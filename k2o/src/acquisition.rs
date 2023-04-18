use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, RecvError, Select, SelectedOperation, Sender};
use human_bytes::human_bytes;
use ipc_test::SharedSlabAllocator;
use opentelemetry::{
    global,
    trace::{self, TraceContextExt, Tracer},
    Context, Key,
};

use crate::{
    assemble::AssemblyResult,
    control::AcquisitionState,
    events::{AcquisitionParams, AcquisitionSize, EventBus, EventMsg, EventReceiver, Events},
    frame::K2Frame,
    ordering::{FrameOrdering, FrameOrderingResult, FrameWithIdx},
    tracing::get_tracer,
    write::{Writer, WriterBuilder},
};

const PRE_ALLOC_CHUNKS: usize = 400; // pre-allocate chunks of this number of frames

enum HandleFramesResult {
    Done { dropped: usize },
    Aborted { dropped: usize },
    Shutdown,
}

pub enum AcquisitionResult<F: K2Frame> {
    Frame(F, u32),
    DroppedFrame(F, u32),
    DroppedFrameOutside(F),
    DoneSuccess { dropped: usize },
    DoneAborted { dropped: usize },
    DoneError, // some possibly unhandled error happened, we don't know a lot here...
}

impl<F: K2Frame> AcquisitionResult<F> {
    pub fn unpack(self) -> Option<F> {
        match self {
            AcquisitionResult::Frame(f, _) => Some(f),
            AcquisitionResult::DroppedFrame(f, _) => Some(f),
            AcquisitionResult::DroppedFrameOutside(f) => Some(f),
            AcquisitionResult::DoneSuccess { dropped: _ } => None,
            AcquisitionResult::DoneAborted { dropped: _ } => None,
            AcquisitionResult::DoneError => None,
        }
    }

    pub fn get_frame(&self) -> Option<&F> {
        match self {
            AcquisitionResult::Frame(f, _) => Some(f),
            AcquisitionResult::DroppedFrame(f, _) => Some(f),
            AcquisitionResult::DroppedFrameOutside(f) => Some(f),
            AcquisitionResult::DoneSuccess { dropped: _ } => None,
            AcquisitionResult::DoneAborted { dropped: _ } => None,
            AcquisitionResult::DoneError => None,
        }
    }
}

pub fn frame_in_acquisition(
    frame_id: u32,
    ref_frame_id: u32,
    params: &AcquisitionParams,
) -> Option<u32> {
    let frame_idx_raw: i64 = frame_id as i64 - ref_frame_id as i64;
    let upper_limit = match params.size {
        AcquisitionSize::Continuous => u32::MAX,
        AcquisitionSize::NumFrames(n) => n,
    };
    if frame_idx_raw >= 0 && frame_idx_raw < upper_limit as i64 {
        Some(frame_idx_raw as u32)
    } else {
        None
    }
}

fn next_hop_ordered<F: K2Frame>(
    ordering: &mut FrameOrdering<F>,
    next_hop_tx: &Sender<AcquisitionResult<F>>,
    result: FrameWithIdx<F>,
) {
    match ordering.handle_frame(result) {
        FrameOrderingResult::Dropped => {}
        FrameOrderingResult::Buffered => {}
        FrameOrderingResult::NextFrame(result) => {
            next_hop_tx.send(result.into()).unwrap();
        }
    }

    // possibly send out buffered frames:
    while let Some(buffered_result) = ordering.maybe_get_next_frame() {
        next_hop_tx.send(buffered_result.into()).unwrap();
    }
}

// TODO: maybe need a drain state which we enter in case of errors, to get
// rid of frames in the pipeline that belong to a canceled acquisition or we could add
// an "acquisition generation" as metadata to the messages, so we can safely
// discard frames that don't belong to whatever acquisition we think we are
// currently working on

struct FrameHandler<'a, F: K2Frame> {
    channel: &'a Receiver<AssemblyResult<F>>,
    next_hop_tx: &'a Sender<AcquisitionResult<F>>,
    events_rx: &'a EventReceiver,
    writer_builder: &'a dyn WriterBuilder,
    shm: &'a mut SharedSlabAllocator,
    params: AcquisitionParams,
    ref_frame_id: u32,
    ordering: FrameOrdering<F>,

    counter: usize,
    dropped: usize,
    dropped_outside: usize,

    ref_ts: Instant,
    ref_bytes_written: usize,
}

impl<'a, F: K2Frame> FrameHandler<'a, F> {
    fn new(
        channel: &'a Receiver<AssemblyResult<F>>,
        next_hop_tx: &'a Sender<AcquisitionResult<F>>,
        events_rx: &'a EventReceiver,
        writer_builder: &'a dyn WriterBuilder,
        shm: &'a mut SharedSlabAllocator,
        params: AcquisitionParams,
        ref_frame_id: u32,
    ) -> Self {
        FrameHandler {
            channel,
            next_hop_tx,
            events_rx,
            writer_builder,
            shm,
            params,
            ref_frame_id,
            ordering: FrameOrdering::new(0),
            counter: 0,
            dropped: 0,
            dropped_outside: 0,
            ref_ts: Instant::now(),
            ref_bytes_written: 0,
        }
    }

    #[must_use]
    fn handle_frames(mut self) -> HandleFramesResult {
        let tracer = get_tracer();
        let span = tracer.start("handle_frames");
        let _guard = trace::mark_span_as_active(span);
        let mut sel = Select::new();
        let op_events = sel.recv(self.events_rx);
        let op_frames = sel.recv(self.channel);

        let frame_shape = F::get_shape_for_binning(&self.params.binning);
        let mut writer: Box<dyn Writer> = self
            .writer_builder
            .open_for_writing(&self.params.size, &frame_shape, std::mem::size_of::<u16>())
            .expect("failed to open for writing");
        writer
            .resize(PRE_ALLOC_CHUNKS)
            .expect("failed to pre-allocate");

        loop {
            let oper = sel.select();
            match oper.index() {
                i if i == op_events => match oper.recv(self.events_rx) {
                    Ok(EventMsg::Shutdown {}) => return HandleFramesResult::Shutdown,
                    Ok(EventMsg::CancelAcquisition {}) => {
                        return HandleFramesResult::Aborted {
                            dropped: self.dropped,
                        }
                    }
                    Ok(_) => continue,
                    Err(RecvError) => {
                        return HandleFramesResult::Shutdown;
                    }
                },
                i if i == op_frames => {
                    // FIXME: if we are in `AcquisitionSync::Immediately` mode,
                    // we need to handle  wrap-around (or reset) of frame IDs gracefully.
                    // so 1) we need to detect the wrap-around, and 2) generate a derived
                    // frame index, which takes this wrap around into account. We may need
                    // to keep a "generation" counter, which is incremented for each
                    // wrap-around or reset. Note that frames may be retired from the
                    // pipeline out-of-order, which the logic needs to account for.
                    match self.handle_frame(oper, &mut writer) {
                        Some(result) => return result,
                        None => continue,
                    }
                }
                _ => unreachable!(),
            }
        }
    }

    #[must_use]
    fn handle_frame(
        &mut self,
        oper: SelectedOperation,
        writer: &mut Box<dyn Writer>,
    ) -> Option<HandleFramesResult> {
        let cx = Context::current();
        let span = cx.span();

        match oper.recv(self.channel) {
            Ok(AssemblyResult::AssembledFrame(frame)) => {
                let frame_id = frame.get_frame_id();
                span.add_event(
                    "handle_assembled_frame",
                    vec![Key::new("frame_id").i64(frame_id as i64)],
                );
                self.handle_assembled_frame(frame, writer)
            }
            Ok(AssemblyResult::AssemblyTimeout { frame, frame_id }) => {
                span.add_event("timeout", vec![Key::new("frame_id").i64(frame_id as i64)]);
                self.timeout(frame_id, frame);
                None
            }
            Err(RecvError) => Some(HandleFramesResult::Shutdown),
        }
    }

    #[must_use]
    fn handle_assembled_frame(
        &mut self,
        frame: F,
        writer: &mut Box<dyn Writer>,
    ) -> Option<HandleFramesResult> {
        let frame_idx_raw: i64 = frame.get_frame_id() as i64 - self.ref_frame_id as i64;
        let upper_limit = match self.params.size {
            AcquisitionSize::Continuous => u32::MAX,
            AcquisitionSize::NumFrames(n) => n,
        };
        if frame_idx_raw >= 0 && (frame_idx_raw as usize) % PRE_ALLOC_CHUNKS == 0 {
            // pre-allocate in chunks of PRE_ALLOC_CHUNKS frames
            let new_size = core::cmp::min(upper_limit as usize, self.counter + PRE_ALLOC_CHUNKS);
            writer.resize(new_size).expect("could not resize");
        }
        if let Some(frame_idx) =
            frame_in_acquisition(frame.get_frame_id(), self.ref_frame_id, &self.params)
        {
            let out_frame_idx_base = frame_idx * F::get_num_subframes(&self.params.binning);
            let frame_shape = F::get_shape_for_binning(&self.params.binning);
            let frame_size_bytes = frame_shape.0 * frame_shape.1 * std::mem::size_of::<u16>(); // assumes u16 data
            for subframe_idx in frame.subframe_indexes(&self.params.binning) {
                let subframe = frame.get_subframe(subframe_idx, &self.params.binning);
                writer.write_frame(&subframe, out_frame_idx_base + subframe_idx);
                self.ref_bytes_written += frame_size_bytes;
                self.counter += 1;
            }
            if self.counter % 100 == 0 {
                self.print_stats(&frame);
            }
            if let AcquisitionSize::NumFrames(num) = self.params.size {
                // FIXME: NumFrames should always be a
                // multiple of the number of subframes,
                // otherwise this check can fail!
                if self.counter == num as usize {
                    if self.counter % 100 != 0 {
                        self.print_stats(&frame);
                    }
                    let result = FrameWithIdx::Frame(frame, frame_idx);
                    next_hop_ordered(&mut self.ordering, self.next_hop_tx, result);
                    self.ordering.dump_if_nonempty();
                    assert!(self.ordering.is_empty());
                    return Some(HandleFramesResult::Done {
                        dropped: self.dropped,
                    });
                }
            }
            let result = FrameWithIdx::Frame(frame, frame_idx);
            next_hop_ordered(&mut self.ordering, self.next_hop_tx, result);
            None
        } else {
            self.dropped_outside += 1;
            frame.free_payload(self.shm);
            None
        }
    }

    fn timeout(&mut self, frame_id: u32, frame: F) {
        let ctx = Context::current();
        if let Some(frame_idx) = frame_in_acquisition(frame_id, self.ref_frame_id, &self.params) {
            ctx.span().add_event(
                "timeout_in_acquisition",
                vec![Key::new("frame_id").i64(frame_id as i64)],
            );
            self.dropped += 1;
            self.counter += 1;
            let result = FrameWithIdx::DroppedFrame(frame, frame_idx);
            next_hop_ordered(&mut self.ordering, self.next_hop_tx, result);
        } else {
            ctx.span().add_event(
                "timeout_outside_acquisition",
                vec![Key::new("frame_id").i64(frame_id as i64)],
            );
            self.dropped_outside += 1;
            frame.free_payload(self.shm);
        }
    }

    fn print_stats(&mut self, frame: &F) {
        let now = Instant::now();
        let latency = frame.get_created_timestamp().elapsed();
        let channel_size = self.channel.len();
        let delta_t = now - self.ref_ts;
        let throughput = {
            if delta_t > Duration::ZERO {
                let bytes_per_second = self.ref_bytes_written as f64 / delta_t.as_secs_f64();
                human_bytes(bytes_per_second)
            } else {
                String::from("")
            }
        };
        let fps = {
            if delta_t > Duration::ZERO {
                format!("{}", 100.0 / delta_t.as_secs_f64())
            } else {
                String::from("")
            }
        };
        println!("frame counter={} frame_id={} dropped={} dropped_outside={}, latency first block -> frame written={:?} channel.len()={} write throughput={}/s fps={}",
                self.counter, frame.get_frame_id(), self.dropped, self.dropped_outside, latency, channel_size, throughput, fps);

        self.ref_ts = Instant::now();
        self.ref_bytes_written = 0;
    }
}

///
/// Instantiate a writer, receive and write N frames, and forward frames
/// to the next hop channel.
///
/// Filters out frames that don't belong to the current acquisition.
///

pub fn acquisition_loop<F: K2Frame>(
    channel: &Receiver<AssemblyResult<F>>,
    next_hop_tx: &Sender<AcquisitionResult<F>>,
    events_rx: &EventReceiver,
    events: &Events,
    writer_builder: Box<dyn WriterBuilder>,
    mut shm: SharedSlabAllocator,
) {
    let tracer = get_tracer();

    tracer.in_span("acquisition_loop", |cx| {
        let span = cx.span();

        let mut state: AcquisitionState = AcquisitionState::default();

        let mut sel = Select::new();
        let op_events = sel.recv(events_rx);
        let op_frames = sel.recv(channel);

        loop {
            let oper = sel.select();
            match oper.index() {
                i if i == op_events => {
                    let msg_result = oper.recv(events_rx);
                    match msg_result {
                        Ok(EventMsg::Arm { params }) => {
                            state = AcquisitionState::Armed {
                                params: params.clone(),
                            };

                            // Forward the start event for sectors, making sure we get the
                            // `AcquisitionStartedSector` event only after we are in `Armed` state.
                            // If instead the sectors would react to `StartAcquisition` like
                            // we do here, we could possibly get the response from the
                            // sectors before we are transitioning to the `Armed` state,
                            // meaning we don't have the acquisition parameters yet etc...
                            events.send(&EventMsg::ArmSectors { params });
                        }
                        Ok(EventMsg::AcquisitionStartedSector {
                            sector_id: _,
                            frame_id,
                        }) => {
                            match state {
                                AcquisitionState::Armed { params } => {
                                    state = AcquisitionState::AcquisitionStarted {
                                        params: params.clone(),
                                        frame_id,
                                    };
                                    events.send(&EventMsg::AcquisitionStarted {
                                        frame_id,
                                        params: params.clone(),
                                    });
                                    println!("acquisition started, first frame_id = {}", frame_id);

                                    Context::current()
                                        .span()
                                        .add_event("AcquisitionStarted", vec![]);
                                    let fh = FrameHandler::new(
                                        channel,
                                        next_hop_tx,
                                        events_rx,
                                        &*writer_builder, // lol
                                        &mut shm,
                                        params,
                                        frame_id,
                                    );
                                    let write_result = fh.handle_frames();
                                    eprintln!("handle_frames done.");
                                    events.send(&EventMsg::AcquisitionEnded {});
                                    Context::current()
                                        .span()
                                        .add_event("AcquisitionEnded", vec![]);
                                    match write_result {
                                        HandleFramesResult::Done { dropped } => {
                                            next_hop_tx
                                                .send(AcquisitionResult::DoneSuccess { dropped })
                                                .unwrap();
                                            continue;
                                        }
                                        HandleFramesResult::Aborted { dropped } => {
                                            next_hop_tx
                                                .send(AcquisitionResult::DoneAborted { dropped })
                                                .unwrap();
                                            continue;
                                        }
                                        HandleFramesResult::Shutdown => {
                                            next_hop_tx.send(AcquisitionResult::DoneError).unwrap();
                                            break;
                                        }
                                    }
                                }
                                AcquisitionState::AcquisitionStarted {
                                    params: _,
                                    frame_id: _,
                                } => {
                                    // we are only interested in the event from the first sector that starts the acquisition:
                                    println!(
                                        "ignoring AcuisitionStartedSector in AcquisitionStarted state"
                                    );
                                }
                                AcquisitionState::Idle
                                | AcquisitionState::AcquisitionFinishing { params: _, frame_id: _ }
                                | AcquisitionState::Startup
                                | AcquisitionState::Shutdown => {
                                    panic!(
                                        "AcquisitionStartedSector received in invalid state {state:?}"
                                    );
                                }
                            }
                        }
                        Ok(EventMsg::Shutdown {}) => break,
                        Ok(_) => continue,
                        Err(RecvError) => {
                            next_hop_tx.send(AcquisitionResult::DoneError).unwrap();
                            break;
                        }
                    }
                }
                i if i == op_frames => {
                    match oper.recv(channel) {
                        Ok(AssemblyResult::AssembledFrame(frame)) => {
                            let frame_id = frame.get_frame_id();
                            frame.free_payload(&mut shm);
                            span.add_event("DroppedFrameOutside", vec![
                                Key::new("frame_id").i64(frame_id as i64),
                            ]);
                        }
                        Ok(AssemblyResult::AssemblyTimeout { frame, frame_id: _ }) => {
                            // timeout outside of an acquisition - not so interesting
                            // to count this as an event
                            let frame_id = frame.get_frame_id();
                            frame.free_payload(&mut shm);
                            span.add_event("AssemblyTimeout", vec![
                                Key::new("frame_id").i64(frame_id as i64),
                            ]);
                            continue;
                        }
                        Err(RecvError) => break,
                    }
                }
                _ => unreachable!(),
            }
        }
    });
    global::force_flush_tracer_provider();
}

// /// Track until which index we have received all frames, and the possibly
// /// "non-contiguous" set of other frames "on the right" which we have received.
// struct TrackFrames {
//     /// Marker dividing the index space of the acquisition into frames that we have
//     /// finished processing, and those that are still in flight or to be received.
//     ///
//     /// This points at the first frame in the todo-part, so a `0` at the beginning
//     /// means everything still needs to be received.
//     ///
//     /// Finished processing also includes dropped frames, where we have waited long
//     /// enough and didn't get all the data for the full frame.
//     dense_until: usize,
//
//     /// Set of all indices `i` that have been successfully received, but where
//     /// another index `j` exists, such that the frame `j` has not been received fully.
//     leading: HashSet<usize>,
//
//     /// Separate tracker for dropped frames
//     dropped: HashSet<usize>,
// }
//
// impl TrackFrames {
//     pub fn new() -> Self {
//         TrackFrames {
//             dense_until: 0,
//             leading: HashSet::new(),
//             dropped: HashSet::new(),
//         }
//     }
//
//     /// After changing `leading` or `dense_until`, call this function to check
//     /// if we can move `dense_until` even further, and remove items from
//     /// `leading`.
//     fn maybe_move_marker(&mut self) {
//         if self.leading.len() == 0 {
//             return; // no need to adjust if `self.leading` is empty
//         }
//         let max_leading = self
//             .leading
//             .iter()
//             .max()
//             .expect("`leading` is not empty, so should have a maximum");
//         for idx in self.dense_until..=*max_leading {
//             if self.leading.contains(&idx) {
//                 self.leading.remove(&idx);
//                 self.dense_until = idx + 1;
//             } else {
//                 // first frame which is not done yet encountered, so we keep this and
//                 // the following in the `leading` set.
//                 return;
//             }
//         }
//     }
//
//     fn track_frame<F: K2Frame>(&mut self, frame: &F, rel_idx: usize) {
//         if rel_idx == self.dense_until {
//             // fast path: the frame is exactly the next "expected" frame:
//             self.dense_until += 1;
//             self.maybe_move_marker();
//             return;
//         } else if rel_idx > self.dense_until {
//             // anything else
//             self.leading.insert(rel_idx);
//             self.maybe_move_marker();
//             return;
//         } else {
//             panic!(
//                 "cannot track a frame with idx {} < {}",
//                 rel_idx, self.dense_until
//             );
//         }
//     }
//
//     pub fn track_frame_done<F: K2Frame>(&mut self, frame: &F, rel_idx: usize) {
//         self.track_frame(frame, rel_idx);
//     }
//
//     pub fn track_frame_dropped<F: K2Frame>(&mut self, frame: &F, rel_idx: usize) {
//         self.track_frame(frame, rel_idx);
//         self.dropped.insert(rel_idx);
//     }
// }
