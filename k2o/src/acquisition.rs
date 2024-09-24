use std::{
    collections::HashSet,
    time::{Duration, Instant},
};

use common::{background_thread::AcquisitionSize, tracing::get_tracer};
use crossbeam::channel::{Receiver, RecvError, Select, SelectedOperation, Sender};
use human_bytes::human_bytes;
use ipc_test::SharedSlabAllocator;
use log::{error, info, warn};
use opentelemetry::{
    trace::{self, Span, TraceContextExt, Tracer},
    Context, Key,
};
use partialdebug::placeholder::PartialDebug;

use crate::{
    assemble::AssemblyResult,
    control::AcquisitionState,
    events::{AcquisitionParams, EventBus, EventMsg, EventReceiver, Events},
    frame::{FrameMeta, GenericFrame, K2Frame},
    ordering::{FrameOrdering, FrameOrderingResult, FrameWithIdx},
};

const PRE_ALLOC_CHUNKS: usize = 400; // pre-allocate chunks of this number of frames

enum HandleFramesResult {
    Done { dropped: usize },
    Aborted { dropped: usize },
    Shutdown,
}

#[derive(PartialDebug)]
pub enum AcquisitionResult<F> {
    Frame(F, u32),
    DroppedFrame(F, u32),
    DroppedFrameOutside(F),
    DoneSuccess {
        dropped: usize,
        acquisition_id: usize,
    },
    DoneAborted {
        dropped: usize,
        acquisition_id: usize,
    },

    /// This is the result if some threads upstream closed their end of the
    /// channel and we get a receive error, while an acquisition is running -
    /// the system is probably shutting down (or crashing).
    DoneShuttingDown {
        acquisition_id: usize,
    },

    /// This is the result if some threads upstream closed their end of the
    /// channel while no current acquisition is known
    ShutdownIdle,
}

impl<F> AcquisitionResult<F> {
    pub fn unpack(self) -> Option<F> {
        match self {
            AcquisitionResult::Frame(f, _) => Some(f),
            AcquisitionResult::DroppedFrame(f, _) => Some(f),
            AcquisitionResult::DroppedFrameOutside(f) => Some(f),
            AcquisitionResult::DoneSuccess {
                dropped: _,
                acquisition_id: _,
            } => None,
            AcquisitionResult::DoneAborted {
                dropped: _,
                acquisition_id: _,
            } => None,
            AcquisitionResult::DoneShuttingDown { acquisition_id: _ } => None,
            AcquisitionResult::ShutdownIdle => None,
        }
    }

    pub fn get_frame(&self) -> Option<&F> {
        match self {
            AcquisitionResult::Frame(f, _) => Some(f),
            AcquisitionResult::DroppedFrame(f, _) => Some(f),
            AcquisitionResult::DroppedFrameOutside(f) => Some(f),
            AcquisitionResult::DoneSuccess {
                dropped: _,
                acquisition_id: _,
            } => None,
            AcquisitionResult::DoneAborted {
                dropped: _,
                acquisition_id: _,
            } => None,
            AcquisitionResult::DoneShuttingDown { acquisition_id: _ } => None,
            AcquisitionResult::ShutdownIdle => None,
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
        AcquisitionSize::NumFrames(n) => n.try_into().expect("too many frames in acquisition size"),
        AcquisitionSize::Auto => u32::MAX, // FIXME: is this the correct default?
    };
    if frame_idx_raw >= 0 && frame_idx_raw < upper_limit as i64 {
        Some(frame_idx_raw as u32)
    } else {
        None
    }
}

/// Take `result` and either directly send it on to the `next_hop_tx` channel,
/// or buffer it in `ordering`.
fn next_hop_ordered(
    ordering: &mut FrameOrdering,
    next_hop_tx: &Sender<AcquisitionResult<GenericFrame>>,
    result: FrameWithIdx,
) {
    match ordering.handle_frame(result) {
        FrameOrderingResult::Dropped => {}
        FrameOrderingResult::Buffered => {}
        FrameOrderingResult::NextFrame(result) => {
            let result: AcquisitionResult<GenericFrame> = result.into();
            next_hop_tx.send(result).unwrap();
        }
    }

    // possibly send out buffered frames:
    while let Some(buffered_result) = ordering.maybe_get_next_frame() {
        let buffered_result: AcquisitionResult<GenericFrame> = buffered_result.into();
        next_hop_tx.send(buffered_result).unwrap();
    }
}

// TODO: maybe need a drain state which we enter in case of errors, to get
// rid of frames in the pipeline that belong to a canceled acquisition or we could add
// an "acquisition generation" as metadata to the messages, so we can safely
// discard frames that don't belong to whatever acquisition we think we are
// currently working on

struct FrameHandler<'a, F: K2Frame> {
    acquisition_id: usize,

    channel: &'a Receiver<AssemblyResult<F>>,
    next_hop_tx: &'a Sender<AcquisitionResult<GenericFrame>>,
    events_rx: &'a EventReceiver,
    shm: &'a mut SharedSlabAllocator,
    params: AcquisitionParams,
    ref_frame_id: u32,
    ordering: FrameOrdering,

    counter: usize,
    dropped: usize,
    dropped_outside: usize,
    dropped_frame_ids: HashSet<u32>,

    ref_ts: Instant,
    ref_bytes_written: usize,
}

impl<'a, F: K2Frame> FrameHandler<'a, F> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        acquisition_id: usize,
        channel: &'a Receiver<AssemblyResult<F>>,
        next_hop_tx: &'a Sender<AcquisitionResult<GenericFrame>>,
        events_rx: &'a EventReceiver,
        shm: &'a mut SharedSlabAllocator,
        params: AcquisitionParams,
        ref_frame_id: u32,
    ) -> Self {
        FrameHandler {
            acquisition_id,
            channel,
            next_hop_tx,
            events_rx,
            shm,
            params,
            ref_frame_id,
            ordering: FrameOrdering::new(0),
            counter: 0,
            dropped: 0,
            dropped_outside: 0,
            dropped_frame_ids: HashSet::new(),
            ref_ts: Instant::now(),
            ref_bytes_written: 0,
        }
    }

    #[must_use]
    fn handle_frames(mut self) -> HandleFramesResult {
        let tracer = get_tracer();
        let mut span = tracer.start("handle_frames");
        span.add_event(
            "start",
            vec![Key::new("ref_frame_id").i64(self.ref_frame_id as i64)],
        );
        let _guard = trace::mark_span_as_active(span);
        let mut sel = Select::new();
        let op_events = sel.recv(self.events_rx);
        let op_frames = sel.recv(self.channel);

        let frame_shape = F::get_shape_for_binning(&self.params.binning);
        let pixel_size_bytes = F::get_pixel_size_bytes();

        loop {
            let oper = sel.select();
            match oper.index() {
                i if i == op_events => match oper.recv(self.events_rx) {
                    Ok(EventMsg::Shutdown {}) => return HandleFramesResult::Shutdown,
                    Ok(EventMsg::CancelAcquisition { acquisition_id: _ }) => {
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
                    match self.handle_frame(oper) {
                        Some(result) => return result,
                        None => continue,
                    }
                }
                _ => unreachable!(),
            }
        }
    }

    #[must_use]
    fn handle_frame(&mut self, oper: SelectedOperation) -> Option<HandleFramesResult> {
        let cx = Context::current();
        let span = cx.span();

        match oper.recv(self.channel) {
            Ok(AssemblyResult::AssembledFrame(frame)) => {
                let frame_id = frame.get_frame_id();
                span.add_event(
                    "handle_assembled_frame",
                    vec![Key::new("frame_id").i64(frame_id as i64)],
                );
                if frame.get_acquisition_id() == self.acquisition_id {
                    self.handle_assembled_frame(frame)
                } else {
                    warn!("dropped assembled frame from unrelated acquisition");
                    frame.free_payload(self.shm);
                    None
                }
            }
            Ok(AssemblyResult::AssemblyTimeout { frame, frame_id }) => {
                span.add_event("timeout", vec![Key::new("frame_id").i64(frame_id as i64)]);
                let frame_meta = frame.get_meta();
                if frame.get_acquisition_id() == self.acquisition_id {
                    self.timeout(frame_id, frame);
                } else {
                    warn!("dropped frame from unrelated acquisition");
                    frame.free_payload(self.shm);
                }

                // handle the case that the last frame was dropped:
                if let AcquisitionSize::NumFrames(num) = self.params.size {
                    if self.counter == num as usize {
                        if self.counter % 100 != 0 {
                            self.print_stats(&frame_meta);
                        }
                        return Some(HandleFramesResult::Done {
                            dropped: self.dropped,
                        });
                    }
                }
                None
            }
            Err(RecvError) => Some(HandleFramesResult::Shutdown),
        }
    }

    #[must_use]
    fn handle_assembled_frame(&mut self, frame: F) -> Option<HandleFramesResult> {
        let frame_idx_raw: i64 = frame.get_frame_id() as i64 - self.ref_frame_id as i64;
        let upper_limit = match self.params.size {
            AcquisitionSize::Continuous => u32::MAX,
            AcquisitionSize::NumFrames(n) => n.try_into().expect("too many frames in acqusition size"),
            AcquisitionSize::Auto => u32::MAX,
        };
        if frame_idx_raw >= 0 && (frame_idx_raw as usize) % PRE_ALLOC_CHUNKS == 0 {
            // pre-allocate in chunks of PRE_ALLOC_CHUNKS frames
            let new_size = core::cmp::min(upper_limit as usize, self.counter + PRE_ALLOC_CHUNKS);
        }
        if let Some(frame_idx) =
            frame_in_acquisition(frame.get_frame_id(), self.ref_frame_id, &self.params)
        {
            let out_frame_idx_base = frame_idx * F::get_num_subframes(&self.params.binning);
            let frame_shape = F::get_shape_for_binning(&self.params.binning);
            let pixel_size_bytes = F::get_pixel_size_bytes();
            let frame_size_bytes = frame_shape.0 * frame_shape.1 * pixel_size_bytes;
            for subframe_idx in frame.subframe_indexes(&self.params.binning) {
                let subframe = frame.get_subframe(subframe_idx, &self.params.binning, self.shm);
                // writer.write_frame(&subframe, out_frame_idx_base + subframe_idx);
                self.ref_bytes_written += frame_size_bytes;
                self.counter += 1;
            }
            let frame_meta = frame.get_meta();
            if self.counter % 100 == 0 {
                self.print_stats(&frame_meta);
            }
            if let AcquisitionSize::NumFrames(num) = self.params.size {
                // FIXME: NumFrames should always be a
                // multiple of the number of subframes,
                // otherwise this check can fail!
                if self.counter == num as usize {
                    if self.counter % 100 != 0 {
                        self.print_stats(&frame_meta);
                    }
                    let result = FrameWithIdx::Frame(frame.into_generic(), frame_idx);
                    next_hop_ordered(&mut self.ordering, self.next_hop_tx, result);
                    self.ordering.dump_if_nonempty();
                    if !self.ordering.is_empty() {
                        info!("self.counter = {}, num = {}", self.counter, num);
                    }
                    assert!(self.ordering.is_empty());
                    return Some(HandleFramesResult::Done {
                        dropped: self.dropped,
                    });
                }
            }
            let result = FrameWithIdx::Frame(frame.into_generic(), frame_idx);
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
            // only increment the counter if we haven't seen this frame ID before:
            if self.dropped_frame_ids.insert(frame.get_frame_id()) {
                self.counter += 1; // FIXME: might need to increment by number of subframes?
            }
            let result = FrameWithIdx::DroppedFrame(frame.into_generic(), frame_idx);
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

    fn print_stats(&mut self, frame_meta: &FrameMeta) {
        let now = Instant::now();
        let latency = frame_meta.get_created_timestamp().elapsed();
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
        info!("acq#{} frame counter={} frame_id={} dropped={} dropped_outside={}, latency first block -> frame written={:?} channel.len()={} write throughput={}/s fps={}",
            frame_meta.get_acquisition_id(), self.counter, frame_meta.get_frame_id(),
            self.dropped, self.dropped_outside, latency, channel_size,
            throughput, fps
        );

        self.ref_ts = Instant::now();
        self.ref_bytes_written = 0;
    }
}

///
/// Instantiate a writer, receive and write N frames, and forward frames
/// to the next hop channel. This is started in a background thread.
///
/// Filters out frames that don't belong to the current acquisition.
///

pub fn acquisition_loop<F: K2Frame>(
    channel: &Receiver<AssemblyResult<F>>,
    next_hop_tx: &Sender<AcquisitionResult<GenericFrame>>,
    events_rx: &EventReceiver,
    events: &Events,
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
                        Ok(EventMsg::Arm { params, acquisition_id }) => {
                            state = AcquisitionState::Armed {
                                params: params.clone(),
                                acquisition_id,
                            };

                            // Forward the start event for sectors, making sure we get the
                            // `AcquisitionStartedSector` event only after we are in `Armed` state.
                            // If instead the sectors would react to `StartAcquisition` like
                            // we do here, we could possibly get the response from the
                            // sectors before we are transitioning to the `Armed` state,
                            // meaning we don't have the acquisition parameters yet etc...
                            events.send(&EventMsg::ArmSectors { params, acquisition_id});
                        }
                        Ok(EventMsg::AcquisitionStartedSector {
                            sector_id: _,
                            frame_id,
                            acquisition_id: acquisition_id_outer,
                        }) => {
                            match state {
                                AcquisitionState::Armed { params , acquisition_id } => {
                                    if acquisition_id != acquisition_id_outer {
                                        error!("acquisition id mismatch; {acquisition_id} != {acquisition_id_outer}");
                                    }
                                    state = AcquisitionState::AcquisitionStarted {
                                        params: params.clone(),
                                        frame_id,
                                        acquisition_id,
                                    };
                                    events.send(&EventMsg::AcquisitionStarted {
                                        frame_id,
                                        params: params.clone(),
                                        acquisition_id,
                                    });
                                    info!("acquisition started, first frame_id = {}", frame_id);

                                    Context::current()
                                        .span()
                                        .add_event("AcquisitionStarted", vec![]);
                                    let fh = FrameHandler::new(
                                        acquisition_id,
                                        channel,
                                        next_hop_tx,
                                        events_rx,
                                        &mut shm,
                                        params,
                                        frame_id,
                                    );
                                    let write_result = fh.handle_frames();
                                    info!("handle_frames done.");
                                    events.send(&EventMsg::AcquisitionEnded { acquisition_id });
                                    Context::current()
                                        .span()
                                        .add_event("AcquisitionEnded", vec![]);
                                    match write_result {
                                        HandleFramesResult::Done { dropped } => {
                                            next_hop_tx
                                                .send(AcquisitionResult::DoneSuccess { dropped, acquisition_id })
                                                .unwrap();
                                            continue;
                                        }
                                        HandleFramesResult::Aborted { dropped } => {
                                            next_hop_tx
                                                .send(AcquisitionResult::DoneAborted { dropped, acquisition_id })
                                                .unwrap();
                                            continue;
                                        }
                                        HandleFramesResult::Shutdown => {
                                            next_hop_tx.send(AcquisitionResult::DoneShuttingDown { acquisition_id }).unwrap();
                                            break;
                                        }
                                    }
                                }
                                AcquisitionState::AcquisitionStarted {
                                    params: _,
                                    frame_id: _,
                                    acquisition_id,
                                } => {
                                    // we are only interested in the event from the first sector that starts the acquisition:
                                    warn!(
                                        "ignoring AcuisitionStartedSector in AcquisitionStarted state for acq#{acquisition_id}"
                                    );
                                }
                                AcquisitionState::Idle
                                | AcquisitionState::AcquisitionFinishing { params: _, frame_id: _, acquisition_id: _ }
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
                            next_hop_tx.send(AcquisitionResult::ShutdownIdle).unwrap();
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
}
