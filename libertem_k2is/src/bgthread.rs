use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crossbeam_channel::{unbounded, Receiver, RecvTimeoutError, SendError, Sender, TryRecvError};
use ipc_test::{SHMHandle, SharedSlabAllocator};
use k2o::acquisition::{acquisition_loop, frame_in_acquisition, AcquisitionResult};
use k2o::assemble::{assembler_main, AssemblyResult};
use k2o::block::{BlockRouteInfo, K2Block};
use k2o::control::{control_loop, AcquisitionState, StateError, StateTracker};
use k2o::events::{AcquisitionParams, ChannelEventBus, EventBus, EventMsg, Events, MessagePump};
use k2o::frame::K2Frame;
use k2o::helpers::{set_cpu_affinity, CPU_AFF_WRITER};
use k2o::recv::recv_decode_loop;
use k2o::tracing::get_tracer;
use k2o::write::WriterBuilder;
use opentelemetry::trace::Tracer;
use opentelemetry::{global, Context};

#[derive(Debug, Clone)]
pub struct AddrConfig {
    top: String,
    bottom: String,
}

impl AddrConfig {
    pub fn new(top: &str, bottom: &str) -> Self {
        Self {
            top: top.to_string(),
            bottom: bottom.to_string(),
        }
    }

    pub fn addr_for_sector(&self, sector: u8) -> String {
        if sector < 4 {
            self.bottom.clone()
        } else {
            self.top.clone()
        }
    }

    pub fn port_for_sector(&self, sector_id: u8) -> u32 {
        2001 + sector_id as u32
    }
}

fn k2_bg_thread<
    const PACKET_SIZE: usize, // FIXME: use associated B::PACKET_SIZE instead of this generic?
    F: K2Frame,
    B: K2Block,
>(
    events: &Events,
    writer_builder: Box<dyn WriterBuilder>,
    addr_config: &AddrConfig,
    pump: MessagePump,
    writer_dest_channel: Sender<AcquisitionResult<F>>, // either going to the consumer, or back to the assembly
    shm: SharedSlabAllocator,
) {
    let tracer = get_tracer();
    tracer.in_span("start_threads", |_cx| {
        let ids = 0..=7u8;

        let (recycle_blocks_tx, recycle_blocks_rx) = unbounded::<B>();

        let ctx = Context::current();

        crossbeam::scope(|s| {
            let (assembly_tx, assembly_rx) = unbounded::<(B, BlockRouteInfo)>();
            for sector_id in ids {
                let tx = assembly_tx.clone();
                let recycle_clone_rx = recycle_blocks_rx.clone();
                let recycle_clone_tx = recycle_blocks_tx.clone();
                let events_rx = events.subscribe();
                let addr = addr_config.addr_for_sector(sector_id);
                let port = addr_config.port_for_sector(sector_id);
                let decode_ctx = ctx.clone();
                s.builder()
                    .name(format!("recv-decode-{}", sector_id))
                    .spawn(move |_| {
                        let _guard = decode_ctx.attach();
                        recv_decode_loop::<B, PACKET_SIZE>(
                            sector_id,
                            port,
                            &tx,
                            &recycle_clone_rx,
                            &recycle_clone_tx,
                            &events_rx,
                            events,
                            addr,
                        );
                    })
                    .expect("spawn recv+decode thread");
            }

            let (full_frames_tx, full_frames_rx) = unbounded::<AssemblyResult<F>>();
            // let (recycle_frames_tx, recycle_frames_rx) = unbounded::<F>();

            let asm_events_rx = events.subscribe();
            let asm_ctx = ctx.clone();
            let asm_shm_handle = shm.get_handle();

            // assembly main thread:
            s.builder()
                .name("assembly".to_string())
                .spawn(move |_| {
                    let asm_shm =
                        SharedSlabAllocator::connect(asm_shm_handle.fd, &asm_shm_handle.info)
                            .expect("connect to shm");

                    //set_cpu_affinity(CPU_AFF_ASSEMBLY);
                    //frame_assembler(&assembly_rx, &full_frames_tx);
                    let _guard = asm_ctx.attach();
                    assembler_main(
                        &assembly_rx,
                        &full_frames_tx,
                        &recycle_blocks_tx,
                        asm_events_rx,
                        asm_shm,
                    );
                })
                .expect("could not spawn assembly thread");

            let writer_ctx = ctx.clone();

            let acq_shm_handle = shm.get_handle();

            // acquisition/writer thread:
            let w1rx = full_frames_rx;
            let writer_events_rx = events.subscribe();
            s.builder()
                .name("acquisition".to_string())
                .spawn(move |_| {
                    set_cpu_affinity(CPU_AFF_WRITER);
                    let _guard = writer_ctx.attach();
                    let acq_shm =
                        SharedSlabAllocator::connect(acq_shm_handle.fd, &acq_shm_handle.info)
                            .expect("connect to shm");

                    acquisition_loop(
                        &w1rx,
                        &writer_dest_channel,
                        &writer_events_rx,
                        events,
                        writer_builder,
                        acq_shm,
                    );
                })
                .expect("could not spawn acquisition thread");

            events.send(&EventMsg::Init {});

            control_loop(events, &Some(pump));
        })
        .unwrap();
    });
    global::force_flush_tracer_provider();
}

///
/// The background thread is started when arming the acquisition object,
/// and stops when `Acquisition::stop` is called.
///
/// In turn, this thread starts and manages receiver/decoder/assembly/writer threads.
///
/// Communication with the background thread(s) is handled via the event bus,
/// that is, `k2o::events::Events` and `k2o::events::EventReceiver`.
///
pub fn start_bg_thread<F: K2Frame + 'static>(
    events: Events,
    writer_builder: Box<dyn WriterBuilder>,
    addr_config: AddrConfig,
    pump: MessagePump,

    // Channel from the writer to the next frame consumer
    tx_from_writer: Sender<AcquisitionResult<F>>,

    shm_handle: SHMHandle,
) -> JoinHandle<()>
where
    [(); F::Block::PACKET_SIZE]:,
{
    let thread_builder = std::thread::Builder::new();
    let ctx = Context::current();
    thread_builder
        .name("k2_bg_thread".to_string())
        .spawn(move || {
            let _guard = ctx.attach();
            let shm = SharedSlabAllocator::connect(shm_handle.fd, &shm_handle.info)
                .expect("connect to shm");
            k2_bg_thread::<{ F::Block::PACKET_SIZE }, F, F::Block>(
                &events,
                writer_builder,
                &addr_config,
                pump,
                tx_from_writer,
                shm,
            );
        })
        .expect("failed to start k2 background thread")
}

pub struct AcquisitionRuntime<F: K2Frame> {
    // bg_thread is an Option so we are able to join by moving out of it
    bg_thread: Option<JoinHandle<()>>,
    main_events_tx: Sender<EventMsg>,
    main_events_rx: Receiver<EventMsg>,

    //
    // To be able to have a circular data flow, we need to have access to two
    // channels:
    //
    // 1) Receiving frames after they have been written to disk
    // 2) Sending frames back into circulation (basically "deallocating" them)
    //
    // As we can't easily reach into the data structures created in the
    // background thread, we create them outside and pass the other end of the
    // channel down.
    // FIXME: make these generic over frame type!
    /// This is where an "external" frame consumer gets their frames:
    rx_writer_to_consumer: Receiver<AcquisitionResult<F>>,

    enable_frame_consumer: bool,

    state_tracker: StateTracker,

    shm: SharedSlabAllocator,
}

#[derive(Debug)]
pub enum RuntimeError {
    Timeout,
    Disconnected,
    ConfigurationError,
}

impl<T> From<SendError<T>> for RuntimeError {
    fn from(_: SendError<T>) -> Self {
        RuntimeError::Disconnected
    }
}

impl From<RecvTimeoutError> for RuntimeError {
    fn from(e: RecvTimeoutError) -> Self {
        match e {
            RecvTimeoutError::Timeout => RuntimeError::Timeout,
            RecvTimeoutError::Disconnected => RuntimeError::Disconnected,
        }
    }
}

impl<F: K2Frame + 'static> AcquisitionRuntime<F> {
    pub fn new(
        writer_builder: Box<dyn WriterBuilder>,
        addr_config: &AddrConfig,
        enable_frame_consumer: bool,
        shm: SHMHandle,
    ) -> Self
    where
        [(); F::Block::PACKET_SIZE]:,
    {
        let events: Events = ChannelEventBus::new();
        let pump = MessagePump::new(&events);
        let (main_events_tx, main_events_rx) = pump.get_ext_channels();

        let (tx_writer_to_consumer, rx_writer_to_consumer) = unbounded::<AcquisitionResult<F>>();

        // Two main configuration options:
        // 1) Writer enabled or not (currently can't disable writer)
        // 2) "Frame Consumer" enabled or not
        //
        // They are configured by wiring up channels in the correct way.
        let bg_thread = if enable_frame_consumer {
            // Frame consumer enabled -> after writing, the writer thread should send the frames
            // to the `tx_frame_consumer` channel:
            Some(start_bg_thread::<F>(
                events,
                writer_builder,
                addr_config.clone(),
                pump,
                tx_writer_to_consumer,
                shm,
            ))
        } else {
            // Directly recycle after writing:
            todo!("implement a tx_writer_to_consumer that just frees the shm");
        };

        let mut state_tracker = StateTracker::new();

        // Wait for the init event, so we know the background thread is started.
        // We currently don't wait for all threads, but that's fine, as it's
        // most important that the receiver for control messages is already
        // listening, as events are buffered in a channel:
        loop {
            match main_events_rx.recv_timeout(Duration::from_millis(5000)) {
                Ok(msg) => match state_tracker.set_state_from_msg(&msg) {
                    Ok(AcquisitionState::Idle) => {
                        break;
                    }
                    Ok(_) => continue,
                    Err(StateError::InvalidTransition { from, msg }) => {
                        panic!("invalid state transition: from={from:?} msg={msg:?}")
                    }
                },
                Err(RecvTimeoutError::Timeout) => {
                    global::force_flush_tracer_provider();
                    panic!("timeout while waiting for init event");
                }
                Err(RecvTimeoutError::Disconnected) => panic!("error while waiting for init event"),
            }
        }

        let shm = SharedSlabAllocator::connect(shm.fd, &shm.info).expect("connect to shm");

        AcquisitionRuntime {
            bg_thread,
            main_events_tx,
            main_events_rx,
            rx_writer_to_consumer,
            enable_frame_consumer,
            state_tracker,
            shm,
        }
    }

    /// Receive at most one event from the event bus and update the state
    pub fn update_state(&mut self) {
        match self.main_events_rx.try_recv() {
            Ok(msg) => match self.state_tracker.set_state_from_msg(&msg) {
                Ok(_) => {}
                Err(StateError::InvalidTransition { from, msg }) => {
                    panic!("invalid state transition: from={from:?} msg={msg:?}")
                }
            },
            Err(TryRecvError::Disconnected) => panic!("lost connection to background thread"),
            Err(TryRecvError::Empty) => {}
        }
    }

    pub fn is_done(&self) -> bool {
        // FIXME: out of band message means we can still have data in the queue
        // from this acquisition! Maybe add checks if we have received a
        // sentinel value like AcquisitionResult::DoneSuccess
        matches!(
            self.state_tracker.state,
            AcquisitionState::Idle | AcquisitionState::Shutdown
        )
    }

    pub fn get_next_frame(&self) -> Result<AcquisitionResult<F>, RuntimeError> {
        // FIXME: can we make this a non-issue somehow?
        if !self.enable_frame_consumer {
            return Err(RuntimeError::ConfigurationError);
        }
        let acquisition_result = self
            .rx_writer_to_consumer
            .recv_timeout(Duration::from_millis(100))?;
        // FIXME! frames are not yet ordered by index
        // so sentinels can come out of order (ugh!)
        if matches!(
            acquisition_result,
            AcquisitionResult::DoneSuccess { .. }
                | AcquisitionResult::DoneError { .. }
                | AcquisitionResult::DoneAborted { .. }
        ) {
            self.main_events_tx.send(EventMsg::ProcessingDone)?;
        }
        Ok(acquisition_result)
    }

    pub fn frame_done(&mut self, frame: AcquisitionResult<F>) -> Result<(), RuntimeError> {
        // TODO: keep track of which frames we have seen here, and once we have
        // seen all of them, send `EventMsg::ProcesingDone`
        if !self.enable_frame_consumer {
            return Err(RuntimeError::ConfigurationError);
        }
        let inner = frame.unpack();
        if let Some(f) = inner {
            f.free_payload(&mut self.shm);
        }
        Ok(())
    }

    pub fn get_frame_slot(&mut self, frame: AcquisitionResult<F>) -> Option<usize> {
        let frame_inner = frame.unpack()?;

        // FIXME: this can be done much earlier! after assembly
        // then explicitly replaced with a frame object that is only a
        // reference into shm!
        let slot = frame_inner.writing_done(&mut self.shm);

        Some(slot.slot_idx)
    }

    pub fn arm(&self, params: AcquisitionParams) -> Result<(), RuntimeError> {
        self.main_events_tx.send(EventMsg::Arm { params })?;
        Ok(())
    }

    pub fn stop(&mut self) -> Result<(), RuntimeError> {
        // FIXME: do we need to do anything special if an acquisition is
        // currently running?
        // self.events
        //     .send(&k2o::events::EventMsg::CancelAcquisition {});
        self.main_events_tx
            .send(k2o::events::EventMsg::Shutdown {})?;
        global::force_flush_tracer_provider();
        Ok(())
    }

    pub fn try_join(&mut self) -> Option<()> {
        if let Some(join_handle) = self.bg_thread.take() {
            if !join_handle.is_finished() {
                self.bg_thread = Some(join_handle);
                return None;
            } else {
                return Some(());
            }
        }
        global::force_flush_tracer_provider();
        Some(())
    }

    pub fn try_join_timeout(&mut self, timeout: Duration) -> Result<(), RuntimeError> {
        if let Some(join_handle) = self.bg_thread.take() {
            let deadline = Instant::now() + timeout;
            while !join_handle.is_finished() && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(100));
            }
            return if !join_handle.is_finished() {
                self.bg_thread = Some(join_handle);
                Err(RuntimeError::Timeout)
            } else {
                join_handle
                    .join()
                    .expect("could not join background thread!");
                Ok(())
            };
        } else {
            Ok(()) // join on non-running thread is not an error
        }
    }

    pub fn wait_predicate<P>(&mut self, timeout: Duration, pred: P) -> Option<()>
    where
        P: Fn(&Self) -> bool,
    {
        let deadline = Instant::now() + timeout;
        loop {
            self.update_state();
            if pred(self) {
                return Some(());
            }
            std::thread::sleep(Duration::from_millis(10));
            if Instant::now() > deadline {
                return None;
            }
        }
    }

    /// Wait until the acquisition has ended or the timeout expired, returning `()` for success and `None` for timeout.
    pub fn wait_until_complete(&mut self, timeout: Duration) -> Option<()> {
        self.wait_predicate(timeout, |slf| slf.is_done())
    }

    /// Wait until the arm command succeeded, returning `()` for success and `None` for timeout.
    pub fn wait_for_arm(&mut self, timeout: Duration) -> Option<()> {
        println!("wait_for_arm");
        self.wait_predicate(timeout, |slf| {
            matches!(
                slf.state_tracker.state,
                AcquisitionState::Armed { params: _ }
            )
        })
    }

    pub fn wait_for_start(&mut self, timeout: Duration) -> Option<()> {
        self.wait_predicate(timeout, |slf| {
            matches!(
                slf.state_tracker.state,
                AcquisitionState::AcquisitionStarted {
                    params: _,
                    frame_id: _
                }
            )
        })
    }

    ///
    /// Is the `frame_id` in the current acquisition?
    /// If so, returns the frame's index in the acquisition.
    ///
    pub fn frame_in_acquisition(&self, frame_id: u32) -> Option<u32> {
        let (params, ref_frame_id) = match &self.state_tracker.state {
            AcquisitionState::Startup
            | AcquisitionState::Idle
            | AcquisitionState::Armed { params: _ }
            | AcquisitionState::Shutdown => {
                return None;
            }
            AcquisitionState::AcquisitionStarted { params, frame_id }
            | AcquisitionState::AcquisitionFinishing { params, frame_id } => (params, frame_id),
        };
        frame_in_acquisition(frame_id, *ref_frame_id, params)
    }
}
