use crate::acquisition::{acquisition_loop, frame_in_acquisition, AcquisitionResult};
use crate::assemble::{assembler_main, AssemblyResult};
use crate::block::{BlockRouteInfo, K2Block};
use crate::block_is::K2ISBlock;
use crate::block_summit::K2SummitBlock;
use crate::control::{control_loop, AcquisitionState, StateError, StateTracker};
use crate::events::{AcquisitionParams, ChannelEventBus, EventBus, EventMsg, Events, MessagePump};
use crate::frame::{GenericFrame, K2Frame};
use crate::frame_is::K2ISFrame;
use crate::frame_summit::K2SummitFrame;
use crate::helpers::{set_cpu_affinity, CPU_AFF_WRITER};
use crate::params::CameraMode;
use crate::recv::recv_decode_loop;
use crate::tracing::get_tracer;
use crossbeam_channel::{unbounded, Receiver, RecvTimeoutError, SendError, Sender, TryRecvError};
use ipc_test::{SHMHandle, SharedSlabAllocator};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use log::debug;
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
    addr_config: &AddrConfig,
    pump: MessagePump,
    writer_dest_channel: Sender<AcquisitionResult<GenericFrame>>, // either going to the consumer, or back to the assembly
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
            let asm_shm_handle = shm.get_handle().os_handle;

            // assembly main thread:
            s.builder()
                .name("assembly".to_string())
                .spawn(move |_| {
                    let asm_shm =
                        SharedSlabAllocator::connect(&asm_shm_handle).expect("connect to shm");

                    //set_cpu_affinity(CPU_AFF_ASSEMBLY);
                    //frame_assembler(&assembly_rx, &full_frames_tx);
                    let _guard = asm_ctx.attach();
                    assembler_main::<F, B>(
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
                    let acq_shm = SharedSlabAllocator::connect(&acq_shm_handle.os_handle)
                        .expect("connect to shm");

                    acquisition_loop(
                        &w1rx,
                        &writer_dest_channel,
                        &writer_events_rx,
                        events,
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
pub fn start_bg_thread<F: K2Frame, const PACKET_SIZE: usize>(
    events: Events,
    addr_config: AddrConfig,
    pump: MessagePump,

    // Channel from the writer to the next frame consumer
    tx_from_writer: Sender<AcquisitionResult<GenericFrame>>,

    shm_handle: SHMHandle,
) -> JoinHandle<()> {
    let thread_builder = std::thread::Builder::new();
    let ctx = Context::current();
    thread_builder
        .name("k2_bg_thread".to_string())
        .spawn(move || {
            let _guard = ctx.attach();
            let shm = SharedSlabAllocator::connect(&shm_handle.os_handle).expect("connect to shm");
            k2_bg_thread::<PACKET_SIZE, F, F::Block>(
                &events,
                &addr_config,
                pump,
                tx_from_writer,
                shm,
            );
        })
        .expect("failed to start k2 background thread")
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

pub enum UpdateStateResult {
    DidUpdate,
    NoNewMessage,
}

pub enum WaitResult {
    Timeout,
    PredSuccess,
}

impl WaitResult {
    pub fn is_success(&self) -> bool {
        matches!(self, WaitResult::PredSuccess)
    }
}

/// The `AcquisitionRuntime` starts and communicates with a background thread,
/// keeps track of the state via the `StateTracker`, and owns the shared memory
/// area.
///
/// This runtime is kept alive over multiple acquisitions, and as such the
/// parameters (like IS/Summit mode, network settings, shm socket path, frame
/// iterator settings) cannot be changed without restarting the runtime.
///
/// It is possible to change per-acquisition settings, though, like filename and
/// file writer settings, number of frames for the acquisition, and the camera
/// sync mode.
pub struct AcquisitionRuntime {
    // bg_thread is an Option so we are able to join by moving out of it
    bg_thread: Option<JoinHandle<()>>,
    main_events_tx: Sender<EventMsg>,
    main_events_rx: Receiver<EventMsg>,

    /// This is where an "external" frame consumer gets their frames:
    rx_writer_to_consumer: Receiver<AcquisitionResult<GenericFrame>>,

    enable_frame_consumer: bool,

    state_tracker: StateTracker,

    shm: SharedSlabAllocator,

    current_acquisition_id: usize,
}

impl AcquisitionRuntime {
    pub fn new(
        addr_config: &AddrConfig,
        enable_frame_consumer: bool,
        shm: SHMHandle,
        mode: CameraMode,
    ) -> Self {
        let events: Events = ChannelEventBus::new();
        let pump = MessagePump::new(&events);
        let (main_events_tx, main_events_rx) = pump.get_ext_channels();

        let (tx_writer_to_consumer, rx_writer_to_consumer) =
            unbounded::<AcquisitionResult<GenericFrame>>();

        // Two main configuration options:
        // 1) Writer enabled or not (currently can't disable writer)
        // 2) "Frame Consumer" enabled or not
        //
        // They are configured by wiring up channels in the correct way.
        let os_handle = shm.os_handle.clone();
        let bg_thread = if enable_frame_consumer {
            // Frame consumer enabled -> after writing, the writer thread should send the frames
            // to the `tx_frame_consumer` channel
            match mode {
                CameraMode::IS => Some(start_bg_thread::<K2ISFrame, { K2ISBlock::PACKET_SIZE }>(
                    events,
                    addr_config.clone(),
                    pump,
                    tx_writer_to_consumer,
                    shm,
                )),
                CameraMode::Summit => Some(start_bg_thread::<
                    K2SummitFrame,
                    { K2SummitBlock::PACKET_SIZE },
                >(
                    events,
                    addr_config.clone(),
                    pump,
                    tx_writer_to_consumer,
                    shm,
                )),
            }
        } else {
            // Directly recycle after writing:
            todo!("implement a tx_writer_to_consumer that just frees the shm");
        };

        let mut state_tracker = StateTracker::new();

        // Wait for the init event, so we know the background thread is started.
        // We currently don't wait for all threads, but that's fine, as it's
        // most important that the receiver for control messages is already
        // listening, as events are buffered in a channel:
        let tracer = get_tracer();
        tracer.in_span("AcquisitionRuntime wait_for_init", |_cx| loop {
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
        });

        let shm = SharedSlabAllocator::connect(&os_handle).expect("connect to shm");

        AcquisitionRuntime {
            bg_thread,
            main_events_tx,
            main_events_rx,
            rx_writer_to_consumer,
            enable_frame_consumer,
            state_tracker,
            shm,
            current_acquisition_id: 0,
        }
    }

    /// Receive at most one event from the event bus and update the state.
    pub fn update_state(&mut self) -> UpdateStateResult {
        match self.main_events_rx.try_recv() {
            Ok(msg) => match self.state_tracker.set_state_from_msg(&msg) {
                Ok(_) => UpdateStateResult::DidUpdate,
                Err(StateError::InvalidTransition { from, msg }) => {
                    panic!("invalid state transition: from={from:?} msg={msg:?}")
                }
            },
            Err(TryRecvError::Disconnected) => panic!("lost connection to background thread"),
            Err(TryRecvError::Empty) => UpdateStateResult::NoNewMessage,
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

    pub fn get_next_frame(&self) -> Result<AcquisitionResult<GenericFrame>, RuntimeError> {
        // FIXME: can we make this a non-issue somehow?
        if !self.enable_frame_consumer {
            return Err(RuntimeError::ConfigurationError);
        }
        let acquisition_result = self
            .rx_writer_to_consumer
            .recv_timeout(Duration::from_millis(100))?;
        // FIXME! frames are not yet ordered by index
        // so sentinels can come out of order (ugh!)
        match acquisition_result {
            AcquisitionResult::DoneSuccess {
                acquisition_id,
                dropped: _,
            }
            | AcquisitionResult::DoneShuttingDown { acquisition_id }
            | AcquisitionResult::DoneAborted {
                acquisition_id,
                dropped: _,
            } => {
                self.main_events_tx
                    .send(EventMsg::ProcessingDone { acquisition_id })?;
                debug!("AcquisitionRuntime::get_next_frame: {acquisition_result:?}");
            }
            _ => {}
        };
        Ok(acquisition_result)
    }

    pub fn frame_done(
        &mut self,
        frame: AcquisitionResult<GenericFrame>,
    ) -> Result<(), RuntimeError> {
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

    pub fn get_frame_slot(&mut self, frame: AcquisitionResult<GenericFrame>) -> Option<usize> {
        let frame_inner = frame.unpack()?;
        Some(frame_inner.into_slot(&self.shm).slot_idx)
    }

    pub fn arm(&mut self, params: AcquisitionParams) -> Result<(), RuntimeError> {
        self.current_acquisition_id += 1;
        self.main_events_tx.send(EventMsg::Arm {
            params,
            acquisition_id: self.current_acquisition_id,
        })?;
        Ok(())
    }

    pub fn get_current_acquisition_id(&self) -> usize {
        self.current_acquisition_id
    }

    pub fn stop(&mut self) -> Result<(), RuntimeError> {
        // FIXME: do we need to do anything special if an acquisition is
        // currently running?
        // self.events
        //     .send(&k2o::events::EventMsg::CancelAcquisition {});
        self.main_events_tx.send(EventMsg::Shutdown {})?;
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

    pub fn wait_predicate<P>(&mut self, timeout: Duration, pred: P) -> WaitResult
    where
        P: Fn(&Self) -> bool,
    {
        let deadline = Instant::now() + timeout;
        // bail out of predicate is true already:
        if pred(self) {
            log::trace!("predicate success");
            return WaitResult::PredSuccess;
        }
        loop {
            log::trace!("updating state");
            let update_result = self.update_state();
            // only sleep if there was no new message, so we can handle
            // storms of events efficiently here:
            match update_result {
                UpdateStateResult::DidUpdate => {
                    log::trace!("checking predicate");
                    if pred(self) {
                        log::trace!("predicate success");
                        return WaitResult::PredSuccess;
                    }
                }
                UpdateStateResult::NoNewMessage => {
                    log::trace!("no state change, sleeping");
                    std::thread::sleep(Duration::from_millis(1));
                }
            }
            if Instant::now() > deadline {
                log::trace!("wait_predicate: timeout");
                return WaitResult::Timeout;
            }
        }
    }

    /// Wait until the acquisition has ended or the timeout expired
    pub fn wait_until_complete(&mut self, timeout: Duration) -> WaitResult {
        self.wait_predicate(timeout, |slf| slf.is_done())
    }

    /// Wait until the arm command succeeded
    pub fn wait_for_arm(&mut self, timeout: Duration) -> WaitResult {
        debug!("wait_for_arm");
        self.wait_predicate(timeout, |slf| {
            matches!(
                slf.state_tracker.state,
                AcquisitionState::Armed {
                    params: _,
                    acquisition_id: _
                }
            )
        })
    }

    pub fn wait_for_start(&mut self, timeout: Duration) -> WaitResult {
        self.wait_predicate(timeout, |slf| {
            matches!(
                slf.state_tracker.state,
                AcquisitionState::AcquisitionStarted {
                    params: _,
                    frame_id: _,
                    acquisition_id: _,
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
            | AcquisitionState::Armed {
                params: _,
                acquisition_id: _,
            }
            | AcquisitionState::Shutdown => {
                return None;
            }
            AcquisitionState::AcquisitionStarted {
                params,
                frame_id,
                acquisition_id: _,
            }
            | AcquisitionState::AcquisitionFinishing {
                params,
                frame_id,
                acquisition_id: _,
            } => (params, frame_id),
        };
        frame_in_acquisition(frame_id, *ref_frame_id, params)
    }
}
