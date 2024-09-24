use std::{
    sync::mpsc::{channel, Receiver, RecvTimeoutError, Sender, TryRecvError},
    thread::JoinHandle,
    time::Duration,
};

use common::{
    background_thread::{AcquisitionSize, BackgroundThread, BackgroundThreadSpawnError, ConcreteAcquisitionSize, ControlMsg, ReceiverMsg},
    frame_stack::FrameStackHandle,
};
use crossbeam::channel::{
    unbounded, Receiver as CReceiver, RecvTimeoutError as CRecvTimeoutError, Sender as CSender,
};
use ipc_test::SharedSlabAllocator;
use k2o::{
    acquisition::AcquisitionResult,
    block::K2Block,
    block_is::K2ISBlock,
    block_summit::K2SummitBlock,
    events::{
        AcquisitionParams, AcquisitionSync, ChannelEventBus, EventBus, EventMsg,
        Events, MessagePump,
    },
    frame::GenericFrame,
    frame_is::K2ISFrame,
    frame_summit::K2SummitFrame,
    recv::RecvConfig,
    runtime::{start_bg_thread, AddrConfig, AssemblyConfig},
};
use log::{debug, error, info, warn};
use opentelemetry::Context;

use crate::{
    config::{K2AcquisitionConfig, K2DetectorConnectionConfig, K2Mode},
    frame_meta::K2FrameMeta,
};

type K2ControlMsg = ControlMsg<()>;

type K2ReceiverMsg = ReceiverMsg<K2FrameMeta, K2AcquisitionConfig>;

#[derive(Debug, Clone, thiserror::Error)]
pub enum AcquisitionError {
    #[error("cancelled by the user")]
    Cancelled,

    #[error("disconnected")]
    Disconnected,

    #[error("thread stopped")]
    ThreadStopped,

    #[error("receiver state error: {msg}")]
    StateError { msg: String },
}

#[derive(Debug, Clone, Copy)]
enum PassiveAcquisitionControlFlow {
    Shutdown,
    Continue,
}

/// With a running acquisition, check for control messages;
/// especially convert `ControlMsg::StopThread` to `AcquisitionError::Cancelled`.
fn check_for_control(control_channel: &Receiver<K2ControlMsg>) -> Result<(), AcquisitionError> {
    match control_channel.try_recv() {
        Ok(m @ ControlMsg::StartAcquisitionPassive { .. }) => Err(AcquisitionError::StateError {
            msg: format!("received {m:?} while an acquisition was already running")
        }),
        Ok(ControlMsg::StopThread) => Err(AcquisitionError::ThreadStopped),
        Ok(ControlMsg::SpecializedControlMsg { msg: _ }) => {
            panic!("unsupported SpecializedControlMsg")
        }
        Ok(ControlMsg::CancelAcquisition) => Err(AcquisitionError::Cancelled),
        Err(TryRecvError::Disconnected) => Err(AcquisitionError::Disconnected),
        Err(TryRecvError::Empty) => Ok(()),
    }
}

/// Start a passive acquisition. This always synchronizes with the sync flag,
/// meaning with the STEM scan.
fn passive_acquisition(
    acquisition_size: AcquisitionSize,
    config: &K2DetectorConnectionConfig,
    to_thread_r: &Receiver<K2ControlMsg>,
    from_thread_s: &Sender<K2ReceiverMsg>,
    main_events_tx: &CSender<EventMsg>,
    rx_writer_to_consumer: &CReceiver<AcquisitionResult<GenericFrame>>,
) -> Result<PassiveAcquisitionControlFlow, AcquisitionError> {
    main_events_tx
        .send(EventMsg::Arm {
            params: AcquisitionParams {
                size: acquisition_size,
                sync: AcquisitionSync::Immediately,
                binning: k2o::events::Binning::Bin1x,
            },
            acquisition_id: 1,
        })
        .unwrap();

    let effective_frame_shape = config.effective_shape();

    let acq_size = match acquisition_size {
        AcquisitionSize::Auto => ConcreteAcquisitionSize::Continuous,
        AcquisitionSize::NumFrames(n) => ConcreteAcquisitionSize::NumFrames(n),
        AcquisitionSize::Continuous => ConcreteAcquisitionSize::Continuous,
    };

    from_thread_s
        .send(ReceiverMsg::AcquisitionStart {
            pending_acquisition: K2AcquisitionConfig::new(acq_size, effective_frame_shape),
        })
        .unwrap();

    'acquisition: loop {
        check_for_control(&to_thread_r)?;

        match rx_writer_to_consumer.recv_timeout(Duration::from_millis(120)) {
            Ok(AcquisitionResult::Frame(f, idx)) => {
                let meta = vec![K2FrameMeta::new(
                    f.acquisition_id,
                    f.frame_id,
                    idx,
                    config.mode.get_frame_type(),
                    config.mode.get_bytes_per_pixel(),
                    config.crop_to_image_data,
                )];
                let slot_info = f.into_payload();
                let frame_stack = FrameStackHandle::new(slot_info, meta, vec![0], 0);
                from_thread_s
                    .send(ReceiverMsg::FrameStack { frame_stack })
                    .unwrap();
            }
            Ok(AcquisitionResult::DoneSuccess {
                dropped,
                acquisition_id,
            }) => {
                info!("acquisition {acquisition_id} done with {dropped} frames dropped");
                from_thread_s
                    .send(ReceiverMsg::Finished { frame_stack: None })
                    .unwrap();
                break 'acquisition;
            }
            Ok(AcquisitionResult::DoneAborted {
                dropped,
                acquisition_id,
            }) => {
                warn!("aborted acquisition {acquisition_id}");
                from_thread_s.send(ReceiverMsg::Cancelled).unwrap();
                break 'acquisition;
            }
            Ok(AcquisitionResult::DoneShuttingDown { acquisition_id }) => {
                warn!("shutting down; current acquisition {acquisition_id}");
                from_thread_s.send(ReceiverMsg::Cancelled).unwrap();
                return Ok(PassiveAcquisitionControlFlow::Shutdown);
            }
            Ok(result) => {
                info!("some result received: {result:?}");
            }
            Err(e) => {
                continue 'acquisition;
            }
        }
    }

    Ok(PassiveAcquisitionControlFlow::Continue)
}

fn background_thread(
    config: &K2DetectorConnectionConfig,
    to_thread_r: &Receiver<K2ControlMsg>,
    from_thread_s: &Sender<K2ReceiverMsg>,
    mut shm: SharedSlabAllocator,
) -> Result<(), AcquisitionError> {
    let events: Events = ChannelEventBus::new();
    let pump = MessagePump::new(&events);
    let (main_events_tx, main_events_rx) = pump.get_ext_channels();

    let (tx_writer_to_consumer, rx_writer_to_consumer) =
        unbounded::<AcquisitionResult<GenericFrame>>();

    let asm_config = AssemblyConfig::new(Duration::from_millis(25), config.assembly_realtime);
    let recv_config = RecvConfig::new(config.recv_realtime);
    let addr_config = AddrConfig::new(&config.local_addr_top, &config.local_addr_bottom);

    // FIXME: join this on drop, and/or use a threading scope
    let inner_bg_thread = match config.mode {
        K2Mode::IS => Some(start_bg_thread::<K2ISFrame, { K2ISBlock::PACKET_SIZE }>(
            events,
            addr_config,
            pump,
            tx_writer_to_consumer,
            shm.get_handle(),
            &asm_config,
            &recv_config,
        )),
        K2Mode::Summit => Some(start_bg_thread::<
            K2SummitFrame,
            { K2SummitBlock::PACKET_SIZE },
        >(
            events,
            addr_config,
            pump,
            tx_writer_to_consumer,
            shm.get_handle(),
            &asm_config,
            &recv_config,
        )),
    };

    loop {
        match main_events_rx.recv_timeout(Duration::from_millis(100)) {
            Ok(EventMsg::Init) => {
                info!("init done");
                break;
            }
            Ok(e) => {
                info!("unexpected event: {e:?}");
                continue;
            }
            Err(_timeout) => {
                info!("still waiting for init...");
                continue;
            }
        }
    }

    from_thread_s.send(ReceiverMsg::ReceiverArmed).unwrap();

    'outer: loop {
        loop {
            // control: main threads tells us what to do.
            let control = to_thread_r.recv_timeout(Duration::from_millis(100));
            match control {
                Ok(ControlMsg::StartAcquisitionPassive { acquisition_size }) => {
                    match passive_acquisition(
                        acquisition_size,
                        config,
                        to_thread_r,
                        from_thread_s,
                        &main_events_tx,
                        &rx_writer_to_consumer,
                    ) {
                        Ok(PassiveAcquisitionControlFlow::Continue) => {
                            info!("passive acquisition returned, waiting for next")
                        }
                        Ok(PassiveAcquisitionControlFlow::Shutdown) => {
                            warn!("shutting down background thread");
                            break 'outer;
                        }
                        Err(AcquisitionError::Cancelled) => {
                            info!("acquisition cancelled by user");
                            from_thread_s.send(ReceiverMsg::Cancelled).unwrap();
                            continue 'outer;
                        }
                        e @ Err(
                            AcquisitionError::Disconnected | AcquisitionError::ThreadStopped,
                        ) => {
                            info!("background_thread: terminating: {e:?}");
                            return Ok(());
                        }
                        Err(e) => {
                            error!("background_thread: error: {}; re-connecting", e);
                            from_thread_s
                                .send(ReceiverMsg::FatalError { error: Box::new(e) })
                                .unwrap();
                            continue 'outer;
                        }
                    }
                }
                Ok(ControlMsg::CancelAcquisition) => {
                    warn!(
                        "background_thread: got a CancelAcquisition message in main loop; ignoring"
                    );
                }
                Ok(ControlMsg::StopThread) => {
                    debug!("background_thread: got a StopThread message");
                    break 'outer;
                }
                Err(RecvTimeoutError::Disconnected) => {
                    warn!("background_thread: control channel has disconnected");
                    break 'outer;
                }
                Err(RecvTimeoutError::Timeout) => (), // no message, nothing to do
                Ok(ControlMsg::SpecializedControlMsg { msg: _ }) => {
                    panic!("ControlMsg::SpecializesControlMsg is unused for K2");
                }
            }
        }
    }
    main_events_tx.send(EventMsg::Shutdown).unwrap();
    inner_bg_thread.unwrap().join().unwrap();
    debug!("background_thread: is done");
    Ok(())
}

fn background_thread_wrap(
    config: &K2DetectorConnectionConfig,
    to_thread_r: &Receiver<K2ControlMsg>,
    from_thread_s: &Sender<K2ReceiverMsg>,
    shm: SharedSlabAllocator,
) {
    if let Err(err) = background_thread(config, to_thread_r, from_thread_s, shm) {
        log::error!("background_thread err'd: {}", err.to_string());
        // NOTE: `shm` is dropped in case of an error, so anyone who tries to connect afterwards
        // will get an error
        from_thread_s
            .send(ReceiverMsg::FatalError {
                error: Box::new(err),
            })
            .unwrap();
    }
}

pub struct K2BackgroundThread {
    bg_thread: JoinHandle<()>,
    to_thread: Sender<ControlMsg<()>>,
    from_thread: Receiver<ReceiverMsg<K2FrameMeta, K2AcquisitionConfig>>,
}

impl BackgroundThread for K2BackgroundThread {
    type FrameMetaImpl = K2FrameMeta;
    type AcquisitionConfigImpl = K2AcquisitionConfig;
    type ExtraControl = ();

    fn channel_to_thread(&mut self) -> &mut Sender<ControlMsg<Self::ExtraControl>> {
        &mut self.to_thread
    }

    fn channel_from_thread(
        &mut self,
    ) -> &mut Receiver<ReceiverMsg<Self::FrameMetaImpl, Self::AcquisitionConfigImpl>> {
        &mut self.from_thread
    }

    fn join(self) {
        if let Err(e) = self.bg_thread.join() {
            // FIXME: should we have an error boundary here instead and stop the panic?
            std::panic::resume_unwind(e)
        }
    }
}

impl K2BackgroundThread {
    pub fn spawn(
        config: &K2DetectorConnectionConfig,
        shm: &SharedSlabAllocator,
    ) -> Result<Self, BackgroundThreadSpawnError> {
        let (to_thread_s, to_thread_r) = channel();
        let (from_thread_s, from_thread_r) = channel();

        let builder = std::thread::Builder::new();
        let shm = shm.clone_and_connect()?;
        let config = config.clone();
        let ctx = Context::current();

        debug!("connection config: {config:?}");

        Ok(Self {
            bg_thread: builder
                .name("bg_thread".to_owned())
                .spawn(move || {
                    let _ctx_guard = ctx.attach();
                    background_thread_wrap(&config, &to_thread_r, &from_thread_s, shm)
                })
                .map_err(BackgroundThreadSpawnError::SpawnFailed)?,
            from_thread: from_thread_r,
            to_thread: to_thread_s,
        })
    }
}
