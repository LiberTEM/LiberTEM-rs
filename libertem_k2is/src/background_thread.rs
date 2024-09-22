use std::{
    sync::mpsc::{channel, Receiver, RecvTimeoutError, Sender},
    thread::JoinHandle,
    time::Duration,
};

use common::background_thread::{
    BackgroundThread, BackgroundThreadSpawnError, ControlMsg, ReceiverMsg,
};
use ipc_test::SharedSlabAllocator;
use log::{debug, error, info, warn};
use opentelemetry::Context;

use crate::{
    config::{K2AcquisitionConfig, K2DetectorConnectionConfig},
    frame_meta::K2FrameMeta,
};

type K2ControlMsg = ControlMsg<()>;

type K2ReceiverMsg = ReceiverMsg<K2FrameMeta, K2AcquisitionConfig>;

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

fn background_thread(
    config: &K2DetectorConnectionConfig,
    to_thread_r: &Receiver<K2ControlMsg>,
    from_thread_s: &Sender<K2ReceiverMsg>,
    mut shm: SharedSlabAllocator,
) -> Result<(), AcquisitionError> {
    'outer: loop {
        loop {
            // control: main threads tells us what to do
            let control = to_thread_r.recv_timeout(Duration::from_millis(100));
            match control {
                Ok(ControlMsg::StartAcquisitionPassive) => {
                    match passive_acquisition(to_thread_r, from_thread_s, config, &mut shm) {
                        Ok(_) => {}
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
