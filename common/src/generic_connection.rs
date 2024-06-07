use std::{
    fmt::Debug,
    marker::PhantomData,
    path::PathBuf,
    sync::mpsc::RecvTimeoutError,
    time::{Duration, Instant},
};

use ipc_test::SharedSlabAllocator;
use log::{trace, warn};
use stats::Stats;

use crate::{
    background_thread::{BackgroundThread, BackgroundThreadSpawnError, ControlMsg, ReceiverMsg},
    frame_stack::{FrameMeta, FrameStackHandle},
};

pub trait DetectorConnectionConfig: Clone {
    /// calculate number of SHM slots
    fn get_shm_num_slots(&self) -> usize;

    /// calculate SHM slot size
    fn get_shm_slot_size(&self) -> usize;

    /// should huge pages be enabled, if available?
    fn get_shm_enable_huge_pages(&self) -> bool;

    /// SHM handle path
    fn get_shm_handle_path(&self) -> String;
}

pub trait PendingAcquisition: Debug {
    fn num_frames(&self) -> usize;
}

#[derive(thiserror::Error, Debug)]
pub enum ConnectionError {
    #[error("priodic callback returned an error: {0}")]
    PeriodicCallbackError(Box<dyn std::error::Error + 'static + Sync + Send>),

    #[error("a general fatal error occurred: {0}")]
    FatalError(Box<dyn std::error::Error + 'static + Sync + Send>),

    #[error("could not create SHM area (num_slots={num_slots}, slot_size={slot_size}, total_size={total_size}): {err:?}")]
    ShmCreateError {
        num_slots: usize,
        slot_size: usize,
        total_size: usize,
        err: Box<dyn std::error::Error + 'static + Sync + Send>,
    },

    #[error("background thread is dead")]
    Disconnected,

    #[error("background thread failed to start: {0}")]
    SpawnFailed(#[from] BackgroundThreadSpawnError),

    #[error("unexpected message: {0}")]
    UnexpectedMessage(String),
}

/// "mirror" of `RecvTimeoutError`
#[derive(Debug, thiserror::Error)]
pub enum NextTimeoutError {
    #[error("timeout expired")]
    Timeout,

    #[error("the background thread is dead and the channel is disconnected")]
    Disconnected,
}

impl From<RecvTimeoutError> for NextTimeoutError {
    fn from(value: RecvTimeoutError) -> Self {
        match value {
            RecvTimeoutError::Timeout => Self::Timeout,
            RecvTimeoutError::Disconnected => Self::Disconnected,
        }
    }
}

impl From<NextTimeoutError> for ConnectionError {
    fn from(value: NextTimeoutError) -> Self {
        Self::FatalError(Box::new(value))
    }
}

#[derive(Debug)]
pub enum ConnectionStatus {
    Running,
    Idle,
}

pub struct GenericConnection<M, B, P>
where
    M: FrameMeta,
    B: BackgroundThread<M, P>,
    P: PendingAcquisition,
{
    remainder: Vec<FrameStackHandle<M>>,
    shm: SharedSlabAllocator,
    stats: Stats,
    status: ConnectionStatus,
    bg_thread: B,

    // TODO: get rid of this?
    _marker: PhantomData<P>,
}

impl<M: FrameMeta, B, P> GenericConnection<M, B, P>
where
    B: BackgroundThread<M, P>,
    P: PendingAcquisition,
{
    pub fn new<D>(config: &D) -> Result<Self, ConnectionError>
    where
        D: DetectorConnectionConfig,
    {
        let num_slots = config.get_shm_num_slots();
        let slot_size = config.get_shm_slot_size();
        let enable_huge_pages = config.get_shm_enable_huge_pages();
        let shm = SharedSlabAllocator::new(
            num_slots,
            slot_size,
            enable_huge_pages,
            &PathBuf::from(config.get_shm_handle_path()),
        )
        .map_err(|e| ConnectionError::ShmCreateError {
            num_slots,
            slot_size,
            total_size: num_slots * slot_size,
            err: Box::new(e),
        })?;
        Ok(Self {
            remainder: Vec::new(),
            shm,
            stats: Stats::new(),
            status: ConnectionStatus::Idle,
            bg_thread: BackgroundThread::spawn(config)?,
            _marker: PhantomData,
        })
    }

    pub fn get_status(&self) -> ConnectionStatus {
        self.status
    }

    fn adjust_status(&mut self, msg: &ReceiverMsg<M, P>) {
        match msg {
            ReceiverMsg::AcquisitionStart { .. } => {
                self.status = ConnectionStatus::Running;
            }
            ReceiverMsg::Finished { .. } => {
                self.status = ConnectionStatus::Idle;
            }
            _ => {}
        }
    }

    /// Get the next message from the background thread, waiting at most
    /// `timeout`.
    ///
    /// If a `NextTimeoutError::Disconnected` error is encountered,
    pub fn next_timeout(
        &mut self,
        timeout: Duration,
    ) -> Result<ReceiverMsg<M, P>, NextTimeoutError> {
        let msg = self.bg_thread.channel_from_thread().recv_timeout(timeout)?;
        self.adjust_status(&msg);
        Ok(msg)
    }

    /// Wait until the detector is armed, or until `timeout` expires. Returns
    /// a `PendingAcquisition`, or `None` in case of timeout.
    pub fn wait_for_arm<E>(
        &mut self,
        timeout: Duration,
        periodic_callback: impl Fn() -> Result<(), E>,
    ) -> Result<Option<P>, ConnectionError>
    where
        E: std::error::Error + 'static + Send + Sync,
    {
        let deadline = Instant::now() + timeout;
        let step = Duration::from_millis(100);

        loop {
            if let Err(e) = periodic_callback() {
                return Err(ConnectionError::PeriodicCallbackError(Box::new(e)));
            }

            let timeout_rem = deadline - Instant::now();
            let this_timeout = timeout_rem.min(step);
            let res = self.next_timeout(this_timeout);
            if let Err(NextTimeoutError::Timeout) = &res {
                if Instant::now() > deadline {
                    return Ok(None);
                } else {
                    continue;
                }
            };
            let res = res?;

            match res {
                ReceiverMsg::AcquisitionStart {
                    pending_acquisition,
                } => return Ok(Some(pending_acquisition)),
                msg @ ReceiverMsg::Finished { .. } | msg @ ReceiverMsg::FrameStack { .. } => {
                    // FIXME: we might want to log + ignore instead?
                    let err = format!("unexpected message: {:?}", msg);
                    return Err(ConnectionError::UnexpectedMessage(err));
                }
                ReceiverMsg::FatalError { error } => {
                    return Err(ConnectionError::FatalError(error))
                }
            }
        }
    }

    fn start_passive(&mut self) -> Result<(), ConnectionError> {
        self.bg_thread
            .channel_to_thread()
            .send(ControlMsg::StartAcquisitionPassive)
            .map_err(|e| ConnectionError::Disconnected)
    }

    /// Receive the next frame stack from the background thread and handle any
    /// other control messages.
    fn recv_next_stack_impl<E: std::error::Error + 'static + Send + Sync>(
        &mut self,
        periodic_callback: impl Fn() -> Result<(), E>,
    ) -> Result<Option<FrameStackHandle<M>>, ConnectionError> {
        // first, check if there is anything on the remainder list:
        if let Some(frame_stack) = self.remainder.pop() {
            return Ok(Some(frame_stack));
        }

        match self.status {
            ConnectionStatus::Running => {}
            ConnectionStatus::Idle => return Ok(None),
        };

        loop {
            if let Err(e) = periodic_callback() {
                return Err(ConnectionError::PeriodicCallbackError(Box::new(e)));
            }

            let recv_result = self.next_timeout(Duration::from_millis(100));
            if let Err(NextTimeoutError::Timeout) = &recv_result {
                continue;
            }
            let msg = recv_result?;

            match msg {
                ReceiverMsg::AcquisitionStart { .. } => {
                    // FIXME: in case of "passive" mode, we should actually not hit this,
                    // as the "outer" structure (`DectrisConnection`) handles it?
                    continue;
                }
                ReceiverMsg::FatalError { error } => {
                    return Err(ConnectionError::FatalError(error));
                }
                ReceiverMsg::Finished { frame_stack } => {
                    self.stats.log_stats();
                    self.stats.reset();
                    return Ok(Some(frame_stack));
                }
                ReceiverMsg::FrameStack { frame_stack } => {
                    return Ok(Some(frame_stack));
                }
            }
        }
    }

    pub fn get_next_stack_impl<E: std::error::Error + 'static + Send + Sync>(
        &mut self,
        max_size: usize,
        periodic_callback: impl Fn() -> Result<(), E>,
    ) -> Result<Option<FrameStackHandle<M>>, ConnectionError> {
        let res = self.recv_next_stack_impl(periodic_callback);
        match res {
            Ok(Some(frame_stack)) => {
                if frame_stack.len() > max_size {
                    // split `FrameStackHandle` into two:
                    trace!(
                        "FrameStackHandle::split_at({max_size}); len={}",
                        frame_stack.len()
                    );
                    self.stats.count_split();
                    let (left, right) = frame_stack.split_at(max_size, &mut self.shm);
                    self.remainder.push(right);
                    assert!(left.len() <= max_size);
                    return Ok(Some(left));
                }
                assert!(frame_stack.len() <= max_size);
                Ok(Some(frame_stack))
            }
            Ok(None) => Ok(None),
            e @ Err(_) => e,
        }
    }

    fn close(self) {
        if self
            .bg_thread
            .channel_to_thread()
            .send(ControlMsg::StopThread)
            .is_err()
        {
            warn!("channel to background thread disconnected, probably already dead");
        }
        self.bg_thread.join();
    }
}
