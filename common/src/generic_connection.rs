use std::{
    fmt::Debug,
    marker::PhantomData,
    path::PathBuf,
    sync::mpsc::RecvTimeoutError,
    time::{Duration, Instant},
};

use ipc_test::{slab::SlabInitError, SharedSlabAllocator};
use log::{debug, info, trace, warn};
use stats::Stats;

use crate::{
    background_thread::{BackgroundThread, BackgroundThreadSpawnError, ControlMsg, ReceiverMsg},
    frame_stack::{FrameMeta, FrameStackHandle, SplitError},
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

pub trait AcquisitionConfig: Debug {
    /// total number of frames in the acquisition
    fn num_frames(&self) -> usize;
}

#[derive(thiserror::Error, Debug)]
pub enum ConnectionError {
    #[error("periodic callback returned an error: {0}")]
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

    #[error("could not connect to SHM area: {0}")]
    ShmConnectError(#[from] SlabInitError),

    #[error("background thread is dead")]
    Disconnected,

    #[error("operation timed out")]
    Timeout,

    #[error("operation cancelled")]
    Cancelled,

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

fn split_with_wait<M: FrameMeta, E>(
    shm: &mut SharedSlabAllocator,
    frame_stack: FrameStackHandle<M>,
    max_size: usize,
    periodic_callback: impl Fn() -> Result<(), E>,
) -> Result<(FrameStackHandle<M>, FrameStackHandle<M>), ConnectionError>
where
    E: std::error::Error + 'static + Send + Sync,
{
    let mut frame_stack = frame_stack;
    loop {
        if let Err(e) = periodic_callback() {
            return Err(ConnectionError::PeriodicCallbackError(Box::new(e)));
        }

        match frame_stack.split_at(max_size, shm) {
            Ok((a, b)) => return Ok((a, b)),
            Err(SplitError::ShmFull(old_frame_stack)) => {
                trace!("shm is full; waiting...");
                std::thread::sleep(Duration::from_millis(1));
                frame_stack = old_frame_stack;
                continue;
            }
        };
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ConnectionStatus {
    Idle,
    Armed,
    Running,
}

pub struct GenericConnection<B, AC>
where
    B: BackgroundThread,
    AC: AcquisitionConfig,
{
    remainder: Vec<FrameStackHandle<B::FrameMetaImpl>>,
    shm: SharedSlabAllocator,
    stats: Stats,
    status: ConnectionStatus,
    bg_thread: B,

    // TODO: get rid of this?
    _marker: PhantomData<AC>,
}

impl<B, AC> GenericConnection<B, AC>
where
    B: BackgroundThread,
    AC: AcquisitionConfig,
{
    /// Create a new `GenericConnection`, taking ownership of the `bg_thread` passed in.
    pub fn new(bg_thread: B, shm: &SharedSlabAllocator) -> Result<Self, ConnectionError> {
        let shm = shm.clone_and_connect()?;
        Ok(Self {
            remainder: Vec::new(),
            shm,
            stats: Stats::new(),
            status: ConnectionStatus::Idle,
            bg_thread,
            _marker: PhantomData,
        })
    }

    /// Instantiate a `SharedSlabAllocator` from the configuration in `config`.
    pub fn shm_from_config<D>(config: &D) -> Result<SharedSlabAllocator, ConnectionError>
    where
        D: DetectorConnectionConfig,
    {
        let num_slots = config.get_shm_num_slots();
        let slot_size = config.get_shm_slot_size();
        let enable_huge_pages = config.get_shm_enable_huge_pages();
        SharedSlabAllocator::new(
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
        })
    }

    pub fn get_status(&self) -> ConnectionStatus {
        self.status
    }

    fn adjust_status(&mut self, msg: &ReceiverMsg<B::FrameMetaImpl, B::AcquisitionConfigImpl>) {
        match msg {
            ReceiverMsg::AcquisitionStart { .. } => {
                debug!("adjust_status: now Running");
                self.status = ConnectionStatus::Running;
            }
            ReceiverMsg::Finished { .. } => {
                debug!("adjust_status: now Idle");
                self.status = ConnectionStatus::Idle;
            }
            ReceiverMsg::ReceiverArmed => {
                debug!("adjust_status: now Armed");
                self.status = ConnectionStatus::Armed;
            }
            ReceiverMsg::FrameStack { .. } => {
                trace!("adjust_status: FrameStack {{ .. }}");
            }
            ReceiverMsg::FatalError { error } => {
                log::warn!("adjust_status: fatal error: {error:?}; going back to idle state");
                self.status = ConnectionStatus::Idle;
            }
            ReceiverMsg::Cancelled => {
                log::warn!("adjust_status: acquisition cancelled");
                self.status = ConnectionStatus::Idle;
            } // other => {
              //     trace!("adjust_status: other message: {other:?}");
              // }
        }
    }

    /// Get the next message from the background thread, waiting at most
    /// `timeout`.
    ///
    /// If a `NextTimeoutError::Disconnected` error is encountered, the background
    /// thread is no longer running.
    ///
    /// When handling the results, care must be taken not to leak memory,
    /// meaning the variants of `ReceiverMsg` that contain a `FrameStackHandle`
    /// must be either returned to user code or free'd. When ignored, they will
    /// fill up the shared memory.
    pub fn next_timeout(
        &mut self,
        timeout: Duration,
    ) -> Result<ReceiverMsg<B::FrameMetaImpl, B::AcquisitionConfigImpl>, NextTimeoutError> {
        let msg = self.bg_thread.channel_from_thread().recv_timeout(timeout)?;
        self.adjust_status(&msg);
        Ok(msg)
    }

    /// Wait until the detector is armed, or until `timeout` expires. Returns an
    /// `AcquisitionConfig` matching the `BackgroundThread` impl, or `None` in
    /// case of timeout.
    ///
    /// `periodic_callback` will be called about every 100ms - if that
    /// returns an `Err`, a `ConnectionError::PeriodicCallbackError`
    /// will be returned.
    ///
    /// If `timeout` is none, wait indefinitely, or until the
    /// `periodic_callback` returns an error.
    pub fn wait_for_arm<E>(
        &mut self,
        timeout: Option<Duration>,
        periodic_callback: impl Fn() -> Result<(), E>,
    ) -> Result<Option<B::AcquisitionConfigImpl>, ConnectionError>
    where
        E: std::error::Error + 'static + Send + Sync,
    {
        // if an acquisition is already running, cancel and wait for idle status:
        if self.is_running() {
            self.cancel(&timeout, &periodic_callback)?;
        }

        if let Some(timeout) = timeout {
            self.wait_for_arm_inner(timeout, periodic_callback)
        } else {
            // wait indefinitely:
            loop {
                let res = self.wait_for_arm_inner(Duration::from_millis(100), &periodic_callback);
                match res {
                    Err(ConnectionError::Timeout) => continue,
                    Ok(None) => continue,
                    other => break other, // error, or Some(result)
                }
            }
        }
    }

    fn wait_for_arm_inner<E>(
        &mut self,
        timeout: Duration,
        periodic_callback: impl Fn() -> Result<(), E>,
    ) -> Result<Option<B::AcquisitionConfigImpl>, ConnectionError>
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
                // the receiver should have been armed before, but no harm; just
                // wait for the next message:
                ReceiverMsg::ReceiverArmed => continue,

                ReceiverMsg::AcquisitionStart {
                    pending_acquisition,
                } => return Ok(Some(pending_acquisition)),
                ReceiverMsg::FatalError { error } => {
                    return Err(ConnectionError::FatalError(error))
                }
                ReceiverMsg::FrameStack { frame_stack } => {
                    frame_stack.free_slot(&mut self.shm);
                    return Err(ConnectionError::UnexpectedMessage(
                        "ReceiverMsg::FrameStack in wait_for_arm".to_owned(),
                    ));
                }
                ReceiverMsg::Finished { frame_stack } => {
                    frame_stack.free_slot(&mut self.shm);
                    return Err(ConnectionError::UnexpectedMessage(
                        "ReceiverMsg::Finished in wait_for_arm".to_owned(),
                    ));
                }
                ReceiverMsg::Cancelled => {
                    return Err(ConnectionError::UnexpectedMessage(
                        "ReceiverMsg::Cancelled in wait_for_arm".to_owned(),
                    ));
                }
            }
        }
    }

    /// Wait for a status change, or until `timeout` expires. If `timeout` is
    /// `None`, wait indefinitely. If the status is already equal to
    /// `desired_status`, return immediately.
    ///
    /// `periodic_callback` will be called about every 100ms - if that
    /// returns an `Err`, a `ConnectionError::PeriodicCallbackError`
    /// will be returned.
    pub fn wait_for_status<E>(
        &mut self,
        desired_status: ConnectionStatus,
        timeout: Option<Duration>,
        periodic_callback: impl Fn() -> Result<(), E>,
    ) -> Result<(), ConnectionError>
    where
        E: std::error::Error + 'static + Send + Sync,
    {
        if let Some(timeout) = timeout {
            self.wait_for_status_inner(desired_status, timeout, periodic_callback)
        } else {
            // wait indefinitely:
            loop {
                let res = self.wait_for_status_inner(
                    desired_status,
                    Duration::from_millis(100),
                    &periodic_callback,
                );
                if let Err(ConnectionError::Timeout) = &res {
                    continue;
                } else {
                    break res;
                }
            }
        }
    }

    fn wait_for_status_inner<E>(
        &mut self,
        desired_status: ConnectionStatus,
        timeout: Duration,
        periodic_callback: impl Fn() -> Result<(), E>,
    ) -> Result<(), ConnectionError>
    where
        E: std::error::Error + 'static + Send + Sync,
    {
        debug!("wait_for_status: waiting for {desired_status:?}...");
        let deadline = Instant::now() + timeout;
        let step = Duration::from_millis(100);

        if self.status == desired_status {
            debug!("wait_for_status: already in desired status: {desired_status:?}");
            return Ok(());
        }

        loop {
            if let Err(e) = periodic_callback() {
                return Err(ConnectionError::PeriodicCallbackError(Box::new(e)));
            }

            let timeout_rem = deadline - Instant::now();
            let this_timeout = timeout_rem.min(step);
            let res = self.next_timeout(this_timeout);
            if let Err(NextTimeoutError::Timeout) = &res {
                if Instant::now() > deadline {
                    return Err(ConnectionError::Timeout);
                } else {
                    continue;
                }
            };
            let res = res?;
            match res {
                ReceiverMsg::FrameStack { frame_stack } => {
                    trace!("wait_for_status: ignoring received FrameStackHandle");
                    frame_stack.free_slot(&mut self.shm);
                }
                ReceiverMsg::Finished { frame_stack } => {
                    warn!("wait_for_status: ignoring FrameStackHandle received in ReceiverMsg::Finished message");
                    frame_stack.free_slot(&mut self.shm);
                }
                ReceiverMsg::FatalError { error } => {
                    return Err(ConnectionError::FatalError(error));
                }
                ReceiverMsg::ReceiverArmed => {
                    trace!("wait_for_status: received ReceiverMsg::ReceiverArmed");
                }
                ReceiverMsg::AcquisitionStart {
                    pending_acquisition: _,
                } => {
                    trace!("wait_for_status: received ReceiverMsg::AcquisitionStart");
                }
                ReceiverMsg::Cancelled => {
                    trace!("wait_for_status: received ReceiverMsg::Cancelled");
                }
            }
            if self.status == desired_status {
                debug!("wait_for_status: successfully got status {desired_status:?}");
                return Ok(());
            }
        }
    }

    /// Start a new passive acquisition.
    pub fn start_passive<E>(
        &mut self,
        periodic_callback: impl Fn() -> Result<(), E>,
        timeout: &Option<Duration>,
    ) -> Result<(), ConnectionError>
    where
        E: std::error::Error + 'static + Send + Sync,
    {
        if self.status == ConnectionStatus::Armed {
            // already armed, don't have to do anything
            debug!("start_passive: already armed, nothing to do");
            return Ok(());
        }

        self.bg_thread
            .channel_to_thread()
            .send(ControlMsg::StartAcquisitionPassive)
            .map_err(|_e| ConnectionError::Disconnected)?;

        self.wait_for_status(ConnectionStatus::Armed, *timeout, periodic_callback)
    }

    /// Receive the next frame stack from the background thread and handle any
    /// other control messages.
    fn recv_next_stack<E: std::error::Error + 'static + Send + Sync>(
        &mut self,
        periodic_callback: impl Fn() -> Result<(), E>,
    ) -> Result<Option<FrameStackHandle<B::FrameMetaImpl>>, ConnectionError> {
        // first, check if there is anything on the remainder list:
        if let Some(frame_stack) = self.remainder.pop() {
            return Ok(Some(frame_stack));
        }

        match self.status {
            ConnectionStatus::Running | ConnectionStatus::Armed => {}
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
                m @ ReceiverMsg::ReceiverArmed => {
                    info!("recv_next_stack: unexpected message: {m:?}");
                    continue;
                }
                ReceiverMsg::AcquisitionStart { .. } => {
                    // FIXME: in case of "passive" mode, we should actually not hit this,
                    // as `wait_for_arm` consumes it?
                    continue;
                }
                ReceiverMsg::FatalError { error } => {
                    return Err(ConnectionError::FatalError(error));
                }
                ReceiverMsg::Finished { frame_stack } => {
                    // Finished here means we have seen all frame stacks of the acquisition,
                    // it does _not_ mean that the data consumer has processed them all.

                    // do stats update here to make sure we count the last frame stack!
                    self.stats.count_stats_item(&frame_stack);
                    self.stats.log_stats();
                    self.stats.reset();

                    return Ok(Some(frame_stack));
                }
                ReceiverMsg::FrameStack { frame_stack } => {
                    self.stats.count_stats_item(&frame_stack);
                    return Ok(Some(frame_stack));
                }
                ReceiverMsg::Cancelled => {
                    return Err(ConnectionError::Cancelled);
                }
            }
        }
    }

    pub fn get_next_stack<E: std::error::Error + 'static + Send + Sync>(
        &mut self,
        max_size: usize,
        periodic_callback: impl Fn() -> Result<(), E>,
    ) -> Result<Option<FrameStackHandle<B::FrameMetaImpl>>, ConnectionError> {
        let res = self.recv_next_stack(&periodic_callback);
        match res {
            Ok(Some(frame_stack)) => {
                if frame_stack.len() > max_size {
                    // split `FrameStackHandle` into two:
                    trace!(
                        "FrameStackHandle::split_at({max_size}); len={}",
                        frame_stack.len()
                    );
                    self.stats.count_split();
                    let (left, right) =
                        split_with_wait(&mut self.shm, frame_stack, max_size, &periodic_callback)?;
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

    pub fn is_running(&self) -> bool {
        self.get_status() == ConnectionStatus::Running
    }

    pub fn cancel<E>(
        &mut self,
        timeout: &Option<Duration>,
        periodic_callback: impl Fn() -> Result<(), E>,
    ) -> Result<(), ConnectionError>
    where
        E: std::error::Error + 'static + Send + Sync,
    {
        self.bg_thread
            .channel_to_thread()
            .send(ControlMsg::CancelAcquisition)
            .map_err(|_| ConnectionError::Disconnected)?;

        self.wait_for_status(ConnectionStatus::Idle, *timeout, periodic_callback)
    }

    pub fn log_shm_stats(&self) {
        let shm = &self.shm;
        let free = shm.num_slots_free();
        let total = shm.num_slots_total();
        self.stats.log_stats();
        info!("shm stats free/total: {}/{}", free, total);
    }

    pub fn reset_stats(&mut self) {
        self.stats.reset();
    }

    pub fn send_specialized(&mut self, msg: B::ExtraControl) -> Result<(), ConnectionError> {
        self.bg_thread
            .channel_to_thread()
            .send(ControlMsg::SpecializedControlMsg { msg })
            .map_err(|_| ConnectionError::Disconnected)?;
        Ok(())
    }

    pub fn close(mut self) {
        debug!("GenericConnection::close");
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
