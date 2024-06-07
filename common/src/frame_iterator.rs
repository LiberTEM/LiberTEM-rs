use std::{error::Error, time::Duration};

use ipc_test::SharedSlabAllocator;
use log::trace;
use stats::Stats;

use crate::{
    background_thread::{BackgroundThread, ReceiverMsg},
    frame_stack::{FrameMeta, FrameStackHandle},
    generic_connection::{GenericConnection, PendingAcquisition},
};

#[derive(thiserror::Error, Debug)]
pub enum ChunkedIterError {
    #[error("receiver is closed")]
    ReceiverClosed,

    #[error("unrecoverable error: {0}")]
    UnrecoverableError(Box<dyn Error + 'static + Send + Sync>),

    #[error("periodic callback error: {0}")]
    PeriodicCallbackError(Box<dyn Error + 'static + Send + Sync>),
}

pub struct FrameChunkedIterator<'a, 'b, 'c, M, B, P>
where
    M: FrameMeta,
    B: BackgroundThread<M, P>,
    P: PendingAcquisition,
{
    receiver: &'a mut GenericConnection<M, B, P>,
    shm: &'b mut SharedSlabAllocator,
    stats: &'c mut Stats,
}

impl<'a, 'b, 'c, M, B, P> FrameChunkedIterator<'a, 'b, 'c, M, B, P>
where
    M: FrameMeta,
    B: BackgroundThread<M, P>,
    P: PendingAcquisition,
{
    /// Get the next frame stack. Mainly handles splitting logic for boundary
    /// conditions and delegates communication with the background thread to `recv_next_stack_impl`
    pub fn get_next_stack_impl<E: std::error::Error + 'static + Send + Sync>(
        &mut self,
        max_size: usize,
        periodic_callback: impl Fn() -> Result<(), E>,
    ) -> Result<Option<FrameStackHandle<M>>, ChunkedIterError> {
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
                    let (left, right) = frame_stack.split_at(max_size, self.shm);
                    self.receiver.get_remainder_mut().push(right);
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

    /// Receive the next frame stack from the background thread and handle any
    /// other control messages.
    fn recv_next_stack_impl<E: std::error::Error + 'static + Send + Sync>(
        &mut self,
        periodic_callback: impl Fn() -> Result<(), E>,
    ) -> Result<Option<FrameStackHandle<M>>, ChunkedIterError> {
        // first, check if there is anything on the remainder list:
        if let Some(frame_stack) = self.receiver.get_remainder_mut().pop() {
            return Ok(Some(frame_stack));
        }

        match self.receiver.get_status() {
            ReceiverStatus::Closed => {
                return Err(ChunkedIterError::ReceiverClosed);
            }
            ReceiverStatus::Idle => return Ok(None),
            ReceiverStatus::Running => {}
            ReceiverStatus::Initializing => todo!(),
            ReceiverStatus::Armed => todo!(),
            ReceiverStatus::Cancelling => todo!(),
            ReceiverStatus::Finished => todo!(),
            ReceiverStatus::Ready => todo!(),
            ReceiverStatus::Shutdown => todo!(),
        }

        let recv = &mut self.receiver;

        loop {
            if let Err(e) = periodic_callback() {
                return Err(ChunkedIterError::PeriodicCallbackError(Box::new(e)));
            }

            let recv_result = recv.next_timeout(Duration::from_millis(100));

            match recv_result {
                None => {
                    continue;
                }
                Some(ReceiverMsg::AcquisitionStart { .. }) => {
                    // FIXME: in case of "passive" mode, we should actually not hit this,
                    // as the "outer" structure (`DectrisConnection`) handles it?
                    continue;
                }
                Some(ReceiverMsg::FatalError { error }) => {
                    return Err(ChunkedIterError::UnrecoverableError(error));
                }
                Some(ReceiverMsg::Finished { frame_stack }) => {
                    self.stats.log_stats();
                    self.stats.reset();
                    return Ok(Some(frame_stack));
                }
                Some(ReceiverMsg::FrameStack { frame_stack }) => {
                    return Ok(Some(frame_stack));
                }
            }
        }
    }

    /// Create a ``FrameChunkedIterator``. The iterator doesn't have its own
    /// state, and it's meant to be instantiated only temporarily.
    pub fn new(
        receiver: &'a mut R,
        shm: &'b mut SharedSlabAllocator,
        stats: &'c mut Stats,
    ) -> Self {
        FrameChunkedIterator {
            receiver,
            shm,
            stats,
        }
    }
}
