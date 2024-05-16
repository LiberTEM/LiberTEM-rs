use std::{error::Error, time::Duration};

use ipc_test::SharedSlabAllocator;
use log::trace;
use stats::Stats;

use crate::{
    frame_stack::{FrameMeta, FrameStackHandle},
    generic_receiver::{Receiver, ReceiverMsg, ReceiverStatus},
};

#[derive(thiserror::Error, Debug)]
pub enum ChunkedIterError {
    #[error("receiver is closed")]
    ReceiverClosed,

    #[error("unrecoverable error: {0}")]
    UnrecoverableError(Box<dyn Error>),

    #[error("periodic callback error: {0}")]
    PeriodicCallbackError(Box<dyn Error>),
}

pub struct FrameChunkedIterator<'a, 'b, 'c, 'd, M, R>
where
    M: FrameMeta,
    R: Receiver<M>,
{
    receiver: &'a mut R,
    shm: &'b mut SharedSlabAllocator,
    remainder: &'c mut Vec<FrameStackHandle<M>>,
    stats: &'d mut Stats,
}

impl<'a, 'b, 'c, 'd, M, R> FrameChunkedIterator<'a, 'b, 'c, 'd, M, R>
where
    M: FrameMeta,
    R: Receiver<M>,
{
    /// Create a ``FrameChunkedIterator``. The iterator doesn't have its own
    /// state, and it's meant to be instantiated only temporarily.
    pub fn new(
        receiver: &'a mut R,
        shm: &'b mut SharedSlabAllocator,
        remainder: &'c mut Vec<FrameStackHandle<M>>,
        stats: &'d mut Stats,
    ) -> Self {
        Self { receiver, shm, remainder, stats }
    }

    /// Get the next frame stack. Mainly handles splitting logic for boundary
    /// conditions and delegates communication with the background thread to `recv_next_stack_impl`
    pub fn get_next_stack_impl<E: std::error::Error>(
        &mut self,
        max_size: usize,
        periodic_callback: dyn Fn() -> Result<(), E>,
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

    /// Receive the next frame stack from the background thread and handle any
    /// other control messages.
    fn recv_next_stack_impl<E: std::error::Error>(
        &mut self,
        periodic_callback: dyn Fn() -> Result<(), E>,
    ) -> Result<Option<FrameStackHandle<M>>, ChunkedIterError> {
        // first, check if there is anything on the remainder list:
        if let Some(frame_stack) = self.remainder.pop() {
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
                let a = Box::new(e);
                return Err(ChunkedIterError::PeriodicCallbackError(Box::new(e)));
            }

            // FIXME: need to drop the GIL while receiving the next frame stack!
            // need to figure out how to do this, if we can just wrap the whole
            // thing into an `allow_threads` scope, or if we need yet another
            // "callback"-like pattern...

            let recv_result = recv.next_timeout(Duration::from_millis(100));

            match recv_result {
                None => {
                    continue;
                }
                Some(ReceiverMsg::AcquisitionStart {
                    series: _,
                    detector_config: _,
                }) => {
                    // FIXME: in case of "passive" mode, we should actually not hit this,
                    // as the "outer" structure (`DectrisConnection`) handles it?
                    continue;
                }
                Some(ResultMsg::SerdeError { msg, recvd_msg }) => {
                    return Err(exceptions::PyRuntimeError::new_err(format!(
                        "serialization error: {}, message: {}",
                        msg, recvd_msg
                    )))
                }
                Some(ResultMsg::Error { msg }) => {
                    return Err(exceptions::PyRuntimeError::new_err(msg))
                }
                Some(ResultMsg::End { frame_stack }) => {
                    self.stats.log_stats();
                    self.stats.reset();
                    return Ok(Some(frame_stack));
                }
                Some(ResultMsg::FrameStack { frame_stack }) => {
                    return Ok(Some(frame_stack));
                }
            }
        }
    }

    fn new(
        receiver: &'a mut DectrisReceiver,
        shm: &'b mut SharedSlabAllocator,
        remainder: &'c mut Vec<FrameStackHandle<DectrisFrameMeta>>,
        stats: &'d mut Stats,
    ) -> PyResult<Self> {
        Ok(FrameChunkedIterator {
            receiver,
            shm,
            remainder,
            stats,
        })
    }
}
