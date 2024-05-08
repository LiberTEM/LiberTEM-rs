use std::{error::Error, time::Duration};

use ipc_test::SharedSlabAllocator;
use log::trace;
use stats::Stats;

use crate::{
    frame_stack::{FrameMeta, FrameStackHandle},
    generic_receiver::{Receiver, ReceiverStatus},
};

#[derive(thiserror::Error, Debug)]
pub enum ChunkedIterError {
    #[error("receiver is closed")]
    ReceiverClosed,

    #[error("unrecoverable error: {0}")]
    UnrecoverableError(Box<dyn Error>),
}

struct FrameChunkedIterator<'a, 'b, 'c, 'd, M, R>
where
    M: FrameMeta,
    R: Receiver,
{
    receiver: &'a mut R,
    shm: &'b mut SharedSlabAllocator,
    remainder: &'c mut Vec<FrameStackHandle<M>>,
    stats: &'d mut Stats,
}

impl<'a, 'b, 'c, 'd, M, R> FrameChunkedIterator<'a, 'b, 'c, 'd, M, R>
where
    M: FrameMeta,
    R: Receiver,
{
    /// Get the next frame stack. Mainly handles splitting logic for boundary
    /// conditions and delegates communication with the background thread to `recv_next_stack_impl`
    pub fn get_next_stack_impl(
        &mut self,
        max_size: usize,
    ) -> Result<Option<FrameStackHandle<M>>, ChunkedIterError> {
        let res = self.recv_next_stack_impl();
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
    fn recv_next_stack_impl(&mut self) -> Result<Option<FrameStackHandle<M>>, ChunkedIterError> {
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
            // FIXME: should we add a generic "check for cancel" callback instead?
            py.check_signals()?;

            // FIXME: how do we allow Python threads here, in generic code?
            // FIXME: do we need to pass down the `Python` object??
            let recv_result = py.allow_threads(|| {
                let next: Result<Option<ResultMsg>, Infallible> =
                    Ok(recv.next_timeout(Duration::from_millis(100)));
                next
            })?;

            match recv_result {
                None => {
                    continue;
                }
                Some(ResultMsg::AcquisitionStart {
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
