use std::time::Duration;

use crate::frame_stack::{FrameMeta, FrameStackHandle};

/// Messages from the background thread to the foreground code
#[derive(PartialEq, Eq, Debug)]
pub enum ReceiverMsg<M: FrameMeta> {
    /// A frame stack has been received and is ready for processing
    FrameStack {
        frame_stack: FrameStackHandle<M>,
    },

    /// 
    Stuff,
}

pub enum ReceiverStatus {
    Initializing,
    Idle,
    Armed,
    Running,
    Cancelling,
    Finished,
    Ready,
    Shutdown,
    Closed,
}

pub trait Receiver<M: FrameMeta> {
    fn get_status(&self) -> ReceiverStatus;

    /// Get the next message from the background thread,
    /// waiting at most `timeout`; returns `None` if no the timeout was hit.
    fn next_timeout(&mut self, timeout: Duration) -> Option<ReceiverMsg<M>>;
}
