use std::{
    io,
    sync::mpsc::{Receiver, Sender},
};

use crate::{
    frame_stack::{FrameMeta, FrameStackHandle},
    generic_connection::{DetectorConnectionConfig, PendingAcquisition},
};

/// Messages from the background thread to the foreground code
#[derive(Debug)]
pub enum ReceiverMsg<M: FrameMeta, P: PendingAcquisition> {
    /// A frame stack has been received and is ready for processing
    FrameStack { frame_stack: FrameStackHandle<M> },

    /// The acquisition is finished, `frame_stack` contains the remaining frames
    /// that were received.
    Finished { frame_stack: FrameStackHandle<M> },

    /// A non-recoverable error occurred, the underlying connection
    /// to the detector system should re-connect.
    FatalError {
        error: Box<dyn std::error::Error + 'static + Send + Sync>,
    },

    /// The acquisition has started, meaning that we are starting to receive
    /// data.
    // FIXME: do we need a separate message for arm vs. really starting to receive data?
    // these two things are often two distinct states of the receiver
    AcquisitionStart { pending_acquisition: P },
}

/// Control messages from the foreground code to the background thread
#[derive(Debug)]
pub enum ControlMsg {
    /// Stop processing ASAP
    StopThread,

    /// Start listening for any acquisitions starting
    StartAcquisitionPassive,
    // TODO: for DECTRIS, we have `StartAcquisition { series: u64 }`, can we get
    // away without that?
}

#[derive(thiserror::Error, Debug)]
pub enum BackgroundThreadSpawnError {
    #[error("could not spawn background thread: {0}")]
    SpawnFailed(#[from] io::Error),
}

// TODO: how does this look like? method to start the thread? join it?
pub trait BackgroundThread<M: FrameMeta, P: PendingAcquisition> {
    fn spawn<D: DetectorConnectionConfig>(config: &D) -> Result<Self, BackgroundThreadSpawnError>
    where
        Self: std::marker::Sized;

    fn channel_to_thread(&self) -> &mut Sender<ControlMsg>;

    fn channel_from_thread(&self) -> &mut Receiver<ReceiverMsg<M, P>>;

    fn join(self);
}
