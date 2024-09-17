use std::{
    fmt::Debug,
    io,
    sync::mpsc::{Receiver, Sender},
};

use ipc_test::slab::SlabInitError;

use crate::{
    frame_stack::{FrameMeta, FrameStackHandle},
    generic_connection::AcquisitionConfig,
};

/// Messages from the background thread to the foreground code
#[derive(Debug)]
pub enum ReceiverMsg<M: FrameMeta, P: AcquisitionConfig> {
    /// A frame stack has been received and is ready for processing
    FrameStack { frame_stack: FrameStackHandle<M> },

    /// The acquisition is finished, `frame_stack` contains the remaining frames
    /// that were received.
    Finished { frame_stack: FrameStackHandle<M> },

    /// The acquisition was cancelled, as requested
    /// by `ControlMsg::CancelAcquisition`
    Cancelled,

    /// A non-recoverable error occurred, the underlying connection
    /// to the detector system should re-connect.
    FatalError {
        error: Box<dyn std::error::Error + 'static + Send + Sync>,
    },

    /// The receiver is armed for starting an acquisition
    ReceiverArmed,

    /// The acquisition has started, meaning that we are starting to receive
    /// data.
    // FIXME: do we need a separate message for arm vs. really starting to receive data?
    // these two things are often two distinct states of the receiver
    AcquisitionStart { pending_acquisition: P },
}

/// Control messages from the foreground code to the background thread
#[derive(Debug)]
pub enum ControlMsg<CM: Debug> {
    /// Stop processing ASAP
    StopThread,

    /// Start listening for any acquisitions starting
    StartAcquisitionPassive,

    /// Cancel the currently running acquisition, if any
    CancelAcquisition,

    /// Detector-specific control message
    SpecializedControlMsg { msg: CM },
}

#[derive(thiserror::Error, Debug)]
pub enum BackgroundThreadSpawnError {
    #[error("could not spawn background thread: {0}")]
    SpawnFailed(#[from] io::Error),

    #[error("shm clone/connect failed: {0}")]
    ShmConnectError(#[from] SlabInitError),
}

pub trait BackgroundThread {
    type FrameMetaImpl: FrameMeta;
    type AcquisitionConfigImpl: AcquisitionConfig;
    type ExtraControl: Debug;

    fn channel_to_thread(&mut self) -> &mut Sender<ControlMsg<Self::ExtraControl>>;

    fn channel_from_thread(
        &mut self,
    ) -> &mut Receiver<ReceiverMsg<Self::FrameMetaImpl, Self::AcquisitionConfigImpl>>;

    fn join(self);
}
