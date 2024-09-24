use std::{
    fmt::Debug,
    io,
    sync::mpsc::{Receiver, Sender},
};

use ipc_test::slab::SlabInitError;
use pyo3::{pyclass, pymethods};

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
    Finished {
        frame_stack: Option<FrameStackHandle<M>>,
    },

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

/// Like AcquisitionSize, but with the `Auto` resolved to either a number of frames or `Continuous`
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConcreteAcquisitionSize {
    /// Set the number of frames to the given value
    NumFrames(usize),

    /// Acquire data until a cancel command is received from the user
    Continuous,
}

/// Configured acquisition size
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AcquisitionSize {
    /// Automatically determine number of frames from acquisition headers or similar
    #[default]
    Auto,

    /// Set the number of frames to the given value
    NumFrames(usize),

    /// Acquire data until a cancel command is received from the user
    Continuous,
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct PyAcquisitionSize {
    inner: AcquisitionSize,
}

impl PyAcquisitionSize {
    pub fn inner(&self) -> AcquisitionSize {
        self.inner
    }
}

#[pymethods]
impl PyAcquisitionSize {
    #[staticmethod]
    pub fn from_num_frames(num_frames: usize) -> Self {
        Self {
            inner: AcquisitionSize::NumFrames(num_frames),
        }
    }

    #[staticmethod]
    pub fn auto() -> Self {
        Self {
            inner: AcquisitionSize::Auto,
        }
    }

    #[staticmethod]
    pub fn continuous() -> Self {
        Self {
            inner: AcquisitionSize::Continuous,
        }
    }
}

/// Control messages from the foreground code to the background thread
#[derive(Debug)]
pub enum ControlMsg<CM: Debug> {
    /// Stop processing ASAP
    StopThread,

    /// Start listening for any acquisitions starting. Depending on the
    /// detector, the acquisition size needs to be passed in, or it can be
    /// determined automatically.
    StartAcquisitionPassive { acquisition_size: AcquisitionSize },

    /// Cancel the currently running acquisition, if any, going back to idle.
    /// Afterwards, another acquisition can be started, for example via
    /// `StartAcquisitionPassive`.
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
