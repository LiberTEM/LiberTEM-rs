use std::{
    fmt::Debug,
    io,
    sync::mpsc::{Receiver, Sender},
};

use ipc_test::slab::SlabInitError;
use pyo3::{exceptions::PyRuntimeError, pyclass, pymethods, PyResult};

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

impl From<ConcreteAcquisitionSize> for AcquisitionSize {
    fn from(value: ConcreteAcquisitionSize) -> Self {
        match value {
            ConcreteAcquisitionSize::NumFrames(n) => Self::NumFrames(n),
            ConcreteAcquisitionSize::Continuous => Self::Continuous,
        }
    }
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

    pub fn from_acquisition_size(size: AcquisitionSize) -> Self {
        Self { inner: size }
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

    /// Get the number of frames in this acquisition, raising a RuntimeError if there is none
    /// (i.e. the size is auto or continuous)
    pub fn get_num_frames(&self) -> PyResult<usize> {
        match &self.inner {
            AcquisitionSize::NumFrames(n) => Ok(*n),
            AcquisitionSize::Continuous | AcquisitionSize::Auto => {
                Err(PyRuntimeError::new_err(format!(
                    "PyAcquisitionSize::get_num_frames called on {:?}",
                    &self.inner
                )))
            }
        }
    }

    pub fn is_continuous(&self) -> bool {
        self.inner == AcquisitionSize::Continuous
    }

    pub fn is_auto(&self) -> bool {
        self.inner == AcquisitionSize::Auto
    }

    /// Do we know the number of frames in this acquisition?
    pub fn num_frames_known(&self) -> bool {
        matches!(self.inner, AcquisitionSize::NumFrames(_))
    }
}

/// Control messages from the foreground code to the background thread
#[derive(Debug)]
pub enum ControlMsg<CM: Debug> {
    /// Start listening for any acquisitions starting. Depending on the
    /// detector, the acquisition size needs to be passed in, or it can be
    /// determined automatically.
    StartAcquisitionPassive { acquisition_size: AcquisitionSize },

    /// Cancel the currently running acquisition, if any, going back to idle.
    /// Afterwards, another acquisition can be started, for example via
    /// `StartAcquisitionPassive`.
    CancelAcquisition,

    /// Stop processing ASAP
    StopThread,

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
