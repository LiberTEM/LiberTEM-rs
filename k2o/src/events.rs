use std::{
    fmt::Debug,
    sync::{Arc, Mutex},
    time::Duration,
};

use crossbeam_channel::{select, unbounded, Receiver, RecvError, SendError, Sender};
use log::{debug, info};

use crate::write::{DirectWriterBuilder, MMapWriterBuilder, NoopWriterBuilder, WriterBuilder};
#[cfg(feature = "hdf5")]
use k2o::write::HDF5WriterBuilder;

pub trait EventBus<T: Clone + Debug> {
    /// Send `msg` to all subscribers
    fn send(&self, msg: &T);

    /// Subscribe gives you a channel where you can receive events
    /// currently you cannot unsubscribe, so this must not be used for short-lived threads!
    fn subscribe(&self) -> Receiver<T>;
}

type ChannelList<T> = Vec<(Sender<T>, Receiver<T>)>;

/// Event bus based on crossbeam channel pairs
pub struct ChannelEventBus<T: Clone + Debug> {
    channels: Arc<Mutex<ChannelList<T>>>,
}

impl<T: Clone + Debug> ChannelEventBus<T> {
    pub fn new() -> Self {
        Self {
            channels: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

impl<T: Clone + Debug> Default for ChannelEventBus<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone + Debug> EventBus<T> for ChannelEventBus<T> {
    fn subscribe(&self) -> Receiver<T> {
        let (tx, rx) = unbounded::<T>();
        let mut channels = self.channels.lock().unwrap();
        channels.push((tx, rx.clone()));
        rx
    }

    fn send(&self, msg: &T) {
        debug!("Event: {:?}", msg);
        // FIXME: don't panic
        for (tx, _) in self.channels.lock().unwrap().iter() {
            tx.send(msg.clone()).unwrap();
        }
    }
}

#[derive(Debug)]
pub enum MessagePumpError {
    Disconnected,
}

impl<T> From<SendError<T>> for MessagePumpError {
    fn from(_: SendError<T>) -> Self {
        MessagePumpError::Disconnected
    }
}

impl From<RecvError> for MessagePumpError {
    fn from(_: RecvError) -> Self {
        MessagePumpError::Disconnected
    }
}

pub struct MessagePump {
    // our ends:
    rx_from_ext_to_bus: Receiver<EventMsg>,
    tx_from_bus_to_ext: Sender<EventMsg>,

    // the external ends:
    tx_from_ext_to_bus: Sender<EventMsg>,
    rx_from_bus_to_ext: Receiver<EventMsg>,

    // the receiver from the bus; the other end we can't own or reference here:
    rx_from_bus: Receiver<EventMsg>,
}

impl MessagePump {
    pub fn new(events: &Events) -> Self {
        let rx_from_bus = events.subscribe();

        let (tx_from_bus_to_ext, rx_from_bus_to_ext) = unbounded::<EventMsg>();
        let (tx_from_ext_to_bus, rx_from_ext_to_bus) = unbounded::<EventMsg>();

        MessagePump {
            rx_from_ext_to_bus,
            tx_from_bus_to_ext,
            tx_from_ext_to_bus,
            rx_from_bus_to_ext,
            rx_from_bus,
        }
    }

    /// Pump messages between external channels and the event bus.
    /// This is non-blocking and does nothing if there are no events.
    /// Will return a `MessagePumpError` in case an event is received and
    /// channel we try to forward to is disconnected.
    pub fn do_pump(&self, events: &Events) -> Result<(), MessagePumpError> {
        // from the bus to the external channel:
        if let Ok(msg) = self.rx_from_bus.try_recv() {
            self.tx_from_bus_to_ext.send(msg)?;
        }
        // from the external channel to the bus:
        if let Ok(msg) = self.rx_from_ext_to_bus.try_recv() {
            events.send(&msg);
        }
        Ok(())
    }

    pub fn do_pump_timeout(
        &self,
        events: &Events,
        timeout: Duration,
    ) -> Result<(), MessagePumpError> {
        select! {
            // from the bus to the external channel:
            recv(self.rx_from_bus) -> msg => {
                self.tx_from_bus_to_ext.send(msg?)?;
            }
            // from the external channel to the bus:
            recv(self.rx_from_ext_to_bus) -> msg => {
                events.send(&msg?);
            }
            default(timeout) => return Ok(()),
        }
        Ok(())
    }

    pub fn get_ext_channels(&self) -> (Sender<EventMsg>, Receiver<EventMsg>) {
        (
            self.tx_from_ext_to_bus.clone(),
            self.rx_from_bus_to_ext.clone(),
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AcquisitionSize {
    Continuous,
    NumFrames(u32),
}

impl From<Option<u32>> for AcquisitionSize {
    fn from(size: Option<u32>) -> Self {
        match size {
            None => AcquisitionSize::Continuous,
            Some(num_frames) => AcquisitionSize::NumFrames(num_frames),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AcquisitionSync {
    /// Wait for hardware synchronization. This means we first wait for
    /// the frame id to wrap around to 1, and then for the sync flag to be
    /// set in the received data.
    WaitForSync,

    /// Ignore synchronization completely and just start acquiring data,
    /// handling wrap-arounds gracefully
    Immediately,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Binning {
    /// 400 fps
    Bin1x,

    /// 800 fps
    Bin2x,

    /// 1200 fps
    Bin4x,

    /// 1600 fps
    Bin8x,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WriterType {
    Direct,
    Mmap,
    #[cfg(feature = "hdf5")]
    HDF5,
}

pub enum WriterTypeError {
    InvalidWriterType,
}

impl TryFrom<&str> for WriterType {
    type Error = WriterTypeError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "direct" => Ok(Self::Direct),
            "mmap" => Ok(Self::Mmap),
            #[cfg(feature = "hdf5")]
            "hdf5" => Ok(Self::HDF5),
            _ => Err(WriterTypeError::InvalidWriterType),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WriterSettings {
    Disabled,
    Enabled {
        method: WriterType,
        filename: String, // maybe change to a path type?
    },
}

impl WriterSettings {
    pub fn disabled() -> Self {
        Self::Disabled
    }

    pub fn new(method: &str, filename: &str) -> Result<Self, WriterTypeError> {
        Ok(Self::Enabled {
            method: WriterType::try_from(method)?,
            filename: filename.to_owned(),
        })
    }

    pub fn get_writer_builder(&self) -> Box<dyn WriterBuilder> {
        match self {
            Self::Disabled => NoopWriterBuilder::new(),
            Self::Enabled { method, filename } => match &method {
                WriterType::Direct => DirectWriterBuilder::for_filename(filename),
                WriterType::Mmap => MMapWriterBuilder::for_filename(filename),
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AcquisitionParams {
    pub size: AcquisitionSize,
    pub sync: AcquisitionSync,
    pub binning: Binning,
    pub writer_settings: WriterSettings,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EventMsg {
    /// Ready for commands
    Init,

    /// Sent when the sector `sector_id` has started the acquisition, for
    /// example when it has received the first block with the sync flag set;
    /// synchronizing to the given `frame_id`
    AcquisitionStartedSector {
        sector_id: u8,
        frame_id: u32,
        acquisition_id: usize,
    },

    /// Send when any sector has started the acquisition
    AcquisitionStarted {
        frame_id: u32,
        params: AcquisitionParams,
        acquisition_id: usize,
    },

    /// Send when a acquisition should be started
    Arm {
        params: AcquisitionParams,
        acquisition_id: usize,
    },

    /// Send when the sector receiver threads should start acquiring data
    ArmSectors {
        params: AcquisitionParams,
        acquisition_id: usize,
    },

    /// Send when the acquisition has ended (successfully or with an error)
    // FIXME: need to distinguish cases!
    AcquisitionEnded {
        acquisition_id: usize,
    },

    /// Send when the final consumer is done processing all data
    ProcessingDone {
        acquisition_id: usize,
    },

    /// Send when the currently running acquisition should be stopped.
    /// Depending on the `AcquisitionSize`, this can either be before finishing
    /// acquisition of the fixed number of frames, or for continuous,
    /// this successfully finishes the acquisition.
    CancelAcquisition {
        acquisition_id: usize,
    },

    /// Generic fatal acquisition error
    AcquisitionError {
        msg: String,
    },

    Shutdown,
}

pub type Events = ChannelEventBus<EventMsg>;
pub type EventReceiver = Receiver<EventMsg>;
