#![allow(clippy::borrow_deref_ref)]

use log::info;
use serde::{Deserialize, Serialize};

use pyo3::prelude::*;
use uuid::Uuid;
use zmq::{Context, Message, Socket, SocketEvent};

#[derive(Serialize, Deserialize, Debug, Clone)]
#[pyclass]
pub struct DSeriesAndType {
    pub series: u64,
    pub htype: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[pyclass]
pub struct DHeader {
    pub htype: String,
    pub header_detail: String,
    pub series: u64,
}

#[pymethods]
impl DHeader {
    #[new]
    fn new(series: u64) -> Self {
        DHeader {
            htype: "dheader-1.0".to_string(),
            header_detail: "basic".to_string(),
            series,
        }
    }
}

#[allow(clippy::upper_case_acronyms)]
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[pyclass]
pub enum TriggerMode {
    #[serde(rename = "exte")]
    EXTE,
    #[serde(rename = "inte")]
    INTE,
    #[serde(rename = "exts")]
    EXTS,
    #[serde(rename = "ints")]
    INTS,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[pyclass]
pub struct DetectorConfig {
    pub ntrigger: u64,
    pub nimages: u64,
    trigger_mode: TriggerMode,
    pub x_pixels_in_detector: u32,
    pub y_pixels_in_detector: u32,
    pub bit_depth_image: u32,
}

impl DetectorConfig {
    pub fn get_num_images(&self) -> u64 {
        match self.trigger_mode {
            TriggerMode::EXTE | TriggerMode::INTE => self.ntrigger,
            TriggerMode::EXTS | TriggerMode::INTS => self.nimages * self.ntrigger,
        }
    }

    pub fn get_shape(&self) -> (u32, u32) {
        (self.y_pixels_in_detector, self.x_pixels_in_detector)
    }

    pub fn get_num_pixels(&self) -> u64 {
        self.y_pixels_in_detector as u64 * self.x_pixels_in_detector as u64
    }
}

#[pymethods]
impl DetectorConfig {
    pub fn get_trigger_mode(slf: PyRef<Self>) -> TriggerMode {
        slf.trigger_mode.clone()
    }

    pub fn get_num_frames(slf: PyRef<Self>) -> u64 {
        slf.get_num_images()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[pyclass]
pub struct DImage {
    pub htype: String,

    /// the current series id
    pub series: u64,

    /// frame index, starting at 0
    pub frame: u64,

    /// md5 hash of the image data
    pub hash: String,
}

#[pymethods]
impl DImage {
    #[new]
    fn new(frame: u64, series: u64, hash: &str) -> Self {
        DImage {
            htype: "dimage-1.0".to_string(),
            series,
            frame,
            hash: hash.to_string(),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[pyclass]
pub enum PixelType {
    #[serde(rename = "uint8")]
    Uint8,
    #[serde(rename = "uint16")]
    Uint16,
    #[serde(rename = "uint32")]
    Uint32,
}

#[derive(PartialEq, Eq, Serialize, Deserialize, Debug, Clone)]
#[pyclass]
pub struct DImageD {
    pub htype: String,
    pub shape: Vec<u64>,
    #[serde(rename = "type")]
    pub type_: PixelType,
    pub encoding: String, // [bs<BIT>][[-]lz4][<|>]
}

#[pymethods]
impl DImageD {
    #[new]
    fn new(shape: Vec<u64>, type_: PixelType, encoding: &str) -> Self {
        DImageD {
            htype: "dimage_d-1.0".to_string(),
            shape,
            type_,
            encoding: encoding.to_string(),
        }
    }
}

/// "footer" sent for each frame. all times in nanoseconds
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[pyclass]
pub struct DConfig {
    pub htype: String, // constant dconfig-1.0
    pub start_time: u64,
    pub stop_time: u64,
    pub real_time: u64,
}

#[pymethods]
impl DConfig {
    #[new]
    fn new(start_time: u64, stop_time: u64, real_time: u64) -> Self {
        DConfig {
            htype: "dconfig-1.0".to_string(),
            start_time,
            stop_time,
            real_time,
        }
    }
}

#[derive(PartialEq, Eq, Clone, Serialize, Deserialize, Debug)]
pub struct FrameMeta {
    pub dimage: DImage,
    pub dimaged: DImageD,
    pub dconfig: DConfig,
    pub data_length_bytes: usize,
}

impl FrameMeta {
    /// Get the number of elements in this frame (`prod(shape)`)
    pub fn get_size(&self) -> u64 {
        self.dimaged.shape.iter().product()
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[pyclass]
pub struct DSeriesEnd {
    pub htype: String,
    pub series: u64,
}

#[pymethods]
impl DSeriesEnd {
    #[new]
    fn new(series: u64) -> Self {
        DSeriesEnd {
            htype: "dseries_end-1.0".to_string(),
            series,
        }
    }
}

fn monitor_thread(ctx: Context, endpoint: &str, name: &str) {
    let socket = ctx.socket(zmq::PAIR).unwrap();
    socket.connect(endpoint).unwrap();

    let mut msg: Message = Message::new();

    loop {
        // two parts:
        // first part: "number and value"
        socket.recv(&mut msg, 0).unwrap();

        let event = u16::from_ne_bytes(msg[0..2].try_into().unwrap());
        let socket_event: SocketEvent = SocketEvent::from_raw(event);

        // second part: affected endpoint as string
        socket.recv(&mut msg, 0).unwrap();

        let endpoint = String::from_utf8_lossy(&msg);

        info!("monitoring {name}: {socket_event:?} @ {endpoint}");

        if socket_event == SocketEvent::MONITOR_STOPPED {
            break;
        }
    }
}

pub fn setup_monitor(ctx: Context, name: String, socket: &Socket) {
    // set up monitoring:
    let monitor_uuid = Uuid::new_v4();
    let monitor_endpoint = format!("inproc://monitor-{monitor_uuid}");
    socket
        .monitor(&monitor_endpoint, zmq::SocketEvent::ALL as i32)
        .unwrap();

    std::thread::Builder::new()
        .name(format!("sender-monitor-{monitor_uuid}"))
        .spawn(move || {
            monitor_thread(ctx, &monitor_endpoint, &name);
        })
        .expect("should be able to start monitor thread");
}
