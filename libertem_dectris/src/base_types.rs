#![allow(clippy::borrow_deref_ref)]

use std::ops::Deref;

use common::{frame_stack::FrameMeta, generic_connection::AcquisitionConfig};
use log::info;
use serde::{Deserialize, Serialize};

use pyo3::prelude::*;
use uuid::Uuid;
use zmq::{Context, Message, Socket, SocketEvent};

#[derive(Deserialize, Serialize, PartialEq, Eq, Debug, Clone)]
#[serde(try_from = "String")]
pub struct NonEmptyString(String);

impl TryFrom<String> for NonEmptyString {
    type Error = String;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        if value.is_empty() {
            Err("empty string provided where non-empty was expected".to_owned())
        } else {
            Ok(NonEmptyString(value))
        }
    }
}

impl ToString for NonEmptyString {
    fn to_string(&self) -> String {
        self.0.clone()
    }
}

impl Deref for NonEmptyString {
    type Target = String;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[pyclass]
pub struct DSeriesAndType {
    pub series: u64,
    pub htype: NonEmptyString,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[pyclass]
pub struct DHeader {
    pub htype: NonEmptyString,
    pub header_detail: NonEmptyString,
    pub series: u64,
}

#[pymethods]
impl DHeader {
    #[new]
    fn new(series: u64) -> Self {
        DHeader {
            htype: "dheader-1.0".to_owned().try_into().unwrap(),
            header_detail: "basic".to_owned().try_into().unwrap(),
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

#[derive(Debug)]
#[pyclass]
pub struct DectrisPendingAcquisition {
    detector_config: DetectorConfig,
    series: u64,
}

#[pymethods]
impl DectrisPendingAcquisition {
    pub fn get_series(&self) -> u64 {
        self.series
    }
}

impl DectrisPendingAcquisition {
    pub fn new(detector_config: DetectorConfig, series: u64) -> Self {
        Self {
            detector_config,
            series,
        }
    }

    pub fn get_detector_config(&self) -> DetectorConfig {
        self.detector_config.clone()
    }
}

impl AcquisitionConfig for DectrisPendingAcquisition {
    fn num_frames(&self) -> usize {
        self.detector_config.get_num_images() as usize
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[pyclass]
pub struct DImage {
    pub htype: NonEmptyString,

    /// the current series id
    pub series: u64,

    /// frame index, starting at 0
    pub frame: u64,

    /// md5 hash of the image data
    pub hash: NonEmptyString,
}

#[pymethods]
impl DImage {
    #[new]
    fn new(frame: u64, series: u64, hash: &str) -> Self {
        DImage {
            htype: "dimage-1.0".to_owned().try_into().unwrap(),
            series,
            frame,
            hash: hash.to_owned().try_into().unwrap(),
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
    pub htype: NonEmptyString,
    pub shape: (u64, u64),
    #[serde(rename = "type")]
    pub type_: PixelType,
    pub encoding: NonEmptyString, // [bs<BIT>][[-]lz4][<|>]
}

#[pymethods]
impl DImageD {
    #[new]
    fn new(shape: (u64, u64), type_: PixelType, encoding: &str) -> Self {
        DImageD {
            htype: "dimage_d-1.0".to_string().try_into().unwrap(),
            shape,
            type_,
            encoding: encoding.to_string().try_into().unwrap(),
        }
    }
}

/// "footer" sent for each frame. all times in nanoseconds
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[pyclass]
pub struct DConfig {
    pub htype: NonEmptyString, // constant dconfig-1.0
    pub start_time: u64,
    pub stop_time: u64,
    pub real_time: u64,
}

#[pymethods]
impl DConfig {
    #[new]
    fn new(start_time: u64, stop_time: u64, real_time: u64) -> Self {
        DConfig {
            htype: "dconfig-1.0".to_string().try_into().unwrap(),
            start_time,
            stop_time,
            real_time,
        }
    }
}

#[derive(PartialEq, Eq, Clone, Serialize, Deserialize, Debug)]
pub struct DectrisFrameMeta {
    pub dimage: DImage,
    pub dimaged: DImageD,
    pub dconfig: DConfig,
    pub data_length_bytes: usize,
}

#[derive(Debug)]
pub enum Endianess {
    Little,
    Big,
}

impl Endianess {
    pub fn as_string(&self) -> String {
        match self {
            Endianess::Little => "<".to_owned(),
            Endianess::Big => ">".to_owned(),
        }
    }
}

impl DectrisFrameMeta {
    /// number of pixels in the uncompressed frame (from the shape)
    pub fn get_number_of_pixels(&self) -> usize {
        self.dimaged.shape.0 as usize * self.dimaged.shape.1 as usize
    }

    /// endianess after decompression (little/big)
    pub fn get_endianess(&self) -> Endianess {
        match self.dimaged.encoding.chars().last().unwrap() {
            '>' => Endianess::Big,
            '<' => Endianess::Little,
            _ => {
                panic!("malformed encoding field");
            }
        }
    }
}

impl FrameMeta for DectrisFrameMeta {
    fn get_data_length_bytes(&self) -> usize {
        self.data_length_bytes
    }

    fn get_dtype_string(&self) -> String {
        let endianess = self.get_endianess();
        // TODO: &'static str instead?
        match (endianess, &self.dimaged.type_) {
            (Endianess::Little, PixelType::Uint8) => "uint8".to_owned(),
            (Endianess::Little, PixelType::Uint16) => "<u2".to_owned(),
            (Endianess::Little, PixelType::Uint32) => "<u4".to_owned(),
            (Endianess::Big, PixelType::Uint8) => "uint8".to_owned(),
            (Endianess::Big, PixelType::Uint16) => ">u2".to_owned(),
            (Endianess::Big, PixelType::Uint32) => ">u4".to_owned(),
        }
    }

    fn get_shape(&self) -> (u64, u64) {
        self.dimaged.shape
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[pyclass]
pub struct DSeriesEnd {
    pub htype: NonEmptyString,
    pub series: u64,
}

#[pymethods]
impl DSeriesEnd {
    #[new]
    fn new(series: u64) -> Self {
        DSeriesEnd {
            htype: "dseries_end-1.0".to_string().try_into().unwrap(),
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
