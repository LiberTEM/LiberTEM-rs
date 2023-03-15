#![allow(clippy::borrow_deref_ref)]

use std::fs;

use log::info;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

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

pub struct DumpRecordFile {
    filename: String,
    mmap: memmap2::Mmap,
}

impl Clone for DumpRecordFile {
    fn clone(&self) -> Self {
        DumpRecordFile::new(&self.filename)
    }
}

impl DumpRecordFile {
    pub fn new(filename: &str) -> Self {
        let file = fs::File::open(filename).expect("file should exist and be readable");
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file) }.unwrap();
        DumpRecordFile {
            mmap,
            filename: filename.to_string(),
        }
    }

    /// read and decode a message from the "zeromq dump file" format,
    /// which is just le-i64 size + raw bytes messages
    ///
    /// in case the message is not a json message, returns None,
    /// otherwise it returns the parsed message as serde_json::Value
    ///
    /// Note: don't use in performance-critical code path; it's only
    /// meant for initialization and reading the first few messages
    pub fn read_json(&self, offset: usize) -> (Option<serde_json::Value>, usize) {
        let (msg, size) = self.read_msg_raw(offset);
        let value: Result<serde_json::Value, _> = serde_json::from_slice(msg);

        match value {
            Err(_) => (None, size),
            Ok(json) => (Some(json), size),
        }
    }

    pub fn read_size(&self, offset: usize) -> usize {
        i64::from_le_bytes(self.mmap[offset..offset + 8].try_into().unwrap()) as usize
    }

    pub fn read_msg_raw(&self, offset: usize) -> (&[u8], usize) {
        let size = i64::from_le_bytes(self.mmap[offset..offset + 8].try_into().unwrap()) as usize;
        (&self.mmap[offset + 8..offset + 8 + size], size)
    }

    /// find the offset of the first header of the given htype
    pub fn offset_for_first_header(&self, expected_htype: &str) -> Option<usize> {
        let mut current_offset = 0;
        while current_offset < self.mmap.len() {
            let (value, size) = self.read_json(current_offset);

            if let Some(val) = value {
                let htype = val
                    .as_object()
                    .expect("all json messages should be objects")
                    .get("htype");
                if let Some(htype_str) = htype {
                    if htype_str == expected_htype {
                        return Some(current_offset);
                    }
                }
            }

            current_offset += size + 8;
        }
        None
    }

    pub fn get_cursor(&self) -> RecordCursor {
        RecordCursor::new(self)
    }

    fn get_size(&self) -> usize {
        self.mmap.len()
    }
}

pub struct CursorPos {
    pub current_offset: usize,
    pub current_msg_index: usize,
}

pub struct RecordCursor {
    file: DumpRecordFile,
    current_offset: usize,

    /// the message index of the message that will be returned next by `read_raw_msg`
    current_msg_index: usize,
}

impl RecordCursor {
    pub fn new(file: &DumpRecordFile) -> Self {
        RecordCursor {
            file: file.clone(),
            current_offset: 0,
            current_msg_index: 0,
        }
    }

    pub fn set_pos(&mut self, pos: CursorPos) {
        self.current_msg_index = pos.current_msg_index;
        self.current_offset = pos.current_offset;
    }

    pub fn get_pos(&self) -> CursorPos {
        CursorPos {
            current_offset: self.current_offset,
            current_msg_index: self.current_msg_index,
        }
    }

    /// seek such that `index` is the next message that will be read
    pub fn seek_to_msg_idx(&mut self, index: usize) {
        self.current_offset = 0;
        self.current_msg_index = 0;

        while self.current_msg_index < index {
            self.read_raw_msg();
        }
    }

    pub fn seek_to_first_header_of_type(&mut self, header_type: &str) {
        self.current_offset = self
            .file
            .offset_for_first_header(header_type)
            .expect("header should exist");
    }

    pub fn read_raw_msg(&mut self) -> &[u8] {
        let (msg, size) = self.file.read_msg_raw(self.current_offset);
        self.current_offset += size + 8;
        self.current_msg_index += 1;
        msg
    }

    pub fn read_and_deserialize<T>(&mut self) -> Result<T, serde_json::error::Error>
    where
        T: DeserializeOwned,
    {
        let msg = self.read_raw_msg();
        serde_json::from_slice::<T>(msg)
    }

    pub fn peek_size(&self) -> usize {
        self.file.read_size(self.current_offset)
    }

    pub fn is_at_end(&self) -> bool {
        self.current_offset == self.file.get_size()
    }

    pub fn get_msg_idx(&self) -> usize {
        self.current_msg_index
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
