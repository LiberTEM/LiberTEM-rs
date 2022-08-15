use std::fs;

use serde::{de::DeserializeOwned, Deserialize, Serialize};

use pyo3::prelude::*;
use serde_json::json;
use zmq::{Context, Socket, SocketType::PUSH};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DHeader {
    pub htype: String,
    pub header_detail: String,
    pub series: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
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

#[derive(Serialize, Deserialize, Debug, Clone)]
#[pyclass]
pub struct DetectorConfig {
    pub ntrigger: u64,
    pub nimages: u64,
    trigger_mode: TriggerMode,
}

impl DetectorConfig {
    pub fn get_num_images(&self) -> u64 {
        match self.trigger_mode {
            TriggerMode::EXTE | TriggerMode::INTE => self.ntrigger,
            TriggerMode::EXTS | TriggerMode::INTS => self.nimages * self.ntrigger,
        }
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
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct DImage {
    pub htype: String,

    /// the current series id
    pub series: u64,

    /// frame index, starting at 0
    pub frame: u64,

    /// md5 hash of the image data
    pub hash: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[pyclass]
#[derive(PartialEq)]
pub enum PixelType {
    #[serde(rename = "uint8")]
    Uint8,
    #[serde(rename = "uint16")]
    Uint16,
    #[serde(rename = "uint32")]
    Uint32,
}

#[derive(PartialEq, Serialize, Deserialize, Debug, Clone)]
pub struct DImageD {
    pub htype: String,
    pub shape: Vec<u64>,
    #[serde(rename = "type")]
    pub type_: PixelType,
    pub encoding: String, // [bs<BIT>][[-]lz4][<|>]
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct DConfig {
    pub htype: String,
    pub start_time: u64,
    pub stop_time: u64,
    pub real_time: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DSeriesEnd {
    pub htype: String,
    pub series: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[pyclass]
#[derive(PartialEq)]
pub struct FrameData {
    pub dimage: DImage,
    pub dimaged: DImageD,

    /// the raw, undecoded data for this frame, probably compressed
    pub image_data: Vec<u8>,

    pub dconfig: DConfig,
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
        let size = i64::from_le_bytes(self.mmap[offset..offset + 8].try_into().unwrap()) as usize;
        size
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

            match value {
                Some(val) => {
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
                None => {}
            }

            current_offset += size + 8;
        }
        return None;
    }

    pub fn get_cursor(&self) -> RecordCursor {
        return RecordCursor::new(self);
    }

    fn get_size(&self) -> usize {
        return self.mmap.len();
    }
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
        return msg;
    }

    pub fn read_and_deserialize<'de, T>(&mut self) -> Result<T, serde_json::error::Error>
    where
        T: DeserializeOwned,
    {
        let msg = self.read_raw_msg().clone();
        return serde_json::from_slice::<T>(&msg);
    }

    pub fn peek_size(&self) -> usize {
        return self.file.read_size(self.current_offset);
    }

    pub fn is_at_end(&self) -> bool {
        return self.current_offset == self.file.get_size();
    }

    pub fn get_msg_idx(&self) -> usize {
        self.current_msg_index
    }
}

#[derive(Debug)]
pub enum SendError {
    Timeout,
    Other,
}

impl From<zmq::Error> for SendError {
    fn from(e: zmq::Error) -> Self {
        match e {
            zmq::Error::EAGAIN => SendError::Timeout,
            _ => SendError::Other,
        }
    }
}

pub struct FrameSender {
    socket: Socket,
    cursor: RecordCursor,
    detector_config: DetectorConfig,
    series: u64,
    nimages: u64,
}

impl FrameSender {
    pub fn new(uri: &str, filename: &str) -> Self {
        let ctx = Context::new();
        let socket = ctx
            .socket(PUSH)
            .expect("context should be able to create a socket");
        socket
            .bind(uri)
            .expect("should be possible to bind the zmq socket");
        socket
            .set_sndhwm(4 * 256)
            .expect("should be possible to set sndhwn");

        let file = DumpRecordFile::new(filename);

        // temporary cursor to deserialize headers:
        let mut cursor = file.get_cursor();

        cursor.seek_to_first_header_of_type("dheader-1.0");
        let dheader_raw = cursor.read_raw_msg();
        let dheader: DHeader = serde_json::from_slice(&dheader_raw)
            .expect("json should match our serialization schema");

        println!("{dheader:?}");

        let detector_config: DetectorConfig = cursor.read_and_deserialize().unwrap();
        println!("{detector_config:?}");

        let nimages = detector_config.get_num_images();
        let series = dheader.series;

        FrameSender {
            socket,
            cursor: file.get_cursor(),
            series,
            nimages,
            detector_config,
        }
    }

    pub fn get_detector_config(&self) -> &DetectorConfig {
        &self.detector_config
    }

    pub fn send_frame(&mut self) -> Result<(), SendError> {
        let socket = &self.socket;
        let cursor = &mut self.cursor;

        // We can't just simply blockingly send here, as that will
        // block Ctrl-C when used in Python (the SIGINT handler
        // only sets a flag, it can't interrupt native code)
        // what we can do instead: in "real life", when the high watermark
        // is exceeded, frames are dropped (i.e. "catastrophal" results)
        // So I think it is warranted to go into an error state if the
        // consumer can't keep up.

        // FIXME: We may want to add a "replay speed" later to limit the message
        // rate to something sensible.

        // milliseconds
        socket.set_sndtimeo(1000)?;

        let m = cursor.read_raw_msg();
        socket.send(m, zmq::SNDMORE)?;

        let m = cursor.read_raw_msg();
        socket.send(m, zmq::SNDMORE)?;
        let m = cursor.read_raw_msg();
        socket.send(m, zmq::SNDMORE)?;

        let m = cursor.read_raw_msg();
        socket.send(m, 0)?;

        // back to infinity for the other messages
        // FIXME: might want to have a global timeout later
        // to not have hangs from the Python side in any circumstance
        socket.set_sndtimeo(-1)?;

        Ok(())
    }

    fn send_msg_at_cursor(&mut self) {
        let socket = &self.socket;
        let cursor = &mut self.cursor;

        let m = cursor.read_raw_msg();
        socket.send(m, 0).unwrap();
    }

    pub fn send_headers(&mut self) {
        let cursor = &mut self.cursor;
        cursor.seek_to_first_header_of_type("dheader-1.0");

        // dheader
        self.send_msg_at_cursor();

        // detector config
        self.send_msg_at_cursor();
    }

    pub fn send_frames(&mut self) {
        for _ in 0..self.nimages {
            self.send_frame().expect("send_frame should not time out");
        }
    }

    pub fn send_footer(&mut self) {
        // for simplicity, always "emulate" the footer message
        let footer_json = json!({
            "htype": "dseries_end-1.0",
            "series": self.series,
        });
        self.socket.send(&footer_json.to_string(), 0).unwrap();
    }

    pub fn get_num_frames(&self) -> u64 {
        self.nimages
    }
}
