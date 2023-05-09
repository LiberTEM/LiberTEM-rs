use std::{fs, sync::Arc};

use serde::de::DeserializeOwned;

pub struct DumpRecordFile {
    data_source: Arc<dyn AsRef<[u8]> + Send + Sync>,
}

impl Clone for DumpRecordFile {
    fn clone(&self) -> Self {
        DumpRecordFile::from_raw_data(Arc::clone(&self.data_source))
    }
}

impl DumpRecordFile {
    pub fn from_file(filename: &str) -> Self {
        let file = fs::File::open(filename).expect("file should exist and be readable");
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file) }.unwrap();
        DumpRecordFile {
            data_source: Arc::new(mmap),
            // filename: filename.to_string(),
        }
    }

    pub fn from_raw_data(data_source: Arc<dyn AsRef<[u8]> + Send + Sync>) -> Self {
        Self { data_source }
    }

    pub fn get_data_slice(&self) -> &[u8] {
        (*self.data_source).as_ref()
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

    pub fn read_msg_raw(&self, offset: usize) -> (&[u8], usize) {
        let data = self.get_data_slice();
        let size = i64::from_le_bytes(data[offset..offset + 8].try_into().unwrap()) as usize;
        (&data[offset + 8..offset + 8 + size], size)
    }

    /// find the offset of the first header of the given htype
    pub fn offset_for_first_header(&self, expected_htype: &str) -> Option<usize> {
        let data = self.get_data_slice();
        let mut current_offset = 0;
        while current_offset < data.len() {
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
        let data = self.get_data_slice();
        data.len()
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

    pub fn is_at_end(&self) -> bool {
        self.current_offset == self.file.get_size()
    }

    pub fn get_msg_idx(&self) -> usize {
        self.current_msg_index
    }
}
