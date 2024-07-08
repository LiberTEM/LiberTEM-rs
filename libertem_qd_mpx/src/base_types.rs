use std::collections::HashMap;

use common::{
    frame_stack::FrameMeta,
    generic_connection::{AcquisitionConfig, DetectorConnectionConfig},
};
use pyo3::pyclass;
use serde::{Deserialize, Serialize};

#[derive(Debug, thiserror::Error)]
pub enum FrameMetaParseError {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QdFrameMeta {}

impl QdFrameMeta {
    pub fn parse_bytes(input: &[u8]) -> Result<Self, FrameMetaParseError> {
        todo!()
    }
}

impl FrameMeta for QdFrameMeta {
    fn get_data_length_bytes(&self) -> usize {
        todo!()
    }

    fn get_dtype_string(&self) -> String {
        todo!()
    }

    fn get_shape(&self) -> (u64, u64) {
        todo!()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AcqHeaderParseError {}

#[derive(Debug)]
#[pyclass]
pub struct QdAcquisitionHeader {
    frames_in_acquisition: usize,
    raw_kv: HashMap<String, String>,
}

impl AcquisitionConfig for QdAcquisitionHeader {
    fn num_frames(&self) -> usize {
        self.frames_in_acquisition
    }
}

impl QdAcquisitionHeader {
    pub fn get_raw_kv(&self) -> &HashMap<String, String> {
        &self.raw_kv
    }

    pub fn parse_bytes(input: &[u8]) -> Result<Self, AcqHeaderParseError> {
        todo!();
    }
}

#[derive(Clone, Debug)]
pub struct QdDetectorConnConfig {
    pub data_host: String,
    pub data_port: usize,

    /// number of frames per frame stack; approximated because of compression
    pub frame_stack_size: usize,

    /// approx. number of bytes per frame, used for sizing frame stacks together
    /// with `frame_stack_size`
    pub bytes_per_frame: usize,

    num_slots: usize,
    enable_huge_pages: bool,
    shm_handle_path: String,
}

impl QdDetectorConnConfig {
    pub fn new(
        data_host: &str,
        data_port: usize,
        frame_stack_size: usize,
        bytes_per_frame: usize,
        num_slots: usize,
        enable_huge_pages: bool,
        shm_handle_path: &str,
    ) -> Self {
        Self {
            data_host: data_host.to_owned(),
            data_port,
            frame_stack_size,
            bytes_per_frame,
            num_slots,
            enable_huge_pages,
            shm_handle_path: shm_handle_path.to_owned(),
        }
    }
}

impl DetectorConnectionConfig for QdDetectorConnConfig {
    fn get_shm_num_slots(&self) -> usize {
        self.num_slots
    }

    fn get_shm_slot_size(&self) -> usize {
        self.frame_stack_size * self.bytes_per_frame
    }

    fn get_shm_enable_huge_pages(&self) -> bool {
        self.enable_huge_pages
    }

    fn get_shm_handle_path(&self) -> String {
        self.shm_handle_path.clone()
    }
}
