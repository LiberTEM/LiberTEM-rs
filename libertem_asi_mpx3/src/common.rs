use common::frame_stack::FrameMeta;
use pyo3::{pyclass, pymethods};
use serde::{Deserialize, Serialize};

#[derive(PartialEq, Eq, Clone, Serialize, Deserialize, Debug)]
#[pyclass]
pub enum DType {
    U8,
    U16,
}

impl DType {
    pub fn from_maxval(maxval: u32) -> Self {
        if maxval < 256 {
            DType::U8
        } else {
            DType::U16
        }
    }

    pub fn num_bytes(&self) -> usize {
        match self {
            DType::U8 => 1,
            DType::U16 => 2,
        }
    }
}

#[pymethods]
impl DType {
    fn as_string(&self) -> String {
        match self {
            Self::U8 => "uint8".to_string(),
            Self::U16 => "uint16".to_string(),
        }
    }
}

#[derive(PartialEq, Eq, Clone, Serialize, Deserialize, Debug)]
pub struct ASIMpxFrameMeta {
    pub sequence: u64,
    pub dtype: DType,
    pub width: u16,
    pub height: u16,

    /// The exact length of the data for this frame
    pub data_length_bytes: usize,

    /// The length of the header in bytes
    pub header_length_bytes: usize,
}

impl ASIMpxFrameMeta {
    /// Get the number of elements in this frame (`prod(shape)`)
    pub fn get_size(&self) -> usize {
        self.width as usize * self.height as usize
    }
}

impl FrameMeta for ASIMpxFrameMeta {
    fn get_data_length_bytes(&self) -> usize {
        self.data_length_bytes
    }
}
