use common::{
    frame_stack::FrameMeta,
    generic_connection::{AcquisitionConfig, DetectorConnectionConfig},
};
use pyo3::{pyclass, pymethods};
use serde::{Deserialize, Serialize};
use serval_client::DetectorConfig;

#[derive(PartialEq, Eq, Clone, Serialize, Deserialize, Debug)]
#[pyclass(eq, eq_int)]
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

    fn get_dtype_string(&self) -> String {
        self.dtype.as_string()
    }

    fn get_shape(&self) -> (u64, u64) {
        (self.width as u64, self.height as u64)
    }
}

// FIXME: can we divide this more cleanly - PendingAcquisition probably
// shouldn't be a `pyclass`? It depends on PyDetectorConfig...

#[derive(Debug)]
#[pyclass(name = "DetectorConfig")]
pub struct PyDetectorConfig {
    config: DetectorConfig,
}

#[pymethods]
impl PyDetectorConfig {
    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }

    fn get_n_triggers(&self) -> u64 {
        self.config.n_triggers
    }
}

impl PyDetectorConfig {
    pub fn new(config: DetectorConfig) -> Self {
        Self { config }
    }
}

#[derive(Debug)]
#[pyclass]
pub struct PendingAcquisition {
    config: DetectorConfig,
    first_frame_meta: ASIMpxFrameMeta,
}

impl PendingAcquisition {
    pub fn new(config: &DetectorConfig, first_frame_meta: &ASIMpxFrameMeta) -> Self {
        Self {
            config: config.clone(),
            first_frame_meta: first_frame_meta.clone(),
        }
    }
}

#[pymethods]
impl PendingAcquisition {
    fn get_detector_config(&self) -> PyDetectorConfig {
        PyDetectorConfig {
            config: self.config.clone(),
        }
    }

    fn get_frame_width(&self) -> u16 {
        self.first_frame_meta.width
    }

    fn get_frame_height(&self) -> u16 {
        self.first_frame_meta.height
    }
}

impl AcquisitionConfig for PendingAcquisition {
    fn num_frames(&self) -> usize {
        self.config.n_triggers as usize
    }
}

#[derive(Debug, Clone)]
pub struct ASIMpxDetectorConnConfig {
    pub data_uri: String,
    pub api_uri: String,

    /// number of frames per frame stack (approx?)
    pub frame_stack_size: usize,

    /// approx. number of bytes per frame
    pub bytes_per_frame: usize,

    num_slots: usize,
    enable_huge_pages: bool,
    shm_handle_path: String,
}

impl ASIMpxDetectorConnConfig {
    pub fn new(
        data_uri: &str,
        api_uri: &str,
        frame_stack_size: usize,
        bytes_per_frame: usize,
        num_slots: usize,
        enable_huge_pages: bool,
        shm_handle_path: &str,
    ) -> Self {
        Self {
            data_uri: data_uri.to_owned(),
            api_uri: api_uri.to_owned(),
            frame_stack_size,
            bytes_per_frame,
            num_slots,
            enable_huge_pages,
            shm_handle_path: shm_handle_path.to_owned(),
        }
    }
}

impl DetectorConnectionConfig for ASIMpxDetectorConnConfig {
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
