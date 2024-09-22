use common::generic_connection::{AcquisitionConfig, DetectorConnectionConfig};
use pyo3::pyclass;

#[pyclass]
#[derive(Debug, Clone)]
pub struct K2AcquisitionConfig {
    num_frames: usize,
}

impl AcquisitionConfig for K2AcquisitionConfig {
    fn num_frames(&self) -> usize {
        self.num_frames
    }
}

#[derive(Debug, Clone, Copy)]
#[pyclass]
pub enum K2Mode {
    IS,
    Summit,
}

impl K2Mode {
    pub fn get_bytes_per_frame(&self) -> usize {
        let shape = self.get_frame_shape();
        (shape.0 * shape.1) as usize * self.get_bytes_per_pixel()
    }

    pub fn get_bytes_per_pixel(&self) -> usize {
        2
    }

    pub fn get_frame_shape(&self) -> (u64, u64) {
        match self {
            K2Mode::IS => (1860, 2048),
            K2Mode::Summit => (3840, 4096),
        }
    }
}

#[derive(Debug, Clone)]
pub struct K2DetectorConnectionConfig {
    mode: K2Mode,

    /// IP address of the local interface that receives data from the top detector
    local_addr_top: String,

    /// IP address of the local interface that receives data from the bottom detector
    local_addr_bottom: String,

    /// Number of frames per frame stack
    frame_stack_size: usize,

    num_slots: usize,
    enable_huge_pages: bool,
    shm_handle_path: String,
}

impl K2DetectorConnectionConfig {
    pub fn new(
        mode: K2Mode,
        local_addr_top: String,
        local_addr_bottom: String,
        num_slots: usize,
        enable_huge_pages: bool,
        shm_handle_path: String,
        frame_stack_size: usize,
    ) -> Self {
        Self {
            mode,
            local_addr_top,
            local_addr_bottom,
            num_slots,
            enable_huge_pages,
            shm_handle_path,
            frame_stack_size,
        }
    }
}

impl DetectorConnectionConfig for K2DetectorConnectionConfig {
    fn get_shm_num_slots(&self) -> usize {
        self.num_slots
    }

    fn get_shm_slot_size(&self) -> usize {
        self.frame_stack_size * self.mode.get_bytes_per_frame()
    }

    fn get_shm_enable_huge_pages(&self) -> bool {
        self.enable_huge_pages
    }

    fn get_shm_handle_path(&self) -> String {
        self.shm_handle_path.clone()
    }
}
