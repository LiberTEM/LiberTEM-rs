use common::{
    background_thread::{ConcreteAcquisitionSize, PyAcquisitionSize},
    generic_connection::{AcquisitionConfig, DetectorConnectionConfig},
};
use pyo3::{pyclass, pymethods};
use serde::{Deserialize, Serialize};

use crate::frame_meta::K2FrameType;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Shape {
    pub width: usize,
    pub height: usize,
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct K2AcquisitionConfig {
    acquisition_size: ConcreteAcquisitionSize,
    effective_frame_shape: Shape,
}

impl K2AcquisitionConfig {
    pub fn new(acquisition_size: ConcreteAcquisitionSize, effective_frame_shape: Shape) -> Self {
        Self {
            acquisition_size,
            effective_frame_shape,
        }
    }
}

impl AcquisitionConfig for K2AcquisitionConfig {
    fn acquisition_size(&self) -> ConcreteAcquisitionSize {
        self.acquisition_size
    }
}

#[pymethods]
impl K2AcquisitionConfig {
    fn frame_shape(&self) -> (usize, usize) {
        (
            self.effective_frame_shape.height,
            self.effective_frame_shape.width,
        )
    }

    fn acquisition_size(&self) -> PyAcquisitionSize {
        PyAcquisitionSize::from_acquisition_size(self.acquisition_size.into())
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
        let shape = self.get_frame_shape(false);
        shape.width * shape.height * self.get_bytes_per_pixel()
    }

    pub fn get_bytes_per_pixel(&self) -> usize {
        2
    }

    /// There is some "non-image data" in the data stream we receive, which we
    /// may want to crop off. See also `crate::decoder::try_cast_with_crop`.
    pub fn get_frame_shape(&self, crop: bool) -> Shape {
        match (self, crop) {
            (K2Mode::IS, true) => Shape {
                height: 1920,
                width: 2048 - 128,
            },
            (K2Mode::IS, false) => Shape {
                height: 1920,
                width: 2048,
            },
            (K2Mode::Summit, true) => Shape {
                height: 3840,
                width: 4096 - 256,
            },
            (K2Mode::Summit, false) => Shape {
                height: 3840,
                width: 4096,
            },
        }
    }

    pub fn get_frame_type(&self) -> K2FrameType {
        match &self {
            K2Mode::IS => K2FrameType::IS,
            K2Mode::Summit => K2FrameType::Summit,
        }
    }
}

#[derive(Debug, Clone)]
pub struct K2DetectorConnectionConfig {
    pub mode: K2Mode,

    /// IP address of the local interface that receives data from the top detector
    pub local_addr_top: String,

    /// IP address of the local interface that receives data from the bottom detector
    pub local_addr_bottom: String,

    /// Run frame assembly in a prioritized thread
    pub assembly_realtime: bool,

    /// Run UDP receivers in prioritized threads
    pub recv_realtime: bool,

    /// Crop K2CamClient output to actually usable image data
    pub crop_to_image_data: bool,

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
        recv_realtime: bool,
        assembly_realtime: bool,
        crop_to_image_data: bool,
    ) -> Self {
        Self {
            mode,
            local_addr_top,
            local_addr_bottom,
            num_slots,
            enable_huge_pages,
            shm_handle_path,
            frame_stack_size,
            recv_realtime,
            assembly_realtime,
            crop_to_image_data,
        }
    }

    pub fn effective_shape(&self) -> Shape {
        self.mode.get_frame_shape(self.crop_to_image_data)
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
