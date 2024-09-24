use common::frame_stack::FrameMeta;
use serde::{Deserialize, Serialize};

use crate::config::{K2Mode, Shape};

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub enum K2FrameType {
    IS,
    Summit,
    // FIXME: add 800, 1600 fps frame types?
}

impl K2FrameType {
    pub fn as_camera_mode(&self) -> K2Mode {
        match self {
            K2FrameType::IS => K2Mode::IS,
            K2FrameType::Summit => K2Mode::Summit,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct K2FrameMeta {
    acquisition_id: usize,

    /// the frame id as received
    frame_id: u32,

    /// the frame index in the acquisition
    frame_idx: u32,

    // FIXME: replace with the proper type
    // created_timestamp: Instant,
    frame_type: K2FrameType,
    bytes_per_pixel: usize,

    /// whether or not to crop to the "usable image area"
    crop_to_image_data: bool,
}

impl K2FrameMeta {
    pub fn new(
        acquisition_id: usize,
        frame_id: u32,
        frame_idx: u32,
        frame_type: K2FrameType,
        bytes_per_pixel: usize,
        crop_to_image_data: bool,
    ) -> Self {
        Self {
            acquisition_id,
            frame_id,
            frame_idx,
            frame_type,
            bytes_per_pixel,
            crop_to_image_data,
        }
    }

    pub fn get_crop(&self) -> bool {
        self.crop_to_image_data
    }

    pub fn get_frame_type(&self) -> K2FrameType {
        self.frame_type
    }

    pub fn get_raw_frame_shape(&self) -> Shape {
        self.frame_type.as_camera_mode().get_frame_shape(false)
    }

    pub fn get_effective_frame_shape(&self) -> Shape {
        self.frame_type
            .as_camera_mode()
            .get_frame_shape(self.crop_to_image_data)
    }
}

impl FrameMeta for K2FrameMeta {
    fn get_data_length_bytes(&self) -> usize {
        let shape = self.get_shape();
        (shape.0 * shape.1) as usize * self.bytes_per_pixel
    }

    fn get_dtype_string(&self) -> String {
        "u16".to_owned()
    }

    fn get_shape(&self) -> (u64, u64) {
        let shape = self
            .frame_type
            .as_camera_mode()
            .get_frame_shape(self.crop_to_image_data);
        (shape.height as u64, shape.width as u64)
    }
}
