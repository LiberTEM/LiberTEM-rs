use common::frame_stack::FrameMeta;
use serde::{Deserialize, Serialize};

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub enum K2FrameType {
    IS,
    Summit,
    // FIXME: add 800, 1600 fps frame types?
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct K2FrameMeta {
    acquisition_id: usize,
    frame_id: u32,
    // FIXME: replace with the proper type
    // created_timestamp: Instant,
    frame_type: K2FrameType,
    bytes_per_pixel: usize,
}

impl K2FrameMeta {
    pub fn new(
        acquisition_id: usize,
        frame_id: u32,
        frame_type: K2FrameType,
        bytes_per_pixel: usize,
    ) -> Self {
        Self {
            acquisition_id,
            frame_id,
            frame_type,
            bytes_per_pixel,
        }
    }

    pub fn get_frame_type(&self) -> K2FrameType {
        self.frame_type
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
        match self.frame_type {
            K2FrameType::IS => (1860, 2048),
            K2FrameType::Summit => (3840, 4096),
        }
    }
}
