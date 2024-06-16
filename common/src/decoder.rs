use ndarray::ArrayViewMut3;

use crate::frame_stack::{FrameMeta, FrameStackHandle};

pub trait Decoder: Default {
    fn decode<M, T>(
        &self,
        input: &FrameStackHandle<M>,
        dest: ArrayViewMut3<'_, T>,
        start_idx: usize,
        end_idx: usize,
    ) -> Result<(), DecoderError>
    where
        M: FrameMeta;
}

#[derive(Debug, thiserror::Error)]
pub enum DecoderError {}
