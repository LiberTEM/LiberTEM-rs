use ipc_test::SharedSlabAllocator;
use ndarray::ArrayViewMut3;
use num::NumCast;
use zerocopy::{AsBytes, FromBytes};

use crate::frame_stack::{FrameMeta, FrameStackHandle};

pub trait Decoder: Default {
    type FrameMeta: FrameMeta;

    /// Decode a range designated by `start_idx` and `end_idx` of `input` in
    /// `shm` into `output`, converting the data to `T` if safely possible,
    /// returning an error otherwise.
    ///
    /// Note that the end index is exclusive, like the range
    /// `start_idx..end_idx`, and these indices are independent of the indices
    /// into `output`. For example, `decode(shm, input, output, 1, 2)` will decode
    /// the second frame from the frame stack into `output[0, :, :]` (using numpy
    /// slicing notation here)
    fn decode<T>(
        &self,
        shm: &SharedSlabAllocator,
        input: &FrameStackHandle<Self::FrameMeta>,
        output: &mut ArrayViewMut3<'_, T>,
        start_idx: usize,
        end_idx: usize,
    ) -> Result<(), DecoderError>
    where
        T: 'static + AsBytes + FromBytes + Copy + NumCast;

    fn zero_copy_available(
        &self,
        handle: &FrameStackHandle<Self::FrameMeta>,
    ) -> Result<bool, DecoderError>;
}

#[derive(Debug, thiserror::Error)]
pub enum DecoderError {
    #[error("decoding of frame failed: {msg}")]
    FrameDecodeFailed { msg: String },
}
