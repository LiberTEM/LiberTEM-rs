use std::{any::type_name, fmt::Debug};

use ipc_test::SharedSlabAllocator;
use ndarray::ArrayViewMut3;
use num::{NumCast, ToPrimitive};
use zerocopy::{AsBytes, FromBytes};

use crate::frame_stack::{FrameMeta, FrameStackHandle};

pub trait DecoderTargetPixelType: AsBytes + FromBytes + NumCast + Copy + Debug + 'static {
    const SIZE_OF: usize = std::mem::size_of::<Self>();
}

impl DecoderTargetPixelType for u8 {}
impl DecoderTargetPixelType for u16 {}
impl DecoderTargetPixelType for u32 {}
impl DecoderTargetPixelType for u64 {}

impl DecoderTargetPixelType for i8 {}
impl DecoderTargetPixelType for i16 {}
impl DecoderTargetPixelType for i32 {}
impl DecoderTargetPixelType for i64 {}

impl DecoderTargetPixelType for f32 {}
impl DecoderTargetPixelType for f64 {}

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
        T: DecoderTargetPixelType;

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

pub fn try_cast_if_safe<I, O>(input: &[I], output: &mut [O]) -> Result<(), DecoderError>
where
    O: Copy + NumCast,
    I: Copy + ToPrimitive + Debug,
{
    for (dest, src) in output.iter_mut().zip(input.iter()) {
        let converted = NumCast::from(*src);
        if let Some(value) = converted {
            *dest = value;
        } else {
            return Err(DecoderError::FrameDecodeFailed {
                msg: format!(
                    "dtype conversion error: {src:?} does not fit {0}",
                    type_name::<O>()
                ),
            });
        }
    }

    Ok(())
}

pub fn decode_ints_be<O, FromT>(input: &[u8], output: &mut [O]) -> Result<(), DecoderError>
where
    O: DecoderTargetPixelType,
    FromT: FromBytes + ToPrimitive + Debug + Copy,
{
    if input.len() % O::SIZE_OF != 0 {
        return Err(DecoderError::FrameDecodeFailed {
            msg: format!(
                "input length {} is not divisible by {}",
                input.len(),
                O::SIZE_OF,
            ),
        });
    }

    let chunks = input.chunks_exact(O::SIZE_OF);
    for (in_chunk, out_dest) in chunks.zip(output.iter_mut()) {
        let swapped =
            FromT::read_from_prefix(in_chunk).ok_or_else(|| DecoderError::FrameDecodeFailed {
                msg: format!(
                    "dtype conversion error: could not byteswap {0}",
                    type_name::<FromT>()
                ),
            })?;

        *out_dest = if let Some(value) = NumCast::from(swapped) {
            value
        } else {
            return Err(DecoderError::FrameDecodeFailed {
                msg: format!(
                    "dtype conversion error: {swapped:?} does not fit {0}",
                    type_name::<O>()
                ),
            });
        }
    }

    Ok(())
}
