use std::{any::type_name, fmt::Debug, mem::size_of};

use ipc_test::SharedSlabAllocator;
use ndarray::ArrayViewMut3;
use num::{cast::AsPrimitive, Bounded, Num, NumCast, ToPrimitive};
use zerocopy::{AsBytes, FromBytes};

use crate::frame_stack::{FrameMeta, FrameStackHandle};

pub trait DecoderTargetPixelType:
    AsBytes + FromBytes + NumCast + Copy + Debug + Bounded + Num + 'static
{
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
        T: DecoderTargetPixelType,
        u8: AsPrimitive<T>,
        u16: AsPrimitive<T>;

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
    O: Copy + NumCast + Debug,
    I: Copy + ToPrimitive + Debug,
{
    for (dest, src) in output.iter_mut().zip(input.iter()) {
        *dest = try_cast_primitive(*src)?;
    }

    Ok(())
}

/// Helper function to cast a primitive value or emit a `DecoderError` if it's
/// not safely possible.
#[inline(always)]
pub fn try_cast_primitive<O, I>(input_value: I) -> Result<O, DecoderError>
where
    O: Copy + ToPrimitive + NumCast + Debug,
    I: ToPrimitive + Debug + Copy,
{
    if let Some(value) = NumCast::from(input_value) {
        Ok(value)
    } else {
        Err(DecoderError::FrameDecodeFailed {
            msg: format!(
                "dtype conversion error: {input_value:?} does not fit {0}",
                type_name::<O>()
            ),
        })
    }
}

pub fn decode_ints_be<'a, O, ItemType>(
    input: &'a [u8],
    output: &mut [O],
) -> Result<(), DecoderError>
where
    O: DecoderTargetPixelType,
    ItemType: ToPrimitive + Debug + Copy + num::traits::FromBytes,
    &'a <ItemType as num::traits::FromBytes>::Bytes: std::convert::TryFrom<&'a [u8]>,
    <&'a <ItemType as num::traits::FromBytes>::Bytes as std::convert::TryFrom<&'a [u8]>>::Error:
        Debug,
    <ItemType as num::traits::FromBytes>::Bytes: 'a,
{
    if input.len() % std::mem::size_of::<ItemType>() != 0 {
        return Err(DecoderError::FrameDecodeFailed {
            msg: format!(
                "input length {} is not divisible by item size {}",
                input.len(),
                std::mem::size_of::<ItemType>(),
            ),
        });
    }

    if input.len() / std::mem::size_of::<ItemType>() != output.len() {
        return Err(DecoderError::FrameDecodeFailed {
            msg: format!(
                "input length {} does not match output size {} for type {} item size {}",
                input.len(),
                output.len(),
                type_name::<ItemType>(),
                size_of::<ItemType>(),
            ),
        });
    }

    let chunks = input.chunks_exact(std::mem::size_of::<ItemType>());
    for (in_chunk, out_dest) in chunks.zip(output.iter_mut()) {
        let swapped = ItemType::from_be_bytes(in_chunk.try_into().expect("chunked"));
        *out_dest = try_cast_primitive(swapped)?;
    }

    Ok(())
}

#[cfg(test)]
mod test {
    use std::io::Write;

    use super::{decode_ints_be, try_cast_if_safe};

    #[test]
    fn test_decode_ints_be() {
        let orig = (0..65535).collect::<Vec<u16>>();
        let mut encoded = Vec::<u8>::new();

        for i in orig.iter() {
            encoded.write_all(&i.to_be_bytes()).unwrap();
        }

        // decode u16be from `orig` into different integer and float formats:
        let mut result = vec![0u16; 65535];
        decode_ints_be::<_, u16>(&encoded, &mut result).unwrap();
        for (a, b) in orig.iter().zip(result.iter()) {
            assert_eq!(a, b);
        }

        let mut result = vec![0u32; 65535];
        decode_ints_be::<_, u16>(&encoded, &mut result).unwrap();
        for (a, b) in orig.iter().zip(result.iter()) {
            assert_eq!(*a as u32, *b);
        }

        let mut result = vec![0u64; 65535];
        decode_ints_be::<_, u16>(&encoded, &mut result).unwrap();
        for (a, b) in orig.iter().zip(result.iter()) {
            assert_eq!(*a as u64, *b);
        }

        let mut result = vec![0.0f32; 65535];
        decode_ints_be::<_, u16>(&encoded, &mut result).unwrap();
        for (a, b) in orig.iter().zip(result.iter()) {
            assert_eq!(*a as f32, *b);
        }
    }

    #[test]
    fn test_try_cast_u8() {
        let orig = (0..255).collect::<Vec<u8>>();

        let mut output = vec![0u8; 255];
        try_cast_if_safe(&orig, &mut output).unwrap();
        for (a, b) in orig.iter().zip(output.iter()) {
            assert_eq!(*a, *b);
        }

        let mut output = vec![0u16; 255];
        try_cast_if_safe(&orig, &mut output).unwrap();
        for (a, b) in orig.iter().zip(output.iter()) {
            assert_eq!(*a as u16, *b);
        }

        let mut output = vec![0u32; 255];
        try_cast_if_safe(&orig, &mut output).unwrap();
        for (a, b) in orig.iter().zip(output.iter()) {
            assert_eq!(*a as u32, *b);
        }

        let mut output = vec![0u64; 255];
        try_cast_if_safe(&orig, &mut output).unwrap();
        for (a, b) in orig.iter().zip(output.iter()) {
            assert_eq!(*a as u64, *b);
        }

        let mut output = vec![0f32; 255];
        try_cast_if_safe(&orig, &mut output).unwrap();
        for (a, b) in orig.iter().zip(output.iter()) {
            assert_eq!(*a as f32, *b);
        }

        let mut output = vec![0f64; 255];
        try_cast_if_safe(&orig, &mut output).unwrap();
        for (a, b) in orig.iter().zip(output.iter()) {
            assert_eq!(*a as f64, *b);
        }
    }
}
