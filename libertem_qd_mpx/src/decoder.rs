use std::{any::type_name, fmt::Debug};

use common::decoder::{
    decode_ints_be, try_cast_if_safe, Decoder, DecoderError, DecoderTargetPixelType,
};
use num::{NumCast, ToPrimitive};
use numpy::ndarray::s;
use zerocopy::FromBytes;

use crate::base_types::{DType, QdFrameMeta};

#[derive(Debug, Default)]
pub struct QdDecoder {}

impl Decoder for QdDecoder {
    type FrameMeta = QdFrameMeta;

    fn decode<T>(
        &self,
        shm: &ipc_test::SharedSlabAllocator,
        input: &common::frame_stack::FrameStackHandle<Self::FrameMeta>,
        output: &mut numpy::ndarray::ArrayViewMut3<'_, T>,
        start_idx: usize,
        end_idx: usize,
    ) -> Result<(), DecoderError>
    where
        T: DecoderTargetPixelType,
    {
        // Data comes in different pixel formats:
        //
        // 1) Big endian unsigned integer (u8, u16, u32)
        // 2) "scrambled" raw formats
        // 2.1) r1: raw 1bit per pixel packed integers - a single u64 contains
        //      data for 64 pixels
        // 2.2) r6: raw 6bit per pixel integers - pixels need to be re-ordered
        //      in groups of 8
        // 2.3) r12: raw 12bit per pixel integers - pixels need to be re-ordered
        //      in groups of 4; big endian
        // 2.4) r24: sent as "two consecutive frames" on the network (not fully
        //      implemented yet!)
        //
        // Then, there is the other dimension of layout: in case multiple
        // sensors are put together, what is their geometry? We have these
        // layout possibilities:
        //
        // 1x1
        // 2x2
        // Nx1
        // 2x2G
        // Nx1G
        //
        // The G suffix means that two "gap" pixels are inserted between the sensors,
        //
        // For now, we support 1x1 and 2x2 layouts (including the G suffix),
        // others TBD.
        //
        // If the resulting array looks like this:
        //
        // _________
        // | 1 | 2 |
        // ---------
        // | 3 | 4 |
        // ---------
        //
        // It maps to the input data like this:
        //
        // [4 | 3 | 2 | 1]
        //
        // (note that quadrants 3 and 4 are also flipped in x and y direction in the
        // resulting array, compared to the original data)
        //

        input.with_slot(shm, |slot| {
            for ((frame_meta, out_idx), in_idx) in
                input.get_meta().iter().zip(0..).zip(start_idx..end_idx)
            {
                let mut out_view = output.slice_mut(s![out_idx, .., ..]);
                let out_slice =
                    out_view
                        .as_slice_mut()
                        .ok_or_else(|| DecoderError::FrameDecodeFailed {
                            msg: "out slice not C-order contiguous".to_owned(),
                        })?;
                let raw_input_data = input.get_slice_for_frame(in_idx, slot);

                match &frame_meta.layout {
                    crate::base_types::Layout::L1x1 => {
                        self.decode_frame_single(frame_meta, raw_input_data, out_slice)?
                    }
                    crate::base_types::Layout::L2x2 => {
                        self.decode_frame_quad(frame_meta, raw_input_data, out_slice)?
                    }
                    crate::base_types::Layout::L2x2G => {
                        self.decode_frame_quad(frame_meta, raw_input_data, out_slice)?
                    }
                    layout @ (crate::base_types::Layout::LNx1
                    | crate::base_types::Layout::LNx1G) => {
                        return Err(DecoderError::FrameDecodeFailed {
                            msg: format!("unsupported layout: {layout:?}"),
                        });
                    }
                }
            }
            Ok(())
        })
    }

    fn zero_copy_available(
        &self,
        _handle: &common::frame_stack::FrameStackHandle<Self::FrameMeta>,
    ) -> Result<bool, DecoderError> {
        // FIXME: add zero copy support, only really important for the u8 case
        // (as the other integer formats are big endian, so on most system they
        // need to be converted)
        Ok(false)
    }
}

impl QdDecoder {
    fn decode_frame_quad<O>(
        &self,
        frame_meta: &QdFrameMeta,
        input: &[u8],
        output: &mut [O],
    ) -> Result<(), DecoderError>
    where
        O: DecoderTargetPixelType,
    {
        match frame_meta.dtype {
            // this one could be zero-copy:
            DType::U01 | DType::U08 => try_cast_if_safe(input, output),

            // these need byte swaps:
            DType::U16 => decode_ints_be::<_, u16>(input, output),
            DType::U32 => decode_ints_be::<_, u32>(input, output),
            DType::U64 => decode_ints_be::<_, u64>(input, output),

            // this is any of the raw formats; dispatch on the "original counter depth" value:
            DType::R64 => {
                todo!();
                if let Some(mq1a) = &frame_meta.mq1a {
                    match mq1a.counter_depth {
                        1 => decode_r1(input, output),
                        6 => decode_r6(input, output),
                        12 => decode_r12(input, output),
                        // 24 => decode_r24(input, output),
                        _ => Err(DecoderError::FrameDecodeFailed {
                            msg: format!("unsupported counter depth: {}", mq1a.counter_depth),
                        }),
                    }
                } else {
                    Err(DecoderError::FrameDecodeFailed { msg: "in raw mode, but no M1QA header available - counter depth not available!".to_owned() })
                }
            }
        }
    }

    fn decode_frame_single<O>(
        &self,
        frame_meta: &QdFrameMeta,
        input: &[u8],
        output: &mut [O],
    ) -> Result<(), DecoderError>
    where
        O: DecoderTargetPixelType,
    {
        match frame_meta.dtype {
            // this one could be zero-copy:
            DType::U01 | DType::U08 => try_cast_if_safe(input, output),

            // these need byte swaps:
            DType::U16 => decode_ints_be::<_, u16>(input, output),
            DType::U32 => decode_ints_be::<_, u32>(input, output),
            DType::U64 => decode_ints_be::<_, u64>(input, output),

            // this is any of the raw formats; dispatch on the "original counter depth" value:
            DType::R64 => {
                if let Some(mq1a) = &frame_meta.mq1a {
                    match mq1a.counter_depth {
                        1 => decode_r1(input, output),
                        6 => decode_r6(input, output),
                        12 => decode_r12(input, output),
                        // 24 => decode_r24(input, output),
                        _ => Err(DecoderError::FrameDecodeFailed {
                            msg: format!("unsupported counter depth: {}", mq1a.counter_depth),
                        }),
                    }
                } else {
                    Err(DecoderError::FrameDecodeFailed { msg: "in raw mode, but no M1QA header available - counter depth not available!".to_owned() })
                }
            }
        }
    }
}

/// Decode from raw 1bit format to O. input length must be divisible by 8 and
/// output must be 8 times larger than input.
fn decode_r1<O>(input: &[u8], output: &mut [O]) -> Result<(), DecoderError>
where
    O: Copy + ToPrimitive + NumCast,
{
    if input.len() % 8 != 0 {
        return Err(DecoderError::FrameDecodeFailed {
            msg: format!("input length {} is not divisible by 8", input.len()),
        });
    }

    if output.len() * 8 < input.len() {
        return Err(DecoderError::FrameDecodeFailed {
            msg: format!(
                "output length {} should be 8 times as large as input length {}",
                output.len(),
                input.len()
            ),
        });
    }

    let chunks = input.chunks_exact(8);
    for (in_chunk, out_chunk) in chunks.zip(output.chunks_exact_mut(64)) {
        let value = u64::from_be_bytes(in_chunk.try_into().expect("chunked by 8 bytes"));
        for (i, out_dest) in out_chunk.iter_mut().enumerate() {
            let out_value = (value >> i) & 0x1;
            *out_dest = if let Some(value) = NumCast::from(out_value) {
                value
            } else {
                return Err(DecoderError::FrameDecodeFailed {
                    msg: format!(
                        "dtype conversion error: {out_value:?} does not fit {0}",
                        type_name::<O>()
                    ),
                });
            }
        }
    }

    Ok(())
}

/// Decode from raw 6bit format to O. input length must be divisible by 8
fn decode_r6<O>(input: &[u8], output: &mut [O]) -> Result<(), DecoderError>
where
    O: Copy + ToPrimitive + NumCast,
{
    if input.len() % 8 != 0 {
        return Err(DecoderError::FrameDecodeFailed {
            msg: format!("input length {} is not divisible by 8", input.len()),
        });
    }

    if output.len() != input.len() {
        return Err(DecoderError::FrameDecodeFailed {
            msg: format!(
                "output length {} should be the same as input length {}",
                output.len(),
                input.len()
            ),
        });
    }

    let chunks = input.chunks_exact(8);
    for (in_chunk, out_chunk) in chunks.zip(output.chunks_exact_mut(8)) {
        for (i, o) in in_chunk.iter().zip(out_chunk.iter_mut().rev()) {
            *o = if let Some(value) = NumCast::from(*i) {
                value
            } else {
                todo!();
            }
        }
    }

    Ok(())
}

/// Decode from raw 12bit format to O. input length must be divisible by 8
fn decode_r12<O>(input: &[u8], output: &mut [O]) -> Result<(), DecoderError>
where
    O: Copy + ToPrimitive + NumCast + Debug,
{
    if input.len() % 8 != 0 {
        return Err(DecoderError::FrameDecodeFailed {
            msg: format!("input length {} is not divisible by 8", input.len()),
        });
    }

    if output.len() * 2 < input.len() {
        return Err(DecoderError::FrameDecodeFailed {
            msg: format!(
                "output length {} needs to match input length {}",
                output.len(),
                input.len()
            ),
        });
    }

    let chunks = input.chunks_exact(8);
    for (in_chunk, out_chunk) in chunks.zip(output.chunks_exact_mut(4)) {
        for (value_chunk, out_value) in in_chunk.chunks_exact(2).zip(out_chunk.iter_mut().rev()) {
            let value: u16 = FromBytes::read_from_prefix(value_chunk).expect("chunked by 2");
            *out_value = if let Some(value) = NumCast::from(value) {
                value
            } else {
                return Err(DecoderError::FrameDecodeFailed {
                    msg: format!(
                        "dtype conversion error: {out_value:?} does not fit {0}",
                        type_name::<O>()
                    ),
                });
            }
        }
    }

    Ok(())
}
