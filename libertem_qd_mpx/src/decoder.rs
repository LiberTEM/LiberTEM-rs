use std::fmt::Debug;

use common::decoder::{
    decode_ints_be, try_cast_if_safe, try_cast_primitive, Decoder, DecoderError,
    DecoderTargetPixelType,
};
use itertools::Itertools;
use num::{NumCast, ToPrimitive};
use numpy::ndarray::s;

use crate::base_types::{DType, Layout, QdFrameMeta};

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
        // Without taking encoding into account, it maps to the input data like this:
        //
        // [4 | 3 | 2 | 1]
        //
        // This is an array of shape (256, 1024), where the first row contains
        // the first rows of all quadrants, but quadrants 3 and 4 are also
        // flipped in x and y direction.
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
                self.decode_frame(frame_meta, raw_input_data, out_slice)?;
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
    pub fn decode_frame<O>(
        &self,
        frame_meta: &QdFrameMeta,
        input: &[u8],
        output: &mut [O],
    ) -> Result<(), DecoderError>
    where
        O: DecoderTargetPixelType,
    {
        match &frame_meta.layout {
            crate::base_types::Layout::L1x1 => self.decode_frame_single(frame_meta, input, output),
            crate::base_types::Layout::L2x2 => self.decode_frame_quad(frame_meta, input, output),
            crate::base_types::Layout::L2x2G => self.decode_frame_quad(frame_meta, input, output),
            layout @ (crate::base_types::Layout::LNx1 | crate::base_types::Layout::LNx1G) => {
                Err(DecoderError::FrameDecodeFailed {
                    msg: format!("unsupported layout: {layout:?}"),
                })
            }
        }
    }

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
                if let Some(mq1a) = &frame_meta.mq1a {
                    match mq1a.counter_depth {
                        1 => R1::decode_2x2_raw(input, output, &frame_meta.layout),
                        6 => R6::decode_2x2_raw(input, output, &frame_meta.layout),
                        12 => R12::decode_2x2_raw(input, output, &frame_meta.layout),
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
                        1 => R1::decode_all(input, &mut output.iter_mut()),
                        6 => R6::decode_all(input, &mut output.iter_mut()),
                        12 => R12::decode_all(input, &mut output.iter_mut()),
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

pub trait RawType {
    /// Bits per pixel excluding padding
    const COUNTER_DEPTH: usize;

    /// how many output pixel values are contained in a single 8-byte chunk?
    const PIXELS_PER_CHUNK: usize;

    /// How many bytes per row on a single chip in raw encoding?
    const BYTES_PER_CHIP_ROW: usize;

    /// take a raw input chunk (8 bytes) and decode it into an output slice
    fn decode_chunk<'a, O, OI>(input: &[u8; 8], output: &mut OI) -> Result<(), DecoderError>
    where
        O: Copy + ToPrimitive + NumCast + Debug + 'a,
        OI: Iterator<Item = &'a mut O> + ExactSizeIterator;

    /// Decode all values from `input` into `output`.
    fn decode_all<'a, O, OI>(input: &[u8], output: &mut OI) -> Result<(), DecoderError>
    where
        O: Copy + ToPrimitive + NumCast + Debug + 'a,
        OI: Iterator<Item = &'a mut O> + DoubleEndedIterator + ExactSizeIterator + Itertools,
    {
        if input.len() % 8 != 0 {
            return Err(DecoderError::FrameDecodeFailed {
                msg: format!("input length {} is not divisible by 8", input.len()),
            });
        }

        let output_pixels = output.len();
        let input_pixels = (input.len() / 8) * Self::PIXELS_PER_CHUNK;

        if output_pixels != input_pixels {
            return Err(DecoderError::FrameDecodeFailed {
                msg: format!(
                    "output length {} should match input pixels ({}; length={})",
                    output_pixels,
                    input_pixels,
                    input.len()
                ),
            });
        }

        let chunks = input.chunks_exact(8);
        for in_chunk in chunks {
            <Self as RawType>::decode_chunk(in_chunk.try_into().expect("chunked by 8"), output)?;
        }

        Ok(())
    }

    fn decode_2x2_raw<O>(
        input: &[u8],
        output: &mut [O],
        _layout: &Layout,
    ) -> Result<(), DecoderError>
    where
        O: Copy + ToPrimitive + NumCast + Debug,
    {
        let rows_per_chip: usize = 256; // FIXME: `ROIROWS` support means this would be dynamic? how does that work with raw format?
        let cols_per_chip: usize = 256;
        let output_row_size: usize = 512; // FIXME: 2x2G -> 514

        eprintln!(
            "input.len()={}, output.len()={}",
            input.len(),
            &mut output.len(),
        );

        let (out_top, out_bottom) = output.split_at_mut(output_row_size * rows_per_chip);

        let mut bottom_rows_rev = out_bottom.chunks_exact_mut(output_row_size).rev();
        let mut top_rows = out_top.chunks_exact_mut(output_row_size);

        // input rows come interleaved: Q4/Q3/Q2/Q1 -> we need to de-interleave here!
        // iterate over input rows by four, while taking one row from top and one from bottom:
        let in_by_four_rows = input.chunks_exact(4 * Self::BYTES_PER_CHIP_ROW);

        eprintln!(
            "top_len={} bottom_len={} in_by_four_len={}",
            top_rows.len(),
            bottom_rows_rev.len(),
            in_by_four_rows.len()
        );

        for four_rows in in_by_four_rows {
            let bottom_row =
                bottom_rows_rev
                    .next()
                    .ok_or_else(|| DecoderError::FrameDecodeFailed {
                        msg: "eof bottom".to_owned(),
                    })?;
            // Q4:
            Self::decode_all(
                &four_rows[0..Self::BYTES_PER_CHIP_ROW],
                &mut bottom_row[cols_per_chip..].iter_mut().rev(),
            )?;
            // Q3:
            Self::decode_all(
                &four_rows[Self::BYTES_PER_CHIP_ROW..2 * Self::BYTES_PER_CHIP_ROW],
                &mut bottom_row[0..cols_per_chip].iter_mut().rev(),
            )?;

            let top_row = top_rows
                .next()
                .ok_or_else(|| DecoderError::FrameDecodeFailed {
                    msg: "eof top".to_owned(),
                })?;

            // Q2:
            Self::decode_all(
                &four_rows[2 * Self::BYTES_PER_CHIP_ROW..3 * Self::BYTES_PER_CHIP_ROW],
                &mut top_row[cols_per_chip..].iter_mut(),
            )?;
            // Q1:
            Self::decode_all(
                &four_rows[3 * Self::BYTES_PER_CHIP_ROW..],
                &mut top_row[0..cols_per_chip].iter_mut(),
            )?;
        }

        Ok(())
    }
}

#[derive(Debug)]
pub struct R1 {}

impl RawType for R1 {
    const COUNTER_DEPTH: usize = 1;
    const PIXELS_PER_CHUNK: usize = 64;
    const BYTES_PER_CHIP_ROW: usize = 32;

    fn decode_chunk<'a, O, OI>(input: &[u8; 8], output: &mut OI) -> Result<(), DecoderError>
    where
        O: Copy + ToPrimitive + NumCast + Debug + 'a,
        OI: Iterator<Item = &'a mut O> + ExactSizeIterator,
    {
        let value = u64::from_be_bytes(*input);
        let old_len = output.len();
        assert!(output.len() >= 64);
        for (out_dest, bin_digit) in output.take(64).zip(0..) {
            let out_value = (value >> bin_digit) & 0x1;
            *out_dest = try_cast_primitive(out_value)?;
        }
        let diff = old_len - output.len();

        assert_eq!(diff, 64);

        Ok(())
    }
}

#[derive(Debug)]
pub struct R6 {}

impl RawType for R6 {
    const COUNTER_DEPTH: usize = 6;
    const PIXELS_PER_CHUNK: usize = 8;
    const BYTES_PER_CHIP_ROW: usize = 256;

    fn decode_chunk<'a, O, OI>(input: &[u8; 8], output: &mut OI) -> Result<(), DecoderError>
    where
        O: Copy + ToPrimitive + NumCast + Debug + 'a,
        OI: Iterator<Item = &'a mut O>,
    {
        for (i, o) in input.iter().rev().zip(output) {
            *o = try_cast_primitive(*i)?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct R12 {}

impl RawType for R12 {
    const COUNTER_DEPTH: usize = 12;
    const PIXELS_PER_CHUNK: usize = 4;
    const BYTES_PER_CHIP_ROW: usize = 512;

    fn decode_chunk<'a, O, OI>(input: &[u8; 8], output: &mut OI) -> Result<(), DecoderError>
    where
        O: Copy + ToPrimitive + NumCast + Debug + 'a,
        OI: Iterator<Item = &'a mut O>,
    {
        for (value_chunk, out_value) in input.chunks_exact(2).rev().zip(output) {
            let value = u16::from_be_bytes(value_chunk.try_into().expect("chunked by 2 bytes"));
            *out_value = try_cast_primitive(value)?;
        }

        Ok(())
    }
}
