use std::fmt::Debug;

use common::decoder::{
    decode_ints_be, try_cast_if_safe, try_cast_primitive, Decoder, DecoderError,
    DecoderTargetPixelType,
};
use log::trace;
use num::{cast::AsPrimitive, Bounded, Num, NumCast, PrimInt, ToPrimitive};
use numpy::ndarray::s;
use zerocopy::FromBytes;

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
        u8: AsPrimitive<T>,
        u16: AsPrimitive<T>,
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
        // Without taking chunk encoding into account, it maps to the input data like this:
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
        O: DecoderTargetPixelType + 'static,
        u8: AsPrimitive<O>,
        u16: AsPrimitive<O>,
    {
        match &frame_meta.layout {
            crate::base_types::Layout::L1x1 => {
                self.decode_frame_single_chip(frame_meta, input, output)
            }
            crate::base_types::Layout::L2x2 => self.decode_frame_quad(frame_meta, input, output),
            crate::base_types::Layout::L2x2G => self.decode_frame_quad(frame_meta, input, output),
            crate::base_types::Layout::LNx1 | crate::base_types::Layout::LNx1G => {
                self.decode_frame_eels(frame_meta, input, output)
            }
        }
    }

    fn decode_frame_eels<O>(
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
            DType::R64 => Err(DecoderError::FrameDecodeFailed {
                msg: format!(
                    "unsupported layout for raw decoding: {:?}",
                    frame_meta.layout
                ),
            }),
        }
    }

    fn decode_frame_quad<O>(
        &self,
        frame_meta: &QdFrameMeta,
        input: &[u8],
        output: &mut [O],
    ) -> Result<(), DecoderError>
    where
        O: DecoderTargetPixelType + 'static,
        u8: AsPrimitive<O>,
        u16: AsPrimitive<O>,
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
                        1 => R1::decode_2x2_raw::<_, 64>(input, output, &frame_meta.layout),
                        6 => R6::decode_2x2_raw::<_, 8>(input, output, &frame_meta.layout),
                        12 => R12::decode_2x2_raw::<_, 4>(input, output, &frame_meta.layout),
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

    fn decode_frame_single_chip<O>(
        &self,
        frame_meta: &QdFrameMeta,
        input: &[u8],
        output: &mut [O],
    ) -> Result<(), DecoderError>
    where
        O: DecoderTargetPixelType,
        u8: AsPrimitive<O>,
        u16: AsPrimitive<O>,
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
                        1 => R1::decode_all::<_, 64>(input, output),
                        6 => R6::decode_all::<_, 8>(input, output),
                        12 => R12::decode_all::<_, 4>(input, output),
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
    type Intermediate: Num;

    /// Bits per pixel excluding padding
    const COUNTER_DEPTH: usize;

    /// how many output pixel values are contained in a single 8-byte chunk?
    const PIXELS_PER_CHUNK: usize;

    /// How many bytes per row on a single chip in raw encoding?
    const BYTES_PER_CHIP_ROW: usize;

    /// take a raw input chunk (8 bytes) and decode it into an output slice
    fn decode_chunk<'a, O, const OUT_PER_CHUNK: usize>(
        input: &[u8; 8],
        output: &mut [O; OUT_PER_CHUNK],
    ) -> Result<(), DecoderError>
    where
        O: Copy + ToPrimitive + NumCast + Bounded + Debug + 'a + 'static,
        Self::Intermediate: AsPrimitive<O>;

    /// Decode all values from `input` into `output`.
    fn decode_all<'a, O, const OUT_PER_CHUNK: usize>(
        input: &[u8],
        output: &mut [O],
    ) -> Result<(), DecoderError>
    where
        O: Copy + ToPrimitive + NumCast + Num + Bounded + Debug + 'a + 'static,
        Self::Intermediate: AsPrimitive<O>,
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
        for (in_chunk, out_chunk) in chunks.zip(output.chunks_exact_mut(OUT_PER_CHUNK)) {
            <Self as RawType>::decode_chunk::<_, OUT_PER_CHUNK>(
                in_chunk.try_into().expect("chunked by 8"),
                out_chunk.try_into().unwrap(),
            )?;
        }

        Ok(())
    }

    fn decode_2x2_raw<O, const OUT_PER_CHUNK: usize>(
        input: &[u8],
        output: &mut [O],
        layout: &Layout,
    ) -> Result<(), DecoderError>
    where
        O: Copy + ToPrimitive + NumCast + Debug + Num + Bounded + 'static,
        Self::Intermediate: AsPrimitive<O>,
    {
        let rows_per_chip: usize = 256; // FIXME: `ROIROWS` support means this would be dynamic? how does that work with raw format?
        const COLS_PER_CHIP: usize = 256;
        let output_row_size: usize = if *layout == Layout::L2x2G { 514 } else { 512 };

        trace!(
            "input.len()={}, output.len()={}",
            input.len(),
            &mut output.len(),
        );

        let (out_top, out_bottom) = output.split_at_mut(output_row_size * rows_per_chip);

        let out_bottom = if *layout == Layout::L2x2G {
            &mut out_bottom[2 * output_row_size..]
        } else {
            out_bottom
        };

        let mut bottom_rows_rev = out_bottom.chunks_exact_mut(output_row_size).rev();
        let mut top_rows = out_top.chunks_exact_mut(output_row_size);

        // input rows come interleaved: Q4/Q3/Q2/Q1 -> we need to de-interleave here!
        // iterate over input rows by four, while taking one row from top and one from bottom:
        let in_by_four_rows = input.chunks_exact(4 * Self::BYTES_PER_CHIP_ROW);

        let col_gap_offset = if *layout == Layout::L2x2G { 2 } else { 0 };

        trace!(
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
            let mut tmp: [O; COLS_PER_CHIP] = [O::zero(); COLS_PER_CHIP];
            Self::decode_all::<_, OUT_PER_CHUNK>(
                &four_rows[0..Self::BYTES_PER_CHIP_ROW],
                &mut tmp,
            )?;
            for (o, i) in bottom_row[COLS_PER_CHIP + col_gap_offset..]
                .iter_mut()
                .rev()
                .zip(tmp.into_iter())
            {
                *o = i;
            }

            // Q3:
            let mut tmp: [O; COLS_PER_CHIP] = [O::zero(); COLS_PER_CHIP];
            Self::decode_all::<_, OUT_PER_CHUNK>(
                &four_rows[Self::BYTES_PER_CHIP_ROW..2 * Self::BYTES_PER_CHIP_ROW],
                &mut tmp,
            )?;
            for (o, i) in bottom_row[0..COLS_PER_CHIP]
                .iter_mut()
                .rev()
                .zip(tmp.into_iter())
            {
                *o = i;
            }

            let top_row = top_rows
                .next()
                .ok_or_else(|| DecoderError::FrameDecodeFailed {
                    msg: "eof top".to_owned(),
                })?;

            // Q2:
            Self::decode_all::<_, OUT_PER_CHUNK>(
                &four_rows[2 * Self::BYTES_PER_CHIP_ROW..3 * Self::BYTES_PER_CHIP_ROW],
                &mut top_row[COLS_PER_CHIP + col_gap_offset..],
            )?;
            // Q1:
            Self::decode_all::<_, OUT_PER_CHUNK>(
                &four_rows[3 * Self::BYTES_PER_CHIP_ROW..],
                &mut top_row[0..COLS_PER_CHIP],
            )?;
        }

        Ok(())
    }

    fn encode_2x2_raw<'a, I>(input: &[I], output: &mut [u8]) -> Result<(), DecoderError>
    where
        I: Copy + ToPrimitive + PrimInt + NumCast + Debug + 'a,
    {
        let cols_per_chip: usize = 256;
        let rows_per_chip: usize = 256;

        let input_chunks = input.chunks_exact(2 * cols_per_chip);

        // encode q1 and q2 into the right part of the [4 | 3 | 2 | 1] array of shape (256, 1024):
        for (row_out, row_in) in output
            .chunks_exact_mut(Self::BYTES_PER_CHIP_ROW * 4)
            .zip(input_chunks)
        {
            let (_row_out_q4q3, row_out_q2q1) =
                &mut row_out.split_at_mut(2 * Self::BYTES_PER_CHIP_ROW);
            let (row_out_q2, row_out_q1) = row_out_q2q1.split_at_mut(Self::BYTES_PER_CHIP_ROW);

            // in the input, q1 comes first:
            let (in_q1, in_q2) = row_in.split_at(cols_per_chip);

            Self::encode_all(&mut in_q1.iter(), row_out_q1)?;
            Self::encode_all(&mut in_q2.iter(), row_out_q2)?;
        }

        let input_chunks = input.chunks_exact(2 * cols_per_chip).skip(rows_per_chip);

        // q3 and q4:
        // encode q1 and q2 into the right part of the [4 | 3 | 2 | 1] array of shape (256, 1024):
        // reverse the rows to flip in y direction:
        for (row_out, row_in) in output
            .chunks_exact_mut(Self::BYTES_PER_CHIP_ROW * 4)
            .rev()
            .zip(input_chunks)
        {
            let (row_out_q4q3, _row_out_q2q1) =
                &mut row_out.split_at_mut(2 * Self::BYTES_PER_CHIP_ROW);
            let (row_out_q4, row_out_q3) = row_out_q4q3.split_at_mut(Self::BYTES_PER_CHIP_ROW);

            // in the input, q3 comes first:
            Self::encode_all(&mut row_in[..cols_per_chip].iter().rev(), row_out_q3)?;
            Self::encode_all(&mut row_in[cols_per_chip..].iter().rev(), row_out_q4)?;
        }

        Ok(())
    }

    fn encode_chunk<'a, I, II>(input: &mut II, output: &mut [u8; 8]) -> Result<(), DecoderError>
    where
        I: Copy + ToPrimitive + PrimInt + NumCast + Debug + 'a,
        II: Iterator<Item = &'a I> + ExactSizeIterator;

    fn encode_all<'a, I, II>(input: &mut II, output: &mut [u8]) -> Result<(), DecoderError>
    where
        I: Copy + ToPrimitive + PrimInt + NumCast + Debug + 'a,
        II: Iterator<Item = &'a I> + ExactSizeIterator,
    {
        if output.len() % 8 != 0 {
            return Err(DecoderError::FrameDecodeFailed {
                msg: format!("output length {} is not divisible by 8", output.len()),
            });
        }

        let output_pixels = (output.len() / 8) * Self::PIXELS_PER_CHUNK;
        let input_pixels = input.len();

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

        let chunks = output.chunks_exact_mut(8);
        for out_chunk in chunks {
            let mut tmp_chunk = [0u8; 8];
            <Self as RawType>::encode_chunk(input, &mut tmp_chunk)?;
            out_chunk.copy_from_slice(&tmp_chunk);
        }

        Ok(())
    }
}

#[derive(Debug)]
pub struct R1 {}

impl RawType for R1 {
    type Intermediate = u8;

    const COUNTER_DEPTH: usize = 1;
    const PIXELS_PER_CHUNK: usize = 64;
    const BYTES_PER_CHIP_ROW: usize = 32;

    fn decode_chunk<'a, O, const OUT_PER_CHUNK: usize>(
        input: &[u8; 8],
        output: &mut [O; OUT_PER_CHUNK],
    ) -> Result<(), DecoderError>
    where
        O: Copy + ToPrimitive + NumCast + Debug + 'a + 'static,
        Self::Intermediate: AsPrimitive<O>,
    {
        let value = u64::from_be_bytes(*input);
        for (out_dest, bin_digit) in output.iter_mut().zip(0..OUT_PER_CHUNK) {
            let out_value = ((value >> bin_digit) & 0x1) as Self::Intermediate;
            *out_dest = (out_value).as_();
        }

        Ok(())
    }

    fn encode_chunk<'a, I, II>(input: &mut II, output: &mut [u8; 8]) -> Result<(), DecoderError>
    where
        I: Copy + ToPrimitive + PrimInt + NumCast + Debug + 'a,
        II: Iterator<Item = &'a I> + ExactSizeIterator,
    {
        let mut dest: u64 = 0;
        for (value, idx) in input.take(64).zip(0..64) {
            // clamp to 1bit value and shift into place:
            let tmp = *value & I::from(0x1).unwrap();
            let tmp: u64 = try_cast_primitive(tmp)?;
            dest |= tmp << idx;
        }
        output.copy_from_slice(&dest.to_be_bytes());

        Ok(())
    }
}

#[derive(Debug)]
pub struct R6 {}

impl RawType for R6 {
    type Intermediate = u8;

    const COUNTER_DEPTH: usize = 6;
    const PIXELS_PER_CHUNK: usize = 8;
    const BYTES_PER_CHIP_ROW: usize = 256;

    fn decode_chunk<'a, O, const OUT_PER_CHUNK: usize>(
        input: &[u8; 8],
        output: &mut [O; OUT_PER_CHUNK],
    ) -> Result<(), DecoderError>
    where
        O: Copy + ToPrimitive + NumCast + Bounded + Debug + 'a + 'static,
        Self::Intermediate: AsPrimitive<O>,
    {
        assert!(output.len() >= 8);
        for (i, o) in input.iter().rev().zip(output.iter_mut()) {
            *o = (*i).as_();
        }
        Ok(())
    }

    fn encode_chunk<'a, I, II>(input: &mut II, output: &mut [u8; 8]) -> Result<(), DecoderError>
    where
        I: Copy + ToPrimitive + NumCast + Debug + 'a,
        II: Iterator<Item = &'a I> + ExactSizeIterator,
    {
        assert!(input.len() >= 8, "input.len() = {}", input.len());

        for (i, o) in input.take(8).zip(output.iter_mut().rev()) {
            *o = try_cast_primitive(*i)?;
        }

        Ok(())
    }
}

#[derive(Debug)]
pub struct R12 {}

impl RawType for R12 {
    type Intermediate = u16;

    const COUNTER_DEPTH: usize = 12;
    const PIXELS_PER_CHUNK: usize = 4;
    const BYTES_PER_CHIP_ROW: usize = 512;

    fn decode_chunk<'a, O, const OUT_PER_CHUNK: usize>(
        input: &[u8; 8],
        output: &mut [O; OUT_PER_CHUNK],
    ) -> Result<(), DecoderError>
    where
        O: Copy + ToPrimitive + NumCast + Debug + 'a + 'static,
        Self::Intermediate: AsPrimitive<O>,
    {
        let values: &[u16; 4] = FromBytes::ref_from(input).unwrap();
        #[cfg(target_endian = "little")]
        {
            output[3] = values[0].swap_bytes().as_();
            output[2] = values[1].swap_bytes().as_();
            output[1] = values[2].swap_bytes().as_();
            output[0] = values[3].swap_bytes().as_();
        }

        #[cfg(target_endian = "big")]
        {
            output[3] = values[0].as_();
            output[2] = values[1].as_();
            output[1] = values[2].as_();
            output[0] = values[3].as_();
        }

        Ok(())
    }

    fn encode_chunk<'a, I, II>(input: &mut II, output: &mut [u8; 8]) -> Result<(), DecoderError>
    where
        I: Copy + ToPrimitive + NumCast + Debug + 'a,
        II: Iterator<Item = &'a I> + ExactSizeIterator,
    {
        assert!(input.len() >= 4, "input.len() = {}", input.len());

        for (i, o) in input.take(4).zip(output.chunks_exact_mut(2).rev()) {
            o.copy_from_slice(&u16::to_be_bytes(try_cast_primitive(*i)?))
        }

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use std::path::PathBuf;

    use common::{
        decoder::{Decoder, DecoderError},
        frame_stack::FrameStackForWriting,
    };
    use ipc_test::SharedSlabAllocator;
    use num::cast::AsPrimitive;
    use numpy::ndarray::Array3;
    use rand::Rng;
    use tempfile::{tempdir, TempDir};

    use crate::{
        base_types::{DType, Layout, QdFrameMeta},
        decoder::R6,
    };

    use super::{QdDecoder, RawType, R1, R12};

    fn generic_encode_decode<
        R: RawType,
        const ENCODED_SIZE: usize,
        const SIZE_PX: usize,
        const OUT_PER_CHUNK: usize,
    >()
    where
        R::Intermediate: AsPrimitive<u32>,
    {
        let mut rng = rand::thread_rng();
        let mut input_data = vec![0; SIZE_PX];
        input_data.fill_with(|| rng.gen::<u32>() % (2u32.pow(R::COUNTER_DEPTH as u32)));

        let input_data = input_data;

        let mut encoded = vec![0u8; ENCODED_SIZE];
        R::encode_all(&mut input_data.iter(), &mut encoded).unwrap();

        eprintln!("input_data: {:X?}", &input_data[..]);
        eprintln!("encoded: {:X?}", &encoded[..]);

        let mut output_data = vec![0; SIZE_PX];
        R::decode_all::<_, OUT_PER_CHUNK>(&encoded, &mut output_data).unwrap();

        assert_eq!(output_data, input_data);
    }

    fn generic_encode_decode_chunk<R: RawType, const SIZE_PX: usize, const OUT_PER_CHUNK: usize>()
    where
        R::Intermediate: AsPrimitive<u32>,
    {
        const ENCODED_SIZE: usize = 16;

        let mut rng = rand::thread_rng();
        let mut input_data = vec![0; SIZE_PX];
        input_data.fill_with(|| rng.gen::<u32>() % (2u32.pow(R::COUNTER_DEPTH as u32)));

        let input_data = input_data;

        let mut encoded = [0u8; ENCODED_SIZE];
        let mut input_iter = input_data.iter();

        let mut chunk = [0u8; 8];
        R::encode_chunk(&mut input_iter, &mut chunk).unwrap();
        encoded[0..8].copy_from_slice(&chunk);
        eprintln!("encoded c1: {:x?}", &encoded[0..8]);

        R::encode_chunk(&mut input_iter, &mut chunk).unwrap();
        encoded[8..].copy_from_slice(&chunk);
        eprintln!("encoded c2: {:x?}", &encoded[8..]);

        eprintln!("input_data: {:X?}", &input_data[..]);
        eprintln!("encoded: {:X?}", &encoded[..]);

        let mut output_data = vec![0; SIZE_PX];
        R::decode_all::<_, OUT_PER_CHUNK>(&encoded, &mut output_data).unwrap();

        assert_eq!(output_data, input_data);
    }

    fn generic_quad_encode_decode<
        R: RawType,
        const ENCODED_SIZE: usize,
        const SIZE_PX: usize,
        const OUT_PER_CHUNK: usize,
    >(
        layout: &Layout,
    ) where
        R::Intermediate: AsPrimitive<u32>,
    {
        let mut rng = rand::thread_rng();
        let mut input_data = vec![0; SIZE_PX];
        input_data.fill_with(|| rng.gen::<u32>() % (2u32.pow(R::COUNTER_DEPTH as u32)));

        let mut encoded = vec![0u8; ENCODED_SIZE];
        R::encode_2x2_raw(&input_data, &mut encoded).unwrap();

        let mut output_data = vec![0; SIZE_PX];
        R::decode_2x2_raw::<_, OUT_PER_CHUNK>(&encoded, &mut output_data, layout).unwrap();

        let matches = output_data
            .iter()
            .zip(input_data.iter())
            .fold(0, |acc, (out, exp)| if out == exp { acc + 1 } else { acc });

        eprintln!("number of matches: {}", matches);
        eprintln!("input_data: {:X?}", &input_data[..]);
        eprintln!("encoded: {:X?}", &encoded[..]);

        assert_eq!(output_data, input_data);
    }

    /// Generate 512x512 input data, encode, then decode with a gap
    /// (the encode part is not so interesting, it's mostly about
    /// synthesizing the zero'd gap pixels)
    fn generic_quad_encode_decode_with_gap<
        R: RawType,
        const ENCODED_SIZE: usize,
        const OUT_PER_CHUNK: usize,
    >()
    where
        R::Intermediate: AsPrimitive<u32>,
    {
        const SIZE_PX_PAYLOAD: usize = 512 * 512;
        const SIZE_PX_W_GAP: usize = 514 * 514;
        let layout = Layout::L2x2G;

        let mut rng = rand::thread_rng();
        let input_data = {
            let mut input = vec![0; SIZE_PX_PAYLOAD];
            input.fill_with(|| rng.gen::<u32>() % (2u32.pow(R::COUNTER_DEPTH as u32)));
            input
        };

        let encoded = {
            let mut encoded = vec![0u8; ENCODED_SIZE];
            R::encode_2x2_raw(&input_data, &mut encoded).unwrap();
            encoded
        };

        {
            // to double check, try the no-gap variant, too:
            let mut output_data_no_gap = vec![0; SIZE_PX_PAYLOAD];
            R::decode_2x2_raw::<_, OUT_PER_CHUNK>(&encoded, &mut output_data_no_gap, &Layout::L2x2)
                .unwrap();
            assert_eq!(output_data_no_gap, input_data);
        }

        // from the non-gapped input data, build the expected, gapped, data:
        let expected_data = {
            let mut expected = vec![0u32; SIZE_PX_W_GAP];

            let top_rows_out = expected.chunks_exact_mut(514).take(256);
            let mut sanity = 0;
            for (o, i) in top_rows_out.zip(input_data.chunks_exact(512).take(256)) {
                // skip two columns here:
                o[0..256].copy_from_slice(&i[0..256]);
                o[256 + 2..].copy_from_slice(&i[256..]);
                sanity += 1;
            }
            assert_eq!(sanity, 256);

            // skip two rows in the output starting from 256 (see the `.skip(..) calls below`)
            let bottom_rows_out = expected.chunks_exact_mut(514).skip(256 + 2);
            let mut sanity = 0;
            for (o, i) in bottom_rows_out.zip(input_data.chunks_exact(512).skip(256)) {
                // skip two columns here:
                o[0..256].copy_from_slice(&i[0..256]);
                o[256 + 2..].copy_from_slice(&i[256..]);
                sanity += 1;
            }
            assert_eq!(sanity, 256);

            expected
        };

        // double check that there are zero'd rows...
        for v in expected_data[514 * 256..514 * (256 + 2)].iter() {
            assert_eq!(*v, 0);
        }

        // and cols:
        for row in expected_data.chunks_exact(514) {
            assert_eq!(&row[256..256 + 2], &[0, 0]);
        }

        let output_data = {
            let mut output_data = vec![0; SIZE_PX_W_GAP];
            R::decode_2x2_raw::<_, OUT_PER_CHUNK>(&encoded, &mut output_data, &layout).unwrap();
            output_data
        };

        let matches = output_data
            .iter()
            .zip(expected_data.iter())
            .fold(0, |acc, (out, exp)| if out == exp { acc + 1 } else { acc });

        eprintln!("number of matches: {}", matches);

        // before checking all output_data, check the stuff that should be zero:
        for v in output_data[514 * 256..514 * (256 + 2)].iter() {
            assert_eq!(*v, 0);
        }
        for row in output_data.chunks_exact(514) {
            assert_eq!(&row[256..256 + 2], &[0, 0]);
        }

        eprintln!("input_data: {:X?}", &input_data[..]);
        eprintln!("encoded: {:X?}", &encoded[..]);

        assert_eq!(output_data, expected_data);
    }

    #[test]
    fn test_r1_encode_decode() {
        generic_encode_decode::<R1, 8192, 65536, 64>();
    }

    #[test]
    fn test_r1_encode_decode_small() {
        generic_encode_decode::<R1, 16, 128, 64>();
    }

    #[test]
    fn test_r1_encode_decode_small_chunk() {
        generic_encode_decode_chunk::<R1, 128, 64>();
    }

    #[test]
    fn test_r6_encode_decode() {
        generic_encode_decode::<R6, 65536, 65536, 8>();
    }

    #[test]
    fn test_r6_encode_decode_small() {
        generic_encode_decode::<R6, 16, 16, 8>();
    }

    #[test]
    fn test_r6_encode_decode_small_chunk() {
        generic_encode_decode_chunk::<R6, 16, 8>();
    }

    #[test]
    fn test_r12_encode_decode() {
        generic_encode_decode::<R12, 131072, 65536, 4>();
    }

    #[test]
    fn test_r12_encode_decode_small() {
        generic_encode_decode::<R12, 32, 16, 4>();
    }

    #[test]
    fn test_r12_encode_decode_small_chunk() {
        generic_encode_decode_chunk::<R12, 8, 4>();
    }

    #[test]
    fn test_r1_encode_decode_quad_raw() {
        const ENCODED_SIZE: usize = 512 * 512 / 8;
        const SIZE_PX: usize = 512 * 512;
        generic_quad_encode_decode::<R1, ENCODED_SIZE, SIZE_PX, 64>(&Layout::L2x2);
    }

    #[test]
    fn test_r6_encode_decode_quad_raw() {
        const ENCODED_SIZE: usize = 512 * 512;
        const SIZE_PX: usize = 512 * 512;
        generic_quad_encode_decode::<R6, ENCODED_SIZE, SIZE_PX, 8>(&Layout::L2x2);
    }

    #[test]
    fn test_r12_encode_decode_quad_raw() {
        const ENCODED_SIZE: usize = 512 * 512 * 2;
        const SIZE_PX: usize = 512 * 512;
        generic_quad_encode_decode::<R12, ENCODED_SIZE, SIZE_PX, 4>(&Layout::L2x2);
    }

    #[test]
    fn test_r1_encode_decode_quad_raw_layout_g() {
        const ENCODED_SIZE: usize = 512 * 512 / 8;
        generic_quad_encode_decode_with_gap::<R1, ENCODED_SIZE, 64>();
    }

    #[test]
    fn test_r6_encode_decode_quad_raw_layout_g() {
        const ENCODED_SIZE: usize = 512 * 512;
        generic_quad_encode_decode_with_gap::<R6, ENCODED_SIZE, 8>();
    }

    #[test]
    fn test_r12_encode_decode_quad_raw_layout_g() {
        const ENCODED_SIZE: usize = 512 * 512 * 2;
        generic_quad_encode_decode_with_gap::<R12, ENCODED_SIZE, 4>();
    }

    fn get_socket_path() -> (TempDir, PathBuf) {
        let socket_dir = tempdir().unwrap();
        let socket_as_path = socket_dir.path().join("stuff.socket");

        (socket_dir, socket_as_path)
    }

    fn make_test_frame_meta(
        dtype: &DType,
        layout: &Layout,
        counter_depth: u8,
        frame_size_bytes: usize,
    ) -> QdFrameMeta {
        let (width_raw, height_raw, width, height, num_chips) = match layout {
            Layout::L1x1 => (256, 256, 256, 256, 1),
            Layout::L2x2 => (1024, 256, 512, 512, 4),
            Layout::LNx1 => (1024, 256, 1024, 256, 2), // eels-like setup
            Layout::L2x2G => (1024, 256, 514, 514, 4),
            Layout::LNx1G => (1024, 256, 1024, 256, 2), // eels-like setup; FIXME: 2px gap?
        };

        QdFrameMeta::new(
            768 + 1 + frame_size_bytes,
            1,
            768,
            num_chips,
            width_raw,
            height_raw,
            width,
            height,
            dtype.clone(),
            layout.clone(),
            // FIXME: this is a lie, but the value is not used yet in decoding, so we get away with it..
            0xFF,
            "".to_owned(),
            0.0,
            0,
            crate::base_types::ColourMode::Single,
            crate::base_types::Gain::HGM,
            Some(crate::base_types::MQ1A {
                timestamp_ext: "".to_owned(),
                acquisition_time_shutter_ns: 0,
                counter_depth,
            }),
        )
    }

    #[test]
    fn test_decoder_single_chip() {
        let bytes_per_frame = 256 * 256 * 2;
        let decoder = QdDecoder::default();
        let frame_meta = make_test_frame_meta(&DType::R64, &Layout::L1x1, 12, bytes_per_frame);

        let (_socket_dir, socket_as_path) = get_socket_path();

        let slot_size = bytes_per_frame;
        let mut shm = SharedSlabAllocator::new(1, slot_size, false, &socket_as_path).unwrap();
        let slot = shm.get_mut().expect("get a free shm slot");
        let mut fs = FrameStackForWriting::new(slot, 1, bytes_per_frame);

        let input_data: Vec<u16> = (0..256 * 256).map(|i| (i % 0xFFFF) as u16).collect();

        fs.write_frame(&frame_meta, |buf| {
            assert_eq!(buf.len(), bytes_per_frame);
            R12::encode_all(&mut input_data.iter(), buf).unwrap();
            Ok::<_, ()>(())
        })
        .unwrap();
        let fsh = fs.writing_done(&mut shm).unwrap();

        let mut decoded = Array3::from_shape_simple_fn([1, 256, 256], || 0u16);

        decoder
            .decode(&shm, &fsh, &mut decoded.view_mut(), 0, 1)
            .unwrap();

        assert_eq!(input_data, decoded.as_slice().unwrap());
    }

    #[test]
    fn test_decoder_quad_l2x2() {
        let bytes_per_frame = 512 * 512 * 2;
        let decoder = QdDecoder::default();
        let frame_meta = make_test_frame_meta(&DType::R64, &Layout::L2x2, 12, bytes_per_frame);

        let (_socket_dir, socket_as_path) = get_socket_path();

        let slot_size = bytes_per_frame;
        let mut shm = SharedSlabAllocator::new(1, slot_size, false, &socket_as_path).unwrap();
        let slot = shm.get_mut().expect("get a free shm slot");
        let mut fs = FrameStackForWriting::new(slot, 1, bytes_per_frame);

        let input_data: Vec<u16> = (0..512 * 512).map(|i| (i % 0xFFFF) as u16).collect();

        fs.write_frame(&frame_meta, |buf| {
            assert_eq!(buf.len(), bytes_per_frame);
            R12::encode_2x2_raw(&input_data, buf).unwrap();
            Ok::<_, ()>(())
        })
        .unwrap();
        let fsh = fs.writing_done(&mut shm).unwrap();

        let mut decoded = Array3::from_shape_simple_fn([1, 512, 512], || 0u16);

        decoder
            .decode(&shm, &fsh, &mut decoded.view_mut(), 0, 1)
            .unwrap();

        assert_eq!(input_data, decoded.as_slice().unwrap());
    }

    #[test]
    fn test_decoder_quad_unsupported_variants() {
        fn _inner(layout: &Layout) {
            let bytes_per_frame = 512 * 512 * 2;
            let decoder = QdDecoder::default();
            let frame_meta = make_test_frame_meta(&DType::R64, layout, 12, bytes_per_frame);

            let (_socket_dir, socket_as_path) = get_socket_path();

            let slot_size = bytes_per_frame;
            let mut shm = SharedSlabAllocator::new(1, slot_size, false, &socket_as_path).unwrap();
            let slot = shm.get_mut().expect("get a free shm slot");
            let mut fs = FrameStackForWriting::new(slot, 1, bytes_per_frame);

            let input_data: Vec<u16> = (0..512 * 512).map(|i| (i % 0xFFFF) as u16).collect();

            fs.write_frame(&frame_meta, |buf| {
                assert_eq!(buf.len(), bytes_per_frame);
                R12::encode_2x2_raw(&input_data, buf).unwrap();
                Ok::<_, ()>(())
            })
            .unwrap();
            let fsh = fs.writing_done(&mut shm).unwrap();

            let mut decoded = Array3::from_shape_simple_fn([1, 512, 512], || 0u16);

            assert!(matches!(
                decoder.decode(&shm, &fsh, &mut decoded.view_mut(), 0, 1),
                Err(DecoderError::FrameDecodeFailed { msg: _ })
            ))
        }
        _inner(&Layout::LNx1);
        _inner(&Layout::LNx1G);
    }

    #[test]
    fn test_decoder_eels_like() {
        fn _inner(layout: &Layout) {
            let bytes_per_frame = 1024 * 256 * 2;
            let decoder = QdDecoder::default();
            let frame_meta = make_test_frame_meta(&DType::U16, layout, 12, bytes_per_frame);

            let (_socket_dir, socket_as_path) = get_socket_path();

            let slot_size = bytes_per_frame;
            let mut shm = SharedSlabAllocator::new(1, slot_size, false, &socket_as_path).unwrap();
            let slot = shm.get_mut().expect("get a free shm slot");
            let mut fs = FrameStackForWriting::new(slot, 1, bytes_per_frame);

            let input_data: Vec<u16> = (0..1024 * 256).map(|i| (i % 0xFFFF) as u16).collect();

            fs.write_frame(&frame_meta, |buf| {
                assert_eq!(buf.len(), bytes_per_frame);

                for (i, o) in input_data.iter().zip(buf.chunks_exact_mut(2)) {
                    o.copy_from_slice(&i.to_be_bytes());
                }

                Ok::<_, ()>(())
            })
            .unwrap();
            let fsh = fs.writing_done(&mut shm).unwrap();

            let mut decoded = Array3::from_shape_simple_fn([1, 1024, 256], || 0u16);
            decoder
                .decode(&shm, &fsh, &mut decoded.view_mut(), 0, 1)
                .unwrap();
            assert_eq!(input_data, decoded.as_slice().unwrap());
        }
        _inner(&Layout::LNx1);
        _inner(&Layout::LNx1G);
    }
}
