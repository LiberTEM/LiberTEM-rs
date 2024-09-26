use std::fmt::Debug;

use common::{
    decoder::{self, try_cast_if_safe, Decoder, DecoderError},
    frame_stack::{self},
};
use ipc_test::SharedSlabAllocator;
use num::{NumCast, ToPrimitive};
use numpy::ndarray::{s, ArrayViewMut3};
use zerocopy::FromBytes;

use crate::frame_meta::K2FrameMeta;

/// Same as `common::decoder::try_cast_if_safe`, but crop the frame,
/// removing a stripe at the right side.
/// (useful for getting rid of non-frame-data on K2{IS,Summit})
pub fn try_cast_with_crop<I, O>(
    input: &[I],
    output: &mut [O],
    frame_width: usize,
    frame_width_orig: usize,
) -> Result<(), DecoderError>
where
    O: Copy + NumCast + Debug,
    I: Copy + ToPrimitive + Debug,
{
    let in_rows = input.chunks_exact(frame_width_orig);
    let out_rows = output.chunks_exact_mut(frame_width);

    assert_eq!(
        in_rows.len(),
        out_rows.len(),
        "{} != {}; frame_width_orig={}, frame_width={}",
        in_rows.len(),
        out_rows.len(),
        frame_width_orig,
        frame_width,
    );

    for (in_row, out_row) in in_rows.zip(out_rows) {
        try_cast_if_safe(&in_row[0..frame_width], out_row)?;
    }

    // assert!(in_rows.remainder().len() == 0);
    // assert!(out_rows.into_remainder().len() == 0);

    Ok(())
}

#[derive(Debug, Clone, Default)]
pub struct K2Decoder {}

impl Decoder for K2Decoder {
    type FrameMeta = K2FrameMeta;

    fn decode<T>(
        &self,
        shm: &SharedSlabAllocator,
        input: &frame_stack::FrameStackHandle<Self::FrameMeta>,
        output: &mut ArrayViewMut3<'_, T>,
        start_idx: usize,
        end_idx: usize,
    ) -> Result<(), decoder::DecoderError>
    where
        T: decoder::DecoderTargetPixelType,
        u8: num::cast::AsPrimitive<T>,
        u16: num::cast::AsPrimitive<T>,
    {
        // the background thread already converts the integer data to u16,
        // possibly from the 12bit raw format, so all that's left here is to
        // convert/copy to target pixel type `T`:
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
                let data_as_u16: &[u16] =
                    FromBytes::slice_from(raw_input_data).ok_or_else(|| {
                        DecoderError::FrameDecodeFailed {
                            msg: "could not interprete input data as u16".to_owned(),
                        }
                    })?;
                if frame_meta.get_crop() {
                    try_cast_with_crop(
                        data_as_u16,
                        out_slice,
                        frame_meta.get_effective_frame_shape().width,
                        frame_meta.get_raw_frame_shape().width,
                    )?;
                } else {
                    try_cast_if_safe(data_as_u16, out_slice)?;
                }
            }

            Ok(())
        })
    }

    fn zero_copy_available(
        &self,
        _handle: &frame_stack::FrameStackHandle<Self::FrameMeta>,
    ) -> Result<bool, decoder::DecoderError> {
        Ok(false)
    }
}
