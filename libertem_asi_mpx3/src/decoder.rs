use common::decoder::{try_cast_if_safe, Decoder, DecoderError, DecoderTargetPixelType};
use numpy::ndarray::{s, ArrayViewMut3};
use zerocopy::FromBytes;

use crate::base_types::{ASIMpxFrameMeta, DType};

#[derive(Debug, Default)]
pub struct ASIMpxDecoder {}

impl Decoder for ASIMpxDecoder {
    type FrameMeta = ASIMpxFrameMeta;

    fn decode<T>(
        &self,
        shm: &ipc_test::SharedSlabAllocator,
        input: &common::frame_stack::FrameStackHandle<Self::FrameMeta>,
        output: &mut ArrayViewMut3<'_, T>,
        start_idx: usize,
        end_idx: usize,
    ) -> Result<(), DecoderError>
    where
        T: DecoderTargetPixelType,
    {
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
                let dtype = &frame_meta.dtype;
                let raw_input_data = input.get_slice_for_frame(in_idx, slot);

                match dtype {
                    DType::U8 => {
                        try_cast_if_safe(raw_input_data, out_slice)?;
                    }
                    DType::U16 => {
                        let data_as_u16: &[u16] = FromBytes::slice_from(raw_input_data)
                            .ok_or_else(|| DecoderError::FrameDecodeFailed {
                                msg: "could not interprete input data as u16".to_owned(),
                            })?;
                        try_cast_if_safe(data_as_u16, out_slice)?;
                    }
                }
            }

            Ok(())
        })
    }

    fn zero_copy_available(
        &self,
        _handle: &common::frame_stack::FrameStackHandle<Self::FrameMeta>,
    ) -> Result<bool, common::decoder::DecoderError> {
        // FIXME: impl zero copy!
        Ok(false)
    }
}
