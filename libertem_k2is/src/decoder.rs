use common::{
    decoder::{self, try_cast_if_safe, Decoder, DecoderError},
    frame_stack,
};
use ipc_test::SharedSlabAllocator;
use numpy::ndarray::{s, ArrayViewMut3};
use zerocopy::FromBytes;

use crate::frame_meta::K2FrameMeta;

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
            for ((_frame_meta, out_idx), in_idx) in
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
                try_cast_if_safe(data_as_u16, out_slice)?;
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
