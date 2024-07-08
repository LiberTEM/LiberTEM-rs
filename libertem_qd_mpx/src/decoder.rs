use common::decoder::Decoder;

use crate::base_types::QdFrameMeta;

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
    ) -> Result<(), common::decoder::DecoderError>
    where
        T: 'static + zerocopy::AsBytes + zerocopy::FromBytes + Copy + num::NumCast,
    {
        todo!()
    }

    fn zero_copy_available(
        &self,
        handle: &common::frame_stack::FrameStackHandle<Self::FrameMeta>,
    ) -> Result<bool, common::decoder::DecoderError> {
        todo!()
    }
}
