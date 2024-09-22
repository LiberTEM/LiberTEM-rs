use pyo3::{pyclass, pymodule, types::PyModule, Bound, PyResult};
use serde::{Deserialize, Serialize};

use common::{
    background_thread::BackgroundThread, decoder::Decoder, frame_stack::FrameMeta,
    generic_connection::AcquisitionConfig, impl_py_cam_client, impl_py_connection,
};

#[pymodule]
fn libertem_k2is(m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct K2FrameMeta {}

impl FrameMeta for K2FrameMeta {
    fn get_data_length_bytes(&self) -> usize {
        todo!()
    }

    fn get_dtype_string(&self) -> String {
        todo!()
    }

    fn get_shape(&self) -> (u64, u64) {
        todo!()
    }
}

#[pyclass]
#[derive(Debug, Clone)]
struct K2AcquisitionConfig {}

impl AcquisitionConfig for K2AcquisitionConfig {
    fn num_frames(&self) -> usize {
        todo!()
    }
}

struct K2BackgroundThread {}

impl BackgroundThread for K2BackgroundThread {
    type FrameMetaImpl = K2FrameMeta;

    type AcquisitionConfigImpl = K2AcquisitionConfig;

    type ExtraControl = ();

    fn channel_to_thread(
        &mut self,
    ) -> &mut std::sync::mpsc::Sender<common::background_thread::ControlMsg<Self::ExtraControl>>
    {
        todo!()
    }

    fn channel_from_thread(
        &mut self,
    ) -> &mut std::sync::mpsc::Receiver<
        common::background_thread::ReceiverMsg<Self::FrameMetaImpl, Self::AcquisitionConfigImpl>,
    > {
        todo!()
    }

    fn join(self) {
        todo!()
    }
}

impl_py_connection!(
    _PyK2ISConnection,
    K2FrameStack,
    K2FrameMeta,
    K2BackgroundThread,
    K2AcquisitionConfig,
    libertem_k2is
);

#[derive(Debug, Clone, Default)]
struct K2Decoder {}

impl Decoder for K2Decoder {
    type FrameMeta = K2FrameMeta;

    fn decode<T>(
        &self,
        shm: &ipc_test::SharedSlabAllocator,
        input: &common::frame_stack::FrameStackHandle<Self::FrameMeta>,
        output: &mut numpy::ndarray::ArrayViewMut3<'_, T>,
        start_idx: usize,
        end_idx: usize,
    ) -> Result<(), common::decoder::DecoderError>
    where
        T: common::decoder::DecoderTargetPixelType,
        u8: num::cast::AsPrimitive<T>,
        u16: num::cast::AsPrimitive<T>,
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

impl_py_cam_client!(
    _PyK2ISCamClient,
    K2Decoder,
    K2FrameStack,
    K2FrameMeta,
    libertem_k2is
);
