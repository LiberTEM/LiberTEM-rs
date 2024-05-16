use pyo3::{PyResult, Python};
use serde::{Deserialize, Serialize};

use crate::{
    frame_iterator::FrameChunkedIterator,
    frame_stack::{FrameMeta, FrameStackHandle},
    generic_receiver::{self, Receiver},
};

#[derive(Serialize, Deserialize, Debug, Clone)]
struct FakeFrameMeta {}

impl FrameMeta for FakeFrameMeta {
    fn get_data_length_bytes(&self) -> usize {
        todo!()
    }
}

struct FakeReceiver {}

impl Receiver<FakeFrameMeta> for FakeReceiver {
    fn get_status(&self) -> generic_receiver::ReceiverStatus {
        todo!()
    }

    fn next_timeout(
        &mut self,
        timeout: std::time::Duration,
    ) -> Option<generic_receiver::ReceiverMsg<FakeFrameMeta>> {
        todo!()
    }
}

struct PyFrameIterator {}

impl PyFrameIterator {
    fn get_next_stack(&mut self, max_size: usize, py: Python<'_>) -> PyResult<()> {
        let remainder: Vec<FrameStackHandle<FakeFrameMeta>> = Vec::new();

        let iter = FrameChunkedIterator::new(todo!(), todo!(), todo!(), todo!());

        py.allow_threads(|| {
            iter.get_next_stack_impl(max_size, || {
                py.check_signals()?;
                Ok(())
            });
        });


        Ok(())
    }
}
