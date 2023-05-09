use ipc_test::SharedSlabAllocator;
use log::trace;
use std::ffi::c_int;

use crate::{common::DType, exceptions::ConnectionError, frame_stack::FrameStackHandle};
use pyo3::{exceptions::PyRuntimeError, ffi::PyMemoryView_FromMemory, prelude::*, FromPyPointer};

#[pyclass]
pub struct CamClient {
    shm: Option<SharedSlabAllocator>,
}

#[allow(non_upper_case_globals)]
const PyBUF_READ: c_int = 0x100;

impl CamClient {
    fn get_memoryview(&self, py: Python, raw_data: &[u8]) -> PyObject {
        let ptr = raw_data.as_ptr();
        let length = raw_data.len();

        let mv = unsafe {
            PyMemoryView_FromMemory(ptr as *mut i8, length.try_into().unwrap(), PyBUF_READ)
        };
        let from_ptr: &PyAny = unsafe { FromPyPointer::from_owned_ptr(py, mv) };
        from_ptr.into_py(py)
    }
}

#[pymethods]
impl CamClient {
    #[new]
    fn new(handle_path: &str) -> PyResult<Self> {
        match SharedSlabAllocator::connect(handle_path) {
            Ok(shm) => Ok(CamClient { shm: Some(shm) }),
            Err(e) => {
                let msg = format!("failed to connect to SHM: {:?}", e);
                Err(ConnectionError::new_err(msg))
            }
        }
    }

    fn get_frames(
        &self,
        handle: &FrameStackHandle,
        py: Python,
    ) -> PyResult<Vec<(PyObject, DType)>> {
        let slot: ipc_test::Slot = if let Some(shm) = &self.shm {
            shm.get(handle.slot.slot_idx)
        } else {
            return Err(PyRuntimeError::new_err("can't decompress with closed SHM"));
        };

        Ok(handle
            .get_meta()
            .iter()
            .zip(0..)
            .map(|(meta, idx)| {
                let image_data = handle.get_slice_for_frame(idx, &slot);
                let memory_view = self.get_memoryview(py, image_data);

                (memory_view, meta.dtype.clone())
            })
            .collect())
    }

    fn done(mut slf: PyRefMut<Self>, handle: &FrameStackHandle) -> PyResult<()> {
        let slot_idx = handle.slot.slot_idx;
        if let Some(shm) = &mut slf.shm {
            shm.free_idx(slot_idx);
            Ok(())
        } else {
            Err(PyRuntimeError::new_err(
                "CamClient.done called with SHM closed",
            ))
        }
    }

    fn close(&mut self) {
        self.shm.take();
    }
}

impl Drop for CamClient {
    fn drop(&mut self) {
        trace!("CamClient::drop");
    }
}
