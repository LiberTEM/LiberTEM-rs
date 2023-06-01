use std::ffi::c_int;

use ipc_test::{SharedSlabAllocator, Slot};
use pyo3::{ffi::PyMemoryView_FromMemory, prelude::*, FromPyPointer};

#[allow(non_upper_case_globals)]
const PyBUF_READ: c_int = 0x100; // somehow not exported by pyo3? oh no...

#[pyclass]
pub struct FrameRef {
    slot_idx: usize,
    memview: PyObject,
}

#[pymethods]
impl FrameRef {
    fn get_memoryview(&self) -> Py<PyAny> {
        self.memview.clone()
    }

    // FIXME: should have some safety - hide the slot index business a bit better
    // and only pass in a whole object, where the inner memoryview can hopefully be
    // properly cleaned up
}

#[pyclass]
pub struct CamClient {
    shm: SharedSlabAllocator,
}

#[pymethods]
impl CamClient {
    #[new]
    fn new(socket_path: &str) -> Self {
        let shm = SharedSlabAllocator::connect(socket_path).expect("connect to shm");
        CamClient { shm }
    }

    fn get_frame_ref(&self, py: Python, slot: usize) -> FrameRef {
        // FIXME: crimes below. need to verify safety, and define the rules that the
        // Python side needs to follow
        let slot_r: Slot = self.shm.get(slot);
        let mv = unsafe {
            PyMemoryView_FromMemory(
                slot_r.ptr as *mut i8,
                slot_r.size.try_into().unwrap(),
                PyBUF_READ,
            )
        };
        let from_ptr: &PyAny = unsafe { FromPyPointer::from_owned_ptr(py, mv) };
        let memview = from_ptr.into_py(py);

        FrameRef {
            memview,
            slot_idx: slot,
        }
    }

    fn done(mut slf: PyRefMut<Self>, slot_idx: usize) {
        slf.shm.free_idx(slot_idx)
    }
}
