use std::ffi::c_int;

use ipc_test::Slot;
use pyo3::{ffi, pyclass, pymethods};

// FIXME: can/should we make this Send'able?
#[pyclass(unsendable)]
struct SlotBuffer {
    slot: Slot,
}

impl SlotBuffer {}

#[pymethods]
impl SlotBuffer {
    unsafe fn __getbuffer__(&self, view: *mut ffi::Py_buffer, flags: c_int) {
        todo!()
    }
}
