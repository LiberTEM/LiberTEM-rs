use bincode::serialize;
use common::frame_stack::FrameStackHandle;
use pyo3::{
    prelude::*,
    types::{PyBytes, PyType},
};
use serde::{Deserialize, Serialize};

use crate::common::ASIMpxFrameMeta;

/// serializable handle for a stack of frames that live in shm
#[pyclass(name = "FrameStackHandle")]
#[derive(PartialEq, Eq, Serialize, Deserialize, Debug)]
pub struct PyFrameStackHandle {
    handle: Option<FrameStackHandle<ASIMpxFrameMeta>>,
}

impl PyFrameStackHandle {
    pub fn new(handle: FrameStackHandle<ASIMpxFrameMeta>) -> Self {
        PyFrameStackHandle {
            handle: Some(handle),
        }
    }
    pub fn take(&mut self) -> Option<FrameStackHandle<ASIMpxFrameMeta>> {
        self.handle.take()
    }
}

#[pymethods]
impl PyFrameStackHandle {
    pub fn serialize(&self, py: Python) -> PyResult<Py<PyBytes>> {
        let bytes: &PyBytes = PyBytes::new(py, serialize(self).unwrap().as_slice());
        Ok(bytes.into())
    }

    #[classmethod]
    fn deserialize(_cls: &PyType, serialized: &PyBytes) -> PyResult<Self> {
        Self::deserialize_impl(serialized)
    }

    fn get_frame_id(slf: PyRef<Self>) -> PyResult<u64> {
        Ok(slf.first_meta()?.sequence)
    }

    fn get_shape(slf: PyRef<Self>) -> PyResult<(u16, u16)> {
        let meta = slf.first_meta()?;
        Ok((meta.height, meta.width))
    }

    fn __len__(slf: PyRef<Self>) -> usize {
        slf.len()
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use tempfile::{tempdir, TempDir};

    fn get_socket_path() -> (TempDir, PathBuf) {
        let socket_dir = tempdir().unwrap();
        let socket_as_path = socket_dir.path().join("stuff.socket");

        (socket_dir, socket_as_path)
    }
}
