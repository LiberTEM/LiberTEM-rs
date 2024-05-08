use common::frame_stack::FrameStackHandle;
use pyo3::{
    exceptions::{self, PyRuntimeError},
    prelude::*,
    types::{PyBytes, PyType},
};
use serde::{Deserialize, Serialize};

use crate::common::{DectrisFrameMeta, PixelType};

/// serializable handle for a stack of frames that live in shm
#[pyclass(name = "FrameStackHandle")]
#[derive(PartialEq, Eq, Serialize, Deserialize, Debug)]
pub struct PyFrameStackHandle {
    handle: Option<FrameStackHandle<DectrisFrameMeta>>,
}

impl PyFrameStackHandle {
    pub fn new(stack: FrameStackHandle<DectrisFrameMeta>) -> Self {
        PyFrameStackHandle {
            handle: Some(stack),
        }
    }

    pub fn get_inner(&self) -> &Option<FrameStackHandle<DectrisFrameMeta>> {
        &self.handle
    }

    pub fn take(&mut self) -> Option<FrameStackHandle<DectrisFrameMeta>> {
        self.handle.take()
    }

    pub fn try_get_inner(&self) -> PyResult<&FrameStackHandle<DectrisFrameMeta>> {
        if let Some(handle) = &self.handle {
            Ok(handle)
        } else {
            Err(PyRuntimeError::new_err(
                "operation on free'd FrameStackHandle".to_owned(),
            ))
        }
    }
}

#[pymethods]
impl PyFrameStackHandle {
    pub fn serialize<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes: Bound<'py, PyBytes> =
            PyBytes::new_bound(py, self.try_get_inner()?.serialize()?.as_slice());
        Ok(bytes)
    }

    #[classmethod]
    fn deserialize<'py>(
        _cls: Bound<'py, PyType>,
        serialized: Bound<'py, PyBytes>,
    ) -> PyResult<Self> {
        Ok(Self {
            handle: Some(FrameStackHandle::deserialize_impl(serialized.as_bytes())?),
        })
    }

    fn get_series_id(&self) -> PyResult<u64> {
        Ok(self.try_get_inner()?.first_meta()?.dimage.series)
    }

    fn get_frame_id(&self) -> PyResult<u64> {
        Ok(self.try_get_inner()?.first_meta()?.dimage.frame)
    }

    fn get_hash(&self) -> PyResult<String> {
        Ok(self.try_get_inner()?.first_meta()?.dimage.hash.clone())
    }

    fn get_pixel_type(&self) -> PyResult<String> {
        Ok(match &self.try_get_inner()?.first_meta()?.dimaged.type_ {
            PixelType::Uint8 => "uint8".to_string(),
            PixelType::Uint16 => "uint16".to_string(),
            PixelType::Uint32 => "uint32".to_string(),
        })
    }

    fn get_encoding(&self) -> PyResult<String> {
        Ok(self.try_get_inner()?.first_meta()?.dimaged.encoding.clone())
    }

    /// return endianess in numpy notation
    fn get_endianess(&self) -> PyResult<String> {
        match self
            .try_get_inner()?
            .first_meta()?
            .dimaged
            .encoding
            .chars()
            .last()
        {
            Some(c) => Ok(c.to_string()),
            None => Err(exceptions::PyValueError::new_err(
                "encoding should be non-empty".to_string(),
            )),
        }
    }

    fn get_shape(&self) -> PyResult<Vec<u64>> {
        Ok(self.try_get_inner()?.first_meta()?.dimaged.shape.clone())
    }

    fn __len__(&self) -> PyResult<usize> {
        Ok(self.try_get_inner()?.len())
    }

    pub fn is_empty(&self) -> PyResult<bool> {
        Ok(self.try_get_inner()?.len() > 0)
    }
}
