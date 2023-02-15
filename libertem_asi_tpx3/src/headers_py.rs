use pyo3::pymethods;

use crate::headers::{AcquisitionStart, DType};

#[pymethods]
impl AcquisitionStart {
    pub fn __repr__(&self) -> String {
        format!("{self:?}")
    }
}

#[pymethods]
impl DType {
    pub fn __str__(&self) -> String {
        self.to_str().to_string()
    }
}