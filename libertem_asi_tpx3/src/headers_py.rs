use pyo3::pymethods;

use crate::headers::AcquisitionStart;

#[pymethods]
impl AcquisitionStart {
    pub fn __repr__(&self) -> String {
        format!("{self:?}")
    }
}