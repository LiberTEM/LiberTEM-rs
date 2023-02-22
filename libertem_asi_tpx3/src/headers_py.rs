use pyo3::pymethods;

use crate::headers::{AcquisitionStart, DType};

#[pymethods]
impl AcquisitionStart {
    pub fn __repr__(&self) -> String {
        format!("{self:?}")
    }

    pub fn get_nav_shape(&self) -> (u16, u16) {
        self.nav_shape
    }

    pub fn get_sig_shape(&self) -> (u16, u16) {
        self.sig_shape
    }

    pub fn get_indptr_dtype(&self) -> String {
        self.indptr_dtype.to_str().to_string()
    }

    pub fn get_indices_dtype(&self) -> String {
        self.indices_dtype.to_str().to_string()
    }
}

#[pymethods]
impl DType {
    pub fn __str__(&self) -> String {
        self.to_str().to_string()
    }
}
