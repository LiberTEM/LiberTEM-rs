// use pyo3::{pyclass, pymethods};
//
// use crate::frame_stack::FrameStackHandle;

// we can't have generics in pyclasses, so we have to generate impls via macros
// see also: https://pyo3.rs/v0.21.2/class.html#no-generic-parameters

#[macro_export]
macro_rules! impl_py_frame_stack {
    ($name: ident, $frame_meta_type: ident) => {
        #[pyclass]
        pub struct $name {
            inner: FrameStackHandle<$frame_meta_type>,
        }

        impl $name {
            fn new(inner: FrameStackHandle<$frame_meta_type>) -> Self {
                Self { inner }
            }
        }

        #[pymethods]
        impl $name {}
    };
}
