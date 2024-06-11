// we can't have generics in pyclasses, so we have to generate impls via macros
// see also: https://pyo3.rs/v0.21.2/class.html#no-generic-parameters

/// This macro generates Python-specific business logic, like dropping the GIL
/// or type conversions. All other logic is implemented in `GenericCamClient`.
///
#[macro_export]
macro_rules! impl_py_cam_client {
    (
        $name: ident,
        $decoder_type: ident,
        $frame_meta_type: ident,
        $mod: ident
    ) => {
        mod impl_cam_client {
            use pyo3::{
                create_exception,
                exceptions::PyException,
                prelude::*,
            };
            use common::{
                cam_client::GenericCamClient,
                decoder::Decoder,
                frame_stack::FrameStackHandle,
            };

            create_exception!($mod, PyCamClientError, PyException);

            #[pyclass]
            pub struct $name {
                client_impl: GenericCamClient<$decoder_type>,
            }

            impl $name {}

            #[pymethods]
            impl $name {
                #[new]
                pub fn new(handle_path: &str) -> PyResult<Self> {
                    Ok(Self {
                        client_impl: GenericCamClient::new(handle_path)
                            .map_err({ |e| PyCamClientError::new_err(e.to_string()) })?,
                    })
                }

                /// Decode into a pre-allocated array.
                ///
                /// This supports user-allocated memory, which enables things like copying
                /// directly into CUDA locked memory and thus getting rid of a memcpy in the
                /// case of CUDA.
                pub fn decode_into_buffer(
                    &self,
                    input: &FrameStackHandle<$frame_meta_type>,
                    out: Bound<'py, PyAny>,
                ) -> PyResult<()> {
                    todo!("decode_into_buffer: Python types; error handling; dtype dispatch");
                    self.client_impl.decode_into_buffer(input, out)
                }

                /// Decode a range of frames into a pre-allocated array.
                ///
                /// This allows for decoding only the data that will be processed
                /// immediately afterwards, allowing for more cache-efficient operations.
                pub fn decode_range_into_buffer(
                    &self,
                    input: &FrameStackHandle<$frame_meta_type>,
                    out: ArrayViewMut3<'_, T>,
                    start_idx: usize,
                    end_idx: usize,
                ) -> PyResult<()> {
                    todo!("decode_range_into_buffer: Python types; dtype dispatch");
                    Ok(self.decoder.decode(input, out, start_idx, end_idx)?)
                }

                /// Free the given `FrameStackHandle`. When calling this, no Python objects
                /// may have references to the memory of the `handle`.
                pub fn frame_stack_done(
                    &mut self,
                    handle: FrameStackHandle<$frame_meta_type>,
                ) -> PyResult<()> {
                    todo!("frame_stack_done: Python types; err handling");
                    self.client_impl.frame_stack_done(handle)
                }

                pub fn close(&mut self) -> PyResult<()> {
                    self.client_impl.close().map_err(|e| {
                        PyCamClientError::new_err(format!("close failed: {e}"))
                    })
                }
            }
        }

        use impl_cam_client::{$name, PyCamClientError};
    };
}
