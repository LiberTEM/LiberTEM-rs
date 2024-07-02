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
        $py_frame_stack: ident,
        $frame_meta_type: ident,
        $mod: ident
    ) => {
        mod impl_cam_client {
            use common::{
                cam_client::GenericCamClient, decoder::Decoder, frame_stack::FrameStackHandle,
            };
            use num::NumCast;
            use numpy::{
                dtype_bound, Element, PyArray3, PyArrayDescrMethods, PyArrayMethods,
                PyUntypedArray, PyUntypedArrayMethods,
            };
            use pyo3::{create_exception, exceptions::PyException, prelude::*};
            use zerocopy::{AsBytes, FromBytes};

            create_exception!($mod, PyCamClientError, PyException);

            #[pyclass]
            pub struct $name {
                client_impl: GenericCamClient<super::$decoder_type>,
            }

            impl $name {
                fn decode_impl<'py, T: Element + AsBytes + FromBytes + Copy + NumCast + 'static>(
                    &self,
                    input: &super::$py_frame_stack,
                    out: &Bound<'py, PyArray3<T>>,
                    py: Python<'_>,
                ) -> PyResult<()> {
                    let mut out_rw = out.try_readwrite()?;
                    let mut out_arr = out_rw.as_array_mut();
                    return self
                        .client_impl
                        .decode_into_buffer(input.try_get_inner()?, &mut out_arr)
                        .map_err(|e| PyCamClientError::new_err(format!("decode failed: {e}")));
                }
            }

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
                /// Fow now, this has to be a numpy array, but using the buffer
                /// protocol, this can, for example, be a reference to pinned
                /// memory for efficient use with CUDA.
                pub fn decode_into_buffer<'py>(
                    &self,
                    input: &super::$py_frame_stack,
                    out: &Bound<'py, PyUntypedArray>,
                    py: Python<'_>,
                ) -> PyResult<()> {
                    if out.dtype().is_equiv_to(&dtype_bound::<u8>(py)) {
                        let out_u8 = out.downcast::<PyArray3<u8>>()?;
                    }

                    let arr_u16: Result<&Bound<'py, PyArray3<u16>>, _> = out.downcast();
                    let arr_u32: Result<&Bound<'py, PyArray3<u32>>, _> = out.downcast();
                    let arr_i8: Result<&Bound<'py, PyArray3<i8>>, _> = out.downcast();
                    let arr_i16: Result<&Bound<'py, PyArray3<i16>>, _> = out.downcast();
                    let arr_i32: Result<&Bound<'py, PyArray3<i32>>, _> = out.downcast();

                    // todo!("decode_into_buffer: Python types; error handling; dtype dispatch");

                    Ok(())
                }

                /// Decode a range of frames into a pre-allocated array.
                ///
                /// This allows for decoding only the data that will be processed
                /// immediately afterwards, allowing for more cache-efficient operations.
                pub fn decode_range_into_buffer<'py>(
                    &self,
                    input: &super::$py_frame_stack,
                    out: Bound<'py, PyAny>,
                    start_idx: usize,
                    end_idx: usize,
                ) -> PyResult<()> {
                    todo!("decode_range_into_buffer: Python types; dtype dispatch");
                    //Ok(self.client_impl.decode_range_into_buffer(input, out, start_idx, end_idx)?)
                }

                /// Free the given `FrameStackHandle`. When calling this, no Python objects
                /// may have references to the memory of the `handle`.
                pub fn frame_stack_done(
                    &mut self,
                    handle: &mut super::$py_frame_stack,
                ) -> PyResult<()> {
                    let inner_handle = handle.take().ok_or_else(|| {
                        PyCamClientError::new_err(
                            "trying to take already free'd frame stack handle",
                        )
                    })?;
                    self.client_impl
                        .frame_stack_done(inner_handle)
                        .map_err(|e| {
                            PyCamClientError::new_err(format!(
                                "GenericCamClient::frame_stack_done: {e}"
                            ))
                        })
                }

                pub fn close(&mut self) -> PyResult<()> {
                    self.client_impl
                        .close()
                        .map_err(|e| PyCamClientError::new_err(format!("close failed: {e}")))
                }
            }
        }

        use impl_cam_client::{$name, PyCamClientError};
    };
}
