// we can't have generics in pyclasses, so we have to generate impls via macros
// see also: https://pyo3.rs/v0.21.2/class.html#no-generic-parameters

#[macro_export]
macro_rules! decode_for_dtype {
    (
        $dtype: ident,
        $self: ident,
        $input: ident,
        $out: ident,
        $start_idx: ident,
        $end_idx: ident,
        $py: ident
    ) => {
        if $out.dtype().is_equiv_to(&dtype_bound::<$dtype>($py)) {
            let out_downcast = $out.downcast::<PyArray3<$dtype>>()?;
            $self.decode_impl($input, out_downcast, $start_idx, $end_idx, $py)?;
            return Ok(());
        }
    };
}

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
                decoder::{Decoder, DecoderTargetPixelType},
                frame_stack::FrameStackHandle,
                generic_cam_client::GenericCamClient,
            };
            use ipc_test::SharedSlabAllocator;
            use num::{
                cast::AsPrimitive,
                complex::{Complex32, Complex64},
                NumCast,
            };
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
                fn decode_impl<'py, T>(
                    &self,
                    input: &super::$py_frame_stack,
                    out: &Bound<'py, PyArray3<T>>,
                    start_idx: usize,
                    end_idx: usize,
                    py: Python<'_>,
                ) -> PyResult<()>
                where
                    T: Element + DecoderTargetPixelType,
                    u8: AsPrimitive<T>,
                    u16: AsPrimitive<T>,
                {
                    let mut out_rw = out.try_readwrite()?;
                    let mut out_arr = out_rw.as_array_mut();
                    return self
                        .client_impl
                        .decode_range_into_buffer(
                            input.try_get_inner()?,
                            &mut out_arr,
                            start_idx,
                            end_idx,
                        )
                        .map_err(|e| PyCamClientError::new_err(format!("decode failed: {e}")));
                }

                pub fn get_shm(&self) -> PyResult<&SharedSlabAllocator> {
                    self.client_impl
                        .get_shm()
                        .map_err(|e| PyCamClientError::new_err(e.to_string()))
                }
            }

            #[pymethods]
            impl $name {
                #[new]
                pub fn new(handle_path: &str) -> PyResult<Self> {
                    Ok(Self {
                        client_impl: GenericCamClient::new(handle_path)
                            .map_err(|e| PyCamClientError::new_err(e.to_string()))?,
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
                    self.decode_range_into_buffer(input, out, 0, input.__len__()?, py)
                }

                /// Decode a range of frames into a pre-allocated array.
                ///
                /// This allows for decoding only the data that will be processed
                /// immediately afterwards, allowing for more cache-efficient operations.
                pub fn decode_range_into_buffer<'py>(
                    &self,
                    input: &super::$py_frame_stack,
                    out: &Bound<'py, PyUntypedArray>,
                    start_idx: usize,
                    end_idx: usize,
                    py: Python<'_>,
                ) -> PyResult<()> {
                    if start_idx >= end_idx || end_idx > input.__len__()? {
                        return Err(PyCamClientError::new_err(format!(
                            "invalid start or end index: [{start_idx},{end_idx})"
                        )));
                    }

                    $crate::decode_for_dtype!(u8, self, input, out, start_idx, end_idx, py);
                    $crate::decode_for_dtype!(u16, self, input, out, start_idx, end_idx, py);
                    $crate::decode_for_dtype!(u32, self, input, out, start_idx, end_idx, py);
                    $crate::decode_for_dtype!(u64, self, input, out, start_idx, end_idx, py);

                    $crate::decode_for_dtype!(i8, self, input, out, start_idx, end_idx, py);
                    $crate::decode_for_dtype!(i16, self, input, out, start_idx, end_idx, py);
                    $crate::decode_for_dtype!(i32, self, input, out, start_idx, end_idx, py);
                    $crate::decode_for_dtype!(i64, self, input, out, start_idx, end_idx, py);

                    $crate::decode_for_dtype!(f32, self, input, out, start_idx, end_idx, py);
                    $crate::decode_for_dtype!(f64, self, input, out, start_idx, end_idx, py);

                    // FIXME: figure out zerocopy::AsBytes for Complex types
                    // $crate::decode_for_dtype!(Complex32, self, input, out, start_idx, end_idx, py);
                    // $crate::decode_for_dtype!(Complex64, self, input, out, start_idx, end_idx, py);

                    Err(PyCamClientError::new_err(format!(
                        "unknown output dtype: {:?}",
                        out.dtype()
                    )))
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
