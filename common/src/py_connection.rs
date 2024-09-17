// we can't have generics in pyclasses, so we have to generate impls via macros
// see also: https://pyo3.rs/v0.21.2/class.html#no-generic-parameters

/// This macro generates Python-specific business logic, like dropping the GIL
/// or type conversions. All other logic is implemented in `GenericConnection`.
#[macro_export]
macro_rules! impl_py_connection {
    (
        $name: ident,
        $name_frame_stack: ident,
        $frame_meta_type: ident,
        $background_thread_type: ident,
        $pending_acquisition_type: ident,
        $mod: ident
    ) => {
        mod impl_connection {
            use bincode::serialize;
            use common::{
                background_thread::BackgroundThread,
                decoder::Decoder,
                frame_stack::{FrameMeta, FrameStackHandle},
                generic_cam_client::GenericCamClient,
                generic_connection::{ConnectionStatus, GenericConnection},
            };
            use ipc_test::SharedSlabAllocator;
            use num::NumCast;
            use numpy::{
                dtype_bound, Element, PyArray3, PyArrayDescrMethods, PyArrayMethods,
                PyUntypedArray, PyUntypedArrayMethods,
            };
            use pyo3::{
                create_exception,
                exceptions::PyException,
                prelude::*,
                types::{PyBytes, PyType},
            };
            use stats::Stats;
            use std::time::Duration;
            use zerocopy::{AsBytes, FromBytes};

            create_exception!($mod, PyConnectionError, PyException);

            #[pyclass]
            pub struct $name {
                conn_impl: Option<
                    GenericConnection<
                        super::$background_thread_type,
                        super::$pending_acquisition_type,
                    >,
                >,
                shm: Option<SharedSlabAllocator>,
                remainder: Vec<FrameStackHandle<super::$frame_meta_type>>,
                stats: Stats,
            }

            impl $name {
                pub fn new(
                    shm: SharedSlabAllocator,
                    conn_impl: GenericConnection<
                        super::$background_thread_type,
                        super::$pending_acquisition_type,
                    >,
                ) -> Self {
                    Self {
                        conn_impl: Some(conn_impl),
                        shm: Some(shm),
                        remainder: Vec::new(),
                        stats: Stats::new(),
                    }
                }

                fn get_conn_mut(
                    &mut self,
                ) -> PyResult<
                    &mut GenericConnection<
                        super::$background_thread_type,
                        super::$pending_acquisition_type,
                    >,
                > {
                    match &mut self.conn_impl {
                        None => Err(PyConnectionError::new_err("connection is closed")),
                        Some(c) => Ok(c),
                    }
                }

                fn get_conn(
                    &self,
                ) -> PyResult<
                    &GenericConnection<
                        super::$background_thread_type,
                        super::$pending_acquisition_type,
                    >,
                > {
                    match &self.conn_impl {
                        None => Err(PyConnectionError::new_err("connection is closed")),
                        Some(c) => Ok(c),
                    }
                }

                fn get_shm_mut(&mut self) -> PyResult<&mut SharedSlabAllocator> {
                    match &mut self.shm {
                        None => Err(PyConnectionError::new_err("shm is closed")),
                        Some(shm) => Ok(shm),
                    }
                }

                fn get_shm(&self) -> PyResult<&SharedSlabAllocator> {
                    match &self.shm {
                        None => Err(PyConnectionError::new_err("shm is closed")),
                        Some(shm) => Ok(shm),
                    }
                }

                pub fn send_specialized(
                    &mut self,
                    msg: <super::$background_thread_type as BackgroundThread>::ExtraControl,
                ) -> PyResult<()> {
                    let mut conn = self.get_conn_mut()?;
                    conn.send_specialized(msg)
                        .map_err(|e| PyConnectionError::new_err(e.to_string()))?;
                    Ok(())
                }

                pub fn wait_for_status(
                    &mut self,
                    desired_status: ConnectionStatus,
                    timeout: Option<Duration>,
                ) -> PyResult<()> {
                    let mut conn = self.get_conn_mut()?;
                    conn.wait_for_status(desired_status, timeout, || {
                        // re-acquire GIL to check if we need to break
                        Python::with_gil(|py| py.check_signals())?;
                        Ok::<_, PyErr>(())
                    })
                    .map_err(|e| PyConnectionError::new_err(e.to_string()))
                }
            }

            #[pymethods]
            impl $name {
                pub fn get_next_stack(
                    &mut self,
                    max_size: usize,
                    py: Python<'_>,
                ) -> PyResult<Option<$name_frame_stack>> {
                    let conn_impl = self.get_conn_mut()?;
                    match py.allow_threads(|| {
                        conn_impl.get_next_stack(max_size, || {
                            // re-acquire GIL to check if we need to break
                            Python::with_gil(|py| py.check_signals())?;
                            Ok::<_, PyErr>(())
                        })
                    }) {
                        Ok(None) => Ok(None),
                        Ok(Some(stack)) => Ok(Some($name_frame_stack::new(stack))),
                        Err(e) => Err(PyConnectionError::new_err(e.to_string())),
                    }
                }

                pub fn wait_for_arm(
                    &mut self,
                    timeout: Option<f32>,
                    py: Python<'_>,
                ) -> PyResult<Option<super::$pending_acquisition_type>> {
                    let timeout = timeout.map(Duration::from_secs_f32);
                    py.allow_threads(|| {
                        let conn_impl = self.get_conn_mut()?;
                        conn_impl
                            .wait_for_arm(timeout, || {
                                // re-acquire GIL to check if we need to break
                                Python::with_gil(|py| py.check_signals())?;
                                Ok::<_, PyErr>(())
                            })
                            .map_err(|e| PyConnectionError::new_err(e.to_string()))
                    })
                }

                pub fn get_socket_path(&self) -> PyResult<String> {
                    let shm = self.get_shm()?;
                    Ok(shm.get_handle().os_handle)
                }

                pub fn close(&mut self) -> PyResult<()> {
                    if let Some(mut conn_impl) = self.conn_impl.take() {
                        conn_impl.log_shm_stats();
                        conn_impl.reset_stats();
                        conn_impl.close();
                        Ok(())
                    } else {
                        Err(PyConnectionError::new_err("already closed".to_owned()))
                    }
                }

                pub fn is_running(&self) -> PyResult<bool> {
                    let conn_impl = self.get_conn()?;
                    Ok(conn_impl.is_running())
                }

                pub fn cancel(&mut self, timeout: Option<f32>, py: Python<'_>) -> PyResult<()> {
                    let conn_impl = self.get_conn_mut()?;

                    let timeout = timeout.map(Duration::from_secs_f32);
                    py.allow_threads(|| {
                        let conn_impl = self.get_conn_mut()?;
                        conn_impl
                            .cancel(&timeout, || {
                                // re-acquire GIL to check if we need to break
                                Python::with_gil(|py| py.check_signals())?;
                                Ok::<_, PyErr>(())
                            })
                            .map_err(|e| {
                                PyConnectionError::new_err(format!("cancellation failed: {e}"))
                            })
                    })
                }

                pub fn start_passive(
                    &mut self,
                    timeout: Option<f32>,
                    py: Python<'_>,
                ) -> PyResult<()> {
                    let timeout = timeout.map(Duration::from_secs_f32);

                    py.allow_threads(|| {
                        let conn_impl = self.get_conn_mut()?;
                        conn_impl
                            .start_passive(
                                || {
                                    Python::with_gil(|py| py.check_signals())?;
                                    Ok::<_, PyErr>(())
                                },
                                &timeout,
                            )
                            .map_err(|e| {
                                PyConnectionError::new_err(format!("start_passive failed: {e}"))
                            })?;

                        conn_impl
                            .wait_for_status(ConnectionStatus::Armed, timeout, || {
                                // re-acquire GIL to check if we need to break
                                Python::with_gil(|py| py.check_signals())?;
                                Ok::<_, PyErr>(())
                            })
                            .map_err(|e| PyConnectionError::new_err(e.to_string()))?;

                        Ok(())
                    })
                }

                pub fn log_shm_stats(&self) -> PyResult<()> {
                    let conn_impl = self.get_conn()?;
                    conn_impl.log_shm_stats();
                    Ok(())
                }
            }

            #[pyclass]
            pub struct $name_frame_stack {
                inner: Option<FrameStackHandle<super::$frame_meta_type>>,
            }

            impl $name_frame_stack {
                pub fn new(inner: FrameStackHandle<super::$frame_meta_type>) -> Self {
                    Self { inner: Some(inner) }
                }

                pub fn try_get_inner(
                    &self,
                ) -> PyResult<&FrameStackHandle<super::$frame_meta_type>> {
                    if let Some(inner) = &self.inner {
                        Ok(inner)
                    } else {
                        Err(PyConnectionError::new_err(
                            "operation on free'd FrameStackHandle".to_owned(),
                        ))
                    }
                }

                pub fn take(&mut self) -> Option<FrameStackHandle<super::$frame_meta_type>> {
                    self.inner.take()
                }

                pub fn deserialize_impl<'py>(serialized: Bound<'py, PyBytes>) -> PyResult<Self> {
                    let data = serialized.as_bytes();
                    let inner: FrameStackHandle<super::$frame_meta_type> =
                        bincode::deserialize(data).map_err(|e| {
                            let msg = format!("could not deserialize FrameStackHandle: {e:?}");
                            PyConnectionError::new_err(msg)
                        })?;

                    Ok(Self { inner: Some(inner) })
                }
            }

            #[pymethods]
            impl $name_frame_stack {
                pub fn __len__(&self) -> PyResult<usize> {
                    Ok(self.try_get_inner()?.len())
                }

                pub fn get_dtype_string(&self) -> PyResult<String> {
                    Ok(self.try_get_inner()?.first_meta().get_dtype_string())
                }

                pub fn get_shape(&self) -> PyResult<(u64, u64)> {
                    Ok(self.try_get_inner()?.first_meta().get_shape())
                }

                pub fn serialize<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
                    let bytes: Bound<'py, PyBytes> = PyBytes::new_bound(
                        py,
                        serialize(self.try_get_inner()?).unwrap().as_slice(),
                    );
                    Ok(bytes.into())
                }

                #[classmethod]
                pub fn deserialize<'py>(
                    _cls: Bound<'py, PyType>,
                    serialized: Bound<'py, PyBytes>,
                ) -> PyResult<Self> {
                    Self::deserialize_impl(serialized)
                }
            }

            impl $name_frame_stack {
                pub fn get_meta(&self) -> PyResult<&[super::$frame_meta_type]> {
                    Ok(&self.try_get_inner()?.get_meta())
                }
            }
        }

        use impl_connection::{$name, $name_frame_stack, PyConnectionError};
    };
}
