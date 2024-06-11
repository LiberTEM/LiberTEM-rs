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
        create_exception!($mod, PyConnectionError, PyException);

        #[pyclass]
        pub struct $name {
            conn_impl: Option<GenericConnection<$background_thread_type, $pending_acquisition_type>>,
            shm: Option<SharedSlabAllocator>,
            remainder: Vec<FrameStackHandle<$frame_meta_type>>,
            stats: Stats,
        }

        impl $name {
            pub fn new(shm: SharedSlabAllocator, conn_impl: GenericConnection<$background_thread_type, $pending_acquisition_type>) -> Self {
                Self {
                    conn_impl: Some(conn_impl),
                    shm: Some(shm),
                    remainder: Vec::new(),
                    stats: Stats::new(),
                }
            }

            fn get_conn_mut(&mut self) -> PyResult<&mut GenericConnection<$background_thread_type, $pending_acquisition_type>> {
                match &mut self.conn_impl {
                    None => Err(PyConnectionError::new_err("connection is closed")),
                    Some(c) => Ok(c),
                }
            }

            fn get_conn(&self) -> PyResult<&GenericConnection<$background_thread_type, $pending_acquisition_type>> {
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

            fn send_specialized(&mut self, msg: <$background_thread_type as BackgroundThread>::ExtraControl) -> PyResult<()> {
                let mut conn = self.get_conn_mut()?;
                conn.send_specialized(msg).map_err(|e| PyConnectionError::new_err(e.to_string()))?;
                Ok(())
            }

            fn wait_for_status(
                &mut self,
                desired_status: ConnectionStatus,
                timeout: Duration,
            ) -> PyResult<()>
            {
                let mut conn = self.get_conn_mut()?;
                conn.wait_for_status(desired_status, timeout, || {
                    // re-acquire GIL to check if we need to break
                    Python::with_gil(|py| py.check_signals())?;
                    Ok::<_, PyErr>(())
                }).map_err(|e| PyConnectionError::new_err(e.to_string()))
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
                    Ok(Some(stack)) => {
                        Ok(Some($name_frame_stack::new(stack)))
                    },
                    Err(e) => {
                        Err(PyConnectionError::new_err(e.to_string()))
                    }
                }
            }

            pub fn wait_for_arm(&mut self, timeout: f32) -> PyResult<Option<$pending_acquisition_type>> {
                let conn_impl = self.get_conn_mut()?;
                conn_impl.wait_for_arm(Duration::from_secs_f32(timeout), || {
                    // re-acquire GIL to check if we need to break
                    Python::with_gil(|py| py.check_signals())?;
                    Ok::<_, PyErr>(())
                }).map_err(|e| {
                    PyConnectionError::new_err(e.to_string())
                })
            }

            pub fn get_socket_path(&self) -> PyResult<String> {
                let shm = self.get_shm()?;
                Ok(shm.get_handle().os_handle)
            }

            pub fn close(&mut self) -> PyResult<()> {
                let conn_impl = self.get_conn_mut()?;
                conn_impl.log_shm_stats();
                conn_impl.reset_stats();
                Ok(())
            }

            pub fn is_running(&self) -> PyResult<bool> {
                let conn_impl = self.get_conn()?;
                Ok(conn_impl.is_running())
            }

            pub fn start_passive(&mut self) -> PyResult<()> {
                let conn_impl = self.get_conn_mut()?;
                conn_impl.start_passive(|| {
                    Python::with_gil(|py| py.check_signals())?;
                    Ok::<_, PyErr>(())
                }).map_err(|e| {
                    PyConnectionError::new_err(format!("start_passive failed: {e}"))
                })?;

                conn_impl.wait_for_status(ConnectionStatus::Armed, Duration::from_millis(100), || {
                    // re-acquire GIL to check if we need to break
                    Python::with_gil(|py| py.check_signals())?;
                    Ok::<_, PyErr>(())
                }).map_err(|e| PyConnectionError::new_err(e.to_string()))?;

                Ok(())
            }

            pub fn log_shm_stats(&self) -> PyResult<()> {
                let conn_impl = self.get_conn()?;
                conn_impl.log_shm_stats();
                Ok(())
            }
        }

        #[pyclass]
        pub struct $name_frame_stack {
            inner: Option<FrameStackHandle<$frame_meta_type>>,
        }

        impl $name_frame_stack {
            fn new(inner: FrameStackHandle<$frame_meta_type>) -> Self {
                Self { inner: Some(inner) }
            }

            fn try_get_inner(&self) -> PyResult<&FrameStackHandle<$frame_meta_type>> {
                if let Some(inner) = &self.inner {
                    Ok(inner)
                } else {
                    Err(PyConnectionError::new_err("operation on free'd FrameStackHandle".to_owned()))
                }
            }

            pub fn take(&mut self) -> Option<FrameStackHandle<$frame_meta_type>> {
                self.inner.take()
            }
        }

        #[pymethods]
        impl $name_frame_stack {
            fn __len__(&self) -> PyResult<usize> {
                Ok(self.try_get_inner()?.len())
            }

            fn get_dtype_string(&self) -> PyResult<String> {
                Ok(self.try_get_inner()?.first_meta().get_dtype_string())
            }

            fn get_shape(&self) -> PyResult<(u64, u64)> {
                Ok(self.try_get_inner()?.first_meta().get_shape())
            }
        }
    };
}
