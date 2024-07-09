use common::generic_connection::GenericConnection;
use numpy::PyUntypedArray;
use pyo3::{
    pyclass, pymethods, pymodule,
    types::{PyBytes, PyModule, PyType},
    Bound, PyResult, Python,
};

use common::{impl_py_cam_client, impl_py_connection};

use crate::base_types::{QdDetectorConnConfig, QdFrameMeta};
use crate::decoder::QdDecoder;
use crate::{background_thread::QdBackgroundThread, base_types::QdAcquisitionHeader};

#[pymodule]
fn libertem_qd_mpx(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<QdConnection>()?;
    m.add_class::<QdFrameStack>()?;
    m.add_class::<CamClient>()?;

    let env = env_logger::Env::default()
        .filter_or("LIBERTEM_QD_LOG_LEVEL", "error")
        .write_style_or("LIBERTEM_QD_LOG_STYLE", "always");
    env_logger::Builder::from_env(env)
        .format_timestamp_micros()
        .init();

    Ok(())
}

impl_py_connection!(
    _PyQdConnection,
    _PyQdFrameStack,
    QdFrameMeta,
    QdBackgroundThread,
    QdAcquisitionHeader,
    libertem_qd_mpx
);

#[pyclass]
struct QdConnection {
    conn: _PyQdConnection,
}

#[pymethods]
impl QdConnection {
    #[new]
    fn new(
        data_host: &str,
        data_port: usize,
        frame_stack_size: usize,
        shm_handle_path: &str,
        num_slots: Option<usize>,
        bytes_per_frame: Option<usize>,
        huge: Option<bool>,
    ) -> PyResult<Self> {
        // NOTE: these values don't have to be exact and are mostly important
        // for performance tuning
        let num_slots = num_slots.unwrap_or(2000);
        let bytes_per_frame = bytes_per_frame.unwrap_or(256 * 256 * 2);

        let config = QdDetectorConnConfig::new(
            data_host,
            data_port,
            frame_stack_size,
            bytes_per_frame,
            num_slots,
            huge.unwrap_or(false),
            shm_handle_path,
        );

        let shm =
            GenericConnection::<QdBackgroundThread, QdAcquisitionHeader>::shm_from_config(&config)
                .map_err(|e| PyConnectionError::new_err(e.to_string()))?;

        let bg_thread = QdBackgroundThread::spawn(&config, &shm)
            .map_err(|e| PyConnectionError::new_err(e.to_string()))?;
        let generic_conn =
            GenericConnection::<QdBackgroundThread, QdAcquisitionHeader>::new(bg_thread, &shm)
                .map_err(|e| PyConnectionError::new_err(e.to_string()))?;

        let conn = _PyQdConnection::new(shm, generic_conn);

        Ok(QdConnection { conn })
    }

    fn wait_for_arm(&mut self, timeout: f32) -> PyResult<Option<QdAcquisitionHeader>> {
        self.conn.wait_for_arm(timeout)
    }

    fn get_socket_path(&self) -> PyResult<String> {
        self.conn.get_socket_path()
    }

    fn is_running(&self) -> PyResult<bool> {
        self.conn.is_running()
    }

    fn start_passive(&mut self) -> PyResult<()> {
        self.conn.start_passive()
    }

    fn close(&mut self) -> PyResult<()> {
        self.conn.close()
    }

    fn get_next_stack(
        &mut self,
        max_size: usize,
        py: Python<'_>,
    ) -> PyResult<Option<QdFrameStack>> {
        let stack_inner = self.conn.get_next_stack(max_size, py)?;
        Ok(stack_inner.map(QdFrameStack::new))
    }
}

#[pyclass]
struct QdFrameStack {
    inner: _PyQdFrameStack,
}

#[pymethods]
impl QdFrameStack {
    fn __len__(&self) -> PyResult<usize> {
        self.inner.__len__()
    }

    fn get_dtype_string(&self) -> PyResult<String> {
        self.inner.get_dtype_string()
    }

    fn get_shape(&self) -> PyResult<(u64, u64)> {
        self.inner.get_shape()
    }

    fn serialize<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        self.inner.serialize(py)
    }

    #[classmethod]
    fn deserialize<'py>(
        _cls: Bound<'py, PyType>,
        serialized: Bound<'py, PyBytes>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: _PyQdFrameStack::deserialize_impl(serialized)?,
        })
    }
}

impl QdFrameStack {
    pub fn new(inner: _PyQdFrameStack) -> Self {
        Self { inner }
    }

    fn get_inner(&self) -> &_PyQdFrameStack {
        &self.inner
    }

    fn get_inner_mut(&mut self) -> &mut _PyQdFrameStack {
        &mut self.inner
    }
}

impl_py_cam_client!(
    _PyQdCamClient,
    QdDecoder,
    _PyQdFrameStack,
    QdFrameMeta,
    libertem_qd_mpx
);

#[pyclass]
struct CamClient {
    inner: _PyQdCamClient,
}

#[pymethods]
impl CamClient {
    #[new]
    fn new(handle_path: &str) -> PyResult<Self> {
        Ok(Self {
            inner: _PyQdCamClient::new(handle_path)?,
        })
    }

    fn decode_range_into_buffer<'py>(
        &self,
        input: &QdFrameStack,
        out: &Bound<'py, PyUntypedArray>,
        start_idx: usize,
        end_idx: usize,
        py: Python<'py>,
    ) -> PyResult<()> {
        self.inner
            .decode_range_into_buffer(input.get_inner(), out, start_idx, end_idx, py)
    }

    fn done(&mut self, handle: &mut QdFrameStack) -> PyResult<()> {
        self.inner.frame_stack_done(handle.get_inner_mut())
    }

    fn close(&mut self) -> PyResult<()> {
        self.inner.close()
    }
}
