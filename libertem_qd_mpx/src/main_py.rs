use std::{str::FromStr, time::Duration};

use common::generic_connection::GenericConnection;
use numpy::PyUntypedArray;
use pyo3::{
    exceptions::PyValueError, pyclass, pymethods, pymodule, types::PyModule, Bound, PyResult,
    Python,
};

use common::{impl_py_cam_client, impl_py_connection};

use crate::base_types::{QdAcquisitionConfig, QdDetectorConnConfig, QdFrameMeta, RecoveryStrategy};
use crate::decoder::QdDecoder;
use crate::{background_thread::QdBackgroundThread, base_types::QdAcquisitionHeader};

#[pymodule]
fn libertem_qd_mpx(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<QdConnection>()?;
    m.add_class::<QdFrameStack>()?;
    m.add_class::<CamClient>()?;
    m.add_class::<QdAcquisitionHeader>()?;
    m.add_class::<QdAcquisitionConfig>()?;

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
    QdFrameStack,
    QdFrameMeta,
    QdBackgroundThread,
    QdAcquisitionConfig,
    libertem_qd_mpx
);

#[pyclass]
struct QdConnection {
    conn: _PyQdConnection,
}

#[pymethods]
impl QdConnection {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        data_host: &str,
        data_port: usize,
        frame_stack_size: usize,
        shm_handle_path: &str,
        drain: Option<bool>,
        num_slots: Option<usize>,
        bytes_per_frame: Option<usize>,
        huge: Option<bool>,
        recovery_strategy: Option<&str>,
    ) -> PyResult<Self> {
        // NOTE: these values don't have to be exact and are mostly important
        // for performance tuning
        let num_slots = num_slots.unwrap_or(2000);
        let bytes_per_frame = bytes_per_frame.unwrap_or(256 * 256 * 2);

        let drain = if drain.unwrap_or(false) {
            Some(Duration::from_millis(100))
        } else {
            None
        };

        let recovery_strategy = recovery_strategy
            .map_or_else(
                || Ok(RecoveryStrategy::default()),
                RecoveryStrategy::from_str,
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let config = QdDetectorConnConfig::new(
            data_host,
            data_port,
            frame_stack_size,
            bytes_per_frame,
            num_slots,
            huge.unwrap_or(false),
            shm_handle_path,
            drain,
            recovery_strategy,
        );

        let shm =
            GenericConnection::<QdBackgroundThread, QdAcquisitionConfig>::shm_from_config(&config)
                .map_err(|e| PyConnectionError::new_err(e.to_string()))?;

        let bg_thread = QdBackgroundThread::spawn(&config, &shm)
            .map_err(|e| PyConnectionError::new_err(e.to_string()))?;
        let generic_conn =
            GenericConnection::<QdBackgroundThread, QdAcquisitionConfig>::new(bg_thread, &shm)
                .map_err(|e| PyConnectionError::new_err(e.to_string()))?;

        let conn = _PyQdConnection::new(shm, generic_conn);

        Ok(QdConnection { conn })
    }

    fn wait_for_arm(
        &mut self,
        timeout: Option<f32>,
        py: Python<'_>,
    ) -> PyResult<Option<QdAcquisitionConfig>> {
        self.conn.wait_for_arm(timeout, py)
    }

    fn get_socket_path(&self) -> PyResult<String> {
        self.conn.get_socket_path()
    }

    fn is_running(&self) -> PyResult<bool> {
        self.conn.is_running()
    }

    fn start_passive(&mut self, timeout: Option<f32>, py: Python<'_>) -> PyResult<()> {
        self.conn.start_passive(timeout, py)
    }

    fn close(&mut self) -> PyResult<()> {
        self.conn.close()
    }

    fn get_next_stack(
        &mut self,
        max_size: usize,
        py: Python<'_>,
    ) -> PyResult<Option<QdFrameStack>> {
        self.conn.get_next_stack(max_size, py)
    }
}

impl_py_cam_client!(
    _PyQdCamClient,
    QdDecoder,
    QdFrameStack,
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
            .decode_range_into_buffer(input, out, start_idx, end_idx, py)
    }

    fn done(&mut self, handle: &mut QdFrameStack) -> PyResult<()> {
        self.inner.frame_stack_done(handle)
    }

    fn close(&mut self) -> PyResult<()> {
        self.inner.close()
    }
}
