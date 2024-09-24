use common::background_thread::PyAcquisitionSize;
use common::generic_connection::GenericConnection;
use common::tracing::{span_from_py, tracing_from_env};
use pyo3::{pyclass, pymethods, Python};
use pyo3::{pymodule, types::PyModule, Bound, PyResult};

use common::{impl_py_cam_client, impl_py_connection};

use crate::background_thread::K2BackgroundThread;
use crate::config::{K2AcquisitionConfig, K2DetectorConnectionConfig, K2Mode};
use crate::decoder::K2Decoder;
use crate::frame_meta::K2FrameMeta;

#[pymodule]
fn libertem_k2is(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let env = env_logger::Env::default()
        .filter_or("LIBERTEM_K2IS_LOG_LEVEL", "error")
        .write_style_or("LIBERTEM_K2IS_LOG_STYLE", "always");
    env_logger::Builder::from_env(env)
        .format_timestamp_micros()
        .init();

    tracing_from_env("libertem-k2is".to_owned());

    m.add_class::<K2Connection>()?;
    m.add_class::<K2FrameStack>()?;
    m.add_class::<K2Mode>()?;
    // m.add_class::<CamClient>()?; FIXME
    m.add_class::<_PyK2CamClient>()?;
    m.add_class::<K2AcquisitionConfig>()?;
    m.add_class::<PyAcquisitionSize>()?;

    Ok(())
}

impl_py_connection!(
    _PyK2Connection,
    K2FrameStack,
    K2FrameMeta,
    K2BackgroundThread,
    K2AcquisitionConfig,
    libertem_k2is
);

#[pyclass]
struct K2Connection {
    conn: _PyK2Connection,
}

#[pymethods]
impl K2Connection {
    #[new]
    fn new(
        local_addr_top: &str,
        local_addr_bottom: &str,
        frame_stack_size: usize,
        shm_handle_path: &str,
        mode: Option<K2Mode>,
        huge: Option<bool>,
        crop_to_image_data: Option<bool>,
        py: Python,
    ) -> PyResult<Self> {
        let _trace_guard = span_from_py(py, "K2Connection::new")?;

        // to have some slack, we need some more memory in IS mode:
        let mode = mode.unwrap_or(K2Mode::IS);

        let num_slots = match mode {
            K2Mode::IS => 1600,    // about 2 seconds of buffering
            K2Mode::Summit => 100, // TODO: update with realistic value here
        };

        let crop_to_image_data = crop_to_image_data.unwrap_or(true);

        let config = K2DetectorConnectionConfig::new(
            mode,
            local_addr_top.to_owned(),
            local_addr_bottom.to_owned(),
            num_slots,
            huge.unwrap_or(false),
            shm_handle_path.to_owned(),
            1, // FIXME: pass down `frame_stack_size` once a value != 1 is supported
            false,
            false,
            crop_to_image_data,
        );

        let shm =
            GenericConnection::<K2BackgroundThread, K2AcquisitionConfig>::shm_from_config(&config)
                .map_err(|e| PyConnectionError::new_err(e.to_string()))?;

        let bg_thread = K2BackgroundThread::spawn(&config, &shm)
            .map_err(|e| PyConnectionError::new_err(e.to_string()))?;

        let generic_conn =
            GenericConnection::<K2BackgroundThread, K2AcquisitionConfig>::new(bg_thread, &shm)
                .map_err(|e| PyConnectionError::new_err(e.to_string()))?;

        let conn = _PyK2Connection::new(shm, generic_conn);
        Ok(Self { conn })
    }

    fn wait_for_arm(
        &mut self,
        timeout: Option<f32>,
        py: Python<'_>,
    ) -> PyResult<Option<K2AcquisitionConfig>> {
        self.conn.wait_for_arm(timeout, py)
    }

    fn get_socket_path(&self) -> PyResult<String> {
        self.conn.get_socket_path()
    }

    fn is_running(&self) -> PyResult<bool> {
        self.conn.is_running()
    }

    fn start_passive(
        &mut self,
        timeout: Option<f32>,
        acquisition_size: Option<PyAcquisitionSize>,
        py: Python<'_>,
    ) -> PyResult<()> {
        let _trace_guard = span_from_py(py, "K2Connection::start_passive")?;

        self.conn.start_passive(timeout, acquisition_size, py)
    }

    fn passive_is_running(&self) -> PyResult<bool> {
        self.conn.passive_is_running()
    }

    fn close(&mut self, py: Python) -> PyResult<()> {
        self.conn.close(py)
    }

    fn cancel(&mut self, timeout: Option<f32>, py: Python) -> PyResult<()> {
        self.conn.cancel(timeout, py)
    }

    fn get_next_stack(
        &mut self,
        max_size: usize,
        py: Python<'_>,
    ) -> PyResult<Option<K2FrameStack>> {
        self.conn.get_next_stack(max_size, py)
    }
}

impl_py_cam_client!(
    _PyK2CamClient,
    K2Decoder,
    K2FrameStack,
    K2FrameMeta,
    libertem_k2is
);
