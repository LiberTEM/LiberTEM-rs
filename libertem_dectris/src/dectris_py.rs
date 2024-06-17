//! This module should be a wrapper around the generic functionality of the `common`
//! crate, and only contain logic that is specific to the detector. The interface
//! exported to Python should be as uniform as possible compared to other detectors,
//! pending future unification with full compatability between detectors.
use std::time::Duration;

use common::{
    background_thread::BackgroundThread,
    frame_stack::FrameMeta,
    generic_connection::{ConnectionStatus, GenericConnection},
};

use crate::{
    background_thread::{DectrisBackgroundThread, DectrisDetectorConnConfig, DectrisExtraControl},
    cam_client::CamClient,
    common::{
        DConfig, DHeader, DImage, DImageD, DSeriesEnd, DectrisFrameMeta, DectrisPendingAcquisition,
        DetectorConfig, PixelType, TriggerMode,
    },
    exceptions::{DecompressError, TimeoutError},
    sim::DectrisSim,
};

use common::{frame_stack::FrameStackHandle, impl_py_connection};
use ipc_test::SharedSlabAllocator;
use pyo3::{create_exception, exceptions::PyException, prelude::*};
use stats::Stats;

#[pymodule]
fn libertem_dectris(py: Python, m: Bound<'_, PyModule>) -> PyResult<()> {
    // FIXME: logging integration deadlocks on close(), when trying to acquire
    // the GIL
    // pyo3_log::init();

    m.add_class::<DectrisFrameStack>()?;
    m.add_class::<DectrisConnection>()?;
    m.add_class::<PixelType>()?;
    m.add_class::<DectrisSim>()?;
    m.add_class::<DetectorConfig>()?;
    m.add_class::<TriggerMode>()?;
    m.add_class::<CamClient>()?;
    m.add("TimeoutError", py.get_type_bound::<TimeoutError>())?;
    m.add("DecompressError", py.get_type_bound::<DecompressError>())?;

    register_header_module(py, &m)?;

    let env = env_logger::Env::default()
        .filter_or("LIBERTEM_DECTRIS_LOG_LEVEL", "error")
        .write_style_or("LIBERTEM_DECTRIS_LOG_STYLE", "always");
    env_logger::init_from_env(env);

    Ok(())
}

fn register_header_module(py: Python<'_>, parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let headers_module = PyModule::new_bound(py, "headers")?;
    headers_module.add_class::<DHeader>()?;
    headers_module.add_class::<DImage>()?;
    headers_module.add_class::<DImageD>()?;
    headers_module.add_class::<DConfig>()?;
    headers_module.add_class::<DSeriesEnd>()?;
    parent_module.add_submodule(&headers_module)?;
    Ok(())
}

impl_py_connection!(
    _PyDectrisConnection,
    _PyDectrisFrameStack,
    DectrisFrameMeta,
    DectrisBackgroundThread,
    DectrisPendingAcquisition,
    libertem_dectris
);

#[pyclass]
struct DectrisConnection {
    conn: _PyDectrisConnection,
}

#[pymethods]
impl DectrisConnection {
    #[new]
    fn new(
        uri: &str,
        frame_stack_size: usize,
        handle_path: &str,
        num_slots: Option<usize>,
        bytes_per_frame: Option<usize>,
        huge: Option<bool>,
    ) -> PyResult<Self> {
        let num_slots = num_slots.map_or_else(|| 2000, |x| x);
        let bytes_per_frame = bytes_per_frame.map_or_else(|| 512 * 512 * 2, |x| x);
        let config = DectrisDetectorConnConfig::new(
            uri,
            frame_stack_size,
            bytes_per_frame,
            num_slots,
            huge.map_or_else(|| false, |x| x),
            handle_path,
        );

        let shm = GenericConnection::<DectrisBackgroundThread, DectrisPendingAcquisition>::shm_from_config(&config).map_err(|e| {
            PyConnectionError::new_err(format!("could not init shm: {}", e))
        })?;
        let bg_thread = DectrisBackgroundThread::spawn(&config, &shm)
            .map_err(|e| PyConnectionError::new_err(e.to_string()))?;
        let generic_conn =
            GenericConnection::<DectrisBackgroundThread, DectrisPendingAcquisition>::new(
                bg_thread, &shm,
            )
            .map_err(|e| PyConnectionError::new_err(e.to_string()))?;

        let conn = _PyDectrisConnection::new(shm, generic_conn);

        Ok(Self { conn })
    }

    /// Wait until the detector is armed, or until the timeout expires (in seconds)
    /// Returns `None` in case of timeout, the detector config otherwise.
    /// This method drops the GIL to allow concurrent Python threads.
    fn wait_for_arm(&mut self, timeout: f32) -> PyResult<Option<(DetectorConfig, u64)>> {
        let res = self.conn.wait_for_arm(timeout)?;
        Ok(res.map(|config| (config.get_detector_config(), config.get_series())))
    }

    fn get_socket_path(&self) -> PyResult<String> {
        self.conn.get_socket_path()
    }

    fn is_running(&self) -> PyResult<bool> {
        self.conn.is_running()
    }

    fn start(&mut self, series: u64) -> PyResult<()> {
        self.conn
            .send_specialized(DectrisExtraControl::StartAcquisitionWithSeries { series })?;
        self.conn
            .wait_for_status(ConnectionStatus::Armed, Duration::from_millis(100))?;
        Ok(())
    }

    /// Start listening for global acquisition headers on the zeromq socket.
    fn start_passive(&mut self) -> PyResult<()> {
        self.conn.start_passive()
    }

    fn close(&mut self) -> PyResult<()> {
        self.conn.close()?;
        Ok(())
    }

    fn log_shm_stats(&self) -> PyResult<()> {
        self.conn.log_shm_stats()
    }

    fn get_next_stack(
        &mut self,
        max_size: usize,
        py: Python<'_>,
    ) -> PyResult<Option<DectrisFrameStack>> {
        let stack_inner = self.conn.get_next_stack(max_size, py)?;
        Ok(stack_inner.map(DectrisFrameStack::new))
    }
}

#[pyclass(name = "FrameStackHandle")]
pub struct DectrisFrameStack {
    inner: _PyDectrisFrameStack,
}

impl DectrisFrameStack {
    pub fn new(inner: _PyDectrisFrameStack) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl DectrisFrameStack {
    fn __len__(&self) -> PyResult<usize> {
        self.inner.__len__()
    }

    fn get_dtype_string(&self) -> PyResult<String> {
        self.inner.get_dtype_string()
    }

    fn get_shape(&self) -> PyResult<(u64, u64)> {
        self.inner.get_shape()
    }

    // fn get_series_id
    // fn get_frame_id
    // fn get_hash
    // anything else?
}

#[pyclass(name = "CamClient")]
pub struct DectrisCamClient {}
