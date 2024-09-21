#![allow(clippy::borrow_deref_ref)]

use crate::{
    base_types::{
        ASIMpxDetectorConnConfig, ASIMpxFrameMeta, DType, PendingAcquisition, PyDetectorConfig,
    },
    exceptions::{ConnectionError, TimeoutError},
};

use common::{
    generic_connection::GenericConnection, impl_py_cam_client, impl_py_connection,
    tracing::span_from_py,
};

use log::trace;
use numpy::PyUntypedArray;
use pyo3::{
    exceptions::PyRuntimeError,
    prelude::*,
    types::{PyBytes, PyType},
};
use serval_client::{DetectorInfo, DetectorLayout, ServalClient};

use crate::background_thread::ASIMpxBackgroundThread;
use crate::decoder::ASIMpxDecoder;

use std::ffi::c_int;

use pyo3::ffi::PyMemoryView_FromMemory;

#[pymodule]
fn libertem_asi_mpx3(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // FIXME: logging integration deadlocks on close(), when trying to acquire
    // the GIL
    // pyo3_log::init();

    m.add_class::<PyFrameStackHandle>()?;
    m.add_class::<ServalConnection>()?;
    m.add_class::<DType>()?;
    m.add_class::<PyDetectorConfig>()?;
    m.add_class::<PyDetectorInfo>()?;
    m.add_class::<PyServalClient>()?;
    m.add_class::<CamClient>()?;
    m.add("TimeoutError", py.get_type_bound::<TimeoutError>())?;

    register_header_module(py, m)?;

    let env = env_logger::Env::default()
        .filter_or("LIBERTEM_ASI_LOG_LEVEL", "error")
        .write_style_or("LIBERTEM_ASI_LOG_STYLE", "always");
    env_logger::Builder::from_env(env)
        .format_timestamp_micros()
        .init();

    Ok(())
}

fn register_header_module(py: Python<'_>, parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let headers_module = PyModule::new_bound(py, "headers")?;
    parent_module.add_submodule(&headers_module)?;
    Ok(())
}

#[derive(Debug)]
#[pyclass(name = "DetectorInfo")]
struct PyDetectorInfo {
    info: DetectorInfo,
}

#[pymethods]
impl PyDetectorInfo {
    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }

    fn get_pix_count(&self) -> u64 {
        self.info.pix_count
    }
}

#[derive(Debug)]
#[pyclass(name = "DetectorLayout")]
struct PyDetectorLayout {
    info: DetectorLayout,
}

#[pymethods]
impl PyDetectorLayout {
    fn __repr__(&self) -> String {
        format!("{:?}", self.info)
    }
}

#[pyclass(name = "ServalAPIClient")]
struct PyServalClient {
    client: ServalClient,
    base_url: String,
}

#[pymethods]
impl PyServalClient {
    #[new]
    fn new(base_url: &str) -> Self {
        Self {
            client: ServalClient::new(base_url),
            base_url: base_url.to_string(),
        }
    }

    fn __repr__(&self) -> String {
        format!("<ServalClient base_url={}>", self.base_url)
    }

    fn get_detector_config(&self) -> PyResult<PyDetectorConfig> {
        self.client
            .get_detector_config()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
            .map(PyDetectorConfig::new)
    }

    fn get_detector_info(&self) -> PyResult<PyDetectorInfo> {
        self.client
            .get_detector_info()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
            .map(|value| PyDetectorInfo { info: value })
    }
}

#[pyclass]
struct ServalConnection {
    conn: _PyASIMpxConnection,
}

#[pymethods]
impl ServalConnection {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        data_uri: &str,
        api_uri: &str,
        frame_stack_size: usize,
        handle_path: String,
        num_slots: Option<usize>,
        bytes_per_frame: Option<usize>,
        huge: Option<bool>,
        py: Python,
    ) -> PyResult<Self> {
        let _trace_guard = span_from_py(py, "ServalConnection::new")?;

        let num_slots = num_slots.map_or_else(|| 2000, |x| x);
        let bytes_per_frame = bytes_per_frame.map_or_else(|| 512 * 512 * 2, |x| x);

        let config = ASIMpxDetectorConnConfig::new(
            data_uri,
            api_uri,
            frame_stack_size,
            bytes_per_frame,
            num_slots,
            huge.unwrap_or(false),
            &handle_path,
        );

        let shm = GenericConnection::<ASIMpxBackgroundThread, PendingAcquisition>::shm_from_config(
            &config,
        )
        .map_err(|e| PyConnectionError::new_err(format!("could not init shm: {e}")))?;

        let bg_thread = ASIMpxBackgroundThread::spawn(&config, &shm)
            .map_err(|e| ConnectionError::new_err(e.to_string()))?;

        let generic_conn =
            GenericConnection::<ASIMpxBackgroundThread, PendingAcquisition>::new(bg_thread, &shm)
                .map_err(|e| PyConnectionError::new_err(e.to_string()))?;

        let conn = _PyASIMpxConnection::new(shm, generic_conn);

        Ok(Self { conn })
    }

    fn wait_for_arm(
        &mut self,
        timeout: Option<f32>,
        py: Python<'_>,
    ) -> PyResult<Option<PendingAcquisition>> {
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

    fn close(&mut self, py: Python) -> PyResult<()> {
        self.conn.close(py)
    }

    fn get_next_stack(
        &mut self,
        max_size: usize,
        py: Python,
    ) -> PyResult<Option<PyFrameStackHandle>> {
        let stack_inner = self.conn.get_next_stack(max_size, py)?;
        Ok(stack_inner.map(PyFrameStackHandle::new))
    }
}

impl_py_connection!(
    _PyASIMpxConnection,
    _PyASIMpxFrameStack,
    ASIMpxFrameMeta,
    ASIMpxBackgroundThread,
    PendingAcquisition,
    libertem_asi_mpx3
);

impl_py_cam_client!(
    _PyASIMpxCamClient,
    ASIMpxDecoder,
    _PyASIMpxFrameStack,
    ASIMpxFrameMeta,
    libertem_asi_mpx3
);

#[pyclass(name = "FrameStackHandle")]
pub struct PyFrameStackHandle {
    inner: _PyASIMpxFrameStack,
}

#[pymethods]
impl PyFrameStackHandle {
    fn serialize<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        self.inner.serialize(py)
    }

    #[classmethod]
    fn deserialize<'py>(
        _cls: Bound<'py, PyType>,
        serialized: Bound<'py, PyBytes>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: _PyASIMpxFrameStack::deserialize_impl(serialized)?,
        })
    }

    fn get_frame_id(&self) -> PyResult<u64> {
        Ok(self.inner.try_get_inner()?.first_meta().sequence)
    }

    fn get_shape(&self) -> PyResult<(u64, u64)> {
        self.inner.get_shape()
    }

    fn __len__(&self) -> PyResult<usize> {
        self.inner.__len__()
    }
}

impl PyFrameStackHandle {
    fn new(inner: _PyASIMpxFrameStack) -> Self {
        Self { inner }
    }

    fn get_inner(&self) -> &_PyASIMpxFrameStack {
        &self.inner
    }

    fn get_inner_mut(&mut self) -> &mut _PyASIMpxFrameStack {
        &mut self.inner
    }
}

#[allow(non_upper_case_globals)]
const PyBUF_READ: c_int = 0x100;

#[pyclass]
pub struct CamClient {
    inner: _PyASIMpxCamClient,
}

impl CamClient {
    fn get_memoryview(&self, py: Python, raw_data: &[u8]) -> PyObject {
        let ptr = raw_data.as_ptr();
        let length = raw_data.len();

        let mv = unsafe {
            PyMemoryView_FromMemory(ptr as *mut i8, length.try_into().unwrap(), PyBUF_READ)
        };

        let from_ptr: Bound<PyAny> = unsafe {
            //FromPyPointer::from_owned_ptr(py, mv)

            Bound::from_owned_ptr(py, mv)
        };
        from_ptr.into_py(py)
    }
}

#[pymethods]
impl CamClient {
    #[new]
    fn new(py: Python, handle_path: &str) -> PyResult<Self> {
        Ok(Self {
            inner: _PyASIMpxCamClient::new(py, handle_path)?,
        })
    }

    // FIXME: add safe replacement for this function
    // (probably needs Python 3.11+)
    #[deprecated]
    fn get_frames(
        &self,
        handle: &PyFrameStackHandle,
        py: Python,
    ) -> PyResult<Vec<(PyObject, DType)>> {
        let shm = self.inner.get_shm()?;

        // safety: Python code must not use memoryviews after calling `done`.
        let slot = unsafe { shm.get(handle.get_inner().try_get_inner()?.get_slot().slot_idx) };

        let meta = handle.get_inner().get_meta()?;
        let inner = handle.get_inner().try_get_inner()?;

        Ok(meta
            .iter()
            .zip(0..)
            .map(|(meta, frame_idx)| {
                let image_data = inner.get_slice_for_frame(frame_idx, &slot);
                let memory_view = self.get_memoryview(py, image_data);

                (memory_view, meta.dtype.clone())
            })
            .collect())
    }

    fn decode_range_into_buffer<'py>(
        &self,
        input: &PyFrameStackHandle,
        out: &Bound<'py, PyUntypedArray>,
        start_idx: usize,
        end_idx: usize,
        py: Python<'py>,
    ) -> PyResult<()> {
        self.inner
            .decode_range_into_buffer(input.get_inner(), out, start_idx, end_idx, py)
    }

    fn done(&mut self, handle: &mut PyFrameStackHandle) -> PyResult<()> {
        self.inner.frame_stack_done(handle.get_inner_mut())
    }

    fn close(&mut self) -> PyResult<()> {
        self.inner.close()
    }
}

impl Drop for CamClient {
    fn drop(&mut self) {
        trace!("CamClient::drop");
    }
}
