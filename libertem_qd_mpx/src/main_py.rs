use std::time::Duration;

use common::{decoder::DecoderTargetPixelType, generic_connection::GenericConnection};
use numpy::{Element, PyArray1, PyArrayMethods, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::{
    exceptions::PyValueError,
    pyclass, pymethods, pymodule,
    types::{PyBytes, PyModule, PyType},
    Bound, PyResult, Python,
};

use common::{impl_py_cam_client, impl_py_connection};

use crate::base_types::{DType, Layout, QdDetectorConnConfig, QdFrameMeta};
use crate::decoder::QdDecoder;
use crate::{background_thread::QdBackgroundThread, base_types::QdAcquisitionHeader};

#[pymodule]
fn libertem_qd_mpx(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<QdConnection>()?;
    m.add_class::<QdFrameStack>()?;
    m.add_class::<CamClient>()?;
    m.add_class::<PyQdDecoder>()?;

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
    config: QdDetectorConnConfig,
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

        let config = QdDetectorConnConfig::new(
            data_host,
            data_port,
            frame_stack_size,
            bytes_per_frame,
            num_slots,
            huge.unwrap_or(false),
            shm_handle_path,
            drain,
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

        Ok(QdConnection { conn, config })
    }

    fn wait_for_arm(
        &mut self,
        timeout: Option<f32>,
        py: Python<'_>,
    ) -> PyResult<Option<QdAcquisitionHeader>> {
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

#[pyclass(name = "QdDecoder")]
struct PyQdDecoder {
    layout: Layout,
    dtype: DType,
    counter_depth: u8,
}

impl PyQdDecoder {
    fn decode_impl<O>(&self, input: Vec<u8>, output: &Bound<'_, PyArray1<O>>) -> PyResult<()>
    where
        O: Element + DecoderTargetPixelType,
    {
        let decoder = QdDecoder::default();
        let (width, height, num_chips) = match &self.layout {
            Layout::L1x1 => (256, 256, 1),
            Layout::L2x2 => (512, 512, 4),
            Layout::LNx1 => todo!(),
            Layout::L2x2G => (514, 514, 4),
            Layout::LNx1G => todo!(),
        };

        let frame_meta = QdFrameMeta::new(
            0,
            1,
            768,
            num_chips,
            width,
            height,
            self.dtype.clone(),
            self.layout.clone(),
            // FIXME: this is a lie, but the value is not used yet in decoding, so we get away with it..
            0xFF,
            "".to_owned(),
            0.0,
            0,
            crate::base_types::ColourMode::Single,
            crate::base_types::Gain::HGM,
            Some(crate::base_types::MQ1A {
                timestamp_ext: "".to_owned(),
                acquisition_time_shutter_ns: 0,
                counter_depth: self.counter_depth,
            }),
        );

        let mut out_arr = output.try_readwrite().unwrap();
        let out_slice = out_arr.as_slice_mut().unwrap();

        decoder
            .decode_frame(&frame_meta, &input, out_slice)
            .unwrap();

        Ok(())
    }
}

#[pymethods]
impl PyQdDecoder {
    #[new]
    fn new(layout: &str, dtype: &str, counter_depth: u8) -> PyResult<Self> {
        Ok(Self {
            layout: layout
                .parse()
                .map_err(|e| PyValueError::new_err(format!("failed to parse layout: {e}")))?,
            dtype: dtype
                .parse()
                .map_err(|e| PyValueError::new_err(format!("failed to parse dtype: {e}")))?,
            counter_depth,
        })
    }

    /// Note: this function is meant for testing only and is not optimized for
    /// performance or good error handling!
    fn decode_to_u64(&self, input: Vec<u8>, output: &Bound<'_, PyArray1<u64>>) -> PyResult<()> {
        eprintln!("decode_to_u64: {} {:?}", input.len(), output.shape());
        self.decode_impl(input, output)
    }

    /// Note: this function is meant for testing only and is not optimized for
    /// performance or good error handling!
    fn decode_to_u8(&self, input: Vec<u8>, output: &Bound<'_, PyArray1<u8>>) -> PyResult<()> {
        self.decode_impl(input, output)
    }

    /// Note: this function is meant for testing only and is not optimized for
    /// performance or good error handling!
    fn decode_to_f32(&self, input: Vec<u8>, output: &Bound<'_, PyArray1<f32>>) -> PyResult<()> {
        self.decode_impl(input, output)
    }

    /// Note: this function is meant for testing only and is not optimized for
    /// performance or good error handling!
    fn decode_to_f64(&self, input: Vec<u8>, output: &Bound<'_, PyArray1<f64>>) -> PyResult<()> {
        self.decode_impl(input, output)
    }
}
