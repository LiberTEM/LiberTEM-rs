mod bgthread;
mod shm_helpers;

use env_logger::Builder;
use ipc_test::SharedSlabAllocator;
use shm_helpers::{CamClient, FrameRef};
use std::{
    panic,
    path::Path,
    sync::{Arc, Barrier},
    time::{Duration, Instant},
};
use tokio::runtime::Runtime;

use bgthread::{AcquisitionRuntime, AddrConfig, RuntimeError};

use k2o::{
    acquisition::AcquisitionResult,
    block::K2Block,
    block_is::K2ISBlock,
    events::{AcquisitionParams, AcquisitionSync, WriterSettings, WriterTypeError},
    frame::{GenericFrame, K2Frame},
    frame_is::K2ISFrame,
    frame_summit::K2SummitFrame,
    params::CameraMode,
    tracing::{get_tracer, init_tracer},
    write::{DirectWriterBuilder, MMapWriterBuilder, WriterBuilder},
};

#[cfg(feature = "hdf5")]
use k2o::write::HDF5WriterBuilder;

use opentelemetry::{
    trace::{self, SpanContext, SpanId, TraceContextExt, TraceFlags, TraceId, TraceState, Tracer},
    Context, ContextGuard,
};
use pyo3::{exceptions, prelude::*};

fn tracing_thread() {
    let thread_builder = std::thread::Builder::new();

    // for waiting until tracing is initialized:
    let barrier = Arc::new(Barrier::new(2));
    let barrier_bg = Arc::clone(&barrier);

    thread_builder
        .name("tracing".to_string())
        .spawn(move || {
            let rt = Runtime::new().unwrap();

            rt.block_on(async {
                init_tracer().unwrap();
                barrier_bg.wait();

                // do we need to keep this thread alive like this? I think so!
                // otherwise we get:
                // OpenTelemetry trace error occurred. cannot send span to the batch span processor because the channel is closed
                loop {
                    tokio::time::sleep(Duration::from_secs(10)).await;
                }
            });
        })
        .unwrap();

    barrier.wait();
}

fn get_py_span_context(py: Python) -> PyResult<SpanContext> {
    let extract_span_context = PyModule::from_code(
        py,
        "
from opentelemetry import trace
span = trace.get_current_span()
span_context = span.get_span_context()",
        "",
        "",
    )?;

    let span_context_py = extract_span_context.getattr("span_context")?;

    let trace_id_py: u128 = span_context_py.getattr("trace_id")?.extract()?;
    let span_id_py: u64 = span_context_py.getattr("span_id")?.extract()?;
    let trace_flags_py: u8 = span_context_py.getattr("trace_flags")?.extract()?;

    let trace_id = TraceId::from_bytes(trace_id_py.to_be_bytes());
    let span_id = SpanId::from_bytes(span_id_py.to_be_bytes());
    let trace_flags = TraceFlags::new(trace_flags_py);

    // FIXME: Python has a list of something here, wtf is that & do we need it?
    let trace_state = TraceState::default();

    let span_context = SpanContext::new(trace_id, span_id, trace_flags, false, trace_state);

    Ok(span_context)
}

fn get_tracing_context(py: Python) -> PyResult<Context> {
    let span_context = get_py_span_context(py)?;
    let context = Context::default().with_remote_span_context(span_context);

    Ok(context)
}

fn span_from_py(py: Python, name: &str) -> PyResult<ContextGuard> {
    let tracer = get_tracer();
    let context = get_tracing_context(py)?;
    let span = tracer.start_with_context(name.to_string(), &context);
    Ok(trace::mark_span_as_active(span))
}

#[pymodule]
fn k2opy(_py: Python, m: &PyModule) -> PyResult<()> {
    // FIXME: add an atexit handler calling `global::shutdown_tracer_provider`
    // so we don't lose any spans at shutdown
    tracing_thread();

    m.add_class::<Acquisition>()?;
    m.add_class::<Cam>()?;
    m.add_class::<SyncFlags>()?;
    m.add_class::<PyMode>()?;
    m.add_class::<PyAcquisitionParams>()?;
    m.add_class::<PyWriter>()?;

    m.add_class::<CamClient>()?;
    m.add_class::<FrameRef>()?;

    let env = env_logger::Env::default()
        .filter_or("LIBERTEM_K2IS_LOG_LEVEL", "error")
        .write_style_or("LIBERTEM_K2IS_LOG_STYLE", "always");
    Builder::new()
        .parse_env(env)
        .format_timestamp_micros()
        .init();

    Ok(())
}

#[pyclass(name = "Sync")]
#[derive(Clone, Debug, PartialEq, Eq)]
enum SyncFlags {
    WaitForSync,
    Immediately,
}

impl From<SyncFlags> for AcquisitionSync {
    fn from(sf: SyncFlags) -> Self {
        match sf {
            SyncFlags::WaitForSync => AcquisitionSync::WaitForSync,
            SyncFlags::Immediately => AcquisitionSync::Immediately,
        }
    }
}

#[pyclass(name = "Mode")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
enum PyMode {
    #[default]
    IS,
    Summit,
}

impl From<CameraMode> for PyMode {
    fn from(value: CameraMode) -> Self {
        match value {
            CameraMode::IS => PyMode::IS,
            CameraMode::Summit => PyMode::Summit,
        }
    }
}

impl Into<CameraMode> for PyMode {
    fn into(self) -> CameraMode {
        match self {
            PyMode::IS => CameraMode::IS,
            PyMode::Summit => CameraMode::Summit,
        }
    }
}

#[pyclass(name = "Writer")]
#[derive(Debug, Clone)]
struct PyWriter {
    pub settings: WriterSettings,
}

impl PyWriter {
    pub fn get_setttings(&self) -> &WriterSettings {
        &self.settings
    }
}

#[pymethods]
impl PyWriter {
    #[new]
    fn new(filename: &str, method: &str) -> PyResult<Self> {
        let settings = match WriterSettings::new(method, filename) {
            Ok(s) => s,
            Err(WriterTypeError::InvalidWriterType) => {
                let msg = format!(
                    "unknown method {method}, choose one: mmap, direct, hdf5 (optional feature)"
                );
                return Err(exceptions::PyValueError::new_err(msg));
            }
        };

        Ok(PyWriter { settings })
    }
}

#[pyclass(name = "AcquisitionParams")]
#[derive(Debug, Clone)]
struct PyAcquisitionParams {
    pub size: Option<u32>,
    pub sync: SyncFlags,
    pub writer_settings: WriterSettings,
}

#[pymethods]
impl PyAcquisitionParams {
    #[new]
    fn new(size: Option<u32>, sync: SyncFlags, writer: Option<PyWriter>) -> Self {
        let writer_settings = match writer {
            None => WriterSettings::disabled(),
            Some(py_writer) => py_writer.get_setttings().clone(),
        };
        PyAcquisitionParams {
            size,
            sync,
            writer_settings,
        }
    }
}

#[pyclass(name = "Frame")]
struct PyFrame {
    acquisition_result: Option<AcquisitionResult<GenericFrame>>,
    frame_idx: u32,
}

impl PyFrame {
    fn new(result: AcquisitionResult<GenericFrame>, frame_idx: u32) -> Self {
        PyFrame {
            acquisition_result: Some(result),
            frame_idx,
        }
    }

    fn consume_frame_data(&mut self) -> AcquisitionResult<GenericFrame> {
        self.acquisition_result.take().unwrap()
    }
}

#[pymethods]
impl PyFrame {
    fn is_dropped(&self) -> bool {
        matches!(
            &self.acquisition_result,
            Some(AcquisitionResult::DroppedFrame(..))
                | Some(AcquisitionResult::DroppedFrameOutside(..))
        )
    }

    fn get_idx(&self) -> u32 {
        self.frame_idx
    }
}

///
/// An `Acquisition` is an object to perform a single acquisition, that is,
/// acquire a potentially unlimited number of frames and iterate over them or
/// write them to disk.
#[pyclass]
struct Acquisition {
    params: PyAcquisitionParams,
    camera_mode: CameraMode,
    id: usize,
}

impl Acquisition {
    pub fn new(params: PyAcquisitionParams, camera_mode: CameraMode, id: usize) -> Self {
        Acquisition {
            params,
            camera_mode,
            id,
        }
    }
}

impl From<RuntimeError> for PyErr {
    fn from(val: RuntimeError) -> Self {
        let msg = format!("runtime error: {val:?}");
        exceptions::PyRuntimeError::new_err(msg)
    }
}

#[pymethods]
impl Acquisition {
    fn get_id(&self) -> usize {
        self.id
    }

    fn __repr__(&self) -> String {
        let id = self.id;
        format!("<Acquisition id='{id}'>")
    }
}

#[pyclass]
struct Cam {
    addr_config: AddrConfig,
    camera_mode: PyMode,
    shm_path: String,
    enable_frame_iterator: bool,
    shm: Option<SharedSlabAllocator>,
    runtime: Option<AcquisitionRuntime>,
}

#[pymethods]
impl Cam {
    #[new]
    fn new(
        local_addr_top: &str,
        local_addr_bottom: &str,
        mode: PyMode,
        shm_path: &str,
        enable_frame_iterator: bool,
        py: Python,
    ) -> PyResult<Self> {
        let _guard = span_from_py(py, "Cam::new")?;

        let path = Path::new(&shm_path);
        let (num_slots, slot_size) = match mode {
            PyMode::IS => (2000, 2048 * 1860 * 2),
            PyMode::Summit => (500, 4096 * 3840 * 2),
        };
        let tracer = get_tracer();
        let shm = tracer.in_span("Cam shm_setup", |_cx| {
            SharedSlabAllocator::new(num_slots, slot_size, true, path).expect("create shm")
        });
        let addr_config = AddrConfig::new(local_addr_top, local_addr_bottom);

        let runtime = AcquisitionRuntime::new(
            &addr_config,
            enable_frame_iterator,
            shm.get_handle(),
            mode.into(),
        );

        Ok(Cam {
            addr_config,
            camera_mode: mode,
            shm_path: shm_path.to_owned(),
            enable_frame_iterator,
            shm: Some(shm),
            runtime: Some(runtime),
        })
    }

    fn wait_for_start(&mut self, py: Python) -> PyResult<()> {
        // TODO: total deadline for initialization?
        // TODO: currently the API can be used wrong, i.e. calling
        // `get_next_frame` before this function means it will throw away
        // perfectly fine frames that actually belong to the beginning of the
        // acquisition. This can be prevented: instead, the user should get back
        // an object from this function which is the actual frame iterator. Or
        // we should make the wait implicit and integrate it into
        // `get_next_frame`... hmm.
        let _guard = span_from_py(py, "Cam::wait_for_start")?;

        if let Some(runtime) = &mut self.runtime {
            loop {
                if runtime.wait_for_start(Duration::from_millis(100)).is_some() {
                    break;
                }
                py.check_signals()?;
            }
        } else {
            return Err(exceptions::PyRuntimeError::new_err(
                "acquisition is not running",
            ));
        }

        Ok(())
    }

    ///
    /// Wait for the current acquisition to complete. This is only needed
    /// if we are only writing to a file, and not consuming the
    /// frame iterator, which takes care of this synchronization
    /// otherwise.
    ///
    /// Also succeeds if the runtime is already shut down.
    ///
    // FIXME: timeout?
    fn wait_until_complete(&mut self, py: Python) -> PyResult<()> {
        let _guard = span_from_py(py, "Acquisition::wait_until_complete")?;

        if let Some(runtime) = &mut self.runtime {
            loop {
                match runtime.wait_until_complete(Duration::from_millis(100)) {
                    Some(_) => return Ok(()),
                    None => {
                        py.check_signals()?;
                    }
                }
            }
        }
        Ok(())
    }

    fn stop(&mut self, timeout: Option<f32>, py: Python) -> PyResult<()> {
        let _guard = span_from_py(py, "Acquisition::stop")?;

        let timeout_float: f32 = timeout.unwrap_or(30_f32);
        match &mut self.runtime {
            None => Err(exceptions::PyRuntimeError::new_err(
                "trying to stop while not running",
            )),
            Some(runtime) => {
                if runtime.stop().is_err() {
                    return Err(exceptions::PyRuntimeError::new_err(
                        "connection to background thread lost",
                    ));
                }
                let timeout = Duration::from_secs_f32(timeout_float);
                let deadline = Instant::now() + timeout;
                while Instant::now() < deadline {
                    if runtime.try_join().is_some() {
                        self.shm.take();
                        return Ok(());
                    }
                    std::thread::sleep(Duration::from_millis(100));
                    py.check_signals()?;
                }
                // deadline exceeded
                Err(exceptions::PyRuntimeError::new_err(
                    "timeout while waiting for background thread to stop",
                ))
            }
        }
    }

    fn get_next_frame(&mut self, py: Python) -> PyResult<Option<PyFrame>> {
        let _guard = span_from_py(py, "Cam::get_next_frame")?;

        if !self.enable_frame_iterator {
            return Err(exceptions::PyRuntimeError::new_err(
                "get_next_frame called without enable_frame_iterator",
            ));
        }
        if let Some(runtime) = &mut self.runtime {
            loop {
                runtime.update_state();
                match runtime.get_next_frame() {
                    Err(RuntimeError::Timeout) => {
                        py.check_signals()?;
                        // FIXME: break out of here after some larger timeout of no data received?
                        continue;
                    }
                    Err(e) => return Err(e.into()),
                    Ok(result) => {
                        py.check_signals()?;
                        match &result {
                            AcquisitionResult::Frame(frame, _frame_idx)
                            | AcquisitionResult::DroppedFrame(frame, _frame_idx) => {
                                // FIXME: support for subframes! maybe on the pyframe object itself?
                                if let Some(frame_idx) =
                                    runtime.frame_in_acquisition(frame.get_frame_id())
                                {
                                    return Ok(Some(PyFrame::new(result, frame_idx)));
                                } else {
                                    let frame_id = result.get_frame().unwrap().get_frame_id();
                                    println!("recycling frame {frame_id}");
                                    runtime.frame_done(result)?;
                                }
                            }
                            AcquisitionResult::DroppedFrameOutside(_) => {
                                runtime.frame_done(result)?;
                            }
                            AcquisitionResult::ShutdownIdle
                            | AcquisitionResult::DoneShuttingDown { acquisition_id: _ } => {
                                return Err(exceptions::PyRuntimeError::new_err(
                                    "acquisition runtime is shutting down",
                                ));
                            }
                            AcquisitionResult::DoneAborted {
                                dropped,
                                acquisition_id,
                            }
                            | AcquisitionResult::DoneSuccess {
                                dropped,
                                acquisition_id,
                            } => {
                                eprintln!("dropped {dropped} frames in this acquisition");
                                return Ok(None);
                            }
                        }
                    }
                }
            }
        } else {
            Err(exceptions::PyRuntimeError::new_err(
                "acquisition is not running",
            ))
        }
    }

    fn frame_done(&mut self, frame: &mut PyFrame) -> PyResult<()> {
        if let Some(runtime) = &mut self.runtime {
            runtime.frame_done(frame.consume_frame_data())?;
            Ok(())
        } else {
            Err(exceptions::PyRuntimeError::new_err(
                "acquisition is not running",
            ))
        }
    }

    fn get_frame_slot(&mut self, frame: &mut PyFrame) -> PyResult<usize> {
        if let Some(runtime) = &mut self.runtime {
            let slot = runtime.get_frame_slot(frame.consume_frame_data());
            if let Some(idx) = slot {
                Ok(idx)
            } else {
                Err(exceptions::PyRuntimeError::new_err(
                    "acquisition is not running",
                ))
            }
        } else {
            Err(exceptions::PyRuntimeError::new_err(
                "acquisition is not running",
            ))
        }
    }

    fn get_frame_shape(&self) -> (usize, usize) {
        match self.camera_mode {
            PyMode::IS => (1860, 2048),
            PyMode::Summit => (3840, 4096),
        }
    }

    /// Arm the runtime for the next acquisition
    fn make_acquisition(
        &mut self,
        py: Python,
        params: PyAcquisitionParams,
    ) -> PyResult<Acquisition> {
        let _guard = span_from_py(py, "Cam::make_acquisition")?;
        let p = params.clone();
        let acq_params = AcquisitionParams {
            size: p.size.into(),
            sync: p.sync.into(),
            binning: k2o::events::Binning::Bin1x,
            writer_settings: p.writer_settings,
        };
        if let Some(runtime) = &mut self.runtime {
            if runtime.arm(acq_params).is_err() {
                return Err(exceptions::PyRuntimeError::new_err(
                    "connection to background thread lost",
                ));
            }
            let tracer = get_tracer();

            tracer.in_span("AcquisitionRuntime::wait_for_arm", |_cx| -> PyResult<()> {
                loop {
                    py.check_signals()?;
                    if runtime.wait_for_arm(Duration::from_millis(100)).is_some() {
                        return Ok(());
                    }
                }
            })?;
            Ok(Acquisition::new(
                params,
                self.camera_mode.into(),
                runtime.get_current_acquisition_id(),
            ))
        } else {
            Err(exceptions::PyRuntimeError::new_err(
                "invalid state - acquisition runtime not available",
            ))
        }
    }
}
