mod bgthread;
mod shm_helpers;

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
    block::{K2Block, K2ISBlock},
    events::{AcquisitionParams, AcquisitionSync},
    frame::{K2Frame, K2ISFrame},
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
    m.add_class::<PyAcquisitionParams>()?;
    m.add_class::<PyWriter>()?;

    m.add_class::<CamClient>()?;
    m.add_class::<FrameRef>()?;

    let env = env_logger::Env::default()
        .filter_or("LIBERTEM_K2IS_LOG_LEVEL", "error")
        .write_style_or("LIBERTEM_K2IS_LOG_STYLE", "always");
    env_logger::init_from_env(env);

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

#[pyclass(name = "Writer")]
#[derive(Debug, Clone)]
struct PyWriter {
    pub filename: String,
    pub method: String,
}

impl PyWriter {
    fn get_writer_builder(&self, filename: &str) -> PyResult<Box<dyn WriterBuilder>> {
        let wb: Box<dyn WriterBuilder + Send> = match self.method.as_str() {
            "direct" => DirectWriterBuilder::for_filename(filename),
            "mmap" => MMapWriterBuilder::for_filename(filename),
            #[cfg(feature = "hdf5")]
            "hdf5" => HDF5WriterBuilder::for_filename(filename),
            _ => {
                let meth = &self.method;
                let msg = format!("unknown method {meth}, choose one: mmap, direct, hdf5");
                return Err(exceptions::PyValueError::new_err(msg));
            }
        };
        Ok(wb)
    }
}

#[pymethods]
impl PyWriter {
    #[new]
    fn new(filename: &str, method: &str) -> Self {
        PyWriter {
            filename: filename.to_string(),
            method: method.to_string(),
        }
    }
}

#[pyclass(name = "AcquisitionParams")]
#[derive(Debug, Clone)]
struct PyAcquisitionParams {
    pub size: Option<u32>,
    pub sync: SyncFlags,
    pub writer: Option<PyWriter>,
    pub enable_frame_iterator: bool,
    pub shm_path: String,
}

#[pymethods]
impl PyAcquisitionParams {
    #[new]
    fn new(
        size: Option<u32>,
        sync: SyncFlags,
        writer: Option<PyWriter>,
        enable_frame_iterator: bool,
        shm_path: String,
    ) -> Self {
        PyAcquisitionParams {
            size,
            sync,
            writer,
            enable_frame_iterator,
            shm_path,
        }
    }
}

#[pyclass(name = "Frame")]
struct PyFrame {
    acquisition_result: Option<AcquisitionResult<K2ISFrame>>,
    frame_idx: u32,
}

impl PyFrame {
    fn new(result: AcquisitionResult<K2ISFrame>, frame_idx: u32) -> Self {
        PyFrame {
            acquisition_result: Some(result),
            frame_idx,
        }
    }

    fn consume_frame_data(&mut self) -> AcquisitionResult<K2ISFrame> {
        self.acquisition_result.take().unwrap()
    }
}

#[pymethods]
impl PyFrame {
    fn is_dropped(slf: PyRef<Self>) -> bool {
        matches!(
            &slf.acquisition_result,
            Some(AcquisitionResult::DroppedFrame(..))
                | Some(AcquisitionResult::DroppedFrameOutside(..))
        )
    }

    fn get_idx(slf: PyRef<Self>) -> u32 {
        slf.frame_idx
    }
}

///
/// An `Acquisition` is an object representing acquisition parameters,
#[pyclass]
struct Acquisition {
    params: PyAcquisitionParams,
    addr_config: AddrConfig,

    // FIXME: assumes IS mode currently
    runtime: Option<AcquisitionRuntime<K2ISFrame>>,
    shm: SharedSlabAllocator,
}

impl Acquisition {
    pub fn new(
        params: PyAcquisitionParams,
        addr_config: &AddrConfig,
        shm: SharedSlabAllocator,
    ) -> Self {
        Acquisition {
            params,
            addr_config: addr_config.clone(),
            runtime: None,
            shm,
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
    fn wait_for_start(mut slf: PyRefMut<Self>, py: Python) -> PyResult<()> {
        // TODO: total deadline for initialization?
        // TODO: currently the API can be used wrong, i.e. calling
        // `get_next_frame` before this function means it will throw away
        // perfectly fine frames that actually belong to the beginning of the
        // acquisition. This can be prevented: instead, the user should get back
        // an object from this function which is the actual frame iterator. Or
        // we should make the wait implicit and integrate it into
        // `get_next_frame`... hmm.

        if let Some(runtime) = &mut slf.runtime {
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

    fn get_next_frame(mut slf: PyRefMut<Self>, py: Python) -> PyResult<Option<PyFrame>> {
        if !slf.params.enable_frame_iterator {
            return Err(exceptions::PyRuntimeError::new_err(
                "get_next_frame called without enable_frame_iterator",
            ));
        }
        if let Some(runtime) = &mut slf.runtime {
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
                                    let frame_id = result.get_frame().unwrap().frame_id;
                                    println!("recycling frame {frame_id}");
                                    runtime.frame_done(result)?;
                                }
                            }
                            AcquisitionResult::DroppedFrameOutside(_) => {
                                runtime.frame_done(result)?;
                            }
                            AcquisitionResult::DoneError => {
                                // FIXME: propagate error as python exception?
                                panic!("WHAT?!");
                                return Ok(None);
                            }
                            AcquisitionResult::DoneAborted { dropped }
                            | AcquisitionResult::DoneSuccess { dropped } => {
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

    fn frame_done(mut slf: PyRefMut<Self>, frame: &mut PyFrame) -> PyResult<()> {
        if let Some(runtime) = &mut slf.runtime {
            runtime.frame_done(frame.consume_frame_data())?;
            Ok(())
        } else {
            Err(exceptions::PyRuntimeError::new_err(
                "acquisition is not running",
            ))
        }
    }

    fn get_frame_slot(mut slf: PyRefMut<Self>, frame: &mut PyFrame) -> PyResult<usize> {
        if let Some(runtime) = &mut slf.runtime {
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

    fn get_frame_shape(_slf: PyRef<Self>) {}

    ///
    /// Start the receiver. Starts receiver threads and, if configured,
    /// instantiates a writer.
    ///
    /// Immediately returns after initialization, but the frame iterator
    /// will block until the data actually arrives.
    ///
    fn arm(mut slf: PyRefMut<Self>, py: Python) -> PyResult<()> {
        let _guard = span_from_py(py, "Acquisition::arm")?;

        let wb = if let Some(writer) = &slf.params.writer {
            writer.get_writer_builder(&writer.filename)?
        } else {
            Box::<k2o::write::NoopWriterBuilder>::default()
        };

        let sync: AcquisitionSync = slf.params.sync.clone().into();
        let p: AcquisitionParams = AcquisitionParams {
            size: slf.params.size.into(),
            sync,
            binning: k2o::events::Binning::Bin1x,
        };
        let mut runtime = AcquisitionRuntime::new::<{ <K2ISBlock as K2Block>::PACKET_SIZE }>(
            wb,
            &slf.addr_config,
            slf.params.enable_frame_iterator,
            slf.shm.get_handle(),
        );
        if runtime.arm(p).is_err() {
            return Err(exceptions::PyRuntimeError::new_err(
                "connection to background thread lost",
            ));
        }
        loop {
            py.check_signals()?;
            if runtime.wait_for_arm(Duration::from_millis(100)).is_some() {
                break;
            }
        }
        slf.runtime = Some(runtime);
        Ok(())
    }

    fn stop(mut slf: PyRefMut<Self>, timeout: Option<f32>, py: Python) -> PyResult<()> {
        let _guard = span_from_py(py, "Acquisition::stop")?;

        let timeout_float: f32 = timeout.unwrap_or(30_f32);
        match &mut slf.runtime {
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

    ///
    /// Wait for the acquisition to complete. This is only needed
    /// if we are only writing to a file, and not consuming the
    /// frame iterator, which takes care of this synchronization
    /// otherwise.
    ///
    /// Also succeeds if the runtime is already shut down.
    ///
    // FIXME: timeout?
    fn wait_until_complete(mut slf: PyRefMut<Self>, py: Python) -> PyResult<()> {
        let _guard = span_from_py(py, "Acquisition::wait_until_complete")?;

        if let Some(runtime) = &mut slf.runtime {
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
}

#[pyclass]
struct Cam {
    addr_config: AddrConfig,
}

#[pymethods]
impl Cam {
    #[new]
    fn new(local_addr_top: &str, local_addr_bottom: &str) -> Self {
        Cam {
            addr_config: AddrConfig::new(local_addr_top, local_addr_bottom),
        }
    }

    fn make_acquisition(
        slf: PyRef<Self>,
        py: Python,
        params: PyAcquisitionParams,
    ) -> PyResult<Acquisition> {
        let _guard = span_from_py(py, "Cam::make_acquisition")?;
        let path = Path::new(&params.shm_path);
        let shm = SharedSlabAllocator::new(1000, 2048 * 1860 * 2, true, path).expect("create shm");
        Ok(Acquisition::new(params, &slf.addr_config, shm))
    }
}
