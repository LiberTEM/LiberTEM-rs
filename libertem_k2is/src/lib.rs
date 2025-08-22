pub mod background_thread;
pub mod config;
pub mod decoder;
pub mod frame_meta;
pub mod main_py;

use common::tracing::{get_tracer, span_from_py, tracing_from_env};
use env_logger::Builder;
use ipc_test::SharedSlabAllocator;
use log::info;
use opentelemetry::trace::Tracer;
use std::{
    path::Path,
    time::{Duration, Instant},
};

use k2o::runtime::{AcquisitionRuntimeConfig, AddrConfig, RuntimeError, WaitResult};

use k2o::{
    acquisition::AcquisitionResult,
    events::{AcquisitionParams, AcquisitionSync},
    frame::GenericFrame,
    params::CameraMode,
    runtime::AcquisitionRuntime,
};

use pyo3::{exceptions, prelude::*, types::PyType};

/*

#[pymodule]
fn libertem_k2is(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    tracing_from_env("libertem-k2is".to_owned());

    m.add_class::<Acquisition>()?;
    m.add_class::<Cam>()?;
    m.add_class::<SyncFlags>()?;
    m.add_class::<PyMode>()?;
    m.add_class::<PyAcquisitionParams>()?;

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

impl From<PyMode> for CameraMode {
    fn from(val: PyMode) -> Self {
        match val {
            PyMode::IS => CameraMode::IS,
            PyMode::Summit => CameraMode::Summit,
        }
    }
}

#[pymethods]
impl PyMode {
    #[classmethod]
    fn from_string(_cls: &Bound<'_, PyType>, mode: &str) -> PyResult<Self> {
        match mode.to_lowercase().as_ref() {
            "is" => Ok(Self::IS),
            "summit" => Ok(Self::Summit),
            _ => Err(exceptions::PyValueError::new_err(format!(
                "unknown mode: {}",
                mode
            ))),
        }
    }
}

#[pyclass(name = "AcquisitionParams")]
#[derive(Debug, Clone)]
struct PyAcquisitionParams {
    pub size: Option<u32>,
    pub sync: SyncFlags,
}

#[pymethods]
impl PyAcquisitionParams {
    #[new]
    fn new(sync: SyncFlags, size: Option<u32>) -> Self {
        PyAcquisitionParams { size, sync }
    }
}

#[pyclass(name = "Frame")]
struct PyFrame {
    acquisition_result: Option<AcquisitionResult<GenericFrame>>,
    frame_idx: u32,
    frame_id: u32,
}

impl PyFrame {
    fn new(result: AcquisitionResult<GenericFrame>, frame_idx: u32, frame_id: u32) -> Self {
        PyFrame {
            acquisition_result: Some(result),
            frame_idx,
            frame_id,
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

    fn get_id(&self) -> u32 {
        self.frame_id
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

#[pymethods]
impl Acquisition {
    fn get_id(&self) -> usize {
        self.id
    }

    fn get_mode(&self) -> PyMode {
        self.camera_mode.into()
    }

    fn get_params(&self) -> PyAcquisitionParams {
        self.params.clone()
    }

    fn __repr__(&self) -> String {
        let id = self.id;
        format!("<Acquisition id='{id}'>")
    }
}

fn convert_runtime_error(val: RuntimeError) -> PyErr {
    let msg = format!("runtime error: {val:?}");
    exceptions::PyRuntimeError::new_err(msg)
}

#[pyclass]
struct Cam {
    camera_mode: PyMode,
    enable_frame_iterator: bool,
    shm: Option<SharedSlabAllocator>,
    runtime: Option<AcquisitionRuntime>,
}

#[pymethods]
impl Cam {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python,
        local_addr_top: &str,
        local_addr_bottom: &str,
        mode: PyMode,
        shm_path: &str,
        enable_frame_iterator: Option<bool>,
        recv_realtime: Option<bool>,
        assembly_realtime: Option<bool>,
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
        let enable_frame_iterator = enable_frame_iterator.unwrap_or(true);
        let runtime_config = AcquisitionRuntimeConfig::new(
            enable_frame_iterator,
            recv_realtime.unwrap_or(true),
            assembly_realtime.unwrap_or(true),
            mode.into(),
            addr_config,
        );

        let runtime = AcquisitionRuntime::new(&runtime_config, shm.get_handle());

        Ok(Cam {
            camera_mode: mode,
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
                if runtime
                    .wait_for_start(Duration::from_millis(100))
                    .is_success()
                {
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
                    WaitResult::PredSuccess => return Ok(()),
                    WaitResult::Timeout => {
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
                    Err(e) => return Err(convert_runtime_error(e)),
                    Ok(result) => {
                        py.check_signals()?;
                        match &result {
                            AcquisitionResult::Frame(frame, _frame_idx)
                            | AcquisitionResult::DroppedFrame(frame, _frame_idx) => {
                                // FIXME: support for subframes! maybe on the pyframe object itself?
                                if let Some(frame_idx) =
                                    runtime.frame_in_acquisition(frame.get_frame_id())
                                {
                                    let frame_id = frame.get_frame_id();
                                    return Ok(Some(PyFrame::new(result, frame_idx, frame_id)));
                                } else {
                                    let frame_id = result.get_frame().unwrap().get_frame_id();
                                    println!("recycling frame {frame_id}");
                                    runtime.frame_done(result).map_err(convert_runtime_error)?;
                                }
                            }
                            AcquisitionResult::DroppedFrameOutside(_) => {
                                runtime.frame_done(result).map_err(convert_runtime_error)?;
                            }
                            AcquisitionResult::ShutdownIdle
                            | AcquisitionResult::DoneShuttingDown { acquisition_id: _ } => {
                                return Err(exceptions::PyRuntimeError::new_err(
                                    "acquisition runtime is shutting down",
                                ));
                            }
                            AcquisitionResult::DoneAborted {
                                dropped,
                                acquisition_id: _,
                            }
                            | AcquisitionResult::DoneSuccess {
                                dropped,
                                acquisition_id: _,
                            } => {
                                info!("dropped {dropped} frames in this acquisition");
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
            runtime
                .frame_done(frame.consume_frame_data())
                .map_err(convert_runtime_error)?;
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
                    if runtime
                        .wait_for_arm(Duration::from_millis(100))
                        .is_success()
                    {
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

*/
