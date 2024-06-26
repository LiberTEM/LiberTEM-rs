#![allow(clippy::borrow_deref_ref)]

use std::{
    convert::Infallible,
    path::PathBuf,
    time::{Duration, Instant},
};

use crate::{
    cam_client::CamClient,
    common::{DType, FrameMeta},
    exceptions::{ConnectionError, TimeoutError},
    frame_stack::FrameStackHandle,
    receiver::{ReceiverStatus, ResultMsg, ServalReceiver},
};

use ipc_test::SharedSlabAllocator;
use log::{info, trace};
use pyo3::{
    exceptions::{self, PyRuntimeError},
    prelude::*,
};
use serval_client::{DetectorConfig, DetectorInfo, DetectorLayout, ServalClient};
use stats::Stats;

#[pymodule]
fn libertem_asi_mpx3(py: Python, m: &PyModule) -> PyResult<()> {
    // FIXME: logging integration deadlocks on close(), when trying to acquire
    // the GIL
    // pyo3_log::init();

    m.add_class::<FrameStackHandle>()?;
    m.add_class::<ServalConnection>()?;
    m.add_class::<DType>()?;
    m.add_class::<PyDetectorConfig>()?;
    m.add_class::<PyDetectorInfo>()?;
    m.add_class::<PyServalClient>()?;
    m.add_class::<CamClient>()?;
    m.add("TimeoutError", py.get_type::<TimeoutError>())?;

    register_header_module(py, m)?;

    let env = env_logger::Env::default()
        .filter_or("LIBERTEM_ASI_LOG_LEVEL", "error")
        .write_style_or("LIBERTEM_ASI_LOG_STYLE", "always");
    env_logger::Builder::from_env(env)
        .format_timestamp_micros()
        .init();

    Ok(())
}

fn register_header_module(py: Python<'_>, parent_module: &PyModule) -> PyResult<()> {
    let headers_module = PyModule::new(py, "headers")?;
    parent_module.add_submodule(headers_module)?;
    Ok(())
}

#[derive(Debug)]
#[pyclass(name = "DetectorConfig")]
struct PyDetectorConfig {
    config: DetectorConfig,
}

#[pymethods]
impl PyDetectorConfig {
    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }

    fn get_n_triggers(&self) -> u64 {
        self.config.n_triggers
    }
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
        format!("{:?}", self)
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
            .map(|value| PyDetectorConfig { config: value })
    }

    fn get_detector_info(&self) -> PyResult<PyDetectorInfo> {
        self.client
            .get_detector_info()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
            .map(|value| PyDetectorInfo { info: value })
    }
}

struct FrameChunkedIterator<'a, 'b, 'c, 'd> {
    receiver: &'a mut ServalReceiver,
    shm: &'b mut SharedSlabAllocator,
    remainder: &'c mut Vec<FrameStackHandle>,
    stats: &'d mut Stats,
}

impl<'a, 'b, 'c, 'd> FrameChunkedIterator<'a, 'b, 'c, 'd> {
    /// Get the next frame stack. Mainly handles splitting logic for boundary
    /// conditions and delegates communication with the background thread to `recv_next_stack_impl`
    pub fn get_next_stack_impl(
        &mut self,
        py: Python,
        max_size: usize,
    ) -> PyResult<Option<FrameStackHandle>> {
        let res = self.recv_next_stack_impl(py);
        match res {
            Ok(Some(frame_stack)) => {
                if frame_stack.len() > max_size {
                    // split `FrameStackHandle` into two:
                    trace!(
                        "FrameStackHandle::split_at({max_size}); len={}",
                        frame_stack.len()
                    );
                    self.stats.count_split();
                    let (left, right) = frame_stack.split_at(max_size, self.shm);
                    self.remainder.push(right);
                    assert!(left.len() <= max_size);
                    return Ok(Some(left));
                }
                assert!(frame_stack.len() <= max_size);
                Ok(Some(frame_stack))
            }
            Ok(None) => Ok(None),
            e @ Err(_) => e,
        }
    }

    /// Receive the next frame stack from the background thread and handle any
    /// other control messages.
    fn recv_next_stack_impl(&mut self, py: Python) -> PyResult<Option<FrameStackHandle>> {
        // first, check if there is anything on the remainder list:
        if let Some(frame_stack) = self.remainder.pop() {
            return Ok(Some(frame_stack));
        }

        match self.receiver.status {
            ReceiverStatus::Closed => {
                return Err(exceptions::PyRuntimeError::new_err("receiver is closed"))
            }
            ReceiverStatus::Idle => return Ok(None),
            ReceiverStatus::Running => {}
        }

        let recv = &mut self.receiver;

        loop {
            py.check_signals()?;

            let recv_result = py.allow_threads(|| {
                let next: Result<Option<ResultMsg>, Infallible> =
                    Ok(recv.next_timeout(Duration::from_millis(100)));
                next
            })?;

            match recv_result {
                None => {
                    continue;
                }
                Some(ResultMsg::ParseError { msg }) => {
                    return Err(exceptions::PyRuntimeError::new_err(msg))
                }
                Some(ResultMsg::AcquisitionStart {
                    detector_config: _,
                    first_frame_meta: _,
                }) => {
                    // FIXME: in case of "passive" mode, we should actually not hit this,
                    // as the "outer" structure (`ServalConnection`) handles it?
                    continue;
                }
                Some(ResultMsg::Error { msg }) => {
                    return Err(exceptions::PyRuntimeError::new_err(msg))
                }
                Some(ResultMsg::End { frame_stack }) => {
                    self.stats.log_stats();
                    self.stats.reset();
                    return Ok(Some(frame_stack));
                }
                Some(ResultMsg::FrameStack { frame_stack }) => {
                    return Ok(Some(frame_stack));
                }
            }
        }
    }

    fn new(
        receiver: &'a mut ServalReceiver,
        shm: &'b mut SharedSlabAllocator,
        remainder: &'c mut Vec<FrameStackHandle>,
        stats: &'d mut Stats,
    ) -> PyResult<Self> {
        Ok(FrameChunkedIterator {
            receiver,
            shm,
            remainder,
            stats,
        })
    }
}

#[pyclass]
struct ServalConnection {
    receiver: ServalReceiver,
    remainder: Vec<FrameStackHandle>,
    local_shm: SharedSlabAllocator,
    stats: Stats,
}

impl ServalConnection {
    fn start_passive_impl(&mut self) -> PyResult<()> {
        self.receiver
            .start_passive()
            .map_err(|err| exceptions::PyRuntimeError::new_err(err.msg))
    }

    fn close_impl(&mut self) {
        self.receiver.close();
    }
}

#[pyclass]
struct PendingAcquisition {
    config: DetectorConfig,
    first_frame_meta: FrameMeta,
}

#[pymethods]
impl PendingAcquisition {
    fn get_detector_config(&self) -> PyDetectorConfig {
        PyDetectorConfig {
            config: self.config.clone(),
        }
    }

    fn get_frame_width(&self) -> u16 {
        self.first_frame_meta.width
    }

    fn get_frame_height(&self) -> u16 {
        self.first_frame_meta.height
    }
}

#[pymethods]
impl ServalConnection {
    #[new]
    fn new(
        data_uri: &str,
        api_uri: &str,
        frame_stack_size: usize,
        handle_path: String,
        num_slots: Option<usize>,
        bytes_per_frame: Option<usize>,
        huge: Option<bool>,
    ) -> PyResult<Self> {
        let num_slots = num_slots.map_or_else(|| 2000, |x| x);
        let bytes_per_frame = bytes_per_frame.map_or_else(|| 512 * 512 * 2, |x| x);
        let slot_size = frame_stack_size * bytes_per_frame;
        let shm = match SharedSlabAllocator::new(
            num_slots,
            slot_size,
            huge.map_or_else(|| false, |x| x),
            &PathBuf::from(handle_path),
        ) {
            Ok(shm) => shm,
            Err(e) => {
                let total_size = num_slots * slot_size;
                let msg = format!("could not create SHM area (num_slots={num_slots}, slot_size={slot_size} total_size={total_size} huge={huge:?}): {e:?}");
                return Err(ConnectionError::new_err(msg));
            }
        };

        let local_shm = shm.clone_and_connect().expect("clone SHM");

        Ok(Self {
            receiver: ServalReceiver::new(data_uri, api_uri, frame_stack_size, shm),
            remainder: Vec::new(),
            local_shm,
            stats: Stats::new(),
        })
    }

    /// Wait until the detector is armed, or until the timeout expires (in seconds)
    /// Returns `None` in case of timeout, the detector config otherwise.
    /// This method drops the GIL to allow concurrent Python threads.
    fn wait_for_arm(&mut self, timeout: f32, py: Python) -> PyResult<Option<PendingAcquisition>> {
        let timeout = Duration::from_secs_f32(timeout);
        let deadline = Instant::now() + timeout;
        let step = Duration::from_millis(100);

        loop {
            py.check_signals()?;

            let res = py.allow_threads(|| {
                let timeout_rem = deadline - Instant::now();
                let this_timeout = timeout_rem.min(step);
                self.receiver.next_timeout(this_timeout)
            });

            match res {
                Some(ResultMsg::AcquisitionStart {
                    detector_config,
                    first_frame_meta,
                }) => {
                    return Ok(Some(PendingAcquisition {
                        config: detector_config,
                        first_frame_meta,
                    }))
                }
                msg @ Some(ResultMsg::End { .. }) | msg @ Some(ResultMsg::FrameStack { .. }) => {
                    let err = format!("unexpected message: {:?}", msg);
                    return Err(PyRuntimeError::new_err(err));
                }
                Some(ResultMsg::ParseError { msg }) => return Err(PyRuntimeError::new_err(msg)),
                Some(ResultMsg::Error { msg }) => return Err(PyRuntimeError::new_err(msg)),
                None => {
                    // timeout
                    if Instant::now() > deadline {
                        return Ok(None);
                    } else {
                        continue;
                    }
                }
            }
        }
    }

    fn get_socket_path(&self) -> PyResult<String> {
        Ok(self.local_shm.get_handle().os_handle)
    }

    fn is_running(slf: PyRef<Self>) -> bool {
        slf.receiver.status == ReceiverStatus::Running
    }

    /// Start listening for global acquisition headers on the zeromq socket.
    /// Call `wait_for_arm` to wait
    fn start_passive(mut slf: PyRefMut<Self>) -> PyResult<()> {
        slf.start_passive_impl()
    }

    fn close(mut slf: PyRefMut<Self>) {
        slf.stats.log_stats();
        slf.stats.reset();
        slf.close_impl();
    }

    fn get_next_stack(
        &mut self,
        py: Python,
        max_size: usize,
    ) -> PyResult<Option<FrameStackHandle>> {
        let mut iter = FrameChunkedIterator::new(
            &mut self.receiver,
            &mut self.local_shm,
            &mut self.remainder,
            &mut self.stats,
        )?;
        iter.get_next_stack_impl(py, max_size).map(|maybe_stack| {
            if let Some(frame_stack) = &maybe_stack {
                self.stats.count_stats_item(frame_stack);
            }
            maybe_stack
        })
    }

    fn log_shm_stats(&self) {
        let free = self.local_shm.num_slots_free();
        let total = self.local_shm.num_slots_total();
        self.stats.log_stats();
        info!("shm stats free/total: {}/{}", free, total);
    }
}
