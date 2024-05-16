#![allow(clippy::borrow_deref_ref)]

use std::{
    convert::Infallible,
    path::PathBuf,
    time::{Duration, Instant},
};

use crate::{
    cam_client::CamClient,
    common::{
        DConfig, DHeader, DImage, DImageD, DSeriesEnd, DectrisFrameMeta, DetectorConfig, PixelType,
        TriggerMode,
    },
    exceptions::{ConnectionError, DecompressError, TimeoutError},
    frame_stack_py::PyFrameStackHandle,
    receiver::{DectrisReceiver, ReceiverStatus, ResultMsg},
    sim::DectrisSim,
};

use common::frame_stack::FrameStackHandle;
use ipc_test::SharedSlabAllocator;
use log::{info, trace};
use pyo3::{
    exceptions::{self, PyRuntimeError},
    prelude::*,
};
use stats::Stats;

#[pymodule]
fn libertem_dectris(py: Python, m: Bound<'_, PyModule>) -> PyResult<()> {
    // FIXME: logging integration deadlocks on close(), when trying to acquire
    // the GIL
    // pyo3_log::init();

    m.add_class::<PyFrameStackHandle>()?;
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

#[pyclass]
struct DectrisConnection {
    receiver: DectrisReceiver,
    remainder: Vec<FrameStackHandle<DectrisFrameMeta>>,
    local_shm: SharedSlabAllocator,
    stats: Stats,
}

impl DectrisConnection {
    fn start_series_impl(&mut self, series: u64) -> PyResult<()> {
        self.receiver
            .start_series(series)
            .map_err(|err| exceptions::PyRuntimeError::new_err(err.msg))
    }

    fn start_passive_impl(&mut self) -> PyResult<()> {
        self.receiver
            .start_passive()
            .map_err(|err| exceptions::PyRuntimeError::new_err(err.msg))
    }

    fn close_impl(&mut self) {
        self.receiver.close();
    }
}

#[pymethods]
impl DectrisConnection {
    #[new]
    fn new(
        uri: &str,
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
            receiver: DectrisReceiver::new(uri, frame_stack_size, shm),
            remainder: Vec::new(),
            local_shm,
            stats: Stats::new(),
        })
    }

    /// Wait until the detector is armed, or until the timeout expires (in seconds)
    /// Returns `None` in case of timeout, the detector config otherwise.
    /// This method drops the GIL to allow concurrent Python threads.
    fn wait_for_arm(
        &mut self,
        timeout: f32,
        py: Python,
    ) -> PyResult<Option<(DetectorConfig, u64)>> {
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
                    series,
                    detector_config,
                }) => return Ok(Some((detector_config, series))),
                msg @ Some(ResultMsg::End { .. }) | msg @ Some(ResultMsg::FrameStack { .. }) => {
                    let err = format!("unexpected message: {:?}", msg);
                    return Err(PyRuntimeError::new_err(err));
                }
                Some(ResultMsg::Error { msg }) => return Err(PyRuntimeError::new_err(msg)),
                Some(ResultMsg::SerdeError { msg, recvd_msg: _ }) => {
                    return Err(PyRuntimeError::new_err(msg))
                }
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

    fn start(mut slf: PyRefMut<Self>, series: u64) -> PyResult<()> {
        slf.start_series_impl(series)
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
    ) -> PyResult<Option<PyFrameStackHandle>> {
        let mut iter = FrameChunkedIterator::new(
            &mut self.receiver,
            &mut self.local_shm,
            &mut self.remainder,
            &mut self.stats,
        )?;
        py.allow_threads(|| {
            iter.get_next_stack_impl(py, max_size).map(|maybe_stack| {
                if let Some(frame_stack) = &maybe_stack {
                    self.stats.count_stats_item(frame_stack);
                }
                maybe_stack.map(PyFrameStackHandle::new)
            })
        })
    }

    fn log_shm_stats(&self) {
        let free = self.local_shm.num_slots_free();
        let total = self.local_shm.num_slots_total();
        self.stats.log_stats();
        info!("shm stats free/total: {}/{}", free, total);
    }
}
