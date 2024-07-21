#![allow(clippy::borrow_deref_ref)]

use std::{
    convert::Infallible,
    path::PathBuf,
    time::{Duration, Instant},
};

use crate::{
    background_thread::{ReceiverStatus, ResultMsg, TPXReceiver},
    cam_client::CamClient,
    chunk_stack::ChunkStackHandle,
    exceptions::{ConnectionError, TimeoutError},
    headers::{AcquisitionEnd, AcquisitionStart, DType, FormatType, ScanEnd, ScanStart},
    headers_py::make_sim_data,
};

use ipc_test::SharedSlabAllocator;
use log::{info, trace, warn};
use pyo3::{
    exceptions::{self, PyRuntimeError},
    prelude::*,
};
use stats::Stats;

#[pymodule]
fn libertem_asi_tpx3(py: Python, m: Bound<'_, PyModule>) -> PyResult<()> {
    // FIXME: logging integration deadlocks on close(), when trying to acquire
    // the GIL
    // pyo3_log::init();

    m.add_class::<ChunkStackHandle>()?;
    m.add_class::<ASITpx3Connection>()?;
    m.add_class::<DType>()?;
    m.add_class::<FormatType>()?;
    m.add_class::<AcquisitionStart>()?;
    m.add_class::<ScanStart>()?;
    m.add_class::<ScanEnd>()?;
    m.add_class::<AcquisitionEnd>()?;
    m.add_class::<CamClient>()?;
    m.add_function(wrap_pyfunction!(make_sim_data, &m)?)?;
    m.add("TimeoutError", py.get_type_bound::<TimeoutError>())?;

    // register_header_module(py, m)?;

    let env = env_logger::Env::default()
        .filter_or("LIBERTEM_TPX_LOG_LEVEL", "error")
        .write_style_or("LIBERTEM_TPX_LOG_STYLE", "always");
    env_logger::init_from_env(env);

    Ok(())
}

// fn register_header_module(py: Python<'_>, parent_module: &PyModule) -> PyResult<()> {
//     let headers_module = PyModule::new(py, "headers")?;
//     headers_module.add_class::<AcquisitionStart>()?;
//     headers_module.add_class::<AcquisitionEnd>()?;
//     headers_module.add_class::<ScanStart>()?;
//     headers_module.add_class::<ScanEnd>()?;
//     headers_module.add_class::<ArrayChunk>()?;
//     parent_module.add_submodule(headers_module)?;
//     Ok(())
// }

struct ChunkIterator<'a, 'b, 'c, 'd> {
    receiver: &'a mut TPXReceiver,
    shm: &'b mut SharedSlabAllocator,
    remainder: &'c mut Vec<ChunkStackHandle>,
    stats: &'d mut Stats,
}

impl<'a, 'b, 'c, 'd> ChunkIterator<'a, 'b, 'c, 'd> {
    /// Get the next frame stack. Mainly handles splitting logic for boundary
    /// conditions and delegates communication with the background thread to `recv_next_stack_impl`
    pub fn get_next_stack_impl(
        &mut self,
        py: Python,
        max_size: u32,
    ) -> PyResult<Option<ChunkStackHandle>> {
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
    fn recv_next_stack_impl(&mut self, py: Python) -> PyResult<Option<ChunkStackHandle>> {
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
                Some(ResultMsg::AcquisitionStart { header: _ }) => {
                    // FIXME: in case of "passive" mode, we should actually not hit this,
                    // as the "outer" structure (`*Connection`) handles it?
                    continue;
                }
                Some(ResultMsg::ScanStart { header: _ }) => {
                    continue;
                }
                Some(ResultMsg::SerdeError { msg, recvd_msg }) => {
                    return Err(exceptions::PyRuntimeError::new_err(format!(
                        "serialization error: {msg}, message: {recvd_msg}",
                    )))
                }
                Some(ResultMsg::AcquisitionError { msg }) => {
                    return Err(exceptions::PyRuntimeError::new_err(msg))
                }
                Some(ResultMsg::ReceiverError { msg }) => {
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
        receiver: &'a mut TPXReceiver,
        shm: &'b mut SharedSlabAllocator,
        remainder: &'c mut Vec<ChunkStackHandle>,
        stats: &'d mut Stats,
    ) -> PyResult<Self> {
        Ok(ChunkIterator {
            receiver,
            shm,
            remainder,
            stats,
        })
    }
}

#[pyclass]
struct ASITpx3Connection {
    receiver: TPXReceiver,
    remainder: Vec<ChunkStackHandle>,
    local_shm: SharedSlabAllocator,
    stats: Stats,
}

impl ASITpx3Connection {
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
impl ASITpx3Connection {
    #[new]
    fn new(
        uri: &str,
        chunks_per_stack: usize,
        handle_path: String,
        num_slots: Option<usize>,
        bytes_per_chunk: Option<usize>,
        huge: Option<bool>,
    ) -> PyResult<Self> {
        let num_slots = num_slots.map_or_else(|| 2000, |x| x);
        let bytes_per_chunk = bytes_per_chunk.map_or_else(|| 512 * 512 * 2, |x| x);
        let slot_size = chunks_per_stack * bytes_per_chunk;
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

        info!(
            "shm created, num_slots={num_slots}, slot_size={slot_size}, total_size={}",
            num_slots * slot_size
        );

        let local_shm = shm.clone_and_connect().expect("clone SHM");

        Ok(Self {
            receiver: TPXReceiver::new(uri, chunks_per_stack, shm),
            remainder: Vec::new(),
            local_shm,
            stats: Stats::new(),
        })
    }

    /// Wait until the detector is armed, or until the timeout expires (in seconds)
    /// Returns `None` in case of timeout, the detector config otherwise.
    /// This method drops the GIL to allow concurrent Python threads.
    fn wait_for_arm(&mut self, timeout: f32, py: Python) -> PyResult<Option<AcquisitionStart>> {
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
                    header,
                    // }) => return Ok(Some(header)),
                }) => {
                    trace!("got ResultMsg::AcquisitionStart");
                    return Ok(Some(header));
                }
                msg @ Some(ResultMsg::FrameStack { .. }) | msg @ Some(ResultMsg::End { .. }) => {
                    warn!("discarding unexpected message {msg:?}");
                    continue;
                }
                Some(ResultMsg::ScanStart { header: _ }) => {
                    todo!("what do we do here?");
                }
                Some(ResultMsg::ReceiverError { msg }) => return Err(PyRuntimeError::new_err(msg)),
                Some(ResultMsg::AcquisitionError { msg }) => {
                    return Err(PyRuntimeError::new_err(msg))
                }
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

    /// Start listening for global acquisition headers on the tcp socket.
    /// Call `wait_for_arm` to wait
    fn start_passive(mut slf: PyRefMut<Self>) -> PyResult<()> {
        slf.start_passive_impl()
    }

    // FIXME: results in "Already Borrowed" if called while another thread is running in `self`,
    // for example `wait_for_arm`, which calls `allow_threads`...
    fn close(&mut self) {
        self.stats.log_stats();
        self.stats.reset();
        self.close_impl();
    }

    fn get_next_stack(&mut self, py: Python, max_size: u32) -> PyResult<Option<ChunkStackHandle>> {
        let mut iter = ChunkIterator::new(
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
