#![allow(clippy::borrow_deref_ref)]

use std::{
    convert::Infallible,
    sync::{self, atomic::AtomicBool},
    thread::JoinHandle,
    time::{Duration, Instant},
};

use crate::{
    bs::decompress_lz4_into,
    cam_client::CamClient,
    common::{
        DConfig, DHeader, DImage, DImageD, DSeriesEnd, DetectorConfig, FrameData, PixelType,
        TriggerMode,
    },
    exceptions::{ConnectionError, DecompressError, TimeoutError},
    frame_stack::FrameStackHandle,
    receiver::{DectrisReceiver, ReceiverStatus, ResultMsg},
    shm::serve_shm_handle,
    sim::DectrisSim,
};

use bincode::serialize;

use ipc_test::SharedSlabAllocator;
use log::{debug, info, trace};
use numpy::PyArray2;
use pyo3::{
    exceptions::{self, PyRuntimeError},
    prelude::*,
    types::{PyBytes, PyType},
};

#[pymodule]
fn libertem_dectris(py: Python, m: &PyModule) -> PyResult<()> {
    // FIXME: logging integration deadlocks on close(), when trying to acquire
    // the GIL
    // pyo3_log::init();

    m.add_class::<Frame>()?;
    m.add_class::<FrameStack>()?;
    m.add_class::<FrameStackHandle>()?;
    m.add_class::<DectrisConnection>()?;
    m.add_class::<PixelType>()?;
    m.add_class::<DectrisSim>()?;
    m.add_class::<DetectorConfig>()?;
    m.add_class::<TriggerMode>()?;
    m.add_class::<CamClient>()?;
    m.add("TimeoutError", py.get_type::<TimeoutError>())?;
    m.add("DecompressError", py.get_type::<DecompressError>())?;

    register_header_module(py, m)?;

    let env = env_logger::Env::default()
        .filter_or("LIBERTEM_DECTRIS_LOG_LEVEL", "error")
        .write_style_or("LIBERTEM_DECTRIS_LOG_STYLE", "always");
    env_logger::init_from_env(env);

    Ok(())
}

fn register_header_module(py: Python<'_>, parent_module: &PyModule) -> PyResult<()> {
    let headers_module = PyModule::new(py, "headers")?;
    headers_module.add_class::<DHeader>()?;
    headers_module.add_class::<DImage>()?;
    headers_module.add_class::<DImageD>()?;
    headers_module.add_class::<DConfig>()?;
    headers_module.add_class::<DSeriesEnd>()?;
    parent_module.add_submodule(headers_module)?;
    Ok(())
}

#[pyclass(module = "libertem_dectris")]
#[derive(Clone)]
pub struct Frame {
    frame: FrameData,
}

impl Frame {
    fn with_data_cloned(frame: &FrameData) -> Self {
        Frame {
            frame: frame.clone(),
        }
    }

    fn get_size(&self) -> u64 {
        return self.frame.dimaged.shape.iter().product();
    }

    fn decompress_into_impl<T: numpy::Element>(&self, out: &PyArray2<T>) -> PyResult<()> {
        let mut out_rw = out.readwrite();
        let out_slice = out_rw.as_slice_mut().expect("`out` must be C-contiguous");
        let out_size = usize::try_from(self.get_size()).unwrap(); // number of elements
        let out_ptr: *mut T = out_slice.as_mut_ptr().cast();
        match decompress_lz4_into(&self.frame.image_data[12..], out_ptr, out_size, None) {
            Ok(()) => Ok(()),
            Err(e) => {
                let msg = format!("decompression failed: {e:?}");
                Err(DecompressError::new_err(msg))
            }
        }
    }
}

#[pymethods]
impl Frame {
    #[new]
    fn new(data: &PyBytes, dimage: &DImage, dimaged: &DImageD, dconfig: &DConfig) -> Self {
        let frame_data: FrameData = FrameData {
            dimage: dimage.clone(),
            dimaged: dimaged.clone(),
            image_data: data.as_bytes().into(),
            dconfig: dconfig.clone(),
        };

        Frame { frame: frame_data }
    }

    fn __repr__(&self) -> String {
        let frame = &self.frame;
        let series = frame.dimage.series;
        let shape = &frame.dimaged.shape;
        let idx = frame.dimage.frame;
        format!("<Frame idx={idx} series={series} shape={shape:?}>")
    }

    fn get_image_data(slf: PyRef<Self>, py: Python) -> Py<PyBytes> {
        let bytes: &PyBytes = PyBytes::new(py, slf.frame.image_data.as_slice());
        bytes.into()
    }

    fn decompress_into(slf: PyRef<Self>, out: &PyAny) -> PyResult<()> {
        let arr_u8: Result<&PyArray2<u8>, _> = out.downcast();
        let arr_u16: Result<&PyArray2<u16>, _> = out.downcast();
        let arr_u32: Result<&PyArray2<u32>, _> = out.downcast();

        match slf.frame.dimaged.encoding.as_str() {
            "bs32-lz4<" => {
                slf.decompress_into_impl(arr_u32.unwrap())?;
            }
            "bs16-lz4<" => {
                slf.decompress_into_impl(arr_u16.unwrap())?;
            }
            "bs8-lz4<" => {
                slf.decompress_into_impl(arr_u8.unwrap())?;
            }
            e => {
                let msg = format!("can't deal with encoding {e}");
                return Err(exceptions::PyValueError::new_err(msg));
            }
        }
        Ok(())
    }

    fn get_series_id(slf: PyRef<Self>) -> u64 {
        slf.frame.dimage.series
    }

    fn get_frame_id(slf: PyRef<Self>) -> u64 {
        slf.frame.dimage.frame
    }

    fn get_hash(slf: PyRef<Self>) -> String {
        slf.frame.dimage.hash.clone()
    }

    fn get_pixel_type(slf: PyRef<Self>) -> String {
        match &slf.frame.dimaged.type_ {
            PixelType::Uint8 => "uint8".to_string(),
            PixelType::Uint16 => "uint16".to_string(),
            PixelType::Uint32 => "uint32".to_string(),
        }
    }

    fn get_encoding(slf: PyRef<Self>) -> String {
        slf.frame.dimaged.encoding.clone()
    }

    /// return endianess in numpy notation
    fn get_endianess(slf: PyRef<Self>) -> PyResult<String> {
        match slf.frame.dimaged.encoding.chars().last() {
            Some(c) => Ok(c.to_string()),
            None => Err(exceptions::PyValueError::new_err(
                "encoding should be non-empty".to_string(),
            )),
        }
    }

    fn get_shape(slf: PyRef<Self>) -> Vec<u64> {
        slf.frame.dimaged.shape.clone()
    }
}

impl From<Frame> for FrameData {
    fn from(frame: Frame) -> Self {
        frame.frame
    }
}

#[pyclass]
struct FrameStack {
    frames: Vec<FrameData>,
}

impl FrameStack {
    fn empty() -> Self {
        FrameStack {
            frames: Vec::with_capacity(128),
        }
    }

    fn with_data(frames: Vec<FrameData>) -> Self {
        FrameStack { frames }
    }

    fn len(&self) -> usize {
        self.frames.len()
    }

    fn push(&mut self, frame: FrameData) {
        self.frames.push(frame);
    }

    fn get(&self, key: usize) -> Option<&FrameData> {
        if let Some(item) = self.frames.get(key) {
            Some(item)
        } else {
            None
        }
    }
}

#[pymethods]
impl FrameStack {
    #[new]
    fn new() -> Self {
        FrameStack::empty()
    }

    /// create a frame stack from a list of frames
    /// NOTE: this probably doesn't perform well, and is only meant for testing
    #[classmethod]
    fn from_frame_list(_cls: &PyType, frames: Vec<Frame>) -> Self {
        FrameStack {
            frames: frames.into_iter().map(|f| f.into()).collect(),
        }
    }

    #[classmethod]
    fn deserialize(_cls: &PyType, serialized: &PyBytes) -> Self {
        let data = serialized.as_bytes();
        FrameStack::with_data(bincode::deserialize(data).unwrap())
    }

    fn serialize(slf: PyRef<Self>, py: Python) -> PyResult<Py<PyBytes>> {
        let bytes: &PyBytes = PyBytes::new(py, serialize(&slf.frames).unwrap().as_slice());
        Ok(bytes.into())
    }

    fn __len__(slf: PyRef<Self>) -> usize {
        slf.frames.len()
    }

    fn __getitem__(slf: PyRef<Self>, key: usize) -> PyResult<Frame> {
        if let Some(item) = slf.get(key) {
            Ok(Frame::with_data_cloned(item))
        } else {
            Err(exceptions::PyIndexError::new_err("frame not found"))
        }
    }
}

pub struct Stats {
    /// total number of bytes (compressed) that have flown through the system
    payload_size_sum: usize,

    /// maximum size of compressed frames seen
    frame_size_max: usize,

    /// minimum size of compressed frames seen
    frame_size_min: usize,

    /// sum of the size of the slots used
    slots_size_sum: usize,

    /// number of frames seen
    num_frames: usize,

    /// number of times a frame stack was split
    split_count: usize,
}

impl Stats {
    pub fn new() -> Self {
        Self {
            payload_size_sum: 0,
            slots_size_sum: 0,
            frame_size_max: 0,
            frame_size_min: usize::MAX,
            num_frames: 0,
            split_count: 0,
        }
    }

    pub fn count_frame_stack(&mut self, frame_stack: &FrameStackHandle) {
        self.payload_size_sum += frame_stack.payload_size();
        self.slots_size_sum += frame_stack.slot_size();
        self.frame_size_max = self.frame_size_max.max(
            frame_stack
                .get_meta()
                .iter()
                .max_by_key(|fm| fm.data_length_bytes)
                .map_or(self.frame_size_max, |fm| fm.data_length_bytes),
        );
        self.frame_size_min = self.frame_size_min.min(
            frame_stack
                .get_meta()
                .iter()
                .min_by_key(|fm| fm.data_length_bytes)
                .map_or(self.frame_size_min, |fm| fm.data_length_bytes),
        );
        self.num_frames += frame_stack.len();
    }

    pub fn count_split(&mut self) {
        self.split_count += 1;
    }

    pub fn log_stats(&self) {
        let efficiency = self.payload_size_sum as f32 / self.slots_size_sum as f32;
        debug!(
            "Stats: frames seen: {}, total payload size: {}, total slot size used: {}, min frame size: {}, max frame size: {}, splits: {}, shm efficiency: {}",
            self.num_frames, self.payload_size_sum, self.slots_size_sum, self.frame_size_min, self.frame_size_max, self.split_count, efficiency,
        );
    }
}

impl Default for Stats {
    fn default() -> Self {
        Self::new()
    }
}

struct FrameChunkedIterator<'a, 'b, 'c, 'd> {
    receiver: &'a mut DectrisReceiver,
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
                Some(ResultMsg::AcquisitionStart {
                    series: _,
                    detector_config: _,
                }) => {
                    // FIXME: in case of "passive" mode, we should actually not hit this,
                    // as the "outer" structure (`DectrisConnection`) handles it?
                    continue;
                }
                Some(ResultMsg::SerdeError { msg, recvd_msg }) => {
                    return Err(exceptions::PyRuntimeError::new_err(format!(
                        "serialization error: {}, message: {}",
                        msg, recvd_msg
                    )))
                }
                Some(ResultMsg::Error { msg }) => {
                    return Err(exceptions::PyRuntimeError::new_err(msg))
                }
                Some(ResultMsg::End { frame_stack }) => {
                    self.stats.log_stats();
                    return Ok(Some(frame_stack));
                }
                Some(ResultMsg::FrameStack { frame_stack }) => {
                    return Ok(Some(frame_stack));
                }
            }
        }
    }

    fn new(
        receiver: &'a mut DectrisReceiver,
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
struct DectrisConnection {
    receiver: DectrisReceiver,
    remainder: Vec<FrameStackHandle>,
    local_shm: SharedSlabAllocator,
    shm_stop_event: Option<sync::Arc<AtomicBool>>,
    shm_join_handle: Option<JoinHandle<()>>,
    shm_socket_path: Option<String>,
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
        if let Some(stop_event) = &self.shm_stop_event {
            stop_event.store(true, sync::atomic::Ordering::Relaxed);
            if let Some(join_handle) = self.shm_join_handle.take() {
                join_handle.join().expect("join background thread");
            }
        }
        self.shm_socket_path = None;
    }
}

#[pymethods]
impl DectrisConnection {
    #[new]
    fn new(
        uri: &str,
        frame_stack_size: usize,
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
            shm_stop_event: None,
            shm_join_handle: None,
            shm_socket_path: None,
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

    fn serve_shm(mut slf: PyRefMut<Self>, socket_path: &str) -> PyResult<()> {
        let (stop_event, join_handle) = serve_shm_handle(slf.receiver.shm_handle, socket_path);
        slf.shm_stop_event = Some(stop_event);
        slf.shm_join_handle = Some(join_handle);
        slf.shm_socket_path = Some(socket_path.to_string());
        Ok(())
    }

    fn get_socket_path(&self) -> PyResult<String> {
        if let Some(path) = &self.shm_socket_path {
            Ok(path.clone())
        } else {
            Err(PyRuntimeError::new_err("not serving shm at the moment"))
        }
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
                self.stats.count_frame_stack(frame_stack);
            }
            maybe_stack
        })
    }

    fn log_shm_stats(&self) {
        let free = self.local_shm.num_slots_free();
        let total = self.local_shm.num_slots_total();
        info!("shm stats free/total: {}/{}", free, total);
    }
}
