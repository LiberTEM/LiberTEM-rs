use std::{
    convert::Infallible,
    fmt::Display,
    thread::JoinHandle,
    time::{Duration, Instant},
};

use crate::common::{
    self, DConfig, DHeader, DImage, DImageD, DSeriesEnd, DetectorConfig, FrameData, FrameSender,
    PixelType, TriggerMode,
};

use bincode::serialize;
use crossbeam_channel::{unbounded, Receiver, RecvTimeoutError, SendError, Sender, TryRecvError};
use log::{debug, info};
use pyo3::{
    create_exception, exceptions,
    prelude::*,
    types::{PyBytes, PyType},
};
use zmq::{Message, Socket};

#[pymodule]
fn libertem_dectris(py: Python, m: &PyModule) -> PyResult<()> {
    // FIXME: logging integration deadlocks on close(), when trying to acquire
    // the GIL
    // pyo3_log::init();

    m.add_class::<Frame>().unwrap();
    m.add_class::<FrameIterator>().unwrap();
    m.add_class::<FrameStack>().unwrap();
    m.add_class::<FrameChunkedIterator>().unwrap();
    m.add_class::<PixelType>().unwrap();
    m.add_class::<DectrisSim>().unwrap();
    m.add_class::<DetectorConfig>().unwrap();
    m.add_class::<TriggerMode>().unwrap();
    m.add("TimeoutError", py.get_type::<TimeoutError>())?;
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

    fn with_data(frame: FrameData) -> Self {
        Frame { frame }
    }
}

#[pymethods]
impl Frame {
    fn __repr__(&self) -> String {
        let frame = &self.frame;
        let series = frame.dimage.series;
        let shape = &frame.dimaged.shape;
        let idx = frame.dimage.frame;
        return format!("<Frame idx={idx} series={series} shape={shape:?}>");
    }

    fn get_image_data(slf: PyRef<Self>, py: Python) -> Py<PyBytes> {
        let bytes: &PyBytes = PyBytes::new(py, &slf.frame.image_data.as_slice());
        return bytes.into();
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
        return match &slf.frame.dimaged.type_ {
            PixelType::Uint8 => "uint8".to_string(),
            PixelType::Uint16 => "uint16".to_string(),
            PixelType::Uint32 => "uint32".to_string(),
        };
    }

    fn get_encoding(slf: PyRef<Self>) -> String {
        slf.frame.dimaged.encoding.clone()
    }

    fn get_shape(slf: PyRef<Self>) -> Vec<u64> {
        slf.frame.dimaged.shape.clone()
    }
}

pub enum ControlMsg {
    StopThread,
    StartAcquisition { series: u64 },
}

#[derive(PartialEq)]
pub enum ResultMsg {
    Error { msg: String }, // generic error response, might need to specialize later
    Frame { frame: FrameData },
    End,
}

#[derive(PartialEq)]
pub enum ReceiverStatus {
    Idle,
    Running,
    Closed,
}

pub struct DectrisReceiver {
    bg_thread: Option<JoinHandle<()>>,
    to_thread: Sender<ControlMsg>,
    from_thread: Receiver<ResultMsg>,
    pub status: ReceiverStatus,
}

fn recv_part(
    msg: &mut Message,
    socket: &Socket,
    control_channel: &Receiver<ControlMsg>,
) -> Result<(), AcquisitionError> {
    loop {
        match socket.recv(msg, 0) {
            Ok(_) => break,
            Err(zmq::Error::EAGAIN) => {
                check_for_control(control_channel)?;
                continue;
            }
            Err(err) => AcquisitionError::ZmqError { err },
        };
    }
    Ok(())
}

fn recv_frame(
    socket: &Socket,
    control_channel: &Receiver<ControlMsg>,
) -> Result<FrameData, AcquisitionError> {
    let mut msg: Message = Message::new();
    let mut data: Vec<u8> = Vec::with_capacity(512 * 512 * 4);

    recv_part(&mut msg, socket, control_channel)?;
    let dimage: DImage = serde_json::from_str(msg.as_str().unwrap()).unwrap();

    recv_part(&mut msg, socket, control_channel)?;
    let dimaged: DImageD = serde_json::from_str(msg.as_str().unwrap()).unwrap();

    // compressed image data:
    recv_part(&mut msg, socket, control_channel)?;
    data.truncate(0);
    data.extend_from_slice(&msg);

    // DConfig:
    recv_part(&mut msg, socket, control_channel)?;
    let dconfig: DConfig = serde_json::from_str(msg.as_str().unwrap()).unwrap();

    Ok(FrameData {
        dimage,
        dimaged,
        image_data: data,
        dconfig,
    })
}

#[derive(Debug, Clone)]
enum AcquisitionError {
    Disconnected,
    SeriesMismatch,
    Cancelled,
    ZmqError { err: zmq::Error },
}

impl Display for AcquisitionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "acquisition has stopped")
    }
}

fn check_for_control(control_channel: &Receiver<ControlMsg>) -> Result<(), AcquisitionError> {
    match control_channel.try_recv() {
        Ok(ControlMsg::StartAcquisition { series: _ }) => {
            panic!("received StartAcquisition while an acquisition was already running");
        }
        Ok(ControlMsg::StopThread) => {
            return Err(AcquisitionError::Cancelled);
        }
        Err(TryRecvError::Disconnected) => {
            return Err(AcquisitionError::Cancelled);
        }
        Err(TryRecvError::Empty) => Ok(()),
    }
}

fn acquisition(
    detector_config: DetectorConfig,
    to_thread_r: &Receiver<ControlMsg>,
    from_thread_s: &Sender<ResultMsg>,
    socket: &Socket,
    series: u64,
) -> Result<(), AcquisitionError> {
    let t0 = Instant::now();
    let mut last_control_check = Instant::now();
    loop {
        if last_control_check.elapsed() > Duration::from_millis(300) {
            last_control_check = Instant::now();
            check_for_control(&to_thread_r)?;
        }

        let frame = recv_frame(&socket, &to_thread_r)?;

        if frame.dimage.series != series {
            return Err(AcquisitionError::SeriesMismatch);
        }

        // we will be done after this frame:
        let done = frame.dimage.frame == detector_config.get_num_images() - 1;

        // send to our queue:
        match from_thread_s.send(ResultMsg::Frame { frame }) {
            Ok(_) => (),
            Err(SendError(_)) => return Err(AcquisitionError::Disconnected),
        }

        if done {
            let elapsed = t0.elapsed();
            info!("done in {elapsed:?}, reading acquisition footer...");

            let mut msg: Message = Message::new();

            socket.recv(&mut msg, 0).unwrap();
            let footer: DSeriesEnd = serde_json::from_str(msg.as_str().unwrap()).unwrap();
            let series = footer.series;
            info!("series {series} done");

            match from_thread_s.send(ResultMsg::End) {
                Ok(_) => (),
                Err(SendError(_)) => return Err(AcquisitionError::Disconnected),
            }
            return Ok(());
        }
    }
}

/// convert `AcquisitionError`s to messages on `from_threads_s`
fn background_thread_wrap(to_thread_r: &Receiver<ControlMsg>, from_thread_s: &Sender<ResultMsg>) {
    match background_thread(to_thread_r, from_thread_s) {
        Ok(()) => {}
        Err(err) => {
            from_thread_s
                .send(ResultMsg::Error {
                    msg: err.to_string(),
                })
                .unwrap();
            return;
        }
    }
}

fn background_thread(
    to_thread_r: &Receiver<ControlMsg>,
    from_thread_s: &Sender<ResultMsg>,
) -> Result<(), AcquisitionError> {
    let ctx = zmq::Context::new();
    let socket = ctx.socket(zmq::PULL).unwrap();
    socket.set_rcvtimeo(1000).unwrap();
    socket.connect("tcp://127.0.0.1:9999").unwrap();
    socket.set_rcvhwm(4 * 256).unwrap();

    loop {
        // control: main threads tells us to quit
        let control = to_thread_r.recv_timeout(Duration::from_millis(100));
        match control {
            Ok(ControlMsg::StartAcquisition { series }) => {
                let mut msg: Message = Message::new();
                recv_part(&mut msg, &socket, &to_thread_r)?;
                // panic in case of any other header type:
                let dheader: DHeader = serde_json::from_str(msg.as_str().unwrap()).unwrap();
                debug!("dheader: {dheader:?}");

                // second message: the header itself
                recv_part(&mut msg, &socket, &to_thread_r)?;
                let detector_config: DetectorConfig =
                    serde_json::from_str(msg.as_str().unwrap()).unwrap();

                match acquisition(detector_config, to_thread_r, from_thread_s, &socket, series) {
                    Ok(_) => {}
                    Err(AcquisitionError::Disconnected | AcquisitionError::Cancelled) => {
                        return Ok(());
                    }
                    e => {
                        return e;
                    }
                }
            }
            Ok(ControlMsg::StopThread) => {
                break;
            }
            Err(RecvTimeoutError::Disconnected) => {
                break;
            }
            Err(RecvTimeoutError::Timeout) => (), // no message, nothing to do
        }
    }
    Ok(())
}

pub struct ReceiverError {
    msg: String,
}

impl Display for ReceiverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let msg = &self.msg;
        write!(f, "{msg}")
    }
}

impl DectrisReceiver {
    pub fn new() -> Self {
        let (to_thread_s, to_thread_r) = unbounded();
        let (from_thread_s, from_thread_r) = unbounded();

        let builder = std::thread::Builder::new();

        DectrisReceiver {
            bg_thread: Some(
                builder
                    .name("bg_thread".to_string())
                    .spawn(move || background_thread_wrap(&to_thread_r, &from_thread_s))
                    .expect("failed to start background thread"),
            ),
            from_thread: from_thread_r,
            to_thread: to_thread_s,
            status: ReceiverStatus::Idle,
        }
    }

    pub fn next(&mut self) -> ResultMsg {
        let result_msg = self
            .from_thread
            .recv()
            .expect("background thread should be running");
        if result_msg == ResultMsg::End {
            self.status = ReceiverStatus::Idle;
        }
        return result_msg;
    }

    pub fn next_timeout(&mut self, timeout: Duration) -> Option<ResultMsg> {
        let result_msg = self.from_thread.recv_timeout(timeout);

        match result_msg {
            Ok(result) => {
                if result == ResultMsg::End {
                    self.status = ReceiverStatus::Idle;
                }
                return Some(result);
            }
            Err(e) => match e {
                RecvTimeoutError::Disconnected => {
                    panic!("background thread should be running")
                }
                RecvTimeoutError::Timeout => return None,
            },
        }
    }

    pub fn start(&mut self, series: u64) -> Result<(), ReceiverError> {
        if self.status == ReceiverStatus::Closed {
            return Err(ReceiverError {
                msg: "receiver is closed".to_string(),
            });
        }
        self.to_thread
            .send(ControlMsg::StartAcquisition { series })
            .expect("background thread should be running");
        self.status = ReceiverStatus::Running;
        Ok(())
    }

    pub fn close(&mut self) {
        self.to_thread.send(ControlMsg::StopThread).unwrap();
        if let Some(join_handle) = self.bg_thread.take() {
            join_handle
                .join()
                .expect("could not join background thread!");
        } else {
            panic!("expected to have a join handle, had None instead");
        }
        self.status = ReceiverStatus::Closed;
    }
}

#[pyclass]
pub struct FrameIterator {
    receiver: DectrisReceiver,
}

#[pymethods]
impl FrameIterator {
    #[new]
    fn new() -> Self {
        FrameIterator {
            receiver: DectrisReceiver::new(),
        }
    }

    fn start(mut slf: PyRefMut<Self>, series: u64) -> PyResult<()> {
        slf.receiver
            .start(series)
            .map_err(|err| return exceptions::PyRuntimeError::new_err(err.msg))
    }

    fn close(mut slf: PyRefMut<Self>) {
        slf.receiver.close();
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<Self>, py: Python) -> PyResult<Option<Frame>> {
        loop {
            match slf.receiver.next_timeout(Duration::from_millis(100)) {
                Some(ResultMsg::Error { msg }) => {
                    return Err(exceptions::PyRuntimeError::new_err(msg));
                }
                Some(ResultMsg::End) => return Ok(None),
                Some(ResultMsg::Frame { frame }) => {
                    return Ok(Some(Frame::with_data_cloned(&frame)))
                }
                None => {
                    py.check_signals()?;
                    py.allow_threads(|| {
                        spin_sleep::sleep(Duration::from_millis(1));
                    });
                    continue;
                }
            }
        }
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

    #[classmethod]
    fn deserialize(_cls: &PyType, serialized: &PyBytes) -> Self {
        let data = serialized.as_bytes();
        FrameStack::with_data(bincode::deserialize(&data).unwrap())
    }

    fn serialize(slf: PyRef<Self>, py: Python) -> PyResult<Py<PyBytes>> {
        let bytes: &PyBytes = PyBytes::new(py, serialize(&slf.frames).unwrap().as_slice());
        return Ok(bytes.into());
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

#[pyclass]
struct FrameChunkedIterator {
    receiver: DectrisReceiver,
}

#[pymethods]
impl FrameChunkedIterator {
    #[new]
    fn new() -> Self {
        FrameChunkedIterator {
            receiver: DectrisReceiver::new(),
        }
    }

    fn start(mut slf: PyRefMut<Self>, series: u64) -> PyResult<()> {
        slf.receiver
            .start(series)
            .map_err(|err| return exceptions::PyRuntimeError::new_err(err.msg))
    }

    fn close(mut slf: PyRefMut<Self>) {
        slf.receiver.close();
    }

    fn is_running(slf: PyRef<Self>) -> bool {
        return slf.receiver.status == ReceiverStatus::Running;
    }

    fn get_next_stack(
        mut slf: PyRefMut<Self>,
        py: Python,
        max_size: usize,
    ) -> PyResult<FrameStack> {
        let mut stack = FrameStack::empty();

        match slf.receiver.status {
            ReceiverStatus::Closed => {
                return Err(exceptions::PyRuntimeError::new_err("receiver is closed"))
            }
            ReceiverStatus::Idle => return Ok(stack),
            ReceiverStatus::Running => {}
        }

        let recv = &mut slf.receiver;

        loop {
            py.check_signals()?;

            let recv_result = py.allow_threads(|| {
                let next: Result<Option<ResultMsg>, Infallible> =
                    Ok(recv.next_timeout(Duration::from_millis(100)));
                return next;
            })?;

            match recv_result {
                None => {
                    continue;
                }
                Some(ResultMsg::Error { msg }) => {
                    return Err(exceptions::PyRuntimeError::new_err(msg))
                }
                Some(ResultMsg::End) => {
                    return Ok(stack);
                }
                Some(ResultMsg::Frame { frame }) => {
                    stack.push(frame);
                    if stack.len() >= max_size {
                        return Ok(stack);
                    }
                }
            }
        }
    }
}

create_exception!(
    libertem_dectris,
    TimeoutError,
    exceptions::PyException,
    "Timeout while communicating"
);

#[pyclass]
struct DectrisSim {
    frame_sender: FrameSender,
    dwelltime: Option<u64>, // in µseconds
}

#[pymethods]
impl DectrisSim {
    #[new]
    fn new(uri: &str, filename: &str, dwelltime: Option<u64>) -> Self {
        return DectrisSim {
            frame_sender: FrameSender::new(uri, filename),
            dwelltime,
        };
    }

    fn get_detector_config(slf: PyRef<Self>) -> DetectorConfig {
        slf.frame_sender.get_detector_config().clone()
    }

    fn send_headers(mut slf: PyRefMut<Self>) {
        slf.frame_sender.send_headers();
    }

    /// send `nframes`, if given, or all frames in the acquisition, from the
    /// current position in the file
    fn send_frames(mut slf: PyRefMut<Self>, py: Python, nframes: Option<u64>) -> PyResult<()> {
        let mut t0 = Instant::now();
        let start_time = Instant::now();

        let effective_nframes = match nframes {
            None => slf.frame_sender.get_num_frames(),
            Some(n) => n,
        };

        for frame_idx in 0..effective_nframes {
            // can't `allow_threads` because the zmq socket is not Send
            //py.allow_threads(|| {
            match slf.frame_sender.send_frame() {
                Err(common::SendError::Timeout) => {
                    return Err(TimeoutError::new_err(
                        "timeout while sending frames".to_string(),
                    ));
                }
                Err(_) => {
                    return Err(exceptions::PyRuntimeError::new_err(
                        "error while sending frames".to_string(),
                    ));
                }
                Ok(_) => {} // Ok(())
            }
            //})?;

            // dwelltime
            // FIXME: for continuous mode, u64 might not be enough for elapsed time,
            // so maybe it's better to carry around a "budget" that can be negative
            // if a frame hasn't been sent out in time etc.
            if let Some(dt) = slf.dwelltime {
                let elapsed_us = start_time.elapsed().as_micros() as u64;
                let target_time_us = (frame_idx + 1) * dt;
                if elapsed_us < target_time_us {
                    let delta = target_time_us - elapsed_us;
                    spin_sleep::sleep(Duration::from_micros(delta));
                }
            }

            // run Python signal handlers every now and then
            if t0.elapsed() > Duration::from_millis(300) {
                t0 = Instant::now();
                py.check_signals()?;

                // also drop GIL once in a while
                py.allow_threads(|| {
                    spin_sleep::sleep(Duration::from_micros(5));
                });
            }
        }

        Ok(())
    }

    fn send_footer(mut slf: PyRefMut<Self>) {
        slf.frame_sender.send_footer();
    }
}
