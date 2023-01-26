#![allow(clippy::borrow_deref_ref)]

use std::{
    convert::Infallible,
    fmt::Display,
    mem::replace,
    sync::{self, atomic::AtomicBool},
    thread::JoinHandle,
    time::{Duration, Instant},
};

use crate::{
    bs::decompress_lz4_into,
    common::{
        setup_monitor, DConfig, DHeader, DImage, DImageD, DSeriesEnd, DSeriesOnly, DetectorConfig,
        FrameData, PixelType, TriggerMode,
    },
    exceptions::{ConnectionError, DecompressError, TimeoutError},
    shm_recv::{serve_shm_handle, CamClient, FrameStackForWriting, FrameStackHandle},
    sim::DectrisSim,
};

use bincode::serialize;
use crossbeam_channel::{unbounded, Receiver, RecvTimeoutError, SendError, Sender, TryRecvError};
use ipc_test::{SHMHandle, SharedSlabAllocator};
use log::{debug, info, trace, warn};
use numpy::PyArray2;
use pyo3::{
    exceptions::{self, PyRuntimeError},
    prelude::*,
    types::{PyBytes, PyType},
};
use zmq::{Message, Socket};

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

pub enum ControlMsg {
    StopThread,

    /// Wait for DHeader messages and latch onto acquisitions,
    /// until the background thread is stopped.
    StartAcquisitionPassive,

    /// Wait for a specific series to start
    StartAcquisition {
        series: u64,
    },
}

///
#[derive(PartialEq, Eq, Debug)]
pub enum ResultMsg {
    Error {
        msg: String,
    }, // generic error response, might need to specialize later
    SerdeError {
        msg: String,
        recvd_msg: String,
    },
    AcquisitionStart {
        series: u64,
        detector_config: DetectorConfig,
    },
    FrameStack {
        frame_stack: FrameStackHandle,
    },
    End {
        frame_stack: FrameStackHandle,
    },
}

#[derive(PartialEq, Eq)]
pub enum ReceiverStatus {
    Idle,
    Running,
    Closed,
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

/// Receive a frame into a `FrameStackForWriting`, reusing the `Message` objects
/// that are passed in.
fn recv_frame_into(
    socket: &Socket,
    control_channel: &Receiver<ControlMsg>,
    msg: &mut Message,
    msg_image: &mut Message,
) -> Result<(DImage, DImageD, DConfig), AcquisitionError> {
    recv_part(msg, socket, control_channel)?;
    let dimage_res: Result<DImage, _> = serde_json::from_str(msg.as_str().unwrap());

    let dimage = match dimage_res {
        Ok(image) => image,
        Err(err) => {
            return Err(AcquisitionError::SerdeError {
                msg: err.to_string(),
                recvd_msg: msg
                    .as_str()
                    .map_or_else(|| "".to_string(), |m| m.to_string()),
            });
        }
    };

    recv_part(msg, socket, control_channel)?;
    let dimaged_res: Result<DImageD, _> = serde_json::from_str(msg.as_str().unwrap());

    let dimaged = match dimaged_res {
        Ok(image) => image,
        Err(err) => {
            return Err(AcquisitionError::SerdeError {
                msg: err.to_string(),
                recvd_msg: msg
                    .as_str()
                    .map_or_else(|| "".to_string(), |m| m.to_string()),
            });
        }
    };

    // compressed image data:
    recv_part(msg_image, socket, control_channel)?;

    // DConfig:
    recv_part(msg, socket, control_channel)?;
    let dconfig: DConfig = serde_json::from_str(msg.as_str().unwrap()).unwrap();

    Ok((dimage, dimaged, dconfig))
}

#[derive(Debug, Clone)]
enum AcquisitionError {
    Disconnected,
    SeriesMismatch,
    FrameIdMismatch { expected_id: u64, got_id: u64 },
    SerdeError { recvd_msg: String, msg: String },
    Cancelled,
    ZmqError { err: zmq::Error },
    BufferFull,
    StateError { msg: String },
}

impl Display for AcquisitionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AcquisitionError::ZmqError { err } => {
                write!(f, "zmq error {err}")
            }
            AcquisitionError::Cancelled => {
                write!(f, "acquisition cancelled")
            }
            AcquisitionError::SerdeError { recvd_msg, msg } => {
                write!(f, "deserialization failed: {msg}; got msg {recvd_msg}")
            }
            AcquisitionError::SeriesMismatch => {
                write!(f, "series mismatch")
            }
            AcquisitionError::FrameIdMismatch {
                expected_id,
                got_id,
            } => {
                write!(f, "frame id mismatch; got {got_id}, expected {expected_id}")
            }
            AcquisitionError::Disconnected => {
                write!(f, "other end has disconnected")
            }
            AcquisitionError::BufferFull => {
                write!(f, "shm buffer is full")
            }
            AcquisitionError::StateError { msg } => {
                write!(f, "state error: {msg}")
            }
        }
    }
}

/// With a running acquisition, check for control messages;
/// especially convert `ControlMsg::StopThread` to `AcquisitionError::Cancelled`.
fn check_for_control(control_channel: &Receiver<ControlMsg>) -> Result<(), AcquisitionError> {
    match control_channel.try_recv() {
        Ok(ControlMsg::StartAcquisition { series: _ }) => Err(AcquisitionError::StateError {
            msg: "received StartAcquisition while an acquisition was already running".to_string(),
        }),
        Ok(ControlMsg::StartAcquisitionPassive) => Err(AcquisitionError::StateError {
            msg: "received StartAcquisitionPassive while an acquisition was already running"
                .to_string(),
        }),
        Ok(ControlMsg::StopThread) => Err(AcquisitionError::Cancelled),
        Err(TryRecvError::Disconnected) => Err(AcquisitionError::Cancelled),
        Err(TryRecvError::Empty) => Ok(()),
    }
}

/// Passively listen for global acquisition headers
/// and automatically latch on to them.
fn passive_acquisition(
    control_channel: &Receiver<ControlMsg>,
    from_thread_s: &Sender<ResultMsg>,
    socket: &Socket,
    frame_stack_size: usize,
    shm: &mut SharedSlabAllocator,
) -> Result<(), AcquisitionError> {
    loop {
        let mut msg: Message = Message::new();

        // block until we get a message:
        recv_part(&mut msg, socket, control_channel)?;

        if msg[0] == b'{' {
            let dheader_res: Result<DHeader, _> = serde_json::from_str(msg.as_str().unwrap());
            let dheader: DHeader = match dheader_res {
                Ok(header) => header,
                Err(_err) => {
                    // not a DHeader, ignore
                    continue;
                }
            };
            debug!("dheader: {dheader:?}");

            // second message: the header itself
            recv_part(&mut msg, socket, control_channel)?;
            let detector_config: DetectorConfig =
                serde_json::from_str(msg.as_str().unwrap()).unwrap();

            acquisition(
                detector_config,
                control_channel,
                from_thread_s,
                socket,
                dheader.series,
                frame_stack_size,
                shm,
            )?;

            let free = shm.num_slots_free();
            let total = shm.num_slots_total();
            info!("passive acquisition done; free slots: {}/{}", free, total);
        } else {
            // probably a binary message: skip this
            continue;
        }

        check_for_control(control_channel)?;
    }
}

fn acquisition(
    detector_config: DetectorConfig,
    to_thread_r: &Receiver<ControlMsg>,
    from_thread_s: &Sender<ResultMsg>,
    socket: &Socket,
    series: u64,
    frame_stack_size: usize,
    shm: &mut SharedSlabAllocator,
) -> Result<(), AcquisitionError> {
    let t0 = Instant::now();
    let mut last_control_check = Instant::now();

    let mut expected_frame_id = 0;

    match from_thread_s.send(ResultMsg::AcquisitionStart {
        series,
        detector_config: detector_config.clone(),
    }) {
        Ok(_) => (),
        Err(SendError(_)) => return Err(AcquisitionError::Disconnected),
    }

    // approx uppper bound of image size in bytes
    let approx_size_bytes = detector_config.get_num_pixels()
        * (detector_config.bit_depth_image as f32 / 8.0f32).ceil() as u64;

    let slot = match shm.get_mut() {
        None => return Err(AcquisitionError::BufferFull),
        Some(x) => x,
    };
    let mut frame_stack =
        FrameStackForWriting::new(slot, frame_stack_size, approx_size_bytes as usize);

    let mut msg = Message::new();
    let mut msg_image = Message::new();

    loop {
        if last_control_check.elapsed() > Duration::from_millis(300) {
            last_control_check = Instant::now();
            check_for_control(to_thread_r)?;
        }

        let (dimage, dimaged, dconfig) =
            recv_frame_into(socket, to_thread_r, &mut msg, &mut msg_image)?;

        if dimage.series != series {
            return Err(AcquisitionError::SeriesMismatch);
        }

        if dimage.frame != expected_frame_id {
            return Err(AcquisitionError::FrameIdMismatch {
                expected_id: expected_frame_id,
                got_id: dimage.frame,
            });
        }

        if !frame_stack.can_fit(msg_image.len()) {
            // send to our queue:
            let handle = {
                let slot = match shm.get_mut() {
                    None => return Err(AcquisitionError::BufferFull),
                    Some(x) => x,
                };
                let new_frame_stack =
                    FrameStackForWriting::new(slot, frame_stack_size, approx_size_bytes as usize);
                let old_frame_stack = replace(&mut frame_stack, new_frame_stack);
                old_frame_stack.writing_done(shm)
            };
            match from_thread_s.send(ResultMsg::FrameStack {
                frame_stack: handle,
            }) {
                Ok(_) => (),
                Err(SendError(_)) => return Err(AcquisitionError::Disconnected),
            }
        }

        let frame = frame_stack.frame_done(dimage, dimaged, dconfig, &msg_image);

        expected_frame_id += 1;

        // we will be done after this frame:
        let done = frame.dimage.frame == detector_config.get_num_images() - 1;

        if done {
            let elapsed = t0.elapsed();
            info!("done in {elapsed:?}, reading acquisition footer...");

            let mut msg: Message = Message::new();

            socket.recv(&mut msg, 0).unwrap();
            let footer: DSeriesEnd = serde_json::from_str(msg.as_str().unwrap()).unwrap();
            let series = footer.series;
            info!("series {series} done");

            let handle = frame_stack.writing_done(shm);

            match from_thread_s.send(ResultMsg::End {
                frame_stack: handle,
            }) {
                Ok(_) => (),
                Err(SendError(_)) => return Err(AcquisitionError::Disconnected),
            }
            return Ok(());
        }
    }
}

/// convert `AcquisitionError`s to messages on `from_threads_s`
fn background_thread_wrap(
    to_thread_r: &Receiver<ControlMsg>,
    from_thread_s: &Sender<ResultMsg>,
    uri: String,
    frame_stack_size: usize,
    shm: SharedSlabAllocator,
) {
    if let Err(err) = background_thread(to_thread_r, from_thread_s, uri, frame_stack_size, shm) {
        log::error!("background_thread err'd: {}", err.to_string());
        // NOTE: `shm` is dropped in case of an error, so anyone who tries to connect afterwards
        // will get an error
        from_thread_s
            .send(ResultMsg::Error {
                msg: err.to_string(),
            })
            .unwrap();
    }
}

fn drain_if_mismatch(
    msg: &mut Message,
    socket: &Socket,
    series: u64,
    control_channel: &Receiver<ControlMsg>,
) -> Result<(), AcquisitionError> {
    loop {
        let series_res: Result<DSeriesOnly, _> = serde_json::from_str(msg.as_str().unwrap());

        if let Ok(recvd_series) = series_res {
            // everything is ok, we can go ahead:
            if recvd_series.series == series {
                return Ok(());
            }
        }

        debug!(
            "drained message header: {} expected series {}",
            msg.as_str().unwrap(),
            series
        );

        // throw away message parts that are part of the mismatched message:
        while msg.get_more() {
            recv_part(msg, socket, control_channel)?;

            if let Some(msg_str) = msg.as_str() {
                debug!("drained message part: {}", msg_str);
            } else {
                debug!("drained non-utf message part");
            }
        }

        // receive the next message:
        recv_part(msg, socket, control_channel)?;
    }
}

fn background_thread(
    to_thread_r: &Receiver<ControlMsg>,
    from_thread_s: &Sender<ResultMsg>,
    uri: String,
    frame_stack_size: usize,
    mut shm: SharedSlabAllocator,
) -> Result<(), AcquisitionError> {
    let ctx = zmq::Context::new();
    let socket = ctx.socket(zmq::PULL).unwrap();
    socket.set_rcvtimeo(1000).unwrap();
    socket.connect(&uri).unwrap();
    socket.set_rcvhwm(4 * 1024).unwrap();

    setup_monitor(ctx, "DectrisReceiver".to_string(), &socket);

    loop {
        // control: main threads tells us to quit
        let control = to_thread_r.recv_timeout(Duration::from_millis(100));
        match control {
            Ok(ControlMsg::StartAcquisitionPassive) => {
                match passive_acquisition(
                    to_thread_r,
                    from_thread_s,
                    &socket,
                    frame_stack_size,
                    &mut shm,
                ) {
                    Ok(_) => {}
                    Err(AcquisitionError::Disconnected | AcquisitionError::Cancelled) => {
                        return Ok(());
                    }
                    e => {
                        return e;
                    }
                }
            }
            Ok(ControlMsg::StartAcquisition { series }) => {
                let mut msg: Message = Message::new();
                recv_part(&mut msg, &socket, to_thread_r)?;

                drain_if_mismatch(&mut msg, &socket, series, to_thread_r)?;

                let dheader_res: Result<DHeader, _> = serde_json::from_str(msg.as_str().unwrap());
                let dheader: DHeader = match dheader_res {
                    Ok(header) => header,
                    Err(err) => {
                        from_thread_s
                            .send(ResultMsg::SerdeError {
                                msg: err.to_string(),
                                recvd_msg: msg
                                    .as_str()
                                    .map_or_else(|| "".to_string(), |m| m.to_string()),
                            })
                            .unwrap();
                        log::error!(
                            "background_thread: serialization error: {}",
                            err.to_string()
                        );
                        break;
                    }
                };
                debug!("dheader: {dheader:?}");

                // second message: the header itself
                recv_part(&mut msg, &socket, to_thread_r)?;
                let detector_config: DetectorConfig =
                    serde_json::from_str(msg.as_str().unwrap()).unwrap();

                match acquisition(
                    detector_config,
                    to_thread_r,
                    from_thread_s,
                    &socket,
                    series,
                    frame_stack_size,
                    &mut shm,
                ) {
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
                debug!("background_thread: got a StopThread message");
                break;
            }
            Err(RecvTimeoutError::Disconnected) => {
                debug!("background_thread: control channel has disconnected");
                break;
            }
            Err(RecvTimeoutError::Timeout) => (), // no message, nothing to do
        }
    }
    debug!("background_thread: is done");
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

/// Start a background thread that received data from the zeromq socket and
/// puts it into shared memory.
pub struct DectrisReceiver {
    bg_thread: Option<JoinHandle<()>>,
    to_thread: Sender<ControlMsg>,
    from_thread: Receiver<ResultMsg>,
    pub status: ReceiverStatus,
    pub shm_handle: SHMHandle,
}

impl DectrisReceiver {
    pub fn new(uri: &str, frame_stack_size: usize, shm: SharedSlabAllocator) -> Self {
        let (to_thread_s, to_thread_r) = unbounded();
        let (from_thread_s, from_thread_r) = unbounded();

        let builder = std::thread::Builder::new();
        let uri = uri.to_string();

        let shm_handle = shm.get_handle();

        DectrisReceiver {
            bg_thread: Some(
                builder
                    .name("bg_thread".to_string())
                    .spawn(move || {
                        background_thread_wrap(
                            &to_thread_r,
                            &from_thread_s,
                            uri.to_string(),
                            frame_stack_size,
                            shm,
                        )
                    })
                    .expect("failed to start background thread"),
            ),
            from_thread: from_thread_r,
            to_thread: to_thread_s,
            status: ReceiverStatus::Idle,
            shm_handle,
        }
    }

    fn adjust_status(&mut self, msg: &ResultMsg) {
        match msg {
            ResultMsg::AcquisitionStart { .. } => {
                self.status = ReceiverStatus::Running;
            }
            ResultMsg::End { .. } => {
                self.status = ReceiverStatus::Idle;
            }
            _ => {}
        }
    }

    pub fn recv(&mut self) -> ResultMsg {
        let result_msg = self
            .from_thread
            .recv()
            .expect("background thread should be running");
        self.adjust_status(&result_msg);
        result_msg
    }

    pub fn next_timeout(&mut self, timeout: Duration) -> Option<ResultMsg> {
        let result_msg = self.from_thread.recv_timeout(timeout);

        match result_msg {
            Ok(result) => {
                self.adjust_status(&result);
                Some(result)
            }
            Err(e) => match e {
                RecvTimeoutError::Disconnected => {
                    panic!("background thread should be running")
                }
                RecvTimeoutError::Timeout => None,
            },
        }
    }

    pub fn start_series(&mut self, series: u64) -> Result<(), ReceiverError> {
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

    pub fn start_passive(&mut self) -> Result<(), ReceiverError> {
        if self.status == ReceiverStatus::Closed {
            return Err(ReceiverError {
                msg: "receiver is closed".to_string(),
            });
        }
        self.to_thread
            .send(ControlMsg::StartAcquisitionPassive)
            .expect("background thread should be running");
        self.status = ReceiverStatus::Running;
        Ok(())
    }

    pub fn close(&mut self) {
        if self.to_thread.send(ControlMsg::StopThread).is_err() {
            warn!("could not stop background thread, probably already dead");
        }
        if let Some(join_handle) = self.bg_thread.take() {
            join_handle
                .join()
                .expect("could not join background thread!");
        } else {
            warn!("did not have a bg thread join handle, cannot join!");
        }
        self.status = ReceiverStatus::Closed;
    }
}

impl Default for DectrisReceiver {
    fn default() -> Self {
        let shm = SharedSlabAllocator::new(5000, 512 * 512 * 2, true).expect("create shm");
        Self::new("tcp://127.0.0.1:9999", 1, shm)
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
