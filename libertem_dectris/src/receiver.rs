use std::{
    fmt::Display,
    mem::replace,
    thread::JoinHandle,
    time::{Duration, Instant},
};

use crossbeam_channel::{unbounded, Receiver, RecvTimeoutError, SendError, Sender, TryRecvError};
use ipc_test::{SHMHandle, SharedSlabAllocator};
use log::{debug, error, info, trace, warn};
use zmq::{Message, Socket};

use crate::{
    common::{
        setup_monitor, DConfig, DHeader, DImage, DImageD, DSeriesAndType, DSeriesEnd,
        DetectorConfig,
    },
    frame_stack::{FrameStackForWriting, FrameStackHandle},
};

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
    ConfigurationError { msg: String },
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
            AcquisitionError::ConfigurationError { msg } => {
                write!(f, "configuration error: {msg}")
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

            if dheader.header_detail == "none" {
                return Err(AcquisitionError::ConfigurationError {
                    msg: "header_detail must be 'basic' or 'all', is 'none'".to_string(),
                });
            }

            // second message: the header itself
            recv_part(&mut msg, socket, control_channel)?;

            if let Some(msg_str) = msg.as_str() {
                debug!("detector config: {}", msg_str);
            } else {
                warn!("non-string received as detector config!")
            }

            let detector_config: DetectorConfig =
                serde_json::from_str(msg.as_str().unwrap()).unwrap();

            debug!("detector config: {}", msg.as_str().unwrap());

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

    debug!("acquisition starting");

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

    debug!("starting receive loop");

    loop {
        if last_control_check.elapsed() > Duration::from_millis(300) {
            last_control_check = Instant::now();
            check_for_control(to_thread_r)?;
            trace!("acquisition progress: expected_frame_id={expected_frame_id}");
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

            // FIXME: panic on timeout
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
        let series_res: Result<DSeriesAndType, _> = serde_json::from_str(msg.as_str().unwrap());

        if let Ok(recvd_series) = series_res {
            // everything is ok, we can go ahead:
            if recvd_series.series == series && recvd_series.htype == "dheader-1.0" {
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
    'outer: loop {
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
                        Err(e) => {
                            let msg = format!("passive_acquisition error: {}", e);
                            from_thread_s.send(ResultMsg::Error { msg }).unwrap();
                            error!("background_thread: error: {}; re-connecting", e);
                            continue 'outer;
                        }
                    }
                }
                Ok(ControlMsg::StartAcquisition { series }) => {
                    let mut msg: Message = Message::new();
                    recv_part(&mut msg, &socket, to_thread_r)?;

                    drain_if_mismatch(&mut msg, &socket, series, to_thread_r)?;

                    let dheader_res: Result<DHeader, _> =
                        serde_json::from_str(msg.as_str().unwrap());
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

                    if let Some(msg_str) = msg.as_str() {
                        debug!("detector config: {}", msg_str);
                    } else {
                        warn!("non-string received as detector config!")
                    }

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
                        Err(e) => {
                            let msg = format!("acquisition error: {}", e);
                            from_thread_s.send(ResultMsg::Error { msg }).unwrap();
                            error!("background_thread: error: {}; re-connecting", e);
                            continue 'outer;
                        }
                    }
                }
                Ok(ControlMsg::StopThread) => {
                    debug!("background_thread: got a StopThread message");
                    break 'outer;
                }
                Err(RecvTimeoutError::Disconnected) => {
                    debug!("background_thread: control channel has disconnected");
                    break 'outer;
                }
                Err(RecvTimeoutError::Timeout) => (), // no message, nothing to do
            }
        }
    }
    debug!("background_thread: is done");
    Ok(())
}

pub struct ReceiverError {
    pub msg: String,
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
