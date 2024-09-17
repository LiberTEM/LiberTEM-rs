use std::{
    convert::Infallible,
    fmt::Display,
    mem::replace,
    ops::Deref,
    sync::mpsc::{channel, Receiver, RecvTimeoutError, SendError, Sender, TryRecvError},
    thread::JoinHandle,
    time::{Duration, Instant},
};

use common::{
    background_thread::{self, BackgroundThread, ControlMsg, ReceiverMsg},
    frame_stack::{FrameStackForWriting, FrameStackWriteError},
    generic_connection::DetectorConnectionConfig,
};
use ipc_test::{slab::ShmError, SharedSlabAllocator};
use log::{debug, error, info, trace, warn};
use zmq::{Message, Socket};

use crate::base_types::{
    setup_monitor, DConfig, DHeader, DImage, DImageD, DSeriesAndType, DSeriesEnd, DectrisFrameMeta,
    DectrisPendingAcquisition, DetectorConfig,
};

type DectrisControlMsg = ControlMsg<DectrisExtraControl>;

/// Receive a message into `msg`, and periodically check for control messages on
/// `control_channel` which are converted into `AcquisitionError`s. Return once
/// a message has been read into `msg`.
fn recv_part(
    msg: &mut Message,
    socket: &Socket,
    control_channel: &Receiver<DectrisControlMsg>,
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
    control_channel: &Receiver<DectrisControlMsg>,
    msg: &mut Message,
    msg_image: &mut Message,
) -> Result<(DImage, DImageD, DConfig), AcquisitionError> {
    recv_part(msg, socket, control_channel)?;
    let dimage_res: Result<DImage, _> = serde_json::from_str(msg.as_str().unwrap());
    let dimage = dimage_res.map_err(|err| AcquisitionError::serde_from_msg(&err, msg))?;

    recv_part(msg, socket, control_channel)?;
    let dimaged_res: Result<DImageD, _> = serde_json::from_str(msg.as_str().unwrap());
    let dimaged = dimaged_res.map_err(|err| AcquisitionError::serde_from_msg(&err, msg))?;

    // compressed image data:
    recv_part(msg_image, socket, control_channel)?;

    // DConfig:
    recv_part(msg, socket, control_channel)?;
    let dconfig: DConfig = serde_json::from_str(msg.as_str().unwrap())
        .map_err(|err| AcquisitionError::serde_from_msg(&err, msg))?;

    Ok((dimage, dimaged, dconfig))
}

#[derive(Debug, Clone, thiserror::Error)]
enum AcquisitionError {
    Disconnected,
    SeriesMismatch,
    FrameIdMismatch { expected_id: u64, got_id: u64 },
    SerdeError { recvd_msg: String, msg: String },
    StopThread,
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
            AcquisitionError::StopThread => {
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
            AcquisitionError::Cancelled => {
                write!(f, "acquisition cancelled")
            }
        }
    }
}

impl<T> From<SendError<T>> for AcquisitionError {
    fn from(_value: SendError<T>) -> Self {
        AcquisitionError::Disconnected
    }
}

impl AcquisitionError {
    fn serde_from_msg(err: &serde_json::Error, msg: &Message) -> Self {
        Self::SerdeError {
            msg: err.to_string(),
            recvd_msg: msg
                .as_str()
                .map_or_else(|| "".to_string(), |m| m.to_string()),
        }
    }
}

/// With a running acquisition, check for control messages;
/// especially convert `ControlMsg::StopThread` to `AcquisitionError::Cancelled`.
fn check_for_control(
    control_channel: &Receiver<DectrisControlMsg>,
) -> Result<(), AcquisitionError> {
    match control_channel.try_recv() {
        Ok(ControlMsg::SpecializedControlMsg {
            msg: DectrisExtraControl::StartAcquisitionWithSeries { series: _ },
        }) => Err(AcquisitionError::StateError {
            msg: "received StartAcquisition while an acquisition was already running".to_string(),
        }),
        Ok(ControlMsg::StartAcquisitionPassive) => Err(AcquisitionError::StateError {
            msg: "received StartAcquisitionPassive while an acquisition was already running"
                .to_string(),
        }),
        Ok(ControlMsg::StopThread) => {
            debug!("check_for_control: StopThread received");

            Err(AcquisitionError::StopThread)
        }
        Ok(ControlMsg::CancelAcquisition) => {
            debug!("check_for_control: CancelAcquisition");

            Err(AcquisitionError::Cancelled)
        }
        Err(TryRecvError::Disconnected) => {
            debug!("check_for_control: Disconnected");

            Err(AcquisitionError::StopThread)
        }
        Err(TryRecvError::Empty) => Ok(()),
    }
}

fn serialization_error(
    from_thread_s: &Sender<ReceiverMsg<DectrisFrameMeta, DectrisPendingAcquisition>>,
    msg: &Message,
    err: &serde_json::Error,
) {
    log::error!(
        "background_thread: serialization error: {}",
        err.to_string()
    );
    from_thread_s
        .send(ReceiverMsg::FatalError {
            error: Box::new(AcquisitionError::SerdeError {
                recvd_msg: msg
                    .as_str()
                    .map_or_else(|| "".to_string(), |m| m.to_string()),
                msg: err.to_string(),
            }),
        })
        .unwrap();
}

fn make_frame_stack(
    shm: &mut SharedSlabAllocator,
    capacity: usize,
    bytes_per_frame: usize,
    to_thread_r: &Receiver<DectrisControlMsg>,
) -> Result<FrameStackForWriting<DectrisFrameMeta>, AcquisitionError> {
    loop {
        // keep some slots free for splitting frame stacks
        if shm.num_slots_free() < 3 && shm.num_slots_total() >= 3 {
            trace!("shm is almost full; waiting and creating backpressure...");
            check_for_control(to_thread_r)?;
            std::thread::sleep(Duration::from_millis(1));
            continue;
        }

        match shm.try_get_mut() {
            Ok(slot) => return Ok(FrameStackForWriting::new(slot, capacity, bytes_per_frame)),
            Err(ShmError::NoSlotAvailable) => {
                trace!("shm is full; waiting and creating backpressure...");
                check_for_control(to_thread_r)?;
                std::thread::sleep(Duration::from_millis(1));
                continue;
            }
        }
    }
}

/// Passively listen for global acquisition headers
/// and automatically latch on to them.
fn passive_acquisition(
    control_channel: &Receiver<DectrisControlMsg>,
    from_thread_s: &Sender<ReceiverMsg<DectrisFrameMeta, DectrisPendingAcquisition>>,
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

            if dheader.header_detail.deref() == "none" {
                return Err(AcquisitionError::ConfigurationError {
                    msg: "header_detail must be 'basic' or 'all', is 'none'".to_string(),
                });
            }

            // second message: the header itself
            recv_part(&mut msg, socket, control_channel)?;

            let detector_config: DetectorConfig = if let Some(msg_str) = msg.as_str() {
                debug!("detector config: {}", msg_str);
                match serde_json::from_str(msg_str) {
                    Ok(header) => header,
                    Err(err) => {
                        serialization_error(from_thread_s, &msg, &err);
                        continue;
                    }
                }
            } else {
                warn!("non-string received as detector config! ignoring message.");
                continue;
            };

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
    to_thread_r: &Receiver<DectrisControlMsg>,
    from_thread_s: &Sender<ReceiverMsg<DectrisFrameMeta, DectrisPendingAcquisition>>,
    socket: &Socket,
    series: u64,
    frame_stack_size: usize,
    shm: &mut SharedSlabAllocator,
) -> Result<(), AcquisitionError> {
    let t0 = Instant::now();
    let mut last_control_check = Instant::now();

    let mut expected_frame_id = 0;

    match from_thread_s.send(ReceiverMsg::AcquisitionStart {
        pending_acquisition: DectrisPendingAcquisition::new(detector_config.clone(), series),
    }) {
        Ok(_) => (),
        Err(SendError(_)) => return Err(AcquisitionError::Disconnected),
    }

    debug!("acquisition starting");

    // approx uppper bound of image size in bytes
    let approx_size_bytes = detector_config.get_num_pixels() as usize
        * (detector_config.bit_depth_image as f32 / 8.0f32).ceil() as usize;

    let mut frame_stack = make_frame_stack(shm, frame_stack_size, approx_size_bytes, to_thread_r)?;

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
                let new_frame_stack =
                    make_frame_stack(shm, frame_stack_size, approx_size_bytes, to_thread_r)?;
                let old_frame_stack = replace(&mut frame_stack, new_frame_stack);
                old_frame_stack.writing_done(shm)
            };
            match handle {
                Ok(frame_stack) => {
                    from_thread_s.send(ReceiverMsg::FrameStack { frame_stack })?;
                }
                Err(FrameStackWriteError::Empty) => {
                    warn!("acquisition: unexpected empty frame stack")
                }
                Err(FrameStackWriteError::NonEmpty) => {
                    warn!("acquisition: unexpected non-empty frame stack")
                }
                Err(FrameStackWriteError::TooSmall) => {
                    warn!("acquisition: frame stack too small")
                }
            }
        }

        let meta = DectrisFrameMeta {
            dimage,
            dimaged,
            dconfig,
            data_length_bytes: msg_image.len(),
        };

        frame_stack
            .write_frame(&meta, |buf| {
                buf.copy_from_slice(&msg_image);
                Ok::<_, Infallible>(())
            })
            .unwrap();

        expected_frame_id += 1;

        // we will be done after this frame:
        let done = meta.dimage.frame == detector_config.get_num_images() - 1;

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
            match handle {
                Ok(frame_stack) => from_thread_s.send(ReceiverMsg::Finished { frame_stack })?,
                Err(FrameStackWriteError::Empty) => {
                    warn!("acquisition: unexpected empty frame stack")
                }
                Err(FrameStackWriteError::NonEmpty) => {
                    warn!("acquisition: unexpected non-empty frame stack")
                }
                Err(FrameStackWriteError::TooSmall) => {
                    warn!("acquisition: frame stack too small")
                }
            }

            return Ok(());
        }
    }
}

/// convert `AcquisitionError`s to messages on `from_threads_s`
fn background_thread_wrap(
    to_thread_r: &Receiver<DectrisControlMsg>,
    from_thread_s: &Sender<ReceiverMsg<DectrisFrameMeta, DectrisPendingAcquisition>>,
    uri: String,
    frame_stack_size: usize,
    shm: SharedSlabAllocator,
) {
    if let Err(err) = background_thread(to_thread_r, from_thread_s, uri, frame_stack_size, shm) {
        log::error!("background_thread err'd: {}", err.to_string());
        // NOTE: `shm` is dropped in case of an error, so anyone who tries to connect afterwards
        // will get an error
        from_thread_s
            .send(ReceiverMsg::FatalError {
                error: Box::new(err),
            })
            .unwrap();
    }
    info!("background_thread_wrap: done");
}

fn drain_if_mismatch(
    msg: &mut Message,
    socket: &Socket,
    series: u64,
    control_channel: &Receiver<DectrisControlMsg>,
) -> Result<(), AcquisitionError> {
    loop {
        let series_res: Result<DSeriesAndType, _> = serde_json::from_str(msg.as_str().unwrap());

        if let Ok(recvd_series) = series_res {
            // everything is ok, we can go ahead:
            if recvd_series.series == series && recvd_series.htype.deref() == "dheader-1.0" {
                return Ok(());
            }
        }

        trace!(
            "drained message header: {} expected series {}",
            msg.as_str().unwrap(),
            series
        );

        // throw away message parts that are part of the mismatched message:
        while msg.get_more() {
            recv_part(msg, socket, control_channel)?;

            if let Some(msg_str) = msg.as_str() {
                trace!("drained message part: {}", msg_str);
            } else {
                trace!("drained non-utf message part");
            }
        }

        // receive the next message:
        recv_part(msg, socket, control_channel)?;
    }
}

fn background_thread(
    to_thread_r: &Receiver<DectrisControlMsg>,
    from_thread_s: &Sender<ReceiverMsg<DectrisFrameMeta, DectrisPendingAcquisition>>,
    uri: String,
    frame_stack_size: usize,
    mut shm: SharedSlabAllocator,
) -> Result<(), AcquisitionError> {
    'outer: loop {
        let thread_id = std::thread::current().id();
        info!("background_thread: connecting to {uri} ({thread_id:?})");
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
                    from_thread_s.send(ReceiverMsg::ReceiverArmed).unwrap();
                    match passive_acquisition(
                        to_thread_r,
                        from_thread_s,
                        &socket,
                        frame_stack_size,
                        &mut shm,
                    ) {
                        Ok(_) => {}
                        Err(AcquisitionError::Cancelled) => {
                            from_thread_s.send(ReceiverMsg::Cancelled).unwrap();
                            continue 'outer;
                        }
                        Err(
                            msg @ (AcquisitionError::Disconnected | AcquisitionError::StopThread),
                        ) => {
                            debug!("background_thread: passive_acquisition returned {msg:?}");
                            break 'outer;
                        }
                        Err(e) => {
                            from_thread_s
                                .send(ReceiverMsg::FatalError {
                                    error: Box::new(e.clone()),
                                })
                                .unwrap();
                            error!("background_thread: error: {}; re-connecting", e);
                            continue 'outer;
                        }
                    }
                }
                Ok(ControlMsg::SpecializedControlMsg {
                    msg: DectrisExtraControl::StartAcquisitionWithSeries { series },
                }) => {
                    from_thread_s.send(ReceiverMsg::ReceiverArmed).unwrap();

                    let mut msg: Message = Message::new();
                    recv_part(&mut msg, &socket, to_thread_r)?;

                    drain_if_mismatch(&mut msg, &socket, series, to_thread_r)?;

                    let dheader_res: Result<DHeader, _> =
                        serde_json::from_str(msg.as_str().unwrap());
                    let dheader: DHeader = match dheader_res {
                        Ok(header) => header,
                        Err(err) => {
                            serialization_error(from_thread_s, &msg, &err);
                            continue 'outer;
                        }
                    };
                    debug!("dheader: {dheader:?}");

                    // second message: the header itself
                    recv_part(&mut msg, &socket, to_thread_r)?;

                    let detector_config: DetectorConfig = if let Some(msg_str) = msg.as_str() {
                        debug!("detector config: {}", msg_str);
                        match serde_json::from_str(msg_str) {
                            Ok(header) => header,
                            Err(err) => {
                                serialization_error(from_thread_s, &msg, &err);
                                continue 'outer;
                            }
                        }
                    } else {
                        warn!("non-string received as detector config! re-connecting");
                        continue 'outer;
                    };

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
                        Err(AcquisitionError::Cancelled) => {
                            from_thread_s.send(ReceiverMsg::Cancelled).unwrap();
                        }
                        Err(AcquisitionError::Disconnected | AcquisitionError::StopThread) => {
                            break 'outer;
                        }
                        Err(e) => {
                            from_thread_s
                                .send(ReceiverMsg::FatalError {
                                    error: Box::new(e.clone()),
                                })
                                .unwrap();
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
                    warn!("background_thread: control channel has disconnected");
                    break 'outer;
                }
                Ok(ControlMsg::CancelAcquisition) => {
                    warn!("background_thread: ignoring ControlMsg::CancelAcquisition outside of running acquisition")
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

#[derive(Debug)]
pub enum DectrisExtraControl {
    StartAcquisitionWithSeries { series: u64 },
}

pub struct DectrisBackgroundThread {
    bg_thread: JoinHandle<()>,
    to_thread: Sender<ControlMsg<DectrisExtraControl>>,
    from_thread: Receiver<ReceiverMsg<DectrisFrameMeta, DectrisPendingAcquisition>>,
}

impl BackgroundThread for DectrisBackgroundThread {
    type FrameMetaImpl = DectrisFrameMeta;
    type AcquisitionConfigImpl = DectrisPendingAcquisition;
    type ExtraControl = DectrisExtraControl;

    fn channel_to_thread(
        &mut self,
    ) -> &mut Sender<background_thread::ControlMsg<Self::ExtraControl>> {
        &mut self.to_thread
    }

    fn channel_from_thread(
        &mut self,
    ) -> &mut Receiver<
        background_thread::ReceiverMsg<Self::FrameMetaImpl, Self::AcquisitionConfigImpl>,
    > {
        &mut self.from_thread
    }

    fn join(self) {
        if let Err(e) = self.bg_thread.join() {
            // FIXME: should we have an error boundary here instead and stop the panic?
            std::panic::resume_unwind(e)
        }
    }
}

impl DectrisBackgroundThread {
    pub fn spawn(
        config: &DectrisDetectorConnConfig,
        shm: &SharedSlabAllocator,
    ) -> Result<Self, background_thread::BackgroundThreadSpawnError>
    where
        Self: std::marker::Sized,
    {
        let (to_thread_s, to_thread_r) = channel();
        let (from_thread_s, from_thread_r) = channel();
        let builder = std::thread::Builder::new();

        let shm = shm.clone_and_connect()?;

        let config = config.clone();

        Ok(Self {
            bg_thread: builder
                .name("bg_thread".to_string())
                .spawn(move || {
                    background_thread_wrap(
                        &to_thread_r,
                        &from_thread_s,
                        config.uri.clone(),
                        config.frame_stack_size,
                        shm,
                    )
                })
                .expect("failed to start background thread"),
            to_thread: to_thread_s,
            from_thread: from_thread_r,
        })
    }
}

#[derive(Clone, Debug)]
pub struct DectrisDetectorConnConfig {
    /// zmq URI of the DCU
    pub uri: String,

    /// number of frames per frame stack; approximated because of compression
    pub frame_stack_size: usize,

    /// approx. number of bytes per frame, used for sizing frame stacks together
    /// with `frame_stack_size`
    pub bytes_per_frame: usize,

    num_slots: usize,
    enable_huge_pages: bool,
    shm_handle_path: String,
}

impl DectrisDetectorConnConfig {
    pub fn new(
        uri: &str,
        frame_stack_size: usize,
        bytes_per_frame: usize,
        num_slots: usize,
        enable_huge_pages: bool,
        shm_handle_path: &str,
    ) -> Self {
        Self {
            uri: uri.to_owned(),
            frame_stack_size,
            bytes_per_frame,
            num_slots,
            enable_huge_pages,
            shm_handle_path: shm_handle_path.to_owned(),
        }
    }
}

impl DetectorConnectionConfig for DectrisDetectorConnConfig {
    fn get_shm_num_slots(&self) -> usize {
        self.num_slots
    }

    fn get_shm_slot_size(&self) -> usize {
        self.frame_stack_size * self.bytes_per_frame
    }

    fn get_shm_enable_huge_pages(&self) -> bool {
        self.enable_huge_pages
    }

    fn get_shm_handle_path(&self) -> String {
        self.shm_handle_path.clone()
    }
}
