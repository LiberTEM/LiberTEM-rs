use std::{
    io::ErrorKind,
    net::TcpStream,
    sync::{
        mpsc::{channel, Receiver, RecvTimeoutError, SendError, Sender, TryRecvError},
        Mutex,
    },
    thread::JoinHandle,
    time::{Duration, Instant},
};

use common::{
    background_thread::{BackgroundThread, BackgroundThreadSpawnError, ControlMsg, ReceiverMsg},
    frame_stack::{FrameStackForWriting, FrameStackWriteError},
    tcp::{self, ReadExactError},
    utils::{num_from_byte_slice, three_way_shift, NumParseError},
};
use ipc_test::{slab::ShmError, SharedSlabAllocator};
use log::{debug, error, info, trace, warn};
use opentelemetry::Context;
use serval_client::{DetectorConfig, ServalClient, ServalError};

use crate::base_types::{ASIMpxDetectorConnConfig, ASIMpxFrameMeta, DType, PendingAcquisition};

type ASIMpxControlMsg = ControlMsg<()>;

#[derive(Debug, thiserror::Error)]
enum ParseError {
    #[error("wrong magic, should be 'P5', got: {got:X?}")]
    WrongMagic { got: [u8; 2] },

    #[error("invalid max val, should be in 0..65536, is {got}")]
    InvalidMaxVal { got: u32 },

    #[error("expected whitespace at {pos}, got byte {got:x} instead")]
    WhiteSpaceExpected { pos: usize, got: u8 },

    #[error("failed to parse number: {err}")]
    Num {
        #[from]
        err: NumParseError,
    },
}

#[derive(PartialEq, Eq)]
pub enum ReceiverStatus {
    Idle,
    Running,
    Closed,
}

/// Peek and parse the first frame header. Retries until either a header was
/// received or a control message arrives in the control channel.
fn peek_header(
    stream: &mut TcpStream,
    control_channel: &Receiver<ASIMpxControlMsg>,
) -> Result<ASIMpxFrameMeta, AcquisitionError> {
    let mut buf: [u8; HEADER_BUF_SIZE] = [0; HEADER_BUF_SIZE];

    // Wait until enough data is in the buffer.
    // Possibly, the sender sends the header and payload separately,
    // in which case we get only a short header, and we need to retry.
    // All because we don't really know how large the header is supposed to be.
    // This is broken for very small frames (where header+data < 512),
    // so if an acquisition only contains <512 bytes in total, we will wait
    // here indefinitely.

    loop {
        match tcp::peek_exact_interruptible(stream, &mut buf, Duration::from_millis(10), 10, || {
            check_for_control(control_channel)
        }) {
            Ok(_) => {
                return Ok(parse_header(&buf, 0)?);
            }
            Err(ReadExactError::PeekError { size }) => {
                trace!("peek  of {size} bytes failed; retrying...");
                continue;
            }
            Err(e) => {
                return Err(AcquisitionError::from(e));
            }
        }
    }
}

const HEADER_BUF_SIZE: usize = 512;

fn parse_header(buf: &[u8; HEADER_BUF_SIZE], sequence: u64) -> Result<ASIMpxFrameMeta, ParseError> {
    let mut pos: usize = 0;

    // Each PGM image consists of the following:
    // •      A "magic number" for identifying the file type.  A pgm image's magic number is the two characters "P5".
    // FIXME: error handling
    if &buf[0..2] != b"P5" {
        return Err(ParseError::WrongMagic {
            got: [buf[0], buf[1]],
        });
    }

    pos += 2;

    // •      Whitespace (blanks, TABs, CRs, LFs).
    while buf[pos].is_ascii_whitespace() && pos < HEADER_BUF_SIZE {
        pos += 1;
    }

    // •      A width, formatted as ASCII characters in decimal.
    let width_start = pos;
    while !buf[pos].is_ascii_whitespace() && pos < HEADER_BUF_SIZE {
        pos += 1;
    }
    let width: u16 = num_from_byte_slice(&buf[width_start..pos])?;

    // •      Whitespace.
    while buf[pos].is_ascii_whitespace() && pos < HEADER_BUF_SIZE {
        pos += 1;
    }

    // •      A height, again in ASCII decimal.
    let height_start = pos;
    while !buf[pos].is_ascii_whitespace() && pos < HEADER_BUF_SIZE {
        pos += 1;
    }
    let height: u16 = num_from_byte_slice(&buf[height_start..pos])?;

    // •      Whitespace.
    while buf[pos].is_ascii_whitespace() && pos < HEADER_BUF_SIZE {
        pos += 1;
    }

    // •      The maximum gray value (Maxval), again in ASCII decimal.  Must be less than 65536, and more than zero.
    // really, more than zero? how do you represent an all-black image in PGM then?
    let maxval_start = pos;
    while !buf[pos].is_ascii_whitespace() && pos < HEADER_BUF_SIZE {
        pos += 1;
    }
    let maxval: u32 = num_from_byte_slice(&buf[maxval_start..pos])?;

    if !(0..65536).contains(&maxval) {
        return Err(ParseError::InvalidMaxVal { got: maxval });
    }

    // •      A single whitespace character (usually a newline).
    if !buf[pos].is_ascii_whitespace() {
        return Err(ParseError::WhiteSpaceExpected { pos, got: buf[pos] });
    }

    // •      A raster of Height rows, [...]
    let _raster_start_pos = pos;

    let dtype: DType = DType::from_maxval(maxval);
    let data_length_bytes = width as usize * height as usize * dtype.num_bytes();

    let header_length_bytes: usize = pos + 1;

    let meta = ASIMpxFrameMeta {
        sequence,
        dtype,
        width,
        height,
        data_length_bytes,
        header_length_bytes,
    };

    trace!("frame header parsed: {meta:?}");

    Ok(meta)
}

fn recv_frame(
    sequence: u64,
    stream: &mut TcpStream,
    control_channel: &Receiver<ASIMpxControlMsg>,
    frame_stack: &mut FrameStackForWriting<ASIMpxFrameMeta>,
    extra_frame_stack: &mut FrameStackForWriting<ASIMpxFrameMeta>,
) -> Result<ASIMpxFrameMeta, AcquisitionError> {
    // TODO:
    // - timeout handling
    // - error handling (parsing, receiving)

    // 1) Read the first N bytes (512?) from the socket into a buffer, and parse the PGM header

    let mut buf: [u8; HEADER_BUF_SIZE] = [0; HEADER_BUF_SIZE];

    tcp::read_exact_interruptible(stream, &mut buf, || check_for_control(control_channel))?;

    // 2) Parse the header, importantly reading width, height, bytes-per-pixel (maxval)

    let meta = parse_header(&buf, sequence)?;

    // 3) Now we know how large the binary part is. Copy the rest of the buffer
    //    into the frame stack, and read the remaining bytes directly into the
    //    shared memory.

    let fs = if frame_stack.can_fit(meta.data_length_bytes) {
        frame_stack
    } else {
        trace!(
            "frame_stack can't fit this frame: {} {}",
            frame_stack.bytes_free(),
            meta.data_length_bytes
        );
        if !extra_frame_stack.is_empty() {
            return Err(AcquisitionError::StateError {
                msg: "extra_frame_stack should be empty".to_string(),
            });
        }
        if !extra_frame_stack.can_fit(meta.data_length_bytes) {
            return Err(AcquisitionError::ConfigurationError {
                msg: format!(
                    "extra_frame_stack can't fit frame; frame size {}, frame stack size {}",
                    meta.data_length_bytes,
                    extra_frame_stack.slot_size()
                ),
            });
        }
        extra_frame_stack
    };

    fs.write_frame(&meta, |dest_buf| {
        // copy the data after the header from our temporary stack buffer:
        let head_src = &buf[meta.header_length_bytes..];
        dest_buf[0..head_src.len()].copy_from_slice(head_src);

        let dest_rest = &mut dest_buf[head_src.len()..];

        tcp::read_exact_interruptible(stream, dest_rest, || check_for_control(control_channel))?;

        Ok::<_, AcquisitionError>(())
    })?;

    Ok(meta)
}

#[derive(Debug, Clone, thiserror::Error)]
enum AcquisitionError {
    #[error("other end has disconnected")]
    Disconnected,

    #[error("background thread stopped")]
    StopThread,

    #[error("acquisition cancelled by user")]
    Cancelled,

    #[error("shm buffer is full")]
    BufferFull,

    #[error("state error: {msg}")]
    StateError { msg: String },

    #[error("configuration error: {msg}")]
    ConfigurationError { msg: String },

    #[error("parse error: {msg}")]
    ParseError { msg: String },

    #[error("connection error: {msg}")]
    ConnectionError { msg: String },

    #[error("serval HTTP API error: {msg}")]
    APIError { msg: String },

    #[error("error writing to shm: {0}")]
    WriteError(#[from] FrameStackWriteError),

    #[error("shm access error: {0}")]
    ShmError(#[from] ShmError),
}

impl From<ParseError> for AcquisitionError {
    fn from(value: ParseError) -> Self {
        AcquisitionError::ParseError {
            msg: format!("{:?}", value),
        }
    }
}

impl<T> From<SendError<T>> for AcquisitionError {
    fn from(_value: SendError<T>) -> Self {
        AcquisitionError::Disconnected
    }
}

impl From<ServalError> for AcquisitionError {
    fn from(value: ServalError) -> Self {
        Self::APIError {
            msg: value.to_string(),
        }
    }
}

impl From<ReadExactError<AcquisitionError>> for AcquisitionError {
    fn from(value: ReadExactError<AcquisitionError>) -> Self {
        match value {
            ReadExactError::Interrupted { size, err } => {
                warn!("interrupted read after {size} bytes; discarding buffer");
                err
            }
            ReadExactError::IOError { err } => Self::from(err),
            ReadExactError::PeekError { size } => Self::ConnectionError {
                msg: format!("could not peek {size} bytes"),
            },
            ReadExactError::Eof => Self::ConnectionError {
                msg: "EOF".to_owned(),
            },
        }
    }
}

impl From<std::io::Error> for AcquisitionError {
    fn from(value: std::io::Error) -> Self {
        Self::ConnectionError {
            msg: format!("i/o error: {value}"),
        }
    }
}

/// With a running acquisition, check for control messages;
/// especially convert `ControlMsg::StopThread` to `AcquisitionError::Cancelled`.
fn check_for_control(control_channel: &Receiver<ASIMpxControlMsg>) -> Result<(), AcquisitionError> {
    match control_channel.try_recv() {
        Ok(ControlMsg::StartAcquisitionPassive) => Err(AcquisitionError::StateError {
            msg: "received StartAcquisitionPassive while an acquisition was already running"
                .to_string(),
        }),
        Ok(ControlMsg::StopThread) => Err(AcquisitionError::StopThread),
        Ok(ControlMsg::SpecializedControlMsg { msg: _ }) => {
            panic!("unsupported SpecializedControlMsg")
        }
        Ok(ControlMsg::CancelAcquisition) => Err(AcquisitionError::Cancelled),
        Err(TryRecvError::Disconnected) => Err(AcquisitionError::Disconnected),
        Err(TryRecvError::Empty) => Ok(()),
    }
}

/// Passively listen for the start of an acquisition
/// and automatically latch on to it.
fn passive_acquisition(
    control_channel: &Receiver<ASIMpxControlMsg>,
    from_thread_s: &Sender<ReceiverMsg<ASIMpxFrameMeta, PendingAcquisition>>,
    frame_stack_size: usize,
    data_uri: &str,
    api_uri: &str,
    shm: &mut SharedSlabAllocator,
) -> Result<(), AcquisitionError> {
    let client = ServalClient::new(api_uri);

    loop {
        trace!("connecting to {data_uri}...");
        check_for_control(control_channel)?;
        let mut stream: TcpStream = match TcpStream::connect(data_uri) {
            Ok(s) => s,
            Err(e) => match e.kind() {
                ErrorKind::ConnectionRefused
                | ErrorKind::TimedOut
                | ErrorKind::ConnectionAborted
                | ErrorKind::ConnectionReset => {
                    // If we re-connect too fast after an acquisition, the
                    // connection might succeed and then be closed from the
                    // other end. That's why we have to handle Connection{Aborted,Reset}
                    // here.
                    std::thread::sleep(Duration::from_millis(10));
                    continue;
                }
                _ => return Err(AcquisitionError::ConnectionError { msg: e.to_string() }),
            },
        };

        stream.set_read_timeout(Some(Duration::from_millis(100)))?;

        // block until we get the first frame:
        let first_frame_meta = match peek_header(&mut stream, control_channel) {
            Ok(m) => m,
            Err(AcquisitionError::ConnectionError { msg }) => {
                warn!("connection error while peeking first frame: {msg}; reconnecting");
                continue;
            }
            Err(e) => return Err(e),
        };

        // then, we should be able to reliably get the detector config
        // (we assume once data arrives, the config is immutable)
        let detector_config = client.get_detector_config()?;

        acquisition(
            control_channel,
            from_thread_s,
            &detector_config,
            &first_frame_meta,
            &mut stream,
            frame_stack_size,
            shm,
        )?;

        let free = shm.num_slots_free()?;
        let total = shm.num_slots_total();
        info!("passive acquisition done; free slots: {}/{}", free, total);

        check_for_control(control_channel)?;
    }
}

fn acquisition(
    to_thread_r: &Receiver<ASIMpxControlMsg>,
    from_thread_s: &Sender<ReceiverMsg<ASIMpxFrameMeta, PendingAcquisition>>,
    detector_config: &DetectorConfig,
    first_frame_meta: &ASIMpxFrameMeta,
    stream: &mut TcpStream,
    frame_stack_size: usize,
    shm: &mut SharedSlabAllocator,
) -> Result<(), AcquisitionError> {
    let t0 = Instant::now();
    let mut last_control_check = Instant::now();

    from_thread_s.send(ReceiverMsg::AcquisitionStart {
        pending_acquisition: PendingAcquisition::new(detector_config, first_frame_meta),
    })?;

    debug!("acquisition starting");

    // approx uppper bound of image size in bytes
    let peek_meta = peek_header(stream, to_thread_r)?;
    let approx_size_bytes = 2 * peek_meta.get_size();

    let slot = match shm.get_mut() {
        None => return Err(AcquisitionError::BufferFull),
        Some(x) => x,
    };
    let mut frame_stack =
        FrameStackForWriting::new(slot, frame_stack_size, approx_size_bytes as usize);

    // in case the frame stack is full, the receiving function needs
    // an alternative destination:
    let extra_slot = match shm.get_mut() {
        None => return Err(AcquisitionError::BufferFull),
        Some(x) => x,
    };
    let mut extra_frame_stack =
        FrameStackForWriting::new(extra_slot, frame_stack_size, approx_size_bytes as usize);

    debug!("starting receive loop");

    let mut sequence = 0;

    loop {
        if last_control_check.elapsed() > Duration::from_millis(300) {
            last_control_check = Instant::now();
            check_for_control(to_thread_r)?;
            trace!("acquisition progress: sequence={sequence}");
        }

        recv_frame(
            sequence,
            stream,
            to_thread_r,
            &mut frame_stack,
            &mut extra_frame_stack,
        )?;
        sequence += 1;

        // If `recv_frame` had to use `extra_frame_stack`, `frame_stack` is
        // finished and we need to exchange the stacks:
        if !extra_frame_stack.is_empty() {
            trace!("got something in `extra_frame_stack`, swapping things around...");
            // approx. the following is happening here:
            // 1) to_send <- frame_stack
            // 2) frame_stack <- extra_frame_stack
            // 3) extra_frame_stack <- new_frame_stack()

            let to_send = {
                let slot = match shm.get_mut() {
                    None => return Err(AcquisitionError::BufferFull),
                    Some(x) => x,
                };
                let new_frame_stack =
                    FrameStackForWriting::new(slot, frame_stack_size, approx_size_bytes as usize);

                let old_frame_stack =
                    three_way_shift(&mut frame_stack, &mut extra_frame_stack, new_frame_stack);
                old_frame_stack.writing_done(shm)?
            };
            // send to our queue:
            from_thread_s.send(ReceiverMsg::FrameStack {
                frame_stack: to_send,
            })?;
        }

        // we will be done after this frame:
        let done = sequence == detector_config.n_triggers;

        if done {
            let elapsed = t0.elapsed();
            info!("done in {elapsed:?}");

            let handle = frame_stack.writing_done(shm)?;
            from_thread_s.send(ReceiverMsg::Finished {
                frame_stack: handle,
            })?;

            if !extra_frame_stack.is_empty() {
                let handle = extra_frame_stack.writing_done(shm)?;
                from_thread_s.send(ReceiverMsg::Finished {
                    frame_stack: handle,
                })?;
            } else {
                // let's not leak the `extra_frame_stack`:
                // FIXME: `FrameStackForWriting` should really free itself,
                // if `writing_done` was not called manually, which might happen
                // in case of error handling.
                // ah, but it can't, because it doesn't have a reference to `shm`! hmm
                extra_frame_stack.free_empty_frame_stack(shm)?;
            }

            return Ok(());
        }
    }
}

/// convert `AcquisitionError`s to messages on `from_threads_s`
fn background_thread_wrap(
    to_thread_r: &Receiver<ASIMpxControlMsg>,
    from_thread_s: &Sender<ReceiverMsg<ASIMpxFrameMeta, PendingAcquisition>>,
    data_uri: &str,
    api_uri: &str,
    frame_stack_size: usize,
    shm: SharedSlabAllocator,
) {
    if let Err(err) = background_thread(
        to_thread_r,
        from_thread_s,
        data_uri,
        api_uri,
        frame_stack_size,
        shm,
    ) {
        log::error!("background_thread err'd: {}", err.to_string());
        // NOTE: `shm` is dropped in case of an error, so anyone who tries to connect afterwards
        // will get an error
        from_thread_s
            .send(ReceiverMsg::FatalError {
                error: Box::new(err),
            })
            .unwrap();
    }
}

fn background_thread(
    to_thread_r: &Receiver<ASIMpxControlMsg>,
    from_thread_s: &Sender<ReceiverMsg<ASIMpxFrameMeta, PendingAcquisition>>,
    data_uri: &str,
    api_uri: &str,
    frame_stack_size: usize,
    mut shm: SharedSlabAllocator,
) -> Result<(), AcquisitionError> {
    'outer: loop {
        loop {
            // control: main threads tells us to quit
            let control = to_thread_r.recv_timeout(Duration::from_millis(100));
            match control {
                Ok(ControlMsg::CancelAcquisition) => {
                    warn!("background_thread: ControlMsg::CancelAcquisition without running acquisition");
                }
                Ok(ControlMsg::StartAcquisitionPassive) => {
                    match passive_acquisition(
                        to_thread_r,
                        from_thread_s,
                        frame_stack_size,
                        data_uri,
                        api_uri,
                        &mut shm,
                    ) {
                        Ok(_) => {}
                        Err(AcquisitionError::Cancelled) => {
                            from_thread_s.send(ReceiverMsg::Cancelled).unwrap();
                            continue 'outer;
                        }
                        e @ Err(AcquisitionError::Disconnected | AcquisitionError::StopThread) => {
                            info!("background_thread: terminating: {e:?}");
                            return Ok(());
                        }
                        Err(e) => {
                            error!("background_thread: error: {}; re-connecting", e);
                            from_thread_s
                                .send(ReceiverMsg::FatalError { error: Box::new(e) })
                                .unwrap();
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
                Ok(ControlMsg::SpecializedControlMsg { msg: _ }) => {
                    panic!("ControlMsg::SpecializesControlMsg is unused for ASI MPX3");
                }
            }
        }
    }
    debug!("background_thread: is done");
    Ok(())
}

pub struct ASIMpxBackgroundThread {
    bg_thread: JoinHandle<()>,
    to_thread: Sender<ControlMsg<()>>,
    from_thread: Mutex<Receiver<ReceiverMsg<ASIMpxFrameMeta, PendingAcquisition>>>,
}

impl BackgroundThread for ASIMpxBackgroundThread {
    type FrameMetaImpl = ASIMpxFrameMeta;
    type AcquisitionConfigImpl = PendingAcquisition;
    type ExtraControl = ();

    fn channel_to_thread(&mut self) -> &mut Sender<ControlMsg<Self::ExtraControl>> {
        &mut self.to_thread
    }

    fn channel_from_thread(
        &mut self,
    ) -> &mut Mutex<
        std::sync::mpsc::Receiver<ReceiverMsg<Self::FrameMetaImpl, Self::AcquisitionConfigImpl>>,
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

impl ASIMpxBackgroundThread {
    pub fn spawn(
        config: &ASIMpxDetectorConnConfig,
        shm: &SharedSlabAllocator,
    ) -> Result<Self, BackgroundThreadSpawnError> {
        let (to_thread_s, to_thread_r) = channel();
        let (from_thread_s, from_thread_r) = channel();

        let builder = std::thread::Builder::new();
        let data_uri = config.data_uri.to_owned();
        let api_uri = config.api_uri.to_owned();
        let shm = shm.clone_and_connect()?;
        let frame_stack_size = config.frame_stack_size;
        let ctx = Context::current();

        Ok(Self {
            bg_thread: builder
                .name("bg_thread".to_string())
                .spawn(move || {
                    let _guard = ctx.attach();
                    background_thread_wrap(
                        &to_thread_r,
                        &from_thread_s,
                        &data_uri,
                        &api_uri,
                        frame_stack_size,
                        shm,
                    )
                })
                .map_err(BackgroundThreadSpawnError::SpawnFailed)?,
            from_thread: Mutex::new(from_thread_r),
            to_thread: to_thread_s,
        })
    }
}
