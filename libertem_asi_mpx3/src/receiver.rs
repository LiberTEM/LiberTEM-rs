use std::{
    fmt::{Debug, Display},
    io::{ErrorKind, Read},
    mem::replace,
    net::TcpStream,
    str::FromStr,
    thread::JoinHandle,
    time::{Duration, Instant},
};

use crossbeam_channel::{unbounded, Receiver, RecvTimeoutError, SendError, Sender, TryRecvError};
use ipc_test::{SHMHandle, SharedSlabAllocator};
use log::{debug, error, info, trace, warn};
use serval_client::{DetectorConfig, ServalClient, ServalError};

use crate::{
    common::{ASIMpxFrameMeta, DType},
    frame_stack_py::{FrameStackForWriting, FrameStackHandle},
};

#[derive(PartialEq, Debug)]
pub enum ResultMsg {
    Error {
        msg: String,
    }, // generic error response, might need to specialize later

    /// The frame header failed to parse
    ParseError {
        msg: String,
    },

    AcquisitionStart {
        detector_config: DetectorConfig,
        first_frame_meta: ASIMpxFrameMeta,
    },

    /// A stack of frames, part of an acquisition
    FrameStack {
        frame_stack: FrameStackHandle,
    },

    /// The last stack of frames of an acquisition
    /// (can possibly be empty!)
    End {
        frame_stack: FrameStackHandle,
    },
}

#[derive(Debug)]
enum ParseError {
    WrongMagic { got: [u8; 2] },
    Eof,
    InvalidMaxVal { got: u32 },
    WhiteSpaceExpected { pos: usize, got: u8 },
}

pub enum ControlMsg {
    StopThread,

    /// Wait for any acquisition to start on a given host/port
    StartAcquisitionPassive,
}

#[derive(PartialEq, Eq)]
pub enum ReceiverStatus {
    Idle,
    Running,
    Closed,
}

// FIXME: error return type
fn num_from_byte_slice<T: FromStr>(bytes: &[u8]) -> T
where
    <T as std::str::FromStr>::Err: Debug,
{
    // This should not become a bottleneck, but in case it does,
    // there is the `atoi` crate, which provides this functionality
    // without going via UTF8 first.
    let s = std::str::from_utf8(bytes).unwrap();
    s.parse().unwrap()
}

/// Peek and parse the first frame header
fn peek_header(stream: &mut TcpStream) -> Result<ASIMpxFrameMeta, AcquisitionError> {
    let mut buf: [u8; HEADER_BUF_SIZE] = [0; HEADER_BUF_SIZE];
    // FIXME: error handling, timeout, ...

    let mut nbytes = 0;

    // Ugh.. wait until enough data is in the buffer
    // possibly, the sender sends the header and payload separately,
    // in which case we get only a short header, and we need to retry.
    // All because we don't really know how large the header is supposed to be.
    // This is broken for very small frames (where header+data < 512),
    // so if an acquisition only contains <512 bytes in total, we will wait
    // here indefinitely.
    while nbytes < HEADER_BUF_SIZE {
        nbytes = match stream.peek(&mut buf) {
            Ok(n) => n,
            Err(e) => return Err(AcquisitionError::ConnectionError { msg: e.to_string() }),
        }
        // FIXME: timeout!!
    }

    Ok(parse_header(&buf, 0)?)
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
    let width: u16 = num_from_byte_slice(&buf[width_start..pos]);

    // •      Whitespace.
    while buf[pos].is_ascii_whitespace() && pos < HEADER_BUF_SIZE {
        pos += 1;
    }

    // •      A height, again in ASCII decimal.
    let height_start = pos;
    while !buf[pos].is_ascii_whitespace() && pos < HEADER_BUF_SIZE {
        pos += 1;
    }
    let height: u16 = num_from_byte_slice(&buf[height_start..pos]);

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
    let maxval: u32 = num_from_byte_slice(&buf[maxval_start..pos]);

    if !(0..65536).contains(&maxval) {
        return Err(ParseError::InvalidMaxVal { got: maxval });
    }

    // •      A single whitespace character (usually a newline).
    if !buf[pos].is_ascii_whitespace() {
        return Err(ParseError::WhiteSpaceExpected { pos, got: buf[pos] });
    }

    // •      A raster of Height rows, [...]
    let raster_start_pos = pos;

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

/// Puts `new` into `right`, `right` into `left` and returns the old `left`
fn three_way_shift<T>(left: &mut T, right: &mut T, new: T) -> T {
    let old_right = replace(right, new);
    replace(left, old_right)
}

fn recv_frame(
    sequence: u64,
    stream: &mut TcpStream,
    control_channel: &Receiver<ControlMsg>,
    frame_stack: &mut FrameStackForWriting,
    extra_frame_stack: &mut FrameStackForWriting,
) -> Result<ASIMpxFrameMeta, AcquisitionError> {
    // TODO:
    // - timeout handling
    // - error handling (parsing, receiving)

    // 1) Read the first N bytes (512?) from the socket into a buffer, and parse the PGM header

    let mut buf: [u8; HEADER_BUF_SIZE] = [0; HEADER_BUF_SIZE];

    // FIXME: need to have a timeout here!
    // In the happy case, this succeeds, or we get a
    // ConnectionReset/ConnectionAborted, but in case the network inbetween is
    // going bad, we might block here indefinitely. But we must regularly check
    // for control messages from the `control_channel`, which we can't do here
    // like this.
    match stream.read_exact(&mut buf) {
        Ok(_) => {}
        Err(e) => {
            // any kind of connection error means something is gone bad
            return Err(AcquisitionError::ConnectionError { msg: e.to_string() });
        }
    }

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

        // FIXME: this blocks - we need to check for control messages every now and then
        match stream.read_exact(dest_rest) {
            Ok(_) => Ok(()),
            Err(e) => Err(AcquisitionError::ConnectionError { msg: e.to_string() }),
        }
    })?;

    Ok(meta)
}

#[derive(Debug, Clone)]
enum AcquisitionError {
    Disconnected,
    Cancelled,
    BufferFull,
    StateError { msg: String },
    ConfigurationError { msg: String },
    ParseError { msg: String },
    ConnectionError { msg: String },
    APIError { msg: String },
}

impl Display for AcquisitionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AcquisitionError::Cancelled => {
                write!(f, "acquisition cancelled")
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
            AcquisitionError::ParseError { msg } => {
                write!(f, "parse error: {msg}")
            }
            AcquisitionError::ConnectionError { msg } => {
                write!(f, "connection error: {msg}")
            }
            AcquisitionError::APIError { msg } => {
                write!(f, "serval HTTP API error: {msg}")
            }
        }
    }
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

/// With a running acquisition, check for control messages;
/// especially convert `ControlMsg::StopThread` to `AcquisitionError::Cancelled`.
fn check_for_control(control_channel: &Receiver<ControlMsg>) -> Result<(), AcquisitionError> {
    match control_channel.try_recv() {
        Ok(ControlMsg::StartAcquisitionPassive) => Err(AcquisitionError::StateError {
            msg: "received StartAcquisitionPassive while an acquisition was already running"
                .to_string(),
        }),
        Ok(ControlMsg::StopThread) => Err(AcquisitionError::Cancelled),
        Err(TryRecvError::Disconnected) => Err(AcquisitionError::Cancelled),
        Err(TryRecvError::Empty) => Ok(()),
    }
}

/// Passively listen for the start of an acquisition
/// and automatically latch on to it.
fn passive_acquisition(
    control_channel: &Receiver<ControlMsg>,
    from_thread_s: &Sender<ResultMsg>,
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

        // block until we get the first frame:
        let first_frame_meta = match peek_header(&mut stream) {
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

        let free = shm.num_slots_free();
        let total = shm.num_slots_total();
        info!("passive acquisition done; free slots: {}/{}", free, total);

        check_for_control(control_channel)?;
    }
}

fn acquisition(
    to_thread_r: &Receiver<ControlMsg>,
    from_thread_s: &Sender<ResultMsg>,
    detector_config: &DetectorConfig,
    first_frame_meta: &ASIMpxFrameMeta,
    stream: &mut TcpStream,
    frame_stack_size: usize,
    shm: &mut SharedSlabAllocator,
) -> Result<(), AcquisitionError> {
    let t0 = Instant::now();
    let mut last_control_check = Instant::now();

    from_thread_s.send(ResultMsg::AcquisitionStart {
        detector_config: detector_config.clone(),
        first_frame_meta: first_frame_meta.clone(),
    })?;

    debug!("acquisition starting");

    // approx uppper bound of image size in bytes
    let peek_meta = peek_header(stream)?;
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
                old_frame_stack.writing_done(shm)
            };
            // send to our queue:
            from_thread_s.send(ResultMsg::FrameStack {
                frame_stack: to_send,
            })?;
        }

        // we will be done after this frame:
        let done = sequence == detector_config.n_triggers;

        if done {
            let elapsed = t0.elapsed();
            info!("done in {elapsed:?}");

            let handle = frame_stack.writing_done(shm);
            from_thread_s.send(ResultMsg::End {
                frame_stack: handle,
            })?;

            if !extra_frame_stack.is_empty() {
                let handle = extra_frame_stack.writing_done(shm);
                from_thread_s.send(ResultMsg::End {
                    frame_stack: handle,
                })?;
            } else {
                // let's not leak the `extra_frame_stack`:
                // FIXME: `FrameStackForWriting` should really free itself,
                // if `writing_done` was not called manually, which might happen
                // in case of error handling.
                // ah, but it can't, because it doesn't have a reference to `shm`! hmm
                let handle = extra_frame_stack.writing_done(shm);
                shm.free_idx(handle.slot.slot_idx);
            }

            return Ok(());
        }
    }
}

/// convert `AcquisitionError`s to messages on `from_threads_s`
fn background_thread_wrap(
    to_thread_r: &Receiver<ControlMsg>,
    from_thread_s: &Sender<ResultMsg>,
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
            .send(ResultMsg::Error {
                msg: err.to_string(),
            })
            .unwrap();
    }
}

fn background_thread(
    to_thread_r: &Receiver<ControlMsg>,
    from_thread_s: &Sender<ResultMsg>,
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

/// Start a background thread that receives data from the socket and
/// puts it into shared memory.
pub struct ServalReceiver {
    bg_thread: Option<JoinHandle<()>>,
    to_thread: Sender<ControlMsg>,
    from_thread: Receiver<ResultMsg>,
    pub status: ReceiverStatus,
    pub shm_handle: SHMHandle,
}

impl ServalReceiver {
    pub fn new(
        data_uri: &str,
        api_uri: &str,
        frame_stack_size: usize,
        shm: SharedSlabAllocator,
    ) -> Self {
        let (to_thread_s, to_thread_r) = unbounded();
        let (from_thread_s, from_thread_r) = unbounded();

        let builder = std::thread::Builder::new();
        let data_uri = data_uri.to_string();
        let api_uri = api_uri.to_string();

        let shm_handle = shm.get_handle();

        ServalReceiver {
            bg_thread: Some(
                builder
                    .name("bg_thread".to_string())
                    .spawn(move || {
                        background_thread_wrap(
                            &to_thread_r,
                            &from_thread_s,
                            &data_uri,
                            &api_uri,
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
