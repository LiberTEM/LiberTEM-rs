use std::{
    fmt::{Debug, Display},
    io::Read,
    mem::replace,
    net::TcpStream,
    str::FromStr,
    thread::JoinHandle,
    time::{Duration, Instant},
};

use crossbeam_channel::{unbounded, Receiver, RecvTimeoutError, SendError, Sender, TryRecvError};
use ipc_test::{SHMHandle, SharedSlabAllocator};
use log::{debug, error, info, trace, warn};

use crate::{
    common::{DType, FrameMeta},
    frame_stack::{FrameStackForWriting, FrameStackHandle},
};

#[derive(PartialEq, Eq, Debug)]
pub enum ResultMsg {
    Error {
        msg: String,
    }, // generic error response, might need to specialize later

    /// The frame header failed to parse
    ParseError {
        msg: String,
    },

    AcquisitionStart {
        // series: u64,
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
fn peek_header(stream: &mut TcpStream) -> FrameMeta {
    const HEADER_BUF_SIZE: usize = 512;
    let mut buf: [u8; HEADER_BUF_SIZE] = [0; HEADER_BUF_SIZE];
    // FIXME: error handling, timeout, ...
    let nbytes = stream.peek(&mut buf).unwrap();
    todo!("extract header parsing logic from `recv_frame` and use it here");
}

fn recv_frame(
    sequence: u64,
    stream: &mut TcpStream,
    control_channel: &Receiver<ControlMsg>,
    frame_stack: &mut FrameStackForWriting,
    extra_frame_stack: &mut FrameStackForWriting,
) -> Result<FrameMeta, AcquisitionError> {
    // TODO:
    // - timeout handling
    // - error handling (parsing, receiving)

    // 1) Read the first N bytes (512?) from the socket into a buffer, and parse the PGM header

    const HEADER_BUF_SIZE: usize = 512;
    let mut buf: [u8; HEADER_BUF_SIZE] = [0; HEADER_BUF_SIZE];

    match stream.read_exact(&mut buf) {
        Ok(_) => {}
        Err(_) => todo!(),
    }

    // 2) Parse the header, importantly reading width, height, bytes-per-pixel (maxval)

    let mut pos: usize = 0;

    // Each PGM image consists of the following:
    // •      A "magic number" for identifying the file type.  A pgm image's magic number is the two characters "P5".
    // FIXME: error handling
    assert_eq!(&buf[0..2], b"P5");
    pos += 2;

    // •      Whitespace (blanks, TABs, CRs, LFs).
    while buf[pos].is_ascii_whitespace() && pos < HEADER_BUF_SIZE {
        pos += 1;
    }

    // •      A width, formatted as ASCII characters in decimal.
    let mut width_start = pos;
    while !buf[pos].is_ascii_whitespace() && pos < HEADER_BUF_SIZE {
        pos += 1;
    }
    let width: u16 = num_from_byte_slice(&buf[width_start..pos]);

    // •      Whitespace.
    while buf[pos].is_ascii_whitespace() && pos < HEADER_BUF_SIZE {
        pos += 1;
    }

    // •      A height, again in ASCII decimal.
    let mut height_start = pos;
    while !buf[pos].is_ascii_whitespace() && pos < HEADER_BUF_SIZE {
        pos += 1;
    }
    let height: u16 = num_from_byte_slice(&buf[height_start..pos]);

    // •      Whitespace.
    while buf[pos].is_ascii_whitespace() && pos < HEADER_BUF_SIZE {
        pos += 1;
    }

    // •      The maximum gray value (Maxval), again in ASCII decimal.  Must be less than 65536, and more than zero.
    let mut maxval_start = pos;
    while !buf[pos].is_ascii_whitespace() && pos < HEADER_BUF_SIZE {
        pos += 1;
    }
    let maxval: u32 = num_from_byte_slice(&buf[maxval_start..pos]);

    // really, more than zero? how do you represent an all-black image in PGM then?
    assert!(maxval < 65536 && maxval >= 0);

    // •      A single whitespace character (usually a newline).
    assert!(buf[pos].is_ascii_whitespace());

    // •      A raster of Height rows, [...]
    let raster_start_pos = pos;

    let dtype: DType = DType::from_maxval(maxval);
    let data_length_bytes = width as usize * height as usize * dtype.num_bytes();

    // 3) Now we know how large the binary part is. Copy the rest of the buffer
    //    into the frame stack, and read the remaining bytes directly into the
    //    shared memory.

    let meta = FrameMeta {
        sequence,
        dtype,
        width,
        height,
        data_length_bytes,
    };

    let mut fs = if frame_stack.can_fit(data_length_bytes) {
        frame_stack
    } else {
        assert!(extra_frame_stack.len() == 0);
        assert!(extra_frame_stack.can_fit(data_length_bytes));
        extra_frame_stack
    };

    fs.write_frame(&meta, |dest_buf| {
        let head_src = &buf[pos..];
        dest_buf[0..head_src.len()].copy_from_slice(&head_src);

        let mut dest_rest = &mut dest_buf[head_src.len()..];

        // FIXME: error handling (can we pass the error through as result of the closure?)
        // FIXME: this blocks - we need to check for control messages every now and then
        stream.read_exact(&mut dest_rest).unwrap();
    });

    Ok(meta)
}

#[derive(Debug, Clone)]
enum AcquisitionError {
    Disconnected,
    SeriesMismatch,
    Cancelled,
    BufferFull,
    StateError { msg: String },
    ConfigurationError { msg: String },
}

impl Display for AcquisitionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AcquisitionError::Cancelled => {
                write!(f, "acquisition cancelled")
            }
            AcquisitionError::SeriesMismatch => {
                write!(f, "series mismatch")
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
    shm: &mut SharedSlabAllocator,
) -> Result<(), AcquisitionError> {
    loop {
        let stream: &mut TcpStream;

        // FIXME: try to connect, then

        // block until we get the first frame:
        let first_frame = peek_header(stream);

        let num_triggers = 0; // FIXME: ask the HTTP API about the detector config

        acquisition(
            control_channel,
            from_thread_s,
            num_triggers,
            stream,
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
    num_triggers: usize,
    stream: &mut TcpStream,
    frame_stack_size: usize,
    shm: &mut SharedSlabAllocator,
) -> Result<(), AcquisitionError> {
    let t0 = Instant::now();
    let mut last_control_check = Instant::now();

    match from_thread_s.send(ResultMsg::AcquisitionStart {}) {
        Ok(_) => (),
        Err(SendError(_)) => return Err(AcquisitionError::Disconnected),
    }

    debug!("acquisition starting");

    // approx uppper bound of image size in bytes
    let peek_meta = peek_header(stream);
    let approx_size_bytes = 2 * peek_meta.get_size();

    let slot = match shm.get_mut() {
        None => return Err(AcquisitionError::BufferFull),
        Some(x) => x,
    };
    let mut frame_stack =
        FrameStackForWriting::new(slot, frame_stack_size, approx_size_bytes as usize);

    // in case the frame stack is full, the receiving function needs
    // an alternative destination:
    let mut extra_frame_stack =
        FrameStackForWriting::new(slot, frame_stack_size, approx_size_bytes as usize);

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
            &to_thread_r,
            &mut frame_stack,
            &mut extra_frame_stack,
        );
        sequence += 1;

        // If `recv_frame` had to use `extra_frame_stack`, `frame_stack` is
        // finished and we need to exchange the stacks:
        if extra_frame_stack.len() > 0 {
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
                let old_frame_stack = replace(&mut frame_stack, new_frame_stack);
                old_frame_stack.writing_done(shm)
            };
            // send to our queue:
            match from_thread_s.send(ResultMsg::FrameStack {
                frame_stack: to_send,
            }) {
                Ok(_) => (),
                Err(SendError(_)) => return Err(AcquisitionError::Disconnected),
            }
        }

        // we will be done after this frame:
        let done = sequence == num_triggers as u64;

        if done {
            let elapsed = t0.elapsed();
            info!("done in {elapsed:?}");

            let handle = frame_stack.writing_done(shm);
            match from_thread_s.send(ResultMsg::End {
                frame_stack: handle,
            }) {
                Ok(_) => (),
                Err(SendError(_)) => return Err(AcquisitionError::Disconnected),
            }

            if extra_frame_stack.len() > 0 {
                let handle = extra_frame_stack.writing_done(shm);
                match from_thread_s.send(ResultMsg::End {
                    frame_stack: handle,
                }) {
                    Ok(_) => (),
                    Err(SendError(_)) => return Err(AcquisitionError::Disconnected),
                }
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

fn background_thread(
    to_thread_r: &Receiver<ControlMsg>,
    from_thread_s: &Sender<ResultMsg>,
    uri: String,
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
    pub fn new(uri: &str, frame_stack_size: usize, shm: SharedSlabAllocator) -> Self {
        let (to_thread_s, to_thread_r) = unbounded();
        let (from_thread_s, from_thread_r) = unbounded();

        let builder = std::thread::Builder::new();
        let uri = uri.to_string();

        let shm_handle = shm.get_handle();

        ServalReceiver {
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
