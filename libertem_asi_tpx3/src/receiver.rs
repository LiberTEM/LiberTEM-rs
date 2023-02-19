use std::{
    fmt::Display,
    mem::replace,
    net::TcpStream,
    thread::JoinHandle,
    time::{Duration, Instant},
};

use crossbeam_channel::{unbounded, Receiver, RecvTimeoutError, SendError, Sender, TryRecvError};
use ipc_test::{SHMHandle, SharedSlabAllocator};
use log::{debug, error, info, trace, warn};

use crate::{
    chunk_stack::{ChunkCSRLayout, ChunkStackForWriting, ChunkStackHandle},
    csr_view_raw::CSRViewRaw,
    headers::{AcquisitionStart, HeaderTypes, ScanStart},
    stream::{stream_recv_chunk, stream_recv_header, StreamError},
};

///
#[derive(PartialEq, Eq, Debug)]
pub enum ResultMsg {
    Error { msg: String }, // generic error response, might need to specialize later
    SerdeError { msg: String, recvd_msg: String },
    AcquisitionStart { header: AcquisitionStart },
    ScanStart { header: ScanStart },
    FrameStack { frame_stack: ChunkStackHandle },
    End { frame_stack: ChunkStackHandle },
}

pub enum ControlMsg {
    StopThread,

    /// Wait for `AcquisitionStart` / `ScanStart` messages and latch onto acquisitions,
    /// until the background thread is stopped.
    StartAcquisitionPassive,
}

#[derive(PartialEq, Eq)]
pub enum ReceiverStatus {
    Idle,
    Running,
    Closed,
}

#[derive(Debug)]
enum AcquisitionError {
    Disconnected,
    Cancelled,
    BufferFull,
    SlotSizeTooSmall {
        slot_size: usize,
        chunk_size: usize,
    },

    /// Example: an unexpected header was received
    StateError {
        msg: String,
    },

    /// An error occurred while reading the socket
    StreamError {
        err: StreamError,
    },
}

impl From<StreamError> for AcquisitionError {
    fn from(err: StreamError) -> Self {
        AcquisitionError::StreamError { err }
    }
}

impl<T> From<SendError<T>> for AcquisitionError {
    fn from(value: SendError<T>) -> Self {
        AcquisitionError::Disconnected
    }
}

impl Display for AcquisitionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AcquisitionError::Cancelled => {
                write!(f, "acquisition cancelled")
            }
            AcquisitionError::SlotSizeTooSmall {
                slot_size,
                chunk_size,
            } => {
                write!(f, "slot size {slot_size} too small for chunk {chunk_size}")
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
            AcquisitionError::StreamError { err } => {
                write!(f, "stream error: {err:?}")
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

/// Passively listen for global acquisition and scan headers
/// and automatically latch on to them.
fn wait_for_acquisition(
    control_channel: &Receiver<ControlMsg>,
    from_thread_s: &Sender<ResultMsg>,
    stream: &mut TcpStream,
    frame_stack_size: usize,
    shm: &mut SharedSlabAllocator,
) -> Result<(), AcquisitionError> {
    // message headers are always 32 bytes; so we put them on the stack here:
    let mut header_bytes: [u8; 32] = [0; 32];
    loop {
        info!("waiting for acquisition header...");
        let header = stream_recv_header(stream, &mut header_bytes)?;

        trace!("got a header: {header:?}");

        match header {
            HeaderTypes::AcquisitionStart { header } => {
                wait_for_scan(
                    header,
                    control_channel,
                    from_thread_s,
                    stream,
                    frame_stack_size,
                    shm,
                )?;

                let free = shm.num_slots_free();
                let total = shm.num_slots_total();
                info!("passive acquisition done; free slots: {}/{}", free, total);
            }
            other => {
                // we haven't received an `AcquisitionStart` header, so we don't
                // know the size of array chunks => we can't synchronize to the stream
                // again, so we need to bail out:
                let msg = format!("expected `AcquisitionStart` header, got {other:?}");
                return Err(AcquisitionError::StateError { msg });
            }
        }

        check_for_control(control_channel)?;
    }
}

fn wait_for_scan(
    acquisition_header: AcquisitionStart,
    control_channel: &Receiver<ControlMsg>,
    from_thread_s: &Sender<ResultMsg>,
    stream: &mut TcpStream,
    frame_stack_size: usize,
    shm: &mut SharedSlabAllocator,
) -> Result<(), AcquisitionError> {
    // message headers are always 32 bytes; so we put them on the stack here:
    let mut header_bytes: [u8; 32] = [0; 32];
    info!("waiting for scan header...");
    loop {
        let header = stream_recv_header(stream, &mut header_bytes)?;
        trace!("got a header: {header:?}");

        match header {
            HeaderTypes::ScanStart { header } => {
                handle_scan(
                    acquisition_header.clone(),
                    header,
                    control_channel,
                    from_thread_s,
                    stream,
                    frame_stack_size,
                    shm,
                )?;

                let free = shm.num_slots_free();
                let total = shm.num_slots_total();
                info!("passive scan done; free slots: {}/{}", free, total);
            }
            HeaderTypes::AcquisitionEnd { header } => {
                info!(
                    "AcquisitionEnd header received for sequence {}",
                    header.sequence
                );
                return Ok(());
            }
            other => {
                // for now, be strict about what headers we expect:
                let msg = format!("expected `ScanStart` header, got {other:?}");
                return Err(AcquisitionError::StateError { msg });
            }
        }

        check_for_control(control_channel)?;
    }
}

///
/// Handle one scan: the `AcquisitionStart` and `ScanStart` messages should
/// already be consumed, the following messages should be:
///
/// - one or more `ArrayChunk` messages and their payload data
/// - exactly one `ScanEnd`
///
/// We don't consume any more messages; the caller should then handle either
///
/// - `AcquisitionEnd` if this was the last scan of the acquisition
/// - `ScanStart` if another scan follows
///
fn handle_scan(
    acquisition_header: AcquisitionStart,
    scan_header: ScanStart,
    to_thread_r: &Receiver<ControlMsg>,
    from_thread_s: &Sender<ResultMsg>,
    stream: &mut TcpStream,
    chunks_per_stack: usize,
    shm: &mut SharedSlabAllocator,
) -> Result<(), AcquisitionError> {
    let t0 = Instant::now();
    let mut last_control_check = Instant::now();

    trace!("sending ResultMsg::AcquisitionStart message");
    from_thread_s.send(ResultMsg::AcquisitionStart {
        header: acquisition_header.clone(),
    })?;

    // approx uppper bound of image size in bytes
    // FIXME! maybe we can approximate this based on dwell time / max. hit rate etc.?
    let approx_size_bytes = 1024 * 1024;

    let slot = match shm.get_mut() {
        None => return Err(AcquisitionError::BufferFull),
        Some(x) => x,
    };
    let mut chunk_stack =
        ChunkStackForWriting::new(slot, chunks_per_stack, approx_size_bytes as usize);

    loop {
        if last_control_check.elapsed() > Duration::from_millis(100) {
            last_control_check = Instant::now();
            check_for_control(to_thread_r)?;
        }

        let mut header_bytes: [u8; 32] = [0; 32];
        let header = stream_recv_header(stream, &mut header_bytes)?;

        match header {
            HeaderTypes::ArrayChunk { header } => {
                let nbytes = header.get_chunk_size_bytes(&acquisition_header);

                if chunk_stack.total_size() < nbytes {
                    // chunk is too large for the slot size, bail out...
                    return Err(AcquisitionError::SlotSizeTooSmall {
                        slot_size: chunk_stack.total_size(),
                        chunk_size: nbytes,
                    });
                }

                if !chunk_stack.can_fit(nbytes) {
                    // send to our queue:
                    let handle = {
                        let slot = match shm.get_mut() {
                            None => return Err(AcquisitionError::BufferFull),
                            Some(x) => x,
                        };
                        let new_frame_stack = ChunkStackForWriting::new(
                            slot,
                            chunks_per_stack,
                            approx_size_bytes as usize,
                        );
                        let old_chunk_stack = replace(&mut chunk_stack, new_frame_stack);
                        old_chunk_stack.writing_done(shm)
                    };

                    // XXX debugging stuff here....
                    // {
                    //     let slot_r = shm.get(handle.slot.slot_idx);
                    //     assert_eq!(
                    //         handle.total_bytes_used - handle.total_bytes_padding,
                    //         handle.get_layout().iter().map(|l| l.data_length_bytes).sum()
                    //     );
                    //     handle.get_chunk_views_raw(&slot_r);
                    // }

                    from_thread_s.send(ResultMsg::FrameStack {
                        frame_stack: handle,
                    })?;
                }
                let sizes = header.get_sizes(&acquisition_header);
                // FIXME: alignment will change!
                // offsets will be aligned in the future, and will come from the chunk header
                let layout = ChunkCSRLayout {
                    indptr_dtype: acquisition_header.indptr_dtype,
                    indptr_offset: 0,
                    indptr_size: sizes.indptr,
                    indices_dtype: acquisition_header.indices_dtype,
                    indices_offset: sizes.indptr,
                    indices_size: sizes.indices,
                    value_dtype: header.value_dtype,
                    value_offset: sizes.indptr + sizes.indices,
                    value_size: sizes.values,
                    nframes: header.nframes,
                    nnz: header.length,
                    data_length_bytes: nbytes,
                };
                let buf = chunk_stack.slice_for_writing(nbytes, layout.clone());
                stream_recv_chunk(stream, buf)?;
                assert_eq!(buf.len(), nbytes);
                CSRViewRaw::from_bytes_with_layout(buf, &layout);
            }
            HeaderTypes::ScanEnd { header } => {
                let elapsed = t0.elapsed();
                info!(
                    "scan {} done in {elapsed:?}, reading acquisition footer...",
                    header.sequence
                );

                let handle = chunk_stack.writing_done(shm);
                from_thread_s.send(ResultMsg::End {
                    frame_stack: handle,
                })?;
                return Ok(());
            }
            other_header => {
                let msg = format!("unexpected header in acquisition: {other_header:?}");
                return Err(AcquisitionError::StateError { msg });
            }
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
    remote: String,
    frame_stack_size: usize,
    mut shm: SharedSlabAllocator,
) -> Result<(), AcquisitionError> {
    // automatically reconnect on errors
    'outer: loop {
        info!("connecting to {remote}");
        let mut stream = match TcpStream::connect(&remote) {
            Ok(s) => s,
            Err(e) => {
                error!("failed to connect to {remote} ({e}), retrying...");
                std::thread::sleep(Duration::from_secs(1));
                continue;
            }
        };
        info!("connected to {remote}");
        // stream.set_read_timeout(Some(Duration::from_millis(100))).unwrap();

        // handle control messages and start acquisition
        'inner: loop {
            // control: main threads tells us to start or quit
            let control = to_thread_r.recv_timeout(Duration::from_millis(100));
            match control {
                Ok(ControlMsg::StartAcquisitionPassive) => {
                    match wait_for_acquisition(
                        to_thread_r,
                        from_thread_s,
                        &mut stream,
                        frame_stack_size,
                        &mut shm,
                    ) {
                        Ok(_) => {}
                        Err(AcquisitionError::Disconnected) => {
                            return Ok(()); // the channel is gone, so :shrug:
                        }
                        Err(AcquisitionError::Cancelled) => {
                            break 'inner; // reconnect
                        }
                        e => {
                            return e; // other error, give up? maybe we can retry instead?
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
pub struct TPXReceiver {
    bg_thread: Option<JoinHandle<()>>,
    to_thread: Sender<ControlMsg>,
    from_thread: Receiver<ResultMsg>,
    pub status: ReceiverStatus,
    pub shm_handle: SHMHandle,
}

impl TPXReceiver {
    pub fn new(uri: &str, frame_stack_size: usize, shm: SharedSlabAllocator) -> Self {
        let (to_thread_s, to_thread_r) = unbounded();
        let (from_thread_s, from_thread_r) = unbounded();

        let builder = std::thread::Builder::new();
        let uri = uri.to_string();

        let shm_handle = shm.get_handle();

        TPXReceiver {
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

impl Default for TPXReceiver {
    fn default() -> Self {
        let shm = SharedSlabAllocator::new(5000, 512 * 512 * 2, true).expect("create shm");
        Self::new("tcp://127.0.0.1:9999", 1, shm)
    }
}
