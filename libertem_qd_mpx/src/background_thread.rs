use std::{
    io::{self, ErrorKind, Read},
    net::TcpStream,
    sync::mpsc::{channel, Receiver, RecvTimeoutError, SendError, Sender, TryRecvError},
    thread::JoinHandle,
    time::{Duration, Instant},
};

use common::{
    background_thread::{BackgroundThread, BackgroundThreadSpawnError, ControlMsg, ReceiverMsg},
    frame_stack::{FrameMeta, FrameStackForWriting, FrameStackWriteError},
    generic_connection::AcquisitionConfig,
    tcp::{self, ReadExactError},
    utils::{num_from_byte_slice, three_way_shift, NumParseError},
};
use ipc_test::{slab::ShmError, SharedSlabAllocator};
use log::{debug, error, info, trace, warn};

use crate::base_types::{
    AcqHeaderParseError, FrameMetaParseError, QdAcquisitionHeader, QdDetectorConnConfig,
    QdFrameMeta, PREFIX_SIZE,
};

type QdControlMsg = ControlMsg<()>;

type QdReceiverMsg = ReceiverMsg<QdFrameMeta, QdAcquisitionHeader>;

#[derive(Debug, thiserror::Error)]
pub enum AcquisitionError {
    #[error("channel disconnected")]
    Disconnected,

    #[error("acquisition was cancelled by the user")]
    Cancelled,

    #[error("receiver state error: {msg}")]
    StateError { msg: String },

    #[error("connection error: {msg}")]
    ConnectionError { msg: String },

    #[error("parse error: {msg}")]
    HeaderParseError { msg: String },

    #[error("configuration error: {msg}")]
    ConfigurationError { msg: String },

    #[error("shm buffer full")]
    NoSlotAvailable,

    #[error("error writing to shm: {0}")]
    WriteError(#[from] FrameStackWriteError),

    #[error("I/O error: {source}")]
    IOError {
        #[from]
        source: io::Error,
    },

    #[error("peek error: could not peek {nbytes}")]
    PeekError { nbytes: usize },
}

impl From<NumParseError> for AcquisitionError {
    fn from(value: NumParseError) -> Self {
        Self::HeaderParseError {
            msg: value.to_string(),
        }
    }
}

impl From<AcqHeaderParseError> for AcquisitionError {
    fn from(value: AcqHeaderParseError) -> Self {
        AcquisitionError::HeaderParseError {
            msg: value.to_string(),
        }
    }
}

impl From<FrameMetaParseError> for AcquisitionError {
    fn from(value: FrameMetaParseError) -> Self {
        AcquisitionError::HeaderParseError {
            msg: value.to_string(),
        }
    }
}

impl<T> From<SendError<T>> for AcquisitionError {
    fn from(_value: SendError<T>) -> Self {
        AcquisitionError::Disconnected
    }
}

impl From<ShmError> for AcquisitionError {
    fn from(value: ShmError) -> Self {
        match value {
            ShmError::NoSlotAvailable => AcquisitionError::NoSlotAvailable,
        }
    }
}

impl From<ReadExactError<AcquisitionError>> for AcquisitionError {
    fn from(value: ReadExactError<AcquisitionError>) -> Self {
        match value {
            ReadExactError::Interrupted { size, err } => {
                if size > 0 {
                    warn!("interrupted read after {size} bytes; discarding buffer");
                }
                err
            }
            ReadExactError::IOError { err } => AcquisitionError::from(err),
            ReadExactError::PeekError { size } => AcquisitionError::PeekError { nbytes: size },
        }
    }
}

pub struct QdBackgroundThread {
    bg_thread: JoinHandle<()>,
    to_thread: Sender<QdControlMsg>,
    from_thread: Receiver<QdReceiverMsg>,
}

/// With a running acquisition, check for control messages;
/// especially convert `ControlMsg::StopThread` to `AcquisitionError::Cancelled`.
fn check_for_control(control_channel: &Receiver<QdControlMsg>) -> Result<(), AcquisitionError> {
    match control_channel.try_recv() {
        Ok(ControlMsg::StartAcquisitionPassive) => Err(AcquisitionError::StateError {
            msg: "received StartAcquisitionPassive while an acquisition was already running"
                .to_string(),
        }),
        Ok(ControlMsg::StopThread) => Err(AcquisitionError::Cancelled),
        Ok(ControlMsg::SpecializedControlMsg { msg: _ }) => {
            panic!("unsupported SpecializedControlMsg")
        }
        Err(TryRecvError::Disconnected) => Err(AcquisitionError::Disconnected),
        Err(TryRecvError::Empty) => Ok(()),
    }
}

/// Fill `buf` from `stream` (like TcpStream::read_exact), but periodically check
/// `to_thread_r` for control messages, which allows to interrupt the acquisition.
/// Assumes a blocking socket with a timeout. In case of interruption, data will be discarded
/// (it's still in `buf`, but we don't keep track how much we read...)
fn read_exact_interruptible(
    stream: &mut impl Read,
    buf: &mut [u8],
    to_thread_r: &Receiver<QdControlMsg>,
) -> Result<(), AcquisitionError> {
    common::tcp::read_exact_interruptible(stream, buf, || check_for_control(to_thread_r))?;

    Ok(())
}

/// Read from `stream` until we hit the timeout. Original socket timeout is
/// overwritten and should be restored by the caller.
fn drain_until_timeout(
    stream: &mut TcpStream,
    to_thread_r: &Receiver<QdControlMsg>,
    timeout: &Duration,
) -> Result<usize, AcquisitionError> {
    let mut tmp = vec![0; 1024 * 1024];
    let mut total_drained: usize = 0;
    stream.set_read_timeout(Some(*timeout))?;
    loop {
        check_for_control(to_thread_r)?;
        total_drained += match stream.read(&mut tmp) {
            Ok(size) => {
                trace!("drain_until_timeout: drained {size}");
                if size == 0 {
                    // EOF: okay....
                    warn!("drained until EOF!");
                    return Ok(total_drained);
                }
                size
            }
            Err(e) => match e.kind() {
                ErrorKind::TimedOut | ErrorKind::WouldBlock => {
                    trace!("drain_until_timeout: timeout: {e:?}");
                    return Ok(total_drained);
                }
                kind => {
                    trace!("drain_until_timeout: other error kind: {kind:?}");
                    return Err(e.into());
                }
            },
        };
    }
}

/// Fill `buf` from `stream` (like TcpStream::peek), but periodically check
/// `to_thread_r` for control messages, which allows to interrupt the acquisition.
/// Assumes a blocking socket with a timeout.
fn peek_exact_interruptible(
    stream: &mut TcpStream,
    buf: &mut [u8],
    to_thread_r: &Receiver<QdControlMsg>,
) -> Result<(), AcquisitionError> {
    tcp::peek_exact_interruptible(stream, buf, Duration::from_millis(10), 10, || {
        check_for_control(to_thread_r)
    })?;

    Ok(())
}

/// Parse the length from a string in the form of 'MPX,<length>', where
/// `<length>` is a 10 digit number (possibly with leading zeros).
fn parse_mpx_length(buf: &[u8]) -> Result<usize, AcquisitionError> {
    let magic = &buf[0..4];
    if magic != b"MPX," {
        return Err(AcquisitionError::HeaderParseError {
            msg: format!(
                "expected 'MPX,', got: '{:X?}' ('{}') (maybe previous read was short?)",
                magic,
                String::from_utf8_lossy(magic)
            ),
        });
    }
    Ok(num_from_byte_slice(&buf[4..14])?)
}

/// Read MPX message from `stream`
///
/// Message is in the form: "MPX,<length>,<MSGTYPE>,<MSGPAYLOAD>"
///
/// Where <length> is the length of ",<MSGTYPE>,<MSGPAYLOAD>" and is encoded
/// as a 10-character decimal number w/ leading zeros.
///
/// The message is read into a newly allocated `Vec`, so this is not the
/// appropriate function to use for receiving payload data.
fn read_mpx_message(
    stream: &mut impl Read,
    to_thread_r: &Receiver<QdControlMsg>,
) -> Result<MPXMessage, AcquisitionError> {
    // read and parse the 'MPX,<length>,' part:
    let mut prefix = [0u8; PREFIX_SIZE];
    read_exact_interruptible(stream, &mut prefix, to_thread_r)?;
    trace!(
        "read_mpx_message: prefix={prefix:X?} ({})",
        std::str::from_utf8(&prefix).unwrap()
    );
    let length = parse_mpx_length(&prefix)?;
    let mut payload = vec![0u8; length - 1];
    read_exact_interruptible(stream, &mut payload, to_thread_r)?;
    Ok(MPXMessage { length, payload })
}

/// Read a frame header, based on the information in `first_frame_meta`,
/// validating that the header+payload length is the same as the first frame.
///
/// This includes reading the MPX prefix.
fn read_frame_header(
    stream: &mut impl Read,
    to_thread_r: &Receiver<QdControlMsg>,
    first_frame_meta: &QdFrameMeta,
    scratch_buf: &mut Vec<u8>,
) -> Result<QdFrameMeta, AcquisitionError> {
    // read the full header in one go:
    scratch_buf.resize(first_frame_meta.get_total_size_header(), 0);
    read_exact_interruptible(stream, scratch_buf, to_thread_r)?;

    trace!(
        "got raw frame header: '{}'",
        String::from_utf8_lossy(scratch_buf)
    );
    // parse and validate "mpx length":
    let length = parse_mpx_length(&scratch_buf[..PREFIX_SIZE])?;
    if length != first_frame_meta.get_mpx_length() {
        return Err(AcquisitionError::HeaderParseError {
            msg: format!(
                "unexpected length difference: expected {} got {}",
                first_frame_meta.get_mpx_length(),
                length
            ),
        });
    }

    Ok(QdFrameMeta::parse_bytes(
        &scratch_buf[PREFIX_SIZE..],
        length,
    )?)
}

fn read_frame_payload_into(
    stream: &mut impl Read,
    to_thread_r: &Receiver<QdControlMsg>,
    frame_meta: &QdFrameMeta,
    out: &mut [u8],
) -> Result<(), AcquisitionError> {
    assert_eq!(frame_meta.get_data_length_bytes(), out.len());
    read_exact_interruptible(stream, out, to_thread_r)
}

pub struct MPXMessage {
    length: usize,
    payload: Vec<u8>,
}

/// Peek MPX message from `stream`
///
/// Message is in the form: "MPX,<length>,<MSGTYPE>,<MSGPAYLOAD>"
///
/// Where <length> is the length of ",<MSGTYPE>,<MSGPAYLOAD>" and is encoded
/// as a 10-character decimal number w/ leading zeros.
fn peek_mpx_message(
    stream: &mut TcpStream,
    to_thread_r: &Receiver<QdControlMsg>,
    max_length: usize,
) -> Result<MPXMessage, AcquisitionError> {
    // read and parse the 'MPX,<length>,' part:
    let mut prefix = [0u8; PREFIX_SIZE];
    peek_exact_interruptible(stream, &mut prefix, to_thread_r)?;

    trace!("peek_mpx_message: {}", String::from_utf8_lossy(&prefix));

    let magic = &prefix[0..4];
    if magic != b"MPX," {
        return Err(AcquisitionError::HeaderParseError {
            msg: format!(
                "expected 'MPX,', got: '{:X?}' ('{}'; maybe previous read was short?)",
                magic,
                String::from_utf8_lossy(magic),
            ),
        });
    }
    let length: usize = num_from_byte_slice(&prefix[4..14])?;

    // subtle difference w/ `read_mpx_message`: we need to have a larger buffer here,
    // because here we get the prefix again!
    let mut prefix_and_payload = vec![0u8; max_length.min(length + PREFIX_SIZE - 1)];
    peek_exact_interruptible(stream, &mut prefix_and_payload, to_thread_r)?;
    Ok(MPXMessage {
        payload: prefix_and_payload[PREFIX_SIZE..].to_vec(),
        length,
    })
}

fn read_acquisition_header(
    stream: &mut impl Read,
    to_thread_r: &Receiver<QdControlMsg>,
) -> Result<QdAcquisitionHeader, AcquisitionError> {
    let acq_header_raw = read_mpx_message(stream, to_thread_r)?;
    Ok(QdAcquisitionHeader::parse_bytes(&acq_header_raw.payload)?)
}

fn peek_first_frame_header(
    stream: &mut TcpStream,
    to_thread_r: &Receiver<QdControlMsg>,
) -> Result<QdFrameMeta, AcquisitionError> {
    let msg = peek_mpx_message(stream, to_thread_r, 2048)?;
    trace!(
        "parsing first frame header: '{}'",
        String::from_utf8_lossy(&msg.payload)
    );
    Ok(QdFrameMeta::parse_bytes(&msg.payload, msg.length)?)
}

/// Receive a frame directly into shared memory
fn recv_frame(
    stream: &mut impl Read,
    to_thread_r: &Receiver<QdControlMsg>,
    first_frame_meta: &QdFrameMeta,
    frame_stack: &mut FrameStackForWriting<QdFrameMeta>,
    extra_frame_stack: &mut FrameStackForWriting<QdFrameMeta>,
) -> Result<QdFrameMeta, AcquisitionError> {
    let mut scratch_buf: Vec<u8> = Vec::new();
    let frame_meta = read_frame_header(stream, to_thread_r, first_frame_meta, &mut scratch_buf)?;
    let payload_size = frame_meta.get_data_length_bytes();

    trace!("parsed frame header: {frame_meta:?}");

    let fs = if frame_stack.can_fit(payload_size) {
        frame_stack
    } else {
        trace!(
            "frame_stack can't fit this frame: {} {}",
            frame_stack.bytes_free(),
            payload_size,
        );
        if !extra_frame_stack.is_empty() {
            return Err(AcquisitionError::StateError {
                msg: "extra_frame_stack should be empty".to_owned(),
            });
        }
        if !extra_frame_stack.can_fit(payload_size) {
            return Err(AcquisitionError::ConfigurationError {
                msg: format!(
                    "extra_frame_stack can't fit frame; frame size {}, frame stack size {}",
                    payload_size,
                    extra_frame_stack.slot_size()
                ),
            });
        }
        extra_frame_stack
    };

    // FIXME: with a vectored read, we could read multiple frames into the frame stack,
    // the idea being we we vector the header reads into a separate buffer from the payload,
    // with the payload being in SHM.
    // This is really only needed if the current approach turns out to be too
    // slow for some reason.

    fs.write_frame(&frame_meta, |dest_buf| {
        read_frame_payload_into(stream, to_thread_r, &frame_meta, dest_buf)
    })?;

    Ok(frame_meta)
}

fn make_frame_stack(
    shm: &mut SharedSlabAllocator,
    config: &QdDetectorConnConfig,
    meta: &QdFrameMeta,
    to_thread_r: &Receiver<QdControlMsg>,
) -> Result<FrameStackForWriting<QdFrameMeta>, AcquisitionError> {
    loop {
        // keep some slots free for splitting frame stacks
        if shm.num_slots_free() < 3 && shm.num_slots_total() >= 3 {
            trace!("shm is almost full; waiting and creating backpressure...");
            check_for_control(to_thread_r)?;
            std::thread::sleep(Duration::from_millis(1));
            continue;
        }

        match shm.try_get_mut() {
            Ok(slot) => {
                return Ok(FrameStackForWriting::new(
                    slot,
                    config.frame_stack_size,
                    meta.get_data_length_bytes(),
                ))
            }
            Err(ShmError::NoSlotAvailable) => {
                trace!("shm is full; waiting and creating backpressure...");
                check_for_control(to_thread_r)?;
                std::thread::sleep(Duration::from_millis(1));
                continue;
            }
        }
    }
}

fn acquisition(
    to_thread_r: &Receiver<QdControlMsg>,
    from_thread_s: &Sender<QdReceiverMsg>,
    acquisition_header: &QdAcquisitionHeader,
    first_frame_meta: &QdFrameMeta,
    stream: &mut TcpStream,
    config: &QdDetectorConnConfig,
    shm: &mut SharedSlabAllocator,
) -> Result<(), AcquisitionError> {
    let t0 = Instant::now();
    let mut last_control_check = Instant::now();

    from_thread_s.send(ReceiverMsg::AcquisitionStart {
        pending_acquisition: acquisition_header.clone(),
    })?;

    let mut frame_stack = make_frame_stack(shm, config, first_frame_meta, to_thread_r)?;
    let mut extra_frame_stack = make_frame_stack(shm, config, first_frame_meta, to_thread_r)?;

    let mut sequence = 0;

    loop {
        if last_control_check.elapsed() > Duration::from_millis(300) {
            last_control_check = Instant::now();
            check_for_control(to_thread_r)?;
            trace!("acquisition progress: sequence={sequence}");
        }

        let meta = recv_frame(
            stream,
            to_thread_r,
            first_frame_meta,
            &mut frame_stack,
            &mut extra_frame_stack,
        )?;

        if !extra_frame_stack.is_empty() {
            let to_send = {
                let new_frame_stack = make_frame_stack(shm, config, first_frame_meta, to_thread_r)?;
                let old_frame_stack =
                    three_way_shift(&mut frame_stack, &mut extra_frame_stack, new_frame_stack);
                old_frame_stack.writing_done(shm)?
            };
            // send to our queue:
            from_thread_s.send(ReceiverMsg::FrameStack {
                frame_stack: to_send,
            })?;
        }

        sequence += 1;

        if sequence != meta.get_sequence() {
            warn!("sequence number mismatch; did we lose a frame?");
            // if this happens for some reason, we must adjust the sequence
            // number to properly terminate the acquisition:
            sequence = meta.get_sequence();
        }

        let done = sequence as usize == acquisition_header.num_frames();
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

fn passive_acquisition(
    to_thread_r: &Receiver<QdControlMsg>,
    from_thread_s: &Sender<QdReceiverMsg>,
    config: &QdDetectorConnConfig,
    shm: &mut SharedSlabAllocator,
) -> Result<(), AcquisitionError> {
    let host = &config.data_host;
    let port = config.data_port;

    let data_uri = format!("{host}:{port}");
    info!("connecting to {}...", &data_uri);

    check_for_control(to_thread_r)?;
    let mut stream: TcpStream = loop {
        break match TcpStream::connect(&data_uri) {
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
    };

    info!("connected to {}.", &data_uri);

    if let Some(timeout) = &config.drain {
        info!("draining for {timeout:?}...");
        let drained = drain_until_timeout(&mut stream, to_thread_r, timeout)?;
        if drained > 0 {
            info!("drained {drained} bytes of garbage");
        }
    }

    from_thread_s.send(ReceiverMsg::ReceiverArmed)?;

    stream.set_read_timeout(Some(Duration::from_millis(100)))?;

    loop {
        // wait for the acquisition header, which is sent when the detector
        // is armed with STARTACQUISITION:
        let acquisition_header: QdAcquisitionHeader =
            read_acquisition_header(&mut stream, to_thread_r)?;

        info!("acquisition header: {:?}", acquisition_header);

        // Wait for the first frame, which is sent when the acquisition actually starts
        // (for example, if it is triggered with SOFTTRIGGER or by the configured
        // hardware trigger):
        let first_frame_meta: QdFrameMeta = peek_first_frame_header(&mut stream, to_thread_r)?;
        // FIXME: send a status update via `from_thread_s`?

        acquisition(
            to_thread_r,
            from_thread_s,
            &acquisition_header,
            &first_frame_meta,
            &mut stream,
            config,
            shm,
        )?;

        let free = shm.num_slots_free();
        let total = shm.num_slots_total();
        info!("passive acquisition done; free slots: {}/{}", free, total);

        check_for_control(to_thread_r)?;
    }
}

fn background_thread(
    config: &QdDetectorConnConfig,
    to_thread_r: &Receiver<QdControlMsg>,
    from_thread_s: &Sender<QdReceiverMsg>,
    mut shm: SharedSlabAllocator,
) -> Result<(), AcquisitionError> {
    'outer: loop {
        loop {
            // control: main threads tells us to quit
            let control = to_thread_r.recv_timeout(Duration::from_millis(100));
            match control {
                Ok(ControlMsg::StartAcquisitionPassive) => {
                    match passive_acquisition(to_thread_r, from_thread_s, config, &mut shm) {
                        Ok(_) => {}
                        e @ Err(AcquisitionError::Disconnected | AcquisitionError::Cancelled) => {
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
                    warn!("background_thread: control channel has disconnected");
                    break 'outer;
                }
                Err(RecvTimeoutError::Timeout) => (), // no message, nothing to do
                Ok(ControlMsg::SpecializedControlMsg { msg: _ }) => {
                    panic!("ControlMsg::SpecializesControlMsg is unused for QD MPX");
                }
            }
        }
    }
    debug!("background_thread: is done");
    Ok(())
}

fn background_thread_wrap(
    config: &QdDetectorConnConfig,
    to_thread_r: &Receiver<QdControlMsg>,
    from_thread_s: &Sender<QdReceiverMsg>,
    shm: SharedSlabAllocator,
) {
    if let Err(err) = background_thread(config, to_thread_r, from_thread_s, shm) {
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

impl QdBackgroundThread {
    pub fn spawn(
        config: &QdDetectorConnConfig,
        shm: &SharedSlabAllocator,
    ) -> Result<Self, BackgroundThreadSpawnError> {
        let (to_thread_s, to_thread_r) = channel();
        let (from_thread_s, from_thread_r) = channel();

        let builder = std::thread::Builder::new();
        let shm = shm.clone_and_connect()?;
        let config = config.clone();

        debug!("connection config: {config:?}");

        Ok(Self {
            bg_thread: builder
                .name("bg_thread".to_string())
                .spawn(move || background_thread_wrap(&config, &to_thread_r, &from_thread_s, shm))
                .map_err(BackgroundThreadSpawnError::SpawnFailed)?,
            from_thread: from_thread_r,
            to_thread: to_thread_s,
        })
    }
}

impl BackgroundThread for QdBackgroundThread {
    type FrameMetaImpl = QdFrameMeta;
    type AcquisitionConfigImpl = QdAcquisitionHeader;
    type ExtraControl = ();

    fn channel_to_thread(
        &mut self,
    ) -> &mut std::sync::mpsc::Sender<common::background_thread::ControlMsg<Self::ExtraControl>>
    {
        &mut self.to_thread
    }

    fn channel_from_thread(
        &mut self,
    ) -> &mut std::sync::mpsc::Receiver<
        common::background_thread::ReceiverMsg<Self::FrameMetaImpl, Self::AcquisitionConfigImpl>,
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

#[cfg(test)]
mod test {
    use std::{
        io::Write,
        net::{SocketAddr, TcpListener, TcpStream},
        path::PathBuf,
        sync::mpsc::channel,
        time::Duration,
    };

    use common::generic_connection::{ConnectionError, GenericConnection};
    use ipc_test::SharedSlabAllocator;
    use tempfile::{tempdir, TempDir};

    use crate::base_types::{QdAcquisitionHeader, QdDetectorConnConfig};

    use super::QdBackgroundThread;

    fn make_test_server() -> (TcpListener, SocketAddr) {
        let fake_server = TcpListener::bind("127.0.0.1:0").unwrap();
        let local_addr = fake_server.local_addr().unwrap();

        (fake_server, local_addr)
    }

    fn get_socket_path() -> (TempDir, PathBuf) {
        let socket_dir = tempdir().unwrap();
        let socket_as_path = socket_dir.path().join("stuff.socket");

        (socket_dir, socket_as_path)
    }

    fn wrap_into_mpx(full_msg: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();

        let mpx = format!("MPX,{:010},", full_msg.len() + 1);
        out.write_all(mpx.as_bytes()).unwrap();
        out.write_all(full_msg).unwrap();

        out
    }

    fn send_frames(stream: &mut TcpStream, number_of_frames: usize) {
        for idx in 0..number_of_frames {
            // copies all over.. but that's ok, it's only test code!
            let data_offset = 384;
            let header = format!(
                "MQ1,{:06},{data_offset:05},01,0256,0256,U08,   1x1,01,2020-05-18 16:51:49.971626,0.000555,0,0,0,1.200000E+2,5.110000E+2,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,3RX,175,511,000,000,000,000,000,000,125,255,125,125,100,100,082,100,087,030,128,004,255,129,128,176,168,511,511,MQ1A,2020-05-18T14:51:49.971626178Z,555000ns,6,",
                idx + 1,
            );
            let header_bytes = header.as_bytes();
            let padding = vec![0x0u8; data_offset - header_bytes.len()];

            let mut frame_msg = Vec::new();
            frame_msg.write_all(header_bytes).unwrap();
            frame_msg.write_all(&padding).unwrap();
            assert_eq!(frame_msg.len(), data_offset); // data_offset is the "full header length"

            let data = vec![idx as u8; 256 * 256];
            frame_msg.write_all(&data).unwrap();

            let full_frame_msg = wrap_into_mpx(&frame_msg);
            // eprintln!("full_frame_msg: {}", String::from_utf8_lossy(&full_frame_msg));

            stream.write_all(&full_frame_msg).unwrap();
        }
    }

    fn send_acq_header(stream: &mut TcpStream, number_of_frames: usize) {
        let hdr = format!(
            r"HDR,	
Time and Date Stamp (day, mnth, yr, hr, min, s):	18/05/2020 16:51:48
Chip ID:	W529_F5,-,-,-
Chip Type (Medipix 3.0, Medipix 3.1, Medipix 3RX):	Medipix 3RX
Assembly Size (NX1, 2X2):	   1x1
Chip Mode  (SPM, CSM, CM, CSCM):	SPM
Counter Depth (number):	6
Gain:	SLGM
Active Counters:	Alternating
Thresholds (keV):	1.200000E+2,5.110000E+2,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0
DACs:	175,511,000,000,000,000,000,000,125,255,125,125,100,100,082,100,087,030,128,004,255,129,128,176,168,511,511
bpc File:	c:\MERLIN_Quad_Config\W529_F5\W529_F5_SPM.bpc,,,
DAC File:	c:\MERLIN_Quad_Config\W529_F5\W529_F5_SPM.dacs,,,
Gap Fill Mode:	Distribute
Flat Field File:	None
Dead Time File:	Dummy (C:\<NUL>\)
Acquisition Type (Normal, Th_scan, Config):	Normal
Frames in Acquisition (Number):	{number_of_frames}
Frames per Trigger (Number):	1
Trigger Start (Positive, Negative, Internal):	Rising Edge LVDS
Trigger Stop (Positive, Negative, Internal):	Internal
Sensor Bias (V):	120 V
Sensor Polarity (Positive, Negative):	Positive
Temperature (C):	Board Temp 37.384918 Deg C
Humidity (%):	Board Humidity 1.331848 
Medipix Clock (MHz):	120MHz
Readout System:	Merlin Quad
Software Version:	0.67.0.9
End	                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
"
        );
        let hdr = wrap_into_mpx(hdr.as_bytes());
        stream.write_all(&hdr).unwrap();
    }

    #[test]
    fn test_connect() {
        let env = env_logger::Env::default()
            .filter_or("LIBERTEM_QD_LOG_LEVEL", "error")
            .write_style_or("LIBERTEM_QD_LOG_STYLE", "always");
        let _ = env_logger::Builder::from_env(env)
            .format_timestamp_micros()
            .try_init();

        let (fake_server, local_addr) = make_test_server();

        let (_socket_dir, socket_as_path) = get_socket_path();
        let bytes_per_frame = 2 * 256 * 256;
        let slot_size = bytes_per_frame;
        let shm = SharedSlabAllocator::new(1, slot_size, false, &socket_as_path).unwrap();

        let config = &QdDetectorConnConfig::new(
            "127.0.0.1",
            local_addr.port() as usize,
            1,
            bytes_per_frame,
            1,
            false,
            socket_as_path.to_str().unwrap(),
            None,
        );

        std::thread::scope(move |s| {
            let (shutdown_s, shutdown_r) = channel();

            let server_thread = s.spawn(move || {
                // the server accepts our connection, and...
                let (_stream, _addr) = fake_server.accept().unwrap();
                shutdown_r.recv_timeout(Duration::from_millis(100)).unwrap();
            });

            // we let the `GenericConnection` drive the background thread:
            let bg = QdBackgroundThread::spawn(config, &shm).unwrap();
            let mut conn: GenericConnection<QdBackgroundThread, QdAcquisitionHeader> =
                GenericConnection::new(bg, &shm).unwrap();
            conn.start_passive(
                || Ok::<(), ConnectionError>(()),
                &Some(Duration::from_millis(100)),
            )
            .unwrap();

            // allow the server thread to end:
            shutdown_s.send(()).unwrap();

            // tear down the connection and join the server thread:
            conn.close();
            server_thread.join().unwrap();

            assert_eq!(shm.num_slots_total(), shm.num_slots_free());
        });
    }

    #[test]
    fn test_wait_for_arm() {
        let env = env_logger::Env::default()
            .filter_or("LIBERTEM_QD_LOG_LEVEL", "error")
            .write_style_or("LIBERTEM_QD_LOG_STYLE", "always");
        let _ = env_logger::Builder::from_env(env)
            .format_timestamp_micros()
            .try_init();

        let (fake_server, local_addr) = make_test_server();

        let (_socket_dir, socket_as_path) = get_socket_path();
        let bytes_per_frame = 2 * 256 * 256;
        let slot_size = bytes_per_frame;
        let shm = SharedSlabAllocator::new(1, slot_size, false, &socket_as_path).unwrap();

        let config = &QdDetectorConnConfig::new(
            "127.0.0.1",
            local_addr.port() as usize,
            1,
            bytes_per_frame,
            1,
            false,
            socket_as_path.to_str().unwrap(),
            None,
        );

        std::thread::scope(move |s| {
            let (shutdown_s, shutdown_r) = channel();

            let server_thread = s.spawn(move || {
                // the server accepts our connection, and...
                let (mut stream, _addr) = fake_server.accept().unwrap();

                // simlate: someone armed the detector ("STARTACQUISITION"), meaning
                // the acquisition header is sent out and a real detector would now wait
                // for either a SOFTTRIGGER or a hardware triggering signal:
                send_acq_header(&mut stream, 16384);

                shutdown_r.recv_timeout(Duration::from_millis(500)).unwrap();
            });

            // we let the `GenericConnection` drive the background thread:
            let bg = QdBackgroundThread::spawn(config, &shm).unwrap();
            let mut conn: GenericConnection<QdBackgroundThread, QdAcquisitionHeader> =
                GenericConnection::new(bg, &shm).unwrap();
            conn.start_passive(
                || Ok::<(), ConnectionError>(()),
                &Some(Duration::from_millis(100)),
            )
            .unwrap();

            conn.wait_for_arm(Some(Duration::from_millis(100)), || {
                Ok::<(), ConnectionError>(())
            })
            .unwrap();

            // allow the server thread to end:
            shutdown_s.send(()).unwrap();

            // tear down the connection/bg thread and join the server thread:
            conn.close();
            server_thread.join().unwrap();

            assert_eq!(shm.num_slots_total(), shm.num_slots_free());
        });
    }

    #[test]
    fn test_bg_thread_crash() {
        let env = env_logger::Env::default()
            .filter_or("LIBERTEM_QD_LOG_LEVEL", "error")
            .write_style_or("LIBERTEM_QD_LOG_STYLE", "always");
        let _ = env_logger::Builder::from_env(env)
            .format_timestamp_micros()
            .try_init();

        let (fake_server, local_addr) = make_test_server();

        let (_socket_dir, socket_as_path) = get_socket_path();
        let bytes_per_frame = 256 * 256;
        let slot_size = bytes_per_frame;
        let mut shm = SharedSlabAllocator::new(100, slot_size, false, &socket_as_path).unwrap();

        let config = &QdDetectorConnConfig::new(
            "127.0.0.1",
            local_addr.port() as usize,
            1,
            bytes_per_frame,
            1,
            false,
            socket_as_path.to_str().unwrap(),
            None,
        );

        std::thread::scope(move |s| {
            let (shutdown_s, shutdown_r) = channel();

            let server_thread = s.spawn(move || {
                // the server accepts our connection, and...
                let (mut stream, _addr): (std::net::TcpStream, SocketAddr) =
                    fake_server.accept().unwrap();

                // simulate: we send some garbage, the background thread should
                // handle that and reconnect:
                let hdr = r"PROTOCOL_VIOLATION";
                stream.write_all(hdr.as_bytes()).unwrap();

                // now, let's accept again and process the next connection cleanly:
                let (mut stream, _addr): (std::net::TcpStream, SocketAddr) =
                    fake_server.accept().unwrap();

                send_acq_header(&mut stream, 16);
                send_frames(&mut stream, 16);

                shutdown_r.recv_timeout(Duration::from_millis(100)).unwrap();
            });

            // we let the `GenericConnection` drive the background thread:
            let bg = QdBackgroundThread::spawn(config, &shm).unwrap();
            let mut conn: GenericConnection<QdBackgroundThread, QdAcquisitionHeader> =
                GenericConnection::new(bg, &shm).unwrap();
            conn.start_passive(
                || Ok::<(), ConnectionError>(()),
                &Some(Duration::from_millis(100)),
            )
            .unwrap();

            let res = conn.wait_for_arm(Some(Duration::from_millis(100)), || {
                Ok::<(), ConnectionError>(())
            });
            assert!(res.is_err());

            assert!(matches!(res, Err(ConnectionError::FatalError(_))));

            // ok, we should be able to reconnect:
            conn.start_passive(
                || Ok::<(), ConnectionError>(()),
                &Some(Duration::from_millis(100)),
            )
            .unwrap();

            conn.wait_for_arm(Some(Duration::from_millis(100)), || {
                Ok::<(), ConnectionError>(())
            })
            .unwrap();

            let mut idx = 0;
            loop {
                let stack = conn
                    .get_next_stack(1, || Ok::<_, std::io::Error>(()))
                    .unwrap();

                if let Some(stack) = stack {
                    eprintln!("got a stack with {} frames", stack.len());
                    stack.with_slot(&shm, |s| {
                        assert_eq!(s.as_slice(), vec![idx as u8; 256 * 256],);
                    });
                    stack.free_slot(&mut shm);
                    idx += 1;
                } else {
                    eprintln!("done!");
                    break;
                }
            }

            assert_eq!(idx, 16);

            // allow the server thread to end:
            shutdown_s.send(()).unwrap();

            // tear down the connection/bg thread and join the server thread:
            conn.close();
            server_thread.join().unwrap();

            assert_eq!(shm.num_slots_total(), shm.num_slots_free());
        });
    }

    #[test]
    fn test_full_acquisition() {
        let env = env_logger::Env::default()
            .filter_or("LIBERTEM_QD_LOG_LEVEL", "error")
            .write_style_or("LIBERTEM_QD_LOG_STYLE", "always");
        let _ = env_logger::Builder::from_env(env)
            .format_timestamp_micros()
            .try_init();

        let (fake_server, local_addr) = make_test_server();

        let (_socket_dir, socket_as_path) = get_socket_path();
        let bytes_per_frame = 256 * 256;
        let slot_size = bytes_per_frame;
        let mut shm = SharedSlabAllocator::new(100, slot_size, false, &socket_as_path).unwrap();

        let config = &QdDetectorConnConfig::new(
            "127.0.0.1",
            local_addr.port() as usize,
            1,
            bytes_per_frame,
            1,
            false,
            socket_as_path.to_str().unwrap(),
            None,
        );

        std::thread::scope(move |s| {
            let (shutdown_s, shutdown_r) = channel();

            let server_thread = s.spawn(move || {
                // the server accepts our connection, and...
                let (mut stream, _addr) = fake_server.accept().unwrap();

                // simlate: someone armed the detector ("STARTACQUISITION"), meaning
                // the acquisition header is sent out and a real detector would now wait
                // for either a SOFTTRIGGER or a hardware triggering signal:
                send_acq_header(&mut stream, 16);

                // someone sent a trigger signal to the detector trigger input, so let's start to "record some data":
                send_frames(&mut stream, 16);

                shutdown_r.recv_timeout(Duration::from_millis(500)).unwrap();
            });

            // we let the `GenericConnection` drive the background thread:
            let bg = QdBackgroundThread::spawn(config, &shm).unwrap();
            let mut conn: GenericConnection<QdBackgroundThread, QdAcquisitionHeader> =
                GenericConnection::new(bg, &shm).unwrap();
            conn.start_passive(
                || Ok::<(), ConnectionError>(()),
                &Some(Duration::from_millis(100)),
            )
            .unwrap();

            conn.wait_for_arm(Some(Duration::from_millis(100)), || {
                Ok::<(), ConnectionError>(())
            })
            .unwrap();

            let mut idx = 0;
            loop {
                let stack = conn
                    .get_next_stack(1, || Ok::<_, std::io::Error>(()))
                    .unwrap();

                if let Some(stack) = stack {
                    eprintln!("got a stack with {} frames", stack.len());
                    stack.with_slot(&shm, |s| {
                        assert_eq!(s.as_slice(), vec![idx as u8; 256 * 256],);
                    });
                    stack.free_slot(&mut shm);
                } else {
                    eprintln!("done!");
                    break;
                }
                idx += 1;
            }

            // allow the server thread to end:
            shutdown_s.send(()).unwrap();

            // tear down the connection/bg thread and join the server thread:
            conn.close();
            server_thread.join().unwrap();

            assert_eq!(shm.num_slots_total(), shm.num_slots_free());
        });
    }

    #[test]
    fn test_shared_memory_backpressure() {
        // This test deliberately creates a shared memory area that is too
        // small, and makes sure the code behaves well in this situation,
        // meaning it generates backpressure on the tcp socket.

        let env = env_logger::Env::default()
            .filter_or("LIBERTEM_QD_LOG_LEVEL", "error")
            .write_style_or("LIBERTEM_QD_LOG_STYLE", "always");
        let _ = env_logger::Builder::from_env(env)
            .format_timestamp_micros()
            .try_init();

        let (fake_server, local_addr) = make_test_server();

        let (_socket_dir, socket_as_path) = get_socket_path();
        let bytes_per_frame = 256 * 256;

        // NOTE: only 5 slots, meaning we can have three slots for splitting
        // plus some for users
        let mut shm = SharedSlabAllocator::new(5, bytes_per_frame, false, &socket_as_path).unwrap();

        let config = &QdDetectorConnConfig::new(
            "127.0.0.1",
            local_addr.port() as usize,
            1,
            bytes_per_frame,
            5,
            false,
            socket_as_path.to_str().unwrap(),
            None,
        );

        std::thread::scope(move |s| {
            let (shutdown_s, shutdown_r) = channel();

            let server_thread = s.spawn(move || {
                // the server accepts our connection, and...
                let (mut stream, _addr) = fake_server.accept().unwrap();

                // simlate: someone armed the detector ("STARTACQUISITION"), meaning
                // the acquisition header is sent out and a real detector would now wait
                // for either a SOFTTRIGGER or a hardware triggering signal:
                send_acq_header(&mut stream, 256);

                // someone sent a trigger signal to the detector trigger input, so let's start to "record some data":
                send_frames(&mut stream, 256);

                shutdown_r.recv_timeout(Duration::from_secs(10)).unwrap();
            });

            // we let the `GenericConnection` drive the background thread:
            let bg = QdBackgroundThread::spawn(config, &shm).unwrap();
            let mut conn: GenericConnection<QdBackgroundThread, QdAcquisitionHeader> =
                GenericConnection::new(bg, &shm).unwrap();
            conn.start_passive(
                || Ok::<(), ConnectionError>(()),
                &Some(Duration::from_millis(100)),
            )
            .unwrap();

            conn.wait_for_arm(Some(Duration::from_millis(100)), || {
                Ok::<(), ConnectionError>(())
            })
            .unwrap();

            let mut idx = 0;
            loop {
                let stack = conn
                    .get_next_stack(1, || Ok::<_, std::io::Error>(()))
                    .unwrap();

                // let's be a bit slow in consuming the data:
                std::thread::sleep(Duration::from_millis(1));

                if let Some(stack) = stack {
                    eprintln!("got a stack with {} frames", stack.len());
                    stack.with_slot(&shm, |s| {
                        assert_eq!(s.as_slice(), vec![idx as u8; 256 * 256],);
                    });
                    stack.free_slot(&mut shm);
                } else {
                    eprintln!("done!");
                    break;
                }
                idx += 1;
            }

            // allow the server thread to end:
            shutdown_s.send(()).unwrap();

            // tear down the connection/bg thread and join the server thread:
            conn.close();
            server_thread.join().unwrap();

            assert_eq!(shm.num_slots_total(), shm.num_slots_free());
        });
    }

    #[test]
    fn test_drain() {
        let env = env_logger::Env::default()
            .filter_or("LIBERTEM_QD_LOG_LEVEL", "error")
            .write_style_or("LIBERTEM_QD_LOG_STYLE", "always");
        let _ = env_logger::Builder::from_env(env)
            .format_timestamp_micros()
            .try_init();

        let (fake_server, local_addr) = make_test_server();

        let (_socket_dir, socket_as_path) = get_socket_path();
        let bytes_per_frame = 256 * 256;
        let slot_size = bytes_per_frame;

        let mut shm = SharedSlabAllocator::new(100, slot_size, false, &socket_as_path).unwrap();

        let config = &QdDetectorConnConfig::new(
            "127.0.0.1",
            local_addr.port() as usize,
            1,
            bytes_per_frame,
            1,
            false,
            socket_as_path.to_str().unwrap(),
            Some(Duration::from_millis(100)),
        );

        std::thread::scope(move |s| {
            let (shutdown_s, shutdown_r) = channel();
            let (arm_s, arm_r) = channel();

            let server_thread = s.spawn(move || {
                // the server accepts our connection, and...
                let (mut stream, _addr) = fake_server.accept().unwrap();

                // just write some straight garbage:
                stream.write_all(&vec![0xFFu8; 10024 * 1024]).unwrap();

                // wait for the arm signal (timeout so we don't wait forever here):
                arm_r.recv_timeout(Duration::from_secs(10)).unwrap();

                // simlate: someone armed the detector ("STARTACQUISITION"), meaning
                // the acquisition header is sent out and a real detector would now wait
                // for either a SOFTTRIGGER or a hardware triggering signal:
                send_acq_header(&mut stream, 256);

                // someone sent a trigger signal to the detector trigger input, so let's start to "record some data":
                send_frames(&mut stream, 256);

                shutdown_r.recv_timeout(Duration::from_secs(10)).unwrap();
            });

            // we let the `GenericConnection` drive the background thread:
            let bg = QdBackgroundThread::spawn(config, &shm).unwrap();
            let mut conn: GenericConnection<QdBackgroundThread, QdAcquisitionHeader> =
                GenericConnection::new(bg, &shm).unwrap();
            conn.start_passive(
                || Ok::<(), ConnectionError>(()),
                &Some(Duration::from_millis(200)),
            )
            .unwrap();

            // arm the fake detector and send the acquisition header:
            arm_s.send(()).unwrap();

            conn.wait_for_arm(Some(Duration::from_millis(100)), || {
                Ok::<(), ConnectionError>(())
            })
            .unwrap();

            let mut idx = 0;
            loop {
                let stack = conn
                    .get_next_stack(1, || Ok::<_, std::io::Error>(()))
                    .unwrap();

                // let's be a bit slow in consuming the data:
                std::thread::sleep(Duration::from_millis(1));

                if let Some(stack) = stack {
                    eprintln!("got a stack with {} frames", stack.len());
                    stack.with_slot(&shm, |s| {
                        assert_eq!(s.as_slice(), vec![idx as u8; 256 * 256],);
                    });
                    stack.free_slot(&mut shm);
                } else {
                    eprintln!("done!");
                    break;
                }
                idx += 1;
            }

            // allow the server thread to end:
            shutdown_s.send(()).unwrap();

            // tear down the connection/bg thread and join the server thread:
            conn.close();
            server_thread.join().unwrap();

            assert_eq!(shm.num_slots_total(), shm.num_slots_free());
        });
    }
}
