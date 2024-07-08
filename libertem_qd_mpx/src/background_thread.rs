use std::{
    io::{self, ErrorKind, Read},
    net::TcpStream,
    sync::mpsc::{channel, Receiver, RecvTimeoutError, Sender, TryRecvError},
    thread::JoinHandle,
    time::Duration,
};

use common::{
    background_thread::{BackgroundThread, BackgroundThreadSpawnError, ControlMsg, ReceiverMsg},
    utils::{num_from_byte_slice, NumParseError},
};
use ipc_test::SharedSlabAllocator;
use log::{debug, error, info, trace, warn};

use crate::base_types::{
    AcqHeaderParseError, FrameMetaParseError, QdAcquisitionHeader, QdDetectorConnConfig,
    QdFrameMeta,
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

    #[error("I/O error: {source}")]
    IOError {
        #[from]
        source: io::Error,
    },
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
/// Assumes a blocking socket with a timeout.
fn read_exact_interruptible(
    stream: &mut impl Read,
    buf: &mut [u8],
    to_thread_r: &Receiver<QdControlMsg>,
) -> Result<(), AcquisitionError> {
    let mut buf_sliced = buf;
    loop {
        check_for_control(to_thread_r)?;
        match stream.read(buf_sliced) {
            Ok(size) => {
                buf_sliced = &mut buf_sliced[size..];
                // it's full! we are done...
                if buf_sliced.is_empty() {
                    return Ok(());
                }
            }
            Err(e) => match e.kind() {
                ErrorKind::WouldBlock | ErrorKind::TimedOut => {
                    check_for_control(to_thread_r)?;
                    continue;
                }
                _ => return Err(AcquisitionError::from(e)),
            },
        }
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
    loop {
        check_for_control(to_thread_r)?;
        match stream.peek(buf) {
            Ok(size) => {
                if size == buf.len() {
                    // it's full! we are done...
                    return Ok(());
                }
            }
            Err(e) => match e.kind() {
                // in this case, we couldn't peek the full size, so we try again
                ErrorKind::WouldBlock | ErrorKind::TimedOut => {
                    check_for_control(to_thread_r)?;
                    continue;
                }
                _ => return Err(AcquisitionError::from(e)),
            },
        }
    }
}

/// Read MPX message from `stream`
///
/// Message is in the form: "MPX,<length>,<MSGTYPE>,<MSGPAYLOAD>"
///
/// Where <length> is the length of "<MSGTYPE>,<MSGPAYLOAD>" and is encoded
/// as a 10-character decimal number w/ leading zeros.
fn read_mpx_message(
    stream: &mut impl Read,
    to_thread_r: &Receiver<QdControlMsg>,
) -> Result<Vec<u8>, AcquisitionError> {
    // read and parse the 'MPX,<length>,' part:
    let mut prefix = [0u8; 15];
    read_exact_interruptible(stream, &mut prefix, to_thread_r)?;

    let magic = &prefix[0..4];
    if magic != b"MPX," {
        return Err(AcquisitionError::HeaderParseError {
            msg: format!(
                "expected 'MPX,', got: '{:X?}' (maybe previous read was short?)",
                magic
            ),
        });
    }
    let length = num_from_byte_slice(&prefix[4..14])?;

    let mut payload = vec![0u8; length];
    read_exact_interruptible(stream, &mut payload, to_thread_r)?;

    Ok(payload)
}

/// Peek MPX message from `stream`
///
/// Message is in the form: "MPX,<length>,<MSGTYPE>,<MSGPAYLOAD>"
///
/// Where <length> is the length of "<MSGTYPE>,<MSGPAYLOAD>" and is encoded
/// as a 10-character decimal number w/ leading zeros.
fn peek_mpx_message(
    stream: &mut TcpStream,
    to_thread_r: &Receiver<QdControlMsg>,
) -> Result<Vec<u8>, AcquisitionError> {
    // read and parse the 'MPX,<length>,' part:
    let mut prefix = [0u8; 15];
    peek_exact_interruptible(stream, &mut prefix, to_thread_r)?;

    let magic = &prefix[0..4];
    if magic != b"MPX," {
        return Err(AcquisitionError::HeaderParseError {
            msg: format!(
                "expected 'MPX,', got: '{:X?}' (maybe previous read was short?)",
                magic
            ),
        });
    }
    let length: usize = num_from_byte_slice(&prefix[4..14])?;

    // subtle difference w/ `read_mpx_message`: we need to have a larger buffer here,
    // because here we get the prefix again!
    let mut prefix_and_payload = vec![0u8; length + 15];
    peek_exact_interruptible(stream, &mut prefix_and_payload, to_thread_r)?;
    Ok(prefix_and_payload[15..].to_vec())
}

fn read_acquisition_header(
    stream: &mut impl Read,
    to_thread_r: &Receiver<QdControlMsg>,
) -> Result<QdAcquisitionHeader, AcquisitionError> {
    let acq_header_raw = read_mpx_message(stream, to_thread_r)?;
    Ok(QdAcquisitionHeader::parse_bytes(&acq_header_raw)?)
}

fn peek_first_frame_header(
    stream: &mut TcpStream,
    to_thread_r: &Receiver<QdControlMsg>,
) -> Result<QdFrameMeta, AcquisitionError> {
    let msg = peek_mpx_message(stream, to_thread_r)?;
    Ok(QdFrameMeta::parse_bytes(&msg)?)
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
    todo!();
}

fn passive_acquisition(
    to_thread_r: &Receiver<QdControlMsg>,
    from_thread_s: &Sender<QdReceiverMsg>,
    config: &QdDetectorConnConfig,
    shm: &mut SharedSlabAllocator,
) -> Result<(), AcquisitionError> {
    let host = &config.data_host;
    let port = config.data_port;

    loop {
        let data_uri = format!("{host}:{port}");
        trace!("connecting to {data_uri}...");

        check_for_control(to_thread_r)?;
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

        // wait for the acquisition header, which is sent when the detector
        // is armed with STARTACQUISITION:
        let acquisition_header: QdAcquisitionHeader =
            read_acquisition_header(&mut stream, to_thread_r)?;

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
            &config,
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
                    match passive_acquisition(to_thread_r, from_thread_s, &config, &mut shm) {
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
