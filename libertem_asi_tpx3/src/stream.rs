use std::net::TcpStream;

use common::tcp::{read_exact_interruptible, ReadExactError};

use crate::{
    background_thread::ControlError,
    headers::{HeaderTypes, WireFormatError},
};

#[derive(Debug)]
pub enum StreamError {
    Timeout,
    IoError(std::io::Error),
    Eof,
    FormatError(WireFormatError),
    ControlError(ControlError),
}

impl From<std::io::Error> for StreamError {
    fn from(value: std::io::Error) -> Self {
        match value.kind() {
            std::io::ErrorKind::TimedOut => StreamError::Timeout,
            _ => StreamError::IoError(value),
        }
    }
}

impl From<WireFormatError> for StreamError {
    fn from(value: WireFormatError) -> Self {
        Self::FormatError(value)
    }
}

impl From<ControlError> for StreamError {
    fn from(value: ControlError) -> Self {
        Self::ControlError(value)
    }
}

impl From<ReadExactError<ControlError>> for StreamError {
    fn from(value: ReadExactError<ControlError>) -> Self {
        match value {
            ReadExactError::Interrupted { size: _, err } => Self::ControlError(err),
            ReadExactError::IOError { err } => Self::IoError(err),
            ReadExactError::PeekError { size: _ } => Self::Timeout,
            ReadExactError::Eof => Self::Eof,
        }
    }
}

pub fn stream_recv_header(
    stream: &mut TcpStream,
    header_bytes: &mut [u8; 32],
    timeout_cb: impl Fn() -> Result<(), ControlError>,
) -> Result<HeaderTypes, StreamError> {
    read_exact_interruptible(stream, header_bytes, timeout_cb)?;
    Ok(HeaderTypes::from_bytes(header_bytes)?)
}

/// Fill `buf` with data
pub fn stream_recv_chunk(
    stream: &mut TcpStream,
    buf: &mut [u8],
    timeout_cb: impl Fn() -> Result<(), ControlError>,
) -> Result<(), StreamError> {
    //
    // Possible situations to handle:
    //
    // - Timeout -> the slot contains partial data and should be discarded
    //   => the connection should be closed in this case, probably, because by throwing the error
    //      we lose the state about the position in the stream
    //      (and/or, the error type can include the number of bytes we were still expecting)
    //
    // - EAGAIN, EWOULDBLOCK: Interrupted in the middle, we just try again
    //   => Either we manage to successfully read the rest of the data without hitting a timeout
    //   => Or we then hit a timeout, see above
    //
    // - A control message arrived from the main thread, which we need to handle.
    //
    read_exact_interruptible(stream, buf, timeout_cb)?;
    Ok(())
}
