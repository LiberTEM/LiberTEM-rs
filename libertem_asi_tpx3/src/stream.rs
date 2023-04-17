use std::{io::Read, net::TcpStream};

use log::warn;

use crate::{
    headers::{HeaderTypes, WireFormatError},
    receiver::ControlError,
};

#[derive(Debug)]
pub enum StreamError {
    Timeout,
    IoError(std::io::Error),
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

type RecvCallback<'a> = Box<dyn Fn(&std::io::ErrorKind) -> Result<(), StreamError> + 'a>;

pub fn stream_recv_header(
    stream: &mut TcpStream,
    header_bytes: &mut [u8; 32],
    timeout_cb: RecvCallback,
) -> Result<HeaderTypes, StreamError> {
    // TODO: timeout? can't block indefinitely - can we make this atomic?

    loop {
        match stream.read_exact(header_bytes) {
            Ok(_) => break,
            Err(e) => match e.kind() {
                std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut => {
                    warn!("stream error: {e}");
                    timeout_cb(&e.kind())?;
                    continue;
                }
                _ => return Err(e.into()),
            },
        }
    }

    Ok(HeaderTypes::from_bytes(header_bytes)?)
}

/// Fill `buf` with data
pub fn stream_recv_chunk(stream: &mut TcpStream, buf: &mut [u8]) -> Result<(), StreamError> {
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

    // Cases:
    // - all goes well, `buf` is filled with data from the stream
    //    - this includes if it was interrupted in the meantime
    // - in case of read timeout, the error is passed down
    stream.read_exact(buf)?;
    Ok(())
}
