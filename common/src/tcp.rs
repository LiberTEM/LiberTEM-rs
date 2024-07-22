use std::{
    io::{ErrorKind, Read},
    net::TcpStream,
    time::Duration,
};

use log::trace;

#[derive(thiserror::Error, Debug)]
pub enum ReadExactError<E> {
    #[error("read interrupted after {size} bytes; error: {err}")]
    Interrupted { size: usize, err: E },

    #[error("i/o error")]
    IOError {
        #[from]
        err: std::io::Error,
    },

    #[error("eof")]
    Eof,

    #[error("could not peek {size} bytes")]
    PeekError { size: usize },
}

/// Fill `buf` from `stream` (like TcpStream::read_exact); if we hit a read timeout
/// on the stream, invoke `f`. If `f` returns an `Err`, stop reading from the stream.
///
/// In case of interruption, `buf` may contain a partial result, and
/// `ReadExactError::Interrupted` will contain the number of bytes read in the
/// `size` field.
pub fn read_exact_interruptible<E, F>(
    stream: &mut impl Read,
    buf: &mut [u8],
    f: F,
) -> Result<(), ReadExactError<E>>
where
    F: Fn() -> Result<(), E>,
{
    let total_to_read = buf.len();
    let mut buf_sliced = buf;
    let mut bytes_read: usize = 0;
    loop {
        if let Err(e) = f() {
            return Err(ReadExactError::Interrupted {
                size: bytes_read,
                err: e,
            });
        }
        match stream.read(buf_sliced) {
            Ok(size) => {
                bytes_read += size;
                buf_sliced = &mut buf_sliced[size..];
                // it's full! we are done...
                if bytes_read == total_to_read {
                    return Ok(());
                }
                if size == 0 {
                    // we aren't done, but got an EOF... propagate that as an error
                    return Err(ReadExactError::Eof);
                }
            }
            Err(e) => match e.kind() {
                ErrorKind::WouldBlock | ErrorKind::TimedOut => {
                    continue;
                }
                _ => return Err(ReadExactError::from(e)),
            },
        }
    }
}

/// Fill `buf` from `stream`; if we hit a read timeout on the stream, invoke
/// `f`. If `f` returns an `Err`, stop peeking.
///
/// In case of interruption, `buf` may contain a partial result, which is also
/// still in the internal queue of the socket.
///
/// Retry `max_retries` times with a sleep of `retry_interval` in between
/// retries; this allows to break free if the socket buffer is too small to peek
/// the requested amount of data, and sleeping in between means we don't
/// completely hog a CPU core.
pub fn peek_exact_interruptible<E, F>(
    stream: &mut TcpStream,
    buf: &mut [u8],
    retry_interval: Duration,
    max_retries: usize,
    f: F,
) -> Result<(), ReadExactError<E>>
where
    F: Fn() -> Result<(), E>,
{
    loop {
        if let Err(e) = f() {
            return Err(ReadExactError::Interrupted {
                // we may write bytes to `buf`, but they are not consumed from the socket,
                // so we report zero here:
                size: 0,
                err: e,
            });
        }

        let mut retries = 0;
        match stream.peek(buf) {
            Ok(size) => {
                if size == buf.len() {
                    // it's full! we are done...
                    return Ok(());
                } else {
                    trace!(
                        "peek_exact_interruptible: not full; {size} != {}",
                        buf.len()
                    );
                    retries += 1;
                    if retries > max_retries {
                        return Err(ReadExactError::PeekError { size: buf.len() });
                    }
                    // we are only using this for peeking at the first frame, or its header,
                    // so we can be a bit sleepy here:
                    std::thread::sleep(retry_interval);
                }
            }
            Err(e) => {
                trace!("peek_exact_interruptible: err: {e}");
                match e.kind() {
                    // in this case, we couldn't peek the full size, so we try again
                    ErrorKind::WouldBlock | ErrorKind::TimedOut => {
                        continue;
                    }
                    _ => return Err(ReadExactError::from(e)),
                }
            }
        }
    }
}
