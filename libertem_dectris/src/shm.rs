use std::fs::remove_file;
use std::io::{self, Read, Write};
use std::os::fd::AsRawFd;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::Path;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::thread::JoinHandle;

use ipc_test::{SHMHandle, SHMInfo};
use log::debug;
use nix::poll::{PollFd, PollFlags};
use sendfd::{RecvWithFd, SendWithFd};

fn handle_connection(mut stream: UnixStream, handle: SHMHandle) {
    let fds = [handle.fd];
    let info = bincode::serialize(&handle.info).expect("serialize shm info");

    stream
        .write_all(&info.len().to_be_bytes())
        .expect("send shm info size");
    stream
        .send_with_fd(&info, &fds)
        .expect("send shm info with fds");
}

/// start a thread that serves shm handles at the given socket path
pub fn serve_shm_handle(handle: SHMHandle, socket_path: &str) -> (Arc<AtomicBool>, JoinHandle<()>) {
    let path = Path::new(socket_path);

    let stop_event = Arc::new(AtomicBool::new(false));

    if path.exists() {
        remove_file(path).expect("remove existing socket");
    }

    let listener = UnixListener::bind(socket_path).unwrap();

    let outer_stop = Arc::clone(&stop_event);

    listener
        .set_nonblocking(true)
        .expect("set to nonblocking accept");

    let join_handle = std::thread::spawn(move || {
        // Stolen from the example on `UnixListener`:
        // accept connections and process them, spawning a new thread for each one

        loop {
            if stop_event.load(Ordering::Relaxed) {
                debug!("stopping `serve_shm_handle` thread");
                break;
            }
            let stream = listener.accept();
            match stream {
                Ok((stream, _addr)) => {
                    /* connection succeeded */
                    std::thread::spawn(move || handle_connection(stream, handle));
                }
                Err(err) => {
                    /* EAGAIN / EWOULDBLOCK */
                    if err.kind() == io::ErrorKind::WouldBlock {
                        let fd = listener.as_raw_fd();
                        let flags = PollFlags::POLLIN;
                        let pollfd = PollFd::new(fd, flags);
                        nix::poll::poll(&mut [pollfd], 100).expect("poll for socket to be ready");
                        continue;
                    }
                    /* connection failed */
                    log::error!("connection failed: {err}");
                    break;
                }
            }
        }
    });

    (outer_stop, join_handle)
}

fn read_size(mut stream: &UnixStream) -> usize {
    let mut buf: [u8; std::mem::size_of::<usize>()] = [0; std::mem::size_of::<usize>()];
    stream.read_exact(&mut buf).expect("read message size");
    usize::from_be_bytes(buf)
}

/// connect to the given unix domain socket and grab a SHM handle
pub(crate) fn recv_shm_handle(socket_path: &str) -> SHMHandle {
    let stream = UnixStream::connect(socket_path).expect("connect to socket");

    let mut fds: [i32; 1] = [0];

    let size = read_size(&stream);
    let mut bytes: Vec<u8> = vec![0; size];

    stream
        .recv_with_fd(bytes.as_mut_slice(), &mut fds)
        .expect("read initial message with fds");

    let info: SHMInfo = bincode::deserialize(&bytes[..]).expect("deserialize SHMInfo object");

    SHMHandle { fd: fds[0], info }
}
