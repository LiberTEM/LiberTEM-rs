#![forbid(clippy::unwrap_used)]
//! Raw memory backend using memfd with huge page support
use std::{
    fs::{remove_file, File},
    io::{self, Read, Write},
    ops::Deref,
    os::{
        fd::{AsFd, AsRawFd, FromRawFd, RawFd},
        unix::net::{UnixListener, UnixStream},
    },
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread::JoinHandle,
};

use log::debug;
use memfd::{FileSeal, HugetlbSize, MemfdOptions};
use memmap2::{MmapOptions, MmapRaw};
use nix::poll::{PollFd, PollFlags};
use sendfd::{RecvWithFd, SendWithFd};
use serde::{de::DeserializeOwned, Serialize};

use crate::common::ShmConnectError;

fn read_size(mut stream: &UnixStream) -> Result<usize, ShmConnectError> {
    let mut buf: [u8; std::mem::size_of::<usize>()] = [0; std::mem::size_of::<usize>()];
    stream.read_exact(&mut buf)?;
    Ok(usize::from_be_bytes(buf))
}

/// connect to the given unix domain socket and grab a SHM handle
fn recv_shm_handle<H>(socket_path: &Path) -> Result<(H, RawFd), ShmConnectError>
where
    H: DeserializeOwned,
{
    let stream = UnixStream::connect(socket_path)?;

    let mut fds: [i32; 1] = [0];

    let size = read_size(&stream)?;

    // message must be longer than 0:
    assert!(size > 0);

    let mut bytes: Vec<u8> = vec![0; size];

    stream.recv_with_fd(bytes.as_mut_slice(), &mut fds)?;

    let payload: H = bincode::deserialize(&bytes[..])?;
    Ok((payload, fds[0]))
}

fn handle_connection(
    mut stream: UnixStream,
    fd: RawFd,
    init_data_serialized: &[u8],
) -> Result<(), ShmConnectError> {
    let fds = [fd];

    // message must not be empty:
    assert!(!init_data_serialized.is_empty());

    stream.write_all(&init_data_serialized.len().to_be_bytes())?;
    stream.send_with_fd(init_data_serialized, &fds)?;

    Ok(())
}

/// start a thread that serves shm handles at the given socket path
pub fn serve_shm_handle<I>(
    init_data: I,
    fd: RawFd,
    socket_path: &Path,
) -> Result<(Arc<AtomicBool>, JoinHandle<()>), ShmConnectError>
where
    I: Serialize,
{
    let stop_event = Arc::new(AtomicBool::new(false));

    if socket_path.exists() {
        remove_file(socket_path).expect("remove existing socket");
    }

    let listener = UnixListener::bind(socket_path)?;
    let outer_stop = Arc::clone(&stop_event);
    let init_data_serialized = bincode::serialize(&init_data)?;

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
                    let my_init = init_data_serialized.clone();
                    std::thread::spawn(move || {
                        handle_connection(stream, fd, &my_init)
                            .expect("could not let other side connect")
                    });
                }
                Err(err) => {
                    /* EAGAIN / EWOULDBLOCK */
                    if err.kind() == io::ErrorKind::WouldBlock {
                        let flags = PollFlags::POLLIN;
                        let pollfd = PollFd::new(listener.as_fd(), flags);
                        if let Err(e) = nix::poll::poll(&mut [pollfd], 100u16) {
                            log::error!("poll failed: {e}");
                        }
                        continue;
                    }
                    /* connection failed */
                    log::error!("connection failed: {err}");
                    break;
                }
            }
        }
    });

    Ok((outer_stop, join_handle))
}

pub struct MemfdShm {
    mmap: MmapRaw,

    #[allow(dead_code)] // we keep the file around such that it is cleaned up on drop
    file: File,
    socket_path: PathBuf,
    bg_thread: Option<(Arc<AtomicBool>, JoinHandle<()>)>,
}

impl MemfdShm {
    /// Create a new memfd-backed shared memory allocation
    ///
    /// The `socket_path` should point to name in a directory with proper
    /// permissions. The socket will be removed if it already exists.
    ///
    /// `size` is the total size specified in bytes, and can not be changed
    /// after creation.
    ///
    /// If `enable_huge` is specified and not enough huge pages are available
    /// from the operating system, mapping the memory area can fail.
    ///
    pub fn new<I>(
        enable_huge: bool,
        socket_path: &Path,
        size: usize,
        init_data: I,
    ) -> Result<Self, ShmConnectError>
    where
        I: Serialize,
    {
        let memfd_options = MemfdOptions::default().allow_sealing(true);
        let memfd_options = if enable_huge {
            memfd_options.hugetlb(Some(HugetlbSize::Huge2MB))
        } else {
            memfd_options
        };
        let memfd = memfd_options
            .create("MemfdShm")
            .map_err(|e| ShmConnectError::Other { msg: e.to_string() })?;
        let file = memfd.as_file();
        file.set_len(size as u64)?;

        memfd
            .add_seals(&[FileSeal::SealShrink, FileSeal::SealGrow])
            .map_err(|e| ShmConnectError::Other { msg: e.to_string() })?;
        memfd
            .add_seal(FileSeal::SealSeal)
            .map_err(|e| ShmConnectError::Other { msg: e.to_string() })?;

        let file = memfd.into_file();
        let mmap = MmapOptions::new().map_raw(&file)?;

        let bg_thread = serve_shm_handle(&init_data, file.as_raw_fd(), socket_path)?;

        Ok(Self {
            mmap,
            file,
            socket_path: socket_path.to_owned(),
            bg_thread: Some(bg_thread),
        })
    }

    pub fn as_mut_ptr(&self) -> *mut u8 {
        self.mmap.as_mut_ptr()
    }

    pub fn get_handle(&self) -> String {
        self.socket_path.to_string_lossy().deref().to_owned()
    }

    pub fn connect<I>(handle: &str) -> Result<(Self, I), ShmConnectError>
    where
        I: DeserializeOwned,
    {
        let socket_path = Path::new(handle);
        let (init_data, fd) = recv_shm_handle::<I>(socket_path)?;

        // safety: we exlusively own the fd, which we just received via
        // the unix domain socket, so it must be open and valid.
        let file = unsafe { File::from_raw_fd(fd) };
        let mmap = MmapOptions::new().map_raw(&file)?;

        Ok((
            Self {
                mmap,
                file,
                socket_path: socket_path.to_owned(),
                bg_thread: None,
            },
            init_data,
        ))
    }
}

impl Drop for MemfdShm {
    fn drop(&mut self) {
        if let Some((stop_flag, join_handle)) = self.bg_thread.take() {
            stop_flag.store(true, Ordering::Relaxed);
            join_handle
                .join()
                .expect("could not join background thread!");
        }
    }
}

pub type Shm = MemfdShm;

#[cfg(test)]
mod tests {
    use crate::{shm::Shm, test_utils::TempDir, SlabInfo};

    #[test]
    fn test_debug_send() {
        fn need_send<S>(_x: S)
        where
            S: Send,
        {
            // nothing in here, just should give us a compilation error
        }

        let handle_path = TempDir::new("handle_path");

        let slab_info: SlabInfo = SlabInfo {
            num_slots: 7,
            slot_size: 1,
            total_size: 7,
        };

        let s: Shm = Shm::new(
            false,
            &handle_path.join("handle.sock"),
            1024 * 1024,
            slab_info,
        )
        .unwrap();

        need_send(s);
    }

    #[test]
    fn test_debug_sync() {
        fn need_sync<S>(_x: S)
        where
            S: Sync,
        {
            // nothing in here, just should give us a compilation error
        }

        let handle_path = TempDir::new("handle_path");

        let slab_info: SlabInfo = SlabInfo {
            num_slots: 7,
            slot_size: 1,
            total_size: 7,
        };

        let s: Shm = Shm::new(
            false,
            &handle_path.join("handle.sock"),
            1024 * 1024,
            slab_info,
        )
        .unwrap();

        need_sync(s);
    }
}
