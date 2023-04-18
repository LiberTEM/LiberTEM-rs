use std::{
    ffi::c_int,
    fs::remove_file,
    io::{Read, Write},
    os::unix::net::{UnixListener, UnixStream},
    path::Path,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

use ipc_test::{SHMHandle, SHMInfo, SharedSlabAllocator, Slot};
use pyo3::{ffi::PyMemoryView_FromMemory, prelude::*, FromPyPointer};
use sendfd::{RecvWithFd, SendWithFd};

use crate::Cam;

#[allow(non_upper_case_globals)]
const PyBUF_READ: c_int = 0x100; // somehow not exported by pyo3? oh no...

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
pub fn serve_shm_handle(handle: SHMHandle, socket_path: &str) -> Arc<AtomicBool> {
    let path = Path::new(socket_path);

    let stop_event = Arc::new(AtomicBool::new(false));

    if path.exists() {
        remove_file(path).expect("remove existing socket");
    }

    let listener = UnixListener::bind(socket_path).unwrap();

    let outer_stop = Arc::clone(&stop_event);

    // FIXME: when does the thread shut down? who sets the stop event?
    // FIXME: what is the lifetime of the shared memory? bound to the acquisition?
    std::thread::spawn(move || {
        // Stolen from the example on `UnixListener`:
        // accept connections and process them, spawning a new thread for each one

        // FIXME: can hang here indefinitely, if no connections come in
        for stream in listener.incoming() {
            if stop_event.load(Ordering::Relaxed) {
                break;
            }
            match stream {
                Ok(stream) => {
                    /* connection succeeded */
                    std::thread::spawn(move || handle_connection(stream, handle));
                }
                Err(err) => {
                    /* connection failed */
                    println!("connection failed: {err}");
                    break;
                }
            }
        }
    });

    outer_stop
}

fn read_size(mut stream: &UnixStream) -> usize {
    let mut buf: [u8; std::mem::size_of::<usize>()] = [0; std::mem::size_of::<usize>()];
    stream.read_exact(&mut buf).expect("read message size");
    usize::from_be_bytes(buf)
}

/// connect to the given unix domain socket and grab a SHM handle
fn recv_shm_handle(socket_path: &str) -> SHMHandle {
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

#[pyclass]
pub struct FrameRef {
    slot_idx: usize,
    memview: PyObject,
}

#[pymethods]
impl FrameRef {
    fn get_memoryview(slf: PyRef<Self>) -> Py<PyAny> {
        slf.memview.clone()
    }

    // FIXME: should have some safety - hide the slot index business a bit better
    // and only pass in a whole object, where the inner memoryview can hopefully be
    // properly cleaned up
}

#[pyclass]
pub struct CamClient {
    shm: SharedSlabAllocator,
}

#[pymethods]
impl CamClient {
    #[new]
    fn new(socket_path: &str) -> Self {
        let handle = recv_shm_handle(socket_path);
        let shm = SharedSlabAllocator::connect(handle.fd, &handle.info).expect("connect to shm");
        CamClient { shm }
    }

    fn get_frame_ref(slf: PyRef<Self>, py: Python, slot: usize) -> FrameRef {
        // FIXME: crimes below. need to verify safety, and define the rules that the
        // Python side needs to follow
        let slot_r: Slot = slf.shm.get(slot);
        let mv = unsafe {
            PyMemoryView_FromMemory(
                slot_r.ptr as *mut i8,
                slot_r.size.try_into().unwrap(),
                PyBUF_READ,
            )
        };
        let from_ptr: &PyAny = unsafe { FromPyPointer::from_owned_ptr(py, mv) };
        let memview = from_ptr.into_py(py);

        FrameRef {
            memview,
            slot_idx: slot,
        }
    }

    fn done(mut slf: PyRefMut<Self>, slot_idx: usize) {
        slf.shm.free_idx(slot_idx)
    }
}
