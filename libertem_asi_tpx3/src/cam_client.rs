use std::ffi::c_int;

use ipc_test::SharedSlabAllocator;
use log::trace;
use pyo3::{exceptions::PyRuntimeError, ffi::PyMemoryView_FromMemory, prelude::*, FromPyPointer};

use crate::{exceptions::ConnectionError, frame_stack::ChunkStackHandle, shm::recv_shm_handle};

#[pyclass]
pub struct CamClient {
    shm: Option<SharedSlabAllocator>,
}

impl CamClient {}

#[allow(non_upper_case_globals)]
const PyBUF_READ: c_int = 0x100;

#[pymethods]
impl CamClient {
    #[new]
    fn new(socket_path: &str) -> PyResult<Self> {
        let handle = recv_shm_handle(socket_path);
        match SharedSlabAllocator::connect(handle.fd, &handle.info) {
            Ok(shm) => Ok(CamClient { shm: Some(shm) }),
            Err(e) => {
                let msg = format!("failed to connect to SHM: {e:?}");
                Err(ConnectionError::new_err(msg))
            }
        }
    }

    fn get_chunks(
        &self,
        handle: &ChunkStackHandle,
        py: Python,
    ) -> PyResult<Vec<PyObject>> {
        if let Some(shm) = &self.shm {
            let slot_r = shm.get(handle.slot.slot_idx);

            let mut cursor: isize = 0;

            Ok(handle.get_meta().iter().map(|meta| {
                let mut ptr = slot_r.ptr;
                let mv = unsafe {
                    ptr = ptr.offset(cursor);
                    PyMemoryView_FromMemory(
                        ptr as *mut i8,
                        slot_r.size.try_into().unwrap(),
                        PyBUF_READ,
                    )
                };
                let from_ptr: &PyAny = unsafe { FromPyPointer::from_owned_ptr(py, mv) };
                cursor += isize::try_from(meta.data_length_bytes).unwrap();
                from_ptr.into_py(py)
            }).collect())
        } else {
            Err(PyRuntimeError::new_err("CamClient.get_chunk called with SHM closed"))
        }
    }

    fn done(mut slf: PyRefMut<Self>, handle: &ChunkStackHandle) -> PyResult<()> {
        let slot_idx = handle.slot.slot_idx;
        if let Some(shm) = &mut slf.shm {
            shm.free_idx(slot_idx);
            Ok(())
        } else {
            Err(PyRuntimeError::new_err(
                "CamClient.done called with SHM closed",
            ))
        }
    }

    fn close(&mut self) {
        self.shm.take();
    }
}

impl Drop for CamClient {
    fn drop(&mut self) {
        trace!("CamClient::drop");
    }
}

#[cfg(test)]
mod tests {
    use numpy::PyArray;
    use tempfile::tempdir;

    use ipc_test::SharedSlabAllocator;
    use pyo3::{prepare_freethreaded_python, Python};
    use zerocopy::AsBytes;

    use crate::{
        cam_client::CamClient,
        frame_stack::{ChunkStackForWriting, ChunkStackHandle, ChunkMeta},
        shm::serve_shm_handle, headers::DType,
    };

    #[test]
    fn test_cam_client() {
        let mut shm = SharedSlabAllocator::new(1, 4096, false).unwrap();
        let slot = shm.get_mut().expect("get a free shm slot");
        let mut fs = ChunkStackForWriting::new(slot, 1, 512);

        // some predictable test data:
        let in_: Vec<u16> = (0..256).map(|i| i % 16).collect();

        assert_eq!(fs.cursor, 0);
        let in_bytes = in_.as_bytes();
        let meta = ChunkMeta {
            value_dtype: DType::U16,
            indptr_dtype: DType::U32,
            indices_dtype: DType::U16,
            nframes: 1,
            length: in_.len() as u32,
            data_length_bytes: in_bytes.len(),
        };
        let slice = fs.slice_for_writing(in_bytes.len(), meta);
        slice.copy_from_slice(in_bytes);
        assert_eq!(fs.cursor, in_bytes.len());

        // we have one frame in there:
        assert_eq!(fs.len(), 1);

        let fs_handle = fs.writing_done(&mut shm);

        // we still have one frame in there:
        assert_eq!(fs_handle.len(), 1);

        // initialize a Python interpreter so we are able to construct a PyBytes instance:
        prepare_freethreaded_python();

        // roundtrip serialize/deserialize:
        Python::with_gil(|py| {
            let bytes = fs_handle.serialize(py).unwrap();
            let new_handle = ChunkStackHandle::deserialize_impl(bytes.as_ref(py)).unwrap();
            assert_eq!(fs_handle, new_handle);
        });

        // start to serve the shm connection via a unix domain socket:
        let handle = shm.get_handle();
        let socket_dir = tempdir().unwrap();
        let socket_as_path = socket_dir.into_path().join("stuff.socket");
        let socket_path = socket_as_path.to_string_lossy();
        serve_shm_handle(handle, &socket_path);

        let client = CamClient::new(&socket_path).unwrap();

        let slot_r: ipc_test::Slot = shm.get(fs_handle.slot.slot_idx);
        let slice = slot_r.as_slice();
        println!("{slice:x?}");

        Python::with_gil(|py| {
            let flat: Vec<u16> = (0..256).collect();
            let out = PyArray::from_vec(py, flat).reshape((1, 16, 16)).unwrap();
            let chunks = client.get_chunks(&fs_handle, py).unwrap();

            out.readonly()
                .as_slice()
                .unwrap()
                .iter()
                .zip(0..)
                .for_each(|(&item, idx)| {
                    assert_eq!(item, in_[idx]);
                    assert_eq!(item, (idx % 16) as u16);
                });
        });
    }
}
