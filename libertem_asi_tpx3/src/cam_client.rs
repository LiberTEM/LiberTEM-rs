use std::ffi::c_int;

use ipc_test::SharedSlabAllocator;
use log::trace;
use pyo3::{exceptions::PyRuntimeError, ffi::PyMemoryView_FromMemory, prelude::*};

use crate::{
    chunk_stack::{ChunkCSRLayout, ChunkStackHandle},
    exceptions::ConnectionError,
};
#[cfg(test)]
use zerocopy::{AsBytes, FromBytes};

#[pyclass]
pub struct CamClient {
    shm: Option<SharedSlabAllocator>,
}

impl CamClient {
    fn get_memoryview(&self, py: Python, raw_data: &[u8]) -> PyObject {
        let ptr = raw_data.as_ptr();
        let length = raw_data.len();

        let mv = unsafe {
            PyMemoryView_FromMemory(ptr as *mut i8, length.try_into().unwrap(), PyBUF_READ)
        };
        unsafe { Py::from_owned_ptr(py, mv) }
    }

    #[cfg(test)]
    fn get_slices<'a, I, IP, V>(
        &'a self,
        slot_r: &'a ipc_test::Slot,
        handle: &'a ChunkStackHandle,
    ) -> Vec<(&[IP], &[I], &[V])>
    where
        I: numpy::Element + FromBytes + AsBytes + std::marker::Copy,
        IP: numpy::Element + FromBytes + AsBytes + std::marker::Copy,
        V: numpy::Element + FromBytes + AsBytes + std::marker::Copy,
    {
        handle
            .get_chunk_views::<I, IP, V>(slot_r)
            .iter()
            .map(|view| (view.indptr, view.indices, view.values))
            .collect()
    }
}

#[allow(non_upper_case_globals)]
const PyBUF_READ: c_int = 0x100;

#[pymethods]
impl CamClient {
    #[new]
    fn new(handle_path: &str) -> PyResult<Self> {
        match SharedSlabAllocator::connect(handle_path) {
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
    ) -> PyResult<Vec<(ChunkCSRLayout, PyObject, PyObject, PyObject)>> {
        if let Some(shm) = &self.shm {
            let slot_r = shm.get(handle.slot.slot_idx);

            Ok(handle
                .get_chunk_views_raw(&slot_r)
                .iter()
                .map(|(view, layout)| {
                    let indptr = self.get_memoryview(py, view.get_indptr_raw());
                    let indices = self.get_memoryview(py, view.get_indices_raw());
                    let values = self.get_memoryview(py, view.get_values_raw());
                    (layout.clone(), indptr, indices, values)
                })
                .collect())
        } else {
            Err(PyRuntimeError::new_err(
                "CamClient.get_chunk called with SHM closed",
            ))
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
    use tempfile::tempdir;

    use ipc_test::SharedSlabAllocator;
    use pyo3::{prepare_freethreaded_python, Python};

    use crate::{
        cam_client::CamClient,
        chunk_stack::{ChunkCSRLayout, ChunkStackForWriting, ChunkStackHandle},
        csr_view::CSRViewMut,
        headers::DType,
        sparse_csr::CSRSizes,
    };

    #[test]
    fn test_cam_client() {
        let socket_dir = tempdir().unwrap();
        let socket_as_path = socket_dir.into_path().join("stuff.socket");
        let handle_path = socket_as_path.to_str().unwrap();

        let mut shm = SharedSlabAllocator::new(1, 4096, false, &socket_as_path).unwrap();
        let slot = shm.get_mut().expect("get a free shm slot");

        // let's make a plan first:
        const NNZ: u32 = 12;
        const NROWS: u32 = 7;
        const SIZES: CSRSizes = CSRSizes::new::<u32, u32, u32>(NNZ, NROWS);

        let mut fs = ChunkStackForWriting::new(slot, 1);

        assert_eq!(fs.cursor, 0);
        let meta = ChunkCSRLayout {
            nframes: NROWS,
            nnz: NNZ,
            data_length_bytes: SIZES.total(),
            indptr_dtype: DType::U32,
            indptr_offset: 0,
            indptr_size: SIZES.indptr,
            indices_dtype: DType::U32,
            indices_offset: SIZES.indptr,
            indices_size: SIZES.indices,
            value_dtype: DType::U32,
            value_offset: SIZES.indptr + SIZES.indices,
            value_size: SIZES.values,
        };
        meta.validate();
        let slice = fs.slice_for_writing(SIZES.total(), meta.clone());
        let mut view_mut: CSRViewMut<u32, u32, u32> = CSRViewMut::from_bytes(slice, &SIZES);
        let values: Vec<u32> = (0..12).map(|i| (1 << (i % 16))).collect();
        let indices: Vec<u32> = (0..12).collect();
        let indptr: Vec<u32> = vec![0, 4, 8, 12, 12, 12, 12, 12];
        view_mut.copy_from_slices(&indptr, &indices, &values);

        println!("values: {values:?}");
        assert_eq!(values.len(), 12);
        println!("indices: {indices:?}");
        assert_eq!(indices.len(), 12);
        println!("indptr: {indptr:?}");
        assert_eq!(indptr.len() as u32, NROWS + 1);
        println!("meta: {meta:?}");

        assert_eq!(fs.cursor, SIZES.total());
        assert_eq!(fs.num_frames(), 7);

        let fs_handle = fs.writing_done(&mut shm);

        assert_eq!(fs_handle.len(), NROWS);

        // initialize a Python interpreter so we are able to construct a PyBytes instance:
        prepare_freethreaded_python();

        // roundtrip serialize/deserialize:
        Python::with_gil(|py| {
            let bytes = fs_handle.serialize(py).unwrap();
            let new_handle = ChunkStackHandle::deserialize_impl(&bytes).unwrap();
            assert_eq!(fs_handle, new_handle);
        });

        // See that we can get the data out again, unchanged:
        let client = CamClient::new(handle_path).unwrap();
        let slot_r: ipc_test::Slot = shm.get(fs_handle.slot.slot_idx);
        let slice = slot_r.as_slice();
        println!("{slice:x?}");
        let slices = client.get_slices::<u32, u32, u32>(&slot_r, &fs_handle);
        assert_eq!(slices.len(), 1);

        assert_eq!(
            slices.first().unwrap(),
            &(&indptr[..], &indices[..], &values[..])
        );
    }
}
