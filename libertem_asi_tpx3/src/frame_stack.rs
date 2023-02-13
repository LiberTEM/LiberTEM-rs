use bincode::serialize;
use ipc_test::{SharedSlabAllocator, SlotForWriting, SlotInfo};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyBytes, PyType},
};
use serde::{Deserialize, Serialize};

use crate::headers::DType;

/// Information about one array chunk
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize, Clone)]
#[pyclass]
pub struct ChunkMeta {
    /// data type of individual pixels
    pub value_dtype: DType,
    pub indptr_dtype: DType,
    pub indices_dtype: DType,

    /// number of frames in this array chunk => rows in the CSR array
    pub nframes: u32,

    /// number of non-zero elements in this array chunk
    pub length: u32,

    /// length of all sub-arrays together in bytes
    pub data_length_bytes: usize,
}

/// The data arrives as chunks of N frames (for example for a full line in a rectangular scan).
/// We then put multiple of these array chunks into one shared memory slot.
pub struct ChunkStackForWriting {
    slot: SlotForWriting,
    meta: Vec<ChunkMeta>,

    /// where in the slot do the chunks begin? this can be unevenly spaced
    offsets: Vec<usize>,

    /// offset where the next chunk will be written
    pub(crate) cursor: usize,

    /// number of bytes reserved for each array chunk (currently: scan line)
    /// because of varying occupancy, this is just the "planning" number
    bytes_per_chunk: usize,
}

impl ChunkStackForWriting {
    pub fn new(slot: SlotForWriting, capacity: usize, bytes_per_chunk: usize) -> Self {
        ChunkStackForWriting {
            slot,
            cursor: 0,
            bytes_per_chunk,
            // reserve a bit more, as we don't know the upper bound of frames
            // per stack and using a bit more memory is better than having to
            // resize the vectors
            meta: Vec::with_capacity(2 * capacity),
            offsets: Vec::with_capacity(2 * capacity),
        }
    }

    pub fn len(&self) -> usize {
        self.meta.len()
    }

    pub fn can_fit(&self, num_bytes: usize) -> bool {
        self.slot.size - self.cursor >= num_bytes
    }

    /// Total slot size in bytes
    pub fn total_size(&self) -> usize {
        self.slot.size
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn slice_for_writing(&mut self, nbytes: usize, meta: ChunkMeta) -> &mut [u8] {
        let start = self.cursor;
        let stop = start + nbytes;
        self.cursor += nbytes;
        self.meta.push(meta);
        let dest = &mut self.slot.as_slice_mut()[start..stop];
        dest
    }

    pub fn writing_done(self, shm: &mut SharedSlabAllocator) -> ChunkStackHandle {
        let slot_info = shm.writing_done(self.slot);

        ChunkStackHandle {
            slot: slot_info,
            meta: self.meta,
            offsets: self.offsets,
            bytes_per_frame: self.bytes_per_chunk,
        }
    }
}

/// serializable handle for a stack of frames that live in shm
#[pyclass]
#[derive(PartialEq, Eq, Serialize, Deserialize, Debug)]
pub struct ChunkStackHandle {
    pub(crate) slot: SlotInfo,
    meta: Vec<ChunkMeta>,
    pub(crate) offsets: Vec<usize>,
    pub(crate) bytes_per_frame: usize,
}

impl ChunkStackHandle {
    pub fn len(&self) -> usize {
        self.meta.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// total number of _useful_ bytes in this frame stack
    pub fn payload_size(&self) -> usize {
        self.meta.iter().map(|fm| fm.data_length_bytes).sum()
    }

    /// total number of bytes allocated for the slot
    pub fn slot_size(&self) -> usize {
        self.slot.size
    }

    pub fn get_meta(&self) -> &Vec<ChunkMeta> {
        &self.meta
    }

    pub(crate) fn deserialize_impl(serialized: &PyBytes) -> PyResult<Self> {
        let data = serialized.as_bytes();
        bincode::deserialize(data).map_err(|e| {
            let msg = format!("could not deserialize FrameStackHandle: {e:?}");
            PyRuntimeError::new_err(msg)
        })
    }

    pub fn get_slice_for_frame<'a>(&'a self, frame_idx: usize, slot: &'a ipc_test::Slot) -> &[u8] {
        let slice = slot.as_slice();
        let in_offset = self.offsets[frame_idx];
        let size = self.meta[frame_idx].data_length_bytes;
        &slice[in_offset..in_offset + size]
    }

    /// Split self at `mid` and create two new `FrameStackHandle`s.
    /// The first will contain frames with indices [0..mid), the second [mid..len)`
    pub fn split_at(self, mid: usize, shm: &mut SharedSlabAllocator) -> (Self, Self) {
        todo!("implement chunk stack splitting!");

        // FIXME: this whole thing is falliable, so modify return type to Result<> (or PyResult<>?)
        let bytes_mid = self.offsets[mid];
        let (left, right) = {
            let slot: ipc_test::Slot = shm.get(self.slot.slot_idx);
            let slice = slot.as_slice();

            let mut slot_left = shm.get_mut().expect("shm slot for writing");
            let slice_left = slot_left.as_slice_mut();

            let mut slot_right = shm.get_mut().expect("shm slot for writing");
            let slice_right = slot_right.as_slice_mut();

            slice_left[..bytes_mid].copy_from_slice(&slice[..bytes_mid]);
            slice_right[..(slice.len() - bytes_mid)].copy_from_slice(&slice[bytes_mid..]);

            let left = shm.writing_done(slot_left);
            let right = shm.writing_done(slot_right);

            shm.free_idx(self.slot.slot_idx);

            (left, right)
        };

        let (left_meta, right_meta) = self.meta.split_at(mid);
        let (left_offsets, right_offsets) = self.offsets.split_at(mid);

        (
            ChunkStackHandle {
                slot: left,
                meta: left_meta.to_vec(),
                offsets: left_offsets.to_vec(),
                bytes_per_frame: self.bytes_per_frame,
            },
            ChunkStackHandle {
                slot: right,
                meta: right_meta.to_vec(),
                offsets: right_offsets.iter().map(|o| o - bytes_mid).collect(),
                bytes_per_frame: self.bytes_per_frame,
            },
        )
    }

    fn first_meta(&self) -> PyResult<&ChunkMeta> {
        self.meta.first().map_or_else(
            || Err(PyValueError::new_err("empty frame stack".to_string())),
            Ok,
        )
    }
}

#[pymethods]
impl ChunkStackHandle {
    pub fn serialize(&self, py: Python) -> PyResult<Py<PyBytes>> {
        let bytes: &PyBytes = PyBytes::new(py, serialize(self).unwrap().as_slice());
        Ok(bytes.into())
    }

    #[classmethod]
    fn deserialize(_cls: &PyType, serialized: &PyBytes) -> PyResult<Self> {
        Self::deserialize_impl(serialized)
    }

    fn get_pixel_type(slf: PyRef<Self>) -> PyResult<String> {
        Ok(match &slf.first_meta()?.value_dtype {
            DType::U8 => "uint8".to_string(),
            DType::U16 => "uint16".to_string(),
            DType::U32 => "uint32".to_string(),
            DType::U64 => "uint64".to_string(),
            DType::U1 | DType::U4 => todo!("packed integer types U1 and U4 not supported yet"),
        })
    }

    fn __len__(slf: PyRef<Self>) -> usize {
        slf.len()
    }

    fn __repr__(&self) -> String {
        format!("{self:?}")
    }
}

#[cfg(test)]
mod tests {
    use crate::{frame_stack::ChunkMeta, headers::DType};

    use super::ChunkStackForWriting;
    use ipc_test::{SharedSlabAllocator, Slot};

    #[test]
    fn test_frame_stack() {
        let mut shm = SharedSlabAllocator::new(1, 4096, false).unwrap();
        let slot = shm.get_mut().expect("get a free shm slot");
        let mut fs = ChunkStackForWriting::new(slot, 1, 256);
        assert_eq!(fs.cursor, 0);
        let meta = ChunkMeta {
            value_dtype: DType::U8,
            indptr_dtype: DType::U8,
            indices_dtype: DType::U8,
            nframes: 1,
            length: 1,
            data_length_bytes: 1,
        };
        let slice = fs.slice_for_writing(1, meta);
        slice[0] = 42;
        assert_eq!(fs.cursor, 1);

        let _fs_handle = fs.writing_done(&mut shm);
    }

    #[test]
    fn test_split_frame_stack_handle() {
        // need at least three slots: one is the source, two for the results.
        let mut shm = SharedSlabAllocator::new(3, 4096, false).unwrap();
        let slot = shm.get_mut().expect("get a free shm slot");
        let mut fs = ChunkStackForWriting::new(slot, 2, 16);
        assert_eq!(fs.cursor, 0);
        let meta = ChunkMeta {
            value_dtype: DType::U8,
            indptr_dtype: DType::U8,
            indices_dtype: DType::U8,
            nframes: 1,
            length: 16,
            data_length_bytes: 16,
        };
        let slice = fs.slice_for_writing(16, meta);
        slice.copy_from_slice(&[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]);
        assert_eq!(fs.cursor, 16);
        let meta = ChunkMeta {
            value_dtype: DType::U8,
            indptr_dtype: DType::U8,
            indices_dtype: DType::U8,
            nframes: 1,
            length: 16,
            data_length_bytes: 16,
        };
        let slice = fs.slice_for_writing(16, meta);
        slice.copy_from_slice(&[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]);
        assert_eq!(fs.cursor, 32);
        assert_eq!(fs.offsets, vec![0, 16]);

        println!("{:?}", fs.slot.as_slice());

        let fs_handle = fs.writing_done(&mut shm);

        let slot_r = shm.get(fs_handle.slot.slot_idx);
        let slice_0 = fs_handle.get_slice_for_frame(0, &slot_r);
        assert_eq!(slice_0.len(), 16);
        for &elem in slice_0 {
            assert_eq!(elem, 1);
        }
        let slice_1 = fs_handle.get_slice_for_frame(1, &slot_r);
        assert_eq!(slice_1.len(), 16);
        for &elem in slice_1 {
            assert_eq!(elem, 2);
        }

        let old_meta_len = fs_handle.meta.len();

        let (a, b) = fs_handle.split_at(1, &mut shm);

        let slot_a: Slot = shm.get(a.slot.slot_idx);
        let slot_b: Slot = shm.get(b.slot.slot_idx);
        let slice_a = &slot_a.as_slice()[..16];
        let slice_b = &slot_b.as_slice()[..16];
        println!("{:?}", slice_a);
        println!("{:?}", slice_b);
        for &elem in slice_a {
            assert_eq!(elem, 1);
        }
        for &elem in slice_b {
            assert_eq!(elem, 2);
        }
        assert_eq!(slice_a, a.get_slice_for_frame(0, &slot_a));
        assert_eq!(slice_b, b.get_slice_for_frame(0, &slot_b));

        assert_eq!(a.meta.len() + b.meta.len(), old_meta_len);
        assert_eq!(a.offsets.len() + b.offsets.len(), 2);

        // when the split is done, there should be one free shm slot:
        assert_eq!(shm.num_slots_free(), 1);

        // and we can free them again:
        shm.free_idx(a.slot.slot_idx);
        shm.free_idx(b.slot.slot_idx);

        assert_eq!(shm.num_slots_free(), 3);
    }
}
