use ipc_test::{SharedSlabAllocator, Slot, SlotForWriting, SlotInfo};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    PyErr,
};
use serde::{Deserialize, Serialize};

pub trait FrameMeta: Clone + Serialize {
    /// Length of the data that belongs to the frame corresponding to this meta object
    fn get_data_length_bytes(&self) -> usize;
}

#[derive(thiserror::Error, Debug)]
pub enum FrameStackError {
    #[error("no frames in this stack")]
    Empty,

    #[error("could not serialize / deserialize: {0}")]
    SerdeError(#[from] bincode::Error),
}

impl From<FrameStackError> for PyErr {
    fn from(value: FrameStackError) -> Self {
        match value {
            FrameStackError::Empty => PyValueError::new_err(value.to_string()),
            FrameStackError::SerdeError(e) => PyRuntimeError::new_err(e.to_string()),
        }
    }
}

pub struct FrameStackForWriting<M>
where
    M: FrameMeta,
{
    slot: SlotForWriting,
    meta: Vec<M>,

    /// where in the slot do the frames begin? this can be unevenly spaced
    offsets: Vec<usize>,

    /// offset where the next frame will be written
    cursor: usize,

    /// number of bytes reserved for each frame
    /// as some frames compress better or worse, this is just the "planning" number,
    /// and not a guarantee!
    bytes_per_frame: usize,
}

impl<M> FrameStackForWriting<M>
where
    M: FrameMeta,
{
    pub fn new(slot: SlotForWriting, capacity: usize, bytes_per_frame: usize) -> Self {
        FrameStackForWriting {
            slot,
            cursor: 0,
            bytes_per_frame,
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

    pub fn get_cursor(&self) -> usize {
        self.cursor
    }

    pub fn can_fit(&self, num_bytes: usize) -> bool {
        self.slot.size - self.cursor >= num_bytes
    }

    pub fn slot_size(&self) -> usize {
        self.slot.size
    }

    pub fn bytes_free(&self) -> usize {
        self.slot.size - self.cursor
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Temporarily take mutable ownership of a buffer
    /// for a single frame, and receive the data into it.
    pub fn write_frame<E>(
        &mut self,
        meta: &M,
        mut fill_buffer: impl FnMut(&mut [u8]) -> Result<(), E>,
    ) -> Result<(), E> {
        let start = self.cursor;
        let stop = start + meta.get_data_length_bytes();
        let dest = &mut self.slot.as_slice_mut()[start..stop];
        fill_buffer(dest)?;

        self.meta.push(meta.clone());
        self.offsets.push(self.cursor);
        self.cursor += meta.get_data_length_bytes();

        Ok(())
    }

    pub fn writing_done(self, shm: &mut SharedSlabAllocator) -> FrameStackHandle<M> {
        let slot_info = shm.writing_done(self.slot);

        FrameStackHandle {
            slot: slot_info,
            meta: self.meta,
            offsets: self.offsets,
            bytes_per_frame: self.bytes_per_frame,
        }
    }
}

/// serializable handle for a stack of frames that live in shm
#[derive(PartialEq, Eq, Serialize, Deserialize, Debug)]
pub struct FrameStackHandle<M>
where
    M: FrameMeta,
{
    pub(crate) slot: SlotInfo,
    meta: Vec<M>,
    pub(crate) offsets: Vec<usize>,
    pub(crate) bytes_per_frame: usize,
}

impl<M> FrameStackHandle<M>
where
    M: FrameMeta,
{
    pub fn len(&self) -> usize {
        self.meta.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// total number of _useful_ bytes in this frame stack
    pub fn payload_size(&self) -> usize {
        self.meta.iter().map(|fm| fm.get_data_length_bytes()).sum()
    }

    /// total number of bytes allocated for the slot
    pub fn slot_size(&self) -> usize {
        self.slot.size
    }

    pub fn get_meta(&self) -> &Vec<M> {
        &self.meta
    }

    pub fn deserialize_impl(serialized: &[u8]) -> Result<Self, FrameStackError>
    where
        M: for<'a> Deserialize<'a>,
    {
        Ok(bincode::deserialize(serialized)?)
    }

    pub fn serialize(&self) -> Result<Vec<u8>, FrameStackError> {
        Ok(bincode::serialize(self)?)
    }

    pub fn get_slice_for_frame<'a>(&'a self, frame_idx: usize, slot: &'a ipc_test::Slot) -> &[u8] {
        let slice = slot.as_slice();
        let in_offset = self.offsets[frame_idx];
        let size = self.meta[frame_idx].get_data_length_bytes();
        &slice[in_offset..in_offset + size]
    }

    /// Split self at `mid` and create two new `FrameStackHandle`s.
    /// The first will contain frames with indices [0..mid), the second [mid..len)`
    pub fn split_at(self, mid: usize, shm: &mut SharedSlabAllocator) -> (Self, Self) {
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
            FrameStackHandle {
                slot: left,
                meta: left_meta.to_vec(),
                offsets: left_offsets.to_vec(),
                bytes_per_frame: self.bytes_per_frame,
            },
            FrameStackHandle {
                slot: right,
                meta: right_meta.to_vec(),
                offsets: right_offsets.iter().map(|o| o - bytes_mid).collect(),
                bytes_per_frame: self.bytes_per_frame,
            },
        )
    }

    pub fn first_meta(&self) -> Result<&M, FrameStackError> {
        self.meta
            .first()
            .map_or_else(|| Err(FrameStackError::Empty), Ok)
    }

    pub fn with_slot<T>(&self, shm: &SharedSlabAllocator, mut f: impl FnMut(&Slot) -> T) -> T {
        let slot_r = shm.get(self.slot.slot_idx);
        f(&slot_r)
    }

    pub fn free_slot(self, shm: &mut SharedSlabAllocator) {
        shm.free_idx(self.slot.slot_idx);
    }
}

#[cfg(test)]
mod tests {
    use std::{io::Write, path::PathBuf};

    use ipc_test::{SharedSlabAllocator, Slot};
    use serde::{Deserialize, Serialize};
    use tempfile::{tempdir, TempDir};

    use crate::frame_stack::FrameStackForWriting;

    use super::FrameMeta;

    #[derive(Serialize, Deserialize, Clone)]
    struct MyMeta {
        data_length: usize,
    }

    impl FrameMeta for MyMeta {
        fn get_data_length_bytes(&self) -> usize {
            self.data_length
        }
    }

    fn get_socket_path() -> (TempDir, PathBuf) {
        let socket_dir = tempdir().unwrap();
        let socket_as_path = socket_dir.path().join("stuff.socket");

        (socket_dir, socket_as_path)
    }

    #[test]
    fn test_frame_stack() {
        let (_socket_dir, socket_as_path) = get_socket_path();
        let mut shm = SharedSlabAllocator::new(1, 4096, false, &socket_as_path).unwrap();
        let slot = shm.get_mut().expect("get a free shm slot");
        let mut fs = FrameStackForWriting::new(slot, 1, 256);
        let meta = MyMeta { data_length: 1 };
        assert_eq!(fs.cursor, 0);
        fs.write_frame(&meta, |mut buf| buf.write_all(&[42]))
            .unwrap();
        assert_eq!(fs.cursor, 1);

        let _fs_handle = fs.writing_done(&mut shm);
    }

    #[test]
    fn test_split_frame_stack_handle() {
        let (_socket_dir, socket_as_path) = get_socket_path();
        // need at least three slots: one is the source, two for the results.
        let mut shm = SharedSlabAllocator::new(3, 4096, false, &socket_as_path).unwrap();
        let slot = shm.get_mut().expect("get a free shm slot");
        let mut fs = FrameStackForWriting::new(slot, 2, 16);
        assert_eq!(fs.cursor, 0);
        let meta = MyMeta { data_length: 16 };
        fs.write_frame(&meta, |mut buf| {
            buf.write_all(&[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        })
        .unwrap();
        assert_eq!(fs.cursor, 16);
        fs.write_frame(&meta, |mut buf| {
            buf.write_all(&[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
        })
        .unwrap();
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
