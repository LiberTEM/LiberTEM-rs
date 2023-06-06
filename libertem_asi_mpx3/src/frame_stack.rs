use bincode::serialize;
use ipc_test::{SharedSlabAllocator, SlotForWriting, SlotInfo};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyBytes, PyType},
};
use serde::{Deserialize, Serialize};

use crate::common::FrameMeta;

pub struct FrameStackForWriting {
    slot: SlotForWriting,
    meta: Vec<FrameMeta>,

    /// where in the slot do the frames begin? this can be unevenly spaced
    offsets: Vec<usize>,

    /// offset where the next frame will be written
    pub(crate) cursor: usize,

    /// number of bytes reserved for each frame
    /// as some frames compress better or worse, this is just the "planning" number
    bytes_per_frame: usize,
}

impl FrameStackForWriting {
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
        meta: &FrameMeta,
        mut fill_buffer: impl FnMut(&mut [u8]) -> Result<(), E>,
    ) -> Result<(), E> {
        // FIXME: `fill_buffer` should return a `Result`
        let start = self.cursor;
        let stop = start + meta.data_length_bytes;
        let dest = &mut self.slot.as_slice_mut()[start..stop];
        fill_buffer(dest)?;

        self.meta.push(meta.clone());
        self.offsets.push(self.cursor);
        self.cursor += meta.data_length_bytes;

        Ok(())
    }

    pub fn writing_done(self, shm: &mut SharedSlabAllocator) -> FrameStackHandle {
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
#[pyclass]
#[derive(PartialEq, Eq, Serialize, Deserialize, Debug)]
pub struct FrameStackHandle {
    pub(crate) slot: SlotInfo,
    meta: Vec<FrameMeta>,
    pub(crate) offsets: Vec<usize>,
    pub(crate) bytes_per_frame: usize,
}

impl FrameStackHandle {
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

    pub fn get_meta(&self) -> &Vec<FrameMeta> {
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

    fn first_meta(&self) -> PyResult<&FrameMeta> {
        self.meta.first().map_or_else(
            || Err(PyValueError::new_err("empty frame stack".to_string())),
            Ok,
        )
    }
}

#[pymethods]
impl FrameStackHandle {
    pub fn serialize(&self, py: Python) -> PyResult<Py<PyBytes>> {
        let bytes: &PyBytes = PyBytes::new(py, serialize(self).unwrap().as_slice());
        Ok(bytes.into())
    }

    #[classmethod]
    fn deserialize(_cls: &PyType, serialized: &PyBytes) -> PyResult<Self> {
        Self::deserialize_impl(serialized)
    }

    fn get_frame_id(slf: PyRef<Self>) -> PyResult<u64> {
        Ok(slf.first_meta()?.sequence)
    }

    fn get_shape(slf: PyRef<Self>) -> PyResult<(u16, u16)> {
        let meta = slf.first_meta()?;
        Ok((meta.height, meta.width))
    }

    fn __len__(slf: PyRef<Self>) -> usize {
        slf.len()
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::FrameStackForWriting;
    use ipc_test::{SharedSlabAllocator, Slot};
    use tempfile::{tempdir, TempDir};

    fn get_socket_path() -> (TempDir, PathBuf) {
        let socket_dir = tempdir().unwrap();
        let socket_as_path = socket_dir.path().join("stuff.socket");

        (socket_dir, socket_as_path)
    }
}
