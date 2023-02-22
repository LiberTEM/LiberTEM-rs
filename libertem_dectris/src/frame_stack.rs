use bincode::serialize;
use ipc_test::{SharedSlabAllocator, SlotForWriting, SlotInfo};
use pyo3::{
    exceptions::{self, PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyBytes, PyType},
};
use serde::{Deserialize, Serialize};

use crate::common::{DConfig, DImage, DImageD, FrameMeta, PixelType};

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

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// take frame metadata and put it into our list
    /// and copy data into the right shm position
    /// (we can't have zero-copy directly into shm with 0mq sadly, as far as I
    /// can tell)
    /// TODO: in a general library, this should also have a zero-copy api available!
    pub fn frame_done(
        &mut self,
        dimage: DImage,
        dimaged: DImageD,
        dconfig: DConfig,
        data: &[u8],
    ) -> FrameMeta {
        let start = self.cursor;
        let stop = start + data.len();
        let dest = &mut self.slot.as_slice_mut()[start..stop];
        // FIXME: return error on slice length mismatch, don't panic
        dest.copy_from_slice(data);
        let meta = FrameMeta {
            dimage,
            dimaged,
            dconfig,
            data_length_bytes: data.len(),
        };
        self.meta.push(meta.clone());
        self.offsets.push(self.cursor);
        self.cursor += data.len();
        meta
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

    fn get_series_id(slf: PyRef<Self>) -> PyResult<u64> {
        Ok(slf.first_meta()?.dimage.series)
    }

    fn get_frame_id(slf: PyRef<Self>) -> PyResult<u64> {
        Ok(slf.first_meta()?.dimage.frame)
    }

    fn get_hash(slf: PyRef<Self>) -> PyResult<String> {
        Ok(slf.first_meta()?.dimage.hash.clone())
    }

    fn get_pixel_type(slf: PyRef<Self>) -> PyResult<String> {
        Ok(match &slf.first_meta()?.dimaged.type_ {
            PixelType::Uint8 => "uint8".to_string(),
            PixelType::Uint16 => "uint16".to_string(),
            PixelType::Uint32 => "uint32".to_string(),
        })
    }

    fn get_encoding(slf: PyRef<Self>) -> PyResult<String> {
        Ok(slf.first_meta()?.dimaged.encoding.clone())
    }

    /// return endianess in numpy notation
    fn get_endianess(slf: PyRef<Self>) -> PyResult<String> {
        match slf.first_meta()?.dimaged.encoding.chars().last() {
            Some(c) => Ok(c.to_string()),
            None => Err(exceptions::PyValueError::new_err(
                "encoding should be non-empty".to_string(),
            )),
        }
    }

    fn get_shape(slf: PyRef<Self>) -> PyResult<Vec<u64>> {
        Ok(slf.first_meta()?.dimaged.shape.clone())
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

    #[test]
    fn test_frame_stack() {
        let (_socket_dir, socket_as_path) = get_socket_path();
        let mut shm = SharedSlabAllocator::new(1, 4096, false, &socket_as_path).unwrap();
        let slot = shm.get_mut().expect("get a free shm slot");
        let mut fs = FrameStackForWriting::new(slot, 1, 256);
        let dimage = crate::common::DImage {
            htype: "".to_string(),
            series: 1,
            frame: 1,
            hash: "".to_string(),
        };
        let dimaged = crate::common::DImageD {
            htype: "".to_string(),
            shape: vec![512, 512],
            type_: crate::common::PixelType::Uint16,
            encoding: ">bslz4".to_string(),
        };
        let dconfig = crate::common::DConfig {
            htype: "".to_string(),
            start_time: 0,
            stop_time: 0,
            real_time: 0,
        };
        assert_eq!(fs.cursor, 0);
        fs.frame_done(dimage, dimaged, dconfig, &[42]);
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
        let dimage = crate::common::DImage {
            htype: "".to_string(),
            series: 1,
            frame: 1,
            hash: "".to_string(),
        };
        let dimaged = crate::common::DImageD {
            htype: "".to_string(),
            shape: vec![512, 512],
            type_: crate::common::PixelType::Uint16,
            encoding: ">bslz4".to_string(),
        };
        let dconfig = crate::common::DConfig {
            htype: "".to_string(),
            start_time: 0,
            stop_time: 0,
            real_time: 0,
        };
        assert_eq!(fs.cursor, 0);
        fs.frame_done(
            dimage.clone(),
            dimaged.clone(),
            dconfig.clone(),
            &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        );
        assert_eq!(fs.cursor, 16);
        fs.frame_done(
            dimage,
            dimaged,
            dconfig,
            &[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        );
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
