use bincode::serialize;
use ipc_test::{SharedSlabAllocator, SlotForWriting, SlotInfo, Slot};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyBytes, PyType},
};
use serde::{Deserialize, Serialize};
use zerocopy::{AsBytes, FromBytes};

use crate::{
    csr_view::CSRView,
    headers::DType,
    sparse_csr::{CSRSizes, SparseCSR}, csr_view_raw::CSRViewRaw,
};

/// Information about one array chunk and the layout of the CSR sub-arrays in memory
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize, Clone)]
#[pyclass]
pub struct ChunkCSRLayout {
    /// data type of individual pixels
    pub indptr_dtype: DType,
    pub indices_dtype: DType,
    pub value_dtype: DType,

    /// number of frames in this array chunk => rows in the CSR array
    pub nframes: u32,

    /// number of non-zero elements in this array chunk
    pub nnz: u32,

    /// length of all sub-arrays together in bytes
    pub data_length_bytes: usize,

    /// offsets and sizes of the array parts in bytes
    pub indptr_offset: usize,
    pub indptr_size: usize,

    pub indices_offset: usize,
    pub indices_size: usize,

    pub value_offset: usize,
    pub value_size: usize,
}

impl ChunkCSRLayout {
    pub fn validate(&self) {
        // validate length and sizes:
        assert_eq!(
            self.data_length_bytes,
            self.indptr_size + self.indices_size + self.value_size
        );
        assert_eq!(
            self.indices_size,
            self.nnz as usize * self.indices_dtype.size()
        );
        assert_eq!(
            self.indptr_size,
            (self.nframes + 1) as usize * self.indptr_dtype.size()
        );
        assert_eq!(self.value_size, self.nnz as usize * self.value_dtype.size());
    }
}

/// The data arrives as chunks of N frames (for example for a full line in a rectangular scan).
/// We then put multiple of these array chunks into one shared memory slot.
pub struct ChunkStackForWriting {
    slot: SlotForWriting,
    meta: Vec<ChunkCSRLayout>,

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

    /// number of _frames_ in this chunk stack
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

    pub fn slice_for_writing(&mut self, nbytes: usize, meta: ChunkCSRLayout) -> &mut [u8] {
        let start = self.cursor;
        let stop = start + nbytes;
        self.cursor += nbytes;
        self.meta.push(meta);
        self.offsets.push(start);
        let dest = &mut self.slot.as_slice_mut()[start..stop];
        dest
    }

    pub fn writing_done(self, shm: &mut SharedSlabAllocator) -> ChunkStackHandle {
        let slot_info = shm.writing_done(self.slot);

        ChunkStackHandle {
            slot: slot_info,
            meta: self.meta,
            offsets: self.offsets,
            bytes_per_chunk: self.bytes_per_chunk,
        }
    }
}

/// serializable handle for a stack of frames that live in shm
#[pyclass]
#[derive(PartialEq, Eq, Serialize, Deserialize, Debug)]
pub struct ChunkStackHandle {
    pub(crate) slot: SlotInfo,
    meta: Vec<ChunkCSRLayout>,
    pub(crate) offsets: Vec<usize>,
    pub(crate) bytes_per_chunk: usize,
}

impl ChunkStackHandle {
    pub fn len(&self) -> u32 {
        self.meta.iter().map(|layout| layout.nframes).sum()
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

    pub fn get_meta(&self) -> &Vec<ChunkCSRLayout> {
        &self.meta
    }

    pub(crate) fn deserialize_impl(serialized: &PyBytes) -> PyResult<Self> {
        let data = serialized.as_bytes();
        bincode::deserialize(data).map_err(|e| {
            let msg = format!("could not deserialize FrameStackHandle: {e:?}");
            PyRuntimeError::new_err(msg)
        })
    }

    /// Split self at array row index `mid` and create two new `FrameStackHandle`s.
    ///
    /// The first will contain frames/rows with indices [0..mid), the second [mid..len)`
    ///
    /// The frames in the `FrameStackHandle`s can be divided up into multiple chunks.
    ///
    /// The length of the left stack returned is equal to `mid`.
    ///
    // FIXME: take into account the alignment of the chunks in the slot
    pub fn split_at(self, mid: u32, shm: &mut SharedSlabAllocator) -> (Self, Self) {
        // FIXME: this whole thing is falliable, so modify return type to Result<> (or PyResult<>?)

        // First, let's plan our operation:
        //
        // - We want to copy 0 to N full chunks, such that the number of "frames"
        //   in all these chunks together is smaller than `mid`.
        // - Then, if we still have space, split the next chunk to completely fill up to `mid`,
        //   and take the "right" part of the split into the right stack, and copy over any
        //   remaining chunks.

        let mut chunks_left: Vec<(ChunkCSRLayout, usize)> = Vec::new();
        let mut chunks_right: Vec<(ChunkCSRLayout, usize)> = Vec::new();
        let mut chunk_split: Option<(ChunkCSRLayout, usize, usize)> = None;

        let mut frames_in_left = 0;
        let mut split_found = false;

        for (m, offset) in self.meta.iter().zip(&self.offsets) {
            if split_found {
                // we already found our split point, so everything else needs to
                // be copied wholesale to the right stack:
                chunks_right.push((m.clone(), *offset));
            } else if frames_in_left + m.nframes < mid {
                // the whole chunk fits into the planned left stack:
                frames_in_left += m.nframes;
                chunks_left.push((m.clone(), *offset));
            } else {
                // the chunk doesn't fit into the left stack, so that's the
                // one we need to split:
                let split_size_left = mid - frames_in_left;
                chunk_split = Some((m.clone(), *offset, split_size_left as usize));
                split_found = true;
            }
        }

        if !split_found {
            panic!(
                "split not found! mid={mid} chunks_left.len()={}, chunks_right.len()={} meta={:?} offsets={:?}",
                chunks_left.len(), chunks_right.len(), self.meta, self.offsets
            );
        }

        assert!(split_found);
        let (chunk_split_meta, chunk_split_offset, split_frames_in_left_part) =
            chunk_split.unwrap();

        let mut left_meta: Vec<ChunkCSRLayout> = Vec::new();
        let mut left_offsets: Vec<usize> = Vec::new();
        let mut right_meta: Vec<ChunkCSRLayout> = Vec::new();
        let mut right_offsets: Vec<usize> = Vec::new();

        let (left, right) = {
            let slot: ipc_test::Slot = shm.get(self.slot.slot_idx);
            let slice = slot.as_slice();

            let mut slot_left = shm.get_mut().expect("shm slot for writing");
            let slice_left = slot_left.as_slice_mut();

            // cursor points to free space in the slot:
            let mut left_cursor = 0;

            for (m, offset) in chunks_left.into_iter() {
                // offset and other meta information from these chunks can be taken as-is; they are not moved
                // relative to the slot beginning:
                left_offsets.push(offset);

                // the payload data can be copied, too:
                let dst = &mut slice_left[offset..offset + m.data_length_bytes];
                let src = &slice[offset..offset + m.data_length_bytes];
                dst.copy_from_slice(src);

                left_cursor = offset + m.data_length_bytes; // FIXME: align here?

                left_meta.push(m);
            }

            let mut slot_right = shm.get_mut().expect("shm slot for writing");
            let slice_right = slot_right.as_slice_mut();

            let split_slice_src =
                &slice[chunk_split_offset..chunk_split_offset + chunk_split_meta.data_length_bytes];
            let m = &chunk_split_meta;
            let csr_to_split = SparseCSR::from_bytes(
                split_slice_src,
                m.indices_dtype,
                m.indptr_dtype,
                m.value_dtype,
                m.nnz,
                m.nframes,
            );

            let relative_mid = split_frames_in_left_part;
            let (lm, rm) = csr_to_split.get_split_info(relative_mid);
            lm.validate();
            rm.validate();

            let split_dst_left = &mut slice_left[left_cursor..left_cursor + lm.data_length_bytes];
            let split_dst_right = &mut slice_right[0..rm.data_length_bytes];

            let (split_left_meta, split_right_meta) =
                csr_to_split.split_into(relative_mid, split_dst_left, split_dst_right);

            left_meta.push(split_left_meta);
            left_offsets.push(left_cursor);

            right_meta.push(split_right_meta);
            right_offsets.push(0);

            let mut right_cursor = split_dst_right.len();

            for (m, offset) in chunks_right.into_iter() {
                // calculate new offset as a running cursor:
                right_offsets.push(right_cursor);

                // FIXME: actually copy over the data itself!
                let dst = &mut slice_right[right_cursor..right_cursor + m.data_length_bytes];
                let src = &slice[offset..offset + m.data_length_bytes];
                dst.copy_from_slice(src);

                right_cursor += m.data_length_bytes; // FIXME: align here?

                right_meta.push(m);
            }

            let left = shm.writing_done(slot_left);
            let right = shm.writing_done(slot_right);

            shm.free_idx(self.slot.slot_idx);

            (left, right)
        };

        (
            ChunkStackHandle {
                slot: left,
                meta: left_meta,
                offsets: left_offsets,
                bytes_per_chunk: self.bytes_per_chunk,
            },
            ChunkStackHandle {
                slot: right,
                meta: right_meta,
                offsets: right_offsets,
                bytes_per_chunk: self.bytes_per_chunk,
            },
        )
    }

    pub fn get_chunk_views<'a, I, IP, V>(
        &'a self,
        slot_r: &'a Slot,
    ) -> Vec<CSRView<I, IP, V>>
    where
        I: numpy::Element + FromBytes + AsBytes,
        IP: numpy::Element + FromBytes + AsBytes,
        V: numpy::Element + FromBytes + AsBytes,
    {
        let raw_data = slot_r.as_slice();
        let mut cursor: usize = 0;

        self
            .get_meta()
            .iter()
            .map(|layout| {
                let arr_data = &raw_data[cursor..layout.data_length_bytes];
                let view = CSRView::from_bytes(arr_data, layout.nnz, layout.nframes);
                cursor += layout.data_length_bytes;
                view
            })
            .collect()
    }

    pub fn get_chunk_views_raw<'a>(
        &'a self,
        slot_r: &'a Slot,
    ) -> Vec<CSRViewRaw> {
        let raw_data = slot_r.as_slice();
        let mut cursor: usize = 0;

        self
            .get_meta()
            .iter()
            .map(|layout| {
                let arr_data = &raw_data[cursor..layout.data_length_bytes];
                CSRViewRaw::from_bytes_with_layout(arr_data, &layout)
            })
            .collect()
    }

    fn first_meta(&self) -> PyResult<&ChunkCSRLayout> {
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
        slf.len() as usize
    }

    fn __repr__(&self) -> String {
        format!("{self:?}")
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        chunk_stack::ChunkCSRLayout,
        csr_view::{CSRView, CSRViewMut},
        headers::DType,
        sparse_csr::CSRSizes,
    };

    use super::ChunkStackForWriting;
    use ipc_test::{SharedSlabAllocator, Slot};

    #[test]
    fn test_chunk_stack() {
        let mut shm = SharedSlabAllocator::new(1, 4096, false).unwrap();
        let slot = shm.get_mut().expect("get a free shm slot");
        let mut fs = ChunkStackForWriting::new(slot, 1, 256);
        assert_eq!(fs.cursor, 0);
        let meta = ChunkCSRLayout {
            nframes: 1,
            nnz: 1,
            data_length_bytes: 3,
            indptr_dtype: DType::U8,
            indptr_offset: 0,
            indptr_size: 1,
            indices_dtype: DType::U8,
            indices_offset: 1,
            indices_size: 1,
            value_dtype: DType::U8,
            value_offset: 2,
            value_size: 1,
        };
        let slice = fs.slice_for_writing(1, meta);
        slice[0] = 42;
        assert_eq!(fs.cursor, 1);

        let _fs_handle = fs.writing_done(&mut shm);
    }

    #[test]
    fn test_split_chunk_stack_handle() {
        // first case tested here: split a chunk stack that contains a single chunk into two

        // need at least three slots: one is the source, two for the results.
        let mut shm = SharedSlabAllocator::new(3, 4096, false).unwrap();
        let slot = shm.get_mut().expect("get a free shm slot");

        // let's make a plan first:
        const NNZ: u32 = 12;
        const NROWS: u32 = 7;
        const SIZES: CSRSizes = CSRSizes::new::<u32, u32, u32>(NNZ, NROWS);

        let mut fs = ChunkStackForWriting::new(slot, 1, SIZES.total());
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
        let mut view_mut: CSRViewMut<u32, u32, u32> = CSRViewMut::from_bytes(slice, NNZ, NROWS);

        // generate some predictable pattern:
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

        println!("{:?}", &fs.slot.as_slice()[..SIZES.total()]);

        let fs_handle = fs.writing_done(&mut shm);

        let slot_r = shm.get(fs_handle.slot.slot_idx);

        let old_meta_len = fs_handle.meta.len();

        let (a, b) = fs_handle.split_at(2, &mut shm);

        let slot_a: Slot = shm.get(a.slot.slot_idx);
        let slot_b: Slot = shm.get(b.slot.slot_idx);
        let slice_a = &slot_a.as_slice();
        let slice_b = &slot_b.as_slice();

        println!("a.first_meta() = {:?}", a.first_meta());
        println!("b.first_meta() = {:?}", b.first_meta());

        // this testcase only splits a single chunk, which is then at the beginning of the slot:
        let view_a: CSRView<u32, u32, u32> = CSRView::from_bytes(slice_a, 8, 2);
        assert_eq!(view_a.indptr, &[0, 4, 8]);
        assert_eq!(view_a.indices, &[0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(view_a.values, &[1, 2, 4, 8, 16, 32, 64, 128]);

        // when the split is done, there should be one free shm slot:
        assert_eq!(shm.num_slots_free(), 1);

        // and we can free them again:
        shm.free_idx(a.slot.slot_idx);
        shm.free_idx(b.slot.slot_idx);

        assert_eq!(shm.num_slots_free(), 3);
    }
}
