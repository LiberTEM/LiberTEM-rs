use bincode::serialize;
use ipc_test::{SharedSlabAllocator, SlotForWriting, SlotInfo, Slot};
use log::trace;
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
    sparse_csr::{CSRSizes, CSRSplitter}, csr_view_raw::CSRViewRaw, common::align_to,
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

#[pymethods]
impl ChunkCSRLayout {
    pub fn get_nframes(&self) -> u32 {
        self.nframes
    }

    pub fn get_nnz(&self) -> u32 {
        self.nnz
    }

    pub fn get_value_dtype(&self) -> String {
        self.value_dtype.to_str().to_string()
    }

    pub fn get_indices_dtype(&self) -> String {
        self.indices_dtype.to_str().to_string()
    }

    pub fn get_indptr_dtype(&self) -> String {
        self.indptr_dtype.to_str().to_string()
    }
}

/// The data arrives as chunks of N frames (for example for a full line in a rectangular scan).
/// We then put multiple of these array chunks into one shared memory slot.
pub struct ChunkStackForWriting {
    slot: SlotForWriting,
    layout: Vec<ChunkCSRLayout>,

    /// where in the slot do the chunks begin? this can be unevenly spaced
    offsets: Vec<usize>,

    /// offset where the next chunk will be written
    pub(crate) cursor: usize,

    padding_bytes: usize,

    /// number of bytes reserved for each array chunk (currently: scan line)
    /// because of varying occupancy, this is just the "planning" number
    bytes_per_chunk: usize,
}

impl ChunkStackForWriting {
    pub fn new(slot: SlotForWriting, capacity: usize, bytes_per_chunk: usize) -> Self {
        ChunkStackForWriting {
            slot,
            cursor: 0,
            padding_bytes: 0,
            bytes_per_chunk,
            // reserve a bit more, as we don't know the upper bound of frames
            // per stack and using a bit more memory is better than having to
            // resize the vectors
            layout: Vec::with_capacity(2 * capacity),
            offsets: Vec::with_capacity(2 * capacity),
        }
    }

    /// number of frames in this chunk stack
    pub fn num_frames(&self) -> usize {
        self.layout.iter().map(|layout| layout.get_nframes() as usize).sum()
    }

    pub fn num_chunks(&self) -> usize {
        self.layout.len()
    }

    pub fn can_fit(&self, num_bytes: usize) -> bool {
        // FIXME: parametrize alignment?
        let num_bytes = align_to(num_bytes, 8);
        self.slot.size - self.cursor >= num_bytes
    }

    /// Total slot size in bytes
    pub fn total_size(&self) -> usize {
        self.slot.size
    }

    pub fn is_empty(&self) -> bool {
        self.num_frames() == 0
    }

    pub fn slice_for_writing(&mut self, nbytes: usize, layout: ChunkCSRLayout) -> &mut [u8] {
        let start = self.cursor;
        let stop = start + nbytes;
        layout.validate();
        self.layout.push(layout);
        self.offsets.push(start);
        let total_size = self.total_size();
        let slice = self.slot.as_slice_mut();
        assert!(start < slice.len(), "start < slice.len() (start={}, len={})", start, slice.len());
        assert!(stop <= slice.len(), "stop <= slice.len() (stop={}, len={})", stop, slice.len());
        let dest = &mut slice[start..stop];
        // FIXME: parametrize alignment?
        let padding = align_to(nbytes, 8) - nbytes;
        self.padding_bytes += padding;
        self.cursor += nbytes + padding;
        assert!(self.cursor <= total_size);
        dest
    }

    pub fn writing_done(self, shm: &mut SharedSlabAllocator) -> ChunkStackHandle {
        assert!(self.cursor <= self.slot.size);
        let slot_info = shm.writing_done(self.slot);

        ChunkStackHandle {
            slot: slot_info,
            layout: self.layout,
            offsets: self.offsets,
            total_bytes_used: self.cursor,
            total_bytes_padding: self.padding_bytes,
        }
    }
}

/// serializable handle for a stack of frames that live in shm
#[pyclass]
#[derive(PartialEq, Eq, Serialize, Deserialize, Debug)]
pub struct ChunkStackHandle {
    pub(crate) slot: SlotInfo,
    layout: Vec<ChunkCSRLayout>,
    pub(crate) offsets: Vec<usize>,
    pub(crate) total_bytes_used: usize,
    pub(crate) total_bytes_padding: usize,
}

impl ChunkStackHandle {
    pub fn len(&self) -> u32 {
        self.layout.iter().map(|layout| layout.nframes).sum()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// total number of _useful_ bytes in this frame stack
    pub fn payload_size(&self) -> usize {
        self.layout.iter().map(|fm| fm.data_length_bytes).sum()
    }

    /// total number of bytes allocated for the slot
    pub fn slot_size(&self) -> usize {
        self.slot.size
    }

    pub fn get_layout(&self) -> &Vec<ChunkCSRLayout> {
        &self.layout
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

        for (layout, offset) in self.layout.iter().zip(&self.offsets) {
            if split_found {
                // we already found our split point, so everything else needs to
                // be copied wholesale to the right stack:
                chunks_right.push((layout.clone(), *offset));
            } else if frames_in_left + layout.nframes < mid {
                // the whole chunk fits into the planned left stack:
                frames_in_left += layout.nframes;
                chunks_left.push((layout.clone(), *offset));
            } else {
                // the chunk doesn't fit into the left stack, so that's the
                // one we need to split:
                let split_size_left = mid - frames_in_left;
                chunk_split = Some((layout.clone(), *offset, split_size_left as usize));
                split_found = true;
            }
        }

        if !split_found {
            panic!(
                "split not found! mid={mid} chunks_left.len()={}, chunks_right.len()={} layout={:?} offsets={:?}",
                chunks_left.len(), chunks_right.len(), self.layout, self.offsets
            );
        }

        assert!(split_found);
        let (chunk_split_layout, chunk_split_offset, split_frames_in_left_part) =
            chunk_split.unwrap();

        let mut left_layout: Vec<ChunkCSRLayout> = Vec::new();
        let mut left_offsets: Vec<usize> = Vec::new();
        let mut left_padding = 0;
        let mut right_layout: Vec<ChunkCSRLayout> = Vec::new();
        let mut right_offsets: Vec<usize> = Vec::new();
        let mut right_padding = 0;

        let (left, left_cursor, right, right_cursor) = {
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

                left_layout.push(m);
            }

            let mut slot_right = shm.get_mut().expect("shm slot for writing");
            let slice_right = slot_right.as_slice_mut();

            let split_slice_src =
                &slice[chunk_split_offset..chunk_split_offset + chunk_split_layout.data_length_bytes];
            let csr_to_split = CSRSplitter::from_bytes(
                split_slice_src,
                chunk_split_layout,
            );

            let relative_mid = split_frames_in_left_part;
            let (lm, rm) = csr_to_split.get_split_info(relative_mid);
            lm.validate();
            rm.validate();

            let split_dst_left = &mut slice_left[left_cursor..left_cursor + lm.data_length_bytes];
            let split_dst_right = &mut slice_right[0..rm.data_length_bytes];

            let (split_left_layout, split_right_layout) =
                csr_to_split.split_into(relative_mid, split_dst_left, split_dst_right);

            left_layout.push(split_left_layout);
            left_offsets.push(left_cursor);

            right_layout.push(split_right_layout);
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

                right_layout.push(m);
            }

            let left = shm.writing_done(slot_left);
            let right = shm.writing_done(slot_right);

            shm.free_idx(self.slot.slot_idx);

            (left, left_cursor, right, right_cursor)
        };

        (
            ChunkStackHandle {
                slot: left,
                layout: left_layout,
                offsets: left_offsets,
                total_bytes_used: left_cursor,
                total_bytes_padding: left_padding,
            },
            ChunkStackHandle {
                slot: right,
                layout: right_layout,
                offsets: right_offsets,
                total_bytes_used: right_cursor,
                total_bytes_padding: right_padding,
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

        self
            .get_layout()
            .iter()
            .zip(&self.offsets)
            .map(|(layout, offset)| {
                let arr_data = &raw_data[*offset..layout.data_length_bytes];
                let view = CSRView::from_bytes(arr_data, layout.nnz, layout.nframes);
                view
            })
            .collect()
    }

    pub fn get_chunk_views_raw<'a>(
        &'a self,
        slot_r: &'a Slot,
    ) -> Vec<(CSRViewRaw, ChunkCSRLayout)> {
        let raw_data = slot_r.as_slice();

        self
            .get_layout()
            .iter()
            .zip(&self.offsets)
            .map(|(layout, offset)| {
                let arr_data = &raw_data[*offset..layout.data_length_bytes];
                layout.validate();
                trace!("constructing csr view from layout {layout:?} with arr_data length {}", arr_data.len());
                (CSRViewRaw::from_bytes_with_layout(arr_data, layout), layout.clone())
            })
            .collect()
    }

    fn first_layout(&self) -> PyResult<&ChunkCSRLayout> {
        self.layout.first().map_or_else(
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
        Ok(match &slf.first_layout()?.value_dtype {
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
        let layout = ChunkCSRLayout {
            nframes: 1,
            nnz: 1,
            data_length_bytes: 4,
            indptr_dtype: DType::U8,
            indptr_offset: 0,
            indptr_size: 2,
            indices_dtype: DType::U8,
            indices_offset: 2,
            indices_size: 1,
            value_dtype: DType::U8,
            value_offset: 3,
            value_size: 1,
        };
        layout.validate();
        let slice = fs.slice_for_writing(1, layout);
        slice[0] = 42;
        assert_eq!(fs.cursor, 8);

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

        let layout = ChunkCSRLayout {
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
        layout.validate();

        let slice = fs.slice_for_writing(SIZES.total(), layout.clone());
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
        println!("layout: {layout:?}");

        assert_eq!(fs.cursor, SIZES.total());

        println!("{:?}", &fs.slot.as_slice()[..SIZES.total()]);

        let fs_handle = fs.writing_done(&mut shm);

        let slot_r = shm.get(fs_handle.slot.slot_idx);

        let old_layout_len = fs_handle.layout.len();

        let (a, b) = fs_handle.split_at(2, &mut shm);

        let slot_a: Slot = shm.get(a.slot.slot_idx);
        let slot_b: Slot = shm.get(b.slot.slot_idx);
        let slice_a = &slot_a.as_slice();
        let slice_b = &slot_b.as_slice();

        println!("a.first_layout() = {:?}", a.first_layout());
        println!("b.first_layout() = {:?}", b.first_layout());

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

    #[test]
    fn test_can_fit() {
        // slot size is rounded up to 4k blocks
        let mut shm = SharedSlabAllocator::new(3, 4096, false).unwrap();
        let slot = shm.get_mut().expect("get a free shm slot");

        let mut fs = ChunkStackForWriting::new(slot, 1, 4096);

        assert!(fs.can_fit(4096));
        assert!(fs.can_fit(1024));
        assert!(fs.can_fit(512));
        assert!(fs.can_fit(0));
        assert!(fs.can_fit(1));

        println!("{}", fs.total_size());
        assert!(!fs.can_fit(4097));

        const NNZ: u32 = 12;
        const NROWS: u32 = 7;
        const SIZES: CSRSizes = CSRSizes::new::<u32, u32, u32>(NNZ, NROWS);
        let layout = ChunkCSRLayout {
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
        layout.validate();

        fs.slice_for_writing(1, layout.clone());
        assert_eq!(fs.padding_bytes, 7);
        assert_eq!(fs.cursor, 8);

        assert!(!fs.can_fit(4096));
        assert!(fs.can_fit(4088));

        fs.slice_for_writing(4088, layout.clone());
        assert_eq!(fs.padding_bytes, 7);
        assert_eq!(fs.cursor, 4096);
        assert!(!fs.can_fit(1));
    }
}
