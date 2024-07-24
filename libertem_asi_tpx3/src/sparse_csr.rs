use std::mem::size_of;

use zerocopy::{AsBytes, FromBytes};

use crate::{
    chunk_stack::ChunkCSRLayout,
    csr_view_raw::{CSRViewRaw, CSRViewRawMut},
    headers::{AcquisitionStart, ArrayChunk, DType},
};

///
/// Helper for splitting CSR arrays
///
/// It's assumed that the parts are in the following order:
///
/// - rowind / indptr
/// - coords / indices
/// - values / data
///
/// Currently, there is also assumed to not be any padding between the parts.
pub struct CSRSplitter<'a> {
    raw_data: &'a [u8],
    layout: ChunkCSRLayout,
}

impl<'a> CSRSplitter<'a> {
    pub fn from_bytes(raw_data: &'a [u8], layout: ChunkCSRLayout) -> Self {
        Self { raw_data, layout }
    }

    pub fn get_split_info(&self, mid: usize) -> (ChunkCSRLayout, ChunkCSRLayout) {
        match self.layout.indptr_dtype {
            DType::U8 => self.get_split_info_generic::<u8>(mid),
            DType::U16 => self.get_split_info_generic::<u16>(mid),
            DType::U32 => self.get_split_info_generic::<u32>(mid),
            DType::U64 | DType::U1 | DType::U4 => {
                panic!(
                    "indptr type {:?} not supported (yet?)",
                    self.layout.indptr_dtype
                )
            }
        }
    }

    /// Get the layouts for left/right chunks
    fn get_split_info_generic<IP>(&self, mid: usize) -> (ChunkCSRLayout, ChunkCSRLayout)
    where
        IP: numpy::Element + FromBytes + AsBytes + Copy + std::ops::Sub<Output = IP>,
        u32: From<IP>,
    {
        let view: CSRViewRaw = CSRViewRaw::from_bytes_with_layout(self.raw_data, &self.layout);

        let left_nnz = view.get_indptr::<IP>()[mid].into();
        let left_nframes = mid;
        let left_indptr_size = size_of::<IP>() * (left_nframes + 1);
        let left_indices_size = left_nnz as usize * self.layout.indices_dtype.size();
        let left_values_size = left_nnz as usize * self.layout.value_dtype.size();
        let left_size = left_indices_size + left_indptr_size + left_values_size;

        let right_nnz = self.layout.nnz - left_nnz;
        let right_nframes = self.layout.nframes as usize - left_nframes;
        let right_indptr_size = size_of::<IP>() * (right_nframes + 1);
        let right_indices_size = right_nnz as usize * self.layout.indices_dtype.size();
        let right_values_size = right_nnz as usize * self.layout.value_dtype.size();
        let right_size = right_indices_size + right_indptr_size + right_values_size;

        assert!(
            left_nframes > 0,
            "left_nframes = 0; self.layout.nframes = {}, mid={}",
            self.layout.nframes,
            mid
        );
        assert!(
            right_nframes > 0,
            "right_nframes = 0; self.layout.nframes = {}, mid={}",
            self.layout.nframes,
            mid
        );

        // FIXME: alignemnt of array parts!
        (
            ChunkCSRLayout {
                indptr_dtype: self.layout.indptr_dtype,
                indices_dtype: self.layout.indices_dtype,
                value_dtype: self.layout.value_dtype,
                nframes: left_nframes.try_into().unwrap(),
                nnz: left_nnz,
                data_length_bytes: left_size,
                indptr_offset: 0,
                indptr_size: left_indptr_size,
                indices_offset: left_indptr_size,
                indices_size: left_indices_size,
                value_offset: left_indptr_size + left_indices_size,
                value_size: left_values_size,
            },
            ChunkCSRLayout {
                indptr_dtype: self.layout.indptr_dtype,
                indices_dtype: self.layout.indices_dtype,
                value_dtype: self.layout.value_dtype,
                nframes: right_nframes.try_into().unwrap(),
                nnz: right_nnz,
                data_length_bytes: right_size,
                indptr_offset: 0,
                indptr_size: right_indptr_size,
                indices_offset: right_indptr_size,
                indices_size: right_indices_size,
                value_offset: right_indptr_size + right_indices_size,
                value_size: right_values_size,
            },
        )
    }

    // FIXME: Result error type
    pub fn split_into(
        &self,
        mid: usize,
        left: &mut [u8],
        right: &mut [u8],
    ) -> (ChunkCSRLayout, ChunkCSRLayout) {
        match self.layout.indptr_dtype {
            DType::U8 => self.split_generic::<u8>(mid, left, right),
            DType::U16 => self.split_generic::<u16>(mid, left, right),
            DType::U32 => self.split_generic::<u32>(mid, left, right),
            DType::U64 | DType::U1 | DType::U4 => {
                panic!(
                    "indptr type {:?} not supported (yet?)",
                    self.layout.indptr_dtype
                )
            }
        }
    }

    /// splits at `mid` and copies results into `left`/`right`.
    /// `left` will contain values for rows with indices [0..mid), `right` will contain [mid..len).
    ///
    /// This is generic over the index pointer type `IP`, as that is the only one we actually
    /// need to interprete in our code - the indices and values we can copy over unseen.
    ///
    /// Returns the layouts of the matrices written into `left` and `right`
    fn split_generic<IP>(
        &self,
        mid: usize,
        left: &mut [u8],
        right: &mut [u8],
    ) -> (ChunkCSRLayout, ChunkCSRLayout)
    where
        IP: numpy::Element + FromBytes + AsBytes + Copy + std::ops::Sub<Output = IP>,
        u32: From<IP>,
    {
        let view: CSRViewRaw = CSRViewRaw::from_bytes_with_layout(self.raw_data, &self.layout);

        let (layout_a, layout_b) = self.get_split_info_generic::<IP>(mid);

        // to find nnz values, we need to look up in the `indptr` array
        // where the values/coords for the left part stop:
        let left_nnz = layout_a.nnz;
        let left_rows = layout_a.nframes;
        let mut left: CSRViewRawMut = CSRViewRawMut::from_bytes_with_layout(left, &layout_a);
        left.copy_into_indices_raw(
            &view.get_indices_raw()[..left_nnz as usize * self.layout.indices_dtype.size()],
        );
        left.copy_into_values_raw(
            &view.get_values_raw()[..left_nnz as usize * self.layout.value_dtype.size()],
        );
        left.copy_into_indptr(&view.get_indptr::<IP>()[..left_rows as usize + 1]);

        // This takes the symmetric parts of the slices above
        let mut right: CSRViewRawMut = CSRViewRawMut::from_bytes_with_layout(right, &layout_b);
        assert_eq!(
            right.get_indices_raw().len(),
            view.get_indices_raw().len() - (left_nnz as usize * self.layout.indices_dtype.size())
        );
        assert_eq!(
            right.get_values_raw().len(),
            view.get_values_raw().len() - (left_nnz as usize * self.layout.value_dtype.size())
        );
        right.copy_into_indices_raw(
            &view.get_indices_raw()[left_nnz as usize * self.layout.indices_dtype.size()..],
        );
        right.copy_into_values_raw(
            &view.get_values_raw()[left_nnz as usize * self.layout.value_dtype.size()..],
        );
        right
            .get_indptr::<IP>()
            .copy_from_slice(&view.get_indptr()[left_rows as usize..]);

        // offset correction: the first `indptr` should point to the first elements in `values`/`indices`,
        // the relative offsets should still be valid.
        let first: IP = right.get_indptr()[0];
        right
            .get_indptr::<IP>()
            .iter_mut()
            .for_each(|ptr| *ptr = *ptr - first);
        (layout_a, layout_b)
    }
}

/// Calculate the size in bytes for the array parts and in total.
#[derive(Debug)]
pub struct CSRSizes {
    pub indptr: usize,

    /// padding _before_ indptr
    pub indptr_padding: usize,

    pub indices: usize,

    /// padding _before_ indices
    pub indices_padding: usize,

    pub values: usize,

    /// padding _before_ values
    pub values_padding: usize,

    pub nnz: u32,
    pub nrows: u32,
}

impl CSRSizes {
    pub const fn new<I, IP, V>(nnz: u32, nrows: u32) -> Self
    where
        I: numpy::Element + FromBytes + AsBytes,
        IP: numpy::Element + FromBytes + AsBytes,
        V: numpy::Element + FromBytes + AsBytes,
    {
        let indptr_size = (nrows as usize + 1) * size_of::<IP>();
        let indices_size = nnz as usize * size_of::<I>();
        let values_size = nnz as usize * size_of::<V>();

        Self {
            indptr: indptr_size,
            indices: indices_size,
            values: values_size,
            nnz,
            nrows,
            indptr_padding: 0,
            indices_padding: 0,
            values_padding: 0,
        }
    }

    pub const fn new_dyn(
        nnz: u32,
        nrows: u32,
        indptr_dtype: DType,
        indices_dtype: DType,
        values_dtype: DType,
    ) -> Self {
        let indptr_size = (nrows as usize + 1) * indptr_dtype.size();
        let indices_size = nnz as usize * indices_dtype.size();
        let values_size = nnz as usize * values_dtype.size();

        Self {
            indptr: indptr_size,
            indices: indices_size,
            values: values_size,
            nnz,
            nrows,
            indptr_padding: 0,
            indices_padding: 0,
            values_padding: 0,
        }
    }

    pub fn from_headers(acquisition_header: &AcquisitionStart, chunk_header: &ArrayChunk) -> Self {
        let nnz = chunk_header.length;
        let indptr_size =
            (chunk_header.nframes as usize + 1) * acquisition_header.indptr_dtype.size();
        let indices_size = nnz as usize * acquisition_header.indices_dtype.size();
        let values_size = nnz as usize * chunk_header.value_dtype.size();

        // in this case, indptr_padding is always zero:
        let indptr_padding = 0;

        // ... so indices_padding must be indices_offset minus the indptr_size:
        let indices_padding = chunk_header.indices_offset as usize - indptr_size;

        // ..........|..|......
        //           ^  \- values_offset
        //            \- indices_offset + indices_size
        let values_padding = chunk_header.values_offset as usize
            - (chunk_header.indices_offset as usize + indices_size);

        Self {
            indptr: indptr_size,
            indices: indices_size,
            values: values_size,
            nnz,
            nrows: chunk_header.nframes,
            indptr_padding,
            indices_padding,
            values_padding,
        }
    }

    pub fn from_layout(layout: &ChunkCSRLayout) -> Self {
        let nnz = layout.nnz;
        let indptr_size = layout.indptr_size;
        let indices_size = layout.indices_size;
        let values_size = layout.value_size;

        // in this case, indptr_padding is always zero:
        let indptr_padding = 0;

        // ... so indices_padding must be indices_offset minus the indptr_size:
        let indices_padding = layout.indices_offset - indptr_size;

        // ..........|..|......
        //           ^  \- values_offset
        //            \- indices_offset + indices_size
        let values_padding = layout.value_offset - (layout.indices_offset + indices_size);

        Self {
            indptr: indptr_size,
            indices: indices_size,
            values: values_size,
            nnz,
            nrows: layout.nframes,
            indptr_padding,
            indices_padding,
            values_padding,
        }
    }

    /// total size in bytes, including padding
    pub const fn total(&self) -> usize {
        let padding_bytes = self.indptr_padding + self.indices_padding + self.values_padding;
        self.indptr + self.indices + self.values + padding_bytes
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        chunk_stack::ChunkCSRLayout,
        csr_view::{CSRView, CSRViewMut},
        headers::DType,
        sparse_csr::{CSRSizes, CSRSplitter},
    };

    #[test]
    fn test_split_csr() {
        // first, let's build a sparse CSR array that is backed by an [u8]
        // (in real usage, this lives in the SHM or is read from the socket
        // into some buffer)

        const NNZ: u32 = 12;
        const NROWS: u32 = 3;
        const SIZES: CSRSizes = CSRSizes::new::<u16, u16, u16>(NNZ, NROWS);
        let mut buf: [u8; SIZES.total()] = [0; SIZES.total()];
        println!("SIZES: {SIZES:?}");

        let layout = ChunkCSRLayout {
            nframes: NROWS,
            nnz: NNZ,
            data_length_bytes: SIZES.total(),
            indptr_dtype: DType::U16,
            indptr_offset: 0,
            indptr_size: SIZES.indptr,
            indices_dtype: DType::U16,
            indices_offset: SIZES.indptr,
            indices_size: SIZES.indices,
            value_dtype: DType::U16,
            value_offset: SIZES.indptr + SIZES.indices,
            value_size: SIZES.values,
        };
        layout.validate();

        let view_mut: CSRViewMut<u16, u16, u16> = CSRViewMut::from_bytes(&mut buf, &SIZES);

        // generate some predictable pattern:
        let values: Vec<u16> = (0..12).map(|i| (1 << (i % 16))).collect();
        println!("values: {values:?}");
        assert_eq!(values.len(), 12);
        view_mut.values.copy_from_slice(&values);

        // put the values somewhere in the rows; doesn't really matter where:
        let indices: Vec<u16> = (0..12).collect();
        println!("indices: {indices:?}");
        assert_eq!(indices.len(), 12);
        view_mut.indices.copy_from_slice(&indices);

        // indices to where the rows start and end:
        // we have three rows, so we should have 4 entries here:
        let indptr: Vec<u16> = vec![0, 4, 8, 12];
        println!("indptr: {indptr:?}");
        assert_eq!(indptr.len() as u32, NROWS + 1);
        view_mut.indptr.copy_from_slice(&indptr);

        // now, we want to split into two parts; let's have two row in the first
        // part and one row in the second part:

        const SIZES_LEFT: CSRSizes = CSRSizes::new::<u16, u16, u16>(8, 2);
        let mut left_buf: [u8; SIZES_LEFT.total()] = [0; SIZES_LEFT.total()];
        println!("SIZES_LEFT: {SIZES_LEFT:?}");

        const SIZES_RIGHT: CSRSizes = CSRSizes::new::<u16, u16, u16>(4, 1);
        let mut right_buf: [u8; SIZES_RIGHT.total()] = [0; SIZES_RIGHT.total()];
        println!("SIZES_RIGHT: {SIZES_RIGHT:?}");

        // the actual split happens here:
        let csr = CSRSplitter::from_bytes(&buf, layout);
        csr.split_into(2, &mut left_buf, &mut right_buf);

        // now, view the results:
        let left_view: CSRView<u16, u16, u16> = CSRView::from_bytes(&left_buf, &SIZES_LEFT);
        assert_eq!(left_view.indices, &[0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(left_view.indptr, &[0, 4, 8]);
        assert_eq!(left_view.values, &[1, 2, 4, 8, 16, 32, 64, 128]);

        let right_view: CSRView<u16, u16, u16> = CSRView::from_bytes(&right_buf, &SIZES_RIGHT);

        assert_eq!(right_view.indices, &[8, 9, 10, 11]);
        assert_eq!(right_view.indptr, &[0, 4]);
        assert_eq!(right_view.values, &[256, 512, 1024, 2048]);
    }

    #[test]
    fn test_some_empty_rows_at_the_end_split_csr() {
        // first, let's build a sparse CSR array that is backed by an [u8]
        // (in real usage, this lives in the SHM or is read from the socket
        // into some buffer)

        const NNZ: u32 = 12;
        const NROWS: u32 = 7;
        const SIZES: CSRSizes = CSRSizes::new::<u16, u16, u16>(NNZ, NROWS);
        let mut buf: [u8; SIZES.total()] = [0; SIZES.total()];
        println!("SIZES: {SIZES:?}");

        let layout = ChunkCSRLayout {
            nframes: NROWS,
            nnz: NNZ,
            data_length_bytes: SIZES.total(),
            indptr_dtype: DType::U16,
            indptr_offset: 0,
            indptr_size: SIZES.indptr,
            indices_dtype: DType::U16,
            indices_offset: SIZES.indptr,
            indices_size: SIZES.indices,
            value_dtype: DType::U16,
            value_offset: SIZES.indptr + SIZES.indices,
            value_size: SIZES.values,
        };
        layout.validate();

        let view_mut: CSRViewMut<u16, u16, u16> = CSRViewMut::from_bytes(&mut buf, &SIZES);

        // generate some predictable pattern:
        let values: Vec<u16> = (0..12).map(|i| (1 << (i % 16))).collect();
        println!("values: {values:?}");
        assert_eq!(values.len(), 12);
        view_mut.values.copy_from_slice(&values);

        // put the values somewhere in the rows; doesn't really matter where:
        let indices: Vec<u16> = (0..12).collect();
        println!("indices: {indices:?}");
        assert_eq!(indices.len(), 12);
        view_mut.indices.copy_from_slice(&indices);

        // indices to where the rows start and end:
        // we have 7 rows, so we should have 8 entries here:
        let indptr: Vec<u16> = vec![0, 4, 8, 12, 12, 12, 12, 12];
        println!("indptr: {indptr:?}");
        assert_eq!(indptr.len() as u32, NROWS + 1);
        view_mut.indptr.copy_from_slice(&indptr);

        // now, we want to split into two parts; let's have two row in the first
        // part and the rest in the second part (in this case, one row with values
        // and 4 empty ones):

        const SIZES_LEFT: CSRSizes = CSRSizes::new::<u16, u16, u16>(8, 2);
        let mut left_buf: [u8; SIZES_LEFT.total()] = [0; SIZES_LEFT.total()];
        println!("SIZES_LEFT: {SIZES_LEFT:?}");

        const SIZES_RIGHT: CSRSizes = CSRSizes::new::<u16, u16, u16>(4, 5);
        let mut right_buf: [u8; SIZES_RIGHT.total()] = [0; SIZES_RIGHT.total()];
        println!("SIZES_RIGHT: {SIZES_RIGHT:?}");

        // the actual split happens here:
        let csr = CSRSplitter::from_bytes(&buf, layout);
        csr.split_into(2, &mut left_buf, &mut right_buf);

        // now, view the results:
        let left_view: CSRView<u16, u16, u16> = CSRView::from_bytes(&left_buf, &SIZES_LEFT);
        assert_eq!(left_view.nnz, left_view.values.len() as u32);
        assert_eq!(left_view.nnz, left_view.indices.len() as u32);
        assert_eq!(left_view.nrows + 1, left_view.indptr.len() as u32);

        assert_eq!(left_view.indices, &[0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(left_view.indptr, &[0, 4, 8]);
        assert_eq!(left_view.values, &[1, 2, 4, 8, 16, 32, 64, 128]);

        let right_view: CSRView<u16, u16, u16> = CSRView::from_bytes(&right_buf, &SIZES_RIGHT);
        assert_eq!(right_view.nnz, right_view.values.len() as u32);
        assert_eq!(right_view.nnz, right_view.indices.len() as u32);
        assert_eq!(right_view.nrows + 1, right_view.indptr.len() as u32);

        assert_eq!(right_view.indices, &[8, 9, 10, 11]);
        assert_eq!(right_view.indptr, &[0, 4, 4, 4, 4, 4]);
        assert_eq!(right_view.values, &[256, 512, 1024, 2048]);
    }
}
