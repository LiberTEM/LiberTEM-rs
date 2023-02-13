use std::mem::size_of;

use zerocopy::{AsBytes, FromBytes};

use crate::{
    csr_view_raw::{CSRViewRaw, CSRViewRawMut},
    headers::DType,
};

///
/// Interprete a slice of bytes as a CSR array.
///
/// It's assumed that the parts are in the following order:
///
/// - rowind / indptr
/// - coords / indices
/// - values / data
///
/// Currently, there is also assumed to not be any padding between the parts.
pub struct SparseCSR<'a> {
    raw_data: &'a [u8],

    indices_dtype: DType,
    indptr_dtype: DType,
    values_dtype: DType,

    nnz: u32,
    nrows: u32,
}

impl<'a> SparseCSR<'a> {
    pub fn from_bytes(
        raw_data: &'a [u8],
        indices_dtype: DType,
        indptr_dtype: DType,
        values_dtype: DType,
        nnz: u32,
        nrows: u32,
    ) -> Self {
        Self {
            raw_data,
            indices_dtype,
            indptr_dtype,
            values_dtype,
            nnz,
            nrows,
        }
    }

    // FIXME: Result error type
    pub fn split_into(&self, mid: usize, left: &mut [u8], right: &mut [u8]) {
        match self.indptr_dtype {
            DType::U8 => self.split_generic::<u8>(mid, left, right),
            DType::U16 => self.split_generic::<u16>(mid, left, right),
            DType::U32 => self.split_generic::<u32>(mid, left, right),
            DType::U64 | DType::U1 | DType::U4 => {
                panic!("indptr type {:?} not supported (yet?)", self.indptr_dtype)
            }
        }
    }

    /// splits at `mid` and copies results into `left`/`right`.
    /// `left` will contain values for rows with indices [0..mid), `right` will contain [mid..len).
    ///
    /// This is generic over the index pointer type `IP`, as that is the only one we actually
    /// need to interprete in our code - the indices and values we can copy over unseen.
    fn split_generic<IP>(&self, mid: usize, left: &mut [u8], right: &mut [u8])
    where
        IP: numpy::Element + FromBytes + AsBytes + Copy + std::ops::Sub<Output = IP>,
        u32: From<IP>,
    {
        let view: CSRViewRaw = CSRViewRaw::from_bytes(
            self.raw_data,
            self.nnz,
            self.nrows,
            self.indptr_dtype,
            self.indices_dtype,
            self.values_dtype,
        );

        // to find nnz values, we need to look up in the `indptr` array
        // where the values/coords for the left part stop:
        let left_nnz = view.get_indptr::<IP>()[mid].try_into().unwrap();
        let left_rows: u32 = mid.try_into().unwrap();
        let mut left: CSRViewRawMut = CSRViewRawMut::from_bytes(
            left,
            left_nnz,
            left_rows,
            self.indptr_dtype,
            self.indices_dtype,
            self.values_dtype,
        );
        left.copy_into_indices_raw(
            &view.get_indices_raw()[..left_nnz as usize * self.indices_dtype.size()]
        );
        left.copy_into_values_raw(
            &view.get_values_raw()[..left_nnz as usize * self.values_dtype.size()],
        );
        left.copy_into_indptr(
            &view.get_indptr::<IP>()[..left_rows as usize + 1]
        );

        // This takes the symmetric parts of the slices above
        let right_nnz = self.nnz - left_nnz;
        let right_rows: u32 = self.nrows - left_rows;
        let mut right: CSRViewRawMut = CSRViewRawMut::from_bytes(
            right,
            right_nnz,
            right_rows,
            self.indptr_dtype,
            self.indices_dtype,
            self.values_dtype,
        );
        right.copy_into_indices_raw(
            &view.get_indices_raw()[right_nnz as usize * self.indices_dtype.size()..],
        );
        right.copy_into_values_raw(
            &view.get_values_raw()[right_nnz as usize * self.values_dtype.size()..],
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
    }
}

/// Calculate the size in bytes for the array parts and in total.
#[derive(Debug)]
pub struct CSRSizes {
    pub indptr: usize,
    pub indices: usize,
    pub values: usize,
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
        }
    }

    pub const fn total(&self) -> usize {
        self.indptr + self.indices + self.values
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        csr_view::{CSRView, CSRViewMut},
        headers::DType,
        sparse_csr::{CSRSizes, SparseCSR},
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

        let view_mut: CSRViewMut<u16, u16, u16> = CSRViewMut::from_bytes(&mut buf, NNZ, NROWS);

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
        let csr = SparseCSR::from_bytes(&buf, DType::U16, DType::U16, DType::U16, 12, 3);
        csr.split_into(2, &mut left_buf, &mut right_buf);

        // now, view the results:
        let left_view: CSRView<u16, u16, u16> = CSRView::from_bytes(&left_buf, 8, 2);
        assert_eq!(left_view.indices, &[0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(left_view.indptr, &[0, 4, 8]);
        assert_eq!(left_view.values, &[1, 2, 4, 8, 16, 32, 64, 128]);

        let right_view: CSRView<u16, u16, u16> = CSRView::from_bytes(&right_buf, 4, 1);

        assert_eq!(right_view.indices, &[8, 9, 10, 11]);
        assert_eq!(right_view.indptr, &[0, 4]);
        assert_eq!(right_view.values, &[256, 512, 1024, 2048]);
    }

    #[test]
    fn test_split_csr_some_empty_rows_at_the_end() {
        // first, let's build a sparse CSR array that is backed by an [u8]
        // (in real usage, this lives in the SHM or is read from the socket
        // into some buffer)

        const NNZ: u32 = 12;
        const NROWS: u32 = 7;
        const SIZES: CSRSizes = CSRSizes::new::<u16, u16, u16>(NNZ, NROWS);
        let mut buf: [u8; SIZES.total()] = [0; SIZES.total()];
        println!("SIZES: {SIZES:?}");

        let view_mut: CSRViewMut<u16, u16, u16> = CSRViewMut::from_bytes(&mut buf, NNZ, NROWS);

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
        let csr = SparseCSR::from_bytes(&buf, DType::U16, DType::U16, DType::U16, 12, 7);
        csr.split_into(2, &mut left_buf, &mut right_buf);

        // now, view the results:
        let left_view: CSRView<u16, u16, u16> = CSRView::from_bytes(&left_buf, 8, 2);
        assert_eq!(left_view.nnz, left_view.values.len() as u32);
        assert_eq!(left_view.nnz, left_view.indices.len() as u32);
        assert_eq!(left_view.nrows + 1, left_view.indptr.len() as u32);

        assert_eq!(left_view.indices, &[0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(left_view.indptr, &[0, 4, 8]);
        assert_eq!(left_view.values, &[1, 2, 4, 8, 16, 32, 64, 128]);

        let right_view: CSRView<u16, u16, u16> = CSRView::from_bytes(&right_buf, 4, 5);
        assert_eq!(right_view.nnz, right_view.values.len() as u32);
        assert_eq!(right_view.nnz, right_view.indices.len() as u32);
        assert_eq!(right_view.nrows + 1, right_view.indptr.len() as u32);

        assert_eq!(right_view.indices, &[8, 9, 10, 11]);
        assert_eq!(right_view.indptr, &[0, 4, 4, 4, 4, 4]);
        assert_eq!(right_view.values, &[256, 512, 1024, 2048]);
    }
}
