use std::mem::size_of;

use zerocopy::{AsBytes, FromBytes, LayoutVerified};

use crate::headers::DType;

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
        // FIXME: type dispatch here
        self.split_generic::<u16, u16, u16>(mid, left, right);
    }

    /// splits at `mid` and copies results into `left`/`right`.
    /// `left` will contain values for rows with indices [0..mid), `right` will contain [mid..len).
    fn split_generic<I, IP, V>(&self, mid: usize, left: &mut [u8], right: &mut [u8])
    where
        I: numpy::Element + FromBytes + AsBytes + Copy,
        IP: numpy::Element + FromBytes + AsBytes + Copy + std::ops::Sub<Output = IP>,
        V: numpy::Element + FromBytes + AsBytes + Copy,
        u32: From<IP>,
    {
        let view: SparseCSRView<I, IP, V> =
            SparseCSRView::from_bytes(self.raw_data, self.nnz, self.nrows);

        // to find nnz values, we need to look up in the `indptr` array
        // where the values/coords for the left part stop:
        let left_nnz = view.indptr[mid].try_into().unwrap();
        let left_rows: u32 = mid.try_into().unwrap();
        let left: SparseCSRViewMut<I, IP, V> =
            SparseCSRViewMut::from_bytes(left, left_nnz, left_rows);
        left.indices
            .copy_from_slice(&view.indices[..left_nnz as usize]);
        left.values
            .copy_from_slice(&view.values[..left_nnz as usize]);
        left.indptr
            .copy_from_slice(&view.indptr[..left_rows as usize + 1]);

        let right_nnz = self.nnz - left_nnz;
        let right_rows: u32 = self.nrows - left_rows;
        let right: SparseCSRViewMut<I, IP, V> =
            SparseCSRViewMut::from_bytes(right, right_nnz, right_rows);
        right
            .indices
            .copy_from_slice(&view.indices[left_nnz as usize..]);
        right
            .values
            .copy_from_slice(&view.values[left_nnz as usize..]);
        right
            .indptr
            .copy_from_slice(&view.indptr[left_rows as usize..]);

        // offset correction:
        let first = right.indptr[0];
        right.indptr.iter_mut().for_each(|ptr| *ptr = *ptr - first);
    }
}

///
pub struct SparseCSRView<'a, I, IP, V> {
    pub indices: &'a [I],
    pub indptr: &'a [IP],
    pub values: &'a [V],

    pub nnz: u32,
    pub nrows: u32,
}

impl<'a, I, V, IP> SparseCSRView<'a, I, IP, V>
where
    I: numpy::Element + FromBytes + AsBytes,
    IP: numpy::Element + FromBytes + AsBytes,
    V: numpy::Element + FromBytes + AsBytes,
{
    pub fn from_bytes(raw_data: &'a [u8], nnz: u32, nrows: u32) -> Self {
        let sizes = CSRSizes::new::<I, IP, V>(nnz, nrows);

        let indptr_raw = &raw_data[0..sizes.indptr];
        let indices_raw = &raw_data[sizes.indptr..sizes.indptr + sizes.indices];
        let values_raw =
            &raw_data[sizes.indptr + sizes.indices..sizes.indptr + sizes.indices + sizes.values];

        // FIXME: error handling
        // FIXME: all of these need to be properly aligned according to
        // their data types - in practice that means there should be
        // padding between these slices and we need to properly adjust the sizing calculation
        // according to these alignment requirements
        let indptr: &[IP] = LayoutVerified::new_slice(indptr_raw).unwrap().into_slice();
        let indices: &[I] = LayoutVerified::new_slice(indices_raw).unwrap().into_slice();
        let values: &[V] = LayoutVerified::new_slice(values_raw).unwrap().into_slice();

        Self {
            indices,
            indptr,
            values,
            nnz,
            nrows,
        }
    }

    /// check if the CSR matrix is valid
    pub fn assert_valid(&self) {
        // 1) All `indptr` values should be in bounds of `indices`, if they are part
        //    of a row that contains more than zero values.
        //    Especially at the end, there can be repetitions of empty ranges, which
        //    can point (start and end!) directly after the end of `indices`.
        //
        // 2) `values` and `indices` should have the same length. That is true
        //    by construction, as both are built according to `nnz`
        //
        // 3) `indptr` should have an entry for each row.
    }
}

pub struct SparseCSRViewMut<'a, I, IP, V> {
    pub indices: &'a mut [I],
    pub indptr: &'a mut [IP],
    pub values: &'a mut [V],

    pub nnz: u32,
    pub nrows: u32,
}

impl<'a, I, V, IP> SparseCSRViewMut<'a, I, IP, V>
where
    I: numpy::Element + FromBytes + AsBytes,
    IP: numpy::Element + FromBytes + AsBytes,
    V: numpy::Element + FromBytes + AsBytes,
{
    pub fn from_bytes(raw_data: &'a mut [u8], nnz: u32, nrows: u32) -> Self {
        let sizes = CSRSizes::new::<I, IP, V>(nnz, nrows);

        let (indptr_raw, rest) = raw_data.split_at_mut(sizes.indptr);
        let (indices_raw, values_raw) = rest.split_at_mut(sizes.indices);
        assert_eq!(sizes.values, values_raw.len());

        // FIXME: error handling, alignment
        let indptr: &mut [IP] = LayoutVerified::new_slice(indptr_raw)
            .unwrap()
            .into_mut_slice();
        let indices: &mut [I] = LayoutVerified::new_slice(indices_raw)
            .unwrap()
            .into_mut_slice();
        let values: &mut [V] = LayoutVerified::new_slice(values_raw)
            .unwrap()
            .into_mut_slice();

        Self {
            indices,
            indptr,
            values,
            nnz,
            nrows,
        }
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

    pub const fn total(&self) -> usize {
        self.indptr + self.indices + self.values
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        headers::DType,
        sparse_csr::{CSRSizes, SparseCSR, SparseCSRView},
    };

    use super::SparseCSRViewMut;

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

        let view_mut: SparseCSRViewMut<u16, u16, u16> =
            SparseCSRViewMut::from_bytes(&mut buf, NNZ, NROWS);

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
        let left_view: SparseCSRView<u16, u16, u16> = SparseCSRView::from_bytes(&left_buf, 8, 2);
        assert_eq!(left_view.indices, &[0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(left_view.indptr, &[0, 4, 8]);
        assert_eq!(left_view.values, &[1, 2, 4, 8, 16, 32, 64, 128]);

        let right_view: SparseCSRView<u16, u16, u16> = SparseCSRView::from_bytes(&right_buf, 4, 1);

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

        let view_mut: SparseCSRViewMut<u16, u16, u16> =
            SparseCSRViewMut::from_bytes(&mut buf, NNZ, NROWS);

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
        let left_view: SparseCSRView<u16, u16, u16> = SparseCSRView::from_bytes(&left_buf, 8, 2);
        assert_eq!(left_view.nnz, left_view.values.len() as u32);
        assert_eq!(left_view.nnz, left_view.indices.len() as u32);
        assert_eq!(left_view.nrows + 1, left_view.indptr.len() as u32);

        assert_eq!(left_view.indices, &[0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(left_view.indptr, &[0, 4, 8]);
        assert_eq!(left_view.values, &[1, 2, 4, 8, 16, 32, 64, 128]);

        let right_view: SparseCSRView<u16, u16, u16> = SparseCSRView::from_bytes(&right_buf, 4, 5);
        assert_eq!(right_view.nnz, right_view.values.len() as u32);
        assert_eq!(right_view.nnz, right_view.indices.len() as u32);
        assert_eq!(right_view.nrows + 1, right_view.indptr.len() as u32);

        assert_eq!(right_view.indices, &[8, 9, 10, 11]);
        assert_eq!(right_view.indptr, &[0, 4, 4, 4, 4, 4]);
        assert_eq!(right_view.values, &[256, 512, 1024, 2048]);
    }
}
