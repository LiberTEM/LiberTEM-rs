use zerocopy::{FromBytes, AsBytes, LayoutVerified};

use crate::sparse_csr::CSRSizes;

///
pub struct CSRView<'a, I, IP, V> {
    pub indices: &'a [I],
    pub indptr: &'a [IP],
    pub values: &'a [V],

    pub nnz: u32,
    pub nrows: u32,
}

impl<'a, I, V, IP> CSRView<'a, I, IP, V>
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

pub struct CSRViewMut<'a, I, IP, V> {
    pub indices: &'a mut [I],
    pub indptr: &'a mut [IP],
    pub values: &'a mut [V],

    pub nnz: u32,
    pub nrows: u32,
}

impl<'a, I, V, IP> CSRViewMut<'a, I, IP, V>
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
