use zerocopy::{AsBytes, FromBytes, LayoutVerified};

use crate::{headers::DType, sparse_csr::CSRSizes};

/// A view into a CSR array stored in `raw_data`, only interpreting the `indptr` array
/// as concrete values, leaving `indices` and `values` as bytes.
pub struct CSRViewRaw<'a> {
    raw_data: &'a [u8],
    pub nnz: u32,
    pub nrows: u32,
    indptr_dtype: DType,
    indices_dtype: DType,
    values_dtype: DType,
}

impl<'a> CSRViewRaw<'a> {
    pub fn from_bytes(
        raw_data: &'a [u8],
        nnz: u32,
        nrows: u32,
        indptr_dtype: DType,
        indices_dtype: DType,
        values_dtype: DType,
    ) -> Self {
        Self {
            raw_data,
            nnz,
            nrows,
            indices_dtype,
            values_dtype,
            indptr_dtype,
        }
    }

    fn get_sizes(&self) -> CSRSizes {
        CSRSizes::new_dyn(
            self.nnz,
            self.nrows,
            self.indptr_dtype,
            self.indices_dtype,
            self.values_dtype,
        )
    }

    pub fn get_indptr<IP>(&self) -> &'a [IP]
    where
        IP: numpy::Element + FromBytes + AsBytes,
    {
        let sizes = self.get_sizes();
        let indptr_raw = &self.raw_data[0..sizes.indptr];
        LayoutVerified::new_slice(indptr_raw).unwrap().into_slice()
    }

    pub fn get_indices_raw(&self) -> &'a [u8] {
        let sizes = self.get_sizes();
        &self.raw_data[sizes.indptr..sizes.indptr + sizes.indices]
    }

    pub fn get_values_raw(&self) -> &'a [u8] {
        let sizes = self.get_sizes();
        &self.raw_data[sizes.indptr + sizes.indices..sizes.indptr + sizes.indices + sizes.values]
    }
}

pub struct CSRViewRawMut<'a> {
    raw_data: &'a mut [u8],
    pub nnz: u32,
    pub nrows: u32,
    indptr_dtype: DType,
    indices_dtype: DType,
    values_dtype: DType,
}

impl<'a> CSRViewRawMut<'a> {
    pub fn from_bytes(
        raw_data: &'a mut [u8],
        nnz: u32,
        nrows: u32,
        indptr_dtype: DType,
        indices_dtype: DType,
        values_dtype: DType,
    ) -> Self {
        Self {
            raw_data,
            nnz,
            nrows,
            indices_dtype,
            values_dtype,
            indptr_dtype,
        }
    }

    fn get_sizes(&self) -> CSRSizes {
        CSRSizes::new_dyn(
            self.nnz,
            self.nrows,
            self.indptr_dtype,
            self.indices_dtype,
            self.values_dtype,
        )
    }

    pub fn get_indptr<IP>(&mut self) -> &mut [IP]
    where
        IP: numpy::Element + FromBytes + AsBytes,
    {
        let sizes = self.get_sizes();
        let indptr_raw = &mut self.raw_data[0..sizes.indptr];
        LayoutVerified::new_slice(indptr_raw)
            .unwrap()
            .into_mut_slice()
    }

    pub fn copy_into_indptr<IP>(&mut self, src: &[IP])
    where
        IP: numpy::Element + FromBytes + AsBytes + Copy,
    {
        self.get_indptr().copy_from_slice(src);
    }

    pub fn get_indices_raw(&mut self) -> &mut [u8] {
        let sizes = self.get_sizes();
        &mut self.raw_data[sizes.indptr..sizes.indptr + sizes.indices]
    }

    pub fn copy_into_indices_raw(&mut self, src: &[u8]) {
        self.get_indices_raw().copy_from_slice(src);
    }

    pub fn get_values_raw(&mut self) -> &mut [u8] {
        let sizes = self.get_sizes();
        &mut self.raw_data
            [sizes.indptr + sizes.indices..sizes.indptr + sizes.indices + sizes.values]
    }

    pub fn copy_into_values_raw(&mut self, src: &[u8]) {
        self.get_values_raw().copy_from_slice(src);
    }
}
