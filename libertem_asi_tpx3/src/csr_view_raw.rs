use zerocopy::{AsBytes, FromBytes, Ref};

use crate::{chunk_stack::ChunkCSRLayout, sparse_csr::CSRSizes};

/// A view into a CSR array stored in `raw_data`, only interpreting the `indptr` array
/// as concrete values, leaving `indices` and `values` as bytes.
///
pub struct CSRViewRaw<'a> {
    raw_data: &'a [u8],
    layout: ChunkCSRLayout,
}

impl<'a> CSRViewRaw<'a> {
    pub fn from_bytes_with_layout(raw_data: &'a [u8], layout: &ChunkCSRLayout) -> Self {
        Self {
            raw_data,
            layout: layout.clone(),
        }
    }

    fn get_sizes(&self) -> CSRSizes {
        CSRSizes::from_layout(&self.layout)
    }

    pub fn get_indptr<IP>(&self) -> &'a [IP]
    where
        IP: numpy::Element + FromBytes + AsBytes,
    {
        let sizes = self.get_sizes();
        let offset = self.layout.indptr_offset;
        let indptr_raw = &self.raw_data[offset..offset + sizes.indptr];
        Ref::new_slice(indptr_raw).unwrap().into_slice()
    }

    pub fn get_indptr_raw(&self) -> &'a [u8] {
        let sizes = self.get_sizes();
        let offset = self.layout.indptr_offset;
        &self.raw_data[offset..offset + sizes.indptr]
    }

    pub fn get_indices_raw(&self) -> &'a [u8] {
        let sizes = self.get_sizes();
        let offset = self.layout.indices_offset;
        &self.raw_data[offset..offset + sizes.indices]
    }

    pub fn get_values_raw(&self) -> &'a [u8] {
        let sizes = self.get_sizes();
        let offset = self.layout.value_offset;
        &self.raw_data[offset..offset + sizes.values]
    }
}

pub struct CSRViewRawMut<'a> {
    raw_data: &'a mut [u8],
    layout: ChunkCSRLayout,
}

impl<'a> CSRViewRawMut<'a> {
    pub fn from_bytes_with_layout(raw_data: &'a mut [u8], layout: &ChunkCSRLayout) -> Self {
        Self {
            raw_data,
            layout: layout.clone(),
        }
    }

    fn get_sizes(&self) -> CSRSizes {
        CSRSizes::from_layout(&self.layout)
    }

    pub fn get_indptr<IP>(&mut self) -> &mut [IP]
    where
        IP: numpy::Element + FromBytes + AsBytes,
    {
        let sizes = self.get_sizes();
        let offset = self.layout.indptr_offset;
        let indptr_raw = &mut self.raw_data[offset..offset + sizes.indptr];
        Ref::new_slice(indptr_raw).unwrap().into_mut_slice()
    }

    pub fn get_indptr_raw(&mut self) -> &mut [u8] {
        let sizes = self.get_sizes();
        let offset = self.layout.indptr_offset;
        &mut self.raw_data[offset..offset + sizes.indptr]
    }

    pub fn copy_into_indptr<IP>(&mut self, src: &[IP])
    where
        IP: numpy::Element + FromBytes + AsBytes + Copy,
    {
        self.get_indptr().copy_from_slice(src);
    }

    pub fn get_indices_raw(&mut self) -> &mut [u8] {
        let sizes = self.get_sizes();
        let offset = self.layout.indices_offset;
        &mut self.raw_data[offset..offset + sizes.indices]
    }

    pub fn copy_into_indices_raw(&mut self, src: &[u8]) {
        self.get_indices_raw().copy_from_slice(src);
    }

    pub fn get_values_raw(&mut self) -> &mut [u8] {
        let offset = self.layout.value_offset;
        let sizes = self.get_sizes();
        &mut self.raw_data[offset..offset + sizes.values]
    }

    pub fn copy_into_values_raw(&mut self, src: &[u8]) {
        self.get_values_raw().copy_from_slice(src);
    }
}
