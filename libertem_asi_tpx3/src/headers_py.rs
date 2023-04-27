use log::debug;
use pyo3::{pyfunction, pymethods};

use crate::{
    chunk_stack::ChunkCSRLayout,
    csr_view::CSRViewMut,
    headers::{
        AcquisitionEnd, AcquisitionStart, ArrayChunk, DType, FormatType, HeaderTypes, ScanEnd,
        ScanStart,
    },
    sparse_csr::CSRSizes,
};

#[pymethods]
impl AcquisitionStart {
    #[new]
    fn new_py(
        version: u8,
        format_type: FormatType,
        nav_shape: (u16, u16),
        indptr_dtype: DType,
        sig_shape: (u16, u16),
        indices_dtype: DType,
        sequence: u32,
    ) -> Self {
        AcquisitionStart::new(
            version,
            format_type,
            nav_shape,
            indptr_dtype,
            sig_shape,
            indices_dtype,
            sequence,
        )
    }

    pub fn __repr__(&self) -> String {
        format!("{self:?}")
    }

    pub fn get_nav_shape(&self) -> (u16, u16) {
        self.nav_shape
    }

    pub fn get_sig_shape(&self) -> (u16, u16) {
        self.sig_shape
    }

    pub fn get_indptr_dtype(&self) -> String {
        self.indptr_dtype.to_str().to_string()
    }

    pub fn get_indices_dtype(&self) -> String {
        self.indices_dtype.to_str().to_string()
    }
}

#[pymethods]
impl DType {
    pub fn __str__(&self) -> String {
        self.to_str().to_string()
    }
}

#[pymethods]
impl ScanStart {
    #[new]
    fn new_py(sequence: u32, metadata_length: u64) -> Self {
        ScanStart::new(sequence, metadata_length)
    }
}

#[pymethods]
impl ArrayChunk {
    #[new]
    fn new_py(
        value_dtype: DType,
        nframes: u32,
        length: u32,
        indices_offset: u32,
        values_offset: u32,
    ) -> Self {
        ArrayChunk::new(value_dtype, nframes, length, indices_offset, values_offset)
    }
}

#[pymethods]
impl ScanEnd {
    #[new]
    fn new_py(sequence: u32) -> Self {
        ScanEnd::new(sequence)
    }
}

#[pymethods]
impl AcquisitionEnd {
    #[new]
    fn new_py(sequence: u32) -> Self {
        AcquisitionEnd::new(sequence)
    }
}

#[pyfunction]
pub fn make_sim_data(
    nav_shape: (u16, u16),
    indptr: Vec<u32>,
    indices: Vec<u32>,
    values: Vec<u32>,
) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::new();
    let nframes = nav_shape.0 as u32 * nav_shape.1 as u32;
    let acquisition_start_header = HeaderTypes::AcquisitionStart {
        header: AcquisitionStart::new(
            7,
            FormatType::CSR,
            nav_shape,
            DType::U32,
            (512, 512),
            DType::U32,
            1,
        ),
    };
    debug!("{acquisition_start_header:?}");
    out.extend_from_slice(&acquisition_start_header.to_bytes());

    let scan_start_header = HeaderTypes::ScanStart {
        header: ScanStart::new(1, 0),
    };
    debug!("{scan_start_header:?}");
    out.extend_from_slice(&scan_start_header.to_bytes());

    let nnz = values.len() as u32;
    let sizes: CSRSizes = CSRSizes::new::<u32, u32, u32>(nnz, nframes);
    // FIXME: correct padding?
    let layout = ChunkCSRLayout {
        nframes,
        nnz,
        data_length_bytes: sizes.total(),
        indptr_dtype: DType::U32,
        indptr_offset: 0,
        indptr_size: sizes.indptr,
        indices_dtype: DType::U32,
        indices_offset: sizes.indptr,
        indices_size: sizes.indices,
        value_dtype: DType::U32,
        value_offset: sizes.indptr + sizes.indices,
        value_size: sizes.values,
    };
    layout.validate();

    debug!("layout: {:?}", layout);
    debug!("sizes: {:?}", sizes);

    let mut chunk: Vec<u8> = (0..sizes.total()).map(|_| 0).collect();
    // debug!("total size: {} {}", sizes.total(), chunk.);
    let mut view_mut: CSRViewMut<u32, u32, u32> =
        CSRViewMut::from_bytes_with_layout(&mut chunk, &layout);
    view_mut.copy_from_slices(&indptr, &indices, &values);

    let array_chunk_header = HeaderTypes::ArrayChunk {
        header: ArrayChunk::new(
            DType::U32,
            nframes,
            nnz,
            layout.indices_offset as u32,
            layout.value_offset as u32,
        ),
    };
    debug!("{array_chunk_header:?}");
    out.extend_from_slice(&array_chunk_header.to_bytes());
    out.extend_from_slice(&chunk);

    let scan_end_header = HeaderTypes::ScanEnd {
        header: ScanEnd::new(1),
    };
    out.extend_from_slice(&scan_end_header.to_bytes());

    let acquisition_end_header = HeaderTypes::AcquisitionEnd {
        header: AcquisitionEnd::new(1),
    };
    out.extend_from_slice(&acquisition_end_header.to_bytes());

    out
}
