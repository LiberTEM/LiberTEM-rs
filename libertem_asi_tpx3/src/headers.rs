use bincode::{Options, ErrorKind};
use log::trace;
use pyo3::pyclass;
use serde::{Serialize, Deserialize};

#[derive(PartialEq, Eq, Clone, Debug)]
pub enum HeaderTypes {
    AcquisitionStart { header: AcquisitionStart },
    ScanStart { header: ScanStart },
    ArrayChunk { header: ArrayChunk },
    ScanEnd { header: ScanEnd },
    AcquisitionEnd { header: AcquisitionEnd },
}

#[derive(Debug)]
pub enum WireFormatError {
    UnknownHeader { id: u8 },
    UnknownVersion,
    SerdeError { err: Box<ErrorKind> },
}

impl From<Box<ErrorKind>> for WireFormatError {
    fn from(err: Box<ErrorKind>) -> Self {
        Self::SerdeError { err }
    }
}

impl HeaderTypes {
    pub fn from_bytes(bytes: &[u8; 32]) -> Result<Self, WireFormatError> {
        let options = bincode::DefaultOptions::new().with_fixint_encoding();

        Ok(match bytes[0] {
            0x00 => {
                let header: AcquisitionStart = options.deserialize_from(bytes.as_slice())?;
                HeaderTypes::AcquisitionStart { header }
            }
            0x01 => {
                let header: ScanStart = options.deserialize_from(bytes.as_slice())?;
                HeaderTypes::ScanStart { header }
            }
            0x02 => {
                let mut header: ArrayChunk = options.deserialize_from(bytes.as_slice())?;
                // // XXX: these values are wrong in our test data!
                // header.value_dtype = DType::U32;
                // header.nframes = 512*512;

                // let mut header_bytes: [u8; 32] = [0; 32];

                // options.serialize_into(&mut header_bytes[..], &header).unwrap();

                // trace!("fixed-up header: {:02x?}", &header_bytes[..]);

                HeaderTypes::ArrayChunk { header }
            }
            0x03 => {
                let header: ScanEnd = options.deserialize_from(bytes.as_slice())?;
                HeaderTypes::ScanEnd { header }
            }
            0x04 => {
                let header: AcquisitionEnd = options.deserialize_from(bytes.as_slice())?;
                HeaderTypes::AcquisitionEnd { header }
            }
            _ => {
                return Err(WireFormatError::UnknownHeader { id: bytes[0] });
            }
        })
    }
}

#[derive(PartialEq, Eq, Clone, Debug, Copy)]
#[repr(u8)]
pub enum DType {
    U1,
    U4,
    U8,
    U16,
    U32,
    U64,
}

impl DType {
    pub fn from_u8(value: u8) -> DType {
        match value {
            0 => DType::U1,
            1 => DType::U4,
            2 => DType::U8,
            3 => DType::U16,
            4 => DType::U32,
            5 => DType::U64,
            _ => panic!("Unknown value: {value}"),
        }
    }

    /// element size in bytes for the uncompressed formats
    pub const fn size(&self) -> usize {
        match self {
            DType::U8 => 1,
            DType::U16 => 2,
            DType::U32 => 4,
            DType::U64 => 8,
            DType::U1 => todo!(),
            DType::U4 => todo!(),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
#[repr(u8)]
pub enum FormatType {
    /// Sorted CSR
    CSR,

    /// Sorted CSR with ToT as value
    ToT,
}

impl FormatType {
    pub fn from_u8(value: u8) -> FormatType {
        match value {
            0 => FormatType::CSR,
            1 => FormatType::ToT,
            _ => panic!("Unknown value: {value}"),
        }
    }
}


#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Clone)]
#[pyclass]
pub struct AcquisitionStart {
    tag: u8, // const. 0x00

    /// 
    pub version: u8,
    pub format_type: FormatType,

    /// scan box limit: ~24kx24k
    pub nav_shape: (u16, u16),
    pub indptr_dtype: DType,

    pub sig_shape: (u16, u16),
    pub indices_dtype: DType,

    /// identifier
    pub sequence: u32,
    reserved: [u8;15],

    // for future: maybe `scan_pattern: Vec<(u64, u64)>,`
    // -> separate message
}

impl AcquisitionStart {
    pub fn new(version: u8, format_type: FormatType, nav_shape: (u16, u16), indptr_dtype: DType, sig_shape: (u16, u16), indices_dtype: DType, sequence: u32) -> Self {
        AcquisitionStart {
            tag: 0,
            version,
            format_type,
            nav_shape,
            indptr_dtype,
            sig_shape,
            indices_dtype,
            sequence,
            reserved: [0;15]
        }
    }
}

/// Sent at the beginning of each scan
/// followed by `metadata_length` bytes of JSON encoded metadata
/// (`metadata_length` can be 0)
#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Clone)]
pub struct ScanStart {
    tag: u8, // const. 0x01

    /// sequence number for scans in this acquisition
    pub sequence: u32,
    pub metadata_length: u64,
    reserved: [u8;19],
}

impl ScanStart {
    pub fn new(sequence: u32, metadata_length: u64) -> Self {
        ScanStart {
            tag: 1,
            sequence,
            metadata_length,
            reserved: [0;19]
        }
    }
}

/// Line or "stack of frames" as sparse data
/// on the wire, this is followed by the the arrays of sparse data.
/// the data is encoded as little endian integers, according to the
/// dtypes specified in the acquisition header and the value dtype
/// for each chunk.
/// 
/// sizes in bytes:
/// size of the indptr part: size_of::<indptr_dtype> * (nframes + 1)
/// size of the indices part: size_of::<indices_dtype> * length
/// size of the values part: size_of::<value_dtype> * length 
#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Clone)]
pub struct ArrayChunk {
    tag: u8,  // const. 0x02

    /// data type of individual pixels
    pub value_dtype: DType,

    /// number of frames in this chunk
    pub nframes: u32,
    
    /// number of non-zero elements in the array
    pub length: u32,
    reserved: [u8;22],
}

impl ArrayChunk {
    pub fn new(value_dtype: DType, nframes: u32, length: u32) -> Self {
        ArrayChunk {
            tag: 2,
            value_dtype,
            nframes,
            length,
            reserved: [0;22]
        }
    }

    pub fn get_chunk_size_bytes(&self, acquisition_header: &AcquisitionStart) -> usize {
        let indptr_size = (self.nframes as usize + 1) * acquisition_header.indptr_dtype.size();
        let indices_size = self.length as usize * acquisition_header.indices_dtype.size();
        let values_size = self.length as usize * self.value_dtype.size();

        // TODO: alignment/padding?
        values_size + indices_size + indptr_size
    }
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Clone)]
pub struct ScanEnd {
    tag: u8,  // const. 0x03

    /// the same sequence number as in the matching `ScanStart`
    pub sequence: u32,
    reserved: [u8;27],
}


impl ScanEnd {
    pub fn new(sequence: u32) -> Self {
        ScanEnd {
            tag: 3,
            sequence,
            reserved: [0;27]
        }
    }
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Clone)]
pub struct AcquisitionEnd {
    tag: u8,  // const. 0x04

    /// the same sequence id as in the matching `AcquisitionStart`
    pub sequence: u32,
    reserved: [u8;27],
}

impl AcquisitionEnd {
    pub fn new(sequence: u32) -> Self {
        AcquisitionEnd {
            tag: 4,
            sequence,
            reserved: [0;27]
        }
    }
}