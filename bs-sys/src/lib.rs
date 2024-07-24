#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

mod bs_bindings {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

#[derive(Debug, PartialEq, Eq)]
#[repr(i64)]
pub enum BitshuffleError {
    AllocationError = -1,
    MissingSSE = -11,
    MissingAVX = -12,
    InputSizeMult8 = -80,
    BlockSizeMult8 = -81,
    DecompressionError = -91,
    SizeMismatch,
    InternalError(i64),
    Other,
}

impl From<i64> for BitshuffleError {
    fn from(code: i64) -> Self {
        match code {
            -1 => Self::AllocationError,
            -11 => Self::MissingSSE,
            -12 => Self::MissingAVX,
            -80 => Self::InputSizeMult8,
            -81 => Self::BlockSizeMult8,
            -91 => Self::DecompressionError,
            code => Self::InternalError(code),
        }
    }
}

///
/// Bound on size of data of type `T` compressed with `compress_lz4`.
///
/// # Arguments
///
/// * `size_in_elems` - the number of elements in your input data
/// * `block_size` - the bitshuffle blocksize, default block size is chosen if `None`
///
pub fn compress_lz4_bound<T>(size_in_elems: usize, block_size: Option<usize>) -> usize {
    let block_size = block_size.unwrap_or(0);
    let elem_size = std::mem::size_of::<T>();
    // size : number of elements in input
    // elem_size : element size of typed data
    // block_size : Process in blocks of this many elements.
    unsafe { bs_bindings::bshuf_compress_lz4_bound(size_in_elems, elem_size, block_size) }
}

///
/// Bitshuffle and compress the data using LZ4.
///
/// # Arguments
///
/// * `in_` - the input array of type `T`
/// * `block_size` - the bitshuffle blocksize, default block size is chosen if `None`
///
pub fn compress_lz4<T>(in_: &[T], block_size: Option<usize>) -> Result<Vec<u8>, BitshuffleError> {
    let block_size = block_size.unwrap_or(0);
    let c_in = in_.as_ptr().cast();
    let elem_size = std::mem::size_of::<T>();
    let size_in_elems = in_.len();

    let max_out_size_bytes =
        unsafe { bs_bindings::bshuf_compress_lz4_bound(size_in_elems, elem_size, block_size) };
    let mut out: Vec<u8> = Vec::with_capacity(max_out_size_bytes);
    let bytes_used = unsafe {
        let c_out = out.as_mut_ptr().cast();

        // in : input buffer, must be of size * elem_size bytes
        // out : output buffer, must be large enough to hold data.
        // size : number of elements in input
        // elem_size : element size of typed data
        // block_size : Process in blocks of this many elements.
        bs_bindings::bshuf_compress_lz4(c_in, c_out, size_in_elems, elem_size, block_size)
    };
    if bytes_used < 0 {
        return Err(bytes_used.into());
    }
    // safety:
    // 1) count is less than or equal to capacity (should be given by construction, ...):
    assert!(usize::try_from(bytes_used).unwrap() <= out.capacity());
    // 2) bshuf_compress_lz4 promises that is has written `count` bytes in `out`
    unsafe { out.set_len(bytes_used as usize) }
    Ok(out)
}

///
/// Decompress data, then un-bitshuffle it and store it in a newly-allocated
/// `Vec<T>`. `out_size`, `block_size` and `T` have to match the parameters used
/// when compressing the data.
///
/// # Arguments
///
/// * `in_` - bitshuffled and compressed data as bytes
/// * `out_size` - number of elements we expect to get back
/// * `block_size` - the bitshuffle blocksize, default block size is chosen if `None`
///
pub fn decompress_lz4<T>(
    in_: &[u8],
    out_size: usize, // number of elements
    block_size: Option<usize>,
) -> Result<Vec<T>, BitshuffleError> {
    let block_size = block_size.unwrap_or(0);
    let mut out: Vec<T> = Vec::with_capacity(out_size);
    let out_ptr: *mut T = out.as_mut_ptr();
    unsafe {
        decompress_lz4_into(in_, out_ptr, out_size, Some(block_size))?;
        out.set_len(out_size)
    };
    Ok(out)
}

///
/// Decompress data, then un-bitshuffle it and store it in `out`.
/// `out_size`, `block_size` and `T` have to match the parameters used when
/// compressing the data.
///
/// # Arguments
///
/// * `in_` - bitshuffled and compressed data as bytes
/// * `out` - pointer to memory where the results should be stored
/// * `block_size` - the bitshuffle blocksize, default block size is chosen if `None`
///
/// # Safety
///
/// The memory pointed to by `out` must be large enough to fit the output, i.e.
/// at least `std::mem::size_of::<T> * out_size`.
pub unsafe fn decompress_lz4_into<T>(
    in_: &[u8],
    out: *mut T, // FIXME: replace with slice of MaybeUninit from Vec::spare_capacity_mut?
    out_size: usize, // number of elements
    block_size: Option<usize>,
) -> Result<(), BitshuffleError> {
    let block_size = block_size.unwrap_or(0);
    let in_ptr = in_.as_ptr().cast();
    let elem_size = std::mem::size_of::<T>();

    unsafe {
        let out_ptr = out.cast();
        let count =
            bs_bindings::bshuf_decompress_lz4(in_ptr, out_ptr, out_size, elem_size, block_size);
        if count < 0 {
            return Err(count.into());
        }
        if count != i64::try_from(in_.len()).unwrap() {
            return Err(BitshuffleError::SizeMismatch);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_bitshuffle_lz4_u8() {
        let input = "this is some data".as_bytes();
        eprintln!("{input:?}");

        let compressed = compress_lz4(input, None).unwrap();
        eprintln!("{compressed:?}");

        let decompressed: Vec<u8> = decompress_lz4(&compressed, input.len(), None).unwrap();
        eprintln!("{decompressed:?}");

        assert_eq!(input, decompressed);
    }

    #[test]
    fn roundtrip_bitshuffle_lz4_i32() {
        let input: Vec<i32> = vec![16, 43242, 12, -32123, 975, -78, 2, 0];
        eprintln!("{input:?}");

        let compressed = compress_lz4(&input, None).unwrap();
        eprintln!("{compressed:?}");

        let decompressed: Vec<i32> = decompress_lz4(&compressed, input.len(), None).unwrap();
        eprintln!("{decompressed:?}");

        assert_eq!(input, decompressed);
    }

    #[test]
    fn roundtrip_bitshuffle_decompress_too_short() {
        let input: Vec<i32> = vec![16, 43242, 12, -32123, 975, -78, 2, 0];
        eprintln!("{input:?}");

        let compressed = compress_lz4(&input, None).unwrap();
        eprintln!("{compressed:?}");

        let err: Result<_, _> = decompress_lz4::<Vec<i32>>(&compressed, input.len() - 1, None);
        assert_eq!(err, Err(BitshuffleError::SizeMismatch));

        let err: Result<_, _> = decompress_lz4::<Vec<i32>>(&compressed, 0, None);
        assert_eq!(err, Err(BitshuffleError::SizeMismatch));

        let err: Result<_, _> = decompress_lz4::<Vec<i32>>(&compressed, input.len() + 1, None);
        assert_eq!(err, Err(BitshuffleError::DecompressionError));

        let err: Result<_, _> = decompress_lz4::<Vec<i32>>(&compressed, input.len() + 1000, None);
        assert_eq!(err, Err(BitshuffleError::InternalError(-1001)));
    }

    #[test]
    fn roundtrip_bitshuffle_lz4_u64() {
        let input: Vec<u64> = vec![
            6605535394741060369,
            1943729968968667107,
            4434805337173085311,
            3857399390108534620,
            42,
            1399976486432680367,
            0,
        ];

        let compressed = compress_lz4(&input, None).unwrap();
        let decompressed: Vec<u64> = decompress_lz4(&compressed, input.len(), None).unwrap();
        assert_eq!(input, decompressed);
    }

    #[test]
    fn roundtrip_bitshuffle_lz4_i64() {
        let input: Vec<i64> = vec![
            6605535394741060369,
            1943729968968667107,
            -4434805337173085311,
            3857399390108534620,
            42,
            -1399976486432680367,
            0,
        ];
        eprintln!("{input:?}");

        let compressed = compress_lz4(&input, None).unwrap();
        eprintln!("{compressed:?}");

        let decompressed: Vec<i64> = decompress_lz4(&compressed, input.len(), None).unwrap();
        eprintln!("{decompressed:?}");

        assert_eq!(input, decompressed);
    }

    #[test]
    fn roundtrip_bitshuffle_lz4_f64() {
        let input: Vec<f64> = vec![
            6605535394741060369.0,
            1943729968968667107.0,
            -4434805337173085311.0,
            3857399390108534620.0,
            42.0,
            -1399976486432680367.0,
            -1.8645594,
            0.63130624,
            -0.74550843,
            -0.25914887,
            -0.43326423,
            -0.311869,
            -0.25334516,
            -0.31615132,
            -1.84247401,
            -0.55763137,
            0.74976962,
            0.17890207,
            0.48589075,
            0.11130236,
            0.47587833,
            0.0,
            -0.0,
        ];
        eprintln!("{input:?}");

        let compressed = compress_lz4(&input, None).unwrap();
        eprintln!("{compressed:?}");

        let decompressed: Vec<f64> = decompress_lz4(&compressed, input.len(), None).unwrap();
        eprintln!("{decompressed:?}");

        assert_eq!(input, decompressed);
    }
}
