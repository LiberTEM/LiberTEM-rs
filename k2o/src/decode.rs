//!
//! # K2IS encoding/decoding functionality
//!
//! This module contains functionality for encoding and decoding K2IS
//! files and network packets.
//!
//! # Binary file and
//!
//! # 12bit "little endian" encoding
//!
//! The encoding can be described by just looking at a single input pair (a, b)
//! of u16 values. Let's say our inputs are `0xABC` and `0xDEF`.
//! The encoder now outputs three bytes.
//! - the first output byte is made up of the two least significant nibbles of `a`, so `0xBC`
//! - the second output byte is a combination from the two input values:
//!         - the least significant nibble is the most significant nibble from `a`, so `0xA`
//!         - the most significant nibble is the least significant niddle from `b`, so `0xF`
//! - the third output byte is the two most significant nibbles from `b`, so `0xDE`
//!
//! So, to summarize:
//!
//! input:  ABC DEF
//! output: BC FA DE
use std::convert::TryInto;

use num::cast::AsPrimitive;

pub const HEADER_SIZE: usize = 40;

// These are no longer constant; they are now const generics (or derived from such):
// pub const PACKET_SIZE: usize = 0x5758;  // or 0xc028
// pub const PAYLOAD_SIZE: usize = PACKET_SIZE - HEADER_SIZE;
// pub const DECODED_SIZE: usize = 930 * 16; // number of u16 values -> can also be 32 * 768 for summit mode

///
/// Decodes 12bit "little endian" values into integers
///
/// # Arguments
/// * `bytes` - the bytes that should be decoded. Should be `PACKET_SIZE` long (`0x5758` or `0xc028`)
/// * `out` - the slice where the decoded integers should be written to. Should be `DECODED_SIZE` long (`14880`).
///
pub fn decode<const PACKET_SIZE: usize>(bytes: &[u8], out: &mut [u16]) -> () {
    // make sure input/output are bounded to PACKET_SIZE and DECODED_SIZE
    //const DECODED_SIZE: usize = (PACKET_SIZE - HEADER_SIZE) * 2 / 3;
    let input = &bytes[..PACKET_SIZE];
    let decoded = &mut out[..(PACKET_SIZE - HEADER_SIZE) * 2 / 3];

    assert_eq!(input.len(), PACKET_SIZE);
    assert_eq!(decoded.len(), (PACKET_SIZE - HEADER_SIZE) * 2 / 3);

    for i in 0..(PACKET_SIZE - HEADER_SIZE) / 3 {
        let fst = input[HEADER_SIZE + i * 3] as u16;
        let mid = input[HEADER_SIZE + i * 3 + 1] as u16;
        let lst = input[HEADER_SIZE + i * 3 + 2] as u16;

        let a = fst | (mid & 0x0F) << 8;
        let b = (mid & 0xF0) >> 4 | lst << 4;

        decoded[i * 2] = a;
        decoded[i * 2 + 1] = b;
    }
}

///
/// Decodes 12bit "little endian" values into generic type D, converted by function f: Fn(u16) -> D
///
/// # Arguments
/// * `bytes` - the bytes that should be decoded. Should be `PACKET_SIZE` long (`0x5758`)
/// * `out` - the slice where the decoded integers should be written to. Should be `DECODED_SIZE` long (`14880`).
///
pub fn decode_map<D, F, const PACKET_SIZE: usize, const DECODED_SIZE: usize>(
    bytes: &[u8],
    out: &mut [D],
    f: F,
) -> ()
where
    F: Fn(u16) -> D,
{
    // make sure input/output are bounded to PACKET_SIZE and DECODED_SIZE
    let input = &bytes[..PACKET_SIZE];
    let decoded = &mut out[..DECODED_SIZE];

    assert_eq!(input.len(), PACKET_SIZE);
    assert_eq!(decoded.len(), DECODED_SIZE);

    for i in 0..(PACKET_SIZE - HEADER_SIZE) / 3 {
        let fst = input[HEADER_SIZE + i * 3] as u16;
        let mid = input[HEADER_SIZE + i * 3 + 1] as u16;
        let lst = input[HEADER_SIZE + i * 3 + 2] as u16;

        let a = fst | (mid & 0x0F) << 8;
        let b = (mid & 0xF0) >> 4 | lst << 4;

        decoded[i * 2] = f(a);
        decoded[i * 2 + 1] = f(b);
    }
}

///
/// Decodes 12bit "little endian" values into generic type D, converted by function f: Fn(u16) -> D
///
/// # Arguments
/// * `bytes` - the bytes that should be decoded. Should be `PACKET_SIZE` long (`0x5758`)
/// * `out` - the slice where the decoded integers should be written to. Should be `DECODED_SIZE` long (`14880`).
///
pub fn decode_converted<D, const PACKET_SIZE: usize, const DECODED_SIZE: usize>(
    bytes: &[u8],
    out: &mut [D],
) -> ()
where
    u16: AsPrimitive<D>,
    D: Copy + 'static,
{
    // make sure input/output are bounded to PACKET_SIZE and DECODED_SIZE
    let input = &bytes[..PACKET_SIZE];
    let decoded = &mut out[..DECODED_SIZE];

    assert_eq!(input.len(), PACKET_SIZE);
    assert_eq!(decoded.len(), DECODED_SIZE);

    for i in 0..(PACKET_SIZE - HEADER_SIZE) / 3 {
        let fst = input[HEADER_SIZE + i * 3] as u16;
        let mid = input[HEADER_SIZE + i * 3 + 1] as u16;
        let lst = input[HEADER_SIZE + i * 3 + 2] as u16;

        let a = fst | (mid & 0x0F) << 8;
        let b = (mid & 0xF0) >> 4 | lst << 4;

        decoded[i * 2] = a.as_();
        decoded[i * 2 + 1] = b.as_();
    }
}

///
/// Decodes 12bit "little endian" values into integers. Alternative
/// implementation that in unrolled (does not seem to be good for performance in all cases!)
///
/// # Arguments
/// * `bytes` - the bytes that should be decoded. Should be `PACKET_SIZE` long (`0x5758`)
/// * `out` - the slice where the decoded integers should be written to. Should be `DECODED_SIZE` long (`14880`).
///
pub fn decode_unrolled<const PACKET_SIZE: usize, const DECODED_SIZE: usize>(
    bytes: &[u8],
    out: &mut [u16],
) -> () {
    let in_chunk_size = 3 * 3 * 16;
    let out_chunk_size = 3 * 2 * 16;

    assert_eq!((PACKET_SIZE - HEADER_SIZE) % in_chunk_size, 0);
    assert_eq!(DECODED_SIZE % out_chunk_size, 0);

    let input = &bytes[..PACKET_SIZE];
    let decoded = &mut out[..DECODED_SIZE];

    assert_eq!(decoded.len(), DECODED_SIZE);
    assert_eq!(input.len(), PACKET_SIZE);

    if true {
        assert_eq!(bytes.len() % PACKET_SIZE, 0);
        assert_eq!(&decoded.len() % DECODED_SIZE, 0);

        assert_eq!((bytes.len() - HEADER_SIZE) % in_chunk_size, 0);
        assert_eq!(&decoded.len() % out_chunk_size, 0);
    }

    let in_chunks_outer = input[HEADER_SIZE..].chunks_exact(in_chunk_size);
    let out_chunks_outer = decoded[..DECODED_SIZE].chunks_exact_mut(out_chunk_size);

    for (in_chunk_outer, out_chunk_outer) in in_chunks_outer.zip(out_chunks_outer) {
        let in_chunks = in_chunk_outer.chunks_exact(3);
        let out_chunks = out_chunk_outer.chunks_exact_mut(2);

        for (in_chunk, out_chunk) in in_chunks.zip(out_chunks) {
            let fst = in_chunk[0] as u16;
            let mid = in_chunk[1] as u16;
            let lst = in_chunk[2] as u16;

            out_chunk[0] = fst | (mid & 0x0F) << 8;
            out_chunk[1] = mid >> 4 | lst << 4;
        }
    }
}

pub fn decode_u32(bytes: &[u8]) -> u32 {
    return u32::from_be_bytes(bytes.try_into().unwrap());
}

pub fn decode_u16(bytes: &[u8]) -> u16 {
    return u16::from_be_bytes(bytes.try_into().unwrap());
}

pub fn decode_u16_vec<const PACKET_SIZE: usize>(bytes: &[u8], out: &mut Vec<u16>) {
    for i in 0..(PACKET_SIZE - HEADER_SIZE) / 2 {
        let in_bytes = &bytes[i * 2..i * 2 + 2];
        out[i] = decode_u16(&in_bytes);
    }
}

pub fn decode_packet_size(bytes: &[u8]) -> u32 {
    return decode_u32(&bytes[36..40]);
}

/// Encode the u16 values from `inp` into 12 bit "little endian" encoding.
///
/// # Arguments
///
/// * `inp` - The input integers
/// * `out` - A mutable byte slice where the encoded values will be written
///
pub fn encode(inp: &Vec<u16>, out: &mut [u8]) -> () {
    // pre-condition: out_chunks should have no remainder
    assert_eq!(out.len() % 3, 0);
    // pre-condition: output size should be 3/2 of input size
    assert_eq!(out.len(), inp.len() * 3 / 2);

    let in_chunks = inp.chunks_exact(2);
    // pre-condition: in_chunks should have no remainder
    assert_eq!(in_chunks.remainder().len(), 0);
    let out_chunks = out.chunks_exact_mut(3);

    for (in_chunk, out_chunk) in in_chunks.zip(out_chunks) {
        let a = in_chunk[0];
        let b = in_chunk[1];

        out_chunk[0] = (a & 0xFF) as u8;
        out_chunk[1] = ((a & 0xF00) >> 8) as u8 | ((b & 0xF) << 4) as u8;
        out_chunk[2] = ((b & 0xFF0) >> 4) as u8;
    }
}

#[cfg(test)]
mod tests {
    use crate::decode::decode;
    use crate::decode::decode_converted;
    use crate::decode::decode_map;
    use crate::decode::decode_unrolled;
    use crate::decode::encode;
    //use crate::decode::DECODED_SIZE;
    //use crate::decode::PACKET_SIZE;
    use crate::decode::HEADER_SIZE;

    const PACKET_SIZE: usize = 0x5758;
    const DECODED_SIZE: usize = (PACKET_SIZE - HEADER_SIZE) * 2 / 3;

    fn make_test_data() -> Vec<u16> {
        let mut original_values: Vec<u16> = Vec::new();
        original_values.extend((0u16..=0xFFF).cycle().take(DECODED_SIZE));
        return original_values;
    }

    #[test]
    fn encode_works() {
        let mut original_values: Vec<u16> = Vec::new();
        original_values.extend([0xABC, 0xDEF].iter().copied());
        let mut encoded: [u8; PACKET_SIZE] = [0; PACKET_SIZE];
        let mut encoded_used = &mut encoded[40..43];
        encode(&original_values, &mut encoded_used);

        println!("{:03X?}", &original_values);
        println!("{:02X?}", &encoded_used);

        let mut decoded: [u16; DECODED_SIZE] = [0; DECODED_SIZE];
        decode::<PACKET_SIZE>(&encoded, &mut decoded);

        println!("{:03X?}", &decoded[..2]);

        assert_eq!(&decoded[..2], &original_values[..2]);
    }

    #[test]
    fn roundtrip_decode() {
        let original_values = make_test_data();

        let mut encoded: [u8; PACKET_SIZE] = [0; PACKET_SIZE];
        let mut encoded_used = &mut encoded[40..];
        encode(&original_values, &mut encoded_used);

        println!("{:03X?}", &original_values[..16]); // arbitrary cut-off
        println!("{:02X?}", &encoded_used[..24]); // not-so-arbitrary cut-off

        let mut decoded: [u16; DECODED_SIZE] = [0; DECODED_SIZE];
        decode::<PACKET_SIZE>(&encoded, &mut decoded);

        println!("{:03X?}", &decoded[..16]); // arbitrary cut-off

        assert_eq!(&decoded.to_vec(), &original_values);
    }

    #[test]
    fn roundtrip_decode_unrolled() {
        let original_values = make_test_data();

        let mut encoded: [u8; PACKET_SIZE] = [0; PACKET_SIZE];
        let mut encoded_used = &mut encoded[40..];
        encode(&original_values, &mut encoded_used);

        println!("{:03X?}", &original_values);
        println!("{:02X?}", &encoded_used);

        let mut decoded: [u16; DECODED_SIZE] = [0; DECODED_SIZE];
        decode_unrolled::<PACKET_SIZE, DECODED_SIZE>(&encoded, &mut decoded);

        println!("{:03X?}", &decoded[..16]); // arbitrary cut-off

        assert_eq!(&decoded.to_vec(), &original_values);
    }

    #[test]
    fn roundtrip_decode_map() {
        let original_values = make_test_data();

        let mut encoded: [u8; PACKET_SIZE] = [0; PACKET_SIZE];
        let mut encoded_used = &mut encoded[40..];
        encode(&original_values, &mut encoded_used);

        println!("{:03X?}", &original_values);
        println!("{:02X?}", &encoded_used);

        let mut decoded: [u16; DECODED_SIZE] = [0; DECODED_SIZE];
        decode_map::<_, _, PACKET_SIZE, DECODED_SIZE>(&encoded, &mut decoded, |x| x);

        println!("{:03X?}", &decoded[..16]); // arbitrary cut-off

        assert_eq!(&decoded.to_vec(), &original_values);
    }

    #[test]
    fn roundtrip_decode_converted() {
        let original_values = make_test_data();

        let mut encoded: [u8; PACKET_SIZE] = [0; PACKET_SIZE];
        let mut encoded_used = &mut encoded[40..];
        encode(&original_values, &mut encoded_used);

        println!("{:03X?}", &original_values);
        println!("{:02X?}", &encoded_used);

        let mut decoded: [u16; DECODED_SIZE] = [0; DECODED_SIZE];
        decode_converted::<_, PACKET_SIZE, DECODED_SIZE>(&encoded, &mut decoded);

        println!("{:03X?}", &decoded[..16]); // arbitrary cut-off

        assert_eq!(&decoded.to_vec(), &original_values);
    }

    #[test]
    fn roundtrip_decode_converted_f32() {
        //! encode from test data to u12, from u12 to f32 and back to original u16
        let original_values = make_test_data();

        let mut encoded: [u8; PACKET_SIZE] = [0; PACKET_SIZE];
        let mut encoded_used = &mut encoded[40..];
        encode(&original_values, &mut encoded_used);

        println!("{:03X?}", &original_values);
        println!("{:02X?}", &encoded_used);

        let mut decoded: [f32; DECODED_SIZE] = [0.0; DECODED_SIZE];
        decode_converted::<_, PACKET_SIZE, DECODED_SIZE>(&encoded, &mut decoded);

        println!("{:03X?}", &decoded[..16]); // arbitrary cut-off

        let decoded_back_to_u16: Vec<u16> = decoded.iter().map(|x| *x as u16).collect();

        assert_eq!(&decoded_back_to_u16, &original_values);
    }
}
