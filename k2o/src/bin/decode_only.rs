use k2o::decode::decode_unrolled;
use k2o::decode::HEADER_SIZE;

fn main() {
    const PACKET_SIZE: usize = 0x5758;
    const DECODED_SIZE: usize = (PACKET_SIZE - HEADER_SIZE) * 2 / 3;
    let input: [u8; PACKET_SIZE] = [0; PACKET_SIZE];
    let mut out: [u16; DECODED_SIZE] = [0; DECODED_SIZE];
    for _ in 0..32 * 400 * 8 * 4 * 2 {
        decode_unrolled::<PACKET_SIZE, DECODED_SIZE>(&input, &mut out);
    }
}
