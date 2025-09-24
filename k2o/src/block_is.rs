use std::time::Instant;

use ndarray::{ArrayBase, ArrayView, Dim, ViewRepr};

use crate::{
    block::K2Block,
    decode::{decode, decode_u16, decode_u32},
};

#[derive(Debug)]
pub struct K2ISBlock {
    sync: u32, // should be constant 0xFFFF0055
    // padding1: [u8; 4],
    version: u8,
    flags: u8,
    // padding2: [u8; 6],
    block_count: u32,

    /// sector width
    width: u16,

    /// sector height
    height: u16,

    frame_id: u32,

    /// start index in x direction
    pixel_x_start: u16,

    /// start index in y direction
    pixel_y_start: u16,

    /// end index in x direction, inclusive
    pixel_x_end: u16,

    /// end index in y direction, inclusive
    pixel_y_end: u16,

    block_size: u32, // for IS mode: fixed 0x5758; can be different in summit mode (0xc028)
    payload: Vec<u16>, // the already decoded payload
    sector_id: u8,   // not part of the actual data received, added as "metadata"
    decode_timestamp: Instant,
    // TODO: receive timestamp as an Instant
    acquisition_id: usize,
}

impl K2ISBlock {
    /// return a dummy block with specified fields, matching IS specs
    /// NOTE: only meant for testing!
    pub fn empty_for_pos(x: u16, y: u16, frame_id: u32, acquisition_id: usize) -> K2ISBlock {
        let payload: Vec<u16> = vec![0; Self::DECODED_SIZE];
        K2ISBlock {
            sync: 0xFFFF0055,
            version: 1,
            flags: 0x01,
            block_count: 0,
            width: 256,
            height: 1860,
            frame_id,
            pixel_x_start: x,
            pixel_y_start: y,
            pixel_x_end: x + 16 - 1,
            pixel_y_end: y + 930 - 1,
            block_size: 0x5758,
            payload,
            sector_id: 0,
            decode_timestamp: Instant::now(),
            acquisition_id,
        }
    }

    /// return a block for the given input data and specified position
    /// NOTE: not optimized for performance - creates a copy of the data
    /// only use for testing!
    pub fn from_vec_and_pos(
        data: &[u16],
        x: u16,
        y: u16,
        frame_id: u32,
        acquisition_id: usize,
    ) -> K2ISBlock {
        let payload = data.to_vec(); // create an owned copy here
        K2ISBlock {
            sync: 0xFFFF0055,
            version: 1,
            flags: 0x01,
            block_count: 0,
            width: 256,
            height: 1860,
            frame_id,
            pixel_x_start: x,
            pixel_y_start: y,
            pixel_x_end: x + 16 - 1,
            pixel_y_end: y + 930 - 1,
            block_size: 0x5758,
            payload,
            sector_id: 0,
            decode_timestamp: Instant::now(),
            acquisition_id,
        }
    }
}

impl K2Block for K2ISBlock {
    fn from_bytes(bytes: &[u8], sector_id: u8, acquisition_id: usize) -> K2ISBlock {
        // FIXME: we don't really need to initialize the vector, as it will be overwritten by `decode` just below...
        // FIXME: use MaybeUninit stuff from nightly?
        let mut payload = vec![0; Self::DECODED_SIZE];

        // println!("{}", payload.len());
        // payload.fill(0);
        // println!("{}", payload.len());
        decode::<{ Self::PACKET_SIZE }>(bytes, &mut payload);

        K2ISBlock {
            sync: decode_u32(&bytes[0..4]),
            version: bytes[8],
            flags: bytes[9],
            block_count: decode_u32(&bytes[16..20]),
            width: decode_u16(&bytes[20..22]),
            height: decode_u16(&bytes[22..24]),
            frame_id: decode_u32(&bytes[24..28]),
            pixel_x_start: decode_u16(&bytes[28..30]),
            pixel_y_start: decode_u16(&bytes[30..32]),
            pixel_x_end: decode_u16(&bytes[32..34]),
            pixel_y_end: decode_u16(&bytes[34..36]),
            block_size: decode_u32(&bytes[36..40]),
            payload,
            sector_id,
            decode_timestamp: Instant::now(),
            acquisition_id,
        }
    }

    fn replace_with(&mut self, bytes: &[u8], sector_id: u8, acquisition_id: usize) {
        decode::<{ Self::PACKET_SIZE }>(bytes, &mut self.payload);

        self.sync = decode_u32(&bytes[0..4]);
        self.version = bytes[8];
        self.flags = bytes[9];
        self.block_count = decode_u32(&bytes[16..20]);
        self.width = decode_u16(&bytes[20..22]);
        self.height = decode_u16(&bytes[22..24]);
        self.frame_id = decode_u32(&bytes[24..28]);
        self.pixel_x_start = decode_u16(&bytes[28..30]);
        self.pixel_y_start = decode_u16(&bytes[30..32]);
        self.pixel_x_end = decode_u16(&bytes[32..34]);
        self.pixel_y_end = decode_u16(&bytes[34..36]);
        self.block_size = decode_u32(&bytes[36..40]);
        self.sector_id = sector_id;
        self.decode_timestamp = Instant::now();
        self.acquisition_id = acquisition_id;
    }

    fn as_array(&self) -> ArrayBase<ViewRepr<&u16>, Dim<[usize; 2]>> {
        let height = self.pixel_y_end - self.pixel_y_start + 1;
        let width = self.pixel_x_end - self.pixel_x_start + 1;
        let view =
            ArrayView::from_shape((height as usize, width as usize), &self.payload[..]).unwrap();
        view
    }

    fn as_vec(&self) -> &Vec<u16> {
        &self.payload
    }

    /// return a dummy block, matching IS specs
    /// NOTE: only meant for testing!
    fn empty(first_frame_id: u32, acquisition_id: usize) -> Self {
        let payload: Vec<u16> = vec![0; Self::DECODED_SIZE];
        K2ISBlock {
            sync: 0xFFFF0055,
            version: 1,
            flags: 0x01,
            block_count: 0,
            width: 256,
            height: 1860,
            frame_id: first_frame_id,
            pixel_x_start: 0,
            pixel_y_start: 0,
            pixel_x_end: 15,
            pixel_y_end: 929,
            block_size: 0x5758,
            payload,
            sector_id: 0,
            decode_timestamp: Instant::now(),
            acquisition_id,
        }
    }

    fn get_flags(&self) -> u8 {
        self.flags
    }

    fn get_sector_width(&self) -> u16 {
        self.width
    }

    fn get_sector_height(&self) -> u16 {
        self.height
    }

    fn get_x_start(&self) -> u16 {
        self.pixel_x_start
    }

    fn get_y_start(&self) -> u16 {
        self.pixel_y_start
    }

    fn get_x_end(&self) -> u16 {
        self.pixel_x_end
    }

    fn get_y_end(&self) -> u16 {
        self.pixel_y_end
    }

    fn get_frame_id(&self) -> u32 {
        self.frame_id
    }

    fn get_sector_id(&self) -> u8 {
        self.sector_id
    }

    const PACKET_SIZE: usize = 0x5758;
    const DECODED_SIZE: usize = 14880;
    const BLOCK_WIDTH: usize = 16;
    const BLOCK_HEIGHT: usize = 930;
    const SECTOR_WIDTH: usize = 256;
    const SECTOR_HEIGHT: usize = 2 * 930;
    const BLOCKS_PER_SECTOR: u8 = 32;

    fn get_decoded_timestamp(&self) -> Instant {
        self.decode_timestamp
    }

    fn validate(&self) {
        assert_eq!(self.sync, 0xFFFF0055);
        assert_eq!(self.block_size, 0x5758)
    }

    fn get_acquisition_id(&self) -> usize {
        self.acquisition_id
    }
}
