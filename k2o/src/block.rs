use std::time::Instant;

use ndarray::{ArrayBase, ArrayView, Dim, ViewRepr};

use crate::decode::{decode, decode_u16, decode_u16_vec, decode_u32};

pub trait K2Block: Send {
    const PACKET_SIZE: usize;
    const DECODED_SIZE: usize;
    const BLOCK_WIDTH: usize;
    const BLOCK_HEIGHT: usize;
    const SECTOR_WIDTH: usize;
    const SECTOR_HEIGHT: usize;
    const BLOCKS_PER_SECTOR: u8;

    fn from_bytes(bytes: &[u8], sector_id: u8) -> Self;
    fn replace_with(&mut self, bytes: &[u8], sector_id: u8);
    fn as_array(&self) -> ArrayBase<ViewRepr<&u16>, Dim<[usize; 2]>>;
    fn as_vec(&self) -> &Vec<u16>;
    fn empty(first_frame_id: u32) -> Self;
    fn get_flags(&self) -> u8;
    fn sync_is_set(&self) -> bool {
        (self.get_flags() & 0x01) == 0x01
    }
    fn get_sector_width(&self) -> u16;
    fn get_sector_height(&self) -> u16;
    fn get_x_start(&self) -> u16;
    fn get_y_start(&self) -> u16;
    fn get_x_end(&self) -> u16;
    fn get_y_end(&self) -> u16;
    fn get_x_offset(&self) -> u16 {
        self.get_sector_width() * (self.get_sector_id() as u16)
    }
    fn get_frame_id(&self) -> u32;
    fn get_sector_id(&self) -> u8;
    fn get_decoded_timestamp(&self) -> Instant;
    fn validate(&self);
}

/// routing info, extracted from the block itself
/// why? to have the block itself and routing info separately
/// so they don't cause cache contention issues (example: if we have a single thread that
/// routes blocks around, we don't want to forcibly move parts of the block
/// itself to the core that is running said thread)
#[derive(Clone, Copy)]
pub struct BlockRouteInfo {
    frame_id: u32,
    sector_id: u8,
}

impl BlockRouteInfo {
    pub fn new<B: K2Block>(block: &B) -> BlockRouteInfo {
        BlockRouteInfo {
            frame_id: block.get_frame_id(),
            sector_id: block.get_sector_id(),
        }
    }
    pub fn get_frame_id(&self) -> u32 {
        self.frame_id
    }

    pub fn get_sector_id(&self) -> u8 {
        self.sector_id
    }
}
