use std::{ops::Range, time::Instant};

use ipc_test::{SharedSlabAllocator, Slot, SlotInfo};
use ndarray::{s, ArrayView2, ArrayViewMut2};

use crate::{block::K2Block, events::Binning, helpers::Shape2};

pub trait FrameForWriting: Sized {
    const FRAME_WIDTH: usize;
    const FRAME_HEIGHT: usize;

    /// The associated block type
    type Block: K2Block;

    /// The matching read-only frame type
    type ReadOnlyFrame: K2Frame<FrameForWriting = Self>;

    fn get_frame_id(&self) -> u32;
    fn get_created_timestamp(&self) -> Instant;
    fn get_modified_timestamp(&self) -> Instant;
    fn set_modified_timestamp(&mut self, ts: &Instant);
    fn get_payload(&self) -> &[u16];
    fn get_payload_mut(&mut self) -> &mut [u16];
    fn get_size_bytes() -> usize {
        Self::FRAME_WIDTH * Self::FRAME_HEIGHT * std::mem::size_of::<u16>()
    }

    fn empty(frame_id: u32, shm: &mut SharedSlabAllocator, acquisition_id: usize) -> Self {
        Self::empty_with_timestamp::<Self::Block>(frame_id, &Instant::now(), shm, acquisition_id)
    }

    fn empty_from_block<B: K2Block>(
        block: &B,
        shm: &mut SharedSlabAllocator,
        acquisition_id: usize,
    ) -> Self {
        Self::empty_with_timestamp::<B>(
            block.get_frame_id(),
            &block.get_decoded_timestamp(),
            shm,
            acquisition_id,
        )
    }
    fn empty_with_timestamp<B: K2Block>(
        frame_id: u32,
        ts: &Instant,
        shm: &mut SharedSlabAllocator,
        acquisition_id: usize,
    ) -> Self;

    fn writing_done(self, shm: &mut SharedSlabAllocator) -> Self::ReadOnlyFrame;

    fn reset(&mut self) {
        self.get_tracker_mut().fill(false);
    }

    fn reset_from_block<B: K2Block>(&mut self, block: &B) {
        self.reset_with_timestamp::<B>(block.get_frame_id(), &block.get_decoded_timestamp())
    }

    fn reset_with_timestamp<B: K2Block>(&mut self, frame_id: u32, ts: &Instant);

    fn as_array(&self) -> ArrayView2<u16> {
        let view =
            ArrayView2::from_shape((Self::FRAME_HEIGHT, Self::FRAME_WIDTH), self.get_payload())
                .unwrap();
        view
    }

    fn as_array_mut(&mut self) -> ArrayViewMut2<u16> {
        let view = ArrayViewMut2::from_shape(
            (Self::FRAME_HEIGHT, Self::FRAME_WIDTH),
            &mut self.get_payload_mut()[..],
        )
        .unwrap();
        view
    }

    ///
    /// Copy the data from `block` over to this `K2Frame`
    ///
    /// This assumes that the block belongs to this frame and will panic if this is not the case.
    ///
    /// Needs to handle the following artifacts:
    /// 1) Duplicate packets: with the assumption that this is a very rare event, we can let the assignment
    ///    continue as normal, as long as the tracking whether a block-position is already filled is
    ///    idempotent. If a duplicate comes in after a frame has finished, a new frame is kept around,
    ///    which is dropped after the timeout (because it is very unlikely that _all_ blocks of a frame
    ///    will be duplicated)
    /// 2) Dropped packets: we can't keep frames with missing data around indefinitely, so we track the
    ///    last-modified timestamp and time out once we are certain the packet is not only late but really
    ///    dropped
    /// 3) Out-of-order delivery
    ///
    fn assign_block(&mut self, block: &impl K2Block) {
        let block_arr = block.as_array();
        assert_eq!(block.get_frame_id(), self.get_frame_id());
        let mut frame_arr = ArrayViewMut2::from_shape(
            (Self::FRAME_HEIGHT, Self::FRAME_WIDTH),
            &mut self.get_payload_mut()[..],
        )
        .unwrap();
        let sector_x_offset = block.get_x_offset();
        let start_y = block.get_y_start() as usize;
        let start_x = (block.get_x_start() + sector_x_offset) as usize;
        let end_y = (block.get_y_end() + 1) as usize;
        let end_x = (block.get_x_end() + 1 + sector_x_offset) as usize;
        let mut dest_slice = frame_arr.slice_mut(s![start_y..end_y, start_x..end_x,]);

        dest_slice.assign(&block_arr);
        self.track_block(block);
    }

    fn is_finished(&self) -> bool {
        return self.get_tracker().iter().all(|&x| x);
    }

    fn get_tracker(&self) -> &Vec<bool>;

    fn get_tracker_mut(&mut self) -> &mut Vec<bool>;

    fn track_block<B: K2Block>(&mut self, block: &B) {
        // blocks are identified according to their (y, x) start coordinates
        let sector_x_offset = block.get_x_offset();
        let start_x = (block.get_x_start() + sector_x_offset) as usize;
        let start_y = block.get_y_start() as usize;

        let block_width = (block.get_x_end() - block.get_x_start() + 1) as usize;
        let block_height = (block.get_y_end() - block.get_y_start() + 1) as usize;
        let num_blocks_x_direction = Self::FRAME_WIDTH / block_width;

        assert_eq!(block_width, B::BLOCK_WIDTH);
        assert_eq!(block_height, B::BLOCK_HEIGHT);

        let block_id_x = start_x / block_width;
        let block_id_y = start_y / block_height;
        let block_id = num_blocks_x_direction * block_id_y + block_id_x;

        // println!("for start_x={} start_y={} have block_id={} block_id_x={} block_id_y={}", start_x, start_y, block_id, block_id_x, block_id_y);
        let tracker = self.get_tracker_mut();
        tracker.get(block_id).or_else(|| {
            println!("index out of bounds: {block_id} ...");
            println!(
                "start_x = {start_x}, start_y = {start_y}, num_blocks_x_direction = {num_blocks_x_direction}",
            );
            panic!("can't continue, sorry");
        });

        tracker[block_id] = true;

        // update "mtime"
        self.set_modified_timestamp(&Instant::now());
    }

    fn get_acquisition_id(&self) -> usize;
}

pub trait K2Frame: Send {
    const FRAME_WIDTH: usize;
    const FRAME_HEIGHT: usize;

    type Block: K2Block;
    type FrameForWriting: FrameForWriting<ReadOnlyFrame = Self>;

    fn into_slot(self, shm: &SharedSlabAllocator) -> Slot;
    fn free_payload(self, shm: &mut SharedSlabAllocator);
    fn get_frame_id(&self) -> u32;
    fn get_created_timestamp(&self) -> Instant;
    fn get_modified_timestamp(&self) -> Instant;
    fn get_pixel_size_bytes() -> usize {
        std::mem::size_of::<u16>()
    }
    fn get_size_bytes() -> usize {
        Self::FRAME_WIDTH * Self::FRAME_HEIGHT * Self::get_pixel_size_bytes()
    }

    fn into_generic(self) -> GenericFrame;

    fn get_shape_for_binning(binning: &Binning) -> Shape2;
    fn get_num_subframes(binning: &Binning) -> u32;
    fn subframe_indexes(&self, binning: &Binning) -> Range<u32>;
    fn get_subframe(&self, index: u32, binning: &Binning, shm: &SharedSlabAllocator) -> SubFrame;
    fn get_acquisition_id(&self) -> usize;

    /// to grab an independent copy of only some metadata of the frame
    fn get_meta(&self) -> FrameMeta {
        FrameMeta {
            acquisition_id: self.get_acquisition_id(),
            frame_id: self.get_frame_id(),
            created_timestamp: self.get_created_timestamp(),
        }
    }
}

pub struct FrameMeta {
    acquisition_id: usize,
    frame_id: u32,
    created_timestamp: Instant,
}

impl FrameMeta {
    pub fn new(acquisition_id: usize, frame_id: u32, created_timestamp: Instant) -> Self {
        Self {
            acquisition_id,
            frame_id,
            created_timestamp,
        }
    }
    pub fn get_acquisition_id(&self) -> usize {
        self.acquisition_id
    }

    pub fn get_created_timestamp(&self) -> Instant {
        self.created_timestamp
    }

    pub fn get_frame_id(&self) -> u32 {
        self.frame_id
    }
}

pub struct SubFrame {
    binning: Binning,
    shape: Shape2,
    slot: Slot,

    /// start index in u16 ("pixel")
    pub start: usize,

    /// end index in u16 ("pixel")
    pub end: usize,
}

impl SubFrame {
    pub fn new(binning: Binning, shape: Shape2, slot: Slot, start: usize, end: usize) -> SubFrame {
        SubFrame {
            binning,
            shape,
            slot,
            start,
            end,
        }
    }

    pub fn apply_to_payload(&self, f: impl Fn(&[u16])) {
        let payload_raw: &[u16] = &bytemuck::cast_slice(self.slot.as_slice())[self.start..self.end];
        f(payload_raw)
    }

    pub fn apply_to_payload_raw(&self, f: impl Fn(&[u8])) {
        let payload_sliced: &[u16] =
            &bytemuck::cast_slice(self.slot.as_slice())[self.start..self.end];
        let payload_raw = bytemuck::cast_slice(payload_sliced);
        f(payload_raw)
    }

    pub fn apply_to_payload_array(&self, mut f: impl FnMut(ArrayView2<u16>)) {
        let payload_sliced: &[u16] =
            &bytemuck::cast_slice(self.slot.as_slice())[self.start..self.end];
        let view = ArrayView2::from_shape(self.shape, payload_sliced).unwrap();
        f(view)
    }

    pub fn get_binning(&self) -> Binning {
        self.binning
    }
}

/// Once decoding, assembly etc. is finished we convert to this generic
/// frame type for downstream processing
pub struct GenericFrame {
    /// a reference to the decoded payload of the whole frame
    pub payload: SlotInfo,

    /// the frame id as received
    pub frame_id: u32,

    /// when the first block was received, to handle dropped packets
    pub created_timestamp: Instant,

    /// when the last block was received, to handle dropped packets
    pub modified_timestamp: Instant,

    /// Acquisition id/generation
    pub acquisition_id: usize,
}

impl GenericFrame {
    pub fn new(
        payload: SlotInfo,
        frame_id: u32,
        created_timestamp: Instant,
        modified_timestamp: Instant,
        acquisition_id: usize,
    ) -> GenericFrame {
        GenericFrame {
            payload,
            frame_id,
            created_timestamp,
            modified_timestamp,
            acquisition_id,
        }
    }

    pub fn get_frame_id(&self) -> u32 {
        self.frame_id
    }

    pub fn into_slot(self, shm: &SharedSlabAllocator) -> Slot {
        shm.get(self.payload.slot_idx)
    }

    pub fn free_payload(self, shm: &mut SharedSlabAllocator) {
        let slot_r = self.into_slot(shm);
        shm.free_idx(slot_r.slot_idx);
    }
}
