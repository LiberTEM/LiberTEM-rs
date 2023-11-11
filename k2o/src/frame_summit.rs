use std::{ops::Range, time::Instant};

use ipc_test::{SharedSlabAllocator, SlotForWriting, SlotInfo};

use crate::{
    block::{K2Block, K2SummitBlock},
    events::Binning,
    frame::{FrameForWriting, K2Frame, SubFrame},
    helpers::Shape2,
};

pub struct K2SummitFrame {
    /// the decoded payload of the whole frame
    pub payload: SlotInfo,

    subframe_idx: u8,

    /// the frame id as received
    pub frame_id: u32,

    /// when the first block was received, to handle dropped packets
    pub created_timestamp: Instant,

    /// when the last block was received, to handle dropped packets
    pub modified_timestamp: Instant,
}

pub struct K2SummitFrameForWriting {
    /// the decoded payload of the whole frame
    pub payload: SlotForWriting,
    // pub payload_slices: Vec<ArrayBase<ViewRepr<& u16>, Dim<[usize; 2]>>>,
    /// used to keep track of which block positions we have seen
    tracker: Vec<bool>,

    /// the frame id as received
    pub frame_id: u32,

    subframe_idx: u8,

    /// when the first block was received, to handle dropped packets
    pub created_timestamp: Instant,

    /// when the last block was received, to handle dropped packets
    pub modified_timestamp: Instant,
}

impl K2SummitFrameForWriting {}

impl FrameForWriting for K2SummitFrameForWriting {
    const FRAME_WIDTH: usize = 4096;
    const FRAME_HEIGHT: usize = 3840;

    fn get_frame_id(&self) -> u32 {
        self.frame_id
    }

    fn get_created_timestamp(&self) -> Instant {
        self.created_timestamp
    }

    fn get_modified_timestamp(&self) -> Instant {
        self.modified_timestamp
    }

    fn get_payload(&self) -> &[u16] {
        let slot_as_u16: &[u16] = bytemuck::cast_slice(self.payload.as_slice());
        slot_as_u16
    }

    fn empty_with_timestamp<B: K2Block>(
        frame_id: u32,
        ts: &Instant,
        shm: &mut SharedSlabAllocator,
    ) -> Self {
        let mut payload = shm.get_mut().expect("get free SHM slot");
        assert!(payload.size >= Self::FRAME_WIDTH * Self::FRAME_HEIGHT);

        for item in payload.as_slice_mut() {
            *item = 0;
        }

        // how many blocks are there? in IS-mode, we have
        // FRAME_WIDTH=2048, FRAME_HEIGHT=1860
        // block_width == 16, block_height == 930
        // -> we have 1860/930=2 blocks in y direction, and 2048/16=128 blocks in x direction
        // -> 256 blocks per frame
        // or, alternatively: 2048*1860/930/16 == 256
        let tracker: Vec<bool> = vec![false; 8 * B::BLOCKS_PER_SECTOR as usize];

        Self {
            payload,
            tracker,
            frame_id,
            created_timestamp: *ts,
            modified_timestamp: Instant::now(),
            subframe_idx: 0,
        }
    }

    fn reset_with_timestamp<B: K2Block>(&mut self, frame_id: u32, ts: &Instant) {
        self.reset();
        self.created_timestamp = *ts;
        self.modified_timestamp = *ts;
        self.frame_id = frame_id;
        self.subframe_idx = 0;
    }

    fn set_modified_timestamp(&mut self, ts: &Instant) {
        self.modified_timestamp = *ts;
    }

    fn get_tracker_mut(&mut self) -> &mut Vec<bool> {
        &mut self.tracker
    }

    fn get_payload_mut(&mut self) -> &mut [u16] {
        let slot_as_u16: &mut [u16] = bytemuck::cast_slice_mut(self.payload.as_slice_mut());
        slot_as_u16
    }

    fn get_tracker(&self) -> &Vec<bool> {
        &self.tracker
    }

    type Block = K2SummitBlock;

    type ReadOnlyFrame = K2SummitFrame;

    fn writing_done(self, shm: &mut SharedSlabAllocator) -> Self::ReadOnlyFrame {
        let slot_info = shm.writing_done(self.payload);
        K2SummitFrame {
            payload: slot_info,
            subframe_idx: self.subframe_idx,
            frame_id: self.frame_id,
            created_timestamp: self.created_timestamp,
            modified_timestamp: self.modified_timestamp,
        }
    }
}

impl K2Frame for K2SummitFrame {
    type Block = K2SummitBlock;

    fn get_shape_for_binning(_binning: &Binning) -> Shape2 {
        (Self::FRAME_HEIGHT, Self::FRAME_WIDTH)
    }

    fn subframe_indexes(&self, _binning: &Binning) -> Range<u32> {
        0..1
    }

    fn get_subframe(&self, index: u32, binning: &Binning, shm: &SharedSlabAllocator) -> SubFrame {
        let slot = shm.get(self.payload.slot_idx);
        let start = 0;
        assert!(index == 0, "can't handle sub frames yet");
        let end = start + Self::FRAME_HEIGHT * Self::FRAME_WIDTH;

        SubFrame::new(
            *binning,
            Self::get_shape_for_binning(binning),
            slot,
            start,
            end,
        )
    }

    fn get_num_subframes(_binning: &Binning) -> u32 {
        1
    }

    const FRAME_WIDTH: usize = 4096;
    const FRAME_HEIGHT: usize = 3840;

    type FrameForWriting = K2SummitFrameForWriting;

    fn get_frame_id(&self) -> u32 {
        self.frame_id
    }

    fn get_created_timestamp(&self) -> Instant {
        self.created_timestamp
    }

    fn get_modified_timestamp(&self) -> Instant {
        self.modified_timestamp
    }

    fn into_slot(self, shm: &SharedSlabAllocator) -> ipc_test::Slot {
        shm.get(self.payload.slot_idx)
    }
}
