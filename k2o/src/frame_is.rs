use std::{ops::Range, time::Instant};

use ipc_test::{SharedSlabAllocator, Slot, SlotForWriting, SlotInfo};
use itertools::Itertools;

use crate::{
    block::K2Block,
    block_is::K2ISBlock,
    events::Binning,
    frame::{FrameForWriting, GenericFrame, K2Frame, SubFrame},
    helpers::Shape2,
};

pub struct K2ISFrameForWriting {
    /// the decoded payload of the whole frame
    pub payload: SlotForWriting,
    /// used to keep track of which block positions we have seen
    tracker: Vec<bool>,

    subframe_idx: u8,

    /// the frame id as received
    pub frame_id: u32,

    /// when the first block was received, to handle dropped packets
    pub created_timestamp: Instant,

    /// when the last block was received, to handle dropped packets
    pub modified_timestamp: Instant,

    /// Acquisition id/generation
    pub acquisition_id: usize,
}

impl FrameForWriting for K2ISFrameForWriting {
    fn empty_with_timestamp<B: K2Block>(
        frame_id: u32,
        ts: &Instant,
        shm: &mut SharedSlabAllocator,
        acquisition_id: usize,
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
        let tracker: Vec<bool> =
            vec![false; Self::FRAME_WIDTH * Self::FRAME_HEIGHT / B::DECODED_SIZE];

        Self {
            payload,
            tracker,
            frame_id,
            created_timestamp: *ts,
            modified_timestamp: Instant::now(),
            subframe_idx: 0,
            acquisition_id,
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

    fn writing_done(self, shm: &mut SharedSlabAllocator) -> Self::ReadOnlyFrame {
        let slot_info = shm.writing_done(self.payload);
        K2ISFrame {
            payload: slot_info,
            subframe_idx: self.subframe_idx,
            frame_id: self.frame_id,
            created_timestamp: self.created_timestamp,
            modified_timestamp: self.modified_timestamp,
            acquisition_id: self.acquisition_id,
        }
    }

    const FRAME_WIDTH: usize = 2048;
    const FRAME_HEIGHT: usize = 1860;

    type Block = K2ISBlock;
    type ReadOnlyFrame = K2ISFrame;

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

    fn get_acquisition_id(&self) -> usize {
        self.acquisition_id
    }
}

impl K2ISFrameForWriting {
    pub fn dump_finished_state(&self) {
        let counts = self.tracker.iter().counts();
        println!(
            "have {} hits and {} still missing",
            counts.get(&true).unwrap_or(&0),
            counts.get(&false).unwrap_or(&0)
        );
    }

    fn into_readonly(self, slot_info: SlotInfo) -> K2ISFrame {
        K2ISFrame {
            payload: slot_info,
            subframe_idx: self.subframe_idx,
            frame_id: self.frame_id,
            created_timestamp: self.created_timestamp,
            modified_timestamp: self.modified_timestamp,
            acquisition_id: self.acquisition_id,
        }
    }
}

pub struct K2ISFrame {
    /// a reference to the decoded payload of the whole frame
    pub payload: SlotInfo,

    subframe_idx: u8,

    /// the frame id as received
    pub frame_id: u32,

    /// when the first block was received, to handle dropped packets
    pub created_timestamp: Instant,

    /// when the last block was received, to handle dropped packets
    pub modified_timestamp: Instant,

    /// Acquisition id/generation
    pub acquisition_id: usize,
}

impl K2Frame for K2ISFrame {
    const FRAME_WIDTH: usize = 2048;
    const FRAME_HEIGHT: usize = 1860;

    type Block = K2ISBlock;
    type FrameForWriting = K2ISFrameForWriting;

    fn get_created_timestamp(&self) -> Instant {
        self.created_timestamp
    }

    fn get_modified_timestamp(&self) -> Instant {
        self.modified_timestamp
    }

    fn get_shape_for_binning(binning: &Binning) -> Shape2 {
        match binning {
            Binning::Bin1x => (Self::FRAME_HEIGHT, Self::FRAME_WIDTH),
            Binning::Bin2x => (895, Self::FRAME_WIDTH),
            Binning::Bin4x => (574, Self::FRAME_WIDTH),
            Binning::Bin8x => (414, Self::FRAME_WIDTH),
        }
    }

    fn subframe_indexes(&self, binning: &Binning) -> Range<u32> {
        match binning {
            Binning::Bin1x => 0..1,
            Binning::Bin2x => 0..2,
            Binning::Bin4x => 0..3,
            Binning::Bin8x => 0..4,
        }
    }

    fn get_subframe(&self, index: u32, binning: &Binning, shm: &SharedSlabAllocator) -> SubFrame {
        let skip_rows: usize = match binning {
            Binning::Bin1x => 0,
            Binning::Bin2x => 4,
            Binning::Bin4x => 4,
            Binning::Bin8x => 4,
        } + (index as usize * 66);
        let shape = Self::get_shape_for_binning(binning);
        let slot = shm.get(self.payload.slot_idx);
        // let payload_raw = bytemuck::cast_slice(slot.as_slice());

        let start = shape.1 * skip_rows;
        let end = start + shape.0 * shape.1;
        // let payload = &payload_raw[start..end];

        SubFrame::new(*binning, shape, slot, start, end)
    }

    fn get_num_subframes(binning: &Binning) -> u32 {
        match binning {
            Binning::Bin1x => 1,
            Binning::Bin2x => 2,
            Binning::Bin4x => 3,
            Binning::Bin8x => 4,
        }
    }

    fn into_generic(self) -> GenericFrame {
        GenericFrame::new(
            self.payload,
            self.frame_id,
            self.created_timestamp,
            self.modified_timestamp,
            self.acquisition_id,
        )
    }

    fn into_slot(self, shm: &SharedSlabAllocator) -> Slot {
        shm.get(self.payload.slot_idx)
    }

    fn free_payload(self, shm: &mut SharedSlabAllocator) {
        let slot_r = self.into_slot(shm);
        shm.free_idx(slot_r.slot_idx);
    }

    fn get_frame_id(&self) -> u32 {
        self.frame_id
    }

    fn get_acquisition_id(&self) -> usize {
        self.acquisition_id
    }
}
