use std::{ops::Range, time::Instant, vec};

use ipc_test::{SharedSlabAllocator, Slot, SlotForWriting, SlotInfo};
use itertools::Itertools;
use ndarray::{s, ArrayView2, ArrayViewMut2};

use crate::{
    block::{K2Block, K2ISBlock},
    events::Binning,
    helpers::Shape2,
};

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

    fn empty(frame_id: u32, shm: &mut SharedSlabAllocator) -> Self {
        Self::empty_with_timestamp::<Self::Block>(frame_id, &Instant::now(), shm)
    }

    fn empty_from_block<B: K2Block>(block: &B, shm: &mut SharedSlabAllocator) -> Self {
        Self::empty_with_timestamp::<B>(block.get_frame_id(), &block.get_decoded_timestamp(), shm)
    }
    fn empty_with_timestamp<B: K2Block>(
        frame_id: u32,
        ts: &Instant,
        shm: &mut SharedSlabAllocator,
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
}

pub trait K2Frame: Send + Sized {
    const FRAME_WIDTH: usize;
    const FRAME_HEIGHT: usize;

    type Block: K2Block;
    type FrameForWriting: FrameForWriting<ReadOnlyFrame = Self>;

    fn get_frame_id(&self) -> u32;
    fn get_created_timestamp(&self) -> Instant;
    fn get_modified_timestamp(&self) -> Instant;
    fn get_size_bytes() -> usize {
        Self::FRAME_WIDTH * Self::FRAME_HEIGHT * std::mem::size_of::<u16>()
    }

    fn free_payload(self, shm: &mut SharedSlabAllocator) {
        let slot_r = self.into_slot(shm);
        shm.free_idx(slot_r.slot_idx);
    }

    fn into_slot(self, smh: &SharedSlabAllocator) -> Slot;

    // fn as_array(&self, shm: &SharedSlabAllocator) -> ArrayView2<u16> {
    //     let view = ArrayView2::from_shape(
    //         (Self::FRAME_HEIGHT, Self::FRAME_WIDTH),
    //         self.get_payload(shm),
    //     )
    //     .unwrap();
    //     view
    // }

    fn get_shape_for_binning(binning: &Binning) -> Shape2;
    fn get_num_subframes(binning: &Binning) -> u32;
    fn subframe_indexes(&self, binning: &Binning) -> Range<u32>;
    fn get_subframe(&self, index: u32, binning: &Binning, shm: &SharedSlabAllocator) -> SubFrame;
}

pub struct SubFrame {
    binning: Binning,
    shape: Shape2,
    slot: Slot,

    /// start index in u16
    start: usize,

    /// end index in u16
    end: usize,
}

impl SubFrame {
    // pub fn get_payload(&self) -> &'a [u16] {
    //     self.payload
    // }
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

    // pub fn as_array(&self) -> ArrayView2<u16> {
    //     let view = ArrayView2::from_shape(self.shape, self.get_payload()).unwrap();
    //     view
    // }
}

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
}

impl FrameForWriting for K2ISFrameForWriting {
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
        let tracker: Vec<bool> =
            vec![false; Self::FRAME_WIDTH * Self::FRAME_HEIGHT / B::DECODED_SIZE];

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

    fn writing_done(self, shm: &mut SharedSlabAllocator) -> Self::ReadOnlyFrame {
        let slot_info = shm.writing_done(self.payload);
        K2ISFrame {
            payload: slot_info,
            subframe_idx: self.subframe_idx,
            frame_id: self.frame_id,
            created_timestamp: self.created_timestamp,
            modified_timestamp: self.modified_timestamp,
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
        todo!("return type? Slot, which the called can convert to a slice?")
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
}

impl K2Frame for K2ISFrame {
    const FRAME_WIDTH: usize = 2048;
    const FRAME_HEIGHT: usize = 1860;

    type Block = K2ISBlock;
    type FrameForWriting = K2ISFrameForWriting;

    fn get_frame_id(&self) -> u32 {
        self.frame_id
    }

    fn get_created_timestamp(&self) -> Instant {
        self.created_timestamp
    }

    fn get_modified_timestamp(&self) -> Instant {
        self.modified_timestamp
    }

    fn into_slot(self, shm: &SharedSlabAllocator) -> Slot {
        shm.get(self.payload.slot_idx)
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

        SubFrame {
            binning: *binning,
            slot,
            shape,
            start,
            end,
        }
    }

    fn get_num_subframes(binning: &Binning) -> u32 {
        match binning {
            Binning::Bin1x => 1,
            Binning::Bin2x => 2,
            Binning::Bin4x => 3,
            Binning::Bin8x => 4,
        }
    }
}

pub struct K2SummitFrame {
    /// the decoded payload of the whole frame
    pub payload: SlotInfo,
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

    fn writing_done(self, shm: &mut SharedSlabAllocator) -> SlotInfo {
        shm.writing_done(self.payload)
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

        SubFrame {
            binning: *binning,
            slot,
            shape: Self::get_shape_for_binning(binning),
            start,
            end,
        }
    }

    fn get_num_subframes(_binning: &Binning) -> u32 {
        1
    }
}
