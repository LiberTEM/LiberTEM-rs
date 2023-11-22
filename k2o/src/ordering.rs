use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashSet},
};

use itertools::Itertools;

use crate::{acquisition::AcquisitionResult, frame::GenericFrame};

pub enum FrameOrderingResult {
    Buffered,
    Dropped,
    NextFrame(FrameWithIdx),
}

pub enum FrameWithIdx {
    Frame(GenericFrame, u32),
    DroppedFrame(GenericFrame, u32),
}

impl From<FrameWithIdx> for AcquisitionResult<GenericFrame> {
    fn from(val: FrameWithIdx) -> Self {
        match val {
            FrameWithIdx::Frame(frame, frame_idx) => AcquisitionResult::Frame(frame, frame_idx),
            FrameWithIdx::DroppedFrame(frame, frame_idx) => {
                AcquisitionResult::DroppedFrame(frame, frame_idx)
            }
        }
    }
}

impl FrameWithIdx {
    fn get_frame(&self) -> &GenericFrame {
        match self {
            FrameWithIdx::Frame(frame, _idx) => frame,
            FrameWithIdx::DroppedFrame(frame, _idx) => frame,
        }
    }

    fn get_idx(&self) -> u32 {
        match self {
            FrameWithIdx::Frame(_, idx) | FrameWithIdx::DroppedFrame(_, idx) => *idx,
        }
    }
}

impl PartialEq for FrameWithIdx {
    fn eq(&self, other: &Self) -> bool {
        self.get_frame().get_frame_id() == other.get_frame().get_frame_id()
    }
}

impl PartialOrd for FrameWithIdx {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for FrameWithIdx {}

impl Ord for FrameWithIdx {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.get_frame()
            .get_frame_id()
            .cmp(&other.get_frame().get_frame_id())
    }
}

impl FrameOrderingResult {
    pub fn is_frame(&self) -> bool {
        matches!(self, Self::NextFrame(..))
    }

    pub fn is_buffered(&self) -> bool {
        matches!(self, Self::Buffered)
    }

    pub fn is_dropped(&self) -> bool {
        matches!(self, Self::Dropped)
    }
}

pub struct FrameOrdering {
    frame_buffer: BinaryHeap<Reverse<FrameWithIdx>>,
    next_expected_frame_idx: u32,
    dropped_idxs: HashSet<u32>,
}

impl FrameOrdering {
    // 100 => ~200ms of frames, so should be enough such that if we are unlucky
    // and miss two frames in a 100ms window, we still don't have to re-allocate
    const DEFAULT_CAPACITY: usize = 100;

    pub fn new(first_expected_frame_idx: u32) -> Self {
        Self {
            frame_buffer: BinaryHeap::with_capacity(Self::DEFAULT_CAPACITY),
            next_expected_frame_idx: first_expected_frame_idx,
            dropped_idxs: HashSet::new(),
        }
    }

    pub fn handle_frame(&mut self, frame_w_idx: FrameWithIdx) -> FrameOrderingResult {
        let expected_frame_idx = self.next_expected_frame_idx;

        // Frame indices can be repeated, in case we drop a frame and later a
        // block of said frame arrives, which starts a new frame with the old index.
        // filter these out here:
        if self.dropped_idxs.contains(&frame_w_idx.get_idx()) {
            return FrameOrderingResult::Dropped;
        }
        if let FrameWithIdx::DroppedFrame(_, idx) = frame_w_idx {
            self.dropped_idxs.insert(idx);
        }

        if frame_w_idx.get_idx() == expected_frame_idx {
            self.next_expected_frame_idx = expected_frame_idx + 1;
            FrameOrderingResult::NextFrame(frame_w_idx)
        } else {
            self.insert_sorted(frame_w_idx);
            FrameOrderingResult::Buffered
        }
    }

    pub fn maybe_get_next_frame(&mut self) -> Option<FrameWithIdx> {
        // cases:
        // 1) `self.frame_buffer` is empty -> no next frame
        // 2) `self.frame_buffer` is non-empty and starts the expected frame -> emit first frame

        match self.frame_buffer.peek() {
            Some(Reverse(frame)) => {
                let expected_frame_idx = self.next_expected_frame_idx;
                if frame.get_idx() == expected_frame_idx {
                    if let Some(Reverse(f)) = self.frame_buffer.pop() {
                        self.next_expected_frame_idx = f.get_idx() + 1;
                        Some(f)
                    } else {
                        unreachable!("should have Some(frame) here, we just peeked!")
                    }
                } else {
                    None
                }
            }
            None => None,
        }
    }

    /// Insert `frame` into the buffer at the correct position. Will panic if
    /// this `FrameOrdering` instance is not compacted.
    fn insert_sorted(&mut self, frame_w_idx: FrameWithIdx) {
        self.frame_buffer.push(Reverse(frame_w_idx));
    }

    pub fn is_empty(&self) -> bool {
        self.frame_buffer.len() == 0
    }

    pub(crate) fn dump_if_nonempty(&self) {
        if !self.is_empty() {
            println!("\n\nDUMP START");
            println!("next expected frame idx: {}", self.next_expected_frame_idx);
            for rf in self.frame_buffer.iter().sorted() {
                let Reverse(f) = rf;
                print!("{:?} - ", f.get_idx());
            }
            println!("\n\nDUMP END");
        }
    }
}

#[cfg(test)]
mod tests {
    use ipc_test::SharedSlabAllocator;
    use tempfile::tempdir;

    use crate::{
        frame::{FrameForWriting, K2Frame},
        frame_is::{K2ISFrame, K2ISFrameForWriting},
        ordering::FrameWithIdx,
    };

    use super::FrameOrdering;

    #[test]
    fn test_direct_return() {
        let socket_dir = tempdir().unwrap();
        let socket_as_path = socket_dir.into_path().join("test.socket");

        let mut ordering: FrameOrdering = FrameOrdering::new(0);

        let mut ssa = SharedSlabAllocator::new(
            10,
            K2ISFrame::FRAME_HEIGHT * K2ISFrame::FRAME_WIDTH * std::mem::size_of::<u16>(),
            false,
            &socket_as_path,
        )
        .expect("create SHM area for testing");

        let f1w: K2ISFrameForWriting = K2ISFrameForWriting::empty(1, &mut ssa, 0);
        let f1 = f1w.writing_done(&mut ssa);
        let f2w: K2ISFrameForWriting = K2ISFrameForWriting::empty(2, &mut ssa, 0);
        let f2 = f2w.writing_done(&mut ssa);
        let f3w: K2ISFrameForWriting = K2ISFrameForWriting::empty(3, &mut ssa, 0);
        let f3 = f3w.writing_done(&mut ssa);

        assert_eq!(ordering.frame_buffer.len(), 0);
        assert!(ordering
            .handle_frame(FrameWithIdx::Frame(f1.into_generic(), 0))
            .is_frame());
        assert_eq!(ordering.frame_buffer.len(), 0);
        assert!(ordering
            .handle_frame(FrameWithIdx::Frame(f2.into_generic(), 1))
            .is_frame());
        assert_eq!(ordering.frame_buffer.len(), 0);
        assert!(ordering
            .handle_frame(FrameWithIdx::Frame(f3.into_generic(), 2))
            .is_frame());

        assert_eq!(ordering.next_expected_frame_idx, 3);

        assert_eq!(ordering.frame_buffer.len(), 0);
        assert!(ordering.maybe_get_next_frame().is_none());
    }

    #[test]
    fn test_one_missing() {
        let socket_dir = tempdir().unwrap();
        let socket_as_path = socket_dir.into_path().join("test.socket");
        let mut ordering: FrameOrdering = FrameOrdering::new(0);

        let mut ssa = SharedSlabAllocator::new(
            10,
            K2ISFrame::FRAME_HEIGHT * K2ISFrame::FRAME_WIDTH * std::mem::size_of::<u16>(),
            false,
            &socket_as_path,
        )
        .expect("create SHM area for testing");

        let f1w: K2ISFrameForWriting = K2ISFrameForWriting::empty(1, &mut ssa, 0);
        let f1 = f1w.writing_done(&mut ssa);
        let f2w: K2ISFrameForWriting = K2ISFrameForWriting::empty(2, &mut ssa, 0);
        let f2 = f2w.writing_done(&mut ssa);
        let f3w: K2ISFrameForWriting = K2ISFrameForWriting::empty(3, &mut ssa, 0);
        let f3 = f3w.writing_done(&mut ssa);

        assert!(ordering
            .handle_frame(FrameWithIdx::Frame(f1.into_generic(), 0))
            .is_frame());
        assert!(ordering
            .handle_frame(FrameWithIdx::Frame(f3.into_generic(), 2))
            .is_buffered());

        // f1 was the last emitted frame, f3 is buffered, f2 is expected:
        assert_eq!(ordering.next_expected_frame_idx, 1);
        assert_eq!(ordering.frame_buffer.len(), 1);
        assert!(ordering.maybe_get_next_frame().is_none());

        // now, we push in f2, which is directly emitted, because its frame id
        // matches the currently expected id (f1 + 1):
        assert!(ordering
            .handle_frame(FrameWithIdx::Frame(f2.into_generic(), 1))
            .is_frame());
        assert_eq!(ordering.next_expected_frame_idx, 2);
        assert_eq!(ordering.frame_buffer.len(), 1);

        assert!(
            // we get a frame, and it has to be f3 with idx 2:
            match ordering.maybe_get_next_frame() {
                Some(frame) => {
                    frame.get_idx() == 2
                }
                None => false,
            }
        );
    }

    #[test]
    fn test_multiple_buffered() {
        let socket_dir = tempdir().unwrap();
        let socket_as_path = socket_dir.into_path().join("test.socket");
        let mut ordering: FrameOrdering = FrameOrdering::new(0);

        let mut ssa = SharedSlabAllocator::new(
            10,
            K2ISFrame::FRAME_HEIGHT * K2ISFrame::FRAME_WIDTH * std::mem::size_of::<u16>(),
            false,
            &socket_as_path,
        )
        .expect("create SHM area for testing");

        let f1w: K2ISFrameForWriting = K2ISFrameForWriting::empty(1, &mut ssa, 0);
        let f1 = f1w.writing_done(&mut ssa);
        let f2w: K2ISFrameForWriting = K2ISFrameForWriting::empty(2, &mut ssa, 0);
        let f2 = f2w.writing_done(&mut ssa);
        let f3w: K2ISFrameForWriting = K2ISFrameForWriting::empty(3, &mut ssa, 0);
        let f3 = f3w.writing_done(&mut ssa);
        let f4w: K2ISFrameForWriting = K2ISFrameForWriting::empty(4, &mut ssa, 0);
        let f4 = f4w.writing_done(&mut ssa);

        assert!(ordering
            .handle_frame(FrameWithIdx::Frame(f1.into_generic(), 0))
            .is_frame());
        assert!(ordering
            .handle_frame(FrameWithIdx::Frame(f3.into_generic(), 2))
            .is_buffered());
        assert!(ordering
            .handle_frame(FrameWithIdx::Frame(f4.into_generic(), 3))
            .is_buffered());

        // f1 was the last emitted frame, f3 and f4 are buffered:
        assert_eq!(ordering.next_expected_frame_idx, 1);
        assert_eq!(ordering.frame_buffer.len(), 2);
        assert!(ordering.maybe_get_next_frame().is_none());

        // now, we push in f2, which is directly emitted, because its frame id
        // matches the currently expected id (f1 + 1):
        assert!(ordering
            .handle_frame(FrameWithIdx::Frame(f2.into_generic(), 1))
            .is_frame());
        assert_eq!(ordering.next_expected_frame_idx, 2);
        assert_eq!(ordering.frame_buffer.len(), 2);

        assert!(
            // we get a frame, and it has to be frame id 3:
            match ordering.maybe_get_next_frame() {
                Some(frame) => {
                    frame.get_idx() == 2
                }
                None => false,
            }
        );

        assert!(
            // we get a frame, and it has to be frame id 4:
            match ordering.maybe_get_next_frame() {
                Some(frame) => {
                    frame.get_idx() == 3
                }
                None => false,
            }
        );
    }

    #[test]
    fn test_multiple_holes() {
        let socket_dir = tempdir().unwrap();
        let socket_as_path = socket_dir.into_path().join("test.socket");
        // something like this: f1 _ f3 _ f5
        let mut ordering: FrameOrdering = FrameOrdering::new(0);

        let mut ssa = SharedSlabAllocator::new(
            10,
            K2ISFrame::FRAME_HEIGHT * K2ISFrame::FRAME_WIDTH * std::mem::size_of::<u16>(),
            false,
            &socket_as_path,
        )
        .expect("create SHM area for testing");

        let f1w: K2ISFrameForWriting = K2ISFrameForWriting::empty(1, &mut ssa, 0);
        let f1 = f1w.writing_done(&mut ssa);
        let f2w: K2ISFrameForWriting = K2ISFrameForWriting::empty(2, &mut ssa, 0);
        let f2 = f2w.writing_done(&mut ssa);
        let f3w: K2ISFrameForWriting = K2ISFrameForWriting::empty(3, &mut ssa, 0);
        let f3 = f3w.writing_done(&mut ssa);
        let f4w: K2ISFrameForWriting = K2ISFrameForWriting::empty(4, &mut ssa, 0);
        let f4 = f4w.writing_done(&mut ssa);
        let f5w: K2ISFrameForWriting = K2ISFrameForWriting::empty(5, &mut ssa, 0);
        let f5 = f5w.writing_done(&mut ssa);

        assert!(ordering
            .handle_frame(FrameWithIdx::Frame(f1.into_generic(), 0))
            .is_frame());
        assert!(ordering
            .handle_frame(FrameWithIdx::Frame(f3.into_generic(), 2))
            .is_buffered());
        assert!(ordering
            .handle_frame(FrameWithIdx::Frame(f5.into_generic(), 4))
            .is_buffered());

        // f1 was the last emitted frame, f3 and f5 are buffered:
        assert_eq!(ordering.next_expected_frame_idx, 1);
        assert_eq!(ordering.frame_buffer.len(), 2);
        assert!(ordering.maybe_get_next_frame().is_none());

        // now, we push in f2, which is directly emitted, because its frame id
        // matches the currently expected id (f1 + 1):
        assert!(ordering
            .handle_frame(FrameWithIdx::Frame(f2.into_generic(), 1))
            .is_frame());
        assert_eq!(ordering.next_expected_frame_idx, 2);
        assert_eq!(ordering.frame_buffer.len(), 2);

        // now, we push in f4, which is buffered:
        assert!(ordering
            .handle_frame(FrameWithIdx::Frame(f4.into_generic(), 3))
            .is_buffered());
        assert_eq!(ordering.next_expected_frame_idx, 2);
        assert_eq!(ordering.frame_buffer.len(), 3);

        // and we can now consume the buffered frames:
        assert!(match ordering.maybe_get_next_frame() {
            Some(frame) => {
                frame.get_idx() == 2
            }
            None => false,
        });
        assert_eq!(ordering.frame_buffer.len(), 2);
        assert!(match ordering.maybe_get_next_frame() {
            Some(frame) => {
                frame.get_idx() == 3
            }
            None => false,
        });
        assert_eq!(ordering.frame_buffer.len(), 1);
        assert!(match ordering.maybe_get_next_frame() {
            Some(frame) => {
                frame.get_idx() == 4
            }
            None => false,
        });
        assert_eq!(ordering.frame_buffer.len(), 0);
        assert!(ordering.maybe_get_next_frame().is_none());
    }

    #[test]
    fn test_drop_duplicates() {
        let socket_dir = tempdir().unwrap();
        let socket_as_path = socket_dir.into_path().join("test.socket");
        let mut ordering: FrameOrdering = FrameOrdering::new(0);

        let mut ssa = SharedSlabAllocator::new(
            10,
            K2ISFrame::FRAME_HEIGHT * K2ISFrame::FRAME_WIDTH * std::mem::size_of::<u16>(),
            false,
            &socket_as_path,
        )
        .expect("create SHM area for testing");

        let f1w: K2ISFrameForWriting = K2ISFrameForWriting::empty(1, &mut ssa, 0);
        let f1 = f1w.writing_done(&mut ssa);
        let f2w: K2ISFrameForWriting = K2ISFrameForWriting::empty(2, &mut ssa, 0);
        let f2 = f2w.writing_done(&mut ssa);
        let f3w: K2ISFrameForWriting = K2ISFrameForWriting::empty(3, &mut ssa, 0);
        let f3 = f3w.writing_done(&mut ssa);
        let f4w: K2ISFrameForWriting = K2ISFrameForWriting::empty(4, &mut ssa, 0);
        let f4 = f4w.writing_done(&mut ssa);

        assert!(ordering
            .handle_frame(FrameWithIdx::DroppedFrame(f1.into_generic(), 0))
            .is_frame());
        assert!(ordering
            .handle_frame(FrameWithIdx::Frame(f2.into_generic(), 1))
            .is_frame());
        assert!(ordering
            .handle_frame(FrameWithIdx::DroppedFrame(f3.into_generic(), 0))
            .is_dropped()); // note duplicate index!
        assert!(ordering
            .handle_frame(FrameWithIdx::Frame(f4.into_generic(), 2))
            .is_frame());
    }
}
