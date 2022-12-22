use crate::common::DConfig;
use crate::common::DImage;
use crate::common::DImageD;
use ipc_test::SharedSlabAllocator;
use ipc_test::SlotForWriting;
use ipc_test::SlotInfo;

#[derive(PartialEq, Eq, Clone)]
pub struct FrameMeta {
    pub dimage: DImage,
    pub dimaged: DImageD,
    pub dconfig: DConfig,
    pub data_length_bytes: usize,
}

pub struct FrameStackForWriting {
    slot: SlotForWriting,
    meta: Vec<FrameMeta>,
    capacity: usize,
    bytes_per_frame: usize,
}

impl FrameStackForWriting {
    pub fn new(slot: SlotForWriting, capacity: usize, bytes_per_frame: usize) -> Self {
        FrameStackForWriting {
            slot,
            capacity, // number of frames
            bytes_per_frame,
            meta: Vec::with_capacity(capacity),
        }
    }

    pub fn len(&self) -> usize {
        self.meta.len()
    }

    pub fn is_full(&self) -> bool {
        self.len() >= self.capacity
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// take frame metadata and put it into our list
    /// and copy data into shared the right shm position
    /// (we can't have zero-copy directly into shm with 0mq sadly, as far as I
    /// can tell)
    /// TODO: in a general library, this should also have a zero-copy api available!
    pub fn frame_done(
        &mut self,
        dimage: DImage,
        dimaged: DImageD,
        dconfig: DConfig,
        data: &[u8],
    ) -> FrameMeta {
        let idx = self.len();
        // FIXME: alignment per frame?
        let start = idx * self.bytes_per_frame;
        let stop = start + self.bytes_per_frame;
        let dest = &mut self.slot.as_slice_mut()[start..stop];
        // FIXME: return error on slice length mismatch, don't panic
        dest[..data.len()].copy_from_slice(data);
        let meta = FrameMeta {
            dimage,
            dimaged,
            dconfig,
            data_length_bytes: data.len(),
        };
        self.meta.push(meta);
        meta.clone()
    }

    pub fn writing_done(self, shm: &mut SharedSlabAllocator) -> FrameStackHandle {
        let slot_info = shm.writing_done(self.slot);

        FrameStackHandle {
            slot: slot_info,
            meta: self.meta,
            bytes_per_frame: self.bytes_per_frame,
        }
    }
}

/// serializable handle for a stack of frames that live in shm
#[derive(PartialEq, Eq)]
pub struct FrameStackHandle {
    slot: SlotInfo,
    meta: Vec<FrameMeta>,
    bytes_per_frame: usize,
}

struct FrameStackView<'a> {
    handle: &'a FrameStackHandle,
}

impl<'a> FrameStackView<'a> {
    fn from_handle(handle: &FrameStackHandle, shm: &SharedSlabAllocator) -> Self {
        todo!();
    }

    fn get_frame_view(&self, frame_idx: usize) {
        todo!();
    }
}

#[cfg(test)]
mod test {
    use ipc_test::SharedSlabAllocator;

    use super::{FrameStackForWriting, FrameStackView};

    #[test]
    fn test_frame_stack() {
        let mut shm = SharedSlabAllocator::new(1, 4096, false).unwrap();
        let slot = shm.get_mut().expect("get a free shm slot");
        let mut fs = FrameStackForWriting::new(slot, 16, 1);
        let dimage = crate::common::DImage {
            htype: "".to_string(),
            series: 1,
            frame: 1,
            hash: "".to_string(),
        };
        let dimaged = crate::common::DImageD {
            htype: "".to_string(),
            shape: vec![512, 512],
            type_: crate::common::PixelType::Uint16,
            encoding: ">bslz4".to_string(),
        };
        let dconfig = crate::common::DConfig {
            htype: "".to_string(),
            start_time: 0,
            stop_time: 0,
            real_time: 0,
        };
        fs.frame_done(dimage, dimaged, dconfig, &[42]);

        let fs_handle = fs.writing_done(&mut shm);

        // then, in another thread/process far far away...
        let view = FrameStackView::from_handle(&fs_handle, &shm);
        view.get_frame_view(0) // .as_slice().get_memoryview() --> to Python
    }
}
