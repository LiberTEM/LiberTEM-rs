use stats::GetStats;

use crate::chunk_stack::ChunkStackHandle;

impl GetStats for ChunkStackHandle {
    fn payload_size(&self) -> usize {
        self.payload_size()
    }

    fn slot_size(&self) -> usize {
        self.slot_size()
    }

    fn max_frame_size(&self, old_max: usize) -> usize {
        self.get_layout()
            .iter()
            .max_by_key(|fm| fm.data_length_bytes)
            .map_or(old_max, |fm| fm.data_length_bytes)
    }

    fn min_frame_size(&self, old_min: usize) -> usize {
        self.get_layout()
            .iter()
            .min_by_key(|fm| fm.data_length_bytes)
            .map_or(old_min, |fm| fm.data_length_bytes)
    }

    fn num_frames(&self) -> usize {
        self.len() as usize
    }
}
