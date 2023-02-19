use log::debug;

use crate::chunk_stack::ChunkStackHandle;

pub struct Stats {
    /// total number of bytes (compressed) that have flown through the system
    payload_size_sum: usize,

    /// maximum size of compressed frames seen
    frame_size_max: usize,

    /// minimum size of compressed frames seen
    frame_size_min: usize,

    /// sum of the size of the slots used
    slots_size_sum: usize,

    /// number of frames seen
    num_frames: usize,

    /// number of times a frame stack was split
    split_count: usize,
}

impl Stats {
    pub fn new() -> Self {
        Self {
            payload_size_sum: 0,
            slots_size_sum: 0,
            frame_size_max: 0,
            frame_size_min: usize::MAX,
            num_frames: 0,
            split_count: 0,
        }
    }

    pub fn count_chunk_stack(&mut self, chunk_stack: &ChunkStackHandle) {
        self.payload_size_sum += chunk_stack.payload_size();
        self.slots_size_sum += chunk_stack.slot_size();
        self.frame_size_max = self.frame_size_max.max(
            chunk_stack
                .get_layout()
                .iter()
                .max_by_key(|fm| fm.data_length_bytes)
                .map_or(self.frame_size_max, |fm| fm.data_length_bytes),
        );
        self.frame_size_min = self.frame_size_min.min(
            chunk_stack
                .get_layout()
                .iter()
                .min_by_key(|fm| fm.data_length_bytes)
                .map_or(self.frame_size_min, |fm| fm.data_length_bytes),
        );
        self.num_frames += chunk_stack.len() as usize;
    }

    pub fn count_split(&mut self) {
        self.split_count += 1;
    }

    pub fn log_stats(&self) {
        let efficiency = self.payload_size_sum as f32 / self.slots_size_sum as f32;
        debug!(
            "Stats: frames seen: {}, total payload size: {}, total slot size used: {}, min chunk size: {}, max chunk size: {}, splits: {}, shm efficiency: {}",
            self.num_frames, self.payload_size_sum, self.slots_size_sum, self.frame_size_min, self.frame_size_max, self.split_count, efficiency,
        );
    }
}

impl Default for Stats {
    fn default() -> Self {
        Self::new()
    }
}
