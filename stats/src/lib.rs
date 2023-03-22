use log::info;

pub trait GetStats {
    /// Useful payload size of the whole item/stack in bytes
    fn payload_size(&self) -> usize;

    /// Whole slot size in bytes
    fn slot_size(&self) -> usize;

    /// The largest "frame" in this item in bytes
    fn max_frame_size(&self, old_max: usize) -> usize;

    /// The smallest "frame" in this item in bytes
    fn min_frame_size(&self, old_min: usize) -> usize;

    /// The number of actual "frames" in this item
    fn num_frames(&self) -> usize;
}

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

    pub fn reset(&mut self) {
        self.payload_size_sum = 0;
        self.slots_size_sum = 0;
        self.frame_size_max = 0;
        self.frame_size_min = usize::MAX;
        self.num_frames = 0;
        self.split_count = 0;
    }

    pub fn count_stats_item<S: GetStats>(&mut self, stats_item: &S) {
        self.payload_size_sum += stats_item.payload_size();
        self.slots_size_sum += stats_item.slot_size();
        self.frame_size_max = self
            .frame_size_max
            .max(stats_item.max_frame_size(self.frame_size_max));
        self.frame_size_min = self
            .frame_size_min
            .min(stats_item.min_frame_size(self.frame_size_min));
        self.num_frames += stats_item.num_frames();
    }

    pub fn count_split(&mut self) {
        self.split_count += 1;
    }

    pub fn log_stats(&self) {
        let efficiency = self.payload_size_sum as f32 / self.slots_size_sum as f32;
        info!(
            "Stats: frames seen: {}, total payload size: {}, total slot size used: {}, min size: {}, max size: {}, splits: {}, shm efficiency: {}",
            self.num_frames, self.payload_size_sum, self.slots_size_sum, self.frame_size_min, self.frame_size_max, self.split_count, efficiency,
        );
    }
}

impl Default for Stats {
    fn default() -> Self {
        Self::new()
    }
}
