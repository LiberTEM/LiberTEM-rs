pub(crate) mod freestack;
pub mod slab;

pub use slab::{SHMHandle, SharedSlabAllocator, SlabInfo, Slot, SlotForWriting, SlotInfo};

// for testing:
// #[cfg_attr(target_os = "linux", path = "backend_memfd.rs")]
// #[cfg_attr(target_os = "linux", path = "backend_shm.rs")]

#[cfg_attr(target_os = "linux", path = "backend_memfd.rs")]
#[cfg_attr(not(target_os = "linux"), path = "backend_shm.rs")]
pub mod shm;

/// Makes `size` a multiple of `alignment`, rounding up.
fn align_to(size: usize, alignment: usize) -> usize {
    // while waiting for div_ceil to be stable, have this monstrosity:
    let div = size / alignment;
    let rem = size % alignment;

    if rem > 0 {
        alignment * (div + 1)
    } else {
        alignment * div
    }
}

#[cfg(test)]
mod test {
    use crate::align_to;

    #[test]
    fn test_align() {
        assert_eq!(align_to(0, 4096), 0);
        assert_eq!(align_to(1, 4096), 4096);
        assert_eq!(align_to(4096, 4096), 4096);
        assert_eq!(align_to(4097, 4096), 2 * 4096);
    }
}
