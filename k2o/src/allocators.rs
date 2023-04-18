use std::alloc::{Allocator, Global, Layout};

pub struct PageAlignedAllocator;

unsafe impl Allocator for PageAlignedAllocator {
    fn allocate(&self, layout: Layout) -> Result<std::ptr::NonNull<[u8]>, std::alloc::AllocError> {
        Global.allocate(layout.align_to(4096).unwrap())
    }

    unsafe fn deallocate(&self, ptr: std::ptr::NonNull<u8>, layout: Layout) {
        Global.deallocate(ptr, layout)
    }
}
