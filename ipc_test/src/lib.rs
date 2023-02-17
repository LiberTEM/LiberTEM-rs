use std::{
    fs::File,
    slice::{from_raw_parts, from_raw_parts_mut},
    sync::atomic::{AtomicU8, Ordering}, str::FromStr,
};
use std::backtrace::Backtrace;

use memmap2::{MmapOptions, MmapRaw};
use shared_memory::{ShmemConf, Shmem};

use anyhow::{Context, Result};
use raw_sync::locks::{LockImpl, LockInit, Mutex};
use serde::{Deserialize, Serialize};

/// A handle for reading from a shared memory slot
pub struct Slot {
    pub ptr: *const u8,
    pub size: usize,
    pub slot_idx: usize,
}

impl Slot {
    pub fn as_slice(&self) -> &[u8] {
        unsafe { from_raw_parts(self.ptr, self.size) }
    }
}

/// A handle for writing to a shared memory slot
pub struct SlotForWriting {
    pub ptr: *mut u8,
    pub size: usize,
    slot_idx: usize, // private while writing to a slot
}

// Safety: if you have a `SlotForWriting`, you have exclusive access
// for writing into the slot this "token" points at
unsafe impl Send for SlotForWriting {}

pub struct SlotIdx {
    pub ptr: *mut u8,
    pub size: usize,
    pub slot_idx: usize,
}

#[derive(Serialize, Deserialize, PartialEq, Eq, Debug)]
pub struct SlotInfo {
    pub size: usize,
    pub slot_idx: usize,
}

impl SlotForWriting {
    pub fn as_slice(&self) -> &[u8] {
        unsafe { from_raw_parts(self.ptr, self.size) }
    }

    pub fn as_slice_mut(&mut self) -> &mut [u8] {
        unsafe { from_raw_parts_mut(self.ptr, self.size) }
    }
}

///
/// A handle for the whole shared memory region
///
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SHMHandle {
    pub id: String,
    pub info: SHMInfo,
}

impl SHMHandle {}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct SHMInfo {
    pub num_slots: usize,
    pub slot_size: usize,
    pub total_len: usize,
}

pub struct SharedSlabAllocator {
    /// total number of slots
    num_slots: usize,

    /// size of each slot in bytes
    slot_size: usize,

    page_size: usize,
    total_len: usize,
    shmem: Shmem
}

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

/// A stack as an array that is stored somewhere in (shared) memory
struct FreeStack {
    ///
    /// Pointer to the underlying memory
    ///
    /// The layout is as follows:
    ///
    ///
    /// +------------+--------+-----+--------+
    /// | top: usize | elem 0 | ... | elem N |
    /// +------------+--------+-----+--------+
    ///
    /// All elements are `usize`, too.
    ///
    base_ptr: *mut u8,

    /// Total size of memory that we have available
    size_in_bytes: usize,
}

impl FreeStack {
    pub fn new(base_ptr: *mut u8, size_in_bytes: usize) -> Self {
        FreeStack {
            base_ptr,
            size_in_bytes,
        }
    }

    pub fn get_stack_idx(&self) -> usize {
        let slice = unsafe { from_raw_parts(self.base_ptr, std::mem::size_of::<usize>()) };
        *bytemuck::from_bytes(slice)
    }

    fn set_stack_idx(&mut self, idx: usize) {
        let slice = unsafe { from_raw_parts_mut(self.base_ptr, std::mem::size_of::<usize>()) };
        let slice_usize: &mut [usize] = bytemuck::cast_slice_mut(slice);
        slice_usize[0] = idx;
    }

    fn get_elems(&self) -> &[usize] {
        let slice = unsafe {
            let elems_ptr = self
                .base_ptr
                .offset(std::mem::size_of::<usize>().try_into().unwrap());
            from_raw_parts(elems_ptr, self.size_in_bytes - std::mem::size_of::<usize>())
        };
        bytemuck::cast_slice(slice)
    }

    fn get_elems_mut(&mut self) -> &mut [usize] {
        let slice = unsafe {
            let elems_ptr = self
                .base_ptr
                .offset(std::mem::size_of::<usize>().try_into().unwrap());
            from_raw_parts_mut(elems_ptr, self.size_in_bytes - std::mem::size_of::<usize>())
        };
        bytemuck::cast_slice_mut(slice)
    }

    pub fn pop(&mut self) -> Option<usize> {
        let idx = self.get_stack_idx();

        if idx == 0 {
            None
        } else {
            let elems = self.get_elems();
            let elem = elems[idx - 1];

            self.set_stack_idx(idx - 1);

            Some(elem)
        }
    }

    pub fn push(&mut self, new_elem: usize) {
        let idx = self.get_stack_idx();
        let elems = self.get_elems_mut();

        elems[idx] = new_elem;

        self.set_stack_idx(idx + 1);
    }
}

///
/// Single-producer multiple consumer communication via shared memory
///
/// ## Layout of the shared memory:
///
/// +-----------------+---------------------+----------------------+--------+--------+-----+----------+
/// | Free-list Mutex | Free-list of size N | N * sync slots (64B) | Slot 0 | Slot 1 | ... | Slot N-1 |
/// +-----------------+---------------------+----------------------+--------+--------+-----+----------+
/// ^                  ^                     ^                      ^- align(64 + (N + 1)*sizeof(usize) + 64 * N)
/// 0                  64                    64 + (N + 1)*sizeof(usize)
///
/// The free-list itself is actually a stack, made up of an `usize` denoting the
/// current number of items on the stack, and up to `N` indices, stored as `usize`s.
///
/// Allocated slots will be aligned, and thus their size can be larger (but
/// never smaller) than requested.
///
/// ## Safety
///
/// This API is low level and needs to be used with care. For example, a
/// misbehaving consumer can request (read-only) access to slots that are currently being
/// written to by another thread/process.
///
/// The `SharedSlabAllocator` instance of the producer must live at least until
/// the consumer has connected, otherwise the `File` and memory map will be closed
/// and connection from the consumer will fail.
///
impl SharedSlabAllocator {
    const MUTEX_SIZE: usize = 64;
    const SYNC_SLOT_SIZE: usize = 64;
    const ALIGNMENT: usize = 4096; // FIXME: does this need to be a runtime thing?

    pub fn new(num_slots: usize, slot_size: usize) -> Result<Self> {
        let free_list_size = std::mem::size_of::<usize>() * (1 + num_slots);
        let slot_sync_size = Self::SYNC_SLOT_SIZE * num_slots;
        let overhead: usize = align_to(
            free_list_size + slot_sync_size + Self::MUTEX_SIZE,
            Self::ALIGNMENT,
        );

        let slot_size = align_to(slot_size, Self::ALIGNMENT);

        // total length needs to be aligned to huge page size:
        let total_len = align_to(
            slot_size * num_slots + overhead,
            4 * 1024, // FIXME: support other page sizes
        );

        let shmem = ShmemConf::new().size(total_len).create().unwrap();

        let handle = SHMHandle{
            id: String::from(shmem.get_os_id()),
            info: SHMInfo {
                num_slots,
                slot_size,
                total_len,
            },
        };

        let mut slab = SharedSlabAllocator {
            page_size: page_size::get(),
            num_slots,
            slot_size: handle.info.slot_size,
            shmem,
            total_len,
        };
        slab.init_structures();
        Ok(slab)
    }

    pub fn clone_and_connect(&self) -> Result<Self> {
        let handle = self.get_handle();
        Self::connect(handle)
    }

    fn init_structures(&mut self) -> Box<dyn LockImpl> {
        let ptr = self.shmem.as_ptr();
        let free_list_ptr = unsafe { ptr.offset(Self::MUTEX_SIZE.try_into().unwrap()) };
        let (lock, used_size) = unsafe { Mutex::new(ptr, free_list_ptr).unwrap() };

        if used_size > Self::MUTEX_SIZE {
            panic!("Mutex size larger than expected!");
        }

        // populate free-list
        let mut free_list = Self::get_free_list(free_list_ptr, self.num_slots);
        for i in 0..self.num_slots {
            free_list.push(i);
        }

        lock
    }

    fn from_handle(handle: SHMHandle, init_structures: bool) -> Result<Self> {
        let id = handle.id;
        let num_slots = handle.info.num_slots;
        let total_len = handle.info.total_len;
        let shmem = ShmemConf::new().os_id(id).size(total_len).open().unwrap();
        let mut slab = SharedSlabAllocator {
            page_size: page_size::get(),
            num_slots,
            slot_size: handle.info.slot_size,
            shmem,
            total_len,
        };
        if init_structures {
            slab.init_structures();
        }
        Ok(slab)
    }
    
    pub fn connect(handle: SHMHandle) -> Result<Self> {
        Self::from_handle(handle, false)
    }

    fn get_mutex(&self) -> Box<dyn LockImpl> {
        let ptr = self.shmem.as_ptr();
        let free_list_ptr = unsafe { ptr.offset(Self::MUTEX_SIZE.try_into().unwrap()) };
        let (lock, _) = unsafe { Mutex::from_existing(ptr, free_list_ptr).unwrap() };
        lock
    }

    pub fn get_handle(&self) -> SHMHandle {
        SHMHandle {
            id: String::from(self.shmem.get_os_id()),
            info: SHMInfo {
                num_slots: self.num_slots,
                slot_size: self.slot_size,
                total_len: self.total_len,
            },
        }
    }

    /// Get a `SlotForWriting`, which contains a mutable pointer to a shared
    /// memory slot. Once you are done writing, you can exchange this to a
    /// `SlotInfo` struct using `writing_done`, which can then be sent to
    /// a consumer.
    pub fn get_mut(&mut self) -> Option<SlotForWriting> {
        let slot_idx: usize = self.pop_free_slot_idx()?;
        Some(SlotForWriting {
            ptr: self.get_mut_ptr_for_slot(slot_idx),
            slot_idx,
            size: self.slot_size,
        })
    }

    /// Exchange the `SlotForWriting` token into
    /// a `SlotInfo` that can be sent to readers
    /// which can not be used to write anymore
    pub fn writing_done(&mut self, slot: SlotForWriting) -> SlotInfo {
        self.release(slot.slot_idx);
        SlotInfo {
            size: slot.size,
            slot_idx: slot.slot_idx,
        }
    }

    pub fn num_slots_free(&self) -> usize {
        let mutex = self.get_mutex();
        let guard = mutex.lock().unwrap();
        let stack = Self::get_free_list(*guard, self.num_slots);
        stack.get_stack_idx()
    }

    pub fn num_slots_total(&self) -> usize {
        self.num_slots
    }

    pub fn get_slot_size(&self) -> usize {
        self.slot_size
    }

    pub fn get(&self, slot_idx: usize) -> Slot {
        self.acquire(slot_idx);
        Slot {
            ptr: self.get_ptr_for_slot(slot_idx),
            slot_idx,
            size: self.slot_size,
        }
    }

    pub fn free_idx(&mut self, slot_idx: usize) {
        let mutex = self.get_mutex();
        let guard = mutex.lock().unwrap();
        let mut stack = Self::get_free_list(*guard, self.num_slots);
        stack.push(slot_idx);
    }

    ///
    /// "When coupled with a load, if the loaded value was written by a store
    /// operation with Release (or stronger) ordering, then all subsequent
    /// operations become ordered after that store. In particular, all
    /// subsequent loads will see data written before the store."
    ///
    fn acquire(&self, slot_idx: usize) {
        let ptr = self.get_slot_sync_ptr(slot_idx);
        let atomic = unsafe { &*(ptr as *const AtomicU8) };
        atomic.load(Ordering::Acquire);
    }

    ///
    /// "When coupled with a store, all previous operations become ordered before
    /// any load of this value with Acquire (or stronger) ordering. In
    /// particular, all previous writes become visible to all threads that
    /// perform an Acquire (or stronger) load of this value."
    ///
    fn release(&mut self, slot_idx: usize) {
        let ptr = self.get_slot_sync_ptr(slot_idx);
        let atomic = unsafe { &*(ptr as *const AtomicU8) };
        atomic.store(0, Ordering::Release);
    }

    fn get_slot_sync_ptr(&self, slot_idx: usize) -> *mut u8 {
        let offset = self.get_slot_sync_offset(slot_idx);
        let base_ptr = self.shmem.as_ptr();
        unsafe { base_ptr.offset(offset) }
    }

    /// Offset where the synchornization variable for slot `slot_idx` is stored
    fn get_slot_sync_offset(&self, slot_idx: usize) -> isize {
        let free_list_size = std::mem::size_of::<usize>() * (1 + self.num_slots);
        let sync_zero: isize = (free_list_size + Self::MUTEX_SIZE).try_into().unwrap();
        sync_zero + (slot_idx * Self::SYNC_SLOT_SIZE) as isize
    }

    /// Offset where the payload data for slot `slot_idx` is stored
    fn get_slot_offset(&self, slot_idx: usize) -> isize {
        let free_list_size = std::mem::size_of::<usize>() * (1 + self.num_slots);
        let slot_sync_size = Self::SYNC_SLOT_SIZE * self.num_slots;
        let slot_zero: isize = align_to(
            free_list_size + slot_sync_size + Self::MUTEX_SIZE,
            Self::ALIGNMENT,
        )
        .try_into()
        .unwrap();

        slot_zero + (slot_idx * self.slot_size) as isize
    }

    fn get_ptr_for_slot(&self, slot_idx: usize) -> *const u8 {
        let slot_offset = self.get_slot_offset(slot_idx);
        let base_ptr = self.shmem.as_ptr();
        unsafe { base_ptr.offset(slot_offset) }
    }

    fn get_mut_ptr_for_slot(&self, slot_idx: usize) -> *mut u8 {
        let slot_offset = self.get_slot_offset(slot_idx);
        let base_ptr = self.shmem.as_ptr();
        unsafe { base_ptr.offset(slot_offset) }
    }

    fn get_free_list(base_ptr: *mut u8, num_slots: usize) -> FreeStack {
        let stack_ptr =
            unsafe { base_ptr.offset(std::mem::size_of::<usize>().try_into().unwrap()) };
        let free_list_size = std::mem::size_of::<usize>() * (1 + num_slots);
        FreeStack::new(stack_ptr, free_list_size)
    }

    fn pop_free_slot_idx(&mut self) -> Option<usize> {
        let mutex = self.get_mutex();
        let guard = mutex.lock().unwrap();
        let mut stack = Self::get_free_list(*guard, self.num_slots);
        stack.pop()
    }
}

impl Drop for SharedSlabAllocator {
    fn drop(&mut self) {
        // println!("SharedSlabAllocator::drop: {}", Backtrace::force_capture())
    }
}

#[cfg(test)]
mod test {
    use std::time::Duration;

    use memmap2::MmapOptions;
    use shared_memory::ShmemConf;

    use crate::{align_to, FreeStack, SharedSlabAllocator, SlotForWriting};

    #[test]
    fn test_align() {
        assert_eq!(align_to(0, 4096), 0);
        assert_eq!(align_to(1, 4096), 4096);
        assert_eq!(align_to(4096, 4096), 4096);
        assert_eq!(align_to(4097, 4096), 2 * 4096);
    }

    #[test]
    fn test_stack_happy() {
        // raw memory length in bytes
        let raw_length = 4096;

        // get some "raw" memory
        let mut mmap = MmapOptions::new()
            .len(raw_length)
            .map_anon()
            .expect("could not mmap");

        let base_ptr = mmap.as_mut_ptr();
        let mut s = FreeStack::new(base_ptr, raw_length);

        assert!(s.pop().is_none());

        s.push(1);
        s.push(2);
        s.push(3);

        assert_eq!(s.pop(), Some(3));
        assert_eq!(s.pop(), Some(2));
        assert_eq!(s.pop(), Some(1));
        assert!(s.pop().is_none());
    }

    #[test]
    #[should_panic(expected = "index out of bounds: the len is 3 but the index is 3")]
    fn test_stack_unhappy() {
        // raw memory length in bytes
        let raw_length = std::mem::size_of::<usize>() * 4;

        // get some "raw" memory
        let mut mmap = MmapOptions::new()
            .len(raw_length)
            .map_anon()
            .expect("could not mmap");

        let base_ptr = mmap.as_mut_ptr();
        let mut s = FreeStack::new(base_ptr, raw_length);

        assert!(s.pop().is_none());

        s.push(1);
        s.push(2);
        s.push(3);
        s.push(4); // panic! out of bounds.
    }

    #[test]
    fn test_simple_alloc_free() {
        let mut ssa = SharedSlabAllocator::new(4, 255).unwrap();
        let mut slotw: SlotForWriting = ssa.get_mut().unwrap();
        assert_eq!(slotw.slot_idx, 3); // we get the slot from the top
        let slot_size = slotw.size;

        assert_eq!(slotw.as_slice_mut().len(), align_to(255, 4096));
        assert_eq!(slot_size, align_to(255, 4096));

        for (item, idx) in slotw.as_slice_mut().iter_mut().zip(0..slot_size) {
            *item = idx as u8;
        }

        for i in 0..255 {
            assert_eq!(slotw.as_slice_mut()[i], i as u8);
        }

        let slotr = ssa.get(slotw.slot_idx);
        for i in 0..255u8 {
            assert_eq!(slotr.as_slice()[i as usize], i);
        }
        ssa.free_idx(slotw.slot_idx);
    }

    #[test]
    fn test_connect() {
        let mut ssa = SharedSlabAllocator::new(4, 255).unwrap();
        let mut slotw: SlotForWriting = ssa.get_mut().unwrap();
        assert_eq!(slotw.slot_idx, 3); // we get the slot from the top
        let slot_size = slotw.size;

        assert_eq!(slotw.as_slice_mut().len(), align_to(255, 4096));
        assert_eq!(slot_size, align_to(255, 4096));

        for (item, idx) in slotw.as_slice_mut().iter_mut().zip(0..slot_size) {
            *item = idx as u8;
        }

        let handle = ssa.get_handle();
        println!("handle: {handle:?}");
        let mut ssa2 = SharedSlabAllocator::connect(handle)
            .expect("should be able to connect to existing shared memory");

        let slotr = ssa2.get(slotw.slot_idx);
        for i in 0..255u8 {
            assert_eq!(slotr.as_slice()[i as usize], i);
        }
        ssa2.free_idx(slotw.slot_idx);
    }

    #[test]
    fn test_clone_and_connect() {
        let mut ssa = SharedSlabAllocator::new(4, 255).unwrap();
        let mut slotw: SlotForWriting = ssa.get_mut().unwrap();
        assert_eq!(slotw.slot_idx, 3); // we get the slot from the top
        let slot_size = slotw.size;

        assert_eq!(slotw.as_slice_mut().len(), align_to(255, 4096));
        assert_eq!(slot_size, align_to(255, 4096));

        for (item, idx) in slotw.as_slice_mut().iter_mut().zip(0..slot_size) {
            *item = idx as u8;
        }

        let handle = ssa.get_handle();
        println!("handle: {handle:?}");
        let mut ssa2 = ssa
            .clone_and_connect()
            .expect("should be able to connect to existing shared memory");

        let slotr = ssa2.get(slotw.slot_idx);
        for i in 0..255u8 {
            assert_eq!(slotr.as_slice()[i as usize], i);
        }
        ssa2.free_idx(slotw.slot_idx);
    }

    #[test]
    fn test_connect_threaded() {
        let mut ssa = SharedSlabAllocator::new(4, 255).unwrap();
        let handle = ssa.get_handle();

        // Channels for sending the slot index from the producer to the child thread:
        let (sender, receiver) = crossbeam::channel::unbounded::<usize>();

        // Back channel to signal that we are done initializing in the child thread:
        let (init_s, init_r) = crossbeam::channel::unbounded::<()>();

        // One more to handle any crashes of the child thread gracefully:
        let (done_s, done_r) = crossbeam::channel::unbounded::<()>();

        crossbeam::scope(|s| {
            s.spawn(|_| {
                // "connect" to shared memory:
                let mut ssa2 = SharedSlabAllocator::connect(handle)
                    .expect("should be able to connect to existing shared memory");

                // just for testing that our test is robust-ish:
                //panic!("paaaniccccc!");

                // Signal the main thread that we have connected successfully
                // and allow it to continue running:
                init_s.send(()).unwrap();

                // wait for a message:
                let idx = receiver.recv().unwrap();
                let slotr = ssa2.get(idx);
                for i in 0..255u8 {
                    assert_eq!(slotr.as_slice()[i as usize], i);
                }

                // We are done with the data in the slot, make it available
                // to the producer:
                ssa2.free_idx(idx);

                // for keeping the test robust: signal main thread we are done
                done_s.send(()).unwrap();
            });

            // As we are sharing the fd with the child thread,
            // we need to wait until the child has mmapped the file,
            // otherwise we can run ahead and close the file on this thread,
            // before the child has had the chance to mmap.
            init_r.recv_timeout(Duration::from_millis(200)).unwrap();


            // write into shm:
            let mut slotw: SlotForWriting = ssa.get_mut().unwrap();
            for (item, idx) in slotw.as_slice_mut().iter_mut().zip(0..255) {
                *item = idx as u8;
            }

            // Let the child thread know
            sender.send(slotw.slot_idx).unwrap();

            done_r.recv_timeout(Duration::from_millis(100)).unwrap();
        })
        .unwrap();
    }

    #[test]
    fn test_shmem_assumptions() {
        fn from_handle(id: &str) {
            ShmemConf::new().os_id(id).size(69420).open().unwrap();
        }
        let id = {
            let shmem = ShmemConf::new().size(69420).create().unwrap();
            // works: the original still is open
            from_handle(shmem.get_os_id());
            shmem.get_os_id().to_string()
        };
        // fails, the file was removed already
        from_handle(&id);
    }
}
