use std::{
    path::Path,
    slice::{from_raw_parts, from_raw_parts_mut},
    sync::atomic::{AtomicU8, Ordering},
    thread::JoinHandle,
    time::Duration,
};

use crossbeam::channel::{bounded, Sender};
use raw_sync::locks::{LockImpl, LockInit, Mutex};
use serde::{Deserialize, Serialize};

use crate::{align_to, freestack::FreeStack, shm::Shm};

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
// unsafe impl Send for SlotForWriting {}

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
    pub os_handle: String,
    pub info: SlabInfo,
}

impl SHMHandle {}

#[derive(Debug, thiserror::Error)]
pub enum ShmError {
    #[error("no slot available")]
    NoSlotAvailable,
}

/// Additional information needed to re-crate a `SharedSlabAllocator` in a
/// different process
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct SlabInfo {
    pub num_slots: usize,
    pub slot_size: usize,
    pub total_size: usize, // including management overheads
}

pub struct SharedSlabAllocator {
    /// total number of slots
    num_slots: usize,

    /// size of each slot in bytes
    slot_size: usize,

    /// total size including overheads
    total_size: usize,

    shm: Shm,

    bg_thread: Option<(JoinHandle<()>, Sender<()>)>,
}

#[derive(thiserror::Error, Debug)]
pub enum SlabInitError {}

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

    fn get_alignment() -> usize {
        align_to(page_size::get(), 4096)
    }

    ///
    /// Create a new `SharedSlabAllocator` oject including a shared memory area.
    ///
    /// `shm_path` is a shm-backend-specific file location that will store something like
    /// a socket, handle or some additional meta-information. Make sure it points to a
    /// name inside a directory with restricted permissions!
    pub fn new(
        num_slots: usize,
        slot_size: usize,
        huge_pages: bool,
        shm_path: &Path,
    ) -> Result<Self, SlabInitError> {
        let free_list_size = std::mem::size_of::<usize>() * (1 + num_slots);
        let slot_sync_size = Self::SYNC_SLOT_SIZE * num_slots;
        let overhead: usize = align_to(
            free_list_size + slot_sync_size + Self::MUTEX_SIZE,
            Self::get_alignment(),
        );

        let slot_size = align_to(slot_size, Self::get_alignment());

        let total_size = align_to(
            slot_size * num_slots + overhead,
            2 * 1024 * 1024, // FIXME: support other huge page sizes
        );

        let slab_info: SlabInfo = SlabInfo {
            num_slots,
            slot_size,
            total_size,
        };
        let shm = Shm::new(huge_pages, shm_path, total_size, slab_info);

        Self::from_shm_and_slab_info(shm, slab_info, true)
    }

    pub fn from_shm_and_slab_info(
        shm: Shm,
        slab_info: SlabInfo,
        init_structures: bool,
    ) -> Result<Self, SlabInitError> {
        let ptr = shm.as_mut_ptr();
        let free_list_ptr = unsafe { ptr.offset(Self::MUTEX_SIZE.try_into().unwrap()) };

        let (_lock, bg_thread) = if init_structures {
            let (lock, used_size) = unsafe { Mutex::new(ptr, free_list_ptr).unwrap() };

            if used_size > Self::MUTEX_SIZE {
                panic!("Mutex size larger than expected!");
            }

            // populate free-list
            let mut free_list = Self::get_free_list(free_list_ptr, slab_info.num_slots);
            for i in 0..slab_info.num_slots {
                free_list.push(i);
            }

            // XXX on Windows, a Mutex is no longer usable if the last handle is
            // dropped, as opposed to POSIX, where it's enough to initialize
            // some memory region, where we can attach later. So here we do a
            // stupid hack and keep it alive in a background thread:
            let (init_chan_s, init_chan_r) = bounded::<()>(0);
            let (cleanup_chan_s, cleanup_chan_r) = bounded::<()>(0);

            // crimes to ship the pointers to the bg thread:
            #[derive(Clone)]
            struct B {
                mutex_ptr: *mut u8,
                data_ptr: *mut u8,
            }
            unsafe impl Send for B {}
            let b: B = B {
                mutex_ptr: ptr,
                data_ptr: free_list_ptr,
            };
            let j = std::thread::spawn(move || {
                let b = b.clone();
                let _mtx = unsafe { Mutex::from_existing(b.mutex_ptr, b.data_ptr) };
                // we are done initializing...
                init_chan_s.send(()).unwrap();
                // so we just keep the mutex open and wait until we are cleaned up:
                cleanup_chan_r.recv().unwrap();
            });

            // wait for initialization of the thread:
            init_chan_r
                .recv_timeout(Duration::from_millis(100))
                .expect("background thread did not initialize");

            (lock, Some((j, cleanup_chan_s)))
        } else {
            let (lock, used_size) = unsafe { Mutex::from_existing(ptr, free_list_ptr).unwrap() };

            if used_size > Self::MUTEX_SIZE {
                panic!("Mutex size larger than expected!");
            }

            (lock, None)
        };

        Ok(Self {
            num_slots: slab_info.num_slots,
            slot_size: slab_info.slot_size,
            shm,
            total_size: slab_info.total_size,
            bg_thread,
        })
    }

    pub fn connect(handle_path: &str) -> Result<Self, SlabInitError> {
        let (shm, slab_info): (_, SlabInfo) = Shm::connect(handle_path);
        Self::from_shm_and_slab_info(shm, slab_info, false)
    }

    pub fn clone_and_connect(&self) -> Result<Self, SlabInitError> {
        let handle = self.get_handle();
        Self::connect(&handle.os_handle)
    }

    fn get_mutex(&self) -> Box<dyn LockImpl> {
        let ptr = self.shm.as_mut_ptr();
        let free_list_ptr = unsafe { ptr.offset(Self::MUTEX_SIZE.try_into().unwrap()) };
        let (lock, _) = unsafe { Mutex::from_existing(ptr, free_list_ptr).unwrap() };
        lock
    }

    pub fn get_slab_info(&self) -> SlabInfo {
        SlabInfo {
            num_slots: self.num_slots,
            slot_size: self.slot_size,
            total_size: self.total_size,
        }
    }

    pub fn get_handle(&self) -> SHMHandle {
        SHMHandle {
            os_handle: self.shm.get_handle(),
            info: SlabInfo {
                num_slots: self.num_slots,
                slot_size: self.slot_size,
                total_size: self.total_size,
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

    pub fn try_get_mut(&mut self) -> Result<SlotForWriting, ShmError> {
        self.get_mut().ok_or(ShmError::NoSlotAvailable)
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

    /// Pointer to where the synchornization variable for slot `slot_idx` is stored
    fn get_slot_sync_ptr(&self, slot_idx: usize) -> *mut u8 {
        let offset = self.get_slot_sync_offset(slot_idx);
        let base_ptr = self.shm.as_mut_ptr();
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
            Self::get_alignment(),
        )
        .try_into()
        .unwrap();

        slot_zero + (slot_idx * self.slot_size) as isize
    }

    fn get_ptr_for_slot(&self, slot_idx: usize) -> *const u8 {
        let slot_offset = self.get_slot_offset(slot_idx);
        let base_ptr = self.shm.as_mut_ptr();
        unsafe { base_ptr.offset(slot_offset) }
    }

    fn get_mut_ptr_for_slot(&self, slot_idx: usize) -> *mut u8 {
        let slot_offset = self.get_slot_offset(slot_idx);
        let base_ptr = self.shm.as_mut_ptr();
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
        // trace!("SharedSlabAllocator::drop:\n{}", Backtrace::force_capture());

        // clean up our hack:
        if let Some((join, channel)) = self.bg_thread.take() {
            channel.send(()).unwrap();
            join.join().unwrap();
        }
    }
}

#[cfg(test)]
mod test {
    use std::{path::PathBuf, time::Duration};
    use tempfile::{tempdir, TempDir};

    use crate::{
        align_to,
        slab::{SharedSlabAllocator, SlotForWriting},
    };

    fn get_socket_path() -> (TempDir, PathBuf) {
        let socket_dir = tempdir().unwrap();
        let socket_as_path = socket_dir.path().join("stuff.socket");

        (socket_dir, socket_as_path)
    }

    #[test]
    fn test_simple_alloc_free() {
        let (_socket_dir, socket_as_path) = get_socket_path();

        let mut ssa = SharedSlabAllocator::new(4, 255, false, &socket_as_path).unwrap();
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
        let (_socket_dir, socket_as_path) = get_socket_path();

        let mut ssa = SharedSlabAllocator::new(4, 255, false, &socket_as_path).unwrap();
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
        let mut ssa2 = SharedSlabAllocator::connect(&handle.os_handle)
            .expect("should be able to connect to existing shared memory");

        let slotr = ssa2.get(slotw.slot_idx);
        for i in 0..255u8 {
            assert_eq!(slotr.as_slice()[i as usize], i);
        }
        ssa2.free_idx(slotw.slot_idx);
    }

    #[test]
    fn test_clone_and_connect() {
        let (_socket_dir, socket_as_path) = get_socket_path();

        let mut ssa = SharedSlabAllocator::new(4, 255, false, &socket_as_path).unwrap();
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
    fn test_threaded_connect() {
        let (_socket_dir, socket_as_path) = get_socket_path();

        let mut ssa = SharedSlabAllocator::new(4, 255, false, &socket_as_path).unwrap();
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
                let mut ssa2 = SharedSlabAllocator::connect(&handle.os_handle)
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
            init_r.recv().unwrap();

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
}
