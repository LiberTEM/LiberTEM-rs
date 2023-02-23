use std::slice::{from_raw_parts, from_raw_parts_mut};

/// A stack as an array that is stored somewhere in (shared) memory
pub struct FreeStack {
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

#[cfg(test)]
mod test {
    use memmap2::MmapOptions;

    use crate::freestack::FreeStack;

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
}
