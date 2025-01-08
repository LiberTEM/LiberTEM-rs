use std::fs::OpenOptions;

use memmap2::{MmapMut, MmapOptions};
use ndarray_npy::{write_zeroed_npy, ViewMutNpyExt};

pub fn main() {
    let filename = "/cachedata/alex/foo.npy";
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(filename)
        .unwrap();

    let mut mmap = unsafe { MmapMut::map_mut(&file).unwrap() };

    let shape = (4000, 1860, 2048);
    let mut view = ndarray::ArrayViewMut3::<u16>::view_mut_npy(&mut mmap).unwrap();

    println!("sum={}", view.sum());
}
