use std::{fs::File, os::unix::prelude::AsRawFd};

use ndarray::Ix;
use nix::{
    fcntl::{fallocate, FallocateFlags},
    sched::CpuSet,
    unistd::Pid,
};

use crate::{
    block::K2Block,
    decode::{decode_packet_size, HEADER_SIZE},
    net::create_mcast_socket,
};

/// pin the current thread to a specific CPU
pub fn set_cpu_affinity(cpu_id: usize) {
    let mut cpu_set = CpuSet::new();
    cpu_set.set(cpu_id).expect("could not set CPU affinity!");
    nix::sched::sched_setaffinity(Pid::from_raw(0), &cpu_set).expect("could not set CPU affinity!");
}

pub const CPU_AFF_DECODE_START: usize = 10;
pub const CPU_AFF_WRITER: usize = 18;
pub const CPU_AFF_ASSEMBLY: usize = 19;

/// Operation modes for `preallocate`
#[derive(PartialEq)]
pub enum AllocateMode {
    AllocateOnly,
    ZeroFill,
}

///
/// Create a new file and allocate space for `num_frames` frames of size `bytes_per_frame`.
///
pub fn preallocate(filename: &str, bytes_per_frame: usize, num_frames: usize, mode: AllocateMode) {
    let file = File::options()
        .create(true)
        .write(true)
        .open(filename)
        .unwrap();
    let length = bytes_per_frame * num_frames as usize;
    let mut flags = FallocateFlags::empty();
    if mode == AllocateMode::AllocateOnly {
        flags.insert(FallocateFlags::FALLOC_FL_KEEP_SIZE);
    }
    if length == 0 {
        return;
    }
    fallocate(file.as_raw_fd(), flags, 0, length.try_into().unwrap()).unwrap_or_else(|e| {
        panic!(
            "fallocate {} for num_frames={} failed: {}",
            filename, num_frames, e
        )
    });
}

pub type Shape3 = (Ix, Ix, Ix);
pub type Shape2 = (Ix, Ix);

pub fn recv_single<const PACKET_SIZE: usize, B: K2Block>(sector_id: u8) -> B {
    let port: u32 = 2001 + (sector_id as u32);
    let socket = create_mcast_socket(port, "225.1.1.1", "192.168.10.99");
    let mut buf: [u8; PACKET_SIZE] = [0; PACKET_SIZE];
    let (number_of_bytes, _src_addr) = socket.recv_from(&mut buf).expect("recv_from failed");
    assert_eq!(number_of_bytes, PACKET_SIZE);
    return B::from_bytes(&buf, sector_id);
}

///
/// Receive a single packet and read the PACKET_SIZE from it
///
pub fn recv_and_get_init() -> u32 {
    // finding the PACKET_SIZE: we just have a look at the first packet from the first sector
    // we can't use the normal decoding in K2ISBlock.from_bytes here, because *drumroll* we don't know
    // the size yet :)
    const PORT: u32 = 2001;
    let socket = create_mcast_socket(PORT, "225.1.1.1", "192.168.10.99");
    let mut buf: [u8; HEADER_SIZE] = [0; HEADER_SIZE];
    let (number_of_bytes, _src_addr) = socket.recv_from(&mut buf).expect("recv_from failed");
    assert_eq!(number_of_bytes, HEADER_SIZE);
    return decode_packet_size(&buf);
}
