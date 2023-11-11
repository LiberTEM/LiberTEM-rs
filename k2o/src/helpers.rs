use std::{cmp, fs::File, os::unix::prelude::AsRawFd, time::Duration};

use log::info;
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
    let length = bytes_per_frame * num_frames;
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
    B::from_bytes(&buf, sector_id)
}

///
/// Receive a single packet and read the PACKET_SIZE from it
///
pub fn recv_and_get_init() -> u32 {
    // finding the PACKET_SIZE: we just have a look at the first packet from the first sector
    // we can't use the normal decoding in K2ISBlock.from_bytes here, because *drumroll* we don't know
    // the size yet :)
    const PORT: u32 = 2005;
    let group = "225.1.1.1";
    let local = "192.168.10.99";
    let socket = create_mcast_socket(PORT, group, local);
    info!("created multicast socket for group {group} local {local} port {PORT}");
    let mut buf: [u8; HEADER_SIZE] = [0; HEADER_SIZE];
    let (number_of_bytes, _src_addr) = socket.recv_from(&mut buf).expect("recv_from failed");
    info!("got initial packet");
    info!("initial packet: {buf:?}");
    assert_eq!(number_of_bytes, HEADER_SIZE);
    decode_packet_size(&buf)
}

pub fn make_realtime(prio: u32) -> Result<u32, Box<dyn std::error::Error>> {
    let c = dbus::blocking::Connection::new_system()?;

    let proxy = c.with_proxy(
        "org.freedesktop.RealtimeKit1",
        "/org/freedesktop/RealtimeKit1",
        Duration::from_millis(10000),
    );
    use dbus::blocking::stdintf::org_freedesktop_dbus::Properties;

    // Make sure we don't fail by wanting too much
    let max_prio: i32 = proxy.get("org.freedesktop.RealtimeKit1", "MaxRealtimePriority")?;
    let prio = cmp::min(prio, max_prio as u32);

    // Enforce RLIMIT_RTPRIO, also a must before asking rtkit for rtprio
    let max_rttime: i64 = proxy.get("org.freedesktop.RealtimeKit1", "RTTimeUSecMax")?;
    let new_limit = libc::rlimit64 {
        rlim_cur: max_rttime as u64,
        rlim_max: max_rttime as u64,
    };
    let mut old_limit = new_limit;
    if unsafe { libc::getrlimit64(libc::RLIMIT_RTTIME, &mut old_limit) } < 0 {
        return Err(Box::from("getrlimit failed"));
    }
    if unsafe { libc::setrlimit64(libc::RLIMIT_RTTIME, &new_limit) } < 0 {
        return Err(Box::from("setrlimit failed"));
    }

    // Finally, let's ask rtkit to make us realtime
    let thread_id = unsafe { libc::syscall(libc::SYS_gettid) };
    let r = proxy.method_call(
        "org.freedesktop.RealtimeKit1",
        "MakeThreadRealtime",
        (thread_id as u64, prio),
    );

    if r.is_err() {
        unsafe { libc::setrlimit64(libc::RLIMIT_RTTIME, &old_limit) };
    }

    r?;
    Ok(prio)
}
