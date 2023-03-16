use std::{
    io::Read,
    net::TcpStream,
    num::Wrapping,
    time::{Duration, Instant},
};

use clap::Parser;
use ipc_test::{SHMHandle, SharedSlabAllocator, SlotInfo};

#[derive(Parser, Debug)]
struct Args {
    #[arg(short, long)]
    socket_path: String,

    #[arg(short, long, default_value_t = 10000)]
    num_items: usize,
}

fn read_size(mut stream: &TcpStream) -> usize {
    let mut buf: [u8; std::mem::size_of::<usize>()] = [0; std::mem::size_of::<usize>()];
    stream.read_exact(&mut buf).expect("read message size");
    usize::from_be_bytes(buf)
}

const ITEMSIZE: usize = std::mem::size_of::<u16>();
const SLOT_SIZE_ITEMS: usize = 512 * 512;
const SLOT_SIZE_BYTES: usize = SLOT_SIZE_ITEMS * ITEMSIZE;

fn main() {
    let args = Args::parse();

    let mut stream = TcpStream::connect("127.0.0.1:9123").expect("connect to socket");

    let size = read_size(&stream);
    let mut bytes: Vec<u8> = vec![0; size];

    stream
        .read_exact(bytes.as_mut_slice())
        .expect("read initial message with fds");

    let shm_handle: SHMHandle =
        bincode::deserialize(&bytes[..]).expect("deserialize SHMInfo object");

    let mut ssa = SharedSlabAllocator::connect(&shm_handle.os_handle).unwrap();

    let mut sum: f64 = 0.0;
    let mut idx: usize = 0;

    let mut t0 = Instant::now();
    let mut bytes_processed = 0;

    loop {
        if idx == args.num_items {
            break;
        }

        let size = read_size(&stream);
        let mut bytes: Vec<u8> = vec![0; size];
        stream
            .read_exact(bytes.as_mut_slice())
            .expect("read slot info");

        let slot_info: SlotInfo = bincode::deserialize(&bytes[..]).expect("deserialize slot info");
        let slot = ssa.get(slot_info.slot_idx);
        let contents: &[u16] = bytemuck::cast_slice(slot.as_slice());

        // emulate work: sum up all values
        let mut sum_part: Wrapping<u64> = Wrapping(0);
        // slots can be larger than requested, so cut off any remainder here:
        let contents_part = &contents[0..(SLOT_SIZE_ITEMS)];
        for item in contents_part {
            sum_part += Wrapping(*item as u64);
        }
        // some additional "work":
        //std::thread::sleep(Duration::from_micros(1));

        ssa.free_idx(slot_info.slot_idx);

        sum += sum_part.0 as f64;
        bytes_processed += SLOT_SIZE_BYTES;

        if t0.elapsed() > Duration::from_secs(1) {
            let slots_free = ssa.num_slots_free();
            println!(
                "idx: {idx:5}, sum: {sum_part}, throughput: {:7.2} MiB/s, slots free: {slots_free}",
                bytes_processed as f32 / 1024.0 / 1024.0
            );
            bytes_processed = 0;
            t0 = Instant::now();
        }

        idx += 1;
    }

    println!("sum: {}", sum);
}
