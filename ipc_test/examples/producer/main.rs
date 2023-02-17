use std::{
    os::unix::net::{UnixListener, UnixStream},
    thread, time::Duration, io::Write, num::Wrapping, path::Path, fs::remove_file,
};

use clap::Parser;
use ipc_test::SharedSlabAllocator;
use sendfd::SendWithFd;

const ITEMSIZE: usize = std::mem::size_of::<u16>();
const SLOT_SIZE_ITEMS: usize = 512*512;
const SLOT_SIZE_BYTES: usize = SLOT_SIZE_ITEMS*ITEMSIZE;

fn handle_connection(mut stream: UnixStream, num_slots: usize, send_num_items: usize, huge: bool) {
    println!("handling consumer");
    let mut ssa = SharedSlabAllocator::new(
        num_slots,
        SLOT_SIZE_BYTES,
        huge,
    ).unwrap();
    println!("created shm area of total size {} MiB", (ssa.num_slots_total() * ssa.get_slot_size()) / 1024 / 1024);
    let handle = ssa.get_handle();
    let fds = [handle.fd];
    let info = bincode::serialize(&handle.info).expect("serialize shm info");

    stream.write_all(&info.len().to_be_bytes()).expect("send shm info size");
    stream.send_with_fd(&info, &fds).expect("send shm info with fds");

    let mut items_sent: usize = 0;

    loop {
        match ssa.get_mut() {
            Some(mut slotw) => {
                let slot_as_u16: &mut [u16] = bytemuck::cast_slice_mut(slotw.as_slice_mut());
                // slots can be larger than requested, so cut off any remainder here:
                let slot_as_u16 = &mut slot_as_u16[0..(SLOT_SIZE_ITEMS)];
                let mut value: Wrapping<u16> = Wrapping(0);
                for item in slot_as_u16.iter_mut() {
                    *item = value.0;
                    //*item = 0; // value.0;
                    value += 1;
                }

                let slot_info = ssa.writing_done(slotw);
                let slot_info_bin = bincode::serialize(&slot_info).expect("serialize slot info");

                stream.write_all(&slot_info_bin.len().to_be_bytes()).expect("send slot info size");
                stream.write_all(&slot_info_bin).expect("send slot info");

                items_sent += 1;

                if items_sent == send_num_items {
                    break;
                }
            },
            None => thread::sleep(Duration::from_millis(1)),
        }
    }

    println!("done sending {} items", send_num_items);
    while ssa.num_slots_free() < ssa.num_slots_total() {
        thread::sleep(Duration::from_millis(100));
    }
    println!("done!")
}

#[derive(Parser, Debug, Clone)]
struct Args {
    #[arg(short, long)]
    socket_path: String,

    #[arg(long, default_value_t=100)]
    slots: usize,

    #[arg(short, long, default_value_t=10000)]
    num_items: usize,

    #[arg(short, long)]
    disable_huge: bool,
}

fn main() {
    let args = Args::parse();

    let path = Path::new(&args.socket_path);
    if path.exists() {
        remove_file(path).expect("remove existing socket");
    }

    let listener = UnixListener::bind(args.socket_path).unwrap();

    // Stolen from the example on `UnixListener`:
    // accept connections and process them, spawning a new thread for each one
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                /* connection succeeded */
                thread::spawn(move || handle_connection(stream, args.slots, args.num_items, !args.disable_huge));
            }
            Err(err) => {
                /* connection failed */
                println!("connection failed: {}", err);
                break;
            }
        }
    }
}
