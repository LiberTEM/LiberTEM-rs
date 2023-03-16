use std::{
    fs::remove_file,
    io::Write,
    net::{TcpListener, TcpStream},
    num::Wrapping,
    path::Path,
    thread,
    time::Duration,
};

use clap::Parser;
use ipc_test::SharedSlabAllocator;

const ITEMSIZE: usize = std::mem::size_of::<u16>();
const SLOT_SIZE_ITEMS: usize = 512 * 512;
const SLOT_SIZE_BYTES: usize = SLOT_SIZE_ITEMS * ITEMSIZE;

fn handle_connection(
    mut stream: TcpStream,
    num_slots: usize,
    send_num_items: usize,
    huge: bool,
    shm_path: &Path,
) {
    println!("handling consumer");
    let mut ssa = SharedSlabAllocator::new(num_slots, SLOT_SIZE_BYTES, huge, shm_path).unwrap();
    println!(
        "created shm area of total size {} MiB",
        (ssa.num_slots_total() * ssa.get_slot_size()) / 1024 / 1024
    );
    let handle = ssa.get_handle();
    let info = bincode::serialize(&handle).expect("serialize shm handle");

    stream
        .write_all(&info.len().to_be_bytes())
        .expect("send shm info size");
    stream.write_all(&info).expect("send shm info with fds");

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

                stream
                    .write_all(&slot_info_bin.len().to_be_bytes())
                    .expect("send slot info size");
                stream.write_all(&slot_info_bin).expect("send slot info");

                items_sent += 1;

                if items_sent == send_num_items {
                    break;
                }
            }
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

    #[arg(short, long, default_value_t = 100)]
    slots: usize,

    #[arg(short, long, default_value_t = 10000)]
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

    let listener = TcpListener::bind("127.0.0.1:9123").unwrap();

    // Stolen from the example on `UnixListener`:
    // accept connections and process them, spawning a new thread for each one
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                /* connection succeeded */

                let path = path.to_owned();

                thread::spawn(move || {
                    handle_connection(
                        stream,
                        args.slots,
                        args.num_items,
                        !args.disable_huge,
                        &path,
                    )
                });
            }
            Err(err) => {
                /* connection failed */
                println!("connection failed: {}", err);
                break;
            }
        }
    }
}
