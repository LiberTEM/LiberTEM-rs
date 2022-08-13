mod common;

use std::time::{Duration, Instant};

use clap::Parser;

use common::DHeader;
use zmq::{Message, Socket};

use crate::common::{DConfig, DImage, DImageD};

fn wait_for_header(socket: &Socket) -> DHeader {
    let mut msg: Message = Message::new();
    socket.recv(&mut msg, 0).unwrap();
    // panic in case of any other header type:
    let dheader: DHeader = serde_json::from_str(msg.as_str().unwrap()).unwrap();
    println!("dheader: {dheader:?}");
    return dheader;
}

#[derive(clap::Parser)]
struct Cli {
    /// number of frames to acquire
    num_frames: u64,
}

pub fn main() {
    let ctx = zmq::Context::new();

    let socket = ctx.socket(zmq::PULL).unwrap();
    socket.connect("tcp://127.0.0.1:9999").unwrap();

    socket.set_rcvhwm(4 * 256).unwrap();

    let cli = Cli::parse();

    let num_frames = cli.num_frames;

    let mut msg: Message = Message::new();

    // NOTE: only for benchmarking, the message types probably don't match up
    // (as there can be some extra messages in there...)

    // first message: the "header-header", that defines what variant of the header follows:
    wait_for_header(&socket);

    // second message: the header itself (ignored here)
    socket.recv(&mut msg, 0).unwrap();

    // just benchmark reading messages for N frames:
    let t0 = Instant::now();

    let mut image_data: Vec<u8> = Vec::with_capacity(512 * 512 * 4);

    for idx in 0..num_frames {
        // DImage:
        // socket.poll(PollEvents::POLLIN, 100).unwrap();
        socket.recv(&mut msg, 0).unwrap();
        let dimage: DImage = serde_json::from_str(msg.as_str().unwrap()).unwrap();

        // DImageD:
        // socket.poll(PollEvents::POLLIN, 100).unwrap();
        socket.recv(&mut msg, 0).unwrap();
        let dimaged: DImageD = serde_json::from_str(msg.as_str().unwrap()).unwrap();

        // compressed image data:
        // socket.poll(PollEvents::POLLIN, 100).unwrap();
        socket.recv(&mut msg, 0).unwrap();
        image_data.truncate(0);
        image_data.extend_from_slice(&msg);

        // DConfig:
        // socket.poll(PollEvents::POLLIN, 100).unwrap();
        socket.recv(&mut msg, 0).unwrap();
        let dconfig: DConfig = serde_json::from_str(msg.as_str().unwrap()).unwrap();

        // std::thread::sleep is not accurate for small durations,
        // probably because the system call beneath this has "sleep for at least" semantics.
        // use an alternative that spins here:
        spin_sleep::sleep(Duration::from_micros(50));
    }
    let elapsed = t0.elapsed();
    let fps = num_frames as f64 / elapsed.as_secs_f64();

    println!("time passed: {elapsed:?} ({fps} fps)");

    // drain the rest of the zmq socket:
    socket.set_rcvtimeo(1000).unwrap();
    loop {
        let res = socket.recv(&mut msg, 0);
        match res {
            Ok(_) => {
                let footer = msg.as_str().unwrap();
                println!("footer: {footer}");
            }
            Err(zmq::Error::EAGAIN) => {
                println!("timeout, we are probably done");
                break;
            }
            Err(e) => panic!("unexpected error: {e}"),
        }
    }
}
