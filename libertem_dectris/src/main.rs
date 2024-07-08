mod base_types;
mod bin_fmt;
mod exceptions;
mod sim;
mod sim_data_source;
mod sim_gen;

use crate::base_types::DHeader;
use crate::base_types::DImage;
use crate::base_types::DetectorConfig;
use crate::bin_fmt::{write_raw_msg, write_raw_msg_fh, write_serializable};
use crate::sim::FrameSender;
use crate::sim_data_source::DumpRecordFile;
use crate::sim_gen::make_sim_data;
use std::collections::HashMap;
use std::io::Write;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;
use zmq::Message;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct Cli {
    #[clap(subcommand)]
    action: Action,
    filename: String,
}

#[derive(Subcommand)]
enum Action {
    Cat {
        /// start at this message index (zero-based, inclusive)
        start_idx: usize,

        /// stop at this message index (zero-based, inclusive)
        end_idx: usize,
    },
    Inspect {
        /// display the first N messages
        #[clap(long)]
        head: Option<usize>,

        /// display a summary of all messages
        #[clap(short, long, action)]
        summary: bool,
    },
    Repeat {
        repetitions: usize,
    },
    Sim {
        uri: String,
    },
    SimMocked {
        num_frames: usize,
        uri: String,
    },
    Record {
        uri: String,
    },
}

fn action_cat(cli: &Cli, start_idx: usize, end_idx: usize) {
    let file = DumpRecordFile::from_file(&cli.filename);
    let mut cursor = file.get_cursor();

    eprintln!("writing from {start_idx} to {end_idx}");

    cursor.seek_to_msg_idx(start_idx);

    while cursor.get_msg_idx() <= end_idx {
        let msg = cursor.read_raw_msg();
        let length = (msg.len() as i64).to_le_bytes();
        std::io::stdout().write_all(&length).unwrap();
        std::io::stdout().write_all(msg).unwrap();
    }
}

fn inspect_dump_msg(raw_msg: &[u8], idx: usize) {
    let value_result: Result<serde_json::Value, _> = serde_json::from_slice(raw_msg);
    match value_result {
        Ok(value) => {
            let fmt_value = serde_json::to_string_pretty(&value).expect("pretty please");
            // let fmt_value = value.to_string();
            println!("msg {idx}:\n\n{fmt_value}\n");
        }
        Err(_) => {
            let len = raw_msg.len();
            println!("msg {idx}: <binary> ({len} bytes)");
        }
    }
}

fn get_msg_type(maybe_value: &Option<serde_json::Value>) -> String {
    match maybe_value {
        None => "<binary>".to_string(),
        Some(value) => {
            let htype = value
                .as_object()
                .expect("all json messages should be objects")
                .get("htype");
            if let Some(htype_str) = htype {
                if htype_str.is_string() {
                    htype_str
                        .as_str()
                        .expect("htype should be string here")
                        .to_string()
                } else {
                    "<unknown>".to_string()
                }
            } else {
                "<unknown>".to_string()
            }
        }
    }
}

fn get_summary(filename: &str) -> HashMap<String, usize> {
    let file = DumpRecordFile::from_file(filename);
    let mut cursor = file.get_cursor();

    let mut msg_map = HashMap::<String, usize>::new();

    while !cursor.is_at_end() {
        let raw_msg = cursor.read_raw_msg();
        let value = try_parse(raw_msg);
        let msg_type = get_msg_type(&value);
        msg_map.entry(msg_type).and_modify(|e| *e += 1).or_insert(1);
    }

    msg_map
}

fn inspect_print_summary(filename: &str) {
    let summary = get_summary(filename);

    println!("messages summary:");
    for (msg_type, count) in summary {
        println!("type {msg_type}: {count}");
    }
}

fn try_parse(raw_msg: &[u8]) -> Option<serde_json::Value> {
    let value_result: Result<serde_json::Value, _> = serde_json::from_slice(raw_msg);
    match value_result {
        Ok(value) => Some(value),
        Err(_) => None,
    }
}

fn action_inspect(cli: &Cli, head: Option<usize>, summary: bool) {
    let file = DumpRecordFile::from_file(&cli.filename);
    let mut cursor = file.get_cursor();

    match head {
        Some(head) => {
            for i in 0..head {
                let raw_msg = cursor.read_raw_msg();
                inspect_dump_msg(raw_msg, i);
            }
        }
        None => {
            let mut i = 0;
            while !cursor.is_at_end() {
                let raw_msg = cursor.read_raw_msg();
                inspect_dump_msg(raw_msg, i);
                i += 1;
            }
        }
    }

    if summary {
        inspect_print_summary(&cli.filename);
    }
}

fn action_repeat(cli: &Cli, repetitions: usize) {
    let file = DumpRecordFile::from_file(&cli.filename);
    let mut cursor = file.get_cursor();

    cursor.seek_to_first_header_of_type("dheader-1.0");
    let dheader = cursor.read_raw_msg();

    write_raw_msg(dheader);

    // detector config
    let detector_config_msg = cursor.read_raw_msg();
    let _detector_config: DetectorConfig = serde_json::from_slice(detector_config_msg).unwrap();
    let mut detector_config_value: serde_json::Value =
        serde_json::from_slice::<serde_json::Value>(detector_config_msg).unwrap();

    // XXX the header may lie about the number of images:
    let summary = get_summary(&cli.filename);
    let nimages = summary.get("<binary>").unwrap();
    let dest_num_images = nimages * repetitions;

    let new_det_config = detector_config_value.as_object_mut().unwrap();
    new_det_config
        .entry("nimages")
        .and_modify(|v| *v = 1.into());
    new_det_config
        .entry("trigger_mode")
        .and_modify(|v| *v = "exte".to_string().into());
    new_det_config
        .entry("ntrigger")
        .and_modify(|v| *v = dest_num_images.into());

    write_serializable(&detector_config_value);

    let mut idx = 0;
    for _ in 0..repetitions {
        let mut rep_cursor = file.get_cursor();
        rep_cursor.seek_to_first_header_of_type("dheader-1.0");
        let _dheader: DHeader = rep_cursor.read_and_deserialize().unwrap(); // discard dheader
        rep_cursor.read_raw_msg(); // discard detector config

        for _ in 0..*nimages {
            let mut dimage: DImage = rep_cursor
                .read_and_deserialize()
                .expect("failed to read dimage header");
            dimage.frame = idx;
            write_serializable(&dimage);

            let dimaged = rep_cursor.read_raw_msg();
            write_raw_msg(dimaged);

            let image = rep_cursor.read_raw_msg();
            write_raw_msg(image);

            // NOTE: we don't fake the timestamps (yet)
            let config = rep_cursor.read_raw_msg();
            write_raw_msg(config);

            idx += 1;
        }
    }
}

fn action_sim(filename: &str, uri: &str) {
    let mut sender = FrameSender::from_file(uri, filename, false);
    sender.send_headers(|| Some(())).unwrap();
    for _ in 0..sender.get_num_frames() {
        sender.send_frame().unwrap();
        std::thread::sleep(Duration::from_millis(1));
    }
    sender.send_footer();
}

fn action_sim_mocked(num_frames: usize, uri: &str) {
    let data = make_sim_data(num_frames);
    let drf = DumpRecordFile::from_raw_data(Arc::new(data));
    let mut sender = FrameSender::new(uri, &drf, false);
    sender.send_headers(|| Some(())).unwrap();
    for _ in 0..sender.get_num_frames() {
        sender.send_frame().unwrap();
        std::thread::sleep(Duration::from_millis(1));
    }
    sender.send_footer();
}

fn action_record(filename: &String, uri: &str) {
    let raw_file = std::fs::File::create(filename).unwrap();
    let mut file = std::io::BufWriter::new(raw_file);
    let ctx = zmq::Context::new();
    let socket = ctx.socket(zmq::PULL).unwrap();

    let mut size: usize = 0;

    socket.connect(uri).unwrap();
    socket.set_rcvhwm(4 * 2048).unwrap();

    let mut first_header: Message = Message::new();

    println!("synchronizing to first header");
    // synchronize to first header:
    let dheader = loop {
        // First acquisition header: Protocol etc
        socket.recv(&mut first_header, 0).unwrap();
        let first_header = first_header.as_str();
        if let Some(first_header) = first_header {
            let dheader_res: Result<DHeader, _> = serde_json::from_str(first_header);
            if let Ok(dheader_res) = dheader_res {
                break dheader_res;
            }
        }
    };

    println!("got first header: {dheader:?}");

    let start = Instant::now();

    socket.set_rcvtimeo(1000).unwrap();

    let mut detector_config_msg: Message = Message::new();
    // Second acquisition header: Detector config
    socket.recv(&mut detector_config_msg, 0).unwrap();

    let detector_config: DetectorConfig = serde_json::from_slice(&detector_config_msg).unwrap();

    println!("detector config: {:?}", detector_config);

    let num_msg = detector_config.get_num_images() * 4 + 1;

    println!("expecting {} messages", num_msg);

    write_raw_msg_fh(&first_header, &mut file);
    write_raw_msg_fh(&detector_config_msg, &mut file);

    size += first_header.len();
    size += detector_config_msg.len();

    let mut buf: Message = Message::new();

    for i in 0..num_msg {
        loop {
            match socket.recv(&mut buf, 0) {
                Ok(_) => break,
                Err(zmq::Error::EAGAIN) => {
                    println!("EAGAIN at {num_msg}, offset: {}", num_msg - i);
                    continue;
                }
                Err(err) => {
                    panic!("{}", err.to_string());
                }
            }
        }
        write_raw_msg_fh(&buf, &mut file);
        size += buf.len();
    }

    println!("Received {} bytes in {:?} s.", size, start.elapsed());
}

pub fn main() {
    let cli = Cli::parse();

    let env = env_logger::Env::default()
        .filter_or("LIBERTEM_DECTRIS_LOG_LEVEL", "error")
        .write_style_or("LIBERTEM_DECTRIS_LOG_STYLE", "always");
    env_logger::init_from_env(env);

    match cli.action {
        Action::Cat { start_idx, end_idx } => action_cat(&cli, start_idx, end_idx),
        Action::Inspect { head, summary } => action_inspect(&cli, head, summary),
        Action::Repeat { repetitions } => action_repeat(&cli, repetitions),
        Action::Sim { uri } => action_sim(&cli.filename, &uri),
        Action::SimMocked { uri, num_frames } => action_sim_mocked(num_frames, &uri),
        Action::Record { uri } => action_record(&cli.filename, &uri),
    }
}
