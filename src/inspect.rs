use std::collections::HashMap;
mod common;

use crate::common::DumpRecordFile;
use clap::Parser;

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct Cli {
    filename: String,

    /// display the first N messages
    #[clap(short, long)]
    head: Option<usize>,

    /// display a summary of all messages
    #[clap(short, long, action)]
    summary: bool,
}

fn dump_msg(raw_msg: &[u8], idx: usize) {
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
            if let Some(htype_str) = htype && htype_str.is_string() {
                htype_str.as_str().expect("htype should be string here").to_string()
            } else {
                "<unknown>".to_string()
            }
        }
    }
}

fn print_summary(filename: &str) {
    let file = DumpRecordFile::new(&filename);
    let mut cursor = file.get_cursor();

    let mut msg_map = HashMap::<String, usize>::new();

    while !cursor.is_at_end() {
        let raw_msg = cursor.read_raw_msg();
        let value = try_parse(&raw_msg);
        let msg_type = get_msg_type(&value);
        msg_map.entry(msg_type).and_modify(|e| *e += 1).or_insert(1);
    }

    println!("messages summary:");
    for (msg_type, count) in msg_map {
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

pub fn main() {
    let cli = Cli::parse();

    let file = DumpRecordFile::new(&cli.filename);
    let mut cursor = file.get_cursor();

    match cli.head {
        Some(head) => {
            for i in 0..head {
                let raw_msg = cursor.read_raw_msg();
                dump_msg(raw_msg, i);
            }
        }
        None => {
            let mut i = 0;
            while !cursor.is_at_end() {
                let raw_msg = cursor.read_raw_msg();
                dump_msg(raw_msg, i);
                i += 1;
            }
        }
    }

    if cli.summary {
        print_summary(&cli.filename);
    }
}
