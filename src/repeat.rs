mod common;

use std::{
    collections::HashMap,
    io::{self, Write},
};

use crate::common::{DHeader, DImage, DetectorConfig, DumpRecordFile};
use clap::Parser;
use serde::Serialize;

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct Cli {
    filename: String,

    repetitions: usize,
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

fn try_parse(raw_msg: &[u8]) -> Option<serde_json::Value> {
    let value_result: Result<serde_json::Value, _> = serde_json::from_slice(raw_msg);
    match value_result {
        Ok(value) => Some(value),
        Err(_) => None,
    }
}

fn get_summary(filename: &str) -> HashMap<String, usize> {
    let file = DumpRecordFile::new(&filename);
    let mut cursor = file.get_cursor();

    let mut msg_map = HashMap::<String, usize>::new();

    while !cursor.is_at_end() {
        let raw_msg = cursor.read_raw_msg();
        let value = try_parse(&raw_msg);
        let msg_type = get_msg_type(&value);
        msg_map.entry(msg_type).and_modify(|e| *e += 1).or_insert(1);
    }

    return msg_map;
}

fn write_raw_msg(msg: &[u8]) {
    let length = (msg.len() as i64).to_le_bytes();
    io::stdout().write(&length).unwrap();
    io::stdout().write_all(msg).unwrap();
}

fn write_serializable<T>(value: &T)
where
    T: Serialize,
{
    let binding = serde_json::to_string(&value).expect("serialization should not fail");
    let msg_raw = binding.as_bytes();
    write_raw_msg(&msg_raw);
}

pub fn main() {
    let cli = Cli::parse();

    let file = DumpRecordFile::new(&cli.filename);
    let mut cursor = file.get_cursor();

    cursor.seek_to_first_header_of_type("dheader-1.0");
    let dheader = cursor.read_raw_msg();

    write_raw_msg(&dheader);

    // detector config
    let detector_config_msg = cursor.read_raw_msg();
    let _detector_config: DetectorConfig = serde_json::from_slice(detector_config_msg).unwrap();
    let mut detector_config_value: serde_json::Value =
        serde_json::from_slice::<serde_json::Value>(detector_config_msg)
            .unwrap()
            .to_owned();

    // XXX the heaer may lie about the number of images:
    let summary = get_summary(&cli.filename);
    let nimages = summary.get("<binary>").unwrap();
    let dest_num_images = nimages * cli.repetitions;

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
    for _ in 0..cli.repetitions {
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
            write_raw_msg(&dimaged);

            let image = rep_cursor.read_raw_msg();
            write_raw_msg(&image);

            // NOTE: we don't fake the timestamps (yet)
            let config = rep_cursor.read_raw_msg();
            write_raw_msg(&config);

            idx += 1;
        }
    }
}
