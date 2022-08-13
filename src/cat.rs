mod common;

use std::io::{self, Write};

use crate::common::DumpRecordFile;
use clap::Parser;

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct Cli {
    filename: String,

    /// start at this message index (zero-based, inclusive)
    start_idx: usize,

    /// stop at this message index (zero-based, inclusive)
    end_idx: usize,
}

pub fn main() {
    let cli = Cli::parse();

    let file = DumpRecordFile::new(&cli.filename);
    let mut cursor = file.get_cursor();

    let start_idx = cli.start_idx;
    let end_idx = cli.end_idx;

    eprintln!("writing from {start_idx} to {end_idx}");

    cursor.seek_to_msg_idx(start_idx);

    while cursor.get_msg_idx() <= end_idx {
        let msg = cursor.read_raw_msg();
        let length = (msg.len() as i64).to_le_bytes();
        io::stdout().write(&length).unwrap();
        io::stdout().write_all(msg).unwrap();
    }
}
