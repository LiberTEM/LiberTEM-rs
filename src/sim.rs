mod common;

use crate::common::{DHeader, DumpRecordFile, FrameSender};
use clap::Parser;

#[derive(Parser)]
struct Cli {
    filename: String,
}

pub fn main() {
    let cli = Cli::parse();

    let mut sender = FrameSender::new(&cli.filename);
    sender.send_headers();
    sender.send_frames();
    sender.send_footer();
}
