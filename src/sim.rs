mod common;

use crate::common::FrameSender;
use clap::Parser;

#[derive(Parser)]
struct Cli {
    filename: String,
    uri: String,
}

pub fn main() {
    let cli = Cli::parse();

    let mut sender = FrameSender::new(&cli.uri, &cli.filename);
    sender.send_headers();
    sender.send_frames();
    sender.send_footer();
}
