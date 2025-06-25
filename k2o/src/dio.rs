use nix::libc::O_DIRECT;
use std::fs::OpenOptions;
use std::os::unix::prelude::OpenOptionsExt;

pub fn open_direct(path: &str) -> Result<std::fs::File, std::io::Error> {
    return OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .custom_flags(O_DIRECT)
        .open(path);
}
