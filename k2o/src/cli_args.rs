use clap::Parser;

use crate::events::WriterType;

#[derive(clap::ArgEnum, Clone, Copy, Debug)]
pub enum Mode {
    IS,
    Summit,
}

#[derive(clap::ArgEnum, Clone, Copy, Debug, PartialEq)]
pub enum WriteMode {
    Direct,
    MMAP,
    HDF5,
}

impl Into<WriterType> for WriteMode {
    fn into(self) -> WriterType {
        match self {
            WriteMode::Direct => WriterType::Direct,
            WriteMode::MMAP => WriterType::Mmap,
            #[cfg(not(feature = "hdf5"))]
            WriteMode::HDF5 => panic!("hdf5 not supported"),
            #[cfg(feature = "hdf5")]
            WriteMode::HDF5 => WriterType::HDF5,
        }
    }
}

/// Test program - arm and perform a single acquisition
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
pub struct Args {
    /// Camera mode, determined automatically from first frame if not given
    #[clap(short, long, arg_enum)]
    pub mode: Option<Mode>,

    /// Write mode
    #[clap(short = 'w', long, arg_enum, default_value = "direct")]
    pub write_mode: WriteMode,

    /// Path to the file where the data should be written to
    /// (will be created if it doesn't exist, overwritten otherwise)
    #[clap(short = 'o', long, default_value = "/cachedata/alex/foo.raw")]
    pub write_to: String,

    /// Disable allocation re-use
    #[clap(short = 'r', long)]
    pub disable_reuse: bool,

    /// Shared memory socket path
    #[clap(short = 's', long, default_value = "/run/user/1000/k2is-shm-socket")]
    pub shm_path: String,
}
