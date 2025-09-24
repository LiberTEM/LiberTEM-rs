use clap::Parser;

#[derive(clap::ValueEnum, Clone, Copy, Debug)]
pub enum Mode {
    IS,
    Summit,
}

/// Test program - arm and perform a single acquisition
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
pub struct Args {
    /// Camera mode, determined automatically from first frame if not given
    #[clap(short, long, value_enum)]
    pub mode: Option<Mode>,

    /// Disable allocation re-use
    #[clap(short = 'r', long)]
    pub disable_reuse: bool,

    /// Shared memory socket path
    #[clap(short = 's', long, default_value = "/run/user/1000/k2is-shm-socket")]
    pub shm_path: String,
}
