#[derive(thiserror::Error, Debug)]
pub enum ShmConnectError {
    #[error("I/O error: {0}")]
    IOError(#[from] std::io::Error),

    #[error("serialization error: {0}")]
    SerializationError(#[from] bincode::Error),

    #[error("other error: {msg}")]
    Other { msg: String },
}
