pub enum ReceiverStatus {
    Initializing,
    Idle,
    Armed,
    Running,
    Cancelling,
    Finished,
    Ready,
    Shutdown,
    Closed,
}

pub trait Receiver {
    fn get_status(&self) -> ReceiverStatus;
}
