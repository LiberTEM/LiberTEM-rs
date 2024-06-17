use crate::frame_stack::FrameMeta;

// TODO: this is the rich "connection status" enum we should use in the future,
// for now we use something much simpler
enum ReceiverStatus {
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

trait Receiver<M: FrameMeta> {
    fn get_status(&self) -> ReceiverStatus;
}
