use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use crossbeam_channel::{unbounded, Receiver, RecvTimeoutError, SendError, Sender};
use ipc_test::SharedSlabAllocator;
use opentelemetry::Context;

use crate::{
    block::{BlockRouteInfo, K2Block},
    events::{EventMsg, EventReceiver},
    frame::{FrameForWriting, K2Frame},
    helpers::{set_cpu_affinity, CPU_AFF_ASSEMBLY},
};

pub struct PendingFrames<F: K2Frame> {
    pub frames: HashMap<u32, F::FrameForWriting>,
    timeout: Duration,
}

impl<F: K2Frame> PendingFrames<F> {
    pub fn new() -> Self {
        PendingFrames {
            frames: HashMap::new(),
            timeout: Duration::from_millis(100),
        }
    }

    pub fn assign_block<B: K2Block>(&mut self, block: &B, shm: &mut SharedSlabAllocator) {
        let frame = match self.frames.get_mut(&block.get_frame_id()) {
            None => {
                let frame = F::FrameForWriting::empty_from_block(block, shm);
                self.frames.insert(block.get_frame_id(), frame);
                self.frames.get_mut(&block.get_frame_id()).unwrap()
            }
            Some(frame) => frame,
        };
        frame.assign_block(block);
    }

    /// call a callback function `cb` on finished frames and remove from list of pending frames
    /// as we keep the finished data around by default, it's possible to call this function
    /// only for every N received blocks, or only if a certain time has passed, without losing
    /// data (possibly small performance loss because of data locality, if you wait for too long)
    pub fn retire_finished<E, CB: Fn(F, &SharedSlabAllocator) -> Result<(), E>>(
        &mut self,
        shm: &mut SharedSlabAllocator,
        cb: CB,
    ) -> Result<(), E> {
        let mut to_remove: Vec<u32> = Vec::with_capacity(16);

        for (_, frame) in self.frames.iter() {
            if frame.is_finished() {
                to_remove.push(frame.get_frame_id());
            }
        }

        for frame_id in to_remove {
            let frame_for_writing = self.frames.remove(&frame_id).unwrap();
            let frame = frame_for_writing.writing_done(shm);
            cb(frame, shm)?;
        }

        Ok(())
    }

    pub fn retire_timed_out<E, CB: Fn(F) -> Result<(), E>>(
        &mut self,
        shm: &mut SharedSlabAllocator,
        cb: CB,
    ) -> Result<(), E> {
        let mut to_remove: Vec<u32> = Vec::with_capacity(16);
        let now = Instant::now();

        for (_, frame) in self.frames.iter() {
            let delta = now - frame.get_modified_timestamp();
            if !frame.is_finished() && delta > self.timeout {
                to_remove.push(frame.get_frame_id());
            }
        }

        for frame_id in to_remove {
            let frame_for_writing = self.frames.remove(&frame_id).unwrap();
            let frame = frame_for_writing.writing_done(shm);
            cb(frame)?;
        }

        Ok(())
    }

    pub fn num_finished(&self) -> usize {
        return self
            .frames
            .iter()
            .filter(|(_, frame)| frame.is_finished())
            .count();
    }

    pub fn num_unfinished(&self) -> usize {
        return self
            .frames
            .iter()
            .filter(|(_, frame)| !frame.is_finished())
            .count();
    }
}

impl<F: K2Frame> Default for PendingFrames<F> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
enum AssemblyError {
    Disconnected,
}

pub enum AssemblyResult<F: K2Frame> {
    AssembledFrame(F),
    AssemblyTimeout { frame: F, frame_id: u32 },
}

fn assembly_worker<F: K2Frame, B: K2Block>(
    blocks_rx: Receiver<B>,
    frames_tx: Sender<AssemblyResult<F>>,
    recycle_blocks_tx: &Sender<B>,
    stop_event: Arc<AtomicBool>,
    handle_path: &str,
) -> Result<(), AssemblyError> {
    let mut pending: PendingFrames<F> = PendingFrames::new();
    let mut shm = SharedSlabAllocator::connect(handle_path).expect("connect to SHM");

    loop {
        match blocks_rx.recv_timeout(Duration::from_millis(100)) {
            Ok(block) => {
                pending.assign_block(&block, &mut shm);
                recycle_blocks_tx.send(block).unwrap();
            }
            Err(RecvTimeoutError::Timeout) => {
                if stop_event.load(Ordering::Relaxed) {
                    // FIXME: do we need to do anything with accumulated data?
                    break;
                }
            }
            Err(RecvTimeoutError::Disconnected) => {
                // the other end has gone away, so we can quit, too.
                break;
            }
        }

        pending.retire_finished(&mut shm, |frame, _| {
            if let Err(SendError(_)) = frames_tx.send(AssemblyResult::AssembledFrame(frame)) {
                // can't retire frames if the other end is disconnected, so we die, too:
                return Err(AssemblyError::Disconnected);
            }
            Ok(())
        })?;

        pending.retire_timed_out(&mut shm, |frame: F| {
            let frame_id = frame.get_frame_id();
            if let Err(SendError(_)) =
                frames_tx.send(AssemblyResult::AssemblyTimeout { frame, frame_id })
            {
                // can't handle timed out frames if the other end is disconnected, so we die, too:
                return Err(AssemblyError::Disconnected);
            }
            Ok(())
        })?;
    }

    Ok(())
}

pub fn assembler_main<F: K2Frame, B: K2Block>(
    blocks_rx: &Receiver<(B, BlockRouteInfo)>,
    frames_tx: &Sender<AssemblyResult<F>>,
    recycle_blocks_tx: &Sender<B>,
    events_rx: EventReceiver,
    shm: SharedSlabAllocator,
) {
    let pool_size = 4;
    let mut worker_channels: Vec<Sender<B>> = Vec::with_capacity(pool_size);

    let stop_event = Arc::new(AtomicBool::new(false));

    let ctx = Context::current();

    for _ in 0..128 {
        recycle_blocks_tx.send(B::empty(0)).unwrap();
    }

    crossbeam::scope(|s| {
        for idx in 0..pool_size {
            let asm_worker_ctx = ctx.clone();
            let (tx, rx) = unbounded::<B>();

            let this_stop_event = Arc::clone(&stop_event);

            let shm_handle = shm.get_handle();

            s.builder()
                .name(format!("asm-frame-{idx}"))
                .spawn(move |_| {
                    asm_worker_ctx.attach();
                    set_cpu_affinity(CPU_AFF_ASSEMBLY + idx);
                    if assembly_worker::<F, B>(
                        rx.clone(),
                        frames_tx.clone(),
                        &recycle_blocks_tx.clone(),
                        this_stop_event,
                        &shm_handle.os_handle,
                    )
                    .is_err()
                    {
                        eprintln!("disconnect in asm-frame");
                    }
                })
                .expect("could not spawn asm-frame");
            worker_channels.push(tx);
        }

        loop {
            match blocks_rx.recv_timeout(Duration::from_millis(100)) {
                Ok((block, block_route)) => {
                    let frame_id = block_route.get_frame_id();
                    let chan = worker_channels[(frame_id as usize) % pool_size].clone();
                    chan.send(block).unwrap();
                }
                Err(RecvTimeoutError::Disconnected) => {
                    stop_event.store(true, Ordering::Relaxed);
                    break;
                }
                // FIXME: might want to be a bit more eager here to make shutdown faster:
                Err(RecvTimeoutError::Timeout) => {
                    if let Ok(msg) = events_rx.try_recv() {
                        if msg == (EventMsg::Shutdown {}) {
                            stop_event.store(true, Ordering::Relaxed);
                            break;
                        }
                    }
                }
            }
        }
    })
    .unwrap();
}

#[cfg(test)]
mod tests {
    use ipc_test::SharedSlabAllocator;
    use ndarray::s;
    use ndarray::ArrayView;
    use tempfile::tempdir;

    use crate::assemble::PendingFrames;
    use crate::block::K2Block;
    use crate::block::K2ISBlock;
    use crate::decode::HEADER_SIZE;
    use crate::events::Binning;
    use crate::frame::FrameForWriting;
    use crate::frame::K2ISFrame;
    use crate::frame::K2ISFrameForWriting;

    use super::K2Frame;

    const PACKET_SIZE: usize = 0x5758;
    const DECODED_SIZE: usize = (PACKET_SIZE - HEADER_SIZE) * 2 / 3;

    fn make_test_data() -> Vec<u16> {
        let mut original_values: Vec<u16> = Vec::new();
        original_values.extend((0u16..=0xFFF).cycle().take(DECODED_SIZE));
        original_values
    }

    #[test]
    fn k2frame_assign_blocks_happy_case() {
        let socket_dir = tempdir().unwrap();
        let socket_as_path = socket_dir.into_path().join("stuff.socket");

        const FRAME_ID: u32 = 42;
        let mut ssa = SharedSlabAllocator::new(
            10,
            K2ISFrame::FRAME_HEIGHT * K2ISFrame::FRAME_WIDTH * std::mem::size_of::<u16>(),
            false,
            &socket_as_path,
        )
        .expect("create SHM area for testing");
        let mut frame: K2ISFrameForWriting = K2ISFrameForWriting::empty(FRAME_ID, &mut ssa);

        assert!(!frame.is_finished());

        let payload = &make_test_data();
        let payload_view = ArrayView::from_shape((930, 16), &payload[..]).unwrap();

        for y_idx in 0..2 {
            for x_idx in 0..128 {
                let start_x = x_idx * 16;
                let start_y = y_idx * 930;

                let block: K2ISBlock =
                    K2ISBlock::from_vec_and_pos(payload, start_x, start_y, FRAME_ID);
                frame.assign_block(&block);
            }
        }

        frame.dump_finished_state();

        assert!(frame.is_finished());

        let frame_arr = frame.as_array();

        // all slices contain the test data pattern:
        for y_idx in 0..2 {
            for x_idx in 0..128 {
                let start_x = x_idx * 16;
                let start_y = y_idx * 930;

                let slice: ndarray::ArrayBase<ndarray::ViewRepr<&u16>, ndarray::Dim<[usize; 2]>> =
                    frame_arr.slice(s![start_y..start_y + 930, start_x..start_x + 16]);
                assert!(payload_view.abs_diff_eq(&slice, 0));

                println!("{payload_view}");
            }
        }
    }

    #[test]
    fn pending_frames_assign_blocks_happy_case() {
        let socket_dir = tempdir().unwrap();
        let socket_as_path = socket_dir.into_path().join("stuff.socket");

        const FRAME_ID: u32 = 42;
        let mut pending_frames: PendingFrames<K2ISFrame> = PendingFrames::new();
        let mut ssa = SharedSlabAllocator::new(
            10,
            K2ISFrame::FRAME_HEIGHT * K2ISFrame::FRAME_WIDTH * std::mem::size_of::<u16>(),
            false,
            &socket_as_path,
        )
        .expect("create SHM area for testing");

        assert_eq!(pending_frames.num_finished(), 0);
        assert_eq!(pending_frames.num_unfinished(), 0);

        let payload = &make_test_data();
        let payload_view = ArrayView::from_shape((930, 16), &payload[..]).unwrap();

        for y_idx in 0..2 {
            for x_idx in 0..128 {
                let start_x = x_idx * 16;
                let start_y = y_idx * 930;

                let block: K2ISBlock =
                    K2ISBlock::from_vec_and_pos(payload, start_x, start_y, FRAME_ID);
                pending_frames.assign_block(&block, &mut ssa);
            }
        }

        assert_eq!(pending_frames.num_finished(), 1);
        assert_eq!(pending_frames.num_unfinished(), 0);

        pending_frames
            .retire_finished(&mut ssa, |frame, ssa| {
                let subframe = frame.get_subframe(0, &Binning::Bin1x, ssa);
                subframe.apply_to_payload_array(|frame_arr| {
                    // all slices contain the test data pattern:
                    for y_idx in 0..2 {
                        for x_idx in 0..128 {
                            let start_x = x_idx * 16;
                            let start_y = y_idx * 930;
                            let slice =
                                frame_arr.slice(s![start_y..start_y + 930, start_x..start_x + 16]);
                            assert!(payload_view.abs_diff_eq(&slice, 0));
                            println!("{payload_view}");
                        }
                    }
                });

                Ok::<_, ()>(())
            })
            .unwrap();
    }

    #[test]
    fn assemly_single_block() {
        let socket_dir = tempdir().unwrap();
        let socket_as_path = socket_dir.into_path().join("stuff.socket");

        const FRAME_ID: u32 = 42;
        let mut ssa = SharedSlabAllocator::new(
            10,
            K2ISFrame::FRAME_HEIGHT * K2ISFrame::FRAME_WIDTH * std::mem::size_of::<u16>(),
            false,
            &socket_as_path,
        )
        .expect("create SHM area for testing");
        let mut pending_frames: PendingFrames<K2ISFrame> = PendingFrames::new();

        assert_eq!(pending_frames.num_finished(), 0);
        assert_eq!(pending_frames.num_unfinished(), 0);

        let payload = &make_test_data();

        let start_x = 2032;
        let start_y = 930;

        let block: K2ISBlock = K2ISBlock::from_vec_and_pos(payload, start_x, start_y, FRAME_ID);
        assert_eq!(block.get_x_end(), 2047);
        assert_eq!(block.get_y_end(), 1859);
        pending_frames.assign_block(&block, &mut ssa);

        assert_eq!(pending_frames.num_finished(), 0);
        assert_eq!(pending_frames.num_unfinished(), 1);

        pending_frames
            .retire_finished(&mut ssa, |_, _| -> Result<(), ()> {
                panic!("this should not be called, as we don't have any finished frames yet");
            })
            .unwrap();
    }
}
