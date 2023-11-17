use std::{io::ErrorKind, time::Duration};

use crossbeam_channel::{Receiver, Sender, TryRecvError};
use log::{debug, error, info, warn};

use crate::{
    block::{BlockRouteInfo, K2Block},
    events::{AcquisitionSync, EventBus, EventMsg, EventReceiver, Events},
    helpers::{make_realtime, set_cpu_affinity, CPU_AFF_DECODE_START},
    net::create_mcast_socket,
};

#[derive(PartialEq)]
enum RecvState {
    /// Receiving UDP packets, but not doing anything with them
    Idle,

    /// Receiving and decoding blocks, but not passing them on downstream to assembly,
    /// goes to `Receiving` state once a block with sync flag has been received
    WaitForSync { acquisition_id: usize },

    /// When receiving the next block, go to the `WaitForFrame` state
    WaitForNext { acquisition_id: usize },

    /// Start receiving and decoding at the start of the specified frame id,
    /// regardless of sync status
    WaitForFrame {
        acquisition_id: usize,
        frame_id: u32,
    },

    /// Recveicing and decoding packets, and sending the decoded packets down
    /// the pipeline
    Receiving { acquisition_id: usize },
}

fn block_for_bytes<B: K2Block>(
    buf: &[u8],
    chan: &Receiver<B>,
    acquisition_id: usize,
    sector_id: u8,
) -> B {
    let block: B = {
        let maybe_block = chan.try_recv();
        match maybe_block {
            Err(_) => B::from_bytes(buf, sector_id, acquisition_id),
            Ok(mut b) => {
                b.replace_with(buf, sector_id, acquisition_id);
                b
            }
        }
    };
    block.validate();
    block
}

/// receive and decode a block from the specified sector, and send it to the
/// central assembly thread
#[allow(clippy::too_many_arguments)]
pub fn recv_decode_loop<B: K2Block, const PACKET_SIZE: usize>(
    sector_id: u8,
    port: u32,
    assembly_channel: &Sender<(B, BlockRouteInfo)>,
    recycle_blocks_rx: &Receiver<B>,
    recycle_blocks_tx: &Sender<B>,
    events_rx: &EventReceiver,
    events: &Events,
    local_addr: String,
) {
    let socket = create_mcast_socket(port, "225.1.1.1", &local_addr);
    let mut buf: [u8; PACKET_SIZE] = [0; PACKET_SIZE];

    match make_realtime(50) {
        Ok(_) => info!("successfully enabled realtime priority"),
        Err(e) => error!("failed to set realtime priority: {e:?}"),
    }

    socket
        .set_read_timeout(Some(Duration::from_millis(10)))
        .unwrap();

    let mut state = RecvState::Idle;

    info!("Listening on {local_addr}:{port} for sector {sector_id}");

    let aff = CPU_AFF_DECODE_START + sector_id as usize;
    info!("Pinning to CPU {aff} for sector {sector_id}");
    set_cpu_affinity(aff);

    loop {
        match events_rx.try_recv() {
            Ok(EventMsg::ArmSectors {
                params,
                acquisition_id,
            }) => {
                debug!("sector {sector_id} waiting for acquisition {acquisition_id}");
                state = match params.sync {
                    AcquisitionSync::Immediately => RecvState::WaitForNext { acquisition_id },
                    AcquisitionSync::WaitForSync => RecvState::WaitForSync { acquisition_id },
                };
            }
            Ok(EventMsg::Shutdown {}) => break,
            Err(TryRecvError::Disconnected) => break,
            Err(TryRecvError::Empty) => {} // this is fine.
            Ok(_) => {}                    // unknown event in the channel, just continue.
        }

        // no matter the current state, we receive the UDP packets and read them
        // into a buffer:
        match socket.recv_from(&mut buf) {
            Ok((number_of_bytes, _src_addr)) => {
                assert_eq!(number_of_bytes, B::PACKET_SIZE);
            }
            Err(error) if error.kind() == ErrorKind::WouldBlock => continue,
            Err(error) if error.kind() == ErrorKind::Interrupted => continue,
            Err(_) => panic!("recv_from failed"),
        }

        // if we are not armed for acquisition, we don't decode the buffer into
        // a block:
        if state == RecvState::Idle {
            continue;
        }

        match state {
            RecvState::WaitForFrame {
                acquisition_id,
                frame_id,
            } => {
                let block = block_for_bytes(&buf, recycle_blocks_rx, acquisition_id, sector_id);
                if block.get_frame_id() == frame_id {
                    events.send(&EventMsg::AcquisitionStartedSector {
                        sector_id,
                        frame_id: block.get_frame_id(),
                        acquisition_id,
                    });
                    state = RecvState::Receiving { acquisition_id };
                    // send out the first block:
                    let route_info = BlockRouteInfo::new(&block);
                    if assembly_channel.send((block, route_info)).is_err() {
                        events.send(&EventMsg::AcquisitionError {
                            msg: "failed to send to assembly threads".to_string(),
                        });
                        break;
                    }
                }
            }
            RecvState::WaitForNext { acquisition_id } => {
                let block = block_for_bytes(&buf, recycle_blocks_rx, acquisition_id, sector_id);
                state = RecvState::WaitForFrame {
                    acquisition_id,
                    frame_id: block.get_frame_id() + 1,
                }
            }
            RecvState::WaitForSync { acquisition_id } => {
                let block = block_for_bytes(&buf, recycle_blocks_rx, acquisition_id, sector_id);
                if block.sync_is_set() {
                    events.send(&EventMsg::AcquisitionStartedSector {
                        sector_id,
                        frame_id: block.get_frame_id(),
                        acquisition_id,
                    });
                    state = RecvState::Receiving { acquisition_id };
                } else {
                    let block = block_for_bytes(&buf, recycle_blocks_rx, acquisition_id, sector_id);
                    // recycle blocks directly if we don't forward them to the frame
                    // assembly thread:
                    recycle_blocks_tx.send(block).unwrap();

                    // FIXME: this is an opportunity to de-allocate
                    // memory, if we had buffered "too much" beforehand.
                    // Add a "high watermark" queue fill level and only recycle
                    // blocks if we are below.
                }
            }
            RecvState::Receiving { acquisition_id } => {
                let block = block_for_bytes(&buf, recycle_blocks_rx, acquisition_id, sector_id);
                let route_info = BlockRouteInfo::new(&block);
                let l = assembly_channel.len();
                if l > 400 && l % 100 == 0 {
                    warn!("assembly_channel is backed up, len={l} sector={sector_id}");
                }
                if l > 10000 {
                    // make sure we don't consume all available memory:
                    error!("too many blocks in assembly_channel, bailing out");
                }
                if assembly_channel.send((block, route_info)).is_err() {
                    events.send(&EventMsg::AcquisitionError {
                        msg: "failed to send to assembly threads".to_string(),
                    });
                    break;
                }
            }
            RecvState::Idle => {
                // we use a "fake" acquisition id here, because there is no acquisition running:
                let block = block_for_bytes(&buf, recycle_blocks_rx, 0, sector_id);
                recycle_blocks_tx.send(block).unwrap();
            }
        }
    }
}
