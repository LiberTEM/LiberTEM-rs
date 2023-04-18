use std::{io::ErrorKind, time::Duration};

use crossbeam_channel::{Receiver, Sender, TryRecvError};

use crate::{
    block::{BlockRouteInfo, K2Block},
    events::{AcquisitionSync, EventBus, EventMsg, EventReceiver, Events},
    helpers::{set_cpu_affinity, CPU_AFF_DECODE_START},
    net::create_mcast_socket,
};

#[derive(PartialEq)]
enum RecvState {
    /// Receiving UDP packets, but not doing anything with them
    Idle,

    /// Receiving and decoding blocks, but not passing them on downstream to assembly,
    /// goes to `Receiving` state once a block with sync flag has been received
    WaitForSync,

    /// Start receiving and decoding at the next block, regardless of sync status
    WaitForNext,

    /// Recveicing and decoding packets, and sending the decoded packets down
    /// the pipeline
    Receiving,
}

/// receive and decode a block from the specified sector, and send it to the
/// central assembly thread
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

    socket
        .set_read_timeout(Some(Duration::from_millis(10)))
        .unwrap();

    let mut state = RecvState::Idle;

    println!("listening on {local_addr}:{port} for sector {sector_id}");

    set_cpu_affinity(CPU_AFF_DECODE_START + sector_id as usize);

    loop {
        match events_rx.try_recv() {
            Ok(EventMsg::ArmSectors { params }) => {
                println!("sector {sector_id} waiting for acquisition");
                state = match params.sync {
                    AcquisitionSync::Immediately => RecvState::WaitForNext,
                    AcquisitionSync::WaitForSync => RecvState::WaitForSync,
                };
            }
            Ok(EventMsg::Shutdown {}) => break,
            Err(TryRecvError::Disconnected) => break,
            Err(TryRecvError::Empty) => {} // this is fine.
            Ok(_) => {}                    // unknown event in the channel, just continue.
        }

        match socket.recv_from(&mut buf) {
            Ok((number_of_bytes, _src_addr)) => {
                assert_eq!(number_of_bytes, B::PACKET_SIZE);
            }
            Err(error) if error.kind() == ErrorKind::WouldBlock => continue,
            Err(error) if error.kind() == ErrorKind::Interrupted => continue,
            Err(_) => panic!("recv_from failed"),
        }

        if state == RecvState::Idle {
            continue;
        }

        let block: B = {
            let maybe_block = recycle_blocks_rx.try_recv();
            match maybe_block {
                Err(_) => B::from_bytes(&buf, sector_id),
                Ok(mut b) => {
                    b.replace_with(&buf, sector_id);
                    b
                }
            }
        };
        block.validate();

        match state {
            RecvState::WaitForNext => {
                events.send(&EventMsg::AcquisitionStartedSector {
                    sector_id,
                    frame_id: block.get_frame_id(),
                });
                state = RecvState::Receiving;
            }
            RecvState::WaitForSync if block.sync_is_set() => {
                events.send(&EventMsg::AcquisitionStartedSector {
                    sector_id,
                    frame_id: block.get_frame_id(),
                });
                state = RecvState::Receiving;
            }
            RecvState::Receiving => {
                let route_info = BlockRouteInfo::new(&block);
                let l = assembly_channel.len();
                if l > 400 && l % 100 == 0 {
                    println!("assembly_channel is backed up, len={l} sector={sector_id}");
                }
                assembly_channel.send((block, route_info)).unwrap();
            }
            RecvState::Idle | RecvState::WaitForSync => {
                // recycle blocks directly if we don't forward them to the frame
                // assembly thread:
                recycle_blocks_tx.send(block).unwrap();

                // FIXME: this is an opportunity to de-allocate
                // memory, if we had buffered "too much" beforehand.
                // Add a "high watermark" queue fill level and only recycle
                // blocks if we are below.
            }
        }
    }
}
