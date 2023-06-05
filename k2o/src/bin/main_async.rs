extern crate crossbeam;
extern crate crossbeam_channel;

use k2o::block::{K2Block, K2ISBlock};
use std::net::Ipv4Addr;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use k2o::decode::{decode_packet_size, HEADER_SIZE};

async fn create_mcast_socket_tokio(port: u32, group: &str, local: &str) -> tokio::net::UdpSocket {
    // Compared to Python version, we create a bound socket here (we don't
    // really need one, but shouldn't hurt)
    let group_w_port = format!("{group}:{port}", group = group, port = port);
    let socket = tokio::net::UdpSocket::bind(&group_w_port)
        .await
        .unwrap_or_else(|err| {
            panic!("couldn't bind to {}: {:?}", group_w_port, err);
        });

    // I'm leaving out the step of setting SO_REUSEADDR and SO_REUSEPORT, as I
    // think we don't need them - we have to "route" all packets from/to a single port
    // to the same process anyways, we can't distribute packets to different procesess.
    let group_addr: Ipv4Addr = group.parse().expect("failed to parse group address");
    let local_addr: Ipv4Addr = local.parse().expect("failed to parse local address");

    // Compared to the Python version, we are using the address of the local interface here,
    // instead of using the interface index. Let's see if this works out!
    socket
        .join_multicast_v4(group_addr, local_addr)
        .unwrap_or_else(|err| {
            panic!(
                "could not join multicast group {} on local addr {}: {:?}",
                group_addr, local_addr, err
            );
        });

    return socket;
}

pub async fn recv_decode_loop<const PACKET_SIZE: usize, const DECODED_SIZE: usize>(
    channel: &'static mpsc::Sender<K2ISBlock>,
) -> Vec<JoinHandle<()>> {
    let ids = 0..=7;

    let join_handles = ids.map(|id| {
        let port: u32 = 2001 + id;
        // let cpu = (10 + id) as usize;
        return tokio::spawn(async move {
            /*let mut cpu_set = CpuSet::new();
            cpu_set.set(cpu).expect("could not set CPU affinity!");
            nix::sched::sched_setaffinity(Pid::from_raw(0), &cpu_set)
                .expect("could not set CPU affinity!");*/

            let socket = create_mcast_socket_tokio(port, "225.1.1.1", "192.168.10.99").await;

            let mut buf: [u8; PACKET_SIZE] = [0; PACKET_SIZE];
            let mut pls: u64;
            let mut counter: i64 = 0;
            loop {
                pls = 0;
                let (number_of_bytes, _src_addr) =
                    socket.recv_from(&mut buf).await.expect("recv_from failed");
                assert_eq!(number_of_bytes, PACKET_SIZE);
                let block: K2ISBlock = K2ISBlock::from_bytes(&buf, id as u8);
                block.validate();
                channel.send(block).await;
                /*
                for x in &block.payload {
                    pls = pls + (*x as u64);
                }
                counter += 1;
                if counter % 1000 == 0 {
                    println!("counter={} pls={} frame_id={} port={}", counter, pls, block.frame_id, port);
                }
                */
            }
            // return (); // not really, because it will never execute because of the infinite loop above
        });
    });
    return Vec::from_iter(join_handles);
}

///
/// Receive a single packet and read the PACKET_SIZE from it
///
async fn recv_and_get_packet_size() -> u32 {
    // finding the PACKET_SIZE: we just have a look at the first packet from the first sector
    const PORT: u32 = 2001;
    let socket = create_mcast_socket_tokio(PORT, "225.1.1.1", "192.168.10.99").await;
    let mut buf: [u8; HEADER_SIZE] = [0; HEADER_SIZE];
    let (number_of_bytes, _src_addr) = socket.recv_from(&mut buf).await.expect("recv_from failed");
    assert_eq!(number_of_bytes, HEADER_SIZE);
    return decode_packet_size(&buf);
}

#[tokio::main]
pub async fn main() {
    let packet_size = recv_and_get_packet_size().await;

    /*
    println!("packet_size = {}", packet_size);

    let (tx, mut rx) = mpsc::channel(1024);
    let (tx_is, mut rx_is) = mpsc::channel::<K2ISBlock<0x5758, 14880>>(1024);

    if packet_size == 0x5758 { // IS mode
        let join_handles: Vec<JoinHandle<_>> = recv_decode_loop::<0x5758, 14880>(&tx_is).await;
        while let Some(block) = rx_is.recv().await {

        }
        futures::future::join_all(join_handles).await;
    } else if packet_size == 0xc028 { // Summit mode
        // FIXME: this is proably not correct - it looks like this mode uses a different encoding
        // from the block header, the payload size per block is 32*768 = 24576
        //
        // From the PACKET_SIZE, if we assume 12bit encoding like in IS mode,
        // we have DECODED_SIZE = (PACKET_SIZE-HEADER_SIZE)*2/3 = (0xc028-40)*2/3 = 32768
        //
        // If we instead assume u16 "encoding", we get (0xc028-40)/2 = 24576 as expected
        let join_handles = recv_decode_loop::<0xc028, 24576>(&tx).await;
        futures::future::join_all(join_handles).await;
    } else {
        panic!("Unknown packet size, aborting")
    }
    */
}
