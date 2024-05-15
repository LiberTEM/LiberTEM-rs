use std::net::Ipv4Addr;
use std::net::UdpSocket;

pub fn create_mcast_socket(port: u32, group: &str, local: &str) -> UdpSocket {
    // Compared to Python version, we create a bound socket here (we don't
    // really need one, but shouldn't hurt)
    let group_w_port = format!("{group}:{port}", group = group, port = port);
    let socket = UdpSocket::bind(&group_w_port).unwrap_or_else(|_err| {
        let error = format!("couldn't bind to {}", group_w_port);
        panic!("{}", error);
    });

    // I'm leaving out the step of setting SO_REUSEADDR and SO_REUSEPORT, as I
    // think we don't need them - we have to "route" all packets from/to a single port
    // to the same process anyways, we can't distribute packets to different procesess.

    let group_addr: Ipv4Addr = group.parse().expect("failed to parse group address");
    let local_addr: Ipv4Addr = local.parse().expect("failed to parse local address");

    // socket.connect(group_w_port).unwrap();

    // Compared to the Python version, we are using the address of the local interface here,
    // instead of using the interface index. Let's see if this works out!
    socket
        .join_multicast_v4(&group_addr, &local_addr)
        .expect("could not join multicast group");

    socket
}
