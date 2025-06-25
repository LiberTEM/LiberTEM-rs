use std::net::AddrParseError;
use std::net::Ipv4Addr;
use std::net::UdpSocket;

#[derive(thiserror::Error, Debug)]
pub enum K2oConnectionError {
    #[error("failed to parse address {address}: {err}")]
    ParseError {
        address: String,
        err: AddrParseError,
    },

    #[error("could not join multicast group {group_addr} on {local_addr}: {err}")]
    JoinError {
        group_addr: String,
        local_addr: String,
        err: std::io::Error,
    },

    #[error("couldn't bind to {group_w_port}: {err}")]
    BindError {
        group_w_port: String,
        err: std::io::Error,
    },

    #[error("I/O error: {err}")]
    IoError { err: std::io::Error },
}

pub fn create_mcast_socket(
    port: u32,
    group: &str,
    local: &str,
) -> Result<UdpSocket, K2oConnectionError> {
    // Compared to Python version, we create a bound socket here (we don't
    // really need one, but shouldn't hurt)
    let group_w_port = format!("{group}:{port}", group = group, port = port);
    let socket = UdpSocket::bind(&group_w_port)
        .map_err(|err| K2oConnectionError::BindError { group_w_port, err })?;

    // I'm leaving out the step of setting SO_REUSEADDR and SO_REUSEPORT, as I
    // think we don't need them - we have to "route" all packets from/to a single port
    // to the same process anyways, we can't distribute packets to different procesess.

    let group_addr: Ipv4Addr = group
        .parse()
        .map_err(|err| K2oConnectionError::ParseError {
            address: group.to_owned(),
            err,
        })?;
    let local_addr: Ipv4Addr = local
        .parse()
        .map_err(|err| K2oConnectionError::ParseError {
            address: local.to_owned(),
            err,
        })?;

    // socket.connect(group_w_port).unwrap();

    // Compared to the Python version, we are using the address of the local interface here,
    // instead of using the interface index. Let's see if this works out!
    socket
        .join_multicast_v4(&group_addr, &local_addr)
        .map_err(|err| K2oConnectionError::JoinError {
            group_addr: group_addr.to_string(),
            local_addr: local_addr.to_string(),
            err,
        })?;

    Ok(socket)
}
