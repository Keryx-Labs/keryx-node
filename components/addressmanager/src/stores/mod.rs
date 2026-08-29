use std::net::{IpAddr, Ipv6Addr};

pub use keryx_utils::networking::NetAddress;

pub(super) mod address_store;
pub(super) mod banned_address_store;

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
pub struct AddressKey(Ipv6Addr, u16);

impl AddressKey {
    pub fn new(ip: Ipv6Addr, port: u16) -> Self {
        Self(ip, port)
    }
}

impl From<NetAddress> for AddressKey {
    fn from(value: NetAddress) -> Self {
        AddressKey::new(
            match value.ip.0 {
                IpAddr::V4(ip) => ip.to_ipv6_mapped(),
                IpAddr::V6(ip) => ip,
            },
            value.port,
        )
    }
}
