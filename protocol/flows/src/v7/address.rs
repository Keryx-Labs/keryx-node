use crate::{flow_context::FlowContext, flow_trait::Flow};
use itertools::Itertools;
use keryx_addressmanager::NetAddress;
use keryx_core::{debug, task::tick::TickReason};
use keryx_p2p_lib::{
    IncomingRoute, Router,
    common::ProtocolError,
    dequeue, dequeue_with_timeout, make_message,
    pb::{AddressesMessage, RequestAddressesMessage, kaspad_message::Payload},
};
use keryx_utils::networking::IpAddress;
use rand::{Rng, seq::SliceRandom};
use std::{
    sync::{Arc, Weak},
    time::Duration,
};

/// The maximum number of addresses that are sent in a single kaspa Addresses message.
const MAX_ADDRESSES_SEND: usize = 1000;

/// The maximum number of addresses that can be received in a single kaspa Addresses response.
/// If a peer exceeds this value we consider it a protocol error.
const MAX_ADDRESSES_RECEIVE: usize = 2500;

/// How long we wait for a peer to answer a `RequestAddresses`. Missing the deadline is not fatal —
/// see [`ReceiveAddressesFlow`].
const ADDRESS_RESPONSE_TIMEOUT: Duration = Duration::from_secs(120);

/// Interval between address requests once the local store is comfortably stocked.
const ADDRESS_REQUEST_INTERVAL: Duration = Duration::from_secs(600);

/// Interval used while the store holds fewer than [`WELL_STOCKED_ADDRESS_COUNT`] addresses, i.e.
/// while discovery is the bottleneck.
const ADDRESS_REQUEST_INTERVAL_STARVED: Duration = Duration::from_secs(60);

/// Address-store size above which we consider discovery satisfied and slow the requests down.
const WELL_STOCKED_ADDRESS_COUNT: usize = 64;

/// Upper bound of the random delay before the first request, so that a node coming up with many
/// peers does not ask all of them in the same instant.
const ADDRESS_REQUEST_JITTER: Duration = Duration::from_secs(5);

/// Asks a peer for its known addresses, periodically.
///
/// This used to ask exactly once per connection and then exit, which made the address store grow
/// only when a *new* connection was established — precisely what a node cannot do once the store has
/// drained. Re-requesting on an interval is backward compatible: `SendAddressesFlow` on every
/// deployed node already answers `RequestAddresses` in an unbounded loop, so no protocol change is
/// involved.
///
/// A missed response is also no longer fatal. `Flow::launch` closes the whole router when a flow
/// returns `Err`, so the old 120s timeout on this one message meant that a peer which simply chose
/// not to share addresses cost us an otherwise healthy block-relay connection.
pub struct ReceiveAddressesFlow {
    ctx: FlowContext,
    // Weak, so waiting between requests does not keep a closed connection's router alive
    router: Weak<Router>,
    incoming_route: IncomingRoute,
}

#[async_trait::async_trait]
impl Flow for ReceiveAddressesFlow {
    fn router(&self) -> Option<Arc<Router>> {
        self.router.upgrade()
    }

    async fn start(&mut self) -> Result<(), ProtocolError> {
        self.start_impl().await
    }
}

impl ReceiveAddressesFlow {
    pub fn new(ctx: FlowContext, router: Arc<Router>, incoming_route: IncomingRoute) -> Self {
        Self { ctx, router: Arc::downgrade(&router), incoming_route }
    }

    async fn start_impl(&mut self) -> Result<(), ProtocolError> {
        let jitter = Duration::from_millis(rand::thread_rng().gen_range(0..ADDRESS_REQUEST_JITTER.as_millis() as u64));
        if let TickReason::Shutdown = self.ctx.tick_service.tick(jitter).await {
            return Ok(());
        }
        loop {
            let Some(router) = self.router.upgrade() else {
                return Err(ProtocolError::ConnectionClosed);
            };
            router
                .enqueue(make_message!(
                    Payload::RequestAddresses,
                    RequestAddressesMessage { include_all_subnetworks: false, subnetwork_id: None }
                ))
                .await?;
            drop(router);

            match dequeue_with_timeout!(self.incoming_route, Payload::Addresses, ADDRESS_RESPONSE_TIMEOUT) {
                Ok(msg) => {
                    let address_list: Vec<(IpAddress, u16)> = msg.try_into()?;
                    if address_list.len() > MAX_ADDRESSES_RECEIVE {
                        // Over-sharing is a genuine protocol violation, so this one still disconnects
                        return Err(ProtocolError::OtherOwned(format!(
                            "address count {} exceeded {}",
                            address_list.len(),
                            MAX_ADDRESSES_RECEIVE
                        )));
                    }
                    let mut amgr_lock = self.ctx.address_manager.lock();
                    let learned =
                        address_list.into_iter().filter(|&(ip, port)| amgr_lock.add_address(NetAddress::new(ip, port))).count();
                    drop(amgr_lock);
                    debug!("Learned {} new addresses from peer {:?}", learned, self.router.upgrade().map(|r| r.to_string()));
                }
                Err(ProtocolError::Timeout(_)) => {
                    debug!("Peer did not answer our address request in time, will retry later");
                }
                Err(err) => return Err(err),
            }

            let interval = if self.ctx.address_manager.lock().address_count() < WELL_STOCKED_ADDRESS_COUNT {
                ADDRESS_REQUEST_INTERVAL_STARVED
            } else {
                ADDRESS_REQUEST_INTERVAL
            };
            if let TickReason::Shutdown = self.ctx.tick_service.tick(interval).await {
                return Ok(());
            }
        }
    }
}

pub struct SendAddressesFlow {
    ctx: FlowContext,
    router: Arc<Router>,
    incoming_route: IncomingRoute,
}

#[async_trait::async_trait]
impl Flow for SendAddressesFlow {
    fn router(&self) -> Option<Arc<Router>> {
        Some(self.router.clone())
    }

    async fn start(&mut self) -> Result<(), ProtocolError> {
        self.start_impl().await
    }
}

impl SendAddressesFlow {
    pub fn new(ctx: FlowContext, router: Arc<Router>, incoming_route: IncomingRoute) -> Self {
        Self { ctx, router, incoming_route }
    }

    async fn start_impl(&mut self) -> Result<(), ProtocolError> {
        loop {
            dequeue!(self.incoming_route, Payload::RequestAddresses)?;
            let addresses = self.ctx.address_manager.lock().iterate_addresses().collect_vec();
            let address_list = addresses
                .choose_multiple(&mut rand::thread_rng(), MAX_ADDRESSES_SEND)
                .map(|addr| (addr.ip, addr.port).into())
                .collect();
            self.router.enqueue(make_message!(Payload::Addresses, AddressesMessage { address_list })).await?;
        }
    }
}
