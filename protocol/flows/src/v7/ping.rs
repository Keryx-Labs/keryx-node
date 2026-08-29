use crate::{flow_context::FlowContext, flow_trait::Flow};
use keryx_core::{debug, task::tick::TickReason};
use keryx_p2p_lib::{
    IncomingRoute, Router,
    common::ProtocolError,
    dequeue, dequeue_with_timeout, make_message,
    pb::{PingMessage, PongMessage, kaspad_message::Payload},
};
use rand::Rng;
use std::{
    collections::VecDeque,
    sync::{Arc, Weak},
    time::{Duration, Instant},
};

/// Flow for managing a loop receiving pings and responding with pongs
pub struct ReceivePingsFlow {
    _ctx: FlowContext,
    router: Arc<Router>,
    incoming_route: IncomingRoute,
}

#[async_trait::async_trait]
impl Flow for ReceivePingsFlow {
    fn router(&self) -> Option<Arc<Router>> {
        Some(self.router.clone())
    }

    async fn start(&mut self) -> Result<(), ProtocolError> {
        self.start_impl().await
    }
}

impl ReceivePingsFlow {
    pub fn new(ctx: FlowContext, router: Arc<Router>, incoming_route: IncomingRoute) -> Self {
        Self { _ctx: ctx, router, incoming_route }
    }

    async fn start_impl(&mut self) -> Result<(), ProtocolError> {
        loop {
            // We dequeue without a timeout in this case, responding to pings whenever they arrive
            let ping = dequeue!(self.incoming_route, Payload::Ping)?;
            debug!("P2P Flows, got ping request with nonce {}", ping.nonce);
            let pong = make_message!(Payload::Pong, PongMessage { nonce: ping.nonce });
            self.router.enqueue(pong).await?;
        }
    }
}

pub const PING_INTERVAL: Duration = Duration::from_secs(30);

/// How long we wait for a pong before counting a strike.
const PONG_TIMEOUT: Duration = Duration::from_secs(60);

/// Consecutive missed pongs tolerated before the peer is dropped.
///
/// A single miss is not evidence of a dead peer. `Pong` shares the peer's single, unprioritized
/// outgoing queue with block bodies, so an IBD chunk of `IBD_BATCH_SIZE` bodies — each carrying a
/// PoM proof of a few hundred KiB — can legitimately sit in front of it, and the receiving side can
/// be stalled just as long validating them. Dropping the connection on the first miss turned a slow
/// minute into a lost peer, and (through `record_ping_timeout`) into a 24h ban of an honest node.
const MAX_CONSECUTIVE_PING_TIMEOUTS: u32 = 3;

/// Flow for managing a loop sending pings and waiting for pongs
pub struct SendPingsFlow {
    ctx: FlowContext,

    // We use a weak reference to avoid this flow from holding the router during timer waiting if the connection was closed
    router: Weak<Router>,
    peer: String,
    incoming_route: IncomingRoute,
}

#[async_trait::async_trait]
impl Flow for SendPingsFlow {
    fn router(&self) -> Option<Arc<Router>> {
        self.router.upgrade()
    }

    async fn start(&mut self) -> Result<(), ProtocolError> {
        self.start_impl().await
    }
}

impl SendPingsFlow {
    pub fn new(ctx: FlowContext, router: Arc<Router>, incoming_route: IncomingRoute) -> Self {
        let peer = router.to_string();
        Self { ctx, router: Arc::downgrade(&router), peer, incoming_route }
    }

    async fn start_impl(&mut self) -> Result<(), ProtocolError> {
        let mut consecutive_timeouts = 0u32;
        // Nonces still unanswered, with the instant they were sent. Because we now tolerate missed
        // pongs, a pong for an earlier round can legitimately arrive during a later one — it is
        // proof of liveness, not a protocol violation, so we match against the whole window instead
        // of only the latest nonce.
        let mut outstanding: VecDeque<(u64, Instant)> = VecDeque::with_capacity(MAX_CONSECUTIVE_PING_TIMEOUTS as usize);
        loop {
            // Wait `PING_INTERVAL` between pings
            if let TickReason::Shutdown = self.ctx.tick_service.tick(PING_INTERVAL).await {
                return Ok(());
            }

            // Create a fresh random nonce for each ping
            let nonce = rand::thread_rng().r#gen::<u64>();
            let ping = make_message!(Payload::Ping, PingMessage { nonce });
            let Some(router) = self.router.upgrade() else {
                return Err(ProtocolError::ConnectionClosed);
            };
            router.enqueue(ping).await?;
            if outstanding.len() == MAX_CONSECUTIVE_PING_TIMEOUTS as usize {
                outstanding.pop_front();
            }
            outstanding.push_back((nonce, Instant::now()));

            let pong = match dequeue_with_timeout!(self.incoming_route, Payload::Pong, PONG_TIMEOUT) {
                Err(e @ ProtocolError::Timeout(_)) => {
                    consecutive_timeouts += 1;
                    // Only a peer that has never answered a single pong on this connection is
                    // counted towards an automatic ban. That is precisely the phantom-node
                    // signature the ban was written for — connect silently, occupy a slot, never
                    // speak — and it structurally excludes an honest peer that is merely stalled.
                    if router.last_ping_duration() == 0
                        && let Some(cm) = self.ctx.connection_manager()
                    {
                        cm.record_ping_timeout(router.net_address().ip()).await;
                    }
                    if consecutive_timeouts >= MAX_CONSECUTIVE_PING_TIMEOUTS {
                        return Err(e);
                    }
                    debug!(
                        "Ping timeout {}/{} with peer {} (nonce: {}), still giving it a chance",
                        consecutive_timeouts, MAX_CONSECUTIVE_PING_TIMEOUTS, self.peer, nonce
                    );
                    continue;
                }
                Err(e) => return Err(e),
                Ok(p) => p,
            };
            let Some(&(_, sent_at)) = outstanding.iter().find(|(sent_nonce, _)| *sent_nonce == pong.nonce) else {
                return Err(ProtocolError::Other("nonce mismatch between ping and pong"));
            };
            debug!("Successful ping with peer {} (nonce: {})", self.peer, pong.nonce);
            consecutive_timeouts = 0;
            outstanding.clear();
            router.set_last_ping_duration(sent_at.elapsed().as_millis() as u64);
        }
    }
}
