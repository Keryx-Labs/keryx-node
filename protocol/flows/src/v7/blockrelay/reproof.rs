//! Bounded per-peer proof repair, independent of live block inventory processing.
use crate::{
    flow_context::{FlowContext, RequestScope},
    flow_trait::Flow,
};
use keryx_consensus_core::{block::Block, blockstatus::BlockStatus};
use keryx_core::{debug, info};
use keryx_hashes::Hash;
use keryx_p2p_lib::{
    IncomingRoute, Router,
    common::ProtocolError,
    convert::header::{HeaderFormat, Versioned},
    dequeue_with_timeout, make_request,
    pb::{RequestRelayBlocksMessage, kaspad_message::Payload},
};
use std::{sync::Arc, time::Duration};

const POM_REPROOF_MIN_INTERVAL: Duration = Duration::from_secs(1);

/// Exactly one repair flow per peer. One in-flight block, a capacity-one response route,
/// and the existing bounded/deduplicated global queue provide backpressure. There are no
/// per-candidate tasks. Live relay flows never await this flow or consume its responses.
pub struct PomReproofFlow {
    ctx: FlowContext,
    router: Arc<Router>,
    msg_route: IncomingRoute,
    header_format: HeaderFormat,
}

#[async_trait::async_trait]
impl Flow for PomReproofFlow {
    fn router(&self) -> Option<Arc<Router>> {
        Some(self.router.clone())
    }
    async fn start(&mut self) -> Result<(), ProtocolError> {
        self.start_impl().await
    }
}

impl PomReproofFlow {
    pub fn new(ctx: FlowContext, router: Arc<Router>, header_format: HeaderFormat) -> Self {
        let msg_route = router.subscribe_with_capacity(vec![], 1);
        Self { ctx, router, msg_route, header_format }
    }

    async fn start_impl(&mut self) -> Result<(), ProtocolError> {
        let mut interval = tokio::time::interval_at(tokio::time::Instant::now() + POM_REPROOF_MIN_INTERVAL, POM_REPROOF_MIN_INTERVAL);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            // An idle flow must terminate when Router::close drops its route sender.
            tokio::select! {
                msg = self.msg_route.recv() => return Err(if msg.is_none() {
                    ProtocolError::ConnectionClosed
                } else {
                    ProtocolError::Other("unsolicited PoM repair response")
                }),
                _ = interval.tick() => {}
            }
            let Some(hash) = self.ctx.take_pom_reproof_candidates(1).into_iter().next() else { continue };
            let session = self.ctx.consensus().unguarded_session();
            if session.async_get_block_status(hash).await == Some(BlockStatus::StatusHeaderOnly) {
                self.ctx.abandon_pom_reproof(hash);
                debug!("PoM re-proof: {} is header-only here, dropped from the re-fetch queue", hash);
                continue;
            }
            drop(session);
            if let Err(e) = self.try_readopt_pom_proof(hash).await {
                self.ctx.enqueue_pom_reproof(hash);
                // Never reuse a response route after a timeout: a late response could belong
                // to the previous request. Flow::launch closes this connection on error.
                return Err(e);
            }
        }
    }

    /// Re-fetches a block whose possession proof is missing locally and adopts the proof it
    /// carries. Every path that fails to obtain a usable proof re-queues the hash at the tail, so
    /// another peer (or the same one, once healed itself) gets a turn.
    ///
    /// Two cases: the block is stored but naked (graft the proof onto it), or it was never
    /// inserted because the proof-required relay path skipped it (submit the proof-carrying block
    /// through the enforcing path — there is no stored header to graft onto).
    async fn try_readopt_pom_proof(&mut self, requested_hash: Hash) -> Result<(), ProtocolError> {
        let Some((block, request_scope)) = self.request_block(requested_hash).await? else {
            // Another flow owns this request; the candidate would be lost when that scope closes.
            self.ctx.enqueue_pom_reproof(requested_hash);
            return Ok(());
        };
        request_scope.report_obtained();
        let Some(proof) = block.pom_proof else {
            self.ctx.enqueue_pom_reproof(requested_hash);
            debug!("PoM re-proof: peer {} also serves {} without its proof — re-queued", self.router, requested_hash);
            return Ok(());
        };
        let session = self.ctx.consensus().unguarded_session();
        // Adoption grafts the proof onto a block we already store. A block skipped by the
        // proof-required relay path was never inserted, so there is no stored header to graft
        // onto — submit the proof-carrying block itself through the enforcing path instead.
        if session.async_get_block_status(requested_hash).await.is_none() {
            let block =
                Block { header: block.header, transactions: block.transactions, pom_proof: Some(proof), pom_tier: block.pom_tier };
            match session.validate_and_insert_block(block).block_task.await {
                Ok(_) => info!("PoM re-proof: inserted {} with the proof served by peer {}", requested_hash, self.router),
                Err(e) => {
                    self.ctx.enqueue_pom_reproof(requested_hash);
                    debug!(
                        "PoM re-proof: proof-carrying {} from peer {} still rejected: {} — re-queued",
                        requested_hash, self.router, e
                    );
                }
            }
            return Ok(());
        }
        match session.async_adopt_pom_proof(requested_hash, (*proof).clone()).await {
            Ok(true) => info!("PoM re-proof: adopted the possession proof of {} from peer {}", requested_hash, self.router),
            Ok(false) => {}
            Err(e) => {
                self.ctx.enqueue_pom_reproof(requested_hash);
                debug!("PoM re-proof: proof of {} from peer {} not adopted: {} — re-queued", requested_hash, self.router, e);
            }
        }
        Ok(())
    }

    async fn request_block(&mut self, requested_hash: Hash) -> Result<Option<(Block, RequestScope<Hash>)>, ProtocolError> {
        let Some(request_scope) = self.ctx.try_adding_block_request(requested_hash) else { return Ok(None) };
        self.router
            .enqueue(make_request!(
                Payload::RequestRelayBlocks,
                RequestRelayBlocksMessage { hashes: vec![requested_hash.into()] },
                self.msg_route.id()
            ))
            .await?;
        let msg = dequeue_with_timeout!(self.msg_route, Payload::Block)?;
        let block: Block = Versioned(self.header_format, msg).try_into()?;
        if block.hash() != requested_hash {
            return Err(ProtocolError::OtherOwned(format!(
                "requested PoM repair block {} but got block {}",
                requested_hash,
                block.hash()
            )));
        }
        Ok(Some((block, request_scope)))
    }
}
