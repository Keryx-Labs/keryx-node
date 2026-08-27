use crate::{flow_context::FlowContext, flow_trait::Flow};
use keryx_core::debug;
use keryx_p2p_lib::{
    IncomingRoute, Router, common::ProtocolError, convert::block::PomWireFormat, dequeue_with_request_id, make_response,
    pb::kaspad_message::Payload,
};
use std::sync::Arc;

pub struct HandleBlockBodyRequests {
    ctx: FlowContext,
    router: Arc<Router>,
    incoming_route: IncomingRoute,
    pom_format: PomWireFormat,
}

#[async_trait::async_trait]
impl Flow for HandleBlockBodyRequests {
    fn router(&self) -> Option<Arc<Router>> {
        Some(self.router.clone())
    }

    async fn start(&mut self) -> Result<(), ProtocolError> {
        self.start_impl().await
    }
}

impl HandleBlockBodyRequests {
    pub fn new(ctx: FlowContext, router: Arc<Router>, incoming_route: IncomingRoute, pom_format: PomWireFormat) -> Self {
        Self { ctx, router, incoming_route, pom_format }
    }

    async fn start_impl(&mut self) -> Result<(), ProtocolError> {
        loop {
            let (msg, request_id) = dequeue_with_request_id!(self.incoming_route, Payload::RequestBlockBodies)?;
            // The requester's own proof horizon (see the field doc in p2p.proto). Absent from
            // pre-v11 peers, in which case nothing is stripped.
            let pom_proof_min_daa = msg.pom_proof_min_daa;
            let hashes: Vec<_> = msg.try_into()?;
            debug!("got request for {} blocks bodies", hashes.len());
            let session = self.ctx.consensus().unguarded_session();

            for hash in hashes {
                // Fetch the full block (not just the body) so the proven PoM tier AND the
                // possession proof travel with the body: a syncing peer needs the tier to validate
                // the coinbase tier-reward split, and the proof so the block it persists can later
                // be relayed to proof-enforcing peers (otherwise it is served "naked" and rejected
                // with "PoM possession proof missing").
                //
                // Never depth-strip against OUR virtual: the receiver's virtual lags behind ours
                // during IBD, so a block "deep" for us can still be recent for the receiver, which
                // would persist it naked and later be rejected by proof-enforcing relay peers. We
                // strip only when the REQUESTER told us its own horizon, which by construction
                // cannot drop a proof it would have kept.
                let block = session.async_get_block(hash).await?;
                let below_horizon = pom_proof_min_daa.is_some_and(|min_daa| block.header.daa_score < min_daa);
                if !below_horizon {
                    self.ctx.warn_if_serving_naked_pom_block(&block);
                }
                let mut body_msg: keryx_p2p_lib::pb::BlockBodyMessage = block.transactions.as_ref().into();
                // The tier is kept even when the proof is dropped: it is consensus-critical for the
                // coinbase tier-reward split.
                body_msg.pom_tier = block.pom_tier.map(|t| t as u32).or_else(|| block.pom_proof.as_ref().map(|p| p.tier as u32));
                if !below_horizon {
                    let pom = self.ctx.encode_pom_proof_cached(self.pom_format, &block);
                    body_msg.pom_proof = pom.legacy;
                    body_msg.pom_proof_deduped = pom.deduped;
                }
                self.router.enqueue(make_response!(Payload::BlockBody, body_msg, request_id)).await?;
            }
        }
    }
}
