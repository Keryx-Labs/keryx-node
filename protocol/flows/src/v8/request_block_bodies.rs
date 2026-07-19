use crate::{flow_context::FlowContext, flow_trait::Flow};
use keryx_consensus_core::config::params::POM_PROOF_RETENTION_DEPTH;
use keryx_core::debug;
use keryx_p2p_lib::{
    IncomingRoute, Router, common::ProtocolError, dequeue_with_request_id, make_response, pb::kaspad_message::Payload,
};
use std::sync::Arc;

pub struct HandleBlockBodyRequests {
    ctx: FlowContext,
    router: Arc<Router>,
    incoming_route: IncomingRoute,
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
    pub fn new(ctx: FlowContext, router: Arc<Router>, incoming_route: IncomingRoute) -> Self {
        Self { ctx, router, incoming_route }
    }

    async fn start_impl(&mut self) -> Result<(), ProtocolError> {
        loop {
            let (msg, request_id) = dequeue_with_request_id!(self.incoming_route, Payload::RequestBlockBodies)?;
            let hashes: Vec<_> = msg.try_into()?;
            debug!("got request for {} blocks bodies", hashes.len());
            let session = self.ctx.consensus().unguarded_session();
            let virtual_daa = session.get_virtual_daa_score();

            for hash in hashes {
                // Fetch the full block (not just the body) so the proven PoM tier AND the
                // possession proof travel with the body: a syncing peer needs the tier to validate
                // the coinbase tier-reward split, and the proof so the block it persists can later
                // be relayed to proof-enforcing peers (otherwise it is served "naked" and rejected
                // with "PoM possession proof missing"). Blocks beyond the proof retention window
                // can never be relayed as recent, so ship them without the 200+ KB proof the
                // receiver would only persist for its GC to delete later.
                let block = session.async_get_block(hash).await?;
                self.ctx.warn_if_serving_naked_pom_block(&block);
                let deep = virtual_daa.saturating_sub(block.header.daa_score) > POM_PROOF_RETENTION_DEPTH;
                let mut body_msg: keryx_p2p_lib::pb::BlockBodyMessage = block.transactions.as_ref().into();
                body_msg.pom_tier =
                    block.pom_tier.map(|t| t as u32).or_else(|| block.pom_proof.as_ref().map(|p| p.tier as u32));
                body_msg.pom_proof = if deep { None } else { block.pom_proof.as_ref().map(|p| p.to_wire_bytes()) };
                self.router.enqueue(make_response!(Payload::BlockBody, body_msg, request_id)).await?;
            }
        }
    }
}
