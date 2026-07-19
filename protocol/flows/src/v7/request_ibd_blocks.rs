use crate::{flow_context::FlowContext, flow_trait::Flow};
use keryx_consensus_core::config::params::POM_PROOF_RETENTION_DEPTH;
use keryx_core::debug;
use keryx_p2p_lib::{
    IncomingRoute, Router, common::ProtocolError, convert::header::HeaderFormat, dequeue_with_request_id, make_response,
    pb::kaspad_message::Payload,
};
use std::sync::Arc;

pub struct HandleIbdBlockRequests {
    ctx: FlowContext,
    router: Arc<Router>,
    incoming_route: IncomingRoute,
    header_format: HeaderFormat,
}

#[async_trait::async_trait]
impl Flow for HandleIbdBlockRequests {
    fn router(&self) -> Option<Arc<Router>> {
        Some(self.router.clone())
    }

    async fn start(&mut self) -> Result<(), ProtocolError> {
        self.start_impl().await
    }
}

impl HandleIbdBlockRequests {
    pub fn new(ctx: FlowContext, router: Arc<Router>, incoming_route: IncomingRoute, header_format: HeaderFormat) -> Self {
        Self { ctx, router, incoming_route, header_format }
    }

    async fn start_impl(&mut self) -> Result<(), ProtocolError> {
        loop {
            let (msg, request_id) = dequeue_with_request_id!(self.incoming_route, Payload::RequestIbdBlocks)?;
            let hashes: Vec<_> = msg.try_into()?;

            debug!("got request for {} IBD blocks", hashes.len());
            let session = self.ctx.consensus().unguarded_session();
            let virtual_daa = session.get_virtual_daa_score();

            for hash in hashes {
                let mut block = session.async_get_block(hash).await?;
                self.ctx.warn_if_serving_naked_pom_block(&block);
                // Blocks beyond the proof retention window can never be relayed as recent, so ship
                // them without the 200+ KB proof (tier kept for the coinbase tier-reward split).
                if virtual_daa.saturating_sub(block.header.daa_score) > POM_PROOF_RETENTION_DEPTH {
                    block.pom_tier = block.pom_tier.or_else(|| block.pom_proof.as_ref().map(|p| p.tier));
                    block.pom_proof = None;
                }
                self.router.enqueue(make_response!(Payload::IbdBlock, (self.header_format, &block).into(), request_id)).await?;
            }
        }
    }
}
