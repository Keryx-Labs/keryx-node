use crate::{flow_context::FlowContext, flow_trait::Flow};
use keryx_core::debug;
use keryx_p2p_lib::{
    IncomingRoute, Router,
    common::ProtocolError,
    convert::{block::PomWireFormat, header::HeaderFormat},
    dequeue_with_request_id, make_response,
    pb::kaspad_message::Payload,
};
use std::sync::Arc;

pub struct HandleIbdBlockRequests {
    ctx: FlowContext,
    router: Arc<Router>,
    incoming_route: IncomingRoute,
    header_format: HeaderFormat,
    pom_format: PomWireFormat,
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
    pub fn new(
        ctx: FlowContext,
        router: Arc<Router>,
        incoming_route: IncomingRoute,
        header_format: HeaderFormat,
        pom_format: PomWireFormat,
    ) -> Self {
        Self { ctx, router, incoming_route, header_format, pom_format }
    }

    async fn start_impl(&mut self) -> Result<(), ProtocolError> {
        loop {
            let (msg, request_id) = dequeue_with_request_id!(self.incoming_route, Payload::RequestIbdBlocks)?;
            // The requester's own proof horizon (see the field doc in p2p.proto). Absent from
            // pre-v11 peers, in which case nothing is stripped.
            let pom_proof_min_daa = msg.pom_proof_min_daa;
            let hashes: Vec<_> = msg.try_into()?;

            debug!("got request for {} IBD blocks", hashes.len());
            let session = self.ctx.consensus().unguarded_session();

            for hash in hashes {
                let mut block = session.async_get_block(hash).await?;
                // Never depth-strip against OUR virtual: the receiver's virtual lags behind ours
                // during IBD, so a block that is "deep" for us can still be recent for the receiver
                // — it would persist the block naked and later be rejected by proof-enforcing relay
                // peers. Proofs we no longer have (GC'd) are absent anyway.
                //
                // We do strip when the REQUESTER states its own horizon: that is its policy applied
                // at the source, so it can never drop a proof the receiver would have kept — and it
                // saves shipping ~440 KB that the receiver was going to discard on arrival.
                if pom_proof_min_daa.is_some_and(|min_daa| block.header.daa_score < min_daa) {
                    // Keep the tier: it is consensus-critical for the coinbase tier-reward split.
                    block.pom_tier = block.pom_tier.or_else(|| block.pom_proof.as_ref().map(|p| p.tier));
                    block.pom_proof = None;
                } else {
                    self.ctx.warn_if_serving_naked_pom_block(&block);
                }
                self.router
                    .enqueue(make_response!(
                        Payload::IbdBlock,
                        (self.header_format, self.ctx.encode_pom_proof_cached(self.pom_format, &block), &block).into(),
                        request_id
                    ))
                    .await?;
            }
        }
    }
}
