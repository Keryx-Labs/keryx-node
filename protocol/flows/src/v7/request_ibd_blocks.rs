use crate::{flow_context::FlowContext, flow_trait::Flow};
use keryx_consensus_core::errors::consensus::ConsensusResult;
use keryx_core::debug;
use keryx_p2p_lib::{
    IncomingRoute, Router, common::ProtocolError, convert::header::HeaderFormat, dequeue_with_request_id, make_response,
    pb::kaspad_message::Payload,
};
use std::sync::Arc;

const CONSENSUS_READ_BATCH_SIZE: usize = 32;

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

            // Amortize the consensus read guard and spawn_blocking transition across a bounded
            // number of blocks instead of scheduling one blocking task per requested hash.
            for hash_batch in hashes.chunks(CONSENSUS_READ_BATCH_SIZE) {
                let hashes = hash_batch.to_vec();
                let blocks = session
                    .clone()
                    .spawn_blocking(move |c| {
                        hashes
                            .into_iter()
                            .map(|hash| c.get_block(hash))
                            .collect::<ConsensusResult<Vec<_>>>()
                    })
                    .await?;

                for block in blocks {
                    self.ctx.warn_if_serving_naked_pom_block(&block);
                    // Always ship the possession proof when we have it. Depth-stripping it against OUR
                    // virtual is unsound: the receiver's virtual lags behind ours during IBD, so a block
                    // that is "deep" for us can still be recent for the receiver — it would persist the
                    // block naked and later be rejected by proof-enforcing relay peers (the 2026-07-31
                    // naked-band wedge). Proofs we no longer have (GC'd) are absent anyway.
                    self.router
                        .enqueue(make_response!(Payload::IbdBlock, (self.header_format, &block).into(), request_id))
                        .await?;
                }
            }
        }
    }
}
