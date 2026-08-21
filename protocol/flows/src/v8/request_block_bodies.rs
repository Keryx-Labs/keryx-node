use crate::{flow_context::FlowContext, flow_trait::Flow};
use keryx_consensus_core::errors::consensus::ConsensusResult;
use keryx_core::debug;
use keryx_p2p_lib::{
    IncomingRoute, Router, common::ProtocolError, dequeue_with_request_id, make_response, pb::kaspad_message::Payload,
};
use std::sync::Arc;

const CONSENSUS_READ_BATCH_SIZE: usize = 32;

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

            // Keep both database reads and potentially-large PoM serialization off the async
            // runtime. A bounded group amortizes spawn_blocking overhead without letting one peer
            // pin the consensus reader or allocate an unbounded response buffer.
            for hash_batch in hashes.chunks(CONSENSUS_READ_BATCH_SIZE) {
                let hashes = hash_batch.to_vec();
                let ctx = self.ctx.clone();
                let body_messages = session
                    .clone()
                    .spawn_blocking(move |c| {
                        hashes
                            .into_iter()
                            .map(|hash| {
                                let block = c.get_block(hash)?;
                                ctx.warn_if_serving_naked_pom_block(&block);
                                let mut body_msg: keryx_p2p_lib::pb::BlockBodyMessage = block.transactions.as_ref().into();
                                body_msg.pom_tier = block
                                    .pom_tier
                                    .map(|tier| tier as u32)
                                    .or_else(|| block.pom_proof.as_ref().map(|proof| proof.tier as u32));
                                body_msg.pom_proof = block.pom_proof.as_ref().map(|proof| proof.to_wire_bytes());
                                Ok(body_msg)
                            })
                            .collect::<ConsensusResult<Vec<_>>>()
                    })
                    .await?;

                for body_msg in body_messages {
                    self.router.enqueue(make_response!(Payload::BlockBody, body_msg, request_id)).await?;
                }
            }
        }
    }
}
