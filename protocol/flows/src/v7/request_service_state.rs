use crate::{flow_context::FlowContext, flow_trait::Flow};
use keryx_core::debug;
use keryx_hashes::Hash;
use keryx_p2p_lib::{
    IncomingRoute, Router,
    common::ProtocolError,
    dequeue, make_message,
    pb::{DoneServiceStateChunksMessage, ServiceStateChunkMessage, kaspad_message::Payload},
};
use std::sync::Arc;

/// Rows per chunk: canonical rows are at most 53 bytes, so a chunk stays well under the
/// message size limit.
const SERVICE_STATE_CHUNK_ROWS: usize = 10_000;

pub struct RequestServiceStateFlow {
    ctx: FlowContext,
    router: Arc<Router>,
    incoming_route: IncomingRoute,
}

#[async_trait::async_trait]
impl Flow for RequestServiceStateFlow {
    fn router(&self) -> Option<Arc<Router>> {
        Some(self.router.clone())
    }

    async fn start(&mut self) -> Result<(), ProtocolError> {
        self.start_impl().await
    }
}

impl RequestServiceStateFlow {
    pub fn new(ctx: FlowContext, router: Arc<Router>, incoming_route: IncomingRoute) -> Self {
        Self { ctx, router, incoming_route }
    }

    async fn start_impl(&mut self) -> Result<(), ProtocolError> {
        loop {
            let pruning_point: Hash = dequeue!(self.incoming_route, Payload::RequestServiceState)?.try_into()?;
            self.handle_request(pruning_point).await?
        }
    }

    async fn handle_request(&mut self, pruning_point: Hash) -> Result<(), ProtocolError> {
        let consensus = self.ctx.consensus();
        let session = consensus.session().await;
        let rows = session.async_get_service_state_rows(pruning_point).await?;
        drop(session);
        debug!("Serving {} service-state rows for pruning point {}", rows.len(), pruning_point);
        for chunk in rows.chunks(SERVICE_STATE_CHUNK_ROWS) {
            self.router
                .enqueue(make_message!(Payload::ServiceStateChunk, ServiceStateChunkMessage { rows: chunk.to_vec() }))
                .await?;
        }
        self.router.enqueue(make_message!(Payload::DoneServiceStateChunks, DoneServiceStateChunksMessage {})).await?;
        Ok(())
    }
}
