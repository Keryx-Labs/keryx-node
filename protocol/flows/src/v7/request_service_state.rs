use crate::{flow_context::FlowContext, flow_trait::Flow};
use keryx_core::debug;
use keryx_hashes::Hash;
use keryx_p2p_lib::{
    IncomingRoute, Router,
    common::ProtocolError,
    dequeue, make_message,
    pb::{
        DoneServiceLedgerSnapshotChunksMessage, DoneServiceStateChunksMessage, ServiceLedgerSnapshotChunkMessage,
        ServiceStateChunkMessage, kaspad_message::Payload,
    },
};
use std::sync::Arc;

/// Rows per chunk: canonical rows are at most 53 bytes, so a chunk stays well under the
/// message size limit.
const SERVICE_STATE_CHUNK_ROWS: usize = 10_000;

/// Snapshot bytes per chunk.
const SERVICE_LEDGER_SNAPSHOT_CHUNK_BYTES: usize = 512 * 1024;

/// First protocol version that takes the ledger snapshot instead of the handoff band.
pub const SERVICE_LEDGER_SNAPSHOT_PROTOCOL_VERSION: u32 = 12;

pub struct RequestServiceStateFlow {
    ctx: FlowContext,
    router: Arc<Router>,
    incoming_route: IncomingRoute,
    protocol_version: u32,
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
    pub fn new(ctx: FlowContext, router: Arc<Router>, incoming_route: IncomingRoute, protocol_version: u32) -> Self {
        Self { ctx, router, incoming_route, protocol_version }
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
        // A v12 peer holding the snapshot at this pruning point takes rows up to it only and
        // re-derives everything above from the snapshot; other peers take the handoff band.
        let snapshot = if self.protocol_version >= SERVICE_LEDGER_SNAPSHOT_PROTOCOL_VERSION {
            session.async_get_service_ledger_snapshot(pruning_point).await?
        } else {
            None
        };
        let handoff_daa = if snapshot.is_some() { 0 } else { service_state_handoff_daa(self.protocol_version) };
        let rows = session.async_get_service_state_rows(pruning_point, handoff_daa).await?;
        drop(session);
        debug!("Serving {} service-state rows for pruning point {}", rows.len(), pruning_point);
        for chunk in rows.chunks(SERVICE_STATE_CHUNK_ROWS) {
            self.router
                .enqueue(make_message!(Payload::ServiceStateChunk, ServiceStateChunkMessage { rows: chunk.to_vec() }))
                .await?;
        }
        self.router.enqueue(make_message!(Payload::DoneServiceStateChunks, DoneServiceStateChunksMessage {})).await?;
        if self.protocol_version >= SERVICE_LEDGER_SNAPSHOT_PROTOCOL_VERSION {
            if let Some(bytes) = snapshot {
                debug!("Serving a {}-byte service-ledger snapshot for pruning point {}", bytes.len(), pruning_point);
                for chunk in bytes.chunks(SERVICE_LEDGER_SNAPSHOT_CHUNK_BYTES) {
                    self.router
                        .enqueue(make_message!(
                            Payload::ServiceLedgerSnapshotChunk,
                            ServiceLedgerSnapshotChunkMessage { chunk: chunk.to_vec() }
                        ))
                        .await?;
                }
            }
            self.router
                .enqueue(make_message!(Payload::DoneServiceLedgerSnapshotChunks, DoneServiceLedgerSnapshotChunksMessage {}))
                .await?;
        }
        Ok(())
    }
}

/// Event-daa span above the pruning point a peer of `protocol_version` ships and accepts as
/// service-state handoff rows.
pub fn service_state_handoff_daa(protocol_version: u32) -> u64 {
    if protocol_version >= 11 { u64::MAX } else { keryx_consensus_core::collateral::SERVICE_STATE_HANDOFF_DAA }
}
