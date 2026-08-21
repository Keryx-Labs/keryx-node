use crate::{flow_context::FlowContext, flow_trait::Flow};
use keryx_core::{debug, warn};
use keryx_p2p_lib::{
    IncomingRoute, Router,
    common::ProtocolError,
    convert::header::HeaderFormat,
    dequeue_with_request_id, make_message, make_response,
    pb::{InvRelayBlockMessage, kaspad_message::Payload},
};
use std::sync::Arc;

pub struct HandleRelayBlockRequests {
    ctx: FlowContext,
    router: Arc<Router>,
    incoming_route: IncomingRoute,
    header_format: HeaderFormat,
}

#[async_trait::async_trait]
impl Flow for HandleRelayBlockRequests {
    fn router(&self) -> Option<Arc<Router>> {
        Some(self.router.clone())
    }

    async fn start(&mut self) -> Result<(), ProtocolError> {
        self.start_impl().await
    }
}

impl HandleRelayBlockRequests {
    pub fn new(ctx: FlowContext, router: Arc<Router>, incoming_route: IncomingRoute, header_format: HeaderFormat) -> Self {
        Self { ctx, router, incoming_route, header_format }
    }

    async fn start_impl(&mut self) -> Result<(), ProtocolError> {
        // We begin by sending the current sink to the new peer. This is to help nodes to exchange
        // state even if no new blocks arrive for some reason.
        // Note: in go-keryxd this was done via a dedicated one-time flow.
        self.send_sink().await?;

        loop {
            let (msg, request_id) = dequeue_with_request_id!(self.incoming_route, Payload::RequestRelayBlocks)?;
            let hashes: Vec<_> = msg.try_into()?;

            let session = self.ctx.consensus().unguarded_session();

            for hash in hashes {
                // A peer may request a block for which we still retain the header/status while the
                // body has already been pruned or is temporarily unavailable during IBD recovery.
                // `async_get_block(...)?` used to turn this local storage condition into a protocol
                // error, which sent a Reject and tore down an otherwise healthy P2P connection.
                // Check body availability first and simply decline to serve such requests. The
                // requester can retry the hash from another peer instead of both sides entering a
                // reconnect loop.
                if !session.async_get_block_status(hash).await.is_some_and(|status| status.has_block_body()) {
                    debug!("relay request for {} from peer {} cannot be served: full block body is not available locally", hash, self.router);
                    continue;
                }

                let block = match session.async_get_block(hash).await {
                    Ok(block) => block,
                    Err(err) => {
                        // Status/body stores can race pruning. Treat the race as a local serving
                        // miss, not as peer misbehaviour and not as a reason to kill the connection.
                        warn!(
                            "relay request for {} from peer {} became unavailable while reading the full block: {} — keeping peer connected",
                            hash, self.router, err
                        );
                        continue;
                    }
                };
                self.ctx.warn_if_serving_naked_pom_block(&block);
                self.router.enqueue(make_response!(Payload::Block, (self.header_format, &block).into(), request_id)).await?;
                debug!("relayed block with hash {} to peer {}", hash, self.router);
            }
        }
    }

    async fn send_sink(&mut self) -> Result<(), ProtocolError> {
        let session = self.ctx.consensus().unguarded_session();
        let is_in_transitional_ibd_state = session.async_is_consensus_in_transitional_ibd_state().await;
        drop(session);
        // The sink may miss block body while in a transitional state, hence syncing with others must be prevented in the meanwhile
        if is_in_transitional_ibd_state {
            return Ok(());
        }
        let sink = self.ctx.consensus().unguarded_session().async_get_sink().await;
        if sink == self.ctx.config.genesis.hash {
            return Ok(());
        }
        self.router.enqueue(make_message!(Payload::InvRelayBlock, InvRelayBlockMessage { hash: Some(sink.into()) })).await?;
        Ok(())
    }
}
