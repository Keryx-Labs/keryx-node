use crate::common::ProtocolError;
use crate::core::hub::HubEvent;
use crate::pb::{
    KaspadMessage, p2p_client::P2pClient as ProtoP2pClient, p2p_server::P2p as ProtoP2p, p2p_server::P2pServer as ProtoP2pServer,
};
use crate::{ConnectionInitializer, Router};
use futures::FutureExt;
use keryx_core::{debug, info};
use keryx_utils::networking::NetAddress;
use keryx_utils_tower::{
    counters::TowerConnectionCounters,
    middleware::{BodyExt, CountBytesBody, MapRequestBodyLayer, MapResponseBodyLayer, ServiceBuilder},
};
use std::net::ToSocketAddrs;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use tokio::sync::mpsc::{Sender as MpscSender, channel as mpsc_channel};
use tokio::sync::oneshot::{Sender as OneshotSender, channel as oneshot_channel};
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;
use tonic::transport::{Error as TonicError, Server as TonicServer};
use tonic::{Request, Response, Status as TonicStatus, Streaming};

#[derive(Error, Debug)]
pub enum ConnectionError {
    #[error("missing socket address")]
    NoAddress,

    #[error("{0}")]
    IoError(#[from] std::io::Error),

    #[error("{0}")]
    TonicError(#[from] TonicError),

    #[error("{0}")]
    TonicStatus(#[from] TonicStatus),

    #[error("{0}")]
    ProtocolError(#[from] ProtocolError),
}

/// Maximum P2P decoded gRPC message size to send and receive
const P2P_MAX_MESSAGE_SIZE: usize = 1024 * 1024 * 1024; // 1GB

/// Handles Router creation for both server and client-side new connections
#[derive(Clone)]
pub struct ConnectionHandler {
    /// Cloned on each new connection so that routers can communicate with a central hub
    hub_sender: MpscSender<HubEvent>,
    initializer: Arc<dyn ConnectionInitializer>,
    counters: Arc<TowerConnectionCounters>,
}

impl ConnectionHandler {
    pub(crate) fn new(
        hub_sender: MpscSender<HubEvent>,
        initializer: Arc<dyn ConnectionInitializer>,
        counters: Arc<TowerConnectionCounters>,
    ) -> Self {
        Self { hub_sender, initializer, counters }
    }

    /// Launches a P2P server listener loop
    pub(crate) fn serve(&self, serve_address: NetAddress) -> Result<OneshotSender<()>, ConnectionError> {
        let (termination_sender, termination_receiver) = oneshot_channel::<()>();
        let connection_handler = self.clone();
        info!("P2P Server starting on: {}", serve_address);

        let bytes_tx = self.counters.bytes_tx.clone();
        let bytes_rx = self.counters.bytes_rx.clone();

        tokio::spawn(async move {
            let proto_server = ProtoP2pServer::new(connection_handler)
                .accept_compressed(tonic::codec::CompressionEncoding::Gzip)
                .send_compressed(tonic::codec::CompressionEncoding::Gzip)
                .max_decoding_message_size(P2P_MAX_MESSAGE_SIZE);

            // Inbound connections had no keepalive at all, so a NAT or firewall dropping its state
            // left a half-open connection that only the 30s ping flow would eventually notice —
            // and that flow is the one which disconnects and counts strikes towards a ban.
            // HTTP/2 PING is a transport frame every hyper/tonic peer answers, so this needs no
            // protocol change and interoperates with unpatched nodes.
            let serve_result = TonicServer::builder()
                .tcp_keepalive(Some(Duration::from_millis(Self::keep_alive())))
                .http2_keepalive_interval(Some(Duration::from_millis(Self::http2_keep_alive_interval())))
                .http2_keepalive_timeout(Some(Duration::from_millis(Self::http2_keep_alive_timeout())))
                .layer(MapRequestBodyLayer::new(move |body| CountBytesBody::new(body, bytes_rx.clone()).boxed_unsync()))
                .layer(MapResponseBodyLayer::new(move |body| CountBytesBody::new(body, bytes_tx.clone())))
                .add_service(proto_server)
                .serve_with_shutdown(serve_address.into(), termination_receiver.map(drop))
                .await;

            match serve_result {
                Ok(_) => info!("P2P Server stopped: {}", serve_address),
                Err(err) => panic!("P2P, Server {serve_address} stopped with error: {err:?}"),
            }
        });
        Ok(termination_sender)
    }

    /// Connect to a new peer
    pub(crate) async fn connect(&self, peer_address: String) -> Result<Arc<Router>, ConnectionError> {
        let Some(socket_address) = peer_address.to_socket_addrs()?.next() else {
            return Err(ConnectionError::NoAddress);
        };
        let peer_address = format!("http://{}", peer_address); // Add scheme prefix as required by Tonic

        let channel = tonic::transport::Endpoint::new(peer_address)?
            .timeout(Duration::from_millis(Self::communication_timeout()))
            .connect_timeout(Duration::from_millis(Self::connect_timeout()))
            .tcp_keepalive(Some(Duration::from_millis(Self::keep_alive())))
            .http2_keep_alive_interval(Duration::from_millis(Self::http2_keep_alive_interval()))
            .keep_alive_timeout(Duration::from_millis(Self::http2_keep_alive_timeout()))
            .keep_alive_while_idle(true)
            .connect()
            .await?;

        let channel = ServiceBuilder::new()
            .layer(MapResponseBodyLayer::new(move |body| CountBytesBody::new(body, self.counters.bytes_rx.clone())))
            .layer(MapRequestBodyLayer::new(move |body| CountBytesBody::new(body, self.counters.bytes_tx.clone()).boxed_unsync()))
            .service(channel);

        let mut client = ProtoP2pClient::new(channel)
            .send_compressed(tonic::codec::CompressionEncoding::Gzip)
            .accept_compressed(tonic::codec::CompressionEncoding::Gzip)
            .max_decoding_message_size(P2P_MAX_MESSAGE_SIZE);

        let (outgoing_route, outgoing_receiver) = mpsc_channel(Self::outgoing_network_channel_size());
        let incoming_stream = client.message_stream(ReceiverStream::new(outgoing_receiver)).await?.into_inner();

        let router = Router::new(socket_address, true, self.hub_sender.clone(), incoming_stream, outgoing_route).await;

        // For outbound peers, we perform the initialization as part of the connect logic
        match self.initializer.initialize_connection(router.clone()).await {
            Ok(()) => {
                // Notify the central Hub about the new peer
                self.hub_sender.send(HubEvent::NewPeer(router.clone())).await.expect("hub receiver should never drop before senders");
            }

            Err(err) => {
                router.try_sending_reject_message(&err).await;
                // Ignoring the new router
                router.close().await;
                debug!("P2P, handshake failed for outbound peer {}: {}", router, err);
                return Err(ConnectionError::ProtocolError(err));
            }
        }

        Ok(router)
    }

    /// Connect to a new peer with `retry_attempts` retries and `retry_interval` duration between each attempt
    pub(crate) async fn connect_with_retry(
        &self,
        address: String,
        retry_attempts: u8,
        retry_interval: Duration,
    ) -> Result<Arc<Router>, ConnectionError> {
        let mut counter = 0;
        loop {
            counter += 1;
            match self.connect(address.clone()).await {
                Ok(router) => {
                    debug!("P2P, Client connected, peer: {:?}", address);
                    return Ok(router);
                }
                Err(ConnectionError::ProtocolError(err)) => {
                    // On protocol errors we avoid retrying
                    debug!("P2P, connect retry #{} failed with error {:?}, peer: {:?}, aborting retries", counter, err, address);
                    return Err(ConnectionError::ProtocolError(err));
                }
                Err(err) => {
                    debug!("P2P, connect retry #{} failed with error {:?}, peer: {:?}", counter, err, address);
                    if counter < retry_attempts {
                        // Await `retry_interval` time before retrying
                        tokio::time::sleep(retry_interval).await;
                    } else {
                        debug!("P2P, Client connection retry #{} - all failed", retry_attempts);
                        return Err(err);
                    }
                }
            }
        }
    }

    // TODO: revisit the below constants
    fn outgoing_network_channel_size() -> usize {
        // TODO: this number is taken from go-keryxd and should be re-evaluated
        (1 << 17) + 256
    }

    /// Wait for a request's response headers. Note this is a tower per-request timeout: for the
    /// long-lived `message_stream` streaming RPC it bounds the handshake of the stream, not its
    /// lifetime.
    fn communication_timeout() -> u64 {
        30_000
    }

    fn keep_alive() -> u64 {
        30_000
    }

    /// Interval between HTTP/2 PING frames on an otherwise idle connection.
    fn http2_keep_alive_interval() -> u64 {
        30_000
    }

    /// How long we wait for the peer to answer an HTTP/2 PING before considering the connection dead.
    fn http2_keep_alive_timeout() -> u64 {
        20_000
    }

    /// TCP + HTTP/2 + gRPC connection establishment budget.
    ///
    /// One second is below a single SYN retransmit, so any packet loss — or simply a peer more than
    /// a few hundred milliseconds away — failed deterministically. Each such failure is recorded
    /// against the address, and since `Adaptor::connect_peer` makes exactly one attempt per round,
    /// a handful of unlucky rounds used to write an honest intercontinental peer off entirely.
    fn connect_timeout() -> u64 {
        8_000
    }
}

#[tonic::async_trait]
impl ProtoP2p for ConnectionHandler {
    type MessageStreamStream = Pin<Box<dyn futures::Stream<Item = Result<KaspadMessage, TonicStatus>> + Send + 'static>>;

    /// Handle the new arriving **server** connections
    async fn message_stream(
        &self,
        request: Request<Streaming<KaspadMessage>>,
    ) -> Result<Response<Self::MessageStreamStream>, TonicStatus> {
        let Some(remote_address) = request.remote_addr() else {
            return Err(TonicStatus::new(tonic::Code::InvalidArgument, "Incoming connection opening request has no remote address"));
        };

        // Build the in/out pipes
        let (outgoing_route, outgoing_receiver) = mpsc_channel(Self::outgoing_network_channel_size());
        let incoming_stream = request.into_inner();

        // Build the router object
        let router = Router::new(remote_address, false, self.hub_sender.clone(), incoming_stream, outgoing_route).await;

        // Notify the central Hub about the new peer
        self.hub_sender.send(HubEvent::NewPeer(router)).await.expect("hub receiver should never drop before senders");

        // Give tonic a receiver stream (messages sent to it will be forwarded to the network peer)
        Ok(Response::new(Box::pin(ReceiverStream::new(outgoing_receiver).map(Ok)) as Self::MessageStreamStream))
    }
}
