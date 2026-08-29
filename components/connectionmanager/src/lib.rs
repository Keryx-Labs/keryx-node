use std::{
    cmp::{Reverse, min},
    collections::{HashMap, HashSet},
    net::{IpAddr, SocketAddr, ToSocketAddrs},
    sync::Arc,
    time::{Duration, Instant, SystemTime},
};

use duration_string::DurationString;
use futures_util::future::{join_all, try_join_all};
use itertools::Itertools;
use keryx_addressmanager::{AddressManager, NetAddress};
use keryx_core::{debug, info, warn};
use keryx_p2p_lib::{ConnectionError, Peer, common::ProtocolError};
use keryx_utils::{networking::PrefixBucket, triggers::SingleTrigger};
use parking_lot::Mutex as ParkingLotMutex;
use rand::{seq::SliceRandom, thread_rng};
use tokio::{
    select,
    sync::{
        Mutex as TokioMutex,
        mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel},
    },
};

/// Ping-timeout strikes tolerated within [`PING_TIMEOUT_WINDOW`] before an IP is banned.
///
/// Only peers that have *never* answered a pong on the connection reach this counter (see
/// `SendPingsFlow`), which is the phantom-node signature the mechanism was written for. An honest
/// peer that stalls under load is therefore structurally excluded.
const PING_TIMEOUT_BAN_THRESHOLD: u32 = 5;
const PING_TIMEOUT_WINDOW: Duration = Duration::from_secs(1800); // 30 minutes

/// Cadence of the connection loop, by how far we are from the outbound target. A node in cold start
/// or recovery cannot afford a 30s wait between rounds; a satisfied node should stay quiet. The
/// per-address dial cooldown is what makes the fast cadence safe -- without it a 2s loop would just
/// hammer the same dead addresses.
const LOOP_INTERVAL_STARVED: Duration = Duration::from_secs(2);
const LOOP_INTERVAL_BELOW_TARGET: Duration = Duration::from_secs(10);
const LOOP_INTERVAL_SATISFIED: Duration = Duration::from_secs(30);

/// Max dial rounds per loop iteration. Without a cap a single iteration could walk the whole
/// address store (up to 4096 entries) in serialized rounds, blocking the event loop and -- worse --
/// postponing the DNS seeding at the end of the iteration, which is the only recovery path when the
/// store is full of dead addresses.
const MAX_DIAL_ROUNDS_PER_ITERATION: usize = 4;

/// Per-address dial backoff: `30s * 2^min(consecutive_failures - 1, 4)`, i.e. 30s .. 8min.
const DIAL_COOLDOWN_BASE: Duration = Duration::from_secs(30);
const MAX_ACCOUNTABLE_DIAL_FAILURES: u32 = 5;

/// How often the peering health line is logged.
const HEALTH_LOG_INTERVAL: Duration = Duration::from_secs(60);

/// Minimum spacing between two DNS seeder queries, whatever the loop cadence.
const DNS_SEED_MIN_INTERVAL: Duration = Duration::from_secs(30);

/// How long an inbound peer may stay connected without ever answering a pong before it is treated
/// as a silent squatter for eviction purposes. Comfortably above two ping intervals.
const INBOUND_SILENCE_GRACE: Duration = Duration::from_secs(300);

/// How long an outbound connection must last before its dial history is forgiven.
///
/// A dial that completes is not yet evidence of a usable peer: the session can die seconds later on
/// a flow error — a peer asking us for a block body we only have the header of, for instance, which
/// tears down the whole router. Until a connection clears this bar it keeps its exponential dial
/// backoff, so a flapping peer is retried at 30s, 1m, 2m, ... instead of on every loop iteration.
const STABLE_CONNECTION_GRACE: Duration = Duration::from_secs(60);

/// How long a protocol version learned from a handshake is trusted for outbound preference.
/// The map is keyed by remote-supplied IPs, so it needs a bound like the other per-IP trackers.
const KNOWN_PROTOCOL_TTL: Duration = Duration::from_secs(86400); // 24 hours
/// Hard ceiling on the version map, mirroring `MAX_ADDRESSES` in the address store.
const MAX_KNOWN_PROTOCOL_ENTRIES: usize = 4096;

/// The first protocol version that encodes a v4 PoM proof as a compact multiproof. Older peers serve
/// the same proof in the legacy encoding — correct, just several times heavier on the wire.
const COMPACT_POM_PROTOCOL_VERSION: u32 = 11;
/// How many candidates to draw per free outbound slot before ranking them by protocol preference.
const CANDIDATE_POOL_FACTOR: usize = 3;

fn canonical_ip(ip: IpAddr) -> IpAddr {
    match ip {
        IpAddr::V6(ip) => ip.to_ipv4_mapped().map(IpAddr::V4).unwrap_or(IpAddr::V6(ip)),
        ip => ip,
    }
}

fn is_fixed_seed_ip(seeders: &[&str], ip: IpAddr) -> bool {
    let ip = canonical_ip(ip);
    seeders.iter().filter_map(|seeder| seeder.parse().ok()).any(|seeder| canonical_ip(seeder) == ip)
}

/// Eviction rank for an inbound peer: higher is evicted first.
///
/// 1. A peer that has never answered a pong although it has been connected for longer than two ping
///    intervals is a silent squatter — exactly the phantom-node shape — and goes first.
/// 2. Then the newest connections, so a burst of arrivals cannot displace established peers.
/// 3. Ties are broken by how represented the peer's /16 (or /64) prefix already is among our inbound
///    peers, which keeps a single network from filling the slots.
fn inbound_eviction_rank(time_connected_ms: u64, last_ping_duration: u64, prefix_redundancy: usize) -> (bool, Reverse<u64>, usize) {
    let silent = last_ping_duration == 0 && time_connected_ms > INBOUND_SILENCE_GRACE.as_millis() as u64;
    (silent, Reverse(time_connected_ms), prefix_redundancy)
}

/// Exponential dial backoff after `failures` consecutive failed dials: 30s, 1m, 2m, 4m, 8m.
fn dial_backoff(failures: u32) -> Duration {
    DIAL_COOLDOWN_BASE * 2u32.pow(min(failures.max(1), MAX_ACCOUNTABLE_DIAL_FAILURES) - 1)
}

/// Per-address dial backoff.
///
/// Its own type because the part that matters is counter-intuitive and regressed once: a
/// *successful* dial also takes a hold, and only a connection that lasts clears it. Reaching that
/// behaviour through a live [`ConnectionManager`] would need a running p2p adaptor, so it lives here
/// where it can be tested directly.
#[derive(Default)]
struct DialCooldown {
    entries: HashMap<NetAddress, (Instant, u32)>,
}

impl DialCooldown {
    fn is_cooling_down(&self, address: NetAddress, now: Instant) -> bool {
        self.entries.get(&address).is_some_and(|(until, _)| *until > now)
    }

    /// Escalates the backoff for `address`, one step per call.
    fn back_off(&mut self, address: NetAddress, now: Instant) {
        let failures = self.entries.get(&address).map_or(1, |(_, failures)| failures.saturating_add(1));
        self.entries.insert(address, (now + dial_backoff(failures), failures));
    }

    /// Forgets the history of `address`: its next dial is unrestricted and its backoff restarts from
    /// [`DIAL_COOLDOWN_BASE`].
    fn forgive(&mut self, address: NetAddress) {
        self.entries.remove(&address);
    }

    /// Drops elapsed holds. The map is keyed by remote-supplied addresses, so it needs a bound.
    fn gc(&mut self, now: Instant) {
        self.entries.retain(|_, (until, _)| *until > now);
    }

    fn len(&self) -> usize {
        self.entries.len()
    }
}

struct PingTimeoutRecord {
    count: u32,
    window_start: Instant,
}

pub struct ConnectionManager {
    p2p_adaptor: Arc<keryx_p2p_lib::Adaptor>,
    outbound_target: usize,
    inbound_limit: usize,
    dns_seeders: &'static [&'static str],
    ban_exempt_seeders: &'static [&'static str],
    default_port: u16,
    address_manager: Arc<ParkingLotMutex<AddressManager>>,
    connection_requests: TokioMutex<HashMap<SocketAddr, ConnectionRequest>>,
    force_next_iteration: UnboundedSender<()>,
    shutdown_signal: SingleTrigger,
    ping_timeout_tracker: ParkingLotMutex<HashMap<IpAddr, PingTimeoutRecord>>,
    dial_cooldown: ParkingLotMutex<DialCooldown>,
    last_health_log: ParkingLotMutex<Instant>,
    last_dns_seed: ParkingLotMutex<Option<Instant>>,
    /// Last known protocol version per IP, learned from successful handshakes, with the instant it
    /// was learned. Used to prefer v11 peers while keeping v10 as fallback; expired by
    /// `KNOWN_PROTOCOL_TTL` so a peer that upgrades is not written off forever.
    known_protocols: ParkingLotMutex<HashMap<IpAddr, (u32, Instant)>>,
}

#[derive(Clone, Debug)]
struct ConnectionRequest {
    next_attempt: SystemTime,
    is_permanent: bool,
    attempts: u32,
}

impl ConnectionRequest {
    fn new(is_permanent: bool) -> Self {
        Self { next_attempt: SystemTime::now(), is_permanent, attempts: 0 }
    }
}

impl ConnectionManager {
    pub fn new(
        p2p_adaptor: Arc<keryx_p2p_lib::Adaptor>,
        outbound_target: usize,
        inbound_limit: usize,
        dns_seeders: &'static [&'static str],
        ban_exempt_seeders: &'static [&'static str],
        default_port: u16,
        address_manager: Arc<ParkingLotMutex<AddressManager>>,
    ) -> Arc<Self> {
        let (tx, rx) = unbounded_channel::<()>();
        let manager = Arc::new(Self {
            p2p_adaptor,
            outbound_target,
            inbound_limit,
            address_manager,
            connection_requests: Default::default(),
            force_next_iteration: tx,
            shutdown_signal: SingleTrigger::new(),
            dns_seeders,
            ban_exempt_seeders,
            default_port,
            ping_timeout_tracker: Default::default(),
            dial_cooldown: Default::default(),
            last_health_log: ParkingLotMutex::new(Instant::now()),
            last_dns_seed: Default::default(),
            known_protocols: Default::default(),
        });
        manager.clone().start_event_loop(rx);
        manager.trigger_iteration();
        manager
    }

    /// Records the protocol version actually negotiated with `ip`. Called from the handshake path;
    /// best-effort and never blocks the flow.
    pub fn record_peer_protocol_version(&self, ip: IpAddr, version: u32) {
        let ip = canonical_ip(ip);
        let now = Instant::now();
        let mut known = self.known_protocols.lock();
        if known.len() >= MAX_KNOWN_PROTOCOL_ENTRIES && !known.contains_key(&ip) {
            // Full and this is a new IP: drop what is already expired, and if that frees nothing,
            // drop the oldest entry. Either way the map cannot be grown without bound by a peer
            // cycling through addresses.
            known.retain(|_, (_, learned_at)| now.duration_since(*learned_at) <= KNOWN_PROTOCOL_TTL);
            if known.len() >= MAX_KNOWN_PROTOCOL_ENTRIES
                && let Some(oldest) = known.iter().min_by_key(|(_, (_, learned_at))| *learned_at).map(|(ip, _)| *ip)
            {
                known.remove(&oldest);
            }
        }
        known.insert(ip, (version, now));
    }

    /// Dial preference, lowest first. Every version here serves a valid v4 PoM proof; v11 differs
    /// only in encoding it as a compact multiproof, so this ranks peers by what a block costs on the
    /// wire, not by what they are capable of. It is therefore a preference over a candidate pool and
    /// **not** a filter: an older peer is still dialed when nothing better is on offer, which is what
    /// keeps the network from partitioning along the version line while v11 adoption is partial.
    fn peer_protocol_preference(ip: IpAddr, known: &HashMap<IpAddr, (u32, Instant)>, now: Instant) -> u8 {
        let version = known
            .get(&canonical_ip(ip))
            .filter(|(_, learned_at)| now.duration_since(*learned_at) <= KNOWN_PROTOCOL_TTL)
            .map(|(version, _)| *version);
        match version {
            Some(v) if v >= COMPACT_POM_PROTOCOL_VERSION => 0, // compact multiproof encoding
            None => 1,                                         // never met, or the observation expired
            Some(10) => 2,                                     // legacy encoding, deliberately still dialed
            Some(_) => 3,                                      // older
        }
    }

    fn start_event_loop(self: Arc<Self>, mut rx: UnboundedReceiver<()>) {
        tokio::spawn(async move {
            loop {
                if self.shutdown_signal.trigger.is_triggered() {
                    break;
                }
                let delay = self.next_iteration_delay();
                select! {
                    _ = rx.recv() => self.clone().handle_event().await,
                    _ = tokio::time::sleep(delay) => self.clone().handle_event().await,
                    _ = self.shutdown_signal.listener.clone() => break,
                }
            }
            debug!("Connection manager event loop exiting");
        });
    }

    /// Time to wait before the next unforced iteration.
    fn next_iteration_delay(&self) -> Duration {
        let outbound = self.p2p_adaptor.active_peers().into_iter().filter(|peer| peer.is_outbound()).count();
        if outbound * 2 < self.outbound_target {
            LOOP_INTERVAL_STARVED
        } else if outbound < self.outbound_target {
            LOOP_INTERVAL_BELOW_TARGET
        } else {
            LOOP_INTERVAL_SATISFIED
        }
    }

    /// Requests an immediate extra iteration of the connection loop.
    fn trigger_iteration(&self) {
        if self.force_next_iteration.send(()).is_err() {
            debug!("Connection manager event loop is gone, cannot force an iteration");
        }
    }

    async fn handle_event(self: Arc<Self>) {
        debug!("Starting connection loop iteration");
        let peers = self.p2p_adaptor.active_peers();
        let peer_by_address: HashMap<SocketAddr, Peer> = peers.into_iter().map(|peer| (peer.net_address(), peer)).collect();

        self.forgive_stable_peers(&peer_by_address);
        self.handle_connection_requests(&peer_by_address).await;
        self.handle_outbound_connections(&peer_by_address).await;
        self.handle_inbound_connections(&peer_by_address).await;
        self.gc_transient_state();
        self.log_health(&peer_by_address);
    }

    /// Drops expired entries from the per-IP trackers. Both are keyed by remote-controlled IPs, so
    /// without this they grow without bound over the lifetime of the process.
    fn gc_transient_state(&self) {
        self.ping_timeout_tracker.lock().retain(|_, record| record.window_start.elapsed() <= PING_TIMEOUT_WINDOW);
        self.dial_cooldown.lock().gc(Instant::now());
        let now = Instant::now();
        self.known_protocols.lock().retain(|_, (_, learned_at)| now.duration_since(*learned_at) <= KNOWN_PROTOCOL_TTL);
    }

    /// One line per minute answering "which peering problem do I have?": a small address store means
    /// discovery is the bottleneck, a healthy store with low outbound means dials are failing, and
    /// `advertised=NONE` or a stuck `inbound=0` means this node is unreachable from the outside.
    fn log_health(&self, peer_by_address: &HashMap<SocketAddr, Peer>) {
        {
            let mut last = self.last_health_log.lock();
            if last.elapsed() < HEALTH_LOG_INTERVAL {
                return;
            }
            *last = Instant::now();
        }
        let outbound = peer_by_address.values().filter(|peer| peer.is_outbound()).count();
        let inbound = peer_by_address.len() - outbound;
        let (known, banned, advertised) = {
            let mut amgr = self.address_manager.lock();
            let banned = amgr.get_all_banned_addresses().len();
            let advertised = amgr.best_local_address();
            (amgr.address_count(), banned, advertised)
        };
        let cooling = self.dial_cooldown.lock().len();
        let (v11_out, v10_out, other_out) = {
            let mut v11 = 0usize;
            let mut v10 = 0usize;
            let mut other = 0usize;
            for peer in peer_by_address.values().filter(|p| p.is_outbound()) {
                let props = peer.properties();
                match props.protocol_version {
                    v if v >= 11 => v11 += 1,
                    10 => v10 += 1,
                    _ => other += 1,
                }
            }
            (v11, v10, other)
        };
        info!(
            "P2P health: outbound={}/{} (v11={} v10={} other={}) inbound={}/{} | addresses={} cooling_down={} banned={} | advertised={}",
            outbound,
            self.outbound_target,
            v11_out,
            v10_out,
            other_out,
            inbound,
            self.inbound_limit,
            known,
            cooling,
            banned,
            advertised.map(|addr| addr.to_string()).unwrap_or_else(|| "NONE (this node cannot receive inbound peers)".to_owned()),
        );
    }

    pub async fn add_connection_request(&self, address: SocketAddr, is_permanent: bool) {
        // If the request already exists, it resets the attempts count and overrides the `is_permanent` setting.
        self.connection_requests.lock().await.insert(address, ConnectionRequest::new(is_permanent));
        self.trigger_iteration(); // We force the next iteration of the connection loop.
    }

    pub async fn stop(&self) {
        self.shutdown_signal.trigger.trigger()
    }

    async fn handle_connection_requests(self: &Arc<Self>, peer_by_address: &HashMap<SocketAddr, Peer>) {
        let mut requests = self.connection_requests.lock().await;
        let mut new_requests = HashMap::with_capacity(requests.len());
        for (address, request) in requests.iter() {
            let address = *address;
            let request = request.clone();
            let is_connected = peer_by_address.contains_key(&address);
            if is_connected && !request.is_permanent {
                // The peer is connected and the request is not permanent - no need to keep the request
                continue;
            }

            if !is_connected && request.next_attempt <= SystemTime::now() {
                debug!("Connecting to peer request {}", address);
                match self.p2p_adaptor.connect_peer(address.to_string()).await {
                    Err(err) => {
                        debug!("Failed connecting to peer request: {}, {}", address, err);
                        if request.is_permanent {
                            const MAX_ACCOUNTABLE_ATTEMPTS: u32 = 4;
                            let retry_duration =
                                Duration::from_secs(30u64 * 2u64.pow(min(request.attempts, MAX_ACCOUNTABLE_ATTEMPTS)));
                            debug!("Will retry peer request {} in {}", address, DurationString::from(retry_duration));
                            new_requests.insert(
                                address,
                                ConnectionRequest {
                                    next_attempt: SystemTime::now() + retry_duration,
                                    attempts: request.attempts + 1,
                                    is_permanent: true,
                                },
                            );
                        }
                    }
                    Ok(_) if request.is_permanent => {
                        // Permanent requests are kept forever
                        new_requests.insert(address, ConnectionRequest::new(true));
                    }
                    Ok(_) => {}
                }
            } else {
                new_requests.insert(address, request);
            }
        }

        *requests = new_requests;
    }

    async fn handle_outbound_connections(self: &Arc<Self>, peer_by_address: &HashMap<SocketAddr, Peer>) {
        // Exclude every active peer, not only the outbound ones. For an inbound peer
        // `net_address()` carries the remote *ephemeral* port, so it never matched an address-store
        // entry and we kept re-dialing nodes we were already connected to. Those dials fail with
        // `PeerAlreadyExists` — correctly not counted as a failure, but they burned a dial slot on
        // every single round of a well-connected node.
        // Outbound peers by their dialed address; inbound peers by the address they advertised at
        // handshake (their `net_address()` carries the remote ephemeral port).
        let active_addresses: HashSet<(IpAddr, u16)> = peer_by_address
            .values()
            .filter_map(|peer| {
                if peer.is_outbound() {
                    Some(NetAddress::from(peer.net_address()))
                } else {
                    peer.properties().advertised_address
                }
            })
            .map(|addr| (canonical_ip(addr.ip.into()), addr.port))
            .collect();
        let active_outbound: HashSet<keryx_addressmanager::NetAddress> =
            peer_by_address.values().filter(|peer| peer.is_outbound()).map(|peer| peer.net_address().into()).collect();
        if active_outbound.len() >= self.outbound_target {
            return;
        }

        let mut missing_connections = self.outbound_target - active_outbound.len();
        let mut addr_iter = self.address_manager.lock().iterate_prioritized_random_addresses(active_outbound);
        let mut progressing = true;
        let mut exhausted = false;
        let mut cooling_down = 0usize;
        let mut rounds = 0usize;
        while !exhausted && missing_connections > 0 && rounds < MAX_DIAL_ROUNDS_PER_ITERATION {
            if self.shutdown_signal.trigger.is_triggered() {
                return;
            }
            rounds += 1;
            // Gather a pool larger than needed and rank it by proof encoding: a node whose outbound
            // slots all went to pre-v11 peers still syncs, but pays the legacy proof encoding on every
            // block — which is the bandwidth the v4 proof made expensive in the first place.
            let mut candidates = Vec::with_capacity(missing_connections * CANDIDATE_POOL_FACTOR);
            while candidates.len() < missing_connections * CANDIDATE_POOL_FACTOR {
                let Some(net_addr) = addr_iter.next() else {
                    exhausted = true;
                    break;
                };
                if active_addresses.contains(&(canonical_ip(net_addr.ip.into()), net_addr.port)) {
                    continue;
                }
                if self.is_cooling_down(net_addr) {
                    cooling_down += 1;
                    continue;
                }
                candidates.push(net_addr);
            }
            // `sort_by_key` is stable, so the failure-weighted random order the store handed us is
            // preserved inside each preference tier.
            let now = Instant::now();
            let known = self.known_protocols.lock().clone();
            candidates.sort_by_key(|addr| Self::peer_protocol_preference(addr.ip.into(), &known, now));
            let addrs_to_connect = candidates.into_iter().take(missing_connections).collect::<Vec<_>>();
            if addrs_to_connect.is_empty() {
                continue;
            }
            let jobs = addrs_to_connect
                .iter()
                .map(|net_addr| {
                    let socket_addr = SocketAddr::new(net_addr.ip.into(), net_addr.port).to_string();
                    debug!("Connecting to {}", &socket_addr);
                    self.p2p_adaptor.connect_peer(socket_addr)
                })
                .collect_vec();

            if progressing {
                // Log only if progress was made
                info!(
                    "Connection manager: has {}/{} outgoing P2P connections, trying to obtain {} additional connection(s)...",
                    self.outbound_target - missing_connections,
                    self.outbound_target,
                    jobs.len(),
                );
                progressing = false;
            } else {
                debug!(
                    "Connection manager: outgoing: {}/{} , connecting: {}, iterator: {}",
                    self.outbound_target - missing_connections,
                    self.outbound_target,
                    jobs.len(),
                    addr_iter.len(),
                );
            }
            for (res, net_addr) in (join_all(jobs).await).into_iter().zip(addrs_to_connect) {
                match res {
                    Ok(_) => {
                        self.address_manager.lock().mark_connection_success(net_addr);
                        // Deliberately a hold, not a clear. Clearing here meant that a peer which
                        // connected cleanly and then died on a flow error carried no backoff at
                        // all, so the loop re-dialed it on the very next iteration — every 2s under
                        // the starved cadence. `forgive_stable_peers` lifts the hold once the
                        // connection has actually lasted.
                        self.back_off_dial(net_addr);
                        missing_connections -= 1;
                        progressing = true;
                    }
                    Err(ConnectionError::ProtocolError(ProtocolError::PeerAlreadyExists(_))) => {
                        // We avoid marking the existing connection as connection failure
                        debug!("Failed connecting to {:?}, peer already exists", net_addr);
                    }
                    Err(err) => {
                        debug!("Failed connecting to {:?}, err: {}", net_addr, err);
                        self.address_manager.lock().mark_connection_failure(net_addr);
                        self.back_off_dial(net_addr);
                    }
                }
            }
        }

        if missing_connections == 0 {
            return;
        }

        // Every known address is connected, cooling down or written off. Decay the failure counts so
        // that a node which was offline long enough to saturate its whole store can recover instead
        // of treating every address as hopeless forever.
        if exhausted && cooling_down == 0 {
            self.address_manager.lock().decay_connection_failures();
        }

        if self.dns_seeders.is_empty() {
            return;
        }
        {
            let mut last = self.last_dns_seed.lock();
            if last.is_some_and(|at| at.elapsed() < DNS_SEED_MIN_INTERVAL) {
                return;
            }
            *last = Some(Instant::now());
        }
        let learned = if exhausted || missing_connections > self.outbound_target / 2 {
            // If the store is exhausted, or we are missing more than half of our target, query all
            // seeders in parallel. This is always the case on a new node start-up and is the most
            // resilient strategy in such a case.
            self.dns_seed_many(self.dns_seeders.len()).await
        } else {
            // Try to obtain at least twice the number of missing connections
            self.dns_seed_with_address_target(2 * missing_connections).await
        };
        // Seeding used to be the last thing an iteration did, so freshly learned addresses sat
        // unused until the next tick — a wasted round on every cold start, which is exactly when the
        // node can least afford one. Re-run the loop right away, but only when we actually learned
        // something new, so this can never turn into a spin.
        if learned > 0 {
            self.trigger_iteration();
        }
    }

    /// Clears the dial backoff of outbound peers that have stayed connected past
    /// [`STABLE_CONNECTION_GRACE`], so a peer that is simply long-lived never accumulates history,
    /// while one that keeps connecting and dying keeps escalating.
    fn forgive_stable_peers(&self, peer_by_address: &HashMap<SocketAddr, Peer>) {
        let stable = peer_by_address
            .values()
            .filter(|peer| peer.is_outbound() && peer.time_connected() > STABLE_CONNECTION_GRACE.as_millis() as u64)
            .map(|peer| NetAddress::from(peer.net_address()))
            .collect_vec();
        if stable.is_empty() {
            return;
        }
        let mut cooldown = self.dial_cooldown.lock();
        for address in stable {
            cooldown.forgive(address);
        }
    }

    /// Whether `address` is still inside its dial backoff window.
    fn is_cooling_down(&self, address: NetAddress) -> bool {
        self.dial_cooldown.lock().is_cooling_down(address, Instant::now())
    }

    /// Escalates the exponential dial backoff for `address`.
    ///
    /// Called both when a dial fails outright and when one succeeds — see the call site for why a
    /// success also backs off. This is what makes the fast loop cadence safe: without it, a starved
    /// node would re-dial the same unreachable (or flapping) addresses every 2s.
    fn back_off_dial(&self, address: NetAddress) {
        self.dial_cooldown.lock().back_off(address, Instant::now());
    }

    async fn handle_inbound_connections(self: &Arc<Self>, peer_by_address: &HashMap<SocketAddr, Peer>) {
        let active_inbound = peer_by_address.values().filter(|peer| !peer.is_outbound()).collect_vec();
        let active_inbound_len = active_inbound.len();
        if self.inbound_limit >= active_inbound_len {
            return;
        }

        // Eviction used to be uniformly random, which dropped long-lived, well-behaved peers as
        // readily as the freshly arrived ones that caused the overflow. Rank worst-first instead
        // (see `inbound_eviction_rank`) and never evict a peer we deliberately keep.
        let mut prefix_counter: HashMap<PrefixBucket, usize> = HashMap::new();
        for peer in active_inbound.iter() {
            *prefix_counter.entry(NetAddress::from(peer.net_address()).prefix_bucket()).or_insert(0) += 1;
        }
        let mut candidates = Vec::with_capacity(active_inbound_len);
        for peer in active_inbound {
            let ip = canonical_ip(peer.net_address().ip());
            if is_fixed_seed_ip(self.ban_exempt_seeders, ip) || self.ip_has_permanent_connection(ip).await {
                continue;
            }
            let redundancy = prefix_counter.get(&NetAddress::from(peer.net_address()).prefix_bucket()).copied().unwrap_or(1);
            candidates.push((inbound_eviction_rank(peer.time_connected(), peer.last_ping_duration(), redundancy), peer));
        }
        candidates.sort_by(|(a, _), (b, _)| b.cmp(a));

        let futures = candidates
            .into_iter()
            .take(active_inbound_len - self.inbound_limit)
            .map(|(_, peer)| {
                debug!("Disconnecting from {} because we're above the inbound limit", peer.net_address());
                self.p2p_adaptor.terminate(peer.key())
            })
            .collect_vec();
        join_all(futures).await;
    }

    /// Queries DNS seeders in random order, one after the other, until obtaining
    /// `min_addresses_to_fetch` addresses. Returns the number of *newly learned* addresses.
    async fn dns_seed_with_address_target(self: &Arc<Self>, min_addresses_to_fetch: usize) -> usize {
        let cmgr = self.clone();
        match tokio::task::spawn_blocking(move || cmgr.dns_seed_with_address_target_blocking(min_addresses_to_fetch)).await {
            Ok(learned) => learned,
            Err(err) => {
                warn!("DNS seeding task failed: {}", err);
                0
            }
        }
    }

    fn dns_seed_with_address_target_blocking(self: &Arc<Self>, mut min_addresses_to_fetch: usize) -> usize {
        let shuffled_dns_seeders = self.dns_seeders.choose_multiple(&mut thread_rng(), self.dns_seeders.len());
        let mut learned = 0;
        for &seeder in shuffled_dns_seeders {
            // Query seeders sequentially until reaching the desired number of addresses
            let addrs_len = self.dns_seed_single(seeder);
            learned += addrs_len;
            if addrs_len >= min_addresses_to_fetch {
                break;
            } else {
                min_addresses_to_fetch -= addrs_len;
            }
        }
        learned
    }

    /// Queries `num_seeders_to_query` random DNS seeders in parallel. Returns the number of newly
    /// learned addresses.
    async fn dns_seed_many(self: &Arc<Self>, num_seeders_to_query: usize) -> usize {
        info!("Querying {} DNS seeders", num_seeders_to_query);
        let shuffled_dns_seeders = self.dns_seeders.choose_multiple(&mut thread_rng(), num_seeders_to_query);
        let jobs = shuffled_dns_seeders.map(|seeder| {
            let cmgr = self.clone();
            tokio::task::spawn_blocking(move || cmgr.dns_seed_single(seeder))
        });
        match try_join_all(jobs).await {
            Ok(counts) => counts.into_iter().sum(),
            Err(err) => {
                warn!("DNS seeding task failed: {}", err);
                0
            }
        }
    }

    /// Query a single DNS seeder and add the obtained addresses to the address manager. Returns the
    /// number of addresses we did not already know — a seeder that only echoes back what we have is
    /// not progress, and the caller uses this to decide whether an immediate re-dial is warranted.
    ///
    /// DNS lookup is a blocking i/o operation so this function is assumed to be called
    /// from a blocking execution context.
    fn dns_seed_single(self: &Arc<Self>, seeder: &str) -> usize {
        info!("Querying DNS seeder {}", seeder);
        // Since the DNS lookup protocol doesn't come with a port, we must assume that the default port is used.
        let addrs = match (seeder, self.default_port).to_socket_addrs() {
            Ok(addrs) => addrs,
            Err(e) => {
                warn!("Error connecting to DNS seeder {}: {}", seeder, e);
                return 0;
            }
        };

        let addrs_len = addrs.len();
        let mut amgr_lock = self.address_manager.lock();
        let learned = addrs.filter(|addr| amgr_lock.add_address(NetAddress::new(addr.ip().into(), addr.port()))).count();
        drop(amgr_lock);
        info!("Retrieved {} addresses from DNS seeder {} ({} new)", addrs_len, seeder, learned);
        learned
    }

    /// Bans the given IP and disconnects from all the peers with that IP.
    ///
    /// _GO-KASPAD: BanByIP_
    pub async fn ban(&self, ip: IpAddr) -> bool {
        let ip = canonical_ip(ip);
        if self.ip_has_permanent_connection(ip).await {
            return false;
        }
        self.address_manager.lock().ban(ip.into());
        for peer in self.p2p_adaptor.active_peers() {
            if canonical_ip(peer.net_address().ip()) == ip {
                self.p2p_adaptor.terminate(peer.key()).await;
            }
        }
        true
    }

    pub async fn ban_automatically(&self, ip: IpAddr) -> bool {
        let ip = canonical_ip(ip);
        if is_fixed_seed_ip(self.ban_exempt_seeders, ip) || self.ip_has_permanent_connection(ip).await {
            return false;
        }
        self.ban(ip).await
    }

    /// Records a ping timeout for the given IP. Bans it after PING_TIMEOUT_BAN_THRESHOLD
    /// timeouts within PING_TIMEOUT_WINDOW — targets phantom nodes that flood inbound slots
    /// by connecting silently then immediately reconnecting after each timeout.
    pub async fn record_ping_timeout(&self, ip: IpAddr) {
        let ip = canonical_ip(ip);
        if is_fixed_seed_ip(self.ban_exempt_seeders, ip) || self.ip_has_permanent_connection(ip).await {
            return;
        }
        let should_ban = {
            let mut tracker = self.ping_timeout_tracker.lock();
            let now = Instant::now();
            let record = tracker.entry(ip).or_insert(PingTimeoutRecord { count: 0, window_start: now });
            if record.window_start.elapsed() > PING_TIMEOUT_WINDOW {
                record.count = 0;
                record.window_start = now;
            }
            record.count += 1;
            if record.count >= PING_TIMEOUT_BAN_THRESHOLD {
                tracker.remove(&ip);
                true
            } else {
                false
            }
        };
        if should_ban && self.ban_automatically(ip).await {
            warn!("Banning peer {} after {} ping timeouts within {:?}", ip, PING_TIMEOUT_BAN_THRESHOLD, PING_TIMEOUT_WINDOW);
        }
    }

    /// Returns whether the given address is banned.
    pub async fn is_banned(&self, address: &SocketAddr) -> bool {
        !self.is_permanent(address).await && self.address_manager.lock().is_banned(address.ip().into())
    }

    /// Returns whether the given address is a permanent request.
    pub async fn is_permanent(&self, address: &SocketAddr) -> bool {
        self.connection_requests.lock().await.contains_key(address)
    }

    /// Returns whether the given IP has some permanent request.
    pub async fn ip_has_permanent_connection(&self, ip: IpAddr) -> bool {
        let ip = canonical_ip(ip);
        self.connection_requests.lock().await.iter().any(|(address, request)| request.is_permanent && canonical_ip(address.ip()) == ip)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn fixed_seed_ip_is_ban_exempt() {
        let seeders = &["seed.example.net", "192.0.2.1"];

        assert!(is_fixed_seed_ip(seeders, "192.0.2.1".parse().unwrap()));
        assert!(is_fixed_seed_ip(seeders, "::ffff:192.0.2.1".parse().unwrap()));
        assert!(!is_fixed_seed_ip(seeders, "127.0.0.1".parse().unwrap()));
    }

    #[test]
    fn dial_backoff_grows_and_caps() {
        assert_eq!(dial_backoff(0), DIAL_COOLDOWN_BASE, "a first failure must not be treated as a zeroth one");
        assert_eq!(dial_backoff(1), DIAL_COOLDOWN_BASE);
        assert_eq!(dial_backoff(2), DIAL_COOLDOWN_BASE * 2);
        assert_eq!(dial_backoff(MAX_ACCOUNTABLE_DIAL_FAILURES), DIAL_COOLDOWN_BASE * 16);
        assert_eq!(dial_backoff(MAX_ACCOUNTABLE_DIAL_FAILURES + 100), DIAL_COOLDOWN_BASE * 16, "backoff must saturate");
    }

    fn test_addr() -> NetAddress {
        NetAddress::from_str("1.2.3.4:22111").unwrap()
    }

    /// The regression this guards: clearing the backoff on a successful dial meant a peer that
    /// connected cleanly and then died on a flow error — being asked for a block body we only have
    /// the header of, say — was re-dialed on the very next loop iteration, i.e. every 2s under the
    /// starved cadence.
    #[test]
    fn a_dial_that_connects_still_takes_a_hold() {
        let addr = test_addr();
        let t0 = Instant::now();
        let mut cooldown = DialCooldown::default();

        cooldown.back_off(addr, t0);
        assert!(cooldown.is_cooling_down(addr, t0), "a peer we just connected to must not be re-dialable at once");
        assert!(!cooldown.is_cooling_down(addr, t0 + DIAL_COOLDOWN_BASE + Duration::from_secs(1)));
    }

    #[test]
    fn flapping_escalates_instead_of_resetting() {
        let addr = test_addr();
        let t0 = Instant::now();
        let mut cooldown = DialCooldown::default();

        cooldown.back_off(addr, t0);
        cooldown.back_off(addr, t0);
        assert!(
            cooldown.is_cooling_down(addr, t0 + DIAL_COOLDOWN_BASE + Duration::from_secs(1)),
            "a second connect-then-die must double the hold, not restart it"
        );
        assert!(!cooldown.is_cooling_down(addr, t0 + DIAL_COOLDOWN_BASE * 2 + Duration::from_secs(1)));
    }

    #[test]
    fn a_connection_that_lasts_clears_its_history() {
        let addr = test_addr();
        let t0 = Instant::now();
        let mut cooldown = DialCooldown::default();

        cooldown.back_off(addr, t0);
        cooldown.back_off(addr, t0);
        cooldown.forgive(addr);
        assert!(!cooldown.is_cooling_down(addr, t0), "a stable peer must be immediately re-dialable if it drops");

        cooldown.back_off(addr, t0);
        assert!(
            !cooldown.is_cooling_down(addr, t0 + DIAL_COOLDOWN_BASE + Duration::from_secs(1)),
            "and its backoff must restart from the base rather than resume where it left off"
        );
    }

    #[test]
    fn elapsed_holds_are_collected() {
        let addr = test_addr();
        let t0 = Instant::now();
        let mut cooldown = DialCooldown::default();

        cooldown.back_off(addr, t0);
        cooldown.gc(t0);
        assert_eq!(cooldown.len(), 1, "a live hold survives collection");
        cooldown.gc(t0 + DIAL_COOLDOWN_BASE + Duration::from_secs(1));
        assert_eq!(cooldown.len(), 0, "an elapsed one does not");
    }

    #[test]
    fn inbound_eviction_prefers_squatters_then_newcomers() {
        let hour = Duration::from_secs(3600).as_millis() as u64;
        let grace = INBOUND_SILENCE_GRACE.as_millis() as u64;

        let established = inbound_eviction_rank(hour, 42, 1);
        let newcomer = inbound_eviction_rank(1_000, 7, 1);
        let squatter = inbound_eviction_rank(grace + 1, 0, 1);
        let young_and_quiet = inbound_eviction_rank(1_000, 0, 1);

        // Higher rank is evicted first
        assert!(squatter > newcomer, "a peer that never ponged goes before a merely new one");
        assert!(newcomer > established, "a burst of arrivals must not displace established peers");
        assert!(young_and_quiet < squatter, "silence only counts after the grace period");

        // Prefix redundancy breaks ties between otherwise identical peers
        assert!(inbound_eviction_rank(hour, 42, 5) > inbound_eviction_rank(hour, 42, 1));
    }

    /// v10 serves a perfectly valid v4 PoM proof, just in the legacy encoding rather than the compact
    /// multiproof v11 added — so this ranks by wire cost, and v10 must stay dialable. A peer we have
    /// never met also ranks above one known to be old: it may well be v11.
    #[test]
    fn outbound_dialing_prefers_v11_but_keeps_v10() {
        let now = Instant::now();
        let ip = |s: &str| IpAddr::from_str(s).unwrap();
        let known: HashMap<IpAddr, (u32, Instant)> = [
            (ip("1.1.1.1"), (11u32, now)),
            (ip("2.2.2.2"), (10u32, now)),
            (ip("3.3.3.3"), (9u32, now)),
            (ip("4.4.4.4"), (12u32, now)),
        ]
        .into_iter()
        .collect();

        let pref = |s: &str| ConnectionManager::peer_protocol_preference(ip(s), &known, now);
        assert!(pref("1.1.1.1") < pref("5.5.5.5"), "a known v11 peer is dialed before an unknown one");
        assert!(pref("5.5.5.5") < pref("2.2.2.2"), "an unknown peer is dialed before a known v10 one");
        assert!(pref("2.2.2.2") < pref("3.3.3.3"), "v10 is a fallback, still ahead of anything older");
        assert_eq!(pref("4.4.4.4"), pref("1.1.1.1"), "anything at or above v11 ranks the same");
    }

    /// v4-mapped and plain forms must land on the same entry, or the preference silently misses.
    #[test]
    fn protocol_observations_match_v4_mapped_addresses() {
        let now = Instant::now();
        let known: HashMap<IpAddr, (u32, Instant)> = [(IpAddr::from_str("1.1.1.1").unwrap(), (11u32, now))].into_iter().collect();

        assert_eq!(
            ConnectionManager::peer_protocol_preference(IpAddr::from_str("::ffff:1.1.1.1").unwrap(), &known, now),
            0,
            "a v4-mapped address must resolve to the same observation"
        );
    }

    /// The map is keyed by remote-supplied IPs, so a stale entry must not pin a peer to an old
    /// version forever — a peer that upgrades has to become indistinguishable from an unknown one.
    #[test]
    fn a_stale_protocol_observation_is_ignored() {
        let now = Instant::now();
        let learned_at = now - KNOWN_PROTOCOL_TTL - Duration::from_secs(1);
        let ip = IpAddr::from_str("2.2.2.2").unwrap();
        let known: HashMap<IpAddr, (u32, Instant)> = [(ip, (10u32, learned_at))].into_iter().collect();
        let unknown = HashMap::new();

        assert_eq!(
            ConnectionManager::peer_protocol_preference(ip, &known, now),
            ConnectionManager::peer_protocol_preference(ip, &unknown, now),
            "an expired observation must rank as unknown, not as known-old"
        );
    }
}
