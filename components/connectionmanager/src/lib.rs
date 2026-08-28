use std::{
    cmp::min,
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
use keryx_utils::triggers::SingleTrigger;
use parking_lot::Mutex as ParkingLotMutex;
use rand::{seq::SliceRandom, thread_rng};
use tokio::{
    select,
    sync::{
        Mutex as TokioMutex,
        mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel},
    },
    time::{MissedTickBehavior, interval},
};

const PING_TIMEOUT_BAN_THRESHOLD: u32 = 3;
const PING_TIMEOUT_WINDOW: Duration = Duration::from_secs(600); // 10 minutes

/// How long a protocol version learned from a handshake is trusted for outbound preference.
/// A peer that upgrades must be able to stop looking old, so the observation expires.
const KNOWN_PROTOCOL_TTL: Duration = Duration::from_secs(86400); // 24 hours
/// Hard ceiling on the version map, mirroring `MAX_ADDRESSES` in the address store. The map is keyed
/// by remote-supplied IPs, so it needs a bound.
const MAX_KNOWN_PROTOCOL_ENTRIES: usize = 4096;
/// The first protocol version that encodes a v4 PoM proof as a compact multiproof. Older peers
/// serve the same proof in the legacy encoding — correct, just several times heavier on the wire.
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
    /// Last protocol version negotiated with each IP, plus the instant it was learned. Used to
    /// prefer peers that serve the compact proof encoding while keeping older ones dialable.
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
            known_protocols: Default::default(),
        });
        manager.clone().start_event_loop(rx);
        manager.force_next_iteration.send(()).unwrap();
        manager
    }

    /// Records the protocol version actually negotiated with `ip`. Called from the handshake path;
    /// best-effort and never blocks the flow.
    pub fn record_peer_protocol_version(&self, ip: IpAddr, version: u32) {
        let ip = canonical_ip(ip);
        let now = Instant::now();
        let mut known = self.known_protocols.lock();
        if known.len() >= MAX_KNOWN_PROTOCOL_ENTRIES && !known.contains_key(&ip) {
            // Full, and this is a new IP: drop what already expired, and if that frees nothing, drop
            // the oldest observation. Either way a peer cycling through addresses cannot grow the map.
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

    /// Drops expired protocol observations. Without this the map grows over the process lifetime.
    fn gc_known_protocols(&self) {
        let now = Instant::now();
        self.known_protocols.lock().retain(|_, (_, learned_at)| now.duration_since(*learned_at) <= KNOWN_PROTOCOL_TTL);
    }

    fn start_event_loop(self: Arc<Self>, mut rx: UnboundedReceiver<()>) {
        let mut ticker = interval(Duration::from_secs(30));
        ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);
        tokio::spawn(async move {
            loop {
                if self.shutdown_signal.trigger.is_triggered() {
                    break;
                }
                select! {
                    _ = rx.recv() => self.clone().handle_event().await,
                    _ = ticker.tick() => self.clone().handle_event().await,
                    _ = self.shutdown_signal.listener.clone() => break,
                }
            }
            debug!("Connection manager event loop exiting");
        });
    }

    async fn handle_event(self: Arc<Self>) {
        debug!("Starting connection loop iteration");
        let peers = self.p2p_adaptor.active_peers();
        let peer_by_address: HashMap<SocketAddr, Peer> = peers.into_iter().map(|peer| (peer.net_address(), peer)).collect();

        self.handle_connection_requests(&peer_by_address).await;
        self.handle_outbound_connections(&peer_by_address).await;
        self.handle_inbound_connections(&peer_by_address).await;
        self.gc_known_protocols();
    }

    pub async fn add_connection_request(&self, address: SocketAddr, is_permanent: bool) {
        // If the request already exists, it resets the attempts count and overrides the `is_permanent` setting.
        self.connection_requests.lock().await.insert(address, ConnectionRequest::new(is_permanent));
        self.force_next_iteration.send(()).unwrap(); // We force the next iteration of the connection loop.
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
        let active_outbound: HashSet<keryx_addressmanager::NetAddress> =
            peer_by_address.values().filter(|peer| peer.is_outbound()).map(|peer| peer.net_address().into()).collect();
        if active_outbound.len() >= self.outbound_target {
            return;
        }

        let mut missing_connections = self.outbound_target - active_outbound.len();
        let mut addr_iter = self.address_manager.lock().iterate_prioritized_random_addresses(active_outbound);
        let mut progressing = true;
        let mut connecting = true;
        while connecting && missing_connections > 0 {
            if self.shutdown_signal.trigger.is_triggered() {
                return;
            }
            // Gather a pool larger than needed and rank it by proof encoding: a node whose outbound
            // slots all went to pre-v11 peers still syncs, but pays the legacy proof encoding on every
            // block — which is the bandwidth the v4 proof made expensive in the first place.
            let mut candidates = Vec::with_capacity(missing_connections * CANDIDATE_POOL_FACTOR);
            while candidates.len() < missing_connections * CANDIDATE_POOL_FACTOR {
                let Some(net_addr) = addr_iter.next() else {
                    connecting = false;
                    break;
                };
                candidates.push(net_addr);
            }
            // `sort_by_key` is stable, so the failure-weighted random order the address store handed
            // us survives inside each preference tier — the ranking re-orders tiers, it does not
            // replace the selection policy.
            let now = Instant::now();
            let known = self.known_protocols.lock().clone();
            candidates.sort_by_key(|net_addr| Self::peer_protocol_preference(net_addr.ip.into(), &known, now));

            let mut addrs_to_connect = Vec::with_capacity(missing_connections);
            let mut jobs = Vec::with_capacity(missing_connections);
            for net_addr in candidates.into_iter().take(missing_connections) {
                let socket_addr = SocketAddr::new(net_addr.ip.into(), net_addr.port).to_string();
                debug!("Connecting to {}", &socket_addr);
                addrs_to_connect.push(net_addr);
                jobs.push(self.p2p_adaptor.connect_peer(socket_addr.clone()));
            }

            if progressing && !jobs.is_empty() {
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
                    }
                }
            }
        }

        if missing_connections > 0 && !self.dns_seeders.is_empty() {
            if missing_connections > self.outbound_target / 2 {
                // If we are missing more than half of our target, query all in parallel.
                // This will always be the case on new node start-up and is the most resilient strategy in such a case.
                self.dns_seed_many(self.dns_seeders.len()).await;
            } else {
                // Try to obtain at least twice the number of missing connections
                self.dns_seed_with_address_target(2 * missing_connections).await;
            }
        }
    }

    async fn handle_inbound_connections(self: &Arc<Self>, peer_by_address: &HashMap<SocketAddr, Peer>) {
        let active_inbound = peer_by_address.values().filter(|peer| !peer.is_outbound()).collect_vec();
        let active_inbound_len = active_inbound.len();
        if self.inbound_limit >= active_inbound_len {
            return;
        }

        let mut futures = Vec::with_capacity(active_inbound_len - self.inbound_limit);
        for peer in active_inbound.choose_multiple(&mut thread_rng(), active_inbound_len - self.inbound_limit) {
            debug!("Disconnecting from {} because we're above the inbound limit", peer.net_address());
            futures.push(self.p2p_adaptor.terminate(peer.key()));
        }
        join_all(futures).await;
    }

    /// Queries DNS seeders in random order, one after the other, until obtaining `min_addresses_to_fetch` addresses
    async fn dns_seed_with_address_target(self: &Arc<Self>, min_addresses_to_fetch: usize) {
        let cmgr = self.clone();
        tokio::task::spawn_blocking(move || cmgr.dns_seed_with_address_target_blocking(min_addresses_to_fetch)).await.unwrap();
    }

    fn dns_seed_with_address_target_blocking(self: &Arc<Self>, mut min_addresses_to_fetch: usize) {
        let shuffled_dns_seeders = self.dns_seeders.choose_multiple(&mut thread_rng(), self.dns_seeders.len());
        for &seeder in shuffled_dns_seeders {
            // Query seeders sequentially until reaching the desired number of addresses
            let addrs_len = self.dns_seed_single(seeder);
            if addrs_len >= min_addresses_to_fetch {
                break;
            } else {
                min_addresses_to_fetch -= addrs_len;
            }
        }
    }

    /// Queries `num_seeders_to_query` random DNS seeders in parallel
    async fn dns_seed_many(self: &Arc<Self>, num_seeders_to_query: usize) -> usize {
        info!("Querying {} DNS seeders", num_seeders_to_query);
        let shuffled_dns_seeders = self.dns_seeders.choose_multiple(&mut thread_rng(), num_seeders_to_query);
        let jobs = shuffled_dns_seeders.map(|seeder| {
            let cmgr = self.clone();
            tokio::task::spawn_blocking(move || cmgr.dns_seed_single(seeder))
        });
        try_join_all(jobs).await.unwrap().into_iter().sum()
    }

    /// Query a single DNS seeder and add the obtained addresses to the address manager.
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
        info!("Retrieved {} addresses from DNS seeder {}", addrs_len, seeder);
        let mut amgr_lock = self.address_manager.lock();
        for addr in addrs {
            amgr_lock.add_address(NetAddress::new(addr.ip().into(), addr.port()));
        }

        addrs_len
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

    #[test]
    fn fixed_seed_ip_is_ban_exempt() {
        let seeders = &["seed.example.net", "192.0.2.1"];

        assert!(is_fixed_seed_ip(seeders, "192.0.2.1".parse().unwrap()));
        assert!(is_fixed_seed_ip(seeders, "::ffff:192.0.2.1".parse().unwrap()));
        assert!(!is_fixed_seed_ip(seeders, "127.0.0.1".parse().unwrap()));
    }

    /// v10 serves a perfectly valid v4 PoM proof, just in the legacy encoding rather than the compact
    /// multiproof v11 added — so this ranks by wire cost, and v10 must stay dialable. A peer we have
    /// never met also ranks above one known to be old: it may well be v11.
    #[test]
    fn outbound_dialing_prefers_v11_but_keeps_v10() {
        let now = Instant::now();
        let ip = |s: &str| s.parse::<IpAddr>().unwrap();
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
        let known: HashMap<IpAddr, (u32, Instant)> = [("1.1.1.1".parse::<IpAddr>().unwrap(), (11u32, now))].into_iter().collect();

        assert_eq!(
            ConnectionManager::peer_protocol_preference("::ffff:1.1.1.1".parse().unwrap(), &known, now),
            0,
            "a v4-mapped address must resolve to the same observation"
        );
    }

    /// A stale entry must not pin a peer to an old version forever: a peer that upgrades has to
    /// become indistinguishable from one we have never met.
    #[test]
    fn a_stale_protocol_observation_is_ignored() {
        let now = Instant::now();
        let learned_at = now - KNOWN_PROTOCOL_TTL - Duration::from_secs(1);
        let ip: IpAddr = "2.2.2.2".parse().unwrap();
        let known: HashMap<IpAddr, (u32, Instant)> = [(ip, (10u32, learned_at))].into_iter().collect();
        let unknown = HashMap::new();

        assert_eq!(
            ConnectionManager::peer_protocol_preference(ip, &known, now),
            ConnectionManager::peer_protocol_preference(ip, &unknown, now),
            "an expired observation must rank as unknown, not as known-old"
        );
    }
}
