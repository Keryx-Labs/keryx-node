//! Fetch preferred peers from the public explorer (same source as krx-node-finder).
//! No TCP port probing — the node handles reachability at runtime.

use serde::{Deserialize, Serialize};
use std::net::{SocketAddr, ToSocketAddrs};
use std::time::Duration;

/// Mainnet DNS seeders (mirrors consensus params).
pub const DNS_SEEDERS: &[&str] = &["seed.keryx-labs.com", "141.95.35.181"];
const PEERS_API_URL: &str = "https://keryx-labs.com/api/v1/peers";
pub const DEFAULT_P2P_PORT: u16 = 22111;
const MAX_SELECTED: usize = 8;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PeerProbeResult {
    pub address: String,
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rtt_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PeerProbeReport {
    pub results: Vec<PeerProbeResult>,
    pub selected: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warning: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProbePeersArgs {
    pub addresses: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct ApiPeer {
    ip: String,
    #[serde(default)]
    connected: bool,
}

pub async fn probe_peers(args: ProbePeersArgs) -> PeerProbeReport {
    let mut candidates = args.addresses.unwrap_or_default();
    candidates.retain(|a| !a.trim().is_empty());

    if candidates.is_empty() {
        candidates = collect_candidates().await;
    }

    // Dedupe while preserving order
    let mut seen = std::collections::HashSet::new();
    candidates.retain(|a| seen.insert(normalize_addr(a)));

    if candidates.is_empty() {
        return PeerProbeReport {
            results: vec![],
            selected: vec![],
            warning: Some(
                "No peers from the network API or DNS seed. The node will rely on DNS seeding at runtime."
                    .into(),
            ),
        };
    }

    let selected: Vec<String> = candidates.into_iter().take(MAX_SELECTED).collect();
    let results: Vec<PeerProbeResult> = selected
        .iter()
        .map(|address| PeerProbeResult {
            address: address.clone(),
            ok: true,
            rtt_ms: None,
            error: None,
        })
        .collect();

    PeerProbeReport {
        results,
        selected,
        warning: None,
    }
}

async fn collect_candidates() -> Vec<String> {
    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();

    // 1) Live peer directory from keryx-labs.com (same as krx-node-finder)
    if let Ok(api_peers) = fetch_network_peers().await {
        for addr in api_peers {
            let key = normalize_addr(&addr);
            if seen.insert(key) {
                out.push(addr);
            }
        }
    }

    // 2) DNS seed A-records as fallback/extra
    if let Ok(dns) = tokio::task::spawn_blocking(resolve_dns_seed_peers).await {
        for addr in dns {
            let key = normalize_addr(&addr);
            if seen.insert(key) {
                out.push(addr);
            }
        }
    }

    out
}

async fn fetch_network_peers() -> Result<Vec<String>, String> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(12))
        .user_agent("keryx-node-launcher/1.0")
        .build()
        .map_err(|e| e.to_string())?;
    let resp = client
        .get(PEERS_API_URL)
        .header("accept", "application/json")
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("peers API HTTP {}", resp.status()));
    }
    let peers: Vec<ApiPeer> = resp.json().await.map_err(|e| e.to_string())?;

    // All unique IPv4; explorer-connected first (same ordering idea as krx-node-finder).
    let mut unique: std::collections::HashMap<String, bool> = std::collections::HashMap::new();
    for p in &peers {
        let ip = p.ip.trim();
        if ip.parse::<std::net::Ipv4Addr>().is_ok() {
            let entry = unique.entry(ip.to_string()).or_insert(false);
            *entry |= p.connected;
        }
    }

    let mut ordered: Vec<(String, bool)> = unique.into_iter().collect();
    ordered.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    Ok(ordered
        .into_iter()
        .map(|(ip, _)| format!("{ip}:{DEFAULT_P2P_PORT}"))
        .collect())
}

fn resolve_dns_seed_peers() -> Vec<String> {
    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for seeder in DNS_SEEDERS {
        let host_port = if seeder.contains(':') {
            seeder.to_string()
        } else {
            format!("{seeder}:{DEFAULT_P2P_PORT}")
        };
        if seeder.parse::<std::net::IpAddr>().is_ok() {
            let key = normalize_addr(&host_port);
            if seen.insert(key) {
                out.push(host_port.clone());
            }
        }
        if let Ok(iter) = host_port.to_socket_addrs() {
            for sa in iter {
                let formatted = format_socket_addr(sa);
                let key = normalize_addr(&formatted);
                if seen.insert(key) {
                    out.push(formatted);
                }
            }
        }
    }
    out
}

fn format_socket_addr(sa: SocketAddr) -> String {
    match sa {
        SocketAddr::V4(v4) => format!("{}:{}", v4.ip(), v4.port()),
        SocketAddr::V6(v6) => format!("[{}]:{}", v6.ip(), v6.port()),
    }
}

fn normalize_addr(addr: &str) -> String {
    addr.trim().to_ascii_lowercase()
}
