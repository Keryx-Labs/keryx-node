use crate::config::{now_ms, SyncPhase, SyncStatus};
use futures::StreamExt;
use parking_lot::Mutex as ParkingMutex;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tauri::{AppHandle, Emitter};
use tokio::sync::{mpsc, oneshot, Mutex};
use tonic::transport::Channel;
use tonic::Request;

pub mod protowire {
    tonic::include_proto!("protowire");
}

use protowire::{
    kaspad_request, kaspad_response, rpc_client::RpcClient, GetBlockDagInfoRequestMessage,
    GetInfoRequestMessage, GetServerInfoRequestMessage, KaspadRequest, KaspadResponse,
};

const NETWORK_URL: &str = "https://keryx-labs.com/api/v1/network";
const POLL_MS: u64 = 3000;
const NETWORK_REFRESH_MS: u64 = 30_000;
const REQUEST_TIMEOUT_MS: u64 = 5000;
const SAMPLE_WINDOW: usize = 12;

#[derive(Default, Clone)]
struct NodeProbe {
    server_version: Option<String>,
    is_synced: Option<bool>,
    mempool_size: Option<u64>,
    virtual_daa_score: Option<u64>,
    block_count: Option<u64>,
    header_count: Option<u64>,
}

#[derive(Clone, Copy)]
struct DaaSample {
    t: u64,
    daa: u64,
}

#[derive(Clone, Copy)]
struct PercentSample {
    t: u64,
    percent: f64,
}

#[derive(Clone)]
struct IbdProgress {
    phase: SyncPhase,
    percent: f64,
    processed: u64,
    tip_timestamp: Option<String>,
    updated_at: u64,
}

pub fn resolve_local_grpc_port(rpclisten: &str) -> u16 {
    let raw = rpclisten.trim();
    if raw.is_empty() {
        return 22110;
    }
    if let Some(idx) = raw.rfind(':') {
        if let Ok(port) = raw[idx + 1..].trim().parse::<u16>() {
            return port;
        }
    }
    raw.parse::<u16>().unwrap_or(22110)
}

async fn fetch_network_height() -> Option<u64> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(8))
        .user_agent("Keryx-Node-Launcher/1.0")
        .build()
        .ok()?;
    let resp = client
        .get(NETWORK_URL)
        .header("accept", "application/json")
        .send()
        .await
        .ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let data: serde_json::Value = resp.json().await.ok()?;
    let height = data.get("height")?.as_u64()?;
    (height > 0).then_some(height)
}

fn estimate_sync_percent(
    is_synced: Option<bool>,
    daa: Option<u64>,
    network_height: Option<u64>,
) -> Option<f64> {
    if is_synced == Some(true) {
        return Some(100.0);
    }
    let network_height = network_height.filter(|&h| h > 0)?;
    let daa = daa.filter(|&d| d > 0)?;
    let raw = (daa as f64 / network_height as f64) * 100.0;
    Some(((raw * 10.0).floor() / 10.0).clamp(0.0, 99.9))
}

fn format_eta(seconds: Option<f64>) -> Option<String> {
    let seconds = seconds.filter(|s| s.is_finite() && *s >= 0.0)?;
    if seconds < 60.0 {
        return Some("~<1m".into());
    }
    let s = seconds.round() as u64;
    let days = s / 86400;
    let hours = (s % 86400) / 3600;
    let mins = (s % 3600) / 60;
    if days > 0 {
        Some(format!("~{days}d {hours}h"))
    } else if hours > 0 {
        Some(format!("~{hours}h {mins}m"))
    } else {
        Some(format!("~{mins}m"))
    }
}

fn format_duration(seconds: f64) -> String {
    let s = seconds.max(0.0).floor() as u64;
    let days = s / 86400;
    let hours = (s % 86400) / 3600;
    let mins = (s % 3600) / 60;
    let secs = s % 60;
    if days > 0 {
        format!("{days}d {hours}h {mins}m")
    } else if hours > 0 {
        format!("{hours}h {mins}m {secs}s")
    } else if mins > 0 {
        format!("{mins}m {secs}s")
    } else {
        format!("{secs}s")
    }
}

fn estimate_eta_seconds(samples: &[DaaSample], daa: u64, network_height: u64) -> Option<f64> {
    if samples.len() < 2 {
        return None;
    }
    let first = samples[0];
    let last = *samples.last().unwrap();
    let dt = (last.t.saturating_sub(first.t)) as f64 / 1000.0;
    let d_daa = last.daa.saturating_sub(first.daa) as f64;
    if dt < 3.0 || d_daa <= 0.0 {
        return None;
    }
    let rate = d_daa / dt;
    let remaining = network_height.saturating_sub(daa) as f64;
    if remaining <= 0.0 {
        return Some(0.0);
    }
    let eta = remaining / rate;
    if !eta.is_finite() || eta > 86400.0 * 30.0 {
        None
    } else {
        Some(eta)
    }
}

fn rpc_ok(err: &Option<protowire::RpcError>) -> bool {
    err.as_ref().map(|e| e.message.is_empty()).unwrap_or(true)
}

struct PendingMap {
    map: ParkingMutex<HashMap<u64, oneshot::Sender<Option<KaspadResponse>>>>,
}

impl PendingMap {
    fn new() -> Self {
        Self {
            map: ParkingMutex::new(HashMap::new()),
        }
    }

    fn insert(&self, id: u64, tx: oneshot::Sender<Option<KaspadResponse>>) {
        self.map.lock().insert(id, tx);
    }

    fn take(&self, id: u64) -> Option<oneshot::Sender<Option<KaspadResponse>>> {
        self.map.lock().remove(&id)
    }

    fn clear_all(&self) {
        let mut map = self.map.lock();
        for (_, tx) in map.drain() {
            let _ = tx.send(None);
        }
    }
}

struct PersistentGrpc {
    port: AtomicU64,
    dead: AtomicBool,
    tx: Mutex<Option<mpsc::Sender<KaspadRequest>>>,
    pending: Arc<PendingMap>,
    seq: AtomicU64,
}

impl PersistentGrpc {
    fn new() -> Self {
        Self {
            port: AtomicU64::new(22110),
            dead: AtomicBool::new(true),
            tx: Mutex::new(None),
            pending: Arc::new(PendingMap::new()),
            seq: AtomicU64::new(0),
        }
    }

    fn is_alive(&self) -> bool {
        !self.dead.load(Ordering::SeqCst)
    }

    async fn connect(&self, port: u16) {
        let current = self.port.load(Ordering::SeqCst) as u16;
        if self.is_alive() && current == port {
            return;
        }
        self.close().await;
        self.port.store(port as u64, Ordering::SeqCst);

        let Ok(endpoint) = Channel::from_shared(format!("http://127.0.0.1:{port}")) else {
            self.dead.store(true, Ordering::SeqCst);
            return;
        };
        let endpoint = endpoint.connect_timeout(Duration::from_secs(2));
        let Ok(channel) = endpoint.connect().await else {
            self.dead.store(true, Ordering::SeqCst);
            return;
        };

        let mut client = RpcClient::new(channel)
            .max_decoding_message_size(64 * 1024 * 1024)
            .max_encoding_message_size(64 * 1024 * 1024);

        let (out_tx, out_rx) = mpsc::channel::<KaspadRequest>(32);
        let outbound = tokio_stream::wrappers::ReceiverStream::new(out_rx);

        let Ok(response) = client.message_stream(Request::new(outbound)).await else {
            self.dead.store(true, Ordering::SeqCst);
            return;
        };

        let mut inbound = response.into_inner();
        *self.tx.lock().await = Some(out_tx);
        self.dead.store(false, Ordering::SeqCst);

        let pending = self.pending.clone();
        let dead = Arc::new(AtomicBool::new(false));
        // Keep a handle so the task can signal death; mirrored via request failures too.
        let dead_flag = dead.clone();
        tokio::spawn(async move {
            while let Some(item) = inbound.next().await {
                match item {
                    Ok(resp) => {
                        let id = resp.id;
                        if let Some(tx) = pending.take(id) {
                            let _ = tx.send(Some(resp));
                        }
                    }
                    Err(_) => break,
                }
            }
            pending.clear_all();
            dead_flag.store(true, Ordering::SeqCst);
        });
        // If the stream dies, subsequent requests fail and mark self.dead.
        let _ = dead;
    }

    async fn close(&self) {
        *self.tx.lock().await = None;
        self.dead.store(true, Ordering::SeqCst);
        self.pending.clear_all();
    }

    async fn request(&self, payload: kaspad_request::Payload) -> Option<KaspadResponse> {
        if !self.is_alive() {
            return None;
        }
        let id = self.seq.fetch_add(1, Ordering::SeqCst) + 1;
        let (tx, rx) = oneshot::channel();
        self.pending.insert(id, tx);

        let msg = KaspadRequest {
            id,
            payload: Some(payload),
        };

        let send_ok = {
            let guard = self.tx.lock().await;
            if let Some(sender) = guard.as_ref() {
                sender.send(msg).await.is_ok()
            } else {
                false
            }
        };

        if !send_ok {
            self.pending.take(id);
            self.dead.store(true, Ordering::SeqCst);
            return None;
        }

        match tokio::time::timeout(Duration::from_millis(REQUEST_TIMEOUT_MS), rx).await {
            Ok(Ok(resp)) => resp,
            _ => {
                self.pending.take(id);
                None
            }
        }
    }

    async fn query_status(&self) -> Option<NodeProbe> {
        let port = self.port.load(Ordering::SeqCst) as u16;
        if !self.is_alive() {
            self.connect(port).await;
        }
        if !self.is_alive() {
            return None;
        }

        let (info_res, dag_res, server_res) = tokio::join!(
            self.request(kaspad_request::Payload::GetInfoRequest(GetInfoRequestMessage {})),
            self.request(kaspad_request::Payload::GetBlockDagInfoRequest(
                GetBlockDagInfoRequestMessage {}
            )),
            self.request(kaspad_request::Payload::GetServerInfoRequest(
                GetServerInfoRequestMessage {}
            )),
        );

        if info_res.is_none() && dag_res.is_none() && server_res.is_none() {
            self.dead.store(true, Ordering::SeqCst);
            return None;
        }

        let mut result = NodeProbe::default();

        if let Some(resp) = info_res {
            if let Some(kaspad_response::Payload::GetInfoResponse(info)) = resp.payload {
                if rpc_ok(&info.error) {
                    if !info.server_version.is_empty() {
                        result.server_version = Some(info.server_version);
                    }
                    result.is_synced = Some(info.is_synced);
                    result.mempool_size = Some(info.mempool_size);
                }
            }
        }

        if let Some(resp) = server_res {
            if let Some(kaspad_response::Payload::GetServerInfoResponse(server)) = resp.payload {
                if rpc_ok(&server.error) {
                    if !server.server_version.is_empty() {
                        result.server_version = Some(server.server_version);
                    }
                    if server.is_synced {
                        result.is_synced = Some(true);
                    } else if result.is_synced.is_none() {
                        result.is_synced = Some(server.is_synced);
                    }
                    result.virtual_daa_score = Some(server.virtual_daa_score);
                }
            }
        }

        if let Some(resp) = dag_res {
            if let Some(kaspad_response::Payload::GetBlockDagInfoResponse(dag)) = resp.payload {
                if rpc_ok(&dag.error) {
                    result.virtual_daa_score = Some(dag.virtual_daa_score);
                    result.block_count = Some(dag.block_count);
                    result.header_count = Some(dag.header_count);
                }
            }
        }

        Some(result)
    }
}

struct SyncInner {
    enabled: bool,
    network_height: Option<u64>,
    last_network_fetch: u64,
    samples: Vec<DaaSample>,
    percent_samples: Vec<PercentSample>,
    ibd: Option<IbdProgress>,
    last_status: SyncStatus,
    grpc: PersistentGrpc,
    ticking: bool,
    sync_started_at: Option<u64>,
    sync_finished_at: Option<u64>,
    saw_unsynced: bool,
    on_sync_complete: Option<Box<dyn Fn(f64) + Send + Sync>>,
    cancel: Option<tokio::sync::watch::Sender<bool>>,
}

pub struct SyncMonitor {
    inner: Arc<Mutex<SyncInner>>,
}

impl SyncMonitor {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(SyncInner {
                enabled: false,
                network_height: None,
                last_network_fetch: 0,
                samples: Vec::new(),
                percent_samples: Vec::new(),
                ibd: None,
                last_status: SyncStatus::empty(),
                grpc: PersistentGrpc::new(),
                ticking: false,
                sync_started_at: None,
                sync_finished_at: None,
                saw_unsynced: false,
                on_sync_complete: None,
                cancel: None,
            })),
        }
    }

    pub async fn set_on_sync_complete<F>(&self, f: Option<F>)
    where
        F: Fn(f64) + Send + Sync + 'static,
    {
        self.inner.lock().await.on_sync_complete =
            f.map(|cb| Box::new(cb) as Box<dyn Fn(f64) + Send + Sync>);
    }

    pub async fn get_last(&self) -> SyncStatus {
        self.inner.lock().await.last_status.clone()
    }

    pub async fn ingest_log(&self, line: &str) {
        let mut inner = self.inner.lock().await;
        if let Some(caps) = parse_ibd(line) {
            inner.saw_unsynced = true;
            inner.ibd = Some(IbdProgress {
                phase: caps.phase,
                percent: caps.percent,
                processed: caps.processed,
                tip_timestamp: caps.tip,
                updated_at: now_ms(),
            });
            inner.percent_samples.push(PercentSample {
                t: now_ms(),
                percent: caps.percent,
            });
            if inner.percent_samples.len() > SAMPLE_WINDOW {
                inner.percent_samples.remove(0);
            }
            return;
        }

        if let Some(headers_batch) = parse_headers_throughput(line) {
            if headers_batch > 0 {
                if let Some(ibd) = inner.ibd.as_mut() {
                    if ibd.phase == SyncPhase::Headers {
                        ibd.updated_at = now_ms();
                    }
                }
            }
        }
    }

    pub async fn start(&self, app: AppHandle, rpclisten: Option<&str>) {
        let port = resolve_local_grpc_port(rpclisten.unwrap_or(""));
        {
            let mut inner = self.inner.lock().await;
            if let Some(tx) = inner.cancel.take() {
                let _ = tx.send(true);
            }
            let (tx, _rx) = tokio::sync::watch::channel(false);
            inner.cancel = Some(tx);
            inner.enabled = true;
            inner.samples.clear();
            inner.percent_samples.clear();
            inner.ibd = None;
            inner.sync_started_at = Some(now_ms());
            inner.sync_finished_at = None;
            inner.saw_unsynced = false;
        }

        {
            let inner = self.inner.lock().await;
            inner.grpc.connect(port).await;
        }

        let monitor = Self {
            inner: self.inner.clone(),
        };
        let app_clone = app.clone();
            let mut rx = {
                let inner = self.inner.lock().await;
                inner
                    .cancel
                    .as_ref()
                    .map(|tx| tx.subscribe())
                    .expect("cancel channel just created")
            };
        tokio::spawn(async move {
            loop {
                monitor.tick(&app_clone).await;
                tokio::select! {
                    _ = tokio::time::sleep(Duration::from_millis(POLL_MS)) => {}
                    changed = rx.changed() => {
                        if changed.is_err() || *rx.borrow() {
                            break;
                        }
                    }
                }
            }
        });

        self.tick(&app).await;
    }

    pub async fn stop(&self, app: &AppHandle) {
        let mut inner = self.inner.lock().await;
        inner.enabled = false;
        if let Some(tx) = inner.cancel.take() {
            let _ = tx.send(true);
        }
        inner.grpc.close().await;
        inner.samples.clear();
        inner.percent_samples.clear();
        inner.ibd = None;
        inner.sync_started_at = None;
        inner.sync_finished_at = None;
        inner.saw_unsynced = false;
        inner.last_status = SyncStatus::empty();
        let status = inner.last_status.clone();
        drop(inner);
        let _ = app.emit("sync-status", status);
    }

    async fn tick(&self, app: &AppHandle) {
        {
            let mut inner = self.inner.lock().await;
            if !inner.enabled || inner.ticking {
                return;
            }
            inner.ticking = true;
        }

        let result = self.tick_inner().await;

        let mut inner = self.inner.lock().await;
        inner.ticking = false;
        if let Some(status) = result {
            inner.last_status = status.clone();
            drop(inner);
            let _ = app.emit("sync-status", status);
        }
    }

    async fn tick_inner(&self) -> Option<SyncStatus> {
        let now = now_ms();

        {
            let need_fetch = {
                let inner = self.inner.lock().await;
                inner.network_height.is_none()
                    || now.saturating_sub(inner.last_network_fetch) > NETWORK_REFRESH_MS
            };
            if need_fetch {
                if let Some(h) = fetch_network_height().await {
                    let mut inner = self.inner.lock().await;
                    inner.network_height = Some(h);
                    inner.last_network_fetch = now;
                }
            }
        }

        let probe = {
            let inner = self.inner.lock().await;
            // query_status uses interior mutability
            inner.grpc.query_status().await
        };

        let mut inner = self.inner.lock().await;

        let Some(probe) = probe else {
            if let Some(ibd) = inner.ibd.clone() {
                inner.saw_unsynced = true;
                let timing = timing_fields(&inner, now, Some(false));
                let eta = eta_from_percent_samples(&inner.percent_samples);
                return Some(SyncStatus {
                    available: true,
                    is_synced: Some(false),
                    sync_percent: Some(ibd.percent),
                    phase: ibd.phase,
                    network_height: inner.network_height,
                    tip_timestamp: ibd.tip_timestamp,
                    eta_seconds: eta,
                    eta_label: format_eta(eta),
                    header_count: (ibd.phase == SyncPhase::Headers).then_some(ibd.processed),
                    block_count: (ibd.phase == SyncPhase::Blocks).then_some(ibd.processed),
                    virtual_daa_score: None,
                    server_version: None,
                    mempool_size: None,
                    error: None,
                    sync_elapsed_seconds: timing.0,
                    sync_duration_seconds: timing.1,
                    sync_time_label: timing.2,
                    updated_at: now_ms(),
                });
            }
            let timing = timing_fields(&inner, now, None);
            return Some(SyncStatus {
                available: false,
                is_synced: None,
                sync_percent: None,
                phase: SyncPhase::Connecting,
                virtual_daa_score: None,
                network_height: inner.network_height,
                header_count: None,
                block_count: None,
                eta_seconds: None,
                eta_label: None,
                tip_timestamp: None,
                server_version: None,
                mempool_size: None,
                error: Some("RPC unavailable (node starting or gRPC disabled)".into()),
                sync_elapsed_seconds: timing.0,
                sync_duration_seconds: timing.1,
                sync_time_label: timing.2,
                updated_at: now_ms(),
            });
        };

        let daa = probe.virtual_daa_score;
        mark_synced_if_needed(&mut inner, now, probe.is_synced);

        if let Some(d) = daa {
            if d > 0 && probe.is_synced != Some(true) {
                inner.samples.push(DaaSample { t: now, daa: d });
                if inner.samples.len() > SAMPLE_WINDOW {
                    inner.samples.remove(0);
                }
            } else if probe.is_synced == Some(true) {
                inner.samples.clear();
                inner.percent_samples.clear();
                inner.ibd = None;
            }
        }

        let ibd_fresh = inner
            .ibd
            .as_ref()
            .map(|i| now.saturating_sub(i.updated_at) < 120_000)
            .unwrap_or(false);
        let in_ibd = inner.ibd.is_some()
            && probe.is_synced != Some(true)
            && (daa.is_none() || daa == Some(0) || ibd_fresh);

        let (sync_percent, phase, eta_seconds) = if probe.is_synced == Some(true) {
            (Some(100.0), SyncPhase::Synced, None)
        } else if in_ibd {
            let ibd = inner.ibd.as_ref().unwrap();
            (
                Some(ibd.percent),
                ibd.phase,
                eta_from_percent_samples(&inner.percent_samples),
            )
        } else {
            let pct = estimate_sync_percent(probe.is_synced, daa, inner.network_height);
            let phase = if daa.map(|d| d > 0).unwrap_or(false) {
                SyncPhase::Catchup
            } else if probe.header_count.map(|h| h > 0).unwrap_or(false) {
                SyncPhase::Headers
            } else {
                SyncPhase::Catchup
            };
            let eta = match (daa, inner.network_height) {
                (Some(d), Some(h)) => estimate_eta_seconds(&inner.samples, d, h),
                _ => None,
            };
            (pct, phase, eta)
        };

        let timing = timing_fields(&inner, now, probe.is_synced);
        Some(SyncStatus {
            available: true,
            is_synced: probe.is_synced,
            sync_percent,
            phase,
            virtual_daa_score: daa,
            network_height: inner.network_height,
            header_count: probe.header_count,
            block_count: probe.block_count,
            eta_seconds,
            eta_label: if probe.is_synced == Some(true) {
                Some("Synced".into())
            } else {
                format_eta(eta_seconds)
            },
            tip_timestamp: inner.ibd.as_ref().and_then(|i| i.tip_timestamp.clone()),
            server_version: probe.server_version.clone(),
            mempool_size: probe.mempool_size,
            error: None,
            sync_elapsed_seconds: timing.0,
            sync_duration_seconds: timing.1,
            sync_time_label: timing.2,
            updated_at: now_ms(),
        })
    }
}

fn timing_fields(
    inner: &SyncInner,
    now: u64,
    is_synced: Option<bool>,
) -> (Option<f64>, Option<f64>, Option<String>) {
    let Some(started) = inner.sync_started_at else {
        return (None, None, None);
    };
    if is_synced == Some(true) {
        if let Some(finished) = inner.sync_finished_at {
            let dur = (finished.saturating_sub(started)) as f64 / 1000.0;
            let label = if inner.saw_unsynced {
                format!("Synced in {}", format_duration(dur))
            } else {
                format!("Already synced ({} up)", format_duration(dur))
            };
            return (Some(dur), Some(dur), Some(label));
        }
        if !inner.saw_unsynced {
            let dur = (now.saturating_sub(started)) as f64 / 1000.0;
            return (Some(dur), Some(dur), Some("Already synced".into()));
        }
    }
    let elapsed = (now.saturating_sub(started)) as f64 / 1000.0;
    (
        Some(elapsed),
        None,
        Some(format!("Elapsed {}", format_duration(elapsed))),
    )
}

fn mark_synced_if_needed(inner: &mut SyncInner, now: u64, is_synced: Option<bool>) {
    if is_synced == Some(false) || (is_synced.is_none() && inner.ibd.is_some()) {
        inner.saw_unsynced = true;
    }
    if is_synced == Some(true) && inner.sync_finished_at.is_none() && inner.sync_started_at.is_some()
    {
        inner.sync_finished_at = Some(now);
        if inner.saw_unsynced {
            let started = inner.sync_started_at.unwrap();
            let dur = (now.saturating_sub(started)) as f64 / 1000.0;
            if let Some(cb) = inner.on_sync_complete.as_ref() {
                cb(dur);
            }
        }
    }
}

fn eta_from_percent_samples(samples: &[PercentSample]) -> Option<f64> {
    if samples.len() < 2 {
        return None;
    }
    let first = samples[0];
    let last = *samples.last().unwrap();
    let dt = (last.t.saturating_sub(first.t)) as f64 / 1000.0;
    let d_p = last.percent - first.percent;
    if dt < 3.0 || d_p <= 0.05 {
        return None;
    }
    let rate = d_p / dt;
    let remaining = (100.0 - last.percent).max(0.0);
    let eta = remaining / rate;
    if !eta.is_finite() || eta > 86400.0 * 30.0 {
        None
    } else {
        Some(eta)
    }
}

struct IbdCaps {
    phase: SyncPhase,
    percent: f64,
    processed: u64,
    tip: Option<String>,
}

fn parse_ibd(line: &str) -> Option<IbdCaps> {
    let idx = line.find("IBD:")?;
    let rest = line[idx + 4..].trim_start();
    if !rest.starts_with("Processed") {
        return None;
    }
    let rest = rest["Processed".len()..].trim_start();
    let (num_str, rest) = split_first_token(rest)?;
    let processed: u64 = num_str.parse().ok()?;
    let rest = rest.trim_start();
    let lower = rest.to_ascii_lowercase();
    let phase = if lower.starts_with("block headers") {
        SyncPhase::Headers
    } else if lower.starts_with("blocks") {
        SyncPhase::Blocks
    } else {
        return None;
    };
    let pct_start = rest.find('(')?;
    let after_paren = &rest[pct_start + 1..];
    let pct_end = after_paren.find('%')?;
    let percent: f64 = after_paren[..pct_end].parse().ok()?;
    let percent = percent.clamp(0.0, 100.0);
    let tip = lower
        .find("last block timestamp:")
        .map(|pos| rest[pos + "last block timestamp:".len()..].trim().to_string())
        .filter(|s| !s.is_empty());
    Some(IbdCaps {
        phase,
        percent,
        processed,
        tip,
    })
}

fn parse_headers_throughput(line: &str) -> Option<u64> {
    let lower = line.to_ascii_lowercase();
    if !lower.contains("headers in the last") {
        return None;
    }
    let idx = lower.find("and ")?;
    let after = line[idx + 4..].trim_start();
    let (num, _) = split_first_token(after)?;
    num.parse().ok()
}

fn split_first_token(s: &str) -> Option<(&str, &str)> {
    let s = s.trim_start();
    let end = s.find(char::is_whitespace).unwrap_or(s.len());
    if end == 0 {
        None
    } else {
        Some((&s[..end], &s[end..]))
    }
}
