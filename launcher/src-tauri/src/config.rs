use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum NetworkKind {
    Mainnet,
    Testnet,
    Devnet,
    Simnet,
}

impl Default for NetworkKind {
    fn default() -> Self {
        Self::Mainnet
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum RocksDbPreset {
    Default,
    Hdd,
    #[serde(rename = "hdd-qd1")]
    HddQd1,
}

impl Default for RocksDbPreset {
    fn default() -> Self {
        Self::Hdd
    }
}

impl RocksDbPreset {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Hdd => "hdd",
            Self::HddQd1 => "hdd-qd1",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LauncherConfig {
    pub install_dir: String,
    pub binary_source: String,
    pub appdir: String,
    pub datadir_zip: String,
    pub extract_zip_on_install: bool,
    pub network: NetworkKind,
    pub testnet_suffix: u32,
    pub ram_scale: f64,
    pub rocksdb_preset: RocksDbPreset,
    pub rocksdb_cache_size: u64,
    pub rocksdb_rate_limit_mb: u64,
    pub rocksdb_wal_dir: String,
    pub peers: Vec<String>,
    pub connect_only: bool,
    pub log_level: String,
    pub logdir: String,
    pub no_log_files: bool,
    pub utxoindex: bool,
    pub archival: bool,
    pub outbound_target: u32,
    pub inbound_limit: u32,
    pub rpc_max_clients: u32,
    pub rpclisten: String,
    pub rpclisten_json: String,
    pub rpclisten_borsh: String,
    pub listen: String,
    pub externalip: String,
    pub unsafe_rpc: bool,
    pub disable_upnp: bool,
    pub disable_dns_seeding: bool,
    pub disable_grpc: bool,
    pub enable_unsynced_mining: bool,
    pub reset_db: bool,
    pub retention_period_days: String,
    pub async_threads: u32,
    pub user_agent_comments: String,
}

impl Default for LauncherConfig {
    fn default() -> Self {
        Self {
            install_dir: String::new(),
            binary_source: String::new(),
            appdir: String::new(),
            datadir_zip: String::new(),
            extract_zip_on_install: false,
            network: NetworkKind::Mainnet,
            testnet_suffix: 10,
            ram_scale: 1.0,
            rocksdb_preset: RocksDbPreset::Hdd,
            rocksdb_cache_size: 0,
            rocksdb_rate_limit_mb: 48,
            rocksdb_wal_dir: String::new(),
            peers: Vec::new(),
            connect_only: false,
            log_level: "INFO".into(),
            logdir: String::new(),
            no_log_files: false,
            utxoindex: false,
            archival: false,
            outbound_target: 8,
            inbound_limit: 128,
            rpc_max_clients: 128,
            rpclisten: String::new(),
            rpclisten_json: String::new(),
            rpclisten_borsh: String::new(),
            listen: String::new(),
            externalip: String::new(),
            unsafe_rpc: false,
            disable_upnp: false,
            disable_dns_seeding: false,
            disable_grpc: false,
            enable_unsynced_mining: false,
            reset_db: false,
            retention_period_days: String::new(),
            async_threads: 0,
            user_agent_comments: String::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NodeStatus {
    pub running: bool,
    pub pid: Option<u32>,
    pub install_ready: bool,
    pub binary_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProgressEvent {
    pub phase: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub percent: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LauncherState {
    pub config: LauncherConfig,
    pub status: NodeStatus,
    pub setup_complete: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SyncPhase {
    Idle,
    Connecting,
    Headers,
    Blocks,
    Catchup,
    Synced,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SyncStatus {
    pub available: bool,
    pub is_synced: Option<bool>,
    pub sync_percent: Option<f64>,
    pub phase: SyncPhase,
    pub virtual_daa_score: Option<u64>,
    pub network_height: Option<u64>,
    pub header_count: Option<u64>,
    pub block_count: Option<u64>,
    pub eta_seconds: Option<f64>,
    pub eta_label: Option<String>,
    pub sync_elapsed_seconds: Option<f64>,
    pub sync_duration_seconds: Option<f64>,
    pub sync_time_label: Option<String>,
    pub tip_timestamp: Option<String>,
    pub server_version: Option<String>,
    pub mempool_size: Option<u64>,
    pub error: Option<String>,
    pub updated_at: u64,
}

impl SyncStatus {
    pub fn empty() -> Self {
        Self {
            available: false,
            is_synced: None,
            sync_percent: None,
            phase: SyncPhase::Idle,
            virtual_daa_score: None,
            network_height: None,
            header_count: None,
            block_count: None,
            eta_seconds: None,
            eta_label: None,
            sync_elapsed_seconds: None,
            sync_duration_seconds: None,
            sync_time_label: None,
            tip_timestamp: None,
            server_version: None,
            mempool_size: None,
            error: None,
            updated_at: now_ms(),
        }
    }
}

pub fn now_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

pub fn build_argv(config: &LauncherConfig) -> Vec<String> {
    let mut args = Vec::new();
    let appdir = if !config.appdir.is_empty() {
        config.appdir.clone()
    } else if !config.install_dir.is_empty() {
        let base = config.install_dir.trim_end_matches(['/', '\\']);
        format!("{base}/data")
    } else {
        String::new()
    };
    if !appdir.is_empty() {
        args.push(format!("--appdir={appdir}"));
    }

    match config.network {
        NetworkKind::Testnet => {
            args.push("--testnet".into());
            if config.testnet_suffix != 10 {
                args.push(format!("--netsuffix={}", config.testnet_suffix));
            }
        }
        NetworkKind::Devnet => args.push("--devnet".into()),
        NetworkKind::Simnet => args.push("--simnet".into()),
        NetworkKind::Mainnet => {}
    }

    if config.ram_scale > 0.0 {
        args.push(format!("--ram-scale={}", config.ram_scale));
    }
    args.push(format!("--rocksdb-preset={}", config.rocksdb_preset.as_str()));
    if config.rocksdb_cache_size > 0 {
        args.push(format!("--rocksdb-cache-size={}", config.rocksdb_cache_size));
    }
    if config.rocksdb_preset != RocksDbPreset::Default && config.rocksdb_rate_limit_mb > 0 {
        args.push(format!("--rocksdb-rate-limit-mb={}", config.rocksdb_rate_limit_mb));
    }
    let wal = config.rocksdb_wal_dir.trim();
    if !wal.is_empty() {
        args.push(format!("--rocksdb-wal-dir={wal}"));
    }

    let peer_flag = if config.connect_only { "--connect" } else { "--addpeer" };
    for peer in config.peers.iter().map(|p| p.trim()).filter(|p| !p.is_empty()) {
        args.push(format!("{peer_flag}={peer}"));
    }

    let log_level = config.log_level.trim();
    if !log_level.is_empty() {
        args.push(format!("--loglevel={log_level}"));
    }
    let logdir = config.logdir.trim();
    if !logdir.is_empty() {
        args.push(format!("--logdir={logdir}"));
    }
    if config.no_log_files {
        args.push("--nologfiles".into());
    }
    if config.utxoindex {
        args.push("--utxoindex".into());
    }
    if config.archival {
        args.push("--archival".into());
    }
    if config.outbound_target > 0 {
        args.push(format!("--outpeers={}", config.outbound_target));
    }
    if config.inbound_limit > 0 {
        args.push(format!("--maxinpeers={}", config.inbound_limit));
    }
    if config.rpc_max_clients > 0 {
        args.push(format!("--rpcmaxclients={}", config.rpc_max_clients));
    }
    let rpclisten = config.rpclisten.trim();
    if !rpclisten.is_empty() {
        args.push(format!("--rpclisten={rpclisten}"));
    }
    let rpclisten_json = config.rpclisten_json.trim();
    if !rpclisten_json.is_empty() {
        args.push(format!("--rpclisten-json={rpclisten_json}"));
    }
    let rpclisten_borsh = config.rpclisten_borsh.trim();
    if !rpclisten_borsh.is_empty() {
        args.push(format!("--rpclisten-borsh={rpclisten_borsh}"));
    }
    let listen = config.listen.trim();
    if !listen.is_empty() {
        args.push(format!("--listen={listen}"));
    }
    let externalip = config.externalip.trim();
    if !externalip.is_empty() {
        args.push(format!("--externalip={externalip}"));
    }
    if config.unsafe_rpc {
        args.push("--unsaferpc".into());
    }
    if config.disable_upnp {
        args.push("--disable-upnp".into());
    }
    if config.disable_dns_seeding {
        args.push("--nodnsseed".into());
    }
    if config.disable_grpc {
        args.push("--nogrpc".into());
    }
    if config.enable_unsynced_mining {
        args.push("--enable-unsynced-mining".into());
    }
    if config.reset_db {
        args.push("--reset-db".into());
    }
    let retention = config.retention_period_days.trim();
    if !retention.is_empty() {
        args.push(format!("--retention-period-days={retention}"));
    }
    if config.async_threads > 0 {
        args.push(format!("--async-threads={}", config.async_threads));
    }
    let ua = config.user_agent_comments.trim();
    if !ua.is_empty() {
        for c in ua.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()) {
            args.push(format!("--uacomment={c}"));
        }
    }

    args
}

pub fn resolve_appdir(config: &LauncherConfig) -> String {
    let trimmed = config.appdir.trim();
    if !trimmed.is_empty() {
        return trimmed.to_string();
    }
    let install = config.install_dir.trim();
    if !install.is_empty() {
        let base = install.trim_end_matches(['/', '\\']);
        return format!("{base}\\data");
    }
    String::new()
}

/// Relative datadir under `--appdir` for the selected network (`keryx-mainnet/datadir`, …).
pub fn network_datadir_rel(config: &LauncherConfig) -> std::path::PathBuf {
    let folder = match config.network {
        NetworkKind::Mainnet => "keryx-mainnet".to_string(),
        NetworkKind::Testnet => format!("keryx-testnet-{}", config.testnet_suffix),
        NetworkKind::Devnet => "keryx-devnet".to_string(),
        NetworkKind::Simnet => "keryx-simnet".to_string(),
    };
    std::path::PathBuf::from(folder).join("datadir")
}

/// If `path` is a directory, look for `keryxd.exe` / `keryxd` inside it (non-recursive first, then shallow).
pub fn resolve_binary_source(path: &str) -> Option<std::path::PathBuf> {
    let p = std::path::Path::new(path.trim());
    if !p.exists() {
        return None;
    }
    if p.is_file() {
        return Some(p.to_path_buf());
    }
    if p.is_dir() {
        #[cfg(windows)]
        {
            let exe = p.join("keryxd.exe");
            if exe.is_file() {
                return Some(exe);
            }
        }
        let unix = p.join("keryxd");
        if unix.is_file() {
            return Some(unix);
        }
        // One level deep (release zip layout)
        if let Ok(entries) = std::fs::read_dir(p) {
            for entry in entries.flatten() {
                let child = entry.path();
                if child.is_dir() {
                    #[cfg(windows)]
                    {
                        let exe = child.join("keryxd.exe");
                        if exe.is_file() {
                            return Some(exe);
                        }
                    }
                    let nested = child.join("keryxd");
                    if nested.is_file() {
                        return Some(nested);
                    }
                }
            }
        }
    }
    None
}

pub fn binary_path_in_install(install_dir: &str) -> std::path::PathBuf {
    let name = if cfg!(windows) { "keryxd.exe" } else { "keryxd" };
    std::path::Path::new(install_dir).join("bin").join(name)
}

pub fn is_install_ready(install_dir: &str) -> bool {
    !install_dir.is_empty() && binary_path_in_install(install_dir).is_file()
}

/// Directory that contains the launcher executable (MSI install folder / portable root).
pub fn launcher_dir() -> std::path::PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| std::path::PathBuf::from("."))
}

/// Suggested (install_dir, binary_source) for first run.
pub fn suggested_paths() -> (String, String) {
    let install = suggested_install_dir();
    let binary = suggested_binary();
    (install, binary)
}

/// Default node install root = launcher folder (self-contained / portable).
fn suggested_install_dir() -> String {
    launcher_dir().display().to_string()
}

fn suggested_binary() -> String {
    let repo_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("..");
    #[cfg(windows)]
    {
        let win = repo_root.join("target").join("release").join("keryxd.exe");
        if win.is_file() {
            return win.display().to_string();
        }
    }
    let unix = repo_root.join("target").join("release").join("keryxd");
    if unix.is_file() {
        return unix.display().to_string();
    }
    String::new()
}
