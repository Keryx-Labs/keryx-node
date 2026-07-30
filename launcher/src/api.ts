import { invoke } from '@tauri-apps/api/core';
import { listen, type UnlistenFn } from '@tauri-apps/api/event';

export interface LauncherConfig {
  installDir: string;
  binarySource: string;
  appdir: string;
  datadirZip: string;
  extractZipOnInstall: boolean;
  network: 'mainnet' | 'testnet' | 'devnet' | 'simnet';
  testnetSuffix: number;
  ramScale: number;
  rocksdbPreset: 'default' | 'hdd' | 'hdd-qd1';
  rocksdbCacheSize: number;
  rocksdbRateLimitMb: number;
  rocksdbWalDir: string;
  peers: string[];
  connectOnly: boolean;
  logLevel: string;
  logdir: string;
  noLogFiles: boolean;
  utxoindex: boolean;
  archival: boolean;
  outboundTarget: number;
  inboundLimit: number;
  rpcMaxClients: number;
  rpclisten: string;
  rpclistenJson: string;
  rpclistenBorsh: string;
  listen: string;
  externalip: string;
  unsafeRpc: boolean;
  disableUpnp: boolean;
  disableDnsSeeding: boolean;
  disableGrpc: boolean;
  enableUnsyncedMining: boolean;
  resetDb: boolean;
  retentionPeriodDays: string;
  asyncThreads: number;
  userAgentComments: string;
}

export interface NodeStatus {
  running: boolean;
  pid: number | null;
  installReady: boolean;
  binaryPath: string | null;
}

export interface LauncherState {
  config: LauncherConfig;
  status: NodeStatus;
  setupComplete: boolean;
}

export interface ProgressEvent {
  phase: string;
  message: string;
  percent?: number;
}

export interface SyncStatus {
  available: boolean;
  isSynced: boolean | null;
  syncPercent: number | null;
  phase: 'idle' | 'connecting' | 'headers' | 'blocks' | 'catchup' | 'synced';
  virtualDaaScore: number | null;
  networkHeight: number | null;
  headerCount: number | null;
  blockCount: number | null;
  etaSeconds: number | null;
  etaLabel: string | null;
  syncElapsedSeconds: number | null;
  syncDurationSeconds: number | null;
  syncTimeLabel: string | null;
  tipTimestamp: string | null;
  serverVersion: string | null;
  mempoolSize: number | null;
  error: string | null;
  updatedAt: number;
}

export interface InstallResult {
  ok: boolean;
  error?: string;
  state?: LauncherState;
}

export interface LatestRelease {
  tag: string;
  name: string;
  size: number;
  downloadUrl: string;
}

export interface DownloadResult {
  ok: boolean;
  error?: string;
  binaryPath?: string;
  tag?: string;
}

export interface DiskInfo {
  media: string;
  drive: string | null;
  freeGb: number | null;
  confidence: string;
}

export interface PerformanceRecommendation {
  profile: string;
  rocksdbPreset: string;
  ramScale: number;
  rocksdbCacheSize: number;
  rocksdbRateLimitMb: number;
  budgetMb: number;
  estimateMb: number;
}

export interface MachineResources {
  totalRamMb: number;
  availableRamMb: number;
  disk: DiskInfo;
  recommendation: PerformanceRecommendation;
  warning?: string;
}

export interface RecommendInput {
  totalRamMb: number;
  availableRamMb: number;
  profile: string;
  drive?: string | null;
}

export interface PeerProbeResult {
  address: string;
  ok: boolean;
  rttMs?: number;
  error?: string;
}

export interface PeerProbeReport {
  results: PeerProbeResult[];
  selected: string[];
  warning?: string;
}

export interface KeryxApi {
  getState: () => Promise<LauncherState & { sync?: SyncStatus; lastSyncDurationSeconds?: number | null }>;
  getSyncStatus: () => Promise<SyncStatus>;
  saveConfig: (config: LauncherConfig) => Promise<LauncherState>;
  pickDirectory: (opts?: { title?: string; defaultPath?: string }) => Promise<string | null>;
  pickFile: (opts?: {
    title?: string;
    filters?: { name: string; extensions: string[] }[];
    defaultPath?: string;
  }) => Promise<string | null>;
  installNode: (config: LauncherConfig) => Promise<InstallResult>;
  startNode: (config: LauncherConfig) => Promise<InstallResult>;
  stopNode: () => Promise<InstallResult>;
  resetSetup: (clearConfig?: boolean) => Promise<LauncherState>;
  buildCommandPreview: (config: LauncherConfig) => Promise<string>;
  openPath: (p: string) => Promise<void>;
  openExternal: (url: string) => Promise<void>;
  fetchLatestRelease: () => Promise<LatestRelease>;
  downloadAndInstallRelease: (args: {
    downloadUrl: string;
    installDir: string;
  }) => Promise<DownloadResult>;
  detectMachineResources: (path: string) => Promise<MachineResources>;
  recommendPerformance: (input: RecommendInput) => Promise<PerformanceRecommendation>;
  probePeers: (addresses?: string[]) => Promise<PeerProbeReport>;
  onLog: (cb: (line: string) => void) => () => void;
  onStatus: (cb: (status: NodeStatus) => void) => () => void;
  onProgress: (cb: (progress: ProgressEvent) => void) => () => void;
  onSyncStatus: (cb: (sync: SyncStatus) => void) => () => void;
}

function eventUnsub(promise: Promise<UnlistenFn>): () => void {
  let unlisten: UnlistenFn | null = null;
  void promise.then((fn) => {
    unlisten = fn;
  });
  return () => {
    unlisten?.();
  };
}

export const api: KeryxApi = {
  getState: () => invoke('get_state'),
  getSyncStatus: () => invoke('get_sync_status'),
  saveConfig: (config) => invoke('save_config', { config }),
  pickDirectory: (opts) => invoke('pick_directory', { opts: opts ?? null }),
  pickFile: (opts) => invoke('pick_file', { opts: opts ?? null }),
  installNode: (config) => invoke('install_node', { config }),
  startNode: (config) => invoke('start_node', { config }),
  stopNode: () => invoke('stop_node'),
  resetSetup: (clearConfig) => invoke('reset_setup', { clearConfig: clearConfig ?? false }),
  buildCommandPreview: (config) => invoke('build_command_preview', { config }),
  openPath: (p) => invoke('open_path', { path: p }),
  openExternal: (url) => invoke('open_external', { url }),
  fetchLatestRelease: () => invoke('fetch_latest_release'),
  downloadAndInstallRelease: (args) => invoke('download_and_install_release', { args }),
  detectMachineResources: (path) => invoke('detect_machine_resources', { path }),
  recommendPerformance: (input) => invoke('recommend_performance', { input }),
  probePeers: (addresses) =>
    invoke('probe_peers', { args: addresses ? { addresses } : null }),
  onLog: (cb) => eventUnsub(listen<string>('node-log', (e) => cb(e.payload))),
  onStatus: (cb) => eventUnsub(listen<NodeStatus>('node-status', (e) => cb(e.payload))),
  onProgress: (cb) => eventUnsub(listen<ProgressEvent>('progress', (e) => cb(e.payload))),
  onSyncStatus: (cb) => eventUnsub(listen<SyncStatus>('sync-status', (e) => cb(e.payload))),
};
