/** Config-driven UI. Persist via Tauri store on every save/install/start. */

import {
  api,
  type LauncherConfig,
  type LauncherState,
  type LatestRelease,
  type MachineResources,
  type NodeStatus,
  type PerformanceRecommendation,
  type ProgressEvent,
  type SyncStatus,
} from './api';

const WIZARD_STEPS = 6;

let config: LauncherConfig;
let setupComplete = false;
let wizardStep = 1;
let activeTab: 'control' | 'settings' = 'control';
let nodeRunning = false;

let binaryMode: 'download' | 'local' = 'download';
let syncMode: 'scratch' | 'zip' = 'scratch';
let storageProfile: 'ssd' | 'hdd' = 'hdd';
let latestRelease: LatestRelease | null = null;
let machineResources: MachineResources | null = null;
let lastSuggestion: PerformanceRecommendation | null = null;
let userOverrodePerformance = false;
let performanceLoadedForPath = '';
let peersTuned = false;
let peerTuneSummary = 'Peers: none yet (will auto-tune on Install step)';
let peerTuneBestRtt: number | null = null;

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Missing #${id}`);
  return el as T;
};

function setInput(id: string, v: string | number | boolean): void {
  const el = document.getElementById(id) as HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement | null;
  if (!el) return;
  if (el instanceof HTMLInputElement && el.type === 'checkbox') el.checked = Boolean(v);
  else el.value = String(v ?? '');
}

function getInput(id: string): string {
  return (document.getElementById(id) as HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement).value;
}

function getCheck(id: string): boolean {
  const el = document.getElementById(id) as HTMLInputElement | null;
  return el ? el.checked : false;
}

function estimateProcessMb(isSsd: boolean, ramScale: number, cacheMb: number): number {
  const memtableMb = isSsd ? Math.max(256 * ramScale, 64) : Math.max(768 * ramScale, 512);
  const cachesMb = (64 + 128 + 128 + 64) * ramScale;
  return cacheMb + memtableMb + cachesMb + 1500;
}

/** Pull wizard DOM → config */
function harvestWizard(): void {
  config.installDir = getInput('installDir').trim();
  config.appdir = getInput('appdir').trim();
  config.network = getInput('network') as LauncherConfig['network'];
  config.binarySource = getInput('binarySource').trim();
  config.datadirZip = getInput('datadirZip').trim();
  config.extractZipOnInstall = syncMode === 'zip' && !!config.datadirZip;
  if (syncMode === 'scratch') {
    config.datadirZip = '';
    config.extractZipOnInstall = false;
  }
  config.ramScale = Number(getInput('ramScale')) || 1;
  config.rocksdbCacheSize = Number(getInput('rocksdbCacheSize')) || 0;
  config.rocksdbRateLimitMb = Number(getInput('rocksdbRateLimitMb')) || 0;
  config.rocksdbPreset = storageProfile === 'ssd' ? 'default' : 'hdd';
  if (storageProfile === 'ssd') config.rocksdbRateLimitMb = 0;
}

/** Pull settings DOM → config */
function harvestSettings(): void {
  config.installDir = getInput('s_installDir').trim();
  config.appdir = getInput('s_appdir').trim();
  config.binarySource = getInput('s_binarySource').trim();
  config.network = getInput('s_network') as LauncherConfig['network'];
  config.testnetSuffix = Number(getInput('s_testnetSuffix')) || 10;
  config.ramScale = Number(getInput('s_ramScale')) || 1;
  config.rocksdbPreset = getInput('s_rocksdbPreset') as LauncherConfig['rocksdbPreset'];
  config.rocksdbCacheSize = Number(getInput('s_rocksdbCacheSize')) || 0;
  config.rocksdbRateLimitMb = Number(getInput('s_rocksdbRateLimitMb')) || 0;
  config.rocksdbWalDir = getInput('s_rocksdbWalDir').trim();
  config.connectOnly = getCheck('s_connectOnly');
  config.peers = getInput('s_peers')
    .split(/\r?\n/)
    .map((s) => s.trim())
    .filter(Boolean);
  config.listen = getInput('s_listen').trim();
  config.externalip = getInput('s_externalip').trim();
  config.outboundTarget = Number(getInput('s_outboundTarget')) || 0;
  config.inboundLimit = Number(getInput('s_inboundLimit')) || 0;
  config.rpclisten = getInput('s_rpclisten').trim();
  config.rpclistenJson = getInput('s_rpclistenJson').trim();
  config.rpclistenBorsh = getInput('s_rpclistenBorsh').trim();
  config.rpcMaxClients = Number(getInput('s_rpcMaxClients')) || 0;
  config.disableUpnp = getCheck('s_disableUpnp');
  config.disableDnsSeeding = getCheck('s_disableDnsSeeding');
  config.disableGrpc = getCheck('s_disableGrpc');
  config.unsafeRpc = getCheck('s_unsafeRpc');
  config.enableUnsyncedMining = getCheck('s_enableUnsyncedMining');
  config.logLevel = getInput('s_logLevel');
  config.asyncThreads = Number(getInput('s_asyncThreads')) || 0;
  config.retentionPeriodDays = getInput('s_retentionPeriodDays').trim();
  config.userAgentComments = getInput('s_userAgentComments').trim();
  config.logdir = getInput('s_logdir').trim();
  config.noLogFiles = getCheck('s_noLogFiles');
  config.utxoindex = getCheck('s_utxoindex');
  config.archival = getCheck('s_archival');
  config.resetDb = getCheck('s_resetDb');
}

function paintWizard(): void {
  setInput('installDir', config.installDir);
  setInput('appdir', config.appdir);
  setInput('network', config.network);
  setInput('binarySource', config.binarySource);
  setInput('datadirZip', config.datadirZip);
  setInput('ramScale', config.ramScale);
  setInput('rocksdbCacheSize', config.rocksdbCacheSize);
  setInput('rocksdbRateLimitMb', config.rocksdbRateLimitMb);
  storageProfile = config.rocksdbPreset === 'default' ? 'ssd' : 'hdd';
  syncMode = config.datadirZip && config.extractZipOnInstall ? 'zip' : 'scratch';
  if (config.binarySource) binaryMode = 'local';
  paintBinaryMode();
  paintSyncMode();
  paintStorageProfile();
  updateRamPreview();
}

function paintSettings(): void {
  setInput('s_installDir', config.installDir);
  setInput('s_appdir', config.appdir);
  setInput('s_binarySource', config.binarySource);
  setInput('s_network', config.network);
  setInput('s_testnetSuffix', config.testnetSuffix);
  setInput('s_ramScale', config.ramScale);
  setInput('s_rocksdbPreset', config.rocksdbPreset);
  setInput('s_rocksdbCacheSize', config.rocksdbCacheSize);
  setInput('s_rocksdbRateLimitMb', config.rocksdbRateLimitMb);
  setInput('s_rocksdbWalDir', config.rocksdbWalDir);
  setInput('s_connectOnly', config.connectOnly);
  setInput('s_peers', config.peers.join('\n'));
  setInput('s_listen', config.listen);
  setInput('s_externalip', config.externalip);
  setInput('s_outboundTarget', config.outboundTarget);
  setInput('s_inboundLimit', config.inboundLimit);
  setInput('s_rpclisten', config.rpclisten);
  setInput('s_rpclistenJson', config.rpclistenJson);
  setInput('s_rpclistenBorsh', config.rpclistenBorsh);
  setInput('s_rpcMaxClients', config.rpcMaxClients);
  setInput('s_disableUpnp', config.disableUpnp);
  setInput('s_disableDnsSeeding', config.disableDnsSeeding);
  setInput('s_disableGrpc', config.disableGrpc);
  setInput('s_unsafeRpc', config.unsafeRpc);
  setInput('s_enableUnsyncedMining', config.enableUnsyncedMining);
  setInput('s_logLevel', config.logLevel);
  setInput('s_asyncThreads', config.asyncThreads);
  setInput('s_retentionPeriodDays', config.retentionPeriodDays);
  setInput('s_userAgentComments', config.userAgentComments);
  setInput('s_logdir', config.logdir);
  setInput('s_noLogFiles', config.noLogFiles);
  setInput('s_utxoindex', config.utxoindex);
  setInput('s_archival', config.archival);
  setInput('s_resetDb', config.resetDb);
}

function currentConfig(): LauncherConfig {
  if (!setupComplete) harvestWizard();
  else if (activeTab === 'settings') harvestSettings();
  return config;
}

function applyState(state: LauncherState): void {
  config = state.config;
  setupComplete = state.setupComplete;
  if (setupComplete) activeTab = 'control';
  paintWizard();
  paintSettings();
  setStatus(state.status);
  renderMode();
}

function setViewVisible(el: HTMLElement, visible: boolean): void {
  el.hidden = !visible;
  el.classList.toggle('is-hidden', !visible);
}

function renderMode(): void {
  const wizard = $('viewWizard');
  const control = $('viewControl');
  const settings = $('viewSettings');
  const tabs = $('mainTabs');

  if (!setupComplete) {
    setViewVisible(wizard, true);
    setViewVisible(control, false);
    setViewVisible(settings, false);
    setViewVisible(tabs, false);
    showWizardStep(wizardStep);
    return;
  }

  setViewVisible(wizard, false);
  setViewVisible(tabs, true);
  setViewVisible(control, activeTab === 'control');
  setViewVisible(settings, activeTab === 'settings');
  tabs.querySelectorAll<HTMLButtonElement>('.tab').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.tab === activeTab);
  });
  if (activeTab === 'settings') {
    paintSettings();
    void refreshPreview();
  }
}

function paintBinaryMode(): void {
  document.querySelectorAll<HTMLButtonElement>('[data-binary-mode]').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.binaryMode === binaryMode);
  });
  $('binaryDownloadPanel').hidden = binaryMode !== 'download';
  $('binaryLocalPanel').hidden = binaryMode !== 'local';
}

function paintSyncMode(): void {
  document.querySelectorAll<HTMLButtonElement>('[data-sync-mode]').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.syncMode === syncMode);
  });
  $('syncZipPanel').hidden = syncMode !== 'zip';
}

function paintStorageProfile(): void {
  document.querySelectorAll<HTMLButtonElement>('[data-storage-profile]').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.storageProfile === storageProfile);
  });
  const rateInput = $('rocksdbRateLimitMb') as HTMLInputElement;
  const note = $('rateLimitNote');
  if (storageProfile === 'ssd') {
    rateInput.disabled = true;
    rateInput.value = '0';
    note.textContent = 'Not used on SSD';
  } else {
    rateInput.disabled = false;
    note.textContent = 'Lower if the disk is shared with the OS';
  }
  updateRamPreview();
}

function applySuggestion(rec: PerformanceRecommendation, force = false): void {
  lastSuggestion = rec;
  if (userOverrodePerformance && !force) {
    updateRamPreview();
    return;
  }
  storageProfile = rec.profile === 'ssd' ? 'ssd' : 'hdd';
  config.rocksdbPreset = (rec.rocksdbPreset as LauncherConfig['rocksdbPreset']) || (storageProfile === 'ssd' ? 'default' : 'hdd');
  config.ramScale = rec.ramScale;
  config.rocksdbCacheSize = rec.rocksdbCacheSize;
  config.rocksdbRateLimitMb = rec.rocksdbRateLimitMb;
  setInput('ramScale', config.ramScale);
  setInput('rocksdbCacheSize', config.rocksdbCacheSize);
  setInput('rocksdbRateLimitMb', config.rocksdbRateLimitMb);
  paintStorageProfile();
  updateRamPreview();
}

function updateRamPreview(): void {
  const scaleEl = document.getElementById('ramScale') as HTMLInputElement | null;
  const cacheEl = document.getElementById('rocksdbCacheSize') as HTMLInputElement | null;
  if (!scaleEl || !cacheEl) return;
  const scale = Number(scaleEl.value) || 1;
  const cache = Number(cacheEl.value) || 0;
  $('ramScaleValue').textContent = scale.toFixed(2).replace(/\.?0+$/, '') || String(scale);
  const isSsd = storageProfile === 'ssd';
  const estimateMb = estimateProcessMb(isSsd, scale, cache);
  const budgetMb = lastSuggestion?.budgetMb ?? machineResources?.recommendation.budgetMb ?? 0;
  const total = machineResources?.totalRamMb ?? 0;
  const avail = machineResources?.availableRamMb ?? 0;
  const estGb = estimateMb / 1024;
  const parts = [`Estimated RAM (theoretical): ~${estGb.toFixed(1)} GB (${storageProfile.toUpperCase()} profile)`];
  if (total > 0) {
    parts.push(`Machine now: ${(avail / 1024).toFixed(1)} GB free of ${(total / 1024).toFixed(0)} GB`);
  }
  $('ramMeterLabel').textContent = parts.join(' · ');
  const fill = $('ramMeterFill');
  if (budgetMb > 0) {
    fill.style.width = `${Math.max(4, Math.min(100, (estimateMb / budgetMb) * 100))}%`;
    fill.classList.toggle('over', estimateMb > budgetMb);
  } else {
    fill.style.width = '40%';
    fill.classList.remove('over');
  }
}

async function loadReleaseMeta(): Promise<void> {
  const meta = $('releaseMeta');
  try {
    latestRelease = await api.fetchLatestRelease();
    const mb = (latestRelease.size / (1024 * 1024)).toFixed(1);
    meta.textContent = `Latest: ${latestRelease.tag} · ${latestRelease.name} · ${mb} MB`;
  } catch (e) {
    meta.textContent = `Could not reach GitHub Releases: ${e}`;
    latestRelease = null;
  }
}

async function loadMachineResources(force = false): Promise<void> {
  const path = getInput('appdir').trim() || getInput('installDir').trim();
  if (!path) {
    setHwLoading(false);
    $('hwMeta').textContent = 'Set install / appdir first so we can detect the disk.';
    return;
  }
  if (!force && performanceLoadedForPath === path && machineResources) {
    setHwLoading(false);
    return;
  }

  const steps = [
    'Checking available memory…',
    'Detecting disk type (SSD or HDD)…',
    'Measuring free space on the data drive…',
    'Calculating recommended RAM scale, cache, and write rate…',
  ];
  setHwLoading(true, steps[0]);
  let stepIdx = 0;
  const tick = window.setInterval(() => {
    stepIdx = Math.min(stepIdx + 1, steps.length - 1);
    setHwLoadingDetail(steps[stepIdx]);
  }, 900);

  try {
    machineResources = await api.detectMachineResources(path);
    performanceLoadedForPath = path;
    const d = machineResources.disk;
    const drive = d.drive || '?';
    const free = d.freeGb != null ? ` · ~${d.freeGb.toFixed(0)} GB free` : '';
    $('hwMeta').textContent = `Detected: ${(machineResources.totalRamMb / 1024).toFixed(0)} GB RAM · disk ${drive} = ${d.media}${free}`;
    const warn = $('hwWarning');
    if (machineResources.warning) {
      warn.hidden = false;
      warn.textContent = machineResources.warning;
    } else {
      warn.hidden = true;
    }
    const recProfile = machineResources.recommendation.profile === 'ssd' ? 'ssd' : 'hdd';
    $('ssdBadge').hidden = recProfile !== 'ssd';
    $('hddBadge').hidden = recProfile !== 'hdd';
    applySuggestion(machineResources.recommendation, force);
  } catch (e) {
    $('hwMeta').textContent = `Hardware detection failed: ${e}`;
  } finally {
    window.clearInterval(tick);
    setHwLoading(false);
  }
}

function setHwLoading(loading: boolean, detail?: string): void {
  const panel = $('hwLoading');
  const results = $('hwResults');
  panel.hidden = !loading;
  results.classList.toggle('is-loading', loading);
  if (loading && detail) setHwLoadingDetail(detail);
  // Keep Continue disabled while analyzing so the user waits for suggestions
  const next = $('btnWizardNext') as HTMLButtonElement;
  if (wizardStep === 5) next.disabled = loading;
}

function setHwLoadingDetail(detail: string): void {
  $('hwLoadingDetail').textContent = detail;
}

async function showWizardStep(step: number): Promise<void> {
  wizardStep = Math.max(1, Math.min(WIZARD_STEPS, step));
  document.querySelectorAll<HTMLElement>('[data-wizard-step]').forEach((panel) => {
    panel.hidden = Number(panel.dataset.wizardStep) !== wizardStep;
  });
  document.querySelectorAll<HTMLElement>('#stepIndicators .step').forEach((el) => {
    const n = Number(el.dataset.step);
    el.classList.toggle('active', n === wizardStep);
    el.classList.toggle('done', n < wizardStep);
  });

  ($('btnWizardBack') as HTMLButtonElement).disabled = wizardStep <= 1;
  const isLast = wizardStep === WIZARD_STEPS;
  $('btnWizardNext').hidden = isLast;
  ($('btnWizardNext') as HTMLButtonElement).disabled = false;
  $('btnWizardInstall').hidden = !isLast;

  if (wizardStep === 2 && binaryMode === 'download') {
    void loadReleaseMeta();
  }
  if (wizardStep === 5) {
    await loadMachineResources(false);
  }
  if (isLast) {
    harvestWizard();
    paintInstallSummary();
    if (!peersTuned) {
      await autoTunePeers(true);
      paintInstallSummary();
    }
  }
}

function paintInstallSummary(): void {
  const profileLabel = storageProfile === 'ssd' ? 'SSD' : 'HDD';
  $('installSummary').textContent = [
    `Install:  ${config.installDir}`,
    `Appdir:   ${config.appdir}`,
    `Binary:   ${config.binarySource || '(download / install path)'}`,
    `Sync:     ${syncMode === 'zip' ? `restore ${config.datadirZip}` : 'from scratch (peers)'}`,
    `Network:  ${config.network}`,
    `Storage:  ${profileLabel} (${config.rocksdbPreset})`,
    `RAM:      scale ${config.ramScale} · cache ${config.rocksdbCacheSize} MB · rate ${config.rocksdbRateLimitMb || 'n/a'} MB/s`,
    peerTuneSummary,
  ].join('\n');
}

async function autoTunePeers(fromWizard: boolean): Promise<void> {
  setPeerLoading(true, 'Fetching peer IPs from keryx-labs.com…');

  const warnEl = document.getElementById('peerTuneWarning');
  if (warnEl) warnEl.hidden = true;

  try {
    const report = await api.probePeers();
    config.peers = report.selected;
    config.connectOnly = false;
    config.disableDnsSeeding = false;
    peersTuned = true;
    peerTuneBestRtt = null;
    if (report.selected.length === 0) {
      peerTuneSummary = 'Peers: none (DNS seed only at runtime)';
    } else {
      peerTuneSummary = `Peers: ${report.selected.length} from network directory`;
    }
    if (report.warning && warnEl) {
      warnEl.hidden = false;
      warnEl.textContent = report.warning;
    }
    if (!fromWizard) {
      setInput('s_peers', config.peers.join('\n'));
    }
    await api.saveConfig(config);
  } catch (e) {
    peersTuned = true;
    peerTuneSummary = 'Peers: none (fetch failed — DNS seed only)';
    if (warnEl) {
      warnEl.hidden = false;
      warnEl.textContent = `Peer auto-tune failed: ${e}`;
    }
  } finally {
    setPeerLoading(false);
  }
}

function setPeerLoading(loading: boolean, detail?: string): void {
  const panel = document.getElementById('peerLoading');
  if (!panel) return;
  panel.hidden = !loading;
  if (loading && detail) setPeerLoadingDetail(detail);
  if (wizardStep === 6) {
    ($('btnWizardInstall') as HTMLButtonElement).disabled = loading;
    ($('btnWizardBack') as HTMLButtonElement).disabled = loading || wizardStep <= 1;
  }
}

function setPeerLoadingDetail(detail: string): void {
  const el = document.getElementById('peerLoadingDetail');
  if (el) el.textContent = detail;
}

function formatDaa(n: number | null): string {
  if (n == null || !Number.isFinite(n)) return '—';
  return Math.floor(n).toLocaleString('en-US');
}

let lastSyncDurationSeconds: number | null = null;

function formatDurationUi(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  const days = Math.floor(s / 86400);
  const hours = Math.floor((s % 86400) / 3600);
  const mins = Math.floor((s % 3600) / 60);
  const secs = s % 60;
  if (days > 0) return `${days}d ${hours}h ${mins}m`;
  if (hours > 0) return `${hours}h ${mins}m ${secs}s`;
  if (mins > 0) return `${mins}m ${secs}s`;
  return `${secs}s`;
}

function resetSyncUi(): void {
  const badge = $('syncBadge');
  badge.className = 'sync-badge mono waiting';
  badge.textContent = 'Sync —';
  $('syncEta').textContent = '';
  $('syncBarFill').style.width = '0%';
  if (lastSyncDurationSeconds != null) {
    $('syncTime').textContent = `Last sync took ${formatDurationUi(lastSyncDurationSeconds)}`;
    $('syncMeta').textContent = 'Start the node to track a new sync.';
  } else {
    $('syncTime').textContent = '';
    $('syncMeta').textContent = 'Start the node to track sync progress.';
  }
}

function paintSync(sync: SyncStatus): void {
  const badge = $('syncBadge');
  const eta = $('syncEta');
  const fill = $('syncBarFill');
  const meta = $('syncMeta');
  const syncTime = $('syncTime');

  if (!nodeRunning) {
    resetSyncUi();
    return;
  }

  if (!sync.available && sync.phase === 'connecting') {
    badge.className = 'sync-badge mono waiting';
    badge.textContent = 'Sync · connecting…';
    eta.textContent = '';
    fill.style.width = '0%';
    syncTime.textContent = sync.syncTimeLabel || '';
    meta.textContent = sync.error || 'Waiting for gRPC on localhost…';
    return;
  }

  if (sync.isSynced === true || sync.phase === 'synced') {
    badge.className = 'sync-badge mono synced';
    badge.textContent = 'Synced · 100%';
    eta.textContent = 'ETA — done';
    fill.style.width = '100%';
  } else if (typeof sync.syncPercent === 'number') {
    badge.className = 'sync-badge mono syncing';
    const phaseLabel =
      sync.phase === 'headers'
        ? 'IBD headers'
        : sync.phase === 'blocks'
          ? 'IBD blocks'
          : 'Syncing';
    badge.textContent = `${phaseLabel} · ${sync.syncPercent.toFixed(sync.phase === 'catchup' ? 1 : 0)}%`;
    eta.textContent = sync.etaLabel ? `ETA ${sync.etaLabel}` : 'ETA · calculating…';
    fill.style.width = `${Math.max(0, Math.min(100, sync.syncPercent))}%`;
  } else {
    badge.className = 'sync-badge mono waiting';
    badge.textContent = sync.phase === 'headers' ? 'IBD headers · …' : 'Syncing · …';
    eta.textContent = '';
    fill.style.width = '0%';
  }

  syncTime.textContent = sync.syncTimeLabel || '';
  if (typeof sync.syncDurationSeconds === 'number') {
    lastSyncDurationSeconds = sync.syncDurationSeconds;
  }

  const parts: string[] = [];
  if (sync.headerCount != null) parts.push(`headers ${formatDaa(sync.headerCount)}`);
  if (sync.blockCount != null) parts.push(`blocks ${formatDaa(sync.blockCount)}`);
  parts.push(`DAA ${formatDaa(sync.virtualDaaScore)} / ${formatDaa(sync.networkHeight)}`);
  if (sync.tipTimestamp) parts.push(`tip ${sync.tipTimestamp}`);
  if (sync.mempoolSize != null) parts.push(`mempool ${sync.mempoolSize}`);
  if (sync.serverVersion) parts.push(`v${sync.serverVersion}`);
  meta.textContent = parts.join(' · ');
}

function setRunButtons(running: boolean): void {
  nodeRunning = running;
  const toggle = $('btnRunToggle') as HTMLButtonElement;
  const restart = $('btnRestart') as HTMLButtonElement;
  toggle.disabled = false;
  toggle.dataset.running = running ? 'true' : 'false';
  toggle.textContent = running ? 'Stop' : 'Start node';
  toggle.classList.toggle('primary', !running);
  toggle.classList.toggle('danger', running);
  restart.disabled = !running;
  $('controlTitle').textContent = running ? 'Running' : 'Ready';
  if (!running) resetSyncUi();
}

async function startNodeAction(): Promise<boolean> {
  clearLogs();
  appendLog('[launcher] Starting…\n');
  const cfg = currentConfig();
  const r = await api.startNode(cfg);
  if (!r.ok) {
    appendLog(`[launcher] Start failed: ${r.error}\n`);
    setRunButtons(false);
    return false;
  }
  if (r.state) applyState(r.state);
  setRunButtons(true);
  return true;
}

async function stopNodeAction(): Promise<boolean> {
  appendLog('[launcher] Stopping…\n');
  const r = await api.stopNode();
  if (!r.ok) {
    appendLog(`[launcher] Stop failed: ${r.error}\n`);
    setRunButtons(nodeRunning);
    return false;
  }
  setRunButtons(false);
  resetSyncUi();
  clearLogs();
  appendLog('[launcher] Stopped.\n');
  return true;
}

function setStatus(status: NodeStatus): void {
  const pill = $('statusPill');
  if (status.running) {
    pill.classList.add('running');
    $('statusText').textContent = `Running${status.pid ? ` · pid ${status.pid}` : ''}`;
    setRunButtons(true);
  } else {
    pill.classList.remove('running');
    $('statusText').textContent = setupComplete ? 'Installed · Stopped' : 'Setup';
    setRunButtons(false);
  }
  $('controlPath').textContent = status.binaryPath || config.installDir || '—';
}

function clearLogs(): void {
  $('logs').textContent = '';
}

function appendLog(line: string): void {
  const logs = $('logs');
  logs.textContent += line;
  if (logs.textContent.length > 400_000) logs.textContent = logs.textContent.slice(-300_000);
  logs.scrollTop = logs.scrollHeight;
}

function showProgress(p: ProgressEvent): void {
  $('progressWrap').hidden = false;
  $('progressMsg').textContent = `[${p.phase}] ${p.message}`;
  if (typeof p.percent === 'number') {
    $('progressFill').style.width = `${Math.max(0, Math.min(100, p.percent))}%`;
  }
}

async function refreshPreview(): Promise<void> {
  $('cmdPreview').textContent = await api.buildCommandPreview(currentConfig());
}

function validateStep(step: number): string | null {
  harvestWizard();
  if (step === 1) {
    if (!config.installDir) return 'Choose an install directory.';
    if (!config.appdir) return 'Choose an app data directory.';
  }
  if (step === 2) {
    if (!config.binarySource) {
      return binaryMode === 'download'
        ? 'Download the latest release first (or switch to a local build).'
        : 'Select the keryxd.exe file or the folder that contains it.';
    }
  }
  if (step === 4 && syncMode === 'zip') {
    if (!config.datadirZip) return 'Choose a datadir.zip, or switch to sync from scratch.';
  }
  return null;
}

function setupTooltips(): void {
  let bubble = document.getElementById('tipBubble');
  if (!bubble) {
    bubble = document.createElement('div');
    bubble.id = 'tipBubble';
    bubble.setAttribute('role', 'tooltip');
    document.body.appendChild(bubble);
  }

  const pad = 12;
  const gap = 10;

  const hide = () => {
    bubble!.classList.remove('is-visible');
    bubble!.setAttribute('aria-hidden', 'true');
  };

  const showFor = (el: HTMLElement) => {
    const text = el.getAttribute('data-tip');
    if (!text) return;
    bubble!.textContent = text;
    bubble!.classList.add('is-visible');
    bubble!.setAttribute('aria-hidden', 'false');

    const tipRect = el.getBoundingClientRect();
    const bRect = bubble!.getBoundingClientRect();
    const vw = window.innerWidth;
    const vh = window.innerHeight;

    let top = tipRect.top - bRect.height - gap;
    if (top < pad) top = tipRect.bottom + gap;
    if (top + bRect.height > vh - pad) top = Math.max(pad, vh - pad - bRect.height);

    let left = tipRect.left + tipRect.width / 2 - bRect.width / 2;
    left = Math.min(Math.max(pad, left), vw - pad - bRect.width);

    bubble!.style.top = `${Math.round(top)}px`;
    bubble!.style.left = `${Math.round(left)}px`;
  };

  document.querySelectorAll<HTMLElement>('.tip[data-tip]').forEach((el) => {
    el.addEventListener('mouseenter', () => showFor(el));
    el.addEventListener('mouseleave', hide);
    el.addEventListener('focus', () => showFor(el));
    el.addEventListener('blur', hide);
  });

  document.getElementById('viewSettings')?.addEventListener('scroll', hide, { passive: true });
  window.addEventListener('resize', hide);
}

function markPerfDirty(): void {
  userOverrodePerformance = true;
  updateRamPreview();
}

async function init(): Promise<void> {
  setupTooltips();
  const state = await api.getState();
  const extended = state as LauncherState & {
    sync?: SyncStatus;
    lastSyncDurationSeconds?: number | null;
  };
  if (typeof extended.lastSyncDurationSeconds === 'number') {
    lastSyncDurationSeconds = extended.lastSyncDurationSeconds;
  }
  applyState(state);
  if (setupComplete) {
    activeTab = 'control';
    renderMode();
  } else {
    wizardStep = 1;
    renderMode();
  }

  api.onLog(appendLog);
  api.onStatus((s) => setStatus(s));
  api.onProgress(showProgress);
  api.onSyncStatus(paintSync);
  resetSyncUi();
  const initialSync = await api.getSyncStatus();
  paintSync(initialSync);

  document.querySelectorAll<HTMLButtonElement>('.tab').forEach((btn) => {
    btn.addEventListener('click', () => {
      if (activeTab === 'settings') harvestSettings();
      activeTab = btn.dataset.tab as 'control' | 'settings';
      renderMode();
    });
  });

  document.querySelectorAll<HTMLButtonElement>('[data-browse]').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const id = btn.dataset.browse!;
      const picked = await api.pickDirectory({ title: 'Select folder', defaultPath: getInput(id) || undefined });
      if (!picked) return;
      setInput(id, picked);
      if (id === 'installDir' && !getInput('appdir')) setInput('appdir', `${picked}\\data`);
      if (id === 's_installDir' && !getInput('s_appdir')) setInput('s_appdir', `${picked}\\data`);
      if (id === 'appdir' || id === 'installDir') {
        performanceLoadedForPath = '';
        userOverrodePerformance = false;
      }
    });
  });

  document.querySelectorAll<HTMLButtonElement>('[data-browse-file]').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const id = btn.dataset.browseFile!;
      const isZip = id.toLowerCase().includes('zip') || id === 'datadirZip';
      const picked = await api.pickFile({
        title: 'Select file',
        defaultPath: getInput(id) || undefined,
        filters: isZip
          ? [{ name: 'ZIP', extensions: ['zip'] }]
          : [{ name: 'Executable', extensions: ['exe'] }, { name: 'All', extensions: ['*'] }],
      });
      if (picked) setInput(id, picked);
    });
  });

  document.querySelectorAll<HTMLButtonElement>('[data-binary-mode]').forEach((btn) => {
    btn.addEventListener('click', () => {
      binaryMode = btn.dataset.binaryMode as 'download' | 'local';
      paintBinaryMode();
      if (binaryMode === 'download') void loadReleaseMeta();
    });
  });

  document.querySelectorAll<HTMLButtonElement>('[data-sync-mode]').forEach((btn) => {
    btn.addEventListener('click', () => {
      syncMode = btn.dataset.syncMode as 'scratch' | 'zip';
      paintSyncMode();
    });
  });

  document.querySelectorAll<HTMLButtonElement>('[data-storage-profile]').forEach((btn) => {
    btn.addEventListener('click', async () => {
      storageProfile = btn.dataset.storageProfile as 'ssd' | 'hdd';
      paintStorageProfile();
      if (machineResources) {
        const rec = await api.recommendPerformance({
          totalRamMb: machineResources.totalRamMb,
          availableRamMb: machineResources.availableRamMb,
          profile: storageProfile,
          drive: machineResources.disk.drive,
        });
        userOverrodePerformance = false;
        applySuggestion(rec, true);
        userOverrodePerformance = true; // profile choice is intentional
      }
    });
  });

  $('btnBrowseBinary').addEventListener('click', async () => {
    const picked = await api.pickFile({
      title: 'Select keryxd.exe',
      defaultPath: getInput('binarySource') || undefined,
      filters: [{ name: 'Executable', extensions: ['exe'] }, { name: 'All', extensions: ['*'] }],
    });
    if (picked) setInput('binarySource', picked);
  });

  $('btnBrowseBinaryDir').addEventListener('click', async () => {
    const picked = await api.pickDirectory({
      title: 'Select folder containing keryxd.exe',
      defaultPath: getInput('binarySource') || undefined,
    });
    if (picked) setInput('binarySource', picked);
  });

  $('btnDownloadRelease').addEventListener('click', async () => {
    harvestWizard();
    $('downloadError').hidden = true;
    if (!config.installDir) {
      alert('Set the install directory on step 1 first.');
      return;
    }
    if (!latestRelease) {
      try {
        latestRelease = await api.fetchLatestRelease();
      } catch (e) {
        $('downloadError').hidden = false;
        $('downloadError').textContent = String(e);
        return;
      }
    }
    const btn = $('btnDownloadRelease') as HTMLButtonElement;
    btn.disabled = true;
    $('progressWrap').hidden = false;
    const r = await api.downloadAndInstallRelease({
      downloadUrl: latestRelease.downloadUrl,
      installDir: config.installDir,
    });
    btn.disabled = false;
    if (!r.ok) {
      $('downloadError').hidden = false;
      $('downloadError').textContent = r.error || 'Download failed';
      return;
    }
    if (r.binaryPath) {
      setInput('binarySource', r.binaryPath);
      config.binarySource = r.binaryPath;
    }
    $('releaseMeta').textContent = `Installed ${latestRelease.tag} → ${r.binaryPath}`;
  });

  $('btnResetSuggested').addEventListener('click', async () => {
    userOverrodePerformance = false;
    await loadMachineResources(true);
  });

  $('ramScale').addEventListener('input', markPerfDirty);
  $('rocksdbCacheSize').addEventListener('input', markPerfDirty);
  $('rocksdbRateLimitMb').addEventListener('input', markPerfDirty);

  $('btnWizardBack').addEventListener('click', () => {
    harvestWizard();
    void showWizardStep(wizardStep - 1);
  });

  $('btnWizardNext').addEventListener('click', async () => {
    const err = validateStep(wizardStep);
    if (err) {
      alert(err);
      return;
    }
    harvestWizard();
    await api.saveConfig(config);
    await showWizardStep(wizardStep + 1);
  });

  $('btnWizardSave').addEventListener('click', async () => {
    harvestWizard();
    await api.saveConfig(config);
    const msg = $('progressMsg');
    $('progressWrap').hidden = false;
    msg.textContent = 'Progress saved — you can close and continue later.';
  });

  $('btnWizardInstall').addEventListener('click', async () => {
    const err = validateStep(1) || validateStep(2) || validateStep(4);
    if (err) {
      alert(err);
      return;
    }
    harvestWizard();
    $('installError').hidden = true;
    const btn = $('btnWizardInstall') as HTMLButtonElement;
    btn.disabled = true;
    appendLog('[launcher] Installing…\n');
    const r = await api.installNode(config);
    btn.disabled = false;
    if (!r.ok) {
      $('installError').hidden = false;
      $('installError').textContent = r.error || 'Install failed';
      return;
    }
    if (r.state) applyState(r.state);
    activeTab = 'control';
    renderMode();
    appendLog('[launcher] Install complete. Ready to start.\n');
  });

  $('btnRunToggle').addEventListener('click', async () => {
    const toggle = $('btnRunToggle') as HTMLButtonElement;
    toggle.disabled = true;
    ($('btnRestart') as HTMLButtonElement).disabled = true;
    if (nodeRunning) await stopNodeAction();
    else await startNodeAction();
  });

  $('btnRestart').addEventListener('click', async () => {
    if (!nodeRunning) return;
    const toggle = $('btnRunToggle') as HTMLButtonElement;
    const restart = $('btnRestart') as HTMLButtonElement;
    toggle.disabled = true;
    restart.disabled = true;
    appendLog('[launcher] Restarting…\n');
    const stopped = await stopNodeAction();
    if (!stopped) {
      setRunButtons(true);
      return;
    }
    await startNodeAction();
  });

  $('btnSave').addEventListener('click', async () => {
    harvestSettings();
    await api.saveConfig(config);
    paintWizard();
    $('savedMsg').hidden = false;
    setTimeout(() => {
      $('savedMsg').hidden = true;
    }, 1500);
    await refreshPreview();
  });

  $('btnReinstall').addEventListener('click', async () => {
    harvestSettings();
    const r = await api.installNode(config);
    if (!r.ok) alert(r.error || 'Re-install failed');
    else if (r.state) applyState(r.state);
  });

  $('btnAutoTunePeers').addEventListener('click', async () => {
    const btn = $('btnAutoTunePeers') as HTMLButtonElement;
    btn.disabled = true;
    btn.textContent = 'Tuning…';
    peersTuned = false;
    await autoTunePeers(false);
    paintSettings();
    await refreshPreview();
    btn.disabled = false;
    btn.textContent = 'Auto-tune peers';
  });

  $('btnClearPeers').addEventListener('click', async () => {
    config.peers = [];
    peersTuned = false;
    peerTuneSummary = 'Peers: none (DNS seed only)';
    setInput('s_peers', '');
    await api.saveConfig(config);
    await refreshPreview();
  });

  $('btnOpenInstall').addEventListener('click', async () => {
    harvestSettings();
    if (config.installDir) await api.openPath(config.installDir);
  });

  $('btnRunSetupAgain').addEventListener('click', async () => {
    const ok = await openConfirmModal({
      title: 'Run setup again?',
      body: 'This opens the first-run wizard again. Your current launcher settings stay saved.',
      confirmLabel: 'Open wizard',
      danger: false,
    });
    if (!ok) return;
    harvestSettings();
    await api.saveConfig(config);
    await runSetupFromScratch(false);
  });

  $('btnControlResetSetup').addEventListener('click', async () => {
    const ok = await openConfirmModal({
      title: 'Setup from scratch?',
      body: 'This stops the node if it is running, clears your launcher settings, and opens the first-run wizard again. Blockchain data on disk is not deleted — only the launcher configuration.',
      confirmLabel: 'Reset & reconfigure',
      danger: true,
    });
    if (!ok) return;
    if (nodeRunning) {
      appendLog('[launcher] Stopping before reset…\n');
      await api.stopNode();
      setRunButtons(false);
    }
    await runSetupFromScratch(true);
  });

  document.querySelectorAll('#viewSettings input, #viewSettings select, #viewSettings textarea').forEach((el) => {
    el.addEventListener('change', () => void refreshPreview());
  });
}

function openConfirmModal(opts: {
  title: string;
  body: string;
  confirmLabel: string;
  danger?: boolean;
}): Promise<boolean> {
  const backdrop = $('confirmModal');
  $('confirmModalTitle').textContent = opts.title;
  $('confirmModalBody').textContent = opts.body;
  const okBtn = $('confirmModalOk') as HTMLButtonElement;
  okBtn.textContent = opts.confirmLabel;
  okBtn.classList.toggle('danger', opts.danger !== false);
  okBtn.classList.toggle('primary', opts.danger === false);
  backdrop.hidden = false;

  return new Promise((resolve) => {
    const cleanup = () => {
      backdrop.hidden = true;
      okBtn.removeEventListener('click', onOk);
      cancelBtn.removeEventListener('click', onCancel);
      backdrop.removeEventListener('click', onBackdrop);
      window.removeEventListener('keydown', onKey);
    };
    const onOk = () => {
      cleanup();
      resolve(true);
    };
    const onCancel = () => {
      cleanup();
      resolve(false);
    };
    const onBackdrop = (e: MouseEvent) => {
      if (e.target === backdrop) onCancel();
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onCancel();
    };
    const cancelBtn = $('confirmModalCancel');
    okBtn.addEventListener('click', onOk);
    cancelBtn.addEventListener('click', onCancel);
    backdrop.addEventListener('click', onBackdrop);
    window.addEventListener('keydown', onKey);
    okBtn.focus();
  });
}

async function runSetupFromScratch(clearConfig: boolean): Promise<void> {
  const next = await api.resetSetup(clearConfig);
  userOverrodePerformance = false;
  performanceLoadedForPath = '';
  peersTuned = false;
  peerTuneSummary = 'Peers: none yet (will auto-tune on Install step)';
  peerTuneBestRtt = null;
  binaryMode = 'download';
  syncMode = 'scratch';
  storageProfile = 'hdd';
  // Force wizard even if applyState would prefer Control from a stale flag.
  setupComplete = false;
  applyState({ ...next, setupComplete: false });
  wizardStep = 1;
  renderMode();
}

void init();
