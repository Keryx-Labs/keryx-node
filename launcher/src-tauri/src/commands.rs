use crate::config::{
    binary_path_in_install, build_argv, is_install_ready, suggested_paths, LauncherConfig,
    LauncherState, SyncStatus,
};
use crate::node_manager::NodeManager;
use crate::store::Store;
use crate::sync_monitor::SyncMonitor;
use serde::Serialize;
use std::sync::Arc;
use tauri::{AppHandle, State};
use tauri_plugin_dialog::{DialogExt, FilePath};
use tauri_plugin_opener::OpenerExt;
use tokio::sync::Mutex;

pub struct AppState {
    pub store: Mutex<Store>,
    pub nodes: Arc<NodeManager>,
    pub sync: Arc<SyncMonitor>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InstallResult {
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub state: Option<LauncherState>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StateResponse {
    #[serde(flatten)]
    pub state: LauncherState,
    pub sync: SyncStatus,
    pub last_sync_duration_seconds: Option<f64>,
    pub last_sync_finished_at: Option<u64>,
}

fn merge_suggestions(mut config: LauncherConfig) -> LauncherConfig {
    let (install, binary) = suggested_paths();
    if config.install_dir.is_empty() {
        config.install_dir = install.clone();
    }
    if config.appdir.is_empty() && !config.install_dir.is_empty() {
        config.appdir = format!(
            "{}\\data",
            config.install_dir.trim_end_matches(['/', '\\'])
        );
    }
    if config.binary_source.is_empty() {
        config.binary_source = binary;
    }
    config
}

async fn load_config(store: &Store) -> LauncherConfig {
    merge_suggestions(store.get().config.clone())
}

async fn build_state(state: &AppState) -> LauncherState {
    let mut store = state.store.lock().await;
    let mut config = load_config(&store).await;
    // Reflect live install dir into node manager
    state.nodes.set_install_dir(config.install_dir.clone()).await;

    let ready = is_install_ready(&config.install_dir);
    let mut setup_complete = store.get().setup_complete;
    // If the binary disappeared, force the wizard back. Do NOT auto-complete when the
    // binary exists — that blocked "Setup from scratch" / reset_setup from re-entering
    // the onboarding while an old install was still on disk.
    if setup_complete && !ready {
        setup_complete = false;
        store.get_mut().setup_complete = false;
        let _ = store.save();
    }

    // Keep store config suggestions filled for next load without wiping user paths
    if store.get().config.install_dir.is_empty() {
        store.get_mut().config.install_dir = config.install_dir.clone();
    }
    if store.get().config.appdir.is_empty() {
        store.get_mut().config.appdir = config.appdir.clone();
    }
    if store.get().config.binary_source.is_empty() {
        store.get_mut().config.binary_source = config.binary_source.clone();
    }
    let _ = store.save();

    config = store.get().config.clone();
    let config = merge_suggestions(config);

    LauncherState {
        config,
        setup_complete,
        status: state.nodes.status().await,
    }
}

#[tauri::command]
pub async fn get_state(state: State<'_, AppState>) -> Result<StateResponse, String> {
    let launcher = build_state(&state).await;
    let sync = state.sync.get_last().await;
    let store = state.store.lock().await;
    Ok(StateResponse {
        state: launcher,
        sync,
        last_sync_duration_seconds: store.get().last_sync_duration_seconds,
        last_sync_finished_at: store.get().last_sync_finished_at,
    })
}

#[tauri::command]
pub async fn get_sync_status(state: State<'_, AppState>) -> Result<serde_json::Value, String> {
    let sync = state.sync.get_last().await;
    let store = state.store.lock().await;
    let mut value = serde_json::to_value(sync).map_err(|e| e.to_string())?;
    if let Some(obj) = value.as_object_mut() {
        obj.insert(
            "lastSyncDurationSeconds".into(),
            serde_json::json!(store.get().last_sync_duration_seconds),
        );
    }
    Ok(value)
}

#[tauri::command]
pub async fn save_config(
    state: State<'_, AppState>,
    config: LauncherConfig,
) -> Result<LauncherState, String> {
    {
        let mut store = state.store.lock().await;
        store.get_mut().config = config;
        store.save()?;
    }
    Ok(build_state(&state).await)
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PickOpts {
    pub title: Option<String>,
    pub default_path: Option<String>,
    pub filters: Option<Vec<FileFilter>>,
}

#[derive(serde::Deserialize)]
pub struct FileFilter {
    pub name: String,
    pub extensions: Vec<String>,
}

#[tauri::command]
pub async fn pick_directory(
    app: AppHandle,
    opts: Option<PickOpts>,
) -> Result<Option<String>, String> {
    let title = opts
        .as_ref()
        .and_then(|o| o.title.clone())
        .unwrap_or_else(|| "Select folder".into());
    let default_path = opts.and_then(|o| o.default_path);
    tokio::task::spawn_blocking(move || {
        let mut builder = app.dialog().file().set_title(title);
        if let Some(default) = default_path {
            builder = builder.set_directory(default);
        }
        let picked = builder.blocking_pick_folder();
        Ok(picked.map(|p| match p {
            FilePath::Path(path) => path.display().to_string(),
            FilePath::Url(url) => url.to_string(),
        }))
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
pub async fn pick_file(app: AppHandle, opts: Option<PickOpts>) -> Result<Option<String>, String> {
    let title = opts
        .as_ref()
        .and_then(|o| o.title.clone())
        .unwrap_or_else(|| "Select file".into());
    let default_path = opts.as_ref().and_then(|o| o.default_path.clone());
    let filters = opts.and_then(|o| o.filters);
    tokio::task::spawn_blocking(move || {
        let mut builder = app.dialog().file().set_title(title);
        if let Some(ref default) = default_path {
            if let Some(parent) = std::path::Path::new(default).parent() {
                builder = builder.set_directory(parent);
            }
        }
        if let Some(ref filters) = filters {
            for f in filters {
                let exts: Vec<&str> = f.extensions.iter().map(|s| s.as_str()).collect();
                builder = builder.add_filter(&f.name, &exts);
            }
        }
        let picked = builder.blocking_pick_file();
        Ok(picked.map(|p| match p {
            FilePath::Path(path) => path.display().to_string(),
            FilePath::Url(url) => url.to_string(),
        }))
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
pub async fn install_node(
    app: AppHandle,
    state: State<'_, AppState>,
    config: LauncherConfig,
) -> Result<InstallResult, String> {
    {
        let mut store = state.store.lock().await;
        store.get_mut().config = config.clone();
        store.save()?;
    }
    // Release gRPC monitor before swapping the binary (it may keep the process busy).
    state.sync.stop(&app).await;
    match state.nodes.install(&app, &config).await {
        Ok(()) => {
            {
                let mut store = state.store.lock().await;
                store.get_mut().setup_complete = true;
                store.save()?;
            }
            Ok(InstallResult {
                ok: true,
                error: None,
                state: Some(build_state(&state).await),
            })
        }
        Err(error) => Ok(InstallResult {
            ok: false,
            error: Some(error),
            state: None,
        }),
    }
}

#[tauri::command]
pub async fn start_node(
    app: AppHandle,
    state: State<'_, AppState>,
    config: LauncherConfig,
) -> Result<InstallResult, String> {
    {
        let mut store = state.store.lock().await;
        store.get_mut().config = config.clone();
        store.save()?;
    }
    match state.nodes.start(&app, &config).await {
        Ok(()) => {
            let sync = state.sync.clone();
            let rpclisten = config.rpclisten.clone();
            let app2 = app.clone();
            tokio::spawn(async move {
                tokio::time::sleep(std::time::Duration::from_millis(1500)).await;
                sync.start(app2, Some(&rpclisten)).await;
            });
            Ok(InstallResult {
                ok: true,
                error: None,
                state: Some(build_state(&state).await),
            })
        }
        Err(error) => Ok(InstallResult {
            ok: false,
            error: Some(error),
            state: None,
        }),
    }
}

#[tauri::command]
pub async fn stop_node(app: AppHandle, state: State<'_, AppState>) -> Result<InstallResult, String> {
    state.sync.stop(&app).await;
    match state.nodes.stop(&app).await {
        Ok(()) => Ok(InstallResult {
            ok: true,
            error: None,
            state: Some(build_state(&state).await),
        }),
        Err(error) => Ok(InstallResult {
            ok: false,
            error: Some(error),
            state: None,
        }),
    }
}

#[tauri::command]
pub async fn reset_setup(
    state: State<'_, AppState>,
    clear_config: Option<bool>,
) -> Result<LauncherState, String> {
    {
        let mut store = state.store.lock().await;
        store.get_mut().setup_complete = false;
        if clear_config.unwrap_or(false) {
            store.get_mut().config = LauncherConfig::default();
        }
        store.save()?;
    }
    Ok(build_state(&state).await)
}

#[tauri::command]
pub async fn build_command_preview(config: LauncherConfig) -> Result<String, String> {
    let bin = if is_install_ready(&config.install_dir) {
        binary_path_in_install(&config.install_dir)
            .display()
            .to_string()
    } else if !config.binary_source.is_empty() {
        config.binary_source.clone()
    } else {
        "keryxd".into()
    };
    Ok(format!("{bin} {}", build_argv(&config).join(" ")))
}

#[tauri::command]
pub async fn open_path(app: AppHandle, path: String) -> Result<(), String> {
    if path.is_empty() {
        return Ok(());
    }
    app.opener()
        .open_path(path, None::<&str>)
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn open_external(app: AppHandle, url: String) -> Result<(), String> {
    if url.is_empty() {
        return Ok(());
    }
    app.opener()
        .open_url(url, None::<&str>)
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn fetch_latest_release() -> Result<crate::release::LatestRelease, String> {
    crate::release::fetch_latest_release().await
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DownloadReleaseArgs {
    pub download_url: String,
    pub install_dir: String,
}

#[tauri::command]
pub async fn download_and_install_release(
    app: AppHandle,
    args: DownloadReleaseArgs,
) -> Result<crate::release::DownloadResult, String> {
    match crate::release::download_and_install_release(&app, &args.download_url, &args.install_dir)
        .await
    {
        Ok(r) => Ok(r),
        Err(error) => Ok(crate::release::DownloadResult {
            ok: false,
            error: Some(error),
            binary_path: None,
            tag: None,
        }),
    }
}

#[tauri::command]
pub async fn detect_machine_resources(path: String) -> Result<crate::resources::MachineResources, String> {
    Ok(tokio::task::spawn_blocking(move || {
        crate::resources::detect_machine_resources(&path)
    })
    .await
    .map_err(|e| e.to_string())?)
}

#[tauri::command]
pub async fn recommend_performance(
    input: crate::resources::RecommendInput,
) -> Result<crate::resources::PerformanceRecommendation, String> {
    Ok(crate::resources::recommend_performance(input))
}

#[tauri::command]
pub async fn probe_peers(
    args: Option<crate::peer_probe::ProbePeersArgs>,
) -> Result<crate::peer_probe::PeerProbeReport, String> {
    Ok(crate::peer_probe::probe_peers(args.unwrap_or(crate::peer_probe::ProbePeersArgs {
        addresses: None,
    }))
    .await)
}
