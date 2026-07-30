mod commands;
mod config;
mod node_manager;
mod peer_probe;
mod release;
mod resources;
mod store;
mod sync_monitor;

use commands::AppState;
use node_manager::NodeManager;
use store::Store;
use sync_monitor::SyncMonitor;
use std::sync::Arc;
use tauri::Manager;
use tokio::sync::Mutex;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let nodes = Arc::new(NodeManager::new());
    let sync = Arc::new(SyncMonitor::new());

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .setup({
            let nodes = nodes.clone();
            let sync = sync.clone();
            move |app| {
                let store = Store::load();
                let sync_for_cb = sync.clone();
                let app_handle = app.handle().clone();

                // Persist completed sync duration
                let store_path_note = app.handle().clone();
                tauri::async_runtime::block_on(async {
                    let nodes2 = nodes.clone();
                    let sync2 = sync.clone();
                    nodes2
                        .set_log_listener(Some(move |line: String| {
                            let sync = sync2.clone();
                            tauri::async_runtime::spawn(async move {
                                sync.ingest_log(&line).await;
                            });
                        }))
                        .await;

                    sync_for_cb
                        .set_on_sync_complete(Some(move |duration: f64| {
                            let handle = store_path_note.clone();
                            tauri::async_runtime::spawn(async move {
                                if let Some(state) = handle.try_state::<AppState>() {
                                    let mut store = state.store.lock().await;
                                    store.get_mut().last_sync_duration_seconds = Some(duration);
                                    store.get_mut().last_sync_finished_at =
                                        Some(config::now_ms());
                                    let _ = store.save();
                                }
                            });
                        }))
                        .await;
                });

                app.manage(AppState {
                    store: Mutex::new(store),
                    nodes,
                    sync,
                });

                let _ = app_handle;
                Ok(())
            }
        })
        .invoke_handler(tauri::generate_handler![
            commands::get_state,
            commands::get_sync_status,
            commands::save_config,
            commands::pick_directory,
            commands::pick_file,
            commands::install_node,
            commands::start_node,
            commands::stop_node,
            commands::reset_setup,
            commands::build_command_preview,
            commands::open_path,
            commands::open_external,
            commands::fetch_latest_release,
            commands::download_and_install_release,
            commands::detect_machine_resources,
            commands::recommend_performance,
            commands::probe_peers,
        ])
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { .. } = event {
                let app = window.app_handle().clone();
                tauri::async_runtime::spawn(async move {
                    if let Some(state) = app.try_state::<AppState>() {
                        state.sync.stop(&app).await;
                        let _ = state.nodes.stop(&app).await;
                    }
                });
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running Keryx Node Launcher");
}
