//! Fetch and install keryxd from GitHub Releases.

use crate::config::{binary_path_in_install, ProgressEvent};
use crate::node_manager::expand_archive;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tauri::{AppHandle, Emitter};
use tokio::io::AsyncWriteExt;

const RELEASES_API: &str = "https://api.github.com/repos/Keryx-Labs/keryx-node/releases/latest";
const USER_AGENT: &str = "keryx-node-launcher/1.0";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LatestRelease {
    pub tag: String,
    pub name: String,
    pub size: u64,
    pub download_url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DownloadResult {
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub binary_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tag: Option<String>,
}

#[derive(Deserialize)]
struct GhRelease {
    tag_name: String,
    #[allow(dead_code)]
    name: Option<String>,
    assets: Vec<GhAsset>,
}

#[derive(Deserialize)]
struct GhAsset {
    name: String,
    size: u64,
    browser_download_url: String,
}

fn emit_progress(app: &AppHandle, phase: &str, message: &str, percent: Option<f64>) {
    let _ = app.emit(
        "progress",
        ProgressEvent {
            phase: phase.into(),
            message: message.into(),
            percent,
        },
    );
}

pub async fn fetch_latest_release() -> Result<LatestRelease, String> {
    let client = reqwest::Client::builder()
        .user_agent(USER_AGENT)
        .build()
        .map_err(|e| e.to_string())?;
    let resp = client
        .get(RELEASES_API)
        .send()
        .await
        .map_err(|e| format!("GitHub API request failed: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("GitHub API returned {}", resp.status()));
    }
    let release: GhRelease = resp
        .json()
        .await
        .map_err(|e| format!("Failed to parse GitHub release JSON: {e}"))?;

    let asset = release
        .assets
        .into_iter()
        .find(|a| {
            let n = a.name.to_ascii_lowercase();
            n.contains("win64") || n.contains("windows") || n.contains("win-amd64")
        })
        .ok_or_else(|| {
            "No Windows (win64) asset found in the latest GitHub release".to_string()
        })?;

    Ok(LatestRelease {
        tag: release.tag_name,
        name: asset.name,
        size: asset.size,
        download_url: asset.browser_download_url,
    })
}

pub async fn download_and_install_release(
    app: &AppHandle,
    download_url: &str,
    install_dir: &str,
) -> Result<DownloadResult, String> {
    let install_dir = install_dir.trim();
    if install_dir.is_empty() {
        return Err("Install directory is required before downloading".into());
    }

    let bin_dir = Path::new(install_dir).join("bin");
    let staging = Path::new(install_dir).join(".download-staging");
    std::fs::create_dir_all(&bin_dir).map_err(|e| e.to_string())?;
    if staging.exists() {
        let _ = std::fs::remove_dir_all(&staging);
    }
    std::fs::create_dir_all(&staging).map_err(|e| e.to_string())?;

    let zip_path = staging.join("release.zip");
    emit_progress(app, "download", "Downloading latest release…", Some(2.0));

    let client = reqwest::Client::builder()
        .user_agent(USER_AGENT)
        .build()
        .map_err(|e| e.to_string())?;

    let resp = client
        .get(download_url)
        .send()
        .await
        .map_err(|e| format!("Download failed: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("Download returned HTTP {}", resp.status()));
    }

    let total = resp.content_length().unwrap_or(0);
    let mut file = tokio::fs::File::create(&zip_path)
        .await
        .map_err(|e| e.to_string())?;
    let mut stream = resp.bytes_stream();
    let mut downloaded: u64 = 0;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("Download stream error: {e}"))?;
        file.write_all(&chunk).await.map_err(|e| e.to_string())?;
        downloaded += chunk.len() as u64;
        if total > 0 {
            let pct = 5.0 + (downloaded as f64 / total as f64) * 55.0;
            emit_progress(
                app,
                "download",
                &format!(
                    "Downloading… {:.1} / {:.1} MB",
                    downloaded as f64 / (1024.0 * 1024.0),
                    total as f64 / (1024.0 * 1024.0)
                ),
                Some(pct),
            );
        }
    }
    file.flush().await.map_err(|e| e.to_string())?;
    drop(file);

    emit_progress(app, "download", "Extracting release zip…", Some(65.0));
    let extract_dir = staging.join("extracted");
    std::fs::create_dir_all(&extract_dir).map_err(|e| e.to_string())?;
    expand_archive(zip_path.to_str().unwrap_or_default(), &extract_dir).await?;

    emit_progress(app, "download", "Locating keryxd.exe…", Some(80.0));
    let exe = find_keryxd_exe(&extract_dir)
        .ok_or_else(|| "keryxd.exe not found inside the release zip".to_string())?;

    let dest = binary_path_in_install(install_dir);
    std::fs::copy(&exe, &dest).map_err(|e| e.to_string())?;

    #[cfg(windows)]
    {
        if let Some(src_dir) = exe.parent() {
            if let Ok(entries) = std::fs::read_dir(src_dir) {
                for entry in entries.flatten() {
                    let name = entry.file_name();
                    let name_str = name.to_string_lossy();
                    if name_str.to_ascii_lowercase().ends_with(".dll") {
                        let _ = std::fs::copy(entry.path(), bin_dir.join(&*name));
                    }
                }
            }
        }
    }

    let _ = std::fs::remove_dir_all(&staging);
    emit_progress(app, "download", "Release installed into bin/", Some(100.0));

    Ok(DownloadResult {
        ok: true,
        error: None,
        binary_path: Some(dest.display().to_string()),
        tag: None,
    })
}

fn find_keryxd_exe(root: &Path) -> Option<PathBuf> {
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path
                .file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.eq_ignore_ascii_case("keryxd.exe") || n == "keryxd")
                .unwrap_or(false)
            {
                return Some(path);
            }
        }
    }
    None
}
