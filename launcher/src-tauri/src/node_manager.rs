use crate::config::{
    binary_path_in_install, build_argv, is_install_ready, network_datadir_rel, resolve_appdir,
    resolve_binary_source, LauncherConfig, NodeStatus, ProgressEvent,
};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use tauri::{AppHandle, Emitter, Manager};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;

pub struct NodeManager {
    child: Arc<Mutex<Option<Child>>>,
    pid: Arc<Mutex<Option<u32>>>,
    log_buf: Arc<Mutex<String>>,
    install_dir: Arc<Mutex<String>>,
    on_log_line: Arc<Mutex<Option<Box<dyn Fn(String) + Send + Sync>>>>,
}

impl NodeManager {
    pub fn new() -> Self {
        Self {
            child: Arc::new(Mutex::new(None)),
            pid: Arc::new(Mutex::new(None)),
            log_buf: Arc::new(Mutex::new(String::new())),
            install_dir: Arc::new(Mutex::new(String::new())),
            on_log_line: Arc::new(Mutex::new(None)),
        }
    }

    pub async fn set_install_dir(&self, dir: String) {
        *self.install_dir.lock().await = dir;
    }

    pub async fn set_log_listener<F>(&self, f: Option<F>)
    where
        F: Fn(String) + Send + Sync + 'static,
    {
        *self.on_log_line.lock().await = f.map(|cb| Box::new(cb) as Box<dyn Fn(String) + Send + Sync>);
    }

    pub async fn status(&self) -> NodeStatus {
        let install_dir = self.install_dir.lock().await.clone();
        let child_guard = self.child.lock().await;
        let running = child_guard.is_some();
        let pid = *self.pid.lock().await;
        let ready = is_install_ready(&install_dir);
        NodeStatus {
            running,
            pid,
            install_ready: ready,
            binary_path: if ready {
                Some(binary_path_in_install(&install_dir).display().to_string())
            } else {
                None
            },
        }
    }

    async fn emit_log(&self, app: &AppHandle, chunk: &str) {
        let _ = app.emit("node-log", chunk);
        let mut buf = self.log_buf.lock().await;
        buf.push_str(chunk);
        while let Some(nl) = buf.find('\n') {
            let mut line = buf[..nl].to_string();
            *buf = buf[nl + 1..].to_string();
            if line.ends_with('\r') {
                line.pop();
            }
            if let Some(cb) = self.on_log_line.lock().await.as_ref() {
                cb(line);
            }
        }
    }

    async fn emit_status(&self, app: &AppHandle) {
        let status = self.status().await;
        let _ = app.emit("node-status", status);
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

    pub async fn install(&self, app: &AppHandle, config: &LauncherConfig) -> Result<(), String> {
        let install_dir = config.install_dir.trim();
        if install_dir.is_empty() {
            return Err("Install directory is required".into());
        }

        // Destination exe cannot be overwritten while a node process still holds it.
        Self::emit_progress(app, "install", "Stopping any running node before install…", Some(2.0));
        let _ = self.stop(app).await;
        #[cfg(windows)]
        {
            kill_keryxd_image().await;
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        }

        let bin_dir = Path::new(install_dir).join("bin");
        let data_dir = {
            let resolved = resolve_appdir(config);
            if resolved.is_empty() {
                Path::new(install_dir).join("data")
            } else {
                PathBuf::from(resolved)
            }
        };
        std::fs::create_dir_all(&bin_dir).map_err(|e| e.to_string())?;
        std::fs::create_dir_all(&data_dir).map_err(|e| e.to_string())?;

        let mut source = config.binary_source.trim().to_string();
        if source.is_empty() {
            let candidates = candidate_binaries(app);
            source = candidates
                .into_iter()
                .find(|c| Path::new(c).is_file())
                .unwrap_or_default();
        }
        // Allow pointing at a folder that contains keryxd.exe
        let resolved = resolve_binary_source(&source)
            .or_else(|| {
                if Path::new(&source).is_file() {
                    Some(PathBuf::from(&source))
                } else {
                    None
                }
            });
        let source_path = resolved.ok_or_else(|| {
            "keryxd binary not found. Download the latest release, pick a local folder/exe, or build target/release/keryxd."
                .to_string()
        })?;
        let source = source_path.display().to_string();
        if !Path::new(&source).is_file() {
            return Err(format!("keryxd binary not found at {source}"));
        }

        let dest = binary_path_in_install(install_dir);
        Self::emit_progress(
            app,
            "install",
            &format!("Copying binary → {}", dest.display()),
            Some(10.0),
        );
        if !paths_equal(Path::new(&source), &dest) {
            copy_file_replace(Path::new(&source), &dest).await?;
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&dest).map_err(|e| e.to_string())?.permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(&dest, perms).map_err(|e| e.to_string())?;
        }

        #[cfg(windows)]
        {
            if let Some(src_dir) = Path::new(&source).parent() {
                if let Ok(entries) = std::fs::read_dir(src_dir) {
                    for entry in entries.flatten() {
                        let name = entry.file_name();
                        let name_str = name.to_string_lossy();
                        if name_str.to_ascii_lowercase().ends_with(".dll") {
                            let dest_dll = bin_dir.join(&*name);
                            if !paths_equal(&entry.path(), &dest_dll) {
                                let _ = copy_file_replace(&entry.path(), &dest_dll).await;
                            }
                        }
                    }
                }
            }
        }

        let zip = config.datadir_zip.trim();
        if config.extract_zip_on_install && !zip.is_empty() {
            if !Path::new(zip).is_file() {
                return Err(format!("datadir.zip not found: {zip}"));
            }
            self.extract_datadir_zip(app, zip, &data_dir, config).await?;
        }

        Self::emit_progress(app, "install", "Install complete", Some(100.0));
        self.set_install_dir(install_dir.to_string()).await;
        self.emit_status(app).await;
        Ok(())
    }

    async fn extract_datadir_zip(
        &self,
        app: &AppHandle,
        zip_path: &str,
        appdir: &Path,
        config: &LauncherConfig,
    ) -> Result<(), String> {
        Self::emit_progress(
            app,
            "extract",
            "Extracting datadir.zip (this can take a while on HDD)…",
            Some(20.0),
        );

        let staging = appdir.join(".extract-staging");
        if staging.exists() {
            let _ = std::fs::remove_dir_all(&staging);
        }
        std::fs::create_dir_all(&staging).map_err(|e| e.to_string())?;

        expand_archive(zip_path, &staging).await?;

        let mut candidates = vec![
            staging.join("datadir"),
            staging.join("keryx-mainnet").join("datadir"),
            staging.join("keryx-testnet").join("datadir"),
            staging
                .join(format!("keryx-testnet-{}", config.testnet_suffix))
                .join("datadir"),
            staging.join("keryx-devnet").join("datadir"),
            staging.join("keryx-simnet").join("datadir"),
        ];
        let mut source_datadir = candidates.drain(..).find(|c| c.exists());
        if source_datadir.is_none() && staging.join("consensus").exists() {
            source_datadir = Some(staging.clone());
        }
        let source_datadir = source_datadir
            .ok_or_else(|| "Zip does not look like a keryxd datadir (expected consensus/ or datadir/)".to_string())?;

        let network_folder = appdir.join(network_datadir_rel(config));
        if let Some(parent) = network_folder.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        if network_folder.exists() {
            Self::emit_progress(app, "extract", "Removing existing datadir before restore…", Some(60.0));
            std::fs::remove_dir_all(&network_folder).map_err(|e| e.to_string())?;
        }
        Self::emit_progress(
            app,
            "extract",
            &format!("Moving extracted chain → {}", network_folder.display()),
            Some(80.0),
        );
        std::fs::rename(&source_datadir, &network_folder).map_err(|e| e.to_string())?;
        let _ = std::fs::remove_dir_all(&staging);
        Self::emit_progress(app, "extract", "Datadir ready", Some(95.0));
        Ok(())
    }

    pub async fn start(&self, app: &AppHandle, config: &LauncherConfig) -> Result<(), String> {
        {
            let guard = self.child.lock().await;
            if guard.is_some() {
                return Err("Node is already running".into());
            }
        }

        let install_dir = config.install_dir.trim();
        if !is_install_ready(install_dir) {
            return Err("Install the node first (binary missing under installDir/bin)".into());
        }

        let bin = binary_path_in_install(install_dir);
        let argv = build_argv(config);
        let preview = format!("$ {} {}\n", bin.display(), argv.join(" "));
        self.emit_log(app, &preview).await;

        *self.log_buf.lock().await = String::new();
        self.set_install_dir(install_dir.to_string()).await;

        let mut cmd = Command::new(&bin);
        cmd.args(&argv)
            .current_dir(install_dir)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(false);
        #[cfg(windows)]
        {
            cmd.creation_flags(0x0800_0000);
        }

        let mut child = cmd.spawn().map_err(|e| format!("Failed to spawn keryxd: {e}"))?;
        let pid = child.id();
        *self.pid.lock().await = pid;

        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        *self.child.lock().await = Some(child);

        let app_out = app.clone();
        let mgr_out = self.clone_handles();
        if let Some(out) = stdout {
            tokio::spawn(async move {
                let mut lines = BufReader::new(out).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    mgr_out.emit_log(&app_out, &format!("{line}\n")).await;
                }
            });
        }

        let app_err = app.clone();
        let mgr_err = self.clone_handles();
        if let Some(err) = stderr {
            tokio::spawn(async move {
                let mut lines = BufReader::new(err).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    mgr_err.emit_log(&app_err, &format!("{line}\n")).await;
                }
            });
        }

        // Poll try_wait without holding the Child across a long await — otherwise stop()
        // can never acquire the process handle (and used to block forever on the mutex).
        let app_wait = app.clone();
        let child_slot = self.child.clone();
        let pid_slot = self.pid.clone();
        let log_buf = self.log_buf.clone();
        let on_log = self.on_log_line.clone();
        let install_dir_slot = self.install_dir.clone();
        tokio::spawn(async move {
            let code = loop {
                tokio::time::sleep(std::time::Duration::from_millis(200)).await;
                let mut guard = child_slot.lock().await;
                let Some(child) = guard.as_mut() else {
                    // stop() already took ownership to kill/reap.
                    break None;
                };
                match child.try_wait() {
                    Ok(Some(status)) => {
                        let code = status.code();
                        *guard = None;
                        break code;
                    }
                    Ok(None) => continue,
                    Err(_) => {
                        *guard = None;
                        break None;
                    }
                }
            };
            *pid_slot.lock().await = None;
            {
                let mut buf = log_buf.lock().await;
                if !buf.is_empty() {
                    let mut line = buf.clone();
                    if line.ends_with('\r') {
                        line.pop();
                    }
                    buf.clear();
                    if let Some(cb) = on_log.lock().await.as_ref() {
                        cb(line);
                    }
                }
            }
            let msg = format!("\n[keryxd exited] code={code:?}\n");
            let _ = app_wait.emit("node-log", msg);
            let install_dir = install_dir_slot.lock().await.clone();
            let ready = is_install_ready(&install_dir);
            let status = NodeStatus {
                running: false,
                pid: None,
                install_ready: ready,
                binary_path: if ready {
                    Some(binary_path_in_install(&install_dir).display().to_string())
                } else {
                    None
                },
            };
            let _ = app_wait.emit("node-status", status);
        });

        self.emit_status(app).await;
        Ok(())
    }

    pub async fn stop(&self, app: &AppHandle) -> Result<(), String> {
        let pid = {
            let guard_pid = *self.pid.lock().await;
            let guard = self.child.lock().await;
            let child_pid = guard.as_ref().and_then(|c| c.id());
            child_pid.or(guard_pid)
        };

        #[cfg(windows)]
        {
            if let Some(pid) = pid {
                let _ = Command::new("taskkill")
                    .args(["/PID", &pid.to_string(), "/T", "/F"])
                    .creation_flags(0x0800_0000)
                    .status()
                    .await;
            }
            // Fallback if our Child handle was lost / PID stale / orphaned tree.
            kill_keryxd_image().await;
        }

        let mut guard = self.child.lock().await;
        if let Some(mut child) = guard.take() {
            drop(guard);
            #[cfg(not(windows))]
            {
                let _ = child.kill().await;
            }
            #[cfg(windows)]
            {
                let _ = child.start_kill();
            }
            let wait = tokio::time::timeout(std::time::Duration::from_secs(8), child.wait()).await;
            if wait.is_err() {
                let _ = child.start_kill();
                let _ = tokio::time::timeout(std::time::Duration::from_secs(3), child.wait()).await;
            }
        } else {
            drop(guard);
            #[cfg(not(windows))]
            if let Some(pid) = pid {
                let _ = Command::new("kill").args(["-TERM", &pid.to_string()]).status().await;
            }
        }

        *self.pid.lock().await = None;
        self.emit_status(app).await;
        Ok(())
    }

    fn clone_handles(&self) -> Self {
        Self {
            child: self.child.clone(),
            pid: self.pid.clone(),
            log_buf: self.log_buf.clone(),
            install_dir: self.install_dir.clone(),
            on_log_line: self.on_log_line.clone(),
        }
    }
}

fn paths_equal(a: &Path, b: &Path) -> bool {
    if a == b {
        return true;
    }
    let norm = |p: &Path| -> String {
        let s = p.to_string_lossy().replace('/', "\\");
        #[cfg(windows)]
        {
            return s.to_ascii_lowercase();
        }
        #[cfg(not(windows))]
        {
            s.to_string()
        }
    };
    if norm(a) == norm(b) {
        return true;
    }
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(ca), Ok(cb)) => {
            #[cfg(windows)]
            {
                ca.to_string_lossy()
                    .eq_ignore_ascii_case(&cb.to_string_lossy())
            }
            #[cfg(not(windows))]
            {
                ca == cb
            }
        }
        _ => false,
    }
}

fn is_sharing_violation(err: &std::io::Error) -> bool {
    match err.raw_os_error() {
        Some(32) | Some(33) | Some(5) => true, // sharing / lock / access denied
        _ => false,
    }
}

async fn copy_file_replace(source: &Path, dest: &Path) -> Result<(), String> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }

    let mut last_err = None;
    for attempt in 0..6 {
        match std::fs::copy(source, dest) {
            Ok(_) => return Ok(()),
            Err(e) if is_sharing_violation(&e) => {
                last_err = Some(e);
                // Move the locked file aside so we can write a fresh copy.
                if dest.exists() {
                    let bak = dest.with_extension(format!("old-{}", attempt));
                    let _ = std::fs::remove_file(&bak);
                    if std::fs::rename(dest, &bak).is_ok() {
                        // Best-effort cleanup of the sideline copy later.
                        let _ = std::fs::remove_file(&bak);
                    }
                }
                #[cfg(windows)]
                {
                    if attempt == 1 {
                        kill_keryxd_image().await;
                    }
                }
                tokio::time::sleep(std::time::Duration::from_millis(350 + attempt as u64 * 200))
                    .await;
            }
            Err(e) => return Err(e.to_string()),
        }
    }

    Err(format!(
        "Could not replace {}: {}. Stop the node (and any other keryxd.exe) and try again.",
        dest.display(),
        last_err
            .map(|e| e.to_string())
            .unwrap_or_else(|| "file in use".into())
    ))
}

#[cfg(windows)]
async fn kill_keryxd_image() {
    let _ = Command::new("taskkill")
        .args(["/IM", "keryxd.exe", "/F", "/T"])
        .creation_flags(0x0800_0000)
        .status()
        .await;
}

fn candidate_binaries(app: &AppHandle) -> Vec<String> {
    let mut out = Vec::new();
    // Dev: repo target/release relative to launcher/
    let launcher_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let repo_root = launcher_root.join("..");
    #[cfg(windows)]
    {
        out.push(
            repo_root
                .join("target")
                .join("release")
                .join("keryxd.exe")
                .display()
                .to_string(),
        );
    }
    out.push(
        repo_root
            .join("target")
            .join("release")
            .join("keryxd")
            .display()
            .to_string(),
    );

    if let Ok(resource) = app.path().resource_dir() {
        #[cfg(windows)]
        {
            out.push(resource.join("bundled").join("keryxd.exe").display().to_string());
        }
        out.push(resource.join("bundled").join("keryxd").display().to_string());
    }
    out
}

pub async fn expand_archive(zip_path: &str, dest: &Path) -> Result<(), String> {
    #[cfg(windows)]
    {
        let zip_escaped = zip_path.replace('\'', "''");
        let dest_escaped = dest.display().to_string().replace('\'', "''");
        let status = Command::new("powershell.exe")
            .args([
                "-NoProfile",
                "-Command",
                &format!(
                    "Expand-Archive -LiteralPath '{zip_escaped}' -DestinationPath '{dest_escaped}' -Force"
                ),
            ])
            .creation_flags(0x0800_0000)
            .status()
            .await
            .map_err(|e| e.to_string())?;
        if status.success() {
            Ok(())
        } else {
            Err(format!("Expand-Archive failed with code {:?}", status.code()))
        }
    }
    #[cfg(not(windows))]
    {
        let status = Command::new("unzip")
            .args(["-o", zip_path, "-d", &dest.display().to_string()])
            .status()
            .await
            .map_err(|e| e.to_string())?;
        if status.success() {
            Ok(())
        } else {
            Err(format!("unzip failed with code {:?}", status.code()))
        }
    }
}
