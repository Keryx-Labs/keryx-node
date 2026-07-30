use crate::config::{launcher_dir, LauncherConfig};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StoreShape {
    pub config: LauncherConfig,
    pub setup_complete: bool,
    pub last_sync_duration_seconds: Option<f64>,
    pub last_sync_finished_at: Option<u64>,
}

impl Default for StoreShape {
    fn default() -> Self {
        Self {
            config: LauncherConfig::default(),
            setup_complete: false,
            last_sync_duration_seconds: None,
            last_sync_finished_at: None,
        }
    }
}

pub struct Store {
    path: PathBuf,
    data: StoreShape,
}

impl Store {
    pub fn load() -> Self {
        let path = store_path();
        let data = if path.is_file() {
            fs::read_to_string(&path)
                .ok()
                .and_then(|s| serde_json::from_str(&s).ok())
                .unwrap_or_default()
        } else {
            StoreShape::default()
        };
        Self { path, data }
    }

    pub fn get(&self) -> &StoreShape {
        &self.data
    }

    pub fn get_mut(&mut self) -> &mut StoreShape {
        &mut self.data
    }

    pub fn save(&self) -> Result<(), String> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let json = serde_json::to_string_pretty(&self.data).map_err(|e| e.to_string())?;
        fs::write(&self.path, json).map_err(|e| e.to_string())
    }
}

/// Portable store: lives next to the launcher executable (install folder), not AppData.
fn store_path() -> PathBuf {
    launcher_dir().join("keryx-launcher.json")
}
