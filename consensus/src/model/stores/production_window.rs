//! Window daa table of the production-index snapshot imported at the pruning point: daa score of
//! every selected-chain index in `[bottom_index, sample_index]`. Cleared once the pruning point
//! leaves that sample.
use std::sync::Arc;

use keryx_database::prelude::{BatchDbWriter, CachedDbItem, DB, StoreError};
use parking_lot::RwLock;
use keryx_database::registry::DatabaseStorePrefixes;
use rocksdb::{IteratorMode, ReadOptions, WriteBatch};
use serde::{Deserialize, Serialize};

/// Bounds of the imported window (chain indices, network numbering).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImportedProductionWindow {
    pub bottom_index: u64,
    pub sample_index: u64,
}

#[derive(Clone)]
pub struct DbProductionWindowStore {
    db: Arc<DB>,
    daa_prefix: u8,
    bounds: Arc<RwLock<CachedDbItem<ImportedProductionWindow>>>,
    // IBD fast path: `chain_index_daa` is called ~18× per validated block
    // inside `chain_index_at_or_below_daa` binary search. Each call was a
    // RocksDB `get` on the window table (190k keys for the 864k DAA window).
    // Cache the whole table in RAM after `install` so the hot path is a
    // single `RwLock` read + vec index, not a DB seek.
    cached_window: Arc<RwLock<Option<(u64, Vec<u64>)>>>,
}

impl DbProductionWindowStore {
    pub fn new(db: Arc<DB>) -> Self {
        Self {
            db: Arc::clone(&db),
            daa_prefix: DatabaseStorePrefixes::ProductionWindowDaa as u8,
            bounds: Arc::new(RwLock::new(CachedDbItem::new(db, DatabaseStorePrefixes::ProductionImportedWindow.into()))),
            cached_window: Arc::new(RwLock::new(None)),
        }
    }

    fn daa_key(&self, index: u64) -> Vec<u8> {
        let mut key = Vec::with_capacity(9);
        key.push(self.daa_prefix);
        key.extend_from_slice(&index.to_be_bytes());
        key
    }

    /// Replaces the table with `daa[i]` = daa of chain index `bottom_index + i`, in one batch.
    pub fn install(&self, batch: &mut WriteBatch, bottom_index: u64, sample_index: u64, daa: &[u64]) -> Result<(), StoreError> {
        self.clear(batch);
        for (i, d) in daa.iter().enumerate() {
            batch.put(self.daa_key(bottom_index + i as u64), d.to_le_bytes());
        }
        *self.cached_window.write() = Some((bottom_index, daa.to_vec()));
        self.bounds.write().write(BatchDbWriter::new(batch), &ImportedProductionWindow { bottom_index, sample_index })
    }

    /// Drops the table and its bounds.
    pub fn clear(&self, batch: &mut WriteBatch) {
        batch.delete_range(vec![self.daa_prefix], vec![self.daa_prefix + 1]);
        *self.cached_window.write() = None;
        let _ = self.bounds.write().remove(BatchDbWriter::new(batch));
    }

    /// Bounds of the imported window, if one is installed.
    pub fn imported(&self) -> Option<ImportedProductionWindow> {
        self.bounds.read().read().ok()
    }

    /// daa of chain `index` from the imported table.
    pub fn daa_at(&self, index: u64) -> Option<u64> {
        // Fast path: cached vec
        {
            let guard = self.cached_window.read();
            if let Some((bottom, vec)) = guard.as_ref() {
                if index >= *bottom {
                    if let Some(v) = vec.get((index - *bottom) as usize) {
                        return Some(*v);
                    }
                }
                // Cache installed but index outside it => not in window
                if self.bounds.read().read().is_ok() {
                    return None;
                }
            }
        }
        // Cold start: window is on disk but cache is empty (restart). Warm it once.
        if self.cached_window.read().is_none() {
            if let Ok(imported) = self.bounds.read().read() {
                let mut vec = Vec::with_capacity((imported.sample_index - imported.bottom_index + 1) as usize);
                let mut ok = true;
                for idx in imported.bottom_index..=imported.sample_index {
                    match self.db.get(self.daa_key(idx)).ok().flatten() {
                        Some(v) => vec.push(u64::from_le_bytes(v[..8].try_into().expect("daa is 8 bytes"))),
                        None => {
                            ok = false;
                            break;
                        }
                    }
                }
                if ok {
                    *self.cached_window.write() = Some((imported.bottom_index, vec));
                    // retry via cache
                    return self.daa_at(index);
                }
            }
        }
        self.db.get(self.daa_key(index)).ok().flatten().map(|v| u64::from_le_bytes(v[..8].try_into().expect("daa is 8 bytes")))
    }

    /// True when the table holds at least one entry.
    pub fn is_installed(&self) -> bool {
        let mut opts = ReadOptions::default();
        opts.set_iterate_range(rocksdb::PrefixRange([self.daa_prefix].as_slice()));
        self.db.iterator_opt(IteratorMode::Start, opts).next().is_some()
    }
}
