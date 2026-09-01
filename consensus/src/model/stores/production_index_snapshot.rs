/// RocksDB store of the canonical production-index snapshot taken at every pruning sample
/// (see `ProductionIndexSnapshot`), keyed by the sample block hash. Bounded by the GC in
/// `advance_service_ledger`.
use std::sync::Arc;

use keryx_database::prelude::{CachePolicy, CachedDbAccess, DB, DirectDbWriter, StoreError};
use keryx_database::registry::DatabaseStorePrefixes;
use keryx_hashes::Hash;

use super::service_ledger_snapshot::SnapshotBlob;

#[derive(Clone)]
pub struct DbProductionIndexSnapshotStore {
    db: Arc<DB>,
    access: CachedDbAccess<Hash, SnapshotBlob>,
}

impl DbProductionIndexSnapshotStore {
    pub fn new(db: Arc<DB>) -> Self {
        Self {
            db: Arc::clone(&db),
            access: CachedDbAccess::new(db, CachePolicy::Empty, DatabaseStorePrefixes::ProductionIndexSnapshot.into()),
        }
    }

    pub fn set(&self, sample: Hash, bytes: Vec<u8>) -> Result<(), StoreError> {
        self.access.write(DirectDbWriter::new(&self.db), sample, SnapshotBlob(bytes))
    }

    pub fn get(&self, sample: Hash) -> Result<Option<Vec<u8>>, StoreError> {
        match self.access.read(sample) {
            Ok(blob) => Ok(Some(blob.0)),
            Err(StoreError::KeyNotFound(_)) => Ok(None),
            Err(e) => Err(e),
        }
    }

    pub fn has(&self, sample: Hash) -> Result<bool, StoreError> {
        self.access.has(sample)
    }

    pub fn delete(&self, sample: Hash) -> Result<(), StoreError> {
        self.access.delete(DirectDbWriter::new(&self.db), sample)
    }

    /// Every stored (sample hash, snapshot bytes).
    pub fn entries(&self) -> Vec<(Hash, Vec<u8>)> {
        self.access
            .iterator()
            .filter_map(|r| r.ok())
            .filter_map(|(key, blob)| {
                key.get(..32).and_then(|k| <[u8; 32]>::try_from(k).ok()).map(|k| (Hash::from_bytes(k), blob.0))
            })
            .collect()
    }
}
