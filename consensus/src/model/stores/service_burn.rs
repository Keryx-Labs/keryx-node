/// RocksDB store of service-bond burned escrow outpoints. Written only for finality-deep misses
/// (reorg-immune), so writes are monotone and idempotent and never rolled back. Read at boot to
/// rebuild the RAM burned-set consulted by transaction validation.
use std::sync::Arc;

use keryx_database::prelude::{CachedDbAccess, CachePolicy, DirectDbWriter, StoreError, DB};
use keryx_database::registry::DatabaseStorePrefixes;

use super::ai_slash::OutpointKey;

#[derive(Clone)]
pub struct DbServiceBurnStore {
    db: Arc<DB>,
    access: CachedDbAccess<OutpointKey, u64>,
}

impl DbServiceBurnStore {
    pub fn new(db: Arc<DB>, cache_policy: CachePolicy) -> Self {
        Self { db: Arc::clone(&db), access: CachedDbAccess::new(db, cache_policy, DatabaseStorePrefixes::ServiceBurn.into()) }
    }

    pub fn set(&self, key: OutpointKey, miss_daa: u64) -> Result<(), StoreError> {
        self.access.write(DirectDbWriter::new(&self.db), key, miss_daa)
    }

    /// All burned outpoints with their miss daa, for the boot load.
    pub fn iterator(&self) -> impl Iterator<Item = Result<(Box<[u8]>, u64), Box<dyn std::error::Error>>> + '_ {
        self.access.iterator()
    }

    /// Deletes every burn with a miss daa above `daa`; returns how many.
    pub fn delete_above(&self, daa: u64) -> Result<usize, StoreError> {
        let keys: Vec<OutpointKey> = self
            .access
            .iterator()
            .filter_map(|r| r.ok())
            .filter(|(_, miss_daa)| *miss_daa > daa)
            .map(|(k, _)| {
                let tx_id: [u8; 32] = k[..32].try_into().unwrap();
                OutpointKey::new(tx_id.into(), u32::from_le_bytes(k[32..36].try_into().unwrap()))
            })
            .collect();
        self.access.delete_many(DirectDbWriter::new(&self.db), &mut keys.iter().cloned())?;
        Ok(keys.len())
    }
}
