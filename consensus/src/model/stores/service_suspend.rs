/// RocksDB store of service-bond production suspensions. Written only for finality-deep third
/// strikes (reorg-immune), so writes are monotone and never rolled back. Keyed by the miner's
/// escrow key; the value is the deadline DAA up to which that miner's blocks are rejected. Read at
/// boot to rebuild the RAM suspension map consulted by block validation.
use std::sync::Arc;

use keryx_database::prelude::{CachedDbAccess, CachePolicy, DirectDbWriter, StoreError, DB};
use keryx_database::registry::DatabaseStorePrefixes;
use keryx_hashes::Hash;

#[derive(Clone)]
pub struct DbServiceSuspendStore {
    db: Arc<DB>,
    access: CachedDbAccess<Hash, u64>,
}

impl DbServiceSuspendStore {
    pub fn new(db: Arc<DB>, cache_policy: CachePolicy) -> Self {
        Self { db: Arc::clone(&db), access: CachedDbAccess::new(db, cache_policy, DatabaseStorePrefixes::ServiceSuspend.into()) }
    }

    /// Records a suspension, keeping the later deadline if the miner is already suspended.
    pub fn set(&self, miner: Hash, until_daa: u64) -> Result<(), StoreError> {
        self.access.write(DirectDbWriter::new(&self.db), miner, until_daa)
    }

    /// All suspensions with their deadline DAA, for the boot load.
    pub fn iterator(&self) -> impl Iterator<Item = Result<(Box<[u8]>, u64), Box<dyn std::error::Error>>> + '_ {
        self.access.iterator()
    }
}
