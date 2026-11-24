use std::sync::Arc;

use keryx_consensus_core::BlockHasher;
use keryx_consensus_core::coinbase::TierReserve;
use keryx_database::prelude::CachePolicy;
use keryx_database::prelude::DB;
use keryx_database::prelude::StoreError;
use keryx_database::prelude::{BatchDbWriter, CachedDbAccess, DirectDbWriter};
use keryx_database::registry::DatabaseStorePrefixes;
use keryx_hashes::Hash;
use rocksdb::WriteBatch;

/// Read access to the tier reserve state after each UTXO-valid chain block, read as the starting
/// reserve of its chain children's coinbase.
pub trait TierReserveStoreReader {
    fn get(&self, hash: Hash) -> Result<TierReserve, StoreError>;
    fn has(&self, hash: Hash) -> Result<bool, StoreError>;
}

/// A DB + cache implementation of `TierReserveStoreReader`, with concurrency support.
#[derive(Clone)]
pub struct DbTierReserveStore {
    db: Arc<DB>,
    access: CachedDbAccess<Hash, TierReserve, BlockHasher>,
}

impl DbTierReserveStore {
    pub fn new(db: Arc<DB>, cache_policy: CachePolicy) -> Self {
        Self { db: Arc::clone(&db), access: CachedDbAccess::new(db, cache_policy, DatabaseStorePrefixes::TierReserve.into()) }
    }

    pub fn clone_with_new_cache(&self, cache_policy: CachePolicy) -> Self {
        Self::new(Arc::clone(&self.db), cache_policy)
    }

    pub fn insert_batch(&self, batch: &mut WriteBatch, hash: Hash, reserve: TierReserve) -> Result<(), StoreError> {
        self.access.write(BatchDbWriter::new(batch), hash, reserve)
    }

    /// Sets the reserve of `hash` outside a batch (snapshot import at a new pruning point).
    pub fn set(&self, hash: Hash, reserve: TierReserve) -> Result<(), StoreError> {
        self.access.write(DirectDbWriter::new(&self.db), hash, reserve)
    }

    pub fn delete_batch(&self, batch: &mut WriteBatch, hash: Hash) -> Result<(), StoreError> {
        self.access.delete(BatchDbWriter::new(batch), hash)
    }
}

impl TierReserveStoreReader for DbTierReserveStore {
    fn get(&self, hash: Hash) -> Result<TierReserve, StoreError> {
        self.access.read(hash)
    }

    fn has(&self, hash: Hash) -> Result<bool, StoreError> {
        self.access.has(hash)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use keryx_database::prelude::StoreResultExt;
    use keryx_database::{create_temp_db, prelude::ConnBuilder};

    #[test]
    fn tier_reserve_round_trip() {
        let (_lifetime, db) = create_temp_db!(ConnBuilder::default().with_files_limit(10));
        let store = DbTierReserveStore::new(db.clone(), CachePolicy::Count(16));
        let (a, b): (Hash, Hash) = (1.into(), 2.into());
        let mut batch = WriteBatch::default();
        let (ra, rb) = (TierReserve { reserve: 7, ask_average: 3 }, TierReserve { reserve: u64::MAX, ask_average: 0 });
        store.insert_batch(&mut batch, a, ra).unwrap();
        store.insert_batch(&mut batch, b, rb).unwrap();
        db.write(batch).unwrap();
        assert_eq!(store.get(a).unwrap(), ra);
        assert_eq!(store.get(b).unwrap(), rb);
        assert_eq!(store.get(9.into()).optional().unwrap(), None);
        let mut batch = WriteBatch::default();
        store.delete_batch(&mut batch, a).unwrap();
        db.write(batch).unwrap();
        assert!(!store.has(a).unwrap());
    }
}
