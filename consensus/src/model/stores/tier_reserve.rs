use std::sync::Arc;

use keryx_consensus_core::BlockHasher;
use keryx_database::prelude::CachePolicy;
use keryx_database::prelude::DB;
use keryx_database::prelude::StoreError;
use keryx_database::prelude::{BatchDbWriter, CachedDbAccess};
use keryx_database::registry::DatabaseStorePrefixes;
use keryx_hashes::Hash;
use rocksdb::WriteBatch;

/// Read access to the tier reserve (sompi) after each UTXO-valid chain block, read as the starting
/// reserve of its chain children's coinbase.
pub trait TierReserveStoreReader {
    fn get(&self, hash: Hash) -> Result<u64, StoreError>;
    fn has(&self, hash: Hash) -> Result<bool, StoreError>;
}

/// A DB + cache implementation of `TierReserveStoreReader`, with concurrency support.
#[derive(Clone)]
pub struct DbTierReserveStore {
    db: Arc<DB>,
    access: CachedDbAccess<Hash, u64, BlockHasher>,
}

impl DbTierReserveStore {
    pub fn new(db: Arc<DB>, cache_policy: CachePolicy) -> Self {
        Self { db: Arc::clone(&db), access: CachedDbAccess::new(db, cache_policy, DatabaseStorePrefixes::TierReserve.into()) }
    }

    pub fn clone_with_new_cache(&self, cache_policy: CachePolicy) -> Self {
        Self::new(Arc::clone(&self.db), cache_policy)
    }

    pub fn insert_batch(&self, batch: &mut WriteBatch, hash: Hash, reserve: u64) -> Result<(), StoreError> {
        self.access.write(BatchDbWriter::new(batch), hash, reserve)
    }

    pub fn delete_batch(&self, batch: &mut WriteBatch, hash: Hash) -> Result<(), StoreError> {
        self.access.delete(BatchDbWriter::new(batch), hash)
    }
}

impl TierReserveStoreReader for DbTierReserveStore {
    fn get(&self, hash: Hash) -> Result<u64, StoreError> {
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
        store.insert_batch(&mut batch, a, 7).unwrap();
        store.insert_batch(&mut batch, b, u64::MAX).unwrap();
        db.write(batch).unwrap();
        assert_eq!(store.get(a).unwrap(), 7);
        assert_eq!(store.get(b).unwrap(), u64::MAX);
        assert_eq!(store.get(9.into()).optional().unwrap(), None);
        let mut batch = WriteBatch::default();
        store.delete_batch(&mut batch, a).unwrap();
        db.write(batch).unwrap();
        assert!(!store.has(a).unwrap());
    }
}
