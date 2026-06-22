use std::sync::Arc;

use keryx_consensus_core::BlockHasher;
use keryx_database::prelude::CachePolicy;
use keryx_database::prelude::DB;
use keryx_database::prelude::StoreError;
use keryx_database::prelude::{BatchDbWriter, CachedDbAccess};
use keryx_database::registry::DatabaseStorePrefixes;
use keryx_hashes::Hash;
use rocksdb::WriteBatch;

/// Read access to the computed holder-ratio bracket (`RATIO_REWARD_BPS` multiplier) of each block,
/// written once the producer's `balance ÷ windowed_production` ratio has been resolved against the
/// balance and production indexes. Mirrors `pom_tier::DbPomTierStore`: the ratio is a chain-state
/// aggregate that is not recoverable from the block alone, so the resolved bps is persisted here
/// for the ratio-reward coinbase split read by the virtual processor.
///
/// The value is a basis-points multiplier in `[RATIO_REWARD_BPS[0], RATIO_REWARD_BPS_DIVISOR]`.
pub trait RatioBpsStoreReader {
    fn get(&self, hash: Hash) -> Result<u64, StoreError>;
    fn has(&self, hash: Hash) -> Result<bool, StoreError>;
}

/// A DB + cache implementation of `RatioBpsStoreReader`, with concurrency support.
#[derive(Clone)]
pub struct DbRatioBpsStore {
    db: Arc<DB>,
    access: CachedDbAccess<Hash, u64, BlockHasher>,
}

impl DbRatioBpsStore {
    pub fn new(db: Arc<DB>, cache_policy: CachePolicy) -> Self {
        Self { db: Arc::clone(&db), access: CachedDbAccess::new(db, cache_policy, DatabaseStorePrefixes::RatioBps.into()) }
    }

    pub fn clone_with_new_cache(&self, cache_policy: CachePolicy) -> Self {
        Self::new(Arc::clone(&self.db), cache_policy)
    }

    // This is append only
    pub fn insert_batch(&self, batch: &mut WriteBatch, hash: Hash, bps: u64) -> Result<(), StoreError> {
        if self.access.has(hash)? {
            return Err(StoreError::HashAlreadyExists(hash));
        }
        self.access.write(BatchDbWriter::new(batch), hash, bps)?;
        Ok(())
    }

    pub fn delete_batch(&self, batch: &mut WriteBatch, hash: Hash) -> Result<(), StoreError> {
        self.access.delete(BatchDbWriter::new(batch), hash)
    }
}

impl RatioBpsStoreReader for DbRatioBpsStore {
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
    fn ratio_bps_round_trip() {
        let (_lifetime, db) = create_temp_db!(ConnBuilder::default().with_files_limit(10));
        let store = DbRatioBpsStore::new(db.clone(), CachePolicy::Count(16));
        let (a, b): (Hash, Hash) = (1.into(), 2.into());

        // Write two brackets in a batch.
        let mut batch = WriteBatch::default();
        store.insert_batch(&mut batch, a, 10_000).unwrap();
        store.insert_batch(&mut batch, b, 4_000).unwrap();
        db.write(batch).unwrap();

        // Read back.
        assert_eq!(store.get(a).unwrap(), 10_000);
        assert_eq!(store.get(b).unwrap(), 4_000);
        assert!(store.has(a).unwrap());

        // Missing key: get is KeyNotFound (None via optional), has is false.
        let c: Hash = 9.into();
        assert!(!store.has(c).unwrap());
        assert_eq!(store.get(c).optional().unwrap(), None);

        // Append-only: re-inserting the same hash errors.
        let mut batch = WriteBatch::default();
        assert!(store.insert_batch(&mut batch, a, 5_200).is_err());

        // Delete is a no-op on a missing key (pre-fork blocks have no bracket), and removes a present one.
        let mut batch = WriteBatch::default();
        store.delete_batch(&mut batch, c).unwrap(); // missing → Ok
        store.delete_batch(&mut batch, a).unwrap();
        db.write(batch).unwrap();
        assert!(!store.has(a).unwrap());
    }
}
