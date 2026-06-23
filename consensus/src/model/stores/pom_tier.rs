use std::sync::Arc;

use keryx_consensus_core::BlockHasher;
use keryx_database::prelude::CachePolicy;
use keryx_database::prelude::DB;
use keryx_database::prelude::StoreError;
use keryx_database::prelude::{BatchDbWriter, CachedDbAccess};
use keryx_database::registry::DatabaseStorePrefixes;
use keryx_hashes::Hash;
use rocksdb::WriteBatch;

/// Read access to the proven PoM tier (`PomProof::tier`) of each block, written at
/// body-commit time once the possession proof has been verified against the per-tier
/// root `R_T`. The tier is unavailable later (the `pom_proof` is dropped when a block
/// is reconstructed from storage), so it is persisted here for the tier-reward coinbase
/// split read by the virtual processor.
pub trait PomTierStoreReader {
    fn get(&self, hash: Hash) -> Result<u8, StoreError>;
    fn has(&self, hash: Hash) -> Result<bool, StoreError>;
}

/// A DB + cache implementation of `PomTierStoreReader`, with concurrency support.
#[derive(Clone)]
pub struct DbPomTierStore {
    db: Arc<DB>,
    access: CachedDbAccess<Hash, u8, BlockHasher>,
}

impl DbPomTierStore {
    pub fn new(db: Arc<DB>, cache_policy: CachePolicy) -> Self {
        Self { db: Arc::clone(&db), access: CachedDbAccess::new(db, cache_policy, DatabaseStorePrefixes::PomTier.into()) }
    }

    pub fn clone_with_new_cache(&self, cache_policy: CachePolicy) -> Self {
        Self::new(Arc::clone(&self.db), cache_policy)
    }

    // This is append only
    pub fn insert_batch(&self, batch: &mut WriteBatch, hash: Hash, tier: u8) -> Result<(), StoreError> {
        if self.access.has(hash)? {
            return Err(StoreError::HashAlreadyExists(hash));
        }
        self.access.write(BatchDbWriter::new(batch), hash, tier)?;
        Ok(())
    }

    pub fn delete_batch(&self, batch: &mut WriteBatch, hash: Hash) -> Result<(), StoreError> {
        self.access.delete(BatchDbWriter::new(batch), hash)
    }
}

impl PomTierStoreReader for DbPomTierStore {
    fn get(&self, hash: Hash) -> Result<u8, StoreError> {
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
    fn pom_tier_round_trip() {
        let (_lifetime, db) = create_temp_db!(ConnBuilder::default().with_files_limit(10));
        let store = DbPomTierStore::new(db.clone(), CachePolicy::Count(16));
        let (a, b): (Hash, Hash) = (1.into(), 2.into());

        // Write two tiers in a batch.
        let mut batch = WriteBatch::default();
        store.insert_batch(&mut batch, a, 3).unwrap();
        store.insert_batch(&mut batch, b, 0).unwrap();
        db.write(batch).unwrap();

        // Read back.
        assert_eq!(store.get(a).unwrap(), 3);
        assert_eq!(store.get(b).unwrap(), 0);
        assert!(store.has(a).unwrap());

        // Missing key: get is KeyNotFound (None via optional), has is false.
        let c: Hash = 9.into();
        assert!(!store.has(c).unwrap());
        assert_eq!(store.get(c).optional().unwrap(), None);

        // Append-only: re-inserting the same hash errors.
        let mut batch = WriteBatch::default();
        assert!(store.insert_batch(&mut batch, a, 1).is_err());

        // Delete is a no-op on a missing key (pre-fork blocks have no tier), and removes a present one.
        let mut batch = WriteBatch::default();
        store.delete_batch(&mut batch, c).unwrap(); // missing → Ok
        store.delete_batch(&mut batch, a).unwrap();
        db.write(batch).unwrap();
        assert!(!store.has(a).unwrap());
    }
}
