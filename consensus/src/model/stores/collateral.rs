use std::sync::Arc;

use keryx_consensus_core::{BlockHasher, collateral::CollateralEntry};
use keryx_database::prelude::{BatchDbWriter, CachedDbAccess, CachePolicy, DirectDbWriter, StoreError, DB};
use keryx_database::registry::DatabaseStorePrefixes;
use keryx_hashes::Hash;
use rocksdb::WriteBatch;

pub trait CollateralStoreReader {
    fn get(&self, miner_hash: Hash) -> Result<CollateralEntry, StoreError>;
}

pub trait CollateralStore: CollateralStoreReader {
    fn set(&self, miner_hash: Hash, entry: CollateralEntry) -> Result<(), StoreError>;
}

#[derive(Clone)]
pub struct DbCollateralStore {
    db: Arc<DB>,
    access: CachedDbAccess<Hash, CollateralEntry, BlockHasher>,
}

impl DbCollateralStore {
    pub fn new(db: Arc<DB>, cache_policy: CachePolicy) -> Self {
        Self {
            db: Arc::clone(&db),
            access: CachedDbAccess::new(db, cache_policy, DatabaseStorePrefixes::MinerCollateral.into()),
        }
    }

    pub fn set_batch(&self, batch: &mut WriteBatch, miner_hash: Hash, entry: CollateralEntry) -> Result<(), StoreError> {
        self.access.write(BatchDbWriter::new(batch), miner_hash, entry)?;
        Ok(())
    }
}

impl CollateralStoreReader for DbCollateralStore {
    fn get(&self, miner_hash: Hash) -> Result<CollateralEntry, StoreError> {
        self.access.read(miner_hash)
    }
}

impl CollateralStore for DbCollateralStore {
    fn set(&self, miner_hash: Hash, entry: CollateralEntry) -> Result<(), StoreError> {
        self.access.write(DirectDbWriter::new(&self.db), miner_hash, entry)?;
        Ok(())
    }
}
