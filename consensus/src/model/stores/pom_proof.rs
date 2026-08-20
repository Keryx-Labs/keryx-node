use std::sync::Arc;

use keryx_consensus_core::BlockHasher;
use keryx_consensus_core::pom::{PomProof, PomProofPreH4, PomProofPreV3, PomProofPreV4};
use keryx_database::prelude::CachePolicy;
use keryx_database::prelude::DB;
use keryx_database::prelude::StoreError;
use keryx_database::prelude::{BatchDbWriter, CachedDbAccess};
use keryx_database::registry::DatabaseStorePrefixes;
use keryx_hashes::Hash;
use rocksdb::WriteBatch;

/// Read access to the full PoM possession proof of each block, persisted at body-commit time.
/// Required so a block can be re-served to peers (relay / IBD) with its proof attached:
/// `get_block` reconstructs the block from storage, and without this store `pom_proof` would be
/// `None`, causing peers to reject the served block with `PoM possession proof missing`. Only
/// blocks at/after `pom_activation` carry a proof; pre-fork blocks have no entry here.
pub trait PomProofStoreReader {
    fn get(&self, hash: Hash) -> Result<PomProof, StoreError>;
    fn has(&self, hash: Hash) -> Result<bool, StoreError>;
}

/// A DB + cache implementation of `PomProofStoreReader`, with concurrency support.
#[derive(Clone)]
pub struct DbPomProofStore {
    db: Arc<DB>,
    access: CachedDbAccess<Hash, PomProof, BlockHasher>,
}

impl DbPomProofStore {
    pub fn new(db: Arc<DB>, cache_policy: CachePolicy) -> Self {
        Self { db: Arc::clone(&db), access: CachedDbAccess::new(db, cache_policy, DatabaseStorePrefixes::PomProof.into()) }
    }

    pub fn clone_with_new_cache(&self, cache_policy: CachePolicy) -> Self {
        Self::new(Arc::clone(&self.db), cache_policy)
    }

    // This is append only
    pub fn insert_batch(&self, batch: &mut WriteBatch, hash: Hash, proof: &PomProof) -> Result<(), StoreError> {
        if self.access.has(hash)? {
            return Err(StoreError::HashAlreadyExists(hash));
        }
        self.access.write(BatchDbWriter::new(batch), hash, proof.clone())?;
        Ok(())
    }

    pub fn delete_batch(&self, batch: &mut WriteBatch, hash: Hash) -> Result<(), StoreError> {
        self.access.delete(BatchDbWriter::new(batch), hash)
    }
}

impl PomProofStoreReader for DbPomProofStore {
    fn get(&self, hash: Hash) -> Result<PomProof, StoreError> {
        // Records written before the H6 `v3` field existed are the pre-V3 positional layout;
        // before the H4 `steps_v2` field, the pre-H4 one. The grown `PomProof` under-flows on
        // their bytes, so decode falls back down the era chain (same mechanism as the utxoset
        // store, one more era deep). KeyNotFound short-circuits — only decode shapes chain.
        match self.access.read_with_decode_fallback::<PomProofPreV4>(hash) {
            Err(e @ StoreError::KeyNotFound(_)) => Err(e),
            Err(_) => match self.access.read_with_decode_fallback::<PomProofPreV3>(hash) {
                Err(_) => self.access.read_with_decode_fallback::<PomProofPreH4>(hash),
                ok => ok,
            },
            ok => ok,
        }
    }

    fn has(&self, hash: Hash) -> Result<bool, StoreError> {
        self.access.has(hash)
    }
}
