/// RocksDB-backed store for OPoI synthetic-liveness annotations (Level-1, option C).
///
/// Keyed by **block hash**: records that block H's own body provided a valid
/// synthetic OPoI answer for H's coinbase miner, and at which epoch. The value is
/// an immutable function of H's content, so reading it for an ancestor block is
/// fully deterministic (no mutable global state, no reorg-dependent split risk) —
/// this is what lets enforcement (Step 4, option C) verify a miner's liveness by
/// following a `/live:<H>` coinbase reference + a reachability ancestor check,
/// instead of reading a mutable per-miner map.
///
/// Recording is unconditional (fills before the hardfork gate). Enforcement —
/// rejecting a block whose miner has no fresh self/ancestor answer — is gated by
/// `synthetic_liveness_activation`.
use std::{mem::size_of, sync::Arc};

use keryx_consensus_core::BlockHasher;
use keryx_database::prelude::{BatchDbWriter, CachedDbAccess, CachePolicy, DirectDbWriter, StoreError, DB};
use keryx_database::registry::DatabaseStorePrefixes;
use keryx_hashes::Hash;
use keryx_utils::mem_size::MemSizeEstimator;
use rocksdb::WriteBatch;
use serde::{Deserialize, Serialize};

/// What a block contributes to OPoI liveness: a valid synthetic answer for
/// `escrow_pubkey` (the block's own coinbase miner) at `epoch`.
#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct LivenessAnnotation {
    /// The block's coinbase escrow Schnorr pubkey — the miner this answer proves live.
    pub escrow_pubkey: [u8; 32],
    /// The synthetic epoch the answer satisfied (`daa / SYNTHETIC_EPOCH_BLOCKS`).
    pub epoch: u64,
}

impl MemSizeEstimator for LivenessAnnotation {
    fn estimate_mem_bytes(&self) -> usize {
        size_of::<Self>()
    }
}

pub trait MinerLivenessStoreReader {
    /// Returns block `H`'s liveness annotation, or `None` if `H` provided none.
    fn get(&self, block: Hash) -> Result<Option<LivenessAnnotation>, StoreError>;
}

pub trait MinerLivenessStore: MinerLivenessStoreReader {
    fn set(&self, block: Hash, annotation: LivenessAnnotation) -> Result<(), StoreError>;
}

#[derive(Clone)]
pub struct DbMinerLivenessStore {
    db: Arc<DB>,
    access: CachedDbAccess<Hash, LivenessAnnotation, BlockHasher>,
}

impl DbMinerLivenessStore {
    pub fn new(db: Arc<DB>, cache_policy: CachePolicy) -> Self {
        Self {
            db: Arc::clone(&db),
            access: CachedDbAccess::new(db, cache_policy, DatabaseStorePrefixes::MinerLiveness.into()),
        }
    }

    pub fn set_batch(&self, batch: &mut WriteBatch, block: Hash, annotation: LivenessAnnotation) -> Result<(), StoreError> {
        self.access.write(BatchDbWriter::new(batch), block, annotation)?;
        Ok(())
    }
}

impl MinerLivenessStoreReader for DbMinerLivenessStore {
    fn get(&self, block: Hash) -> Result<Option<LivenessAnnotation>, StoreError> {
        match self.access.read(block) {
            Ok(annotation) => Ok(Some(annotation)),
            Err(StoreError::KeyNotFound(_)) => Ok(None),
            Err(e) => Err(e),
        }
    }
}

impl MinerLivenessStore for DbMinerLivenessStore {
    fn set(&self, block: Hash, annotation: LivenessAnnotation) -> Result<(), StoreError> {
        self.access.write(DirectDbWriter::new(&self.db), block, annotation)?;
        Ok(())
    }
}
