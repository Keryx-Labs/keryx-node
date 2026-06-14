/// RocksDB-backed store for OPoI synthetic-liveness tracking (Level-1).
///
/// Maps a miner's escrow public key (the 32-byte Schnorr key it publishes in its
/// coinbase `/escrow:` field) to the last synthetic-task epoch it answered.
///
/// Recording is unconditional (the store fills before the hardfork gate so the
/// network has history at activation). Enforcement — rejecting a block whose
/// coinbase escrow miner has a stale epoch — is gated separately by
/// `synthetic_liveness_activation` and lives in body validation (Step 4).
use std::{fmt, mem::size_of, sync::Arc};

use keryx_database::prelude::{BatchDbWriter, CachedDbAccess, CachePolicy, DirectDbWriter, StoreError, DB};
use keryx_database::registry::DatabaseStorePrefixes;
use keryx_utils::mem_size::MemSizeEstimator;
use rocksdb::WriteBatch;
use serde::{Deserialize, Serialize};

/// 32-byte key: a miner's escrow Schnorr public key.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct EscrowPubkeyKey([u8; 32]);

impl EscrowPubkeyKey {
    pub fn new(pubkey: [u8; 32]) -> Self {
        Self(pubkey)
    }
}

impl AsRef<[u8]> for EscrowPubkeyKey {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl fmt::Display for EscrowPubkeyKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", hex::encode(self.0))
    }
}

/// Per-miner liveness record: the most recent synthetic-task epoch answered.
#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct MinerLivenessRecord {
    pub last_epoch: u64,
}

impl MemSizeEstimator for MinerLivenessRecord {
    fn estimate_mem_bytes(&self) -> usize {
        size_of::<Self>()
    }
}

pub trait MinerLivenessStoreReader {
    /// Returns the last synthetic-task epoch answered by `pubkey`, or `None` if
    /// the miner has never answered (so it has no recorded liveness yet).
    fn last_epoch(&self, pubkey: EscrowPubkeyKey) -> Result<Option<u64>, StoreError>;
}

pub trait MinerLivenessStore: MinerLivenessStoreReader {
    fn set(&self, pubkey: EscrowPubkeyKey, last_epoch: u64) -> Result<(), StoreError>;
}

#[derive(Clone)]
pub struct DbMinerLivenessStore {
    db: Arc<DB>,
    access: CachedDbAccess<EscrowPubkeyKey, MinerLivenessRecord>,
}

impl DbMinerLivenessStore {
    pub fn new(db: Arc<DB>, cache_policy: CachePolicy) -> Self {
        Self {
            db: Arc::clone(&db),
            access: CachedDbAccess::new(db, cache_policy, DatabaseStorePrefixes::MinerLiveness.into()),
        }
    }

    pub fn set_batch(&self, batch: &mut WriteBatch, pubkey: EscrowPubkeyKey, last_epoch: u64) -> Result<(), StoreError> {
        self.access.write(BatchDbWriter::new(batch), pubkey, MinerLivenessRecord { last_epoch })?;
        Ok(())
    }
}

impl MinerLivenessStoreReader for DbMinerLivenessStore {
    fn last_epoch(&self, pubkey: EscrowPubkeyKey) -> Result<Option<u64>, StoreError> {
        match self.access.read(pubkey) {
            Ok(record) => Ok(Some(record.last_epoch)),
            Err(StoreError::KeyNotFound(_)) => Ok(None),
            Err(e) => Err(e),
        }
    }
}

impl MinerLivenessStore for DbMinerLivenessStore {
    fn set(&self, pubkey: EscrowPubkeyKey, last_epoch: u64) -> Result<(), StoreError> {
        self.access.write(DirectDbWriter::new(&self.db), pubkey, MinerLivenessRecord { last_epoch })?;
        Ok(())
    }
}
