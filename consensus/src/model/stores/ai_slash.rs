/// RocksDB-backed stores for the OPoI slash mechanism (Phase 3 A4).
///
/// Two stores:
/// - `DbAiResponseStore`: confirmed AiResponse txs, keyed by response_hash.
/// - `DbAiSlashedStore`: slashed escrow outpoints, keyed by outpoint bytes.

use std::{fmt, mem::size_of, sync::Arc};

use keryx_consensus_core::BlockHasher;
use keryx_database::prelude::{BatchDbWriter, CachedDbAccess, CachePolicy, DirectDbWriter, StoreError, DB};
use keryx_database::registry::DatabaseStorePrefixes;
use keryx_hashes::Hash;
use keryx_utils::mem_size::MemSizeEstimator;
use rocksdb::WriteBatch;
use serde::{Deserialize, Serialize};

// ── AiResponseStore ─────────────��────────────────���───────────────────────────

/// Data recorded when an AiResponse tx is confirmed.
#[derive(Clone, Serialize, Deserialize)]
pub struct AiResponseRecord {
    /// Blue score of the block that included this AiResponse.
    pub inclusion_blue_score: u64,
    /// Transaction ID of the coinbase in the same block.
    /// The escrow output is always at index 1 of this coinbase.
    pub coinbase_tx_id: Hash,
    /// `blake2b(raw_AiRequest_payload)[0..32]` — copied from AiResponse.request_hash.
    /// Used by Phase 3 C fraud verification to re-derive the expected commitment.
    pub request_hash: [u8; 32],
    /// First 32 bytes of `AiResponse.result` as published by the miner.
    /// Phase 3 C: must equal `model_fixed::forward(request_hash)`.
    /// A challenger proves fraud by showing this value is wrong.
    pub claimed_commitment: [u8; 32],
}

impl MemSizeEstimator for AiResponseRecord {
    fn estimate_mem_bytes(&self) -> usize {
        size_of::<Self>()
    }
}

pub trait AiResponseStoreReader {
    fn get(&self, response_hash: Hash) -> Result<AiResponseRecord, StoreError>;
    fn has(&self, response_hash: Hash) -> Result<bool, StoreError>;
}

pub trait AiResponseStore: AiResponseStoreReader {
    fn set(&self, response_hash: Hash, record: AiResponseRecord) -> Result<(), StoreError>;
}

#[derive(Clone)]
pub struct DbAiResponseStore {
    db: Arc<DB>,
    access: CachedDbAccess<Hash, AiResponseRecord, BlockHasher>,
}

impl DbAiResponseStore {
    pub fn new(db: Arc<DB>, cache_policy: CachePolicy) -> Self {
        Self {
            db: Arc::clone(&db),
            access: CachedDbAccess::new(db, cache_policy, DatabaseStorePrefixes::AiResponse.into()),
        }
    }

    pub fn set_batch(&self, batch: &mut WriteBatch, response_hash: Hash, record: AiResponseRecord) -> Result<(), StoreError> {
        self.access.write(BatchDbWriter::new(batch), response_hash, record)?;
        Ok(())
    }
}

impl AiResponseStoreReader for DbAiResponseStore {
    fn get(&self, response_hash: Hash) -> Result<AiResponseRecord, StoreError> {
        self.access.read(response_hash)
    }

    fn has(&self, response_hash: Hash) -> Result<bool, StoreError> {
        match self.access.read(response_hash) {
            Ok(_) => Ok(true),
            Err(StoreError::KeyNotFound(_)) => Ok(false),
            Err(e) => Err(e),
        }
    }
}

impl AiResponseStore for DbAiResponseStore {
    fn set(&self, response_hash: Hash, record: AiResponseRecord) -> Result<(), StoreError> {
        self.access.write(DirectDbWriter::new(&self.db), response_hash, record)?;
        Ok(())
    }
}

// ── AiSlashedStore ──────────────────────────────────��──────────────────────��──

/// 36-byte key encoding a TransactionOutpoint: [tx_id: 32][index: 4 LE].
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct OutpointKey([u8; 36]);

impl OutpointKey {
    pub fn new(tx_id: Hash, index: u32) -> Self {
        let mut bytes = [0u8; 36];
        bytes[..32].copy_from_slice(&tx_id.as_bytes());
        bytes[32..].copy_from_slice(&index.to_le_bytes());
        Self(bytes)
    }
}

impl AsRef<[u8]> for OutpointKey {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl fmt::Display for OutpointKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", hex::encode(self.0))
    }
}

pub trait AiSlashedStoreReader {
    fn is_slashed(&self, key: OutpointKey) -> Result<bool, StoreError>;
}

pub trait AiSlashedStore: AiSlashedStoreReader {
    /// Record a slash. `slash_blue_score` is stored for audit purposes.
    fn set(&self, key: OutpointKey, slash_blue_score: u64) -> Result<(), StoreError>;
}

#[derive(Clone)]
pub struct DbAiSlashedStore {
    db: Arc<DB>,
    access: CachedDbAccess<OutpointKey, u64>,
}

impl DbAiSlashedStore {
    pub fn new(db: Arc<DB>, cache_policy: CachePolicy) -> Self {
        Self {
            db: Arc::clone(&db),
            access: CachedDbAccess::new(db, cache_policy, DatabaseStorePrefixes::AiSlashed.into()),
        }
    }

    pub fn set_batch(&self, batch: &mut WriteBatch, key: OutpointKey, slash_blue_score: u64) -> Result<(), StoreError> {
        self.access.write(BatchDbWriter::new(batch), key, slash_blue_score)?;
        Ok(())
    }
}

impl AiSlashedStoreReader for DbAiSlashedStore {
    fn is_slashed(&self, key: OutpointKey) -> Result<bool, StoreError> {
        match self.access.read(key) {
            Ok(_) => Ok(true),
            Err(StoreError::KeyNotFound(_)) => Ok(false),
            Err(e) => Err(e),
        }
    }
}

impl AiSlashedStore for DbAiSlashedStore {
    fn set(&self, key: OutpointKey, slash_blue_score: u64) -> Result<(), StoreError> {
        self.access.write(DirectDbWriter::new(&self.db), key, slash_blue_score)?;
        Ok(())
    }
}
