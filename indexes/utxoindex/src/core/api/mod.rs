use keryx_consensus_core::{
    BlockHashSet,
    tx::{ScriptPublicKey, ScriptPublicKeys, TransactionOutpoint},
    utxo::utxo_diff::UtxoDiff,
};
use keryx_consensusmanager::spawn_blocking;
use keryx_database::prelude::StoreResult;
use keryx_hashes::Hash;
use keryx_index_core::indexed_utxos::BalanceByScriptPublicKey;
use parking_lot::RwLock;
use std::{collections::HashSet, fmt::Debug, sync::Arc};

use crate::{
    errors::UtxoIndexResult,
    model::{CompactUtxoCollection, CompactUtxoEntry, UtxoChanges, UtxoSetByScriptPublicKey},
};

///Utxoindex API targeted at retrieval calls.
pub trait UtxoIndexApi: Send + Sync + Debug {
    /// Retrieve circulating supply from the utxoindex db.
    ///
    /// Note: Use a read lock when accessing this method
    fn get_circulating_supply(&self) -> StoreResult<u64>;

    /// Retrieve utxos by script public keys supply from the utxoindex db.
    ///
    /// Note: Use a read lock when accessing this method
    fn get_utxos_by_script_public_keys(&self, script_public_keys: ScriptPublicKeys) -> StoreResult<UtxoSetByScriptPublicKey>;

    fn get_balance_by_script_public_keys(&self, script_public_keys: ScriptPublicKeys) -> StoreResult<BalanceByScriptPublicKey>;

    /// Retrieve up to `limit` utxos of one script public key, resuming strictly after
    /// `resume_after` when provided.
    ///
    /// Note: Use a read lock when accessing this method. This is the bounded-hold building
    /// block: callers chunk whale-sized scans so the lock is never held for a full bucket.
    fn get_utxos_by_script_public_key_chunk(
        &self,
        script_public_key: &ScriptPublicKey,
        resume_after: Option<TransactionOutpoint>,
        limit: usize,
    ) -> StoreResult<Vec<(TransactionOutpoint, CompactUtxoEntry)>>;

    // This can have a big memory footprint, so it should be used only for tests.
    fn get_all_outpoints(&self) -> StoreResult<HashSet<TransactionOutpoint>>;

    /// Retrieve the stored tips of the utxoindex (used for testing purposes).
    ///
    /// Note: Use a read lock when accessing this method
    fn get_utxo_index_tips(&self) -> StoreResult<Arc<BlockHashSet>>;

    /// Checks if the utxoindex's db is synced with consensus.
    ///
    /// Note:
    /// 1) Use a read lock when accessing this method
    /// 2) due to potential sync-gaps is_synced is unreliable while consensus is actively resolving virtual states.  
    fn is_synced(&self) -> UtxoIndexResult<bool>;

    /// Update the utxoindex with the given utxo_diff, and tips.
    ///
    /// Note: Use a write lock when accessing this method
    fn update(&mut self, utxo_diff: Arc<UtxoDiff>, tips: Arc<Vec<Hash>>) -> UtxoIndexResult<UtxoChanges>;

    /// Resync the utxoindex from the consensus db
    ///
    /// Note: Use a write lock when accessing this method
    fn resync(&mut self) -> UtxoIndexResult<()>;
}

/// Upper bound on UTXO entries fetched per utxoindex read-lock hold.
///
/// Whale buckets are walked in chunks with the lock RELEASED between chunks: the
/// virtual-processor writer updates the index on every virtual state (~10/s), and
/// the RwLock is write-preferring, so a multi-second full-bucket scan parks the
/// writer and convoys every subsequent reader (GetBalance et al.) behind it —
/// measured as 9-20 s balance calls whenever a 50k+ UTXO address was being fetched.
///
/// Trade-off: a chunked result is assembled across lock releases and is not one
/// atomic snapshot of the set. RPC consumers already tolerate this — the mempool
/// is the final arbiter at broadcast time.
const UTXO_SCAN_CHUNK: usize = 8192;

/// Async proxy for the UTXO index
#[derive(Debug, Clone)]
pub struct UtxoIndexProxy {
    inner: Arc<RwLock<dyn UtxoIndexApi>>,
}

impl UtxoIndexProxy {
    pub fn new(inner: Arc<RwLock<dyn UtxoIndexApi>>) -> Self {
        Self { inner }
    }

    pub async fn get_circulating_supply(self) -> StoreResult<u64> {
        spawn_blocking(move || self.inner.read().get_circulating_supply()).await.unwrap()
    }

    pub async fn get_utxos_by_script_public_keys(self, script_public_keys: ScriptPublicKeys) -> StoreResult<UtxoSetByScriptPublicKey> {
        spawn_blocking(move || {
            let mut result = UtxoSetByScriptPublicKey::new();
            for script_public_key in script_public_keys {
                let mut collection = CompactUtxoCollection::new();
                let mut resume_after: Option<TransactionOutpoint> = None;
                loop {
                    let chunk =
                        self.inner.read().get_utxos_by_script_public_key_chunk(&script_public_key, resume_after, UTXO_SCAN_CHUNK)?;
                    let exhausted = chunk.len() < UTXO_SCAN_CHUNK;
                    resume_after = chunk.last().map(|(outpoint, _)| *outpoint);
                    collection.extend(chunk);
                    if exhausted {
                        break;
                    }
                }
                result.insert(script_public_key, collection);
            }
            Ok(result)
        })
        .await
        .unwrap()
    }

    pub async fn get_balance_by_script_public_keys(
        self,
        script_public_keys: ScriptPublicKeys,
    ) -> StoreResult<BalanceByScriptPublicKey> {
        spawn_blocking(move || {
            let mut result = BalanceByScriptPublicKey::new();
            for script_public_key in script_public_keys {
                let mut balance: u64 = 0;
                let mut resume_after: Option<TransactionOutpoint> = None;
                loop {
                    let chunk =
                        self.inner.read().get_utxos_by_script_public_key_chunk(&script_public_key, resume_after, UTXO_SCAN_CHUNK)?;
                    let exhausted = chunk.len() < UTXO_SCAN_CHUNK;
                    resume_after = chunk.last().map(|(outpoint, _)| *outpoint);
                    balance += chunk.iter().map(|(_, entry)| entry.amount).sum::<u64>();
                    if exhausted {
                        break;
                    }
                }
                result.insert(script_public_key, balance);
            }
            Ok(result)
        })
        .await
        .unwrap()
    }

    /// Count the UTXOs of one script public key without materializing the set,
    /// walking the bucket in bounded-lock chunks like the retrieval methods.
    pub async fn get_utxo_count_by_script_public_key(self, script_public_key: ScriptPublicKey) -> StoreResult<u64> {
        spawn_blocking(move || {
            let mut count: u64 = 0;
            let mut resume_after: Option<TransactionOutpoint> = None;
            loop {
                let chunk =
                    self.inner.read().get_utxos_by_script_public_key_chunk(&script_public_key, resume_after, UTXO_SCAN_CHUNK)?;
                count += chunk.len() as u64;
                if chunk.len() < UTXO_SCAN_CHUNK {
                    break;
                }
                resume_after = chunk.last().map(|(outpoint, _)| *outpoint);
            }
            Ok(count)
        })
        .await
        .unwrap()
    }

    pub async fn update(self, utxo_diff: Arc<UtxoDiff>, tips: Arc<Vec<Hash>>) -> UtxoIndexResult<UtxoChanges> {
        spawn_blocking(move || self.inner.write().update(utxo_diff, tips)).await.unwrap()
    }
}
