use std::ops::Deref;
use std::sync::Arc;

use arc_swap::ArcSwap;
use keryx_consensus_core::api::stats::VirtualStateStats;
use keryx_consensus_core::{
    BlockHashMap, BlockHashSet, HashMapCustomHasher, block::VirtualStateApproxId, coinbase::BlockRewardData,
    config::genesis::GenesisBlock, tx::TransactionId, utxo::utxo_diff::UtxoDiff,
};
use keryx_database::prelude::{BatchDbWriter, CachedDbItem, DirectDbWriter, StoreResultExt};
use keryx_database::prelude::{CachePolicy, StoreResult};
use keryx_database::prelude::{DB, StoreError};
use keryx_database::registry::DatabaseStorePrefixes;
use keryx_hashes::Hash;
use keryx_muhash::MuHash;
use rocksdb::WriteBatch;
use serde::{Deserialize, Serialize};

use super::ghostdag::GhostdagData;
use super::utxo_set::{DbUtxoSetStore, UtxoDiffPreH4};

#[derive(Clone, Serialize, Deserialize, Default)]
pub struct VirtualState {
    pub parents: Vec<Hash>,
    pub ghostdag_data: GhostdagData,
    pub daa_score: u64,
    pub bits: u32,
    pub past_median_time: u64,
    pub multiset: MuHash,
    pub utxo_diff: UtxoDiff, // This is the UTXO diff from the selected tip to the virtual. i.e., if this diff is applied on the past UTXO of the selected tip, we'll get the virtual UTXO set.
    pub accepted_tx_ids: Vec<TransactionId>, // TODO: consider saving `accepted_id_merkle_root` directly
    pub mergeset_rewards: BlockHashMap<BlockRewardData>,
    pub mergeset_non_daa: BlockHashSet,
}

impl VirtualState {
    pub fn new(
        parents: Vec<Hash>,
        daa_score: u64,
        bits: u32,
        past_median_time: u64,
        multiset: MuHash,
        utxo_diff: UtxoDiff,
        accepted_tx_ids: Vec<TransactionId>,
        mergeset_rewards: BlockHashMap<BlockRewardData>,
        mergeset_non_daa: BlockHashSet,
        ghostdag_data: GhostdagData,
    ) -> Self {
        Self {
            parents,
            ghostdag_data,
            daa_score,
            bits,
            past_median_time,
            multiset,
            utxo_diff,
            accepted_tx_ids,
            mergeset_rewards,
            mergeset_non_daa,
        }
    }

    pub fn from_genesis(genesis: &GenesisBlock, ghostdag_data: GhostdagData) -> Self {
        Self {
            parents: vec![genesis.hash],
            ghostdag_data,
            daa_score: genesis.daa_score,
            bits: genesis.bits,
            past_median_time: genesis.timestamp,
            multiset: MuHash::new(),
            utxo_diff: UtxoDiff::default(), // Virtual diff is initially empty since genesis receives no reward
            accepted_tx_ids: genesis.build_genesis_transactions().into_iter().map(|tx| tx.id()).collect(),
            mergeset_rewards: BlockHashMap::new(),
            mergeset_non_daa: BlockHashSet::from_iter(std::iter::once(genesis.hash)),
        }
    }

    pub fn to_virtual_state_approx_id(&self) -> VirtualStateApproxId {
        VirtualStateApproxId::new(self.daa_score, self.ghostdag_data.blue_work, self.ghostdag_data.selected_parent)
    }
}

impl From<&VirtualState> for VirtualStateStats {
    fn from(state: &VirtualState) -> Self {
        Self {
            num_parents: state.parents.len() as u32,
            daa_score: state.daa_score,
            bits: state.bits,
            past_median_time: state.past_median_time,
        }
    }
}

/// Represents the "last known good" virtual state. To be used by any logic which does not want to wait
/// for a possible virtual state write to complete but can rather settle with the last known state
#[derive(Clone, Default)]
pub struct LkgVirtualState {
    inner: Arc<ArcSwap<VirtualState>>,
}

/// Guard for accessing the last known good virtual state (lock-free)
/// It's a simple newtype over arc_swap::Guard just to avoid explicit dependency
pub struct LkgVirtualStateGuard(arc_swap::Guard<Arc<VirtualState>>);

impl Deref for LkgVirtualStateGuard {
    type Target = Arc<VirtualState>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl LkgVirtualState {
    /// Provides a temporary borrow to the last known good virtual state.
    pub fn load(&self) -> LkgVirtualStateGuard {
        LkgVirtualStateGuard(self.inner.load())
    }

    /// Loads the last known good virtual state.
    pub fn load_full(&self) -> Arc<VirtualState> {
        self.inner.load_full()
    }

    // Kept private in order to make sure it is only updated by DbVirtualStateStore
    fn store(&self, virtual_state: Arc<VirtualState>) {
        self.inner.store(virtual_state)
    }
}

/// Used in order to group virtual related stores under a single lock
pub struct VirtualStores {
    pub state: DbVirtualStateStore,
    pub utxo_set: DbUtxoSetStore,
}

impl VirtualStores {
    pub fn new(db: Arc<DB>, lkg_virtual_state: LkgVirtualState, utxoset_cache_policy: CachePolicy) -> Self {
        Self {
            state: DbVirtualStateStore::new(db.clone(), lkg_virtual_state),
            utxo_set: DbUtxoSetStore::new(db, utxoset_cache_policy, DatabaseStorePrefixes::VirtualUtxoset.into()),
        }
    }
}

/// Reader API for `VirtualStateStore`.
pub trait VirtualStateStoreReader {
    fn get(&self) -> StoreResult<Arc<VirtualState>>;
}

pub trait VirtualStateStore: VirtualStateStoreReader {
    fn set(&mut self, state: Arc<VirtualState>) -> StoreResult<()>;
}

/// A DB + cache implementation of `VirtualStateStore` trait
#[derive(Clone)]
pub struct DbVirtualStateStore {
    db: Arc<DB>,
    access: CachedDbItem<Arc<VirtualState>>,
    /// The "last known good" virtual state
    lkg_virtual_state: LkgVirtualState,
}

/// Pre-H4 layout of the serialized virtual state: identical field order, with the embedded
/// `utxo_diff` holding pre-`effective_daa` utxo entries. A datadir written by an older binary
/// stores this layout under the same key; reads fall back to it and backfill the coin-age
/// anchor with the pre-H4 invariant (`effective_daa = block_daa_score`), like the utxoset store.
#[derive(Deserialize)]
struct VirtualStatePreH4 {
    parents: Vec<Hash>,
    ghostdag_data: GhostdagData,
    daa_score: u64,
    bits: u32,
    past_median_time: u64,
    multiset: MuHash,
    utxo_diff: UtxoDiffPreH4,
    accepted_tx_ids: Vec<TransactionId>,
    mergeset_rewards: BlockHashMap<BlockRewardData>,
    mergeset_non_daa: BlockHashSet,
}

impl From<VirtualStatePreH4> for Arc<VirtualState> {
    fn from(s: VirtualStatePreH4) -> Self {
        Arc::new(VirtualState {
            parents: s.parents,
            ghostdag_data: s.ghostdag_data,
            daa_score: s.daa_score,
            bits: s.bits,
            past_median_time: s.past_median_time,
            multiset: s.multiset,
            utxo_diff: s.utxo_diff.into(),
            accepted_tx_ids: s.accepted_tx_ids,
            mergeset_rewards: s.mergeset_rewards,
            mergeset_non_daa: s.mergeset_non_daa,
        })
    }
}

impl DbVirtualStateStore {
    pub fn new(db: Arc<DB>, lkg_virtual_state: LkgVirtualState) -> Self {
        let access = CachedDbItem::new(db.clone(), DatabaseStorePrefixes::VirtualState.into());
        // Init the LKG cache from DB store data
        lkg_virtual_state.store(access.read_with_decode_fallback::<VirtualStatePreH4>().optional().unwrap().unwrap_or_default());
        Self { db, access, lkg_virtual_state }
    }

    pub fn clone_with_new_cache(&self) -> Self {
        Self::new(self.db.clone(), self.lkg_virtual_state.clone())
    }

    pub fn is_initialized(&self) -> StoreResult<bool> {
        match self.access.read() {
            Ok(_) => Ok(true),
            Err(StoreError::KeyNotFound(_)) => Ok(false),
            Err(e) => Err(e),
        }
    }

    pub fn set_batch(&mut self, batch: &mut WriteBatch, state: Arc<VirtualState>) -> StoreResult<()> {
        self.lkg_virtual_state.store(state.clone()); // Keep the LKG cache up-to-date
        self.access.write(BatchDbWriter::new(batch), &state)
    }
}

impl VirtualStateStoreReader for DbVirtualStateStore {
    fn get(&self) -> StoreResult<Arc<VirtualState>> {
        self.access.read_with_decode_fallback::<VirtualStatePreH4>()
    }
}

impl VirtualStateStore for DbVirtualStateStore {
    fn set(&mut self, state: Arc<VirtualState>) -> StoreResult<()> {
        self.lkg_virtual_state.store(state.clone()); // Keep the LKG cache up-to-date
        self.access.write(DirectDbWriter::new(&self.db), &state)
    }
}
