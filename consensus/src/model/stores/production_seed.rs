use keryx_database::prelude::DB;
use keryx_database::prelude::StoreResult;
use keryx_database::prelude::StoreResultExt;
use keryx_database::prelude::{BatchDbWriter, CachedDbItem};
use keryx_database::registry::DatabaseStorePrefixes;
use rocksdb::WriteBatch;
use std::sync::Arc;

/// Fast-sync catch-up tracking for the ratio-reward windowed-production index.
///
/// `import_pruning_point_utxo_set` clears the windowed-production prefix index on every pruning-point
/// UTXO import (fast sync) instead of seeding it, on the assumption that the whole `ratio_reward_window`
/// lies above the pruning point and gets rebuilt exactly by the body-IBD that follows. That holds
/// only while the pruning point itself is still before the ratio-reward activation height; once the
/// chain has advanced past `pruning_depth` beyond that height, every fresh fast sync starts its
/// production index at zero exactly when the activation is already live, and needs a full
/// `ratio_reward_window` of catch-up before its computed `ratio_bps` matches long-running nodes —
/// during which it disqualifies every ratio-reward coinbase it sees.
///
/// This store records the virtual selected-chain index at the moment of the last import, so the
/// node can tell whether it is still inside that catch-up window (see `trust_coinbase` in
/// `VirtualStateProcessor`) and self-expire the relaxation once the window has organically refilled.
pub trait ProductionIndexSeedStoreReader {
    fn get(&self) -> StoreResult<u64>;
}

pub trait ProductionIndexSeedStore: ProductionIndexSeedStoreReader {
    fn set_batch(&mut self, batch: &mut WriteBatch, seeded_at_index: u64) -> StoreResult<()>;
}

#[derive(Clone)]
pub struct DbProductionIndexSeedStore {
    access: CachedDbItem<u64>,
}

impl DbProductionIndexSeedStore {
    pub fn new(db: Arc<DB>) -> Self {
        Self { access: CachedDbItem::new(db, DatabaseStorePrefixes::ProductionIndexSeededAt.into()) }
    }

    pub fn clone_with_new_cache(&self, db: Arc<DB>) -> Self {
        Self::new(db)
    }

    /// `None` if the node has never imported a pruning-point UTXO snapshot (built from genesis, so the
    /// windowed-production prefix index was never cleared/reset and has no catch-up gap), or pre-dates
    /// this store.
    pub fn get_optional(&self) -> Option<u64> {
        self.access.read().optional().unwrap()
    }

    /// Clears the fast-sync catch-up marker when a selected-chain rewind rebuilds the prefix
    /// index from its retained chain. Keeping the old marker would unnecessarily trust coinbases
    /// after recovery, even though the rebuilt index is already authoritative.
    pub fn clear_batch(&mut self, batch: &mut WriteBatch) -> StoreResult<()> {
        self.access.delete_all(BatchDbWriter::new(batch))
    }
}

impl ProductionIndexSeedStoreReader for DbProductionIndexSeedStore {
    fn get(&self) -> StoreResult<u64> {
        self.access.read()
    }
}

impl ProductionIndexSeedStore for DbProductionIndexSeedStore {
    fn set_batch(&mut self, batch: &mut WriteBatch, seeded_at_index: u64) -> StoreResult<()> {
        self.access.write(BatchDbWriter::new(batch), &seeded_at_index)
    }
}
