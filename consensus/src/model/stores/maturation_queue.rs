//! Coin-age (holder-reward v3) maturation queue — the time-driven half of strategy A.
//!
//! A coin's bucket classification (`age_buckets`) shifts WITHOUT a transaction when its age
//! crosses `W`: at `daa = effective_daa + W` it must move from the immature ramp to the mature
//! face-value bucket. This store holds one entry per IMMATURE coin, keyed by its maturity score
//! (big-endian, so a prefix scan yields due coins in order), and is swept at each virtual commit:
//! every entry with `maturity_daa <= new virtual score` is promoted into `b_mat` and removed.
//! The per-commit sweep cost is bounded by the coins created `W` ago (one block's output mass per
//! elapsed score unit) — never by an address's fragmentation.
//!
//! Reorg safety: promotions are monotone in the virtual daa score. The single-key WATERMARK
//! records the highest swept score; if a commit ever observes a LOWER virtual score (deep reorg
//! past the maturity boundary), incremental state is invalid and the caller performs a full
//! coin-age rebuild from the UTXO set (exact — every entry carries its anchor) instead of
//! attempting fragile incremental demotion.

use std::sync::Arc;

use keryx_consensus_core::tx::{ScriptPublicKey, TransactionOutpoint};
use keryx_database::prelude::CachePolicy;
use keryx_database::prelude::DB;
use keryx_database::prelude::StoreError;
use keryx_database::prelude::{BatchDbWriter, CachedDbAccess, StoreResultExt};
use keryx_database::registry::DatabaseStorePrefixes;
use keryx_hashes::HASH_SIZE;
use keryx_utils::mem_size::MemSizeEstimator;
use rocksdb::WriteBatch;
use serde::{Deserialize, Serialize};

/// Queue key: 8-byte big-endian maturity score, then the 36-byte outpoint. BE ordering makes the
/// RocksDB prefix iteration yield due coins first, so the sweep is a bounded range scan.
const QUEUE_KEY_SIZE: usize = size_of::<u64>() + HASH_SIZE + size_of::<u32>();

#[derive(Eq, Hash, PartialEq, Debug, Clone, Copy)]
pub struct MaturationKey([u8; QUEUE_KEY_SIZE]);

impl MaturationKey {
    pub fn new(maturity_daa: u64, outpoint: &TransactionOutpoint) -> Self {
        let mut bytes = [0u8; QUEUE_KEY_SIZE];
        bytes[..8].copy_from_slice(&maturity_daa.to_be_bytes());
        bytes[8..8 + HASH_SIZE].copy_from_slice(&outpoint.transaction_id.as_bytes());
        bytes[8 + HASH_SIZE..].copy_from_slice(&outpoint.index.to_le_bytes());
        Self(bytes)
    }

    /// The maturity score encoded in raw key bytes read back from an iterator.
    pub fn maturity_of(raw: &[u8]) -> u64 {
        u64::from_be_bytes(raw[..8].try_into().expect("queue key starts with a be u64"))
    }
}

impl AsRef<[u8]> for MaturationKey {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl std::fmt::Display for MaturationKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "maturity={}", Self::maturity_of(&self.0))
    }
}

/// Queue value: the data needed to promote the coin without re-reading the UTXO set.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaturationEntry {
    pub script_public_key: ScriptPublicKey,
    pub amount: u64,
    pub anchor: u64,
}

impl MemSizeEstimator for MaturationEntry {}

#[derive(Clone)]
pub struct DbMaturationQueueStore {
    access: CachedDbAccess<MaturationKey, MaturationEntry>,
    watermark: CachedDbAccess<String, u64>,
}

const WATERMARK_KEY: &str = "wm";

impl DbMaturationQueueStore {
    pub fn new(db: Arc<DB>) -> Self {
        Self {
            // Range-scanned once then deleted — a cache would only hold dead entries.
            access: CachedDbAccess::new(db.clone(), CachePolicy::Empty, DatabaseStorePrefixes::MaturationQueue.into()),
            watermark: CachedDbAccess::new(db, CachePolicy::Count(1), DatabaseStorePrefixes::CoinAgeWatermark.into()),
        }
    }

    /// Enqueues an immature coin at its maturity score (`anchor + W`).
    pub fn insert_batch(
        &self,
        batch: &mut WriteBatch,
        maturity_daa: u64,
        outpoint: &TransactionOutpoint,
        entry: MaturationEntry,
    ) -> Result<(), StoreError> {
        self.access.write(BatchDbWriter::new(batch), MaturationKey::new(maturity_daa, outpoint), entry)
    }

    /// Removes a queued coin (spent while immature, or just promoted).
    pub fn delete_batch(&self, batch: &mut WriteBatch, maturity_daa: u64, outpoint: &TransactionOutpoint) -> Result<(), StoreError> {
        self.access.delete(BatchDbWriter::new(batch), MaturationKey::new(maturity_daa, outpoint))
    }

    /// All queue entries due at/below `daa_bound`, in maturity order — the promotion sweep input.
    /// Returns the raw key bytes alongside each entry so the caller can delete exactly what it saw.
    pub fn due(&self, daa_bound: u64) -> Vec<(Box<[u8]>, MaturationEntry)> {
        self.due_range(0, daa_bound)
    }

    /// Queue entries with maturity in `(from_exclusive, to_inclusive]`, in maturity order. The
    /// sweep promotes exactly one such disjoint range per commit (from the previous watermark to
    /// the new virtual score); the ratio read path scans the same shape to reconcile the committed
    /// split with a POV score that differs from the watermark (see `eff_balance_for_spk`).
    pub fn due_range(&self, from_exclusive: u64, to_inclusive: u64) -> Vec<(Box<[u8]>, MaturationEntry)> {
        if from_exclusive >= to_inclusive {
            return Vec::new();
        }
        // Seek to the first possible key of maturity `from+1` (zeroed outpoint sorts first).
        let seek = MaturationKey::new(from_exclusive + 1, &TransactionOutpoint::default());
        self.access
            .seek_iterator(None, Some(seek), usize::MAX, false)
            .map(|res| res.unwrap())
            .take_while(|(raw, _)| MaturationKey::maturity_of(raw) <= to_inclusive)
            .collect()
    }

    /// Deletes every retained entry with maturity at/below `bound` — the retention pruning pass.
    /// Promoted entries are kept for a bounded horizon so the read path can DEMOTE when validating
    /// a POV below the watermark (deep side-chain); beyond the horizon they are garbage.
    pub fn prune_below(&self, batch: &mut WriteBatch, bound: u64) -> Result<(), StoreError> {
        for (raw, _) in self.due(bound) {
            self.delete_raw_batch(batch, &raw)?;
        }
        Ok(())
    }

    /// The outpoint encoded in raw key bytes returned by `due`/`due_range`.
    pub fn outpoint_of(raw: &[u8]) -> TransactionOutpoint {
        let transaction_id = keryx_hashes::Hash::from_slice(&raw[8..8 + HASH_SIZE]);
        let index = u32::from_le_bytes(raw[8 + HASH_SIZE..].try_into().expect("queue key ends with a le u32"));
        TransactionOutpoint::new(transaction_id, index)
    }

    /// Deletes a swept entry by its raw key bytes (as returned by `due`).
    pub fn delete_raw_batch(&self, batch: &mut WriteBatch, raw: &[u8]) -> Result<(), StoreError> {
        let mut bytes = [0u8; QUEUE_KEY_SIZE];
        bytes.copy_from_slice(raw);
        self.access.delete(BatchDbWriter::new(batch), MaturationKey(bytes))
    }

    /// The highest virtual score the queue has been swept to (`None` before the first sweep).
    pub fn get_watermark(&self) -> Result<Option<u64>, StoreError> {
        self.watermark.read(WATERMARK_KEY.to_string()).optional()
    }

    pub fn set_watermark_batch(&self, batch: &mut WriteBatch, daa_score: u64) -> Result<(), StoreError> {
        self.watermark.write(BatchDbWriter::new(batch), WATERMARK_KEY.to_string(), daa_score)
    }

    /// Clears the queue AND the watermark (before a full coin-age rebuild).
    pub fn clear(&self, batch: &mut WriteBatch) -> Result<(), StoreError> {
        self.access.delete_all(BatchDbWriter::new(batch))?;
        self.watermark.delete_all(BatchDbWriter::new(batch))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use keryx_consensus_core::tx::ScriptVec;
    use keryx_database::{create_temp_db, prelude::ConnBuilder};

    #[test]
    fn maturation_queue_sweep_order_and_bound() {
        let (_lifetime, db) = create_temp_db!(ConnBuilder::default().with_files_limit(10));
        let store = DbMaturationQueueStore::new(db.clone());
        let spk = ScriptPublicKey::new(0, ScriptVec::from_slice(&[7; 3]));
        let entry = |amount| MaturationEntry { script_public_key: spk.clone(), amount, anchor: 1 };
        let op = |n: u64| TransactionOutpoint::new(keryx_hashes::Hash::from_u64_word(n), 0);

        let mut batch = WriteBatch::default();
        store.insert_batch(&mut batch, 300, &op(3), entry(30)).unwrap();
        store.insert_batch(&mut batch, 100, &op(1), entry(10)).unwrap();
        store.insert_batch(&mut batch, 200, &op(2), entry(20)).unwrap();
        db.write(batch).unwrap();

        // Due at 250: maturities 100 and 200, in order; 300 stays queued.
        let due = store.due(250);
        assert_eq!(due.iter().map(|(_, e)| e.amount).collect::<Vec<_>>(), vec![10, 20]);

        // Sweep: delete what was seen, advance the watermark.
        let mut batch = WriteBatch::default();
        for (raw, _) in &due {
            store.delete_raw_batch(&mut batch, raw).unwrap();
        }
        store.set_watermark_batch(&mut batch, 250).unwrap();
        db.write(batch).unwrap();

        assert_eq!(store.get_watermark().unwrap(), Some(250));
        assert_eq!(store.due(u64::MAX).iter().map(|(_, e)| e.amount).collect::<Vec<_>>(), vec![30]);
    }
}
