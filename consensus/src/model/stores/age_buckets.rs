//! Coin-age (holder-reward v3) bucket index — strategy A of the spec (per-coin maturity cap with
//! O(1) reward lookup). One `AgeBuckets` aggregate per payout SPK, split at the maturity boundary
//! `age >= W` (i.e. `effective_daa <= d − W`):
//!
//! - `b_mat` — Σ amount over MATURE coins (each contributes its face value);
//! - `b_imm` / `a_imm` — Σ amount / Σ amount·effective_daa over IMMATURE coins (linear ramp).
//!
//! `eff_balance = b_mat + (d·b_imm − a_imm)/W` (see `coin_age::eff_balance_from_buckets`) replaces
//! the raw balance in the ratio bracket at/after `coin_age_activation`. The lookup cost is bounded
//! by block mass regardless of how fragmented an address is — the reason strategy A was chosen
//! over on-demand UTXO iteration (unbounded, a consensus-DoS surface via dust fragmentation).
//!
//! Maintained in lockstep with the virtual UTXO set (`apply_age_diff`) and rebuilt from it at
//! startup (each UTXO carries `effective_daa`, so the rebuild is exact). Until the maturation
//! queue lands, coins classified immature at insert are re-classified only by the startup rebuild;
//! nothing reads the store before `coin_age_activation`, so the interim drift is inert.

use std::sync::Arc;

use keryx_consensus_core::tx::ScriptPublicKey;
use keryx_database::prelude::CachePolicy;
use keryx_database::prelude::DB;
use keryx_database::prelude::StoreError;
use keryx_database::prelude::{BatchDbWriter, CachedDbAccess, StoreResultExt};
use keryx_database::registry::DatabaseStorePrefixes;
use keryx_utils::mem_size::MemSizeEstimator;
use rocksdb::WriteBatch;
use serde::{Deserialize, Serialize};

use super::address_amount::ScriptPublicKeyBucket;

/// Per-SPK coin-age aggregates. `a_imm` is `u128`: the anchor mass `Σ amount·effective_daa`
/// (sompi × DAA score) overflows `u64` at supply scale.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgeBuckets {
    pub b_mat: u64,
    pub b_imm: u64,
    pub a_imm: u128,
}

impl AgeBuckets {
    pub fn is_empty(&self) -> bool {
        self.b_mat == 0 && self.b_imm == 0 && self.a_imm == 0
    }
}

// Fixed-size Copy struct: the blanket size_of-based estimation is exact.
impl MemSizeEstimator for AgeBuckets {}

pub trait AgeBucketsStoreReader {
    fn get(&self, script_public_key: &ScriptPublicKey) -> Result<AgeBuckets, StoreError>;
}

/// A DB + cache implementation of `AgeBucketsStoreReader`, mirroring `DbAddressAmountStore`
/// (same SPK key encoding, same delete-on-empty keyspace discipline).
#[derive(Clone)]
pub struct DbAgeBucketsStore {
    access: CachedDbAccess<ScriptPublicKeyBucket, AgeBuckets>,
}

impl DbAgeBucketsStore {
    pub fn new(db: Arc<DB>, cache_policy: CachePolicy) -> Self {
        Self { access: CachedDbAccess::new(db, cache_policy, DatabaseStorePrefixes::AgeBuckets.into()) }
    }

    /// Sets the aggregates for `script_public_key`. Writing an all-zero value deletes the entry
    /// (indistinguishable from absence on read), keeping the keyspace to non-empty addresses only.
    pub fn set_batch(&self, batch: &mut WriteBatch, script_public_key: &ScriptPublicKey, buckets: AgeBuckets) -> Result<(), StoreError> {
        let key = ScriptPublicKeyBucket::from(script_public_key);
        if buckets.is_empty() {
            self.access.delete(BatchDbWriter::new(batch), key)
        } else {
            self.access.write(BatchDbWriter::new(batch), key, buckets)
        }
    }

    /// Clears the whole index (used before a startup rebuild or a fast-sync re-seed).
    pub fn clear(&self, batch: &mut WriteBatch) -> Result<(), StoreError> {
        self.access.delete_all(BatchDbWriter::new(batch))
    }

    /// Test helper: collects the whole index into a map (see `DbAddressAmountStore::collect`).
    #[cfg(test)]
    pub fn collect(&self) -> std::collections::HashMap<ScriptPublicKey, AgeBuckets> {
        self.access
            .iterator()
            .map(|res| {
                let (key, buckets) = res.unwrap();
                (ScriptPublicKey::from(ScriptPublicKeyBucket::from_bytes(key.to_vec())), buckets)
            })
            .collect()
    }
}

impl AgeBucketsStoreReader for DbAgeBucketsStore {
    fn get(&self, script_public_key: &ScriptPublicKey) -> Result<AgeBuckets, StoreError> {
        // Absent key ⇒ empty aggregates (no coins held for this address).
        Ok(self.access.read(ScriptPublicKeyBucket::from(script_public_key)).optional()?.unwrap_or_default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use keryx_consensus_core::tx::{ScriptPublicKey, ScriptVec};
    use keryx_database::{create_temp_db, prelude::ConnBuilder};

    fn spk(byte: u8) -> ScriptPublicKey {
        ScriptPublicKey::new(0, ScriptVec::from_slice(&[byte; 3]))
    }

    #[test]
    fn age_buckets_round_trip() {
        let (_lifetime, db) = create_temp_db!(ConnBuilder::default().with_files_limit(10));
        let store = DbAgeBucketsStore::new(db.clone(), CachePolicy::Count(16));
        let a = spk(1);

        // Absent key reads as empty.
        assert_eq!(store.get(&a).unwrap(), AgeBuckets::default());

        // Write, read back (u128 anchor mass round-trips).
        let buckets = AgeBuckets { b_mat: 1_000, b_imm: 500, a_imm: u128::from(u64::MAX) * 3 };
        let mut batch = WriteBatch::default();
        store.set_batch(&mut batch, &a, buckets).unwrap();
        db.write(batch).unwrap();
        assert_eq!(store.get(&a).unwrap(), buckets);

        // All-zero deletes the entry.
        let mut batch = WriteBatch::default();
        store.set_batch(&mut batch, &a, AgeBuckets::default()).unwrap();
        db.write(batch).unwrap();
        assert_eq!(store.get(&a).unwrap(), AgeBuckets::default());
    }
}
