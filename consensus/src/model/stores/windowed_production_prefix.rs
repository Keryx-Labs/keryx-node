//! Gold-standard ratio-reward production index: a per-SPK **prefix sum** over the selected chain.
//!
//! # Why this exists
//! The ratio-reward needs, for a payout SPK at selected-chain block `b`, the sum of coinbase base
//! miner-cuts that SPK produced over the trailing window of `W` chain blocks. The legacy
//! `WindowedProduction` store kept this as a single mutable running sum maintained by `+delta`/
//! `−delta` folds. That value is **path-dependent**: two nodes on the identical chain could hold
//! different sums depending on the update history they happened to live through (from-genesis vs
//! fast-sync import, reorgs one saw and the other did not, pruning timing). Path dependence ⇒
//! coinbase divergence ⇒ chain splits.
//!
//! # The invariant this store guarantees
//! We instead store, sparsely at every chain index where an SPK produced, the **cumulative**
//! production for that SPK from genesis through that index:
//!
//! ```text
//!   cumulative_at(spk, t) = Σ block_production(spk) over selected-chain [1, t]
//!   windowed(spk, b, W)   = cumulative_at(spk, b) − cumulative_at(spk, b − W)
//! ```
//!
//! `cumulative_at` is a **pure function of the selected chain** — never of the update path — so any
//! two nodes on the same chain compute the identical windowed value by construction. A reorg is a
//! plain truncation (delete the entries above the split, re-append the new chain's); there is no
//! slide arithmetic and no saturating clamp that can silently lose state.
//!
//! # Storage layout (two registry prefixes)
//! - **entries** (`WindowedProductionPrefix`): key `SPK || be(index)` → `u64` cumulative. Big-endian
//!   index so RocksDB lexicographic order == numeric order *within* one SPK, enabling a reverse seek
//!   (`SeekForPrev`) to the largest entry `≤ t`.
//! - **floor** (`WindowedProductionFloor`): key `SPK` → `u64` cumulative up to the current pruning
//!   floor. When a block below the floor is pruned its per-block entry is collapsed here, so
//!   `cumulative_at` for an SPK whose last sub-floor production was pruned still returns the exact
//!   cumulative. Safe because `W < pruning_depth`: the window never reaches below the floor, so the
//!   floor only ever serves as the `cum(b−W)` baseline, never as an in-window term.

use std::collections::HashMap;
use std::sync::Arc;

use keryx_consensus_core::tx::ScriptPublicKey;
use keryx_database::prelude::{StoreError, DB};
use keryx_database::registry::DatabaseStorePrefixes;
use rocksdb::{Direction, IteratorMode, ReadOptions, WriteBatch};

use super::address_amount::ScriptPublicKeyBucket;

/// Read API for the prefix-sum production index.
pub trait WindowedProductionPrefixStoreReader {
    /// Cumulative production for `spk` over selected-chain `[1, index]` (a pure function of the chain).
    fn cumulative_at(&self, spk: &ScriptPublicKey, index: u64) -> Result<u64, StoreError>;

    /// Windowed production for `spk` ending at chain block `b` over a window of `w` blocks:
    /// `cumulative_at(b) − cumulative_at(b − w)` (or `cumulative_at(b)` when `b ≤ w`, i.e. the window
    /// reaches genesis). Always non-negative since cumulative is monotonic in the index.
    fn windowed(&self, spk: &ScriptPublicKey, b: u64, w: u64) -> Result<u64, StoreError> {
        let hi = self.cumulative_at(spk, b)?;
        if b <= w {
            return Ok(hi);
        }
        let lo = self.cumulative_at(spk, b - w)?;
        Ok(hi.saturating_sub(lo))
    }
}

/// A DB-backed prefix-sum production index. Reads hit committed state directly (no cache layer — the
/// reverse seek is a single `SeekForPrev` and queries are ~one-per-reward-payout); writes go through
/// the caller's consensus `WriteBatch` so they commit atomically with the selected-chain change.
#[derive(Clone)]
pub struct DbWindowedProductionPrefixStore {
    db: Arc<DB>,
    entries_prefix: u8,
    floor_prefix: u8,
}

impl DbWindowedProductionPrefixStore {
    pub fn new(db: Arc<DB>) -> Self {
        Self {
            db,
            entries_prefix: DatabaseStorePrefixes::WindowedProductionPrefix.into(),
            floor_prefix: DatabaseStorePrefixes::WindowedProductionFloor.into(),
        }
    }

    /// On-disk entry key: `entries_prefix || SPK_bucket || be(index)`.
    fn entry_key(&self, spk: &ScriptPublicKey, index: u64) -> Vec<u8> {
        let bucket = ScriptPublicKeyBucket::from(spk);
        let mut key = Vec::with_capacity(1 + bucket.as_ref().len() + 8);
        key.push(self.entries_prefix);
        key.extend_from_slice(bucket.as_ref());
        key.extend_from_slice(&index.to_be_bytes());
        key
    }

    /// The `entries_prefix || SPK_bucket` range that holds exactly one SPK's index→cumulative entries.
    fn spk_range(&self, spk: &ScriptPublicKey) -> Vec<u8> {
        let bucket = ScriptPublicKeyBucket::from(spk);
        let mut p = Vec::with_capacity(1 + bucket.as_ref().len());
        p.push(self.entries_prefix);
        p.extend_from_slice(bucket.as_ref());
        p
    }

    /// On-disk floor key: `floor_prefix || SPK_bucket`.
    fn floor_key(&self, spk: &ScriptPublicKey) -> Vec<u8> {
        let bucket = ScriptPublicKeyBucket::from(spk);
        let mut key = Vec::with_capacity(1 + bucket.as_ref().len());
        key.push(self.floor_prefix);
        key.extend_from_slice(bucket.as_ref());
        key
    }

    fn read_floor(&self, spk: &ScriptPublicKey) -> Result<u64, StoreError> {
        Ok(self.db.get(self.floor_key(spk))?.map(decode_u64).unwrap_or(0))
    }

    /// Raw cumulative write for one `(spk, index)` entry. Used by [`Self::extend`]; exposed for the
    /// rebuild path.
    pub fn put_cumulative(&self, batch: &mut WriteBatch, spk: &ScriptPublicKey, index: u64, cumulative: u64) {
        batch.put(self.entry_key(spk, index), cumulative.to_le_bytes());
    }

    /// Set the per-SPK floor baseline (cumulative up to the current pruning floor).
    pub fn put_floor(&self, batch: &mut WriteBatch, spk: &ScriptPublicKey, cumulative: u64) {
        batch.put(self.floor_key(spk), cumulative.to_le_bytes());
    }

    /// Extend the index along a selected-chain change in a single batch, EXACTLY (no slide, no clamp).
    ///
    /// `common` is the split index (the shared chain tip below the change). `removals` are the
    /// `(spk, index)` of chain blocks leaving the chain (index > `common`); `additions` are the
    /// `(spk, index, cut)` of chain blocks joining it, **in ascending index order** (`common+1, …`).
    ///
    /// Each addition's cumulative is seeded from `cumulative_at(spk, common)` — which a reverse seek
    /// computes correctly from committed state regardless of the about-to-be-removed entries (they sit
    /// at index > `common`, so the `≤ common` seek skips them) — then accumulated in memory across the
    /// additions so multiple blocks by the same SPK in one batch chain correctly.
    pub fn extend(
        &self,
        batch: &mut WriteBatch,
        common: u64,
        removals: &[(ScriptPublicKey, u64)],
        additions: &[(ScriptPublicKey, u64, u64)],
    ) -> Result<(), StoreError> {
        // Delete the entries of removed chain blocks first (distinct keys from additions even when
        // indices overlap, since the producer SPK differs; if they ever coincide the later put wins).
        for (spk, index) in removals {
            batch.delete(self.entry_key(spk, *index));
        }
        // Per-SPK running cumulative, seeded lazily from the committed cumulative at the split.
        let mut acc: HashMap<ScriptPublicKey, u64> = HashMap::new();
        for (spk, index, cut) in additions {
            let base = match acc.get(spk) {
                Some(v) => *v,
                None => self.cumulative_at(spk, common)?,
            };
            let new_cum = base + cut;
            self.put_cumulative(batch, spk, *index, new_cum);
            acc.insert(spk.clone(), new_cum);
        }
        Ok(())
    }

    /// Collapse the per-block entries of blocks being pruned into the per-SPK floor baseline, so
    /// `cumulative_at` stays exact for SPKs whose last sub-floor production has been pruned. `items`
    /// are the `(spk, index, cumulative)` of the pruned chain blocks (any order). The floor for an SPK
    /// becomes the max cumulative among its collapsed entries (cumulative is monotonic in index, so
    /// the max corresponds to the highest pruned index — order-independent and idempotent).
    pub fn collapse_to_floor(&self, batch: &mut WriteBatch, items: &[(ScriptPublicKey, u64, u64)]) -> Result<(), StoreError> {
        let mut new_floor: HashMap<ScriptPublicKey, u64> = HashMap::new();
        for (spk, index, cumulative) in items {
            batch.delete(self.entry_key(spk, *index));
            let e = new_floor.entry(spk.clone()).or_insert(0);
            *e = (*e).max(*cumulative);
        }
        for (spk, cumulative) in new_floor {
            // Existing floor (if any) is from a lower index ⇒ strictly smaller cumulative, so the
            // collapsed max supersedes it; still take the max for total order-independence.
            let merged = self.read_floor(&spk)?.max(cumulative);
            self.put_floor(batch, &spk, merged);
        }
        Ok(())
    }

    /// Fold every retained per-block entry with chain index `< floor_index` into its SPK's floor
    /// baseline (deleting the entry), so the index stays bounded to ~the pruning window instead of
    /// growing with the whole chain. Processes at most `entry_budget` entries per call, so a large
    /// first-run backlog drains gradually over successive pruning messages. Returns the count deleted.
    ///
    /// SAFE for consensus. `windowed_production_for_block` clamps every window bottom to
    /// `max(m_idx − W, pruning_idx)`, whose minimum — a block just above the pruning point — is
    /// `pruning_idx − W`; the caller passes `floor_index = pruning_idx − ratio_reward_window`, so no
    /// consensus read ever touches a collapsed index. Query-preserving at *every* intermediate commit
    /// too: entries are visited in ascending index order and the lowest are deleted first, so a per-SPK
    /// floor is never set above a still-retained entry — a reverse seek always returns the correct
    /// highest-retained cumulative (see `collapse_below_preserves_queries`). Because it changes no value
    /// the consensus reads, it needs no cross-node coordination (a collapsed and a freshly-built node
    /// return identical `windowed()`), and can never cause a UTXO / consensus divergence.
    pub fn collapse_below(&self, batch: &mut WriteBatch, floor_index: u64, entry_budget: u64) -> Result<u64, StoreError> {
        let mut opts = ReadOptions::default();
        opts.set_iterate_range(rocksdb::PrefixRange([self.entries_prefix].as_slice()));
        let it = self.db.iterator_opt(IteratorMode::Start, opts);

        // Per-SPK max cumulative among this call's collapsed entries (bucket bytes → cumulative).
        let mut folded: HashMap<Vec<u8>, u64> = HashMap::new();
        let mut deleted = 0u64;
        for item in it {
            if deleted >= entry_budget {
                break;
            }
            let (key, value) = item?;
            // Key layout: entries_prefix(1) || spk_bucket(N) || be(index)(8).
            let index = u64::from_be_bytes(key[key.len() - 8..].try_into().expect("index is 8 bytes"));
            if index >= floor_index {
                continue; // at/above the floor: the consensus may still read it — keep.
            }
            let bucket = key[1..key.len() - 8].to_vec();
            batch.delete(key.as_ref());
            let e = folded.entry(bucket).or_insert(0);
            *e = (*e).max(decode_u64(value.to_vec()));
            deleted += 1;
        }
        // Merge each collapsed max into the SPK's committed floor (monotonic, order-independent).
        for (bucket, cumulative) in folded {
            let mut fk = Vec::with_capacity(1 + bucket.len());
            fk.push(self.floor_prefix);
            fk.extend_from_slice(&bucket);
            let existing = self.db.get(&fk)?.map(decode_u64).unwrap_or(0);
            batch.put(fk, existing.max(cumulative).to_le_bytes());
        }
        Ok(deleted)
    }

    /// True if the index holds no entries at all (fresh prefix, or a datadir created before this
    /// store existed) — the trigger for the one-time from-chain build/migration at startup.
    pub fn is_empty(&self) -> bool {
        let mut opts = ReadOptions::default();
        opts.set_iterate_range(rocksdb::PrefixRange([self.entries_prefix].as_slice()));
        self.db.iterator_opt(IteratorMode::Start, opts).next().is_none()
    }

    /// Wipe the entire index (entries + floor). Used by the from-chain rebuild before re-deriving.
    pub fn clear(&self, batch: &mut WriteBatch) {
        for prefix in [self.entries_prefix, self.floor_prefix] {
            // delete_range covers [prefix, prefix+1) — the whole single-byte-prefixed keyspace.
            batch.delete_range(vec![prefix], vec![prefix + 1]);
        }
    }
}

impl WindowedProductionPrefixStoreReader for DbWindowedProductionPrefixStore {
    fn cumulative_at(&self, spk: &ScriptPublicKey, index: u64) -> Result<u64, StoreError> {
        let range = self.spk_range(spk);
        // Seek key = range || be(index). Reverse direction ⇒ SeekForPrev: the largest key ≤ seek.
        let mut seek = range.clone();
        seek.extend_from_slice(&index.to_be_bytes());
        let mut opts = ReadOptions::default();
        opts.set_iterate_range(rocksdb::PrefixRange(range.as_slice()));
        let mut it = self.db.iterator_opt(IteratorMode::From(seek.as_slice(), Direction::Reverse), opts);
        if let Some(item) = it.next() {
            let (_key, value) = item?;
            // PrefixRange bounds the iterator to this SPK, so any hit is a real entry ≤ index.
            return Ok(decode_u64(value.as_ref().to_vec()));
        }
        // No retained entry ≤ index for this SPK ⇒ the floor baseline (0 if never pruned/seen).
        self.read_floor(spk)
    }
}

fn decode_u64(bytes: Vec<u8>) -> u64 {
    u64::from_le_bytes(bytes[..8].try_into().expect("production index value is 8 bytes"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use keryx_consensus_core::tx::{ScriptPublicKey, ScriptVec};
    use keryx_database::{create_temp_db, prelude::ConnBuilder};

    fn spk(byte: u8) -> ScriptPublicKey {
        ScriptPublicKey::new(0, ScriptVec::from_slice(&[byte, byte, byte]))
    }

    /// Brute-force oracle: the windowed sum computed directly from the per-block (spk, cut) chain.
    /// `chain[i]` is the producer/cut of selected-chain index `i+1` (indices are 1-based).
    fn brute_windowed(chain: &[(ScriptPublicKey, u64)], target_spk: &ScriptPublicKey, b: u64, w: u64) -> u64 {
        let lo = if b > w { b - w + 1 } else { 1 };
        let mut sum = 0u64;
        for idx in lo..=b {
            let (s, c) = &chain[(idx - 1) as usize];
            if s == target_spk {
                sum += c;
            }
        }
        sum
    }

    /// Drive the store forward one block at a time (the common, no-reorg path) and assert every
    /// windowed query matches the brute-force oracle for every SPK at every tip.
    #[test]
    fn forward_matches_brute_force() {
        let (_lt, db) = create_temp_db!(ConnBuilder::default().with_files_limit(10));
        let store = DbWindowedProductionPrefixStore::new(db.clone());
        let w = 5u64;
        // A deliberately uneven production pattern across 3 SPKs over 20 blocks.
        let chain: Vec<(ScriptPublicKey, u64)> = (1..=20u64)
            .map(|i| (spk((i % 3) as u8 + 1), (i * 7 % 11) + 1))
            .collect();

        for (k, (s, c)) in chain.iter().enumerate() {
            let index = k as u64 + 1;
            let mut batch = WriteBatch::default();
            // Forward extension: common = previous tip = index-1, one addition.
            store.extend(&mut batch, index - 1, &[], &[(s.clone(), index, *c)]).unwrap();
            db.write(batch).unwrap();

            // Verify against brute force for all SPKs at this tip.
            for sb in 1..=3u8 {
                let q = spk(sb);
                let got = store.windowed(&q, index, w).unwrap();
                let want = brute_windowed(&chain[..index as usize], &q, index, w);
                assert_eq!(got, want, "spk {sb} at tip {index} (w={w})");
            }
        }
    }

    /// A reorg: build a chain, then replace its top with a different suffix. The result must equal the
    /// brute force over the NEW chain — proving truncation + re-append is exact.
    #[test]
    fn reorg_matches_brute_force() {
        let (_lt, db) = create_temp_db!(ConnBuilder::default().with_files_limit(10));
        let store = DbWindowedProductionPrefixStore::new(db.clone());
        let w = 4u64;

        // Original chain, indices 1..=10.
        let mut chain: Vec<(ScriptPublicKey, u64)> = (1..=10u64).map(|i| (spk((i % 2) as u8 + 1), i + 1)).collect();
        for (k, (s, c)) in chain.iter().enumerate() {
            let index = k as u64 + 1;
            let mut batch = WriteBatch::default();
            store.extend(&mut batch, index - 1, &[], &[(s.clone(), index, *c)]).unwrap();
            db.write(batch).unwrap();
        }

        // Reorg: split at common = 6, remove indices 7..=10, add a different suffix 7..=11.
        let common = 6u64;
        let removals: Vec<(ScriptPublicKey, u64)> =
            (7..=10u64).map(|i| (chain[(i - 1) as usize].0.clone(), i)).collect();
        let new_suffix: Vec<(ScriptPublicKey, u64)> =
            vec![(spk(3), 100), (spk(3), 200), (spk(1), 300), (spk(3), 400), (spk(2), 500)];
        let additions: Vec<(ScriptPublicKey, u64, u64)> =
            new_suffix.iter().enumerate().map(|(j, (s, c))| (s.clone(), common + 1 + j as u64, *c)).collect();

        let mut batch = WriteBatch::default();
        store.extend(&mut batch, common, &removals, &additions).unwrap();
        db.write(batch).unwrap();

        // Build the expected new chain (indices 1..=11) for the oracle.
        chain.truncate(common as usize);
        chain.extend(new_suffix);
        let tip = chain.len() as u64;
        for sb in 1..=3u8 {
            let q = spk(sb);
            for b in 1..=tip {
                let got = store.windowed(&q, b, w).unwrap();
                let want = brute_windowed(&chain, &q, b, w);
                assert_eq!(got, want, "post-reorg spk {sb} at b {b}");
            }
        }
    }

    /// Pruning: collapse the entries below a floor into the per-SPK baseline and assert `cumulative_at`
    /// (hence `windowed`) is unchanged for queries at/above the floor — the whole point of the floor.
    #[test]
    fn floor_collapse_preserves_queries() {
        let (_lt, db) = create_temp_db!(ConnBuilder::default().with_files_limit(10));
        let store = DbWindowedProductionPrefixStore::new(db.clone());
        let w = 6u64;

        // SPK 1 produces early then goes quiet; SPK 2 produces late — the exact case the floor exists
        // for (cum(b−W) of SPK 1 must survive even after its early entries are pruned).
        let chain: Vec<(ScriptPublicKey, u64)> = vec![
            (spk(1), 10), (spk(1), 20), (spk(2), 5), (spk(2), 5), (spk(1), 30),
            (spk(2), 7), (spk(2), 7), (spk(2), 7), (spk(2), 7), (spk(2), 7),
            (spk(2), 7), (spk(2), 7), (spk(2), 7), (spk(2), 7), (spk(2), 7),
        ];
        for (k, (s, c)) in chain.iter().enumerate() {
            let index = k as u64 + 1;
            let mut batch = WriteBatch::default();
            store.extend(&mut batch, index - 1, &[], &[(s.clone(), index, *c)]).unwrap();
            db.write(batch).unwrap();
        }
        let tip = chain.len() as u64;

        // Precondition the floor relies on (guaranteed in production by `W < pruning_depth`): the
        // window bottom `b − W` never reaches below the highest collapsed index. We collapse indices
        // 1..=5, so valid queries need `b − W ≥ 5`, i.e. `b ≥ 5 + W`. (Querying further back would ask
        // for a cumulative finer-grained than the floor retains — which never happens at the live tip.)
        let highest_collapsed = 5u64;
        let min_b = highest_collapsed + w;

        // Snapshot queries before pruning.
        let before: Vec<(u8, u64, u64)> = (1..=2u8)
            .flat_map(|sb| (min_b..=tip).map(move |b| (sb, b)))
            .map(|(sb, b)| (sb, b, store.windowed(&spk(sb), b, w).unwrap()))
            .collect();

        // Prune everything below floor index 6: collapse entries at indices 1..=5 into the floor.
        let floor = 6u64;
        let mut cum_so_far: HashMap<ScriptPublicKey, u64> = HashMap::new();
        let items: Vec<(ScriptPublicKey, u64, u64)> = (1..=5u64)
            .map(|index| {
                let (s, c) = &chain[(index - 1) as usize];
                let e = cum_so_far.entry(s.clone()).or_insert(0);
                *e += c;
                (s.clone(), index, *e)
            })
            .collect();
        let mut batch = WriteBatch::default();
        store.collapse_to_floor(&mut batch, &items).unwrap();
        db.write(batch).unwrap();

        // Every query at/above the floor must be identical to before the collapse.
        for (sb, b, want) in before {
            let got = store.windowed(&spk(sb), b, w).unwrap();
            assert_eq!(got, want, "post-collapse spk {sb} at b {b} (floor {floor})");
        }
    }

    /// `collapse_below` (the wired GC path) must preserve every at/above-floor query — and do so
    /// after EACH budgeted partial pass, not just at the end (the ascending delete-lowest-first
    /// invariant is what makes an intermediate commit safe).
    #[test]
    fn collapse_below_preserves_queries() {
        let (_lt, db) = create_temp_db!(ConnBuilder::default().with_files_limit(10));
        let store = DbWindowedProductionPrefixStore::new(db.clone());
        let w = 6u64;
        let chain: Vec<(ScriptPublicKey, u64)> = vec![
            (spk(1), 10), (spk(1), 20), (spk(2), 5), (spk(2), 5), (spk(1), 30),
            (spk(2), 7), (spk(2), 7), (spk(2), 7), (spk(2), 7), (spk(2), 7),
            (spk(2), 7), (spk(2), 7), (spk(2), 7), (spk(2), 7), (spk(2), 7),
        ];
        for (k, (s, c)) in chain.iter().enumerate() {
            let index = k as u64 + 1;
            let mut batch = WriteBatch::default();
            store.extend(&mut batch, index - 1, &[], &[(s.clone(), index, *c)]).unwrap();
            db.write(batch).unwrap();
        }
        let tip = chain.len() as u64;
        // Collapse indices < 6; valid queries need `b − w ≥ 5`, i.e. `b ≥ 6 + w` (the production
        // invariant `floor_index = pruning_idx − W`, `min_b = pruning_idx`).
        let floor_index = 6u64;
        let min_b = floor_index + w;

        let before: Vec<(u8, u64, u64)> = (1..=2u8)
            .flat_map(|sb| (min_b..=tip).map(move |b| (sb, b)))
            .map(|(sb, b)| (sb, b, store.windowed(&spk(sb), b, w).unwrap()))
            .collect();

        // Tiny budget forces multiple partial passes (incl. mid-bucket cuts).
        loop {
            let mut batch = WriteBatch::default();
            let n = store.collapse_below(&mut batch, floor_index, 2).unwrap();
            db.write(batch).unwrap();
            for (sb, b, want) in &before {
                assert_eq!(store.windowed(&spk(*sb), *b, w).unwrap(), *want, "post partial-collapse spk {sb} b {b}");
            }
            if n == 0 {
                break;
            }
        }
    }

    /// Cross-SPK isolation: a reverse seek for one SPK must never read a neighbouring SPK's entry.
    #[test]
    fn cross_spk_isolation() {
        let (_lt, db) = create_temp_db!(ConnBuilder::default().with_files_limit(10));
        let store = DbWindowedProductionPrefixStore::new(db.clone());
        let mut batch = WriteBatch::default();
        // Two SPKs whose buckets are adjacent; only spk(2) has an entry at index 5.
        store.put_cumulative(&mut batch, &spk(2), 5, 999);
        db.write(batch).unwrap();
        // spk(1) and spk(3) have nothing ⇒ cumulative 0 at any index, not spk(2)'s 999.
        assert_eq!(store.cumulative_at(&spk(1), 10).unwrap(), 0);
        assert_eq!(store.cumulative_at(&spk(3), 10).unwrap(), 0);
        assert_eq!(store.cumulative_at(&spk(2), 10).unwrap(), 999);
        assert_eq!(store.cumulative_at(&spk(2), 4).unwrap(), 0); // before the entry
    }
}
