//! Assignment of the network-model ranges to escrow keys, one table per finality epoch.

use keryx_hashes::Hash;
use std::collections::BTreeMap;

/// Tables a ledger snapshot carries: the one computed at its sample and the two before it.
pub const RANGE_TABLES_KEPT: u64 = 3;

/// Load ratio between the heaviest and the lightest range above which a key is moved.
const REBALANCE_RATIO: u64 = 2;

/// Escrow key → range, keys strictly ascending.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RangeTable(Vec<(Hash, u8)>);

impl RangeTable {
    /// Builds a table from entries with strictly ascending keys.
    pub fn from_entries(entries: Vec<(Hash, u8)>) -> Result<Self, &'static str> {
        if entries.windows(2).any(|w| w[0].0 >= w[1].0) {
            return Err("range table keys out of order");
        }
        Ok(Self(entries))
    }

    /// The `(key, range)` entries, ascending keys.
    pub fn entries(&self) -> &[(Hash, u8)] {
        &self.0
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Range held by `key`, if listed.
    pub fn get(&self, key: &Hash) -> Option<u8> {
        self.0.binary_search_by(|(k, _)| k.cmp(key)).ok().map(|i| self.0[i].1)
    }

    /// Range `key` must mine under this table: its entry, else its home range.
    pub fn range_of(&self, key: &Hash, n_ranges: usize) -> u8 {
        self.get(key).unwrap_or_else(|| home_range(key, n_ranges))
    }
}

/// Epoch-independent range of a key absent from the table.
pub fn home_range(key: &Hash, n_ranges: usize) -> u8 {
    let mut h = blake3::Hasher::new();
    h.update(b"KeryxHomeRangeV1");
    h.update(&key.as_bytes());
    let word = u64::from_le_bytes(h.finalize().as_bytes()[..8].try_into().unwrap());
    (word % n_ranges.max(1) as u64) as u8
}

/// Table of the next epoch: every key with a range-tier blue in `producers` (one item per blue)
/// keeps its range under `prev` (home range if new), weighted by its blue count; while a range is
/// empty or out of ratio, the key of the heaviest range whose weight is closest to half the spread
/// moves to the lightest one.
pub fn next_range_table(prev: &RangeTable, producers: impl IntoIterator<Item = Hash>, n_ranges: usize) -> RangeTable {
    let n = n_ranges.max(1);
    let mut weight: BTreeMap<Hash, u64> = BTreeMap::new();
    for key in producers {
        *weight.entry(key).or_default() += 1;
    }
    let mut assign: BTreeMap<Hash, u8> = weight.keys().map(|k| (*k, prev.range_of(k, n).min(n as u8 - 1))).collect();
    let mut load = vec![0u64; n];
    for (k, r) in assign.iter() {
        load[*r as usize] += weight[k];
    }
    for _ in 0..(assign.len() * n + 1) {
        let hi = (0..n).max_by(|a, b| load[*a].cmp(&load[*b]).then(b.cmp(a))).unwrap();
        let lo = (0..n).min_by(|a, b| load[*a].cmp(&load[*b]).then(a.cmp(b))).unwrap();
        let spread = load[hi] - load[lo];
        if hi == lo || !(load[lo] == 0 || load[hi] > REBALANCE_RATIO * load[lo]) {
            break;
        }
        let candidate = assign
            .iter()
            .filter(|(_, r)| **r as usize == hi)
            .map(|(k, _)| (weight[k], *k))
            .filter(|(w, _)| *w < spread)
            .min_by_key(|(w, k)| ((2 * w).abs_diff(spread), *w, *k));
        let Some((w, key)) = candidate else { break };
        assign.insert(key, lo as u8);
        load[hi] -= w;
        load[lo] += w;
    }
    RangeTable(assign.into_iter().collect())
}

/// Epoch of the table judging a blue of epoch `blue_epoch` merged by a chain block of epoch
/// `chain_epoch`: the blue's own previous epoch, never older than two epochs before the chain block.
pub fn judging_table_epoch(blue_epoch: u64, chain_epoch: u64) -> Option<u64> {
    let e = blue_epoch.min(chain_epoch).max(chain_epoch.saturating_sub(1));
    e.checked_sub(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(b: u8) -> Hash {
        Hash::from_bytes([b; 32])
    }

    fn blues(spec: &[(u8, u64)]) -> Vec<Hash> {
        spec.iter().flat_map(|(b, n)| std::iter::repeat_n(key(*b), *n as usize)).collect()
    }

    fn loads(t: &RangeTable, w: &[(u8, u64)], n: usize) -> Vec<u64> {
        let mut l = vec![0; n];
        for (b, c) in w {
            l[t.get(&key(*b)).unwrap() as usize] += c;
        }
        l
    }

    #[test]
    fn home_range_is_stable_and_spread() {
        let mut seen = [0u32; 4];
        for b in 0..=255u8 {
            let r = home_range(&key(b), 4);
            assert_eq!(home_range(&key(b), 4), r);
            seen[r as usize] += 1;
        }
        assert!(seen.iter().all(|c| *c > 32), "{seen:?}");
        assert_eq!(home_range(&key(7), 1), 0);
    }

    #[test]
    fn empty_ranges_get_filled_from_four_keys() {
        let w = [(1, 10), (2, 10), (3, 10), (4, 10)];
        let t = next_range_table(&RangeTable::default(), blues(&w), 4);
        assert_eq!(t.entries().len(), 4);
        assert_eq!(loads(&t, &w, 4), vec![10, 10, 10, 10]);
    }

    #[test]
    fn keys_keep_their_range_when_balanced() {
        let prev = RangeTable::from_entries(vec![(key(1), 3), (key(2), 2), (key(3), 1), (key(4), 0)]).unwrap();
        let t = next_range_table(&prev, blues(&[(1, 12), (2, 9), (3, 7), (4, 10)]), 4);
        assert_eq!(t, prev);
    }

    #[test]
    fn an_out_of_ratio_range_is_rebalanced_with_few_moves() {
        let prev = RangeTable::from_entries(vec![(key(1), 0), (key(2), 0), (key(3), 1), (key(4), 2), (key(5), 3)]).unwrap();
        let w = [(1, 20), (2, 15), (3, 10), (4, 10), (5, 10)];
        let t = next_range_table(&prev, blues(&w), 4);
        let l = loads(&t, &w, 4);
        assert!(*l.iter().max().unwrap() <= 2 * *l.iter().min().unwrap(), "{l:?}");
        assert_eq!(t.get(&key(1)), Some(0));
        let moved = t.entries().iter().filter(|(k, r)| prev.get(k) != Some(*r)).count();
        assert!(moved <= 2, "{t:?}");
    }

    #[test]
    fn a_single_key_never_moves_onto_an_empty_range() {
        let prev = RangeTable::from_entries(vec![(key(1), 2)]).unwrap();
        let t = next_range_table(&prev, blues(&[(1, 50)]), 4);
        assert_eq!(t.get(&key(1)), Some(2));
    }

    #[test]
    fn absent_keys_leave_and_new_keys_start_home() {
        let prev = RangeTable::from_entries(vec![(key(1), 0), (key(2), 1)]).unwrap();
        let t = next_range_table(&prev, blues(&[(1, 5), (9, 5)]), 1);
        assert_eq!(t.get(&key(2)), None);
        assert_eq!(t.get(&key(9)), Some(home_range(&key(9), 1)));
    }

    #[test]
    fn rebalance_terminates_with_many_small_keys() {
        let w: Vec<(u8, u64)> = (0..200u8).map(|b| (b, 1 + (b as u64 % 7))).collect();
        let t = next_range_table(&RangeTable::default(), blues(&w), 4);
        let l = loads(&t, &w, 4);
        let (mx, mn) = (*l.iter().max().unwrap(), *l.iter().min().unwrap());
        assert!(mn > 0 && mx <= 2 * mn, "{l:?}");
    }

    #[test]
    fn judging_epoch_follows_the_blue_within_one_epoch() {
        assert_eq!(judging_table_epoch(10, 10), Some(9));
        assert_eq!(judging_table_epoch(9, 10), Some(8));
        assert_eq!(judging_table_epoch(5, 10), Some(8));
        assert_eq!(judging_table_epoch(11, 10), Some(9));
        assert_eq!(judging_table_epoch(0, 0), None);
    }

    #[test]
    fn unordered_entries_are_refused() {
        assert!(RangeTable::from_entries(vec![(key(2), 0), (key(1), 0)]).is_err());
        assert!(RangeTable::from_entries(vec![(key(1), 0), (key(1), 1)]).is_err());
    }
}
