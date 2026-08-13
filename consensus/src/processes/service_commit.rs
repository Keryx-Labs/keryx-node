/// Sealed service-state commitment index.
///
/// A MuHash accumulator over every finality-flushed service-bond row (escrow-burn rows and
/// strike-log rows), snapshotted after each flushed event daa. `commitment_at(daa)` is the
/// accumulator finalized over exactly the rows with event daa at or below `daa` — the value
/// `Header::service_state_hash` commits for the header's pruning point. Rows are only ever
/// added (finality-deep, reorg-immune), so the index is a pure function of the two service
/// stores and is rebuilt from them at boot.
use keryx_hashes::Hash;
use keryx_muhash::MuHash;
use parking_lot::RwLock;

/// Canonical byte form of an escrow-burn row. Also the unit a service-state transfer ships.
pub fn burn_row_bytes(txid: Hash, index: u32, daa: u64) -> [u8; 45] {
    let mut bytes = [0u8; 45];
    bytes[0] = 0x01;
    bytes[1..33].copy_from_slice(&txid.as_bytes());
    bytes[33..37].copy_from_slice(&index.to_le_bytes());
    bytes[37..45].copy_from_slice(&daa.to_le_bytes());
    bytes
}

/// Canonical byte form of a strike-log row.
pub fn strike_row_bytes(daa: u64, miner: Hash, count: u32, last_daa: u64) -> [u8; 53] {
    let mut bytes = [0u8; 53];
    bytes[0] = 0x02;
    bytes[1..9].copy_from_slice(&daa.to_le_bytes());
    bytes[9..41].copy_from_slice(&miner.as_bytes());
    bytes[41..45].copy_from_slice(&count.to_le_bytes());
    bytes[45..53].copy_from_slice(&last_daa.to_le_bytes());
    bytes
}

struct Inner {
    acc: MuHash,
    /// Finalized commitment after all rows with event daa at or below the key.
    snapshots: std::collections::BTreeMap<u64, Hash>,
}

pub struct ServiceCommitIndex {
    inner: RwLock<Inner>,
}

impl Default for ServiceCommitIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl ServiceCommitIndex {
    pub fn new() -> Self {
        Self { inner: RwLock::new(Inner { acc: MuHash::new(), snapshots: Default::default() }) }
    }

    /// Adds one canonical row to the accumulator. Rows of one event daa must all be added
    /// before that daa is sealed; MuHash is commutative, so their order is free.
    pub fn add_row(&self, row: &[u8]) {
        self.inner.write().acc.add_element(row);
    }

    /// Seals the accumulator state as the commitment for `daa` (all rows at or below it added).
    pub fn seal(&self, daa: u64) {
        let mut inner = self.inner.write();
        let value = inner.acc.finalize();
        inner.snapshots.insert(daa, value);
    }

    /// The sealed commitment as of `daa`: the snapshot at the greatest event daa at or below
    /// it, or the zero hash when no event has ever been sealed that deep.
    pub fn commitment_at(&self, daa: u64) -> Hash {
        self.inner.read().snapshots.range(..=daa).next_back().map(|(_, h)| *h).unwrap_or_default()
    }

    /// Rebuilds the index from the full row set, `rows` being `(event daa, canonical bytes)` in
    /// any order. Pure function of the set: sorting and per-daa sealing happen here.
    pub fn rebuild(&self, mut rows: Vec<(u64, Vec<u8>)>) {
        rows.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        let mut inner = self.inner.write();
        inner.acc = MuHash::new();
        inner.snapshots.clear();
        let mut i = 0;
        while i < rows.len() {
            let daa = rows[i].0;
            while i < rows.len() && rows[i].0 == daa {
                inner.acc.add_element(&rows[i].1);
                i += 1;
            }
            let value = inner.acc.finalize();
            inner.snapshots.insert(daa, value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn incremental_and_rebuilt_indexes_agree() {
        let a = Hash::from_bytes([1u8; 32]);
        let b = Hash::from_bytes([2u8; 32]);
        let rows: Vec<(u64, Vec<u8>)> = vec![
            (100, burn_row_bytes(a, 1, 100).to_vec()),
            (100, strike_row_bytes(100, b, 1, 100).to_vec()),
            (250, strike_row_bytes(250, b, 2, 250).to_vec()),
        ];

        let inc = ServiceCommitIndex::new();
        inc.add_row(&rows[0].1);
        inc.add_row(&rows[1].1);
        inc.seal(100);
        inc.add_row(&rows[2].1);
        inc.seal(250);

        // rebuild from the same set, deliberately out of order
        let rebuilt = ServiceCommitIndex::new();
        rebuilt.rebuild(vec![rows[2].clone(), rows[0].clone(), rows[1].clone()]);

        assert_eq!(inc.commitment_at(99), Hash::default());
        assert_ne!(inc.commitment_at(100), Hash::default());
        assert_eq!(inc.commitment_at(100), rebuilt.commitment_at(100));
        assert_eq!(inc.commitment_at(180), inc.commitment_at(100), "between events the last seal holds");
        assert_eq!(inc.commitment_at(250), rebuilt.commitment_at(250));
        assert_eq!(inc.commitment_at(u64::MAX), rebuilt.commitment_at(250));
        assert_ne!(inc.commitment_at(100), inc.commitment_at(250));
    }

    #[test]
    fn row_order_within_a_daa_is_free() {
        let a = Hash::from_bytes([7u8; 32]);
        let r1 = burn_row_bytes(a, 0, 50).to_vec();
        let r2 = burn_row_bytes(a, 1, 50).to_vec();

        let x = ServiceCommitIndex::new();
        x.add_row(&r1);
        x.add_row(&r2);
        x.seal(50);

        let y = ServiceCommitIndex::new();
        y.add_row(&r2);
        y.add_row(&r1);
        y.seal(50);

        assert_eq!(x.commitment_at(50), y.commitment_at(50));
    }
}
