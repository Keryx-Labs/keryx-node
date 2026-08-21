use std::sync::Arc;

use itertools::Itertools;
use keryx_consensus_core::pruning::PruningPointProof;
use keryx_database::prelude::DB;
use keryx_database::prelude::StoreResult;
use keryx_database::prelude::{BatchDbWriter, CachedDbItem, DirectDbWriter};
use keryx_database::registry::DatabaseStorePrefixes;
use keryx_hashes::Hash;
use keryx_hashes::ZERO_HASH;
use rocksdb::WriteBatch;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Serialize, Deserialize)]
struct PruningPointInfo {
    pruning_point: Hash,
    _candidate: Hash, // Obsolete field. Kept only for avoiding the DB upgrade logic. TODO: remove all together
    index: u64,
}

impl PruningPointInfo {
    pub fn new(pruning_point: Hash, index: u64) -> Self {
        Self { pruning_point, _candidate: ZERO_HASH, index }
    }

    pub fn decompose(self) -> (Hash, u64) {
        (self.pruning_point, self.index)
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct PruningProofDescriptor {
    /// The pruning point associated with this proof descriptor
    pub(crate) pruning_point: Hash,
    /// Indicates whether this descriptor was received from an external source (IBD) or was built locally
    pub(crate) external: bool,
    /// The per-level tips
    pub(crate) tips: Vec<Hash>,
    /// The per-level roots
    pub(crate) roots: Vec<Hash>,
    /// The per-level header counts (used to sanity check loading logic)
    pub(crate) counts: Vec<u64>,
}

impl PruningProofDescriptor {
    pub fn new(pruning_point: Hash, tips: Vec<Hash>, roots: Vec<Hash>, counts: Vec<u64>) -> Self {
        Self { pruning_point, external: false, tips, roots, counts }
    }

    pub(crate) fn from_proof(proof: &PruningPointProof, pruning_point: Hash, external: bool) -> Self {
        let (tips, roots, counts) = proof
            .iter()
            .map(|level| (level.last().expect("validated").hash, level.first().expect("validated").hash, level.len() as u64))
            .multiunzip();
        let desc = Self { pruning_point, external, tips, roots, counts };
        assert_eq!(desc.tips[0], pruning_point);
        desc
    }
}

/// Restart-persistent ordered membership for a locally-built pruning proof.
///
/// This is deliberately stored separately from [`PruningProofDescriptor`] so existing descriptor
/// records remain binary-compatible. The descriptor is still the authority for pruning-point,
/// root, tip, and count matching; this item only avoids rediscovering the same ordered hashes.
#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct PruningProofHashIndex {
    format_version: u8,
    pruning_point: Hash,
    levels: Vec<Vec<Hash>>,
}

impl PruningProofHashIndex {
    const FORMAT_VERSION: u8 = 1;

    pub(crate) fn from_proof(proof: &PruningPointProof, descriptor: &PruningProofDescriptor) -> Self {
        assert!(!descriptor.external, "external proof descriptors must not be persisted as local hash indexes");
        Self {
            format_version: Self::FORMAT_VERSION,
            pruning_point: descriptor.pruning_point,
            levels: proof.iter().map(|level| level.iter().map(|header| header.hash).collect()).collect(),
        }
    }

    pub(crate) fn matches_descriptor(&self, descriptor: &PruningProofDescriptor) -> bool {
        if self.format_version != Self::FORMAT_VERSION
            || descriptor.external
            || self.pruning_point != descriptor.pruning_point
            || self.levels.len() != descriptor.counts.len()
            || descriptor.counts.len() != descriptor.roots.len()
            || descriptor.roots.len() != descriptor.tips.len()
        {
            return false;
        }

        self.levels.iter().enumerate().all(|(level, hashes)| {
            hashes.len() as u64 == descriptor.counts[level]
                && hashes.first() == Some(&descriptor.roots[level])
                && hashes.last() == Some(&descriptor.tips[level])
        })
    }

    pub(crate) fn levels(&self) -> &[Vec<Hash>] {
        &self.levels
    }
}

/// Reader API for `PruningStore`.
pub trait PruningStoreReader {
    fn pruning_point(&self) -> StoreResult<Hash>;
    fn pruning_point_index(&self) -> StoreResult<u64>;

    /// Returns the pruning point and its index
    fn pruning_point_and_index(&self) -> StoreResult<(Hash, u64)>;

    /// Represent the point after which data is fully held (i.e., history is consecutive from this point and up to virtual).
    /// This is usually a pruning point that is at or below the retention period requirement (and for archival
    /// nodes it will remain the initial syncing point or the last pruning point before turning to an archive).
    /// At every pruning point movement, this is adjusted to the next pruning point sample that satisfies the required
    /// retention period.
    fn retention_period_root(&self) -> StoreResult<Hash>;

    // During pruning, this is a reference to the retention root before the pruning point move.
    // After pruning, this is updated to point to the retention period root.
    // This checkpoint is used to determine if pruning has successfully completed.
    fn retention_checkpoint(&self) -> StoreResult<Hash>;

    /// Returns a compact descriptor of the pruning proof.
    ///
    /// The descriptor contains succinct, per-level metadata sufficient to guide
    /// reconstruction of the full proof from other consensus stores.
    ///
    /// The returned descriptor may lag behind the current pruning point.
    fn pruning_proof_descriptor(&self) -> StoreResult<Arc<PruningProofDescriptor>>;
}

pub trait PruningStore: PruningStoreReader {
    fn set(&mut self, pruning_point: Hash, index: u64) -> StoreResult<()>;
}

/// A DB + cache implementation of `PruningStore` trait, with concurrent readers support.
#[derive(Clone)]
pub struct DbPruningStore {
    db: Arc<DB>,
    access: CachedDbItem<PruningPointInfo>,
    retention_checkpoint_access: CachedDbItem<Hash>,
    retention_period_root_access: CachedDbItem<Hash>,
    pruning_proof_descriptor_access: CachedDbItem<Arc<PruningProofDescriptor>>,
    pruning_proof_hash_index_access: CachedDbItem<Arc<PruningProofHashIndex>>,
}

impl DbPruningStore {
    pub fn new(db: Arc<DB>) -> Self {
        Self {
            db: Arc::clone(&db),
            access: CachedDbItem::new(db.clone(), DatabaseStorePrefixes::PruningPoint.into()),
            retention_checkpoint_access: CachedDbItem::new(db.clone(), DatabaseStorePrefixes::RetentionCheckpoint.into()),
            retention_period_root_access: CachedDbItem::new(db.clone(), DatabaseStorePrefixes::RetentionPeriodRoot.into()),
            pruning_proof_descriptor_access: CachedDbItem::new(db.clone(), DatabaseStorePrefixes::PruningProofDescriptor.into()),
            pruning_proof_hash_index_access: CachedDbItem::new(db, DatabaseStorePrefixes::PruningProofHashIndex.into()),
        }
    }

    pub fn clone_with_new_cache(&self) -> Self {
        Self::new(Arc::clone(&self.db))
    }

    pub fn set_batch(&mut self, batch: &mut WriteBatch, pruning_point: Hash, index: u64) -> StoreResult<()> {
        self.access.write(BatchDbWriter::new(batch), &PruningPointInfo::new(pruning_point, index))
    }

    pub fn set_retention_checkpoint(&mut self, batch: &mut WriteBatch, retention_checkpoint: Hash) -> StoreResult<()> {
        self.retention_checkpoint_access.write(BatchDbWriter::new(batch), &retention_checkpoint)
    }

    pub fn set_retention_period_root(&mut self, batch: &mut WriteBatch, retention_period_root: Hash) -> StoreResult<()> {
        self.retention_period_root_access.write(BatchDbWriter::new(batch), &retention_period_root)
    }

    pub fn set_pruning_proof_descriptor(&mut self, descriptor: PruningProofDescriptor) -> StoreResult<()> {
        self.pruning_proof_descriptor_access.write(DirectDbWriter::new(&self.db), &Arc::new(descriptor))
    }

    pub(crate) fn set_pruning_proof_hash_index(&mut self, index: PruningProofHashIndex) -> StoreResult<()> {
        self.pruning_proof_hash_index_access.write(DirectDbWriter::new(&self.db), &Arc::new(index))
    }

    pub(crate) fn pruning_proof_hash_index(&self) -> StoreResult<Arc<PruningProofHashIndex>> {
        self.pruning_proof_hash_index_access.read()
    }
}

impl PruningStoreReader for DbPruningStore {
    fn pruning_point(&self) -> StoreResult<Hash> {
        Ok(self.access.read()?.pruning_point)
    }

    fn pruning_point_index(&self) -> StoreResult<u64> {
        Ok(self.access.read()?.index)
    }

    fn pruning_point_and_index(&self) -> StoreResult<(Hash, u64)> {
        Ok(self.access.read()?.decompose())
    }

    fn retention_checkpoint(&self) -> StoreResult<Hash> {
        self.retention_checkpoint_access.read()
    }

    fn retention_period_root(&self) -> StoreResult<Hash> {
        self.retention_period_root_access.read()
    }

    fn pruning_proof_descriptor(&self) -> StoreResult<Arc<PruningProofDescriptor>> {
        self.pruning_proof_descriptor_access.read()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use keryx_consensus_core::header::Header;

    fn proof_for_hashes(levels: &[Vec<Hash>]) -> PruningPointProof {
        levels
            .iter()
            .map(|level| level.iter().copied().map(|hash| Arc::new(Header::from_precomputed_hash(hash, vec![]))).collect())
            .collect()
    }

    #[test]
    fn local_hash_index_requires_an_exact_descriptor_match() {
        let levels = vec![vec![Hash::from_u64_word(1), Hash::from_u64_word(2)], vec![Hash::from_u64_word(3)]];
        let proof = proof_for_hashes(&levels);
        let descriptor = PruningProofDescriptor::from_proof(&proof, levels[0][1], false);
        let index = PruningProofHashIndex::from_proof(&proof, &descriptor);
        assert!(index.matches_descriptor(&descriptor));

        let mut changed_count = descriptor.clone();
        changed_count.counts[0] += 1;
        assert!(!index.matches_descriptor(&changed_count));

        let mut changed_root = descriptor.clone();
        changed_root.roots[0] = Hash::from_u64_word(4);
        assert!(!index.matches_descriptor(&changed_root));

        let mut external = descriptor;
        external.external = true;
        assert!(!index.matches_descriptor(&external));
    }
}

impl PruningStore for DbPruningStore {
    fn set(&mut self, pruning_point: Hash, index: u64) -> StoreResult<()> {
        self.access.write(DirectDbWriter::new(&self.db), &PruningPointInfo::new(pruning_point, index))
    }
}
