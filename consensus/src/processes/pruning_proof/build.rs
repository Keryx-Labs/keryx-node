use std::{
    cell::Cell,
    cmp::Reverse,
    collections::{BinaryHeap, VecDeque},
    ops::DerefMut,
    sync::Arc,
    time::Instant,
};

use itertools::Itertools;
use keryx_consensus_core::{
    BlockHashMap, BlockHashSet, BlockLevel, HashMapCustomHasher, KType,
    blockhash::{BlockHashes, ORIGIN},
    header::Header,
    pruning::PruningPointProof,
};
use keryx_core::{debug, info, trace, warn};
use keryx_database::prelude::*;
use keryx_hashes::Hash;
use keryx_utils::binary_heap::TopK;
use parking_lot::RwLock;

use crate::{
    model::{
        services::reachability::ReachabilityService,
        stores::{
            ghostdag::{DbGhostdagStore, GhostdagStore, GhostdagStoreReader},
            headers::{HeaderStoreReader, HeaderWithBlockLevel},
            pruning::{PruningProofDescriptor, PruningProofHashIndex, PruningStoreReader},
            reachability::{DbReachabilityStore, ReachabilityStoreReader},
            relations::{DbRelationsStore, RelationsStoreReader},
        },
    },
    processes::{
        ghostdag::{ordering::SortableBlock, protocol::GhostdagManager},
        pruning_proof::{GhostdagReaderExt, ProofInternalError},
        reachability::inquirer as reachability,
        relations::RelationsStoreExtensions,
    },
};

use super::{ProofInternalResult, PruningProofManager};
use crate::model::services::reachability::MTReachabilityService;

#[derive(Clone)]
struct LevelProofContext {
    ghostdag_store: Arc<DbGhostdagStore>,
    tip: Hash,
    root: Hash,
    count: u64,
}

/// A relations-store reader restricted to the future of a fixed root block (including the root).
///
/// Only parents and children that lie within the root’s future are exposed.
/// This provides a consistent, root-relative view of relations when operating on
/// proofs or subgraphs confined to that region of the DAG.
#[derive(Clone)]
struct FutureIntersectRelations<T: RelationsStoreReader, U: ReachabilityService> {
    relations_store: T,
    reachability_service: U,
    root: Hash,
}

impl<T: RelationsStoreReader, U: ReachabilityService> FutureIntersectRelations<T, U> {
    fn new(relations_store: T, reachability_service: U, root: Hash) -> Self {
        Self { relations_store, reachability_service, root }
    }
}

impl<T: RelationsStoreReader, U: ReachabilityService> RelationsStoreReader for FutureIntersectRelations<T, U> {
    fn get_parents(&self, hash: Hash) -> Result<BlockHashes, StoreError> {
        self.relations_store.get_parents(hash).map(|hashes| {
            hashes
                .iter()
                .copied()
                .filter(|&h| self.reachability_service.try_is_dag_ancestor_of(self.root, h).optional().unwrap().unwrap_or(false))
                .collect_vec()
                .into()
        })
    }

    fn get_children(&self, hash: Hash) -> StoreResult<ReadLock<BlockHashSet>> {
        assert!(self.reachability_service.is_dag_ancestor_of(self.root, hash), "future(root) invariant violated");
        self.relations_store.get_children(hash)
    }

    fn has(&self, hash: Hash) -> Result<bool, StoreError> {
        Ok(self.relations_store.has(hash)? && self.reachability_service.is_dag_ancestor_of(self.root, hash))
    }

    fn counts(&self) -> Result<(usize, usize), StoreError> {
        unreachable!("not expected to be called in this context")
    }
}

/// Utility for creating retry-indexed temporary ghostdag stores.
///
/// Each call to `new_store` returns a fresh temporary `DbGhostdagStore` for the
/// given level, using an incrementing retry index to avoid namespace collisions.
/// This is used when multiple ghostdag attempts may be required for the same level.
struct GhostdagStoreFactory {
    db: Arc<DB>,
    cache_policy: CachePolicy,
    level: BlockLevel,
    retries: u8,
}

impl GhostdagStoreFactory {
    fn new(db: Arc<DB>, cache_policy: CachePolicy, level: BlockLevel) -> Self {
        Self { db, cache_policy, level, retries: 0 }
    }

    /// Creates a fresh temporary ghostdag store for the next retry attempt
    fn new_store(&mut self) -> Arc<DbGhostdagStore> {
        self.retries += 1;
        Arc::new(DbGhostdagStore::new_temp(self.db.clone(), self.level, self.cache_policy, self.cache_policy, self.retries - 1))
    }
}

/// Utility for creating retry-indexed temporary reachability stores.
struct ReachabilityStoreFactory {
    db: Arc<DB>,
    cache_policy: CachePolicy,
    level: BlockLevel,
    retries: u8,
}

impl ReachabilityStoreFactory {
    fn new(db: Arc<DB>, cache_policy: CachePolicy, level: BlockLevel) -> Self {
        Self { db, cache_policy, level, retries: 0 }
    }

    fn new_store(&mut self) -> Arc<RwLock<DbReachabilityStore>> {
        self.retries += 1;
        Arc::new(RwLock::new(DbReachabilityStore::with_block_level_retry(
            self.db.clone(),
            self.cache_policy,
            self.cache_policy,
            self.level,
            self.retries - 1,
        )))
    }
}

impl PruningProofManager {
    /// Builds a pruning-point proof for `pp` by computing per-level MLS proof contexts and
    /// collecting the headers in `future(root) ∩ past(tip)` for each level.
    /// Temporary stores are used during construction, and headers are shared (via arcs)
    /// across levels in the final proof.
    pub(crate) fn build_pruning_point_proof(&self, pp: Hash) -> PruningPointProof {
        // Sanity rebuilding must remain independent of the persisted materialization cache.
        self.build_pruning_point_proof_inner(pp, super::next_pp_build_diag_id(), false)
    }

    pub(crate) fn build_pruning_point_proof_with_diag_id(&self, pp: Hash, build_id: u64) -> PruningPointProof {
        self.build_pruning_point_proof_inner(pp, build_id, true)
    }

    fn build_pruning_point_proof_inner(&self, pp: Hash, build_id: u64, allow_persisted_index: bool) -> PruningPointProof {
        let total_started = Instant::now();
        info!("PP-BUILD-DIAG id={} stage=build_enter pruning_point={}", build_id, pp);

        let descriptor_lock_started = Instant::now();
        info!("PP-BUILD-DIAG id={} stage=descriptor_lock_wait_start", build_id);
        let pruning_point_read = self.pruning_point_store.read();
        info!(
            "PP-BUILD-DIAG id={} stage=descriptor_lock_acquired elapsed_ms={}",
            build_id,
            descriptor_lock_started.elapsed().as_millis()
        );
        let descriptor = pruning_point_read.pruning_proof_descriptor().optional().unwrap();
        drop(pruning_point_read);
        info!(
            "PP-BUILD-DIAG id={} stage=proof_descriptor_check present={} matches_pruning_point={} external={} elapsed_ms={}",
            build_id,
            descriptor.is_some(),
            descriptor.as_ref().is_some_and(|descriptor| descriptor.pruning_point == pp),
            descriptor.as_ref().is_some_and(|descriptor| descriptor.external),
            total_started.elapsed().as_millis()
        );
        if let Some(descriptor) = descriptor.as_ref() {
            // Use a locally built descriptor (when it matches the current pruning point) for fast reconstruction.
            // Otherwise, recalculate the descriptor to optimize proof size.
            if descriptor.pruning_point == pp && !descriptor.external {
                if allow_persisted_index {
                    let persisted_index = self.pruning_point_store.read().pruning_proof_hash_index().optional();
                    match persisted_index {
                        Ok(Some(index)) if index.matches_descriptor(descriptor) => {
                            let load_started = Instant::now();
                            info!("PP-BUILD-DIAG id={} stage=local_hash_index_load_start", build_id);
                            match self.proof_from_hash_index(index.as_ref()) {
                                Ok(proof) => {
                                    info!(
                                        "PP-BUILD-DIAG id={} stage=local_hash_index_load_complete headers={} elapsed_ms={}",
                                        build_id,
                                        proof.iter().map(|level| level.len()).sum::<usize>(),
                                        load_started.elapsed().as_millis()
                                    );
                                    return proof;
                                }
                                Err(err) => warn!(
                                    "PP-BUILD-DIAG id={} stage=local_hash_index_load_failed error={} fallback=descriptor",
                                    build_id, err
                                ),
                            }
                        }
                        Ok(Some(_)) => info!("PP-BUILD-DIAG id={} stage=local_hash_index_mismatch fallback=descriptor", build_id),
                        Ok(None) => info!("PP-BUILD-DIAG id={} stage=local_hash_index_missing fallback=descriptor", build_id),
                        Err(err) => {
                            warn!("PP-BUILD-DIAG id={} stage=local_hash_index_read_failed error={} fallback=descriptor", build_id, err)
                        }
                    }
                }

                info!("PP-BUILD-DIAG id={} stage=local_descriptor_rebuild_start", build_id);
                let proof = self.proof_from_descriptor(descriptor, build_id, allow_persisted_index);
                if allow_persisted_index {
                    self.persist_local_hash_index(descriptor, &proof, build_id);
                }
                info!(
                    "PP-BUILD-DIAG id={} stage=local_descriptor_rebuild_complete headers={} elapsed_ms={}",
                    build_id,
                    proof.iter().map(|level| level.len()).sum::<usize>(),
                    total_started.elapsed().as_millis()
                );
                return proof;
            }
        }

        let new_descriptor = match pp == self.genesis_hash {
            true => {
                // Genesis case - create a proof where all levels hold genesis only
                let (tips, roots, counts) = (0..=self.max_block_level).map(|_| (self.genesis_hash, self.genesis_hash, 1)).multiunzip();
                PruningProofDescriptor::new(self.genesis_hash, tips, roots, counts)
            }
            false => {
                // General case
                info!("PP-BUILD-DIAG id={} stage=root_calculation_start pruning_point={}", build_id, pp);
                let descriptor = self.calc_new_proof(pp, descriptor.as_deref(), build_id);
                info!(
                    "PP-BUILD-DIAG id={} stage=root_calculation_complete elapsed_ms={}",
                    build_id,
                    total_started.elapsed().as_millis()
                );
                descriptor
            }
        };

        info!("PP-BUILD-DIAG id={} stage=proof_finalization_start", build_id);
        let proof = self.proof_from_descriptor(&new_descriptor, build_id, false);
        info!(
            "PP-BUILD-DIAG id={} stage=proof_finalization_complete headers={} elapsed_ms={}",
            build_id,
            proof.iter().map(|level| level.len()).sum::<usize>(),
            total_started.elapsed().as_millis()
        );

        let descriptor_write_started = Instant::now();
        info!("PP-BUILD-DIAG id={} stage=descriptor_write_lock_wait_start", build_id);
        let mut pruning_point_write = self.pruning_point_store.write();
        info!(
            "PP-BUILD-DIAG id={} stage=descriptor_write_lock_acquired elapsed_ms={}",
            build_id,
            descriptor_write_started.elapsed().as_millis()
        );
        pruning_point_write.set_pruning_proof_descriptor(new_descriptor.clone()).unwrap();
        if allow_persisted_index {
            pruning_point_write.set_pruning_proof_hash_index(PruningProofHashIndex::from_proof(&proof, &new_descriptor)).unwrap();
        }
        drop(pruning_point_write);
        info!("PP-BUILD-DIAG id={} stage=build_complete elapsed_ms={}", build_id, total_started.elapsed().as_millis());

        proof
    }

    /// Reconstructs the pruning proof described by `descriptor` by loading headers from storage
    /// and collecting, per level, the blocks in `future(root) ∩ past(tip)`.
    ///
    /// Uses a local header-arc cache to deduplicate headers shared across levels.
    fn proof_from_descriptor(
        &self,
        descriptor: &PruningProofDescriptor,
        build_id: u64,
        fast_local_level_zero: bool,
    ) -> PruningPointProof {
        let phase_started = Instant::now();
        info!(
            "PP-BUILD-DIAG id={} stage=descriptor_reconstruction_start pruning_point={} levels={} external={}",
            build_id,
            descriptor.pruning_point,
            descriptor.tips.len(),
            descriptor.external
        );
        // The pruning proof can contain many duplicate headers (across levels), so we use a local cache in order
        // to make sure we hold a single Arc per header
        let mut cache: BlockHashMap<Arc<Header>> = BlockHashMap::with_capacity(4 * self.pruning_proof_m as usize);
        let mut get_header = |hash| cache.entry(hash).or_insert_with_key(|&hash| self.headers_store.get_header(hash).unwrap()).clone();

        let temp_db_started = Instant::now();
        info!("PP-BUILD-DIAG id={} stage=temp_db_create_start phase=descriptor_reconstruction", build_id);
        let (_db_lifetime, temp_db) = keryx_database::create_temp_db!(ConnBuilder::default().with_files_limit(10));
        info!(
            "PP-BUILD-DIAG id={} stage=temp_db_create_complete phase=descriptor_reconstruction elapsed_ms={}",
            build_id,
            temp_db_started.elapsed().as_millis()
        );
        let cache_policy = CachePolicy::Count(2 * self.pruning_proof_m as usize);

        let proof: PruningPointProof = (0..=self.max_block_level)
            .map(|level| {
                let level_started = Instant::now();
                let level_idx = level as usize;
                let tip = descriptor.tips[level_idx];
                let root = descriptor.roots[level_idx];
                let expected_count = descriptor.counts[level_idx];
                info!(
                    "PP-BUILD-DIAG id={} stage=descriptor_level_start level={} root={} tip={} expected_headers={}",
                    build_id, level, root, tip, expected_count
                );

                if level == 0 && fast_local_level_zero {
                    return self.proof_level_zero_from_local_relations(root, tip, expected_count, build_id, &mut get_header);
                }

                let reachability_probe_started = Instant::now();
                info!("PP-BUILD-DIAG id={} stage=reachability_lock_wait_start phase=descriptor level={}", build_id, level);
                let reachability_probe = self.reachability_store.read();
                info!(
                    "PP-BUILD-DIAG id={} stage=reachability_lock_acquired phase=descriptor level={} elapsed_ms={}",
                    build_id,
                    level,
                    reachability_probe_started.elapsed().as_millis()
                );
                drop(reachability_probe);

                let mut headers = VecDeque::with_capacity(2 * self.pruning_proof_m as usize);
                let mut relations_store = DbRelationsStore::new_temp(temp_db.clone(), level, 0, cache_policy, cache_policy);

                let mut queue = BinaryHeap::<SortableBlock>::new();
                let mut visited = BlockHashSet::new();
                let mut parent_edges = 0usize;
                queue.push(SortableBlock::new(tip, get_header(tip).blue_work));

                while let Some(SortableBlock { hash: current, .. }) = queue.pop() {
                    if !visited.insert(current) {
                        continue;
                    }
                    if visited.len() % 10_000 == 0 {
                        info!(
                            "PP-BUILD-DIAG id={} stage=descriptor_backward_walk_progress level={} visited={} queued={} parent_edges={} elapsed_ms={}",
                            build_id,
                            level,
                            visited.len(),
                            queue.len(),
                            parent_edges,
                            level_started.elapsed().as_millis()
                        );
                    }

                    // We are only interested in the exact diamond future(root) ⋂ past(tip)
                    if !self.reachability_service.is_dag_ancestor_of(root, current) {
                        continue;
                    }

                    let header = get_header(current);
                    let parents: BlockHashes = self.reachable_parents_at_level(level, &header).collect::<Vec<_>>().into();
                    parent_edges += parents.len();
                    for parent in parents.iter().copied() {
                        queue.push(SortableBlock::new(parent, get_header(parent).blue_work));
                    }
                    relations_store.insert(current, parents).unwrap();
                }

                // Bottom-up traversal from root using the relations collected above
                let mut bottom_up_queue: BinaryHeap<_> = Default::default();
                let mut bottom_up_visited = BlockHashSet::new();
                let mut child_edges = 0usize;
                bottom_up_queue.push(Reverse(SortableBlock::new(root, get_header(root).blue_work)));

                while let Some(Reverse(SortableBlock { hash: current, .. })) = bottom_up_queue.pop() {
                    if !bottom_up_visited.insert(current) {
                        continue;
                    }
                    if bottom_up_visited.len() % 10_000 == 0 {
                        info!(
                            "PP-BUILD-DIAG id={} stage=descriptor_forward_walk_progress level={} visited={} queued={} headers={} child_edges={} elapsed_ms={}",
                            build_id,
                            level,
                            bottom_up_visited.len(),
                            bottom_up_queue.len(),
                            headers.len(),
                            child_edges,
                            level_started.elapsed().as_millis()
                        );
                    }

                    if !self.reachability_service.is_dag_ancestor_of(current, tip) {
                        continue;
                    }

                    headers.push_back(get_header(current));

                    let children = relations_store.get_children(current).unwrap();
                    child_edges += children.read().len();
                    for &child in children.read().iter() {
                        bottom_up_queue.push(Reverse(SortableBlock::new(child, get_header(child).blue_work)));
                    }
                }

                assert_eq!(
                    expected_count,
                    headers.len() as u64,
                    "rebuilt proof level {} count {} does not match the expected descriptor count {}",
                    level,
                    headers.len(),
                    expected_count
                );
                info!(
                    "PP-BUILD-DIAG id={} stage=descriptor_level_complete level={} headers={} backward_visited={} forward_visited={} parent_edges={} child_edges={} elapsed_ms={}",
                    build_id,
                    level,
                    headers.len(),
                    visited.len(),
                    bottom_up_visited.len(),
                    parent_edges,
                    child_edges,
                    level_started.elapsed().as_millis()
                );
                headers.into()
            })
            .collect();
        drop(get_header);
        info!(
            "PP-BUILD-DIAG id={} stage=descriptor_reconstruction_complete headers={} unique_headers={} elapsed_ms={}",
            build_id,
            proof.iter().map(|level| level.len()).sum::<usize>(),
            cache.len(),
            phase_started.elapsed().as_millis()
        );
        proof
    }

    /// Materializes the level-0 descriptor diamond directly from the canonical local relations
    /// store. This produces the same bottom-up `future(root) ∩ past(tip)` ordering as the generic
    /// reconstruction, without first walking the entire diamond backwards to rebuild temporary
    /// parent/child relations.
    fn proof_level_zero_from_local_relations(
        &self,
        root: Hash,
        tip: Hash,
        expected_count: u64,
        build_id: u64,
        get_header: &mut impl FnMut(Hash) -> Arc<Header>,
    ) -> Vec<Arc<Header>> {
        let started = Instant::now();
        let relations_store = self.relations_store.read().clone();
        let mut headers = Vec::with_capacity(expected_count.try_into().unwrap_or(usize::MAX));
        let mut queue = BinaryHeap::new();
        let mut discovered = BlockHashSet::new();
        discovered.insert(root);
        queue.push(Reverse(SortableBlock::new(root, get_header(root).blue_work)));

        while let Some(Reverse(SortableBlock { hash: current, .. })) = queue.pop() {
            headers.push(get_header(current));
            for &child in relations_store.get_children(current).unwrap().read().iter() {
                // A non-ancestor of `tip` cannot have an ancestor-of-`tip` descendant. Check each
                // candidate only once even when it has many parents in the descriptor diamond.
                if discovered.insert(child) && self.reachability_service.is_dag_ancestor_of(child, tip) {
                    queue.push(Reverse(SortableBlock::new(child, get_header(child).blue_work)));
                }
            }
        }

        assert_eq!(
            headers.len() as u64,
            expected_count,
            "fast local descriptor reconstruction count mismatch: expected {}, got {}",
            expected_count,
            headers.len()
        );
        info!(
            "PP-BUILD-DIAG id={} stage=local_descriptor_level_zero_complete headers={} elapsed_ms={}",
            build_id,
            headers.len(),
            started.elapsed().as_millis()
        );
        headers
    }

    fn proof_from_hash_index(&self, index: &PruningProofHashIndex) -> StoreResult<PruningPointProof> {
        const HEADER_LOAD_CHUNK_SIZE: usize = 8_192;
        index
            .levels()
            .iter()
            .map(|level| {
                let mut headers = Vec::with_capacity(level.len());
                for chunk in level.chunks(HEADER_LOAD_CHUNK_SIZE) {
                    headers.extend(self.headers_store.get_headers_many(chunk)?);
                }
                Ok(headers)
            })
            .collect()
    }

    fn persist_local_hash_index(&self, descriptor: &PruningProofDescriptor, proof: &PruningPointProof, build_id: u64) {
        debug_assert!(!descriptor.external);
        let started = Instant::now();
        self.pruning_point_store.write().set_pruning_proof_hash_index(PruningProofHashIndex::from_proof(proof, descriptor)).unwrap();
        info!("PP-BUILD-DIAG id={} stage=local_hash_index_persist_complete elapsed_ms={}", build_id, started.elapsed().as_millis());
    }

    /// Computes level-proof contexts for all levels, processing levels from high to low to satisfy
    /// MLS inter-level constraints, and aggregates the results into a pruning-proof descriptor.
    fn calc_new_proof(&self, pp: Hash, prev_descriptor: Option<&PruningProofDescriptor>, build_id: u64) -> PruningProofDescriptor {
        let calculation_started = Instant::now();
        info!(
            "PP-BUILD-DIAG id={} stage=calculate_descriptor_start pruning_point={} previous_descriptor={} previous_external={}",
            build_id,
            pp,
            prev_descriptor.is_some(),
            prev_descriptor.is_some_and(|descriptor| descriptor.external)
        );
        info!("PP-BUILD-DIAG id={} stage=temp_db_create_start phase=calculate_descriptor", build_id);
        let (_db_lifetime, temp_db) = keryx_database::create_temp_db!(ConnBuilder::default().with_files_limit(10));
        info!(
            "PP-BUILD-DIAG id={} stage=temp_db_create_complete phase=calculate_descriptor elapsed_ms={}",
            build_id,
            calculation_started.elapsed().as_millis()
        );
        let pp_header_started = Instant::now();
        info!("PP-BUILD-DIAG id={} stage=pruning_point_header_read_start pruning_point={}", build_id, pp);
        let pp_header = self.headers_store.get_header_with_block_level(pp).unwrap();
        info!(
            "PP-BUILD-DIAG id={} stage=pruning_point_header_read_complete pruning_point={} selected_tip={} block_level={} elapsed_ms={}",
            build_id,
            pp,
            pp_header.header.hash,
            pp_header.block_level,
            pp_header_started.elapsed().as_millis()
        );

        let mut level_proof_ctxs: Vec<Option<LevelProofContext>> = vec![None; (self.max_block_level + 1).into()];

        for level in (0..=self.max_block_level).rev() {
            let level_started = Instant::now();
            let level_idx = level as usize;
            info!("PP-BUILD-DIAG id={} stage=level_start level={}", build_id, level);
            let required_block = if level != self.max_block_level {
                let LevelProofContext { ghostdag_store: next_level_gd_store, tip: next_level_tip, .. } =
                    level_proof_ctxs[level_idx + 1].as_ref().unwrap();

                let block_at_depth_m_at_next_level = next_level_gd_store
                    .block_at_depth(*next_level_tip, self.pruning_proof_m)
                    .map_err(|err| format!("next level: {}, err: {}", level + 1, err))
                    .unwrap();
                Some(block_at_depth_m_at_next_level)
            } else {
                None
            };
            let level_ctx = self
                .calc_level_proof_context(
                    &pp_header,
                    level,
                    required_block,
                    prev_descriptor.as_ref().map(|d| d.tips[level_idx]),
                    prev_descriptor.as_ref().map(|d| d.roots[level_idx]),
                    temp_db.clone(),
                    build_id,
                )
                .unwrap_or_else(|e| panic!("calc_level_proof_context failed for level {level}: {e}"));
            info!(
                "PP-BUILD-DIAG id={} stage=level_complete level={} root={} tip={} headers={} elapsed_ms={}",
                build_id,
                level,
                level_ctx.root,
                level_ctx.tip,
                level_ctx.count,
                level_started.elapsed().as_millis()
            );
            level_proof_ctxs[level_idx] = Some(level_ctx);
        }

        let (tips, roots, counts) = level_proof_ctxs.into_iter().map(Option::unwrap).map(|l| (l.tip, l.root, l.count)).multiunzip();
        info!(
            "PP-BUILD-DIAG id={} stage=calculate_descriptor_complete elapsed_ms={}",
            build_id,
            calculation_started.elapsed().as_millis()
        );
        PruningProofDescriptor::new(pp, tips, roots, counts)
    }

    /// Computes a level-proof context by incrementally expanding the level relations subgraph and
    /// periodically attempting to anchor a proof between a candidate `root` and the selected `tip`.
    ///
    /// High-level flow:
    /// 1. Determine the selected `tip` at `level` (either the pruning point itself, or its approximate
    ///    selected parent at that level).
    /// 2. Traverse *backward* from the selected `tip` while populating a temporary relations store.
    ///    Traversal is performed in reverse-topological order so that all derived data
    ///    (e.g. future-size tracking, ghostdag inputs) is observed consistently.
    /// 3. Whenever the current block is a viable root candidate (sufficient base-level depth,
    ///    sufficient future size, and inclusion constraints), populate a temporary ghostdag store
    ///    for the region `future(root) ∩ past(tip)` and test whether it satisfies the
    ///    proof-level requirements.
    /// 4. If a candidate fails due to insufficient realized blue depth (due to reds),
    ///    increase the future-size threshold and continue searching further into the past.
    ///
    /// If `required_block` is provided, the chosen root must lie in its past.
    /// Typically, this block is the one at depth `M` from the *next* level, as mandated by the
    /// MLS (mining-in-log-space) protocol. Since level proofs are computed from higher levels
    /// to lower ones, the caller already has this block available and supplies it here to
    /// constrain root selection and ensure inter-level consistency.
    fn calc_level_proof_context(
        &self,
        pp_header: &HeaderWithBlockLevel,
        level: BlockLevel,
        required_block: Option<Hash>,
        prev_tip: Option<Hash>,
        prev_root: Option<Hash>,
        db: Arc<DB>,
        build_id: u64,
    ) -> ProofInternalResult<LevelProofContext> {
        let level_started = Instant::now();
        info!(
            "PP-BUILD-DIAG id={} stage=root_calculation_level_start level={} previous_tip={:?} previous_root={:?} required_block={:?}",
            build_id, level, prev_tip, prev_root, required_block
        );
        let reachability_probe_started = Instant::now();
        info!("PP-BUILD-DIAG id={} stage=reachability_lock_wait_start phase=root_calculation level={}", build_id, level);
        let reachability_probe = self.reachability_store.read();
        info!(
            "PP-BUILD-DIAG id={} stage=reachability_lock_acquired phase=root_calculation level={} elapsed_ms={}",
            build_id,
            level,
            reachability_probe_started.elapsed().as_millis()
        );
        drop(reachability_probe);

        // Select the tip at this level:
        // - If the pruning point level >= level, use it.
        // - Otherwise, use the approximate selected parent at level.
        info!("PP-BUILD-DIAG id={} stage=level_tip_selection_start level={}", build_id, level);
        let tip = if pp_header.block_level >= level {
            pp_header.header.hash
        } else {
            // When advancing from a previous descriptor, require `prev_tip` to lie in the past of the new selected tip.
            // This preserves monotonicity across successive proofs (see `prev_root` rationale below).
            //
            // Note: such a parent always exists because the new pruning point is in the future of previous pruning points.
            self.reachable_parents_at_level(level, &pp_header.header)
                .filter(|&p| prev_tip.is_none_or(|prev_tip| self.reachability_service.is_dag_ancestor_of(prev_tip, p)))
                .map(|p| self.headers_store.get_header(p).expect("reachable"))
                .max_by_key(|h| SortableBlock::new(h.hash, h.blue_work))
                .ok_or_else(|| ProofInternalError::NotEnoughHeadersToBuildProof("no reachable parents".to_string()))?
                .hash
        };
        info!(
            "PP-BUILD-DIAG id={} stage=level_tip_selection_complete level={} tip={} elapsed_ms={}",
            build_id,
            level,
            tip,
            level_started.elapsed().as_millis()
        );

        // Base-level blue score of the selected tip, taken directly from the header.
        // This is distinct from the *locally computed* blue score later derived from
        // the temporary ghostdag instance at this level.
        let tip_header_score = self.headers_store.get_blue_score(tip).unwrap();

        // Proof thresholds:
        // - required_future_size gates root candidacy based on how much future mass a root covers.
        // - required_base_level_depth is a base-level (header) blue-depth guard: if level 0
        //   lacks sufficient blues here, we avoid premature ghostdag attempts.
        let mut required_future_size = 2 * self.pruning_proof_m;
        let required_base_level_depth = (self.pruning_proof_m as f64 * 2.1) as u64; // ~= 2100 for M=1000

        // If no explicit required block is provided, default to `tip`.
        // Typically, `required_block` is the block at depth `M` from the *next* level, per the MLS protocol
        let required = required_block.unwrap_or(tip);

        // Backward traversal from `tip` in reverse-topological order
        // to maintain consistency for all derived computations.
        let mut queue = BinaryHeap::<SortableBlock>::new();
        let mut visited = BlockHashSet::new();
        queue.push(SortableBlock { hash: tip, blue_work: self.headers_store.get_header(tip).unwrap().blue_work });

        let cache_policy = CachePolicy::Count(2 * self.pruning_proof_m as usize);

        // A single shared relations store is maintained for the entire search of this level.
        let mut relations_store = DbRelationsStore::new_temp(db.clone(), level, 0, cache_policy, cache_policy);

        // For each visited block, store the size of its (known) future up to `tip`.
        let mut future_sizes_map = BlockHashMap::<u64>::new();
        let mut traversal_count = 0usize;
        let mut parent_edges = 0usize;
        let root_attempts = Cell::new(0usize);

        // Each ghostdag attempt uses a fresh temp store namespace (indexed internally by `retries`).
        let mut ghostdag_factory = GhostdagStoreFactory::new(db.clone(), cache_policy, level);
        let mut reachability_factory = ReachabilityStoreFactory::new(db.clone(), cache_policy, level);

        // Track a few high-future-size candidates for a final fallback pass
        let mut best_future_roots = TopK::<(u64, Hash), 8>::new();

        // Try to realize a level-proof from a candidate root
        let mut try_root = |relations_store: &DbRelationsStore, root: Hash, future_size: u64| -> Option<LevelProofContext> {
            let attempt = root_attempts.get() + 1;
            root_attempts.set(attempt);
            let attempt_started = Instant::now();
            info!(
                "PP-BUILD-DIAG id={} stage=root_attempt_start level={} attempt={} root={} tip={} future_size={} required={}",
                build_id, level, attempt, root, tip, future_size, required
            );
            // Populate ghostdag for `future(root) ∩ past(tip)` and test depth requirements.
            let (ghostdag_store, has_required_block, count) = self.populate_level_proof_ghostdag_data(
                relations_store,
                &mut ghostdag_factory,
                &mut reachability_factory,
                root,
                tip,
                required,
                level,
                self.ghostdag_k,
                build_id,
                attempt,
            );

            // Realized blue depth for this root, computed from the level-specific ghostdag
            let current_level_score = ghostdag_store.get_blue_score(tip).unwrap();
            info!(
                "PP-BUILD-DIAG id={} stage=root_attempt_complete level={} attempt={} root={} headers={} has_required={} blue_score={} elapsed_ms={}",
                build_id,
                level,
                attempt,
                root,
                count,
                has_required_block,
                current_level_score,
                attempt_started.elapsed().as_millis()
            );

            // Log all non-trivial cases
            if tip != self.genesis_hash {
                debug!(
                    "level: {}, future: {}, blue score: {}, count: {}, retries: {}",
                    level, future_size, current_level_score, count, ghostdag_factory.retries
                );
            }

            // Success:
            // - Must include the required block (the block at depth M from the next level)
            // - If root is genesis, required-block inclusion is sufficient
            // - Otherwise require at least `2M` blue depth at this level
            if has_required_block && (root == self.genesis_hash || current_level_score >= 2 * self.pruning_proof_m) {
                Some(LevelProofContext { ghostdag_store, tip, root, count })
            } else {
                None
            }
        };

        while let Some(SortableBlock { hash: current, .. }) = queue.pop() {
            if !visited.insert(current) {
                continue;
            }
            traversal_count += 1;
            if traversal_count % 10_000 == 0 {
                info!(
                    "PP-BUILD-DIAG id={} stage=level_walk_progress level={} headers={} queued={} parent_edges={} root_attempts={} elapsed_ms={}",
                    build_id,
                    level,
                    traversal_count,
                    queue.len(),
                    parent_edges,
                    root_attempts.get(),
                    level_started.elapsed().as_millis()
                );
            }

            if let Some(prev_root) = prev_root {
                // When advancing from a previous descriptor, use `prev_root` as a boundary for root selection.
                if !self.reachability_service.is_dag_ancestor_of(prev_root, current) {
                    continue;
                }
            }

            let header = self.headers_store.get_header(current).unwrap();

            // Collect reachable parents at this level
            let parents: BlockHashes = self.reachable_parents_at_level(level, &header).collect::<Vec<_>>().into();
            parent_edges += parents.len();

            // Persist relations for `current`
            relations_store.insert(current, parents.clone()).unwrap();

            trace!("Level: {} | Counting future size of {}", level, current);
            let future_size = self.count_future_size(&relations_store, current, &future_sizes_map, build_id, level);
            future_sizes_map.insert(current, future_size);
            trace!("Level: {} | Hash: {} | Future Size: {}", level, current, future_size);

            // Base-level depth from `tip`, measured using *header* blue scores.
            let base_level_depth = tip_header_score.saturating_sub(header.blue_score);

            // Root candidacy conditions:
            // - Must be in the past of `required`
            // - And one of:
            //   (a) genesis
            //   (b) sufficiently large future and sufficiently deep base-level distance
            if self.reachability_service.is_dag_ancestor_of(current, required) {
                // If the root appears immediately viable, attempt ghostdag now.
                // A successful attempt requires ≥ 2M realized blues at this level.
                if current == self.genesis_hash
                    || (future_size >= required_future_size && base_level_depth >= required_base_level_depth)
                {
                    let root = current;
                    if let Some(level_ctx) = try_root(&relations_store, root, future_size) {
                        return Ok(level_ctx);
                    }

                    // Large enough future with insufficient blue depth implies reds; increase the
                    // future-size threshold and retry further in the past.
                    required_future_size = (required_future_size as f64 * 1.1) as u64;
                } else if future_size >= 2 * self.pruning_proof_m {
                    // Minimum precondition for reaching ≥ 2M blues is future_size ≥ 2M.
                    // Defer ghostdag and keep as a fallback candidate.
                    best_future_roots.push((future_size, current));
                }
            }

            // Continue expanding the backward traversal.
            for &p in parents.iter() {
                queue.push(SortableBlock { hash: p, blue_work: self.headers_store.get_header(p).unwrap().blue_work });
            }
        }

        // Use the previous proof's root as the fallback anchor when progressing proofs.
        // With a fixed root, ghostdag selection is deterministic, and if the new tip is in the future of the
        // previous tip then blue score/work can only increase — so once the 2M (or genesis) invariant holds,
        // it continues to hold for all future progressions.
        if let Some(root) = prev_root {
            let future_size = *future_sizes_map.get(&root).expect("exhausted traversal");
            if let Some(level_ctx) = try_root(&relations_store, root, future_size) {
                return Ok(level_ctx);
            }
        }

        // Final fallback: give a last chance to a few high-future-size roots.
        // This is only needed for migrating nodes without a stored descriptor yet, and can be removed
        // once all nodes persist descriptors (along with the whole top-k fallback path).
        for (future_size, root) in best_future_roots.into_sorted_iter_ascending().collect_vec().into_iter().rev() {
            if let Some(level_ctx) = try_root(&relations_store, root, future_size) {
                return Ok(level_ctx);
            }
        }

        panic!("Failed to find sufficient root for level {level} after exhausting all known headers.");
    }

    /// Given a current hash, count the blocks in its future.
    ///
    /// The algorithm works as follows:
    /// 1. Identify the dominant child (the one with the largest future) to minimize traversal,
    ///    since most of the future is expected to be covered by it.
    /// 2. Perform a BFS over all other children and their futures, skipping blocks that are
    ///    already in the future of the dominant child.
    ///
    /// This is conceptually similar to mergeset calculation logic
    /// (effectively a traversal over the reversed mergeset).
    ///
    /// Assumes `future_sizes` is populated for all children of `current` (caller is expected to be doing a topological BFS).
    fn count_future_size(
        &self,
        relations: &DbRelationsStore,
        current: Hash,
        future_sizes: &BlockHashMap<u64>,
        build_id: u64,
        level: BlockLevel,
    ) -> u64 {
        let started = Instant::now();
        // Seed the BFS queue with all children of the current hash
        let mut queue: VecDeque<_> = relations.get_children(current).unwrap().read().iter().copied().collect();
        let mut visited = BlockHashSet::new();
        let mut relation_edges = 0usize;

        struct Entry {
            child: Hash,
            fut_size: u64,
        }

        // Future sizes are guaranteed to exist due to the topological BFS invariant
        let dominant_entry = queue
            .iter()
            .copied()
            .map(|child| Entry { child, fut_size: *future_sizes.get(&child).expect("topological bfs") })
            .max_by_key(|e| e.fut_size);

        let mut count = 0;

        if let Some(Entry { child: dominant_child, fut_size }) = dominant_entry {
            // Fully account for the dominant child future (+1 for itself) and exclude it from the traversal
            count += fut_size + 1;
            visited.insert(dominant_child);

            while let Some(hash) = queue.pop_front() {
                if !visited.insert(hash) {
                    continue;
                }

                // Skip blocks that are already in the future of the dominant child
                if self.reachability_service.is_dag_ancestor_of(dominant_child, hash) {
                    continue;
                }

                count += 1;
                let children = relations.get_children(hash).unwrap();
                relation_edges += children.read().len();
                for &child in children.read().iter() {
                    queue.push_back(child);
                }
                if visited.len() % 10_000 == 0 {
                    info!(
                        "PP-BUILD-DIAG id={} stage=future_size_walk_progress level={} root={} visited={} queued={} relation_edges={} elapsed_ms={}",
                        build_id,
                        level,
                        current,
                        visited.len(),
                        queue.len(),
                        relation_edges,
                        started.elapsed().as_millis()
                    );
                }
            }
        }

        if visited.len() >= 10_000 {
            info!(
                "PP-BUILD-DIAG id={} stage=future_size_walk_complete level={} root={} visited={} relation_edges={} future_size={} elapsed_ms={}",
                build_id,
                level,
                current,
                visited.len(),
                relation_edges,
                count,
                started.elapsed().as_millis()
            );
        }

        trace!("Counted future size of {} as {}", current, count);
        count
    }

    /// Forward-traverses from `root` toward `tip`, and inserts ghostdag data for each visited block.
    ///
    /// Traversal is restricted to `future(root) ∩ past(tip)` (i.e., blocks in the antipast of `tip` are ignored).
    /// Returns `true` iff `required_block` was encountered during traversal.
    fn populate_level_proof_ghostdag_data(
        &self,
        relations_store: &DbRelationsStore,
        ghostdag_factory: &mut GhostdagStoreFactory,
        reachability_factory: &mut ReachabilityStoreFactory,
        root: Hash,
        tip: Hash,
        required_block: Hash,
        level: BlockLevel,
        ghostdag_k: KType,
        build_id: u64,
        attempt: usize,
    ) -> (Arc<DbGhostdagStore>, bool, u64) {
        let started = Instant::now();
        info!(
            "PP-BUILD-DIAG id={} stage=ghostdag_population_start level={} attempt={} root={} tip={} required={}",
            build_id, level, attempt, root, tip, required_block
        );
        debug!("Populating GD for root {} at level {} (retry {})", root, level, ghostdag_factory.retries.saturating_sub(1));

        let ghostdag_store = ghostdag_factory.new_store();
        let reachability_store = reachability_factory.new_store();
        let reachability_service = MTReachabilityService::new(reachability_store.clone());

        // Init reachability with ORIGIN and add root as its only child
        reachability::init(reachability_store.write().deref_mut()).unwrap();
        reachability::add_block(reachability_store.write().deref_mut(), root, ORIGIN, &mut [].into_iter()).unwrap();

        // Restrict relations to `future(root)` via level reachability
        let relations_view = FutureIntersectRelations::new(relations_store, reachability_service.clone(), root);

        // Create a ghostdag manager over the restricted relations view
        let ghostdag_manager = GhostdagManager::with_level(
            root,
            ghostdag_k,
            ghostdag_store.clone(),
            &relations_view,
            self.headers_store.clone(),
            reachability_service.clone(),
            level,
            self.max_block_level,
            self.pom_maxlevel_v4_activation,
        );

        // No need to initialize origin since we have a single root
        ghostdag_store.insert(root, Arc::new(ghostdag_manager.genesis_ghostdag_data())).unwrap();

        // Bottom-up topological traversal from `root` toward `tip`
        let mut queue: BinaryHeap<_> = Default::default();
        let mut visited = BlockHashSet::new();
        let mut count = 1; // counting root
        for child in relations_view.get_children(root).unwrap().read().iter().copied() {
            queue.push(Reverse(SortableBlock { hash: child, blue_work: self.headers_store.get_header(child).unwrap().blue_work }));
        }

        let mut has_required_block = root == required_block;
        let mut reachability_tip = root;

        while let Some(Reverse(SortableBlock { hash: current, .. })) = queue.pop() {
            if !visited.insert(current) {
                continue;
            }

            // We only care about `future(root) ∩ past(tip)`
            if !self.reachability_service.is_dag_ancestor_of(current, tip) {
                continue;
            }

            has_required_block |= current == required_block;
            count += 1;
            if count % 10_000 == 0 {
                info!(
                    "PP-BUILD-DIAG id={} stage=ghostdag_population_progress level={} attempt={} root={} headers={} queued={} visited={} elapsed_ms={}",
                    build_id,
                    level,
                    attempt,
                    root,
                    count,
                    queue.len(),
                    visited.len(),
                    started.elapsed().as_millis()
                );
            }

            let parents = relations_view.get_parents(current).unwrap();
            assert!(!parents.is_empty(), "non-root blocks must have parents");

            let ghostdag_data = Arc::new(ghostdag_manager.ghostdag(parents.as_slice()));
            ghostdag_store.insert(current, ghostdag_data.clone()).unwrap();

            reachability_tip = ghostdag_manager.find_selected_parent([reachability_tip, current]);

            let mut level_reachability = reachability_store.write();
            let mut reachability_mergeset = ghostdag_data
                .unordered_mergeset_without_selected_parent()
                .filter(|hash| level_reachability.has(*hash).unwrap())
                .collect_vec()
                .into_iter();

            reachability::add_block(
                level_reachability.deref_mut(),
                current,
                ghostdag_data.selected_parent,
                &mut reachability_mergeset,
            )
            .unwrap();

            if current == reachability_tip {
                reachability::hint_virtual_selected_parent(level_reachability.deref_mut(), current).unwrap();
            }
            drop(level_reachability);

            for child in relations_view.get_children(current).unwrap().read().iter().copied() {
                queue.push(Reverse(SortableBlock { hash: child, blue_work: self.headers_store.get_header(child).unwrap().blue_work }));
            }
        }

        // Returned for sanity testing by the caller
        info!(
            "PP-BUILD-DIAG id={} stage=ghostdag_population_complete level={} attempt={} root={} headers={} has_required={} elapsed_ms={}",
            build_id,
            level,
            attempt,
            root,
            count,
            has_required_block,
            started.elapsed().as_millis()
        );
        (ghostdag_store, has_required_block, count)
    }

    /// Returns the header's parents at `level` that are reachable according to the reachability service,
    /// i.e., parents for which reachability data exists in the database.
    ///
    /// This function enforces the reachability / storage invariants described in the
    /// [crate-level documentation](crate): only parents with reachability data are returned.
    /// By convention, the returned hashes are therefore also guaranteed to have a header
    /// entry in the database.
    fn reachable_parents_at_level<'a>(&'a self, level: u8, header: &'a Header) -> impl Iterator<Item = Hash> + 'a {
        // `parents_at_level` may include candidates that are not currently in the database.
        // This is fine here: we only need *some* sufficiently-deep reachable root for a proof at this level,
        // not necessarily the "best" / most complete set of candidates.
        self.parents_manager
            .parents_at_level(header, level)
            .iter()
            .copied()
            // Filtering by header existence alone is not enough: we may store headers of past pruning points,
            // but those are not part of the reachable DAG for proof purposes.
            .filter(|&p| self.reachability_service.has_reachability_data(p))
    }
}
