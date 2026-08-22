use crate::{
    consensus::{
        services::{ConsensusServices, DbWindowManager},
        storage::ConsensusStorage,
    },
    errors::{BlockProcessResult, RuleError},
    model::{
        services::reachability::MTReachabilityService,
        stores::{
            DB,
            block_transactions::DbBlockTransactionsStore,
            ghostdag::DbGhostdagStore,
            headers::DbHeadersStore,
            pom_proof::DbPomProofStore,
            pom_tier::DbPomTierStore,
            reachability::DbReachabilityStore,
            statuses::{DbStatusesStore, StatusesStore, StatusesStoreBatchExtensions, StatusesStoreReader},
            tips::{DbTipsStore, TipsStore},
        },
    },
    pipeline::{
        ProcessingCounters,
        deps_manager::{BlockProcessingMessage, BlockTaskDependencyManager, TaskId, VirtualStateProcessingMessage},
    },
    processes::{coinbase::CoinbaseManager, transaction_validator::TransactionValidator},
};
use crossbeam_channel::{Receiver, Sender};
use keryx_consensus_core::{
    KType,
    block::Block,
    blockstatus::BlockStatus::{self, StatusHeaderOnly, StatusInvalid},
    config::{genesis::GenesisBlock, params::{ForkActivation, Params}},
    mass::{Mass, MassCalculator, MassOps},
    pom::PomProof,
    tx::Transaction,
};
use keryx_consensus_notify::{
    notification::{BlockAddedNotification, Notification},
    root::ConsensusNotificationRoot,
};
use keryx_consensusmanager::SessionLock;
use keryx_hashes::Hash;
use keryx_notify::notifier::Notify;
use parking_lot::RwLock;
use rayon::ThreadPool;
use rocksdb::WriteBatch;
use std::collections::HashSet;
use std::sync::{Arc, atomic::Ordering};

pub struct BlockBodyProcessor {
    // Channels
    receiver: Receiver<BlockProcessingMessage>,
    sender: Sender<VirtualStateProcessingMessage>,

    // Thread pool
    pub(super) thread_pool: Arc<ThreadPool>,

    // DB
    pub(super) db: Arc<DB>,

    // Config
    pub(super) max_block_mass: u64,
    pub(super) genesis: GenesisBlock,
    pub(super) ghostdag_k: KType,
    pub(super) skip_opoi: bool,
    /// PoM possession activation — when active at a block's daa_score, its `pom_proof` is verified.
    pub(super) pom_activation: ForkActivation,
    /// H2 lineup gate — selects the 5-tier `pom_tiers` set when active at a block's daa_score.
    pub(super) very_light_activation: ForkActivation,
    /// H3 gate — when active at a block's daa_score, `check_pom_proof` additionally pins
    /// `proof.final_state == header.pom_final_state` (the header commitment the block level
    /// and header-only PoW check derive from).
    pub(super) pom_level_activation: ForkActivation,
    pub(super) reward_routing_activation: ForkActivation,
    /// H4 gate — when active at a block's daa_score, `check_pom_proof` uses the recompute-from-chunks
    /// verifier (`verify_pom_proof_v2`, all K transitions re-walked) instead of the 32/256 spot-check.
    /// Bundled into H4 alongside coin-age (same `coin_age_verification_activation` DAA).
    pub(super) coin_age_verification_activation: ForkActivation,
    /// H5 gate — when active at a block's daa_score, `check_pom_proof` re-walks with the
    /// non-foldable mix64-chained transition (`verify_pom_proof_v2(.., walk_v2= true)`); pre-H5
    /// blocks re-walk with the frozen v1 fold. Single H5 bundle gate (`Params::h5_activation`).
    pub(super) h5_activation: ForkActivation,
    /// H5.1 emergency relaunch: walk seed derives from the v2-salted pph words at/after the gate
    /// (`POM_H5_1_PPH_SALT`). Seed only — the header pow fold stays on the H3 salt.
    pub(super) h5_1_activation: ForkActivation,
    /// H5.2 chain anchoring: walk seed derives from the v3-salted pph words at/after the gate
    /// (`POM_H5_2_PPH_SALT`). Seed only — the header pow fold stays on the H3 salt.
    pub(super) h5_2_activation: ForkActivation,
    /// H6 gate — when active at a block's daa_score, `check_pom_proof` requires the v3
    /// matrix-walk witness and verifies it with `verify_pom_proof_v3_container` (spot-checked
    /// state commitments; the verifier never re-walks). Same seed/target/final_hash inputs as
    /// the v2 path; `header.pom_final_state` carries `pom_v3::fold64(roots[K])`.
    pub(super) pom_v3_activation: ForkActivation,
    pub(super) pom_v4_activation: ForkActivation,

    // Stores
    pub(super) statuses_store: Arc<RwLock<DbStatusesStore>>,
    pub(super) _ghostdag_store: Arc<DbGhostdagStore>,
    pub(super) headers_store: Arc<DbHeadersStore>,
    pub(super) block_transactions_store: Arc<DbBlockTransactionsStore>,
    /// Proven PoM tier per block, persisted at commit for the tier-reward coinbase split.
    pub(super) pom_tier_store: Arc<DbPomTierStore>,
    /// Full PoM possession proof per block, persisted at commit so the block can be re-served
    /// (relay/IBD) with its proof attached. See `DbPomProofStore`.
    pub(super) pom_proof_store: Arc<DbPomProofStore>,
    pub(super) body_tips_store: Arc<RwLock<DbTipsStore>>,

    // Managers and services
    pub(super) _reachability_service: MTReachabilityService<DbReachabilityStore>,
    pub(super) coinbase_manager: CoinbaseManager,
    pub(crate) mass_calculator: MassCalculator,
    pub(super) transaction_validator: TransactionValidator,
    pub(super) window_manager: DbWindowManager,

    // Pruning lock
    pruning_lock: SessionLock,

    // Dependency manager
    task_manager: BlockTaskDependencyManager,

    // Notifier
    notification_root: Arc<ConsensusNotificationRoot>,

    // Counters
    counters: Arc<ProcessingCounters>,

    /// Negative cache of PoM witnesses that failed verification, keyed by
    /// (block hash, witness wire digest). The witness travels OUTSIDE the block
    /// hash, so a bad witness must never invalidate the hash (witness poisoning) — but
    /// without this cache, dropping the invalidation would let a peer force
    /// unbounded re-verification of the same bad witness. Bounded two-generation
    /// set; see `BadWitnessCache`.
    bad_witness_cache: RwLock<BadWitnessCache>,
}

/// Bounded two-generation set: inserts go to the current generation; when it
/// fills up, generations rotate and the oldest entries fall away. Membership
/// checks look at both generations, so an entry lives for at least one and at
/// most two generation lifetimes. No LRU bookkeeping, O(1) everything.
#[derive(Default)]
pub(super) struct BadWitnessCache {
    current: HashSet<(Hash, [u8; 32])>,
    previous: HashSet<(Hash, [u8; 32])>,
}

impl BadWitnessCache {
    /// Per-generation capacity: 4096 entries × 40 bytes ≈ 160 KiB per generation.
    /// An attacker cannot use rotation to flush their own entry usefully: filling
    /// a generation costs 4096 distinct failed verifications.
    const GENERATION_CAP: usize = 4096;

    pub(super) fn contains(&self, block_hash: Hash, witness_digest: &[u8; 32]) -> bool {
        let key = (block_hash, *witness_digest);
        self.current.contains(&key) || self.previous.contains(&key)
    }

    pub(super) fn insert(&mut self, block_hash: Hash, witness_digest: [u8; 32]) {
        if self.current.len() >= Self::GENERATION_CAP {
            self.previous = std::mem::take(&mut self.current);
        }
        self.current.insert((block_hash, witness_digest));
    }
}

/// True when a body-validation error is scoped to the PoM WITNESS rather than the
/// block itself. The witness travels outside the block hash (transport attachment),
/// so none of these may ever mark the HASH invalid: the same block can arrive later
/// with its honest witness. Marking the hash would let a single crafted witness
/// poison a valid block permanently, reject every descendant and wedge the node
/// (witness poisoning). The delivery is still rejected — and the relay flow still drops the
/// peer that served it.
pub(super) fn witness_scoped_error(e: &RuleError) -> bool {
    matches!(
        e,
        RuleError::BadPomProof(_)
            | RuleError::BadPomProofV3(_)
            | RuleError::BadPomProofV4(_)
            | RuleError::PomFinalStateMismatch(_, _)
            | RuleError::PomUnknownTier(_)
    )
}

/// Whether a body-validation failure persists `StatusInvalid` for the block hash.
/// Base exemptions (every era):
/// - MissingParents: the block may become valid once its parents arrive.
/// - BadMerkleRoot: a later delivery may carry the transactions matching the root.
/// - PrunedBlock: rejects this body delivery, not the block as a whole.
/// - PomProofMissing: the proof is a transport-level attachment (stripped by IBD
///   beyond the retention window, garbage-collected, or dropped by a lagging peer);
///   its absence only rejects this delivery.
/// - KnownBadPomWitness: cache short-circuit, only ever produced when the fix is active.
///
/// With `bad_witness_rejects_delivery` (H6 gate, keyed on the block's own daa_score)
/// every witness-scoped error is exempted as well — see `witness_scoped_error`.
/// Pre-gate blocks keep the historical behavior: a present-but-wrong witness marks
/// invalid.
pub(super) fn marks_block_invalid(e: &RuleError, bad_witness_rejects_delivery: bool) -> bool {
    if matches!(
        e,
        RuleError::BadMerkleRoot(_, _)
            | RuleError::MissingParents(_)
            | RuleError::PrunedBlock
            | RuleError::PomProofMissing
            | RuleError::KnownBadPomWitness
    ) {
        return false;
    }
    !(bad_witness_rejects_delivery && witness_scoped_error(e))
}

impl BlockBodyProcessor {
    #[cfg(test)]
    pub(crate) fn pom_tier_store(&self) -> &Arc<DbPomTierStore> {
        &self.pom_tier_store
    }

    pub fn new(
        receiver: Receiver<BlockProcessingMessage>,
        sender: Sender<VirtualStateProcessingMessage>,
        thread_pool: Arc<ThreadPool>,

        params: &Params,
        db: Arc<DB>,
        storage: &Arc<ConsensusStorage>,
        services: &Arc<ConsensusServices>,

        pruning_lock: SessionLock,
        notification_root: Arc<ConsensusNotificationRoot>,
        counters: Arc<ProcessingCounters>,
    ) -> Self {
        Self {
            receiver,
            sender,
            thread_pool,
            db,

            max_block_mass: params.max_block_mass,
            genesis: params.genesis.clone(),
            ghostdag_k: params.ghostdag_k(),
            skip_opoi: params.skip_proof_of_work,
            pom_activation: params.pom_activation,
            very_light_activation: params.very_light_activation,
            pom_level_activation: params.pom_level_activation,
            reward_routing_activation: params.reward_routing_activation,
            coin_age_verification_activation: params.coin_age_verification_activation,
            h5_activation: params.h5_activation,
            h5_1_activation: params.h5_1_activation,
            h5_2_activation: params.h5_2_activation,
            pom_v3_activation: params.pom_v3_activation,
            pom_v4_activation: params.pom_v4_activation,

            statuses_store: storage.statuses_store.clone(),
            _ghostdag_store: storage.ghostdag_store.clone(),
            headers_store: storage.headers_store.clone(),
            block_transactions_store: storage.block_transactions_store.clone(),
            pom_tier_store: storage.pom_tier_store.clone(),
            pom_proof_store: storage.pom_proof_store.clone(),
            body_tips_store: storage.body_tips_store.clone(),

            _reachability_service: services.reachability_service.clone(),
            coinbase_manager: services.coinbase_manager.clone(),
            mass_calculator: services.mass_calculator.clone(),
            transaction_validator: services.transaction_validator.clone(),
            window_manager: services.window_manager.clone(),

            pruning_lock,
            task_manager: BlockTaskDependencyManager::new(),
            notification_root,
            counters,
            bad_witness_cache: RwLock::new(BadWitnessCache::default()),
        }
    }

    pub fn worker(self: &Arc<BlockBodyProcessor>) {
        while let Ok(msg) = self.receiver.recv() {
            match msg {
                BlockProcessingMessage::Exit => break,
                BlockProcessingMessage::Process(task, block_result_transmitter, virtual_result_transmitter) => {
                    if let Some(task_id) = self.task_manager.register(task, block_result_transmitter, virtual_result_transmitter) {
                        let processor = self.clone();
                        self.thread_pool.spawn(move || {
                            processor.queue_block(task_id);
                        });
                    }
                }
            };
        }

        // Wait until all workers are idle before exiting
        self.task_manager.wait_for_idle();

        // Pass the exit signal on to the following processor
        self.sender.send(VirtualStateProcessingMessage::Exit).unwrap();
    }

    fn queue_block(self: &Arc<BlockBodyProcessor>, task_id: TaskId) {
        if let Some(task) = self.task_manager.try_begin(task_id) {
            let res = self.process_body(task.block(), task.is_trusted(), task.skip_pom_proof());

            let dependent_tasks = self.task_manager.end(task, |task, block_result_transmitter, virtual_state_result_transmitter| {
                let _ = block_result_transmitter.send(res.clone());
                if res.is_err() || !task.requires_virtual_processing() {
                    // We don't care if receivers were dropped
                    let _ = virtual_state_result_transmitter.send(res.clone());
                } else {
                    self.sender.send(VirtualStateProcessingMessage::Process(task, virtual_state_result_transmitter)).unwrap();
                }
            });

            for dep in dependent_tasks {
                let processor = self.clone();
                self.thread_pool.spawn(move || processor.queue_block(dep));
            }
        }
    }

    fn process_body(self: &Arc<BlockBodyProcessor>, block: &Block, is_trusted: bool, skip_pom_proof: bool) -> BlockProcessResult<BlockStatus> {
        let _prune_guard = self.pruning_lock.blocking_read();
        let status = self.statuses_store.read().get(block.hash()).unwrap();
        match status {
            StatusInvalid => return Err(RuleError::KnownInvalid),
            StatusHeaderOnly => {} // Proceed to body processing
            _ if status.has_block_body() => return Ok(status),
            _ => panic!("unexpected block status {status:?}"),
        }

        // Witness-poisoning fix, gated at the H6 (pom_v3) fork by the block's OWN daa_score — the same
        // deterministic key the proof-era selection uses, so every updated node applies the
        // same policy to the same block. Pre-gate behavior is byte-identical to the previous
        // release (a present-but-wrong witness still marks StatusInvalid below).
        let bad_witness_rejects_delivery = self.pom_v3_activation.is_active(block.header.daa_score);

        // Negative witness cache: a witness that already failed for this block is rejected
        // without re-verification (pre-v3 verifiers re-walk all K transitions — unbounded
        // re-verification would be a cheap CPU DoS).
        let witness_digest = if bad_witness_rejects_delivery { block.pom_proof.as_ref().map(|p| p.wire_digest()) } else { None };
        if let Some(digest) = witness_digest {
            if self.bad_witness_cache.read().contains(block.hash(), &digest) {
                return Err(RuleError::KnownBadPomWitness);
            }
        }

        let mass = match self.validate_body(block, is_trusted, skip_pom_proof) {
            Ok(mass) => mass,
            Err(e) => {
                // Remember failed witnesses per (block, witness) — see the cache above.
                if witness_scoped_error(&e) {
                    if let Some(digest) = witness_digest {
                        self.bad_witness_cache.write().insert(block.hash(), digest);
                    }
                }
                if marks_block_invalid(&e, bad_witness_rejects_delivery) {
                    self.statuses_store.write().set(block.hash(), BlockStatus::StatusInvalid).unwrap();
                }
                return Err(e);
            }
        };

        // Persist the PoM possession proof (verified in `check_pom_proof`): the full proof so the
        // block can be re-served to peers (relay/IBD) with its proof attached, plus the tier alone
        // for the tier-reward coinbase split read by the virtual processor. The in-memory
        // `block.pom_proof` is dropped once a block is reloaded from storage, so both must be
        // captured here while it is still attached to the block.
        //
        // The IBD path skips `check_pom_proof` at validation (`skip_pom_proof`), but a carried
        // proof would still be persisted here and re-served later. Never persist an UNVERIFIED
        // proof — a malicious syncer could feed a bogus one that proof-enforcing relay peers would
        // then reject from US. Verify it now; on failure persist the block naked (the tier still
        // travels separately) and let the re-proof loop fetch a valid proof from another peer.
        let pom_proof = if skip_pom_proof && block.pom_proof.is_some() {
            match self.check_pom_proof(block) {
                Ok(()) => block.pom_proof.clone(),
                Err(e) => {
                    keryx_core::warn!(
                        "IBD-carried PoM proof of {} failed verification ({}) — persisting the block without it",
                        block.hash(),
                        e
                    );
                    None
                }
            }
        } else {
            block.pom_proof.clone()
        };
        // IBD may accept bodies without a carried proof, but their tier is not authenticated.
        // Never persist an unverified tier claim for later coinbase reward calculation.
        let pom_tier = if skip_pom_proof && pom_proof.is_none() { None } else { block.pom_tier };
        self.commit_body(block.hash(), block.header.direct_parents(), block.transactions.clone(), pom_proof, pom_tier);

        // Send a BlockAdded notification
        self.notification_root
            .notify(Notification::BlockAdded(BlockAddedNotification::new(block.to_owned())))
            .expect("expecting an open unbounded channel");

        // Report counters
        self.counters.body_counts.fetch_add(1, Ordering::Relaxed);
        self.counters.txs_counts.fetch_add(block.transactions.len() as u64, Ordering::Relaxed);
        self.counters.mass_counts.fetch_add(mass.max(), Ordering::Relaxed);
        Ok(BlockStatus::StatusUTXOPendingVerification)
    }

    fn validate_body(self: &Arc<BlockBodyProcessor>, block: &Block, is_trusted: bool, skip_pom_proof: bool) -> BlockProcessResult<Mass> {
        let mass = self.validate_body_in_isolation(block, skip_pom_proof)?;
        if !is_trusted {
            self.validate_body_in_context(block)?;
        }
        Ok(mass)
    }

    fn commit_body(
        self: &Arc<BlockBodyProcessor>,
        hash: Hash,
        parents: &[Hash],
        transactions: Arc<Vec<Transaction>>,
        pom_proof: Option<Arc<PomProof>>,
        pom_tier: Option<u8>,
    ) {
        let mut batch = WriteBatch::default();

        // This is an append only store so it requires no lock.
        self.block_transactions_store.insert_batch(&mut batch, hash, transactions).unwrap();

        // Append-only: persist the possession proof (full proof for re-serving + tier alone for the
        // tier-reward split) when the block carried one. On the IBD path the full proof is absent but
        // the tier travels separately (`block.pom_tier`) — persist it so the coinbase tier-reward
        // split is reconstructible. `proof.tier` is authoritative when a proof is present.
        if let Some(proof) = &pom_proof {
            self.pom_proof_store.insert_batch(&mut batch, hash, proof).unwrap();
            self.pom_tier_store.insert_batch(&mut batch, hash, proof.tier).unwrap();
        } else if let Some(tier) = pom_tier {
            self.pom_tier_store.insert_batch(&mut batch, hash, tier).unwrap();
        }

        let mut body_tips_write_guard = self.body_tips_store.write();
        body_tips_write_guard.add_tip_batch(&mut batch, hash, parents).unwrap();
        let statuses_write_guard =
            self.statuses_store.set_batch(&mut batch, hash, BlockStatus::StatusUTXOPendingVerification).unwrap();

        self.db.write(batch).unwrap();

        // Calling the drops explicitly after the batch is written in order to avoid possible errors.
        drop(statuses_write_guard);
        drop(body_tips_write_guard);
    }

    pub fn process_genesis(self: &Arc<BlockBodyProcessor>) {
        // Init tips store
        let mut batch = WriteBatch::default();
        let mut body_tips_write_guard = self.body_tips_store.write();
        body_tips_write_guard.init_batch(&mut batch, &[]).unwrap();
        self.db.write(batch).unwrap();
        drop(body_tips_write_guard);

        // Write the genesis body
        self.commit_body(self.genesis.hash, &[], Arc::new(self.genesis.build_genesis_transactions()), None, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use keryx_consensus_core::pom::PomVerifyError;
    use keryx_consensus_core::pom_v3::PomV3VerifyError;
    use keryx_consensus_core::pom_v4::PomV4VerifyError;

    #[test]
    fn test_witness_errors_never_mark_the_block_invalid_post_gate() {
        // Post-gate (witness-poisoning fix active): witness-scoped errors reject the delivery,
        // never persist StatusInvalid.
        let witness_scoped = [
            RuleError::BadPomProof(PomVerifyError::TargetNotMet),
            RuleError::BadPomProofV3(PomV3VerifyError::MissingV3),
            RuleError::BadPomProofV4(PomV4VerifyError::MissingV4),
            RuleError::PomFinalStateMismatch(1, 2),
            RuleError::PomUnknownTier(7),
        ];
        for e in &witness_scoped {
            assert!(!marks_block_invalid(e, true), "{e:?} must not poison the block hash post-gate");
            // Pre-gate: historical behavior is preserved byte-identically — a
            // present-but-wrong witness still marks invalid.
            assert!(marks_block_invalid(e, false), "{e:?} must keep marking invalid pre-gate");
        }
        // Era-independent exemptions.
        for e in [RuleError::PomProofMissing, RuleError::KnownBadPomWitness] {
            assert!(!marks_block_invalid(&e, true));
            assert!(!marks_block_invalid(&e, false));
        }
        // Genuine block defects still mark invalid in both eras.
        assert!(marks_block_invalid(&RuleError::DuplicateTransactions(Default::default()), true));
        assert!(marks_block_invalid(&RuleError::DuplicateTransactions(Default::default()), false));
    }

    #[test]
    fn test_bad_witness_cache_membership_and_rotation() {
        let mut cache = BadWitnessCache::default();
        let block = Hash::from_u64_word(1);
        let witness_a = [0xaa; 32];
        let witness_b = [0xbb; 32];

        assert!(!cache.contains(block, &witness_a));
        cache.insert(block, witness_a);
        assert!(cache.contains(block, &witness_a), "failed witness must be remembered");
        // Same witness on another block, and another witness on the same block, are distinct.
        assert!(!cache.contains(Hash::from_u64_word(2), &witness_a));
        assert!(!cache.contains(block, &witness_b), "an honest replacement witness must not be blocked");

        // Rotation: an entry survives one full generation, dies after two.
        for i in 0..BadWitnessCache::GENERATION_CAP as u64 {
            cache.insert(Hash::from_u64_word(1000 + i), [0x11; 32]);
        }
        assert!(cache.contains(block, &witness_a), "entry must survive the first rotation (previous generation)");
        for i in 0..BadWitnessCache::GENERATION_CAP as u64 {
            cache.insert(Hash::from_u64_word(100_000 + i), [0x22; 32]);
        }
        assert!(!cache.contains(block, &witness_a), "entry must age out after two rotations");
    }
}
