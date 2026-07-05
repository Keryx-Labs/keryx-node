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
use std::sync::{Arc, atomic::Ordering};

pub struct BlockBodyProcessor {
    // Channels
    receiver: Receiver<BlockProcessingMessage>,
    sender: Sender<VirtualStateProcessingMessage>,

    // Thread pool
    pub(super) thread_pool: Arc<ThreadPool>,

    // DB
    db: Arc<DB>,

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

    // Stores
    pub(super) statuses_store: Arc<RwLock<DbStatusesStore>>,
    pub(super) _ghostdag_store: Arc<DbGhostdagStore>,
    pub(super) _headers_store: Arc<DbHeadersStore>,
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
}

impl BlockBodyProcessor {
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

            statuses_store: storage.statuses_store.clone(),
            _ghostdag_store: storage.ghostdag_store.clone(),
            _headers_store: storage.headers_store.clone(),
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

        let mass = match self.validate_body(block, is_trusted, skip_pom_proof) {
            Ok(mass) => mass,
            Err(e) => {
                // We mark invalid blocks with status StatusInvalid except in the
                // case of the following errors:
                // MissingParents - If we got MissingParents the block shouldn't be
                // considered as invalid because it could be added later on when its
                // parents are present.
                // BadMerkleRoot - if we get BadMerkleRoot we shouldn't mark the
                // block as invalid because later on we can get the block with
                // transactions that fits the merkle root.
                // PrunedBlock - PrunedBlock is an error that rejects a block body and
                // not the block as a whole, so we shouldn't mark it as invalid.
                if !matches!(e, RuleError::BadMerkleRoot(_, _) | RuleError::MissingParents(_) | RuleError::PrunedBlock) {
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
        let pom_proof = block.pom_proof.clone();
        self.commit_body(block.hash(), block.header.direct_parents(), block.transactions.clone(), pom_proof, block.pom_tier);

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
