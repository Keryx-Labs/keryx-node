use crate::{
    consensus::{
        services::{
            ConsensusServices, DbBlockDepthManager, DbDagTraversalManager, DbGhostdagManager, DbParentsManager, DbPruningPointManager,
            DbWindowManager,
        },
        storage::ConsensusStorage,
    },
    constants::BLOCK_VERSION,
    errors::RuleError,
    model::{
        services::{
            reachability::{MTReachabilityService, ReachabilityService},
            relations::MTRelationsService,
        },
        stores::{
            DB,
            acceptance_data::{AcceptanceDataStoreReader, DbAcceptanceDataStore},
            block_transactions::{BlockTransactionsStoreReader, DbBlockTransactionsStore},
            block_window_cache::{BlockWindowCacheStore, BlockWindowCacheWriter},
            daa::DbDaaStore,
            depth::{DbDepthStore, DepthStoreReader},
            ghostdag::{DbGhostdagStore, GhostdagData, GhostdagStoreReader},
            headers::{DbHeadersStore, HeaderStoreReader},
            address_amount::DbAddressAmountStore,
            age_buckets::{AgeBuckets, DbAgeBucketsStore},
            maturation_queue::{DbMaturationQueueStore, MaturationEntry},
            windowed_production_prefix::DbWindowedProductionPrefixStore,
            past_pruning_points::DbPastPruningPointsStore,
            pom_tier::DbPomTierStore,
            production_seed::{DbProductionIndexSeedStore, ProductionIndexSeedStore},
            pruning::{DbPruningStore, PruningStoreReader},
            pruning_meta::PruningMetaStores,
            pruning_samples::DbPruningSamplesStore,
            reachability::DbReachabilityStore,
            relations::{DbRelationsStore, RelationsStoreReader},
            selected_chain::{DbSelectedChainStore, SelectedChainStore, SelectedChainStoreReader},
            statuses::{DbStatusesStore, StatusesStore, StatusesStoreBatchExtensions, StatusesStoreReader},
            tips::{DbTipsStore, TipsStoreReader},
            utxo_diffs::{DbUtxoDiffsStore, UtxoDiffsStoreReader},
            utxo_multisets::{DbUtxoMultisetsStore, UtxoMultisetsStoreReader},
            virtual_state::{LkgVirtualState, VirtualState, VirtualStateStoreReader, VirtualStores},
        },
    },
    params::Params,
    pipeline::{
        ProcessingCounters, deps_manager::VirtualStateProcessingMessage, pruning_processor::processor::PruningProcessingMessage,
        virtual_processor::utxo_validation::UtxoProcessingContext,
    },
    processes::{
        coinbase::CoinbaseManager,
        ghostdag::ordering::SortableBlock,
        transaction_validator::{TransactionValidator, errors::TxResult, tx_validation_in_utxo_context::TxValidationFlags},
        window::WindowManager,
    },
};
use keryx_consensus_core::{
    BlockHashMap, BlockHashSet, ChainPath,
    acceptance_data::AcceptanceData,
    api::args::{TransactionValidationArgs, TransactionValidationBatchArgs},
    block::{BlockTemplate, MutableBlock, TemplateBuildMode, TemplateTransactionSelector},
    blockstatus::BlockStatus::{StatusDisqualifiedFromChain, StatusUTXOValid},
    coinbase::MinerData,
    config::{genesis::GenesisBlock, params::ForkActivation},
    header::Header,
    merkle::calc_hash_merkle_root,
    mining_rules::MiningRules,
    pruning::PruningPointsList,
    tx::{MutableTransaction, ScriptPublicKey, Transaction},
    utxo::{
        utxo_diff::UtxoDiff,
        utxo_view::{UtxoView, UtxoViewComposition},
    },
};
use keryx_consensus_notify::{
    notification::{
        NewBlockTemplateNotification, Notification, SinkBlueScoreChangedNotification, UtxosChangedNotification,
        VirtualChainChangedNotification, VirtualDaaScoreChangedNotification,
    },
    root::ConsensusNotificationRoot,
};
use keryx_consensusmanager::SessionLock;
use keryx_core::{debug, info, time::unix_now, trace, warn};
use keryx_database::prelude::{StoreError, StoreResultExt, StoreResultUnitExt};
use keryx_hashes::{Hash, ZERO_HASH};
use keryx_muhash::MuHash;
use keryx_notify::{events::EventType, notifier::Notify};
use once_cell::unsync::Lazy;

use super::errors::{PruningImportError, PruningImportResult};
use crossbeam_channel::{Receiver as CrossbeamReceiver, Sender as CrossbeamSender};
use itertools::Itertools;
use keryx_consensus_core::tx::ValidatedTransaction;
use keryx_utils::binary_heap::BinaryHeapExtensions;
use parking_lot::{RwLock, RwLockUpgradableReadGuard};
use rand::{Rng, seq::SliceRandom};
use rayon::{
    ThreadPool,
    prelude::{IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator},
};
use rocksdb::WriteBatch;
use std::{
    cmp::min,
    collections::{BinaryHeap, HashMap, VecDeque},
    ops::Deref,
    sync::{Arc, atomic::Ordering},
};

pub struct VirtualStateProcessor {
    // Channels
    receiver: CrossbeamReceiver<VirtualStateProcessingMessage>,
    pruning_sender: CrossbeamSender<PruningProcessingMessage>,
    pruning_receiver: CrossbeamReceiver<PruningProcessingMessage>,

    // Thread pool
    pub(super) thread_pool: Arc<ThreadPool>,

    // DB
    db: Arc<DB>,

    // Config
    pub(super) genesis: GenesisBlock,
    pub(super) max_block_parents: u8,
    pub(super) mergeset_size_limit: u64,

    // Stores
    pub(super) statuses_store: Arc<RwLock<DbStatusesStore>>,
    pub(super) ghostdag_store: Arc<DbGhostdagStore>,
    pub(super) headers_store: Arc<DbHeadersStore>,
    pub(super) daa_excluded_store: Arc<DbDaaStore>,
    pub(super) block_transactions_store: Arc<DbBlockTransactionsStore>,
    /// Proven PoM tier per block, read to scale the tier-reward coinbase split.
    pub(super) pom_tier_store: Arc<DbPomTierStore>,
    pub(super) pruning_point_store: Arc<RwLock<DbPruningStore>>,
    pub(super) past_pruning_points_store: Arc<DbPastPruningPointsStore>,
    pub(super) body_tips_store: Arc<RwLock<DbTipsStore>>,
    pub(super) depth_store: Arc<DbDepthStore>,
    pub(super) selected_chain_store: Arc<RwLock<DbSelectedChainStore>>,
    pub(super) production_index_seed_store: Arc<RwLock<DbProductionIndexSeedStore>>,
    pub(super) pruning_samples_store: Arc<DbPruningSamplesStore>,

    // Utxo-related stores
    pub(super) utxo_diffs_store: Arc<DbUtxoDiffsStore>,
    pub(super) utxo_multisets_store: Arc<DbUtxoMultisetsStore>,
    pub(super) acceptance_data_store: Arc<DbAcceptanceDataStore>,
    /// Ratio-reward balance index (Stage 2b): payout SPK → Σ unspent amount, kept in lockstep
    /// with the virtual UTXO set. Read to anchor `ratio_bps` at a block's selected-parent view.
    pub(super) address_balance_store: Arc<DbAddressAmountStore>,
    pub(super) age_buckets_store: Arc<DbAgeBucketsStore>,
    pub(super) maturation_queue_store: Arc<DbMaturationQueueStore>,
    /// Gold-standard prefix-sum production index (pure function of the chain): producer payout SPK → Σ
    /// `base_miner_cut` cumulated over the selected chain, kept in lockstep with it. The ratio
    /// denominator (windowed production at a block's selected-parent view) is the difference of two
    /// cumulatives; read by `ratio_bps_by_block`.
    pub(super) windowed_production_prefix_store: Arc<DbWindowedProductionPrefixStore>,
    /// Memo for `block_productions` (chain block → its era-aware production contribution list),
    /// i.e. a parsed-coinbase/mergeset cache. During catch-up, `resolve_virtual` validates the
    /// whole prev_sink→new_sink path in one batch while the committed selected chain still ends at
    /// prev_sink, so every block on the path takes the side-chain (Case B) branch of
    /// `windowed_production_for_block` and re-reads the SAME growing prefix of coinbases —
    /// quadratic RocksDB reads without this memo (measured: virtual thread pegged at ~4
    /// UTXO-validated blocks/s on an IBD catch-up). Bounded by periodic clear.
    pub(super) block_production_cache: parking_lot::RwLock<BlockHashMap<std::sync::Arc<Vec<(ScriptPublicKey, u64)>>>>,
    pub(super) virtual_stores: Arc<RwLock<VirtualStores>>,
    pub(super) pruning_meta_stores: Arc<RwLock<PruningMetaStores>>,

    /// The "last known good" virtual state. To be used by any logic which does not want to wait
    /// for a possible virtual state write to complete but can rather settle with the last known state
    pub lkg_virtual_state: LkgVirtualState,

    // Managers and services
    pub(super) ghostdag_manager: DbGhostdagManager,
    pub(super) reachability_service: MTReachabilityService<DbReachabilityStore>,
    pub(super) relations_service: MTRelationsService<DbRelationsStore>,
    pub(super) dag_traversal_manager: DbDagTraversalManager,
    pub(super) window_manager: DbWindowManager,
    pub(super) coinbase_manager: CoinbaseManager,
    pub(super) transaction_validator: TransactionValidator,
    pub(super) pruning_point_manager: DbPruningPointManager,
    pub(super) parents_manager: DbParentsManager,
    pub(super) depth_manager: DbBlockDepthManager,

    // block window caches
    pub(super) block_window_cache_for_difficulty: Arc<BlockWindowCacheStore>,
    pub(super) block_window_cache_for_past_median_time: Arc<BlockWindowCacheStore>,

    // Pruning lock
    pub(super) pruning_lock: SessionLock,

    // Notifier
    notification_root: Arc<ConsensusNotificationRoot>,

    // Counters
    counters: Arc<ProcessingCounters>,

    // Mining Rule
    _mining_rules: Arc<MiningRules>,

    // OPoI slash stores (Phase 3 A4) — persisted to RocksDB
    pub(super) ai_response_store: Arc<crate::model::stores::ai_slash::DbAiResponseStore>,

    // OPoI Phase 3 hardfork: model capability enforcement activation score
    pub(super) model_cap_enforcement_activation: ForkActivation,
    pub(super) inference_reward_minimums: &'static [([u8; 32], u64)],

    // OPoI v2 hardfork: uncensored lineup swap, DAA-gated
    pub(super) opoi_v2_activation: ForkActivation,
    pub(super) inference_reward_minimums_v2: &'static [([u8; 32], u64)],

    // H2 inference_reward minimums (5-tier: adds Qwen3-1.7B + 70B-Q2), DAA-gated at a future score
    pub(super) inference_min_h2_activation: ForkActivation,
    pub(super) inference_reward_minimums_v2_h2: &'static [([u8; 32], u64)],

    // SALT v2 hardfork: KeryxHash domain separation switch activation score
    pub(super) pow_salt_v2_activation: ForkActivation,

    // SALT v4 hardfork (chain relaunch on stock difficulty) activation score
    pub(super) pow_salt_v4_activation: ForkActivation,

    // PoM possession activation: also gates the tier-reward coinbase split (a proven
    // tier only exists under PoM). Empty tier-bps map before this score ⇒ no penalty.
    pub(super) pom_activation: ForkActivation,
    // H2 lineup gate — selects the 5-tier `tier_reward_bps` schedule when active at a block's daa.
    pub(super) very_light_activation: ForkActivation,
    // Ratio-reward activation: empty ratio-bps map before this score ⇒ no penalty.
    pub(super) ratio_reward_activation: ForkActivation,
    // H3 gate: per-blue production accounting + DAA-sized ratio window (see `block_productions`
    // and `production_window_ctx`). Same activation as the PoM block-level hardfork.
    pub(super) pom_level_activation: ForkActivation,
    // Coinbase ratio/tier VERIFICATION boundary: trust coinbases below this score (non-revalidatable
    // pre-relaunch history), verify with the prefix-sum at/above. Consensus rule (see params doc).
    pub(super) ratio_verification_activation: ForkActivation,
    // Trailing selected-chain window length (blocks) for the ratio-reward production index.
    pub(super) ratio_reward_window: u64,
    // H3 window length in DAA score (fixed real-time duration, per-blue era).
    pub(super) ratio_reward_window_daa: u64,
    // Coin-age holder-reward v3 (H4) gate: at/after this score, utxo-diff population assigns
    // FIFO-inherited `effective_daa` anchors (see `UtxoDiff::add_transaction`) and the ratio
    // numerator switches to the per-coin-capped effective balance. Dormant (`never()`) until H4.
    pub(super) coin_age_activation: ForkActivation,
    // Coin-age maturity period (DAA score): the mature/immature bucket boundary (see `apply_age_diff`).
    pub(super) coin_age_maturity_w: u64,
    // Retention horizon (DAA score) for PROMOTED maturation-queue entries, enabling read-path
    // demotion for POVs below the watermark (side chains). Finality-deep — beyond it a POV would
    // be rejected anyway.
    pub(super) coin_age_retention: u64,

    // Skip ratio/tier coinbase verification while following the chain. Three independent reasons,
    // ORed together and re-checked live (see `trust_coinbase()`) rather than fixed at construction:
    //  - `is_archival`: an archival node retains blocks below the pruning point, so its
    //    windowed-production fold subtracts leaving blocks that pruned nodes (the canonical
    //    majority) never see → it computes a different ratio and would disqualify the whole chain.
    //    Permanent for the node's lifetime.
    //  - `KERYX_TRUST_COINBASE` env: manual operator override, also permanent.
    //  - fast-sync catch-up window: `import_pruning_point_utxo_set` clears (does not seed) the
    //    windowed-production index, on the assumption the whole `ratio_reward_window` lies above
    //    the pruning point. That assumption breaks once the chain has advanced past `pruning_depth`
    //    beyond the ratio-reward activation height — every fresh fast sync then starts its index at
    //    zero while activation is already live, and needs a full window of catch-up before its
    //    computed ratio matches long-running nodes. This case self-expires: once
    //    `current_tip_index - seeded_at_index >= ratio_reward_window`, the index has organically
    //    refilled and verification re-enables itself with no operator action.
    // In every case the header `utxo_commitment` (checked first, always) already pins the resulting
    // UTXO set to the canonical chain, so trusting the block's coinbase outputs is safe.
    pub(super) is_archival: bool,
    // `KERYX_TRUST_COINBASE` operator override, read once at construction (it never changes at
    // runtime) so the per-block `trust_coinbase()` hot path avoids a fresh env lookup per coinbase.
    pub(super) trust_coinbase_env: bool,
}

impl VirtualStateProcessor {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        receiver: CrossbeamReceiver<VirtualStateProcessingMessage>,
        pruning_sender: CrossbeamSender<PruningProcessingMessage>,
        pruning_receiver: CrossbeamReceiver<PruningProcessingMessage>,
        thread_pool: Arc<ThreadPool>,
        params: &Params,
        db: Arc<DB>,
        storage: &Arc<ConsensusStorage>,
        services: &Arc<ConsensusServices>,
        pruning_lock: SessionLock,
        notification_root: Arc<ConsensusNotificationRoot>,
        counters: Arc<ProcessingCounters>,
        mining_rules: Arc<MiningRules>,
        is_archival: bool,
    ) -> Self {
        // Archival nodes retain blocks below the pruning point, so their windowed-production fold
        // subtracts leaving blocks that pruned nodes (the canonical majority) never see — diverging
        // the ratio/tier coinbase. They therefore MUST trust it (the header utxo_commitment still
        // pins the resulting UTXO set). Auto-enabled by `--archival`; `KERYX_TRUST_COINBASE=1` also
        // forces it on. See field doc.
        let trust_coinbase_env = std::env::var("KERYX_TRUST_COINBASE").is_ok();
        if is_archival || trust_coinbase_env {
            debug!(
                "Ratio/tier coinbase verification DISABLED ({}) — following the chain on UTXO-commitment trust only.",
                if is_archival { "archival node" } else { "KERYX_TRUST_COINBASE set" }
            );
        }
        Self {
            receiver,
            pruning_sender,
            pruning_receiver,
            thread_pool,

            genesis: params.genesis.clone(),
            max_block_parents: params.max_block_parents(),
            mergeset_size_limit: params.mergeset_size_limit(),

            db,
            statuses_store: storage.statuses_store.clone(),
            headers_store: storage.headers_store.clone(),
            ghostdag_store: storage.ghostdag_store.clone(),
            daa_excluded_store: storage.daa_excluded_store.clone(),
            block_transactions_store: storage.block_transactions_store.clone(),
            pom_tier_store: storage.pom_tier_store.clone(),
            pruning_point_store: storage.pruning_point_store.clone(),
            past_pruning_points_store: storage.past_pruning_points_store.clone(),
            body_tips_store: storage.body_tips_store.clone(),
            depth_store: storage.depth_store.clone(),
            selected_chain_store: storage.selected_chain_store.clone(),
            production_index_seed_store: storage.production_index_seed_store.clone(),
            pruning_samples_store: storage.pruning_samples_store.clone(),
            utxo_diffs_store: storage.utxo_diffs_store.clone(),
            utxo_multisets_store: storage.utxo_multisets_store.clone(),
            acceptance_data_store: storage.acceptance_data_store.clone(),
            address_balance_store: storage.address_balance_store.clone(),
            age_buckets_store: storage.age_buckets_store.clone(),
            maturation_queue_store: storage.maturation_queue_store.clone(),
            windowed_production_prefix_store: storage.windowed_production_prefix_store.clone(),
            block_production_cache: parking_lot::RwLock::new(BlockHashMap::default()),
            virtual_stores: storage.virtual_stores.clone(),
            pruning_meta_stores: storage.pruning_meta_stores.clone(),
            lkg_virtual_state: storage.lkg_virtual_state.clone(),

            block_window_cache_for_difficulty: storage.block_window_cache_for_difficulty.clone(),
            block_window_cache_for_past_median_time: storage.block_window_cache_for_past_median_time.clone(),

            ghostdag_manager: services.ghostdag_manager.clone(),
            reachability_service: services.reachability_service.clone(),
            relations_service: services.relations_service.clone(),
            dag_traversal_manager: services.dag_traversal_manager.clone(),
            window_manager: services.window_manager.clone(),
            coinbase_manager: services.coinbase_manager.clone(),
            transaction_validator: services.transaction_validator.clone(),
            pruning_point_manager: services.pruning_point_manager.clone(),
            parents_manager: services.parents_manager.clone(),
            depth_manager: services.depth_manager.clone(),

            pruning_lock,
            notification_root,
            counters,
            _mining_rules: mining_rules,
            ai_response_store: storage.ai_response_store.clone(),

            model_cap_enforcement_activation: params.model_cap_enforcement_activation,
            inference_reward_minimums: params.inference_reward_minimums,

            opoi_v2_activation: params.opoi_v2_activation,
            inference_reward_minimums_v2: params.inference_reward_minimums_v2,

            inference_min_h2_activation: params.inference_min_h2_activation,
            inference_reward_minimums_v2_h2: params.inference_reward_minimums_v2_h2,

            pow_salt_v2_activation: params.pow_salt_v2_activation,
            pow_salt_v4_activation: params.pow_salt_v4_activation,
            pom_activation: params.pom_activation,
            very_light_activation: params.very_light_activation,
            ratio_reward_activation: params.ratio_reward_activation,
            pom_level_activation: params.pom_level_activation,
            ratio_verification_activation: params.ratio_verification_activation,
            ratio_reward_window: params.ratio_reward_window,
            ratio_reward_window_daa: params.ratio_reward_window_daa,
            coin_age_activation: params.coin_age_activation,
            coin_age_maturity_w: params.coin_age_maturity_w,
            coin_age_retention: params.finality_depth(),
            is_archival,
            trust_coinbase_env,
        }
    }

    /// Whether ratio/tier coinbase verification should be skipped for the chain we're currently
    /// following. See the field doc on `is_archival` for the three ORed conditions. Re-evaluated on
    /// every call (cheap: one cached env lookup + at most one store read) rather than fixed at
    /// construction, so the fast-sync catch-up relaxation auto-expires once the window refills.
    pub(super) fn trust_coinbase(&self) -> bool {
        self.is_archival || self.trust_coinbase_env || self.in_production_catchup_window()
    }

    /// True while we're still inside our own post-import catch-up window: fewer than
    /// `ratio_reward_window` selected-chain blocks have been committed since the last
    /// pruning-point UTXO import cleared the windowed-production prefix index. `false` if the node
    /// has never imported a snapshot (built from genesis — no gap to begin with).
    fn in_production_catchup_window(&self) -> bool {
        let seeded_at = match self.production_index_seed_store.read().get_optional() {
            Some(idx) => idx,
            None => return false,
        };
        let tip_idx = match self.selected_chain_store.read().get_tip() {
            Ok((idx, _)) => idx,
            Err(_) => return false,
        };
        tip_idx.saturating_sub(seeded_at) < self.ratio_reward_window
    }

    pub fn worker(self: &Arc<Self>) {
        'outer: while let Ok(msg) = self.receiver.recv() {
            if msg.is_exit_message() {
                break;
            }

            // Once a task arrived, collect all pending tasks from the channel.
            // This is done since virtual processing is not a per-block
            // operation, so it benefits from max available info

            let messages: Vec<VirtualStateProcessingMessage> = std::iter::once(msg).chain(self.receiver.try_iter()).collect();
            trace!("virtual processor received {} tasks", messages.len());

            self.resolve_virtual();

            let statuses_read = self.statuses_store.read();
            for msg in messages {
                match msg {
                    VirtualStateProcessingMessage::Exit => break 'outer,
                    VirtualStateProcessingMessage::Process(task, virtual_state_result_transmitter) => {
                        // We don't care if receivers were dropped
                        let _ = virtual_state_result_transmitter.send(Ok(statuses_read.get(task.block().hash()).unwrap()));
                    }
                };
            }
        }

        // Pass the exit signal on to the following processor
        self.pruning_sender.send(PruningProcessingMessage::Exit).unwrap();
    }

    fn resolve_virtual(self: &Arc<Self>) {
        let pruning_point = self.pruning_point_store.read().pruning_point().unwrap();
        let virtual_read = self.virtual_stores.upgradable_read();
        let prev_state = virtual_read.state.get().unwrap();
        let finality_point = self.virtual_finality_point(&prev_state.ghostdag_data, pruning_point);

        // PRUNE SAFETY: in order to avoid locking the prune lock throughout virtual resolving we make sure
        // to only process blocks in the future of the finality point (F) which are never pruned (since finality depth << pruning depth).
        // This is justified since:
        //      1. Tips which are not in the future of F definitely don't have F on their chain
        //         hence cannot become the next sink (due to finality violation).
        //      2. Such tips cannot be merged by virtual since they are violating the merge depth
        //         bound (merge depth <= finality depth).
        // (both claims are true by induction for any block in their past as well)
        let prune_guard = self.pruning_lock.blocking_read();
        let tips = self
            .body_tips_store
            .read()
            .get()
            .unwrap()
            .read()
            .iter()
            .copied()
            .filter(|&h| self.reachability_service.is_dag_ancestor_of(finality_point, h))
            .collect_vec();
        drop(prune_guard);
        let prev_sink = prev_state.ghostdag_data.selected_parent;
        let mut accumulated_diff = prev_state.utxo_diff.clone().to_reversed();

        let (new_sink, virtual_parent_candidates) =
            self.sink_search_algorithm(&virtual_read, &mut accumulated_diff, prev_sink, tips, finality_point, pruning_point);
        let (virtual_parents, virtual_ghostdag_data) = self.pick_virtual_parents(new_sink, virtual_parent_candidates, pruning_point);
        assert_eq!(virtual_ghostdag_data.selected_parent, new_sink);

        let sink_multiset = self.utxo_multisets_store.get(new_sink).unwrap();
        let chain_path = self.dag_traversal_manager.calculate_chain_path(prev_sink, new_sink, None);
        let sink_ghostdag_data = Lazy::new(|| self.ghostdag_store.get_data(new_sink).unwrap());
        // Cache the DAA and Median time windows of the sink for future use, as well as prepare for virtual's window calculations
        self.cache_sink_windows(new_sink, prev_sink, &sink_ghostdag_data);

        let new_virtual_state = self
            .calculate_and_commit_virtual_state(
                virtual_read,
                virtual_parents,
                virtual_ghostdag_data,
                sink_multiset,
                &mut accumulated_diff,
                &chain_path,
            )
            .expect("all possible rule errors are unexpected here");

        let compact_sink_ghostdag_data = if let Some(sink_ghostdag_data) = Lazy::get(&sink_ghostdag_data) {
            // If we had to retrieve the full data, we convert it to compact
            sink_ghostdag_data.to_compact()
        } else {
            // Else we query the compact data directly.
            self.ghostdag_store.get_compact_data(new_sink).unwrap()
        };

        // Update the pruning processor about the virtual state change
        // Empty the channel before sending the new message. If pruning processor is busy, this step makes sure
        // the internal channel does not grow with no need (since we only care about the most recent message)
        let _consume = self.pruning_receiver.try_iter().count();
        self.pruning_sender.send(PruningProcessingMessage::Process { sink_ghostdag_data: compact_sink_ghostdag_data }).unwrap();

        // Emit notifications
        let accumulated_diff = Arc::new(accumulated_diff);
        let virtual_parents = Arc::new(new_virtual_state.parents.clone());
        self.notification_root
            .notify(Notification::NewBlockTemplate(NewBlockTemplateNotification {}))
            .expect("expecting an open unbounded channel");
        self.notification_root
            .notify(Notification::UtxosChanged(UtxosChangedNotification::new(accumulated_diff, virtual_parents)))
            .expect("expecting an open unbounded channel");
        self.notification_root
            .notify(Notification::SinkBlueScoreChanged(SinkBlueScoreChangedNotification::new(compact_sink_ghostdag_data.blue_score)))
            .expect("expecting an open unbounded channel");
        self.notification_root
            .notify(Notification::VirtualDaaScoreChanged(VirtualDaaScoreChangedNotification::new(new_virtual_state.daa_score)))
            .expect("expecting an open unbounded channel");
        if self.notification_root.has_subscription(EventType::VirtualChainChanged) {
            // check for subscriptions before the heavy lifting
            let added_chain_blocks_acceptance_data =
                chain_path.added.iter().copied().map(|added| self.acceptance_data_store.get(added).unwrap()).collect_vec();
            self.notification_root
                .notify(Notification::VirtualChainChanged(VirtualChainChangedNotification::new(
                    chain_path.added.into(),
                    chain_path.removed.into(),
                    Arc::new(added_chain_blocks_acceptance_data),
                )))
                .expect("expecting an open unbounded channel");
        }
    }

    pub(crate) fn virtual_finality_point(&self, virtual_ghostdag_data: &GhostdagData, pruning_point: Hash) -> Hash {
        let finality_point = self.depth_manager.calc_finality_point(virtual_ghostdag_data, pruning_point);
        if self.reachability_service.is_chain_ancestor_of(pruning_point, finality_point) {
            finality_point
        } else {
            // At the beginning of IBD when virtual finality point might be below the pruning point
            // or disagreeing with the pruning point chain, we take the pruning point itself as the finality point
            pruning_point
        }
    }

    /// Calculates the UTXO state of `to` starting from the state of `from`.
    /// The provided `diff` is assumed to initially hold the UTXO diff of `from` from virtual.
    /// The function returns the top-most UTXO-valid block on `chain(to)` which is ideally
    /// `to` itself (with the exception of returning `from` if `to` is already known to be UTXO disqualified).
    /// When returning it is guaranteed that `diff` holds the diff of the returned block from virtual
    fn calculate_utxo_state_relatively(&self, stores: &VirtualStores, diff: &mut UtxoDiff, from: Hash, to: Hash) -> Hash {
        // Avoid reorging if disqualified status is already known
        if self.statuses_store.read().get(to).unwrap() == StatusDisqualifiedFromChain {
            return from;
        }

        let mut split_point: Option<Hash> = None;

        // Walk down to the reorg split point
        for current in self.reachability_service.default_backward_chain_iterator(from) {
            if self.reachability_service.is_chain_ancestor_of(current, to) {
                split_point = Some(current);
                break;
            }

            let mergeset_diff = self.utxo_diffs_store.get(current).unwrap();
            // Apply the diff in reverse
            diff.with_diff_in_place(&mergeset_diff.as_reversed()).unwrap();
        }

        let split_point = split_point.expect("chain iterator was expected to reach the reorg split point");
        debug!("VIRTUAL PROCESSOR, found split point: {split_point}");

        // A variable holding the most recent UTXO-valid block on `chain(to)` (note that it's maintained such
        // that 'diff' is always its UTXO diff from virtual)
        let mut diff_point = split_point;

        // Walk back up to the new virtual selected parent candidate
        let mut chain_block_counter = 0;
        let mut chain_disqualified_counter = 0;
        for (selected_parent, current) in self.reachability_service.forward_chain_iterator(split_point, to, true).tuple_windows() {
            if selected_parent != diff_point {
                // This indicates that the selected parent is disqualified, propagate up and continue
                let statuses_guard = self.statuses_store.upgradable_read();
                if statuses_guard.get(current).unwrap() != StatusDisqualifiedFromChain {
                    RwLockUpgradableReadGuard::upgrade(statuses_guard).set(current, StatusDisqualifiedFromChain).unwrap();
                    chain_disqualified_counter += 1;
                }
                continue;
            }

            match self.utxo_diffs_store.get(current) {
                Ok(mergeset_diff) => {
                    diff.with_diff_in_place(mergeset_diff.deref()).unwrap();
                    diff_point = current;
                }
                Err(StoreError::KeyNotFound(_)) => {
                    if self.statuses_store.read().get(current).unwrap() == StatusDisqualifiedFromChain {
                        // Current block is already known to be disqualified
                        continue;
                    }

                    let header = self.headers_store.get_header(current).unwrap();
                    let mergeset_data = self.ghostdag_store.get_data(current).unwrap();
                    let pov_daa_score = header.daa_score;

                    let selected_parent_multiset_hash = self.utxo_multisets_store.get(selected_parent).unwrap();
                    let selected_parent_utxo_view = (&stores.utxo_set).compose(&*diff);

                    let mut ctx = UtxoProcessingContext::new(mergeset_data.into(), selected_parent_multiset_hash);

                    self.calculate_utxo_state(&mut ctx, &selected_parent_utxo_view, pov_daa_score);
                    // `diff` is the diff from the committed virtual to `current`'s selected parent; pass
                    // it so the ratio-reward bracket can be evaluated at `current`'s view (see
                    // `ratio_bps_by_block`): balance index (at virtual) + this diff + current's mergeset diff.
                    let res = self.verify_expected_utxo_state(&mut ctx, &selected_parent_utxo_view, &header, &*diff);

                    if let Err(rule_error) = res {
                        info!("Block {} is disqualified from virtual chain: {}", current, rule_error);
                        self.statuses_store.write().set(current, StatusDisqualifiedFromChain).unwrap();
                        chain_disqualified_counter += 1;
                    } else {
                        debug!("VIRTUAL PROCESSOR, UTXO validated for {current}");

                        // Accumulate the diff
                        diff.with_diff_in_place(&ctx.mergeset_diff).unwrap();
                        // Update the diff point
                        diff_point = current;
                        // Commit UTXO data for current chain block
                        self.commit_utxo_state(
                            current,
                            ctx.mergeset_diff,
                            ctx.multiset_hash,
                            ctx.mergeset_acceptance_data,
                            ctx.pruning_sample_from_pov.expect("verified"),
                        );
                        // Count the number of UTXO-processed chain blocks
                        chain_block_counter += 1;
                    }
                }
                Err(err) => panic!("unexpected error {err}"),
            }
        }
        // Report counters
        self.counters.chain_block_counts.fetch_add(chain_block_counter, Ordering::Relaxed);
        if chain_disqualified_counter > 0 {
            self.counters.chain_disqualified_counts.fetch_add(chain_disqualified_counter, Ordering::Relaxed);
        }

        diff_point
    }

    fn commit_utxo_state(
        &self,
        current: Hash,
        mergeset_diff: UtxoDiff,
        multiset: MuHash,
        acceptance_data: AcceptanceData,
        pruning_sample_from_pov: Hash,
    ) {
        let mut batch = WriteBatch::default();
        self.utxo_diffs_store.insert_batch(&mut batch, current, Arc::new(mergeset_diff)).unwrap();
        self.utxo_multisets_store.insert_batch(&mut batch, current, multiset).unwrap();
        self.acceptance_data_store.insert_batch(&mut batch, current, Arc::new(acceptance_data)).unwrap();
        // Note we call idempotent since this field can be populated during IBD with headers proof
        self.pruning_samples_store.insert_batch(&mut batch, current, pruning_sample_from_pov).idempotent().unwrap();
        let write_guard = self.statuses_store.set_batch(&mut batch, current, StatusUTXOValid).unwrap();
        self.db.write(batch).unwrap();
        // Calling the drops explicitly after the batch is written in order to avoid possible errors.
        drop(write_guard);
    }

    fn calculate_and_commit_virtual_state(
        &self,
        virtual_read: RwLockUpgradableReadGuard<'_, VirtualStores>,
        virtual_parents: Vec<Hash>,
        virtual_ghostdag_data: GhostdagData,
        selected_parent_multiset: MuHash,
        accumulated_diff: &mut UtxoDiff,
        chain_path: &ChainPath,
    ) -> Result<Arc<VirtualState>, RuleError> {
        let new_virtual_state = self.calculate_virtual_state(
            &virtual_read,
            virtual_parents,
            virtual_ghostdag_data,
            selected_parent_multiset,
            accumulated_diff,
        )?;
        self.commit_virtual_state(virtual_read, new_virtual_state.clone(), accumulated_diff, chain_path);
        Ok(new_virtual_state)
    }

    pub(super) fn calculate_virtual_state(
        &self,
        virtual_stores: &VirtualStores,
        virtual_parents: Vec<Hash>,
        virtual_ghostdag_data: GhostdagData,
        selected_parent_multiset: MuHash,
        accumulated_diff: &mut UtxoDiff,
    ) -> Result<Arc<VirtualState>, RuleError> {
        let selected_parent_utxo_view = (&virtual_stores.utxo_set).compose(&*accumulated_diff);
        let mut ctx = UtxoProcessingContext::new((&virtual_ghostdag_data).into(), selected_parent_multiset);

        // Calc virtual DAA score, difficulty bits and past median time
        let virtual_daa_window = self.window_manager.block_daa_window(&virtual_ghostdag_data)?;
        let virtual_bits = self.window_manager.calculate_difficulty_bits(&virtual_ghostdag_data, &virtual_daa_window);
        let virtual_past_median_time = self.window_manager.calc_past_median_time(&virtual_ghostdag_data)?.0;

        // Calc virtual UTXO state relative to selected parent
        self.calculate_utxo_state(&mut ctx, &selected_parent_utxo_view, virtual_daa_window.daa_score);

        // Update the accumulated diff
        accumulated_diff.with_diff_in_place(&ctx.mergeset_diff).unwrap();

        // Build the new virtual state
        Ok(Arc::new(VirtualState::new(
            virtual_parents,
            virtual_daa_window.daa_score,
            virtual_bits,
            virtual_past_median_time,
            ctx.multiset_hash,
            ctx.mergeset_diff,
            ctx.accepted_tx_ids,
            ctx.mergeset_rewards,
            virtual_daa_window.mergeset_non_daa,
            virtual_ghostdag_data,
        )))
    }

    fn commit_virtual_state(
        &self,
        virtual_read: RwLockUpgradableReadGuard<'_, VirtualStores>,
        new_virtual_state: Arc<VirtualState>,
        accumulated_diff: &UtxoDiff,
        chain_path: &ChainPath,
    ) {
        let mut batch = WriteBatch::default();
        let mut virtual_write = RwLockUpgradableReadGuard::upgrade(virtual_read);
        let mut selected_chain_write = self.selected_chain_store.write();

        // Apply the accumulated diff to the virtual UTXO set
        virtual_write.utxo_set.write_diff_batch(&mut batch, accumulated_diff).unwrap();

        // Ratio-reward (Stage 2b): advance the balance index by the SAME diff, in the SAME batch,
        // so it stays byte-for-byte in lockstep with the virtual UTXO set (a torn write would let
        // the index drift). Ungated: it is a passive aggregate with no consensus effect until
        // `ratio_bps_by_block` reads it post-activation; maintaining it from genesis keeps it exact
        // for from-genesis nodes (fast-sync reconstruction from the pruning-point snapshot is 2b-3).
        self.apply_balance_diff(&mut batch, accumulated_diff);

        // Coin-age (v3): sweep the maturation queue FIRST (promote coins whose age crossed W by
        // the new virtual score), then advance the bucket index by the SAME diff, in the SAME
        // batch (lockstep, like the balance index above). Ungated passive aggregates — read only
        // at/after `coin_age_activation`. A `true` sweep result = the virtual score moved below
        // the promotion watermark (deep reorg): rebuild everything from the UTXO set post-commit.
        let coin_age_needs_rebuild = self.sweep_maturation_queue(&mut batch, new_virtual_state.daa_score);
        self.apply_age_diff(&mut batch, accumulated_diff, new_virtual_state.daa_score);

        // Ratio-reward (Stage 2b-2b): advance the prefix-sum production index along the SAME chain
        // path, in the SAME batch, BEFORE `apply_changes` mutates the selected chain — so the index
        // reads the pre-change chain (its current anchor) and stays in lockstep with it. Pure function
        // of the chain (no path dependence) — see `advance_production_prefix`; read by `ratio_bps_by_block`.
        self.advance_production_prefix(&mut batch, chain_path, &*selected_chain_write);

        // Update virtual state
        virtual_write.state.set_batch(&mut batch, new_virtual_state).unwrap();

        // Update the virtual selected chain
        selected_chain_write.apply_changes(&mut batch, chain_path).unwrap();

        // Flush the batch changes
        self.db.write(batch).unwrap();

        // Calling the drops explicitly after the batch is written in order to avoid possible errors.
        drop(virtual_write);
        drop(selected_chain_write);

        // Deep-reorg guard (see `sweep_maturation_queue`): re-derive the coin-age state exactly
        // from the just-committed UTXO set. Rare to nonexistent in practice (the virtual daa
        // score is effectively monotone); correctness beats incremental demotion here.
        if coin_age_needs_rebuild {
            self.rebuild_age_buckets_index();
        }
    }

    /// Rebuild the ratio-reward BALANCE index (`address_balance_store`) directly from the current
    /// virtual UTXO set — the authoritative source — at startup. The balance index is the ratio
    /// numerator (`Σ unspent amount` per payout SPK). A from-genesis node maintains it incrementally
    /// and a fast-synced node seeds it from the pruning-point UTXO snapshot at import; but a datadir
    /// restored from a snapshot built by an older binary (or before this index existed) carries a
    /// STALE balance index that no longer matches its own UTXO set — making the ratio numerator differ
    /// across nodes (the cause of the post-relaunch balance divergence: e.g. 81 G vs 242 T for the same
    /// SPK). Recomputing it from the UTXO set makes every node's numerator the canonical `Σ UTXO`, so
    /// the same SPK yields the same balance everywhere. Run on every start; incremental lockstep
    /// maintenance (`apply_balance_diff`) carries it forward exactly from this canonical baseline.
    pub(crate) fn rebuild_address_balance_index(&self) {
        let virtual_read = self.virtual_stores.read();
        let mut balances: HashMap<ScriptPublicKey, u64> = HashMap::new();
        for item in virtual_read.utxo_set.iterator() {
            let (_, entry) = item.unwrap();
            let acc = balances.entry(entry.script_public_key.clone()).or_default();
            *acc = acc.saturating_add(entry.amount);
        }
        let n = balances.len();
        info!("address-balance rebuild: recomputing the ratio-reward balance index from the UTXO set ({} addresses)...", n);
        let mut batch = WriteBatch::default();
        self.address_balance_store.clear(&mut batch).unwrap();
        for (spk, amount) in balances {
            self.address_balance_store.set_batch(&mut batch, &spk, amount).unwrap();
        }
        self.db.write(batch).unwrap();
        info!("address-balance rebuild: done — {} addresses written (canonical Σ UTXO).", n);
    }

    /// Rebuild the coin-age BUCKET index (`age_buckets_store`) from the current virtual UTXO set at
    /// startup — the exact mirror of `rebuild_address_balance_index` for the v3 aggregates. Every
    /// UTXO carries its `effective_daa` anchor, so classifying each coin against the CURRENT virtual
    /// score re-derives `{b_mat, b_imm, a_imm}` exactly — which also absorbs any coin that matured
    /// in place since the last run (until the maturation-queue promotions maintain that in-flight).
    pub(crate) fn rebuild_age_buckets_index(&self) {
        let virtual_read = self.virtual_stores.read();
        // A freshly-created (staging) consensus has no virtual state yet — skip, like the
        // windowed-production rebuild does. The index is seeded once the state exists: at
        // pruning-point import for fast sync (`import_pruning_point_utxo_set`) or next startup.
        let Some(state) = virtual_read.state.get().optional().unwrap() else {
            warn!("age-buckets rebuild: virtual state is not initialized yet; skipping");
            return;
        };
        let daa_score = state.daa_score;
        let mature_bound = daa_score.saturating_sub(self.coin_age_maturity_w);
        let mut buckets: HashMap<ScriptPublicKey, AgeBuckets> = HashMap::new();
        for item in virtual_read.utxo_set.iterator() {
            let (_, entry) = item.unwrap();
            let b = buckets.entry(entry.script_public_key.clone()).or_default();
            if entry.effective_daa <= mature_bound {
                b.b_mat = b.b_mat.saturating_add(entry.amount);
            } else {
                b.b_imm = b.b_imm.saturating_add(entry.amount);
                b.a_imm = b.a_imm.saturating_add(entry.amount as u128 * entry.effective_daa as u128);
            }
        }
        let n = buckets.len();
        info!("age-buckets rebuild: recomputing the coin-age bucket index from the UTXO set ({} addresses)...", n);
        let mut batch = WriteBatch::default();
        self.age_buckets_store.clear(&mut batch).unwrap();
        self.maturation_queue_store.clear(&mut batch).unwrap();
        for (spk, b) in buckets {
            self.age_buckets_store.set_batch(&mut batch, &spk, b).unwrap();
        }
        // Reseed the maturation queue (immature coins only) and pin the watermark at the score
        // the classification was taken at — the sweep resumes incrementally from here.
        let mut queued = 0u64;
        for item in virtual_read.utxo_set.iterator() {
            let (outpoint, entry) = item.unwrap();
            if entry.effective_daa > mature_bound {
                let e = MaturationEntry {
                    script_public_key: entry.script_public_key.clone(),
                    amount: entry.amount,
                    anchor: entry.effective_daa,
                };
                self.maturation_queue_store.insert_batch(&mut batch, entry.effective_daa + self.coin_age_maturity_w, &outpoint, e).unwrap();
                queued += 1;
            }
        }
        self.maturation_queue_store.set_watermark_batch(&mut batch, daa_score).unwrap();
        self.db.write(batch).unwrap();
        info!("age-buckets rebuild: done — {} addresses written, {} immature coins queued (split at d−W).", n, queued);
    }

    /// One-time from-chain build of the gold-standard prefix-sum production index (run at startup when
    /// the store is empty — a datadir predating it, or a fresh prefix). Walks the selected chain over a
    /// generous range above any practical reorg depth, accumulating each producer SPK's cumulative
    /// (baseline 0 at the range start) and writing one entry per production index. Indices below the
    /// pruning point on a pruned node are simply unavailable and skipped: `windowed` is a *difference*,
    /// so the absolute baseline is immaterial, and `W < pruning_depth` keeps every queried index inside
    /// the built range. Pure function of the chain ⇒ every node derives identical windowed values.
    pub(crate) fn rebuild_windowed_production_prefix_index(&self) {
        let sc = self.selected_chain_store.read();
        let tip_idx = match sc.get_tip() {
            Ok((idx, _)) => idx,
            Err(_) => {
                warn!("windowed-production prefix build: no selected-chain tip; skipping");
                return;
            }
        };
        let lo = tip_idx.saturating_sub(2 * self.ratio_reward_window).max(1);
        info!("windowed-production prefix build: deriving canonical index over selected-chain [{}, {}]...", lo, tip_idx);
        let mut batch = WriteBatch::default();
        self.windowed_production_prefix_store.clear(&mut batch);
        let mut cum: HashMap<ScriptPublicKey, u64> = HashMap::new();
        let mut written = 0u64;
        for i in lo..=tip_idx {
            if let Ok(h) = sc.get_by_index(i) {
                // Era-aware (H3): per paid mergeset blue at/after `pom_level_activation`, the chain
                // block's own producer below — identical to what lockstep maintenance appended, so a
                // fresh rebuild and an incrementally-maintained index derive byte-identical values.
                for (spk, cut) in self.block_productions(h) {
                    let c = cum.entry(spk.clone()).or_insert(0);
                    *c += cut;
                    self.windowed_production_prefix_store.put_cumulative(&mut batch, &spk, i, *c);
                    written += 1;
                }
            }
        }
        self.db.write(batch).unwrap();
        info!("windowed-production prefix build: done — {} entries written (canonical baseline).", written);
    }

    /// Startup hook: build the prefix-sum index from the chain only when it is empty (a datadir
    /// predating this store, or a fresh prefix). Once populated it stays current via lockstep
    /// maintenance, so this is a no-op on subsequent boots.
    pub(crate) fn rebuild_windowed_production_prefix_index_on_start(&self) {
        if self.windowed_production_prefix_store.is_empty() {
            self.rebuild_windowed_production_prefix_index();
        }
    }

    /// Caches the DAA and Median time windows of the sink block (if needed). Following, virtual's window calculations will
    /// naturally hit the cache finding the sink's windows and building upon them.
    fn cache_sink_windows(&self, new_sink: Hash, prev_sink: Hash, sink_ghostdag_data: &impl Deref<Target = Arc<GhostdagData>>) {
        // We expect that the `new_sink` is cached (or some close-enough ancestor thereof) if it is equal to the `prev_sink`,
        // Hence we short-circuit the check of the keys in such cases, thereby reducing the access of the read-lock
        if new_sink != prev_sink {
            // this is only important for ibd performance, as we incur expensive cache misses otherwise.
            // this occurs because we cannot rely on header processing to pre-cache in this scenario.
            if !self.block_window_cache_for_difficulty.contains_key(&new_sink) {
                self.block_window_cache_for_difficulty
                    .insert(new_sink, self.window_manager.block_daa_window(sink_ghostdag_data.deref()).unwrap().window);
            };

            if !self.block_window_cache_for_past_median_time.contains_key(&new_sink) {
                self.block_window_cache_for_past_median_time
                    .insert(new_sink, self.window_manager.calc_past_median_time(sink_ghostdag_data.deref()).unwrap().1);
            };
        }
    }

    /// Returns the max number of tips to consider as virtual parents in a single virtual resolve operation.
    ///
    /// Guaranteed to be `>= self.max_block_parents`
    fn max_virtual_parent_candidates(&self, max_block_parents: usize) -> usize {
        // Limit to max_block_parents x 3 candidates. This way we avoid going over thousands of tips when the network isn't healthy.
        // There's no specific reason for a factor of 3, and its not a consensus rule, just an estimation for reducing the amount
        // of candidates considered.
        max_block_parents * 3
    }

    /// Searches for the next valid sink block (SINK = Virtual selected parent). The search is performed
    /// in the inclusive past of `tips`.
    /// The provided `diff` is assumed to initially hold the UTXO diff of `prev_sink` from virtual.
    /// The function returns with `diff` being the diff of the new sink from previous virtual.
    /// In addition to the found sink the function also returns a queue of additional virtual
    /// parent candidates ordered in descending blue work order.
    pub(super) fn sink_search_algorithm(
        &self,
        stores: &VirtualStores,
        diff: &mut UtxoDiff,
        prev_sink: Hash,
        tips: Vec<Hash>,
        finality_point: Hash,
        pruning_point: Hash,
    ) -> (Hash, VecDeque<Hash>) {
        // TODO (relaxed): additional tests

        let mut heap = tips
            .into_iter()
            .map(|block| SortableBlock { hash: block, blue_work: self.ghostdag_store.get_blue_work(block).unwrap() })
            .collect::<BinaryHeap<_>>();

        // The initial diff point is the previous sink
        let mut diff_point = prev_sink;

        // We maintain the following invariant: `heap` is an antichain.
        // It holds at step 0 since tips are an antichain, and remains through the loop
        // since we check that every pushed block is not in the past of current heap
        // (and it can't be in the future by induction)
        loop {
            let candidate = heap.pop().expect("valid sink must exist").hash;
            let candidate_status = self.statuses_store.read().get(candidate).unwrap();
            // Header-only blocks have no body => no UTXO state, and an abandoned higher-blue-work
            // overhang (e.g. a salt fork left in the datadir after a relaunch) can form a long
            // header-only branch. Skip them WITHOUT walking their parents — doing so would traverse
            // the whole overhang on every virtual resolve, an O(branch length) blow-up.
            if candidate_status.is_header_only() {
                continue;
            }
            // Disqualified blocks have a body but known-invalid UTXO state: they can never be the
            // sink. PoM-GATED behavior change (deliberately fork-gated, like every other consensus
            // change since the storm — no modification to the live pre-PoM chain):
            //   • pre-`pom_activation`: legacy — skip WITHOUT walking parents (the historically
            //     deployed v1.2.6 behavior; preserves exact sink resolution for the current chain).
            //   • from `pom_activation`: still skip the (known-failing) UTXO computation, but WALK
            //     this block's parents, because the valid sink may lie BELOW a disqualified block
            //     (a valid block built on top of one). A disqualified branch is bounded (real
            //     bodies), so this is not the header-only overhang case. Fixes a sink-search stall.
            if candidate_status == StatusDisqualifiedFromChain {
                if !self.pom_activation.is_active(self.headers_store.get_daa_score(candidate).unwrap()) {
                    continue;
                }
                // pom-active: fall through to the parent walk below (no UTXO computation).
            } else if self.reachability_service.is_chain_ancestor_of(finality_point, candidate) {
                diff_point = self.calculate_utxo_state_relatively(stores, diff, diff_point, candidate);
                if diff_point == candidate {
                    // This indicates that candidate has valid UTXO state and that `diff` represents its diff from virtual

                    // All blocks with lower blue work than filtering_root are:
                    // 1. not in its future (bcs blue work is monotonic),
                    // 2. will be removed eventually by the bounded merge check.
                    // Hence as an optimization we prefer removing such blocks in advance to allow valid tips to be considered.
                    let filtering_root = self.depth_store.merge_depth_root(candidate).unwrap();
                    let filtering_blue_work = self.ghostdag_store.get_blue_work(filtering_root).unwrap_or_default();
                    return (
                        candidate,
                        heap.into_sorted_iter().take_while(|s| s.blue_work >= filtering_blue_work).map(|s| s.hash).collect(),
                    );
                } else {
                    debug!("Block candidate {} has invalid UTXO state and is ignored from Virtual chain.", candidate)
                }
            } else if finality_point != pruning_point {
                // `finality_point == pruning_point` indicates we are at IBD start hence no warning required
                warn!("Finality Violation Detected. Block {} violates finality and is ignored from Virtual chain.", candidate);
            }
            // PRUNE SAFETY: see comment within [`resolve_virtual`]
            let prune_guard = self.pruning_lock.blocking_read();
            for parent in self.relations_service.get_parents(candidate).unwrap().iter().copied() {
                if self.reachability_service.is_dag_ancestor_of(finality_point, parent)
                    && !self.reachability_service.is_dag_ancestor_of_any(parent, &mut heap.iter().map(|sb| sb.hash))
                {
                    heap.push(SortableBlock { hash: parent, blue_work: self.ghostdag_store.get_blue_work(parent).unwrap() });
                }
            }
            drop(prune_guard);
        }
    }

    /// Picks the virtual parents according to virtual parent selection pruning constrains.
    /// Assumes:
    ///     1. `selected_parent` is a UTXO-valid block
    ///     2. `candidates` are an antichain ordered in descending blue work order
    ///     3. `candidates` do not contain `selected_parent` and `selected_parent.blue work > max(candidates.blue_work)`  
    pub(super) fn pick_virtual_parents(
        &self,
        selected_parent: Hash,
        mut candidates: VecDeque<Hash>,
        pruning_point: Hash,
    ) -> (Vec<Hash>, GhostdagData) {
        // TODO (relaxed): additional tests

        // Mergeset increasing might traverse DAG areas which are below the finality point and which theoretically
        // can borderline with pruned data, hence we acquire the prune lock to ensure data consistency. Note that
        // the final selected mergeset can never be pruned (this is the essence of the prunality proof), however
        // we might touch such data prior to validating the bounded merge rule. All in all, this function is short
        // enough so we avoid making further optimizations
        let _prune_guard = self.pruning_lock.blocking_read();
        let max_block_parents = self.max_block_parents as usize;
        let mergeset_size_limit = self.mergeset_size_limit;
        let max_candidates = self.max_virtual_parent_candidates(max_block_parents);

        // Prioritize half the blocks with highest blue work and pick the rest randomly to ensure diversity between nodes
        if candidates.len() > max_candidates {
            // make_contiguous should be a no op since the deque was just built
            let slice = candidates.make_contiguous();

            // Keep slice[..max_block_parents / 2] as is, choose max_candidates - max_block_parents / 2 in random
            // from the remainder of the slice while swapping them to slice[max_block_parents / 2..max_candidates].
            //
            // Inspired by rand::partial_shuffle (which lacks the guarantee on chosen elements location).
            for i in max_block_parents / 2..max_candidates {
                let j = rand::thread_rng().gen_range(i..slice.len()); // i < max_candidates < slice.len()
                slice.swap(i, j);
            }

            // Truncate the unchosen elements
            candidates.truncate(max_candidates);
        } else if candidates.len() > max_block_parents / 2 {
            // Fallback to a simpler algo in this case
            candidates.make_contiguous()[max_block_parents / 2..].shuffle(&mut rand::thread_rng());
        }

        let mut virtual_parents = Vec::with_capacity(min(max_block_parents, candidates.len() + 1));
        virtual_parents.push(selected_parent);
        let mut mergeset_size = 1; // Count the selected parent

        // Try adding parents as long as mergeset size and number of parents limits are not reached
        while let Some(candidate) = candidates.pop_front() {
            if mergeset_size >= mergeset_size_limit || virtual_parents.len() >= max_block_parents {
                break;
            }
            match self.mergeset_increase(&virtual_parents, candidate, mergeset_size_limit - mergeset_size) {
                MergesetIncreaseResult::Accepted { increase_size } => {
                    mergeset_size += increase_size;
                    virtual_parents.push(candidate);
                }
                MergesetIncreaseResult::Rejected { new_candidate } => {
                    // If we already have a candidate in the past of new candidate then skip.
                    if self.reachability_service.is_any_dag_ancestor(&mut candidates.iter().copied(), new_candidate) {
                        continue; // TODO (optimization): not sure this check is needed if candidates invariant as antichain is kept
                    }
                    // Remove all candidates which are in the future of the new candidate
                    candidates.retain(|&h| !self.reachability_service.is_dag_ancestor_of(new_candidate, h));
                    candidates.push_back(new_candidate);
                }
            }
        }
        assert!(mergeset_size <= mergeset_size_limit);
        assert!(virtual_parents.len() <= max_block_parents);
        self.remove_bounded_merge_breaking_parents(virtual_parents, pruning_point)
    }

    fn mergeset_increase(&self, selected_parents: &[Hash], candidate: Hash, budget: u64) -> MergesetIncreaseResult {
        /*
        Algo:
            Traverse past(candidate) \setminus past(selected_parents) and make
            sure the increase in mergeset size is within the available budget
        */

        let candidate_parents = self.relations_service.get_parents(candidate).unwrap();
        let mut queue: VecDeque<_> = candidate_parents.iter().copied().collect();
        let mut visited: BlockHashSet = queue.iter().copied().collect();
        let mut mergeset_increase = 1u64; // Starts with 1 to count for the candidate itself

        while let Some(current) = queue.pop_front() {
            if self.reachability_service.is_dag_ancestor_of_any(current, &mut selected_parents.iter().copied()) {
                continue;
            }
            mergeset_increase += 1;
            if mergeset_increase > budget {
                return MergesetIncreaseResult::Rejected { new_candidate: current };
            }

            let current_parents = self.relations_service.get_parents(current).unwrap();
            for &parent in current_parents.iter() {
                if visited.insert(parent) {
                    queue.push_back(parent);
                }
            }
        }
        MergesetIncreaseResult::Accepted { increase_size: mergeset_increase }
    }

    fn remove_bounded_merge_breaking_parents(
        &self,
        mut virtual_parents: Vec<Hash>,
        current_pruning_point: Hash,
    ) -> (Vec<Hash>, GhostdagData) {
        let mut ghostdag_data = self.ghostdag_manager.ghostdag(&virtual_parents);
        let merge_depth_root = self.depth_manager.calc_merge_depth_root(&ghostdag_data, current_pruning_point);
        let mut kosherizing_blues: Option<Vec<Hash>> = None;
        let mut bad_reds = Vec::new();

        //
        // Note that the code below optimizes for the usual case where there are no merge-bound-violating blocks.
        //

        // Find red blocks violating the merge bound and which are not kosherized by any blue
        for red in ghostdag_data.mergeset_reds.iter().copied() {
            if self.reachability_service.is_dag_ancestor_of(merge_depth_root, red) {
                continue;
            }
            // Lazy load the kosherizing blocks since this case is extremely rare
            if kosherizing_blues.is_none() {
                kosherizing_blues = Some(self.depth_manager.kosherizing_blues(&ghostdag_data, merge_depth_root).collect());
            }
            if !self.reachability_service.is_dag_ancestor_of_any(red, &mut kosherizing_blues.as_ref().unwrap().iter().copied()) {
                bad_reds.push(red);
            }
        }

        if !bad_reds.is_empty() {
            // Remove all parents which lead to merging a bad red
            virtual_parents.retain(|&h| !self.reachability_service.is_any_dag_ancestor(&mut bad_reds.iter().copied(), h));
            // Recompute ghostdag data since parents changed
            ghostdag_data = self.ghostdag_manager.ghostdag(&virtual_parents);
        }

        (virtual_parents, ghostdag_data)
    }

    fn validate_mempool_transaction_impl(
        &self,
        mutable_tx: &mut MutableTransaction,
        virtual_utxo_view: &impl UtxoView,
        virtual_daa_score: u64,
        virtual_past_median_time: u64,
        args: &TransactionValidationArgs,
    ) -> TxResult<()> {
        self.transaction_validator.validate_tx_in_isolation(&mutable_tx.tx)?;
        self.transaction_validator.validate_tx_in_header_context_with_args(
            &mutable_tx.tx,
            virtual_daa_score,
            virtual_past_median_time,
        )?;
        self.validate_mempool_transaction_in_utxo_context(mutable_tx, virtual_utxo_view, virtual_daa_score, args)?;
        Ok(())
    }

    pub fn validate_mempool_transaction(&self, mutable_tx: &mut MutableTransaction, args: &TransactionValidationArgs) -> TxResult<()> {
        let virtual_read = self.virtual_stores.read();
        let virtual_state = virtual_read.state.get().unwrap();
        let virtual_utxo_view = &virtual_read.utxo_set;
        let virtual_daa_score = virtual_state.daa_score;
        let virtual_past_median_time = virtual_state.past_median_time;
        // Run within the thread pool since par_iter might be internally applied to inputs
        self.thread_pool.install(|| {
            self.validate_mempool_transaction_impl(mutable_tx, virtual_utxo_view, virtual_daa_score, virtual_past_median_time, args)
        })
    }

    pub fn validate_mempool_transactions_in_parallel(
        &self,
        mutable_txs: &mut [MutableTransaction],
        args: &TransactionValidationBatchArgs,
    ) -> Vec<TxResult<()>> {
        let virtual_read = self.virtual_stores.read();
        let virtual_state = virtual_read.state.get().unwrap();
        let virtual_utxo_view = &virtual_read.utxo_set;
        let virtual_daa_score = virtual_state.daa_score;
        let virtual_past_median_time = virtual_state.past_median_time;

        self.thread_pool.install(|| {
            mutable_txs
                .par_iter_mut()
                .map(|mtx| {
                    self.validate_mempool_transaction_impl(
                        mtx,
                        &virtual_utxo_view,
                        virtual_daa_score,
                        virtual_past_median_time,
                        args.get(&mtx.id()),
                    )
                })
                .collect::<Vec<TxResult<()>>>()
        })
    }

    fn populate_mempool_transaction_impl(
        &self,
        mutable_tx: &mut MutableTransaction,
        virtual_utxo_view: &impl UtxoView,
    ) -> TxResult<()> {
        self.populate_mempool_transaction_in_utxo_context(mutable_tx, virtual_utxo_view)?;
        Ok(())
    }

    pub fn populate_mempool_transaction(&self, mutable_tx: &mut MutableTransaction) -> TxResult<()> {
        let virtual_read = self.virtual_stores.read();
        let virtual_utxo_view = &virtual_read.utxo_set;
        self.populate_mempool_transaction_impl(mutable_tx, virtual_utxo_view)
    }

    pub fn populate_mempool_transactions_in_parallel(&self, mutable_txs: &mut [MutableTransaction]) -> Vec<TxResult<()>> {
        let virtual_read = self.virtual_stores.read();
        let virtual_utxo_view = &virtual_read.utxo_set;
        self.thread_pool.install(|| {
            mutable_txs
                .par_iter_mut()
                .map(|mtx| self.populate_mempool_transaction_impl(mtx, &virtual_utxo_view))
                .collect::<Vec<TxResult<()>>>()
        })
    }

    fn validate_block_template_transactions_in_parallel<V: UtxoView + Sync>(
        &self,
        txs: &[Transaction],
        virtual_state: &VirtualState,
        utxo_view: &V,
    ) -> Vec<TxResult<u64>> {
        self.thread_pool
            .install(|| txs.par_iter().map(|tx| self.validate_block_template_transaction(tx, virtual_state, &utxo_view)).collect())
    }

    fn validate_block_template_transaction(
        &self,
        tx: &Transaction,
        virtual_state: &VirtualState,
        utxo_view: &impl UtxoView,
    ) -> TxResult<u64> {
        // No need to validate the transaction in isolation since we rely on the mining manager to submit transactions
        // which were previously validated through `validate_mempool_transaction_and_populate`, hence we only perform
        // in-context validations
        self.transaction_validator.validate_tx_in_header_context_with_args(
            tx,
            virtual_state.daa_score,
            virtual_state.past_median_time,
        )?;
        let ValidatedTransaction { calculated_fee, .. } =
            self.validate_transaction_in_utxo_context(tx, utxo_view, virtual_state.daa_score, TxValidationFlags::Full)?;
        Ok(calculated_fee)
    }

    pub fn build_block_template(
        &self,
        miner_data: MinerData,
        mut tx_selector: Box<dyn TemplateTransactionSelector>,
        build_mode: TemplateBuildMode,
    ) -> Result<BlockTemplate, RuleError> {
        //
        // TODO (relaxed): additional tests
        //

        // We call for the initial tx batch before acquiring the virtual read lock,
        // optimizing for the common case where all txs are valid. Following selection calls
        // are called within the lock in order to preserve validness of already validated txs
        let mut txs = tx_selector.select_transactions();
        let mut calculated_fees = Vec::with_capacity(txs.len());
        let virtual_read = self.virtual_stores.read();
        let virtual_state = virtual_read.state.get().unwrap();
        let virtual_utxo_view = &virtual_read.utxo_set;

        let mut invalid_transactions = HashMap::new();
        let results = self.validate_block_template_transactions_in_parallel(&txs, &virtual_state, &virtual_utxo_view);
        for (tx, res) in txs.iter().zip(results) {
            match res {
                Err(e) => {
                    invalid_transactions.insert(tx.id(), e);
                    tx_selector.reject_selection(tx.id());
                }
                Ok(fee) => {
                    calculated_fees.push(fee);
                }
            }
        }

        let mut has_rejections = !invalid_transactions.is_empty();
        if has_rejections {
            txs.retain(|tx| !invalid_transactions.contains_key(&tx.id()));
        }

        while has_rejections {
            has_rejections = false;
            let next_batch = tx_selector.select_transactions(); // Note that once next_batch is empty the loop will exit
            let next_batch_results =
                self.validate_block_template_transactions_in_parallel(&next_batch, &virtual_state, &virtual_utxo_view);
            for (tx, res) in next_batch.into_iter().zip(next_batch_results) {
                match res {
                    Err(e) => {
                        invalid_transactions.insert(tx.id(), e);
                        tx_selector.reject_selection(tx.id());
                        has_rejections = true;
                    }
                    Ok(fee) => {
                        txs.push(tx);
                        calculated_fees.push(fee);
                    }
                }
            }
        }

        // Check whether this was an overall successful selection episode. We pass this decision
        // to the selector implementation which has the broadest picture and can use mempool config
        // and context
        match (build_mode, tx_selector.is_successful()) {
            (TemplateBuildMode::Standard, false) => return Err(RuleError::InvalidTransactionsInNewBlock(invalid_transactions)),
            (TemplateBuildMode::Standard, true) | (TemplateBuildMode::Infallible, _) => {}
        }

        // At this point we can safely drop the read lock
        drop(virtual_read);

        // Build the template
        self.build_block_template_from_virtual_state(virtual_state, miner_data, txs, calculated_fees)
    }

    pub(crate) fn validate_block_template_transactions(
        &self,
        txs: &[Transaction],
        virtual_state: &VirtualState,
        utxo_view: &impl UtxoView,
    ) -> Result<(), RuleError> {
        // Search for invalid transactions
        let mut invalid_transactions = HashMap::new();
        for tx in txs.iter() {
            if let Err(e) = self.validate_block_template_transaction(tx, virtual_state, utxo_view) {
                invalid_transactions.insert(tx.id(), e);
            }
        }
        if !invalid_transactions.is_empty() { Err(RuleError::InvalidTransactionsInNewBlock(invalid_transactions)) } else { Ok(()) }
    }

    pub(crate) fn build_block_template_from_virtual_state(
        &self,
        virtual_state: Arc<VirtualState>,
        miner_data: MinerData,
        mut txs: Vec<Transaction>,
        calculated_fees: Vec<u64>,
    ) -> Result<BlockTemplate, RuleError> {
        // [`calc_block_parents`] can use deep blocks below the pruning point for this calculation, so we
        // need to hold the pruning lock.
        let _prune_guard = self.pruning_lock.blocking_read();
        let pruning_point = self.pruning_point_store.read().pruning_point().unwrap();
        let header_pruning_point =
            self.pruning_point_manager.expected_header_pruning_point(virtual_state.ghostdag_data.to_compact()).pruning_point;
        let tier_bps_by_block =
            self.tier_bps_by_block(&virtual_state.ghostdag_data, &virtual_state.mergeset_non_daa, virtual_state.daa_score);
        // Build path: the rewarding block is virtual itself, so its view is the committed virtual the
        // balance index already reflects ⇒ no view correction needed (`view_diffs = &[]`).
        let ratio_bps_by_block = self.ratio_bps_by_block(
            &virtual_state.ghostdag_data,
            &virtual_state.mergeset_non_daa,
            &virtual_state.mergeset_rewards,
            virtual_state.daa_score,
            &[],
        );
        let coinbase = self
            .coinbase_manager
            .expected_coinbase_transaction(
                virtual_state.daa_score,
                miner_data.clone(),
                &virtual_state.ghostdag_data,
                &virtual_state.mergeset_rewards,
                &virtual_state.mergeset_non_daa,
                &tier_bps_by_block,
                &ratio_bps_by_block,
            )
            .unwrap();
        txs.insert(0, coinbase.tx);
        let version = BLOCK_VERSION;
        let parents_by_level = self.parents_manager.calc_block_parents(pruning_point, &virtual_state.parents);
        let hash_merkle_root = calc_hash_merkle_root(txs.iter());

        let accepted_id_merkle_root = self
            .calc_accepted_id_merkle_root(virtual_state.accepted_tx_ids.iter().copied(), virtual_state.ghostdag_data.selected_parent);
        let utxo_commitment = virtual_state.multiset.clone().finalize();
        // Past median time is the exclusive lower bound for valid block time, so we increase by 1 to get the valid min
        let min_block_time = virtual_state.past_median_time + 1;
        // Difficulty-reset hardfork: the persisted virtual `bits` is only refreshed when a new block
        // is processed, but on a frozen chain none can be found at the inherited (too-high) difficulty.
        // While the reset window is active we override the template to genesis so the first block is
        // mineable; it validates under the same rule in `calculate_difficulty_bits`, and its insertion
        // re-resolves the virtual normally. Outside the window this is a no-op (uses `virtual_state.bits`).
        let template_bits = self.window_manager.reset_difficulty_bits(virtual_state.daa_score).unwrap_or(virtual_state.bits);
        let header = Header::new_finalized(
            version,
            parents_by_level,
            hash_merkle_root,
            accepted_id_merkle_root,
            utxo_commitment,
            u64::max(min_block_time, unix_now()),
            template_bits,
            0,
            virtual_state.daa_score,
            virtual_state.ghostdag_data.blue_work,
            virtual_state.ghostdag_data.blue_score,
            header_pruning_point,
            0, // pom_final_state: filled by the miner from the winning walk (H3), like the nonce
        );
        let selected_parent_hash = virtual_state.ghostdag_data.selected_parent;
        let selected_parent_timestamp = self.headers_store.get_timestamp(selected_parent_hash).unwrap();
        let selected_parent_daa_score = self.headers_store.get_daa_score(selected_parent_hash).unwrap();
        Ok(BlockTemplate::new(
            MutableBlock::new(header, txs),
            miner_data,
            coinbase.has_red_reward,
            coinbase.red_reward_output_index,
            selected_parent_timestamp,
            selected_parent_daa_score,
            selected_parent_hash,
            calculated_fees,
        ))
    }

    /// Make sure pruning point-related stores are initialized
    pub fn init(self: &Arc<Self>) {
        let pruning_point_read = self.pruning_point_store.upgradable_read();
        if pruning_point_read.pruning_point().optional().unwrap().is_none() {
            let mut pruning_point_write = RwLockUpgradableReadGuard::upgrade(pruning_point_read);
            let mut pruning_meta_write = self.pruning_meta_stores.write();
            let mut batch = WriteBatch::default();
            self.past_pruning_points_store.insert_batch(&mut batch, 0, self.genesis.hash).idempotent().unwrap();
            pruning_point_write.set_batch(&mut batch, self.genesis.hash, 0).unwrap();
            pruning_point_write.set_retention_checkpoint(&mut batch, self.genesis.hash).unwrap();
            pruning_point_write.set_retention_period_root(&mut batch, self.genesis.hash).unwrap();
            pruning_meta_write.set_utxoset_position(&mut batch, self.genesis.hash).unwrap();
            self.db.write(batch).unwrap();
            drop(pruning_point_write);
            drop(pruning_meta_write);
        }
    }

    /// Initializes UTXO state of genesis and points virtual at genesis.
    /// Note that pruning point-related stores are initialized by `init`
    pub fn process_genesis(self: &Arc<Self>) {
        // Write the UTXO state of genesis
        self.commit_utxo_state(self.genesis.hash, UtxoDiff::default(), MuHash::new(), AcceptanceData::default(), ZERO_HASH);

        // Init the virtual selected chain store
        let mut batch = WriteBatch::default();
        let mut selected_chain_write = self.selected_chain_store.write();
        selected_chain_write.init_with_pruning_point(&mut batch, self.genesis.hash).unwrap();
        self.db.write(batch).unwrap();
        drop(selected_chain_write);

        // Init virtual state
        self.commit_virtual_state(
            self.virtual_stores.upgradable_read(),
            Arc::new(VirtualState::from_genesis(&self.genesis, self.ghostdag_manager.ghostdag(&[self.genesis.hash]))),
            &Default::default(),
            &Default::default(),
        );
    }

    /// Finalizes the pruning point utxoset state and imports the pruning point utxoset *to* virtual utxoset
    pub fn import_pruning_point_utxo_set(
        &self,
        new_pruning_point: Hash,
        mut imported_utxo_multiset: MuHash,
    ) -> PruningImportResult<()> {
        info!("Importing the UTXO set of the pruning point {}", new_pruning_point);
        let new_pruning_point_header = self.headers_store.get_header(new_pruning_point).unwrap();
        let imported_utxo_multiset_hash = imported_utxo_multiset.finalize();
        if imported_utxo_multiset_hash != new_pruning_point_header.utxo_commitment {
            return Err(PruningImportError::ImportedMultisetHashMismatch(
                new_pruning_point_header.utxo_commitment,
                imported_utxo_multiset_hash,
            ));
        }

        {
            // Set the pruning point utxoset position to the new point we just verified
            let mut batch = WriteBatch::default();
            let mut pruning_meta_write = self.pruning_meta_stores.write();
            pruning_meta_write.set_utxoset_position(&mut batch, new_pruning_point).unwrap();
            self.db.write(batch).unwrap();
            drop(pruning_meta_write);
        }

        {
            // Copy the pruning-point UTXO set into virtual's UTXO set
            let pruning_meta_read = self.pruning_meta_stores.read();
            let mut virtual_write = self.virtual_stores.write();

            virtual_write.utxo_set.clear().unwrap();
            for chunk in &pruning_meta_read.utxo_set.iterator().map(|iter_result| iter_result.unwrap()).chunks(1000) {
                virtual_write.utxo_set.write_from_iterator_without_cache(chunk).unwrap();
            }

            // Ratio-reward (Stage 2b): a from-genesis node builds the balance index from every UTXO
            // diff since genesis; a fast-synced node skips that history, so seed the index directly
            // from the imported pruning-point UTXO snapshot (grouped per payout SPK). The incremental
            // lockstep maintenance in `commit_virtual_state` then carries it forward from here.
            //
            // The windowed-production prefix index is NOT seeded from the snapshot (there is nothing to
            // seed it with — the snapshot is a UTXO set, not a production history) — we only clear it so
            // a re-import over a populated DB cannot leave stale aggregates. This was historically
            // assumed safe because `W < pruning_depth` ⇒ the whole window lies above the pruning point
            // and gets rebuilt exactly by the IBD that follows, completing before the window matters.
            // That holds only while the pruning point is still before the ratio-reward activation
            // height; once the chain has advanced past `pruning_depth` beyond it, every fresh fast sync
            // now lands its import after activation, and needs a full window of catch-up before its
            // computed ratio matches long-running nodes. We record the pre-import chain position here so
            // `trust_coinbase()` can detect and bound exactly that catch-up period (self-expiring once
            // the window organically refills) instead of disqualifying every block it sees in the
            // meantime. See `production_seed` module doc and `is_archival` field doc for the full
            // rationale.
            let mut balances: HashMap<ScriptPublicKey, u64> = HashMap::new();
            for item in virtual_write.utxo_set.iterator() {
                let (_, entry) = item.unwrap();
                let acc = balances.entry(entry.script_public_key.clone()).or_default();
                *acc = acc.saturating_add(entry.amount);
            }
            let production_seed_index = match self.selected_chain_store.read().get_tip() {
                Ok((idx, _)) => idx,
                Err(_) => 0,
            };
            let mut batch = WriteBatch::default();
            self.address_balance_store.clear(&mut batch).unwrap();
            // Reset the prefix-sum production index: forward lockstep maintenance rebuilds it from the
            // import point (baseline immaterial — `windowed` is a difference), and the fast-sync
            // catch-up window trusts the coinbase until it has refilled past the import point.
            self.windowed_production_prefix_store.clear(&mut batch);
            self.production_index_seed_store.write().set_batch(&mut batch, production_seed_index).unwrap();
            for (spk, amount) in balances {
                self.address_balance_store.set_batch(&mut batch, &spk, amount).unwrap();
            }
            self.db.write(batch).unwrap();
            info!(
                "Ratio-reward windowed-production index reset at import (selected-chain index {}); \
                 ratio/tier coinbase verification will be auto-trusted for the next ~{} blocks of \
                 catch-up (see KERYX_TRUST_COINBASE / is_archival for the other ways this can happen).",
                production_seed_index, self.ratio_reward_window
            );
        }

        let virtual_read = self.virtual_stores.upgradable_read();

        // Validate transactions of the pruning point itself
        let new_pruning_point_transactions = self.block_transactions_store.get(new_pruning_point).unwrap();
        let validated_transactions = self.validate_transactions_in_parallel(
            &new_pruning_point_transactions,
            &virtual_read.utxo_set,
            new_pruning_point_header.daa_score,
            TxValidationFlags::Full,
        );
        if validated_transactions.len() < new_pruning_point_transactions.len() - 1 {
            // Some non-coinbase transactions are invalid
            return Err(PruningImportError::NewPruningPointTxErrors);
        }

        {
            // Submit partial UTXO state for the pruning point.
            // Note we only have and need the multiset; acceptance data and utxo-diff are irrelevant.
            let mut batch = WriteBatch::default();
            self.utxo_multisets_store.set_batch(&mut batch, new_pruning_point, imported_utxo_multiset.clone()).unwrap();

            let statuses_write = self.statuses_store.set_batch(&mut batch, new_pruning_point, StatusUTXOValid).unwrap();
            self.db.write(batch).unwrap();
            drop(statuses_write);
        }

        // Calculate the virtual state, treating the pruning point as the only virtual parent
        let virtual_parents = vec![new_pruning_point];
        let virtual_ghostdag_data = self.ghostdag_manager.ghostdag(&virtual_parents);

        self.calculate_and_commit_virtual_state(
            virtual_read,
            virtual_parents,
            virtual_ghostdag_data,
            imported_utxo_multiset.clone(),
            &mut UtxoDiff::default(),
            &ChainPath::default(),
        )?;

        // Coin-age (v3): seed the age-bucket index from the just-imported UTXO set (every entry
        // carries its `effective_daa`). The startup rebuild was skipped on this consensus while it
        // had no virtual state; the state exists now, and the imported set is the exact baseline —
        // without this, a fast-synced node validates holder-reward with empty buckets once the
        // post-import trust window expires.
        self.rebuild_age_buckets_index();

        Ok(())
    }

    pub fn are_pruning_points_violating_finality(&self, pp_list: PruningPointsList) -> bool {
        // Ideally we would want to check if the last known pruning point has the finality point
        // in its chain, but in some cases it's impossible: let `lkp` be the last known pruning
        // point from the list, and `fup` be the first unknown pruning point (the one following `lkp`).
        // fup.blue_score - lkp.blue_score ≈ finality_depth (±k), so it's possible for `lkp` not to
        // have the finality point in its past. So we have no choice but to check if `lkp`
        // has `finality_point.finality_point` in its chain (in the worst case `fup` is one block
        // above the current finality point, and in this case `lkp` will be a few blocks above the
        // finality_point.finality_point), meaning this function can only detect finality violations
        // in depth of 2*finality_depth, and can give false negatives for smaller finality violations.
        let current_pp = self.pruning_point_store.read().pruning_point().unwrap();
        let vf = self.virtual_finality_point(&self.lkg_virtual_state.load().ghostdag_data, current_pp);
        let vff = self.depth_manager.calc_finality_point(&self.ghostdag_store.get_data(vf).unwrap(), current_pp);

        let last_known_pp = pp_list.iter().rev().find(|pp| match self.statuses_store.read().get(pp.hash).optional().unwrap() {
            Some(status) => status.is_valid(),
            None => false,
        });

        if let Some(last_known_pp) = last_known_pp {
            !self.reachability_service.is_chain_ancestor_of(vff, last_known_pp.hash)
        } else {
            // If no pruning point is known, there's definitely a finality violation
            // (normally at least genesis should be known).
            true
        }
    }

    /// Executes `op` within the thread pool associated with this processor.
    pub fn install<OP, R>(&self, op: OP) -> R
    where
        OP: FnOnce() -> R + Send,
        R: Send,
    {
        self.thread_pool.install(op)
    }

    #[cfg(test)]
    pub(crate) fn test_ai_response_store(
        &self,
    ) -> &Arc<crate::model::stores::ai_slash::DbAiResponseStore> {
        &self.ai_response_store
    }

}

enum MergesetIncreaseResult {
    Accepted { increase_size: u64 },
    Rejected { new_candidate: Hash },
}
