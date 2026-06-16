use super::VirtualStateProcessor;
use crate::{
    errors::{
        BlockProcessResult,
        RuleError::{
            AiRequestEscrowBelowInferenceReward, AiRequestFeeBelowInferenceReward,
            AiRequestInferenceRewardBelowMinimum, AiRequestInvalidEscrowScript,
            AiRequestMissingEscrowOutput, AiRequestPriorityFeeBelowMinimum,
            AiResponseModelCapMissing, AiResponseModelMismatch, AiResponsePayloadMalformed, BadAcceptedIDMerkleRoot,
            BadCoinbaseTransaction, BadUTXOCommitment, InvalidTransactionsInUtxoContext,
            SyntheticLivenessNoEscrow, SyntheticLivenessStale, WrongHeaderPruningPoint,
        },
    },
    model::stores::{
        block_transactions::BlockTransactionsStoreReader,
        daa::DaaStoreReader,
        ghostdag::{CompactGhostdagData, GhostdagData},
        headers::HeaderStoreReader,
    },
    processes::{
        pruning::PruningPointReply,
        transaction_validator::{
            errors::{TxResult, TxRuleError},
            tx_validation_in_utxo_context::TxValidationFlags,
        },
    },
};
use crate::model::stores::ai_slash::{AiResponseRecord, AiResponseStore, AiResponseStoreReader};
use crate::model::services::reachability::ReachabilityService;
use crate::model::stores::miner_liveness::{LivenessAnnotation, MinerLivenessStore, MinerLivenessStoreReader};
use keryx_consensus_core::{
    BlockHashMap, BlockHashSet, HashMapCustomHasher,
    acceptance_data::{AcceptedTxEntry, MergesetBlockAcceptanceData},
    api::args::TransactionValidationArgs,
    coinbase::*,
    hashing,
    header::Header,
    muhash::MuHashExtensions,
    tx::{MutableTransaction, PopulatedTransaction, Transaction, TransactionId, ValidatedTransaction, VerifiableTransaction},
    utxo::{
        utxo_diff::UtxoDiff,
        utxo_view::{UtxoView, UtxoViewComposition},
    },
};
use keryx_core::{info, trace, warn};
use keryx_hashes::Hash;
use keryx_inference::{AiRequestPayload, AiResponsePayload, INFERENCE_REWARD_TOKEN_STEP, parse_ai_caps};
use keryx_inference::synthetic::{derive_synthetic_request, synthetic_seed, SYNTHETIC_EPOCH_BLOCKS};
use keryx_consensus_core::config::params::SYNTHETIC_LIVENESS_GRACE_EPOCHS;
use keryx_consensus_core::config::params::{LLAMA_3_3_70B_MODEL_ID_LEGACY, TIER_MODEL_IDS, TIER_REWARD_BPS};
use keryx_muhash::MuHash;
use keryx_txscript::script_class::ScriptClass;
use keryx_utils::refs::Refs;

use rayon::prelude::*;
use smallvec::{SmallVec, smallvec};
use std::{iter::once, ops::Deref};

pub(crate) mod crescendo {
    use keryx_core::{info, log::CRESCENDO_KEYWORD};
    use std::sync::{
        Arc,
        atomic::{AtomicU8, Ordering},
    };

    #[derive(Clone)]
    pub(crate) struct _CrescendoLogger {
        steps: Arc<AtomicU8>,
    }

    impl _CrescendoLogger {
        pub fn _new() -> Self {
            Self { steps: Arc::new(AtomicU8::new(Self::_ACTIVATE)) }
        }

        const _ACTIVATE: u8 = 0;

        pub fn _report_activation(&self) -> bool {
            if self.steps.compare_exchange(Self::_ACTIVATE, Self::_ACTIVATE + 1, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
                info!(target: CRESCENDO_KEYWORD, "[Crescendo] [--------- Crescendo activated for UTXO state processing rules ---------]");
                true
            } else {
                false
            }
        }
    }
}

/// A context for processing the UTXO state of a block with respect to its selected parent.
/// Note this can also be the virtual block.
pub(super) struct UtxoProcessingContext<'a> {
    pub ghostdag_data: Refs<'a, GhostdagData>,
    pub multiset_hash: MuHash,
    pub mergeset_diff: UtxoDiff,
    pub accepted_tx_ids: Vec<TransactionId>,
    pub mergeset_acceptance_data: Vec<MergesetBlockAcceptanceData>,
    pub mergeset_rewards: BlockHashMap<BlockRewardData>,
    pub pruning_sample_from_pov: Option<Hash>,
}

impl<'a> UtxoProcessingContext<'a> {
    pub fn new(ghostdag_data: Refs<'a, GhostdagData>, selected_parent_multiset_hash: MuHash) -> Self {
        let mergeset_size = ghostdag_data.mergeset_size();
        Self {
            ghostdag_data,
            multiset_hash: selected_parent_multiset_hash,
            mergeset_diff: UtxoDiff::default(),
            accepted_tx_ids: Vec::with_capacity(1), // We expect at least the selected parent coinbase tx
            mergeset_rewards: BlockHashMap::with_capacity(mergeset_size),
            mergeset_acceptance_data: Vec::with_capacity(mergeset_size),
            pruning_sample_from_pov: Default::default(),
        }
    }

    pub fn selected_parent(&self) -> Hash {
        self.ghostdag_data.selected_parent
    }
}

impl VirtualStateProcessor {
    /// Calculates UTXO state and transaction acceptance data relative to the selected parent state
    pub(super) fn calculate_utxo_state<V: UtxoView + Sync>(
        &self,
        ctx: &mut UtxoProcessingContext,
        selected_parent_utxo_view: &V,
        pov_daa_score: u64,
    ) {
        let selected_parent_transactions = self.block_transactions_store.get(ctx.selected_parent()).unwrap();
        let validated_coinbase = ValidatedTransaction::new_coinbase(&selected_parent_transactions[0]);

        ctx.mergeset_diff.add_transaction(&validated_coinbase, pov_daa_score).unwrap();
        ctx.multiset_hash.add_transaction(&validated_coinbase, pov_daa_score);
        let validated_coinbase_id = validated_coinbase.id();
        ctx.accepted_tx_ids.push(validated_coinbase_id);

        for (i, (merged_block, txs)) in once((ctx.selected_parent(), selected_parent_transactions))
            .chain(
                ctx.ghostdag_data
                    .consensus_ordered_mergeset_without_selected_parent(self.ghostdag_store.deref())
                    .map(|b| (b, self.block_transactions_store.get(b).unwrap())),
            )
            .enumerate()
        {
            // Create a composed UTXO view from the selected parent UTXO view + the mergeset UTXO diff
            let composed_view = selected_parent_utxo_view.compose(&ctx.mergeset_diff);

            // The first block in the mergeset is always the selected parent
            let is_selected_parent = i == 0;

            // No need to fully validate selected parent transactions since selected parent txs were already validated
            // as part of selected parent UTXO state verification with the exact same UTXO context.
            let validation_flags = if is_selected_parent { TxValidationFlags::SkipScriptChecks } else { TxValidationFlags::Full };
            let (validated_transactions, inner_multiset) =
                self.validate_transactions_with_muhash_in_parallel(&txs, &composed_view, pov_daa_score, validation_flags);

            ctx.multiset_hash.combine(&inner_multiset);

            let mut block_fee = 0u64;
            for (validated_tx, _) in validated_transactions.iter() {
                ctx.mergeset_diff.add_transaction(validated_tx, pov_daa_score).unwrap();
                ctx.accepted_tx_ids.push(validated_tx.id());
                block_fee += validated_tx.calculated_fee;
            }

            ctx.mergeset_acceptance_data.push(MergesetBlockAcceptanceData {
                block_hash: merged_block,
                // For the selected parent, we prepend the coinbase tx
                accepted_transactions: is_selected_parent
                    .then_some(AcceptedTxEntry { transaction_id: validated_coinbase_id, index_within_block: 0 })
                    .into_iter()
                    .chain(
                        validated_transactions
                            .into_iter()
                            .map(|(tx, tx_idx)| AcceptedTxEntry { transaction_id: tx.id(), index_within_block: tx_idx }),
                    )
                    .collect(),
            });

            let coinbase_data = self.coinbase_manager.deserialize_coinbase_payload(&txs[0].payload).unwrap();
            let escrow_spk =
                self.coinbase_manager.parse_escrow_from_extra_data(coinbase_data.miner_data.extra_data);
            ctx.mergeset_rewards.insert(
                merged_block,
                BlockRewardData::new_with_escrow(
                    coinbase_data.subsidy,
                    block_fee,
                    coinbase_data.miner_data.script_public_key,
                    escrow_spk,
                ),
            );

            // OPoI Phase 3 A3: register AiResponse txs and process AiChallenge txs.
            // Called after parallel validation so write lock does not overlap with
            // the read lock taken in validate_transaction_in_utxo_context.
            let coinbase_tx_id = txs[0].id();
            self.process_ai_txs_for_slash(&txs, pov_daa_score, coinbase_tx_id, merged_block);
        }
    }

    /// Verify that the current block fully respects its own UTXO view. We define a block as
    /// UTXO valid if all the following conditions hold:
    ///     1. The block header includes the expected `utxo_commitment`.
    ///     2. The block header includes the expected `accepted_id_merkle_root`.
    ///     3. The block header includes the expected `pruning_point`.
    ///     4. The block coinbase transaction rewards the mergeset blocks correctly.
    ///     5. All non-coinbase block transactions are valid against its own UTXO view.
    pub(super) fn verify_expected_utxo_state<V: UtxoView + Sync>(
        &self,
        ctx: &mut UtxoProcessingContext,
        selected_parent_utxo_view: &V,
        header: &Header,
    ) -> BlockProcessResult<()> {
        // Verify header UTXO commitment
        let expected_commitment = ctx.multiset_hash.finalize();
        if expected_commitment != header.utxo_commitment {
            return Err(BadUTXOCommitment(header.hash, header.utxo_commitment, expected_commitment));
        }
        trace!("correct commitment: {}, {}", header.hash, expected_commitment);

        // Verify header accepted_id_merkle_root
        let expected_accepted_id_merkle_root =
            self.calc_accepted_id_merkle_root(ctx.accepted_tx_ids.iter().copied(), ctx.selected_parent());

        if expected_accepted_id_merkle_root != header.accepted_id_merkle_root {
            return Err(BadAcceptedIDMerkleRoot(header.hash, header.accepted_id_merkle_root, expected_accepted_id_merkle_root));
        }

        let txs = self.block_transactions_store.get(header.hash).unwrap();

        // Verify coinbase transaction
        self.verify_coinbase_transaction(
            &txs[0],
            header.daa_score,
            &ctx.ghostdag_data,
            &ctx.mergeset_rewards,
            &self.daa_excluded_store.get_mergeset_non_daa(header.hash).unwrap(),
        )?;

        // Verify the header pruning point
        let reply = self.verify_header_pruning_point(header, ctx.ghostdag_data.to_compact())?;
        ctx.pruning_sample_from_pov = Some(reply.pruning_sample);

        // Hardfork activation banner — one consolidated line per activation DAA, logged
        // ONCE. The virtual-chain re-resolution at the boundary re-processes the same block
        // many times; without this guard each pass re-emitted every banner (log spam).
        {
            use std::sync::atomic::Ordering;
            static LAST_LOGGED_DAA: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(u64::MAX);
            // Short component names (no parentheticals); OPoI v2 names the fork headline.
            let mut newly: Vec<&str> = Vec::new();
            if header.daa_score == self.synthetic_liveness_activation.daa_score() { newly.push("synthetic-liveness"); }
            if header.daa_score == self.tier_reward_activation.daa_score() { newly.push("tier-reward"); }
            if header.daa_score == self.model_cap_enforcement_activation.daa_score() { newly.push("model-cap"); }
            if header.daa_score == self.pow_salt_v5_activation.daa_score() { newly.push("SALT v5"); }
            if header.daa_score == self.pow_salt_v4_activation.daa_score() { newly.push("SALT v4"); }
            if header.daa_score == self.pow_salt_v2_activation.daa_score() { newly.push("SALT v2"); }
            let opoi = header.daa_score == self.opoi_v2_activation.daa_score();
            if (opoi || !newly.is_empty()) && LAST_LOGGED_DAA.swap(header.daa_score, Ordering::Relaxed) != header.daa_score {
                match (opoi, newly.is_empty()) {
                    (true, true) => info!("=== HARDFORK OPoI v2 ACTIVATED at DAA {} ===", header.daa_score),
                    (true, false) => info!("=== HARDFORK OPoI v2 ACTIVATED at DAA {} — also: {} ===", header.daa_score, newly.join(" + ")),
                    (false, _) => info!("=== HARDFORK ACTIVATED at DAA {} — {} ===", header.daa_score, newly.join(" + ")),
                }
            }
        }

        // Enforcement (runs every block while active; unchanged).
        if self.opoi_v2_activation.is_active(header.daa_score) {
            check_ai_response_v2(&txs)?;
        } else if self.model_cap_enforcement_activation.is_active(header.daa_score) {
            self.check_ai_response_model_caps(&txs)?;
        }
        self.enforce_synthetic_liveness(&txs, header)?;

        // Verify all transactions are valid in context
        let current_utxo_view = selected_parent_utxo_view.compose(&ctx.mergeset_diff);
        let validated_transactions =
            self.validate_transactions_in_parallel(&txs, &current_utxo_view, header.daa_score, TxValidationFlags::Full);
        if validated_transactions.len() < txs.len() - 1 {
            // Some non-coinbase transactions are invalid
            return Err(InvalidTransactionsInUtxoContext(txs.len() - 1 - validated_transactions.len(), txs.len() - 1));
        }

        // Enforce AiRequest inference_reward minimums and fee coverage after activation.
        // The 70B entry differs across the OPoI v2 gate (legacy IQ3 id before, Q4_K_M after),
        // so historical blocks revalidate against the table the network enforced at the time.
        if self.model_cap_enforcement_activation.is_active(header.daa_score) {
            let minimums = if self.opoi_v2_activation.is_active(header.daa_score) {
                self.inference_reward_minimums
            } else {
                keryx_consensus_core::config::params::INFERENCE_REWARD_MINIMUMS_LEGACY
            };
            check_ai_request_inference_rewards(&txs, &validated_transactions, minimums)?;
        }

        Ok(())
    }

    fn verify_header_pruning_point(
        &self,
        header: &Header,
        ghostdag_data: CompactGhostdagData,
    ) -> BlockProcessResult<PruningPointReply> {
        let reply = self.pruning_point_manager.expected_header_pruning_point(ghostdag_data);
        if reply.pruning_point != header.pruning_point {
            return Err(WrongHeaderPruningPoint(reply.pruning_point, header.pruning_point));
        }
        Ok(reply)
    }

    fn verify_coinbase_transaction(
        &self,
        coinbase: &Transaction,
        daa_score: u64,
        ghostdag_data: &GhostdagData,
        mergeset_rewards: &BlockHashMap<BlockRewardData>,
        mergeset_non_daa: &BlockHashSet,
    ) -> BlockProcessResult<()> {
        // Extract only miner data from the provided coinbase
        let miner_data = self.coinbase_manager.deserialize_coinbase_payload(&coinbase.payload).unwrap().miner_data;
        let tier_bps_by_block = self.tier_bps_by_block(ghostdag_data, mergeset_non_daa, daa_score);
        let expected_coinbase = self
            .coinbase_manager
            .expected_coinbase_transaction(
                daa_score,
                miner_data,
                ghostdag_data,
                mergeset_rewards,
                mergeset_non_daa,
                &tier_bps_by_block,
            )
            .unwrap()
            .tx;
        if hashing::tx::hash(coinbase) != hashing::tx::hash(&expected_coinbase) { Err(BadCoinbaseTransaction) } else { Ok(()) }
    }

    /// Tier-reward map consumed by `expected_coinbase_transaction`: for each rewarded blue,
    /// the subsidy multiplier (bps) of the highest model tier it declares in `ai:cap`. Both
    /// the validator and the template builder derive it the same way from immutable block
    /// bodies, so the coinbase they produce agrees deterministically. Returns an empty map
    /// before `tier_reward_activation` (⇒ every cut paid in full, no penalty, no burn).
    pub(super) fn tier_bps_by_block(
        &self,
        ghostdag_data: &GhostdagData,
        mergeset_non_daa: &BlockHashSet,
        pov_daa_score: u64,
    ) -> BlockHashMap<u64> {
        let mut map = BlockHashMap::new();
        if !self.tier_reward_activation.is_active(pov_daa_score) {
            return map;
        }
        for blue in ghostdag_data.mergeset_blues.iter().filter(|h| !mergeset_non_daa.contains(h)) {
            let txs = self.block_transactions_store.get(*blue).unwrap();
            map.insert(*blue, tier_reward_bps(&txs[0].payload));
        }
        map
    }

    fn check_ai_response_model_caps(&self, txs: &[Transaction]) -> BlockProcessResult<()> {
        check_ai_response_model_caps(txs)
    }

    /// Validates transactions against the provided `utxo_view` and returns a vector with all transactions
    /// which passed the validation along with their original index within the containing block
    pub(crate) fn validate_transactions_in_parallel<'a, V: UtxoView + Sync>(
        &self,
        txs: &'a Vec<Transaction>,
        utxo_view: &V,
        pov_daa_score: u64,
        flags: TxValidationFlags,
    ) -> Vec<(ValidatedTransaction<'a>, u32)> {
        self.thread_pool.install(|| {
            txs
                .par_iter() // We can do this in parallel without complications since block body validation already ensured
                            // that all txs within each block are independent
                .enumerate()
                .skip(1) // Skip the coinbase tx.
                .filter_map(|(i, tx)| self.validate_transaction_in_utxo_context(tx, &utxo_view, pov_daa_score, flags).ok().map(|vtx| (vtx, i as u32)))
                .collect()
        })
    }

    /// Same as validate_transactions_in_parallel except during the iteration this will also
    /// calculate the muhash in parallel for valid transactions
    pub(crate) fn validate_transactions_with_muhash_in_parallel<'a, V: UtxoView + Sync>(
        &self,
        txs: &'a Vec<Transaction>,
        utxo_view: &V,
        pov_daa_score: u64,
        flags: TxValidationFlags,
    ) -> (SmallVec<[(ValidatedTransaction<'a>, u32); 2]>, MuHash) {
        self.thread_pool.install(|| {
            txs
                .par_iter() // We can do this in parallel without complications since block body validation already ensured
                            // that all txs within each block are independent
                .enumerate()
                .skip(1) // Skip the coinbase tx.
                .filter_map(|(i, tx)| self.validate_transaction_in_utxo_context(tx, &utxo_view, pov_daa_score, flags).ok().map(|vtx| {
                    let mh = MuHash::from_transaction(&vtx, pov_daa_score);
                    (smallvec![(vtx, i as u32)], mh)
                }
                ))
                .reduce(
                    || (smallvec![], MuHash::new()),
                    |mut a, mut b| {
                        a.0.append(&mut b.0);
                        a.1.combine(&b.1);
                        a
                    },
                )
        })
    }

    /// Attempts to populate the transaction with UTXO entries and performs all utxo-related tx validations
    pub(super) fn validate_transaction_in_utxo_context<'a>(
        &self,
        transaction: &'a Transaction,
        utxo_view: &impl UtxoView,
        pov_daa_score: u64,
        flags: TxValidationFlags,
    ) -> TxResult<ValidatedTransaction<'a>> {
        // OPoI slashing removed (v1.2.3): the slashed-escrow enforcement was non-deterministic
        // under a multi-challenger flood — the recorded challenger_spk is last-writer-wins, so
        // different nodes accepted/rejected the same escrow spend differently, producing diverging
        // UTXO commitments and fragmenting consensus. On top of that the verifiable commitment was
        // lost in the result->ipfs_cid migration, so every honest AiResponse was slashable by anyone.
        // Escrows are therefore always spendable now; no slashed-escrow check is performed.

        let mut entries = Vec::with_capacity(transaction.inputs.len());
        for input in transaction.inputs.iter() {
            if let Some(entry) = utxo_view.get(&input.previous_outpoint) {
                entries.push(entry);
            } else {
                // Missing at least one input. For perf considerations, we report once a single miss is detected and avoid collecting all possible misses.
                return Err(TxRuleError::MissingTxOutpoints);
            }
        }
        let populated_tx = PopulatedTransaction::new(transaction, entries);
        let res = self.transaction_validator.validate_populated_transaction_and_get_fee(&populated_tx, pov_daa_score, flags, None);
        match res {
            Ok(calculated_fee) => Ok(ValidatedTransaction::new(populated_tx, calculated_fee)),
            Err(tx_rule_error) => {
                // TODO (relaxed): aggregate by error types and log through the monitor (in order to not flood the logs)
                info!("Rejecting transaction {} due to transaction rule error: {}", transaction.id(), tx_rule_error);
                Err(tx_rule_error)
            }
        }
    }

    /// Populates the mempool transaction with maximally found UTXO entry data
    pub(crate) fn populate_mempool_transaction_in_utxo_context(
        &self,
        mutable_tx: &mut MutableTransaction,
        utxo_view: &impl UtxoView,
    ) -> TxResult<()> {
        let mut has_missing_outpoints = false;
        for i in 0..mutable_tx.tx.inputs.len() {
            if mutable_tx.entries[i].is_some() {
                // We prefer a previously populated entry if such exists
                continue;
            }
            if let Some(entry) = utxo_view.get(&mutable_tx.tx.inputs[i].previous_outpoint) {
                mutable_tx.entries[i] = Some(entry);
            } else {
                // We attempt to fill as much as possible UTXO entries, hence we do not break in this case but rather continue looping
                has_missing_outpoints = true;
            }
        }
        if has_missing_outpoints {
            return Err(TxRuleError::MissingTxOutpoints);
        }
        Ok(())
    }

    /// Populates the mempool transaction with maximally found UTXO entry data and proceeds to validation if all found
    pub(super) fn validate_mempool_transaction_in_utxo_context(
        &self,
        mutable_tx: &mut MutableTransaction,
        utxo_view: &impl UtxoView,
        pov_daa_score: u64,
        args: &TransactionValidationArgs,
    ) -> TxResult<()> {
        self.populate_mempool_transaction_in_utxo_context(mutable_tx, utxo_view)?;

        // Calc the contextual storage mass
        let contextual_mass = self
            .transaction_validator
            .mass_calculator
            .calc_contextual_masses(&mutable_tx.as_verifiable())
            .ok_or(TxRuleError::MassIncomputable)?;

        // Set the inner mass field
        mutable_tx.tx.set_mass(contextual_mass.storage_mass);

        // At this point we know all UTXO entries are populated, so we can safely pass the tx as verifiable
        let mass_and_feerate_threshold = args
            .feerate_threshold
            .map(|threshold| (contextual_mass.max(mutable_tx.calculated_non_contextual_masses.unwrap()), threshold));
        let calculated_fee = self.transaction_validator.validate_populated_transaction_and_get_fee(
            &mutable_tx.as_verifiable(),
            pov_daa_score,
            TxValidationFlags::SkipMassCheck, // we can skip the mass check since we just set it
            mass_and_feerate_threshold,
        )?;
        mutable_tx.calculated_fee = Some(calculated_fee);
        Ok(())
    }

    /// Scans a block's transactions for AiResponse and AiChallenge txs and updates the slash state.
    ///
    /// Called sequentially AFTER `validate_transactions_with_muhash_in_parallel` so there is no
    /// lock contention with the read lock in `validate_transaction_in_utxo_context`.
    fn process_ai_txs_for_slash(&self, txs: &[Transaction], pov_daa_score: u64, coinbase_tx_id: TransactionId, block_hash: Hash) {
        // Register confirmed AiResponse txs.
        for tx in txs.iter().skip(1) {
            if tx.is_ai_response() {
                let hash = blake2b_simd::blake2b(&tx.payload);
                let mut response_hash_bytes = [0u8; 32];
                response_hash_bytes.copy_from_slice(&hash.as_bytes()[..32]);
                let response_hash = Hash::from_bytes(response_hash_bytes);

                // Extract request_hash and claimed_commitment (record-only, not consensus-read).
                // V2 payloads carry a real result_commitment; v1 records fall back to the
                // sha2-256 digest embedded in the IPFS multihash (bytes [2..34]).
                let (request_hash, claimed_commitment) =
                    if let Some(resp) = AiResponsePayload::deserialize(&tx.payload) {
                        let commitment: [u8; 32] = resp
                            .result_commitment
                            .unwrap_or_else(|| resp.response_ipfs_cid[2..34].try_into().unwrap());
                        (resp.request_hash, commitment)
                    } else {
                        ([0u8; 32], [0u8; 32])
                    };

                let record = AiResponseRecord {
                    inclusion_blue_score: pov_daa_score,
                    coinbase_tx_id,
                    request_hash,
                    claimed_commitment,
                };
                // Log only the first registration of a given response_hash. The same
                // AiResponse is re-included in many block bodies across the DAG, so an
                // unconditional log spams INFO once per body. The .set() itself stays
                // unconditional (last-writer-wins): the record's inclusion_blue_score and
                // coinbase_tx_id are consensus-critical and must keep their original
                // semantics — only the logging is gated, so this is consensus-neutral.
                let already_known = self.ai_response_store.has(response_hash).unwrap_or(false);
                if let Err(e) = self.ai_response_store.set(response_hash, record) {
                    warn!("OPoI: failed to register AiResponse in DB: {}", e);
                } else if !already_known {
                    info!("OPoI: registered AiResponse response_hash={}", hex::encode(response_hash_bytes));
                }
            }
        }

        // OPoI slashing removed (v1.2.3): AiChallenge txs are no longer processed. The fraud-proof
        // slash was non-deterministic (last-writer-wins challenger_spk under a multi-challenger
        // flood) and slashed honest AiResponses (commitment lost in the result->ipfs_cid migration),
        // which fragmented consensus. No slash is recorded and verify_fraud_proof is never run.
        // AiResponses are still registered above for record-keeping only (no longer consensus-read).

        // OPoI synthetic-liveness annotation (Level-1, option C). Unconditional: if this
        // accepted block's own body answers the synthetic task for its own coinbase miner,
        // annotate the block hash with (escrow_pubkey, epoch). The annotation is an
        // immutable function of the block, so a descendant can later prove its liveness by
        // referencing this block (`/live:<hash>`) + a deterministic reachability check.
        if let Some(pubkey) = parse_coinbase_escrow_pubkey(&txs[0].payload) {
            let epoch = pov_daa_score / SYNTHETIC_EPOCH_BLOCKS;
            if let Some(answered_epoch) = synthetic_answer_epoch(txs, &pubkey, epoch) {
                let annotation = LivenessAnnotation { escrow_pubkey: pubkey, epoch: answered_epoch };
                if let Err(e) = self.miner_liveness_store.set(block_hash, annotation) {
                    warn!("OPoI: failed to record liveness annotation: {}", e);
                } else {
                    trace!("OPoI: liveness annotation block={} miner={} epoch={}", block_hash, hex::encode(pubkey), answered_epoch);
                }
            }
        }
    }

    /// Enforces OPoI synthetic liveness for a selected-chain block (Level-1, option C).
    /// Gated by `synthetic_liveness_activation`. A block is liveness-valid iff its
    /// coinbase miner proves a fresh synthetic answer, either:
    ///   (a) self-contained — this block's own body answers for this/last epoch; or
    ///   (b) by reference — its coinbase carries `/live:<H>` pointing to a reachable
    ///       ancestor block `H` that answered for the same miner within the grace window.
    /// Both paths read only deterministic state (the block's own body, plus immutable
    /// per-block annotations consulted via reachability), so no reorg-dependent split.
    fn enforce_synthetic_liveness(&self, txs: &[Transaction], header: &Header) -> BlockProcessResult<()> {
        if !self.synthetic_liveness_activation.is_active(header.daa_score) {
            return Ok(());
        }
        let epoch = header.daa_score / SYNTHETIC_EPOCH_BLOCKS;
        // Defer enforcement until at least one FULL epoch has elapsed since opoi_v2. The
        // first fully-v2 epoch runs UNENFORCED — the miner answers it freely and that answer
        // gets annotated. Enforcement then begins the next epoch, where every block can
        // reference that already-established answer via `/live:` from block one. This makes
        // the hardfork transition seamless: no transient disqualifications, and no epoch
        // straddling opoi_v2 (which could hold v1 answers) is ever enforced.
        if epoch * SYNTHETIC_EPOCH_BLOCKS < self.opoi_v2_activation.daa_score().saturating_add(SYNTHETIC_EPOCH_BLOCKS) {
            return Ok(());
        }
        let Some(pubkey) = parse_coinbase_escrow_pubkey(&txs[0].payload) else {
            return Err(SyntheticLivenessNoEscrow(header.hash));
        };
        // (a) self-contained: recomputed on this block's own (immutable) body.
        if synthetic_answer_epoch(txs, &pubkey, epoch).is_some() {
            return Ok(());
        }
        // (b) /live:<H> reference to a reachable, recent ancestor for the same miner.
        if let Some(anchor) = parse_coinbase_live_ref(&txs[0].payload) {
            // Read the cheap immutable annotation first; only known (annotated) ancestors
            // reach the reachability check, so it never sees an unknown hash.
            if let Ok(Some(annot)) = self.miner_liveness_store.get(anchor) {
                if annot.escrow_pubkey == pubkey
                    && epoch.saturating_sub(annot.epoch) <= 1 + SYNTHETIC_LIVENESS_GRACE_EPOCHS
                    && self.reachability_service.is_dag_ancestor_of(anchor, header.hash)
                {
                    return Ok(());
                }
            }
        }
        Err(SyntheticLivenessStale(header.hash, hex::encode(pubkey), epoch))
    }

    /// Calculates the accepted_id_merkle_root based on the current DAA score and the accepted tx ids
    /// refer KIP-15 for more details
    pub(super) fn calc_accepted_id_merkle_root(
        &self,
        accepted_tx_ids: impl ExactSizeIterator<Item = Hash>,
        selected_parent: Hash,
    ) -> Hash {
        keryx_merkle::merkle_hash(
            self.headers_store.get_header(selected_parent).unwrap().accepted_id_merkle_root,
            keryx_merkle::calc_merkle_root(accepted_tx_ids),
        )
    }
}

/// Parses the 32-byte miner escrow Schnorr pubkey from a coinbase payload's
/// `/escrow:<64-hex>` marker. Returns `None` when the marker is absent or
/// malformed — such a block declares no escrow identity and records no liveness.
/// Mirrors `CoinbaseManager::parse_escrow_from_extra_data` but yields raw bytes.
fn parse_coinbase_escrow_pubkey(coinbase_payload: &[u8]) -> Option<[u8; 32]> {
    const MARKER: &[u8] = b"/escrow:";
    const HEX_LEN: usize = 64;
    let pos = coinbase_payload.windows(MARKER.len()).position(|w| w == MARKER)?;
    let hex_start = pos + MARKER.len();
    let hex_end = hex_start + HEX_LEN;
    if hex_end > coinbase_payload.len() {
        return None;
    }
    let hex_str = std::str::from_utf8(&coinbase_payload[hex_start..hex_end]).ok()?;
    let mut out = [0u8; 32];
    hex::decode_to_slice(hex_str, &mut out).ok()?;
    Some(out)
}

/// Tier-reward multiplier (basis points) for a block, derived from the highest
/// model tier it declares in its coinbase `ai:cap` field. Heavier model ⇒ larger
/// share of the subsidy; an unknown / absent declaration falls back to the floor
/// tier. The legacy IQ3 70B id maps to the same rank as the Q4_K_M id. Pure and
/// deterministic over immutable block content — safe as a consensus input.
fn tier_reward_bps(coinbase_payload: &[u8]) -> u64 {
    let rank = parse_ai_caps(coinbase_payload)
        .iter()
        .filter_map(|id| {
            TIER_MODEL_IDS
                .iter()
                .position(|m| m == id)
                .or_else(|| (*id == LLAMA_3_3_70B_MODEL_ID_LEGACY).then_some(4))
        })
        .max()
        .unwrap_or(0);
    TIER_REWARD_BPS[rank]
}

/// Expected `request_hash` (= `blake2b(payload)[..32]`) of the synthetic task for
/// `(epoch, escrow_pubkey, model_id)`. A v2 AiResponse matching this proves the
/// miner answered its protocol-issued liveness task for that epoch.
fn expected_synthetic_request_hash(epoch: u64, escrow_pubkey: &[u8; 32], model_id: [u8; 32]) -> [u8; 32] {
    let seed = synthetic_seed(epoch, escrow_pubkey);
    // Single-element model set ⇒ the picker always returns `model_id`, so the
    // hash binds the task to exactly the model the miner claims it served.
    let req = derive_synthetic_request(&seed, &[model_id], epoch).expect("single-element model set is non-empty");
    let mut out = [0u8; 32];
    out.copy_from_slice(&blake2b_simd::blake2b(&req.serialize()).as_bytes()[..32]);
    out
}

/// Returns the synthetic epoch in `{epoch, epoch-1}` that `txs` (a block body)
/// answers for `pubkey`, or `None`. Scans the block's v2 `AiResponse` txs for one
/// whose `request_hash` matches the synthetic task derived for `(pubkey, e, model_id)`.
/// Accepting `epoch-1` absorbs inclusion lag across an epoch boundary. Pure over the
/// block body — deterministic and immutable.
fn synthetic_answer_epoch(txs: &[Transaction], pubkey: &[u8; 32], epoch: u64) -> Option<u64> {
    for tx in txs.iter().skip(1) {
        if !tx.is_ai_response() {
            continue;
        }
        let Some(resp) = AiResponsePayload::deserialize(&tx.payload) else { continue };
        let Some(model_id) = resp.model_id else { continue };
        if resp.request_hash == expected_synthetic_request_hash(epoch, pubkey, model_id) {
            return Some(epoch);
        }
        if epoch > 0 && resp.request_hash == expected_synthetic_request_hash(epoch - 1, pubkey, model_id) {
            return Some(epoch - 1);
        }
    }
    None
}

/// Parses a `/live:<64-hex block hash>` liveness reference from a coinbase payload.
/// Returns `None` when the marker is absent or malformed.
fn parse_coinbase_live_ref(coinbase_payload: &[u8]) -> Option<Hash> {
    const MARKER: &[u8] = b"/live:";
    const HEX_LEN: usize = 64;
    let pos = coinbase_payload.windows(MARKER.len()).position(|w| w == MARKER)?;
    let hex_start = pos + MARKER.len();
    let hex_end = hex_start + HEX_LEN;
    if hex_end > coinbase_payload.len() {
        return None;
    }
    let hex_str = std::str::from_utf8(&coinbase_payload[hex_start..hex_end]).ok()?;
    let mut out = [0u8; 32];
    hex::decode_to_slice(hex_str, &mut out).ok()?;
    Some(Hash::from_bytes(out))
}

/// Rejects the block if any AiRequest tx violates inference_reward/priority_fee/escrow rules:
/// - inference_reward below the per-model minimum
/// - priority_fee below MIN_AI_REQUEST_PRIORITY_FEE
/// - UTXO fee < priority_fee (inference_reward now goes to output[1] escrow, not fee)
/// - output[1] missing, not a CSV P2PK script, or value < inference_reward
fn check_ai_request_inference_rewards(
    txs: &[Transaction],
    validated: &[(keryx_consensus_core::tx::ValidatedTransaction<'_>, u32)],
    minimums: &[([u8; 32], u64)],
) -> BlockProcessResult<()> {
    let fee_map: std::collections::HashMap<TransactionId, u64> =
        validated.iter().map(|(vt, _)| (vt.id(), vt.calculated_fee)).collect();

    for tx in txs.iter().skip(1) {
        if !tx.is_ai_request() {
            continue;
        }
        if let Some(req) = AiRequestPayload::deserialize(&tx.payload) {
            // Check inference_reward minimum per model_id, including token-count surcharge.
            // effective_min = base[model] + ceil(max_tokens / 64) * TOKEN_STEP
            if let Some(&(_, base_reward)) = minimums.iter().find(|(id, _)| *id == req.model_id) {
                let token_surcharge = ((req.max_tokens as u64 + 63) / 64) * INFERENCE_REWARD_TOKEN_STEP;
                let effective_min = base_reward + token_surcharge;
                if req.inference_reward < effective_min {
                    return Err(AiRequestInferenceRewardBelowMinimum(
                        tx.id(),
                        req.inference_reward,
                        effective_min,
                        hex::encode(req.model_id),
                    ));
                }
            }
            // Check priority_fee minimum (burned as TX fee).
            if req.priority_fee < keryx_inference::MIN_AI_REQUEST_PRIORITY_FEE {
                return Err(AiRequestPriorityFeeBelowMinimum(
                    tx.id(),
                    req.priority_fee,
                    keryx_inference::MIN_AI_REQUEST_PRIORITY_FEE,
                ));
            }
            // Check UTXO fee covers at least priority_fee (inference_reward is in output[1]).
            if let Some(&calculated_fee) = fee_map.get(&tx.id()) {
                if calculated_fee < req.priority_fee {
                    return Err(AiRequestFeeBelowInferenceReward(tx.id(), calculated_fee, req.priority_fee));
                }
            }
            // Validate escrow output[1]: CSV P2PK script with value >= inference_reward.
            if tx.outputs.len() < 2 {
                return Err(AiRequestMissingEscrowOutput(tx.id()));
            }
            let escrow_out = &tx.outputs[1];
            if !ScriptClass::is_csv_pay_to_pubkey(escrow_out.script_public_key.script()) {
                return Err(AiRequestInvalidEscrowScript(tx.id()));
            }
            if escrow_out.value < req.inference_reward {
                return Err(AiRequestEscrowBelowInferenceReward(tx.id(), escrow_out.value, req.inference_reward));
            }
        }
    }
    Ok(())
}

/// Rejects the block if any `AiResponse` tx uses a model_id not declared in the coinbase
/// `/ai:cap:` field.  Only runs after `model_cap_enforcement_activation`.
///
/// Strategy: build a map `blake2b(AiRequest_payload)[0..32] → model_id` from the
/// AiRequest txs in this block (miners include the requests they answer), then for
/// each AiResponse check its `request_hash` against that map.  If the AiRequest lives
/// in an earlier block the response is skipped — cross-block enforcement is Phase 4.
/// If the miner declared no caps at all (not yet upgraded), enforcement is also skipped.
fn check_ai_response_model_caps(txs: &[Transaction]) -> BlockProcessResult<()> {
    let declared_caps = parse_ai_caps(&txs[0].payload);
    if declared_caps.is_empty() {
        return Ok(());
    }

    // blake2b(AiRequest_payload)[0..32] → model_id
    let mut request_model_map: std::collections::HashMap<[u8; 32], [u8; 32]> =
        std::collections::HashMap::new();
    for tx in txs.iter().skip(1) {
        if tx.is_ai_request() {
            if let Some(req) = AiRequestPayload::deserialize(&tx.payload) {
                let digest = blake2b_simd::blake2b(&tx.payload);
                let mut key = [0u8; 32];
                key.copy_from_slice(&digest.as_bytes()[..32]);
                request_model_map.insert(key, req.model_id);
            }
        }
    }

    for tx in txs.iter().skip(1) {
        if !tx.is_ai_response() {
            continue;
        }
        if let Some(resp) = AiResponsePayload::deserialize(&tx.payload) {
            if let Some(&model_id) = request_model_map.get(&resp.request_hash) {
                if !declared_caps.contains(&model_id) {
                    return Err(AiResponseModelCapMissing(tx.id(), hex::encode(model_id)));
                }
            }
            // request not in same block → AiRequest came from an earlier block → skip
        }
    }
    Ok(())
}

/// OPoI v2 (from `opoi_v2_activation`) — supersedes `check_ai_response_model_caps`.
///
/// Every `AiResponse` tx must:
///   1. parse as the 142-byte v2 payload (model_id + result_commitment present) — the
///      commitment binds the off-chain IPFS content to the chain for the future challenger;
///   2. declare a model_id present in the coinbase `ai:cap:` field — unconditional, so the
///      cross-block escape of the Phase 3 check (request in an earlier block ⇒ no check)
///      is closed without any store lookup (deterministic, block-local);
///   3. when the matching AiRequest is in the same block, claim the same model_id the
///      request asked for. Cross-block, a lying model_id is on-chain provable fraud for
///      the v2 challenger (reveal the request payload hashing to `request_hash`).
fn check_ai_response_v2(txs: &[Transaction]) -> BlockProcessResult<()> {
    let declared_caps = parse_ai_caps(&txs[0].payload);

    // blake2b(AiRequest_payload)[0..32] → model_id, for the same-block consistency check.
    let mut request_model_map: std::collections::HashMap<[u8; 32], [u8; 32]> =
        std::collections::HashMap::new();
    for tx in txs.iter().skip(1) {
        if tx.is_ai_request() {
            if let Some(req) = AiRequestPayload::deserialize(&tx.payload) {
                let digest = blake2b_simd::blake2b(&tx.payload);
                let mut key = [0u8; 32];
                key.copy_from_slice(&digest.as_bytes()[..32]);
                request_model_map.insert(key, req.model_id);
            }
        }
    }

    for tx in txs.iter().skip(1) {
        if !tx.is_ai_response() {
            continue;
        }
        let Some(resp) = AiResponsePayload::deserialize(&tx.payload) else {
            return Err(AiResponsePayloadMalformed(tx.id()));
        };
        let (Some(model_id), Some(_commitment)) = (resp.model_id, resp.result_commitment) else {
            // v1 (78-byte) responses are no longer acceptable past the gate.
            return Err(AiResponsePayloadMalformed(tx.id()));
        };
        if !declared_caps.contains(&model_id) {
            return Err(AiResponseModelCapMissing(tx.id(), hex::encode(model_id)));
        }
        if let Some(&req_model) = request_model_map.get(&resp.request_hash) {
            if req_model != model_id {
                return Err(AiResponseModelMismatch(tx.id(), hex::encode(model_id), hex::encode(req_model)));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use keryx_consensus_core::subnets;

    use super::*;

    // ── synthetic-liveness helper tests (Level-1) ─────────────────────────────

    #[test]
    fn synthetic_request_hash_is_deterministic() {
        let pk = [9u8; 32];
        let model = [3u8; 32];
        assert_eq!(
            expected_synthetic_request_hash(7, &pk, model),
            expected_synthetic_request_hash(7, &pk, model)
        );
    }

    #[test]
    fn synthetic_request_hash_binds_epoch_miner_and_model() {
        let pk = [9u8; 32];
        let other_pk = [10u8; 32];
        let model = [3u8; 32];
        let other_model = [4u8; 32];
        let base = expected_synthetic_request_hash(7, &pk, model);
        assert_ne!(base, expected_synthetic_request_hash(8, &pk, model), "epoch must change the hash");
        assert_ne!(base, expected_synthetic_request_hash(7, &other_pk, model), "miner must change the hash");
        assert_ne!(base, expected_synthetic_request_hash(7, &pk, other_model), "model must change the hash");
    }

    #[test]
    fn node_hash_matches_miner_side_derivation() {
        // A compliant miner derives the same request_hash from the shared seed,
        // so its AiResponse.request_hash will match what the node recomputes.
        let pk = [5u8; 32];
        let model = [2u8; 32];
        let epoch = 42u64;
        let seed = synthetic_seed(epoch, &pk);
        let req = derive_synthetic_request(&seed, &[model], epoch).unwrap();
        let mut miner_side = [0u8; 32];
        miner_side.copy_from_slice(&blake2b_simd::blake2b(&req.serialize()).as_bytes()[..32]);
        assert_eq!(miner_side, expected_synthetic_request_hash(epoch, &pk, model));
    }

    // ── tier-reward helper tests ──────────────────────────────────────────────

    /// Builds a coinbase-shaped payload declaring the given model_ids in `ai:cap`.
    /// The 21-byte prefix clears the 19-byte binary header `parse_ai_caps` skips.
    fn caps_payload(ids: &[[u8; 32]]) -> Vec<u8> {
        let caps = ids.iter().map(hex::encode).collect::<Vec<_>>().join(",");
        format!("HDR_padding_19+++++++/ai:cap:{caps}/").into_bytes()
    }

    #[test]
    fn tier_reward_bps_maps_each_tier() {
        for (rank, id) in TIER_MODEL_IDS.iter().enumerate() {
            assert_eq!(tier_reward_bps(&caps_payload(&[*id])), TIER_REWARD_BPS[rank], "rank {rank}");
        }
    }

    #[test]
    fn tier_reward_bps_takes_the_highest_declared_tier() {
        // Declaring a light model alongside the flagship earns the flagship's share.
        let payload = caps_payload(&[TIER_MODEL_IDS[0], TIER_MODEL_IDS[5], TIER_MODEL_IDS[2]]);
        assert_eq!(tier_reward_bps(&payload), TIER_REWARD_BPS[5]);
    }

    #[test]
    fn tier_reward_bps_legacy_70b_maps_to_ultra_rank() {
        assert_eq!(tier_reward_bps(&caps_payload(&[LLAMA_3_3_70B_MODEL_ID_LEGACY])), TIER_REWARD_BPS[4]);
    }

    #[test]
    fn tier_reward_bps_floors_unknown_or_absent() {
        // No ai:cap field at all, and a declared-but-unrecognised id, both floor.
        assert_eq!(tier_reward_bps(b"HDR_padding_19+++++++/no/caps/here"), TIER_REWARD_BPS[0]);
        assert_eq!(tier_reward_bps(&caps_payload(&[[0xEEu8; 32]])), TIER_REWARD_BPS[0]);
        assert_eq!(
            TIER_REWARD_BPS[5],
            keryx_consensus_core::config::params::TIER_REWARD_BPS_DIVISOR,
            "top tier must be the 100% reference"
        );
    }

    #[test]
    fn coinbase_escrow_pubkey_parsing() {
        let pk = [0xABu8; 32];
        let payload = format!("0.2.8/2025-01-01/deadbeef/ai:v1:aabbccdd11223344/escrow:{}", hex::encode(pk));
        assert_eq!(parse_coinbase_escrow_pubkey(payload.as_bytes()), Some(pk));
        // No marker, and a truncated hex tail, both yield None.
        assert_eq!(parse_coinbase_escrow_pubkey(b"0.2.8/no/escrow/here"), None);
        let truncated = "x/escrow:abcd";
        assert_eq!(parse_coinbase_escrow_pubkey(truncated.as_bytes()), None);
    }

    #[test]
    fn coinbase_live_ref_parsing() {
        let h = [0xCDu8; 32];
        let payload = format!("0.2.8/escrow:{}/live:{}", hex::encode([1u8; 32]), hex::encode(h));
        assert_eq!(parse_coinbase_live_ref(payload.as_bytes()), Some(Hash::from_bytes(h)));
        assert_eq!(parse_coinbase_live_ref(b"no live ref"), None);
        assert_eq!(parse_coinbase_live_ref(b"/live:abcd"), None); // truncated
    }

    #[test]
    fn synthetic_answer_epoch_matches_only_right_miner_and_epoch() {
        let pubkey = [9u8; 32];
        let model = [3u8; 32];
        let epoch = 5u64;
        // Build a v2 AiResponse carrying the synthetic request_hash for (epoch, pubkey, model).
        let request_hash = expected_synthetic_request_hash(epoch, &pubkey, model);
        let resp = AiResponsePayload::new_v2(request_hash, 1000, dummy_cid(), 16, model, [0u8; 32]);
        let resp_tx = Transaction::new(0, vec![], vec![], 0, subnets::SUBNETWORK_ID_AI_RESPONSE, 0, resp.serialize());
        let txs = vec![make_coinbase_no_caps(), resp_tx];

        assert_eq!(synthetic_answer_epoch(&txs, &pubkey, epoch), Some(epoch));
        // One epoch ahead still matches the answer as the previous epoch (inclusion lag).
        assert_eq!(synthetic_answer_epoch(&txs, &pubkey, epoch + 1), Some(epoch));
        // Wrong miner or an unrelated epoch ⇒ no match.
        assert_eq!(synthetic_answer_epoch(&txs, &[7u8; 32], epoch), None);
        assert_eq!(synthetic_answer_epoch(&txs, &pubkey, epoch + 5), None);
    }

    // ── helpers ───────────────────────────────────────────────────────────────

    fn make_coinbase_with_caps(model_ids: &[[u8; 32]]) -> Transaction {
        let mut payload = vec![0u8; 53];
        let caps_str = model_ids.iter().map(hex::encode).collect::<Vec<_>>().join(",");
        let extra = format!("0.2.8/2025-01-01/00000000deadbeef/ai:v1:aabbccdd11223344/ai:cap:{}", caps_str);
        payload.extend_from_slice(extra.as_bytes());
        Transaction::new(0, vec![], vec![], 0, subnets::SUBNETWORK_ID_COINBASE, 0, payload)
    }

    fn make_coinbase_no_caps() -> Transaction {
        let mut payload = vec![0u8; 53];
        payload.extend_from_slice(b"0.2.8/2025-01-01/00000000deadbeef/ai:v1:aabbccdd11223344");
        Transaction::new(0, vec![], vec![], 0, subnets::SUBNETWORK_ID_COINBASE, 0, payload)
    }

    fn dummy_cid() -> [u8; 34] {
        let mut cid = [0u8; 34];
        cid[0] = 0x12;
        cid[1] = 0x20;
        cid
    }

    fn make_ai_request(model_id: [u8; 32]) -> Transaction {
        let req = AiRequestPayload::new(model_id, 100, 1_000_000, 30_000_000, b"test prompt".to_vec());
        Transaction::new(0, vec![], vec![], 0, subnets::SUBNETWORK_ID_AI_REQUEST, 0, req.serialize())
    }

    fn make_ai_response_for(request_tx: &Transaction) -> Transaction {
        let digest = blake2b_simd::blake2b(&request_tx.payload);
        let mut request_hash = [0u8; 32];
        request_hash.copy_from_slice(&digest.as_bytes()[..32]);
        let resp = AiResponsePayload::new(request_hash, 1000, dummy_cid(), 128);
        Transaction::new(0, vec![], vec![], 0, subnets::SUBNETWORK_ID_AI_RESPONSE, 0, resp.serialize())
    }

    fn make_ai_response_orphan(request_hash: [u8; 32]) -> Transaction {
        let resp = AiResponsePayload::new(request_hash, 1000, dummy_cid(), 128);
        Transaction::new(0, vec![], vec![], 0, subnets::SUBNETWORK_ID_AI_RESPONSE, 0, resp.serialize())
    }

    // ── check_ai_response_model_caps ─────────────────────────────────────────

    #[test]
    fn no_caps_declared_skips_enforcement() {
        let model_id = [0xAAu8; 32];
        let req = make_ai_request(model_id);
        let resp = make_ai_response_for(&req);
        let txs = vec![make_coinbase_no_caps(), req, resp];
        assert!(check_ai_response_model_caps(&txs).is_ok());
    }

    #[test]
    fn declared_model_is_accepted() {
        let model_id = [0x11u8; 32];
        let req = make_ai_request(model_id);
        let resp = make_ai_response_for(&req);
        let txs = vec![make_coinbase_with_caps(&[model_id]), req, resp];
        assert!(check_ai_response_model_caps(&txs).is_ok());
    }

    #[test]
    fn undeclared_model_is_rejected() {
        let declared = [0x22u8; 32];
        let used = [0x33u8; 32];
        let req = make_ai_request(used);
        let resp = make_ai_response_for(&req);
        let txs = vec![make_coinbase_with_caps(&[declared]), req, resp];
        assert!(matches!(check_ai_response_model_caps(&txs), Err(AiResponseModelCapMissing(_, _))));
    }

    #[test]
    fn response_for_request_from_earlier_block_is_skipped() {
        let model_id = [0x44u8; 32];
        let orphan_hash = [0xFFu8; 32];
        let resp = make_ai_response_orphan(orphan_hash);
        let txs = vec![make_coinbase_with_caps(&[model_id]), resp];
        assert!(check_ai_response_model_caps(&txs).is_ok());
    }

    #[test]
    fn multiple_responses_one_undeclared_is_rejected() {
        let declared = [0x55u8; 32];
        let undeclared = [0x66u8; 32];
        let req_ok = make_ai_request(declared);
        let req_bad = make_ai_request(undeclared);
        let resp_ok = make_ai_response_for(&req_ok);
        let resp_bad = make_ai_response_for(&req_bad);
        let txs = vec![make_coinbase_with_caps(&[declared]), req_ok, req_bad, resp_ok, resp_bad];
        assert!(matches!(check_ai_response_model_caps(&txs), Err(AiResponseModelCapMissing(_, _))));
    }

    // ── check_ai_response_v2 ─────────────────────────────────────────────────

    fn make_ai_response_v2_for(request_tx: &Transaction, model_id: [u8; 32]) -> Transaction {
        let digest = blake2b_simd::blake2b(&request_tx.payload);
        let mut request_hash = [0u8; 32];
        request_hash.copy_from_slice(&digest.as_bytes()[..32]);
        let resp = AiResponsePayload::new_v2(request_hash, 1000, dummy_cid(), 128, model_id, [0xCCu8; 32]);
        Transaction::new(0, vec![], vec![], 0, subnets::SUBNETWORK_ID_AI_RESPONSE, 0, resp.serialize())
    }

    fn make_ai_response_v2_orphan(request_hash: [u8; 32], model_id: [u8; 32]) -> Transaction {
        let resp = AiResponsePayload::new_v2(request_hash, 1000, dummy_cid(), 128, model_id, [0xCCu8; 32]);
        Transaction::new(0, vec![], vec![], 0, subnets::SUBNETWORK_ID_AI_RESPONSE, 0, resp.serialize())
    }

    #[test]
    fn v2_accepts_declared_model_same_block() {
        let model = [1u8; 32];
        let req = make_ai_request(model);
        let resp = make_ai_response_v2_for(&req, model);
        let txs = vec![make_coinbase_with_caps(&[model]), req, resp];
        assert!(check_ai_response_v2(&txs).is_ok());
    }

    #[test]
    fn v2_rejects_v1_payload_past_gate() {
        let model = [1u8; 32];
        let req = make_ai_request(model);
        let resp_v1 = make_ai_response_for(&req);
        let txs = vec![make_coinbase_with_caps(&[model]), req, resp_v1];
        assert!(matches!(check_ai_response_v2(&txs), Err(AiResponsePayloadMalformed(_))));
    }

    #[test]
    fn v2_rejects_undeclared_model_cross_block() {
        // Request lives in an earlier block (orphan response here) — Phase 3 skipped
        // this case entirely; v2 must still reject an undeclared model_id.
        let declared = [1u8; 32];
        let undeclared = [2u8; 32];
        let resp = make_ai_response_v2_orphan([0xABu8; 32], undeclared);
        let txs = vec![make_coinbase_with_caps(&[declared]), resp];
        assert!(matches!(check_ai_response_v2(&txs), Err(AiResponseModelCapMissing(_, _))));
    }

    #[test]
    fn v2_accepts_declared_model_cross_block() {
        let declared = [1u8; 32];
        let resp = make_ai_response_v2_orphan([0xABu8; 32], declared);
        let txs = vec![make_coinbase_with_caps(&[declared]), resp];
        assert!(check_ai_response_v2(&txs).is_ok());
    }

    #[test]
    fn v2_rejects_model_mismatch_same_block() {
        // Miner declares both models, request asks for A, response claims B → mismatch.
        let model_a = [1u8; 32];
        let model_b = [2u8; 32];
        let req = make_ai_request(model_a);
        let resp = make_ai_response_v2_for(&req, model_b);
        let txs = vec![make_coinbase_with_caps(&[model_a, model_b]), req, resp];
        assert!(matches!(check_ai_response_v2(&txs), Err(AiResponseModelMismatch(_, _, _))));
    }

    #[test]
    fn v2_rejects_response_when_no_caps_declared() {
        // A non-upgraded miner (no ai:cap field) cannot carry AiResponse txs past the gate.
        let model = [1u8; 32];
        let resp = make_ai_response_v2_orphan([0xABu8; 32], model);
        let txs = vec![make_coinbase_no_caps(), resp];
        assert!(matches!(check_ai_response_v2(&txs), Err(AiResponseModelCapMissing(_, _))));
    }

    #[test]
    fn test_rayon_reduce_retains_order() {
        // this is an independent test to replicate the behavior of
        // validate_txs_in_parallel and validate_txs_with_muhash_in_parallel
        // and assert that the order of data is retained when doing par_iter
        let data: Vec<u16> = (1..=1000).collect();

        let collected: Vec<u16> = data
            .par_iter()
            .filter_map(|a| {
                let chance: f64 = rand::random();
                if chance < 0.05 {
                    return None;
                }
                Some(*a)
            })
            .collect();

        println!("collected len: {}", collected.len());

        collected.iter().tuple_windows().for_each(|(prev, curr)| {
            // Data was originally sorted, so we check if they remain sorted after filtering
            assert!(prev < curr, "expected {} < {} if original sort was preserved", prev, curr);
        });

        let reduced: SmallVec<[u16; 2]> = data
            .par_iter()
            .filter_map(|a: &u16| {
                let chance: f64 = rand::random();
                if chance < 0.05 {
                    return None;
                }
                Some(smallvec![*a])
            })
            .reduce(
                || smallvec![],
                |mut arr, mut curr_data| {
                    arr.append(&mut curr_data);
                    arr
                },
            );

        println!("reduced len: {}", reduced.len());

        reduced.iter().tuple_windows().for_each(|(prev, curr)| {
            // Data was originally sorted, so we check if they remain sorted after filtering
            assert!(prev < curr, "expected {} < {} if original sort was preserved", prev, curr);
        });
    }
}
