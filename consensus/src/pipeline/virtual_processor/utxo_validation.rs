use super::VirtualStateProcessor;
use crate::{
    errors::{
        BlockProcessResult,
        RuleError::{
            AiRequestEscrowBelowInferenceReward, AiRequestFeeBelowInferenceReward,
            AiRequestInferenceRewardBelowMinimum, AiRequestInvalidEscrowScript,
            AiRequestMissingEscrowOutput, AiRequestPriorityFeeBelowMinimum,
            AiResponseModelCapMissing, BadAcceptedIDMerkleRoot, BadCoinbaseTransaction, BadUTXOCommitment,
            InvalidTransactionsInUtxoContext, WrongHeaderPruningPoint,
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
use crate::model::stores::address_amount::AddressAmountStoreReader;
use crate::model::stores::ai_slash::{AiResponseRecord, AiResponseStore, AiResponseStoreReader};
use crate::model::stores::pom_tier::PomTierStoreReader;
use crate::model::stores::selected_chain::SelectedChainStoreReader;
use keryx_consensus_core::config::params::{TIER_REWARD_BPS_DIVISOR, ratio_reward_bps, tier_reward_bps};
use keryx_database::prelude::StoreResultExt;
use keryx_consensus_core::{
    BlockHashMap, BlockHashSet, ChainPath, HashMapCustomHasher,
    acceptance_data::{AcceptedTxEntry, MergesetBlockAcceptanceData},
    api::args::TransactionValidationArgs,
    coinbase::*,
    hashing,
    header::Header,
    muhash::MuHashExtensions,
    tx::{MutableTransaction, PopulatedTransaction, ScriptPublicKey, Transaction, TransactionId, ValidatedTransaction, VerifiableTransaction},
    utxo::{
        utxo_diff::UtxoDiff,
        utxo_view::{UtxoView, UtxoViewComposition},
    },
};
use keryx_core::{info, trace, warn};
use keryx_hashes::Hash;
use keryx_inference::{AiRequestPayload, AiResponsePayload, INFERENCE_REWARD_TOKEN_STEP, parse_ai_caps};
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
            self.process_ai_txs_for_slash(&txs, pov_daa_score, coinbase_tx_id);
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
        // Diff from the committed virtual to this block's selected parent (for ratio-reward balances).
        sp_diff: &UtxoDiff,
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

        // Verify coinbase transaction. The two diffs (committed-virtual → selected parent, then
        // selected parent → this block via its mergeset) let the ratio-reward bracket be evaluated at
        // this block's own view from the virtual-anchored balance index.
        //
        // Skipped while `trust_coinbase()` holds (archival node, `KERYX_TRUST_COINBASE` operator
        // opt-in, or still inside our own fast-sync production-index catch-up window — see its doc):
        // the `utxo_commitment` verified above already pins this block's resulting UTXO set to the
        // canonical chain, so the block's coinbase outputs are trusted without re-deriving the ratio
        // bracket — which such a node cannot yet reproduce for the post-fork canonical chain.
        if !self.trust_coinbase() {
            self.verify_coinbase_transaction(
                &txs[0],
                header.daa_score,
                &ctx.ghostdag_data,
                &ctx.mergeset_rewards,
                &self.daa_excluded_store.get_mergeset_non_daa(header.hash).unwrap(),
                &[sp_diff, &ctx.mergeset_diff],
            )?;
        }

        // Verify the header pruning point
        let reply = self.verify_header_pruning_point(header, ctx.ghostdag_data.to_compact())?;
        ctx.pruning_sample_from_pov = Some(reply.pruning_sample);

        // SALT v2 hardfork: log once at the exact activation DAA score.
        if header.daa_score == self.pow_salt_v2_activation.daa_score() {
            info!(
                "=== SALT v2 HARDFORK ACTIVATED at DAA {} — KeryxHash domain salt switched to v2, pre-v1.2.2 miners now rejected ===",
                header.daa_score
            );
        }

        // SALT v4 hardfork (chain relaunch on stock difficulty): log once at activation.
        if header.daa_score == self.pow_salt_v4_activation.daa_score() {
            info!(
                "=== SALT v4 HARDFORK ACTIVATED at DAA {} — KeryxHash salt switched to v4, stock difficulty (no reset); chain relaunched off the abandoned SALT-v3 spiral, older binaries now rejected ===",
                header.daa_score
            );
        }

        // Bundled hardfork (OPoI v2 + PoM + holder-reward share one mainnet activation DAA). Emit a
        // single consolidated banner listing whichever of the three activate exactly at this block's
        // DAA score. The gates are independent fields, so on a network that staggers them the banner
        // still fires correctly at each distinct activation DAA; `never()` (= u64::MAX) never matches.
        {
            let mut lines: Vec<&str> = Vec::new();
            if header.daa_score == self.pom_activation.daa_score() {
                lines.push("  PoM           — Proof-of-Model mining live; kHeavyHash retired (1 GPU = 1 tier); non-PoM miners rejected");
            }
            if header.daa_score == self.opoi_v2_activation.daa_score() {
                lines.push("  OPoI v2       — uncensored model lineup now enforced");
            }
            if header.daa_score == self.ratio_reward_activation.daa_score() {
                lines.push("  Holder-reward — miner cut weighted by KRX holdings; the shortfall is burned");
            }
            if !lines.is_empty() {
                info!("════════════════ KERYX HARDFORK · DAA {} ════════════════", header.daa_score);
                for line in lines {
                    info!("{line}");
                }
                info!("═══════════════════════════════════════════════════════════════");
            }
        }

        // OPoI Phase 3 hardfork: enforce model capability declarations after activation.
        if self.model_cap_enforcement_activation.is_active(header.daa_score) {
            if header.daa_score == self.model_cap_enforcement_activation.daa_score() {
                info!(
                    "=== OPoI HARDFORK ACTIVATED at DAA {} — UTXO escrow + model cap enforcement now live ===",
                    header.daa_score
                );
            }
            self.check_ai_response_model_caps(&txs)?;
        }

        // Verify all transactions are valid in context
        let current_utxo_view = selected_parent_utxo_view.compose(&ctx.mergeset_diff);
        let validated_transactions =
            self.validate_transactions_in_parallel(&txs, &current_utxo_view, header.daa_score, TxValidationFlags::Full);
        if validated_transactions.len() < txs.len() - 1 {
            // Some non-coinbase transactions are invalid
            return Err(InvalidTransactionsInUtxoContext(txs.len() - 1 - validated_transactions.len(), txs.len() - 1));
        }

        // Enforce AiRequest inference_reward minimums and fee coverage after activation.
        if self.model_cap_enforcement_activation.is_active(header.daa_score) {
            // OPoI v2 hardfork: swap to the uncensored lineup at/after activation. DAA-gated so
            // IBD re-validates historical (pre-v2) blocks against the legacy lineup unchanged.
            // (Activation is announced by the consolidated KERYX HARDFORK banner above.)
            let minimums = if self.opoi_v2_activation.is_active(header.daa_score) {
                self.inference_reward_minimums_v2
            } else {
                self.inference_reward_minimums
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
        // Diffs from the committed virtual to this block's own view, for ratio-reward balances.
        view_diffs: &[&UtxoDiff],
    ) -> BlockProcessResult<()> {
        // Extract only miner data from the provided coinbase
        let miner_data = self.coinbase_manager.deserialize_coinbase_payload(&coinbase.payload).unwrap().miner_data;
        let tier_bps_by_block = self.tier_bps_by_block(ghostdag_data, mergeset_non_daa, daa_score);
        let ratio_bps_by_block = self.ratio_bps_by_block(ghostdag_data, mergeset_non_daa, mergeset_rewards, daa_score, view_diffs);
        let expected_coinbase = self
            .coinbase_manager
            .expected_coinbase_transaction(
                daa_score,
                miner_data,
                ghostdag_data,
                mergeset_rewards,
                mergeset_non_daa,
                &tier_bps_by_block,
                &ratio_bps_by_block,
            )
            .unwrap()
            .tx;
        if hashing::tx::hash(coinbase) != hashing::tx::hash(&expected_coinbase) {
            // TEMP DIAGNOSTIC: pinpoint why the coinbase differs (tier vs ratio vs amounts).
            warn!(
                "COINBASE MISMATCH daa={} tier_bps={:?} ratio_bps={:?} actual_outs={:?} expected_outs={:?}",
                daa_score,
                tier_bps_by_block,
                ratio_bps_by_block,
                coinbase.outputs.iter().map(|o| o.value).collect::<Vec<_>>(),
                expected_coinbase.outputs.iter().map(|o| o.value).collect::<Vec<_>>(),
            );
            Err(BadCoinbaseTransaction)
        } else {
            Ok(())
        }
    }

    /// Tier-reward map consumed by `expected_coinbase_transaction`: for each rewarded blue, the
    /// subsidy multiplier (bps) of its cryptographically-proven PoM tier (persisted at body commit
    /// in `pom_tier_store`). Both the validator and the template builder derive it identically from
    /// the same store, so the coinbase they produce agrees deterministically. Returns an empty map
    /// before `pom_activation` (⇒ every miner cut paid in full, no penalty, no burn). A blue with no
    /// stored tier (cannot happen for a valid post-fork block — `check_pom_proof` requires the proof)
    /// is simply left out, falling back to the full cut on the coinbase side.
    pub(super) fn tier_bps_by_block(
        &self,
        ghostdag_data: &GhostdagData,
        mergeset_non_daa: &BlockHashSet,
        pov_daa_score: u64,
    ) -> BlockHashMap<u64> {
        let mut map = BlockHashMap::new();
        if !self.pom_activation.is_active(pov_daa_score) {
            return map;
        }
        // Reward schedule gated per block by `very_light_activation` (5-tier H2 vs legacy 4-tier),
        // keyed on this block's own daa_score to match `pom_tiers` and stay canonical under IBD.
        let schedule = tier_reward_bps(self.very_light_activation.is_active(pov_daa_score));
        for blue in ghostdag_data.mergeset_blues.iter().filter(|h| !mergeset_non_daa.contains(h)) {
            if let Some(tier) = self.pom_tier_store.get(*blue).optional().unwrap() {
                let bps = schedule.get(tier as usize).copied().unwrap_or(TIER_REWARD_BPS_DIVISOR);
                map.insert(*blue, bps);
            }
        }
        map
    }

    /// Ratio-reward map consumed by `expected_coinbase_transaction`: for each rewarded blue, the
    /// holder-ratio bracket multiplier (bps), computed **inline at this (rewarding) block's view**
    /// from the consensus balance + production indexes — NOT read from any per-block store.
    ///
    /// Why inline (Stage 2b option B): a per-block stored bracket would have to be written for every
    /// blue, but blues are only UTXO-committed when they sit on the selected chain. A side-blue's
    /// stored value would be missing — or, worse, non-deterministically present after a reorg
    /// (whether a block was ever a transient chain candidate depends on each node's processing
    /// order) — which diverges the expected coinbase across nodes → consensus split. Computing the
    /// bracket inline from each rewarding block's own (intrinsic, reorg-stable) view removes the
    /// store entirely and covers every blue identically on all nodes.
    ///
    /// `view_diffs` are the UTXO diffs from the node's committed virtual to THIS block's view (empty
    /// on the build path, where the rewarding block is virtual itself). Per blue, the balance at
    /// this view = the virtual-anchored balance index corrected by those diffs, restricted to the
    /// blue's payout SPK (taken from `mergeset_rewards`, already derived for the coinbase). Returns
    /// an empty map before `ratio_reward_activation` (⇒ full miner cut, no penalty). Compounds with
    /// `tier_bps_by_block` in the coinbase manager.
    pub(super) fn ratio_bps_by_block(
        &self,
        ghostdag_data: &GhostdagData,
        mergeset_non_daa: &BlockHashSet,
        mergeset_rewards: &BlockHashMap<BlockRewardData>,
        pov_daa_score: u64,
        view_diffs: &[&UtxoDiff],
    ) -> BlockHashMap<u64> {
        let mut map = BlockHashMap::new();
        if !self.ratio_reward_activation.is_active(pov_daa_score) {
            return map;
        }
        // Windowed-production correction (Stage 2b-2b): the production index lags at the committed
        // virtual's selected chain; correct every payout SPK to THIS block's selected-parent window
        // with one chain-path delta, shared by all blues (mirror of the balance `view_diffs`). Floor
        // at one block's base miner cut so a newcomer with no recent production divides by one block.
        let prod_correction = self.production_window_correction(ghostdag_data.selected_parent);
        let prod_floor = self.coinbase_manager.base_miner_cut(pov_daa_score).max(1);
        for blue in ghostdag_data.mergeset_blues.iter().filter(|h| !mergeset_non_daa.contains(h)) {
            // Payout SPK = the blue's own miner cut, already resolved into the reward data.
            if let Some(reward) = mergeset_rewards.get(blue) {
                let spk = &reward.script_public_key;
                let base = self.address_balance_store.get(spk).unwrap() as i128;
                let delta: i128 = view_diffs.iter().map(|d| balance_delta_for_spk(d, spk)).sum();
                let balance = (base + delta).max(0) as u64;
                let prod_base = self.windowed_production_store.get(spk).unwrap() as i128;
                let prod_delta = prod_correction.get(spk).copied().unwrap_or(0);
                let production = ((prod_base + prod_delta).max(0) as u64).max(prod_floor);
                map.insert(*blue, ratio_reward_bps(balance, production));
            }
        }

        // Optional self-check (env KERYX_RATIO_SELFCHECK=1): verify the maintained windowed-production
        // index (store + correction) equals the DIRECT window recompute for each rewarded blue. Catches
        // any residual non-determinism live. O(W) per call — enable only briefly to validate a relaunch.
        if std::env::var("KERYX_RATIO_SELFCHECK").is_ok() {
            let sc = self.selected_chain_store.read();
            if let Ok(sp_idx) = sc.get_by_hash(ghostdag_data.selected_parent) {
                let w = self.ratio_reward_window;
                let lo = sp_idx.saturating_sub(w - 1).max(1);
                let mut direct: std::collections::HashMap<ScriptPublicKey, u64> = std::collections::HashMap::new();
                for i in lo..=sp_idx {
                    if let Ok(h) = sc.get_by_index(i) {
                        if let Some((spk, cut)) = self.block_production(h) {
                            *direct.entry(spk).or_default() += cut;
                        }
                    }
                }
                for blue in ghostdag_data.mergeset_blues.iter().filter(|h| !mergeset_non_daa.contains(h)) {
                    if let Some(reward) = mergeset_rewards.get(blue) {
                        let spk = &reward.script_public_key;
                        let used = ((self.windowed_production_store.get(spk).unwrap() as i128
                            + prod_correction.get(spk).copied().unwrap_or(0))
                        .max(0) as u64)
                            .max(prod_floor);
                        let truth = direct.get(spk).copied().unwrap_or(0).max(prod_floor);
                        if used != truth {
                            warn!(
                                "RATIO-SELFCHECK MISMATCH daa={} blue={} used_prod={} direct_prod={} drift={}",
                                pov_daa_score, blue, used, truth, used as i128 - truth as i128
                            );
                        }
                    }
                }
            }
        }
        map
    }

    /// Reads selected-chain block `hash`'s production contribution: its producer payout SPK (the
    /// `miner_data` SPK in its own coinbase) and the base (un-scaled) miner cut of one block subsidy
    /// at its DAA score. `None` if that base cut is 0 (tail emission edge) ⇒ no contribution. This is
    /// the per-block unit summed by the windowed-production index (one number per chain block,
    /// attributed to its producer — deliberately not the per-output paid amount, see `base_miner_cut`).
    pub(super) fn block_production(&self, hash: Hash) -> Option<(ScriptPublicKey, u64)> {
        let cut = self.coinbase_manager.base_miner_cut(self.headers_store.get_daa_score(hash).unwrap());
        if cut == 0 {
            return None;
        }
        let txs = self.block_transactions_store.get(hash).unwrap();
        let spk = self.coinbase_manager.deserialize_coinbase_payload(&txs[0].payload).unwrap().miner_data.script_public_key;
        Some((spk, cut))
    }

    /// Folds the per-SPK production delta of moving the windowed-production index along `chain_path`
    /// (from a chain whose tip has index `from_tip`) into `deltas`. The window is the last
    /// `RATIO_REWARD_WINDOW` selected-chain blocks (a block count, slid in O(1) via the chain index):
    /// - **top**: `removed` blocks leave the window (subtract), `added` blocks join it (add);
    /// - **bottom**: as the tip moves by `Δ = added − removed`, the low boundary slides by the same
    ///   amount — blocks at indices `(from_tip−W, to_tip−W]` exit (subtract) when `Δ>0`, or re-enter
    ///   (add) when `Δ<0`. Those deep blocks sit far below any reorg split, so reading them by index
    ///   on `sc` (the pre-change chain) is reorg-stable and identical on every node.
    fn fold_production_window_delta(
        &self,
        chain_path: &ChainPath,
        from_tip: u64,
        sc: &impl SelectedChainStoreReader,
        deltas: &mut std::collections::HashMap<ScriptPublicKey, i128>,
    ) {
        let w = self.ratio_reward_window as i128;
        let from_tip = from_tip as i128;
        let removed_len = chain_path.removed.len() as i128;
        let added_len = chain_path.added.len() as i128;
        let common = from_tip - removed_len; // common-ancestor index (shared chain below the reorg)
        let to_tip = common + added_len;

        // Disjoint-window case: when the reorg/jump exceeds the window (`added` or `removed` > W — e.g.
        // a reorg across the difficulty-reset genesis-diff burst), the incremental top+bottom-slide
        // below is unsafe: it would (a) subtract/add `removed`/`added` blocks that lie BELOW the window
        // — over-counting, which the saturating store then silently loses — and (b) read bottom-slide
        // indices past the reorg split, which aren't on the pre-change `sc`. Recompute both windows
        // directly instead (subtract the whole old window, add the whole new window). O(W), only on
        // such large jumps; the per-block path keeps the O(1) incremental branch below.
        if added_len > w || removed_len > w {
            // Old window [from_tip-W+1, from_tip] — entirely on `sc` (the pre-change chain).
            for i in (from_tip - w + 1).max(1)..=from_tip {
                if let Some((spk, cut)) = self.block_production(sc.get_by_index(i as u64).unwrap()) {
                    *deltas.entry(spk).or_default() -= cut as i128;
                }
            }
            // New window [to_tip-W+1, to_tip]: indices ≤ common come from `sc` (shared low chain),
            // indices > common come from `added` (added[k] is at index common+1+k, ascending).
            let new_lo = (to_tip - w + 1).max(1);
            for i in new_lo..=common {
                if let Some((spk, cut)) = self.block_production(sc.get_by_index(i as u64).unwrap()) {
                    *deltas.entry(spk).or_default() += cut as i128;
                }
            }
            let first_added_k = (new_lo - (common + 1)).max(0) as usize;
            for h in chain_path.added.iter().skip(first_added_k) {
                if let Some((spk, cut)) = self.block_production(*h) {
                    *deltas.entry(spk).or_default() += cut as i128;
                }
            }
            return;
        }

        // Overlap case (`added` and `removed` ≤ W): every removed/added block falls inside its window,
        // so the top is a straight subtract/add, and the bottom shifts by only `Δ = added − removed`.
        for h in &chain_path.removed {
            if let Some((spk, cut)) = self.block_production(*h) {
                *deltas.entry(spk).or_default() -= cut as i128;
            }
        }
        for h in &chain_path.added {
            if let Some((spk, cut)) = self.block_production(*h) {
                *deltas.entry(spk).or_default() += cut as i128;
            }
        }

        // Bottom slide: as the tip moves, the window's low boundary slides by the same amount — see
        // `window_bottom_slide` for the exact index range + sign (the off-by-one-prone arithmetic,
        // unit-tested in isolation). Window blocks are retained on disk (W < pruning depth).
        if let Some((sign, lo, hi)) = window_bottom_slide(from_tip, to_tip, w) {
            for i in lo..=hi {
                if let Some((spk, cut)) = self.block_production(sc.get_by_index(i as u64).unwrap()) {
                    *deltas.entry(spk).or_default() += sign * cut as i128;
                }
            }
        }
    }

    /// Ratio-reward (Stage 2b-2b) — advances the windowed-production index along `chain_path`, kept in
    /// lockstep with the selected chain (called from `commit_virtual_state` in the SAME batch as the
    /// selected-chain `apply_changes`, BEFORE it runs, so `sc` still reflects the pre-change chain).
    /// Ungated/passive: maintained from genesis so it is exact for from-genesis nodes; only read once
    /// `ratio_reward_activation` fires. Folds one net delta per producer SPK, then writes each once
    /// (a value returning to 0 deletes its entry via `set_batch`).
    pub(super) fn advance_production_window(
        &self,
        batch: &mut rocksdb::WriteBatch,
        chain_path: &ChainPath,
        sc: &impl SelectedChainStoreReader,
    ) {
        let from_tip = sc.get_tip().unwrap().0;
        let mut deltas: std::collections::HashMap<ScriptPublicKey, i128> = std::collections::HashMap::new();
        self.fold_production_window_delta(chain_path, from_tip, sc, &mut deltas);
        for (spk, delta) in deltas {
            if delta == 0 {
                continue;
            }
            let new_value = (self.windowed_production_store.get(&spk).unwrap() as i128 + delta).max(0) as u64;
            self.windowed_production_store.set_batch(batch, &spk, new_value).unwrap();
        }
    }

    /// Ratio-reward (Stage 2b-2b) — per-SPK production delta from the committed virtual's window to
    /// block `m_sp`'s window (`m_sp` = the rewarding block's selected parent). The windowed-production
    /// index is anchored at the committed virtual (its selected-chain tip); this corrects it to the
    /// rewarding block's own view, exactly mirroring how the balance `view_diffs` correct the balance
    /// index. Empty when `m_sp` is already the committed tip (the build path, where the rewarding
    /// block is virtual itself). Deterministic: a function of `m_sp`'s intrinsic DAG position.
    fn production_window_correction(&self, m_sp: Hash) -> std::collections::HashMap<ScriptPublicKey, i128> {
        let mut deltas = std::collections::HashMap::new();
        let sc = self.selected_chain_store.read();
        let (committed_tip_index, committed_tip) = sc.get_tip().unwrap();
        if m_sp == committed_tip {
            return deltas;
        }
        let chain_path = self.dag_traversal_manager.calculate_chain_path(committed_tip, m_sp, None);
        self.fold_production_window_delta(&chain_path, committed_tip_index, &*sc, &mut deltas);
        deltas
    }

    /// Ratio-reward (Stage 2b) — advances the balance index by `diff`, keeping it in lockstep with
    /// the virtual UTXO set (called from `commit_virtual_state` with the same `accumulated_diff`, in
    /// the same batch). Folds the diff into one net delta per payout SPK, then read-modify-writes
    /// each touched address once; a balance returning to 0 deletes its entry (via `set_batch`).
    pub(super) fn apply_balance_diff(&self, batch: &mut rocksdb::WriteBatch, diff: &UtxoDiff) {
        let mut deltas: std::collections::HashMap<ScriptPublicKey, i128> = std::collections::HashMap::new();
        for entry in diff.add.values() {
            *deltas.entry(entry.script_public_key.clone()).or_default() += entry.amount as i128;
        }
        for entry in diff.remove.values() {
            *deltas.entry(entry.script_public_key.clone()).or_default() -= entry.amount as i128;
        }
        for (spk, delta) in deltas {
            if delta == 0 {
                continue;
            }
            let new_balance = (self.address_balance_store.get(&spk).unwrap() as i128 + delta).max(0) as u64;
            self.address_balance_store.set_batch(batch, &spk, new_balance).unwrap();
        }
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
    fn process_ai_txs_for_slash(&self, txs: &[Transaction], pov_daa_score: u64, coinbase_tx_id: TransactionId) {
        // Register confirmed AiResponse txs.
        for tx in txs.iter().skip(1) {
            if tx.is_ai_response() {
                let hash = blake2b_simd::blake2b(&tx.payload);
                let mut response_hash_bytes = [0u8; 32];
                response_hash_bytes.copy_from_slice(&hash.as_bytes()[..32]);
                let response_hash = Hash::from_bytes(response_hash_bytes);

                // Extract request_hash and claimed_commitment for Phase 3 C fraud verification.
                // Commitment = sha2-256 digest embedded in the IPFS multihash (bytes [2..34]).
                let (request_hash, claimed_commitment) =
                    if let Some(resp) = AiResponsePayload::deserialize(&tx.payload) {
                        let commitment: [u8; 32] = resp.response_ipfs_cid[2..34].try_into().unwrap();
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
/// Ratio-reward (Stage 2b) — net amount (`added − removed`) attributable to `spk` within a UTXO
/// diff. Used to translate the virtual-anchored balance index to a rewarding block's own view in
/// `ratio_bps_by_block`. `i128` carries the signed intermediate; the caller floors the corrected
/// balance at 0.
fn balance_delta_for_spk(diff: &UtxoDiff, spk: &ScriptPublicKey) -> i128 {
    let added: i128 = diff.add.values().filter(|e| &e.script_public_key == spk).map(|e| e.amount as i128).sum();
    let removed: i128 = diff.remove.values().filter(|e| &e.script_public_key == spk).map(|e| e.amount as i128).sum();
    added - removed
}

/// Ratio-reward (Stage 2b-2b) — pure index arithmetic of the windowed-production **bottom slide**.
/// When the selected-chain tip moves from `from_tip` to `to_tip` (window length `w`), the window's
/// low boundary slides by the same amount: chain blocks that leave the window must be subtracted
/// (`sign = -1`), or — on a reorg that shortens the chain — those that re-enter must be re-added
/// (`sign = +1`). Returns `Some((sign, lo, hi))` over the inclusive chain-index range `[lo, hi]`
/// (with `lo >= 1`, since index 0 is the genesis/pruning anchor and is never a production block), or
/// `None` when nothing crosses the boundary (no net tip move, or an early/short chain whose window
/// has not yet reached index 1). Split out from `fold_production_window_delta` to be unit-tested in
/// isolation — this is the off-by-one-prone core.
fn window_bottom_slide(from_tip: i128, to_tip: i128, w: i128) -> Option<(i128, i128, i128)> {
    let (from_low, to_low) = (from_tip - w, to_tip - w);
    let (lo, hi, sign) = if to_low > from_low {
        (from_low + 1, to_low, -1) // window grew at the top ⇒ blocks exit at the bottom
    } else if to_low < from_low {
        (to_low + 1, from_low, 1) // window shrank (reorg) ⇒ blocks re-enter at the bottom
    } else {
        return None;
    };
    let lo = lo.max(1);
    (lo <= hi).then_some((sign, lo, hi))
}

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

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use keryx_consensus_core::subnets;

    use super::*;

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

#[cfg(test)]
mod ratio_window_tests {
    use super::window_bottom_slide;

    #[test]
    fn advance_by_one_exits_oldest() {
        // tip 10 → 11, W=3: window {8,9,10} → {9,10,11}; index 8 exits (subtract).
        assert_eq!(window_bottom_slide(10, 11, 3), Some((-1, 8, 8)));
    }

    #[test]
    fn advance_by_three_exits_three() {
        // tip 10 → 13, W=3: indices 8,9,10 all exit.
        assert_eq!(window_bottom_slide(10, 13, 3), Some((-1, 8, 10)));
    }

    #[test]
    fn reorg_reenters_oldest() {
        // tip 11 → 10, W=3: index 8 re-enters the window (re-add).
        assert_eq!(window_bottom_slide(11, 10, 3), Some((1, 8, 8)));
    }

    #[test]
    fn no_tip_move_is_none() {
        assert_eq!(window_bottom_slide(10, 10, 3), None);
    }

    #[test]
    fn early_chain_skips_genesis_anchor() {
        // Window not yet full (tip ≤ W): nothing has aged past index 0, so no bottom touch.
        assert_eq!(window_bottom_slide(2, 3, 3), None);
        // First real exit happens once the boundary reaches index 1.
        assert_eq!(window_bottom_slide(3, 4, 3), Some((-1, 1, 1)));
    }

    #[test]
    fn advance_then_reorg_cancels_exactly() {
        // The forward exit and its reorg re-entry hit the SAME index with opposite signs ⇒ net zero,
        // which is what makes the production index reorg-exact.
        assert_eq!(window_bottom_slide(10, 11, 3), Some((-1, 8, 8)));
        assert_eq!(window_bottom_slide(11, 10, 3), Some((1, 8, 8)));
    }
}
