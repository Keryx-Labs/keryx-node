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
        ghostdag::{CompactGhostdagData, GhostdagData, GhostdagStoreReader},
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
use crate::model::stores::age_buckets::{AgeBuckets, AgeBucketsStoreReader};
use crate::model::stores::maturation_queue::{DbMaturationQueueStore, MaturationEntry};
use crate::model::stores::ai_slash::{AiResponseRecord, AiResponseStore, AiResponseStoreReader};
use crate::model::stores::pom_tier::PomTierStoreReader;
use crate::model::stores::selected_chain::SelectedChainStoreReader;
use crate::model::stores::windowed_production_prefix::WindowedProductionPrefixStoreReader;
use keryx_consensus_core::coin_age::eff_balance_from_buckets;
use keryx_consensus_core::config::params::{INFERENCE_REWARD_MINIMUMS_V2_H4, TIER_REWARD_BPS_DIVISOR, ratio_reward_bps, ratio_reward_bps_v2, tier_reward_bps};
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
use keryx_core::{debug, info, trace, warn};
use keryx_hashes::Hash;
use keryx_inference::{AiRequestPayload, AiResponsePayload, INFERENCE_REWARD_TOKEN_STEP, parse_ai_caps};
use keryx_muhash::MuHash;
use keryx_txscript::script_class::ScriptClass;
use keryx_utils::refs::Refs;

use rayon::prelude::*;
use smallvec::{SmallVec, smallvec};
use std::{
    iter::once,
    ops::Deref,
    sync::atomic::{AtomicBool, Ordering},
};

/// One-shot guard for the H4 banner below. Unlike the other hardfork banners — which match the
/// activation DAA score exactly — H4 fires on the FIRST block seen at or after the gate. A chain
/// block's `daa_score` advances by its mergeset's DAA-added count, so at 10 BPS it routinely
/// SKIPS the exact activation value and an equality-matched banner never prints. Logging only, so
/// a process-global guard is fine: it has no bearing on consensus.
static COIN_AGE_BANNER_LOGGED: AtomicBool = AtomicBool::new(false);

/// Max DAA a block may sit past the H4 gate and still trigger the activation banner. The gate uses
/// an at-or-after match (an exact-equality banner would be skipped at 10 BPS), which alone is true
/// forever after the fork — so a node that boots already synced far beyond H4 re-prints the banner
/// on every restart, its first validated chain block always being "at or after" the long-passed
/// gate. Bounding to gate + this window keeps the print to the actual crossing (live, or during IBD
/// where the first post-gate chain block sits a handful of DAA past the gate) and stays silent once
/// the chain has moved on. ~1 day at 10 BPS — orders of magnitude above any crossing lag.
const H4_BANNER_MAX_LAG: u64 = 864_000;

/// Pre-resolved production-window context of a single validated block (its selected parent
/// `m_sp`), shared by every rewarded blue of that block — see [`VirtualStateProcessor::production_window_ctx`].
pub(super) enum ProductionWindowCtx {
    /// `m_sp` is a committed selected-chain block: window = `(bottom, m_idx]` on the prefix index.
    OnChain { m_idx: u64, bottom: u64 },
    /// `m_sp` is on a side chain (mid-reorg / catch-up resolve batch): committed part `(lo, common]`
    /// on the prefix index + the side-chain production above `lo`, pre-aggregated per SPK.
    SideChain { common: u64, lo: u64, side_by_spk: std::collections::HashMap<ScriptPublicKey, u64> },
}

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

        // Coin-age era flag (holder-reward v3), derived from the POV block's own daa score —
        // same gating discipline as every other fork so IBD re-validation stays canonical.
        let coin_age_active = self.coin_age_activation.is_active(pov_daa_score);

        ctx.mergeset_diff.add_transaction(&validated_coinbase, pov_daa_score, coin_age_active).unwrap();
        ctx.multiset_hash.add_transaction(&validated_coinbase, pov_daa_score, self.coin_age_activation);
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
                ctx.mergeset_diff.add_transaction(validated_tx, pov_daa_score, coin_age_active).unwrap();
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
        // Coinbase ratio/tier verification. We ALWAYS compute the expected coinbase and log any
        // mismatch (so the producer's and every validator's computation can be compared across nodes);
        // we only REJECT when `enforce` holds. Enforcement requires the relaunch-frontier gate
        // (`ratio_verification_activation`, so non-revalidatable pre-relaunch history is trusted — its
        // `utxo_commitment`, checked above, pins the state) AND the node not being in a trust window
        // (archival / `KERYX_TRUST_COINBASE` / fast-sync catch-up). With the gate set to `never()`,
        // enforcement is OFF (observe-only) network-wide — the relaunch runs while we confirm the
        // prefix-sum makes all nodes agree, then enforcement is switched on by setting the gate DAA.
        let enforce = self.ratio_verification_activation.is_active(header.daa_score) && !self.trust_coinbase();
        self.verify_coinbase_transaction(
            &txs[0],
            header.daa_score,
            &ctx.ghostdag_data,
            &ctx.mergeset_rewards,
            &self.daa_excluded_store.get_mergeset_non_daa(header.hash).unwrap(),
            &[sp_diff, &ctx.mergeset_diff],
            enforce,
        )?;

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

        // H3 hardfork (PoM block-level): log once at the exact activation DAA score.
        if header.daa_score == self.pom_level_activation.daa_score() {
            info!("════════════════ KERYX HARDFORK H3 · DAA {} ════════════════", header.daa_score);
            info!("  Header        — pomFinalState committed in the block hash; header-only PoW checks restored");
            info!("  Block levels  — real levels back (bounded pruning proof, from-scratch IBD)");
            info!("  PoM salt      — walk + pow folds now salted; pre-H3 binaries rejected");
            info!("  Ratio v2      — production counted per paid blue over a DAA-sized 24h window");
            info!("  Coinbase cap  — output limit aligned with the OPoI builder (3*(K+1)+4)");
            info!("═══════════════════════════════════════════════════════════════");
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

        // H4 hardfork (coin-age holder-reward v3): fire on the FIRST block at or after the gate,
        // not on an exact DAA match — a chain block's daa_score advances by its mergeset's DAA-added
        // count and routinely skips the exact activation value at 10 BPS. Bounded to a window past
        // the gate (H4_BANNER_MAX_LAG) so a node booting already synced far beyond H4 no longer
        // re-prints it on every restart. `compare_exchange` keeps it to one print per process (the
        // first post-gate block within the window, whether reached live or during IBD).
        if self.coin_age_activation.is_active(header.daa_score)
            && header.daa_score < self.coin_age_activation.daa_score() + H4_BANNER_MAX_LAG
            && COIN_AGE_BANNER_LOGGED.compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed).is_ok()
        {
            // Header carries the GATE score (the fork's identity, always exact), not this block's —
            // a node restarting long after the fork also prints this once, and `header.daa_score`
            // would then read as if H4 had just fired. The observed block goes in the footer.
            info!("════════════════ KERYX HARDFORK H4 · DAA {} ════════════════", self.coin_age_activation.daa_score());
            info!("  Holder-reward — ratio numerator is now the coin-age effective balance, not the balance snapshot");
            info!("  Coin age      — FIFO carry-over anchors per output; age resets on transfer, survives consolidation");
            info!("  Maturity      — a coin ramps linearly to full weight over W = {} DAA", self.coin_age_maturity_w);
            info!("  UTXO muhash   — per-coin effective_daa now committed in the multiset");
            info!("  Rotation      — moving a pot to a fresh address no longer buys the top bracket");
            info!("  Bracket table — floor 50% (was 40%), ramp to 100% now spans 90 days (was 30)");
            info!("  (first block seen at/after the gate: daa {})", header.daa_score);
            info!("═══════════════════════════════════════════════════════════════");
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
            let minimums = if self.coin_age_activation.is_active(header.daa_score) {
                // H4 candle-free lineup floors (new bareme 0.5/1/1.5/2.5/4 KRX). New model_ids, so
                // the H2 table no longer matches any served model post-H4.
                INFERENCE_REWARD_MINIMUMS_V2_H4
            } else if self.inference_min_h2_activation.is_active(header.daa_score) {
                // H2 (5-tier) floors: adds Qwen3-1.7B + 70B-Q2, absent from the v2 table.
                self.inference_reward_minimums_v2_h2
            } else if self.opoi_v2_activation.is_active(header.daa_score) {
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
        // When false (observe-only): compute the expected coinbase and LOG any mismatch, but do NOT
        // reject the block. Lets the network run while we confirm the producer and validators compute
        // the identical coinbase (logs comparable across nodes) before enforcement is switched on.
        enforce: bool,
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
            // Diagnostic: pinpoint why the coinbase differs (tier vs ratio vs amounts). Logged at WARN
            // only when it causes a real rejection (`enforce`); in observe-only it would fire for every
            // trusted transition/history block, so it stays at DEBUG to avoid spamming normal logs.
            let detail = format!(
                "COINBASE MISMATCH enforce={} daa={} tier_bps={:?} ratio_bps={:?} actual_outs={:?} expected_outs={:?}",
                enforce,
                daa_score,
                tier_bps_by_block,
                ratio_bps_by_block,
                coinbase.outputs.iter().map(|o| o.value).collect::<Vec<_>>(),
                expected_coinbase.outputs.iter().map(|o| o.value).collect::<Vec<_>>(),
            );
            if enforce {
                warn!("{}", detail);
                return Err(BadCoinbaseTransaction);
            }
            // Observe-only: block is accepted; keep the comparison at debug level.
            debug!("{}", detail);
            Ok(())
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
        // Windowed production is read from the gold-standard prefix-sum index, evaluated at THIS
        // block's selected-parent window (Case A/B inside `windowed_production_for_block`). It is a pure
        // function of the chain — no path-dependent running sum, no slide arithmetic, no saturating
        // clamp — so every node computes the identical denominator. Floor at one block's base miner cut
        // so a newcomer with no recent production divides by one block. The balance numerator keeps its
        // own committed-index + `view_diffs` correction (that index is exact and not the divergence source).
        let prod_floor = self.coinbase_manager.base_miner_cut(pov_daa_score).max(1);
        // Coin-age era (v3): the numerator switches from the instantaneous balance to the
        // per-coin-capped effective balance — rotation-resistant (a fresh address's coins carry
        // age 0 and contribute nothing until they ripen over W).
        let coin_age_active = self.coin_age_activation.is_active(pov_daa_score);
        let w = self.ratio_reward_window;
        // Window context depends only on the block's selected parent — resolve it ONCE and share it
        // across every rewarded blue (it embeds the full side-chain aggregation in the Case B shape).
        let window_ctx = self.production_window_ctx(ghostdag_data.selected_parent, w);
        for blue in ghostdag_data.mergeset_blues.iter().filter(|h| !mergeset_non_daa.contains(h)) {
            // Payout SPK = the blue's own miner cut, already resolved into the reward data.
            if let Some(reward) = mergeset_rewards.get(blue) {
                let spk = &reward.script_public_key;
                let balance = if coin_age_active {
                    self.eff_balance_for_spk(spk, pov_daa_score, view_diffs)
                } else {
                    let base = self.address_balance_store.get(spk).unwrap() as i128;
                    let delta: i128 = view_diffs.iter().map(|d| balance_delta_for_spk(d, spk)).sum();
                    (base + delta).max(0) as u64
                };
                let production = self.windowed_production_with_ctx(spk, &window_ctx).max(prod_floor);
                // Recalibrated bracket table ships bundled with H4 (one hardfork, one gate).
                let bps = if self.coin_age_activation.is_active(pov_daa_score) {
                    ratio_reward_bps_v2(balance, production)
                } else {
                    ratio_reward_bps(balance, production)
                };
                map.insert(*blue, bps);
            }
        }

        // Targeted diagnostic (env KERYX_RATIO_DEBUG=1): dump the exact ratio inputs per rewarded blue
        // — selected-parent chain index, balance (numerator), windowed production from the prefix index
        // (O(log), cheap), the floor, and the resulting bracket. Run on the producer (build) and the
        // validator (verify) and diff the two lines to localize a cross-node disagreement: differing
        // `sp_idx` ⇒ chain/index mismatch; differing `balance` ⇒ numerator; differing `prod_prefix` with
        // same `sp_idx` ⇒ window/prefix mismatch. NOTE: deliberately NO O(W) direct-sum recompute here —
        // it runs on the build path per template and an 864k-block scan stalls template production (~40s),
        // starving the miner. The prefix value is the cross-node comparison we need.
        if std::env::var("KERYX_RATIO_DEBUG").is_ok() {
            let sc = self.selected_chain_store.read();
            if let Ok(sp_idx) = sc.get_by_hash(ghostdag_data.selected_parent) {
                for blue in ghostdag_data.mergeset_blues.iter().filter(|h| !mergeset_non_daa.contains(h)) {
                    if let Some(reward) = mergeset_rewards.get(blue) {
                        let spk = &reward.script_public_key;
                        // Resolve `balance` exactly as the numerator above does — at/after
                        // `coin_age_activation` that is the coin-age effective balance, NOT the
                        // instantaneous snapshot. Recomputing the snapshot here would print a value
                        // the bracket never saw, so a post-H4 cross-node diff on this line would
                        // compare the wrong quantity and hide the real disagreement.
                        let balance = if coin_age_active {
                            self.eff_balance_for_spk(spk, pov_daa_score, view_diffs)
                        } else {
                            let base = self.address_balance_store.get(spk).unwrap() as i128;
                            let delta: i128 = view_diffs.iter().map(|d| balance_delta_for_spk(d, spk)).sum();
                            (base + delta).max(0) as u64
                        };
                        let prefix = self.windowed_production_with_ctx(spk, &window_ctx);
                        // Also emit the producer's script-public-key (version + script hex) so an
                        // external tailer can key `prod_prefix`/`balance` by address without a
                        // separate blue->coinbase lookup. Appended last to keep existing parsers valid.
                        let spk_hex: String = spk.script().iter().map(|b| format!("{:02x}", b)).collect();
                        debug!(
                            "RATIO-DEBUG daa={} blue={} sp_idx={} balance={} prod_prefix={} floor={} ratio_bps={} spk_ver={} spk={}",
                            pov_daa_score, blue, sp_idx, balance, prefix, prod_floor,
                            map.get(blue).copied().unwrap_or(0), spk.version(), spk_hex
                        );
                    }
                }
            }
        }

        // Optional self-check (env KERYX_RATIO_SELFCHECK=1): verify BOTH the legacy maintained index
        // (store + correction) AND the new gold-standard prefix-sum index equal the DIRECT window
        // recompute for each rewarded blue. This is the equivalence oracle that proves the prefix index
        // before it becomes the consensus value. O(W) per call — enable only briefly (e.g. a relaunch).
        if std::env::var("KERYX_RATIO_SELFCHECK").is_ok() {
            let w = self.ratio_reward_window;
            // Prefix-index value per rewarded blue, computed FIRST so each `windowed_production_for_block`
            // takes and releases the selected-chain read lock before we hold it below (no nested re-lock).
            let mut prefix_vals: std::collections::HashMap<Hash, u64> = std::collections::HashMap::new();
            for blue in ghostdag_data.mergeset_blues.iter().filter(|h| !mergeset_non_daa.contains(h)) {
                if let Some(reward) = mergeset_rewards.get(blue) {
                    let v = self.windowed_production_for_block(&reward.script_public_key, ghostdag_data.selected_parent, w);
                    prefix_vals.insert(*blue, v.max(prod_floor));
                }
            }
            let sc = self.selected_chain_store.read();
            if let Ok(sp_idx) = sc.get_by_hash(ghostdag_data.selected_parent) {
                // Era-aware window bottom (exclusive), mirroring `production_window_ctx`:
                // legacy = last `w` chain blocks; H3 = daa-sized window found by binary search.
                let sp_header = self.headers_store.get_header(ghostdag_data.selected_parent).unwrap();
                let bottom = if self.pom_level_activation.is_active(sp_header.daa_score) {
                    let pruning_idx = sc.get_by_hash(sp_header.pruning_point).unwrap_or(0);
                    let daa_bound = sp_header.daa_score.saturating_sub(self.ratio_reward_window_daa);
                    self.chain_index_at_or_below_daa(&*sc, daa_bound, sp_idx, pruning_idx)
                } else {
                    sp_idx.saturating_sub(w)
                };
                let lo = (bottom + 1).max(1);
                let mut direct: std::collections::HashMap<ScriptPublicKey, u64> = std::collections::HashMap::new();
                for i in lo..=sp_idx {
                    if let Ok(h) = sc.get_by_index(i) {
                        for (spk, cut) in self.block_productions(h) {
                            *direct.entry(spk).or_default() += cut;
                        }
                    }
                }
                for blue in ghostdag_data.mergeset_blues.iter().filter(|h| !mergeset_non_daa.contains(h)) {
                    if let Some(reward) = mergeset_rewards.get(blue) {
                        let spk = &reward.script_public_key;
                        let truth = direct.get(spk).copied().unwrap_or(0).max(prod_floor);
                        let prefix = prefix_vals.get(blue).copied().unwrap_or(prod_floor);
                        if prefix != truth {
                            warn!(
                                "RATIO-SELFCHECK MISMATCH (prefix) daa={} blue={} prefix_prod={} direct_prod={} drift={}",
                                pov_daa_score, blue, prefix, truth, prefix as i128 - truth as i128
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

    /// Production contributions attributed at chain block `hash`'s index in the prefix-sum index,
    /// era-aware. The era is gated by the CHAIN BLOCK's own daa_score — a pure per-block property,
    /// so the index remains a pure, IBD-re-derivable function of the chain across the fork:
    /// - pre-`pom_level_activation` (legacy): one entry — the chain block's own producer. Only
    ///   selected-chain producers accumulated production, undercounting badly-peered miners whose
    ///   blocks are merged as blues (~1.7× connectivity bias).
    /// - at/after `pom_level_activation` (H3): one entry per PAID mergeset blue of the chain block
    ///   (non-DAA blues excluded — the exact set the coinbase pays and `ratio_bps_by_block`
    ///   iterates), each = (blue's own coinbase SPK, `base_miner_cut(blue.daa_score)`). Production
    ///   becomes the exact mirror of payment: every blue is merged by exactly one chain block, so
    ///   every paid block is counted exactly once, connectivity-bias-free.
    pub(super) fn block_productions(&self, hash: Hash) -> Vec<(ScriptPublicKey, u64)> {
        if self.pom_level_activation.is_active(self.headers_store.get_daa_score(hash).unwrap()) {
            let ghostdag_data = self.ghostdag_store.get_data(hash).unwrap();
            let non_daa = self.daa_excluded_store.get_mergeset_non_daa(hash).unwrap();
            ghostdag_data
                .mergeset_blues
                .iter()
                .filter(|b| !non_daa.contains(b))
                .filter_map(|b| self.block_production(*b))
                .collect()
        } else {
            self.block_production(hash).into_iter().collect()
        }
    }

    /// Memoized [`block_productions`]. A chain block's contribution list (its era, its mergeset,
    /// the blues' coinbase SPKs and base cuts) is immutable per hash — reorgs never change it — so
    /// entries are safe to keep indefinitely; the cache is only cleared to bound memory.
    /// This is what breaks the quadratic RocksDB-read blowup of side-chain (Case B) windowed
    /// production during catch-up: block k of a resolve batch re-reads the same k−1 mergesets
    /// block k−1 just read.
    pub(super) fn block_productions_cached(&self, hash: Hash) -> std::sync::Arc<Vec<(ScriptPublicKey, u64)>> {
        if let Some(v) = self.block_production_cache.read().get(&hash) {
            return v.clone();
        }
        let v = std::sync::Arc::new(self.block_productions(hash));
        let mut cache = self.block_production_cache.write();
        if cache.len() >= 200_000 {
            cache.clear();
        }
        cache.insert(hash, v.clone());
        v
    }

    /// Gold-standard prefix-sum maintenance — advances the production index along `chain_path`, kept in
    /// lockstep with the selected chain (called from `commit_virtual_state` in the SAME batch as the
    /// selected-chain `apply_changes`, BEFORE it runs, so `sc` still reflects the pre-change chain).
    /// Ungated/passive: maintained from genesis so it is exact for from-genesis nodes; only read once
    /// `ratio_reward_activation` fires. Translates the selected-chain `chain_path` into the
    /// `(common, removals, additions)` the prefix store extends with. EXACT and path-independent: the store
    /// seeds each addition from `cumulative_at(spk, common)` — a reverse seek that naturally ignores
    /// the about-to-be-removed entries (they sit at index > `common`) — and re-derives cumulatives, so
    /// there is no slide arithmetic and no saturating clamp that could silently drift.
    ///
    /// Index assignment mirrors the selected-chain store: `common = from_tip − |removed|`; a removed
    /// block `removed[j]` sat at index `from_tip − j` (removed is tip→split order); an added block
    /// `added[k]` lands at `common + 1 + k` (added is split→tip order). Producers with a zero base cut
    /// (`block_production == None`, tail-emission edge) contribute no entry — identical to the legacy
    /// fold skipping them, and correct since a zero cut never changes a cumulative.
    pub(super) fn advance_production_prefix(
        &self,
        batch: &mut rocksdb::WriteBatch,
        chain_path: &ChainPath,
        sc: &impl SelectedChainStoreReader,
    ) {
        let from_tip = sc.get_tip().unwrap().0;
        let common = from_tip - chain_path.removed.len() as u64;
        // Era-aware (H3): a chain block contributes one entry per paid mergeset blue, so several
        // (spk, index[, cut]) tuples can share an index. Deletion of duplicate keys is idempotent;
        // on the addition side `extend`'s per-SPK running accumulator chains same-key puts, so the
        // last write carries the summed cumulative — no pre-aggregation needed.
        let removals: Vec<(ScriptPublicKey, u64)> = chain_path
            .removed
            .iter()
            .enumerate()
            .flat_map(|(j, h)| {
                self.block_productions_cached(*h).iter().map(|(spk, _)| (spk.clone(), from_tip - j as u64)).collect::<Vec<_>>()
            })
            .collect();
        let additions: Vec<(ScriptPublicKey, u64, u64)> = chain_path
            .added
            .iter()
            .enumerate()
            .flat_map(|(k, h)| {
                self.block_productions_cached(*h)
                    .iter()
                    .map(|(spk, cut)| (spk.clone(), common + 1 + k as u64, *cut))
                    .collect::<Vec<_>>()
            })
            .collect();
        self.windowed_production_prefix_store.extend(batch, common, &removals, &additions).unwrap();
    }

    /// Windowed production for `spk` as seen by the block whose selected parent is `m_sp`, read from
    /// the gold-standard prefix-sum index, with the window FLOORED at `m_sp`'s committed pruning point
    /// (option C). The window is `(max(idx(m_sp) − W, idx(pruning_point)), idx(m_sp)]` — the last `W`
    /// chain-blocks, but never reaching below the pruning point.
    ///
    /// Why the floor: a pruned node only retains the selected chain back to the pruning point, and
    /// across the pre-relaunch (high-DAG-width) history that is FEWER than `W` chain-blocks — so it
    /// physically cannot reproduce an un-clamped `W`-window (it computes a truncated, larger ratio than
    /// an archival node). Clamping BOTH archival and pruned nodes to the same consensus pruning point
    /// makes them sum production over the identical block set `(pruning_point, m_sp]`, hence identical
    /// values. The pruning point is read from `m_sp`'s HEADER (a consensus value every validator
    /// shares), not the node's local pruning state (which lags during sync). Absolute chain indices may
    /// differ across nodes (archival from genesis vs pruned re-based), but the cumulative DIFFERENCE
    /// over the same block range is offset-independent, so the result agrees.
    ///
    /// **Case A** — `m_sp` on the committed chain: `cum(idx) − cum(floor)`.
    /// **Case B** — `m_sp` off-chain (mid-reorg): committed-prefix part `(floor, common]` + the
    /// side-chain `added` blocks above the floor, summed directly.
    ///
    /// The window resolution depends only on `m_sp`, so it is split out into
    /// [`production_window_ctx`], computed ONCE per validated block; the per-SPK query is
    /// [`windowed_production_with_ctx`]. Keeping them fused per SPK (the previous shape) walked
    /// the full committed-tip→`m_sp` chain path and re-read every side-chain coinbase for EVERY
    /// rewarded blue of EVERY block of a catch-up resolve batch — quadratic in batch length, and
    /// the measured cause of an IBD catch-up crawling at ~4 UTXO-validated blocks/s.
    pub(super) fn windowed_production_for_block(&self, spk: &ScriptPublicKey, m_sp: Hash, w: u64) -> u64 {
        let ctx = self.production_window_ctx(m_sp, w);
        self.windowed_production_with_ctx(spk, &ctx)
    }

    /// Resolves the production-window context of the block whose selected parent is `m_sp` —
    /// everything of `windowed_production_for_block` that does not depend on the queried SPK.
    /// Case B pre-aggregates the side-chain production into a per-SPK map (one pass over the
    /// chain path, mergesets served by `block_productions_cached`), so per-blue queries are O(1)
    /// map lookups + two prefix-store reads.
    ///
    /// Window sizing is era-gated on `m_sp`'s daa_score (a header value — identical for the
    /// producer and every validator):
    /// - pre-`pom_level_activation`: the last `w` SELECTED-CHAIN blocks (legacy; ~4.6 real days
    ///   at mainnet mergeset width, drifting with it);
    /// - at/after: the chain blocks whose daa_score lies in `(m_sp.daa − ratio_reward_window_daa,
    ///   m_sp.daa]` — a FIXED 24h regardless of DAG width. The bottom index is found by binary
    ///   search (daa is strictly increasing along the selected chain: every chain block adds at
    ///   least itself to the DAA count).
    ///
    /// Both eras keep the pruning-point clamp (option C, see `windowed_production_for_block`).
    pub(super) fn production_window_ctx(&self, m_sp: Hash, w: u64) -> ProductionWindowCtx {
        let sc = self.selected_chain_store.read();
        let m_sp_header = self.headers_store.get_header(m_sp).unwrap();
        // Window floor: chain index of m_sp's committed pruning point (consensus). 0 if not on the
        // selected chain (no clamp) — should not happen for a valid block's pruning point.
        let pruning_idx = sc.get_by_hash(m_sp_header.pruning_point).unwrap_or(0);
        let h3 = self.pom_level_activation.is_active(m_sp_header.daa_score);
        // H3 window bottom in daa units — entries strictly above this daa are inside the window.
        let daa_bound = m_sp_header.daa_score.saturating_sub(self.ratio_reward_window_daa);
        if let Ok(m_idx) = sc.get_by_hash(m_sp) {
            // Case A: m_sp is a committed chain block.
            let bottom = if h3 {
                self.chain_index_at_or_below_daa(&*sc, daa_bound, m_idx, pruning_idx)
            } else {
                m_idx.saturating_sub(w)
            }
            .max(pruning_idx);
            return ProductionWindowCtx::OnChain { m_idx, bottom };
        }
        // Case B: m_sp is on a side chain. Reconstruct its window = committed-prefix part + side delta.
        let (committed_tip_index, committed_tip) = sc.get_tip().unwrap();
        let chain_path = self.dag_traversal_manager.calculate_chain_path(committed_tip, m_sp, None);
        let common = committed_tip_index - chain_path.removed.len() as u64;
        let m = common + chain_path.added.len() as u64; // m_sp's index along its OWN selected chain
        // Window bottom over the COMMITTED part, floored at the pruning point. In the H3 era the
        // committed part of the window is bounded by daa (searched up to `common`); side-chain
        // added blocks are filtered by their own daa below instead of by index.
        let lo = if h3 { self.chain_index_at_or_below_daa(&*sc, daa_bound, common, pruning_idx) } else { m.saturating_sub(w) }
            .max(pruning_idx);
        // Side part: added[k] sits at index common+1+k; include those inside the window
        // (legacy: index > lo; H3: the block's own daa above the daa bound).
        let mut side_by_spk: std::collections::HashMap<ScriptPublicKey, u64> = std::collections::HashMap::new();
        for (k, h) in chain_path.added.iter().enumerate() {
            let in_window = if h3 {
                self.headers_store.get_daa_score(*h).unwrap() > daa_bound
            } else {
                common + 1 + k as u64 > lo
            };
            if in_window {
                for (s, cut) in self.block_productions_cached(*h).iter() {
                    *side_by_spk.entry(s.clone()).or_default() += cut;
                }
            }
        }
        ProductionWindowCtx::SideChain { common, lo, side_by_spk }
    }

    /// Largest selected-chain index in `[search floor, hi_idx]` whose block's daa_score is
    /// ≤ `bound_daa` — the exclusive window bottom for a daa-sized production window. Binary
    /// search, valid because daa_score is strictly increasing along the selected chain. The search
    /// floor is `max(hi_idx − ratio_reward_window_daa, floor_idx)`: the chain gains at most one
    /// index per daa point, so the bottom can never sit more than `ratio_reward_window_daa`
    /// indices below `hi_idx`; `floor_idx` (the pruning clamp) keeps every probe inside retained,
    /// consensus-shared history. If even the floor's daa exceeds the bound (window truncated by
    /// pruning), the floor itself is returned — the caller clamps to it anyway.
    fn chain_index_at_or_below_daa(
        &self,
        sc: &impl SelectedChainStoreReader,
        bound_daa: u64,
        hi_idx: u64,
        floor_idx: u64,
    ) -> u64 {
        let daa_at = |i: u64| self.headers_store.get_daa_score(sc.get_by_index(i).unwrap()).unwrap();
        let mut lo = hi_idx.saturating_sub(self.ratio_reward_window_daa).max(floor_idx);
        let mut hi = hi_idx;
        if lo >= hi || daa_at(lo) > bound_daa {
            return lo;
        }
        while lo < hi {
            let mid = lo + (hi - lo + 1) / 2;
            if daa_at(mid) <= bound_daa { lo = mid } else { hi = mid - 1 }
        }
        lo
    }

    /// Windowed production of `spk` under a pre-resolved [`ProductionWindowCtx`]. Byte-identical
    /// result to the previous fused computation: Case A/B formulas unchanged, only hoisted.
    pub(super) fn windowed_production_with_ctx(&self, spk: &ScriptPublicKey, ctx: &ProductionWindowCtx) -> u64 {
        match ctx {
            ProductionWindowCtx::OnChain { m_idx, bottom } => {
                let hi = self.windowed_production_prefix_store.cumulative_at(spk, *m_idx).unwrap();
                let lo_cum = self.windowed_production_prefix_store.cumulative_at(spk, *bottom).unwrap();
                hi.saturating_sub(lo_cum)
            }
            ProductionWindowCtx::SideChain { common, lo, side_by_spk } => {
                // Shared part: committed-chain indices (lo, common] (empty when the whole window is side-chain).
                let shared = if lo < common {
                    let hi = self.windowed_production_prefix_store.cumulative_at(spk, *common).unwrap();
                    let bottom = self.windowed_production_prefix_store.cumulative_at(spk, *lo).unwrap();
                    hi.saturating_sub(bottom)
                } else {
                    0
                };
                shared + side_by_spk.get(spk).copied().unwrap_or(0)
            }
        }
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

    /// Coin-age (holder-reward v3) — advances the bucket index by `diff`, in lockstep with the
    /// virtual UTXO set (same batch as `apply_balance_diff`). Each entry is classified at the new
    /// virtual score: MATURE (`effective_daa <= d − W`) contributes its face value to `b_mat`,
    /// IMMATURE contributes `(amount, amount·effective_daa)` to `(b_imm, a_imm)`. Deltas are
    /// folded per SPK first, then each touched address is read-modify-written once; all-zero
    /// aggregates delete their entry. Maintained ungated (passive aggregate, same discipline as
    /// the balance index); nothing reads it before `coin_age_activation`, and the startup rebuild
    /// re-derives it exactly from the UTXO set — which also re-classifies any coin that matured
    /// in place until the maturation-queue promotions land.
    pub(super) fn apply_age_diff(&self, batch: &mut rocksdb::WriteBatch, diff: &UtxoDiff, pov_daa_score: u64) {
        // (b_mat delta, b_imm delta, a_imm delta) per SPK. i128 accommodates sompi × DAA products.
        let mut deltas: std::collections::HashMap<ScriptPublicKey, (i128, i128, i128)> = std::collections::HashMap::new();
        let mature_bound = pov_daa_score.saturating_sub(self.coin_age_maturity_w);
        for (outpoint, entry) in diff.add.iter() {
            let d = deltas.entry(entry.script_public_key.clone()).or_default();
            if entry.effective_daa <= mature_bound {
                d.0 += entry.amount as i128;
            } else {
                d.1 += entry.amount as i128;
                d.2 += (entry.amount as i128) * (entry.effective_daa as i128);
                // Immature coin: enqueue at its maturity score so the sweep promotes it in time.
                let queued = MaturationEntry {
                    script_public_key: entry.script_public_key.clone(),
                    amount: entry.amount,
                    anchor: entry.effective_daa,
                };
                self.maturation_queue_store.insert_batch(batch, entry.effective_daa + self.coin_age_maturity_w, outpoint, queued).unwrap();
            }
        }
        for (outpoint, entry) in diff.remove.iter() {
            let d = deltas.entry(entry.script_public_key.clone()).or_default();
            if entry.effective_daa <= mature_bound {
                d.0 -= entry.amount as i128;
            } else {
                d.1 -= entry.amount as i128;
                d.2 -= (entry.amount as i128) * (entry.effective_daa as i128);
                // Spent while immature: drop its pending promotion.
                self.maturation_queue_store.delete_batch(batch, entry.effective_daa + self.coin_age_maturity_w, outpoint).unwrap();
            }
        }
        for (spk, (dm, div, dia)) in deltas {
            if (dm, div, dia) == (0, 0, 0) {
                continue;
            }
            let b = self.age_buckets_store.get(&spk).unwrap();
            let next = AgeBuckets {
                b_mat: (b.b_mat as i128 + dm).max(0) as u64,
                b_imm: (b.b_imm as i128 + div).max(0) as u64,
                a_imm: (b.a_imm as i128 + dia).max(0) as u128,
            };
            self.age_buckets_store.set_batch(batch, &spk, next).unwrap();
        }
    }

    /// Coin-age maturation sweep — the time-driven bucket transition. Promotes every queued coin
    /// whose maturity score (`anchor + W`) fell at/below the NEW virtual score: `b_imm/a_imm →
    /// b_mat`, queue entry deleted, watermark advanced. Runs BEFORE `apply_age_diff` in the same
    /// commit so a coin that is both due and spent is first promoted, then removed as mature —
    /// mirroring the remove-path classification (which sees it at/below `d − W`).
    ///
    /// When the virtual score moves BELOW the watermark (side-chain re-anchor — routine during
    /// IBD catch-up, where virtual commits alternate between the syncer chain and the local
    /// near-tip sink), the promotions in `(new, watermark]` are unwound in place by demoting
    /// their retained queue entries — the write-path mirror of the read-path demotion in
    /// `eff_balance_for_spk`, with the same spent-after-maturing guard (`diff` here plays the
    /// role of `view_diffs`: a spend at score ≥ due > new score cannot be in the new virtual's
    /// past, so the reorg diff re-adds such a coin and `apply_age_diff` re-classifies it).
    ///
    /// Returns `true` only when the drop exceeds `coin_age_retention` — the retained promotions
    /// needed to unwind were pruned (never in practice: retention = finality depth) — and the
    /// caller must run a full `rebuild_age_buckets_index` after the commit instead.
    pub(super) fn sweep_maturation_queue(&self, batch: &mut rocksdb::WriteBatch, new_daa_score: u64, diff: &UtxoDiff) -> bool {
        let watermark = self.maturation_queue_store.get_watermark().unwrap().unwrap_or(new_daa_score);
        if new_daa_score < watermark {
            if watermark - new_daa_score > self.coin_age_retention {
                return true;
            }
            for (raw, due) in self.maturation_queue_store.due_range(new_daa_score, watermark) {
                // Pure re-add (spent-after-maturing restore) — skip, `apply_age_diff` re-adds the
                // coin on the immature side. In add AND remove (same outpoint re-anchored with a
                // different `effective_daa`) — demote: the remove folds on the immature side and
                // must land on the demoted value.
                let outpoint = DbMaturationQueueStore::outpoint_of(&raw);
                if diff.add.contains_key(&outpoint) && !diff.remove.contains_key(&outpoint) {
                    continue;
                }
                let b = self.age_buckets_store.get(&due.script_public_key).unwrap();
                let next = AgeBuckets {
                    b_mat: b.b_mat.saturating_sub(due.amount),
                    b_imm: b.b_imm.saturating_add(due.amount),
                    a_imm: b.a_imm.saturating_add(due.amount as u128 * due.anchor as u128),
                };
                self.age_buckets_store.set_batch(batch, &due.script_public_key, next).unwrap();
                // The entry stays queued: the next forward sweep past its due re-promotes it.
            }
            self.maturation_queue_store.set_watermark_batch(batch, new_daa_score).unwrap();
            return false;
        }
        for (_, due) in self.maturation_queue_store.due_range(watermark, new_daa_score) {
            // Consecutive read-modify-writes on the same SPK stay coherent through the store's
            // write-through cache (set_batch inserts the cache before the batch lands).
            let b = self.age_buckets_store.get(&due.script_public_key).unwrap();
            let next = AgeBuckets {
                b_mat: b.b_mat.saturating_add(due.amount),
                b_imm: b.b_imm.saturating_sub(due.amount),
                a_imm: b.a_imm.saturating_sub(due.amount as u128 * due.anchor as u128),
            };
            self.age_buckets_store.set_batch(batch, &due.script_public_key, next).unwrap();
            // NOTE: the promoted entry is NOT deleted — it is retained for `coin_age_retention`
            // scores so the read path can DEMOTE when a POV falls below the watermark (side
            // chains, see `eff_balance_for_spk`). Retention pruning below reclaims it.
        }
        self.maturation_queue_store.prune_below(batch, new_daa_score.saturating_sub(self.coin_age_retention)).unwrap();
        self.maturation_queue_store.set_watermark_batch(batch, new_daa_score).unwrap();
        false
    }

    /// Coin-age numerator (holder-reward v3): the per-coin-capped effective balance of `spk` at
    /// the POV block's view — the consensus replacement for the raw balance at/after
    /// `coin_age_activation`. Cross-node determinism requires reconciling two node-local anchors
    /// onto the POV's:
    ///
    /// 1. **Split reconciliation** — the committed buckets are split at the node-local WATERMARK,
    ///    not at the POV score. Retained queue entries bridge the gap: maturities in
    ///    `(watermark, pov]` are promotions the POV already sees (add to `b_mat`), maturities in
    ///    `(pov, watermark]` are promotions the POV does NOT yet see (demote back). A demotion
    ///    entry whose outpoint is re-added by `view_diffs` is skipped — the coin is absent from
    ///    the committed store (spent after maturing), and the content correction below re-adds it
    ///    on the right side of the POV split.
    /// 2. **Content correction** — `view_diffs` (committed virtual → POV view) entries are folded
    ///    at the POV split (`effective_daa <= pov − W`), mirroring `balance_delta_for_spk`.
    ///
    /// Both adjustments are bounded: the split scan by `|pov − watermark|` (mergeset depth in
    /// practice, capped by the retention horizon), the content fold by the diff size.
    pub(super) fn eff_balance_for_spk(&self, spk: &ScriptPublicKey, pov_daa_score: u64, view_diffs: &[&UtxoDiff]) -> u64 {
        let b = self.age_buckets_store.get(spk).unwrap();
        let (mut mat, mut imm_v, mut imm_a) = (b.b_mat as i128, b.b_imm as i128, b.a_imm as i128);
        let watermark = self.maturation_queue_store.get_watermark().unwrap().unwrap_or(pov_daa_score);
        if pov_daa_score >= watermark {
            // Promotions the POV sees but the committed split does not yet.
            for (_, e) in self.maturation_queue_store.due_range(watermark, pov_daa_score) {
                if &e.script_public_key == spk {
                    mat += e.amount as i128;
                    imm_v -= e.amount as i128;
                    imm_a -= (e.amount as i128) * (e.anchor as i128);
                }
            }
        } else {
            // Promotions the committed split holds but the POV does not see yet: demote, unless
            // the coin is absent from the committed store (re-added by the view diffs).
            for (raw, e) in self.maturation_queue_store.due_range(pov_daa_score, watermark) {
                if &e.script_public_key == spk {
                    let outpoint = DbMaturationQueueStore::outpoint_of(&raw);
                    if view_diffs.iter().any(|d| d.add.contains_key(&outpoint)) {
                        continue;
                    }
                    mat -= e.amount as i128;
                    imm_v += e.amount as i128;
                    imm_a += (e.amount as i128) * (e.anchor as i128);
                }
            }
        }
        // Content correction at the POV split.
        let mature_bound = pov_daa_score.saturating_sub(self.coin_age_maturity_w);
        for diff in view_diffs {
            let (dm, div, dia) = age_delta_for_spk(diff, spk, mature_bound);
            mat += dm;
            imm_v += div;
            imm_a += dia;
        }
        eff_balance_from_buckets(mat.max(0) as u64, imm_v.max(0) as u64, imm_a.max(0) as u128, pov_daa_score, self.coin_age_maturity_w)
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
                    let mh = MuHash::from_transaction(&vtx, pov_daa_score, self.coin_age_activation);
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
/// Coin-age view-diff correction for one SPK: (b_mat, b_imm, a_imm) deltas with every entry
/// classified at the POV split (`effective_daa <= mature_bound`). The bucket-space mirror of
/// `balance_delta_for_spk`.
fn age_delta_for_spk(diff: &UtxoDiff, spk: &ScriptPublicKey, mature_bound: u64) -> (i128, i128, i128) {
    let (mut dm, mut div, mut dia) = (0i128, 0i128, 0i128);
    let mut fold = |entry: &keryx_consensus_core::tx::UtxoEntry, sign: i128| {
        if entry.effective_daa <= mature_bound {
            dm += sign * entry.amount as i128;
        } else {
            div += sign * entry.amount as i128;
            dia += sign * (entry.amount as i128) * (entry.effective_daa as i128);
        }
    };
    for entry in diff.add.values().filter(|e| &e.script_public_key == spk) {
        fold(entry, 1);
    }
    for entry in diff.remove.values().filter(|e| &e.script_public_key == spk) {
        fold(entry, -1);
    }
    (dm, div, dia)
}

fn balance_delta_for_spk(diff: &UtxoDiff, spk: &ScriptPublicKey) -> i128 {
    let added: i128 = diff.add.values().filter(|e| &e.script_public_key == spk).map(|e| e.amount as i128).sum();
    let removed: i128 = diff.remove.values().filter(|e| &e.script_public_key == spk).map(|e| e.amount as i128).sum();
    added - removed
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

