use super::VirtualStateProcessor;
use crate::{
    errors::{
        BlockProcessResult,
        RuleError::{
            AiRequestEscrowBelowInferenceReward, AiRequestFeeBelowInferenceReward,
            AiRequestInferenceRewardBelowMinimum, AiRequestInvalidEscrowScript,
            AiRequestMissingEscrowOutput, AiRequestPriorityFeeBelowMinimum,
            AiRequestMaxTokensExceeded, AiResponseModelCapMissing, AiResponseV2BeforeActivation, BadAcceptedIDMerkleRoot,
            BadCoinbaseTransaction, BadServiceStateCommitment, BadUTXOCommitment, InvalidTransactionsInUtxoContext,
            WrongHeaderPruningPoint,
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
use crate::model::stores::pruning::PruningStoreReader;
use crate::model::stores::selected_chain::SelectedChainStoreReader;
use crate::model::stores::windowed_production_prefix::WindowedProductionPrefixStoreReader;
use keryx_consensus_core::coin_age::eff_balance_from_buckets;
use keryx_consensus_core::config::params::{INFERENCE_REWARD_MINIMUMS_V2_H4, INFERENCE_REWARD_MINIMUMS_V2_H6, TIER_REWARD_BPS_DIVISOR, ratio_reward_bps, ratio_reward_bps_v2, tier_reward_bps};
use keryx_database::prelude::StoreResultExt;
use keryx_consensus_core::{
    BlockHashMap, BlockHashSet, ChainPath, HashMapCustomHasher,
    acceptance_data::{AcceptedTxEntry, MergesetBlockAcceptanceData},
    api::args::TransactionValidationArgs,
    coinbase::*,
    hashing,
    header::Header,
    muhash::MuHashExtensions,
    tx::{
        MutableTransaction, PopulatedTransaction, ScriptPublicKey, Transaction, TransactionId, TransactionOutpoint,
        ValidatedTransaction, VerifiableTransaction,
    },
    utxo::{
        utxo_diff::UtxoDiff,
        utxo_view::{UtxoView, UtxoViewComposition},
    },
};
use keryx_core::{debug, info, trace, warn};
use keryx_hashes::Hash;
use keryx_consensus_core::collateral::AI_REQUEST_MAX_TOKENS_CAP;
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
static H5_3_BANNER_LOGGED: AtomicBool = AtomicBool::new(false);
static H5_4_BANNER_LOGGED: AtomicBool = AtomicBool::new(false);
static H6_BANNER_LOGGED: AtomicBool = AtomicBool::new(false);
static H7_BANNER_LOGGED: AtomicBool = AtomicBool::new(false);
static H8_BANNER_LOGGED: AtomicBool = AtomicBool::new(false);

/// Whether a hardfork banner should print for the block crossing `activation`. Latching alone is
/// not enough: IBD re-validates the historical crossing block, and a network whose gate is active
/// from genesis (`daa_score() == 0`, e.g. H4 on the testnet) keeps every young chain inside the
/// lag window — both re-printed the banner on every sync. A banner announces a LIVE crossing:
/// the gate must be a real fork (score > 0), the block must sit inside the lag window past it,
/// and its timestamp must be recent wall-clock (a replayed historical crossing is old news).
fn banner_should_fire(activation: keryx_consensus_core::config::params::ForkActivation, header: &Header) -> bool {
    /// Max wall-clock age (ms) of a crossing block for its banner to count as live.
    const BANNER_LIVE_WINDOW_MS: u64 = 3_600_000;
    activation.is_active(header.daa_score)
        && activation.daa_score() > 0
        && header.daa_score < activation.daa_score() + BANNER_MAX_LAG
        && keryx_core::time::unix_now().saturating_sub(header.timestamp) < BANNER_LIVE_WINDOW_MS
}

/// Max DAA a block may sit past the H4 gate and still trigger the activation banner. The gate uses
/// an at-or-after match (an exact-equality banner would be skipped at 10 BPS), which alone is true
/// forever after the fork — so a node that boots already synced far beyond a gate re-prints its banner
/// on every restart, its first validated chain block always being "at or after" the long-passed
/// gate. Bounding to gate + this window keeps the print to the actual crossing (live, or during IBD
/// where the first post-gate chain block sits a handful of DAA past the gate) and stays silent once
/// the chain has moved on. ~1 day at 10 BPS — orders of magnitude above any crossing lag.
const BANNER_MAX_LAG: u64 = 864_000;

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
                self.coinbase_manager.parse_escrow_from_extra_data(coinbase_data.miner_data.extra_data, pov_daa_score);
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
        // Coinbase ratio/tier verification. Enforcement requires the relaunch-frontier gate
        // (`ratio_verification_activation`, so non-revalidatable pre-relaunch history is trusted — its
        // `utxo_commitment`, checked above, pins the state) AND the node not being in a trust window
        // (archival / `KERYX_TRUST_COINBASE` / fast-sync catch-up). With the gate set to `never()`,
        // enforcement is OFF (observe-only) network-wide — the relaunch runs while we confirm the
        // prefix-sum makes all nodes agree, then enforcement is switched on by setting the gate DAA.
        //
        // When not enforcing, the expected coinbase is only computed under `KERYX_RATIO_DEBUG`
        // (cross-node comparison logs). The ratio balances fold `sp_diff`, which grows with the
        // distance from the committed virtual — computing it per block turns a long re-validation
        // walk quadratic, so a trusted transition must not pay for a comparison it discards.
        let enforce = self.ratio_verification_activation.is_active(header.daa_score) && !self.trust_coinbase();
        if enforce || (std::env::var("KERYX_RATIO_DEBUG").is_ok() && !self.trust_coinbase()) {
            self.verify_coinbase_transaction(
                &txs[0],
                header.daa_score,
                &ctx.ghostdag_data,
                &ctx.mergeset_rewards,
                &self.daa_excluded_store.get_mergeset_non_daa(header.hash).unwrap(),
                &[sp_diff, &ctx.mergeset_diff],
                enforce,
            )?;
        }

        // Sealed service-state commitment: the header must commit the canonical service state
        // at its own pruning point. This runs in chain order (the local flush frontier is at
        // least finality-deep past the pruning point), and is skipped in the same trust windows
        // as the coinbase check — a node that cannot yet reproduce the fold trusts the
        // utxo-commitment-pinned chain instead.
        if keryx_consensus_core::pom::service_commit_active(header.daa_score) && !self.trust_coinbase() {
            let pp_daa = self.headers_store.get_daa_score(header.pruning_point).unwrap();
            let expected = self.service_commit_index.commitment_at(pp_daa);
            if header.service_state_hash != expected {
                return Err(BadServiceStateCommitment(header.hash, header.service_state_hash, expected));
            }
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
        // the gate (BANNER_MAX_LAG) so a node booting already synced far beyond H4 no longer
        // re-prints it on every restart. `compare_exchange` keeps it to one print per process (the
        // first post-gate block within the window, whether reached live or during IBD).
        if banner_should_fire(self.coin_age_activation, header)
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

        // H5.3 relaunch banner. Same latching shape as H4 — fires on the first block at or AFTER
        // the gate, never on strict equality: the DAA score is a cumulative count and routinely
        // skips the exact activation value at 10 BPS, so an equality test would leave the banner
        // silent and its absence would read as "the fork did not activate".
        if banner_should_fire(self.difficulty_reset_activation_h5_3, header)
            && H5_3_BANNER_LOGGED.compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed).is_ok()
        {
            info!("════════════════ KERYX HARDFORK H5.3 · DAA {} ════════════════", self.difficulty_reset_activation_h5_3.daa_score());
            info!("  Relaunch      — chain restarted at the last score preceding the coin-age divergence incident");
            info!("  Difficulty    — reset window open: blocks build at genesis bits until the DAA re-converges");
            info!("  Separation    — the abandoned branch carries the inherited bits and is rejected from here on");
            info!("  Miners        — unchanged: no walk-seed rotation, existing rigs keep mining");
            info!("  (first block seen at/after the gate: daa {})", header.daa_score);
            info!("═══════════════════════════════════════════════════════════════");
        }

        // H5.4 relaunch banner. Same latching shape as H4/H5.3 — fires on the first block at or
        // AFTER the gate, never on strict equality (the DAA score routinely skips the exact
        // activation value at 10 BPS).
        if banner_should_fire(self.difficulty_reset_activation_h5_4, header)
            && H5_4_BANNER_LOGGED.compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed).is_ok()
        {
            info!("════════════════ KERYX HARDFORK H5.4 · DAA {} ════════════════", self.difficulty_reset_activation_h5_4.daa_score());
            info!("  Relaunch      — chain restarted from the pre-incident base after the PoM proof-transport wedge");
            info!("  Difficulty    — reset window open: blocks build at genesis bits until the DAA re-converges");
            info!("  Separation    — un-upgraded nodes expect the inherited (decayed) bits and are cut off from here on");
            info!("  Miners        — unchanged: no walk-seed rotation, existing rigs keep mining");
            info!("  (first block seen at/after the gate: daa {})", header.daa_score);
            info!("═══════════════════════════════════════════════════════════════");
        }

        // H6 banner. Same latching shape as the others — fires once, on the first block at or
        // after the gate, only for a live crossing (see `banner_should_fire`).
        if banner_should_fire(self.pom_v3_activation, header)
            && H6_BANNER_LOGGED.compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed).is_ok()
        {
            info!("════════════════ KERYX HARDFORK H6 · DAA {} ════════════════", self.pom_v3_activation.daa_score());
            info!("  PoM v3        — matrix-walk possession proof; new model lineup (Qwen3.5-9B tier 0)");
            info!("  Escrow        — MANDATORY: blocks without `/escrow:` + a valid `/esig:` delegation cert are invalid");
            info!("  Identity      — strikes, suspensions and standing follow the payout address, not the hot escrow key");
            info!("  Sealed state  — burns/strikes/sightings committed in headers, downloaded and verified at IBD");
            info!("  Standing      — fresh identities mine at the floor tier rate for the probation window");
            info!("  Service-bond  — silent cohort members escalate: burn → slash-all → suspension; serving resets");
            info!("  Escrow lock   — CSV extended to ~22h; ~10h of claims stay burnable");
            info!("  Difficulty    — reset window open at the gate");
            info!("  (first block seen at/after the gate: daa {})", header.daa_score);
            info!("═══════════════════════════════════════════════════════════════");
        }

        // H7 banner. Same latching shape as the others — fires once, on the first block at or
        // after the gate, only for a live crossing (see `banner_should_fire`).
        if banner_should_fire(self.service_bond_v2_activation, header)
            && H7_BANNER_LOGGED.compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed).is_ok()
        {
            info!("════════════════ KERYX HARDFORK H7 · DAA {} ════════════════", self.service_bond_v2_activation.daa_score());
            info!("  Service bond  — v2: the audit stops striking miners for losing a race");
            info!("  Window        — every cohort member gets a {} DAA base (~5 min) to see and serve a request", keryx_consensus_core::collateral::SERVICE_WINDOW_BASE_DAA_V2);
            info!("  Cohort        — eligibility tightens to {} DAA past the last proven tier block", keryx_consensus_core::collateral::SERVICE_ELIGIBILITY_WINDOW_DAA_V2);
            info!("  First miss    — uniform {}-claim burn; a young identity no longer loses its whole vault", keryx_consensus_core::collateral::STRIKE_1_BURN_CLAIMS);
            info!("  Standing      — probation-only: a strike keeps its burn but no longer demotes the reward rate");
            info!("  Rate-limit    — a served response no longer disarms the one-strike-per-interval limit");
            info!("  Miners        — unchanged: no model or walk changes, existing rigs keep mining");
            info!("  (first block seen at/after the gate: daa {})", header.daa_score);
            info!("═══════════════════════════════════════════════════════════════");
        }

        if banner_should_fire(self.reward_routing_activation, header)
            && H8_BANNER_LOGGED.compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed).is_ok()
        {
            info!("════════════════ KERYX HARDFORK H8 · DAA {} ════════════════", self.reward_routing_activation.daa_score());
            info!("  Inference     — the reward goes to the FIRST miner whose response is accepted, not a client-designated key");
            info!("  Requests      — lock their reward in a keyless vault output; no answer within the horizon = burned");
            info!("  Coinbase      — mints finalized rewards to the winner's payout address (up to {} per block)", keryx_consensus_core::collateral::MAX_REWARD_MINTS_PER_BLOCK);
            info!("  Miners        — unchanged: no model or walk changes, existing rigs keep mining");
            info!("  (first block seen at/after the gate: daa {})", header.daa_score);
            info!("═══════════════════════════════════════════════════════════════");
        }

        // Signed (v2) AiResponse payloads only become valid at the service-bond gate; before it
        // this keeps the fixed 78-byte rule every deployed node enforces. The gate also brings
        // the max_tokens cap on AiRequests.
        if !self.pom_v3_activation.is_active(header.daa_score) {
            for tx in txs.iter().skip(1) {
                if tx.is_ai_response() && tx.payload.len() != keryx_inference::AI_RESPONSE_PAYLOAD_LEN {
                    return Err(AiResponseV2BeforeActivation(tx.id()));
                }
            }
        } else {
            for tx in txs.iter().skip(1) {
                if tx.is_ai_request() {
                    if let Some(req) = AiRequestPayload::deserialize(&tx.payload) {
                        if req.max_tokens > AI_REQUEST_MAX_TOKENS_CAP {
                            return Err(AiRequestMaxTokensExceeded(tx.id(), req.max_tokens, AI_REQUEST_MAX_TOKENS_CAP));
                        }
                    }
                }
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

            // Fast-fail: every AiRequest rule the transaction alone decides (reward and fee
            // floors, escrow output structure) is checked HERE rather than after the parallel
            // UTXO validation below — only `calculated_fee >= priority_fee` genuinely needs the
            // UTXO result. A single malformed or underpaid AiRequest poisons every block that
            // includes it, and each one otherwise costs a full UTXO pass to reach a verdict the
            // transaction already determines. Scheduling only: the rules, and the resulting
            // disqualification, are unchanged, so patched and unpatched nodes agree and no gate
            // is needed. (The complementary fix is upstream — keeping such a tx out of the
            // mempool so honest miners never include it and lose their block.)
            check_ai_request_payload_rules_all(&txs, self.ai_reward_minimums(header.daa_score), self.reward_routing_activation.is_active(header.daa_score))?;
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
            check_ai_request_inference_rewards(&txs, &validated_transactions, self.ai_reward_minimums(header.daa_score), self.reward_routing_activation.is_active(header.daa_score))?;
        }

        Ok(())
    }

    /// The `AiRequest` reward-minimum table in force at `daa_score`. DAA-gated so IBD re-validates
    /// historical blocks against the lineup of their own era: H4 swaps to the candle-free floors
    /// (new model_ids, so the H2 table matches nothing post-H4), H2 adds Qwen3-1.7B and 70B-Q2, and
    /// OPoI v2 introduced the uncensored lineup. Resolved in one place so the pre-UTXO fast path,
    /// the full block check and mempool admission cannot read different tables for the same score.
    pub(super) fn ai_reward_minimums(&self, daa_score: u64) -> &[([u8; 32], u64)] {
        if self.pom_v3_activation.is_active(daa_score) {
            INFERENCE_REWARD_MINIMUMS_V2_H6
        } else if self.coin_age_activation.is_active(daa_score) {
            INFERENCE_REWARD_MINIMUMS_V2_H4
        } else if self.inference_min_h2_activation.is_active(daa_score) {
            self.inference_reward_minimums_v2_h2
        } else if self.opoi_v2_activation.is_active(daa_score) {
            self.inference_reward_minimums_v2
        } else {
            self.inference_reward_minimums
        }
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
        let suspended_blues = self.suspended_blues(ghostdag_data, mergeset_non_daa, daa_score);
        let reward_mints = self.service_reward_mints_for(ghostdag_data.selected_parent);
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
                &suspended_blues,
                &reward_mints,
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
    /// Blue block hashes whose producer is under a finality-deep service-bond suspension as of this
    /// block — their miner cut is burned by `expected_coinbase_transaction`. Derived from the
    /// reorg-immune suspended set (populated in-order during virtual resolution, finality-deep), so
    /// every H6 node computes the identical set at this block's view. Empty pre-H6.
    pub(super) fn suspended_blues(
        &self,
        ghostdag_data: &GhostdagData,
        mergeset_non_daa: &BlockHashSet,
        pov_daa_score: u64,
    ) -> BlockHashSet {
        let mut set = BlockHashSet::new();
        if !self.pom_v3_activation.is_active(pov_daa_score) || self.service_suspended.read().is_empty() {
            return set;
        }
        for blue in ghostdag_data.mergeset_blues.iter().filter(|h| !mergeset_non_daa.contains(h)) {
            let txs = self.block_transactions_store.get(*blue).unwrap();
            let coinbase = self.coinbase_manager.deserialize_coinbase_payload(&txs[0].payload).unwrap();
            // Identity is the payout SPK key: rotating the hot escrow key does not shed a
            // suspension.
            let identity = keryx_consensus_core::collateral::miner_key(&coinbase.miner_data.script_public_key);
            if self.is_producer_suspended(&identity, pov_daa_score) {
                set.insert(*blue);
            }
        }
        set
    }

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
        // Reward schedule gated per block by daa_score (5-tier H6 once pom_v3 is live, else 5-tier H2,
        // else legacy 4-tier), keyed on this block's own daa_score to match `pom_tiers` under IBD.
        let schedule = tier_reward_bps(
            self.very_light_activation.is_active(pov_daa_score),
            self.pom_v3_activation.is_active(pov_daa_score),
        );
        // H6: the tier bonus is gated on standing — an identity in probation earns the floor
        // rate whatever tier it proves, so rotating identities forfeits the bonus for the whole
        // probation. Before service_bond_v2 a strike as of the lagged anchor also demotes.
        let standing_gate = self.pom_v3_activation.is_active(pov_daa_score);
        for blue in ghostdag_data.mergeset_blues.iter().filter(|h| !mergeset_non_daa.contains(h)) {
            if let Some(tier) = self.pom_tier_store.get(*blue).optional().unwrap() {
                let mut bps = schedule.get(tier as usize).copied().unwrap_or(TIER_REWARD_BPS_DIVISOR);
                if standing_gate {
                    let txs = self.block_transactions_store.get(*blue).unwrap();
                    let coinbase = self.coinbase_manager.deserialize_coinbase_payload(&txs[0].payload).unwrap();
                    let identity = keryx_consensus_core::collateral::miner_key(&coinbase.miner_data.script_public_key);
                    if !self.service_standing_at(&identity, pov_daa_score) {
                        bps = schedule[0];
                    }
                }
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
            let own_pp = self.pruning_point_store.read().pruning_point().unwrap();
            let sc = self.selected_chain_store.read();
            if let Ok(sp_idx) = sc.get_by_hash(ghostdag_data.selected_parent) {
                // Era-aware window bottom (exclusive), mirroring `production_window_ctx`:
                // legacy = last `w` chain blocks; H3 = daa-sized window found by binary search.
                let sp_header = self.headers_store.get_header(ghostdag_data.selected_parent).unwrap();
                let bottom = if self.pom_level_activation.is_active(sp_header.daa_score) {
                    let daa_bound = sp_header.daa_score.saturating_sub(self.ratio_reward_window_daa);
                    let pruning_idx = self.reward_window_floor(&*sc, sp_header.pruning_point, own_pp, daa_bound);
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
        // Read before the chain lock so the two locks are never held in inverse order.
        let own_pp = self.pruning_point_store.read().pruning_point().unwrap();
        let sc = self.selected_chain_store.read();
        let m_sp_header = self.headers_store.get_header(m_sp).unwrap();
        let h3 = self.pom_level_activation.is_active(m_sp_header.daa_score);
        // H3 window bottom in daa units — entries strictly above this daa are inside the window.
        let daa_bound = m_sp_header.daa_score.saturating_sub(self.ratio_reward_window_daa);
        let pruning_idx = self.reward_window_floor(&*sc, m_sp_header.pruning_point, own_pp, daa_bound);
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
    /// Chain-index floor for the daa-window searches (reward and service-bond): the header's
    /// committed pruning point when it is still on the retained selected chain, else the local
    /// retention boundary. A node
    /// prunes ahead of the blocks it re-validates during a restart catch-up, so their header
    /// pruning points can fall below retention; the substitute floor is consensus-neutral
    /// because the window bottom always sits above any retained floor (window < pruning depth).
    /// If even the boundary's daa exceeds `daa_bound`, the window bottom itself was pruned and
    /// no local computation can match the network — fail loud instead of diverging.
    pub(super) fn reward_window_floor(
        &self,
        sc: &impl SelectedChainStoreReader,
        header_pp: Hash,
        own_pp: Hash,
        daa_bound: u64,
    ) -> u64 {
        self.window_floor_in_retention(sc, header_pp, own_pp, daa_bound).unwrap_or_else(|| {
            panic!("the validation window reaches below the pruned horizon; local history cannot revalidate it — resync from a fresh datadir")
        })
    }

    /// Fallible form of [`Self::reward_window_floor`]: `None` when the window bottom itself sits
    /// below retained history, so no local floor can reproduce the network's search.
    pub(super) fn window_floor_in_retention(
        &self,
        sc: &impl SelectedChainStoreReader,
        header_pp: Hash,
        own_pp: Hash,
        daa_bound: u64,
    ) -> Option<u64> {
        match sc.get_by_hash(header_pp) {
            Ok(idx) => Some(idx),
            Err(_) => (self.headers_store.get_daa_score(own_pp).unwrap() <= daa_bound).then(|| sc.get_by_hash(own_pp).unwrap()),
        }
    }

    pub(super) fn chain_index_at_or_below_daa(
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
        deltas.retain(|_, d| *d != 0);
        if deltas.is_empty() {
            return;
        }
        let spks: Vec<ScriptPublicKey> = deltas.keys().cloned().collect();
        let (balances, hits, misses) = self.address_balance_store.get_many(&spks).unwrap();
        self.counters.address_balance_cache_hits.fetch_add(hits, std::sync::atomic::Ordering::Relaxed);
        self.counters.address_balance_cache_misses.fetch_add(misses, std::sync::atomic::Ordering::Relaxed);
        for (spk, balance) in spks.iter().zip(balances) {
            let delta = deltas[spk];
            let new_balance = (balance as i128 + delta).max(0) as u64;
            self.address_balance_store.set_batch(batch, spk, new_balance).unwrap();
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
        // ORDER MATTERS for the maturation queue: removes MUST be processed before adds. A tx
        // re-accepted by a different chain block during a shallow reorg puts the SAME outpoint in
        // diff.remove AND diff.add with the SAME inherited `effective_daa` (only `block_daa_score`
        // differs, so the pair survives diff algebra) — same queue key on both sides. With adds
        // first, the remove's delete lands after the add's insert in the batch and silently kills
        // the re-added coin's maturation entry: the coin then sits in `b_imm` forever (its
        // promotion never fires), which is exactly the network-wide coin-age drift signature.
        // Removes-first makes the add's insert land last, so the surviving coin keeps its entry.
        // The bucket deltas are order-independent (accumulated in the map, applied once below).
        for (outpoint, entry) in diff.remove.iter() {
            let d = deltas.entry(entry.script_public_key.clone()).or_default();
            if entry.effective_daa <= mature_bound {
                d.0 -= entry.amount as i128;
                // Spent while mature: drop its retained promotion too. A spent coin can never
                // need a demotion (a reorg re-introducing it re-adds it through `diff.add`,
                // which re-classifies and re-enqueues it), while a lingering entry gets demoted
                // by later score drops against a coin that no longer exists, corrupting the
                // buckets. Invariant (matching the startup reseed): every queued entry's coin
                // exists in the virtual UTXO set.
                self.maturation_queue_store.delete_batch(batch, entry.effective_daa + self.coin_age_maturity_w, outpoint).unwrap();
            } else {
                d.1 -= entry.amount as i128;
                d.2 -= (entry.amount as i128) * (entry.effective_daa as i128);
                // Spent while immature: drop its pending promotion.
                self.maturation_queue_store.delete_batch(batch, entry.effective_daa + self.coin_age_maturity_w, outpoint).unwrap();
            }
        }
        for (outpoint, entry) in diff.add.iter() {
            let d = deltas.entry(entry.script_public_key.clone()).or_default();
            if entry.effective_daa <= mature_bound {
                d.0 += entry.amount as i128;
                // Inserted already mature (inherited anchor — consolidation keeps its age): enqueue
                // it anyway, in the RETAINED role (due is in the past, so forward sweeps skip it).
                // Without an entry the coin is invisible to the demotion sweep: a virtual re-anchor
                // crossing its maturity boundary desyncs the store from the `d − W` classification,
                // and a spend inside that window then drains the WRONG bucket (silently, on any
                // address whose immature mass covers the amount).
                let queued = MaturationEntry {
                    script_public_key: entry.script_public_key.clone(),
                    amount: entry.amount,
                    anchor: entry.effective_daa,
                };
                self.maturation_queue_store.insert_batch(batch, entry.effective_daa + self.coin_age_maturity_w, outpoint, queued).unwrap();
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
        deltas.retain(|_, d| *d != (0, 0, 0));
        if deltas.is_empty() {
            return;
        }
        let spks: Vec<ScriptPublicKey> = deltas.keys().cloned().collect();
        let (buckets, hits, misses) = self.age_buckets_store.get_many(&spks).unwrap();
        self.counters.age_buckets_cache_hits.fetch_add(hits, std::sync::atomic::Ordering::Relaxed);
        self.counters.age_buckets_cache_misses.fetch_add(misses, std::sync::atomic::Ordering::Relaxed);
        for (spk, b) in spks.iter().zip(buckets) {
            let (dm, div, dia) = deltas[spk];
            let (nm, ni, na) = (b.b_mat as i128 + dm, b.b_imm as i128 + div, b.a_imm as i128 + dia);
            if nm < 0 || ni < 0 || na < 0 {
                warn!(
                    "coin-age: bucket underflow while applying a diff (b_mat {} {:+}, b_imm {} {:+}, a_imm {} {:+}) — clamping to 0; the index is inconsistent and will be re-derived at next startup",
                    b.b_mat, dm, b.b_imm, div, b.a_imm, dia
                );
            }
            let next = AgeBuckets { b_mat: nm.max(0) as u64, b_imm: ni.max(0) as u64, a_imm: na.max(0) as u128 };
            self.age_buckets_store.set_batch(batch, spk, next).unwrap();
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

            // Fold all demotions by SPK first. The old implementation performed one
            // age-bucket read/modify/write per queued coin; during IBD re-anchors this
            // can serialize a large maturation range on the virtual-processor thread.
            let mut demotions: std::collections::HashMap<ScriptPublicKey, (u64, u128)> =
                std::collections::HashMap::new();

            for (raw, due) in self.maturation_queue_store.due_range(new_daa_score, watermark) {
                // Pure re-add (spent-after-maturing restore) — skip, `apply_age_diff` re-adds the
                // coin on the immature side. In add AND remove (same outpoint re-anchored with a
                // different `effective_daa`) — demote: the remove folds on the immature side and
                // must land on the demoted value.
                let outpoint = DbMaturationQueueStore::outpoint_of(&raw);
                if diff.add.contains_key(&outpoint) && !diff.remove.contains_key(&outpoint) {
                    continue;
                }

                debug!("coin-age sweep: demote {} anchor {} spk {}", due.amount, due.anchor, hex::encode(due.script_public_key.script()));
                let d = demotions.entry(due.script_public_key.clone()).or_default();
                d.0 = d.0.saturating_add(due.amount);
                d.1 = d.1.saturating_add((due.amount as u128).saturating_mul(due.anchor as u128));
            }

            if !demotions.is_empty() {
                let spks: Vec<ScriptPublicKey> = demotions.keys().cloned().collect();
                let (buckets, hits, misses) = self.age_buckets_store.get_many(&spks).unwrap();
                self.counters.age_buckets_cache_hits.fetch_add(hits, std::sync::atomic::Ordering::Relaxed);
                self.counters.age_buckets_cache_misses.fetch_add(misses, std::sync::atomic::Ordering::Relaxed);

                for (spk, b) in spks.iter().zip(buckets) {
                    let (amount, weighted_anchor) = demotions[spk];

                    if b.b_mat < amount {
                        warn!(
                            "coin-age sweep: demotion underflows b_mat ({} < {}) — clamping; queue entries with no backing coin (ghost)",
                            b.b_mat, amount
                        );
                    }

                    let next = AgeBuckets {
                        b_mat: b.b_mat.saturating_sub(amount),
                        b_imm: b.b_imm.saturating_add(amount),
                        a_imm: b.a_imm.saturating_add(weighted_anchor),
                    };
                    self.age_buckets_store.set_batch(batch, spk, next).unwrap();
                }
            }

            self.maturation_queue_store.set_watermark_batch(batch, new_daa_score).unwrap();
            return false;
        }

        // Fold all promotions by SPK, then read/modify/write each age bucket once.
        let mut promotions: std::collections::HashMap<ScriptPublicKey, (u64, u128)> =
            std::collections::HashMap::new();

        for (_, due) in self.maturation_queue_store.due_range(watermark, new_daa_score) {
            debug!("coin-age sweep: promote {} anchor {} spk {}", due.amount, due.anchor, hex::encode(due.script_public_key.script()));
            let d = promotions.entry(due.script_public_key.clone()).or_default();
            d.0 = d.0.saturating_add(due.amount);
            d.1 = d.1.saturating_add((due.amount as u128).saturating_mul(due.anchor as u128));

            // NOTE: the promoted entry is NOT deleted — it is retained for `coin_age_retention`
            // scores so the read path can DEMOTE when a POV falls below the watermark (side
            // chains, see `eff_balance_for_spk`). Retention pruning below reclaims it.
        }

        if !promotions.is_empty() {
            let spks: Vec<ScriptPublicKey> = promotions.keys().cloned().collect();
            let (buckets, hits, misses) = self.age_buckets_store.get_many(&spks).unwrap();
            self.counters.age_buckets_cache_hits.fetch_add(hits, std::sync::atomic::Ordering::Relaxed);
            self.counters.age_buckets_cache_misses.fetch_add(misses, std::sync::atomic::Ordering::Relaxed);

            for (spk, b) in spks.iter().zip(buckets) {
                let (amount, weighted_anchor) = promotions[spk];

                if b.b_imm < amount || b.a_imm < weighted_anchor {
                    warn!(
                        "coin-age sweep: promotion underflows the immature bucket (b_imm {} < {} or a_imm {} < {}) — clamping; queue entries with no backing coin (ghost)",
                        b.b_imm,
                        amount,
                        b.a_imm,
                        weighted_anchor
                    );
                }

                let next = AgeBuckets {
                    b_mat: b.b_mat.saturating_add(amount),
                    b_imm: b.b_imm.saturating_sub(amount),
                    a_imm: b.a_imm.saturating_sub(weighted_anchor),
                };
                self.age_buckets_store.set_batch(batch, spk, next).unwrap();
            }
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
                    // Mirror of the write-path guard in `sweep_maturation_queue`: skip ONLY the
                    // pure re-add (spent-after-maturing restore). A coin both re-added and
                    // removed (re-anchored with a different `effective_daa`) MUST be demoted so
                    // the content correction's remove folds on the demoted value — skipping it
                    // here leaves it mature while the remove folds immature, shifting the
                    // numerator by the coin's amount for any POV below the (node-local)
                    // watermark: a cross-node divergence.
                    if view_diffs.iter().any(|d| d.add.contains_key(&outpoint))
                        && !view_diffs.iter().any(|d| d.remove.contains_key(&outpoint))
                    {
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

    /// If any of `outpoints` is a burned escrow outpoint, returns the full burned set formatted as
    /// space-separated `txid:index`. A claiming miner reads this to slash exactly the burned members
    /// and re-batch the rest instead of bisecting to find them. `None` when none are burned.
    /// Burned outpoints among `outpoints` that a POV at `pov_daa_score` must already see.
    ///
    /// A burn enters the set once the sink reaches `event daa + finality`, and the flush runs after
    /// that sink's own blocks are validated — so a live node first applies it strictly above that
    /// score. The set itself carries no such bound, and a
    /// node holding rows above the blocks it is replaying (fresh sync, restart catch-up) would
    /// otherwise reject claims the network accepted before the burn existed.
    fn burned_outpoints_msg<'b>(
        &self,
        outpoints: impl Iterator<Item = &'b TransactionOutpoint>,
        pov_daa_score: u64,
    ) -> Option<String> {
        let guard = self.service_burned.read();
        let mut msg = String::new();
        for o in outpoints {
            if guard.get(o).is_some_and(|&burn_daa| burn_daa.saturating_add(self.finality_depth) < pov_daa_score) {
                if !msg.is_empty() {
                    msg.push(' ');
                }
                msg.push_str(&format!("{}:{}", o.transaction_id, o.index));
            }
        }
        (!msg.is_empty()).then_some(msg)
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

        // Service-bond (H6): a finality-deep miss burns the miner's escrow claims — spending one is
        // invalid forever. The set only ever contains reorg-immune entries, so every POV reaches the
        // same verdict. Report the full burned set (not the first), so a claiming miner slashes
        // exactly those and re-batches the rest. A burned outpoint is still present in the view
        // (burn is an overlay, not a deletion), so this never masks a genuine missing-input.
        if let Some(msg) = self.burned_outpoints_msg(transaction.inputs.iter().map(|i| &i.previous_outpoint), pov_daa_score) {
            info!("Rejecting transaction {} at daa {}: spend of burned escrow {}", transaction.id(), pov_daa_score, msg);
            return Err(TxRuleError::SpendOfBurnedEscrow(msg));
        }
        let mut entries = Vec::with_capacity(transaction.inputs.len());
        for input in transaction.inputs.iter() {
            if let Some(mut entry) = utxo_view.get(&input.previous_outpoint) {
                if let Some(anchor) = historical_anchor_override(&input.previous_outpoint)
                    && entry.effective_daa != anchor
                {
                    info!("Historical anchor override applied for {}", input.previous_outpoint);
                    entry.effective_daa = anchor;
                }
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
        if let Some(msg) = self.burned_outpoints_msg(mutable_tx.tx.inputs.iter().map(|i| &i.previous_outpoint), pov_daa_score) {
            return Err(TxRuleError::SpendOfBurnedEscrow(msg));
        }

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

/// Age anchors the canonical chain committed for specific historical spends, where clean
/// re-derivation computes a different value. Applied at input population so re-validation of the
/// canonical chain reproduces the committed state byte-for-byte. Keyed by outpoint; every listed
/// coin is long spent, so the table is inert outside historical validation.
const HISTORICAL_ANCHOR_OVERRIDES: &[(Hash, u32, u64)] = &[(
    // aeb4e536e444210419a3bf2fae8e582816ad36339be0f429c0eaac611e3bcab3:1 — spent at DAA 74807554
    // committing anchor 74780462; re-derivation yields 74780464.
    Hash::from_bytes([
        0xae, 0xb4, 0xe5, 0x36, 0xe4, 0x44, 0x21, 0x04, 0x19, 0xa3, 0xbf, 0x2f, 0xae, 0x8e, 0x58, 0x28, 0x16, 0xad, 0x36, 0x33,
        0x9b, 0xe0, 0xf4, 0x29, 0xc0, 0xea, 0xac, 0x61, 0x1e, 0x3b, 0xca, 0xb3,
    ]),
    1,
    74780462,
)];

fn historical_anchor_override(outpoint: &TransactionOutpoint) -> Option<u64> {
    HISTORICAL_ANCHOR_OVERRIDES
        .iter()
        .find(|(txid, index, _)| outpoint.transaction_id == *txid && outpoint.index == *index)
        .map(|&(_, _, anchor)| anchor)
}

/// The `AiRequest` rules that need nothing but the transaction itself: the per-model
/// `inference_reward` floor (`base[model] + ceil(max_tokens / 64) * TOKEN_STEP`) and the
/// `priority_fee` floor. Defined once and called from both the pre-UTXO fast path
/// (`check_ai_request_payload_rules_all`) and the full post-UTXO check, so the two cannot drift
/// apart and reach different verdicts for the same block.
fn check_ai_request_payload_rules(tx: &Transaction, req: &AiRequestPayload, minimums: &[([u8; 32], u64)]) -> BlockProcessResult<()> {
    if let Some(&(_, base_reward)) = minimums.iter().find(|(id, _)| *id == req.model_id) {
        let token_surcharge = ((req.max_tokens as u64 + 63) / 64) * INFERENCE_REWARD_TOKEN_STEP;
        let effective_min = base_reward + token_surcharge;
        if req.inference_reward < effective_min {
            return Err(AiRequestInferenceRewardBelowMinimum(tx.id(), req.inference_reward, effective_min, hex::encode(req.model_id)));
        }
    }
    if req.priority_fee < keryx_inference::MIN_AI_REQUEST_PRIORITY_FEE {
        return Err(AiRequestPriorityFeeBelowMinimum(tx.id(), req.priority_fee, keryx_inference::MIN_AI_REQUEST_PRIORITY_FEE));
    }
    Ok(())
}

/// Structural validation of the escrow output — `outputs[1]` present, a CSV P2PK script, and worth
/// at least `inference_reward`. Reads only the transaction's own outputs, so it needs no UTXO
/// context either; shared by the same two call sites as [`check_ai_request_payload_rules`].
fn check_ai_request_escrow_output(tx: &Transaction, req: &AiRequestPayload, routed: bool) -> BlockProcessResult<()> {
    if tx.outputs.len() < 2 {
        return Err(AiRequestMissingEscrowOutput(tx.id()));
    }
    let escrow_out = &tx.outputs[1];
    if routed {
        // H8 routing: the reward locks in the canonical keyless vault; the coinbase mints it
        // to the first accepted responder.
        if escrow_out.script_public_key.version() != 0
            || escrow_out.script_public_key.script() != &keryx_inference::INFERENCE_VAULT_SCRIPT[..]
        {
            return Err(AiRequestInvalidEscrowScript(tx.id()));
        }
    } else if !ScriptClass::is_csv_pay_to_pubkey(escrow_out.script_public_key.script()) {
        return Err(AiRequestInvalidEscrowScript(tx.id()));
    }
    if escrow_out.value < req.inference_reward {
        return Err(AiRequestEscrowBelowInferenceReward(tx.id(), escrow_out.value, req.inference_reward));
    }
    Ok(())
}

/// Every `AiRequest` rule that the transaction alone decides, run BEFORE the parallel UTXO
/// validation: a block whose only defect is a malformed or underpaid `AiRequest` is rejected
/// without first paying for the full UTXO pass. Only `calculated_fee >= priority_fee` genuinely
/// needs the UTXO result and stays behind.
///
/// Scheduling, not a rule change: every rule here is enforced identically by
/// `check_ai_request_inference_rewards` afterwards, so a patched and an unpatched node disqualify
/// exactly the same blocks and no gate is needed. A block violating both a payload rule and the
/// fee rule now reports the payload error instead of the fee one — same disqualification, only the
/// logged reason differs.
fn check_ai_request_payload_rules_all(txs: &[Transaction], minimums: &[([u8; 32], u64)], routed: bool) -> BlockProcessResult<()> {
    for tx in txs.iter().skip(1) {
        check_ai_request_tx_payload_rules(tx, minimums, routed)?;
    }
    Ok(())
}

/// Same rules for a SINGLE transaction, with no block around it — the entry point used by mempool
/// admission (`validate_mempool_transaction_impl`) so a transaction that would poison every block
/// including it never reaches a template. A non-`AiRequest` transaction passes untouched.
pub(super) fn check_ai_request_tx_payload_rules(tx: &Transaction, minimums: &[([u8; 32], u64)], routed: bool) -> BlockProcessResult<()> {
    if !tx.is_ai_request() {
        return Ok(());
    }
    if let Some(req) = AiRequestPayload::deserialize(&tx.payload) {
        check_ai_request_payload_rules(tx, &req, minimums)?;
        check_ai_request_escrow_output(tx, &req, routed)?;
    }
    Ok(())
}

/// Rejects the block if any AiRequest tx violates inference_reward/priority_fee/escrow rules:
/// - inference_reward below the per-model minimum
/// - priority_fee below MIN_AI_REQUEST_PRIORITY_FEE
/// - UTXO fee < priority_fee (inference_reward now goes to output[1] escrow, not fee)
/// - output[1] missing, not a CSV P2PK script, or value < inference_reward
///
/// All of these except the UTXO-fee one are also enforced earlier by
/// `check_ai_request_payload_rules_all`; they stay here so this remains the single authoritative
/// check, and so the original error precedence within a transaction is preserved.

fn check_ai_request_inference_rewards(
    txs: &[Transaction],
    validated: &[(keryx_consensus_core::tx::ValidatedTransaction<'_>, u32)],
    minimums: &[([u8; 32], u64)],
    routed: bool,
) -> BlockProcessResult<()> {
    let fee_map: std::collections::HashMap<TransactionId, u64> =
        validated.iter().map(|(vt, _)| (vt.id(), vt.calculated_fee)).collect();

    for tx in txs.iter().skip(1) {
        if !tx.is_ai_request() {
            continue;
        }
        if let Some(req) = AiRequestPayload::deserialize(&tx.payload) {
            // inference_reward and priority_fee minimums (shared with the pre-UTXO fast path).
            check_ai_request_payload_rules(tx, &req, minimums)?;
            // The one rule that genuinely needs UTXO validation: the fee covers priority_fee
            // (inference_reward itself goes to the output[1] escrow, not to the fee).
            if let Some(&calculated_fee) = fee_map.get(&tx.id()) {
                if calculated_fee < req.priority_fee {
                    return Err(AiRequestFeeBelowInferenceReward(tx.id(), calculated_fee, req.priority_fee));
                }
            }
            // Escrow output[1] structure (shared with the pre-UTXO fast path).
            check_ai_request_escrow_output(tx, &req, routed)?;
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
    use keryx_consensus_core::tx::TransactionOutput;
    use keryx_txscript::opcodes;

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

    // ── check_ai_request_payload_rules_all (pre-UTXO fast path) ──────────────

    /// `<seq_len> <seq bytes> OP_CSV OP_DATA_32 <32-byte key> OP_CHECKSIG` — the shape
    /// `ScriptClass::is_csv_pay_to_pubkey` accepts (`crypto/txscript/src/script_class.rs`).
    fn csv_p2pk_script() -> ScriptPublicKey {
        let mut s = vec![3u8, 0x60, 0x8c, 0x00]; // seq_len = 3, then the sequence bytes
        s.push(opcodes::codes::OpCheckSequenceVerify);
        s.push(opcodes::codes::OpData32);
        s.extend_from_slice(&[0xABu8; 32]);
        s.push(opcodes::codes::OpCheckSig);
        ScriptPublicKey::new(0, s.into())
    }

    /// `max_tokens` here is 100 ⇒ ceil(100/64) = 2 surcharge steps on top of the base. The tx
    /// carries a well-formed escrow `outputs[1]` funded at `inference_reward`, so the only thing
    /// under test is the reward floor.
    fn ai_request_with_reward(model_id: [u8; 32], inference_reward: u64) -> Transaction {
        let req = AiRequestPayload::new(model_id, 100, inference_reward, 30_000_000, b"test prompt".to_vec());
        let outputs = vec![
            TransactionOutput::new(1, ScriptPublicKey::new(0, vec![].into())),
            TransactionOutput::new(inference_reward, csv_p2pk_script()),
        ];
        Transaction::new(0, vec![], outputs, 0, subnets::SUBNETWORK_ID_AI_REQUEST, 0, req.serialize())
    }

    fn vault_script() -> ScriptPublicKey {
        ScriptPublicKey::new(0, keryx_inference::INFERENCE_VAULT_SCRIPT.to_vec().into())
    }

    /// Same shape as [`ai_request_with_reward`], with `outputs[1]` under the caller's script and
    /// value — the two things the H8 routing branch decides on.
    fn ai_request_with_escrow(model_id: [u8; 32], inference_reward: u64, value: u64, script: ScriptPublicKey) -> Transaction {
        let req = AiRequestPayload::new(model_id, 100, inference_reward, 30_000_000, b"test prompt".to_vec());
        let outputs =
            vec![TransactionOutput::new(1, ScriptPublicKey::new(0, vec![].into())), TransactionOutput::new(value, script)];
        Transaction::new(0, vec![], outputs, 0, subnets::SUBNETWORK_ID_AI_REQUEST, 0, req.serialize())
    }

    /// Reward high enough to clear the effective minimum for `max_tokens = 100`.
    fn routed_reward(base: u64) -> u64 {
        base + 2 * INFERENCE_REWARD_TOKEN_STEP
    }

    #[test]
    fn vault_escrow_is_accepted_past_the_routing_gate() {
        let model_id = [0x77u8; 32];
        let base = 400_000_000u64;
        let reward = routed_reward(base);
        let txs = vec![make_coinbase_no_caps(), ai_request_with_escrow(model_id, reward, reward, vault_script())];
        assert!(check_ai_request_payload_rules_all(&txs, &[(model_id, base)], true).is_ok());
    }

    /// The cutover is sharp in both directions: a client that keeps naming a miner past the gate
    /// is refused, exactly as a client locking the vault before it.
    #[test]
    fn csv_escrow_is_rejected_past_the_routing_gate() {
        let model_id = [0x77u8; 32];
        let base = 400_000_000u64;
        let reward = routed_reward(base);
        let txs = vec![make_coinbase_no_caps(), ai_request_with_escrow(model_id, reward, reward, csv_p2pk_script())];
        assert!(matches!(
            check_ai_request_payload_rules_all(&txs, &[(model_id, base)], true),
            Err(AiRequestInvalidEscrowScript(_))
        ));
    }

    #[test]
    fn vault_escrow_is_rejected_before_the_routing_gate() {
        let model_id = [0x77u8; 32];
        let base = 400_000_000u64;
        let reward = routed_reward(base);
        let txs = vec![make_coinbase_no_caps(), ai_request_with_escrow(model_id, reward, reward, vault_script())];
        assert!(matches!(
            check_ai_request_payload_rules_all(&txs, &[(model_id, base)], false),
            Err(AiRequestInvalidEscrowScript(_))
        ));
    }

    /// The vault is matched on script *and* version: the coinbase mint reads the canonical form
    /// only, so a look-alike at another version must not lock a reward.
    #[test]
    fn vault_escrow_at_a_nonzero_script_version_is_rejected() {
        let model_id = [0x77u8; 32];
        let base = 400_000_000u64;
        let reward = routed_reward(base);
        let odd_version = ScriptPublicKey::new(1, keryx_inference::INFERENCE_VAULT_SCRIPT.to_vec().into());
        let txs = vec![make_coinbase_no_caps(), ai_request_with_escrow(model_id, reward, reward, odd_version)];
        assert!(matches!(
            check_ai_request_payload_rules_all(&txs, &[(model_id, base)], true),
            Err(AiRequestInvalidEscrowScript(_))
        ));
    }

    /// The mint pays `inference_reward`, so a vault holding less than that would mint coins the
    /// request never locked.
    #[test]
    fn underfunded_vault_is_rejected_past_the_routing_gate() {
        let model_id = [0x77u8; 32];
        let base = 400_000_000u64;
        let reward = routed_reward(base);
        let txs = vec![make_coinbase_no_caps(), ai_request_with_escrow(model_id, reward, reward - 1, vault_script())];
        assert!(matches!(
            check_ai_request_payload_rules_all(&txs, &[(model_id, base)], true),
            Err(AiRequestEscrowBelowInferenceReward(_, _, _))
        ));
    }

    #[test]
    fn missing_escrow_output_is_rejected_past_the_routing_gate() {
        let model_id = [0x77u8; 32];
        let base = 400_000_000u64;
        let req = AiRequestPayload::new(model_id, 100, routed_reward(base), 30_000_000, b"test prompt".to_vec());
        let outputs = vec![TransactionOutput::new(1, ScriptPublicKey::new(0, vec![].into()))];
        let tx = Transaction::new(0, vec![], outputs, 0, subnets::SUBNETWORK_ID_AI_REQUEST, 0, req.serialize());
        assert!(matches!(
            check_ai_request_payload_rules_all(&[make_coinbase_no_caps(), tx], &[(model_id, base)], true),
            Err(AiRequestMissingEscrowOutput(_))
        ));
    }

    #[test]
    fn reward_at_the_effective_minimum_is_accepted() {
        let model_id = [0x55u8; 32];
        let base = 400_000_000u64;
        let minimums = [(model_id, base)];
        let effective_min = base + 2 * INFERENCE_REWARD_TOKEN_STEP;
        let txs = vec![make_coinbase_no_caps(), ai_request_with_reward(model_id, effective_min)];
        assert!(check_ai_request_payload_rules_all(&txs, &minimums, false).is_ok());
    }

    #[test]
    fn reward_one_sompi_below_the_effective_minimum_is_rejected() {
        let model_id = [0x55u8; 32];
        let base = 400_000_000u64;
        let minimums = [(model_id, base)];
        let effective_min = base + 2 * INFERENCE_REWARD_TOKEN_STEP;
        let txs = vec![make_coinbase_no_caps(), ai_request_with_reward(model_id, effective_min - 1)];
        assert!(matches!(
            check_ai_request_payload_rules_all(&txs, &minimums, false),
            Err(AiRequestInferenceRewardBelowMinimum(_, _, _, _))
        ));
    }

    /// The token surcharge is the part a sender is most likely to omit — paying only the base
    /// must still be rejected (the on-chain case of 2026-07-30: 400M paid, 420M required).
    #[test]
    fn base_reward_without_the_token_surcharge_is_rejected() {
        let model_id = [0x55u8; 32];
        let base = 400_000_000u64;
        let minimums = [(model_id, base)];
        let txs = vec![make_coinbase_no_caps(), ai_request_with_reward(model_id, base)];
        assert!(matches!(
            check_ai_request_payload_rules_all(&txs, &minimums, false),
            Err(AiRequestInferenceRewardBelowMinimum(_, _, _, _))
        ));
    }

    /// A model absent from the table has no floor to enforce — the era's own gating decides which
    /// table is in force, and an unknown model_id must not be rejected by this rule.
    #[test]
    fn unknown_model_id_has_no_minimum() {
        let minimums = [([0x55u8; 32], 400_000_000u64)];
        let txs = vec![make_coinbase_no_caps(), ai_request_with_reward([0x66u8; 32], 1)];
        assert!(check_ai_request_payload_rules_all(&txs, &minimums, false).is_ok());
    }

    /// Rule moved forward on nectopower's review: a below-minimum `priority_fee` needs no UTXO
    /// context either, so switching which field is invalidated must not restore the slow path.
    #[test]
    fn priority_fee_below_minimum_is_rejected_early() {
        let model_id = [0x55u8; 32];
        let minimums = [(model_id, 400_000_000u64)];
        let reward = 400_000_000 + 2 * INFERENCE_REWARD_TOKEN_STEP;
        let req = AiRequestPayload::new(model_id, 100, reward, keryx_inference::MIN_AI_REQUEST_PRIORITY_FEE - 1, b"p".to_vec());
        let outputs =
            vec![TransactionOutput::new(1, ScriptPublicKey::new(0, vec![].into())), TransactionOutput::new(reward, csv_p2pk_script())];
        let tx = Transaction::new(0, vec![], outputs, 0, subnets::SUBNETWORK_ID_AI_REQUEST, 0, req.serialize());
        assert!(matches!(
            check_ai_request_payload_rules_all(&[make_coinbase_no_caps(), tx], &minimums, false),
            Err(AiRequestPriorityFeeBelowMinimum(_, _, _))
        ));
    }

    /// Same reasoning for the escrow output: it lives in the transaction, so an underfunded or
    /// missing escrow is decided before the UTXO pass, not after.
    #[test]
    fn underfunded_and_missing_escrow_are_rejected_early() {
        let model_id = [0x55u8; 32];
        let minimums = [(model_id, 400_000_000u64)];
        let reward = 400_000_000 + 2 * INFERENCE_REWARD_TOKEN_STEP;
        let req = AiRequestPayload::new(model_id, 100, reward, 30_000_000, b"p".to_vec());

        let underfunded = vec![
            TransactionOutput::new(1, ScriptPublicKey::new(0, vec![].into())),
            TransactionOutput::new(reward - 1, csv_p2pk_script()),
        ];
        let tx = Transaction::new(0, vec![], underfunded, 0, subnets::SUBNETWORK_ID_AI_REQUEST, 0, req.serialize());
        assert!(matches!(
            check_ai_request_payload_rules_all(&[make_coinbase_no_caps(), tx], &minimums, false),
            Err(AiRequestEscrowBelowInferenceReward(_, _, _))
        ));

        let tx = Transaction::new(0, vec![], vec![], 0, subnets::SUBNETWORK_ID_AI_REQUEST, 0, req.serialize());
        assert!(matches!(
            check_ai_request_payload_rules_all(&[make_coinbase_no_caps(), tx], &minimums, false),
            Err(AiRequestMissingEscrowOutput(_))
        ));
    }

    /// Mempool admission uses the single-transaction entry point, with no block around it. The
    /// real case of 2026-07-30: 400 000 000 sompi paid where 420 000 000 was required, a request
    /// that poisoned every block including it. A non-AiRequest transaction must pass untouched.
    #[test]
    fn single_tx_entry_point_rejects_the_underpaid_request_and_passes_others() {
        let model_id = [0x55u8; 32];
        let minimums = [(model_id, 400_000_000u64)];
        let underpaid = ai_request_with_reward(model_id, 400_000_000);
        assert!(matches!(
            check_ai_request_tx_payload_rules(&underpaid, &minimums, false),
            Err(AiRequestInferenceRewardBelowMinimum(_, _, _, _))
        ));

        let ok = ai_request_with_reward(model_id, 400_000_000 + 2 * INFERENCE_REWARD_TOKEN_STEP);
        assert!(check_ai_request_tx_payload_rules(&ok, &minimums, false).is_ok());

        // A plain transaction carries no AiRequest payload and is none of this rule's business.
        let plain = Transaction::new(0, vec![], vec![], 0, subnets::SUBNETWORK_ID_NATIVE, 0, vec![]);
        assert!(check_ai_request_tx_payload_rules(&plain, &minimums, false).is_ok());
    }

    /// The fast path and the full check must reach the same verdict — that equivalence is the
    /// reason running the rule earlier needs no gate.
    #[test]
    fn fast_path_matches_the_full_check_verdict() {
        let model_id = [0x55u8; 32];
        let base = 400_000_000u64;
        let minimums = [(model_id, base)];
        let bad = ai_request_with_reward(model_id, base);
        let txs = vec![make_coinbase_no_caps(), bad];
        let fast = check_ai_request_payload_rules_all(&txs, &minimums, false);
        let full = check_ai_request_inference_rewards(&txs, &[], &minimums, false);
        assert!(matches!(fast, Err(AiRequestInferenceRewardBelowMinimum(_, _, _, _))));
        assert!(matches!(full, Err(AiRequestInferenceRewardBelowMinimum(_, _, _, _))));
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

