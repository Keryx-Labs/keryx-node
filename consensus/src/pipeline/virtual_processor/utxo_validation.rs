use super::VirtualStateProcessor;
use crate::{
    errors::{
        BlockProcessResult,
        RuleError::{
            BadAcceptedIDMerkleRoot, BadCoinbaseTransaction, BadUTXOCommitment, InvalidTransactionsInUtxoContext,
            WrongHeaderPruningPoint,
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
use crate::model::stores::ai_slash::{AiResponseRecord, AiResponseStore, AiResponseStoreReader, AiSlashedStore, AiSlashedStoreReader, OutpointKey};
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
use keryx_consensus_core::collateral::CHALLENGE_WINDOW_BLOCKS;
use keryx_core::{info, trace, warn};
use keryx_hashes::Hash;
use keryx_inference::{AiChallengePayload, AiResponsePayload, FraudProofResult, FRAUD_PROOF_LEN, verify_fraud_proof};
use keryx_consensus_core::tx::TransactionOutpoint;
use keryx_muhash::MuHash;
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

        // Verify all transactions are valid in context
        let current_utxo_view = selected_parent_utxo_view.compose(&ctx.mergeset_diff);
        let validated_transactions =
            self.validate_transactions_in_parallel(&txs, &current_utxo_view, header.daa_score, TxValidationFlags::Full);
        if validated_transactions.len() < txs.len() - 1 {
            // Some non-coinbase transactions are invalid
            return Err(InvalidTransactionsInUtxoContext(txs.len() - 1 - validated_transactions.len(), txs.len() - 1));
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
        let expected_coinbase = self
            .coinbase_manager
            .expected_coinbase_transaction(daa_score, miner_data, ghostdag_data, mergeset_rewards, mergeset_non_daa)
            .unwrap()
            .tx;
        if hashing::tx::hash(coinbase) != hashing::tx::hash(&expected_coinbase) { Err(BadCoinbaseTransaction) } else { Ok(()) }
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
        // OPoI Phase 3 A4: reject any spend of a slashed escrow outpoint (DB-backed, rayon-safe).
        // AiChallenge txs are exempted: they are the legitimate mechanism for spending slashed escrows.
        if !transaction.is_ai_challenge() {
            for input in transaction.inputs.iter() {
                let key = OutpointKey::new(input.previous_outpoint.transaction_id, input.previous_outpoint.index);
                if self.ai_slashed_store.is_slashed(key).unwrap_or(false) {
                    return Err(TxRuleError::SpendingSlashedEscrow);
                }
            }
        }

        // OPoI Phase 3 D: AiChallenge with inputs — validate fraud proof and escrow outpoint.
        // An AiChallenge that has inputs is a spending transaction: the escrow moves to the challenger.
        // Validation must pass before the standard UTXO population below to give a clear error.
        if transaction.is_ai_challenge() && !transaction.inputs.is_empty() {
            if let Some(challenge) = AiChallengePayload::deserialize(&transaction.payload) {
                let rh = Hash::from_bytes(challenge.response_hash);
                let record = self.ai_response_store.get(rh).map_err(|_| TxRuleError::AiChallengeUnknownResponse)?;

                // Challenge window must be open.
                if pov_daa_score > record.inclusion_blue_score + CHALLENGE_WINDOW_BLOCKS as u64 {
                    return Err(TxRuleError::AiChallengeOutsideWindow);
                }

                // Fraud proof validation (Phase 3 C re-execution proof).
                if !challenge.proof_data.is_empty() {
                    if challenge.proof_data.len() == FRAUD_PROOF_LEN {
                        let submitted_rh: &[u8; 32] = challenge.proof_data.as_slice().try_into().unwrap();
                        if submitted_rh != &record.request_hash {
                            return Err(TxRuleError::AiChallengeBadRequestHash);
                        }
                        if verify_fraud_proof(&record.request_hash, &record.claimed_commitment) != FraudProofResult::Valid {
                            return Err(TxRuleError::AiChallengeProofInvalid);
                        }
                    } else {
                        return Err(TxRuleError::AiChallengeProofInvalid);
                    }
                }

                // The tx must spend the correct escrow outpoint: (coinbase_tx_id, index 1).
                let expected = TransactionOutpoint { transaction_id: record.coinbase_tx_id, index: 1 };
                let has_correct_input = transaction.inputs.iter().any(|inp| inp.previous_outpoint == expected);
                if !has_correct_input {
                    return Err(TxRuleError::AiChallengeWrongEscrowInput);
                }
            }
        }

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
                let (request_hash, claimed_commitment) =
                    if let Some(resp) = AiResponsePayload::deserialize(&tx.payload) {
                        let commitment: [u8; 32] = if resp.result.len() >= 32 {
                            resp.result[0..32].try_into().unwrap()
                        } else {
                            [0u8; 32] // under-length result → zero commitment → always challengeable
                        };
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
                if let Err(e) = self.ai_response_store.set(response_hash, record) {
                    warn!("OPoI: failed to register AiResponse in DB: {}", e);
                } else {
                    info!("OPoI: registered AiResponse response_hash={}", hex::encode(response_hash_bytes));
                }
            }
        }

        // Process AiChallenge txs — update slash state.
        //
        // AiChallenge with inputs (Phase 3 D): fund transfer already validated in
        // validate_transaction_in_utxo_context; here we just mark the escrow as slashed
        // for auditing and to prevent duplicate challenges.
        //
        // AiChallenge with 0 inputs (Phase A compat):
        //   proof_data empty (0 bytes)       → slash unconditionally within window.
        //   proof_data 32 bytes (request_hash) → re-execution proof.
        for tx in txs.iter().skip(1) {
            if tx.is_ai_challenge() {
                if let Some(challenge) = AiChallengePayload::deserialize(&tx.payload) {
                    let rh = Hash::from_bytes(challenge.response_hash);

                    // Phase 3 D: spending AiChallenge — escrow already transferred via UTXO.
                    // Just mark as slashed for auditing / prevent re-challenge.
                    if !tx.inputs.is_empty() {
                        if let Ok(record) = self.ai_response_store.get(rh) {
                            let key = OutpointKey::new(record.coinbase_tx_id, 1);
                            if let Err(e) = self.ai_slashed_store.set(key, pov_daa_score) {
                                warn!("OPoI: failed to record slash (spending challenge) in DB: {}", e);
                            } else {
                                info!("OPoI: escrow spent by challenger — response_hash={}", hex::encode(challenge.response_hash));
                            }
                        }
                        continue; // fund transfer was handled by UTXO; skip 0-input logic below
                    }

                    // 0-input AiChallenge (Phase A compat + Phase 3 C re-execution).
                    let slash_accepted = if challenge.proof_data.is_empty() {
                        // Phase A compatibility: empty proof → unconditional slash within window.
                        true
                    } else if challenge.proof_data.len() == FRAUD_PROOF_LEN {
                        // Phase 3 C: re-execution fraud proof.
                        // proof_data = request_hash (32 bytes).
                        let submitted_request_hash: &[u8; 32] =
                            challenge.proof_data.as_slice().try_into().unwrap();
                        match self.ai_response_store.get(rh) {
                            Ok(ref rec) if submitted_request_hash == &rec.request_hash => {
                                match verify_fraud_proof(&rec.request_hash, &rec.claimed_commitment) {
                                    FraudProofResult::Valid => {
                                        info!(
                                            "OPoI: fraud proven via re-execution — response_hash={}",
                                            hex::encode(challenge.response_hash)
                                        );
                                        true
                                    }
                                    FraudProofResult::Invalid => {
                                        warn!(
                                            "OPoI: challenge rejected — miner commitment matches re-execution — response_hash={}",
                                            hex::encode(challenge.response_hash)
                                        );
                                        false
                                    }
                                }
                            }
                            Ok(_) => {
                                warn!(
                                    "OPoI: challenger submitted wrong request_hash — response_hash={}",
                                    hex::encode(challenge.response_hash)
                                );
                                false
                            }
                            Err(_) => false,
                        }
                    } else {
                        warn!(
                            "OPoI: proof_data length {} not recognised — response_hash={}",
                            challenge.proof_data.len(),
                            hex::encode(challenge.response_hash)
                        );
                        false
                    };

                    if slash_accepted {
                        match self.ai_response_store.get(rh) {
                            Ok(record) if pov_daa_score <= record.inclusion_blue_score + CHALLENGE_WINDOW_BLOCKS as u64 => {
                                let key = OutpointKey::new(record.coinbase_tx_id, 1);
                                if let Err(e) = self.ai_slashed_store.set(key, pov_daa_score) {
                                    warn!("OPoI: failed to record slash in DB: {}", e);
                                } else {
                                    info!("OPoI: escrow slashed — response_hash={}", hex::encode(challenge.response_hash));
                                }
                            }
                            Ok(_) => warn!(
                                "OPoI: challenge outside window — response_hash={}",
                                hex::encode(challenge.response_hash)
                            ),
                            Err(_) => warn!(
                                "OPoI: challenge for unknown response — response_hash={}",
                                hex::encode(challenge.response_hash)
                            ),
                        }
                    }
                }
            }
        }
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

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use super::*;

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
