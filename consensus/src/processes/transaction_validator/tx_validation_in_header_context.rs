//! Groups transaction validations that depend on the containing header and/or
//! its past headers (but do not depend on UTXO state or other transactions in
//! the containing block)

use super::{
    TransactionValidator,
    errors::{TxResult, TxRuleError},
};
use crate::constants::LOCK_TIME_THRESHOLD;
use keryx_consensus_core::tx::Transaction;

pub(crate) enum LockTimeType {
    Finalized,
    DaaScore,
    Time,
}

pub(crate) enum LockTimeArg {
    Finalized,
    DaaScore(u64),
    MedianTime(u64),
}

impl TransactionValidator {
    pub(crate) fn validate_tx_in_header_context_with_args(
        &self,
        tx: &Transaction,
        ctx_daa_score: u64,
        ctx_block_time: u64,
    ) -> TxResult<()> {
        self.validate_tx_in_header_context(
            tx,
            match Self::get_lock_time_type(tx) {
                LockTimeType::Finalized => LockTimeArg::Finalized,
                LockTimeType::DaaScore => LockTimeArg::DaaScore(ctx_daa_score),
                LockTimeType::Time => LockTimeArg::MedianTime(ctx_block_time),
            },
            ctx_daa_score,
        )
    }

    pub(crate) fn validate_tx_in_header_context(
        &self,
        tx: &Transaction,
        lock_time_arg: LockTimeArg,
        ctx_daa_score: u64,
    ) -> TxResult<()> {
        self.check_tx_is_finalized(tx, lock_time_arg)?;
        self.check_ai_response_payload_era(tx, ctx_daa_score)
    }

    /// AiResponse payloads have an exact length per era: 78 bytes before `opoi_v2_activation`
    /// (matches what pre-v2 binaries enforce statelessly), 142 bytes from it (model_id +
    /// result_commitment). Runs with the containing-block DAA in consensus paths and with
    /// the virtual DAA in mempool/template paths, so stale-format txs are evicted from
    /// templates at the gate crossing instead of producing an invalid block.
    fn check_ai_response_payload_era(&self, tx: &Transaction, ctx_daa_score: u64) -> TxResult<()> {
        if !tx.is_ai_response() {
            return Ok(());
        }
        let expected = if self.opoi_v2_activation.is_active(ctx_daa_score) {
            keryx_inference::AI_RESPONSE_PAYLOAD_V2_LEN
        } else {
            keryx_inference::AI_RESPONSE_PAYLOAD_LEN
        };
        let len = tx.payload.len();
        if len < expected {
            return Err(TxRuleError::AiPayloadTooShort(len, expected));
        }
        if len > expected {
            return Err(TxRuleError::AiPayloadTooLong(len, expected));
        }
        Ok(())
    }

    pub(crate) fn get_lock_time_type(tx: &Transaction) -> LockTimeType {
        match tx.lock_time {
            // Lock time of zero means the transaction is finalized.
            0 => LockTimeType::Finalized,

            // The lock time field of a transaction is either a block DAA score at
            // which the transaction is finalized or a timestamp depending on if the
            // value is before the LOCK_TIME_THRESHOLD. When it is under the
            // threshold it is a DAA score
            t if t < LOCK_TIME_THRESHOLD => LockTimeType::DaaScore,

            // ..and when equal or above the threshold it represents time
            _t => LockTimeType::Time,
        }
    }

    fn check_tx_is_finalized(&self, tx: &Transaction, lock_time_arg: LockTimeArg) -> TxResult<()> {
        let block_time_or_daa_score = match lock_time_arg {
            LockTimeArg::Finalized => return Ok(()),
            LockTimeArg::DaaScore(ctx_daa_score) => ctx_daa_score,
            LockTimeArg::MedianTime(ctx_block_time) => ctx_block_time,
        };

        if tx.lock_time < block_time_or_daa_score {
            return Ok(());
        }

        // At this point, the transaction's lock time hasn't occurred yet, but
        // the transaction might still be finalized if the sequence number
        // for all transaction inputs is maxed out.
        for (i, input) in tx.inputs.iter().enumerate() {
            if input.sequence != u64::MAX {
                return Err(TxRuleError::NotFinalized(i));
            }
        }

        Ok(())
    }
}
