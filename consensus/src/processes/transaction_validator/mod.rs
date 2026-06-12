pub mod errors;
pub mod tx_validation_in_header_context;
pub mod tx_validation_in_isolation;
pub mod tx_validation_in_utxo_context;
use std::sync::Arc;

use keryx_txscript::{
    SigCacheKey,
    caches::{Cache, TxScriptCacheCounters},
};

use keryx_consensus_core::{KType, mass::MassCalculator};

#[derive(Clone)]
pub struct TransactionValidator {
    max_tx_inputs: usize,
    max_tx_outputs: usize,
    max_signature_script_len: usize,
    max_script_public_key_len: usize,
    coinbase_payload_script_public_key_max_len: u8,
    coinbase_maturity: u64,
    ghostdag_k: KType,
    sig_cache: Cache<SigCacheKey, bool>,

    pub(crate) mass_calculator: MassCalculator,

    /// OPoI v2 gate — AiResponse payloads must be exactly 78 bytes before it and
    /// exactly 142 bytes (model_id + result_commitment) from it.
    opoi_v2_activation: keryx_consensus_core::config::params::ForkActivation,
}

impl TransactionValidator {
    pub fn new(
        max_tx_inputs: usize,
        max_tx_outputs: usize,
        max_signature_script_len: usize,
        max_script_public_key_len: usize,
        coinbase_payload_script_public_key_max_len: u8,
        coinbase_maturity: u64,
        ghostdag_k: KType,
        counters: Arc<TxScriptCacheCounters>,
        mass_calculator: MassCalculator,
        opoi_v2_activation: keryx_consensus_core::config::params::ForkActivation,
    ) -> Self {
        Self {
            max_tx_inputs,
            max_tx_outputs,
            max_signature_script_len,
            max_script_public_key_len,
            coinbase_payload_script_public_key_max_len,
            coinbase_maturity,
            ghostdag_k,
            sig_cache: Cache::with_counters(10_000, counters),
            mass_calculator,
            opoi_v2_activation,
        }
    }

    pub fn new_for_tests(
        max_tx_inputs: usize,
        max_tx_outputs: usize,
        max_signature_script_len: usize,
        max_script_public_key_len: usize,
        coinbase_payload_script_public_key_max_len: u8,
        coinbase_maturity: u64,
        ghostdag_k: KType,
        counters: Arc<TxScriptCacheCounters>,
    ) -> Self {
        Self {
            max_tx_inputs,
            max_tx_outputs,
            max_signature_script_len,
            max_script_public_key_len,
            coinbase_payload_script_public_key_max_len,
            coinbase_maturity,
            ghostdag_k,
            sig_cache: Cache::with_counters(10_000, counters),
            mass_calculator: MassCalculator::new(0, 0, 0, 0),
            // Tests target pre-v2 behaviour unless they construct a full validator.
            opoi_v2_activation: keryx_consensus_core::config::params::ForkActivation::never(),
        }
    }
}
