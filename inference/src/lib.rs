/// Keryx OPoI (Optimistic Proof of Inference) — Phase 1
///
/// Provides a deterministic synthetic MLP that every miner must execute
/// per block template.  The result is committed in the coinbase `extra_data`
/// field, making inference mandatory for block production.
///
/// Phase 1 is CPU-only and uses no downloaded model — weights are derived
/// deterministically from `MODEL_SEED`.  Future phases will:
///   - Verify outputs on-chain via optimistic fraud proofs (Phase 2)
///   - Swap the synthetic MLP for real SLM inference via IPFS (Phase 3)

pub mod model;
pub mod task;

pub use task::{InferenceResult, InferenceTask};

use candle_core::Device;

/// Errors returned by the inference engine.
#[derive(Debug, thiserror::Error)]
pub enum InferenceError {
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),
}

/// Runs the OPoI synthetic MLP on the given 64-bit miner nonce.
///
/// This is the single entry-point used by the miner.  It constructs an
/// `InferenceTask` from the nonce, runs the forward pass on CPU, and returns
/// an `InferenceResult` whose `as_hex8()` is appended to `extra_data`.
pub fn run_inference(nonce: u64) -> Result<InferenceResult, InferenceError> {
    let task = InferenceTask::from_nonce(nonce);
    let output = model::forward(&task.input, &Device::Cpu)?;
    Ok(InferenceResult { output })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inference_is_deterministic() {
        let r1 = run_inference(0xDEAD_BEEF_CAFE_1337).unwrap();
        let r2 = run_inference(0xDEAD_BEEF_CAFE_1337).unwrap();
        assert_eq!(r1.output, r2.output, "same nonce must produce same output");
    }

    #[test]
    fn different_nonces_produce_different_outputs() {
        let r1 = run_inference(1).unwrap();
        let r2 = run_inference(2).unwrap();
        assert_ne!(r1.output, r2.output, "different nonces should differ");
    }

    #[test]
    fn hex8_is_16_chars() {
        let r = run_inference(42).unwrap();
        assert_eq!(r.as_hex8().len(), 16);
    }
}
