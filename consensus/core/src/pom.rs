//! Proof-of-Model — possession-only post-PoW witness types.
//!
//! These carry the succinct, weightless-verifiable proof that the block producer held
//! tier `tier`'s full quantized weight blob in VRAM while mining. The proof is a
//! Block-level field (NOT part of `pre_pow_hash`): the Fiat-Shamir challenges derive
//! from the winning `nonce`/`pow_value`, so the proof can only be built post-PoW.
//!
//! Verification (see `header_processor::post_pow_validation`, build order §5) is 100 %
//! deterministic integer/Merkle checks against the per-tier root `R_T` pinned in
//! `config::params` — no ε. The canonical chunk layout (32 B chunks, name-sorted GGUF
//! tensors) is produced by the offline `pom-rt-builder`. See `POM_CONSENSUS_SPEC.md`.

use borsh::{BorshDeserialize, BorshSerialize};
use serde::{Deserialize, Serialize};

/// A PoM tier binding: the model whose possession this tier proves, plus its canonical
/// 32 B-chunk Merkle root `R_T` and chunk count `N` (from the offline `pom-rt-builder`).
/// Pinned per network in `config::params` (`POM_TIERS`); the tier index is the slice
/// position. `model_id` links the tier to the miner's declared model (cross-checked in
/// `post_pow_validation`), so a miner cannot claim a tier it does not hold.
#[derive(Clone, Copy, Debug)]
pub struct PomTier {
    pub model_id: [u8; 32],
    pub root: [u8; 32],
    pub chunks: u64,
}

/// One Fiat-Shamir-opened step of the possession walk.
///
/// Lets a weightless verifier re-check a single step `i`: that `state_before` is the
/// committed trace leaf at `i`, that the weight chunk at index `state_before % N` is
/// genuine (Merkle path to `R_T`), and that `transition(state_before, chunk)` is the
/// committed trace leaf at `i+1`.
#[derive(Clone, Debug, Serialize, Deserialize, BorshSerialize, BorshDeserialize)]
#[serde(rename_all = "camelCase")]
pub struct PomOpening {
    /// Walk state before this step, i.e. `state[i]`. The chunk index is `state_before % N`.
    pub state_before: u64,
    /// The 32-byte weight chunk read at that index (candle's exact quantized bytes).
    pub chunk: [u8; 32],
    /// Merkle path proving `leaf(chunk)` is at index `state_before % N` under `R_T`.
    pub weight_path: Vec<[u8; 32]>,
    /// Merkle path proving `leaf(state_before)` is at step index `i` under `trace_root`.
    pub trace_path_before: Vec<[u8; 32]>,
    /// Merkle path proving `leaf(transition(state_before, chunk))` is at step `i+1` under `trace_root`.
    pub trace_path_after: Vec<[u8; 32]>,
}

/// Post-PoW possession witness, carried at the `Block` level (outside `pre_pow_hash`).
#[derive(Clone, Debug, Serialize, Deserialize, BorshSerialize, BorshDeserialize)]
#[serde(rename_all = "camelCase")]
pub struct PomProof {
    /// Tier index — selects `R_T`, `target_T` and the reward bracket.
    pub tier: u8,
    /// Merkle root committing the full execution trace `state[0..=K]`.
    pub trace_root: [u8; 32],
    /// Claimed PoW value = `kHeavyHash(final_state)`; checked `<= target_T`.
    pub pow_value: [u8; 32],
    /// Final walk state `state[K]`; `kHeavyHash(final_state)` must equal `pow_value`.
    pub final_state: u64,
    /// Merkle path proving `leaf(final_state)` is the last trace leaf under `trace_root`.
    pub final_trace_path: Vec<[u8; 32]>,
    /// The `t` Fiat-Shamir openings, in challenge order.
    pub openings: Vec<PomOpening>,
}

impl PomProof {
    /// Number of opened steps (`t`).
    pub fn opening_count(&self) -> usize {
        self.openings.len()
    }

    /// Rough in-memory / wire byte size (for cache accounting). Dominated by Merkle paths.
    pub fn approx_bytes(&self) -> usize {
        let path_bytes = |p: &Vec<[u8; 32]>| p.len() * 32;
        let openings: usize = self
            .openings
            .iter()
            .map(|o| 32 + 8 + path_bytes(&o.weight_path) + path_bytes(&o.trace_path_before) + path_bytes(&o.trace_path_after))
            .sum();
        1 + 32 + 32 + 8 + path_bytes(&self.final_trace_path) + openings
    }
}
