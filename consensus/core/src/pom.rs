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
use keryx_utils::mem_size::MemSizeEstimator;
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
    /// Merkle path proving `leaf(seed)` is the FIRST trace leaf (index 0) — binds the
    /// committed trace to this block's canonical seed (no forged start state).
    pub initial_trace_path: Vec<[u8; 32]>,
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
        1 + 32 + 32 + 8 + path_bytes(&self.initial_trace_path) + path_bytes(&self.final_trace_path) + openings
    }
}

impl MemSizeEstimator for PomProof {
    fn estimate_mem_bytes(&self) -> usize {
        size_of::<Self>() + self.approx_bytes()
    }
}

// ---------------------------------------------------------------------------
// Weightless verifier (build order §5c). Vendored byte-exact from `pom-core`
// (the reference, with its 15 tests); the canonical hash is blake3 (matches the
// `pom-rt-builder` R_T). `final_hash` (kHeavyHash) is supplied by the pipeline.
// ---------------------------------------------------------------------------

/// 32 B chunk = 4 little-endian u64 words.
pub const POM_CHUNK_WORDS: usize = 4;
/// Domain-separation salt folded into the walk seed (matches miner kernel / pom-core).
pub const POM_SEED_SALT: u64 = 0x4B65727978500; // "KeryxP"

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PomVerifyError {
    WrongOpeningCount,
    PowValueMismatch,
    TargetNotMet,
    BadInitialState,
    BadFinalTracePath,
    BadStateBeforePath,
    BadWeightPath,
    BadStateAfterPath,
}

#[inline]
fn blake(bytes: &[u8]) -> [u8; 32] {
    *blake3::hash(bytes).as_bytes()
}

#[inline]
fn mix64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d049bb133111eb);
    x ^= x >> 31;
    x
}

/// Initial walk state from a folded PoW seed (matches `pom-core::seed_state`).
#[inline]
pub fn pom_seed_state(pow_seed: u64) -> u64 {
    mix64(pow_seed ^ POM_SEED_SALT)
}

#[inline]
fn pph_words(pre_pow_hash: &[u8; 32]) -> [u64; 4] {
    let mut w = [0u64; 4];
    for (i, wi) in w.iter_mut().enumerate() {
        *wi = u64::from_le_bytes(pre_pow_hash[i * 8..i * 8 + 8].try_into().unwrap());
    }
    w
}

/// Canonical PoM block seed = initial walk state. mix64-fold of (nonce, time, pre_pow_hash).
/// BYTE-IDENTICAL to the miner kernel `pom_mine.cu::pom_seed_fold` and `src/pom.rs`.
pub fn pom_block_seed(pre_pow_hash: &[u8; 32], timestamp: u64, nonce: u64) -> u64 {
    let p = pph_words(pre_pow_hash);
    let mut s = mix64(nonce ^ 0x4B65727978531);
    s = mix64(s ^ timestamp);
    s = mix64(s ^ p[0]);
    s = mix64(s ^ p[1]);
    s = mix64(s ^ p[2]);
    s = mix64(s ^ p[3]);
    s
}

/// Canonical PoM pow value (256-bit, little-endian) compared against the target. mix64-fold of
/// (final_state, pre_pow_hash) — the memory-hardness is already paid by the K data-dependent
/// reads. BYTE-IDENTICAL to the miner kernel `pom_mine.cu::pom_pow_fold` and `src/pom.rs`.
pub fn pom_pow_value(final_state: u64, pre_pow_hash: &[u8; 32]) -> [u8; 32] {
    let p = pph_words(pre_pow_hash);
    let o0 = mix64(final_state ^ p[0] ^ 0x9E3779B97F4A7C15);
    let o1 = mix64(o0 ^ p[1] ^ 0xC2B2AE3D27D4EB4F);
    let o2 = mix64(o1 ^ p[2] ^ 0x165667B19E3779F9);
    let o3 = mix64(o2 ^ p[3] ^ 0xD6E8FEB86659FD93);
    let mut out = [0u8; 32];
    out[0..8].copy_from_slice(&o0.to_le_bytes());
    out[8..16].copy_from_slice(&o1.to_le_bytes());
    out[16..24].copy_from_slice(&o2.to_le_bytes());
    out[24..32].copy_from_slice(&o3.to_le_bytes());
    out
}

#[inline]
fn transition(state: u64, chunk: &[u64; POM_CHUNK_WORDS]) -> u64 {
    let mut h = state;
    for &w in chunk.iter() {
        h ^= w;
    }
    mix64(h)
}

#[inline]
fn chunk_to_words(c: &[u8; 32]) -> [u64; POM_CHUNK_WORDS] {
    let mut w = [0u64; POM_CHUNK_WORDS];
    for (i, wi) in w.iter_mut().enumerate() {
        *wi = u64::from_le_bytes(c[i * 8..i * 8 + 8].try_into().unwrap());
    }
    w
}

#[inline]
fn trace_leaf(state: u64) -> [u8; 32] {
    blake(&state.to_le_bytes())
}

fn hash_pair(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut buf = [0u8; 64];
    buf[..32].copy_from_slice(left);
    buf[32..].copy_from_slice(right);
    blake(&buf)
}

/// Little-endian 256-bit `a <= b`.
fn le_leq(a: &[u8; 32], b: &[u8; 32]) -> bool {
    for i in (0..32).rev() {
        if a[i] < b[i] {
            return true;
        }
        if a[i] > b[i] {
            return false;
        }
    }
    true
}

fn verify_merkle(leaf: [u8; 32], index: u64, path: &[[u8; 32]], root: &[u8; 32]) -> bool {
    let mut acc = leaf;
    let mut idx = index;
    for sib in path {
        acc = if idx & 1 == 0 { hash_pair(&acc, sib) } else { hash_pair(sib, &acc) };
        idx >>= 1;
    }
    &acc == root
}

/// Fiat-Shamir challenge step-indices (byte-layout identical to `pom-core::challenges`).
fn fs_challenges(pre_pow_hash: &[u8; 32], nonce: u64, trace_root: &[u8; 32], pow_value: &[u8; 32], t: usize, k: u32) -> Vec<u32> {
    let mut fs = [0u8; 104];
    fs[..32].copy_from_slice(pre_pow_hash);
    fs[32..40].copy_from_slice(&nonce.to_le_bytes());
    fs[40..72].copy_from_slice(trace_root);
    fs[72..104].copy_from_slice(pow_value);
    let seed = blake(&fs);
    let mut out = Vec::with_capacity(t);
    for j in 0..t as u64 {
        let mut buf = [0u8; 40];
        buf[..32].copy_from_slice(&seed);
        buf[32..].copy_from_slice(&j.to_le_bytes());
        let d = blake(&buf);
        let v = u64::from_le_bytes(d[..8].try_into().unwrap());
        out.push((v % k as u64) as u32);
    }
    out
}

/// Deterministically verify a possession proof against tier root `r_t` and `target`,
/// without the weights. `seed` is this block's canonical initial walk state; `final_hash`
/// is the node's `kHeavyHash` fold of `final_state` to the 256-bit pow value.
#[allow(clippy::too_many_arguments)]
pub fn verify_pom_proof<Hf: Fn(u64) -> [u8; 32]>(
    pre_pow_hash: &[u8; 32],
    nonce: u64,
    seed: u64,
    proof: &PomProof,
    n_chunks: u64,
    k: u32,
    t: usize,
    r_t: &[u8; 32],
    target: &[u8; 32],
    final_hash: Hf,
) -> Result<(), PomVerifyError> {
    if proof.openings.len() != t {
        return Err(PomVerifyError::WrongOpeningCount);
    }
    if final_hash(proof.final_state) != proof.pow_value {
        return Err(PomVerifyError::PowValueMismatch);
    }
    if !le_leq(&proof.pow_value, target) {
        return Err(PomVerifyError::TargetNotMet);
    }
    if !verify_merkle(trace_leaf(seed), 0, &proof.initial_trace_path, &proof.trace_root) {
        return Err(PomVerifyError::BadInitialState);
    }
    if !verify_merkle(trace_leaf(proof.final_state), k as u64, &proof.final_trace_path, &proof.trace_root) {
        return Err(PomVerifyError::BadFinalTracePath);
    }
    let chs = fs_challenges(pre_pow_hash, nonce, &proof.trace_root, &proof.pow_value, t, k);
    for (op, &i) in proof.openings.iter().zip(chs.iter()) {
        let i = i as u64;
        if !verify_merkle(trace_leaf(op.state_before), i, &op.trace_path_before, &proof.trace_root) {
            return Err(PomVerifyError::BadStateBeforePath);
        }
        let off = op.state_before % n_chunks;
        if !verify_merkle(blake(&op.chunk), off, &op.weight_path, r_t) {
            return Err(PomVerifyError::BadWeightPath);
        }
        let state_after = transition(op.state_before, &chunk_to_words(&op.chunk));
        if !verify_merkle(trace_leaf(state_after), i + 1, &op.trace_path_after, &proof.trace_root) {
            return Err(PomVerifyError::BadStateAfterPath);
        }
    }
    Ok(())
}

#[cfg(test)]
mod verify_tests {
    use super::*;

    fn merkle_root(leaves: &[[u8; 32]]) -> [u8; 32] {
        let mut level = leaves.to_vec();
        while level.len() > 1 {
            let mut next = Vec::with_capacity(level.len().div_ceil(2));
            let mut i = 0;
            while i < level.len() {
                let r = if i + 1 < level.len() { level[i + 1] } else { level[i] };
                next.push(hash_pair(&level[i], &r));
                i += 2;
            }
            level = next;
        }
        level[0]
    }

    fn merkle_proof(leaves: &[[u8; 32]], index: usize) -> Vec<[u8; 32]> {
        let mut path = Vec::new();
        let mut level = leaves.to_vec();
        let mut idx = index;
        while level.len() > 1 {
            let sib_idx = if idx & 1 == 0 { idx + 1 } else { idx - 1 };
            let sib = if sib_idx < level.len() { level[sib_idx] } else { level[idx] };
            path.push(sib);
            let mut next = Vec::with_capacity(level.len().div_ceil(2));
            let mut i = 0;
            while i < level.len() {
                let r = if i + 1 < level.len() { level[i + 1] } else { level[i] };
                next.push(hash_pair(&level[i], &r));
                i += 2;
            }
            idx >>= 1;
            level = next;
        }
        path
    }

    fn synth_chunk(off: u64) -> [u64; POM_CHUNK_WORDS] {
        let mut c = [0u64; POM_CHUNK_WORDS];
        for (j, w) in c.iter_mut().enumerate() {
            *w = mix64(off.wrapping_mul(POM_CHUNK_WORDS as u64) + j as u64 + 1);
        }
        c
    }

    fn words_to_bytes(w: &[u64; POM_CHUNK_WORDS]) -> [u8; 32] {
        let mut b = [0u8; 32];
        for (i, wi) in w.iter().enumerate() {
            b[i * 8..i * 8 + 8].copy_from_slice(&wi.to_le_bytes());
        }
        b
    }

    // Inline prover (test-only) to exercise the verifier end-to-end.
    fn build(n_chunks: u64, k: u32, t: usize, pph: &[u8; 32], nonce: u64) -> (PomProof, [u8; 32], u64) {
        let seed = pom_seed_state(nonce);
        let weight_leaves: Vec<[u8; 32]> = (0..n_chunks).map(|o| blake(&words_to_bytes(&synth_chunk(o)))).collect();
        let r_t = merkle_root(&weight_leaves);

        let mut trace = Vec::with_capacity(k as usize + 1);
        let mut state = seed;
        trace.push(state);
        let mut off = state % n_chunks;
        for _ in 0..k {
            state = transition(state, &synth_chunk(off));
            trace.push(state);
            off = state % n_chunks;
        }
        let trace_leaves: Vec<[u8; 32]> = trace.iter().map(|&s| trace_leaf(s)).collect();
        let trace_root = merkle_root(&trace_leaves);
        let final_state = trace[k as usize];
        let pow_value = trace_leaf(final_state); // stand-in final_hash for tests
        let chs = fs_challenges(pph, nonce, &trace_root, &pow_value, t, k);
        let openings = chs
            .iter()
            .map(|&i| {
                let i = i as usize;
                let sb = trace[i];
                let off = sb % n_chunks;
                PomOpening {
                    state_before: sb,
                    chunk: words_to_bytes(&synth_chunk(off)),
                    weight_path: merkle_proof(&weight_leaves, off as usize),
                    trace_path_before: merkle_proof(&trace_leaves, i),
                    trace_path_after: merkle_proof(&trace_leaves, i + 1),
                }
            })
            .collect();
        let proof = PomProof {
            tier: 0,
            trace_root,
            pow_value,
            final_state,
            initial_trace_path: merkle_proof(&trace_leaves, 0),
            final_trace_path: merkle_proof(&trace_leaves, k as usize),
            openings,
        };
        (proof, r_t, seed)
    }

    fn fh(s: u64) -> [u8; 32] {
        trace_leaf(s)
    }

    #[test]
    fn node_verify_roundtrip_and_tamper() {
        let (n, k, t) = (4096u64, 256u32, 32usize);
        let pph = blake(b"pph");
        let nonce = 0xabc;
        let (proof, r_t, seed) = build(n, k, t, &pph, nonce);
        let pass = [0xffu8; 32];

        // honest
        assert_eq!(verify_pom_proof(&pph, nonce, seed, &proof, n, k, t, &r_t, &pass, fh), Ok(()));
        // wrong tier root
        let wrong = blake(b"wrong");
        assert_eq!(verify_pom_proof(&pph, nonce, seed, &proof, n, k, t, &wrong, &pass, fh), Err(PomVerifyError::BadWeightPath));
        // wrong seed
        assert_eq!(
            verify_pom_proof(&pph, nonce, pom_seed_state(nonce ^ 1), &proof, n, k, t, &r_t, &pass, fh),
            Err(PomVerifyError::BadInitialState)
        );
        // tampered chunk
        let mut p2 = proof.clone();
        p2.openings[1].chunk[0] ^= 0xff;
        assert_eq!(verify_pom_proof(&pph, nonce, seed, &p2, n, k, t, &r_t, &pass, fh), Err(PomVerifyError::BadWeightPath));
        // target not met
        assert_eq!(verify_pom_proof(&pph, nonce, seed, &proof, n, k, t, &r_t, &[0u8; 32], fh), Err(PomVerifyError::TargetNotMet));
    }
}
