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
use std::sync::atomic::{AtomicU64, Ordering};

/// DAA score at which `Header::pom_final_state` becomes consensus (hashed into the block
/// hash, cross-checked against `PomProof::final_state`, and used to derive the block level).
/// u64::MAX means "never" (default — disabled until explicitly initialised from
/// `Params::pom_level_activation`). Same startup-init pattern as the PoW salts.
static POM_LEVEL_ACTIVATION_DAA: AtomicU64 = AtomicU64::new(u64::MAX);

/// Called once at startup with the value from `Params::pom_level_activation`.
/// Header hashing (`hashing::header::hash`) has no access to `Params`, so the
/// activation is published through this global exactly like the PoW salt forks.
pub fn init_pom_level_activation(daa_score: u64) {
    POM_LEVEL_ACTIVATION_DAA.store(daa_score, Ordering::Relaxed);
}

/// Whether the PoM block-level fork is active for a block at `daa_score`.
#[inline(always)]
pub fn pom_level_active(daa_score: u64) -> bool {
    daa_score >= POM_LEVEL_ACTIVATION_DAA.load(Ordering::Relaxed)
}

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

/// One step of the H4 recompute-from-chunks walk record: the 32 B weight chunk the walk
/// read at this step, plus its inclusion path under the tier root `R_T`. The chunk index
/// is NOT carried — the verifier derives it (`state % N`) while re-walking, so a prover
/// cannot open a chunk at an index the walk never visited.
#[derive(Clone, Debug, Serialize, Deserialize, BorshSerialize, BorshDeserialize)]
#[serde(rename_all = "camelCase")]
pub struct PomStep {
    /// The 32-byte weight chunk read at this step (candle's exact quantized bytes).
    pub chunk: [u8; 32],
    /// Merkle path proving `leaf(chunk)` is at the derived index `state % N` under `R_T`.
    pub weight_path: Vec<[u8; 32]>,
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
    /// H4 recompute-from-chunks walk record: the K chunks the winning walk read, in step
    /// order, each Merkle-proven under `R_T`. When present the verifier re-walks all K
    /// transitions itself (`verify_pom_proof_v2`) and derives `final_state` — no trace
    /// tree, no spot-check, nothing taken on the prover's word (the 32/256 spot-check this
    /// replaces accepted a forged `final_state` ~88% of the time). `None` on every pre-H4
    /// proof. Trailing field: on the borsh wire a `None` proof is re-encoded through the
    /// `PomProofPreH4` layout (`to_wire_bytes`) so it stays byte-identical for not-yet-updated
    /// peers; in the node-local bincode DB, old records (written before this field existed)
    /// decode via the `PomProofPreH4` fallback in `DbPomProofStore`.
    pub steps_v2: Option<Vec<PomStep>>,
}

/// Exact pre-H4 layout of `PomProof` (no `steps_v2`). Decode-fallback target for legacy
/// wire/DB bytes and byte-identical encode source for proofs without the v2 extension —
/// an updated node re-serving a pre-H4 block must emit the exact bytes a not-yet-updated
/// peer expects. Same mechanism as `UtxoEntryPreH4` / `HeaderWithBlockLevelPreH3`.
#[derive(Clone, Debug, Serialize, Deserialize, BorshSerialize, BorshDeserialize)]
#[serde(rename_all = "camelCase")]
pub struct PomProofPreH4 {
    pub tier: u8,
    pub trace_root: [u8; 32],
    pub pow_value: [u8; 32],
    pub final_state: u64,
    pub initial_trace_path: Vec<[u8; 32]>,
    pub final_trace_path: Vec<[u8; 32]>,
    pub openings: Vec<PomOpening>,
}

impl From<PomProofPreH4> for PomProof {
    fn from(p: PomProofPreH4) -> Self {
        Self {
            tier: p.tier,
            trace_root: p.trace_root,
            pow_value: p.pow_value,
            final_state: p.final_state,
            initial_trace_path: p.initial_trace_path,
            final_trace_path: p.final_trace_path,
            openings: p.openings,
            steps_v2: None,
        }
    }
}

impl From<&PomProof> for PomProofPreH4 {
    fn from(p: &PomProof) -> Self {
        Self {
            tier: p.tier,
            trace_root: p.trace_root,
            pow_value: p.pow_value,
            final_state: p.final_state,
            initial_trace_path: p.initial_trace_path.clone(),
            final_trace_path: p.final_trace_path.clone(),
            openings: p.openings.clone(),
        }
    }
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
        let steps: usize =
            self.steps_v2.as_ref().map_or(0, |steps| steps.iter().map(|s| 32 + path_bytes(&s.weight_path)).sum());
        1 + 32 + 32 + 8 + path_bytes(&self.initial_trace_path) + path_bytes(&self.final_trace_path) + openings + steps
    }

    /// Canonical wire (borsh) encoding, era-exact: a proof without the v2 extension encodes
    /// byte-identically to the pre-H4 layout, so re-served pre-H4 blocks stay readable by
    /// not-yet-updated peers. ALL borsh encode sites MUST use this instead of `borsh::to_vec`.
    pub fn to_wire_bytes(&self) -> Vec<u8> {
        if self.steps_v2.is_none() {
            borsh::to_vec(&PomProofPreH4::from(self)).expect("PomProof borsh serialize")
        } else {
            borsh::to_vec(self).expect("PomProof borsh serialize")
        }
    }

    /// Decode the canonical wire (borsh) encoding, either era. A pre-H4 stream ends exactly
    /// where the `steps_v2` option tag would sit, so the full decode fails cleanly and the
    /// legacy fallback applies; a v2 stream fails the legacy decode (trailing bytes), so the
    /// two layouts can never be confused. ALL borsh decode sites MUST use this.
    pub fn from_wire_bytes(bytes: &[u8]) -> std::io::Result<Self> {
        borsh::from_slice::<PomProof>(bytes).or_else(|_| borsh::from_slice::<PomProofPreH4>(bytes).map(PomProof::from))
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
    // --- H4 recompute-from-chunks (verify_pom_proof_v2) ---
    /// Post-H4 proof without the `steps_v2` walk record.
    MissingSteps,
    /// `steps_v2` length differs from the walk length `K`.
    WrongStepCount,
    /// Post-H4 proof still carries legacy trace-tree fields (must be canonically empty).
    NonCanonicalLegacyFields,
    /// The claimed `final_state` differs from the state derived by re-walking the chunks.
    FinalStateMismatch,
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

/// H3 domain salt applied to the pre_pow_hash words feeding both PoM folds at/after
/// `pom_level_activation`. Forced-update mechanism (same spirit as the kHeavyHash matrix
/// salts): every walk trajectory and pow value changes at the gate, so pre-H3 binaries —
/// even ones patched to echo `pom_final_state` — produce proofs that verify false.
/// Derivation: sha256("keryx-h3-pom-pph-salt") read as 4 little-endian u64 words.
pub const POM_H3_PPH_SALT: [u64; 4] = [0x7C99D381176D4EC4, 0xC2E28E3E28118C36, 0xD496CE1B129B76CA, 0x47CF0979FA580BCE];

#[inline]
fn pph_words_h3(pre_pow_hash: &[u8; 32]) -> [u64; 4] {
    let mut w = pph_words(pre_pow_hash);
    for (wi, si) in w.iter_mut().zip(POM_H3_PPH_SALT.iter()) {
        *wi ^= si;
    }
    w
}

#[inline]
fn pom_block_seed_from_words(p: &[u64; 4], timestamp: u64, nonce: u64) -> u64 {
    let mut s = mix64(nonce ^ 0x4B65727978531);
    s = mix64(s ^ timestamp);
    s = mix64(s ^ p[0]);
    s = mix64(s ^ p[1]);
    s = mix64(s ^ p[2]);
    s = mix64(s ^ p[3]);
    s
}

/// Canonical PoM block seed = initial walk state. mix64-fold of (nonce, time, pre_pow_hash).
/// Pre-H3 era only. BYTE-IDENTICAL to the miner kernel `pom_mine.cu::pom_seed_fold` and `src/pom.rs`.
pub fn pom_block_seed(pre_pow_hash: &[u8; 32], timestamp: u64, nonce: u64) -> u64 {
    pom_block_seed_from_words(&pph_words(pre_pow_hash), timestamp, nonce)
}

/// H3-era block seed: same fold over the salted pre_pow_hash words. The miner kernel is
/// unchanged — it folds whatever pph words the host feeds it (the host salts them post-gate).
pub fn pom_block_seed_h3(pre_pow_hash: &[u8; 32], timestamp: u64, nonce: u64) -> u64 {
    pom_block_seed_from_words(&pph_words_h3(pre_pow_hash), timestamp, nonce)
}

#[inline]
fn pom_pow_value_from_words(final_state: u64, p: &[u64; 4]) -> [u8; 32] {
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

/// Canonical PoM pow value (256-bit, little-endian) compared against the target. mix64-fold of
/// (final_state, pre_pow_hash) — the memory-hardness is already paid by the K data-dependent
/// reads. Pre-H3 era only. BYTE-IDENTICAL to the miner kernel `pom_mine.cu::pom_pow_fold` and `src/pom.rs`.
pub fn pom_pow_value(final_state: u64, pre_pow_hash: &[u8; 32]) -> [u8; 32] {
    pom_pow_value_from_words(final_state, &pph_words(pre_pow_hash))
}

/// H3-era pow value: same fold over the salted pre_pow_hash words.
pub fn pom_pow_value_h3(final_state: u64, pre_pow_hash: &[u8; 32]) -> [u8; 32] {
    pom_pow_value_from_words(final_state, &pph_words_h3(pre_pow_hash))
}

/// Pre-H5 possession transition (FROZEN — validates all blocks below `h5_activation`). The 4 chunk
/// words are XOR-folded into a single 64-bit accumulator before one `mix64`, so only their XOR
/// (8 bytes) is load-bearing — a miner could hold a 4×-smaller possession table. Kept verbatim for
/// historical re-derivation; never change it.
#[inline]
fn transition_v1(state: u64, chunk: &[u64; POM_CHUNK_WORDS]) -> u64 {
    let mut h = state;
    for &w in chunk.iter() {
        h ^= w;
    }
    mix64(h)
}

/// H5 possession transition (active at/after `h5_activation`). `mix64` is chained through each of
/// the 4 chunk words, so all 32 bytes are load-bearing and order-dependent — the v1 XOR-fold
/// shortcut is closed and a miner must hold the full chunk. Same weights, same `R_T` (the change is
/// in the walk, not the weight commitment); only the walk output `final_state` differs per era.
#[inline]
fn transition_v2(state: u64, chunk: &[u64; POM_CHUNK_WORDS]) -> u64 {
    let mut h = state;
    for &w in chunk.iter() {
        h = mix64(h ^ w);
    }
    h
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
    // Bound the path to the u64 index bit-width: any Merkle tree addressable by a u64 index has at
    // most 64 levels, so every honest inclusion path is <= 64 siblings. Rejecting longer paths early
    // caps verification work — a malicious proof cannot force an unbounded hashing loop — and changes
    // no accept decision, since an over-length path never hashes to the correct root below anyway.
    if path.len() > 64 {
        return false;
    }
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
        // Pre-H4 spot-check path — only ever runs for pre-H4 (hence pre-H5) blocks, always walk v1.
        let state_after = transition_v1(op.state_before, &chunk_to_words(&op.chunk));
        if !verify_merkle(trace_leaf(state_after), i + 1, &op.trace_path_after, &proof.trace_root) {
            return Err(PomVerifyError::BadStateAfterPath);
        }
    }
    Ok(())
}

/// H4 verifier — recompute-from-chunks. The prover supplies the K chunks its winning walk
/// read (in step order), each Merkle-proven under the tier root `r_t`; the verifier re-walks
/// every transition itself, deriving each chunk index (`state % N`) and the `final_state`.
/// No transition is taken on the prover's word — this closes the soundness hole of the 32/256
/// spot-check (`verify_pom_proof`), where the 224 unopened transitions let a single honest
/// walk back a hash-grinded `final_state` with ~88% acceptance.
///
/// Legacy trace-tree fields must be canonically empty (`trace_root == 0`, empty paths and
/// openings): one canonical encoding per post-H4 proof, no dead 47 KB payload. `final_state`
/// and `pow_value` stay meaningful but are CHECKED against the derived walk, never trusted.
/// The caller still cross-checks `final_state` against `Header::pom_final_state` (H3 rule).
pub fn verify_pom_proof_v2<Hf: Fn(u64) -> [u8; 32]>(
    seed: u64,
    proof: &PomProof,
    n_chunks: u64,
    k: u32,
    r_t: &[u8; 32],
    target: &[u8; 32],
    final_hash: Hf,
    walk_v2: bool,
) -> Result<(), PomVerifyError> {
    let steps = proof.steps_v2.as_ref().ok_or(PomVerifyError::MissingSteps)?;
    if steps.len() != k as usize {
        return Err(PomVerifyError::WrongStepCount);
    }
    if proof.trace_root != [0u8; 32]
        || !proof.initial_trace_path.is_empty()
        || !proof.final_trace_path.is_empty()
        || !proof.openings.is_empty()
    {
        return Err(PomVerifyError::NonCanonicalLegacyFields);
    }
    // H5 era selection: `walk_v2` (from `h5_activation` on the block's daa_score) picks the
    // non-foldable mix64-chained transition; pre-H5 blocks re-walk with the frozen v1 fold.
    let transition = if walk_v2 { transition_v2 } else { transition_v1 };
    let mut state = seed;
    for step in steps.iter() {
        let off = state % n_chunks;
        if !verify_merkle(blake(&step.chunk), off, &step.weight_path, r_t) {
            return Err(PomVerifyError::BadWeightPath);
        }
        state = transition(state, &chunk_to_words(&step.chunk));
    }
    if state != proof.final_state {
        return Err(PomVerifyError::FinalStateMismatch);
    }
    let pow_value = final_hash(state);
    if pow_value != proof.pow_value {
        return Err(PomVerifyError::PowValueMismatch);
    }
    if !le_leq(&pow_value, target) {
        return Err(PomVerifyError::TargetNotMet);
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
            state = transition_v1(state, &synth_chunk(off));
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
            steps_v2: None,
        };
        (proof, r_t, seed)
    }

    // Inline v2 prover (test-only): walk once recording the chunks, open every step under R_T.
    // `walk_v2` selects the era transition so both the frozen v1 fold and the H5 mix64 chain are
    // exercised end-to-end against the verifier.
    fn build_v2(n_chunks: u64, k: u32, seed: u64, walk_v2: bool) -> (PomProof, [u8; 32]) {
        let weight_leaves: Vec<[u8; 32]> = (0..n_chunks).map(|o| blake(&words_to_bytes(&synth_chunk(o)))).collect();
        let r_t = merkle_root(&weight_leaves);

        let transition = if walk_v2 { transition_v2 } else { transition_v1 };
        let mut steps = Vec::with_capacity(k as usize);
        let mut state = seed;
        for _ in 0..k {
            let off = state % n_chunks;
            steps.push(PomStep { chunk: words_to_bytes(&synth_chunk(off)), weight_path: merkle_proof(&weight_leaves, off as usize) });
            state = transition(state, &synth_chunk(off));
        }
        let final_state = state;
        let proof = PomProof {
            tier: 0,
            trace_root: [0u8; 32],
            pow_value: trace_leaf(final_state), // stand-in final_hash for tests
            final_state,
            initial_trace_path: vec![],
            final_trace_path: vec![],
            openings: vec![],
            steps_v2: Some(steps),
        };
        (proof, r_t)
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

    #[test]
    fn v2_verify_roundtrip_and_tamper() {
        let (n, k) = (4096u64, 256u32);
        let seed = pom_seed_state(0xabc);
        let (proof, r_t) = build_v2(n, k, seed, false);
        let pass = [0xffu8; 32];

        // honest
        assert_eq!(verify_pom_proof_v2(seed, &proof, n, k, &r_t, &pass, fh, false), Ok(()));
        // wrong tier root
        assert_eq!(verify_pom_proof_v2(seed, &proof, n, k, &blake(b"wrong"), &pass, fh, false), Err(PomVerifyError::BadWeightPath));
        // wrong seed: the very first derived index no longer matches the opened path
        assert_eq!(verify_pom_proof_v2(seed ^ 1, &proof, n, k, &r_t, &pass, fh, false), Err(PomVerifyError::BadWeightPath));
        // tampered chunk
        let mut p2 = proof.clone();
        p2.steps_v2.as_mut().unwrap()[100].chunk[0] ^= 0xff;
        assert_eq!(verify_pom_proof_v2(seed, &p2, n, k, &r_t, &pass, fh, false), Err(PomVerifyError::BadWeightPath));
        // truncated walk
        let mut p3 = proof.clone();
        p3.steps_v2.as_mut().unwrap().pop();
        assert_eq!(verify_pom_proof_v2(seed, &p3, n, k, &r_t, &pass, fh, false), Err(PomVerifyError::WrongStepCount));
        // missing steps entirely
        let mut p4 = proof.clone();
        p4.steps_v2 = None;
        assert_eq!(verify_pom_proof_v2(seed, &p4, n, k, &r_t, &pass, fh, false), Err(PomVerifyError::MissingSteps));
        // legacy trace-tree fields must stay canonically empty
        let mut p5 = proof.clone();
        p5.trace_root = [1u8; 32];
        assert_eq!(verify_pom_proof_v2(seed, &p5, n, k, &r_t, &pass, fh, false), Err(PomVerifyError::NonCanonicalLegacyFields));
        let mut p6 = proof.clone();
        p6.openings.push(PomOpening {
            state_before: 0,
            chunk: [0u8; 32],
            weight_path: vec![],
            trace_path_before: vec![],
            trace_path_after: vec![],
        });
        assert_eq!(verify_pom_proof_v2(seed, &p6, n, k, &r_t, &pass, fh, false), Err(PomVerifyError::NonCanonicalLegacyFields));
        // target not met
        assert_eq!(verify_pom_proof_v2(seed, &proof, n, k, &r_t, &[0u8; 32], fh, false), Err(PomVerifyError::TargetNotMet));
    }

    /// The spot-check exploit scenario, replayed against v2: keep one honest walk record but
    /// grind `final_state`/`pow_value` in pure hashing. Under the 32/256 spot-check this was
    /// accepted ~88% of the time; under v2 the verifier derives `final_state` from the chunks
    /// itself, so EVERY forged value is rejected — no probabilistic escape hatch.
    #[test]
    fn v2_forged_final_state_always_rejected() {
        let (n, k) = (4096u64, 256u32);
        let seed = pom_seed_state(0xdead);
        let (proof, r_t) = build_v2(n, k, seed, false);
        let pass = [0xffu8; 32];
        assert_eq!(verify_pom_proof_v2(seed, &proof, n, k, &r_t, &pass, fh, false), Ok(()));

        for grind in 0..1000u64 {
            let forged_state = mix64(proof.final_state ^ grind.wrapping_add(1));
            let mut forged = proof.clone();
            forged.final_state = forged_state;
            forged.pow_value = fh(forged_state); // self-consistent fold, exactly like the exploit
            assert_eq!(
                verify_pom_proof_v2(seed, &forged, n, k, &r_t, &pass, fh, false),
                Err(PomVerifyError::FinalStateMismatch),
                "forged final_state accepted at grind {grind}"
            );
        }
    }

    /// H5 era-gating of the walk: the frozen v1 fold and the mix64-chained v2 walk produce
    /// different `final_state`s from the same weights/seed, and a proof built for one era is
    /// rejected when re-walked under the other (the derived chunk indices diverge). This is exactly
    /// how a block validated on the wrong side of `h5_activation` fails.
    #[test]
    fn v2_walk_era_gating() {
        let (n, k) = (4096u64, 256u32);
        let seed = pom_seed_state(0xf00d);
        let pass = [0xffu8; 32];

        let (v1_proof, r_t) = build_v2(n, k, seed, false);
        let (v2_proof, _) = build_v2(n, k, seed, true);

        // Same weights + seed, different walk → different derived final_state.
        assert_ne!(v1_proof.final_state, v2_proof.final_state);

        // Each proof verifies only under its own era.
        assert_eq!(verify_pom_proof_v2(seed, &v1_proof, n, k, &r_t, &pass, fh, false), Ok(()));
        assert_eq!(verify_pom_proof_v2(seed, &v2_proof, n, k, &r_t, &pass, fh, true), Ok(()));

        // Cross-era: re-walking with the wrong transition diverges → rejected.
        assert!(verify_pom_proof_v2(seed, &v2_proof, n, k, &r_t, &pass, fh, false).is_err());
        assert!(verify_pom_proof_v2(seed, &v1_proof, n, k, &r_t, &pass, fh, true).is_err());
    }

    /// Era-exact wire encoding: a proof without `steps_v2` MUST encode byte-identically to
    /// the pre-H4 layout (not-yet-updated peers must keep decoding re-served pre-H4 blocks),
    /// and both eras must round-trip through the canonical decode helper.
    #[test]
    fn wire_bytes_era_exact_and_roundtrip() {
        let (n, k, t) = (4096u64, 256u32, 32usize);
        let pph = blake(b"wire-pph");
        let (v1, _, _) = build(n, k, t, &pph, 0xabc);
        assert!(v1.steps_v2.is_none());

        // v1 wire bytes == exact legacy encoding (what a pre-H4 binary emits and expects).
        let legacy = borsh::to_vec(&PomProofPreH4::from(&v1)).unwrap();
        assert_eq!(v1.to_wire_bytes(), legacy, "pre-H4 proof must stay byte-identical on the wire");

        // Legacy bytes decode back with steps_v2 == None.
        let back = PomProof::from_wire_bytes(&legacy).unwrap();
        assert!(back.steps_v2.is_none());
        assert_eq!(back.final_state, v1.final_state);
        assert_eq!(back.openings.len(), v1.openings.len());

        // v2 round-trips through the same helpers.
        let seed = pom_seed_state(0xabc);
        let (v2, r_t) = build_v2(n, k, seed, false);
        let bytes = v2.to_wire_bytes();
        let back2 = PomProof::from_wire_bytes(&bytes).unwrap();
        assert_eq!(back2.steps_v2.as_ref().unwrap().len(), k as usize);
        assert_eq!(verify_pom_proof_v2(seed, &back2, n, k, &r_t, &[0xffu8; 32], fh, false), Ok(()));

        // A truncated / garbage stream fails both decodes.
        assert!(PomProof::from_wire_bytes(&bytes[..bytes.len() - 1]).is_err());
    }

    /// Bincode decode fallback the `DbPomProofStore` relies on: a record written by a pre-H4 binary
    /// is the `PomProofPreH4` positional layout; the grown `PomProof` under-flows on it, so decode
    /// must fall back to the old layout and backfill `steps_v2 = None` — never a silent mis-decode.
    #[test]
    fn bincode_pre_h4_record_decodes_via_fallback() {
        let (n, k, t) = (4096u64, 256u32, 32usize);
        let pph = blake(b"db-pph");
        let (v1, _, _) = build(n, k, t, &pph, 0xabc);

        // Old binary wrote the 7-field layout.
        let old_bytes = bincode::serialize(&PomProofPreH4::from(&v1)).unwrap();
        // New `PomProof` (8 fields) under-flows on it...
        assert!(bincode::deserialize::<PomProof>(&old_bytes).is_err());
        // ...so the store's fallback kicks in and backfills steps_v2 = None.
        let recovered: PomProof = bincode::deserialize::<PomProofPreH4>(&old_bytes).unwrap().into();
        assert!(recovered.steps_v2.is_none());
        assert_eq!(recovered.final_state, v1.final_state);
        assert_eq!(recovered.openings.len(), v1.openings.len());

        // A v2 record decodes as PomProof directly (primary path, no fallback).
        let seed = pom_seed_state(0xabc);
        let (v2, _) = build_v2(n, k, seed, false);
        let v2_bytes = bincode::serialize(&v2).unwrap();
        let back = bincode::deserialize::<PomProof>(&v2_bytes).unwrap();
        assert_eq!(back.steps_v2.as_ref().unwrap().len(), k as usize);
    }
}
