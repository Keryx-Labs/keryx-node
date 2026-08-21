//! Proof-of-Model v4. The verifier re-walks and derives `final_state`; the proof carries the K
//! tiles read plus one Merkle range proof per tile against the per-tier `R_T` (unchanged,
//! `pom-rt-builder`). A tile is `POM_V4_TILE_CHUNKS` consecutive canonical chunks, opened as an
//! aligned subtree of depth `TILE_SUBTREE_DEPTH`. The transition is integer-only and the miner
//! GPU kernel must reproduce this module byte-for-byte.

use crate::pom::{blake, hash_pair, mix64, verify_merkle};
use crate::pom_v3::{fold64, rho8, rho_tweak, snippet_fold};
use borsh::{BorshDeserialize, BorshSerialize};
use serde::{Deserialize, Serialize};

/// State dimension: the walk state is a D x D int8 matrix (1 KB at D = 32).
pub const POM_V4_D: usize = 32;
/// Walk length in steps (tiles multiplied per nonce).
pub const POM_V4_K: usize = 256;
/// Canonical R_T chunk size (bytes) — pinned by `pom-rt-builder`, do not change.
pub const POM_V4_CHUNK_BYTES: usize = 32;
/// Tile size: `POM_V4_TILE_CHUNKS` canonical chunks = 1 KB (= D x D).
pub const POM_V4_TILE_BYTES: usize = POM_V4_D * POM_V4_D;
pub const POM_V4_TILE_CHUNKS: u64 = (POM_V4_TILE_BYTES / POM_V4_CHUNK_BYTES) as u64;
/// Offset-chain snippet: the first 32 bytes (= first canonical chunk) of a tile.
pub const POM_V4_SNIPPET_BYTES: usize = 32;
/// Aligned-subtree depth of a whole tile (`log2(POM_V4_TILE_CHUNKS)` = log2(32) = 5).
const TILE_SUBTREE_DEPTH: u32 = 5;
const TILE_LEAVES: usize = POM_V4_TILE_BYTES / POM_V4_CHUNK_BYTES;

/// Domain salts. Derivation: sha256(label), first 8 bytes read as little-endian u64
/// (same convention as the v3 / H3 / H5 salts).
/// sha256("keryx-v4-s0-row-salt")
pub const POM_V4_S0_ROW_SALT: u64 = 0x03421325594C3C51;
/// sha256("keryx-v4-offset-first-salt")
pub const POM_V4_OFFSET_FIRST_SALT: u64 = 0x6D1CCF96AC4D76F9;
/// sha256("keryx-v4-offset-step-salt")
pub const POM_V4_OFFSET_STEP_SALT: u64 = 0x89050E78D34609EF;

/// One walk state: a D x D int8 matrix, row-major (`state[row * D + col]`).
pub type V4State = Vec<u8>; // POM_V4_D * POM_V4_D bytes

/// Merkle range proof for one tile: the path from the tile's aligned-subtree root (folded from
/// the tile's own 32 chunk leaves at depth `TILE_SUBTREE_DEPTH`) up to `R_T`. The tile bytes are
/// the leaves, carried in `PomProofV4::tiles`, so no separate leaf field.
#[derive(Clone, Debug, Serialize, Deserialize, BorshSerialize, BorshDeserialize)]
#[serde(rename_all = "camelCase")]
pub struct PomV4RangeProof {
    pub path: Vec<[u8; 32]>,
}

/// Post-PoW witness for the v4 walk, carried at the Block level like `PomProof`.
#[derive(Clone, Debug, Serialize, Deserialize, BorshSerialize, BorshDeserialize)]
#[serde(rename_all = "camelCase")]
pub struct PomProofV4 {
    /// Tier index — selects `R_T` and the reward bracket (authenticated by the proof itself:
    /// range proofs only verify against the claimed tier's root).
    pub tier: u8,
    /// The K tiles read during the walk, in step order. `tiles[t-1]` is `POM_V4_TILE_BYTES`
    /// bytes; its first 32 bytes are the offset-chain snippet for step `t`.
    pub tiles: Vec<Vec<u8>>,
    /// One Merkle range proof per tile, against `R_T[tier]`.
    pub merkle: Vec<PomV4RangeProof>,
}

impl PomProofV4 {
    /// Rough wire size (cache accounting).
    pub fn approx_bytes(&self) -> usize {
        let tiles: usize = self.tiles.iter().map(|t| t.len()).sum();
        let paths: usize = self.merkle.iter().map(|m| m.path.len() * 32).sum();
        1 + tiles + paths
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PomV4VerifyError {
    /// `tiles` length != K or `merkle` length != K.
    WrongShape,
    /// A tile's byte length is not `POM_V4_TILE_BYTES`.
    WrongTileShape,
    /// A tile fails its range proof against `R_T` at the walk-derived offset.
    BadTilePath,
    /// The blob is too small to hold a single tile.
    BlobTooSmall,
    // --- container-level (verify_pom_proof_v4_container) ---
    /// Post-v4-gate proof without the `v4` witness.
    MissingV4,
    /// Post-v4-gate proof still carries legacy or v3 content (must be canonically empty).
    NonCanonicalLegacyFields,
    /// Container `tier` differs from the v4 witness `tier`.
    TierMismatch,
    /// Container `final_state` differs from the verifier-derived walk result.
    FinalStateMismatch,
    /// Container `pow_value` differs from the era pow fold of `final_state`.
    PowValueMismatch,
    /// The era pow fold of `final_state` does not meet the tier target.
    TargetNotMet,
}

// --- walk primitives (the miner kernel must reproduce these byte-for-byte) ---

/// Signed int8 dot product of a state row and a tile column (both D bytes).
/// No overflow: |acc| <= D * 128 * 128 < 2^19 at D = 32.
#[inline]
pub fn v4_dot_i8(row: &[u8], col: &[u8]) -> i32 {
    debug_assert_eq!(row.len(), POM_V4_D);
    debug_assert_eq!(col.len(), POM_V4_D);
    row.iter().zip(col.iter()).map(|(&a, &b)| (a as i8 as i32) * (b as i8 as i32)).sum()
}

/// S_0 from the canonical block seed: row r is a mix64 keystream, 4 little-endian bytes per squeeze.
pub fn v4_initial_state(seed: u64) -> V4State {
    let mut s = vec![0u8; POM_V4_D * POM_V4_D];
    for r in 0..POM_V4_D {
        let mut h = mix64(seed ^ POM_V4_S0_ROW_SALT.wrapping_add(r as u64));
        for k4 in 0..POM_V4_D / 4 {
            h = mix64(h);
            s[r * POM_V4_D + k4 * 4..r * POM_V4_D + k4 * 4 + 4].copy_from_slice(&(h as u32).to_le_bytes());
        }
    }
    s
}

/// Tile offset for step 1 (no previous tile: seed-only).
#[inline]
pub fn v4_first_offset(seed: u64, n_tiles: u64) -> u64 {
    mix64(seed ^ POM_V4_OFFSET_FIRST_SALT) % n_tiles
}

/// Tile offset for step t+1 from the snippet of the tile read at step t (t in 1..=K-1).
#[inline]
pub fn v4_next_offset(seed: u64, step: u64, snippet: &[u8; POM_V4_SNIPPET_BYTES], n_tiles: u64) -> u64 {
    mix64(seed ^ (step + 1).wrapping_mul(POM_V4_OFFSET_STEP_SALT) ^ snippet_fold(snippet)) % n_tiles
}

/// One walk transition written into a caller-provided state buffer. The verifier uses this
/// allocation-free form for all K steps; `v4_transition` remains the simple reference wrapper.
#[inline]
fn v4_transition_into(state: &[u8], tile: &[u8], step: u32, next: &mut [u8]) {
    debug_assert_eq!(state.len(), POM_V4_TILE_BYTES);
    debug_assert_eq!(tile.len(), POM_V4_TILE_BYTES);
    debug_assert_eq!(next.len(), POM_V4_TILE_BYTES);
    for x in 0..POM_V4_D {
        let row = &state[x * POM_V4_D..(x + 1) * POM_V4_D];
        for j in 0..POM_V4_D {
            let col = &tile[j * POM_V4_D..(j + 1) * POM_V4_D];
            next[x * POM_V4_D + j] = rho8(v4_dot_i8(row, col), rho_tweak(step, x as u32, j as u32));
        }
    }
}

/// One walk transition: S_t = rho(S_{t-1} x tile), entrywise, column j of the tile being
/// `tile[j*D .. (j+1)*D]`.
pub fn v4_transition(state: &V4State, tile: &[u8], step: u32) -> V4State {
    let mut next = vec![0u8; POM_V4_TILE_BYTES];
    v4_transition_into(state, tile, step, &mut next);
    next
}

/// Merkle root over the D rows of a state (leaf = blake3(row); D = 32 is a power of two so the
/// tree is complete, no duplicate-last). The fixed scratch array avoids heap allocation in the
/// verifier's final-state fold.
pub fn v4_state_root(state: &V4State) -> [u8; 32] {
    debug_assert_eq!(state.len(), POM_V4_TILE_BYTES);
    let mut level = [[0u8; 32]; POM_V4_D];
    for (r, slot) in level.iter_mut().enumerate() {
        *slot = blake(&state[r * POM_V4_D..(r + 1) * POM_V4_D]);
    }
    let mut width = POM_V4_D;
    while width > 1 {
        for i in 0..width / 2 {
            let left = level[i * 2];
            let right = level[i * 2 + 1];
            level[i] = hash_pair(&left, &right);
        }
        width /= 2;
    }
    level[0]
}

/// Number of whole tiles in a canonical blob (`n_chunks` 32 B chunks). The sub-tile remainder is
/// never walked (covered by `R_T`; only completeness of aligned subtrees matters).
#[inline]
pub fn v4_n_tiles(n_chunks: u64) -> u64 {
    n_chunks / POM_V4_TILE_CHUNKS
}

/// Fold a full leaf level up one level (duplicate-last on an odd count — matches `pom-rt-builder`).
fn fold_level(level: &[[u8; 32]]) -> Vec<[u8; 32]> {
    level.chunks(2).map(|p| if p.len() == 2 { hash_pair(&p[0], &p[1]) } else { hash_pair(&p[0], &p[0]) }).collect()
}

/// Fold a whole tile's `POM_V4_TILE_CHUNKS` chunk leaves into its aligned subtree root
/// (depth `TILE_SUBTREE_DEPTH`, always complete since `POM_V4_TILE_CHUNKS` is a power of two).
/// Uses fixed scratch storage because this function runs 256 times for every v4 verification.
pub fn v4_tile_subtree_root(tile: &[u8]) -> [u8; 32] {
    debug_assert_eq!(tile.len(), POM_V4_TILE_BYTES);
    debug_assert_eq!(TILE_LEAVES, 1usize << TILE_SUBTREE_DEPTH);
    let mut level = [[0u8; 32]; TILE_LEAVES];
    for (slot, chunk) in level.iter_mut().zip(tile.chunks_exact(POM_V4_CHUNK_BYTES)) {
        *slot = blake(chunk);
    }
    let mut width = TILE_LEAVES;
    while width > 1 {
        for i in 0..width / 2 {
            let left = level[i * 2];
            let right = level[i * 2 + 1];
            level[i] = hash_pair(&left, &right);
        }
        width /= 2;
    }
    level[0]
}

// --- verifier (weightless; this is what the node runs per block, live only) ---

/// Re-walk and verify a v4 proof against the tier root `r_t` and its canonical chunk count. On
/// success returns the derived `final_state`; the caller pins it against the header and checks
/// the target (container wiring in `check_pom_proof`).
pub fn verify_pom_proof_v4(
    seed: u64,
    proof: &PomProofV4,
    r_t: &[u8; 32],
    n_chunks: u64,
) -> Result<u64, PomV4VerifyError> {
    if proof.tiles.len() != POM_V4_K || proof.merkle.len() != POM_V4_K {
        return Err(PomV4VerifyError::WrongShape);
    }
    let n_tiles = v4_n_tiles(n_chunks);
    if n_tiles == 0 {
        return Err(PomV4VerifyError::BlobTooSmall);
    }

    // Two 1 KiB buffers are reused for the full walk. The previous implementation allocated a
    // fresh Vec on every one of the 256 transitions, which amplified allocator pressure when many
    // recent IBD/relay proofs were verified concurrently.
    let mut state = v4_initial_state(seed);
    let mut next = vec![0u8; POM_V4_TILE_BYTES];
    let mut off = v4_first_offset(seed, n_tiles);
    for step in 1..=POM_V4_K {
        let tile = &proof.tiles[step - 1];
        if tile.len() != POM_V4_TILE_BYTES {
            return Err(PomV4VerifyError::WrongTileShape);
        }
        if !verify_merkle(v4_tile_subtree_root(tile), off, &proof.merkle[step - 1].path, r_t) {
            return Err(PomV4VerifyError::BadTilePath);
        }
        v4_transition_into(&state, tile, step as u32, &mut next);
        std::mem::swap(&mut state, &mut next);
        if step < POM_V4_K {
            let snippet: [u8; POM_V4_SNIPPET_BYTES] = tile[..POM_V4_SNIPPET_BYTES].try_into().unwrap();
            off = v4_next_offset(seed, step as u64, &snippet, n_tiles);
        }
    }
    Ok(fold64(&v4_state_root(&state)))
}

/// Full v4 validation of a container `PomProof`, mirroring the v3 container's role in
/// `check_pom_proof`: canonical-shape checks on the legacy/v3 fields, the header-pin
/// consistency (`final_state`/`pow_value`), the tier target, and the re-walk.
#[allow(clippy::too_many_arguments)]
pub fn verify_pom_proof_v4_container<Hf: Fn(u64) -> [u8; 32]>(
    seed: u64,
    proof: &crate::pom::PomProof,
    n_chunks: u64,
    r_t: &[u8; 32],
    target: &[u8; 32],
    final_hash: Hf,
) -> Result<(), PomV4VerifyError> {
    let v4 = proof.v4.as_ref().ok_or(PomV4VerifyError::MissingV4)?;
    if proof.trace_root != [0u8; 32]
        || !proof.initial_trace_path.is_empty()
        || !proof.final_trace_path.is_empty()
        || !proof.openings.is_empty()
        || proof.steps_v2.is_some()
        || proof.v3.is_some()
    {
        return Err(PomV4VerifyError::NonCanonicalLegacyFields);
    }
    if proof.tier != v4.tier {
        return Err(PomV4VerifyError::TierMismatch);
    }
    let derived = verify_pom_proof_v4(seed, v4, r_t, n_chunks)?;
    if proof.final_state != derived {
        return Err(PomV4VerifyError::FinalStateMismatch);
    }
    let pv = final_hash(proof.final_state);
    if pv != proof.pow_value {
        return Err(PomV4VerifyError::PowValueMismatch);
    }
    if !crate::pom::le_leq(&pv, target) {
        return Err(PomV4VerifyError::TargetNotMet);
    }
    Ok(())
}

// --- reference prover (tests; the production miner builds these bytes on GPU) ---

/// Full walk trace for one nonce over an in-memory canonical blob.
pub struct V4Walk {
    pub states: Vec<V4State>,
    pub offsets: Vec<u64>,
    pub snippets: Vec<[u8; 32]>,
}

/// Reference walk over an in-memory canonical byte stream.
pub fn v4_walk(seed: u64, blob: &[u8]) -> Result<V4Walk, PomV4VerifyError> {
    let n_tiles = v4_n_tiles((blob.len() / POM_V4_CHUNK_BYTES) as u64);
    if n_tiles == 0 {
        return Err(PomV4VerifyError::BlobTooSmall);
    }
    let mut states = Vec::with_capacity(POM_V4_K + 1);
    let mut offsets = Vec::with_capacity(POM_V4_K);
    let mut snippets = Vec::with_capacity(POM_V4_K);
    states.push(v4_initial_state(seed));
    let mut off = v4_first_offset(seed, n_tiles);
    for step in 1..=POM_V4_K as u32 {
        let tile = &blob[(off as usize) * POM_V4_TILE_BYTES..(off as usize + 1) * POM_V4_TILE_BYTES];
        let snippet: [u8; 32] = tile[..POM_V4_SNIPPET_BYTES].try_into().unwrap();
        offsets.push(off);
        snippets.push(snippet);
        states.push(v4_transition(states.last().unwrap(), tile, step));
        if (step as usize) < POM_V4_K {
            off = v4_next_offset(seed, step as u64, &snippet, n_tiles);
        }
    }
    Ok(V4Walk { states, offsets, snippets })
}

/// Merkle path from an aligned tile subtree (index = tile offset) up to `R_T`.
fn v4_tile_path(leaf_hashes: &[[u8; 32]], tile_offset: u64) -> Vec<[u8; 32]> {
    let mut level: Vec<[u8; 32]> = leaf_hashes.to_vec();
    for _ in 0..TILE_SUBTREE_DEPTH {
        level = fold_level(&level);
    }
    let mut idx = tile_offset;
    let mut path = Vec::new();
    while level.len() > 1 {
        let sib = if (idx ^ 1) < level.len() as u64 { level[(idx ^ 1) as usize] } else { level[idx as usize] };
        path.push(sib);
        level = fold_level(&level);
        idx >>= 1;
    }
    path
}

/// Reference prover: walk, collect the read tiles and their range proofs.
pub fn v4_prove(seed: u64, tier: u8, blob: &[u8], leaf_hashes: &[[u8; 32]]) -> Result<PomProofV4, PomV4VerifyError> {
    let walk = v4_walk(seed, blob)?;
    let mut tiles = Vec::with_capacity(POM_V4_K);
    let mut merkle = Vec::with_capacity(POM_V4_K);
    for &off in &walk.offsets {
        let tile = blob[(off as usize) * POM_V4_TILE_BYTES..(off as usize + 1) * POM_V4_TILE_BYTES].to_vec();
        tiles.push(tile);
        merkle.push(PomV4RangeProof { path: v4_tile_path(leaf_hashes, off) });
    }
    Ok(PomProofV4 { tier, tiles, merkle })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random blob of `n_tiles` whole tiles (+ a ragged tail), with its full
    /// 32 B-chunk leaf level.
    fn test_blob(n_tiles: usize) -> (Vec<u8>, Vec<[u8; 32]>) {
        let tail_chunks = 3usize; // deliberately not a multiple of POM_V4_TILE_CHUNKS
        let n_bytes = n_tiles * POM_V4_TILE_BYTES + tail_chunks * POM_V4_CHUNK_BYTES;
        let mut blob = vec![0u8; n_bytes];
        let mut h = 0xDEADBEEFu64;
        for b in blob.iter_mut() {
            h = mix64(h);
            *b = h as u8;
        }
        let leaves: Vec<[u8; 32]> = blob.chunks(POM_V4_CHUNK_BYTES).map(blake).collect();
        (blob, leaves)
    }

    fn r_t_root(leaf_hashes: &[[u8; 32]]) -> [u8; 32] {
        let mut level = leaf_hashes.to_vec();
        while level.len() > 1 {
            level = fold_level(&level);
        }
        level[0]
    }

    fn reference_state_root(state: &V4State) -> [u8; 32] {
        let mut level: Vec<[u8; 32]> =
            (0..POM_V4_D).map(|r| blake(&state[r * POM_V4_D..(r + 1) * POM_V4_D])).collect();
        while level.len() > 1 {
            level = fold_level(&level);
        }
        level[0]
    }

    fn reference_tile_subtree_root(tile: &[u8]) -> [u8; 32] {
        let mut level: Vec<[u8; 32]> = tile.chunks(POM_V4_CHUNK_BYTES).map(blake).collect();
        for _ in 0..TILE_SUBTREE_DEPTH {
            level = fold_level(&level);
        }
        level[0]
    }

    #[test]
    fn fixed_scratch_helpers_match_reference() {
        let (blob, _) = test_blob(2);
        let tile = &blob[..POM_V4_TILE_BYTES];
        let state = v4_initial_state(0xA5A5_5A5A_1234_5678);
        assert_eq!(v4_tile_subtree_root(tile), reference_tile_subtree_root(tile));
        assert_eq!(v4_state_root(&state), reference_state_root(&state));

        let expected = v4_transition(&state, tile, 17);
        let mut actual = vec![0u8; POM_V4_TILE_BYTES];
        v4_transition_into(&state, tile, 17, &mut actual);
        assert_eq!(actual, expected);
    }

    #[test]
    fn honest_proof_verifies_and_returns_final_state() {
        let (blob, leaves) = test_blob(64);
        let r_t = r_t_root(&leaves);
        let n_chunks = (blob.len() / POM_V4_CHUNK_BYTES) as u64;
        let seed = 0x1234_5678_9ABC_DEF0;

        let proof = v4_prove(seed, 2, &blob, &leaves).unwrap();
        let fs = verify_pom_proof_v4(seed, &proof, &r_t, n_chunks).unwrap();
        // The verifier's re-walk must agree with the reference walk's final state.
        let walk = v4_walk(seed, &blob).unwrap();
        assert_eq!(fs, fold64(&v4_state_root(walk.states.last().unwrap())));
    }

    #[test]
    fn tampered_tile_is_rejected() {
        let (blob, leaves) = test_blob(64);
        let r_t = r_t_root(&leaves);
        let n_chunks = (blob.len() / POM_V4_CHUNK_BYTES) as u64;
        let seed = 0x0FED_CBA9_8765_4321;

        let mut proof = v4_prove(seed, 2, &blob, &leaves).unwrap();
        // Flip one weight byte in the first tile — no longer a genuine model tile.
        proof.tiles[0][0] ^= 0x01;
        assert_eq!(verify_pom_proof_v4(seed, &proof, &r_t, n_chunks), Err(PomV4VerifyError::BadTilePath));
    }

    #[test]
    fn grind_final_state_is_not_free() {
        let (blob, leaves) = test_blob(64);
        let r_t = r_t_root(&leaves);
        let n_chunks = (blob.len() / POM_V4_CHUNK_BYTES) as u64;
        let seed = 0xABCD_1234_ABCD_1234;

        let honest = v4_prove(seed, 2, &blob, &leaves).unwrap();
        let fs_honest = verify_pom_proof_v4(seed, &honest, &r_t, n_chunks).unwrap();

        // A substituted genuine tile either fails its range proof at the derived offset or
        // yields a different final_state (rejected by the header pin).
        let mut forged = honest.clone();
        forged.tiles[10] = blob[0..POM_V4_TILE_BYTES].to_vec();
        forged.merkle[10] = PomV4RangeProof { path: v4_tile_path(&leaves, 0) };
        match verify_pom_proof_v4(seed, &forged, &r_t, n_chunks) {
            Err(_) => {}
            Ok(fs) => assert_ne!(fs, fs_honest, "a re-rolled state must not reproduce the honest final_state"),
        }
        // Sanity: the honest final_state is stable across re-verification.
        assert_eq!(fs_honest, verify_pom_proof_v4(seed, &honest, &r_t, n_chunks).unwrap());
    }

    #[test]
    fn wrong_shape_is_rejected() {
        let (blob, leaves) = test_blob(64);
        let r_t = r_t_root(&leaves);
        let n_chunks = (blob.len() / POM_V4_CHUNK_BYTES) as u64;
        let seed = 7;
        let mut proof = v4_prove(seed, 0, &blob, &leaves).unwrap();
        proof.tiles.pop();
        assert_eq!(verify_pom_proof_v4(seed, &proof, &r_t, n_chunks), Err(PomV4VerifyError::WrongShape));
    }

    fn container(v4: PomProofV4, final_state: u64, pow: [u8; 32]) -> crate::pom::PomProof {
        crate::pom::PomProof {
            tier: v4.tier,
            trace_root: [0u8; 32],
            pow_value: pow,
            final_state,
            initial_trace_path: vec![],
            final_trace_path: vec![],
            openings: vec![],
            steps_v2: None,
            v3: None,
            v4: Some(v4),
        }
    }

    #[test]
    fn container_honest_passes() {
        let (blob, leaves) = test_blob(64);
        let r_t = r_t_root(&leaves);
        let n_chunks = (blob.len() / POM_V4_CHUNK_BYTES) as u64;
        let seed = 0x5151_5151_5151_5151;
        let pph = [3u8; 32];
        let _nonce = 99u64;

        let v4 = v4_prove(seed, 2, &blob, &leaves).unwrap();
        let fs = verify_pom_proof_v4(seed, &v4, &r_t, n_chunks).unwrap();
        let final_hash = |s: u64| crate::pom::pom_pow_value_h3(s, &pph);
        let pow = final_hash(fs);
        let target = [0xffu8; 32];
        let proof = container(v4, fs, pow);
        assert!(verify_pom_proof_v4_container(seed, &proof, n_chunks, &r_t, &target, final_hash).is_ok());
    }

    #[test]
    fn container_wrong_final_state_rejected() {
        let (blob, leaves) = test_blob(64);
        let r_t = r_t_root(&leaves);
        let n_chunks = (blob.len() / POM_V4_CHUNK_BYTES) as u64;
        let seed = 0x2626_2626_2626_2626;
        let pph = [8u8; 32];
        let _nonce = 5u64;

        let v4 = v4_prove(seed, 2, &blob, &leaves).unwrap();
        let fs = verify_pom_proof_v4(seed, &v4, &r_t, n_chunks).unwrap();
        let final_hash = |s: u64| crate::pom::pom_pow_value_h3(s, &pph);
        // Claim a final_state that isn't what the walk derives.
        let bad = fs ^ 1;
        let proof = container(v4, bad, final_hash(bad));
        let target = [0xffu8; 32];
        assert_eq!(
            verify_pom_proof_v4_container(seed, &proof, n_chunks, &r_t, &target, final_hash),
            Err(PomV4VerifyError::FinalStateMismatch)
        );
    }
}
