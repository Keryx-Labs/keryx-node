//! Proof-of-Model v3 — the H6 matrix-state walk (reference implementation = executable spec).
//!
//! The v1/v2 walk proves possession with a hash chain over 32 B chunks; the work spent is
//! hashing. The v3 walk makes the PoW itself int8 matrix arithmetic over the committed weight
//! bytes: one step multiplies the D x D state by a data-selected 64 KB weight tile and applies
//! a non-linear, position-dependent byte reduction. Verification is by spot-checking committed
//! per-step states — the verifier NEVER re-walks (K * D^3 = 4.3 GMACs/block would put IBD back
//! to days, the H3 lesson), it re-checks `POM_V3_CHECKS` single entries at D MACs each.
//!
//! Soundness model and attack catalogue: `H6_soundness_spot_checking_2026-08-06.md`. The two
//! red-team results that shaped this code:
//!   * offsets derive from (seed, step, previous-tile snippet) and the proof carries the FULL
//!     snippet list — the verifier derives every offset from that list alone. Deriving offsets
//!     from the state is fatally steerable (A3); verifying snippet links only locally without
//!     the listed anchor is fatally shiftable (A9).
//!   * the byte reduction `rho` is non-linear AND position/step-tweaked: plain `acc & 0xff`
//!     makes the walk linear over Z/256 (backward repair, A4), and any fixed point of rho
//!     (e.g. rho(0) = 0) makes a forged constant state self-consistent for free (A10).
//!
//! This module is intentionally standalone (pure functions over byte slices): the node wires it
//! into `check_pom_proof` era dispatch separately, the miner's GPU kernel must reproduce it
//! byte-for-byte, and the tests double as the cheat-miner harness (each attack class from the
//! study is exercised as a rejection case).
//!
//! Weight commitment: UNCHANGED — the existing per-tier `R_T` (blake3 over the canonical 32 B
//! chunk stream, `pom-rt-builder`). A v3 "tile" is 2048 consecutive canonical chunks (64 KB), a
//! "column" is 8 consecutive chunks (256 B), opened as an 8-leaf aligned subtree range proof.

use crate::pom::{blake, hash_pair, mix64, verify_merkle};
use borsh::{BorshDeserialize, BorshSerialize};
use serde::{Deserialize, Serialize};

/// State dimension: the walk state is a D x D int8 matrix (64 KB).
pub const POM_V3_D: usize = 256;
/// Walk length in steps (tiles multiplied per nonce).
pub const POM_V3_K: usize = 256;
/// Spot-checks per proof. Soundness: a prover missing/forging a fraction `f` of its steps
/// survives verification with probability ~e^(-m*f) (study §4).
pub const POM_V3_CHECKS: usize = 32;
/// Canonical R_T chunk size (bytes) — pinned by `pom-rt-builder`, do not change.
pub const POM_V3_CHUNK_BYTES: usize = 32;
/// Tile size: 2048 canonical chunks = 64 KB.
pub const POM_V3_TILE_BYTES: usize = 65536;
pub const POM_V3_TILE_CHUNKS: u64 = (POM_V3_TILE_BYTES / POM_V3_CHUNK_BYTES) as u64;
/// Offset-chain snippet: the first 32 bytes (= first canonical chunk) of a tile.
pub const POM_V3_SNIPPET_BYTES: usize = 32;
/// Chunks per opened column (256 B) and leaf-depth of the column subtree (2^3 = 8).
pub const POM_V3_COL_CHUNKS: u64 = (POM_V3_D / POM_V3_CHUNK_BYTES) as u64;
const COL_SUBTREE_DEPTH: u32 = 3;

/// Domain salts. Derivation: sha256(label), first 8 bytes read as little-endian u64
/// (same convention as the H3/H5 pph salts).
/// sha256("keryx-h6-s0-row-salt")
pub const POM_V3_S0_ROW_SALT: u64 = 0x6B61F28F3CC48744;
/// sha256("keryx-h6-offset-first-salt")
pub const POM_V3_OFFSET_FIRST_SALT: u64 = 0x3F1F886D659E316A;
/// sha256("keryx-h6-offset-step-salt")
pub const POM_V3_OFFSET_STEP_SALT: u64 = 0xD4C194F3ADB3B1C7;
/// Domain prefixes for the blake3-based derivations.
const CHECKS_DOMAIN: &[u8] = b"keryx-h6-checks-v3";

/// One walk state: a D x D int8 matrix, row-major (`state[row * D + col]`).
pub type V3State = Vec<u8>; // POM_V3_D * POM_V3_D bytes

/// One spot-check opening: everything needed to re-check a single entry (t, x, j) of the walk.
#[derive(Clone, Debug, Serialize, Deserialize, BorshSerialize, BorshDeserialize)]
#[serde(rename_all = "camelCase")]
pub struct PomV3Opening {
    /// Row `x` of the committed state S_{t-1} (D bytes).
    pub row_before: Vec<u8>,
    /// Merkle path proving `leaf(row_before)` is row `x` under `roots[t-1]`.
    pub path_before: Vec<[u8; 32]>,
    /// Row `x` of the committed state S_t (D bytes).
    pub row_after: Vec<u8>,
    /// Merkle path proving `leaf(row_after)` is row `x` under `roots[t]`.
    pub path_after: Vec<[u8; 32]>,
    /// Column `j` of the tile read at step `t`: 256 weight bytes = 8 canonical chunks.
    pub tile_col: Vec<u8>,
    /// Merkle path from the 8-leaf column subtree root to `R_T` (subtree index
    /// `tile_offset * 256 + j`).
    pub col_path: Vec<[u8; 32]>,
    /// Merkle path proving `leaf(snippets[t-1])` is the FIRST chunk of the tile at the
    /// list-derived offset `i_t` under `R_T` — anchors the offset chain (A9).
    pub snippet_path: Vec<[u8; 32]>,
}

/// Post-PoW witness for the v3 (H6) walk, carried at the Block level like `PomProof`.
#[derive(Clone, Debug, Serialize, Deserialize, BorshSerialize, BorshDeserialize)]
#[serde(rename_all = "camelCase")]
pub struct PomProofV3 {
    /// Tier index — selects `R_T` and the reward bracket (authenticated by the proof itself:
    /// openings only verify against the claimed tier's root).
    pub tier: u8,
    /// Per-step state commitments `root(S_0) ..= root(S_K)` (K+1 entries). `roots[K]` is the
    /// lottery binding: the container pins `final_state = fold64(roots[K])` and the era pow
    /// fold of `final_state` must meet the tier target (header-only verifiable, H3 pin intact).
    pub roots: Vec<[u8; 32]>,
    /// The K offset-chain snippets: `snippets[t-1]` = first 32 B of the tile read at step t.
    /// The verifier derives EVERY tile offset from this list (never link-by-link).
    pub snippets: Vec<[u8; 32]>,
    /// The `POM_V3_CHECKS` openings, in PRF challenge order.
    pub checks: Vec<PomV3Opening>,
}

impl PomProofV3 {
    /// Rough wire size (cache accounting), dominated by the openings' Merkle paths.
    pub fn approx_bytes(&self) -> usize {
        let path = |p: &Vec<[u8; 32]>| p.len() * 32;
        let checks: usize = self
            .checks
            .iter()
            .map(|c| {
                c.row_before.len()
                    + c.row_after.len()
                    + c.tile_col.len()
                    + path(&c.path_before)
                    + path(&c.path_after)
                    + path(&c.col_path)
                    + path(&c.snippet_path)
            })
            .sum();
        1 + self.roots.len() * 32 + self.snippets.len() * 32 + checks
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PomV3VerifyError {
    /// `roots` length != K+1, `snippets` length != K, or `checks` length != POM_V3_CHECKS.
    WrongShape,
    /// An opening's row/column byte length is not D / not 8 chunks.
    WrongOpeningShape,
    /// `roots[0]` differs from the S_0 tree the verifier derives from the block seed.
    BadInitialRoot,
    /// A `row_before` opening fails its Merkle path against `roots[t-1]`.
    BadRowBeforePath,
    /// A `row_after` opening fails its Merkle path against `roots[t]`.
    BadRowAfterPath,
    /// A tile-column opening fails its range proof against `R_T`.
    BadColumnPath,
    /// A listed snippet fails its Merkle path against `R_T` at the list-derived offset.
    BadSnippetPath,
    /// The recomputed entry `rho(row_before . tile_col)` differs from `row_after[j]`.
    TransitionMismatch,
    /// The blob is too small to hold a single tile.
    BlobTooSmall,
    // --- container-level (verify_pom_proof_v3_container) ---
    /// Post-v3-gate proof without the `v3` witness.
    MissingV3,
    /// Post-v3-gate proof still carries legacy trace/openings/steps content (must be
    /// canonically empty).
    NonCanonicalLegacyFields,
    /// Container `tier` differs from the v3 witness `tier`.
    TierMismatch,
    /// Container `final_state` differs from `fold64(roots[K])` (breaks the H3 header pin).
    FinalStateMismatch,
    /// Container `pow_value` differs from the era pow fold of `final_state`.
    PowValueMismatch,
    /// The era pow fold of `final_state` does not meet the tier target.
    TargetNotMet,
}

// --- walk primitives (the miner kernel must reproduce these byte-for-byte) ---

/// Non-linear position/step-tweaked byte reduction (A4 + A10).
/// `acc` is the exact int32 dot product; the tweak folds (step, row, col).
#[inline]
pub fn rho8(acc: i32, tweak: u32) -> u8 {
    let mut z = (acc as u32) ^ tweak;
    z = z.wrapping_mul(0x9E3779B9);
    z ^= z >> 16;
    z = z.wrapping_mul(0x85EBCA6B);
    z ^= z >> 13;
    (z & 0xff) as u8
}

/// The rho tweak for entry (step t in 1..=K, row x, col j).
#[inline]
pub fn rho_tweak(step: u32, row: u32, col: u32) -> u32 {
    step.wrapping_mul(0x9E3779B9).wrapping_add(row.wrapping_mul(0xC2B2AE35)).wrapping_add(col.wrapping_mul(0x85EBCA6B))
}

/// Signed int8 dot product of a state row and a tile column (both D bytes).
/// No overflow: |acc| <= 256 * 128 * 128 < 2^22.
#[inline]
pub fn dot_i8(row: &[u8], col: &[u8]) -> i32 {
    debug_assert_eq!(row.len(), POM_V3_D);
    debug_assert_eq!(col.len(), POM_V3_D);
    row.iter().zip(col.iter()).map(|(&a, &b)| (a as i8 as i32) * (b as i8 as i32)).sum()
}

/// S_0 from the canonical block seed (the same era-salted `pom_block_seed_*` fold as v1/v2):
/// row r is a mix64 keystream, 4 little-endian bytes per squeeze.
pub fn v3_initial_state(seed: u64) -> V3State {
    let mut s = vec![0u8; POM_V3_D * POM_V3_D];
    for r in 0..POM_V3_D {
        let mut h = mix64(seed ^ POM_V3_S0_ROW_SALT.wrapping_add(r as u64));
        for k4 in 0..POM_V3_D / 4 {
            h = mix64(h);
            s[r * POM_V3_D + k4 * 4..r * POM_V3_D + k4 * 4 + 4].copy_from_slice(&(h as u32).to_le_bytes());
        }
    }
    s
}

/// Fold of a 32 B snippet: 8 little-endian u32 words chained through mix64.
#[inline]
pub fn snippet_fold(snippet: &[u8; POM_V3_SNIPPET_BYTES]) -> u64 {
    let mut sf = 0u64;
    for w in 0..8 {
        let word = u32::from_le_bytes(snippet[w * 4..w * 4 + 4].try_into().unwrap()) as u64;
        sf = mix64(sf ^ word);
    }
    sf
}

/// Tile offset for step 1 (no previous tile: seed-only).
#[inline]
pub fn v3_first_offset(seed: u64, n_tiles: u64) -> u64 {
    mix64(seed ^ POM_V3_OFFSET_FIRST_SALT) % n_tiles
}

/// Tile offset for step t+1 from the snippet of the tile read at step t (t in 1..=K-1).
#[inline]
pub fn v3_next_offset(seed: u64, step: u64, snippet: &[u8; POM_V3_SNIPPET_BYTES], n_tiles: u64) -> u64 {
    mix64(seed ^ (step + 1).wrapping_mul(POM_V3_OFFSET_STEP_SALT) ^ snippet_fold(snippet)) % n_tiles
}

/// One walk transition: S_t = rho(S_{t-1} x tile), entrywise, column j of the tile being
/// `tile[j*D .. (j+1)*D]` (a tile column = 256 consecutive canonical-stream bytes).
pub fn v3_transition(state: &V3State, tile: &[u8], step: u32) -> V3State {
    debug_assert_eq!(tile.len(), POM_V3_TILE_BYTES);
    let mut next = vec![0u8; POM_V3_D * POM_V3_D];
    for x in 0..POM_V3_D {
        let row = &state[x * POM_V3_D..(x + 1) * POM_V3_D];
        for j in 0..POM_V3_D {
            let col = &tile[j * POM_V3_D..(j + 1) * POM_V3_D];
            next[x * POM_V3_D + j] = rho8(dot_i8(row, col), rho_tweak(step, x as u32, j as u32));
        }
    }
    next
}

/// Merkle root over the D rows of a state (leaf = blake3(row), D = 256 is an exact power of
/// two so the tree is complete: depth 8, no duplicate-last).
pub fn v3_state_root(state: &V3State) -> [u8; 32] {
    let mut level: Vec<[u8; 32]> = (0..POM_V3_D).map(|r| blake(&state[r * POM_V3_D..(r + 1) * POM_V3_D])).collect();
    while level.len() > 1 {
        level = level.chunks(2).map(|p| hash_pair(&p[0], &p[1])).collect();
    }
    level[0]
}

/// Merkle path for row `x` inside a state (siblings bottom-up, matches `verify_merkle`).
fn v3_state_row_path(state: &V3State, x: usize) -> Vec<[u8; 32]> {
    let mut level: Vec<[u8; 32]> = (0..POM_V3_D).map(|r| blake(&state[r * POM_V3_D..(r + 1) * POM_V3_D])).collect();
    let mut path = Vec::with_capacity(8);
    let mut idx = x;
    while level.len() > 1 {
        path.push(level[idx ^ 1]);
        level = level.chunks(2).map(|p| hash_pair(&p[0], &p[1])).collect();
        idx >>= 1;
    }
    path
}

/// The PRF deriving the spot-check triples (t, x, j). Binds the header (pre_pow_hash, nonce)
/// and the ENTIRE deterministic commitment (all roots, all snippets) — no salt, no prover
/// freedom (A5): re-rolling any committed byte re-rolls root_K, i.e. the lottery.
pub fn v3_check_points(
    pre_pow_hash: &[u8; 32],
    nonce: u64,
    roots: &[[u8; 32]],
    snippets: &[[u8; 32]],
) -> Vec<(u32, u32, u32)> {
    let mut acc = Vec::with_capacity(CHECKS_DOMAIN.len() + 32 + 8 + 64);
    acc.extend_from_slice(CHECKS_DOMAIN);
    acc.extend_from_slice(pre_pow_hash);
    acc.extend_from_slice(&nonce.to_le_bytes());
    let mut roots_buf = Vec::with_capacity(roots.len() * 32);
    roots.iter().for_each(|r| roots_buf.extend_from_slice(r));
    acc.extend_from_slice(&blake(&roots_buf));
    let mut snip_buf = Vec::with_capacity(snippets.len() * 32);
    snippets.iter().for_each(|s| snip_buf.extend_from_slice(s));
    acc.extend_from_slice(&blake(&snip_buf));
    let prf_seed = blake(&acc);

    (0..POM_V3_CHECKS as u64)
        .map(|i| {
            let mut buf = [0u8; 40];
            buf[..32].copy_from_slice(&prf_seed);
            buf[32..].copy_from_slice(&i.to_le_bytes());
            let d = blake(&buf);
            // K and D are powers of two: the modulo is bias-free.
            let t = 1 + (u64::from_le_bytes(d[..8].try_into().unwrap()) % POM_V3_K as u64) as u32;
            let x = (u64::from_le_bytes(d[8..16].try_into().unwrap()) % POM_V3_D as u64) as u32;
            let j = (u64::from_le_bytes(d[16..24].try_into().unwrap()) % POM_V3_D as u64) as u32;
            (t, x, j)
        })
        .collect()
}

// --- prover (reference; the production miner re-walks on GPU and only builds trees for the
// --- winning nonce, but MUST produce these exact bytes) ---

/// Full walk trace for one nonce over an in-memory canonical blob.
pub struct V3Walk {
    /// S_0 ..= S_K (K+1 states).
    pub states: Vec<V3State>,
    /// Tile offsets i_1 ..= i_K.
    pub offsets: Vec<u64>,
    /// Snippets of the tiles read at steps 1..=K.
    pub snippets: Vec<[u8; 32]>,
}

/// Number of whole tiles in a canonical blob (`n_chunks` 32 B chunks). The sub-tile remainder
/// is never walked (it stays covered by R_T; only completeness of 8-leaf subtrees matters).
pub fn v3_n_tiles(n_chunks: u64) -> u64 {
    n_chunks / POM_V3_TILE_CHUNKS
}

/// Reference walk over an in-memory canonical byte stream (tests / cheat-miner harness).
pub fn v3_walk(seed: u64, blob: &[u8]) -> Result<V3Walk, PomV3VerifyError> {
    let n_tiles = v3_n_tiles((blob.len() / POM_V3_CHUNK_BYTES) as u64);
    if n_tiles == 0 {
        return Err(PomV3VerifyError::BlobTooSmall);
    }
    let mut states = Vec::with_capacity(POM_V3_K + 1);
    let mut offsets = Vec::with_capacity(POM_V3_K);
    let mut snippets = Vec::with_capacity(POM_V3_K);
    states.push(v3_initial_state(seed));
    let mut off = v3_first_offset(seed, n_tiles);
    for step in 1..=POM_V3_K as u32 {
        let tile = &blob[(off as usize) * POM_V3_TILE_BYTES..(off as usize + 1) * POM_V3_TILE_BYTES];
        let snippet: [u8; 32] = tile[..POM_V3_SNIPPET_BYTES].try_into().unwrap();
        offsets.push(off);
        snippets.push(snippet);
        let next = v3_transition(states.last().unwrap(), tile, step);
        states.push(next);
        if (step as usize) < POM_V3_K {
            off = v3_next_offset(seed, step as u64, &snippet, n_tiles);
        }
    }
    Ok(V3Walk { states, offsets, snippets })
}

/// Merkle path under `R_T` for an aligned run of leaves, returned as (subtree_index, path):
/// the caller folds the run into its subtree root itself. `leaf_hashes` is the FULL leaf
/// level of the tier tree — reference-only (tests); the production prover streams this.
fn rt_subtree_path(leaf_hashes: &[[u8; 32]], first_leaf: u64, depth: u32) -> ([u8; 32], u64, Vec<[u8; 32]>) {
    // Fold the whole leaf level up `depth` levels (duplicate-last at each level, matching
    // pom-rt-builder's carried-peak construction for non-power-of-two counts).
    let mut level: Vec<[u8; 32]> = leaf_hashes.to_vec();
    for _ in 0..depth {
        level = fold_level(&level);
    }
    let mut idx = first_leaf >> depth;
    let subtree_root = level[idx as usize];
    let mut path = Vec::new();
    while level.len() > 1 {
        let sib = if (idx ^ 1) < level.len() as u64 { level[(idx ^ 1) as usize] } else { level[idx as usize] };
        path.push(sib);
        level = fold_level(&level);
        idx >>= 1;
    }
    (subtree_root, first_leaf >> depth, path)
}

fn fold_level(level: &[[u8; 32]]) -> Vec<[u8; 32]> {
    level
        .chunks(2)
        .map(|p| if p.len() == 2 { hash_pair(&p[0], &p[1]) } else { hash_pair(&p[0], &p[0]) })
        .collect()
}

/// Fold 8 column-chunk leaves into their aligned subtree root (depth 3, always complete).
fn col_subtree_root(col: &[u8]) -> [u8; 32] {
    let leaves: Vec<[u8; 32]> =
        col.chunks(POM_V3_CHUNK_BYTES).map(blake).collect();
    let l1 = fold_level(&leaves);
    let l2 = fold_level(&l1);
    hash_pair(&l2[0], &l2[1])
}

/// Reference prover: walk, commit, derive checks, open. `blob` is the canonical byte stream
/// of the tier (whole tiles only are walked); `leaf_hashes` its full 32 B-chunk leaf level.
pub fn v3_prove(
    pre_pow_hash: &[u8; 32],
    nonce: u64,
    seed: u64,
    tier: u8,
    blob: &[u8],
    leaf_hashes: &[[u8; 32]],
) -> Result<PomProofV3, PomV3VerifyError> {
    let walk = v3_walk(seed, blob)?;
    let roots: Vec<[u8; 32]> = walk.states.iter().map(v3_state_root).collect();
    let points = v3_check_points(pre_pow_hash, nonce, &roots, &walk.snippets);
    let checks = points
        .iter()
        .map(|&(t, x, j)| {
            let (ti, xi, ji) = (t as usize, x as usize, j as usize);
            let off = walk.offsets[ti - 1];
            let s_before = &walk.states[ti - 1];
            let s_after = &walk.states[ti];
            let col_first_leaf = off * POM_V3_TILE_CHUNKS + j as u64 * POM_V3_COL_CHUNKS;
            let (_, _, col_path) = rt_subtree_path(leaf_hashes, col_first_leaf, COL_SUBTREE_DEPTH);
            let (_, _, snippet_path) = rt_subtree_path(leaf_hashes, off * POM_V3_TILE_CHUNKS, 0);
            PomV3Opening {
                row_before: s_before[xi * POM_V3_D..(xi + 1) * POM_V3_D].to_vec(),
                path_before: v3_state_row_path(s_before, xi),
                row_after: s_after[xi * POM_V3_D..(xi + 1) * POM_V3_D].to_vec(),
                path_after: v3_state_row_path(s_after, xi),
                tile_col: blob[(off as usize) * POM_V3_TILE_BYTES + ji * POM_V3_D
                    ..(off as usize) * POM_V3_TILE_BYTES + (ji + 1) * POM_V3_D]
                    .to_vec(),
                col_path,
                snippet_path,
            }
        })
        .collect();
    Ok(PomProofV3 { tier, roots, snippets: walk.snippets, checks })
}

// --- verifier (weightless; this is what the node runs per block) ---

/// Deterministically verify a v3 proof against the tier root `r_t` (existing R_T, unchanged)
/// and its canonical chunk count. Cost: K+2 mix64 chains for the offsets, one S_0 tree
/// (~256 row hashes), and per check ~2 row paths + 1 range proof + 1 snippet path + D MACs.
/// The caller checks the lottery (`era_pow_fold(fold64(roots[K])) <= target`) and the
/// header binding separately — see `verify_pom_proof_v3_container`.
pub fn verify_pom_proof_v3(
    pre_pow_hash: &[u8; 32],
    nonce: u64,
    seed: u64,
    proof: &PomProofV3,
    r_t: &[u8; 32],
    n_chunks: u64,
) -> Result<(), PomV3VerifyError> {
    if proof.roots.len() != POM_V3_K + 1 || proof.snippets.len() != POM_V3_K || proof.checks.len() != POM_V3_CHECKS {
        return Err(PomV3VerifyError::WrongShape);
    }
    let n_tiles = v3_n_tiles(n_chunks);
    if n_tiles == 0 {
        return Err(PomV3VerifyError::BlobTooSmall);
    }

    // S_0 is a pure function of the seed: recompute its root (no trust in roots[0]).
    if v3_state_root(&v3_initial_state(seed)) != proof.roots[0] {
        return Err(PomV3VerifyError::BadInitialRoot);
    }

    // Derive the FULL offset chain from the listed snippets (A9: list is the anchor).
    let mut offsets = Vec::with_capacity(POM_V3_K);
    let mut off = v3_first_offset(seed, n_tiles);
    for step in 1..=POM_V3_K as u64 {
        offsets.push(off);
        if (step as usize) < POM_V3_K {
            off = v3_next_offset(seed, step, &proof.snippets[step as usize - 1], n_tiles);
        }
    }

    let points = v3_check_points(pre_pow_hash, nonce, &proof.roots, &proof.snippets);
    for (point, open) in points.iter().zip(proof.checks.iter()) {
        let &(t, x, j) = point;
        let (ti, xi, ji) = (t as usize, x as usize, j as u64);
        if open.row_before.len() != POM_V3_D || open.row_after.len() != POM_V3_D || open.tile_col.len() != POM_V3_D {
            return Err(PomV3VerifyError::WrongOpeningShape);
        }
        // State rows against their committed per-step roots.
        if !verify_merkle(blake(&open.row_before), xi as u64, &open.path_before, &proof.roots[ti - 1]) {
            return Err(PomV3VerifyError::BadRowBeforePath);
        }
        if !verify_merkle(blake(&open.row_after), xi as u64, &open.path_after, &proof.roots[ti]) {
            return Err(PomV3VerifyError::BadRowAfterPath);
        }
        // Tile column against R_T: 8 aligned chunk leaves -> depth-3 subtree root -> path.
        let off_t = offsets[ti - 1];
        let subtree_index = off_t * (POM_V3_TILE_CHUNKS >> COL_SUBTREE_DEPTH) + ji;
        if !verify_merkle(col_subtree_root(&open.tile_col), subtree_index, &open.col_path, r_t) {
            return Err(PomV3VerifyError::BadColumnPath);
        }
        // Listed snippet against R_T at the list-derived tile offset (offset-chain anchor).
        if !verify_merkle(blake(&proof.snippets[ti - 1]), off_t * POM_V3_TILE_CHUNKS, &open.snippet_path, r_t) {
            return Err(PomV3VerifyError::BadSnippetPath);
        }
        // The transition itself: one entry, D MACs + rho.
        if rho8(dot_i8(&open.row_before, &open.tile_col), rho_tweak(t, x, j)) != open.row_after[ji as usize] {
            return Err(PomV3VerifyError::TransitionMismatch);
        }
    }
    Ok(())
}

/// 64-bit header fold of the final-state commitment: `Header::pom_final_state` carries this at
/// and after the v3 gate, so every header-only mechanism (H3 pin, `pom_pow_value_h3` header PoW,
/// block levels, pruning proofs) works unchanged. The full 256-bit binding lives in the proof:
/// re-rolling any committed byte re-rolls `roots[K]`, and colliding this 64-bit fold on purpose
/// costs ~2^64 walks — far beyond the lottery it would cheat.
#[inline]
pub fn fold64(root: &[u8; 32]) -> u64 {
    u64::from_le_bytes(root[..8].try_into().unwrap())
}

/// Full v3 validation of a container `PomProof`, mirroring `verify_pom_proof_v2`'s role in
/// `check_pom_proof`: canonical-shape checks on the legacy fields, header-pin consistency
/// (`final_state`/`pow_value`), the tier target, and the structural spot-check verification.
/// `final_hash` is the same era pow fold the header-only path uses (H3-salted).
#[allow(clippy::too_many_arguments)]
pub fn verify_pom_proof_v3_container<Hf: Fn(u64) -> [u8; 32]>(
    pre_pow_hash: &[u8; 32],
    nonce: u64,
    seed: u64,
    proof: &crate::pom::PomProof,
    n_chunks: u64,
    r_t: &[u8; 32],
    target: &[u8; 32],
    final_hash: Hf,
) -> Result<(), PomV3VerifyError> {
    let v3 = proof.v3.as_ref().ok_or(PomV3VerifyError::MissingV3)?;
    // Legacy fields must be canonical placeholders: a v3 proof that also smuggles v1/v2
    // content has no meaning and only inflates relay/storage.
    if proof.trace_root != [0u8; 32]
        || !proof.initial_trace_path.is_empty()
        || !proof.final_trace_path.is_empty()
        || !proof.openings.is_empty()
        || proof.steps_v2.is_some()
    {
        return Err(PomV3VerifyError::NonCanonicalLegacyFields);
    }
    if proof.tier != v3.tier {
        return Err(PomV3VerifyError::TierMismatch);
    }
    // Structure + offsets + S_0 + all spot-checks (validates shapes before we index roots[K]).
    verify_pom_proof_v3(pre_pow_hash, nonce, seed, v3, r_t, n_chunks)?;
    // Header pin: final_state is the 64-bit fold of the final-state commitment.
    if proof.final_state != fold64(&v3.roots[POM_V3_K]) {
        return Err(PomV3VerifyError::FinalStateMismatch);
    }
    // Lottery: same era fold and target comparison as the header-only PoW path.
    let pv = final_hash(proof.final_state);
    if pv != proof.pow_value {
        return Err(PomV3VerifyError::PowValueMismatch);
    }
    if !crate::pom::le_leq(&pv, target) {
        return Err(PomV3VerifyError::TargetNotMet);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random blob of `n_tiles` whole tiles (+ a ragged tail to exercise
    /// the duplicate-last R_T shape), with its full leaf level.
    fn test_blob(n_tiles: usize) -> (Vec<u8>, Vec<[u8; 32]>) {
        let tail_chunks = 5usize; // deliberately not a multiple of 8
        let n_bytes = n_tiles * POM_V3_TILE_BYTES + tail_chunks * POM_V3_CHUNK_BYTES;
        let mut blob = vec![0u8; n_bytes];
        let mut h = 0xDEADBEEFu64;
        for b in blob.iter_mut() {
            h = mix64(h);
            *b = h as u8;
        }
        let leaves: Vec<[u8; 32]> = blob.chunks(POM_V3_CHUNK_BYTES).map(blake).collect();
        (blob, leaves)
    }

    fn rt_root(leaves: &[[u8; 32]]) -> [u8; 32] {
        let mut level = leaves.to_vec();
        while level.len() > 1 {
            level = fold_level(&level);
        }
        level[0]
    }

    const PPH: [u8; 32] = [7u8; 32];
    const NONCE: u64 = 0x1234_5678_9ABC_DEF0;
    const SEED: u64 = 0x0F1E_2D3C_4B5A_6978;

    fn honest_setup() -> (Vec<u8>, Vec<[u8; 32]>, [u8; 32], u64, PomProofV3) {
        let (blob, leaves) = test_blob(16);
        let root = rt_root(&leaves);
        let n_chunks = (blob.len() / POM_V3_CHUNK_BYTES) as u64;
        let proof = v3_prove(&PPH, NONCE, SEED, 0, &blob, &leaves).unwrap();
        (blob, leaves, root, n_chunks, proof)
    }

    #[test]
    fn honest_proof_verifies() {
        let (_, _, root, n_chunks, proof) = honest_setup();
        assert_eq!(verify_pom_proof_v3(&PPH, NONCE, SEED, &proof, &root, n_chunks), Ok(()));
        // Sanity: proof stays in the ~100 KB envelope from the study.
        assert!(proof.approx_bytes() < 150 * 1024, "proof size {} B", proof.approx_bytes());
    }

    #[test]
    fn rho_has_no_zero_fixed_point() {
        // A10: an all-zero state must NOT be self-reproducing. rho(0, tweak) == 0 for every
        // entry of a step would be required; assert it fails immediately.
        let z = (0..POM_V3_D as u32).filter(|&j| rho8(0, rho_tweak(1, 0, j)) == 0).count();
        assert!(z < POM_V3_D / 8, "rho(0) collapses to zero on {z}/256 entries");
    }

    #[test]
    fn forged_transition_rejected() {
        // A1: forge every transition (garbage states, self-consistent trees) — any sampled
        // check must catch the broken matmul.
        let (blob, leaves, root, n_chunks, honest) = honest_setup();
        let mut walk = v3_walk(SEED, &blob).unwrap();
        for state in walk.states.iter_mut().skip(1) {
            for b in state.iter_mut() {
                *b = b.wrapping_add(1);
            }
        }
        let roots: Vec<[u8; 32]> = walk.states.iter().map(v3_state_root).collect();
        let points = v3_check_points(&PPH, NONCE, &roots, &walk.snippets);
        let mut proof = honest.clone();
        proof.roots = roots;
        let template = honest.checks[0].clone();
        proof.checks = points
            .iter()
            .map(|&(t, x, _)| {
                let (ti, xi) = (t as usize, x as usize);
                let mut c = template.clone();
                c.row_before = walk.states[ti - 1][xi * POM_V3_D..(xi + 1) * POM_V3_D].to_vec();
                c.path_before = v3_state_row_path(&walk.states[ti - 1], xi);
                c.row_after = walk.states[ti][xi * POM_V3_D..(xi + 1) * POM_V3_D].to_vec();
                c.path_after = v3_state_row_path(&walk.states[ti], xi);
                c
            })
            .collect();
        // Rebuild honest column/snippet openings for the sampled points so ONLY the
        // transition is wrong.
        let honest_full = v3_prove_at_points(&points, &walk, &blob, &leaves, &mut proof);
        assert!(matches!(
            verify_pom_proof_v3(&PPH, NONCE, SEED, &honest_full, &root, n_chunks),
            Err(PomV3VerifyError::TransitionMismatch) | Err(PomV3VerifyError::BadRowBeforePath)
        ));
    }

    /// Rebuild the tile/snippet parts of each opening honestly for arbitrary points.
    fn v3_prove_at_points(
        points: &[(u32, u32, u32)],
        walk: &V3Walk,
        blob: &[u8],
        leaves: &[[u8; 32]],
        proof: &mut PomProofV3,
    ) -> PomProofV3 {
        for (open, &(t, _, j)) in proof.checks.iter_mut().zip(points.iter()) {
            let (ti, ji) = (t as usize, j as usize);
            let off = walk.offsets[ti - 1] as usize;
            open.tile_col = blob[off * POM_V3_TILE_BYTES + ji * POM_V3_D..off * POM_V3_TILE_BYTES + (ji + 1) * POM_V3_D].to_vec();
            let (_, _, col_path) =
                rt_subtree_path(leaves, off as u64 * POM_V3_TILE_CHUNKS + ji as u64 * POM_V3_COL_CHUNKS, COL_SUBTREE_DEPTH);
            open.col_path = col_path;
            let (_, _, snippet_path) = rt_subtree_path(leaves, off as u64 * POM_V3_TILE_CHUNKS, 0);
            open.snippet_path = snippet_path;
        }
        proof.snippets = walk.snippets.clone();
        proof.clone()
    }

    #[test]
    fn chain_shift_lie_rejected() {
        // A9: lie about one snippet to redirect the offset chain — the snippet-vs-R_T check
        // at the list-derived offset must fail wherever sampled (here: lie about ALL of them,
        // pointing the whole chain elsewhere).
        let (blob, leaves, root, n_chunks, _) = honest_setup();
        let walk = v3_walk(SEED, &blob).unwrap();
        let mut lied_snippets = walk.snippets.clone();
        for s in lied_snippets.iter_mut() {
            s[0] ^= 1;
        }
        let roots: Vec<[u8; 32]> = walk.states.iter().map(v3_state_root).collect();
        let points = v3_check_points(&PPH, NONCE, &roots, &lied_snippets);
        let mut proof = PomProofV3 {
            tier: 0,
            roots,
            snippets: lied_snippets,
            checks: vec![
                PomV3Opening {
                    row_before: vec![0; POM_V3_D],
                    path_before: vec![],
                    row_after: vec![0; POM_V3_D],
                    path_after: vec![],
                    tile_col: vec![0; POM_V3_D],
                    col_path: vec![],
                    snippet_path: vec![],
                };
                POM_V3_CHECKS
            ],
        };
        for (open, &(t, x, _)) in proof.checks.iter_mut().zip(points.iter()) {
            let (ti, xi) = (t as usize, x as usize);
            open.row_before = walk.states[ti - 1][xi * POM_V3_D..(xi + 1) * POM_V3_D].to_vec();
            open.path_before = v3_state_row_path(&walk.states[ti - 1], xi);
            open.row_after = walk.states[ti][xi * POM_V3_D..(xi + 1) * POM_V3_D].to_vec();
            open.path_after = v3_state_row_path(&walk.states[ti], xi);
        }
        let full = v3_prove_at_points(&points, &walk, &blob, &leaves, &mut proof.clone());
        let mut shifted = full;
        shifted.snippets = proof.snippets.clone();
        assert!(verify_pom_proof_v3(&PPH, NONCE, SEED, &shifted, &root, n_chunks).is_err());
    }

    #[test]
    fn tampered_weight_byte_rejected() {
        // A7-adjacent: a prover whose blob differs by one byte in an opened column fails the
        // range proof against the true R_T.
        let (blob, leaves, root, n_chunks, proof) = honest_setup();
        let mut bad = proof.clone();
        bad.checks[0].tile_col[17] ^= 0xFF;
        assert!(matches!(
            verify_pom_proof_v3(&PPH, NONCE, SEED, &bad, &root, n_chunks),
            Err(PomV3VerifyError::BadColumnPath) | Err(PomV3VerifyError::TransitionMismatch)
        ));
        let _ = (blob, leaves);
    }

    #[test]
    fn forged_initial_root_rejected() {
        let (_, _, root, n_chunks, proof) = honest_setup();
        let mut bad = proof.clone();
        bad.roots[0][0] ^= 1;
        assert_eq!(verify_pom_proof_v3(&PPH, NONCE, SEED, &bad, &root, n_chunks), Err(PomV3VerifyError::BadInitialRoot));
    }

    #[test]
    fn wrong_shape_rejected() {
        let (_, _, root, n_chunks, proof) = honest_setup();
        let mut bad = proof.clone();
        bad.roots.pop();
        assert_eq!(verify_pom_proof_v3(&PPH, NONCE, SEED, &bad, &root, n_chunks), Err(PomV3VerifyError::WrongShape));
        let mut bad = proof.clone();
        bad.snippets.pop();
        assert_eq!(verify_pom_proof_v3(&PPH, NONCE, SEED, &bad, &root, n_chunks), Err(PomV3VerifyError::WrongShape));
    }

    #[test]
    fn container_verifies_and_round_trips_wire() {
        use crate::pom::PomProof;
        let (_, _, root, n_chunks, v3) = honest_setup();
        let final_state = fold64(&v3.roots[POM_V3_K]);
        let final_hash = |s: u64| blake(&s.to_le_bytes());
        let container = PomProof {
            tier: v3.tier,
            trace_root: [0u8; 32],
            pow_value: final_hash(final_state),
            final_state,
            initial_trace_path: vec![],
            final_trace_path: vec![],
            openings: vec![],
            steps_v2: None,
            v3: Some(v3),
        };
        let target = [0xFFu8; 32]; // trivial target: this test is not about the lottery
        assert_eq!(
            verify_pom_proof_v3_container(&PPH, NONCE, SEED, &container, n_chunks, &root, &target, final_hash),
            Ok(())
        );
        // Era-exact wire round-trip: a v3-bearing proof survives encode/decode; the decoded
        // struct is identical (same verify result).
        let decoded = PomProof::from_wire_bytes(&container.to_wire_bytes()).unwrap();
        assert_eq!(
            verify_pom_proof_v3_container(&PPH, NONCE, SEED, &decoded, n_chunks, &root, &target, final_hash),
            Ok(())
        );
        // A v3-era proof with impossible target is rejected.
        assert_eq!(
            verify_pom_proof_v3_container(&PPH, NONCE, SEED, &container, n_chunks, &root, &[0u8; 32], final_hash),
            Err(PomV3VerifyError::TargetNotMet)
        );
        // Smuggled legacy content is rejected.
        let mut smuggled = container.clone();
        smuggled.trace_root = [1u8; 32];
        assert_eq!(
            verify_pom_proof_v3_container(&PPH, NONCE, SEED, &smuggled, n_chunks, &root, &target, final_hash),
            Err(PomV3VerifyError::NonCanonicalLegacyFields)
        );
        // A proof without the v3 witness is rejected at the gate.
        let mut naked = container.clone();
        naked.v3 = None;
        assert_eq!(
            verify_pom_proof_v3_container(&PPH, NONCE, SEED, &naked, n_chunks, &root, &target, final_hash),
            Err(PomV3VerifyError::MissingV3)
        );
    }

}
