//! Proof-of-Model v5: the v4 walk over the tier's weight tiles and its context-pad tiles. The miner
//! GPU kernel must reproduce this module byte-for-byte.

use crate::pom::{blake, mix64, verify_merkle};
use crate::pom_v3::{fold64, snippet_fold};
use crate::pom_v4::{
    check_pow_pin, finish_container, v4_container_witness, v4_first_offset, v4_initial_state_into, v4_n_tiles, v4_state_root,
    v4_tile_subtree_root, v4_transition_into, PomProofV4, PomV4VerifyError, POM_V4_D, POM_V4_K, POM_V4_OFFSET_STEP_SALT,
    POM_V4_SNIPPET_BYTES, POM_V4_TILE_BYTES,
};

/// Bytes of one pad entry.
pub const POM_V5_P_ENTRY_BYTES: usize = 32;
/// Pad entries per pad tile.
pub const POM_V5_TILE_ENTRIES: u64 = (POM_V4_TILE_BYTES / POM_V5_P_ENTRY_BYTES) as u64;
/// ARX rounds per pad entry.
pub const POM_V5_P_ROUNDS: u32 = 1_920;

const PAD_DOMAIN: &[u8] = b"KERYX-POM-V5-PAD";
const PAD_IV: [u64; 4] = [0x510e_527f_ade6_82d1, 0x9b05_688c_2b3e_6c1f, 0x1f83_d9ab_fb41_bd6b, 0x5be0_cd19_137e_2179];

/// Seed of the context pad of `tier`, bound to its model and weight root.
pub fn v5_pad_seed(tier: u8, model_id: &[u8; 32], r_t: &[u8; 32]) -> [u8; 32] {
    let mut buf = Vec::with_capacity(PAD_DOMAIN.len() + 1 + 64);
    buf.extend_from_slice(PAD_DOMAIN);
    buf.push(tier);
    buf.extend_from_slice(model_id);
    buf.extend_from_slice(r_t);
    blake(&buf)
}

#[inline(always)]
fn round4(a: &mut u64, b: &mut u64, c: &mut u64, d: &mut u64) {
    *a = a.wrapping_add(*b);
    *d ^= *a;
    *d = d.rotate_left(32);
    *c = c.wrapping_add(*d);
    *b ^= *c;
    *b = b.rotate_left(24);
    *a = a.wrapping_add(*b);
    *d ^= *a;
    *d = d.rotate_left(16);
    *c = c.wrapping_add(*d);
    *b ^= *c;
    *b = b.rotate_left(63);
}

fn pad_entry_into(pad_seed: &[u8; 32], index: u64, rounds: u32, out: &mut [u8]) {
    let w = |i: usize| u64::from_le_bytes(pad_seed[i * 8..i * 8 + 8].try_into().unwrap());
    let (mut a, mut b, mut c, mut d) = (w(0) ^ PAD_IV[0], w(1) ^ index ^ PAD_IV[1], w(2) ^ PAD_IV[2], w(3) ^ PAD_IV[3]);
    for _ in 0..rounds {
        round4(&mut a, &mut b, &mut c, &mut d);
    }
    out[0..8].copy_from_slice(&a.to_le_bytes());
    out[8..16].copy_from_slice(&b.to_le_bytes());
    out[16..24].copy_from_slice(&c.to_le_bytes());
    out[24..32].copy_from_slice(&d.to_le_bytes());
}

/// Pad entry `index` of the pad seeded by `pad_seed`.
pub fn v5_pad_entry(pad_seed: &[u8; 32], index: u64) -> [u8; POM_V5_P_ENTRY_BYTES] {
    let mut out = [0u8; POM_V5_P_ENTRY_BYTES];
    pad_entry_into(pad_seed, index, POM_V5_P_ROUNDS, &mut out);
    out
}

/// Pad tile `tile`: pad entries `tile * POM_V5_TILE_ENTRIES ..` in order.
pub fn v5_pad_tile(pad_seed: &[u8; 32], tile: u64) -> Vec<u8> {
    let mut out = vec![0u8; POM_V4_TILE_BYTES];
    for (k, entry) in out.chunks_exact_mut(POM_V5_P_ENTRY_BYTES).enumerate() {
        pad_entry_into(pad_seed, tile * POM_V5_TILE_ENTRIES + k as u64, POM_V5_P_ROUNDS, entry);
    }
    out
}

fn v5_pad_steps(n_weight_tiles: u64, n_pad_tiles: u64) -> u64 {
    let k = POM_V4_K as u64;
    let n = n_weight_tiles + n_pad_tiles;
    ((k * n_pad_tiles + n / 2) / n).clamp(1, k - 1)
}

fn v5_is_pad_step(step: u64, pad_steps: u64) -> bool {
    let k = POM_V4_K as u64;
    (step * pad_steps) / k != ((step - 1) * pad_steps) / k
}

fn v5_state_fold(state: &[u8]) -> u64 {
    let mut h = 0u64;
    for r in 0..POM_V4_D {
        let w = u64::from_le_bytes(state[r * POM_V4_D..r * POM_V4_D + 8].try_into().unwrap());
        h ^= mix64(w ^ r as u64);
    }
    h
}

fn v5_first_offset(seed: u64, n_weight_tiles: u64) -> u64 {
    v4_first_offset(seed, n_weight_tiles)
}

fn v5_next_offset(
    seed: u64,
    step: u64,
    snippet: &[u8; POM_V4_SNIPPET_BYTES],
    state_fold: u64,
    n_weight_tiles: u64,
    n_pad_tiles: u64,
    pad_steps: u64,
) -> u64 {
    let raw = mix64(seed ^ (step + 1).wrapping_mul(POM_V4_OFFSET_STEP_SALT) ^ snippet_fold(snippet) ^ state_fold);
    if v5_is_pad_step(step + 1, pad_steps) {
        n_weight_tiles + raw % n_pad_tiles
    } else {
        raw % n_weight_tiles
    }
}

/// Runs the v5 walk and returns its `final_state`.
fn v5_walk<F>(seed: u64, n_weight_tiles: u64, n_pad_tiles: u64, pad_seed: &[u8; 32], mut fetch: F) -> Result<u64, PomV4VerifyError>
where
    F: FnMut(usize, u64) -> Result<Option<Vec<u8>>, PomV4VerifyError>,
{
    let pad_steps = v5_pad_steps(n_weight_tiles, n_pad_tiles);
    let mut states = [[0u8; POM_V4_TILE_BYTES]; 2];
    v4_initial_state_into(&mut states[0], seed);
    let mut off = v5_first_offset(seed, n_weight_tiles);
    for step in 1..=POM_V4_K {
        let tile = match fetch(step - 1, off)? {
            Some(t) => t,
            None => v5_pad_tile(pad_seed, off - n_weight_tiles),
        };
        let (lo, hi) = states.split_at_mut(1);
        let (cur, next) = if step % 2 == 1 { (&lo[0], &mut hi[0]) } else { (&hi[0], &mut lo[0]) };
        if step < POM_V4_K {
            let snippet: [u8; POM_V4_SNIPPET_BYTES] = tile[..POM_V4_SNIPPET_BYTES].try_into().unwrap();
            off = v5_next_offset(seed, step as u64, &snippet, v5_state_fold(cur), n_weight_tiles, n_pad_tiles, pad_steps);
        }
        v4_transition_into(next, cur, &tile, step as u32);
    }
    Ok(fold64(&v4_state_root(&states[POM_V4_K % 2])))
}

fn verify_weight_paths(proof: &PomProofV4, weights: &[(usize, u64)], r_t: &[u8; 32]) -> bool {
    let ok = |&(i, off): &(usize, u64)| verify_merkle(v4_tile_subtree_root(&proof.tiles[i]), off, &proof.merkle[i].path, r_t);
    #[cfg(not(target_arch = "wasm32"))]
    {
        use rayon::prelude::*;
        weights.par_iter().all(ok)
    }
    #[cfg(target_arch = "wasm32")]
    {
        weights.iter().all(ok)
    }
}

fn check_tile_space(n_weight_tiles: u64, n_pad_tiles: u64) -> Result<(), PomV4VerifyError> {
    if n_pad_tiles == 0 {
        return Err(PomV4VerifyError::NoContextPad);
    }
    if n_weight_tiles == 0 {
        return Err(PomV4VerifyError::BlobTooSmall);
    }
    Ok(())
}

/// Verifies `proof` against `R_T` and the context pad and returns the derived `final_state`.
pub fn verify_pom_proof_v5(
    seed: u64,
    proof: &PomProofV4,
    r_t: &[u8; 32],
    n_chunks: u64,
    pad_seed: &[u8; 32],
    n_pad_tiles: u64,
) -> Result<u64, PomV4VerifyError> {
    if proof.tiles.len() != POM_V4_K || proof.merkle.len() != POM_V4_K {
        return Err(PomV4VerifyError::WrongShape);
    }
    let n_weight_tiles = v4_n_tiles(n_chunks);
    check_tile_space(n_weight_tiles, n_pad_tiles)?;
    let mut weights = Vec::with_capacity(POM_V4_K);
    let final_state = v5_walk(seed, n_weight_tiles, n_pad_tiles, pad_seed, |i, off| {
        if off < n_weight_tiles {
            if proof.tiles[i].len() != POM_V4_TILE_BYTES {
                return Err(PomV4VerifyError::WrongTileShape);
            }
            weights.push((i, off));
            Ok(Some(proof.tiles[i].clone()))
        } else {
            if !proof.tiles[i].is_empty() || !proof.merkle[i].path.is_empty() {
                return Err(PomV4VerifyError::WrongTileShape);
            }
            Ok(None)
        }
    })?;
    if !verify_weight_paths(proof, &weights, r_t) {
        return Err(PomV4VerifyError::BadTilePath);
    }
    Ok(final_state)
}

/// Full v5 validation of a container `PomProof`: the v4 container rules with the v5 walk.
#[allow(clippy::too_many_arguments)]
pub fn verify_pom_proof_v5_container<Hf: Fn(u64) -> [u8; 32]>(
    seed: u64,
    proof: &crate::pom::PomProof,
    n_chunks: u64,
    r_t: &[u8; 32],
    pad_seed: &[u8; 32],
    n_pad_tiles: u64,
    target: &[u8; 32],
    final_hash: Hf,
) -> Result<(), PomV4VerifyError> {
    let v4 = v4_container_witness(proof)?;
    check_pow_pin(proof, target, &final_hash)?;
    let derived = verify_pom_proof_v5(seed, v4, r_t, n_chunks, pad_seed, n_pad_tiles)?;
    finish_container(proof, derived, target, final_hash)
}

/// Reference prover over an in-memory weight blob and its chunk leaves.
pub fn v5_prove(
    seed: u64,
    tier: u8,
    blob: &[u8],
    leaf_hashes: &[[u8; 32]],
    pad_seed: &[u8; 32],
    n_pad_tiles: u64,
) -> Result<(PomProofV4, u64), PomV4VerifyError> {
    let n_weight_tiles = v4_n_tiles((blob.len() / crate::pom_v4::POM_V4_CHUNK_BYTES) as u64);
    check_tile_space(n_weight_tiles, n_pad_tiles)?;
    let mut proof = PomProofV4 { tier, tiles: Vec::with_capacity(POM_V4_K), merkle: Vec::with_capacity(POM_V4_K) };
    let final_state = v5_walk(seed, n_weight_tiles, n_pad_tiles, pad_seed, |_, off| {
        if off < n_weight_tiles {
            let t = blob[(off as usize) * POM_V4_TILE_BYTES..(off as usize + 1) * POM_V4_TILE_BYTES].to_vec();
            proof.tiles.push(t.clone());
            proof.merkle.push(crate::pom_v4::PomV4RangeProof { path: crate::pom_v4::v4_tile_path(leaf_hashes, off) });
            Ok(Some(t))
        } else {
            proof.tiles.push(Vec::new());
            proof.merkle.push(crate::pom_v4::PomV4RangeProof { path: Vec::new() });
            Ok(None)
        }
    })?;
    Ok((proof, final_state))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pom_v4::{verify_pom_proof_v4, POM_V4_CHUNK_BYTES};

    fn test_blob(n_tiles: usize) -> (Vec<u8>, Vec<[u8; 32]>, [u8; 32]) {
        let mut blob = vec![0u8; n_tiles * POM_V4_TILE_BYTES + 3 * POM_V4_CHUNK_BYTES];
        let mut h = 0xC0FFEEu64;
        for b in blob.iter_mut() {
            h = mix64(h);
            *b = h as u8;
        }
        let leaves: Vec<[u8; 32]> = blob.chunks(POM_V4_CHUNK_BYTES).map(blake).collect();
        let mut level = leaves.clone();
        while level.len() > 1 {
            level = crate::pom_v4::fold_level(&level);
        }
        (blob, leaves, level[0])
    }

    fn n_chunks(blob: &[u8]) -> u64 {
        (blob.len() / POM_V4_CHUNK_BYTES) as u64
    }

    fn hex(b: &[u8]) -> String {
        b.iter().map(|x| format!("{x:02x}")).collect()
    }

    #[test]
    fn pad_vectors_shared_with_the_miner() {
        let seed = v5_pad_seed(1, &[7; 32], &[9; 32]);
        assert_eq!(hex(&seed), "cb678ed7a91fb53d40ad3b966d0219e92c75ead7761d6404bd533cc7ddf34f96");
        assert_eq!(hex(&v5_pad_entry(&seed, 5)), "0601327f5974ca61e9a78cd447afdda4cd64050bac6ddf552ce1a165f04727e0");
        assert_eq!(hex(&blake(&v5_pad_tile(&seed, 3))), "fa8d888e763b63542c23ad2a3e2ac715bf180f51f6b0c357a0399595593d55c2");
        let (blob, leaves, r_t) = test_blob(64);
        let (proof, fs) = v5_prove(0x55AA, 1, &blob, &leaves, &seed, 61).unwrap();
        assert_eq!(verify_pom_proof_v5(0x55AA, &proof, &r_t, n_chunks(&blob), &seed, 61), Ok(fs));
        assert_eq!(fs, 5_477_356_865_520_795_736);
    }

    #[test]
    fn pad_tile_is_its_entries_in_order() {
        let seed = v5_pad_seed(0, &[1; 32], &[2; 32]);
        let tile = v5_pad_tile(&seed, 7);
        for k in 0..POM_V5_TILE_ENTRIES {
            assert_eq!(&tile[k as usize * 32..k as usize * 32 + 32], &v5_pad_entry(&seed, 7 * POM_V5_TILE_ENTRIES + k));
        }
    }

    #[test]
    fn honest_proof_reads_both_kinds_of_tiles_and_verifies() {
        let (blob, leaves, r_t) = test_blob(64);
        let pad_seed = v5_pad_seed(0, &[1; 32], &r_t);
        let seed = 0x1357_9BDF_2468_ACE0;
        let (proof, fs) = v5_prove(seed, 0, &blob, &leaves, &pad_seed, 64).unwrap();
        let pad_steps = v5_pad_steps(64, 64);
        assert_eq!(pad_steps, 128);
        for (i, t) in proof.tiles.iter().enumerate() {
            assert_eq!(t.is_empty(), v5_is_pad_step(i as u64 + 1, pad_steps), "step {} reads the wrong kind of tile", i + 1);
        }
        assert_eq!(verify_pom_proof_v5(seed, &proof, &r_t, n_chunks(&blob), &pad_seed, 64), Ok(fs));
        assert!(verify_pom_proof_v4(seed, &proof, &r_t, n_chunks(&blob)).is_err());
        let other = v5_pad_seed(1, &[1; 32], &r_t);
        assert_ne!(verify_pom_proof_v5(seed, &proof, &r_t, n_chunks(&blob), &other, 64), Ok(fs));
    }

    #[test]
    fn pad_steps_follow_the_pad_share() {
        for (n_w, n_p) in [(7_566_000u64, 1_007_616u64), (5_670_000, 703_488), (15_700_000, 1_807_360), (1_000, 1)] {
            let k = v5_pad_steps(n_w, n_p);
            let share = POM_V4_K as f64 * n_p as f64 / (n_w + n_p) as f64;
            assert!((k as f64 - share).abs() <= 1.0, "{k} pad steps for a share of {share}");
            assert_eq!((1..=POM_V4_K as u64).filter(|&s| v5_is_pad_step(s, k)).count() as u64, k);
            assert!(!v5_is_pad_step(1, k));
        }
    }

    #[test]
    fn malformed_proofs_are_rejected() {
        let (blob, leaves, r_t) = test_blob(64);
        let pad_seed = v5_pad_seed(0, &[1; 32], &r_t);
        let seed = 0x2468;
        let (proof, _) = v5_prove(seed, 0, &blob, &leaves, &pad_seed, 64).unwrap();
        let n = n_chunks(&blob);
        assert_eq!(verify_pom_proof_v5(seed, &proof, &r_t, n, &pad_seed, 0), Err(PomV4VerifyError::NoContextPad));

        let w = proof.tiles.iter().position(|t| !t.is_empty()).unwrap();
        let mut bad = proof.clone();
        bad.tiles[w][0] ^= 1;
        assert!(verify_pom_proof_v5(seed, &bad, &r_t, n, &pad_seed, 64).is_err());

        let p = proof.tiles.iter().position(|t| t.is_empty()).unwrap();
        let mut filled = proof.clone();
        filled.tiles[p] = v5_pad_tile(&pad_seed, 0);
        assert_eq!(verify_pom_proof_v5(seed, &filled, &r_t, n, &pad_seed, 64), Err(PomV4VerifyError::WrongTileShape));

        let mut pathed = proof.clone();
        pathed.merkle[p].path = proof.merkle[w].path.clone();
        assert_eq!(verify_pom_proof_v5(seed, &pathed, &r_t, n, &pad_seed, 64), Err(PomV4VerifyError::WrongTileShape));

        let v4 = crate::pom_v4::v4_prove(seed, 0, &blob, &leaves).unwrap();
        assert!(verify_pom_proof_v5(seed, &v4, &r_t, n, &pad_seed, 64).is_err());

        let mut short = proof.clone();
        short.tiles.pop();
        assert_eq!(verify_pom_proof_v5(seed, &short, &r_t, n, &pad_seed, 64), Err(PomV4VerifyError::WrongShape));
    }

    #[test]
    fn tampered_last_weight_tile_fails_its_path() {
        let (blob, leaves, r_t) = test_blob(64);
        let pad_seed = v5_pad_seed(0, &[1; 32], &r_t);
        let seed = 0x9999;
        let (mut proof, _) = v5_prove(seed, 0, &blob, &leaves, &pad_seed, 64).unwrap();
        let w = proof.tiles.iter().rposition(|t| !t.is_empty()).unwrap();
        proof.tiles[w][POM_V4_TILE_BYTES - 1] ^= 1;
        assert_eq!(verify_pom_proof_v5(seed, &proof, &r_t, n_chunks(&blob), &pad_seed, 64), Err(PomV4VerifyError::BadTilePath));
    }

    #[test]
    fn container_pins_the_v5_final_state() {
        let (blob, leaves, r_t) = test_blob(64);
        let pad_seed = v5_pad_seed(2, &[4; 32], &r_t);
        let seed = 0x5151;
        let pph = [3u8; 32];
        let (v5, fs) = v5_prove(seed, 2, &blob, &leaves, &pad_seed, 32).unwrap();
        let final_hash = |s: u64| crate::pom::pom_pow_value_h3(s, &pph);
        let container = |final_state: u64| crate::pom::PomProof {
            tier: 2,
            trace_root: [0u8; 32],
            pow_value: final_hash(final_state),
            final_state,
            initial_trace_path: vec![],
            final_trace_path: vec![],
            openings: vec![],
            steps_v2: None,
            v3: None,
            v4: Some(v5.clone()),
            task: None,
        };
        let target = [0xffu8; 32];
        let check = |p: &crate::pom::PomProof| {
            verify_pom_proof_v5_container(seed, p, n_chunks(&blob), &r_t, &pad_seed, 32, &target, final_hash)
        };
        assert!(check(&container(fs)).is_ok());
        assert_eq!(check(&container(fs ^ 1)), Err(PomV4VerifyError::FinalStateMismatch));
        let mut forged = container(fs);
        forged.pow_value[0] ^= 1;
        let mut gutted = forged.clone();
        gutted.v4.as_mut().unwrap().tiles[0].clear();
        assert_eq!(check(&gutted), Err(PomV4VerifyError::PowValueMismatch), "pow_value is checked before the walk");
    }
}
