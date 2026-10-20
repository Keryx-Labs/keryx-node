//! Proof-of-Model v5: the v4 walk with a per-tier context pad. The miner GPU kernel must
//! reproduce this module byte-for-byte.

use crate::pom::{blake, mix64};
use crate::pom_v3::fold64;
use crate::pom_v4::{
    finish_container, v4_check_tiles, v4_container_witness, v4_initial_state_into, v4_state_root, v4_transition_into, PomProofV4,
    PomV4VerifyError, POM_V4_D, POM_V4_K, POM_V4_TILE_BYTES,
};

/// One pad read every `POM_V5_P_STRIDE` steps.
pub const POM_V5_P_STRIDE: usize = 4;
/// Bytes of one pad entry.
pub const POM_V5_P_ENTRY_BYTES: usize = POM_V4_D;
/// ARX rounds per pad entry.
pub const POM_V5_P_ROUNDS: u32 = 61_440;
/// Domain separation of the pad index.
pub const POM_V5_P_SALT: u64 = 0x0850_3190_C199_C415;

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

fn pad_entry_rounds(pad_seed: &[u8; 32], index: u64, rounds: u32) -> [u8; POM_V5_P_ENTRY_BYTES] {
    let w = |i: usize| u64::from_le_bytes(pad_seed[i * 8..i * 8 + 8].try_into().unwrap());
    let (mut a, mut b, mut c, mut d) = (w(0) ^ PAD_IV[0], w(1) ^ index ^ PAD_IV[1], w(2) ^ PAD_IV[2], w(3) ^ PAD_IV[3]);
    for _ in 0..rounds {
        round4(&mut a, &mut b, &mut c, &mut d);
    }
    let mut out = [0u8; POM_V5_P_ENTRY_BYTES];
    out[0..8].copy_from_slice(&a.to_le_bytes());
    out[8..16].copy_from_slice(&b.to_le_bytes());
    out[16..24].copy_from_slice(&c.to_le_bytes());
    out[24..32].copy_from_slice(&d.to_le_bytes());
    out
}

/// Pad entry `index` of the pad seeded by `pad_seed`.
pub fn v5_pad_entry(pad_seed: &[u8; 32], index: u64) -> [u8; POM_V5_P_ENTRY_BYTES] {
    pad_entry_rounds(pad_seed, index, POM_V5_P_ROUNDS)
}

/// State row and pad index read after the transition of `step`, when `step` is a pad step.
pub fn v5_pad_read(seed: u64, state: &[u8], step: u32, n_pad: u64) -> Option<(usize, u64)> {
    if step as usize % POM_V5_P_STRIDE != 0 {
        return None;
    }
    let row = (step as usize / POM_V5_P_STRIDE) % POM_V4_D;
    let r = &state[row * POM_V4_D..(row + 1) * POM_V4_D];
    let h = (0..4).fold(0u64, |h, i| h ^ u64::from_le_bytes(r[i * 8..i * 8 + 8].try_into().unwrap()));
    Some((row, mix64(seed ^ h ^ (step as u64).wrapping_mul(POM_V5_P_SALT)) % n_pad))
}

/// `final_state` of the v5 walk over the K `tiles` read in step order, pad entries supplied by
/// `pad`.
pub fn v5_final_state<F: FnMut(u64) -> [u8; POM_V5_P_ENTRY_BYTES]>(seed: u64, tiles: &[Vec<u8>], n_pad: u64, mut pad: F) -> u64 {
    let mut a = [0u8; POM_V4_TILE_BYTES];
    let mut b = [0u8; POM_V4_TILE_BYTES];
    v4_initial_state_into(&mut a, seed);
    let mut src_is_a = true;
    for step in 1..=POM_V4_K {
        let tile = &tiles[step - 1];
        let dst: &mut [u8] = if src_is_a {
            v4_transition_into(&mut b, &a, tile, step as u32);
            &mut b
        } else {
            v4_transition_into(&mut a, &b, tile, step as u32);
            &mut a
        };
        if let Some((row, idx)) = v5_pad_read(seed, dst, step as u32, n_pad) {
            let entry = pad(idx);
            for (s, e) in dst[row * POM_V4_D..(row + 1) * POM_V4_D].iter_mut().zip(entry.iter()) {
                *s ^= e;
            }
        }
        src_is_a = !src_is_a;
    }
    let final_buf: &[u8] = if src_is_a { &a } else { &b };
    fold64(&v4_state_root(final_buf))
}

/// Verifies the tiles of `proof` against `R_T` and re-walks with the context pad; returns the
/// derived `final_state`.
pub fn verify_pom_proof_v5(
    seed: u64,
    proof: &PomProofV4,
    r_t: &[u8; 32],
    n_chunks: u64,
    pad_seed: &[u8; 32],
    n_pad: u64,
) -> Result<u64, PomV4VerifyError> {
    if n_pad == 0 {
        return Err(PomV4VerifyError::NoContextPad);
    }
    v4_check_tiles(seed, proof, r_t, n_chunks)?;
    Ok(v5_final_state(seed, &proof.tiles, n_pad, |i| v5_pad_entry(pad_seed, i)))
}

/// Full v5 validation of a container `PomProof`: the v4 container rules with the v5 re-walk.
#[allow(clippy::too_many_arguments)]
pub fn verify_pom_proof_v5_container<Hf: Fn(u64) -> [u8; 32]>(
    seed: u64,
    proof: &crate::pom::PomProof,
    n_chunks: u64,
    r_t: &[u8; 32],
    pad_seed: &[u8; 32],
    n_pad: u64,
    target: &[u8; 32],
    final_hash: Hf,
) -> Result<(), PomV4VerifyError> {
    let v4 = v4_container_witness(proof)?;
    let derived = verify_pom_proof_v5(seed, v4, r_t, n_chunks, pad_seed, n_pad)?;
    finish_container(proof, derived, target, final_hash)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pom_v4::{v4_prove, verify_pom_proof_v4, POM_V4_CHUNK_BYTES};

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

    #[test]
    fn pad_entry_is_deterministic_and_index_bound() {
        let seed = v5_pad_seed(1, &[7; 32], &[9; 32]);
        assert_eq!(v5_pad_entry(&seed, 5), v5_pad_entry(&seed, 5));
        assert_ne!(v5_pad_entry(&seed, 5), v5_pad_entry(&seed, 6));
        assert_ne!(seed, v5_pad_seed(2, &[7; 32], &[9; 32]));
        assert_ne!(pad_entry_rounds(&seed, 5, POM_V5_P_ROUNDS), pad_entry_rounds(&seed, 5, POM_V5_P_ROUNDS - 1));
    }

    fn hex(b: &[u8]) -> String {
        b.iter().map(|x| format!("{x:02x}")).collect()
    }

    #[test]
    fn pad_vectors_shared_with_the_miner() {
        let seed = v5_pad_seed(1, &[7; 32], &[9; 32]);
        assert_eq!(hex(&seed), "cb678ed7a91fb53d40ad3b966d0219e92c75ead7761d6404bd533cc7ddf34f96");
        assert_eq!(hex(&v5_pad_entry(&seed, 5)), "f849c84e259e59e8fef1d78da4de56d6f9df69aad05f571404d9a6372b35911c");
        let state: Vec<u8> = (0..(POM_V4_D * POM_V4_D) as u32).map(|i| (i * 37 + 11) as u8).collect();
        assert_eq!(v5_pad_read(0xABCD, &state, 12, 1_000_003), Some((3, 172_680)));
        let tiles: Vec<Vec<u8>> =
            (0..POM_V4_K as u32).map(|t| (0..POM_V4_TILE_BYTES as u32).map(|b| (t * 31 + b * 7 + 3) as u8).collect()).collect();
        assert_eq!(v5_final_state(0x55AA, &tiles, 4_099, |i| v5_pad_entry(&seed, i)), 10_975_683_372_132_145_722);
    }

    #[test]
    fn pad_is_read_every_stride_steps() {
        let state = [0x5Au8; POM_V4_TILE_BYTES];
        let reads = (1..=POM_V4_K as u32).filter(|&s| v5_pad_read(3, &state, s, 1_000).is_some()).count();
        assert_eq!(reads, POM_V4_K / POM_V5_P_STRIDE);
        let (row, idx) = v5_pad_read(3, &state, 8, 1_000).unwrap();
        assert_eq!(row, 2);
        assert!(idx < 1_000);
    }

    #[test]
    fn honest_proof_verifies_and_depends_on_the_pad() {
        let (blob, leaves, r_t) = test_blob(64);
        let seed = 0x1357_9BDF_2468_ACE0;
        let pad_seed = v5_pad_seed(0, &[1; 32], &r_t);
        let n_pad = 4_096;
        let proof = v4_prove(seed, 0, &blob, &leaves).unwrap();

        let table: Vec<[u8; 32]> = (0..n_pad).map(|i| v5_pad_entry(&pad_seed, i)).collect();
        let resident = v5_final_state(seed, &proof.tiles, n_pad, |i| table[i as usize]);
        let verified = verify_pom_proof_v5(seed, &proof, &r_t, n_chunks(&blob), &pad_seed, n_pad).unwrap();
        assert_eq!(verified, resident);

        assert_ne!(verified, verify_pom_proof_v4(seed, &proof, &r_t, n_chunks(&blob)).unwrap());
        let other_seed = v5_pad_seed(1, &[1; 32], &r_t);
        assert_ne!(verified, verify_pom_proof_v5(seed, &proof, &r_t, n_chunks(&blob), &other_seed, n_pad).unwrap());
        let mut one_wrong = table.clone();
        let mut touched = Vec::new();
        v5_final_state(seed, &proof.tiles, n_pad, |i| {
            touched.push(i);
            table[i as usize]
        });
        one_wrong[touched[POM_V4_K / POM_V5_P_STRIDE - 1] as usize][0] ^= 1;
        assert_ne!(verified, v5_final_state(seed, &proof.tiles, n_pad, |i| one_wrong[i as usize]));
    }

    #[test]
    fn tampered_tile_and_missing_pad_are_rejected() {
        let (blob, leaves, r_t) = test_blob(64);
        let seed = 0x2468;
        let pad_seed = v5_pad_seed(0, &[1; 32], &r_t);
        let mut proof = v4_prove(seed, 0, &blob, &leaves).unwrap();
        assert_eq!(verify_pom_proof_v5(seed, &proof, &r_t, n_chunks(&blob), &pad_seed, 0), Err(PomV4VerifyError::NoContextPad));
        proof.tiles[3][0] ^= 1;
        assert_eq!(verify_pom_proof_v5(seed, &proof, &r_t, n_chunks(&blob), &pad_seed, 64), Err(PomV4VerifyError::BadTilePath));
    }

    #[test]
    fn container_pins_the_v5_final_state() {
        let (blob, leaves, r_t) = test_blob(64);
        let seed = 0x5151;
        let pph = [3u8; 32];
        let pad_seed = v5_pad_seed(2, &[4; 32], &r_t);
        let n_pad = 1_024;
        let v4 = v4_prove(seed, 2, &blob, &leaves).unwrap();
        let fs = verify_pom_proof_v5(seed, &v4, &r_t, n_chunks(&blob), &pad_seed, n_pad).unwrap();
        let fs_v4 = verify_pom_proof_v4(seed, &v4, &r_t, n_chunks(&blob)).unwrap();
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
            v4: Some(v4.clone()),
            task: None,
        };
        let target = [0xffu8; 32];
        let check = |p: &crate::pom::PomProof| {
            verify_pom_proof_v5_container(seed, p, n_chunks(&blob), &r_t, &pad_seed, n_pad, &target, final_hash)
        };
        assert!(check(&container(fs)).is_ok());
        assert_eq!(check(&container(fs_v4)), Err(PomV4VerifyError::FinalStateMismatch));
    }
}
