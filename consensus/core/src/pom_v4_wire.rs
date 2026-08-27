//! Compact p2p encoding of a v4 PoM proof: one Merkle multiproof instead of `POM_V4_K` independent
//! paths.
//!
//! All `POM_V4_K` tile paths climb to the SAME per-tier root `R_T`, so their upper levels are
//! shared, yet the canonical (borsh) form ships every path in full. Measured on the tier-0 shape
//! (`consensus/core/examples/pom_v4_size.rs`): 5 888 siblings on the wire against 3 505 distinct
//! ones actually needed — 40.5 % of the path field is redundant.
//!
//! Two kinds of redundancy are removed, and it is worth knowing which is which, because the gzip
//! already running on every p2p channel reaches only the first:
//!
//!   * a sibling that literally repeats one carried by another tile (deflate can back-reference it,
//!     but only within its 32 KiB window — about two thirds of them in practice);
//!   * a sibling that is itself an *ancestor* of some other tile, so the receiver can fold it from
//!     that tile's own subtree. These appear exactly once in the byte stream, so no compressor of
//!     any window size can ever remove them.
//!
//! **Nothing positional goes on the wire.** Tile offsets are re-derived from the block seed and the
//! tiles' leading snippets via [`v4_offset_chain`], and the tree shape from the tier's chunk count,
//! so both sides independently reach the same node ordering. That is what keeps the encoding at
//! pure payload.
//!
//! This is a transport encoding only: [`decode_v4_deduped`] reconstructs a byte-identical
//! [`PomProof`], which is what then gets verified, stored and re-served to peers that speak the
//! older format. Consensus never sees this module.

use crate::hashing::header::hash_override_nonce_time;
use crate::header::Header;
use crate::pom::{PomProof, hash_pair, pom_block_seed_v4};
use crate::pom_v4::{
    POM_V4_K, POM_V4_TILE_BYTES, PomProofV4, PomV4RangeProof, v4_n_tiles, v4_offset_chain, v4_tile_level_len, v4_tile_path_len,
    v4_tile_subtree_root,
};
use std::collections::BTreeMap;

/// Fixed part of the encoding: `tier | pow_value | final_state | node_count`.
const HEADER_BYTES: usize = 1 + 32 + 8 + 4;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PomWireError {
    /// Not a v4 proof, or the legacy/v3 container fields are non-empty. Only canonical v4 proofs
    /// have a compact form — everything the compact form omits is required to be empty.
    NotCanonicalV4,
    /// `tiles`/`merkle` length is not `POM_V4_K`, or a tile is not `POM_V4_TILE_BYTES`.
    WrongShape,
    /// A path length disagrees with the tree implied by the tier's chunk count.
    WrongPathLen,
    /// The blob is too small to hold a single tile.
    BlobTooSmall,
    /// Two tiles that meet at the same tree node disagree about the value above it — the proof is
    /// internally inconsistent and has no well-defined compact form.
    InconsistentPaths,
    /// Truncated, over-long, or otherwise malformed byte stream.
    Malformed,
    /// Tier index is not present in the v4 tier table.
    UnknownTier(u8),
}

/// Seed and canonical chunk count for a v4 block, from its header alone.
///
/// Both are what [`encode_v4_deduped`]/[`decode_v4_deduped`] need, and both are derivable by any
/// peer holding the header — which is why the compact form can omit them.
///
/// The tier table is pinned to `POM_TIERS_H6`: `pom_v4_activation` is strictly later than
/// `pom_v3_activation`, so every v4 block selects that table in `pom_tiers`. A tier outside it is
/// rejected here rather than guessed, and the caller falls back to the legacy encoding.
pub fn v4_wire_context(header: &Header, tier: u8) -> Result<(u64, u64), PomWireError> {
    let tiers = crate::config::params::POM_TIERS_H6;
    let t = tiers.get(tier as usize).ok_or(PomWireError::UnknownTier(tier))?;
    let pre_pow_hash = hash_override_nonce_time(header, 0, 0).as_bytes();
    Ok((pom_block_seed_v4(&pre_pow_hash, header.timestamp, header.nonce), t.chunks))
}

/// Tier byte of a compact encoding, read without decoding it.
///
/// The tier selects `n_chunks`, which the decode needs in order to know the tree shape — so it has
/// to be readable first. It is the first byte by construction; `decode_v4_deduped` re-reads and
/// range-checks it, so a bogus value here only costs a failed lookup.
pub fn deduped_tier(bytes: &[u8]) -> Option<u8> {
    bytes.first().copied()
}

/// True when every field the compact form does not carry is canonically empty, exactly as
/// `verify_pom_proof_v4_container` demands.
fn is_canonical_v4(proof: &PomProof) -> bool {
    proof.v4.is_some()
        && proof.trace_root == [0u8; 32]
        && proof.initial_trace_path.is_empty()
        && proof.final_trace_path.is_empty()
        && proof.openings.is_empty()
        && proof.steps_v2.is_none()
        && proof.v3.is_none()
}

fn shape_check(v4: &PomProofV4) -> Result<(), PomWireError> {
    if v4.tiles.len() != POM_V4_K || v4.merkle.len() != POM_V4_K {
        return Err(PomWireError::WrongShape);
    }
    if v4.tiles.iter().any(|t| t.len() != POM_V4_TILE_BYTES) {
        return Err(PomWireError::WrongShape);
    }
    Ok(())
}

/// Drive one bottom-up pass over the tree, calling `visit` once per level with the set of nodes
/// known at that level and the level's width.
///
/// Encoder and decoder must agree on the node ordering down to the last detail, so both go through
/// this one walker rather than reimplementing the traversal. The known-set is keyed by node index
/// and iterated ascending (`BTreeMap`), which *is* the canonical order.
///
/// `visit` receives `(level, level_len, known)` and returns the completed level — `known` plus every
/// sibling needed to fold it — from which the next level is derived.
fn walk_levels<F>(level0: BTreeMap<u64, [u8; 32]>, level0_len: u64, plen: usize, mut visit: F) -> Result<(), PomWireError>
where
    F: FnMut(usize, u64, &BTreeMap<u64, [u8; 32]>) -> Result<BTreeMap<u64, [u8; 32]>, PomWireError>,
{
    let mut cur = level0;
    let mut level_len = level0_len;
    for level in 0..plen {
        let full = visit(level, level_len, &cur)?;

        // Fold to the next level. `fold_level` pairs the odd tail with itself, so a right child
        // past the end reuses the left one.
        let mut next: BTreeMap<u64, [u8; 32]> = BTreeMap::new();
        for &idx in cur.keys() {
            let p = idx >> 1;
            if next.contains_key(&p) {
                continue;
            }
            let l = *full.get(&(p * 2)).ok_or(PomWireError::Malformed)?;
            let r = if p * 2 + 1 < level_len { *full.get(&(p * 2 + 1)).ok_or(PomWireError::Malformed)? } else { l };
            next.insert(p, hash_pair(&l, &r));
        }
        cur = next;
        level_len = level_len.div_ceil(2);
    }
    Ok(())
}

/// Level 0 of the walk: each tile's subtree root, keyed by tile index so repeated offsets collapse
/// to one node (both tiles carry the same bytes, so both fold to the same root).
///
/// Folding 256 tiles is ~16 k blake3 compressions and dominates both directions, so it runs on
/// rayon exactly as `verify_tile_merkle_parallel` does — sequential on wasm, which has no pool.
fn tile_level(offsets: &[u64; POM_V4_K], tiles: &[Vec<u8>]) -> BTreeMap<u64, [u8; 32]> {
    #[cfg(not(target_arch = "wasm32"))]
    let roots: Vec<[u8; 32]> = {
        use rayon::prelude::*;
        tiles.par_iter().map(|t| v4_tile_subtree_root(t)).collect()
    };
    #[cfg(target_arch = "wasm32")]
    let roots: Vec<[u8; 32]> = tiles.iter().map(|t| v4_tile_subtree_root(t)).collect();

    let mut level0: BTreeMap<u64, [u8; 32]> = BTreeMap::new();
    for (i, &off) in offsets.iter().enumerate() {
        level0.entry(off).or_insert(roots[i]);
    }
    level0
}

/// Whether `idx`'s sibling has to be supplied at a level of width `level_len`: not when it is the
/// self-paired odd tail (out of range), and not when another tile already carries it.
#[inline]
fn needs_sibling(idx: u64, level_len: u64, known: &BTreeMap<u64, [u8; 32]>) -> Option<u64> {
    let sib = idx ^ 1;
    if sib < level_len && !known.contains_key(&sib) { Some(sib) } else { None }
}

/// Encode a canonical v4 proof into the compact form.
///
/// `seed`/`n_chunks` come from [`v4_wire_context`]. Returns an error rather than a degraded encoding
/// for anything unexpected, so the caller can fall back to the canonical bytes; a proof whose paths
/// are structurally atypical (the verifier only checks that a path folds to `R_T`, not that it was
/// built canonically) is refused here instead of being silently mangled.
pub fn encode_v4_deduped(proof: &PomProof, seed: u64, n_chunks: u64) -> Result<Vec<u8>, PomWireError> {
    if !is_canonical_v4(proof) {
        return Err(PomWireError::NotCanonicalV4);
    }
    let v4 = proof.v4.as_ref().expect("checked by is_canonical_v4");
    shape_check(v4)?;

    let n_tiles = v4_n_tiles(n_chunks);
    if n_tiles == 0 {
        return Err(PomWireError::BlobTooSmall);
    }
    let level0_len = v4_tile_level_len(n_chunks);
    let plen = v4_tile_path_len(level0_len);
    if v4.merkle.iter().any(|m| m.path.len() != plen) {
        return Err(PomWireError::WrongPathLen);
    }

    let offsets = v4_offset_chain(seed, &v4.tiles, n_tiles);

    let level0 = tile_level(&offsets, &v4.tiles);

    let mut nodes: Vec<[u8; 32]> = Vec::new();
    walk_levels(level0, level0_len, plen, |level, level_len, known| {
        // What each tile claims the sibling of its ancestor is, at this level.
        let mut claimed: BTreeMap<u64, [u8; 32]> = BTreeMap::new();
        for (i, &off) in offsets.iter().enumerate() {
            let anc = off >> level;
            let v = v4.merkle[i].path[level];
            // Tiles meeting at the same ancestor must agree; otherwise the proof has no single
            // well-defined compact form and we refuse it.
            if let Some(prev) = claimed.insert(anc, v)
                && prev != v
            {
                return Err(PomWireError::InconsistentPaths);
            }
        }

        let mut full = known.clone();
        for &idx in known.keys() {
            if let Some(sib) = needs_sibling(idx, level_len, known) {
                let v = *claimed.get(&idx).ok_or(PomWireError::Malformed)?;
                nodes.push(v);
                full.insert(sib, v);
            }
        }
        Ok(full)
    })?;

    let mut out = Vec::with_capacity(HEADER_BYTES + POM_V4_K * POM_V4_TILE_BYTES + nodes.len() * 32);
    out.push(v4.tier);
    out.extend_from_slice(&proof.pow_value);
    out.extend_from_slice(&proof.final_state.to_le_bytes());
    out.extend_from_slice(&(nodes.len() as u32).to_le_bytes());
    for t in &v4.tiles {
        out.extend_from_slice(t);
    }
    for n in &nodes {
        out.extend_from_slice(n);
    }
    Ok(out)
}

/// Decode the compact form back into a byte-identical [`PomProof`].
///
/// The result is what gets verified, stored, and re-served to peers on the older format, so it must
/// match the prover's proof exactly — [`encode_v4_deduped`] callers are expected to confirm that by
/// round-tripping before they rely on it.
pub fn decode_v4_deduped(bytes: &[u8], seed: u64, n_chunks: u64) -> Result<PomProof, PomWireError> {
    if bytes.len() < HEADER_BYTES {
        return Err(PomWireError::Malformed);
    }
    let tier = bytes[0];
    let mut pow_value = [0u8; 32];
    pow_value.copy_from_slice(&bytes[1..33]);
    let final_state = u64::from_le_bytes(bytes[33..41].try_into().map_err(|_| PomWireError::Malformed)?);
    let node_count = u32::from_le_bytes(bytes[41..45].try_into().map_err(|_| PomWireError::Malformed)?) as usize;

    let tiles_bytes = POM_V4_K * POM_V4_TILE_BYTES;
    // Exact-length check: a compact proof has a fully determined size, so anything else is
    // malformed. This also bounds `node_count` before it is used to size anything.
    //
    // Checked throughout: `node_count` is attacker-controlled and `usize` is 32-bit on wasm, where
    // `node_count * 32` would otherwise wrap and let a short buffer pass the length check.
    let expected = node_count
        .checked_mul(32)
        .and_then(|n| n.checked_add(HEADER_BYTES))
        .and_then(|n| n.checked_add(tiles_bytes))
        .ok_or(PomWireError::Malformed)?;
    if bytes.len() != expected {
        return Err(PomWireError::Malformed);
    }

    let n_tiles = v4_n_tiles(n_chunks);
    if n_tiles == 0 {
        return Err(PomWireError::BlobTooSmall);
    }
    let level0_len = v4_tile_level_len(n_chunks);
    let plen = v4_tile_path_len(level0_len);

    let tiles: Vec<Vec<u8>> =
        bytes[HEADER_BYTES..HEADER_BYTES + tiles_bytes].chunks_exact(POM_V4_TILE_BYTES).map(|t| t.to_vec()).collect();
    let node_base = HEADER_BYTES + tiles_bytes;
    let supplied: Vec<[u8; 32]> =
        bytes[node_base..].chunks_exact(32).map(|c| c.try_into().expect("chunks_exact(32) yields 32 bytes")).collect();

    let offsets = v4_offset_chain(seed, &tiles, n_tiles);

    // Same level-0 construction as the encoder: first tile wins on a repeated offset. A proof whose
    // duplicate tiles disagree is not rejected here — it simply fails Merkle verification later,
    // which is the check that is authoritative anyway.
    let level0 = tile_level(&offsets, &tiles);

    let mut paths: Vec<Vec<[u8; 32]>> = (0..POM_V4_K).map(|_| Vec::with_capacity(plen)).collect();
    let mut cursor = 0usize;
    walk_levels(level0, level0_len, plen, |level, level_len, known| {
        let mut full = known.clone();
        for &idx in known.keys() {
            if let Some(sib) = needs_sibling(idx, level_len, known) {
                let v = *supplied.get(cursor).ok_or(PomWireError::Malformed)?;
                cursor += 1;
                full.insert(sib, v);
            }
        }

        // Each tile's path entry: its ancestor's sibling, or the ancestor itself on the odd tail —
        // mirroring `v4_tile_path`'s `if (idx ^ 1) < level.len() { .. } else { level[idx] }`.
        for (i, &off) in offsets.iter().enumerate() {
            let anc = off >> level;
            let sib = anc ^ 1;
            let key = if sib < level_len { sib } else { anc };
            paths[i].push(*full.get(&key).ok_or(PomWireError::Malformed)?);
        }
        Ok(full)
    })?;

    // Every supplied node must have been consumed; a trailing remainder means the sender and we
    // disagree about the tree, which would make the reconstruction wrong in ways verification might
    // not localise.
    if cursor != supplied.len() {
        return Err(PomWireError::Malformed);
    }

    Ok(PomProof {
        tier,
        trace_root: [0u8; 32],
        pow_value,
        final_state,
        initial_trace_path: vec![],
        final_trace_path: vec![],
        openings: vec![],
        steps_v2: None,
        v3: None,
        v4: Some(PomProofV4 { tier, tiles, merkle: paths.into_iter().map(|path| PomV4RangeProof { path }).collect() }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pom_v4::{POM_V4_TILE_CHUNKS, v4_prove};
    use rand::{Rng, SeedableRng, rngs::StdRng};

    /// Build a real blob, prove against it, and return the proof plus its chunk count — so the
    /// paths under test are ones the actual prover produced, not ones shaped by hand.
    fn real_proof(seed: u64, n_tiles: u64, rng_seed: u64) -> (PomProof, u64) {
        let mut rng = StdRng::seed_from_u64(rng_seed);
        let mut blob = vec![0u8; (n_tiles as usize) * POM_V4_TILE_BYTES];
        rng.fill(&mut blob[..]);

        let n_chunks = n_tiles * POM_V4_TILE_CHUNKS;
        let leaves: Vec<[u8; 32]> = blob.chunks_exact(32).map(crate::pom::blake).collect();
        let v4 = v4_prove(seed, 0, &blob, &leaves).expect("prove");
        let proof = PomProof {
            tier: 0,
            trace_root: [0u8; 32],
            pow_value: [0xab; 32],
            final_state: 0xdead_beef_cafe_f00d,
            initial_trace_path: vec![],
            final_trace_path: vec![],
            openings: vec![],
            steps_v2: None,
            v3: None,
            v4: Some(v4),
        };
        (proof, n_chunks)
    }

    fn assert_round_trip(seed: u64, n_tiles: u64, rng_seed: u64) {
        let (proof, n_chunks) = real_proof(seed, n_tiles, rng_seed);
        let enc = encode_v4_deduped(&proof, seed, n_chunks).expect("encode");
        let dec = decode_v4_deduped(&enc, seed, n_chunks).expect("decode");

        let a = proof.v4.as_ref().unwrap();
        let b = dec.v4.as_ref().unwrap();
        assert_eq!(a.tiles, b.tiles, "tiles differ (n_tiles={n_tiles})");
        for i in 0..POM_V4_K {
            assert_eq!(a.merkle[i].path, b.merkle[i].path, "path {i} differs (n_tiles={n_tiles})");
        }
        // The whole point: the canonical bytes must come back identical, because the reconstructed
        // proof is stored and re-served to peers still on the legacy format.
        assert_eq!(proof.to_wire_bytes(), dec.to_wire_bytes(), "canonical wire bytes differ (n_tiles={n_tiles})");
        assert_eq!(proof.wire_digest(), dec.wire_digest(), "wire digest differs (n_tiles={n_tiles})");
    }

    #[test]
    fn round_trip_power_of_two() {
        assert_round_trip(0x1234_5678, 256, 1);
    }

    /// Odd level widths exercise `fold_level`'s self-paired tail at several heights — the case that
    /// silently corrupts reconstruction if the encoder and decoder disagree about it.
    ///
    /// Sizes are kept small on purpose: the reference prover rebuilds the whole tree once per tile
    /// (`v4_tile_path`), so cost grows as `K * n_chunks` and a realistic mainnet blob would take
    /// minutes in a debug build. The tail behaviour under test is a property of the level widths,
    /// which these reproduce at every height.
    #[test]
    fn round_trip_odd_level_widths() {
        for n_tiles in [3, 5, 7, 9, 17, 33, 63, 65, 127, 129, 255, 257, 511, 513] {
            assert_round_trip(0xABCD_EF01 ^ n_tiles, n_tiles, n_tiles);
        }
    }

    /// With very few tiles the 256 walk steps collide constantly, so most paths are shared and the
    /// level-0 map is far smaller than `POM_V4_K`.
    #[test]
    fn round_trip_heavy_offset_collisions() {
        for n_tiles in [1, 2, 4, 8] {
            assert_round_trip(0x5555 ^ n_tiles, n_tiles, n_tiles + 100);
        }
    }

    #[test]
    fn round_trip_varied_seeds() {
        for s in 0..8u64 {
            assert_round_trip(0xDEAD_0000 + s, 333, s);
        }
    }

    /// The compact form must be materially smaller than the canonical one — that is its only
    /// reason to exist.
    #[test]
    fn compact_form_is_smaller() {
        let (proof, n_chunks) = real_proof(42, 512, 7);
        let enc = encode_v4_deduped(&proof, 42, n_chunks).unwrap();
        let canonical = proof.to_wire_bytes();
        assert!(enc.len() < canonical.len(), "compact {} not smaller than canonical {}", enc.len(), canonical.len());
    }

    #[test]
    fn rejects_non_canonical_container() {
        let (mut proof, n_chunks) = real_proof(9, 512, 9);
        proof.trace_root = [7u8; 32];
        assert_eq!(encode_v4_deduped(&proof, 9, n_chunks), Err(PomWireError::NotCanonicalV4));
    }

    #[test]
    fn rejects_truncated_and_overlong_input() {
        let (proof, n_chunks) = real_proof(11, 512, 11);
        let enc = encode_v4_deduped(&proof, 11, n_chunks).unwrap();

        assert!(matches!(decode_v4_deduped(&enc[..enc.len() - 1], 11, n_chunks), Err(PomWireError::Malformed)));
        assert!(matches!(decode_v4_deduped(&enc[..4], 11, n_chunks), Err(PomWireError::Malformed)));

        let mut long = enc.clone();
        long.extend_from_slice(&[0u8; 32]);
        assert!(matches!(decode_v4_deduped(&long, 11, n_chunks), Err(PomWireError::Malformed)));
    }

    /// Decoding under a different seed derives a different offset chain, so the reconstruction must
    /// not match — and must not panic either.
    #[test]
    fn wrong_seed_does_not_reconstruct() {
        let (proof, n_chunks) = real_proof(21, 400, 21);
        let enc = encode_v4_deduped(&proof, 21, n_chunks).unwrap();
        // Refusing outright is equally fine; what must not happen is a matching reconstruction.
        if let Ok(dec) = decode_v4_deduped(&enc, 22, n_chunks) {
            assert_ne!(proof.to_wire_bytes(), dec.to_wire_bytes(), "different seed must not reconstruct");
        }
    }

    /// A blob smaller than one tile has no addressable tile; both directions must say so rather
    /// than divide by zero deriving the offset chain.
    #[test]
    fn rejects_blob_too_small() {
        let (proof, _) = real_proof(3, 64, 3);
        assert_eq!(encode_v4_deduped(&proof, 3, 0), Err(PomWireError::BlobTooSmall));
        assert!(matches!(
            decode_v4_deduped(&[0u8; HEADER_BYTES + POM_V4_K * POM_V4_TILE_BYTES], 3, 0),
            Err(PomWireError::BlobTooSmall)
        ));
    }
}
