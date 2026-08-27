//! What a PoM v4 proof actually costs on the wire, and what is recoverable.
//!
//! The measurement harness behind `pom_v4_wire`: it answers, with numbers rather than arithmetic,
//! three questions, in order of how much they change the plan:
//!
//!   1. How big is a v4 proof per tier, exactly? (`POM_V4_K` tiles + one Merkle range proof each)
//!   2. What does the gzip we ALREADY run on every p2p channel actually buy on it?
//!      (`protocol/p2p/src/core/connection_handler.rs` enables tonic gzip both directions, so any
//!      further generic compression layer would stack on top of this, not add to it.)
//!   3. How many Merkle siblings are redundant? All `POM_V4_K` paths climb to the SAME per-tier
//!      root `R_T`, so their upper levels are shared, yet each tile ships its own full path today.
//!
//! Run: `cargo run --release -p keryx-consensus-core --example pom_v4_size`
//!
//! CAVEAT on the compression numbers: real tiles are quantized GGUF weight bytes. This example
//! fills them from a CSPRNG, which is the incompressible worst case. Q4_K/Q8_0 data is close to
//! that but not identical, so treat the gzip column as a lower bound on the ratio and re-run
//! against real tier bytes (via the tier-root builder) before making a final call on any codec
//! question. What the example does NOT approximate is the sibling redundancy — it is exact, being
//! purely structural (it depends only on the tile offsets, not on the byte values).

use flate2::{Compression, write::GzEncoder};
use keryx_consensus_core::config::params::POM_TIERS_H6;
use keryx_consensus_core::pom::PomProof;
use keryx_consensus_core::pom_v4::{
    POM_V4_CHUNK_BYTES, POM_V4_K, POM_V4_TILE_BYTES, POM_V4_TILE_CHUNKS, PomProofV4, PomV4RangeProof, v4_offset_chain,
    v4_tile_subtree_root,
};
use keryx_consensus_core::pom_v4_wire::{decode_v4_deduped, encode_v4_deduped};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;

/// Local mirror of `pom::hash_pair` (which is `pub(crate)`). Must stay byte-identical to it, or the
/// tree built here stops matching the one the verifier folds.
fn hash_pair(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut buf = [0u8; 64];
    buf[..32].copy_from_slice(left);
    buf[32..].copy_from_slice(right);
    *blake3::hash(&buf).as_bytes()
}

/// Number of independent offset draws averaged for the dedup figures. The count varies by a few
/// nodes between nonces; one draw would read as false precision.
const TRIALS: usize = 200;

/// Size of the tile-root level: the level whose leaves are whole tiles. `fold_level` ceil-halves,
/// and repeated ceil-halving is a single ceil, so this is `ceil(n_chunks / POM_V4_TILE_CHUNKS)`.
///
/// Deliberately NOT `v4_n_tiles`, which FLOORS: when `n_chunks` is not a multiple of 32 the tile
/// level carries one extra partial node that no offset ever addresses. Tier 4
/// (`chunks = 927_994_064`) is exactly that case — 28_999_814 addressable tiles under a level of
/// 28_999_815 nodes. Conflating the two shifts the tree shape by one and silently breaks path
/// reconstruction, so the two quantities are kept apart here on purpose.
fn tile_level_len(n_chunks: u64) -> u64 {
    n_chunks.div_ceil(POM_V4_TILE_CHUNKS)
}

/// Addressable tile count — the range the walk's offsets are reduced modulo.
fn n_tiles(n_chunks: u64) -> u64 {
    n_chunks / POM_V4_TILE_CHUNKS
}

/// Merkle path length from a tile-subtree root up to `R_T`: the number of `fold_level` steps that
/// take the tile level down to a single root. Mirrors the `while level.len() > 1` loop in
/// `pom_v4::v4_tile_path`.
fn path_len(mut level_len: u64) -> usize {
    let mut n = 0;
    while level_len > 1 {
        level_len = level_len.div_ceil(2);
        n += 1;
    }
    n
}

/// Exact number of distinct Merkle siblings needed to prove ALL `offsets` at once.
///
/// Walks the tree bottom-up exactly as `fold_level` builds it. At each level a known node needs its
/// sibling supplied unless the sibling is itself known (shared with another tile) or does not exist
/// (the odd-tail node that `fold_level` pairs with itself — `hash_pair(p0, p0)`, which is why
/// `v4_tile_path` falls back to `level[idx]` when `idx ^ 1` is out of range).
///
/// This is the multiproof node count; the difference against `offsets.len() * path_len` is the
/// redundancy currently on the wire.
fn multiproof_node_count(offsets: &[u64], level0_len: u64) -> usize {
    let mut cur: BTreeSet<u64> = offsets.iter().copied().collect();
    let mut level_len = level0_len;
    let mut supplied = 0usize;
    while level_len > 1 {
        for &idx in cur.iter() {
            let sib = idx ^ 1;
            // Out of range => the self-paired odd tail, derivable. Already known => shared.
            if sib < level_len && !cur.contains(&sib) {
                supplied += 1;
            }
        }
        cur = cur.iter().map(|&i| i >> 1).collect();
        level_len = level_len.div_ceil(2);
    }
    supplied
}

/// gzip at the default level — what tonic's `CompressionEncoding::Gzip` applies today.
fn gzip_len(bytes: &[u8]) -> usize {
    let mut enc = GzEncoder::new(Vec::new(), Compression::default());
    enc.write_all(bytes).unwrap();
    enc.finish().unwrap().len()
}

/// Number of 32-byte blocks in `paths_bytes` that are exact repeats of an earlier one, and how many
/// of those repeats fall within gzip's 32 KiB window (and so are reachable by it) versus beyond it.
///
/// This is the measurement that explains the gzip column: the redundancy is a set of duplicated
/// siblings, and deflate can only turn one into a back-reference if its previous occurrence is still
/// inside the window. It also bounds what a large-window codec (zstd, whose window spans the whole
/// payload) could add without any format change.
fn duplicate_reach(paths_bytes: &[u8]) -> (usize, usize, usize) {
    use std::collections::HashMap;
    const W: usize = 32 * 1024;
    let mut last_seen: HashMap<&[u8], usize> = HashMap::new();
    let (mut dups, mut in_window, mut out_of_window) = (0, 0, 0);
    for (i, block) in paths_bytes.chunks_exact(32).enumerate() {
        let pos = i * 32;
        if let Some(&prev) = last_seen.get(block) {
            dups += 1;
            if pos - prev <= W { in_window += 1 } else { out_of_window += 1 }
        }
        last_seen.insert(block, pos);
    }
    (dups, in_window, out_of_window)
}

/// Stand-in for a tree node the prover would have but we are not materialising: a deterministic
/// function of its position, so that any two tiles that share this node see the SAME bytes.
fn prf(level: usize, idx: u64) -> [u8; 32] {
    let mut buf = [0u8; 16];
    buf[..8].copy_from_slice(&(level as u64).to_le_bytes());
    buf[8..].copy_from_slice(&idx.to_le_bytes());
    *blake3::hash(&buf).as_bytes()
}

/// Build the `POM_V4_K` per-tile Merkle paths so they are MUTUALLY CONSISTENT, and return the
/// multiproof node list alongside them.
///
/// Consistency is the whole point: two tiles sharing an ancestor must carry byte-identical
/// siblings above the meet point, because that shared redundancy is exactly what gzip may or may
/// not find and what the multiproof removes outright. Random per-path bytes would understate
/// gzip and so overstate the dedup win.
///
/// Only the ~9 k nodes actually touched are materialised — supplied siblings from `prf`, internal
/// nodes folded from their children with `hash_pair`, matching `fold_level` including its
/// self-paired odd tail.
fn build_consistent_paths(offsets: &[u64], tiles: &[Vec<u8>], level0_len: u64, plen: usize) -> (Vec<PomV4RangeProof>, Vec<[u8; 32]>) {
    let mut paths: Vec<Vec<[u8; 32]>> = vec![Vec::with_capacity(plen); offsets.len()];
    let mut nodes: Vec<[u8; 32]> = Vec::new();

    // Level 0: the REAL tile-subtree roots, folded from the tile bytes exactly as the verifier
    // does. Rooting the tree anywhere else (a positional stand-in, say) would make the paths
    // internally consistent yet unrelated to the tiles, and the production encoder would rightly
    // fail to round-trip them. Keyed by tile index, so duplicate offsets collapse — which is what
    // really happens, since both tiles carry the same bytes.
    let mut cur: BTreeMap<u64, [u8; 32]> = BTreeMap::new();
    for (i, &o) in offsets.iter().enumerate() {
        cur.entry(o).or_insert_with(|| v4_tile_subtree_root(&tiles[i]));
    }
    let mut level_len = level0_len;

    for level in 0..plen {
        // Complete the level: supply every sibling that is neither known nor self-paired. Iterate
        // ascending (BTreeMap order) — this is the canonical node order the real encoder/decoder
        // must agree on.
        let mut full = cur.clone();
        for (&idx, _) in cur.iter() {
            let sib = idx ^ 1;
            if sib < level_len && !cur.contains_key(&sib) {
                let v = prf(level, sib);
                nodes.push(v);
                full.insert(sib, v);
            }
        }

        // Each tile's path entry at this level: its ancestor's sibling, self on the odd tail
        // (mirrors `v4_tile_path`'s `if (idx ^ 1) < level.len() { .. } else { level[idx] }`).
        for (i, &off) in offsets.iter().enumerate() {
            let anc = off >> level;
            let sib = anc ^ 1;
            let key = if sib < level_len { sib } else { anc };
            paths[i].push(*full.get(&key).expect("sibling materialised above"));
        }

        // Fold to the next level.
        let mut next: BTreeMap<u64, [u8; 32]> = BTreeMap::new();
        for (&idx, _) in cur.iter() {
            let p = idx >> 1;
            if next.contains_key(&p) {
                continue;
            }
            let l = *full.get(&(p * 2)).expect("left child known");
            let r = if p * 2 + 1 < level_len { *full.get(&(p * 2 + 1)).expect("right child known") } else { l };
            next.insert(p, hash_pair(&l, &r));
        }
        cur = next;
        level_len = level_len.div_ceil(2);
    }

    (paths.into_iter().map(|path| PomV4RangeProof { path }).collect(), nodes)
}

/// A canonical v4 container: every legacy/v3 field empty, as `verify_pom_proof_v4_container`
/// requires (`NonCanonicalLegacyFields` otherwise).
fn v4_proof(tier: u8, tiles: Vec<Vec<u8>>, merkle: Vec<PomV4RangeProof>) -> PomProof {
    PomProof {
        tier,
        trace_root: [0u8; 32],
        pow_value: [0xab; 32],
        final_state: 0x1234_5678_9abc_def0,
        initial_trace_path: vec![],
        final_trace_path: vec![],
        openings: vec![],
        steps_v2: None,
        v3: None,
        v4: Some(PomProofV4 { tier, tiles, merkle }),
    }
}

fn main() {
    println!("PoM v4 proof size analysis  (POM_V4_K = {POM_V4_K}, tile = {POM_V4_TILE_BYTES} B = {POM_V4_TILE_CHUNKS} chunks)\n");

    // ---- 1 + 3: exact sizes and the exact structural redundancy, per H6 tier ----
    println!("Per-tier wire size and multiproof saving (structural, exact):\n");
    println!(
        "{:>4}  {:>13}  {:>11}  {:>4}  {:>9}  {:>9}  {:>10}  {:>10}  {:>7}",
        "tier", "chunks", "tile level", "path", "sibs now", "sibs ded", "wire now", "wire ded", "saving"
    );

    for (tier, t) in POM_TIERS_H6.iter().enumerate() {
        let level0 = tile_level_len(t.chunks);
        let tiles = n_tiles(t.chunks);
        let plen = path_len(level0);

        // Average the dedup count over independent offset draws.
        let mut total = 0usize;
        for trial in 0..TRIALS {
            let mut rng = StdRng::seed_from_u64(0xC0FFEE ^ (tier as u64) << 32 ^ trial as u64);
            let offsets: Vec<u64> = (0..POM_V4_K).map(|_| rng.gen_range(0..tiles)).collect();
            total += multiproof_node_count(&offsets, level0);
        }
        let deduped_sibs = total / TRIALS;
        let legacy_sibs = POM_V4_K * plen;

        // Legacy borsh: container + tiles (4 B Vec prefix each) + one path per tile (4 B prefix each).
        let container = 1 + 32 + 32 + 8 + 4 + 4 + 4 + 1 + 1 + 1;
        let tiles_field = 4 + POM_V4_K * (4 + POM_V4_TILE_BYTES);
        let merkle_field = 4 + POM_V4_K * (4 + plen * 32);
        let wire_now = container + 1 + tiles_field + merkle_field;

        // Deduped: fixed-size tiles (no per-tile prefix), explicit offsets, one node list.
        // tier + pow_value + final_state + level0_len + offsets + tiles + nodes
        let wire_ded = 1 + 32 + 8 + 8 + (4 + POM_V4_K * 4) + (4 + POM_V4_K * POM_V4_TILE_BYTES) + (4 + deduped_sibs * 32);

        println!(
            "{:>4}  {:>13}  {:>11}  {:>4}  {:>9}  {:>9}  {:>10}  {:>10}  {:>6.1}%",
            tier,
            t.chunks,
            level0,
            plen,
            legacy_sibs,
            deduped_sibs,
            wire_now,
            wire_ded,
            100.0 * (wire_now - wire_ded) as f64 / wire_now as f64
        );
    }

    // ---- 2: what gzip buys, on the tier-0 shape ----
    let t0 = &POM_TIERS_H6[0];
    let level0 = tile_level_len(t0.chunks);
    let tiles_n = n_tiles(t0.chunks);
    let plen = path_len(level0);

    println!("\n\nCompression, tier 0 shape (path = {plen}):\n");

    let mut rng = StdRng::seed_from_u64(0xDEADBEEF);

    // Tiles: incompressible stand-in for quantized weight bytes (see the CAVEAT above).
    let tiles: Vec<Vec<u8>> = (0..POM_V4_K)
        .map(|_| {
            let mut t = vec![0u8; POM_V4_TILE_BYTES];
            rng.fill(&mut t[..]);
            t
        })
        .collect();

    // Offsets come from the real walk derivation rather than a draw, so the proof built below is
    // one the production encoder accepts — which is what makes the end-to-end check at the bottom
    // meaningful at mainnet scale without materialising a 4.8 GB blob.
    let walk_seed = 0xC0DE_1234_5678_9ABCu64;
    let offsets: Vec<u64> = v4_offset_chain(walk_seed, &tiles, tiles_n).to_vec();

    // Paths, built mutually consistent so the real shared-sibling redundancy is present in the
    // bytes — that is what decides whether gzip already captures part of the dedup win.
    let (merkle, nodes) = build_consistent_paths(&offsets, &tiles, level0, plen);

    let proof = v4_proof(0, tiles.clone(), merkle.clone());
    let wire = proof.to_wire_bytes();

    // Component-wise, to see which half of the proof gzip can and cannot touch.
    let tiles_bytes: Vec<u8> = tiles.concat();
    let paths_bytes: Vec<u8> = merkle.iter().flat_map(|m| m.path.iter().flatten().copied()).collect();

    // The deduped encoding, byte-for-byte as it would go on the wire:
    // tier | pow_value | final_state | level0_len | offsets (u32 each) | tiles (fixed) | nodes
    let mut deduped = Vec::with_capacity(400_000);
    deduped.push(0u8);
    deduped.extend_from_slice(&[0xab; 32]);
    deduped.extend_from_slice(&0x1234_5678_9abc_def0u64.to_le_bytes());
    deduped.extend_from_slice(&level0.to_le_bytes());
    deduped.extend_from_slice(&(POM_V4_K as u32).to_le_bytes());
    for &o in &offsets {
        deduped.extend_from_slice(&(o as u32).to_le_bytes());
    }
    for t in &tiles {
        deduped.extend_from_slice(t);
    }
    deduped.extend_from_slice(&(nodes.len() as u32).to_le_bytes());
    for n in &nodes {
        deduped.extend_from_slice(n);
    }

    // Same siblings, level-major instead of tile-major: all 256 siblings for level 0, then level 1,
    // and so on. Duplicated siblings live within a level, and one level is only 8 KB, so this
    // ordering brings every duplicate inside gzip's 32 KB window. Measuring it isolates how much of
    // the gap is purely window reach — i.e. what a large-window codec like zstd could recover with
    // no format change at all.
    let paths_level_major: Vec<u8> = (0..plen).flat_map(|l| merkle.iter().flat_map(move |m| m.path[l]).collect::<Vec<u8>>()).collect();

    println!("{:>30}  {:>9}  {:>9}  {:>8}", "payload", "raw", "gzip", "gzip saves");
    for (name, buf) in [
        ("full proof today (borsh)", &wire),
        ("  tiles only (weight bytes)", &tiles_bytes),
        ("  merkle paths, tile-major", &paths_bytes),
        ("  merkle paths, level-major", &paths_level_major),
        ("full proof deduped", &deduped),
    ] {
        let g = gzip_len(buf);
        println!("{:>30}  {:>9}  {:>9}  {:>7.2}%", name, buf.len(), g, 100.0 * (buf.len() as f64 - g as f64) / buf.len() as f64);
    }

    let (dups, in_win, out_win) = duplicate_reach(&paths_bytes);
    println!("\nWhy gzip gets what it gets (duplicate 32 B siblings in the path field, tile-major):");
    println!("  exact repeats of an earlier sibling : {dups}");
    println!("  within gzip's 32 KiB window         : {in_win}  <- deflate can back-reference these");
    println!("  beyond it                           : {out_win}  <- only a large-window codec reaches these");

    let legacy_sibs = POM_V4_K * plen;
    let saved = (legacy_sibs - nodes.len()) * 32;

    println!("\nStructural redundancy in the merkle field (tier 0, this offset draw):");
    println!("  siblings on the wire today : {legacy_sibs} ({} B)", legacy_sibs * 32);
    println!("  distinct siblings needed   : {} ({} B)", nodes.len(), nodes.len() * 32);
    println!(
        "  redundant                  : {} ({saved} B, {:.1}% of the path field)",
        legacy_sibs - nodes.len(),
        100.0 * saved as f64 / (legacy_sibs * 32) as f64
    );

    // The comparison that actually matters: bytes a peer receives now vs after the change, both
    // measured AFTER the gzip that is already on the channel.
    let now_raw = wire.len();
    let now_gz = gzip_len(&wire);
    let ded_raw = deduped.len();
    let ded_gz = gzip_len(&deduped);

    println!("\nWhat a peer actually receives (the metric CountBytesBody reports):");
    let baseline = now_gz;
    let rel = |b: usize| 100.0 * (b as f64 - baseline as f64) / baseline as f64;

    // Same information as today, only the merkle field transposed to level-major, so gzip can reach
    // every duplicate. Reversing a transposition needs no tree math, which makes this the cheapest
    // conceivable format change and a MEASURED lower bound on what a large-window codec would buy.
    let mut reordered = Vec::with_capacity(wire.len());
    reordered.push(0u8);
    reordered.extend_from_slice(&[0xab; 32]);
    reordered.extend_from_slice(&0x1234_5678_9abc_def0u64.to_le_bytes());
    for t in &tiles {
        reordered.extend_from_slice(t);
    }
    reordered.extend_from_slice(&paths_level_major);
    let reordered_gz = gzip_len(&reordered);

    println!("  today: legacy + gzip          : {now_gz} B  (raw {now_raw})   <- baseline");
    println!("  level-major + gzip            : {reordered_gz} B  ({:+.1}%)", rel(reordered_gz));
    println!("  multiproof, uncompressed      : {ded_raw} B  ({:+.1}%)", rel(ded_raw));
    println!("  multiproof + gzip (pointless) : {ded_gz} B  ({:+.1}%)", rel(ded_gz));
    println!(
        "\n  Best case for any pure-codec change (large window, no format change) is bounded below by\n  \
         the redundancy-free size {ded_raw} B plus a match token per duplicate: about {} B ({:+.1}%).\n  \
         So swapping gzip for zstd can recover at most ~{:.0}% of what the multiproof recovers.",
        ded_raw + dups * 4,
        rel(ded_raw + dups * 4),
        100.0 * (baseline - (ded_raw + dups * 4)) as f64 / (baseline - ded_raw) as f64
    );

    println!(
        "\nRatio check: tiles are {:.1}% of the raw proof, paths {:.1}%.",
        100.0 * (POM_V4_K * POM_V4_TILE_BYTES) as f64 / wire.len() as f64,
        100.0 * (legacy_sibs * 32) as f64 / wire.len() as f64
    );
    // End-to-end through the production encoder, at real tier-0 dimensions.
    let real_compact = encode_v4_deduped(&proof, walk_seed, t0.chunks).expect("encode");
    let decoded = decode_v4_deduped(&real_compact, walk_seed, t0.chunks).expect("decode");
    assert_eq!(proof.to_wire_bytes(), decoded.to_wire_bytes(), "round-trip must be byte-exact");

    // Best of N, after a warm-up: the first call also pays for spinning up the rayon pool, and a
    // single shot reads as noise rather than cost.
    let best_us = |mut f: Box<dyn FnMut()>| {
        for _ in 0..3 {
            f();
        }
        (0..20)
            .map(|_| {
                let t = std::time::Instant::now();
                f();
                t.elapsed().as_secs_f64() * 1e6
            })
            .fold(f64::INFINITY, f64::min)
    };
    let enc_us = best_us(Box::new(|| {
        let _ = encode_v4_deduped(&proof, walk_seed, t0.chunks).unwrap();
    }));
    let dec_us = best_us(Box::new(|| {
        let _ = decode_v4_deduped(&real_compact, walk_seed, t0.chunks).unwrap();
    }));

    println!("\nProduction encoder at tier-0 dimensions (round-trip verified byte-exact):");
    println!("  compact encoding : {} B ({:+.1}% vs the {now_gz} B a peer gets today)", real_compact.len(), rel(real_compact.len()));
    println!("  encode {enc_us:.0} us | decode {dec_us:.0} us");
    println!(
        "  At 10 BPS x 8 peers, encoding per peer costs {:.1}% of a core; once per block, {:.2}%.",
        100.0 * (enc_us * 1e-6) * 10.0 * 8.0,
        100.0 * (enc_us * 1e-6) * 10.0
    );

    println!("Chunk size is {POM_V4_CHUNK_BYTES} B; a tile is {POM_V4_TILE_CHUNKS} chunks.");
}
