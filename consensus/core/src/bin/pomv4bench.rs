//! Per-proof cost of PoM v4 verification, measured on the real functions.
//!
//! Times the two dominant parts of `verify_pom_proof_v4`: the K-step re-walk and the per-tile
//! Merkle leaf hashing. Synthetic tiles: the cost is structural, not data-dependent.

use keryx_consensus_core::pom_v4::{
    POM_V4_K, POM_V4_TILE_BYTES, v4_initial_state_into, v4_state_root, v4_tile_subtree_root, v4_transition_into,
};
use std::time::Instant;

fn tiles(n: usize) -> Vec<Vec<u8>> {
    // deterministic filler; verification cost does not depend on the values
    (0..n)
        .map(|i| {
            let mut t = vec![0u8; POM_V4_TILE_BYTES];
            let mut x = 0x9E3779B97F4A7C15u64 ^ (i as u64);
            for b in t.iter_mut() {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                *b = (x >> 24) as u8;
            }
            t
        })
        .collect()
}

fn main() {
    let reps: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(20);
    let tl = tiles(POM_V4_K);

    let mut a = [0u8; POM_V4_TILE_BYTES];
    let mut b = [0u8; POM_V4_TILE_BYTES];

    // warm
    v4_initial_state_into(&mut a, 1);
    let mut src_is_a = true;
    for (i, t) in tl.iter().enumerate() {
        if src_is_a {
            v4_transition_into(&mut b, &a, t, i as u32);
        } else {
            v4_transition_into(&mut a, &b, t, i as u32);
        }
        src_is_a = !src_is_a;
    }
    std::hint::black_box(if src_is_a { &a } else { &b });

    let t0 = Instant::now();
    for r in 0..reps {
        v4_initial_state_into(&mut a, r as u64);
        let mut src_is_a = true;
        for (i, t) in tl.iter().enumerate() {
            if src_is_a {
                v4_transition_into(&mut b, &a, t, i as u32);
            } else {
                v4_transition_into(&mut a, &b, t, i as u32);
            }
            src_is_a = !src_is_a;
        }
        let st: &[u8] = if src_is_a { &a } else { &b };
        std::hint::black_box(v4_state_root(st));
    }
    let walk = t0.elapsed().as_secs_f64() / reps as f64;

    let t1 = Instant::now();
    for _ in 0..reps {
        for t in tl.iter() {
            std::hint::black_box(v4_tile_subtree_root(t));
        }
    }
    let merkle = t1.elapsed().as_secs_f64() / reps as f64;

    println!("K = {POM_V4_K}, tile = {POM_V4_TILE_BYTES} B, reps = {reps}");
    println!("  re-walk  ({POM_V4_K} transitions) : {:>9.3} ms / proof", walk * 1e3);
    println!("  merkle leaves ({POM_V4_K} tiles)  : {:>9.3} ms / proof", merkle * 1e3);
    println!("  total mesure                     : {:>9.3} ms / proof", (walk + merkle) * 1e3);
    println!();
    for bps in [10.0f64, 15.0] {
        let cores = (walk + merkle) * bps;
        println!("  a {bps:.0} blocs/s, 1 verification par bloc : {:.2} coeur(s) ({:.0} %)", cores, cores * 100.0);
    }
}