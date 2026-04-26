/// Genesis block miner for Keryx mainnet.
///
/// Iterates nonces in parallel (rayon) until a valid PoW is found for the
/// genesis header, then prints the values to paste into genesis.rs.
///
/// Usage:
///   cargo run -p keryx-pow --bin mine-genesis --release
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use keryx_consensus_core::{
    config::genesis::GENESIS,
    header::{CompressedParents, Header},
    hashing,
    merkle::calc_hash_merkle_root,
};
use keryx_hashes::ZERO_HASH;
use keryx_pow::State;
use rayon::prelude::*;

// How many nonces each rayon chunk tests before checking for a solution.
const CHUNK_SIZE: u64 = 1 << 18; // 262 144

fn main() {
    let txs = GENESIS.build_genesis_transactions();
    let merkle_root = calc_hash_merkle_root(txs.iter());

    // Build the candidate header (nonce will be overwritten by each worker).
    let template = Header::new_finalized(
        GENESIS.version,
        CompressedParents::default(),
        merkle_root,
        ZERO_HASH,
        GENESIS.utxo_commitment,
        GENESIS.timestamp,
        GENESIS.bits,
        0, // nonce placeholder
        GENESIS.daa_score,
        0.into(),
        0,
        ZERO_HASH,
    );

    let state = State::new(&template);

    let found = Arc::new(AtomicBool::new(false));
    let winning_nonce = Arc::new(AtomicU64::new(0));
    let hashes_done = Arc::new(AtomicU64::new(0));

    let start = Instant::now();

    // Progress reporter thread
    let hashes_clone = Arc::clone(&hashes_done);
    let found_clone = Arc::clone(&found);
    std::thread::spawn(move || {
        while !found_clone.load(Ordering::Relaxed) {
            std::thread::sleep(std::time::Duration::from_secs(5));
            let h = hashes_clone.load(Ordering::Relaxed);
            let elapsed = start.elapsed().as_secs_f64();
            eprintln!("[{:.0}s] {:.2} MH/s — {} hashes tried", elapsed, h as f64 / elapsed / 1e6, h);
        }
    });

    // Split the full u64 nonce space into chunks and process in parallel.
    let chunk_count = u64::MAX / CHUNK_SIZE;
    (0..chunk_count).into_par_iter().find_any(|&chunk| {
        if found.load(Ordering::Relaxed) {
            return true;
        }
        let base = chunk * CHUNK_SIZE;
        for nonce in base..base + CHUNK_SIZE {
            if state.check_pow(nonce).0 {
                found.store(true, Ordering::Relaxed);
                winning_nonce.store(nonce, Ordering::Relaxed);
                return true;
            }
        }
        hashes_done.fetch_add(CHUNK_SIZE, Ordering::Relaxed);
        false
    });

    let nonce = winning_nonce.load(Ordering::Relaxed);
    let elapsed = start.elapsed();

    // Rebuild header with winning nonce to compute the final block hash.
    let final_header = Header::new_finalized(
        GENESIS.version,
        CompressedParents::default(),
        merkle_root,
        ZERO_HASH,
        GENESIS.utxo_commitment,
        GENESIS.timestamp,
        GENESIS.bits,
        nonce,
        GENESIS.daa_score,
        0.into(),
        0,
        ZERO_HASH,
    );

    let block_hash = hashing::header::hash(&final_header);

    println!("\n=== Genesis mined in {:.2?} ===", elapsed);
    println!("nonce: {:#x},", nonce);
    println!("hash_merkle_root: Hash::from_bytes({:#04x?}),", merkle_root.as_bytes());
    println!("hash: Hash::from_bytes({:#04x?}),", block_hash.as_bytes());
}
