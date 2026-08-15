//! setsolve — read-only pruning-utxoset audit + single-coin anchor solver (node must be STOPPED).
//!
//! Throwaway tool, same shape as `anchorscan`: build with
//! `cargo build --release -p keryx-consensus --bin setsolve`, copy the binary out and remove this
//! file from the repo afterwards.
//!
//! Walks the PRUNING utxoset (prefix 11), recomputes its muhash with the consensus encoding, then
//! tries single-entry anchor substitutions over the candidate window to reach the target
//! commitment.
//!
//! Usage: setsolve <consensus-db-dir> <coin-age-gate-daa> <win-lo> <win-hi> <derived-hex> <target-hex> [span]

use keryx_consensus_core::{
    config::params::ForkActivation,
    hashing::HasherExtensions,
    muhash::MuHashExtensions,
    tx::{ScriptPublicKey, TransactionId, TransactionOutpoint, UtxoEntry},
};
use keryx_hashes::HasherBase;
use keryx_muhash::MuHash;
use serde::Deserialize;
use std::str::FromStr;

#[derive(Deserialize)]
struct UtxoEntryL {
    amount: u64,
    script_public_key: ScriptPublicKey,
    block_daa_score: u64,
    is_coinbase: bool,
    effective_daa: u64,
}

#[derive(Deserialize)]
struct UtxoEntryPreH4L {
    amount: u64,
    script_public_key: ScriptPublicKey,
    block_daa_score: u64,
    is_coinbase: bool,
}

const P_PRUNING_UTXOSET: u8 = 11;

/// Byte-identical replica of the consensus `write_utxo` (private in core) — validated by the
/// derived-hash MATCH check before any solver conclusion is trusted.
fn write_utxo_replica(writer: &mut impl HasherBase, entry: &UtxoEntry, outpoint: &TransactionOutpoint, coin_age_activation: ForkActivation) {
    writer
        .update(outpoint.transaction_id)
        .update(outpoint.index.to_le_bytes())
        .update(entry.block_daa_score.to_le_bytes())
        .update(entry.amount.to_le_bytes())
        .write_bool(entry.is_coinbase)
        .update(entry.script_public_key.version().to_le_bytes())
        .write_var_bytes(entry.script_public_key.script());
    if coin_age_activation.is_active(entry.block_daa_score) {
        writer.update(entry.effective_daa.to_le_bytes());
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let db_path = args.next().expect("usage: setsolve <db-dir> <gate> <win-lo> <win-hi> <derived-hex> <target-hex> [span]");
    let gate: u64 = args.next().expect("gate").parse().unwrap();
    let win_lo: u64 = args.next().expect("win-lo").parse().unwrap();
    let win_hi: u64 = args.next().expect("win-hi").parse().unwrap();
    let derived_hex = args.next().expect("derived hash hex");
    let target_hex = args.next().expect("target hash hex");
    let span: i64 = args.next().map(|s| s.parse().unwrap()).unwrap_or(64);
    let activation = ForkActivation::new(gate);

    let t0 = std::time::Instant::now();
    let opts = rocksdb::Options::default();
    let db = rocksdb::DB::open_for_read_only(&opts, std::path::Path::new(&db_path), false)
        .expect("open db read-only (is the node stopped?)");
    eprintln!("[{:>6.1}s] db opened (read-only), gate={gate}, window=[{win_lo},{win_hi}], span=±{span}", t0.elapsed().as_secs_f64());

    let mut mh = MuHash::new();
    let (mut total, mut in_window, mut inherited) = (0u64, 0u64, 0u64);
    // (outpoint, entry) pairs inside the candidate window carrying an inherited anchor.
    let mut candidates: Vec<(TransactionOutpoint, UtxoEntry)> = Vec::new();

    for item in db.iterator(rocksdb::IteratorMode::From(&[P_PRUNING_UTXOSET], rocksdb::Direction::Forward)) {
        let (k, v) = item.unwrap();
        if k.first() != Some(&P_PRUNING_UTXOSET) {
            break;
        }
        total += 1;
        if total % 5_000_000 == 0 {
            eprintln!("[{:>6.1}s] scanned {} entries, {} candidates", t0.elapsed().as_secs_f64(), total, candidates.len());
        }

        let entry = match bincode::deserialize::<UtxoEntryL>(&v) {
            Ok(e) => UtxoEntry::new_aged(e.amount, e.script_public_key, e.block_daa_score, e.is_coinbase, e.effective_daa),
            Err(_) => {
                let e: UtxoEntryPreH4L = bincode::deserialize(&v).unwrap();
                UtxoEntry::new_aged(e.amount, e.script_public_key, e.block_daa_score, e.is_coinbase, e.block_daa_score)
            }
        };

        // Key is prefix ‖ txid ‖ trimmed LE output index (see `UtxoKey::as_ref`).
        let txid = TransactionId::from_str(&hex::encode(&k[1..33])).unwrap();
        let mut idx_bytes = [0u8; 4];
        let tail = &k[33..];
        idx_bytes[..tail.len()].copy_from_slice(tail);
        let outpoint = TransactionOutpoint::new(txid, u32::from_le_bytes(idx_bytes));

        mh.add_utxo(&outpoint, &entry, activation);

        if (win_lo..=win_hi).contains(&entry.block_daa_score) {
            in_window += 1;
            if entry.effective_daa != entry.block_daa_score {
                inherited += 1;
                candidates.push((outpoint, entry));
            }
        }
    }

    // Persist the aggregated multiset so later solver iterations skip the 3-minute rescan.
    std::fs::write("setsolve-muhash.bin", bincode::serialize(&mh).unwrap()).unwrap();
    let derived = mh.clone().finalize();
    println!("scanned      : {total} entries in {:.1}s", t0.elapsed().as_secs_f64());
    println!("in window    : {in_window} (of which {inherited} carry an inherited anchor -> candidates)");
    println!("derived hash : {derived}");
    println!("expected     : {derived_hex} ({})", if format!("{derived}") == derived_hex { "MATCH — probe encoding is sound" } else { "MISMATCH — probe encoding differs, solver results are void" });

    for (op, e) in &candidates {
        println!(
            "candidate {}:{} amount={} block_daa={} effective_daa={} coinbase={}",
            op.transaction_id, op.index, e.amount, e.block_daa_score, e.effective_daa, e.is_coinbase
        );
    }

    // Single-entry substitution: current × rm(entry) × add(entry') == target?
    println!("solving over {} candidates × {} hypotheses each...", candidates.len(), 2 * span);
    let mut hits = 0u32;
    for (i, (op, e)) in candidates.iter().enumerate() {
        if i > 0 && i % 200 == 0 {
            eprintln!("[{:>6.1}s] solver at candidate {}/{}", t0.elapsed().as_secs_f64(), i, candidates.len());
        }
        let mut base = mh.clone();
        {
            let mut w = base.remove_element_builder();
            write_utxo_replica(&mut w, e, op, activation);
            w.finalize();
        }
        for delta in -span..=span {
            if delta == 0 {
                continue;
            }
            let hyp = e.effective_daa.wrapping_add_signed(delta);
            let mut m = base.clone();
            let e2 = UtxoEntry::new_aged(e.amount, e.script_public_key.clone(), e.block_daa_score, e.is_coinbase, hyp);
            m.add_utxo(op, &e2, activation);
            if format!("{}", m.finalize()) == target_hex {
                println!("HIT: {}:{} anchor {} -> {} (delta {delta}) reaches the target commitment", op.transaction_id, op.index, e.effective_daa, hyp);
                hits += 1;
            }
        }
    }
    if hits == 0 {
        println!("no single-entry substitution reaches the target — mutation is multi-entry or outside the window/span");
    }

    // Ghost-residue mode: the committed multiset may carry an uncancelled add/remove pair for a
    // coin that is long SPENT — a residue no materialized set can reproduce. Enumerate ordered
    // anchor pairs (a, b) and test derived × add(e@a) × remove(e@b) == target.
    let ghost: Vec<String> = std::env::args().skip(8).collect();
    if ghost.len() == 7 {
        let g_txid = TransactionId::from_str(&ghost[0]).unwrap();
        let g_idx: u32 = ghost[1].parse().unwrap();
        let g_amount: u64 = ghost[2].parse().unwrap();
        let g_spk = ScriptPublicKey::from_vec(0, hex::decode(&ghost[3]).unwrap());
        let g_daa: u64 = ghost[4].parse().unwrap();
        let (g_lo, g_hi): (u64, u64) = (ghost[5].parse().unwrap(), ghost[6].parse().unwrap());
        let g_op = TransactionOutpoint::new(g_txid, g_idx);
        println!("ghost mode: {}:{} amount={} block_daa={} anchors [{},{}]², testing {} ordered pairs...", g_txid, g_idx, g_amount, g_daa, g_lo, g_hi, (g_hi - g_lo + 1).pow(2));
        let mut ghost_hits = 0u32;
        for a in g_lo..=g_hi {
            for b in g_lo..=g_hi {
                if a == b {
                    continue;
                }
                let mut m = mh.clone();
                let e_add = UtxoEntry::new_aged(g_amount, g_spk.clone(), g_daa, false, a);
                m.add_utxo(&g_op, &e_add, activation);
                {
                    let mut w = m.remove_element_builder();
                    let e_rem = UtxoEntry::new_aged(g_amount, g_spk.clone(), g_daa, false, b);
                    write_utxo_replica(&mut w, &e_rem, &g_op, activation);
                    w.finalize();
                }
                if format!("{}", m.finalize()) == target_hex {
                    println!("GHOST HIT: residue add(anchor={a}) x remove(anchor={b}) on {}:{} reaches the target commitment", g_txid, g_idx);
                    ghost_hits += 1;
                }
            }
        }
        if ghost_hits == 0 {
            println!("no single ghost pair reaches the target — residue involves more coins or other fields");
        }
    }
}
