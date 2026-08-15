//! residue — measures the ghost residue between the committed multiset lineage and the
//! materialized UTXO set on one datadir (node must be STOPPED). Throwaway tool, `anchorscan` shape.
//!
//! The committed lineage (VirtualState.multiset, validated against every header commitment) may
//! carry uncancelled add/remove pairs for long-spent coins; no materialized set can reproduce
//! those. R = multiset ÷ muhash(set). If R also maps the materialized pruning set onto the
//! pruning point's committed hash, R is the window-constant to pin in the fix.
//!
//! Usage: residue <consensus-db-dir> <coin-age-gate-daa> <pruning-muhash-bin> <pp-target-hex>

use keryx_consensus::model::stores::virtual_state::VirtualState;
use keryx_consensus_core::{
    config::params::ForkActivation,
    muhash::MuHashExtensions,
    tx::{ScriptPublicKey, TransactionId, TransactionOutpoint, UtxoEntry},
};
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

const P_VIRTUAL_UTXOSET: u8 = 27;
const P_VIRTUAL_STATE: u8 = 28;

/// Group inverse through the serde layer: MuHash serializes as (numerator, denominator); swapping
/// the two 384-byte halves yields the inverse element. Validated by the self-check in main.
fn inverse(mh: &MuHash) -> MuHash {
    let mut bytes = bincode::serialize(mh).unwrap();
    assert_eq!(bytes.len(), 768, "unexpected MuHash serde layout");
    let (a, b) = bytes.split_at_mut(384);
    a.swap_with_slice(b);
    bincode::deserialize(&bytes).unwrap()
}

fn main() {
    let mut args = std::env::args().skip(1);
    let db_path = args.next().expect("usage: residue <db-dir> <gate> <pruning-muhash-bin> <pp-target-hex>");
    let gate: u64 = args.next().expect("gate").parse().unwrap();
    let pp_bin = args.next().expect("pruning muhash bin path");
    let pp_target = args.next().expect("pp target hex");
    let activation = ForkActivation::new(gate);

    let t0 = std::time::Instant::now();
    let opts = rocksdb::Options::default();
    let db = rocksdb::DB::open_for_read_only(&opts, std::path::Path::new(&db_path), false)
        .expect("open db read-only (is the node stopped?)");

    // Committed lineage: the persisted virtual multiset, block-by-block validated against every
    // header utxo_commitment on the way here.
    let vs_bytes = db.get(&[P_VIRTUAL_STATE]).unwrap().expect("virtual state present");
    let vs: VirtualState = bincode::deserialize(&vs_bytes).unwrap();
    let mut committed = vs.multiset.clone();
    println!("virtual daa        : {}", vs.daa_score);
    println!("committed multiset : {}", committed.finalize());

    // Materialized side: full muhash over the virtual UTXO set.
    let mut set_mh = MuHash::new();
    let mut total = 0u64;
    for item in db.iterator(rocksdb::IteratorMode::From(&[P_VIRTUAL_UTXOSET], rocksdb::Direction::Forward)) {
        let (k, v) = item.unwrap();
        if k.first() != Some(&P_VIRTUAL_UTXOSET) {
            break;
        }
        total += 1;
        if total % 10_000_000 == 0 {
            eprintln!("[{:>6.1}s] scanned {} virtual entries", t0.elapsed().as_secs_f64(), total);
        }
        let entry = match bincode::deserialize::<UtxoEntryL>(&v) {
            Ok(e) => UtxoEntry::new_aged(e.amount, e.script_public_key, e.block_daa_score, e.is_coinbase, e.effective_daa),
            Err(_) => {
                let e: UtxoEntryPreH4L = bincode::deserialize(&v).unwrap();
                UtxoEntry::new_aged(e.amount, e.script_public_key, e.block_daa_score, e.is_coinbase, e.block_daa_score)
            }
        };
        let txid = TransactionId::from_str(&hex::encode(&k[1..33])).unwrap();
        let mut idx_bytes = [0u8; 4];
        let tail = &k[33..];
        idx_bytes[..tail.len()].copy_from_slice(tail);
        set_mh.add_utxo(&TransactionOutpoint::new(txid, u32::from_le_bytes(idx_bytes)), &entry, activation);
    }
    println!("materialized set   : {} ({} entries in {:.1}s)", set_mh.clone().finalize(), total, t0.elapsed().as_secs_f64());

    // Self-check of the serde-swap inverse: B ∘ B⁻¹ must equal the empty multiset.
    let mut ident = set_mh.clone();
    ident.combine(&inverse(&set_mh));
    assert_eq!(ident.finalize(), MuHash::new().finalize(), "inverse self-check failed — serde layout assumption is wrong");

    if committed.clone().finalize() == set_mh.clone().finalize() {
        println!("NO residue at the virtual tip — hypothesis refuted on this datadir.");
        return;
    }

    // R = committed ÷ materialized.
    let mut r = committed.clone();
    r.combine(&inverse(&set_mh));
    let r_bytes = r.clone().serialize();
    println!("residue R (384B LE): {}", hex::encode(r_bytes));

    // Decisive test: does the SAME R map the materialized pruning set onto its committed hash?
    let pp_mh: MuHash = bincode::deserialize(&std::fs::read(&pp_bin).expect("pruning muhash bin")).unwrap();
    let mut pp_fixed = pp_mh.clone();
    pp_fixed.combine(&r);
    let pp_result = pp_fixed.finalize();
    println!("pruning set x R    : {}", pp_result);
    println!("pp committed       : {}", pp_target);
    if format!("{pp_result}") == pp_target {
        println!("VERDICT: R is WINDOW-CONSTANT — muhash(set) x R == commitment holds at both measured points. Pin R.");
    } else {
        println!("VERDICT: R differs between tip and pruning point — residue accreted between the two; per-point residues needed.");
    }
}
