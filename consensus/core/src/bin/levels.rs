//! Block-level distribution measured on real headers, per difficulty regime and per anchor.
//!
//! Reads block headers as JSON (wRPC shape), recomputes each header's PoW value with the same
//! functions the node uses, and reports the level each anchor would assign. Read-only: nothing
//! here feeds consensus.
//!
//! usage: levels <headers.json>... [--anchors 225,230,250]
//!
//! Input is either a JSON array of header objects, or an object carrying `blocks[].header`.

use keryx_consensus_core::BlueWorkType;
use keryx_consensus_core::hashing::header::hash_override_nonce_time;
use keryx_consensus_core::header::Header;
use keryx_consensus_core::pom::pom_pow_value_h3;
use keryx_hashes::Hash;
use keryx_math::Uint256;
use std::collections::BTreeMap;

const DEFAULT_ANCHORS: [u32; 6] = [225, 230, 234, 236, 238, 250];

fn s(v: &serde_json::Value, k: &str) -> String {
    v[k].as_str().unwrap_or_default().to_string()
}

fn u64f(v: &serde_json::Value, k: &str) -> u64 {
    match &v[k] {
        serde_json::Value::String(x) => x.parse().unwrap_or(0),
        serde_json::Value::Number(n) => n.as_u64().unwrap_or(0),
        _ => 0,
    }
}

/// Rebuilds the header, then mirrors `calc_pom_pow`: pre-pow hash with nonce/time zeroed, folded
/// through the committed walk state.
fn pow_and_target(hd: &serde_json::Value) -> Option<(u32, u32, u64)> {
    let parents: Vec<Vec<Hash>> = hd["parentsByLevel"]
        .as_array()?
        .iter()
        .map(|lvl| lvl.as_array().unwrap().iter().filter_map(|x| x.as_str()?.parse::<Hash>().ok()).collect())
        .collect();
    let timestamp = u64f(hd, "timestamp");
    let nonce = u64f(hd, "nonce");
    let daa = u64f(hd, "daaScore");
    let bits = u64f(hd, "bits") as u32;
    let header = Header::new_finalized(
        u64f(hd, "version") as u16,
        parents.try_into().ok()?,
        s(hd, "hashMerkleRoot").parse().ok()?,
        s(hd, "acceptedIdMerkleRoot").parse().ok()?,
        s(hd, "utxoCommitment").parse().ok()?,
        timestamp,
        bits,
        nonce,
        daa,
        BlueWorkType::from_hex(&s(hd, "blueWork")).ok()?,
        u64f(hd, "blueScore"),
        s(hd, "pruningPoint").parse().ok()?,
        u64f(hd, "pomFinalState"),
        s(hd, "serviceStateHash").parse().unwrap_or_default(),
        u64f(hd, "pomTier") as u8,
    );
    let pre: [u8; 32] = hash_override_nonce_time(&header, 0, 0).as_bytes();
    let pow = Uint256::from_le_bytes(pom_pow_value_h3(header.pom_final_state, &pre));
    let target = Uint256::from_compact_target_bits(bits);
    Some((pow.bits(), target.bits(), daa))
}

fn level(anchor: u32, pow_bits: u32) -> u32 {
    (anchor as i64 - pow_bits as i64).max(0) as u32
}

fn headers(v: &serde_json::Value) -> Vec<serde_json::Value> {
    if let Some(a) = v.as_array() {
        return a.iter().map(|b| b.get("header").unwrap_or(b).clone()).collect();
    }
    if let Some(a) = v.get("blocks").and_then(|b| b.as_array()) {
        return a.iter().map(|b| b.get("header").unwrap_or(b).clone()).collect();
    }
    v.get("header").map(|h| vec![h.clone()]).unwrap_or_else(|| vec![v.clone()])
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut files = Vec::new();
    let mut anchors: Vec<u32> = DEFAULT_ANCHORS.to_vec();
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--anchors" {
            anchors = args[i + 1].split(',').filter_map(|x| x.trim().parse().ok()).collect();
            i += 2;
        } else {
            files.push(args[i].clone());
            i += 1;
        }
    }
    if files.is_empty() {
        eprintln!("usage: levels <headers.json>... [--anchors 225,230,250]");
        std::process::exit(2);
    }

    // regime key = target.bits(); value = (pow_bits samples, daa range)
    let mut regimes: BTreeMap<u32, (Vec<u32>, u64, u64)> = BTreeMap::new();
    let mut skipped = 0usize;
    for f in &files {
        let txt = match std::fs::read_to_string(f) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("skip {f}: {e}");
                continue;
            }
        };
        let v: serde_json::Value = match serde_json::from_str(&txt) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("skip {f}: {e}");
                continue;
            }
        };
        for hd in headers(&v) {
            match pow_and_target(&hd) {
                Some((pb, tb, daa)) => {
                    let e = regimes.entry(tb).or_insert((Vec::new(), u64::MAX, 0));
                    e.0.push(pb);
                    e.1 = e.1.min(daa);
                    e.2 = e.2.max(daa);
                }
                None => skipped += 1,
            }
        }
    }

    let total: usize = regimes.values().map(|(v, _, _)| v.len()).sum();
    println!("headers analysed: {total}   unparsed: {skipped}");
    println!("level = max(anchor - pow.bits(), 0);  'niv+1' = levels 0..=level a block occupies");
    println!();

    for (tbits, (pow_bits, daa_lo, daa_hi)) in &regimes {
        let n = pow_bits.len() as f64;
        println!("== regime target.bits() = {tbits}   blocs = {}   daa {daa_lo}..{daa_hi}", pow_bits.len());
        println!(
            "{:>8} {:>7} {:>7} {:>8} {:>9} {:>9} {:>9}",
            "ancre", "niv min", "niv moy", "niv+1 moy", ">=min+1", ">=min+2", ">=min+3"
        );
        for a in &anchors {
            let levels: Vec<u32> = pow_bits.iter().map(|pb| level(*a, *pb)).collect();
            let lo = *levels.iter().min().unwrap_or(&0);
            let mean: f64 = levels.iter().map(|l| *l as f64).sum::<f64>() / n;
            let occ: f64 = levels.iter().map(|l| (*l + 1) as f64).sum::<f64>() / n;
            let frac = |k: u32| levels.iter().filter(|l| **l >= lo + k).count() as f64 / n;
            println!(
                "{:>8} {:>7} {:>7.2} {:>8.2} {:>8.2}% {:>8.2}% {:>8.2}%",
                a,
                lo,
                mean,
                occ,
                100.0 * frac(1),
                100.0 * frac(2),
                100.0 * frac(3)
            );
        }
        println!();
    }
}
