//! Off-node PoM v3 re-walk detector. Reads the byte-exact canonical tier blob once, then for
//! each block header (RPC JSON) redoes the honest walk and checks whether the committed
//! `pom_final_state` equals `fold64(v3_state_root(S_K))`. A mismatch means the final state was
//! not produced by an honest walk over the pinned weights — the H6 lottery-grind shortcut.

use std::io::Read;
use keryx_consensus_core::header::Header;
use keryx_consensus_core::hashing::header::hash_override_nonce_time;
use keryx_consensus_core::pom::pom_block_seed_h5_2;
use keryx_consensus_core::pom_v3::{fold64, v3_state_root, v3_walk, POM_V3_K};
use keryx_consensus_core::BlueWorkType;
use keryx_hashes::Hash;

fn h(v: &serde_json::Value, k: &str) -> String { v[k].as_str().unwrap().to_string() }
fn u64f(v: &serde_json::Value, k: &str) -> u64 {
    match &v[k] {
        serde_json::Value::String(s) => s.parse().unwrap(),
        serde_json::Value::Number(n) => n.as_u64().unwrap(),
        _ => panic!("field {k} not a u64"),
    }
}

fn check(path: &str, blob: &[u8]) {
    let hd: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let parents: Vec<Vec<Hash>> = hd["parentsByLevel"].as_array().unwrap().iter()
        .map(|lvl| lvl.as_array().unwrap().iter()
            .map(|x| x.as_str().unwrap().parse::<Hash>().unwrap()).collect())
        .collect();
    let timestamp = u64f(&hd, "timestamp");
    let nonce = u64f(&hd, "nonce");
    let daa = u64f(&hd, "daaScore");
    let pfs = u64f(&hd, "pomFinalState");
    let tier = u64f(&hd, "pomTier") as u8;
    let header = Header::new_finalized(
        u64f(&hd, "version") as u16,
        parents.try_into().unwrap(),
        h(&hd, "hashMerkleRoot").parse().unwrap(),
        h(&hd, "acceptedIdMerkleRoot").parse().unwrap(),
        h(&hd, "utxoCommitment").parse().unwrap(),
        timestamp, u64f(&hd, "bits") as u32, nonce, daa,
        BlueWorkType::from_hex(&h(&hd, "blueWork")).unwrap(),
        u64f(&hd, "blueScore"),
        h(&hd, "pruningPoint").parse().unwrap(),
        pfs,
        h(&hd, "serviceStateHash").parse().unwrap_or_default(),
        tier,
    );
    let pre: [u8; 32] = hash_override_nonce_time(&header, 0, 0).as_bytes();
    let seed = pom_block_seed_h5_2(&pre, timestamp, nonce);
    let walk = v3_walk(seed, blob).expect("walk");
    let computed = fold64(&v3_state_root(&walk.states[POM_V3_K]));
    let ok = computed == pfs;
    println!("{:<22} tier={tier} daa={daa}  committed={pfs:#018x}  walk={computed:#018x}  {}",
        path.rsplit('/').next().unwrap(),
        if ok { "MATCH (honest)" } else { "MISMATCH (grind)" });
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: rewalk <tier.blob> <header.json>...");
        std::process::exit(2);
    }
    eprintln!("reading blob {} ...", &args[1]);
    let mut blob = Vec::new();
    std::fs::File::open(&args[1]).unwrap().read_to_end(&mut blob).unwrap();
    eprintln!("blob {} bytes; walking each header (K={POM_V3_K})\n", blob.len());
    for p in &args[2..] { check(p, &blob); }
}
