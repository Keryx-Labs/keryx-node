//! Dump the sealed service-bond rows of a node datadir (read-only RocksDB secondary) and
//! recompute the MuHash commitment over the rows at or below a pruning-point daa.
use keryx_consensus::processes::service_commit::{burn_row_bytes, first_seen_row_bytes, reward_row_bytes, strike_row_bytes};
use keryx_consensus_core::collateral::{RewardEntry, StrikeEntry};
use keryx_hashes::Hash;
use keryx_muhash::MuHash;
use std::env;

fn h32(b: &[u8]) -> Hash {
    let mut a = [0u8; 32];
    a.copy_from_slice(&b[..32]);
    Hash::from_bytes(a)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: dump_service_state <consensus-db-path> [pp_daa]");
        std::process::exit(2);
    }
    let db_path = &args[1];
    let pp_daa: Option<u64> = args.get(2).and_then(|s| s.parse().ok());
    let from_daa: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0);
    let secondary = std::env::temp_dir().join(format!("keryx-svc-secondary-{}", std::process::id()));
    let mut opts = rocksdb::Options::default();
    opts.set_max_open_files(-1);
    let live = std::env::var("KERYX_DUMP_SECONDARY").is_ok();
    let db = if live {
        let db = match rocksdb::DB::open_as_secondary(&opts, std::path::Path::new(db_path), secondary.as_path()) {
            Ok(db) => db,
            Err(e) => {
                eprintln!("open_as_secondary failed: {e}");
                std::process::exit(1);
            }
        };
        if let Err(e) = db.try_catch_up_with_primary() {
            eprintln!("warning: catch_up failed: {e}");
        }
        db
    } else {
        match rocksdb::DB::open_for_read_only(&opts, std::path::Path::new(db_path), false) {
            Ok(db) => db,
            Err(e) => {
                eprintln!("open_for_read_only failed: {e}");
                std::process::exit(1);
            }
        }
    };
    if args.get(2).map(|s| s.as_str()) == Some("--prefixes") {
        for prefix in 0u8..=255 {
            let mut n = 0u64;
            let iter = db.iterator(rocksdb::IteratorMode::From(&[prefix], rocksdb::Direction::Forward));
            for item in iter {
                let (k, _) = match item {
                    Ok(kv) => kv,
                    Err(e) => {
                        eprintln!("iteration error at prefix {prefix}: {e}");
                        break;
                    }
                };
                if k.first() != Some(&prefix) {
                    break;
                }
                n += 1;
            }
            if n > 0 {
                println!("prefix {prefix}: {n} keys");
            }
        }
        let _ = std::fs::remove_dir_all(&secondary);
        return;
    }
    let mut rows: Vec<(u64, String, Vec<u8>)> = Vec::new();
    let mut scan = |prefix: u8, f: &mut dyn FnMut(&[u8], &[u8], &mut Vec<(u64, String, Vec<u8>)>)| {
        let iter = db.iterator(rocksdb::IteratorMode::From(&[prefix], rocksdb::Direction::Forward));
        for item in iter {
            let (k, v) = match item {
                Ok(kv) => kv,
                Err(e) => {
                    eprintln!("iteration error at prefix {prefix}: {e}");
                    break;
                }
            };
            if k.first() != Some(&prefix) {
                break;
            }
            f(&k[1..], &v, &mut rows);
        }
    };
    scan(196, &mut |k, v, rows| {
        let txid = h32(k);
        let index = u32::from_le_bytes(k[32..36].try_into().unwrap());
        let daa: u64 = bincode::deserialize(v).unwrap();
        rows.push((daa, format!("burn {daa} {txid}:{index}"), burn_row_bytes(txid, index, daa).to_vec()));
    });
    scan(198, &mut |k, v, rows| {
        let daa = u64::from_be_bytes(k[..8].try_into().unwrap());
        let miner = h32(&k[8..40]);
        let e: StrikeEntry = bincode::deserialize(v).unwrap();
        rows.push((
            daa,
            format!("strike {daa} {miner} count={} last_daa={}", e.count, e.last_daa),
            strike_row_bytes(daa, miner, e.count, e.last_daa).to_vec(),
        ));
    });
    scan(199, &mut |k, v, rows| {
        let miner = h32(k);
        let daa: u64 = bincode::deserialize(v).unwrap();
        rows.push((daa, format!("first_seen {daa} {miner}"), first_seen_row_bytes(miner, daa).to_vec()));
    });
    scan(200, &mut |k, v, rows| {
        let mut rh = [0u8; 32];
        rh.copy_from_slice(&k[..32]);
        let e: RewardEntry = bincode::deserialize(v).unwrap();
        let spk = e.spk.as_ref().map(|s| hex::encode(s.script())).unwrap_or_else(|| "burned".into());
        rows.push((
            e.daa,
            format!("reward {} {} winner={} amount={} spk={}", e.daa, hex::encode(rh), e.winner, e.amount, spk),
            reward_row_bytes(rh, e.winner, e.amount, e.daa, e.spk.as_ref()),
        ));
    });
    rows.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    for (_, line, _) in &rows {
        println!("{line}");
    }
    let commit = |bound: u64| {
        let mut acc = MuHash::new();
        let mut n = 0usize;
        for (d, _, b) in &rows {
            if *d > from_daa && *d <= bound {
                acc.add_element(b);
                n += 1;
            }
        }
        (n, if n == 0 { Hash::default() } else { acc.finalize() })
    };
    if let Some(pp) = pp_daa {
        let (n, h) = commit(pp);
        eprintln!("rows in ({from_daa}, {pp}]: {n} muhash {h}");
    }
    let (n, h) = commit(u64::MAX);
    eprintln!("rows total: {n} muhash {h}");
    let _ = std::fs::remove_dir_all(&secondary);
}
