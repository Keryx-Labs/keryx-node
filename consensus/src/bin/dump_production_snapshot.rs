//! Dump the production-index and service-ledger snapshots held by a node datadir (read-only
//! RocksDB; secondary mode with KERYX_DUMP_SECONDARY so a running node can be read) and print,
//! per sample, the stored hash plus a chain-numbering-independent digest.
use keryx_consensus_core::collateral::{ProductionIndexSnapshot, ServiceLedgerSnapshot};
use keryx_hashes::Hash;
use std::env;
use std::str::FromStr;

const PRODUCTION_PREFIX: u8 = 202;
const LEDGER_PREFIX: u8 = 201;

fn open(db_path: &str) -> rocksdb::DB {
    let mut opts = rocksdb::Options::default();
    opts.set_max_open_files(256);
    if env::var("KERYX_DUMP_SECONDARY").is_ok() {
        let secondary = env::temp_dir().join(format!("keryx-prod-secondary-{}", std::process::id()));
        let db = rocksdb::DB::open_as_secondary(&opts, std::path::Path::new(db_path), secondary.as_path())
            .unwrap_or_else(|e| panic!("open_as_secondary failed: {e}"));
        if let Err(e) = db.try_catch_up_with_primary() {
            eprintln!("warning: catch_up failed: {e}");
        }
        db
    } else {
        rocksdb::DB::open_for_read_only(&opts, std::path::Path::new(db_path), false)
            .unwrap_or_else(|e| panic!("open_for_read_only failed: {e}"))
    }
}

/// Stored values are a bincode `SnapshotBlob(Vec<u8>)`: an 8-byte LE length then the bytes.
fn blob(v: &[u8]) -> &[u8] {
    if v.len() >= 8 {
        let n = u64::from_le_bytes(v[..8].try_into().unwrap()) as usize;
        if n == v.len() - 8 {
            return &v[8..];
        }
    }
    v
}

fn scan(db: &rocksdb::DB, prefix: u8) -> Vec<(Hash, Vec<u8>)> {
    let mut out = Vec::new();
    for item in db.iterator(rocksdb::IteratorMode::From(&[prefix], rocksdb::Direction::Forward)) {
        let (k, v) = item.expect("iteration error");
        if k.first() != Some(&prefix) {
            break;
        }
        if k.len() != 33 {
            continue;
        }
        out.push((Hash::from_bytes(k[1..33].try_into().unwrap()), blob(&v).to_vec()));
    }
    out
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: dump_production_snapshot <consensus-db-path> [--full] [sample_hash]");
        std::process::exit(2);
    }
    let full = args.iter().any(|a| a == "--full");
    let only: Option<Hash> = args.iter().skip(2).find(|a| a.len() == 64).map(|a| Hash::from_str(a).expect("bad hash"));
    let db = open(&args[1]);
    if let Some(pos) = args.iter().position(|a| a == "--peek") {
        let start: u8 = args[pos + 1].parse().expect("prefix byte");
        let mut ro = rocksdb::ReadOptions::default();
        ro.set_total_order_seek(true);
        let mut n = 0;
        for item in db.iterator_opt(rocksdb::IteratorMode::From(&[start], rocksdb::Direction::Forward), ro) {
            let (k, v) = item.expect("iteration error");
            println!("key len={} head={} value len={}", k.len(), hex::encode(&k[..k.len().min(8)]), v.len());
            n += 1;
            if n >= 8 {
                break;
            }
        }
        return;
    }
    if let Some(pos) = args.iter().position(|a| a == "--histogram") {
        let start: u8 = args[pos + 1].parse().expect("prefix byte");
        let mut ro = rocksdb::ReadOptions::default();
        ro.set_total_order_seek(true);
        let mut counts = std::collections::BTreeMap::new();
        for item in db.iterator_opt(rocksdb::IteratorMode::From(&[start], rocksdb::Direction::Forward), ro) {
            let (k, _) = item.expect("iteration error");
            *counts.entry(k[0]).or_insert(0u64) += 1;
        }
        for (p, n) in counts {
            println!("prefix {p}: {n} keys");
        }
        return;
    }
    if args.iter().any(|a| a == "--prefixes") {
        for prefix in 0u8..=255 {
            let mut n = 0u64;
            for item in db.iterator(rocksdb::IteratorMode::From(&[prefix], rocksdb::Direction::Forward)) {
                let (k, _) = item.expect("iteration error");
                if k.first() != Some(&prefix) || n >= 5 {
                    break;
                }
                n += 1;
            }
            if n > 0 {
                println!("prefix {prefix}: >= {n} keys");
            }
        }
        return;
    }

    for (sample, bytes) in scan(&db, LEDGER_PREFIX) {
        if only.is_some_and(|o| o != sample) {
            continue;
        }
        match ServiceLedgerSnapshot::from_bytes(&bytes) {
            Ok(s) => println!(
                "LEDGER sample={sample} hash={} bytes={} recent_producers={}",
                ServiceLedgerSnapshot::hash_of_bytes(&bytes),
                bytes.len(),
                s.recent_producers.len()
            ),
            Err(e) => println!("LEDGER sample={sample} undecodable: {e}"),
        }
    }
    for (sample, bytes) in scan(&db, PRODUCTION_PREFIX) {
        if only.is_some_and(|o| o != sample) {
            continue;
        }
        let snap = match ProductionIndexSnapshot::from_bytes(&bytes) {
            Ok(s) => s,
            Err(e) => {
                println!("PRODUCTION sample={sample} undecodable: {e}");
                continue;
            }
        };
        let normalized = ProductionIndexSnapshot {
            bottom_index: 0,
            sample_index: snap.sample_index - snap.bottom_index,
            floors: snap.floors.clone(),
            entries: snap
                .entries
                .iter()
                .map(|(spk, e)| (spk.clone(), e.iter().map(|(i, c)| (i - snap.bottom_index, *c)).collect()))
                .collect(),
            window_daa: snap.window_daa.clone(),
        };
        let n_entries: usize = snap.entries.iter().map(|(_, e)| e.len()).sum();
        println!(
            "PRODUCTION sample={sample} hash={} encoding=v{} bottom_index={} sample_index={} span={} floors={} floors_sum={} groups={} entries={} window_daa={} [{}..{}] normalized_hash={}",
            ProductionIndexSnapshot::hash_of_bytes(&bytes),
            bytes.first().copied().unwrap_or(0),
            snap.bottom_index,
            snap.sample_index,
            snap.sample_index - snap.bottom_index,
            snap.floors.len(),
            snap.floors.iter().map(|(_, v)| *v).sum::<u64>(),
            snap.entries.len(),
            n_entries,
            snap.window_daa.len(),
            snap.window_daa.first().copied().unwrap_or(0),
            snap.window_daa.last().copied().unwrap_or(0),
            ProductionIndexSnapshot::hash_of_bytes(&normalized.to_bytes()),
        );
        println!("  relative_hash={}", snap.relative_hash());
        if full {
            for (spk, v) in normalized.floors.iter() {
                println!("  FLOOR {} {}", hex::encode(spk.script()), v);
            }
            for (spk, e) in normalized.entries.iter() {
                println!("  ENTRIES {} {:?}", hex::encode(spk.script()), e);
            }
            println!("  WINDOW_DAA {:?}", normalized.window_daa);
        }
    }
}
