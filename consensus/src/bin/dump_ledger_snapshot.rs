//! Write the raw service-ledger snapshot bytes stored for one sample block (read-only RocksDB;
//! secondary mode with KERYX_DUMP_SECONDARY so a running node can be read).
use keryx_hashes::Hash;
use std::env;
use std::str::FromStr;

const LEDGER_PREFIX: u8 = 201;

fn open(db_path: &str) -> rocksdb::DB {
    let mut opts = rocksdb::Options::default();
    opts.set_max_open_files(256);
    if env::var("KERYX_DUMP_SECONDARY").is_ok() {
        let secondary = env::temp_dir().join(format!("keryx-ledger-secondary-{}", std::process::id()));
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

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: dump_ledger_snapshot <consensus-db-path> <sample_hash> <out_file>");
        std::process::exit(2);
    }
    let sample = Hash::from_str(&args[2]).expect("bad hash");
    let db = open(&args[1]);
    let mut key = vec![LEDGER_PREFIX];
    key.extend_from_slice(&sample.as_bytes());
    match db.get(&key).expect("read error") {
        Some(v) => {
            let bytes = blob(&v);
            std::fs::write(&args[3], bytes).expect("write out_file");
            println!("LEDGER sample={sample} bytes={} written to {}", bytes.len(), args[3]);
        }
        None => {
            println!("LEDGER sample={sample} <absent>");
            std::process::exit(1);
        }
    }
}
