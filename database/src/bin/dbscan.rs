// One-off diagnostic: bucket a consensus RocksDB by key-prefix byte (key+value bytes & entry count).
// Read-only open with the project's exact rocksdb version (SST footer compatible).
use rocksdb::{IteratorMode, Options, ReadOptions, DB};
use std::collections::BTreeMap;

fn main() {
    let path = std::env::args().nth(1).expect("usage: dbscan <db_path>");
    let mut opts = Options::default();
    opts.create_if_missing(false);
    let db = DB::open_for_read_only(&opts, &path, false).expect("open db read-only");

    let mut count: BTreeMap<u8, u64> = BTreeMap::new();
    let mut bytes: BTreeMap<u8, u64> = BTreeMap::new();

    let mut ro = ReadOptions::default();
    ro.set_verify_checksums(false);
    let iter = db.iterator_opt(IteratorMode::Start, ro);
    for item in iter {
        let (k, v) = item.expect("iter");
        if let Some(&p) = k.first() {
            *count.entry(p).or_default() += 1;
            *bytes.entry(p).or_default() += (k.len() + v.len()) as u64;
        }
    }

    let mut total = 0u64;
    let mut rows: Vec<(u8, u64, u64)> = bytes.iter().map(|(&p, &b)| (p, b, count[&p])).collect();
    rows.sort_by(|a, b| b.1.cmp(&a.1));
    println!("{:<6} {:<6} {:>12} {:>14}", "pref", "hex", "MB", "entries");
    for (p, b, c) in &rows {
        println!("{:<6} 0x{:02X}   {:>12.1} {:>14}", p, p, *b as f64 / 1048576.0, c);
        total += b;
    }
    println!("TOTAL  {:.1} MB", total as f64 / 1048576.0);
}
