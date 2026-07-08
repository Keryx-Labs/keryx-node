//! Read-only secondary-mode reader for the ratio-reward production index (throwaway debug tool).
//! Opens the live consensus RocksDB as a SECONDARY (no lock contention with the running node) and
//! answers point queries over the selected-chain and windowed-production-prefix stores so an external
//! driver can reconstruct `windowed_production` exactly, using the same key encodings the node writes.
//!
//! Prefixes (keryx_database::registry): ChainHashByIndex=20, ChainIndexByHash=21,
//! WindowedProductionPrefix=46, WindowedProductionFloor=47. U64Key = le64. Prefix value = le64.
//! cumulative_at = reverse-seek within `46||bucket` for the largest index <= target (le64 value),
//! else floor `47||bucket`.

use std::io::{self, BufRead, Write};

use rocksdb::{DB, Direction, IteratorMode, Options, ReadOptions};

fn hex_dec(s: &str) -> Vec<u8> {
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
}
fn hex_enc(b: &[u8]) -> String {
    b.iter().map(|x| format!("{:02x}", x)).collect()
}
fn le64(v: &[u8]) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&v[..8]);
    u64::from_le_bytes(a)
}

fn cumulative_at(db: &DB, bucket: &[u8], index: u64) -> u64 {
    let mut range = vec![46u8];
    range.extend_from_slice(bucket);
    let mut seek = range.clone();
    seek.extend_from_slice(&index.to_be_bytes());
    let mut ro = ReadOptions::default();
    ro.set_iterate_range(rocksdb::PrefixRange(range.as_slice()));
    let mut it = db.iterator_opt(IteratorMode::From(seek.as_slice(), Direction::Reverse), ro);
    if let Some(Ok((_k, v))) = it.next() {
        return le64(&v);
    }
    // floor 47||bucket
    let mut fk = vec![47u8];
    fk.extend_from_slice(bucket);
    match db.get(&fk).unwrap() {
        Some(v) => le64(&v),
        None => 0,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let primary = &args[1];
    let secondary = &args[2];
    let mut opts = Options::default();
    opts.create_if_missing(false);
    let db = DB::open_as_secondary(&opts, primary, secondary).expect("open_as_secondary");
    db.try_catch_up_with_primary().ok();

    let stdin = io::stdin();
    let mut out = io::stdout();
    for line in stdin.lock().lines() {
        let line = line.unwrap();
        let t: Vec<&str> = line.split_whitespace().collect();
        if t.is_empty() {
            continue;
        }
        match t[0] {
            "catchup" => {
                db.try_catch_up_with_primary().ok();
                writeln!(out, "ok").unwrap();
            }
            // h <index> -> hash hex at ChainHashByIndex
            "h" => {
                let idx: u64 = t[1].parse().unwrap();
                let mut k = vec![20u8];
                k.extend_from_slice(&idx.to_le_bytes());
                match db.get(&k).unwrap() {
                    Some(v) => writeln!(out, "{}", hex_enc(&v)).unwrap(),
                    None => writeln!(out, "NONE").unwrap(),
                }
            }
            // i <hash_hex> -> chain index at ChainIndexByHash
            "i" => {
                let h = hex_dec(t[1]);
                let mut k = vec![21u8];
                k.extend_from_slice(&h);
                match db.get(&k).unwrap() {
                    Some(v) => writeln!(out, "{}", le64(&v)).unwrap(),
                    None => writeln!(out, "NONE").unwrap(),
                }
            }
            // c <bucket_hex> <index> -> cumulative_at
            "c" => {
                let bucket = hex_dec(t[1]);
                let idx: u64 = t[2].parse().unwrap();
                writeln!(out, "{}", cumulative_at(&db, &bucket, idx)).unwrap();
            }
            // e <bucket_hex> -> dump all (index cumulative) entries for this SPK
            "e" => {
                let bucket = hex_dec(t[1]);
                let mut range = vec![46u8];
                range.extend_from_slice(&bucket);
                let mut ro = ReadOptions::default();
                ro.set_iterate_range(rocksdb::PrefixRange(range.as_slice()));
                let it = db.iterator_opt(IteratorMode::From(range.as_slice(), Direction::Forward), ro);
                let mut n = 0u64;
                for item in it {
                    let (k, v) = item.unwrap();
                    let mut ib = [0u8; 8];
                    ib.copy_from_slice(&k[k.len() - 8..]);
                    let index = u64::from_be_bytes(ib);
                    writeln!(out, "E {} {}", index, le64(&v)).unwrap();
                    n += 1;
                }
                writeln!(out, "ecount {}", n).unwrap();
            }
            // f <bucket_hex> -> floor value
            "f" => {
                let bucket = hex_dec(t[1]);
                let mut fk = vec![47u8];
                fk.extend_from_slice(&bucket);
                match db.get(&fk).unwrap() {
                    Some(v) => writeln!(out, "{}", le64(&v)).unwrap(),
                    None => writeln!(out, "0").unwrap(),
                }
            }
            // raw <prefix_num> [n] -> dump first n raw keys (hex) with that prefix byte
            "raw" => {
                let pfx: u8 = t[1].parse().unwrap();
                let n: usize = t.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);
                let range = vec![pfx];
                let mut ro = ReadOptions::default();
                ro.set_iterate_range(rocksdb::PrefixRange(range.as_slice()));
                let it = db.iterator_opt(IteratorMode::From(range.as_slice(), Direction::Forward), ro);
                let mut c = 0usize;
                for item in it {
                    if c >= n { break; }
                    let (k, v) = item.unwrap();
                    writeln!(out, "K {} = {}", hex_enc(&k), hex_enc(&v)).unwrap();
                    c += 1;
                }
                writeln!(out, "rawcount_shown {}", c).unwrap();
            }
            _ => writeln!(out, "?").unwrap(),
        }
        out.flush().unwrap();
    }
}
