// Throwaway diagnostic: print the VirtualState (prefix 28) raw value of a consensus RocksDB
// opened read-only, and scan it for LE u64s in a plausible DAA range to locate
// blue_score / daa_score without deserializing the full struct.
use rocksdb::{Options, DB};

fn main() {
    let path = std::env::args().nth(1).expect("usage: dbpeek <consensus_db_path>");
    let mut opts = Options::default();
    opts.create_if_missing(false);
    let db = DB::open_for_read_only(&opts, &path, false).expect("open db read-only");

    let val = db
        .get([28u8])
        .expect("read")
        .expect("no VirtualState key (prefix 28) in this db");
    println!("VirtualState value: {} bytes", val.len());

    // Any LE u64 in a broad plausible range; blue_score appears before daa_score in the layout.
    for off in 0..val.len().saturating_sub(8) {
        let v = u64::from_le_bytes(val[off..off + 8].try_into().unwrap());
        if (50_000_000..70_000_000).contains(&v) {
            println!("offset {:>6}: {}", off, v);
        }
        // timestamps (past_median_time, ms since epoch) for sanity
        if (1_700_000_000_000..1_800_000_000_000).contains(&v) {
            println!("offset {:>6}: {} (timestamp ms)", off, v);
        }
    }
}
