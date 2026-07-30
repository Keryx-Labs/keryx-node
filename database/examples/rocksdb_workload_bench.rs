//! Comparative benchmark of the node's dominant RocksDB write pattern, old options vs new.
//!
//! The workload replays what a 10-BPS Keryx node actually does to its consensus DB on every block:
//! one ~228 KB `PomProof` (K=256 walk steps, each with a Merkle path), a handful of small hot
//! consensus records (header, ghostdag, statuses, reachability, utxo diff, acceptance data), a few
//! point reads of those hot keys, and the proof-GC delete of the block that just left the retention
//! window — all committed in a single `WriteBatch`, exactly as `commit_header` / the virtual
//! processor do.
//!
//! It reports the numbers that decide whether a node fits on a spinning disk: write amplification
//! (how many physical bytes RocksDB writes per logical byte the node hands it), on-disk footprint,
//! wall-clock throughput, hot-read latency, and the index/filter memory that lives outside every
//! configured budget.
//!
//! Variants:
//! * `legacy-default` / `legacy-hdd` — the option sets as they were before the optimization patch,
//!   copied verbatim from `rocksdb_preset.rs` at commit 95c338fd.
//! * `opt-default` / `opt-hdd` — the current presets, applied through the real production code path
//!   (`RocksDbPreset::apply_to_options`), including the shared cache / memtable budget.
//!
//! Usage:
//!   cargo run --release -p keryx-database --example rocksdb_workload_bench -- \
//!       <variant> <db_dir> [blocks] [retention]

use keryx_database::prelude::{RocksDbPreset, RocksDbResources};
use rocksdb::{BlockBasedOptions, Cache, DBCompressionType, DBWithThreadMode, MultiThreaded, Options, WriteBatch};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Size of one post-H4 PoM proof: 256 walk steps x (32 B chunk + ~31 x 32 B Merkle path).
const POM_PROOF_BYTES: usize = 228 * 1024;
/// Shared cache / memtable budgets handed to the optimized variants (the daemon's defaults at
/// `--ram-scale=1.0`).
const SHARED_CACHE_BYTES: usize = 256 * 1024 * 1024;
const SHARED_MEMTABLE_BYTES_DEFAULT: usize = 256 * 1024 * 1024;
const SHARED_MEMTABLE_BYTES_HDD: usize = 768 * 1024 * 1024;
/// Same file budget for every variant, so the comparison isn't skewed by FD limits.
const MAX_OPEN_FILES: i32 = 500;
/// Hot-key prefixes, mirroring `DatabaseStorePrefixes`: (prefix, value size).
const HOT_RECORDS: &[(u8, usize)] = &[
    (0x01, 200),  // headers
    (0x02, 500),  // ghostdag
    (0x03, 1),    // statuses
    (0x04, 80),   // reachability
    (0x05, 1024), // utxo diffs
    (0x06, 500),  // acceptance data
    (0x07, 120),  // daa / depth
    (0x08, 64),   // relations
];
const POM_PREFIX: u8 = 0xF0;

type Db = DBWithThreadMode<MultiThreaded>;

// ---------------------------------------------------------------------------------------------
// Option sets
// ---------------------------------------------------------------------------------------------

/// Pre-patch `apply_default`: parallelism + level-style compaction, nothing else. No block cache
/// is configured, so RocksDB falls back to its 8 MB per-DB default; index and filter blocks live
/// outside it entirely.
fn legacy_default(opts: &mut Options, parallelism: usize) {
    if parallelism > 1 {
        opts.increase_parallelism(parallelism as i32);
    }
    opts.optimize_level_style_compaction(64 * 1024 * 1024);
}

/// Pre-patch `apply_hdd`, verbatim.
fn legacy_hdd(opts: &mut Options, parallelism: usize) {
    if parallelism > 1 {
        opts.increase_parallelism(parallelism as i32);
    }
    let write_buffer_size = 256 * 1024 * 1024;
    opts.optimize_level_style_compaction(write_buffer_size);
    opts.set_write_buffer_size(write_buffer_size);
    opts.set_target_file_size_base(256 * 1024 * 1024);
    opts.set_target_file_size_multiplier(1);
    opts.set_max_bytes_for_level_base(1024 * 1024 * 1024);
    opts.set_level_compaction_dynamic_level_bytes(true);
    opts.set_level_zero_file_num_compaction_trigger(1);
    opts.set_compaction_pri(rocksdb::CompactionPri::OldestSmallestSeqFirst);
    opts.set_compaction_readahead_size(4 * 1024 * 1024);
    opts.set_compression_type(DBCompressionType::Lz4);
    opts.set_bottommost_compression_type(DBCompressionType::Zstd);
    opts.set_compression_options(-1, 22, 0, 64 * 1024);
    opts.set_zstd_max_train_bytes(8 * 1024 * 1024);
    let mut block_opts = BlockBasedOptions::default();
    block_opts.set_bloom_filter(18.0, false);
    block_opts.set_partition_filters(true);
    block_opts.set_format_version(5);
    block_opts.set_index_type(rocksdb::BlockBasedIndexType::TwoLevelIndexSearch);
    block_opts.set_cache_index_and_filter_blocks(true);
    // A per-connection cache, as before: every DB the node opened built its own.
    block_opts.set_block_cache(&Cache::new_lru_cache(256 * 1024 * 1024));
    block_opts.set_block_size(256 * 1024);
    opts.set_block_based_table_factory(&block_opts);
    opts.set_ratelimiter(12 * 1024 * 1024, 100_000, 10);
    opts.set_enable_blob_files(true);
    opts.set_min_blob_size(512);
    opts.set_blob_file_size(256 * 1024 * 1024);
    opts.set_blob_compression_type(DBCompressionType::Zstd);
    opts.set_enable_blob_gc(true);
    opts.set_blob_gc_age_cutoff(0.9);
    opts.set_blob_gc_force_threshold(0.1);
    opts.set_blob_compaction_readahead_size(8 * 1024 * 1024);
}

fn build_options(variant: &str, parallelism: usize) -> Options {
    let mut opts = Options::default();
    match variant {
        "legacy-default" => legacy_default(&mut opts, parallelism),
        "legacy-hdd" => legacy_hdd(&mut opts, parallelism),
        "opt-default" => {
            let resources = RocksDbResources::new(SHARED_CACHE_BYTES, SHARED_MEMTABLE_BYTES_DEFAULT, None);
            RocksDbPreset::Default.apply_to_options(&mut opts, parallelism, 64 * 1024 * 1024, Some(&resources));
        }
        "opt-hdd" => {
            let resources = RocksDbResources::new(SHARED_CACHE_BYTES, SHARED_MEMTABLE_BYTES_HDD, None);
            RocksDbPreset::Hdd.apply_to_options(&mut opts, parallelism, 64 * 1024 * 1024, Some(&resources));
        }
        // opt-hdd plus periodic compaction. With ~99.8% of the bytes in blob files the LSM holds
        // almost no SST, so RocksDB's size-based compaction triggers rarely fire — and blob GC only
        // runs inside an SST compaction. Periodic compaction gives it a time-based trigger instead,
        // which is the only thing that can reclaim blob garbage in this shape. 30s here so the
        // mechanism is observable inside a ~2 minute run; production would use minutes-to-hours.
        "opt-hdd-periodic" => {
            let resources = RocksDbResources::new(SHARED_CACHE_BYTES, SHARED_MEMTABLE_BYTES_HDD, None);
            RocksDbPreset::Hdd.apply_to_options(&mut opts, parallelism, 64 * 1024 * 1024, Some(&resources));
            opts.set_periodic_compaction_seconds(30);
        }
        "hdd-qd1" => {
            let resources = RocksDbResources::new(SHARED_CACHE_BYTES, SHARED_MEMTABLE_BYTES_HDD, Some(30 * 1024 * 1024));
            RocksDbPreset::HddQd1.apply_to_options(&mut opts, 1, 64 * 1024 * 1024, Some(&resources));
        }
        other => panic!("unknown variant '{other}' (legacy-default | legacy-hdd | opt-default | opt-hdd | hdd-qd1)"),
    }
    opts.enable_statistics();
    // hdd-qd1 deliberately keeps every file open; the others share one budget for comparability.
    if variant != "hdd-qd1" {
        opts.set_max_open_files(MAX_OPEN_FILES);
    }
    opts.create_if_missing(true);
    opts
}

// ---------------------------------------------------------------------------------------------
// Workload
// ---------------------------------------------------------------------------------------------

/// Deterministic per-block key: prefix || 32-byte "hash" derived from the index.
fn key(prefix: u8, index: u64) -> Vec<u8> {
    let mut k = Vec::with_capacity(33);
    k.push(prefix);
    // Spread indices across the keyspace the way block hashes do, so writes are not append-ordered.
    let scattered = index.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    k.extend_from_slice(&scattered.to_be_bytes());
    k.extend_from_slice(&index.to_be_bytes());
    k.extend_from_slice(&[0u8; 16]);
    k
}

/// A proof-shaped payload: Merkle paths are hash bytes, i.e. incompressible. Deriving them from the
/// index keeps the bytes pseudo-random rather than a compressible constant run.
fn proof_payload(index: u64) -> Vec<u8> {
    let mut v = vec![0u8; POM_PROOF_BYTES];
    let mut state = index.wrapping_mul(0xD6E8_FEB8_6659_FD93).wrapping_add(1);
    for chunk in v.chunks_mut(8) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let bytes = state.to_le_bytes();
        chunk.copy_from_slice(&bytes[..chunk.len()]);
    }
    v
}

struct Report {
    wall: Duration,
    logical_bytes: u64,
    flush_write_bytes: u64,
    compact_write_bytes: u64,
    compact_read_bytes: u64,
    on_disk_bytes: u64,
    sst_bytes: u64,
    blob_bytes: u64,
    table_readers_mem: u64,
    block_cache_usage: u64,
    hot_read: Duration,
    hot_reads: u64,
    settle: Duration,
    /// Footprint and physical writes as the run left them, before the forced full compaction.
    on_disk_before_compact: u64,
    physical_before_compact: u64,
    full_compact: Duration,
    /// True when compaction was still pending at the settle deadline — the run left unpaid
    /// compaction debt, so its write-amplification and footprint numbers are optimistic.
    settle_timed_out: bool,
}

fn run(variant: &str, db_dir: &Path, blocks: u64, retention: u64) -> Report {
    let parallelism = num_cpus::get();
    let opts = build_options(variant, parallelism);
    let db = Db::open(&opts, db_dir).expect("failed to open DB");

    let mut logical_bytes = 0u64;
    let started = Instant::now();
    let mut hot_read = Duration::ZERO;
    let mut hot_reads = 0u64;

    for index in 0..blocks {
        let mut batch = WriteBatch::default();

        // The block's PoM proof.
        let proof = proof_payload(index);
        batch.put(key(POM_PREFIX, index), &proof);
        logical_bytes += (POM_PROOF_BYTES + 33) as u64;

        // The small hot consensus records committed alongside it.
        for &(prefix, size) in HOT_RECORDS {
            let value = vec![(index % 251) as u8; size];
            batch.put(key(prefix, index), &value);
            logical_bytes += (size + 33) as u64;
        }

        // Proof GC: the block that just left the retention window.
        if index >= retention {
            batch.delete(key(POM_PREFIX, index - retention));
        }

        db.write(batch).expect("batch write failed");

        // Point reads of recent hot keys, as validation of a new block does against its ancestors.
        if index % 4 == 0 && index > 0 {
            let read_started = Instant::now();
            for &(prefix, _) in HOT_RECORDS.iter().take(4) {
                let target = index.saturating_sub(1 + (index % 64));
                let _ = db.get_pinned(key(prefix, target)).expect("read failed");
                hot_reads += 1;
            }
            hot_read += read_started.elapsed();
        }
    }

    // Let RocksDB settle so pending compactions are counted, not left as unpaid debt that would
    // flatter whichever variant queued more of it.
    db.flush().expect("flush failed");
    let settle_deadline = Duration::from_secs(900);
    let settle_started = Instant::now();
    let mut settle_timed_out = true;
    while settle_started.elapsed() < settle_deadline {
        let pending = db.property_int_value("rocksdb.compaction-pending").ok().flatten().unwrap_or(0);
        let running = db.property_int_value("rocksdb.num-running-compactions").ok().flatten().unwrap_or(0);
        if pending == 0 && running == 0 {
            settle_timed_out = false;
            break;
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    let settle = settle_started.elapsed();
    let wall = started.elapsed();

    // Footprint and physical writes before reclaiming any deferred garbage.
    let on_disk_before_compact = dir_size(db_dir, None);
    let stats_before = opts.get_statistics().unwrap_or_default();
    let physical_before_compact =
        ticker(&stats_before, "rocksdb.flush.write.bytes") + ticker(&stats_before, "rocksdb.compact.write.bytes");

    // A variant that defers compaction (and with it blob GC) shows a smaller amplification and a
    // larger footprint than one that pays as it goes — the two are not comparable until the
    // deferred work is done. Forcing a full compaction settles which of the two is happening: if
    // the extra bytes are uncollected garbage they disappear here, and the I/O it took to collect
    // them lands in the post-compaction amplification.
    let compact_started = Instant::now();
    db.compact_range::<&[u8], &[u8]>(None, None);
    let full_compact = compact_started.elapsed();

    // Ticker lines are `rocksdb.<name> COUNT : <value>`.
    fn ticker(stats: &str, name: &str) -> u64 {
        stats
            .lines()
            .find(|line| line.split_whitespace().next() == Some(name))
            .and_then(|line| line.rsplit(':').next())
            .and_then(|value| value.trim().parse().ok())
            .unwrap_or(0)
    }
    let stats = opts.get_statistics().unwrap_or_default();
    let ticker = |name: &str| ticker(&stats, name);

    let report = Report {
        wall,
        logical_bytes,
        flush_write_bytes: ticker("rocksdb.flush.write.bytes"),
        compact_write_bytes: ticker("rocksdb.compact.write.bytes"),
        compact_read_bytes: ticker("rocksdb.compact.read.bytes"),
        on_disk_bytes: dir_size(db_dir, None),
        sst_bytes: dir_size(db_dir, Some("sst")),
        blob_bytes: dir_size(db_dir, Some("blob")),
        table_readers_mem: db.property_int_value("rocksdb.estimate-table-readers-mem").ok().flatten().unwrap_or(0),
        block_cache_usage: db.property_int_value("rocksdb.block-cache-usage").ok().flatten().unwrap_or(0),
        hot_read,
        hot_reads,
        settle,
        settle_timed_out,
        on_disk_before_compact,
        physical_before_compact,
        full_compact,
    };
    drop(db);
    report
}

/// Total size of `dir`, optionally restricted to files with the given extension.
fn dir_size(dir: &Path, extension: Option<&str>) -> u64 {
    let Ok(entries) = std::fs::read_dir(dir) else { return 0 };
    entries
        .filter_map(Result::ok)
        .map(|entry| match entry.file_type() {
            Ok(ft) if ft.is_dir() => dir_size(&entry.path(), extension),
            Ok(_) => match extension {
                Some(ext) if entry.path().extension().and_then(|e| e.to_str()) != Some(ext) => 0,
                _ => entry.metadata().map(|m| m.len()).unwrap_or(0),
            },
            Err(_) => 0,
        })
        .sum()
}

fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: {} <variant> <db_dir> [blocks] [retention]", args[0]);
        eprintln!("  variant: legacy-default | opt-default | legacy-hdd | opt-hdd");
        std::process::exit(2);
    }
    let variant = args[1].clone();
    let db_dir = PathBuf::from(&args[2]);
    let blocks: u64 = args.get(3).map(|v| v.parse().expect("blocks")).unwrap_or(2_000);
    let retention: u64 = args.get(4).map(|v| v.parse().expect("retention")).unwrap_or(800);

    if db_dir.exists() {
        std::fs::remove_dir_all(&db_dir).expect("failed to clear db dir");
    }
    std::fs::create_dir_all(&db_dir).expect("failed to create db dir");

    println!("variant={variant} blocks={blocks} retention={retention} dir={}", db_dir.display());
    let r = run(&variant, &db_dir, blocks, retention);

    let physical = r.flush_write_bytes + r.compact_write_bytes;
    let amplification = if r.logical_bytes > 0 { physical as f64 / r.logical_bytes as f64 } else { 0.0 };
    let amplification_before = if r.logical_bytes > 0 { r.physical_before_compact as f64 / r.logical_bytes as f64 } else { 0.0 };
    let hot_read_us = if r.hot_reads > 0 { r.hot_read.as_secs_f64() * 1e6 / r.hot_reads as f64 } else { 0.0 };

    println!("RESULT variant={variant}");
    println!("  wall_seconds            {:.1}", r.wall.as_secs_f64());
    println!("  write_seconds           {:.1}", (r.wall - r.settle).as_secs_f64());
    println!(
        "  settle_seconds          {:.1}{}",
        r.settle.as_secs_f64(),
        if r.settle_timed_out { " (TIMED OUT - unpaid compaction debt)" } else { "" }
    );
    println!("  blocks_per_second       {:.1}", blocks as f64 / (r.wall - r.settle).as_secs_f64());
    println!("  logical_written_MiB     {:.1}", mib(r.logical_bytes));
    println!("  flush_written_MiB       {:.1}", mib(r.flush_write_bytes));
    println!("  compaction_written_MiB  {:.1}", mib(r.compact_write_bytes));
    println!("  compaction_read_MiB     {:.1}", mib(r.compact_read_bytes));
    println!("  physical_written_MiB    {:.1}", mib(physical));
    println!("  write_amplification     {amplification:.2}x  (as-left: {amplification_before:.2}x)");
    println!("  full_compact_seconds    {:.1}", r.full_compact.as_secs_f64());
    println!("  on_disk_asleft_MiB      {:.1}", mib(r.on_disk_before_compact));
    println!("  on_disk_MiB             {:.1}", mib(r.on_disk_bytes));
    println!("  on_disk_sst_MiB         {:.1}", mib(r.sst_bytes));
    println!("  on_disk_blob_MiB        {:.1}", mib(r.blob_bytes));
    println!("  table_readers_mem_MiB   {:.1}", mib(r.table_readers_mem));
    println!("  block_cache_usage_MiB   {:.1}", mib(r.block_cache_usage));
    println!("  hot_read_latency_us     {hot_read_us:.1}");
}
