//! RocksDB configuration presets for different use cases
//!
//! This module provides pre-configured RocksDB option sets optimized for different
//! deployment scenarios.

use rocksdb::{Cache, Options, WriteBufferManager};
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, Ordering};

/// Kill-switch for key-value separation, set once at startup (`--rocksdb-no-blob-files`).
/// Existing blob files remain readable: only new writes return inline to the LSM, and
/// compaction drains the old blob files over time.
static BLOB_FILES_DISABLED: AtomicBool = AtomicBool::new(false);

/// Disable blob-file writing for every database opened after this call.
pub fn disable_blob_files() {
    BLOB_FILES_DISABLED.store(true, Ordering::Relaxed);
}

fn blob_files_disabled() -> bool {
    BLOB_FILES_DISABLED.load(Ordering::Relaxed)
}

/// Values at/above this size are written to blob files instead of being carried inline in the LSM
/// (RocksDB key-value separation, aka BlobDB).
///
/// The node writes one ~296 KB v4 `PomProof` per block at 10 BPS. Inline, every one of those is
/// rewritten by every compaction that touches its SST — an order-of-magnitude write amplification
/// on a value that is written once, read rarely (relay re-serve only) and deleted wholesale by the
/// proof GC. Separated, the LSM carries only a small pointer, so compaction moves metadata instead
/// of payload and the block cache stops being flushed by proof bytes.
///
/// 4 KiB keeps every hot consensus record (ghostdag, reachability, headers, UTXO entries) inline —
/// only proofs and large block bodies cross the threshold.
const BLOB_MIN_VALUE_BYTES: u64 = 4 * 1024;

/// ZSTD level for the bottommost level of the HDD preset. Level 22 (the previous value) costs
/// ~50x the CPU of level 6 for a few percent of size on already-compact binary records, and on a
/// spinning disk the bottleneck is compaction throughput, not the last few percent of space.
const HDD_BOTTOMMOST_ZSTD_LEVEL: i32 = 6;

/// Default blob-file size for the SSD preset. A v4 proof is ~296 KB; 128 MB files hold ~430
/// proofs each, matching the 1500-DAA serve window without creating a swarm of tiny blob files.
const SSD_BLOB_FILE_BYTES: u64 = 128 * 1024 * 1024;

/// Floor for the dedicated blob cache. ~296 KB × 1500 DAA serve-window proofs ≈ 450 MB if every
/// DAA carried one proof; at 10 BPS the selected-chain window of 2000 chain blocks is ~2000
/// live proofs ≈ 600 MB. We cannot pin the whole window on a 4 GB box, but 256 MB keeps the
/// hottest ~850 proofs off the SST LRU so header/UTXO reads stay in cache during catch-up.
pub const DA_1500_BLOB_CACHE_MIN_BYTES: usize = 256 * 1024 * 1024;

/// Default background-write rate limit for the HDD preset, in bytes/s.
///
/// The logical write rate of a 10-BPS node is ~2.5 MB/s; with compaction amplification the
/// physical rate is several times that. The previous 12 MB/s limit sat below the sustained
/// requirement, so compaction debt accumulated until RocksDB stalled writes. 48 MB/s still
/// smooths I/O spikes on a spinning disk while leaving compaction able to keep up.
/// Override with `--rocksdb-rate-limit-mb`.
pub const DEFAULT_HDD_RATE_LIMIT_BYTES_PER_SEC: u64 = 48 * 1024 * 1024;

/// Memory resources shared by **every** RocksDB instance the process opens (meta, active
/// consensus, staging consensus, utxoindex).
///
/// Both RocksDB objects held here are handles over a single shared allocation, so passing one
/// `RocksDbResources` to all connections gives the process **one** block-cache budget and **one**
/// memtable budget. Building them per-connection (the previous behavior) multiplied whatever the
/// operator asked for by the number of open databases — `--rocksdb-cache-size=2048` allocated 8 GB,
/// and the HDD preset's 256 MB write buffer could reach 1.5 GB of memtables *per database*.
#[derive(Clone)]
pub struct RocksDbResources {
    block_cache: Cache,
    /// Dedicated blob cache so ~296 KB v4 PoM proofs do not evict hot SST blocks from `block_cache`.
    blob_cache: Cache,
    write_buffer_manager: WriteBufferManager,
    rate_limit_bytes_per_sec: Option<u64>,
}

// The RocksDB handles held here are opaque, so the only thing worth printing is the one tunable.
impl std::fmt::Debug for RocksDbResources {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RocksDbResources").field("rate_limit_bytes_per_sec", &self.rate_limit_bytes_per_sec).finish_non_exhaustive()
    }
}

impl RocksDbResources {
    /// * `block_cache_bytes` — total block cache across all DBs (holds data blocks plus, since the
    ///   presets set `cache_index_and_filter_blocks`, the SST index and filter blocks).
    /// * `write_buffer_bytes` — total memtable allowance across all DBs. RocksDB flushes early
    ///   rather than stalling writers when the budget is reached (`allow_stall = false`).
    /// * `rate_limit_bytes_per_sec` — background-write rate limit (HDD presets only).
    ///
    /// The blob cache is sized at 3/4 of the block cache, floored at
    /// [`DA_1500_BLOB_CACHE_MIN_BYTES`], so the 1500-DAA proof window does not flush UTXO/header
    /// SST blocks. No extra operator flag — `--rocksdb-cache-size` / `--ram-scale` grow both.
    pub fn new(block_cache_bytes: usize, write_buffer_bytes: usize, rate_limit_bytes_per_sec: Option<u64>) -> Self {
        let blob_cache_bytes = (block_cache_bytes.saturating_mul(3) / 4).max(DA_1500_BLOB_CACHE_MIN_BYTES);
        Self {
            block_cache: Cache::new_lru_cache(block_cache_bytes),
            blob_cache: Cache::new_lru_cache(blob_cache_bytes),
            write_buffer_manager: WriteBufferManager::new_write_buffer_manager(write_buffer_bytes, false),
            rate_limit_bytes_per_sec,
        }
    }
}

/// Available RocksDB configuration presets
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RocksDbPreset {
    /// Default configuration - balanced for general use on SSD/NVMe
    /// - 64MB write buffer
    /// - Standard compression
    /// - BlobDB for large values (keeps PoM proofs out of LSM compaction)
    /// - Optimized for fast storage
    #[default]
    Default,

    /// HDD configuration - optimized for hard disk drives
    /// - 256MB write buffer (4x default)
    /// - Aggressive compression (LZ4 + ZSTD)
    /// - BlobDB enabled for large values
    /// - Autotuned rate limiting to prevent I/O spikes
    /// - Optimized for sequential writes and reduced write amplification
    ///
    /// Recommended for archival nodes on HDD storage.
    Hdd,

    /// HDD tuned for queue-depth-1 transports (USB Bulk-Only Transport / no NCQ).
    /// One flush + one compaction thread, larger readaheads, all files kept open.
    /// Prefer plain `hdd` on native SATA where NCQ makes concurrency free.
    HddQd1,
}

impl FromStr for RocksDbPreset {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "default" => Ok(Self::Default),
            "hdd" => Ok(Self::Hdd),
            "hdd-qd1" | "hdd_qd1" => Ok(Self::HddQd1),
            _ => Err(format!("Unknown RocksDB preset: '{}'. Valid options: default, hdd, hdd-qd1", s)),
        }
    }
}

impl std::fmt::Display for RocksDbPreset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Default => write!(f, "default"),
            Self::Hdd => write!(f, "hdd"),
            Self::HddQd1 => write!(f, "hdd-qd1"),
        }
    }
}

impl RocksDbPreset {
    /// Apply the preset configuration to RocksDB options
    ///
    /// # Arguments
    /// * `opts` - RocksDB options to configure
    /// * `parallelism` - Number of background threads
    /// * `mem_budget` - Memtable memory budget (only used for Default preset, HDD uses fixed 256MB)
    /// * `resources` - Process-wide shared cache / memtable budget. `None` (tests, tooling) keeps
    ///   the per-connection defaults.
    pub fn apply_to_options(&self, opts: &mut Options, parallelism: usize, mem_budget: usize, resources: Option<&RocksDbResources>) {
        match self {
            Self::Default => self.apply_default(opts, parallelism, mem_budget, resources),
            Self::Hdd => self.apply_hdd(opts, parallelism, resources),
            Self::HddQd1 => self.apply_hdd_qd1(opts, resources),
        }
    }

    /// Apply default preset configuration
    fn apply_default(&self, opts: &mut Options, parallelism: usize, mem_budget: usize, resources: Option<&RocksDbResources>) {
        if parallelism > 1 {
            opts.increase_parallelism(parallelism as i32);
        }

        // Use the provided memory budget (typically 64MB)
        opts.optimize_level_style_compaction(mem_budget);

        // `optimize_level_style_compaction` leaves `max_write_buffer_number` at 6, so the effective
        // memtable ceiling is 6x the write buffer it just derived (`mem_budget / 4`). Pin it to a
        // value that still allows a flush to overlap incoming writes without hoarding memory.
        opts.set_max_write_buffer_number(4);

        let mut block_opts = rocksdb::BlockBasedOptions::default();
        block_opts.set_bloom_filter(10.0, false);
        block_opts.set_format_version(5);
        if let Some(resources) = resources {
            // Index and filter blocks are only charged against the block cache when there *is* a
            // shared cache to charge them to. Without this, they live outside every budget and grow
            // with the database (the node keeps up to `fd_budget / 2` SSTs open per consensus DB).
            block_opts.set_cache_index_and_filter_blocks(true);
            block_opts.set_pin_l0_filter_and_index_blocks_in_cache(true);
            block_opts.set_block_cache(&resources.block_cache);
        }
        opts.set_block_based_table_factory(&block_opts);

        // Key-value separation: see `BLOB_MIN_VALUE_BYTES`.
        if !blob_files_disabled() {
            opts.set_enable_blob_files(true);
            opts.set_min_blob_size(BLOB_MIN_VALUE_BYTES);
            opts.set_blob_file_size(SSD_BLOB_FILE_BYTES);
            opts.set_blob_compression_type(rocksdb::DBCompressionType::Lz4);
            opts.set_enable_blob_gc(true);
            // Fraction of the oldest blob files eligible for relocation during compaction. At 0.9
            // nearly every file qualifies, and live blobs are rewritten far faster than they are
            // created. 0.25 keeps the 1500-DAA live window (proofs still being re-served) out of GC.
            opts.set_blob_gc_age_cutoff(0.25);
            opts.set_blob_gc_force_threshold(0.2);
        }

        if let Some(resources) = resources {
            opts.set_blob_cache(&resources.blob_cache);
            opts.set_write_buffer_manager(&resources.write_buffer_manager);
        }

        // Smooth the 10 BPS × ~296 KB proof write stream so flushes do not stall the body processor.
        opts.set_bytes_per_sync(2 * 1024 * 1024);
        opts.set_wal_bytes_per_sync(2 * 1024 * 1024);
        opts.set_avoid_unnecessary_blocking_io(true);
    }

    /// Apply HDD preset configuration (HDD-optimized settings)
    fn apply_hdd(&self, opts: &mut Options, parallelism: usize, resources: Option<&RocksDbResources>) {
        if parallelism > 1 {
            opts.increase_parallelism(parallelism as i32);
        }

        // Memory and write buffer settings (256MB for better batching on HDD)
        let write_buffer_size = 256 * 1024 * 1024; // 256MB

        // Optimize for level-style compaction with archive-appropriate memory
        // This sets up LSM tree parameters
        opts.optimize_level_style_compaction(write_buffer_size);

        // Re-set write_buffer_size after optimize_level_style_compaction()
        // because optimize_level_style_compaction() internally overrides it to size/4
        opts.set_write_buffer_size(write_buffer_size);

        // ...and cap how many of those 256MB buffers may exist at once. The value left behind by
        // `optimize_level_style_compaction` is 6, i.e. up to 1.5 GB of memtables *per database* —
        // multiplied by every DB the node opens. 3 keeps room for a flush to overlap writes.
        opts.set_max_write_buffer_number(3);

        // LSM Tree Structure - Optimized for large (4TB+) archives
        // 256 MB SST files reduce file count dramatically (500K → 16K files for 4TB)
        opts.set_target_file_size_base(256 * 1024 * 1024); // 256 MB SST files
        opts.set_target_file_size_multiplier(1); // Same size across all levels
        opts.set_max_bytes_for_level_base(1024 * 1024 * 1024); // 1 GB L1 base
        opts.set_level_compaction_dynamic_level_bytes(true); // Minimize space amplification

        // Compaction settings
        // Compacting L0 at every single file minimizes read amplification but forces a compaction
        // per flush — the opposite of what a spinning disk wants, since each pass rewrites data
        // that the next flush would have let it batch. 4 trades a little read amplification for
        // meaningfully less write amplification.
        opts.set_level_zero_file_num_compaction_trigger(4);

        // Prioritize compacting older/smaller files first
        use rocksdb::CompactionPri;
        opts.set_compaction_pri(CompactionPri::OldestSmallestSeqFirst);

        // Read-ahead for compactions (4MB - good for sequential HDD reads)
        opts.set_compaction_readahead_size(4 * 1024 * 1024);

        // Compression strategy: LZ4 for all levels, ZSTD for bottommost
        use rocksdb::DBCompressionType;

        // Set default compression to LZ4 (fast)
        opts.set_compression_type(DBCompressionType::Lz4);

        // Enable bottommost level compression with ZSTD
        opts.set_bottommost_compression_type(DBCompressionType::Zstd);

        // ZSTD options for the bottommost level only, so the level choice cannot leak into the LZ4
        // levels. Larger dictionaries (64 KB) improve compression on large archives.
        opts.set_bottommost_compression_options(
            -1,                        // window_bits (let ZSTD choose optimal)
            HDD_BOTTOMMOST_ZSTD_LEVEL, // level
            0,                         // strategy (default)
            64 * 1024,                 // dict_bytes (64 KB dictionary)
            true,                      // enabled
        );

        // Train ZSTD dictionaries on 8 MB of sample data (~125x dictionary size)
        opts.set_bottommost_zstd_max_train_bytes(8 * 1024 * 1024, true);

        // Block-based table options for better caching
        use rocksdb::BlockBasedOptions;
        let mut block_opts = BlockBasedOptions::default();

        // Partitioned Bloom filters (18 bits per key for better false-positive rate)
        block_opts.set_bloom_filter(18.0, false); // 18 bits per key
        block_opts.set_partition_filters(true); // Partition for large databases
        block_opts.set_format_version(5); // Latest format with optimizations
        block_opts.set_index_type(rocksdb::BlockBasedIndexType::TwoLevelIndexSearch);

        // Cache index and filter blocks in block cache for faster queries
        block_opts.set_cache_index_and_filter_blocks(true);
        block_opts.set_pin_l0_filter_and_index_blocks_in_cache(true);

        // Block cache: the process-wide shared cache when one was provided, otherwise a local
        // 256MB one (safe for low-RAM systems). Sized via --ram-scale / --rocksdb-cache-size.
        match resources {
            Some(resources) => block_opts.set_block_cache(&resources.block_cache),
            None => block_opts.set_block_cache(&Cache::new_lru_cache(256 * 1024 * 1024)),
        }

        // Set block size (256KB - better for sequential HDD reads)
        block_opts.set_block_size(256 * 1024);

        opts.set_block_based_table_factory(&block_opts);

        // Rate limiting: prevent I/O spikes on HDD. Autotuned adapts when the disk is shared vs
        // dedicated, while still honouring the configured ceiling (`--rocksdb-rate-limit-mb`).
        let rate_limit =
            resources.and_then(|resources| resources.rate_limit_bytes_per_sec).unwrap_or(DEFAULT_HDD_RATE_LIMIT_BYTES_PER_SEC);
        opts.set_auto_tuned_ratelimiter(rate_limit as i64, 100_000, 10);

        // Enable BlobDB for large values (reduces write amplification)
        if !blob_files_disabled() {
            opts.set_enable_blob_files(true);
            opts.set_min_blob_size(BLOB_MIN_VALUE_BYTES);
            opts.set_blob_file_size(256 * 1024 * 1024); // 256MB blob files
            opts.set_blob_compression_type(DBCompressionType::Zstd); // Compress blobs
            opts.set_enable_blob_gc(true); // Enable garbage collection
            opts.set_blob_gc_age_cutoff(0.25); // oldest 25% of blob files are relocation-eligible
            opts.set_blob_gc_force_threshold(0.1); // Force GC at 10% garbage
            opts.set_blob_compaction_readahead_size(8 * 1024 * 1024); // 8 MB blob readahead
        }

        if let Some(resources) = resources {
            opts.set_blob_cache(&resources.blob_cache);
            opts.set_write_buffer_manager(&resources.write_buffer_manager);
        }
    }

    /// HDD + queue-depth-1 constraints (USB BOT / no NCQ). See `docs/storage-performance.md`.
    fn apply_hdd_qd1(&self, opts: &mut Options, resources: Option<&RocksDbResources>) {
        // Start from the HDD preset (single background parallelism), then constrain concurrency.
        self.apply_hdd(opts, 1, resources);

        // One flush + one compaction. `increase_parallelism(num_cpus)` would queue N background jobs
        // against a device that can serve one command at a time, starving foreground reads.
        opts.set_max_background_jobs(2);
        opts.set_max_subcompactions(1);

        // Fewer, larger reads: at ~8 ms per seek, a 16 MB sequential read costs about what two random
        // 4 KiB reads do.
        opts.set_compaction_readahead_size(16 * 1024 * 1024);
        opts.set_blob_compaction_readahead_size(16 * 1024 * 1024);
        opts.set_blob_file_size(512 * 1024 * 1024);

        // Keep every SST open: each reopen is another command round-trip. Index/filter memory stays
        // bounded because the preset charges it to the shared block cache.
        opts.set_max_open_files(-1);

        opts.set_optimize_filters_for_hits(true);
        opts.set_avoid_unnecessary_blocking_io(true);

        // Sync incrementally instead of letting dirty pages accumulate into one burst that would hold
        // the single command slot for hundreds of milliseconds.
        opts.set_bytes_per_sync(1024 * 1024);
        opts.set_wal_bytes_per_sync(1024 * 1024);
    }

    /// Get a human-readable description of the preset
    pub fn description(&self) -> &'static str {
        match self {
            Self::Default => "Default preset - balanced for SSD/NVMe (64MB write buffer, BlobDB for large values)",
            Self::Hdd => {
                "HDD preset - optimized for hard disk drives (256MB write buffer, BlobDB, aggressive compression, autotuned rate limiting)"
            }
            Self::HddQd1 => {
                "HDD queue-depth-1 preset - for USB BOT / no-NCQ disks (single compaction thread, large readahead, all files open)"
            }
        }
    }

    /// Get the recommended use case for this preset
    pub fn use_case(&self) -> &'static str {
        match self {
            Self::Default => "General purpose nodes on SSD/NVMe storage",
            Self::Hdd => "Archival nodes on HDD storage (--archival flag recommended)",
            Self::HddQd1 => "Nodes whose datadir sits behind USB Bulk-Only Transport or another QD1 link",
        }
    }

    /// Get memory requirements for this preset
    pub fn memory_requirements(&self) -> &'static str {
        match self {
            Self::Default => "~4GB minimum, scales with --ram-scale",
            Self::Hdd | Self::HddQd1 => {
                "~4GB minimum (256MB write buffer + 256MB cache + overhead), 8GB+ recommended for public RPC"
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preset_from_str() {
        assert_eq!(RocksDbPreset::from_str("default").unwrap(), RocksDbPreset::Default);
        assert_eq!(RocksDbPreset::from_str("Default").unwrap(), RocksDbPreset::Default);
        assert_eq!(RocksDbPreset::from_str("hdd").unwrap(), RocksDbPreset::Hdd);
        assert_eq!(RocksDbPreset::from_str("HDD").unwrap(), RocksDbPreset::Hdd);
        assert_eq!(RocksDbPreset::from_str("hdd-qd1").unwrap(), RocksDbPreset::HddQd1);
        assert!(RocksDbPreset::from_str("unknown").is_err());
    }

    /// Both presets must be applicable with and without shared resources (tests and tooling open
    /// connections without them).
    #[test]
    fn test_presets_apply_with_and_without_resources() {
        let resources = RocksDbResources::new(16 * 1024 * 1024, 16 * 1024 * 1024, Some(1024 * 1024));
        for preset in [RocksDbPreset::Default, RocksDbPreset::Hdd, RocksDbPreset::HddQd1] {
            let mut opts = Options::default();
            preset.apply_to_options(&mut opts, 2, 64 * 1024 * 1024, None);
            let mut opts = Options::default();
            preset.apply_to_options(&mut opts, 2, 64 * 1024 * 1024, Some(&resources));
        }
    }
}
