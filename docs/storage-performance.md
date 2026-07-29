# Storage and Memory Performance

This document records where a Keryx node's disk, RAM and CPU actually go, what was changed to reduce
that footprint, and the measurements behind both. It exists so the next person tuning this does not
have to re-derive the analysis — and so the numbers can be challenged with a reproducible benchmark
rather than argued from intuition.

Everything described here is **node-local policy**. None of it touches consensus, the wire format,
or any DAA-gated constant: a patched and an unpatched node accept exactly the same blocks. The one
change that would require network coordination is called out explicitly at the end.

## 1. Where the resources go

### Disk: the PoM proof dominates everything else

Each post-H4 block carries a `PomProof` recording the full `K = 256`-step possession walk, every step
with its Merkle path under the tier root — about **228 KB per block** (see the rationale on
`POM_PROOF_RETENTION_DEPTH` in `consensus/core/src/config/params.rs`).

At 10 BPS that is:

| | |
|---|---|
| Logical proof writes | ~2.3 MB/s → **~197 GB/day** |
| Live proof set (retention 25 000) | ~6 GB, in continuous churn (written and deleted at the same rate) |

Before the change every one of those 228 KB values lived **inline in the LSM**, so each was rewritten
by every compaction that touched its SST. Two consequences, both measured in §3:

- write amplification multiplied the 2.3 MB/s into tens of MB/s of physical disk traffic;
- the hot, small consensus records (ghostdag, reachability, headers, UTXO entries) shared their SSTs
  and their block cache with proof bytes, so ordinary lookups paid for the churn.

### RAM: three budgets that were not budgets

1. **The PoM proof cache was sized by item count.** `pom_proof_store` used the shared header-data
   policy — `Count(10_000)`. At ~228 KB per proof that reaches **~2.3 GB**, and because item counts
   are not scaled by `--ram-scale`, the cache ignored that flag entirely.
2. **Cache and memtable budgets were per-connection.** The node opens at least four RocksDB instances
   (meta, active consensus, staging consensus, utxoindex) and each `build()` allocated its own cache.
   `--rocksdb-cache-size=2048` therefore allocated **8 GB**, and the HDD preset's 256 MB write buffer
   with `max_write_buffer_number` left at 6 could reach **1.5 GB of memtables per database**.
3. **Index and filter blocks lived outside every budget.** The default preset configured no block
   cache and no `cache_index_and_filter_blocks`, so the index/filter memory of up to `fd_budget / 2`
   open SSTs per consensus DB grew with the database and was accounted nowhere.

### CPU: the threads exist, the work is serial

The rayon pools already default to one thread per core (`block_processors_num_threads: 0` means
"rayon default"). Low CPU utilisation is not a pool-sizing problem; it is a lack of parallel width
plus cores parked on I/O:

- the virtual processor is single-consumer by construction — one virtual state at a time, with
  `par_iter` only *within* a block, over transactions;
- `commit_header` serialises reachability staging and holds relations/statuses write locks across the
  `db.write(batch)` call, so on a slow disk every core waits on one commit;
- `apply_balance_diff`, `apply_age_diff` and `windowed_production_prefix` do random point reads inside
  the virtual-commit critical path (see §5 — these are **not** fixed).

## 2. What changed

| Area | Change | File |
|---|---|---|
| Disk | BlobDB key-value separation in the `default` preset, `min_blob_size = 4 KiB` — proofs leave the LSM, hot records stay inline | `database/src/db/rocksdb_preset.rs` |
| RAM | `RocksDbResources`: one block cache + one `WriteBufferManager` shared by every DB the process opens | `database/src/db/rocksdb_preset.rs`, `keryxd/src/daemon.rs` |
| RAM | `pom_proof_store` switched from `Count(10_000)` to a byte-tracked 64 MB budget that honours `--ram-scale` | `consensus/src/consensus/storage.rs` |
| RAM | `cache_index_and_filter_blocks` + `pin_l0_filter_and_index_blocks_in_cache` when a shared cache exists | `database/src/db/rocksdb_preset.rs` |
| RAM | Explicit `max_write_buffer_number` (4 default / 3 HDD) instead of the 6 left by `optimize_level_style_compaction` | `database/src/db/rocksdb_preset.rs` |
| CPU | HDD bottommost ZSTD level 22 → 6, applied via `set_bottommost_compression_options` so it cannot leak into the LZ4 levels | `database/src/db/rocksdb_preset.rs` |
| I/O | HDD `level_zero_file_num_compaction_trigger` 1 → 4 | `database/src/db/rocksdb_preset.rs` |
| I/O | HDD background write rate limit 12 → 48 MB/s (autotuned), tunable via `--rocksdb-rate-limit-mb` | `database/src/db/rocksdb_preset.rs`, `keryxd/src/args.rs` |
| I/O | Dedicated blob cache (¼ block cache) so proofs do not evict hot SST blocks | `database/src/db/rocksdb_preset.rs` |
| I/O | Optional `--rocksdb-preset=hdd-qd1` for USB BOT / no-NCQ | `database/src/db/rocksdb_preset.rs` |
| RAM | `address_balance` / `age_buckets` byte-budget caches (128 MB × `--ram-scale`) + MultiGet RMW | `consensus/src/consensus/storage.rs`, `database/src/access.rs` |
| RAM | SeekForPrev result cache on `windowed_production_prefix` (64 MB × `--ram-scale`) | `consensus/src/model/stores/windowed_production_prefix.rs` |
| CPU | Parallel Merkle path checks in `verify_pom_proof` / `verify_pom_proof_v2` (rayon, native) | `consensus/core/src/pom.rs` |

The rate limiter deserves a note: at 12 MB/s it sat *below* the sustained physical write rate of a
10-BPS node, so compaction debt accumulated until RocksDB stalled writes. §3 shows this directly —
`legacy-hdd` produced identical numbers over USB and over SATA, because the limiter, not the disk,
was the bottleneck in both.

## 3. Benchmark

`database/examples/rocksdb_workload_bench.rs` replays the node's dominant write pattern: one 228 KB
proof per block, the small hot consensus records alongside it, point reads of recent hot keys, and
the proof-GC delete of the block leaving the retention window — all in a single `WriteBatch`, as
`commit_header` and the virtual processor do.

```bash
cargo run --release -p keryx-database --example rocksdb_workload_bench -- \
    <legacy-default|opt-default|legacy-hdd|opt-hdd|hdd-qd1|opt-hdd-periodic> <db_dir> [blocks] [retention]
```

The `legacy-*` variants carry the pre-change option sets copied verbatim from the v1.4.0 source; the
`opt-*` variants call the real production code path (`RocksDbPreset::apply_to_options`), so the
comparison is against shipped behaviour, not a reconstruction.

The bench forces a full compaction at the end and reports the footprint both as the run left it and
after that compaction. This matters: a configuration that defers compaction shows a flattering
amplification and an inflated footprint, and the two are not comparable until the deferred work is
paid.

### Results — HDD (WD Red WD30EFZX, SATA), 12 000 blocks, 2.7 GB logical

| variant | write amplification | physical written | time to steady state | final on disk |
|---|---:|---:|---:|---:|
| `legacy-default` | 3.31x | 8 949 MiB | 197 s | 894 MiB |
| `opt-default` | 2.85x | 7 712 MiB | 169 s | 1 000 MiB |
| `legacy-hdd` | 1.59x | 4 291 MiB | 415 s | 893 MiB |
| **`opt-hdd`** | **1.54x** | 4 155 MiB | **124 s** | 893 MiB |
| `hdd-qd1` | 1.35x | **3 643 MiB** | 147 s | 892 MiB |

**3.4x faster to the same steady state, at the same final footprint.** The mechanism is visible in
the file breakdown: the LSM shrank from 894 MiB of SST to 1.7 MiB, with 891 MiB in blob files. That
is the set compaction sweeps repeatedly and that competes with hot keys for cache.

Two results worth keeping:

- **`opt-hdd` peaks at ~2x the live set before compaction reclaims it** (1 751 MiB as-left → 893 MiB
  after a forced compaction). This is deferred blob garbage, not a footprint regression: with ~99.8%
  of bytes in blobs the LSM holds almost no SST, RocksDB's compaction triggers are SST-size based, and
  blob GC only runs inside an SST compaction. The effect is **exaggerated by the synthetic workload** —
  a real node holds gigabytes of small-value SST (UTXO set, headers, ghostdag, reachability) whose
  ordinary churn keeps compactions firing. `opt-hdd-periodic` confirms periodic compaction eliminates
  the peak, at **+48% physical writes**; that trade was judged not worth making by default.
- **The worse the transport, the more the optimisation pays.** Over a USB Bulk-Only-Transport bridge
  `legacy-default` reached 6.86x amplification and the optimisation cut 48% of wall time; over SATA the
  same optimisation cuts 14%. `legacy-hdd` scored identically on both, because its 12 MB/s limiter
  bounded it below either transport.

### Results — SSD (NVMe), same workload

| variant | write amplification | physical written | SST | blob |
|---|---:|---:|---:|---:|
| `legacy-default` | 3.81x | 10 291 MiB | 1 021 MiB | 0 |
| `opt-default` | 2.84x | 7 673 MiB | 33 MiB | 967 MiB |

Compaction read traffic dropped 47% (9 272 → 4 958 MiB).

## 4. Hardware reference points

Measured with an unbuffered tester (`FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH`, 4096-aligned,
single-threaded — i.e. queue depth 1), 2 GB file:

| | WD Red WD30EFZX (SATA) | WD Red (USB BOT) | Seagate ST2000DM008 (USB BOT) |
|---|---:|---:|---:|
| Sequential read | 112.9 MB/s | 115.1 MB/s | 85.1 MB/s |
| Sequential write | 96.0 MB/s | 113.4 MB/s | 62.9 MB/s |
| Random 4K read | 112 IOPS / 8.9 ms | 131 IOPS / 7.6 ms | 72 IOPS / 13.9 ms |
| Random 4K write | 553 IOPS / 1.8 ms † | 493 IOPS / 2.0 ms † | 75 IOPS / 13.4 ms (p95 76 ms) |

† Not mechanical. Sub-2 ms on a 5400-rpm drive is impossible; these writes are absorbed by the drive's
DRAM cache because the write-through flag is not honoured end to end. Do not plan against them.

Conclusions that generalise:

- **~110 random IOPS is the budget on a spinning disk**, and it is a mechanical limit (seek +
  rotational latency), not a transport one. At queue depth 1, USB BOT measured *the same* as native
  SATA — BOT only costs when I/O is concurrent, which is exactly what RocksDB background compaction
  plus foreground reads produce.
- Numbers above are short-stroked over a 2 GB file; a real multi-hundred-GB datadir seeks farther and
  will be slower.
- **SMR drives are disqualified.** The ST2000DM008 is shingled; its 76 ms p95 random write is the
  signature of the persistent cache zone filling. RocksDB compaction is precisely the sustained
  random-write workload SMR handles worst.
- **USB enclosures are a reliability risk, not just a performance one.** During sustained write load
  the drive dropped off the bus entirely (`disk` event 51 ×7, `Ntfs` event 140 — *"failed to flush data
  to the transaction log. Corruption may occur"*) and did not return without a physical power cycle. A
  node whose datadir vanishes mid-write is a corruption scenario no tuning addresses. Prefer direct
  SATA; if USB is unavoidable, use a single-bay enclosure with its own supply and no hub.

## 5. Operator guidance

For a spinning disk, the preset choice matters more than anything else — `legacy-default` on HDD wrote
**18.5 GB for 2.7 GB of data** over USB. Always set the preset:

```
keryxd --rocksdb-preset=hdd \
       --rocksdb-cache-size=8192 \
       --ram-scale=2.0 \
       --rocksdb-wal-dir=/path/on/ssd
```

- `--rocksdb-cache-size` is now a **process-wide** budget. Before the change it was allocated per
  database, so this flag delivered roughly 4x what was asked for; the value can now be trusted.
  With ~110 IOPS on the disk, every cache hit saves ~9 ms — this is the single largest lever on
  spinning storage, and RAM is the cheapest way to buy it.
- `--rocksdb-wal-dir` on an SSD removes all log writes from the spinning disk's queue.
- `--rocksdb-rate-limit-mb` (default 48) throttles background writes. Lower it if the disk is shared;
  raise it if compaction cannot keep up and writes stall.

## 6. Remaining / follow-ups

**Read path in the virtual-commit critical section — mitigated (not eliminated).**
`address_balance` / `age_buckets` now use a 128 MB byte-budget cache (scaled by `--ram-scale`)
instead of `Count(10_000)`, and RMW batches go through RocksDB `MultiGet`. The production-prefix
store caches `SeekForPrev` results within a block and invalidates on `extend`/collapse/`clear`.
Hit/miss rates are logged every 10s by the consensus monitor
(`Virtual-index cache: balance …; age …; production-prefix …`). Measure on a real HDD IBD before
claiming the ~22 BPS ceiling is gone — cold-start and large UTXO churn can still miss.

**Parallel PoM Merkle verification — done.** `verify_pom_proof` / `verify_pom_proof_v2` run path
checks via rayon on native targets (walk transitions stay sequential). IBD still skips PoM verify.

**Queue-depth-1 tuning — shipped as `--rocksdb-preset=hdd-qd1`.** Prefer plain `hdd` on SATA/NCQ;
use `hdd-qd1` only for USB BOT / no-NCQ.

**Blob cache + autotuned rate limiter — done** in both Default and HDD presets (blob cache = ¼ of
the process-wide block cache, min 64 MB).

**Lowering `POM_PROOF_RETENTION_DEPTH`.** Dropping it from 25 000 to ~5 000 would cut the live proof
store from ~6 GB to ~1.2 GB and the churn proportionally. This is **not** a consensus rule, but it
**cannot be changed unilaterally**: a peer decides whether to skip proof verification using its own
copy of the constant (`protocol/flows/src/ibd/flow.rs`, `protocol/flows/src/v8/request_block_bodies.rs`),
so a node that GCs earlier and serves a proofless block inside the window its peer still expects gets
rejected with *"PoM possession proof missing"*. It requires a coordinated network release.

**Reducing `POM_WALK_STEPS` / `POM_OPENINGS`.** These are the root cause of the 228 KB proof size, but
changing them is a consensus fork requiring miner lockstep. Recorded as a candidate for a future
H-fork, not a tuning option.
