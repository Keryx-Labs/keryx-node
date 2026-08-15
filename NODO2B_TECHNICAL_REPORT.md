# Keryx Node — Adaptive IBD NODO-2B

## Base

- Direct fork base: `Keryx-Labs/keryx-node`
- Pinned source commit: `0984f950f39c917ba91b4f668e519d0daa6467a6`
- Keryx node release line: `v1.4.4`
- Production-selected implementation: **NODO-2B**
- NODO-2C intentionally excluded.

## Functional source scope

Relative to the pinned Keryx base, NODO-2B intentionally changes only:

1. `protocol/flows/src/flow_context.rs`
   - shared per-peer IBD cooldown state;
   - cooldown lookup/expiry;
   - rejection of cooling peers at the shared IBD arbitration boundary.

2. `protocol/flows/src/ibd/flow.rs`
   - bounded IBD throughput probe;
   - slow-peer yield below the configured threshold;
   - 15-second shared cooldown;
   - handoff back to IBD arbitration.

`README.md` and this report are documentation-only changes.

All other source files are restored byte-for-byte from the pinned Keryx base.

## Adaptive slow-peer parameters

| Parameter | Value |
|---|---:|
| IBD batch | 99 blocks |
| Probe activation | 792 blocks |
| Probe sample | 198 blocks |
| Slow threshold | 25.0 blocks/s |
| Shared cooldown | 15 s |

## Mainnet evidence — 2026-08-11

Exact peer IP addresses are intentionally omitted.

```text
13:39:32.642  IBD started with <peer-a>
13:39:52.261  slow-peer yield: 15.83 blocks/s
13:39:52.273  IBD started with <peer-b>
13:39:58.979  fast-peer probe PASS: 34.66 blocks/s
13:41:03.475  IBD completed successfully
```

Observed handoff from slow-peer yield to the next IBD acquisition: approximately
12 ms.

## Mining validation

```text
Blocks found:       6
Blocks submitted:   6
Rejected:           0
Hashrate samples:   98
Mean hashrate:      19.274 MH/s
Max hashrate:       20.030 MH/s
```

The node independently logged six submit-block events.

## Live processing during pruning

The node continued accepting relay traffic during pruning. Audit reached:

```text
traversed: 114000
pruned:    104965
```

## Allocator status

NODO-2B makes **no allocator change**.

`utils/alloc/Cargo.toml` remains identical to the pinned Keryx v1.4.4 base, including its existing mimalloc configuration.


## Scope

NODO-2B changes IBD peer arbitration only. It does not intentionally change
consensus rules, block validity, transaction validity, mining validity, or the
normal fast-peer path.
