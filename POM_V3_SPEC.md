# PoM v3 (H6) — Bit-Exact Specification of the Matrix-State Walk

Status: FROZEN for the H6 fork. Any deviation is a consensus fork.
Reference implementation: `consensus/core/src/pom_v3.rs` (the executable spec — on any
discrepancy between this document and that module's behavior, the module wins and this
document must be fixed).
Audience: the official CUDA miner port, third-party miner forks (byte-exact required,
including the Vulkan fork), and independent verifier implementations.
Companion: `POM_CONSENSUS_SPEC.md` (v1/v2 eras, container mechanics, R_T construction).

## 1. Overview

At and after `pom_v3_activation` (per-block gate, keyed on the block's own `daa_score`),
the proof-of-work IS int8 matrix arithmetic over the committed model weights: each nonce
walks K = 256 steps, each step multiplying a 256 x 256 int8 state by a data-selected
64 KB weight tile, reduced entrywise by a non-linear position-tweaked byte function.
The block carries a succinct witness (`PomProofV3`); the verifier NEVER re-walks — it
re-derives the offset chain, recomputes S_0, and spot-checks 32 PRF-sampled single
entries against per-step state commitments and the unchanged per-tier root `R_T`.

All integers little-endian unless stated. All hashes blake3-256.

## 2. Constants

| Name | Value | Meaning |
|---|---|---|
| `D` | 256 | state dimension (state = D x D int8, 64 KB, row-major) |
| `K` | 256 | walk steps per nonce |
| `CHECKS` | 32 | spot-checks per proof (survival of fraction-f cheating ~ e^(-32 f)) |
| `CHUNK_BYTES` | 32 | canonical R_T chunk (pinned by `pom-rt-builder`, unchanged) |
| `TILE_BYTES` | 65536 | tile = 2048 consecutive canonical chunks |
| `TILE_CHUNKS` | 2048 | |
| `SNIPPET_BYTES` | 32 | offset-chain snippet = FIRST canonical chunk of a tile |
| `COL_CHUNKS` | 8 | chunks per opened tile column (256 B) |
| `COL_SUBTREE_DEPTH` | 3 | column subtree: 8 leaves, always complete |

Domain salts, derived as `first 8 bytes of sha256(label), read little-endian u64`
(same convention as the H3/H5 pph salts):

| Salt | Label | Value |
|---|---|---|
| `S0_ROW_SALT` | `keryx-h6-s0-row-salt` | `0x6B61F28F3CC48744` |
| `OFFSET_FIRST_SALT` | `keryx-h6-offset-first-salt` | `0x3F1F886D659E316A` |
| `OFFSET_STEP_SALT` | `keryx-h6-offset-step-salt` | `0xD4C194F3ADB3B1C7` |

Domain prefix (raw ASCII bytes, no length prefix): `CHECKS_DOMAIN = "keryx-h6-checks-v3"`.

## 3. Primitives

### 3.1 `mix64` (unchanged from v1/v2, splitmix64 finalizer)

```
x ^= x >> 30;  x *= 0xBF58476D1CE4E5B9;
x ^= x >> 27;  x *= 0x94D049BB133111EB;
x ^= x >> 31;  return x;             // all u64, wrapping
```

### 3.2 `rho8(acc: i32, tweak: u32) -> u8` — the byte reduction (DECIDED: 5-op form)

```
z = (acc as u32) ^ tweak;            // reinterpret i32 -> u32 (two's complement)
z *= 0x9E3779B9;   z ^= z >> 16;     // wrapping u32 mul
z *= 0x85EBCA6B;   z ^= z >> 13;
return z & 0xFF;
```

Decision record (2026-08-07): the cheaper 2-op variant (single multiply-shift) is
REJECTED. The 5-op murmur-style finalizer is what `matbench_v3.cu` measured — walk cost
-11 to -16 % vs the v2 form at a 2.5x resident/host moat, accepted 2026-08-06 — and its
avalanche quality is what the A4 (backward repair needs non-linearity over Z/256) and
A10 (no cheap fixed points / absorbing states) defenses rest on. rho cost applies to
attackers exactly as to honest miners, so weakening it buys throughput for both and
security for neither. The verifier only evaluates rho 32 times per block — irrelevant.

### 3.3 `rho_tweak(step t: u32, row x: u32, col j: u32) -> u32`

```
t * 0x9E3779B9  +  x * 0xC2B2AE35  +  j * 0x85EBCA6B     // wrapping u32
```

`t` is 1-based (1..=K). Every entry of every step gets a distinct tweak stream; the
all-zero state is not self-consistent anywhere (A10).

### 3.4 `dot_i8(row: &[u8; D], col: &[u8; D]) -> i32`

Bytes are reinterpreted as int8 (two's complement). `acc = sum(row[k] * col[k])`,
exact int32 (|acc| <= 256 * 128 * 128 < 2^22 — no overflow, no saturation).

### 3.5 `snippet_fold(snippet: &[u8; 32]) -> u64`

`sf = 0; for w in 0..8 { sf = mix64(sf ^ (u32_le(snippet[4w..4w+4]) as u64)) }`.

### 3.6 Merkle conventions

- Leaf hash = `blake3(payload)`. Node = `blake3(left || right)` (64-byte concat).
- `verify_merkle(leaf, index, path, root)`: fold bottom-up; at each level the sibling
  goes right if `index` bit is 0, left if 1; `index >>= 1`. Paths longer than 64 are
  rejected outright.
- State trees (D = 256 rows): COMPLETE, depth 8, no duplicate-last.
- `R_T` (the existing per-tier tree from `pom-rt-builder`, UNCHANGED): folds with
  duplicate-last at EVERY level — an odd node hashes with itself. Inclusion paths near
  the ragged right edge therefore contain self-siblings; the verifier is agnostic (it
  just folds whatever siblings the path supplies).
- Column subtree (8 chunk leaves, depth 3): always complete. Tiles are 2048 chunks and
  columns are 8-chunk aligned, so a column NEVER straddles the duplicate-last edge:
  its subtree index under R_T is exact.

## 4. The walk (per nonce)

Inputs: `seed` (the era-salted block seed: `pom_block_seed_h5_2(pre_pow_hash words,
timestamp, nonce)` — H5.2 form, unchanged by H6), the tier blob (canonical 32 B chunk
stream of `pom-rt-builder`, `n_chunks` total).

1. `n_tiles = floor(n_chunks / 2048)`. The sub-tile remainder is NEVER walked; it stays
   committed under R_T and needs no special handling. `n_tiles == 0` is invalid
   (`BlobTooSmall`) — every real tier is far larger than one tile.
2. `S_0`: for each row `r` in 0..D: `h = mix64(seed ^ (S0_ROW_SALT + r))`; then 64
   squeezes: `h = mix64(h)`, append `(h as u32)` little-endian (4 bytes) — 256 bytes per
   row. (`S0_ROW_SALT + r` is wrapping u64 add; `^` binds after the add.)
3. Offsets (tile indices in 0..n_tiles), derived ONLY from seed + snippet list:
   - `i_1 = mix64(seed ^ OFFSET_FIRST_SALT) % n_tiles`
   - `i_{t+1} = mix64(seed ^ ((t+1) * OFFSET_STEP_SALT) ^ snippet_fold(snippet_t)) % n_tiles`
     for t in 1..=K-1, where `snippet_t` = first 32 bytes of the tile read at step t and
     `(t+1) * OFFSET_STEP_SALT` is wrapping u64 mul of the 1-based step successor.
   - The modulo is taken as plain `%` on u64. n_tiles is not a power of two: the bias is
     <= 2^-40 for any real tier size and is accepted (frozen).
4. Transition, for step t in 1..=K over tile `T = blob[i_t * 65536 .. (i_t+1) * 65536]`:
   - column `j` of T = bytes `T[j*256 .. (j+1)*256]` (256 CONSECUTIVE canonical-stream
     bytes = 8 consecutive chunks; the tile is column-major for the walk).
   - `S_t[x][j] = rho8(dot_i8(S_{t-1}.row(x), T.col(j)), rho_tweak(t, x, j))` for all
     (x, j) in D x D.
5. Commitments: `roots[t] = state_root(S_t)` for t in 0..=K (row leaves, depth-8 tree).
6. Lottery: `final_state = fold64(roots[K])` = first 8 bytes of `roots[K]` as u64 LE.
   The winning condition is `era_pow_fold(final_state) <= tier_target`, where
   `era_pow_fold` is the H3-salted header fold (`pom_pow_value_h3`) — UNCHANGED by H6.
   There is NO other v3 lottery function. (`Header.pom_final_state` carries
   `final_state`; header-only PoW, block levels and pruning proofs are era-stable.)

## 5. The witness (`PomProofV3`)

Borsh wire layout, field order (inside the `PomProof` container's trailing `v3` Option):

```
tier:     u8
roots:    Vec<[u8;32]>   // MUST be exactly K+1 = 257 entries, S_0 first
snippets: Vec<[u8;32]>   // MUST be exactly K = 256 entries; snippets[t-1] = first chunk
                         // of the tile read at step t
checks:   Vec<PomV3Opening>  // MUST be exactly CHECKS = 32, in PRF challenge order
```

`PomV3Opening` field order: `row_before: Vec<u8>` (D bytes), `path_before: Vec<[u8;32]>`,
`row_after: Vec<u8>` (D bytes), `path_after: Vec<[u8;32]>`, `tile_col: Vec<u8>` (D bytes),
`col_path: Vec<[u8;32]>`, `snippet_path: Vec<[u8;32]>`.

Container rules (`PomProof` with `v3 = Some(..)`): the legacy fields MUST be canonical
placeholders — `trace_root` all-zero, `initial_trace_path`/`final_trace_path`/`openings`
empty, `steps_v2 = None`; `tier` mirrors `v3.tier`; `final_state = fold64(roots[K])`;
`pow_value = era_pow_fold(final_state)`. Era-exact wire: a proof with `v3` borsh-encodes
the full struct; without, it re-encodes through `PomProofPreV3` (see `pom.rs`).

Approximate size: 257*32 + 256*32 + 32 openings * ~3.3 KB ~ 108 KB.

## 6. The spot-check PRF

```
prf_seed = blake3( "keryx-h6-checks-v3" || pre_pow_hash || nonce_le64
                   || blake3(roots[0] || .. || roots[K])
                   || blake3(snippets[0] || .. || snippets[K-1]) )
for i in 0..32:
    d   = blake3( prf_seed || i_le64 )          // 40-byte input
    t_i = 1 + (u64_le(d[0..8])   % K)           // 1..=K   (bias-free, K power of two)
    x_i =      u64_le(d[8..16])  % D            // 0..D-1
    j_i =      u64_le(d[16..24]) % D            // 0..D-1
```

The PRF binds the header AND the entire deterministic commitment: re-rolling any
committed byte re-rolls `roots[K]`, i.e. the lottery itself (A5 — no prover freedom).

## 7. Verification (exact order, fail-fast)

Given (`pre_pow_hash`, `nonce`, `seed`, container, tier `R_T` + `n_chunks`, target):

1. `v3` present, else `MissingV3`. Legacy placeholders canonical, else
   `NonCanonicalLegacyFields`. `container.tier == v3.tier`, else `TierMismatch`.
2. Shapes: 257 roots, 256 snippets, 32 checks (`WrongShape`); per-opening byte lengths
   (`WrongOpeningShape`, checked per opening in loop order).
3. `n_tiles > 0`, else `BlobTooSmall`.
4. Recompute `S_0` from the seed; its root MUST equal `roots[0]` (`BadInitialRoot`).
5. Derive ALL offsets `i_1..i_K` from (seed, snippet list) — NEVER from opened data.
6. Derive the 32 check points (§6). For each, in order:
   a. `row_before` against `roots[t-1]` at index x (`BadRowBeforePath`);
   b. `row_after` against `roots[t]` at index x (`BadRowAfterPath`);
   c. `tile_col` folded to its depth-3 subtree root, verified against `R_T` at subtree
      index `i_t * 256 + j` (`BadColumnPath`);
   d. `snippets[t-1]` as a single leaf against `R_T` at leaf index `i_t * 2048`
      (`BadSnippetPath`) — the offset-chain anchor (A9);
   e. `rho8(dot_i8(row_before, tile_col), rho_tweak(t, x, j)) == row_after[j]`
      (`TransitionMismatch`).
7. `container.final_state == fold64(roots[K])` (`FinalStateMismatch`);
   `container.pow_value == era_pow_fold(final_state)` (`PowValueMismatch`);
   `pow_value <= tier_target` little-endian (`TargetNotMet`).

Verifier cost: K+2 mix64 chains, one S_0 derivation + tree (~256 row hashes), and per
check ~2 row paths + 1 range proof + 1 leaf path + D MACs — microseconds per block.

## 8. Miner-side requirements (normative)

- The GPU kernel MUST reproduce §3-§4 byte-for-byte (int8 two's complement, exact i32
  accumulation, wrapping u32/u64 arithmetic, little-endian folds). dp4a implementations
  must take care with the i8 reinterpretation of the canonical bytes.
- Proof construction runs ONLY for the winning nonce: re-walk it, build the K+1 state
  trees, list the K snippets, derive the checks (§6), open them (§5).
- Era switch: generate a v3 proof IF AND ONLY IF the TEMPLATE's `daa_score` is at/after
  `pom_v3_activation`. Never key on wall clock or on the tip: a pre-gate template solved
  late still carries a PreV3 proof, a post-gate template always carries v3.
- The walk cannot start before the tier blob is resident and R_T-verified (unchanged
  from v1/v2; possession is the point).

## 9. Frozen decisions log

| Date | Decision |
|---|---|
| 2026-08-06 | Offsets from (seed, step, snippet list), never from state (A3/A9). |
| 2026-08-06 | rho non-linear + positional tweak (A4/A10), cost measured -1.2 %. |
| 2026-08-06 | Verifier never re-walks; 32 spot-checks; e^(-mf) soundness model. |
| 2026-08-06 | R_T, chunking and header layout unchanged; `final_state = fold64(roots[K])`. |
| 2026-08-07 | rho = 5-op finalizer (this doc §3.2); 2-op variant rejected. |
| 2026-08-07 | Single lottery: `era_pow_fold(fold64(roots[K]))`; the vestigial standalone v3 pow fold was removed from the reference. |
| 2026-08-07 | Tile remainder: floor division, remainder never walked (§4.1). |
| 2026-08-07 | R_T range proofs inherit duplicate-last folding; column subtrees always complete and aligned (§3.6). |
