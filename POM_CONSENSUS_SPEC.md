# Proof-of-Model (PoM) — Consensus Spec (possession-only, v0)

Branch `PoM` (keryx-node-hardfork). Companion miner branch: `keryx-miner@PoM`.

Goal: make the PoW itself **read the model weights**, so that producing a valid
block requires the full weight blob resident in VRAM, and let a **weightless node**
verify this with a succinct, **100% deterministic** proof (no ε). This proves
**possession** (gate 1), never **correctness** (gate 2 = the ε/determinism wall,
explicitly out of scope — see `project_possession_only_first_step`).

This document is design-only. No consensus code is changed yet.

---

## 0. What the microbench already settled (`pom-microbench/`, 2026-06-19)

- A data-dependent memory-hard walk over the blob **enforces full-VRAM residency**:
  removing 25 % of the blob costs **40–130×** hashrate (monotonic to 162× at 10 %).
- **K (reads/attempt) = pure difficulty/hashrate knob** — security ratio invariant in K.
- **chunk = 32 B = sweet spot** (= one Ampere L2 sector): max honest bandwidth
  (~150 GB/s) AND max attacker penalty (~250×). 32 B is also a clean Merkle-leaf size.
- Caveat: models the PCIe attacker. A 2-GPU **NVLink** pool would reduce the penalty
  (rarer/costlier attack) — to be stated in the threat model, not solved here.

These give us concrete constants: **chunk = 32 B**, **K tuned per tier for difficulty**.

---

## 1. Threat model

- **Proven:** the producer held the full weight blob of the claimed tier in fast
  (VRAM-class) memory at mining time. Partial possession / disk-streaming / model
  substitution is economically dead (microbench) and cryptographically caught (§4).
- **NOT proven:** that the inference *answer* is correct (gate 2). Correctness stays
  optimistic (challenge window + future ZK for disputes, per whitepaper).
- **Residual:** NVLink/NVSwitch pooling could let two GPUs back one tier with a
  smaller penalty than PCIe. Accepted for v0; revisit if it materialises.

---

## 2. Canonical PoM-PoW function

Replaces the matrix/`keryx_hash` step (today: `State::calculate_pow`,
`consensus/pow/src/lib.rs`) for PoM blocks. The blob `W_T` is the GGUF weights of
the tier `T` model, viewed as `N` fixed **32-byte chunks**.

```
seed       = PowHash(pre_pow_hash || time || pad).finalize_with_nonce(nonce)   // unchanged front-end
state[0]   = mix(seed)
for i in 0..K:
    off    = state[i] mod N                       // data-dependent => non-prefetchable
    chunk  = W_T[off]                             // 32 B = 4 u64, read from VRAM
    state[i+1] = transition(state[i], chunk)      // cheap XOR-accumulate + mix64 (see microbench kernel)
pow_value  = kHeavyHash(state[K])                 // reuse existing final hash
accept iff pow_value <= target_T                  // per-tier target (§5)
```

`mix`, `transition`, `kHeavyHash` are byte-exact and shared by miner and verifier.
The miner already does this walk at mining frequency (the microbench kernel is the
prototype). The walk **is** the proof-of-work; possession is a side effect of it
being unaffordable without the resident blob.

---

## 3. The circular-dependency constraint (why the proof is post-PoW)

`pre_pow_hash = hash_override_nonce_time(header, 0, 0)` commits **everything except
nonce and time** — including `hash_merkle_root` (hence the coinbase). The Fiat-Shamir
challenges (§4) derive from `pow_value`, which derives from the winning `nonce`.

⇒ **The proof cannot live in the coinbase / any pre-pow-committed field.** It is
post-PoW witness data, exactly like the nonce. It is therefore carried at the
**`Block` level** (alongside `header` + `transactions`), **excluded from
`pre_pow_hash`**, and validated in the already-existing
`consensus/src/pipeline/header_processor/post_pow_validation.rs`.

No new `Header` field is required: the proof is bound to the block because its
verification re-derives the walk seed from `(pre_pow_hash, nonce)` already in the
header. A swapped proof simply fails verification.

---

## 4. Weightless verification (Fiat-Shamir reveal)

The verifier has **`R_T` only** (the per-tier Merkle root, §6), never the weights.
The producer commits to the full execution and opens a random subset.

**Prover builds `PomProof`:**
1. Run the walk (§2), recording every `state[i]`.
2. `trace_root = MerkleRoot(state[0..=K])`  (commitment to the whole execution).
3. `pow_value = kHeavyHash(state[K])`.
4. `challenges = FS(pre_pow_hash, nonce, trace_root, pow_value)` → `t` step indices in `0..K`.
5. For each challenged step `i`: reveal `{ state[i], state[i+1], off=state[i] mod N,
   chunk = W_T[off], weight_path_i (chunk→R_T), trace_path_i (state[i],state[i+1]→trace_root) }`.

`PomProof = { trace_root, pow_value, openings[t] }`.

**Verifier (`post_pow_validation`, deterministic, no weights):**
- Re-derive `challenges = FS(pre_pow_hash, nonce, trace_root, pow_value)`.
- For each opening: check `off == state[i] mod N`; `weight_path_i` proves `chunk` at
  leaf `off` under `R_T`; `trace_path_i` proves `(state[i], state[i+1])` under
  `trace_root`; and `state[i+1] == transition(state[i], chunk)`.
- Check `pow_value <= target_T` and that `state[K]` (committed as last trace leaf,
  opened once) hashes to `pow_value`.
- Reject on any mismatch (`RuleError::InvalidPoW` or a new `RuleError::InvalidPomProof`).

Every check is byte-level integer comparison ⇒ **consensus-clean, no ε**. This is the
exact contrast with inference verification.

**Soundness (sketch, needs review):** the prover commits `trace_root` before learning
`challenges` (FS) ⇒ cannot adaptively patch openings. To pass with a fabricated step
(no real chunk at `off`) the prover must grind a `nonce`/trace whose `t` challenged
steps all land on the chunks it *does* hold. With held fraction `f`, that is
≈ `f^t` per attempt ⇒ pick `t` so `f^t` is infeasible for the largest `f` a cheating
config could plausibly hold (combined with the microbench penalty on actually
fetching the rest). Start at **t ≈ 32** and calibrate.

**Proof size (32 B chunks, N≈2^28 for a ~8 GiB tier, K≈1024, t=32):**
per opening ≈ 32 (state) + 32 (state') + 32 (chunk) + 28·32 (weight path) + 10·32
(trace path) ≈ 1.3 KiB ⇒ **≈ 42 KiB / block**. Within `max_block_mass = 500_000`.
(Trade-off knobs: smaller `t`, coarser leaves, or a vector commitment with shorter paths.)

---

## 5. Difficulty: GLOBAL, single target — NO per-tier difficulty

Multi-tier is the target (each miner declares + proves the highest model it holds),
but per-tier difficulty is **not** needed. Measured (microbench, 2026-06-19, chunk
32 B, K=1024, resident):

| blob (tier) | honest H/s |
|---|---|
| 2 GiB | 6.30M |
| 8 GiB | 5.38M |
| 20 GiB | 4.03M |

Honest hashrate declines only **~1.5× over a 10× blob-size range** (sub-linear: the
walk is latency-bound on `K` random 32 B reads, not bound by `N`). Therefore:

- A **single global `target`** does not starve high tiers — a 70B miner does at most
  ~1.5× fewer attempts than an 8B miner.
- The **reward-by-tier gradient** (0.5 → 4.0 KRX, ×8) **dwarfs** that 1.5× penalty, so
  the incentive is unambiguous: **declare the highest tier your GPU can hold** (and
  prove it). Self-regulating "1 GPU = 1 tier".
- You cannot declare a tier without proving possession (§4); holding the model needs
  the VRAM (hardware barrier). Each GPU settles at its max tier naturally.

⇒ **No parallel lanes, no per-tier retargeting, no GHOSTDAG interaction.** Difficulty
machinery (`difficulty.rs`, `calc_level_from_pow`) is untouched. The tier is pure
metadata + possession proof + reward bracket — it never enters difficulty.

---

## 6. Tier binding & reward integration (reuse what exists)

- `R_T` (Merkle root over the 32 B chunks of each tier's GGUF) pinned in
  `consensus/core/src/config/params.rs`, next to the existing `*_MODEL_ID` consts.
  One `R_T` per tier; `model_id` already identifies the model, `R_T` adds the
  byte-level commitment used by §4.
- The miner already declares its model/tier (`ai:cap`, model id). The PoM proof binds
  the *actual* blob to `R_T`, so a miner can no longer claim a tier it does not hold.
- Reward-by-tier already exists: `check_ai_request_inference_rewards`
  (`utxo_validation.rs:544`) gated by `model_cap_enforcement_activation`, tables
  `INFERENCE_REWARD_MINIMUMS{,_V2}`. PoM makes the tier claim **un-gameable**, which is
  the whole point of `project_stake_to_mine_phase2`'s reward-by-model.

---

## 7. Activation gating

New fork `pom_activation: ForkActivation` on `Params` (mirror of
`opoi_v2_activation`): mainnet `never()` until H chosen, testnet a throwaway value,
sim/devnet `always()`. PoW salt / IBD divergence rules apply — PoM blocks before H
must validate under the legacy self-verifying PoW; the proof requirement starts at H.
(IBD re-validation is why this MUST be DAA-gated — same lesson as
`project_ibd_utxo_commitment_split` and `project_opoi_v2_gate_sync`.)

---

## 8. Exact integration points (grounded in current code)

| Concern | File / symbol |
|---|---|
| Walk + final hash | `consensus/pow/src/lib.rs` `State::calculate_pow` (+ new PoM path) |
| Memory-hard kernel (prototype) | `pom-microbench/cuda/kernel.cu` `pom_walk` |
| PoW check call site | `header_processor/pre_ghostdag_validation.rs:104` `state.check_pow` |
| Post-PoW proof verify (new) | `header_processor/post_pow_validation.rs` |
| Per-tier target / block level | `consensus/src/processes/difficulty.rs`, `keryx_pow::calc_level_from_pow` |
| `R_T` consts + `pom_activation` | `consensus/core/src/config/params.rs` (near `*_MODEL_ID`, line ~11–92) |
| Reward-by-tier (reuse) | `consensus/src/pipeline/virtual_processor/utxo_validation.rs:544` |
| New block-level `PomProof` | `consensus/core/src/block.rs` (Block struct) + P2P/serde |
| New error variant | `consensus/core/src/errors/...` `InvalidPomProof` |

---

## 9. Open questions (prototype before committing consensus)

1. **`transition` / `mix` final form** — lock the byte-exact functions shared by
   miner kernel + node verifier (the microbench uses XOR-accumulate + splitmix64).
2. **`t` calibration** — soundness `f^t` vs proof size; target ~42 KiB / `t≈32`.
3. **`trace_root` cost** — recording K states + Merkle tree at mining frequency;
   does it dent honest hashrate? (microbench measured the *walk* only, not the commit.)
4. **Kernel ↔ candle aliasing** — the PoW kernel must index the SAME VRAM buffer as the
   quantized candle weights (zero duplicated VRAM). Feasibility unproven.
5. **Per-tier difficulty — RESOLVED, dropped** (§5): global difficulty works, no
   GHOSTDAG interaction. Multi-tier ships from day 1 via per-tier `R_T` + reward bracket.
6. **NVLink pooling** — quantify the penalty drop on a real 2-GPU NVLink rig if available.
7. **Hashrate vs blob size** — measured ~1.5× over 10× size; re-confirm on the actual
   tier blobs (8B/32B/70B GGUF) once `R_T` blobs exist.

---

## 10. Suggested build order

1. Lock `transition`/`mix`/`kHeavyHash` byte-exact (shared crate, miner + node).
2. Microbench extension: add the `trace_root` commit to measure honest-hashrate cost (Q3).
3. `R_T` builder (offline tool: GGUF → 32 B chunks → Merkle root) + pin **all tiers** in params.
4. `PomProof` struct + serde + Block plumbing (no enforcement yet).
5. `post_pow_validation` verifier (multi-tier: select `R_T` by declared tier; `pom_activation` gated, testnet `always`).
6. Miner: emit `PomProof` from the real walk (reuse kernel), tier = highest model it holds.
7. End-to-end on a fresh testnet with ≥2 tiers (e.g. 8B + 32B) — difficulty global, untouched.
```
