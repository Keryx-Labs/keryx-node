# PoM v3 — final_state is not bound to the walk (nonce-replay / work inflation)

**Severity:** High — mineable work/reward inflation; difficulty cannot throttle the attacker.
**Component:** `consensus/core/src/pom_v3.rs` (`verify_pom_proof_v3`), reached via `check_pom_proof`.
**Found:** 2026-08-20, on live mainnet, by the krx.suprnova.cc pool operator.
**Regression test:** `pom_v3.rs::tests::unsampled_state_root_is_not_bound` (added in this PR — currently passes, i.e. reproduces the flaw).

---

## Summary

`verify_pom_proof_v3` spot-checks only `POM_V3_CHECKS = 32` single entries out of the `POM_V3_K = 256`
**prover-committed** per-step state roots. `roots[0]` is pinned to `S_0(seed)` and `roots[K]` feeds the
lottery (`final_state = fold64(roots[K])`), but the intermediate state chain `roots[1..K-1]` is only
ever touched at the 32 PRF-sampled steps. Any step the challenge does not sample is **never opened**, so
it can hold arbitrary content and the proof still verifies.

Because of this, a miner can do **one** real matrix-walk, obtain one golden `final_state` (with a tiny
pow), and present it under **unlimited distinct nonces**: only `roots[0]` (nonce-bound) and the 32
samples must be made consistent, and the chain between `roots[0]` and the golden `roots[K]` is free to
craft. Since the difficulty hash `pom_pow_value(final_state, pre_pow_hash)` has **no nonce input**
(`consensus/core/src/pom.rs:456`), every replay clears any share/network target and **vardiff cannot
throttle it**.

This is the same class as the v1 possession forgery that PoM v2 closed by *recompute-from-chunks*; the
v3 move to succinct 32/256 spot-checking re-opened it.

---

## Live evidence (mainnet, 2026-08-20)

Captured 40 consecutive **accepted** shares from one wallet (full headers, nonces, `pomProofHex`,
`final_state`, `pow_hex`) and verified everything independently:

1. **Difficulty math is sound and passing.** A from-scratch reimplementation of `pom_pow_value_h3`
   reproduces the node's `pow_hex` **byte-for-byte, 40/40**, and `pow <= target` holds for every share
   (achieved ≈ 2.09×10¹⁰ vs assigned 1e-5, and ~67,000× above a 312,500 vardiff ceiling). The pow gate
   is not the bug.
2. **The pow is nonce-independent.** 40 distinct nonces collapse to **2 distinct pow hashes** (2 = the
   capture spanned 2 block templates; within one template the pow is a single fixed value for every
   nonce).
3. **Full possession verification accepts the replay.** Re-running the node's `Verify` with
   `force_possession = true` **accepts the same `final_state` for 11+ adjacent nonces** (`…c23a, …c63a,
   …ca3a, …ce3a`, stepping by 4), and the re-derived `final_state` equals the claimed one. An honest
   nonce-seeded walk cannot produce the same endpoint for different nonces — the proof is not binding
   the walk to the nonce.

---

## Root cause

`verify_pom_proof_v3` (`consensus/core/src/pom_v3.rs`):

- `roots[0]` is bound to the seed (`v3_state_root(v3_initial_state(seed)) == roots[0]`) — good.
- Offsets are derived from `(seed, snippet list)` — good in isolation.
- **but** the loop runs `POM_V3_CHECKS = 32` times, each opening a single `(t, x, j)` entry: it verifies
  one row of `roots[t-1]` and one row of `roots[t]` plus one transition. Nothing forces the *whole*
  chain `roots[0] → … → roots[K]` to be the deterministic walk from `S_0(seed)`. The prover commits the
  states; only 32 single entries out of 256×256×256 are ever checked, and the endpoint is pinned by a
  64-bit fold (`fold64`).

So the succinct proof certifies "I possess the model and here is *a* valid-looking chain of states,"
not "this specific nonce's walk ends at this `final_state`." The `unsampled_state_root_is_not_bound`
test corrupts a mid-chain `roots[t*]` to garbage (choosing a corruption whose PRF sampling avoids `t*`
and `t*+1`) and shows `verify_pom_proof_v3` still returns `Ok(())`.

---

## Impact

- **Work/reward inflation:** one real walk → unlimited credited shares/blocks. Observed a single wallet
  credited ~10 accepted shares/second, all clearing max difficulty, on our pool.
- **Difficulty is not a control:** since pow ⟂ nonce, network-difficulty / vardiff retargeting rejects
  nothing.
- At consensus level this inflates a miner's effective share of block production and breaks the intended
  PoM soundness model (`H6_soundness_spot_checking` / `POM_V3_SPEC.md` §4 `e^(-m·f)`), which assumes a
  forged step is caught with probability ~`checks/steps`. The observed attack keeps every *sampled*
  entry individually consistent, so raising the check count or anchoring all snippets does **not** close
  it.

---

## Fix direction (for maintainer decision)

The invariant to restore: **the accepted `final_state` must be provably the endpoint of the walk seeded
by this share's nonce.** Spot-checking single entries over prover-committed states cannot enforce that.

- **Recommended — recompute-from-chunks (as PoM v2):** re-derive each state from the running walk
  (`off = state % n_chunks` from the *running* state, never a committed one) and derive `final_state`
  directly, so possession is checked on every step of *this nonce's* walk. This is what closed the v1
  forgery. Cost is the reason v3 went succinct (≈4.3 GMACs/block), so this is a real performance vs.
  soundness call the maintainers own — possibly gated behind a new activation DAA and/or bounded by
  sampling a large *contiguous* segment rather than 32 scattered single entries.
- Whatever the mechanism, it must be **gated at a hardfork activation** — a stricter verifier that
  takes effect un-gated would fork the network.

Pool-side stopgap already deployed on krx.suprnova.cc (reject reused `(wallet, final_state)`) contains
it for our pool only; solo miners and other pools remain exposed until the consensus fix ships.

Happy to run a patched verifier against the captured attack proofs before any fork — contact via the PR.
