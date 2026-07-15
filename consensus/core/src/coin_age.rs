//! Coin-age holder-reward (v3) — core age arithmetic.
//!
//! The ratio-reward's balance numerator is an instantaneous snapshot, which the "rotation"
//! exploit games: mine hard on an address until its production drives the ratio bracket down,
//! then move the whole pot to a FRESH address (production zero ⇒ top bracket again) and repeat.
//! v3 replaces the snapshot with a persistence measure — how much × how long coins actually
//! stayed put — so a pot can no longer certify a new address the moment it lands.
//! See KERYX-KRX/coin_age_holder_reward_spec.md (spec v3). Everything here is pure arithmetic:
//! no stores, no activation checks — callers gate on `coin_age_activation`.
//!
//! Two building blocks:
//! - [`assign_output_effective_daa`]: the FIFO age carry-over rule (spec §3) run per transaction,
//!   deciding each output's `effective_daa` (its age anchor) from the spent inputs.
//! - [`eff_balance_from_buckets`]: the per-coin-capped effective balance (spec §2) from the three
//!   per-SPK aggregates `{B_mat, B_imm, A_imm}`, replacing the raw balance in the ratio bracket.
//!
//! The pair has ONE load-bearing property, and it is the one to check before touching anything
//! here: merging coins must never raise `eff_balance` — not at the merge, and not at any later
//! DAA. A rule that only balances the books at the merge instant leaks through the ramp that
//! follows (see [`assign_output_effective_daa`]). Assert against `eff_balance` over a spread of
//! future DAAs; an intermediate quantity like Σ v·e being conserved proves nothing, because
//! `eff_balance` is concave in age and concavity does not commute with averaging.
//!
//! Determinism: integer arithmetic only, floor division, `u128` intermediates (sompi × DAA
//! products overflow `u64`), and FIFO ordering keyed on `(effective_daa, tx input order)` — both
//! consensus data — so every node derives byte-identical results.

use crate::tx::ScriptPublicKey;
use std::collections::HashMap;

/// A spent input's view for age assignment: its payout SPK, amount (sompi) and age anchor.
/// Order in the slice MUST be the transaction's input order (consensus data, the FIFO tie-break).
pub type AgedInput<'a> = (&'a ScriptPublicKey, u64, u64);

/// Effective-DAA (age anchor) assignment for every output of a transaction — the FIFO carry-over
/// rule of spec §3. Outputs are grouped by SPK; for each output SPK `S`:
///
/// - `S` absent from the inputs (pure recipient / transfer / rotation hop) ⇒ `e_out = current_daa`
///   (age resets to zero — age is not transferable across addresses).
/// - `S` present in the inputs (change / consolidation) ⇒ FIFO: the spent value
///   `spent_v = max(0, in_v − out_v)` consumes the OLDEST inputs first (their age is destroyed);
///   `e_out` is the anchor of the YOUNGEST surviving input value, i.e. the minimum surviving age.
///
/// FIFO is the only choice a holder cannot game in their favor (LIFO would let a churning shell
/// shield an old core). Merging coins of the SAME maturity class is free: if every input is
/// mature, the youngest is mature too, so the merged coin is mature and `eff_balance` is
/// untouched — the consolidation case that matters (spec §4①).
///
/// # Why the youngest, and why nothing weaker is sound
///
/// A merge replaces N coins by ONE, and one coin carries ONE anchor — the age DISTRIBUTION is
/// destroyed. `eff_balance` is a sum of N kinked ramps before the merge and a single kinked ramp
/// after, so no anchor can make the two agree at every future DAA: at best they agree at one
/// instant. Matching at the merge instant (the value-weighted average, capped or not) is exactly
/// what leaks — the merged coin then ramps at its FULL value `V/w` per block instead of only the
/// fresh part ramping, so an old pile drags fresh coins to maturity `V/I` times too fast, and
/// spamming self-consolidation becomes profitable.
///
/// The youngest surviving anchor is the unique maximal safe choice — not a conservative guess.
/// Safety demands `V·g(a+t) ≤ Σ vᵢ·g(aᵢ+t)` for all `t ≥ 0` (`g` = the per-coin ramp of
/// [`eff_balance_from_buckets`]). Evaluate at `t = w − a`: the merged coin is exactly mature, so
/// the left side is `V`; the right side reaches `V` only once EVERY input is mature, which forces
/// `aᵢ ≥ a` for all `i`, i.e. `a ≤ min aᵢ`. Any anchor above the minimum is therefore breakable,
/// always at the same spot — the merged coin maturing before its own components.
///
/// The price is that merging ACROSS maturity classes loses age down to the youngest input (mixing
/// a fresh coin into a mature pile resets the pile). That loss is one-sided by construction: this
/// rule can only ever under-credit, never mint. Wallets should consolidate within a maturity
/// class.
///
/// Returns one `effective_daa` per output, in output order. Every output of the same SPK gets the
/// same anchor. Coinbase transactions have no inputs ⇒ every output anchors at `current_daa`.
///
/// Ties on equal anchors keep the tx input order (stable sort) — canonical on every node.
pub fn assign_output_effective_daa(
    inputs: &[AgedInput<'_>],
    outputs: &[(&ScriptPublicKey, u64)],
    current_daa: u64,
) -> Vec<u64> {
    // Group input (amount, anchor) pairs by SPK, preserving tx input order within each group.
    let mut by_spk: HashMap<&ScriptPublicKey, Vec<(u64, u64)>> = HashMap::new();
    for (spk, amount, anchor) in inputs {
        by_spk.entry(spk).or_default().push((*amount, *anchor));
    }
    // Total output value per SPK (an SPK's outputs all share one anchor, so sum them first).
    let mut out_v: HashMap<&ScriptPublicKey, u64> = HashMap::new();
    for (spk, amount) in outputs {
        *out_v.entry(spk).or_default() += amount;
    }
    // Resolve one anchor per output SPK, then map back onto the output list.
    let mut anchor_by_spk: HashMap<&ScriptPublicKey, u64> = HashMap::new();
    for (&spk, &total_out) in out_v.iter() {
        let anchor = match by_spk.get(spk) {
            None => current_daa, // pure recipient: reset
            Some(group) => fifo_survivor_anchor(group, total_out, current_daa),
        };
        anchor_by_spk.insert(spk, anchor);
    }
    outputs.iter().map(|(spk, _)| anchor_by_spk[spk]).collect()
}

/// FIFO survivor anchor for one SPK group (spec §3 inner rule). `group` is the SPK's spent
/// inputs as (amount, anchor) in tx input order; `out_v` its total output value (> 0).
///
/// The spent value (`in_v − out_v`, when positive) consumes the oldest anchors first; the outputs
/// then take the anchor of the YOUNGEST surviving input — the largest surviving anchor. Any
/// net-received value (`out_v > in_v`) is age zero, so it drags the whole output to `current_daa`.
/// See [`assign_output_effective_daa`] for why the youngest, and why nothing weaker is sound.
/// The ≤ `current_daa` invariant holds since every anchor is a past DAA.
fn fifo_survivor_anchor(group: &[(u64, u64)], out_v: u64, current_daa: u64) -> u64 {
    let in_v: u64 = group.iter().map(|(v, _)| v).sum();
    if in_v == 0 || out_v > in_v {
        // Nothing carries, or net-received value enters at age zero and sets the minimum.
        return current_daa;
    }
    // Oldest first; stable ⇒ equal anchors keep tx input order (canonical tie-break).
    let mut sorted: Vec<(u64, u64)> = group.to_vec();
    sorted.sort_by_key(|&(_, anchor)| anchor);
    // Consume the spent value from the oldest end; a partially spent input keeps its remainder.
    // `out_v > 0` and `in_v ≥ out_v`, so at least one input survives and the loop always assigns.
    let mut spent_v = in_v - out_v;
    let mut youngest = 0u64; // max anchor over surviving value = min surviving age
    for (v, anchor) in sorted {
        let consumed = spent_v.min(v);
        spent_v -= consumed;
        if v > consumed {
            youngest = youngest.max(anchor);
        }
    }
    youngest
}

/// Per-coin-capped effective balance (spec §2) from the three per-SPK bucket aggregates at the
/// evaluation DAA `d`:
///
/// - `b_mat` — Σ amount over coins with age ≥ `w` (mature: each contributes its face value);
/// - `b_imm` / `a_imm` — Σ amount / Σ amount·effective_daa over coins with age < `w` (immature:
///   each contributes `v·age/w`, i.e. the linear ramp).
///
/// `eff_balance = B_mat + (d·B_imm − A_imm) / w`. The cap is PER COIN — an age-0 coin contributes
/// zero regardless of how over-mature the rest of the address is, which is what kills the
/// "maturity battery" (an old holder instantly maturing a fresh pot under an aggregate cap,
/// spec §4④). The immature term is < `b_imm` by construction (each age < `w`); the explicit
/// clamps only guard rounding/inconsistent-input edges. `u128` avoids overflow on `d·B_imm`.
pub fn eff_balance_from_buckets(b_mat: u64, b_imm: u64, a_imm: u128, current_daa: u64, w: u64) -> u64 {
    let ramp_mass = ((current_daa as u128) * (b_imm as u128)).saturating_sub(a_imm);
    let immature = (ramp_mass / (w.max(1) as u128)).min(b_imm as u128) as u64;
    b_mat.saturating_add(immature)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tx::ScriptVec;

    fn spk(byte: u8) -> ScriptPublicKey {
        ScriptPublicKey::new(0, ScriptVec::from_slice(&[byte; 3]))
    }

    const W: u64 = 6_048_000; // 7 days at 10 BPS (COIN_AGE_MATURITY_W)
    const D: u64 = 50_000_000; // evaluation DAA for the tests

    /// Anchor of a coin created `age` DAA before the evaluation point `D`.
    fn aged(age: u64) -> u64 {
        D - age
    }

    /// `eff_balance` of a set of (amount, anchor) coins held by one SPK, evaluated at DAA `d` —
    /// the per-coin-capped sum the buckets aggregate. Used to assert what a merge does to the
    /// only number that matters: the ratio numerator.
    fn eff_at(coins: &[(u64, u64)], d: u64) -> u64 {
        let (mut b_mat, mut b_imm, mut a_imm) = (0u64, 0u64, 0u128);
        for &(v, anchor) in coins {
            if d - anchor >= W {
                b_mat += v;
            } else {
                b_imm += v;
                a_imm += v as u128 * anchor as u128;
            }
        }
        eff_balance_from_buckets(b_mat, b_imm, a_imm, d, W)
    }

    fn eff_of(coins: &[(u64, u64)]) -> u64 {
        eff_at(coins, D)
    }

    /// Merge `coins` (all one SPK) into a single output at `D` and assert the result can never
    /// out-earn the unmerged set — checked at the merge instant AND all the way through maturity,
    /// which is where a rule that only balances at `t = 0` leaks.
    fn assert_merge_never_pays(coins: &[(u64, u64)]) {
        let a = spk(1);
        let total: u64 = coins.iter().map(|(v, _)| v).sum();
        let inputs: Vec<AgedInput<'_>> = coins.iter().map(|&(v, anchor)| (&a, v, anchor)).collect();
        let outputs = [(&a, total)];
        let merged = assign_output_effective_daa(&inputs, &outputs, D)[0];
        for t in [0, 1, W / 100, W / 10, W / 3, W / 2, W - 1, W, W + 1, 2 * W, 10 * W] {
            let (before, after) = (eff_at(coins, D + t), eff_at(&[(total, merged)], D + t));
            assert!(after <= before, "merge pays at t={t}: unmerged {before} -> merged {after}");
        }
    }

    #[test]
    fn consolidating_one_maturity_class_is_free() {
        // Spec §4①, the case that matters: every input is mature, so the youngest is mature too
        // ⇒ the merged coin is mature ⇒ eff_balance is untouched. Consolidating a mature stock
        // costs nothing, which is what keeps the UTXO-set cleanup incentive alive.
        let a = spk(1);
        let coins = [(10u64, aged(3 * W)), (20, aged(2 * W)), (30, aged(W))];
        let inputs: Vec<AgedInput<'_>> = coins.iter().map(|&(v, e)| (&a, v, e)).collect();
        let outputs = [(&a, 60u64)];
        let merged = assign_output_effective_daa(&inputs, &outputs, D)[0];
        assert_eq!(merged, aged(W)); // the youngest survivor, still exactly mature
        assert_eq!(eff_of(&[(60, merged)]), eff_of(&coins)); // 60 either way — free
        assert_merge_never_pays(&coins);
    }

    #[test]
    fn consolidation_takes_the_youngest_survivor() {
        // Same-SPK consolidation spends nothing net ⇒ every input survives ⇒ the output anchors
        // on the youngest of them. No weighted average: averaging is what let an old pile lend
        // its surplus age to fresh coins.
        let a = spk(1);
        let inputs = [(&a, 10u64, aged(W / 2)), (&a, 20, aged(W / 4))];
        let outputs = [(&a, 30u64)];
        assert_eq!(assign_output_effective_daa(&inputs, &outputs, D), vec![aged(W / 4)]);
    }

    #[test]
    fn fifo_spends_oldest_first_on_partial_spend() {
        // Spec §4②: mixed-age stock, partial spend. The oldest input is consumed by the spent
        // value; the change keeps the YOUNGEST anchor — FIFO destroys the most age, never less
        // (a holder cannot shield an old core by churning young coins).
        let a = spk(1);
        let b = spk(2);
        let inputs = [(&a, 100u64, aged(W / 2)), (&a, 100, aged(W / 4))];
        let outputs = [(&b, 100u64), (&a, 100)];
        let anchors = assign_output_effective_daa(&inputs, &outputs, D);
        assert_eq!(anchors[0], D); // recipient B: reset to now
        assert_eq!(anchors[1], aged(W / 4)); // change at A: the young survivor, old age destroyed
    }

    #[test]
    fn partially_spent_oldest_input_keeps_its_remainder() {
        // spent_v smaller than the oldest input: the remainder of that input survives with its
        // own anchor and blends with the younger input.
        let a = spk(1);
        let b = spk(2);
        let inputs = [(&a, 100u64, aged(W / 2)), (&a, 100, aged(W / 4))];
        let outputs = [(&a, 150u64), (&b, 50)];
        let anchors = assign_output_effective_daa(&inputs, &outputs, D);
        // 50 spent off the oldest; survivors are 50 at age W/2 and 100 at age W/4 — the change
        // takes the youngest of them, not their mean.
        assert_eq!(anchors[0], aged(W / 4));
        assert_eq!(anchors[1], D);
    }

    #[test]
    fn pure_recipient_resets_to_now() {
        // Spec §4③ (the rotation hop): an output SPK absent from the inputs anchors at the
        // current DAA — age is not transferable, a fresh address starts at zero.
        let a = spk(1);
        let b = spk(2);
        let inputs = [(&a, 1_000u64, 42u64)];
        let outputs = [(&b, 1_000u64)];
        assert_eq!(assign_output_effective_daa(&inputs, &outputs, D), vec![D]);
    }

    #[test]
    fn net_receive_resets_to_now() {
        // Spec §3 "entrée nette": out_v > in_v for S ⇒ S nets value in, and that value is age
        // zero. Age zero IS the minimum, so the whole output resets — no dilution, a reset.
        let a = spk(1);
        let b = spk(2);
        let inputs = [(&a, 700u64, aged(W / 2)), (&b, 300, aged(W))];
        let outputs = [(&a, 1_000u64)];
        assert_eq!(assign_output_effective_daa(&inputs, &outputs, D), vec![D]);
    }

    #[test]
    fn coinbase_outputs_anchor_at_now() {
        // No inputs (coinbase): every output starts at the current DAA.
        let a = spk(1);
        let outputs = [(&a, 5_000u64)];
        assert_eq!(assign_output_effective_daa(&[], &outputs, D), vec![D]);
    }

    #[test]
    fn same_spk_outputs_share_one_anchor() {
        // All outputs of one SPK get the same anchor (the rule resolves per SPK, not per output).
        let a = spk(1);
        let inputs = [(&a, 100u64, aged(W / 2))];
        let outputs = [(&a, 30u64), (&a, 70)];
        let anchors = assign_output_effective_daa(&inputs, &outputs, D);
        assert_eq!(anchors[0], anchors[1]);
        assert_eq!(anchors[0], aged(W / 2));
    }

    #[test]
    fn anchor_never_exceeds_current_daa() {
        let a = spk(1);
        let inputs = [(&a, u64::MAX / 2, D - 1)];
        let outputs = [(&a, u64::MAX / 2), (&spk(2), 1u64)];
        for anchor in assign_output_effective_daa(&inputs, &outputs, D) {
            assert!(anchor <= D);
        }
    }

    #[test]
    fn mixing_a_fresh_coin_resets_the_whole_output() {
        // The price of the rule, pinned: one age-0 coin merged into an over-mature pile drags the
        // output to age 0, whatever the values. Value-blind and brutal — but forced (see the
        // maximality argument on `assign_output_effective_daa`), and it errs against the holder.
        let a = spk(1);
        let inputs = [(&a, 100u64, aged(3 * W)), (&a, 100, D)];
        let outputs = [(&a, 200u64)];
        assert_eq!(assign_output_effective_daa(&inputs, &outputs, D), vec![D]);
    }

    #[test]
    fn maturity_battery_dies_through_consolidation_both_ways() {
        // Spec §4④, the case `maturity_battery_is_dead` misses: battery and fresh pot CONSOLIDATED
        // instead of left separate. Both size ratios, because they fail differently — and the
        // dangerous one is the big pile / small pot, which the earlier `w`-cap rule got wrong.
        let (m, p) = (1_000_000u64, 40_000_000u64);
        assert_merge_never_pays(&[(m, aged(3 * W)), (p, D)]); // small battery, big pot
        assert_merge_never_pays(&[(p, aged(3 * W)), (m, D)]); // big battery, small pot
    }

    #[test]
    fn merging_never_accelerates_a_fresh_coin() {
        // The leak the `w`-cap rule left open, pinned as a test: a huge over-mature pile plus one
        // fresh coin. Under that rule the merge was neutral at t=0 but the merged coin ramped at
        // its FULL value, so the fresh coin matured ~V/I times too early — merging every block
        // paid. Here the merge can never beat leaving the coin alone, at any t.
        assert_merge_never_pays(&[(1_000_000_000_000u64, aged(3 * W)), (238_464_000, D)]);
    }

    #[test]
    fn consolidation_never_pays_on_a_miner_stock() {
        // The real shape: a long over-mature tail plus a fresh ramp still climbing — the stock
        // that minted +38.28e9 on testnet at daa 8568. Sweeping it all into one coin must never
        // out-earn leaving it alone, checked through maturity and beyond.
        let coins: Vec<(u64, u64)> = (0..40u64)
            .map(|i| (1_000_000 + i * 7, aged(if i < 20 { 3 * W + i * 1_000 } else { (i - 19) * (W / 40) })))
            .collect();
        assert_merge_never_pays(&coins);
    }

    #[test]
    fn maturity_battery_is_dead() {
        // Spec §4④: 1M KRX held ≥ W (mature) + a fresh 40M pot (age 0 ⇒ a_imm = b_imm·d).
        // Under the per-coin cap the fresh pot contributes ZERO — the old holding cannot
        // "pre-charge" maturity for incoming coins (the aggregate-cap exploit).
        // NOTE: this only covers coins left SEPARATE. Merging them is the carry-over rule's job:
        // see `maturity_battery_dies_through_consolidation_both_ways`.
        let mature = 1_000_000u64;
        let fresh = 40_000_000u64;
        assert_eq!(eff_balance_from_buckets(mature, fresh, fresh as u128 * D as u128, D, W), mature);
    }

    #[test]
    fn half_matured_ramp_counts_half() {
        // Spec §4 table: 3_500 held at half maturity ⇒ eff_balance 1_750.
        let v = 3_500u64;
        let anchor = D - W / 2; // created half a maturity period ago
        let a_imm = v as u128 * anchor as u128;
        assert_eq!(eff_balance_from_buckets(0, v, a_imm, D, W), v / 2);
    }

    #[test]
    fn fully_fresh_counts_zero_and_mature_counts_full() {
        let v = 3_500u64;
        assert_eq!(eff_balance_from_buckets(0, v, v as u128 * D as u128, D, W), 0); // age 0 ⇒ nothing
        assert_eq!(eff_balance_from_buckets(v, 0, 0, D, W), v); // mature ⇒ face value
    }

    #[test]
    fn eff_balance_wide_values() {
        // u128 path: near-max supply-scale immature mass at a large DAA must not overflow and
        // stays clamped at b_imm.
        let b_imm = 21_000_000u64 * 100_000_000; // whole-supply-scale sompi
        let eff = eff_balance_from_buckets(0, b_imm, 0, u64::MAX / 4, 1);
        assert_eq!(eff, b_imm); // ramp saturates at face value, never above
    }
}
