//! Coin-age holder-reward (v3) — core age arithmetic.
//!
//! The ratio-reward's balance numerator is an instantaneous snapshot, which the "rotation"
//! exploit games: mine hard on an address until its production drives the ratio bracket down,
//! then move the whole pot to a FRESH address (production zero ⇒ top bracket again) and repeat.
//! v3 replaces the snapshot with a persistence measure — how much × how long coins actually
//! stayed put — so a pot can no longer certify a new address the moment it lands.
//! See KERYX-KRX/coin_age_holder_reward_spec.md (spec v2). Everything here is pure arithmetic:
//! no stores, no activation checks — callers gate on `coin_age_activation`.
//!
//! Two building blocks:
//! - [`assign_output_effective_daa`]: the FIFO age carry-over rule (spec §3) run per transaction,
//!   deciding each output's `effective_daa` (its age anchor) from the spent inputs.
//! - [`eff_balance_from_buckets`]: the per-coin-capped effective balance (spec §2) from the three
//!   per-SPK aggregates `{B_mat, B_imm, A_imm}`, replacing the raw balance in the ratio bracket.
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
///   `e_out` is the value-weighted average anchor of the surviving (youngest) input value, plus
///   any net-received value weighted at `current_daa`.
///
/// FIFO is the only choice a holder cannot game in their favor (LIFO would let a churning shell
/// shield an old core); consolidation to the same SPK spends nothing net, so the full weighted
/// average survives — consolidating is free (spec §4①).
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
/// The spent value (`in_v − out_v`, when positive) consumes the oldest anchors first; what
/// remains — the youngest `preserved = min(in_v, out_v)` of input value — carries its
/// value-weighted average anchor into the outputs, diluted by any net-new value at `current_daa`.
/// Floor division; the ≤ `current_daa` invariant holds since every anchor is a past DAA.
fn fifo_survivor_anchor(group: &[(u64, u64)], out_v: u64, current_daa: u64) -> u64 {
    let in_v: u64 = group.iter().map(|(v, _)| v).sum();
    if in_v == 0 {
        return current_daa;
    }
    // Oldest first; stable ⇒ equal anchors keep tx input order (canonical tie-break).
    let mut sorted: Vec<(u64, u64)> = group.to_vec();
    sorted.sort_by_key(|&(_, anchor)| anchor);
    // Consume the spent value from the oldest end; a partially spent input keeps its remainder.
    let mut spent_v = in_v.saturating_sub(out_v);
    let mut survivor_mass: u128 = 0; // Σ v·anchor over surviving value
    for (v, anchor) in sorted {
        let consumed = spent_v.min(v);
        spent_v -= consumed;
        let surviving = v - consumed;
        survivor_mass += (surviving as u128) * (anchor as u128);
    }
    let preserved = in_v.min(out_v); // surviving input value, > 0 here
    let new_v = out_v - preserved; // net-received value enters at age zero (anchor = now)
    let total_mass = survivor_mass + (new_v as u128) * (current_daa as u128);
    (total_mass / (out_v as u128)) as u64
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

    #[test]
    fn consolidation_carries_full_weighted_average() {
        // Spec §4①: same-SPK consolidation spends nothing net ⇒ the full value-weighted
        // average anchor survives. Age mass is conserved (up to the documented floor).
        let a = spk(1);
        let inputs = [(&a, 10u64, 100u64), (&a, 20, 200)];
        let outputs = [(&a, 30u64)];
        let anchors = assign_output_effective_daa(&inputs, &outputs, D);
        assert_eq!(anchors, vec![(10 * 100 + 20 * 200) / 30]); // = 166
    }

    #[test]
    fn fifo_spends_oldest_first_on_partial_spend() {
        // Spec §4②: mixed-age stock, partial spend. The oldest input (anchor 100) is consumed
        // by the spent value; the change keeps the YOUNGEST anchor — FIFO destroys the most age,
        // never less (a holder cannot shield an old core by churning young coins).
        let a = spk(1);
        let b = spk(2);
        let inputs = [(&a, 100u64, 100u64), (&a, 100, 900)];
        let outputs = [(&b, 100u64), (&a, 100)];
        let anchors = assign_output_effective_daa(&inputs, &outputs, D);
        assert_eq!(anchors[0], D); // recipient B: reset to now
        assert_eq!(anchors[1], 900); // change at A: the young survivor, old age destroyed
    }

    #[test]
    fn partially_spent_oldest_input_keeps_its_remainder() {
        // spent_v smaller than the oldest input: the remainder of that input survives with its
        // own anchor and blends with the younger input.
        let a = spk(1);
        let b = spk(2);
        let inputs = [(&a, 100u64, 100u64), (&a, 100, 900)];
        let outputs = [(&a, 150u64), (&b, 50)];
        let anchors = assign_output_effective_daa(&inputs, &outputs, D);
        assert_eq!(anchors[0], (50 * 100 + 100 * 900) / 150); // = 633
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
    fn net_receive_dilutes_toward_now() {
        // Spec §3 "entrée nette": out_v > in_v for S ⇒ the excess enters at age zero and
        // dilutes the carried anchor toward `current_daa`.
        let a = spk(1);
        let b = spk(2);
        let inputs = [(&a, 700u64, 1_000u64), (&b, 300, 5_000)];
        let outputs = [(&a, 1_000u64)];
        let anchors = assign_output_effective_daa(&inputs, &outputs, D);
        let expected = ((700u128 * 1_000 + 300u128 * D as u128) / 1_000) as u64;
        assert_eq!(anchors, vec![expected]); // b's anchor is destroyed — only a's value carries age
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
        let inputs = [(&a, 100u64, 400u64)];
        let outputs = [(&a, 30u64), (&a, 70)];
        let anchors = assign_output_effective_daa(&inputs, &outputs, D);
        assert_eq!(anchors[0], anchors[1]);
        assert_eq!(anchors[0], 400);
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
    fn maturity_battery_is_dead() {
        // Spec §4④: 1M KRX held ≥ W (mature) + a fresh 40M pot (age 0 ⇒ a_imm = b_imm·d).
        // Under the per-coin cap the fresh pot contributes ZERO — the old holding cannot
        // "pre-charge" maturity for incoming coins (the aggregate-cap exploit).
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
