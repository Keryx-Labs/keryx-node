//! Consensus script-public-key rule (0x22): spends of listed outputs are invalid at/after
//! `SPK_RULE_ACTIVATION_DAA`.
//!
//! A listed output stays in the UTXO set (so `utxo_commitment` is unchanged and IBD/archival
//! re-derivation is identical) but can never be spent: `validate_populated_transaction` rejects any
//! transaction that consumes one. The check is a deterministic byte-equality on the spent output's
//! script, so every node running the same build agrees.
//!
//! ACTIVATION is driven by the single H4 flip point (`H4_ACTIVATION_DAA`), `u64::MAX` (inert) until
//! the H4 DAA is set at release — same gate as the rest of H4. Kept out of any public branch until
//! activation.

/// DAA score at/after which a spend of any `SPK_RULE_SCRIPTS` output is invalid. Gated at the H4 flip
/// point so it arms in lockstep with the rest of H4 (`u64::MAX` = inert while H4 is unscheduled).
pub const SPK_RULE_ACTIVATION_DAA: u64 = crate::config::params::H4_ACTIVATION_DAA;

/// Listed script public keys, as raw `ScriptPublicKey::script()` bytes. From the activation score
/// on, any transaction spending a UTXO whose script matches one of these is rejected.
pub const SPK_RULE_SCRIPTS: &[&[u8]] = &[
    // keryx:qp4adsxmlyl9g494zld2vx2tmx8ezsjv6ha6eqhhr7mlmqvnaf9wxdu5rw2ra
    // P2PK: 0x20 <32-byte pubkey> 0xac.
    &[
        0x20, 0x6b, 0xd6, 0xc0, 0xdb, 0xf9, 0x3e, 0x54, 0x54, 0xb5, 0x17, 0xda, 0xa6, 0x19, 0x4b, 0xd9, 0x8f, 0x91,
        0x42, 0x4c, 0xd5, 0xfb, 0xac, 0x82, 0xf7, 0x1f, 0xb7, 0xfd, 0x81, 0x93, 0xea, 0x4a, 0xe3, 0xac,
    ],
];

/// Whether the rule is active for a block at `daa_score`.
#[inline]
pub fn spk_rule_active(daa_score: u64) -> bool {
    daa_score >= SPK_RULE_ACTIVATION_DAA
}

/// Whether `script` (an output's `script_public_key.script()` bytes) is on the rule list.
#[inline]
pub fn spk_rule_matches(script: &[u8]) -> bool {
    SPK_RULE_SCRIPTS.iter().any(|s| *s == script)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_SPK: &[u8] = &[
        0x20, 0x6b, 0xd6, 0xc0, 0xdb, 0xf9, 0x3e, 0x54, 0x54, 0xb5, 0x17, 0xda, 0xa6, 0x19, 0x4b, 0xd9, 0x8f, 0x91,
        0x42, 0x4c, 0xd5, 0xfb, 0xac, 0x82, 0xf7, 0x1f, 0xb7, 0xfd, 0x81, 0x93, 0xea, 0x4a, 0xe3, 0xac,
    ];

    #[test]
    fn listed_spk_matches() {
        assert!(spk_rule_matches(SAMPLE_SPK));
        // A byte-different script (last byte flipped) does not match.
        let mut other = SAMPLE_SPK.to_vec();
        *other.last_mut().unwrap() ^= 0xff;
        assert!(!spk_rule_matches(&other));
        // Empty / short scripts do not match.
        assert!(!spk_rule_matches(&[]));
        assert!(!spk_rule_matches(&SAMPLE_SPK[..10]));
    }

    #[test]
    fn gate_boundary_is_value_agnostic() {
        // The rule is active from exactly SPK_RULE_ACTIVATION_DAA onward, whatever that value is
        // (u64::MAX while H4 is unscheduled → inert). Written against the constant, not a literal,
        // so it holds for the inert build AND any release DAA. Underflow-safe: at u64::MAX the
        // `- 1` is u64::MAX - 1, still below the gate.
        assert!(spk_rule_active(SPK_RULE_ACTIVATION_DAA));
        assert!(!spk_rule_active(SPK_RULE_ACTIVATION_DAA - 1));
    }
}
