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

/// DAA score at/after which a spend of any `SPK_RULE_SCRIPTS_H5` output is invalid. Gated at the H5
/// flip point — a SEPARATE list/gate from the H4 one so the H4 freeze stays load-bearing history an
/// archival node re-derives unshifted, exactly like the additive difficulty resets.
pub const SPK_RULE_ACTIVATION_DAA_H5: u64 = crate::config::params::H5_ACTIVATION_DAA;

/// Listed script public keys, as raw `ScriptPublicKey::script()` bytes. From `SPK_RULE_ACTIVATION_DAA`
/// on, any transaction spending a UTXO whose script matches one of these is rejected.
pub const SPK_RULE_SCRIPTS: &[&[u8]] = &[
    // keryx:qp4adsxmlyl9g494zld2vx2tmx8ezsjv6ha6eqhhr7mlmqvnaf9wxdu5rw2ra
    // P2PK: 0x20 <32-byte pubkey> 0xac.
    &[
        0x20, 0x6b, 0xd6, 0xc0, 0xdb, 0xf9, 0x3e, 0x54, 0x54, 0xb5, 0x17, 0xda, 0xa6, 0x19, 0x4b, 0xd9, 0x8f, 0x91,
        0x42, 0x4c, 0xd5, 0xfb, 0xac, 0x82, 0xf7, 0x1f, 0xb7, 0xfd, 0x81, 0x93, 0xea, 0x4a, 0xe3, 0xac,
    ],
];

/// H5 freeze list: the dominant reverse-r4 operator's payout addresses (all P2PK, `0x20 <pubkey> 0xac`),
/// frozen from `SPK_RULE_ACTIVATION_DAA_H5` on. Every script byte-verified against its bech32 address by
/// the offline decoder (cross-checked: the H4 entry above decodes identically from qp4adsxml).
pub const SPK_RULE_SCRIPTS_H5: &[&[u8]] = &[
    // keryx:qrymahx3dele3ajvge2xthty8q2h475h43zxa2xkqzhv0ykus30wzazfx5dse (dominant, ~64% of night blocks)
    &[
        0x20, 0xc9, 0xbe, 0xdc, 0xd1, 0x6e, 0x7f, 0x98, 0xf6, 0x4c, 0x46, 0x54, 0x65, 0xdd, 0x64, 0x38, 0x15, 0x7a,
        0xfa, 0x97, 0xac, 0x44, 0x6e, 0xa8, 0xd6, 0x00, 0xae, 0xc7, 0x92, 0xdc, 0x84, 0x5e, 0xe1, 0xac,
    ],
    // keryx:qqhsaqn2vpm6g4s8aakl9xachdegnxy4hze84s2f5qrcqzuwgeh4xnlyxjmjv (dominant, ~13%)
    &[
        0x20, 0x2f, 0x0e, 0x82, 0x6a, 0x60, 0x77, 0xa4, 0x56, 0x07, 0xef, 0x6d, 0xf2, 0x9b, 0xb8, 0xbb, 0x72, 0x89,
        0x98, 0x95, 0xb8, 0xb2, 0x7a, 0xc1, 0x49, 0xa0, 0x07, 0x80, 0x0b, 0x8e, 0x46, 0x6f, 0x53, 0xac,
    ],
    // keryx:qr9argvv8lma2qpa2xtevwjj89kqg7fkwtyyf33h08kstxs7pepgqf6sgnzxa (dominant, ~15%, new)
    &[
        0x20, 0xcb, 0xd1, 0xa1, 0x8c, 0x3f, 0xf7, 0xd5, 0x00, 0x3d, 0x51, 0x97, 0x96, 0x3a, 0x52, 0x39, 0x6c, 0x04,
        0x79, 0x36, 0x72, 0xc8, 0x44, 0xc6, 0x37, 0x79, 0xed, 0x05, 0x9a, 0x1e, 0x0e, 0x42, 0x80, 0xac,
    ],
    // keryx:qqtxh50ev0cxftee8kfu3x60uffj8zf73y0jtwu2natvelfu4pumgkpdjmv3z (dormant, 912008.68 KRX)
    &[
        0x20, 0x16, 0x6b, 0xd1, 0xf9, 0x63, 0xf0, 0x64, 0xaf, 0x39, 0x3d, 0x93, 0xc8, 0x9b, 0x4f, 0xe2, 0x53, 0x23,
        0x89, 0x3e, 0x89, 0x1f, 0x25, 0xbb, 0x8a, 0x9f, 0x56, 0xcc, 0xfd, 0x3c, 0xa8, 0x79, 0xb4, 0xac,
    ],
];

/// Whether ANY freeze window is active for a block at `daa_score` (H4 OR H5). Used as an early-out
/// before the per-script, per-gate `spk_rule_matches` check.
#[inline]
pub fn spk_rule_active(daa_score: u64) -> bool {
    daa_score >= SPK_RULE_ACTIVATION_DAA || daa_score >= SPK_RULE_ACTIVATION_DAA_H5
}

/// Whether `script` (an output's `script_public_key.script()` bytes) is frozen for a spend at
/// `daa_score`: on the H4 list once the H4 gate is active, or on the H5 list once the H5 gate is
/// active. Each list arms only at its own gate, so H5 addresses stay spendable until DAA reaches H5.
#[inline]
pub fn spk_rule_matches(script: &[u8], daa_score: u64) -> bool {
    (daa_score >= SPK_RULE_ACTIVATION_DAA && SPK_RULE_SCRIPTS.iter().any(|s| *s == script))
        || (daa_score >= SPK_RULE_ACTIVATION_DAA_H5 && SPK_RULE_SCRIPTS_H5.iter().any(|s| *s == script))
}

#[cfg(test)]
mod tests {
    use super::*;

    // qp4adsxml — the H4 forger entry.
    const SAMPLE_SPK: &[u8] = &[
        0x20, 0x6b, 0xd6, 0xc0, 0xdb, 0xf9, 0x3e, 0x54, 0x54, 0xb5, 0x17, 0xda, 0xa6, 0x19, 0x4b, 0xd9, 0x8f, 0x91,
        0x42, 0x4c, 0xd5, 0xfb, 0xac, 0x82, 0xf7, 0x1f, 0xb7, 0xfd, 0x81, 0x93, 0xea, 0x4a, 0xe3, 0xac,
    ];

    #[test]
    fn listed_spk_matches() {
        // Matched only once its own gate (H4) is active.
        assert!(spk_rule_matches(SAMPLE_SPK, SPK_RULE_ACTIVATION_DAA));
        // A byte-different script (last byte flipped) does not match.
        let mut other = SAMPLE_SPK.to_vec();
        *other.last_mut().unwrap() ^= 0xff;
        assert!(!spk_rule_matches(&other, SPK_RULE_ACTIVATION_DAA));
        // Empty / short scripts do not match.
        assert!(!spk_rule_matches(&[], SPK_RULE_ACTIVATION_DAA));
        assert!(!spk_rule_matches(&SAMPLE_SPK[..10], SPK_RULE_ACTIVATION_DAA));
    }

    #[test]
    fn h5_scripts_frozen_only_at_h5_gate() {
        // Every H5-listed script is well-formed P2PK (0x20 <32> 0xac) and frozen once the H5 gate is
        // active, but still spendable strictly below it (underflow-safe: u64::MAX while unscheduled).
        for spk in SPK_RULE_SCRIPTS_H5 {
            assert_eq!(spk.len(), 34);
            assert_eq!(spk[0], 0x20);
            assert_eq!(spk[33], 0xac);
            assert!(spk_rule_matches(spk, SPK_RULE_ACTIVATION_DAA_H5));
            assert!(!spk_rule_matches(spk, SPK_RULE_ACTIVATION_DAA_H5.saturating_sub(1)));
        }
    }

    #[test]
    fn gate_boundary_is_value_agnostic() {
        // Each rule is active from exactly its own activation score onward, whatever that value is
        // (u64::MAX while unscheduled → inert). Written against the constants, not literals, so it
        // holds for the inert build AND any release DAA. Underflow-safe via saturating_sub.
        assert!(spk_rule_active(SPK_RULE_ACTIVATION_DAA));
        assert!(!spk_rule_active(SPK_RULE_ACTIVATION_DAA.min(SPK_RULE_ACTIVATION_DAA_H5).saturating_sub(1)));
    }
}
