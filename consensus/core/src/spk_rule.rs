//! Consensus script-public-key rule (0x22): spends of listed outputs are invalid at/after
//! `SPK_RULE_ACTIVATION_DAA`.
//!
//! A listed output stays in the UTXO set (so `utxo_commitment` is unchanged and IBD/archival
//! re-derivation is identical) but can never be spent: `validate_populated_transaction` rejects any
//! transaction that consumes one. The check is a deterministic byte-equality on the spent output's
//! script, so every node running the same build agrees.
//!
//! ACTIVATION is `u64::MAX` (inert) in this tree. It is set to a real DAA score only in a private,
//! coordinated deployment build, kept out of any public branch until activation.

/// DAA score at/after which a spend of any `SPK_RULE_SCRIPTS` output is invalid. `u64::MAX` = inert.
/// TESTNET: gated at 3000 (same as `coin_age_verification_activation`) for validation.
/// MUST revert to `u64::MAX` before any mainnet/public build.
pub const SPK_RULE_ACTIVATION_DAA: u64 = 3000;

/// Listed script public keys, as raw `ScriptPublicKey::script()` bytes. From the activation score
/// on, any transaction spending a UTXO whose script matches one of these is rejected.
pub const SPK_RULE_SCRIPTS: &[&[u8]] = &[
    // keryx:qp4adsxmlyl9g494zld2vx2tmx8ezsjv6ha6eqhhr7mlmqvnaf9wxdu5rw2ra
    // P2PK: 0x20 <32-byte pubkey> 0xac.
    &[
        0x20, 0x6b, 0xd6, 0xc0, 0xdb, 0xf9, 0x3e, 0x54, 0x54, 0xb5, 0x17, 0xda, 0xa6, 0x19, 0x4b, 0xd9, 0x8f, 0x91,
        0x42, 0x4c, 0xd5, 0xfb, 0xac, 0x82, 0xf7, 0x1f, 0xb7, 0xfd, 0x81, 0x93, 0xea, 0x4a, 0xe3, 0xac,
    ],
    // keryxtest:qzxzrajspw5hrfu6vg2x2p6r88tfvf063yn5v44hxa6z6mjs89p5c0rthz4hh
    // Testnet validation target. P2PK: 0x20 <32-byte pubkey> 0xac.
    &[
        0x20, 0x8c, 0x21, 0xf6, 0x50, 0x0b, 0xa9, 0x71, 0xa7, 0x9a, 0x62, 0x14, 0x65, 0x07, 0x43, 0x39, 0xd6, 0x96,
        0x25, 0xfa, 0x89, 0x27, 0x46, 0x56, 0xb7, 0x37, 0x74, 0x2d, 0x6e, 0x50, 0x39, 0x43, 0x4c, 0xac,
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
    fn testnet_gate_active() {
        // TESTNET build: gated at 3000. (Mainnet/public builds MUST restore u64::MAX.)
        assert_eq!(SPK_RULE_ACTIVATION_DAA, 3000);
        assert!(!spk_rule_active(2999));
        assert!(spk_rule_active(3000));
    }
}
