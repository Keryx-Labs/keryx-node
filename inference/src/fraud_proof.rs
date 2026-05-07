/// OPoI ZK Fraud Proof — Phase 3 B stub.
///
/// Defines the wire format and verification API for Groth16 challenges.
/// Phase 3 B: `verify_fraud_proof` returns `StubNotImplemented` for
/// well-formed proofs; actual arkworks BN254 Groth16 verification lands
/// in Phase 3 C once the circuit VK is finalised.
///
/// Wire format of `proof_data` inside `AiChallengePayload`:
///   A (G1, compressed, 32 bytes) — not A (negated, BN254 convention)
///   B (G2, compressed, 64 bytes)
///   C (G1, compressed, 32 bytes)
///   Total: 128 bytes (arkworks compressed BN254 Groth16 serialisation)
///
/// Public inputs (prepended before `proof_data` by the verifier, not on-chain):
///   nonce_salted: [u8; 8 LE]  — nonce ^ PHASE2_OPOI_SALT
///   claimed_tag:  [u8; 8]     — the 8 bytes the miner published (hex-decoded)
///
/// The circuit proves: tag_fixed(nonce_salted) != claimed_tag, i.e.
/// the miner lied about their OPoI output.

/// Byte length of a compressed arkworks BN254 Groth16 proof.
/// A (32) + B (64) + C (32) = 128 bytes.
pub const GROTH16_PROOF_LEN: usize = 128;

/// Result returned by `verify_fraud_proof`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FraudProofResult {
    /// Proof is cryptographically valid: the miner published a wrong OPoI tag.
    Valid,
    /// Proof bytes are malformed, wrong length, or public inputs are inconsistent.
    Invalid,
    /// Phase 3 B stub: format is correct but the verifying key is not yet deployed.
    /// Consensus records the attempt but does not slash.
    StubNotImplemented,
}

/// Verify a Groth16 fraud proof submitted in an `AiChallenge` transaction.
///
/// `response_hash` is the blake2b-256 of the disputed `AiResponse` payload —
/// used to derive the public inputs for the circuit.
///
/// Returns `FraudProofResult`:
/// - `Invalid` for empty or malformed `proof_data` (wrong byte length)
/// - `StubNotImplemented` for correctly sized proofs (Phase 3 B)
/// - `Valid` will be returned by the Phase 3 C implementation
pub fn verify_fraud_proof(response_hash: &[u8; 32], proof_data: &[u8]) -> FraudProofResult {
    if proof_data.len() != GROTH16_PROOF_LEN {
        return FraudProofResult::Invalid;
    }
    // Phase 3 C: parse Groth16 proof, load hardcoded VK, verify against
    // public inputs derived from response_hash.
    let _ = response_hash;
    FraudProofResult::StubNotImplemented
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_proof_is_invalid() {
        assert_eq!(verify_fraud_proof(&[0u8; 32], &[]), FraudProofResult::Invalid);
    }

    #[test]
    fn wrong_length_proof_is_invalid() {
        assert_eq!(verify_fraud_proof(&[0u8; 32], &[0u8; 64]), FraudProofResult::Invalid);
        assert_eq!(verify_fraud_proof(&[0u8; 32], &[0u8; 256]), FraudProofResult::Invalid);
    }

    #[test]
    fn correctly_sized_proof_is_stub() {
        let proof = vec![0xABu8; GROTH16_PROOF_LEN];
        assert_eq!(verify_fraud_proof(&[1u8; 32], &proof), FraudProofResult::StubNotImplemented);
    }
}
