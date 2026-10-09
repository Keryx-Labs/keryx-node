//! Private-inference era rules for AI transactions, shared by the block body check (validity)
//! and mempool admission (the same rule evaluated at the virtual score).

use keryx_consensus_core::{collateral::SERVICE_LEDGER_HORIZON_DAA, config::params::ForkActivation, tx::Transaction};
use keryx_inference::{
    AI_RESPONSE_PAYLOAD_V2_LEN, AiRequestPayload, AiResponsePayload, MAX_AI_REQUEST_PAYLOAD_LEN, PrivateRequestEnvelope,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrivateEraViolation {
    /// An AiRequest longer than the historical maximum before the gate.
    RequestTooLongBeforeActivation(usize),
    /// An AiResponse carrying an inline body before the gate.
    ResponseBodyBeforeActivation(usize),
    /// An AiRequest whose prompt is not a well-formed envelope after the gate.
    RequestNotPrivate(String),
    /// An AiResponse without an inline body once the transition window is over.
    ResponseWithoutBody,
}

impl std::fmt::Display for PrivateEraViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RequestTooLongBeforeActivation(len) => {
                write!(f, "AiRequest payload of {len} bytes exceeds {MAX_AI_REQUEST_PAYLOAD_LEN} before the private-inference activation")
            }
            Self::ResponseBodyBeforeActivation(len) => {
                write!(f, "AiResponse payload of {len} bytes carries an inline body before the private-inference activation")
            }
            Self::RequestNotPrivate(e) => write!(f, "AiRequest prompt is not a private-inference envelope: {e}"),
            Self::ResponseWithoutBody => write!(f, "AiResponse carries no inline private body"),
        }
    }
}

/// Checks one transaction against the private-inference era at `daa_score`.
///
/// Before the gate: an AiRequest stays within the historical maximum and an AiResponse carries no
/// inline body. From the gate on: every AiRequest prompt is a well-formed envelope. From the gate
/// plus the ledger horizon on: every AiResponse carries an inline body.
pub fn check_private_inference_era(tx: &Transaction, daa_score: u64, activation: ForkActivation) -> Result<(), PrivateEraViolation> {
    if !activation.is_active(daa_score) {
        if tx.is_ai_request() && tx.payload.len() > MAX_AI_REQUEST_PAYLOAD_LEN {
            return Err(PrivateEraViolation::RequestTooLongBeforeActivation(tx.payload.len()));
        }
        if tx.is_ai_response() && tx.payload.len() > AI_RESPONSE_PAYLOAD_V2_LEN {
            return Err(PrivateEraViolation::ResponseBodyBeforeActivation(tx.payload.len()));
        }
        return Ok(());
    }
    if tx.is_ai_request() {
        let req = AiRequestPayload::deserialize(&tx.payload)
            .ok_or_else(|| PrivateEraViolation::RequestNotPrivate("undecodable payload".to_string()))?;
        PrivateRequestEnvelope::parse(&req.prompt).map_err(|e| PrivateEraViolation::RequestNotPrivate(e.to_string()))?;
    }
    if tx.is_ai_response() && daa_score >= activation.daa_score().saturating_add(SERVICE_LEDGER_HORIZON_DAA) {
        let has_body = AiResponsePayload::deserialize(&tx.payload).is_some_and(|r| r.private_body.is_some());
        if !has_body {
            return Err(PrivateEraViolation::ResponseWithoutBody);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use keryx_consensus_core::subnets::{SUBNETWORK_ID_AI_REQUEST, SUBNETWORK_ID_AI_RESPONSE};
    use keryx_inference::{AiResponder, escrow_pubkey_of, seal_request};

    const GATE: u64 = 1_000;

    fn tx(subnet: keryx_consensus_core::subnets::SubnetworkId, payload: Vec<u8>) -> Transaction {
        Transaction::new(0, vec![], vec![], 0, subnet, 0, payload)
    }

    fn plaintext_request() -> Vec<u8> {
        AiRequestPayload::new([5u8; 32], 64, 1, 1, b"plain prompt".to_vec()).serialize()
    }

    fn private_request() -> Vec<u8> {
        let key = escrow_pubkey_of(&[7u8; 32]).unwrap();
        seal_request([5u8; 32], 64, 1, 1, b"sealed prompt", &[key]).unwrap().0.serialize()
    }

    fn response(body: Option<Vec<u8>>) -> Vec<u8> {
        let responder = AiResponder { escrow_pubkey: [0x33u8; 32], signature: [0x44u8; 64] };
        let r = AiResponsePayload::new_v2([7u8; 32], 1, [0x12u8; 34], 1, responder);
        match body {
            Some(b) => r.with_private_body(b).serialize(),
            None => r.serialize(),
        }
    }

    #[test]
    fn before_the_gate_plaintext_is_valid_and_private_extensions_are_not() {
        let gate = ForkActivation::new(GATE);
        let at = GATE - 1;
        assert!(check_private_inference_era(&tx(SUBNETWORK_ID_AI_REQUEST, plaintext_request()), at, gate).is_ok());
        assert!(check_private_inference_era(&tx(SUBNETWORK_ID_AI_RESPONSE, response(None)), at, gate).is_ok());
        let long = vec![0u8; MAX_AI_REQUEST_PAYLOAD_LEN + 1];
        assert_eq!(
            check_private_inference_era(&tx(SUBNETWORK_ID_AI_REQUEST, long), at, gate),
            Err(PrivateEraViolation::RequestTooLongBeforeActivation(MAX_AI_REQUEST_PAYLOAD_LEN + 1))
        );
        assert!(matches!(
            check_private_inference_era(&tx(SUBNETWORK_ID_AI_RESPONSE, response(Some(vec![1u8; 40]))), at, gate),
            Err(PrivateEraViolation::ResponseBodyBeforeActivation(_))
        ));
        // A dormant gate never activates.
        let never = ForkActivation::never();
        assert!(check_private_inference_era(&tx(SUBNETWORK_ID_AI_REQUEST, plaintext_request()), u64::MAX - 1, never).is_ok());
    }

    #[test]
    fn from_the_gate_every_request_is_sealed() {
        let gate = ForkActivation::new(GATE);
        assert!(check_private_inference_era(&tx(SUBNETWORK_ID_AI_REQUEST, private_request()), GATE, gate).is_ok());
        assert!(matches!(
            check_private_inference_era(&tx(SUBNETWORK_ID_AI_REQUEST, plaintext_request()), GATE, gate),
            Err(PrivateEraViolation::RequestNotPrivate(_))
        ));
        // Wearing the marker is not enough: the envelope must parse.
        let fake = AiRequestPayload::new([5u8; 32], 64, 1, 1, vec![0x00, b'K', b'X', b'P', 1, 2, 3]).serialize();
        assert!(matches!(
            check_private_inference_era(&tx(SUBNETWORK_ID_AI_REQUEST, fake), GATE, gate),
            Err(PrivateEraViolation::RequestNotPrivate(_))
        ));
    }

    #[test]
    fn bodyless_responses_last_one_ledger_horizon_past_the_gate() {
        let gate = ForkActivation::new(GATE);
        let end = GATE + SERVICE_LEDGER_HORIZON_DAA;
        let bodyless = || tx(SUBNETWORK_ID_AI_RESPONSE, response(None));
        assert!(check_private_inference_era(&bodyless(), GATE, gate).is_ok());
        assert!(check_private_inference_era(&bodyless(), end - 1, gate).is_ok());
        assert_eq!(check_private_inference_era(&bodyless(), end, gate), Err(PrivateEraViolation::ResponseWithoutBody));
        let v1 = tx(SUBNETWORK_ID_AI_RESPONSE, AiResponsePayload::new([7u8; 32], 1, [0x12u8; 34], 1).serialize());
        assert_eq!(check_private_inference_era(&v1, end, gate), Err(PrivateEraViolation::ResponseWithoutBody));
        assert!(check_private_inference_era(&tx(SUBNETWORK_ID_AI_RESPONSE, response(Some(vec![1u8; 40]))), end, gate).is_ok());
    }
}
