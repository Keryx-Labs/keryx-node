//! Verifiable-inference era rules for AI transactions, shared by the block body check (validity)
//! and mempool admission (the same rule evaluated at the virtual score).

use keryx_consensus_core::{collateral::SERVICE_LEDGER_HORIZON_DAA, config::params::ForkActivation, tx::Transaction};
use keryx_inference::AiResponsePayload;
use keryx_verifiable_inference::opening::{Opening, SignedPayload, seed_block};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifiableEraViolation {
    /// An opening transaction before the gate.
    OpeningBeforeActivation,
    /// An AiResponse carrying a trace commitment before the gate.
    TraceBeforeActivation,
    /// An AiResponse with an inline body but no trace commitment once the transition window is over.
    ResponseWithoutTrace,
    /// An opening whose payload does not decode, or that names its seed block before
    /// `trace_seed_bind_activation` or does not name it after.
    MalformedOpening,
    /// An AiResponse body longer than the pre-chain maximum before `trace_chain_activation`.
    BodyTooLongBeforeChain(usize),
}

impl std::fmt::Display for VerifiableEraViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OpeningBeforeActivation => write!(f, "opening transaction before the verifiable-inference activation"),
            Self::TraceBeforeActivation => write!(f, "AiResponse carries a trace commitment before the verifiable-inference activation"),
            Self::ResponseWithoutTrace => write!(f, "AiResponse carries no trace commitment"),
            Self::MalformedOpening => write!(f, "opening payload does not decode"),
            Self::BodyTooLongBeforeChain(len) => write!(f, "AiResponse body of {len} bytes exceeds {} before the stage-2 activation", keryx_inference::MAX_AI_RESPONSE_PRIVATE_BODY_LEN_H14),
        }
    }
}

/// Checks one transaction against the verifiable-inference era at `daa_score`.
pub fn check_verifiable_inference_era(
    tx: &Transaction,
    daa_score: u64,
    activation: ForkActivation,
    chain_activation: ForkActivation,
    seed_bind_activation: ForkActivation,
) -> Result<(), VerifiableEraViolation> {
    let response = if tx.is_ai_response() { AiResponsePayload::deserialize(&tx.payload) } else { None };
    if !chain_activation.is_active(daa_score) {
        let body = response.as_ref().and_then(|r| r.private_body.as_ref()).map_or(0, Vec::len);
        if body > keryx_inference::MAX_AI_RESPONSE_PRIVATE_BODY_LEN_H14 {
            return Err(VerifiableEraViolation::BodyTooLongBeforeChain(body));
        }
    }
    if !activation.is_active(daa_score) {
        if tx.is_ai_opening() {
            return Err(VerifiableEraViolation::OpeningBeforeActivation);
        }
        if response.as_ref().is_some_and(|r| r.trace.is_some()) {
            return Err(VerifiableEraViolation::TraceBeforeActivation);
        }
        return Ok(());
    }
    if tx.is_ai_opening() {
        let signed = SignedPayload::split(&tx.payload).ok_or(VerifiableEraViolation::MalformedOpening)?;
        if seed_block(signed.body).is_some() != seed_bind_activation.is_active(daa_score) {
            return Err(VerifiableEraViolation::MalformedOpening);
        }
        Opening::decode(signed.body).ok_or(VerifiableEraViolation::MalformedOpening)?;
    }
    if daa_score >= activation.daa_score().saturating_add(SERVICE_LEDGER_HORIZON_DAA)
        && response.as_ref().is_some_and(|r| r.private_body.is_some() && r.trace.is_none())
    {
        return Err(VerifiableEraViolation::ResponseWithoutTrace);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use keryx_consensus_core::{
        subnets::{SUBNETWORK_ID_AI_OPENING, SUBNETWORK_ID_AI_RESPONSE, SubnetworkId},
        tx::Transaction,
    };
    use keryx_inference::{AiResponder, TraceCommitment};
    use keryx_verifiable_inference::opening::Opening;

    const GATE: u64 = 1_000;

    fn tx(subnet: SubnetworkId, payload: Vec<u8>) -> Transaction {
        Transaction::new(0, vec![], vec![], 0, subnet, 0, payload)
    }

    fn response(trace: bool) -> Vec<u8> {
        let r = AiResponsePayload::new_v2([7u8; 32], 1, [0u8; 34], 1, AiResponder { escrow_pubkey: [1; 32], signature: [2; 64] })
            .with_private_body(vec![3u8; 40]);
        let r = if trace { r.with_trace(TraceCommitment { trace_root: [4; 32], n_prompt: 3, n_pos: 9, stage: 1, tokens_root: None }) } else { r };
        r.serialize()
    }

    fn opening() -> Vec<u8> {
        let mut p = Opening { response_txid: [5; 32], points: vec![], prefix: None, links: vec![], experts: vec![], resid: None }.encode();
        p.extend_from_slice(&[6u8; 96]);
        p
    }

    #[test]
    fn dormant_gate_rejects_everything_new() {
        let gate = ForkActivation::never();
        assert_eq!(check_verifiable_inference_era(&tx(SUBNETWORK_ID_AI_OPENING, opening()), 5, gate, ForkActivation::never(), ForkActivation::never()), Err(VerifiableEraViolation::OpeningBeforeActivation));
        assert_eq!(check_verifiable_inference_era(&tx(SUBNETWORK_ID_AI_RESPONSE, response(true)), 5, gate, ForkActivation::never(), ForkActivation::never()), Err(VerifiableEraViolation::TraceBeforeActivation));
        assert!(check_verifiable_inference_era(&tx(SUBNETWORK_ID_AI_RESPONSE, response(false)), u64::MAX - 1, gate, ForkActivation::never(), ForkActivation::never()).is_ok());
    }

    #[test]
    fn active_gate_rules() {
        let gate = ForkActivation::new(GATE);
        assert!(check_verifiable_inference_era(&tx(SUBNETWORK_ID_AI_OPENING, opening()), GATE, gate, ForkActivation::never(), ForkActivation::never()).is_ok());
        assert!(check_verifiable_inference_era(&tx(SUBNETWORK_ID_AI_RESPONSE, response(true)), GATE, gate, ForkActivation::never(), ForkActivation::never()).is_ok());
        assert_eq!(
            check_verifiable_inference_era(&tx(SUBNETWORK_ID_AI_OPENING, vec![1u8; 200]), GATE, gate, ForkActivation::never(), ForkActivation::never()),
            Err(VerifiableEraViolation::MalformedOpening)
        );
        // Bodies without a trace stay valid during the transition window, not after it.
        assert!(check_verifiable_inference_era(&tx(SUBNETWORK_ID_AI_RESPONSE, response(false)), GATE, gate, ForkActivation::never(), ForkActivation::never()).is_ok());
        let end = GATE + SERVICE_LEDGER_HORIZON_DAA;
        assert_eq!(
            check_verifiable_inference_era(&tx(SUBNETWORK_ID_AI_RESPONSE, response(false)), end, gate, ForkActivation::never(), ForkActivation::never()),
            Err(VerifiableEraViolation::ResponseWithoutTrace)
        );
        assert!(check_verifiable_inference_era(&tx(SUBNETWORK_ID_AI_RESPONSE, response(true)), end, gate, ForkActivation::never(), ForkActivation::never()).is_ok());
    }

    #[test]
    fn a_long_body_waits_for_the_stage_2_gate() {
        let (gate, chain) = (ForkActivation::new(GATE), ForkActivation::new(2 * GATE));
        let body = |len: usize| {
            AiResponsePayload::new_v2([7u8; 32], 1, [0u8; 34], 1, AiResponder { escrow_pubkey: [1; 32], signature: [2; 64] })
                .with_private_body(vec![3u8; len])
                .serialize()
        };
        let (h14, max) = (keryx_inference::MAX_AI_RESPONSE_PRIVATE_BODY_LEN_H14, keryx_inference::MAX_AI_RESPONSE_PRIVATE_BODY_LEN);
        assert!(check_verifiable_inference_era(&tx(SUBNETWORK_ID_AI_RESPONSE, body(h14)), GATE, gate, chain, ForkActivation::never()).is_ok());
        assert_eq!(
            check_verifiable_inference_era(&tx(SUBNETWORK_ID_AI_RESPONSE, body(h14 + 1)), 2 * GATE - 1, gate, chain, ForkActivation::never()),
            Err(VerifiableEraViolation::BodyTooLongBeforeChain(h14 + 1))
        );
        assert!(check_verifiable_inference_era(&tx(SUBNETWORK_ID_AI_RESPONSE, body(max)), 2 * GATE, gate, chain, ForkActivation::never()).is_ok());
    }

    #[test]
    fn opening_names_its_seed_block_exactly_from_its_gate() {
        let gate = ForkActivation::new(GATE);
        let bind = ForkActivation::new(2 * GATE);
        let body = Opening { response_txid: [5; 32], points: vec![], prefix: None, links: vec![], experts: vec![], resid: None }.encode();
        let mut p = keryx_verifiable_inference::opening::seeded(&body, &[8; 32]).unwrap();
        p.extend_from_slice(&[6u8; 96]);
        let seeded = tx(SUBNETWORK_ID_AI_OPENING, p);
        assert_eq!(check_verifiable_inference_era(&seeded, 2 * GATE - 1, gate, ForkActivation::never(), bind), Err(VerifiableEraViolation::MalformedOpening));
        assert!(check_verifiable_inference_era(&seeded, 2 * GATE, gate, ForkActivation::never(), bind).is_ok());
        assert!(check_verifiable_inference_era(&tx(SUBNETWORK_ID_AI_OPENING, opening()), 2 * GATE - 1, gate, ForkActivation::never(), bind).is_ok());
        assert_eq!(
            check_verifiable_inference_era(&tx(SUBNETWORK_ID_AI_OPENING, opening()), 2 * GATE, gate, ForkActivation::never(), bind),
            Err(VerifiableEraViolation::MalformedOpening)
        );
    }
}
