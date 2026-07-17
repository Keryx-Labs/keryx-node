use super::error::ConversionError;
use super::header::{HeaderFormat, Versioned};
use crate::pb as protowire;
use keryx_consensus_core::{block::Block, pom::PomProof, tx::Transaction};
type BlockBody = Vec<Transaction>;
// ----------------------------------------------------------------------------
// consensus_core to protowire
// ----------------------------------------------------------------------------

impl From<(HeaderFormat, &Block)> for protowire::BlockMessage {
    fn from(value: (HeaderFormat, &Block)) -> Self {
        let (header_format, block) = value;
        Self {
            header: Some((header_format, block.header.as_ref()).into()),
            transactions: block.transactions.iter().map(|tx| tx.into()).collect(),
            // `to_wire_bytes` keeps a pre-H4 proof (steps_v2 == None) byte-identical to the legacy
            // layout, so not-yet-updated peers still decode re-served pre-H4 blocks.
            pom_proof: block.pom_proof.as_ref().map(|p| p.to_wire_bytes()),
            // Carry the tier explicitly (falls back to the proof's tier) so it survives IBD even
            // when the full proof is absent (legacy blocks).
            pom_tier: block.pom_tier.or_else(|| block.pom_proof.as_ref().map(|p| p.tier)).map(|t| t as u32),
        }
    }
}
impl From<&BlockBody> for protowire::BlockBodyMessage {
    fn from(block_body: &BlockBody) -> Self {
        // `pom_tier`/`pom_proof` are set by the IBD body serving flow (it has the block hash to
        // look them up); this `BlockBody` (= just transactions) carries neither.
        Self { transactions: block_body.iter().map(|tx| tx.into()).collect(), pom_tier: None, pom_proof: None }
    }
}

// ----------------------------------------------------------------------------
// protowire to consensus_core
// ----------------------------------------------------------------------------

impl TryFrom<Versioned<protowire::BlockMessage>> for Block {
    type Error = ConversionError;

    fn try_from(value: Versioned<protowire::BlockMessage>) -> Result<Self, Self::Error> {
        let Versioned(header_format, block) = value;
        let header = block.header.ok_or(ConversionError::NoneValue)?;
        let txs = block.transactions.into_iter().map(|i| i.try_into()).collect::<Result<Vec<Transaction>, Self::Error>>()?;
        let mut blk = Block::new(Versioned(header_format, header).try_into()?, txs);
        if let Some(bytes) = block.pom_proof {
            let proof = PomProof::from_wire_bytes(&bytes).map_err(|_| ConversionError::PomProofDecode)?;
            blk = blk.with_pom_proof(proof);
        }
        blk = blk.with_pom_tier(block.pom_tier.map(|t| t as u8));
        Ok(blk)
    }
}

impl TryFrom<protowire::BlockBodyMessage> for BlockBody {
    type Error = ConversionError;
    fn try_from(body_message: protowire::BlockBodyMessage) -> Result<Self, Self::Error> {
        let blk_body: BlockBody =
            body_message.transactions.into_iter().map(|i| i.try_into()).collect::<Result<Vec<Transaction>, ConversionError>>()?;
        Ok(blk_body)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use keryx_consensus_core::pom::{PomOpening, PomProof};
    use keryx_hashes::Hash;

    fn dummy_proof() -> PomProof {
        PomProof {
            tier: 1,
            trace_root: [7u8; 32],
            pow_value: [9u8; 32],
            final_state: 0x1234,
            initial_trace_path: vec![[1u8; 32], [2u8; 32]],
            final_trace_path: vec![[3u8; 32]],
            openings: vec![PomOpening {
                state_before: 42,
                chunk: [5u8; 32],
                weight_path: vec![[6u8; 32], [7u8; 32]],
                trace_path_before: vec![[8u8; 32]],
                trace_path_after: vec![[9u8; 32]],
            }],
            steps_v2: None,
        }
    }

    fn dummy_proof_v2() -> PomProof {
        use keryx_consensus_core::pom::PomStep;
        PomProof {
            tier: 4,
            trace_root: [0u8; 32],
            pow_value: [3u8; 32],
            final_state: 0xbeef,
            initial_trace_path: vec![],
            final_trace_path: vec![],
            openings: vec![],
            steps_v2: Some(vec![
                PomStep { chunk: [1u8; 32], weight_path: vec![[2u8; 32]] },
                PomStep { chunk: [3u8; 32], weight_path: vec![[4u8; 32], [5u8; 32]] },
            ]),
        }
    }

    #[test]
    fn pom_proof_survives_p2p_roundtrip() {
        let block = Block::from_precomputed_hash(Hash::from_bytes([1u8; 32]), vec![]).with_pom_proof(dummy_proof());
        let msg: protowire::BlockMessage = (HeaderFormat::Legacy, &block).into();
        assert!(msg.pom_proof.is_some());
        let back: Block = Versioned(HeaderFormat::Legacy, msg).try_into().unwrap();
        let p = back.pom_proof.expect("proof preserved over the wire");
        assert_eq!(p.tier, 1);
        assert_eq!(p.trace_root, [7u8; 32]);
        assert_eq!(p.final_state, 0x1234);
        assert_eq!(p.openings.len(), 1);
        assert_eq!(p.openings[0].state_before, 42);
        assert_eq!(p.openings[0].weight_path.len(), 2);
    }

    #[test]
    fn pom_proof_survives_body_message_roundtrip() {
        // Mirrors the IBD body-sync path that wedged the network on 2026-06-29: the serving flow
        // (`v8::request_block_bodies`) borsh-encodes the proof into `BlockBodyMessage.pom_proof` and
        // the tier into `pom_tier`; the receiving flow (`ibd::flow`) borsh-decodes them back. The
        // `From<&BlockBody>` conversion itself drops both (it only has transactions), so this guards
        // the manual encode/decode the flows perform — the exact step that was missing before.
        let proof = dummy_proof();

        // Serve side (request_block_bodies): start from the transaction-only conversion, then attach.
        let mut body: protowire::BlockBodyMessage = (&BlockBody::new()).into();
        assert!(body.pom_proof.is_none() && body.pom_tier.is_none());
        body.pom_tier = Some(proof.tier as u32);
        body.pom_proof = Some(proof.to_wire_bytes());

        // Receive side (ibd::flow): decode tier + proof back out.
        assert_eq!(body.pom_tier.map(|t| t as u8), Some(1));
        let decoded = PomProof::from_wire_bytes(body.pom_proof.as_deref().unwrap()).expect("proof preserved over the body wire");
        assert_eq!(decoded.tier, proof.tier);
        assert_eq!(decoded.trace_root, proof.trace_root);
        assert_eq!(decoded.final_state, proof.final_state);
        assert_eq!(decoded.openings.len(), 1);
        assert_eq!(decoded.openings[0].state_before, 42);
        assert_eq!(decoded.openings[0].weight_path.len(), 2);
    }

    #[test]
    fn no_proof_roundtrips_as_none() {
        let block = Block::from_precomputed_hash(Hash::from_bytes([2u8; 32]), vec![]);
        let msg: protowire::BlockMessage = (HeaderFormat::Legacy, &block).into();
        assert!(msg.pom_proof.is_none());
        let back: Block = Versioned(HeaderFormat::Legacy, msg).try_into().unwrap();
        assert!(back.pom_proof.is_none());
    }

    #[test]
    fn v2_proof_survives_p2p_roundtrip() {
        let block = Block::from_precomputed_hash(Hash::from_bytes([3u8; 32]), vec![]).with_pom_proof(dummy_proof_v2());
        let msg: protowire::BlockMessage = (HeaderFormat::Legacy, &block).into();
        let back: Block = Versioned(HeaderFormat::Legacy, msg).try_into().unwrap();
        let p = back.pom_proof.expect("v2 proof preserved over the wire");
        assert_eq!(p.tier, 4);
        let steps = p.steps_v2.as_ref().expect("steps_v2 preserved");
        assert_eq!(steps.len(), 2);
        assert_eq!(steps[1].weight_path.len(), 2);
        assert!(p.openings.is_empty());
    }

    /// A pre-H4 proof must serialize to the EXACT bytes the legacy `PomProofPreH4` layout emits —
    /// the invariant that lets a not-yet-updated peer keep decoding re-served pre-H4 blocks.
    #[test]
    fn pre_h4_proof_wire_bytes_are_legacy_exact() {
        use keryx_consensus_core::pom::PomProofPreH4;
        let p = dummy_proof();
        assert_eq!(p.to_wire_bytes(), borsh::to_vec(&PomProofPreH4::from(&p)).unwrap());
    }
}
