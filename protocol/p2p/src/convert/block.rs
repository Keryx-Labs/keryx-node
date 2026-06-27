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
            // borsh is infallible for in-memory PomProof (plain data).
            pom_proof: block.pom_proof.as_ref().map(|p| borsh::to_vec(p.as_ref()).expect("PomProof borsh serialize")),
            // Carry the tier explicitly (falls back to the proof's tier) so it survives IBD even
            // when the full proof is absent (legacy blocks).
            pom_tier: block.pom_tier.or_else(|| block.pom_proof.as_ref().map(|p| p.tier)).map(|t| t as u32),
        }
    }
}
impl From<&BlockBody> for protowire::BlockBodyMessage {
    fn from(block_body: &BlockBody) -> Self {
        // `pom_tier` is set by the IBD body serving flow (it has the block hash to look it up);
        // this `BlockBody` (= just transactions) carries no tier.
        Self { transactions: block_body.iter().map(|tx| tx.into()).collect(), pom_tier: None }
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
            let proof: PomProof = borsh::from_slice(&bytes).map_err(|_| ConversionError::PomProofDecode)?;
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
    fn no_proof_roundtrips_as_none() {
        let block = Block::from_precomputed_hash(Hash::from_bytes([2u8; 32]), vec![]);
        let msg: protowire::BlockMessage = (HeaderFormat::Legacy, &block).into();
        assert!(msg.pom_proof.is_none());
        let back: Block = Versioned(HeaderFormat::Legacy, msg).try_into().unwrap();
        assert!(back.pom_proof.is_none());
    }
}
