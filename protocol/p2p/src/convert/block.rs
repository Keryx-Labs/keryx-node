use super::error::ConversionError;
use super::header::{HeaderFormat, Versioned};
use crate::pb as protowire;
use keryx_consensus_core::{block::Block, header::Header, pom::PomProof, pom_v4_wire, tx::Transaction};
type BlockBody = Vec<Transaction>;

/// Which encoding a peer gets for a v4 possession proof.
///
/// Separate from [`HeaderFormat`] because it flips at a different protocol version (11, against 9),
/// and derived from the negotiated version at flow registration so a single peer is always served
/// one consistent format.
///
/// Only the SEND side needs this: the receive side is self-describing, since the compact form
/// arrives in its own protobuf field.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PomWireFormat {
    /// Full borsh `PomProof` in `pom_proof` — every tile carries its whole Merkle path.
    Legacy,
    /// Multiproof encoding in `pom_proof_deduped`, dropping the ~40% of sibling hashes that the
    /// receiver can derive. See `keryx_consensus_core::pom_v4_wire`.
    Deduped,
}

impl From<u32> for PomWireFormat {
    fn from(version: u32) -> Self {
        if version >= 11 { Self::Deduped } else { Self::Legacy }
    }
}

/// A proof already encoded for one peer. At most one field is `Some`, and which one depends on the
/// peer's protocol version.
///
/// This is a parameter of the message conversions rather than something they compute, for two
/// reasons: building the compact form costs ~1.3 ms of hashing and callers serving one block to
/// many peers want to do it once (`FlowContext::encode_pom_proof_cached`), and making it explicit
/// means a serving path cannot silently ship a block with no proof, which peers then reject with
/// `PoM possession proof missing`.
#[derive(Debug, Default, Clone)]
pub struct EncodedPomProof {
    /// Full borsh `PomProof`, for `pom_proof`.
    pub legacy: Option<Vec<u8>>,
    /// Compact multiproof form, for `pom_proof_deduped`.
    pub deduped: Option<Vec<u8>>,
}

/// Encode a proof for one peer, preferring the compact form when the peer speaks it.
///
/// Any reason the compact form cannot be produced — a non-v4 proof, an unknown tier, paths that are
/// not canonically shaped — falls back to the legacy bytes rather than failing the serve. The proof
/// still reaches the peer; it is just bigger. That fallback is what makes this safe to deploy: the
/// compact form is an optimisation, never a correctness dependency.
pub fn encode_pom_proof(format: PomWireFormat, header: &Header, proof: Option<&std::sync::Arc<PomProof>>) -> EncodedPomProof {
    let Some(proof) = proof else { return EncodedPomProof::default() };
    if format == PomWireFormat::Deduped
        && let Ok(compact) = pom_v4_wire::v4_wire_context(header, proof.tier)
            .and_then(|(seed, n_chunks)| pom_v4_wire::encode_v4_deduped(proof, seed, n_chunks))
    {
        return EncodedPomProof { legacy: None, deduped: Some(compact) };
    }
    // `to_wire_bytes` keeps a pre-H4 proof (steps_v2 == None) byte-identical to the legacy
    // layout, so not-yet-updated peers still decode re-served pre-H4 blocks.
    EncodedPomProof { legacy: Some(proof.to_wire_bytes()), deduped: None }
}

/// Decode whichever proof encoding a peer sent. `header` is needed for the compact form only, to
/// re-derive the walk seed and tree shape that the encoding deliberately omits.
pub fn decode_pom_proof(
    header: &Header,
    pom_proof: Option<Vec<u8>>,
    pom_proof_deduped: Option<Vec<u8>>,
) -> Result<Option<PomProof>, ConversionError> {
    if let Some(bytes) = pom_proof_deduped {
        let tier = pom_v4_wire::deduped_tier(&bytes).ok_or(ConversionError::PomProofDecode)?;
        let (seed, n_chunks) = pom_v4_wire::v4_wire_context(header, tier).map_err(|_| ConversionError::PomProofDecode)?;
        let proof = pom_v4_wire::decode_v4_deduped(&bytes, seed, n_chunks).map_err(|_| ConversionError::PomProofDecode)?;
        return Ok(Some(proof));
    }
    if let Some(bytes) = pom_proof {
        return Ok(Some(PomProof::from_wire_bytes(&bytes).map_err(|_| ConversionError::PomProofDecode)?));
    }
    Ok(None)
}
// ----------------------------------------------------------------------------
// consensus_core to protowire
// ----------------------------------------------------------------------------

impl From<(HeaderFormat, EncodedPomProof, &Block)> for protowire::BlockMessage {
    fn from(value: (HeaderFormat, EncodedPomProof, &Block)) -> Self {
        let (header_format, pom, block) = value;
        Self {
            header: Some((header_format, block.header.as_ref()).into()),
            transactions: block.transactions.iter().map(|tx| tx.into()).collect(),
            pom_proof: pom.legacy,
            pom_proof_deduped: pom.deduped,
            // Carry the tier explicitly (falls back to the proof's tier) so it survives IBD even
            // when the full proof is absent (legacy blocks).
            pom_tier: block.pom_tier.or_else(|| block.pom_proof.as_ref().map(|p| p.tier)).map(|t| t as u32),
        }
    }
}
impl From<&BlockBody> for protowire::BlockBodyMessage {
    fn from(block_body: &BlockBody) -> Self {
        // `pom_tier`/`pom_proof*` are set by the IBD body serving flow (it has the block hash to
        // look them up); this `BlockBody` (= just transactions) carries none of them.
        Self {
            transactions: block_body.iter().map(|tx| tx.into()).collect(),
            pom_tier: None,
            pom_proof: None,
            pom_proof_deduped: None,
        }
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
        let hdr: Header = Versioned(header_format, header).try_into()?;
        // Decode the proof against the header: the compact encoding omits the walk seed and tree
        // shape precisely because both are derivable from it.
        let proof = decode_pom_proof(&hdr, block.pom_proof, block.pom_proof_deduped)?;
        let mut blk = Block::new(hdr, txs);
        if let Some(proof) = proof {
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
            v3: None,
            v4: None,
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
            v3: None,
            v4: None,
        }
    }

    /// A v11 peer must still receive pre-v4 proofs in the legacy field: only v4 proofs have a
    /// compact form, and everything older has to keep flowing untouched.
    #[test]
    fn deduped_format_falls_back_for_non_v4_proofs() {
        for proof in [dummy_proof(), dummy_proof_v2()] {
            let block = Block::from_precomputed_hash(Hash::from_bytes([1u8; 32]), vec![]).with_pom_proof(proof);
            let msg: protowire::BlockMessage = (
                HeaderFormat::Legacy,
                encode_pom_proof(PomWireFormat::Deduped, block.header.as_ref(), block.pom_proof.as_ref()),
                &block,
            )
                .into();
            assert!(msg.pom_proof.is_some(), "non-v4 proof must fall back to the legacy field");
            assert!(msg.pom_proof_deduped.is_none());

            // And it still decodes, which is the point of the fallback.
            let back: Block = Versioned(HeaderFormat::Legacy, msg).try_into().unwrap();
            assert!(back.pom_proof.is_some());
        }
    }

    /// The two encodings are mutually exclusive: a peer never has to guess which one it got, and a
    /// proof is never paid for twice.
    #[test]
    fn exactly_one_proof_encoding_is_set() {
        for format in [PomWireFormat::Legacy, PomWireFormat::Deduped] {
            let block = Block::from_precomputed_hash(Hash::from_bytes([2u8; 32]), vec![]).with_pom_proof(dummy_proof_v2());
            let msg: protowire::BlockMessage =
                (HeaderFormat::Legacy, encode_pom_proof(format, block.header.as_ref(), block.pom_proof.as_ref()), &block).into();
            assert_eq!(msg.pom_proof.is_some() as u8 + msg.pom_proof_deduped.is_some() as u8, 1, "{format:?}");
        }
    }

    /// A block with no proof stays with no proof, in either format.
    #[test]
    fn absent_proof_stays_absent() {
        for format in [PomWireFormat::Legacy, PomWireFormat::Deduped] {
            let block = Block::from_precomputed_hash(Hash::from_bytes([3u8; 32]), vec![]);
            let msg: protowire::BlockMessage =
                (HeaderFormat::Legacy, encode_pom_proof(format, block.header.as_ref(), block.pom_proof.as_ref()), &block).into();
            assert!(msg.pom_proof.is_none() && msg.pom_proof_deduped.is_none(), "{format:?}");
        }
    }

    /// The version threshold is what keeps a v11 node from sending a v10 peer bytes it cannot parse.
    #[test]
    fn wire_format_follows_protocol_version() {
        for v in 0..11 {
            assert_eq!(PomWireFormat::from(v), PomWireFormat::Legacy, "version {v}");
        }
        for v in [11, 12, 50] {
            assert_eq!(PomWireFormat::from(v), PomWireFormat::Deduped, "version {v}");
        }
    }

    #[test]
    fn pom_proof_survives_p2p_roundtrip() {
        let block = Block::from_precomputed_hash(Hash::from_bytes([1u8; 32]), vec![]).with_pom_proof(dummy_proof());
        let msg: protowire::BlockMessage =
            (HeaderFormat::Legacy, encode_pom_proof(PomWireFormat::Legacy, block.header.as_ref(), block.pom_proof.as_ref()), &block)
                .into();
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
        let msg: protowire::BlockMessage =
            (HeaderFormat::Legacy, encode_pom_proof(PomWireFormat::Legacy, block.header.as_ref(), block.pom_proof.as_ref()), &block)
                .into();
        assert!(msg.pom_proof.is_none());
        let back: Block = Versioned(HeaderFormat::Legacy, msg).try_into().unwrap();
        assert!(back.pom_proof.is_none());
    }

    #[test]
    fn proofless_forged_tier_survives_p2p_roundtrip() {
        let block = Block::from_precomputed_hash(Hash::from_bytes([4u8; 32]), vec![]).with_pom_tier(Some(4));
        let msg: protowire::BlockMessage =
            (HeaderFormat::Legacy, encode_pom_proof(PomWireFormat::Legacy, block.header.as_ref(), block.pom_proof.as_ref()), &block)
                .into();

        assert!(msg.pom_proof.is_none());
        assert_eq!(msg.pom_tier, Some(4));

        let back: Block = Versioned(HeaderFormat::Legacy, msg).try_into().unwrap();
        assert!(back.pom_proof.is_none());
        assert_eq!(back.pom_tier, Some(4));
    }

    #[test]
    fn v2_proof_survives_p2p_roundtrip() {
        let block = Block::from_precomputed_hash(Hash::from_bytes([3u8; 32]), vec![]).with_pom_proof(dummy_proof_v2());
        let msg: protowire::BlockMessage =
            (HeaderFormat::Legacy, encode_pom_proof(PomWireFormat::Legacy, block.header.as_ref(), block.pom_proof.as_ref()), &block)
                .into();
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
