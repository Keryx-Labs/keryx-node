/// On-chain AI request/response payload types.
///
/// Shared between the consensus validator (node) and the miner so both
/// sides always agree on the binary layout.

/// Binary payload layout for `SUBNETWORK_ID_AI_REQUEST` transactions:
/// `[model_id: 32] [max_tokens: 4 LE] [fee: 8 LE] [prompt…]`
pub const MIN_AI_REQUEST_PAYLOAD_LEN: usize = 44;
pub const MAX_AI_REQUEST_PAYLOAD_LEN: usize = 4_096;

/// Binary payload layout for `SUBNETWORK_ID_AI_RESPONSE` transactions:
/// `[request_hash: 32] [challenge_window_end: 8 LE] [result…]`
pub const MIN_AI_RESPONSE_PAYLOAD_LEN: usize = 40;
pub const MAX_AI_RESPONSE_PAYLOAD_LEN: usize = 8_192;

/// Hex-encoded subnetwork IDs as returned by the keryxd gRPC API.
/// Used by the miner to filter transactions from block templates.
pub const SUBNETWORK_ID_AI_REQUEST_HEX: &str = "0300000000000000000000000000000000000000";
pub const SUBNETWORK_ID_AI_RESPONSE_HEX: &str = "0400000000000000000000000000000000000000";

/// Payload of a `SUBNETWORK_ID_AI_REQUEST` transaction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AiRequestPayload {
    /// 32-byte identifier for the target model (e.g. hash of model weights).
    pub model_id: [u8; 32],
    /// Maximum number of tokens to generate.
    pub max_tokens: u32,
    /// Tip in sompi offered to the miner who answers this request.
    pub fee: u64,
    /// Raw prompt bytes (UTF-8 recommended, not enforced at this layer).
    pub prompt: Vec<u8>,
}

impl AiRequestPayload {
    pub fn new(model_id: [u8; 32], max_tokens: u32, fee: u64, prompt: Vec<u8>) -> Self {
        Self { model_id, max_tokens, fee, prompt }
    }

    pub fn serialize(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(MIN_AI_REQUEST_PAYLOAD_LEN + self.prompt.len());
        out.extend_from_slice(&self.model_id);
        out.extend_from_slice(&self.max_tokens.to_le_bytes());
        out.extend_from_slice(&self.fee.to_le_bytes());
        out.extend_from_slice(&self.prompt);
        out
    }

    pub fn deserialize(data: &[u8]) -> Option<Self> {
        if data.len() < MIN_AI_REQUEST_PAYLOAD_LEN || data.len() > MAX_AI_REQUEST_PAYLOAD_LEN {
            return None;
        }
        let model_id: [u8; 32] = data[0..32].try_into().ok()?;
        let max_tokens = u32::from_le_bytes(data[32..36].try_into().ok()?);
        let fee = u64::from_le_bytes(data[36..44].try_into().ok()?);
        let prompt = data[44..].to_vec();
        Some(Self { model_id, max_tokens, fee, prompt })
    }

    /// Parse from a hex-encoded payload string (keryxd gRPC format).
    pub fn from_hex(payload_hex: &str) -> Option<Self> {
        let bytes = hex::decode(payload_hex).ok()?;
        Self::deserialize(&bytes)
    }
}

/// Payload of a `SUBNETWORK_ID_AI_RESPONSE` transaction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AiResponsePayload {
    /// Transaction ID of the `AiRequest` this response answers.
    pub request_hash: [u8; 32],
    /// Blue score at which the challenge window closes (miner's escrow is locked until then).
    pub challenge_window_end: u64,
    /// Raw inference result bytes.
    pub result: Vec<u8>,
}

impl AiResponsePayload {
    pub fn new(request_hash: [u8; 32], challenge_window_end: u64, result: Vec<u8>) -> Self {
        Self { request_hash, challenge_window_end, result }
    }

    pub fn serialize(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(MIN_AI_RESPONSE_PAYLOAD_LEN + self.result.len());
        out.extend_from_slice(&self.request_hash);
        out.extend_from_slice(&self.challenge_window_end.to_le_bytes());
        out.extend_from_slice(&self.result);
        out
    }

    pub fn deserialize(data: &[u8]) -> Option<Self> {
        if data.len() < MIN_AI_RESPONSE_PAYLOAD_LEN || data.len() > MAX_AI_RESPONSE_PAYLOAD_LEN {
            return None;
        }
        let request_hash: [u8; 32] = data[0..32].try_into().ok()?;
        let challenge_window_end = u64::from_le_bytes(data[32..40].try_into().ok()?);
        let result = data[40..].to_vec();
        Some(Self { request_hash, challenge_window_end, result })
    }

    /// Parse from a hex-encoded payload string (keryxd gRPC format).
    pub fn from_hex(payload_hex: &str) -> Option<Self> {
        let bytes = hex::decode(payload_hex).ok()?;
        Self::deserialize(&bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ai_request_roundtrip() {
        let req = AiRequestPayload::new(
            [42u8; 32],
            256,
            1_000_000,
            b"What is the capital of France?".to_vec(),
        );
        let bytes = req.serialize();
        let parsed = AiRequestPayload::deserialize(&bytes).unwrap();
        assert_eq!(req, parsed);
    }

    #[test]
    fn ai_response_roundtrip() {
        let resp = AiResponsePayload::new([7u8; 32], 900_000, b"Paris.".to_vec());
        let bytes = resp.serialize();
        let parsed = AiResponsePayload::deserialize(&bytes).unwrap();
        assert_eq!(resp, parsed);
    }

    #[test]
    fn ai_request_rejects_too_short() {
        assert!(AiRequestPayload::deserialize(&[0u8; 10]).is_none());
    }

    #[test]
    fn ai_response_rejects_too_short() {
        assert!(AiResponsePayload::deserialize(&[0u8; 10]).is_none());
    }

    #[test]
    fn ai_request_rejects_oversized() {
        let huge = vec![0u8; MAX_AI_REQUEST_PAYLOAD_LEN + 1];
        assert!(AiRequestPayload::deserialize(&huge).is_none());
    }
}
