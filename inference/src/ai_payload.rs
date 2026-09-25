/// On-chain AI request/response payload types.
///
/// Shared between the consensus validator (node) and the miner so both
/// sides always agree on the binary layout.

/// Binary payload layout for `SUBNETWORK_ID_AI_REQUEST` transactions:
/// `[model_id: 32] [max_tokens: 4 LE] [inference_reward: 8 LE] [priority_fee: 8 LE] [prompt…]`
///
/// The prompt is plaintext unless it starts with the private-inference marker, in which case
/// it is a [`crate::private::PrivateRequestEnvelope`] readable only by the responders it names.
pub const MIN_AI_REQUEST_PAYLOAD_LEN: usize = 52;
/// The fixed bytes before the prompt — bound as associated data by private envelopes.
pub const AI_REQUEST_HEADER_LEN: usize = MIN_AI_REQUEST_PAYLOAD_LEN;

/// Minimum priority_fee (sompi) for an AiRequest — matches the network flat minimum tx fee (0.3 KRX).
/// Requesters may set a higher value to get their request processed faster.
pub const MIN_AI_REQUEST_PRIORITY_FEE: u64 = 30_000_000;

/// Surcharge added to the per-model inference_reward minimum per 64-token increment of max_tokens.
/// Enforced alongside model_cap_enforcement_activation (same hardfork gate).
/// Formula: effective_min = base[model] + ceil(max_tokens / 64) * TOKEN_STEP  (0.005 KRX per step)
pub const INFERENCE_REWARD_TOKEN_STEP: u64 = 5_000_000;
pub const MAX_AI_REQUEST_PAYLOAD_LEN: usize = 4_096;

/// Binary payload layout for `SUBNETWORK_ID_AI_RESPONSE` transactions:
/// `[request_hash: 32] [challenge_window_end: 8 LE] [response_ipfs_cid: 34] [response_length: 4 LE]`
/// Fixed 78 bytes — result is stored off-chain on IPFS, CID pinned by the miner.
///
/// V2 (service-bond era) appends `[responder_escrow_pubkey: 32] [schnorr_signature: 64]`,
/// the signature covering the 78 v1 bytes.
///
/// A V2 payload may append one extension (private-inference era, gated by
/// `private_inference_activation`): `[ext_kind: 1 = AI_RESPONSE_EXT_PRIVATE_BODY] [ext_len: 4 LE]
/// [body: ext_len]` carrying the answer inline — a private-inference envelope encrypted to the
/// requester. The responder signature then covers `v1 bytes || extension bytes`, so a relayer
/// cannot swap the body under a signed head. Valid lengths: exactly 78, exactly 174, or
/// `174 + 5 + ext_len` with `1 <= ext_len <= MAX_AI_RESPONSE_PRIVATE_BODY_LEN`.
pub const AI_RESPONSE_PAYLOAD_LEN: usize = 78;
pub const AI_RESPONSE_PAYLOAD_V2_LEN: usize = AI_RESPONSE_PAYLOAD_LEN + 32 + 64;
/// Extension kind: an inline (encrypted) answer body.
pub const AI_RESPONSE_EXT_PRIVATE_BODY: u8 = 0x01;
/// `[ext_kind: 1] [ext_len: 4 LE]`.
pub const AI_RESPONSE_EXT_HEADER_LEN: usize = 1 + 4;
/// Largest inline body: a 4 096-token answer with envelope overhead fits comfortably.
pub const MAX_AI_RESPONSE_PRIVATE_BODY_LEN: usize = 32 * 1024;
pub const MIN_AI_RESPONSE_PAYLOAD_LEN: usize = AI_RESPONSE_PAYLOAD_LEN;
pub const MAX_AI_RESPONSE_PAYLOAD_LEN: usize =
    AI_RESPONSE_PAYLOAD_V2_LEN + AI_RESPONSE_EXT_HEADER_LEN + MAX_AI_RESPONSE_PRIVATE_BODY_LEN;

/// Binary payload layout for `SUBNETWORK_ID_AI_CHALLENGE` transactions:
/// `[response_hash: 32] [challenger_deposit: 8 LE] [challenger_spk_version: 2 LE] [challenger_spk: 32] [proof_data…]`
/// `proof_data` is empty for Phase 3 A2b stubs; 32 bytes (request_hash) for Phase 3 C re-execution.
/// `challenger_spk_version` + `challenger_spk` identify where the slashed escrow goes after CSV expiry.
pub const MIN_AI_CHALLENGE_PAYLOAD_LEN: usize = 74;
pub const MAX_AI_CHALLENGE_PAYLOAD_LEN: usize = 32_768;

/// Hex-encoded subnetwork IDs as returned by the keryxd gRPC API.
/// Used by the miner to filter transactions from block templates.
pub const SUBNETWORK_ID_AI_REQUEST_HEX: &str = "0300000000000000000000000000000000000000";
pub const SUBNETWORK_ID_AI_RESPONSE_HEX: &str = "0400000000000000000000000000000000000000";
pub const SUBNETWORK_ID_AI_CHALLENGE_HEX: &str = "0500000000000000000000000000000000000000";

/// Canonical keyless reward-vault script of a routed AiRequest (`OP_RETURN "aivault"`):
/// provably unspendable, so the vaulted amount burns unless a coinbase mints it to the
/// first accepted responder.
pub const INFERENCE_VAULT_SCRIPT: [u8; 9] = [0x6a, 0x07, b'a', b'i', b'v', b'a', b'u', b'l', b't'];

/// Payload of a `SUBNETWORK_ID_AI_REQUEST` transaction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AiRequestPayload {
    /// 32-byte identifier for the target model (e.g. hash of model weights).
    pub model_id: [u8; 32],
    /// Maximum number of tokens to generate.
    pub max_tokens: u32,
    /// Sompi paid to the miner who fulfils this request (redirected from fee burn to miner payout).
    pub inference_reward: u64,
    /// Sompi burned as a network fee (minimum MIN_AI_REQUEST_PRIORITY_FEE; set higher for priority).
    pub priority_fee: u64,
    /// Raw prompt bytes (UTF-8 recommended, not enforced at this layer).
    pub prompt: Vec<u8>,
}

impl AiRequestPayload {
    pub fn new(model_id: [u8; 32], max_tokens: u32, inference_reward: u64, priority_fee: u64, prompt: Vec<u8>) -> Self {
        Self { model_id, max_tokens, inference_reward, priority_fee, prompt }
    }

    /// The fixed header: everything before the prompt.
    pub fn header_bytes(&self) -> [u8; AI_REQUEST_HEADER_LEN] {
        Self::header_bytes_of(&self.model_id, self.max_tokens, self.inference_reward, self.priority_fee)
    }

    /// [`Self::header_bytes`] from the fields alone.
    pub fn header_bytes_of(
        model_id: &[u8; 32],
        max_tokens: u32,
        inference_reward: u64,
        priority_fee: u64,
    ) -> [u8; AI_REQUEST_HEADER_LEN] {
        let mut out = [0u8; AI_REQUEST_HEADER_LEN];
        out[0..32].copy_from_slice(model_id);
        out[32..36].copy_from_slice(&max_tokens.to_le_bytes());
        out[36..44].copy_from_slice(&inference_reward.to_le_bytes());
        out[44..52].copy_from_slice(&priority_fee.to_le_bytes());
        out
    }

    /// Whether the prompt carries the private-inference envelope marker. Cheap; a well-formed
    /// envelope is only guaranteed by [`crate::private::PrivateRequestEnvelope::parse`].
    pub fn is_private(&self) -> bool {
        crate::private::PrivateRequestEnvelope::is_private(&self.prompt)
    }

    pub fn serialize(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(AI_REQUEST_HEADER_LEN + self.prompt.len());
        out.extend_from_slice(&self.header_bytes());
        out.extend_from_slice(&self.prompt);
        out
    }

    pub fn deserialize(data: &[u8]) -> Option<Self> {
        if data.len() < MIN_AI_REQUEST_PAYLOAD_LEN || data.len() > MAX_AI_REQUEST_PAYLOAD_LEN {
            return None;
        }
        let model_id: [u8; 32] = data[0..32].try_into().ok()?;
        let max_tokens = u32::from_le_bytes(data[32..36].try_into().ok()?);
        let inference_reward = u64::from_le_bytes(data[36..44].try_into().ok()?);
        let priority_fee = u64::from_le_bytes(data[44..52].try_into().ok()?);
        let prompt = data[52..].to_vec();
        Some(Self { model_id, max_tokens, inference_reward, priority_fee, prompt })
    }

    /// Parse from a hex-encoded payload string (keryxd gRPC format).
    pub fn from_hex(payload_hex: &str) -> Option<Self> {
        let bytes = hex::decode(payload_hex).ok()?;
        Self::deserialize(&bytes)
    }
}

/// Responder identity of a V2 AiResponse: the miner's escrow pubkey and its schnorr
/// signature over the 78 v1 payload bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AiResponder {
    pub escrow_pubkey: [u8; 32],
    pub signature: [u8; 64],
}

/// Payload of a `SUBNETWORK_ID_AI_RESPONSE` transaction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AiResponsePayload {
    /// blake2b-256 hash of the AiRequest payload this response answers.
    pub request_hash: [u8; 32],
    /// Blue score at which the challenge window closes (miner's escrow is locked until then).
    pub challenge_window_end: u64,
    /// Raw 34-byte IPFS multihash (sha2-256): [0x12, 0x20, <32 bytes digest>].
    /// The full inference result is stored off-chain; fetch via IPFS gateway using this CID.
    pub response_ipfs_cid: [u8; 34],
    /// Number of tokens generated by the miner.
    pub response_length: u32,
    /// V2 responder identity; `None` for a v1 payload.
    pub responder: Option<AiResponder>,
    /// Inline answer body (the `AI_RESPONSE_EXT_PRIVATE_BODY` extension): a private-inference
    /// envelope sealed to the requester. Only a V2 payload may carry one, and the responder
    /// signature covers it. `None` when the answer lives on IPFS alone.
    pub private_body: Option<Vec<u8>>,
}

impl AiResponsePayload {
    pub fn new(request_hash: [u8; 32], challenge_window_end: u64, response_ipfs_cid: [u8; 34], response_length: u32) -> Self {
        Self { request_hash, challenge_window_end, response_ipfs_cid, response_length, responder: None, private_body: None }
    }

    pub fn new_v2(
        request_hash: [u8; 32],
        challenge_window_end: u64,
        response_ipfs_cid: [u8; 34],
        response_length: u32,
        responder: AiResponder,
    ) -> Self {
        Self { request_hash, challenge_window_end, response_ipfs_cid, response_length, responder: Some(responder), private_body: None }
    }

    /// Attaches an inline answer body. The responder must sign [`Self::signed_bytes`] AFTER
    /// attaching it, since the signature covers the extension.
    pub fn with_private_body(mut self, body: Vec<u8>) -> Self {
        self.private_body = Some(body);
        self
    }

    /// The v1 bytes: the fixed 78-byte head.
    fn v1_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(AI_RESPONSE_PAYLOAD_LEN);
        out.extend_from_slice(&self.request_hash);
        out.extend_from_slice(&self.challenge_window_end.to_le_bytes());
        out.extend_from_slice(&self.response_ipfs_cid);
        out.extend_from_slice(&self.response_length.to_le_bytes());
        out
    }

    /// The extension bytes (`[kind] [len LE] [body]`), empty without a body.
    pub fn extension_bytes(&self) -> Vec<u8> {
        match &self.private_body {
            Some(body) => {
                let mut out = Vec::with_capacity(AI_RESPONSE_EXT_HEADER_LEN + body.len());
                out.push(AI_RESPONSE_EXT_PRIVATE_BODY);
                out.extend_from_slice(&(body.len() as u32).to_le_bytes());
                out.extend_from_slice(body);
                out
            }
            None => Vec::new(),
        }
    }

    /// The message covered by the V2 responder signature: the 78 v1 bytes, followed by the
    /// extension bytes when an inline body is attached (so the body cannot be swapped under a
    /// signed head). Byte-identical to the v1 bytes for every body-less payload.
    pub fn signed_bytes(&self) -> Vec<u8> {
        let mut out = self.v1_bytes();
        out.extend_from_slice(&self.extension_bytes());
        out
    }

    pub fn serialize(&self) -> Vec<u8> {
        let mut out = self.v1_bytes();
        if let Some(r) = &self.responder {
            out.reserve(32 + 64);
            out.extend_from_slice(&r.escrow_pubkey);
            out.extend_from_slice(&r.signature);
        }
        out.extend_from_slice(&self.extension_bytes());
        out
    }

    pub fn deserialize(data: &[u8]) -> Option<Self> {
        let has_responder = data.len() >= AI_RESPONSE_PAYLOAD_V2_LEN;
        let has_extension = data.len() > AI_RESPONSE_PAYLOAD_V2_LEN;
        if !has_responder && data.len() != AI_RESPONSE_PAYLOAD_LEN {
            return None;
        }
        let request_hash: [u8; 32] = data[0..32].try_into().ok()?;
        let challenge_window_end = u64::from_le_bytes(data[32..40].try_into().ok()?);
        let response_ipfs_cid: [u8; 34] = data[40..74].try_into().ok()?;
        let response_length = u32::from_le_bytes(data[74..78].try_into().ok()?);
        let responder = if has_responder {
            Some(AiResponder { escrow_pubkey: data[78..110].try_into().ok()?, signature: data[110..174].try_into().ok()? })
        } else {
            None
        };
        let private_body = if has_extension {
            // Only the inline-body extension exists; the declared length must account for
            // every remaining byte so the encoding stays canonical.
            let ext = &data[AI_RESPONSE_PAYLOAD_V2_LEN..];
            if ext.len() < AI_RESPONSE_EXT_HEADER_LEN || ext[0] != AI_RESPONSE_EXT_PRIVATE_BODY {
                return None;
            }
            let len = u32::from_le_bytes(ext[1..5].try_into().ok()?) as usize;
            if len == 0 || len > MAX_AI_RESPONSE_PRIVATE_BODY_LEN || len != ext.len() - AI_RESPONSE_EXT_HEADER_LEN {
                return None;
            }
            Some(ext[AI_RESPONSE_EXT_HEADER_LEN..].to_vec())
        } else {
            None
        };
        Some(Self { request_hash, challenge_window_end, response_ipfs_cid, response_length, responder, private_body })
    }

    /// Parse from a hex-encoded payload string (keryxd gRPC format).
    pub fn from_hex(payload_hex: &str) -> Option<Self> {
        let bytes = hex::decode(payload_hex).ok()?;
        Self::deserialize(&bytes)
    }

    /// Encode the raw multihash as a base58btc CIDv0 string (e.g. "Qm...").
    pub fn cid_v0(&self) -> String {
        base58btc_encode(&self.response_ipfs_cid)
    }
}

/// Payload of a `SUBNETWORK_ID_AI_CHALLENGE` transaction.
///
/// Submitted by anyone who believes a miner published a fraudulent AiResponse.
/// The `challenger_deposit` is burned if the challenge is invalid.
///
/// If the challenge is accepted (proof validates via re-execution), the miner's escrow outpoint
/// is recorded as slashed with the challenger's `challenger_spk`.  After the CSV lock (36,000
/// blocks) expires, the miner can spend the escrow but only if an output goes to `challenger_spk`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AiChallengePayload {
    /// Transaction payload hash of the disputed `AiResponse` (blake2b-256).
    pub response_hash: [u8; 32],
    /// Sompi deposited by the challenger (burned if challenge fails).
    pub challenger_deposit: u64,
    /// SPK version for the challenger's receiving address (standard = 0).
    pub challenger_spk_version: u16,
    /// 32-byte script identifying the challenger's receiving address.
    /// After the slash is confirmed and CSV expires, any spend of the escrow must
    /// include an output with this exact script_public_key.
    pub challenger_spk: [u8; 32],
    /// Re-execution fraud proof: `request_hash` (32 bytes) in Phase 3 C, empty in Phase A.
    pub proof_data: Vec<u8>,
}

impl AiChallengePayload {
    pub fn new(
        response_hash: [u8; 32],
        challenger_deposit: u64,
        challenger_spk_version: u16,
        challenger_spk: [u8; 32],
        proof_data: Vec<u8>,
    ) -> Self {
        Self { response_hash, challenger_deposit, challenger_spk_version, challenger_spk, proof_data }
    }

    pub fn serialize(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(MIN_AI_CHALLENGE_PAYLOAD_LEN + self.proof_data.len());
        out.extend_from_slice(&self.response_hash);
        out.extend_from_slice(&self.challenger_deposit.to_le_bytes());
        out.extend_from_slice(&self.challenger_spk_version.to_le_bytes());
        out.extend_from_slice(&self.challenger_spk);
        out.extend_from_slice(&self.proof_data);
        out
    }

    pub fn deserialize(data: &[u8]) -> Option<Self> {
        if data.len() < MIN_AI_CHALLENGE_PAYLOAD_LEN || data.len() > MAX_AI_CHALLENGE_PAYLOAD_LEN {
            return None;
        }
        let response_hash: [u8; 32] = data[0..32].try_into().ok()?;
        let challenger_deposit = u64::from_le_bytes(data[32..40].try_into().ok()?);
        let challenger_spk_version = u16::from_le_bytes(data[40..42].try_into().ok()?);
        let challenger_spk: [u8; 32] = data[42..74].try_into().ok()?;
        let proof_data = data[74..].to_vec();
        Some(Self { response_hash, challenger_deposit, challenger_spk_version, challenger_spk, proof_data })
    }

    pub fn from_hex(payload_hex: &str) -> Option<Self> {
        let bytes = hex::decode(payload_hex).ok()?;
        Self::deserialize(&bytes)
    }
}

/// Base58btc encoding (Bitcoin/IPFS alphabet) for CIDv0 strings.
fn base58btc_encode(input: &[u8]) -> String {
    const ALPHABET: &[u8] = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    let mut digits: Vec<u8> = vec![0];
    for &byte in input {
        let mut carry = byte as u32;
        for d in digits.iter_mut() {
            carry += (*d as u32) << 8;
            *d = (carry % 58) as u8;
            carry /= 58;
        }
        while carry > 0 {
            digits.push((carry % 58) as u8);
            carry /= 58;
        }
    }
    let leading_zeros = input.iter().take_while(|&&b| b == 0).count();
    let mut out = String::with_capacity(leading_zeros + digits.len());
    for _ in 0..leading_zeros { out.push('1'); }
    for d in digits.iter().rev() { out.push(ALPHABET[*d as usize] as char); }
    out
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
            30_000_000,
            b"What is the capital of France?".to_vec(),
        );
        let bytes = req.serialize();
        let parsed = AiRequestPayload::deserialize(&bytes).unwrap();
        assert_eq!(req, parsed);
    }

    #[test]
    fn ai_response_roundtrip() {
        let cid = [0x12, 0x20, 0xAAu8, 0xBB, 0xCC, 0xDD,
                   1,2,3,4,5,6,7,8,9,10,11,12,13,14,
                   15,16,17,18,19,20,21,22,23,24,25,26,27,28];
        let resp = AiResponsePayload::new([7u8; 32], 900_000, cid, 128);
        let bytes = resp.serialize();
        assert_eq!(bytes.len(), AI_RESPONSE_PAYLOAD_LEN);
        let parsed = AiResponsePayload::deserialize(&bytes).unwrap();
        assert_eq!(resp, parsed);
    }

    #[test]
    fn ai_response_rejects_wrong_size() {
        assert!(AiResponsePayload::deserialize(&[0u8; 40]).is_none());
        assert!(AiResponsePayload::deserialize(&[0u8; 77]).is_none());
        assert!(AiResponsePayload::deserialize(&[0u8; 79]).is_none());
        assert!(AiResponsePayload::deserialize(&[0u8; 173]).is_none());
        // 174 + a truncated or zero-length extension header
        assert!(AiResponsePayload::deserialize(&[0u8; 175]).is_none());
        assert!(AiResponsePayload::deserialize(&[0u8; 179]).is_none());
    }

    fn v2_with_body(body: Vec<u8>) -> AiResponsePayload {
        let responder = AiResponder { escrow_pubkey: [0x33u8; 32], signature: [0x44u8; 64] };
        AiResponsePayload::new_v2([7u8; 32], 900_000, [0x12u8; 34], 128, responder).with_private_body(body)
    }

    #[test]
    fn ai_response_private_body_roundtrip() {
        let body = b"\x00KXP\x01 sealed answer bytes".to_vec();
        let resp = v2_with_body(body.clone());
        let bytes = resp.serialize();
        assert_eq!(bytes.len(), AI_RESPONSE_PAYLOAD_V2_LEN + AI_RESPONSE_EXT_HEADER_LEN + body.len());
        assert_eq!(bytes[AI_RESPONSE_PAYLOAD_V2_LEN], AI_RESPONSE_EXT_PRIVATE_BODY);
        let parsed = AiResponsePayload::deserialize(&bytes).unwrap();
        assert_eq!(parsed, resp);
        assert_eq!(parsed.private_body.as_deref(), Some(body.as_slice()));

        // The signed message covers the head and the extension, not the responder fields.
        let signed = resp.signed_bytes();
        assert_eq!(signed.len(), AI_RESPONSE_PAYLOAD_LEN + AI_RESPONSE_EXT_HEADER_LEN + body.len());
        assert_eq!(&signed[..AI_RESPONSE_PAYLOAD_LEN], &bytes[..AI_RESPONSE_PAYLOAD_LEN]);
        assert_eq!(&signed[AI_RESPONSE_PAYLOAD_LEN..], &bytes[AI_RESPONSE_PAYLOAD_V2_LEN..]);
        // Body-less payloads keep the historical 78-byte signed message.
        let plain = AiResponsePayload::new([7u8; 32], 900_000, [0x12u8; 34], 128);
        assert_eq!(plain.signed_bytes().len(), AI_RESPONSE_PAYLOAD_LEN);

        // The largest body is accepted, one byte more is not.
        let max = v2_with_body(vec![1u8; MAX_AI_RESPONSE_PRIVATE_BODY_LEN]).serialize();
        assert_eq!(max.len(), MAX_AI_RESPONSE_PAYLOAD_LEN);
        assert!(AiResponsePayload::deserialize(&max).is_some());
        let over = v2_with_body(vec![1u8; MAX_AI_RESPONSE_PRIVATE_BODY_LEN + 1]).serialize();
        assert!(AiResponsePayload::deserialize(&over).is_none());
    }

    #[test]
    fn ai_response_rejects_malformed_extensions() {
        let good = v2_with_body(vec![9u8; 40]).serialize();
        // Unknown extension kind.
        let mut bad = good.clone();
        bad[AI_RESPONSE_PAYLOAD_V2_LEN] = 0x02;
        assert!(AiResponsePayload::deserialize(&bad).is_none());
        // Declared length shorter or longer than the remaining bytes.
        let mut bad = good.clone();
        bad[AI_RESPONSE_PAYLOAD_V2_LEN + 1..AI_RESPONSE_PAYLOAD_V2_LEN + 5].copy_from_slice(&39u32.to_le_bytes());
        assert!(AiResponsePayload::deserialize(&bad).is_none());
        let mut bad = good.clone();
        bad[AI_RESPONSE_PAYLOAD_V2_LEN + 1..AI_RESPONSE_PAYLOAD_V2_LEN + 5].copy_from_slice(&41u32.to_le_bytes());
        assert!(AiResponsePayload::deserialize(&bad).is_none());
        // A v1 head followed by an extension is not a layout: no responder, no body.
        let mut v1_ext = AiResponsePayload::new([7u8; 32], 1, [0u8; 34], 1).serialize();
        v1_ext.extend_from_slice(&[AI_RESPONSE_EXT_PRIVATE_BODY, 1, 0, 0, 0, 0xAA]);
        assert!(AiResponsePayload::deserialize(&v1_ext).is_none());
    }

    #[test]
    fn ai_request_header_bytes_prefix_the_payload() {
        let req = AiRequestPayload::new([5u8; 32], 77, 1_000, 2_000, b"prompt".to_vec());
        let bytes = req.serialize();
        assert_eq!(&bytes[..AI_REQUEST_HEADER_LEN], &req.header_bytes());
        assert_eq!(&bytes[AI_REQUEST_HEADER_LEN..], b"prompt");
        assert!(!req.is_private());
        let private = AiRequestPayload::new([5u8; 32], 77, 1_000, 2_000, vec![0x00, b'K', b'X', b'P', 1]);
        assert!(private.is_private());
    }

    #[test]
    fn ai_response_v2_roundtrip() {
        let cid = [0x12, 0x20, 0xAAu8, 0xBB, 0xCC, 0xDD,
                   1,2,3,4,5,6,7,8,9,10,11,12,13,14,
                   15,16,17,18,19,20,21,22,23,24,25,26,27,28];
        let responder = AiResponder { escrow_pubkey: [0x33u8; 32], signature: [0x44u8; 64] };
        let resp = AiResponsePayload::new_v2([7u8; 32], 900_000, cid, 128, responder);
        let bytes = resp.serialize();
        assert_eq!(bytes.len(), AI_RESPONSE_PAYLOAD_V2_LEN);
        let parsed = AiResponsePayload::deserialize(&bytes).unwrap();
        assert_eq!(resp, parsed);
        assert_eq!(&bytes[..AI_RESPONSE_PAYLOAD_LEN], resp.signed_bytes().as_slice());
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

    #[test]
    fn ai_challenge_roundtrip() {
        let ch = AiChallengePayload::new([0xABu8; 32], 500_000, 0, [0xCDu8; 32], b"stub_proof".to_vec());
        let bytes = ch.serialize();
        let parsed = AiChallengePayload::deserialize(&bytes).unwrap();
        assert_eq!(ch, parsed);
    }

    #[test]
    fn ai_challenge_empty_proof_roundtrip() {
        let ch = AiChallengePayload::new([1u8; 32], 1_000, 0, [2u8; 32], vec![]);
        let bytes = ch.serialize();
        let parsed = AiChallengePayload::deserialize(&bytes).unwrap();
        assert_eq!(ch, parsed);
    }

    #[test]
    fn ai_challenge_rejects_too_short() {
        assert!(AiChallengePayload::deserialize(&[0u8; 10]).is_none());
    }

    #[test]
    fn ai_challenge_spk_roundtrip() {
        let spk = [0x42u8; 32];
        let ch = AiChallengePayload::new([0u8; 32], 0, 1, spk, [0xAAu8; 32].to_vec());
        let bytes = ch.serialize();
        let parsed = AiChallengePayload::deserialize(&bytes).unwrap();
        assert_eq!(parsed.challenger_spk_version, 1);
        assert_eq!(parsed.challenger_spk, spk);
        assert_eq!(parsed.proof_data, [0xAAu8; 32]);
    }
}
