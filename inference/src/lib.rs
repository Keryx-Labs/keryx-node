/// Keryx OPoI (Optimistic Proof of Inference) — Phase 1 + Phase 2
///
/// Phase 1: synthetic f32 MLP (candle-core), tag embedded in coinbase.
/// Phase 2: fixed-point i32/i64 MLP — bit-exact on all hardware.
///   Tags are verified on-chain; collateral is slashed for fraud.
/// Phase 3 (future): real SLM weights distributed via IPFS.

pub mod model;
pub mod model_fixed;
pub mod task;

pub use task::{InferenceResult, InferenceTask};

use candle_core::Device;

// ── Phase 2 — Verification helpers ───────────────────────────────────────────

/// Minimum offset in a coinbase payload before the extra_data begins.
/// Layout: blue_score(8) + subsidy(8) + spk_version(2) + spk_len(1) = 19 bytes.
/// The real offset is 19 + spk_len, but scanning from 19 avoids the dense binary
/// header where a false `/ai:v1:` match is theoretically possible.
const COINBASE_MIN_BINARY_HEADER: usize = 19;

/// Scans raw coinbase payload bytes for an OPoI tag.
///
/// Looks for the ASCII byte sequence `/ai:v1:` followed by 16 lowercase hex chars,
/// preceded by `/{nonce_hex16}`.  Returns `(nonce, claimed_tag_hex16)` on success.
///
/// Searches only from byte offset `COINBASE_MIN_BINARY_HEADER` to skip the
/// binary-encoded fields that precede `extra_data`.
pub fn parse_opoi(payload: &[u8]) -> Option<(u64, String)> {
    const MARKER: &[u8] = b"/ai:v1:";
    const NONCE_HEX_LEN: usize = 16;
    const TAG_HEX_LEN: usize = 16;

    // Skip the fixed binary header to avoid spurious matches.
    let search_start = COINBASE_MIN_BINARY_HEADER.min(payload.len());
    let search_slice = &payload[search_start..];

    // Find the marker byte sequence.
    let relative_pos = search_slice.windows(MARKER.len()).position(|w| w == MARKER)?;
    let marker_pos = search_start + relative_pos; // absolute position in payload

    // Extract ai_tag (16 hex chars after the marker).
    let tag_start = marker_pos + MARKER.len();
    if payload.len() < tag_start + TAG_HEX_LEN {
        return None;
    }
    let tag_bytes = &payload[tag_start..tag_start + TAG_HEX_LEN];
    let claimed_tag = std::str::from_utf8(tag_bytes).ok()?;
    if !claimed_tag.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }

    // Extract nonce: the 16 hex chars immediately before the marker, preceded by '/'.
    // Layout: .../{nonce_hex16}/ai:v1:{tag_hex16}
    if marker_pos < NONCE_HEX_LEN + 1 {
        return None; // not enough room for "/{nonce}"
    }
    let slash_pos = marker_pos - NONCE_HEX_LEN - 1;
    if payload[slash_pos] != b'/' {
        return None;
    }
    let nonce_bytes = &payload[slash_pos + 1..marker_pos];
    let nonce_hex = std::str::from_utf8(nonce_bytes).ok()?;
    if !nonce_hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let nonce = u64::from_str_radix(nonce_hex, 16).ok()?;

    Some((nonce, claimed_tag.to_string()))
}

/// Phase 1 — verifies via candle-core f32 MLP (non-deterministic across hardware).
/// Kept for reference only; NOT used in consensus after Phase 2 activation.
pub fn verify_tag(nonce: u64, claimed_hex8: &str) -> bool {
    match run_inference(nonce) {
        Ok(result) => result.as_hex8() == claimed_hex8,
        Err(_) => false,
    }
}

// ── Phase 2 — Fixed-point verification ───────────────────────────────────────

/// Runs the fixed-point MLP on `nonce` and returns the 32-byte output.
/// Bit-exact on every platform — used for on-chain tag verification in Phase 2.
pub fn run_inference_fixed(nonce: u64) -> [u8; 32] {
    let task = InferenceTask::from_nonce(nonce);
    model_fixed::forward(&task.input)
}

/// Returns the 16-char hex OPoI tag produced by the fixed-point model for `nonce`.
pub fn tag_fixed(nonce: u64) -> String {
    let output = run_inference_fixed(nonce);
    hex::encode(&output[..8])
}

/// Verifies that `claimed_hex16` matches the fixed-point model output for `nonce`.
/// This is the authoritative check used by consensus during Phase 2.
pub fn verify_tag_fixed(nonce: u64, claimed_hex16: &str) -> bool {
    tag_fixed(nonce) == claimed_hex16
}

/// Errors returned by the inference engine.
#[derive(Debug, thiserror::Error)]
pub enum InferenceError {
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),
}

/// Runs the OPoI synthetic MLP on the given 64-bit miner nonce.
///
/// This is the single entry-point used by the miner.  It constructs an
/// `InferenceTask` from the nonce, runs the forward pass on CPU, and returns
/// an `InferenceResult` whose `as_hex8()` is appended to `extra_data`.
pub fn run_inference(nonce: u64) -> Result<InferenceResult, InferenceError> {
    let task = InferenceTask::from_nonce(nonce);
    let output = model::forward(&task.input, &Device::Cpu)?;
    Ok(InferenceResult { output })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inference_is_deterministic() {
        let r1 = run_inference(0xDEAD_BEEF_CAFE_1337).unwrap();
        let r2 = run_inference(0xDEAD_BEEF_CAFE_1337).unwrap();
        assert_eq!(r1.output, r2.output, "same nonce must produce same output");
    }

    #[test]
    fn different_nonces_produce_different_outputs() {
        let r1 = run_inference(1).unwrap();
        let r2 = run_inference(2).unwrap();
        assert_ne!(r1.output, r2.output, "different nonces should differ");
    }

    #[test]
    fn hex8_is_16_chars() {
        let r = run_inference(42).unwrap();
        assert_eq!(r.as_hex8().len(), 16);
    }

    // ── Phase 2 tests ─────────────────────────────────────────────────────────

    fn make_coinbase_payload(nonce: u64, tag: &str) -> Vec<u8> {
        // Binary header: blue_score(8) + subsidy(8) + spk_version(2) + spk_len(1) + spk(34)
        let mut payload = vec![0u8; 19 + 34]; // 53 bytes binary prefix
        // Append ASCII extra_data
        let extra = format!("0.2.1/2025-01-01/{:016x}/ai:v1:{}", nonce, tag);
        payload.extend_from_slice(extra.as_bytes());
        payload
    }

    #[test]
    fn parse_opoi_finds_valid_tag() {
        let nonce = 0xABCD_1234_5678_EF01u64;
        let result = run_inference(nonce).unwrap();
        let tag = result.as_hex8();
        let payload = make_coinbase_payload(nonce, &tag);

        let parsed = parse_opoi(&payload);
        assert!(parsed.is_some(), "should find OPoI tag");
        let (parsed_nonce, parsed_tag) = parsed.unwrap();
        assert_eq!(parsed_nonce, nonce);
        assert_eq!(parsed_tag, tag);
    }

    #[test]
    fn parse_opoi_returns_none_when_missing() {
        let payload: Vec<u8> = b"\x00\x00\x00\x00\x00\x00\x00\x00plain/extra/data/without/ai/tag".to_vec();
        assert!(parse_opoi(&payload).is_none());
    }

    #[test]
    fn verify_tag_accepts_correct_tag() {
        let nonce = 99u64;
        let result = run_inference(nonce).unwrap();
        assert!(verify_tag(nonce, &result.as_hex8()));
    }

    #[test]
    fn verify_tag_rejects_wrong_tag() {
        let nonce = 42u64;
        assert!(!verify_tag(nonce, "0000000000000000"));
    }
}
