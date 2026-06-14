/// Synthetic OPoI liveness tasks — Level-1 anti "zero-inference" net.
///
/// The protocol issues one synthetic inference task per epoch, derived
/// deterministically from chain state. Once `synthetic_liveness_activation` is
/// live, every miner must answer the current task on-chain (an `AiResponse` tx)
/// within the liveness window, or its blocks are rejected. The task input is
/// chain-derived (not miner-chosen), so it cannot be precomputed or self-dealt.
///
/// This module is intentionally PURE: it derives the canonical
/// [`AiRequestPayload`] from a 32-byte `seed` and the miner's declared model
/// set. The caller computes `request_hash = blake2b-256(payload.serialize())`
/// (blake2b lives in the consensus/miner crates, not here) and matches it
/// against on-chain `AiResponse` txs. Keeping the hash out of this crate avoids
/// pulling a hashing dependency into the inference engine.
use crate::ai_payload::AiRequestPayload;

/// Length of a synthetic-liveness epoch, in blocks. The protocol issues exactly
/// one synthetic task per epoch; `epoch = daa_score / SYNTHETIC_EPOCH_BLOCKS`.
/// Shared by the node (recording/enforcement) and the miner (scheduling) so both
/// agree on epoch boundaries. ~36k blocks ≈ 1 hour at 10 BPS.
pub const SYNTHETIC_EPOCH_BLOCKS: u64 = 36_000;

/// Tokens the synthetic task asks for. Deliberately small — this proves
/// liveness ("you are online and serving the model you declared"), not a real
/// workload. Correctness of the answer is the (future) Level-2 challenger's job.
pub const SYNTHETIC_MAX_TOKENS: u32 = 16;

/// ASCII prefix of every synthetic prompt. Lets tooling and the miner recognise
/// a protocol-issued liveness task versus an organic, fee-bearing AiRequest.
pub const SYNTHETIC_PROMPT_PREFIX: &[u8] = b"keryx-opoi-liveness:";

/// Domain-separation tag mixed into the synthetic-task seed (versioned).
/// The seed is `blake2b(SYNTHETIC_SEED_DOMAIN || epoch_le || escrow_pubkey)`
/// today (2a: predictable per epoch+miner). The 2b hardening mixes in a
/// finalized chain anchor by bumping this tag to `…-v2` — the bump alone
/// guarantees old and new seeds never collide across the upgrade.
pub const SYNTHETIC_SEED_DOMAIN: &[u8] = b"keryx-opoi-synthetic-seed-v1";

/// Picks one `model_id` from the miner's declared set, indexed by the seed.
///
/// `declared_models` MUST be the miner's on-chain `ai:cap:` set in its canonical
/// (as-parsed) order, so every node selects the same target. Returns `None` if
/// the set is empty (a miner that declared no capability has nothing to prove
/// and — once enforcement is live — cannot produce a valid block).
fn pick_model(seed: &[u8; 32], declared_models: &[[u8; 32]]) -> Option<[u8; 32]> {
    if declared_models.is_empty() {
        return None;
    }
    // First 8 seed bytes → index into the declared set.
    let idx = (u64::from_le_bytes(seed[..8].try_into().unwrap()) as usize) % declared_models.len();
    Some(declared_models[idx])
}

/// Builds the deterministic synthetic prompt for `(seed, epoch)`.
/// Fixed given the inputs so every node derives byte-identical content.
fn synthetic_prompt(seed: &[u8; 32], epoch: u64) -> Vec<u8> {
    let mut p = Vec::with_capacity(SYNTHETIC_PROMPT_PREFIX.len() + 8 + 32);
    p.extend_from_slice(SYNTHETIC_PROMPT_PREFIX);
    p.extend_from_slice(&epoch.to_le_bytes());
    p.extend_from_slice(seed);
    p
}

/// Derives the canonical synthetic [`AiRequestPayload`] for an epoch.
///
/// * `seed` — 32 bytes the caller computes as `blake2b(epoch_anchor_hash || epoch)`.
///   It must be unpredictable before the epoch's anchor block exists yet
///   deterministic for all nodes afterwards.
/// * `declared_models` — the miner's on-chain `ai:cap:` model set (canonical
///   order). Restricting the target to declared models means a `--light` miner
///   is never asked for a model it never claimed.
/// * `epoch` — `daa_score / SYNTHETIC_EPOCH_BLOCKS`.
///
/// Returns `None` when the miner declared no models. The synthetic request
/// carries zero `inference_reward`/`priority_fee`: it is protocol-issued, never
/// posted as a transaction, and never paid — `request_hash` is only a binding
/// label that the answering `AiResponse` must reference.
/// Canonical synthetic-task seed for `(epoch, escrow_pubkey)`, shared verbatim by
/// the node (recording/enforcement) and the miner (answering) so both sides always
/// derive byte-identical tasks. A drift here would brick honest miners, so this is
/// the single source of truth.
///
/// 2a: `blake2b(SYNTHETIC_SEED_DOMAIN || epoch_le || escrow_pubkey)[..32]` —
/// predictable per epoch+miner. The 2b hardening mixes a finalized chain anchor in
/// here and bumps `SYNTHETIC_SEED_DOMAIN`; no call site changes.
pub fn synthetic_seed(epoch: u64, escrow_pubkey: &[u8; 32]) -> [u8; 32] {
    let mut buf = Vec::with_capacity(SYNTHETIC_SEED_DOMAIN.len() + 8 + 32);
    buf.extend_from_slice(SYNTHETIC_SEED_DOMAIN);
    buf.extend_from_slice(&epoch.to_le_bytes());
    buf.extend_from_slice(escrow_pubkey);
    let mut out = [0u8; 32];
    out.copy_from_slice(&blake2b_simd::blake2b(&buf).as_bytes()[..32]);
    out
}

pub fn derive_synthetic_request(
    seed: &[u8; 32],
    declared_models: &[[u8; 32]],
    epoch: u64,
) -> Option<AiRequestPayload> {
    let model_id = pick_model(seed, declared_models)?;
    Some(AiRequestPayload::new(
        model_id,
        SYNTHETIC_MAX_TOKENS,
        0, // inference_reward — protocol-issued, never paid
        0, // priority_fee — not a real fee-bearing request
        synthetic_prompt(seed, epoch),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai_payload::MIN_AI_REQUEST_PAYLOAD_LEN;

    fn m(b: u8) -> [u8; 32] {
        [b; 32]
    }

    #[test]
    fn deterministic_for_same_inputs() {
        let seed = [7u8; 32];
        let models = [m(1), m(2), m(3)];
        let a = derive_synthetic_request(&seed, &models, 42).unwrap();
        let b = derive_synthetic_request(&seed, &models, 42).unwrap();
        assert_eq!(a.serialize(), b.serialize());
    }

    #[test]
    fn empty_declared_set_yields_none() {
        let seed = [0u8; 32];
        assert!(derive_synthetic_request(&seed, &[], 0).is_none());
    }

    #[test]
    fn single_model_set_always_picks_it() {
        let only = m(9);
        for epoch in 0..50 {
            let mut seed = [0u8; 32];
            seed[..8].copy_from_slice(&(epoch as u64).to_le_bytes());
            let req = derive_synthetic_request(&seed, &[only], epoch).unwrap();
            assert_eq!(req.model_id, only);
        }
    }

    #[test]
    fn model_selection_varies_with_seed() {
        let models = [m(1), m(2), m(3), m(4), m(5)];
        let mut seen = std::collections::HashSet::new();
        for i in 0u64..256 {
            let mut seed = [0u8; 32];
            seed[..8].copy_from_slice(&i.to_le_bytes());
            let req = derive_synthetic_request(&seed, &models, 0).unwrap();
            seen.insert(req.model_id);
        }
        // Over many seeds the picker must reach more than one declared model.
        assert!(seen.len() > 1, "seed-indexed picker should spread across the declared set");
    }

    #[test]
    fn different_epochs_produce_different_requests() {
        let seed = [3u8; 32];
        let models = [m(1)];
        let e1 = derive_synthetic_request(&seed, &models, 100).unwrap();
        let e2 = derive_synthetic_request(&seed, &models, 101).unwrap();
        // Same model (single-element set) but the prompt embeds the epoch.
        assert_eq!(e1.model_id, e2.model_id);
        assert_ne!(e1.serialize(), e2.serialize());
    }

    #[test]
    fn prompt_is_tagged_and_payload_is_well_formed() {
        let seed = [5u8; 32];
        let req = derive_synthetic_request(&seed, &[m(1)], 1).unwrap();
        assert!(req.prompt.starts_with(SYNTHETIC_PROMPT_PREFIX));
        assert_eq!(req.inference_reward, 0);
        assert_eq!(req.priority_fee, 0);
        assert_eq!(req.max_tokens, SYNTHETIC_MAX_TOKENS);
        // Round-trips through the on-chain payload codec within size bounds.
        let bytes = req.serialize();
        assert!(bytes.len() >= MIN_AI_REQUEST_PAYLOAD_LEN);
        assert_eq!(AiRequestPayload::deserialize(&bytes), Some(req));
    }
}
