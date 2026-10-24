use keryx_hashes::{Hash, Hasher, TransactionHash};
use keryx_utils::mem_size::MemSizeEstimator;
use serde::{Deserialize, Serialize};

use crate::tx::ScriptPublicKey;

/// Fraction of each accepted block subsidy held in escrow as miner collateral (basis points).
/// 2 000 BPS = 20 %.
pub const COLLATERAL_RATE_BPS: u64 = 2_000;

/// Number of blocks during which an OPoI result may be challenged after its block is accepted.
/// At 10 BPS, 36 000 blocks ≈ 1 hour — enough time for any active node to detect and submit
/// a challenge, while keeping the escrow lock reasonable for honest miners.
pub const CHALLENGE_WINDOW_BLOCKS: u64 = 36_000;

/// DAA window during which a claim stays burnable (~10 h at 10 BPS): a disposable identity
/// leaves this much production on the table. Bounded by the cold-refold reach — the vault must
/// be rebuildable from the chain a pruned node retains (see the boot assert).
pub const SERVICE_BURNABLE_WINDOW_DAA: u64 = 360_000;

/// Escrow CSV lock at/after the service-bond gate: burnable window (360 000) + finality depth
/// (432 000), ≈ 22 h at 10 BPS. A claim created at C is burnable by misses up to
/// C + SERVICE_BURNABLE_WINDOW_DAA, enforceable at most finality later — this lock guarantees
/// the burn is always in force before the claim unlocks.
pub const SERVICE_BOND_CSV_WINDOW_BLOCKS: u64 = 792_000;

/// Per-miner collateral balance tracked on-chain.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct CollateralEntry {
    pub accumulated_sompi: u64,
}

impl MemSizeEstimator for CollateralEntry {
    fn estimate_mem_bytes(&self) -> usize {
        size_of::<Self>()
    }
}

/// Returns a stable 32-byte store key derived from a miner's ScriptPublicKey.
///
/// Encodes `[version_le (2 bytes), script…]` and hashes with TransactionHash (blake2b).
/// This must remain stable across node restarts — never change the encoding.
pub fn miner_key(spk: &ScriptPublicKey) -> Hash {
    let mut data = Vec::with_capacity(2 + spk.script().len());
    data.extend_from_slice(&spk.version().to_le_bytes());
    data.extend_from_slice(spk.script());
    TransactionHash::hash(data)
}

/// Store key of an announced escrow pubkey, verbatim. The escrow key is the HOT key: it signs
/// V2 AiResponses and spends the CSV escrow outputs. The service-ledger identity that takes the
/// penalties is [`miner_key`] of the payout SPK — the escrow key is only bound to it through a
/// delegation cert.
pub fn escrow_miner_key(pubkey: &[u8; 32]) -> Hash {
    Hash::from_bytes(*pubkey)
}

/// Marker of the escrow announcement in the coinbase extra_data:
/// `/escrow:<64 hex chars of the 32-byte x-only schnorr pubkey>`.
pub const ESCROW_MARKER: &[u8] = b"/escrow:";
/// Marker of the escrow delegation cert in the coinbase extra_data:
/// `/esig:<128 hex chars of the 64-byte schnorr signature>`.
pub const ESIG_MARKER: &[u8] = b"/esig:";

/// Domain separator of the escrow delegation signature.
pub const ESCROW_DELEGATION_DOMAIN: &[u8] = b"KeryxEscrowDelegationV1";

fn parse_hex_after_marker<const N: usize>(extra_data: &[u8], marker: &[u8]) -> Option<[u8; N]> {
    let pos = extra_data.windows(marker.len()).position(|w| w == marker)?;
    let hex_start = pos + marker.len();
    if hex_start + N * 2 > extra_data.len() {
        return None;
    }
    let mut out = [0u8; N];
    for (i, byte) in out.iter_mut().enumerate() {
        let hi = std::str::from_utf8(&extra_data[hex_start + i * 2..hex_start + i * 2 + 2]).ok()?;
        *byte = u8::from_str_radix(hi, 16).ok()?;
    }
    Some(out)
}

/// The announced escrow pubkey in a coinbase extra_data, if any.
pub fn parse_escrow_pubkey(extra_data: &[u8]) -> Option<[u8; 32]> {
    parse_hex_after_marker::<32>(extra_data, ESCROW_MARKER)
}

/// The escrow delegation signature in a coinbase extra_data, if any.
pub fn parse_escrow_esig(extra_data: &[u8]) -> Option<[u8; 64]> {
    parse_hex_after_marker::<64>(extra_data, ESIG_MARKER)
}


/// The message a payout key signs (once, offline) to delegate service duty to an escrow key.
pub fn escrow_delegation_message(escrow_pubkey: &[u8; 32]) -> [u8; 32] {
    let mut hasher = blake2b_simd::Params::new().hash_length(32).to_state();
    hasher.update(ESCROW_DELEGATION_DOMAIN);
    hasher.update(escrow_pubkey);
    let mut msg = [0u8; 32];
    msg.copy_from_slice(hasher.finalize().as_bytes());
    msg
}

/// Verifies an escrow delegation: `sig` must be a schnorr signature over
/// [`escrow_delegation_message`] by the x-only key inside the standard schnorr P2PK payout
/// script (`0x20 <key32> OP_CHECKSIG`, version 0). Any other payout script form cannot carry a
/// delegation and fails.
pub fn verify_escrow_delegation(payout_version: u16, payout_script: &[u8], escrow_pubkey: &[u8; 32], sig: &[u8; 64]) -> bool {
    if payout_version != 0 || payout_script.len() != 34 || payout_script[0] != 0x20 || payout_script[33] != 0xac {
        return false;
    }
    let Ok(payout_key) = secp256k1::XOnlyPublicKey::from_slice(&payout_script[1..33]) else {
        return false;
    };
    let msg = secp256k1::Message::from_digest(escrow_delegation_message(escrow_pubkey));
    let Ok(signature) = secp256k1::schnorr::Signature::from_slice(sig) else {
        return false;
    };
    secp256k1::SECP256K1.verify_schnorr(&signature, &msg, &payout_key).is_ok()
}

/// Domain separator of the V2 AiResponse responder signature.
pub const RESPONDER_SIG_DOMAIN: &[u8] = b"KeryxServiceResponderV1";

/// The message a V2 responder signs: the domain-tagged blake2b-256 of the payload's signed
/// bytes (`AiResponsePayload::signed_bytes`).
pub fn responder_signature_message(signed_bytes: &[u8]) -> [u8; 32] {
    let mut hasher = blake2b_simd::Params::new().hash_length(32).to_state();
    hasher.update(RESPONDER_SIG_DOMAIN);
    hasher.update(signed_bytes);
    let mut msg = [0u8; 32];
    msg.copy_from_slice(hasher.finalize().as_bytes());
    msg
}

/// Verifies a V2 responder signature: schnorr by `escrow_pubkey` over
/// [`responder_signature_message`] of the payload's signed bytes.
pub fn verify_responder_signature(escrow_pubkey: &[u8; 32], signature: &[u8; 64], signed_bytes: &[u8]) -> bool {
    let msg = responder_signature_message(signed_bytes);
    let Ok(pk) = secp256k1::XOnlyPublicKey::from_slice(escrow_pubkey) else {
        return false;
    };
    let Ok(sig) = secp256k1::schnorr::Signature::from_slice(signature) else {
        return false;
    };
    secp256k1::SECP256K1.verify_schnorr(&sig, &secp256k1::Message::from_digest(msg), &pk).is_ok()
}

/// Delay between a traced response's acceptance and the chain block whose hash seeds its draw.
pub const TRACE_SEED_DELAY_DAA: u64 = 100;
/// Domain separator of the trace draw seed.
pub const TRACE_DRAW_DOMAIN: &[u8] = b"KeryxTraceDrawV1";
/// Window to open the next phase of an attention opening once its child is drawn.
pub const ATTN_PHASE_WINDOW_DAA: u64 = 3_000;
/// Domain separator of the child drawn after an attention phase.
pub const ATTN_CHILD_DOMAIN: &[u8] = b"KeryxAttnChildV1";

/// A signed response carrying a trace commitment, accepted in a chain block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TraceResponseEvent {
    pub response_txid: [u8; 32],
    pub request_hash: [u8; 32],
    pub responder: Hash,
    pub escrow_pubkey: [u8; 32],
    pub trace_root: [u8; 32],
    pub n_prompt: u32,
    pub n_pos: u32,
    /// Plaintext length of the sealed answer.
    pub answer_len: u32,
    pub stage: u8,
    pub tokens: bool,
}

/// An opening accepted in a chain block whose signature verifies under `escrow_pubkey`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TraceOpeningEvent {
    pub response_txid: [u8; 32],
    pub escrow_pubkey: [u8; 32],
    pub body: Vec<u8>,
}

/// Trace inputs of one chain block.
pub struct TraceInputs<'a> {
    pub chain_hash: Hash,
    pub responses: &'a [TraceResponseEvent],
    pub openings: &'a [TraceOpeningEvent],
    /// Audit rules of a tier for a response accepted at a DAA score.
    pub tier: &'a dyn Fn(u8, u64) -> Option<crate::config::params::TraceRules>,
    /// Plaintext prompt length of the block's accepted requests, by request hash.
    pub prompt_lens: &'a [([u8; 32], u32)],
    /// Whether the shape rule is active at this block.
    pub shape: bool,
    /// Whether a trace root is admitted once at this block.
    pub unique_root: bool,
    /// Whether the sanction rule is active at this block: a missing opening strikes without the
    /// interval guard, and a response under the audit rule resets the streak at its opening.
    pub sanction: bool,
    /// Whether the context rule is active at this block: a silent cohort member is excused on a
    /// prompt that may exceed the tier context floor unless a member's traced response proved
    /// it fit.
    pub context: bool,
    /// Whether the draw rule is active at this block: the reward of a routed request goes to a
    /// draw among the members credited within its draw window.
    pub draw: bool,
    /// Whether the v3 window is active at this block: an audit armed here is judged on the
    /// measured work of the request and the queue of each member.
    pub window_v3: bool,
    /// Whether a response accepted more than once by this block is judged once by the shape
    /// and root rules.
    pub dedup: bool,
    /// Whether an opening naming its seed block is judged only against an audit drawn from that
    /// block, and passed over otherwise.
    pub seed_bind: bool,
}

const TRACE_TIER_UNKNOWN: u8 = u8::MAX;

/// The committed shape of a traced response whose request is not pending yet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TraceShape {
    n_prompt: u32,
    n_pos: u32,
    /// Text length of the sealed answer.
    answer_len: u32,
}

fn trace_shape_ok(req: &PendingRequest, shape: &TraceShape, prefix_offset: Option<u32>) -> bool {
    keryx_verifiable_inference::shape::shape_ok(req.prompt_len, shape.answer_len, req.max_tokens, shape.n_prompt, shape.n_pos)
        && prefix_offset.is_none_or(|o| keryx_verifiable_inference::prefix::fits(shape.n_prompt, o))
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TraceAudit {
    request_hash: [u8; 32],
    responder: Hash,
    escrow_pubkey: [u8; 32],
    trace_root: [u8; 32],
    n_prompt: u32,
    n_pos: u32,
    accepted_daa: u64,
    tier: u8,
    max_tokens: u32,
    identities: Vec<Hash>,
    seed: Option<[u8; 32]>,
    deadline_daa: u64,
    /// Pieces already proven when the audit needs several ([`PIECE_TRACE`], [`PIECE_ATTN`],
    /// [`PIECE_RECUR`]).
    pieces: u8,
    /// Progress of an attention opening of several phases.
    attn: Option<AttnProgress>,
}

/// Accepted phases of an attention opening still short of its last.
#[derive(Clone, Debug, PartialEq, Eq)]
struct AttnProgress {
    /// Child drawn after each accepted phase.
    path: Vec<u64>,
    /// Partials root and child count the last accepted phase committed.
    root: [u8; 32],
    n: u64,
    /// DAA of the chain block that accepted the last phase.
    accepted_daa: u64,
    /// Whether the child of the last accepted phase is drawn.
    drawn: bool,
    /// Draw weights of the `n` children.
    weights: Vec<u16>,
}

impl AttnProgress {
    /// Path of the phase the audit awaits, `None` until the child of the last accepted phase is
    /// drawn.
    fn awaited(&self) -> Option<&[u64]> {
        self.drawn.then_some(&self.path[..])
    }
}

/// Seed of the child drawn after attention phase `phase` of a response.
pub fn attn_child_seed(chain_hash: &Hash, response_txid: &[u8; 32], phase: u8) -> [u8; 32] {
    let mut hasher = blake2b_simd::Params::new().hash_length(32).to_state();
    hasher.update(ATTN_CHILD_DOMAIN);
    hasher.update(&chain_hash.as_bytes());
    hasher.update(response_txid);
    hasher.update(&[phase]);
    let mut seed = [0u8; 32];
    seed.copy_from_slice(hasher.finalize().as_bytes());
    seed
}

/// The trace opening of a several-piece audit is proven.
const PIECE_TRACE: u8 = 1;
/// The attention opening of a several-piece audit is proven.
const PIECE_ATTN: u8 = 2;
/// The recurrent opening of a several-piece audit is proven.
const PIECE_RECUR: u8 = 4;
const PIECE_ALL: u8 = PIECE_TRACE | PIECE_ATTN | PIECE_RECUR;

/// Pieces an audit under `rules` needs.
fn required_pieces(rules: Option<&crate::config::params::TraceRules>) -> u8 {
    match rules {
        Some(r) if r.attention => {
            PIECE_TRACE | PIECE_ATTN | if r.chain.as_ref().is_some_and(|c| c.n_recur > 0) { PIECE_RECUR } else { 0 }
        }
        _ => PIECE_TRACE,
    }
}

/// Draw seed of a traced response.
pub fn trace_seed(chain_hash: &Hash, response_txid: &[u8; 32]) -> [u8; 32] {
    let mut hasher = blake2b_simd::Params::new().hash_length(32).to_state();
    hasher.update(TRACE_DRAW_DOMAIN);
    hasher.update(&chain_hash.as_bytes());
    hasher.update(response_txid);
    let mut seed = [0u8; 32];
    seed.copy_from_slice(hasher.finalize().as_bytes());
    seed
}

/// Checks that an opening body proves every drawn point of `audit` under `seed`, the prefix
/// point included when the rules carry a prefix context; `Err` names the failing check.
fn trace_opening_valid(
    rules: &crate::config::params::TraceRules,
    audit: &TraceAudit,
    seed: &[u8; 32],
    txid: &[u8; 32],
    body: &[u8],
) -> Result<(), &'static str> {
    use keryx_verifiable_inference::{chain, draw::POINTS, opening::Opening, prefix, verify};
    let Some(opening) = Opening::decode(body) else { return Err("malformed body") };
    if &opening.response_txid != txid {
        return Err("response id mismatch");
    }
    if opening.points.len() != POINTS as usize {
        return Err("point count");
    }
    let mut resp = verify::ResponseCtx { trace_root: audit.trace_root, n_prompt: audit.n_prompt, n_pos: audit.n_pos };
    if let Some(cc) = &rules.chain {
        let (Some(pc), Some(pp), Some(rp)) = (&rules.prefix, &opening.prefix, &opening.resid) else { return Err("stage-2 part missing") };
        match (rules.attention, &opening.roots) {
            (true, Some((pos_root, seq_root))) => {
                if keryx_verifiable_inference::seq::trace_root(pos_root, seq_root) != audit.trace_root {
                    return Err("trace root split");
                }
                resp.trace_root = *pos_root;
            }
            (true, None) => return Err("sequence root missing"),
            (false, Some(_)) => return Err("sequence root before its rule"),
            (false, None) => {}
        }
        if opening.links.len() != opening.points.len() || opening.experts.len() != opening.points.len() {
            return Err("link count");
        }
        for (i, (p, (tok, link))) in opening.points.iter().zip(&opening.links).enumerate() {
            let ids = opening.experts.get(i).and_then(Option::as_ref);
            if chain::verify_chain_point(&rules.ctx, cc, &resp, seed, i as u32, p, *tok, link, ids).is_err() {
                return Err(if i == 0 { "point 0" } else { "point 1" });
            }
        }
        chain::verify_chain_prefix_point(&rules.ctx, cc, pc, &resp, seed, &audit.request_hash, pp).map_err(|_| "prefix point")?;
        return chain::verify_resid_point(&rules.ctx, cc, &resp, seed, rp).map_err(|_| "residual point");
    }
    if opening.resid.is_some() {
        return Err("stage-2 opening before its rule");
    }
    for (i, p) in opening.points.iter().enumerate() {
        if verify::verify_point(&rules.ctx, &resp, seed, i as u32, p).is_err() {
            return Err(if i == 0 { "point 0" } else { "point 1" });
        }
    }
    match (&rules.prefix, &opening.prefix) {
        (None, _) => Ok(()),
        (Some(_), None) => Err("prefix point missing"),
        (Some(pc), Some(p)) => {
            prefix::verify_prefix_point(&rules.ctx, pc, &resp, seed, &audit.request_hash, p).map(|_| ()).map_err(|_| "prefix point")
        }
    }
}

/// The piece an opening body, seeded or not, proves.
fn body_piece(body: &[u8]) -> u8 {
    use keryx_verifiable_inference::opening::{VERSION_ATTN, VERSION_RECUR, inner_body};
    match inner_body(body).and_then(|b| b.first().copied()) {
        Some(VERSION_ATTN) => PIECE_ATTN,
        Some(VERSION_RECUR) => PIECE_RECUR,
        _ => PIECE_TRACE,
    }
}

/// Checks that a recurrent opening body proves the recurrent point of `audit` under `seed`.
fn recur_opening_valid(
    rules: &crate::config::params::TraceRules,
    audit: &TraceAudit,
    seed: &[u8; 32],
    txid: &[u8; 32],
    body: &[u8],
) -> Result<(), &'static str> {
    use keryx_verifiable_inference::{opening::{RecurOpening, inner_body}, recur, verify};
    let op = inner_body(body).as_deref().and_then(RecurOpening::decode).ok_or("malformed recurrent body")?;
    if &op.response_txid != txid {
        return Err("response id mismatch");
    }
    let cc = rules.chain.as_ref().ok_or("recurrence without stage-2 rules")?;
    let resp = verify::ResponseCtx { trace_root: audit.trace_root, n_prompt: audit.n_prompt, n_pos: audit.n_pos };
    recur::verify_recur_point(&rules.ctx, cc, &resp, seed, &op).map_err(|_| "recurrent point")
}

/// Checks that an attention opening body proves the phase `audit` awaits of its attention point
/// under `seed`.
fn attn_opening_valid(
    rules: &crate::config::params::TraceRules,
    audit: &TraceAudit,
    seed: &[u8; 32],
    txid: &[u8; 32],
    body: &[u8],
) -> Result<keryx_verifiable_inference::chain::AttnStep, &'static str> {
    use keryx_verifiable_inference::{chain, opening::{AttnOpening, inner_body}, verify};
    let op = inner_body(body).as_deref().and_then(AttnOpening::decode).ok_or("malformed attention body")?;
    if &op.response_txid != txid {
        return Err("response id mismatch");
    }
    let (path, root) = match &audit.attn {
        Some(p) => (&p.path[..], Some(&p.root)),
        None => (&[][..], None),
    };
    #[cfg(test)]
    if let Some(r) = ATTN_ORACLE.with(|o| o.borrow().as_ref().map(|f| f(path, root, &op))) {
        return r;
    }
    let cc = rules.chain.as_ref().ok_or("attention without stage-2 rules")?;
    let resp = verify::ResponseCtx { trace_root: audit.trace_root, n_prompt: audit.n_prompt, n_pos: audit.n_pos };
    chain::verify_attn_phase(&rules.ctx, cc, &resp, seed, path, root, &op).map_err(|_| "attention point")
}

/// Verdict of an attention phase from its path, the previous partials root and the opening.
#[cfg(test)]
type AttnOracle = Box<dyn Fn(&[u64], Option<&[u8; 32]>, &keryx_verifiable_inference::opening::AttnOpening) -> Result<keryx_verifiable_inference::chain::AttnStep, &'static str>>;

#[cfg(test)]
thread_local! {
    /// Stands in for the attention verifier in the ledger tests of the calling thread.
    static ATTN_ORACLE: std::cell::RefCell<Option<AttnOracle>> = const { std::cell::RefCell::new(None) };
}

/// Domain separator of the trace opening signature.
pub const OPENING_SIG_DOMAIN: &[u8] = b"KeryxTraceOpeningV1";

/// The message an opening is signed over: the domain-tagged blake2b-256 of its body.
pub fn opening_signature_message(body: &[u8]) -> [u8; 32] {
    let mut hasher = blake2b_simd::Params::new().hash_length(32).to_state();
    hasher.update(OPENING_SIG_DOMAIN);
    hasher.update(body);
    let mut msg = [0u8; 32];
    msg.copy_from_slice(hasher.finalize().as_bytes());
    msg
}

/// Verifies an opening signature: schnorr by `escrow_pubkey` over [`opening_signature_message`].
pub fn verify_opening_signature(escrow_pubkey: &[u8; 32], signature: &[u8; 64], body: &[u8]) -> bool {
    let msg = opening_signature_message(body);
    let Ok(pk) = secp256k1::XOnlyPublicKey::from_slice(escrow_pubkey) else {
        return false;
    };
    let Ok(sig) = secp256k1::schnorr::Signature::from_slice(signature) else {
        return false;
    };
    secp256k1::SECP256K1.verify_schnorr(&sig, &secp256k1::Message::from_digest(msg), &pk).is_ok()
}

/// Deterministically select one index in `0..n` from a 32-byte seed (a block hash chosen after
/// the request). Assigns the single responsible miner for an inference request from the eligible
/// (recently-active tier) set. `None` for an empty set.
pub fn assign_index(seed: &[u8; 32], n: usize) -> Option<usize> {
    if n == 0 {
        return None;
    }
    let x = u64::from_le_bytes(seed[..8].try_into().unwrap());
    Some((x % n as u64) as usize)
}

/// Number of escrow claims burned at the first consecutive missed assignment.
pub const STRIKE_1_BURN_CLAIMS: u32 = 5;

/// Minimum DAA between two consecutive strikes on the same miner (~1 h). Any further miss inside
/// this interval is a no-op: it neither escalates the strike count nor burns escrow. The guard-rail
/// that separates "offline for ten minutes" (or a request flood) from "refusing to serve for
/// hours" — a genuinely dead miner still escalates, one strike per interval, reaching suspension in
/// ~3 intervals.
pub const SERVICE_STRIKE_INTERVAL_DAA: u64 = 36_000;

/// Production suspension applied at the third consecutive strike (24 h). Enforced finality-deep
/// (like escrow burns): a miner suspended at miss daa `T` has his miner cut burned — his blocks
/// are still valid and merged, he is simply paid nothing — over
/// `[T + finality, T + finality + SERVICE_SUSPENSION_DAA]`, so the full 24 h bites after the
/// reorg-immune finality delay. (Blocks are NOT rejected: rejecting a suspended producer's blue
/// would strand the honest miner that merges it — see `coinbase.rs` zero-output guard.)
pub const SERVICE_SUSPENSION_DAA: u64 = 864_000;

/// DAA window, ending at the assignment seed block, in which a miner must have produced a proven
/// tier block to be service-eligible. ~10 minutes at 10 BPS.
pub const SERVICE_ELIGIBILITY_WINDOW_DAA: u64 = 6_000;

/// Service-eligibility window once `service_bond_v2_activation` is live (~5 minutes at 10 BPS):
/// an identity leaves every cohort this long after its last proven block.
pub const SERVICE_ELIGIBILITY_WINDOW_DAA_V2: u64 = 3_000;

/// Widest window a service-providers query may cover (~1 hour at 10 BPS).
pub const MAX_SERVICE_PROVIDERS_WINDOW_DAA: u64 = 36_000;

/// Cohort members a private request may leave out and still be served: miners that entered the
/// cohort between sealing and arming.
pub fn private_cohort_tolerance(cohort_len: usize) -> usize {
    cohort_len.div_ceil(4).max(3)
}

/// Largest cohort a private request is sealed to in full; a larger one is sealed to its
/// [`private_target_cohort`]. Must equal the envelope's recipient cap.
pub const PRIVATE_COHORT_TARGET: usize = 128;

/// Seed of a private request's target cohort: the outpoint its first input spends
/// (`txid ‖ index u32 LE`).
pub type PrivateCohortSeed = [u8; 36];

/// The [`PrivateCohortSeed`] of a request whose first input spends `outpoint`.
pub fn private_cohort_seed(outpoint: &crate::tx::TransactionOutpoint) -> PrivateCohortSeed {
    let mut seed = [0u8; 36];
    seed[..32].copy_from_slice(&outpoint.transaction_id.as_bytes());
    seed[32..].copy_from_slice(&outpoint.index.to_le_bytes());
    seed
}

/// Rank of escrow key `key` in the target cohort drawn from `seed`; lower ranks are kept.
pub fn private_cohort_rank(seed: &PrivateCohortSeed, key: &[u8; 32]) -> [u8; 32] {
    let mut state = blake2b_simd::Params::new().hash_length(32).to_state();
    state.update(b"KeryxPrivateCohortV1");
    state.update(seed);
    state.update(key);
    let mut rank = [0u8; 32];
    rank.copy_from_slice(state.finalize().as_bytes());
    rank
}

/// The escrow keys a private request seeded with `seed` must cover among `cohort`: all of them
/// up to [`PRIVATE_COHORT_TARGET`], else the [`PRIVATE_COHORT_TARGET`] of lowest rank. Sorted,
/// deduplicated.
pub fn private_target_cohort(seed: &PrivateCohortSeed, cohort: &[Hash]) -> Vec<Hash> {
    let mut keys: Vec<Hash> = cohort.to_vec();
    keys.sort_unstable();
    keys.dedup();
    if keys.len() > PRIVATE_COHORT_TARGET {
        keys.sort_by_cached_key(|k| (private_cohort_rank(seed, &k.as_bytes()), *k));
        keys.truncate(PRIVATE_COHORT_TARGET);
        keys.sort_unstable();
    }
    keys
}

/// Standing evaluation lag AND probation length (~14 h at 10 BPS): an identity is in standing at
/// POV `p` iff, looking only at events with daa ≤ `p − LAG`, it has been sighted (first certified
/// block) and its strike count reads zero. The lag is finality + the ledger horizon, so every
/// event the evaluation may read is finality-flushed on every node long before any POV that reads
/// it — standing is a pure function of reorg-immune data, identical live, on catch-up and on
/// refold. The probation is the lag itself: a fresh identity earns the floor tier rate for ~14 h.
pub const SERVICE_STANDING_LAG_DAA: u64 = 504_000;

/// Fixed part of the service window: assignment detection, propagation and inclusion. 30 s.
pub const SERVICE_WINDOW_BASE_DAA: u64 = 300;

/// Window base once `service_bond_v2_activation` is live (~5 minutes at 10 BPS): every cohort
/// member gets at least this long to see and serve a request, a restarting rig included.
pub const SERVICE_WINDOW_BASE_DAA_V2: u64 = 3_000;

/// Hard cap on an AiRequest's `max_tokens` at/after the service-bond gate — matches the web
/// interface maximum. Bounds the service window any single request can demand and rejects
/// nonsense values.
pub const AI_REQUEST_MAX_TOKENS_CAP: u32 = 4_096;

/// DAA window an assigned miner has, from his assignment seed block, for the request to be served
/// before it counts as a miss: a fixed base plus a per-requested-token allowance floored at the
/// generation speed of the tier's model class (measured medians ~7-10 tok/s, ×2 margin).
pub fn service_window_daa(tier: u8, max_tokens: u32) -> u64 {
    service_window_daa_at(tier, max_tokens, false)
}

/// [`service_window_daa`] with the base selected by whether `service_bond_v2_activation` is
/// active at the daa the audit arms.
pub fn service_window_daa_at(tier: u8, max_tokens: u32, v2: bool) -> u64 {
    let per_token_daa: u64 = match tier {
        0..=2 => 2, // 0.2 s/token — 5 tok/s floor
        3 => 3,     // 0.3 s/token
        _ => 4,     // 0.4 s/token — 2.5 tok/s floor
    };
    let base = if v2 { SERVICE_WINDOW_BASE_DAA_V2 } else { SERVICE_WINDOW_BASE_DAA };
    base + max_tokens.min(AI_REQUEST_MAX_TOKENS_CAP) as u64 * per_token_daa
}

/// Window base once `service_window_v3_activation` is live.
pub const SERVICE_WINDOW_BASE_DAA_V3: u64 = 150;

/// Per-tier generation allowance of the v3 window, in hundredths of a DAA per generated token.
pub const SERVICE_WINDOW_RATE_V3: [u64; 5] = [90, 180, 85, 70, 35];

/// Requests queued ahead of a request at one responder beyond which no lateness is imputed to
/// it on that request. Must equal the miner's request queue size.
pub const SERVICE_QUEUE_CAP: usize = 64;

/// Serving work of one request under the v3 window, in DAA: `n_prompt` prompt tokens (or
/// sealed bytes, which bound them) and `generated` generated tokens.
pub fn service_work_daa(tier: u8, n_prompt: u32, generated: u32) -> u64 {
    let rate = SERVICE_WINDOW_RATE_V3[(tier as usize).min(SERVICE_WINDOW_RATE_V3.len() - 1)];
    2 * (n_prompt as u64).div_ceil(100) + (rate * generated.min(AI_REQUEST_MAX_TOKENS_CAP) as u64).div_ceil(100)
}

/// Service window of one request under `service_window_v3_activation`, queue excluded.
pub fn service_window_daa_v3(tier: u8, n_prompt: u32, generated: u32) -> u64 {
    SERVICE_WINDOW_BASE_DAA_V3 + service_work_daa(tier, n_prompt, generated)
}

/// Length of the reward draw of a request: half its v3 window at full budget.
pub fn reward_draw_window_daa(tier: u8, prompt_len: u32, max_tokens: u32) -> u64 {
    service_window_daa_v3(tier, prompt_len, max_tokens).div_ceil(2)
}

/// Domain separator of the reward draw seed.
pub const REWARD_DRAW_DOMAIN: &[u8] = b"KeryxRewardDrawV1";

/// Seed of the reward draw of `request_hash` closed by the chain block `chain_hash`.
pub fn reward_draw_seed(chain_hash: &Hash, request_hash: &[u8; 32]) -> [u8; 32] {
    let mut hasher = blake2b_simd::Params::new().hash_length(32).to_state();
    hasher.update(REWARD_DRAW_DOMAIN);
    hasher.update(&chain_hash.as_bytes());
    hasher.update(request_hash);
    let mut seed = [0u8; 32];
    seed.copy_from_slice(hasher.finalize().as_bytes());
    seed
}

fn upper_median(mut values: Vec<u32>) -> Option<u32> {
    values.sort_unstable();
    values.get(values.len() / 2).copied()
}

/// DAA horizon beyond which per-request ledger state is forgotten: pending requests expire and
/// vault claims drop out. This bounds the request/vault memory a fold must warm up; strike
/// counts are NOT horizon-bound — they persist in the strike log and only reset on a served
/// response or an executed suspension.
pub const SERVICE_LEDGER_HORIZON_DAA: u64 = 72_000;

/// How long a served request hash stays remembered so a later acceptance of the same hash cannot
/// arm a second audit (~24 h at 10 BPS). A hash can repeat because request identity is the payload
/// digest: the same prompt, sent twice, is the same request. The second audit would be
/// unanswerable — responders dedupe on that hash and the identical response is already on chain —
/// so the whole cohort would be struck for a request nobody can serve.
pub const SERVICE_AUDITED_MEMORY_DAA: u64 = 864_000;

/// Ceiling, above a syncee's pruning point, for service rows it cannot re-derive itself: such
/// events come from requests accepted at most an eligibility window above the pruning point
/// (deeper cohort windows cross unretained history), and a request stops generating events one
/// ledger horizon after acceptance. The service-state transfer ships every flushed row at or
/// below `pruning_point + SERVICE_STATE_HANDOFF_DAA`; the syncee re-derives only above it.
pub const SERVICE_STATE_HANDOFF_DAA: u64 = SERVICE_ELIGIBILITY_WINDOW_DAA + SERVICE_LEDGER_HORIZON_DAA;

/// Hard cap on inference-reward mint outputs per coinbase (H8 routing). Wins beyond it, in
/// canonical `(event daa, request hash)` order, stay burned — bounds the coinbase output count.
pub const MAX_REWARD_MINTS_PER_BLOCK: usize = 64;

/// How long an authenticated response is held when its request has not been accepted yet. An
/// AiResponse carries no inputs, so nothing orders its acceptance against its request's: the
/// selected chain can accept it first. Sized on the consensus merge depth
/// (`BPS * MERGE_DEPTH_DURATION`) — past it a block can no longer be merged, so its transactions
/// can no longer be accepted and the inversion is out of reach.
pub const SERVICE_EARLY_RESPONSE_HORIZON_DAA: u64 = 36_000;

/// Ceiling on distinct request hashes held by the above. Eviction takes the oldest first and only
/// returns an inversion to the behaviour it had before it was handled — it never creates a miss.
const MAX_EARLY_RESPONSE_HASHES: usize = 4_096;

/// Penalty applied to a miner for a missed service assignment, by consecutive-miss count.
/// A successful serve resets the count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ServicePenalty {
    None,
    /// Burn `n` escrow claims (n blocks' worth of the miner's accumulated escrow).
    BurnClaims(u32),
    /// Burn the miner's entire still-locked pending escrow.
    SlashAllPending,
    /// Suspend the miner: his miner cut is burned (blocks stay valid and merged, he is paid
    /// nothing) for [`SERVICE_SUSPENSION_DAA`] once the suspension is finality-deep. Also drains
    /// any escrow re-accumulated past the drain.
    Suspend,
}

/// Penalty for the `consecutive_misses`-th consecutive miss (0 = served, no penalty).
pub fn strike_penalty(consecutive_misses: u32, established: bool) -> ServicePenalty {
    strike_penalty_at(consecutive_misses, established, false)
}

/// [`strike_penalty`] with the first-miss step selected by whether
/// `service_bond_v2_activation` is active at the daa the strike lands.
/// Before the gate, an identity without standing skips the gentle first step;
/// at and after it, the first miss costs the same for every identity.
pub fn strike_penalty_at(consecutive_misses: u32, established: bool, v2: bool) -> ServicePenalty {
    match (consecutive_misses, established || v2) {
        (0, _) => ServicePenalty::None,
        (1, true) => ServicePenalty::BurnClaims(STRIKE_1_BURN_CLAIMS),
        (1, false) | (2, _) => ServicePenalty::SlashAllPending,
        _ => ServicePenalty::Suspend,
    }
}

/// Fold one assignment outcome into a miner's consecutive-miss counter: a miss increments,
/// a served assignment resets to 0. The reset is what keeps an honest miner's occasional
/// miss from ever escalating.
pub fn update_strikes(current: u32, missed: bool) -> u32 {
    if missed {
        current + 1
    } else {
        0
    }
}

/// Eligible responsible-miner set for a request targeting `target_tier`'s model: the distinct
/// `(identity, delegated escrow key)` pairs that produced at least one `target_tier` block in
/// the recent window. Sorted and deduped so every node derives the identical set. `recent` is
/// `(identity, tier, escrow key)` for the recently-active window (order irrelevant).
pub fn eligible_pairs(recent: &[(Hash, u8, Hash)], target_tier: u8) -> Vec<(Hash, Hash)> {
    let mut set: Vec<(Hash, Hash)> = recent.iter().filter(|(_, t, _)| *t == target_tier).map(|(m, _, e)| (*m, *e)).collect();
    set.sort_unstable();
    set.dedup();
    set
}

/// Draws the responsible miner from `eligible`, skipping `excluded` (miners that already missed
/// this request). Falls back to the full set when exclusion empties it, so a lone producer stays
/// drawable — his repeat misses are what escalates.
pub fn draw_assignment(eligible: &[Hash], excluded: &[Hash], seed: &[u8; 32]) -> Option<Hash> {
    let filtered: Vec<Hash> = eligible.iter().copied().filter(|m| !excluded.contains(m)).collect();
    let pool = if filtered.is_empty() { eligible } else { &filtered };
    assign_index(seed, pool.len()).map(|i| pool[i])
}

/// Point-in-time view of the service-bond enforcement state, for RPC monitoring.
#[derive(Clone, Debug, Default)]
pub struct ServiceStrikesSnapshot {
    pub virtual_daa_score: u64,
    /// Live strike entries: (miner, consecutive misses, last strike daa).
    pub strikes: Vec<(Hash, u32, u64)>,
    /// Production suspensions: (miner, suspended-until daa).
    pub suspended: Vec<(Hash, u64)>,
    /// Misses awaiting finality: (miner, miss daa, consecutive misses, burned claims, burned
    /// sompi, missed request hash).
    pub pending_burns: Vec<(Hash, u64, u32, u32, u64, [u8; 32])>,
    /// Strikes taken over the whole retained log: (miner, count). Unlike `strikes`, this never
    /// resets — a served response and an executed suspension both clear the live counter, so it
    /// is the only figure that answers "how often has this miner failed". Display only.
    pub lifetime_strikes: Vec<(Hash, u32)>,
}

/// One service-eligible identity of a tier at the current sink: the miner (payout-SPK key), the
/// escrow key it announces — what a private request is sealed to — and the tier it proved.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ServiceProvider {
    pub tier: u8,
    pub model_id: [u8; 32],
    pub identity: Hash,
    pub escrow_pubkey: [u8; 32],
}

/// Display snapshot of the currently eligible responders per tier (`getServiceProviders`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ServiceProvidersSnapshot {
    pub virtual_daa_score: u64,
    pub providers: Vec<ServiceProvider>,
}

/// One escrow claim of a miner: a CSV-locked coinbase escrow output he can claim after the lock,
/// unless burned by a service penalty first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EscrowClaim {
    pub outpoint: crate::tx::TransactionOutpoint,
    pub value: u64,
    pub daa: u64,
}

/// A missed service assignment: the request's window closed with no accepted response. `burned`
/// lists the concrete escrow claims the penalty takes, newest first (the freshest claims have the
/// most CSV lock left, so they are the ones guaranteed still unclaimed).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServiceMiss {
    pub request_hash: [u8; 32],
    pub miner: Hash,
    pub consecutive_misses: u32,
    /// `Suspend` flags a production suspension for `miner`; the enforcement layer turns it into a
    /// finality-deep, reorg-immune deadline from the miss's own daa.
    pub penalty: ServicePenalty,
    pub burned: Vec<EscrowClaim>,
    /// The miss is an audited response left unopened, not a silent cohort member.
    pub opening_missing: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PendingRequest {
    tier: u8,
    max_tokens: u32,
    accepted_daa: u64,
    /// Sompi locked in the request's keyless reward vault (0 for pre-routing-gate requests,
    /// which lock their reward to a designated escrow key instead).
    reward: u64,
    /// First identity credited with an accepted response — the one the reward mints to.
    winner: Option<Hash>,
    audit: Option<Audit>,
    /// Escrow keys that answered between this request's acceptance and its arming — the one-block
    /// gap where no audit exists yet to credit them. Drained into `responded` when the audit arms.
    /// Bounded by the AiResponses of a single chain block's acceptance data, then cleared.
    early_responders: Vec<Hash>,
    /// Private inference: the escrow keys (sorted, deduplicated) the requester sealed the prompt
    /// to; empty for a public request. A private request's audit cohort is the eligible tier set
    /// restricted to these keys, so only a named responder is obligated to answer, credited when
    /// it does, and paid the vault.
    recipients: Vec<Hash>,
    /// Private inference: the seed of the target cohort the recipients must cover; zero for a
    /// public request.
    cohort_seed: PrivateCohortSeed,
    /// Plaintext prompt length; zero before the shape rule.
    prompt_len: u32,
    /// A cohort member's well-formed traced response proved the prompt fits the tier context floor.
    fit_proven: bool,
    /// Open reward draw under the draw rule.
    draw: Option<RewardDraw>,
}

impl PendingRequest {
    /// `(prompt tokens, generated tokens)` the request is judged on: the upper medians of its
    /// traced member answers, else the sealed prompt length and the full budget.
    fn reference_shape(&self) -> (u32, u32) {
        let shapes: Vec<(u32, u32)> =
            self.audit.as_ref().and_then(|a| a.v3.as_ref()).map(|w| w.answers.iter().filter_map(|(_, _, s)| *s).collect()).unwrap_or_default();
        let p = upper_median(shapes.iter().map(|(p, _)| *p).collect()).unwrap_or(self.prompt_len);
        let g = upper_median(shapes.iter().map(|(_, g)| *g).collect()).unwrap_or(self.max_tokens);
        (p, g)
    }

    fn reference_work(&self) -> u64 {
        let (p, g) = self.reference_shape();
        service_work_daa(self.tier, p, g)
    }
}

/// A reward draw: opened by the first credited response, closed `W` DAA later.
#[derive(Clone, Debug, PartialEq, Eq)]
struct RewardDraw {
    open_daa: u64,
    close_daa: u64,
    /// Credited identities and the daa they were credited at, credit order.
    candidates: Vec<(Hash, u64)>,
}

/// One cohort audit: every declared miner of the request's tier must respond before the window
/// closes; the silent ones are struck when it does. `cohort` holds the sorted identities
/// (payout-SPK keys); `delegations` maps each delegated escrow key back to its identities, so
/// a response signed by the hot escrow key credits the right identity.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Audit {
    cohort: Vec<Hash>,
    delegations: Vec<(Hash, Hash)>,
    responded: Vec<Hash>,
    window_end_daa: u64,
    /// Present when armed under the v3 window.
    v3: Option<QueuedWindow>,
}

/// Per-member state of an audit armed under the v3 window.
#[derive(Clone, Debug, PartialEq, Eq)]
struct QueuedWindow {
    arm_daa: u64,
    /// Cohort order: work queued ahead of the request (DAA), and whether the queue exceeded
    /// [`SERVICE_QUEUE_CAP`].
    queue: Vec<(u64, bool)>,
    /// Credited members.
    answers: Vec<Answer>,
}

/// A credited member: identity, credit daa, traced `(n_prompt, generated)`.
type Answer = (Hash, u64, Option<(u32, u32)>);

/// One miner's strike state: consecutive-miss count and the daa of the last actual strike.
/// Also the persisted strike-log record: `{0, 0}` marks a served-response reset, `{0, daa}` an
/// executed suspension (the daa keeps the rate-limit window armed and re-derives the suspension
/// deadline: `daa + finality + SERVICE_SUSPENSION_DAA`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StrikeEntry {
    pub count: u32,
    pub last_daa: u64,
}

impl MemSizeEstimator for StrikeEntry {
    fn estimate_mem_bytes(&self) -> usize {
        size_of::<Self>()
    }
}

/// Persisted record of a finality-deep inference-reward win: winner identity, vaulted amount,
/// event daa, and the payout script the mint goes to (`None` leaves the amount burned).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RewardEntry {
    pub winner: Hash,
    pub amount: u64,
    pub daa: u64,
    pub spk: Option<crate::tx::ScriptPublicKey>,
}

impl MemSizeEstimator for RewardEntry {
    fn estimate_mem_bytes(&self) -> usize {
        size_of::<Self>()
    }
}

/// An inference reward won: the first identity credited with an accepted response to a
/// routing-gated request. Minted to `spk` by a coinbase once finality-deep; `None` (identity
/// with no known payout script) leaves the vaulted amount burned.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServiceReward {
    pub request_hash: [u8; 32],
    pub winner: Hash,
    pub amount: u64,
    pub spk: Option<crate::tx::ScriptPublicKey>,
}

/// The strike-affecting outcomes of folding one chain block: the misses it closes and the miners
/// whose streak a served response reset. Both are persisted to the strike log once finality-deep.
#[derive(Clone, Debug, Default)]
pub struct FoldOutcome {
    pub misses: Vec<ServiceMiss>,
    /// Openings judged by this fold.
    pub verdicts: Vec<TraceVerdict>,
    /// Inference rewards decided by this fold (first accepted response of a routed request).
    pub rewards: Vec<ServiceReward>,
    /// Served-response streak resets as (identity, preserved last-strike daa). The preserved daa
    /// keeps the strike rate-limit armed across a serve; 0 before the v2 gate.
    pub resets: Vec<(Hash, u64)>,
    /// Identities sighted (first certified block) in this fold, new relative to the baseline.
    pub sightings: Vec<Hash>,
    /// Claims dropped by the burnable-window purge in this fold, in pop order — the undo log
    /// needs them to reverse the block.
    pub expired: Vec<(Hash, EscrowClaim)>,
    /// Silent cohort members left unstruck under the context rule, as (request hash, identity).
    pub excused: Vec<([u8; 32], Hash)>,
    /// Members whose answer came past their v3 deadline, as (request hash, identity).
    pub late: Vec<([u8; 32], Hash)>,
}

/// The judgement of one opening: `cause` is `None` when it closed the audit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TraceVerdict {
    pub response_txid: [u8; 32],
    pub request_hash: [u8; 32],
    pub responder: Hash,
    pub cause: Option<&'static str>,
}

/// Reorg-restore state of everything but the vault (see [`ServiceLedger::light_snapshot`]).
#[derive(Clone, Debug)]
pub struct LightSnapshot {
    pending: std::collections::BTreeMap<[u8; 32], PendingRequest>,
    early_responses: std::collections::BTreeMap<[u8; 32], (u64, Vec<Hash>)>,
    strikes: std::collections::BTreeMap<Hash, StrikeEntry>,
    first_seen: std::collections::BTreeMap<Hash, u64>,
    base: std::sync::Arc<std::collections::BTreeMap<Hash, StrikeEntry>>,
    first_seen_base: std::sync::Arc<std::collections::BTreeMap<Hash, u64>>,
    producer_spk: std::collections::BTreeMap<Hash, crate::tx::ScriptPublicKey>,
    audited: std::collections::BTreeMap<[u8; 32], u64>,
    trace_audits: std::collections::BTreeMap<[u8; 32], TraceAudit>,
    early_shapes: std::collections::BTreeMap<([u8; 32], Hash), TraceShape>,
    trace_roots: std::collections::BTreeMap<[u8; 32], u64>,
}

/// Request-lifecycle ledger, folded once per selected-chain block. Deterministic: state is a pure
/// function of the accepted requests/responses stream, the cohort function and the persisted
/// baseline, with BTreeMap ordering. The strike map is a delta over `base` — the persisted
/// records at the fold anchor — and only the request/vault memory is window-bounded.
#[derive(Clone, Debug, Default)]
pub struct ServiceLedger {
    pending: std::collections::BTreeMap<[u8; 32], PendingRequest>,
    /// Authenticated responses whose request is not pending yet, by request hash: the daa they
    /// were first seen at, and the escrow keys that signed them. Drained into the request's
    /// `early_responders` the moment it is accepted.
    early_responses: std::collections::BTreeMap<[u8; 32], (u64, Vec<Hash>)>,
    /// Strike delta folded since the anchor; overlays `base` (delta wins, `count: 0` tombstones).
    strikes: std::collections::BTreeMap<Hash, StrikeEntry>,
    /// Per-miner still-locked escrow claims, chain order (newest at the back).
    vault: std::collections::BTreeMap<Hash, std::collections::VecDeque<EscrowClaim>>,
    /// Finality-anchored strike baseline (the persisted records at the fold anchor). Shared so
    /// per-chain-block snapshots stay cheap to clone.
    base: std::sync::Arc<std::collections::BTreeMap<Hash, StrikeEntry>>,
    /// First-sighting delta folded since the anchor, over the persisted baseline. Only used to
    /// deduplicate sighting events — standing itself reads the lagged persisted state.
    first_seen: std::collections::BTreeMap<Hash, u64>,
    /// Finality-anchored first-sighting baseline.
    first_seen_base: std::sync::Arc<std::collections::BTreeMap<Hash, u64>>,
    /// Activation daa of the v2 service windows; `None` = not armed. Configuration, not folded
    /// state — installed on every fold entry, untouched by snapshots.
    window_v2_activation_daa: Option<u64>,
    /// Burnable window override (test networks); `None` = `SERVICE_BURNABLE_WINDOW_DAA`.
    burnable_window_daa: Option<u64>,
    /// Activation daa of vault reward routing (requests accepted at or after it mint their
    /// reward to the first accepted responder); `None` = not armed. Configuration, like above.
    reward_routing_daa: Option<u64>,
    /// Request hashes already admitted, with the daa they were admitted at. Gated by
    /// `reward_routing_daa`: past the gate a repeat acceptance of a remembered hash is ignored
    /// instead of arming a second, unanswerable audit.
    audited: std::collections::BTreeMap<[u8; 32], u64>,
    /// Latest coinbase payout script seen per identity, folded from chain-block producers —
    /// resolves a reward winner to a mintable script.
    producer_spk: std::collections::BTreeMap<Hash, crate::tx::ScriptPublicKey>,
    /// Traced responses awaiting their draw or their opening, by response transaction id.
    trace_audits: std::collections::BTreeMap<[u8; 32], TraceAudit>,
    /// Shapes of traced responses whose request was not pending yet, by `(request hash,
    /// responder)`; judged when the request is admitted, dropped with `early_responses`.
    early_shapes: std::collections::BTreeMap<([u8; 32], Hash), TraceShape>,
    /// Trace roots admitted under the root rule, with the daa of their response; a repeat
    /// within [`SERVICE_AUDITED_MEMORY_DAA`] is not an answer.
    trace_roots: std::collections::BTreeMap<[u8; 32], u64>,
}

impl ServiceLedger {
    /// Folds one selected-chain block into the ledger and returns the misses it closes.
    ///
    /// `requests` are the block's accepted AiRequests as `(request_hash, tier, max_tokens)`;
    /// `responses` its accepted AiResponses as `(request_hash, verified responder)`; `escrows` the
    /// escrow claims this block's coinbase creates, keyed by producing miner; `cohort` resolves a
    /// tier to its full declared-miner set at this block. Every cohort member must respond before
    /// the request's window closes; requests are admitted before their block's responses and
    /// responses are applied before expiries, so an answer landing in the opening block or in the
    /// closing one counts just the same.
    pub fn on_chain_block(
        &mut self,
        daa: u64,
        requests: &[([u8; 32], u8, u32)],
        responses: &[([u8; 32], Option<Hash>)],
        escrows: &[(Hash, EscrowClaim)],
        is_established: impl Fn(&Hash) -> bool,
        cohort: impl FnMut(u8) -> Vec<(Hash, Hash)>,
    ) -> FoldOutcome {
        self.fold_inner(daa, requests, &[], &[], responses, escrows, &[], None, None, is_established, cohort)
    }

    /// [`Self::on_chain_block`] with the reward-routing inputs: per-request vaulted amounts and
    /// the block's `(identity, payout script)` producers.
    #[allow(clippy::too_many_arguments)]
    pub fn on_chain_block_with_rewards(
        &mut self,
        daa: u64,
        requests: &[([u8; 32], u8, u32)],
        request_rewards: &[([u8; 32], u64)],
        responses: &[([u8; 32], Option<Hash>)],
        escrows: &[(Hash, EscrowClaim)],
        producers: &[(Hash, crate::tx::ScriptPublicKey)],
        is_established: impl Fn(&Hash) -> bool,
        cohort: impl FnMut(u8) -> Vec<(Hash, Hash)>,
    ) -> FoldOutcome {
        self.fold_inner(daa, requests, request_rewards, &[], responses, escrows, producers, None, None, is_established, cohort)
    }

    /// [`Self::on_chain_block_with_rewards`] with the private-inference input: per-request
    /// recipient sets (sorted escrow keys). A request absent here, or with an empty set, is public.
    #[allow(clippy::too_many_arguments)]
    pub fn on_chain_block_with_recipients(
        &mut self,
        daa: u64,
        requests: &[([u8; 32], u8, u32)],
        request_rewards: &[([u8; 32], u64)],
        request_recipients: &[([u8; 32], PrivateCohortSeed, Vec<Hash>)],
        responses: &[([u8; 32], Option<Hash>)],
        escrows: &[(Hash, EscrowClaim)],
        producers: &[(Hash, crate::tx::ScriptPublicKey)],
        is_established: impl Fn(&Hash) -> bool,
        cohort: impl FnMut(u8) -> Vec<(Hash, Hash)>,
    ) -> FoldOutcome {
        self.fold_inner(daa, requests, request_rewards, request_recipients, responses, escrows, producers, None, None, is_established, cohort)
    }

    /// [`Self::on_chain_block_with_recipients`] with the trace inputs of the block.
    #[allow(clippy::too_many_arguments)]
    pub fn on_chain_block_with_traces(
        &mut self,
        daa: u64,
        requests: &[([u8; 32], u8, u32)],
        request_rewards: &[([u8; 32], u64)],
        request_recipients: &[([u8; 32], PrivateCohortSeed, Vec<Hash>)],
        responses: &[([u8; 32], Option<Hash>)],
        escrows: &[(Hash, EscrowClaim)],
        producers: &[(Hash, crate::tx::ScriptPublicKey)],
        traces: &TraceInputs,
        is_established: impl Fn(&Hash) -> bool,
        cohort: impl FnMut(u8) -> Vec<(Hash, Hash)>,
    ) -> FoldOutcome {
        self.fold_inner(daa, requests, request_rewards, request_recipients, responses, escrows, producers, None, Some(traces), is_established, cohort)
    }

    /// Folds a chain block whose strike events are already persisted (daa at or below the store
    /// frontier): request and vault memory evolve normally, escrow claims already burned are
    /// dropped through `is_burned` (the exact historical truth from the burn store), and the
    /// strike map is left untouched — the baseline already carries those events.
    pub fn on_chain_block_warmup(
        &mut self,
        daa: u64,
        requests: &[([u8; 32], u8, u32)],
        responses: &[([u8; 32], Option<Hash>)],
        escrows: &[(Hash, EscrowClaim)],
        is_burned: &dyn Fn(&crate::tx::TransactionOutpoint) -> bool,
        cohort: impl FnMut(u8) -> Vec<(Hash, Hash)>,
    ) {
        self.fold_inner(daa, requests, &[], &[], responses, escrows, &[], Some(is_burned), None, |_| true, cohort);
    }

    /// [`Self::on_chain_block_warmup`] with the reward-routing inputs.
    #[allow(clippy::too_many_arguments)]
    pub fn on_chain_block_warmup_with_rewards(
        &mut self,
        daa: u64,
        requests: &[([u8; 32], u8, u32)],
        request_rewards: &[([u8; 32], u64)],
        responses: &[([u8; 32], Option<Hash>)],
        escrows: &[(Hash, EscrowClaim)],
        producers: &[(Hash, crate::tx::ScriptPublicKey)],
        is_burned: &dyn Fn(&crate::tx::TransactionOutpoint) -> bool,
        cohort: impl FnMut(u8) -> Vec<(Hash, Hash)>,
    ) {
        self.fold_inner(daa, requests, request_rewards, &[], responses, escrows, producers, Some(is_burned), None, |_| true, cohort);
    }

    /// [`Self::on_chain_block_warmup_with_rewards`] with the private-inference input.
    #[allow(clippy::too_many_arguments)]
    pub fn on_chain_block_warmup_with_recipients(
        &mut self,
        daa: u64,
        requests: &[([u8; 32], u8, u32)],
        request_rewards: &[([u8; 32], u64)],
        request_recipients: &[([u8; 32], PrivateCohortSeed, Vec<Hash>)],
        responses: &[([u8; 32], Option<Hash>)],
        escrows: &[(Hash, EscrowClaim)],
        producers: &[(Hash, crate::tx::ScriptPublicKey)],
        is_burned: &dyn Fn(&crate::tx::TransactionOutpoint) -> bool,
        cohort: impl FnMut(u8) -> Vec<(Hash, Hash)>,
    ) {
        self.fold_inner(daa, requests, request_rewards, request_recipients, responses, escrows, producers, Some(is_burned), None, |_| true, cohort);
    }

    /// [`Self::on_chain_block_warmup_with_recipients`] with the trace inputs of the block.
    #[allow(clippy::too_many_arguments)]
    pub fn on_chain_block_warmup_with_traces(
        &mut self,
        daa: u64,
        requests: &[([u8; 32], u8, u32)],
        request_rewards: &[([u8; 32], u64)],
        request_recipients: &[([u8; 32], PrivateCohortSeed, Vec<Hash>)],
        responses: &[([u8; 32], Option<Hash>)],
        escrows: &[(Hash, EscrowClaim)],
        producers: &[(Hash, crate::tx::ScriptPublicKey)],
        traces: &TraceInputs,
        is_burned: &dyn Fn(&crate::tx::TransactionOutpoint) -> bool,
        cohort: impl FnMut(u8) -> Vec<(Hash, Hash)>,
    ) {
        self.fold_inner(daa, requests, request_rewards, request_recipients, responses, escrows, producers, Some(is_burned), Some(traces), |_| true, cohort);
    }

    #[allow(clippy::too_many_arguments)]
    fn fold_inner(
        &mut self,
        daa: u64,
        requests: &[([u8; 32], u8, u32)],
        request_rewards: &[([u8; 32], u64)],
        request_recipients: &[([u8; 32], PrivateCohortSeed, Vec<Hash>)],
        responses: &[([u8; 32], Option<Hash>)],
        escrows: &[(Hash, EscrowClaim)],
        producers: &[(Hash, crate::tx::ScriptPublicKey)],
        warmup_burned: Option<&dyn Fn(&crate::tx::TransactionOutpoint) -> bool>,
        traces: Option<&TraceInputs>,
        is_established: impl Fn(&Hash) -> bool,
        mut cohort: impl FnMut(u8) -> Vec<(Hash, Hash)>,
    ) -> FoldOutcome {
        let warmup = warmup_burned.is_some();
        for (identity, spk) in producers {
            self.producer_spk.insert(*identity, spk.clone());
        }
        let mut sightings: Vec<Hash> = Vec::new();
        let mut expired: Vec<(Hash, EscrowClaim)> = Vec::new();
        for (miner, claim) in escrows {
            // First certified block of this identity: report it once for persistence. A warmup
            // fold skips — its sightings are already in the baseline.
            if !warmup && !self.first_seen.contains_key(miner) && !self.first_seen_base.contains_key(miner) {
                self.first_seen.insert(*miner, claim.daa);
                sightings.push(*miner);
            }
            if warmup_burned.is_some_and(|is_burned| is_burned(&claim.outpoint)) {
                continue;
            }
            self.vault.entry(*miner).or_default().push_back(*claim);
        }
        let burnable = self.burnable_window_daa.unwrap_or(SERVICE_BURNABLE_WINDOW_DAA);
        for (miner, claims) in self.vault.iter_mut() {
            while claims.front().is_some_and(|c| c.daa + burnable <= daa) {
                expired.push((*miner, claims.pop_front().unwrap()));
            }
        }
        self.vault.retain(|_, claims| !claims.is_empty());

        let mut outcome = FoldOutcome { sightings, expired, ..Default::default() };

        self.pending.retain(|_, r| r.accepted_daa + SERVICE_LEDGER_HORIZON_DAA > daa);
        self.early_responses.retain(|_, (seen, _)| *seen + SERVICE_EARLY_RESPONSE_HORIZON_DAA > daa);
        let early_hashes: std::collections::BTreeSet<[u8; 32]> = self.early_responses.keys().copied().collect();
        let keep_shapes = traces.is_some_and(|tr| tr.context || tr.window_v3);
        let pending = &self.pending;
        self.early_shapes
            .retain(|(rh, _), _| early_hashes.contains(rh) || (keep_shapes && pending.get(rh).is_some_and(|r| r.audit.is_none())));

        // This block's own requests are admitted before its responses are folded: a response
        // accepted in the same chain block as its request must find the request pending, or it is
        // dropped and its author struck for an answer he did give. Arming still waits for the next
        // block (`daa > accepted_daa` below), so the cohort is unchanged.
        let gated = self.reward_routing_daa.is_some_and(|a| daa >= a);
        let draw_rule = traces.is_some_and(|tr| tr.draw);
        let window_v3 = traces.is_some_and(|tr| tr.window_v3);
        if gated {
            self.audited.retain(|_, seen| *seen + SERVICE_AUDITED_MEMORY_DAA > daa);
        }
        for (rh, tier, max_tokens) in requests {
            // Past the gate, a hash already admitted never opens a second audit: the responders
            // that served it cannot serve it again (same dedup key, same response transaction),
            // so the repeat would strike a whole cohort for an unanswerable assignment.
            if gated && self.audited.contains_key(rh) {
                continue;
            }
            if gated {
                self.audited.insert(*rh, daa);
            }
            let mut early = self.early_responses.remove(rh).map(|(_, keys)| keys).unwrap_or_default();
            let prompt_len =
                traces.filter(|tr| tr.shape).and_then(|tr| tr.prompt_lens.iter().find(|(h, _)| h == rh).map(|(_, l)| *l)).unwrap_or(0);
            // A re-accepted hash (identical payload resubmitted) must not reset the running
            // audit: overwriting would re-arm the window and push the miss out forever.
            let entry = self.pending.entry(*rh).or_insert(PendingRequest {
                tier: *tier,
                max_tokens: *max_tokens,
                accepted_daa: daa,
                reward: request_rewards.iter().find(|(h, _)| h == rh).map(|(_, v)| *v).unwrap_or(0),
                winner: None,
                audit: None,
                early_responders: Vec::new(),
                recipients: request_recipients.iter().find(|(h, _, _)| h == rh).map(|(_, _, r)| r.clone()).unwrap_or_default(),
                cohort_seed: request_recipients.iter().find(|(h, _, _)| h == rh).map(|(_, s, _)| *s).unwrap_or([0u8; 36]),
                prompt_len,
                fit_proven: false,
                draw: None,
            });
            if let Some(tr) = traces.filter(|tr| tr.shape) {
                let prefix_offset = (tr.tier)(entry.tier, daa).and_then(|r| r.prefix.map(|p| p.offset));
                let shapes: Vec<(Hash, TraceShape)> = self
                    .early_shapes
                    .range((*rh, Hash::from_bytes([0u8; 32]))..=(*rh, Hash::from_bytes([0xffu8; 32])))
                    .map(|((_, key), shape)| (*key, *shape))
                    .collect();
                early.retain(|key| shapes.iter().find(|(k, _)| k == key).is_none_or(|(_, shape)| trace_shape_ok(entry, shape, prefix_offset)));
                // Under the context rule and the v3 window the well-formed early shapes wait for
                // the arming, where cohort membership is known.
                let keep = tr.context || tr.window_v3;
                self.early_shapes.retain(|(h, key), _| h != rh || (keep && early.contains(key)));
            }
            // Answers that beat their own request to acceptance.
            if entry.audit.is_none() {
                for key in early {
                    if !entry.early_responders.contains(&key) {
                        entry.early_responders.push(key);
                    }
                }
            }
        }

        // A traced response whose commitment does not fit the sealed sizes of its request and
        // answer is not an answer: neither credited nor audited.
        let mut misshaped: Vec<([u8; 32], Hash)> = Vec::new();
        // An input-less response can be accepted more than once by one chain block.
        let mut seen_txids = std::collections::BTreeSet::new();
        let trace_responses: Vec<&TraceResponseEvent> = traces
            .map(|tr| tr.responses.iter().filter(|ev| !tr.dedup || seen_txids.insert(ev.response_txid)).collect())
            .unwrap_or_default();
        if let Some(tr) = traces.filter(|tr| tr.shape) {
            for ev in trace_responses.iter() {
                let answer_len = if ev.tokens {
                    keryx_verifiable_inference::answer::text_len(ev.answer_len, ev.n_pos.saturating_sub(ev.n_prompt)).unwrap_or(u32::MAX)
                } else {
                    ev.answer_len
                };
                let shape = TraceShape { n_prompt: ev.n_prompt, n_pos: ev.n_pos, answer_len };
                match self.pending.get(&ev.request_hash) {
                    Some(req) => {
                        let rules = (tr.tier)(req.tier, daa);
                        let prefix_offset = rules.as_ref().and_then(|r| r.prefix.as_ref().map(|p| p.offset));
                        let stage_ok = rules.as_ref().is_none_or(|r| r.chain.is_none() || (ev.stage == keryx_verifiable_inference::chain::STAGE_CHAIN && ev.tokens));
                        if !stage_ok || !trace_shape_ok(req, &shape, prefix_offset) {
                            misshaped.push((ev.request_hash, ev.responder));
                        } else if (tr.context || tr.window_v3) && req.audit.is_none() {
                            self.early_shapes.insert((ev.request_hash, ev.responder), shape);
                        }
                    }
                    None => {
                        self.early_shapes.insert((ev.request_hash, ev.responder), shape);
                    }
                }
            }
        }

        if traces.is_some_and(|tr| tr.unique_root) {
            self.trace_roots.retain(|_, seen| *seen + SERVICE_AUDITED_MEMORY_DAA > daa);
            for ev in trace_responses.iter() {
                if misshaped.contains(&(ev.request_hash, ev.responder)) {
                    continue;
                }
                if let std::collections::btree_map::Entry::Vacant(e) = self.trace_roots.entry(ev.trace_root) {
                    e.insert(daa);
                } else {
                    misshaped.push((ev.request_hash, ev.responder));
                }
            }
        }

        // An authenticated response (signed by a delegated escrow key) marks every cohort
        // identity that delegated to it as having served this audit and resets their streak.
        // Anyone else's response is ignored by the ledger. A response landing before the audit
        // arms is parked on the request and credited when it does.
        for (rh, responder) in responses {
            let Some(r) = responder else { continue };
            if misshaped.contains(&(*rh, *r)) {
                continue;
            }
            let mut served: Vec<Hash> = Vec::new();
            let req_tier: u8;
            {
                let Some(req) = self.pending.get_mut(rh) else {
                    let entry = self.early_responses.entry(*rh).or_insert((daa, Vec::new()));
                    if !entry.1.contains(r) {
                        entry.1.push(*r);
                    }
                    continue;
                };
                req_tier = req.tier;
                let req_max_tokens = req.max_tokens;
                let Some(audit) = req.audit.as_mut() else {
                    if !req.early_responders.contains(r) {
                        req.early_responders.push(*r);
                    }
                    continue;
                };
                let matched: Vec<Hash> = audit.delegations.iter().filter(|(e, _)| e == r).map(|(_, id)| *id).collect();
                for identity in matched {
                    if audit.cohort.binary_search(&identity).is_ok() && !audit.responded.contains(&identity) {
                        audit.responded.push(identity);
                        served.push(identity);
                    }
                }
                if let Some(w) = audit.v3.as_mut() {
                    let shape = traces
                        .and_then(|tr| tr.responses.iter().find(|ev| ev.request_hash == *rh && ev.responder == *r))
                        .map(|ev| (ev.n_prompt, ev.n_pos.saturating_sub(ev.n_prompt)));
                    for identity in served.iter() {
                        w.answers.push((*identity, daa, shape));
                    }
                }
                if !served.is_empty()
                    && traces.is_some_and(|tr| {
                        tr.context
                            && tr.responses.iter().any(|ev| {
                                ev.request_hash == *rh && ev.responder == *r && crate::config::params::prompt_fits_floor(req_tier, ev.n_prompt, req_max_tokens)
                            })
                    })
                {
                    req.fit_proven = true;
                }
            }
            self.credit_reward(rh, &served, daa, draw_rule, warmup, &mut outcome);
            // Under the sanction rule a traced response that will be audited resets the streak at
            // its accepted opening, not here.
            let reset_at_opening = self.reset_waits_for_opening(traces, rh, r, req_tier, daa);
            for identity in served {
                // A warmup fold leaves strikes alone: the baseline already carries this reset.
                if !warmup && !reset_at_opening && self.strike_state(&identity).is_some_and(|e| e.count > 0) {
                    let preserved = self.reset_preserved_last_daa(&identity, daa);
                    self.strikes.insert(identity, StrikeEntry { count: 0, last_daa: preserved });
                    outcome.resets.push((identity, preserved));
                }
            }
        }

        while self.early_responses.len() > MAX_EARLY_RESPONSE_HASHES {
            let victim =
                self.early_responses.iter().min_by_key(|(hash, (seen, _))| (*seen, **hash)).map(|(hash, _)| *hash).unwrap();
            self.early_responses.remove(&victim);
            self.early_shapes.retain(|(rh, _), _| *rh != victim);
        }

        if let Some(tr) = traces {
            self.fold_traces(daa, tr, &misshaped, warmup, &is_established, &mut outcome);
            let closing: Vec<[u8; 32]> =
                self.pending.iter().filter(|(_, r)| r.draw.as_ref().is_some_and(|d| daa >= d.close_daa)).map(|(h, _)| *h).collect();
            for rh in closing {
                self.close_draw(&rh, &tr.chain_hash, warmup, &mut outcome);
            }
        }

        let hashes: Vec<[u8; 32]> = self.pending.keys().copied().collect();
        let mut served_at_arm: Vec<Hash> = Vec::new();
        let mut armed_v3: Vec<[u8; 32]> = Vec::new();
        for rh in hashes {
            let req = self.pending.get(&rh).unwrap();
            match &req.audit {
                Some(a) if daa > a.window_end_daa => {
                    let audit = a.clone();
                    let late: Vec<Hash> = match &audit.v3 {
                        Some(w) => {
                            let (p, g) = req.reference_shape();
                            let allowed = w.arm_daa + service_window_daa_v3(req.tier, p, g);
                            w.answers
                                .iter()
                                .filter(|(id, at, _)| {
                                    audit
                                        .cohort
                                        .binary_search(id)
                                        .ok()
                                        .and_then(|i| w.queue.get(i))
                                        .is_some_and(|(queued, capped)| !capped && *at > allowed + queued)
                                })
                                .map(|(id, _, _)| *id)
                                .collect()
                        }
                        None => Vec::new(),
                    };
                    let excused = traces.is_some_and(|tr| tr.context)
                        && !req.fit_proven
                        && crate::config::params::safe_prompt_len(req.tier, req.max_tokens).is_some_and(|safe| req.prompt_len > safe);
                    self.drop_draw_candidates(&rh, &late);
                    if let Some(tr) = traces {
                        self.close_draw(&rh, &tr.chain_hash, warmup, &mut outcome);
                    }
                    for miner in audit.cohort.iter().filter(|m| !audit.responded.contains(m) || late.contains(m)) {
                        if warmup {
                            // The miss is already persisted: its burns came in through `is_burned`
                            // and its strike lives in the baseline.
                            continue;
                        }
                        let is_late = late.contains(miner);
                        if excused && !is_late {
                            outcome.excused.push((rh, *miner));
                            continue;
                        }
                        if is_late {
                            outcome.late.push((rh, *miner));
                        }
                        // Rate-limit: a strike lands at most once per interval. A miss inside the
                        // interval of the miner's last strike is a no-op — no escalation, no burn.
                        if self.strike_state(miner).is_some_and(|e| e.last_daa > 0 && daa < e.last_daa + SERVICE_STRIKE_INTERVAL_DAA)
                        {
                            continue;
                        }
                        let count = self.consecutive_misses(miner) + 1;
                        self.strikes.insert(*miner, StrikeEntry { count, last_daa: daa });
                        let penalty = strike_penalty_at(count, is_established(miner), self.v2_at(daa));
                        let burned = self.burn(miner, penalty);
                        if penalty == ServicePenalty::Suspend {
                            // The third strike executes the full drain and the suspension; the
                            // streak restarts so a later miss escalates from one instead of
                            // re-suspending forever. The strike daa keeps the rate-limit armed.
                            self.strikes.insert(*miner, StrikeEntry { count: 0, last_daa: daa });
                        }
                        outcome.misses.push(ServiceMiss {
                            request_hash: rh,
                            miner: *miner,
                            consecutive_misses: count,
                            penalty,
                            burned,
                            opening_missing: false,
                        });
                    }
                    self.pending.remove(&rh);
                }
                Some(_) => {}
                None if daa > req.accepted_daa => {
                    let mut set = cohort(req.tier);
                    // A private request is sealed to its target cohort. One that leaves out more
                    // than the tolerance is unservable. Otherwise it obligates (and pays) only
                    // the members it was sealed to: the others cannot read it, so they are
                    // neither audited for it nor able to claim its vault.
                    if !req.recipients.is_empty() {
                        let escrows: Vec<Hash> = set.iter().map(|(_, escrow)| *escrow).collect();
                        let target = private_target_cohort(&req.cohort_seed, &escrows);
                        let uncovered = target.iter().filter(|e| req.recipients.binary_search(e).is_err()).count();
                        if uncovered > private_cohort_tolerance(target.len()) {
                            set.clear();
                        } else {
                            set.retain(|(_, escrow)| req.recipients.binary_search(escrow).is_ok());
                        }
                    }
                    if set.is_empty() {
                        self.pending.remove(&rh);
                        self.early_shapes.retain(|(h, _), _| *h != rh);
                    } else {
                        let mut ids: Vec<Hash> = set.iter().map(|(id, _)| *id).collect();
                        ids.sort_unstable();
                        ids.dedup();
                        let mut delegations: Vec<(Hash, Hash)> = set.iter().map(|(id, esc)| (*esc, *id)).collect();
                        delegations.sort_unstable();
                        delegations.dedup();
                        let window =
                            service_window_daa_at(req.tier, req.max_tokens, self.v2_at(daa));
                        // Credit whoever answered before the audit existed: same membership test
                        // as the live path, so an early answer and a late one are worth the same.
                        let req_tier = req.tier;
                        let req_max_tokens = req.max_tokens;
                        let early = std::mem::take(&mut self.pending.get_mut(&rh).unwrap().early_responders);
                        let mut responded: Vec<Hash> = Vec::new();
                        let mut answers: Vec<Answer> = Vec::new();
                        let mut fit = false;
                        for key in early.iter() {
                            let waits = self.reset_waits_for_opening(traces, &rh, key, req_tier, daa);
                            let shape = self.early_shapes.get(&(rh, *key)).map(|s| (s.n_prompt, s.n_pos.saturating_sub(s.n_prompt)));
                            let mut member = false;
                            for identity in delegations.iter().filter(|(e, _)| e == key).map(|(_, id)| id) {
                                if ids.binary_search(identity).is_ok() && !responded.contains(identity) {
                                    responded.push(*identity);
                                    answers.push((*identity, daa, shape));
                                    member = true;
                                    if !waits {
                                        served_at_arm.push(*identity);
                                    }
                                }
                            }
                            if member && self.early_shapes.get(&(rh, *key)).is_some_and(|s| crate::config::params::prompt_fits_floor(req_tier, s.n_prompt, req_max_tokens)) {
                                fit = true;
                            }
                        }
                        self.early_shapes.retain(|(h, _), _| *h != rh);
                        let credited = responded.clone();
                        let req = self.pending.get_mut(&rh).unwrap();
                        if fit && traces.is_some_and(|tr| tr.context) {
                            req.fit_proven = true;
                        }
                        let v3 = window_v3.then(|| {
                            armed_v3.push(rh);
                            QueuedWindow { arm_daa: daa, queue: Vec::new(), answers }
                        });
                        req.audit = Some(Audit { cohort: ids, delegations, responded, window_end_daa: daa + window, v3 });
                        self.credit_reward(&rh, &credited, daa, draw_rule, warmup, &mut outcome);
                    }
                }
                None => {}
            }
        }

        for rh in armed_v3 {
            self.set_queued_window(&rh, daa);
        }

        for identity in served_at_arm {
            // A warmup fold leaves strikes alone: the baseline already carries this reset.
            if !warmup && self.strike_state(&identity).is_some_and(|e| e.count > 0) {
                let preserved = self.reset_preserved_last_daa(&identity, daa);
                self.strikes.insert(identity, StrikeEntry { count: 0, last_daa: preserved });
                outcome.resets.push((identity, preserved));
            }
        }

        outcome
    }

    /// Fills the tier, token budget and sanctioned identities of a trace audit from its request,
    /// once that request's audit is armed.
    fn resolve_trace_audit(&self, a: &mut TraceAudit) {
        if !a.identities.is_empty() {
            return;
        }
        let Some(req) = self.pending.get(&a.request_hash) else { return };
        let Some(audit) = req.audit.as_ref() else { return };
        a.tier = req.tier;
        a.max_tokens = req.max_tokens;
        a.identities = audit
            .delegations
            .iter()
            .filter(|(e, id)| *e == a.responder && audit.cohort.binary_search(id).is_ok())
            .map(|(_, id)| *id)
            .collect();
    }

    /// Records the block's traced responses, draws the ripe ones, judges the openings and closes
    /// the audits whose deadline passed.
    fn fold_traces(
        &mut self,
        daa: u64,
        tr: &TraceInputs,
        misshaped: &[([u8; 32], Hash)],
        warmup: bool,
        is_established: &dyn Fn(&Hash) -> bool,
        outcome: &mut FoldOutcome,
    ) {
        for ev in tr.responses {
            if self.trace_audits.contains_key(&ev.response_txid) || misshaped.contains(&(ev.request_hash, ev.responder)) {
                continue;
            }
            let mut a = TraceAudit {
                request_hash: ev.request_hash,
                responder: ev.responder,
                escrow_pubkey: ev.escrow_pubkey,
                trace_root: ev.trace_root,
                n_prompt: ev.n_prompt,
                n_pos: ev.n_pos,
                accepted_daa: daa,
                tier: TRACE_TIER_UNKNOWN,
                max_tokens: 0,
                identities: Vec::new(),
                seed: None,
                deadline_daa: 0,
                pieces: 0,
                attn: None,
            };
            self.resolve_trace_audit(&mut a);
            self.trace_audits.insert(ev.response_txid, a);
        }

        let ripe: Vec<[u8; 32]> = self
            .trace_audits
            .iter()
            .filter(|(_, a)| a.seed.is_none() && daa >= a.accepted_daa + TRACE_SEED_DELAY_DAA)
            .map(|(k, _)| *k)
            .collect();
        for txid in ripe {
            let mut a = self.trace_audits.remove(&txid).unwrap();
            self.resolve_trace_audit(&mut a);
            let seed = trace_seed(&tr.chain_hash, &txid);
            let rules = if a.tier != TRACE_TIER_UNKNOWN && !a.identities.is_empty() { (tr.tier)(a.tier, a.accepted_daa) } else { None };
            if rules.is_some_and(|r| r.audit_all || keryx_verifiable_inference::draw::audited(&seed)) {
                a.deadline_daa = daa + service_window_daa_at(a.tier, a.max_tokens, self.v2_at(daa));
                a.seed = Some(seed);
                self.trace_audits.insert(txid, a);
            }
        }

        for (txid, a) in self.trace_audits.iter_mut() {
            if let Some(p) = a.attn.as_mut().filter(|p| !p.drawn && daa >= p.accepted_daa + TRACE_SEED_DELAY_DAA) {
                let child = attn_child_seed(&tr.chain_hash, txid, p.path.len() as u8);
                p.path.push(keryx_verifiable_inference::attn_blocks::draw_weighted_child(&child, &p.weights));
                p.drawn = true;
            }
        }

        for op in tr.openings {
            let Some(a) = self.trace_audits.get(&op.response_txid) else { continue };
            let Some(seed) = a.seed else { continue };
            if op.escrow_pubkey != a.escrow_pubkey {
                continue;
            }
            if let Some(block) = keryx_verifiable_inference::opening::seed_block(&op.body)
                && (!tr.seed_bind || trace_seed(&Hash::from_bytes(block), &op.response_txid) != seed)
            {
                continue;
            }
            let rules = (tr.tier)(a.tier, a.accepted_daa);
            let required = required_pieces(rules.as_ref());
            let piece = body_piece(&op.body);
            if piece & required == 0 || a.pieces & piece != 0 {
                continue;
            }
            if piece == PIECE_ATTN
                && let Some(path) = keryx_verifiable_inference::opening::attn_path(&op.body)
                && Some(&path[..]) != a.attn.as_ref().map_or(Some(&[][..]), AttnProgress::awaited)
            {
                continue;
            }
            let mut a = self.trace_audits.remove(&op.response_txid).unwrap();
            let mut next = None;
            let verdict = match &rules {
                Some(r) if piece == PIECE_ATTN => attn_opening_valid(r, &a, &seed, &op.response_txid, &op.body).map(|step| {
                    if let keryx_verifiable_inference::chain::AttnStep::Next { root, n, weights } = step {
                        next = Some((root, n, weights));
                    }
                }),
                Some(r) if piece == PIECE_RECUR => recur_opening_valid(r, &a, &seed, &op.response_txid, &op.body),
                Some(r) => trace_opening_valid(r, &a, &seed, &op.response_txid, &op.body),
                None => Err("no rules"),
            };
            if let Some((root, n, weights)) = next {
                let path = a.attn.take().map(|p| p.path).unwrap_or_default();
                a.attn = Some(AttnProgress { path, root, n, accepted_daa: daa, drawn: false, weights });
                a.deadline_daa = a.deadline_daa.max(daa + TRACE_SEED_DELAY_DAA + ATTN_PHASE_WINDOW_DAA);
                self.trace_audits.insert(op.response_txid, a);
                continue;
            }
            if verdict.is_ok() && a.pieces | piece != required {
                a.pieces |= piece;
                if piece == PIECE_ATTN {
                    a.attn = None;
                }
                self.trace_audits.insert(op.response_txid, a);
                continue;
            }
            outcome.verdicts.push(TraceVerdict {
                response_txid: op.response_txid,
                request_hash: a.request_hash,
                responder: a.responder,
                cause: verdict.err(),
            });
            if verdict.is_ok() && tr.sanction && !warmup {
                for miner in a.identities.iter() {
                    if self.strike_state(miner).is_some_and(|e| e.count > 0) {
                        let preserved = self.reset_preserved_last_daa(miner, daa);
                        self.strikes.insert(*miner, StrikeEntry { count: 0, last_daa: preserved });
                        outcome.resets.push((*miner, preserved));
                    }
                }
            }
            if verdict.is_err() {
                self.drop_draw_candidates(&a.request_hash, &a.identities);
                for miner in a.identities.iter() {
                    if warmup {
                        continue;
                    }
                    let count = (self.consecutive_misses(miner) + 1).max(2);
                    let penalty = strike_penalty_at(count, is_established(miner), self.v2_at(daa));
                    self.strikes.insert(*miner, StrikeEntry { count, last_daa: daa });
                    let burned = self.burn(miner, penalty);
                    if penalty == ServicePenalty::Suspend {
                        self.strikes.insert(*miner, StrikeEntry { count: 0, last_daa: daa });
                    }
                    outcome.misses.push(ServiceMiss {
                        request_hash: a.request_hash,
                        miner: *miner,
                        consecutive_misses: count,
                        penalty,
                        burned,
                        opening_missing: false,
                    });
                }
            }
        }

        let late: Vec<[u8; 32]> =
            self.trace_audits.iter().filter(|(_, a)| a.seed.is_some() && daa > a.deadline_daa).map(|(k, _)| *k).collect();
        for txid in late {
            let a = self.trace_audits.remove(&txid).unwrap();
            self.drop_draw_candidates(&a.request_hash, &a.identities);
            for miner in a.identities.iter() {
                if warmup {
                    continue;
                }
                if !tr.sanction
                    && self.strike_state(miner).is_some_and(|e| e.last_daa > 0 && daa < e.last_daa + SERVICE_STRIKE_INTERVAL_DAA)
                {
                    continue;
                }
                let count = self.consecutive_misses(miner) + 1;
                self.strikes.insert(*miner, StrikeEntry { count, last_daa: daa });
                let penalty = strike_penalty_at(count, is_established(miner), self.v2_at(daa));
                let burned = self.burn(miner, penalty);
                if penalty == ServicePenalty::Suspend {
                    self.strikes.insert(*miner, StrikeEntry { count: 0, last_daa: daa });
                }
                outcome.misses.push(ServiceMiss {
                    request_hash: a.request_hash,
                    miner: *miner,
                    consecutive_misses: count,
                    penalty,
                    burned,
                    opening_missing: true,
                });
            }
        }

        self.trace_audits.retain(|_, a| a.seed.is_some() || a.accepted_daa + SERVICE_LEDGER_HORIZON_DAA > daa);
    }

    /// Last-strike daa a served-response reset carries forward, so a serve no longer disarms
    /// the strike rate-limit. 0 (the disarming legacy value) before the v2 gate.
    fn reset_preserved_last_daa(&self, identity: &Hash, daa: u64) -> u64 {
        if self.v2_at(daa) {
            self.strike_state(identity).map(|e| e.last_daa).unwrap_or(0)
        } else {
            0
        }
    }

    /// Takes the escrow claims a penalty burns out of the miner's vault: the `n` newest for
    /// `BurnClaims(n)`, everything still locked for `SlashAllPending` — and for `Suspend` too, so
    /// claims re-accumulated past the second strike stay burnable while the streak lasts.
    fn burn(&mut self, miner: &Hash, penalty: ServicePenalty) -> Vec<EscrowClaim> {
        let Some(claims) = self.vault.get_mut(miner) else {
            return Vec::new();
        };
        let take = match penalty {
            ServicePenalty::None => 0,
            ServicePenalty::BurnClaims(n) => (n as usize).min(claims.len()),
            ServicePenalty::SlashAllPending | ServicePenalty::Suspend => claims.len(),
        };
        let burned: Vec<EscrowClaim> = (0..take).map(|_| claims.pop_back().unwrap()).collect();
        if claims.is_empty() {
            self.vault.remove(miner);
        }
        burned
    }

    /// The miner's still-locked escrow claims, chain order (newest last).
    pub fn vault_claims(&self, miner: &Hash) -> Vec<EscrowClaim> {
        self.vault.get(miner).map(|claims| claims.iter().copied().collect()).unwrap_or_default()
    }

    /// The private-inference recipients of a pending request: `Some(sorted escrow keys)` while
    /// the request is pending and was sealed to named responders; `None` when it is unknown,
    /// expired or public. Mempool admission reads it to accept an inline response body only
    /// from a named responder of a live private request.
    /// Escrow key expected to sign the opening of a drawn response still awaiting it.
    pub fn awaited_opening_key(&self, response_txid: &[u8; 32]) -> Option<[u8; 32]> {
        self.trace_audits.get(response_txid).filter(|a| a.seed.is_some()).map(|a| a.escrow_pubkey)
    }

    /// Draw seed of a drawn response still awaiting its opening.
    pub fn awaited_opening_seed(&self, response_txid: &[u8; 32]) -> Option<[u8; 32]> {
        self.trace_audits.get(response_txid).and_then(|a| a.seed)
    }

    /// Path of the attention phase a drawn response awaits; `None` while a child is still to
    /// draw or when no attention opening is awaited.
    pub fn awaited_attention_path(&self, response_txid: &[u8; 32]) -> Option<Vec<u64>> {
        let a = self.trace_audits.get(response_txid).filter(|a| a.seed.is_some() && a.pieces & PIECE_ATTN == 0)?;
        match &a.attn {
            None => Some(Vec::new()),
            Some(p) => p.awaited().map(<[u64]>::to_vec),
        }
    }

    pub fn private_recipients(&self, request_hash: &[u8; 32]) -> Option<&[Hash]> {
        self.pending.get(request_hash).filter(|r| !r.recipients.is_empty()).map(|r| r.recipients.as_slice())
    }

    /// The armed audit cohort (sorted identities) of a pending request, if armed. Diagnostics.
    pub fn audit_cohort(&self, request_hash: &[u8; 32]) -> Option<Vec<Hash>> {
        self.pending.get(request_hash).and_then(|r| r.audit.as_ref()).map(|a| a.cohort.clone())
    }

    /// Installs the persisted strike baseline the delta folds over (the store content at the
    /// fold anchor).
    pub fn set_base(&mut self, base: std::sync::Arc<std::collections::BTreeMap<Hash, StrikeEntry>>) {
        self.base = base;
    }

    /// Installs the persisted first-sighting baseline (dedup source for sighting events).
    pub fn set_first_seen_base(&mut self, base: std::sync::Arc<std::collections::BTreeMap<Hash, u64>>) {
        self.first_seen_base = base;
    }

    /// Drops the folded strike and first-sighting deltas and installs the persisted baselines.
    pub fn set_persisted_baselines(
        &mut self,
        base: std::sync::Arc<std::collections::BTreeMap<Hash, StrikeEntry>>,
        first_seen_base: std::sync::Arc<std::collections::BTreeMap<Hash, u64>>,
    ) {
        self.strikes.clear();
        self.first_seen.clear();
        self.base = base;
        self.first_seen_base = first_seen_base;
    }

    /// Whether `service_bond_v2_activation` is live at `daa`.
    /// Under the sanction rule a traced response headed for an audit resets the streak at its
    /// accepted opening, not when it is credited.
    fn reset_waits_for_opening(&self, traces: Option<&TraceInputs>, request_hash: &[u8; 32], responder: &Hash, tier: u8, daa: u64) -> bool {
        let Some(tr) = traces.filter(|tr| tr.sanction) else { return false };
        let audited = |at: u64| (tr.tier)(tier, at).is_some_and(|rules| rules.audit_all);
        tr.responses.iter().any(|ev| ev.request_hash == *request_hash && ev.responder == *responder) && audited(daa)
            || self.trace_audits.values().any(|a| a.request_hash == *request_hash && a.responder == *responder && audited(a.accepted_daa))
    }

    fn v2_at(&self, daa: u64) -> bool {
        self.window_v2_activation_daa.is_some_and(|a| daa >= a)
    }

    /// Installs the `service_bond_v2_activation` daa the arming path reads.
    pub fn set_window_v2_activation(&mut self, daa: u64) {
        self.window_v2_activation_daa = Some(daa);
    }

    /// Installs the network's burnable window.
    pub fn set_burnable_window(&mut self, daa: u64) {
        self.burnable_window_daa = Some(daa);
    }

    /// Installs the reward-routing activation daa.
    pub fn set_reward_routing_activation(&mut self, daa: u64) {
        self.reward_routing_daa = Some(daa);
    }

    /// Awards `rh` to `identity` if the request is routed, unawarded, and carries a reward.
    /// A warmup fold sets the winner without emitting — the reward row is already persisted.
    fn maybe_award(&mut self, rh: &[u8; 32], identity: Hash, warmup: bool, outcome: &mut FoldOutcome) {
        if self.reward_routing_daa.is_none() {
            return;
        }
        let spk = self.producer_spk.get(&identity).cloned();
        let routed_from = self.reward_routing_daa.unwrap();
        let Some(req) = self.pending.get_mut(rh) else { return };
        if req.accepted_daa < routed_from || req.winner.is_some() || req.reward == 0 {
            return;
        }
        req.winner = Some(identity);
        if !warmup {
            outcome.rewards.push(ServiceReward { request_hash: *rh, winner: identity, amount: req.reward, spk });
        }
    }

    /// Credits `served` on the reward of `rh`. An open draw takes them as candidates until it
    /// closes; otherwise the first credit opens a draw under the draw rule, or wins outright.
    fn credit_reward(&mut self, rh: &[u8; 32], served: &[Hash], daa: u64, draw_rule: bool, warmup: bool, outcome: &mut FoldOutcome) {
        let Some(&first) = served.first() else { return };
        let Some(routed_from) = self.reward_routing_daa else { return };
        let Some(req) = self.pending.get_mut(rh) else { return };
        if let Some(draw) = req.draw.as_mut() {
            if daa <= draw.close_daa {
                for identity in served {
                    if !draw.candidates.iter().any(|(c, _)| c == identity) {
                        draw.candidates.push((*identity, daa));
                    }
                }
            }
            return;
        }
        if req.accepted_daa < routed_from || req.winner.is_some() || req.reward == 0 {
            return;
        }
        if draw_rule {
            let close_daa = daa + reward_draw_window_daa(req.tier, req.prompt_len, req.max_tokens);
            req.draw = Some(RewardDraw { open_daa: daa, close_daa, candidates: served.iter().map(|id| (*id, daa)).collect() });
            return;
        }
        self.maybe_award(rh, first, warmup, outcome);
    }

    /// Removes identities the audit sanctioned on `rh` from its open draw.
    fn drop_draw_candidates(&mut self, rh: &[u8; 32], identities: &[Hash]) {
        if identities.is_empty() {
            return;
        }
        if let Some(draw) = self.pending.get_mut(rh).and_then(|r| r.draw.as_mut()) {
            draw.candidates.retain(|(id, _)| !identities.contains(id));
        }
    }

    /// Draws the winner of `rh` among its remaining candidates, seeded by the closing chain
    /// block; with none left the reward stays burned.
    fn close_draw(&mut self, rh: &[u8; 32], chain_hash: &Hash, warmup: bool, outcome: &mut FoldOutcome) {
        let Some(req) = self.pending.get_mut(rh) else { return };
        let Some(draw) = req.draw.take() else { return };
        let mut candidates: Vec<Hash> = draw.candidates.iter().map(|(id, _)| *id).collect();
        candidates.sort_unstable();
        candidates.dedup();
        let Some(i) = assign_index(&reward_draw_seed(chain_hash, rh), candidates.len()) else {
            req.reward = 0;
            return;
        };
        let winner = candidates[i];
        req.winner = Some(winner);
        if !warmup {
            let spk = self.producer_spk.get(&winner).cloned();
            outcome.rewards.push(ServiceReward { request_hash: *rh, winner, amount: req.reward, spk });
        }
    }

    /// Sets the per-member queue and the deadline of an audit just armed under the v3 window:
    /// each member's queue is the work of the other armed requests it has not answered yet.
    fn set_queued_window(&mut self, rh: &[u8; 32], daa: u64) {
        let Some(req) = self.pending.get(rh) else { return };
        let Some(audit) = req.audit.as_ref() else { return };
        let queue: Vec<(u64, bool)> = audit
            .cohort
            .iter()
            .map(|m| {
                let ahead: Vec<u64> = self
                    .pending
                    .iter()
                    .filter(|(h, j)| {
                        *h != rh && j.audit.as_ref().is_some_and(|ja| ja.cohort.binary_search(m).is_ok() && !ja.responded.contains(m))
                    })
                    .map(|(_, j)| j.reference_work())
                    .collect();
                (ahead.iter().sum(), ahead.len() > SERVICE_QUEUE_CAP)
            })
            .collect();
        let own = service_window_daa_v3(req.tier, req.prompt_len, req.max_tokens);
        let horizon_end = req.accepted_daa + SERVICE_LEDGER_HORIZON_DAA - 2;
        let end = (daa + own + queue.iter().map(|(q, _)| *q).max().unwrap_or(0)).min(horizon_end);
        let audit = self.pending.get_mut(rh).unwrap().audit.as_mut().unwrap();
        audit.window_end_daa = end;
        if let Some(w) = audit.v3.as_mut() {
            w.queue = queue;
        }
    }

    /// The small reorg-restore state: everything but the vault (whose restore goes through the
    /// per-block undo log — cloning a full burnable window per chain block does not scale).
    pub fn light_snapshot(&self) -> LightSnapshot {
        LightSnapshot {
            pending: self.pending.clone(),
            early_responses: self.early_responses.clone(),
            strikes: self.strikes.clone(),
            first_seen: self.first_seen.clone(),
            base: self.base.clone(),
            first_seen_base: self.first_seen_base.clone(),
            producer_spk: self.producer_spk.clone(),
            audited: self.audited.clone(),
            trace_audits: self.trace_audits.clone(),
            early_shapes: self.early_shapes.clone(),
            trace_roots: self.trace_roots.clone(),
        }
    }

    /// Restores everything but the vault from a light snapshot.
    pub fn restore_light(&mut self, snap: &LightSnapshot) {
        self.pending = snap.pending.clone();
        self.early_responses = snap.early_responses.clone();
        self.strikes = snap.strikes.clone();
        self.first_seen = snap.first_seen.clone();
        self.base = snap.base.clone();
        self.first_seen_base = snap.first_seen_base.clone();
        self.producer_spk = snap.producer_spk.clone();
        self.audited = snap.audited.clone();
        self.trace_audits = snap.trace_audits.clone();
        self.early_shapes = snap.early_shapes.clone();
        self.trace_roots = snap.trace_roots.clone();
    }

    /// Reverses one folded block's vault mutations — the exact inverse of the fold's op order
    /// (adds at the back, then window-expiry pops at the front, then burn pops at the back):
    /// burns are re-pushed first (newest last), then expired claims re-enter at the front
    /// (oldest first), then the block's own adds pop off the back.
    pub fn undo_vault(&mut self, added: &[(Hash, EscrowClaim)], misses: &[ServiceMiss], expired: &[(Hash, EscrowClaim)]) {
        for miss in misses.iter().rev() {
            for claim in miss.burned.iter().rev() {
                self.vault.entry(miss.miner).or_default().push_back(*claim);
            }
        }
        for (miner, claim) in expired.iter().rev() {
            self.vault.entry(*miner).or_default().push_front(*claim);
        }
        for (miner, claim) in added.iter().rev() {
            let claims = self.vault.get_mut(miner).expect("undo of an add requires the deque to exist");
            let popped = claims.pop_back().expect("undo of an add requires the claim to be present");
            debug_assert_eq!(popped, *claim);
            if claims.is_empty() {
                self.vault.remove(miner);
            }
        }
    }

    /// The miner's strike state: the folded delta, falling back to the persisted baseline.
    fn strike_state(&self, miner: &Hash) -> Option<StrikeEntry> {
        self.strikes.get(miner).or_else(|| self.base.get(miner)).copied()
    }

    /// The miner's consecutive-miss count. Never expires by time: only a served response or an
    /// executed suspension resets it.
    pub fn consecutive_misses(&self, miner: &Hash) -> u32 {
        self.strike_state(miner).map(|e| e.count).unwrap_or(0)
    }

    /// Currently pending (accepted, unserved, unexpired) request count.
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    /// Live strike entries as (miner, count, last strike daa): the baseline and the delta merged
    /// (delta wins), zero counts excluded.
    pub fn strike_entries(&self) -> Vec<(Hash, u32, u64)> {
        let mut merged = self.base.as_ref().clone();
        for (m, e) in self.strikes.iter() {
            merged.insert(*m, *e);
        }
        merged.into_iter().filter(|(_, e)| e.count > 0).map(|(m, e)| (m, e.count, e.last_daa)).collect()
    }
}

/// Self-contained image of the ledger after one chain block: the effective state, baselines
/// folded in, so two nodes holding the same state encode the same bytes.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ServiceLedgerSnapshot {
    vault: std::collections::BTreeMap<Hash, Vec<EscrowClaim>>,
    pending: std::collections::BTreeMap<[u8; 32], PendingRequest>,
    early_responses: std::collections::BTreeMap<[u8; 32], (u64, Vec<Hash>)>,
    strikes: std::collections::BTreeMap<Hash, StrikeEntry>,
    first_seen: std::collections::BTreeMap<Hash, u64>,
    audited: std::collections::BTreeMap<[u8; 32], u64>,
    producer_spk: std::collections::BTreeMap<Hash, ScriptPublicKey>,
    /// `(chain daa, identity, tier, escrow key)` of the paid blues of the chain blocks inside
    /// the eligibility window ending at the snapshot block, chain order — what a cohort walk
    /// below the pruning point reads.
    pub recent_producers: Vec<(u64, Hash, u8, Hash)>,
    trace_audits: std::collections::BTreeMap<[u8; 32], TraceAudit>,
    early_shapes: std::collections::BTreeMap<([u8; 32], Hash), TraceShape>,
    trace_roots: std::collections::BTreeMap<[u8; 32], u64>,
}

const SNAPSHOT_ENCODING_VERSION: u8 = 1;
/// Encoding 1 plus a trailing private-recipient section (`[n: u32] n × [request_hash: 32]
/// [cohort seed: 36] [k: u32] [k × escrow key: 32]`, pending order, ascending keys). Emitted only when a pending
/// request carries recipients, so every snapshot without one keeps its historical bytes and
/// hash, and one byte form per state keeps the encoding canonical.
const SNAPSHOT_ENCODING_VERSION_PRIVATE: u8 = 2;
/// Private section (possibly empty) followed by the trace audits.
const SNAPSHOT_ENCODING_VERSION_TRACE: u8 = 3;
/// Private and trace sections (possibly empty) followed by the prompt lengths of the pending
/// requests that carry one (`[n: u32] n × [request_hash: 32] [prompt_len: u32]`, pending order)
/// and the early shapes (`[n: u32] n × [request_hash: 32] [responder: 32] [n_prompt: u32]
/// [n_pos: u32] [answer_len: u32]`, key order).
const SNAPSHOT_ENCODING_VERSION_SHAPE: u8 = 4;
/// Private, trace and shape sections (possibly empty) followed by the admitted trace roots
/// (`[n: u32] n × [root: 32] [daa: u64]`, key order).
const SNAPSHOT_ENCODING_VERSION_ROOT: u8 = 5;
/// Private, trace, shape and root sections (possibly empty) followed by the pending requests
/// whose prompt was proven to fit the tier context floor (`[n: u32] n × [request_hash: 32]`,
/// pending order).
const SNAPSHOT_ENCODING_VERSION_CONTEXT: u8 = 6;
/// Sections of version 6 (possibly empty) followed by the open reward draws (`[n: u32] n ×
/// [request_hash: 32] [open_daa: u64] [close_daa: u64] [k: u32] k × [identity: 32] [daa: u64]`,
/// pending order) and the v3 windows (`[n: u32] n × [request_hash: 32] [arm_daa: u64] [q: u32]
/// q × [queued: u64] [capped: u8] [a: u32] a × [identity: 32] [daa: u64] [traced: u8]
/// ([n_prompt: u32] [generated: u32] when traced)`, pending order, queue in cohort order).
const SNAPSHOT_ENCODING_VERSION_QUEUE: u8 = 7;
/// Sections of version 7 (possibly empty) followed by the trace audits holding one proven piece
/// of two (`[n: u32] n × [response_txid: 32] [piece: u8]`, key order).
const SNAPSHOT_ENCODING_VERSION_ATTN: u8 = 8;
/// Sections of version 8 (possibly empty) followed by the attention openings short of their
/// last phase (`[n: u32] n × [response_txid: 32] [k: u8] k × [child: u64] [root: 32]
/// [children: u64] [accepted_daa: u64] [drawn: u8] children × [weight: u16]`, key order).
const SNAPSHOT_ENCODING_VERSION_PHASES: u8 = 10;
/// Children an attention phase may commit.
const MAX_ATTN_CHILDREN: u64 = 1 << 16;

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.pos.checked_add(n).ok_or("length overflow")?;
        let slice = self.bytes.get(self.pos..end).ok_or("truncated snapshot")?;
        self.pos = end;
        Ok(slice)
    }
    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16, String> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn hash(&mut self) -> Result<Hash, String> {
        Ok(Hash::from_bytes(self.take(32)?.try_into().unwrap()))
    }
    fn key(&mut self) -> Result<[u8; 32], String> {
        Ok(self.take(32)?.try_into().unwrap())
    }
    fn hashes(&mut self) -> Result<Vec<Hash>, String> {
        let n = self.u32()? as usize;
        (0..n).map(|_| self.hash()).collect()
    }
}

fn put_uvarint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

fn read_uvarint(r: &mut Reader<'_>) -> Result<u64, String> {
    let mut v: u64 = 0;
    for shift in (0..64).step_by(7) {
        let b = r.u8()?;
        v |= ((b & 0x7f) as u64) << shift;
        if b & 0x80 == 0 {
            if b == 0 && shift > 0 {
                return Err("non-canonical varint".into());
            }
            return Ok(v);
        }
    }
    Err("varint overflow".into())
}

fn put_hashes(out: &mut Vec<u8>, hashes: &[Hash]) {
    out.extend_from_slice(&(hashes.len() as u32).to_le_bytes());
    for h in hashes {
        out.extend_from_slice(&h.as_bytes());
    }
}

impl ServiceLedgerSnapshot {
    /// Canonical byte form; the input of [`Self::hash`] and of the sync transfer.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        let private: Vec<(&[u8; 32], &PrivateCohortSeed, &Vec<Hash>)> = self
            .pending
            .iter()
            .filter(|(_, r)| !r.recipients.is_empty())
            .map(|(rh, r)| (rh, &r.cohort_seed, &r.recipients))
            .collect();
        let shaped = self.pending.values().any(|r| r.prompt_len > 0) || !self.early_shapes.is_empty();
        let fit: Vec<&[u8; 32]> = self.pending.iter().filter(|(_, r)| r.fit_proven).map(|(rh, _)| rh).collect();
        let drawn: Vec<(&[u8; 32], &RewardDraw)> = self.pending.iter().filter_map(|(rh, r)| r.draw.as_ref().map(|d| (rh, d))).collect();
        let windowed: Vec<(&[u8; 32], &QueuedWindow)> =
            self.pending.iter().filter_map(|(rh, r)| r.audit.as_ref().and_then(|a| a.v3.as_ref()).map(|w| (rh, w))).collect();
        let halves: Vec<(&[u8; 32], u8)> = self.trace_audits.iter().filter(|(_, a)| a.pieces != 0).map(|(k, a)| (k, a.pieces)).collect();
        let phased: Vec<(&[u8; 32], &AttnProgress)> = self.trace_audits.iter().filter_map(|(k, a)| a.attn.as_ref().map(|p| (k, p))).collect();
        let version = if !phased.is_empty() {
            SNAPSHOT_ENCODING_VERSION_PHASES
        } else if !halves.is_empty() {
            SNAPSHOT_ENCODING_VERSION_ATTN
        } else if !drawn.is_empty() || !windowed.is_empty() {
            SNAPSHOT_ENCODING_VERSION_QUEUE
        } else if !fit.is_empty() {
            SNAPSHOT_ENCODING_VERSION_CONTEXT
        } else if !self.trace_roots.is_empty() {
            SNAPSHOT_ENCODING_VERSION_ROOT
        } else if shaped {
            SNAPSHOT_ENCODING_VERSION_SHAPE
        } else if !self.trace_audits.is_empty() {
            SNAPSHOT_ENCODING_VERSION_TRACE
        } else if private.is_empty() {
            SNAPSHOT_ENCODING_VERSION
        } else {
            SNAPSHOT_ENCODING_VERSION_PRIVATE
        };
        out.push(version);
        out.extend_from_slice(&(self.vault.len() as u32).to_le_bytes());
        for (miner, claims) in self.vault.iter() {
            out.extend_from_slice(&miner.as_bytes());
            out.extend_from_slice(&(claims.len() as u32).to_le_bytes());
            for c in claims {
                out.extend_from_slice(&c.outpoint.transaction_id.as_bytes());
                out.extend_from_slice(&c.outpoint.index.to_le_bytes());
                out.extend_from_slice(&c.value.to_le_bytes());
                out.extend_from_slice(&c.daa.to_le_bytes());
            }
        }
        out.extend_from_slice(&(self.pending.len() as u32).to_le_bytes());
        for (rh, r) in self.pending.iter() {
            out.extend_from_slice(rh);
            out.push(r.tier);
            out.extend_from_slice(&r.max_tokens.to_le_bytes());
            out.extend_from_slice(&r.accepted_daa.to_le_bytes());
            out.extend_from_slice(&r.reward.to_le_bytes());
            match r.winner {
                Some(w) => {
                    out.push(1);
                    out.extend_from_slice(&w.as_bytes());
                }
                None => out.push(0),
            }
            match &r.audit {
                Some(a) => {
                    out.push(1);
                    put_hashes(&mut out, &a.cohort);
                    out.extend_from_slice(&(a.delegations.len() as u32).to_le_bytes());
                    for (e, id) in a.delegations.iter() {
                        out.extend_from_slice(&e.as_bytes());
                        out.extend_from_slice(&id.as_bytes());
                    }
                    put_hashes(&mut out, &a.responded);
                    out.extend_from_slice(&a.window_end_daa.to_le_bytes());
                }
                None => out.push(0),
            }
            put_hashes(&mut out, &r.early_responders);
        }
        out.extend_from_slice(&(self.early_responses.len() as u32).to_le_bytes());
        for (rh, (seen, keys)) in self.early_responses.iter() {
            out.extend_from_slice(rh);
            out.extend_from_slice(&seen.to_le_bytes());
            put_hashes(&mut out, keys);
        }
        out.extend_from_slice(&(self.strikes.len() as u32).to_le_bytes());
        for (m, e) in self.strikes.iter() {
            out.extend_from_slice(&m.as_bytes());
            out.extend_from_slice(&e.count.to_le_bytes());
            out.extend_from_slice(&e.last_daa.to_le_bytes());
        }
        out.extend_from_slice(&(self.first_seen.len() as u32).to_le_bytes());
        for (m, daa) in self.first_seen.iter() {
            out.extend_from_slice(&m.as_bytes());
            out.extend_from_slice(&daa.to_le_bytes());
        }
        out.extend_from_slice(&(self.audited.len() as u32).to_le_bytes());
        for (rh, daa) in self.audited.iter() {
            out.extend_from_slice(rh);
            out.extend_from_slice(&daa.to_le_bytes());
        }
        out.extend_from_slice(&(self.producer_spk.len() as u32).to_le_bytes());
        for (m, spk) in self.producer_spk.iter() {
            out.extend_from_slice(&m.as_bytes());
            out.extend_from_slice(&spk.version().to_le_bytes());
            out.extend_from_slice(&(spk.script().len() as u16).to_le_bytes());
            out.extend_from_slice(spk.script());
        }
        out.extend_from_slice(&(self.recent_producers.len() as u32).to_le_bytes());
        for (daa, id, tier, escrow) in self.recent_producers.iter() {
            out.extend_from_slice(&daa.to_le_bytes());
            out.extend_from_slice(&id.as_bytes());
            out.push(*tier);
            out.extend_from_slice(&escrow.as_bytes());
        }
        if version >= SNAPSHOT_ENCODING_VERSION_PRIVATE {
            out.extend_from_slice(&(private.len() as u32).to_le_bytes());
            for (rh, seed, recipients) in private {
                out.extend_from_slice(rh);
                out.extend_from_slice(seed);
                put_hashes(&mut out, recipients);
            }
        }
        if version >= SNAPSHOT_ENCODING_VERSION_TRACE {
            out.extend_from_slice(&(self.trace_audits.len() as u32).to_le_bytes());
            for (txid, a) in self.trace_audits.iter() {
                out.extend_from_slice(txid);
                out.extend_from_slice(&a.request_hash);
                out.extend_from_slice(&a.responder.as_bytes());
                out.extend_from_slice(&a.escrow_pubkey);
                out.extend_from_slice(&a.trace_root);
                out.extend_from_slice(&a.n_prompt.to_le_bytes());
                out.extend_from_slice(&a.n_pos.to_le_bytes());
                out.extend_from_slice(&a.accepted_daa.to_le_bytes());
                out.push(a.tier);
                out.extend_from_slice(&a.max_tokens.to_le_bytes());
                put_hashes(&mut out, &a.identities);
                match &a.seed {
                    Some(s) => {
                        out.push(1);
                        out.extend_from_slice(s);
                        out.extend_from_slice(&a.deadline_daa.to_le_bytes());
                    }
                    None => out.push(0),
                }
            }
        }
        if version >= SNAPSHOT_ENCODING_VERSION_SHAPE {
            let lens: Vec<(&[u8; 32], u32)> = self.pending.iter().filter(|(_, r)| r.prompt_len > 0).map(|(rh, r)| (rh, r.prompt_len)).collect();
            out.extend_from_slice(&(lens.len() as u32).to_le_bytes());
            for (rh, len) in lens {
                out.extend_from_slice(rh);
                out.extend_from_slice(&len.to_le_bytes());
            }
            out.extend_from_slice(&(self.early_shapes.len() as u32).to_le_bytes());
            for ((rh, key), shape) in self.early_shapes.iter() {
                out.extend_from_slice(rh);
                out.extend_from_slice(&key.as_bytes());
                out.extend_from_slice(&shape.n_prompt.to_le_bytes());
                out.extend_from_slice(&shape.n_pos.to_le_bytes());
                out.extend_from_slice(&shape.answer_len.to_le_bytes());
            }
        }
        if version >= SNAPSHOT_ENCODING_VERSION_ROOT {
            out.extend_from_slice(&(self.trace_roots.len() as u32).to_le_bytes());
            for (root, daa) in self.trace_roots.iter() {
                out.extend_from_slice(root);
                out.extend_from_slice(&daa.to_le_bytes());
            }
        }
        if version >= SNAPSHOT_ENCODING_VERSION_CONTEXT {
            out.extend_from_slice(&(fit.len() as u32).to_le_bytes());
            for rh in fit {
                out.extend_from_slice(rh);
            }
        }
        if version >= SNAPSHOT_ENCODING_VERSION_QUEUE {
            out.extend_from_slice(&(drawn.len() as u32).to_le_bytes());
            for (rh, d) in drawn {
                out.extend_from_slice(rh);
                out.extend_from_slice(&d.open_daa.to_le_bytes());
                out.extend_from_slice(&d.close_daa.to_le_bytes());
                out.extend_from_slice(&(d.candidates.len() as u32).to_le_bytes());
                for (id, at) in d.candidates.iter() {
                    out.extend_from_slice(&id.as_bytes());
                    out.extend_from_slice(&at.to_le_bytes());
                }
            }
            out.extend_from_slice(&(windowed.len() as u32).to_le_bytes());
            for (rh, w) in windowed {
                out.extend_from_slice(rh);
                out.extend_from_slice(&w.arm_daa.to_le_bytes());
                out.extend_from_slice(&(w.queue.len() as u32).to_le_bytes());
                for (queued, capped) in w.queue.iter() {
                    out.extend_from_slice(&queued.to_le_bytes());
                    out.push(*capped as u8);
                }
                out.extend_from_slice(&(w.answers.len() as u32).to_le_bytes());
                for (id, at, shape) in w.answers.iter() {
                    out.extend_from_slice(&id.as_bytes());
                    out.extend_from_slice(&at.to_le_bytes());
                    match shape {
                        Some((n_prompt, generated)) => {
                            out.push(1);
                            out.extend_from_slice(&n_prompt.to_le_bytes());
                            out.extend_from_slice(&generated.to_le_bytes());
                        }
                        None => out.push(0),
                    }
                }
            }
        }
        if version >= SNAPSHOT_ENCODING_VERSION_ATTN {
            out.extend_from_slice(&(halves.len() as u32).to_le_bytes());
            for (txid, piece) in halves {
                out.extend_from_slice(txid);
                out.push(piece);
            }
        }
        if version >= SNAPSHOT_ENCODING_VERSION_PHASES {
            out.extend_from_slice(&(phased.len() as u32).to_le_bytes());
            for (txid, p) in phased {
                out.extend_from_slice(txid);
                out.push(p.path.len() as u8);
                for c in &p.path {
                    out.extend_from_slice(&c.to_le_bytes());
                }
                out.extend_from_slice(&p.root);
                out.extend_from_slice(&p.n.to_le_bytes());
                out.extend_from_slice(&p.accepted_daa.to_le_bytes());
                out.push(p.drawn as u8);
                for w in &p.weights {
                    out.extend_from_slice(&w.to_le_bytes());
                }
            }
        }
        out
    }

    /// Parses [`Self::to_bytes`] output; rejects malformed, unordered or trailing data.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let mut r = Reader { bytes, pos: 0 };
        let version = r.u8()?;
        if !matches!(
            version,
            SNAPSHOT_ENCODING_VERSION
                | SNAPSHOT_ENCODING_VERSION_PRIVATE
                | SNAPSHOT_ENCODING_VERSION_TRACE
                | SNAPSHOT_ENCODING_VERSION_SHAPE
                | SNAPSHOT_ENCODING_VERSION_ROOT
                | SNAPSHOT_ENCODING_VERSION_CONTEXT
                | SNAPSHOT_ENCODING_VERSION_QUEUE
                | SNAPSHOT_ENCODING_VERSION_ATTN
                | SNAPSHOT_ENCODING_VERSION_PHASES
        ) {
            return Err("unsupported snapshot encoding".into());
        }
        let mut snap = Self::default();
        let n = r.u32()?;
        for _ in 0..n {
            let miner = r.hash()?;
            let k = r.u32()?;
            let mut claims = Vec::with_capacity(k.min(1 << 16) as usize);
            for _ in 0..k {
                let txid = r.hash()?;
                let index = r.u32()?;
                let value = r.u64()?;
                let daa = r.u64()?;
                claims.push(EscrowClaim { outpoint: crate::tx::TransactionOutpoint::new(txid, index), value, daa });
            }
            if claims.is_empty() || snap.vault.insert(miner, claims).is_some() {
                return Err("malformed vault".into());
            }
        }
        let n = r.u32()?;
        for _ in 0..n {
            let rh = r.key()?;
            let tier = r.u8()?;
            let max_tokens = r.u32()?;
            let accepted_daa = r.u64()?;
            let reward = r.u64()?;
            let winner = match r.u8()? {
                0 => None,
                1 => Some(r.hash()?),
                _ => return Err("malformed winner".into()),
            };
            let audit = match r.u8()? {
                0 => None,
                1 => {
                    let cohort = r.hashes()?;
                    let k = r.u32()? as usize;
                    let mut delegations = Vec::with_capacity(k.min(1 << 16));
                    for _ in 0..k {
                        let e = r.hash()?;
                        let id = r.hash()?;
                        delegations.push((e, id));
                    }
                    let responded = r.hashes()?;
                    let window_end_daa = r.u64()?;
                    Some(Audit { cohort, delegations, responded, window_end_daa, v3: None })
                }
                _ => return Err("malformed audit".into()),
            };
            let early_responders = r.hashes()?;
            let req = PendingRequest {
                tier,
                max_tokens,
                accepted_daa,
                reward,
                winner,
                audit,
                early_responders,
                recipients: Vec::new(),
                cohort_seed: [0u8; 36],
                prompt_len: 0,
                fit_proven: false,
                draw: None,
            };
            if snap.pending.insert(rh, req).is_some() {
                return Err("malformed pending".into());
            }
        }
        let n = r.u32()?;
        for _ in 0..n {
            let rh = r.key()?;
            let seen = r.u64()?;
            let keys = r.hashes()?;
            if snap.early_responses.insert(rh, (seen, keys)).is_some() {
                return Err("malformed early responses".into());
            }
        }
        let n = r.u32()?;
        for _ in 0..n {
            let m = r.hash()?;
            let count = r.u32()?;
            let last_daa = r.u64()?;
            if snap.strikes.insert(m, StrikeEntry { count, last_daa }).is_some() {
                return Err("malformed strikes".into());
            }
        }
        let n = r.u32()?;
        for _ in 0..n {
            let m = r.hash()?;
            let daa = r.u64()?;
            if snap.first_seen.insert(m, daa).is_some() {
                return Err("malformed first_seen".into());
            }
        }
        let n = r.u32()?;
        for _ in 0..n {
            let rh = r.key()?;
            let daa = r.u64()?;
            if snap.audited.insert(rh, daa).is_some() {
                return Err("malformed audited".into());
            }
        }
        let n = r.u32()?;
        for _ in 0..n {
            let m = r.hash()?;
            let version = r.u16()?;
            let len = r.u16()? as usize;
            let script = r.take(len)?;
            if snap.producer_spk.insert(m, ScriptPublicKey::from_vec(version, script.to_vec())).is_some() {
                return Err("malformed producer_spk".into());
            }
        }
        let n = r.u32()?;
        for _ in 0..n {
            let daa = r.u64()?;
            let id = r.hash()?;
            let tier = r.u8()?;
            let escrow = r.hash()?;
            snap.recent_producers.push((daa, id, tier, escrow));
        }
        if version >= SNAPSHOT_ENCODING_VERSION_PRIVATE {
            let n = r.u32()?;
            if n == 0 && version == SNAPSHOT_ENCODING_VERSION_PRIVATE {
                return Err("empty private section".into());
            }
            for _ in 0..n {
                let rh = r.key()?;
                let seed: PrivateCohortSeed = r.take(36)?.try_into().unwrap();
                let keys = r.hashes()?;
                if keys.is_empty() || keys.windows(2).any(|w| w[0] >= w[1]) {
                    return Err("malformed private recipients".into());
                }
                match snap.pending.get_mut(&rh) {
                    Some(req) if req.recipients.is_empty() => {
                        req.recipients = keys;
                        req.cohort_seed = seed;
                    }
                    _ => return Err("malformed private section".into()),
                }
            }
        }
        if version >= SNAPSHOT_ENCODING_VERSION_TRACE {
            let n = r.u32()?;
            if n == 0 && version == SNAPSHOT_ENCODING_VERSION_TRACE {
                return Err("empty trace section".into());
            }
            for _ in 0..n {
                let txid = r.key()?;
                let mut a = TraceAudit {
                    request_hash: r.key()?,
                    responder: r.hash()?,
                    escrow_pubkey: r.key()?,
                    trace_root: r.key()?,
                    n_prompt: r.u32()?,
                    n_pos: r.u32()?,
                    accepted_daa: r.u64()?,
                    tier: r.u8()?,
                    max_tokens: r.u32()?,
                    identities: r.hashes()?,
                    seed: None,
                    deadline_daa: 0,
                    pieces: 0,
                    attn: None,
                };
                match r.u8()? {
                    0 => {}
                    1 => {
                        a.seed = Some(r.key()?);
                        a.deadline_daa = r.u64()?;
                    }
                    _ => return Err("malformed trace audit".into()),
                }
                if snap.trace_audits.insert(txid, a).is_some() {
                    return Err("malformed trace section".into());
                }
            }
        }
        if version >= SNAPSHOT_ENCODING_VERSION_SHAPE {
            let n = r.u32()?;
            for _ in 0..n {
                let rh = r.key()?;
                let len = r.u32()?;
                match snap.pending.get_mut(&rh) {
                    Some(req) if req.prompt_len == 0 && len > 0 => req.prompt_len = len,
                    _ => return Err("malformed prompt lengths".into()),
                }
            }
            let n = r.u32()?;
            for _ in 0..n {
                let rh = r.key()?;
                let key = r.hash()?;
                let shape = TraceShape { n_prompt: r.u32()?, n_pos: r.u32()?, answer_len: r.u32()? };
                if snap.early_shapes.insert((rh, key), shape).is_some() {
                    return Err("malformed early shapes".into());
                }
            }
        }
        if version >= SNAPSHOT_ENCODING_VERSION_ROOT {
            let n = r.u32()?;
            if n == 0 && version == SNAPSHOT_ENCODING_VERSION_ROOT {
                return Err("empty root section".into());
            }
            for _ in 0..n {
                let root = r.key()?;
                let daa = r.u64()?;
                if snap.trace_roots.insert(root, daa).is_some() {
                    return Err("malformed trace roots".into());
                }
            }
        }
        if version >= SNAPSHOT_ENCODING_VERSION_CONTEXT {
            let n = r.u32()?;
            if n == 0 && version == SNAPSHOT_ENCODING_VERSION_CONTEXT {
                return Err("empty context section".into());
            }
            for _ in 0..n {
                let rh = r.key()?;
                match snap.pending.get_mut(&rh) {
                    Some(req) if !req.fit_proven => req.fit_proven = true,
                    _ => return Err("malformed context section".into()),
                }
            }
        }
        if version >= SNAPSHOT_ENCODING_VERSION_QUEUE {
            let n_draws = r.u32()?;
            for _ in 0..n_draws {
                let rh = r.key()?;
                let open_daa = r.u64()?;
                let close_daa = r.u64()?;
                let k = r.u32()? as usize;
                let mut candidates = Vec::with_capacity(k.min(1 << 16));
                for _ in 0..k {
                    let id = r.hash()?;
                    let at = r.u64()?;
                    candidates.push((id, at));
                }
                match snap.pending.get_mut(&rh) {
                    Some(req) if req.draw.is_none() => req.draw = Some(RewardDraw { open_daa, close_daa, candidates }),
                    _ => return Err("malformed draw section".into()),
                }
            }
            let n_windows = r.u32()?;
            if n_draws == 0 && n_windows == 0 && version == SNAPSHOT_ENCODING_VERSION_QUEUE {
                return Err("empty queue section".into());
            }
            for _ in 0..n_windows {
                let rh = r.key()?;
                let arm_daa = r.u64()?;
                let q = r.u32()? as usize;
                let mut queue = Vec::with_capacity(q.min(1 << 16));
                for _ in 0..q {
                    let queued = r.u64()?;
                    let capped = match r.u8()? {
                        0 => false,
                        1 => true,
                        _ => return Err("malformed queue".into()),
                    };
                    queue.push((queued, capped));
                }
                let k = r.u32()? as usize;
                let mut answers = Vec::with_capacity(k.min(1 << 16));
                for _ in 0..k {
                    let id = r.hash()?;
                    let at = r.u64()?;
                    let shape = match r.u8()? {
                        0 => None,
                        1 => Some((r.u32()?, r.u32()?)),
                        _ => return Err("malformed answer".into()),
                    };
                    answers.push((id, at, shape));
                }
                match snap.pending.get_mut(&rh).and_then(|req| req.audit.as_mut()) {
                    Some(a) if a.v3.is_none() && a.cohort.len() == queue.len() => a.v3 = Some(QueuedWindow { arm_daa, queue, answers }),
                    _ => return Err("malformed window section".into()),
                }
            }
        }
        if version >= SNAPSHOT_ENCODING_VERSION_ATTN {
            let n = r.u32()?;
            if n == 0 && version == SNAPSHOT_ENCODING_VERSION_ATTN {
                return Err("empty attention section".into());
            }
            for _ in 0..n {
                let txid = r.key()?;
                let piece = r.u8()?;
                match snap.trace_audits.get_mut(&txid) {
                    Some(a) if a.pieces == 0 && a.seed.is_some() && piece != 0 && piece & !PIECE_ALL == 0 && piece != PIECE_ALL => a.pieces = piece,
                    _ => return Err("malformed attention section".into()),
                }
            }
        }
        if version >= SNAPSHOT_ENCODING_VERSION_PHASES {
            let n = r.u32()?;
            if n == 0 {
                return Err("empty phase section".into());
            }
            for _ in 0..n {
                let txid = r.key()?;
                let k = r.u8()? as usize;
                let path = (0..k).map(|_| r.u64()).collect::<Result<Vec<_>, _>>()?;
                let root = r.key()?;
                let children = r.u64()?;
                let accepted_daa = r.u64()?;
                let drawn = match r.u8()? {
                    0 => false,
                    1 => true,
                    _ => return Err("malformed phase section".into()),
                };
                if !(1..=MAX_ATTN_CHILDREN).contains(&children) {
                    return Err("malformed phase section".into());
                }
                let weights = (0..children).map(|_| r.u16()).collect::<Result<Vec<_>, _>>()?;
                let ok = weights.iter().all(|&w| w >= 1) && (!drawn || path.last().is_some_and(|&c| c < children));
                match snap.trace_audits.get_mut(&txid) {
                    Some(a) if ok && a.attn.is_none() && a.seed.is_some() && a.pieces & PIECE_ATTN == 0 => {
                        a.attn = Some(AttnProgress { path, root, n: children, accepted_daa, drawn, weights })
                    }
                    _ => return Err("malformed phase section".into()),
                }
            }
        }
        if r.pos != bytes.len() {
            return Err("trailing bytes".into());
        }
        if snap.to_bytes() != bytes {
            return Err("non-canonical snapshot".into());
        }
        Ok(snap)
    }

    /// Domain-separated digest of the canonical bytes.
    pub fn hash(&self) -> Hash {
        Self::hash_of_bytes(&self.to_bytes())
    }

    /// [`Self::hash`] over bytes already encoded.
    pub fn hash_of_bytes(bytes: &[u8]) -> Hash {
        let mut hasher = blake2b_simd::Params::new().hash_length(32).personal(b"KeryxLedgerSnap").to_state();
        hasher.update(bytes);
        Hash::from_bytes(hasher.finalize().as_bytes().try_into().unwrap())
    }
}

/// Header service-state commitment past `service_ledger_activation`: the sealed rows and the
/// ledger snapshot, both at the header's pruning point.
pub fn service_commitment_v2(rows: Hash, ledger: Hash) -> Hash {
    let mut hasher = blake2b_simd::Params::new().hash_length(32).personal(b"KeryxSvcCommit").to_state();
    hasher.update(&rows.as_bytes());
    hasher.update(&ledger.as_bytes());
    Hash::from_bytes(hasher.finalize().as_bytes().try_into().unwrap())
}

/// Header service-state commitment past `production_index_activation`: sealed rows, ledger
/// snapshot and production-index snapshot, all at the header's pruning point.
pub fn service_commitment_v3(rows: Hash, ledger: Hash, production: Hash) -> Hash {
    let mut hasher = blake2b_simd::Params::new().hash_length(32).personal(b"KeryxSvcCommit3").to_state();
    hasher.update(&rows.as_bytes());
    hasher.update(&ledger.as_bytes());
    hasher.update(&production.as_bytes());
    Hash::from_bytes(hasher.finalize().as_bytes().try_into().unwrap())
}

/// Header service-state commitment past `exact_verification_activation`: v3 with the production
/// snapshot hashed in its window-relative form (`ProductionIndexSnapshot::relative_hash`).
pub fn service_commitment_v4(rows: Hash, ledger: Hash, production: Hash) -> Hash {
    let mut hasher = blake2b_simd::Params::new().hash_length(32).personal(b"KeryxSvcCommit4").to_state();
    hasher.update(&rows.as_bytes());
    hasher.update(&ledger.as_bytes());
    hasher.update(&production.as_bytes());
    Hash::from_bytes(hasher.finalize().as_bytes().try_into().unwrap())
}

const PRODUCTION_SNAPSHOT_ENCODING_V1: u8 = 1;
const PRODUCTION_SNAPSHOT_ENCODING_V2: u8 = 2;

/// Canonical snapshot of the ratio-reward production index at a pruning sample: per-SPK cumulative
/// production at the window bottom (`floors`), the sparse prefix entries inside the window, both
/// sorted by canonical SPK bytes, and the daa score of every chain index in the window.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ProductionIndexSnapshot {
    /// Chain index of the window bottom the floors are cumulative at.
    pub bottom_index: u64,
    /// Chain index of the sample block the snapshot is taken at.
    pub sample_index: u64,
    /// `(spk, cumulative_at(spk, bottom_index))`, sorted by (version, script).
    pub floors: Vec<(ScriptPublicKey, u64)>,
    /// `(spk, entries)` with entries `(chain_index, cumulative)` strictly increasing in
    /// `(bottom_index, sample_index]`; groups sorted by (version, script).
    pub entries: Vec<(ScriptPublicKey, Vec<(u64, u64)>)>,
    /// daa score of chain indices `bottom_index..=sample_index`, strictly increasing; empty on a
    /// v1 snapshot.
    pub window_daa: Vec<u64>,
}

impl ProductionIndexSnapshot {
    /// Whether the snapshot carries the window daa table.
    pub fn has_window_table(&self) -> bool {
        !self.window_daa.is_empty()
    }

    /// The snapshot re-based to its window: chain indices from `bottom_index`, cumulative values
    /// from each SPK's floor. Identical on every node holding the same window production, whatever
    /// its own chain numbering and index baseline.
    pub fn relative(&self) -> Self {
        let floor_of = |spk: &ScriptPublicKey| self.floors.iter().find(|(s, _)| s == spk).map(|(_, v)| *v).unwrap_or(0);
        Self {
            bottom_index: 0,
            sample_index: self.sample_index.saturating_sub(self.bottom_index),
            floors: self.floors.iter().map(|(spk, _)| (spk.clone(), 0)).collect(),
            entries: self
                .entries
                .iter()
                .map(|(spk, e)| {
                    let floor = floor_of(spk);
                    (spk.clone(), e.iter().map(|(i, c)| (i.saturating_sub(self.bottom_index), c.saturating_sub(floor))).collect())
                })
                .collect(),
            window_daa: self.window_daa.clone(),
        }
    }

    /// Hash of the re-based snapshot's canonical encoding.
    pub fn relative_hash(&self) -> Hash {
        Self::hash_of_bytes(&self.relative().to_bytes())
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.push(if self.window_daa.is_empty() { PRODUCTION_SNAPSHOT_ENCODING_V1 } else { PRODUCTION_SNAPSHOT_ENCODING_V2 });
        out.extend_from_slice(&self.bottom_index.to_le_bytes());
        out.extend_from_slice(&self.sample_index.to_le_bytes());
        out.extend_from_slice(&(self.floors.len() as u32).to_le_bytes());
        for (spk, cum) in self.floors.iter() {
            put_spk(&mut out, spk);
            out.extend_from_slice(&cum.to_le_bytes());
        }
        out.extend_from_slice(&(self.entries.len() as u32).to_le_bytes());
        for (spk, points) in self.entries.iter() {
            put_spk(&mut out, spk);
            out.extend_from_slice(&(points.len() as u32).to_le_bytes());
            for (index, cum) in points.iter() {
                out.extend_from_slice(&index.to_le_bytes());
                out.extend_from_slice(&cum.to_le_bytes());
            }
        }
        if let Some((first, rest)) = self.window_daa.split_first() {
            out.extend_from_slice(&(self.window_daa.len() as u32).to_le_bytes());
            out.extend_from_slice(&first.to_le_bytes());
            let mut prev = *first;
            for daa in rest {
                put_uvarint(&mut out, daa - prev);
                prev = *daa;
            }
        }
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let mut r = Reader { bytes, pos: 0 };
        let version = r.u8()?;
        if version != PRODUCTION_SNAPSHOT_ENCODING_V1 && version != PRODUCTION_SNAPSHOT_ENCODING_V2 {
            return Err("unknown production snapshot version".into());
        }
        let mut snap = Self { bottom_index: r.u64()?, sample_index: r.u64()?, ..Default::default() };
        if snap.sample_index < snap.bottom_index {
            return Err("malformed production window".into());
        }
        let n = r.u32()?;
        for _ in 0..n {
            let spk = read_spk(&mut r)?;
            let cum = r.u64()?;
            if snap.floors.last().is_some_and(|(prev, _)| spk_key(prev) >= spk_key(&spk)) {
                return Err("unsorted floors".into());
            }
            snap.floors.push((spk, cum));
        }
        let m = r.u32()?;
        for _ in 0..m {
            let spk = read_spk(&mut r)?;
            if snap.entries.last().is_some_and(|(prev, _)| spk_key(prev) >= spk_key(&spk)) {
                return Err("unsorted entries".into());
            }
            let k = r.u32()?;
            let mut points = Vec::with_capacity(k as usize);
            let mut prev = None;
            for _ in 0..k {
                let index = r.u64()?;
                let cum = r.u64()?;
                if index <= snap.bottom_index || index > snap.sample_index || prev.is_some_and(|p| index <= p) {
                    return Err("malformed production entries".into());
                }
                prev = Some(index);
                points.push((index, cum));
            }
            snap.entries.push((spk, points));
        }
        if version == PRODUCTION_SNAPSHOT_ENCODING_V2 {
            let n = r.u32()? as u64;
            if n == 0 || n != snap.sample_index - snap.bottom_index + 1 {
                return Err("malformed production window table".into());
            }
            let mut daa = r.u64()?;
            snap.window_daa.reserve(n as usize);
            snap.window_daa.push(daa);
            for _ in 1..n {
                let delta = read_uvarint(&mut r)?;
                if delta == 0 {
                    return Err("non-increasing production window table".into());
                }
                daa = daa.checked_add(delta).ok_or("production window table overflow")?;
                snap.window_daa.push(daa);
            }
        }
        if r.pos != bytes.len() {
            return Err("trailing bytes".into());
        }
        if snap.to_bytes() != bytes {
            return Err("non-canonical production snapshot".into());
        }
        Ok(snap)
    }

    /// daa score of chain index `index`, when the window table covers it.
    pub fn window_daa_at(&self, index: u64) -> Option<u64> {
        if index < self.bottom_index {
            return None;
        }
        self.window_daa.get((index - self.bottom_index) as usize).copied()
    }

    /// Domain-separated digest of the canonical bytes.
    pub fn hash(&self) -> Hash {
        Self::hash_of_bytes(&self.to_bytes())
    }

    /// [`Self::hash`] over bytes already encoded.
    pub fn hash_of_bytes(bytes: &[u8]) -> Hash {
        let mut hasher = blake2b_simd::Params::new().hash_length(32).personal(b"KeryxProdSnap").to_state();
        hasher.update(bytes);
        Hash::from_bytes(hasher.finalize().as_bytes().try_into().unwrap())
    }
}

fn spk_key(spk: &ScriptPublicKey) -> (u16, &[u8]) {
    (spk.version(), spk.script())
}

fn put_spk(out: &mut Vec<u8>, spk: &ScriptPublicKey) {
    out.extend_from_slice(&spk.version().to_le_bytes());
    out.extend_from_slice(&(spk.script().len() as u16).to_le_bytes());
    out.extend_from_slice(spk.script());
}

fn read_spk(r: &mut Reader<'_>) -> Result<ScriptPublicKey, String> {
    let version = r.u16()?;
    let len = r.u16()? as usize;
    let script = r.take(len)?;
    Ok(ScriptPublicKey::from_vec(version, script.to_vec()))
}

impl ServiceLedger {
    /// The effective state as a canonical snapshot: strike and sighting baselines folded into
    /// their deltas, vault claims in chain order.
    pub fn snapshot(&self) -> ServiceLedgerSnapshot {
        let mut strikes = self.base.as_ref().clone();
        for (m, e) in self.strikes.iter() {
            strikes.insert(*m, *e);
        }
        let mut first_seen = self.first_seen_base.as_ref().clone();
        for (m, d) in self.first_seen.iter() {
            first_seen.insert(*m, *d);
        }
        ServiceLedgerSnapshot {
            vault: self.vault.iter().map(|(m, c)| (*m, c.iter().copied().collect())).collect(),
            pending: self.pending.clone(),
            early_responses: self.early_responses.clone(),
            strikes,
            first_seen,
            audited: self.audited.clone(),
            producer_spk: self.producer_spk.clone(),
            recent_producers: Vec::new(),
            trace_audits: self.trace_audits.clone(),
            early_shapes: self.early_shapes.clone(),
            trace_roots: self.trace_roots.clone(),
        }
    }

    /// Replaces the folded state with `snap`; configuration (activation daas) is kept.
    pub fn restore_snapshot(&mut self, snap: &ServiceLedgerSnapshot) {
        self.vault = snap.vault.iter().map(|(m, c)| (*m, c.iter().copied().collect())).collect();
        self.pending = snap.pending.clone();
        self.early_responses = snap.early_responses.clone();
        self.strikes = snap.strikes.clone();
        self.base = std::sync::Arc::new(Default::default());
        self.first_seen = snap.first_seen.clone();
        self.first_seen_base = std::sync::Arc::new(Default::default());
        self.audited = snap.audited.clone();
        self.producer_spk = snap.producer_spk.clone();
        self.trace_audits = snap.trace_audits.clone();
        self.early_shapes = snap.early_shapes.clone();
        self.trace_roots = snap.trace_roots.clone();
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn snapshot_encodes_private_recipients_in_a_versioned_trailer() {
        use crate::collateral::{ServiceLedger, ServiceLedgerSnapshot};
        use keryx_hashes::Hash;
        let b = Hash::from_u64_word(2);
        let public = [1u8; 32];
        let private = [2u8; 32];
        let mut ledger = ServiceLedger::default();
        ledger.set_window_v2_activation(0);
        ledger.set_reward_routing_activation(0);
        ledger.on_chain_block_with_recipients(
            100,
            &[(public, 0, 1), (private, 0, 1)],
            &[],
            &[(private, [7u8; 36], vec![b])],
            &[],
            &[],
            &[],
            |_| true,
            |_| vec![(b, b)],
        );
        let snap = ledger.snapshot();
        let bytes = snap.to_bytes();
        assert_eq!(bytes[0], 2);
        let parsed = ServiceLedgerSnapshot::from_bytes(&bytes).unwrap();
        assert_eq!(parsed, snap);
        let mut restored = ServiceLedger::default();
        restored.restore_snapshot(&parsed);
        assert_eq!(restored.private_recipients(&private), Some(&[b][..]));
        assert_eq!(restored.private_recipients(&public), None);
        assert_eq!(restored.pending.get(&private).unwrap().cohort_seed, [7u8; 36]);

        // Without private requests the encoding is the historical one, byte for byte.
        let mut plain = ServiceLedger::default();
        plain.set_window_v2_activation(0);
        plain.set_reward_routing_activation(0);
        plain.on_chain_block_with_recipients(100, &[(public, 0, 1)], &[], &[], &[], &[], &[], |_| true, |_| vec![(b, b)]);
        let plain_bytes = plain.snapshot().to_bytes();
        assert_eq!(plain_bytes[0], 1);
        assert!(ServiceLedgerSnapshot::from_bytes(&plain_bytes).is_ok());

        // A version-2 header without a trailer, or a trailer naming an unknown request, is rejected.
        let mut bad = plain_bytes.clone();
        bad[0] = 2;
        assert!(ServiceLedgerSnapshot::from_bytes(&bad).is_err());
        let mut bad = bytes.clone();
        let at = bytes.len() - 32 - 4 - 36 - 32; // the request hash of the single trailer entry
        bad[at] ^= 0xFF;
        assert!(ServiceLedgerSnapshot::from_bytes(&bad).is_err());
        // A version-1 header on bytes that carry a trailer is trailing data.
        let mut bad = bytes.clone();
        bad[0] = 1;
        assert!(ServiceLedgerSnapshot::from_bytes(&bad).is_err());
    }

    #[test]
    fn persisted_baselines_replace_the_restored_strike_state() {
        use crate::collateral::{ServiceLedger, ServiceLedgerSnapshot, StrikeEntry};
        use keryx_hashes::Hash;
        let m = Hash::from_u64_word(1);
        let n = Hash::from_u64_word(2);
        let mut snap = ServiceLedgerSnapshot::default();
        snap.strikes.insert(m, StrikeEntry { count: 2, last_daa: 5 });
        let mut ledger = ServiceLedger::default();
        ledger.restore_snapshot(&snap);
        assert_eq!(ledger.consecutive_misses(&m), 2);

        let base = [(m, StrikeEntry { count: 3, last_daa: 9 }), (n, StrikeEntry { count: 1, last_daa: 2 })].into_iter().collect();
        ledger.set_persisted_baselines(std::sync::Arc::new(base), std::sync::Arc::new(Default::default()));
        assert_eq!(ledger.consecutive_misses(&m), 3, "the persisted record wins over the older snapshot delta");
        assert_eq!(ledger.consecutive_misses(&n), 1);
    }

    #[test]
    fn relative_hash_is_independent_of_chain_numbering_and_floors() {
        use crate::collateral::{service_commitment_v3, service_commitment_v4, ProductionIndexSnapshot};
        use crate::tx::ScriptPublicKey;
        use keryx_hashes::Hash;
        let spk = ScriptPublicKey::from_vec(0, vec![0x20; 34]);
        let window_daa: Vec<u64> = (0..=150u64).map(|i| 1_000 + i * 5).collect();
        let a = ProductionIndexSnapshot {
            bottom_index: 100,
            sample_index: 250,
            floors: vec![(spk.clone(), 7)],
            entries: vec![(spk.clone(), vec![(101, 9), (240, 12)])],
            window_daa: window_daa.clone(),
        };
        let b = ProductionIndexSnapshot {
            bottom_index: 5_100,
            sample_index: 5_250,
            floors: vec![(spk.clone(), 1_007)],
            entries: vec![(spk.clone(), vec![(5_101, 1_009), (5_240, 1_012)])],
            window_daa,
        };
        assert_ne!(ProductionIndexSnapshot::hash_of_bytes(&a.to_bytes()), ProductionIndexSnapshot::hash_of_bytes(&b.to_bytes()));
        assert_eq!(a.relative_hash(), b.relative_hash(), "same window production, same relative hash");
        let c = ProductionIndexSnapshot { entries: vec![(spk.clone(), vec![(101, 9), (240, 13)])], ..a.clone() };
        assert_ne!(a.relative_hash(), c.relative_hash(), "a different production changes the relative hash");
        let (rows, ledger) = (Hash::from_u64_word(1), Hash::from_u64_word(2));
        assert_ne!(service_commitment_v3(rows, ledger, a.relative_hash()), service_commitment_v4(rows, ledger, a.relative_hash()));
    }

    #[test]
    fn production_snapshot_roundtrip_is_canonical() {
        use crate::collateral::ProductionIndexSnapshot;
        use crate::tx::ScriptPublicKey;
        let spk_a = ScriptPublicKey::from_vec(0, vec![0x20; 34]);
        let spk_b = ScriptPublicKey::from_vec(0, vec![0x21; 34]);
        let snap = ProductionIndexSnapshot {
            bottom_index: 100,
            sample_index: 250,
            floors: vec![(spk_a.clone(), 7), (spk_b.clone(), 0)],
            entries: vec![(spk_a.clone(), vec![(101, 9), (240, 12)]), (spk_b.clone(), vec![(250, 3)])],
            window_daa: vec![],
        };
        let bytes = snap.to_bytes();
        assert_eq!(bytes[0], 1, "a snapshot without the table keeps the legacy encoding and hash");
        let decoded = ProductionIndexSnapshot::from_bytes(&bytes).unwrap();
        assert_eq!(decoded, snap);
        assert_eq!(ProductionIndexSnapshot::hash_of_bytes(&bytes), snap.hash());

        // With the window daa table: v2 encoding, a different hash, exact round trip.
        let tabled = ProductionIndexSnapshot { window_daa: (0..=150u64).map(|i| 1_000 + i * 5).collect(), ..snap.clone() };
        let tbytes = tabled.to_bytes();
        assert_eq!(tbytes[0], 2);
        assert_ne!(ProductionIndexSnapshot::hash_of_bytes(&tbytes), snap.hash());
        assert_eq!(ProductionIndexSnapshot::from_bytes(&tbytes).unwrap(), tabled);
        assert_eq!(tabled.window_daa_at(100), Some(1_000));
        assert_eq!(tabled.window_daa_at(250), Some(1_750));
        assert_eq!(tabled.window_daa_at(99), None);
        assert_eq!(tabled.window_daa_at(251), None);
        // Large deltas exercise multi-byte varints.
        let wide = ProductionIndexSnapshot { window_daa: (0..=150u64).map(|i| (1 << 40) | (i * 300)).collect(), ..snap.clone() };
        assert_eq!(ProductionIndexSnapshot::from_bytes(&wide.to_bytes()).unwrap(), wide);

        // A table that does not cover exactly [bottom, sample] is rejected.
        let short = ProductionIndexSnapshot { window_daa: vec![1, 2, 3], ..snap.clone() };
        assert!(ProductionIndexSnapshot::from_bytes(&short.to_bytes()).is_err());
        // A non-increasing table is rejected.
        let mut flat = tabled.clone();
        flat.window_daa[10] = flat.window_daa[9];
        assert!(ProductionIndexSnapshot::from_bytes(&flat.to_bytes()).is_err());

        // An empty snapshot is valid and hashes deterministically (the genesis commitment).
        let empty = ProductionIndexSnapshot::default();
        assert_eq!(ProductionIndexSnapshot::from_bytes(&empty.to_bytes()).unwrap(), empty);

        // Unsorted groups are rejected (order malleability would fork the commitment).
        let unsorted = ProductionIndexSnapshot {
            bottom_index: 100,
            sample_index: 250,
            floors: vec![(spk_b.clone(), 0), (spk_a.clone(), 7)],
            entries: vec![],
            window_daa: vec![],
        };
        assert!(ProductionIndexSnapshot::from_bytes(&unsorted.to_bytes()).is_err());

        // An entry outside (bottom, sample] is rejected.
        let out_of_window = ProductionIndexSnapshot {
            bottom_index: 100,
            sample_index: 250,
            floors: vec![],
            entries: vec![(spk_a, vec![(100, 5)])],
            window_daa: vec![],
        };
        assert!(ProductionIndexSnapshot::from_bytes(&out_of_window.to_bytes()).is_err());

        // Trailing garbage is rejected.
        let mut padded = bytes.clone();
        padded.push(0);
        assert!(ProductionIndexSnapshot::from_bytes(&padded).is_err());
    }
    use super::{
        eligible_pairs, private_cohort_rank, private_cohort_tolerance, private_target_cohort, service_window_daa, service_window_daa_at,
        strike_penalty, strike_penalty_at, update_strikes, FoldOutcome, PRIVATE_COHORT_TARGET, ServiceLedger, ServicePenalty, ServiceReward,
        StrikeEntry, AI_REQUEST_MAX_TOKENS_CAP, SERVICE_EARLY_RESPONSE_HORIZON_DAA, SERVICE_LEDGER_HORIZON_DAA,
        SERVICE_STRIKE_INTERVAL_DAA, STRIKE_1_BURN_CLAIMS,
    };
    use keryx_hashes::Hash;

    // Identity == escrow key in most tests; the delegation mapping itself is covered by
    // `response_credits_the_delegating_identity`.
    fn cohort_of(set: &[Hash]) -> impl FnMut(u8) -> Vec<(Hash, Hash)> + '_ {
        move |_tier| set.iter().map(|m| (*m, *m)).collect()
    }

    #[test]
    fn service_window_scales_with_tokens_and_clamps_at_cap() {
        // base 30 s + per-token allowance by tier class
        assert_eq!(service_window_daa(0, 256), 300 + 512);
        assert_eq!(service_window_daa(2, 256), 300 + 512);
        assert_eq!(service_window_daa(3, 256), 300 + 768);
        assert_eq!(service_window_daa(4, 256), 300 + 1024);
        // a request cannot buy more window than the max_tokens cap allows, and the worst
        // possible window stays well inside the ledger horizon
        assert_eq!(service_window_daa(4, u32::MAX), 300 + AI_REQUEST_MAX_TOKENS_CAP as u64 * 4);
        assert!(service_window_daa(4, u32::MAX) < SERVICE_LEDGER_HORIZON_DAA);
    }

    #[test]
    fn only_cohort_member_responses_count() {
        let a = Hash::from_bytes([1u8; 32]);
        let b = Hash::from_bytes([9u8; 32]);
        let set = [a];
        let mut ledger = ServiceLedger::default();
        let w = service_window_daa(0, 256);

        let rh = [7u8; 32];
        assert!(ledger.on_chain_block(100, &[(rh, 0, 256)], &[], &[], |_| true, cohort_of(&set)).misses.is_empty());
        // a response in the audit-opening block is parked and credited when the audit arms
        assert!(ledger.on_chain_block(101, &[], &[(rh, Some(a))], &[], |_| true, cohort_of(&set)).misses.is_empty());
        // a v1 (unsigned) response and a non-member response never count
        assert!(ledger.on_chain_block(102, &[], &[(rh, None), (rh, Some(b))], &[], |_| true, cohort_of(&set)).misses.is_empty());
        assert_eq!(ledger.pending_len(), 1);
        // the member's signed response does; the audit closes clean at its window end
        assert!(ledger.on_chain_block(103, &[], &[(rh, Some(a))], &[], |_| true, cohort_of(&set)).misses.is_empty());
        assert!(ledger.on_chain_block(102 + w, &[], &[], &[], |_| true, cohort_of(&set)).misses.is_empty());
        assert_eq!(ledger.pending_len(), 0);
    }

    /// Private inference: a request sealed to named responders audits, credits and pays only
    /// those responders. A non-recipient's signed answer counts for nothing, nobody outside the
    /// recipient set is struck when the window closes, and the recipient wins the vault.
    #[test]
    fn private_cohort_tolerance_is_a_quarter_with_a_floor_of_three() {
        assert_eq!(private_cohort_tolerance(0), 3);
        assert_eq!(private_cohort_tolerance(12), 3);
        assert_eq!(private_cohort_tolerance(13), 4);
        assert_eq!(private_cohort_tolerance(17), 5);
        assert_eq!(private_cohort_tolerance(45), 12);
    }

    /// A private request must be sealed to the cohort it arms with: up to the tolerance of
    /// members may be missing (they entered after sealing) and are neither obligated nor struck;
    /// one more makes it unservable — nobody is obligated and the vault is never minted.
    #[test]
    fn private_request_leaving_out_more_than_the_tolerance_is_unservable() {
        let set: Vec<Hash> = (1u8..=16).map(|i| Hash::from_bytes([i; 32])).collect();
        let tol = private_cohort_tolerance(set.len());
        assert_eq!(tol, 4);
        let arm = |covered: usize| {
            let mut ledger = ServiceLedger::default();
            ledger.set_window_v2_activation(0);
            ledger.set_reward_routing_activation(0);
            let rh = [7u8; 32];
            let recipients = set[..covered].to_vec();
            ledger.on_chain_block_with_recipients(100, &[(rh, 0, 256)], &[(rh, 5_000)], &[(rh, [0u8; 36], recipients)], &[], &[], &[], |_| true, cohort_of(&set));
            ledger.on_chain_block_with_recipients(101, &[], &[], &[], &[], &[], &[], |_| true, cohort_of(&set));
            (ledger.audit_cohort(&rh), ledger.pending_len())
        };
        let (cohort, pending) = arm(set.len() - tol);
        assert_eq!(cohort, Some(set[..set.len() - tol].to_vec()));
        assert_eq!(pending, 1);
        let (cohort, pending) = arm(set.len() - tol - 1);
        assert_eq!(cohort, None);
        assert_eq!(pending, 0);
    }

    #[test]
    fn private_target_cohort_keeps_everyone_up_to_the_cap_and_the_lowest_ranks_above() {
        let keys = |n: usize| -> Vec<Hash> { (0..n as u64).map(|i| Hash::from_u64_word(i + 1)).collect() };
        let seed = [3u8; 36];
        let full = keys(PRIVATE_COHORT_TARGET);
        assert_eq!(private_target_cohort(&seed, &full), full);

        let big = keys(150);
        let target = private_target_cohort(&seed, &big);
        assert_eq!(target.len(), PRIVATE_COHORT_TARGET);
        assert!(target.windows(2).all(|w| w[0] < w[1]));
        assert!(target.iter().all(|k| big.contains(k)));
        let worst_kept = target.iter().map(|k| private_cohort_rank(&seed, &k.as_bytes())).max().unwrap();
        assert!(big.iter().filter(|k| !target.contains(k)).all(|k| private_cohort_rank(&seed, &k.as_bytes()) > worst_kept));

        let mut shuffled = big.clone();
        shuffled.reverse();
        shuffled.push(big[0]);
        assert_eq!(private_target_cohort(&seed, &shuffled), target);
        assert_ne!(private_target_cohort(&[4u8; 36], &big), target);
    }

    /// Above the cap the coverage is measured against the target cohort: sealing to it is served,
    /// leaving out more than the tolerance of it is not, and a recipient the target dropped by
    /// churn stays obligated (it can read).
    #[test]
    fn private_request_above_the_cap_is_covered_against_its_target_cohort() {
        let seed = [9u8; 36];
        let set: Vec<Hash> = (0..150u64).map(|i| Hash::from_u64_word(i + 1)).collect();
        let target = private_target_cohort(&seed, &set);
        let tol = private_cohort_tolerance(target.len());
        assert_eq!(tol, 32);
        let arm = |recipients: Vec<Hash>, cohort: &[Hash]| {
            let mut ledger = ServiceLedger::default();
            ledger.set_window_v2_activation(0);
            ledger.set_reward_routing_activation(0);
            let rh = [7u8; 32];
            ledger.on_chain_block_with_recipients(100, &[(rh, 0, 256)], &[(rh, 5_000)], &[(rh, seed, recipients)], &[], &[], &[], |_| true, cohort_of(cohort));
            ledger.on_chain_block_with_recipients(101, &[], &[], &[], &[], &[], &[], |_| true, cohort_of(cohort));
            ledger.audit_cohort(&rh)
        };
        assert_eq!(arm(target.clone(), &set), Some(target.clone()));

        let short: Vec<Hash> = target[tol..].to_vec();
        assert_eq!(arm(short.clone(), &set), Some(short));
        let too_short: Vec<Hash> = target[tol + 1..].to_vec();
        assert_eq!(arm(too_short, &set), None);

        let outside: Vec<Hash> = set.iter().filter(|k| !target.contains(k)).copied().collect();
        assert_eq!(arm(outside, &set), None);

        let worst_kept = target.iter().map(|k| private_cohort_rank(&seed, &k.as_bytes())).max().unwrap();
        let evicted = *target.iter().find(|k| private_cohort_rank(&seed, &k.as_bytes()) == worst_kept).unwrap();
        let newcomer =
            (1000u64..).map(Hash::from_u64_word).find(|k| private_cohort_rank(&seed, &k.as_bytes()) < worst_kept).unwrap();
        let mut grown = set.clone();
        grown.push(newcomer);
        assert!(!private_target_cohort(&seed, &grown).contains(&evicted));
        let cohort = arm(target.clone(), &grown).unwrap();
        assert!(cohort.contains(&evicted));
        assert!(!cohort.contains(&newcomer));
    }

    #[test]
    fn private_request_restricts_the_cohort_to_its_recipients() {
        let a = Hash::from_bytes([1u8; 32]);
        let b = Hash::from_bytes([2u8; 32]);
        let c = Hash::from_bytes([3u8; 32]);
        let set = [a, b, c];
        let spk_b = crate::tx::ScriptPublicKey::from_vec(0, vec![0xBB; 34]);
        let mut ledger = ServiceLedger::default();
        ledger.set_window_v2_activation(0);
        ledger.set_reward_routing_activation(0);
        let w = service_window_daa_at(0, 256, true);

        let rh = [7u8; 32];
        let out = ledger.on_chain_block_with_recipients(
            100,
            &[(rh, 0, 256)],
            &[(rh, 5_000)],
            &[(rh, [0u8; 36], vec![b])],
            &[],
            &[],
            &[(b, spk_b.clone())],
            |_| true,
            cohort_of(&set),
        );
        assert!(out.misses.is_empty() && out.rewards.is_empty());
        assert_eq!(ledger.private_recipients(&rh), Some(&[b][..]));
        assert_eq!(ledger.audit_cohort(&rh), None);
        // Arms with the recipient alone.
        ledger.on_chain_block_with_recipients(101, &[], &[], &[], &[], &[], &[], |_| true, cohort_of(&set));
        assert_eq!(ledger.audit_cohort(&rh), Some(vec![b]));
        // A non-recipient's authenticated answer is ignored: no reward.
        let out = ledger.on_chain_block_with_recipients(102, &[], &[], &[], &[(rh, Some(a))], &[], &[], |_| true, cohort_of(&set));
        assert!(out.rewards.is_empty());
        // The recipient answers and wins the vault.
        let out = ledger.on_chain_block_with_recipients(103, &[], &[], &[], &[(rh, Some(b))], &[], &[], |_| true, cohort_of(&set));
        assert_eq!(out.rewards, vec![ServiceReward { request_hash: rh, winner: b, amount: 5_000, spk: Some(spk_b) }]);
        // Window close: nobody is struck for a request they could not read.
        let out = ledger.on_chain_block_with_recipients(102 + w, &[], &[], &[], &[], &[], &[], |_| true, cohort_of(&set));
        assert!(out.misses.is_empty());
        assert_eq!(ledger.pending_len(), 0);
        assert_eq!(ledger.private_recipients(&rh), None);
    }

    /// A silent named responder is struck like any silent cohort member; the rest of the tier
    /// is not.
    #[test]
    fn private_request_to_a_silent_recipient_strikes_only_that_recipient() {
        let a = Hash::from_bytes([1u8; 32]);
        let b = Hash::from_bytes([2u8; 32]);
        let set = [a, b];
        let mut ledger = ServiceLedger::default();
        ledger.set_window_v2_activation(0);
        ledger.set_reward_routing_activation(0);
        let w = service_window_daa_at(0, 256, true);
        let rh = [8u8; 32];
        ledger.on_chain_block_with_recipients(100, &[(rh, 0, 256)], &[], &[(rh, [0u8; 36], vec![b])], &[], &[], &[], |_| true, cohort_of(&set));
        ledger.on_chain_block_with_recipients(101, &[], &[], &[], &[], &[], &[], |_| true, cohort_of(&set));
        let out = ledger.on_chain_block_with_recipients(102 + w, &[], &[], &[], &[], &[], &[], |_| true, cohort_of(&set));
        assert_eq!(out.misses.iter().map(|m| m.miner).collect::<Vec<_>>(), vec![b]);
        assert_eq!(ledger.consecutive_misses(&b), 1);
        assert_eq!(ledger.consecutive_misses(&a), 0);
    }

    /// A private request whose recipients are all outside the eligible set is unservable: it is
    /// dropped at arming, exactly like a request for an empty tier, and strikes nobody.
    #[test]
    fn private_request_to_an_ineligible_recipient_is_unservable() {
        let a = Hash::from_bytes([1u8; 32]);
        let b = Hash::from_bytes([2u8; 32]);
        let set = [a];
        let mut ledger = ServiceLedger::default();
        ledger.set_window_v2_activation(0);
        ledger.set_reward_routing_activation(0);
        let rh = [9u8; 32];
        ledger.on_chain_block_with_recipients(100, &[(rh, 0, 256)], &[], &[(rh, [0u8; 36], vec![b])], &[], &[], &[], |_| true, cohort_of(&set));
        assert_eq!(ledger.pending_len(), 1);
        let out = ledger.on_chain_block_with_recipients(101, &[], &[], &[], &[], &[], &[], |_| true, cohort_of(&set));
        assert!(out.misses.is_empty());
        assert_eq!(ledger.pending_len(), 0);
        // An answer from the non-recipient afterwards is a stray response: nothing to credit.
        let out = ledger.on_chain_block_with_recipients(102, &[], &[], &[], &[(rh, Some(a))], &[], &[], |_| true, cohort_of(&set));
        assert!(out.rewards.is_empty());
    }

    /// An answer that lands before the audit arms is credited through the same recipient filter
    /// as a live one: a non-recipient's early answer earns nothing, a recipient's wins.
    #[test]
    fn early_private_answers_pass_the_recipient_filter() {
        let a = Hash::from_bytes([1u8; 32]);
        let b = Hash::from_bytes([2u8; 32]);
        let set = [a, b];
        let mut ledger = ServiceLedger::default();
        ledger.set_window_v2_activation(0);
        ledger.set_reward_routing_activation(0);
        let rh = [10u8; 32];
        // Both answer in the request's own block: only b's answer survives arming.
        ledger.on_chain_block_with_recipients(
            100,
            &[(rh, 0, 256)],
            &[(rh, 1_000)],
            &[(rh, [0u8; 36], vec![b])],
            &[(rh, Some(a)), (rh, Some(b))],
            &[],
            &[],
            |_| true,
            cohort_of(&set),
        );
        let out = ledger.on_chain_block_with_recipients(101, &[], &[], &[], &[], &[], &[], |_| true, cohort_of(&set));
        assert_eq!(out.rewards.iter().map(|r| r.winner).collect::<Vec<_>>(), vec![b]);
        assert_eq!(ledger.audit_cohort(&rh), Some(vec![b]));
    }

    /// H8 routing: the first identity credited with an accepted response wins the reward,
    /// exactly once, with its payout script resolved from the folded producers; later
    /// responders win nothing, and pre-gate requests emit no reward.
    #[test]
    fn first_accepted_responder_wins_the_reward_once() {
        let a = Hash::from_bytes([1u8; 32]);
        let b = Hash::from_bytes([2u8; 32]);
        let set = [a, b];
        let spk_a = crate::tx::ScriptPublicKey::from_vec(0, vec![0xAA; 34]);
        let mut ledger = ServiceLedger::default();
        ledger.set_window_v2_activation(0);
        ledger.set_reward_routing_activation(200);

        // Pre-gate request: routed emission stays off even with rewards supplied.
        let rh_old = [6u8; 32];
        let out = ledger.on_chain_block_with_rewards(
            100,
            &[(rh_old, 0, 256)],
            &[(rh_old, 5_000)],
            &[],
            &[],
            &[(a, spk_a.clone())],
            |_| true,
            cohort_of(&set),
        );
        assert!(out.rewards.is_empty());
        let out =
            ledger.on_chain_block_with_rewards(101, &[], &[], &[(rh_old, Some(a))], &[], &[], |_| true, cohort_of(&set));
        assert!(out.rewards.is_empty());

        // Post-gate request: the first credited responder wins, once, with a resolved script.
        let rh = [7u8; 32];
        let out = ledger.on_chain_block_with_rewards(
            300,
            &[(rh, 0, 256)],
            &[(rh, 9_000)],
            &[],
            &[],
            &[],
            |_| true,
            cohort_of(&set),
        );
        assert!(out.rewards.is_empty());
        // First accepted response after arming: `b` wins (no known script — stays burned).
        let out = ledger.on_chain_block_with_rewards(302, &[], &[], &[(rh, Some(b))], &[], &[], |_| true, cohort_of(&set));
        assert_eq!(out.rewards.len(), 1);
        assert_eq!(out.rewards[0], ServiceReward { request_hash: rh, winner: b, amount: 9_000, spk: None });
        // A later responder wins nothing.
        let out = ledger.on_chain_block_with_rewards(303, &[], &[], &[(rh, Some(a))], &[], &[], |_| true, cohort_of(&set));
        assert!(out.rewards.is_empty());

        // A winner with a folded producer script gets it resolved.
        let rh2 = [8u8; 32];
        ledger.on_chain_block_with_rewards(
            310,
            &[(rh2, 0, 256)],
            &[(rh2, 4_000)],
            &[],
            &[],
            &[(a, spk_a.clone())],
            |_| true,
            cohort_of(&set),
        );
        let out = ledger.on_chain_block_with_rewards(312, &[], &[], &[(rh2, Some(a))], &[], &[], |_| true, cohort_of(&set));
        assert_eq!(out.rewards.len(), 1);
        assert_eq!(out.rewards[0].spk, Some(spk_a));
    }

    /// Past the gate, re-accepting a hash already admitted must not open a second audit: the
    /// cohort that served it cannot serve it again, so the repeat would strike everyone for an
    /// assignment nobody can honour. Before the gate the old behaviour is untouched.
    #[test]
    fn a_repeat_request_hash_never_arms_a_second_audit() {
        let a = Hash::from_bytes([1u8; 32]);
        let set = [a];
        let rh = [7u8; 32];
        let w = service_window_daa(0, 256);

        // Pre-gate: the repeat is admitted exactly as before.
        let mut legacy = ServiceLedger::default();
        legacy.on_chain_block(100, &[(rh, 0, 256)], &[], &[], |_| true, cohort_of(&set));
        legacy.on_chain_block(101, &[], &[(rh, Some(a))], &[], |_| true, cohort_of(&set));
        legacy.on_chain_block(102 + w, &[], &[], &[], |_| true, cohort_of(&set));
        assert_eq!(legacy.pending_len(), 0);
        legacy.on_chain_block(103 + w, &[(rh, 0, 256)], &[], &[], |_| true, cohort_of(&set));
        assert_eq!(legacy.pending_len(), 1, "pre-gate behaviour must be unchanged");

        // Post-gate: the first audit runs and closes served; the repeat is ignored, so the
        // cohort is never struck for it.
        let mut ledger = ServiceLedger::default();
        ledger.set_reward_routing_activation(100);
        ledger.on_chain_block(100, &[(rh, 0, 256)], &[], &[], |_| true, cohort_of(&set));
        ledger.on_chain_block(101, &[], &[(rh, Some(a))], &[], |_| true, cohort_of(&set));
        assert!(ledger.on_chain_block(102 + w, &[], &[], &[], |_| true, cohort_of(&set)).misses.is_empty());
        assert_eq!(ledger.pending_len(), 0);

        ledger.on_chain_block(103 + w, &[(rh, 0, 256)], &[], &[], |_| true, cohort_of(&set));
        assert_eq!(ledger.pending_len(), 0, "a remembered hash must not be admitted again");
        let misses = ledger.on_chain_block(104 + 2 * w, &[], &[], &[], |_| true, cohort_of(&set)).misses;
        assert!(misses.is_empty(), "the repeat must strike nobody");
    }

    #[test]
    fn answer_in_the_accepting_block_never_strikes() {
        // A miner fast enough to answer inside the chain block that accepts the request used to
        // have his answer dropped — no audit existed yet to credit it — and was struck for a
        // request he had served. Observed on testnet at daa 9132, cohort of one.
        let a = Hash::from_bytes([1u8; 32]);
        let set = [a];
        let mut ledger = ServiceLedger::default();
        let w = service_window_daa(3, 128);

        let rh = [7u8; 32];
        // request and its answer accepted by the same chain block
        let out = ledger.on_chain_block(9132, &[(rh, 3, 128)], &[(rh, Some(a))], &[], |_| true, cohort_of(&set));
        assert!(out.misses.is_empty());
        // the audit arms on the next block and must open already satisfied
        assert!(ledger.on_chain_block(9133, &[], &[], &[], |_| true, cohort_of(&set)).misses.is_empty());
        let misses = ledger.on_chain_block(9134 + w, &[], &[], &[], |_| true, cohort_of(&set)).misses;
        assert!(misses.is_empty(), "an answered request must never strike its answerer");
        assert_eq!(ledger.consecutive_misses(&a), 0);
        assert_eq!(ledger.pending_len(), 0);
    }

    #[test]
    fn answer_accepted_before_its_own_request_never_strikes() {
        // An AiResponse has no inputs, so nothing orders its acceptance against its request's:
        // the selected chain can accept the answer first. It is held until the request lands.
        let a = Hash::from_bytes([1u8; 32]);
        let set = [a];
        let mut ledger = ServiceLedger::default();
        let w = service_window_daa(0, 256);

        let rh = [7u8; 32];
        // the answer is accepted two chain blocks before the request it answers
        assert!(ledger.on_chain_block(100, &[], &[(rh, Some(a))], &[], |_| true, cohort_of(&set)).misses.is_empty());
        assert!(ledger.on_chain_block(101, &[], &[], &[], |_| true, cohort_of(&set)).misses.is_empty());
        ledger.on_chain_block(102, &[(rh, 0, 256)], &[], &[], |_| true, cohort_of(&set));
        ledger.on_chain_block(103, &[], &[], &[], |_| true, cohort_of(&set));
        let misses = ledger.on_chain_block(104 + w, &[], &[], &[], |_| true, cohort_of(&set)).misses;
        assert!(misses.is_empty(), "an answer that beat its request must still count");
        assert_eq!(ledger.consecutive_misses(&a), 0);
    }

    #[test]
    fn early_answer_expires_with_the_merge_depth_horizon() {
        // Held answers are not kept forever: past the merge-depth horizon the request can no
        // longer be accepted, so the parked answer is dropped.
        let a = Hash::from_bytes([1u8; 32]);
        let set = [a];
        let mut ledger = ServiceLedger::default();
        let w = service_window_daa(0, 256);

        let rh = [7u8; 32];
        ledger.on_chain_block(100, &[], &[(rh, Some(a))], &[], |_| true, cohort_of(&set));
        let late = 100 + SERVICE_EARLY_RESPONSE_HORIZON_DAA + 1;
        ledger.on_chain_block(late, &[(rh, 0, 256)], &[], &[], |_| true, cohort_of(&set));
        ledger.on_chain_block(late + 1, &[], &[], &[], |_| true, cohort_of(&set));
        let misses = ledger.on_chain_block(late + 2 + w, &[], &[], &[], |_| true, cohort_of(&set)).misses;
        assert_eq!(misses.len(), 1, "a stale parked answer must not cancel a fresh request");
    }

    #[test]
    fn only_the_silent_member_of_the_cohort_is_struck() {
        // Cohort of three answering in the accepting block: A and B serve, C stays silent.
        // C alone takes the strike.
        let a = Hash::from_bytes([1u8; 32]);
        let b = Hash::from_bytes([2u8; 32]);
        let c = Hash::from_bytes([3u8; 32]);
        let set = [a, b, c];
        let mut ledger = ServiceLedger::default();
        let w = service_window_daa(0, 256);

        let rh = [7u8; 32];
        ledger.on_chain_block(100, &[(rh, 0, 256)], &[(rh, Some(a)), (rh, Some(b))], &[], |_| true, cohort_of(&set));
        ledger.on_chain_block(101, &[], &[], &[], |_| true, cohort_of(&set));
        let misses = ledger.on_chain_block(102 + w, &[], &[], &[], |_| true, cohort_of(&set)).misses;
        assert_eq!(misses.len(), 1);
        assert_eq!(misses[0].miner, c);
        assert_eq!(ledger.consecutive_misses(&a), 0);
        assert_eq!(ledger.consecutive_misses(&b), 0);
        assert_eq!(ledger.consecutive_misses(&c), 1);
    }

    #[test]
    fn strikes_are_rate_limited_to_one_per_interval() {
        // Two silent requests inside one strike interval yield only ONE strike — the guard-rail
        // that stops a burst (or a brief outage) from escalating a miner to suspension.
        let a = Hash::from_bytes([1u8; 32]);
        let set = [a];
        let mut ledger = ServiceLedger::default();
        let w = service_window_daa(0, 256);

        let r1 = [1u8; 32];
        ledger.on_chain_block(100, &[(r1, 0, 256)], &[], &[], |_| true, cohort_of(&set));
        ledger.on_chain_block(101, &[], &[], &[], |_| true, cohort_of(&set));
        let misses = ledger.on_chain_block(102 + w, &[], &[], &[], |_| true, cohort_of(&set)).misses;
        assert_eq!(misses.len(), 1); // strike 1

        // a second miss well within the interval: no strike, no escalation
        let d2 = 200 + w;
        let r2 = [2u8; 32];
        ledger.on_chain_block(d2, &[(r2, 0, 256)], &[], &[], |_| true, cohort_of(&set));
        ledger.on_chain_block(d2 + 1, &[], &[], &[], |_| true, cohort_of(&set));
        let misses = ledger.on_chain_block(d2 + 2 + w, &[], &[], &[], |_| true, cohort_of(&set)).misses;
        assert!(misses.is_empty(), "a miss inside the interval must not strike");
        assert_eq!(ledger.consecutive_misses(&a), 1);
    }

    #[test]
    fn reaccepted_request_keeps_its_armed_audit() {
        // Re-accepting a known hash (identical payload resubmitted) must not reset the
        // audit clock: the original window still expires and the miss still fires.
        let a = Hash::from_bytes([1u8; 32]);
        let set = [a];
        let mut ledger = ServiceLedger::default();
        let w = service_window_daa(0, 256);

        let r1 = [1u8; 32];
        ledger.on_chain_block(100, &[(r1, 0, 256)], &[], &[], |_| true, cohort_of(&set));
        ledger.on_chain_block(101, &[], &[], &[], |_| true, cohort_of(&set));
        // the same hash lands again mid-window: the running audit must survive
        ledger.on_chain_block(102, &[(r1, 0, 256)], &[], &[], |_| true, cohort_of(&set));
        let misses = ledger.on_chain_block(102 + w, &[], &[], &[], |_| true, cohort_of(&set)).misses;
        assert_eq!(misses.len(), 1, "the original window must still expire");
    }

    #[test]
    fn cohort_escalates_across_intervals_to_suspension() {
        let a = Hash::from_bytes([1u8; 32]);
        let b = Hash::from_bytes([2u8; 32]);
        let set = [a, b];
        let mut ledger = ServiceLedger::default();
        let w = service_window_daa(0, 256);
        let i = SERVICE_STRIKE_INTERVAL_DAA;

        // round 1: whole cohort silent -> both strike 1
        let r1 = [1u8; 32];
        ledger.on_chain_block(100, &[(r1, 0, 256)], &[], &[], |_| true, cohort_of(&set));
        ledger.on_chain_block(101, &[], &[], &[], |_| true, cohort_of(&set));
        let misses = ledger.on_chain_block(102 + w, &[], &[], &[], |_| true, cohort_of(&set)).misses;
        assert_eq!(misses.len(), 2);
        assert_eq!((misses[0].miner, misses[0].consecutive_misses), (a, 1));
        assert_eq!(misses[0].penalty, ServicePenalty::BurnClaims(STRIKE_1_BURN_CLAIMS));

        // round 2, one interval later: a serves -> only b strikes, to strike 2
        let d2 = 200 + i;
        let r2 = [2u8; 32];
        ledger.on_chain_block(d2, &[(r2, 0, 256)], &[], &[], |_| true, cohort_of(&set));
        ledger.on_chain_block(d2 + 1, &[], &[], &[], |_| true, cohort_of(&set));
        let outcome = ledger.on_chain_block(d2 + 2, &[], &[(r2, Some(a))], &[], |_| true, cohort_of(&set));
        assert_eq!(outcome.resets, vec![(a, 0)], "a's serve must reset (and report) his streak");
        let misses = ledger.on_chain_block(d2 + 2 + w, &[], &[], &[], |_| true, cohort_of(&set)).misses;
        assert_eq!(misses.len(), 1);
        assert_eq!((misses[0].miner, misses[0].consecutive_misses), (b, 2));
        assert_eq!(misses[0].penalty, ServicePenalty::SlashAllPending);
        assert_eq!(ledger.consecutive_misses(&a), 0);

        // round 3, another interval later: a (reset) restarts at 1, b reaches strike 3 -> Suspend
        let d3 = d2 + i + 100;
        let r3 = [3u8; 32];
        ledger.on_chain_block(d3, &[(r3, 0, 256)], &[], &[], |_| true, cohort_of(&set));
        ledger.on_chain_block(d3 + 1, &[], &[], &[], |_| true, cohort_of(&set));
        let misses = ledger.on_chain_block(d3 + 2 + w, &[], &[], &[], |_| true, cohort_of(&set)).misses;
        assert_eq!(misses.len(), 2);
        assert_eq!((misses[0].miner, misses[0].consecutive_misses), (a, 1));
        assert_eq!((misses[1].miner, misses[1].consecutive_misses), (b, 3));
        assert_eq!(misses[1].penalty, ServicePenalty::Suspend);
    }

    #[test]
    fn late_response_in_closing_block_cancels_the_miss() {
        let a = Hash::from_bytes([1u8; 32]);
        let set = [a];
        let mut ledger = ServiceLedger::default();
        let w = service_window_daa(0, 256);

        let rh = [9u8; 32];
        ledger.on_chain_block(100, &[(rh, 0, 256)], &[], &[], |_| true, cohort_of(&set));
        ledger.on_chain_block(101, &[], &[], &[], |_| true, cohort_of(&set));
        // window is past, but the closing block itself carries the response: no miss
        assert!(ledger.on_chain_block(200 + w, &[], &[(rh, Some(a))], &[], |_| true, cohort_of(&set)).misses.is_empty());
        assert_eq!(ledger.pending_len(), 0);
    }

    #[test]
    fn empty_cohort_drops_the_request() {
        let mut ledger = ServiceLedger::default();
        let rh = [5u8; 32];
        assert!(ledger.on_chain_block(100, &[(rh, 0, 256)], &[], &[], |_| true, |_tier| Vec::new()).misses.is_empty());
        assert!(ledger.on_chain_block(101, &[], &[], &[], |_| true, |_tier| Vec::new()).misses.is_empty());
        assert_eq!(ledger.pending_len(), 0);
    }

    #[test]
    fn penalties_burn_newest_claims_then_drain() {
        use super::EscrowClaim;
        use crate::tx::TransactionOutpoint;

        let a = Hash::from_bytes([1u8; 32]);
        let set = [a];
        let mut ledger = ServiceLedger::default();
        let w = service_window_daa(0, 256);
        let claim = |n: u64, daa: u64| EscrowClaim { outpoint: TransactionOutpoint::new(n.into(), 1), value: n, daa };

        // six claims accumulated, then r1 missed: strike 1 burns the 5 NEWEST
        let escrows: Vec<(Hash, EscrowClaim)> = (1..=6).map(|n| (a, claim(n, 100))).collect();
        let r1 = [1u8; 32];
        ledger.on_chain_block(100, &[(r1, 0, 256)], &[], &escrows, |_| true, cohort_of(&set));
        ledger.on_chain_block(101, &[], &[], &[], |_| true, cohort_of(&set));
        let misses = ledger.on_chain_block(102 + w, &[], &[], &[], |_| true, cohort_of(&set)).misses;
        assert_eq!(misses.len(), 1);
        assert_eq!(misses[0].penalty, ServicePenalty::BurnClaims(STRIKE_1_BURN_CLAIMS));
        assert_eq!(misses[0].burned.iter().map(|c| c.value).collect::<Vec<_>>(), vec![6, 5, 4, 3, 2]);

        // r2 (one interval later) missed too: strike 2 drains the leftover claim plus one
        // accumulated meanwhile
        let i = SERVICE_STRIKE_INTERVAL_DAA;
        let d2 = 200 + i;
        let r2 = [2u8; 32];
        let fresh = [(a, claim(7, d2))];
        ledger.on_chain_block(d2, &[(r2, 0, 256)], &[], &fresh, |_| true, cohort_of(&set));
        ledger.on_chain_block(d2 + 1, &[], &[], &[], |_| true, cohort_of(&set));
        let misses = ledger.on_chain_block(d2 + 2 + w, &[], &[], &[], |_| true, cohort_of(&set)).misses;
        assert_eq!(misses.len(), 1);
        assert_eq!(misses[0].penalty, ServicePenalty::SlashAllPending);
        assert_eq!(misses[0].burned.iter().map(|c| c.value).collect::<Vec<_>>(), vec![7, 1]);

        // r3 (another interval later): strike 3 (Suspend) takes claims re-accumulated past the drain
        let d3 = d2 + i + 100;
        let r3 = [3u8; 32];
        let fresh = [(a, claim(8, d3))];
        ledger.on_chain_block(d3, &[(r3, 0, 256)], &[], &fresh, |_| true, cohort_of(&set));
        ledger.on_chain_block(d3 + 1, &[], &[], &[], |_| true, cohort_of(&set));
        let misses = ledger.on_chain_block(d3 + 2 + w, &[], &[], &[], |_| true, cohort_of(&set)).misses;
        assert_eq!(misses.len(), 1);
        assert_eq!(misses[0].penalty, ServicePenalty::Suspend);
        assert_eq!(misses[0].burned.iter().map(|c| c.value).collect::<Vec<_>>(), vec![8]);
    }

    #[test]
    fn horizon_expires_pendings_but_strikes_persist() {
        let a = Hash::from_bytes([1u8; 32]);
        let set = [a];
        let mut ledger = ServiceLedger::default();
        let w = service_window_daa(0, 256);

        let rh = [9u8; 32];
        ledger.on_chain_block(100, &[(rh, 0, 256)], &[], &[], |_| true, cohort_of(&set));
        ledger.on_chain_block(101, &[], &[], &[], |_| true, cohort_of(&set));
        let misses = ledger.on_chain_block(102 + w, &[], &[], &[], |_| true, cohort_of(&set)).misses;
        assert_eq!(misses.len(), 1);
        assert_eq!(ledger.consecutive_misses(&a), 1);

        // far beyond the horizon: request/vault memory is gone, the strike count is not
        let far = 102 + w + 4 * SERVICE_LEDGER_HORIZON_DAA;
        ledger.on_chain_block(far, &[], &[], &[], |_| true, cohort_of(&set));
        assert_eq!(ledger.pending_len(), 0);
        assert_eq!(ledger.consecutive_misses(&a), 1);
        assert_eq!(ledger.strike_entries().len(), 1);
    }

    #[test]
    fn strikes_reset_only_on_serve() {
        let a = Hash::from_bytes([1u8; 32]);
        let set = [a];
        let mut ledger = ServiceLedger::default();
        let w = service_window_daa(0, 256);

        let r1 = [1u8; 32];
        ledger.on_chain_block(100, &[(r1, 0, 256)], &[], &[], |_| true, cohort_of(&set));
        ledger.on_chain_block(101, &[], &[], &[], |_| true, cohort_of(&set));
        assert_eq!(ledger.on_chain_block(102 + w, &[], &[], &[], |_| true, cohort_of(&set)).misses.len(), 1);

        // only a served response resets — and the reset is reported for persistence
        let far = 102 + w + 4 * SERVICE_LEDGER_HORIZON_DAA;
        let r2 = [2u8; 32];
        ledger.on_chain_block(far, &[(r2, 0, 256)], &[], &[], |_| true, cohort_of(&set));
        ledger.on_chain_block(far + 1, &[], &[], &[], |_| true, cohort_of(&set));
        let outcome = ledger.on_chain_block(far + 2, &[], &[(r2, Some(a))], &[], |_| true, cohort_of(&set));
        assert_eq!(outcome.resets, vec![(a, 0)]);
        assert_eq!(ledger.consecutive_misses(&a), 0);
        assert!(ledger.strike_entries().is_empty());
    }

    #[test]
    fn serve_keeps_the_rate_limit_armed_post_v2() {
        let a = Hash::from_bytes([1u8; 32]);
        let set = [a];
        let mut ledger = ServiceLedger::default();
        ledger.set_window_v2_activation(0);
        let w = service_window_daa_at(0, 256, true);
        let i = SERVICE_STRIKE_INTERVAL_DAA;

        let r1 = [1u8; 32];
        ledger.on_chain_block(100, &[(r1, 0, 256)], &[], &[], |_| true, cohort_of(&set));
        ledger.on_chain_block(101, &[], &[], &[], |_| true, cohort_of(&set));
        let strike_daa = 102 + w;
        assert_eq!(ledger.on_chain_block(strike_daa, &[], &[], &[], |_| true, cohort_of(&set)).misses.len(), 1);

        // the serve resets the streak but reports the preserved strike daa
        let r2 = [2u8; 32];
        ledger.on_chain_block(strike_daa + 10, &[(r2, 0, 256)], &[], &[], |_| true, cohort_of(&set));
        ledger.on_chain_block(strike_daa + 11, &[], &[], &[], |_| true, cohort_of(&set));
        let outcome = ledger.on_chain_block(strike_daa + 12, &[], &[(r2, Some(a))], &[], |_| true, cohort_of(&set));
        assert_eq!(outcome.resets, vec![(a, strike_daa)]);
        assert_eq!(ledger.consecutive_misses(&a), 0);

        // a miss closing inside the pre-serve strike's interval is still absorbed
        let r3 = [3u8; 32];
        ledger.on_chain_block(strike_daa + 20, &[(r3, 0, 256)], &[], &[], |_| true, cohort_of(&set));
        ledger.on_chain_block(strike_daa + 21, &[], &[], &[], |_| true, cohort_of(&set));
        assert!(ledger.on_chain_block(strike_daa + 22 + w, &[], &[], &[], |_| true, cohort_of(&set)).misses.is_empty());

        // past the interval it strikes again, from 1
        let d4 = strike_daa + i + 100;
        let r4 = [4u8; 32];
        ledger.on_chain_block(d4, &[(r4, 0, 256)], &[], &[], |_| true, cohort_of(&set));
        ledger.on_chain_block(d4 + 1, &[], &[], &[], |_| true, cohort_of(&set));
        let misses = ledger.on_chain_block(d4 + 2 + w, &[], &[], &[], |_| true, cohort_of(&set)).misses;
        assert_eq!(misses.len(), 1);
        assert_eq!(misses[0].consecutive_misses, 1);
    }

    #[test]
    fn suspend_resets_the_streak() {
        let a = Hash::from_bytes([1u8; 32]);
        let set = [a];
        let mut ledger = ServiceLedger::default();
        let w = service_window_daa(0, 256);
        let i = SERVICE_STRIKE_INTERVAL_DAA;

        let mut last_penalty = ServicePenalty::None;
        for k in 0..4u64 {
            let d = 1_000 + k * (i + 10);
            let rh = [k as u8 + 1; 32];
            ledger.on_chain_block(d, &[(rh, 0, 256)], &[], &[], |_| true, cohort_of(&set));
            ledger.on_chain_block(d + 1, &[], &[], &[], |_| true, cohort_of(&set));
            let misses = ledger.on_chain_block(d + 2 + w, &[], &[], &[], |_| true, cohort_of(&set)).misses;
            assert_eq!(misses.len(), 1);
            last_penalty = misses[0].penalty;
            if k == 2 {
                assert_eq!(last_penalty, ServicePenalty::Suspend);
                // the executed suspension restarts the streak…
                assert_eq!(ledger.consecutive_misses(&a), 0);
            }
        }
        // …so the fourth miss escalates from one, not into an endless re-suspension
        assert_eq!(last_penalty, ServicePenalty::BurnClaims(STRIKE_1_BURN_CLAIMS));
    }

    #[test]
    fn refold_with_base_and_warmup_matches_incremental() {
        // The cold-start invariant: a refold that overlays the persisted baseline, warms
        // request/vault memory below the store frontier (dropping claims through the persisted
        // burn set) and folds normally above it must reproduce the incremental state and
        // re-emit exactly the events above the frontier.
        use super::EscrowClaim;
        use crate::tx::TransactionOutpoint;

        let a = Hash::from_bytes([1u8; 32]);
        let set = [a];
        let w = service_window_daa(0, 256);
        let i = SERVICE_STRIKE_INTERVAL_DAA;
        let claim = |n: u64, daa: u64| EscrowClaim { outpoint: TransactionOutpoint::new(n.into(), 1), value: n, daa };

        // Four rounds spaced one interval apart: miss, miss, serve, miss.
        let rounds: Vec<(u64, u8, bool)> = (0..4u64).map(|k| (1_000 + k * (i + 10), k as u8 + 1, k == 2)).collect();
        let fold_round =
            |ledger: &mut ServiceLedger, d: u64, n: u8, served: bool, warm: Option<&dyn Fn(&TransactionOutpoint) -> bool>| {
                let rh = [n; 32];
                let escrows = [(a, claim(n as u64, d))];
                let mut outcomes = Vec::new();
                match warm {
                    Some(is_burned) => {
                        ledger.on_chain_block_warmup(d, &[(rh, 0, 256)], &[], &escrows, is_burned, cohort_of(&set));
                        ledger.on_chain_block_warmup(d + 1, &[], &[], &[], is_burned, cohort_of(&set));
                        if served {
                            ledger.on_chain_block_warmup(d + 2, &[], &[(rh, Some(a))], &[], is_burned, cohort_of(&set));
                        }
                        ledger.on_chain_block_warmup(d + 2 + w, &[], &[], &[], is_burned, cohort_of(&set));
                    }
                    None => {
                        outcomes.push((d, ledger.on_chain_block(d, &[(rh, 0, 256)], &[], &escrows, |_| true, cohort_of(&set))));
                        outcomes.push((d + 1, ledger.on_chain_block(d + 1, &[], &[], &[], |_| true, cohort_of(&set))));
                        if served {
                            outcomes.push((d + 2, ledger.on_chain_block(d + 2, &[], &[(rh, Some(a))], &[], |_| true, cohort_of(&set))));
                        }
                        outcomes.push((d + 2 + w, ledger.on_chain_block(d + 2 + w, &[], &[], &[], |_| true, cohort_of(&set))));
                    }
                }
                outcomes
            };

        // Incremental fold over everything, collecting the persisted trace as it goes.
        let mut inc = ServiceLedger::default();
        let mut trace: Vec<(u64, FoldOutcome)> = Vec::new();
        for &(d, n, served) in rounds.iter() {
            trace.extend(fold_round(&mut inc, d, n, served, None));
        }

        // "Persist" everything up to and including round 2's events (the store frontier).
        let cursor = rounds[1].0 + 2 + w;
        let mut base = std::collections::BTreeMap::new();
        let mut first_seen_base = std::collections::BTreeMap::new();
        let mut burned_set = std::collections::HashSet::new();
        for (d, outcome) in trace.iter().filter(|(d, _)| *d <= cursor) {
            for miner in outcome.sightings.iter() {
                first_seen_base.entry(*miner).or_insert(*d);
            }
            for miss in outcome.misses.iter() {
                let record = if miss.penalty == ServicePenalty::Suspend {
                    StrikeEntry { count: 0, last_daa: *d }
                } else {
                    StrikeEntry { count: miss.consecutive_misses, last_daa: *d }
                };
                base.insert(miss.miner, record);
                for c in miss.burned.iter() {
                    burned_set.insert(c.outpoint);
                }
            }
            for (miner, preserved) in outcome.resets.iter() {
                base.insert(*miner, StrikeEntry { count: 0, last_daa: *preserved });
            }
        }

        // Refold: baseline + warmup below the frontier, normal fold above it.
        let mut refolded = ServiceLedger::default();
        refolded.set_base(std::sync::Arc::new(base));
        refolded.set_first_seen_base(std::sync::Arc::new(first_seen_base));
        let is_burned = |op: &TransactionOutpoint| burned_set.contains(op);
        let mut replayed: Vec<(u64, FoldOutcome)> = Vec::new();
        for &(d, n, served) in rounds.iter() {
            if d + 2 + w <= cursor {
                fold_round(&mut refolded, d, n, served, Some(&is_burned));
            } else {
                replayed.extend(fold_round(&mut refolded, d, n, served, None));
            }
        }

        let _to = rounds.last().unwrap().0 + 2 + w;
        assert_eq!(inc.consecutive_misses(&a), refolded.consecutive_misses(&a));
        assert_eq!(inc.strike_entries(), refolded.strike_entries());
        assert_eq!(inc.vault_claims(&a), refolded.vault_claims(&a));
        let events_above = |t: &[(u64, FoldOutcome)]| {
            t.iter()
                .filter(|(d, _)| *d > cursor)
                .flat_map(|(d, o)| o.misses.iter().map(move |m| (*d, m.miner, m.consecutive_misses, m.penalty)))
                .collect::<Vec<_>>()
        };
        assert_eq!(events_above(&trace), events_above(&replayed), "events above the frontier must replay identically");

        // The effective state is the same image from both paths, byte for byte.
        let snap = inc.snapshot();
        assert_eq!(snap, refolded.snapshot());
        let bytes = snap.to_bytes();
        let decoded = super::ServiceLedgerSnapshot::from_bytes(&bytes).unwrap();
        assert_eq!(decoded, snap);
        assert_eq!(decoded.hash(), refolded.snapshot().hash());
        assert!(super::ServiceLedgerSnapshot::from_bytes(&bytes[..bytes.len() - 1]).is_err());

        // A ledger restored from the image continues exactly like the incremental one.
        let mut restored = ServiceLedger::default();
        restored.restore_snapshot(&snap);
        let d = rounds.last().unwrap().0 + i + 10;
        let from_inc = fold_round(&mut inc, d, 9, false, None);
        let from_restored = fold_round(&mut restored, d, 9, false, None);
        assert_eq!(events_above(&from_inc), events_above(&from_restored));
        assert_eq!(inc.snapshot(), restored.snapshot());
    }

    #[test]
    fn eligible_pairs_distinct_sorted_by_tier() {
        let a = Hash::from_bytes([1u8; 32]);
        let b = Hash::from_bytes([2u8; 32]);
        let c = Hash::from_bytes([3u8; 32]);
        let e = Hash::from_bytes([9u8; 32]);
        // a: tier 0 (twice, same delegation), b: tier 0, c: tier 1
        let recent = [(a, 0u8, e), (c, 1u8, e), (a, 0u8, e), (b, 0u8, b)];
        assert_eq!(eligible_pairs(&recent, 0), vec![(a, e), (b, b)]);
        assert_eq!(eligible_pairs(&recent, 1), vec![(c, e)]);
        assert!(eligible_pairs(&recent, 4).is_empty());
    }

    #[test]
    fn response_credits_the_delegating_identity() {
        // Identity (payout SPK key) and escrow (hot) key are distinct: a response signed by the
        // escrow key must credit the delegating identity — and every identity sharing that key.
        let id_a = Hash::from_bytes([1u8; 32]);
        let id_b = Hash::from_bytes([2u8; 32]);
        let esc = Hash::from_bytes([0xEEu8; 32]);
        let mut ledger = ServiceLedger::default();
        let w = service_window_daa(0, 256);
        let pairs = [(id_a, esc), (id_b, esc)];
        let cohort = |_tier: u8| pairs.to_vec();

        let rh = [7u8; 32];
        ledger.on_chain_block(100, &[(rh, 0, 256)], &[], &[], |_| true, cohort);
        ledger.on_chain_block(101, &[], &[], &[], |_| true, cohort);
        // the hot key's response serves BOTH delegating identities
        assert!(ledger.on_chain_block(102, &[], &[(rh, Some(esc))], &[], |_| true, cohort).misses.is_empty());
        assert!(ledger.on_chain_block(102 + w, &[], &[], &[], |_| true, cohort).misses.is_empty());
        assert_eq!(ledger.pending_len(), 0);

        // a response by a key nobody delegated to never counts
        let r2 = [8u8; 32];
        let stranger = Hash::from_bytes([0x55u8; 32]);
        ledger.on_chain_block(200 + w, &[(r2, 0, 256)], &[], &[], |_| true, cohort);
        ledger.on_chain_block(201 + w, &[], &[], &[], |_| true, cohort);
        ledger.on_chain_block(202 + w, &[], &[(r2, Some(stranger))], &[], |_| true, cohort);
        let misses = ledger.on_chain_block(202 + 2 * w, &[], &[], &[], |_| true, cohort).misses;
        assert_eq!(misses.len(), 2, "both identities must miss");
        assert_eq!((misses[0].miner, misses[1].miner), (id_a, id_b));
    }

    #[test]
    fn escrow_delegation_cert_roundtrip() {
        use super::{escrow_delegation_message, parse_escrow_esig, parse_escrow_pubkey, verify_escrow_delegation};

        let payout = secp256k1::Keypair::from_seckey_slice(secp256k1::SECP256K1, &[0xA1u8; 32]).unwrap();
        let escrow = secp256k1::Keypair::from_seckey_slice(secp256k1::SECP256K1, &[0xB2u8; 32]).unwrap();
        let escrow_pubkey = escrow.x_only_public_key().0.serialize();
        let msg = secp256k1::Message::from_digest(escrow_delegation_message(&escrow_pubkey));
        let sig = *payout.sign_schnorr(msg).as_ref();

        // standard schnorr P2PK payout script: 0x20 <key32> OP_CHECKSIG
        let mut script = vec![0x20u8];
        script.extend_from_slice(&payout.x_only_public_key().0.serialize());
        script.push(0xac);

        assert!(verify_escrow_delegation(0, &script, &escrow_pubkey, &sig));
        // wrong version, tampered escrow key, wrong signer, malformed script — all rejected
        assert!(!verify_escrow_delegation(1, &script, &escrow_pubkey, &sig));
        assert!(!verify_escrow_delegation(0, &script, &[0x11u8; 32], &sig));
        let mut other_script = vec![0x20u8];
        other_script.extend_from_slice(&escrow.x_only_public_key().0.serialize());
        other_script.push(0xac);
        assert!(!verify_escrow_delegation(0, &other_script, &escrow_pubkey, &sig));
        assert!(!verify_escrow_delegation(0, &script[..33], &escrow_pubkey, &sig));

        // extra_data wire form: both segments parse back to the same bytes
        let extra = format!(
            "0.5.0/2608131047/escrow:{}/esig:{}/ai:v1:0011223344556677",
            faster_hex::hex_string(&escrow_pubkey),
            faster_hex::hex_string(&sig)
        );
        assert_eq!(parse_escrow_pubkey(extra.as_bytes()), Some(escrow_pubkey));
        assert_eq!(parse_escrow_esig(extra.as_bytes()), Some(sig));
        assert_eq!(parse_escrow_esig(b"0.5.0/escrow:aabb"), None);
    }

    #[test]
    fn vault_undo_reverses_a_folded_block() {
        use super::{EscrowClaim, SERVICE_BURNABLE_WINDOW_DAA};
        use crate::tx::TransactionOutpoint;

        let a = Hash::from_bytes([1u8; 32]);
        let set = [a];
        let mut ledger = ServiceLedger::default();
        let w = service_window_daa(0, 256);
        let claim = |n: u64, daa: u64| EscrowClaim { outpoint: TransactionOutpoint::new(n.into(), 1), value: n, daa };

        // block 1: six claims land
        let adds1: Vec<(Hash, EscrowClaim)> = (1..=6).map(|n| (a, claim(n, 100))).collect();
        let r1 = [1u8; 32];
        ledger.on_chain_block(100, &[(r1, 0, 256)], &[], &adds1, |_| true, cohort_of(&set));
        ledger.on_chain_block(101, &[], &[], &[], |_| true, cohort_of(&set));
        let before_burn = ledger.vault_claims(&a);

        // block 2: the audit closes — strike 1 burns the 5 newest; undo restores them exactly
        let out = ledger.on_chain_block(102 + w, &[], &[], &[], |_| true, cohort_of(&set));
        assert_eq!(out.misses.len(), 1);
        assert_eq!(ledger.vault_claims(&a).len(), 1);
        ledger.undo_vault(&[], &out.misses, &out.expired);
        assert_eq!(ledger.vault_claims(&a), before_burn);

        // block 3, one burnable-window later: one add, the six old claims expire — undo restores
        // both sides of the mutation
        let far = 100 + SERVICE_BURNABLE_WINDOW_DAA;
        let adds3 = [(a, claim(7, far))];
        let out = ledger.on_chain_block(far, &[], &[], &adds3, |_| true, cohort_of(&set));
        assert_eq!(out.expired.len(), 6, "the window purge must report what it dropped");
        assert_eq!(ledger.vault_claims(&a).iter().map(|c| c.value).collect::<Vec<_>>(), vec![7]);
        ledger.undo_vault(&adds3, &out.misses, &out.expired);
        assert_eq!(ledger.vault_claims(&a), before_burn);
    }

    #[test]
    fn young_identity_first_miss_slashes_everything() {
        use super::EscrowClaim;
        use crate::tx::TransactionOutpoint;

        let a = Hash::from_bytes([1u8; 32]);
        let set = [a];
        let mut ledger = ServiceLedger::default();
        let w = service_window_daa(0, 256);
        let claim = |n: u64, daa: u64| EscrowClaim { outpoint: TransactionOutpoint::new(n.into(), 1), value: n, daa };

        // seven claims accumulated, then a miss WITHOUT standing: the whole vault burns at once
        let escrows: Vec<(Hash, EscrowClaim)> = (1..=7).map(|n| (a, claim(n, 100))).collect();
        let r1 = [1u8; 32];
        ledger.on_chain_block(100, &[(r1, 0, 256)], &[], &escrows, |_| false, cohort_of(&set));
        ledger.on_chain_block(101, &[], &[], &[], |_| false, cohort_of(&set));
        let misses = ledger.on_chain_block(102 + w, &[], &[], &[], |_| false, cohort_of(&set)).misses;
        assert_eq!(misses.len(), 1);
        assert_eq!(misses[0].penalty, ServicePenalty::SlashAllPending);
        assert_eq!(misses[0].burned.len(), 7, "a young identity's first miss must drain everything");
    }

    #[test]
    fn young_identity_first_miss_burns_only_the_first_step_post_v2() {
        use super::EscrowClaim;
        use crate::tx::TransactionOutpoint;

        // Same fold as `young_identity_first_miss_slashes_everything`, gate armed: the
        // fold must read the new first step, not just the pure penalty table.
        let a = Hash::from_bytes([1u8; 32]);
        let set = [a];
        let mut ledger = ServiceLedger::default();
        ledger.set_window_v2_activation(0);
        let w = service_window_daa_at(0, 256, true);
        let claim = |n: u64, daa: u64| EscrowClaim { outpoint: TransactionOutpoint::new(n.into(), 1), value: n, daa };

        let escrows: Vec<(Hash, EscrowClaim)> = (1..=7).map(|n| (a, claim(n, 100))).collect();
        let r1 = [1u8; 32];
        ledger.on_chain_block(100, &[(r1, 0, 256)], &[], &escrows, |_| false, cohort_of(&set));
        ledger.on_chain_block(101, &[], &[], &[], |_| false, cohort_of(&set));
        let misses = ledger.on_chain_block(102 + w, &[], &[], &[], |_| false, cohort_of(&set)).misses;
        assert_eq!(misses.len(), 1);
        assert_eq!(misses[0].penalty, ServicePenalty::BurnClaims(STRIKE_1_BURN_CLAIMS));
        assert_eq!(misses[0].burned.len(), STRIKE_1_BURN_CLAIMS as usize);
        // The remaining claims stay locked instead of being drained.
        assert_eq!(ledger.vault_claims(&a).len(), 7 - STRIKE_1_BURN_CLAIMS as usize);
    }

    #[test]
    fn sightings_report_each_identity_once() {
        use super::EscrowClaim;
        use crate::tx::TransactionOutpoint;

        let a = Hash::from_bytes([1u8; 32]);
        let b = Hash::from_bytes([2u8; 32]);
        let set = [a];
        let mut ledger = ServiceLedger::default();
        // b is already in the persisted baseline: never reported again
        ledger.set_first_seen_base(std::sync::Arc::new([(b, 50u64)].into_iter().collect()));
        let claim = |n: u64, daa: u64| EscrowClaim { outpoint: TransactionOutpoint::new(n.into(), 1), value: n, daa };

        let out = ledger.on_chain_block(100, &[], &[], &[(a, claim(1, 100)), (b, claim(2, 100))], |_| true, cohort_of(&set));
        assert_eq!(out.sightings, vec![a]);
        let out = ledger.on_chain_block(200, &[], &[], &[(a, claim(3, 200))], |_| true, cohort_of(&set));
        assert!(out.sightings.is_empty(), "an identity is sighted once");
    }

    #[test]
    fn strike_penalty_escalation() {
        assert_eq!(strike_penalty(0, true), ServicePenalty::None);
        assert_eq!(strike_penalty(1, true), ServicePenalty::BurnClaims(STRIKE_1_BURN_CLAIMS));
        assert_eq!(strike_penalty(2, true), ServicePenalty::SlashAllPending);
        assert_eq!(strike_penalty(3, true), ServicePenalty::Suspend);
        assert_eq!(strike_penalty(9, true), ServicePenalty::Suspend);
        // an identity without standing skips the gentle first step
        assert_eq!(strike_penalty(1, false), ServicePenalty::SlashAllPending);
        assert_eq!(strike_penalty(2, false), ServicePenalty::SlashAllPending);
        assert_eq!(strike_penalty(3, false), ServicePenalty::Suspend);
    }

    #[test]
    fn strike_penalty_first_step_is_uniform_post_v2() {
        // The only cell the gate moves: first miss, identity without standing.
        assert_eq!(strike_penalty_at(1, false, false), ServicePenalty::SlashAllPending);
        assert_eq!(strike_penalty_at(1, false, true), ServicePenalty::BurnClaims(STRIKE_1_BURN_CLAIMS));

        // Every other cell reads the same on both sides of the gate.
        for established in [false, true] {
            for count in [0u32, 2, 3, 9] {
                assert_eq!(
                    strike_penalty_at(count, established, false),
                    strike_penalty_at(count, established, true),
                    "count {count}, established {established}"
                );
            }
            assert_eq!(strike_penalty_at(1, established, true), ServicePenalty::BurnClaims(STRIKE_1_BURN_CLAIMS));
        }

        // Escalation past the first miss is untouched: still slash-all then suspension.
        assert_eq!(strike_penalty_at(2, false, true), ServicePenalty::SlashAllPending);
        assert_eq!(strike_penalty_at(3, false, true), ServicePenalty::Suspend);

        // The pre-gate wrapper keeps the old table.
        assert_eq!(strike_penalty(1, false), strike_penalty_at(1, false, false));
    }

    #[test]
    fn strikes_reset_on_serve() {
        // miss, miss (→ ban territory) then serve resets, then a fresh miss is only strike 1
        let mut c = 0;
        for missed in [true, true, false, true] {
            c = update_strikes(c, missed);
        }
        assert_eq!(c, 1);
        assert_eq!(strike_penalty(c, true), ServicePenalty::BurnClaims(STRIKE_1_BURN_CLAIMS));
        // a long honest run of serves keeps it at 0
        for _ in 0..1000 {
            c = update_strikes(c, false);
        }
        assert_eq!(c, 0);
    }

    mod traces {
        use super::super::{ServiceLedger, ServiceLedgerSnapshot, TraceInputs, TraceOpeningEvent, TraceResponseEvent, trace_seed};
        use keryx_hashes::Hash;
        use crate::config::params::TraceRules;
        use keryx_verifiable_inference::{
            draw,
            testkit::{Model, Response},
        };

        const RH: [u8; 32] = [7u8; 32];
        const TXID: [u8; 32] = [0x51u8; 32];
        /// Second request and response of the root rule runs.
        const RH2: [u8; 32] = [8u8; 32];
        const TXID2: [u8; 32] = [0x52u8; 32];
        const ESCROW_PK: [u8; 32] = [0xA5u8; 32];

        fn miner() -> Hash {
            Hash::from_bytes([1u8; 32])
        }

        fn chain_hash(drawn: bool) -> Hash {
            (0u64..)
                .map(|i| {
                    let mut b = [0u8; 32];
                    b[..8].copy_from_slice(&i.to_le_bytes());
                    Hash::from_bytes(b)
                })
                .find(|h| draw::audited(&trace_seed(h, &TXID)) == drawn)
                .unwrap()
        }

        struct Run {
            ledger: ServiceLedger,
            model: Model,
            auditable: bool,
            /// Shape rule: `(prompt length, answer length)` of the request and its response.
            shape: Option<(u32, u32)>,
            /// Prefix rule: the prefix offset, and the request id the traced prompt carries.
            prefix: Option<(u32, [u8; 32])>,
            /// Audit rule: every traced response is audited, and the root rule admits a second
            /// request `RH2` at 100.
            audit_all: bool,
            /// Sanction rule: a missing opening strikes without the interval guard, the streak
            /// resets at the accepted opening.
            sanction: bool,
            /// Context rule: silence on a prompt that may exceed the tier floor is excused unless
            /// a member's traced response proved it fit.
            context: bool,
            /// Seed-bound opening rule.
            seed_bind: bool,
            /// Token budget of the request `RH`.
            max_tokens: u32,
            /// Cohort of tier 0.
            cohort: Vec<Hash>,
            /// `(n_prompt, n_pos)` of the traced response when set.
            positions: Option<(u32, u32)>,
            /// Attention rule: an audit also needs the attention opening.
            attention: bool,
        }

        impl Run {
            fn new(auditable: bool) -> Self {
                Self {
                    ledger: ServiceLedger::default(),
                    model: Model::new(),
                    auditable,
                    shape: None,
                    prefix: None,
                    audit_all: false,
                    sanction: false,
                    context: false,
                    seed_bind: false,
                    max_tokens: 256,
                    cohort: vec![miner()],
                    positions: None,
                    attention: false,
                }
            }

            /// Context rule over the shape rule: tier 0 floor 8 192, budget 256, so a prompt over
            /// 7 424 bytes is grey.
            fn context(prompt_len: u32) -> Self {
                Self { context: true, ..Self::shaped(prompt_len, 10) }
            }

            fn audited_all() -> Self {
                Self { audit_all: true, ..Self::new(true) }
            }

            fn sanctioned() -> Self {
                Self { sanction: true, ..Self::audited_all() }
            }

            fn shaped(prompt_len: u32, answer_len: u32) -> Self {
                Self { shape: Some((prompt_len, answer_len)), ..Self::new(true) }
            }

            /// Prefix rule at `offset`, traced prompt carrying `carried` as the request id.
            fn prefixed(offset: u32, carried: [u8; 32]) -> Self {
                Self { prefix: Some((offset, carried)), ..Self::shaped(64, 10) }
            }

            fn fold(&mut self, daa: u64, chain: Hash, responses: &[TraceResponseEvent], openings: &[TraceOpeningEvent]) -> Vec<(Hash, u32)> {
                self.fold_with_verdicts(daa, chain, responses, openings, &mut Vec::new())
            }

            /// [`Self::fold`] that also reports the causes of the openings it judged.
            fn fold_with_verdicts(
                &mut self,
                daa: u64,
                chain: Hash,
                responses: &[TraceResponseEvent],
                openings: &[TraceOpeningEvent],
                verdicts: &mut Vec<Option<&'static str>>,
            ) -> Vec<(Hash, u32)> {
                let tier = |t: u8, _at: u64| -> Option<TraceRules> {
                    (self.auditable && t == 0).then(|| TraceRules {
                        ctx: self.model.tier(),
                        prefix: self.prefix.map(|(o, _)| self.model.prefix_ctx(o)),
                        chain: None,
                        audit_all: self.audit_all,
                        attention: self.attention,
                    })
                };
                let lens = [(RH, self.shape.map_or(0, |(p, _)| p))];
                let tr = TraceInputs {
                    chain_hash: chain,
                    responses,
                    openings,
                    tier: &tier,
                    prompt_lens: &lens,
                    shape: self.shape.is_some(),
                    unique_root: self.audit_all,
                    sanction: self.sanction,
                    context: self.context,
                    draw: false,
                    window_v3: false,
                    dedup: false,
                    seed_bind: self.seed_bind,
                };
                let set = self.cohort.clone();
                let resp: Vec<([u8; 32], Option<Hash>)> = responses.iter().map(|e| (e.request_hash, Some(e.responder))).collect();
                let req: Vec<([u8; 32], u8, u32)> = match (daa, self.audit_all) {
                    (100, false) => vec![(RH, 0, self.max_tokens)],
                    (100, true) => vec![(RH, 0, self.max_tokens), (RH2, 0, 256)],
                    _ => vec![],
                };
                let outcome = self.ledger.on_chain_block_with_traces(
                    daa,
                    &req,
                    &[],
                    &[],
                    &resp,
                    &[],
                    &[],
                    &tr,
                    |_| true,
                    |_| set.iter().map(|m| (*m, *m)).collect(),
                );
                verdicts.extend(outcome.verdicts.iter().map(|v| v.cause));
                outcome.misses.iter().map(|m| (m.miner, m.consecutive_misses)).collect()
            }

            /// The traced response of this run: 3 prompt positions, 9 in total (40 and 48 under
            /// the prefix rule).
            fn event(&mut self, honest: bool) -> (keryx_verifiable_inference::testkit::Response, TraceResponseEvent) {
                let r = match (&self.prefix, self.positions) {
                    (Some((offset, carried)), _) => self.model.response_with_prefix(40, 48, honest, Some((carried, *offset))),
                    (None, Some((n_prompt, n_pos))) => self.model.response(n_prompt, n_pos, honest),
                    (None, None) => self.model.response(3, 9, honest),
                };
                let ev = TraceResponseEvent {
                    response_txid: TXID,
                    request_hash: RH,
                    responder: miner(),
                    escrow_pubkey: ESCROW_PK,
                    trace_root: r.trace_root,
                    n_prompt: r.n_prompt,
                    n_pos: r.n_pos,
                    answer_len: self.shape.map_or(0, |(_, a)| a),
                    stage: 1,
                    tokens: false,
                };
                (r, ev)
            }

            fn window_end(&self) -> u64 {
                self.ledger.pending.get(&RH).unwrap().audit.as_ref().unwrap().window_end_daa
            }

            /// Request at 100, traced response at 102, draw at 202: the response and its seed.
            fn drawn(&mut self, honest: bool, drawn: bool) -> (keryx_verifiable_inference::testkit::Response, [u8; 32]) {
                let (r, ev) = self.event(honest);
                let h0 = Hash::from_bytes([0u8; 32]);
                assert!(self.fold(100, h0, &[], &[]).is_empty());
                assert!(self.fold(102, h0, &[ev], &[]).is_empty());
                let ch = chain_hash(drawn);
                assert!(self.fold(202, ch, &[], &[]).is_empty());
                (r, trace_seed(&ch, &TXID))
            }

            fn deadline(&self) -> u64 {
                self.ledger.trace_audits.get(&TXID).map(|a| a.deadline_daa).unwrap_or(0)
            }

            /// [`Self::drawn`] then the opening of this run. Returns the opening deadline.
            fn until_draw(&mut self, honest: bool, drawn: bool) -> (u64, Vec<u8>) {
                let (r, seed) = self.drawn(honest, drawn);
                let body = match &self.prefix {
                    Some((offset, carried)) => self.model.open_with_prefix(&r, TXID, &seed, carried, *offset),
                    None => self.model.open(&r, TXID, &seed),
                }
                .encode();
                (self.deadline(), body)
            }
        }

        fn opening(pk: [u8; 32], body: Vec<u8>) -> TraceOpeningEvent {
            TraceOpeningEvent { response_txid: TXID, escrow_pubkey: pk, body }
        }

        /// Request at 100, audit armed at 101, then the window expires with nobody answering.
        fn silent_window(run: &mut Run) -> Vec<(Hash, u32)> {
            let h0 = Hash::from_bytes([0u8; 32]);
            assert!(run.fold(100, h0, &[], &[]).is_empty());
            assert!(run.fold(101, h0, &[], &[]).is_empty());
            let end = run.window_end();
            run.fold(end + 1, h0, &[], &[])
        }

        #[test]
        fn grey_prompt_silence_is_excused_when_nobody_answers() {
            let mut run = Run::context(8_000);
            assert!(silent_window(&mut run).is_empty());
            assert_eq!(run.ledger.consecutive_misses(&miner()), 0);
            assert!(run.ledger.pending.is_empty());
        }

        #[test]
        fn grey_prompt_silence_is_struck_without_the_rule() {
            let mut run = Run::shaped(8_000, 10);
            assert_eq!(silent_window(&mut run), vec![(miner(), 1)]);
        }

        #[test]
        fn safe_prompt_silence_is_struck() {
            let mut run = Run::context(64);
            assert_eq!(silent_window(&mut run), vec![(miner(), 1)]);
        }

        /// Request at 100, audit armed at 101, the traced response of `miner()` at 102, then the
        /// window expires on the silent second member `b`.
        fn answered_window(run: &mut Run, b: Hash) -> Vec<(Hash, u32)> {
            run.cohort.push(b);
            let (_, ev) = run.event(true);
            let h0 = Hash::from_bytes([0u8; 32]);
            assert!(run.fold(100, h0, &[], &[]).is_empty());
            assert!(run.fold(101, h0, &[], &[]).is_empty());
            assert!(run.fold(102, h0, &[ev], &[]).is_empty());
            let end = run.window_end();
            run.fold(end + 1, h0, &[], &[])
        }

        #[test]
        fn grey_prompt_silence_is_struck_once_a_member_proved_the_fit() {
            let b = Hash::from_bytes([2u8; 32]);
            let mut run = Run::context(8_000);
            run.positions = Some((500, 510));
            assert_eq!(answered_window(&mut run, b), vec![(b, 1)]);
            assert_eq!(run.ledger.consecutive_misses(&miner()), 0);
        }

        #[test]
        fn grey_prompt_silence_is_excused_when_the_answer_exceeded_the_floor() {
            let b = Hash::from_bytes([2u8; 32]);
            let mut run = Run::context(8_000);
            run.max_tokens = 8_000;
            run.positions = Some((500, 510));
            assert!(answered_window(&mut run, b).is_empty());
            assert_eq!(run.ledger.consecutive_misses(&b), 0);
        }

        #[test]
        fn a_misshaped_answer_proves_nothing() {
            let b = Hash::from_bytes([2u8; 32]);
            let mut run = Run::context(8_000);
            // 3 prompt positions for an 8 000-byte prompt: not an answer.
            assert!(answered_window(&mut run, b).is_empty());
        }

        #[test]
        fn an_early_member_answer_proves_the_fit() {
            let b = Hash::from_bytes([2u8; 32]);
            let mut run = Run::context(8_000);
            run.cohort.push(b);
            run.positions = Some((500, 510));
            let (_, ev) = run.event(true);
            let h0 = Hash::from_bytes([0u8; 32]);
            // the answer lands in the request's own block, before the audit arms
            assert!(run.fold(100, h0, &[ev], &[]).is_empty());
            assert!(run.fold(101, h0, &[], &[]).is_empty());
            assert!(run.ledger.early_shapes.is_empty());
            let end = run.window_end();
            assert_eq!(run.fold(end + 1, h0, &[], &[]), vec![(b, 1)]);
        }

        #[test]
        fn a_proven_fit_survives_the_snapshot() {
            let mut run = Run::context(8_000);
            run.positions = Some((500, 510));
            let (_, ev) = run.event(true);
            let h0 = Hash::from_bytes([0u8; 32]);
            assert!(run.fold(100, h0, &[], &[]).is_empty());
            assert!(run.fold(101, h0, &[], &[]).is_empty());
            assert!(run.fold(102, h0, &[ev], &[]).is_empty());
            assert!(run.ledger.pending.get(&RH).unwrap().fit_proven);
            let bytes = run.ledger.snapshot().to_bytes();
            assert_eq!(bytes[0], super::super::SNAPSHOT_ENCODING_VERSION_CONTEXT);
            let back = ServiceLedgerSnapshot::from_bytes(&bytes).unwrap();
            assert!(back.pending.get(&RH).unwrap().fit_proven);
            assert_eq!(back.to_bytes(), bytes);
        }

        #[test]
        fn valid_opening_closes_the_audit() {
            let mut run = Run::new(true);
            let (deadline, body) = run.until_draw(true, true);
            assert!(deadline > 203);
            let h = Hash::from_bytes([3u8; 32]);
            assert!(run.fold(203, h, &[], &[opening(ESCROW_PK, body)]).is_empty());
            assert!(run.ledger.trace_audits.is_empty());
            assert!(run.fold(deadline + 1, h, &[], &[]).is_empty());
            assert_eq!(run.ledger.consecutive_misses(&miner()), 0);
        }

        fn attention_body() -> Vec<u8> {
            use keryx_verifiable_inference::opening::AttnOpening;
            AttnOpening { response_txid: TXID, ..Default::default() }
            .encode()
        }

        #[test]
        fn a_valid_trace_opening_waits_for_the_attention_opening() {
            let mut run = Run { attention: true, ..Run::new(true) };
            let (deadline, body) = run.until_draw(true, true);
            let h = Hash::from_bytes([3u8; 32]);
            assert!(run.fold(203, h, &[], &[opening(ESCROW_PK, body)]).is_empty());
            assert_eq!(run.ledger.trace_audits.get(&TXID).map(|a| a.pieces), Some(crate::collateral::PIECE_TRACE));
            let bytes = run.ledger.snapshot().to_bytes();
            assert_eq!(bytes[0], crate::collateral::SNAPSHOT_ENCODING_VERSION_ATTN);
            assert_eq!(ServiceLedgerSnapshot::from_bytes(&bytes).unwrap().to_bytes(), bytes);
            assert!(run.fold(deadline, h, &[], &[]).is_empty());
            assert_eq!(run.fold(deadline + 1, h, &[], &[]), vec![(miner(), 1)]);
            assert!(run.ledger.trace_audits.is_empty());
        }

        #[test]
        fn a_bad_attention_opening_strikes_immediately() {
            let mut run = Run { attention: true, ..Run::new(true) };
            run.until_draw(true, true);
            let misses = run.fold(203, Hash::from_bytes([3u8; 32]), &[], &[opening(ESCROW_PK, attention_body())]);
            assert_eq!(misses, vec![(miner(), 2)]);
            assert!(run.ledger.trace_audits.is_empty());
        }

        fn phase_body(path: &[u64]) -> Vec<u8> {
            keryx_verifiable_inference::opening::AttnOpening { response_txid: TXID, path: path.to_vec(), partials: vec![path.len() as u8; 4], ..Default::default() }
                .encode()
        }

        type Seen = std::rc::Rc<std::cell::RefCell<Vec<(Vec<u64>, Option<[u8; 32]>, u8)>>>;

        /// Attention phases judged by `step` (by phase) instead of the verifier; returns what each
        /// call saw.
        fn oracle(step: impl Fn(u8) -> Result<keryx_verifiable_inference::chain::AttnStep, &'static str> + 'static) -> Seen {
            let seen: Seen = Default::default();
            let log = seen.clone();
            crate::collateral::ATTN_ORACLE.with(|o| {
                *o.borrow_mut() = Some(Box::new(move |path: &[u64], root: Option<&[u8; 32]>, op: &keryx_verifiable_inference::opening::AttnOpening| {
                    log.borrow_mut().push((path.to_vec(), root.copied(), op.path.len() as u8));
                    step(op.path.len() as u8)
                }))
            });
            seen
        }

        const R0: [u8; 32] = [0x51; 32];
        const R1: [u8; 32] = [0x52; 32];

        const W0: [u16; 4] = [1, 1, 60_000, 1];
        const W1: [u16; 3] = [5, 5, 5];

        fn three_phases(phase: u8) -> Result<keryx_verifiable_inference::chain::AttnStep, &'static str> {
            use keryx_verifiable_inference::chain::AttnStep;
            Ok(match phase {
                0 => AttnStep::Next { root: R0, n: 4, weights: W0.to_vec() },
                1 => AttnStep::Next { root: R1, n: 3, weights: W1.to_vec() },
                _ => AttnStep::Done,
            })
        }

        /// Trace opening and attention phase 0 accepted at 203.
        fn after_phase_0(step: impl Fn(u8) -> Result<keryx_verifiable_inference::chain::AttnStep, &'static str> + 'static) -> (Run, Seen) {
            let seen = oracle(step);
            let mut run = Run { attention: true, ..Run::new(true) };
            let (_, body) = run.until_draw(true, true);
            let h = Hash::from_bytes([3u8; 32]);
            assert!(run.fold(203, h, &[], &[opening(ESCROW_PK, body), opening(ESCROW_PK, phase_body(&[]))]).is_empty());
            (run, seen)
        }

        #[test]
        fn the_child_seed_matches_its_known_answer() {
            let seed = crate::collateral::attn_child_seed(&Hash::from_bytes([1u8; 32]), &[2u8; 32], 1);
            let hex: String = seed.iter().map(|b| format!("{b:02x}")).collect();
            assert_eq!(hex, "347b8cd184f5fdf5d3ef1828b99bb28d859b019df6b09ece1ec8167ea62e534b");
        }

        #[test]
        fn an_attention_opening_runs_its_phases_on_drawn_children() {
            use crate::collateral::{AttnProgress, attn_child_seed};
            use keryx_verifiable_inference::attn_blocks::draw_weighted_child;
            let (mut run, seen) = after_phase_0(three_phases);
            let h = Hash::from_bytes([3u8; 32]);
            let progress = |run: &Run| run.ledger.trace_audits.get(&TXID).and_then(|a| a.attn.clone());
            assert_eq!(progress(&run), Some(AttnProgress { path: vec![], root: R0, n: 4, accepted_daa: 203, drawn: false, weights: W0.to_vec() }));
            let (h2, h3) = (Hash::from_bytes([0x61u8; 32]), Hash::from_bytes([0x62u8; 32]));
            let c0 = draw_weighted_child(&attn_child_seed(&h2, &TXID, 0), &W0);
            assert_eq!(run.ledger.awaited_attention_path(&TXID), None, "no child drawn yet");
            assert!(run.fold(250, h, &[], &[opening(ESCROW_PK, phase_body(&[c0]))]).is_empty());
            assert_eq!(seen.borrow().len(), 1, "a phase before its child is drawn is passed over");
            assert!(run.fold(303, h2, &[], &[]).is_empty());
            assert_eq!(run.ledger.awaited_attention_path(&TXID), Some(vec![c0]));
            assert_eq!(progress(&run).map(|p| (p.path, p.drawn)), Some((vec![c0], true)));
            let other = (c0 + 1) % 4;
            let wrong = [phase_body(&[c0, 0]), phase_body(&[]), phase_body(&[other])];
            assert!(run.fold(304, h, &[], &wrong.map(|b| opening(ESCROW_PK, b))).is_empty());
            assert_eq!(seen.borrow().len(), 1, "phases built for another path are passed over");
            assert!(run.fold(305, h, &[], &[opening(ESCROW_PK, phase_body(&[c0]))]).is_empty());
            assert_eq!(progress(&run), Some(AttnProgress { path: vec![c0], root: R1, n: 3, accepted_daa: 305, drawn: false, weights: W1.to_vec() }));
            assert!(run.fold(405, h3, &[], &[]).is_empty());
            let c1 = draw_weighted_child(&attn_child_seed(&h3, &TXID, 1), &W1);
            assert!(run.fold(406, h, &[], &[opening(ESCROW_PK, phase_body(&[c0, c1]))]).is_empty());
            assert!(run.ledger.trace_audits.is_empty(), "the last phase closes the audit");
            assert_eq!(run.ledger.awaited_attention_path(&TXID), None);
            assert_eq!(*seen.borrow(), vec![(vec![], None, 0), (vec![c0], Some(R0), 1), (vec![c0, c1], Some(R1), 2)]);
            assert_eq!(run.ledger.consecutive_misses(&miner()), 0);
        }

        #[test]
        fn a_trace_opening_after_phase_0_keeps_the_attention_progress() {
            use crate::collateral::attn_child_seed;
            use keryx_verifiable_inference::attn_blocks::draw_weighted_child;
            for same_block in [true, false] {
                let seen = oracle(three_phases);
                let mut run = Run { attention: true, ..Run::new(true) };
                let (_, trace) = run.until_draw(true, true);
                let h = Hash::from_bytes([3u8; 32]);
                if same_block {
                    assert!(run.fold(203, h, &[], &[opening(ESCROW_PK, phase_body(&[])), opening(ESCROW_PK, trace)]).is_empty());
                } else {
                    assert!(run.fold(203, h, &[], &[opening(ESCROW_PK, phase_body(&[]))]).is_empty());
                    assert!(run.fold(250, h, &[], &[opening(ESCROW_PK, trace)]).is_empty());
                }
                assert_eq!(run.ledger.trace_audits.get(&TXID).map(|a| a.pieces), Some(crate::collateral::PIECE_TRACE));
                assert_eq!(run.ledger.trace_audits.get(&TXID).and_then(|a| a.attn.as_ref().map(|p| (p.root, p.n))), Some((R0, 4)));
                assert_eq!(run.ledger.awaited_attention_path(&TXID), None, "phase 0 is not awaited again");
                assert!(run.fold(260, h, &[], &[opening(ESCROW_PK, phase_body(&[]))]).is_empty());
                assert_eq!(seen.borrow().len(), 1, "a replayed phase 0 is passed over");
                let (h2, h3) = (Hash::from_bytes([0x61u8; 32]), Hash::from_bytes([0x62u8; 32]));
                assert!(run.fold(303, h2, &[], &[]).is_empty());
                let c0 = draw_weighted_child(&attn_child_seed(&h2, &TXID, 0), &W0);
                assert_eq!(run.ledger.awaited_attention_path(&TXID), Some(vec![c0]));
                assert!(run.fold(305, h, &[], &[opening(ESCROW_PK, phase_body(&[c0]))]).is_empty());
                assert!(run.fold(405, h3, &[], &[]).is_empty());
                let c1 = draw_weighted_child(&attn_child_seed(&h3, &TXID, 1), &W1);
                assert!(run.fold(406, h, &[], &[opening(ESCROW_PK, phase_body(&[c0, c1]))]).is_empty());
                assert!(run.ledger.trace_audits.is_empty(), "same block {same_block}");
                assert_eq!(run.ledger.consecutive_misses(&miner()), 0);
            }
        }

        #[test]
        fn phase_0_is_awaited_once_the_response_is_drawn() {
            let mut run = Run { attention: true, ..Run::new(true) };
            assert_eq!(run.ledger.awaited_attention_path(&TXID), None);
            run.until_draw(true, true);
            assert_eq!(run.ledger.awaited_attention_path(&TXID), Some(vec![]));
        }

        #[test]
        fn a_failed_attention_phase_strikes_immediately() {
            let (mut run, _) = after_phase_0(|phase| if phase == 0 { three_phases(0) } else { Err("attention point") });
            let h = Hash::from_bytes([3u8; 32]);
            assert!(run.fold(303, h, &[], &[]).is_empty());
            let path = run.ledger.trace_audits.get(&TXID).and_then(|a| a.attn.clone()).unwrap().path;
            assert_eq!(run.fold(304, h, &[], &[opening(ESCROW_PK, phase_body(&path))]), vec![(miner(), 2)]);
            assert!(run.ledger.trace_audits.is_empty());
        }

        #[test]
        fn a_missing_attention_phase_strikes_after_its_window() {
            use crate::collateral::{ATTN_PHASE_WINDOW_DAA, TRACE_SEED_DELAY_DAA};
            let (mut run, _) = after_phase_0(three_phases);
            let deadline = run.deadline();
            assert!(deadline >= 203 + TRACE_SEED_DELAY_DAA + ATTN_PHASE_WINDOW_DAA);
            let h = Hash::from_bytes([3u8; 32]);
            assert!(run.fold(deadline, h, &[], &[]).is_empty());
            assert_eq!(run.fold(deadline + 1, h, &[], &[]), vec![(miner(), 1)]);
            assert!(run.ledger.trace_audits.is_empty());
        }

        #[test]
        fn attention_progress_survives_the_snapshot_and_malformed_sections_are_refused() {
            let (mut run, _) = after_phase_0(three_phases);
            let check = |run: &Run| {
                let bytes = run.ledger.snapshot().to_bytes();
                assert_eq!(bytes[0], crate::collateral::SNAPSHOT_ENCODING_VERSION_PHASES);
                let back = ServiceLedgerSnapshot::from_bytes(&bytes).unwrap();
                assert_eq!(back.trace_audits.get(&TXID).and_then(|a| a.attn.clone()), run.ledger.trace_audits.get(&TXID).and_then(|a| a.attn.clone()));
                assert_eq!(back.to_bytes(), bytes);
                bytes
            };
            let bytes = check(&run);
            let n = bytes.len();
            let mut undrawn_with_flag = bytes.clone();
            undrawn_with_flag[n - 9] = 1;
            assert!(ServiceLedgerSnapshot::from_bytes(&undrawn_with_flag).is_err());
            let mut bad_flag = bytes.clone();
            bad_flag[n - 9] = 2;
            assert!(ServiceLedgerSnapshot::from_bytes(&bad_flag).is_err());
            let mut childless = bytes.clone();
            childless[n - 25..n - 17].copy_from_slice(&0u64.to_le_bytes());
            assert!(ServiceLedgerSnapshot::from_bytes(&childless).is_err());
            let mut weightless = bytes.clone();
            weightless[n - 2..].copy_from_slice(&0u16.to_le_bytes());
            assert!(ServiceLedgerSnapshot::from_bytes(&weightless).is_err());
            let mut huge = bytes.clone();
            huge[n - 25..n - 17].copy_from_slice(&u64::MAX.to_le_bytes());
            assert!(ServiceLedgerSnapshot::from_bytes(&huge).is_err());
            assert!(ServiceLedgerSnapshot::from_bytes(&childless).is_err());
            assert!(run.fold(303, Hash::from_bytes([0x61u8; 32]), &[], &[]).is_empty());
            let drawn = check(&run);
            let n = drawn.len();
            let mut out_of_range = drawn.clone();
            out_of_range[n - 65..n - 57].copy_from_slice(&4u64.to_le_bytes());
            assert!(ServiceLedgerSnapshot::from_bytes(&out_of_range).is_err());
        }

        #[test]
        fn a_recurrent_opening_is_passed_over_without_its_rule() {
            use keryx_verifiable_inference::opening::RecurOpening;
            let mut run = Run { attention: true, ..Run::new(true) };
            run.until_draw(true, true);
            let body = RecurOpening { response_txid: TXID, ..Default::default() }.encode();
            assert!(run.fold(203, Hash::from_bytes([3u8; 32]), &[], &[opening(ESCROW_PK, body)]).is_empty());
            assert_eq!(run.ledger.trace_audits.get(&TXID).map(|a| a.pieces), Some(0));
        }

        #[test]
        fn pieces_follow_the_rules() {
            use crate::collateral::{PIECE_ATTN, PIECE_RECUR, PIECE_TRACE, required_pieces};
            use crate::config::params::trace_rules;
            assert_eq!(required_pieces(None), PIECE_TRACE);
            assert_eq!(required_pieces(trace_rules(1, true, true, true, false).as_ref()), PIECE_TRACE);
            assert_eq!(required_pieces(trace_rules(1, true, true, true, true).as_ref()), PIECE_TRACE | PIECE_ATTN);
            assert_eq!(required_pieces(trace_rules(0, true, true, true, true).as_ref()), PIECE_TRACE | PIECE_ATTN | PIECE_RECUR);
            assert_eq!(required_pieces(trace_rules(4, true, true, true, true).as_ref()), PIECE_TRACE | PIECE_ATTN | PIECE_RECUR);
        }

        #[test]
        fn a_trace_opening_carries_the_root_halves_under_the_attention_rules() {
            use crate::collateral::{TraceAudit, trace_opening_valid};
            use crate::config::params::trace_rules;
            use keryx_verifiable_inference::draw::POINTS;
            use keryx_verifiable_inference::opening::{EntryOpening, LeafChunk, LinkOpening, Opening, Point, PrefixPoint, ResidAdd, ResidPoint, ResidSrc};
            let (pos_root, seq_root) = ([0x11u8; 32], [0x22u8; 32]);
            let audit = TraceAudit {
                request_hash: RH,
                responder: miner(),
                escrow_pubkey: ESCROW_PK,
                trace_root: keryx_verifiable_inference::seq::trace_root(&pos_root, &seq_root),
                n_prompt: 4,
                n_pos: 9,
                accepted_daa: 0,
                tier: 1,
                max_tokens: 16,
                identities: vec![miner()],
                seed: Some([9; 32]),
                deadline_daa: 0,
                pieces: 0,
                attn: None,
            };
            let body = |roots: Option<([u8; 32], [u8; 32])>| {
                Opening {
                    response_txid: TXID,
                    points: vec![Point::default(); POINTS as usize],
                    prefix: Some(PrefixPoint::default()),
                    links: vec![(0, LinkOpening::None); POINTS as usize],
                    experts: vec![None; POINTS as usize],
                    resid: Some(ResidPoint {
                        n_ubatch: 1,
                        tok: 0,
                        pos_path: vec![],
                        meta_path: vec![],
                        layer: EntryOpening::default(),
                        product: EntryOpening::default(),
                        dst: LeafChunk::default(),
                        src: ResidSrc::Chunk(LeafChunk::default()),
                        add: ResidAdd::Chunk(LeafChunk::default()),
                    }),
                    roots,
                }
                .encode()
            };
            let (with, without) = (trace_rules(1, true, true, true, true).unwrap(), trace_rules(1, true, true, true, false).unwrap());
            let check = |r, roots| trace_opening_valid(r, &audit, &[9; 32], &TXID, &body(roots));
            assert_eq!(check(&with, None), Err("sequence root missing"));
            assert_eq!(check(&with, Some(([0x12; 32], seq_root))), Err("trace root split"));
            assert_eq!(check(&with, Some((pos_root, seq_root))), Err("point 0"));
            assert_eq!(check(&without, Some((pos_root, seq_root))), Err("sequence root before its rule"));
        }

        #[test]
        fn an_attention_opening_is_passed_over_without_its_rule() {
            let mut run = Run::new(true);
            let (deadline, body) = run.until_draw(true, true);
            let h = Hash::from_bytes([3u8; 32]);
            assert!(run.fold(203, h, &[], &[opening(ESCROW_PK, attention_body())]).is_empty());
            assert_eq!(run.ledger.trace_audits.get(&TXID).map(|a| a.pieces), Some(0));
            assert!(run.fold(204, h, &[], &[opening(ESCROW_PK, body)]).is_empty());
            assert!(run.ledger.trace_audits.is_empty());
            assert!(run.fold(deadline + 1, h, &[], &[]).is_empty());
            assert_eq!(run.ledger.consecutive_misses(&miner()), 0);
        }

        #[test]
        fn fraudulent_opening_slashes_immediately() {
            let mut run = Run::new(true);
            let (_, body) = run.until_draw(false, true);
            let misses = run.fold(203, Hash::from_bytes([3u8; 32]), &[], &[opening(ESCROW_PK, body)]);
            assert_eq!(misses, vec![(miner(), 2)]);
            assert!(run.ledger.trace_audits.is_empty());
        }

        /// Two chain blocks that both draw `TXID`, an honest response, and the ledger before the
        /// draw (accepted at 102, drawn at 202).
        fn reorged_draw(run: &mut Run) -> (Response, Hash, Hash, ServiceLedger) {
            let (r, ev) = run.event(true);
            let h0 = Hash::from_bytes([0u8; 32]);
            assert!(run.fold(100, h0, &[], &[]).is_empty());
            assert!(run.fold(102, h0, &[ev], &[]).is_empty());
            let mut drawn = (0u64..)
                .map(|i| {
                    let mut b = [0u8; 32];
                    b[..8].copy_from_slice(&i.to_le_bytes());
                    Hash::from_bytes(b)
                })
                .filter(|h| draw::audited(&trace_seed(h, &TXID)));
            let (block_a, block_b) = (drawn.next().unwrap(), drawn.next().unwrap());
            (r, block_a, block_b, run.ledger.clone())
        }

        fn seeded_body(run: &Run, r: &Response, built_for: &Hash, named: &Hash) -> Vec<u8> {
            keryx_verifiable_inference::opening::seeded(&run.model.open(r, TXID, &trace_seed(built_for, &TXID)).encode(), &named.as_bytes())
                .unwrap()
        }

        #[test]
        fn unseeded_opening_judged_after_its_seed_block_left_the_chain_strikes() {
            let mut run = Run::new(true);
            let (r, block_a, block_b, fork) = reorged_draw(&mut run);
            assert!(run.fold(202, block_a, &[], &[]).is_empty());
            let body = run.model.open(&r, TXID, &trace_seed(&block_a, &TXID)).encode();

            run.ledger = fork;
            assert!(run.fold(202, block_b, &[], &[]).is_empty());
            assert_eq!(run.ledger.awaited_opening_key(&TXID), Some(ESCROW_PK));
            let mut verdicts = Vec::new();
            let misses = run.fold_with_verdicts(203, Hash::from_bytes([3u8; 32]), &[], &[opening(ESCROW_PK, body)], &mut verdicts);
            assert!(matches!(verdicts[..], [Some(_)]), "{verdicts:?}");
            assert_eq!(misses, vec![(miner(), 2)]);
            assert_eq!(run.ledger.awaited_opening_key(&TXID), None);
        }

        #[test]
        fn seeded_opening_of_a_seed_block_that_left_the_chain_waits_for_the_one_of_the_chain() {
            let mut run = Run { seed_bind: true, ..Run::new(true) };
            let (r, block_a, block_b, fork) = reorged_draw(&mut run);
            assert!(run.fold(202, block_a, &[], &[]).is_empty());
            let stale = seeded_body(&run, &r, &block_a, &block_a);

            run.ledger = fork;
            assert!(run.fold(202, block_b, &[], &[]).is_empty());
            let mut verdicts = Vec::new();
            let misses = run.fold_with_verdicts(203, Hash::from_bytes([3u8; 32]), &[], &[opening(ESCROW_PK, stale)], &mut verdicts);
            assert!(verdicts.is_empty(), "{verdicts:?}");
            assert!(misses.is_empty());
            assert_eq!(run.ledger.awaited_opening_key(&TXID), Some(ESCROW_PK));

            let fresh = seeded_body(&run, &r, &block_b, &block_b);
            let misses = run.fold_with_verdicts(204, Hash::from_bytes([4u8; 32]), &[], &[opening(ESCROW_PK, fresh)], &mut verdicts);
            assert_eq!(verdicts, vec![None]);
            assert!(misses.is_empty());
            assert_eq!(run.ledger.awaited_opening_key(&TXID), None);
        }

        #[test]
        fn seeded_opening_before_its_rule_is_passed_over() {
            let mut run = Run::new(true);
            let (r, block_a, _, _) = reorged_draw(&mut run);
            assert!(run.fold(202, block_a, &[], &[]).is_empty());
            let body = seeded_body(&run, &r, &block_a, &block_a);
            let mut verdicts = Vec::new();
            let misses = run.fold_with_verdicts(203, Hash::from_bytes([3u8; 32]), &[], &[opening(ESCROW_PK, body)], &mut verdicts);
            assert!(verdicts.is_empty(), "{verdicts:?}");
            assert!(misses.is_empty());
            assert_eq!(run.ledger.awaited_opening_key(&TXID), Some(ESCROW_PK));
        }

        #[test]
        fn seeded_opening_naming_the_seed_block_is_judged_on_its_points() {
            let mut run = Run { seed_bind: true, ..Run::new(true) };
            let (r, block_a, block_b, _) = reorged_draw(&mut run);
            assert!(run.fold(202, block_b, &[], &[]).is_empty());
            let forged = seeded_body(&run, &r, &block_a, &block_b);
            let mut verdicts = Vec::new();
            let misses = run.fold_with_verdicts(203, Hash::from_bytes([3u8; 32]), &[], &[opening(ESCROW_PK, forged)], &mut verdicts);
            assert!(matches!(verdicts[..], [Some(_)]), "{verdicts:?}");
            assert_eq!(misses, vec![(miner(), 2)]);
            assert_eq!(run.ledger.awaited_opening_key(&TXID), None);
        }

        #[test]
        fn malformed_opening_is_fraud() {
            let mut run = Run::new(true);
            let (_, mut body) = run.until_draw(true, true);
            body.pop();
            let misses = run.fold(203, Hash::from_bytes([3u8; 32]), &[], &[opening(ESCROW_PK, body)]);
            assert_eq!(misses, vec![(miner(), 2)]);
        }

        #[test]
        fn missing_opening_strikes_at_the_deadline() {
            let mut run = Run::new(true);
            let (deadline, _) = run.until_draw(true, true);
            let h = Hash::from_bytes([3u8; 32]);
            assert!(run.fold(deadline, h, &[], &[]).is_empty());
            assert_eq!(run.fold(deadline + 1, h, &[], &[]), vec![(miner(), 1)]);
            assert!(run.ledger.trace_audits.is_empty());
        }

        #[test]
        fn missing_opening_is_rate_limited_under_the_audit_rule_alone() {
            let mut run = Run::audited_all();
            run.ledger.strikes.insert(miner(), crate::collateral::StrikeEntry { count: 1, last_daa: 99 });
            let (deadline, _) = run.until_draw(true, false);
            // The served response resets the streak at once.
            assert_eq!(run.ledger.consecutive_misses(&miner()), 0);
            let h = Hash::from_bytes([3u8; 32]);
            // The unanswered second request strikes at its window end.
            assert_eq!(run.fold(run.window_end() + 1, h, &[], &[]), vec![(miner(), 1)]);
            // The unopened audit falls inside the interval guard of that strike.
            assert!(run.fold(deadline + 1, h, &[], &[]).is_empty());
            assert!(run.ledger.trace_audits.is_empty());
        }

        #[test]
        fn missing_opening_ignores_the_interval_guard_under_the_sanction_rule() {
            let mut run = Run::sanctioned();
            run.ledger.strikes.insert(miner(), crate::collateral::StrikeEntry { count: 1, last_daa: 99 });
            let (deadline, _) = run.until_draw(true, false);
            // Served, credited, but the streak waits for the opening.
            assert_eq!(run.ledger.pending.get(&RH).unwrap().audit.as_ref().unwrap().responded, vec![miner()]);
            assert_eq!(run.ledger.consecutive_misses(&miner()), 1);
            let h = Hash::from_bytes([3u8; 32]);
            // Cohort silence on the second request keeps the interval guard.
            assert!(run.fold(run.window_end() + 1, h, &[], &[]).is_empty());
            // The unopened audit strikes anyway and escalates.
            assert_eq!(run.fold(deadline + 1, h, &[], &[]), vec![(miner(), 2)]);
            assert!(run.ledger.trace_audits.is_empty());
        }

        #[test]
        fn accepted_opening_resets_the_streak_under_the_sanction_rule() {
            let mut run = Run::sanctioned();
            run.ledger.strikes.insert(miner(), crate::collateral::StrikeEntry { count: 1, last_daa: 99 });
            let (deadline, body) = run.until_draw(true, false);
            assert_eq!(run.ledger.consecutive_misses(&miner()), 1);
            let h = Hash::from_bytes([3u8; 32]);
            assert!(run.fold(203, h, &[], &[opening(ESCROW_PK, body)]).is_empty());
            assert_eq!(run.ledger.consecutive_misses(&miner()), 0);
            assert!(run.ledger.trace_audits.is_empty());
            // The unanswered second request strikes at its window end; the closed audit does not.
            assert_eq!(run.fold(run.window_end() + 1, h, &[], &[]), vec![(miner(), 1)]);
            assert!(run.fold(deadline + 1, h, &[], &[]).is_empty());
        }

        #[test]
        fn two_missing_openings_escalate_in_one_fold_under_the_sanction_rule() {
            for sanction in [false, true] {
                let mut run = Run::audited_all();
                run.sanction = sanction;
                let (_, ev) = run.event(true);
                let r2 = run.model.response(3, 10, true);
                let ev2 = TraceResponseEvent {
                    response_txid: TXID2,
                    request_hash: RH2,
                    trace_root: r2.trace_root,
                    n_prompt: r2.n_prompt,
                    n_pos: r2.n_pos,
                    ..ev.clone()
                };
                let h0 = Hash::from_bytes([0u8; 32]);
                assert!(run.fold(100, h0, &[], &[]).is_empty());
                assert!(run.fold(102, h0, &[ev, ev2], &[]).is_empty());
                assert!(run.fold(202, chain_hash(false), &[], &[]).is_empty());
                assert_eq!(run.ledger.trace_audits.len(), 2);
                let deadline = run.deadline();
                let misses = run.fold(deadline + 1, Hash::from_bytes([3u8; 32]), &[], &[]);
                if sanction {
                    assert_eq!(misses, vec![(miner(), 1), (miner(), 2)]);
                } else {
                    assert_eq!(misses, vec![(miner(), 1)]);
                }
                assert!(run.ledger.trace_audits.is_empty());
            }
        }

        #[test]
        fn opening_under_another_key_is_ignored() {
            let mut run = Run::new(true);
            let (deadline, body) = run.until_draw(false, true);
            let h = Hash::from_bytes([3u8; 32]);
            assert!(run.fold(203, h, &[], &[opening([0x11u8; 32], body)]).is_empty());
            assert_eq!(run.ledger.trace_audits.len(), 1);
            assert_eq!(run.fold(deadline + 1, h, &[], &[]), vec![(miner(), 1)]);
        }

        #[test]
        fn undrawn_response_is_dropped() {
            let mut run = Run::new(true);
            let (deadline, body) = run.until_draw(false, false);
            assert_eq!(deadline, 0);
            assert!(run.ledger.trace_audits.is_empty());
            assert!(run.fold(203, Hash::from_bytes([3u8; 32]), &[], &[opening(ESCROW_PK, body)]).is_empty());
        }

        #[test]
        fn every_trace_is_audited_under_the_audit_rule() {
            let mut run = Run::audited_all();
            let (deadline, body) = run.until_draw(true, false);
            assert!(deadline > 203);
            assert_eq!(run.ledger.trace_audits.len(), 1);
            let h = Hash::from_bytes([3u8; 32]);
            let mut verdicts = Vec::new();
            assert!(run.fold_with_verdicts(203, h, &[], &[opening(ESCROW_PK, body)], &mut verdicts).is_empty());
            assert_eq!(verdicts, vec![None]);
            assert!(run.ledger.trace_audits.is_empty());
            // The second request of the run was never answered.
            assert_eq!(run.fold(run.window_end() + 1, h, &[], &[]), vec![(miner(), 1)]);
        }

        #[test]
        fn fraud_verdict_names_the_failing_check() {
            let mut run = Run::prefixed(8, RH);
            let (r, seed) = run.drawn(true, true);
            let body = run.model.open(&r, TXID, &seed).encode();
            let mut verdicts = Vec::new();
            let misses = run.fold_with_verdicts(203, Hash::from_bytes([3u8; 32]), &[], &[opening(ESCROW_PK, body)], &mut verdicts);
            assert_eq!(misses, vec![(miner(), 2)]);
            assert_eq!(verdicts, vec![Some("prefix point missing")]);
        }

        #[test]
        fn reused_trace_root_is_not_an_answer_under_the_root_rule() {
            let mut run = Run::audited_all();
            let (_, ev) = run.event(true);
            let ev2 = TraceResponseEvent { response_txid: TXID2, request_hash: RH2, ..ev.clone() };
            let h0 = Hash::from_bytes([0u8; 32]);
            assert!(run.fold(100, h0, &[], &[]).is_empty());
            assert!(run.fold(102, h0, &[ev, ev2], &[]).is_empty());
            assert_eq!(run.ledger.trace_audits.len(), 1);
            assert!(run.ledger.trace_audits.contains_key(&TXID));
            assert_eq!(run.ledger.pending.get(&RH).unwrap().audit.as_ref().unwrap().responded, vec![miner()]);
            assert!(run.ledger.pending.get(&RH2).unwrap().audit.as_ref().unwrap().responded.is_empty());
            assert_eq!(run.ledger.trace_roots.len(), 1);

            let snap = run.ledger.snapshot();
            let bytes = snap.to_bytes();
            assert_eq!(bytes[0], 5);
            let parsed = ServiceLedgerSnapshot::from_bytes(&bytes).unwrap();
            assert_eq!(parsed, snap);
            assert!(ServiceLedgerSnapshot::from_bytes(&bytes[..bytes.len() - 1]).is_err());
            let mut restored = Run::audited_all();
            restored.ledger.restore_snapshot(&parsed);
            assert_eq!(restored.ledger.trace_roots.len(), 1);
            // The unanswered second request strikes at its deadline; the served one does not.
            let end = restored.window_end();
            assert_eq!(restored.fold(end + 1, chain_hash(false), &[], &[]), vec![(miner(), 1)]);
        }

        #[test]
        fn root_memory_expires() {
            let mut run = Run::audited_all();
            let (_, ev) = run.event(true);
            let h0 = Hash::from_bytes([0u8; 32]);
            assert!(run.fold(100, h0, &[], &[]).is_empty());
            assert!(run.fold(102, h0, &[ev], &[]).is_empty());
            assert_eq!(run.ledger.trace_roots.len(), 1);
            let later = 102 + super::super::SERVICE_AUDITED_MEMORY_DAA;
            run.fold(later - 1, h0, &[], &[]);
            assert_eq!(run.ledger.trace_roots.len(), 1);
            run.fold(later, h0, &[], &[]);
            assert!(run.ledger.trace_roots.is_empty());
        }

        #[test]
        fn tier_without_manifest_is_not_audited() {
            let mut run = Run::new(false);
            let (deadline, _) = run.until_draw(false, true);
            assert_eq!(deadline, 0);
            assert!(run.ledger.trace_audits.is_empty());
        }

        #[test]
        fn prefix_opening_closes_the_audit() {
            let mut run = Run::prefixed(8, RH);
            let (deadline, body) = run.until_draw(true, true);
            assert!(deadline > 203);
            let h = Hash::from_bytes([3u8; 32]);
            assert!(run.fold(203, h, &[], &[opening(ESCROW_PK, body)]).is_empty());
            assert!(run.ledger.trace_audits.is_empty());
            assert!(run.fold(deadline + 1, h, &[], &[]).is_empty());
            assert_eq!(run.ledger.consecutive_misses(&miner()), 0);
        }

        #[test]
        fn opening_without_the_prefix_point_is_fraud_under_the_prefix_rule() {
            let mut run = Run::prefixed(8, RH);
            let (r, seed) = run.drawn(true, true);
            let body = run.model.open(&r, TXID, &seed).encode();
            let misses = run.fold(203, Hash::from_bytes([3u8; 32]), &[], &[opening(ESCROW_PK, body)]);
            assert_eq!(misses, vec![(miner(), 2)]);
            assert!(run.ledger.trace_audits.is_empty());
        }

        #[test]
        fn trace_of_another_request_is_fraud_under_the_prefix_rule() {
            let mut run = Run::prefixed(8, [0x99u8; 32]);
            let (_, body) = run.until_draw(true, true);
            let misses = run.fold(203, Hash::from_bytes([3u8; 32]), &[], &[opening(ESCROW_PK, body)]);
            assert_eq!(misses, vec![(miner(), 2)]);
            assert!(run.ledger.trace_audits.is_empty());
        }

        #[test]
        fn prompt_too_short_for_the_prefix_is_misshaped() {
            // 40 prompt positions hold a prefix at offset 8, not at offset 9.
            let mut run = Run::prefixed(9, RH);
            let (_, ev) = run.event(true);
            let h0 = Hash::from_bytes([0u8; 32]);
            assert!(run.fold(100, h0, &[], &[]).is_empty());
            assert!(run.fold(102, h0, &[ev], &[]).is_empty());
            assert!(run.ledger.trace_audits.is_empty());
            let end = run.window_end();
            assert_eq!(run.fold(end + 1, chain_hash(false), &[], &[]), vec![(miner(), 1)]);
        }

        #[test]
        fn misshaped_trace_is_neither_credited_nor_audited() {
            // 64 prompt bytes need 4 prompt positions; the trace commits 3.
            let mut run = Run::shaped(64, 8);
            let (_, ev) = run.event(true);
            let h0 = Hash::from_bytes([0u8; 32]);
            assert!(run.fold(100, h0, &[], &[]).is_empty());
            assert!(run.fold(102, h0, &[ev], &[]).is_empty());
            assert!(run.ledger.trace_audits.is_empty());
            let end = run.window_end();
            assert_eq!(run.fold(end + 1, chain_hash(false), &[], &[]), vec![(miner(), 1)]);
        }

        #[test]
        fn well_shaped_trace_is_credited_and_audited() {
            // 48 prompt bytes need 3 prompt positions, 60 answer bytes need 6 generated ones.
            let mut run = Run::shaped(48, 60);
            let (_, ev) = run.event(true);
            let h0 = Hash::from_bytes([0u8; 32]);
            assert!(run.fold(100, h0, &[], &[]).is_empty());
            assert!(run.fold(102, h0, &[ev], &[]).is_empty());
            assert_eq!(run.ledger.trace_audits.len(), 1);
            let end = run.window_end();
            assert!(run.fold(end + 1, chain_hash(false), &[], &[]).is_empty());
            assert_eq!(run.ledger.consecutive_misses(&miner()), 0);
        }

        #[test]
        fn answer_over_the_token_budget_is_misshaped() {
            // max_tokens 256 allows 8 192 answer bytes.
            let mut run = Run::shaped(0, 8_193);
            let (_, ev) = run.event(true);
            let h0 = Hash::from_bytes([0u8; 32]);
            assert!(run.fold(100, h0, &[], &[]).is_empty());
            assert!(run.fold(102, h0, &[ev], &[]).is_empty());
            assert!(run.ledger.trace_audits.is_empty());
            let end = run.window_end();
            assert_eq!(run.fold(end + 1, chain_hash(false), &[], &[]), vec![(miner(), 1)]);
        }

        #[test]
        fn early_misshaped_trace_is_dropped_when_its_request_arrives() {
            let mut run = Run::shaped(64, 8);
            let (_, ev) = run.event(true);
            let h0 = Hash::from_bytes([0u8; 32]);
            assert!(run.fold(98, h0, &[ev], &[]).is_empty());
            assert_eq!(run.ledger.early_shapes.len(), 1);
            assert!(run.fold(100, h0, &[], &[]).is_empty());
            assert!(run.ledger.early_shapes.is_empty());
            assert!(run.fold(101, h0, &[], &[]).is_empty());
            let end = run.window_end();
            assert_eq!(run.fold(end + 1, chain_hash(false), &[], &[]), vec![(miner(), 1)]);

            let mut run = Run::shaped(48, 60);
            let (_, ev) = run.event(true);
            assert!(run.fold(98, h0, &[ev], &[]).is_empty());
            assert!(run.fold(100, h0, &[], &[]).is_empty());
            assert!(run.fold(101, h0, &[], &[]).is_empty());
            let end = run.window_end();
            assert!(run.fold(end + 1, chain_hash(false), &[], &[]).is_empty());
            assert_eq!(run.ledger.consecutive_misses(&miner()), 0);
        }

        #[test]
        fn shapes_round_trip_through_the_snapshot() {
            let mut run = Run::shaped(48, 60);
            let (_, ev) = run.event(true);
            let h0 = Hash::from_bytes([0u8; 32]);
            assert!(run.fold(98, h0, &[ev], &[]).is_empty());
            let bytes = run.ledger.snapshot().to_bytes();
            assert_eq!(bytes[0], 4);
            assert_eq!(ServiceLedgerSnapshot::from_bytes(&bytes).unwrap(), run.ledger.snapshot());
            assert!(run.fold(100, h0, &[], &[]).is_empty());
            let snap = run.ledger.snapshot();
            let bytes = snap.to_bytes();
            assert_eq!(bytes[0], 4);
            let parsed = ServiceLedgerSnapshot::from_bytes(&bytes).unwrap();
            assert_eq!(parsed, snap);
            assert!(ServiceLedgerSnapshot::from_bytes(&bytes[..bytes.len() - 1]).is_err());
            let mut restored = Run::shaped(48, 60);
            restored.ledger.restore_snapshot(&parsed);
            assert_eq!(restored.ledger.pending.get(&RH).unwrap().prompt_len, 48);
            assert!(restored.fold(101, h0, &[], &[]).is_empty());
            let end = restored.window_end();
            assert!(restored.fold(end + 1, chain_hash(false), &[], &[]).is_empty());
        }

        #[test]
        fn trace_audits_round_trip_through_the_snapshot() {
            let mut run = Run::new(true);
            assert_eq!(run.ledger.snapshot().to_bytes()[0], 1);
            let (deadline, body) = run.until_draw(true, true);
            let snap = run.ledger.snapshot();
            let bytes = snap.to_bytes();
            assert_eq!(bytes[0], 3);
            let parsed = ServiceLedgerSnapshot::from_bytes(&bytes).unwrap();
            assert_eq!(parsed, snap);
            assert!(ServiceLedgerSnapshot::from_bytes(&bytes[..bytes.len() - 1]).is_err());

            let mut restored = Run::new(true);
            restored.ledger.restore_snapshot(&parsed);
            assert_eq!(restored.ledger.trace_audits.get(&TXID).unwrap().deadline_daa, deadline);
            let h = Hash::from_bytes([3u8; 32]);
            assert!(restored.fold(203, h, &[], &[opening(ESCROW_PK, body)]).is_empty());
            assert_eq!(restored.ledger.snapshot().to_bytes()[0], 1);
        }
    }

    mod draw_and_window {
        use super::super::*;
        use keryx_hashes::Hash;

        const RH: [u8; 32] = [7u8; 32];
        const RH2: [u8; 32] = [8u8; 32];
        const REWARD: u64 = 1_000;

        fn id(b: u8) -> Hash {
            Hash::from_bytes([b; 32])
        }

        fn chain(b: u8) -> Hash {
            Hash::from_bytes([b; 32])
        }

        /// One answer: request, responder, traced `(n_prompt, n_pos)`.
        type Answer = ([u8; 32], Hash, Option<(u32, u32)>);

        struct Net {
            ledger: ServiceLedger,
            cohort: Vec<Hash>,
            draw: bool,
            window_v3: bool,
            /// Root rule: a trace root is admitted once.
            unique_root: bool,
            /// Every traced answer carries the same trace root.
            same_root: bool,
            /// Dedup rule: a response accepted twice by one block is judged once.
            dedup: bool,
        }

        impl Net {
            fn new(draw: bool, window_v3: bool, cohort: Vec<Hash>) -> Self {
                let mut ledger = ServiceLedger::default();
                ledger.set_window_v2_activation(0);
                ledger.set_reward_routing_activation(0);
                Self { ledger, cohort, draw, window_v3, unique_root: false, same_root: false, dedup: false }
            }

            /// Folds one chain block accepting `requests` as `(hash, max_tokens)` on tier 0 and
            /// `answers`.
            fn fold(&mut self, daa: u64, chain: Hash, requests: &[([u8; 32], u32)], answers: &[Answer]) -> FoldOutcome {
                let tier = |_: u8, _: u64| -> Option<crate::config::params::TraceRules> { None };
                let events: Vec<TraceResponseEvent> = answers
                    .iter()
                    .filter_map(|(rh, r, s)| {
                        s.map(|(n_prompt, n_pos)| {
                            let mut txid = r.as_bytes();
                            txid[0] ^= rh[0];
                            TraceResponseEvent {
                                response_txid: txid,
                                request_hash: *rh,
                                responder: *r,
                                escrow_pubkey: [0u8; 32],
                                trace_root: if self.same_root { [0xEEu8; 32] } else { txid },
                                n_prompt,
                                n_pos,
                                answer_len: 0,
                                stage: 1,
                                tokens: false,
                            }
                        })
                    })
                    .collect();
                let tr = TraceInputs {
                    chain_hash: chain,
                    responses: &events,
                    openings: &[],
                    tier: &tier,
                    prompt_lens: &[],
                    shape: false,
                    unique_root: self.unique_root,
                    sanction: false,
                    context: false,
                    draw: self.draw,
                    window_v3: self.window_v3,
                    dedup: self.dedup,
                    seed_bind: false,
                };
                let reqs: Vec<([u8; 32], u8, u32)> = requests.iter().map(|(h, m)| (*h, 0, *m)).collect();
                let rewards: Vec<([u8; 32], u64)> = requests.iter().map(|(h, _)| (*h, REWARD)).collect();
                let resp: Vec<([u8; 32], Option<Hash>)> = answers.iter().map(|(rh, r, _)| (*rh, Some(*r))).collect();
                let set = self.cohort.clone();
                self.ledger.on_chain_block_with_traces(
                    daa,
                    &reqs,
                    &rewards,
                    &[],
                    &resp,
                    &[],
                    &[],
                    &tr,
                    |_| true,
                    |_| set.iter().map(|m| (*m, *m)).collect(),
                )
            }

            fn window_end(&self, rh: &[u8; 32]) -> u64 {
                self.ledger.pending.get(rh).unwrap().audit.as_ref().unwrap().window_end_daa
            }

            fn struck(outcome: &FoldOutcome) -> Vec<Hash> {
                let mut v: Vec<Hash> = outcome.misses.iter().map(|m| m.miner).collect();
                v.sort_unstable();
                v
            }
        }

        /// The same traced response accepted twice by one chain block, as an input-less
        /// transaction merged from two blocks of the mergeset.
        fn twice(a: Hash) -> Vec<Answer> {
            vec![(RH, a, Some((100, 300))), (RH, a, Some((100, 300)))]
        }

        #[test]
        fn a_response_accepted_twice_after_the_arming_is_credited_once() {
            let a = id(1);
            let mut net = Net::new(false, false, vec![a]);
            net.unique_root = true;
            net.dedup = true;
            net.fold(100, chain(0), &[(RH, 256)], &[]);
            net.fold(101, chain(0), &[], &[]);
            net.fold(102, chain(0), &[], &twice(a));
            assert_eq!(net.ledger.pending.get(&RH).unwrap().audit.as_ref().unwrap().responded, vec![a]);
            assert_eq!(net.ledger.trace_audits.len(), 1);
            let end = net.window_end(&RH);
            assert!(net.fold(end + 1, chain(0), &[], &[]).misses.is_empty());
        }

        #[test]
        fn a_response_accepted_twice_with_its_request_is_credited_at_the_arming() {
            let a = id(1);
            let mut net = Net::new(false, false, vec![a]);
            net.unique_root = true;
            net.dedup = true;
            net.fold(100, chain(0), &[(RH, 256)], &twice(a));
            net.fold(101, chain(0), &[], &[]);
            assert_eq!(net.ledger.pending.get(&RH).unwrap().audit.as_ref().unwrap().responded, vec![a]);
            let end = net.window_end(&RH);
            assert!(net.fold(end + 1, chain(0), &[], &[]).misses.is_empty());
        }

        #[test]
        fn before_the_dedup_rule_a_response_accepted_twice_is_not_credited() {
            let a = id(1);
            let mut net = Net::new(false, false, vec![a]);
            net.unique_root = true;
            net.fold(100, chain(0), &[(RH, 256)], &[]);
            net.fold(101, chain(0), &[], &[]);
            net.fold(102, chain(0), &[], &twice(a));
            let end = net.window_end(&RH);
            assert_eq!(Net::struck(&net.fold(end + 1, chain(0), &[], &[])), vec![a]);
        }

        #[test]
        fn a_repeated_root_from_another_response_is_still_refused() {
            let (a, b) = (id(1), id(2));
            let mut net = Net::new(false, false, vec![a, b]);
            net.unique_root = true;
            net.same_root = true;
            net.fold(100, chain(0), &[(RH, 256)], &[]);
            net.fold(101, chain(0), &[], &[]);
            net.fold(102, chain(0), &[], &[(RH, a, Some((100, 300)))]);
            net.fold(103, chain(0), &[], &[(RH, b, Some((100, 300)))]);
            let end = net.window_end(&RH);
            assert_eq!(Net::struck(&net.fold(end + 1, chain(0), &[], &[])), vec![b]);
        }

        #[test]
        fn v3_work_matches_the_measured_coefficients() {
            assert_eq!(service_work_daa(2, 5_000, 2_048), 100 + 1_741);
            assert_eq!(service_window_daa_v3(2, 5_000, 2_048), 1_991);
            assert_eq!(reward_draw_window_daa(2, 5_000, 2_048), 996);
            assert_eq!(service_window_daa_v3(4, 5_000, 2_048), 150 + 100 + 717);
            assert_eq!(reward_draw_window_daa(4, 5_000, 2_048), 484);
            assert_eq!(service_work_daa(1, 0, 4_096), service_work_daa(1, 0, 10_000));
        }

        #[test]
        fn without_the_draw_the_first_credit_wins() {
            let (a, b) = (id(1), id(2));
            let mut net = Net::new(false, false, vec![a, b]);
            net.fold(100, chain(0), &[(RH, 256)], &[]);
            net.fold(101, chain(0), &[], &[]);
            let out = net.fold(102, chain(0), &[], &[(RH, b, None)]);
            assert_eq!(out.rewards.iter().map(|r| r.winner).collect::<Vec<_>>(), vec![b]);
        }

        #[test]
        fn the_draw_picks_among_members_credited_within_its_window() {
            let (a, b, c) = (id(1), id(2), id(3));
            let mut net = Net::new(true, false, vec![a, b, c]);
            net.fold(100, chain(0), &[(RH, 256)], &[]);
            net.fold(101, chain(0), &[], &[]);
            assert!(net.fold(102, chain(0), &[], &[(RH, b, None)]).rewards.is_empty());
            let close = 102 + reward_draw_window_daa(0, 0, 256);
            assert_eq!(close, 102 + 191);
            assert!(net.fold(close - 1, chain(0), &[], &[(RH, a, None)]).rewards.is_empty());
            let closing = chain(9);
            let out = net.fold(close, closing, &[], &[(RH, c, None)]);
            let winner = [a, b, c][assign_index(&reward_draw_seed(&closing, &RH), 3).unwrap()];
            assert_eq!(out.rewards.len(), 1);
            assert_eq!(out.rewards[0].winner, winner);
            assert_eq!(out.rewards[0].amount, REWARD);
            assert_eq!(net.ledger.pending.get(&RH).unwrap().winner, Some(winner));
            assert!(net.ledger.pending.get(&RH).unwrap().draw.is_none());
        }

        #[test]
        fn a_member_credited_after_the_close_is_not_a_candidate() {
            let (a, b) = (id(1), id(2));
            let mut net = Net::new(true, false, vec![a, b]);
            net.fold(100, chain(0), &[(RH, 256)], &[]);
            net.fold(101, chain(0), &[], &[]);
            net.fold(102, chain(0), &[], &[(RH, a, None)]);
            let close = 102 + reward_draw_window_daa(0, 0, 256);
            let out = net.fold(close, chain(9), &[], &[]);
            assert_eq!(out.rewards[0].winner, a);
            assert!(net.fold(close + 1, chain(9), &[], &[(RH, b, None)]).rewards.is_empty());
        }

        #[test]
        fn early_answers_open_the_draw_at_the_arming() {
            let (a, b) = (id(1), id(2));
            let mut net = Net::new(true, false, vec![a, b]);
            net.fold(100, chain(0), &[(RH, 256)], &[(RH, a, None), (RH, b, None)]);
            assert!(net.fold(101, chain(0), &[], &[]).rewards.is_empty());
            let draw = net.ledger.pending.get(&RH).unwrap().draw.clone().unwrap();
            assert_eq!(draw.open_daa, 101);
            assert_eq!(draw.candidates.len(), 2);
        }

        #[test]
        fn a_draw_with_no_candidate_left_burns_the_reward() {
            let (a, b) = (id(1), id(2));
            let mut net = Net::new(true, false, vec![a, b]);
            net.fold(100, chain(0), &[(RH, 256)], &[]);
            net.fold(101, chain(0), &[], &[]);
            net.fold(102, chain(0), &[], &[(RH, a, None)]);
            net.ledger.drop_draw_candidates(&RH, &[a]);
            let close = 102 + reward_draw_window_daa(0, 0, 256);
            assert!(net.fold(close, chain(9), &[], &[]).rewards.is_empty());
            let req = net.ledger.pending.get(&RH).unwrap();
            assert_eq!((req.winner, req.reward), (None, 0));
            assert!(net.fold(close + 1, chain(9), &[], &[(RH, b, None)]).rewards.is_empty());
        }

        #[test]
        fn an_open_draw_closes_before_its_request_is_judged() {
            let a = id(1);
            let mut net = Net::new(true, true, vec![a]);
            net.fold(100, chain(0), &[(RH, 256)], &[]);
            net.fold(101, chain(0), &[], &[]);
            let end = net.window_end(&RH);
            net.fold(end, chain(0), &[], &[(RH, a, Some((0, 256)))]);
            let out = net.fold(end + 1, chain(9), &[], &[]);
            assert_eq!(out.rewards.iter().map(|r| r.winner).collect::<Vec<_>>(), vec![a]);
            assert!(net.ledger.pending.is_empty());
        }

        #[test]
        fn an_open_draw_survives_the_snapshot() {
            let (a, b) = (id(1), id(2));
            let mut net = Net::new(true, false, vec![a, b]);
            net.fold(100, chain(0), &[(RH, 256)], &[]);
            net.fold(101, chain(0), &[], &[]);
            net.fold(102, chain(0), &[], &[(RH, b, None)]);
            let snap = net.ledger.snapshot();
            let bytes = snap.to_bytes();
            assert_eq!(bytes[0], SNAPSHOT_ENCODING_VERSION_QUEUE);
            let parsed = ServiceLedgerSnapshot::from_bytes(&bytes).unwrap();
            assert_eq!(parsed, snap);
            assert!(ServiceLedgerSnapshot::from_bytes(&bytes[..bytes.len() - 1]).is_err());
            let mut restored = Net::new(true, false, vec![a, b]);
            restored.ledger.restore_snapshot(&parsed);
            let close = 102 + reward_draw_window_daa(0, 0, 256);
            assert_eq!(restored.fold(close, chain(9), &[], &[]).rewards, net.fold(close, chain(9), &[], &[]).rewards);
        }

        #[test]
        fn the_v3_deadline_is_judged_on_the_median_answer() {
            let (a, b, c, d) = (id(1), id(2), id(3), id(4));
            let mut net = Net::new(false, true, vec![a, b, c, d]);
            net.fold(100, chain(0), &[(RH, 256)], &[]);
            net.fold(101, chain(0), &[], &[]);
            // full budget, no prompt bytes: 150 + 231
            assert_eq!(net.window_end(&RH), 101 + 381);
            // 200 generated tokens on 100 prompt tokens: allowed until 101 + 150 + 2 + 180
            let allowed = 101 + service_window_daa_v3(0, 100, 200);
            assert_eq!(allowed, 433);
            net.fold(150, chain(0), &[], &[(RH, a, Some((100, 300)))]);
            net.fold(allowed, chain(0), &[], &[(RH, b, Some((100, 300)))]);
            // a padded answer does not move the median nor buy its author time
            net.fold(allowed + 1, chain(0), &[], &[(RH, c, Some((100, 356)))]);
            let out = net.fold(net.window_end(&RH) + 1, chain(0), &[], &[]);
            assert_eq!(Net::struck(&out), vec![c, d]);
            assert_eq!(out.late, vec![(RH, c)]);
            assert_eq!(net.ledger.consecutive_misses(&a), 0);
            assert_eq!(net.ledger.consecutive_misses(&b), 0);
        }

        #[test]
        fn without_any_answer_the_v3_window_runs_the_full_budget() {
            let a = id(1);
            let mut net = Net::new(false, true, vec![a]);
            net.fold(100, chain(0), &[(RH, 256)], &[]);
            net.fold(101, chain(0), &[], &[]);
            let end = net.window_end(&RH);
            assert!(net.fold(end, chain(0), &[], &[]).misses.is_empty());
            assert_eq!(Net::struck(&net.fold(end + 1, chain(0), &[], &[])), vec![a]);
        }

        #[test]
        fn an_untraced_answer_is_judged_on_the_full_budget() {
            let a = id(1);
            let mut net = Net::new(false, true, vec![a]);
            net.fold(100, chain(0), &[(RH, 256)], &[]);
            net.fold(101, chain(0), &[], &[]);
            let end = net.window_end(&RH);
            net.fold(end, chain(0), &[], &[(RH, a, None)]);
            assert!(net.fold(end + 1, chain(0), &[], &[]).misses.is_empty());
        }

        #[test]
        fn the_queue_ahead_extends_the_deadline() {
            let (a, b) = (id(1), id(2));
            let mut net = Net::new(false, true, vec![a, b]);
            net.fold(100, chain(0), &[(RH, 256)], &[]);
            net.fold(101, chain(0), &[], &[]);
            let first_end = net.window_end(&RH);
            // b already served the first request: only a has it queued
            net.fold(102, chain(0), &[(RH2, 256)], &[(RH, b, Some((0, 256)))]);
            net.fold(103, chain(0), &[], &[]);
            let queued = service_work_daa(0, 0, 256);
            assert_eq!(net.window_end(&RH2), 103 + 381 + queued);
            assert_eq!(net.window_end(&RH), first_end);
            let w = net.ledger.pending.get(&RH2).unwrap().audit.as_ref().unwrap().v3.clone().unwrap();
            assert_eq!(w.queue, vec![(queued, false), (0, false)]);
            net.fold(300, chain(0), &[], &[(RH, a, Some((0, 256)))]);
            // past its own work, inside its own work plus the queue
            net.fold(103 + 381 + queued, chain(0), &[], &[(RH2, a, Some((0, 256))), (RH2, b, Some((0, 256)))]);
            assert!(!net.ledger.pending.contains_key(&RH));
            let out = net.fold(net.window_end(&RH2) + 1, chain(0), &[], &[]);
            assert_eq!(Net::struck(&out), vec![b]);
            assert_eq!(out.late, vec![(RH2, b)]);
        }

        #[test]
        fn no_lateness_past_the_queue_cap() {
            let a = id(1);
            let mut net = Net::new(false, true, vec![a]);
            let requests: Vec<([u8; 32], u32)> =
                (0..SERVICE_QUEUE_CAP as u8 + 2).map(|i| ([i + 10; 32], 256)).collect();
            net.fold(100, chain(0), &requests, &[]);
            net.fold(101, chain(0), &[], &[]);
            let rh = requests[0].0;
            let w = net.ledger.pending.get(&rh).unwrap().audit.as_ref().unwrap().v3.clone().unwrap();
            assert_eq!(w.queue, vec![(service_work_daa(0, 0, 256) * (SERVICE_QUEUE_CAP as u64 + 1), true)]);
            let end = net.window_end(&rh);
            let answers: Vec<Answer> = requests.iter().map(|(h, _)| (*h, a, Some((0, 1)))).collect();
            net.fold(end, chain(0), &[], &answers);
            assert!(net.fold(end + 1, chain(0), &[], &[]).misses.is_empty());
        }

        #[test]
        fn the_v3_deadline_stays_under_the_ledger_horizon() {
            let a = id(1);
            let mut net = Net::new(false, true, vec![a]);
            let requests: Vec<([u8; 32], u32)> = (0..40u8).map(|i| ([i + 10; 32], 4_096)).collect();
            net.fold(100, chain(0), &requests, &[]);
            net.fold(101, chain(0), &[], &[]);
            assert_eq!(net.window_end(&requests[0].0), 100 + SERVICE_LEDGER_HORIZON_DAA - 2);
            let out = net.fold(100 + SERVICE_LEDGER_HORIZON_DAA - 1, chain(0), &[], &[]);
            assert_eq!(out.misses.len(), 1);
        }

        #[test]
        fn a_v3_window_survives_the_snapshot() {
            let (a, b) = (id(1), id(2));
            let mut net = Net::new(false, true, vec![a, b]);
            net.fold(100, chain(0), &[(RH, 256)], &[]);
            net.fold(101, chain(0), &[], &[]);
            net.fold(150, chain(0), &[], &[(RH, a, Some((100, 300))), (RH, b, None)]);
            let snap = net.ledger.snapshot();
            let bytes = snap.to_bytes();
            assert_eq!(bytes[0], SNAPSHOT_ENCODING_VERSION_QUEUE);
            let parsed = ServiceLedgerSnapshot::from_bytes(&bytes).unwrap();
            assert_eq!(parsed, snap);
            let mut restored = Net::new(false, true, vec![a, b]);
            restored.ledger.restore_snapshot(&parsed);
            assert_eq!(restored.ledger.snapshot().to_bytes(), bytes);
        }
    }
}
