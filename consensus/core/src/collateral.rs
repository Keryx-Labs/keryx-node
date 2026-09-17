use keryx_hashes::{Hash, Hasher, TransactionHash};
use keryx_utils::mem_size::MemSizeEstimator;
use serde::{Deserialize, Serialize};

use crate::config::params::{NetworkModelLayout, NETWORK_MODEL_MAINNET};
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

/// Verifies a V2 responder signature: schnorr by `escrow_pubkey` over the domain-tagged
/// blake2b-256 of the v1 payload bytes.
pub fn verify_responder_signature(escrow_pubkey: &[u8; 32], signature: &[u8; 64], signed_bytes: &[u8]) -> bool {
    let mut hasher = blake2b_simd::Params::new().hash_length(32).to_state();
    hasher.update(RESPONDER_SIG_DOMAIN);
    hasher.update(signed_bytes);
    let mut msg = [0u8; 32];
    msg.copy_from_slice(hasher.finalize().as_bytes());
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
/// Window base for a request to the network model (~10 minutes at 10 BPS): pipeline assembly
/// across several miners before the first token.
pub const SERVICE_WINDOW_BASE_DAA_NETWORK_MODEL: u64 = 6_000;
/// Per-token allowance for the network model: one network round trip per shard and per token.
pub const SERVICE_WINDOW_NETWORK_MODEL_PER_TOKEN_DAA: u64 = 10;

/// DAA window an assigned miner has, from his assignment seed block, for the request to be served
/// before it counts as a miss: a fixed base plus a per-requested-token allowance floored at the
/// generation speed of the tier's model class (measured medians ~7-10 tok/s, ×2 margin).
pub fn service_window_daa(tier: u8, max_tokens: u32) -> u64 {
    service_window_daa_at(tier, max_tokens, false)
}

/// [`service_window_daa`] with the base selected by whether `service_bond_v2_activation` is
/// active at the daa the audit arms.
pub fn service_window_daa_at(tier: u8, max_tokens: u32, v2: bool) -> u64 {
    use crate::config::params::NETWORK_MODEL_TIER;
    let per_token_daa: u64 = match tier {
        0..=2 => 2, // 0.2 s/token — 5 tok/s floor
        3 => 3,     // 0.3 s/token
        4 => 4,     // 0.4 s/token — 2.5 tok/s floor
        t if t >= NETWORK_MODEL_TIER => SERVICE_WINDOW_NETWORK_MODEL_PER_TOKEN_DAA,
        _ => 4,
    };
    let mut base = if v2 { SERVICE_WINDOW_BASE_DAA_V2 } else { SERVICE_WINDOW_BASE_DAA };
    if tier >= NETWORK_MODEL_TIER {
        base = base.max(SERVICE_WINDOW_BASE_DAA_NETWORK_MODEL);
    }
    base + max_tokens.min(AI_REQUEST_MAX_TOKENS_CAP) as u64 * per_token_daa
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
    use crate::config::params::tier_serves;
    let mut set: Vec<(Hash, Hash)> =
        recent.iter().filter(|(_, t, _)| tier_serves(*t, target_tier)).map(|(m, _, e)| (*m, *e)).collect();
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
    /// Network-model audits only: the identity drawn for each shard tier, tier order. The head
    /// tier's draw is struck on silence; every other draw is struck when a credited response
    /// was served without it. Empty for a lineup audit.
    links: Vec<(u8, Hash)>,
    /// Network-model audits only: `(identity, shard tier)` of every eligible producer, sorted —
    /// a signer must hold the tier it signs for.
    members: Vec<(Hash, u8)>,
}

/// An armed network-model audit as the pipeline head needs it: the escrow keys drawn for each
/// shard tier (a substitute of the same tier is also credited, but the drawn one is struck).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PipelineAssignment {
    pub request_hash: [u8; 32],
    pub accepted_daa: u64,
    pub window_end_daa: u64,
    /// `(shard tier, escrow pubkey)`, tier order; an identity delegating from several escrow
    /// keys appears once per key.
    pub links: Vec<(u8, [u8; 32])>,
}

fn hex32(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

fn unhex32(s: &str) -> Option<[u8; 32]> {
    if s.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, chunk) in s.as_bytes().chunks(2).enumerate() {
        out[i] = u8::from_str_radix(std::str::from_utf8(chunk).ok()?, 16).ok()?;
    }
    Some(out)
}

impl PipelineAssignment {
    /// Wire form for the block template: `hash:accepted:window_end:tier-escrowhex,...`.
    pub fn encode(&self) -> String {
        let links: Vec<String> = self.links.iter().map(|(t, k)| format!("{}-{}", t, hex32(k))).collect();
        format!("{}:{}:{}:{}", hex32(&self.request_hash), self.accepted_daa, self.window_end_daa, links.join(","))
    }

    pub fn decode(s: &str) -> Option<Self> {
        let mut parts = s.split(':');
        let request_hash = unhex32(parts.next()?)?;
        let accepted_daa: u64 = parts.next()?.parse().ok()?;
        let window_end_daa: u64 = parts.next()?.parse().ok()?;
        let mut links = Vec::new();
        for item in parts.next()?.split(',').filter(|i| !i.is_empty()) {
            let (t, k) = item.split_once('-')?;
            links.push((t.parse().ok()?, unhex32(k)?));
        }
        Some(Self { request_hash, accepted_daa, window_end_daa, links })
    }
}

/// A pipeline response's verified links, keyed to its `(request_hash, head escrow key)`
/// response entry: `(shard tier, link escrow key)` per signing link.
pub type ResponseLinks = ([u8; 32], Hash, Vec<(u8, Hash)>);

/// Domain of the derived reward keys of a pipeline response's link shares.
const REWARD_SHARE_DOMAIN: &[u8] = b"KeryxRewardShareV1";

/// Reward-store key of the share paid to the link of `tier` for `request_hash`. The head's share
/// keeps the request hash itself as its key.
pub fn reward_share_key(request_hash: &[u8; 32], tier: u8) -> [u8; 32] {
    let mut hasher = blake2b_simd::Params::new().hash_length(32).to_state();
    hasher.update(REWARD_SHARE_DOMAIN);
    hasher.update(request_hash);
    hasher.update(&[tier]);
    let mut out = [0u8; 32];
    out.copy_from_slice(hasher.finalize().as_bytes());
    out
}

/// Per-shard draw seed: the request hash mixed with the shard tier, so the draws of one request
/// are independent across tiers.
fn shard_draw_seed(request_hash: &[u8; 32], tier: u8) -> [u8; 32] {
    let mut hasher = blake2b_simd::Params::new().hash_length(32).to_state();
    hasher.update(request_hash);
    hasher.update(&[tier]);
    let mut out = [0u8; 32];
    out.copy_from_slice(hasher.finalize().as_bytes());
    out
}

/// VRAM-weighted split of `reward` between the head and its links: `(identity, amount)` per
/// signer, the head first and carrying the rounding remainder.
pub fn split_pipeline_reward(layout: &NetworkModelLayout, reward: u64, head: (u8, Hash), links: &[(u8, Hash)]) -> Vec<(Hash, u64)> {
    let total: u128 = layout.tier_weight(head.0) as u128 + links.iter().map(|(t, _)| layout.tier_weight(*t) as u128).sum::<u128>();
    if total == 0 {
        return vec![(head.1, reward)];
    }
    let mut out = Vec::with_capacity(links.len() + 1);
    let mut paid: u64 = 0;
    for (tier, identity) in links {
        let amount = (reward as u128 * layout.tier_weight(*tier) as u128 / total) as u64;
        paid += amount;
        out.push((*identity, amount));
    }
    out.insert(0, (head.1, reward - paid));
    out
}

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
    /// The network model of this network; `None` = mainnet. Configuration, like the
    /// activation daas — installed on every fold entry, never snapshotted.
    network_model: Option<&'static NetworkModelLayout>,
}

impl ServiceLedger {
    fn layout(&self) -> &'static NetworkModelLayout {
        self.network_model.unwrap_or(&NETWORK_MODEL_MAINNET)
    }

    /// Installs the network model the pipeline audits follow.
    pub fn set_network_model(&mut self, layout: &'static NetworkModelLayout) {
        self.network_model = Some(layout);
    }

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
        self.fold_inner(daa, requests, &[], responses, escrows, &[], &[], None, is_established, cohort)
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
        response_links: &[ResponseLinks],
        is_established: impl Fn(&Hash) -> bool,
        cohort: impl FnMut(u8) -> Vec<(Hash, Hash)>,
    ) -> FoldOutcome {
        self.fold_inner(daa, requests, request_rewards, responses, escrows, producers, response_links, None, is_established, cohort)
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
        self.fold_inner(daa, requests, &[], responses, escrows, &[], &[], Some(is_burned), |_| true, cohort);
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
        response_links: &[ResponseLinks],
        is_burned: &dyn Fn(&crate::tx::TransactionOutpoint) -> bool,
        cohort: impl FnMut(u8) -> Vec<(Hash, Hash)>,
    ) {
        self.fold_inner(daa, requests, request_rewards, responses, escrows, producers, response_links, Some(is_burned), |_| true, cohort);
    }

    #[allow(clippy::too_many_arguments)]
    fn fold_inner(
        &mut self,
        daa: u64,
        requests: &[([u8; 32], u8, u32)],
        request_rewards: &[([u8; 32], u64)],
        responses: &[([u8; 32], Option<Hash>)],
        escrows: &[(Hash, EscrowClaim)],
        producers: &[(Hash, crate::tx::ScriptPublicKey)],
        response_links: &[ResponseLinks],
        warmup_burned: Option<&dyn Fn(&crate::tx::TransactionOutpoint) -> bool>,
        is_established: impl Fn(&Hash) -> bool,
        mut cohort: impl FnMut(u8) -> Vec<(Hash, Hash)>,
    ) -> FoldOutcome {
        use crate::config::params::NETWORK_MODEL_TIER;
        let layout = self.layout();
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

        let mut outcome = FoldOutcome::default();
        outcome.sightings = sightings;
        outcome.expired = expired;

        self.pending.retain(|_, r| r.accepted_daa + SERVICE_LEDGER_HORIZON_DAA > daa);
        self.early_responses.retain(|_, (seen, _)| *seen + SERVICE_EARLY_RESPONSE_HORIZON_DAA > daa);

        // This block's own requests are admitted before its responses are folded: a response
        // accepted in the same chain block as its request must find the request pending, or it is
        // dropped and its author struck for an answer he did give. Arming still waits for the next
        // block (`daa > accepted_daa` below), so the cohort is unchanged.
        let gated = self.reward_routing_daa.is_some_and(|a| daa >= a);
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
            let early = self.early_responses.remove(rh).map(|(_, keys)| keys).unwrap_or_default();
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
            });
            // Answers that beat their own request to acceptance.
            if entry.audit.is_none() {
                for key in early {
                    if !entry.early_responders.contains(&key) {
                        entry.early_responders.push(key);
                    }
                }
            }
        }

        // An authenticated response (signed by a delegated escrow key) marks every cohort
        // identity that delegated to it as having served this audit and resets their streak.
        // Anyone else's response is ignored by the ledger. A response landing before the audit
        // arms is parked on the request and credited when it does.
        for (rh, responder) in responses {
            let Some(r) = responder else { continue };
            let mut served: Vec<Hash> = Vec::new();
            let mut pipeline: Option<(Hash, Vec<(u8, Hash)>)> = None;
            {
                let Some(req) = self.pending.get_mut(rh) else {
                    let entry = self.early_responses.entry(*rh).or_insert((daa, Vec::new()));
                    if !entry.1.contains(r) {
                        entry.1.push(*r);
                    }
                    continue;
                };
                let Some(audit) = req.audit.as_mut() else {
                    if !req.early_responders.contains(r) {
                        req.early_responders.push(*r);
                    }
                    continue;
                };
                let matched: Vec<Hash> = audit.delegations.iter().filter(|(e, _)| e == r).map(|(_, id)| *id).collect();
                if audit.links.is_empty() {
                    for identity in matched {
                        if audit.cohort.binary_search(&identity).is_ok() && !audit.responded.contains(&identity) {
                            audit.responded.push(identity);
                            served.push(identity);
                        }
                    }
                } else {
                    // A pipeline response counts only complete: the head holds the head tier and
                    // every other shard tier is signed by exactly one link holding it. Anything
                    // else is no response at all — the drawn head answers for it at window close.
                    let Some((_, _, links)) = response_links.iter().find(|(h, k, _)| h == rh && k == r) else { continue };
                    let head_ids: Vec<Hash> =
                        matched.iter().copied().filter(|id| audit.members.binary_search(&(*id, layout.head_tier())).is_ok()).collect();
                    let Some(head) = head_ids.first().copied() else { continue };
                    let mut expected: Vec<u8> = layout.shard_tiers().filter(|t| *t != layout.head_tier()).collect();
                    let mut signed: Vec<u8> = links.iter().map(|(t, _)| *t).collect();
                    signed.sort_unstable();
                    expected.sort_unstable();
                    if signed != expected {
                        continue;
                    }
                    let mut link_ids: Vec<(u8, Hash)> = Vec::with_capacity(links.len());
                    let mut complete = true;
                    for (tier, key) in links {
                        let holder = audit
                            .delegations
                            .iter()
                            .filter(|(e, _)| e == key)
                            .map(|(_, id)| *id)
                            .find(|id| audit.members.binary_search(&(*id, *tier)).is_ok());
                        match holder {
                            Some(id) => link_ids.push((*tier, id)),
                            None => {
                                complete = false;
                                break;
                            }
                        }
                    }
                    if !complete {
                        continue;
                    }
                    for identity in head_ids.iter().chain(link_ids.iter().map(|(_, id)| id)) {
                        if !audit.responded.contains(identity) {
                            audit.responded.push(*identity);
                            served.push(*identity);
                        }
                    }
                    pipeline = Some((head, link_ids));
                }
            }
            if let Some((head, link_ids)) = pipeline.take() {
                self.maybe_award_pipeline(rh, head, &link_ids, warmup, &mut outcome);
            } else if let Some(first) = served.first().copied() {
                self.maybe_award(rh, first, warmup, &mut outcome);
            }
            for identity in served {
                // A warmup fold leaves strikes alone: the baseline already carries this reset.
                if !warmup && self.strike_state(&identity).is_some_and(|e| e.count > 0) {
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
        }

        let hashes: Vec<[u8; 32]> = self.pending.keys().copied().collect();
        let mut served_at_arm: Vec<Hash> = Vec::new();
        for rh in hashes {
            let req = self.pending.get(&rh).unwrap();
            match &req.audit {
                Some(a) if daa > a.window_end_daa => {
                    let audit = a.clone();
                    // Lineup: every silent cohort member. Pipeline: the drawn head alone when
                    // nothing was served, else every drawn identity the served response did
                    // without (a substituted head included).
                    let silent: Vec<Hash> = if audit.links.is_empty() {
                        audit.cohort.iter().filter(|m| !audit.responded.contains(m)).copied().collect()
                    } else if audit.responded.is_empty() {
                        audit.links.iter().filter(|(t, _)| *t == layout.head_tier()).map(|(_, id)| *id).collect()
                    } else {
                        let mut v: Vec<Hash> = audit.links.iter().filter(|(_, id)| !audit.responded.contains(id)).map(|(_, id)| *id).collect();
                        v.sort_unstable();
                        v.dedup();
                        v
                    };
                    for miner in silent.iter() {
                        if warmup {
                            // The miss is already persisted: its burns came in through `is_burned`
                            // and its strike lives in the baseline.
                            continue;
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
                        });
                    }
                    self.pending.remove(&rh);
                }
                Some(_) => {}
                None if daa > req.accepted_daa && req.tier == NETWORK_MODEL_TIER => {
                    // One draw per shard tier; a tier nobody produces means the network cannot
                    // assemble the model — the request is dropped, nobody is struck. Responses
                    // that beat the arming are not creditable (they carry no links here).
                    let mut ids: Vec<Hash> = Vec::new();
                    let mut delegations: Vec<(Hash, Hash)> = Vec::new();
                    let mut members: Vec<(Hash, u8)> = Vec::new();
                    let mut links: Vec<(u8, Hash)> = Vec::new();
                    let mut complete = true;
                    for tier in layout.shard_tiers() {
                        let set = cohort(tier);
                        if set.is_empty() {
                            complete = false;
                            break;
                        }
                        let mut tier_ids: Vec<Hash> = set.iter().map(|(id, _)| *id).collect();
                        tier_ids.sort_unstable();
                        tier_ids.dedup();
                        let drawn = draw_assignment(&tier_ids, &[], &shard_draw_seed(&rh, tier)).unwrap();
                        links.push((tier, drawn));
                        members.extend(tier_ids.iter().map(|id| (*id, tier)));
                        ids.extend(tier_ids);
                        delegations.extend(set.iter().map(|(id, esc)| (*esc, *id)));
                    }
                    if !complete {
                        self.pending.remove(&rh);
                    } else {
                        ids.sort_unstable();
                        ids.dedup();
                        delegations.sort_unstable();
                        delegations.dedup();
                        members.sort_unstable();
                        members.dedup();
                        let window = service_window_daa_at(req.tier, req.max_tokens, self.v2_at(daa));
                        let req = self.pending.get_mut(&rh).unwrap();
                        req.early_responders.clear();
                        req.audit = Some(Audit {
                            cohort: ids,
                            delegations,
                            responded: Vec::new(),
                            window_end_daa: daa + window,
                            links,
                            members,
                        });
                    }
                }
                None if daa > req.accepted_daa => {
                    let set = cohort(req.tier);
                    if set.is_empty() {
                        self.pending.remove(&rh);
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
                        let early = std::mem::take(&mut self.pending.get_mut(&rh).unwrap().early_responders);
                        let mut responded: Vec<Hash> = Vec::new();
                        for key in early.iter() {
                            for identity in delegations.iter().filter(|(e, _)| e == key).map(|(_, id)| id) {
                                if ids.binary_search(identity).is_ok() && !responded.contains(identity) {
                                    responded.push(*identity);
                                    served_at_arm.push(*identity);
                                }
                            }
                        }
                        let first_credit = responded.first().copied();
                        self.pending.get_mut(&rh).unwrap().audit = Some(Audit {
                            cohort: ids,
                            delegations,
                            responded,
                            window_end_daa: daa + window,
                            links: Vec::new(),
                            members: Vec::new(),
                        });
                        if let Some(first) = first_credit {
                            self.maybe_award(&rh, first, warmup, &mut outcome);
                        }
                    }
                }
                None => {}
            }
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

    /// The armed network-model audits, oldest first.
    pub fn pipeline_assignments(&self) -> Vec<PipelineAssignment> {
        let mut out: Vec<PipelineAssignment> = self
            .pending
            .iter()
            .filter_map(|(rh, req)| {
                let audit = req.audit.as_ref()?;
                if audit.links.is_empty() {
                    return None;
                }
                let mut links = Vec::new();
                for (tier, identity) in audit.links.iter() {
                    for (escrow, id) in audit.delegations.iter() {
                        if id == identity {
                            links.push((*tier, escrow.as_bytes()));
                        }
                    }
                }
                Some(PipelineAssignment { request_hash: *rh, accepted_daa: req.accepted_daa, window_end_daa: audit.window_end_daa, links })
            })
            .collect();
        out.sort_by_key(|a| (a.accepted_daa, a.request_hash));
        out
    }

    /// The miner's still-locked escrow claims, chain order (newest last).
    pub fn vault_claims(&self, miner: &Hash) -> Vec<EscrowClaim> {
        self.vault.get(miner).map(|claims| claims.iter().copied().collect()).unwrap_or_default()
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

    /// Awards a served pipeline response: the vaulted reward split by shard VRAM between the
    /// head (the recorded winner, keyed by the request hash) and its links (keyed by
    /// [`reward_share_key`]). Same routing and once-only rules as [`Self::maybe_award`].
    fn maybe_award_pipeline(&mut self, rh: &[u8; 32], head: Hash, links: &[(u8, Hash)], warmup: bool, outcome: &mut FoldOutcome) {
        let layout = self.layout();
        if self.reward_routing_daa.is_none() {
            return;
        }
        let routed_from = self.reward_routing_daa.unwrap();
        let Some(req) = self.pending.get_mut(rh) else { return };
        if req.accepted_daa < routed_from || req.winner.is_some() || req.reward == 0 {
            return;
        }
        req.winner = Some(head);
        let reward = req.reward;
        if warmup {
            return;
        }
        let shares = split_pipeline_reward(layout, reward, (layout.head_tier(), head), links);
        for (i, (identity, amount)) in shares.into_iter().enumerate() {
            let key = if i == 0 { *rh } else { reward_share_key(rh, links[i - 1].0) };
            let spk = self.producer_spk.get(&identity).cloned();
            outcome.rewards.push(ServiceReward { request_hash: key, winner: identity, amount, spk });
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
}

const SNAPSHOT_ENCODING_VERSION: u8 = 1;

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
        out.push(SNAPSHOT_ENCODING_VERSION);
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
                    // Flag 2 carries the pipeline fields; a lineup audit keeps the flag-1 bytes
                    // it always had, so every pre-H14 snapshot hash is unchanged.
                    out.push(if a.links.is_empty() { 1 } else { 2 });
                    put_hashes(&mut out, &a.cohort);
                    out.extend_from_slice(&(a.delegations.len() as u32).to_le_bytes());
                    for (e, id) in a.delegations.iter() {
                        out.extend_from_slice(&e.as_bytes());
                        out.extend_from_slice(&id.as_bytes());
                    }
                    put_hashes(&mut out, &a.responded);
                    out.extend_from_slice(&a.window_end_daa.to_le_bytes());
                    if !a.links.is_empty() {
                        out.extend_from_slice(&(a.links.len() as u32).to_le_bytes());
                        for (t, id) in a.links.iter() {
                            out.push(*t);
                            out.extend_from_slice(&id.as_bytes());
                        }
                        out.extend_from_slice(&(a.members.len() as u32).to_le_bytes());
                        for (id, t) in a.members.iter() {
                            out.extend_from_slice(&id.as_bytes());
                            out.push(*t);
                        }
                    }
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
        out
    }

    /// Parses [`Self::to_bytes`] output; rejects malformed, unordered or trailing data.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let mut r = Reader { bytes, pos: 0 };
        if r.u8()? != SNAPSHOT_ENCODING_VERSION {
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
                flag @ (1 | 2) => {
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
                    let mut links = Vec::new();
                    let mut members = Vec::new();
                    if flag == 2 {
                        let n = r.u32()? as usize;
                        if n == 0 {
                            return Err("malformed audit links".into());
                        }
                        for _ in 0..n {
                            let t = r.u8()?;
                            let id = r.hash()?;
                            links.push((t, id));
                        }
                        let m = r.u32()? as usize;
                        for _ in 0..m {
                            let id = r.hash()?;
                            let t = r.u8()?;
                            members.push((id, t));
                        }
                    }
                    Some(Audit { cohort, delegations, responded, window_end_daa, links, members })
                }
                _ => return Err("malformed audit".into()),
            };
            let early_responders = r.hashes()?;
            let req = PendingRequest { tier, max_tokens, accepted_daa, reward, winner, audit, early_responders };
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
    }
}

#[cfg(test)]
mod tests {

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
        let wide = ProductionIndexSnapshot { window_daa: (0..=150u64).map(|i| 1 << 40 | i * 300).collect(), ..snap.clone() };
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
        eligible_pairs, service_window_daa, service_window_daa_at, strike_penalty, strike_penalty_at, update_strikes,
        FoldOutcome, ServiceLedger, ServicePenalty, ServiceReward,
        StrikeEntry, AI_REQUEST_MAX_TOKENS_CAP, SERVICE_EARLY_RESPONSE_HORIZON_DAA, SERVICE_LEDGER_HORIZON_DAA,
        SERVICE_STRIKE_INTERVAL_DAA, STRIKE_1_BURN_CLAIMS,
    };
    use keryx_hashes::Hash;

    use crate::config::params::{
        network_model_shard_tiers, network_model_tier_weight, NETWORK_MODEL_HEAD_TIER, NETWORK_MODEL_MAINNET, NETWORK_MODEL_SHARDS,
        NETWORK_MODEL_TIER,
    };
    use super::{reward_share_key, split_pipeline_reward, PipelineAssignment, ServiceLedgerSnapshot};

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
            &[],
            |_| true,
            cohort_of(&set),
        );
        assert!(out.rewards.is_empty());
        let out =
            ledger.on_chain_block_with_rewards(101, &[], &[], &[(rh_old, Some(a))], &[], &[], &[], |_| true, cohort_of(&set));
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
            &[],
            |_| true,
            cohort_of(&set),
        );
        assert!(out.rewards.is_empty());
        // First accepted response after arming: `b` wins (no known script — stays burned).
        let out = ledger.on_chain_block_with_rewards(302, &[], &[], &[(rh, Some(b))], &[], &[], &[], |_| true, cohort_of(&set));
        assert_eq!(out.rewards.len(), 1);
        assert_eq!(out.rewards[0], ServiceReward { request_hash: rh, winner: b, amount: 9_000, spk: None });
        // A later responder wins nothing.
        let out = ledger.on_chain_block_with_rewards(303, &[], &[], &[(rh, Some(a))], &[], &[], &[], |_| true, cohort_of(&set));
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
            &[],
            |_| true,
            cohort_of(&set),
        );
        let out = ledger.on_chain_block_with_rewards(312, &[], &[], &[(rh2, Some(a))], &[], &[], &[], |_| true, cohort_of(&set));
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
    fn network_model_requests_draw_from_every_shard() {
        use crate::config::params::NETWORK_MODEL_TIER;
        let a = Hash::from_bytes([1u8; 32]);
        let b = Hash::from_bytes([2u8; 32]);
        let c = Hash::from_bytes([3u8; 32]);
        let e = Hash::from_bytes([9u8; 32]);
        let recent = [(a, NETWORK_MODEL_TIER + 1, e), (b, NETWORK_MODEL_TIER + 3, e), (c, 4u8, e)];
        assert_eq!(eligible_pairs(&recent, NETWORK_MODEL_TIER), vec![(a, e), (b, e)]);
        assert_eq!(eligible_pairs(&recent, NETWORK_MODEL_TIER + 1), vec![(a, e)]);
        assert_eq!(eligible_pairs(&recent, 4), vec![(c, e)]);
    }

    /// Per-tier cohorts of a pipeline test network: identity == escrow key, `t*16 + i`.
    fn shard_id(tier: u8, i: u8) -> Hash {
        Hash::from_bytes([tier * 16 + i; 32])
    }

    /// Two producers per shard tier, none for `missing_tier`.
    fn shard_cohorts(missing_tier: Option<u8>) -> impl FnMut(u8) -> Vec<(Hash, Hash)> {
        move |tier| {
            if Some(tier) == missing_tier || tier <= NETWORK_MODEL_TIER {
                return Vec::new();
            }
            (0..2u8).map(|i| (shard_id(tier, i), shard_id(tier, i))).collect()
        }
    }

    fn drawn_links(ledger: &ServiceLedger, rh: &[u8; 32]) -> Vec<(u8, Hash)> {
        ledger.pending.get(rh).unwrap().audit.as_ref().unwrap().links.clone()
    }

    /// A complete pipeline response by `head` with the drawn link of every other tier, unless
    /// `substitute` replaces the drawn identity of that tier.
    fn pipeline_links(links: &[(u8, Hash)], substitute: Option<(u8, Hash)>) -> Vec<(u8, Hash)> {
        links
            .iter()
            .filter(|(t, _)| *t != NETWORK_MODEL_HEAD_TIER)
            .map(|(t, id)| match substitute {
                Some((st, sid)) if st == *t => (*t, sid),
                _ => (*t, *id),
            })
            .collect()
    }

    #[test]
    fn network_model_audit_draws_one_identity_per_shard_tier() {
        let mut ledger = ServiceLedger::default();
        let rh = [7u8; 32];
        ledger.on_chain_block(100, &[(rh, NETWORK_MODEL_TIER, 256)], &[], &[], |_| true, shard_cohorts(None));
        ledger.on_chain_block(101, &[], &[], &[], |_| true, shard_cohorts(None));
        let audit = ledger.pending.get(&rh).unwrap().audit.as_ref().unwrap();
        let tiers: Vec<u8> = audit.links.iter().map(|(t, _)| *t).collect();
        assert_eq!(tiers, network_model_shard_tiers().collect::<Vec<_>>());
        for (t, id) in audit.links.iter() {
            assert!(audit.members.binary_search(&(*id, *t)).is_ok());
        }
        assert_eq!(audit.members.len(), 2 * NETWORK_MODEL_SHARDS.len());
        assert_eq!(audit.cohort.len(), 2 * NETWORK_MODEL_SHARDS.len());
        assert_eq!(audit.window_end_daa, 101 + service_window_daa(NETWORK_MODEL_TIER, 256));
        // deterministic: a second ledger draws the same identities
        let mut other = ServiceLedger::default();
        other.on_chain_block(100, &[(rh, NETWORK_MODEL_TIER, 256)], &[], &[], |_| true, shard_cohorts(None));
        other.on_chain_block(101, &[], &[], &[], |_| true, shard_cohorts(None));
        assert_eq!(drawn_links(&other, &rh), audit.links);
    }

    #[test]
    fn network_model_request_with_an_unproduced_shard_is_dropped_without_strikes() {
        let mut ledger = ServiceLedger::default();
        let rh = [7u8; 32];
        let w = service_window_daa(NETWORK_MODEL_TIER, 256);
        ledger.on_chain_block(100, &[(rh, NETWORK_MODEL_TIER, 256)], &[], &[], |_| true, shard_cohorts(Some(NETWORK_MODEL_TIER + 3)));
        let out = ledger.on_chain_block(101, &[], &[], &[], |_| true, shard_cohorts(Some(NETWORK_MODEL_TIER + 3)));
        assert!(out.misses.is_empty());
        assert_eq!(ledger.pending_len(), 0);
        let out = ledger.on_chain_block(102 + w, &[], &[], &[], |_| true, shard_cohorts(Some(NETWORK_MODEL_TIER + 3)));
        assert!(out.misses.is_empty());
    }

    #[test]
    fn network_model_silence_strikes_only_the_drawn_head() {
        let mut ledger = ServiceLedger::default();
        ledger.set_window_v2_activation(0);
        let rh = [7u8; 32];
        let w = service_window_daa_at(NETWORK_MODEL_TIER, 256, true);
        ledger.on_chain_block(100, &[(rh, NETWORK_MODEL_TIER, 256)], &[], &[], |_| true, shard_cohorts(None));
        ledger.on_chain_block(101, &[], &[], &[], |_| true, shard_cohorts(None));
        let links = drawn_links(&ledger, &rh);
        let head = links.iter().find(|(t, _)| *t == NETWORK_MODEL_HEAD_TIER).unwrap().1;
        let out = ledger.on_chain_block(102 + w, &[], &[], &[], |_| true, shard_cohorts(None));
        assert_eq!(out.misses.len(), 1);
        assert_eq!(out.misses[0].miner, head);
        assert_eq!(out.misses[0].consecutive_misses, 1);
        assert_eq!(ledger.pending_len(), 0);
    }

    #[test]
    fn network_model_served_response_credits_every_signer_and_strikes_the_missing_link() {
        let mut ledger = ServiceLedger::default();
        ledger.set_window_v2_activation(0);
        ledger.set_reward_routing_activation(0);
        let rh = [7u8; 32];
        let w = service_window_daa_at(NETWORK_MODEL_TIER, 256, true);
        ledger.on_chain_block_with_rewards(100, &[(rh, NETWORK_MODEL_TIER, 256)], &[(rh, 104_000)], &[], &[], &[], &[], |_| true, shard_cohorts(None));
        ledger.on_chain_block(101, &[], &[], &[], |_| true, shard_cohorts(None));
        let links = drawn_links(&ledger, &rh);
        let head = links.iter().find(|(t, _)| *t == NETWORK_MODEL_HEAD_TIER).unwrap().1;
        // the drawn link of tier 8 is replaced by the other producer of that tier
        let sub_tier = NETWORK_MODEL_TIER + 3;
        let drawn_sub = links.iter().find(|(t, _)| *t == sub_tier).unwrap().1;
        let substitute = if drawn_sub == shard_id(sub_tier, 0) { shard_id(sub_tier, 1) } else { shard_id(sub_tier, 0) };
        let signed = pipeline_links(&links, Some((sub_tier, substitute)));
        // pre-arm strikes on the head so its reset is observable
        ledger.strikes.insert(head, StrikeEntry { count: 1, last_daa: 50 });
        let out = ledger.on_chain_block_with_rewards(
            105,
            &[],
            &[],
            &[(rh, Some(head))],
            &[],
            &[],
            &[(rh, head, signed.clone())],
            |_| true,
            shard_cohorts(None),
        );
        // reward split by VRAM: weights 8/12/12/16/24/32 of 104 000
        assert_eq!(out.rewards.len(), NETWORK_MODEL_SHARDS.len());
        assert_eq!(out.rewards[0].request_hash, rh);
        assert_eq!(out.rewards[0].winner, head);
        assert_eq!(out.rewards[0].amount, 32_000);
        let total: u64 = out.rewards.iter().map(|r| r.amount).sum();
        assert_eq!(total, 104_000);
        for (r, (tier, id)) in out.rewards[1..].iter().zip(signed.iter()) {
            assert_eq!(r.winner, *id);
            assert_eq!(r.request_hash, reward_share_key(&rh, *tier));
            assert_eq!(r.amount, 104_000 * network_model_tier_weight(*tier) / 104);
        }
        assert!(out.resets.iter().any(|(id, _)| *id == head));
        // window close: the substituted draw alone is struck, the substitute and the rest are clean
        let out = ledger.on_chain_block(102 + w, &[], &[], &[], |_| true, shard_cohorts(None));
        assert_eq!(out.misses.len(), 1);
        assert_eq!(out.misses[0].miner, drawn_sub);
        // a second identical response never pays twice
        assert_eq!(ledger.pending_len(), 0);
    }

    #[test]
    fn network_model_incomplete_or_foreign_responses_are_not_credited() {
        let mut ledger = ServiceLedger::default();
        ledger.set_window_v2_activation(0);
        ledger.set_reward_routing_activation(0);
        let rh = [7u8; 32];
        let w = service_window_daa_at(NETWORK_MODEL_TIER, 256, true);
        ledger.on_chain_block_with_rewards(100, &[(rh, NETWORK_MODEL_TIER, 256)], &[(rh, 104_000)], &[], &[], &[], &[], |_| true, shard_cohorts(None));
        ledger.on_chain_block(101, &[], &[], &[], |_| true, shard_cohorts(None));
        let links = drawn_links(&ledger, &rh);
        let head = links.iter().find(|(t, _)| *t == NETWORK_MODEL_HEAD_TIER).unwrap().1;
        let full = pipeline_links(&links, None);
        // no links at all (a v2-style answer to a pipeline request)
        let out = ledger.on_chain_block_with_rewards(103, &[], &[], &[(rh, Some(head))], &[], &[], &[], |_| true, shard_cohorts(None));
        assert!(out.rewards.is_empty());
        // one tier missing
        let mut short = full.clone();
        short.pop();
        let out = ledger.on_chain_block_with_rewards(104, &[], &[], &[(rh, Some(head))], &[], &[], &[(rh, head, short)], |_| true, shard_cohorts(None));
        assert!(out.rewards.is_empty());
        // a link signed by an identity that does not hold that tier
        let mut wrong = full.clone();
        wrong[0].1 = shard_id(NETWORK_MODEL_TIER + 5, 0);
        let out = ledger.on_chain_block_with_rewards(105, &[], &[], &[(rh, Some(head))], &[], &[], &[(rh, head, wrong)], |_| true, shard_cohorts(None));
        assert!(out.rewards.is_empty());
        // a head that does not hold the head tier
        let fake_head = shard_id(NETWORK_MODEL_TIER + 1, 0);
        let out = ledger.on_chain_block_with_rewards(106, &[], &[], &[(rh, Some(fake_head))], &[], &[], &[(rh, fake_head, full.clone())], |_| true, shard_cohorts(None));
        assert!(out.rewards.is_empty());
        // nothing was credited: the drawn head is struck alone at window close
        let out = ledger.on_chain_block(102 + w, &[], &[], &[], |_| true, shard_cohorts(None));
        assert_eq!(out.misses.len(), 1);
        assert_eq!(out.misses[0].miner, head);
    }

    #[test]
    fn network_model_substitute_head_is_paid_and_the_drawn_head_struck() {
        let mut ledger = ServiceLedger::default();
        ledger.set_window_v2_activation(0);
        ledger.set_reward_routing_activation(0);
        let rh = [7u8; 32];
        let w = service_window_daa_at(NETWORK_MODEL_TIER, 256, true);
        ledger.on_chain_block_with_rewards(100, &[(rh, NETWORK_MODEL_TIER, 256)], &[(rh, 1_040)], &[], &[], &[], &[], |_| true, shard_cohorts(None));
        ledger.on_chain_block(101, &[], &[], &[], |_| true, shard_cohorts(None));
        let links = drawn_links(&ledger, &rh);
        let drawn_head = links.iter().find(|(t, _)| *t == NETWORK_MODEL_HEAD_TIER).unwrap().1;
        let other_head = if drawn_head == shard_id(NETWORK_MODEL_HEAD_TIER, 0) { shard_id(NETWORK_MODEL_HEAD_TIER, 1) } else { shard_id(NETWORK_MODEL_HEAD_TIER, 0) };
        let full = pipeline_links(&links, None);
        let out = ledger.on_chain_block_with_rewards(105, &[], &[], &[(rh, Some(other_head))], &[], &[], &[(rh, other_head, full)], |_| true, shard_cohorts(None));
        assert_eq!(out.rewards[0].winner, other_head);
        assert_eq!(out.rewards[0].amount, 320);
        let out = ledger.on_chain_block(102 + w, &[], &[], &[], |_| true, shard_cohorts(None));
        assert_eq!(out.misses.len(), 1);
        assert_eq!(out.misses[0].miner, drawn_head);
    }

    #[test]
    fn pipeline_assignments_list_the_drawn_escrow_keys_per_tier() {
        let mut ledger = ServiceLedger::default();
        let rh = [7u8; 32];
        ledger.on_chain_block(100, &[(rh, NETWORK_MODEL_TIER, 256), ([8u8; 32], 0, 64)], &[], &[], |_| true, |tier| {
            if tier == 0 { vec![(Hash::from_bytes([1u8; 32]), Hash::from_bytes([1u8; 32]))] } else { shard_cohorts(None)(tier) }
        });
        assert!(ledger.pipeline_assignments().is_empty());
        ledger.on_chain_block(101, &[], &[], &[], |_| true, |tier| {
            if tier == 0 { vec![(Hash::from_bytes([1u8; 32]), Hash::from_bytes([1u8; 32]))] } else { shard_cohorts(None)(tier) }
        });
        let a = ledger.pipeline_assignments();
        assert_eq!(a.len(), 1);
        assert_eq!(a[0].request_hash, rh);
        assert_eq!(a[0].accepted_daa, 100);
        let drawn = drawn_links(&ledger, &rh);
        let expected: Vec<(u8, [u8; 32])> = drawn.iter().map(|(t, id)| (*t, id.as_bytes())).collect();
        assert_eq!(a[0].links, expected);
        let wire = a[0].encode();
        assert_eq!(PipelineAssignment::decode(&wire).unwrap(), a[0]);
        assert!(PipelineAssignment::decode("zz:1:2:").is_none());
    }

    #[test]
    fn pipeline_reward_split_keeps_the_remainder_on_the_head() {
        let head = (NETWORK_MODEL_HEAD_TIER, shard_id(NETWORK_MODEL_HEAD_TIER, 0));
        let links: Vec<(u8, Hash)> =
            network_model_shard_tiers().filter(|t| *t != NETWORK_MODEL_HEAD_TIER).map(|t| (t, shard_id(t, 0))).collect();
        let shares = split_pipeline_reward(&NETWORK_MODEL_MAINNET, 1_000_003, head, &links);
        assert_eq!(shares.len(), NETWORK_MODEL_SHARDS.len());
        assert_eq!(shares.iter().map(|(_, a)| *a).sum::<u64>(), 1_000_003);
        assert_eq!(shares[0].0, head.1);
        assert!(shares[0].1 > shares[1].1);
        // no weights at all: everything to the head
        assert_eq!(split_pipeline_reward(&NETWORK_MODEL_MAINNET, 5, (4, head.1), &[]), vec![(head.1, 5)]);
    }

    #[test]
    fn snapshot_roundtrips_a_pipeline_audit_and_keeps_lineup_bytes() {
        let mut ledger = ServiceLedger::default();
        let rh = [7u8; 32];
        let lineup = [8u8; 32];
        let a = Hash::from_bytes([1u8; 32]);
        ledger.on_chain_block(100, &[(rh, NETWORK_MODEL_TIER, 256), (lineup, 0, 64)], &[], &[], |_| true, |tier| {
            if tier == 0 { vec![(a, a)] } else { shard_cohorts(None)(tier) }
        });
        ledger.on_chain_block(101, &[], &[], &[], |_| true, |tier| if tier == 0 { vec![(a, a)] } else { shard_cohorts(None)(tier) });
        let snap = ledger.snapshot();
        let bytes = snap.to_bytes();
        let decoded = ServiceLedgerSnapshot::from_bytes(&bytes).unwrap();
        assert_eq!(decoded, snap);
        let mut restored = ServiceLedger::default();
        restored.restore_snapshot(&decoded);
        assert_eq!(drawn_links(&restored, &rh), drawn_links(&ledger, &rh));
        assert!(restored.pending.get(&lineup).unwrap().audit.as_ref().unwrap().links.is_empty());
        // a lineup-only ledger encodes exactly as before: flag 1, no link section
        let mut old = ServiceLedger::default();
        old.on_chain_block(100, &[(lineup, 0, 64)], &[], &[], |_| true, cohort_of(&[a]));
        old.on_chain_block(101, &[], &[], &[], |_| true, cohort_of(&[a]));
        let old_bytes = old.snapshot().to_bytes();
        let mut expected = Vec::new();
        expected.push(1u8);
        expected.extend_from_slice(&0u32.to_le_bytes()); // vault
        expected.extend_from_slice(&1u32.to_le_bytes()); // pending
        expected.extend_from_slice(&lineup);
        expected.push(0); // tier
        expected.extend_from_slice(&64u32.to_le_bytes());
        expected.extend_from_slice(&100u64.to_le_bytes());
        expected.extend_from_slice(&0u64.to_le_bytes()); // reward
        expected.push(0); // winner
        expected.push(1); // audit flag: lineup
        assert_eq!(&old_bytes[..expected.len()], expected.as_slice());
    }

    #[test]
    fn network_model_window_is_wider_and_slower() {
        use super::{
            service_window_daa_at, SERVICE_WINDOW_BASE_DAA_NETWORK_MODEL, SERVICE_WINDOW_BASE_DAA_V2,
            SERVICE_WINDOW_NETWORK_MODEL_PER_TOKEN_DAA,
        };
        use crate::config::params::NETWORK_MODEL_TIER;
        assert_eq!(service_window_daa_at(4, 256, true), SERVICE_WINDOW_BASE_DAA_V2 + 256 * 4);
        assert_eq!(
            service_window_daa_at(NETWORK_MODEL_TIER, 256, true),
            SERVICE_WINDOW_BASE_DAA_NETWORK_MODEL + 256 * SERVICE_WINDOW_NETWORK_MODEL_PER_TOKEN_DAA
        );
        assert_eq!(service_window_daa_at(NETWORK_MODEL_TIER + 2, 0, false), SERVICE_WINDOW_BASE_DAA_NETWORK_MODEL);
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
}
