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

/// Escrow CSV lock at/after the service-bond gate: ledger horizon (72 000) + finality depth
/// (432 000), ≈ 14 h at 10 BPS. A claim created at C is burnable by misses up to C + horizon,
/// enforceable at most finality later — this lock guarantees the burn is always in force before
/// the claim unlocks.
pub const SERVICE_BOND_CSV_WINDOW_BLOCKS: u64 = 504_000;

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

/// Service-ledger identity of a miner: its announced escrow pubkey, verbatim. The same key
/// signs V2 AiResponses, receives the CSV escrow outputs, and takes the service penalties —
/// one identity for eligibility, authentication and slashing.
pub fn escrow_miner_key(pubkey: &[u8; 32]) -> Hash {
    Hash::from_bytes(*pubkey)
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
/// (like escrow burns): a miner suspended at miss daa `T` has his blocks rejected over
/// `[T + finality, T + finality + SERVICE_SUSPENSION_DAA]`, so the full 24 h bites after the
/// reorg-immune finality delay.
pub const SERVICE_SUSPENSION_DAA: u64 = 864_000;

/// DAA window, ending at the assignment seed block, in which a miner must have produced a proven
/// tier block to be service-eligible. ~10 minutes at 10 BPS.
pub const SERVICE_ELIGIBILITY_WINDOW_DAA: u64 = 6_000;

/// Fixed part of the service window: assignment detection, propagation and inclusion. 30 s.
pub const SERVICE_WINDOW_BASE_DAA: u64 = 300;

/// Hard cap on an AiRequest's `max_tokens` at/after the service-bond gate — matches the web
/// interface maximum. Bounds the service window any single request can demand and rejects
/// nonsense values.
pub const AI_REQUEST_MAX_TOKENS_CAP: u32 = 4_096;

/// DAA window an assigned miner has, from his assignment seed block, for the request to be served
/// before it counts as a miss: a fixed base plus a per-requested-token allowance floored at the
/// generation speed of the tier's model class (measured medians ~7-10 tok/s, ×2 margin).
pub fn service_window_daa(tier: u8, max_tokens: u32) -> u64 {
    let per_token_daa: u64 = match tier {
        0..=2 => 2, // 0.2 s/token — 5 tok/s floor
        3 => 3,     // 0.3 s/token
        _ => 4,     // 0.4 s/token — 2.5 tok/s floor
    };
    SERVICE_WINDOW_BASE_DAA + max_tokens.min(AI_REQUEST_MAX_TOKENS_CAP) as u64 * per_token_daa
}

/// DAA horizon beyond which service-ledger state is forgotten: pending requests expire and strike
/// entries read as zero. Folding the chain from an empty ledger over this horizon reproduces the
/// exact state, so the ledger is RAM-only and IBD-safe. Set to twice the strike interval so a
/// strike entry survives one inter-strike gap and the count can climb to suspension; the cold-start
/// refold spans twice this horizon and therefore recomputes every live strike.
pub const SERVICE_LEDGER_HORIZON_DAA: u64 = 72_000;

/// Penalty applied to a miner for a missed service assignment, by consecutive-miss count.
/// A successful serve resets the count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ServicePenalty {
    None,
    /// Burn `n` escrow claims (n blocks' worth of the miner's accumulated escrow).
    BurnClaims(u32),
    /// Burn the miner's entire still-locked pending escrow.
    SlashAllPending,
    /// Suspend the miner's block production: his blocks are rejected for [`SERVICE_SUSPENSION_DAA`]
    /// once the suspension is finality-deep. Also drains any escrow re-accumulated past the drain.
    Suspend,
}

/// Penalty for the `consecutive_misses`-th consecutive miss (0 = served, no penalty).
pub fn strike_penalty(consecutive_misses: u32) -> ServicePenalty {
    match consecutive_misses {
        0 => ServicePenalty::None,
        1 => ServicePenalty::BurnClaims(STRIKE_1_BURN_CLAIMS),
        2 => ServicePenalty::SlashAllPending,
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
/// miners who produced at least one `target_tier` block in the recent window. Sorted and deduped
/// so every node derives the same ordering before `assign_index` picks an index into it.
/// `recent` is `(miner_key, tier)` for the recently-active window (order irrelevant).
pub fn eligible_miners(recent: &[(Hash, u8)], target_tier: u8) -> Vec<Hash> {
    let mut set: Vec<Hash> = recent.iter().filter(|(_, t)| *t == target_tier).map(|(m, _)| *m).collect();
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

#[derive(Clone, Debug)]
struct PendingRequest {
    tier: u8,
    max_tokens: u32,
    accepted_daa: u64,
    audit: Option<Audit>,
}

/// One cohort audit: every declared miner of the request's tier must respond before the window
/// closes; the silent ones are struck when it does.
#[derive(Clone, Debug)]
struct Audit {
    cohort: Vec<Hash>,
    responded: Vec<Hash>,
    window_end_daa: u64,
}

#[derive(Clone, Copy, Debug)]
struct StrikeEntry {
    count: u32,
    last_daa: u64,
}

/// RAM-only request-lifecycle ledger, folded once per selected-chain block. Deterministic: state
/// is a pure function of the accepted requests/responses stream and the cohort function, with
/// BTreeMap ordering; any node folding the last [`SERVICE_LEDGER_HORIZON_DAA`] of chain from an
/// empty ledger reaches the identical state. Never persisted.
#[derive(Clone, Debug, Default)]
pub struct ServiceLedger {
    pending: std::collections::BTreeMap<[u8; 32], PendingRequest>,
    strikes: std::collections::BTreeMap<Hash, StrikeEntry>,
    /// Per-miner still-locked escrow claims, chain order (newest at the back).
    vault: std::collections::BTreeMap<Hash, std::collections::VecDeque<EscrowClaim>>,
}

impl ServiceLedger {
    /// Folds one selected-chain block into the ledger and returns the misses it closes.
    ///
    /// `requests` are the block's accepted AiRequests as `(request_hash, tier, max_tokens)`;
    /// `responses` its accepted AiResponses as `(request_hash, verified responder)`; `escrows` the
    /// escrow claims this block's coinbase creates, keyed by producing miner; `cohort` resolves a
    /// tier to its full declared-miner set at this block. Every cohort member must respond before
    /// the request's window closes; responses are applied before expiries, so one landing in the
    /// closing block still counts.
    pub fn on_chain_block(
        &mut self,
        daa: u64,
        requests: &[([u8; 32], u8, u32)],
        responses: &[([u8; 32], Option<Hash>)],
        escrows: &[(Hash, EscrowClaim)],
        mut cohort: impl FnMut(u8) -> Vec<Hash>,
    ) -> Vec<ServiceMiss> {
        for (miner, claim) in escrows {
            self.vault.entry(*miner).or_default().push_back(*claim);
        }
        for claims in self.vault.values_mut() {
            while claims.front().is_some_and(|c| c.daa + SERVICE_LEDGER_HORIZON_DAA <= daa) {
                claims.pop_front();
            }
        }
        self.vault.retain(|_, claims| !claims.is_empty());

        // An authenticated response from a cohort member marks him as having served this audit
        // and resets his streak. Anyone else's response is ignored by the ledger.
        for (rh, responder) in responses {
            let Some(req) = self.pending.get_mut(rh) else { continue };
            let Some(audit) = req.audit.as_mut() else { continue };
            if let Some(r) = responder {
                if audit.cohort.binary_search(r).is_ok() && !audit.responded.contains(r) {
                    audit.responded.push(*r);
                    self.strikes.remove(r);
                }
            }
        }

        self.pending.retain(|_, r| r.accepted_daa + SERVICE_LEDGER_HORIZON_DAA > daa);

        let mut misses = Vec::new();
        let hashes: Vec<[u8; 32]> = self.pending.keys().copied().collect();
        for rh in hashes {
            let req = self.pending.get(&rh).unwrap();
            match &req.audit {
                Some(a) if daa > a.window_end_daa => {
                    let audit = a.clone();
                    for miner in audit.cohort.iter().filter(|m| !audit.responded.contains(m)) {
                        // Rate-limit: a strike lands at most once per interval. A miss inside the
                        // interval of the miner's last strike is a no-op — no escalation, no burn.
                        if self.strikes.get(miner).is_some_and(|e| daa < e.last_daa + SERVICE_STRIKE_INTERVAL_DAA) {
                            continue;
                        }
                        let count = self.consecutive_misses(miner, daa) + 1;
                        self.strikes.insert(*miner, StrikeEntry { count, last_daa: daa });
                        let penalty = strike_penalty(count);
                        let burned = self.burn(miner, penalty);
                        misses.push(ServiceMiss { request_hash: rh, miner: *miner, consecutive_misses: count, penalty, burned });
                    }
                    self.pending.remove(&rh);
                }
                Some(_) => {}
                None if daa > req.accepted_daa => {
                    let set = cohort(req.tier);
                    if set.is_empty() {
                        self.pending.remove(&rh);
                    } else {
                        let window = service_window_daa(req.tier, req.max_tokens);
                        self.pending.get_mut(&rh).unwrap().audit =
                            Some(Audit { cohort: set, responded: Vec::new(), window_end_daa: daa + window });
                    }
                }
                None => {}
            }
        }

        for (rh, tier, max_tokens) in requests {
            // A re-accepted hash (identical payload resubmitted) must not reset the running
            // audit: overwriting would re-arm the window and push the miss out forever.
            self.pending
                .entry(*rh)
                .or_insert(PendingRequest { tier: *tier, max_tokens: *max_tokens, accepted_daa: daa, audit: None });
        }

        misses
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

    /// The miner's consecutive-miss count as of `daa`; entries older than the ledger horizon read
    /// as zero.
    pub fn consecutive_misses(&self, miner: &Hash, daa: u64) -> u32 {
        match self.strikes.get(miner) {
            Some(e) if e.last_daa + SERVICE_LEDGER_HORIZON_DAA > daa => e.count,
            _ => 0,
        }
    }

    /// Currently pending (accepted, unserved, unexpired) request count.
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        eligible_miners, service_window_daa, strike_penalty, update_strikes, ServiceLedger, ServicePenalty,
        AI_REQUEST_MAX_TOKENS_CAP, SERVICE_LEDGER_HORIZON_DAA, SERVICE_STRIKE_INTERVAL_DAA, STRIKE_1_BURN_CLAIMS,
    };
    use keryx_hashes::Hash;

    fn cohort_of(set: &[Hash]) -> impl FnMut(u8) -> Vec<Hash> + '_ {
        move |_tier| set.to_vec()
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
        assert!(ledger.on_chain_block(100, &[(rh, 0, 256)], &[], &[], cohort_of(&set)).is_empty());
        // response in the audit-opening block is applied before the cohort exists: ignored
        assert!(ledger.on_chain_block(101, &[], &[(rh, Some(a))], &[], cohort_of(&set)).is_empty());
        // a v1 (unsigned) response and a non-member response never count
        assert!(ledger.on_chain_block(102, &[], &[(rh, None), (rh, Some(b))], &[], cohort_of(&set)).is_empty());
        assert_eq!(ledger.pending_len(), 1);
        // the member's signed response does; the audit closes clean at its window end
        assert!(ledger.on_chain_block(103, &[], &[(rh, Some(a))], &[], cohort_of(&set)).is_empty());
        assert!(ledger.on_chain_block(102 + w, &[], &[], &[], cohort_of(&set)).is_empty());
        assert_eq!(ledger.pending_len(), 0);
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
        ledger.on_chain_block(100, &[(r1, 0, 256)], &[], &[], cohort_of(&set));
        ledger.on_chain_block(101, &[], &[], &[], cohort_of(&set));
        let misses = ledger.on_chain_block(102 + w, &[], &[], &[], cohort_of(&set));
        assert_eq!(misses.len(), 1); // strike 1

        // a second miss well within the interval: no strike, no escalation
        let d2 = 200 + w;
        let r2 = [2u8; 32];
        ledger.on_chain_block(d2, &[(r2, 0, 256)], &[], &[], cohort_of(&set));
        ledger.on_chain_block(d2 + 1, &[], &[], &[], cohort_of(&set));
        let misses = ledger.on_chain_block(d2 + 2 + w, &[], &[], &[], cohort_of(&set));
        assert!(misses.is_empty(), "a miss inside the interval must not strike");
        assert_eq!(ledger.consecutive_misses(&a, d2 + 2 + w), 1);
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
        ledger.on_chain_block(100, &[(r1, 0, 256)], &[], &[], cohort_of(&set));
        ledger.on_chain_block(101, &[], &[], &[], cohort_of(&set));
        // the same hash lands again mid-window: the running audit must survive
        ledger.on_chain_block(102, &[(r1, 0, 256)], &[], &[], cohort_of(&set));
        let misses = ledger.on_chain_block(102 + w, &[], &[], &[], cohort_of(&set));
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
        ledger.on_chain_block(100, &[(r1, 0, 256)], &[], &[], cohort_of(&set));
        ledger.on_chain_block(101, &[], &[], &[], cohort_of(&set));
        let misses = ledger.on_chain_block(102 + w, &[], &[], &[], cohort_of(&set));
        assert_eq!(misses.len(), 2);
        assert_eq!((misses[0].miner, misses[0].consecutive_misses), (a, 1));
        assert_eq!(misses[0].penalty, ServicePenalty::BurnClaims(STRIKE_1_BURN_CLAIMS));

        // round 2, one interval later: a serves -> only b strikes, to strike 2
        let d2 = 200 + i;
        let r2 = [2u8; 32];
        ledger.on_chain_block(d2, &[(r2, 0, 256)], &[], &[], cohort_of(&set));
        ledger.on_chain_block(d2 + 1, &[], &[], &[], cohort_of(&set));
        ledger.on_chain_block(d2 + 2, &[], &[(r2, Some(a))], &[], cohort_of(&set));
        let misses = ledger.on_chain_block(d2 + 2 + w, &[], &[], &[], cohort_of(&set));
        assert_eq!(misses.len(), 1);
        assert_eq!((misses[0].miner, misses[0].consecutive_misses), (b, 2));
        assert_eq!(misses[0].penalty, ServicePenalty::SlashAllPending);
        assert_eq!(ledger.consecutive_misses(&a, d2 + 2 + w), 0);

        // round 3, another interval later: a (reset) restarts at 1, b reaches strike 3 -> Suspend
        let d3 = d2 + i + 100;
        let r3 = [3u8; 32];
        ledger.on_chain_block(d3, &[(r3, 0, 256)], &[], &[], cohort_of(&set));
        ledger.on_chain_block(d3 + 1, &[], &[], &[], cohort_of(&set));
        let misses = ledger.on_chain_block(d3 + 2 + w, &[], &[], &[], cohort_of(&set));
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
        ledger.on_chain_block(100, &[(rh, 0, 256)], &[], &[], cohort_of(&set));
        ledger.on_chain_block(101, &[], &[], &[], cohort_of(&set));
        // window is past, but the closing block itself carries the response: no miss
        assert!(ledger.on_chain_block(200 + w, &[], &[(rh, Some(a))], &[], cohort_of(&set)).is_empty());
        assert_eq!(ledger.pending_len(), 0);
    }

    #[test]
    fn empty_cohort_drops_the_request() {
        let mut ledger = ServiceLedger::default();
        let rh = [5u8; 32];
        assert!(ledger.on_chain_block(100, &[(rh, 0, 256)], &[], &[], |_tier| Vec::new()).is_empty());
        assert!(ledger.on_chain_block(101, &[], &[], &[], |_tier| Vec::new()).is_empty());
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
        ledger.on_chain_block(100, &[(r1, 0, 256)], &[], &escrows, cohort_of(&set));
        ledger.on_chain_block(101, &[], &[], &[], cohort_of(&set));
        let misses = ledger.on_chain_block(102 + w, &[], &[], &[], cohort_of(&set));
        assert_eq!(misses.len(), 1);
        assert_eq!(misses[0].penalty, ServicePenalty::BurnClaims(STRIKE_1_BURN_CLAIMS));
        assert_eq!(misses[0].burned.iter().map(|c| c.value).collect::<Vec<_>>(), vec![6, 5, 4, 3, 2]);

        // r2 (one interval later) missed too: strike 2 drains the leftover claim plus one
        // accumulated meanwhile
        let i = SERVICE_STRIKE_INTERVAL_DAA;
        let d2 = 200 + i;
        let r2 = [2u8; 32];
        let fresh = [(a, claim(7, d2))];
        ledger.on_chain_block(d2, &[(r2, 0, 256)], &[], &fresh, cohort_of(&set));
        ledger.on_chain_block(d2 + 1, &[], &[], &[], cohort_of(&set));
        let misses = ledger.on_chain_block(d2 + 2 + w, &[], &[], &[], cohort_of(&set));
        assert_eq!(misses.len(), 1);
        assert_eq!(misses[0].penalty, ServicePenalty::SlashAllPending);
        assert_eq!(misses[0].burned.iter().map(|c| c.value).collect::<Vec<_>>(), vec![7, 1]);

        // r3 (another interval later): strike 3 (Suspend) takes claims re-accumulated past the drain
        let d3 = d2 + i + 100;
        let r3 = [3u8; 32];
        let fresh = [(a, claim(8, d3))];
        ledger.on_chain_block(d3, &[(r3, 0, 256)], &[], &fresh, cohort_of(&set));
        ledger.on_chain_block(d3 + 1, &[], &[], &[], cohort_of(&set));
        let misses = ledger.on_chain_block(d3 + 2 + w, &[], &[], &[], cohort_of(&set));
        assert_eq!(misses.len(), 1);
        assert_eq!(misses[0].penalty, ServicePenalty::Suspend);
        assert_eq!(misses[0].burned.iter().map(|c| c.value).collect::<Vec<_>>(), vec![8]);
    }

    #[test]
    fn horizon_expires_pendings_and_strikes() {
        let a = Hash::from_bytes([1u8; 32]);
        let set = [a];
        let mut ledger = ServiceLedger::default();
        let w = service_window_daa(0, 256);

        let rh = [9u8; 32];
        ledger.on_chain_block(100, &[(rh, 0, 256)], &[], &[], cohort_of(&set));
        ledger.on_chain_block(101, &[], &[], &[], cohort_of(&set));
        let misses = ledger.on_chain_block(102 + w, &[], &[], &[], cohort_of(&set));
        assert_eq!(misses.len(), 1);
        assert_eq!(ledger.consecutive_misses(&a, 102 + w), 1);

        // beyond the horizon the strike reads zero and nothing lingers
        let far = 102 + w + SERVICE_LEDGER_HORIZON_DAA;
        assert_eq!(ledger.consecutive_misses(&a, far), 0);
        ledger.on_chain_block(far, &[], &[], &[], cohort_of(&set));
        assert_eq!(ledger.pending_len(), 0);
    }

    #[test]
    fn eligible_miners_distinct_sorted_by_tier() {
        let a = Hash::from_bytes([1u8; 32]);
        let b = Hash::from_bytes([2u8; 32]);
        let c = Hash::from_bytes([3u8; 32]);
        // a: tier 0 (twice), b: tier 0, c: tier 1
        let recent = [(a, 0u8), (c, 1u8), (a, 0u8), (b, 0u8)];
        assert_eq!(eligible_miners(&recent, 0), vec![a, b]);
        assert_eq!(eligible_miners(&recent, 1), vec![c]);
        assert!(eligible_miners(&recent, 4).is_empty());
    }

    #[test]
    fn strike_penalty_escalation() {
        assert_eq!(strike_penalty(0), ServicePenalty::None);
        assert_eq!(strike_penalty(1), ServicePenalty::BurnClaims(STRIKE_1_BURN_CLAIMS));
        assert_eq!(strike_penalty(2), ServicePenalty::SlashAllPending);
        assert_eq!(strike_penalty(3), ServicePenalty::Suspend);
        assert_eq!(strike_penalty(9), ServicePenalty::Suspend);
    }

    #[test]
    fn strikes_reset_on_serve() {
        // miss, miss (→ ban territory) then serve resets, then a fresh miss is only strike 1
        let mut c = 0;
        for missed in [true, true, false, true] {
            c = update_strikes(c, missed);
        }
        assert_eq!(c, 1);
        assert_eq!(strike_penalty(c), ServicePenalty::BurnClaims(STRIKE_1_BURN_CLAIMS));
        // a long honest run of serves keeps it at 0
        for _ in 0..1000 {
            c = update_strikes(c, false);
        }
        assert_eq!(c, 0);
    }

}
