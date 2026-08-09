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

/// Escrow CSV lock at/after the service-bond gate: ledger horizon (36 000) + finality depth
/// (432 000), ≈ 13 h at 10 BPS. A claim created at C is burnable by misses up to C + horizon,
/// enforceable at most finality later — this lock guarantees the burn is always in force before
/// the claim unlocks.
pub const SERVICE_BOND_CSV_WINDOW_BLOCKS: u64 = 468_000;

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

/// DAA window, ending at the assignment seed block, in which a miner must have produced a proven
/// tier block to be service-eligible. ~10 minutes at 10 BPS.
pub const SERVICE_ELIGIBILITY_WINDOW_DAA: u64 = 6_000;

/// DAA window an assigned miner has, from his assignment seed block, for the request to be served
/// before it counts as a miss. Covers propagation plus one inference on the tier's model.
pub fn service_window_daa(tier: u8) -> u64 {
    match tier {
        0 => 1_200,
        1 => 1_800,
        2 => 2_400,
        3 => 3_600,
        _ => 6_000,
    }
}

/// DAA horizon beyond which service-ledger state is forgotten: pending requests expire and strike
/// entries read as zero. Folding the chain from an empty ledger over this horizon reproduces the
/// exact state, so the ledger is RAM-only and IBD-safe. Matches the escrow CSV lock (~1 h) —
/// a penalty must land while the escrow is still locked.
pub const SERVICE_LEDGER_HORIZON_DAA: u64 = 36_000;

/// Penalty applied to a miner for a missed service assignment, by consecutive-miss count.
/// A successful serve resets the count; the ban is a P2P policy, not a consensus slash.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ServicePenalty {
    None,
    /// Burn `n` escrow claims (n blocks' worth of the miner's accumulated escrow).
    BurnClaims(u32),
    /// Burn the miner's entire still-locked pending escrow.
    SlashAllPending,
    /// Ban the miner's IP (enforced at the P2P layer, not consensus).
    BanIp,
}

/// Penalty for the `consecutive_misses`-th consecutive miss (0 = served, no penalty).
pub fn strike_penalty(consecutive_misses: u32) -> ServicePenalty {
    match consecutive_misses {
        0 => ServicePenalty::None,
        1 => ServicePenalty::BurnClaims(STRIKE_1_BURN_CLAIMS),
        2 => ServicePenalty::SlashAllPending,
        _ => ServicePenalty::BanIp,
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
    pub penalty: ServicePenalty,
    pub burned: Vec<EscrowClaim>,
}

#[derive(Clone, Debug)]
struct PendingRequest {
    tier: u8,
    accepted_daa: u64,
    assignment: Option<Assignment>,
    /// Miners that already missed this request, excluded from its re-draws.
    excluded: Vec<Hash>,
}

#[derive(Clone, Copy, Debug)]
struct Assignment {
    miner: Hash,
    window_end_daa: u64,
}

#[derive(Clone, Copy, Debug)]
struct StrikeEntry {
    count: u32,
    last_daa: u64,
}

/// RAM-only request-lifecycle ledger, folded once per selected-chain block. Deterministic: state
/// is a pure function of the accepted requests/responses stream and the draw function, with
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
    /// `requests` are the block's accepted AiRequests as `(request_hash, tier)`; `responses` the
    /// request hashes its accepted AiResponses answer; `escrows` the escrow claims this block's
    /// coinbase creates, keyed by producing miner; `draw` resolves `(tier, excluded)` to the
    /// responsible miner using this block as the assignment seed. Responses are applied before
    /// window closes, so a response landing in the closing block still cancels the miss.
    pub fn on_chain_block(
        &mut self,
        daa: u64,
        requests: &[([u8; 32], u8)],
        responses: &[([u8; 32], Option<Hash>)],
        escrows: &[(Hash, EscrowClaim)],
        mut draw: impl FnMut(u8, &[Hash]) -> Option<Hash>,
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

        // Only the currently assigned miner's authenticated response serves the request; anyone
        // else's is ignored — the assignment is an exclusive audit of one miner. Responses are
        // applied before misses, so one landing in the closing block still cancels.
        for (rh, responder) in responses {
            let served = self
                .pending
                .get(rh)
                .is_some_and(|req| req.assignment.is_some_and(|a| *responder == Some(a.miner)));
            if served {
                let req = self.pending.remove(rh).unwrap();
                self.strikes.remove(&req.assignment.unwrap().miner);
            }
        }

        self.pending.retain(|_, r| r.accepted_daa + SERVICE_LEDGER_HORIZON_DAA > daa);

        let mut misses = Vec::new();
        let hashes: Vec<[u8; 32]> = self.pending.keys().copied().collect();
        for rh in hashes {
            let req = self.pending.get(&rh).unwrap();
            match req.assignment {
                Some(a) if daa > a.window_end_daa => {
                    let count = self.consecutive_misses(&a.miner, daa) + 1;
                    self.strikes.insert(a.miner, StrikeEntry { count, last_daa: daa });
                    let burned = self.burn(&a.miner, strike_penalty(count));
                    misses.push(ServiceMiss {
                        request_hash: rh,
                        miner: a.miner,
                        consecutive_misses: count,
                        penalty: strike_penalty(count),
                        burned,
                    });
                    let req = self.pending.get_mut(&rh).unwrap();
                    req.excluded.push(a.miner);
                    let window = service_window_daa(req.tier);
                    let assignment =
                        draw(req.tier, &req.excluded).map(|miner| Assignment { miner, window_end_daa: daa + window });
                    self.pending.get_mut(&rh).unwrap().assignment = assignment;
                }
                Some(_) => {}
                None if daa > req.accepted_daa => {
                    let window = service_window_daa(req.tier);
                    let assignment =
                        draw(req.tier, &req.excluded).map(|miner| Assignment { miner, window_end_daa: daa + window });
                    self.pending.get_mut(&rh).unwrap().assignment = assignment;
                }
                None => {}
            }
        }

        for (rh, tier) in requests {
            self.pending
                .insert(*rh, PendingRequest { tier: *tier, accepted_daa: daa, assignment: None, excluded: Vec::new() });
        }

        misses
    }

    /// Takes the escrow claims a penalty burns out of the miner's vault: the `n` newest for
    /// `BurnClaims(n)`, everything still locked for `SlashAllPending` — and for `BanIp` too, so
    /// claims re-accumulated past the second strike stay burnable while the streak lasts.
    fn burn(&mut self, miner: &Hash, penalty: ServicePenalty) -> Vec<EscrowClaim> {
        let Some(claims) = self.vault.get_mut(miner) else {
            return Vec::new();
        };
        let take = match penalty {
            ServicePenalty::None => 0,
            ServicePenalty::BurnClaims(n) => (n as usize).min(claims.len()),
            ServicePenalty::SlashAllPending | ServicePenalty::BanIp => claims.len(),
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
        assign_index, draw_assignment, eligible_miners, service_window_daa, strike_penalty, update_strikes,
        ServiceLedger, ServicePenalty, SERVICE_LEDGER_HORIZON_DAA, STRIKE_1_BURN_CLAIMS,
    };
    use keryx_hashes::Hash;

    fn no_draw(_tier: u8, _excluded: &[Hash]) -> Option<Hash> {
        panic!("draw must not be called")
    }

    #[test]
    fn draw_assignment_excludes_then_falls_back() {
        let a = Hash::from_bytes([1u8; 32]);
        let b = Hash::from_bytes([2u8; 32]);
        let seed = [0u8; 32];
        // seed 0 → index 0 of the pool
        assert_eq!(draw_assignment(&[a, b], &[], &seed), Some(a));
        assert_eq!(draw_assignment(&[a, b], &[a], &seed), Some(b));
        // exclusion emptying the set falls back to the full set
        assert_eq!(draw_assignment(&[a, b], &[a, b], &seed), Some(a));
        assert_eq!(draw_assignment(&[], &[], &seed), None);
    }

    #[test]
    fn only_the_assigned_responder_serves() {
        let a = Hash::from_bytes([1u8; 32]);
        let b = Hash::from_bytes([2u8; 32]);
        let eligible = [a];
        let mut ledger = ServiceLedger::default();
        let draw = |_tier: u8, excluded: &[Hash]| draw_assignment(&eligible, excluded, &[0u8; 32]);

        let rh = [7u8; 32];
        assert!(ledger.on_chain_block(100, &[(rh, 0)], &[], &[], draw).is_empty());
        // a volunteer response before any assignment serves nothing
        assert!(ledger.on_chain_block(101, &[], &[(rh, Some(b))], &[], draw).is_empty());
        assert_eq!(ledger.pending_len(), 1);
        // now assigned to a; neither an unsigned (v1) response nor b's serves it
        assert!(ledger.on_chain_block(102, &[], &[(rh, None), (rh, Some(b))], &[], draw).is_empty());
        assert_eq!(ledger.pending_len(), 1);
        // the assignee's own signed response does
        assert!(ledger.on_chain_block(103, &[], &[(rh, Some(a))], &[], draw).is_empty());
        assert_eq!(ledger.pending_len(), 0);
    }

    #[test]
    fn miss_escalation_cascade_and_serve_reset() {
        let a = Hash::from_bytes([1u8; 32]);
        let b = Hash::from_bytes([2u8; 32]);
        let eligible = [a, b];
        let mut ledger = ServiceLedger::default();
        let w = service_window_daa(0);
        let draw = |_tier: u8, excluded: &[Hash]| draw_assignment(&eligible, excluded, &[0u8; 32]);

        // r1 accepted at 100, assigned to a at 101, window closes after 101 + w
        let r1 = [1u8; 32];
        assert!(ledger.on_chain_block(100, &[(r1, 0)], &[], &[], draw).is_empty());
        assert!(ledger.on_chain_block(101, &[], &[], &[], draw).is_empty());
        let misses = ledger.on_chain_block(102 + w, &[], &[], &[], draw);
        assert_eq!(misses.len(), 1);
        assert_eq!(misses[0].miner, a);
        assert_eq!(misses[0].consecutive_misses, 1);
        assert_eq!(misses[0].penalty, ServicePenalty::BurnClaims(STRIKE_1_BURN_CLAIMS));

        // cascade: r1 re-drawn to b (a excluded); b misses too → his own strike 1
        let misses = ledger.on_chain_block(103 + 2 * w, &[], &[], &[], draw);
        assert_eq!(misses.len(), 1);
        assert_eq!(misses[0].miner, b);
        assert_eq!(misses[0].consecutive_misses, 1);

        // exclusion now empties the set → fallback re-draws a; a misses r1 again → strike 2
        let misses = ledger.on_chain_block(104 + 3 * w, &[], &[], &[], draw);
        assert_eq!(misses.len(), 1);
        assert_eq!(misses[0].miner, a);
        assert_eq!(misses[0].consecutive_misses, 2);
        assert_eq!(misses[0].penalty, ServicePenalty::SlashAllPending);

        // a serves his next assignment → his counter resets; a fresh miss is strike 1 again.
        // r1 is finally served here too — an unserved request keeps bouncing forever.
        let daa = 105 + 3 * w;
        let r2 = [2u8; 32];
        assert!(ledger.on_chain_block(daa, &[(r2, 0)], &[(r1, Some(a))], &[], draw).is_empty());
        assert!(ledger.on_chain_block(daa + 1, &[], &[], &[], draw).is_empty()); // assigns a
        assert!(ledger.on_chain_block(daa + 2, &[], &[(r2, Some(a))], &[], draw).is_empty()); // served in window
        assert_eq!(ledger.consecutive_misses(&a, daa + 2), 0);

        let r3 = [3u8; 32];
        assert!(ledger.on_chain_block(daa + 3, &[(r3, 0)], &[], &[], draw).is_empty());
        assert!(ledger.on_chain_block(daa + 4, &[], &[], &[], draw).is_empty());
        let misses = ledger.on_chain_block(daa + 5 + w, &[], &[], &[], draw);
        assert_eq!(misses.len(), 1);
        assert_eq!(misses[0].miner, a);
        assert_eq!(misses[0].consecutive_misses, 1);
    }

    #[test]
    fn late_response_in_closing_block_cancels_the_miss() {
        let a = Hash::from_bytes([1u8; 32]);
        let eligible = [a];
        let mut ledger = ServiceLedger::default();
        let w = service_window_daa(0);
        let draw = |_tier: u8, excluded: &[Hash]| draw_assignment(&eligible, excluded, &[0u8; 32]);

        let rh = [9u8; 32];
        ledger.on_chain_block(100, &[(rh, 0)], &[], &[], draw);
        ledger.on_chain_block(101, &[], &[], &[], draw);
        // window is past, but the same chain block carries the response: served, no miss
        assert!(ledger.on_chain_block(200 + w, &[], &[(rh, Some(a))], &[], draw).is_empty());
        assert_eq!(ledger.pending_len(), 0);
    }

    #[test]
    fn penalties_burn_newest_claims_then_drain() {
        use super::EscrowClaim;
        use crate::tx::TransactionOutpoint;

        let a = Hash::from_bytes([1u8; 32]);
        let eligible = [a];
        let mut ledger = ServiceLedger::default();
        let w = service_window_daa(0);
        let draw = |_tier: u8, excluded: &[Hash]| draw_assignment(&eligible, excluded, &[0u8; 32]);
        let claim = |n: u64, daa: u64| EscrowClaim { outpoint: TransactionOutpoint::new(n.into(), 1), value: n, daa };

        // six claims accumulated, then r1 assigned to a and missed: strike 1 burns the 5 NEWEST
        let escrows: Vec<(Hash, EscrowClaim)> = (1..=6).map(|n| (a, claim(n, 100))).collect();
        let r1 = [1u8; 32];
        ledger.on_chain_block(100, &[(r1, 0)], &[], &escrows, draw);
        ledger.on_chain_block(101, &[], &[], &[], draw);
        let misses = ledger.on_chain_block(102 + w, &[], &[], &[], draw);
        assert_eq!(misses.len(), 1);
        assert_eq!(misses[0].penalty, ServicePenalty::BurnClaims(STRIKE_1_BURN_CLAIMS));
        assert_eq!(misses[0].burned.iter().map(|c| c.value).collect::<Vec<_>>(), vec![6, 5, 4, 3, 2]);

        // strike 2 (still unserved, a re-drawn by fallback): SlashAllPending drains the leftover
        // claim plus one accumulated meanwhile
        let fresh = [(a, claim(7, 102 + w))];
        ledger.on_chain_block(102 + w, &[], &[], &fresh, draw);
        let misses = ledger.on_chain_block(103 + 2 * w, &[], &[], &[], draw);
        assert_eq!(misses.len(), 1);
        assert_eq!(misses[0].penalty, ServicePenalty::SlashAllPending);
        assert_eq!(misses[0].burned.iter().map(|c| c.value).collect::<Vec<_>>(), vec![7, 1]);

        // strike 3 (BanIp): claims re-accumulated past the full slash burn too
        let fresh = [(a, claim(8, 103 + 2 * w))];
        ledger.on_chain_block(103 + 2 * w, &[], &[], &fresh, draw);
        let misses = ledger.on_chain_block(104 + 3 * w, &[], &[], &[], draw);
        assert_eq!(misses.len(), 1);
        assert_eq!(misses[0].penalty, ServicePenalty::BanIp);
        assert_eq!(misses[0].burned.iter().map(|c| c.value).collect::<Vec<_>>(), vec![8]);

        // a miner with an empty vault yields an empty burn list, never a panic
        let r2 = [2u8; 32];
        ledger.on_chain_block(200 + 4 * w, &[(r2, 0)], &[], &[], draw);
        ledger.on_chain_block(201 + 4 * w, &[], &[], &[], draw);
        let misses = ledger.on_chain_block(202 + 5 * w, &[], &[], &[], draw);
        assert_eq!(misses.len(), 2); // r1 still bouncing + r2
        assert!(misses.iter().all(|m| m.burned.is_empty()));
    }

    #[test]
    fn horizon_expires_pendings_and_strikes() {
        let a = Hash::from_bytes([1u8; 32]);
        let eligible = [a];
        let mut ledger = ServiceLedger::default();
        let w = service_window_daa(0);
        let draw = |_tier: u8, excluded: &[Hash]| draw_assignment(&eligible, excluded, &[0u8; 32]);

        let rh = [9u8; 32];
        ledger.on_chain_block(100, &[(rh, 0)], &[], &[], draw);
        ledger.on_chain_block(101, &[], &[], &[], draw);
        let misses = ledger.on_chain_block(102 + w, &[], &[], &[], draw);
        assert_eq!(misses.len(), 1);
        assert_eq!(ledger.consecutive_misses(&a, 102 + w), 1);

        // beyond the horizon the strike reads zero and the request has expired
        let far = 102 + w + SERVICE_LEDGER_HORIZON_DAA;
        assert_eq!(ledger.consecutive_misses(&a, far), 0);
        ledger.on_chain_block(far, &[], &[], &[], draw);
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
        // deterministic assignment over the eligible set
        let set = eligible_miners(&recent, 0);
        let i = assign_index(&[9u8; 32], set.len()).unwrap();
        assert!(i < set.len());
    }

    #[test]
    fn strike_penalty_escalation() {
        assert_eq!(strike_penalty(0), ServicePenalty::None);
        assert_eq!(strike_penalty(1), ServicePenalty::BurnClaims(STRIKE_1_BURN_CLAIMS));
        assert_eq!(strike_penalty(2), ServicePenalty::SlashAllPending);
        assert_eq!(strike_penalty(3), ServicePenalty::BanIp);
        assert_eq!(strike_penalty(9), ServicePenalty::BanIp);
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

    #[test]
    fn assign_index_deterministic_bounded_and_spread() {
        let seed = [7u8; 32];
        assert_eq!(assign_index(&seed, 0), None);
        assert!(assign_index(&seed, 10).unwrap() < 10);
        assert_eq!(assign_index(&seed, 10), assign_index(&seed, 10));

        let mut hits = [0u32; 8];
        for k in 0u8..64 {
            let mut s = [0u8; 32];
            s[0] = k;
            hits[assign_index(&s, 8).unwrap()] += 1;
        }
        assert!(hits.iter().all(|&h| h > 0), "every bucket hit: {hits:?}");
    }
}
