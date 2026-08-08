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

#[cfg(test)]
mod tests {
    use super::{assign_index, strike_penalty, update_strikes, ServicePenalty, STRIKE_1_BURN_CLAIMS};

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
