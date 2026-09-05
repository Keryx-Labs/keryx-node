use crate::tx::{ScriptPublicKey, Transaction};
use serde::{Deserialize, Serialize};

#[derive(PartialEq, Eq, Debug, Clone)]
pub struct MinerData<T: AsRef<[u8]> = Vec<u8>> {
    pub script_public_key: ScriptPublicKey,
    pub extra_data: T,
}

impl<T: AsRef<[u8]>> MinerData<T> {
    pub fn new(script_public_key: ScriptPublicKey, extra_data: T) -> Self {
        Self { script_public_key, extra_data }
    }
}

#[derive(PartialEq, Eq, Debug)]
pub struct CoinbaseData<T: AsRef<[u8]> = Vec<u8>> {
    pub blue_score: u64,
    pub subsidy: u64,
    pub miner_data: MinerData<T>,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct BlockRewardData {
    pub subsidy: u64,
    pub total_fees: u64,
    pub script_public_key: ScriptPublicKey,
    /// Escrow SPK parsed from the block's coinbase extra_data.
    /// `Some` = OPoI miner (20 % escrow output), `None` = standard miner (20 % burned).
    pub escrow_script_public_key: Option<ScriptPublicKey>,
}

impl BlockRewardData {
    pub fn new(subsidy: u64, total_fees: u64, script_public_key: ScriptPublicKey) -> Self {
        Self { subsidy, total_fees, script_public_key, escrow_script_public_key: None }
    }

    pub fn new_with_escrow(
        subsidy: u64,
        total_fees: u64,
        script_public_key: ScriptPublicKey,
        escrow_script_public_key: Option<ScriptPublicKey>,
    ) -> Self {
        Self { subsidy, total_fees, script_public_key, escrow_script_public_key }
    }
}

/// Holds a coinbase transaction along with meta-data obtained during creation
pub struct CoinbaseTransactionTemplate {
    pub tx: Transaction,
    pub has_red_reward: bool,
    /// Index of the red-blocks reward output within the coinbase outputs, if present.
    /// Used by modify_block_template to rewrite the correct output when changing miner address.
    pub red_reward_output_index: Option<usize>,
    /// Per-payout-SPK split of what this coinbase pays, emitted by the builder because it cannot
    /// be reconstructed from the finished transaction: a miner cut is the base cut scaled by the
    /// tier and ratio brackets **of this block's view** (neither map survives validation), and an
    /// inference-reward mint is indistinguishable from a miner cut by script alone. One entry per
    /// SPK, already aggregated over the blues that share it.
    pub payouts: Vec<(ScriptPublicKey, CoinbasePayout)>,
}

/// What one payout SPK earns from a single coinbase, split by source, so income can be reported
/// apart from the shortfall the reward brackets destroy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CoinbasePayout {
    /// Miner cut actually paid, after the tier and ratio brackets. Zero for a suspended producer.
    pub paid: u64,
    /// The base miner cut before those brackets; `base − paid` is what was burned.
    pub base: u64,
    /// Escrow slice that accrued to this producer. Zero when it was burned at emission (a
    /// standard miner announces no escrow key, so the slice is paid to the burn SPK).
    pub escrow: u64,
    /// Inference-reward mints this coinbase routed to this SPK.
    pub inference: u64,
    /// The `base` cut split by the proven model tier of the blue that earned it, indexed by tier.
    /// Display only, and NOT filled by the builder — the builder is handed bracket multipliers in
    /// bps, and mapping bps back to a tier would misread a standing-demoted top tier as the entry
    /// tier, which is the exact confusion this index exists to remove. It is filled from each
    /// blue's own header instead, at the two places that know it.
    ///
    /// `Σ tier_base` equals `base` for blues whose tier is resolvable, and falls short of it for
    /// pre-PoM blues that have none — so a share must be taken over the sum of these, never over
    /// `base`, or an untiered block would silently dilute the mix.
    pub tier_base: [u64; TIER_BUCKETS],
}

/// Tier buckets tracked per payout SPK. Five: the H6 schedule's width, which is also H2's. The
/// legacy pre-H2 schedule had four, and its tiers are a prefix of these, so a shorter schedule
/// simply leaves the top bucket empty.
pub const TIER_BUCKETS: usize = 5;
