pub use super::{
    bps::{Bps, TenBps},
    constants::consensus::*,
    genesis::{DEVNET_GENESIS, GENESIS, GenesisBlock, SIMNET_GENESIS, TESTNET_GENESIS, TESTNET11_GENESIS},
};

// ── Inference reward minimums ─────────────────────────────────────────────────
// model_id = sha2-256(primary_weight_file) = CIDv0_decoded_bytes[2..34].

/// TinyLlama 1.1B — sha2-256(QmdqcmS8aMngiZWYYdeZEaW22N6XRTd9zK5ZCJG1MPmrQ3)
pub const TINYLLAMA_MODEL_ID: [u8; 32] = [
    0xe6, 0x4a, 0xf3, 0x68, 0xec, 0x93, 0x51, 0xa5,
    0xa4, 0xc0, 0xec, 0x7a, 0xe4, 0x7d, 0x42, 0xad,
    0xa7, 0xf6, 0xb3, 0xf1, 0xa6, 0xe6, 0x0f, 0xc7,
    0x3d, 0x0e, 0xb6, 0xca, 0x29, 0x53, 0x64, 0x5c,
];

/// DeepSeek-R1-8B — sha2-256(QmYK1faUGNMYZ2UKeSpUoUoFpRarZQEwfPCHbYNG2ib2mR)
pub const DEEPSEEK_R1_8B_MODEL_ID: [u8; 32] = [
    0x94, 0x29, 0x67, 0x33, 0x16, 0xbc, 0x40, 0xec,
    0x06, 0x67, 0x89, 0x45, 0x34, 0x57, 0x8b, 0x41,
    0x23, 0x6f, 0xc7, 0xee, 0xa4, 0xd9, 0x31, 0xf1,
    0x48, 0x9c, 0x34, 0xc5, 0x83, 0x7f, 0x42, 0xf4,
];

/// DeepSeek-R1-32B — sha2-256(model.gguf) computed locally
pub const DEEPSEEK_R1_32B_MODEL_ID: [u8; 32] = [
    0xbe, 0xd9, 0xb0, 0xf5, 0x51, 0xf5, 0xb9, 0x5b,
    0xf9, 0xda, 0x58, 0x88, 0xa4, 0x8f, 0x0f, 0x87,
    0xc3, 0x7a, 0xd6, 0xb7, 0x25, 0x19, 0xc4, 0xcb,
    0xd7, 0x75, 0xf5, 0x4a, 0xc0, 0xb9, 0xfc, 0x62,
];

/// LLaMA-3.3-70B — Q4_K_M, CIDv0[2..34] of model.gguf (re-quantized from the old IQ3,
/// which candle 0.8.4 cannot read). Must match keryx-miner models.rs LLAMA_3_3_70B.
pub const LLAMA_3_3_70B_MODEL_ID: [u8; 32] = [
    0xed, 0xf4, 0x76, 0xbd, 0x67, 0xa2, 0xf7, 0xb1,
    0x9b, 0x40, 0xa1, 0x7d, 0xef, 0x4c, 0xaa, 0x3c,
    0x84, 0x7b, 0x68, 0xfd, 0xa1, 0x8a, 0x3c, 0x31,
    0x29, 0x35, 0xb0, 0xb3, 0x43, 0xae, 0xb3, 0x3e,
];

/// LLaMA-3.3-70B — legacy IQ3 weights id, recognised only BEFORE `opoi_v2_activation`.
/// Kept so pre-hardfork blocks revalidate identically (the deployed network enforces
/// the IQ3 id); the Q4_K_M id above takes over at the gate.
pub const LLAMA_3_3_70B_MODEL_ID_LEGACY: [u8; 32] = [
    0xaa, 0xd2, 0xcf, 0x33, 0x48, 0xd8, 0xc7, 0xfd,
    0xbd, 0x2c, 0x0d, 0xd5, 0x8e, 0x0d, 0x99, 0x36,
    0x84, 0x50, 0xd4, 0x3c, 0x95, 0x84, 0xae, 0xf8,
    0x1a, 0x46, 0x7d, 0xd3, 0x47, 0x56, 0x13, 0x44,
];

/// Qwen3-32B — Q6_K, CIDv0[2..34] of model.gguf. Must match keryx-miner models.rs QWEN3_32B.
/// Recognised only from `opoi_v2_activation` (5090 tier model added at the hardfork).
pub const QWEN3_32B_MODEL_ID: [u8; 32] = [
    0xf0, 0x7e, 0x57, 0xb1, 0x1e, 0xd6, 0xcc, 0xe5,
    0x63, 0x31, 0xae, 0xff, 0x60, 0xcf, 0xdb, 0x36,
    0x24, 0xbd, 0x97, 0xe7, 0x03, 0x78, 0x8c, 0xba,
    0x02, 0xce, 0x00, 0xfa, 0xe7, 0x9a, 0xb0, 0x43,
];

/// Qwen3-235B-A22B — Q4_K_M, CIDv0[2..34] of model.gguf. Must match keryx-miner
/// models.rs QWEN3_235B. Recognised only from `opoi_v2_activation` (--very-ultra
/// multi-GPU tier added at the hardfork).
/// CID = QmVV8HHdoVEz5bRY28eii3Mb3BunRYhcBwoC1PigdJ3tp3 (pinned Q4_K_M merged gguf).
pub const QWEN3_235B_MODEL_ID: [u8; 32] = [
    0x6a, 0x2d, 0xa1, 0x7d, 0x05, 0x16, 0x65, 0x09,
    0x08, 0x4e, 0x95, 0x1c, 0xe4, 0x1d, 0x8b, 0x77,
    0xf1, 0x49, 0x63, 0xad, 0xab, 0x55, 0xe9, 0xd5,
    0xa9, 0x85, 0x6c, 0x22, 0xa7, 0x99, 0x39, 0xf4,
];

/// Per-model minimum inference_reward in sompi (from `opoi_v2_activation`: Q4_K_M 70B id,
/// Qwen3-32B added, 70B floor raised to 5.0 KRX).
pub const INFERENCE_REWARD_MINIMUMS: &[([u8; 32], u64)] = &[
    (TINYLLAMA_MODEL_ID,         50_000_000),   // 0.5 KRX
    (DEEPSEEK_R1_8B_MODEL_ID,   150_000_000),   // 1.5 KRX
    (DEEPSEEK_R1_32B_MODEL_ID,  250_000_000),   // 2.5 KRX
    (QWEN3_32B_MODEL_ID,        350_000_000),   // 3.5 KRX
    (LLAMA_3_3_70B_MODEL_ID,    500_000_000),   // 5.0 KRX
    (QWEN3_235B_MODEL_ID,       700_000_000),   // 7.0 KRX — flagship multi-GPU tier
];

/// Pre-`opoi_v2_activation` table — identical except the 70B entry uses the legacy IQ3 id.
/// Matches what the network enforced before the hardfork, so historical blocks revalidate.
pub const INFERENCE_REWARD_MINIMUMS_LEGACY: &[([u8; 32], u64)] = &[
    (TINYLLAMA_MODEL_ID,              50_000_000),   // 0.5 KRX
    (DEEPSEEK_R1_8B_MODEL_ID,        150_000_000),   // 1.5 KRX
    (DEEPSEEK_R1_32B_MODEL_ID,       250_000_000),   // 2.5 KRX
    (LLAMA_3_3_70B_MODEL_ID_LEGACY, 400_000_000),   // 4.0 KRX
];

/// Tier-reward — coinbase-subsidy multiplier in basis points per declared model
/// tier, applied from `tier_reward_activation`. Index = tier rank, ordered by
/// `TIER_MODEL_IDS` (0 = lightest model, 5 = heaviest). Monotone: serving a
/// heavier model earns a larger share of the block subsidy. The un-earned delta
/// is never minted — a deflationary "burn" with no burn output. The top tier is
/// the 100% reference so the scheme never inflates emission beyond the schedule.
pub const TIER_REWARD_BPS: [u64; 6] = [
    8_500,  // 0  --light       TinyLlama      -15%
    8_800,  // 1  default       DeepSeek-8B    -12%
    9_100,  // 2  --high        DeepSeek-32B    -9%
    9_400,  // 3  --very-high   Qwen3-32B       -6%
    9_700,  // 4  --ultra       LLaMA-3.3-70B   -3%
    10_000, // 5  --very-ultra  Qwen3-235B       0%
];

/// Basis-points divisor for `TIER_REWARD_BPS`.
pub const TIER_REWARD_BPS_DIVISOR: u64 = 10_000;

/// Balance-reward brackets: a block's coinbase miner cut is additionally scaled
/// (MULTIPLICATIVELY with the tier multiplier) by how much KRX the miner holds at
/// its payout address, proven by `/bal:` outpoints embedded in the coinbase. Applied
/// from `balance_reward_activation`; the un-earned delta is burned (deflationary).
/// Pick the highest bracket whose `BALANCE_REWARD_THRESHOLDS_SOMPI` is met. Reuses
/// `TIER_REWARD_BPS_DIVISOR`. Both factors floored ⇒ never inflates beyond schedule.
pub const BALANCE_REWARD_BPS: [u64; 5] = [
    8_000,  // 0  < 50k KRX         80%
    8_700,  // 1  ≥ 50k KRX         87%
    9_200,  // 2  ≥ 200k KRX        92%
    9_700,  // 3  ≥ 500k KRX        97%
    10_000, // 4  ≥ 1,000,000 KRX  100%
];

/// Minimum balance (in sompi) to earn each `BALANCE_REWARD_BPS` bracket. 1 KRX = 1e8 sompi.
pub const BALANCE_REWARD_THRESHOLDS_SOMPI: [u64; 5] = [
    0,
    50_000 * 100_000_000,
    200_000 * 100_000_000,
    500_000 * 100_000_000,
    1_000_000 * 100_000_000,
];

/// Max number of `/bal:` outpoints a coinbase may reference (bounds coinbase bloat
/// and validation cost). References beyond this are ignored deterministically.
pub const BALANCE_REWARD_MAX_OUTPOINTS: usize = 8;

/// Canonical model ids ordered lightest → heaviest. A block's tier rank is the
/// highest position among the model_ids it declares in its coinbase `ai:cap`
/// field. The legacy IQ3 70B id is mapped to the same rank as the Q4_K_M id by
/// the lookup helper, so a miner straddling the hardfork is not misranked.
pub const TIER_MODEL_IDS: [[u8; 32]; 6] = [
    TINYLLAMA_MODEL_ID,
    DEEPSEEK_R1_8B_MODEL_ID,
    DEEPSEEK_R1_32B_MODEL_ID,
    QWEN3_32B_MODEL_ID,
    LLAMA_3_3_70B_MODEL_ID,
    QWEN3_235B_MODEL_ID,
];

/// Grace, in epochs, before a miner's stale liveness invalidates its blocks.
/// A block is acceptable if its miner answered the synthetic task within the
/// last `1 + GRACE` epochs — tolerates a short outage / model reload without
/// kicking an honest miner. Only consulted once `synthetic_liveness_activation`
/// is live (enforcement is in Step 4; recording is unconditional).
pub const SYNTHETIC_LIVENESS_GRACE_EPOCHS: u64 = 1;
use crate::{
    BlockLevel, KType,
    constants::STORAGE_MASS_PARAMETER,
    network::{NetworkId, NetworkType},
};
use keryx_addresses::Prefix;
use keryx_math::Uint256;
use serde::{Deserialize, Serialize};
use std::{
    cmp::min,
    ops::{Deref, DerefMut},
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForkActivation(u64);

impl ForkActivation {
    const NEVER: u64 = u64::MAX;
    const ALWAYS: u64 = 0;

    pub const fn new(daa_score: u64) -> Self {
        Self(daa_score)
    }

    pub const fn never() -> Self {
        Self(Self::NEVER)
    }

    pub const fn always() -> Self {
        Self(Self::ALWAYS)
    }

    /// Returns the actual DAA score triggering the activation. Should be used only
    /// for cases where the explicit value is required for computations (e.g., coinbase subsidy).
    /// Otherwise, **activation checks should always go through `self.is_active(..)`**
    pub fn daa_score(self) -> u64 {
        self.0
    }

    pub fn is_active(self, current_daa_score: u64) -> bool {
        current_daa_score >= self.0
    }

    /// Checks if the fork was "recently" activated, i.e., in the time frame of the provided range.
    /// This function returns false for forks that were always active, since they were never activated.
    pub fn is_within_range_from_activation(self, current_daa_score: u64, range: u64) -> bool {
        self != Self::always() && self.is_active(current_daa_score) && current_daa_score < self.0 + range
    }

    /// Checks if the fork is expected to be activated "soon", i.e., in the time frame of the provided range.
    /// Returns the distance from activation if so, or `None` otherwise.  
    pub fn is_within_range_before_activation(self, current_daa_score: u64, range: u64) -> Option<u64> {
        if !self.is_active(current_daa_score) && current_daa_score + range > self.0 { Some(self.0 - current_daa_score) } else { None }
    }
}

/// A consensus parameter which depends on forking activation
#[derive(Clone, Copy, Debug)]
pub struct ForkedParam<T: Copy> {
    pre: T,
    post: T,
    activation: ForkActivation,
}

impl<T: Copy> ForkedParam<T> {
    const fn new(pre: T, post: T, activation: ForkActivation) -> Self {
        Self { pre, post, activation }
    }

    pub const fn new_const(val: T) -> Self {
        Self { pre: val, post: val, activation: ForkActivation::never() }
    }

    pub fn activation(&self) -> ForkActivation {
        self.activation
    }

    pub fn get(&self, daa_score: u64) -> T {
        if self.activation.is_active(daa_score) { self.post } else { self.pre }
    }

    /// Returns the value before activation (=pre unless activation = always)
    pub fn before(&self) -> T {
        match self.activation.0 {
            ForkActivation::ALWAYS => self.post,
            _ => self.pre,
        }
    }

    /// Returns the permanent long-term value after activation (=post unless the activation is never scheduled)
    pub fn after(&self) -> T {
        match self.activation.0 {
            ForkActivation::NEVER => self.pre,
            _ => self.post,
        }
    }

    /// Maps the ForkedParam<T> to a new ForkedParam<U> by applying a map function on both pre and post
    pub fn map<U: Copy, F: Fn(T) -> U>(&self, f: F) -> ForkedParam<U> {
        ForkedParam::new(f(self.pre), f(self.post), self.activation)
    }
}

impl<T: Copy + Ord> ForkedParam<T> {
    /// Returns the min of `pre` and `post` values. Useful for non-consensus initializations
    /// which require knowledge of the value bounds.
    ///
    /// Note that if activation is not scheduled (set to never) then pre is always returned,
    /// and if activation is set to always (since inception), post will be returned.
    pub fn lower_bound(&self) -> T {
        match self.activation.0 {
            ForkActivation::NEVER => self.pre,
            ForkActivation::ALWAYS => self.post,
            _ => self.pre.min(self.post),
        }
    }

    /// Returns the max of `pre` and `post` values. Useful for non-consensus initializations
    /// which require knowledge of the value bounds.
    ///
    /// Note that if activation is not scheduled (set to never) then pre is always returned,
    /// and if activation is set to always (since inception), post will be returned.
    pub fn upper_bound(&self) -> T {
        match self.activation.0 {
            ForkActivation::NEVER => self.pre,
            ForkActivation::ALWAYS => self.post,
            _ => self.pre.max(self.post),
        }
    }
}

/// Blockrate-related consensus params.
/// Grouped together under a single struct because they are logically related and
/// in order to easily support **future BPS acceleration hardforks** (by simply adding
/// a forked instance of blockrate params to the main [`Params`]).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BlockrateParams {
    pub target_time_per_block: u64, // (milliseconds)
    pub ghostdag_k: KType,
    pub past_median_time_sample_rate: u64,
    pub difficulty_sample_rate: u64,
    pub max_block_parents: u8,
    pub mergeset_size_limit: u64,
    pub merge_depth: u64,
    pub finality_depth: u64,
    pub pruning_depth: u64,
    pub coinbase_maturity: u64,
}

impl BlockrateParams {
    pub const fn new<const BPS: u64>() -> Self {
        Self {
            target_time_per_block: Bps::<BPS>::target_time_per_block(),
            ghostdag_k: Bps::<BPS>::ghostdag_k(),
            past_median_time_sample_rate: Bps::<BPS>::past_median_time_sample_rate(),
            difficulty_sample_rate: Bps::<BPS>::difficulty_adjustment_sample_rate(),
            max_block_parents: Bps::<BPS>::max_block_parents(),
            mergeset_size_limit: Bps::<BPS>::mergeset_size_limit(),
            merge_depth: Bps::<BPS>::merge_depth_bound(),
            finality_depth: Bps::<BPS>::finality_depth(),
            pruning_depth: Bps::<BPS>::pruning_depth(),
            coinbase_maturity: Bps::<BPS>::coinbase_maturity(),
        }
    }

    pub const fn increase_max_block_parents(mut self, max_block_parents: u8) -> Self {
        if self.max_block_parents < max_block_parents {
            self.max_block_parents = max_block_parents;
        }
        self
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OverrideParams {
    /// Timestamp deviation tolerance (in seconds)
    pub timestamp_deviation_tolerance: Option<u64>,

    /// Size of the sampled block window that is used to calculate the past median time of each block
    pub past_median_time_window_size: Option<usize>,

    /// Size of the sampled block window that is used to calculate the required difficulty of each block
    pub difficulty_window_size: Option<usize>,

    /// The minimum size a difficulty window (full or sampled) must have to trigger a DAA calculation
    pub min_difficulty_window_size: Option<usize>,

    pub coinbase_payload_script_public_key_max_len: Option<u8>,
    pub max_coinbase_payload_len: Option<usize>,

    pub max_tx_inputs: Option<usize>,
    pub max_tx_outputs: Option<usize>,
    pub max_signature_script_len: Option<usize>,
    pub max_script_public_key_len: Option<usize>,
    pub mass_per_tx_byte: Option<u64>,
    pub mass_per_script_pub_key_byte: Option<u64>,
    pub mass_per_sig_op: Option<u64>,
    pub max_block_mass: Option<u64>,

    /// The parameter for scaling inverse KRX value to mass units (KIP-0009)
    pub storage_mass_parameter: Option<u64>,

    /// DAA score after which the pre-deflationary period switches to the deflationary period
    pub deflationary_phase_daa_score: Option<u64>,

    pub pre_deflationary_phase_base_subsidy: Option<u64>,
    pub skip_proof_of_work: Option<bool>,
    pub max_block_level: Option<BlockLevel>,
    pub pruning_proof_m: Option<u64>,

    /// Blockrate-related params
    pub blockrate: Option<BlockrateParams>,

    /// Target time per block prior to the crescendo hardfork (in milliseconds)
    pub pre_crescendo_target_time_per_block: Option<u64>,

    /// Crescendo activation DAA score
    pub crescendo_activation: Option<ForkActivation>,

    /// Model capability enforcement hardfork activation DAA score
    pub model_cap_enforcement_activation: Option<ForkActivation>,

    #[serde(skip)]
    pub inference_reward_minimums: Option<&'static [([u8; 32], u64)]>,
}

impl From<Params> for OverrideParams {
    fn from(p: Params) -> Self {
        Self {
            timestamp_deviation_tolerance: Some(p.timestamp_deviation_tolerance),
            pre_crescendo_target_time_per_block: Some(p.pre_crescendo_target_time_per_block),
            difficulty_window_size: Some(p.difficulty_window_size),
            past_median_time_window_size: Some(p.past_median_time_window_size),
            min_difficulty_window_size: Some(p.min_difficulty_window_size),
            coinbase_payload_script_public_key_max_len: Some(p.coinbase_payload_script_public_key_max_len),
            max_coinbase_payload_len: Some(p.max_coinbase_payload_len),
            max_tx_inputs: Some(p.max_tx_inputs),
            max_tx_outputs: Some(p.max_tx_outputs),
            max_signature_script_len: Some(p.max_signature_script_len),
            max_script_public_key_len: Some(p.max_script_public_key_len),
            mass_per_tx_byte: Some(p.mass_per_tx_byte),
            mass_per_script_pub_key_byte: Some(p.mass_per_script_pub_key_byte),
            mass_per_sig_op: Some(p.mass_per_sig_op),
            max_block_mass: Some(p.max_block_mass),
            storage_mass_parameter: Some(p.storage_mass_parameter),
            deflationary_phase_daa_score: Some(p.deflationary_phase_daa_score),
            pre_deflationary_phase_base_subsidy: Some(p.pre_deflationary_phase_base_subsidy),
            skip_proof_of_work: Some(p.skip_proof_of_work),
            max_block_level: Some(p.max_block_level),
            pruning_proof_m: Some(p.pruning_proof_m),
            blockrate: Some(p.blockrate),
            crescendo_activation: Some(p.crescendo_activation),
            model_cap_enforcement_activation: Some(p.model_cap_enforcement_activation),
            inference_reward_minimums: Some(p.inference_reward_minimums),
        }
    }
}

/// Consensus parameters. Contains settings and configurations which are consensus-sensitive.
/// Changing one of these on a network node would exclude and prevent it from reaching consensus
/// with the other unmodified nodes.
#[derive(Clone, Debug)]
pub struct Params {
    pub dns_seeders: &'static [&'static str],
    pub net: NetworkId,
    pub genesis: GenesisBlock,

    /// Timestamp deviation tolerance (in seconds)
    pub timestamp_deviation_tolerance: u64,

    /// Defines the highest allowed proof of work difficulty value for a block as a [`Uint256`]
    pub max_difficulty_target: Uint256,

    /// Highest allowed proof of work difficulty as a floating number
    pub max_difficulty_target_f64: f64,

    /// Size of the sampled block window that is used to calculate the past median time of each block
    pub past_median_time_window_size: usize,

    /// Size of the sampled block window that is used to calculate the required difficulty of each block
    pub difficulty_window_size: usize,

    /// The minimum size a difficulty window must have to trigger a DAA calculation
    pub min_difficulty_window_size: usize,

    pub coinbase_payload_script_public_key_max_len: u8,
    pub max_coinbase_payload_len: usize,

    pub max_tx_inputs: usize,
    pub max_tx_outputs: usize,
    pub max_signature_script_len: usize,
    pub max_script_public_key_len: usize,

    pub mass_per_tx_byte: u64,
    pub mass_per_script_pub_key_byte: u64,
    pub mass_per_sig_op: u64,
    pub max_block_mass: u64,

    /// The parameter for scaling inverse KRX value to mass units (KIP-0009)
    pub storage_mass_parameter: u64,

    /// DAA score after which the pre-deflationary period switches to the deflationary period
    pub deflationary_phase_daa_score: u64,

    pub pre_deflationary_phase_base_subsidy: u64,
    pub skip_proof_of_work: bool,
    pub max_block_level: BlockLevel,
    pub pruning_proof_m: u64,

    /// Blockrate-related params
    pub blockrate: BlockrateParams,

    /// Target time per block prior to the crescendo hardfork (in milliseconds).
    /// Required permanently in order to calculate the subsidy month from the current DAA score
    pub pre_crescendo_target_time_per_block: u64,

    /// Crescendo activation DAA score
    pub crescendo_activation: ForkActivation,

    /// Model capability enforcement hardfork activation DAA score.
    /// After this score, blocks containing AiResponse txs whose model_id is not
    /// declared in the coinbase ai:cap: field are rejected by consensus.
    pub model_cap_enforcement_activation: ForkActivation,

    /// Per-model minimum inference_reward (sompi) enforced from `model_cap_enforcement_activation`.
    /// AiRequest txs below the minimum for their model_id are rejected.
    /// Fulfilled inference_rewards are redirected from the fee burn to the responding miner.
    pub inference_reward_minimums: &'static [([u8; 32], u64)],

    /// PoW SALT v2 hardfork activation DAA score.
    /// After this score, `KERYX_MATRIX_SALT_V2` is used for matrix generation instead of v1.
    /// Any miner binary compiled against v1 will compute a different matrix and its blocks
    /// will fail PoW validation — this is the forced-update mechanism.
    /// Set to `ForkActivation::never()` to disable (default for mainnet until announced).
    pub pow_salt_v2_activation: ForkActivation,

    /// PoW SALT v4 hardfork activation DAA score (chain relaunch).
    /// After this score, `KERYX_MATRIX_SALT_V4` is used for matrix generation instead of v2.
    /// This forks cleanly away from the abandoned SALT-v3 / diff-spiral chain while keeping
    /// stock difficulty (no genesis reset). Same forced-update mechanism as v2.
    pub pow_salt_v4_activation: ForkActivation,

    /// PoW SALT v5 hardfork activation DAA score. Shipped at the same DAA as `opoi_v2`
    /// so the OPoI hardfork also bumps the matrix salt (`KERYX_MATRIX_SALT_V5`), forcing
    /// every miner onto the updated binary at the cutover. Same forced-update mechanism as v4.
    pub pow_salt_v5_activation: ForkActivation,

    /// OPoI v2 hardfork activation DAA score. From this score:
    /// - AiResponse payloads must be in the 142-byte v2 format (model_id + result_commitment),
    ///   making every response bindable to its off-chain content (future challenger v2);
    /// - the model cap check uses the embedded model_id (cross-block enforcement, no store reads);
    /// - the 70B model_id switches from the legacy IQ3 weights to the Q4_K_M re-quantization.
    /// Slashing/challenge processing stays disabled — this gate only lays the v2 foundations.
    pub opoi_v2_activation: ForkActivation,

    /// Synthetic-liveness hardfork activation DAA score (Level-1 anti "zero-inference").
    /// From this score, a block is rejected unless its coinbase escrow miner has
    /// answered the protocol's synthetic OPoI task within the last
    /// `1 + SYNTHETIC_LIVENESS_GRACE_EPOCHS` epochs. Recording of synthetic
    /// answers happens regardless of this gate; only the rejection is gated here.
    pub synthetic_liveness_activation: ForkActivation,

    /// Tier-reward hardfork activation DAA score. From this score, each block's
    /// coinbase subsidy is scaled by the multiplier of the highest model tier it
    /// declares in its `ai:cap` field (see `TIER_REWARD_BPS` / `TIER_MODEL_IDS`).
    /// The un-earned delta is never minted (deflationary). Before this gate the
    /// full schedule subsidy is paid, so historical blocks revalidate identically.
    pub tier_reward_activation: ForkActivation,

    /// Balance-reward hardfork activation DAA score. From this score, each block's
    /// coinbase miner cut is additionally scaled by the holdings bracket of the
    /// miner (proven by `/bal:` outpoints paying to the coinbase payout SPK; see
    /// `BALANCE_REWARD_BPS` / `BALANCE_REWARD_THRESHOLDS_SOMPI`), multiplicatively
    /// with the tier multiplier. The un-earned delta is burned. Before this gate
    /// the cut is unaffected, so historical blocks revalidate identically.
    /// Gated to the SAME H as `opoi_v2_activation` (the intended last hardfork).
    pub balance_reward_activation: ForkActivation,
}

impl Params {
    /// Returns the past median time sample rate
    #[inline]
    #[must_use]
    pub fn past_median_time_sample_rate(&self) -> u64 {
        self.blockrate.past_median_time_sample_rate
    }

    /// Returns the difficulty sample rate
    #[inline]
    #[must_use]
    pub fn difficulty_sample_rate(&self) -> u64 {
        self.blockrate.difficulty_sample_rate
    }

    /// Returns the target time per block
    #[inline]
    #[must_use]
    pub fn target_time_per_block(&self) -> u64 {
        self.blockrate.target_time_per_block
    }

    /// Returns the expected number of blocks per second
    #[inline]
    #[must_use]
    pub fn bps(&self) -> u64 {
        1000 / self.blockrate.target_time_per_block
    }

    /// Returns the expected number of blocks per second throughout history (currently represented as [`ForkedParam`]).
    /// Required permanently in order to calculate the subsidy month from the current DAA score.
    #[inline]
    #[must_use]
    pub fn bps_history(&self) -> ForkedParam<u64> {
        ForkedParam::new(
            1000 / self.pre_crescendo_target_time_per_block,
            1000 / self.blockrate.target_time_per_block,
            self.crescendo_activation,
        )
    }

    pub fn ghostdag_k(&self) -> KType {
        self.blockrate.ghostdag_k
    }

    pub fn max_block_parents(&self) -> u8 {
        self.blockrate.max_block_parents
    }

    pub fn mergeset_size_limit(&self) -> u64 {
        self.blockrate.mergeset_size_limit
    }

    pub fn merge_depth(&self) -> u64 {
        self.blockrate.merge_depth
    }

    pub fn finality_depth(&self) -> u64 {
        self.blockrate.finality_depth
    }

    pub fn pruning_depth(&self) -> u64 {
        self.blockrate.pruning_depth
    }

    pub fn coinbase_maturity(&self) -> u64 {
        self.blockrate.coinbase_maturity
    }

    pub fn finality_duration_in_milliseconds(&self) -> u64 {
        self.blockrate.target_time_per_block * self.blockrate.finality_depth
    }

    pub fn difficulty_window_duration_in_block_units(&self) -> u64 {
        self.blockrate.difficulty_sample_rate * self.difficulty_window_size as u64
    }

    pub fn expected_difficulty_window_duration_in_milliseconds(&self) -> u64 {
        self.blockrate.target_time_per_block * self.blockrate.difficulty_sample_rate * self.difficulty_window_size as u64
    }

    /// Returns the depth at which the anticone of a chain block is final (i.e., is a permanently closed set).
    /// Based on the analysis at <https://github.com/kaspanet/docs/blob/main/Reference/prunality/Prunality.pdf>
    /// and on the decomposition of merge depth (rule R-I therein) from finality depth (φ)
    pub fn anticone_finalization_depth(&self) -> u64 {
        let anticone_finalization_depth = self.blockrate.finality_depth
            + self.blockrate.merge_depth
            + 4 * self.blockrate.mergeset_size_limit * self.blockrate.ghostdag_k as u64
            + 2 * self.blockrate.ghostdag_k as u64
            + 2;

        // In mainnet it's guaranteed that `self.pruning_depth` is greater
        // than `anticone_finalization_depth`, but for some tests we use
        // a smaller (unsafe) pruning depth, so we return the minimum of
        // the two to avoid a situation where a block can be pruned and
        // not finalized.
        min(self.blockrate.pruning_depth, anticone_finalization_depth)
    }

    pub fn network_name(&self) -> String {
        self.net.to_prefixed()
    }

    pub fn prefix(&self) -> Prefix {
        self.net.into()
    }

    pub fn default_p2p_port(&self) -> u16 {
        self.net.default_p2p_port()
    }

    pub fn default_rpc_port(&self) -> u16 {
        self.net.default_rpc_port()
    }

    pub fn override_params(self, overrides: OverrideParams) -> Self {
        Self {
            dns_seeders: self.dns_seeders,
            net: self.net,
            genesis: self.genesis.clone(),

            timestamp_deviation_tolerance: overrides.timestamp_deviation_tolerance.unwrap_or(self.timestamp_deviation_tolerance),

            max_difficulty_target: self.max_difficulty_target,
            max_difficulty_target_f64: self.max_difficulty_target_f64,

            difficulty_window_size: overrides.difficulty_window_size.unwrap_or(self.difficulty_window_size),
            past_median_time_window_size: overrides.past_median_time_window_size.unwrap_or(self.past_median_time_window_size),
            min_difficulty_window_size: overrides.min_difficulty_window_size.unwrap_or(self.min_difficulty_window_size),

            coinbase_payload_script_public_key_max_len: overrides
                .coinbase_payload_script_public_key_max_len
                .unwrap_or(self.coinbase_payload_script_public_key_max_len),

            max_coinbase_payload_len: overrides.max_coinbase_payload_len.unwrap_or(self.max_coinbase_payload_len),

            max_tx_inputs: overrides.max_tx_inputs.unwrap_or(self.max_tx_inputs),
            max_tx_outputs: overrides.max_tx_outputs.unwrap_or(self.max_tx_outputs),
            max_signature_script_len: overrides.max_signature_script_len.unwrap_or(self.max_signature_script_len),
            max_script_public_key_len: overrides.max_script_public_key_len.unwrap_or(self.max_script_public_key_len),
            mass_per_tx_byte: overrides.mass_per_tx_byte.unwrap_or(self.mass_per_tx_byte),
            mass_per_script_pub_key_byte: overrides.mass_per_script_pub_key_byte.unwrap_or(self.mass_per_script_pub_key_byte),
            mass_per_sig_op: overrides.mass_per_sig_op.unwrap_or(self.mass_per_sig_op),
            max_block_mass: overrides.max_block_mass.unwrap_or(self.max_block_mass),

            storage_mass_parameter: overrides.storage_mass_parameter.unwrap_or(self.storage_mass_parameter),

            deflationary_phase_daa_score: overrides.deflationary_phase_daa_score.unwrap_or(self.deflationary_phase_daa_score),

            pre_deflationary_phase_base_subsidy: overrides
                .pre_deflationary_phase_base_subsidy
                .unwrap_or(self.pre_deflationary_phase_base_subsidy),

            skip_proof_of_work: overrides.skip_proof_of_work.unwrap_or(self.skip_proof_of_work),

            max_block_level: overrides.max_block_level.unwrap_or(self.max_block_level),

            pruning_proof_m: overrides.pruning_proof_m.unwrap_or(self.pruning_proof_m),

            blockrate: overrides.blockrate.clone().unwrap_or(self.blockrate.clone()),

            pre_crescendo_target_time_per_block: overrides
                .pre_crescendo_target_time_per_block
                .unwrap_or(self.pre_crescendo_target_time_per_block),

            crescendo_activation: overrides.crescendo_activation.unwrap_or(self.crescendo_activation),

            model_cap_enforcement_activation: overrides
                .model_cap_enforcement_activation
                .unwrap_or(self.model_cap_enforcement_activation),

            inference_reward_minimums: overrides
                .inference_reward_minimums
                .unwrap_or(self.inference_reward_minimums),

            pow_salt_v2_activation: self.pow_salt_v2_activation,

            pow_salt_v4_activation: self.pow_salt_v4_activation,

            pow_salt_v5_activation: self.pow_salt_v5_activation,

            opoi_v2_activation: self.opoi_v2_activation,

            synthetic_liveness_activation: self.synthetic_liveness_activation,

            tier_reward_activation: self.tier_reward_activation,

            balance_reward_activation: self.balance_reward_activation,
        }
    }
}

impl Deref for Params {
    type Target = BlockrateParams;

    fn deref(&self) -> &Self::Target {
        &self.blockrate
    }
}

impl DerefMut for Params {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.blockrate
    }
}

impl From<NetworkType> for Params {
    fn from(value: NetworkType) -> Self {
        match value {
            NetworkType::Mainnet => MAINNET_PARAMS,
            NetworkType::Testnet => TESTNET_PARAMS,
            NetworkType::Devnet => DEVNET_PARAMS,
            NetworkType::Simnet => SIMNET_PARAMS,
        }
    }
}

impl From<NetworkId> for Params {
    fn from(value: NetworkId) -> Self {
        match value.network_type {
            NetworkType::Mainnet => MAINNET_PARAMS,
            NetworkType::Testnet => match value.suffix {
                Some(10) => TESTNET_PARAMS,
                Some(x) => panic!("Testnet suffix {} is not supported", x),
                None => panic!("Testnet suffix not provided"),
            },
            NetworkType::Devnet => DEVNET_PARAMS,
            NetworkType::Simnet => SIMNET_PARAMS,
        }
    }
}

pub const MAINNET_PARAMS: Params = Params {
    dns_seeders: &["seed.keryx-labs.com"],
    net: NetworkId::new(NetworkType::Mainnet),
    genesis: GENESIS,
    timestamp_deviation_tolerance: TIMESTAMP_DEVIATION_TOLERANCE,
    max_difficulty_target: MAX_DIFFICULTY_TARGET,
    max_difficulty_target_f64: MAX_DIFFICULTY_TARGET_AS_F64,
    past_median_time_window_size: MEDIAN_TIME_SAMPLED_WINDOW_SIZE as usize,
    difficulty_window_size: DIFFICULTY_SAMPLED_WINDOW_SIZE as usize,
    min_difficulty_window_size: MIN_DIFFICULTY_WINDOW_SIZE,
    coinbase_payload_script_public_key_max_len: 150,
    max_coinbase_payload_len: 2048,

    // Limit the cost of calculating compute/transient/storage masses
    max_tx_inputs: 1000,
    max_tx_outputs: 1000,
    // Transient mass enforces a limit of 125Kb, however script engine max scripts size is 10Kb so there's no point in surpassing that.
    max_signature_script_len: 10_000,
    // Compute mass enforces a limit of ~45.5Kb, however script engine max scripts size is 10Kb so there's no point in surpassing that.
    // Note that storage mass will kick in and gradually penalize also for lower lengths (generalized KIP-0009, plurality will be high).
    max_script_public_key_len: 10_000,

    mass_per_tx_byte: 1,
    mass_per_script_pub_key_byte: 10,
    mass_per_sig_op: 1000,
    max_block_mass: 500_000,

    storage_mass_parameter: STORAGE_MASS_PARAMETER,

    // Keryx launches at 10 BPS from genesis with Crescendo always active.
    // No pre-emission bootstrapping phase is needed — the emission schedule starts at block 0.
    deflationary_phase_daa_score: 0,
    pre_deflationary_phase_base_subsidy: TenBps::pre_deflationary_phase_base_subsidy(),
    skip_proof_of_work: false,
    max_block_level: 225,
    pruning_proof_m: 1000,

    blockrate: BlockrateParams::new::<10>(),

    pre_crescendo_target_time_per_block: TenBps::target_time_per_block(),

    crescendo_activation: ForkActivation::new(0),

    // Hardfork activation: 2026-05-28 15:00 UTC — DAA 11_409_033 + ~4_140_000 (115h × 10 BPS).
    model_cap_enforcement_activation: ForkActivation::new(15_550_000),
    inference_reward_minimums: INFERENCE_REWARD_MINIMUMS,

    // PoW SALT v2: emergency activation 2026-05-30 ~15:00 UTC.
    // DAA estimate: 16_501_908 (current) + 774_000 (21.5h × 10 BPS) = 17_275_908 → rounded down for 2 min margin.
    pow_salt_v2_activation: ForkActivation::new(17_275_000),

    // PoW SALT v4: chain relaunch on stock difficulty. At this score the salt switches v2→v4,
    // forking cleanly away from the abandoned SALT-v3 / diff-1-spiral chain. Same DAA as the
    // old v3 gate so a datadir restored from before this point continues seamlessly into v4.
    pow_salt_v4_activation: ForkActivation::new(21_932_751),
    pow_salt_v5_activation: ForkActivation::never(), // set to the opoi_v2 H DAA at the hardfork

    // TODO(hardfork): set to a concrete DAA (~5-7 days ahead) just before the v1.3.0 release.
    opoi_v2_activation: ForkActivation::never(),

    // TODO(hardfork): set to a concrete DAA (~5-7 days ahead) just before the v1.3.0 release.
    synthetic_liveness_activation: ForkActivation::never(),
    tier_reward_activation: ForkActivation::never(),
    balance_reward_activation: ForkActivation::never(),
};

pub const TESTNET_PARAMS: Params = Params {
    dns_seeders: &[],
    net: NetworkId::with_suffix(NetworkType::Testnet, 10),
    genesis: TESTNET_GENESIS,
    timestamp_deviation_tolerance: TIMESTAMP_DEVIATION_TOLERANCE,
    max_difficulty_target: MAX_DIFFICULTY_TARGET,
    max_difficulty_target_f64: MAX_DIFFICULTY_TARGET_AS_F64,
    past_median_time_window_size: MEDIAN_TIME_SAMPLED_WINDOW_SIZE as usize,
    difficulty_window_size: DIFFICULTY_SAMPLED_WINDOW_SIZE as usize,
    min_difficulty_window_size: MIN_DIFFICULTY_WINDOW_SIZE,
    coinbase_payload_script_public_key_max_len: 150,
    max_coinbase_payload_len: 2048,

    // Limit the cost of calculating compute/transient/storage masses
    max_tx_inputs: 1000,
    max_tx_outputs: 1000,
    // Transient mass enforces a limit of 125Kb, however script engine max scripts size is 10Kb so there's no point in surpassing that.
    max_signature_script_len: 10_000,
    // Compute mass enforces a limit of ~45.5Kb, however script engine max scripts size is 10Kb so there's no point in surpassing that.
    // Note that storage mass will kick in and gradually penalize also for lower lengths (generalized KIP-0009, plurality will be high).
    max_script_public_key_len: 10_000,

    mass_per_tx_byte: 1,
    mass_per_script_pub_key_byte: 10,
    mass_per_sig_op: 1000,
    max_block_mass: 500_000,

    storage_mass_parameter: STORAGE_MASS_PARAMETER,

    // Keryx testnet launches at 10 BPS from genesis with Crescendo always active.
    deflationary_phase_daa_score: 0,
    pre_deflationary_phase_base_subsidy: TenBps::pre_deflationary_phase_base_subsidy(),
    skip_proof_of_work: false,
    max_block_level: 250,
    pruning_proof_m: 1000,

    blockrate: BlockrateParams::new::<10>(),

    pre_crescendo_target_time_per_block: TenBps::target_time_per_block(),

    crescendo_activation: ForkActivation::new(0),

    // Testnet rollout (10 BPS): legacy gates active from genesis (salt pinned to v4,
    // no mid-chain transition), new OPoI-v2-era gates flip together at DAA 1_000.
    model_cap_enforcement_activation: ForkActivation::new(0),
    inference_reward_minimums: INFERENCE_REWARD_MINIMUMS,

    pow_salt_v2_activation: ForkActivation::new(0),
    pow_salt_v4_activation: ForkActivation::new(0),
    pow_salt_v5_activation: ForkActivation::new(1_000), // same DAA as opoi_v2 (testnet cutover)

    // New OPoI-v2-era gates: all activate together at 1_000.
    opoi_v2_activation: ForkActivation::new(1_000),
    synthetic_liveness_activation: ForkActivation::new(1_000),
    tier_reward_activation: ForkActivation::new(1_000),
    balance_reward_activation: ForkActivation::new(1_000),
};

pub const SIMNET_PARAMS: Params = Params {
    dns_seeders: &[],
    net: NetworkId::new(NetworkType::Simnet),
    genesis: SIMNET_GENESIS,
    timestamp_deviation_tolerance: TIMESTAMP_DEVIATION_TOLERANCE,
    max_difficulty_target: MAX_DIFFICULTY_TARGET,
    max_difficulty_target_f64: MAX_DIFFICULTY_TARGET_AS_F64,
    past_median_time_window_size: MEDIAN_TIME_SAMPLED_WINDOW_SIZE as usize,
    difficulty_window_size: DIFFICULTY_SAMPLED_WINDOW_SIZE as usize,
    min_difficulty_window_size: MIN_DIFFICULTY_WINDOW_SIZE,

    deflationary_phase_daa_score: TenBps::deflationary_phase_daa_score(),
    pre_deflationary_phase_base_subsidy: TenBps::pre_deflationary_phase_base_subsidy(),
    coinbase_payload_script_public_key_max_len: 150,
    max_coinbase_payload_len: 2048,

    max_tx_inputs: 1000,
    max_tx_outputs: 1000,
    max_signature_script_len: 10_000,
    max_script_public_key_len: 10_000,

    mass_per_tx_byte: 1,
    mass_per_script_pub_key_byte: 10,
    mass_per_sig_op: 1000,
    max_block_mass: 500_000,

    storage_mass_parameter: STORAGE_MASS_PARAMETER,

    skip_proof_of_work: true, // For simnet only, PoW can be simulated by default
    max_block_level: 250,
    pruning_proof_m: PRUNING_PROOF_M,

    // For simnet, we deviate from default 10BPS configuration and allow at least 64 parents in order to support mempool benchmarks out of the box
    blockrate: BlockrateParams::new::<10>().increase_max_block_parents(64),

    pre_crescendo_target_time_per_block: TenBps::target_time_per_block(),

    crescendo_activation: ForkActivation::always(),

    model_cap_enforcement_activation: ForkActivation::always(),
    inference_reward_minimums: INFERENCE_REWARD_MINIMUMS,
    pow_salt_v2_activation: ForkActivation::never(),
    pow_salt_v4_activation: ForkActivation::never(),
    pow_salt_v5_activation: ForkActivation::never(),
    opoi_v2_activation: ForkActivation::never(),
    synthetic_liveness_activation: ForkActivation::never(),
    tier_reward_activation: ForkActivation::never(),
    balance_reward_activation: ForkActivation::never(),
};

pub const DEVNET_PARAMS: Params = Params {
    dns_seeders: &[],
    net: NetworkId::new(NetworkType::Devnet),
    genesis: DEVNET_GENESIS,
    timestamp_deviation_tolerance: TIMESTAMP_DEVIATION_TOLERANCE,
    max_difficulty_target: MAX_DIFFICULTY_TARGET,
    max_difficulty_target_f64: MAX_DIFFICULTY_TARGET_AS_F64,
    past_median_time_window_size: MEDIAN_TIME_SAMPLED_WINDOW_SIZE as usize,
    difficulty_window_size: DIFFICULTY_SAMPLED_WINDOW_SIZE as usize,
    min_difficulty_window_size: MIN_DIFFICULTY_WINDOW_SIZE,
    coinbase_payload_script_public_key_max_len: 150,
    max_coinbase_payload_len: 2048,

    max_tx_inputs: 1000,
    max_tx_outputs: 1000,
    max_signature_script_len: 10_000,
    max_script_public_key_len: 10_000,

    mass_per_tx_byte: 1,
    mass_per_script_pub_key_byte: 10,
    mass_per_sig_op: 1000,
    max_block_mass: 500_000,

    storage_mass_parameter: STORAGE_MASS_PARAMETER,

    deflationary_phase_daa_score: 0,
    pre_deflationary_phase_base_subsidy: TenBps::pre_deflationary_phase_base_subsidy(),
    skip_proof_of_work: false,
    max_block_level: 250,
    pruning_proof_m: 1000,

    blockrate: BlockrateParams::new::<10>(),

    pre_crescendo_target_time_per_block: TenBps::target_time_per_block(),

    crescendo_activation: ForkActivation::always(),

    model_cap_enforcement_activation: ForkActivation::always(),
    inference_reward_minimums: INFERENCE_REWARD_MINIMUMS,
    pow_salt_v2_activation: ForkActivation::never(),
    pow_salt_v4_activation: ForkActivation::never(),
    pow_salt_v5_activation: ForkActivation::never(),
    opoi_v2_activation: ForkActivation::never(),
    synthetic_liveness_activation: ForkActivation::never(),
    tier_reward_activation: ForkActivation::never(),
    balance_reward_activation: ForkActivation::never(),
};
