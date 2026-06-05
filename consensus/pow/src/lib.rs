// public for benchmarks
#[doc(hidden)]
pub mod matrix;
#[cfg(feature = "wasm32-sdk")]
pub mod wasm;
#[doc(hidden)]
pub mod xoshiro;

use std::cmp::max;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::matrix::Matrix;
use keryx_consensus_core::{BlockLevel, hashing, header::Header};
use keryx_hashes::PowHash;
use keryx_math::Uint256;

/// DAA score at which the PoW SALT switches from v1 to v2.
/// u64::MAX means "never" (default — disabled until explicitly initialised from params).
/// Set once at node/miner startup via [`init_pow_salt_v2_activation`].
static POW_SALT_V2_ACTIVATION_DAA: AtomicU64 = AtomicU64::new(u64::MAX);

/// DAA score at which the PoW SALT switches from v2 to v3.
/// u64::MAX means "never" (default). Set once at startup via [`init_pow_salt_v3_activation`].
static POW_SALT_V3_ACTIVATION_DAA: AtomicU64 = AtomicU64::new(u64::MAX);

/// Called once at startup with the value from `Params::pow_salt_v2_activation`.
/// After this point every PoW computation automatically picks the correct salt.
pub fn init_pow_salt_v2_activation(daa_score: u64) {
    POW_SALT_V2_ACTIVATION_DAA.store(daa_score, Ordering::Relaxed);
}

/// Called once at startup with the value from `Params::pow_salt_v3_activation`.
pub fn init_pow_salt_v3_activation(daa_score: u64) {
    POW_SALT_V3_ACTIVATION_DAA.store(daa_score, Ordering::Relaxed);
}

/// Returns the active matrix-salt version (1, 2 or 3) for a block at `daa_score`.
/// Thresholds are monotonic and compared with `>=`, so the first block whose
/// `daa_score` reaches an activation already uses the new salt.
#[inline(always)]
pub(crate) fn active_salt_version(daa_score: u64) -> u8 {
    if daa_score >= POW_SALT_V3_ACTIVATION_DAA.load(Ordering::Relaxed) {
        3
    } else if daa_score >= POW_SALT_V2_ACTIVATION_DAA.load(Ordering::Relaxed) {
        2
    } else {
        1
    }
}

/// State is an intermediate data structure with pre-computed values to speed up mining.
pub struct State {
    pub(crate) matrix: Matrix,
    pub(crate) target: Uint256,
    // PRE_POW_HASH || TIME || 32 zero byte padding; without NONCE
    pub(crate) hasher: PowHash,
}

impl State {
    #[inline]
    pub fn new(header: &Header) -> Self {
        let target = Uint256::from_compact_target_bits(header.bits);
        // Zero out the time and nonce.
        let pre_pow_hash = hashing::header::hash_override_nonce_time(header, 0, 0);
        // PRE_POW_HASH || TIME || 32 zero byte padding || NONCE
        let hasher = PowHash::new(pre_pow_hash, header.timestamp);
        let matrix = Matrix::generate(pre_pow_hash, active_salt_version(header.daa_score));

        Self { matrix, target, hasher }
    }

    #[inline]
    #[must_use]
    /// PRE_POW_HASH || TIME || 32 zero byte padding || NONCE
    pub fn calculate_pow(&self, nonce: u64) -> Uint256 {
        // Hasher already contains PRE_POW_HASH || TIME || 32 zero byte padding; so only the NONCE is missing
        let hash = self.hasher.clone().finalize_with_nonce(nonce);
        let hash = self.matrix.keryx_hash(hash);
        Uint256::from_le_bytes(hash.as_bytes())
    }

    #[inline]
    #[must_use]
    pub fn check_pow(&self, nonce: u64) -> (bool, Uint256) {
        let pow = self.calculate_pow(nonce);
        // The pow hash must be less or equal than the claimed target.
        (pow <= self.target, pow)
    }
}

pub fn calc_block_level(header: &Header, max_block_level: BlockLevel) -> BlockLevel {
    let (block_level, _) = calc_block_level_check_pow(header, max_block_level);
    block_level
}

pub fn calc_block_level_check_pow(header: &Header, max_block_level: BlockLevel) -> (BlockLevel, bool) {
    if header.parents_by_level.is_empty() {
        return (max_block_level, true); // Genesis has the max block level
    }

    let state = State::new(header);
    let (passed, pow) = state.check_pow(header.nonce);
    let block_level = calc_level_from_pow(pow, max_block_level);
    (block_level, passed)
}

pub fn calc_level_from_pow(pow: Uint256, max_block_level: BlockLevel) -> BlockLevel {
    let signed_block_level = max_block_level as i64 - pow.bits() as i64;
    max(signed_block_level, 0) as BlockLevel
}
