use keryx_hashes::{Hash, Hasher, TransactionHash};
use keryx_utils::mem_size::MemSizeEstimator;
use serde::{Deserialize, Serialize};

use crate::tx::ScriptPublicKey;

/// Fraction of each accepted block subsidy auto-accumulated as miner collateral (basis points).
/// 1 000 BPS = 10 %.
pub const COLLATERAL_RATE_BPS: u64 = 1_000;

/// Number of blocks during which an OPoI result may be challenged after its block is accepted.
pub const CHALLENGE_WINDOW_BLOCKS: u64 = 100;

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
