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
