/// Block version used before the H4 relaunch boundary.
pub const BLOCK_VERSION: u16 = 1;

/// Block version required by the H4 relaunch and all later blocks.
pub const H4_RELAUNCH_BLOCK_VERSION: u16 = BLOCK_VERSION + 1;

#[inline]
pub const fn block_version_for_h4_relaunch(daa_score: u64, h4_relaunch_activation_daa: u64) -> u16 {
    if daa_score >= h4_relaunch_activation_daa { H4_RELAUNCH_BLOCK_VERSION } else { BLOCK_VERSION }
}

/// TX_VERSION is the current latest supported transaction version.
pub const TX_VERSION: u16 = 0;

pub const LOCK_TIME_THRESHOLD: u64 = 500_000_000_000;

/// MAX_SCRIPT_PUBLIC_KEY_VERSION is the current latest supported public key script version.
pub const MAX_SCRIPT_PUBLIC_KEY_VERSION: u16 = 0;

/// SompiPerKaspa is the number of sompi in one keryx (1 KRX).
pub const SOMPI_PER_KASPA: u64 = 100_000_000;

/// The parameter for scaling inverse KRX value to mass units (KIP-0009)
pub const STORAGE_MASS_PARAMETER: u64 = SOMPI_PER_KASPA * 10_000;

/// The parameter defining how much mass per byte to charge for when calculating
/// transient storage mass. Since normally the block mass limit is 500_000, this limits
/// block body byte size to 125_000 (KIP-0013).
pub const TRANSIENT_BYTE_TO_MASS_FACTOR: u64 = 4;

/// MaxSompi is the maximum transaction amount allowed in sompi.
pub const MAX_SOMPI: u64 = 29_000_000_000 * SOMPI_PER_KASPA;

// MAX_TX_IN_SEQUENCE_NUM is the maximum sequence number the sequence field
// of a transaction input can be.
pub const MAX_TX_IN_SEQUENCE_NUM: u64 = u64::MAX;

// SEQUENCE_LOCK_TIME_MASK is a mask that extracts the relative lock time
// when masked against the transaction input sequence number.
pub const SEQUENCE_LOCK_TIME_MASK: u64 = 0x00000000ffffffff;

// SEQUENCE_LOCK_TIME_DISABLED is a flag that if set on a transaction
// input's sequence number, the sequence number will not be interpreted
// as a relative lock time.
pub const SEQUENCE_LOCK_TIME_DISABLED: u64 = 1 << 63;

/// UNACCEPTED_DAA_SCORE is used to for UtxoEntries that were created by
/// transactions in the mempool, or otherwise not-yet-accepted transactions.
pub const UNACCEPTED_DAA_SCORE: u64 = u64::MAX;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn h4_relaunch_version_switches_at_the_exact_boundary() {
        const H4_DAA: u64 = 54_766_000;

        assert_eq!(block_version_for_h4_relaunch(H4_DAA - 1, H4_DAA), BLOCK_VERSION);
        assert_eq!(block_version_for_h4_relaunch(H4_DAA, H4_DAA), H4_RELAUNCH_BLOCK_VERSION);
    }
}
