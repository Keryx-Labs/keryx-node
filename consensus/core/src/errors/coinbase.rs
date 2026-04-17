use thiserror::Error;

#[derive(Error, Debug, Clone)]
pub enum CoinbaseError {
    #[error("coinbase payload length is {0} while the minimum allowed length is {1}")]
    PayloadLenBelowMin(usize, usize),

    #[error("coinbase payload length is {0} while the maximum allowed length is {1}")]
    PayloadLenAboveMax(usize, usize),

    #[error("coinbase payload script public key length is {0} while the maximum allowed length is {1}")]
    PayloadScriptPublicKeyLenAboveMax(usize, u8),

    #[error(
        "coinbase payload length is {0} bytes but it needs to be at least {1} bytes long in order to accommodate the script public key"
    )]
    PayloadCantContainScriptPublicKey(usize, usize),

    /// OPoI Phase 2: the miner's claimed inference tag does not match the
    /// expected result for the nonce declared in extra_data.
    #[error("OPoI tag mismatch: nonce {0:#018x} — claimed tag '{1}' does not match expected value")]
    OPoiTagInvalid(u64, String),
}

pub type CoinbaseResult<T> = std::result::Result<T, CoinbaseError>;
