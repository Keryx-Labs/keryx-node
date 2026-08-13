/// RocksDB store of the service-bond strike log. Append-only, written only for finality-deep
/// events (reorg-immune) in chain order, so writes are monotone and idempotent and never rolled
/// back. Keys order by event daa, making the log the canonical event sequence: the fold baseline
/// is the last record per miner, the persisted frontier is the highest key daa, and a row
/// `{count: 0, last_daa > 0}` is an executed suspension whose deadline re-derives as
/// `last_daa + finality + SERVICE_SUSPENSION_DAA`.
use std::{fmt, sync::Arc};

use keryx_consensus_core::collateral::StrikeEntry;
use keryx_database::prelude::{CachedDbAccess, CachePolicy, DirectDbWriter, StoreError, DB};
use keryx_database::registry::DatabaseStorePrefixes;
use keryx_hashes::Hash;

/// `event daa (8 bytes BE) || miner identity (32 bytes)` — big-endian daa first, so RocksDB
/// iteration order is event order.
#[derive(Eq, Hash, PartialEq, Debug, Copy, Clone)]
pub struct StrikeLogKey([u8; 40]);

impl StrikeLogKey {
    pub fn new(daa: u64, miner: Hash) -> Self {
        let mut bytes = [0u8; 40];
        bytes[..8].copy_from_slice(&daa.to_be_bytes());
        bytes[8..].copy_from_slice(&miner.as_bytes());
        Self(bytes)
    }

    pub fn parse(key: &[u8]) -> (u64, Hash) {
        let daa = u64::from_be_bytes(key[..8].try_into().unwrap());
        let miner: [u8; 32] = key[8..40].try_into().unwrap();
        (daa, Hash::from_bytes(miner))
    }
}

impl AsRef<[u8]> for StrikeLogKey {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl fmt::Display for StrikeLogKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", hex::encode(self.0))
    }
}

#[derive(Clone)]
pub struct DbServiceStrikeStore {
    db: Arc<DB>,
    access: CachedDbAccess<StrikeLogKey, StrikeEntry>,
}

impl DbServiceStrikeStore {
    pub fn new(db: Arc<DB>, cache_policy: CachePolicy) -> Self {
        Self { db: Arc::clone(&db), access: CachedDbAccess::new(db, cache_policy, DatabaseStorePrefixes::ServiceStrike.into()) }
    }

    pub fn set(&self, daa: u64, miner: Hash, record: StrikeEntry) -> Result<(), StoreError> {
        self.access.write(DirectDbWriter::new(&self.db), StrikeLogKey::new(daa, miner), record)
    }

    /// The whole log in event order, for the boot load and the refold baseline.
    pub fn iterator(&self) -> impl Iterator<Item = Result<(Box<[u8]>, StrikeEntry), Box<dyn std::error::Error>>> + '_ {
        self.access.iterator()
    }
}
