/// RocksDB store of finality-deep inference-reward wins. Append-only, written only for
/// reorg-immune events in chain order, keyed by request hash; read at boot to rebuild the
/// commitment index and the rewarded-request dedup set.
use std::{fmt, sync::Arc};

use keryx_consensus_core::collateral::RewardEntry;
use keryx_database::prelude::{CachePolicy, CachedDbAccess, DB, DirectDbWriter, StoreError};
use keryx_database::registry::DatabaseStorePrefixes;

#[derive(Eq, Hash, PartialEq, Debug, Copy, Clone)]
pub struct RewardKey(pub [u8; 32]);

impl AsRef<[u8]> for RewardKey {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl fmt::Display for RewardKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", hex::encode(self.0))
    }
}

#[derive(Clone)]
pub struct DbServiceRewardStore {
    db: Arc<DB>,
    access: CachedDbAccess<RewardKey, RewardEntry>,
}

impl DbServiceRewardStore {
    pub fn new(db: Arc<DB>, cache_policy: CachePolicy) -> Self {
        Self { db: Arc::clone(&db), access: CachedDbAccess::new(db, cache_policy, DatabaseStorePrefixes::ServiceReward.into()) }
    }

    pub fn set(&self, key: RewardKey, entry: RewardEntry) -> Result<(), StoreError> {
        self.access.write(DirectDbWriter::new(&self.db), key, entry)
    }

    /// All reward rows, for the boot load.
    pub fn iterator(&self) -> impl Iterator<Item = Result<(Box<[u8]>, RewardEntry), Box<dyn std::error::Error>>> + '_ {
        self.access.iterator()
    }
}
