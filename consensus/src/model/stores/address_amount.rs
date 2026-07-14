use std::fmt::Display;
use std::sync::Arc;

use keryx_consensus_core::tx::{ScriptPublicKey, ScriptPublicKeyVersion, ScriptVec};
use keryx_database::prelude::CachePolicy;
use keryx_database::prelude::DB;
use keryx_database::prelude::StoreError;
use keryx_database::prelude::{BatchDbWriter, CachedDbAccess, StoreResultExt};
use rocksdb::WriteBatch;

/// Size (bytes) of the script-public-key version prefix of a [`ScriptPublicKeyBucket`].
const VERSION_TYPE_SIZE: usize = size_of::<ScriptPublicKeyVersion>();

/// DB key wrapping a [`ScriptPublicKey`]: 2 bytes LE version, 8 bytes LE script length, then the
/// raw script. Same on-disk layout as the (private) bucket of `indexes/utxoindex`, so a per-SPK
/// aggregate keyed here lines up byte-for-byte with the utxoindex view it mirrors.
#[derive(Eq, Hash, PartialEq, Debug, Clone)]
pub struct ScriptPublicKeyBucket(Vec<u8>);

impl ScriptPublicKeyBucket {
    /// Rewraps raw key bytes read back from the DB (iterator paths of stores sharing this key
    /// encoding, e.g. the coin-age `age_buckets` index test helper).
    #[cfg(test)]
    pub(crate) fn from_bytes(bytes: Vec<u8>) -> Self {
        Self(bytes)
    }
}

impl From<&ScriptPublicKey> for ScriptPublicKeyBucket {
    fn from(script_public_key: &ScriptPublicKey) -> Self {
        let mut bytes: Vec<u8> = Vec::with_capacity(VERSION_TYPE_SIZE + size_of::<u64>() + script_public_key.script().len());
        bytes.extend_from_slice(&script_public_key.version().to_le_bytes());
        bytes.extend_from_slice(&(script_public_key.script().len() as u64).to_le_bytes());
        bytes.extend_from_slice(script_public_key.script());
        Self(bytes)
    }
}

impl From<ScriptPublicKeyBucket> for ScriptPublicKey {
    fn from(bucket: ScriptPublicKeyBucket) -> Self {
        let version = ScriptPublicKeyVersion::from_le_bytes(
            <[u8; VERSION_TYPE_SIZE]>::try_from(&bucket.0[..VERSION_TYPE_SIZE]).expect("expected version size"),
        );
        let script_size =
            u64::from_le_bytes(bucket.0[VERSION_TYPE_SIZE..VERSION_TYPE_SIZE + size_of::<u64>()].try_into().unwrap()) as usize;
        let script =
            ScriptVec::from_slice(&bucket.0[VERSION_TYPE_SIZE + size_of::<u64>()..VERSION_TYPE_SIZE + size_of::<u64>() + script_size]);
        Self::new(version, script)
    }
}

impl AsRef<[u8]> for ScriptPublicKeyBucket {
    fn as_ref(&self) -> &[u8] {
        self.0.as_slice()
    }
}

// `CachedDbAccess::read` requires the key to be `ToString` (for error reporting). The utxoindex
// bucket gets this for free via its module; here it is an explicit hex dump of the key bytes.
impl Display for ScriptPublicKeyBucket {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for b in &self.0 {
            write!(f, "{b:02x}")?;
        }
        Ok(())
    }
}

/// Read access to a consensus-maintained per-address `u64` aggregate keyed by payout script-public-key.
///
/// Backs the two ratio-reward (Stage 2) indexes, both of which are exactly this shape — a single
/// `SPK → amount` map kept in lockstep with the UTXO set (wound/unwound by the same diffs on reorg):
/// - **balance index** (`Σ unspent amount`), elevating the non-consensus `--utxoindex` balance view
///   into a deterministic consensus structure;
/// - **windowed-production index** (`Σ coinbase miner-cut over the trailing window W`).
///
/// A missing key reads as `0` (an address with no aggregate); the store deletes entries that fall
/// back to `0` so the keyspace tracks only currently-relevant addresses.
pub trait AddressAmountStoreReader {
    fn get(&self, script_public_key: &ScriptPublicKey) -> Result<u64, StoreError>;
    fn has(&self, script_public_key: &ScriptPublicKey) -> Result<bool, StoreError>;
}

/// A DB + cache implementation of `AddressAmountStoreReader`, with concurrency support.
#[derive(Clone)]
pub struct DbAddressAmountStore {
    access: CachedDbAccess<ScriptPublicKeyBucket, u64>,
}

impl DbAddressAmountStore {
    /// Builds the store under the given prefix. The same type backs the balance and the windowed
    /// production indexes; they differ only by the `prefix` passed in (see `DatabaseStorePrefixes`).
    pub fn new(db: Arc<DB>, cache_policy: CachePolicy, prefix: Vec<u8>) -> Self {
        Self { access: CachedDbAccess::new(db, cache_policy, prefix) }
    }

    /// Sets the aggregate for `script_public_key`. Writing `0` deletes the entry (a zero aggregate
    /// is indistinguishable from absence on read), keeping the keyspace to non-empty addresses only.
    /// The caller computes the new value (read-modify-write is done per address per batch in the
    /// virtual processor, where all deltas of a single applied diff are folded first).
    pub fn set_batch(&self, batch: &mut WriteBatch, script_public_key: &ScriptPublicKey, amount: u64) -> Result<(), StoreError> {
        let key = ScriptPublicKeyBucket::from(script_public_key);
        if amount == 0 {
            self.access.delete(BatchDbWriter::new(batch), key)
        } else {
            self.access.write(BatchDbWriter::new(batch), key, amount)
        }
    }

    /// Clears the whole index (this store's prefix only). Used at fast-sync import to drop any stale
    /// aggregate before re-seeding from the imported pruning-point UTXO snapshot.
    pub fn clear(&self, batch: &mut WriteBatch) -> Result<(), StoreError> {
        self.access.delete_all(BatchDbWriter::new(batch))
    }

    /// Test helper: collects the whole index (this store's prefix only) into a map, reversing the
    /// `ScriptPublicKeyBucket` key encoding back to the originating `ScriptPublicKey`. Used by the
    /// ratio-reward reconstruction-equality test to assert that the incrementally maintained index
    /// matches, key-for-key, a fresh seed grouped from the UTXO snapshot.
    #[cfg(test)]
    pub fn collect(&self) -> std::collections::HashMap<ScriptPublicKey, u64> {
        self.access
            .iterator()
            .map(|res| {
                let (key, amount) = res.unwrap();
                (ScriptPublicKey::from(ScriptPublicKeyBucket(key.to_vec())), amount)
            })
            .collect()
    }
}

impl AddressAmountStoreReader for DbAddressAmountStore {
    fn get(&self, script_public_key: &ScriptPublicKey) -> Result<u64, StoreError> {
        // Absent key ⇒ 0 (no aggregate held for this address).
        Ok(self.access.read(ScriptPublicKeyBucket::from(script_public_key)).optional()?.unwrap_or(0))
    }

    fn has(&self, script_public_key: &ScriptPublicKey) -> Result<bool, StoreError> {
        self.access.has(ScriptPublicKeyBucket::from(script_public_key))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use keryx_consensus_core::tx::ScriptPublicKey;
    use keryx_database::{create_temp_db, prelude::ConnBuilder};

    fn spk(byte: u8) -> ScriptPublicKey {
        ScriptPublicKey::new(0, ScriptVec::from_slice(&[byte, byte, byte]))
    }

    #[test]
    fn address_amount_round_trip() {
        let (_lifetime, db) = create_temp_db!(ConnBuilder::default().with_files_limit(10));
        let store = DbAddressAmountStore::new(db.clone(), CachePolicy::Count(16), vec![39]);
        let (a, b) = (spk(1), spk(2));

        // Absent keys read as 0 and `has` false.
        assert_eq!(store.get(&a).unwrap(), 0);
        assert!(!store.has(&a).unwrap());

        // Write two aggregates in a batch.
        let mut batch = WriteBatch::default();
        store.set_batch(&mut batch, &a, 1_000).unwrap();
        store.set_batch(&mut batch, &b, 42).unwrap();
        db.write(batch).unwrap();

        assert_eq!(store.get(&a).unwrap(), 1_000);
        assert_eq!(store.get(&b).unwrap(), 42);
        assert!(store.has(&a).unwrap());

        // Overwrite (the index is mutable, unlike the append-only block→bps stores).
        let mut batch = WriteBatch::default();
        store.set_batch(&mut batch, &a, 1_500).unwrap();
        db.write(batch).unwrap();
        assert_eq!(store.get(&a).unwrap(), 1_500);

        // Setting 0 deletes the entry ⇒ back to absent/0.
        let mut batch = WriteBatch::default();
        store.set_batch(&mut batch, &a, 0).unwrap();
        db.write(batch).unwrap();
        assert_eq!(store.get(&a).unwrap(), 0);
        assert!(!store.has(&a).unwrap());

        // The SPK bucket round-trips back to the original script-public-key.
        let bucket = ScriptPublicKeyBucket::from(&b);
        assert_eq!(ScriptPublicKey::from(bucket), b);
    }
}
