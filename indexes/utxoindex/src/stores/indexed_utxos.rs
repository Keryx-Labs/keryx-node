use crate::core::model::{CompactUtxoCollection, CompactUtxoEntry, UtxoSetByScriptPublicKey};

use keryx_consensus_core::tx::{
    ScriptPublicKey, ScriptPublicKeyVersion, ScriptPublicKeys, ScriptVec, TransactionIndexType, TransactionOutpoint,
};
use keryx_core::debug;
use keryx_database::prelude::{CachePolicy, CachedDbAccess, DB, DirectDbWriter, StoreResult};
use keryx_database::registry::DatabaseStorePrefixes;
use keryx_hashes::Hash;
use keryx_index_core::indexed_utxos::BalanceByScriptPublicKey;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt::Display;
use std::sync::Arc;

pub const VERSION_TYPE_SIZE: usize = size_of::<ScriptPublicKeyVersion>(); // Const since we need to re-use this a few times.

/// [`ScriptPublicKeyBucket`].
/// Consists of 2 bytes of little endian [VersionType] bytes, followed by the script length (8) and by a variable size of [ScriptVec].
#[derive(Eq, Hash, PartialEq, Debug, Clone)]
struct ScriptPublicKeyBucket(Vec<u8>);

impl From<&ScriptPublicKey> for ScriptPublicKeyBucket {
    fn from(script_public_key: &ScriptPublicKey) -> Self {
        // version (2) + length (8) + dynamic script
        let mut bytes: Vec<u8> = Vec::with_capacity(VERSION_TYPE_SIZE + size_of::<u64>() + script_public_key.script().len());
        bytes.extend_from_slice(&script_public_key.version().to_le_bytes());
        bytes.extend_from_slice(&(script_public_key.script().len() as u64).to_le_bytes()); // TODO: Consider using a smaller integer
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

// Keys:

// TransactionOutpoint:
/// Size of the [TransactionOutpointKey] in bytes.
pub const TRANSACTION_OUTPOINT_KEY_SIZE: usize = keryx_hashes::HASH_SIZE + size_of::<TransactionIndexType>();

/// [TransactionOutpoint] key which references the [CompactUtxoEntry] within a [ScriptPublicKeyBucket]
/// Consists of 32 bytes of [TransactionId], followed by 4 bytes of little endian [TransactionIndexType]
#[derive(Eq, Hash, PartialEq, Debug, Copy, Clone)]
struct TransactionOutpointKey([u8; TRANSACTION_OUTPOINT_KEY_SIZE]);

impl From<TransactionOutpointKey> for TransactionOutpoint {
    fn from(key: TransactionOutpointKey) -> Self {
        let transaction_id = Hash::from_slice(&key.0[..keryx_hashes::HASH_SIZE]);
        let index = TransactionIndexType::from_le_bytes(
            <[u8; size_of::<TransactionIndexType>()]>::try_from(&key.0[keryx_hashes::HASH_SIZE..]).expect("expected index size"),
        );
        Self::new(transaction_id, index)
    }
}

impl From<&TransactionOutpoint> for TransactionOutpointKey {
    fn from(outpoint: &TransactionOutpoint) -> Self {
        let mut bytes = [0; TRANSACTION_OUTPOINT_KEY_SIZE];
        bytes[..keryx_hashes::HASH_SIZE].copy_from_slice(&outpoint.transaction_id.as_bytes());
        bytes[keryx_hashes::HASH_SIZE..].copy_from_slice(&outpoint.index.to_le_bytes());
        Self(bytes)
    }
}

impl AsRef<[u8]> for TransactionOutpointKey {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

/// Full [CompactUtxoEntry] access key.
/// Consists of variable amount of bytes of [ScriptPublicKeyBucket], and 36 bytes of [TransactionOutpointKey]
#[derive(Eq, Hash, PartialEq, Debug, Clone, Serialize, Deserialize)]
struct UtxoEntryFullAccessKey(Arc<Vec<u8>>);

impl Display for UtxoEntryFullAccessKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self) // TODO: Deserialize first
    }
}

impl UtxoEntryFullAccessKey {
    /// Creates a new [UtxoEntryFullAccessKey] from a [ScriptPublicKeyBucket] and [TransactionOutpointKey].
    pub fn new(script_public_key_bucket: ScriptPublicKeyBucket, transaction_outpoint_key: TransactionOutpointKey) -> Self {
        let mut bytes = Vec::with_capacity(script_public_key_bucket.as_ref().len() + TRANSACTION_OUTPOINT_KEY_SIZE);
        bytes.extend_from_slice(script_public_key_bucket.as_ref());
        bytes.extend_from_slice(transaction_outpoint_key.as_ref());
        Self(Arc::new(bytes))
    }

    pub fn extract_outpoint(&self) -> TransactionOutpoint {
        TransactionOutpoint::from(TransactionOutpointKey(self.0[(self.0.len() - TRANSACTION_OUTPOINT_KEY_SIZE)..].try_into().unwrap()))
    }
}

impl AsRef<[u8]> for UtxoEntryFullAccessKey {
    fn as_ref(&self) -> &[u8] {
        self.0.as_slice()
    }
}

// Traits:

pub trait UtxoSetByScriptPublicKeyStoreReader {
    /// Get [UtxoSetByScriptPublicKey] set by queried [ScriptPublicKeys],
    fn get_utxos_from_script_public_keys(&self, script_public_keys: ScriptPublicKeys) -> StoreResult<UtxoSetByScriptPublicKey>;
    fn get_balance_from_script_public_keys(&self, script_public_keys: ScriptPublicKeys) -> StoreResult<BalanceByScriptPublicKey>;
    /// Get up to `limit` UTXO entries of one script public key, resuming strictly after
    /// `resume_after` when provided. Entries come in stable DB key order (outpoint bytes),
    /// so successive calls walk the bucket exactly once. Building block for chunked scans
    /// that bound how long the utxoindex lock is held per call.
    fn get_utxos_from_script_public_key_chunk(
        &self,
        script_public_key: &ScriptPublicKey,
        resume_after: Option<TransactionOutpoint>,
        limit: usize,
    ) -> StoreResult<Vec<(TransactionOutpoint, CompactUtxoEntry)>>;
    fn get_all_outpoints(&self) -> StoreResult<HashSet<TransactionOutpoint>>; // This can have a big memory footprint, so it should be used only for tests.
}

pub trait UtxoSetByScriptPublicKeyStore: UtxoSetByScriptPublicKeyStoreReader {
    /// remove [UtxoSetByScriptPublicKey] from the [UtxoSetByScriptPublicKeyStore].
    fn remove_utxo_entries(&mut self, utxo_entries: &UtxoSetByScriptPublicKey) -> StoreResult<()>;

    /// add [UtxoSetByScriptPublicKey] into the [UtxoSetByScriptPublicKeyStore].
    fn add_utxo_entries(&mut self, utxo_entries: &UtxoSetByScriptPublicKey) -> StoreResult<()>;

    /// removes all entries in the cache and db, besides prefixes themselves.
    fn delete_all(&mut self) -> StoreResult<()>;
}

// Implementations:

#[derive(Clone)]
pub struct DbUtxoSetByScriptPublicKeyStore {
    db: Arc<DB>,
    access: CachedDbAccess<UtxoEntryFullAccessKey, CompactUtxoEntry>,
}

impl DbUtxoSetByScriptPublicKeyStore {
    pub fn new(db: Arc<DB>, cache_policy: CachePolicy) -> Self {
        Self { db: Arc::clone(&db), access: CachedDbAccess::new(db, cache_policy, DatabaseStorePrefixes::UtxoIndex.into()) }
    }
}

impl UtxoSetByScriptPublicKeyStoreReader for DbUtxoSetByScriptPublicKeyStore {
    // compared to go-keryxd this gets transaction outpoints from multiple script public keys at once.
    // TODO: probably ideal way to retrieve is to return a chained iterator which can be used to chunk results and propagate utxo entries
    // to the rpc via pagination, this would alleviate the memory footprint of script public keys with large amount of utxos.
    fn get_utxos_from_script_public_keys(&self, script_public_keys: ScriptPublicKeys) -> StoreResult<UtxoSetByScriptPublicKey> {
        let script_count = script_public_keys.len();
        let mut entries_count: usize = 0;
        let mut utxos_by_script_public_keys = UtxoSetByScriptPublicKey::new();
        for script_public_key in script_public_keys.into_iter() {
            let script_public_key_bucket = ScriptPublicKeyBucket::from(&script_public_key);
            let utxos_by_script_public_keys_inner = CompactUtxoCollection::from_iter(
                self.access.seek_iterator(Some(script_public_key_bucket.as_ref()), None, usize::MAX, false).map(|res| {
                    let (key, entry) = res.unwrap();
                    (TransactionOutpointKey(<[u8; TRANSACTION_OUTPOINT_KEY_SIZE]>::try_from(&key[..]).unwrap()).into(), entry)
                }),
            );
            entries_count += utxos_by_script_public_keys_inner.len();
            utxos_by_script_public_keys.insert(script_public_key, utxos_by_script_public_keys_inner);
        }
        debug!("IDXPRC, Executed a query for the utxo set of {} script public keys yielding {} entries", script_count, entries_count);
        Ok(utxos_by_script_public_keys)
    }

    fn get_balance_from_script_public_keys(&self, script_public_keys: ScriptPublicKeys) -> StoreResult<BalanceByScriptPublicKey> {
        let script_count = script_public_keys.len();
        let mut entries_count: usize = 0;
        let mut balance_by_script_public_keys = BalanceByScriptPublicKey::new();
        for script_public_key in script_public_keys.into_iter() {
            let script_public_key_bucket = ScriptPublicKeyBucket::from(&script_public_key);
            let balance: u64 = self
                .access
                .seek_iterator(Some(script_public_key_bucket.as_ref()), None, usize::MAX, false)
                .map(|res| {
                    entries_count += 1;
                    let (_, entry) = res.unwrap();
                    entry.amount
                })
                .sum();
            balance_by_script_public_keys.insert(script_public_key, balance);
        }
        debug!("IDXPRC, Executed a query for the balance of {} script public keys involving {} entries", script_count, entries_count);
        Ok(balance_by_script_public_keys)
    }

    fn get_utxos_from_script_public_key_chunk(
        &self,
        script_public_key: &ScriptPublicKey,
        resume_after: Option<TransactionOutpoint>,
        limit: usize,
    ) -> StoreResult<Vec<(TransactionOutpoint, CompactUtxoEntry)>> {
        let script_public_key_bucket = ScriptPublicKeyBucket::from(script_public_key);
        let seek_key = resume_after
            .as_ref()
            .map(|outpoint| UtxoEntryFullAccessKey::new(script_public_key_bucket.clone(), TransactionOutpointKey::from(outpoint)));
        let skip_first = seek_key.is_some();
        Ok(self
            .access
            .seek_iterator(Some(script_public_key_bucket.as_ref()), seek_key, limit, skip_first)
            .map(|res| {
                let (key, entry) = res.unwrap();
                (TransactionOutpointKey(<[u8; TRANSACTION_OUTPOINT_KEY_SIZE]>::try_from(&key[..]).unwrap()).into(), entry)
            })
            .collect())
    }

    // This can have a big memory footprint, so it should be used only for tests.
    fn get_all_outpoints(&self) -> StoreResult<HashSet<TransactionOutpoint>> {
        Ok(HashSet::from_iter(
            self.access.iterator().map(|res| UtxoEntryFullAccessKey(Arc::new(res.unwrap().0.to_vec())).extract_outpoint()),
        ))
    }
}

impl UtxoSetByScriptPublicKeyStore for DbUtxoSetByScriptPublicKeyStore {
    fn remove_utxo_entries(&mut self, utxo_entries: &UtxoSetByScriptPublicKey) -> StoreResult<()> {
        if utxo_entries.is_empty() {
            return Ok(());
        }

        let mut writer = DirectDbWriter::new(&self.db);

        let mut to_remove = utxo_entries.iter().flat_map(move |(script_public_key, compact_utxo_collection)| {
            compact_utxo_collection.keys().map(move |transaction_outpoint| {
                UtxoEntryFullAccessKey::new(
                    ScriptPublicKeyBucket::from(script_public_key),
                    TransactionOutpointKey::from(transaction_outpoint),
                )
            })
        });

        self.access.delete_many(&mut writer, &mut to_remove)?;

        Ok(())
    }

    fn add_utxo_entries(&mut self, utxo_entries: &UtxoSetByScriptPublicKey) -> StoreResult<()> {
        if utxo_entries.is_empty() {
            return Ok(());
        }

        let mut writer = DirectDbWriter::new(&self.db);

        let mut to_add = utxo_entries.iter().flat_map(move |(script_public_key, compact_utxo_collection)| {
            compact_utxo_collection.iter().map(move |(transaction_outpoint, compact_utxo)| {
                (
                    UtxoEntryFullAccessKey::new(
                        ScriptPublicKeyBucket::from(script_public_key),
                        TransactionOutpointKey::from(transaction_outpoint),
                    ),
                    *compact_utxo,
                )
            })
        });

        self.access.write_many(&mut writer, &mut to_add)?;

        Ok(())
    }

    /// Removes all entries in the cache and db, besides prefixes themselves.
    fn delete_all(&mut self) -> StoreResult<()> {
        self.access.delete_all(DirectDbWriter::new(&self.db))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use keryx_database::create_temp_db;
    use keryx_database::prelude::ConnBuilder;

    fn spk(byte: u8) -> ScriptPublicKey {
        ScriptPublicKey::new(0, ScriptVec::from_slice(&[byte; 32]))
    }

    #[test]
    fn test_chunked_scan_matches_full_scan() {
        let (_db_lifetime, db) = create_temp_db!(ConnBuilder::default().with_files_limit(10));
        let mut store = DbUtxoSetByScriptPublicKeyStore::new(db, CachePolicy::Empty);

        // 257 entries in the target bucket (not a multiple of the walk limit), plus a
        // neighbor bucket that must never leak into the walk.
        let target = spk(1);
        let neighbor = spk(2);
        let mut target_collection = CompactUtxoCollection::new();
        for i in 0..257u32 {
            target_collection
                .insert(TransactionOutpoint::new(Hash::from_u64_word(i as u64 + 1), i), CompactUtxoEntry::new(i as u64 * 7 + 1, i as u64, i % 2 == 0));
        }
        let mut neighbor_collection = CompactUtxoCollection::new();
        neighbor_collection.insert(TransactionOutpoint::new(Hash::from_u64_word(9999), 0), CompactUtxoEntry::new(42, 0, false));
        let mut to_add = UtxoSetByScriptPublicKey::new();
        to_add.insert(target.clone(), target_collection.clone());
        to_add.insert(neighbor.clone(), neighbor_collection);
        store.add_utxo_entries(&to_add).unwrap();

        // Walk the target bucket in chunks of 100, resuming from the last outpoint.
        let mut walked = CompactUtxoCollection::new();
        let mut resume_after: Option<TransactionOutpoint> = None;
        loop {
            let chunk = store.get_utxos_from_script_public_key_chunk(&target, resume_after, 100).unwrap();
            let exhausted = chunk.len() < 100;
            resume_after = chunk.last().map(|(outpoint, _)| *outpoint);
            for (outpoint, entry) in chunk {
                assert!(walked.insert(outpoint, entry).is_none(), "chunked walk revisited an outpoint");
            }
            if exhausted {
                break;
            }
        }

        assert_eq!(walked.len(), target_collection.len(), "chunked walk must cover the whole bucket exactly once");
        for (outpoint, entry) in target_collection.iter() {
            assert_eq!(walked.get(outpoint).expect("chunked walk missed an outpoint").amount, entry.amount);
        }

        // A limit larger than the bucket returns everything in one call.
        assert_eq!(store.get_utxos_from_script_public_key_chunk(&target, None, usize::MAX).unwrap().len(), target_collection.len());
        // Resuming after the last outpoint returns an empty chunk.
        let full = store.get_utxos_from_script_public_key_chunk(&target, None, usize::MAX).unwrap();
        let last = full.last().map(|(outpoint, _)| *outpoint);
        assert!(store.get_utxos_from_script_public_key_chunk(&target, last, 100).unwrap().is_empty());
    }
}
