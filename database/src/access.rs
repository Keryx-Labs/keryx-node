use crate::{cache::CachePolicy, db::DB, errors::StoreError};

use super::prelude::{Cache, DbKey, DbWriter};
use keryx_utils::mem_size::MemSizeEstimator;
use rocksdb::{Direction, IterateBounds, IteratorMode, ReadOptions};
use serde::{Serialize, de::DeserializeOwned};
use std::{collections::hash_map::RandomState, error::Error, hash::BuildHasher, sync::Arc};

/// A concurrent DB store access with typed caching.
#[derive(Clone)]
pub struct CachedDbAccess<TKey, TData, S = RandomState>
where
    TKey: Clone + std::hash::Hash + Eq + Send + Sync,
    TData: Clone + Send + Sync + MemSizeEstimator,
{
    db: Arc<DB>,

    // Cache
    cache: Cache<TKey, TData, S>,

    // DB bucket/path
    prefix: Vec<u8>,
}

pub type KeyDataResult<TData> = Result<(Box<[u8]>, TData), Box<dyn Error>>;

impl<TKey, TData, S> CachedDbAccess<TKey, TData, S>
where
    TKey: Clone + std::hash::Hash + Eq + Send + Sync,
    TData: Clone + Send + Sync + MemSizeEstimator,
    S: BuildHasher + Default,
{
    pub fn new(db: Arc<DB>, cache_policy: CachePolicy, prefix: Vec<u8>) -> Self {
        Self { db, cache: Cache::new(cache_policy), prefix }
    }

    pub fn read_from_cache(&self, key: &TKey) -> Option<TData> {
        self.cache.get(key)
    }

    /// Batch read: serve hits from the app cache, then one RocksDB `multi_get` for the misses.
    /// Returns one `Option` per input key (`None` = absent). Also returns `(hits, misses)` counts
    /// so callers can feed `ProcessingCounters` without a second cache probe.
    pub fn read_many(&self, keys: &[TKey]) -> Result<(Vec<Option<TData>>, u64, u64), StoreError>
    where
        TKey: Clone + AsRef<[u8]> + ToString,
        TData: DeserializeOwned,
    {
        let mut results: Vec<Option<TData>> = Vec::with_capacity(keys.len());
        let mut miss_indices: Vec<usize> = Vec::new();
        let mut miss_db_keys: Vec<DbKey> = Vec::new();
        let mut hits = 0u64;

        for (i, key) in keys.iter().enumerate() {
            if let Some(data) = self.cache.get(key) {
                results.push(Some(data));
                hits += 1;
            } else {
                results.push(None);
                miss_indices.push(i);
                miss_db_keys.push(DbKey::new(&self.prefix, key.clone()));
            }
        }

        let misses = miss_indices.len() as u64;
        if !miss_db_keys.is_empty() {
            let db_results = self.db.multi_get(miss_db_keys.iter());
            for (slot, db_result) in miss_indices.into_iter().zip(db_results) {
                match db_result {
                    Ok(Some(slice)) => {
                        let data: TData = bincode::deserialize(&slice)?;
                        self.cache.insert(keys[slot].clone(), data.clone());
                        results[slot] = Some(data);
                    }
                    Ok(None) => {}
                    Err(e) => return Err(e.into()),
                }
            }
        }

        Ok((results, hits, misses))
    }

    pub fn has(&self, key: TKey) -> Result<bool, StoreError>
    where
        TKey: Clone + AsRef<[u8]>,
    {
        Ok(self.cache.contains_key(&key) || self.db.get_pinned(DbKey::new(&self.prefix, key))?.is_some())
    }

    pub fn read(&self, key: TKey) -> Result<TData, StoreError>
    where
        TKey: Clone + AsRef<[u8]> + ToString,
        TData: DeserializeOwned, // We need `DeserializeOwned` since the slice coming from `db.get_pinned` has short lifetime
    {
        if let Some(data) = self.cache.get(&key) {
            Ok(data)
        } else {
            let db_key = DbKey::new(&self.prefix, key.clone());
            if let Some(slice) = self.db.get_pinned(&db_key)? {
                let data: TData = bincode::deserialize(&slice)?;
                self.cache.insert(key, data.clone());
                Ok(data)
            } else {
                Err(StoreError::KeyNotFound(db_key))
            }
        }
    }

    pub fn has_with_fallback(&self, fallback_prefix: &[u8], key: TKey) -> Result<bool, StoreError>
    where
        TKey: Clone + AsRef<[u8]>,
    {
        if self.cache.contains_key(&key) {
            Ok(true)
        } else {
            let db_key = DbKey::new(&self.prefix, key.clone());
            if self.db.get_pinned(&db_key)?.is_some() {
                Ok(true)
            } else {
                let db_key = DbKey::new(fallback_prefix, key.clone());
                Ok(self.db.get_pinned(&db_key)?.is_some())
            }
        }
    }

    pub fn read_with_fallback<TFallbackDeser>(&self, fallback_prefix: &[u8], key: TKey) -> Result<TData, StoreError>
    where
        TKey: Clone + AsRef<[u8]> + ToString,
        TData: DeserializeOwned,
        TFallbackDeser: DeserializeOwned + Into<TData>,
    {
        self.read_with_fallbacks::<TData, TFallbackDeser>(fallback_prefix, key)
    }

    /// Like `read_with_fallback`, with an additional in-place decode fallback: if the main-column
    /// bytes fail to decode as `TData`, the SAME bytes are re-tried as `TDecodeFallback` (a
    /// previous layout of `TData`) and converted. Needed when a persisted struct gains a trailing
    /// field: entries written by an older node keep the old layout in the same column, and
    /// bincode (positional, no defaults) fails on them with a length error rather than falling
    /// back. Safe both ways: old bytes decoded as the grown `TData` always underflow (error, never
    /// silently mis-decode), and `TDecodeFallback` is only attempted after `TData` failed.
    pub fn read_with_fallbacks<TDecodeFallback, TPrefixFallback>(
        &self,
        fallback_prefix: &[u8],
        key: TKey,
    ) -> Result<TData, StoreError>
    where
        TKey: Clone + AsRef<[u8]> + ToString,
        TData: DeserializeOwned,
        TDecodeFallback: DeserializeOwned + Into<TData>,
        TPrefixFallback: DeserializeOwned + Into<TData>,
    {
        if let Some(data) = self.cache.get(&key) {
            Ok(data)
        } else {
            let db_key = DbKey::new(&self.prefix, key.clone());
            if let Some(slice) = self.db.get_pinned(&db_key)? {
                let data: TData = match bincode::deserialize(&slice) {
                    Ok(data) => data,
                    Err(_) => bincode::deserialize::<TDecodeFallback>(&slice)?.into(),
                };
                self.cache.insert(key, data.clone());
                Ok(data)
            } else {
                let db_key = DbKey::new(fallback_prefix, key.clone());
                if let Some(slice) = self.db.get_pinned(&db_key)? {
                    let data: TPrefixFallback = bincode::deserialize(&slice)?;
                    let data: TData = data.into();
                    self.cache.insert(key, data.clone());
                    Ok(data)
                } else {
                    Err(StoreError::KeyNotFound(db_key))
                }
            }
        }
    }

    /// Like `read_with_fallbacks`, with TWO in-place decode fallbacks tried in order (newest
    /// layout first). Needed once a persisted struct has grown trailing fields twice: three
    /// layouts coexist in the same column. Same safety argument — shorter (older) layouts always
    /// underflow when read as a longer one, so each fallback is only reached on a genuine error.
    pub fn read_with_fallbacks3<TDecodeFallback1, TDecodeFallback2, TPrefixFallback>(
        &self,
        fallback_prefix: &[u8],
        key: TKey,
    ) -> Result<TData, StoreError>
    where
        TKey: Clone + AsRef<[u8]> + ToString,
        TData: DeserializeOwned,
        TDecodeFallback1: DeserializeOwned + Into<TData>,
        TDecodeFallback2: DeserializeOwned + Into<TData>,
        TPrefixFallback: DeserializeOwned + Into<TData>,
    {
        if let Some(data) = self.cache.get(&key) {
            Ok(data)
        } else {
            let db_key = DbKey::new(&self.prefix, key.clone());
            if let Some(slice) = self.db.get_pinned(&db_key)? {
                let data: TData = match bincode::deserialize(&slice) {
                    Ok(data) => data,
                    Err(_) => match bincode::deserialize::<TDecodeFallback1>(&slice) {
                        Ok(data) => data.into(),
                        Err(_) => bincode::deserialize::<TDecodeFallback2>(&slice)?.into(),
                    },
                };
                self.cache.insert(key, data.clone());
                Ok(data)
            } else {
                let db_key = DbKey::new(fallback_prefix, key.clone());
                if let Some(slice) = self.db.get_pinned(&db_key)? {
                    let data: TPrefixFallback = bincode::deserialize(&slice)?;
                    let data: TData = data.into();
                    self.cache.insert(key, data.clone());
                    Ok(data)
                } else {
                    Err(StoreError::KeyNotFound(db_key))
                }
            }
        }
    }

    /// Like `read`, with an in-place decode fallback (no prefix fallback): if the bytes fail to
    /// decode as `TData`, the SAME bytes are re-tried as `TDecodeFallback` (a previous layout of
    /// `TData`) and converted. See `read_with_fallbacks` for the safety argument (a grown
    /// positional layout under-flows on old bytes — hard error, never a silent mis-decode).
    pub fn read_with_decode_fallback<TDecodeFallback>(&self, key: TKey) -> Result<TData, StoreError>
    where
        TKey: Clone + AsRef<[u8]> + ToString,
        TData: DeserializeOwned,
        TDecodeFallback: DeserializeOwned + Into<TData>,
    {
        if let Some(data) = self.cache.get(&key) {
            Ok(data)
        } else {
            let db_key = DbKey::new(&self.prefix, key.clone());
            if let Some(slice) = self.db.get_pinned(&db_key)? {
                let data: TData = match bincode::deserialize(&slice) {
                    Ok(data) => data,
                    Err(_) => bincode::deserialize::<TDecodeFallback>(&slice)?.into(),
                };
                self.cache.insert(key, data.clone());
                Ok(data)
            } else {
                Err(StoreError::KeyNotFound(db_key))
            }
        }
    }

    /// Like `iterator`, with the same in-place decode fallback as `read_with_decode_fallback` —
    /// needed by stores whose whole keyspace may hold mixed layouts (e.g. a utxoset written by an
    /// older binary and appended to by this one) and which are scanned wholesale at startup.
    pub fn iterator_with_decode_fallback<TDecodeFallback>(&self) -> impl Iterator<Item = KeyDataResult<TData>> + '_
    where
        TKey: Clone + AsRef<[u8]>,
        TData: DeserializeOwned,
        TDecodeFallback: DeserializeOwned + Into<TData>,
    {
        let prefix_key = DbKey::prefix_only(&self.prefix);
        let mut read_opts = ReadOptions::default();
        read_opts.set_iterate_range(rocksdb::PrefixRange(prefix_key.as_ref()));
        self.db.iterator_opt(IteratorMode::From(prefix_key.as_ref(), Direction::Forward), read_opts).map(move |iter_result| {
            match iter_result {
                Ok((key, data_bytes)) => match bincode::deserialize(&data_bytes) {
                    Ok(data) => Ok((key[prefix_key.prefix_len()..].into(), data)),
                    Err(e) => match bincode::deserialize::<TDecodeFallback>(&data_bytes) {
                        Ok(data) => Ok((key[prefix_key.prefix_len()..].into(), data.into())),
                        Err(_) => Err(e.into()),
                    },
                },
                Err(e) => Err(e.into()),
            }
        })
    }

    pub fn iterator(&self) -> impl Iterator<Item = KeyDataResult<TData>> + '_
    where
        TKey: Clone + AsRef<[u8]>,
        TData: DeserializeOwned, // We need `DeserializeOwned` since the slice coming from `db.get_pinned` has short lifetime
    {
        let prefix_key = DbKey::prefix_only(&self.prefix);
        let mut read_opts = ReadOptions::default();
        read_opts.set_iterate_range(rocksdb::PrefixRange(prefix_key.as_ref()));
        self.db.iterator_opt(IteratorMode::From(prefix_key.as_ref(), Direction::Forward), read_opts).map(move |iter_result| {
            match iter_result {
                Ok((key, data_bytes)) => match bincode::deserialize(&data_bytes) {
                    Ok(data) => Ok((key[prefix_key.prefix_len()..].into(), data)),
                    Err(e) => Err(e.into()),
                },
                Err(e) => Err(e.into()),
            }
        })
    }

    pub fn write(&self, mut writer: impl DbWriter, key: TKey, data: TData) -> Result<(), StoreError>
    where
        TKey: Clone + AsRef<[u8]>,
        TData: Serialize,
    {
        let bin_data = bincode::serialize(&data)?;
        self.cache.insert(key.clone(), data);
        writer.put(DbKey::new(&self.prefix, key), bin_data)?;
        Ok(())
    }

    pub fn write_many(
        &self,
        mut writer: impl DbWriter,
        iter: &mut (impl Iterator<Item = (TKey, TData)> + Clone),
    ) -> Result<(), StoreError>
    where
        TKey: Clone + AsRef<[u8]>,
        TData: Serialize,
    {
        let iter_clone = iter.clone();
        self.cache.insert_many(iter);
        for (key, data) in iter_clone {
            let bin_data = bincode::serialize(&data)?;
            writer.put(DbKey::new(&self.prefix, key.clone()), bin_data)?;
        }
        Ok(())
    }

    /// Write directly from an iterator and do not cache any data. NOTE: this action also clears the cache
    pub fn write_many_without_cache(
        &self,
        mut writer: impl DbWriter,
        iter: &mut impl Iterator<Item = (TKey, TData)>,
    ) -> Result<(), StoreError>
    where
        TKey: Clone + AsRef<[u8]>,
        TData: Serialize,
    {
        for (key, data) in iter {
            let bin_data = bincode::serialize(&data)?;
            writer.put(DbKey::new(&self.prefix, key), bin_data)?;
        }
        // We must clear the cache in order to avoid invalidated entries
        self.cache.remove_all();
        Ok(())
    }

    pub fn delete(&self, mut writer: impl DbWriter, key: TKey) -> Result<(), StoreError>
    where
        TKey: Clone + AsRef<[u8]>,
    {
        self.cache.remove(&key);
        writer.delete(DbKey::new(&self.prefix, key))?;
        Ok(())
    }

    pub fn delete_many(&self, mut writer: impl DbWriter, key_iter: &mut (impl Iterator<Item = TKey> + Clone)) -> Result<(), StoreError>
    where
        TKey: Clone + AsRef<[u8]>,
    {
        let key_iter_clone = key_iter.clone();
        self.cache.remove_many(key_iter);
        for key in key_iter_clone {
            writer.delete(DbKey::new(&self.prefix, key.clone()))?;
        }
        Ok(())
    }

    /// Deletes all entries in the store using the underlying rocksdb `delete_range` operation
    pub fn delete_all(&self, mut writer: impl DbWriter) -> Result<(), StoreError>
    where
        TKey: Clone + AsRef<[u8]>,
    {
        self.cache.remove_all();
        let db_key = DbKey::prefix_only(&self.prefix);
        let (from, to) = rocksdb::PrefixRange(db_key.as_ref()).into_bounds();
        writer.delete_range(from.unwrap(), to.unwrap())?;
        Ok(())
    }

    /// A dynamic iterator that can iterate through a specific prefix / bucket, or from a certain start point.
    //TODO: loop and chain iterators for multi-prefix / bucket iterator.
    pub fn seek_iterator(
        &self,
        bucket: Option<&[u8]>,   // iter self.prefix if None, else append bytes to self.prefix.
        seek_from: Option<TKey>, // iter whole range if None
        limit: usize,            // amount to take.
        skip_first: bool,        // skips the first value, (useful in conjunction with the seek-key, as to not re-retrieve).
    ) -> impl Iterator<Item = KeyDataResult<TData>> + '_
    where
        TKey: Clone + AsRef<[u8]>,
        TData: DeserializeOwned,
    {
        let db_key = bucket.map_or_else(
            move || DbKey::prefix_only(&self.prefix),
            move |bucket| {
                let mut key = DbKey::prefix_only(&self.prefix);
                key.add_bucket(bucket);
                key
            },
        );

        let mut read_opts = ReadOptions::default();
        read_opts.set_iterate_range(rocksdb::PrefixRange(db_key.as_ref()));

        let mut db_iterator = match seek_from {
            Some(seek_key) => {
                self.db.iterator_opt(IteratorMode::From(DbKey::new(&self.prefix, seek_key).as_ref(), Direction::Forward), read_opts)
            }
            None => self.db.iterator_opt(IteratorMode::Start, read_opts),
        };

        if skip_first {
            db_iterator.next();
        }

        db_iterator.take(limit).map(move |item| match item {
            Ok((key_bytes, value_bytes)) => match bincode::deserialize::<TData>(value_bytes.as_ref()) {
                Ok(value) => Ok((key_bytes[db_key.prefix_len()..].into(), value)),
                Err(err) => Err(err.into()),
            },
            Err(err) => Err(err.into()),
        })
    }

    /// `seek_iterator` with the same in-place decode fallback as `read_with_decode_fallback`
    /// (mixed-layout keyspaces, e.g. pruning-point utxoset chunk streaming over an old datadir).
    pub fn seek_iterator_with_decode_fallback<TDecodeFallback>(
        &self,
        bucket: Option<&[u8]>,
        seek_from: Option<TKey>,
        limit: usize,
        skip_first: bool,
    ) -> impl Iterator<Item = KeyDataResult<TData>> + '_
    where
        TKey: Clone + AsRef<[u8]>,
        TData: DeserializeOwned,
        TDecodeFallback: DeserializeOwned + Into<TData>,
    {
        let db_key = bucket.map_or_else(
            move || DbKey::prefix_only(&self.prefix),
            move |bucket| {
                let mut key = DbKey::prefix_only(&self.prefix);
                key.add_bucket(bucket);
                key
            },
        );

        let mut read_opts = ReadOptions::default();
        read_opts.set_iterate_range(rocksdb::PrefixRange(db_key.as_ref()));

        let mut db_iterator = match seek_from {
            Some(seek_key) => {
                self.db.iterator_opt(IteratorMode::From(DbKey::new(&self.prefix, seek_key).as_ref(), Direction::Forward), read_opts)
            }
            None => self.db.iterator_opt(IteratorMode::Start, read_opts),
        };

        if skip_first {
            db_iterator.next();
        }

        db_iterator.take(limit).map(move |item| match item {
            Ok((key_bytes, value_bytes)) => match bincode::deserialize::<TData>(value_bytes.as_ref()) {
                Ok(value) => Ok((key_bytes[db_key.prefix_len()..].into(), value)),
                Err(err) => match bincode::deserialize::<TDecodeFallback>(value_bytes.as_ref()) {
                    Ok(value) => Ok((key_bytes[db_key.prefix_len()..].into(), value.into())),
                    Err(_) => Err(err.into()),
                },
            },
            Err(err) => Err(err.into()),
        })
    }

    pub fn prefix(&self) -> &[u8] {
        &self.prefix
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        create_temp_db,
        prelude::{BatchDbWriter, ConnBuilder, DirectDbWriter},
    };
    use keryx_hashes::Hash;
    use rocksdb::WriteBatch;

    #[test]
    fn test_delete_all() {
        let (_lifetime, db) = create_temp_db!(ConnBuilder::default().with_files_limit(10));
        let access = CachedDbAccess::<Hash, u64>::new(db.clone(), CachePolicy::Count(2), vec![1, 2]);

        access.write_many(DirectDbWriter::new(&db), &mut (0..16).map(|i| (i.into(), 2))).unwrap();
        assert_eq!(16, access.iterator().count());
        access.delete_all(DirectDbWriter::new(&db)).unwrap();
        assert_eq!(0, access.iterator().count());

        access.write_many(DirectDbWriter::new(&db), &mut (0..16).map(|i| (i.into(), 2))).unwrap();
        assert_eq!(16, access.iterator().count());
        let mut batch = WriteBatch::default();
        access.delete_all(BatchDbWriter::new(&mut batch)).unwrap();
        assert_eq!(16, access.iterator().count());
        db.write(batch).unwrap();
        assert_eq!(0, access.iterator().count());
    }

    #[test]
    fn test_read_with_fallback() {
        let (_lifetime, db) = create_temp_db!(ConnBuilder::default().with_files_limit(10));
        let primary_prefix = vec![1];
        let fallback_prefix = vec![2];
        let access = CachedDbAccess::<Hash, u64>::new(db.clone(), CachePolicy::Count(10), primary_prefix);
        let fallback_access = CachedDbAccess::<Hash, u64>::new(db.clone(), CachePolicy::Count(10), fallback_prefix.clone());

        let key: Hash = 1.into();
        let value = 100;

        // Write to fallback
        fallback_access.write(DirectDbWriter::new(&db), key, value).unwrap();

        // Read with fallback, should succeed
        let result = access.read_with_fallback::<u64>(&fallback_prefix, key).unwrap();
        assert_eq!(result, value);

        // Key should now be in the primary cache
        assert_eq!(access.read_from_cache(&key).unwrap(), value);
    }

    #[test]
    fn test_has_with_fallback() {
        let (_lifetime, db) = create_temp_db!(ConnBuilder::default().with_files_limit(10));
        let primary_prefix = vec![1];
        let fallback_prefix = vec![2];
        let access = CachedDbAccess::<Hash, u64>::new(db.clone(), CachePolicy::Count(10), primary_prefix);
        let fallback_access = CachedDbAccess::<Hash, u64>::new(db.clone(), CachePolicy::Count(10), fallback_prefix.clone());

        let key_in_fallback: Hash = 1.into();
        let key_not_found: Hash = 2.into();

        // Write to fallback
        fallback_access.write(DirectDbWriter::new(&db), key_in_fallback, 100).unwrap();

        // Check for key in fallback, should exist
        assert!(access.has_with_fallback(&fallback_prefix, key_in_fallback).unwrap());

        // Check for key that doesn't exist, should not be found
        assert!(!access.has_with_fallback(&fallback_prefix, key_not_found).unwrap());
    }
}
