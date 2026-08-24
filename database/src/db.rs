use rocksdb::{DBWithThreadMode, MultiThreaded};
use std::ops::{Deref, DerefMut};
use std::path::PathBuf;

pub use conn_builder::ConnBuilder;
use keryx_utils::fd_budget::FDGuard;
pub use rocksdb_preset::{
    DA_1500_BLOB_CACHE_MIN_BYTES, DEFAULT_HDD_RATE_LIMIT_BYTES_PER_SEC, RocksDbPreset, RocksDbResources, disable_blob_files,
};

mod conn_builder;
mod rocksdb_preset;

/// The DB type used for Kaspad stores
pub struct DB {
    inner: DBWithThreadMode<MultiThreaded>,
    _fd_guard: FDGuard,
}

impl DB {
    pub fn new(inner: DBWithThreadMode<MultiThreaded>, fd_guard: FDGuard) -> Self {
        Self { inner, _fd_guard: fd_guard }
    }
}

impl DerefMut for DB {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl Deref for DB {
    type Target = DBWithThreadMode<MultiThreaded>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

/// Deletes an existing DB if it exists
pub fn delete_db(db_dir: PathBuf) {
    if !db_dir.exists() {
        return;
    }
    let options = rocksdb::Options::default();
    let path = db_dir.to_str().unwrap();
    <DBWithThreadMode<MultiThreaded>>::destroy(&options, path).expect("DB is expected to be deletable");
}
