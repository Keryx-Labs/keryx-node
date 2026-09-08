//! PoM possession proofs in a fixed-size ring, in memory by default or in a file next to the
//! database (`--pom-proof-ring-file`).
//!
//! One ~440 KiB proof per block at 10 BPS lives a few minutes before the pruning GC drops it.
//! Through RocksDB those writes dominate the node's I/O and their dead copies linger in blob
//! files until compaction. Here proofs are appended to one buffer of fixed size, overwritten in
//! place once it wraps; the index lives in memory (rebuilt by a scan at open for the file
//! backend). Consensus never depends on a proof being present (the header pins the state), so a
//! proof lost to a wrap, a crash or a restart only stops that block from being re-served with
//! its proof.

use std::collections::{BTreeMap, HashMap};
use std::fs::{File, OpenOptions};
use std::io;
use std::path::Path;

use keryx_consensus_core::pom::PomProof;
use keryx_database::prelude::{DbKey, StoreError};
use keryx_database::registry::DatabaseStorePrefixes;
use keryx_hashes::Hash;
use parking_lot::Mutex;
use rocksdb::WriteBatch;

/// Read access to the full PoM possession proof of each block, persisted at body-commit time.
/// Required so a block can be re-served to peers (relay / IBD) with its proof attached:
/// `get_block` reconstructs the block from storage, and without this store `pom_proof` would be
/// `None`, causing peers to reject the served block with `PoM possession proof missing`. Only
/// blocks at/after `pom_activation` carry a proof; pre-fork blocks have no entry here.
pub trait PomProofStoreReader {
    fn get(&self, hash: Hash) -> Result<PomProof, StoreError>;
    fn has(&self, hash: Hash) -> Result<bool, StoreError>;
}

pub const RING_FILE_NAME: &str = "pom-proofs.ring";
/// Defaults unless `KERYX_POM_PROOF_RING_MB` says otherwise, sized against the
/// `POM_PROOF_SERVE_DEPTH_DAA` window at 10 BPS: ~1.5x of it in memory, ~3x on disk.
pub const DEFAULT_MEMORY_RING_BYTES: u64 = 1024 * 1024 * 1024;
pub const DEFAULT_FILE_RING_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const RING_MB_ENV: &str = "KERYX_POM_PROOF_RING_MB";
const MIN_RING_BYTES: u64 = 64 * 1024 * 1024;
/// Only used to report the covered window at boot.
const NOMINAL_PROOF_BYTES: u64 = 460 * 1024;

const RECORD_MAGIC: u32 = 0x504F_4D52;
/// magic u32 ‖ seq u64 ‖ block hash 32 ‖ payload len u32 ‖ payload check 8
const HEADER_LEN: usize = 4 + 8 + 32 + 4 + 8;

/// Ring capacity from the environment, floored so a few minutes of proofs always fit.
pub fn ring_capacity_from_env(default: u64) -> u64 {
    std::env::var(RING_MB_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .map(|mb| mb.saturating_mul(1024 * 1024))
        .unwrap_or(default)
        .max(MIN_RING_BYTES)
}

fn payload_check(payload: &[u8]) -> [u8; 8] {
    let digest = blake2b_simd::blake2b(payload);
    digest.as_bytes()[..8].try_into().expect("8 bytes")
}

#[cfg(unix)]
fn file_read_at(file: &File, offset: u64, buf: &mut [u8]) -> io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(buf, offset)
}

#[cfg(unix)]
fn file_write_at(file: &File, offset: u64, buf: &[u8]) -> io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.write_all_at(buf, offset)
}

#[cfg(windows)]
fn file_read_at(file: &File, offset: u64, buf: &mut [u8]) -> io::Result<()> {
    use std::os::windows::fs::FileExt;
    let mut done = 0usize;
    while done < buf.len() {
        let n = file.seek_read(&mut buf[done..], offset + done as u64)?;
        if n == 0 {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "short read in proof ring"));
        }
        done += n;
    }
    Ok(())
}

#[cfg(windows)]
fn file_write_at(file: &File, offset: u64, buf: &[u8]) -> io::Result<()> {
    use std::os::windows::fs::FileExt;
    let mut done = 0usize;
    while done < buf.len() {
        let n = file.seek_write(&buf[done..], offset + done as u64)?;
        if n == 0 {
            return Err(io::Error::new(io::ErrorKind::WriteZero, "short write in proof ring"));
        }
        done += n;
    }
    Ok(())
}

enum Backing {
    Memory(Vec<u8>),
    File(File),
}

impl Backing {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        match self {
            Backing::Memory(mem) => {
                let start = offset as usize;
                let src = mem.get(start..start + buf.len()).ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "read past the proof ring"))?;
                buf.copy_from_slice(src);
                Ok(())
            }
            Backing::File(file) => file_read_at(file, offset, buf),
        }
    }

    fn write_at(&mut self, offset: u64, buf: &[u8]) -> io::Result<()> {
        match self {
            Backing::Memory(mem) => {
                let start = offset as usize;
                let dst = mem.get_mut(start..start + buf.len()).ok_or_else(|| io::Error::new(io::ErrorKind::WriteZero, "write past the proof ring"))?;
                dst.copy_from_slice(buf);
                Ok(())
            }
            Backing::File(file) => file_write_at(file, offset, buf),
        }
    }
}

#[derive(Clone, Copy)]
struct Slot {
    offset: u64,
    len: u32,
}

struct Ring {
    backing: Backing,
    capacity: u64,
    write_pos: u64,
    next_seq: u64,
    by_hash: HashMap<Hash, Slot>,
    by_offset: BTreeMap<u64, Hash>,
}

impl Ring {
    fn with_backing(backing: Backing, capacity: u64) -> Self {
        Self { backing, capacity, write_pos: 0, next_seq: 1, by_hash: HashMap::new(), by_offset: BTreeMap::new() }
    }

    /// Zeroed pages are only materialized as proofs land, so RSS grows with use, not at open.
    fn in_memory(capacity: u64) -> Self {
        Self::with_backing(Backing::Memory(vec![0u8; capacity as usize]), capacity)
    }

    fn open(path: &Path, capacity: u64) -> io::Result<Self> {
        let file = OpenOptions::new().read(true).write(true).create(true).open(path)?;
        if file.metadata()?.len() != capacity {
            file.set_len(capacity)?;
        }
        let mut ring = Self::with_backing(Backing::File(file), capacity);
        ring.recover()?;
        Ok(ring)
    }

    /// Rebuild the index from the file: records are contiguous from offset 0 with strictly
    /// increasing sequence numbers up to the last write; the first bad header, bad check or
    /// non-increasing sequence marks the end (what follows is the overwritten remainder of an
    /// older lap, or zeros).
    fn recover(&mut self) -> io::Result<()> {
        let mut pos = 0u64;
        let mut last_seq = 0u64;
        let mut header = [0u8; HEADER_LEN];
        while pos + HEADER_LEN as u64 <= self.capacity {
            self.backing.read_at(pos, &mut header)?;
            let Some((seq, hash, len, check)) = parse_header(&header) else { break };
            if seq <= last_seq || len == 0 || pos + HEADER_LEN as u64 + len as u64 > self.capacity {
                break;
            }
            let mut payload = vec![0u8; len as usize];
            self.backing.read_at(pos + HEADER_LEN as u64, &mut payload)?;
            if payload_check(&payload) != check {
                break;
            }
            self.by_offset.insert(pos, hash);
            self.by_hash.insert(hash, Slot { offset: pos, len });
            last_seq = seq;
            pos += HEADER_LEN as u64 + len as u64;
        }
        self.write_pos = pos;
        self.next_seq = last_seq + 1;
        Ok(())
    }

    fn append(&mut self, hash: Hash, payload: &[u8]) -> io::Result<()> {
        let total = HEADER_LEN as u64 + payload.len() as u64;
        if total > self.capacity {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "proof larger than the ring"));
        }
        if self.write_pos + total > self.capacity {
            self.write_pos = 0;
        }
        let start = self.write_pos;
        let end = start + total;
        self.evict_range(start, end);

        let seq = self.next_seq;
        let mut buf = Vec::with_capacity(total as usize);
        buf.extend_from_slice(&RECORD_MAGIC.to_le_bytes());
        buf.extend_from_slice(&seq.to_le_bytes());
        buf.extend_from_slice(hash.as_bytes().as_ref());
        buf.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        buf.extend_from_slice(&payload_check(payload));
        buf.extend_from_slice(payload);
        self.backing.write_at(start, &buf)?;

        self.next_seq = seq + 1;
        self.write_pos = end;
        self.by_offset.insert(start, hash);
        self.by_hash.insert(hash, Slot { offset: start, len: payload.len() as u32 });
        Ok(())
    }

    /// Drop every index entry whose record overlaps `[start, end)` — the records about to be
    /// overwritten. Records never overlap each other, so walking down from `end` stops at the
    /// first one ending at or before `start`.
    fn evict_range(&mut self, start: u64, end: u64) {
        let mut victims = Vec::new();
        for (&offset, &hash) in self.by_offset.range(..end).rev() {
            let Some(slot) = self.by_hash.get(&hash) else { victims.push((offset, hash)); continue };
            let record_end = if slot.offset == offset { offset + HEADER_LEN as u64 + slot.len as u64 } else { offset };
            if record_end <= start && slot.offset == offset {
                break;
            }
            victims.push((offset, hash));
        }
        for (offset, hash) in victims {
            self.by_offset.remove(&offset);
            if self.by_hash.get(&hash).is_some_and(|s| s.offset == offset) {
                self.by_hash.remove(&hash);
            }
        }
    }

    fn remove(&mut self, hash: Hash) {
        if let Some(slot) = self.by_hash.remove(&hash) {
            self.by_offset.remove(&slot.offset);
        }
    }

    fn read(&self, hash: Hash) -> io::Result<Option<Vec<u8>>> {
        let Some(slot) = self.by_hash.get(&hash) else { return Ok(None) };
        let mut payload = vec![0u8; slot.len as usize];
        self.backing.read_at(slot.offset + HEADER_LEN as u64, &mut payload)?;
        Ok(Some(payload))
    }
}

fn parse_header(h: &[u8; HEADER_LEN]) -> Option<(u64, Hash, u32, [u8; 8])> {
    if u32::from_le_bytes(h[0..4].try_into().ok()?) != RECORD_MAGIC {
        return None;
    }
    let seq = u64::from_le_bytes(h[4..12].try_into().ok()?);
    let hash = Hash::from_bytes(h[12..44].try_into().ok()?);
    let len = u32::from_le_bytes(h[44..48].try_into().ok()?);
    let check: [u8; 8] = h[48..56].try_into().ok()?;
    Some((seq, hash, len, check))
}

fn io_error(e: io::Error) -> StoreError {
    StoreError::DataInconsistency(format!("PoM proof ring: {e}"))
}

fn window_summary(capacity: u64) -> String {
    let proofs = capacity / NOMINAL_PROOF_BYTES;
    format!("{} MB, ~{} proofs, ~{:.1} min at 10 BPS", capacity / (1024 * 1024), proofs, proofs as f64 / 600.0)
}

/// The ring implementation of `PomProofStoreReader`. `WriteBatch` parameters are accepted for
/// call-site compatibility with the other stores; the ring is written directly.
pub struct DbPomProofStore {
    ring: Mutex<Ring>,
}

impl DbPomProofStore {
    /// In-memory ring. A ring file left in `dir` by an earlier run is removed.
    pub fn in_memory(dir: &Path, capacity: u64) -> Self {
        let stale = dir.join(RING_FILE_NAME);
        match std::fs::remove_file(&stale) {
            Ok(()) => log::info!("removed the ring file {} left by an earlier run", stale.display()),
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => log::warn!("could not remove the stale ring file {}: {e}", stale.display()),
        }
        log::info!("PoM proof ring in memory: {}", window_summary(capacity));
        Self { ring: Mutex::new(Ring::in_memory(capacity)) }
    }

    pub fn open_file(dir: &Path, capacity: u64) -> io::Result<Self> {
        let path = dir.join(RING_FILE_NAME);
        let ring = Ring::open(&path, capacity)?;
        log::info!("PoM proof ring in {}: {}, {} proofs recovered", path.display(), window_summary(capacity), ring.by_hash.len());
        Ok(Self { ring: Mutex::new(ring) })
    }

    pub fn insert_batch(&self, _batch: &mut WriteBatch, hash: Hash, proof: &PomProof) -> Result<(), StoreError> {
        let payload = proof.to_wire_bytes();
        let mut ring = self.ring.lock();
        if ring.by_hash.contains_key(&hash) {
            return Err(StoreError::HashAlreadyExists(hash));
        }
        ring.append(hash, &payload).map_err(io_error)
    }

    pub fn delete_batch(&self, _batch: &mut WriteBatch, hash: Hash) -> Result<(), StoreError> {
        self.ring.lock().remove(hash);
        Ok(())
    }

    fn not_found(hash: Hash) -> StoreError {
        let prefix: Vec<u8> = DatabaseStorePrefixes::PomProof.into();
        StoreError::KeyNotFound(DbKey::new(&prefix, hash))
    }
}

impl PomProofStoreReader for DbPomProofStore {
    fn get(&self, hash: Hash) -> Result<PomProof, StoreError> {
        let payload = self.ring.lock().read(hash).map_err(io_error)?;
        match payload {
            None => Err(Self::not_found(hash)),
            Some(bytes) => PomProof::from_wire_bytes(&bytes).map_err(io_error),
        }
    }

    fn has(&self, hash: Hash) -> Result<bool, StoreError> {
        Ok(self.ring.lock().by_hash.contains_key(&hash))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("keryx-pom-ring-{}-{}", name, std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn temp_ring(name: &str, capacity: u64) -> (PathBuf, Ring) {
        let path = temp_dir(name).join(RING_FILE_NAME);
        let _ = std::fs::remove_file(&path);
        let ring = Ring::open(&path, capacity).unwrap();
        (path, ring)
    }

    /// Both backends, so every behaviour test runs on each.
    fn both_rings(name: &str, capacity: u64) -> Vec<(Option<PathBuf>, Ring)> {
        let (path, file_ring) = temp_ring(name, capacity);
        vec![(None, Ring::in_memory(capacity)), (Some(path), file_ring)]
    }

    fn h(i: u8) -> Hash {
        Hash::from_bytes([i; 32])
    }

    #[test]
    fn round_trip_and_delete() {
        for (path, mut ring) in both_rings("rt", MIN_RING_BYTES) {
            ring.append(h(1), b"alpha").unwrap();
            ring.append(h(2), b"beta").unwrap();
            assert_eq!(ring.read(h(1)).unwrap().as_deref(), Some(&b"alpha"[..]));
            assert_eq!(ring.read(h(2)).unwrap().as_deref(), Some(&b"beta"[..]));
            ring.remove(h(1));
            assert!(ring.read(h(1)).unwrap().is_none());
            assert_eq!(ring.by_offset.len(), 1);
            if let Some(path) = path {
                std::fs::remove_file(path).unwrap();
            }
        }
    }

    #[test]
    fn wrap_evicts_the_overwritten_records_only() {
        // 4 records of 100 bytes fit; the 5th wraps to 0 and evicts the 1st (and only the 1st).
        let record = HEADER_LEN as u64 + 100;
        for (path, mut ring) in both_rings("wrap", MIN_RING_BYTES) {
            ring.capacity = record * 4 + 10;
            for i in 1..=4u8 {
                ring.append(h(i), &[i; 100]).unwrap();
            }
            ring.append(h(5), &[5; 100]).unwrap();
            assert!(ring.read(h(1)).unwrap().is_none());
            assert_eq!(ring.read(h(2)).unwrap().unwrap(), vec![2u8; 100]);
            assert_eq!(ring.read(h(5)).unwrap().unwrap(), vec![5u8; 100]);
            assert_eq!(ring.write_pos, record);
            // A longer record spanning two old ones evicts both.
            ring.append(h(6), &[6; 200]).unwrap();
            assert!(ring.read(h(2)).unwrap().is_none());
            assert!(ring.read(h(3)).unwrap().is_none());
            assert_eq!(ring.read(h(4)).unwrap().unwrap(), vec![4u8; 100]);
            if let Some(path) = path {
                std::fs::remove_file(path).unwrap();
            }
        }
    }

    #[test]
    fn recover_rebuilds_the_index_and_continues_the_sequence() {
        let (path, mut ring) = temp_ring("recover", MIN_RING_BYTES);
        ring.append(h(1), b"one").unwrap();
        ring.append(h(2), b"two").unwrap();
        let write_pos = ring.write_pos;
        drop(ring);
        let mut reopened = Ring::open(&path, MIN_RING_BYTES).unwrap();
        assert_eq!(reopened.write_pos, write_pos);
        assert_eq!(reopened.next_seq, 3);
        assert_eq!(reopened.read(h(2)).unwrap().as_deref(), Some(&b"two"[..]));
        reopened.append(h(3), b"three").unwrap();
        assert_eq!(reopened.read(h(1)).unwrap().as_deref(), Some(&b"one"[..]));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn recover_stops_at_a_corrupted_record() {
        let (path, mut ring) = temp_ring("corrupt", MIN_RING_BYTES);
        ring.append(h(1), b"good").unwrap();
        ring.append(h(2), b"bad!").unwrap();
        let bad_payload_at = ring.by_hash[&h(2)].offset + HEADER_LEN as u64;
        ring.backing.write_at(bad_payload_at, b"BAD!").unwrap();
        drop(ring);
        let reopened = Ring::open(&path, MIN_RING_BYTES).unwrap();
        assert!(reopened.read(h(1)).unwrap().is_some());
        assert!(reopened.read(h(2)).unwrap().is_none());
        assert_eq!(reopened.next_seq, 2);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn in_memory_store_removes_a_stale_ring_file() {
        let dir = temp_dir("stale");
        let path = dir.join(RING_FILE_NAME);
        std::fs::write(&path, b"leftover").unwrap();
        let store = DbPomProofStore::in_memory(&dir, MIN_RING_BYTES);
        assert!(!path.exists());
        assert!(!store.has(h(1)).unwrap());
    }

    #[test]
    fn capacity_env_is_floored() {
        assert!(ring_capacity_from_env(DEFAULT_MEMORY_RING_BYTES) >= MIN_RING_BYTES);
        assert!(ring_capacity_from_env(1) >= MIN_RING_BYTES);
    }
}
