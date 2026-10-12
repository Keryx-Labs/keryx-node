//! Model manifest entries.

use crate::merkle::{self, Hash};

pub const ROLE_PRODUCT: u8 = 0;
pub const ROLE_BIAS: u8 = 1;
pub const ROLE_OUTPUT: u8 = 2;
pub const FORM_NONE: u8 = 0;
pub const FORM_SPLIT: u8 = 1;
pub const FORM_PAIR_GATE: u8 = 2;
pub const FORM_PAIR_UP: u8 = 3;
pub const GLU_SWIGLU: u8 = 1;
pub const GLU_GEGLU: u8 = 2;
pub const NONE: u32 = u32::MAX;

#[derive(Debug, Clone, PartialEq)]
pub struct Entry {
    pub name: String,
    pub wtype: u32,
    pub k: u64,
    pub m: u64,
    pub n_exp: u32,
    pub chunk_base: u64,
    pub n_bytes: u64,
    pub row_bytes: u32,
    pub role: u8,
    pub glu_op: u8,
    pub glu_form: u8,
    pub partner: u32,
    pub bias: u32,
}

impl Entry {
    /// Parse one serialized entry; returns the entry and its byte length.
    pub fn parse(b: &[u8]) -> Option<(Entry, usize)> {
        let nl = u16::from_le_bytes(b.get(0..2)?.try_into().ok()?) as usize;
        let name = String::from_utf8(b.get(2..2 + nl)?.to_vec()).ok()?;
        let r = b.get(2 + nl..2 + nl + 55)?;
        let u32a = |o: usize| u32::from_le_bytes(r[o..o + 4].try_into().unwrap());
        let u64a = |o: usize| u64::from_le_bytes(r[o..o + 8].try_into().unwrap());
        let e = Entry {
            name,
            wtype: u32a(0),
            k: u64a(4),
            m: u64a(12),
            n_exp: u32a(20),
            chunk_base: u64a(24),
            n_bytes: u64a(32),
            row_bytes: u32a(40),
            role: r[44],
            glu_op: r[45],
            glu_form: r[46],
            partner: u32a(47),
            bias: u32a(51),
        };
        Some((e, 2 + nl + 55))
    }

    pub fn leaf(bytes: &[u8]) -> Hash {
        *blake3::hash(bytes).as_bytes()
    }

    /// Chunk range covering a weight row, and the row offset in its first chunk.
    pub fn row_chunks(&self, exp: u64, row: u64) -> Option<(u64, u64, usize)> {
        if row >= self.m || exp >= self.n_exp as u64 {
            return None;
        }
        let start = (exp * self.m + row) * self.row_bytes as u64;
        let end = start + self.row_bytes as u64;
        if end > self.n_bytes / 32 * 32 {
            return None;
        }
        Some((self.chunk_base + start / 32, self.chunk_base + (end - 1) / 32, (start % 32) as usize))
    }
}

/// Inclusion of an entry in the manifest.
pub fn verify_entry(m_t: &Hash, n_entries: u64, index: u64, entry_bytes: &[u8], path: &[Hash]) -> bool {
    merkle::rt_verify_leaf(m_t, n_entries, index, &Entry::leaf(entry_bytes), path)
}

/// Entries with their serialized bytes, and the product count.
pub type Manifest = (Vec<(Entry, Vec<u8>)>, u32);

/// Entries of a serialized manifest and its product count.
pub fn read_file(data: &[u8]) -> Option<Manifest> {
    if data.get(0..8)? != b"KXMANIF1" {
        return None;
    }
    let n = u32::from_le_bytes(data[8..12].try_into().ok()?) as usize;
    let n_products = u32::from_le_bytes(data[12..16].try_into().ok()?);
    let mut o = 16;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let (e, len) = Entry::parse(&data[o..])?;
        out.push((e, data[o..o + len].to_vec()));
        o += len;
    }
    Some((out, n_products))
}
