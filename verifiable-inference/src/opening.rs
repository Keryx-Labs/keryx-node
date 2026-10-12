//! Opening payload format.

use crate::merkle::Hash;

pub const VERSION: u8 = 1;
/// Largest opening body.
pub const MAX_BODY_LEN: usize = 80 * 1024;
/// `[escrow_pubkey: 32] [signature: 64]` after the body.
pub const SIGNATURE_LEN: usize = 32 + 64;
/// Smallest opening payload: version, response id, point count and the signature.
pub const MIN_PAYLOAD_LEN: usize = 1 + 32 + 1 + SIGNATURE_LEN;
pub const MAX_PAYLOAD_LEN: usize = MAX_BODY_LEN + SIGNATURE_LEN;

/// Signed opening payload split into its body, signer key and signature.
pub struct SignedPayload<'a> {
    pub body: &'a [u8],
    pub escrow_pubkey: [u8; 32],
    pub signature: [u8; 64],
}

impl<'a> SignedPayload<'a> {
    pub fn split(payload: &'a [u8]) -> Option<Self> {
        if payload.len() < MIN_PAYLOAD_LEN || payload.len() > MAX_PAYLOAD_LEN {
            return None;
        }
        let (body, tail) = payload.split_at(payload.len() - SIGNATURE_LEN);
        Some(Self { body, escrow_pubkey: tail[..32].try_into().ok()?, signature: tail[32..].try_into().ok()? })
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct EntryOpening {
    pub index: u32,
    pub bytes: Vec<u8>,
    pub path: Vec<Hash>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ChunkRange {
    pub first_chunk: u64,
    pub chunks: Vec<u8>,
    pub proof: Vec<Hash>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Point {
    pub n_ubatch: u32,
    pub pos_path: Vec<Hash>,
    pub meta_path: Vec<Hash>,
    pub entry: EntryOpening,
    pub partner: Option<EntryOpening>,
    pub bias: Option<(EntryOpening, ChunkRange)>,
    pub rows: Vec<ChunkRange>,
    pub x: Vec<u8>,
    pub x_path: Vec<Hash>,
    pub y_total_len: u32,
    pub y_chunk_index: u32,
    pub y_chunk: Vec<u8>,
    pub y_chunk_path: Vec<Hash>,
    pub y_path: Vec<Hash>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Opening {
    pub response_txid: Hash,
    pub points: Vec<Point>,
}

struct W(Vec<u8>);

impl W {
    fn u8(&mut self, v: u8) {
        self.0.push(v);
    }
    fn u16(&mut self, v: u16) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn bytes32(&mut self, b: &[u8]) {
        self.u32(b.len() as u32);
        self.0.extend_from_slice(b);
    }
    fn hashes(&mut self, h: &[Hash]) {
        self.u8(h.len() as u8);
        for x in h {
            self.0.extend_from_slice(x);
        }
    }
    fn entry(&mut self, e: &EntryOpening) {
        self.u32(e.index);
        self.u16(e.bytes.len() as u16);
        self.0.extend_from_slice(&e.bytes);
        self.hashes(&e.path);
    }
    fn range(&mut self, r: &ChunkRange) {
        self.u64(r.first_chunk);
        self.u32((r.chunks.len() / 32) as u32);
        self.0.extend_from_slice(&r.chunks);
        self.hashes(&r.proof);
    }
}

struct R<'a> {
    b: &'a [u8],
    o: usize,
}

impl R<'_> {
    fn take(&mut self, n: usize) -> Option<&[u8]> {
        let s = self.b.get(self.o..self.o.checked_add(n)?)?;
        self.o += n;
        Some(s)
    }
    fn u8(&mut self) -> Option<u8> {
        Some(self.take(1)?[0])
    }
    fn u16(&mut self) -> Option<u16> {
        Some(u16::from_le_bytes(self.take(2)?.try_into().ok()?))
    }
    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }
    fn bytes32(&mut self) -> Option<Vec<u8>> {
        let n = self.u32()? as usize;
        Some(self.take(n)?.to_vec())
    }
    fn hashes(&mut self) -> Option<Vec<Hash>> {
        let n = self.u8()? as usize;
        (0..n).map(|_| self.take(32).map(|s| s.try_into().unwrap())).collect()
    }
    fn entry(&mut self) -> Option<EntryOpening> {
        let index = self.u32()?;
        let n = self.u16()? as usize;
        let bytes = self.take(n)?.to_vec();
        Some(EntryOpening { index, bytes, path: self.hashes()? })
    }
    fn range(&mut self) -> Option<ChunkRange> {
        let first_chunk = self.u64()?;
        let n = self.u32()? as usize;
        let chunks = self.take(n.checked_mul(32)?)?.to_vec();
        Some(ChunkRange { first_chunk, chunks, proof: self.hashes()? })
    }
}

impl Opening {
    pub fn encode(&self) -> Vec<u8> {
        let mut w = W(Vec::new());
        w.u8(VERSION);
        w.0.extend_from_slice(&self.response_txid);
        w.u8(self.points.len() as u8);
        for p in &self.points {
            w.u32(p.n_ubatch);
            w.hashes(&p.pos_path);
            w.hashes(&p.meta_path);
            w.entry(&p.entry);
            match &p.partner {
                Some(e) => {
                    w.u8(1);
                    w.entry(e);
                }
                None => w.u8(0),
            }
            match &p.bias {
                Some((e, r)) => {
                    w.u8(1);
                    w.entry(e);
                    w.range(r);
                }
                None => w.u8(0),
            }
            w.u8(p.rows.len() as u8);
            for r in &p.rows {
                w.range(r);
            }
            w.bytes32(&p.x);
            w.hashes(&p.x_path);
            w.u32(p.y_total_len);
            w.u32(p.y_chunk_index);
            w.bytes32(&p.y_chunk);
            w.hashes(&p.y_chunk_path);
            w.hashes(&p.y_path);
        }
        w.0
    }

    pub fn decode(b: &[u8]) -> Option<Opening> {
        let mut r = R { b, o: 0 };
        if r.u8()? != VERSION {
            return None;
        }
        let response_txid: Hash = r.take(32)?.try_into().ok()?;
        let n = r.u8()? as usize;
        let mut points = Vec::with_capacity(n);
        for _ in 0..n {
            let n_ubatch = r.u32()?;
            let pos_path = r.hashes()?;
            let meta_path = r.hashes()?;
            let entry = r.entry()?;
            let partner = if r.u8()? == 1 { Some(r.entry()?) } else { None };
            let bias = if r.u8()? == 1 { Some((r.entry()?, r.range()?)) } else { None };
            let nr = r.u8()? as usize;
            let rows = (0..nr).map(|_| r.range()).collect::<Option<Vec<_>>>()?;
            points.push(Point {
                n_ubatch,
                pos_path,
                meta_path,
                entry,
                partner,
                bias,
                rows,
                x: r.bytes32()?,
                x_path: r.hashes()?,
                y_total_len: r.u32()?,
                y_chunk_index: r.u32()?,
                y_chunk: r.bytes32()?,
                y_chunk_path: r.hashes()?,
                y_path: r.hashes()?,
            });
        }
        (r.o == b.len()).then_some(Opening { response_txid, points })
    }
}
