//! Opening payload format.

use crate::merkle::Hash;

pub const VERSION: u8 = 1;
/// Version 1 plus a prefix point after the points.
pub const VERSION_PREFIX: u8 = 2;
/// Stage-2 opening: each point carries its token and input link, then the prefix and residual points.
pub const VERSION_CHAIN: u8 = 3;
/// Envelope naming the seed block an opening was built for: `[4] [response id] [seed block]`, then
/// the inner body's version and the rest of that body after its response id.
pub const VERSION_SEEDED: u8 = 4;
/// Largest opening body.
pub const MAX_BODY_LEN: usize = 112 * 1024;
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

/// The layer-0 check of one prefix position: the input of the first product against the
/// embedding of the token the request id puts there.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PrefixPoint {
    pub n_ubatch: u32,
    pub pos_path: Vec<Hash>,
    pub meta_path: Vec<Hash>,
    pub entry: EntryOpening,
    pub x: Vec<u8>,
    pub x_path: Vec<Hash>,
    pub embd: (EntryOpening, ChunkRange),
    pub norm: (EntryOpening, ChunkRange),
}

/// Last chunk of an expert product's `Y` leaf, which ends with the slot expert ids.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ExpertIds {
    pub chunk: Vec<u8>,
    pub chunk_path: Vec<Hash>,
}

/// One 1 KiB chunk of a committed leaf.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LeafChunk {
    pub total_len: u32,
    pub index: u32,
    pub chunk: Vec<u8>,
    pub chunk_path: Vec<Hash>,
    pub leaf_path: Vec<Hash>,
}

/// A whole committed leaf.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LeafBytes {
    pub bytes: Vec<u8>,
    pub leaf_path: Vec<Hash>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LinkOpening {
    None,
    Norm { src: LeafBytes, norm: (EntryOpening, ChunkRange) },
    Token { embd: (EntryOpening, ChunkRange), norm: (EntryOpening, ChunkRange) },
    Source { src: EntryOpening, y: LeafChunk, ids: Option<ExpertIds> },
}

#[derive(Debug, Clone, PartialEq)]
pub enum ResidSrc {
    Chunk(LeafChunk),
    Token(EntryOpening, ChunkRange),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ResidAdd {
    Chunk(LeafChunk),
    Normed { y: LeafBytes, norm: (EntryOpening, ChunkRange) },
    Moe(Box<MoeAdd>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct MoeAdd {
    pub scores: LeafBytes,
    pub bias: (EntryOpening, ChunkRange),
    pub ids: ExpertIds,
    pub slots: Vec<LeafChunk>,
    pub shexp: LeafChunk,
    pub router_row: (EntryOpening, ChunkRange),
    pub gate: EntryOpening,
    pub gate_x: LeafBytes,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResidPoint {
    pub n_ubatch: u32,
    pub tok: u32,
    pub pos_path: Vec<Hash>,
    pub meta_path: Vec<Hash>,
    pub layer: EntryOpening,
    pub product: EntryOpening,
    pub dst: LeafChunk,
    pub src: ResidSrc,
    pub add: ResidAdd,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Opening {
    pub response_txid: Hash,
    pub points: Vec<Point>,
    pub prefix: Option<PrefixPoint>,
    /// Stage 2: `(token, link)` of each point.
    pub links: Vec<(u32, LinkOpening)>,
    /// Stage 2: expert ids of each point drawn on an expert product.
    pub experts: Vec<Option<ExpertIds>>,
    pub resid: Option<ResidPoint>,
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
    fn weights(&mut self, w: &(EntryOpening, ChunkRange)) {
        self.entry(&w.0);
        self.range(&w.1);
    }
    fn leaf_chunk(&mut self, c: &LeafChunk) {
        self.u32(c.total_len);
        self.u32(c.index);
        self.bytes32(&c.chunk);
        self.hashes(&c.chunk_path);
        self.hashes(&c.leaf_path);
    }
    fn leaf_bytes(&mut self, l: &LeafBytes) {
        self.bytes32(&l.bytes);
        self.hashes(&l.leaf_path);
    }
    fn link(&mut self, l: &LinkOpening) {
        match l {
            LinkOpening::None => self.u8(0),
            LinkOpening::Norm { src, norm } => {
                self.u8(1);
                self.leaf_bytes(src);
                self.weights(norm);
            }
            LinkOpening::Token { embd, norm } => {
                self.u8(2);
                self.weights(embd);
                self.weights(norm);
            }
            LinkOpening::Source { src, y, ids } => {
                self.u8(3);
                self.entry(src);
                self.leaf_chunk(y);
                match ids {
                    Some(e) => {
                        self.u8(1);
                        self.bytes32(&e.chunk);
                        self.hashes(&e.chunk_path);
                    }
                    None => self.u8(0),
                }
            }
        }
    }
    fn resid(&mut self, p: &ResidPoint) {
        self.u32(p.n_ubatch);
        self.u32(p.tok);
        self.hashes(&p.pos_path);
        self.hashes(&p.meta_path);
        self.entry(&p.layer);
        self.entry(&p.product);
        self.leaf_chunk(&p.dst);
        match &p.src {
            ResidSrc::Chunk(c) => {
                self.u8(0);
                self.leaf_chunk(c);
            }
            ResidSrc::Token(e, r) => {
                self.u8(1);
                self.entry(e);
                self.range(r);
            }
        }
        match &p.add {
            ResidAdd::Chunk(c) => {
                self.u8(0);
                self.leaf_chunk(c);
            }
            ResidAdd::Normed { y, norm } => {
                self.u8(1);
                self.leaf_bytes(y);
                self.weights(norm);
            }
            ResidAdd::Moe(m) => {
                self.u8(2);
                self.leaf_bytes(&m.scores);
                self.weights(&m.bias);
                self.bytes32(&m.ids.chunk);
                self.hashes(&m.ids.chunk_path);
                self.u8(m.slots.len() as u8);
                for c in &m.slots {
                    self.leaf_chunk(c);
                }
                self.leaf_chunk(&m.shexp);
                self.weights(&m.router_row);
                self.entry(&m.gate);
                self.leaf_bytes(&m.gate_x);
            }
        }
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
    fn weights(&mut self) -> Option<(EntryOpening, ChunkRange)> {
        Some((self.entry()?, self.range()?))
    }
    fn leaf_chunk(&mut self) -> Option<LeafChunk> {
        Some(LeafChunk {
            total_len: self.u32()?,
            index: self.u32()?,
            chunk: self.bytes32()?,
            chunk_path: self.hashes()?,
            leaf_path: self.hashes()?,
        })
    }
    fn leaf_bytes(&mut self) -> Option<LeafBytes> {
        Some(LeafBytes { bytes: self.bytes32()?, leaf_path: self.hashes()? })
    }
    fn link(&mut self) -> Option<LinkOpening> {
        Some(match self.u8()? {
            0 => LinkOpening::None,
            1 => LinkOpening::Norm { src: self.leaf_bytes()?, norm: self.weights()? },
            2 => LinkOpening::Token { embd: self.weights()?, norm: self.weights()? },
            3 => LinkOpening::Source {
                src: self.entry()?,
                y: self.leaf_chunk()?,
                ids: match self.u8()? {
                    0 => None,
                    1 => Some(ExpertIds { chunk: self.bytes32()?, chunk_path: self.hashes()? }),
                    _ => return None,
                },
            },
            _ => return None,
        })
    }
    fn resid(&mut self) -> Option<ResidPoint> {
        let n_ubatch = self.u32()?;
        let tok = self.u32()?;
        let pos_path = self.hashes()?;
        let meta_path = self.hashes()?;
        let layer = self.entry()?;
        let product = self.entry()?;
        let dst = self.leaf_chunk()?;
        let src = match self.u8()? {
            0 => ResidSrc::Chunk(self.leaf_chunk()?),
            1 => ResidSrc::Token(self.entry()?, self.range()?),
            _ => return None,
        };
        let add = match self.u8()? {
            0 => ResidAdd::Chunk(self.leaf_chunk()?),
            1 => ResidAdd::Normed { y: self.leaf_bytes()?, norm: self.weights()? },
            2 => {
                let scores = self.leaf_bytes()?;
                let bias = self.weights()?;
                let ids = ExpertIds { chunk: self.bytes32()?, chunk_path: self.hashes()? };
                let n = self.u8()? as usize;
                let slots = (0..n).map(|_| self.leaf_chunk()).collect::<Option<Vec<_>>>()?;
                ResidAdd::Moe(Box::new(MoeAdd {
                    scores,
                    bias,
                    ids,
                    slots,
                    shexp: self.leaf_chunk()?,
                    router_row: self.weights()?,
                    gate: self.entry()?,
                    gate_x: self.leaf_bytes()?,
                }))
            }
            _ => return None,
        };
        Some(ResidPoint { n_ubatch, tok, pos_path, meta_path, layer, product, dst, src, add })
    }
}

impl Opening {
    pub fn encode(&self) -> Vec<u8> {
        let mut w = W(Vec::new());
        let chain = self.resid.is_some();
        w.u8(if chain {
            VERSION_CHAIN
        } else if self.prefix.is_some() {
            VERSION_PREFIX
        } else {
            VERSION
        });
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
        if chain {
            for (i, (tok, link)) in self.links.iter().enumerate() {
                w.u32(*tok);
                w.link(link);
                match self.experts.get(i).cloned().flatten() {
                    Some(e) => {
                        w.u8(1);
                        w.bytes32(&e.chunk);
                        w.hashes(&e.chunk_path);
                    }
                    None => w.u8(0),
                }
            }
        }
        if let Some(p) = &self.prefix {
            w.u32(p.n_ubatch);
            w.hashes(&p.pos_path);
            w.hashes(&p.meta_path);
            w.entry(&p.entry);
            w.bytes32(&p.x);
            w.hashes(&p.x_path);
            w.entry(&p.embd.0);
            w.range(&p.embd.1);
            w.entry(&p.norm.0);
            w.range(&p.norm.1);
        }
        if let Some(p) = &self.resid {
            w.resid(p);
        }
        w.0
    }

    pub fn decode(b: &[u8]) -> Option<Opening> {
        if b.first() == Some(&VERSION_SEEDED) {
            let inner = *b.get(65)?;
            if inner == VERSION_SEEDED {
                return None;
            }
            let mut v = Vec::with_capacity(b.len() - 32);
            v.push(inner);
            v.extend_from_slice(&b[1..33]);
            v.extend_from_slice(&b[66..]);
            return Self::decode(&v);
        }
        let mut r = R { b, o: 0 };
        let version = r.u8()?;
        if !matches!(version, VERSION | VERSION_PREFIX | VERSION_CHAIN) {
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
        let mut experts = Vec::new();
        let links = if version == VERSION_CHAIN {
            let mut links = Vec::with_capacity(n);
            for _ in 0..n {
                links.push((r.u32()?, r.link()?));
                experts.push(match r.u8()? {
                    0 => None,
                    1 => Some(ExpertIds { chunk: r.bytes32()?, chunk_path: r.hashes()? }),
                    _ => return None,
                });
            }
            links
        } else {
            Vec::new()
        };
        let prefix = if version >= VERSION_PREFIX {
            Some(PrefixPoint {
                n_ubatch: r.u32()?,
                pos_path: r.hashes()?,
                meta_path: r.hashes()?,
                entry: r.entry()?,
                x: r.bytes32()?,
                x_path: r.hashes()?,
                embd: (r.entry()?, r.range()?),
                norm: (r.entry()?, r.range()?),
            })
        } else {
            None
        };
        let resid = if version == VERSION_CHAIN { Some(r.resid()?) } else { None };
        (r.o == b.len()).then_some(Opening { response_txid, points, prefix, links, experts, resid })
    }
}

/// The seed block a seeded opening body names; `None` for an unseeded body.
pub fn seed_block(body: &[u8]) -> Option<Hash> {
    if body.first() != Some(&VERSION_SEEDED) {
        return None;
    }
    body.get(33..65)?.try_into().ok()
}

/// Wraps an unseeded opening body into the seeded envelope naming `seed_block`.
pub fn seeded(body: &[u8], seed_block: &Hash) -> Option<Vec<u8>> {
    let version = *body.first()?;
    if !matches!(version, VERSION | VERSION_PREFIX | VERSION_CHAIN) || body.len() < 33 {
        return None;
    }
    let mut v = Vec::with_capacity(body.len() + 33);
    v.push(VERSION_SEEDED);
    v.extend_from_slice(&body[1..33]);
    v.extend_from_slice(seed_block);
    v.push(version);
    v.extend_from_slice(&body[33..]);
    Some(v)
}
