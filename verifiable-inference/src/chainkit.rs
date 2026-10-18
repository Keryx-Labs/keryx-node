//! Two-layer synthetic tier and stage-2 traces for tests.

use crate::chain::{self, ChainCtx, Resid};
use crate::dot;
use crate::manifest::{self, Chain, Entry, LayerLinks, ProductLink};
use crate::draw;
use crate::merkle::{B3_CHUNK, Hash, b3_chunk_cv, b3_path, b3_root, rt_range_proof};
use crate::opening::{ChunkRange, EntryOpening, LeafBytes, LeafChunk, LinkOpening, Point, ResidAdd, ResidPoint, ResidSrc};
use crate::prefix::expected_input;
use crate::q8;
use crate::quant;
use crate::f16;
use crate::testkit::{q8_1_row, serialize};
use crate::verify::TierCtx;

pub const D: usize = 512;
const HEAD_M: usize = 32;
const Q_M: usize = 32;
const N_LAYERS: usize = 2;
pub const VOCAB: usize = 256;
pub const EPS: f64 = 1e-6;

const PER_LAYER: usize = 5;
const ATTN_OUTPUT: usize = 0;
const ATTN_Q: usize = 1;
const FFN_DOWN: usize = 2;
const FFN_GATE: usize = 3;
const FFN_UP: usize = 4;
const N_PRODUCTS: usize = N_LAYERS * PER_LAYER + 1;
const OUTPUT: usize = N_LAYERS * PER_LAYER;
const EMBD: usize = N_PRODUCTS;
const NORMS: usize = EMBD + 1;
const OUTPUT_NORM: usize = NORMS + 2 * N_LAYERS;
const LAYERS: usize = OUTPUT_NORM + 1;

fn q8_0_rows(seed: u64, rows: usize, scale: f32) -> Vec<u8> {
    let mut out = Vec::with_capacity(rows * D / 32 * 34);
    for r in 0..rows {
        for b in 0..D / 32 {
            out.extend_from_slice(&f16::from_f32(scale).to_le_bytes());
            let h = blake3::hash(&[seed.to_le_bytes(), ((r * D / 32 + b) as u64).to_le_bytes()].concat());
            out.extend_from_slice(h.as_bytes());
        }
    }
    out
}

fn rank(l: usize, p: usize) -> usize {
    l * PER_LAYER + p
}

fn attn_norm(l: usize) -> usize {
    NORMS + 2 * l
}

fn ffn_norm(l: usize) -> usize {
    NORMS + 2 * l + 1
}

pub struct Tier {
    pub weights: Vec<u8>,
    pub chunk_leaves: Vec<Hash>,
    pub entries: Vec<(Entry, Vec<u8>)>,
    pub entry_leaves: Vec<Hash>,
    pub r_t: Hash,
    pub m_t: Hash,
}

/// Committed leaves of one position.
pub struct Position {
    pub tok: u32,
    pub leaves: Vec<Vec<u8>>,
    pub hashes: Vec<Hash>,
}

pub struct Trace {
    pub n_prompt: u32,
    pub positions: Vec<Position>,
    pub pos_roots: Vec<Hash>,
    pub trace_root: Hash,
}

impl Tier {
    pub fn new() -> Self {
        let row_bytes = (D / 32 * 34) as u32;
        let mut weights = Vec::new();
        let mut entries: Vec<Entry> = Vec::new();
        let base = |name: String, role: u8, wtype: u32, m: usize, off: usize, n_bytes: usize, rb: u32| Entry {
            name,
            wtype,
            k: D as u64,
            m: m as u64,
            n_exp: 1,
            chunk_base: (off / 32) as u64,
            n_bytes: n_bytes as u64,
            row_bytes: rb,
            role,
            glu_op: 0,
            glu_form: manifest::FORM_NONE,
            partner: manifest::NONE,
            bias: manifest::NONE,
            chain: Chain::None,
        };
        let link = |layer: usize, link: u8, norm: usize, src: usize| {
            Chain::Product(ProductLink { layer: layer as u32, link, norm: norm as u32, src: src as u32 })
        };
        let none = manifest::NONE as usize;

        let push = |weights: &mut Vec<u8>, bytes: Vec<u8>| {
            let off = weights.len();
            weights.extend_from_slice(&bytes);
            (off, bytes.len())
        };
        for l in 0..N_LAYERS {
            let seed = 100 * (l as u64 + 1);
            for p in 0..PER_LAYER {
                let (name, m) = match p {
                    ATTN_OUTPUT => ("attn_output", D),
                    ATTN_Q => ("attn_q", Q_M),
                    FFN_DOWN => ("ffn_down", D),
                    FFN_GATE => ("ffn_gate", D),
                    _ => ("ffn_up", D),
                };
                let (off, n) = push(&mut weights, q8_0_rows(seed + p as u64, m, 0.0004));
                let mut e = base(format!("blk.{l}.{name}.weight"), manifest::ROLE_PRODUCT, quant::Q8_0, m, off, n, row_bytes);
                e.chain = match p {
                    ATTN_OUTPUT => link(l, manifest::LINK_NONE, none, none),
                    ATTN_Q => link(l, manifest::LINK_ATTN_IN, attn_norm(l), none),
                    FFN_DOWN => link(l, manifest::LINK_GLU, none, rank(l, FFN_GATE)),
                    _ => link(l, manifest::LINK_FFN_IN, ffn_norm(l), none),
                };
                if p == FFN_GATE || p == FFN_UP {
                    e.glu_op = manifest::GLU_SWIGLU;
                    e.glu_form = if p == FFN_GATE { manifest::FORM_PAIR_GATE } else { manifest::FORM_PAIR_UP };
                    e.partner = rank(l, if p == FFN_GATE { FFN_UP } else { FFN_GATE }) as u32;
                }
                entries.push(e);
            }
        }
        let (off, n) = push(&mut weights, q8_0_rows(900, HEAD_M, 0.0004));
        let mut head = base("output.weight".into(), manifest::ROLE_OUTPUT, quant::Q8_0, HEAD_M, off, n, row_bytes);
        head.chain = link(N_LAYERS, manifest::LINK_HEAD, OUTPUT_NORM, none);
        entries.push(head);

        let (off, n) = push(&mut weights, q8_0_rows(3, VOCAB, 0.02));
        entries.push(base("token_embd.weight".into(), manifest::ROLE_EMBD, quant::Q8_0, VOCAB, off, n, row_bytes));
        for (i, name) in ["blk.0.attn_norm", "blk.0.ffn_norm", "blk.1.attn_norm", "blk.1.ffn_norm", "output_norm"].iter().enumerate() {
            let w: Vec<u8> = (0..D).flat_map(|j| (1.0 + 0.001 * ((j + 3 * i) % 17) as f32).to_le_bytes()).collect();
            let (off, n) = push(&mut weights, w);
            entries.push(base(format!("{name}.weight"), manifest::ROLE_NORM, 0, 1, off, n, 4 * D as u32));
        }
        for l in 0..N_LAYERS {
            let mut e = base(format!("blk.{l}"), manifest::ROLE_LAYER, 0, 0, 0, 0, 0);
            e.chain = Chain::Layer(LayerLinks {
                layer: l as u32,
                proj: rank(l, ATTN_OUTPUT) as u32,
                down: rank(l, FFN_DOWN) as u32,
                post_attn: manifest::NONE,
                post_ffn: manifest::NONE,
                scale: manifest::NONE,
            });
            entries.push(e);
        }

        let chunk_leaves: Vec<Hash> = weights.chunks_exact(32).map(|c| *blake3::hash(c).as_bytes()).collect();
        let r_t = rt_range_proof(&chunk_leaves, 0, 1).0;
        let entries: Vec<(Entry, Vec<u8>)> = entries
            .into_iter()
            .map(|e| {
                let mut b = serialize(&e);
                b.extend_from_slice(&manifest::chain_bytes(&e.chain));
                (e, b)
            })
            .collect();
        let entry_leaves: Vec<Hash> = entries.iter().map(|(_, b)| Entry::leaf(b)).collect();
        let m_t = rt_range_proof(&entry_leaves, 0, 1).0;
        Self { weights, chunk_leaves, entries, entry_leaves, r_t, m_t }
    }

    pub fn tier(&self) -> TierCtx {
        TierCtx {
            r_t: self.r_t,
            n_chunks: self.chunk_leaves.len() as u64,
            m_t: self.m_t,
            n_entries: self.entries.len() as u64,
            n_products: N_PRODUCTS as u64,
        }
    }

    pub fn chain_ctx(&self) -> ChainCtx {
        ChainCtx { n_layers: N_LAYERS as u32, layer_base: LAYERS as u32, embd_entry: EMBD as u32, eps: EPS, embd_scale: 1.0 }
    }

    pub(crate) fn entry(&self, i: usize) -> &Entry {
        &self.entries[i].0
    }

    fn tensor(&self, i: usize) -> &[u8] {
        let e = self.entry(i);
        let off = e.chunk_base as usize * 32;
        &self.weights[off..off + e.n_bytes as usize]
    }

    pub(crate) fn row(&self, i: usize, r: usize) -> &[u8] {
        let rb = self.entry(i).row_bytes as usize;
        &self.tensor(i)[r * rb..(r + 1) * rb]
    }

    fn f32_tensor(&self, i: usize) -> Vec<f32> {
        self.tensor(i).chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
    }

    fn matvec(&self, i: usize, x: &[u8]) -> Vec<f64> {
        (0..self.entry(i).m as usize)
            .map(|r| dot::canonical_dot(quant::Q8_0, self.row(i, r), D, x, q8::MMVQ).unwrap().y_hat)
            .collect()
    }

    fn embedding(&self, tok: u32) -> Vec<f32> {
        quant::dequant_row(quant::Q8_0, self.row(EMBD, tok as usize), D).unwrap()
    }

    /// A stage-2 trace of an honest pass over `tokens`, one position per token.
    pub fn trace(&self, tokens: &[u32], n_prompt: u32) -> Trace {
        let np = N_PRODUCTS as u64;
        let n_leaves = chain::n_pos_leaves(np, N_LAYERS as u64) as usize;
        let as_f32 = |v: &[f64]| -> Vec<f32> { v.iter().map(|&a| a as f32).collect() };
        let bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|a| a.to_le_bytes()).collect() };
        let silu = |g: f64| g / (1.0 + libm::exp(-g));

        let empty = |leaves: &mut [Vec<u8>], r: usize| {
            leaves[2 * r].clear();
            leaves[2 * r + 1].clear();
        };

        let mut positions = Vec::with_capacity(tokens.len());
        for (t, &tok) in tokens.iter().enumerate() {
            let mut leaves = vec![Vec::new(); n_leaves];
            let mut h = self.embedding(tok);
            for l in 0..N_LAYERS {
                let x_in = if l == 0 {
                    q8_1_row(&expected_input(&h, &self.f32_tensor(attn_norm(l)), EPS, 1.0))
                } else {
                    q8_1_row(&chain::rms_norm(&h, &self.f32_tensor(attn_norm(l)), EPS))
                };
                let y_q = self.matvec(rank(l, ATTN_Q), &x_in);
                let mix: Vec<f64> = (0..D).map(|i| y_q[i % Q_M]).collect();
                let x_ao = q8_1_row(&mix);
                let y_ao = as_f32(&self.matvec(rank(l, ATTN_OUTPUT), &x_ao));
                let r: Vec<f32> = y_ao.iter().zip(&h).map(|(a, b)| a + b).collect();

                let x_f = q8_1_row(&chain::rms_norm(&r, &self.f32_tensor(ffn_norm(l)), EPS));
                let g = self.matvec(rank(l, FFN_GATE), &x_f);
                let u = self.matvec(rank(l, FFN_UP), &x_f);
                let hglu: Vec<f32> = g.iter().zip(&u).map(|(&a, &b)| (silu(a) * b) as f32).collect();
                let x_d = q8_1_row(&hglu.iter().map(|&a| a as f64).collect::<Vec<_>>());
                let y_d = as_f32(&self.matvec(rank(l, FFN_DOWN), &x_d));
                let out: Vec<f32> = y_d.iter().zip(&r).map(|(a, b)| a + b).collect();

                for (p, x, y) in [
                    (ATTN_Q, &x_in, bytes(&as_f32(&y_q))),
                    (ATTN_OUTPUT, &x_ao, bytes(&y_ao)),
                    (FFN_GATE, &x_f, bytes(&hglu)),
                    (FFN_UP, &x_f, Vec::new()),
                    (FFN_DOWN, &x_d, bytes(&y_d)),
                ] {
                    leaves[2 * rank(l, p)] = x.clone();
                    leaves[2 * rank(l, p) + 1] = y;
                }
                leaves[chain::resid_leaf(np, l as u64, Resid::Attn) as usize] = bytes(&r);
                leaves[chain::resid_leaf(np, l as u64, Resid::Out) as usize] = bytes(&out);
                h = out;
            }
            if t + 1 >= n_prompt as usize {
                let x_o = q8_1_row(&chain::rms_norm(&h, &self.f32_tensor(OUTPUT_NORM), EPS));
                leaves[2 * OUTPUT + 1] = bytes(&as_f32(&self.matvec(OUTPUT, &x_o)));
                leaves[2 * OUTPUT] = x_o;
            } else {
                empty(&mut leaves, OUTPUT);
            }
            leaves[chain::meta_leaf(np, N_LAYERS as u64) as usize] = chain::meta(t as u32, 1, tok).to_vec();
            let hashes: Vec<Hash> = leaves.iter().map(|b| *blake3::hash(b).as_bytes()).collect();
            positions.push(Position { tok, leaves, hashes });
        }
        let pos_roots: Vec<Hash> = positions.iter().map(|p| b3_root(&p.hashes)).collect();
        let trace_root = b3_root(&pos_roots);
        Trace { n_prompt, positions, pos_roots, trace_root }
    }
}

impl Tier {
    pub fn entry_opening(&self, i: usize) -> EntryOpening {
        EntryOpening { index: i as u32, bytes: self.entries[i].1.clone(), path: rt_range_proof(&self.entry_leaves, i as u64, 1).1 }
    }

    fn range(&self, first: u64, last: u64) -> ChunkRange {
        ChunkRange {
            first_chunk: first,
            chunks: self.weights[first as usize * 32..(last as usize + 1) * 32].to_vec(),
            proof: rt_range_proof(&self.chunk_leaves, first, last - first + 1).1,
        }
    }

    fn bytes_range(&self, i: usize, offset: u64, len: u64) -> ChunkRange {
        let base = self.entry(i).chunk_base;
        self.range(base + offset / 32, base + (offset + len - 1) / 32)
    }

    fn weights_of(&self, i: usize) -> (EntryOpening, ChunkRange) {
        (self.entry_opening(i), self.bytes_range(i, 0, self.entry(i).n_bytes))
    }

    fn row_of(&self, i: usize, r: u64) -> (EntryOpening, ChunkRange) {
        let rb = self.entry(i).row_bytes as u64;
        (self.entry_opening(i), self.bytes_range(i, r * rb, rb))
    }
}

fn leaf_bytes(pos: &Position, i: u64) -> LeafBytes {
    LeafBytes { bytes: pos.leaves[i as usize].clone(), leaf_path: b3_path(&pos.hashes, i as usize) }
}

fn leaf_chunk(pos: &Position, i: u64, c: u64) -> LeafChunk {
    let b = &pos.leaves[i as usize];
    let n = b.len().div_ceil(B3_CHUNK).max(1);
    let chunk = b[c as usize * B3_CHUNK..((c as usize + 1) * B3_CHUNK).min(b.len())].to_vec();
    let chunk_path = if n == 1 {
        Vec::new()
    } else {
        let cvs: Vec<Hash> = (0..n).map(|k| b3_chunk_cv(&b[k * B3_CHUNK..((k + 1) * B3_CHUNK).min(b.len())], k as u64)).collect();
        b3_path(&cvs, c as usize)
    };
    LeafChunk { total_len: b.len() as u32, index: c as u32, chunk, chunk_path, leaf_path: b3_path(&pos.hashes, i as usize) }
}

fn n_chunks(len: usize) -> u64 {
    len.div_ceil(B3_CHUNK).max(1) as u64
}

impl Tier {
    /// Point `i` drawn by `seed`, with its token and input link.
    pub fn open_point(&self, trace: &Trace, seed: &Hash, i: u32) -> (Point, u32, LinkOpening) {
        let d = draw::point(seed, i);
        let np = N_PRODUCTS as u64;
        let n_pos = trace.positions.len() as u64;
        let drawn = (d[1] % np) as usize;
        let t = if drawn == OUTPUT {
            let first = trace.n_prompt as u64 - 1;
            first + d[0] % (n_pos - first)
        } else {
            d[0] % n_pos
        };
        let gate = if self.entry(drawn).glu_form == manifest::FORM_PAIR_UP { self.entry(drawn).partner as usize } else { drawn };
        let ge = self.entry(gate);
        let row = d[2] % ge.m;
        let pos = &trace.positions[t as usize];

        let pair = ge.glu_form == manifest::FORM_PAIR_GATE;
        let mut rows = vec![self.row_of(gate, row).1];
        if pair {
            rows.push(self.row_of(ge.partner as usize, row).1);
        }
        let y = &pos.leaves[2 * gate + 1];
        let yc = (row * 4 / B3_CHUNK as u64) as usize;
        let ych = leaf_chunk(pos, 2 * gate as u64 + 1, yc as u64);
        let point = Point {
            n_ubatch: 1,
            pos_path: b3_path(&trace.pos_roots, t as usize),
            meta_path: b3_path(&pos.hashes, chain::meta_leaf(np, N_LAYERS as u64) as usize),
            entry: self.entry_opening(drawn),
            partner: (drawn != gate || pair).then(|| self.entry_opening(if drawn == gate { ge.partner as usize } else { gate })),
            bias: None,
            rows,
            x: pos.leaves[2 * gate].clone(),
            x_path: b3_path(&pos.hashes, 2 * gate),
            y_total_len: y.len() as u32,
            y_chunk_index: yc as u32,
            y_chunk: ych.chunk,
            y_chunk_path: ych.chunk_path,
            y_path: b3_path(&pos.hashes, 2 * gate + 1),
        };

        let Chain::Product(pl) = ge.chain else { panic!("product without link") };
        let layer = pl.layer as u64;
        let link = match pl.link {
            manifest::LINK_NONE => LinkOpening::None,
            manifest::LINK_ATTN_IN if layer == 0 => {
                LinkOpening::Token { embd: self.row_of(EMBD, pos.tok as u64), norm: self.weights_of(pl.norm as usize) }
            }
            manifest::LINK_GLU => {
                let src = pl.src as u64;
                let c = d[3] % n_chunks(pos.leaves[2 * src as usize + 1].len());
                LinkOpening::Glu { src: self.entry_opening(src as usize), y: leaf_chunk(pos, 2 * src + 1, c) }
            }
            link => {
                let leaf = match link {
                    manifest::LINK_ATTN_IN => chain::resid_leaf(np, layer - 1, Resid::Out),
                    manifest::LINK_FFN_IN => chain::resid_leaf(np, layer, Resid::Attn),
                    _ => chain::resid_leaf(np, N_LAYERS as u64 - 1, Resid::Out),
                };
                LinkOpening::Norm { src: leaf_bytes(pos, leaf), norm: self.weights_of(pl.norm as usize) }
            }
        };
        (point, pos.tok, link)
    }
}

impl Tier {
    /// Residual point drawn by `seed`.
    pub fn open_resid(&self, trace: &Trace, seed: &Hash) -> ResidPoint {
        let d = draw::point(seed, chain::RESID_POINT);
        let np = N_PRODUCTS as u64;
        let t = d[0] % trace.positions.len() as u64;
        let sel = d[1] % (2 * N_LAYERS as u64);
        let (layer, kind) = (sel / 2, if sel % 2 == 0 { Resid::Attn } else { Resid::Out });
        let pos = &trace.positions[t as usize];
        let product = if kind == Resid::Attn { rank(layer as usize, ATTN_OUTPUT) } else { rank(layer as usize, FFN_DOWN) };
        let dst_leaf = chain::resid_leaf(np, layer, kind);
        let c = d[2] % n_chunks(pos.leaves[dst_leaf as usize].len());
        let src = match kind {
            Resid::Attn if layer == 0 => {
                let (e, r) = self.row_of(EMBD, pos.tok as u64);
                ResidSrc::Token(e, r)
            }
            Resid::Attn => ResidSrc::Chunk(leaf_chunk(pos, chain::resid_leaf(np, layer - 1, Resid::Out), c)),
            Resid::Out => ResidSrc::Chunk(leaf_chunk(pos, chain::resid_leaf(np, layer, Resid::Attn), c)),
        };
        ResidPoint {
            n_ubatch: 1,
            tok: pos.tok,
            pos_path: b3_path(&trace.pos_roots, t as usize),
            meta_path: b3_path(&pos.hashes, chain::meta_leaf(np, N_LAYERS as u64) as usize),
            layer: self.entry_opening(LAYERS + layer as usize),
            product: self.entry_opening(product),
            dst: leaf_chunk(pos, dst_leaf, c),
            src,
            add: ResidAdd::Chunk(leaf_chunk(pos, 2 * product as u64 + 1, c)),
            scale: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chain::{verify_chain_point, verify_resid_point};
    use crate::verify::ResponseCtx;

    fn honest() -> (Tier, Trace) {
        let tier = Tier::new();
        let tokens: Vec<u32> = (0..12u32).map(|i| (i * 37 + 5) % VOCAB as u32).collect();
        let trace = tier.trace(&tokens, 8);
        for p in &trace.positions {
            for (i, l) in p.leaves.iter().enumerate() {
                let f32_leaf = (i < 2 * N_PRODUCTS && i % 2 == 1) || (i >= 2 * N_PRODUCTS && i < chain::meta_leaf(N_PRODUCTS as u64, N_LAYERS as u64) as usize);
                if f32_leaf {
                    assert!(l.chunks_exact(4).all(|c| f32::from_le_bytes(c.try_into().unwrap()).is_finite()), "leaf {i}");
                }
            }
        }
        (tier, trace)
    }

    fn resp(trace: &Trace) -> ResponseCtx {
        ResponseCtx { trace_root: trace.trace_root, n_prompt: trace.n_prompt, n_pos: trace.positions.len() as u32 }
    }

    #[test]
    fn honest_trace_passes_every_draw() {
        let (tier, trace) = honest();
        let (tc, cc, rc) = (tier.tier(), tier.chain_ctx(), resp(&trace));
        let mut links = [0usize; 5];
        for s in 0..64u32 {
            let seed = *blake3::hash(&s.to_le_bytes()).as_bytes();
            for i in 0..draw::POINTS {
                let (p, tok, link) = tier.open_point(&trace, &seed, i);
                let kind = match &link {
                    LinkOpening::None => 0,
                    LinkOpening::Token { .. } => 1,
                    LinkOpening::Norm { .. } => 2,
                    LinkOpening::Glu { .. } => 3,
                };
                links[kind] += 1;
                verify_chain_point(&tc, &cc, &rc, &seed, i, &p, tok, &link).unwrap_or_else(|e| panic!("seed {s} point {i}: {e:?}"));
            }
            verify_resid_point(&tc, &cc, &rc, &seed, &tier.open_resid(&trace, &seed)).unwrap_or_else(|e| panic!("seed {s} resid: {e:?}"));
        }
        assert!(links[..4].iter().all(|&n| n > 0), "{links:?}");
    }

    #[test]
    fn stage2_opening_round_trips() {
        use crate::opening::{Opening, PrefixPoint};
        let (tier, trace) = honest();
        let seed = [7u8; 32];
        let (points, links): (Vec<_>, Vec<_>) = (0..draw::POINTS)
            .map(|i| {
                let (p, tok, link) = tier.open_point(&trace, &seed, i);
                (p, (tok, link))
            })
            .unzip();
        let o = Opening {
            response_txid: [3; 32],
            points,
            prefix: Some(PrefixPoint::default()),
            links,
            resid: Some(tier.open_resid(&trace, &seed)),
        };
        let body = o.encode();
        assert_eq!(body[0], crate::opening::VERSION_CHAIN);
        assert_eq!(Opening::decode(&body), Some(o));
    }

    #[test]
    fn a_changed_residual_is_refused() {
        let (tier, mut trace) = honest();
        let leaf = chain::resid_leaf(N_PRODUCTS as u64, 0, Resid::Attn) as usize;
        for pos in &mut trace.positions {
            let v = f32::from_le_bytes(pos.leaves[leaf][..4].try_into().unwrap()) + 0.5;
            pos.leaves[leaf][..4].copy_from_slice(&v.to_le_bytes());
            pos.hashes[leaf] = *blake3::hash(&pos.leaves[leaf]).as_bytes();
        }
        trace.pos_roots = trace.positions.iter().map(|p| b3_root(&p.hashes)).collect();
        trace.trace_root = b3_root(&trace.pos_roots);
        let (tc, cc, rc) = (tier.tier(), tier.chain_ctx(), resp(&trace));
        let mut refused = 0;
        for s in 0..64u32 {
            let seed = *blake3::hash(&s.to_le_bytes()).as_bytes();
            let d = draw::point(&seed, chain::RESID_POINT);
            let hits = d[1] % 4 < 2 && d[2] % 2 == 0;
            let r = verify_resid_point(&tc, &cc, &rc, &seed, &tier.open_resid(&trace, &seed));
            if hits {
                assert_eq!(r, Err(crate::verify::Fail::Mismatch), "seed {s}");
                refused += 1;
            }
        }
        assert!(refused > 0);
    }
}
