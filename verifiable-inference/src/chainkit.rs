//! Two-layer synthetic tier and stage-2 traces for tests.

use crate::chain::{self, ChainCtx, Resid};
use crate::dot;
use crate::manifest::{self, Chain, Entry, LayerLinks, ProductLink};
use crate::draw;
use crate::merkle::{B3_CHUNK, Hash, b3_chunk_cv, b3_path, b3_root, rt_range_proof};
use crate::opening::{ChunkRange, EntryOpening, LeafBytes, LeafChunk, LinkOpening, Point, PrefixPoint, ResidAdd, ResidPoint, ResidSrc};
use crate::prefix::{self, PrefixCtx};
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

const PER_LAYER: usize = 7;
const ATTN_Q: usize = 0;
const ATTN_OUTPUT: usize = 1;
const FFN_DOWN: usize = 2;
const FFN_GATE: usize = 3;
const FFN_UP: usize = 4;
const SSM_F_A: usize = 5;
const SSM_F_B: usize = 6;
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
                    ATTN_Q => ("attn_k", Q_M),
                    FFN_DOWN => ("ffn_down", D),
                    FFN_GATE => ("ffn_gate", D),
                    FFN_UP => ("ffn_up", D),
                    SSM_F_A => ("ssm_f_a", D),
                    _ => ("ssm_f_b", D),
                };
                let (off, n) = push(&mut weights, q8_0_rows(seed + p as u64, m, 0.0004));
                let mut e = base(format!("blk.{l}.{name}.weight"), manifest::ROLE_PRODUCT, quant::Q8_0, m, off, n, row_bytes);
                e.chain = match p {
                    ATTN_OUTPUT => link(l, manifest::LINK_NONE, none, none),
                    ATTN_Q => link(l, manifest::LINK_ATTN_IN, attn_norm(l), none),
                    FFN_DOWN => link(l, manifest::LINK_GLU, none, rank(l, FFN_GATE)),
                    SSM_F_A => link(l, manifest::LINK_ATTN_IN, attn_norm(l), none),
                    SSM_F_B => link(l, manifest::LINK_DIRECT, none, rank(l, SSM_F_A)),
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
                moe: None,
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
        ChainCtx {
            n_layers: N_LAYERS as u32,
            layer_base: LAYERS as u32,
            embd_entry: EMBD as u32,
            eps: EPS,
            embd_scale: 1.0,
            trim_last: false,
            n_used: 0,
        }
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
        let n_leaves = chain::n_pos_leaves(np, N_LAYERS as u64, 0) as usize;
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
                let y_fa = as_f32(&self.matvec(rank(l, SSM_F_A), &x_in));
                let x_fb = q8_1_row(&y_fa.iter().map(|&a| a as f64).collect::<Vec<_>>());
                let y_fb = as_f32(&self.matvec(rank(l, SSM_F_B), &x_fb));
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
                    (SSM_F_A, &x_in, bytes(&y_fa)),
                    (SSM_F_B, &x_fb, bytes(&y_fb)),
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
            leaves[chain::meta_leaf(np, N_LAYERS as u64, 0) as usize] = chain::meta(t as u32, 1, tok).to_vec();
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


    fn norm_fragment_of(&self, i: usize, seed: &Hash, point: u32) -> (EntryOpening, ChunkRange) {
        let k = self.entry(i).k;
        let (b0, n) = chain::norm_fragment(seed, point, k);
        let len = (n * 32).min(k - b0 * 32);
        (self.entry_opening(i), self.bytes_range(i, 4 * b0 * 32, 4 * len))
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
            meta_path: b3_path(&pos.hashes, chain::meta_leaf(np, N_LAYERS as u64, 0) as usize),
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
                LinkOpening::Token { embd: self.row_of(EMBD, pos.tok as u64), norm: self.norm_fragment_of(pl.norm as usize, seed, i) }
            }
            manifest::LINK_GLU | manifest::LINK_DIRECT => {
                let src = pl.src as u64;
                let c = d[3] % n_chunks(pos.leaves[2 * src as usize + 1].len());
                LinkOpening::Source { src: self.entry_opening(src as usize), y: leaf_chunk(pos, 2 * src + 1, c), ids: None }
            }
            link => {
                let leaf = match link {
                    manifest::LINK_ATTN_IN => chain::resid_leaf(np, layer - 1, Resid::Out),
                    manifest::LINK_FFN_IN => chain::resid_leaf(np, layer, Resid::Attn),
                    _ => chain::resid_leaf(np, N_LAYERS as u64 - 1, Resid::Out),
                };
                LinkOpening::Norm { src: leaf_bytes(pos, leaf), norm: self.norm_fragment_of(pl.norm as usize, seed, i) }
            }
        };
        (point, pos.tok, link)
    }
}

impl Tier {
    /// Prefix context of the synthetic tier: identity byte table.
    pub fn prefix_ctx(&self, offset: u32) -> PrefixCtx {
        PrefixCtx {
            table: std::array::from_fn(|b| b as u32),
            offset,
            embd_entry: EMBD as u32,
            norm_entry: attn_norm(0) as u32,
            eps: EPS,
            embd_scale: 1.0,
        }
    }

    /// Stage-2 prefix point drawn by `seed` at `offset`.
    pub fn open_prefix(&self, trace: &Trace, seed: &Hash, offset: u32) -> PrefixPoint {
        let np = N_PRODUCTS as u64;
        let t = prefix::position(seed, offset) as usize;
        let pos = &trace.positions[t];
        let k = self.entry(attn_norm(0)).k;
        let (b0, n) = chain::norm_fragment(seed, prefix::PREFIX_POINT, k);
        PrefixPoint {
            n_ubatch: 1,
            pos_path: b3_path(&trace.pos_roots, t),
            meta_path: b3_path(&pos.hashes, chain::meta_leaf(np, N_LAYERS as u64, 0) as usize),
            entry: self.entry_opening(0),
            x: pos.leaves[0].clone(),
            x_path: b3_path(&pos.hashes, 0),
            embd: self.row_of(EMBD, pos.tok as u64),
            norm: (self.entry_opening(attn_norm(0)), self.bytes_range(attn_norm(0), 4 * b0 * 32, 4 * (n * 32).min(k - b0 * 32))),
        }
    }

    /// Residual point drawn by `seed`.
    pub fn open_resid(&self, trace: &Trace, seed: &Hash) -> ResidPoint {
        self.open_resid_with(&self.chain_ctx(), trace, seed)
    }

    /// Residual point drawn by `seed` under `ctx`.
    pub fn open_resid_with(&self, ctx: &ChainCtx, trace: &Trace, seed: &Hash) -> ResidPoint {
        let d = draw::point(seed, chain::RESID_POINT);
        let np = N_PRODUCTS as u64;
        let resp = crate::verify::ResponseCtx { trace_root: trace.trace_root, n_prompt: trace.n_prompt, n_pos: trace.positions.len() as u32 };
        let sel = d[1] % (2 * N_LAYERS as u64);
        let t = chain::resid_position(ctx, &resp, sel / 2, d[0]).expect("residual position");
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
            meta_path: b3_path(&pos.hashes, chain::meta_leaf(np, N_LAYERS as u64, 0) as usize),
            layer: self.entry_opening(LAYERS + layer as usize),
            product: self.entry_opening(product),
            dst: leaf_chunk(pos, dst_leaf, c),
            src,
            add: ResidAdd::Chunk(leaf_chunk(pos, 2 * product as u64 + 1, c)),
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
                let f32_leaf = (i < 2 * N_PRODUCTS && i % 2 == 1) || (i >= 2 * N_PRODUCTS && i < chain::meta_leaf(N_PRODUCTS as u64, N_LAYERS as u64, 0) as usize);
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
                    LinkOpening::Source { .. } => 3,
                };
                links[kind] += 1;
                verify_chain_point(&tc, &cc, &rc, &seed, i, &p, tok, &link, None).unwrap_or_else(|e| panic!("seed {s} point {i}: {e:?}"));
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
            experts: vec![None; links.len()],
            links,
            resid: Some(tier.open_resid(&trace, &seed)),
        };
        let body = o.encode();
        assert_eq!(body[0], crate::opening::VERSION_CHAIN);
        assert_eq!(Opening::decode(&body), Some(o));
    }

    #[test]
    fn a_zeroed_input_is_refused() {
        let (tier, mut trace) = honest();
        let r = rank(1, ATTN_Q);
        for pos in &mut trace.positions {
            let x = vec![0u8; pos.leaves[2 * r].len()];
            let y = vec![0u8; pos.leaves[2 * r + 1].len()];
            pos.leaves[2 * r] = x;
            pos.leaves[2 * r + 1] = y;
            pos.hashes[2 * r] = *blake3::hash(&pos.leaves[2 * r]).as_bytes();
            pos.hashes[2 * r + 1] = *blake3::hash(&pos.leaves[2 * r + 1]).as_bytes();
        }
        trace.pos_roots = trace.positions.iter().map(|p| b3_root(&p.hashes)).collect();
        trace.trace_root = b3_root(&trace.pos_roots);
        let (tc, cc, rc) = (tier.tier(), tier.chain_ctx(), resp(&trace));
        let mut refused = 0;
        for s in 0..256u32 {
            let seed = *blake3::hash(&s.to_le_bytes()).as_bytes();
            for i in 0..draw::POINTS {
                if draw::point(&seed, i)[1] % N_PRODUCTS as u64 != r as u64 {
                    continue;
                }
                let (p, tok, link) = tier.open_point(&trace, &seed, i);
                assert!(verify_chain_point(&tc, &cc, &rc, &seed, i, &p, tok, &link, None).is_err(), "seed {s} point {i}");
                refused += 1;
            }
        }
        assert!(refused > 0);
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

    #[test]
    fn a_trimmed_last_layer_adds_its_output_only_product() {
        let (mut tier, trace) = honest();
        let down = rank(N_LAYERS - 1, FFN_DOWN);
        let (e, bytes) = &mut tier.entries[down];
        e.role = manifest::ROLE_OUTPUT;
        let nl = u16::from_le_bytes([bytes[0], bytes[1]]) as usize;
        bytes[2 + nl + 44] = manifest::ROLE_OUTPUT;
        tier.entry_leaves = tier.entries.iter().map(|(_, b)| Entry::leaf(b)).collect();
        tier.m_t = rt_range_proof(&tier.entry_leaves, 0, 1).0;
        let (tc, rc) = (tier.tier(), resp(&trace));
        let trimmed = ChainCtx { trim_last: true, ..tier.chain_ctx() };
        let (mut ok, mut refused) = (0, 0);
        for s in 0..256u32 {
            let seed = *blake3::hash(&s.to_le_bytes()).as_bytes();
            let sel = draw::point(&seed, chain::RESID_POINT)[1] % (2 * N_LAYERS as u64);
            if sel != 2 * N_LAYERS as u64 - 1 {
                continue;
            }
            let r = verify_resid_point(&tc, &trimmed, &rc, &seed, &tier.open_resid_with(&trimmed, &trace, &seed));
            assert_eq!(r, Ok(()), "seed {s}");
            ok += 1;
            let untrimmed = tier.chain_ctx();
            let r = verify_resid_point(&tc, &untrimmed, &rc, &seed, &tier.open_resid_with(&untrimmed, &trace, &seed));
            assert_eq!(r, Err(crate::verify::Fail::Manifest), "seed {s}");
            refused += 1;
        }
        assert!(ok > 0 && refused > 0);
    }
}

#[cfg(test)]
mod expert_tests {
    use super::*;
    use crate::opening::ExpertIds;
    use crate::verify::{PosShape, ResponseCtx, verify_point_at};

    const N_EXP: usize = 4;
    const N_USED: usize = 2;
    const M: usize = 128;

    #[test]
    fn expert_product_point_verifies() {
        let w = q8_0_rows(77, N_EXP * M, 0.0004);
        let chunk_leaves: Vec<Hash> = w.chunks_exact(32).map(|c| *blake3::hash(c).as_bytes()).collect();
        let r_t = rt_range_proof(&chunk_leaves, 0, 1).0;
        let rb = D / 32 * 34;
        let e = Entry {
            name: "blk.0.ffn_down_exps.weight".into(),
            wtype: quant::Q8_0,
            k: D as u64,
            m: M as u64,
            n_exp: N_EXP as u32,
            chunk_base: 0,
            n_bytes: w.len() as u64,
            row_bytes: rb as u32,
            role: manifest::ROLE_PRODUCT,
            glu_op: 0,
            glu_form: manifest::FORM_NONE,
            partner: manifest::NONE,
            bias: manifest::NONE,
            chain: Chain::None,
        };
        let eb = serialize(&e);
        let m_t = rt_range_proof(&[Entry::leaf(&eb)], 0, 1).0;
        let tier = TierCtx { r_t, n_chunks: chunk_leaves.len() as u64, m_t, n_entries: 1, n_products: 1 };

        let ids: [u32; N_USED] = [3, 1];
        let xs: Vec<Vec<u8>> = (0..N_USED).map(|s| q8_1_row(&(0..D).map(|i| ((i * 7 + s * 13) % 23) as f64 * 0.01 - 0.1).collect::<Vec<_>>())).collect();
        let mut y = Vec::new();
        for (s, &id) in ids.iter().enumerate() {
            for r in 0..M {
                let row = &w[(id as usize * M + r) * rb..(id as usize * M + r + 1) * rb];
                let v = dot::canonical_dot(quant::Q8_0, row, D, &xs[s], q8::MMVQ).unwrap().y_hat as f32;
                y.extend_from_slice(&v.to_le_bytes());
            }
        }
        for &id in &ids {
            y.extend_from_slice(&id.to_le_bytes());
        }
        let x: Vec<u8> = xs.concat();
        let meta = [0u8, 0, 0, 0, 1, 0, 0, 0];
        let leaves = [*blake3::hash(&x).as_bytes(), *blake3::hash(&y).as_bytes(), *blake3::hash(&meta).as_bytes()];
        let pos_root = b3_root(&leaves);
        let resp = ResponseCtx { trace_root: b3_root(&[pos_root]), n_prompt: 1, n_pos: 1 };
        let shape = PosShape { n_leaves: 3, meta_index: 2, tok: None, n_used: N_USED as u32 };
        let pos = Position { tok: 0, leaves: vec![x.clone(), y.clone(), meta.to_vec()], hashes: leaves.to_vec() };

        let mut seen = [false; N_USED];
        for s in 0..32u32 {
            let seed = *blake3::hash(&s.to_le_bytes()).as_bytes();
            let d = draw::point(&seed, 0);
            let (slot, row) = ((d[3] >> 32) % N_USED as u64, d[2] % M as u64);
            seen[slot as usize] = true;
            let exp = ids[slot as usize] as u64;
            let off = (exp * M as u64 + row) * rb as u64;
            let y_off = (slot * M as u64 + row) * 4;
            let yc = leaf_chunk(&pos, 1, y_off / B3_CHUNK as u64);
            let first = leaf_chunk(&pos, 1, (N_USED * M * 4 / B3_CHUNK) as u64);
            let p = Point {
                n_ubatch: 1,
                pos_path: Vec::new(),
                meta_path: b3_path(&leaves, 2),
                entry: EntryOpening { index: 0, bytes: eb.clone(), path: Vec::new() },
                partner: None,
                bias: None,
                rows: vec![ChunkRange {
                    first_chunk: off / 32,
                    chunks: w[(off / 32 * 32) as usize..((off + rb as u64 - 1) / 32 * 32 + 32) as usize].to_vec(),
                    proof: rt_range_proof(&chunk_leaves, off / 32, (off + rb as u64 - 1) / 32 - off / 32 + 1).1,
                }],
                x: x.clone(),
                x_path: b3_path(&leaves, 0),
                y_total_len: y.len() as u32,
                y_chunk_index: yc.index,
                y_chunk: yc.chunk,
                y_chunk_path: yc.chunk_path,
                y_path: b3_path(&leaves, 1),
            };
            let e_ids = ExpertIds { chunk: first.chunk, chunk_path: first.chunk_path };
            verify_point_at(&tier, &resp, &seed, 0, &p, &shape, Some(&e_ids)).unwrap_or_else(|e| panic!("seed {s}: {e:?}"));
            assert!(verify_point_at(&tier, &resp, &seed, 0, &p, &shape, None).is_err());
        }
        assert!(seen.iter().all(|&b| b));
    }
}

#[cfg(test)]
mod moe_tests {
    use super::*;
    use crate::chain::{route_weights, verify_resid_point};
    use crate::opening::{ExpertIds, MoeAdd};
    use crate::verify::ResponseCtx;

    const E: usize = 16;
    const U: usize = 4;
    const SCALE: f32 = 2.5;

    fn vecf(seed: u64, n: usize, a: f32) -> Vec<f32> {
        (0..n).map(|i| a * (blake3::hash(&[seed.to_le_bytes(), (i as u64).to_le_bytes()].concat()).as_bytes()[0] as f32 / 127.5 - 1.0)).collect()
    }

    fn bytes(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|a| a.to_le_bytes()).collect()
    }

    struct Moe {
        weights: Vec<u8>,
        chunk_leaves: Vec<Hash>,
        entries: Vec<Vec<u8>>,
        entry_leaves: Vec<Hash>,
        tier: TierCtx,
        ctx: ChainCtx,
        pos: Position,
        trace_root: Hash,
        n_ubatch: u32,
    }

    fn entry(name: &str, role: u8, k: usize, m: usize, n_exp: u32, off: usize, n: usize, rb: usize, chain: Chain) -> Vec<u8> {
        let e = Entry {
            name: name.into(),
            wtype: if matches!(role, manifest::ROLE_PRODUCT | manifest::ROLE_OUTPUT) { quant::Q8_0 } else { 0 },
            k: k as u64,
            m: m as u64,
            n_exp,
            chunk_base: (off / 32) as u64,
            n_bytes: n as u64,
            row_bytes: rb as u32,
            role,
            glu_op: 0,
            glu_form: manifest::FORM_NONE,
            partner: manifest::NONE,
            bias: manifest::NONE,
            chain,
        };
        let mut b = serialize(&e);
        b.extend_from_slice(&manifest::chain_bytes(&e.chain));
        b
    }

    fn build(order_swap: bool, bump_slot: Option<usize>, batch: u32) -> Moe {
        let router = vecf(1, E * D, 0.05);
        let bias = vecf(2, E, 0.01);
        let norm: Vec<f32> = (0..D).map(|i| 1.0 + 0.001 * (i % 7) as f32).collect();
        let mut weights = bytes(&router);
        let bias_off = weights.len();
        weights.extend_from_slice(&bytes(&bias));
        let norm_off = weights.len();
        weights.extend_from_slice(&bytes(&norm));
        let chunk_leaves: Vec<Hash> = weights.chunks_exact(32).map(|c| *blake3::hash(c).as_bytes()).collect();

        let none = manifest::NONE;
        let last = if batch > 1 { manifest::ROLE_OUTPUT } else { manifest::ROLE_PRODUCT };
        let moe = manifest::MoeLinks { down_exps: 1, router: 4, probs_bias: 5, w_scale: SCALE.to_bits(), ffn_norm: 6, gate_exps: 3 };
        let entries = vec![
            entry("blk.0.attn_output.weight", manifest::ROLE_PRODUCT, D, D, 1, 0, 0, 0, Chain::None),
            entry("blk.0.ffn_down_exps.weight", manifest::ROLE_PRODUCT, D, D, E as u32, 0, 0, 0, Chain::None),
            entry("blk.0.ffn_down_shexp.weight", last, D, D, 1, 0, 0, 0, Chain::None),
            entry("blk.0.ffn_gate_exps.weight", last, D, D, E as u32, 0, 0, 0, Chain::None),
            entry("blk.0.ffn_gate_inp.weight", manifest::ROLE_ROUTER, D, E, 1, 0, 4 * E * D, 4 * D, Chain::None),
            entry("blk.0.exp_probs_b.bias", manifest::ROLE_NORM, E, 1, 1, bias_off, 4 * E, 4 * E, Chain::None),
            entry("blk.0.ffn_norm.weight", manifest::ROLE_NORM, D, 1, 1, norm_off, 4 * D, 4 * D, Chain::None),
            entry("blk.0", manifest::ROLE_LAYER, 0, 0, 0, 0, 0, 0, Chain::Layer(LayerLinks {
                layer: 0, proj: 0, down: 2, post_attn: none, post_ffn: none, scale: none, moe: Some(moe),
            })),
        ];
        let entry_leaves: Vec<Hash> = entries.iter().map(|b| Entry::leaf(b)).collect();
        let tier = TierCtx {
            r_t: rt_range_proof(&chunk_leaves, 0, 1).0,
            n_chunks: chunk_leaves.len() as u64,
            m_t: rt_range_proof(&entry_leaves, 0, 1).0,
            n_entries: entries.len() as u64,
            n_products: 4,
        };
        let ctx = ChainCtx { n_layers: 1, layer_base: 7, embd_entry: 0, eps: EPS, embd_scale: 1.0, trim_last: batch > 1, n_used: U as u32 };

        let r = vecf(3, D, 2.0);
        let x = chain::rms_norm(&r, &norm, EPS);
        let scores: Vec<f32> = (0..E).map(|e| router[e * D..(e + 1) * D].iter().zip(&x).map(|(&w, &v)| w as f64 * v).sum::<f64>() as f32).collect();
        let sel: Vec<f64> = scores.iter().zip(&bias).map(|(&s, &b)| 1.0 / (1.0 + (-(s as f64)).exp()) + b as f64).collect();
        let mut ids: Vec<u32> = (0..E as u32).collect();
        ids.sort_by(|&a, &b| sel[b as usize].total_cmp(&sel[a as usize]));
        ids.truncate(U);
        let w = route_weights(&scores, &bias, &ids, SCALE as f64).unwrap();
        let ys: Vec<Vec<f32>> = (0..U).map(|s| vecf(10 + s as u64, D, 0.3)).collect();
        let sh = vecf(20, D, 0.3);
        let out: Vec<f32> = (0..D).map(|j| (w.iter().zip(&ys).map(|(a, y)| a * y[j] as f64).sum::<f64>() + sh[j] as f64 + r[j] as f64) as f32).collect();
        let mut y_exps: Vec<u8> = ys.iter().flat_map(|y| bytes(y)).collect();
        if let Some(s) = bump_slot {
            let v = f32::from_le_bytes(y_exps[s * D * 4..s * D * 4 + 4].try_into().unwrap()) + 0.25;
            y_exps[s * D * 4..s * D * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        let mut committed = ids.clone();
        if order_swap {
            committed.swap(0, U - 1);
        }
        for id in &committed {
            y_exps.extend_from_slice(&id.to_le_bytes());
        }
        let np = 4u64;
        let n_leaves = chain::n_pos_leaves(np, 1, 1) as usize;
        let mut leaves = vec![Vec::new(); n_leaves];
        leaves[6] = q8_1_row(&x);
        leaves[3] = y_exps;
        leaves[5] = bytes(&sh);
        leaves[chain::resid_leaf(np, 0, Resid::Attn) as usize] = bytes(&r);
        leaves[chain::resid_leaf(np, 0, Resid::Out) as usize] = bytes(&out);
        leaves[chain::router_leaf(np, 1, 0) as usize] = bytes(&scores);
        leaves[chain::meta_leaf(np, 1, 1) as usize] = chain::meta(0, batch, 7).to_vec();
        let hashes: Vec<Hash> = leaves.iter().map(|b| *blake3::hash(b).as_bytes()).collect();
        let trace_root = b3_root(&[b3_root(&hashes)]);
        Moe { weights, chunk_leaves, entries, entry_leaves, tier, ctx, pos: Position { tok: 7, leaves, hashes }, trace_root, n_ubatch: batch }
    }

    impl Moe {
        fn eo(&self, i: usize) -> EntryOpening {
            EntryOpening { index: i as u32, bytes: self.entries[i].clone(), path: rt_range_proof(&self.entry_leaves, i as u64, 1).1 }
        }

        fn range(&self, off: usize, len: usize) -> ChunkRange {
            let (first, last) = ((off / 32) as u64, ((off + len - 1) / 32) as u64);
            ChunkRange {
                first_chunk: first,
                chunks: self.weights[first as usize * 32..(last as usize + 1) * 32].to_vec(),
                proof: rt_range_proof(&self.chunk_leaves, first, last - first + 1).1,
            }
        }

        fn open(&self, seed: &Hash) -> Option<ResidPoint> {
            let d = draw::point(seed, chain::RESID_POINT);
            if d[1] % 2 != 1 {
                return None;
            }
            let np = 4u64;
            let c = d[2] % (D as u64 * 4 / B3_CHUNK as u64);
            let per = (D * 4 / B3_CHUNK) as u64;
            let row = d[3] % E as u64;
            let ids_chunk = leaf_chunk(&self.pos, 3, (U * D * 4 / B3_CHUNK) as u64);
            Some(ResidPoint {
                n_ubatch: self.n_ubatch,
                tok: 7,
                pos_path: Vec::new(),
                meta_path: b3_path(&self.pos.hashes, chain::meta_leaf(np, 1, 1) as usize),
                layer: self.eo(7),
                product: self.eo(2),
                dst: leaf_chunk(&self.pos, chain::resid_leaf(np, 0, Resid::Out), c),
                src: ResidSrc::Chunk(leaf_chunk(&self.pos, chain::resid_leaf(np, 0, Resid::Attn), c)),
                add: ResidAdd::Moe(Box::new(MoeAdd {
                    scores: leaf_bytes(&self.pos, chain::router_leaf(np, 1, 0)),
                    bias: (self.eo(5), self.range(4 * E * D, 4 * E)),
                    ids: ExpertIds { chunk: ids_chunk.chunk, chunk_path: ids_chunk.chunk_path },
                    slots: (0..U as u64).map(|s| leaf_chunk(&self.pos, 3, s * per + c)).collect(),
                    shexp: leaf_chunk(&self.pos, 5, c),
                    router_row: (self.eo(4), self.range(row as usize * 4 * D, 4 * D)),
                    gate: self.eo(3),
                    gate_x: leaf_bytes(&self.pos, 6),
                })),
            })
        }

        fn check(&self, seed: &Hash) -> Option<Result<(), crate::verify::Fail>> {
            let resp = ResponseCtx { trace_root: self.trace_root, n_prompt: 1, n_pos: 1 };
            let p = self.open(seed)?;
            Some(verify_resid_point(&self.tier, &self.ctx, &resp, seed, &p))
        }
    }

    fn seeds() -> impl Iterator<Item = Hash> {
        (0..64u32).map(|s| *blake3::hash(&s.to_le_bytes()).as_bytes())
    }

    #[test]
    fn honest_moe_layer_passes() {
        let m = build(false, None, 1);
        let n = seeds().filter_map(|s| m.check(&s).map(|r| r.unwrap_or_else(|e| panic!("{e:?}")))).count();
        assert!(n > 0);
    }

    #[test]
    fn swapped_experts_are_refused() {
        let m = build(true, None, 1);
        let results: Vec<_> = seeds().filter_map(|s| m.check(&s)).collect();
        assert!(!results.is_empty() && results.iter().all(|r| *r == Err(crate::verify::Fail::Mismatch)));
    }

    #[test]
    fn a_changed_expert_output_is_refused() {
        let m = build(false, Some(1), 1);
        let mut refused = 0;
        for s in seeds() {
            let d = draw::point(&s, chain::RESID_POINT);
            if let Some(r) = m.check(&s) {
                if d[2] % 2 == 0 {
                    assert_eq!(r, Err(crate::verify::Fail::Mismatch));
                    refused += 1;
                }
            }
        }
        assert!(refused > 0);
    }

    #[test]
    fn a_trimmed_last_layer_reads_its_expert_input_as_one_row() {
        let m = build(false, None, 32);
        let n = seeds().filter_map(|s| m.check(&s).map(|r| r.unwrap_or_else(|e| panic!("{e:?}")))).count();
        assert!(n > 0);
    }
}

#[cfg(test)]
mod head_tests {
    use super::*;
    use crate::verify::{PosShape, ResponseCtx, verify_point_at};

    const H: usize = 4;
    const M: usize = 16;

    #[test]
    fn per_head_product_point_verifies() {
        let w = q8_0_rows(91, H * M, 0.0004);
        let chunk_leaves: Vec<Hash> = w.chunks_exact(32).map(|c| *blake3::hash(c).as_bytes()).collect();
        let rb = D / 32 * 34;
        let e = Entry {
            name: "blk.3.attn_k_b.weight".into(),
            wtype: quant::Q8_0,
            k: D as u64,
            m: M as u64,
            n_exp: H as u32,
            chunk_base: 0,
            n_bytes: w.len() as u64,
            row_bytes: rb as u32,
            role: manifest::ROLE_HEADS,
            glu_op: 0,
            glu_form: manifest::FORM_NONE,
            partner: manifest::NONE,
            bias: manifest::NONE,
            chain: Chain::None,
        };
        let eb = serialize(&e);
        let tier = TierCtx {
            r_t: rt_range_proof(&chunk_leaves, 0, 1).0,
            n_chunks: chunk_leaves.len() as u64,
            m_t: rt_range_proof(&[Entry::leaf(&eb)], 0, 1).0,
            n_entries: 1,
            n_products: 1,
        };
        let xs: Vec<Vec<u8>> = (0..H).map(|h| q8_1_row(&(0..D).map(|i| ((i * 5 + h * 11) % 19) as f64 * 0.01 - 0.09).collect::<Vec<_>>())).collect();
        let mut y = Vec::new();
        for (h, xh) in xs.iter().enumerate() {
            for r in 0..M {
                let row = &w[(h * M + r) * rb..(h * M + r + 1) * rb];
                y.extend_from_slice(&(dot::canonical_dot(quant::Q8_0, row, D, xh, q8::MMVQ).unwrap().y_hat as f32).to_le_bytes());
            }
        }
        let x = xs.concat();
        let meta = [0u8, 0, 0, 0, 1, 0, 0, 0];
        let leaves = [*blake3::hash(&x).as_bytes(), *blake3::hash(&y).as_bytes(), *blake3::hash(&meta).as_bytes()];
        let resp = ResponseCtx { trace_root: b3_root(&[b3_root(&leaves)]), n_prompt: 1, n_pos: 1 };
        let shape = PosShape { n_leaves: 3, meta_index: 2, tok: None, n_used: 0 };
        let pos = Position { tok: 0, leaves: vec![x.clone(), y.clone(), meta.to_vec()], hashes: leaves.to_vec() };
        let mut seen = [false; H];
        for s in 0..48u32 {
            let seed = *blake3::hash(&s.to_le_bytes()).as_bytes();
            let d = draw::point(&seed, 0);
            let (h, row) = ((d[3] >> 32) % H as u64, d[2] % M as u64);
            seen[h as usize] = true;
            let off = (h * M as u64 + row) * rb as u64;
            let yc = leaf_chunk(&pos, 1, (h * M as u64 + row) * 4 / B3_CHUNK as u64);
            let mut p = Point {
                n_ubatch: 1,
                pos_path: Vec::new(),
                meta_path: b3_path(&leaves, 2),
                entry: EntryOpening { index: 0, bytes: eb.clone(), path: Vec::new() },
                partner: None,
                bias: None,
                rows: vec![ChunkRange {
                    first_chunk: off / 32,
                    chunks: w[(off / 32 * 32) as usize..((off + rb as u64 - 1) / 32 * 32 + 32) as usize].to_vec(),
                    proof: rt_range_proof(&chunk_leaves, off / 32, (off + rb as u64 - 1) / 32 - off / 32 + 1).1,
                }],
                x: x.clone(),
                x_path: b3_path(&leaves, 0),
                y_total_len: y.len() as u32,
                y_chunk_index: yc.index,
                y_chunk: yc.chunk,
                y_chunk_path: yc.chunk_path,
                y_path: b3_path(&leaves, 1),
            };
            verify_point_at(&tier, &resp, &seed, 0, &p, &shape, None).unwrap_or_else(|e| panic!("seed {s}: {e:?}"));
            let other = ((h + 1) % H as u64) * M as u64 + row;
            let oo = other * rb as u64;
            p.rows = vec![ChunkRange {
                first_chunk: oo / 32,
                chunks: w[(oo / 32 * 32) as usize..((oo + rb as u64 - 1) / 32 * 32 + 32) as usize].to_vec(),
                proof: rt_range_proof(&chunk_leaves, oo / 32, (oo + rb as u64 - 1) / 32 - oo / 32 + 1).1,
            }];
            assert!(verify_point_at(&tier, &resp, &seed, 0, &p, &shape, None).is_err());
        }
        assert!(seen.iter().all(|&b| b));
    }
}

#[cfg(test)]
mod block_task_tests {
    use super::*;
    use crate::block_task::{TASK_LEN, draw_seed, task_tokens, verify_block_task};
    use crate::opening::Opening;

    fn opening(tier: &Tier, trace: &Trace, block_hash: &Hash) -> Opening {
        let seed = draw_seed(block_hash);
        let (points, links): (Vec<_>, Vec<_>) = (0..draw::POINTS)
            .map(|i| {
                let (p, tok, link) = tier.open_point(trace, &seed, i);
                (p, (tok, link))
            })
            .unzip();
        Opening {
            response_txid: *block_hash,
            experts: vec![None; points.len()],
            points,
            prefix: Some(tier.open_prefix(trace, &seed, 0)),
            links,
            resid: Some(tier.open_resid(trace, &seed)),
        }
    }

    #[test]
    fn honest_block_task_passes() {
        let tier = Tier::new();
        let window_seed = [0x5Au8; 32];
        let pc = tier.prefix_ctx(0);
        let trace = tier.trace(&task_tokens(&pc.table, &window_seed), TASK_LEN);
        for s in 0..48u32 {
            let block = *blake3::hash(&s.to_le_bytes()).as_bytes();
            verify_block_task(&tier.tier(), &tier.chain_ctx(), &pc, &trace.trace_root, &window_seed, &block, &opening(&tier, &trace, &block))
                .unwrap_or_else(|e| panic!("block {s}: {e:?}"));
        }
    }

    #[test]
    fn a_task_of_another_window_is_refused() {
        let tier = Tier::new();
        let pc = tier.prefix_ctx(0);
        let trace = tier.trace(&task_tokens(&pc.table, &[0x11u8; 32]), TASK_LEN);
        for s in 0..48u32 {
            let block = *blake3::hash(&s.to_le_bytes()).as_bytes();
            let r = verify_block_task(&tier.tier(), &tier.chain_ctx(), &pc, &trace.trace_root, &[0x22u8; 32], &block, &opening(&tier, &trace, &block));
            assert!(r.is_err(), "block {s}");
        }
    }
}

