//! Synthetic tier and responses for tests.

use crate::dot;
use crate::draw;
use crate::f16;
use crate::manifest::{self, Entry};
use crate::merkle::{Hash, b3_path, b3_root, rt_range_proof};
use crate::opening::{ChunkRange, EntryOpening, Opening, Point};
use crate::q8;
use crate::quant;
use crate::verify::TierCtx;

const K: usize = 512;
const M: usize = 4;
const N_UBATCH: u32 = 1;

/// One Q8_0 product of `M` rows of `K` weights.
pub struct Model {
    weights: Vec<u8>,
    chunk_leaves: Vec<Hash>,
    entry_bytes: Vec<u8>,
    entry: Entry,
    r_t: Hash,
    m_t: Hash,
}

/// A traced response over the synthetic model.
pub struct Response {
    pub trace_root: Hash,
    pub n_prompt: u32,
    pub n_pos: u32,
    x: Vec<Vec<u8>>,
    y: Vec<Vec<u8>>,
    pos_leaves: Vec<Vec<Hash>>,
    pos_roots: Vec<Hash>,
}

fn byte(seed: u64, i: usize) -> u8 {
    blake3::hash(&[seed.to_le_bytes(), (i as u64).to_le_bytes()].concat()).as_bytes()[0]
}

impl Model {
    pub fn new() -> Self {
        let (bb, bv) = quant::block_size(quant::Q8_0).unwrap();
        let row_bytes = K / bv * bb;
        let mut weights = Vec::with_capacity(M * row_bytes);
        for r in 0..M {
            for b in 0..K / bv {
                weights.extend_from_slice(&f16::from_f32(0.01 + 0.001 * (r * 7 + b) as f32).to_le_bytes());
                for j in 0..bv {
                    weights.push(byte(1, (r * K / bv + b) * bv + j));
                }
            }
        }
        let chunk_leaves: Vec<Hash> = weights.chunks_exact(32).map(|c| *blake3::hash(c).as_bytes()).collect();
        let r_t = rt_range_proof(&chunk_leaves, 0, 1).0;
        let entry = Entry {
            name: "blk.0.ffn_down.weight".into(),
            wtype: quant::Q8_0,
            k: K as u64,
            m: M as u64,
            n_exp: 1,
            chunk_base: 0,
            n_bytes: weights.len() as u64,
            row_bytes: row_bytes as u32,
            role: manifest::ROLE_PRODUCT,
            glu_op: 0,
            glu_form: manifest::FORM_NONE,
            partner: manifest::NONE,
            bias: manifest::NONE,
        };
        let mut entry_bytes = Vec::new();
        entry_bytes.extend_from_slice(&(entry.name.len() as u16).to_le_bytes());
        entry_bytes.extend_from_slice(entry.name.as_bytes());
        entry_bytes.extend_from_slice(&entry.wtype.to_le_bytes());
        entry_bytes.extend_from_slice(&entry.k.to_le_bytes());
        entry_bytes.extend_from_slice(&entry.m.to_le_bytes());
        entry_bytes.extend_from_slice(&entry.n_exp.to_le_bytes());
        entry_bytes.extend_from_slice(&entry.chunk_base.to_le_bytes());
        entry_bytes.extend_from_slice(&entry.n_bytes.to_le_bytes());
        entry_bytes.extend_from_slice(&entry.row_bytes.to_le_bytes());
        entry_bytes.extend_from_slice(&[entry.role, entry.glu_op, entry.glu_form]);
        entry_bytes.extend_from_slice(&entry.partner.to_le_bytes());
        entry_bytes.extend_from_slice(&entry.bias.to_le_bytes());
        let m_t = rt_range_proof(&[Entry::leaf(&entry_bytes)], 0, 1).0;
        Self { weights, chunk_leaves, entry_bytes, entry, r_t, m_t }
    }

    pub fn tier(&self) -> TierCtx {
        TierCtx { r_t: self.r_t, n_chunks: self.chunk_leaves.len() as u64, m_t: self.m_t, n_entries: 1, n_products: 1 }
    }

    fn row(&self, r: usize) -> &[u8] {
        let rb = self.entry.row_bytes as usize;
        &self.weights[r * rb..(r + 1) * rb]
    }

    /// A response of `n_pos` positions; `honest == false` shifts every output.
    pub fn response(&self, n_prompt: u32, n_pos: u32, honest: bool) -> Response {
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut pos_leaves = Vec::new();
        for t in 0..n_pos as usize {
            let mut leaf = Vec::with_capacity(K / 32 * 36);
            for b in 0..K / 32 {
                let qs: Vec<u8> = (0..32).map(|j| byte(2 + t as u64, b * 32 + j)).collect();
                let d = 0.02f32;
                let s: f32 = qs.iter().map(|&q| (q as i8) as f32 * d).sum();
                leaf.extend_from_slice(&f16::from_f32(d).to_le_bytes());
                leaf.extend_from_slice(&f16::from_f32(s).to_le_bytes());
                leaf.extend_from_slice(&qs);
            }
            let mut out = Vec::with_capacity(M * 4);
            for r in 0..M {
                let rc = dot::canonical_dot(quant::Q8_0, self.row(r), K, &leaf, q8::MMVQ).unwrap();
                let v = rc.y_hat as f32 + if honest { 0.0 } else { 1.0 + rc.scale as f32 };
                out.extend_from_slice(&v.to_le_bytes());
            }
            let mut meta = [0u8; 8];
            meta[..4].copy_from_slice(&(t as u32).to_le_bytes());
            meta[4..].copy_from_slice(&N_UBATCH.to_le_bytes());
            pos_leaves.push(vec![*blake3::hash(&leaf).as_bytes(), *blake3::hash(&out).as_bytes(), *blake3::hash(&meta).as_bytes()]);
            x.push(leaf);
            y.push(out);
        }
        let pos_roots: Vec<Hash> = pos_leaves.iter().map(|l| b3_root(l)).collect();
        Response { trace_root: b3_root(&pos_roots), n_prompt, n_pos, x, y, pos_leaves, pos_roots }
    }

    /// The opening of `resp` under `seed`.
    pub fn open(&self, resp: &Response, txid: Hash, seed: &Hash) -> Opening {
        let points = (0..draw::POINTS)
            .map(|i| {
                let d = draw::point(seed, i);
                let t = (d[0] % resp.n_pos as u64) as usize;
                let row = d[2] % M as u64;
                let rb = self.entry.row_bytes as u64;
                let first = row * rb / 32;
                let last = (row * rb + rb - 1) / 32;
                let proof = rt_range_proof(&self.chunk_leaves, first, last - first + 1).1;
                Point {
                    n_ubatch: N_UBATCH,
                    pos_path: b3_path(&resp.pos_roots, t),
                    meta_path: b3_path(&resp.pos_leaves[t], 2),
                    entry: EntryOpening { index: 0, bytes: self.entry_bytes.clone(), path: Vec::new() },
                    partner: None,
                    bias: None,
                    rows: vec![ChunkRange {
                        first_chunk: first,
                        chunks: self.weights[first as usize * 32..(last as usize + 1) * 32].to_vec(),
                        proof,
                    }],
                    x: resp.x[t].clone(),
                    x_path: b3_path(&resp.pos_leaves[t], 0),
                    y_total_len: (M * 4) as u32,
                    y_chunk_index: 0,
                    y_chunk: resp.y[t].clone(),
                    y_chunk_path: Vec::new(),
                    y_path: b3_path(&resp.pos_leaves[t], 1),
                }
            })
            .collect();
        Opening { response_txid: txid, points }
    }
}

impl Default for Model {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verify::{ResponseCtx, verify_point};

    #[test]
    fn synthetic_openings_verify() {
        let m = Model::new();
        let tier = m.tier();
        for honest in [true, false] {
            let r = m.response(3, 9, honest);
            let ctx = ResponseCtx { trace_root: r.trace_root, n_prompt: r.n_prompt, n_pos: r.n_pos };
            for s in 0..32u8 {
                let seed = [s; 32];
                let o = Opening::decode(&m.open(&r, [5; 32], &seed).encode()).unwrap();
                for (i, p) in o.points.iter().enumerate() {
                    let res = verify_point(&tier, &ctx, &seed, i as u32, p);
                    assert_eq!(res.is_ok(), honest, "seed {s} point {i}: {res:?}");
                }
            }
        }
    }
}
