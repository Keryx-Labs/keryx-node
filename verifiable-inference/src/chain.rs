//! Stage-2 links between committed activations.

use crate::draw;
use crate::manifest::{self, Chain};
use crate::merkle::{B3_CHUNK, Hash, b3_fold, b3_leaf_from_chunk, b3_verify};
use crate::opening::{ChunkRange, EntryOpening, LeafBytes, LeafChunk, LinkOpening, Point, ResidAdd, ResidPoint, ResidSrc};
use crate::opening::PrefixPoint;
use crate::prefix::{PrefixCtx, THRESHOLD, expected_input, verify_prefix_point_at};
use crate::verify::{Fail, PosShape, Report, ResponseCtx, TierCtx, open_bytes, open_entry, open_row, verify_point_at};
use crate::{q8, quant};

/// Stage byte of a trace carrying the chain leaves.
pub const STAGE_CHAIN: u8 = 2;
/// Draw index of the residual point.
pub const RESID_POINT: u32 = draw::POINTS + 1;

/// Consensus chain data of a tier.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChainCtx {
    pub n_layers: u32,
    /// Manifest index of the layer-0 `ROLE_LAYER` entry; layer `l` is at `layer_base + l`.
    pub layer_base: u32,
    pub embd_entry: u32,
    pub eps: f64,
    pub embd_scale: f64,
}
/// Relative tolerance of a residual addition, per element.
pub const EPS_RESID: f64 = 1e-6;

/// Residual leaves of a layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Resid {
    Attn,
    Out,
}

/// Leaves of a stage-2 position: `X`/`Y` per product, two residuals per layer, then the meta leaf.
pub fn n_pos_leaves(n_products: u64, n_layers: u64) -> u64 {
    2 * n_products + 2 * n_layers + 1
}

/// Leaf index of a residual of `layer`.
pub fn resid_leaf(n_products: u64, layer: u64, kind: Resid) -> u64 {
    2 * n_products + 2 * layer + if kind == Resid::Attn { 0 } else { 1 }
}

/// Leaf index of the meta leaf.
pub fn meta_leaf(n_products: u64, n_layers: u64) -> u64 {
    2 * n_products + 2 * n_layers
}

/// Meta leaf bytes: position, batch size, token consumed at the position.
pub fn meta(t: u32, n_ubatch: u32, tok: u32) -> [u8; 12] {
    let mut m = [0u8; 12];
    m[..4].copy_from_slice(&t.to_le_bytes());
    m[4..8].copy_from_slice(&n_ubatch.to_le_bytes());
    m[8..].copy_from_slice(&tok.to_le_bytes());
    m
}

/// `v * w / rms(v)` in f64.
pub fn rms_norm(v: &[f32], w: &[f32], eps: f64) -> Vec<f64> {
    expected_input(v, w, eps, 1.0)
}

/// `max |q - v / d|` over the blocks of `x` starting at block `first_block`, against `v`.
pub fn block_deviation(x: &[u8], layout: u8, k: usize, first_block: usize, v: &[f64]) -> Option<f64> {
    let blocks = q8::blocks(x, layout, k)?;
    let n = v.len().div_ceil(32);
    let vmax = v.iter().fold(0f64, |m, a| m.max(a.abs()));
    let mut worst = 0f64;
    for (b, (d, q)) in blocks.get(first_block..first_block.checked_add(n)?)?.iter().enumerate() {
        let vals = &v[b * 32..((b + 1) * 32).min(v.len())];
        if !d.is_finite() || vals.iter().any(|a| !a.is_finite()) {
            return None;
        }
        if *d == 0.0 {
            if vals.iter().any(|a| a.abs() > 1e-6 * vmax) {
                return None;
            }
            continue;
        }
        for (j, a) in vals.iter().enumerate() {
            worst = worst.max((q[j] as f64 - a / *d as f64).abs());
        }
    }
    Some(worst)
}

fn within(dev: Option<f64>) -> Result<(), Fail> {
    match dev {
        Some(d) if d <= THRESHOLD => Ok(()),
        _ => Err(Fail::Mismatch),
    }
}

/// A product input normed from a committed residual: `x = Q8(rms_norm(src) * w)`.
pub fn norm_link(x: &[u8], layout: u8, src: &[f32], w: &[f32], eps: f64) -> Result<(), Fail> {
    if src.len() != w.len() || src.is_empty() {
        return Err(Fail::Shape);
    }
    within(block_deviation(x, layout, src.len(), 0, &rms_norm(src, w, eps)))
}

/// A product input taken from the committed GLU output `h`, for the blocks covering `h` from
/// block `first_block`.
pub fn glu_link(x: &[u8], layout: u8, k: usize, first_block: usize, h: &[f32]) -> Result<(), Fail> {
    if h.is_empty() {
        return Err(Fail::Shape);
    }
    let v: Vec<f64> = h.iter().map(|&a| a as f64).collect();
    within(block_deviation(x, layout, k, first_block, &v))
}

/// A residual addition, element by element: `dst = scale * (add + src)`.
pub fn residual_link(dst: &[f32], add: &[f64], src: &[f32], scale: f64) -> Result<(), Fail> {
    if dst.is_empty() || dst.len() != add.len() || dst.len() != src.len() {
        return Err(Fail::Shape);
    }
    for ((&d, &a), &s) in dst.iter().zip(add).zip(src) {
        let s = s as f64;
        let tol = EPS_RESID * scale.abs() * (a.abs() + s.abs()) + f32::MIN_POSITIVE as f64;
        if !((d as f64 - scale * (a + s)).abs() <= tol) {
            return Err(Fail::Mismatch);
        }
    }
    Ok(())
}

/// Leaf layout of a stage-2 position consuming `tok`.
pub fn shape(tier: &TierCtx, ctx: &ChainCtx, tok: u32) -> PosShape {
    let n_layers = ctx.n_layers as u64;
    PosShape { n_leaves: n_pos_leaves(tier.n_products, n_layers), meta_index: meta_leaf(tier.n_products, n_layers), tok: Some(tok) }
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn f32_weights(tier: &TierCtx, w: &(EntryOpening, ChunkRange), index: u32, k: u64, offset: u64, n: u64) -> Result<Vec<f32>, Fail> {
    let e = open_entry(tier, &w.0, index as u64)?;
    if e.role != manifest::ROLE_NORM || e.wtype != 0 || e.k != k {
        return Err(Fail::Manifest);
    }
    Ok(f32s(&open_bytes(tier, &e, &w.1, 4 * offset, 4 * n)?))
}

fn embedding(tier: &TierCtx, ctx: &ChainCtx, w: &(EntryOpening, ChunkRange), tok: u32, k: u64) -> Result<Vec<f32>, Fail> {
    let e = open_entry(tier, &w.0, ctx.embd_entry as u64)?;
    if e.role != manifest::ROLE_EMBD || e.k != k || e.n_exp != 1 {
        return Err(Fail::Manifest);
    }
    let row = open_row(tier, &e, &w.1, tok as u64)?;
    quant::dequant_row(e.wtype, &row, k as usize).ok_or(Fail::Unsupported)
}

fn whole_leaf(pos_root: &Hash, shape: &PosShape, index: u64, l: &LeafBytes, len: u64) -> Result<Vec<f32>, Fail> {
    if l.bytes.len() as u64 != len {
        return Err(Fail::Shape);
    }
    if !b3_verify(pos_root, shape.n_leaves, index, blake3::hash(&l.bytes).as_bytes(), &l.leaf_path) {
        return Err(Fail::Trace);
    }
    Ok(f32s(&l.bytes))
}

fn chunk_leaf(c: &LeafChunk, total_len: u64, index: u64) -> Result<Hash, Fail> {
    if c.total_len as u64 != total_len || c.index as u64 != index {
        return Err(Fail::Shape);
    }
    b3_leaf_from_chunk(total_len, index, &c.chunk, &c.chunk_path).ok_or(Fail::Trace)
}

fn leaf_chunk(pos_root: &Hash, shape: &PosShape, leaf: u64, c: &LeafChunk, total_len: u64, index: u64) -> Result<Vec<f32>, Fail> {
    let h = chunk_leaf(c, total_len, index)?;
    if !b3_verify(pos_root, shape.n_leaves, leaf, &h, &c.leaf_path) {
        return Err(Fail::Trace);
    }
    Ok(f32s(&c.chunk))
}

fn n_chunks(total_len: u64) -> u64 {
    total_len.div_ceil(B3_CHUNK as u64).max(1)
}

/// Verify point `i` of a stage-2 opening and the link of its input.
#[allow(clippy::too_many_arguments)]
pub fn verify_chain_point(
    tier: &TierCtx,
    ctx: &ChainCtx,
    resp: &ResponseCtx,
    seed: &Hash,
    i: u32,
    p: &Point,
    tok: u32,
    link: &LinkOpening,
) -> Result<Report, Fail> {
    let shape = shape(tier, ctx, tok);
    let (report, o) = verify_point_at(tier, resp, seed, i, p, &shape)?;
    let Chain::Product(pl) = o.input.chain else {
        return Err(Fail::Manifest);
    };
    let (k, np, n_layers) = (o.input.k, tier.n_products, ctx.n_layers as u64);
    let layer = pl.layer as u64;
    if pl.link != manifest::LINK_HEAD && layer >= n_layers {
        return Err(Fail::Manifest);
    }
    match (pl.link, link) {
        (manifest::LINK_NONE, LinkOpening::None) => {}
        (manifest::LINK_ATTN_IN, LinkOpening::Token { embd, norm }) if layer == 0 => {
            let w = f32_weights(tier, norm, pl.norm, k, 0, k)?;
            let v = expected_input(&embedding(tier, ctx, embd, tok, k)?, &w, ctx.eps, ctx.embd_scale);
            within(block_deviation(&p.x, o.layout, k as usize, 0, &v))?;
        }
        (manifest::LINK_ATTN_IN | manifest::LINK_FFN_IN | manifest::LINK_HEAD, LinkOpening::Norm { src, norm }) => {
            let leaf = match pl.link {
                manifest::LINK_ATTN_IN if layer > 0 => resid_leaf(np, layer - 1, Resid::Out),
                manifest::LINK_FFN_IN => resid_leaf(np, layer, Resid::Attn),
                manifest::LINK_HEAD if n_layers > 0 => resid_leaf(np, n_layers - 1, Resid::Out),
                _ => return Err(Fail::Shape),
            };
            let h = whole_leaf(&o.pos_root, &shape, leaf, src, 4 * k)?;
            norm_link(&p.x, o.layout, &h, &f32_weights(tier, norm, pl.norm, k, 0, k)?, ctx.eps)?;
        }
        (manifest::LINK_GLU, LinkOpening::Glu { src, y }) => {
            let se = open_entry(tier, src, pl.src as u64)?;
            let m_out = match se.glu_form {
                manifest::FORM_PAIR_GATE => se.m,
                manifest::FORM_SPLIT => se.m / 2,
                _ => return Err(Fail::Manifest),
            };
            if m_out != k {
                return Err(Fail::Manifest);
            }
            let total = 4 * m_out;
            let c = draw::point(seed, i)[3] % n_chunks(total);
            let h = leaf_chunk(&o.pos_root, &shape, 2 * pl.src as u64 + 1, y, total, c)?;
            glu_link(&p.x, o.layout, k as usize, (c * B3_CHUNK as u64 / 128) as usize, &h)?;
        }
        _ => return Err(Fail::Shape),
    }
    Ok(report)
}

/// Verify the prefix point of a stage-2 opening.
pub fn verify_chain_prefix_point(
    tier: &TierCtx,
    ctx: &ChainCtx,
    prefix: &PrefixCtx,
    resp: &ResponseCtx,
    seed: &Hash,
    request_txid: &Hash,
    p: &PrefixPoint,
) -> Result<u64, Fail> {
    verify_prefix_point_at(tier, prefix, resp, seed, request_txid, p, |tok| shape(tier, ctx, tok))
}

/// Verify the residual point of a stage-2 opening.
pub fn verify_resid_point(tier: &TierCtx, ctx: &ChainCtx, resp: &ResponseCtx, seed: &Hash, p: &ResidPoint) -> Result<(), Fail> {
    let (np, n_layers) = (tier.n_products, ctx.n_layers as u64);
    if resp.n_pos == 0 || n_layers == 0 {
        return Err(Fail::Shape);
    }
    let d = draw::point(seed, RESID_POINT);
    let t = d[0] % resp.n_pos as u64;
    let sel = d[1] % (2 * n_layers);
    let (layer, kind) = (sel / 2, if sel % 2 == 0 { Resid::Attn } else { Resid::Out });
    let shape = shape(tier, ctx, p.tok);

    let le = open_entry(tier, &p.layer, ctx.layer_base as u64 + layer)?;
    let ll = match le.chain {
        Chain::Layer(ll) if le.role == manifest::ROLE_LAYER && ll.layer as u64 == layer => ll,
        _ => return Err(Fail::Manifest),
    };
    let rank = if kind == Resid::Attn { ll.proj } else { ll.down } as u64;
    let pe = open_entry(tier, &p.product, rank)?;
    if pe.role != manifest::ROLE_PRODUCT || pe.glu_form != manifest::FORM_NONE || pe.n_exp != 1 {
        return Err(Fail::Manifest);
    }
    let dim = pe.m;
    let total = 4 * dim;
    let c = d[2] % n_chunks(total);

    let dst_leaf = chunk_leaf(&p.dst, total, c)?;
    let pos_root = b3_fold(shape.n_leaves, resid_leaf(np, layer, kind), &dst_leaf, &p.dst.leaf_path).ok_or(Fail::Trace)?;
    if !b3_verify(&resp.trace_root, resp.n_pos as u64, t, &pos_root, &p.pos_path)
        || !b3_verify(&pos_root, shape.n_leaves, shape.meta_index, &shape.meta_hash(t, p.n_ubatch), &p.meta_path)
    {
        return Err(Fail::Trace);
    }
    let dst = f32s(&p.dst.chunk);
    let (off, n) = (c as usize * B3_CHUNK / 4, dst.len());

    let src = match (kind, &p.src) {
        (Resid::Attn, ResidSrc::Token(e, r)) if layer == 0 => {
            let row = embedding(tier, ctx, &(e.clone(), r.clone()), p.tok, dim)?;
            row[off..off + n].iter().map(|&v| (v as f64 * ctx.embd_scale) as f32).collect()
        }
        (Resid::Attn, ResidSrc::Chunk(sc)) if layer > 0 => leaf_chunk(&pos_root, &shape, resid_leaf(np, layer - 1, Resid::Out), sc, total, c)?,
        (Resid::Out, ResidSrc::Chunk(sc)) => leaf_chunk(&pos_root, &shape, resid_leaf(np, layer, Resid::Attn), sc, total, c)?,
        _ => return Err(Fail::Shape),
    };
    let post = if kind == Resid::Attn { ll.post_attn } else { ll.post_ffn };
    let add: Vec<f64> = match (post, &p.add) {
        (manifest::NONE, ResidAdd::Chunk(ac)) => leaf_chunk(&pos_root, &shape, 2 * rank + 1, ac, total, c)?.iter().map(|&v| v as f64).collect(),
        (pn, ResidAdd::Normed { y, norm }) if pn != manifest::NONE => {
            let yv = whole_leaf(&pos_root, &shape, 2 * rank + 1, y, total)?;
            rms_norm(&yv, &f32_weights(tier, norm, pn, dim, 0, dim)?, ctx.eps)[off..off + n].to_vec()
        }
        _ => return Err(Fail::Shape),
    };
    let scale = match (kind, ll.scale, &p.scale) {
        (Resid::Out, sc, Some(w)) if sc != manifest::NONE => f32_weights(tier, w, sc, 1, 0, 1)?[0] as f64,
        (_, sc, None) if kind == Resid::Attn || sc == manifest::NONE => 1.0,
        _ => return Err(Fail::Shape),
    };
    residual_link(&dst, &add, &src, scale)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::q8_1_row;

    const D: usize = 512;
    const EPS: f64 = 1e-6;

    fn vec_of(seed: u32, scale: f32) -> Vec<f32> {
        (0..D).map(|i| scale * ((((i as u32).wrapping_mul(2_654_435_761) ^ seed.wrapping_mul(40_503)) % 2001) as f32 / 1000.0 - 1.0)).collect()
    }

    fn norm_w() -> Vec<f32> {
        (0..D).map(|i| 1.0 + 0.001 * (i % 17) as f32).collect()
    }

    fn honest_input(src: &[f32]) -> Vec<u8> {
        q8_1_row(&rms_norm(src, &norm_w(), EPS))
    }

    #[test]
    fn layout_of_a_stage2_position() {
        assert_eq!(n_pos_leaves(3, 2), 11);
        assert_eq!(resid_leaf(3, 0, Resid::Attn), 6);
        assert_eq!(resid_leaf(3, 1, Resid::Out), 9);
        assert_eq!(meta_leaf(3, 2), 10);
        assert_eq!(&meta(7, 512, 42)[8..], &42u32.to_le_bytes());
    }

    #[test]
    fn honest_norm_link_passes() {
        let h = vec_of(1, 3.0);
        assert_eq!(norm_link(&honest_input(&h), q8::MMVQ, &h, &norm_w(), EPS), Ok(()));
    }

    #[test]
    fn empty_input_fails() {
        let h = vec_of(1, 3.0);
        let x = vec![0u8; D / 32 * 36];
        assert_eq!(norm_link(&x, q8::MMVQ, &h, &norm_w(), EPS), Err(Fail::Mismatch));
    }

    #[test]
    fn input_of_another_source_fails() {
        let (h, other) = (vec_of(1, 3.0), vec_of(2, 3.0));
        assert_eq!(norm_link(&honest_input(&other), q8::MMVQ, &h, &norm_w(), EPS), Err(Fail::Mismatch));
    }

    #[test]
    fn one_input_fails_at_other_positions() {
        let c = honest_input(&vec_of(9, 2.0));
        let failed = (1..=8).filter(|&s| norm_link(&c, q8::MMVQ, &vec_of(s, 3.0), &norm_w(), EPS).is_err()).count();
        assert_eq!(failed, 8);
    }

    #[test]
    fn wrong_norm_weight_fails() {
        let h = vec_of(1, 3.0);
        let w2: Vec<f32> = (0..D).map(|i| 1.0 + 0.05 * (i % 13) as f32).collect();
        assert_eq!(norm_link(&honest_input(&h), q8::MMVQ, &h, &w2, EPS), Err(Fail::Mismatch));
    }

    #[test]
    fn glu_link_checks_the_covered_blocks() {
        let h = vec_of(3, 0.5);
        let x = q8_1_row(&h.iter().map(|&a| a as f64).collect::<Vec<_>>());
        assert_eq!(glu_link(&x, q8::MMVQ, D, 4, &h[128..256]), Ok(()));
        assert_eq!(glu_link(&x, q8::MMVQ, D, 5, &h[128..256]), Err(Fail::Mismatch));
        assert_eq!(glu_link(&x, q8::MMVQ, D, 15, &h[128..256]), Err(Fail::Mismatch));
    }

    #[test]
    fn non_finite_values_fail() {
        let h = vec_of(1, 3.0);
        let mut v: Vec<f64> = rms_norm(&h, &norm_w(), EPS);
        let x = q8_1_row(&v);
        v[3] = f64::NAN;
        assert_eq!(block_deviation(&x, q8::MMVQ, D, 0, &v), None);
        let add: Vec<f64> = h.iter().map(|&a| a as f64).collect();
        let mut dst: Vec<f32> = h.iter().map(|a| 2.0 * a).collect();
        dst[0] = f32::NAN;
        assert_eq!(residual_link(&dst, &add, &h, 1.0), Err(Fail::Mismatch));
    }

    #[test]
    fn residual_addition() {
        let (y, h) = (vec_of(4, 0.3), vec_of(5, 2.0));
        let add: Vec<f64> = y.iter().map(|&a| a as f64).collect();
        let r: Vec<f32> = y.iter().zip(&h).map(|(a, b)| a + b).collect();
        assert_eq!(residual_link(&r, &add, &h, 1.0), Ok(()));
        assert_eq!(residual_link(&h, &add, &h, 1.0), Err(Fail::Mismatch));
        let zero = vec![0f64; D];
        assert_eq!(residual_link(&r, &zero, &h, 1.0), Err(Fail::Mismatch));
        let scaled: Vec<f32> = r.iter().map(|a| a * 0.75).collect();
        assert_eq!(residual_link(&scaled, &add, &h, 0.75), Ok(()));
        assert_eq!(residual_link(&scaled, &add, &h, 1.0), Err(Fail::Mismatch));
    }

    #[test]
    fn post_norm_residual() {
        let (y, h, w) = (vec_of(6, 4.0), vec_of(7, 2.0), norm_w());
        let add = rms_norm(&y, &w, EPS);
        let r: Vec<f32> = add.iter().zip(&h).map(|(a, &b)| (*a as f32) + b).collect();
        assert_eq!(residual_link(&r, &add, &h, 1.0), Ok(()));
        let raw: Vec<f64> = y.iter().map(|&a| a as f64).collect();
        assert_eq!(residual_link(&r, &raw, &h, 1.0), Err(Fail::Mismatch));
    }
}
