//! Stage-2 links between committed activations.

use crate::prefix::{THRESHOLD, expected_input};
use crate::q8;
use crate::verify::Fail;

/// Stage byte of a trace carrying the chain leaves.
pub const STAGE_CHAIN: u8 = 2;
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
        if (d as f64 - scale * (a + s)).abs() > tol {
            return Err(Fail::Mismatch);
        }
    }
    Ok(())
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
