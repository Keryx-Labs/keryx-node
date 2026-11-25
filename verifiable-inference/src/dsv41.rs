//! Compressed attention rows of DeepSeek-V4.1: a block of `ratio` consecutive tokens pooled into
//! one key/value row, read by the attention of the layers that share it.

use crate::attn::{Yarn, rms_norm, rope_norm_yarn};

/// Pooled latent of a block: per value, the members' outputs weighted by the softmax of their
/// gate outputs (no gate: one member), then RMS-normed and scaled by `norm`.
pub fn comp_latent(kvs: &[&[f32]], gates: Option<&[&[f32]]>, norm: &[f32], eps: f64) -> Option<Vec<f64>> {
    let d = norm.len();
    if kvs.is_empty() || kvs.iter().any(|k| k.len() != d) {
        return None;
    }
    let mut out = vec![0f64; d];
    match gates {
        None if kvs.len() == 1 => out.iter_mut().zip(kvs[0]).for_each(|(o, &v)| *o = v as f64),
        Some(g) if g.len() == kvs.len() && g.iter().all(|r| r.len() == d) => {
            for (i, o) in out.iter_mut().enumerate() {
                let m = g.iter().map(|r| r[i] as f64).fold(f64::NEG_INFINITY, f64::max);
                let e: Vec<f64> = g.iter().map(|r| libm::exp(r[i] as f64 - m)).collect();
                let z: f64 = e.iter().sum();
                *o = kvs.iter().zip(&e).map(|(k, w)| k[i] as f64 * w).sum::<f64>() / z;
            }
        }
        _ => return None,
    }
    rms_norm(&mut out, Some(norm), eps);
    out.iter().all(|x| x.is_finite()).then_some(out)
}

/// Rotary part of a compressed row: the last `n_rot` values of `x` rotated at `pos`.
pub fn rope_tail(x: &mut [f64], pos: u64, n_rot: usize, base: f64, yarn: &Yarn) {
    let off = x.len() - n_rot;
    rope_norm_yarn(x, off, pos, n_rot, base, yarn, false);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32s(b: &[u8]) -> Vec<f32> {
        b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
    }

    /// ggml's pooled, normed and rotated rows of 3 blocks at ratio 2 and ratio 1.
    #[test]
    fn compressed_rows_match_ggml() {
        let f = f32s(include_bytes!("../testdata/dsv41-comp.bin"));
        let yarn = Yarn { freq_scale: 1.0 / 16.0, ext_factor: 1.0, attn_factor: (1.0f32 / (1.0 + 0.1 * 16f32.ln())) as f64, beta_fast: 32.0, beta_slow: 1.0, n_ctx_orig: 65_536 };
        let (d, nb) = (512usize, 3usize);
        let mut o = 0;
        for ratio in [2usize, 1] {
            let take = |o: &mut usize, n: usize| {
                let v = f[*o..*o + n].to_vec();
                *o += n;
                v
            };
            let (kv, sc, w, want) = (take(&mut o, d * ratio * nb), take(&mut o, d * ratio * nb), take(&mut o, d), take(&mut o, d * nb));
            for b in 0..nb {
                let kvs: Vec<&[f32]> = (0..ratio).map(|j| &kv[(b * ratio + j) * d..(b * ratio + j + 1) * d]).collect();
                let gs: Vec<&[f32]> = (0..ratio).map(|j| &sc[(b * ratio + j) * d..(b * ratio + j + 1) * d]).collect();
                let gates = (ratio > 1).then_some(&gs[..]);
                let mut x = comp_latent(&kvs, gates, &w, 1e-20).unwrap();
                rope_tail(&mut x, ((b + 1000) * ratio) as u64, 64, 160_000.0, &yarn);
                let dev = x.iter().zip(&want[b * d..(b + 1) * d]).fold(0f64, |m, (a, &e)| m.max((a - e as f64).abs()));
                assert!(dev < 1e-4, "ratio {ratio} block {b}: {dev}");
            }
        }
        assert_eq!(o, f.len());
    }

    #[test]
    fn pooling_needs_one_gate_per_member() {
        let (a, b, w) = (vec![1.0f32; 512], vec![2.0f32; 512], vec![1.0f32; 512]);
        assert!(comp_latent(&[&a, &b], None, &w, 1e-20).is_none());
        assert!(comp_latent(&[&a, &b], Some(&[&a]), &w, 1e-20).is_none());
        let x = comp_latent(&[&a, &b], Some(&[&w, &w]), &w, 1e-20).unwrap();
        assert!(x.iter().all(|&v| (v - 1.0).abs() < 1e-9));
    }
}
