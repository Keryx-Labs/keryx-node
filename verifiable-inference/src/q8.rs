//! Q8_1 activation rows.

use crate::f16;

/// Scale and quantized values of each 32-value block of a row.
pub fn blocks(leaf: &[u8], layout: u8, k: usize) -> Option<Vec<(f32, [i8; 32])>> {
    let mut out = Vec::with_capacity(k.div_ceil(32));
    match layout {
        MMVQ => {
            if leaf.len() < k.div_ceil(32) * 36 {
                return None;
            }
            for blk in leaf.chunks_exact(36).take(k.div_ceil(32)) {
                out.push((f16::read(blk, 0), std::array::from_fn(|j| blk[4 + j] as i8)));
            }
        }
        MMQ_D4 | MMQ_DS4 | MMQ_D2S6 => {
            if leaf.len() < k.div_ceil(128) * 144 {
                return None;
            }
            for blk in leaf.chunks_exact(144).take(k.div_ceil(128)) {
                for sb in 0..4 {
                    let d = match layout {
                        MMQ_D4 => f32::from_le_bytes([blk[4 * sb], blk[4 * sb + 1], blk[4 * sb + 2], blk[4 * sb + 3]]),
                        MMQ_DS4 => f16::read(blk, 4 * sb),
                        _ => f16::read(blk, 2 * (sb / 2)),
                    };
                    out.push((d, std::array::from_fn(|j| blk[16 + 32 * sb + j] as i8)));
                }
            }
        }
        _ => return None,
    }
    out.truncate(k.div_ceil(32));
    Some(out)
}

pub const MMVQ: u8 = 0;
pub const MMQ_D4: u8 = 1;
pub const MMQ_DS4: u8 = 2;
pub const MMQ_D2S6: u8 = 3;

/// Values and stored partial sums of a row.
pub fn decode(leaf: &[u8], layout: u8, k: usize) -> Option<(Vec<f32>, Option<Vec<f32>>)> {
    let mut x = Vec::with_capacity(k);
    let mut s = Vec::new();
    match layout {
        MMVQ => {
            if leaf.len() < k.div_ceil(32) * 36 {
                return None;
            }
            for blk in leaf.chunks_exact(36).take(k.div_ceil(32)) {
                let d = f16::read(blk, 0);
                s.push(f16::read(blk, 2));
                for j in 0..32 {
                    x.push((blk[4 + j] as i8) as f32 * d);
                }
            }
        }
        MMQ_D4 | MMQ_DS4 | MMQ_D2S6 => {
            if leaf.len() < k.div_ceil(128) * 144 {
                return None;
            }
            for blk in leaf.chunks_exact(144).take(k.div_ceil(128)) {
                for sb in 0..4 {
                    let d = match layout {
                        MMQ_D4 => f32::from_le_bytes([blk[4 * sb], blk[4 * sb + 1], blk[4 * sb + 2], blk[4 * sb + 3]]),
                        MMQ_DS4 => {
                            s.push(f16::read(blk, 4 * sb + 2));
                            f16::read(blk, 4 * sb)
                        }
                        _ => f16::read(blk, 2 * (sb / 2)),
                    };
                    for j in 0..32 {
                        x.push((blk[16 + 32 * sb + j] as i8) as f32 * d);
                    }
                }
            }
        }
        _ => return None,
    }
    x.truncate(k);
    let has_s = layout == MMVQ || layout == MMQ_DS4;
    if has_s {
        s.truncate(k / 32);
    }
    Some((x, has_s.then_some(s)))
}
