/// Deterministic 3-layer MLP for OPoI Phase 1.
///
/// Architecture: 32 → 256 → 128 → 32 (all ReLU except last layer uses tanh-clamp)
/// Weights are generated at runtime via a seeded LCG — no file I/O, no model download.
/// Every miner produces identical outputs for identical inputs (fully deterministic).

use candle_core::{DType, Device, Result as CandleResult, Tensor};

use crate::task::MODEL_SEED;

// ── Weight generation ─────────────────────────────────────────────────────────

/// Linear congruential generator — fast, deterministic, portable.
/// Uses Knuth's constants from TAOCP Vol. 2, §3.3.4.
fn lcg_next(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
    // Extract 23 mantissa bits → uniform [0, 1), then shift to [-1, 1)
    let mantissa = (*state >> 41) as f32;
    mantissa / (1u32 << 23) as f32 * 2.0 - 1.0
}

/// Generates a weight matrix of shape `(rows, cols)` using He initialisation.
/// `layer_id` offsets the LCG seed so each layer gets independent random weights.
fn make_weights(rows: usize, cols: usize, layer_id: u64) -> Vec<f32> {
    let mut state = MODEL_SEED.wrapping_add(layer_id.wrapping_mul(0xDEAD_BEEF_CAFE_1337));
    let scale = (2.0_f32 / cols as f32).sqrt(); // He init for ReLU layers
    (0..rows * cols).map(|_| lcg_next(&mut state) * scale).collect()
}

// ── Forward pass ──────────────────────────────────────────────────────────────

/// Runs the MLP forward pass on a 32-byte input.
///
/// Returns a 32-byte output that commits the miner to having performed
/// the computation — included in the coinbase `extra_data`.
pub fn forward(input: &[u8; 32], device: &Device) -> CandleResult<[u8; 32]> {
    // Normalise input bytes to f32 in [-0.5, 0.5]
    let x_data: Vec<f32> = input.iter().map(|&b| b as f32 / 255.0 - 0.5).collect();
    // shape: (1, 32)
    let x = Tensor::from_vec(x_data, (1_usize, 32_usize), device)?;

    // Layer 1 — 32 → 256, ReLU
    let w1 = Tensor::from_vec(make_weights(256, 32, 0), (256_usize, 32_usize), device)?;
    let b1 = Tensor::zeros((256_usize,), DType::F32, device)?;
    // (1,32) × (32,256) → (1,256)
    let h1 = x.matmul(&w1.t()?)?.broadcast_add(&b1)?.relu()?;

    // Layer 2 — 256 → 128, ReLU
    let w2 = Tensor::from_vec(make_weights(128, 256, 1), (128_usize, 256_usize), device)?;
    let b2 = Tensor::zeros((128_usize,), DType::F32, device)?;
    // (1,256) × (256,128) → (1,128)
    let h2 = h1.matmul(&w2.t()?)?.broadcast_add(&b2)?.relu()?;

    // Layer 3 — 128 → 32, no activation (raw logits)
    let w3 = Tensor::from_vec(make_weights(32, 128, 2), (32_usize, 128_usize), device)?;
    let b3 = Tensor::zeros((32_usize,), DType::F32, device)?;
    // (1,128) × (128,32) → (1,32)
    let out = h2.matmul(&w3.t()?)?.broadcast_add(&b3)?;

    // Map logits → [0, 255] bytes via tanh squeeze
    let out_vec: Vec<f32> = out.flatten_all()?.to_vec1()?;
    let mut result = [0u8; 32];
    for (i, &v) in out_vec.iter().enumerate().take(32) {
        // tanh maps ℝ → (-1, 1); shift+scale to [0, 255]
        result[i] = ((v.tanh() * 0.5 + 0.5) * 255.0).round() as u8;
    }
    Ok(result)
}
