/// OPoI Phase 2 verification for the stratum bridge.
///
/// Copied verbatim from keryx-inference/src/model_fixed.rs and task.rs so the bridge
/// has no cross-workspace dependency.  Any change to the miner's model_fixed.rs must
/// be reflected here — both sides must be bit-exact or share validation will break.

// ── Constants (from task.rs) ─────────────────────────────────────────────────

const MODEL_SEED: u64 = 0x4B_65_72_79_78_4F_50_00;

/// Protocol salt XORed into the nonce before the MLP forward pass.
/// Matches `PHASE2_OPOI_SALT` in keryx-inference/src/lib.rs.
pub const PHASE2_OPOI_SALT: u64 = u64::from_le_bytes(*b"KERYX:2\0");

// ── Fixed-point MLP (from model_fixed.rs) ───────────────────────────────────

const NORM_SHIFT: u32 = 10;
const N_IN:  usize = 32;
const N_H1:  usize = 256;
const N_H2:  usize = 128;
const N_OUT: usize = 32;
const HE_L1: i64 = 256;
const HE_L2: i64 = 90;
const HE_L3: i64 = 128;

#[inline]
fn lcg_weight(state: &mut u64, he_scale: i64) -> i32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let raw = ((*state >> 32) as u32) as i32 as i64;
    ((raw * he_scale) >> 31) as i32
}

fn make_weights(rows: usize, cols: usize, layer_id: u64, he: i64) -> Vec<i32> {
    let mut s = MODEL_SEED.wrapping_add(layer_id.wrapping_mul(0xDEAD_BEEF_CAFE_1337));
    (0..rows * cols).map(|_| lcg_weight(&mut s, he)).collect()
}

/// All three layer matrices, generated once — the weights are a pure function
/// of the compile-time MODEL_SEED, so every `forward` call shares this table.
struct Weights {
    w1: Vec<i32>,
    w2: Vec<i32>,
    w3: Vec<i32>,
}

static WEIGHTS: std::sync::LazyLock<Weights> = std::sync::LazyLock::new(|| Weights {
    w1: make_weights(N_H1, N_IN, 0, HE_L1),
    w2: make_weights(N_H2, N_H1, 1, HE_L2),
    w3: make_weights(N_OUT, N_H2, 2, HE_L3),
});

fn forward(input: &[u8; 32]) -> [u8; 32] {
    let Weights { w1, w2, w3 } = &*WEIGHTS;
    let x: Vec<i64> = input.iter().map(|&b| b as i64 - 128).collect();

    let h1: Vec<i64> = (0..N_H1)
        .map(|i| {
            let acc: i64 = (0..N_IN).map(|j| x[j] * w1[i * N_IN + j] as i64).sum();
            (acc >> NORM_SHIFT).max(0)
        })
        .collect();

    let h2: Vec<i64> = (0..N_H2)
        .map(|i| {
            let acc: i64 = (0..N_H1).map(|j| h1[j] * w2[i * N_H1 + j] as i64).sum();
            (acc >> NORM_SHIFT).max(0)
        })
        .collect();

    let h3: Vec<i64> = (0..N_OUT)
        .map(|i| (0..N_H2).map(|j| h2[j] * w3[i * N_H2 + j] as i64).sum())
        .collect();

    let mut out = [0u8; 32];
    for (i, &v) in h3.iter().enumerate() {
        let b = v.to_le_bytes();
        out[i] = b[0] ^ b[1] ^ b[2] ^ b[3] ^ b[4] ^ b[5] ^ b[6] ^ b[7];
    }
    out
}

// ── Public API ───────────────────────────────────────────────────────────────

/// Build the input tensor from a nonce (same layout as InferenceTask::from_nonce).
fn input_from_nonce(nonce: u64) -> [u8; 32] {
    let mut input = [0u8; 32];
    input[..8].copy_from_slice(&nonce.to_le_bytes());
    let seed_bytes = MODEL_SEED.to_le_bytes();
    for chunk in input[8..].chunks_mut(8) {
        let n = chunk.len();
        chunk.copy_from_slice(&seed_bytes[..n]);
    }
    input
}

/// Returns the 16-char hex OPoI tag for a given nonce.
/// Mirrors `keryx_inference::tag_fixed` exactly.
pub fn tag_fixed(nonce: u64) -> String {
    let input = input_from_nonce(nonce ^ PHASE2_OPOI_SALT);
    let output = forward(&input);
    hex::encode(&output[..8])
}

/// Returns `true` when `claimed_hex16` matches the expected tag for `nonce`.
/// Used by the bridge to validate every submitted share.
pub fn verify_tag(nonce: u64, claimed_hex16: &str) -> bool {
    tag_fixed(nonce) == claimed_hex16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regression_nonce_42() {
        // Must match keryx-inference fixed_point_regression_nonce_42.
        let input = input_from_nonce(42);
        let out = forward(&input);
        let expected: [u8; 32] = [
            182, 147, 169, 135, 251, 232, 129,  16,
            221, 172,  47, 152,   9,  81, 226, 160,
              1,  54, 235,  28, 221, 139, 125, 111,
            176, 173, 146,  73, 168, 229, 102, 209,
        ];
        assert_eq!(out, expected);
    }

    #[test]
    fn verify_roundtrip() {
        let nonce = 0xDEAD_BEEF_CAFE_1337u64;
        let tag = tag_fixed(nonce);
        assert!(verify_tag(nonce, &tag));
        assert!(!verify_tag(nonce, "0000000000000000"));
    }

    #[test]
    fn tag_is_16_hex_chars() {
        assert_eq!(tag_fixed(0).len(), 16);
        assert!(tag_fixed(0).chars().all(|c| c.is_ascii_hexdigit()));
    }
}
