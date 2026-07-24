pub use super::{
    bps::{Bps, TenBps},
    constants::consensus::*,
    genesis::{DEVNET_GENESIS, GENESIS, GenesisBlock, SIMNET_GENESIS, TESTNET_GENESIS, TESTNET11_GENESIS},
};

// ── Inference reward minimums ─────────────────────────────────────────────────
// model_id = sha2-256(primary_weight_file) = CIDv0_decoded_bytes[2..34].

/// TinyLlama 1.1B — sha2-256(QmdqcmS8aMngiZWYYdeZEaW22N6XRTd9zK5ZCJG1MPmrQ3)
pub const TINYLLAMA_MODEL_ID: [u8; 32] = [
    0xe6, 0x4a, 0xf3, 0x68, 0xec, 0x93, 0x51, 0xa5,
    0xa4, 0xc0, 0xec, 0x7a, 0xe4, 0x7d, 0x42, 0xad,
    0xa7, 0xf6, 0xb3, 0xf1, 0xa6, 0xe6, 0x0f, 0xc7,
    0x3d, 0x0e, 0xb6, 0xca, 0x29, 0x53, 0x64, 0x5c,
];

/// DeepSeek-R1-8B — sha2-256(QmYK1faUGNMYZ2UKeSpUoUoFpRarZQEwfPCHbYNG2ib2mR)
pub const DEEPSEEK_R1_8B_MODEL_ID: [u8; 32] = [
    0x94, 0x29, 0x67, 0x33, 0x16, 0xbc, 0x40, 0xec,
    0x06, 0x67, 0x89, 0x45, 0x34, 0x57, 0x8b, 0x41,
    0x23, 0x6f, 0xc7, 0xee, 0xa4, 0xd9, 0x31, 0xf1,
    0x48, 0x9c, 0x34, 0xc5, 0x83, 0x7f, 0x42, 0xf4,
];

/// DeepSeek-R1-32B — sha2-256(model.gguf) computed locally
pub const DEEPSEEK_R1_32B_MODEL_ID: [u8; 32] = [
    0xbe, 0xd9, 0xb0, 0xf5, 0x51, 0xf5, 0xb9, 0x5b,
    0xf9, 0xda, 0x58, 0x88, 0xa4, 0x8f, 0x0f, 0x87,
    0xc3, 0x7a, 0xd6, 0xb7, 0x25, 0x19, 0xc4, 0xcb,
    0xd7, 0x75, 0xf5, 0x4a, 0xc0, 0xb9, 0xfc, 0x62,
];

/// LLaMA-3.3-70B — sha2-256(model.gguf) computed locally
pub const LLAMA_3_3_70B_MODEL_ID: [u8; 32] = [
    0xaa, 0xd2, 0xcf, 0x33, 0x48, 0xd8, 0xc7, 0xfd,
    0xbd, 0x2c, 0x0d, 0xd5, 0x8e, 0x0d, 0x99, 0x36,
    0x84, 0x50, 0xd4, 0x3c, 0x95, 0x84, 0xae, 0xf8,
    0x1a, 0x46, 0x7d, 0xd3, 0x47, 0x56, 0x13, 0x44,
];

/// Per-model minimum inference_reward in sompi. Legacy (pre-OPoI-v2) lineup.
pub const INFERENCE_REWARD_MINIMUMS: &[([u8; 32], u64)] = &[
    (TINYLLAMA_MODEL_ID,         50_000_000),   // 0.5 KRX
    (DEEPSEEK_R1_8B_MODEL_ID,   150_000_000),   // 1.5 KRX
    (DEEPSEEK_R1_32B_MODEL_ID,  250_000_000),   // 2.5 KRX
    (LLAMA_3_3_70B_MODEL_ID,   400_000_000),   // 4.0 KRX
];

// ── OPoI v2 lineup (uncensored / abliterated) ─────────────────────────────────
// Active from `opoi_v2_activation`. Weights + tokenizers pinned on the Keryx IPFS
// gateway. model_id = base58-decode(weight CID)[2..34] = sha2-256(model.gguf).

/// Gemma-3-4B-it-abliterated — sha2-256(Qma1CbFzWTNhy2ReVjDG1GvM5q2Uy4VhqTbnS9c641jUQ6)
pub const GEMMA_3_4B_MODEL_ID: [u8; 32] = [
    0xad, 0x50, 0xad, 0x0b, 0xd4, 0x61, 0xd8, 0xab,
    0x44, 0xef, 0xc0, 0x21, 0x49, 0x89, 0xeb, 0x33,
    0x29, 0x16, 0x85, 0xef, 0x4a, 0xde, 0x22, 0xa0,
    0xf4, 0xf2, 0x17, 0xd0, 0x32, 0x66, 0xd8, 0x37,
];

/// Dolphin-3.0-Llama-3.1-8B — sha2-256(QmYJtFpaDnVwAVSbzRo42fsb19nLpt8LHe8WVKoyxd4AkZ)
pub const DOLPHIN_LLAMA3_8B_MODEL_ID: [u8; 32] = [
    0x94, 0x21, 0x06, 0x6a, 0x64, 0x00, 0xc9, 0x8b,
    0xa1, 0x37, 0x11, 0x4f, 0x7f, 0x4b, 0x7d, 0x4a,
    0x2d, 0xdf, 0x13, 0xab, 0x16, 0x3a, 0x5d, 0xe3,
    0x8c, 0x01, 0x84, 0x79, 0x3a, 0xf6, 0x31, 0x3a,
];

/// Qwen3-32B-abliterated — sha2-256(QmVBwp5n3muQJwYNLTHSu3EnzBWviQqfh58FvHvKRfLtam)
pub const QWEN3_32B_MODEL_ID: [u8; 32] = [
    0x65, 0xc6, 0xeb, 0x6f, 0xe1, 0x8b, 0x9e, 0xfd,
    0x80, 0x60, 0xab, 0x9d, 0x2d, 0x03, 0xbb, 0x9b,
    0x01, 0x05, 0x0a, 0x3b, 0x13, 0x78, 0xcb, 0xac,
    0x00, 0x0c, 0x5c, 0xc0, 0xac, 0xdc, 0x0d, 0x2a,
];

/// Llama-3.3-70B-Instruct-abliterated — sha2-256(QmPdTayXcEsfUwMCoMKKcLSv7Dwpp2xVBWELwrG2M7Rhzu)
pub const LLAMA_3_3_70B_ABLITERATED_MODEL_ID: [u8; 32] = [
    0x13, 0x29, 0xfb, 0xe2, 0x1b, 0x3f, 0x36, 0xf6,
    0xd0, 0x06, 0x89, 0xfc, 0xaa, 0x74, 0xf7, 0xa2,
    0x22, 0xb8, 0xcc, 0x4c, 0x08, 0xc0, 0x19, 0x1f,
    0xeb, 0x23, 0x97, 0x55, 0xa7, 0x23, 0x42, 0x1e,
];

// --- H2 lineup refresh (gated by `very_light_activation`). MUST mirror the miner's `models.rs`. ---

/// Qwen3-1.7B-abliterated Q4_K_M (mlabonne base, locally quantized). New `--very-light` tier 0
/// post-H2. CIDv0[2..34] of model.gguf — must match the miner's `QWEN3_1_7B.model_id`.
pub const QWEN3_1_7B_MODEL_ID: [u8; 32] = [
    0x4f, 0x21, 0xdd, 0xeb, 0x7d, 0x62, 0xbd, 0x22,
    0x65, 0xbc, 0x54, 0x23, 0x0d, 0x53, 0x6c, 0xa3,
    0xf1, 0x74, 0x99, 0x27, 0x78, 0x0f, 0x52, 0x8c,
    0x3c, 0x41, 0xfa, 0x29, 0x11, 0xdf, 0x4d, 0x72,
];

/// Llama-3.3-70B-Instruct-abliterated Q2_K_L (bartowski). Replaces the 48 GB Q4 as the post-H2
/// top tier so a 32 GB 5090 can serve it. CIDv0[2..34] of model.gguf — must match the miner's
/// `LLAMA_3_3_70B_Q2.model_id`. Verified: a fresh bartowski download re-hashes to exactly this CID
/// (`QmVjsK1LBMjk24tawUrGyWUEXHQwkcPgeetC5JpNZL7p1J`), so the model_id and R_T below are canonical.
pub const LLAMA_3_3_70B_Q2_MODEL_ID: [u8; 32] = [
    0x6d, 0xf4, 0x6a, 0x78, 0xcb, 0xe4, 0xdc, 0x57,
    0x9f, 0x04, 0xdb, 0xd8, 0x01, 0xf1, 0xa5, 0x20,
    0xb9, 0xea, 0xe2, 0x8c, 0xe7, 0xb5, 0x0c, 0x8d,
    0xa7, 0x87, 0x4b, 0xfa, 0x3f, 0xb5, 0x10, 0x8d,
];

/// Per-model minimum inference_reward in sompi. OPoI v2 lineup, enforced from
/// `opoi_v2_activation` (replaces `INFERENCE_REWARD_MINIMUMS` at that DAA score).
pub const INFERENCE_REWARD_MINIMUMS_V2: &[([u8; 32], u64)] = &[
    (GEMMA_3_4B_MODEL_ID,                 50_000_000),   // 0.5 KRX  (--light)
    (DOLPHIN_LLAMA3_8B_MODEL_ID,         150_000_000),   // 1.5 KRX  (default)
    (QWEN3_32B_MODEL_ID,                 250_000_000),   // 2.5 KRX  (--high)
    (LLAMA_3_3_70B_ABLITERATED_MODEL_ID, 400_000_000),   // 4.0 KRX  (--very-high)
];

/// Per-model minimum inference_reward in sompi. H2 (5-tier) lineup, enforced from
/// `inference_min_h2_activation` (replaces `INFERENCE_REWARD_MINIMUMS_V2` at that DAA score).
/// The v2 table above was never extended when the H2 lineup shipped, so two served models had no
/// enforced floor: `--very-light` Qwen3-1.7B (absent) and the top tier 70B-Q2 (the v2 table still
/// lists the retired 70B-Q4 model_id). This table adds both, mirroring the dApp's advertised floors.
/// Gated at a FUTURE DAA — never at `very_light_activation` (already past): applying a stricter
/// minimum to a historical block would reject it on IBD re-validation and diverge the UTXO set.
/// Node-only enforcement (the miner does not check minimums), so no miner lockstep is required.
pub const INFERENCE_REWARD_MINIMUMS_V2_H2: &[([u8; 32], u64)] = &[
    (QWEN3_1_7B_MODEL_ID,          30_000_000),   // 0.3 KRX  (--very-light)
    (GEMMA_3_4B_MODEL_ID,          50_000_000),   // 0.5 KRX  (--light)
    (DOLPHIN_LLAMA3_8B_MODEL_ID,  150_000_000),   // 1.5 KRX  (default)
    (QWEN3_32B_MODEL_ID,          250_000_000),   // 2.5 KRX  (--high)
    (LLAMA_3_3_70B_Q2_MODEL_ID,   400_000_000),   // 4.0 KRX  (--very-high, Q2_K_L)
];

/// SINGLE flip point for the H4 hard fork on mainnet. `u64::MAX` = dormant (every H4 gate reads as
/// `never()`). Set this to the chosen H4 DAA score at release — it drives BOTH mainnet coin-age
/// gates (`coin_age_activation` + `coin_age_verification_activation`) in one edit. The miner's
/// `COIN_AGE_VERIFICATION_ACTIVATION_DAA` MUST be set to the exact same value (node↔miner lockstep).
/// NOTE: the H4 difficulty reset is a SEPARATE entry (see `difficulty_reset_activations`), because
/// the existing H2 reset at 38_951_445 is load-bearing history that must not move.
pub const H4_ACTIVATION_DAA: u64 = 54_766_000;

/// Single activation gate for the ENTIRE H5 bundle — parallel-block cap, non-foldable `transition()`
/// walk + `verify_merkle` bound, and the tier-0 model swap. `u64::MAX` = dormant; set this to the
/// real DAA at H5 release and every H5 feature flips together in one edit. MUST be mirrored on the
/// miner side (walk + lineup). See KERYX-KRX/H5_hardfork_plan.
pub const H5_ACTIVATION_DAA: u64 = 59_009_013;

/// H5 parallel-block cap: max blocks per selected-parent counted in the DAA score (and paid).
/// The surplus is forced into `mergeset_non_daa` — excluded from both the DAA increment and the
/// coinbase payment — never rejected (rejection at admission is non-deterministic → split).
/// N=20 is the lowest bound that never clips an honest producer: the measured honest ceiling is
/// ~15 blocks/selected-parent, so 20 leaves margin while neutralizing sibling floods (observed up
/// to 44/DAA). Gated by `h5_activation` on the selected parent's DAA score, a pure header-level
/// function so IBD re-derives it identically. See KERYX-KRX/H5_hardfork_plan.
pub const PARALLEL_BLOCK_CAP_N: usize = 20;

// --- H4 lineup refresh (gated by `coin_age_activation`, bundled into H4). Fully candle-independent:
// every model is UNTIED (llama.cpp hosts the walk + inference in one resident copy). MUST mirror the
// miner's `models.rs`. Each `model_id` = CIDv0[2..34] of the pinned GGUF; each `root` from the offline
// builder (`WeightIndex::build_from_gguf`), spike-verified byte-identical. Dormant on mainnet until
// `coin_age_activation` is scheduled. ---

/// EXAONE-4.0-1.2B-abliterated Q4_K_M (LG). H4 tier 0 (--very-light), replaces Qwen3-1.7B.
pub const EXAONE_4_0_1_2B_MODEL_ID: [u8; 32] = [
    0x30, 0x0a, 0x99, 0xb3, 0xa8, 0x5b, 0x0a, 0xb4,
    0x5d, 0x1d, 0x93, 0x0b, 0xb7, 0xb1, 0xd4, 0xb0,
    0xf3, 0x59, 0x83, 0xd5, 0x21, 0xe7, 0x9f, 0xf2,
    0x11, 0x93, 0xa6, 0x90, 0x8d, 0xc4, 0xb8, 0x10,
];

/// Mistral-7B-Instruct-v0.3-abliterated Q6_K (Mistral). H4 tier 1 (--light), replaces Gemma-3-4B.
pub const MISTRAL_7B_V03_MODEL_ID: [u8; 32] = [
    0x8c, 0x2f, 0xea, 0x60, 0x0f, 0x0e, 0xef, 0xe7,
    0x04, 0x87, 0x41, 0xa5, 0x11, 0x9c, 0xb7, 0xbe,
    0x30, 0x30, 0x37, 0xf5, 0x9f, 0xc0, 0x26, 0xe4,
    0x83, 0x82, 0x65, 0x8f, 0x23, 0x58, 0x1e, 0x0a,
];

/// GLM-4-9B-0414-abliterated Q6_K (Zhipu). H4 tier 2 (default), replaces Dolphin-8B.
pub const GLM_4_9B_0414_MODEL_ID: [u8; 32] = [
    0xfa, 0x2f, 0x13, 0xbe, 0x08, 0x50, 0xe2, 0x6c,
    0x5c, 0xe8, 0x6c, 0x7a, 0xc7, 0x9d, 0xa8, 0x5e,
    0x30, 0x0c, 0x1d, 0xa8, 0xb3, 0x29, 0x0f, 0x9a,
    0x18, 0xd4, 0x71, 0x05, 0xf1, 0xf2, 0x14, 0x0a,
];

/// Qwen3.6-27B-abliterated Q4_K_M (Alibaba, arch qwen35 hybrid-SSM). H4 tier 3 (--high), replaces Qwen3-32B.
pub const QWEN3_6_27B_MODEL_ID: [u8; 32] = [
    0xb8, 0xbd, 0xc0, 0x1f, 0xa4, 0x07, 0xea, 0xb9,
    0x43, 0xe4, 0xfe, 0xfc, 0x80, 0x74, 0x83, 0xb3,
    0x9f, 0x81, 0x42, 0x78, 0x52, 0x56, 0x04, 0x9e,
    0x1f, 0x55, 0x96, 0x98, 0xa5, 0x28, 0x47, 0x46,
];

/// Kimi-Linear-48B-A3B-abliterated Q4_K_M (Moonshot, MoE). H4 tier 4 (--very-high), replaces Llama-70B-Q2.
pub const KIMI_LINEAR_48B_MODEL_ID: [u8; 32] = [
    0x3d, 0xc0, 0x93, 0x58, 0xad, 0x75, 0xc6, 0xef,
    0x0c, 0x9c, 0x86, 0xee, 0x4f, 0x47, 0xc4, 0xd6,
    0xac, 0xda, 0x96, 0x1f, 0xec, 0xbd, 0x0e, 0x4f,
    0x9c, 0xf5, 0x5e, 0x8f, 0x0f, 0xdf, 0xfd, 0xdb,
];

/// Per-model minimum inference_reward in sompi. H4 (candle-free) lineup, enforced from
/// `coin_age_activation`. New bareme 0.5/1/1.5/2.5/4 KRX. Node-only enforcement (no miner lockstep).
/// Gated at the H4 boundary (never applied to historical blocks — see the H2 table's rationale).
pub const INFERENCE_REWARD_MINIMUMS_V2_H4: &[([u8; 32], u64)] = &[
    (EXAONE_4_0_1_2B_MODEL_ID,     50_000_000),   // 0.5 KRX  (--very-light)
    (MISTRAL_7B_V03_MODEL_ID,     100_000_000),   // 1.0 KRX  (--light)
    (GLM_4_9B_0414_MODEL_ID,      150_000_000),   // 1.5 KRX  (default)
    (QWEN3_6_27B_MODEL_ID,        250_000_000),   // 2.5 KRX  (--high)
    (KIMI_LINEAR_48B_MODEL_ID,    400_000_000),   // 4.0 KRX  (--very-high)
];

// --- Proof-of-Model possession (post-PoW). See POM_CONSENSUS_SPEC.md. ---

/// Data-dependent 32 B reads per possession-walk attempt (the memory-hard work core).
/// K=256 — chosen compromise: ~25 MH/s on a 3090 with solid possession strictness.
pub const POM_WALK_STEPS: u32 = 256;
/// Fiat-Shamir-opened steps revealed per `PomProof` (soundness `~f^t` vs proof size).
pub const POM_OPENINGS: usize = 32;

/// Selected-chain depth, in chain blocks, behind which a block's persisted `PomProof` may be
/// garbage-collected. A proof is only ever needed to (re-)serve a *recent* block to peers via relay;
/// far older blocks are only ever requested through IBD, which does not verify the proof
/// (`skip_pom_proof`). So keeping proofs for the last `POM_PROOF_RETENTION_DEPTH` DAA-score depth is
/// far more than the relay horizon (a few hundred blocks) and lets the rest be reclaimed. Deleting a
/// proof can never corrupt consensus state: the proof is not part of the UTXO set / state, and the
/// header `utxo_commitment` already pins the state. The GC pass runs unconditionally on every node
/// (see the pruning processor) — no flag, transparent — so pruned datadirs stay bounded by design.
///
/// Lowered 50_000 → 25_000 for the H4 verifier-v2 proofs: v2 records the full K=256-step walk
/// (256 Merkle paths) instead of the v1 32/256 spot-check, so each proof is ~2.7× bigger (~228 KB
/// vs ~48 KB). At 25_000 the pruned proof store stays ~6 GB (vs ~12 GB at 50_000), keeping the
/// bootstrap datadir near its H3 size. Still far above the relay horizon; not consensus (per above).
pub const POM_PROOF_RETENTION_DEPTH: u64 = 25_000;

/// Per-tier possession anchors `R_T` (32 B-chunk blake3 Merkle root) + `N` (chunk count),
/// produced offline by `pom-rt-builder` (canonical: name-sorted GGUF tensors). Tier index =
/// slice position; `model_id` ties the tier to the declared model. Difficulty stays global
/// (no per-tier target — measured ~1.5x hashrate spread over 10x model size).
pub const POM_TIERS: &[crate::pom::PomTier] = &[
    crate::pom::PomTier {
        model_id: GEMMA_3_4B_MODEL_ID,
        root: [
            0x84, 0x6c, 0xaa, 0x40, 0x0c, 0xf0, 0x14, 0x13, 0x21, 0x18, 0x49, 0x5d, 0x22, 0xe4, 0xbf, 0xa2,
            0x42, 0x45, 0x4e, 0xac, 0x0d, 0x83, 0x5c, 0x3f, 0x8e, 0x63, 0x47, 0xd0, 0x13, 0x9d, 0x1b, 0x7e,
        ],
        chunks: 77_604_776,
    },
    crate::pom::PomTier {
        model_id: DOLPHIN_LLAMA3_8B_MODEL_ID,
        root: [
            0x13, 0x3f, 0x62, 0x7b, 0x88, 0x2e, 0xf8, 0x56, 0x78, 0x5a, 0x83, 0x98, 0x6a, 0x9b, 0x1a, 0xdf,
            0xed, 0xff, 0xf0, 0x74, 0x4a, 0x1f, 0x94, 0x21, 0xec, 0x4d, 0xa6, 0xe9, 0x46, 0x68, 0x15, 0xde,
        ],
        chunks: 153_528_426,
    },
    // Qwen3-32B (Q4_K_M, 707 tensors, 18.40 GiB) — R_T from pom-rt-builder streaming Merkle.
    crate::pom::PomTier {
        model_id: QWEN3_32B_MODEL_ID,
        root: [
            0xe2, 0xaa, 0x66, 0x59, 0xaa, 0xb4, 0x38, 0x7e, 0xb5, 0xfd, 0x79, 0x40, 0x9c, 0x0a, 0x1a, 0x68,
            0x86, 0x3a, 0x3d, 0xef, 0x3b, 0x66, 0x2c, 0xb4, 0x06, 0x16, 0x97, 0xf0, 0xea, 0x87, 0xfa, 0x58,
        ],
        chunks: 617_380_448,
    },
    // Llama-3.3-70B (Q4_K_M, 724 tensors, 39.59 GiB) — R_T from pom-rt-builder streaming Merkle.
    crate::pom::PomTier {
        model_id: LLAMA_3_3_70B_ABLITERATED_MODEL_ID,
        root: [
            0x53, 0x5f, 0xc2, 0xac, 0xb6, 0x09, 0x7b, 0x5d, 0xf8, 0x83, 0xec, 0x50, 0x66, 0x9a, 0x7f, 0x48,
            0xdc, 0x9f, 0x3b, 0xd5, 0x98, 0x74, 0x28, 0x59, 0xb8, 0xbb, 0x4c, 0xac, 0x3b, 0x35, 0x26, 0xaa,
        ],
        chunks: 1_328_516_616,
    },
];

/// Post-H2 (5-tier) possession anchors, gated by `very_light_activation`. Inserts `--very-light`
/// Qwen3-1.7B at tier 0 (the existing tiers shift up by one) and replaces the top tier's 70B Q4
/// with the 32 GB-servable Q2_K_L. MUST mirror the miner's `pom_tier_index` post-H2 ordering:
/// Qwen3-1.7B=0, Gemma=1, Dolphin=2, Qwen3-32B=3, Llama-70B-Q2=4.
pub const POM_TIERS_H2: &[crate::pom::PomTier] = &[
    // Qwen3-1.7B (Q4_K_M, 310 tensors, 1.026 GiB) — R_T from pom-rt-builder streaming Merkle.
    crate::pom::PomTier {
        model_id: QWEN3_1_7B_MODEL_ID,
        root: [
            0xd0, 0x9a, 0x0b, 0x1c, 0x26, 0x25, 0x69, 0xc2, 0x39, 0xfa, 0xcc, 0xf6, 0x41, 0xf8, 0xe4, 0x35,
            0x4a, 0x15, 0x77, 0x50, 0x1b, 0xa8, 0x42, 0xbc, 0x64, 0x9a, 0x87, 0x6d, 0xe1, 0xaf, 0x9a, 0x5d,
        ],
        chunks: 34_420_544,
    },
    POM_TIERS[0], // Gemma-3-4B
    POM_TIERS[1], // Dolphin-Llama3-8B
    POM_TIERS[2], // Qwen3-32B
    // Llama-3.3-70B-Q2_K_L (724 tensors, 25.512 GiB) — R_T from pom-rt-builder streaming Merkle.
    // GGUF re-downloaded from bartowski; CID verified == the recorded model_id before computing R_T.
    crate::pom::PomTier {
        model_id: LLAMA_3_3_70B_Q2_MODEL_ID,
        root: [
            0xb9, 0x6c, 0xfc, 0xb5, 0x38, 0xae, 0xb0, 0x66, 0xa1, 0x8c, 0xea, 0xa1, 0x1c, 0x8b, 0x1a, 0x04,
            0x4f, 0x91, 0x32, 0x40, 0x8e, 0x87, 0x04, 0x8e, 0xb7, 0x41, 0xfe, 0x73, 0xed, 0x1b, 0xf6, 0x18,
        ],
        chunks: 856_040_456,
    },
];

/// H4 (candle-free) possession anchors, gated by `coin_age_activation`. Same 5-tier ORDER as H2
/// (tier index → reward position unchanged), swapping every model for an UNTIED one so llama.cpp
/// hosts walk + inference with no candle: EXAONE-1.2B=0, Mistral-7B=1, GLM-4-9B=2, Qwen3.6-27B=3,
/// Kimi-Linear-48B=4. MUST mirror the miner's `pom_tier_index` H4 ordering. Roots spike-verified.
pub const POM_TIERS_H4: &[crate::pom::PomTier] = &[
    crate::pom::PomTier {
        model_id: EXAONE_4_0_1_2B_MODEL_ID,
        root: [
            0xcc, 0x8b, 0x25, 0xc4, 0xe1, 0xaa, 0x7a, 0xb9, 0xbb, 0x99, 0x41, 0xda, 0x16, 0x18, 0xf9, 0xab,
            0x29, 0x38, 0xea, 0x85, 0x07, 0x7b, 0x88, 0x79, 0xeb, 0xd7, 0xd4, 0x91, 0x6a, 0xc3, 0x8d, 0xdd,
        ],
        chunks: 28_943_588,
    },
    crate::pom::PomTier {
        model_id: MISTRAL_7B_V03_MODEL_ID,
        root: [
            0xd7, 0x6a, 0xcb, 0xbe, 0x8c, 0x24, 0x29, 0x81, 0x6c, 0x02, 0xa4, 0xdb, 0xd9, 0xf2, 0x09, 0xa0,
            0x87, 0x85, 0xef, 0x97, 0x5c, 0xd1, 0x38, 0xf5, 0x18, 0x22, 0x76, 0x12, 0x0b, 0xa2, 0x0e, 0xc5,
        ],
        chunks: 185_827_840,
    },
    crate::pom::PomTier {
        model_id: GLM_4_9B_0414_MODEL_ID,
        root: [
            0x1b, 0xa8, 0xb8, 0xb1, 0x34, 0x41, 0x03, 0xfa, 0xa0, 0xa7, 0x47, 0x89, 0xd9, 0x39, 0xc3, 0x3c,
            0x23, 0xba, 0x5c, 0x3c, 0x41, 0xbb, 0x1a, 0x89, 0x5a, 0xb6, 0xe8, 0xbf, 0xec, 0xb0, 0x78, 0x7d,
        ],
        chunks: 258_040_832,
    },
    crate::pom::PomTier {
        model_id: QWEN3_6_27B_MODEL_ID,
        root: [
            0x85, 0x23, 0xf4, 0x14, 0x8d, 0x22, 0xc7, 0x71, 0x3b, 0xfc, 0x11, 0x32, 0xb4, 0xaf, 0x3d, 0x4b,
            0x97, 0x61, 0xa2, 0x03, 0xfb, 0x33, 0xf1, 0x8e, 0xe7, 0x55, 0x67, 0xbd, 0xee, 0x51, 0x2b, 0x0a,
        ],
        chunks: 516_762_688,
    },
    crate::pom::PomTier {
        model_id: KIMI_LINEAR_48B_MODEL_ID,
        root: [
            0x95, 0x74, 0x71, 0x0f, 0xfa, 0xb6, 0x78, 0xf0, 0x68, 0xb4, 0xe6, 0x5a, 0xbe, 0x72, 0x40, 0x86,
            0x2d, 0xa1, 0x5b, 0xb1, 0x6e, 0xa8, 0x2f, 0xd1, 0x62, 0xa9, 0x35, 0x1a, 0x10, 0x51, 0x99, 0x59,
        ],
        chunks: 927_994_064,
    },
];

/// Qwen3-8B-abliterated Q4_K_S (huihui-ai, mradermacher GGUF). H5 tier 0 (--very-light), replaces
/// EXAONE. `model_id` = CIDv0[2..34] of the pinned GGUF (IPFS Qm...ccwHVeZYVzEq6A5ofk76MxrnwzMnSjAVt9PaUQ7zfLXm).
pub const QWEN3_8B_ABLITERATED_MODEL_ID: [u8; 32] = [
    0xd4, 0x2f, 0xa6, 0xee, 0x00, 0xe0, 0x7d, 0x49,
    0xb0, 0x46, 0x09, 0x0a, 0x56, 0xaf, 0x0e, 0x7b,
    0xd6, 0x10, 0x25, 0x93, 0x7c, 0x50, 0x2e, 0x2c,
    0x57, 0x4a, 0x72, 0x87, 0x4c, 0x35, 0x0d, 0x24,
];

/// H5 possession anchors — same 5-tier ORDER as H4, swapping ONLY tier 0's model (raising the
/// tier-0 VRAM floor to ~6 GB). Tiers 1-4 are unchanged from H4 (same models → same R_T). Gated by
/// `h5_activation`.
pub const POM_TIERS_H5: &[crate::pom::PomTier] = &[
    // Tier 0: Qwen3-8B-abliterated Q4_K_S. root = pom-rt-builder R_T over the pinned GGUF's 32 B
    // chunks (name-sorted tensors | blake3 leaf | blake3 tree); chunks = N reported by the builder.
    crate::pom::PomTier {
        model_id: QWEN3_8B_ABLITERATED_MODEL_ID,
        root: [
            0xa1, 0xcb, 0xff, 0xfa, 0xae, 0xb9, 0x71, 0xcb, 0x29, 0x7b, 0x7e, 0x01, 0xff, 0x41, 0x09, 0x72,
            0x3e, 0x43, 0x97, 0x41, 0xcd, 0x42, 0x68, 0x22, 0x5f, 0x0c, 0x30, 0xa3, 0x33, 0xe6, 0x9a, 0x68,
        ],
        chunks: 149_876_736,
    },
    POM_TIERS_H4[1],
    POM_TIERS_H4[2],
    POM_TIERS_H4[3],
    POM_TIERS_H4[4],
];

/// Possession anchors for a block at `daa_score`: the H5 set once `h5_activation` is live (tier-0
/// model swap), else the H4 candle-free set, else the 5-tier H2 set once `very_light_activation`,
/// else the legacy 4-tier set. The choice MUST be made per block from that block's own DAA (never
/// frozen) — an archival/IBD node recomputing an older block under a newer scheme would validate
/// against the wrong anchors and reject the chain.
pub fn pom_tiers(h5_active: bool, coin_age_active: bool, very_light_active: bool) -> &'static [crate::pom::PomTier] {
    if h5_active {
        POM_TIERS_H5
    } else if coin_age_active {
        POM_TIERS_H4
    } else if very_light_active {
        POM_TIERS_H2
    } else {
        POM_TIERS
    }
}

/// Tier-reward — multiplier in basis points applied to the *immediate miner cut* (the 75 %
/// paid at once, after the R&D and escrow cuts) of a block's subsidy, indexed by the block's
/// cryptographically-proven PoM tier (`PomProof::tier`, the slice position in `POM_TIERS`).
/// Heavier model ⇒ larger share kept. The un-earned delta is burned (see the coinbase manager),
/// so the total block reward, the R&D cut and the escrow cut are untouched. The top tier is the
/// 100 % reference. Gated by `pom_activation` (a proven tier only exists under PoM).
///
/// 6-point steps: a compromise between the bench-justified 10-point spread (the PoM walk hashrate is
/// near-flat across tiers — ~5 % drop over an 8× model-size range on a 3090 — so a small step barely
/// beats the dip; see KERYX-KRX/tier_reward_bench.md) and the need to soften the *multiplicative*
/// compound now that tier-reward and holder-reward co-activate at the same mainnet H. The combined
/// miner cut is `tier_bps × ratio_bps`, so a 10-point tier floor stacked on the 40 % holder floor
/// dropped the worst case to 28 %; 6-point steps lift the tier floor to 82 % (worst case ≈ 33 %) while
/// keeping each tier-up worth a meaningful ~+6-7 %.
///   0  Gemma-3-4B        --light       -18%
///   1  Dolphin-Llama3-8B default       -12%
///   2  Qwen3-32B         --high         -6%
///   3  Llama-3.3-70B     --very-high     0%
pub const TIER_REWARD_BPS: [u64; 4] = [8_200, 8_800, 9_400, 10_000];

/// Post-H2 (5-tier) tier-reward schedule, gated by `very_light_activation`. 8-point steps with the
/// 70B-Q2 as the 100 % top, so `--very-light` (Qwen3-1.7B) bottoms out at −32 %: smallest model ⇒
/// weakest possession ⇒ lowest reward, deliberately steep to discourage low-effort farming of the
/// entry tier. Wider than the legacy 4-tier 6-point spread — the H2 curve re-spaces all five tiers.
///   0  Qwen3-1.7B        --very-light  -32%
///   1  Gemma-3-4B        --light       -24%
///   2  Dolphin-Llama3-8B default       -16%
///   3  Qwen3-32B         --high         -8%
///   4  Llama-3.3-70B-Q2  --very-high     0%
pub const TIER_REWARD_BPS_H2: [u64; 5] = [6_800, 7_600, 8_400, 9_200, 10_000];

/// Tier-reward schedule for a block at `daa_score`: 5-tier H2 once `very_light_activation` is live,
/// legacy 4-tier before. Chosen per block from that block's own DAA (never frozen) — same gating
/// discipline as `pom_tiers`, so archival/IBD recomputation of pre-H2 blocks stays canonical.
pub fn tier_reward_bps(very_light_active: bool) -> &'static [u64] {
    if very_light_active {
        &TIER_REWARD_BPS_H2
    } else {
        &TIER_REWARD_BPS
    }
}

/// Basis-points divisor for `TIER_REWARD_BPS` (= the top-tier 100 % reference).
pub const TIER_REWARD_BPS_DIVISOR: u64 = 10_000;

/// Ratio-reward — holder-weighted multiplier (bps) applied to the *immediate miner cut*, indexed
/// by the holder ratio `balance ÷ windowed_production` (see `ratio_reward_bps`). It clones the
/// tier-reward machinery but swaps the proven model-tier input for a ratio bucket computed by the
/// node from chain state (no miner input). The un-earned delta is burned, so the total reward, the
/// R&D cut and the escrow cut are untouched. When the tier-reward is also active the two factors
/// **compound** multiplicatively on the miner cut. Gated by `ratio_reward_activation`.
///
/// 6 brackets, brutal, floor 40 %: a miner holding < 1 window of its own production (a dumper)
/// keeps 40 %; holding ~1 month of production keeps 100 %. See KERYX-KRX/ratio_reward_spec.md.
pub const RATIO_REWARD_BPS: [u64; 6] = [4_000, 5_200, 6_400, 7_600, 8_800, 10_000];

/// Basis-points divisor for `RATIO_REWARD_BPS` (= the top-bracket 100 % reference).
pub const RATIO_REWARD_BPS_DIVISOR: u64 = 10_000;

/// Bracket entry thresholds, expressed as integer multiples of windowed production. Bracket `i`
/// is reached when `balance >= RATIO_REWARD_THRESHOLDS[i] * windowed_production`. Must be sorted
/// ascending and start at 0 (bracket 0 always reachable). Reading: 0/1/3/7/15/30 windows held.
pub const RATIO_REWARD_THRESHOLDS: [u64; 6] = [0, 1, 3, 7, 15, 30];

/// Ratio-reward v2 — recalibrated bracket table, gated by `coin_age_activation` (bundled into H4,
/// one hardfork instead of two): higher floor (50 % instead of 40 %, less burn overall) and a
/// gentler 9-step ramp to 100 % over 90 days. Deliberately stays within 0-100 % — a bracket above
/// 100 % would let the ratio bonus compensate the tier-reward penalty, which was rejected: it
/// would let a miner with the hardware for a big tier deliberately run a small one and, given
/// enough patience, still reach full reward — undermining the tier-reward's entire purpose
/// (rewarding real model capacity). See KERYX-KRX/ratio_reward_spec.md (v2 addendum).
pub const RATIO_REWARD_BPS_V2: [u64; 9] = [5_000, 5_500, 6_000, 6_500, 7_000, 7_500, 8_000, 9_000, 10_000];

/// Bracket entry thresholds for `RATIO_REWARD_BPS_V2`, same semantics as `RATIO_REWARD_THRESHOLDS`
/// (integer multiples of windowed production; 1 window = 24h at 10 BPS today). Reading:
/// 0/3/7/15/30/45/60/75/90 days held.
pub const RATIO_REWARD_THRESHOLDS_V2: [u64; 9] = [0, 3, 7, 15, 30, 45, 60, 75, 90];

/// Length (in blocks) of the trailing window over which a payout address's production (coinbase
/// earned) is summed. 24h at 10 BPS = 864_000 blocks. HARD CONSTRAINT: must stay `< pruning_depth`
/// (~30h) so the window always falls inside retained history and is reconstructible on IBD.
///
/// PRE-H3 ONLY. This window was applied in SELECTED-CHAIN block units, but the chain advances at
/// ~2.2 chain blocks/s (mergesets absorb the 10 BPS DAG width), so "24h" was really ~4.6 real
/// days — and drifting with mergeset size. Superseded at `pom_level_activation` by
/// [`RATIO_REWARD_WINDOW_DAA`].
pub const RATIO_REWARD_WINDOW: u64 = 864_000;

/// H3 ratio-reward window, in DAA score (true block count): 24h at 10 BPS = 864_000 DAA, a FIXED
/// real-time duration regardless of mergeset width. Used for blocks at/after `pom_level_activation`
/// together with per-blue production accounting (production = the base cuts of every PAID mergeset
/// blue, the exact mirror of coinbase payment — replacing the selected-chain-only accounting whose
/// ~1.7× connectivity bias and 4.6-day effective window drifted from the spec). The spec reading
/// "top bracket = 30 days of production held" is exact again. HARD CONSTRAINT: spans far less
/// selected-chain depth than the legacy window (~190k chain blocks), comfortably inside pruning.
pub const RATIO_REWARD_WINDOW_DAA: u64 = 864_000;

/// Coin-age (holder-reward v3) maturity period, in DAA score: 24h at 10 BPS. A coin younger
/// than this counts toward the ratio numerator at the linear prorata of its age (`v·age/W`);
/// at/after it, at its full face value. Set equal to the production window: 24h of holding is
/// enough to break bracket-farming by address hopping, while keeping the maturity ramp short so
/// normal holders are not over-penalised. See
/// KERYX-KRX/coin_age_holder_reward_spec.md §2/§9. Used at/after `coin_age_activation`.
pub const COIN_AGE_MATURITY_W: u64 = 864_000; // 24h at 10 BPS

/// Returns the `RATIO_REWARD_BPS` multiplier for a payout address given its `balance` and its
/// `production` over the trailing window. The caller MUST floor `production` at one block subsidy
/// (a zero-history / freshly-rotated address would otherwise hit the top bracket for free).
pub fn ratio_reward_bps(balance: u64, production: u64) -> u64 {
    ratio_bracket_bps(balance, production, &RATIO_REWARD_THRESHOLDS, &RATIO_REWARD_BPS)
}

/// Same as `ratio_reward_bps`, against the recalibrated `RATIO_REWARD_BPS_V2` table. Gated by
/// `coin_age_activation` (bundled into H4).
pub fn ratio_reward_bps_v2(balance: u64, production: u64) -> u64 {
    ratio_bracket_bps(balance, production, &RATIO_REWARD_THRESHOLDS_V2, &RATIO_REWARD_BPS_V2)
}

/// Division-free bracket scan shared by `ratio_reward_bps`/`ratio_reward_bps_v2`: bracket `i` is
/// reached iff `balance >= thresholds[i] * production`. Thresholds are ascending, so the first
/// failing bracket ends the scan. `u128` math avoids overflow on the `threshold * production`
/// product. `thresholds` and `bps_table` MUST be the same length.
fn ratio_bracket_bps(balance: u64, production: u64, thresholds: &[u64], bps_table: &[u64]) -> u64 {
    let mut bps = bps_table[0];
    for i in 0..thresholds.len() {
        if (balance as u128) >= (thresholds[i] as u128) * (production as u128) {
            bps = bps_table[i];
        } else {
            break;
        }
    }
    bps
}

use crate::{
    BlockLevel, KType,
    constants::STORAGE_MASS_PARAMETER,
    network::{NetworkId, NetworkType},
};
use keryx_addresses::Prefix;
use keryx_math::Uint256;
use serde::{Deserialize, Serialize};
use std::{
    cmp::min,
    ops::{Deref, DerefMut},
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForkActivation(u64);

impl ForkActivation {
    const NEVER: u64 = u64::MAX;
    const ALWAYS: u64 = 0;

    pub const fn new(daa_score: u64) -> Self {
        Self(daa_score)
    }

    pub const fn never() -> Self {
        Self(Self::NEVER)
    }

    pub const fn always() -> Self {
        Self(Self::ALWAYS)
    }

    /// Returns the actual DAA score triggering the activation. Should be used only
    /// for cases where the explicit value is required for computations (e.g., coinbase subsidy).
    /// Otherwise, **activation checks should always go through `self.is_active(..)`**
    pub fn daa_score(self) -> u64 {
        self.0
    }

    pub fn is_active(self, current_daa_score: u64) -> bool {
        current_daa_score >= self.0
    }

    /// Checks if the fork was "recently" activated, i.e., in the time frame of the provided range.
    /// This function returns false for forks that were always active, since they were never activated.
    pub fn is_within_range_from_activation(self, current_daa_score: u64, range: u64) -> bool {
        self != Self::always() && self.is_active(current_daa_score) && current_daa_score < self.0 + range
    }

    /// Checks if the fork is expected to be activated "soon", i.e., in the time frame of the provided range.
    /// Returns the distance from activation if so, or `None` otherwise.  
    pub fn is_within_range_before_activation(self, current_daa_score: u64, range: u64) -> Option<u64> {
        if !self.is_active(current_daa_score) && current_daa_score + range > self.0 { Some(self.0 - current_daa_score) } else { None }
    }
}

/// A consensus parameter which depends on forking activation
#[derive(Clone, Copy, Debug)]
pub struct ForkedParam<T: Copy> {
    pre: T,
    post: T,
    activation: ForkActivation,
}

impl<T: Copy> ForkedParam<T> {
    const fn new(pre: T, post: T, activation: ForkActivation) -> Self {
        Self { pre, post, activation }
    }

    pub const fn new_const(val: T) -> Self {
        Self { pre: val, post: val, activation: ForkActivation::never() }
    }

    pub fn activation(&self) -> ForkActivation {
        self.activation
    }

    pub fn get(&self, daa_score: u64) -> T {
        if self.activation.is_active(daa_score) { self.post } else { self.pre }
    }

    /// Returns the value before activation (=pre unless activation = always)
    pub fn before(&self) -> T {
        match self.activation.0 {
            ForkActivation::ALWAYS => self.post,
            _ => self.pre,
        }
    }

    /// Returns the permanent long-term value after activation (=post unless the activation is never scheduled)
    pub fn after(&self) -> T {
        match self.activation.0 {
            ForkActivation::NEVER => self.pre,
            _ => self.post,
        }
    }

    /// Maps the ForkedParam<T> to a new ForkedParam<U> by applying a map function on both pre and post
    pub fn map<U: Copy, F: Fn(T) -> U>(&self, f: F) -> ForkedParam<U> {
        ForkedParam::new(f(self.pre), f(self.post), self.activation)
    }
}

impl<T: Copy + Ord> ForkedParam<T> {
    /// Returns the min of `pre` and `post` values. Useful for non-consensus initializations
    /// which require knowledge of the value bounds.
    ///
    /// Note that if activation is not scheduled (set to never) then pre is always returned,
    /// and if activation is set to always (since inception), post will be returned.
    pub fn lower_bound(&self) -> T {
        match self.activation.0 {
            ForkActivation::NEVER => self.pre,
            ForkActivation::ALWAYS => self.post,
            _ => self.pre.min(self.post),
        }
    }

    /// Returns the max of `pre` and `post` values. Useful for non-consensus initializations
    /// which require knowledge of the value bounds.
    ///
    /// Note that if activation is not scheduled (set to never) then pre is always returned,
    /// and if activation is set to always (since inception), post will be returned.
    pub fn upper_bound(&self) -> T {
        match self.activation.0 {
            ForkActivation::NEVER => self.pre,
            ForkActivation::ALWAYS => self.post,
            _ => self.pre.max(self.post),
        }
    }
}

/// Blockrate-related consensus params.
/// Grouped together under a single struct because they are logically related and
/// in order to easily support **future BPS acceleration hardforks** (by simply adding
/// a forked instance of blockrate params to the main [`Params`]).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BlockrateParams {
    pub target_time_per_block: u64, // (milliseconds)
    pub ghostdag_k: KType,
    pub past_median_time_sample_rate: u64,
    pub difficulty_sample_rate: u64,
    pub max_block_parents: u8,
    pub mergeset_size_limit: u64,
    pub merge_depth: u64,
    pub finality_depth: u64,
    pub pruning_depth: u64,
    pub coinbase_maturity: u64,
}

impl BlockrateParams {
    pub const fn new<const BPS: u64>() -> Self {
        Self {
            target_time_per_block: Bps::<BPS>::target_time_per_block(),
            ghostdag_k: Bps::<BPS>::ghostdag_k(),
            past_median_time_sample_rate: Bps::<BPS>::past_median_time_sample_rate(),
            difficulty_sample_rate: Bps::<BPS>::difficulty_adjustment_sample_rate(),
            max_block_parents: Bps::<BPS>::max_block_parents(),
            mergeset_size_limit: Bps::<BPS>::mergeset_size_limit(),
            merge_depth: Bps::<BPS>::merge_depth_bound(),
            finality_depth: Bps::<BPS>::finality_depth(),
            pruning_depth: Bps::<BPS>::pruning_depth(),
            coinbase_maturity: Bps::<BPS>::coinbase_maturity(),
        }
    }

    pub const fn increase_max_block_parents(mut self, max_block_parents: u8) -> Self {
        if self.max_block_parents < max_block_parents {
            self.max_block_parents = max_block_parents;
        }
        self
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OverrideParams {
    /// Timestamp deviation tolerance (in seconds)
    pub timestamp_deviation_tolerance: Option<u64>,

    /// Size of the sampled block window that is used to calculate the past median time of each block
    pub past_median_time_window_size: Option<usize>,

    /// Size of the sampled block window that is used to calculate the required difficulty of each block
    pub difficulty_window_size: Option<usize>,

    /// The minimum size a difficulty window (full or sampled) must have to trigger a DAA calculation
    pub min_difficulty_window_size: Option<usize>,

    pub coinbase_payload_script_public_key_max_len: Option<u8>,
    pub max_coinbase_payload_len: Option<usize>,

    pub max_tx_inputs: Option<usize>,
    pub max_tx_outputs: Option<usize>,
    pub max_signature_script_len: Option<usize>,
    pub max_script_public_key_len: Option<usize>,
    pub mass_per_tx_byte: Option<u64>,
    pub mass_per_script_pub_key_byte: Option<u64>,
    pub mass_per_sig_op: Option<u64>,
    pub max_block_mass: Option<u64>,

    /// The parameter for scaling inverse KRX value to mass units (KIP-0009)
    pub storage_mass_parameter: Option<u64>,

    /// DAA score after which the pre-deflationary period switches to the deflationary period
    pub deflationary_phase_daa_score: Option<u64>,

    pub pre_deflationary_phase_base_subsidy: Option<u64>,
    pub skip_proof_of_work: Option<bool>,
    pub max_block_level: Option<BlockLevel>,
    pub pruning_proof_m: Option<u64>,

    /// Blockrate-related params
    pub blockrate: Option<BlockrateParams>,

    /// Target time per block prior to the crescendo hardfork (in milliseconds)
    pub pre_crescendo_target_time_per_block: Option<u64>,

    /// Crescendo activation DAA score
    pub crescendo_activation: Option<ForkActivation>,

    /// Model capability enforcement hardfork activation DAA score
    pub model_cap_enforcement_activation: Option<ForkActivation>,

    #[serde(skip)]
    pub inference_reward_minimums: Option<&'static [([u8; 32], u64)]>,
}

impl From<Params> for OverrideParams {
    fn from(p: Params) -> Self {
        Self {
            timestamp_deviation_tolerance: Some(p.timestamp_deviation_tolerance),
            pre_crescendo_target_time_per_block: Some(p.pre_crescendo_target_time_per_block),
            difficulty_window_size: Some(p.difficulty_window_size),
            past_median_time_window_size: Some(p.past_median_time_window_size),
            min_difficulty_window_size: Some(p.min_difficulty_window_size),
            coinbase_payload_script_public_key_max_len: Some(p.coinbase_payload_script_public_key_max_len),
            max_coinbase_payload_len: Some(p.max_coinbase_payload_len),
            max_tx_inputs: Some(p.max_tx_inputs),
            max_tx_outputs: Some(p.max_tx_outputs),
            max_signature_script_len: Some(p.max_signature_script_len),
            max_script_public_key_len: Some(p.max_script_public_key_len),
            mass_per_tx_byte: Some(p.mass_per_tx_byte),
            mass_per_script_pub_key_byte: Some(p.mass_per_script_pub_key_byte),
            mass_per_sig_op: Some(p.mass_per_sig_op),
            max_block_mass: Some(p.max_block_mass),
            storage_mass_parameter: Some(p.storage_mass_parameter),
            deflationary_phase_daa_score: Some(p.deflationary_phase_daa_score),
            pre_deflationary_phase_base_subsidy: Some(p.pre_deflationary_phase_base_subsidy),
            skip_proof_of_work: Some(p.skip_proof_of_work),
            max_block_level: Some(p.max_block_level),
            pruning_proof_m: Some(p.pruning_proof_m),
            blockrate: Some(p.blockrate),
            crescendo_activation: Some(p.crescendo_activation),
            model_cap_enforcement_activation: Some(p.model_cap_enforcement_activation),
            inference_reward_minimums: Some(p.inference_reward_minimums),
        }
    }
}

/// Consensus parameters. Contains settings and configurations which are consensus-sensitive.
/// Changing one of these on a network node would exclude and prevent it from reaching consensus
/// with the other unmodified nodes.
#[derive(Clone, Debug)]
pub struct Params {
    pub dns_seeders: &'static [&'static str],
    pub net: NetworkId,
    pub genesis: GenesisBlock,

    /// Timestamp deviation tolerance (in seconds)
    pub timestamp_deviation_tolerance: u64,

    /// Defines the highest allowed proof of work difficulty value for a block as a [`Uint256`]
    pub max_difficulty_target: Uint256,

    /// Highest allowed proof of work difficulty as a floating number
    pub max_difficulty_target_f64: f64,

    /// Size of the sampled block window that is used to calculate the past median time of each block
    pub past_median_time_window_size: usize,

    /// Size of the sampled block window that is used to calculate the required difficulty of each block
    pub difficulty_window_size: usize,

    /// The minimum size a difficulty window must have to trigger a DAA calculation
    pub min_difficulty_window_size: usize,

    pub coinbase_payload_script_public_key_max_len: u8,
    pub max_coinbase_payload_len: usize,

    pub max_tx_inputs: usize,
    pub max_tx_outputs: usize,
    pub max_signature_script_len: usize,
    pub max_script_public_key_len: usize,

    pub mass_per_tx_byte: u64,
    pub mass_per_script_pub_key_byte: u64,
    pub mass_per_sig_op: u64,
    pub max_block_mass: u64,

    /// The parameter for scaling inverse KRX value to mass units (KIP-0009)
    pub storage_mass_parameter: u64,

    /// DAA score after which the pre-deflationary period switches to the deflationary period
    pub deflationary_phase_daa_score: u64,

    pub pre_deflationary_phase_base_subsidy: u64,
    pub skip_proof_of_work: bool,
    pub max_block_level: BlockLevel,
    pub pruning_proof_m: u64,

    /// Blockrate-related params
    pub blockrate: BlockrateParams,

    /// Target time per block prior to the crescendo hardfork (in milliseconds).
    /// Required permanently in order to calculate the subsidy month from the current DAA score
    pub pre_crescendo_target_time_per_block: u64,

    /// Crescendo activation DAA score
    pub crescendo_activation: ForkActivation,

    /// Model capability enforcement hardfork activation DAA score.
    /// After this score, blocks containing AiResponse txs whose model_id is not
    /// declared in the coinbase ai:cap: field are rejected by consensus.
    pub model_cap_enforcement_activation: ForkActivation,

    /// Per-model minimum inference_reward (sompi) enforced from `model_cap_enforcement_activation`.
    /// AiRequest txs below the minimum for their model_id are rejected.
    /// Fulfilled inference_rewards are redirected from the fee burn to the responding miner.
    pub inference_reward_minimums: &'static [([u8; 32], u64)],

    /// OPoI v2 hardfork activation DAA score. From this score the uncensored model
    /// lineup (`inference_reward_minimums_v2`) replaces the legacy `inference_reward_minimums`.
    /// DAA-gated so IBD re-validation keeps the legacy table for historical blocks
    /// (swapping it unconditionally would diverge the UTXO set on pre-fork history).
    pub opoi_v2_activation: ForkActivation,

    /// OPoI v2 per-model minimum inference_reward (sompi). Used in place of
    /// `inference_reward_minimums` for blocks at or after `opoi_v2_activation`.
    pub inference_reward_minimums_v2: &'static [([u8; 32], u64)],

    /// Proof-of-Model possession activation DAA score. At/after this score every block must
    /// carry a valid `PomProof` (verified in `post_pow_validation` against `POM_TIERS`).
    /// DAA-gated so IBD re-validation of pre-fork history keeps the legacy self-verifying PoW.
    pub pom_activation: ForkActivation,
    /// H2 lineup refresh (very-light Qwen3-1.7B + 70B-Q2) gate. Selects the 5-tier `POM_TIERS_H2` /
    /// `TIER_REWARD_BPS_H2` over the legacy 4-tier sets. MUST equal the miner's
    /// `VERY_LIGHT_ACTIVATION_DAA` for the running network. Dormant until the H2 DAA is chosen.
    pub very_light_activation: ForkActivation,

    /// PoM block-level hardfork (H3) activation DAA score. At/after this score:
    /// (1) `Header::pom_final_state` becomes consensus — hashed into the block hash and
    /// cross-checked against `PomProof::final_state` in body validation; (2) the PoW value
    /// `pom_pow_value(pom_final_state, pre_pow_hash)` is re-checked against the target at
    /// header validation (header-only, no weights needed); (3) the block level is derived
    /// from that value again instead of being forced to 0, un-degenerating the pruning proof
    /// (level 0 alone carried the whole post-`pom_activation` span — 3.26M headers and
    /// growing — which killed from-scratch IBD). Blocks in the dead zone
    /// [`pom_activation`, here) keep level 0 forever; the proof fully self-heals once the
    /// pruning point passes this score. MUST equal the miner's activation for the running
    /// network — miners must fill `pom_final_state` from the winning walk.
    pub pom_level_activation: ForkActivation,

    /// H2 per-model minimum inference_reward gate. From this score `inference_reward_minimums_v2_h2`
    /// (5-tier, incl. Qwen3-1.7B + 70B-Q2) replaces `inference_reward_minimums_v2`. MUST be a FUTURE
    /// DAA — never `very_light_activation` (already past) — so IBD re-validation of historical blocks
    /// keeps the v2 table; a stricter minimum applied retroactively would reject a past AiRequest and
    /// diverge the UTXO set. Node-only enforcement (no miner involvement).
    pub inference_min_h2_activation: ForkActivation,

    /// H2 (5-tier) per-model minimum inference_reward (sompi). Used in place of
    /// `inference_reward_minimums_v2` for blocks at or after `inference_min_h2_activation`.
    pub inference_reward_minimums_v2_h2: &'static [([u8; 32], u64)],

    /// PoW SALT v2 hardfork activation DAA score.
    /// After this score, `KERYX_MATRIX_SALT_V2` is used for matrix generation instead of v1.
    /// Any miner binary compiled against v1 will compute a different matrix and its blocks
    /// will fail PoW validation — this is the forced-update mechanism.
    /// Set to `ForkActivation::never()` to disable (default for mainnet until announced).
    pub pow_salt_v2_activation: ForkActivation,

    /// PoW SALT v4 hardfork activation DAA score (chain relaunch).
    /// After this score, `KERYX_MATRIX_SALT_V4` is used for matrix generation instead of v2.
    /// This forks cleanly away from the abandoned SALT-v3 / diff-spiral chain while keeping
    /// stock difficulty (no genesis reset). Same forced-update mechanism as v2.
    pub pow_salt_v4_activation: ForkActivation,

    /// Ratio-reward (holder-weighted miner-cut bonus) activation DAA score. At/after this score
    /// the coinbase miner cut is scaled by the producer's holder ratio bracket (`RATIO_REWARD_BPS`,
    /// computed by the node from the balance + windowed-production indexes). DAA-gated so IBD
    /// re-validation of pre-fork history is unaffected (empty map ⇒ full cut, no burn).
    pub ratio_reward_activation: ForkActivation,

    /// Ratio/tier coinbase VERIFICATION boundary. Coinbase ratio/tier outputs are re-derived and
    /// checked only at/after this score; below it they are trusted (the `utxo_commitment` still pins
    /// the UTXO set). Rationale: the pre-relaunch chain's coinbases were built by the old regime with a
    /// path-dependent windowed-production value that NO clean recomputation (legacy rebuild or
    /// prefix-sum) reproduces — so that history is intrinsically non-revalidatable and must be trusted,
    /// while the cleanly re-mined chain at/after the relaunch frontier is fully verified by the
    /// deterministic prefix-sum. Mainnet = the relaunch frontier; other networks = 0 (verify all, no
    /// corrupted history). Distinct from `trust_coinbase()` (node-local: archival/env/catch-up) — this
    /// is a consensus rule, identical on every node, and also covers the IBD/staging path.
    pub ratio_verification_activation: ForkActivation,

    /// Difficulty-reset hardfork activation DAA score (chain relaunch). At/after this score the
    /// difficulty window discards every sample that precedes the reset, so the chain resumes at
    /// `genesis.bits` and the DAA re-converges to the post-fork (PoM-only) hashrate within one
    /// window. Needed when a hardfork sheds most of the pre-fork hashrate (e.g. non-PoM pools cut
    /// off at `pom_activation`), leaving stock difficulty far too high and the chain frozen.
    /// Forward-only: blocks below this score keep their original bits (no re-org). `never()` to disable.
    pub difficulty_reset_activation: ForkActivation,

    /// SECOND difficulty-reset window, for the H4 relaunch. Additive to `difficulty_reset_activation`:
    /// each reset is a self-contained window `[activation, activation + full_window)` that forces
    /// `genesis.bits`. A dedicated field (rather than moving the existing one) because the H2 reset at
    /// `difficulty_reset_activation` is load-bearing consensus history — an archival node re-derives
    /// those blocks, so its window must never shift. Driven by `H4_ACTIVATION_DAA`. `never()` while H4
    /// is unscheduled.
    pub difficulty_reset_activation_h4: ForkActivation,

    /// THIRD difficulty-reset window, for the H5 relaunch. Additive to the previous two (same
    /// self-contained `[activation, activation + full_window)` → `genesis.bits` semantics); a
    /// dedicated field so the H2/H4 resets stay load-bearing consensus history that archival nodes
    /// re-derive unshifted. Driven by `H5_ACTIVATION_DAA` (H5 sheds the ~92% dominant hashrate at
    /// relaunch, so stock difficulty is far too high). `never()` while H5 is unscheduled.
    pub difficulty_reset_activation_h5: ForkActivation,

    /// Single H5 bundle activation, keyed on the selected parent's DAA score. Drives every H5
    /// feature (parallel-block cap now; non-foldable walk + tier-0 swap when they land). Driven by
    /// `H5_ACTIVATION_DAA`. `never()` on nets where H5 does not apply.
    pub h5_activation: ForkActivation,

    /// Length (in blocks) of the trailing selected-chain window over which a payout address's
    /// production (base coinbase miner-cut earned) is summed for the ratio-reward denominator.
    /// Defaults to `RATIO_REWARD_WINDOW`; a Params field (not the const) so tests can shrink it to
    /// exercise the window slide. HARD CONSTRAINT: must stay `< pruning_depth`.
    /// PRE-H3 ONLY — superseded by `ratio_reward_window_daa` at `pom_level_activation`.
    pub ratio_reward_window: u64,

    /// H3 ratio-reward window in DAA score (fixed real-time duration, per-blue accounting era).
    /// Defaults to `RATIO_REWARD_WINDOW_DAA`; a Params field so tests can shrink it.
    pub ratio_reward_window_daa: u64,

    /// Coin-age holder-reward (v3, H4) activation DAA score. At/after this score the ratio-reward
    /// numerator switches from the instantaneous balance to the per-coin-capped effective balance
    /// (`coin_age::eff_balance_from_buckets`), closing the "rotation" exploit (bracket-farming by
    /// hopping the pot across fresh addresses). Requires the `effective_daa` UtxoEntry field
    /// (hard fork: UTXO commitment changes at this boundary). `never()` until H4 is scheduled.
    pub coin_age_activation: ForkActivation,

    /// Coin-age VERIFICATION boundary (mirrors `ratio_verification_activation`): coinbase outputs
    /// are re-derived with the coin-age numerator and enforced only at/after this score, covering
    /// the post-activation migration window where trusted transition blocks may precede full
    /// cross-node determinism. `never()` until H4 is scheduled.
    pub coin_age_verification_activation: ForkActivation,

    /// Coin-age maturity period in DAA score. Defaults to `COIN_AGE_MATURITY_W` (24h); a
    /// Params field (not the const) so tests can shrink it to exercise the maturity ramp and
    /// the immature→mature bucket promotion.
    pub coin_age_maturity_w: u64,
}

impl Params {
    /// Returns the past median time sample rate
    #[inline]
    #[must_use]
    pub fn past_median_time_sample_rate(&self) -> u64 {
        self.blockrate.past_median_time_sample_rate
    }

    /// Returns the difficulty sample rate
    #[inline]
    #[must_use]
    pub fn difficulty_sample_rate(&self) -> u64 {
        self.blockrate.difficulty_sample_rate
    }

    /// Returns the target time per block
    #[inline]
    #[must_use]
    pub fn target_time_per_block(&self) -> u64 {
        self.blockrate.target_time_per_block
    }

    /// Returns the expected number of blocks per second
    #[inline]
    #[must_use]
    pub fn bps(&self) -> u64 {
        1000 / self.blockrate.target_time_per_block
    }

    /// Returns the expected number of blocks per second throughout history (currently represented as [`ForkedParam`]).
    /// Required permanently in order to calculate the subsidy month from the current DAA score.
    #[inline]
    #[must_use]
    pub fn bps_history(&self) -> ForkedParam<u64> {
        ForkedParam::new(
            1000 / self.pre_crescendo_target_time_per_block,
            1000 / self.blockrate.target_time_per_block,
            self.crescendo_activation,
        )
    }

    pub fn ghostdag_k(&self) -> KType {
        self.blockrate.ghostdag_k
    }

    pub fn max_block_parents(&self) -> u8 {
        self.blockrate.max_block_parents
    }

    pub fn mergeset_size_limit(&self) -> u64 {
        self.blockrate.mergeset_size_limit
    }

    pub fn merge_depth(&self) -> u64 {
        self.blockrate.merge_depth
    }

    pub fn finality_depth(&self) -> u64 {
        self.blockrate.finality_depth
    }

    pub fn pruning_depth(&self) -> u64 {
        self.blockrate.pruning_depth
    }

    pub fn coinbase_maturity(&self) -> u64 {
        self.blockrate.coinbase_maturity
    }

    pub fn finality_duration_in_milliseconds(&self) -> u64 {
        self.blockrate.target_time_per_block * self.blockrate.finality_depth
    }

    pub fn difficulty_window_duration_in_block_units(&self) -> u64 {
        self.blockrate.difficulty_sample_rate * self.difficulty_window_size as u64
    }

    pub fn expected_difficulty_window_duration_in_milliseconds(&self) -> u64 {
        self.blockrate.target_time_per_block * self.blockrate.difficulty_sample_rate * self.difficulty_window_size as u64
    }

    /// Returns the depth at which the anticone of a chain block is final (i.e., is a permanently closed set).
    /// Based on the analysis at <https://github.com/kaspanet/docs/blob/main/Reference/prunality/Prunality.pdf>
    /// and on the decomposition of merge depth (rule R-I therein) from finality depth (φ)
    pub fn anticone_finalization_depth(&self) -> u64 {
        let anticone_finalization_depth = self.blockrate.finality_depth
            + self.blockrate.merge_depth
            + 4 * self.blockrate.mergeset_size_limit * self.blockrate.ghostdag_k as u64
            + 2 * self.blockrate.ghostdag_k as u64
            + 2;

        // In mainnet it's guaranteed that `self.pruning_depth` is greater
        // than `anticone_finalization_depth`, but for some tests we use
        // a smaller (unsafe) pruning depth, so we return the minimum of
        // the two to avoid a situation where a block can be pruned and
        // not finalized.
        min(self.blockrate.pruning_depth, anticone_finalization_depth)
    }

    pub fn network_name(&self) -> String {
        self.net.to_prefixed()
    }

    pub fn prefix(&self) -> Prefix {
        self.net.into()
    }

    pub fn default_p2p_port(&self) -> u16 {
        self.net.default_p2p_port()
    }

    pub fn default_rpc_port(&self) -> u16 {
        self.net.default_rpc_port()
    }

    pub fn override_params(self, overrides: OverrideParams) -> Self {
        Self {
            dns_seeders: self.dns_seeders,
            net: self.net,
            genesis: self.genesis.clone(),

            timestamp_deviation_tolerance: overrides.timestamp_deviation_tolerance.unwrap_or(self.timestamp_deviation_tolerance),

            max_difficulty_target: self.max_difficulty_target,
            max_difficulty_target_f64: self.max_difficulty_target_f64,

            difficulty_window_size: overrides.difficulty_window_size.unwrap_or(self.difficulty_window_size),
            past_median_time_window_size: overrides.past_median_time_window_size.unwrap_or(self.past_median_time_window_size),
            min_difficulty_window_size: overrides.min_difficulty_window_size.unwrap_or(self.min_difficulty_window_size),

            coinbase_payload_script_public_key_max_len: overrides
                .coinbase_payload_script_public_key_max_len
                .unwrap_or(self.coinbase_payload_script_public_key_max_len),

            max_coinbase_payload_len: overrides.max_coinbase_payload_len.unwrap_or(self.max_coinbase_payload_len),

            max_tx_inputs: overrides.max_tx_inputs.unwrap_or(self.max_tx_inputs),
            max_tx_outputs: overrides.max_tx_outputs.unwrap_or(self.max_tx_outputs),
            max_signature_script_len: overrides.max_signature_script_len.unwrap_or(self.max_signature_script_len),
            max_script_public_key_len: overrides.max_script_public_key_len.unwrap_or(self.max_script_public_key_len),
            mass_per_tx_byte: overrides.mass_per_tx_byte.unwrap_or(self.mass_per_tx_byte),
            mass_per_script_pub_key_byte: overrides.mass_per_script_pub_key_byte.unwrap_or(self.mass_per_script_pub_key_byte),
            mass_per_sig_op: overrides.mass_per_sig_op.unwrap_or(self.mass_per_sig_op),
            max_block_mass: overrides.max_block_mass.unwrap_or(self.max_block_mass),

            storage_mass_parameter: overrides.storage_mass_parameter.unwrap_or(self.storage_mass_parameter),

            deflationary_phase_daa_score: overrides.deflationary_phase_daa_score.unwrap_or(self.deflationary_phase_daa_score),

            pre_deflationary_phase_base_subsidy: overrides
                .pre_deflationary_phase_base_subsidy
                .unwrap_or(self.pre_deflationary_phase_base_subsidy),

            skip_proof_of_work: overrides.skip_proof_of_work.unwrap_or(self.skip_proof_of_work),

            max_block_level: overrides.max_block_level.unwrap_or(self.max_block_level),

            pruning_proof_m: overrides.pruning_proof_m.unwrap_or(self.pruning_proof_m),

            blockrate: overrides.blockrate.clone().unwrap_or(self.blockrate.clone()),

            pre_crescendo_target_time_per_block: overrides
                .pre_crescendo_target_time_per_block
                .unwrap_or(self.pre_crescendo_target_time_per_block),

            crescendo_activation: overrides.crescendo_activation.unwrap_or(self.crescendo_activation),

            model_cap_enforcement_activation: overrides
                .model_cap_enforcement_activation
                .unwrap_or(self.model_cap_enforcement_activation),

            inference_reward_minimums: overrides
                .inference_reward_minimums
                .unwrap_or(self.inference_reward_minimums),

            opoi_v2_activation: self.opoi_v2_activation,

            inference_reward_minimums_v2: self.inference_reward_minimums_v2,

            pom_activation: self.pom_activation,

            very_light_activation: self.very_light_activation,

            pom_level_activation: self.pom_level_activation,

            inference_min_h2_activation: self.inference_min_h2_activation,
            inference_reward_minimums_v2_h2: self.inference_reward_minimums_v2_h2,

            pow_salt_v2_activation: self.pow_salt_v2_activation,

            pow_salt_v4_activation: self.pow_salt_v4_activation,

            ratio_reward_activation: self.ratio_reward_activation,
            ratio_verification_activation: self.ratio_verification_activation,
            difficulty_reset_activation: self.difficulty_reset_activation,
            difficulty_reset_activation_h4: self.difficulty_reset_activation_h4,
            difficulty_reset_activation_h5: self.difficulty_reset_activation_h5,
            h5_activation: self.h5_activation,

            ratio_reward_window: self.ratio_reward_window,
            ratio_reward_window_daa: self.ratio_reward_window_daa,

            coin_age_activation: self.coin_age_activation,
            coin_age_verification_activation: self.coin_age_verification_activation,
            coin_age_maturity_w: self.coin_age_maturity_w,
        }
    }
}

impl Deref for Params {
    type Target = BlockrateParams;

    fn deref(&self) -> &Self::Target {
        &self.blockrate
    }
}

impl DerefMut for Params {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.blockrate
    }
}

impl From<NetworkType> for Params {
    fn from(value: NetworkType) -> Self {
        match value {
            NetworkType::Mainnet => MAINNET_PARAMS,
            NetworkType::Testnet => TESTNET_PARAMS,
            NetworkType::Devnet => DEVNET_PARAMS,
            NetworkType::Simnet => SIMNET_PARAMS,
        }
    }
}

impl From<NetworkId> for Params {
    fn from(value: NetworkId) -> Self {
        match value.network_type {
            NetworkType::Mainnet => MAINNET_PARAMS,
            NetworkType::Testnet => match value.suffix {
                Some(10) => TESTNET_PARAMS,
                Some(x) => panic!("Testnet suffix {} is not supported", x),
                None => panic!("Testnet suffix not provided"),
            },
            NetworkType::Devnet => DEVNET_PARAMS,
            NetworkType::Simnet => SIMNET_PARAMS,
        }
    }
}

pub const MAINNET_PARAMS: Params = Params {
    // A literal IP is valid here: the seeder string is resolved through
    // `(seeder, default_port).to_socket_addrs()`, which parses an IP address before
    // falling back to a DNS lookup. It acts as a fixed bootstrap peer on port 22111.
    dns_seeders: &["seed.keryx-labs.com", "141.95.35.181"],
    net: NetworkId::new(NetworkType::Mainnet),
    genesis: GENESIS,
    timestamp_deviation_tolerance: TIMESTAMP_DEVIATION_TOLERANCE,
    max_difficulty_target: MAX_DIFFICULTY_TARGET,
    max_difficulty_target_f64: MAX_DIFFICULTY_TARGET_AS_F64,
    past_median_time_window_size: MEDIAN_TIME_SAMPLED_WINDOW_SIZE as usize,
    difficulty_window_size: DIFFICULTY_SAMPLED_WINDOW_SIZE as usize,
    min_difficulty_window_size: MIN_DIFFICULTY_WINDOW_SIZE,
    coinbase_payload_script_public_key_max_len: 150,
    max_coinbase_payload_len: 2048,

    // Limit the cost of calculating compute/transient/storage masses
    max_tx_inputs: 1000,
    max_tx_outputs: 1000,
    // Transient mass enforces a limit of 125Kb, however script engine max scripts size is 10Kb so there's no point in surpassing that.
    max_signature_script_len: 10_000,
    // Compute mass enforces a limit of ~45.5Kb, however script engine max scripts size is 10Kb so there's no point in surpassing that.
    // Note that storage mass will kick in and gradually penalize also for lower lengths (generalized KIP-0009, plurality will be high).
    max_script_public_key_len: 10_000,

    mass_per_tx_byte: 1,
    mass_per_script_pub_key_byte: 10,
    mass_per_sig_op: 1000,
    max_block_mass: 500_000,

    storage_mass_parameter: STORAGE_MASS_PARAMETER,

    // Keryx launches at 10 BPS from genesis with Crescendo always active.
    // No pre-emission bootstrapping phase is needed — the emission schedule starts at block 0.
    deflationary_phase_daa_score: 0,
    pre_deflationary_phase_base_subsidy: TenBps::pre_deflationary_phase_base_subsidy(),
    skip_proof_of_work: false,
    max_block_level: 225,
    pruning_proof_m: 1000,

    blockrate: BlockrateParams::new::<10>(),

    pre_crescendo_target_time_per_block: TenBps::target_time_per_block(),

    crescendo_activation: ForkActivation::new(0),

    // Hardfork activation: 2026-05-28 15:00 UTC — DAA 11_409_033 + ~4_140_000 (115h × 10 BPS).
    model_cap_enforcement_activation: ForkActivation::new(15_550_000),
    inference_reward_minimums: INFERENCE_REWARD_MINIMUMS,

    // OPoI v2: uncensored lineup swap. Mainnet H = DAA 37_780_000 (2026-06-26 18:00 UTC), bundled
    // with PoM + ratio-reward into a single hardfork. MUST equal the miner's OPOI_V2_ACTIVATION_DAA.
    opoi_v2_activation: ForkActivation::new(37_780_000),
    inference_reward_minimums_v2: INFERENCE_REWARD_MINIMUMS_V2,

    // PoM possession: mainnet H = DAA 37_780_000 (2026-06-26 18:00 UTC). This is a mining-algorithm
    // hardfork (kHeavyHash → Proof-of-Model) — every miner MUST run a PoM binary with the pinned
    // GGUF models by H, and pom_activation MUST equal the miner's POM_ACTIVATION_DAA, or its blocks
    // are rejected and it forks off the chain.
    pom_activation: ForkActivation::new(37_780_000),
    very_light_activation: ForkActivation::new(38_951_445), // H2 = frozen frontier; mirrors miner VERY_LIGHT_ACTIVATION_DAA

    // PoM block-level hardfork (H3): restores header-only PoW verification + real block levels
    // (pruning proof un-degeneration, from-scratch IBD), salts the pph words feeding the PoM
    // folds (POM_H3_PPH_SALT, forced update) and aligns the coinbase output cap with the OPoI
    // builder (3*(K+1)+4). Full hardfork — header format + hash change, every node AND miner
    // must upgrade before this score. DAA picked 2026-07-05 08:49 UTC (tip 43,117,871)
    // targeting activation ≈ 2026-07-05 18:00 UTC (~17:55–18:10 at 10–9.7 DAA/s).
    // MUST mirror the miner's H3 activation (POM_LEVEL_ACTIVATION_DAA) for the running network.
    pom_level_activation: ForkActivation::new(43_450_000),

    // H2 inference_reward minimums (adds Qwen3-1.7B + 70B-Q2, missed when the H2 lineup shipped).
    // Gated at the H3 DAA (43_450_000): H3 is a hard fork that forces every node onto v1.3.1,
    // which already carries this floor table, so the minimums activate in the same single event
    // instead of a separate ~2026-07-09 point to coordinate. Soft-fork semantics (new-valid ⊆
    // old-valid). NOT gated at very_light_activation (past) to avoid re-validation divergence.
    inference_min_h2_activation: ForkActivation::new(43_450_000),
    inference_reward_minimums_v2_h2: INFERENCE_REWARD_MINIMUMS_V2_H2,

    // PoW SALT v2: emergency activation 2026-05-30 ~15:00 UTC.
    // DAA estimate: 16_501_908 (current) + 774_000 (21.5h × 10 BPS) = 17_275_908 → rounded down for 2 min margin.
    pow_salt_v2_activation: ForkActivation::new(17_275_000),

    // PoW SALT v4: chain relaunch on stock difficulty. At this score the salt switches v2→v4,
    // forking cleanly away from the abandoned SALT-v3 / diff-1-spiral chain. Same DAA as the
    // old v3 gate so a datadir restored from before this point continues seamlessly into v4.
    pow_salt_v4_activation: ForkActivation::new(21_932_751),

    // Ratio-reward (holder-weighted miner cut). Mainnet activation H = DAA 37_780_000, targeting
    // 2026-06-26 18:00 UTC at 10 BPS (measured: DAA 34_950_043 at 2026-06-23 11:24 UTC; +282_960 s
    // × 10 = +2_829_600, rounded up ~36 s for a small margin so it lands at/after the announced time).
    // Node-only gate (the miner has no ratio-reward logic). Before H the placeholder map is empty ⇒
    // no-op, IBD/old blocks unaffected.
    ratio_reward_activation: ForkActivation::new(37_780_000),
    // Coinbase ratio/tier enforcement boundary. Confirmed cross-node determinism (archival vs pruned
    // compute identical balance + prefix-sum production ⇒ identical coinbase) in observe-only mode, so
    // enforcement is re-enabled at/after this DAA. Set ABOVE the post-relaunch transition blocks (which
    // were mined with a stale balance index before the fix) so they stay trusted; only the cleanly
    // computed blocks above it are enforced. Below this score the coinbase is trusted (utxo_commitment
    // pins state) — covers both the non-revalidatable pre-relaunch history and the transition blocks.
    ratio_verification_activation: ForkActivation::new(38_980_000),
    // Difficulty reset (chain relaunch). H = DAA 37_780_000 shed almost all pre-fork hashrate
    // (non-PoM pools cut off), leaving stock difficulty calibrated to ~456 GH/s while only the
    // PoM hashrate (~tens of MH/s) remained valid → chain froze at the fork.
    // Gated AT the fork DAA (= ratio-reward activation, the clean/corrupt boundary): the relaunch
    // base is a pre-fork datadir synced to the last block with daa < H (the fork-era blocks at
    // daa >= H are auto-rejected by the deterministic coinbase/difficulty), so re-mining starts
    // with virtual_daa_score >= H and the reset fires on the very first re-mined block. The reset
    // filters the difficulty window down to samples with daa_score >= this score — only the top
    // boundary layer (well under MIN_DIFFICULTY_WINDOW_SIZE=150) — so the calc falls back to
    // genesis.bits. The chain relaunches at the launch target and the DAA re-converges upward to
    // the real PoM hashrate within one window. MUST match across all honest nodes.
    difficulty_reset_activation: ForkActivation::new(38_951_445),
    // H4 relaunch difficulty reset — additive, driven by the single H4 flip point.
    difficulty_reset_activation_h4: ForkActivation::new(H4_ACTIVATION_DAA),
    // H5 relaunch difficulty reset — additive, gated at the same DAA as the H5 bundle. The relaunch
    // base is the frozen datadir at the current tip (daa 59_009_012); the gate is tip+1 (59_009_013)
    // so the frozen tip stays on the H4 side and the first re-mined block (daa >= H5) fires the
    // reset, same as H4.
    difficulty_reset_activation_h5: ForkActivation::new(H5_ACTIVATION_DAA),
    // H5 bundle gate — set to the relaunch tip DAA. Every H5 feature flips at this score.
    h5_activation: ForkActivation::new(H5_ACTIVATION_DAA),
    ratio_reward_window: RATIO_REWARD_WINDOW,
    ratio_reward_window_daa: RATIO_REWARD_WINDOW_DAA,

    // Coin-age holder-reward (v3): DORMANT until the H4 hard fork is scheduled. The whole
    // machinery (effective_daa UtxoEntry field, bucket indexes, maturation queue), plus the
    // recalibrated ratio-reward v2 bracket table, gates here — one hardfork, one gate.
    // Both driven by the single `H4_ACTIVATION_DAA` flip point (set it at release).
    coin_age_activation: ForkActivation::new(H4_ACTIVATION_DAA),
    coin_age_verification_activation: ForkActivation::new(H4_ACTIVATION_DAA),
    coin_age_maturity_w: COIN_AGE_MATURITY_W,
};

pub const TESTNET_PARAMS: Params = Params {
    dns_seeders: &[],
    net: NetworkId::with_suffix(NetworkType::Testnet, 10),
    genesis: TESTNET_GENESIS,
    timestamp_deviation_tolerance: TIMESTAMP_DEVIATION_TOLERANCE,
    max_difficulty_target: MAX_DIFFICULTY_TARGET,
    max_difficulty_target_f64: MAX_DIFFICULTY_TARGET_AS_F64,
    past_median_time_window_size: MEDIAN_TIME_SAMPLED_WINDOW_SIZE as usize,
    difficulty_window_size: DIFFICULTY_SAMPLED_WINDOW_SIZE as usize,
    min_difficulty_window_size: MIN_DIFFICULTY_WINDOW_SIZE,
    coinbase_payload_script_public_key_max_len: 150,
    max_coinbase_payload_len: 2048,

    // Limit the cost of calculating compute/transient/storage masses
    max_tx_inputs: 1000,
    max_tx_outputs: 1000,
    // Transient mass enforces a limit of 125Kb, however script engine max scripts size is 10Kb so there's no point in surpassing that.
    max_signature_script_len: 10_000,
    // Compute mass enforces a limit of ~45.5Kb, however script engine max scripts size is 10Kb so there's no point in surpassing that.
    // Note that storage mass will kick in and gradually penalize also for lower lengths (generalized KIP-0009, plurality will be high).
    max_script_public_key_len: 10_000,

    mass_per_tx_byte: 1,
    mass_per_script_pub_key_byte: 10,
    mass_per_sig_op: 1000,
    max_block_mass: 500_000,

    storage_mass_parameter: STORAGE_MASS_PARAMETER,

    // Keryx testnet launches at 10 BPS from genesis with Crescendo always active.
    deflationary_phase_daa_score: 0,
    pre_deflationary_phase_base_subsidy: TenBps::pre_deflationary_phase_base_subsidy(),
    skip_proof_of_work: false,
    max_block_level: 250,
    pruning_proof_m: 1000,

    blockrate: BlockrateParams::new::<10>(),

    pre_crescendo_target_time_per_block: TenBps::target_time_per_block(),

    crescendo_activation: ForkActivation::new(0),

    // Testnet: model capability + inference_reward enforcement ON from genesis, so the
    // legacy lineup is enforced from block 0 and the v2 swap below is the only transition.
    model_cap_enforcement_activation: ForkActivation::always(),
    inference_reward_minimums: INFERENCE_REWARD_MINIMUMS,

    // Testnet mirrors the current mainnet state from genesis: every shipped fork
    // (OPoI v2, PoM possession, H2 lineup, H2 minimums, ratio-reward, H3 block levels)
    // is active at DAA 0, so the only transition exercised on this testnet is the
    // coin-age H4 rehearsal below.
    opoi_v2_activation: ForkActivation::new(0),
    inference_reward_minimums_v2: INFERENCE_REWARD_MINIMUMS_V2,

    // PoM possession: active from genesis (mainnet-state baseline).
    pom_activation: ForkActivation::new(0),
    very_light_activation: ForkActivation::new(0), // H2 5-tier lineup from genesis
    // H3 block levels: active from the first mined block (mainnet-state baseline; the H3
    // transition was rehearsed on the previous testnet). `new(1)` and NOT `new(0)`/`always()`:
    // this activation also drives the global header-hashing switch (`init_pom_level_activation`),
    // and genesis (daa 0) must keep its pinned legacy hash, which does not commit
    // `pom_final_state`. MUST mirror the miner's testnet activation.
    pom_level_activation: ForkActivation::new(1),
    inference_min_h2_activation: ForkActivation::new(0),
    inference_reward_minimums_v2_h2: INFERENCE_REWARD_MINIMUMS_V2_H2,

    // PoW SALT v2: testnet active from genesis (no mid-chain transition — only opoi_v2
    // at DAA 1000 transitions on this testnet). Mainnet keeps new(17_275_000).
    pow_salt_v2_activation: ForkActivation::new(0),

    // PoW SALT v4: active from genesis on testnet to mirror the live mainnet PoW (salt v4)
    // during the pre-PoM era, so the kHeavyHash→PoM transition test is a faithful H rehearsal.
    pow_salt_v4_activation: ForkActivation::new(0),

    // Ratio-reward: active from genesis (mainnet-state baseline).
    ratio_reward_activation: ForkActivation::new(0),
    ratio_verification_activation: ForkActivation::new(0), // no corrupted history on testnet — verify all
    // Testnet has no frozen-chain history to relaunch from; the H2 difficulty reset stays disabled.
    difficulty_reset_activation: ForkActivation::never(),
    // H4 reset ENABLED on testnet (mirrors the coin-age gates at 3_000) so the additive-reset path
    // is exercised end-to-end before mainnet. Harmless on a trivial-difficulty testnet.
    difficulty_reset_activation_h4: ForkActivation::new(3_000),
    // H5 reset ENABLED on testnet at the same gate as the H5 bundle, so the additive-reset path is
    // exercised end-to-end before mainnet. Harmless on a trivial-difficulty testnet.
    difficulty_reset_activation_h5: ForkActivation::new(3_000),
    // H5 bundle gate active early on testnet for validation.
    h5_activation: ForkActivation::new(3_000),
    // Testnet override: shrink the production window to ~100 s (1_000 blocks @ 10 BPS) instead of
    // the 24h mainnet value, so the holder ratio climbs through its brackets within a test session
    // rather than ~30 days. Still well under pruning_depth. Same shrink for the H3 daa window.
    ratio_reward_window: 1_000,
    ratio_reward_window_daa: 1_000,

    // Coin-age holder-reward (v3): the ONLY mid-chain transition on this testnet — from genesis
    // the chain is in the current-mainnet state (everything above at 0), then at DAA 3_000 the H4
    // boundary fires: FIFO anchors start, the per-coin muhash field appears, the ratio numerator
    // switches to the effective balance, the recalibrated ratio-reward v2 bracket table takes
    // over, and the shrunk maturity ramp (W=2_000) + queue promotions kick in. Rehearses the
    // mainnet H4 gate end-to-end.
    coin_age_activation: ForkActivation::new(3_000),
    coin_age_verification_activation: ForkActivation::new(3_000),
    coin_age_maturity_w: 2_000,
};

pub const SIMNET_PARAMS: Params = Params {
    dns_seeders: &[],
    net: NetworkId::new(NetworkType::Simnet),
    genesis: SIMNET_GENESIS,
    timestamp_deviation_tolerance: TIMESTAMP_DEVIATION_TOLERANCE,
    max_difficulty_target: MAX_DIFFICULTY_TARGET,
    max_difficulty_target_f64: MAX_DIFFICULTY_TARGET_AS_F64,
    past_median_time_window_size: MEDIAN_TIME_SAMPLED_WINDOW_SIZE as usize,
    difficulty_window_size: DIFFICULTY_SAMPLED_WINDOW_SIZE as usize,
    min_difficulty_window_size: MIN_DIFFICULTY_WINDOW_SIZE,

    deflationary_phase_daa_score: TenBps::deflationary_phase_daa_score(),
    pre_deflationary_phase_base_subsidy: TenBps::pre_deflationary_phase_base_subsidy(),
    coinbase_payload_script_public_key_max_len: 150,
    max_coinbase_payload_len: 2048,

    max_tx_inputs: 1000,
    max_tx_outputs: 1000,
    max_signature_script_len: 10_000,
    max_script_public_key_len: 10_000,

    mass_per_tx_byte: 1,
    mass_per_script_pub_key_byte: 10,
    mass_per_sig_op: 1000,
    max_block_mass: 500_000,

    storage_mass_parameter: STORAGE_MASS_PARAMETER,

    skip_proof_of_work: true, // For simnet only, PoW can be simulated by default
    max_block_level: 250,
    pruning_proof_m: PRUNING_PROOF_M,

    // For simnet, we deviate from default 10BPS configuration and allow at least 64 parents in order to support mempool benchmarks out of the box
    blockrate: BlockrateParams::new::<10>().increase_max_block_parents(64),

    pre_crescendo_target_time_per_block: TenBps::target_time_per_block(),

    crescendo_activation: ForkActivation::always(),

    model_cap_enforcement_activation: ForkActivation::always(),
    inference_reward_minimums: INFERENCE_REWARD_MINIMUMS,
    opoi_v2_activation: ForkActivation::always(),
    inference_reward_minimums_v2: INFERENCE_REWARD_MINIMUMS_V2,
    // PoM possession: dormant until miner emission (§6) + P2P transport land; flip with §7.
    pom_activation: ForkActivation::never(),
    very_light_activation: ForkActivation::never(),
    pom_level_activation: ForkActivation::never(),
    inference_min_h2_activation: ForkActivation::never(),
    inference_reward_minimums_v2_h2: INFERENCE_REWARD_MINIMUMS_V2_H2,
    pow_salt_v2_activation: ForkActivation::never(),
    pow_salt_v4_activation: ForkActivation::never(),
    ratio_reward_activation: ForkActivation::never(),
    ratio_verification_activation: ForkActivation::new(0), // verify all (no corrupted history)
    difficulty_reset_activation: ForkActivation::never(),
    difficulty_reset_activation_h4: ForkActivation::never(),
    difficulty_reset_activation_h5: ForkActivation::never(),
    h5_activation: ForkActivation::never(),
    ratio_reward_window: RATIO_REWARD_WINDOW,
    ratio_reward_window_daa: RATIO_REWARD_WINDOW_DAA,

    // Coin-age holder-reward (v3): DORMANT until the H4 hard fork is scheduled. The whole
    // machinery (effective_daa UtxoEntry field, bucket indexes, maturation queue) gates here.
    coin_age_activation: ForkActivation::never(),
    coin_age_verification_activation: ForkActivation::never(),
    coin_age_maturity_w: COIN_AGE_MATURITY_W,
};

pub const DEVNET_PARAMS: Params = Params {
    dns_seeders: &[],
    net: NetworkId::new(NetworkType::Devnet),
    genesis: DEVNET_GENESIS,
    timestamp_deviation_tolerance: TIMESTAMP_DEVIATION_TOLERANCE,
    max_difficulty_target: MAX_DIFFICULTY_TARGET,
    max_difficulty_target_f64: MAX_DIFFICULTY_TARGET_AS_F64,
    past_median_time_window_size: MEDIAN_TIME_SAMPLED_WINDOW_SIZE as usize,
    difficulty_window_size: DIFFICULTY_SAMPLED_WINDOW_SIZE as usize,
    min_difficulty_window_size: MIN_DIFFICULTY_WINDOW_SIZE,
    coinbase_payload_script_public_key_max_len: 150,
    max_coinbase_payload_len: 2048,

    max_tx_inputs: 1000,
    max_tx_outputs: 1000,
    max_signature_script_len: 10_000,
    max_script_public_key_len: 10_000,

    mass_per_tx_byte: 1,
    mass_per_script_pub_key_byte: 10,
    mass_per_sig_op: 1000,
    max_block_mass: 500_000,

    storage_mass_parameter: STORAGE_MASS_PARAMETER,

    deflationary_phase_daa_score: 0,
    pre_deflationary_phase_base_subsidy: TenBps::pre_deflationary_phase_base_subsidy(),
    skip_proof_of_work: false,
    max_block_level: 250,
    pruning_proof_m: 1000,

    blockrate: BlockrateParams::new::<10>(),

    pre_crescendo_target_time_per_block: TenBps::target_time_per_block(),

    crescendo_activation: ForkActivation::always(),

    model_cap_enforcement_activation: ForkActivation::always(),
    inference_reward_minimums: INFERENCE_REWARD_MINIMUMS,
    opoi_v2_activation: ForkActivation::always(),
    inference_reward_minimums_v2: INFERENCE_REWARD_MINIMUMS_V2,
    // PoM possession: dormant until miner emission (§6) + P2P transport land; flip with §7.
    pom_activation: ForkActivation::never(),
    very_light_activation: ForkActivation::never(),
    pom_level_activation: ForkActivation::never(),
    inference_min_h2_activation: ForkActivation::never(),
    inference_reward_minimums_v2_h2: INFERENCE_REWARD_MINIMUMS_V2_H2,
    pow_salt_v2_activation: ForkActivation::never(),
    pow_salt_v4_activation: ForkActivation::never(),
    ratio_reward_activation: ForkActivation::never(),
    ratio_verification_activation: ForkActivation::new(0), // verify all (no corrupted history)
    difficulty_reset_activation: ForkActivation::never(),
    difficulty_reset_activation_h4: ForkActivation::never(),
    difficulty_reset_activation_h5: ForkActivation::never(),
    h5_activation: ForkActivation::never(),
    ratio_reward_window: RATIO_REWARD_WINDOW,
    ratio_reward_window_daa: RATIO_REWARD_WINDOW_DAA,

    // Coin-age holder-reward (v3): DORMANT until the H4 hard fork is scheduled. The whole
    // machinery (effective_daa UtxoEntry field, bucket indexes, maturation queue) gates here.
    coin_age_activation: ForkActivation::never(),
    coin_age_verification_activation: ForkActivation::never(),
    coin_age_maturity_w: COIN_AGE_MATURITY_W,
};

#[cfg(test)]
mod ratio_reward_bps_tests {
    use super::*;

    const P: u64 = 1_000_000; // arbitrary windowed production

    #[test]
    fn v1_brackets_unchanged() {
        // Regression lock for the pre-existing table after factoring out `ratio_bracket_bps`.
        assert_eq!(ratio_reward_bps(0, P), 4_000);
        assert_eq!(ratio_reward_bps(1 * P, P), 5_200);
        assert_eq!(ratio_reward_bps(3 * P, P), 6_400);
        assert_eq!(ratio_reward_bps(7 * P, P), 7_600);
        assert_eq!(ratio_reward_bps(15 * P, P), 8_800);
        assert_eq!(ratio_reward_bps(30 * P, P), 10_000);
        assert_eq!(ratio_reward_bps(1_000 * P, P), 10_000); // holding far beyond top bracket stays capped
    }

    #[test]
    fn v2_brackets_exact_boundaries() {
        assert_eq!(ratio_reward_bps_v2(0, P), 5_000);
        assert_eq!(ratio_reward_bps_v2(3 * P, P), 5_500);
        assert_eq!(ratio_reward_bps_v2(7 * P, P), 6_000);
        assert_eq!(ratio_reward_bps_v2(15 * P, P), 6_500);
        assert_eq!(ratio_reward_bps_v2(30 * P, P), 7_000);
        assert_eq!(ratio_reward_bps_v2(45 * P, P), 7_500);
        assert_eq!(ratio_reward_bps_v2(60 * P, P), 8_000);
        assert_eq!(ratio_reward_bps_v2(75 * P, P), 9_000); // note: 85% bracket deliberately skipped
        assert_eq!(ratio_reward_bps_v2(90 * P, P), 10_000);
    }

    #[test]
    fn v2_never_exceeds_100_percent() {
        // No bracket above 100%, unlike a tier-compensation table would need — by design.
        assert!(RATIO_REWARD_BPS_V2.iter().all(|&bps| bps <= RATIO_REWARD_BPS_DIVISOR));
        assert_eq!(ratio_reward_bps_v2(u64::MAX / 2, P), 10_000); // extreme holding still capped at 100%
    }

    #[test]
    fn v2_one_below_each_threshold_stays_in_lower_bracket() {
        // Off-by-one just under each threshold must NOT round up to the next bracket.
        assert_eq!(ratio_reward_bps_v2(3 * P - 1, P), 5_000);
        assert_eq!(ratio_reward_bps_v2(90 * P - 1, P), 9_000);
    }
}
