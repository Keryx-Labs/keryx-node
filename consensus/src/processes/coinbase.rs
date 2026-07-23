use keryx_addresses::{Address, Prefix, Version};
use keryx_consensus_core::{
    BlockHashMap, BlockHashSet,
    coinbase::*,
    collateral::CHALLENGE_WINDOW_BLOCKS,
    config::params::{RATIO_REWARD_BPS_DIVISOR, TIER_REWARD_BPS_DIVISOR},
    errors::coinbase::{CoinbaseError, CoinbaseResult},
    subnets,
    tx::{ScriptPublicKey, ScriptVec, Transaction, TransactionOutput},
};
use keryx_hashes::{Hasher, TransactionHash};
use keryx_txscript::{
    opcodes::codes::{OpCheckSequenceVerify, OpCheckSig},
    script_builder::ScriptBuilder,
    standard::pay_to_address_script,
};
use std::convert::TryInto;

use crate::{constants, model::stores::ghostdag::GhostdagData};

// ── R&D allocation ────────────────────────────────────────────────────────────

/// Protocol-level R&D allocation: 5% of every block reward (500 basis points).
const RD_ALLOCATION_BPS: u64 = 500;
const RD_ALLOCATION_BPS_DIVISOR: u64 = 10_000;

// ── OPoI escrow split ─────────────────────────────────────────────────────────

/// 20 % of each blue-block reward is sent to the miner's escrow SPK (OPoI)
/// or burned (standard miner).  Applied on the gross reward before the R&D cut.
const ESCROW_RATE_BPS: u64 = 2_000;
const ESCROW_RATE_BPS_DIVISOR: u64 = 10_000;

/// Marker searched in coinbase extra_data to locate the escrow public key.
/// Format: `/escrow:<64-hex-chars-of-32-byte-schnorr-pubkey>`
const ESCROW_MARKER: &[u8] = b"/escrow:";
const ESCROW_PUBKEY_HEX_LEN: usize = 64;

/// Seed used to derive the provably-unspendable burn address.
/// `TransactionHash("KERYX_PROOF_OF_BURN_V1")` is not a valid EC point →
/// no private key exists → coins sent here are permanently removed from supply.
const BURN_SEED: &[u8] = b"KERYX_PROOF_OF_BURN_V1";

/// Mainnet address that receives the R&D allocation.
pub const RD_ALLOCATION_ADDRESS: &str =
    "keryx:qp8zp9wnpqhgygpsv25px8whw0ee7md72s0tgy78x5wt7ryk6w525aqm045zv";

/// Consensus bound on coinbase output count for the era selected by `h3` (`pom_level_activation`).
///
/// Pre-H3: the legacy `K + 2` cap inherited from the upstream one-output-per-blue builder.
/// Post-H3: the OPoI builder's structural max — up to three outputs per mergeset blue (fee
/// burn, miner cut, escrow/burn; blues are bounded by K + 1) plus four aggregates (red-fee
/// burn, red reward, R&D allocation, reward burn) = `3 * (K + 1) + 4` (379 on mainnet, K=124).
///
/// The legacy cap contradicts this builder: with zero fees (2 outputs per blue) a chain block
/// merging ≥ 62 blues can only build a coinbase every validator rejects, halting the network
/// — the 2026-07-04 storm peaked at 123 outputs (~60 blues), 2 blues short. Aligning the
/// validator is a consensus loosening, hence the H3 gate (all nodes upgrade at that DAA).
/// Root-caused by Dizzztroyer (PR #16).
pub fn coinbase_outputs_limit(ghostdag_k: u64, h3: bool) -> u64 {
    if h3 { 3 * (ghostdag_k + 1) + 4 } else { ghostdag_k + 2 }
}

// ─────────────────────────────────────────────────────────────────────────────

const LENGTH_OF_BLUE_SCORE: usize = size_of::<u64>();
const LENGTH_OF_SUBSIDY: usize = size_of::<u64>();
const LENGTH_OF_SCRIPT_PUB_KEY_VERSION: usize = size_of::<u16>();
const LENGTH_OF_SCRIPT_PUB_KEY_LENGTH: usize = size_of::<u8>();

const MIN_PAYLOAD_LENGTH: usize =
    LENGTH_OF_BLUE_SCORE + LENGTH_OF_SUBSIDY + LENGTH_OF_SCRIPT_PUB_KEY_VERSION + LENGTH_OF_SCRIPT_PUB_KEY_LENGTH;

// A year is 365.25 days; a month is 365.25/12 = 30.4375 days.
// SECONDS_PER_MONTH = 30.4375 * 24 * 60 * 60
const SECONDS_PER_MONTH: u64 = 2_629_800;

// ── Keryx emission constants ──────────────────────────────────────────────────

/// Genesis block reward per second (sompi/second).
///
/// At 10 BPS this yields 540_000_000 sompi/block = 5.4 KRX/block.
///
/// With monthly-granularity 4-year halvings the discrete geometric series converges:
///   total = S0 × SPM / (1 − 2^(−1/48))
///         ≈ 5.4e9 × 2_629_800 × 69.9
///         ≈ 9.92 B KRX
/// giving a hard cap of ≈ 9.92 B KRX for the main emission phase (< 10 B KRX).
///
/// Note: KRX/second = 54, KRX/block = 5.4 (at 10 BPS: sompi/block = S0 / BPS).
///
/// Emission milestones:
///   Year  4 →  50% mined (~4.96 B KRX) — first halving event
///   Year  8 →  75% mined
///   Year 12 →  87.5% mined
///   Year ~146 → integer reward reaches 0, tail emission activates
const KRX_GENESIS_REWARD_PER_SECOND: u64 = 5_400_000_000;

/// Halving interval in calendar months (48 months = 4-year halving cycle).
///
/// This ensures that no more than 50% of the total supply is emitted during
/// the first 4 years, spreading meaningful block rewards over ~146 years
/// before the tail emission floor takes over.
const KRX_HALVING_PERIOD_MONTHS: u32 = 48;

/// Tail emission rate (sompi/second) applied once the main schedule is
/// exhausted (~146 years after genesis).
///
/// At 10 BPS this equals 1 sompi/block — a non-zero floor that extends
/// miner incentives to 200+ years while keeping the effective supply cap
/// at ≈ 9.92 B KRX.
const KRX_TAIL_EMISSION_PER_SECOND: u64 = 10;

// ─────────────────────────────────────────────────────────────────────────────

/// Builds a CSV-timelocked P2PK script for the OPoI escrow output.
///
/// Script: `<CHALLENGE_WINDOW_BLOCKS> OP_CSV OP_DROP <pubkey_32> OP_CHECKSIG`
///
/// The output cannot be spent until the spending transaction's input sequence
/// satisfies the relative lock (>= CHALLENGE_WINDOW_BLOCKS deep), giving the
/// network the full challenge window to submit a fraud proof before the miner
/// claims their collateral.
fn build_escrow_script(pubkey_bytes: &[u8]) -> Option<ScriptPublicKey> {
    // Keryx's OP_CSV pops its argument — no OP_DROP needed after it.
    let script = ScriptBuilder::new()
        .add_sequence(CHALLENGE_WINDOW_BLOCKS)
        .ok()?
        .add_op(OpCheckSequenceVerify)
        .ok()?
        .add_data(pubkey_bytes)
        .ok()?
        .add_op(OpCheckSig)
        .ok()?
        .drain();
    Some(ScriptPublicKey::from_vec(0, script))
}

/// Returns the provably-unspendable burn SPK derived from BURN_SEED.
///
/// Called by the AiChallenge deposit validator to verify that a challenger's
/// deposit output actually targets the canonical burn address.
pub(crate) fn burn_script_public_key() -> ScriptPublicKey {
    let burn_hash = TransactionHash::hash(BURN_SEED);
    let burn_address = Address::new(Prefix::Mainnet, Version::PubKey, &burn_hash.as_bytes());
    pay_to_address_script(&burn_address)
}

/// Build the per-block emission schedule at construction time.
///
/// Each entry in the returned Vec covers one calendar month starting from
/// `emission_start_daa_score`. 4-year halvings are implemented by decaying
/// the per-second reward as `S0 × 2^(−month / KRX_HALVING_PERIOD_MONTHS)`.
/// The schedule terminates when the integer per-second reward truncates to 0
/// (~1760 months / ~146 years at the genesis constant above).
fn build_emission_schedule(bps: u64) -> Vec<u64> {
    let mut schedule = Vec::new();
    let mut month: u32 = 0;
    loop {
        let reward_per_second = (KRX_GENESIS_REWARD_PER_SECOND as f64
            * 2f64.powf(-(month as f64) / KRX_HALVING_PERIOD_MONTHS as f64))
            as u64;
        if reward_per_second == 0 {
            break;
        }
        schedule.push(reward_per_second.div_ceil(bps));
        month += 1;
    }
    schedule
}

#[derive(Clone)]
pub struct CoinbaseManager {
    coinbase_payload_script_public_key_max_len: u8,
    max_coinbase_payload_len: usize,
    /// DAA score at which the deflationary emission phase begins.
    emission_start_daa_score: u64,
    /// Flat per-block subsidy paid before the deflationary phase starts.
    pre_emission_base_subsidy: u64,
    /// Network block rate (blocks per second). Fixed at 10 for Keryx mainnet.
    bps: u64,
    /// Per-block reward indexed by calendar month since `emission_start_daa_score`.
    /// Covers ≈ 34 years of annual-halving emission.
    emission_schedule: Vec<u64>,
    /// Per-block subsidy applied indefinitely once `emission_schedule` is exhausted.
    tail_emission_per_block: u64,
    /// Script public key for the protocol R&D allocation output (5% of every block reward).
    rd_allocation_script_public_key: ScriptPublicKey,
    /// Provably-unspendable burn SPK — receives the 20% escrow cut of standard miners.
    burn_script_public_key: ScriptPublicKey,
}

/// Struct used to streamline payload parsing
struct PayloadParser<'a> {
    remaining: &'a [u8], // The unparsed remainder
}

impl<'a> PayloadParser<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { remaining: data }
    }

    /// Returns a slice with the first `n` bytes of `remaining`, while setting `remaining` to the remaining part
    fn take(&mut self, n: usize) -> &[u8] {
        let (segment, remaining) = self.remaining.split_at(n);
        self.remaining = remaining;
        segment
    }
}

impl CoinbaseManager {
    pub fn new(
        coinbase_payload_script_public_key_max_len: u8,
        max_coinbase_payload_len: usize,
        emission_start_daa_score: u64,
        pre_emission_base_subsidy: u64,
        bps: u64,
    ) -> Self {
        let emission_schedule = build_emission_schedule(bps);
        let tail_emission_per_block = KRX_TAIL_EMISSION_PER_SECOND.div_ceil(bps);

        let rd_address = Address::try_from(RD_ALLOCATION_ADDRESS)
            .expect("hardcoded R&D allocation address must be valid");
        let rd_allocation_script_public_key = pay_to_address_script(&rd_address);

        let burn_hash = TransactionHash::hash(BURN_SEED);
        let burn_address = Address::new(Prefix::Mainnet, Version::PubKey, &burn_hash.as_bytes());
        let burn_script_public_key = pay_to_address_script(&burn_address);

        Self {
            coinbase_payload_script_public_key_max_len,
            max_coinbase_payload_len,
            emission_start_daa_score,
            pre_emission_base_subsidy,
            bps,
            emission_schedule,
            tail_emission_per_block,
            rd_allocation_script_public_key,
            burn_script_public_key,
        }
    }

    pub fn expected_coinbase_transaction<T: AsRef<[u8]>>(
        &self,
        daa_score: u64,
        miner_data: MinerData<T>,
        ghostdag_data: &GhostdagData,
        mergeset_rewards: &BlockHashMap<BlockRewardData>,
        mergeset_non_daa: &BlockHashSet,
        // Tier-reward: blue block hash → multiplier (bps) for its *miner cut* only.
        // Empty / missing entry ⇒ full `TIER_REWARD_BPS_DIVISOR` (no penalty), so this is a
        // no-op before `pom_activation`. Built by the caller from each merged block's proven
        // PoM tier (see `tier_bps_by_block`).
        tier_bps_by_block: &BlockHashMap<u64>,
        // Ratio-reward: blue block hash → multiplier (bps) for its *miner cut* only, from the
        // producer's holder ratio bracket. Empty / missing entry ⇒ full `RATIO_REWARD_BPS_DIVISOR`
        // (no penalty), so this is a no-op before `ratio_reward_activation`. Compounds
        // multiplicatively with `tier_bps_by_block` (see `ratio_bps_by_block`).
        ratio_bps_by_block: &BlockHashMap<u64>,
    ) -> CoinbaseResult<CoinbaseTransactionTemplate> {
        // × 2 for (miner + escrow/burn) per blue, + 1 for possible red reward, + 1 for R&D
        // allocation, + 1 for the accumulated tier-reward burn
        let mut outputs = Vec::with_capacity(ghostdag_data.mergeset_blues.len() * 2 + 3);
        let mut rd_total = 0u64;
        // Sum of the per-blue miner-cut reductions (tier-reward and ratio-reward combined), burned
        // in a single output below. Keeps the total block reward equal to the schedule subsidy —
        // only the miner's share is penalised.
        let mut reward_burn_total = 0u64;

        // Add outputs for each mergeset blue block (∩ DAA window).
        // Transaction fees: 100% burned — no miner benefit, maximum supply destruction.
        // Block subsidy split:
        //   5%  → R&D allocation (accumulated into a single output at the end)
        //   20% → escrow SPK (OPoI miner) or burn SPK (standard miner)
        //   75% → miner's script_public_key
        // Note that combinatorically it is nearly impossible for a blue block to be non-DAA.
        for blue in ghostdag_data.mergeset_blues.iter().filter(|h| !mergeset_non_daa.contains(h)) {
            let reward_data = mergeset_rewards.get(blue).unwrap();
            // All fees burn 100% — deflationary by design.
            if reward_data.total_fees > 0 {
                outputs.push(TransactionOutput::new(reward_data.total_fees, self.burn_script_public_key.clone()));
            }
            if reward_data.subsidy > 0 {
                let rd_cut = reward_data.subsidy * RD_ALLOCATION_BPS / RD_ALLOCATION_BPS_DIVISOR;
                let escrow_cut = reward_data.subsidy * ESCROW_RATE_BPS / ESCROW_RATE_BPS_DIVISOR;
                rd_total += rd_cut;
                let miner_subsidy = reward_data.subsidy - rd_cut - escrow_cut;
                // Scale only the miner cut; the shortfall is burned below. R&D and escrow keep their
                // full-subsidy base, total unchanged. The tier-reward (proven model) and the
                // ratio-reward (holder ratio) compound multiplicatively: a missing entry in either
                // map is the full divisor (no penalty), so each is independently a no-op before its
                // own activation. Integer divisions are applied in a fixed order for determinism.
                let tier_bps = tier_bps_by_block.get(blue).copied().unwrap_or(TIER_REWARD_BPS_DIVISOR);
                let ratio_bps = ratio_bps_by_block.get(blue).copied().unwrap_or(RATIO_REWARD_BPS_DIVISOR);
                let miner_paid = miner_subsidy * tier_bps / TIER_REWARD_BPS_DIVISOR * ratio_bps / RATIO_REWARD_BPS_DIVISOR;
                reward_burn_total += miner_subsidy - miner_paid;
                outputs.push(TransactionOutput::new(miner_paid, reward_data.script_public_key.clone()));
                let escrow_spk = reward_data
                    .escrow_script_public_key
                    .clone()
                    .unwrap_or_else(|| self.burn_script_public_key.clone());
                outputs.push(TransactionOutput::new(escrow_cut, escrow_spk));
            }
        }

        // Collect all rewards from mergeset reds ∩ DAA window and create a
        // single output rewarding the subsidy to the current block (the "merging" block).
        // Fees from red blocks are burned like all other fees.
        let mut red_subsidy = 0u64;
        let mut red_fees = 0u64;

        for red in ghostdag_data.mergeset_reds.iter() {
            let reward_data = mergeset_rewards.get(red).unwrap();
            if mergeset_non_daa.contains(red) {
                // Non-DAA red: subsidy forfeited, fees burned.
                red_fees += reward_data.total_fees;
            } else {
                // DAA red: subsidy goes to merging miner, fees burned.
                red_subsidy += reward_data.subsidy;
                red_fees += reward_data.total_fees;
            }
        }

        // Burn 100% of fees from red blocks.
        if red_fees > 0 {
            outputs.push(TransactionOutput::new(red_fees, self.burn_script_public_key.clone()));
        }

        // Track the index of the red reward output so modify_block_template can rewrite
        // the correct output when the miner address changes, regardless of output ordering.
        let mut red_reward_output_index: Option<usize> = None;

        if red_subsidy > 0 {
            let rd_cut = red_subsidy * RD_ALLOCATION_BPS / RD_ALLOCATION_BPS_DIVISOR;
            rd_total += rd_cut;
            red_reward_output_index = Some(outputs.len());
            outputs.push(TransactionOutput::new(red_subsidy - rd_cut, miner_data.script_public_key.clone()));
        }

        // Single R&D allocation output — 5% of the total block reward, sent to the protocol treasury.
        if rd_total > 0 {
            outputs.push(TransactionOutput::new(rd_total, self.rd_allocation_script_public_key.clone()));
        }

        // Burn the accumulated miner-cut reductions (tier + ratio) in one output. Appended last so
        // it never shifts `red_reward_output_index`. Zero before both rewards activate.
        if reward_burn_total > 0 {
            outputs.push(TransactionOutput::new(reward_burn_total, self.burn_script_public_key.clone()));
        }

        // Build the current block's payload
        let subsidy = self.calc_block_subsidy(daa_score);
        let payload = self.serialize_coinbase_payload(&CoinbaseData { blue_score: ghostdag_data.blue_score, subsidy, miner_data })?;

        Ok(CoinbaseTransactionTemplate {
            tx: Transaction::new(constants::TX_VERSION, vec![], outputs, 0, subnets::SUBNETWORK_ID_COINBASE, 0, payload),
            has_red_reward: red_subsidy > 0 || red_fees > 0,
            red_reward_output_index,
        })
    }

    pub fn serialize_coinbase_payload<T: AsRef<[u8]>>(&self, data: &CoinbaseData<T>) -> CoinbaseResult<Vec<u8>> {
        let script_pub_key_len = data.miner_data.script_public_key.script().len();
        if script_pub_key_len > self.coinbase_payload_script_public_key_max_len as usize {
            return Err(CoinbaseError::PayloadScriptPublicKeyLenAboveMax(
                script_pub_key_len,
                self.coinbase_payload_script_public_key_max_len,
            ));
        }
        let payload: Vec<u8> = data.blue_score.to_le_bytes().iter().copied()                    // Blue score                   (u64)
            .chain(data.subsidy.to_le_bytes().iter().copied())                                  // Subsidy                      (u64)
            .chain(data.miner_data.script_public_key.version().to_le_bytes().iter().copied())   // Script public key version    (u16)
            .chain((script_pub_key_len as u8).to_le_bytes().iter().copied())                    // Script public key length     (u8)
            .chain(data.miner_data.script_public_key.script().iter().copied())                  // Script public key
            .chain(data.miner_data.extra_data.as_ref().iter().copied())                         // Extra data
            .collect();

        Ok(payload)
    }

    pub fn modify_coinbase_payload<T: AsRef<[u8]>>(&self, mut payload: Vec<u8>, miner_data: &MinerData<T>) -> CoinbaseResult<Vec<u8>> {
        let script_pub_key_len = miner_data.script_public_key.script().len();
        if script_pub_key_len > self.coinbase_payload_script_public_key_max_len as usize {
            return Err(CoinbaseError::PayloadScriptPublicKeyLenAboveMax(
                script_pub_key_len,
                self.coinbase_payload_script_public_key_max_len,
            ));
        }

        // Keep only blue score and subsidy. Note that truncate does not modify capacity, so
        // the usual case where the payloads are the same size will not trigger a reallocation
        payload.truncate(LENGTH_OF_BLUE_SCORE + LENGTH_OF_SUBSIDY);
        payload.extend(
            miner_data.script_public_key.version().to_le_bytes().iter().copied() // Script public key version (u16)
                .chain((script_pub_key_len as u8).to_le_bytes().iter().copied()) // Script public key length  (u8)
                .chain(miner_data.script_public_key.script().iter().copied())    // Script public key
                .chain(miner_data.extra_data.as_ref().iter().copied()),          // Extra data
        );

        Ok(payload)
    }

    /// OPoI Phase 2 — verifies the coinbase inference tag against the fixed-point MLP.
    ///
    /// Parses the `extra_data` portion of `payload` looking for the pattern
    /// `/{nonce_hex16}/ai:v1:{tag_hex16}`, then checks that the claimed tag
    /// matches the output of the deterministic fixed-point model for that nonce.
    ///
    /// The fixed-point model (`model_fixed`) uses pure i32/i64 arithmetic and
    /// produces bit-exact results on every CPU — eliminating the floating-point
    /// non-determinism that blocked Phase 1 value verification.
    ///
    /// Active from genesis (chain relaunch — no fork activation needed).
    pub fn validate_opoi_tag(&self, payload: &[u8]) -> CoinbaseResult<()> {
        match keryx_inference::parse_opoi(payload) {
            Some((nonce, claimed_tag)) => {
                if keryx_inference::verify_tag_fixed(nonce, &claimed_tag) {
                    Ok(())
                } else {
                    Err(CoinbaseError::OPoiTagInvalid(nonce, claimed_tag))
                }
            }
            None => Err(CoinbaseError::OPoiTagMissing),
        }
    }

    /// Parses the miner's escrow public key from coinbase `extra_data` and returns a
    /// CSV-timelocked script for the escrow output.
    ///
    /// Looks for `/escrow:` followed by exactly 64 hex chars (32-byte Schnorr pubkey).
    /// Returns `None` if the marker is absent or the key is malformed — treated as a
    /// standard miner whose escrow cut is sent to the burn address instead.
    pub fn parse_escrow_from_extra_data(&self, extra_data: &[u8]) -> Option<ScriptPublicKey> {
        let marker_pos = extra_data.windows(ESCROW_MARKER.len()).position(|w| w == ESCROW_MARKER)?;
        let hex_start = marker_pos + ESCROW_MARKER.len();
        let hex_end = hex_start + ESCROW_PUBKEY_HEX_LEN;
        if hex_end > extra_data.len() {
            return None;
        }
        let hex_str = std::str::from_utf8(&extra_data[hex_start..hex_end]).ok()?;
        let pubkey_bytes: Vec<u8> = (0..32)
            .map(|i| u8::from_str_radix(&hex_str[i * 2..i * 2 + 2], 16))
            .collect::<Result<Vec<_>, _>>()
            .ok()?;
        build_escrow_script(&pubkey_bytes)
    }

    pub fn deserialize_coinbase_payload<'a>(&self, payload: &'a [u8]) -> CoinbaseResult<CoinbaseData<&'a [u8]>> {
        if payload.len() < MIN_PAYLOAD_LENGTH {
            return Err(CoinbaseError::PayloadLenBelowMin(payload.len(), MIN_PAYLOAD_LENGTH));
        }

        if payload.len() > self.max_coinbase_payload_len {
            return Err(CoinbaseError::PayloadLenAboveMax(payload.len(), self.max_coinbase_payload_len));
        }

        let mut parser = PayloadParser::new(payload);

        let blue_score = u64::from_le_bytes(parser.take(LENGTH_OF_BLUE_SCORE).try_into().unwrap());
        let subsidy = u64::from_le_bytes(parser.take(LENGTH_OF_SUBSIDY).try_into().unwrap());
        let script_pub_key_version = u16::from_le_bytes(parser.take(LENGTH_OF_SCRIPT_PUB_KEY_VERSION).try_into().unwrap());
        let script_pub_key_len = u8::from_le_bytes(parser.take(LENGTH_OF_SCRIPT_PUB_KEY_LENGTH).try_into().unwrap());

        if script_pub_key_len > self.coinbase_payload_script_public_key_max_len {
            return Err(CoinbaseError::PayloadScriptPublicKeyLenAboveMax(
                script_pub_key_len as usize,
                self.coinbase_payload_script_public_key_max_len,
            ));
        }

        if parser.remaining.len() < script_pub_key_len as usize {
            return Err(CoinbaseError::PayloadCantContainScriptPublicKey(
                payload.len(),
                MIN_PAYLOAD_LENGTH + script_pub_key_len as usize,
            ));
        }

        let script_public_key =
            ScriptPublicKey::new(script_pub_key_version, ScriptVec::from_slice(parser.take(script_pub_key_len as usize)));
        let extra_data = parser.remaining;

        Ok(CoinbaseData { blue_score, subsidy, miner_data: MinerData { script_public_key, extra_data } })
    }

    /// Returns the per-block subsidy at the given DAA score.
    ///
    /// Three-phase emission layout:
    /// 1. `daa_score < emission_start_daa_score` → flat `pre_emission_base_subsidy`
    /// 2. Month index within `emission_schedule` → annual-halving reward (≈ 34 years)
    /// 3. Past the schedule                      → `tail_emission_per_block` (200+ years)
    pub fn calc_block_subsidy(&self, daa_score: u64) -> u64 {
        if daa_score < self.emission_start_daa_score {
            return self.pre_emission_base_subsidy;
        }
        let month = self.emission_month(daa_score);
        if month < self.emission_schedule.len() {
            self.emission_schedule[month]
        } else {
            self.tail_emission_per_block
        }
    }

    /// Base (un-scaled) miner cut of a single block subsidy at `daa_score`: the block subsidy minus
    /// the R&D and escrow allocations, BEFORE any tier/ratio-reward scaling. This is the exogenous
    /// "mining output" of one selected-chain block, used by the ratio-reward windowed-production
    /// index (see `ratio_bps_by_block`) — deliberately the base cut, not the paid (post-scaling)
    /// amount, so production stays independent of the reward policy it feeds. Mirrors the per-blue
    /// split in `expected_coinbase_transaction` (`miner_subsidy = subsidy − rd_cut − escrow_cut`).
    pub fn base_miner_cut(&self, daa_score: u64) -> u64 {
        let subsidy = self.calc_block_subsidy(daa_score);
        let rd_cut = subsidy * RD_ALLOCATION_BPS / RD_ALLOCATION_BPS_DIVISOR;
        let escrow_cut = subsidy * ESCROW_RATE_BPS / ESCROW_RATE_BPS_DIVISOR;
        subsidy - rd_cut - escrow_cut
    }

    /// Returns the number of elapsed calendar months since the emission phase started.
    ///
    /// Note: called only when `daa_score >= self.emission_start_daa_score`.
    fn emission_month(&self, daa_score: u64) -> usize {
        // DAA score difference divided by BPS gives elapsed seconds.
        let seconds_since_start = (daa_score - self.emission_start_daa_score) / self.bps;
        (seconds_since_start / SECONDS_PER_MONTH) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::MAINNET_PARAMS;
    use keryx_consensus_core::{
        config::params::Params,
        constants::SOMPI_PER_KASPA,
    };

    #[test]
    fn coinbase_outputs_limit_by_era() {
        let k = MAINNET_PARAMS.ghostdag_k() as u64;
        assert_eq!(k, 124, "10 BPS ghostdag K");
        // Pre-H3: legacy upstream cap. Post-H3: builder structural max (3 per blue + 4 aggregates).
        assert_eq!(coinbase_outputs_limit(k, false), 126);
        assert_eq!(coinbase_outputs_limit(k, true), 379);
        // The H3 bound must cover the worst case the builder can emit: 3 outputs for each of the
        // K+1 mergeset blues plus the 4 aggregate outputs.
        assert_eq!(coinbase_outputs_limit(k, true), 3 * (k + 1) + 4);
    }

    fn create_manager(params: &Params) -> CoinbaseManager {
        CoinbaseManager::new(
            params.coinbase_payload_script_public_key_max_len,
            params.max_coinbase_payload_len,
            params.deflationary_phase_daa_score,
            params.pre_deflationary_phase_base_subsidy,
            params.bps_history().after(),
        )
    }

    #[test]
    fn base_miner_cut_is_subsidy_minus_rd_and_escrow() {
        let cbm = create_manager(&MAINNET_PARAMS);
        // Across the emission phases (flat pre-emission, scheduled, tail), the base miner cut is the
        // block subsidy minus the R&D and escrow allocations, BEFORE any tier/ratio scaling.
        for daa in [0u64, 1, 1_000_000, 100_000_000, 10_000_000_000] {
            let s = cbm.calc_block_subsidy(daa);
            let expected = s - s * RD_ALLOCATION_BPS / RD_ALLOCATION_BPS_DIVISOR - s * ESCROW_RATE_BPS / ESCROW_RATE_BPS_DIVISOR;
            assert_eq!(cbm.base_miner_cut(daa), expected, "daa={daa}");
            assert!(cbm.base_miner_cut(daa) <= s, "base cut never exceeds the subsidy");
        }
    }

    // ── tier-reward ───────────────────────────────────────────────────────────

    use keryx_consensus_core::blockhash::BlockHashes;
    use keryx_consensus_core::config::params::{RATIO_REWARD_BPS, RATIO_REWARD_BPS_DIVISOR, TIER_REWARD_BPS};
    use keryx_consensus_core::{HashKTypeMap, HashMapCustomHasher};
    use keryx_hashes::Hash;

    /// Builds a `GhostdagData` whose mergeset blues are exactly `blues` (no reds), enough to
    /// drive `expected_coinbase_transaction`.
    fn ghostdag_with_blues(blues: Vec<Hash>) -> GhostdagData {
        let selected_parent = blues[0];
        GhostdagData::new(
            0,
            Default::default(),
            selected_parent,
            BlockHashes::new(blues),
            BlockHashes::new(vec![]),
            HashKTypeMap::new(BlockHashMap::new()),
        )
    }

    /// Two blue blocks with equal subsidy but different proven tiers: the floor tier (0, −18 %)
    /// and the top tier (3, 0 %). Only the *miner cut* is scaled, the shortfall is burned, and
    /// the total block reward, the escrow cuts and the R&D cut are untouched.
    #[test]
    fn tier_reward_scales_only_miner_cut() {
        let cbm = create_manager(&MAINNET_PARAMS);
        let subsidy = 1_000_000_000u64;

        let (h_a, h_b): (Hash, Hash) = (1.into(), 2.into());
        let spk_a = ScriptPublicKey::from_vec(0, vec![0xaa]);
        let spk_b = ScriptPublicKey::from_vec(0, vec![0xbb]);
        let escrow_a = ScriptPublicKey::from_vec(0, vec![0xa1]);
        let escrow_b = ScriptPublicKey::from_vec(0, vec![0xb1]);

        let mut mergeset_rewards = BlockHashMap::new();
        mergeset_rewards.insert(h_a, BlockRewardData::new_with_escrow(subsidy, 0, spk_a.clone(), Some(escrow_a.clone())));
        mergeset_rewards.insert(h_b, BlockRewardData::new_with_escrow(subsidy, 0, spk_b.clone(), Some(escrow_b.clone())));

        let ghostdag = ghostdag_with_blues(vec![h_a, h_b]);
        let non_daa = BlockHashSet::new();
        let miner_data = MinerData::new(ScriptPublicKey::from_vec(0, vec![0xcc]), vec![]);

        // A = floor tier (−18 %), B = top tier (full).
        let mut tier_bps = BlockHashMap::new();
        tier_bps.insert(h_a, TIER_REWARD_BPS[0]);
        tier_bps.insert(h_b, TIER_REWARD_BPS[3]);
        // Ratio-reward inactive here → empty map (no compounding), isolates the tier behaviour.
        let ratio_bps = BlockHashMap::new();

        let tx = cbm
            .expected_coinbase_transaction(0, miner_data, &ghostdag, &mergeset_rewards, &non_daa, &tier_bps, &ratio_bps)
            .unwrap()
            .tx;

        let rd = subsidy * RD_ALLOCATION_BPS / RD_ALLOCATION_BPS_DIVISOR;
        let escrow = subsidy * ESCROW_RATE_BPS / ESCROW_RATE_BPS_DIVISOR;
        let full_miner = subsidy - rd - escrow;
        let miner_a = full_miner * TIER_REWARD_BPS[0] / TIER_REWARD_BPS_DIVISOR; // −18 %
        let miner_b = full_miner; // top tier, full

        let value_of = |spk: &ScriptPublicKey| tx.outputs.iter().find(|o| &o.script_public_key == spk).map(|o| o.value);

        // Miner cut scaled by tier.
        assert_eq!(value_of(&spk_a), Some(miner_a), "floor-tier miner cut must be −18 %");
        assert_eq!(value_of(&spk_b), Some(miner_b), "top-tier miner cut must be full");
        assert!(miner_a < miner_b, "serving a heavier model must pay the miner strictly more");

        // Escrow cuts untouched by the tier penalty.
        assert_eq!(value_of(&escrow_a), Some(escrow), "escrow cut must keep its full base");
        assert_eq!(value_of(&escrow_b), Some(escrow), "escrow cut must keep its full base");

        // Total block reward unchanged — the delta is burned, not removed.
        let total: u64 = tx.outputs.iter().map(|o| o.value).sum();
        assert_eq!(total, 2 * subsidy, "tier penalty must not change the total block reward");
    }

    /// Empty tier map (the pre-`pom_activation` state) is a no-op: the miner cut is paid in full,
    /// the total is unchanged, and no extra burn output is appended.
    #[test]
    fn tier_reward_empty_map_is_noop() {
        let cbm = create_manager(&MAINNET_PARAMS);
        let subsidy = 1_000_000_000u64;

        let h_a: Hash = 1.into();
        let spk_a = ScriptPublicKey::from_vec(0, vec![0xaa]);
        let escrow_a = ScriptPublicKey::from_vec(0, vec![0xa1]);

        let mut mergeset_rewards = BlockHashMap::new();
        mergeset_rewards.insert(h_a, BlockRewardData::new_with_escrow(subsidy, 0, spk_a.clone(), Some(escrow_a.clone())));

        let ghostdag = ghostdag_with_blues(vec![h_a]);
        let non_daa = BlockHashSet::new();
        let miner_data = MinerData::new(ScriptPublicKey::from_vec(0, vec![0xcc]), vec![]);

        // No gate → empty maps.
        let tier_bps = BlockHashMap::new();
        let ratio_bps = BlockHashMap::new();

        let tx = cbm
            .expected_coinbase_transaction(0, miner_data, &ghostdag, &mergeset_rewards, &non_daa, &tier_bps, &ratio_bps)
            .unwrap()
            .tx;

        let rd = subsidy * RD_ALLOCATION_BPS / RD_ALLOCATION_BPS_DIVISOR;
        let escrow = subsidy * ESCROW_RATE_BPS / ESCROW_RATE_BPS_DIVISOR;
        let full_miner = subsidy - rd - escrow;

        let value_of = |spk: &ScriptPublicKey| tx.outputs.iter().find(|o| &o.script_public_key == spk).map(|o| o.value);
        assert_eq!(value_of(&spk_a), Some(full_miner), "no gate ⇒ full miner cut");
        // 1 miner + 1 escrow + 1 R&D, and NO tier-burn output appended.
        assert_eq!(tx.outputs.len(), 3, "no penalty ⇒ no extra burn output");
        let total: u64 = tx.outputs.iter().map(|o| o.value).sum();
        assert_eq!(total, subsidy, "total reward equals the single block subsidy");
    }

    /// Tier-reward and ratio-reward compound multiplicatively on the miner cut: a block at the
    /// floor tier (−18 %) AND the floor ratio bracket (40 %) keeps 0.82 × 0.40 = 0.328 of its cut,
    /// the combined shortfall is burned, and the total block reward is unchanged.
    #[test]
    fn tier_and_ratio_reward_compound() {
        let cbm = create_manager(&MAINNET_PARAMS);
        let subsidy = 1_000_000_000u64;

        let h_a: Hash = 1.into();
        let spk_a = ScriptPublicKey::from_vec(0, vec![0xaa]);
        let escrow_a = ScriptPublicKey::from_vec(0, vec![0xa1]);

        let mut mergeset_rewards = BlockHashMap::new();
        mergeset_rewards.insert(h_a, BlockRewardData::new_with_escrow(subsidy, 0, spk_a.clone(), Some(escrow_a.clone())));

        let ghostdag = ghostdag_with_blues(vec![h_a]);
        let non_daa = BlockHashSet::new();
        let miner_data = MinerData::new(ScriptPublicKey::from_vec(0, vec![0xcc]), vec![]);

        let mut tier_bps = BlockHashMap::new();
        tier_bps.insert(h_a, TIER_REWARD_BPS[0]); // floor tier −18 %
        let mut ratio_bps = BlockHashMap::new();
        ratio_bps.insert(h_a, RATIO_REWARD_BPS[0]); // floor bracket 40 %

        let tx = cbm
            .expected_coinbase_transaction(0, miner_data, &ghostdag, &mergeset_rewards, &non_daa, &tier_bps, &ratio_bps)
            .unwrap()
            .tx;

        let rd = subsidy * RD_ALLOCATION_BPS / RD_ALLOCATION_BPS_DIVISOR;
        let escrow = subsidy * ESCROW_RATE_BPS / ESCROW_RATE_BPS_DIVISOR;
        let full_miner = subsidy - rd - escrow;
        let tier_only = full_miner * TIER_REWARD_BPS[0] / TIER_REWARD_BPS_DIVISOR;
        // Same fixed division order as the coinbase manager.
        let expected_miner = tier_only * RATIO_REWARD_BPS[0] / RATIO_REWARD_BPS_DIVISOR;

        let value_of = |spk: &ScriptPublicKey| tx.outputs.iter().find(|o| &o.script_public_key == spk).map(|o| o.value);
        assert_eq!(value_of(&spk_a), Some(expected_miner), "miner cut must be tier × ratio compounded");
        assert!(expected_miner < tier_only, "ratio floor must cut further than tier alone");

        // Escrow untouched by either penalty.
        assert_eq!(value_of(&escrow_a), Some(escrow), "escrow cut must keep its full base");

        // Total unchanged — combined shortfall burned.
        let total: u64 = tx.outputs.iter().map(|o| o.value).sum();
        assert_eq!(total, subsidy, "compound penalty must not change the total block reward");
    }

    #[test]
    fn emission_schedule_sanity_test() {
        let cbm = create_manager(&MAINNET_PARAMS);
        let bps = MAINNET_PARAMS.bps_history().after();

        let expected_month0 = KRX_GENESIS_REWARD_PER_SECOND.div_ceil(bps);
        assert_eq!(cbm.emission_schedule[0], expected_month0, "month-0 reward must match genesis constant");

        // 4-year halving: month 48 must be ~half of month 0 (off-by-one from div_ceil is ok)
        let month48 = cbm.emission_schedule[KRX_HALVING_PERIOD_MONTHS as usize];
        let expected_half = expected_month0 / 2;
        assert!(
            month48 == expected_half || month48 == expected_half + 1,
            "month-48 reward ({}) should be ~half of month-0 ({})",
            month48,
            expected_month0
        );

        // No zero-reward entries in the main schedule
        assert!(cbm.emission_schedule.iter().all(|&r| r > 0), "emission schedule must not contain zero entries");
        assert!(!cbm.emission_schedule.is_empty(), "emission schedule must be non-empty");

        println!(
            "Emission schedule: {} months ≈ {} years",
            cbm.emission_schedule.len(),
            cbm.emission_schedule.len() / 12
        );
        println!("Month-0 per-block reward: {} sompi = {} KRX", expected_month0, expected_month0 / SOMPI_PER_KASPA);
    }

    #[test]
    fn total_emission_approximately_10b_krx_test() {
        let cbm = create_manager(&MAINNET_PARAMS);
        let bps = MAINNET_PARAMS.bps_history().after();

        // Main-phase emission: each schedule entry covers SECONDS_PER_MONTH × bps blocks
        let main_emission: u64 = cbm.emission_schedule.iter().map(|&r| r * SECONDS_PER_MONTH * bps).sum();
        let main_emission_krx = main_emission / SOMPI_PER_KASPA;

        println!("Main-phase emission: {} sompi = {} KRX", main_emission, main_emission_krx);

        // Target band: 9 – 11 B KRX
        assert!(main_emission_krx >= 9_000_000_000, "emission too low: {} KRX", main_emission_krx);
        assert!(main_emission_krx <= 11_000_000_000, "emission too high: {} KRX", main_emission_krx);
    }

    #[test]
    fn tail_emission_test() {
        let cbm = create_manager(&MAINNET_PARAMS);
        let bps = MAINNET_PARAMS.bps_history().after();

        // Construct a DAA score well past the end of the main schedule
        let past_schedule_daa = MAINNET_PARAMS.deflationary_phase_daa_score
            + (cbm.emission_schedule.len() as u64 + 100) * SECONDS_PER_MONTH * bps;

        let subsidy = cbm.calc_block_subsidy(past_schedule_daa);
        let expected_tail = KRX_TAIL_EMISSION_PER_SECOND.div_ceil(bps);
        assert_eq!(subsidy, expected_tail, "tail emission must be active past schedule end");
        assert_eq!(subsidy, cbm.tail_emission_per_block);
    }

    #[test]
    fn halving_test() {
        let cbm = create_manager(&MAINNET_PARAMS);
        let bps = MAINNET_PARAMS.bps_history().after();
        let blocks_per_halving = KRX_HALVING_PERIOD_MONTHS as u64 * SECONDS_PER_MONTH * bps;

        let base = cbm.calc_block_subsidy(MAINNET_PARAMS.deflationary_phase_daa_score);
        let after_one = cbm.calc_block_subsidy(MAINNET_PARAMS.deflationary_phase_daa_score + blocks_per_halving);
        let after_two = cbm.calc_block_subsidy(MAINNET_PARAMS.deflationary_phase_daa_score + 2 * blocks_per_halving);

        // Each halving must cut the reward by ~half (rounding tolerates ±1)
        assert!(
            after_one == base / 2 || after_one == base / 2 + 1,
            "after 1 halving: {} (expected ≈ {})",
            after_one,
            base / 2
        );
        assert!(
            after_two == base / 4 || after_two == base / 4 + 1,
            "after 2 halvings: {} (expected ≈ {})",
            after_two,
            base / 4
        );
    }

    #[test]
    fn pre_emission_phase_test() {
        let cbm = create_manager(&MAINNET_PARAMS);

        // Keryx has no pre-emission bootstrapping phase: emission starts at genesis (daa_score = 0).
        assert_eq!(MAINNET_PARAMS.deflationary_phase_daa_score, 0,
            "Keryx emission must start at genesis — no pre-emission phase");

        // Genesis block (daa_score = 0) must immediately yield the first emission schedule entry.
        assert_eq!(cbm.calc_block_subsidy(0), cbm.emission_schedule[0],
            "genesis block reward must match emission schedule month 0");

        // Block 1 must also yield month-0 reward (same calendar month as genesis).
        assert_eq!(cbm.calc_block_subsidy(1), cbm.emission_schedule[0],
            "block 1 reward must still be in emission schedule month 0");
    }

    #[test]
    fn payload_serialization_test() {
        let cbm = create_manager(&MAINNET_PARAMS);

        let script_data = [33u8, 255];
        let extra_data = [2u8, 3];
        let data = CoinbaseData {
            blue_score: 56,
            subsidy: 1_590_000_000,
            miner_data: MinerData {
                script_public_key: ScriptPublicKey::new(0, ScriptVec::from_slice(&script_data)),
                extra_data: &extra_data as &[u8],
            },
        };

        let payload = cbm.serialize_coinbase_payload(&data).unwrap();
        let deserialized_data = cbm.deserialize_coinbase_payload(&payload).unwrap();

        assert_eq!(data, deserialized_data);
    }

    #[test]
    fn modify_payload_test() {
        let cbm = create_manager(&MAINNET_PARAMS);

        let script_data = [33u8, 255];
        let extra_data = [2u8, 3, 23, 98];
        let data = CoinbaseData {
            blue_score: 56345,
            subsidy: 1_590_000_000,
            miner_data: MinerData {
                script_public_key: ScriptPublicKey::new(0, ScriptVec::from_slice(&script_data)),
                extra_data: &extra_data,
            },
        };

        let data2 = CoinbaseData {
            blue_score: data.blue_score,
            subsidy: data.subsidy,
            miner_data: MinerData {
                script_public_key: ScriptPublicKey::new(0, ScriptVec::from_slice(&[33u8, 255, 33])),
                extra_data: &[2u8, 3, 23, 98, 34, 34] as &[u8],
            },
        };

        let mut payload = cbm.serialize_coinbase_payload(&data).unwrap();
        payload = cbm.modify_coinbase_payload(payload, &data2.miner_data).unwrap();
        let deserialized_data = cbm.deserialize_coinbase_payload(&payload).unwrap();

        assert_eq!(data2, deserialized_data);
    }
}
