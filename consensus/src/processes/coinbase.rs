use keryx_addresses::Address;
use keryx_consensus_core::{
    BlockHashMap, BlockHashSet,
    coinbase::*,
    errors::coinbase::{CoinbaseError, CoinbaseResult},
    subnets,
    tx::{ScriptPublicKey, ScriptVec, Transaction, TransactionOutput},
};
use keryx_txscript::standard::pay_to_address_script;
use std::convert::TryInto;

use crate::{constants, model::stores::ghostdag::GhostdagData};

// ── R&D allocation ────────────────────────────────────────────────────────────

/// Protocol-level R&D allocation: 2% of every block reward (200 basis points).
/// Deducted from each miner output before it is committed to the coinbase transaction.
const RD_ALLOCATION_BPS: u64 = 200;
const RD_ALLOCATION_BPS_DIVISOR: u64 = 10_000;

/// Mainnet address that receives the R&D allocation.
pub const RD_ALLOCATION_ADDRESS: &str =
    "keryx:qp8zp9wnpqhgygpsv25px8whw0ee7md72s0tgy78x5wt7ryk6w525aqm045zv";

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
    /// Script public key for the protocol R&D allocation output (2% of every block reward).
    rd_allocation_script_public_key: ScriptPublicKey,
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

        Self {
            coinbase_payload_script_public_key_max_len,
            max_coinbase_payload_len,
            emission_start_daa_score,
            pre_emission_base_subsidy,
            bps,
            emission_schedule,
            tail_emission_per_block,
            rd_allocation_script_public_key,
        }
    }

    pub fn expected_coinbase_transaction<T: AsRef<[u8]>>(
        &self,
        daa_score: u64,
        miner_data: MinerData<T>,
        ghostdag_data: &GhostdagData,
        mergeset_rewards: &BlockHashMap<BlockRewardData>,
        mergeset_non_daa: &BlockHashSet,
    ) -> CoinbaseResult<CoinbaseTransactionTemplate> {
        // + 1 for possible red reward, + 1 for R&D allocation output
        let mut outputs = Vec::with_capacity(ghostdag_data.mergeset_blues.len() + 2);
        let mut rd_total = 0u64;

        // Add an output for each mergeset blue block (∩ DAA window), paying to the script reported by the block.
        // 2% of each reward is withheld and accumulated into the R&D allocation output.
        // Note that combinatorically it is nearly impossible for a blue block to be non-DAA.
        for blue in ghostdag_data.mergeset_blues.iter().filter(|h| !mergeset_non_daa.contains(h)) {
            let reward_data = mergeset_rewards.get(blue).unwrap();
            let total = reward_data.subsidy + reward_data.total_fees;
            if total > 0 {
                let rd_cut = total * RD_ALLOCATION_BPS / RD_ALLOCATION_BPS_DIVISOR;
                rd_total += rd_cut;
                outputs.push(TransactionOutput::new(total - rd_cut, reward_data.script_public_key.clone()));
            }
        }

        // Collect all rewards from mergeset reds ∩ DAA window and create a
        // single output rewarding all to the current block (the "merging" block).
        // The same 2% R&D cut applies to the merged red reward.
        let mut red_reward = 0u64;

        for red in ghostdag_data.mergeset_reds.iter() {
            let reward_data = mergeset_rewards.get(red).unwrap();
            if mergeset_non_daa.contains(red) {
                red_reward += reward_data.total_fees;
            } else {
                red_reward += reward_data.subsidy + reward_data.total_fees;
            }
        }

        if red_reward > 0 {
            let rd_cut = red_reward * RD_ALLOCATION_BPS / RD_ALLOCATION_BPS_DIVISOR;
            rd_total += rd_cut;
            outputs.push(TransactionOutput::new(red_reward - rd_cut, miner_data.script_public_key.clone()));
        }

        // Single R&D allocation output — 2% of the total block reward, sent to the protocol treasury.
        if rd_total > 0 {
            outputs.push(TransactionOutput::new(rd_total, self.rd_allocation_script_public_key.clone()));
        }

        // Build the current block's payload
        let subsidy = self.calc_block_subsidy(daa_score);
        let payload = self.serialize_coinbase_payload(&CoinbaseData { blue_score: ghostdag_data.blue_score, subsidy, miner_data })?;

        Ok(CoinbaseTransactionTemplate {
            tx: Transaction::new(constants::TX_VERSION, vec![], outputs, 0, subnets::SUBNETWORK_ID_COINBASE, 0, payload),
            has_red_reward: red_reward > 0,
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

    /// OPoI Phase 1 — verifies that the coinbase payload carries an inference tag.
    ///
    /// Parses the `extra_data` portion of `payload` looking for the pattern
    /// `/{nonce_hex16}/ai:v1:{tag_hex16}`.
    ///
    /// **Phase 1 policy (Optimistic)**: only the *presence* and *format* of the
    /// tag are checked, not the MLP output value.  Checking the exact value here
    /// causes false rejections during IBD because the MLP result depends on the
    /// floating-point implementation (candle-core vs. bare Rust vs. GPU kernels)
    /// and may differ across hardware even for the same nonce.  Value verification
    /// is deferred to Phase 2 fraud-proofs, which are the authoritative enforcement
    /// mechanism under the Optimistic Proof of Inference design.
    ///
    /// Validates the OPoI tag format in a coinbase payload.
    /// Consensus enforcement is disabled (Phase 1 optimistic); this function is
    /// kept for Phase 2 fraud-proof verification.
    #[allow(dead_code)]
    pub fn validate_opoi_tag(&self, payload: &[u8]) -> CoinbaseResult<()> {
        match keryx_inference::parse_opoi(payload) {
            Some(_) => {
                // Tag is present and well-formed — value check deferred to Phase 2 fraud-proofs.
                Ok(())
            }
            None => {
                // No OPoI marker — every miner must commit to inference.
                Err(CoinbaseError::OPoiTagMissing)
            }
        }
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
