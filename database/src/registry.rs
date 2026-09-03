use enum_primitive_derive::Primitive;

/// We use `u8::MAX` which is never a valid block level. Also note that through
/// the [`DatabaseStorePrefixes`] enum we make sure it is not used as a prefix as well
pub const SEPARATOR: u8 = u8::MAX;

#[derive(Primitive, Debug, Clone, Copy)]
#[repr(u8)]
pub enum DatabaseStorePrefixes {
    // ---- Consensus ----
    AcceptanceData = 1,
    BlockTransactions = 2,
    NonDaaMergeset = 3,
    BlockDepth = 4,
    Ghostdag = 5,
    GhostdagCompact = 6,
    HeadersSelectedTip = 7,
    // Legacy headers store prefix. CompressedHeaders is used instead
    Headers = 8,
    HeadersCompact = 9,
    PastPruningPoints = 10,
    PruningUtxoset = 11,
    PruningUtxosetPosition = 12,
    PruningPoint = 13,
    RetentionCheckpoint = 14,
    Reachability = 15,
    ReachabilityReindexRoot = 16,
    ReachabilityRelations = 17,
    RelationsParents = 18,
    RelationsChildren = 19,
    ChainHashByIndex = 20,
    ChainIndexByHash = 21,
    ChainHighestIndex = 22,
    Statuses = 23,
    Tips = 24,
    UtxoDiffs = 25,
    UtxoMultisets = 26,
    VirtualUtxoset = 27,
    VirtualState = 28,
    PruningSamples = 29,

    // ---- Decomposed reachability stores ----
    ReachabilityTreeChildren = 30,
    ReachabilityFutureCoveringSet = 31,

    // Stores headers with run-length encoded parents
    CompressedHeaders = 32,

    // Stores a succinct pruning proof descriptor
    PruningProofDescriptor = 33,

    // ---- OPoI Collateral ----
    MinerCollateral = 34,

    // ---- OPoI Slash (Phase 3 A4) ----
    /// Confirmed AiResponse txs: response_hash → AiResponseRecord
    AiResponse = 35,
    /// Slashed escrow outpoints: outpoint_bytes → slash_blue_score
    AiSlashed = 36,

    // ---- PoM tier-reward ----
    /// Proven PoM tier per block: block_hash → tier (u8)
    PomTier = 37,

    // ---- Ratio-reward (holder-weighted miner cut) ----
    // 38 reserved (was RatioBps per-block store; removed — the bracket is now computed inline at the
    // rewarding block's view, see ratio_bps_by_block, so nothing is persisted per block).
    /// Ratio-reward balance index: payout SPK → Σ unspent amount (consensus, lockstep with the UTXO set)
    AddressBalance = 39,

    // ---- Ghostdag Proof
    TempGhostdag = 40,
    TempGhostdagCompact = 41,
    TempRelationsParents = 42,
    TempRelationsChildren = 43,

    // ---- Ratio-reward (cont.) ----
    // 44 retired: the legacy `WindowedProduction` running-sum index, superseded by the path-independent
    // prefix-sum index below (`WindowedProductionPrefix`). Do not reuse this discriminant.

    /// Fast-sync catch-up: virtual selected-chain index at which the windowed-production index was last
    /// reset by a pruning-point UTXO import (see `import_pruning_point_utxo_set`). Single value, no key.
    ProductionIndexSeededAt = 45,

    /// Ratio-reward production PREFIX-SUM index (gold-standard, replaces the path-dependent
    /// `WindowedProduction` running sum): key `SPK || be(chain_index)` → cumulative production for that
    /// SPK over selected-chain [genesis, chain_index]. The windowed value is the pure-function
    /// difference `cum(b) − cum(b−W)`, so every node on the same chain computes the identical number
    /// regardless of its update history. See `windowed_production_prefix`.
    WindowedProductionPrefix = 46,

    /// Floor baseline for `WindowedProductionPrefix`: key `SPK` → cumulative production up to the
    /// current pruning floor, for SPKs whose per-block entries below the floor have been collapsed
    /// (so `cum(b−W)` stays exact after pruning). See `windowed_production_prefix::advance_floor`.
    WindowedProductionFloor = 47,

    /// Coin-age (holder-reward v3) bucket aggregates: key `SPK` → `{b_mat, b_imm, a_imm}` (see
    /// `consensus::model::stores::age_buckets`). Maintained in lockstep with the virtual UTXO set,
    /// rebuilt from it at startup; read by the ratio numerator at/after `coin_age_activation`.
    AgeBuckets = 48,

    /// Coin-age maturation queue: key `be(maturity_daa) || outpoint` → `(SPK, amount, anchor)`
    /// for IMMATURE coins only (see `maturation_queue`). Swept at each virtual commit to promote
    /// coins whose `effective_daa + W` fell at/below the new virtual score.
    MaturationQueue = 49,

    /// Coin-age promotion watermark (single key): the highest virtual daa score up to which the
    /// maturation queue has been swept. A decrease (deep reorg) triggers a full coin-age rebuild.
    CoinAgeWatermark = 51,

    // ---- Retention Period Root ----
    RetentionPeriodRoot = 50,

    // ---- Pruning metadata ----
    PruningUtxosetSyncFlag = 60,
    BodyMissingAnticone = 61,

    // ---- Metadata ----
    MultiConsensusMetadata = 124,
    ConsensusEntries = 125,

    // ---- Components ----
    Addresses = 128,
    BannedAddresses = 129,

    // ---- Indexes ----
    UtxoIndex = 192,
    UtxoIndexTips = 193,
    CirculatingSupply = 194,

    // ---- PoM possession proof ----
    /// Full PoM possession proof per block: block_hash → bincode(PomProof) — bincode, like every
    /// other `CachedDbAccess` store; borsh is the WIRE encoding only (`PomProof::to_wire_bytes`).
    /// Persisted so a block can be re-served (relay/IBD) with its proof; otherwise `get_block`
    /// returns `pom_proof: None` and peers reject the served block (`PoM possession proof missing`).
    PomProof = 195,
    /// Service-bond burned escrow outpoints (finality-deep misses): outpoint → miss daa.
    ServiceBurn = 196,
    /// Service-bond strike log (finality-deep events, append-only): `daa (BE) || miner identity`
    /// → (consecutive misses, last strike daa). The fold's strike baseline is the last record
    /// per miner; counts only reset on a served response or an executed suspension, never by
    /// time. Suspensions are the `{0, daa > 0}` rows. (197 was the retired suspend store.)
    ServiceStrike = 198,
    /// Service-bond first sightings (finality-deep, append-once): miner identity → daa of its
    /// first certified block. The standing/probation clock.
    ServiceFirstSeen = 199,
    /// Inference-reward wins (finality-deep, append-once): request hash → (winner identity,
    /// amount, event daa). Mint dedup and commitment rebuild.
    ServiceReward = 200,
    /// Canonical service-ledger snapshot at each pruning sample: block hash → encoded state.
    ServiceLedgerSnapshot = 201,
    /// Canonical production-index snapshot at each pruning sample: block hash → encoded state.
    ProductionIndexSnapshot = 202,
    /// Window daa table of the production-index snapshot imported at the pruning point:
    /// be(chain index) → daa score.
    ProductionWindowDaa = 203,
    /// Bounds (bottom, sample chain indices) of that imported window.
    ProductionImportedWindow = 204,
    /// Per-SPK prefix sum of the miner cut actually PAID by coinbases (post tier/ratio scaling),
    /// and its pruning floor. Display only: read by `getHolderReward` to separate income from the
    /// burned shortfall. Deliberately a separate keyspace from `WindowedProductionPrefix`, which
    /// is hashed into the service commitment — nothing here enters consensus.
    MinerPaidPrefix = 205,
    MinerPaidFloor = 206,
    /// Per-SPK prefix sum of the escrow cut that ACCRUED to the producer (zero for a standard
    /// miner, whose escrow is burned at emission), and its pruning floor. Display only.
    MinerEscrowPrefix = 207,
    MinerEscrowFloor = 208,
    /// Per-SPK prefix sum of inference-reward mints routed to this SPK by coinbases, and its
    /// pruning floor. Mining income of a different kind, tracked apart from the miner cut because
    /// a mint is indistinguishable from one by script alone. Display only.
    MinerInferencePrefix = 209,
    MinerInferenceFloor = 210,
    /// Single u64: the lowest selected-chain index from which the three payout indexes above are
    /// complete. They are maintained FORWARD from the boot that created them, because a from-chain
    /// backfill has to read the coinbase BODY of every chain block and of each of its mergeset
    /// blues — hours of random reads on a spinning disk. A query whose window bottom predates this
    /// index would silently under-report income and over-report the burn, so reads clamp to it and
    /// report the span actually covered, letting the caller label a partial window honestly.
    MinerPayoutIndexStart = 211,
    /// Per-SPK prefix sums of the BASE miner cut split by the proven model tier of the block that
    /// earned it — one (entries, floor) pair per tier bucket — plus their pruning floors. Display
    /// only, same keyspace shape and same forward-only start marker as the payout indexes above.
    ///
    /// Five buckets rather than one weighted average because a miner is not on one tier: rigs run
    /// different models, so the window holds a MIX, and any single tier reported for it would be
    /// false. From the buckets the weighted tier is a division; from a weighted tier the mix is
    /// unrecoverable.
    MinerTier0Prefix = 212,
    MinerTier0Floor = 213,
    MinerTier1Prefix = 214,
    MinerTier1Floor = 215,
    MinerTier2Prefix = 216,
    MinerTier2Floor = 217,
    MinerTier3Prefix = 218,
    MinerTier3Floor = 219,
    MinerTier4Prefix = 220,
    MinerTier4Floor = 221,

    // ---- Separator ----
    /// Reserved as a separator
    Separator = SEPARATOR,
}

impl From<DatabaseStorePrefixes> for Vec<u8> {
    fn from(value: DatabaseStorePrefixes) -> Self {
        [value as u8].to_vec()
    }
}

impl From<DatabaseStorePrefixes> for u8 {
    fn from(value: DatabaseStorePrefixes) -> Self {
        value as u8
    }
}

impl AsRef<[u8]> for DatabaseStorePrefixes {
    fn as_ref(&self) -> &[u8] {
        // SAFETY: enum has repr(u8)
        std::slice::from_ref(unsafe { &*(self as *const Self as *const u8) })
    }
}

impl IntoIterator for DatabaseStorePrefixes {
    type Item = u8;
    type IntoIter = <[u8; 1] as IntoIterator>::IntoIter;
    fn into_iter(self) -> Self::IntoIter {
        [self as u8].into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_as_ref() {
        let prefix = DatabaseStorePrefixes::AcceptanceData;
        assert_eq!(&[prefix as u8], prefix.as_ref());
        assert_eq!(
            size_of::<u8>(),
            size_of::<DatabaseStorePrefixes>(),
            "DatabaseStorePrefixes is expected to have the same memory layout of u8"
        );
    }
}
