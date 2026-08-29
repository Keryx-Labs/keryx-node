use super::VirtualStateProcessor;
use crate::processes::service_commit;
use crate::model::stores::{
    acceptance_data::AcceptanceDataStoreReader, block_transactions::BlockTransactionsStoreReader, daa::DaaStoreReader,
    ghostdag::GhostdagStoreReader, headers::HeaderStoreReader, pruning::PruningStoreReader,
    selected_chain::SelectedChainStoreReader,
};
use keryx_consensus_core::collateral::{
    eligible_pairs, escrow_miner_key, miner_key, verify_responder_signature, EscrowClaim, FoldOutcome, RewardEntry, ServiceLedger,
    ServiceLedgerSnapshot,
    ServiceMiss, ServicePenalty, ServiceReward, ServiceStrikesSnapshot, StrikeEntry,
    SERVICE_ELIGIBILITY_WINDOW_DAA, SERVICE_ELIGIBILITY_WINDOW_DAA_V2, SERVICE_SUSPENSION_DAA,
};
use keryx_consensus_core::config::params::POM_TIERS_H6;
use keryx_consensus_core::tx::{ScriptPublicKey, TransactionOutpoint};
use keryx_consensus_core::ChainPath;
use keryx_consensus_core::blockhash::BlockHashExtensions;
use keryx_core::{info, warn};
use keryx_hashes::Hash;
use keryx_inference::{AiRequestPayload, AiResponsePayload};
use keryx_txscript::script_class::ScriptClass;


/// The escrow pubkey locked by a CSV escrow script, if the script is one.
fn csv_escrow_pubkey(script: &[u8]) -> Option<[u8; 32]> {
    if !ScriptClass::is_csv_pay_to_pubkey(script) {
        return None;
    }
    let seq_len = script[0] as usize;
    script[seq_len + 3..seq_len + 35].try_into().ok()
}

/// The authenticated responder key of a V2 response: its escrow pubkey, iff the schnorr
/// signature over the v1 payload bytes verifies. `None` for v1 or a bad signature.
fn verified_responder(resp: &AiResponsePayload) -> Option<Hash> {
    let r = resp.responder.as_ref()?;
    verify_responder_signature(&r.escrow_pubkey, &r.signature, &resp.signed_bytes()).then(|| escrow_miner_key(&r.escrow_pubkey))
}

/// Retained per-chain-block ledger snapshots; reorgs deeper than this fall back to a horizon refold.
const SERVICE_SNAPSHOT_CAP: usize = 4_096;

/// One strike-affecting event awaiting finality: a miss (burns, strike record, possibly a
/// suspension), a served-response streak reset, or an identity's first sighting — all persisted
/// so the refold baseline and the standing clock carry them.
enum ServiceEvent {
    Miss(ServiceMiss),
    /// (identity, preserved last-strike daa — keeps the rate-limit armed across a serve).
    Reset(Hash, u64),
    Sighting(Hash),
    /// An inference-reward win (H8 routing): minted by the coinbase of the chain block whose
    /// selected parent finalizes it.
    Reward(ServiceReward),
}

/// RAM mirror of the persisted standing state: first sightings and the full strike history per
/// identity, in event order. Standing is evaluated at a lagged anchor (`pov − STANDING_LAG`), so
/// every row the evaluation reads is finality-flushed on every node long before any POV that
/// reads it — the answer is a pure function of reorg-immune data, identical live, on catch-up
/// and on refold. Also the flush's idempotency source: a row already mirrored is never
/// re-persisted nor re-committed.
#[derive(Default)]
pub(super) struct StandingIndex {
    first_seen: std::collections::HashMap<Hash, u64>,
    /// (event daa, count, last_daa) per identity, ascending by daa (flush order). `last_daa` is
    /// mirrored so a served reset (`last_daa` zero or strictly older than the row daa) stays
    /// distinguishable from an executed suspension (`last_daa` equal to its own row daa) — both
    /// carry a zero count, but only the second is a strike.
    history: std::collections::HashMap<Hash, Vec<(u64, u32, u64)>>,
}

impl StandingIndex {
    /// Records a sighting; false if the identity was already sighted.
    fn record_sighting(&mut self, id: Hash, daa: u64) -> bool {
        match self.first_seen.entry(id) {
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(daa);
                true
            }
            std::collections::hash_map::Entry::Occupied(_) => false,
        }
    }

    /// Records a strike-log row; false if that (identity, daa) row is already mirrored.
    fn record_strike(&mut self, id: Hash, daa: u64, count: u32, last_daa: u64) -> bool {
        let rows = self.history.entry(id).or_default();
        if rows.iter().rev().any(|(d, _, _)| *d == daa) {
            return false;
        }
        rows.push((daa, count, last_daa));
        rows.sort_unstable_by_key(|(d, _, _)| *d);
        true
    }

    /// Strikes an identity has taken over the whole retained log, suspensions included. Display
    /// only: the live counter resets on a served response and on an executed suspension, so it
    /// cannot answer "how often has this miner failed". Derived from the mirror, so no extra
    /// state and no disk read.
    fn lifetime_strikes(&self) -> Vec<(Hash, u32)> {
        self.history
            .iter()
            .filter_map(|(id, rows)| {
                let n = rows.iter().filter(|(daa, count, last)| *count > 0 || (*last > 0 && *last == *daa)).count() as u32;
                (n > 0).then_some((*id, n))
            })
            .collect()
    }

    /// Whether the identity is in standing at `pov`: sighted at or before the lagged anchor.
    /// Before `service_bond_v2_activation` (`v2` false) a non-zero strike count as of that
    /// anchor also revokes standing; at and after it, standing is probation-only — strikes
    /// already carry their own penalty and no longer demote the reward rate.
    fn standing(&self, id: &Hash, pov: u64, v2: bool) -> bool {
        let anchor = pov.saturating_sub(keryx_consensus_core::collateral::SERVICE_STANDING_LAG_DAA);
        if !self.first_seen.get(id).is_some_and(|&f| f <= anchor) {
            return false;
        }
        if v2 {
            return true;
        }
        match self.history.get(id) {
            None => true,
            Some(rows) => {
                let at = rows.partition_point(|(d, _, _)| *d <= anchor);
                at == 0 || rows[at - 1].1 == 0
            }
        }
    }
}

/// Everything needed to reverse one folded chain block's vault mutations. All fields are pure
/// functions of the chain, so entries stay valid across refolds.
struct VaultUndo {
    added: Vec<(Hash, EscrowClaim)>,
    misses: Vec<ServiceMiss>,
    expired: Vec<(Hash, EscrowClaim)>,
}

/// RAM-only service-ledger state folded along the committed selected chain.
#[derive(Default)]
pub(super) struct ServiceLedgerSync {
    ledger: ServiceLedger,
    /// Light (vault-less) ledger state as of each folded chain-block index. The vault itself is
    /// restored through `undo` — cloning a full burnable window per chain block does not scale.
    snapshots: std::collections::BTreeMap<u64, keryx_consensus_core::collateral::LightSnapshot>,
    /// Per-block vault undo entries, keyed like `snapshots`.
    undo: std::collections::BTreeMap<u64, VaultUndo>,
    /// Chain index the ledger is folded up to.
    tip: Option<u64>,
    /// Events awaiting finality depth, in chain order as (chain index, daa, event). Truncated on
    /// reorg like the chain itself; entries deeper than finality are written to the stores.
    queue: std::collections::VecDeque<(u64, u64, ServiceEvent)>,
    /// Highest event daa already persisted to the stores.
    deep_cursor_daa: u64,
    /// Miss daa keyed by `(miner, request_hash, strike count)`, kept across refolds so a miss is
    /// logged once. The count is part of the key: a request hash repeats whenever an identical
    /// payload is resubmitted, and its later strike must still be logged.
    logged: std::collections::HashMap<(Hash, [u8; 32], u32), u64>,
}

/// Logs the misses of one fold that have not been logged yet.
fn log_new_service_misses(
    logged: &mut std::collections::HashMap<(Hash, [u8; 32], u32), u64>,
    daa: u64,
    misses: &[ServiceMiss],
) {
    for miss in misses.iter() {
        if logged.insert((miss.miner, miss.request_hash, miss.consecutive_misses), daa).is_some() {
            continue;
        }
        let burned_total: u64 = miss.burned.iter().map(|c| c.value).sum();
        info!(
            "service-bond: miss #{} by miner {} on request {} → {:?}, {} claims / {} sompi (awaiting finality)",
            miss.consecutive_misses,
            miss.miner,
            hex::encode(miss.request_hash),
            miss.penalty,
            miss.burned.len(),
            burned_total
        );
    }
}

impl VirtualStateProcessor {
    /// `(identity, proven tier, delegated escrow key)` of each paid mergeset blue of chain block
    /// `hash` — the same blue set the coinbase rewards. The identity is [`miner_key`] of the
    /// blue's payout SPK; the escrow key is the hot key it delegated to (cert enforced by block
    /// validity past the gate). The tier is read from the blue's committed `header.pom_tier` (bound
    /// to `proof.tier` in live validation), NOT `pom_tier_store`: the header is committed and
    /// retained deeper than block bodies, so a freshly synced node derives identical cohorts,
    /// whereas the tier store cannot be populated for proofless historical bodies. Blues without an
    /// escrow announcement are skipped.
    pub(super) fn service_producers_of_chain_block(&self, hash: Hash) -> Vec<(Hash, u8, Hash)> {
        let ghostdag_data = self.ghostdag_store.get_data(hash).unwrap();
        let non_daa = self.daa_excluded_store.get_mergeset_non_daa(hash).unwrap();
        ghostdag_data
            .mergeset_blues
            .iter()
            .filter(|b| !non_daa.contains(b))
            .filter_map(|b| {
                let tier = self.headers_store.get_header(*b).unwrap().pom_tier;
                let txs = self.block_transactions_store.get(*b).unwrap();
                let coinbase = self.coinbase_manager.deserialize_coinbase_payload(&txs[0].payload).unwrap();
                let pubkey = crate::processes::coinbase::parse_escrow_pubkey_from_extra_data(coinbase.miner_data.extra_data)?;
                Some((miner_key(&coinbase.miner_data.script_public_key), tier, escrow_miner_key(&pubkey)))
            })
            .collect()
    }

    /// Eligible responsible miners for a `target_tier` request, seen from committed chain block
    /// `seed`: the distinct miner keys with at least one proven `target_tier` blue merged by a
    /// chain block whose daa_score lies in `(seed.daa − window_daa, seed.daa]`, floored at `seed`'s
    /// committed pruning point. A pure function of the chain, so every node derives the identical
    /// set. Empty if `seed` is not a committed chain block.
    #[allow(dead_code)] // consumed by the coming penalty/RPC layer; exercised by tests today
    pub(crate) fn service_eligible_miners_windowed(&self, seed: Hash, target_tier: u8, window_daa: u64) -> Vec<(Hash, Hash)> {
        // Read before the chain lock so the two locks are never held in inverse order.
        let own_pp = self.pruning_point_store.read().pruning_point().unwrap();
        let sc = self.selected_chain_store.read();
        self.service_eligible_miners_in(&*sc, seed, target_tier, window_daa, own_pp)
    }

    fn service_eligible_miners_in(
        &self,
        sc: &impl SelectedChainStoreReader,
        seed: Hash,
        target_tier: u8,
        window_daa: u64,
        own_pp: Hash,
    ) -> Vec<(Hash, Hash)> {
        let Ok(seed_idx) = sc.get_by_hash(seed) else {
            return vec![];
        };
        let seed_header = self.headers_store.get_header(seed).unwrap();
        let daa_bound = seed_header.daa_score.saturating_sub(window_daa);
        // A window crossing below retained history only happens while re-validating blocks near
        // the local pruning point (fresh IBD / restart catch-up). Past the ledger gate the part
        // below the pruning point is read from the imported snapshot; before it the audit arms
        // empty rather than with a cohort no local search can reproduce.
        let (pruning_idx, below) = match self.window_floor_in_retention(sc, seed_header.pruning_point, own_pp, daa_bound) {
            Some(idx) => (idx, false),
            None if self.service_ledger_activation.is_active(seed_header.daa_score) => match sc.get_by_hash(own_pp) {
                Ok(idx) => (idx, true),
                Err(_) => return vec![],
            },
            None => return vec![],
        };
        let bottom = self.chain_index_at_or_below_daa(sc, daa_bound, seed_idx, pruning_idx).max(pruning_idx);
        let mut recent = Vec::new();
        if below {
            let pp_daa = self.headers_store.get_daa_score(own_pp).unwrap();
            recent.extend(
                self.service_imported_producers
                    .read()
                    .iter()
                    .filter(|(daa, _, _, _)| *daa > daa_bound && *daa <= pp_daa)
                    .map(|(_, id, tier, escrow)| (*id, *tier, *escrow)),
            );
        }
        for i in (bottom + 1)..=seed_idx {
            recent.extend(self.service_producers_of_chain_block(sc.get_by_index(i).unwrap()));
        }
        eligible_pairs(&recent, target_tier)
    }

    #[allow(dead_code)]
    pub(crate) fn service_eligible_miners(&self, seed: Hash, target_tier: u8) -> Vec<(Hash, Hash)> {
        self.service_eligible_miners_windowed(seed, target_tier, SERVICE_ELIGIBILITY_WINDOW_DAA)
    }

    /// Accepted AiRequests `(request_hash, tier)` and AiResponses `(request_hash, verified
    /// responder)` of committed chain block `hash`, across its whole mergeset acceptance data.
    /// Requests for models outside the tier lineup are skipped; a v1 response or an invalid
    /// responder signature yields `None` (a volunteer — never serves the assignment).
    fn service_events_of_chain_block(
        &self,
        hash: Hash,
        txid_identity: bool,
    ) -> (Vec<([u8; 32], u8, u32)>, Vec<([u8; 32], u64)>, Vec<([u8; 32], Option<Hash>)>) {
        let mut requests = Vec::new();
        let mut request_rewards = Vec::new();
        let mut responses = Vec::new();
        let acceptance = self.acceptance_data_store.get(hash).unwrap();
        for mbad in acceptance.iter() {
            let txs = self.block_transactions_store.get(mbad.block_hash).unwrap();
            for entry in mbad.accepted_transactions.iter() {
                let tx = &txs[entry.index_within_block as usize];
                if tx.is_ai_request() {
                    if let Some(req) = AiRequestPayload::deserialize(&tx.payload) {
                        if let Some(tier) = POM_TIERS_H6.iter().position(|t| t.model_id == req.model_id) {
                            // Past the gate a request is identified by its transaction id, which is
                            // unique by construction. The payload digest is not: the same prompt with
                            // the same parameters is the same hash, so two senders — or one retry —
                            // used to collide into a single, unanswerable assignment.
                            let mut request_hash = [0u8; 32];
                            if txid_identity {
                                request_hash.copy_from_slice(&tx.id().as_bytes());
                            } else {
                                let digest = blake2b_simd::blake2b(&tx.payload);
                                request_hash.copy_from_slice(&digest.as_bytes()[..32]);
                            }
                            requests.push((request_hash, tier as u8, req.max_tokens));
                            request_rewards.push((request_hash, req.inference_reward));
                        }
                    }
                } else if tx.is_ai_response() {
                    if let Some(resp) = AiResponsePayload::deserialize(&tx.payload) {
                        responses.push((resp.request_hash, verified_responder(&resp)));
                    }
                }
            }
        }
        (requests, request_rewards, responses)
    }

    /// `(identity, coinbase payout script)` of the chain block's producers — the reward-mint
    /// resolution input, from the same walk as [`Self::service_producers_of_chain_block`].
    fn service_producer_spks_of_chain_block(&self, hash: Hash) -> Vec<(Hash, ScriptPublicKey)> {
        let ghostdag_data = self.ghostdag_store.get_data(hash).unwrap();
        let non_daa = self.daa_excluded_store.get_mergeset_non_daa(hash).unwrap();
        ghostdag_data
            .mergeset_blues
            .iter()
            .filter(|b| !non_daa.contains(b))
            .filter_map(|b| {
                let txs = self.block_transactions_store.get(*b).unwrap();
                let coinbase = self.coinbase_manager.deserialize_coinbase_payload(&txs[0].payload).unwrap();
                crate::processes::coinbase::parse_escrow_pubkey_from_extra_data(coinbase.miner_data.extra_data)?;
                Some((miner_key(&coinbase.miner_data.script_public_key), coinbase.miner_data.script_public_key))
            })
            .collect()
    }

    /// The current service-ledger escrow claims of `miner` — future RPC surface, test-read today.
    #[allow(dead_code)]
    pub(crate) fn service_vault_claims(&self, miner: &Hash) -> Vec<EscrowClaim> {
        self.service_ledger.lock().ledger.vault_claims(miner)
    }

    /// Point-in-time service-bond enforcement state: live strikes, suspensions and the misses
    /// still awaiting finality.
    pub(crate) fn service_strikes_snapshot(&self, virtual_daa_score: u64) -> ServiceStrikesSnapshot {
        let mut snapshot = ServiceStrikesSnapshot { virtual_daa_score, ..Default::default() };
        {
            let sync = self.service_ledger.lock();
            snapshot.strikes = sync.ledger.strike_entries();
            snapshot.pending_burns = sync
                .queue
                .iter()
                .filter_map(|(_, daa, event)| match event {
                    ServiceEvent::Miss(miss) => Some((
                        miss.miner,
                        *daa,
                        miss.consecutive_misses,
                        miss.burned.len() as u32,
                        miss.burned.iter().map(|c| c.value).sum(),
                        miss.request_hash,
                    )),
                    ServiceEvent::Reset(..) | ServiceEvent::Sighting(_) | ServiceEvent::Reward(_) => None,
                })
                .collect();
        }
        snapshot.lifetime_strikes = self.service_standing.read().lifetime_strikes();
        snapshot.lifetime_strikes.sort_unstable();
        // Only the suspensions in force at this POV: the map keeps every record ever flushed, so
        // that re-validating an old POV reaches the verdict a live node reached. Reporting it raw
        // leaves expired suspensions on display long after production resumed.
        snapshot.suspended = self
            .service_suspended
            .read()
            .iter()
            .filter(|(_, until)| until.saturating_sub(SERVICE_SUSPENSION_DAA) <= virtual_daa_score && virtual_daa_score < **until)
            .map(|(m, until)| (*m, *until))
            .collect();
        snapshot.suspended.sort_unstable();
        snapshot
    }

    /// Escrow claims created by committed chain block `hash`'s coinbase, keyed by producing miner:
    /// for each paid mergeset blue, the CSV escrow output locking the escrow key that blue's own
    /// coinbase announces. Standard miners (escrow burned at emission) contribute none.
    fn service_escrows_of_chain_block(&self, hash: Hash) -> Vec<(Hash, EscrowClaim)> {
        let daa = self.headers_store.get_daa_score(hash).unwrap();
        let ghostdag_data = self.ghostdag_store.get_data(hash).unwrap();
        let non_daa = self.daa_excluded_store.get_mergeset_non_daa(hash).unwrap();
        let txs = self.block_transactions_store.get(hash).unwrap();
        let coinbase = &txs[0];
        let coinbase_id = coinbase.id();
        let mut claims = Vec::new();
        // Walk the coinbase outputs in lockstep with the paid blues, matching each blue to the CSV
        // output locking its own announced escrow key. Keying on that key rather than on the
        // position after the miner payout keeps the pairing exact when the payout output is absent
        // (a suspended producer's burned cut emits none). The cursor keeps two blues of the same
        // miner in chain order.
        let mut cursor = 0usize;
        for blue in ghostdag_data.mergeset_blues.iter().filter(|b| !non_daa.contains(b)) {
            let blue_txs = self.block_transactions_store.get(*blue).unwrap();
            let blue_coinbase = self.coinbase_manager.deserialize_coinbase_payload(&blue_txs[0].payload).unwrap();
            if blue_coinbase.subsidy == 0 {
                continue;
            }
            let Some(pubkey) = crate::processes::coinbase::parse_escrow_pubkey_from_extra_data(blue_coinbase.miner_data.extra_data)
            else {
                continue;
            };
            let Some(escrow_idx) = (cursor..coinbase.outputs.len()).find(|&i| {
                coinbase.outputs[i].value > 0 && csv_escrow_pubkey(coinbase.outputs[i].script_public_key.script()) == Some(pubkey)
            }) else {
                continue;
            };
            cursor = escrow_idx + 1;
            claims.push((
                miner_key(&blue_coinbase.miner_data.script_public_key),
                EscrowClaim {
                    outpoint: TransactionOutpoint::new(coinbase_id, escrow_idx as u32),
                    value: coinbase.outputs[escrow_idx].value,
                    daa,
                },
            ));
        }
        claims
    }

    /// Folds one committed chain block into `ledger` and returns its outcome. No-op before
    /// `pom_v3_activation` (a per-block property, so the fold is canonical across nodes and IBD).
    /// `warmup` folds a block whose strike events are already persisted: request/vault memory only,
    /// burned claims dropped through the burn set. Events only become enforceable once
    /// finality-deep (see `advance_service_ledger`).
    fn fold_service_chain_block(
        &self,
        ledger: &mut ServiceLedger,
        sc: &impl SelectedChainStoreReader,
        hash: Hash,
        own_pp: Hash,
        live: bool,
        warmup: bool,
    ) -> (FoldOutcome, Vec<(Hash, EscrowClaim)>) {
        let daa = self.headers_store.get_daa_score(hash).unwrap();
        if !self.pom_v3_activation.is_active(daa) {
            return (FoldOutcome::default(), Vec::new());
        }
        ledger.set_window_v2_activation(self.service_bond_v2_activation.daa_score());
        ledger.set_reward_routing_activation(self.reward_routing_activation.daa_score());
        ledger.set_burnable_window(self.service_burnable_window_daa);
        let (requests, request_rewards, responses) = self.service_events_of_chain_block(hash, self.reward_routing_activation.is_active(daa));
        let producers =
            if self.reward_routing_activation.is_active(daa) { self.service_producer_spks_of_chain_block(hash) } else { Vec::new() };
        // Claims whose outpoint is already in the (reorg-immune) burn store are dead on arrival:
        // live claims never are, and the frontier-daa refold must not resurrect a claim whose
        // burning miss is rate-limit-absorbed by the baseline.
        let escrows: Vec<(Hash, EscrowClaim)> = {
            let burned = self.service_burned.read();
            self.service_escrows_of_chain_block(hash).into_iter().filter(|(_, c)| !burned.contains_key(&c.outpoint)).collect()
        };
        if live && !requests.is_empty() {
            for (rh, tier, max_tokens) in requests.iter() {
                info!(
                    "service-bond: request {} accepted at daa {}, tier {}, max_tokens {}",
                    hex::encode(rh),
                    daa,
                    tier,
                    max_tokens
                );
            }
        }
        let eligibility_window = if self.service_bond_v2_activation.is_active(daa) {
            SERVICE_ELIGIBILITY_WINDOW_DAA_V2
        } else {
            SERVICE_ELIGIBILITY_WINDOW_DAA
        };
        let cohort = |tier: u8| {
            let set = self.service_eligible_miners_in(sc, hash, tier, eligibility_window, own_pp);
            // Only the live fold logs: a refold replays the same armings.
            if live {
                info!("service-bond: audit armed at daa {}, tier {}, cohort {}", daa, tier, set.len());
            }
            set
        };
        if warmup {
            let burned = self.service_burned.read();
            ledger.on_chain_block_warmup_with_rewards(
                daa,
                &requests,
                &request_rewards,
                &responses,
                &escrows,
                &producers,
                &|op| burned.contains_key(op),
                cohort,
            );
            (FoldOutcome::default(), escrows)
        } else {
            let outcome = ledger.on_chain_block_with_rewards(
                daa,
                &requests,
                &request_rewards,
                &responses,
                &escrows,
                &producers,
                |id| self.service_standing_at(id, daa),
                cohort,
            );
            (outcome, escrows)
        }
    }

    /// Whether `identity` is in standing at `pov_daa` — sighted at the lagged anchor, and
    /// strike-free there too before the v2 gate. Pure function of finality-flushed rows plus
    /// the POV daa; identical on every node at every POV.
    pub(super) fn service_standing_at(&self, identity: &Hash, pov_daa: u64) -> bool {
        self.service_standing.read().standing(identity, pov_daa, self.service_bond_v2_activation.is_active(pov_daa))
    }

    /// Highest chain index whose daa is at or below `bound_daa`, searched down to `floor_idx`
    /// only. The ratio-reward variant floors its search at `hi_idx - ratio_reward_window_daa`,
    /// which silently truncates a service-ledger refold to a window six times too short.
    fn service_chain_index_at_or_below_daa(
        &self,
        sc: &impl SelectedChainStoreReader,
        bound_daa: u64,
        hi_idx: u64,
        floor_idx: u64,
    ) -> u64 {
        let daa_at = |i: u64| self.headers_store.get_daa_score(sc.get_by_index(i).unwrap()).unwrap();
        let (mut lo, mut hi) = (floor_idx, hi_idx);
        if lo >= hi || daa_at(lo) > bound_daa {
            return lo;
        }
        while lo < hi {
            let mid = lo + (hi - lo + 1) / 2;
            if daa_at(mid) <= bound_daa {
                lo = mid
            } else {
                hi = mid - 1
            }
        }
        lo
    }

    /// Rebuilds the ledger up to chain index `to` by folding the committed chain from an empty
    /// state — the cold-start and deep-reorg path. The strike baseline is reloaded from the
    /// store (the exact state at the persisted frontier `cursor_daa`), blocks at or below the
    /// frontier warm up request/vault memory only (their strike events already live in the
    /// baseline, their burns in the burn set), and blocks above it fold normally and re-queue
    /// their events.
    fn refold_service_ledger(
        &self,
        sc: &impl SelectedChainStoreReader,
        to: u64,
        pruning_point: Hash,
        cursor_daa: u64,
        queue: &mut std::collections::VecDeque<(u64, u64, ServiceEvent)>,
        logged: &mut std::collections::HashMap<(Hash, [u8; 32], u32), u64>,
    ) -> ServiceLedger {
        let mut ledger = ServiceLedger::default();
        let Ok(to_hash) = sc.get_by_index(to) else {
            return ledger;
        };
        let to_daa = self.headers_store.get_daa_score(to_hash).unwrap();
        ledger.set_base(std::sync::Arc::new(self.load_strike_base()));
        ledger.set_first_seen_base(std::sync::Arc::new(
            self.service_standing.read().first_seen.iter().map(|(k, v)| (*k, *v)).collect(),
        ));
        // One burnable window below the persisted frontier keeps every pending request and vault
        // claim readable at the frontier warm. A frontier of zero (nothing persisted yet) falls
        // back to the finality anchor: everything above it is re-derived.
        let start = if cursor_daa > 0 { cursor_daa } else { to_daa.saturating_sub(self.finality_depth) };
        let daa_bound = start.saturating_sub(self.service_burnable_window_daa);
        let pruning_idx = sc.get_by_hash(pruning_point).unwrap_or(0);
        let mut bottom = self.service_chain_index_at_or_below_daa(sc, daa_bound, to, pruning_idx);
        // A persisted sample snapshot at or below `to` is the exact state there: restore it and
        // fold only what lies above.
        let sample = self
            .service_ledger_hashes
            .read()
            .keys()
            .filter_map(|h| sc.get_by_hash(*h).ok().filter(|idx| *idx <= to).map(|idx| (idx, *h)))
            .max();
        if let Some((sample_idx, sample_hash)) = sample {
            let restored = self
                .service_ledger_snapshot_store
                .get(sample_hash)
                .ok()
                .flatten()
                .and_then(|bytes| ServiceLedgerSnapshot::from_bytes(&bytes).ok());
            if let Some(snapshot) = restored {
                ledger.restore_snapshot(&snapshot);
                bottom = sample_idx;
                info!("service-bond: refold from the snapshot at chain index {} (daa {})", sample_idx, self.headers_store.get_daa_score(sample_hash).unwrap());
            }
        }
        if bottom == pruning_idx && pruning_idx > 0 {
            let bottom_daa = self.headers_store.get_daa_score(sc.get_by_index(bottom).unwrap()).unwrap();
            if bottom_daa > daa_bound {
                warn!(
                    "service-bond: cold refold clamped at the pruning point (daa {} > bound {}) — events below it cannot be re-derived",
                    bottom_daa, daa_bound
                );
            }
        }
        for i in (bottom + 1)..=to {
            let hash = sc.get_by_index(i).unwrap();
            let daa = self.headers_store.get_daa_score(hash).unwrap();
            // Strictly below the frontier: fully persisted. AT the frontier: re-derived and
            // re-queued — a crash may have flushed that daa partially, and the flush itself is
            // idempotent (already-mirrored rows are skipped).
            let warmup = daa < cursor_daa;
            let (outcome, _added) = self.fold_service_chain_block(&mut ledger, sc, hash, pruning_point, false, warmup);
            log_new_service_misses(logged, daa, &outcome.misses);
            if daa >= cursor_daa {
                // Sightings first: a partially flushed daa must never persist a claim's burn
                // before its identity's sighting.
                for miner in outcome.sightings {
                    queue.push_back((i, daa, ServiceEvent::Sighting(miner)));
                }
                for miss in outcome.misses {
                    queue.push_back((i, daa, ServiceEvent::Miss(miss)));
                }
                for (miner, preserved) in outcome.resets {
                    queue.push_back((i, daa, ServiceEvent::Reset(miner, preserved)));
                }
                for reward in outcome.rewards {
                    queue.push_back((i, daa, ServiceEvent::Reward(reward)));
                }
            }
        }
        ledger
    }

    /// The refold baseline: the last log record per miner. The log iterates in event order
    /// (daa-BE-prefixed keys), so a plain insert keeps the latest.
    fn load_strike_base(&self) -> std::collections::BTreeMap<Hash, StrikeEntry> {
        let mut base = std::collections::BTreeMap::new();
        for entry in self.service_strike_store.iterator() {
            let (key, record) = entry.unwrap();
            let (_, miner) = crate::model::stores::service_strike::StrikeLogKey::parse(&key);
            base.insert(miner, record);
        }
        base
    }

    /// Advances the service ledger along the committed `chain_path` — called from `resolve_virtual`
    /// right after the virtual state is committed, so the selected-chain store reflects the new
    /// chain. Reorgs restore the snapshot at the common ancestor; a cold start or a reorg deeper
    /// than the retained snapshots refolds the horizon.
    pub(super) fn advance_service_ledger(&self, chain_path: &ChainPath, pruning_point: Hash) {
        let sc = self.selected_chain_store.read();
        let (tip_idx, tip_hash) = sc.get_tip().unwrap();
        if !self.pom_v3_activation.is_active(self.headers_store.get_daa_score(tip_hash).unwrap()) {
            return;
        }
        let tip_daa = self.headers_store.get_daa_score(tip_hash).unwrap();
        let common = tip_idx - chain_path.added.len() as u64;
        let mut sync = self.service_ledger.lock();
        // A reorg (or restore) drops queued misses above the common ancestor with the chain.
        sync.queue.retain(|(idx, _, _)| *idx <= common);
        if sync.tip != Some(common) {
            // A reorg walks the vault back through the per-block undo log and restores the rest
            // from the light snapshot; anything deeper (or a cold start) refolds.
            let undoable = sync.tip.is_some_and(|tip| {
                common <= tip && sync.snapshots.contains_key(&common) && ((common + 1)..=tip).all(|i| sync.undo.contains_key(&i))
            });
            if undoable {
                let tip = sync.tip.unwrap();
                for i in ((common + 1)..=tip).rev() {
                    let VaultUndo { added, misses, expired } = sync.undo.get(&i).unwrap();
                    let (added, misses, expired) = (added.clone(), misses.clone(), expired.clone());
                    sync.ledger.undo_vault(&added, &misses, &expired);
                }
                let snap = sync.snapshots.get(&common).unwrap().clone();
                sync.ledger.restore_light(&snap);
            } else {
                let cursor = sync.deep_cursor_daa;
                let mut queue = std::mem::take(&mut sync.queue);
                queue.clear();
                let mut logged = std::mem::take(&mut sync.logged);
                let ledger = self.refold_service_ledger(&*sc, common, pruning_point, cursor, &mut queue, &mut logged);
                sync.queue = queue;
                sync.logged = logged;
                sync.ledger = ledger;
            }
        }
        sync.snapshots.split_off(&(common + 1));
        sync.undo.split_off(&(common + 1));
        for removed in chain_path.removed.iter() {
            if self.service_ledger_hashes.write().remove(removed).is_some() {
                self.service_ledger_snapshot_store.delete(*removed).unwrap();
            }
        }
        let mut sampled = false;
        for (k, h) in chain_path.added.iter().enumerate() {
            let idx = common + 1 + k as u64;
            let daa = self.headers_store.get_daa_score(*h).unwrap();
            // Blocks at or below the persisted event frontier carry events the stores already
            // hold (transferred sealed state, or this node's own earlier flush): folding them
            // live would re-derive strikes on top of a baseline that already counts them, so the
            // escalation runs away and burns vaults the network never burned. Mirrors the cold
            // refold's own warmup rule.
            let warmup = daa < sync.deep_cursor_daa;
            let (outcome, added) = self.fold_service_chain_block(&mut sync.ledger, &*sc, *h, pruning_point, true, warmup);
            log_new_service_misses(&mut sync.logged, daa, &outcome.misses);
            sync.undo.insert(idx, VaultUndo { added, misses: outcome.misses.clone(), expired: outcome.expired });
            for miner in outcome.sightings {
                sync.queue.push_back((idx, daa, ServiceEvent::Sighting(miner)));
            }
            for miss in outcome.misses {
                sync.queue.push_back((idx, daa, ServiceEvent::Miss(miss)));
            }
            for (miner, preserved) in outcome.resets {
                sync.queue.push_back((idx, daa, ServiceEvent::Reset(miner, preserved)));
            }
            for reward in outcome.rewards {
                sync.queue.push_back((idx, daa, ServiceEvent::Reward(reward)));
            }
            let snapshot = sync.ledger.light_snapshot();
            sync.snapshots.insert(idx, snapshot);
            if self.is_pruning_sample_block(*h) {
                let mut snapshot = sync.ledger.snapshot();
                snapshot.recent_producers = self.recent_producers_below(&*sc, idx, daa);
                let bytes = snapshot.to_bytes();
                self.service_ledger_hashes.write().insert(*h, ServiceLedgerSnapshot::hash_of_bytes(&bytes));
                self.service_ledger_snapshot_store.set(*h, bytes).unwrap();
                sampled = true;
            }
        }
        if sampled {
            self.gc_service_ledger_snapshots(pruning_point);
        }
        while sync.snapshots.len() > SERVICE_SNAPSHOT_CAP {
            sync.snapshots.pop_first();
        }
        while sync.undo.len() > SERVICE_SNAPSHOT_CAP {
            sync.undo.pop_first();
        }
        sync.tip = Some(tip_idx);
        // Bound the logged set by the deepest span a refold can revisit (the finality anchor
        // plus the warmup horizon).
        let logged_span = self.finality_depth + self.service_burnable_window_daa;
        sync.logged.retain(|_, daa| *daa + logged_span > tip_daa);
        // Events now deeper than finality are reorg-immune on every acceptable POV: persist the
        // burned outpoints, the strike records and the suspensions, in chain order, and advance
        // the sealed commitment (one seal per flushed event daa — all events of a daa qualify
        // together, so a daa is never split across flushes).
        let mut sealing_daa: Option<u64> = None;
        while sync.queue.front().is_some_and(|(_, daa, _)| daa + self.finality_depth <= tip_daa) {
            let (_, daa, event) = sync.queue.pop_front().unwrap();
            if sealing_daa.is_some_and(|d| d != daa) {
                self.service_commit_index.seal(sealing_daa.unwrap());
            }
            sealing_daa = Some(daa);
            match event {
                ServiceEvent::Sighting(miner) => {
                    if self.service_standing.write().record_sighting(miner, daa) {
                        self.service_first_seen_store.set(miner, daa).unwrap();
                        self.service_commit_index.add_row(&service_commit::first_seen_row_bytes(miner, daa));
                    }
                }
                ServiceEvent::Reset(miner, preserved) => {
                    if self.service_standing.write().record_strike(miner, daa, 0, preserved) {
                        self.service_strike_store.set(daa, miner, StrikeEntry { count: 0, last_daa: preserved }).unwrap();
                        self.service_commit_index.add_row(&service_commit::strike_row_bytes(daa, miner, 0, preserved));
                    }
                }
                ServiceEvent::Miss(miss) => {
                    for claim in miss.burned.iter() {
                        if self.service_burned.write().insert(claim.outpoint, daa).is_some() {
                            continue;
                        }
                        let key =
                            crate::model::stores::ai_slash::OutpointKey::new(claim.outpoint.transaction_id, claim.outpoint.index);
                        self.service_burn_store.set(key, daa).unwrap();
                        self.service_commit_index.add_row(&service_commit::burn_row_bytes(
                            claim.outpoint.transaction_id,
                            claim.outpoint.index,
                            daa,
                        ));
                    }
                    if !miss.burned.is_empty() {
                        info!(
                            "service-bond: burn FINAL for miner {} — {} claims, miss daa {}",
                            miss.miner,
                            miss.burned.len(),
                            daa
                        );
                    }
                    // Mirror the fold: an executed suspension logs as `{0, daa}` — the streak
                    // restarts, the daa keeps the rate-limit armed and re-derives the deadline.
                    let record = if miss.penalty == ServicePenalty::Suspend {
                        StrikeEntry { count: 0, last_daa: daa }
                    } else {
                        StrikeEntry { count: miss.consecutive_misses, last_daa: daa }
                    };
                    if self.service_standing.write().record_strike(miss.miner, daa, record.count, record.last_daa) {
                        self.service_strike_store.set(daa, miss.miner, record).unwrap();
                        self.service_commit_index.add_row(&service_commit::strike_row_bytes(daa, miss.miner, record.count, record.last_daa));
                    }
                    // A third strike, now reorg-immune, suspends the miner's production. The
                    // deadline is derived from the miss's own daa (deterministic), and the full
                    // window bites from finalization:
                    // [daa + finality, daa + finality + SERVICE_SUSPENSION_DAA].
                    if miss.penalty == ServicePenalty::Suspend {
                        let until = daa + self.finality_depth + SERVICE_SUSPENSION_DAA;
                        self.service_suspended.write().insert(miss.miner, until);
                        info!("service-bond: SUSPENSION FINAL for miner {} until daa {} (miss daa {})", miss.miner, until, daa);
                    }
                }
                ServiceEvent::Reward(reward) => {
                    if self.service_rewarded.write().insert(reward.request_hash) {
                        self.service_reward_store
                            .set(
                                crate::model::stores::service_reward::RewardKey(reward.request_hash),
                                RewardEntry { winner: reward.winner, amount: reward.amount, daa, spk: reward.spk.clone() },
                            )
                            .unwrap();
                        self.service_commit_index.add_row(&service_commit::reward_row_bytes(
                            reward.request_hash,
                            reward.winner,
                            reward.amount,
                            daa,
                            reward.spk.as_ref(),
                        ));
                        self.service_reward_recent.write().entry(daa).or_default().push((
                            reward.request_hash,
                            reward.amount,
                            reward.spk.clone(),
                        ));
                        info!(
                            "service-bond: reward FINAL for request {} → miner {} ({} sompi{})",
                            hex::encode(reward.request_hash),
                            reward.winner,
                            reward.amount,
                            if reward.spk.is_some() { "" } else { ", no payout script — burned" }
                        );
                    }
                }
            }
            sync.deep_cursor_daa = sync.deep_cursor_daa.max(daa);
        }
        if let Some(daa) = sealing_daa {
            self.service_commit_index.seal(daa);
        }
    }

    /// Producers of the chain blocks with daa in `(daa − SERVICE_ELIGIBILITY_WINDOW_DAA, daa]`
    /// ending at chain index `idx`, chain order.
    fn recent_producers_below(&self, sc: &impl SelectedChainStoreReader, idx: u64, daa: u64) -> Vec<(u64, Hash, u8, Hash)> {
        let bound = daa.saturating_sub(SERVICE_ELIGIBILITY_WINDOW_DAA);
        let mut out = Vec::new();
        let mut i = idx;
        loop {
            let Ok(h) = sc.get_by_index(i) else { break };
            let block_daa = self.headers_store.get_daa_score(h).unwrap();
            if block_daa <= bound {
                break;
            }
            for (id, tier, escrow) in self.service_producers_of_chain_block(h) {
                out.push((block_daa, id, tier, escrow));
            }
            if i == 0 {
                break;
            }
            i -= 1;
        }
        out.reverse();
        out
    }

    /// Whether `hash` opens a new finality epoch on its selected chain — the blocks a pruning
    /// point can be chosen among.
    pub(super) fn is_pruning_sample_block(&self, hash: Hash) -> bool {
        let Ok(sp) = self.ghostdag_store.get_selected_parent(hash) else { return false };
        if sp.is_origin() {
            return false;
        }
        let blue = self.headers_store.get_blue_score(hash).unwrap();
        let sp_blue = self.headers_store.get_blue_score(sp).unwrap();
        sp_blue / self.finality_depth < blue / self.finality_depth
    }

    /// Drops sample snapshots more than one finality depth below the pruning point.
    fn gc_service_ledger_snapshots(&self, pruning_point: Hash) {
        let floor = self.headers_store.get_blue_score(pruning_point).unwrap().saturating_sub(self.finality_depth);
        let samples: Vec<Hash> = self.service_ledger_hashes.read().keys().copied().collect();
        for sample in samples {
            let stale = match self.headers_store.get_blue_score(sample) {
                Ok(blue) => blue < floor,
                Err(_) => true,
            };
            if stale {
                self.service_ledger_hashes.write().remove(&sample);
                self.service_ledger_snapshot_store.delete(sample).unwrap();
            }
        }
    }

    /// Installs an imported ledger snapshot taken at chain block `sample` (the new pruning
    /// point): persists it and restarts the fold from it, so every event above `sample` is
    /// re-derived locally.
    pub(crate) fn install_service_ledger_snapshot(
        &self,
        sample: Hash,
        bytes: Vec<u8>,
        snapshot: ServiceLedgerSnapshot,
    ) -> keryx_consensus_core::errors::consensus::ConsensusResult<()> {
        let sample_idx = self
            .selected_chain_store
            .read()
            .get_by_hash(sample)
            .map_err(|_| keryx_consensus_core::errors::consensus::ConsensusError::General("snapshot sample is not a chain block"))?;
        let sample_daa = self.headers_store.get_daa_score(sample).unwrap();
        self.service_ledger_hashes.write().insert(sample, ServiceLedgerSnapshot::hash_of_bytes(&bytes));
        self.service_ledger_snapshot_store.set(sample, bytes).unwrap();
        *self.service_imported_producers.write() = snapshot.recent_producers.clone();
        let mut sync = self.service_ledger.lock();
        let mut ledger = ServiceLedger::default();
        ledger.restore_snapshot(&snapshot);
        sync.ledger = ledger;
        sync.snapshots.clear();
        sync.undo.clear();
        sync.queue.clear();
        sync.logged.clear();
        sync.tip = Some(sample_idx);
        sync.deep_cursor_daa = sync.deep_cursor_daa.max(sample_daa);
        info!("service-bond: ledger restored from the snapshot at {} (daa {})", sample, sample_daa);
        Ok(())
    }

    /// Canonical hash of the persisted ledger snapshot at `sample`, if this node holds it.
    /// Genesis carries the empty ledger.
    pub(super) fn service_ledger_hash_at(&self, sample: Hash) -> Option<Hash> {
        if sample == self.genesis.hash {
            return Some(ServiceLedgerSnapshot::default().hash());
        }
        self.service_ledger_hashes.read().get(&sample).copied()
    }

    /// Boot-time load of the persisted burned outpoints into the RAM set consulted by transaction
    /// validation, of the suspensions (re-derived from the strike log) into the RAM map consulted
    /// by block validation, and of the deep cursor — the persisted event frontier bounding the
    /// cold-start refold.
    pub(crate) fn load_service_burned(&self) {
        let mut rows: Vec<(u64, Vec<u8>)> = Vec::new();
        let mut set = self.service_burned.write();
        let mut cursor = 0u64;
        for entry in self.service_burn_store.iterator() {
            let (key, daa) = entry.unwrap();
            let tx_id_bytes: [u8; 32] = key[..32].try_into().unwrap();
            let index = u32::from_le_bytes(key[32..36].try_into().unwrap());
            set.insert(TransactionOutpoint::new(tx_id_bytes.into(), index), daa);
            rows.push((daa, service_commit::burn_row_bytes(tx_id_bytes.into(), index, daa).to_vec()));
            cursor = cursor.max(daa);
        }
        drop(set);
        let mut standing = self.service_standing.write();
        *standing = StandingIndex::default();
        let mut suspended = self.service_suspended.write();
        for entry in self.service_strike_store.iterator() {
            let (key, record) = entry.unwrap();
            let (event_daa, miner) = crate::model::stores::service_strike::StrikeLogKey::parse(&key);
            cursor = cursor.max(event_daa);
            rows.push((event_daa, service_commit::strike_row_bytes(event_daa, miner, record.count, record.last_daa).to_vec()));
            standing.record_strike(miner, event_daa, record.count, record.last_daa);
            // An executed suspension logs `{0, daa}` with `last_daa` equal to its own event daa;
            // a served reset carries a strictly older (or zero) preserved daa. The log is in
            // event order, so the last (largest) deadline per miner wins.
            if record.count == 0 && record.last_daa == event_daa {
                suspended.insert(miner, record.last_daa + self.finality_depth + SERVICE_SUSPENSION_DAA);
            }
        }
        drop(suspended);
        for entry in self.service_first_seen_store.iterator() {
            let (key, daa) = entry.unwrap();
            let miner: [u8; 32] = key[..32].try_into().unwrap();
            standing.record_sighting(Hash::from_bytes(miner), daa);
            rows.push((daa, service_commit::first_seen_row_bytes(Hash::from_bytes(miner), daa).to_vec()));
            cursor = cursor.max(daa);
        }
        drop(standing);
        {
            let mut rewarded = self.service_rewarded.write();
            let mut recent = self.service_reward_recent.write();
            rewarded.clear();
            recent.clear();
            for entry in self.service_reward_store.iterator() {
                let (key, record) = entry.unwrap();
                let request_hash: [u8; 32] = key[..32].try_into().unwrap();
                rewarded.insert(request_hash);
                recent.entry(record.daa).or_default().push((request_hash, record.amount, record.spk.clone()));
                rows.push((
                    record.daa,
                    service_commit::reward_row_bytes(request_hash, record.winner, record.amount, record.daa, record.spk.as_ref()),
                ));
                cursor = cursor.max(record.daa);
            }
        }
        self.service_commit_index.rebuild(rows);
        self.service_ledger.lock().deep_cursor_daa = cursor;
        let own_pp = self.pruning_point_store.read().pruning_point().ok();
        let mut hashes = self.service_ledger_hashes.write();
        hashes.clear();
        for (sample, bytes) in self.service_ledger_snapshot_store.entries() {
            hashes.insert(sample, ServiceLedgerSnapshot::hash_of_bytes(&bytes));
            if Some(sample) == own_pp {
                if let Ok(snapshot) = ServiceLedgerSnapshot::from_bytes(&bytes) {
                    *self.service_imported_producers.write() = snapshot.recent_producers;
                }
            }
        }
    }

    /// Coinbase mint expectation for a block whose selected parent is `sp`: the reward wins
    /// finalized exactly by `sp`'s fold — event daa in `(parent(sp).daa − finality, sp.daa −
    /// finality]` — in `(daa, request hash)` order. Wins with no payout script stay burned.
    /// `parent(sp)` is `sp`'s own selected parent: the committed selected chain is not consulted,
    /// so the expectation is the same on every chain candidate and from every node's point of view.
    pub(super) fn service_reward_mints_for(&self, sp: Hash) -> Vec<(ScriptPublicKey, u64)> {
        let sp_daa = self.headers_store.get_daa_score(sp).unwrap();
        if !self.reward_routing_activation.is_active(sp_daa) {
            return Vec::new();
        }
        let parent_daa = self
            .ghostdag_store
            .get_selected_parent(sp)
            .ok()
            .filter(|parent| !parent.is_origin())
            .and_then(|parent| self.headers_store.get_daa_score(parent).ok())
            .unwrap_or(0);
        let lo = parent_daa.saturating_sub(self.finality_depth);
        let hi = sp_daa.saturating_sub(self.finality_depth);
        if hi <= lo {
            return Vec::new();
        }
        let recent = self.service_reward_recent.read();
        let mut wins: Vec<([u8; 32], u64, ScriptPublicKey)> = Vec::new();
        for (_, entries) in recent.range((std::ops::Bound::Excluded(lo), std::ops::Bound::Included(hi))) {
            let mut at_daa: Vec<_> = entries.iter().filter_map(|(rh, amount, spk)| spk.clone().map(|s| (*rh, *amount, s))).collect();
            at_daa.sort_unstable_by_key(|(rh, _, _)| *rh);
            wins.extend(at_daa);
        }
        wins.truncate(keryx_consensus_core::collateral::MAX_REWARD_MINTS_PER_BLOCK);
        wins.into_iter().map(|(_, amount, spk)| (spk, amount)).collect()
    }

    /// Whether `producer` is under a finality-deep suspension at `daa_score`. The window is
    /// derived from the record itself (`[until − SERVICE_SUSPENSION_DAA, until)`), NOT from when
    /// this node learned it: a catch-up or fresh node re-validating an old POV must reach the
    /// exact verdict a live node reached — an unbounded lower edge would apply the suspension to
    /// POVs before its finalization.
    pub(super) fn is_producer_suspended(&self, producer: &Hash, daa_score: u64) -> bool {
        self.service_suspended
            .read()
            .get(producer)
            .is_some_and(|&until| until.saturating_sub(SERVICE_SUSPENSION_DAA) <= daa_score && daa_score < until)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use keryx_consensus_core::collateral::RESPONDER_SIG_DOMAIN;
    use keryx_inference::AiResponder;

    fn signed_response(seckey: &[u8; 32], tamper: bool) -> AiResponsePayload {
        let keypair = secp256k1::Keypair::from_seckey_slice(secp256k1::SECP256K1, seckey).unwrap();
        let mut resp = AiResponsePayload::new([7u8; 32], 900_000, [0x12u8; 34], 128);
        let mut hasher = blake2b_simd::Params::new().hash_length(32).to_state();
        hasher.update(RESPONDER_SIG_DOMAIN);
        hasher.update(&resp.signed_bytes());
        let msg = secp256k1::Message::from_digest_slice(hasher.finalize().as_bytes()).unwrap();
        let sig = keypair.sign_schnorr(msg);
        let (xonly, _) = keypair.x_only_public_key();
        resp.responder = Some(AiResponder { escrow_pubkey: xonly.serialize(), signature: *sig.as_ref() });
        if tamper {
            resp.response_length += 1;
        }
        resp
    }

    #[test]
    fn standing_ignores_strikes_post_v2() {
        use keryx_consensus_core::collateral::SERVICE_STANDING_LAG_DAA;

        let a = Hash::from_bytes([1u8; 32]);
        let mut idx = StandingIndex::default();
        idx.record_sighting(a, 0);
        idx.record_strike(a, 100, 1, 100);
        let pov = SERVICE_STANDING_LAG_DAA + 200;

        // Pre-gate a strike at the anchor revokes standing; at the gate it no longer does.
        assert!(!idx.standing(&a, pov, false));
        assert!(idx.standing(&a, pov, true));

        // The sighting probation is era-independent: a young identity has no standing either way.
        let b = Hash::from_bytes([2u8; 32]);
        idx.record_sighting(b, pov - 10);
        assert!(!idx.standing(&b, pov, false));
        assert!(!idx.standing(&b, pov, true));

        // A served reset (count 0) at the anchor restores standing in both eras.
        idx.record_strike(a, 150, 0, 100);
        assert!(idx.standing(&a, pov, false));
        assert!(idx.standing(&a, pov, true));
    }

    #[test]
    fn responder_signature_gates_the_identity() {
        let seckey = [0xC1u8; 32];
        let good = signed_response(&seckey, false);
        let expected = escrow_miner_key(&good.responder.as_ref().unwrap().escrow_pubkey);
        assert_eq!(verified_responder(&good), Some(expected));

        // v1 payload → no identity
        let v1 = AiResponsePayload::new([7u8; 32], 900_000, [0x12u8; 34], 128);
        assert_eq!(verified_responder(&v1), None);

        // signature over different bytes → rejected
        let tampered = signed_response(&seckey, true);
        assert_eq!(verified_responder(&tampered), None);

        // stolen signature under someone else's pubkey → rejected
        let mut stolen = signed_response(&seckey, false);
        stolen.responder.as_mut().unwrap().escrow_pubkey = [0x55u8; 32];
        assert_eq!(verified_responder(&stolen), None);
    }

    #[test]
    fn csv_escrow_pubkey_extracts_the_locked_key() {
        // <seq_len=3> <3 seq bytes> OP_CSV OpData32 <key 32> OP_CHECKSIG
        let mut script = vec![3u8, 0xAA, 0xBB, 0xCC];
        script.push(keryx_txscript::opcodes::codes::OpCheckSequenceVerify);
        script.push(keryx_txscript::opcodes::codes::OpData32);
        script.extend_from_slice(&[0x11u8; 32]);
        script.push(keryx_txscript::opcodes::codes::OpCheckSig);
        assert_eq!(csv_escrow_pubkey(&script), Some([0x11u8; 32]));
        assert_eq!(csv_escrow_pubkey(&[0u8; 10]), None);
    }
}
