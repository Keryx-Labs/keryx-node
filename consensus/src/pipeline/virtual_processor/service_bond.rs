use super::VirtualStateProcessor;
use crate::model::stores::{
    acceptance_data::AcceptanceDataStoreReader, block_transactions::BlockTransactionsStoreReader, daa::DaaStoreReader,
    ghostdag::GhostdagStoreReader, headers::HeaderStoreReader, pom_tier::PomTierStoreReader,
    selected_chain::SelectedChainStoreReader,
};
use keryx_consensus_core::collateral::{
    eligible_miners, escrow_miner_key, EscrowClaim, ServiceLedger, ServiceMiss, ServicePenalty,
    SERVICE_ELIGIBILITY_WINDOW_DAA, SERVICE_LEDGER_HORIZON_DAA, SERVICE_STRIKE_EPOCH_DAA, SERVICE_SUSPENSION_DAA,
};
use keryx_consensus_core::config::params::POM_TIERS_H6;
use keryx_consensus_core::tx::TransactionOutpoint;
use keryx_consensus_core::ChainPath;
use keryx_core::info;
use keryx_database::prelude::StoreResultExt;
use keryx_hashes::Hash;
use keryx_inference::{AiRequestPayload, AiResponsePayload};
use keryx_txscript::script_class::ScriptClass;

/// Domain separator of the V2 AiResponse responder signature.
const RESPONDER_SIG_DOMAIN: &[u8] = b"KeryxServiceResponderV1";

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
    let mut hasher = blake2b_simd::Params::new().hash_length(32).to_state();
    hasher.update(RESPONDER_SIG_DOMAIN);
    hasher.update(&resp.signed_bytes());
    let msg = secp256k1::Message::from_digest_slice(hasher.finalize().as_bytes()).unwrap();
    let pk = secp256k1::XOnlyPublicKey::from_slice(&r.escrow_pubkey).ok()?;
    let sig = secp256k1::schnorr::Signature::from_slice(&r.signature).ok()?;
    secp256k1::SECP256K1.verify_schnorr(&sig, &msg, &pk).ok()?;
    Some(escrow_miner_key(&r.escrow_pubkey))
}

/// Retained per-chain-block ledger snapshots; reorgs deeper than this fall back to a horizon refold.
const SERVICE_SNAPSHOT_CAP: usize = 4_096;

/// RAM-only service-ledger state folded along the committed selected chain.
#[derive(Default)]
pub(super) struct ServiceLedgerSync {
    ledger: ServiceLedger,
    /// Ledger state as of each folded chain-block index, for reorg restore.
    snapshots: std::collections::BTreeMap<u64, ServiceLedger>,
    /// Chain index the ledger is folded up to.
    tip: Option<u64>,
    /// Misses awaiting finality depth, in chain order as (chain index, daa, miss). Truncated on
    /// reorg like the chain itself; entries deeper than finality are written to the burn store.
    queue: std::collections::VecDeque<(u64, u64, ServiceMiss)>,
    /// Highest miss daa already persisted to the burn store.
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
    /// `(miner_key, proven tier)` of each paid mergeset blue of chain block `hash` — the same blue
    /// set the coinbase rewards. The key is the blue's announced escrow pubkey: the identity that
    /// holds the bond, signs V2 responses and takes the penalties. Blues without a stored tier or
    /// without an escrow announcement are skipped.
    pub(super) fn service_producers_of_chain_block(&self, hash: Hash) -> Vec<(Hash, u8)> {
        let ghostdag_data = self.ghostdag_store.get_data(hash).unwrap();
        let non_daa = self.daa_excluded_store.get_mergeset_non_daa(hash).unwrap();
        ghostdag_data
            .mergeset_blues
            .iter()
            .filter(|b| !non_daa.contains(b))
            .filter_map(|b| {
                let tier = self.pom_tier_store.get(*b).optional().unwrap()?;
                let txs = self.block_transactions_store.get(*b).unwrap();
                let coinbase = self.coinbase_manager.deserialize_coinbase_payload(&txs[0].payload).unwrap();
                let pubkey = crate::processes::coinbase::parse_escrow_pubkey_from_extra_data(coinbase.miner_data.extra_data)?;
                Some((escrow_miner_key(&pubkey), tier))
            })
            .collect()
    }

    /// Eligible responsible miners for a `target_tier` request, seen from committed chain block
    /// `seed`: the distinct miner keys with at least one proven `target_tier` blue merged by a
    /// chain block whose daa_score lies in `(seed.daa − window_daa, seed.daa]`, floored at `seed`'s
    /// committed pruning point. A pure function of the chain, so every node derives the identical
    /// set. Empty if `seed` is not a committed chain block.
    #[allow(dead_code)] // consumed by the coming penalty/RPC layer; exercised by tests today
    pub(crate) fn service_eligible_miners_windowed(&self, seed: Hash, target_tier: u8, window_daa: u64) -> Vec<Hash> {
        let sc = self.selected_chain_store.read();
        self.service_eligible_miners_in(&*sc, seed, target_tier, window_daa)
    }

    fn service_eligible_miners_in(
        &self,
        sc: &impl SelectedChainStoreReader,
        seed: Hash,
        target_tier: u8,
        window_daa: u64,
    ) -> Vec<Hash> {
        let Ok(seed_idx) = sc.get_by_hash(seed) else {
            return vec![];
        };
        let seed_header = self.headers_store.get_header(seed).unwrap();
        let pruning_idx = sc.get_by_hash(seed_header.pruning_point).unwrap_or(0);
        let daa_bound = seed_header.daa_score.saturating_sub(window_daa);
        let bottom = self.chain_index_at_or_below_daa(sc, daa_bound, seed_idx, pruning_idx).max(pruning_idx);
        let mut recent = Vec::new();
        for i in (bottom + 1)..=seed_idx {
            recent.extend(self.service_producers_of_chain_block(sc.get_by_index(i).unwrap()));
        }
        eligible_miners(&recent, target_tier)
    }

    #[allow(dead_code)]
    pub(crate) fn service_eligible_miners(&self, seed: Hash, target_tier: u8) -> Vec<Hash> {
        self.service_eligible_miners_windowed(seed, target_tier, SERVICE_ELIGIBILITY_WINDOW_DAA)
    }

    /// Accepted AiRequests `(request_hash, tier)` and AiResponses `(request_hash, verified
    /// responder)` of committed chain block `hash`, across its whole mergeset acceptance data.
    /// Requests for models outside the tier lineup are skipped; a v1 response or an invalid
    /// responder signature yields `None` (a volunteer — never serves the assignment).
    fn service_events_of_chain_block(&self, hash: Hash) -> (Vec<([u8; 32], u8, u32)>, Vec<([u8; 32], Option<Hash>)>) {
        let mut requests = Vec::new();
        let mut responses = Vec::new();
        let acceptance = self.acceptance_data_store.get(hash).unwrap();
        for mbad in acceptance.iter() {
            let txs = self.block_transactions_store.get(mbad.block_hash).unwrap();
            for entry in mbad.accepted_transactions.iter() {
                let tx = &txs[entry.index_within_block as usize];
                if tx.is_ai_request() {
                    if let Some(req) = AiRequestPayload::deserialize(&tx.payload) {
                        if let Some(tier) = POM_TIERS_H6.iter().position(|t| t.model_id == req.model_id) {
                            let digest = blake2b_simd::blake2b(&tx.payload);
                            let mut request_hash = [0u8; 32];
                            request_hash.copy_from_slice(&digest.as_bytes()[..32]);
                            requests.push((request_hash, tier as u8, req.max_tokens));
                        }
                    }
                } else if tx.is_ai_response() {
                    if let Some(resp) = AiResponsePayload::deserialize(&tx.payload) {
                        responses.push((resp.request_hash, verified_responder(&resp)));
                    }
                }
            }
        }
        (requests, responses)
    }

    /// The current service-ledger escrow claims of `miner` — future RPC surface, test-read today.
    #[allow(dead_code)]
    pub(crate) fn service_vault_claims(&self, miner: &Hash) -> Vec<EscrowClaim> {
        self.service_ledger.lock().ledger.vault_claims(miner)
    }

    /// Escrow claims created by committed chain block `hash`'s coinbase, keyed by producing miner:
    /// for each paid mergeset blue, the CSV escrow output that follows the blue's miner payout
    /// output. Standard miners (escrow burned at emission) contribute none.
    fn service_escrows_of_chain_block(&self, hash: Hash) -> Vec<(Hash, EscrowClaim)> {
        let daa = self.headers_store.get_daa_score(hash).unwrap();
        let ghostdag_data = self.ghostdag_store.get_data(hash).unwrap();
        let non_daa = self.daa_excluded_store.get_mergeset_non_daa(hash).unwrap();
        let txs = self.block_transactions_store.get(hash).unwrap();
        let coinbase = &txs[0];
        let coinbase_id = coinbase.id();
        let mut claims = Vec::new();
        // Walk the coinbase outputs in lockstep with the paid blues: per blue with a subsidy the
        // validated layout is [fee burn?, miner payout, escrow/burn], so the escrow candidate is
        // the output right after the first output at/past the cursor paying the blue's SPK.
        let mut cursor = 0usize;
        for blue in ghostdag_data.mergeset_blues.iter().filter(|b| !non_daa.contains(b)) {
            let blue_txs = self.block_transactions_store.get(*blue).unwrap();
            let blue_coinbase = self.coinbase_manager.deserialize_coinbase_payload(&blue_txs[0].payload).unwrap();
            if blue_coinbase.subsidy == 0 {
                continue;
            }
            let spk = blue_coinbase.miner_data.script_public_key;
            let Some(miner_idx) = (cursor..coinbase.outputs.len()).find(|&i| coinbase.outputs[i].script_public_key == spk)
            else {
                continue;
            };
            let escrow_idx = miner_idx + 1;
            cursor = escrow_idx + 1;
            if let Some(escrow_out) = coinbase.outputs.get(escrow_idx) {
                if escrow_out.value > 0 {
                    if let Some(pubkey) = csv_escrow_pubkey(escrow_out.script_public_key.script()) {
                        claims.push((
                            escrow_miner_key(&pubkey),
                            EscrowClaim {
                                outpoint: TransactionOutpoint::new(coinbase_id, escrow_idx as u32),
                                value: escrow_out.value,
                                daa,
                            },
                        ));
                    }
                }
            }
        }
        claims
    }

    /// Folds one committed chain block into `ledger` and returns its misses. No-op before
    /// `pom_v3_activation` (a per-block property, so the fold is canonical across nodes and IBD).
    /// Misses only become enforceable once finality-deep (see `advance_service_ledger`).
    fn fold_service_chain_block(
        &self,
        ledger: &mut ServiceLedger,
        sc: &impl SelectedChainStoreReader,
        hash: Hash,
        live: bool,
    ) -> Vec<ServiceMiss> {
        let daa = self.headers_store.get_daa_score(hash).unwrap();
        if !self.pom_v3_activation.is_active(daa) {
            return Vec::new();
        }
        let (requests, responses) = self.service_events_of_chain_block(hash);
        let escrows = self.service_escrows_of_chain_block(hash);
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
        ledger.on_chain_block(daa, &requests, &responses, &escrows, |tier| {
            let set = self.service_eligible_miners_in(sc, hash, tier, SERVICE_ELIGIBILITY_WINDOW_DAA);
            // Only the live fold logs: a refold replays the same armings.
            if live {
                info!("service-bond: audit armed at daa {}, tier {}, cohort {}", daa, tier, set.len());
            }
            set
        })
    }

    /// Rebuilds the ledger up to chain index `to` by folding the committed chain from an empty
    /// state — the cold-start and deep-reorg path. Strikes reset at every epoch multiple, so any
    /// fold starting at or before `epoch_start - horizon` reproduces the incremental state: the
    /// horizon margin covers the requests and vault claims still readable at the epoch start.
    fn refold_service_ledger(
        &self,
        sc: &impl SelectedChainStoreReader,
        to: u64,
        pruning_point: Hash,
        cursor_daa: u64,
        queue: &mut std::collections::VecDeque<(u64, u64, ServiceMiss)>,
        logged: &mut std::collections::HashMap<(Hash, [u8; 32], u32), u64>,
    ) -> ServiceLedger {
        let mut ledger = ServiceLedger::default();
        let Ok(to_hash) = sc.get_by_index(to) else {
            return ledger;
        };
        let to_daa = self.headers_store.get_daa_score(to_hash).unwrap();
        // Start at or before `epoch_start - horizon` of the earliest daa whose misses may be
        // re-queued (the burn-store cursor, clamped by `to`): the fold then covers every such
        // miss's full epoch with complete request/vault memory, so replayed misses and the state
        // at `to` both match the incremental fold exactly.
        let anchor = cursor_daa.min(to_daa);
        let daa_bound = (anchor - anchor % SERVICE_STRIKE_EPOCH_DAA).saturating_sub(SERVICE_LEDGER_HORIZON_DAA);
        let pruning_idx = sc.get_by_hash(pruning_point).unwrap_or(0);
        let bottom = self.chain_index_at_or_below_daa(sc, daa_bound, to, pruning_idx).max(pruning_idx);
        for i in (bottom + 1)..=to {
            let hash = sc.get_by_index(i).unwrap();
            let daa = self.headers_store.get_daa_score(hash).unwrap();
            let misses = self.fold_service_chain_block(&mut ledger, sc, hash, false);
            log_new_service_misses(logged, daa, &misses);
            for miss in misses {
                if daa > cursor_daa {
                    queue.push_back((i, daa, miss));
                }
            }
        }
        ledger
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
            let restored = sync.snapshots.get(&common).cloned();
            match restored {
                Some(ledger) => sync.ledger = ledger,
                None => {
                    let cursor = sync.deep_cursor_daa;
                    let mut queue = std::mem::take(&mut sync.queue);
                    queue.clear();
                    let mut logged = std::mem::take(&mut sync.logged);
                    let ledger = self.refold_service_ledger(&*sc, common, pruning_point, cursor, &mut queue, &mut logged);
                    sync.queue = queue;
                    sync.logged = logged;
                    sync.ledger = ledger;
                }
            };
        }
        sync.snapshots.split_off(&(common + 1));
        for (k, h) in chain_path.added.iter().enumerate() {
            let idx = common + 1 + k as u64;
            let daa = self.headers_store.get_daa_score(*h).unwrap();
            let misses = self.fold_service_chain_block(&mut sync.ledger, &*sc, *h, true);
            log_new_service_misses(&mut sync.logged, daa, &misses);
            for miss in misses {
                sync.queue.push_back((idx, daa, miss));
            }
            let snapshot = sync.ledger.clone();
            sync.snapshots.insert(idx, snapshot);
        }
        while sync.snapshots.len() > SERVICE_SNAPSHOT_CAP {
            sync.snapshots.pop_first();
        }
        sync.tip = Some(tip_idx);
        // Bound the logged set by the deepest span a refold can revisit.
        sync.logged.retain(|_, daa| *daa + 2 * SERVICE_LEDGER_HORIZON_DAA > tip_daa);
        // Misses now deeper than finality are reorg-immune on every acceptable POV: persist their
        // burned outpoints and arm the spend rule.
        while sync.queue.front().is_some_and(|(_, daa, _)| daa + self.finality_depth <= tip_daa) {
            let (_, daa, miss) = sync.queue.pop_front().unwrap();
            for claim in miss.burned.iter() {
                let key = crate::model::stores::ai_slash::OutpointKey::new(claim.outpoint.transaction_id, claim.outpoint.index);
                self.service_burn_store.set(key, daa).unwrap();
                self.service_burned.write().insert(claim.outpoint);
            }
            if !miss.burned.is_empty() {
                info!(
                    "service-bond: burn FINAL for miner {} — {} claims, miss daa {}",
                    miss.miner,
                    miss.burned.len(),
                    daa
                );
            }
            // A third strike, now reorg-immune, suspends the miner's production. The deadline is
            // derived from the miss's own daa (deterministic), and the full window bites from
            // finalization: [daa + finality, daa + finality + SERVICE_SUSPENSION_DAA].
            if miss.penalty == ServicePenalty::Suspend {
                let until = daa + self.finality_depth + SERVICE_SUSPENSION_DAA;
                let prev = self.service_suspended.write().insert(miss.miner, until);
                if prev.is_none_or(|p| p < until) {
                    self.service_suspend_store.set(miss.miner, until).unwrap();
                }
                info!("service-bond: SUSPENSION FINAL for miner {} until daa {} (miss daa {})", miss.miner, until, daa);
            }
            sync.deep_cursor_daa = sync.deep_cursor_daa.max(daa);
        }
    }

    /// Boot-time load of the persisted burned outpoints into the RAM set consulted by transaction
    /// validation, of the persisted suspensions into the RAM map consulted by block validation, and
    /// of the deep cursor bounding the cold-start refold.
    pub(crate) fn load_service_burned(&self) {
        let mut set = self.service_burned.write();
        let mut cursor = 0u64;
        for entry in self.service_burn_store.iterator() {
            let (key, daa) = entry.unwrap();
            let tx_id_bytes: [u8; 32] = key[..32].try_into().unwrap();
            let index = u32::from_le_bytes(key[32..36].try_into().unwrap());
            set.insert(TransactionOutpoint::new(tx_id_bytes.into(), index));
            cursor = cursor.max(daa);
        }
        drop(set);
        let mut suspended = self.service_suspended.write();
        for entry in self.service_suspend_store.iterator() {
            let (key, until) = entry.unwrap();
            let miner: [u8; 32] = key[..32].try_into().unwrap();
            suspended.insert(Hash::from_bytes(miner), until);
        }
        drop(suspended);
        self.service_ledger.lock().deep_cursor_daa = cursor;
    }

    /// Whether `producer`'s block at `daa_score` is rejected by a finality-deep suspension. A pure
    /// lookup in the reorg-immune suspended set — cheap and identical on every H6 node.
    pub(super) fn is_producer_suspended(&self, producer: &Hash, daa_score: u64) -> bool {
        self.service_suspended.read().get(producer).is_some_and(|&until| daa_score < until)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
