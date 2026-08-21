use crate::{
    flow_context::{BlockLogEvent, FlowContext, RequestScope},
    flow_trait::Flow,
    flowcontext::orphans::OrphanOutput,
};
use keryx_addresses::Address;
use keryx_consensus::processes::coinbase::RD_ALLOCATION_ADDRESS;
use keryx_consensus_core::{
    api::BlockValidationFutures, block::Block, blockstatus::BlockStatus, config::params::POM_PROOF_SERVE_DEPTH_DAA,
    errors::block::RuleError,
};
use keryx_consensusmanager::{BlockProcessingBatch, ConsensusProxy};
use keryx_core::{debug, info, warn};
use keryx_hashes::Hash;
use keryx_txscript::pay_to_address_script;
use keryx_p2p_lib::{
    IncomingRoute, Router, SharedIncomingRoute,
    common::ProtocolError,
    convert::header::{HeaderFormat, Versioned},
    dequeue, dequeue_with_timeout, make_message, make_request,
    pb::{InvRelayBlockMessage, RequestBlockLocatorMessage, RequestRelayBlocksMessage, kaspad_message::Payload},
};
use keryx_utils::channel::{JobSender, JobTrySendError as TrySendError};
use std::{
    collections::VecDeque,
    sync::Arc,
    time::{Duration, Instant},
};

pub struct RelayInvMessage {
    hash: Hash,

    /// Indicates whether this inv is an orphan root of a previously relayed descendent
    /// (i.e. this inv was indirectly queued)
    is_orphan_root: bool,

    /// Indicates whether this inv is already known to be within orphan resolution range
    known_within_range: bool,
}

/// Encapsulates an incoming invs route which also receives data locally
pub struct TwoWayIncomingRoute {
    incoming_route: SharedIncomingRoute,
    indirect_invs: VecDeque<RelayInvMessage>,
}

impl TwoWayIncomingRoute {
    pub fn new(incoming_route: SharedIncomingRoute) -> Self {
        Self { incoming_route, indirect_invs: VecDeque::new() }
    }

    pub fn enqueue_indirect_invs<I: IntoIterator<Item = Hash>>(&mut self, iter: I, known_within_range: bool) {
        // All indirect invs are orphan roots; not all are known to be within orphan resolution range
        self.indirect_invs.extend(iter.into_iter().map(|h| RelayInvMessage { hash: h, is_orphan_root: true, known_within_range }))
    }

    pub async fn dequeue(&mut self) -> Result<RelayInvMessage, ProtocolError> {
        if let Some(inv) = self.indirect_invs.pop_front() {
            Ok(inv)
        } else {
            let msg = dequeue!(self.incoming_route, Payload::InvRelayBlock)?;
            let inv = msg.try_into()?;
            Ok(RelayInvMessage { hash: inv, is_orphan_root: false, known_within_range: false })
        }
    }
}

/// Number of blocks missing R&D allocation a peer may relay before being banned.
/// Honest nodes may relay a handful of pre-enforcement blocks; spammers relay many more.
const RD_VIOLATION_BAN_THRESHOLD: u32 = 5;

/// Re-proof is best-effort repair traffic, not part of the hot relay path. A naked block can be
/// re-queued when the current peer lacks its proof, so rate-limit attempts per relay flow to avoid
/// repeatedly downloading the same large block on every incoming inv while still rotating the
/// global queue across peers quickly enough to self-heal.
const POM_REPROOF_MIN_INTERVAL: Duration = Duration::from_secs(1);

pub struct HandleRelayInvsFlow {
    ctx: FlowContext,
    router: Arc<Router>,
    /// A route specific for invs messages
    invs_route: TwoWayIncomingRoute,
    /// A route for other messages such as Block and BlockLocator
    msg_route: IncomingRoute,
    /// A channel sender for sending blocks to be handled by the IBD flow (of this peer)
    ibd_sender: JobSender<Block>,
    /// Header format determined by protocol version
    header_format: HeaderFormat,
    /// Counts blocks relayed by this peer that were missing the R&D allocation output.
    rd_violation_count: u32,
    /// Last time this relay flow consumed an item from the global PoM re-proof queue.
    last_reproof_attempt: Instant,
}

#[async_trait::async_trait]
impl Flow for HandleRelayInvsFlow {
    fn router(&self) -> Option<Arc<Router>> {
        Some(self.router.clone())
    }

    async fn start(&mut self) -> Result<(), ProtocolError> {
        match self.start_impl().await {
            Err(e) if e.is_ban_worthy() => {
                let peer_ip = self.router.net_address().ip();
                if self.ctx.ban_peer_automatically(peer_ip).await {
                    warn!("Banned peer {} for ban-worthy protocol violation: {}", self.router, e);
                }
                Err(e)
            }
            res => res,
        }
    }
}

impl HandleRelayInvsFlow {
    pub fn new(
        ctx: FlowContext,
        router: Arc<Router>,
        invs_route: SharedIncomingRoute,
        msg_route: IncomingRoute,
        ibd_sender: JobSender<Block>,
        header_format: HeaderFormat,
    ) -> Self {
        Self {
            ctx,
            router,
            invs_route: TwoWayIncomingRoute::new(invs_route),
            msg_route,
            ibd_sender,
            header_format,
            rd_violation_count: 0,
            last_reproof_attempt: Instant::now() - POM_REPROOF_MIN_INTERVAL,
        }
    }

    async fn start_impl(&mut self) -> Result<(), ProtocolError> {
        loop {
            // Loop over incoming block inv messages
            let inv = self.invs_route.dequeue().await?;

            // Self-healing: re-fetch the possession proof of blocks flagged naked-recent (by the
            // serving guard-rail or the IBD receive path). This is deliberately rate-limited so a
            // proofless peer cannot turn the hot relay path into a large-block download loop. A
            // miss is re-queued at the tail and another peer/flow can try it later.
            if self.last_reproof_attempt.elapsed() >= POM_REPROOF_MIN_INTERVAL {
                self.last_reproof_attempt = Instant::now();
                if let Some(naked_hash) = self.ctx.take_pom_reproof_candidates(1).into_iter().next() {
                    if let Err(err) = self.try_readopt_pom_proof(naked_hash).await {
                        // `take_pom_reproof_candidates` removes the hash from the dedup set. Put it
                        // back before propagating transport/protocol failure so a healthy peer can
                        // still repair it after this flow exits.
                        self.ctx.enqueue_pom_reproof(naked_hash);
                        return Err(err);
                    }
                }
            }

            let session = self.ctx.consensus().unguarded_session();
            let is_ibd_in_transitional_state = session.async_is_consensus_in_transitional_ibd_state().await;

            match session.async_get_block_status(inv.hash).await {
                None | Some(BlockStatus::StatusHeaderOnly) => {} // Continue processing this missing inv
                Some(BlockStatus::StatusInvalid) => {
                    // The peer advertises as part of its chain a block we have proven invalid —
                    // that is a validation verdict, not an ambiguous ending, hence ban-worthy.
                    return Err(ProtocolError::WrongChain(format!("sent inv of an invalid block {}", inv.hash)));
                }
                _ => {
                    // Block is already known, skip to next inv
                    debug!("Relay block {} already exists, continuing...", inv.hash);
                    continue;
                }
            }

            match self.ctx.get_orphan_roots_if_known(&session, inv.hash).await {
                OrphanOutput::Unknown => {}           // Keep processing this inv
                OrphanOutput::NoRoots(_) => continue, // Existing orphan w/o missing roots
                OrphanOutput::Roots(roots) => {
                    // Known orphan with roots to enqueue
                    self.enqueue_orphan_roots(inv.hash, roots, inv.known_within_range);
                    continue;
                }
            }

            if self.ctx.is_ibd_running() && !self.ctx.should_mine(&session).await {
                // Note: If the node is considered nearly synced we continue processing relay blocks even though an IBD is in progress.
                // For instance this means that downloading a side-chain from a delayed node does not interop the normal flow of live blocks.
                debug!("Got relay block {} while in IBD and the node is out of sync, continuing...", inv.hash);
                continue;
            }

            // We keep the request scope alive until consensus processes the block
            let Some((block, request_scope)) = self.request_block(inv.hash, self.msg_route.id(), self.header_format).await? else {
                debug!("Relay block {} was already requested from another peer, continuing...", inv.hash);
                continue;
            };
            request_scope.report_obtained();

            if block.is_header_only() {
                return Err(ProtocolError::OtherOwned(format!("sent header of {} where expected block with body", block.hash())));
            }

            // Pre-validate the coinbase before entering the consensus pipeline.
            // Honest nodes may relay a few pre-enforcement blocks — we drop those silently.
            // Peers that exceed RD_VIOLATION_BAN_THRESHOLD are banned: they are either a
            // malicious miner or a node actively spamming invalid blocks.
            if let Err(reason) = Self::check_relay_coinbase(&block) {
                self.rd_violation_count += 1;
                if self.rd_violation_count >= RD_VIOLATION_BAN_THRESHOLD {
                    let peer_ip = self.router.net_address().ip();
                    if self.ctx.ban_peer_automatically(peer_ip).await {
                        warn!(
                            "Peer {} reached {} R&D violations (last: block {} — {}) — banning",
                            peer_ip,
                            self.rd_violation_count,
                            block.hash(),
                            reason
                        );
                    } else {
                        warn!(
                            "Peer {} reached {} R&D violations (last: block {} — {}) — disconnecting without a ban",
                            peer_ip,
                            self.rd_violation_count,
                            block.hash(),
                            reason
                        );
                    }
                    return Err(ProtocolError::OtherOwned(format!(
                        "peer disconnected after {} blocks missing R&D allocation",
                        self.rd_violation_count
                    )));
                }
                warn!(
                    "Relay block {} has invalid coinbase ({}) — dropping (violation {}/{})",
                    block.hash(), reason, self.rd_violation_count, RD_VIOLATION_BAN_THRESHOLD
                );
                continue;
            }

            let blue_work_threshold = session.async_get_virtual_merge_depth_blue_work_threshold().await;
            // Since `blue_work` respects topology, the negation of this condition means that the relay
            // block is not in the future of virtual's merge depth root, and thus cannot be merged unless
            // other valid blocks Kosherize it (in which case it will be obtained once the merger is relayed)
            let broadcast = block.header.blue_work > blue_work_threshold;

            // We do not apply the skip heuristic below if inv was queued indirectly (as an orphan root), since
            // that means the process started by a proper and relevant relay block
            if !inv.is_orphan_root && !broadcast {
                debug!(
                    "Relay block {} has lower blue work than virtual's merge depth root ({} <= {}), hence we are skipping it",
                    inv.hash, block.header.blue_work, blue_work_threshold
                );
                continue;
            }
            // if in a transitional ibd state, do not wait, sync immediately
            if is_ibd_in_transitional_state {
                self.try_trigger_ibd(block)?;
                continue;
            }

            // Orphan-root recovery is a mini-IBD: roots live arbitrarily deep below the network
            // tip, so their possession proofs may be legitimately GC'ed or stripped (the serving
            // side evaluates retention depth against ITS OWN virtual — a receiver that lags far
            // behind would otherwise demand proofs nobody retains; that mismatch is what wedged
            // the network on 2026-07-24/25). Apply the IBD trust model (accumulated work) to
            // roots and skip possession-proof verification; direct tip relays keep full
            // enforcement, so the possession teeth at the live tip are unchanged.
            //
            // EXCEPT, from the H6 gate on, for roots that are still RECENT: skipping the proof does
            // not merely leave the tier unauthenticated, it leaves the PoW itself unverified (the
            // header-only check folds `pom_final_state`, which is grindable at hash speed without
            // the weights — see `check_pow_and_calc_block_level`), so this path would accept a
            // block nobody paid for. Within `POM_PROOF_SERVE_DEPTH_DAA` every honest node still
            // retains and serves the proof, so demanding it cannot starve a root that really
            // exists. Gated on `pom_v3_activation`: under v3 the proof is what authenticates both
            // the tier and the work, so the stricter reading belongs to that era switch — and the
            // gate lets the policy be disarmed by a version bump rather than a rollback. Two more
            // conditions keep the 07-24/25 wedge closed: we only enforce while NEARLY SYNCED (a
            // lagging node's own virtual says nothing about what the network still retains), and a
            // missing proof is a soft skip below — never a peer disconnect.
            let proof_required = orphan_root_proof_required(
                inv.is_orphan_root,
                self.ctx.config.pom_v3_activation.is_active(block.header.daa_score),
                self.ctx.should_mine(&session).await,
                session.get_virtual_daa_score(),
                block.header.daa_score,
            );
            let BlockValidationFutures { block_task, mut virtual_state_task } = if inv.is_orphan_root && !proof_required {
                session.validate_and_insert_block_ibd(block.clone())
            } else {
                session.validate_and_insert_block(block.clone())
            };

            let ancestor_batch = match block_task.await {
                Ok(_) => Default::default(),
                // A recent orphan root served without its proof: skip it for now and queue the
                // re-fetch. NEVER a disconnect — the peer may simply not hold the proof, and
                // banning peers over a propagation hole is how 2026-07-24/25 became a wedge.
                // `PomProofMissing` never marks the block invalid (body processor), so the block
                // stays retryable and enters as soon as any peer serves it proof-carrying.
                Err(RuleError::PomProofMissing) if proof_required => {
                    self.ctx.enqueue_pom_reproof(inv.hash);
                    debug!("Relay: recent orphan root {} arrived without its possession proof — queued for re-fetch", inv.hash);
                    continue;
                }
                Err(RuleError::MissingParents(missing_parents)) => {
                    debug!("Block {} is orphan and has missing parents: {:?}", block.hash(), missing_parents);
                    if let Some(mut ancestor_batch) = self.process_orphan(&session, block.clone(), inv.known_within_range).await? {
                        // Block is not an orphan, retrying with the exact proof policy of the first attempt.
                        let BlockValidationFutures { block_task: block_task_inner, virtual_state_task: virtual_state_task_inner } =
                            if inv.is_orphan_root && !proof_required {
                                session.validate_and_insert_block_ibd(block.clone())
                            } else {
                                session.validate_and_insert_block(block.clone())
                            };
                        virtual_state_task = virtual_state_task_inner;
                        for block_task in ancestor_batch.block_tasks.take().unwrap() {
                            match block_task.await {
                                Ok(_) => {}
                                // We disconnect on invalidness even though this is not a direct relay from this peer, because
                                // current relay is a descendant of this block (i.e. this peer claims all its ancestors are valid)
                                Err(rule_error) => return Err(rule_error.into()),
                            }
                        }

                        match block_task_inner.await {
                            Ok(_) => match ancestor_batch.blocks.len() {
                                0 => debug!("Retried orphan block {} successfully", block.hash()),
                                n => {
                                    self.ctx.log_block_event(BlockLogEvent::Unorphaned(ancestor_batch.blocks[0].hash(), n));
                                    debug!("Unorphaned {} ancestors and retried orphan block {} successfully", n, block.hash())
                                }
                            },
                            // Same soft skip as the first attempt (see above).
                            Err(RuleError::PomProofMissing) if proof_required => {
                                self.ctx.enqueue_pom_reproof(inv.hash);
                                debug!("Relay: retried orphan root {} still proofless — queued for re-fetch", inv.hash);
                                continue;
                            }
                            Err(rule_error) => return Err(rule_error.into()),
                        }
                        ancestor_batch
                    } else {
                        continue;
                    }
                }
                Err(rule_error) => return Err(rule_error.into()),
            };

            // As a policy, we only relay blocks who stand a chance to enter past(virtual).
            // The only mining rule which permanently excludes a block is the merge depth bound
            // (as opposed to "max parents" and "mergeset size limit" rules)
            if broadcast {
                let msgs = ancestor_batch
                    .blocks
                    .iter()
                    .map(|b| make_message!(Payload::InvRelayBlock, InvRelayBlockMessage { hash: Some(b.hash().into()) }))
                    .collect();
                // we filter out the current peer to avoid sending it back invs we know it already has
                self.ctx.hub().broadcast_many(msgs, Some(self.router.key())).await;

                // we filter out the current peer to avoid sending it back the same invs
                self.ctx
                    .hub()
                    .broadcast(
                        make_message!(Payload::InvRelayBlock, InvRelayBlockMessage { hash: Some(inv.hash.into()) }),
                        Some(self.router.key()),
                    )
                    .await;
            }

            // We spawn post-processing as a separate task so that this loop
            // can continue processing the following relay blocks
            let ctx = self.ctx.clone();
            tokio::spawn(async move {
                ctx.on_new_block(&session, ancestor_batch, block, virtual_state_task).await;
                ctx.log_block_event(BlockLogEvent::Relay(inv.hash));
            });
        }
    }

    fn enqueue_orphan_roots(&mut self, _orphan: Hash, roots: Vec<Hash>, known_within_range: bool) {
        self.invs_route.enqueue_indirect_invs(roots, known_within_range)
    }

    /// Re-fetches a block whose possession proof is missing locally and adopts the proof it
    /// carries. A miss is re-queued at the tail of the global repair queue so another peer can
    /// provide it later; the caller rate-limits attempts to keep repair traffic out of the hot path.
    ///
    /// Two cases: the block is stored but naked (graft the proof onto it), or it was never
    /// inserted because the proof-required relay path skipped it (submit the proof-carrying block
    /// through the enforcing path — there is no stored header to graft onto).
    async fn try_readopt_pom_proof(&mut self, requested_hash: Hash) -> Result<(), ProtocolError> {
        let Some((block, request_scope)) = self.request_block(requested_hash, self.msg_route.id(), self.header_format).await? else {
            // Another flow currently owns this request. Re-queue rather than silently losing the
            // repair candidate when the request scope is released.
            self.ctx.enqueue_pom_reproof(requested_hash);
            return Ok(());
        };
        request_scope.report_obtained();
        let Some(proof) = block.pom_proof else {
            self.ctx.enqueue_pom_reproof(requested_hash);
            debug!("PoM re-proof: peer {} also serves {} without its proof — re-queued for another peer", self.router, requested_hash);
            return Ok(());
        };
        let session = self.ctx.consensus().unguarded_session();
        // Adoption grafts the proof onto a block we already store. A block skipped by the
        // proof-required relay path was never inserted, so there is no stored header to graft
        // onto — submit the proof-carrying block itself through the enforcing path instead.
        if session.async_get_block_status(requested_hash).await.is_none() {
            let block =
                Block { header: block.header, transactions: block.transactions, pom_proof: Some(proof), pom_tier: block.pom_tier };
            match session.validate_and_insert_block(block).block_task.await {
                Ok(_) => info!("PoM re-proof: inserted {} with the proof served by peer {}", requested_hash, self.router),
                Err(e) => {
                    self.ctx.enqueue_pom_reproof(requested_hash);
                    debug!(
                        "PoM re-proof: proof-carrying {} from peer {} still rejected: {} — re-queued for another peer",
                        requested_hash, self.router, e
                    );
                }
            }
            return Ok(());
        }
        match session.async_adopt_pom_proof(requested_hash, (*proof).clone()).await {
            Ok(true) => info!("PoM re-proof: adopted the possession proof of {} from peer {}", requested_hash, self.router),
            Ok(false) => {}
            Err(e) => {
                self.ctx.enqueue_pom_reproof(requested_hash);
                debug!("PoM re-proof: proof of {} from peer {} not adopted: {} — re-queued", requested_hash, self.router, e);
            }
        }
        Ok(())
    }

    async fn request_block(
        &mut self,
        requested_hash: Hash,
        request_id: u32,
        header_format: HeaderFormat,
    ) -> Result<Option<(Block, RequestScope<Hash>)>, ProtocolError> {
        // Note: the request scope is returned and should be captured until block processing is completed
        let Some(request_scope) = self.ctx.try_adding_block_request(requested_hash) else {
            return Ok(None);
        };
        self.router
            .enqueue(make_request!(
                Payload::RequestRelayBlocks,
                RequestRelayBlocksMessage { hashes: vec![requested_hash.into()] },
                request_id
            ))
            .await?;
        let msg = dequeue_with_timeout!(self.msg_route, Payload::Block)?;
        let block: Block = Versioned(header_format, msg).try_into()?;
        if block.hash() != requested_hash {
            Err(ProtocolError::OtherOwned(format!("requested block hash {} but got block {}", requested_hash, block.hash())))
        } else {
            Ok(Some((block, request_scope)))
        }
    }

    /// Process the orphan block. Returns `Some(BlockProcessingBatch)` if the block has no missing roots, where
    /// the batch includes ancestor blocks and their consensus processing batch. This indicates a retry is recommended.
    async fn process_orphan(
        &mut self,
        consensus: &ConsensusProxy,
        block: Block,
        mut known_within_range: bool,
    ) -> Result<Option<BlockProcessingBatch>, ProtocolError> {
        // Return if the block has been orphaned from elsewhere already
        if self.ctx.is_known_orphan(block.hash()).await {
            return Ok(None);
        }

        /* We orphan a block if one of the following holds:
                1. It is known to be within orphan resolution range (no-op)
                2. It holds the IBD DAA score heuristic conditions (local op)
                3. We resolve its orphan range by interacting with the peer (peer op)

            Note that we check the conditions by the order of their cost and avoid making expensive calls if not needed.
        */
        let should_orphan = known_within_range || self.check_orphan_ibd_conditions(block.header.daa_score) || {
            // Inner scope to evaluate orphan resolution range and reassign the `known_within_range` variable
            known_within_range = self.check_orphan_resolution_range(consensus, block.hash(), self.msg_route.id()).await?;
            known_within_range
        };

        if should_orphan {
            let hash = block.hash();
            match self.ctx.add_orphan(consensus, block).await {
                // There is a sync gap between consensus and the orphan pool, meaning that consensus might have indicated
                // that this block is orphan, but by the time it got to the orphan pool we discovered it no longer has missing roots.
                // In such a case, the orphan pool will queue the known orphan ancestors to consensus and will return the block processing
                // batch.
                // We signal this to the caller by returning the batch of processed ancestors, indicating a consensus processing retry
                // should be performed for this block as well.
                Some(OrphanOutput::NoRoots(ancestor_batch)) => {
                    return Ok(Some(ancestor_batch));
                }
                Some(OrphanOutput::Roots(roots)) => {
                    self.ctx.log_block_event(BlockLogEvent::Orphaned(hash, roots.len()));
                    self.enqueue_orphan_roots(hash, roots, known_within_range)
                }
                None | Some(OrphanOutput::Unknown) => {}
            }
        } else {
            self.try_trigger_ibd(block)?;
        }
        Ok(None)
    }

    /// Applies an heuristic to check whether we should store the orphan block in the orphan pool for IBD considerations.
    ///
    /// When IBD is going on it is guaranteed to sync all blocks in past(R) where R is the relay block triggering the
    /// IBD. Frequently, if the IBD is short and fast enough, R will be within short distance from the syncer tips once
    /// the IBD is over. However antipast(R) is usually not in orphan resolution range so these blocks will not be kept
    /// leading to another IBD and so on.
    ///
    /// By checking whether the current orphan DAA score is within the range (R - M/10, R + M/2)** we make sure that in this
    /// case we keep ~M/2 blocks in the orphan pool which are all unorphaned when IBD completes (see revalidate_orphans),
    /// and the node reaches full sync state asap. We use M/10 for the lower bound since we only want to cover anticone(R)
    /// in that region (which is expectedly small), whereas the M/2 upper bound is for covering the most early segment in
    /// future(R). Overall we avoid keeping more than ~M/2 in order to not enter the area where blocks start getting evicted
    /// from the orphan pool.
    ///
    /// **where R is the DAA score of R, and M is the orphans pool size limit
    fn check_orphan_ibd_conditions(&self, orphan_daa_score: u64) -> bool {
        if let Some(ibd_daa_score) = self.ctx.ibd_relay_daa_score() {
            let max_orphans = self.ctx.max_orphans() as u64;
            orphan_daa_score + max_orphans / 10 > ibd_daa_score && orphan_daa_score < ibd_daa_score + max_orphans / 2
        } else {
            false
        }
    }

    /// Checks whether the given block hash is within orphan resolution range. This method sends a BlockLocator
    /// request to the peer with a limit of `ctx.orphan_resolution_range`. In the response, if we know one of the
    /// hashes, we should retrieve the given block via unorphaning.
    async fn check_orphan_resolution_range(
        &mut self,
        consensus: &ConsensusProxy,
        hash: Hash,
        request_id: u32,
    ) -> Result<bool, ProtocolError> {
        self.router
            .enqueue(make_request!(
                Payload::RequestBlockLocator,
                RequestBlockLocatorMessage { high_hash: Some(hash.into()), limit: self.ctx.orphan_resolution_range() },
                request_id
            ))
            .await?;
        let msg = dequeue_with_timeout!(self.msg_route, Payload::BlockLocator)?;
        let locator_hashes: Vec<Hash> = msg.try_into()?;
        // Locator hashes are sent from later to earlier, so it makes sense to query consensus in reverse. Technically
        // with current syncer-side implementations (in both go-kaspa and this codebase) we could query only the last one,
        // but we prefer not relying on such details for correctness
        //
        // The current syncer-side implementation sends a full locator even though it suffices to only send the
        // most early block. We keep it this way in order to allow future syncee-side implementations to do more
        // with the full incremental info and because it is only a small set of hashes.
        for h in locator_hashes.into_iter().rev() {
            if consensus.async_get_block_status(h).await.is_some_and(|s| s.has_block_body()) {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Lightweight pre-validation of a relay block's coinbase.
    ///
    /// Checks that the coinbase transaction contains an R&D allocation output before
    /// submitting the block to the full consensus pipeline. This catches miners running
    /// outdated software (without the R&D allocation) early, preventing their blocks from
    /// being relayed across the network and wasting validation resources.
    ///
    /// Returns `Ok(())` if the coinbase looks valid, or `Err(reason)` if it is obviously wrong.
    fn check_relay_coinbase(block: &Block) -> Result<(), &'static str> {
        let coinbase = block.transactions.first().ok_or("block has no transactions")?;
        // Zero outputs means zero-reward block (all blues had subsidy=0) — no R&D cut expected.
        if coinbase.outputs.is_empty() {
            return Ok(());
        }
        let rd_address = Address::try_from(RD_ALLOCATION_ADDRESS).map_err(|_| "invalid R&D address constant")?;
        let rd_script = pay_to_address_script(&rd_address);
        if coinbase.outputs.iter().any(|o| o.script_public_key == rd_script) {
            Ok(())
        } else {
            Err("missing R&D allocation output")
        }
    }

    // Send the block to IBD flow via the dedicated job channel. If the channel has a pending job, we prefer
    // the block with higher blue work, since it is usually more recent
    fn try_trigger_ibd(&self, block: Block) -> Result<(), ProtocolError> {
        match self.ibd_sender.try_send(block.clone(), |b, c| if b.header.blue_work > c.header.blue_work { b } else { c }) {
            Ok(_) | Err(TrySendError::Full(_)) => Ok(()),
            Err(TrySendError::Closed(_)) => Err(ProtocolError::ConnectionClosed), // This indicates that IBD flow has exited
        }
    }
}

/// Whether a relayed block must carry a verified possession proof to be inserted.
///
/// Orphan roots normally take the IBD trust model (accumulated work) because they can sit
/// arbitrarily deep, where proofs are legitimately GC'ed. But skipping the proof also skips the
/// only real PoW check — the header-only fold of `pom_final_state` is grindable at hash speed
/// without holding the weights — so from the H6 gate on, a recent root must not enter that way.
///
/// `h6_active` is `pom_v3_activation` at the block's DAA score: v3 is the era where the proof
/// authenticates the tier and the work, and gating here keeps the switch coordinated and
/// reversible by version bump.
///
/// The two guards below are what keep this from re-creating the 2026-07-24/25 wedge:
/// * `nearly_synced`: a node that lags has no idea what the network still retains; its own
///   virtual DAA is a meaningless yardstick, so it keeps the permissive path.
/// * depth `<= POM_PROOF_SERVE_DEPTH_DAA`: inside the service window every honest node still
///   holds and serves the proof, so demanding it cannot starve a root that genuinely exists.
///
/// A `true` verdict never bans a peer: the caller treats a missing proof as a soft skip plus a
/// re-fetch request (`PomProofMissing` does not mark the block invalid, so it stays retryable).
fn orphan_root_proof_required(
    is_orphan_root: bool,
    h6_active: bool,
    nearly_synced: bool,
    virtual_daa: u64,
    block_daa: u64,
) -> bool {
    is_orphan_root && h6_active && nearly_synced && virtual_daa.saturating_sub(block_daa) <= POM_PROOF_SERVE_DEPTH_DAA
}

#[cfg(test)]
mod tests {
    use super::*;

    const TIP: u64 = 1_000_000;

    #[test]
    fn recent_orphan_root_requires_its_proof() {
        // Inside the service window while synced: the proof exists network-wide, demand it.
        assert!(orphan_root_proof_required(true, true, true, TIP, TIP - 10));
        assert!(orphan_root_proof_required(true, true, true, TIP, TIP - POM_PROOF_SERVE_DEPTH_DAA));
    }

    #[test]
    fn old_orphan_root_stays_exempt() {
        // Past the service window the proof is legitimately gone — demanding it would wedge us.
        assert!(!orphan_root_proof_required(true, true, true, TIP, TIP - POM_PROOF_SERVE_DEPTH_DAA - 1));
        assert!(!orphan_root_proof_required(true, true, true, TIP, 0));
    }

    #[test]
    fn lagging_node_never_demands() {
        // The 2026-07-24/25 wedge shape: a far-behind receiver sees a root that is AHEAD of its
        // own virtual (depth saturates to 0, i.e. "recent") while the network has long GC'ed the
        // proof. Not being nearly synced must keep such a node on the permissive path.
        assert!(!orphan_root_proof_required(true, true, false, 1_000, 500_000));
        assert!(!orphan_root_proof_required(true, true, false, TIP, TIP - 10));
    }

    #[test]
    fn direct_relays_and_pre_h6_blocks_are_unaffected() {
        // Direct relay already goes through the enforcing path unconditionally — this predicate
        // only ever upgrades orphan roots, never downgrades anything.
        assert!(!orphan_root_proof_required(false, true, true, TIP, TIP - 10));
        // Before the H6 gate the permissive orphan-root path is kept verbatim, so shipping this
        // release changes nothing on a network that has not crossed the fork.
        assert!(!orphan_root_proof_required(true, false, true, TIP, TIP - 10));
    }
}
