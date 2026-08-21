use crate::{
    flow_context::FlowContext,
    flow_trait::Flow,
    ibd::{HeadersChunkStream, TrustedEntryStream, negotiate::ChainNegotiationOutput},
};
use futures::future::{Either, join_all, select, try_join_all};
use itertools::Itertools;
use keryx_consensus_core::{
    BlockHashSet,
    api::BlockValidationFuture,
    block::Block,
    config::params::POM_PROOF_SERVE_DEPTH_DAA,
    errors::consensus::ConsensusResult,
    header::Header,
    pom::PomProof,
    pruning::{PruningPointProof, PruningPointsList, PruningProofMetadata},
    trusted::{ExternalGhostdagData, TrustedBlock},
    tx::Transaction,
};
use keryx_consensusmanager::{ConsensusProxy, StagingConsensus, spawn_blocking};
use keryx_core::{debug, info, time::unix_now, warn};
use keryx_hashes::Hash;
use keryx_muhash::MuHash;
use keryx_p2p_lib::{
    IncomingRoute, Router,
    common::ProtocolError,
    convert::{
        header::{HeaderFormat, Versioned},
        model::trusted::TrustedDataPackage,
    },
    dequeue_with_timeout, make_message, make_request,
    pb::{
        RequestAntipastMessage, RequestBlockBodiesMessage, RequestHeadersMessage, RequestIbdBlocksMessage,
        RequestPruningPointAndItsAnticoneMessage, RequestPruningPointProofMessage, RequestPruningPointUtxoSetMessage,
        RequestServiceStateMessage, kaspad_message::Payload,
    },
};
use keryx_utils::channel::JobReceiver;
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::time::sleep;

use super::{HeadersChunk, IBD_BATCH_SIZE, PruningPointUtxosetChunkStream, progress::ProgressReporter};
type BlockBody = Vec<Transaction>;

/// Event daa of a canonical service-state row, `None` for a malformed one. Mirrors the row
/// layouts in `service_commit`.
fn service_row_daa(row: &[u8]) -> Option<u64> {
    match (*row.first()?, row.len()) {
        (0x01, 45) => Some(u64::from_le_bytes(row[37..45].try_into().unwrap())),
        (0x02, 53) => Some(u64::from_le_bytes(row[1..9].try_into().unwrap())),
        (0x03, 41) => Some(u64::from_le_bytes(row[33..41].try_into().unwrap())),
        (0x04, n) if n >= 85 => Some(u64::from_le_bytes(row[73..81].try_into().unwrap())),
        _ => None,
    }
}

/// Flow for managing IBD - Initial Block Download
pub struct IbdFlow {
    pub(super) ctx: FlowContext,
    pub(super) router: Arc<Router>,
    pub(super) incoming_route: IncomingRoute,
    pub(super) body_only_ibd_permitted: bool,
    header_format: HeaderFormat,
    protocol_version: u32,

    // Receives relay blocks from relay flow which are out of orphan resolution range and hence trigger IBD
    relay_receiver: JobReceiver<Block>,
}

#[async_trait::async_trait]
impl Flow for IbdFlow {
    fn router(&self) -> Option<Arc<Router>> {
        Some(self.router.clone())
    }

    async fn start(&mut self) -> Result<(), ProtocolError> {
        self.start_impl().await
    }
}

pub enum IbdType {
    Sync { highest_known_syncer_chain_hash: Hash, is_utxo_stable: bool, is_pp_anticone_synced: bool },
    DownloadHeadersProof,
    PruningCatchUp { highest_known_syncer_chain_hash: Hash },
}

struct QueueChunkOutput {
    jobs: Vec<BlockValidationFuture>,
    daa_score: u64,
    timestamp: u64,
}

impl IbdFlow {
    pub fn new(
        ctx: FlowContext,
        router: Arc<Router>,
        incoming_route: IncomingRoute,
        relay_receiver: JobReceiver<Block>,
        body_only_ibd_permitted: bool,
        header_format: HeaderFormat,
        protocol_version: u32,
    ) -> Self {
        Self { ctx, router, incoming_route, relay_receiver, body_only_ibd_permitted, header_format, protocol_version }
    }

    /// Fetch a set of headers with a single blocking-runtime transition. IBD already knows the
    /// exact hash order, so callers can zip the returned headers with the requested bodies.
    async fn get_headers_batch(consensus: &ConsensusProxy, hashes: &[Hash]) -> Result<Vec<Arc<Header>>, ProtocolError> {
        let hashes = hashes.to_vec();
        Ok(consensus
            .clone()
            .spawn_blocking(move |c| hashes.into_iter().map(|hash| c.get_header(hash)).collect::<ConsensusResult<Vec<_>>>())
            .await?)
    }

    /// Fetch the already-validated header and GhostDAG metadata for trusted body recovery in one
    /// consensus task. This replaces two spawn_blocking round trips per block with one per chunk.
    async fn get_trusted_metadata_batch(
        consensus: &ConsensusProxy,
        hashes: &[Hash],
    ) -> Result<Vec<(Arc<Header>, ExternalGhostdagData)>, ProtocolError> {
        let hashes = hashes.to_vec();
        Ok(consensus
            .clone()
            .spawn_blocking(move |c| {
                hashes
                    .into_iter()
                    .map(|hash| Ok((c.get_header(hash)?, c.get_ghostdag_data(hash)?)))
                    .collect::<ConsensusResult<Vec<_>>>()
            })
            .await?)
    }

    /// Full-block trusted recovery already receives headers from the peer, so only GhostDAG data
    /// needs a local batch lookup.
    async fn get_ghostdag_batch(
        consensus: &ConsensusProxy,
        hashes: &[Hash],
    ) -> Result<Vec<ExternalGhostdagData>, ProtocolError> {
        let hashes = hashes.to_vec();
        Ok(consensus
            .clone()
            .spawn_blocking(move |c| hashes.into_iter().map(|hash| c.get_ghostdag_data(hash)).collect::<ConsensusResult<Vec<_>>>())
            .await?)
    }

    async fn start_impl(&mut self) -> Result<(), ProtocolError> {
        while let Ok(relay_block) = self.relay_receiver.recv().await {
            if let Some(_guard) = self.ctx.try_set_ibd_running(self.router.key(), relay_block.header.daa_score) {
                info!("IBD started with peer {}", self.router);

                match self.ibd(relay_block).await {
                    Ok(_) => info!("IBD with peer {} completed successfully", self.router),
                    Err(e) => {
                        info!("IBD with peer {} completed with error: {}", self.router, e);
                        if e.is_ban_worthy() {
                            let peer_ip = self.router.net_address().ip();
                            if self.ctx.ban_peer_automatically(peer_ip).await {
                                warn!("Banned peer {} for ban-worthy protocol violation: {}", self.router, e);
                            }
                        }
                        return Err(e);
                    }
                }
            }
        }

        Ok(())
    }

    async fn ibd(&mut self, relay_block: Block) -> Result<(), ProtocolError> {
        let mut session = self.ctx.consensus().session().await;

        let negotiation_output = self.negotiate_missing_syncer_chain_segment(&session).await?;
        let ibd_type = self
            .determine_ibd_type(
                &session,
                &relay_block.header,
                negotiation_output.highest_known_syncer_chain_hash,
                negotiation_output.syncer_pruning_point,
            )
            .await?;
        // Body-sync target: normally the syncer's sink, but the highest header below the sync ceiling
        // when one is set (the Sync path updates it from `sync_headers`).
        let mut body_target = negotiation_output.syncer_virtual_selected_parent;
        match ibd_type {
            IbdType::Sync { highest_known_syncer_chain_hash, is_utxo_stable, is_pp_anticone_synced } => {
                let pruning_point = session.async_pruning_point().await;

                info!("syncing ahead from current pruning point");
                // Following IBD catchup a new pruning point is designated and finalized in consensus. Blocks from its anticone (including itself)
                // have undergone normal header verification, but contain no body yet. Processing of new blocks in the pruning point's future cannot proceed
                // since these blocks' parents are missing block data.
                // Hence we explicitly process bodies of the currently body missing anticone blocks as trusted blocks
                // Notice that this is degenerate following sync_with_headers_proof
                // but not necessarily so after sync_headers -
                // as it might sync following a previous pruning_catch_up that crashed before this stage concluded
                if !is_pp_anticone_synced {
                    self.sync_missing_trusted_bodies(&session).await?;
                }
                if !is_utxo_stable
                // Utxo might not be available even if the pruning point block data is.
                // Utxo must be synced before all so the node could function
                {
                    info!(
                        "utxoset corresponding to the current pruning point is incomplete, attempting to download it from {}",
                        self.router
                    );

                    self.sync_new_utxo_set(&session, pruning_point, &relay_block.header).await?;
                }
                // Once utxo is valid, simply sync missing headers
                body_target = self
                    .sync_headers(
                        &session,
                        negotiation_output.syncer_virtual_selected_parent,
                        highest_known_syncer_chain_hash,
                        &relay_block,
                    )
                    .await?;
            }
            IbdType::DownloadHeadersProof => {
                drop(session); // Avoid holding the previous consensus throughout the staging IBD
                let staging = self.ctx.consensus_manager.new_staging_consensus();
                match self.ibd_with_headers_proof(&staging, negotiation_output.syncer_virtual_selected_parent, &relay_block).await {
                    Ok(()) => {
                        spawn_blocking(|| staging.commit()).await.unwrap();
                        info!(
                            "Header download stage of IBD with headers proof completed successfully from {}. Committed staging consensus.",
                            self.router
                        );

                        // This will reobtain the freshly committed staging consensus
                        session = self.ctx.consensus().session().await;
                        // Next, sync a utxoset corresponding to the new pruning point from the syncer.
                        // Note that the new pruning point's anticone need not be downloaded separately as in other IBD types
                        // as it was just downloaded as part of the headers proof.
                        self.sync_new_utxo_set(&session, negotiation_output.syncer_pruning_point, &relay_block.header).await?;
                    }
                    Err(e) => {
                        warn!("IBD with headers proof from {} was unsuccessful ({})", self.router, e);
                        staging.cancel();
                        return Err(e);
                    }
                }
            }
            IbdType::PruningCatchUp { highest_known_syncer_chain_hash } => {
                info!("catching up to new pruning point {} ", negotiation_output.syncer_pruning_point);
                match self.pruning_point_catchup(&session, &negotiation_output, &relay_block, highest_known_syncer_chain_hash).await {
                    Ok(()) => {
                        info!("header stage of pruning catchup from peer {} completed", self.router);
                        self.sync_missing_trusted_bodies(&session).await?;
                        self.sync_new_utxo_set(&session, negotiation_output.syncer_pruning_point, &relay_block.header).await?;
                        // Note that pruning of old data will only occur once virtual has caught up sufficiently far
                    }

                    Err(e) => {
                        warn!("IBD catchup from peer {} was unsuccessful ({})", self.router, e);
                        return Err(e);
                    }
                }
            }
        }

        // Sync missing bodies in the past of the (possibly ceiling-capped) sync target
        self.sync_missing_block_bodies(&session, body_target).await?;

        // Relay block might be in the antipast of syncer sink, thus check its past for missing bodies
        // as well — but skip it under a sync ceiling (the relay block is the corrupted tip above it).
        if self.sync_ceiling().is_none() {
            self.sync_missing_block_bodies(&session, relay_block.hash()).await?;
        }

        // Following IBD we revalidate orphans since many of them might have been processed during the IBD
        // or are now processable
        let (queued_hashes, virtual_processing_tasks) = self.ctx.revalidate_orphans(&session).await;
        let mut unorphaned_hashes = Vec::with_capacity(queued_hashes.len());
        let results = join_all(virtual_processing_tasks).await;
        for (hash, result) in queued_hashes.into_iter().zip(results) {
            match result {
                Ok(_) => unorphaned_hashes.push(hash),
                // We do not return the error and disconnect here since we don't know
                // that this peer was the origin of the orphan block
                Err(e) => warn!("Validation failed for orphan block {}: {}", hash, e),
            }
        }
        match unorphaned_hashes.len() {
            0 => {}
            n => info!("IBD post processing: unorphaned {} blocks ...{}", n, unorphaned_hashes.last().unwrap()),
        }

        Ok(())
    }

    async fn determine_ibd_type(
        &self,
        consensus: &ConsensusProxy,
        relay_header: &Header,
        highest_known_syncer_chain_hash: Option<Hash>,
        syncer_pruning_point: Hash,
    ) -> Result<IbdType, ProtocolError> {
        if let Some(highest_known_syncer_chain_hash) = highest_known_syncer_chain_hash {
            let pruning_point = consensus.async_pruning_point().await;
            let sink = consensus.async_get_sink().await;
            info!("current sink is:{}", sink);
            info!("current pruning point is:{}", pruning_point);
            if consensus.async_is_chain_ancestor_of(pruning_point, highest_known_syncer_chain_hash).await? {
                /// Categorizes the syncer's pruning point position relative to local
                enum SyncerSkew {
                    Lagging,
                    Aligned,
                    Leading,
                }

                let syncer_skew = if syncer_pruning_point == pruning_point {
                    SyncerSkew::Aligned
                } else if consensus.async_is_chain_ancestor_of(pruning_point, syncer_pruning_point).await.unwrap_or(false) {
                    SyncerSkew::Leading
                } else if consensus.async_get_n_last_pruning_points(4 /*syncer lag tolerance*/).await.contains(&syncer_pruning_point) {
                    SyncerSkew::Lagging
                } else {
                    return Err(ProtocolError::Other(
                        "The syncer purports to have data in the recent future but their pruning point could not be easily recognized",
                    ));
                };

                let is_utxo_stable = consensus.async_is_pruning_utxoset_stable().await;
                let is_pp_anticone_synced = consensus.async_is_pruning_point_anticone_fully_synced().await;

                return match (syncer_skew, is_utxo_stable && is_pp_anticone_synced) {
                    (SyncerSkew::Aligned, _) => {
                        Ok(IbdType::Sync { highest_known_syncer_chain_hash, is_utxo_stable, is_pp_anticone_synced })
                    }
                    (SyncerSkew::Lagging, true) => {
                        Ok(IbdType::Sync { highest_known_syncer_chain_hash, is_utxo_stable, is_pp_anticone_synced })
                    }
                    (SyncerSkew::Lagging, false) => Err(ProtocolError::Other(
                        "Local node is in a transitional state requiring external data to stabilize, but the syncer lags behind and is unable to provide said data",
                    )),
                    (SyncerSkew::Leading, true) => {
                        if consensus.async_get_block_status(syncer_pruning_point).await.is_some_and(|b| b.has_block_body()) {
                            // While a leading syncer skew often indicates the need for catchup, in this case
                            // the node is just missing a segment in the future of its current pruning point, that is available to the syncer
                            Ok(IbdType::Sync { highest_known_syncer_chain_hash, is_utxo_stable, is_pp_anticone_synced })
                        } else {
                            Ok(IbdType::PruningCatchUp { highest_known_syncer_chain_hash })
                        }
                    }
                    (SyncerSkew::Leading, false) => Ok(IbdType::PruningCatchUp { highest_known_syncer_chain_hash }),
                };
            }

            // If the pruning point is not in the chain of `highest_known_syncer_chain_hash`, it
            // means it's in its antichain (because if `highest_known_syncer_chain_hash` was in
            // the pruning point's past the pruning point itself would be
            // `highest_known_syncer_chain_hash`). So it means there's a finality conflict.
            //
            let peer_ip = self.router.net_address().ip();
            if self.ctx.ban_peer_automatically(peer_ip).await {
                warn!("Banned peer {} for finality conflict with local pruning point", self.router);
            }
            return Err(ProtocolError::Other("peer is in a finality conflict with the local pruning point"));
        }

        // Option B (KERYX_TRUST_SYNC_FROM_TIP): no shared chain block was found because our tip is
        // below the trusted peer's pruning point. The standard fallback below is headers-proof IBD,
        // which is broken across the PoM/diff-reset hardfork. Instead, trust the `--connect`'d
        // archival peer and catch up forward from our own sink (it serves the blocks below its
        // pruning point). See `trust_sync_from_tip` — gated by env, unsafe for public P2P.
        if self.trust_sync_from_tip() {
            let local_sink = consensus.async_get_sink().await;
            let is_utxo_stable = consensus.async_is_pruning_utxoset_stable().await;
            let is_pp_anticone_synced = consensus.async_is_pruning_point_anticone_fully_synced().await;
            warn!(
                "KERYX_TRUST_SYNC_FROM_TIP: no shared chain block with {} (our tip is below its pruning point) — trusting peer, catching up forward from local sink {}",
                self.router, local_sink
            );
            return Ok(IbdType::Sync { highest_known_syncer_chain_hash: local_sink, is_utxo_stable, is_pp_anticone_synced });
        }

        let hst_header = consensus.async_get_header(consensus.async_get_headers_selected_tip().await).await.unwrap();
        let pruning_depth = self.ctx.config.pruning_depth();
        if relay_header.blue_score >= hst_header.blue_score + pruning_depth && relay_header.blue_work > hst_header.blue_work {
            let finality_duration_in_milliseconds = self.ctx.config.finality_duration_in_milliseconds();
            if unix_now() > consensus.async_creation_timestamp().await + finality_duration_in_milliseconds {
                let fp = consensus.async_finality_point().await;
                let fp_ts = consensus.async_get_header(fp).await?.timestamp;
                if unix_now() < fp_ts + finality_duration_in_milliseconds * 3 / 2 {
                    // We reject the headers proof if the node has a relatively up-to-date finality point and current
                    // consensus has matured for long enough (and not recently synced). This is mostly a spam-protector
                    // since subsequent checks identify these violations as well
                    let peer_ip = self.router.net_address().ip();
                    if self.ctx.ban_peer_automatically(peer_ip).await {
                        warn!(
                            "Banned peer {} for IBD spam (peer has no known block while local consensus is up to date)",
                            self.router
                        );
                    }
                    return Err(ProtocolError::Other(
                        "peer has no known block but local consensus appears to be up to date, this is most likely a spam attempt",
                    ));
                }
            }

            // The relayed block has sufficient blue score and blue work over the current header selected tip
            Ok(IbdType::DownloadHeadersProof)
        } else {
            Err(ProtocolError::Other("peer has no known block but conditions for requesting headers proof are not met"))
        }
    }

    /// This function is triggered when the syncer's pruning point is higher
    /// than ours and we already processed its header before.
    /// so we only need to sync more headers and set it to our new pruning point before proceeding with IBD
    async fn pruning_point_catchup(
        &mut self,
        consensus: &ConsensusProxy,
        negotiation_output: &ChainNegotiationOutput,
        relay_block: &Block,
        highest_known_syncer_chain_hash: Hash,
    ) -> Result<(), ProtocolError> {
        // Before attempting to update to the syncer's pruning point, sync to the latest headers of the syncer,
        // to ensure that we will locally have sufficient headers on top of the syncer's pruning point
        let syncer_pp = negotiation_output.syncer_pruning_point;
        let syncer_sink = negotiation_output.syncer_virtual_selected_parent;
        self.sync_headers(consensus, syncer_sink, highest_known_syncer_chain_hash, relay_block).await?;

        // This function's main effect is to confirm the syncer's pruning point can be finalized into the consensus, and to update
        // all the relevant stores
        consensus.async_intrusive_pruning_point_update(syncer_pp, syncer_sink).await?;

        // A sanity check to confirm that following the intrusive addition of new pruning points,
        // the latest pruning point still correctly agrees with the DAG data,
        // and is the head of a pruning points "chain" leading all the way down to genesis
        // TODO (relaxed): once the catchup functionality has sufficiently matured, consider only doing this test if sanity checks are enabled
        info!("validating pruning points consistency");
        consensus.async_validate_pruning_points(syncer_sink).await.unwrap();
        info!("pruning points consistency validated");
        Ok(())
    }

    async fn ibd_with_headers_proof(
        &mut self,
        staging: &StagingConsensus,
        syncer_virtual_selected_parent: Hash,
        relay_block: &Block,
    ) -> Result<(), ProtocolError> {
        info!("Starting IBD with headers proof with peer {}", self.router);

        let staging_session = staging.session().await;

        let pruning_point = self.sync_and_validate_pruning_proof(&staging_session, relay_block).await?;
        self.sync_headers(&staging_session, syncer_virtual_selected_parent, pruning_point, relay_block).await?;
        staging_session.async_validate_pruning_points(syncer_virtual_selected_parent).await?;
        self.validate_staging_timestamps(&self.ctx.consensus().session().await, &staging_session).await?;
        Ok(())
    }

    async fn sync_and_validate_pruning_proof(&mut self, staging: &ConsensusProxy, relay_block: &Block) -> Result<Hash, ProtocolError> {
        self.router.enqueue(make_message!(Payload::RequestPruningPointProof, RequestPruningPointProofMessage {})).await?;

        // Pruning proof generation and communication might take several minutes, so we allow a long 10 minute timeout
        let msg = dequeue_with_timeout!(self.incoming_route, Payload::PruningPointProof, Duration::from_secs(600))?;
        let proof: PruningPointProof = Versioned(self.header_format, msg).try_into()?;
        info!(
            "Received headers proof with overall {} headers ({} unique)",
            proof.iter().map(|l| l.len()).sum::<usize>(),
            proof.iter().flatten().unique_by(|h| h.hash).count()
        );

        let proof_metadata = PruningProofMetadata::new(relay_block.header.blue_work);

        // Get a new session for current consensus (non staging)
        let consensus = self.ctx.consensus().session().await;

        // The proof is validated in the context of current consensus
        let proof =
            consensus.clone().spawn_blocking(move |c| c.validate_pruning_proof(&proof, &proof_metadata).map(|()| proof)).await?;

        let proof_pruning_point = proof[0].last().expect("was just ensured by validation").hash;

        if proof_pruning_point == self.ctx.config.genesis.hash {
            return Err(ProtocolError::Other("the proof pruning point is the genesis block"));
        }

        if proof_pruning_point == consensus.async_pruning_point().await {
            return Err(ProtocolError::Other("the proof pruning point is the same as the current pruning point"));
        }
        drop(consensus);

        self.router
            .enqueue(make_message!(Payload::RequestPruningPointAndItsAnticone, RequestPruningPointAndItsAnticoneMessage {}))
            .await?;
        // First, all pruning points up to the last are sent
        let msg = dequeue_with_timeout!(self.incoming_route, Payload::PruningPoints)?;
        let pruning_points: PruningPointsList = Versioned(self.header_format, msg).try_into()?;

        if pruning_points.is_empty() || pruning_points.last().unwrap().hash != proof_pruning_point {
            return Err(ProtocolError::Other("the proof pruning point is not equal to the last pruning point in the list"));
        }

        if pruning_points.first().unwrap().hash != self.ctx.config.genesis.hash {
            return Err(ProtocolError::Other("the first pruning point in the list is expected to be genesis"));
        }

        if self.ctx.consensus().session().await.async_are_pruning_points_violating_finality(pruning_points.clone()).await {
            let peer_ip = self.router.net_address().ip();
            if self.ctx.ban_peer_automatically(peer_ip).await {
                warn!("Banned peer {} for sending pruning points that violate finality", self.router);
            }
            return Err(ProtocolError::Other("pruning points are violating finality"));
        }

        {
            let pruning_points_set: BlockHashSet = pruning_points.iter().map(|h| h.hash).collect();
            for level in proof.iter() {
                if let Some(root) = level.first()
                    && root.hash != self.ctx.config.genesis.hash
                    && !pruning_points_set.contains(&root.pruning_point)
                {
                    return Err(ProtocolError::Other("proof and past pruning points are inconsistent with each other"));
                }
            }
        }

        let msg = dequeue_with_timeout!(self.incoming_route, Payload::TrustedData)?;
        let pkg: TrustedDataPackage = Versioned(self.header_format, msg).try_into()?;
        debug!("received trusted data with {} daa entries and {} ghostdag entries", pkg.daa_window.len(), pkg.ghostdag_window.len());

        let mut entry_stream = TrustedEntryStream::new(&self.router, &mut self.incoming_route, self.header_format);
        let Some(pruning_point_entry) = entry_stream.next().await? else {
            return Err(ProtocolError::Other("got `done` message before receiving the pruning point"));
        };

        if pruning_point_entry.block.hash() != proof_pruning_point {
            return Err(ProtocolError::Other("the proof pruning point is not equal to the expected trusted entry"));
        }

        let mut entries = vec![pruning_point_entry];
        while let Some(entry) = entry_stream.next().await? {
            entries.push(entry);
        }
        let mut trusted_set = pkg.build_trusted_subdag(entries)?;

        if self.ctx.config.enable_sanity_checks {
            let con = self.ctx.consensus().unguarded_session_blocking();
            trusted_set = staging
                .clone()
                .spawn_blocking(move |c| {
                    let ref_proof = proof.clone();
                    c.apply_pruning_proof(proof, &trusted_set)?;
                    c.import_pruning_points(pruning_points)?;

                    info!("Building the proof which was just applied (sanity test)");
                    let built_proof = c.get_pruning_point_proof();
                    let mut mismatch_detected = false;
                    for (i, (ref_level, built_level)) in ref_proof.iter().zip(built_proof.iter()).enumerate() {
                        if ref_level.iter().map(|h| h.hash).collect::<BlockHashSet>()
                            != built_level.iter().map(|h| h.hash).collect::<BlockHashSet>()
                        {
                            mismatch_detected = true;
                            warn!("Locally built proof for level {} does not match the applied one", i);
                        }
                    }
                    if mismatch_detected {
                        info!("Validating the locally built proof (sanity test fallback #2)");
                        if let Err(err) = con.validate_pruning_proof(&built_proof, &proof_metadata) {
                            panic!("Locally built proof failed validation: {}", err);
                        }
                        info!("Locally built proof was validated successfully");
                    } else {
                        info!("Proof was locally built successfully");
                    }
                    Result::<_, ProtocolError>::Ok(trusted_set)
                })
                .await?;
        } else {
            trusted_set = staging
                .clone()
                .spawn_blocking(move |c| {
                    c.apply_pruning_proof(proof, &trusted_set)?;
                    c.import_pruning_points(pruning_points)?;
                    Result::<_, ProtocolError>::Ok(trusted_set)
                })
                .await?;
        }

        // Queue trusted blocks in topological order and drain their virtual-state tasks in bounded
        // batches. The old path awaited every block serially even though the consensus API already
        // exposes independent futures and had a TODO to batch them.
        info!("Starting to process {} trusted blocks", trusted_set.len());
        let mut last_time = Instant::now();
        let mut last_index: usize = 0;
        let mut jobs = Vec::with_capacity(IBD_BATCH_SIZE);
        let mut processed = 0usize;
        for tb in trusted_set {
            jobs.push(staging.validate_and_insert_trusted_block(tb).virtual_state_task);
            if jobs.len() == IBD_BATCH_SIZE {
                let batch_len = jobs.len();
                try_join_all(std::mem::take(&mut jobs)).await?;
                processed += batch_len;
                let now = Instant::now();
                let passed = now.duration_since(last_time);
                if passed > Duration::from_secs(1) {
                    info!(
                        "Processed {} trusted blocks in the last {:.2}s (total {})",
                        processed - last_index,
                        passed.as_secs_f64(),
                        processed
                    );
                    last_time = now;
                    last_index = processed;
                }
            }
        }
        if !jobs.is_empty() {
            let batch_len = jobs.len();
            try_join_all(jobs).await?;
            processed += batch_len;
        }
        staging.async_clear_body_missing_anticone_set().await;
        info!("Done processing {} trusted blocks", processed);
        Ok(proof_pruning_point)
    }

    fn sync_ceiling(&self) -> Option<u64> {
        static SYNC_CEILING: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
        *SYNC_CEILING.get_or_init(|| std::env::var("KERYX_SYNC_CEILING_DAA").ok().and_then(|s| s.parse().ok()))
    }

    fn trust_sync_from_tip(&self) -> bool {
        static TRUST_SYNC: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *TRUST_SYNC.get_or_init(|| matches!(std::env::var("KERYX_TRUST_SYNC_FROM_TIP").as_deref(), Ok("1")))
    }

    async fn sync_headers(
        &mut self,
        consensus: &ConsensusProxy,
        syncer_virtual_selected_parent: Hash,
        highest_known_syncer_chain_hash: Hash,
        relay_block: &Block,
    ) -> Result<Hash, ProtocolError> {
        let ceiling = self.sync_ceiling();
        let highest_shared_header_score = consensus.async_get_header(highest_known_syncer_chain_hash).await?.daa_score;
        let mut progress_reporter = ProgressReporter::new(highest_shared_header_score, relay_block.header.daa_score, "block headers");

        self.router
            .enqueue(make_message!(
                Payload::RequestHeaders,
                RequestHeadersMessage {
                    low_hash: Some(highest_known_syncer_chain_hash.into()),
                    high_hash: Some(syncer_virtual_selected_parent.into())
                }
            ))
            .await?;
        let mut chunk_stream = HeadersChunkStream::new(&self.router, &mut self.incoming_route, self.header_format);

        let mut ceiling_hit = false;
        let mut prev: Option<(Vec<BlockValidationFuture>, u64, u64)> = None;
        loop {
            let chunk = match chunk_stream.next().await? {
                Some(chunk) => chunk,
                None => break,
            };
            let (current_daa_score, current_timestamp) = {
                let last_header = chunk.last().expect("chunk is never empty");
                (last_header.daa_score, last_header.timestamp)
            };
            let current_jobs: Vec<BlockValidationFuture> = chunk
                .into_iter()
                .filter(|h| match ceiling {
                    Some(c) if h.daa_score >= c => {
                        ceiling_hit = true;
                        false
                    }
                    _ => true,
                })
                .map(|h| consensus.validate_and_insert_block(Block::from_header_arc(h)).virtual_state_task)
                .collect();
            if let Some((prev_jobs, prev_daa_score, prev_timestamp)) = prev.take() {
                let prev_chunk_len = prev_jobs.len();
                try_join_all(prev_jobs).await?;
                progress_reporter.report(prev_chunk_len, prev_daa_score, prev_timestamp);
            }
            let reported_daa = ceiling.map(|c| current_daa_score.min(c.saturating_sub(1))).unwrap_or(current_daa_score);
            prev = Some((current_jobs, reported_daa, current_timestamp));
        }
        if let Some((prev_jobs, _, _)) = prev {
            let prev_chunk_len = prev_jobs.len();
            try_join_all(prev_jobs).await?;
            progress_reporter.report_completion(prev_chunk_len);
        }

        if ceiling_hit {
            let tip = consensus.async_get_headers_selected_tip().await;
            info!("sync ceiling reached during header download; stopping at headers selected tip {}", tip);
            return Ok(tip);
        }

        if consensus.async_get_block_status(syncer_virtual_selected_parent).await.is_none() {
            return Err(ProtocolError::OtherOwned(format!(
                "did not receive syncer's virtual selected parent {} from peer {} during header download",
                syncer_virtual_selected_parent, self.router
            )));
        }

        self.sync_missing_relay_past_headers(consensus, syncer_virtual_selected_parent, relay_block.hash()).await?;

        Ok(syncer_virtual_selected_parent)
    }

    async fn sync_new_utxo_set(
        &mut self,
        consensus: &ConsensusProxy,
        pruning_point: Hash,
        relay_header: &Header,
    ) -> Result<(), ProtocolError> {
        // A better solution could be to create a copy of the old utxo state for some sort of fallback rather than delete it.
        consensus.async_clear_pruning_utxo_set().await;
        self.sync_pruning_point_utxoset(consensus, pruning_point).await?;
        consensus.async_set_pruning_utxoset_stable().await;
        self.sync_service_state(consensus, pruning_point, relay_header).await?;
        let consensus_manager = self.ctx.consensus_manager.clone();
        spawn_blocking(move || consensus_manager.invoke_consensus_reset_handlers()).await.unwrap();
        self.ctx.on_pruning_point_utxoset_override();
        Ok(())
    }

    async fn sync_service_state(
        &mut self,
        consensus: &ConsensusProxy,
        pruning_point: Hash,
        relay_header: &Header,
    ) -> Result<(), ProtocolError> {
        let pp_daa = consensus.async_get_header(pruning_point).await?.daa_score;
        if !keryx_consensus_core::pom::service_commit_active(pp_daa) {
            return Ok(());
        }
        if self.protocol_version < 10 {
            return Err(ProtocolError::Other("peer cannot serve the service-state handoff window — sync from an upgraded peer"));
        }
        let expected = if relay_header.pruning_point == pruning_point {
            relay_header.service_state_hash
        } else {
            let hst = consensus.async_get_headers_selected_tip().await;
            let hst_header = consensus.async_get_header(hst).await?;
            if hst_header.pruning_point != pruning_point {
                return Err(ProtocolError::Other("no validated header anchors the negotiated pruning point"));
            }
            hst_header.service_state_hash
        };
        info!("downloading the sealed service state for pruning point {}", pruning_point);
        self.router
            .enqueue(make_message!(
                Payload::RequestServiceState,
                RequestServiceStateMessage { pruning_point_hash: Some(pruning_point.into()) }
            ))
            .await?;
        let handoff_cutoff = pp_daa + keryx_consensus_core::collateral::SERVICE_STATE_HANDOFF_DAA;
        let mut rows: Vec<Vec<u8>> = Vec::new();
        let mut prefix_rows = 0usize;
        let mut acc = MuHash::new();
        loop {
            match tokio::time::timeout(keryx_p2p_lib::common::DEFAULT_TIMEOUT, self.incoming_route.recv()).await {
                Ok(Some(msg)) => match msg.payload {
                    Some(Payload::ServiceStateChunk(chunk)) => {
                        for row in chunk.rows {
                            let daa = service_row_daa(&row).ok_or(ProtocolError::Other("malformed service-state row"))?;
                            if daa > handoff_cutoff {
                                return Err(ProtocolError::Other("service-state row beyond the handoff ceiling"));
                            }
                            if daa <= pp_daa {
                                acc.add_element(&row);
                                prefix_rows += 1;
                            }
                            rows.push(row);
                        }
                    }
                    Some(Payload::DoneServiceStateChunks(_)) => break,
                    _ => {
                        return Err(ProtocolError::UnexpectedMessage(
                            stringify!(Payload::ServiceStateChunk | Payload::DoneServiceStateChunks),
                            msg.payload.as_ref().map(|v| v.into()),
                        ));
                    }
                },
                Ok(None) => return Err(ProtocolError::ConnectionClosed),
                Err(_) => return Err(ProtocolError::Timeout(keryx_p2p_lib::common::DEFAULT_TIMEOUT)),
            }
        }
        let computed = if prefix_rows == 0 { Hash::default() } else { acc.finalize() };
        if computed != expected {
            return Err(ProtocolError::OtherOwned(format!(
                "service-state verification failed: peer rows hash to {}, header commits {}",
                computed, expected
            )));
        }
        let handoff_rows = rows.len() - prefix_rows;
        consensus.clone().spawn_blocking(move |c| c.import_service_state(rows)).await?;
        info!("imported {} sealed service-state rows ({} verified, {} handoff)", prefix_rows + handoff_rows, prefix_rows, handoff_rows);
        Ok(())
    }

    async fn sync_missing_relay_past_headers(
        &mut self,
        consensus: &ConsensusProxy,
        syncer_virtual_selected_parent: Hash,
        relay_block_hash: Hash,
    ) -> Result<(), ProtocolError> {
        if consensus.async_get_block_status(relay_block_hash).await.is_some() {
            return Ok(());
        }

        self.router
            .enqueue(make_message!(
                Payload::RequestAntipast,
                RequestAntipastMessage {
                    block_hash: Some(syncer_virtual_selected_parent.into()),
                    context_hash: Some(relay_block_hash.into())
                }
            ))
            .await?;

        let msg = dequeue_with_timeout!(self.incoming_route, Payload::BlockHeaders)?;
        let chunk: HeadersChunk = Versioned(self.header_format, msg).try_into()?;
        let jobs: Vec<BlockValidationFuture> =
            chunk.into_iter().map(|h| consensus.validate_and_insert_block(Block::from_header_arc(h)).virtual_state_task).collect();
        try_join_all(jobs).await?;
        dequeue_with_timeout!(self.incoming_route, Payload::DoneHeaders)?;

        if consensus.async_get_block_status(relay_block_hash).await.is_none() {
            Err(ProtocolError::OtherOwned(format!(
                "did not receive relay block {} from peer {} during header download",
                relay_block_hash, self.router
            )))
        } else {
            Ok(())
        }
    }

    async fn validate_staging_timestamps(
        &self,
        consensus: &ConsensusProxy,
        staging_consensus: &ConsensusProxy,
    ) -> Result<(), ProtocolError> {
        let staging_hst = staging_consensus.async_get_header(staging_consensus.async_get_headers_selected_tip().await).await.unwrap();
        let current_hst = consensus.async_get_header(consensus.async_get_headers_selected_tip().await).await.unwrap();
        if staging_hst.timestamp < current_hst.timestamp || staging_hst.timestamp - current_hst.timestamp < 600_000 {
            Err(ProtocolError::OtherOwned(format!(
                "The difference between the timestamp of the current selected tip ({}) and the \nstaging selected tip ({}) is too small or negative. Aborting IBD...",
                current_hst.timestamp, staging_hst.timestamp
            )))
        } else {
            Ok(())
        }
    }

    async fn sync_pruning_point_utxoset(&mut self, consensus: &ConsensusProxy, pruning_point: Hash) -> Result<(), ProtocolError> {
        info!("downloading the pruning point utxoset, this can take a little while.");
        self.router
            .enqueue(make_message!(
                Payload::RequestPruningPointUtxoSet,
                RequestPruningPointUtxoSetMessage { pruning_point_hash: Some(pruning_point.into()) }
            ))
            .await?;
        let mut chunk_stream = PruningPointUtxosetChunkStream::new(&self.router, &mut self.incoming_route);
        let mut multiset = MuHash::new();
        while let Some(chunk) = chunk_stream.next().await? {
            multiset = consensus
                .clone()
                .spawn_blocking(move |c| {
                    c.append_imported_pruning_point_utxos(&chunk, &mut multiset);
                    multiset
                })
                .await;
        }
        consensus.clone().spawn_blocking(move |c| c.import_pruning_point_utxo_set(pruning_point, multiset)).await?;
        Ok(())
    }

    async fn sync_missing_trusted_bodies(&mut self, consensus: &ConsensusProxy) -> Result<(), ProtocolError> {
        info!("downloading pruning point anticone missing block data");
        let diesembodied_hashes = consensus.async_get_body_missing_anticone().await;
        if self.body_only_ibd_permitted {
            self.sync_missing_trusted_bodies_no_headers(consensus, diesembodied_hashes).await?
        } else {
            self.sync_missing_trusted_bodies_full_blocks(consensus, diesembodied_hashes).await?;
        }
        consensus.async_clear_body_missing_anticone_set().await;
        Ok(())
    }

    async fn sync_missing_trusted_bodies_no_headers(
        &mut self,
        consensus: &ConsensusProxy,
        diesembodied_hashes: Vec<Hash>,
    ) -> Result<(), ProtocolError> {
        for chunk in diesembodied_hashes.chunks(IBD_BATCH_SIZE) {
            self.router
                .enqueue(make_message!(
                    Payload::RequestBlockBodies,
                    RequestBlockBodiesMessage { hashes: chunk.iter().map(|h| h.into()).collect() }
                ))
                .await?;

            // While the peer is filling the route buffer, fetch all local metadata in one blocking
            // task. The previous implementation did two blocking transitions per received body.
            let metadata = Self::get_trusted_metadata_batch(consensus, chunk).await.map_err(|err| {
                ProtocolError::OtherOwned(format!("syncee inconsistency while loading trusted block metadata: {}", err))
            })?;
            let mut jobs = Vec::with_capacity(chunk.len());

            for ((&hash, (blk_header, ghostdag_data)), _) in chunk.iter().zip(metadata).zip(0..) {
                let msg = dequeue_with_timeout!(self.incoming_route, Payload::BlockBody)?;
                let wire_tier = msg.pom_tier.map(|tier| tier as u8);

                // These blocks are at pruning depth and their proofs are discarded. Modern peers
                // send pom_tier separately, so avoid deserializing a potentially 200+ KiB proof
                // solely to throw it away. Parse only as a compatibility fallback when tier is absent.
                let fallback_tier = if wire_tier.is_none() {
                    msg.pom_proof
                        .as_deref()
                        .map(PomProof::from_wire_bytes)
                        .transpose()
                        .map_err(|_| ProtocolError::OtherOwned(format!("invalid pom_proof for trusted block {}", hash)))?
                        .map(|proof| proof.tier)
                } else {
                    None
                };
                let blk_body: BlockBody = msg.try_into()?;
                if blk_body.is_empty() {
                    return Err(ProtocolError::OtherOwned(format!("sent empty block body for block {}", hash)));
                }
                let pom_tier = wire_tier.or(fallback_tier);
                let block = Block { header: blk_header, transactions: blk_body.into(), pom_proof: None, pom_tier };
                jobs.push(
                    consensus
                        .validate_and_insert_trusted_block(TrustedBlock::new(block, ghostdag_data))
                        .virtual_state_task,
                );
            }
            try_join_all(jobs).await?;
        }
        Ok(())
    }

    async fn sync_missing_trusted_bodies_full_blocks(
        &mut self,
        consensus: &ConsensusProxy,
        diesembodied_hashes: Vec<Hash>,
    ) -> Result<(), ProtocolError> {
        for chunk in diesembodied_hashes.chunks(IBD_BATCH_SIZE) {
            self.router
                .enqueue(make_message!(
                    Payload::RequestIbdBlocks,
                    RequestIbdBlocksMessage { hashes: chunk.iter().map(|h| h.into()).collect() }
                ))
                .await?;
            let ghostdag_batch = Self::get_ghostdag_batch(consensus, chunk).await?;
            let mut jobs = Vec::with_capacity(chunk.len());

            for (&hash, ghostdag_data) in chunk.iter().zip(ghostdag_batch) {
                let msg = dequeue_with_timeout!(self.incoming_route, Payload::IbdBlock)?;
                let mut block: Block = Versioned(self.header_format, msg).try_into()?;
                if block.hash() != hash {
                    return Err(ProtocolError::OtherOwned(format!("expected block {} but got {}", hash, block.hash())));
                }
                if block.is_header_only() {
                    return Err(ProtocolError::OtherOwned(format!("sent header of {} where expected block with body", block.hash())));
                }
                block.pom_tier = block.pom_tier.or_else(|| block.pom_proof.as_ref().map(|p| p.tier));
                block.pom_proof = None;
                jobs.push(
                    consensus
                        .validate_and_insert_trusted_block(TrustedBlock::new(block, ghostdag_data))
                        .virtual_state_task,
                );
            }
            try_join_all(jobs).await?;
        }
        Ok(())
    }

    async fn sync_missing_block_bodies(&mut self, consensus: &ConsensusProxy, high: Hash) -> Result<(), ProtocolError> {
        let sleep_task = sleep(Duration::from_secs(2));
        let hashes_task = consensus.async_get_missing_block_body_hashes(high);
        tokio::pin!(sleep_task);
        tokio::pin!(hashes_task);
        let hashes = match select(sleep_task, hashes_task).await {
            Either::Left((_, hashes_task)) => {
                info!(
                    "IBD: searching for missing block bodies to request from peer {}. This operation might take several seconds.",
                    self.router
                );
                hashes_task.await
            }
            Either::Right((hashes_result, _)) => hashes_result,
        }?;
        if hashes.is_empty() {
            return Ok(());
        }

        let bounds = [*hashes.first().expect("hashes was non empty"), *hashes.last().expect("hashes was non empty")];
        let mut bound_headers = Self::get_headers_batch(consensus, &bounds).await?.into_iter();
        let low_header = bound_headers.next().expect("requested low header");
        let high_header = bound_headers.next().expect("requested high header");
        let mut progress_reporter = ProgressReporter::new(low_header.daa_score, high_header.daa_score, "blocks");
        let high_daa = high_header.daa_score;

        let mut iter = hashes.chunks(IBD_BATCH_SIZE);
        let QueueChunkOutput { jobs: mut prev_jobs, daa_score: mut prev_daa_score, timestamp: mut prev_timestamp } =
            self.queue_block_processing_chunk(consensus, iter.next().expect("hashes was non empty"), high_daa).await?;

        for chunk in iter {
            let QueueChunkOutput { jobs: current_jobs, daa_score: current_daa_score, timestamp: current_timestamp } =
                self.queue_block_processing_chunk(consensus, chunk, high_daa).await?;
            let prev_chunk_len = prev_jobs.len();
            try_join_all(prev_jobs).await?;
            progress_reporter.report(prev_chunk_len, prev_daa_score, prev_timestamp);
            prev_daa_score = current_daa_score;
            prev_timestamp = current_timestamp;
            prev_jobs = current_jobs;
        }

        let prev_chunk_len = prev_jobs.len();
        try_join_all(prev_jobs).await?;
        progress_reporter.report_completion(prev_chunk_len);

        Ok(())
    }

    async fn queue_block_processing_chunk(
        &mut self,
        consensus: &ConsensusProxy,
        chunk: &[Hash],
        high_daa: u64,
    ) -> Result<QueueChunkOutput, ProtocolError> {
        if self.body_only_ibd_permitted {
            self.queue_block_processing_chunk_body_only(consensus, chunk, high_daa).await
        } else {
            self.queue_block_processing_chunk_full_block(consensus, chunk, high_daa).await
        }
    }

    async fn queue_block_processing_chunk_full_block(
        &mut self,
        consensus: &ConsensusProxy,
        chunk: &[Hash],
        high_daa: u64,
    ) -> Result<QueueChunkOutput, ProtocolError> {
        let mut jobs = Vec::with_capacity(chunk.len());
        let mut current_daa_score = 0;
        let mut current_timestamp = 0;
        self.router
            .enqueue(make_message!(
                Payload::RequestIbdBlocks,
                RequestIbdBlocksMessage { hashes: chunk.iter().map(|h| h.into()).collect() }
            ))
            .await?;
        for &expected_hash in chunk {
            let msg = dequeue_with_timeout!(self.incoming_route, Payload::IbdBlock)?;
            let mut block: Block = Versioned(self.header_format, msg).try_into()?;
            if block.hash() != expected_hash {
                return Err(ProtocolError::OtherOwned(format!("expected block {} but got {}", expected_hash, block.hash())));
            }
            if block.is_header_only() {
                return Err(ProtocolError::OtherOwned(format!("sent header of {} where expected block with body", block.hash())));
            }
            if high_daa.saturating_sub(block.header.daa_score) > POM_PROOF_SERVE_DEPTH_DAA {
                block.pom_tier = block.pom_tier.or_else(|| block.pom_proof.as_ref().map(|p| p.tier));
                block.pom_proof = None;
            } else if block.pom_proof.is_none() && self.ctx.config.pom_activation.is_active(block.header.daa_score) {
                self.ctx.enqueue_pom_reproof(block.hash());
            }
            current_daa_score = block.header.daa_score;
            current_timestamp = block.header.timestamp;
            jobs.push(consensus.validate_and_insert_block_ibd(block).virtual_state_task);
        }
        Ok(QueueChunkOutput { jobs, daa_score: current_daa_score, timestamp: current_timestamp })
    }

    async fn queue_block_processing_chunk_body_only(
        &mut self,
        consensus: &ConsensusProxy,
        chunk: &[Hash],
        high_daa: u64,
    ) -> Result<QueueChunkOutput, ProtocolError> {
        let mut jobs = Vec::with_capacity(chunk.len());
        let mut current_daa_score = 0;
        let mut current_timestamp = 0;
        self.router
            .enqueue(make_request!(
                Payload::RequestBlockBodies,
                RequestBlockBodiesMessage { hashes: chunk.iter().map(|h| h.into()).collect() },
                self.incoming_route.id()
            ))
            .await?;

        // Fetch all local headers in one blocking consensus task while the peer fills the route.
        let headers = Self::get_headers_batch(consensus, chunk).await.map_err(|err| {
            ProtocolError::OtherOwned(format!("syncee inconsistency while loading block headers for body sync: {}", err))
        })?;

        for ((&expected_hash, blk_header), _) in chunk.iter().zip(headers).zip(0..) {
            let msg = dequeue_with_timeout!(self.incoming_route, Payload::BlockBody)?;
            let wire_tier = msg.pom_tier.map(|t| t as u8);
            let deep = high_daa.saturating_sub(blk_header.daa_score) > POM_PROOF_SERVE_DEPTH_DAA;

            // Historical blocks do not retain possession proofs. If the peer supplied the tier as
            // a separate field, skip deserializing the large proof entirely. Recent blocks and
            // compatibility peers without pom_tier still parse it normally.
            let pom_proof = if deep && wire_tier.is_some() {
                None
            } else {
                msg.pom_proof
                    .as_deref()
                    .map(PomProof::from_wire_bytes)
                    .transpose()
                    .map_err(|_| ProtocolError::OtherOwned(format!("invalid pom_proof for block {}", expected_hash)))?
                    .map(Arc::new)
            };
            let blk_body: BlockBody = msg.try_into()?;
            if blk_body.is_empty() {
                return Err(ProtocolError::OtherOwned(format!("sent empty block body for block {}", expected_hash)));
            }

            let pom_tier = wire_tier.or_else(|| pom_proof.as_ref().map(|proof| proof.tier));
            let pom_proof = if deep {
                None
            } else {
                if pom_proof.is_none() && self.ctx.config.pom_activation.is_active(blk_header.daa_score) {
                    self.ctx.enqueue_pom_reproof(blk_header.hash);
                }
                pom_proof
            };
            let block = Block { header: blk_header, transactions: blk_body.into(), pom_proof, pom_tier };
            current_daa_score = block.header.daa_score;
            current_timestamp = block.header.timestamp;
            jobs.push(consensus.validate_and_insert_block_ibd(block).virtual_state_task);
        }
        Ok(QueueChunkOutput { jobs, daa_score: current_daa_score, timestamp: current_timestamp })
    }
}
