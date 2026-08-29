use crate::{
    consensus::test_consensus::TestConsensus,
    model::{
        services::reachability::ReachabilityService,
        stores::{ai_slash::AiResponseStoreReader, pom_tier::PomTierStoreReader},
    },
};
use keryx_inference::{self, AiResponsePayload, compute_ai_commitment};
use keryx_consensus_core::{
    BlockHashSet,
    api::ConsensusApi,
    block::{Block, BlockTemplate, MutableBlock, TemplateBuildMode, TemplateTransactionSelector},
    blockhash,
    blockstatus::BlockStatus,
    coinbase::MinerData,
    config::{ConfigBuilder, params::MAINNET_PARAMS},
    subnets::SUBNETWORK_ID_AI_RESPONSE,
    tx::{ScriptPublicKey, ScriptVec, Transaction},
};
use crate::constants::TX_VERSION;
use keryx_hashes::Hash;
use std::{collections::VecDeque, thread::JoinHandle};

struct OnetimeTxSelector {
    txs: Option<Vec<Transaction>>,
}

impl OnetimeTxSelector {
    fn new(txs: Vec<Transaction>) -> Self {
        Self { txs: Some(txs) }
    }
}

impl TemplateTransactionSelector for OnetimeTxSelector {
    fn select_transactions(&mut self) -> Vec<Transaction> {
        self.txs.take().unwrap()
    }

    fn reject_selection(&mut self, _tx_id: keryx_consensus_core::tx::TransactionId) {
        unimplemented!()
    }

    fn is_successful(&self) -> bool {
        true
    }
}

struct TestContext {
    consensus: TestConsensus,
    join_handles: Vec<JoinHandle<()>>,
    miner_data: MinerData,
    simulated_time: u64,
    current_templates: VecDeque<BlockTemplate>,
    current_tips: BlockHashSet,
}

impl Drop for TestContext {
    fn drop(&mut self) {
        self.consensus.shutdown(std::mem::take(&mut self.join_handles));
    }
}

impl TestContext {
    fn new(consensus: TestConsensus) -> Self {
        let join_handles = consensus.init();
        let genesis_hash = consensus.params().genesis.hash;
        let simulated_time = consensus.params().genesis.timestamp;
        Self {
            consensus,
            join_handles,
            miner_data: new_miner_data(),
            simulated_time,
            current_templates: Default::default(),
            current_tips: BlockHashSet::from_iter([genesis_hash]),
        }
    }

    pub fn build_block_template_row(&mut self, nonces: impl Iterator<Item = usize>) -> &mut Self {
        for nonce in nonces {
            self.simulated_time += self.consensus.params().target_time_per_block();
            self.current_templates.push_back(self.build_block_template(nonce as u64, self.simulated_time));
        }
        self
    }

    pub fn assert_row_parents(&mut self) -> &mut Self {
        for t in self.current_templates.iter() {
            assert_eq!(self.current_tips, BlockHashSet::from_iter(t.block.header.direct_parents().iter().copied()));
        }
        self
    }

    pub async fn validate_and_insert_row(&mut self) -> &mut Self {
        self.current_tips.clear();
        while let Some(t) = self.current_templates.pop_front() {
            self.current_tips.insert(t.block.header.hash);
            self.validate_and_insert_block(t.block.to_immutable()).await;
        }
        self
    }

    pub async fn build_and_insert_disqualified_chain(&mut self, mut parents: Vec<Hash>, len: usize) -> Hash {
        // The chain will be disqualified since build_block_with_parents builds utxo-invalid blocks
        for _ in 0..len {
            self.simulated_time += self.consensus.params().target_time_per_block();
            let b = self.build_block_with_parents(parents, 0, self.simulated_time);
            parents = vec![b.header.hash];
            self.validate_and_insert_block(b.to_immutable()).await;
        }
        parents[0]
    }

    pub fn build_block_template(&self, nonce: u64, timestamp: u64) -> BlockTemplate {
        let mut t = self
            .consensus
            .build_block_template(
                self.miner_data.clone(),
                Box::new(OnetimeTxSelector::new(Default::default())),
                TemplateBuildMode::Standard,
            )
            .unwrap();
        t.block.header.timestamp = timestamp;
        t.block.header.nonce = nonce;
        t.block.header.finalize();
        t
    }

    pub fn build_block_with_parents(&self, parents: Vec<Hash>, nonce: u64, timestamp: u64) -> MutableBlock {
        let mut b = self.consensus.build_block_with_parents_and_transactions(blockhash::NONE, parents, Default::default());
        b.header.timestamp = timestamp;
        b.header.nonce = nonce;
        b.header.finalize(); // This overrides the NONE hash we passed earlier with the actual hash
        b
    }

    pub async fn validate_and_insert_block(&mut self, block: Block) -> &mut Self {
        let status = self.consensus.validate_and_insert_block(block).virtual_state_task.await.unwrap();
        assert!(status.has_block_body());
        self
    }

    pub fn assert_tips(&mut self) -> &mut Self {
        assert_eq!(BlockHashSet::from_iter(self.consensus.get_tips().into_iter()), self.current_tips);
        self
    }

    pub fn assert_tips_num(&mut self, expected_num: usize) -> &mut Self {
        assert_eq!(BlockHashSet::from_iter(self.consensus.get_tips().into_iter()).len(), expected_num);
        self
    }

    pub fn assert_virtual_parents_subset(&mut self) -> &mut Self {
        assert!(self.consensus.get_virtual_parents().is_subset(&self.current_tips));
        self
    }

    pub fn assert_valid_utxo_tip(&mut self) -> &mut Self {
        // Assert that at least one body tip was resolved with valid UTXO
        assert!(self.consensus.body_tips().iter().copied().any(|h| self.consensus.block_status(h) == BlockStatus::StatusUTXOValid));
        self
    }
}

#[tokio::test]
async fn ibd_does_not_persist_forged_tier_without_pom_proof() {
    let mut params = MAINNET_PARAMS;
    params.pom_activation = keryx_consensus_core::config::params::ForkActivation::always();
    let config = ConfigBuilder::new(params).build();
    let ctx = TestContext::new(TestConsensus::new(&config));
    let timestamp = ctx.simulated_time + ctx.consensus.params().target_time_per_block();
    let block = ctx.build_block_template(0, timestamp).block.to_immutable().with_pom_tier(Some(4));

    assert!(block.pom_proof.is_none());
    assert_eq!(block.pom_tier, Some(4));
    let block_hash = block.hash();
    let normal = ctx.consensus.validate_and_insert_block(block.clone()).virtual_state_task.await;
    let ibd = ctx.consensus.validate_and_insert_block_ibd(block).virtual_state_task.await;
    assert!(matches!(normal, Err(keryx_consensus_core::errors::block::RuleError::PomProofMissing)), "normal={normal:?}");
    assert!(matches!(ibd, Ok(BlockStatus::StatusUTXOValid)), "ibd={ibd:?}");
    assert!(!ctx.consensus.pom_tier_store().has(block_hash).unwrap());
}

#[tokio::test]
async fn template_mining_sanity_test() {
    let config = ConfigBuilder::new(MAINNET_PARAMS).skip_proof_of_work().build();
    let mut ctx = TestContext::new(TestConsensus::new(&config));
    let rounds = 10;
    let width = 3;
    for _ in 0..rounds {
        ctx.build_block_template_row(0..width)
            .assert_row_parents()
            .validate_and_insert_row()
            .await
            .assert_tips()
            .assert_virtual_parents_subset()
            .assert_valid_utxo_tip();
    }
}

#[tokio::test]
async fn antichain_merge_test() {
    let config = ConfigBuilder::new(MAINNET_PARAMS)
        .skip_proof_of_work()
        .edit_consensus_params(|p| {
            p.max_block_parents = 4;
            p.mergeset_size_limit = 10;
        })
        .build();

    let mut ctx = TestContext::new(TestConsensus::new(&config));

    // Build a large 32-wide antichain
    ctx.build_block_template_row(0..32)
        .validate_and_insert_row()
        .await
        .assert_tips()
        .assert_virtual_parents_subset()
        .assert_valid_utxo_tip();

    // Mine a long enough chain s.t. the antichain is fully merged
    for _ in 0..32 {
        ctx.build_block_template_row(0..1).validate_and_insert_row().await.assert_valid_utxo_tip();
    }
    ctx.assert_tips_num(1);
}

#[tokio::test]
async fn basic_utxo_disqualified_test() {
    keryx_core::log::try_init_logger("info");
    let config = ConfigBuilder::new(MAINNET_PARAMS)
        .skip_proof_of_work()
        .edit_consensus_params(|p| {
            p.max_block_parents = 4;
            p.mergeset_size_limit = 10;
        })
        .build();

    let mut ctx = TestContext::new(TestConsensus::new(&config));

    // Mine a valid chain
    for _ in 0..10 {
        ctx.build_block_template_row(0..1).validate_and_insert_row().await.assert_valid_utxo_tip();
    }

    // Get current sink
    let sink = ctx.consensus.get_sink();

    // Mine a longer disqualified chain
    let disqualified_tip = ctx.build_and_insert_disqualified_chain(vec![config.genesis.hash], 20).await;

    assert_ne!(sink, disqualified_tip);
    assert_eq!(sink, ctx.consensus.get_sink());
    assert_eq!(BlockHashSet::from_iter([sink, disqualified_tip]), BlockHashSet::from_iter(ctx.consensus.get_tips().into_iter()));
    assert!(!ctx.consensus.get_virtual_parents().contains(&disqualified_tip));
}

#[tokio::test]
async fn double_search_disqualified_test() {
    // TODO: add non-coinbase transactions and concurrency in order to complicate the test

    keryx_core::log::try_init_logger("info");
    let config = ConfigBuilder::new(MAINNET_PARAMS)
        .skip_proof_of_work()
        .edit_consensus_params(|p| {
            p.max_block_parents = 4;
            p.mergeset_size_limit = 10;
            p.min_difficulty_window_size = p.difficulty_window_size;
        })
        .build();
    let mut ctx = TestContext::new(TestConsensus::new(&config));

    // Mine 3 valid blocks over genesis
    ctx.build_block_template_row(0..3)
        .validate_and_insert_row()
        .await
        .assert_tips()
        .assert_virtual_parents_subset()
        .assert_valid_utxo_tip();

    // Mark the one expected to remain on virtual chain
    let original_sink = ctx.consensus.get_sink();

    // Find the roots to be used for the disqualified chains
    let mut virtual_parents = ctx.consensus.get_virtual_parents();
    assert!(virtual_parents.remove(&original_sink));
    let mut iter = virtual_parents.into_iter();
    let root_1 = iter.next().unwrap();
    let root_2 = iter.next().unwrap();
    assert_eq!(iter.next(), None);

    // Mine a valid chain
    for _ in 0..10 {
        ctx.build_block_template_row(0..1).validate_and_insert_row().await.assert_valid_utxo_tip();
    }

    // Get current sink
    let sink = ctx.consensus.get_sink();

    assert!(ctx.consensus.reachability_service().is_chain_ancestor_of(original_sink, sink));

    // Mine a long disqualified chain
    let disqualified_tip_1 = ctx.build_and_insert_disqualified_chain(vec![root_1], 30).await;

    // And another shorter disqualified chain
    let disqualified_tip_2 = ctx.build_and_insert_disqualified_chain(vec![root_2], 20).await;

    assert_eq!(ctx.consensus.get_block_status(root_1), Some(BlockStatus::StatusUTXOValid));
    assert_eq!(ctx.consensus.get_block_status(root_2), Some(BlockStatus::StatusUTXOValid));

    assert_ne!(sink, disqualified_tip_1);
    assert_ne!(sink, disqualified_tip_2);
    assert_eq!(sink, ctx.consensus.get_sink());
    assert_eq!(
        BlockHashSet::from_iter([sink, disqualified_tip_1, disqualified_tip_2]),
        BlockHashSet::from_iter(ctx.consensus.get_tips().into_iter())
    );
    assert!(!ctx.consensus.get_virtual_parents().contains(&disqualified_tip_1));
    assert!(!ctx.consensus.get_virtual_parents().contains(&disqualified_tip_2));

    // Mine a long enough valid chain s.t. both disqualified chains are fully merged
    for _ in 0..30 {
        ctx.build_block_template_row(0..1).validate_and_insert_row().await.assert_valid_utxo_tip();
    }
    ctx.assert_tips_num(1);
}

fn new_miner_data() -> MinerData {
    let secp = secp256k1::Secp256k1::new();
    let mut rng = rand::thread_rng();
    let (_sk, pk) = secp.generate_keypair(&mut rng);
    let script = ScriptVec::from_slice(&pk.serialize());
    MinerData::new(ScriptPublicKey::new(0, script), keryx_inference::gen_opoi_extra_data(0))
}

// ── OPoI E2E helpers ──────────────────────────────────────────────────────────

fn opoi_config() -> keryx_consensus_core::config::Config {
    ConfigBuilder::new(MAINNET_PARAMS).skip_proof_of_work().build()
}

/// Build an AiResponse TX whose 34-byte response_ipfs_cid carries the given 32-byte commitment
/// in bytes [2..34] (the slice the consensus reads as claimed_commitment).
fn make_ai_response_tx(request_hash: [u8; 32], commitment: [u8; 32]) -> Transaction {
    let mut response_ipfs_cid = [0u8; 34];
    response_ipfs_cid[0] = 0x12; // sha2-256 multihash code
    response_ipfs_cid[1] = 0x20; // digest length (32 bytes)
    response_ipfs_cid[2..34].copy_from_slice(&commitment);
    let payload = AiResponsePayload::new(request_hash, 0, response_ipfs_cid, 0).serialize();
    Transaction::new(TX_VERSION, vec![], vec![], 0, SUBNETWORK_ID_AI_RESPONSE, 0, payload)
}

/// Compute the response_hash key used by the consensus: blake2b(tx.payload)[0..32].
fn response_hash_of(tx: &Transaction) -> Hash {
    let h = blake2b_simd::blake2b(&tx.payload);
    let mut bytes = [0u8; 32];
    bytes.copy_from_slice(&h.as_bytes()[..32]);
    Hash::from_bytes(bytes)
}

// ── OPoI E2E tests ────────────────────────────────────────────────────────────

/// After a block containing an AiResponse TX is accepted, the consensus must
/// have registered the response in the ai_response_store so challengers can
/// look it up by response_hash.
#[tokio::test]
async fn opoi_response_registered_on_chain() {
    let config = opoi_config();
    let tc = TestConsensus::new(&config);
    let handles = tc.init();

    let genesis = config.genesis.hash;
    let request_hash = [0x01u8; 32];
    let commitment = compute_ai_commitment(&request_hash);
    let response_tx = make_ai_response_tx(request_hash, commitment);
    let rh = response_hash_of(&response_tx);

    tc.add_utxo_valid_block_with_parents(1u64.into(), vec![genesis], vec![response_tx]).await.unwrap();

    let record = tc.ai_response_store().get(rh).expect("AiResponse must be registered");
    assert_eq!(record.request_hash, request_hash);
    assert_eq!(record.claimed_commitment, commitment);

    tc.shutdown(handles);
}

// OPoI slashing removed (v1.2.3): the slash-behavior tests (fraud→slash, honest→no-slash,
// unknown→no-slash, outside-window→no-slash) were dropped together with the slashing mechanism.
// Escrows are now always spendable; there is no slash state to assert.

// ── tier-reward E2E (full pipeline: commit → store → coinbase split) ──────────

/// End-to-end: a merged block's miner cut in its merging block's coinbase is scaled by the
/// merged block's cryptographically-proven PoM tier — persisted at body commit (`pom_tier_store`),
/// read back by the virtual processor when it builds the coinbase. The floor tier (0, −18 %) pays
/// its miner exactly 82 % of what the top tier (3, 0 %) pays, while the total block reward is
/// identical (the shortfall is burned). `skip_proof_of_work` skips `check_pom_proof`, so the test
/// can attach a chosen-tier proof without a real possession witness; only `tier` is read.
#[tokio::test]
async fn tier_reward_e2e_scales_merged_block_miner_cut() {
    use keryx_consensus_core::config::params::{ForkActivation, TIER_REWARD_BPS, TIER_REWARD_BPS_DIVISOR};
    use keryx_consensus_core::pom::PomProof;

    fn proof_with_tier(tier: u8) -> PomProof {
        // Contents are irrelevant (check_pom_proof is skipped); only `tier` is persisted/read.
        PomProof {
            tier,
            trace_root: [0; 32],
            pow_value: [0; 32],
            final_state: 0,
            initial_trace_path: vec![],
            final_trace_path: vec![],
            openings: vec![],
            steps_v2: None,
            v3: None,
            v4: None,
        }
    }

    // (total coinbase payout of the block merging A, the part paid to the shared miner SPK).
    async fn payout_for_tier(tier: u8) -> (u64, u64) {
        let mut params = MAINNET_PARAMS;
        params.pom_activation = ForkActivation::always();
        let config = ConfigBuilder::new(params).skip_proof_of_work().build();
        let mut ctx = TestContext::new(TestConsensus::new(&config));
        let miner_spk = ctx.miner_data.script_public_key.clone();

        // Block A over genesis, carrying a possession proof of `tier` → tier stored at commit.
        ctx.simulated_time += ctx.consensus.params().target_time_per_block();
        let a = ctx.build_block_template(0, ctx.simulated_time).block.to_immutable().with_pom_proof(proof_with_tier(tier));
        ctx.validate_and_insert_block(a).await;

        // Block B merges A: its coinbase rewards A, scaling A's miner cut by A's proven tier.
        ctx.simulated_time += ctx.consensus.params().target_time_per_block();
        let mut template_b = ctx.build_block_template(0, ctx.simulated_time);
        let coinbase_b = template_b.block.transactions.remove(0);
        let total: u64 = coinbase_b.outputs.iter().map(|o| o.value).sum();
        let miner: u64 = coinbase_b.outputs.iter().filter(|o| o.script_public_key == miner_spk).map(|o| o.value).sum();
        (total, miner)
    }

    let (total_top, miner_top) = payout_for_tier(3).await; // 0 %
    let (total_floor, miner_floor) = payout_for_tier(0).await; // −18 %

    assert!(miner_top > 0, "top-tier block must pay its miner");
    assert_eq!(total_top, total_floor, "tier penalty must not change the total block reward");
    assert_eq!(
        miner_floor,
        miner_top * TIER_REWARD_BPS[0] / TIER_REWARD_BPS_DIVISOR,
        "floor-tier miner must get exactly 82 % of the top-tier cut"
    );
    assert!(miner_floor < miner_top, "serving a heavier model must pay the miner strictly more");
}

/// A transaction spending a burned escrow outpoint is rejected in the UTXO context, before any
/// entry lookup — the spend-level enforcement of a finality-deep service miss.
#[tokio::test]
async fn burned_escrow_outpoint_spend_is_rejected() {
    use crate::processes::transaction_validator::{errors::TxRuleError, tx_validation_in_utxo_context::TxValidationFlags};
    use keryx_consensus_core::subnets::SUBNETWORK_ID_NATIVE;
    use keryx_consensus_core::tx::{TransactionInput, TransactionOutpoint};
    use keryx_consensus_core::utxo::utxo_collection::UtxoCollection;

    let config = opoi_config();
    let tc = TestConsensus::new(&config);
    let handles = tc.init();
    let vp = tc.virtual_processor().clone();

    let outpoint = TransactionOutpoint::new(7u64.into(), 1);
    vp.service_burned.write().insert(outpoint, 0);
    let tx = Transaction::new(
        TX_VERSION,
        vec![TransactionInput::new(outpoint, vec![], 0, 0)],
        vec![],
        0,
        SUBNETWORK_ID_NATIVE,
        0,
        vec![],
    );
    // A burn binds POVs strictly past `event daa + finality`: the sink reaching that score flushes
    // it only after its own blocks are validated, so the block AT that score is one the network
    // accepted. A node holding the row while replaying below it (fresh sync / restart catch-up)
    // must reach the same verdict.
    let finality = vp.finality_depth;
    let at_edge = vp.validate_transaction_in_utxo_context(&tx, &UtxoCollection::default(), finality, TxValidationFlags::Full);
    assert!(matches!(at_edge, Err(TxRuleError::MissingTxOutpoints)), "burn must not bind at the flush edge");

    let res = vp.validate_transaction_in_utxo_context(&tx, &UtxoCollection::default(), finality + 1, TxValidationFlags::Full);
    assert!(matches!(res, Err(TxRuleError::SpendOfBurnedEscrow(_))));

    // An untouched outpoint still fails only on the missing entry, proving the set is selective.
    let other = TransactionOutpoint::new(8u64.into(), 1);
    let tx2 = Transaction::new(
        TX_VERSION,
        vec![TransactionInput::new(other, vec![], 0, 0)],
        vec![],
        0,
        SUBNETWORK_ID_NATIVE,
        0,
        vec![],
    );
    let res2 = vp.validate_transaction_in_utxo_context(&tx2, &UtxoCollection::default(), 1, TxValidationFlags::Full);
    assert!(matches!(res2, Err(TxRuleError::MissingTxOutpoints)));

    tc.shutdown(handles);
}

/// Service-bond eligibility walk E2E: the audit cohort for a tier, seen from a committed chain
/// block, is the distinct escrow keys of proven blocks of that tier merged inside the DAA window,
/// and a shorter window truncates the set.
#[tokio::test]
async fn service_cohort_from_recent_tier_producers() {
    use keryx_consensus_core::collateral::{escrow_miner_key, miner_key};
    use keryx_consensus_core::config::params::ForkActivation;
    use keryx_consensus_core::pom::PomProof;

    fn proof_with_tier(tier: u8) -> PomProof {
        PomProof {
            tier,
            trace_root: [0; 32],
            pow_value: [0; 32],
            final_state: 0,
            initial_trace_path: vec![],
            final_trace_path: vec![],
            openings: vec![],
            steps_v2: None,
            v3: None,
            v4: None,
        }
    }

    let mut params = MAINNET_PARAMS;
    params.pom_activation = ForkActivation::always();
    // Active v3 gate: the service ledger folds every committed chain block through the real
    // `resolve_virtual` path (empty request stream — the lifecycle itself is unit-tested).
    params.pom_v3_activation = ForkActivation::always();
    let config = ConfigBuilder::new(params).skip_proof_of_work().build();
    let tc = TestConsensus::new(&config);
    let handles = tc.init();
    let genesis = config.genesis.hash;

    // m1 announces an escrow pubkey — his service identity (eligibility key, vault key); m2 does
    // not: his 20 % burns at emission, and without a bond he is not service-eligible at all.
    let mut m1 = new_miner_data();
    let mut extra = m1.extra_data.to_vec();
    extra.extend_from_slice(format!("/escrow:{}", "11".repeat(32)).as_bytes());
    m1.extra_data = extra.into();
    let m2 = new_miner_data();
    let id1 = miner_key(&m1.script_public_key);
    let e1 = escrow_miner_key(&[0x11u8; 32]);

    // Single-parent chain b1..b5; each block's tier is proven at its own body commit and paid
    // (hence walked) once merged by the next chain block. b5 is the seed, so the walk sees b1..b4:
    // tier 0 ← {m1 (b1, b4), m2 (b3, ignored — no escrow)}, tier 1 ← {m2 (b2, ignored)}. b3 also
    // carries an AiResponse to an unknown request, folded into the ledger as a no-op.
    let stray_response = make_ai_response_tx([0x42u8; 32], [0u8; 32]);
    let plan = [(1u64, &m1, 0u8), (2, &m2, 1), (3, &m2, 0), (4, &m1, 0), (5, &m1, 1)];
    let mut parent = genesis;
    for (n, miner, tier) in plan {
        let hash: Hash = n.into();
        let txs = if n == 3 { vec![stray_response.clone()] } else { vec![] };
        // The cohort fold reads the proven tier from the committed `header.pom_tier` (bound to
        // proof.tier by live validation), so set it on each block.
        let mut mutable = tc.build_utxo_valid_block_with_parents(hash, vec![parent], miner.clone(), txs);
        mutable.header.pom_tier = tier;
        let block = mutable.to_immutable().with_pom_proof(proof_with_tier(tier));
        tc.validate_and_insert_block(block).virtual_state_task.await.unwrap();
        parent = hash;
    }

    let vp = tc.virtual_processor().clone();
    let seed: Hash = 5u64.into();

    assert_eq!(vp.service_eligible_miners(seed, 0), vec![(id1, e1)]);
    assert!(vp.service_eligible_miners(seed, 1).is_empty(), "a miner without an escrow bond is never eligible");
    assert!(vp.service_eligible_miners(seed, 4).is_empty());


    // A 1-DAA window covers b5 alone, whose only merged blue is b4 (m1, tier 0).
    assert_eq!(vp.service_eligible_miners_windowed(seed, 0, 1), vec![(id1, e1)]);
    assert!(vp.service_eligible_miners_windowed(seed, 1, 1).is_empty());

    // A non-chain seed yields no eligible set at all.
    assert!(vp.service_eligible_miners(Hash::from_bytes([0xEE; 32]), 0).is_empty());

    // Escrow vault: m1's blues b1 and b4 were merged by committed chain blocks (b2, b5), each
    // leaving one CSV escrow claim keyed by his escrow pubkey; m2 announced none, so his cut
    // burned unclaimed and no vault exists under any key of his.
    let claims = vp.service_vault_claims(&id1);
    assert_eq!(claims.len(), 2);
    assert!(claims.iter().all(|c| c.value > 0));

    tc.shutdown(handles);
}

/// Gold-standard prefix-sum production index E2E: maintained in lockstep through the real
/// `commit_virtual_state` path, its windowed value at the sink (Case A of
/// `windowed_production_for_block`) accumulates one base miner cut per in-window selected-chain block
/// attributed to that block's producer, drops producers that age out of the last `ratio_reward_window`
/// blocks, and chains cumulatively for a producer that appears more than once inside the window.
#[tokio::test]
async fn windowed_production_prefix_accumulates_per_producer_and_slides() {
    use crate::model::stores::headers::HeaderStoreReader;
    use crate::model::stores::selected_chain::SelectedChainStoreReader;
    use crate::model::stores::windowed_production_prefix::WindowedProductionPrefixStoreReader;

    // Tiny window so a short chain exercises both the slide (aged-out producers drop to zero) and
    // cumulative chaining. Index maintenance is ungated (runs from genesis), so no ratio activation
    // is needed to populate the production index.
    let mut params = MAINNET_PARAMS;
    params.ratio_reward_window = 3;
    let w = params.ratio_reward_window;
    let config = ConfigBuilder::new(params).skip_proof_of_work().build();
    let mut ctx = TestContext::new(TestConsensus::new(&config));

    // 7-block single chain (chain indices 1..=7). The same producer mines blocks 6 and 7 (0-indexed
    // i=5,6 ⇒ chain indices 6,7), both inside the last-3 window ⇒ its windowed production must be two
    // base cuts (cumulative chaining within one SPK).
    let repeat = new_miner_data();
    let mut producers: Vec<ScriptPublicKey> = Vec::new();
    for i in 0..7 {
        let md = if i == 5 || i == 6 { repeat.clone() } else { new_miner_data() };
        producers.push(md.script_public_key.clone());
        ctx.miner_data = md;
        ctx.build_block_template_row(0..1).validate_and_insert_row().await;
    }

    let vp = ctx.consensus.virtual_processor().clone();
    let sink = ctx.consensus.get_sink();
    let tip_idx = vp.selected_chain_store.read().get_tip().unwrap().0;
    let one_cut = {
        let sink_daa = ctx.consensus.headers_store().get_daa_score(sink).unwrap();
        vp.coinbase_manager.base_miner_cut(sink_daa)
    };
    assert!(one_cut > 0, "an in-window producer must have a non-zero base cut");

    // Windowed production at the sink, asserting the Case-A block query and the direct prefix query agree.
    let windowed = |spk: &ScriptPublicKey| {
        let via_block = vp.windowed_production_for_block(spk, sink, w);
        let direct = vp.windowed_production_prefix_store.windowed(spk, tip_idx, w).unwrap();
        assert_eq!(via_block, direct, "Case-A block query must match the direct windowed query");
        via_block
    };

    // Window = last 3 selected-chain blocks (chain indices 5,6,7 ⇒ producers i=4,5,6). Producers i=0..3
    // aged out ⇒ dropped to zero; i=4 contributes one cut; the repeat (i=5,6) chains to two cuts.
    for i in 0..4 {
        assert_eq!(windowed(&producers[i]), 0, "producer {i} aged out of the window ⇒ entry dropped");
    }
    assert_eq!(windowed(&producers[4]), one_cut, "a single in-window block contributes exactly one base cut");
    assert_eq!(
        windowed(&repeat.script_public_key),
        2 * one_cut,
        "a producer of two in-window blocks must show two base cuts (cumulative chaining)"
    );
}

/// H3 era of the production index: per-blue accounting + daa-sized window. On a single chain each
/// chain block's sole mergeset blue is its selected parent, so index `i` credits the producer of
/// block `i−1` — payment-mirror semantics: a producer is credited when its block is MERGED, so the
/// sink's own producer is not yet in the window. The window bottom is found by daa (fixed
/// real-time duration) instead of a chain-block count.
///
/// `pom_level_activation` is `new(1)` (not `always()`): the same activation drives the global
/// header-hashing switch (`init_pom_level_activation`), and genesis (daa 0) must keep its pinned
/// legacy hash. Every non-genesis block of this consensus hashes with the (zero) `pom_final_state`
/// committed — internally consistent, and `skip_proof_of_work` bypasses the PoM checks.
#[tokio::test]
async fn windowed_production_prefix_h3_per_blue_daa_window() {
    use crate::model::stores::headers::HeaderStoreReader;
    use keryx_consensus_core::config::params::ForkActivation;

    let mut params = MAINNET_PARAMS;
    params.pom_level_activation = ForkActivation::new(1);
    params.ratio_reward_window_daa = 3;
    // Legacy window deliberately different so a wrong era pick is caught by the assertions below.
    params.ratio_reward_window = 5;
    let legacy_w = params.ratio_reward_window;
    let config = ConfigBuilder::new(params).skip_proof_of_work().build();
    let mut ctx = TestContext::new(TestConsensus::new(&config));

    // 7-block single chain (chain indices 1..=7, daa_score == chain index). The same producer
    // mines blocks at indices 6 and 7 (0-indexed i=5,6).
    let repeat = new_miner_data();
    let mut producers: Vec<ScriptPublicKey> = Vec::new();
    for i in 0..7 {
        let md = if i == 5 || i == 6 { repeat.clone() } else { new_miner_data() };
        producers.push(md.script_public_key.clone());
        ctx.miner_data = md;
        ctx.build_block_template_row(0..1).validate_and_insert_row().await;
    }

    let vp = ctx.consensus.virtual_processor().clone();
    let sink = ctx.consensus.get_sink();
    let sink_daa = ctx.consensus.headers_store().get_daa_score(sink).unwrap();
    // Single chain: daa_score = chain index − 1 (genesis daa 0, block 1 daa 0, block i daa i−1).
    assert_eq!(sink_daa, 6, "single chain ⇒ daa_score = chain index − 1");
    let one_cut = vp.coinbase_manager.base_miner_cut(sink_daa);

    let windowed = |spk: &ScriptPublicKey| vp.windowed_production_for_block(spk, sink, legacy_w);

    // daa window = (6−3, 6] in daa units ⇒ chain indices 5,6,7 (daa 4,5,6), crediting the MERGED
    // blues = the producers of blocks 4,5,6 (0-indexed producers[3], producers[4], producers[5]).
    for i in 0..3 {
        assert_eq!(windowed(&producers[i]), 0, "producer {i} merged below the daa window ⇒ zero");
    }
    assert_eq!(windowed(&producers[3]), one_cut, "block 4's producer is credited at merge index 5");
    assert_eq!(windowed(&producers[4]), one_cut, "block 5's producer is credited at merge index 6");
    // The repeat producer mined blocks 6 and 7, but only block 6 has been MERGED (at index 7);
    // block 7 is the sink itself — its production is credited when a child merges it, exactly like
    // its coinbase payment. Payment-mirror semantics: one cut, not two.
    assert_eq!(windowed(&repeat.script_public_key), one_cut, "the sink's own production is not yet merged ⇒ one cut");
}

/// Fastsync production-window trust (Option A): `trust_coinbase()` must relax ratio-reward coinbase
/// verification for exactly the `ratio_reward_window` selected-chain blocks following a pruning-point
/// UTXO import (the only window during which a fast-synced node's freshly-cleared windowed-production
/// prefix index cannot yet match a from-genesis node's), then self-expire. A node that has never
/// imported a snapshot must never see this relaxation.
///
/// The import is simulated at genesis (mirrors the `set_initial_utxo_set` / integration-test pattern
/// for `import_pruning_point_utxo_set`: an empty multiset trivially matches genesis's own UTXO
/// commitment). `import_pruning_point_utxo_set` recomputes virtual with the imported pruning point as
/// its sole parent, so this must happen *before* any blocks are built on top of genesis — doing it
/// after would silently discard that chain progress from virtual's perspective. Real fast sync never
/// hits that ordering hazard because the imported pruning point is always itself the current chain
/// tip; constructing a *non-genesis* pruning point with a correctly matching multiset needs the full
/// integration-test machinery (see `ratio_reward_balance_index_reconstruction_matches_incremental` /
/// `testing/integration/src/consensus_integration_tests.rs`), which is out of scope for this
/// unit-level test of `trust_coinbase()`'s windowing arithmetic.
#[tokio::test]
async fn fastsync_catchup_window_trusts_then_expires() {
    use crate::model::stores::selected_chain::SelectedChainStoreReader;
    use keryx_consensus_core::api::ConsensusApi;
    use keryx_muhash::MuHash;

    // Tiny window (mirrors `windowed_production_prefix_accumulates_per_producer_and_slides`) so the test
    // exercises the full catch-up-then-expiry cycle in a handful of blocks instead of needing the
    // real mainnet/testnet `ratio_reward_window` (864_000 / 1_000).
    let mut params = MAINNET_PARAMS;
    params.ratio_reward_window = 3;
    let window = params.ratio_reward_window;
    let config = ConfigBuilder::new(params).skip_proof_of_work().build();
    let mut ctx = TestContext::new(TestConsensus::new(&config));
    let vp = ctx.consensus.virtual_processor().clone();

    // Never imported a snapshot ⇒ no catch-up gap to begin with, regardless of chain progress.
    assert_eq!(vp.production_index_seed_store.read().get_optional(), None, "fresh node must have no seed recorded");
    assert!(!vp.trust_coinbase(), "a from-genesis node must never get the fastsync relaxation");

    // Simulate a pruning-point UTXO import (fast sync) at genesis, before any blocks are built (see
    // doc comment above for why ordering matters here). The seeded index it records is the *current*
    // selected-chain tip at the moment of the call — genesis, i.e. index 0.
    let genesis_hash = ctx.consensus.params().genesis.hash;
    ctx.consensus.import_pruning_point_utxo_set(genesis_hash, MuHash::new()).unwrap();

    let seeded_at = vp.production_index_seed_store.read().get_optional().expect("import must record a seed");
    assert_eq!(seeded_at, 0, "import happened at genesis ⇒ seeded index must be 0");
    assert!(vp.trust_coinbase(), "must be trusted immediately after import (0 blocks into the catch-up window)");

    // Still inside the window: ratio_reward_window - 1 more blocks keeps us under the threshold.
    for i in 0..(window - 1) {
        ctx.build_block_template_row(0..1).validate_and_insert_row().await;
        let tip_idx = vp.selected_chain_store.read().get_tip().unwrap().0;
        assert_eq!(tip_idx, i + 1, "single-chain row must advance the selected-chain tip by exactly one block");
        assert!(vp.trust_coinbase(), "must stay trusted while still inside the post-import catch-up window");
    }

    // One more block crosses the window boundary ⇒ the relaxation must self-expire.
    ctx.build_block_template_row(0..1).validate_and_insert_row().await;
    let tip_idx = vp.selected_chain_store.read().get_tip().unwrap().0;
    assert!(tip_idx - seeded_at >= window, "test setup sanity: must have crossed the window");
    assert!(!vp.trust_coinbase(), "must self-expire once a full ratio_reward_window of blocks has passed since import");
}

/// Ratio-reward (Stage 2b) reconstruction-equality: the balance index maintained incrementally from
/// genesis (lockstep with the virtual UTXO set in `commit_virtual_state`) must equal, key-for-key,
/// the index a fast-synced node rebuilds at `import_pruning_point_utxo_set` by grouping the imported
/// UTXO snapshot per payout SPK. If the two diverge, a fast-synced node would compute a different
/// holder bracket than a from-genesis node for the same block → divergent expected coinbase → a
/// consensus split. This pins the property that makes flipping `ratio_reward_activation` safe.
#[tokio::test]
async fn ratio_reward_balance_index_reconstruction_matches_incremental() {
    use crate::model::stores::address_amount::AddressAmountStoreReader;
    use keryx_consensus_core::tx::ScriptPublicKey;
    use std::collections::HashMap;

    // Index maintenance is ungated (runs from genesis), so no ratio activation is needed.
    let config = ConfigBuilder::new(MAINNET_PARAMS).skip_proof_of_work().build();
    let mut ctx = TestContext::new(TestConsensus::new(&config));

    // A handful of width-1 blocks, each built by a distinct random producer SPK, so several payout
    // addresses accrue coinbase (plus the escrow and R&D outputs of every block → multiple SPK kinds).
    for _ in 0..8 {
        ctx.miner_data = new_miner_data();
        ctx.build_block_template_row(0..1).validate_and_insert_row().await;
    }

    let vp = ctx.consensus.virtual_processor().clone();

    // Reconstruction seed: exact mirror of `import_pruning_point_utxo_set` — group the current
    // virtual UTXO snapshot per SPK with `saturating_add`.
    let mut reconstructed: HashMap<ScriptPublicKey, u64> = HashMap::new();
    {
        let virtual_read = vp.virtual_stores.read();
        for item in virtual_read.utxo_set.iterator() {
            let (_, entry) = item.unwrap();
            let acc = reconstructed.entry(entry.script_public_key.clone()).or_default();
            *acc = acc.saturating_add(entry.amount);
        }
    }
    assert!(!reconstructed.is_empty(), "the chain must have produced spendable UTXOs to make this test meaningful");

    // Forward direction (focused message): every UTXO-derived balance is present and equal in the
    // incrementally maintained index.
    for (spk, amount) in &reconstructed {
        assert_eq!(
            vp.address_balance_store.get(spk).unwrap(),
            *amount,
            "incremental balance index disagrees with the UTXO-set reconstruction for an address",
        );
    }

    // Reverse direction: the incremental index holds no extra/stale entry absent from the
    // reconstruction. Full key-set + value equality is exactly the byte-for-byte property a
    // fast-synced node relies on.
    let incremental = vp.address_balance_store.collect();
    assert_eq!(incremental, reconstructed, "balance index and UTXO-set reconstruction must match exactly");
}

/// Adversarial harness for the coin-age write path: drives `sweep_maturation_queue` +
/// `apply_age_diff` directly — in the exact order and batching of `commit_virtual_state` —
/// through score advances, tip re-anchors (score drops) and interleaved spends/re-adds, and
/// after EVERY commit asserts the self-check invariant: the stored buckets must equal the
/// reclassification of the shadow UTXO set at the committed score. Any sequence that breaks
/// equality is a deterministic reproduction of the production coin-age drift (the network-wide
/// `coin-age self-check DIVERGENCE` events: b_mat/b_imm split off while the sum stays exact).
#[tokio::test]
async fn coin_age_maturation_choreography_adversarial() {
    use crate::model::stores::age_buckets::AgeBuckets;
    use keryx_consensus_core::tx::{TransactionOutpoint, UtxoEntry};
    use keryx_consensus_core::utxo::utxo_diff::UtxoDiff;
    use keryx_database::{create_temp_db, prelude::ConnBuilder};
    use rocksdb::WriteBatch;
    use std::collections::HashMap;

    const W: u64 = 864_000; // asserted against params below so a fork change fails loudly
    const B: u64 = 10 * W; // base score: far enough that `score - W` never saturates
    const DUE1: u64 = B + W; // maturity score of a coin anchored at B

    // One commit of a scenario: (virtual score, coins added as (id, spk byte, amount, effective_daa),
    // coin ids spent). Re-add = same id in adds after a spend; re-anchor = same id in adds AND spends.
    type CommitSpec = (u64, Vec<(u64, u8, u64, u64)>, Vec<u64>);

    fn spk_n(byte: u8) -> ScriptPublicKey {
        ScriptPublicKey::new(0, ScriptVec::from_slice(&[byte; 34]))
    }
    fn op(id: u64) -> TransactionOutpoint {
        TransactionOutpoint::new(Hash::from_u64_word(id), 0)
    }

    fn run(name: &str, commits: &[CommitSpec]) {
        let config = ConfigBuilder::new(MAINNET_PARAMS).skip_proof_of_work().build();
        let (_life, db) = create_temp_db!(ConnBuilder::default().with_files_limit(10));
        let (sender, _receiver) = async_channel::unbounded();
        let tc = TestConsensus::with_db(db.clone(), &config, sender);
        let vp = tc.virtual_processor().clone();
        assert_eq!(vp.coin_age_maturity_w, W, "test constants assume the mainnet maturity window");

        let mut shadow: HashMap<TransactionOutpoint, UtxoEntry> = HashMap::new();
        for (i, (score, adds, spends)) in commits.iter().enumerate() {
            let mut diff = UtxoDiff { add: Default::default(), remove: Default::default() };
            for id in spends {
                let e = shadow.remove(&op(*id)).expect("scenario spends a coin absent from the shadow set");
                diff.remove.insert(op(*id), e);
            }
            for (id, spk_byte, amount, eda) in adds {
                assert!(*eda <= *score, "scenario adds a coin anchored in the future");
                let e = UtxoEntry {
                    amount: *amount,
                    script_public_key: spk_n(*spk_byte),
                    block_daa_score: *eda,
                    is_coinbase: true,
                    effective_daa: *eda,
                };
                diff.add.insert(op(*id), e.clone());
                shadow.insert(op(*id), e);
            }

            // Mirror of commit_virtual_state: sweep FIRST, then the diff, one batch, one write.
            let mut batch = WriteBatch::default();
            let needs_rebuild = vp.sweep_maturation_queue(&mut batch, *score, &diff);
            assert!(!needs_rebuild, "{name}: commit {i} dropped beyond the retention horizon — keep scenario drops shallow");
            vp.apply_age_diff(&mut batch, &diff, *score);
            db.write(batch).unwrap();

            // Self-check invariant at the committed score (the sweep pinned the watermark there).
            let bound = score.saturating_sub(W);
            let mut expected: HashMap<ScriptPublicKey, AgeBuckets> = HashMap::new();
            for e in shadow.values() {
                let b = expected.entry(e.script_public_key.clone()).or_default();
                if e.effective_daa <= bound {
                    b.b_mat += e.amount;
                } else {
                    b.b_imm += e.amount;
                    b.a_imm += e.amount as u128 * e.effective_daa as u128;
                }
            }
            expected.retain(|_, b| !b.is_empty());
            let stored = vp.age_buckets_store.collect();
            if stored != expected {
                // Deterministic repro: dump the exact commit sequence that led here so the
                // failing path can be minimized by hand.
                eprintln!("=== {name}: FAILING SEQUENCE (commits 0..={i}) ===");
                for (j, (s, a, r)) in commits.iter().enumerate().take(i + 1) {
                    eprintln!("  commit {j}: score={s} adds={a:?} spends={r:?}");
                }
                assert_eq!(
                    stored, expected,
                    "{name}: coin-age buckets diverged from the UTXO reclassification after commit {i} (score {score})"
                );
            }
        }
    }

    // S1 — baseline: deposit, ride to maturity, promote exactly at the due boundary.
    run(
        "s1_baseline_promotion",
        &[(B, vec![(1, 0xA1, 1_000, B)], vec![]), (DUE1 - 1, vec![], vec![]), (DUE1, vec![], vec![]), (DUE1 + 500, vec![], vec![])],
    );

    // S2 — pure tip oscillation: promote, re-anchor below the due (demote), re-advance (re-promote).
    run(
        "s2_demote_repromote",
        &[
            (B, vec![(1, 0xA1, 1_000, B)], vec![]),
            (DUE1 + 100, vec![], vec![]),
            (DUE1 - 50, vec![], vec![]),
            (DUE1 + 200, vec![], vec![]),
        ],
    );

    // S3 — score drop with the matured coin spent in the SAME commit (demote + immature remove).
    run(
        "s3_drop_with_inflight_spend",
        &[(B, vec![(1, 0xA1, 1_000, B)], vec![]), (DUE1 + 100, vec![], vec![]), (DUE1 - 50, vec![], vec![1])],
    );

    // S4 — spent after maturing, then a reorg restores the coin below its due (the skip-rule case),
    // then maturity again.
    run(
        "s4_spent_after_maturing_reorg_restore",
        &[
            (B, vec![(1, 0xA1, 1_000, B)], vec![]),
            (DUE1 + 100, vec![], vec![]),
            (DUE1 + 200, vec![], vec![1]),
            (DUE1 - 50, vec![(1, 0xA1, 1_000, B)], vec![]),
            (DUE1 + 300, vec![], vec![]),
        ],
    );

    // S5 — re-anchor during a drop: same outpoint removed (old anchor) AND re-added (new anchor)
    // in the demotion commit, then maturity at the NEW due.
    run(
        "s5_reanchor_same_outpoint_in_drop",
        &[
            (B, vec![(1, 0xA1, 1_000, B)], vec![]),
            (DUE1 + 100, vec![], vec![]),
            (DUE1 - 50, vec![(1, 0xA1, 1_000, DUE1 - 60)], vec![1]),
            (DUE1 - 60 + W + 10, vec![], vec![]),
        ],
    );

    // S6 — due and spent in the same commit: the sweep must promote first, the diff then removes
    // on the mature side (the ordering contract stated on `sweep_maturation_queue`).
    run("s6_due_and_spend_same_commit", &[(B, vec![(1, 0xA1, 1_000, B)], vec![]), (DUE1, vec![], vec![1])]);

    // S7 — several SPKs with staggered dues, a drop across a subset of them with an in-flight
    // spend, a fresh deposit while re-advancing.
    run(
        "s7_multi_spk_interleaved_oscillation",
        &[
            (
                B + 90,
                vec![(1, 0xA1, 500, B), (2, 0xA1, 600, B + 40), (3, 0xB2, 700, B + 20), (4, 0xB2, 800, B + 70), (5, 0xC3, 900, B + 90)],
                vec![],
            ),
            (B + W + 50, vec![], vec![]),
            (B + W + 30, vec![], vec![2]),
            (B + W + 60, vec![(6, 0xA1, 700, B + W + 55)], vec![]),
            (B + W + 100, vec![], vec![]),
        ],
    );

    // S8 — oscillation hammer: a due-dense cluster (20 coins, 10 DAA apart, 2 SPKs), then a dozen
    // advance/drop cycles walking through the cluster, each drop carrying a spend and a deposit —
    // the closest static approximation of a busy pool under routine tip re-anchors.
    let mut s8: Vec<CommitSpec> = Vec::new();
    let mut cluster: Vec<(u64, u8, u64, u64)> = Vec::new();
    for i in 0..20u64 {
        cluster.push((100 + i, if i % 2 == 0 { 0xE1 } else { 0xE2 }, 500 + i, B + 10 * i));
    }
    s8.push((B + 200, cluster, vec![]));
    let mut alive: Vec<u64> = (100..120).collect();
    let mut next_id = 200u64;
    for k in 0..12u64 {
        let hi = B + W + 15 * k + 20;
        let lo = B + W + 15 * k + 5;
        s8.push((hi, vec![], vec![]));
        let victim = alive.remove((k as usize * 7) % alive.len());
        s8.push((lo, vec![(next_id, 0xE1, 333 + k, lo - 3)], vec![victim]));
        alive.push(next_id);
        next_id += 1;
    }
    s8.push((B + W + 400, vec![], vec![]));
    run("s8_oscillation_hammer", &s8);

    // S9 — deterministic fuzz walk: thousands of commits mixing score oscillations, fresh and
    // INHERITED anchors (effective_daa far in the past, including exactly at the `score − W`
    // boundary — the consolidation-keeps-age path the static scenarios above cannot reach),
    // spends of barely-mature coins during drops, and same-outpoint re-anchors. The sequence is
    // fully determined by SEED — a failure prints the commit index, so the exact minimal replay
    // can be reconstructed by truncating the generated script.
    for seed in 1u64..=8 {
    let mut rng = 0x5EED_C014_A6E0_0000 + seed;
    let mut next = move || {
        // SplitMix64: deterministic, dependency-free.
        rng = rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut s9: Vec<CommitSpec> = Vec::new();
    let mut score = B + W; // start beyond one full window so inherited anchors can be mature
    let mut floor = score; // never drop below the highest rebuild-free depth we allow
    let mut alive: Vec<u64> = Vec::new();
    let mut next_id = 1_000u64;
    for _ in 0..1_500 {
        // Score walk: mostly forward, ~1/4 shallow re-anchors (bounded well under retention).
        if next() % 4 == 0 && score > floor + 2 {
            let max_drop = (score - floor).min(400);
            score -= 1 + next() % max_drop;
        } else {
            score += 1 + next() % 300;
            if score > floor + 5_000 {
                floor = score - 5_000;
            }
        }
        let mut adds: Vec<(u64, u8, u64, u64)> = Vec::new();
        let mut spends: Vec<u64> = Vec::new();
        // 0..=2 spends of random alive coins.
        for _ in 0..next() % 3 {
            if !alive.is_empty() {
                let idx = (next() % alive.len() as u64) as usize;
                spends.push(alive.swap_remove(idx));
            }
        }
        // ~1/10 commits re-anchor an alive coin: same outpoint spent AND re-added, new anchor.
        // Selected BEFORE the fresh deposits below, so a coin added this very commit can never be
        // picked (the runner folds spends before adds).
        if next() % 10 == 0 && !alive.is_empty() {
            let idx = (next() % alive.len() as u64) as usize;
            let id = alive[idx];
            spends.push(id);
            let eda = score.saturating_sub(next() % (W + 100));
            adds.push((id, 0xD0 + (next() % 12) as u8, 100 + next() % 900, eda.min(score)));
        }
        // 0..=2 deposits over a 12-SPK pool, anchors mixing fresh / inherited / boundary-exact.
        for _ in 0..next() % 3 {
            let spk_byte = 0xD0 + (next() % 12) as u8;
            let eda = match next() % 4 {
                0 => score - next() % 50,                        // fresh (immature)
                1 => score.saturating_sub(W + next() % 300),     // inherited, already mature
                2 => score.saturating_sub(W) + next() % 11,      // exactly around the split bound
                _ => score.saturating_sub(next() % (W + 200)),   // anywhere in (and past) the window
            };
            adds.push((next_id, spk_byte, 100 + next() % 900, eda.min(score)));
            alive.push(next_id);
            next_id += 1;
        }
        s9.push((score, adds, spends));
    }
    run(&format!("s9_fuzz_walk_seed_{seed}"), &s9);
    }

    // S10 — the POOLARIS / izzback hypothesis: a tx re-accepted by a different chain block during
    // a shallow reorg yields the SAME outpoint in diff.add AND diff.remove with the SAME inherited
    // effective_daa (block_daa_score differs, so the pair survives diff algebra). apply_age_diff's
    // add loop inserts the maturation-queue entry, then its remove loop deletes the SAME key in the
    // same batch — the coin survives in the UTXO set, the buckets stay net-consistent (silent), but
    // the queue entry is gone: when the due passes, the promotion never fires. Expected failure at
    // the LAST commit (b_imm keeps the coin the reclassification calls mature) until the loop order
    // is fixed (removes before adds).
    run(
        "s10_readd_same_anchor_kills_queue_entry",
        &[
            (B, vec![(1, 0xA1, 1_000, B)], vec![]),
            // Shallow re-anchor: same outpoint spent AND re-added with the SAME effective_daa
            // (the runner rebuilds the entry from the add tuple — block_daa_score differing in
            // production is what lets this pair reach apply_age_diff; the queue key only depends
            // on (effective_daa + W, outpoint), which is what collides).
            (B + 50, vec![(1, 0xA1, 1_000, B)], vec![1]),
            (DUE1 - 10, vec![], vec![]),
            (DUE1 + 100, vec![], vec![]),
        ],
    );
}

// ── Reward-window floor under a pruned selected-chain index ───────────────────

/// A consensus whose selected-chain index no longer reaches the header pruning points of the
/// blocks under (re)validation: the DAG is 8 blocks wide (daa outruns the chain index, the shape
/// of every freshly initialized store), and the index is pruned below chain index 5 with the
/// pruning point moved there — what a node that pruned ahead of a restart catch-up holds.
async fn pruned_floor_fixture() -> (TestConsensus, std::sync::Arc<keryx_database::prelude::DB>, Vec<JoinHandle<()>>) {
    use crate::model::stores::pruning::PruningStore;
    use crate::model::stores::selected_chain::{SelectedChainStore, SelectedChainStoreReader};
    use keryx_consensus_core::config::params::ForkActivation;
    use keryx_database::prelude::{ConnBuilder, DirectDbWriter};

    let config = ConfigBuilder::new(MAINNET_PARAMS)
        .skip_proof_of_work()
        .edit_consensus_params(|p| {
            p.pom_level_activation = ForkActivation::always();
            p.service_ledger_activation = ForkActivation::always();
            p.ratio_reward_window_daa = 200;
        })
        .build();
    let (db_lifetime, db) = keryx_database::create_temp_db!(ConnBuilder::default().with_files_limit(10));
    let (dummy_sender, _receiver) = async_channel::unbounded();
    let tc = TestConsensus::with_db(db.clone(), &config, dummy_sender);
    // Leak the tempdir guard: its drop asserts zero DB refs, which the should_panic
    // test can never guarantee mid-unwind.
    std::mem::forget(db_lifetime);
    let handles = tc.init();

    let mut prev_row = vec![config.genesis.hash];
    let mut next: u64 = 1;
    for _level in 0..40 {
        let mut row = Vec::with_capacity(8);
        for _ in 0..8 {
            let hash: Hash = next.into();
            next += 1;
            tc.add_utxo_valid_block_with_parents(hash, prev_row.clone(), vec![]).await.unwrap();
            row.push(hash);
        }
        prev_row = row;
    }

    let vp = tc.virtual_processor().clone();
    let pp_hash = {
        let mut sc = vp.selected_chain_store.write();
        let pp_hash = sc.get_by_index(5).unwrap();
        sc.prune_below_point(DirectDbWriter::new(&db), pp_hash).unwrap();
        pp_hash
    };
    vp.pruning_point_store.write().set(pp_hash, 5).unwrap();

    {
        let sc = vp.selected_chain_store.read();
        assert!(sc.get_by_index(0).is_err(), "the index the unclamped search would probe must be gone");
        assert!(sc.get_by_hash(config.genesis.hash).is_err(), "the header pruning point must be unresolvable");
        let (tip_idx, _) = sc.get_tip().unwrap();
        assert!(tip_idx < 200, "the chain index must sit inside the daa window for the floor to matter");
    }
    (tc, db, handles)
}

#[tokio::test]
async fn reward_window_floor_survives_a_pruned_header_pruning_point() {
    use crate::model::stores::headers::HeaderStoreReader;
    use crate::model::stores::selected_chain::SelectedChainStoreReader;
    use crate::pipeline::virtual_processor::utxo_validation::ProductionWindowCtx;

    let (tc, _db, _handles) = pruned_floor_fixture().await;
    let vp = tc.virtual_processor().clone();

    let sc = vp.selected_chain_store.read();
    let (tip_idx, tip) = sc.get_tip().unwrap();
    let tip_header = vp.headers_store.get_header(tip).unwrap();
    let daa_bound = tip_header.daa_score - 200;

    // The reference bottom, walked linearly over the retained index.
    let mut expected = 5;
    for i in 5..=tip_idx {
        if vp.headers_store.get_daa_score(sc.get_by_index(i).unwrap()).unwrap() <= daa_bound {
            expected = i;
        } else {
            break;
        }
    }
    drop(sc);

    match vp.production_window_ctx(tip, 0) {
        ProductionWindowCtx::OnChain { m_idx, bottom } => {
            assert_eq!(m_idx, tip_idx);
            assert_eq!(bottom, expected);
        }
        _ => panic!("the committed tip must resolve as an on-chain window"),
    }
}

#[tokio::test]
#[should_panic(expected = "pruned horizon")]
async fn reward_window_below_the_pruned_horizon_fails_loud() {
    use crate::model::stores::selected_chain::SelectedChainStoreReader;

    let (tc, _db, _handles) = pruned_floor_fixture().await;
    let vp = tc.virtual_processor().clone();

    // An early chain block: its whole daa window sits below the pruned horizon, so no local
    // computation can match what the network computed for it.
    let early = vp.selected_chain_store.read().get_by_index(6).unwrap();
    let _ = vp.production_window_ctx(early, 0);
}

#[tokio::test]
async fn cohort_window_survives_a_pruned_header_pruning_point() {
    let (tc, _db, _handles) = pruned_floor_fixture().await;
    let vp = tc.virtual_processor().clone();
    let tip = {
        use crate::model::stores::selected_chain::SelectedChainStoreReader;
        vp.selected_chain_store.read().get_tip().unwrap().1
    };
    // No tier blocks were mined, so the set is empty — the point is that the window search
    // must not probe below retention on the way there.
    assert!(vp.service_eligible_miners_windowed(tip, 0, 100).is_empty());
}

/// A cohort window that reaches below the pruned horizon arms empty instead of panicking:
/// the fold crosses this band on every fresh IBD / restart catch-up, and the events such an
/// audit could yield are already carried by the service-state transfer.
#[tokio::test]
async fn cohort_window_below_the_pruned_horizon_arms_empty() {
    use crate::model::stores::selected_chain::SelectedChainStoreReader;

    let (tc, _db, _handles) = pruned_floor_fixture().await;
    let vp = tc.virtual_processor().clone();
    let early = vp.selected_chain_store.read().get_by_index(6).unwrap();
    assert!(vp.service_eligible_miners_windowed(early, 0, 100).is_empty());
}

#[tokio::test]
async fn reward_mint_window_is_the_same_off_the_committed_chain() {
    use crate::model::stores::headers::HeaderStoreReader;
    use crate::model::stores::selected_chain::SelectedChainStoreReader;
    use keryx_consensus_core::config::params::ForkActivation;
    use keryx_consensus_core::tx::ScriptPublicKey;

    let config = ConfigBuilder::new(MAINNET_PARAMS)
        .skip_proof_of_work()
        .edit_consensus_params(|p| {
            p.reward_routing_activation = ForkActivation::new(0);
            p.finality_depth = 5;
        })
        .build();
    let mut ctx = TestContext::new(TestConsensus::new(&config));
    let mut chain = Vec::new();
    for _ in 0..20 {
        ctx.build_block_template_row(0..1).validate_and_insert_row().await;
        chain.push(*ctx.current_tips.iter().next().unwrap());
    }
    let fork = chain[9];
    let on_chain = chain[10];

    // A side block on `fork`: it never joins the committed selected chain, which stays on `chain`.
    ctx.simulated_time += ctx.consensus.params().target_time_per_block();
    let side = ctx.build_block_with_parents(vec![fork], 7, ctx.simulated_time);
    let side_hash = side.header.hash;
    ctx.validate_and_insert_block(side.to_immutable()).await;

    let vp = ctx.consensus.virtual_processor().clone();
    assert!(vp.selected_chain_store.read().get_by_hash(side_hash).is_err());
    let side_daa = vp.headers_store.get_daa_score(side_hash).unwrap();
    assert_eq!(side_daa, vp.headers_store.get_daa_score(on_chain).unwrap());
    let f = vp.finality_depth;
    assert!(side_daa > vp.headers_store.get_daa_score(fork).unwrap() && side_daa > f);

    // A finalized win whose event daa sits in the window `(fork.daa - f, side.daa - f]`.
    let spk = ScriptPublicKey::from_vec(0, vec![0x20; 34]);
    vp.service_reward_recent.write().entry(side_daa - f).or_default().push(([9u8; 32], 123_456, Some(spk.clone())));

    let off_chain = vp.service_reward_mints_for(side_hash);
    assert_eq!(off_chain, vec![(spk, 123_456)]);
    assert_eq!(vp.service_reward_mints_for(on_chain), off_chain);
}

/// Inside the post-import catch-up window the node trusts the coinbase: a window floor below
/// retention clamps at the pruning point instead of panicking (the debug dump path), while the
/// template path anchors its window at the node's own pruning point, which the selected-chain
/// index always retains — so the clamp is never consulted there.
#[tokio::test]
async fn trusted_node_clamps_the_window_floor_and_the_template_never_needs_it() {
    use crate::model::stores::headers::HeaderStoreReader;
    use crate::model::stores::production_seed::ProductionIndexSeedStore;
    use crate::model::stores::pruning::PruningStoreReader;
    use crate::model::stores::selected_chain::SelectedChainStoreReader;
    use crate::pipeline::virtual_processor::utxo_validation::ProductionWindowCtx;
    use rocksdb::WriteBatch;

    let (tc, db, _handles) = pruned_floor_fixture().await;
    let vp = tc.virtual_processor().clone();
    assert!(!vp.trust_coinbase(), "the fixture must start outside every trust window");

    // Seed the production index at the tip: fewer than a window of chain blocks since import.
    let (tip_idx, tip) = vp.selected_chain_store.read().get_tip().unwrap();
    let mut batch = WriteBatch::default();
    vp.production_index_seed_store.write().set_batch(&mut batch, tip_idx).unwrap();
    db.write(batch).unwrap();
    assert!(vp.trust_coinbase());

    // The window below the pruned horizon now clamps at the pruning point instead of panicking.
    let pp = vp.pruning_point_store.read().pruning_point().unwrap();
    let pp_idx = vp.selected_chain_store.read().get_by_hash(pp).unwrap();
    let early = vp.selected_chain_store.read().get_by_index(6).unwrap();
    match vp.production_window_ctx(early, 0) {
        ProductionWindowCtx::OnChain { bottom, .. } => assert_eq!(bottom, pp_idx),
        _ => panic!("a committed chain block must resolve as an on-chain window"),
    }

    // The template's window is anchored at the node's own pruning point, always retained: the
    // fallible floor resolves for any daa bound, so the clamp above is never reached there.
    let sc = vp.selected_chain_store.read();
    let tip_daa = vp.headers_store.get_daa_score(tip).unwrap();
    assert_eq!(vp.window_floor_in_retention(&*sc, pp, pp, tip_daa.saturating_sub(vp.ratio_reward_window_daa)), Some(pp_idx));
    assert_eq!(vp.window_floor_in_retention(&*sc, pp, pp, 0), Some(pp_idx));
}

/// Every chain block opening a finality epoch gets its service-ledger snapshot persisted, and
/// nothing else does.
#[tokio::test]
async fn ledger_snapshots_are_persisted_at_pruning_samples() {
    use keryx_consensus_core::collateral::ServiceLedgerSnapshot;
    use keryx_consensus_core::config::params::ForkActivation;

    let mut params = MAINNET_PARAMS;
    params.pom_v3_activation = ForkActivation::always();
    params.blockrate.finality_depth = 10;
    let config = ConfigBuilder::new(params).skip_proof_of_work().build();
    let tc = TestConsensus::new(&config);
    let handles = tc.init();

    let mut parent = config.genesis.hash;
    for n in 1..=45u64 {
        let hash: Hash = n.into();
        tc.add_utxo_valid_block_with_parents(hash, vec![parent], vec![]).await.unwrap();
        parent = hash;
    }

    let vp = tc.virtual_processor().clone();
    let expected: Vec<Hash> = (1..=45u64).map(Hash::from).filter(|h| vp.is_pruning_sample_block(*h)).collect();
    assert_eq!(expected, vec![Hash::from(10u64), Hash::from(20u64), Hash::from(30u64), Hash::from(40u64)]);
    for h in expected.iter() {
        let bytes = vp.service_ledger_snapshot_store.get(*h).unwrap().expect("a sample must carry its snapshot");
        assert_eq!(ServiceLedgerSnapshot::from_bytes(&bytes).unwrap().to_bytes(), bytes);
    }
    let mut stored: Vec<Hash> = vp.service_ledger_snapshot_store.entries().into_iter().map(|(h, _)| h).collect();
    stored.sort();
    assert_eq!(stored, expected);
    assert_eq!(vp.service_ledger_hash_at(config.genesis.hash), Some(ServiceLedgerSnapshot::default().hash()));
    assert_eq!(vp.service_ledger_hash_at(Hash::from(11u64)), None);

    tc.shutdown(handles);
}

/// Below the pruning point, a cohort walk past the ledger gate reads the producers carried by the
/// imported snapshot instead of arming empty.
#[tokio::test]
async fn cohort_below_the_pruned_horizon_reads_the_imported_producers() {
    use crate::model::stores::headers::HeaderStoreReader;
    use crate::model::stores::pruning::PruningStoreReader;
    use crate::model::stores::selected_chain::SelectedChainStoreReader;

    let (tc, _db, _handles) = pruned_floor_fixture().await;
    let vp = tc.virtual_processor().clone();
    let early = vp.selected_chain_store.read().get_by_index(6).unwrap();
    let pp = vp.pruning_point_store.read().pruning_point().unwrap();
    let pp_daa = vp.headers_store.get_daa_score(pp).unwrap();
    let id = Hash::from_bytes([0xAAu8; 32]);
    let escrow = Hash::from_bytes([0xBBu8; 32]);

    assert!(vp.service_eligible_miners_windowed(early, 0, 100).is_empty());
    *vp.service_imported_producers.write() = vec![(pp_daa, id, 0, escrow), (pp_daa, id, 1, escrow)];
    assert_eq!(vp.service_eligible_miners_windowed(early, 0, 100), vec![(id, escrow)]);
    assert_eq!(vp.service_eligible_miners_windowed(early, 1, 100), vec![(id, escrow)]);
    assert!(vp.service_eligible_miners_windowed(early, 2, 100).is_empty());
}
