use crate::{
    consensus::test_consensus::TestConsensus,
    model::{
        services::reachability::ReachabilityService,
        stores::ai_slash::AiResponseStoreReader,
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

// ── OPoI synthetic-liveness E2E (Level-1, option C) ───────────────────────────

/// Coinbase miner_data carrying `/escrow:<pk>`, the declared `/ai:cap:<model>`, and
/// optionally a `/live:<H>` reference.
fn liveness_miner_data(pk: &[u8; 32], model: [u8; 32], live: Option<Hash>) -> keryx_consensus_core::coinbase::MinerData {
    use keryx_consensus_core::tx::ScriptPublicKey;
    let mut extra = format!("test/escrow:{}/ai:cap:{}", hex::encode(pk), hex::encode(model));
    if let Some(h) = live {
        extra.push_str(&format!("/live:{}", hex::encode(h.as_bytes())));
    }
    keryx_consensus_core::coinbase::MinerData::new(ScriptPublicKey::from_vec(0, vec![]), extra.into_bytes())
}

/// A payload-only v2 AiResponse answering the synthetic task for `(epoch, pk, model)`.
fn synthetic_answer_tx(epoch: u64, pk: &[u8; 32], model: [u8; 32]) -> Transaction {
    use keryx_inference::synthetic::{derive_synthetic_request, synthetic_seed};
    let seed = synthetic_seed(epoch, pk);
    let req = derive_synthetic_request(&seed, &[model], epoch).unwrap();
    let mut request_hash = [0u8; 32];
    request_hash.copy_from_slice(&blake2b_simd::blake2b(&req.serialize()).as_bytes()[..32]);
    let mut cid = [0u8; 34];
    cid[0] = 0x12;
    cid[1] = 0x20;
    let resp = AiResponsePayload::new_v2(request_hash, 0, cid, 16, model, [0u8; 32]);
    Transaction::new(TX_VERSION, vec![], vec![], 0, SUBNETWORK_ID_AI_RESPONSE, 0, resp.serialize())
}

/// With the gate active: a self-contained synthetic answer and a `/live:` ancestor
/// reference both keep a block on the chain, while a block proving neither is
/// disqualified — "no inference = no mining", now enforced at consensus.
#[tokio::test]
async fn synthetic_liveness_enforcement_e2e() {
    use keryx_consensus_core::config::params::ForkActivation;
    let mut params = MAINNET_PARAMS;
    params.synthetic_liveness_activation = ForkActivation::always();
    params.opoi_v2_activation = ForkActivation::always(); // allow 142-byte v2 AiResponse payloads
    let config = ConfigBuilder::new(params).skip_proof_of_work().build();
    let tc = TestConsensus::new(&config);
    let handles = tc.init();
    let genesis = config.genesis.hash;
    let pk = [0x11u8; 32];
    let model = [0x22u8; 32];

    // (a) self-contained answer for epoch 0 → stays on the chain.
    let a: Hash = 1u64.into();
    tc.validate_and_insert_block(
        tc.build_utxo_valid_block_with_parents(a, vec![genesis], liveness_miner_data(&pk, model, None), vec![synthetic_answer_tx(0, &pk, model)])
            .to_immutable(),
    )
    .virtual_state_task
    .await
    .unwrap();
    assert_eq!(tc.get_block_status(a), Some(BlockStatus::StatusUTXOValid), "self-contained answer must be accepted");

    // (b) no answer, but /live:<A> referencing the annotated ancestor → accepted.
    let b: Hash = 2u64.into();
    tc.validate_and_insert_block(
        tc.build_utxo_valid_block_with_parents(b, vec![a], liveness_miner_data(&pk, model, Some(a)), vec![]).to_immutable(),
    )
    .virtual_state_task
    .await
    .unwrap();
    assert_eq!(tc.get_block_status(b), Some(BlockStatus::StatusUTXOValid), "/live: ancestor reference must be accepted");

    // (c) no answer and no /live: → disqualified from the virtual chain.
    let c: Hash = 3u64.into();
    tc.validate_and_insert_block(
        tc.build_utxo_valid_block_with_parents(c, vec![a], liveness_miner_data(&pk, model, None), vec![]).to_immutable(),
    )
    .virtual_state_task
    .await
    .unwrap();
    assert_eq!(
        tc.get_block_status(c),
        Some(BlockStatus::StatusDisqualifiedFromChain),
        "a block proving no synthetic liveness must be disqualified"
    );

    tc.shutdown(handles);
}

// ── OPoI tier-reward E2E (miner-cut scaling by declared model tier) ───────────

/// With the gate active, only the *miner's* cut of a merged block's subsidy is
/// scaled by the highest model tier it declared in `ai:cap`; escrow, R&D and the
/// total block reward are untouched, the shortfall being burned. A block merging
/// a floor-tier (TinyLlama) ancestor pays its miner only 85% of the 75% cut vs a
/// top-tier (Qwen3-235B) ancestor, yet both coinbases pay out the same total.
#[tokio::test]
async fn tier_reward_scales_miner_cut_e2e() {
    use keryx_consensus_core::config::params::{
        ForkActivation, QWEN3_235B_MODEL_ID, TINYLLAMA_MODEL_ID, TIER_REWARD_BPS, TIER_REWARD_BPS_DIVISOR,
    };
    use keryx_consensus_core::tx::ScriptPublicKey;

    // Inserts A declaring `model` in ai:cap, then builds child B and returns
    // (total coinbase payout, the part going to A's miner SPK).
    async fn payout_for_tier(model: [u8; 32]) -> (u64, u64) {
        let mut params = MAINNET_PARAMS;
        params.tier_reward_activation = ForkActivation::always();
        params.opoi_v2_activation = ForkActivation::always(); // ai:cap context, no AiResponse needed
        let config = ConfigBuilder::new(params).skip_proof_of_work().build();
        let tc = TestConsensus::new(&config);
        let handles = tc.init();
        let genesis = config.genesis.hash;
        let pk = [0x11u8; 32];

        let a: Hash = 1u64.into();
        let status = tc
            .validate_and_insert_block(
                tc.build_utxo_valid_block_with_parents(a, vec![genesis], liveness_miner_data(&pk, model, None), vec![])
                    .to_immutable(),
            )
            .virtual_state_task
            .await
            .unwrap();
        assert_eq!(status, BlockStatus::StatusUTXOValid, "merged block A must be UTXO-valid");

        let b = tc.build_utxo_valid_block_with_parents(2u64.into(), vec![a], liveness_miner_data(&pk, model, None), vec![]);
        let miner_spk = ScriptPublicKey::from_vec(0, vec![]); // liveness_miner_data uses an empty SPK
        let outs = &b.transactions[0].outputs;
        let total: u64 = outs.iter().map(|o| o.value).sum();
        let miner: u64 = outs.iter().filter(|o| o.script_public_key == miner_spk).map(|o| o.value).sum();
        tc.shutdown(handles);
        (total, miner)
    }

    let (total_top, miner_top) = payout_for_tier(QWEN3_235B_MODEL_ID).await; // rank 5 → 100%
    let (total_floor, miner_floor) = payout_for_tier(TINYLLAMA_MODEL_ID).await; // rank 0 → 85%

    assert!(miner_top > 0, "top-tier block must still pay its miner");
    // Total block reward is identical across tiers — only the split changes (delta burned).
    assert_eq!(total_top, total_floor, "tier penalty must not change the total block reward");
    // Top tier is the 100% reference, so the floor miner gets exactly 85% of the top miner's cut.
    assert_eq!(miner_floor, miner_top * TIER_REWARD_BPS[0] / TIER_REWARD_BPS_DIVISOR, "floor miner must get 85% of the cut");
    assert!(miner_floor < miner_top, "serving a heavier model must pay the miner strictly more");
}

// OPoI slashing removed (v1.2.3): the slash-behavior tests (fraud→slash, honest→no-slash,
// unknown→no-slash, outside-window→no-slash) were dropped together with the slashing mechanism.
// Escrows are now always spendable; there is no slash state to assert.
