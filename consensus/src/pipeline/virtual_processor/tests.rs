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

// OPoI slashing removed (v1.2.3): the slash-behavior tests (fraud→slash, honest→no-slash,
// unknown→no-slash, outside-window→no-slash) were dropped together with the slashing mechanism.
// Escrows are now always spendable; there is no slash state to assert.

// ── tier-reward E2E (full pipeline: commit → store → coinbase split) ──────────

/// End-to-end: a merged block's miner cut in its merging block's coinbase is scaled by the
/// merged block's cryptographically-proven PoM tier — persisted at body commit (`pom_tier_store`),
/// read back by the virtual processor when it builds the coinbase. The floor tier (0, −30 %) pays
/// its miner exactly 70 % of what the top tier (3, 0 %) pays, while the total block reward is
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
    let (total_floor, miner_floor) = payout_for_tier(0).await; // −30 %

    assert!(miner_top > 0, "top-tier block must pay its miner");
    assert_eq!(total_top, total_floor, "tier penalty must not change the total block reward");
    assert_eq!(
        miner_floor,
        miner_top * TIER_REWARD_BPS[0] / TIER_REWARD_BPS_DIVISOR,
        "floor-tier miner must get exactly 70 % of the top-tier cut"
    );
    assert!(miner_floor < miner_top, "serving a heavier model must pay the miner strictly more");
}
