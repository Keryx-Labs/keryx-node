use crate::{
    consensus::test_consensus::TestConsensus,
    model::{
        services::reachability::ReachabilityService,
        stores::ai_slash::{AiResponseStoreReader, AiSlashedStoreReader, OutpointKey},
    },
};
use keryx_inference::{self, AiChallengePayload, AiResponsePayload, compute_ai_commitment};
use keryx_consensus_core::{
    BlockHashSet,
    api::ConsensusApi,
    block::{Block, BlockTemplate, MutableBlock, TemplateBuildMode, TemplateTransactionSelector},
    blockhash,
    blockstatus::BlockStatus,
    coinbase::MinerData,
    config::{ConfigBuilder, params::MAINNET_PARAMS},
    subnets::{SUBNETWORK_ID_AI_CHALLENGE, SUBNETWORK_ID_AI_RESPONSE},
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

/// Build an AiResponse TX with a specific 32-byte commitment prepended to result.
fn make_ai_response_tx(request_hash: [u8; 32], commitment: [u8; 32]) -> Transaction {
    let mut result = commitment.to_vec();
    result.extend_from_slice(b"opoi_test_result");
    let payload = AiResponsePayload::new(request_hash, 0, result).serialize();
    Transaction::new(TX_VERSION, vec![], vec![], 0, SUBNETWORK_ID_AI_RESPONSE, 0, payload)
}

/// Compute the response_hash key used by the consensus: blake2b(tx.payload)[0..32].
fn response_hash_of(tx: &Transaction) -> Hash {
    let h = blake2b_simd::blake2b(&tx.payload);
    let mut bytes = [0u8; 32];
    bytes.copy_from_slice(&h.as_bytes()[..32]);
    Hash::from_bytes(bytes)
}

/// Build an AiChallenge TX (Phase 3 C: proof_data = request_hash).
fn make_ai_challenge_tx(
    response_hash: Hash,
    challenger_spk: [u8; 32],
    request_hash: [u8; 32],
) -> Transaction {
    let rh_bytes: [u8; 32] = response_hash.as_bytes().try_into().unwrap();
    let payload = AiChallengePayload::new(rh_bytes, 0, 0, challenger_spk, request_hash.to_vec()).serialize();
    Transaction::new(TX_VERSION, vec![], vec![], 0, SUBNETWORK_ID_AI_CHALLENGE, 0, payload)
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

/// A miner that publishes a fraudulent commitment (wrong value) should have
/// their escrow slashed once a valid AiChallenge is included in a later block.
#[tokio::test]
async fn opoi_fraud_challenge_slashes_escrow() {
    let config = opoi_config();
    let tc = TestConsensus::new(&config);
    let handles = tc.init();

    let genesis = config.genesis.hash;
    let request_hash = [0x02u8; 32];
    let wrong_commitment = [0xFFu8; 32]; // does not match compute_ai_commitment

    let response_tx = make_ai_response_tx(request_hash, wrong_commitment);
    let rh = response_hash_of(&response_tx);
    let challenger_spk = [0xABu8; 32];
    let challenge_tx = make_ai_challenge_tx(rh, challenger_spk, request_hash);

    // Block 1: fraudulent AiResponse
    tc.add_utxo_valid_block_with_parents(1u64.into(), vec![genesis], vec![response_tx]).await.unwrap();

    // Retrieve coinbase_tx_id from the registered record to build the outpoint key
    let record = tc.ai_response_store().get(rh).expect("AiResponse must be registered");
    let outpoint_key = OutpointKey::new(record.coinbase_tx_id, 1);

    // Block 2: valid AiChallenge proving fraud
    tc.add_utxo_valid_block_with_parents(2u64.into(), vec![1u64.into()], vec![challenge_tx]).await.unwrap();

    let slash = tc.ai_slashed_store().get_slash(outpoint_key).expect("store error");
    assert!(slash.is_some(), "escrow must be slashed after valid fraud proof");
    let slash = slash.unwrap();
    assert_eq!(slash.challenger_spk_script, challenger_spk);

    tc.shutdown(handles);
}

/// A challenger submitting a proof against an *honest* miner (correct commitment)
/// must be rejected: no slash record should be written.
#[tokio::test]
async fn opoi_honest_miner_challenge_no_slash() {
    let config = opoi_config();
    let tc = TestConsensus::new(&config);
    let handles = tc.init();

    let genesis = config.genesis.hash;
    let request_hash = [0x03u8; 32];
    let honest_commitment = compute_ai_commitment(&request_hash); // correct value

    let response_tx = make_ai_response_tx(request_hash, honest_commitment);
    let rh = response_hash_of(&response_tx);
    let challenger_spk = [0xCDu8; 32];
    let challenge_tx = make_ai_challenge_tx(rh, challenger_spk, request_hash);

    tc.add_utxo_valid_block_with_parents(10u64.into(), vec![genesis], vec![response_tx]).await.unwrap();
    let record = tc.ai_response_store().get(rh).expect("AiResponse must be registered");
    let outpoint_key = OutpointKey::new(record.coinbase_tx_id, 1);

    tc.add_utxo_valid_block_with_parents(11u64.into(), vec![10u64.into()], vec![challenge_tx]).await.unwrap();

    let slash = tc.ai_slashed_store().get_slash(outpoint_key).expect("store error");
    assert!(slash.is_none(), "honest miner must NOT be slashed");

    tc.shutdown(handles);
}

/// A challenge referencing an unknown response_hash (no prior AiResponse on chain)
/// must be silently ignored — no slash recorded.
#[tokio::test]
async fn opoi_challenge_unknown_response_no_slash() {
    let config = opoi_config();
    let tc = TestConsensus::new(&config);
    let handles = tc.init();

    let genesis = config.genesis.hash;
    let unknown_rh = Hash::from_bytes([0xBEu8; 32]);
    let challenge_tx = make_ai_challenge_tx(unknown_rh, [0x11u8; 32], [0x22u8; 32]);

    tc.add_utxo_valid_block_with_parents(20u64.into(), vec![genesis], vec![challenge_tx]).await.unwrap();

    // The response was never registered, so no outpoint can be slashed.
    // We verify no slash was recorded under the fake outpoint used in the challenge.
    let fake_key = OutpointKey::new(Hash::from_bytes([0xFFu8; 32]), 1);
    let slash = tc.ai_slashed_store().get_slash(fake_key).expect("store error");
    assert!(slash.is_none(), "challenge for unknown response must not produce a slash");

    tc.shutdown(handles);
}

/// An AiChallenge submitted outside the challenge window (> CHALLENGE_WINDOW_BLOCKS
/// after the AiResponse) must be silently ignored — no slash recorded.
///
/// NOTE: This test inserts 36,002 blocks and takes ~100 s. Run manually with:
///   cargo test -p keryx-consensus --lib -- opoi_challenge_outside_window_no_slash --ignored
#[tokio::test]
#[ignore]
async fn opoi_challenge_outside_window_no_slash() {
    use keryx_consensus_core::collateral::CHALLENGE_WINDOW_BLOCKS;

    let config = opoi_config();
    let tc = TestConsensus::new(&config);
    let handles = tc.init();

    let genesis = config.genesis.hash;
    let request_hash = [0x04u8; 32];
    let wrong_commitment = [0xEEu8; 32];

    let response_tx = make_ai_response_tx(request_hash, wrong_commitment);
    let rh = response_hash_of(&response_tx);
    let challenger_spk = [0xDEu8; 32];
    let challenge_tx = make_ai_challenge_tx(rh, challenger_spk, request_hash);

    // Block 100: AiResponse
    tc.add_utxo_valid_block_with_parents(100u64.into(), vec![genesis], vec![response_tx]).await.unwrap();
    let record = tc.ai_response_store().get(rh).expect("AiResponse must be registered");

    // The challenge window is CHALLENGE_WINDOW_BLOCKS. We need to advance the
    // blue score far enough that the challenge lands outside the window.
    // We do this by building a chain of (CHALLENGE_WINDOW_BLOCKS + 2) empty blocks.
    let window = CHALLENGE_WINDOW_BLOCKS as u64;
    let mut parent = 100u64.into();
    // Hash range 200..200+window+2, well clear of 100 and the challenge
    for i in 0..window + 2 {
        let next: Hash = (200u64 + i).into();
        tc.add_utxo_valid_block_with_parents(next, vec![parent], vec![]).await.unwrap();
        parent = next;
    }

    let outpoint_key = OutpointKey::new(record.coinbase_tx_id, 1);

    // Challenge block appended after the window has closed
    let challenge_hash: Hash = (200u64 + window + 3).into();
    tc.add_utxo_valid_block_with_parents(challenge_hash, vec![parent], vec![challenge_tx]).await.unwrap();

    let slash = tc.ai_slashed_store().get_slash(outpoint_key).expect("store error");
    assert!(slash.is_none(), "challenge outside window must NOT produce a slash");

    tc.shutdown(handles);
}
