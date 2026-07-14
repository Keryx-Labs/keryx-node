use crate::{
    coin_age::assign_output_effective_daa,
    config::params::ForkActivation,
    hashing::HasherExtensions,
    tx::{TransactionOutpoint, UtxoEntry, VerifiableTransaction},
};
use keryx_hashes::HasherBase;
use keryx_muhash::MuHash;

/// Muhash (UTXO commitment) serialization, era-gated for the coin-age hard fork (holder-reward
/// v3). The gate is PER COIN — `effective_daa` joins the element encoding iff the COIN was
/// created at/after `coin_age_activation` (`entry.block_daa_score`), NOT per POV block: the
/// multiset is carried incrementally across blocks, so a removal must hash byte-identically to
/// the addition that put the coin in — a POV-era gate would break the commitment on the first
/// post-fork spend of a pre-fork coin. Pre-fork coins carry `effective_daa == block_daa_score`
/// by invariant, so excluding the field loses nothing; post-fork coins pin their FIFO anchors
/// into the header `utxo_commitment`. Callers pass `params.coin_age_activation` verbatim.
pub trait MuHashExtensions {
    fn add_transaction(&mut self, tx: &impl VerifiableTransaction, block_daa_score: u64, coin_age_activation: ForkActivation);
    fn add_utxo(&mut self, outpoint: &TransactionOutpoint, entry: &UtxoEntry, coin_age_activation: ForkActivation);
    fn from_transaction(tx: &impl VerifiableTransaction, block_daa_score: u64, coin_age_activation: ForkActivation) -> Self;
    fn from_utxo(outpoint: &TransactionOutpoint, entry: &UtxoEntry, coin_age_activation: ForkActivation) -> Self;
}

impl MuHashExtensions for MuHash {
    fn add_transaction(&mut self, tx: &impl VerifiableTransaction, block_daa_score: u64, coin_age_activation: ForkActivation) {
        // Coin-age era of the NEW outputs (created at `block_daa_score`): the hashed entries must
        // carry the SAME FIFO-inherited anchors as the stored ones (`UtxoDiff::add_transaction`),
        // otherwise the commitment and the UTXO set would disagree. Same rule, same inputs, same
        // order — byte-identical results.
        let anchors: Option<Vec<u64>> = if coin_age_activation.is_active(block_daa_score) {
            let inputs: Vec<_> = tx
                .populated_inputs()
                .map(|(_, entry)| (&entry.script_public_key, entry.amount, entry.effective_daa))
                .collect();
            let outputs: Vec<_> = tx.outputs().iter().map(|output| (&output.script_public_key, output.value)).collect();
            Some(assign_output_effective_daa(&inputs, &outputs, block_daa_score))
        } else {
            None
        };

        let tx_id = tx.id();
        for (input, entry) in tx.populated_inputs() {
            let mut writer = self.remove_element_builder();
            write_utxo(&mut writer, entry, &input.previous_outpoint, coin_age_activation);
            writer.finalize();
        }
        for (i, output) in tx.outputs().iter().enumerate() {
            let outpoint = TransactionOutpoint::new(tx_id, i as u32);
            let entry = match &anchors {
                Some(anchors) => {
                    UtxoEntry::new_aged(output.value, output.script_public_key.clone(), block_daa_score, tx.is_coinbase(), anchors[i])
                }
                None => UtxoEntry::new(output.value, output.script_public_key.clone(), block_daa_score, tx.is_coinbase()),
            };
            self.add_utxo(&outpoint, &entry, coin_age_activation);
        }
    }

    fn add_utxo(&mut self, outpoint: &TransactionOutpoint, entry: &UtxoEntry, coin_age_activation: ForkActivation) {
        let mut writer = self.add_element_builder();
        write_utxo(&mut writer, entry, outpoint, coin_age_activation);
        writer.finalize();
    }

    fn from_transaction(tx: &impl VerifiableTransaction, block_daa_score: u64, coin_age_activation: ForkActivation) -> Self {
        let mut mh = Self::new();
        mh.add_transaction(tx, block_daa_score, coin_age_activation);
        mh
    }

    fn from_utxo(outpoint: &TransactionOutpoint, entry: &UtxoEntry, coin_age_activation: ForkActivation) -> Self {
        let mut mh = Self::new();
        mh.add_utxo(outpoint, entry, coin_age_activation);
        mh
    }
}

fn write_utxo(writer: &mut impl HasherBase, entry: &UtxoEntry, outpoint: &TransactionOutpoint, coin_age_activation: ForkActivation) {
    writer
        // Outpoint
        .update(outpoint.transaction_id)
        .update(outpoint.index.to_le_bytes())
        // Utxo entry
        .update(entry.block_daa_score.to_le_bytes())
        .update(entry.amount.to_le_bytes())
        .write_bool(entry.is_coinbase)
        .update(entry.script_public_key.version().to_le_bytes())
        .write_var_bytes(entry.script_public_key.script());
    // Coin-age era (H4), PER COIN: the age anchor joins the commitment iff this coin was created
    // at/after the fork — appended last so the pre-fork element encoding stays byte-identical,
    // and removal always hashes exactly like the addition did (see the trait doc).
    if coin_age_activation.is_active(entry.block_daa_score) {
        writer.update(entry.effective_daa.to_le_bytes());
    }
}
