use crate::{
    coin_age::assign_output_effective_daa,
    hashing::HasherExtensions,
    tx::{TransactionOutpoint, UtxoEntry, VerifiableTransaction},
};
use keryx_hashes::HasherBase;
use keryx_muhash::MuHash;

/// Muhash (UTXO commitment) serialization, era-gated for the coin-age hard fork (holder-reward
/// v3): `coin_age_active` selects whether `effective_daa` participates in the element encoding.
/// Blocks below `coin_age_activation` commit WITHOUT the field — byte-identical to the legacy
/// commitment — while blocks at/after it commit WITH it, pinning the age anchors into the header
/// `utxo_commitment`. Every caller derives the flag from the POV context it is hashing for (the
/// rewarding block, the pruning point being imported/verified, or genesis).
pub trait MuHashExtensions {
    fn add_transaction(&mut self, tx: &impl VerifiableTransaction, block_daa_score: u64, coin_age_active: bool);
    fn add_utxo(&mut self, outpoint: &TransactionOutpoint, entry: &UtxoEntry, coin_age_active: bool);
    fn from_transaction(tx: &impl VerifiableTransaction, block_daa_score: u64, coin_age_active: bool) -> Self;
    fn from_utxo(outpoint: &TransactionOutpoint, entry: &UtxoEntry, coin_age_active: bool) -> Self;
}

impl MuHashExtensions for MuHash {
    fn add_transaction(&mut self, tx: &impl VerifiableTransaction, block_daa_score: u64, coin_age_active: bool) {
        // Coin-age era: the hashed output entries must carry the SAME FIFO-inherited anchors as
        // the stored ones (`UtxoDiff::add_transaction`), otherwise the commitment and the UTXO
        // set would disagree. Same rule, same inputs, same order — byte-identical results.
        let anchors: Option<Vec<u64>> = if coin_age_active {
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
            write_utxo(&mut writer, entry, &input.previous_outpoint, coin_age_active);
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
            self.add_utxo(&outpoint, &entry, coin_age_active);
        }
    }

    fn add_utxo(&mut self, outpoint: &TransactionOutpoint, entry: &UtxoEntry, coin_age_active: bool) {
        let mut writer = self.add_element_builder();
        write_utxo(&mut writer, entry, outpoint, coin_age_active);
        writer.finalize();
    }

    fn from_transaction(tx: &impl VerifiableTransaction, block_daa_score: u64, coin_age_active: bool) -> Self {
        let mut mh = Self::new();
        mh.add_transaction(tx, block_daa_score, coin_age_active);
        mh
    }

    fn from_utxo(outpoint: &TransactionOutpoint, entry: &UtxoEntry, coin_age_active: bool) -> Self {
        let mut mh = Self::new();
        mh.add_utxo(outpoint, entry, coin_age_active);
        mh
    }
}

fn write_utxo(writer: &mut impl HasherBase, entry: &UtxoEntry, outpoint: &TransactionOutpoint, coin_age_active: bool) {
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
    // Coin-age era (H4): the age anchor joins the commitment, appended last so the pre-fork
    // element encoding stays byte-identical below the gate.
    if coin_age_active {
        writer.update(entry.effective_daa.to_le_bytes());
    }
}
