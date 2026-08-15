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

/// Historical commitments carry a multiset residue that no rebuilt UTXO set reproduces;
/// combining a rebuilt set with this constant restores the committed lineage.
pub const COMMITMENT_RESIDUE: [u8; keryx_muhash::SERIALIZED_MUHASH_SIZE] = [
    0xe1, 0xb8, 0xa0, 0xe4, 0x99, 0x1f, 0xd5, 0x2e, 0x68, 0xec, 0x7a, 0xb2, 0x6b, 0xad, 0xab, 0xc3,
    0x3f, 0x4b, 0x28, 0xb6, 0xda, 0x51, 0x4a, 0x62, 0x8a, 0xc5, 0xf8, 0xfa, 0xf6, 0xa0, 0xac, 0x6b,
    0x35, 0xcf, 0x8f, 0x6a, 0xad, 0x0a, 0xdc, 0xd8, 0x5d, 0xb9, 0xe6, 0xb2, 0x61, 0x33, 0x82, 0xf5,
    0x65, 0x8a, 0x63, 0xdc, 0xf5, 0x21, 0xbd, 0x64, 0xe4, 0x27, 0x7e, 0x54, 0x3c, 0xce, 0xe5, 0xaf,
    0x20, 0x3a, 0x6f, 0xff, 0x6e, 0xdd, 0x49, 0xec, 0x09, 0xdc, 0xa2, 0xcf, 0xa0, 0x67, 0x81, 0x85,
    0x04, 0xbe, 0x0a, 0x5e, 0x55, 0x37, 0xf7, 0x66, 0xe3, 0x47, 0x54, 0x33, 0x8e, 0x8b, 0x64, 0x72,
    0xc8, 0xbd, 0xcb, 0x92, 0xa4, 0x75, 0xc3, 0xb3, 0x28, 0xd2, 0x5d, 0x4c, 0xda, 0xf1, 0xbe, 0x4d,
    0xc6, 0x9e, 0xd7, 0x99, 0x9b, 0xbf, 0xcf, 0xe7, 0xe4, 0xdc, 0x3a, 0x7d, 0x9d, 0x59, 0x19, 0xfb,
    0x50, 0xf7, 0x9f, 0x49, 0x79, 0x3b, 0xed, 0xff, 0xbc, 0x75, 0xb3, 0x89, 0xc8, 0xa8, 0xf5, 0x0f,
    0xbf, 0x4b, 0x02, 0xa0, 0x6d, 0x14, 0xd3, 0x24, 0x70, 0xd2, 0x61, 0x56, 0x37, 0x3b, 0xfe, 0xb4,
    0xe6, 0xb8, 0xd7, 0x18, 0xe3, 0x0a, 0xcd, 0x7e, 0x9f, 0x96, 0x2d, 0xab, 0x63, 0x2c, 0x8f, 0x0a,
    0xd1, 0xe9, 0x3b, 0x86, 0xcf, 0x08, 0xef, 0xd1, 0xbb, 0x30, 0x09, 0x40, 0x4a, 0xa0, 0x6b, 0x00,
    0x22, 0x58, 0x00, 0x33, 0xb8, 0xc2, 0x18, 0x53, 0x41, 0x76, 0x41, 0x64, 0x88, 0x43, 0x8a, 0x79,
    0xe0, 0x7b, 0x7c, 0xfb, 0x5a, 0x6f, 0xfe, 0x72, 0xc8, 0x48, 0x48, 0x5d, 0x1a, 0xbe, 0xb3, 0x0a,
    0x56, 0xb3, 0xe1, 0x2d, 0x9b, 0x00, 0x80, 0xad, 0xa8, 0x1d, 0xa5, 0x2d, 0xb1, 0x25, 0xe4, 0x01,
    0x38, 0x0f, 0x3c, 0xc0, 0xd2, 0xa3, 0x58, 0x26, 0x40, 0xe4, 0x6b, 0xa3, 0xea, 0x9a, 0x1d, 0x5f,
    0xea, 0x6d, 0x62, 0x20, 0xd2, 0x9e, 0x4a, 0x8b, 0x98, 0x56, 0x84, 0xad, 0xb7, 0x8e, 0xf1, 0x1c,
    0xdf, 0x22, 0x8c, 0xd1, 0x52, 0x4e, 0x9f, 0x63, 0x72, 0xca, 0xc9, 0x5e, 0x19, 0xe6, 0x9f, 0x68,
    0x95, 0x69, 0xe3, 0x37, 0x78, 0xb2, 0x67, 0xfc, 0x04, 0x47, 0xe4, 0x94, 0xeb, 0xec, 0x8a, 0xc7,
    0xcf, 0xa1, 0xae, 0xd8, 0x70, 0x80, 0x76, 0x08, 0x5e, 0xd4, 0xa2, 0x4e, 0xc7, 0x39, 0x57, 0x72,
    0x19, 0x88, 0x42, 0x99, 0x49, 0x9b, 0x91, 0x6a, 0x2b, 0x46, 0xbb, 0xf3, 0x21, 0xb5, 0x6d, 0x10,
    0x07, 0xc8, 0xc0, 0x61, 0xa8, 0x55, 0x58, 0x6e, 0x72, 0x81, 0xeb, 0x99, 0x61, 0xbb, 0x47, 0xbc,
    0x7d, 0xc0, 0xfb, 0x20, 0x9c, 0x3c, 0x22, 0x9c, 0xf1, 0x96, 0x3a, 0x65, 0x25, 0xfb, 0x15, 0x66,
    0x2d, 0x26, 0x58, 0xaa, 0x44, 0x5a, 0xb4, 0xae, 0xf6, 0x62, 0xa2, 0x3b, 0xfe, 0x11, 0x58, 0xf1,
];

/// Returns `multiset` combined with [`COMMITMENT_RESIDUE`].
pub fn with_commitment_residue(multiset: &MuHash) -> MuHash {
    let mut out = multiset.clone();
    out.combine(&MuHash::deserialize(COMMITMENT_RESIDUE).expect("pinned residue is a valid group element"));
    out
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
