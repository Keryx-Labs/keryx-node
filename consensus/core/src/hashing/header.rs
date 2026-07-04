use super::HasherExtensions;
use crate::header::Header;
use keryx_hashes::{Hash, HasherBase};

/// Returns the header hash using the provided nonce+timestamp instead of those in the header.
///
/// This is the pre-PoW form: `pom_final_state` is NEVER written here. The PoM walk seed
/// derives from this hash, and the walk produces `final_state` — including the field would
/// be circular. It also keeps the pre-PoW byte stream identical across `pom_level_activation`,
/// so miners' pre-PoW hashing is untouched by the fork.
#[inline]
pub fn hash_override_nonce_time(header: &Header, nonce: u64, timestamp: u64) -> Hash {
    hash_internal(header, nonce, timestamp, None)
}

/// Returns the header hash. At/after `pom_level_activation` (gated on the header's own
/// `daa_score`) the block hash commits to `pom_final_state`, making the block level an
/// immutable, header-only derivable property. Pre-fork hashes are byte-identical to legacy.
pub fn hash(header: &Header) -> Hash {
    let pom_final_state =
        if crate::pom::pom_level_active(header.daa_score) { Some(header.pom_final_state) } else { None };
    hash_internal(header, header.nonce, header.timestamp, pom_final_state)
}

#[inline]
fn hash_internal(header: &Header, nonce: u64, timestamp: u64, pom_final_state: Option<u64>) -> Hash {
    let mut hasher = keryx_hashes::BlockHash::new();
    hasher.update(header.version.to_le_bytes()).write_len(header.parents_by_level.expanded_len()); // Write the number of parent levels

    // Write parents at each level
    header.parents_by_level.expanded_iter().for_each(|level| {
        hasher.write_var_array(level);
    });

    // Write all header fields
    hasher
        .update(header.hash_merkle_root)
        .update(header.accepted_id_merkle_root)
        .update(header.utxo_commitment)
        .update(timestamp.to_le_bytes())
        .update(header.bits.to_le_bytes())
        .update(nonce.to_le_bytes())
        .update(header.daa_score.to_le_bytes())
        .update(header.blue_score.to_le_bytes())
        .write_blue_work(header.blue_work)
        .update(header.pruning_point);

    if let Some(final_state) = pom_final_state {
        hasher.update(final_state.to_le_bytes());
    }

    hasher.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BlueWorkType, blockhash};

    #[test]
    fn test_header_hashing() {
        let header = Header::new_finalized(
            1,
            vec![vec![1.into()]].try_into().unwrap(),
            Default::default(),
            Default::default(),
            Default::default(),
            234,
            23,
            567,
            0,
            0.into(),
            0,
            Default::default(),
            0,
        );
        assert_ne!(blockhash::NONE, header.hash);
    }

    #[test]
    fn test_hash_blue_work() {
        let tests: Vec<(BlueWorkType, Vec<u8>)> =
            vec![(0.into(), vec![0, 0, 0, 0, 0, 0, 0, 0]), (123456.into(), vec![3, 0, 0, 0, 0, 0, 0, 0, 1, 226, 64])];

        for test in tests {
            let mut hasher = keryx_hashes::BlockHash::new();
            hasher.write_blue_work(test.0);

            let mut hasher2 = keryx_hashes::BlockHash::new();
            hasher2.update(test.1);
            assert_eq!(hasher.finalize(), hasher2.finalize())
        }
    }
}
