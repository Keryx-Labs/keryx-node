//! N-gram table lookups (Engram) of DeepSeek-V4.1: the rows a token reads, from its committed
//! token history, the token map and the table, each opened against its root.

use crate::manifest::EngramLinks;
use crate::merkle::{Hash, rt_verify_range};
use crate::opening::ChunkRange;

/// Table row of each head of `e` for a token whose mapped id and those of its predecessors are
/// `ids` (`ids[s]` the token `s` positions back, `None` when blocked: before the sequence start
/// or outside the map); a blocked shift blocks every longer one.
pub fn engram_rows(e: &EngramLinks, ids: &[Option<u32>]) -> Option<Vec<u32>> {
    let (n_gram, n_head) = (e.n_gram as usize, e.n_head as usize);
    if ids.len() != n_gram || n_gram < 2 {
        return None;
    }
    let mut toks = vec![0u64; n_gram];
    let mut blocked = false;
    for (s, t) in toks.iter_mut().enumerate() {
        blocked = blocked || ids[s].is_none();
        *t = if blocked { e.pad as u64 } else { ids[s].unwrap() as u64 };
    }
    let mut out = Vec::with_capacity(e.n_cols());
    let mut rolling = toks[0].wrapping_mul(e.mult[0]);
    for o in 1..n_gram {
        rolling ^= toks[o].wrapping_mul(e.mult[o]);
        for h in 0..n_head {
            let col = (o - 1) * n_head + h;
            let p = e.primes[col] as u64;
            if p < 2 {
                return None;
            }
            out.push((rolling % p + e.offsets[col] as u64) as u32);
        }
    }
    Some(out)
}

/// Mapped id of token `tok` from the 32 B chunk of the map holding it; `None` when the chunk is
/// not that of the committed map. Tokens past the map are blocked (`Some(None)`).
pub fn mapped_id(e: &EngramLinks, tok: u32, chunk: &ChunkRange) -> Option<Option<u32>> {
    if tok >= e.map_len {
        return Some(None);
    }
    let n_chunks = (4 * e.map_len as u64).div_ceil(32);
    let c = tok as u64 * 4 / 32;
    if chunk.first_chunk != c || chunk.chunks.len() != 32 {
        return None;
    }
    let leaf: Hash = *blake3::hash(&chunk.chunks).as_bytes();
    if !rt_verify_range(&e.map_root, n_chunks, c, &[leaf], &chunk.proof) {
        return None;
    }
    let o = (tok as usize * 4) % 32;
    Some(Some(u32::from_le_bytes(chunk.chunks[o..o + 4].try_into().unwrap())))
}

/// Values of table row `row` from the chunks covering it; `None` when they are not those of the
/// committed table.
pub fn table_row(e: &EngramLinks, row: u32, r: &ChunkRange) -> Option<Vec<f32>> {
    let d = e.head_dim as usize;
    let rb = crate::quant::row_bytes(e.table_type, d)? as u64;
    let (lo, hi) = (row as u64 * rb, (row as u64 + 1) * rb);
    let (first, last) = (lo / 32, (hi - 1) / 32);
    if last >= e.table_chunks || r.first_chunk != first || r.chunks.len() as u64 != (last - first + 1) * 32 {
        return None;
    }
    let leaves: Vec<Hash> = r.chunks.chunks_exact(32).map(|c| *blake3::hash(c).as_bytes()).collect();
    if !rt_verify_range(&e.table_root, e.table_chunks, first, &leaves, &r.proof) {
        return None;
    }
    let o = (lo - first * 32) as usize;
    crate::quant::dequant_row(e.table_type, &r.chunks[o..o + rb as usize], d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::merkle::rt_range_proof;

    fn links(map: &[u32], table: &[u8]) -> EngramLinks {
        let map_bytes: Vec<u8> = map.iter().flat_map(|v| v.to_le_bytes()).chain(std::iter::repeat(0)).take(map.len().div_ceil(8) * 32).collect();
        let ml: Vec<Hash> = map_bytes.chunks_exact(32).map(|c| *blake3::hash(c).as_bytes()).collect();
        let tl: Vec<Hash> = table.chunks_exact(32).map(|c| *blake3::hash(c).as_bytes()).collect();
        let mut e = EngramLinks {
            module: 0, kv: 0, gate: 0, n_gram: 4, n_head: 2, head_dim: 32, table_type: crate::quant::IQ4_NL, table_root: rt_range_proof(&tl, 0, 1).0,
            table_chunks: tl.len() as u64, map_root: rt_range_proof(&ml, 0, 1).0, map_len: map.len() as u32, pad: 2, out_leaf: 0,
            mult: [0; crate::manifest::ENGRAM_MAX_GRAM], primes: [0; crate::manifest::ENGRAM_MAX_COLS], offsets: [0; crate::manifest::ENGRAM_MAX_COLS],
        };
        e.mult[..4].copy_from_slice(&[76_632_096_046_245, 4_839_876_093_313, 35_959_672_319_349, 73_987_337_458_391]);
        e.primes[..6].copy_from_slice(&[11, 13, 17, 19, 23, 29]);
        e.offsets[..6].copy_from_slice(&[0, 11, 24, 41, 60, 83]);
        e
    }

    /// The hash as llama.cpp writes it, transcribed independently.
    fn reference(e: &EngramLinks, ids: &[i64]) -> Vec<u32> {
        let mut toks = [0u64; 4];
        let mut blocked = ids[0] < 0;
        toks[0] = if blocked { e.pad as u64 } else { ids[0] as u64 };
        for s in 1..4 {
            blocked = blocked || ids[s] < 0;
            toks[s] = if blocked { e.pad as u64 } else { ids[s] as u64 };
        }
        let mut out = vec![0u32; 6];
        let mut rolling = toks[0].wrapping_mul(e.mult[0]);
        for o in 1..4 {
            rolling ^= toks[o].wrapping_mul(e.mult[o]);
            for h in 0..2 {
                let col = (o - 1) * 2 + h;
                out[col] = (rolling % e.primes[col] as u64 + e.offsets[col] as u64) as u32;
            }
        }
        out
    }

    #[test]
    fn rows_follow_the_hash_and_blocking() {
        let e = links(&[0], &[0; 32]);
        for ids in [[5i64, 9, 3, 7], [5, -1, 3, 7], [-1, 9, 3, 7], [5, 9, 3, -1], [0, 0, 0, 0]] {
            let opt: Vec<Option<u32>> = ids.iter().map(|&x| (x >= 0).then_some(x as u32)).collect();
            assert_eq!(engram_rows(&e, &opt).unwrap(), reference(&e, &ids), "{ids:?}");
        }
        let a = engram_rows(&e, &[Some(5), None, Some(3), Some(7)]).unwrap();
        let b = engram_rows(&e, &[Some(5), None, Some(4), Some(1)]).unwrap();
        assert_eq!(a, b, "a blocked shift hides the tokens behind it");
        assert!(engram_rows(&e, &[Some(5), Some(9)]).is_none());
    }

    #[test]
    fn the_map_and_the_table_open_against_their_roots() {
        let map: Vec<u32> = (0..40).map(|t| (t * 7) % 23).collect();
        let row = vec![0x3cu8, 0x00].into_iter().chain((0..16).map(|i| i as u8 * 17)).collect::<Vec<u8>>();
        let table: Vec<u8> = row.iter().copied().cycle().take(18 * 16).collect();
        let e = links(&map, &table);
        let map_bytes: Vec<u8> = map.iter().flat_map(|v| v.to_le_bytes()).collect();
        let ml: Vec<Hash> = map_bytes.chunks(32).map(|c| { let mut p = c.to_vec(); p.resize(32, 0); *blake3::hash(&p).as_bytes() }).collect();
        let chunk = |t: u32| {
            let c = t as usize * 4 / 32;
            let mut b = map_bytes[c * 32..((c + 1) * 32).min(map_bytes.len())].to_vec();
            b.resize(32, 0);
            ChunkRange { first_chunk: c as u64, chunks: b, proof: rt_range_proof(&ml, c as u64, 1).1 }
        };
        assert_eq!(mapped_id(&e, 13, &chunk(13)), Some(Some(map[13])));
        assert_eq!(mapped_id(&e, 13, &chunk(21)), None);
        assert_eq!(mapped_id(&e, 99, &chunk(13)), Some(None));
        let tl: Vec<Hash> = table.chunks_exact(32).map(|c| *blake3::hash(c).as_bytes()).collect();
        let range = |r: u64| {
            let (lo, hi) = (r * 18, (r + 1) * 18);
            let (f, l) = (lo / 32, (hi - 1) / 32);
            ChunkRange { first_chunk: f, chunks: table[(f * 32) as usize..((l + 1) * 32) as usize].to_vec(), proof: rt_range_proof(&tl, f, l - f + 1).1 }
        };
        let want = crate::quant::dequant_row(crate::quant::IQ4_NL, &row, 32).unwrap();
        assert_eq!(table_row(&e, 5, &range(5)), Some(want));
        assert_eq!(table_row(&e, 5, &range(6)), None);
        let mut bad = range(5);
        bad.chunks[3] ^= 1;
        assert_eq!(table_row(&e, 5, &bad), None);
    }

    #[test]
    fn the_envelope_round_trips() {
        use crate::manifest::{Chain, LayerLinks, NONE, chain_bytes};
        let e = links(&[1, 2, 3], &[0; 64]);
        let inner = LayerLinks { layer: 1, proj: 4, down: 9, post_attn: NONE, post_ffn: NONE, scale: NONE, moe: None, attn: None, recur: None, hc: None, dsa: None, comp: None, engram: None };
        let c = Chain::Layer(LayerLinks { engram: Some(e), ..inner });
        let mut ent = vec![1u8, 0, b'x'];
        ent.extend_from_slice(&[0u8; 44]);
        ent.extend_from_slice(&[crate::manifest::ROLE_LAYER, 0, crate::manifest::FORM_NONE]);
        ent.extend_from_slice(&NONE.to_le_bytes());
        ent.extend_from_slice(&NONE.to_le_bytes());
        ent.extend_from_slice(&chain_bytes(&c));
        assert_eq!(crate::manifest::Entry::parse(&ent).map(|(x, n)| (x.chain, n)), Some((c, ent.len())));
        let mut cut = ent.clone();
        cut.truncate(ent.len() - 30);
        assert!(crate::manifest::Entry::parse(&cut).is_none());
    }
}
