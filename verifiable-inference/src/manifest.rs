//! Model manifest entries.

use crate::merkle::{self, Hash};

pub const ROLE_PRODUCT: u8 = 0;
pub const ROLE_BIAS: u8 = 1;
pub const ROLE_OUTPUT: u8 = 2;
/// Token embedding table: `m` rows of `k` values.
pub const ROLE_EMBD: u8 = 3;
/// Layer-0 input norm weights: `k` F32 values.
pub const ROLE_NORM: u8 = 4;
pub const FORM_NONE: u8 = 0;
pub const FORM_SPLIT: u8 = 1;
pub const FORM_PAIR_GATE: u8 = 2;
pub const FORM_PAIR_UP: u8 = 3;
pub const GLU_SWIGLU: u8 = 1;
pub const GLU_GEGLU: u8 = 2;
pub const NONE: u32 = u32::MAX;
pub const ROLE_LAYER: u8 = 5;
/// F32 router weights: `m` rows (experts) of `k` values.
pub const ROLE_ROUTER: u8 = 6;
/// One product per head: `n_exp` heads of `m` rows of `k` weights, head `h` reads input slice `h`.
pub const ROLE_HEADS: u8 = 7;

pub const LINK_NONE: u8 = 0;
pub const LINK_ATTN_IN: u8 = 1;
pub const LINK_FFN_IN: u8 = 2;
pub const LINK_GLU: u8 = 3;
pub const LINK_HEAD: u8 = 4;
pub const LINK_DIRECT: u8 = 5;

const TAG_NONE: u8 = 0;
const TAG_PRODUCT: u8 = 1;
const TAG_LAYER: u8 = 2;
const TAG_MOE_LAYER: u8 = 3;

/// Source of a product's input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProductLink {
    pub layer: u32,
    pub link: u8,
    pub norm: u32,
    pub src: u32,
}

/// Residual additions of a layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerLinks {
    pub layer: u32,
    pub proj: u32,
    pub down: u32,
    pub post_attn: u32,
    pub post_ffn: u32,
    pub scale: u32,
    pub moe: Option<MoeLinks>,
}

/// Routed experts of a layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoeLinks {
    pub down_exps: u32,
    pub router: u32,
    pub probs_bias: u32,
    /// f32 bits.
    pub w_scale: u32,
    pub ffn_norm: u32,
    /// Rank of the gate product of the routed experts, whose input is the router input.
    pub gate_exps: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Chain {
    None,
    Product(ProductLink),
    Layer(LayerLinks),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Entry {
    pub name: String,
    pub wtype: u32,
    pub k: u64,
    pub m: u64,
    pub n_exp: u32,
    pub chunk_base: u64,
    pub n_bytes: u64,
    pub row_bytes: u32,
    pub role: u8,
    pub glu_op: u8,
    pub glu_form: u8,
    pub partner: u32,
    pub bias: u32,
    pub chain: Chain,
}

fn parse_chain(b: &[u8]) -> Option<(Chain, usize)> {
    let u32a = |o: usize| b.get(o..o + 4).map(|s| u32::from_le_bytes(s.try_into().unwrap()));
    match b.first() {
        None => None,
        Some(&TAG_NONE) => Some((Chain::None, 1)),
        Some(&TAG_PRODUCT) => {
            let link = *b.get(5)?;
            Some((Chain::Product(ProductLink { layer: u32a(1)?, link, norm: u32a(6)?, src: u32a(10)? }), 14))
        }
        Some(&tag @ (TAG_LAYER | TAG_MOE_LAYER)) => {
            let moe = if tag == TAG_MOE_LAYER {
                Some(MoeLinks { down_exps: u32a(25)?, router: u32a(29)?, probs_bias: u32a(33)?, w_scale: u32a(37)?, ffn_norm: u32a(41)?, gate_exps: u32a(45)? })
            } else {
                None
            };
            Some((
                Chain::Layer(LayerLinks {
                    layer: u32a(1)?,
                    proj: u32a(5)?,
                    down: u32a(9)?,
                    post_attn: u32a(13)?,
                    post_ffn: u32a(17)?,
                    scale: u32a(21)?,
                    moe,
                }),
                if moe.is_some() { 49 } else { 25 },
            ))
        }
        _ => None,
    }
}

/// Serialized chain extension of an entry.
pub fn chain_bytes(c: &Chain) -> Vec<u8> {
    let mut b = Vec::new();
    match c {
        Chain::None => b.push(TAG_NONE),
        Chain::Product(p) => {
            b.push(TAG_PRODUCT);
            b.extend_from_slice(&p.layer.to_le_bytes());
            b.push(p.link);
            b.extend_from_slice(&p.norm.to_le_bytes());
            b.extend_from_slice(&p.src.to_le_bytes());
        }
        Chain::Layer(l) => {
            b.push(if l.moe.is_some() { TAG_MOE_LAYER } else { TAG_LAYER });
            for v in [l.layer, l.proj, l.down, l.post_attn, l.post_ffn, l.scale] {
                b.extend_from_slice(&v.to_le_bytes());
            }
            if let Some(m) = l.moe {
                for v in [m.down_exps, m.router, m.probs_bias, m.w_scale, m.ffn_norm, m.gate_exps] {
                    b.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
    }
    b
}

impl Entry {
    /// Parse one whole serialized entry (`b` holds exactly one entry); returns the entry and its
    /// byte length. A stage-1 entry has no chain extension.
    pub fn parse(b: &[u8]) -> Option<(Entry, usize)> {
        let (mut e, len) = Self::parse_base(b)?;
        if len == b.len() {
            return Some((e, len));
        }
        let (c, n) = parse_chain(&b[len..])?;
        e.chain = c;
        Some((e, len + n))
    }

    /// Parse the entry at the start of `b` inside a manifest file; `chain` = the file carries a
    /// chain extension after every entry.
    pub fn parse_in_file(b: &[u8], chain: bool) -> Option<(Entry, usize)> {
        let (mut e, len) = Self::parse_base(b)?;
        if !chain {
            return Some((e, len));
        }
        let (c, n) = parse_chain(&b[len..])?;
        e.chain = c;
        Some((e, len + n))
    }

    fn parse_base(b: &[u8]) -> Option<(Entry, usize)> {
        let nl = u16::from_le_bytes(b.get(0..2)?.try_into().ok()?) as usize;
        let name = String::from_utf8(b.get(2..2 + nl)?.to_vec()).ok()?;
        let r = b.get(2 + nl..2 + nl + 55)?;
        let u32a = |o: usize| u32::from_le_bytes(r[o..o + 4].try_into().unwrap());
        let u64a = |o: usize| u64::from_le_bytes(r[o..o + 8].try_into().unwrap());
        let e = Entry {
            name,
            wtype: u32a(0),
            k: u64a(4),
            m: u64a(12),
            n_exp: u32a(20),
            chunk_base: u64a(24),
            n_bytes: u64a(32),
            row_bytes: u32a(40),
            role: r[44],
            glu_op: r[45],
            glu_form: r[46],
            partner: u32a(47),
            bias: u32a(51),
            chain: Chain::None,
        };
        Some((e, 2 + nl + 55))
    }

    pub fn leaf(bytes: &[u8]) -> Hash {
        *blake3::hash(bytes).as_bytes()
    }

    /// Chunk range covering a weight row, and the row offset in its first chunk.
    pub fn row_chunks(&self, exp: u64, row: u64) -> Option<(u64, u64, usize)> {
        if row >= self.m || exp >= self.n_exp as u64 {
            return None;
        }
        let start = (exp * self.m + row) * self.row_bytes as u64;
        let end = start + self.row_bytes as u64;
        if end > self.n_bytes / 32 * 32 {
            return None;
        }
        Some((self.chunk_base + start / 32, self.chunk_base + (end - 1) / 32, (start % 32) as usize))
    }
}

/// Inclusion of an entry in the manifest.
pub fn verify_entry(m_t: &Hash, n_entries: u64, index: u64, entry_bytes: &[u8], path: &[Hash]) -> bool {
    merkle::rt_verify_leaf(m_t, n_entries, index, &Entry::leaf(entry_bytes), path)
}

/// Entries with their serialized bytes, and the product count.
pub type Manifest = (Vec<(Entry, Vec<u8>)>, u32);

/// Entries of a serialized manifest and its product count. `KXMANIF2` carries a chain
/// extension after every entry.
pub fn read_file(data: &[u8]) -> Option<Manifest> {
    let chain = match data.get(0..8)? {
        b"KXMANIF1" => false,
        b"KXMANIF2" => true,
        _ => return None,
    };
    let n = u32::from_le_bytes(data[8..12].try_into().ok()?) as usize;
    let n_products = u32::from_le_bytes(data[12..16].try_into().ok()?);
    let mut o = 16;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let (e, len) = Entry::parse_in_file(&data[o..], chain)?;
        out.push((e, data[o..o + len].to_vec()));
        o += len;
    }
    Some((out, n_products))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_bytes(name: &str, role: u8) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&(name.len() as u16).to_le_bytes());
        b.extend_from_slice(name.as_bytes());
        b.extend_from_slice(&[0u8; 44]);
        b.extend_from_slice(&[role, 0, FORM_NONE]);
        b.extend_from_slice(&NONE.to_le_bytes());
        b.extend_from_slice(&NONE.to_le_bytes());
        b
    }

    #[test]
    fn stage1_entry_parses_without_chain() {
        let b = base_bytes("blk.0.attn_q.weight", ROLE_PRODUCT);
        let (e, len) = Entry::parse(&b).unwrap();
        assert_eq!((len, e.chain), (b.len(), Chain::None));
    }

    #[test]
    fn chain_extensions_round_trip() {
        let p = Chain::Product(ProductLink { layer: 3, link: LINK_FFN_IN, norm: 17, src: NONE });
        let l = Chain::Layer(LayerLinks { layer: 3, proj: 4, down: 9, post_attn: 20, post_ffn: 21, scale: NONE, moe: None });
        let moe = MoeLinks { down_exps: 12, router: 30, probs_bias: 31, w_scale: 2.446f32.to_bits(), ffn_norm: 29, gate_exps: 13 };
        let lm = Chain::Layer(LayerLinks { layer: 4, proj: 5, down: 11, post_attn: NONE, post_ffn: NONE, scale: NONE, moe: Some(moe) });
        for c in [Chain::None, p, l, lm] {
            let mut b = base_bytes("x", ROLE_PRODUCT);
            b.extend_from_slice(&chain_bytes(&c));
            let (e, len) = Entry::parse(&b).unwrap();
            assert_eq!((len, e.chain), (b.len(), c));
        }
        let mut bad = base_bytes("x", ROLE_PRODUCT);
        bad.push(9);
        assert!(Entry::parse(&bad).is_none());
    }

    #[test]
    fn manifest2_file_reads_every_entry() {
        let link = Chain::Product(ProductLink { layer: 0, link: LINK_ATTN_IN, norm: 2, src: NONE });
        let mut f = b"KXMANIF2".to_vec();
        f.extend_from_slice(&3u32.to_le_bytes());
        f.extend_from_slice(&1u32.to_le_bytes());
        let mut want = Vec::new();
        for (name, role, c) in [("blk.0.attn_q.weight", ROLE_PRODUCT, link), ("token_embd.weight", ROLE_EMBD, Chain::None), ("blk.0.attn_norm.weight", ROLE_NORM, Chain::None)] {
            let mut b = base_bytes(name, role);
            b.extend_from_slice(&chain_bytes(&c));
            f.extend_from_slice(&b);
            want.push((b, c));
        }
        let (entries, n_products) = read_file(&f).unwrap();
        assert_eq!(n_products, 1);
        for ((e, bytes), (wb, c)) in entries.iter().zip(&want) {
            assert_eq!((bytes, e.chain), (wb, *c));
        }
    }

    #[test]
    #[ignore]
    fn manifest2_files_from_env() {
        let dir = std::env::var("VI_MANIFEST_DIR").expect("VI_MANIFEST_DIR");
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_none_or(|e| e != "bin") {
                continue;
            }
            let data = std::fs::read(&path).unwrap();
            let (entries, n_products) = read_file(&data).unwrap();
            let leaves: Vec<Hash> = entries.iter().map(|(_, b)| Entry::leaf(b)).collect();
            let m_t = crate::merkle::rt_range_proof(&leaves, 0, 1).0;
            let links = entries.iter().filter(|(e, _)| matches!(e.chain, Chain::Product(_))).count();
            let layers = entries.iter().filter(|(e, _)| matches!(e.chain, Chain::Layer(_))).count();
            for (i, (e, b)) in entries.iter().enumerate() {
                let (p, len) = Entry::parse(b).unwrap();
                assert_eq!((&p, len), (e, b.len()));
                let path = crate::merkle::rt_range_proof(&leaves, i as u64, 1).1;
                assert!(verify_entry(&m_t, entries.len() as u64, i as u64, b, &path));
            }
            println!("{} entries {} products {} links {} layers {} M_T {}", path.display(), entries.len(), n_products, links, layers,
                m_t.iter().map(|b| format!("{b:02x}")).collect::<String>());
        }
    }
}
