//! Stage-2 block task: a traced pass of the tier model over an input drawn from a recent chain block.

use crate::chain::{ChainCtx, RESID_POINT, resid_position, verify_chain_point, verify_chain_prefix_point, verify_resid_point};
use crate::draw;
use crate::merkle::Hash;
use crate::opening::Opening;
use crate::prefix::{PREFIX_LEN, PrefixCtx, expected_token};
use crate::verify::{Fail, ResponseCtx, TierCtx};

/// DAA span of a task window.
pub const WINDOW_DAA: u64 = 64;
/// DAA delay between a window's seed block and its first use.
pub const LAG_DAA: u64 = 16;
/// Positions of the task.
pub const TASK_LEN: u32 = PREFIX_LEN;

const INPUT_DOMAIN: &[u8] = b"KeryxBlockTaskV1";
const DRAW_DOMAIN: &[u8] = b"KeryxBlockTaskDrawV1";

/// Task window of a block at `daa`, or `None` before the first window is usable.
pub fn window(daa: u64) -> Option<u64> {
    daa.checked_sub(LAG_DAA).map(|d| d / WINDOW_DAA)
}

/// First DAA of window `w`: its seed is the first chain block at or after it.
pub fn window_start(w: u64) -> u64 {
    w * WINDOW_DAA
}

/// The 32 bytes that select the task tokens of a window.
pub fn task_bytes(window_seed: &Hash) -> Hash {
    let mut h = blake3::Hasher::new();
    h.update(INPUT_DOMAIN);
    h.update(window_seed);
    *h.finalize().as_bytes()
}

/// Token ids of the task.
pub fn task_tokens(table: &[u32; 256], window_seed: &Hash) -> Vec<u32> {
    task_bytes(window_seed).iter().map(|&b| table[b as usize]).collect()
}

/// Draw seed of a block's task opening.
pub fn draw_seed(block_hash: &Hash) -> Hash {
    let mut h = blake3::Hasher::new();
    h.update(DRAW_DOMAIN);
    h.update(block_hash);
    *h.finalize().as_bytes()
}

/// Task context of a tier: its prefix context with the task at position 0.
pub fn task_prefix(prefix: &PrefixCtx) -> PrefixCtx {
    PrefixCtx { offset: 0, ..prefix.clone() }
}

/// Verify a block's task opening against the task root in its header.
pub fn verify_block_task(
    tier: &TierCtx,
    ctx: &ChainCtx,
    prefix: &PrefixCtx,
    task_root: &Hash,
    window_seed: &Hash,
    block_hash: &Hash,
    opening: &Opening,
) -> Result<(), Fail> {
    let (Some(pp), Some(rp)) = (&opening.prefix, &opening.resid) else { return Err(Fail::Shape) };
    if opening.points.len() != draw::POINTS as usize || opening.links.len() != opening.points.len() || opening.experts.len() != opening.points.len() {
        return Err(Fail::Shape);
    }
    let pc = task_prefix(prefix);
    let bytes = task_bytes(window_seed);
    let tok_at = |t: u64| expected_token(&pc, &bytes, t).ok_or(Fail::Shape);
    let resp = ResponseCtx { trace_root: *task_root, n_prompt: TASK_LEN, n_pos: TASK_LEN };
    let seed = draw_seed(block_hash);
    for (i, (p, (tok, link))) in opening.points.iter().zip(&opening.links).enumerate() {
        let r = verify_chain_point(tier, ctx, &resp, &seed, i as u32, p, *tok, link, opening.experts[i].as_ref())?;
        if *tok != tok_at(r.t)? {
            return Err(Fail::Mismatch);
        }
    }
    verify_chain_prefix_point(tier, ctx, &pc, &resp, &seed, &bytes, pp)?;
    verify_resid_point(tier, ctx, &resp, &seed, rp)?;
    let d = draw::point(&seed, RESID_POINT);
    let layer = (d[1] % (2 * ctx.n_layers as u64)) / 2;
    if rp.tok != tok_at(resid_position(ctx, &resp, layer, d[0])?)? {
        return Err(Fail::Mismatch);
    }
    Ok(())
}
