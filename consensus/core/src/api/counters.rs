use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Default)]
pub struct ProcessingCounters {
    pub blocks_submitted: AtomicU64,
    pub header_counts: AtomicU64,
    pub dep_counts: AtomicU64,
    pub mergeset_counts: AtomicU64,
    pub body_counts: AtomicU64,
    pub txs_counts: AtomicU64,
    pub chain_block_counts: AtomicU64,
    pub chain_disqualified_counts: AtomicU64,
    pub mass_counts: AtomicU64,
    /// App-cache hits for the address-balance index (virtual-commit RMW path).
    pub address_balance_cache_hits: AtomicU64,
    /// App-cache misses for the address-balance index — each miss is a RocksDB point read.
    pub address_balance_cache_misses: AtomicU64,
    /// App-cache hits for the coin-age bucket index.
    pub age_buckets_cache_hits: AtomicU64,
    /// App-cache misses for the coin-age bucket index.
    pub age_buckets_cache_misses: AtomicU64,
    /// Hits on the windowed-production prefix SeekForPrev cache.
    pub windowed_prefix_cache_hits: AtomicU64,
    /// Misses that fall through to a RocksDB `SeekForPrev` on the production prefix index.
    pub windowed_prefix_cache_misses: AtomicU64,
}

impl ProcessingCounters {
    pub fn snapshot(&self) -> ProcessingCountersSnapshot {
        ProcessingCountersSnapshot {
            blocks_submitted: self.blocks_submitted.load(Ordering::Relaxed),
            header_counts: self.header_counts.load(Ordering::Relaxed),
            dep_counts: self.dep_counts.load(Ordering::Relaxed),
            mergeset_counts: self.mergeset_counts.load(Ordering::Relaxed),
            body_counts: self.body_counts.load(Ordering::Relaxed),
            txs_counts: self.txs_counts.load(Ordering::Relaxed),
            chain_block_counts: self.chain_block_counts.load(Ordering::Relaxed),
            chain_disqualified_counts: self.chain_disqualified_counts.load(Ordering::Relaxed),
            mass_counts: self.mass_counts.load(Ordering::Relaxed),
            address_balance_cache_hits: self.address_balance_cache_hits.load(Ordering::Relaxed),
            address_balance_cache_misses: self.address_balance_cache_misses.load(Ordering::Relaxed),
            age_buckets_cache_hits: self.age_buckets_cache_hits.load(Ordering::Relaxed),
            age_buckets_cache_misses: self.age_buckets_cache_misses.load(Ordering::Relaxed),
            windowed_prefix_cache_hits: self.windowed_prefix_cache_hits.load(Ordering::Relaxed),
            windowed_prefix_cache_misses: self.windowed_prefix_cache_misses.load(Ordering::Relaxed),
        }
    }
}

#[derive(Default, Debug, PartialEq, Eq)]
pub struct ProcessingCountersSnapshot {
    pub blocks_submitted: u64,
    pub header_counts: u64,
    pub dep_counts: u64,
    pub mergeset_counts: u64,
    pub body_counts: u64,
    pub txs_counts: u64,
    pub chain_block_counts: u64,
    pub chain_disqualified_counts: u64,
    pub mass_counts: u64,
    pub address_balance_cache_hits: u64,
    pub address_balance_cache_misses: u64,
    pub age_buckets_cache_hits: u64,
    pub age_buckets_cache_misses: u64,
    pub windowed_prefix_cache_hits: u64,
    pub windowed_prefix_cache_misses: u64,
}

impl core::ops::Sub for &ProcessingCountersSnapshot {
    type Output = ProcessingCountersSnapshot;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::Output {
            blocks_submitted: self.blocks_submitted.saturating_sub(rhs.blocks_submitted),
            header_counts: self.header_counts.saturating_sub(rhs.header_counts),
            dep_counts: self.dep_counts.saturating_sub(rhs.dep_counts),
            mergeset_counts: self.mergeset_counts.saturating_sub(rhs.mergeset_counts),
            body_counts: self.body_counts.saturating_sub(rhs.body_counts),
            txs_counts: self.txs_counts.saturating_sub(rhs.txs_counts),
            chain_block_counts: self.chain_block_counts.saturating_sub(rhs.chain_block_counts),
            chain_disqualified_counts: self.chain_disqualified_counts.saturating_sub(rhs.chain_disqualified_counts),
            mass_counts: self.mass_counts.saturating_sub(rhs.mass_counts),
            address_balance_cache_hits: self.address_balance_cache_hits.saturating_sub(rhs.address_balance_cache_hits),
            address_balance_cache_misses: self.address_balance_cache_misses.saturating_sub(rhs.address_balance_cache_misses),
            age_buckets_cache_hits: self.age_buckets_cache_hits.saturating_sub(rhs.age_buckets_cache_hits),
            age_buckets_cache_misses: self.age_buckets_cache_misses.saturating_sub(rhs.age_buckets_cache_misses),
            windowed_prefix_cache_hits: self.windowed_prefix_cache_hits.saturating_sub(rhs.windowed_prefix_cache_hits),
            windowed_prefix_cache_misses: self.windowed_prefix_cache_misses.saturating_sub(rhs.windowed_prefix_cache_misses),
        }
    }
}
