use super::protocol::GhostdagManager;
use crate::model::stores::ghostdag::GhostdagStoreReader;
use crate::model::stores::relations::RelationsStoreReader;
use crate::model::{services::reachability::ReachabilityService, stores::headers::HeaderStoreReader};
use keryx_consensus_core::{BlockHashSet, HashMapCustomHasher};
use keryx_hashes::Hash;
use std::collections::VecDeque;

pub fn unordered_mergeset_without_selected_parent<S: RelationsStoreReader + ?Sized, U: ReachabilityService + ?Sized>(
    relations: &S,
    reachability: &U,
    selected_parent: Hash,
    parents: &[Hash],
) -> BlockHashSet {
    // Typical mergeset at 10 BPS is O(k) (~124). Pre-size so the BFS does not rehash on every
    // near-tip block — this is on the header-processor critical path.
    let hint = parents.len().saturating_mul(4).max(16);
    let mut queue: VecDeque<Hash> = VecDeque::with_capacity(hint);
    for &p in parents {
        if p != selected_parent {
            queue.push_back(p);
        }
    }
    let mut mergeset: BlockHashSet = BlockHashSet::with_capacity(hint);
    for &h in &queue {
        mergeset.insert(h);
    }
    let mut past = BlockHashSet::with_capacity(hint);

    while let Some(current) = queue.pop_front() {
        let current_parents = relations.get_parents(current).unwrap();

        // For each parent of the current block we check whether it is in the past of the selected parent. If not,
        // we add it to the resulting merge-set and queue it for further processing.
        for parent in current_parents.iter() {
            if mergeset.contains(parent) || past.contains(parent) {
                continue;
            }

            if reachability.is_dag_ancestor_of(*parent, selected_parent) {
                past.insert(*parent);
                continue;
            }

            mergeset.insert(*parent);
            queue.push_back(*parent);
        }
    }

    mergeset
}

impl<T: GhostdagStoreReader, S: RelationsStoreReader, U: ReachabilityService, V: HeaderStoreReader> GhostdagManager<T, S, U, V> {
    pub fn ordered_mergeset_without_selected_parent(&self, selected_parent: Hash, parents: &[Hash]) -> Vec<Hash> {
        self.sort_blocks(self.unordered_mergeset_without_selected_parent(selected_parent, parents))
    }

    pub fn unordered_mergeset_without_selected_parent(&self, selected_parent: Hash, parents: &[Hash]) -> BlockHashSet {
        unordered_mergeset_without_selected_parent(&self.relations_store, &self.reachability_service, selected_parent, parents)
    }
}