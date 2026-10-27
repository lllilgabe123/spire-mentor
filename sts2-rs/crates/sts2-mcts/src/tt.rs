//! Open-addressing transposition table for MCTS.
//!
//! Bead `sts2-rs-h2ql` (Wave 4 of
//! `docs/plans/2026-05-17-mcts-leaf-eval-and-search-improvements.md`
//! §4.5). Replaces the prior `std::collections::HashMap<(u64, u8),
//! NodeId>` introduced by `sts2-rs-7z97` with a fixed-size
//! `Vec<Entry>` (2²⁰ slots = ~16 MiB at 16 B/entry) probed via
//! `ahash` of the `(state_hash, depth_bucket)` key. Linear probing
//! with a small probe budget; on overflow we evict the colliding
//! entry (always-replace policy) — MCTS treats the TT as a hint, so
//! a missed hit costs at most one extra node allocation, never
//! correctness.
//!
//! ## Why this shape
//!
//! The prior `HashMap` triggered the predecessor bead's +15%
//! wall-time penalty for three reasons:
//!   1. `DefaultHasher` (SipHash) hashes the `(u64, u8)` tuple even
//!      though `state_hash` already returns a well-mixed `u64`.
//!   2. `HashMap` grows dynamically — every doubling triggers a
//!      `realloc` + rehash of the full table.
//!   3. The `(u64, u8)` key gets boxed-up by the hashing trait, and
//!      `HashMap::get` does two hash mixes (one for the bucket, one
//!      for the SipHash key).
//!
//! Our fixed-size table sidesteps all three: `ahash::AHasher` is
//! 3–5× faster than SipHash on small keys, the backing `Vec` never
//! resizes, and the index derivation is a single mask. The
//! always-replace policy is the standard MCTS choice (Childs et al.
//! 2008 §3.2; sts_lightspeed's `BattleScumSearcher2` uses the same
//! pattern for its `ttHash`). Q-merging is structural: a TT hit
//! returns the existing `NodeId`, and the new edge re-points to it
//! — both parents then back up into the shared node, which is the
//! Childs/Brodeur/Kocsis UCT3 "sum into both paths" rule.

use std::hash::Hasher;

use ahash::AHasher;

use crate::NodeId;

/// Table size as a power of two. `2²⁰` ≈ 1 M slots; at `Entry`
/// size 16 B (8 B key + 4 B NodeId + 4 B flags/padding) the table
/// is ~16 MiB resident. Plan §4.5 picked this size from the
/// lightspeed precedent (`ttSize = 1 << 20`) — the working-set
/// size for a 10–30k-sim search rarely exceeds a few hundred
/// thousand distinct states, so probes terminate fast.
pub const TT_SIZE_LOG2: u32 = 20;
pub const TT_SIZE: usize = 1 << TT_SIZE_LOG2;

/// Linear-probe budget per lookup. After this many failed probes
/// the lookup gives up and reports "miss" (insertion then evicts
/// the slot at the initial index — always-replace).
const PROBE_LIMIT: u8 = 8;

/// One transposition-table entry.
///
/// `key_high` stores the high `64 - TT_SIZE_LOG2` bits of the hashed
/// key so probing can distinguish "different entry that hashed to
/// the same bucket" from "same entry, hit"; the low bits are
/// implicit in the slot index. `node` is meaningful only when
/// `occupied` is true.
#[derive(Clone, Copy, Debug)]
struct Entry {
    key_high: u64,
    node: NodeId,
    occupied: bool,
}

impl Entry {
    const EMPTY: Self = Self {
        key_high: 0,
        node: NodeId(0),
        occupied: false,
    };
}

/// Fixed-size open-addressing transposition table.
/// Default size for chance-node mode: `2¹⁶`
/// slots (~1 MiB at 16 B/entry). Chance-mode combat trees are small
/// (~1k–few-k nodes per search) and the table is re-allocated per `Mcts::new`
/// (one per committed move in closed-loop play), so the full `2²⁰` table's
/// 16 MiB zero-fill dominated wall-time; a table sized to the working set
/// keeps the visit-concentration benefit without the allocation tax.
pub const CHANCE_TT_SIZE_LOG2: u32 = 16;

#[derive(Debug)]
pub struct TranspositionTable {
    entries: Box<[Entry]>,
    /// `entries.len() - 1`, the index mask (len is always a power of two).
    mask: u64,
    /// `log2(entries.len())` — bit width consumed by the slot index.
    size_log2: u32,
    /// Cumulative hit count since construction or [`Self::clear`].
    hits: u64,
}

impl TranspositionTable {
    /// Allocate a fresh, empty TT with the default `2²⁰` size (PIMC path).
    pub fn new() -> Self {
        Self::with_log2(TT_SIZE_LOG2)
    }

    /// Allocate a fresh, empty TT with `2^size_log2` slots. Chance mode uses
    /// this to request a table sized to its smaller working set.
    pub fn with_log2(size_log2: u32) -> Self {
        let size = 1usize << size_log2;
        let entries: Box<[Entry]> = vec![Entry::EMPTY; size].into_boxed_slice();
        Self {
            entries,
            mask: (size as u64) - 1,
            size_log2,
            hits: 0,
        }
    }

    /// Empty every slot and reset the hit counter. Called on
    /// reroot — surviving nodes get re-inserted lazily as the search
    /// re-traverses them.
    pub fn clear(&mut self) {
        for e in &mut self.entries {
            *e = Entry::EMPTY;
        }
        self.hits = 0;
    }

    /// Cumulative number of probes that resolved to an occupied
    /// matching slot since construction or [`Self::clear`].
    pub fn hits(&self) -> u64 {
        self.hits
    }

    /// Probe for `key` WITHOUT bumping the hit counter. Used by the
    /// expansion site so the cycle-guard branch (where a probe
    /// matches an ancestor and the engine declines to wire to it)
    /// doesn't inflate the diagnostic. Call [`Self::record_hit`]
    /// after deciding the hit was used.
    pub fn peek(&self, key: (u64, u8)) -> Option<NodeId> {
        let mixed = mix(key);
        let mut idx = (mixed & self.mask) as usize;
        let key_high = mixed >> self.size_log2;
        let wrap = self.entries.len() - 1;
        for _ in 0..PROBE_LIMIT {
            let e = &self.entries[idx];
            if !e.occupied {
                return None;
            }
            if e.key_high == key_high {
                return Some(e.node);
            }
            idx = (idx + 1) & wrap;
        }
        None
    }

    /// Probe for `key`. Returns the stored [`NodeId`] on hit, or
    /// `None` on miss; bumps [`Self::hits`] on a hit. Linear-probes
    /// up to [`PROBE_LIMIT`] slots from the initial index before
    /// reporting miss.
    pub fn lookup(&mut self, key: (u64, u8)) -> Option<NodeId> {
        if let Some(n) = self.peek(key) {
            self.hits += 1;
            Some(n)
        } else {
            None
        }
    }

    /// Bump the hit counter externally — used by callers that
    /// probed with [`Self::peek`] and then decided the hit was
    /// usable. Cheaper than re-probing.
    pub fn record_hit(&mut self) {
        self.hits += 1;
    }

    /// Insert `(key, node)`. If the natural slot and the next
    /// [`PROBE_LIMIT`] − 1 are occupied by other keys we evict the
    /// entry at the natural slot (always-replace) — MCTS uses the
    /// TT as a hint, so the worst case is the redundant allocation
    /// of one extra node.
    pub fn insert(&mut self, key: (u64, u8), node: NodeId) {
        let mixed = mix(key);
        let initial = (mixed & self.mask) as usize;
        let key_high = mixed >> self.size_log2;
        let mut idx = initial;
        let wrap = self.entries.len() - 1;
        for _ in 0..PROBE_LIMIT {
            let e = &mut self.entries[idx];
            if !e.occupied || e.key_high == key_high {
                *e = Entry {
                    key_high,
                    node,
                    occupied: true,
                };
                return;
            }
            idx = (idx + 1) & wrap;
        }
        // All probe slots collided — evict the natural slot.
        self.entries[initial] = Entry {
            key_high,
            node,
            occupied: true,
        };
    }
}

impl Default for TranspositionTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Mix `(state_hash, depth_bucket)` into a 64-bit hash via `ahash`.
/// `state_hash` is already a well-mixed `u64` (SipHash output from
/// `sts2-core::hashing::state_hash`) but the depth bucket has to be
/// folded back in distinctly; `ahash::AHasher` does both for free
/// at ~1 ns/call on modern x86.
#[inline]
fn mix(key: (u64, u8)) -> u64 {
    let mut h = AHasher::default();
    h.write_u64(key.0);
    h.write_u8(key.1);
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroU32;

    fn n(id: u32) -> NodeId {
        NodeId(id)
    }

    #[test]
    fn empty_table_misses() {
        let mut tt = TranspositionTable::new();
        assert!(tt.lookup((0xdead_beef, 0)).is_none());
        assert_eq!(tt.hits(), 0);
    }

    #[test]
    fn round_trip_lookup_hits() {
        let mut tt = TranspositionTable::new();
        tt.insert((0x1234_5678_9abc_def0, 3), n(42));
        assert_eq!(tt.lookup((0x1234_5678_9abc_def0, 3)), Some(n(42)));
        assert_eq!(tt.hits(), 1);
    }

    #[test]
    fn different_buckets_disambiguate() {
        let mut tt = TranspositionTable::new();
        tt.insert((0xaa, 1), n(10));
        tt.insert((0xaa, 2), n(20));
        assert_eq!(tt.lookup((0xaa, 1)), Some(n(10)));
        assert_eq!(tt.lookup((0xaa, 2)), Some(n(20)));
    }

    #[test]
    fn clear_drops_entries() {
        let mut tt = TranspositionTable::new();
        tt.insert((1, 0), n(1));
        tt.clear();
        assert!(tt.lookup((1, 0)).is_none());
        assert_eq!(tt.hits(), 0);
    }

    /// Insert N entries — every key should round-trip. Exercises
    /// the linear-probe path on natural collisions for keys that
    /// happen to share a bucket.
    #[test]
    fn many_inserts_round_trip() {
        let mut tt = TranspositionTable::new();
        let n_inserts = 10_000u32;
        for i in 1..=n_inserts {
            tt.insert(
                (u64::from(i).wrapping_mul(0x9e37_79b9_7f4a_7c15), 0),
                NodeId(i),
            );
        }
        let mut found = 0;
        for i in 1..=n_inserts {
            if tt.lookup((u64::from(i).wrapping_mul(0x9e37_79b9_7f4a_7c15), 0)) == Some(NodeId(i)) {
                found += 1;
            }
        }
        // Always-replace policy means a small handful of inserts
        // may have been evicted by later natural-collision storms;
        // at TT_SIZE=2²⁰ and 10k inserts we expect ~99% retention.
        assert!(
            found >= 9_950,
            "expected ≥9950 of 10000 round-trips, got {found}"
        );
    }

    /// Bead `sts2-rs-uef4u`: `peek` is the non-mutating probe — it
    /// must never bump the hit counter, even on repeated hits.
    /// `lookup` and `record_hit` do bump it. This is the invariant
    /// the cycle-guard branch in `expansion` relies on.
    #[test]
    fn peek_does_not_bump_hit_counter() {
        let mut tt = TranspositionTable::new();
        tt.insert((0xdead_beef, 4), n(7));

        // Three peeks on a hitting key — hits stays 0.
        for _ in 0..3 {
            assert_eq!(tt.peek((0xdead_beef, 4)), Some(n(7)));
        }
        assert_eq!(tt.hits(), 0, "peek must not bump hits");

        // One peek on a missing key — still no increment.
        assert!(tt.peek((0xcafe_babe, 4)).is_none());
        assert_eq!(tt.hits(), 0);

        // lookup of the same hitting key bumps to 1.
        assert_eq!(tt.lookup((0xdead_beef, 4)), Some(n(7)));
        assert_eq!(tt.hits(), 1);

        // record_hit bumps externally without re-probing.
        tt.record_hit();
        assert_eq!(tt.hits(), 2);
    }

    /// `NodeId(0)` is a valid stored value (the root node has id 0
    /// and is never inserted by the engine itself, but the test
    /// suite plants it to exercise the cycle guard).
    #[test]
    fn node_zero_is_storable() {
        let mut tt = TranspositionTable::new();
        tt.insert((1, 0), NodeId(0));
        assert_eq!(tt.lookup((1, 0)), Some(NodeId(0)));
    }

    // Silence unused import in non-test builds without complicating
    // the cfg gates.
    #[allow(dead_code)]
    fn _nz(v: u32) -> Option<NonZeroU32> {
        NonZeroU32::new(v)
    }
}
