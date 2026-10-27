//! Process-wide counters of the PUCT move prior (GH #830): how often a
//! search's descent read a node's prior, how often a net prior was actually
//! evaluated, and how often such an evaluation fell back to uniform.
//!
//! With the per-node prior cache ([`crate::Mcts::with_prior_cache`]) on, an
//! evaluation is a cache miss, so `evals < lookups`; with it off, every read
//! evaluates and `evals == lookups`. A search without
//! [`crate::MctsConfig::use_puct_prior`] counts nothing. Plain `Relaxed`
//! atomics: they order nothing and never steer a search. Not on wasm32, where
//! the counting functions do nothing and [`snapshot`] reads zeros.

#[cfg(not(target_arch = "wasm32"))]
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

#[cfg(not(target_arch = "wasm32"))]
static LOOKUPS: AtomicU64 = AtomicU64::new(0);
#[cfg(not(target_arch = "wasm32"))]
static EVALS: AtomicU64 = AtomicU64::new(0);
#[cfg(not(target_arch = "wasm32"))]
static UNIFORM_FALLBACKS: AtomicU64 = AtomicU64::new(0);

/// The prior counters since the process started ([`snapshot`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PriorCounts {
    /// PUCT prior reads in a search's descent, cache hits included, under
    /// any prior.
    pub lookups: u64,
    /// Evaluations of a playout-net prior
    /// ([`crate::rollout_net::RolloutNetPrior`]): the searches' prior-cache
    /// misses, or every read with the cache off.
    pub evals: u64,
    /// Those evaluations whose net mass over the node's edges was NaN or
    /// zero, so the prior fell back to uniform.
    pub uniform_fallbacks: u64,
}

/// Count one PUCT prior read in a search's descent.
#[inline]
pub(crate) fn count_lookup() {
    #[cfg(not(target_arch = "wasm32"))]
    LOOKUPS.fetch_add(1, Relaxed);
}

/// Count one evaluation of a playout-net prior.
#[inline]
pub(crate) fn count_eval() {
    #[cfg(not(target_arch = "wasm32"))]
    EVALS.fetch_add(1, Relaxed);
}

/// Count one playout-net prior evaluation that fell back to uniform.
#[inline]
pub(crate) fn count_fallback() {
    #[cfg(not(target_arch = "wasm32"))]
    UNIFORM_FALLBACKS.fetch_add(1, Relaxed);
}

/// The counters now. All zero on wasm32.
#[must_use]
pub fn snapshot() -> PriorCounts {
    #[cfg(not(target_arch = "wasm32"))]
    {
        PriorCounts {
            lookups: LOOKUPS.load(Relaxed),
            evals: EVALS.load(Relaxed),
            uniform_fallbacks: UNIFORM_FALLBACKS.load(Relaxed),
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        PriorCounts::default()
    }
}
