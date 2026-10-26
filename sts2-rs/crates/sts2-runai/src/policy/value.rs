//! `ValuePolicy` — heuristic meta policy with the learned run-value net
//! ([`ValueNet`]) scoring card-shaped picks by `V(resulting state)`
//! (full-run-AI plan §5/M4 E9; bead `sts2-rs-40ikw.8`).
//!
//! On a card-shaped surface, each candidate mutation is applied to the
//! visible run-view, the resulting state is encoded to [`ContextFeatures`]
//! and scored by the value net (expected final-act-reached), and the
//! argmax is the pick. Everything else (pathing, events, shop, rest,
//! combat) stays with the wrapped [`HeuristicPolicy`]. This is the E9
//! "learned-V on the card surface vs Texel-V" arm (it was directly
//! comparable to the since-removed ranker policy on the same surface).

use std::sync::Arc;

use sts2_core::full_run::RunPolicy;
use sts2_core::run_manager::{Phase, RunAction, RunManager};

use crate::eval::ranker_features::ContextFeatures;
use crate::eval::run_value::ValueNet;
use crate::eval::Weights;
use crate::policy::surfaces::pathing;
use crate::policy::{HeuristicPolicy, VScoreSource};
use crate::rollout::{candidates_for_surface, detect_card_surface};
use crate::view::VisibleRunView;

/// Heuristic policy with value-net-scored card-shaped picks.
#[derive(Debug)]
pub struct ValuePolicy {
    inner: HeuristicPolicy,
    net: Arc<ValueNet>,
    /// Decisions actually scored by the net (sweep diagnostics).
    net_decisions: u64,
    last_scores: Option<(Vec<f64>, f64)>,
}

impl ValuePolicy {
    #[must_use]
    pub fn new(net: Arc<ValueNet>) -> Self {
        Self::with_weights(net, Weights::default())
    }

    /// `weights` configure the fallback heuristic (non-card surfaces).
    #[must_use]
    pub fn with_weights(net: Arc<ValueNet>, weights: Weights) -> Self {
        Self {
            inner: HeuristicPolicy::with_weights(weights),
            net,
            net_decisions: 0,
            last_scores: None,
        }
    }

    /// Decisions the net (not the heuristic) decided so far.
    #[must_use]
    pub fn net_decisions(&self) -> u64 {
        self.net_decisions
    }
}

impl RunPolicy for ValuePolicy {
    fn choose(&mut self, rm: &RunManager, actions: &[RunAction]) -> Option<usize> {
        let tier1_idx = self.inner.choose(rm, actions)?;
        let tier1_scores = self.inner.take_v_scores();
        self.last_scores.clone_from(&tier1_scores);

        // M4 V-pathing (plan §5/M4): rank map edges by the learned V of
        // each path's projected economy — the surface the ranker can't
        // touch (a Move doesn't change deck/HP until the room resolves).
        if rm.phase() == Phase::MapChoice {
            let view = VisibleRunView::from_run_manager(rm);
            if let Some(col) = pathing::best_first_col_by_value(&view, self.inner.v(), &self.net) {
                if let Some(idx) = actions
                    .iter()
                    .position(|a| matches!(a, RunAction::Move { col: c } if *c == col))
                {
                    self.net_decisions += 1;
                    return Some(idx);
                }
            }
            return Some(tier1_idx);
        }

        let Some(surface) = detect_card_surface(rm, actions) else {
            return Some(tier1_idx);
        };
        let view = VisibleRunView::from_run_manager(rm);
        let candidates = candidates_for_surface(rm, &view, surface, actions);
        if candidates.len() < 2 {
            return Some(tier1_idx);
        }

        // Score each candidate by V of the state it produces.
        let mut scores = Vec::with_capacity(candidates.len());
        let mut best: Option<(usize, f64)> = None;
        for (mutation, action_idx) in &candidates {
            let resulting = mutation.apply_to_view(&view);
            let v = self.net.value(&ContextFeatures::from_view(&resulting));
            scores.push(v);
            if best.is_none_or(|(_, bv)| v > bv) {
                best = Some((*action_idx, v));
            }
        }
        self.net_decisions += 1;

        // Audit scores: the net's per-candidate V + top-2 margin.
        let mut sorted = scores.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let margin = match sorted.as_slice() {
            [first, second, ..] => first - second,
            _ => 0.0,
        };
        self.last_scores = Some((scores, margin));

        best.map_or(Some(tier1_idx), |(idx, _)| {
            Some(idx.min(actions.len().saturating_sub(1)))
        })
    }
}

impl VScoreSource for ValuePolicy {
    fn take_v_scores(&mut self) -> Option<(Vec<f64>, f64)> {
        self.last_scores.take()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::ranker_features::{context_dim, feature_schema_id};
    use sts2_content::run_callbacks::new_run_manager_for_character;
    use sts2_core::full_run::{drive_full_run, FullRunConfig, RunOutcome};
    use sts2_core::potions::CharacterId;

    /// A schema-matching 2-hidden-unit value net (same shape as
    /// `run_value`'s own `tiny_net_json` fixture): identity feature
    /// standardization (`feat_mu=0`, `feat_sd=1`), so V is a fixed
    /// function of the encoded context. Any well-formed net is enough to
    /// exercise the policy's scoring/argmax plumbing.
    fn fixture_net() -> Arc<ValueNet> {
        let cdim = context_dim();
        let json = serde_json::json!({
            "schema_id": feature_schema_id(),
            "context_dim": cdim,
            "hidden": 2,
            "feat_mu": vec![0.0f32; cdim],
            "feat_sd": vec![1.0f32; cdim],
            "y_mu": 0.0,
            "y_sd": 1.0,
            "w1": vec![vec![0.05f32, -0.05f32]; cdim],
            "b1": [0.1f32, -0.1f32],
            "w2": [1.0f32, -1.0f32],
            "b2": 0.0f32,
        })
        .to_string();
        Arc::new(ValueNet::from_json_slice(json.as_bytes()).expect("fixture net loads"))
    }

    #[test]
    fn new_and_with_weights_start_with_zero_net_decisions() {
        let net = fixture_net();
        let p = ValuePolicy::new(net.clone());
        assert_eq!(p.net_decisions(), 0);
        let p2 = ValuePolicy::with_weights(net, Weights::default());
        assert_eq!(p2.net_decisions(), 0);
    }

    /// Every `choose()` returns an in-range index across the first 60
    /// decisions of a real run, and the value net decides at least one
    /// card-shaped (or V-pathing) pick — proving the net branches are
    /// actually reached, not just the heuristic fallback.
    #[test]
    fn value_policy_returns_valid_indices_and_uses_the_net() {
        let net = fixture_net();
        let mut rm = new_run_manager_for_character(7, CharacterId::Ironclad);
        let mut policy = ValuePolicy::new(net);
        for _ in 0..60 {
            if rm.is_over() {
                break;
            }
            let actions = rm.get_available_actions();
            if actions.is_empty() {
                break;
            }
            let idx = policy.choose(&rm, &actions).expect("a pick");
            assert!(idx < actions.len(), "index {idx} out of range");
            // The VScoreSource surface must hand back a score vector
            // whenever the net actually scored a multi-candidate surface.
            let _ = policy.take_v_scores();
            let _ = rm.take_action(actions[idx]);
        }
        assert!(
            policy.net_decisions() > 0,
            "the value net must decide at least one pick over 60 decisions"
        );
    }

    /// Multi-seed smoke: the value policy drives full runs to a real
    /// terminal outcome (never hitting the `MaxSteps` safety cap).
    #[test]
    fn value_policy_three_seed_smoke() {
        let net = fixture_net();
        let mut total_net_decisions = 0;
        for seed in [3, 5, 11] {
            let mut rm = new_run_manager_for_character(seed, CharacterId::Ironclad);
            let mut policy = ValuePolicy::new(net.clone());
            let cfg = FullRunConfig::default();
            let report = drive_full_run(&mut rm, &mut policy, &cfg);
            assert!(
                !matches!(report.outcome, RunOutcome::MaxSteps),
                "seed {seed} did not terminate: {:?}",
                report.outcome
            );
            total_net_decisions += policy.net_decisions();
        }
        assert!(
            total_net_decisions > 0,
            "the net must decide at least one pick across 3 seeds"
        );
    }
}
