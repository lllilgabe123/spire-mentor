//! Proxy-fidelity / mutual-information study
//! (`docs/plans/2026-06-25-proxy-fidelity-mi-study.md`).
//!
//! Drives self-play runs exactly like [`crate::feature_export`], but at
//! every card-shaped decision it evaluates a BATTERY of deck-quality
//! proxies on the CURRENT deck (the no-op [`CandidateMutation::Baseline`]
//! arm) and records each proxy's full [`DeltaStats`] + a probe-cost field.
//! Each run's eventual outcome (`won` / `final_act`) is stamped onto its
//! rows post-hoc. The Python `proxy_mi_study.py` then ranks the proxies by
//! conditional mutual information per probe-sim against that outcome —
//! answering "which fight-sim panel best predicts the run result, per unit
//! of compute?".
//!
//! Like the feature exporter this is exact: the proxy probes clone the
//! `RunManager` (`fight_delta`/`chain_delta`/`encounter_delta` never touch
//! the run RNG), so adding the battery does not perturb the driven
//! trajectory — the same `(seed, step)` keys join 1:1 to `features.jsonl`.

use serde::{Deserialize, Serialize};

use sts2_content::encounters::{EncounterDef, EncounterTier};
use sts2_core::full_run::{drive_full_run, FullRunConfig, RunPolicy};
use sts2_core::run_manager::{RunAction, RunManager};

use crate::arena::{build_policy, character_token, classify_outcome, ArenaRow, CellSpec};
use crate::deck_archetype::classify_archetype;
use crate::eval::ranker_features::ContextFeatures;
use crate::label::engine_identity;
use crate::rollout::{
    act_encounter_table, candidates_for_surface, chain_delta, detect_card_surface, encounter_delta,
    fight_delta, CandidateMutation, ChainPlan, DeltaStats, EncounterPlan, FightCache, ProbeBudget,
};
use crate::view::VisibleRunView;

/// One proxy variant's recorded outcome at one snapshot. The full
/// `DeltaStats` component vector is kept (not just `score`) so the
/// scalar-metric hypotheses — damage/turn, hp-loss, velocity — fall out
/// as columns the Python scorer can test individually.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProxyStat {
    /// Stable id: `boss_only` | `elite_only` | `boss_elite` | `chain_k2`
    /// | `random3` | `nextact_boss`.
    pub id: String,
    pub win_rate: f64,
    pub loss_rate: f64,
    pub dmg_t5_mean: f64,
    pub hp_lost_mean: f64,
    pub turns_mean: f64,
    /// `DeltaStats::score()` — the production ranking scalar.
    pub score: f64,
    /// `k * n_encounters * sims` — the MI-per-compute denominator. A
    /// relative cost proxy (probe count × per-move sims), not wall time.
    pub probe_sims: u64,
}

/// One decision's proxy battery + the run's eventual outcome (JSONL line
/// in `proxy.jsonl`). Joins to `features.jsonl` on `(seed, step)`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProxyRow {
    pub seed: i32,
    /// Decision index — same counter as the feature exporter / labels.
    pub step: u64,
    pub floor: i32,
    pub act: u8,
    pub surface: String,
    /// Engine identity at export time ([`engine_identity`]).
    pub engine: String,
    pub context: ContextFeatures,
    pub proxies: Vec<ProxyStat>,
    /// Stamped from the run's `RunOutcome` after the trajectory finishes.
    pub won: u8,
    pub final_act: u8,
}

/// splitmix64 finalizer — local copy (rollout's is private) for the
/// `random3` normal-encounter sampling. Same constants.
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn stat(id: &str, d: &DeltaStats, probe_sims: u64) -> ProxyStat {
    let loss_rate = if d.probes == 0 {
        0.0
    } else {
        f64::from(d.losses) / f64::from(d.probes)
    };
    ProxyStat {
        id: id.to_string(),
        win_rate: d.win_rate,
        loss_rate,
        dmg_t5_mean: d.dmg_dealt_t5_mean,
        hp_lost_mean: d.hp_lost_mean,
        turns_mean: d.turns_mean,
        score: d.score(),
        probe_sims,
    }
}

/// Evaluate the whole battery on the *current* deck (no-op candidate).
/// All variants share the same `EncounterPlan`/`decision_key` the
/// production labeler uses, so identities match what the agent would see.
fn eval_battery(
    rm: &RunManager,
    step: u64,
    budget: &ProbeBudget,
    cache: &FightCache,
) -> Vec<ProxyStat> {
    let noop = CandidateMutation::Baseline;
    // Same decision_key the feature exporter uses (picks the elite
    // identity; does NOT feed the combat seeds).
    let key = step ^ (u64::from(rm.seed as u32) << 24);
    let plan = EncounterPlan::for_decision(rm, key);
    let k = u64::from(budget.k);
    let sims = u64::from(budget.sims);
    let mut out = Vec::new();

    // boss_only / elite_only / boss_elite — same primitive, masked plan.
    if plan.boss.is_some() {
        let boss_plan = EncounterPlan {
            elite: None,
            boss: plan.boss,
        };
        out.push(stat(
            "boss_only",
            &fight_delta(rm, &noop, &boss_plan, budget, cache),
            k * sims,
        ));
    }
    if plan.elite.is_some() {
        let elite_plan = EncounterPlan {
            elite: plan.elite,
            boss: None,
        };
        out.push(stat(
            "elite_only",
            &fight_delta(rm, &noop, &elite_plan, budget, cache),
            k * sims,
        ));
    }
    let n_be = u64::from(plan.boss.is_some()) + u64::from(plan.elite.is_some());
    if n_be > 0 {
        out.push(stat(
            "boss_elite",
            &fight_delta(rm, &noop, &plan, budget, cache),
            k * n_be * sims,
        ));
    }

    // chain_k2 — HP-threaded [2×normal, elite, boss].
    let chain = ChainPlan::for_decision(rm, key, 2);
    let n_links = (chain.normals.len()
        + usize::from(chain.base.elite.is_some())
        + usize::from(chain.base.boss.is_some())) as u64;
    if n_links > 0 {
        out.push(stat(
            "chain_k2",
            &chain_delta(rm, &noop, &chain, budget),
            k * n_links * sims,
        ));
    }

    // random3 — mean over three sampled normal-tier encounters.
    if let Some(table) = act_encounter_table(rm.run_state.act_index, rm.run_state.act1) {
        let normals: Vec<&'static EncounterDef> = table
            .iter()
            .filter(|e| e.tier == EncounterTier::Normal)
            .collect();
        if !normals.is_empty() {
            let (mut wr, mut hp, mut dmg, mut turns, mut score) = (0.0, 0.0, 0.0, 0.0, 0.0);
            let (mut losses, mut probes) = (0u32, 0u32);
            for j in 0..3u64 {
                let idx = (splitmix64(key ^ (0xA3A3_0000_0000_0001 ^ j)) as usize) % normals.len();
                let d = encounter_delta(rm, &noop, normals[idx], budget);
                wr += d.win_rate;
                hp += d.hp_lost_mean;
                dmg += d.dmg_dealt_t5_mean;
                turns += d.turns_mean;
                score += d.score();
                losses += d.losses;
                probes += d.probes;
            }
            out.push(ProxyStat {
                id: "random3".to_string(),
                win_rate: wr / 3.0,
                loss_rate: if probes == 0 {
                    0.0
                } else {
                    f64::from(losses) / f64::from(probes)
                },
                dmg_t5_mean: dmg / 3.0,
                hp_lost_mean: hp / 3.0,
                turns_mean: turns / 3.0,
                score: score / 3.0,
                probe_sims: k * 3 * sims,
            });
        }
    }

    // nextact_boss — peek the next act's boss (deck-scaling horizon).
    if let Some(table) = act_encounter_table(rm.run_state.act_index + 1, rm.run_state.act1) {
        if let Some(boss) = table.iter().find(|e| e.tier == EncounterTier::Boss) {
            out.push(stat(
                "nextact_boss",
                &encounter_delta(rm, &noop, boss, budget),
                k * sims,
            ));
        }
    }

    out
}

/// Policy wrapper mirroring `FeatureExportPolicy`'s decision counting and
/// labelability gate; emits a `ProxyRow` battery instead of feature rows.
#[derive(Debug)]
struct ProxyStudyPolicy<P> {
    inner: P,
    seed: i32,
    step: u64,
    budget: ProbeBudget,
    cache: FightCache,
    /// Evaluate the (expensive) battery on every `snapshot_every`-th
    /// eligible decision; `1` = all. Subsampling keeps the offline study
    /// affordable without biasing — log the rate so partial ≠ "full".
    snapshot_every: u64,
    eligible_seen: u64,
    rows: Vec<ProxyRow>,
}

impl<P: RunPolicy> RunPolicy for ProxyStudyPolicy<P> {
    fn prepare(&mut self, rm: &mut RunManager) {
        // Forward the per-decision hook (see LabelingPolicy for rationale);
        // no-op for inner policies that don't override `prepare`.
        self.inner.prepare(rm);
    }

    fn choose(&mut self, rm: &RunManager, actions: &[RunAction]) -> Option<usize> {
        let step = self.step;
        self.step += 1;

        let chosen_idx = self.inner.choose(rm, actions)?;

        if let Some(surface) = detect_card_surface(rm, actions) {
            let view = VisibleRunView::from_run_manager(rm);
            let candidates = candidates_for_surface(rm, &view, surface, actions);
            if candidates.len() >= 2 {
                let key = step ^ (u64::from(rm.seed as u32) << 24);
                let plan = EncounterPlan::for_decision(rm, key);
                if plan.elite.is_some() || plan.boss.is_some() {
                    let take = self.eligible_seen.is_multiple_of(self.snapshot_every);
                    self.eligible_seen += 1;
                    if take {
                        let proxies = eval_battery(rm, step, &self.budget, &self.cache);
                        if !proxies.is_empty() {
                            self.rows.push(ProxyRow {
                                seed: self.seed,
                                step,
                                floor: rm.run_state.total_floor,
                                act: rm.run_state.act_index,
                                surface: surface.token().to_string(),
                                engine: engine_identity().to_string(),
                                context: ContextFeatures::from_view(&view),
                                proxies,
                                won: 0,
                                final_act: 0,
                            });
                        }
                    }
                }
            }
        }

        Some(chosen_idx)
    }
}

/// Drive one self-play run and return its `ArenaRow` + outcome-stamped
/// proxy rows. The trajectory is driven by the FULL deployed policy
/// (`spec.policy` + `spec.combat` + the env-loaded ranker/value weights),
/// via the arena's [`build_policy`] — so the outcome distribution the
/// proxies are scored against is on-policy for the real agent (a heuristic
/// off-policy driver wins ~0 and biases the study). `probe` is the
/// per-proxy MCTS probe budget, independent of the driving combat budget.
#[must_use]
pub fn run_proxy_cell(
    spec: &CellSpec,
    probe: &ProbeBudget,
    snapshot_every: u64,
) -> (ArenaRow, Vec<ProxyRow>) {
    crate::rollout::ensure_content_installed();
    let cfg = FullRunConfig {
        max_steps: spec.max_steps,
        livelock_stall_limit: spec.livelock_stall_limit,
        invincible: false,
        record_trace: false,
    };
    let started = std::time::Instant::now();

    let mut rm = sts2_content::run_callbacks::new_run_manager_for_character_with_ascension(
        spec.seed,
        spec.character,
        spec.ascension,
    )
    .with_neow_start();
    sts2_content::run_callbacks::initialize_run_setup(&mut rm);

    let mut policy = ProxyStudyPolicy {
        inner: build_policy(spec),
        seed: spec.seed,
        step: 0,
        budget: *probe,
        cache: FightCache::default(),
        snapshot_every: snapshot_every.max(1),
        eligible_seen: 0,
        rows: Vec::new(),
    };
    let report = drive_full_run(&mut rm, &mut policy, &cfg);

    let (outcome, won, bin) = classify_outcome(&report.outcome, report.final_phase);

    // Stamp the eventual outcome onto every snapshot of this run.
    let won_u8 = u8::from(won);
    for row in &mut policy.rows {
        row.won = won_u8;
        row.final_act = report.act_index;
    }

    let final_view = VisibleRunView::from_run_manager(&rm);
    let final_deck: Vec<String> = final_view
        .deck
        .iter()
        .map(|dc| {
            if dc.upgraded {
                format!("{:?}+", dc.card_id)
            } else {
                format!("{:?}", dc.card_id)
            }
        })
        .collect();
    let final_archetype = classify_archetype(&final_view).token().to_string();
    let row = ArenaRow {
        seed: spec.seed,
        character: character_token(spec.character).to_string(),
        ascension: spec.ascension,
        act1: spec.act1.resolved_token(spec.seed),
        policy: format!("proxy+{}", crate::arena::policy_label(spec)),
        combat_budget: spec.combat.token().to_string(),
        outcome,
        won,
        steps: report.steps,
        final_phase: format!("{:?}", report.final_phase),
        final_hp: report.final_hp,
        final_gold: report.final_gold,
        act_index: report.act_index,
        final_relics: rm.run_state.relics.len(),
        wall_ms: started.elapsed().as_millis() as u64,
        noncompletion_bin: bin,
        escalated_t2: 0,
        escalated_t3: 0,
        combat_decisions: 0,
        combat_decisions_elite: 0,
        combat_decisions_boss: 0,
        combat_decisions_motif: 0,
        combat_sims_spent: 0,
        combat_fallback_untranslatable: 0,
        combat_fallback_no_outcome: 0,
        combat_fallback_not_surfaced: 0,
        final_deck,
        final_archetype,
        sample_seed: 0,
        potions_overflow_dropped: rm.run_state.potions_overflow_dropped,
        // Label/proxy harnesses never run the PPO net and never bin a
        // panic row (they drive inside the caller's unwind scope).
        ppo_peak_act_slots: 0,
        ppo_peak_tokens: 0,
        reached_second_boss: crate::arena::reached_second_boss(&rm),
        panic_message: None,
    };
    (row, policy.rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::{CombatBudgetKind, PolicyKind};
    use sts2_core::full_run::DEFAULT_LIVELOCK_STALL_LIMIT;
    use sts2_core::potions::CharacterId;

    fn spec(seed: i32) -> CellSpec {
        CellSpec {
            seed,
            character: CharacterId::Ironclad,
            ascension: 0,
            policy: PolicyKind::Heuristic,
            combat: CombatBudgetKind::None,
            sims_override: None,
            boss_sims_override: None,
            max_steps: 3000,
            livelock_stall_limit: DEFAULT_LIVELOCK_STALL_LIMIT,
            act1: crate::arena::Act1Choice::default(),
            audit: false,
            ablate: None,
        }
    }

    /// Smoke: a cheap-probe study run produces outcome-stamped rows whose
    /// proxy batteries are non-empty and whose `(seed, step)` are unique.
    #[test]
    fn proxy_rows_are_stamped_and_keyed() {
        let probe = ProbeBudget {
            sims: 2,
            k: 2,
            t_turns: 1,
            leaf: crate::policy::LeafKind::TexelWide,
            horizon: crate::rollout::LabelHorizon::default(),
            chain_normals: 2,
        };
        let (arena, rows) = run_proxy_cell(&spec(3), &probe, 1);
        assert!(!rows.is_empty(), "study produced rows");
        let mut steps = std::collections::HashSet::new();
        for r in &rows {
            assert!(steps.insert(r.step), "unique step per row");
            assert!(!r.proxies.is_empty(), "battery non-empty");
            assert_eq!(r.won, u8::from(arena.won), "won stamped from run outcome");
            assert!(r.proxies.iter().all(|p| p.probe_sims > 0));
            // boss_elite is the production baseline — present whenever any
            // boss/elite plan exists (which the gate guarantees).
            assert!(r.proxies.iter().any(|p| p.id == "boss_elite"));
        }
        // JSONL round-trip (ids + integer cost; floats aren't asserted
        // exactly — serde_json's shortest-decimal repr makes that brittle).
        let line = serde_json::to_string(&rows[0]).expect("serializes");
        let back: ProxyRow = serde_json::from_str(&line).expect("parses");
        assert_eq!(back.step, rows[0].step);
        assert_eq!(back.proxies.len(), rows[0].proxies.len());
        let ids: Vec<&str> = back.proxies.iter().map(|p| p.id.as_str()).collect();
        let want: Vec<&str> = rows[0].proxies.iter().map(|p| p.id.as_str()).collect();
        assert_eq!(ids, want);
        assert!(back
            .proxies
            .iter()
            .zip(&rows[0].proxies)
            .all(|(a, b)| a.probe_sims == b.probe_sims));
    }
}
