//! GH #830, a net as the move prior: the per-node PUCT prior cache
//! ([`Mcts::with_prior_cache`]) against the uncached search, the prior's
//! blindness to the hidden draw order, and the prior counters
//! ([`crate::prior_counts`]).
//!
//! Lives inside the crate so it can read the cache. The rl2-kind tests play
//! the built-in Ironclad playout net as the prior and compile only when that
//! net does (`cfg(rollout_net_ironclad)`). The bc-kind tests play the
//! built-in Ironclad move prior, the deployed one (`data/prior_nets`,
//! `cfg(prior_net_ironclad)`). The counters are process-wide: nextest runs
//! each test in its own process.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use super::*;
use crate::prior_counts::{snapshot, PriorCounts};
use crate::rollout_net::{RolloutNet, RolloutNetPrior};
#[cfg(rollout_net_ironclad)]
use sts2_content::encounters::{act1_overgrowth, EncounterTier};
#[cfg(rollout_net_ironclad)]
use sts2_content::run_bridge::enter_combat_from_roster;
use sts2_core::run_manager::{Phase, RunManager};
#[cfg(rollout_net_ironclad)]
use sts2_core::runstate::RoomType;
use sts2_core::CharacterId;

const SIMS: u32 = 300;

fn install() {
    sts2_content::cards::dispatch::ensure_installed();
    sts2_content::relics::dispatch::ensure_installed();
    sts2_content::potions::dispatch::ensure_installed();
    sts2_core::relic_effects::ensure_installed();
}

/// The built-in Ironclad playout net's text (`data/rollout_nets`), as
/// `rollout_net.rs` compiles it in.
#[cfg(rollout_net_ironclad)]
const IRONCLAD_NET: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../data/rollout_nets/ironclad.json"
));

/// The built-in Ironclad playout net, parsed afresh (a `RolloutNet` is not
/// `Clone`, and the prior owns its net).
#[cfg(rollout_net_ironclad)]
fn rl2_net() -> RolloutNet {
    install();
    RolloutNet::from_json(IRONCLAD_NET).expect("the built-in net parses")
}

/// That net as the prior, at the gated knobs (mix 0.5, T 1).
#[cfg(rollout_net_ironclad)]
fn rl2_prior() -> SharedPolicy {
    SharedPolicy::new(RolloutNetPrior::new(rl2_net(), 0.5, 1.0))
}

/// The built-in behaviour-cloned Ironclad move prior
/// ([`crate::rollout_net::builtin_prior_json`], `data/prior_nets`): the one
/// the Ironclad run agent deploys, parsed afresh.
#[cfg(prior_net_ironclad)]
fn bc_net() -> RolloutNet {
    install();
    let text = crate::rollout_net::builtin_prior_json(CharacterId::Ironclad)
        .expect("the Ironclad prior is compiled in");
    RolloutNet::from_json(text).expect("the built-in prior parses")
}

/// That net as the prior, at the gated knobs (mix 0.5, T 1).
#[cfg(prior_net_ironclad)]
fn bc_prior() -> SharedPolicy {
    SharedPolicy::new(RolloutNetPrior::new(bc_net(), 0.5, 1.0))
}

/// A fresh Ironclad run (seed `seed`) at its first combat, reached by
/// first-option picks, with its deck doubled before it (20 cards: the
/// second turn still has a draw pile to permute).
fn run_at_first_combat(seed: i32) -> RunManager {
    install();
    let mut rm: RunManager =
        sts2_content::run_callbacks::new_run_manager_for_character_with_ascension(
            seed,
            CharacterId::Ironclad,
            0,
        )
        .with_neow_start();
    sts2_content::run_callbacks::initialize_run_setup(&mut rm);
    let copy = rm.run_state.master_deck.clone();
    rm.run_state.master_deck.extend(copy);
    for _ in 0..64 {
        if rm.phase() == Phase::Combat {
            return rm;
        }
        let actions = rm.get_available_actions();
        assert!(!actions.is_empty(), "run stalled before its first combat");
        let _ = rm.take_action(actions[0]);
    }
    panic!("no combat within 64 run actions");
}

/// The first act-1 encounter of `tier`, built for `rm`'s run the way the
/// run path builds a `room` ([`enter_combat_from_roster`]).
#[cfg(rollout_net_ironclad)]
fn room_fight(rm: &RunManager, tier: EncounterTier, room: RoomType) -> CombatState {
    let enc = act1_overgrowth::ACT1_ENCOUNTERS
        .iter()
        .find(|e| e.tier == tier)
        .expect("act 1 has the tier");
    let roster = enc.canonical_roster();
    let kinds: Vec<_> = roster.iter().map(|(k, _)| *k).collect();
    let hps = sts2_core::monsters::roll_unique_max_hps_for(
        &kinds,
        &mut rm.run_state.rng.niche.clone(),
        0,
        sts2_core::GameVersion::default(),
    );
    let roster: Vec<_> = roster
        .into_iter()
        .zip(hps)
        .map(|((kind, args), hp)| (kind, args, hp))
        .collect();
    let s = enter_combat_from_roster(&rm.run_state, &roster, room);
    assert_eq!(s.player.character, Some(CharacterId::Ironclad));
    s
}

/// The exactness fixtures: a hallway, an elite and a boss of one run.
#[cfg(rollout_net_ironclad)]
fn fixtures() -> Vec<(&'static str, CombatState)> {
    let rm = run_at_first_combat(4);
    let hallway = rm.combat.clone().expect("in combat");
    let elite = room_fight(&rm, EncounterTier::Elite, RoomType::Elite);
    let boss = room_fight(&rm, EncounterTier::Boss, RoomType::Boss);
    assert!(elite.is_elite && boss.is_boss);
    vec![("hallway", hallway), ("elite", elite), ("boss", boss)]
}

/// The deployed honest search (`BeliefDraw` and its dialect, depth 200)
/// with the PUCT prior on.
fn prior_config(sims: u32, seed: u64) -> MctsConfig {
    let mut cfg = MctsConfig {
        max_simulations: sims,
        seed,
        max_rollout_depth: 200,
        hidden_info: HiddenInfoMode::BeliefDraw,
        ..MctsConfig::default()
    };
    cfg.apply_mode_dialect();
    cfg.use_puct_prior = true;
    cfg
}

/// A prior that counts its calls.
#[derive(Clone)]
struct Counted<P> {
    inner: P,
    calls: Arc<AtomicU64>,
}

impl<P: Policy> Policy for Counted<P> {
    fn prior(&self, state: &CombatState, actions: &[Action]) -> SmallVec<[f32; 16]> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.inner.prior(state, actions)
    }
}

/// Every node's statistics, floats as bits, and its child count.
fn tree_bits<P: Policy>(m: &Mcts<P>) -> Vec<(u32, u32, usize)> {
    m.tree
        .arena
        .iter()
        .map(|n| (n.visits, n.value_sum.to_bits(), n.children.len()))
        .collect()
}

/// Entries in the prior cache.
fn cached<P: Policy>(m: &Mcts<P>) -> usize {
    m.prior_cache.iter().flatten().count()
}

fn bits(p: &[f32]) -> Vec<u32> {
    p.iter().map(|x| x.to_bits()).collect()
}

/// A search of `state` under `cfg` and `prior`, the cache `on` or off: its
/// pick, its tree, the prior's calls and the cache's entries.
#[cfg(rollout_net_ironclad)]
fn search_once(
    state: &CombatState,
    cfg: MctsConfig,
    prior: &SharedPolicy,
    on: bool,
) -> (Action, Vec<(u32, u32, usize)>, u64, usize) {
    let calls = Arc::new(AtomicU64::new(0));
    let policy = Counted {
        inner: prior.clone(),
        calls: Arc::clone(&calls),
    };
    let mut m = Mcts::with_policy(state.clone(), cfg, policy)
        .with_prior_cache(on)
        .with_leaf_eval(LeafEval::TexelWide);
    let pick = m.try_search_n(cfg.max_simulations).expect("searchable");
    (
        pick,
        tree_bits(&m),
        calls.load(Ordering::Relaxed),
        cached(&m),
    )
}

/// With the cache on and off, every fixture's search is the same, bit for
/// bit on every node, and picks the same move; the cached one calls the
/// prior less (the cache served reads), the uncached one caches nothing.
#[cfg(rollout_net_ironclad)]
fn assert_cache_exact(kind: &str, prior: &SharedPolicy, fpu: Option<f32>) {
    for (fixture, state) in fixtures() {
        let mut cfg = prior_config(SIMS, 0x5EED);
        cfg.fpu_reduction = fpu;
        let (pick_on, tree_on, calls_on, entries) = search_once(&state, cfg, prior, true);
        let (pick_off, tree_off, calls_off, none) = search_once(&state, cfg, prior, false);
        let what = format!("{kind} prior, {fixture}, fpu {fpu:?}");
        assert_eq!(pick_on, pick_off, "{what}: pick");
        assert_eq!(tree_on.len(), tree_off.len(), "{what}: arena size");
        assert_eq!(tree_on, tree_off, "{what}: (visits, value_sum, children)");
        assert_eq!(none, 0, "{what}: the uncached search cached");
        assert!(
            entries > 0 && calls_on < calls_off,
            "{what}: the cache served nothing ({calls_on} calls cached, {calls_off} uncached)"
        );
    }
}

#[cfg(rollout_net_ironclad)]
#[test]
fn the_prior_cache_is_exact_for_the_rl2_prior() {
    assert_cache_exact("rl2", &rl2_prior(), None);
}

#[cfg(rollout_net_ironclad)]
#[test]
fn the_prior_cache_is_exact_for_the_rl2_prior_under_fpu() {
    assert_cache_exact("rl2", &rl2_prior(), Some(0.2));
}

#[cfg(all(rollout_net_ironclad, prior_net_ironclad))]
#[test]
fn the_prior_cache_is_exact_for_the_bc_prior() {
    assert_cache_exact("bc", &bc_prior(), None);
}

#[cfg(all(rollout_net_ironclad, prior_net_ironclad))]
#[test]
fn the_prior_cache_is_exact_for_the_bc_prior_under_fpu() {
    assert_cache_exact("bc", &bc_prior(), Some(0.2));
}

/// The cache is emptied wherever a node, its state or the prior changes:
/// `with_root_allowed`, `set_policy`, `reroot` and `reroot_observed`
/// (through a chance edge), and `with_prior_cache`. Any prior fills it, so
/// the uniform one does here.
#[test]
fn the_prior_cache_is_cleared_wherever_the_tree_or_the_prior_changes() {
    let start = run_at_first_combat(4).combat.expect("in combat");
    let searched = |cfg: MctsConfig| {
        let mut m = Mcts::new(start.clone(), cfg).with_leaf_eval(LeafEval::TexelWide);
        m.try_search_n(cfg.max_simulations).expect("searchable");
        assert!(cached(&m) > 0, "the search cached no prior");
        m
    };
    let honest = prior_config(SIMS, 11);

    let m = searched(honest);
    let play = legal_actions(&start)[0].clone();
    let m = m.with_root_allowed(&[play]);
    assert_eq!(cached(&m), 0, "with_root_allowed");

    let mut m = searched(honest);
    m.set_policy(UniformPolicy);
    assert_eq!(cached(&m), 0, "set_policy");

    let m = searched(honest).with_prior_cache(true);
    assert_eq!(cached(&m), 0, "with_prior_cache");

    let mut m = searched(honest);
    let chance = m
        .tree
        .node(m.tree.root)
        .children
        .iter()
        .find(|(e, _)| *e == EdgeAction::Other(Action::EndTurn))
        .map(|(_, c)| *c)
        .expect("EndTurn expanded");
    let outcome = m.tree.node(chance).chance_outcomes[0].child;
    let seen = m
        .tree
        .node(outcome)
        .realized
        .as_deref()
        .expect("state")
        .clone();
    let kept = m
        .reroot_observed(&Action::EndTurn, &seen, prior_config(SIMS, 12))
        .expect("a stored outcome matches");
    assert!(kept.via_chance);
    assert_eq!(cached(&m), 0, "reroot_observed");

    let mut clairvoyant = honest;
    clairvoyant.hidden_info = HiddenInfoMode::Clairvoyant;
    let mut m = searched(clairvoyant);
    let (edge, _) = m
        .tree
        .node(m.tree.root)
        .children
        .iter()
        .find(|(e, _)| matches!(e, EdgeAction::PlayCard { .. }))
        .cloned()
        .expect("a card play expanded");
    let play = edge.resolve(&m.tree.root_state).expect("root edge");
    let mut next = m.tree.root_state.clone();
    let _ = core_apply(&mut next, play.clone());
    assert!(m.reroot(&play, next), "reroot keeps the played child");
    assert_eq!(cached(&m), 0, "reroot");
}

/// Tree reuse: a kept root (through `EndTurn`) whose realised hand is in
/// another order than the stored outcome's. The kept tree's cache would
/// serve stale vectors without the clear (asserted: some kept node's new
/// slot holds an entry of its child count whose bits differ from a fresh
/// call), and the cached and uncached trees stay identical on every node.
#[cfg(rollout_net_ironclad)]
fn assert_cache_exact_after_reuse(kind: &str, prior: &SharedPolicy) {
    // A larger first search, so the kept subtree has nodes for a stale
    // entry to land on.
    const FIRST: u32 = 4 * SIMS;
    let start = run_at_first_combat(4).combat.expect("in combat");
    let build = |on: bool| {
        let mut m = Mcts::with_policy(start.clone(), prior_config(FIRST, 11), prior.clone())
            .with_prior_cache(on)
            .with_leaf_eval(LeafEval::TexelWide);
        m.try_search_n(FIRST).expect("searchable");
        m
    };
    let (mut a, mut b) = (build(true), build(false));
    assert_eq!(tree_bits(&a), tree_bits(&b), "{kind}: before the reroot");

    let chance = a
        .tree
        .node(a.tree.root)
        .children
        .iter()
        .find(|(e, _)| *e == EdgeAction::Other(Action::EndTurn))
        .map(|(_, c)| *c)
        .expect("EndTurn expanded");
    let outcome = a
        .tree
        .node(chance)
        .chance_outcomes
        .iter()
        .map(|o| o.child)
        .max_by_key(|&c| a.tree.node(c).visits)
        .expect("EndTurn has an outcome");
    let stored = a
        .tree
        .node(outcome)
        .realized
        .as_deref()
        .expect("state")
        .clone();
    let mut realised = stored.clone();
    realised.player.hand.reverse();
    assert_ne!(
        realised.player.hand, stored.player.hand,
        "{kind}: nothing to reorder"
    );
    assert_eq!(
        sts2_core::chance_outcome_hash(&realised),
        sts2_core::chance_outcome_hash(&stored),
        "{kind}: the reorder changed what the player sees"
    );

    let stale = a.prior_cache.clone();
    let ka = a
        .reroot_observed(&Action::EndTurn, &realised, prior_config(SIMS, 12))
        .expect("a stored outcome matches");
    let kb = b
        .reroot_observed(&Action::EndTurn, &realised, prior_config(SIMS, 12))
        .expect("a stored outcome matches");
    assert!(ka.via_chance && ka == kb, "{kind}: {ka:?} / {kb:?}");
    let root_state = a.tree.node(a.tree.root).realized.as_deref().expect("root");
    assert_ne!(
        root_state.player.hand, stored.player.hand,
        "{kind}: the kept root holds the stored hand order"
    );
    let served_stale = (0..a.tree.arena.len())
        .filter(|&i| {
            let n = &a.tree.arena[i];
            let Some(Some((count, old))) = stale.get(i) else {
                return false;
            };
            if n.kind != NodeKind::Decision || usize::from(*count) != n.children.len() {
                return false;
            }
            let state = n.realized.as_deref().expect("decision state");
            let actions: SmallVec<[Action; 16]> = n
                .children
                .iter()
                .map(|(e, _)| e.resolve(state).unwrap_or(Action::EndTurn))
                .collect();
            bits(&a.policy.prior(state, &actions)) != bits(old)
        })
        .count();
    assert!(
        served_stale > 0,
        "{kind}: no kept node would read a stale entry; the case cannot catch a missing clear"
    );

    let pa = a.try_search_n(SIMS).expect("searchable");
    let pb = b.try_search_n(SIMS).expect("searchable");
    assert_eq!(pa, pb, "{kind}: pick after the reroot");
    assert_eq!(tree_bits(&a), tree_bits(&b), "{kind}: after the reroot");
    assert!(cached(&a) > 0 && cached(&b) == 0, "{kind}");
}

#[cfg(rollout_net_ironclad)]
#[test]
fn the_prior_cache_is_exact_on_a_reused_tree_rl2() {
    assert_cache_exact_after_reuse("rl2", &rl2_prior());
}

#[cfg(all(rollout_net_ironclad, prior_net_ironclad))]
#[test]
fn the_prior_cache_is_exact_on_a_reused_tree_bc() {
    assert_cache_exact_after_reuse("bc", &bc_prior());
}

// ── blind to the hidden draw order ───────────────────────────────────────

/// The child edges a fresh decision node at `state` holds, resolved.
fn edge_actions(state: &CombatState) -> SmallVec<[Action; 16]> {
    decision_edges(state, false)
        .iter()
        .map(|e| e.resolve(state).expect("a fresh edge resolves"))
        .collect()
}

/// `state` with its draw pile reordered three ways (reversed, rotated by
/// one, rotated by two), each asserted to differ from the original and from
/// the others, its top card included.
fn draw_permutations(state: &CombatState) -> Vec<CombatState> {
    assert!(
        state.player.pinned_top.is_empty() && state.player.pinned_bottom.is_empty(),
        "reordering a pinned pile would move what the player knows"
    );
    let draw = &state.player.draw;
    assert!(draw.len() >= 3, "a draw pile of {} cards", draw.len());
    let mut out: Vec<CombatState> = Vec::new();
    for k in 0..3 {
        let mut s = state.clone();
        match k {
            0 => s.player.draw.reverse(),
            1 => s.player.draw.rotate_left(1),
            _ => s.player.draw.rotate_left(2),
        }
        assert_ne!(s.player.draw, *draw, "permutation {k} moved nothing");
        assert_ne!(
            s.player.draw[0], draw[0],
            "permutation {k} kept the top card"
        );
        assert!(out.iter().all(|o| o.player.draw != s.player.draw));
        out.push(s);
    }
    out
}

/// The harness: `prior` at `state` and at its draw permutations, over the
/// same edges. `Err` names the first permutation whose output bits differ.
fn draw_order_blind(prior: &dyn Policy, state: &CombatState) -> Result<(), String> {
    let actions = edge_actions(state);
    let base = bits(&prior.prior(state, &actions));
    for (k, s) in draw_permutations(state).iter().enumerate() {
        let got = bits(&prior.prior(s, &actions));
        if got != base {
            return Err(format!("permutation {k}: {got:?} vs {base:?}"));
        }
    }
    Ok(())
}

/// The two states the harness reads: the run's raw combat (no blindfold)
/// and a node state after the second turn's draw, from a search's tree.
fn honesty_states() -> Vec<(&'static str, CombatState)> {
    let raw = run_at_first_combat(4).combat.expect("in combat");
    let mut m = Mcts::new(raw.clone(), prior_config(SIMS, 11)).with_leaf_eval(LeafEval::TexelWide);
    m.try_search_n(SIMS).expect("searchable");
    let chance = m
        .tree
        .node(m.tree.root)
        .children
        .iter()
        .find(|(e, _)| *e == EdgeAction::Other(Action::EndTurn))
        .map(|(_, c)| *c)
        .expect("EndTurn expanded");
    let outcome = m.tree.node(chance).chance_outcomes[0].child;
    let drawn = m
        .tree
        .node(outcome)
        .realized
        .as_deref()
        .expect("state")
        .clone();
    assert!(drawn.round > raw.round, "not past the next draw");
    vec![("raw", raw), ("post-draw node", drawn)]
}

#[cfg(rollout_net_ironclad)]
#[test]
fn the_rl2_prior_reads_no_draw_order() {
    let prior = RolloutNetPrior::new(rl2_net(), 0.5, 1.0);
    for (name, state) in honesty_states() {
        draw_order_blind(&prior, &state).unwrap_or_else(|e| panic!("rl2 prior, {name}: {e}"));
    }
}

/// The deployed BC prior sees what the player sees: its output does not move
/// when only the hidden draw order does.
#[cfg(prior_net_ironclad)]
#[test]
fn the_bc_prior_reads_no_draw_order() {
    let prior = RolloutNetPrior::new(bc_net(), 0.5, 1.0);
    for (name, state) in honesty_states() {
        draw_order_blind(&prior, &state).unwrap_or_else(|e| panic!("bc prior, {name}: {e}"));
    }
}

/// Negative control: a prior that peeks at the top of the draw pile fails
/// the harness on both states.
#[test]
fn the_draw_order_harness_catches_a_prior_that_peeks() {
    struct Peeks;
    impl Policy for Peeks {
        fn prior(&self, state: &CombatState, actions: &[Action]) -> SmallVec<[f32; 16]> {
            let top = state.player.draw.first().map_or(0, |u| u.get());
            #[allow(clippy::cast_precision_loss)]
            let n = actions.len() as f32;
            let mut p: SmallVec<[f32; 16]> = actions.iter().map(|_| 1.0 / n).collect();
            #[allow(clippy::cast_precision_loss)]
            {
                p[0] += top as f32 * 1e-3;
            }
            p
        }
    }
    for (name, state) in honesty_states() {
        assert!(draw_order_blind(&Peeks, &state).is_err(), "{name}");
    }
}

// ── the prior counters ───────────────────────────────────────────────────

/// Without `use_puct_prior` the descent never reads the prior: no call, no
/// cache entry, every counter 0, whatever the prior is.
#[test]
fn a_prior_less_search_never_touches_the_cache_or_the_counters() {
    let start = run_at_first_combat(4).combat.expect("in combat");
    let mut plain = prior_config(SIMS, 1);
    plain.use_puct_prior = false;
    let calls = Arc::new(AtomicU64::new(0));
    let policy = Counted {
        inner: UniformPolicy,
        calls: Arc::clone(&calls),
    };
    let mut m = Mcts::with_policy(start, plain, policy).with_leaf_eval(LeafEval::TexelWide);
    m.try_search_n(SIMS).expect("searchable");
    assert_eq!(calls.load(Ordering::Relaxed), 0, "the prior was called");
    assert_eq!(cached(&m), 0, "the cache was filled");
    assert_eq!(snapshot(), PriorCounts::default());
}

/// A read is one lookup; a net prior's evaluation is one eval. Uncached,
/// every lookup evaluates (evals == lookups == the prior's calls); cached,
/// only the misses do (evals == calls < lookups). No fallback on a real net.
#[cfg(rollout_net_ironclad)]
#[test]
fn the_counters_count_lookups_and_evals() {
    let start = run_at_first_combat(4).combat.expect("in combat");
    assert_eq!(snapshot(), PriorCounts::default());
    let run = |on: bool| {
        let calls = Arc::new(AtomicU64::new(0));
        let policy = Counted {
            inner: rl2_prior(),
            calls: Arc::clone(&calls),
        };
        let before = snapshot();
        let mut m = Mcts::with_policy(start.clone(), prior_config(SIMS, 1), policy)
            .with_prior_cache(on)
            .with_leaf_eval(LeafEval::TexelWide);
        m.try_search_n(SIMS).expect("searchable");
        let after = snapshot();
        (
            calls.load(Ordering::Relaxed),
            after.lookups - before.lookups,
            after.evals - before.evals,
        )
    };
    let (calls_off, lookups_off, evals_off) = run(false);
    assert!(lookups_off > 0, "no prior read");
    assert_eq!(
        (evals_off, calls_off),
        (lookups_off, lookups_off),
        "uncached: one evaluation per read"
    );
    let (calls_on, lookups_on, evals_on) = run(true);
    assert_eq!(lookups_on, lookups_off, "the cache changed the descent");
    assert_eq!(evals_on, calls_on, "an evaluation is a call");
    assert!(
        evals_on > 0 && evals_on < lookups_on,
        "cached: {evals_on} evals / {lookups_on} lookups"
    );
    assert_eq!(snapshot().uniform_fallbacks, 0);
}

/// A net whose every key bias overflows f32 scores every action `+inf`, so
/// its weights are NaN: the prior falls back to uniform and counts it.
#[test]
fn a_nan_rl2_prior_falls_back_to_uniform_and_counts_it() {
    use crate::rollout_net::{ACTION_DIM, STATE_DIM};
    let (h, d) = (2usize, 2usize);
    let zeros = |r: usize, c: usize| vec![vec![0.0f32; c]; r];
    let net = RolloutNet::from_json(
        &serde_json::json!({
            "state_dim": STATE_DIM, "action_dim": ACTION_DIM, "hidden": h,
            "emb_dim": d, "temperature": 1.0,
            "keys": ["<unk>", "end_turn"],
            "w1": zeros(h, STATE_DIM), "b1": vec![0.0; h], "w2": zeros(d, h),
            "b2": vec![0.0; d], "emb": zeros(2, d), "bias": [1e300, 1e300],
            "proj": zeros(d, ACTION_DIM), "lin": vec![0.0; ACTION_DIM],
        })
        .to_string(),
    )
    .expect("parses");
    let state = run_at_first_combat(4).combat.expect("in combat");
    let actions = edge_actions(&state);
    let before = snapshot();
    let p = RolloutNetPrior::new(net, 0.5, 1.0).prior(&state, &actions);
    #[allow(clippy::cast_precision_loss)]
    let uniform = 1.0 / actions.len() as f32;
    assert_eq!(bits(&p), bits(&vec![uniform; actions.len()]), "{p:?}");
    let after = snapshot();
    assert_eq!(after.uniform_fallbacks, before.uniform_fallbacks + 1);
    assert_eq!(after.evals, before.evals + 1);
}
