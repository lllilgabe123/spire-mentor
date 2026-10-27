//! GH #830, deep bets build 2: `STS2_RUNAI_MCTS_OVERRIDES=tree_reuse=1` on
//! the deployed pilot ([`MctsCombatPolicy`]), which keeps its search tree
//! between its own consecutive decisions in a fight.
//!
//! The overrides are read when a policy is constructed, so each test sets
//! the env var before building one (nextest runs every test in its own
//! process). Default-off identity is pinned by `leaf_playouts_identity.rs`
//! (the policy fight under the Ironclad off switches is byte-identical to
//! its pre-knob golden, and the shipped default to its own); here,
//! `tree_reuse_is_off_by_default` checks that the knob's machinery never
//! runs at default.

use sts2_core::full_run::RunPolicy;
use sts2_core::run_manager::{Phase, RunAction, RunManager};
use sts2_core::CombatState;
use sts2_runai::policy::{CombatBudget, HeuristicPolicy, MctsCombatPolicy, TreeReuseCounts};
use sts2_runai::rollout::ensure_content_installed;

const SIMS: u32 = 200;
const OVERRIDES: &str = "STS2_RUNAI_MCTS_OVERRIDES";

fn fresh_run(seed: i32) -> RunManager {
    ensure_content_installed();
    let mut rm = sts2_content::run_callbacks::new_run_manager_for_character_with_ascension(
        seed,
        sts2_core::potions::CharacterId::Ironclad,
        0,
    )
    .with_neow_start();
    sts2_content::run_callbacks::initialize_run_setup(&mut rm);
    rm
}

/// A fresh run whose deck carries two more deterministic attacks (Iron
/// Wave, Twin Strike), so the opening leaves more than one hand in the draw
/// pile and its hidden order decides what the next turn draws.
fn fresh_run_long_deck(seed: i32) -> RunManager {
    let mut rm = fresh_run(seed);
    for id in [
        sts2_core::cards::CardId::IronWave,
        sts2_core::cards::CardId::TwinStrike,
    ] {
        rm.run_state
            .master_deck
            .push(sts2_content::cards::spec_index::card_pod_from_id(id, false).expect("card spec"));
    }
    rm
}

/// Drive `rm` with first-option picks until it enters its first combat.
fn to_first_combat(rm: &mut RunManager) {
    for _ in 0..64 {
        if rm.phase() == Phase::Combat {
            return;
        }
        let actions = rm.get_available_actions();
        assert!(!actions.is_empty(), "run stalled before its first combat");
        let _ = rm.take_action(actions[0]);
    }
    panic!("no combat within 64 run actions");
}

/// The deployed pilot at `SIMS` simulations in every tier, with the
/// override bundle `overrides` (`None`: unset).
fn pilot(overrides: Option<&str>, run_seed: i32) -> MctsCombatPolicy<HeuristicPolicy> {
    match overrides {
        Some(v) => std::env::set_var(OVERRIDES, v),
        None => std::env::remove_var(OVERRIDES),
    }
    let mut budget = CombatBudget::eval_cheap();
    budget.sims = SIMS;
    budget.boss_sims = SIMS;
    budget.elite_sims = Some(SIMS);
    MctsCombatPolicy::new(HeuristicPolicy::new(), budget, run_seed)
}

/// One decision of `policy` on `rm`, applied. Returns the action and the
/// retained visits the search started with.
fn step(
    policy: &mut MctsCombatPolicy<HeuristicPolicy>,
    rm: &mut RunManager,
) -> (RunAction, Option<u32>) {
    let actions = rm.get_available_actions();
    assert!(!actions.is_empty(), "run stalled");
    let idx = policy.choose(rm, &actions).expect("policy picks");
    let retained = policy.last_retained_visits();
    let _ = rm.take_action(actions[idx]);
    (actions[idx], retained)
}

/// Play the first combat of `seed` to its end; the retained visits of
/// every combat decision.
fn play_first_fight(policy: &mut MctsCombatPolicy<HeuristicPolicy>, seed: i32) -> Vec<Option<u32>> {
    let mut rm = fresh_run(seed);
    to_first_combat(&mut rm);
    let mut out = Vec::new();
    for _ in 0..2_000 {
        if rm.phase() != Phase::Combat {
            return out;
        }
        out.push(step(policy, &mut rm).1);
    }
    panic!("fight did not finish");
}

fn assert_counts_add_up(c: TreeReuseCounts) {
    assert_eq!(
        c.searched,
        c.reused
            + c.fresh_no_tree
            + c.fresh_not_expanded
            + c.fresh_mismatch
            + c.fresh_outcome_unseen
            + c.fresh_edge_set
            + c.fresh_other,
        "{c:?}"
    );
    assert!(c.reused_via_chance <= c.reused, "{c:?}");
}

#[test]
fn tree_reuse_is_off_by_default() {
    let mut p = pilot(None, 5);
    let retained = play_first_fight(&mut p, 5);
    assert!(retained.iter().all(Option::is_none));
    assert_eq!(p.tree_reuse_counts(), TreeReuseCounts::default());
    // And an explicit `tree_reuse=0` is the same.
    let mut p = pilot(Some("tree_reuse=0"), 5);
    let _ = play_first_fight(&mut p, 5);
    assert_eq!(p.tree_reuse_counts(), TreeReuseCounts::default());
}

/// Reuse happens on a real fight, every searched decision is accounted
/// for, and the budget is unchanged: `sims_spent` counts only the new
/// simulations, exactly as without reuse.
#[test]
fn tree_reuse_reuses_within_a_fight_and_accounts_for_every_decision() {
    let mut p = pilot(Some("tree_reuse=1"), 5);
    let retained = play_first_fight(&mut p, 5);
    let c = p.tree_reuse_counts();
    eprintln!("{} combat decisions, counts {c:?}", retained.len());
    assert_counts_add_up(c);
    assert!(c.reused > 0, "{c:?}");
    assert!(c.retained_visits > 0, "{c:?}");
    let (_, _, _, sims_spent) = p.combat_budget_spent();
    assert_eq!(sims_spent, c.new_sims);
    assert_eq!(c.new_sims, c.searched * u64::from(SIMS));
    let per_decision: u64 = retained.iter().flatten().map(|&v| u64::from(v)).sum();
    assert_eq!(per_decision, c.retained_visits);
    assert_eq!(
        retained.iter().filter(|r| r.is_some()).count() as u64,
        c.reused
    );
    // A fresh tree's first decision of the fight has nothing to reuse.
    assert_eq!(retained[0], None);
}

#[test]
fn tree_reuse_is_deterministic() {
    let run = || {
        let mut p = pilot(Some("tree_reuse=1"), 7);
        let mut rm = fresh_run(7);
        to_first_combat(&mut rm);
        let mut log = Vec::new();
        while rm.phase() == Phase::Combat && log.len() < 2_000 {
            log.push(step(&mut p, &mut rm));
        }
        (log, p.tree_reuse_counts(), rm.run_state.current_hp)
    };
    let (a, b) = (run(), run());
    assert!(a.1.reused > 0, "{:?}", a.1);
    assert_eq!(a, b);
}

/// What the player sees of a fight: the observation hash the search dedups
/// on, the hand in order (legal actions index into it) and the belt.
fn observation(c: &CombatState) -> (u64, Vec<String>, String) {
    let hand = c
        .player
        .hand
        .iter()
        .map(|u| {
            let card = sts2_core::card_by_uid(&c.cards, *u).expect("hand card");
            format!("{:?}/{}/{}", card.card_id, card.upgraded, card.cost)
        })
        .collect();
    (
        sts2_core::chance_outcome_hash(c),
        hand,
        format!("{:?}", c.player.potions),
    )
}

/// Run two pilots in lockstep on two games, comparing every decision while
/// the two games look the same to the player. Returns (decisions compared,
/// how many of them reused a tree, rounds the compared decisions spanned,
/// the reuse counts).
fn lockstep(
    mut rm1: RunManager,
    mut rm2: RunManager,
    run_seed: i32,
) -> (usize, usize, u16, TreeReuseCounts) {
    let mut p1 = pilot(Some("tree_reuse=1"), run_seed);
    let mut p2 = pilot(Some("tree_reuse=1"), run_seed);
    let (mut compared, mut reused) = (0, 0);
    let mut rounds = (u16::MAX, 0_u16);
    for _ in 0..2_000 {
        if rm1.phase() != Phase::Combat || rm2.phase() != Phase::Combat {
            break;
        }
        let (c1, c2) = (rm1.combat.as_ref(), rm2.combat.as_ref());
        let (c1, c2) = (c1.expect("combat"), c2.expect("combat"));
        if observation(c1) != observation(c2) {
            break;
        }
        rounds = (rounds.0.min(c1.round), rounds.1.max(c1.round));
        let (a1, r1) = step(&mut p1, &mut rm1);
        let (a2, r2) = step(&mut p2, &mut rm2);
        assert_eq!(
            a1, a2,
            "decision {compared}: the hidden information moved the pick"
        );
        assert_eq!(r1, r2, "decision {compared}: retained visits differ");
        let bits = |p: &MctsCombatPolicy<HeuristicPolicy>| {
            p.last_combat_search().map(|r| {
                r.outcomes
                    .iter()
                    .map(|o| (format!("{:?}", o.action), o.visits, o.value_sum.to_bits()))
                    .collect::<Vec<_>>()
            })
        };
        assert_eq!(
            bits(&p1),
            bits(&p2),
            "decision {compared}: root statistics differ"
        );
        compared += 1;
        reused += usize::from(r1.is_some());
    }
    assert_eq!(p1.tree_reuse_counts(), p2.tree_reuse_counts());
    let counts = p1.tree_reuse_counts();
    eprintln!("lockstep counts {counts:?}");
    (
        compared,
        reused,
        rounds.1.saturating_sub(rounds.0) + 1,
        counts,
    )
}

/// The honesty pin. Two games identical to the player — one with the
/// unknown draw pile reversed and every combat RNG stream (monster AI
/// included) reseeded — must get identical picks and identical root
/// statistics from a reusing pilot for as long as the games look the same
/// (here: the first turn; the reversed pile shows at the next draw). The
/// pile holds more than the next hand, so its order is information a
/// leaking search would act on: rooting the reused tree at the true state
/// instead of its belief root fails here.
#[test]
fn tree_reuse_reads_no_hidden_information() {
    let seed = 5;
    let mut rm1 = fresh_run_long_deck(seed);
    to_first_combat(&mut rm1);
    let mut rm2 = rm1.clone();
    let c2 = rm2.combat.as_mut().expect("combat");
    assert!(c2.player.pinned_top.is_empty() && c2.player.pinned_bottom.is_empty());
    assert!(
        c2.player.draw.len() > 5,
        "{} cards left to draw",
        c2.player.draw.len()
    );
    c2.player.draw.reverse();
    c2.run_rng.reseed_combat_streams(0xC0FFEE, 3);
    c2.rng = sts2_core::dotnet_rng::Rng::from_search_seed(0xBAD5EED);
    assert_ne!(
        sts2_core::state_hash(rm1.combat.as_ref().expect("combat")),
        sts2_core::state_hash(rm2.combat.as_ref().expect("combat"))
    );
    let (compared, reused, rounds, _) = lockstep(rm1, rm2, seed);
    eprintln!("compared {compared} decisions over {rounds} rounds, {reused} reused");
    assert!(compared >= 3, "only {compared} decisions compared");
    assert!(reused >= 1, "no reuse in {compared} compared decisions");
}

/// The same across turns: hidden streams that do not move what the player
/// sees at the next draw (the reshuffle stream, card / potion generation)
/// reseeded in one game. The comparison runs through `EndTurn`, so a reuse
/// through a chance edge is held to the same standard.
#[test]
fn tree_reuse_reads_no_hidden_information_across_turns() {
    let seed = 5;
    let mut rm1 = fresh_run(seed);
    to_first_combat(&mut rm1);
    let mut rm2 = rm1.clone();
    let c2 = rm2.combat.as_mut().expect("combat");
    c2.run_rng.shuffle = sts2_core::dotnet_rng::Rng::new(0x5EED);
    c2.run_rng.combat_card_generation = sts2_core::dotnet_rng::Rng::new(0x5EEE);
    c2.run_rng.combat_potion_generation = sts2_core::dotnet_rng::Rng::new(0x5EEF);
    assert_ne!(
        sts2_core::state_hash(rm1.combat.as_ref().expect("combat")),
        sts2_core::state_hash(rm2.combat.as_ref().expect("combat"))
    );
    let (compared, reused, rounds, counts) = lockstep(rm1, rm2, seed);
    eprintln!("compared {compared} decisions over {rounds} rounds, {reused} reused");
    assert!(reused >= 1, "no reuse in {compared} compared decisions");
    assert!(rounds >= 2, "the comparison never crossed an EndTurn");
    // Seed 5 reuses across that EndTurn: the chance-edge path is compared.
    assert!(counts.reused_via_chance >= 1, "{counts:?}");
}
