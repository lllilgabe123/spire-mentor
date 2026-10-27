//! GH #839 — the search must see gains that outlast the fight.
//!
//! Genetic Algorithm grows its own block by 3 on every play, and since GH
//! #838 that growth carries into every later fight. The combat value only
//! scores what the fight leaves behind (HP, potions, round), so to the
//! search a GA play is 1 energy for 1 block and a lethal Strike always
//! wins the comparison. This test pins that `persistent_weight` flips it in
//! a position where playing GA first costs almost nothing, and that the
//! default `0.0` leaves the search exactly as it was.

mod common;

use common::build_handful_ex;
use sts2_core::cards::{CardId, TargetType};
use sts2_core::{Action, Intent};
use sts2_mcts::{persistent_gain_q, Mcts, MctsConfig};

const SIMS: u32 = 600;
const SEEDS: &[u64] = &[0, 1, 2];

const PLAY_GA: Action = Action::PlayCard {
    hand_index: 0,
    target: None,
};
const PLAY_STRIKE: Action = Action::PlayCard {
    hand_index: 1,
    target: Some(0),
};

/// Act 1, a weak enemy and one energy: either Strike for the immediate
/// kill, or play GA now (growing it for the rest of the run) and kill with
/// the reshuffled Strike next turn for a couple of HP.
fn scenario(seed: u64) -> sts2_core::CombatState {
    build_handful_ex(
        seed,
        &[
            (CardId::GeneticAlgorithm, 0, 1, 1, TargetType::Self_),
            (CardId::StrikeIronclad, 6, 0, 1, TargetType::Enemy),
        ],
        6,  // enemy HP — exactly one Strike
        70, // player HP — never in danger
        Intent::Attack { damage: 3 },
        &[],
        1, // energy — one play this turn
    )
}

fn search(seed: u64, persistent_weight: f32) -> (f32, f32, Action) {
    let mut mcts = Mcts::new(
        scenario(seed),
        MctsConfig {
            max_simulations: SIMS,
            seed,
            persistent_weight,
            ..MctsConfig::default()
        },
    );
    let picked = mcts.search();
    let ga = mcts
        .root_q_for_action(&PLAY_GA)
        .expect("GA is a root child");
    let strike = mcts
        .root_q_for_action(&PLAY_STRIKE)
        .expect("Strike is a root child");
    (ga, strike, picked)
}

#[test]
fn search_grows_genetic_algorithm_before_taking_the_kill() {
    for &seed in SEEDS {
        let (ga, strike, picked) = search(seed, 1.0);
        assert!(
            ga > strike,
            "seed {seed}: growing GA must back up above the immediate kill \
             (ga={ga}, strike={strike})"
        );
        assert_eq!(picked, PLAY_GA, "seed {seed}");
    }
}

/// The rollback seam: with the bonus off, the immediate kill wins, as it
/// did before GH #839.
#[test]
fn persistent_blind_search_takes_the_kill() {
    for &seed in SEEDS {
        let (ga, strike, picked) = search(seed, 0.0);
        assert!(
            strike > ga,
            "seed {seed}: without the bonus the kill must back up above GA \
             (ga={ga}, strike={strike})"
        );
        assert_eq!(picked, PLAY_STRIKE, "seed {seed}");
    }
}

/// `Mcts::with_policy` measures gains from the root, whatever the caller
/// left in the baseline field, so the root itself is worth exactly zero.
#[test]
fn baseline_is_taken_from_the_root() {
    let mcts = Mcts::new(
        scenario(0),
        MctsConfig {
            persistent_weight: 1.0,
            ..MctsConfig::default()
        },
    );
    assert_eq!(
        mcts.config.persistent_baseline,
        sts2_core::PersistentBaseline::of(&scenario(0))
    );
    assert!(persistent_gain_q(&scenario(0), &mcts.config).abs() < 1e-6);
}
