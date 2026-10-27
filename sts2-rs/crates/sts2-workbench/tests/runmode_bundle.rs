//! GH #830: the browser run mode's `RunAgent` plays Ironclad with the run
//! agent's Ironclad defaults and every other character as before.
//!
//! The `runmode_*` goldens cannot show this: they search at 8 / 16 sims,
//! where the EndTurn widening cap binds for neither constant and the draw
//! widening rarely does, so they may come back unchanged either way. Here the
//! constructor path the browser takes (`RunAgent::new`: `eval_cheap`, then
//! the sims) plays a run's first combat at 1000 / 4000 sims twice: with no
//! `STS2_RUNAI_*` variable, exactly as the wasm build runs (its env lookups
//! always fail), and with the two off switches. Ironclad's searches must
//! differ, Silent's must not.
//!
//! The record of each decision is its action and the search it came from
//! (every root action's visits and values). The searches, not the moves:
//! these first fights are easy, and on seeds 1-8 the knobs never changed a
//! move of one. Seed 6 is the longest of those whose searches move.
//!
//! The overrides are read when the agent is built, so the legs run in one
//! process. `STS2_RUNAI_PRIOR_NET` is read once per process; the first combat
//! is a hallway fight, where the built-in prior (elite and boss only) is off
//! either way, so that cannot hide a difference.

use serde_json::Value;
use sts2_runai::policy::{IRONCLAD_SEARCH_KNOBS, IRONCLAD_SEARCH_KNOBS_OFF};
use sts2_workbench::game::Game;
use sts2_workbench::runmode::RunAgent;

const SIMS: u32 = 1000;
const BOSS_SIMS: u32 = 4000;
const SEED: i32 = 6;

type Fight = Vec<(String, Option<Value>)>;

/// Every combat decision of the first combat of a faithful `character` run
/// (seed `seed`, A0) played by a fresh `RunAgent`: the action and the
/// search it came from.
fn first_combat(character: &str, seed: i32) -> Fight {
    let mut game = Game::new();
    game.start_faithful(character, seed, 0).expect("run starts");
    let mut agent = RunAgent::new(seed, SIMS, BOSS_SIMS);
    let mut out = Vec::new();
    for _ in 0..2_000 {
        let Some(rec) = agent.step(&mut game) else {
            break;
        };
        if rec.is_combat {
            out.push((format!("{:?}", rec.action), rec.combat_search));
        } else if !out.is_empty() {
            return out;
        }
        if rec.over {
            break;
        }
    }
    assert!(!out.is_empty(), "{character}: never fought");
    out
}

/// `first_combat` with the two off switches set, then with neither.
fn off_then_default(character: &str, seed: i32) -> (Fight, Fight) {
    std::env::set_var("STS2_RUNAI_MCTS_OVERRIDES", IRONCLAD_SEARCH_KNOBS_OFF);
    std::env::set_var("STS2_RUNAI_PRIOR_NET", "off");
    let off = first_combat(character, seed);
    std::env::remove_var("STS2_RUNAI_MCTS_OVERRIDES");
    let default = first_combat(character, seed);
    (off, default)
}

#[test]
fn ironclad_run_mode_plays_the_ironclad_defaults() {
    let (off, default) = off_then_default("Ironclad", SEED);
    assert_eq!(off.len(), default.len());
    assert_ne!(
        off, default,
        "the Ironclad default did not reach the run-mode agent"
    );
}

/// Silent plays the same fight, search for search, with or without the
/// Ironclad switches. Non-vacuous: the same knobs set explicitly do move
/// Silent's searches in this fight, so a leak of the Ironclad layer would.
#[test]
fn silent_run_mode_is_unchanged_by_the_ironclad_switches() {
    let (off, default) = off_then_default("Silent", SEED);
    assert_eq!(off, default);
    std::env::set_var("STS2_RUNAI_MCTS_OVERRIDES", IRONCLAD_SEARCH_KNOBS);
    let with_the_knobs = first_combat("Silent", SEED);
    assert_ne!(
        with_the_knobs, default,
        "the Ironclad knobs do not bind in this Silent fight: pick another seed"
    );
}
