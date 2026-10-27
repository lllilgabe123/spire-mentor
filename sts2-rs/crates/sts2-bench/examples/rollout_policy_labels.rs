//! `rollout_policy_labels` — training rows for a fitted rollout priority
//! table.
//!
//! The Heuristic rollout's base priorities are the STS1 `sts_lightspeed`
//! ladder plus type/cost/block buckets for the STS2-only rosters
//! (`expert::play_ordering`). This dumps what a high-budget search
//! actually prefers at real fight positions, priced the way the rollout
//! prices them, so `scripts/fit_rollout_priorities.py` can fit one base
//! priority per `(character, card)` against it.
//!
//! Every fixture under `pool_root` is played to the end by the deployed
//! search (`mcts_config_for`, honest `BeliefDraw`, depth 200) at
//! `label_sims`. At each decision with at least two legal card plays, the
//! row records every legal action's rollout pricing
//! (`expert::rollout_pricing`) next to its root visit count and mean value.
//! The search's pick is then applied, so positions follow the distribution
//! the deployed agent itself reaches.
//!
//! The pool must not be one the position books are built from: the books
//! are the gate for whatever table these rows produce. `just
//! harvest-fixtures-rollout` writes a disjoint-seed pool for this.
//!
//! Output is JSONL, one row per decision, flushed per fight so a long run
//! shows progress and a killed run keeps what it finished.
//!
//! ```text
//! cargo run --release -p sts2-bench --example rollout_policy_labels -- \
//!     <pool_root> <out.jsonl> [fixtures_per_character=80] [label_sims=4000] \
//!     [seeds_per_fixture=1]
//! ```

#[path = "common/mod.rs"]
mod common;

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use rayon::prelude::*;
use serde::Serialize;
use sts2_core::{is_terminal, legal_actions, Action, CombatPhase, CombatState};
use sts2_mcts::expert::{rollout_pricing, END_TURN_PRIORITY};
use sts2_mcts::{HiddenInfoMode, Mcts, MctsConfig};

/// Safety cap on actions per fight; a real fight ends far sooner.
const MAX_STEPS: u32 = 600;

#[derive(Serialize)]
struct ActionRow {
    action: String,
    card: Option<&'static str>,
    base: i16,
    priority: i16,
    tracks_base: bool,
    visits: u32,
    mean_value: f32,
}

#[derive(Serialize)]
struct DecisionRow<'a> {
    fixture: &'a str,
    character: &'a str,
    seed: u64,
    step: u32,
    round: u16,
    label_sims: u32,
    /// The deployed rollout would take a Uniform step here (coverage gate).
    uniform_fallback: bool,
    /// The deployed rollout's obvious-kill shortcut would pick this index.
    obvious_kill: Option<usize>,
    chosen: usize,
    actions: Vec<ActionRow>,
}

fn label_config(sims: u32, seed: u64) -> MctsConfig {
    sts2_runai::policy::mcts_config_for(
        HiddenInfoMode::BeliefDraw,
        sims,
        seed,
        200,
        MctsConfig::default().terminal_eval_mode,
    )
}

fn step_seed(fight_seed: u64, step: u32) -> u64 {
    fight_seed ^ u64::from(step).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

/// Play one fixture to the end, returning its decision rows as JSON lines.
fn play_fixture(path: &Path, rel: &str, character: &str, seed: u64, sims: u32) -> Vec<String> {
    let Ok(mut state) = common::realize_fixture(path, seed) else {
        eprintln!("[labels] skip {rel}: does not realize");
        return Vec::new();
    };
    let mut lines = Vec::new();
    let mut step = 0u32;
    while !is_terminal(&state) && step < MAX_STEPS {
        let acts = legal_actions(&state);
        if acts.is_empty() {
            let _ = sts2_core::apply(&mut state, Action::EndTurn);
            step += 1;
            continue;
        }
        let mut m = Mcts::new(state.clone(), label_config(sims, step_seed(seed, step)));
        let chosen_action = m.search_n(sims);
        if state.phase == CombatPhase::PlayerPlayPhase {
            if let Some(line) = decision_row(
                &state,
                &acts,
                &m,
                &chosen_action,
                rel,
                character,
                seed,
                step,
                sims,
            ) {
                lines.push(line);
            }
        }
        let _ = sts2_core::apply(&mut state, chosen_action);
        step += 1;
    }
    lines
}

#[allow(clippy::too_many_arguments)]
fn decision_row(
    state: &CombatState,
    acts: &[Action],
    m: &Mcts,
    chosen_action: &Action,
    rel: &str,
    character: &str,
    seed: u64,
    step: u32,
    sims: u32,
) -> Option<String> {
    let pricing = rollout_pricing(state, acts, END_TURN_PRIORITY);
    // A position teaches the card table something only when the rollout
    // has a choice between at least two card plays.
    let card_plays = pricing.actions.iter().filter(|p| p.card.is_some()).count();
    if card_plays < 2 {
        return None;
    }
    let outcomes = m.root_outcomes();
    let rows: Vec<ActionRow> = acts
        .iter()
        .zip(&pricing.actions)
        .map(|(a, p)| {
            let o = outcomes.iter().find(|o| &o.action == a);
            ActionRow {
                action: format!("{a:?}"),
                card: p.card.map(sts2_core::cards::CardId::to_python_name),
                base: p.base,
                priority: p.priority,
                tracks_base: p.tracks_base,
                visits: o.map_or(0, |o| o.visits),
                mean_value: o.map_or(0.0, |o| o.mean_value),
            }
        })
        .collect();
    let chosen = acts.iter().position(|a| a == chosen_action)?;
    let row = DecisionRow {
        fixture: rel,
        character,
        seed,
        step,
        round: state.round,
        label_sims: sims,
        uniform_fallback: pricing.uniform_fallback,
        obvious_kill: pricing.obvious_kill,
        chosen,
        actions: rows,
    };
    serde_json::to_string(&row).ok()
}

/// Every `*.toml` under `dir`, recursively, sorted for a stable sample.
fn fixtures_under(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else {
            continue;
        };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|x| x == "toml") {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

/// `n` fixtures spread evenly over the sorted list (all of them if fewer).
fn spread<T: Clone>(items: &[T], n: usize) -> Vec<T> {
    if items.len() <= n {
        return items.to_vec();
    }
    (0..n).map(|i| items[i * items.len() / n].clone()).collect()
}

fn main() -> std::io::Result<()> {
    sts2_content::cards::dispatch::ensure_installed();
    sts2_content::relics::dispatch::ensure_installed();
    sts2_content::potions::dispatch::ensure_installed();

    let args: Vec<String> = std::env::args().skip(1).collect();
    assert!(
        args.len() >= 2,
        "usage: rollout_policy_labels <pool_root> <out.jsonl> [fixtures_per_character] \
         [label_sims] [seeds_per_fixture]"
    );
    let pool_root = PathBuf::from(&args[0]);
    let out_path = PathBuf::from(&args[1]);
    let per_char: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(80);
    let sims: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(4000);
    let seeds: u64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(1);

    // Sub-pools nest as `<pool_root>/<sub>/the-<char>/`, so match the
    // character directory anywhere below the root.
    let every = fixtures_under(&pool_root);
    let mut jobs: Vec<(PathBuf, String, String, u64)> = Vec::new();
    for character in ["ironclad", "silent", "defect", "necrobinder", "regent"] {
        let dir_name = format!("the-{character}");
        let all: Vec<PathBuf> = every
            .iter()
            .filter(|p| p.components().any(|c| c.as_os_str() == dir_name.as_str()))
            .cloned()
            .collect();
        let picked = spread(&all, per_char);
        eprintln!(
            "[labels] {character}: {} of {} fixtures under {}",
            picked.len(),
            all.len(),
            pool_root.display()
        );
        for p in picked {
            let rel = p
                .strip_prefix(&pool_root)
                .unwrap_or(&p)
                .display()
                .to_string();
            for s in 0..seeds {
                jobs.push((p.clone(), rel.clone(), character.to_string(), 1 + s));
            }
        }
    }
    eprintln!(
        "[labels] {} fights at {sims} sims -> {}",
        jobs.len(),
        out_path.display()
    );

    let out = Mutex::new(std::io::BufWriter::new(std::fs::File::create(&out_path)?));
    let done = std::sync::atomic::AtomicUsize::new(0);
    let total = jobs.len();
    jobs.par_iter().for_each(|(path, rel, character, seed)| {
        let lines = std::panic::catch_unwind(|| play_fixture(path, rel, character, *seed, sims))
            .unwrap_or_else(|_| {
                eprintln!("[labels] PANIC in {rel} seed {seed}; fight dropped");
                Vec::new()
            });
        {
            let mut w = out.lock().expect("writer lock");
            for l in &lines {
                let _ = writeln!(w, "{l}");
            }
            let _ = w.flush();
        }
        let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if n.is_multiple_of(10) || n == total {
            eprintln!(
                "[labels] {n}/{total} fights done ({rel}: {} rows)",
                lines.len()
            );
        }
    });
    Ok(())
}
