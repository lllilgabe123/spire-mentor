//! `dominance_pairs` — a decision-quality instrument whose answers are
//! **proven by the engine**, not voted on by a deep search (GitHub #492
//! Phase 2).
//!
//! ## Prerequisite: the fixture pool
//!
//! Every entry stores a *path* into `sts2-rs/data/run-fixtures/`, which is
//! git-ignored, so it is absent on a fresh clone and after an agent
//! worktree that produced it is cleaned up. Regenerate it with
//! **`just harvest-fixtures`** (~44 s for all five characters, deterministic
//! — the frozen `state_hash` still matches across hundreds of commits) or
//! `logs/2026-08-12-fixture-pool/harvest.sh` directly.
//! ## Why the position book cannot measure a backup-rule change
//!
//! `combat_position_book`'s `gold_action` is whatever a 50 000-sim search
//! agrees on across 3 seeds and 2 methods. Both oracles use the *same
//! backup rule* as the arm under test, so the book cannot see a defect
//! that lives in the backup rule — and it drops every position where the
//! oracle is split, which is exactly the near-tie set where an
//! aggregation change fires. Measured on #492's repro position, the 50k
//! oracle unanimously picks the move the engine can prove is dominated.
//!
//! ## What this measures instead
//!
//! Take a live decision point. Take two distinct card plays `a` and `b`
//! that are both playable and still playable after each other — i.e. two
//! **orderings of the same pair of cards**, so the two sequences cost the
//! same energy and consume the same hand. Apply both orderings and
//! compare the resulting states.
//!
//! Keep the pair only when the two states are **structurally identical
//! except on a dominance axis** — full `CombatState` equality (which
//! includes both RNG streams, every pile, every power, every intent)
//! after zeroing the axis fields. Then one ordering is weakly better and
//! the other is weakly worse, and no search was consulted to decide that.
//!
//! Axes, all monotone and unambiguous:
//!
//! | axis | direction |
//! |---|---|
//! | `enemy_hp` | lower is better (summed over all enemies) |
//! | `player_hp` | higher is better |
//! | `player_block` | higher is better |
//!
//! Deliberately **not** an axis: enemy debuff stacks. "More Vulnerable is
//! better" is a judgement, not an engine theorem (it is worth nothing if
//! you never attack again), and neutralizing power amounts for a
//! whitelisted debuff set would smuggle that judgement into a suite whose
//! whole value is that it contains none. Potions are excluded for the
//! mechanical reason that `potions` / `potion_slot_indices` /
//! `CombatState::potions` would all have to be neutralized in lockstep.
//!
//! ## What the metric claims, exactly
//!
//! For a kept pair, playing the dominating card first and following
//! through reaches a state that weakly dominates the one reached by the
//! other order. Both cards are affordable together (that is enforced at
//! generation), so an agent that picks either first will play the other
//! second in the ordinary case. The scored quantity is therefore the
//! search's **pairwise preference between the two first moves**, read off
//! one search's root statistics — a paired comparison inside a single
//! tree, not two searches differenced.
//!
//! This is a *local* claim: it does not assert that the dominating card
//! is the globally best root action, only that it is never worse than the
//! other member of its pair. That is enough to be a defect when the
//! search says otherwise, and it is the exact shape of bug
//! `wb-e34ef8b6039bc0f3`.
//!
//! ## Usage
//!
//! ```sh
//! # 1. Freeze the suite (no search runs; minutes).
//! cargo run --release -p sts2-bench --example dominance_pairs -- \
//!     build [pool_root] [out_json] [per_char_fixtures] [warmup_variants]
//!
//! # 2. Score arms against it.
//! cargo run --release -p sts2-bench --example dominance_pairs -- \
//!     score [suite_json] [pool_root] --sims 1000 --seeds 4 \
//!           --arms mean,pm:1,pm:2,pm:4 [--limit N] [--dump cells.csv]
//! ```
//!
//! `--limit` STRIDES across the suite rather than truncating it; the suite
//! is written in pool order, so a prefix is one character. `--dump` writes
//! one CSV row per scored cell, including each pair member's **rank among
//! all root children** — the column that separates an error the agent
//! would actually play from one it orders wrong among moves it will never
//! choose.
//!
//! An arm is `mean` (the deployed control), `pm:<p>[:<shift>]` (globally
//! shifted power mean) or `pmn:<p>` (per-node min-max normalized power
//! mean), any of them suffixed `+dp` to turn on
//! `MctsConfig::dominance_pruning`. Every arm's row carries a `dq_vs_ctl`
//! column — the mean absolute change it made to the scored root Q values
//! relative to the control. An
//! arm reporting `pair_acc` equal to the control AND `dq_vs_ctl` near zero
//! did not fail, it never fired, and those are different findings.

#[path = "common/mod.rs"]
mod common;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sts2_cli::scenario_to_combat_state;
use sts2_cli::sim::scenario::toml_file::load_scenario_file;
use sts2_core::{legal_actions, CombatPhase, CombatState};
use sts2_mcts::{BackupRule, HiddenInfoMode, Mcts, MctsConfig};

use common::dominance::pairs_at;

/// Deployed advisor dialect — same constants `combat_position_book` uses,
/// so a number from this suite is comparable with a `gold_agr` number.
const SEARCH_DEPTH: u32 = 200;
const C_PUCT: f32 = 2.5;
const DRAW_WIDENING_C: f32 = 2.0;

// ── Suite types ─────────────────────────────────────────────────────────

/// Reader-only mirror of the `RealisticCorpus` shape `corpus_gen
/// --runmanager-toml-export` writes; same duplication `mcts_ab.rs` and
/// `combat_position_book.rs` already carry, and for the same reason.
#[derive(Debug, Clone, Deserialize)]
struct HarvestCorpus {
    entries: Vec<HarvestEntry>,
}

#[derive(Debug, Clone, Deserialize)]
struct HarvestEntry {
    tier: String,
    fixture_path: String,
    floor: u16,
    seed: u64,
}

/// One engine-proven ordering pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(clippy::struct_field_names)] // `pair_id` is the suite-wide join key
struct Pair {
    pair_id: u32,
    character: String,
    tier: String,
    floor: u16,
    fixture_rel_path: String,
    seed: u64,
    /// Warm-up variant index. The decision point is reached by playing
    /// `warmup_plies(state, seed ^ variant_salt(v), max_warmup_plies)`, so
    /// this index plus the suite-level `max_warmup_plies` is the whole
    /// reproduction key. Variant `0` reproduces `combat_position_book`'s
    /// convention exactly.
    warmup_variant: u32,
    /// `state_hash` of the scored state. A consumer that realizes the
    /// fixture and gets a different hash is not scoring this position.
    state_hash: u64,
    legal_actions: usize,
    /// Debug rendering of the root action that starts the dominating
    /// ordering, and of the one that starts the dominated ordering. Same
    /// convention as `BookEntry::gold_action` — `Action` has no
    /// `Serialize` impl and every sibling harness already joins on the
    /// debug string.
    better_first: String,
    worse_first: String,
    /// Human-readable `CardId` of each ordering's first card. Purely
    /// descriptive — `better_first` / `worse_first` are the join keys.
    /// Without this a reader auditing the committed suite sees only
    /// `PlayCard { hand_index: 2 }` and cannot tell what the pair is.
    better_card: String,
    worse_card: String,
    /// Which axes the dominating ordering wins on, and by how much
    /// (always `>= 0`; at least one is `> 0`).
    axes: Vec<String>,
    gain_enemy_hp: i32,
    gain_player_hp: i32,
    gain_player_block: i32,
}

#[derive(Debug, Clone, Serialize)]
struct Suite {
    version: u32,
    git_sha: String,
    generator: String,
    max_warmup_plies: u32,
    warmup_variants: u32,
    /// Positions examined, by `(character, tier)` — the denominator for
    /// "how rare is a proven pair".
    examined_by_bucket: BTreeMap<String, usize>,
    ordered_pairs_tried: usize,
    pairs: Vec<Pair>,
}

#[derive(Debug, Clone, Deserialize)]
struct SuiteIn {
    max_warmup_plies: u32,
    pairs: Vec<Pair>,
}

// ── Position realization (shared with the position book) ────────────────

/// Salt for warm-up variant `v`. Variant 0 is the identity, so it
/// reproduces `combat_position_book`'s decision point for the same
/// fixture, seed and ply cap.
fn variant_salt(v: u32) -> u64 {
    u64::from(v).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

/// Why a suite entry produced no scorable state. Splitting `Missing` out
/// of the rest is the whole point: an absent fixture pool and a moved
/// engine have opposite fixes, and until GH #446 this binary reported
/// neither — with the pool absent it printed `0 positions realized`, then a
/// complete zero-filled results table with `cell_p = 1.0000`, and **exited
/// 0**. That is indistinguishable from a genuine null. The sibling
/// `combat_position_book annotate` was hardened against exactly this on
/// 2026-08-24; this is that fix, propagated to the co-metric.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Skip {
    /// The fixture did not load: absent pool, or an unparseable TOML.
    Missing,
    /// It loaded, but warm-up left nothing to score (terminal, wrong
    /// phase, or fewer than two legal actions). A property of the
    /// position, not of the pool.
    NoDecision,
    /// It loaded and realized, but to a different state than the suite's
    /// frozen `state_hash`. This is the only one that means "stale".
    /// Produced by the caller, not by `realize`.
    HashMiss,
}

fn realize(
    path: &Path,
    seed: u64,
    variant: u32,
    max_warmup_plies: u32,
) -> Result<CombatState, Skip> {
    let scenario = load_scenario_file(path).map_err(|_| Skip::Missing)?;
    let mut state = scenario_to_combat_state(&scenario, seed).map_err(|_| Skip::Missing)?;
    let _ = sts2_core::combat_loop::start_combat(&mut state);
    let _ = sts2_core::combat_loop::start_player_turn(&mut state);
    // Vary the warm-up RNG, not the ply CAP. Varying the cap only rescales
    // one draw from a seed-derived stream, so most variants land on the
    // same decision point: measured, 48 cap-variants over the full pool
    // yielded 350 raw pairs of which 226 were duplicates.
    common::warmup_plies(&mut state, seed ^ variant_salt(variant), max_warmup_plies);
    if state.is_over || state.phase != CombatPhase::PlayerPlayPhase {
        return Err(Skip::NoDecision);
    }
    if legal_actions(&state).len() < 2 {
        return Err(Skip::NoDecision);
    }
    Ok(state)
}

// ── Arms ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
struct Arm {
    rule: BackupRule,
    p: f32,
    shift: f32,
    /// `max_rollout_depth`. Deployed is [`SEARCH_DEPTH`]; `--depth` lowers
    /// it so a meaningful fraction of leaves hit the cap and
    /// `evaluate_in_progress` is consulted at all. Diagnostic only — the
    /// `vbzji` experiment found deep-and-few beats shallow-and-many on win
    /// rate, so this is here to test a *mechanism*, not to ship.
    depth: u32,
    /// `MctsConfig::dominance_pruning` — the `+dp` arm suffix.
    dominance_pruning: bool,
}

/// An arm spec, optionally suffixed `+dp` to turn on
/// `MctsConfig::dominance_pruning` (`mean+dp`, `pmn:4+dp`).
fn parse_arm(spec: &str) -> Option<(String, Arm)> {
    if let Some(base) = spec.strip_suffix("+dp") {
        let (name, arm) = parse_arm(base)?;
        return Some((
            format!("{name}+dp"),
            Arm {
                dominance_pruning: true,
                ..arm
            },
        ));
    }
    let mut it = spec.split(':');
    match it.next()? {
        "mean" => Some((
            "mean".to_string(),
            Arm {
                rule: BackupRule::Mean,
                p: 1.0,
                shift: 1.5,
                depth: SEARCH_DEPTH,
                dominance_pruning: false,
            },
        )),
        "pm" => {
            let p: f32 = it.next()?.parse().ok()?;
            let shift: f32 = it.next().and_then(|s| s.parse().ok()).unwrap_or(1.5);
            Some((
                format!("pm:{p}:{shift}"),
                Arm {
                    rule: BackupRule::PowerMean,
                    p,
                    shift,
                    depth: SEARCH_DEPTH,
                    dominance_pruning: false,
                },
            ))
        }
        "pmn" => {
            let p: f32 = it.next()?.parse().ok()?;
            Some((
                format!("pmn:{p}"),
                Arm {
                    rule: BackupRule::PowerMeanNorm,
                    p,
                    shift: 1.5,
                    depth: SEARCH_DEPTH,
                    dominance_pruning: false,
                },
            ))
        }
        _ => None,
    }
}

fn arm_config(arm: Arm, sims: u32, seed: u64) -> MctsConfig {
    MctsConfig {
        max_simulations: sims,
        seed,
        max_rollout_depth: arm.depth,
        hidden_info: HiddenInfoMode::BeliefDraw,
        c_puct: C_PUCT,
        draw_widening_c: DRAW_WIDENING_C,
        backup_rule: arm.rule,
        backup_power_p: arm.p,
        backup_power_shift: arm.shift,
        dominance_pruning: arm.dominance_pruning,
        ..MctsConfig::default()
    }
}

fn arm_seed(k: u32) -> u64 {
    u64::from(k)
        .wrapping_mul(0xD1B5_4A32_D192_ED03)
        .wrapping_add(0x0C1E_0BAD_5EED_0001)
}

/// One scored cell: how the arm ranked the dominating first move against
/// the dominated one, in a single search.
#[derive(Debug, Clone, Copy)]
struct Cell {
    pair_id: u32,
    /// `1` correct, `0` wrong, `-1` the two Q values were exactly equal.
    q_verdict: i8,
    visit_verdict: i8,
    /// `Q(better) - Q(worse)`. Signed, so the mean is an effect size and
    /// not just a rate.
    margin: f32,
    /// The two raw root Q values, kept so an arm's row can report how far
    /// it moved them relative to the control.
    q_better: f32,
    q_worse: f32,
    v_better: u32,
    v_worse: u32,
    /// 0-based rank of each member of the pair among ALL root children,
    /// ordered by Q descending.
    ///
    /// This is the difference between an error that ships and one that
    /// does not. A pair ordered wrong at ranks 4 and 5 costs nothing — the
    /// agent plays neither. A pair ordered wrong at ranks 0 and 1 is the
    /// agent committing to a move the engine proves is dominated. A
    /// headline accuracy that does not separate the two over-states the
    /// defect.
    rank_better: u16,
    rank_worse: u16,
    n_children: u16,
    /// Fraction of the pair's rollouts that ended at a **depth cap**
    /// rather than a real terminal, i.e. the fraction whose value came
    /// from `evaluate_in_progress` at all.
    ///
    /// Load-bearing for any value-side arm: #492 observed that on its
    /// repro position every rollout reached a win terminal, where
    /// `evaluate_victory` reads only `player_hp` / `potions` / `round`.
    /// If that generalises, re-weighting `evaluate_in_progress` is a
    /// change to a function the search never calls on the failing class.
    tie_frac: f32,
    /// Fraction of the pair's rollouts that ended in a WIN terminal.
    win_frac: f32,
    /// Per-member win fractions. The summed `win_frac` says the fight is
    /// won; only the split says whether the search thinks the DOMINATED
    /// order wins more often — which on a proven pair it cannot, and which
    /// is the #512 §2.3 below-chance signature.
    win_frac_better: f32,
    win_frac_worse: f32,
    /// Index of each member among the root's children, i.e. its position
    /// in `legal_actions` order. If accuracy tracks this index rather than
    /// the dominance, the defect is an ordering artefact and not a value
    /// one — the cheapest possible explanation for below-chance accuracy.
    idx_better: u16,
    idx_worse: u16,
}

fn score_position(
    state: &CombatState,
    pairs: &[&Pair],
    arm: Arm,
    sims: u32,
    seed: u64,
) -> Vec<Cell> {
    let mut mcts = Mcts::new(state.clone(), arm_config(arm, sims, seed));
    let _ = mcts.search();
    let outs = mcts.root_outcomes();
    // Rank every root child by Q descending, so a pair's verdict can be
    // read alongside whether the search was ever going to play either.
    let mut order: Vec<usize> = (0..outs.len()).collect();
    order.sort_by(|a, b| {
        outs[*b]
            .mean_value
            .partial_cmp(&outs[*a].mean_value)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut rank = vec![0_u16; outs.len()];
    for (r, &i) in order.iter().enumerate() {
        rank[i] = u16::try_from(r).unwrap_or(u16::MAX);
    }
    let find = |label: &str| {
        outs.iter()
            .position(|o| format!("{:?}", o.action) == label)
            .map(|i| {
                let o = &outs[i];
                (o.mean_value, o.visits, rank[i], o.ties, o.wins, i)
            })
    };
    pairs
        .iter()
        .filter_map(|p| {
            let (bq, bv, br, bties, bwins, bi) = find(&p.better_first)?;
            let (wq, wv, wr, wties, wwins, wi) = find(&p.worse_first)?;
            let cmp = |a: f32, b: f32| -> i8 {
                if (a - b).abs() <= f32::EPSILON {
                    -1
                } else {
                    i8::from(a > b)
                }
            };
            Some(Cell {
                pair_id: p.pair_id,
                q_verdict: cmp(bq, wq),
                #[allow(clippy::cast_precision_loss)]
                visit_verdict: cmp(bv as f32, wv as f32),
                margin: bq - wq,
                q_better: bq,
                q_worse: wq,
                v_better: bv,
                v_worse: wv,
                rank_better: br,
                rank_worse: wr,
                n_children: u16::try_from(outs.len()).unwrap_or(u16::MAX),
                #[allow(clippy::cast_precision_loss)]
                tie_frac: (bties + wties) as f32 / (bv + wv).max(1) as f32,
                #[allow(clippy::cast_precision_loss)]
                win_frac: (bwins + wwins) as f32 / (bv + wv).max(1) as f32,
                #[allow(clippy::cast_precision_loss)]
                win_frac_better: bwins as f32 / bv.max(1) as f32,
                #[allow(clippy::cast_precision_loss)]
                win_frac_worse: wwins as f32 / wv.max(1) as f32,
                idx_better: u16::try_from(bi).unwrap_or(u16::MAX),
                idx_worse: u16::try_from(wi).unwrap_or(u16::MAX),
            })
        })
        .collect()
}

/// Accuracy with exact ties counted as half a point, plus the raw counts
/// so a reader can see whether a rate is carried by ties.
fn accuracy(verdicts: &[i8]) -> (f64, usize, usize, usize) {
    let right = verdicts.iter().filter(|v| **v == 1).count();
    let wrong = verdicts.iter().filter(|v| **v == 0).count();
    let tied = verdicts.iter().filter(|v| **v == -1).count();
    #[allow(clippy::cast_precision_loss)]
    let n = verdicts.len().max(1) as f64;
    #[allow(clippy::cast_precision_loss)]
    let acc = (right as f64 + 0.5 * tied as f64) / n;
    (acc, right, wrong, tied)
}

/// Standard normal CDF via `erf`, for the clustered test below.
fn norm_cdf(z: f64) -> f64 {
    // Abramowitz & Stegun 7.1.26 on |z|/sqrt(2); max abs error 1.5e-7,
    // three orders of magnitude finer than any p-value this reports.
    let x = z.abs() / std::f64::consts::SQRT_2;
    let t = 1.0 / t_denom(x);
    let y = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t
            * (-x * x).exp();
    let erf = if z < 0.0 { -y } else { y };
    f64::midpoint(1.0, erf)
}

fn t_denom(x: f64) -> f64 {
    0.327_591_1_f64.mul_add(x, 1.0)
}

/// **Cluster-robust** paired test. Every cell inside one position comes
/// from the SAME search tree (and cells across seeds from the same
/// position share the position), so the cells are not independent and a
/// cell-level McNemar over-states significance — the same
/// pseudo-replication that manufactured a `p = 0.00000` out of one
/// position in the 2026-08-22 edge-prior study.
///
/// Collapse to one number per position: `d_i` = (arm correct) − (control
/// correct) summed over that position's cells, then a two-sided paired
/// t-test on the `d_i`. Returns `(p, n_clusters, mean_d)`.
fn cluster_paired_p(d: &[f64]) -> (f64, usize, f64) {
    let n = d.len();
    if n < 2 {
        return (1.0, n, 0.0);
    }
    #[allow(clippy::cast_precision_loss)]
    let nf = n as f64;
    let mean = d.iter().sum::<f64>() / nf;
    let var = d.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (nf - 1.0);
    if var <= 0.0 {
        return (if mean.abs() > 0.0 { 0.0 } else { 1.0 }, n, mean);
    }
    let t = mean / (var / nf).sqrt();
    // Normal approximation to the t distribution; with n in the hundreds
    // the difference is in the fourth decimal.
    (2.0 * (1.0 - norm_cdf(t.abs())), n, mean)
}

/// Two-sided exact-binomial (sign) test on the discordant cells of a
/// paired comparison — McNemar without the chi-square approximation,
/// which matters because the discordant counts here are small.
///
/// **Reported for continuity with sibling harnesses, but it is NOT the
/// test to read**: it treats every cell as independent. Read
/// `cluster_p`.
fn mcnemar_p(b: u32, c: u32) -> f64 {
    let n = b + c;
    if n == 0 {
        return 1.0;
    }
    let k = u64::from(b.min(c));
    // Sum the lower tail exactly; n is at most a few thousand.
    let mut log_c = 0.0_f64;
    let mut tail = 0.0_f64;
    #[allow(clippy::cast_precision_loss)]
    for i in 0..=k {
        if i > 0 {
            log_c += ((u64::from(n) - i + 1) as f64).ln() - (i as f64).ln();
        }
        tail += (log_c - f64::from(n) * std::f64::consts::LN_2).exp();
    }
    (2.0 * tail).min(1.0)
}

// ── Modes ───────────────────────────────────────────────────────────────

fn cmd_build(args: &[String]) -> std::io::Result<()> {
    let mut it = args.iter();
    let pool_root = PathBuf::from(
        it.next()
            .cloned()
            .unwrap_or_else(|| "data/run-fixtures".to_string()),
    );
    let out_path = PathBuf::from(
        it.next()
            .cloned()
            .unwrap_or_else(|| "data/dominance_pairs.json".to_string()),
    );
    let per_char: usize = it.next().and_then(|s| s.parse().ok()).unwrap_or(450);
    // Each variant re-realizes the same fixture with a different warmup
    // depth, so one fixture contributes several decision points instead of
    // one. Proven pairs are rare per position; this is the cheap way to N.
    let warmup_variants: u32 = it.next().and_then(|s| s.parse().ok()).unwrap_or(8);
    let max_warmup_plies: u32 = it.next().and_then(|s| s.parse().ok()).unwrap_or(8);

    std::panic::set_hook(Box::new(|_| {}));

    let mut pool_dirs: Vec<PathBuf> = std::fs::read_dir(&pool_root)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.is_dir()
                && p.file_name()
                    .and_then(|s| s.to_str())
                    .is_some_and(|s| s.starts_with("the-"))
        })
        .collect();
    pool_dirs.sort();

    let mut tasks: Vec<(String, HarvestEntry, u32)> = Vec::new();
    let mut examined_by_bucket: BTreeMap<String, usize> = BTreeMap::new();
    for dir in &pool_dirs {
        let character = dir
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .trim_start_matches("the-")
            .to_string();
        let Ok(bytes) = std::fs::read(dir.join("corpus.json")) else {
            eprintln!("[dom] {character}: no corpus.json, skipping");
            continue;
        };
        let Ok(corpus) = serde_json::from_slice::<HarvestCorpus>(&bytes) else {
            eprintln!("[dom] {character}: corpus.json unparseable, skipping");
            continue;
        };
        for entry in corpus.entries.into_iter().take(per_char) {
            *examined_by_bucket
                .entry(format!("{character}/{}", entry.tier))
                .or_default() += warmup_variants as usize;
            for w in 0..warmup_variants {
                tasks.push((character.clone(), entry.clone(), w));
            }
        }
    }
    eprintln!(
        "[dom] {} (fixture, warmup) positions from {} characters",
        tasks.len(),
        pool_dirs.len()
    );

    let found: Vec<(String, HarvestEntry, u32, u64, usize, usize, Vec<_>)> = tasks
        .into_par_iter()
        .filter_map(|(character, entry, w)| {
            let path = PathBuf::from(&entry.fixture_path);
            let state =
                std::panic::catch_unwind(|| realize(&path, entry.seed, w, max_warmup_plies))
                    .ok()?
                    .ok()?;
            let (tried, found) = std::panic::catch_unwind(|| pairs_at(&state)).ok()?;
            if found.is_empty() {
                return Some((
                    character,
                    entry,
                    w,
                    0,
                    legal_actions(&state).len(),
                    tried,
                    Vec::new(),
                ));
            }
            Some((
                character,
                entry,
                w,
                sts2_core::state_hash(&state),
                legal_actions(&state).len(),
                tried,
                found,
            ))
        })
        .collect();

    let mut pairs = Vec::new();
    let mut ordered_pairs_tried = 0_usize;
    // Warmup variants are max-ply CAPS, not exact depths, so different
    // variants of one fixture regularly land on the SAME decision point.
    // Keeping both copies would double-count that pair in every paired
    // test downstream — silently inflating n while adding no information.
    let mut seen: std::collections::HashSet<(u64, String, String)> =
        std::collections::HashSet::new();
    let mut duplicates = 0_usize;
    // Restore the default hook the moment the `catch_unwind` loop above is
    // done. It was suppressed so a panicking fixture does not spam the log,
    // but leaving it installed silences every LATER panic too — including
    // the guards below, which would then exit 101 printing nothing at all
    // (GH #446). `mcts_book_sweep` has always restored it; this did not.
    let _ = std::panic::take_hook();

    for (character, entry, w, hash, n_legal, tried, found) in found {
        ordered_pairs_tried += tried;
        for (better, worse, better_label, worse_label, axes, e, h, b) in found {
            if !seen.insert((hash, format!("{better:?}"), format!("{worse:?}"))) {
                duplicates += 1;
                continue;
            }
            let p = PathBuf::from(&entry.fixture_path);
            let rel = p.strip_prefix(&pool_root).map_or_else(
                |_| {
                    let file = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
                    format!("the-{character}/{file}")
                },
                |q| q.display().to_string(),
            );
            #[allow(clippy::cast_possible_truncation)]
            pairs.push(Pair {
                pair_id: pairs.len() as u32,
                character: character.clone(),
                tier: entry.tier.clone(),
                floor: entry.floor,
                fixture_rel_path: rel,
                seed: entry.seed,
                warmup_variant: w,
                state_hash: hash,
                legal_actions: n_legal,
                better_first: format!("{better:?}"),
                worse_first: format!("{worse:?}"),
                better_card: better_label.clone(),
                worse_card: worse_label.clone(),
                axes,
                gain_enemy_hp: e,
                gain_player_hp: h,
                gain_player_block: b,
            });
        }
    }

    // Never overwrite a good suite with an empty one. With the pool absent
    // every `realize` fails, `pairs` is empty, and this would happily write
    // a valid-looking zero-pair suite over the committed 741 — which then
    // scores 0.0000 on every arm forever. Same false-null family as the
    // `score` guard above (GH #446).
    assert!(
        !pairs.is_empty(),
        "no dominance pairs found — refusing to write an empty suite over {}. \
         If nothing realized, the fixture pool is git-ignored and regenerable: \
         run `just harvest-fixtures` (~44 s) or \
         logs/2026-08-12-fixture-pool/harvest.sh. Pool root: {}",
        out_path.display(),
        pool_root.display()
    );

    let suite = Suite {
        version: 1,
        git_sha: common::git_short_sha(),
        generator: "dominance_pairs".to_string(),
        max_warmup_plies,
        warmup_variants,
        examined_by_bucket,
        ordered_pairs_tried,
        pairs,
    };
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&out_path, serde_json::to_vec_pretty(&suite)?)?;

    let mut by_axis: BTreeMap<String, usize> = BTreeMap::new();
    let mut by_char: BTreeMap<String, usize> = BTreeMap::new();
    for p in &suite.pairs {
        *by_axis.entry(p.axes.join("+")).or_default() += 1;
        *by_char.entry(p.character.clone()).or_default() += 1;
    }
    println!("\n=== dominance-pair suite ===");
    println!("ordered pairs tried : {}", suite.ordered_pairs_tried);
    println!("duplicates dropped  : {duplicates}");
    println!("proven pairs        : {}", suite.pairs.len());
    println!("written             : {}", out_path.display());
    println!("\nby axis:");
    for (k, v) in &by_axis {
        println!("  {k:<28} {v:>5}");
    }
    println!("\nby character:");
    for (k, v) in &by_char {
        println!("  {k:<28} {v:>5}");
    }
    Ok(())
}

fn cmd_score(args: &[String]) -> std::io::Result<()> {
    let mut positional: Vec<String> = Vec::new();
    let mut sims_list: Vec<u32> = vec![1000];
    let mut seeds: u32 = 4;
    let mut arm_specs: Vec<String> = vec!["mean".into(), "pm:2".into()];
    let mut limit: usize = usize::MAX;
    // Per-cell CSV for offline diagnosis. Off by default: a full ladder is
    // ~50k rows and nothing in the summary needs it.
    let mut dump: Option<String> = None;
    // Rollout depth, applied to EVERY arm. Diagnostic for #512 §7.1 probe
    // 3: at the deployed 200 no leaf ever hits the cap, so
    // `evaluate_in_progress` gets no vote at all on this class.
    let mut depth: u32 = SEARCH_DEPTH;
    let mut it = args.iter().peekable();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--sims" => {
                sims_list = it
                    .next()
                    .map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect())
                    .unwrap_or_default();
            }
            "--seeds" => seeds = it.next().and_then(|s| s.parse().ok()).unwrap_or(4),
            "--arms" => {
                arm_specs = it
                    .next()
                    .map(|s| s.split(',').map(str::to_string).collect())
                    .unwrap_or_default();
            }
            "--limit" => limit = it.next().and_then(|s| s.parse().ok()).unwrap_or(usize::MAX),
            "--depth" => {
                depth = it
                    .next()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(SEARCH_DEPTH);
            }
            "--dump" => dump = it.next().cloned(),
            other => positional.push(other.to_string()),
        }
    }
    let suite_path = PathBuf::from(
        positional
            .first()
            .cloned()
            .unwrap_or_else(|| "data/dominance_pairs.json".to_string()),
    );
    let pool_root = PathBuf::from(
        positional
            .get(1)
            .cloned()
            .unwrap_or_else(|| "data/run-fixtures".to_string()),
    );

    let suite: SuiteIn = serde_json::from_slice(&std::fs::read(&suite_path)?)
        .map_err(|e| std::io::Error::other(format!("{}: {e}", suite_path.display())))?;
    let max_warmup_plies = suite.max_warmup_plies;
    let arms: Vec<(String, Arm)> = arm_specs
        .iter()
        .filter_map(|s| parse_arm(s))
        .map(|(name, mut arm)| {
            arm.depth = depth;
            let name = if depth == SEARCH_DEPTH {
                name
            } else {
                format!("{name}@d{depth}")
            };
            (name, arm)
        })
        .collect();
    assert!(!arms.is_empty(), "no parseable arms in --arms");

    std::panic::set_hook(Box::new(|_| {}));

    // Group pairs by position so one search scores every pair it hosts.
    //
    // `--limit` STRIDES rather than truncating. The suite is written in
    // pool order (character, then fixture), so a prefix is a correlated
    // sample: the first 200 pairs are 100% Defect and score `pair_acc`
    // 0.81 against 0.73 for the whole suite. A subset run has to span the
    // suite or its numbers cannot be compared with a full-suite run.
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    let sampled: Vec<&Pair> = if limit >= suite.pairs.len() {
        suite.pairs.iter().collect()
    } else {
        let step = suite.pairs.len() as f64 / limit as f64;
        (0..limit)
            .map(|i| &suite.pairs[((i as f64 * step) as usize).min(suite.pairs.len() - 1)])
            .collect()
    };
    let mut by_pos: BTreeMap<(String, u64, u32), Vec<&Pair>> = BTreeMap::new();
    for p in sampled {
        by_pos
            .entry((p.fixture_rel_path.clone(), p.seed, p.warmup_variant))
            .or_default()
            .push(p);
    }
    eprintln!(
        "[dom] {} pairs across {} positions x {} arms x {} seeds x {:?} sims",
        by_pos.values().map(Vec::len).sum::<usize>(),
        by_pos.len(),
        arms.len(),
        seeds,
        sims_list
    );

    // Realize once per position; a mismatched `state_hash` is a hard stop,
    // not a warning — it means the suite and the engine have diverged and
    // every number below would be about a different position.
    let wanted = by_pos.len();
    let outcomes: Vec<Result<(CombatState, Vec<&Pair>), Skip>> = by_pos
        .into_iter()
        .map(|((rel, seed, w), pairs)| {
            let path = pool_root.join(&rel);
            let state = std::panic::catch_unwind(|| realize(&path, seed, w, max_warmup_plies))
                .unwrap_or(Err(Skip::NoDecision))?;
            let want = pairs[0].state_hash;
            if sts2_core::state_hash(&state) != want {
                // Not `Skip`, because a hash miss is the one failure that
                // genuinely means the suite is stale — counted separately
                // below so the two never get conflated in the summary.
                return Err(Skip::HashMiss);
            }
            Ok((state, pairs))
        })
        .collect();
    // Restore the default hook the moment the `catch_unwind` loop above is
    // done. It was suppressed so a panicking fixture does not spam the log,
    // but leaving it installed silences every LATER panic too — including
    // the guards below, which would then exit 101 printing nothing at all
    // (GH #446). `mcts_book_sweep` has always restored it; this did not.
    let _ = std::panic::take_hook();
    let count = |want: Skip| {
        outcomes
            .iter()
            .filter(|r| r.as_ref().err() == Some(&want))
            .count()
    };
    let (missing, no_decision, hash_miss) = (
        count(Skip::Missing),
        count(Skip::NoDecision),
        count(Skip::HashMiss),
    );
    let positions: Vec<(CombatState, Vec<&Pair>)> = outcomes.into_iter().flatten().collect();
    eprintln!(
        "[dom] {} of {wanted} positions realized and hash-verified \
         (skipped: {missing} fixture missing, {no_decision} no decision, \
         {hash_miss} hash mismatch)",
        positions.len()
    );
    // Refuse to print a results table scored on (almost) nothing. Without
    // this the binary emitted all-zero rows with `cell_p = 1.0000` and exit
    // 0 when the pool was absent — a false null that reads exactly like a
    // real one. `mcts_book_sweep` has had the equivalent guard all along;
    // this binary, the oracle-free co-metric, did not (GH #446).
    assert!(
        positions.len() * 2 >= wanted,
        "only {}/{wanted} suite positions verified — refusing to report a number \
         scored on a partial corpus.\n  {missing} fixtures could not be loaded from \
         {}. The pool is git-ignored and regenerable: run `just harvest-fixtures` \
         (~44 s, deterministic) or logs/2026-08-12-fixture-pool/harvest.sh.\n  \
         {hash_miss} realized to a different state than the suite froze — THAT is \
         a stale suite, and it needs a `dominance_pairs build` rebuild.",
        positions.len(),
        pool_root.display()
    );

    // pair_id -> cluster index (one cluster per position / search tree).
    let mut cluster_of: BTreeMap<u32, usize> = BTreeMap::new();
    for (i, (_, pairs)) in positions.iter().enumerate() {
        for p in pairs {
            cluster_of.insert(p.pair_id, i);
        }
    }
    let n_clusters = positions.len();

    let mut dump_rows: Vec<String> = Vec::new();
    if dump.is_some() {
        dump_rows.push(
            "arm,sims,seed,pair_id,character,tier,axes,gain_enemy_hp,gain_player_block,\
             legal_actions,n_children,q_verdict,visit_verdict,q_better,q_worse,v_better,v_worse,\
             rank_better,rank_worse,tie_frac,win_frac,win_frac_better,win_frac_worse,\
             idx_better,idx_worse,better_card,worse_card"
                .to_string(),
        );
    }
    for sims in &sims_list {
        println!("\n=== {sims} sims, {seeds} seeds ===");
        println!(
            "{:<16} {:>8} {:>7} {:>6} {:>6} {:>6} {:>10} {:>10} {:>12}",
            "arm", "pair_acc", "visit", "right", "wrong", "tie", "margin", "dq_vs_ctl", "cell_p"
        );
        let mut control: Option<Vec<i8>> = None;
        let mut control_q: Vec<(f32, f32)> = Vec::new();
        let mut control_keys: Vec<(u32, u32)> = Vec::new();
        for (name, arm) in &arms {
            let cells: Vec<(u32, Cell)> = positions
                .par_iter()
                .flat_map(|(state, pairs)| {
                    (0..seeds)
                        .into_par_iter()
                        .flat_map(move |k| {
                            score_position(state, pairs, *arm, *sims, arm_seed(k))
                                .into_iter()
                                .map(move |c| (k, c))
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>()
                })
                .collect();
            // Stable cell order so the paired test lines cells up by
            // (pair, seed) and not by whatever order rayon finished in.
            let mut cells = cells;
            cells.sort_by_key(|(k, c)| (c.pair_id, *k));
            let keys: Vec<(u32, u32)> = cells.iter().map(|(k, c)| (c.pair_id, *k)).collect();
            if dump.is_some() {
                for (k, c) in &cells {
                    let Some(p) = suite.pairs.iter().find(|p| p.pair_id == c.pair_id) else {
                        continue;
                    };
                    dump_rows.push(format!(
                        "{name},{sims},{k},{},{},{},{},{},{},{},{},{},{},{:.6},{:.6},{},{},{},{},{:.4},{:.4},{:.4},{:.4},{},{},{},{}",
                        c.pair_id,
                        p.character,
                        p.tier,
                        p.axes.join("+"),
                        p.gain_enemy_hp,
                        p.gain_player_block,
                        p.legal_actions,
                        c.n_children,
                        c.q_verdict,
                        c.visit_verdict,
                        c.q_better,
                        c.q_worse,
                        c.v_better,
                        c.v_worse,
                        c.rank_better,
                        c.rank_worse,
                        c.tie_frac,
                        c.win_frac,
                        c.win_frac_better,
                        c.win_frac_worse,
                        c.idx_better,
                        c.idx_worse,
                        p.better_card,
                        p.worse_card,
                    ));
                }
            }
            let q: Vec<i8> = cells.iter().map(|(_, c)| c.q_verdict).collect();
            let visit_verdicts: Vec<i8> = cells.iter().map(|(_, c)| c.visit_verdict).collect();
            let (qa, right, wrong, tie) = accuracy(&q);
            let (va, _, _, _) = accuracy(&visit_verdicts);
            #[allow(clippy::cast_precision_loss)]
            let margin = cells.iter().map(|(_, c)| f64::from(c.margin)).sum::<f64>()
                / cells.len().max(1) as f64;

            // "Did it fire?" — mean absolute displacement of the scored
            // root Q values vs the control, on the same cells.
            let qs: Vec<(f32, f32)> = cells.iter().map(|(_, c)| (c.q_better, c.q_worse)).collect();
            #[allow(clippy::cast_precision_loss)]
            let dq = if control_q.is_empty() || keys != control_keys {
                0.0
            } else {
                control_q
                    .iter()
                    .zip(qs.iter())
                    .map(|(a, b)| f64::from((a.0 - b.0).abs() + (a.1 - b.1).abs()) / 2.0)
                    .sum::<f64>()
                    / qs.len().max(1) as f64
            };

            let is_control = control.is_none();
            let p_str = match &control {
                None => {
                    control = Some(q.clone());
                    control_q.clone_from(&qs);
                    control_keys.clone_from(&keys);
                    "control".to_string()
                }
                Some(base) => {
                    if keys == control_keys {
                        let (mut b, mut c) = (0_u32, 0_u32);
                        for (x, y) in base.iter().zip(q.iter()) {
                            if *x == 1 && *y == 0 {
                                b += 1;
                            } else if *x == 0 && *y == 1 {
                                c += 1;
                            }
                        }
                        format!("{:.4} ({b}/{c})", mcnemar_p(b, c))
                    } else {
                        "UNPAIRED".to_string()
                    }
                }
            };
            // Cluster-robust paired test — the one to read.
            let cluster_str = match &control {
                Some(base) if !is_control && keys == control_keys && !control_keys.is_empty() => {
                    let mut d = vec![0.0_f64; n_clusters];
                    for ((k, x), y) in keys.iter().zip(base.iter()).zip(q.iter()) {
                        let ci = cluster_of.get(&k.0).copied().unwrap_or(0);
                        let score = |v: i8| {
                            if v == 1 {
                                1.0
                            } else {
                                f64::from(v == -1) * 0.5
                            }
                        };
                        d[ci] += score(*y) - score(*x);
                    }
                    let (cp, nc, md) = cluster_paired_p(&d);
                    format!("{cp:.4} n={nc} d={md:+.3}")
                }
                _ => "control".to_string(),
            };
            println!(
                "{name:<16} {qa:>8.4} {va:>7.4} {right:>6} {wrong:>6} {tie:>6} {margin:>10.5} \
                 {dq:>10.5} {p_str:>12}   cluster_p={cluster_str}"
            );

            // Per-axis breakdown: a regression should say which kind of
            // dominance it lost.
            let mut by_axis: BTreeMap<String, Vec<i8>> = BTreeMap::new();
            for (_, c) in &cells {
                if let Some(p) = suite.pairs.iter().find(|p| p.pair_id == c.pair_id) {
                    by_axis
                        .entry(p.axes.join("+"))
                        .or_default()
                        .push(c.q_verdict);
                }
            }
            for (axis, vs) in &by_axis {
                let (acc, right, wrong, tie) = accuracy(vs);
                println!(
                    "    {axis:<24} {acc:>8.4}  n={} ({right}/{wrong}/{tie})",
                    vs.len()
                );
            }
        }
    }
    if let Some(path) = dump {
        std::fs::write(&path, dump_rows.join("\n") + "\n")?;
        eprintln!("[dom] wrote {} cell rows to {path}", dump_rows.len() - 1);
    }
    Ok(())
}

/// #512 §7.1 probe 1 — `rollout_coverage_fraction` on the suite's own
/// positions, per character.
///
/// The expert rollout policy falls back to Uniform whenever coverage sits
/// below `COVERAGE_THRESHOLD`. Regent once sat at ~2%, which silently
/// dropped every Regent rollout to random play; Regent is also the worst
/// character on this suite (0.560 pair accuracy, and *falling* with
/// budget). If that regression is back, nothing else in the arm list
/// matters for Regent and this probe is the whole answer.
fn cmd_coverage(args: &[String]) -> std::io::Result<()> {
    // `--cards` answers the follow-up question the summary table always
    // provokes: *which* cards are missing from the priority table (GH #515).
    let cards = args.iter().any(|a| a == "--cards");
    let positional: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    let suite_path = PathBuf::from(
        positional
            .first()
            .map_or_else(|| "data/dominance_pairs.json".to_string(), |s| (*s).clone()),
    );
    let pool_root = PathBuf::from(
        positional
            .get(1)
            .map_or_else(|| "data/run-fixtures".to_string(), |s| (*s).clone()),
    );
    let suite: SuiteIn = serde_json::from_slice(&std::fs::read(&suite_path)?)
        .map_err(|e| std::io::Error::other(format!("{}: {e}", suite_path.display())))?;

    std::panic::set_hook(Box::new(|_| {}));

    // One entry per position, not per pair — a position hosting three
    // pairs would otherwise be counted three times.
    let mut seen: BTreeMap<(String, u64, u32), String> = BTreeMap::new();
    for p in &suite.pairs {
        seen.insert(
            (p.fixture_rel_path.clone(), p.seed, p.warmup_variant),
            p.character.clone(),
        );
    }
    let keys: Vec<_> = seen.into_iter().collect();
    let rows: Vec<CoverageRow> = keys
        .par_iter()
        .filter_map(|((rel, seed, w), character)| {
            let path = pool_root.join(rel);
            let state =
                std::panic::catch_unwind(|| realize(&path, *seed, *w, suite.max_warmup_plies))
                    .ok()?
                    .ok()?;
            let actions = legal_actions(&state);
            Some(CoverageRow {
                character: character.clone(),
                fraction: sts2_mcts::expert::rollout_coverage_fraction(&state, &actions),
                uncovered: uncovered_cards(&state, &actions),
            })
        })
        .collect();

    // Restore the default hook the moment the `catch_unwind` loop above is
    // done. It was suppressed so a panicking fixture does not spam the log,
    // but leaving it installed silences every LATER panic too — including
    // the guards below, which would then exit 101 printing nothing at all
    // (GH #446). `mcts_book_sweep` has always restored it; this did not.
    let _ = std::panic::take_hook();

    // Same false-null shape as `score`: with the pool absent this printed a
    // header and no rows, which reads as "measured, nothing to report".
    assert!(
        rows.len() * 2 >= keys.len(),
        "only {}/{} suite positions realized — the fixture pool is git-ignored \
         and regenerable: run `just harvest-fixtures` (~44 s) or \
         logs/2026-08-12-fixture-pool/harvest.sh. Pool root: {}",
        rows.len(),
        keys.len(),
        pool_root.display()
    );

    let mut by_char: BTreeMap<String, Vec<f32>> = BTreeMap::new();
    for r in &rows {
        by_char
            .entry(r.character.clone())
            .or_default()
            .push(r.fraction);
    }
    println!(
        "rollout_coverage_fraction on the {} dominance positions",
        rows.len()
    );
    println!(
        "{:<14} {:>6} {:>8} {:>8} {:>8} {:>10}",
        "character", "n", "mean", "min", "p10", "frac<0.8"
    );
    for (c, fs) in &by_char {
        let mut v = fs.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        #[allow(clippy::cast_precision_loss)]
        let n = v.len() as f32;
        let mean = v.iter().sum::<f32>() / n.max(1.0);
        let p10 = v[(v.len() / 10).min(v.len().saturating_sub(1))];
        #[allow(clippy::cast_precision_loss)]
        let low = v.iter().filter(|f| **f < 0.8).count() as f32 / n.max(1.0);
        println!(
            "{c:<14} {:>6} {mean:>8.4} {:>8.4} {p10:>8.4} {low:>10.4}",
            v.len(),
            v[0]
        );
    }

    if cards {
        report_uncovered(&rows);
    }
    Ok(())
}

/// One realized suite position: who it belongs to, its coverage fraction,
/// and the card ids that fell through to [`NEUTRAL_PRIORITY`].
struct CoverageRow {
    character: String,
    fraction: f32,
    uncovered: Vec<String>,
}

/// Debug names of the cards behind the legal `PlayCard` actions whose
/// `play_ordering` is the [`NEUTRAL_PRIORITY`] fallback — i.e. exactly the
/// cards `rollout_coverage_fraction` counts against the gate. Deduplicated
/// per position so a card sitting in the hand twice is one vote.
fn uncovered_cards(state: &CombatState, actions: &[sts2_core::Action]) -> Vec<String> {
    let mut out: Vec<String> = actions
        .iter()
        .filter_map(|a| match a {
            sts2_core::Action::PlayCard { hand_index, .. } => state
                .player
                .hand
                .get(*hand_index as usize)
                .and_then(|uid| state.cards.get(uid.0.get() as usize - 1)),
            _ => None,
        })
        .filter(|card| {
            sts2_mcts::expert::play_ordering(card.card_id, state.player.character)
                == sts2_mcts::expert::NEUTRAL_PRIORITY
        })
        .map(|card| format!("{:?}", card.card_id))
        .collect();
    out.sort();
    out.dedup();
    out
}

/// Which cards are costing the gate, per character: how many positions each
/// appears in at all, and how many of the *failing* positions (coverage
/// below the threshold) it appears in. The second column is the one that
/// ranks the work — a card in one failing position out of 205 is not what
/// dropped the character below the gate.
fn report_uncovered(rows: &[CoverageRow]) {
    let mut tally: BTreeMap<String, BTreeMap<String, (u32, u32)>> = BTreeMap::new();
    let mut failing: BTreeMap<String, u32> = BTreeMap::new();
    for r in rows {
        let low = r.fraction < sts2_mcts::expert::COVERAGE_THRESHOLD;
        if low {
            *failing.entry(r.character.clone()).or_default() += 1;
        }
        for c in &r.uncovered {
            let e = tally
                .entry(r.character.clone())
                .or_default()
                .entry(c.clone())
                .or_default();
            e.0 += 1;
            if low {
                e.1 += 1;
            }
        }
    }
    for (character, cards) in &tally {
        let n_fail = failing.get(character).copied().unwrap_or(0);
        println!(
            "\nuncovered cards — {character} ({} distinct; {n_fail} positions below the gate)",
            cards.len()
        );
        println!("{:<34} {:>10} {:>12}", "card", "positions", "of which <0.8");
        let mut v: Vec<_> = cards.iter().collect();
        v.sort_by_key(|(name, (all, low))| {
            (
                std::cmp::Reverse(*low),
                std::cmp::Reverse(*all),
                (*name).clone(),
            )
        });
        for (name, (all, low)) in v {
            println!("{name:<34} {all:>10} {low:>12}");
        }
    }
}

fn main() -> std::io::Result<()> {
    sts2_content::cards::dispatch::ensure_installed();
    sts2_content::relics::dispatch::ensure_installed();
    sts2_content::potions::dispatch::ensure_installed();

    // 64 MiB rayon workers. Realizing a harvested fixture runs real card /
    // relic dispatch, and a handful of Act-3 fixtures recurse deep enough
    // through the late-effect bus to blow the default 2 MiB worker stack —
    // which aborts the process rather than raising a panic `catch_unwind`
    // could absorb, so one bad fixture would otherwise take the whole sweep
    // down with no output. Measured: the 54 000-position build aborts at
    // default stack size and completes here.
    rayon::ThreadPoolBuilder::new()
        .stack_size(64 * 1024 * 1024)
        .build_global()
        .expect("rayon global pool is configured once, before any parallel work");

    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("build") => cmd_build(&args[1..]),
        Some("score") => cmd_score(&args[1..]),
        Some("coverage") => cmd_coverage(&args[1..]),
        _ => {
            eprintln!(
                "usage:\n  dominance_pairs build [pool_root] [out_json] [per_char] \
                 [warmup_variants] [max_warmup_plies]\n  dominance_pairs score [suite_json] \
                 [pool_root] [--sims 1000,4000] [--seeds 4] [--arms mean,pm:2] [--limit N] \
                 [--depth 200]\n  dominance_pairs coverage [suite_json] [pool_root]"
            );
            Ok(())
        }
    }
}
