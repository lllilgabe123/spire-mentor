//! `mcts_exploration_arms` — adaptive-exploration arms on the position book.
//!
//! Companion harness to `mcts_cpuct_book_sweep`, which is left untouched so
//! the published #442 table stays reproducible. This one answers a different
//! question.
//!
//! ## Why a second harness
//!
//! `mcts_cpuct_book_sweep` reports a **mean over all 168 book positions**. The
//! complaint that motivated this work — "`c_puct = 2.5` over-explores in some
//! positions and under-explores in others" — is a claim about *variance across
//! positions*, which a mean is structurally blind to. A constant can be the
//! best single number and still be wrong in most individual positions.
//!
//! So every arm here is scored **per volatility stratum** as well as overall.
//! Volatility is measured once per position by an independent reference search
//! (`--ref-sims`, deployed config, one seed) and cached to a sidecar keyed on
//! the book's `state_hash`, so it is paid once and is not a property of any
//! arm. Positions are then split into quartiles by reference root Q spread.
//!
//! **The falsifying test**: if the `c_puct` ladder's optimum is the same in
//! every quartile, there is no adaptivity prize and the whole family below is
//! dead. If it *crosses* — high-spread quartiles preferring low `c`, flat
//! quartiles preferring high `c` — a fixed constant is provably leaving
//! decisions on the table and an adaptive rule can collect them.
//!
//! ## Arms
//!
//! - `c=…` — the plain constant ladder, re-reported per quartile (step 0).
//! - `fpu=…` — first-play-urgency reduction ([`MctsConfig::fpu_reduction`]).
//!   The default `INFINITY` FPU visits every child once before any is
//!   revisited, at every node and every depth; that is a `c_puct`-independent
//!   source of flat visit allocation and it is confounded with any reading of
//!   the constant. Step 1 clears it.
//! - `vstd=…` — KataGo's variance-scaled `c_puct`
//!   ([`MctsConfig::cpuct_utility_stdev_scale`]), step 2.
//! - `sh` — Sequential Halving root budget allocation
//!   ([`MctsConfig::root_sequential_halving`]), step 3.
//!
//! See `docs/research/2026-08-14-adaptive-exploration-beyond-fixed-c-puct.md`.
//!
//! ```sh
//! cargo run --release -p sts2-bench --example mcts_exploration_arms -- \
//!     [book_json] [pool_root] [seeds] [sims] [arm_filter_substring]
//! ```

#[path = "common/mod.rs"]
mod common;

use std::collections::BTreeMap;
use std::path::PathBuf;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use sts2_cli::scenario_to_combat_state;
use sts2_cli::sim::scenario::toml_file::load_scenario_file;
use sts2_core::{Action, CombatPhase, CombatState};
use sts2_mcts::{
    EmbeddingValueNet, HiddenInfoMode, LearnedPolicyPrior, LearnedValueNet, Mcts, MctsConfig,
    SharedPolicy,
};

#[derive(Debug, Clone, Deserialize)]
struct Book {
    oracle_sims: u32,
    chance_agreement: f64,
    entries: Vec<BookEntry>,
}

#[derive(Debug, Clone, Deserialize)]
struct BookEntry {
    character: String,
    tier: String,
    fixture_rel_path: String,
    seed: u64,
    state_hash: u64,
    gold_action: String,
}

/// Position-level volatility, measured once by a reference search and cached.
/// Keyed on `state_hash` so a stale book invalidates it automatically.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct Volatility {
    state_hash: u64,
    /// `max Q - min Q` over the reference search's (transposition-collapsed)
    /// root children. The quantity FINDINGS §4 measured at a median of 0.034.
    q_spread: f64,
    /// Reference search's separation between its top two children.
    top2_gap: f64,
    /// Standard deviation of the rollout values backed up through the root —
    /// the calibration target for `cpuct_utility_stdev_prior`.
    root_stdev: f64,
    /// Standard deviation of every visited node's Q — the Global Std
    /// statistic, and the calibration target for `global_std_c`.
    tree_stdev: f64,
}

/// The deployed dialect. Every arm is a delta from this, so an arm that
/// changes nothing is a bit-for-bit control.
fn base_config(sims: u32, seed: u64) -> MctsConfig {
    MctsConfig {
        max_simulations: sims,
        seed,
        max_rollout_depth: 200,
        hidden_info: HiddenInfoMode::BeliefDraw,
        c_puct: 2.5,
        draw_widening_c: 2.0,
        ..MctsConfig::default()
    }
}

/// A learned prior smoothed toward uniform: `P' = (1-w)/k + w·P`.
///
/// Needed because `c_puct` cannot flatten a learned prior the way it flattens
/// a uniform one. The explore term is `c·P(a)·sqrt(N)/(1+n)`, so an action the
/// net assigns `P ~ 0` gets ~0 bonus **at every `c`** — raising `c` only
/// inflates the actions that were already favoured. Measured: the raw prior
/// sits at `med v/v` 30-60 across the whole ladder from 2.5 to 76, i.e. the
/// search is near-greedy and `c` does not move it. The mixing weight is the
/// knob that does, and it is the standard prior-smoothing shape.
///
/// `w = 0` reduces exactly to `UniformPolicy`, so the ladder carries its own
/// control rather than relying on a separately-run one.
#[derive(Debug)]
struct BlendedPrior<N: sts2_mcts::PolicyNet> {
    inner: LearnedPolicyPrior<N>,
    weight: f32,
}

impl<N: sts2_mcts::PolicyNet> sts2_mcts::Policy for BlendedPrior<N> {
    fn prior(&self, state: &CombatState, actions: &[Action]) -> SmallVec<[f32; 16]> {
        let learned = self.inner.prior(state, actions);
        #[allow(clippy::cast_precision_loss)]
        let uniform = 1.0_f32 / actions.len().max(1) as f32;
        learned
            .into_iter()
            .map(|p| (1.0 - self.weight) * uniform + self.weight * p)
            .collect()
    }
}

/// Total illegal-root-action fallbacks across every cell of the run.
static ILLEGAL_ROOT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// One benchmarked arm: a name, a mutation of the deployed config, and the
/// prior the search consults. `prior: None` means [`SharedPolicy::uniform`],
/// which is byte-identical to the `Mcts::new` path every other arm uses.
struct Arm {
    name: String,
    apply: Box<dyn Fn(&mut MctsConfig) + Sync + Send>,
    prior: Option<SharedPolicy>,
}

fn arms(stdev_prior: f32) -> Vec<Arm> {
    let mut v: Vec<Arm> = Vec::new();
    let mut push = |name: String, f: Box<dyn Fn(&mut MctsConfig) + Sync + Send>| {
        v.push(Arm {
            name,
            apply: f,
            prior: None,
        });
    };

    // Step 0 — the constant ladder, for the per-quartile crossing test and the
    // sims x c sweep. Runs down to 0.1 because the optimum MOVES WITH BUDGET:
    // at 200 sims the best arm was 0.40, the old grid's floor, so the optimum
    // was not bracketed there. It is bracketed now at both ends at every
    // budget measured.
    for c in [0.1_f32, 0.2, 0.4, 0.8, 1.41, 2.5, 4.0, 8.0] {
        push(
            format!("c={c:.2}"),
            Box::new(move |cfg: &mut MctsConfig| cfg.c_puct = c),
        );
    }
    // Step 1 — FPU reduction. `0.0` values an unexpanded action at the parent
    // and approximates the historical breadth-first rule; the penalty only
    // bites once it is commensurate with the explore term it competes against
    // (`c·sqrt(ln N)` ~ 6.6 at c=2.5, N=1000), which is why the ladder runs so
    // far past KataGo's 0.2 — their 0.2 is against a PUCT bonus, not this one.
    for r in [0.0_f32, 1.0, 3.0, 6.0, 9.0] {
        push(
            format!("fpu={r:.2}"),
            Box::new(move |cfg: &mut MctsConfig| cfg.fpu_reduction = Some(r)),
        );
    }
    // Step 2 — KataGo variance-scaled c_puct, prior calibrated to this book.
    for s in [0.4_f32, 0.85] {
        push(
            format!("vstd={s:.2}"),
            Box::new(move |cfg: &mut MctsConfig| {
                cfg.cpuct_utility_stdev_scale = s;
                cfg.cpuct_utility_stdev_prior = stdev_prior;
            }),
        );
    }
    // Step 3 — Global Std: replace the constant with `C · σ` over every
    // visited node's Q. `C = 2` is the paper's generalizing default.
    // Ladder re-centred on the measured tree sigma (~0.1), not on the paper's
    // `C ~ 2`: `C = 25` is what reproduces the deployed effective `c` here.
    for cg in [8.0_f32, 16.0, 25.0, 40.0] {
        push(
            format!("gstd={cg:.1}"),
            Box::new(move |cfg: &mut MctsConfig| cfg.global_std_c = Some(cg)),
        );
    }
    // Step 4 — Sequential Halving at the root (no exploration constant).
    push(
        "sh".to_string(),
        Box::new(|cfg: &mut MctsConfig| cfg.root_sequential_halving = true),
    );
    // Combinations of whatever individually helps: FPU is orthogonal to both
    // the variance scaling and the root schedule, so the interaction is worth
    // one cell each rather than a second full grid.
    push(
        "fpu=3.00+vstd=0.85".to_string(),
        Box::new(move |cfg: &mut MctsConfig| {
            cfg.fpu_reduction = Some(3.0);
            cfg.cpuct_utility_stdev_scale = 0.85;
            cfg.cpuct_utility_stdev_prior = stdev_prior;
        }),
    );
    push(
        "fpu=3.00+sh".to_string(),
        Box::new(|cfg: &mut MctsConfig| {
            cfg.fpu_reduction = Some(3.0);
            cfg.root_sequential_halving = true;
        }),
    );
    // Step 5 — the PUCT branch, with a UNIFORM prior. `use_puct_prior` does
    // two things at once and only one of them is "consult the policy net":
    // it also swaps the explore term from `c·sqrt(ln N / n)` to
    // `c·P·sqrt(N)/(1+n)`. Those have different magnitudes AND different
    // shapes, so a learned-prior arm measured against the deployed control
    // confounds prior quality with the formula switch. Holding `P = 1/k`
    // isolates the formula.
    //
    // The ladder runs an order of magnitude past the UCT one because the two
    // terms are not commensurate *in magnitude*. At the book's median
    // branching (k = 6), evaluated at a uniform visit split (n = N/k = 167),
    // UCT gives `2.5·sqrt(ln 1000 / 167) = 0.51` against PUCT's
    // `2.5·(1/6)·sqrt(1000) / 168 = 0.079` — 6.5x smaller, which predicts an
    // equivalent `c ~ 16`.
    //
    // **Measured, that prediction is wrong** (2026-08-15, 168 positions x 16
    // seeds @1000 sims): `puct=19.0` already loses 2.6pp (cluster p = 0.011)
    // and the flat optimum is `c` in [2.5, 10] at −0.5 to −0.8pp vs the
    // deployed UCT control, none of them significant. The magnitude
    // calculation misleads because it evaluates at a uniform split that
    // neither family produces: read instead through the realized
    // concentration, where the two agree closely — every arm within 1pp of
    // the control, in both families, lands at `med v/v` between 1.4 and 2.0.
    // The grid is kept as-is so the losing tail stays on the record.
    for c in [2.5_f32, 5.0, 10.0, 19.0, 38.0, 76.0] {
        push(
            format!("puct={c:.1}"),
            Box::new(move |cfg: &mut MctsConfig| {
                cfg.use_puct_prior = true;
                cfg.c_puct = c;
            }),
        );
    }
    // Step 6 — the learned prior, same branch, same ladder. Only present when
    // `STS2_ARMS_PRIOR_NET` points at a `value_policy_net.py` weights JSON;
    // paired against the `puct=` arms above, the difference is prior quality
    // and nothing else. A malformed asset panics rather than silently
    // degrading to uniform — a typo that quietly reproduced the uniform arm
    // would read as "the learned prior adds nothing", the exact false null
    // this arm exists to avoid.
    if let Ok(path) = std::env::var("STS2_ARMS_PRIOR_NET") {
        let net = std::sync::Arc::new(
            LearnedValueNet::from_weights_path(std::path::Path::new(&path))
                .unwrap_or_else(|e| panic!("STS2_ARMS_PRIOR_NET={path}: failed to load: {e}")),
        );
        for c in [2.5_f32, 5.0, 10.0, 19.0, 38.0, 76.0] {
            v.push(Arm {
                name: format!("prior={c:.1}"),
                apply: Box::new(move |cfg: &mut MctsConfig| {
                    cfg.use_puct_prior = true;
                    cfg.c_puct = c;
                }),
                prior: Some(SharedPolicy::new(LearnedPolicyPrior::new(
                    std::sync::Arc::clone(&net),
                ))),
            });
        }
        // Blend ladder at the deployed constant. `c` is held at 2.5 because
        // the `prior=` ladder above establishes that `c` is not the knob here;
        // `w` is. `w = 0.00` must reproduce `puct=2.5` exactly — it is the
        // in-run control, and a disagreement between them means the blend
        // plumbing is wrong rather than the prior being weak.
        for w in [0.0_f32, 0.1, 0.25, 0.5, 0.75, 1.0] {
            let net = std::sync::Arc::clone(&net);
            v.push(Arm {
                name: format!("blend={w:.2}"),
                apply: Box::new(|cfg: &mut MctsConfig| {
                    cfg.use_puct_prior = true;
                    cfg.c_puct = 2.5;
                }),
                prior: Some(SharedPolicy::new(BlendedPrior {
                    inner: LearnedPolicyPrior::new(net),
                    weight: w,
                })),
            });
        }
    }
    // The SHIPPED prior. `sts2-py` bakes an `EmbeddingValueNet` policy prior
    // into the wheel via `include_bytes!` and `mcts_play_to_terminal` turns it
    // on by default, gated to Ironclad because it cost Necrobinder ~21pp
    // (`experiments/necro-vs-ic-lift`). That panel also measured it costing
    // *Ironclad* 1.9pp, and closed with: "Choosing the gate width needs a
    // mid-run state source this harness does not have." The book is one — it
    // is mid-run, on-manifold, five characters, and its answers come from a
    // 50000-sim double oracle rather than from the same search being judged.
    // So point the same instrument at the shipped artifact.
    if let Ok(path) = std::env::var("STS2_ARMS_PRIOR_EMBED") {
        let net = std::sync::Arc::new(
            EmbeddingValueNet::from_weights_path(std::path::Path::new(&path))
                .unwrap_or_else(|e| panic!("STS2_ARMS_PRIOR_EMBED={path}: failed to load: {e}")),
        );
        for w in [0.0_f32, 0.1, 0.25, 0.5, 1.0] {
            let net = std::sync::Arc::clone(&net);
            v.push(Arm {
                name: format!("shipped={w:.2}"),
                apply: Box::new(|cfg: &mut MctsConfig| {
                    cfg.use_puct_prior = true;
                    cfg.c_puct = 2.5;
                }),
                prior: Some(SharedPolicy::new(BlendedPrior {
                    inner: LearnedPolicyPrior::new(net),
                    weight: w,
                })),
            });
        }
    }
    v
}

fn trial_seed(t: u32) -> u64 {
    u64::from(t)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(0x51ED_2701)
}

/// Realize a book entry and verify it against the book's state hash. A
/// mismatch means the fixture or engine moved under the book; scoring against
/// a stale gold answer would silently measure noise.
fn realize_verified(pool_root: &PathBuf, e: &BookEntry) -> Option<CombatState> {
    let path = pool_root.join(&e.fixture_rel_path);
    let scenario = load_scenario_file(&path).ok()?;
    let mut state = scenario_to_combat_state(&scenario, e.seed).ok()?;
    let _ = sts2_core::combat_loop::start_combat(&mut state);
    let _ = sts2_core::combat_loop::start_player_turn(&mut state);
    if state.is_over || state.phase != CombatPhase::PlayerPlayPhase {
        return None;
    }
    if sts2_core::state_hash(&state) != e.state_hash {
        eprintln!(
            "[arms] STATE HASH MISMATCH for {} — the book is stale against this \
             engine. Rebuild it (combat_position_book) before trusting any number.",
            e.fixture_rel_path
        );
        return None;
    }
    Some(state)
}

/// Transposition-collapsed root children as `(visits, mean_value)`.
///
/// `root_outcomes()` reports one merged node once per incoming root edge, so
/// duplicated entries carry byte-identical stats and raw counts overstate both
/// the child count and the visit spread (2026-08-11 POSTFIX correction).
fn collapsed_children<P: sts2_mcts::Policy>(m: &Mcts<P>) -> Vec<(u32, f32)> {
    let mut seen: BTreeMap<(u32, u32, u32), (u32, f32)> = BTreeMap::new();
    for o in m.root_outcomes() {
        #[allow(clippy::cast_precision_loss)]
        let mean = o.value_sum / o.visits.max(1) as f32;
        seen.entry((o.visits, o.wins, o.losses))
            .or_insert((o.visits, mean));
    }
    let mut vals: Vec<(u32, f32)> = seen.into_values().collect();
    vals.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    vals
}

/// Reference search: measures the position, not any arm. Deployed config, one
/// seed, `ref_sims` budget.
fn measure_volatility(state: &CombatState, state_hash: u64, ref_sims: u32) -> Volatility {
    let mut m = Mcts::new(state.clone(), base_config(ref_sims, 0x5EED_0001));
    let _ = m.search_n(ref_sims);
    let vals = collapsed_children(&m);
    let (q_spread, top2_gap) = if vals.len() < 2 {
        (0.0, 0.0)
    } else {
        let hi = vals[0].1;
        let lo = vals[vals.len() - 1].1;
        (f64::from(hi - lo), f64::from(vals[0].1 - vals[1].1))
    };
    Volatility {
        state_hash,
        q_spread,
        top2_gap,
        root_stdev: f64::from(m.root_utility_stdev()),
        tree_stdev: f64::from(m.tree_q_stdev()),
    }
}

/// One arm's measurements on one position, averaged over seeds.
#[derive(Clone, Copy)]
struct Cell {
    /// Per-seed hit bitmask (bit `t` set iff seed `t` picked the gold action).
    /// Retained so arms can be compared **paired** — same position, same seed —
    /// rather than by their marginal means, which is what a significance test
    /// on this design requires.
    hits: u64,
    agree: f64,
    vmax_vmin: f64,
    top_share: f64,
    top_gap: f64,
    stability: f64,
}

fn run_cell(state: &CombatState, gold: &str, arm: &Arm, sims: u32, seeds: u32) -> Cell {
    let mut agree = 0.0;
    let mut vmax_vmin = 0.0;
    let mut top_share = 0.0;
    let mut top_gap = 0.0;
    let mut counted = 0.0_f64;
    let mut picks: BTreeMap<String, u32> = BTreeMap::new();
    let mut hits: u64 = 0;

    for t in 0..seeds {
        let mut cfg = base_config(sims, trial_seed(t));
        (arm.apply)(&mut cfg);
        // `SharedPolicy::uniform()` + `use_puct_prior = false` is documented
        // byte-identical to `Mcts::new`, so routing every arm through
        // `with_policy` leaves the pre-existing arms bit-for-bit unchanged.
        let policy = arm.prior.clone().unwrap_or_else(SharedPolicy::uniform);
        let mut m = Mcts::with_policy(state.clone(), cfg, policy);
        let action = m.search_n(sims);
        // Report the illegal-root-action fallback rate (sts2-mcts
        // `illegal_root_actions`). Expected 0; a non-zero count means the
        // search committed to a weaker pick rule than intended on that cell.
        if m.illegal_root_actions() > 0 {
            ILLEGAL_ROOT.fetch_add(
                m.illegal_root_actions(),
                std::sync::atomic::Ordering::Relaxed,
            );
        }
        let pick = format!("{action:?}");
        *picks.entry(pick.clone()).or_default() += 1;
        if pick == gold {
            agree += 1.0;
            if t < 64 {
                hits |= 1_u64 << t;
            }
        }

        let vals = collapsed_children(&m);
        if vals.len() < 2 {
            continue;
        }
        let visits: Vec<u32> = vals.iter().map(|(v, _)| *v).collect();
        let total: u32 = visits.iter().sum();
        let k = vals.len();
        #[allow(clippy::cast_precision_loss)]
        {
            vmax_vmin += f64::from(*visits.iter().max().unwrap())
                / f64::from(*visits.iter().min().unwrap()).max(1.0);
            top_share += (f64::from(*visits.iter().max().unwrap()) / f64::from(total.max(1)))
                / (1.0 / k as f64);
            top_gap += f64::from(vals[0].1 - vals[1].1);
        }
        counted += 1.0;
    }

    let n = f64::from(seeds.max(1));
    let c = counted.max(1.0);
    let modal = picks.values().max().copied().unwrap_or(0);
    Cell {
        hits,
        agree: agree / n,
        vmax_vmin: vmax_vmin / c,
        top_share: top_share / c,
        top_gap: top_gap / c,
        stability: f64::from(modal) / n,
    }
}

/// Complementary error function, Numerical Recipes Chebyshev form
/// (fractional error < 1.2e-7). Enough for reporting a p-value to two
/// significant figures; nothing here turns on the seventh digit.
fn erfc(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let poly = -z * z - 1.265_512_23
        + t * (1.000_023_68
            + t * (0.374_091_96
                + t * (0.096_784_18
                    + t * (-0.186_288_06
                        + t * (0.278_868_07
                            + t * (-1.135_203_98
                                + t * (1.488_515_87 + t * (-0.822_152_23 + t * 0.170_872_77))))))));
    let ans = t * poly.exp();
    if x >= 0.0 {
        ans
    } else {
        2.0 - ans
    }
}

/// Two-sided p for a chi-square statistic on 1 df.
fn chi2_1df_p(x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    erfc((x / 2.0).sqrt())
}

/// Paired comparison of one arm against the control.
struct Paired {
    /// (position, seed) cells the arm got right and the control got wrong.
    b: u64,
    /// (position, seed) cells the control got right and the arm got wrong.
    c: u64,
    /// McNemar chi-square with Yates continuity correction, and its p.
    mcnemar_p: f64,
    /// Mean per-position difference in agreement rate (arm − control).
    mean_delta: f64,
    /// Cluster-robust p: paired t over the 168 per-position rate differences.
    /// The seeds inside a position are **not** independent — the same board,
    /// the same gold answer — so cell-level McNemar treats 168×16 correlated
    /// observations as 2688 independent ones and reports a p that is too
    /// small. This is the honest one; McNemar is reported alongside because
    /// it is what the literature on this design names.
    cluster_p: f64,
}

fn paired(arm: &[(usize, Cell)], ctrl: &[(usize, Cell)], seeds: u32) -> Paired {
    let ctrl_by_pos: BTreeMap<usize, &Cell> = ctrl.iter().map(|(pi, c)| (*pi, c)).collect();
    let mut b = 0_u64;
    let mut c = 0_u64;
    let mut deltas: Vec<f64> = Vec::new();
    let usable = seeds.min(64);
    for (pi, cell) in arm {
        let Some(ref_cell) = ctrl_by_pos.get(pi) else {
            continue;
        };
        for t in 0..usable {
            let a_hit = cell.hits >> t & 1 == 1;
            let c_hit = ref_cell.hits >> t & 1 == 1;
            match (a_hit, c_hit) {
                (true, false) => b += 1,
                (false, true) => c += 1,
                _ => {}
            }
        }
        deltas.push(cell.agree - ref_cell.agree);
    }
    #[allow(clippy::cast_precision_loss)]
    let disc = (b + c) as f64;
    #[allow(clippy::cast_precision_loss)]
    let chi2 = if disc > 0.0 {
        let d = (b as f64 - c as f64).abs() - 1.0;
        if d > 0.0 {
            d * d / disc
        } else {
            0.0
        }
    } else {
        0.0
    };
    #[allow(clippy::cast_precision_loss)]
    let n = deltas.len() as f64;
    let mean_delta = if n > 0.0 {
        deltas.iter().sum::<f64>() / n
    } else {
        0.0
    };
    let cluster_p = if n > 1.0 {
        let var = deltas.iter().map(|d| (d - mean_delta).powi(2)).sum::<f64>() / (n - 1.0);
        let se = (var / n).sqrt();
        if se > 0.0 {
            erfc((mean_delta / se).abs() / std::f64::consts::SQRT_2)
        } else {
            1.0
        }
    } else {
        1.0
    };
    Paired {
        b,
        c,
        mcnemar_p: chi2_1df_p(chi2),
        mean_delta,
        cluster_p,
    }
}

fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[v.len() / 2]
}

fn main() -> std::io::Result<()> {
    sts2_content::cards::dispatch::ensure_installed();
    sts2_content::relics::dispatch::ensure_installed();
    sts2_content::potions::dispatch::ensure_installed();

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut it = args.iter();
    let book_path = PathBuf::from(
        it.next()
            .cloned()
            .unwrap_or_else(|| "data/combat_position_book.json".to_string()),
    );
    let pool_root = PathBuf::from(
        it.next()
            .cloned()
            .unwrap_or_else(|| "data/run-fixtures".to_string()),
    );
    let seeds: u32 = it.next().and_then(|s| s.parse().ok()).unwrap_or(4);
    let sims: u32 = it.next().and_then(|s| s.parse().ok()).unwrap_or(1000);
    let filter: Option<String> = it.next().cloned();
    let ref_sims: u32 = std::env::var("REF_SIMS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8000);
    let vol_cache = PathBuf::from(
        std::env::var("VOL_CACHE").unwrap_or_else(|_| "data/book_volatility.json".to_string()),
    );

    let book: Book =
        serde_json::from_slice(&std::fs::read(&book_path)?).expect("position book parses");
    eprintln!(
        "[arms] book: {} positions (oracle {} sims), chance {:.3}; \
         {seeds} seeds x {sims} sims",
        book.entries.len(),
        book.oracle_sims,
        book.chance_agreement,
    );

    std::panic::set_hook(Box::new(|_| {}));
    let realized: Vec<(BookEntry, CombatState)> = book
        .entries
        .par_iter()
        .filter_map(|e| {
            let s = std::panic::catch_unwind(|| realize_verified(&pool_root, e))
                .ok()
                .flatten()?;
            Some((e.clone(), s))
        })
        .collect();
    let _ = std::panic::take_hook();
    eprintln!(
        "[arms] {} of {} positions verified against the book's state hash",
        realized.len(),
        book.entries.len()
    );
    assert!(
        realized.len() * 2 >= book.entries.len(),
        "more than half the book failed to verify — refusing to report on a stale book"
    );

    // ── Volatility pass (position property, measured once, cached) ────────
    let cached: BTreeMap<u64, Volatility> = std::fs::read(&vol_cache)
        .ok()
        .and_then(|b| serde_json::from_slice::<Vec<Volatility>>(&b).ok())
        .map(|v| v.into_iter().map(|x| (x.state_hash, x)).collect())
        .unwrap_or_default();
    let missing: Vec<usize> = realized
        .iter()
        .enumerate()
        .filter(|(_, (e, _))| !cached.contains_key(&e.state_hash))
        .map(|(i, _)| i)
        .collect();
    if !missing.is_empty() {
        eprintln!(
            "[arms] measuring volatility for {} positions at {ref_sims} ref sims \
             ({} cached)",
            missing.len(),
            cached.len()
        );
    }
    let fresh: Vec<Volatility> = missing
        .par_iter()
        .map(|&i| {
            let (e, s) = &realized[i];
            measure_volatility(s, e.state_hash, ref_sims)
        })
        .collect();
    let mut vol = cached;
    for v in fresh {
        vol.insert(v.state_hash, v);
    }
    let all_vol: Vec<Volatility> = vol.values().copied().collect();
    let _ = std::fs::write(&vol_cache, serde_json::to_vec_pretty(&all_vol).unwrap());

    let mut spreads: Vec<f64> = realized
        .iter()
        .map(|(e, _)| vol[&e.state_hash].q_spread)
        .collect();
    let mut stdevs: Vec<f64> = realized
        .iter()
        .map(|(e, _)| vol[&e.state_hash].root_stdev)
        .collect();
    let mut tree_sds: Vec<f64> = realized
        .iter()
        .map(|(e, _)| vol[&e.state_hash].tree_stdev)
        .collect();
    let med_tree_sd = median(&mut tree_sds);
    let med_stdev = median(&mut stdevs);
    let med_spread = median(&mut spreads.clone());
    eprintln!(
        "[arms] reference root Q spread: median {med_spread:.4}; root utility stdev: \
         median {med_stdev:.4} (this is what `cpuct_utility_stdev_prior` is set \
         to); tree-wide Q stdev: median {med_tree_sd:.4} (multiply `gstd=C` by \
         this to read it as an effective c_puct)"
    );

    // Quartile edges on the reference Q spread. Q1 = flattest positions
    // (least resolved), Q4 = most decided.
    spreads.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let edges = [
        spreads[spreads.len() / 4],
        spreads[spreads.len() / 2],
        spreads[3 * spreads.len() / 4],
    ];
    let quartile = |s: f64| -> usize {
        if s <= edges[0] {
            0
        } else if s <= edges[1] {
            1
        } else if s <= edges[2] {
            2
        } else {
            3
        }
    };
    eprintln!(
        "[arms] Q-spread quartile edges: {:.4} / {:.4} / {:.4}",
        edges[0], edges[1], edges[2]
    );

    #[allow(clippy::cast_possible_truncation)]
    let arm_list: Vec<Arm> = arms(med_stdev as f32)
        .into_iter()
        .filter(|a| {
            // Comma-separated substrings, so one invocation can cover a
            // hand-picked set of arms — at 4000 sims the full grid is ~35
            // minutes and the robustness question only needs a few of them.
            filter
                .as_ref()
                .is_none_or(|f| f.split(',').any(|part| a.name.contains(part.trim())))
        })
        .collect();
    assert!(!arm_list.is_empty(), "arm filter matched nothing");
    eprintln!(
        "[arms] {} arms x {} positions",
        arm_list.len(),
        realized.len()
    );

    // Every (arm, position) cell is independent; flatten so one rayon pool
    // covers the whole grid rather than parallelising per arm.
    let started = std::time::Instant::now();
    let cells: Vec<(usize, usize, Cell)> = arm_list
        .iter()
        .enumerate()
        .flat_map(|(ai, a)| {
            realized
                .iter()
                .enumerate()
                .map(move |(pi, r)| (ai, pi, a, r))
        })
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|(ai, pi, arm, (entry, state))| {
            (
                ai,
                pi,
                run_cell(state, &entry.gold_action, arm, sims, seeds),
            )
        })
        .collect();
    let wall = started.elapsed();

    let mut by_arm: Vec<Vec<(usize, Cell)>> = (0..arm_list.len()).map(|_| Vec::new()).collect();
    for (ai, pi, cell) in cells {
        by_arm[ai].push((pi, cell));
    }

    println!("\n=== exploration arms on the position book ===");
    println!(
        "positions={} seeds={seeds} sims={sims} ref_sims={ref_sims} \
         chance_agreement={:.3} wall={:.1}s",
        realized.len(),
        book.chance_agreement,
        wall.as_secs_f64()
    );
    println!(
        "\n{:>18}  {:>8}  {:>7}  {:>7}  {:>7}  {:>7}  {:>8}  {:>9}  {:>8}  {:>9}",
        "arm",
        "gold_agr",
        "Q1flat",
        "Q2",
        "Q3",
        "Q4sharp",
        "med v/v",
        "med share",
        "med gap",
        "stability"
    );

    let mut best = (0usize, -1.0_f64);
    let mut rows: Vec<(String, f64, [f64; 4])> = Vec::new();
    for (ai, arm) in arm_list.iter().enumerate() {
        let r = &by_arm[ai];
        if r.is_empty() {
            continue;
        }
        #[allow(clippy::cast_precision_loss)]
        let n = r.len() as f64;
        let agr = r.iter().map(|(_, c)| c.agree).sum::<f64>() / n;
        if agr > best.1 {
            best = (ai, agr);
        }
        let mut q_sum = [0.0_f64; 4];
        let mut q_n = [0.0_f64; 4];
        for (pi, c) in r {
            let q = quartile(vol[&realized[*pi].0.state_hash].q_spread);
            q_sum[q] += c.agree;
            q_n[q] += 1.0;
        }
        let mut q = [0.0_f64; 4];
        for i in 0..4 {
            q[i] = if q_n[i] > 0.0 {
                q_sum[i] / q_n[i]
            } else {
                f64::NAN
            };
        }
        let mut vv: Vec<f64> = r.iter().map(|(_, c)| c.vmax_vmin).collect();
        let mut shr: Vec<f64> = r.iter().map(|(_, c)| c.top_share).collect();
        let mut gap: Vec<f64> = r.iter().map(|(_, c)| c.top_gap).collect();
        let stab = r.iter().map(|(_, c)| c.stability).sum::<f64>() / n;
        let tag = if arm.name == "c=2.50" {
            "  <- deployed"
        } else {
            ""
        };
        println!(
            "{:>18}  {agr:>8.3}  {:>7.3}  {:>7.3}  {:>7.3}  {:>7.3}  {:>8.2}  \
             {:>9.3}  {:>8.4}  {stab:>9.3}{tag}",
            arm.name,
            q[0],
            q[1],
            q[2],
            q[3],
            median(&mut vv),
            median(&mut shr),
            median(&mut gap),
        );
        rows.push((arm.name.clone(), agr, q));
    }
    println!(
        "\nbest arm: {} at gold_agr={:.3}",
        arm_list[best.0].name, best.1
    );

    // The crossing test: does the constant ladder peak at a different `c` in
    // the flat quartile than in the sharp one? If not, adaptivity has no
    // headroom to collect and the deployed constant is right everywhere.
    let ladder: Vec<&(String, f64, [f64; 4])> = rows
        .iter()
        .filter(|(n, _, _)| n.starts_with("c="))
        .collect();
    if ladder.len() > 1 {
        println!("\n--- crossing test: best c_puct within each Q-spread quartile ---");
        for (qi, label) in ["Q1 (flattest)", "Q2", "Q3", "Q4 (sharpest)"]
            .iter()
            .enumerate()
        {
            let best_c = ladder
                .iter()
                .max_by(|a, b| {
                    a.2[qi]
                        .partial_cmp(&b.2[qi])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap();
            println!(
                "{label:>15}: best {} at {:.3}  (deployed c=2.50 at {:.3})",
                best_c.0,
                best_c.2[qi],
                ladder
                    .iter()
                    .find(|(n, _, _)| n == "c=2.50")
                    .map_or(f64::NAN, |r| r.2[qi])
            );
        }
    }

    // Paired significance against the deployed control. Marginal means cannot
    // settle a 1pp difference; these run on the same positions AND the same
    // seeds, which is what the design supports.
    if let Some(ci) = arm_list.iter().position(|a| a.name == "c=2.50") {
        println!("\n--- paired vs control (c=2.50), same positions and seeds ---");
        println!(
            "{:>18}  {:>9}  {:>7}  {:>7}  {:>11}  {:>11}",
            "arm", "Δ mean", "b", "c", "McNemar p", "cluster p"
        );
        for (ai, arm) in arm_list.iter().enumerate() {
            if ai == ci {
                continue;
            }
            let p = paired(&by_arm[ai], &by_arm[ci], seeds);
            println!(
                "{:>18}  {:>+9.4}  {:>7}  {:>7}  {:>11.4}  {:>11.4}",
                arm.name, p.mean_delta, p.b, p.c, p.mcnemar_p, p.cluster_p
            );
        }
        println!(
            "b = arm right & control wrong; c = control right & arm wrong. \n\
             cluster p is a paired t over the {} per-position rate differences \n\
             and is the one to read — seeds within a position are correlated, so \n\
             McNemar's cell-level p is anticonservative here.",
            realized.len()
        );
    }

    // Per-character x per-arm. The book is stratified by character and the
    // characters are not interchangeable — Silent carries ~3x Ironclad's rate
    // of transposition-merged root children, and the shipped prior is
    // character-GATED — so a pooled optimum can be the best single number
    // while being wrong for most individual characters. This is the cut that
    // shows it.
    //
    // Per-character n is 30-40, so read the columns as indicative and the
    // pooled column as the powered one; `seeds` is the lever if a column
    // needs to carry weight on its own.
    {
        let mut chars: Vec<String> = realized
            .iter()
            .map(|(e, _)| e.character.clone())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        chars.sort();
        print!("\n{:>18}", "arm");
        for c in &chars {
            print!("{c:>13}");
        }
        println!("{:>13}", "POOLED");
        for (ai, arm) in arm_list.iter().enumerate() {
            let mut per: BTreeMap<&str, (f64, u32)> = BTreeMap::new();
            let mut tot = (0.0_f64, 0_u32);
            for (pi, cell) in &by_arm[ai] {
                let e = &realized[*pi].0;
                let slot = per.entry(e.character.as_str()).or_insert((0.0, 0));
                slot.0 += cell.agree;
                slot.1 += 1;
                tot.0 += cell.agree;
                tot.1 += 1;
            }
            print!("{:>18}", arm.name);
            for c in &chars {
                match per.get(c.as_str()) {
                    Some((sum, n)) if *n > 0 => print!("{:>13.3}", sum / f64::from(*n)),
                    _ => print!("{:>13}", "-"),
                }
            }
            println!("{:>13.3}", tot.0 / f64::from(tot.1.max(1)));
        }
        let illegal = ILLEGAL_ROOT.load(std::sync::atomic::Ordering::Relaxed);
        if illegal > 0 {
            println!(
                "\n[arms] WARNING: {illegal} illegal-root-action fallbacks across this run \
                 (sts2-mcts `illegal_root_actions`) — the search took a weaker pick rule \
                 on those decisions"
            );
        }
        print!("{:>18}", "n positions");
        for c in &chars {
            let n = realized.iter().filter(|(e, _)| &e.character == c).count();
            print!("{n:>13}");
        }
        println!("{:>13}", realized.len());
    }

    // Per-tier for the best arm and the deployed control.
    let deployed_idx = arm_list.iter().position(|a| a.name == "c=2.50");
    for (label, idx) in [("best", Some(best.0)), ("deployed", deployed_idx)] {
        let Some(idx) = idx else { continue };
        let mut per: BTreeMap<String, (f64, u32)> = BTreeMap::new();
        for (pi, cell) in &by_arm[idx] {
            let e = &realized[*pi].0;
            let slot = per.entry(e.tier.clone()).or_insert((0.0, 0));
            slot.0 += cell.agree;
            slot.1 += 1;
        }
        print!("\n{label} ({}) by tier: ", arm_list[idx].name);
        for (tier, (sum, n)) in &per {
            print!("{tier} {:.3} (n={n})  ", sum / f64::from(*n));
        }
        println!();
    }
    Ok(())
}
