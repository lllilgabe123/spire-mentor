//! `surrogate_eval_data` — emit the data to answer two questions about the
//! learned combat surrogate, both on the **realistic** state distribution the
//! run-AI actually visits (not the synthetic training distribution):
//!
//! 1. **Generalization.** Does a surrogate trained on *synthetic* states still
//!    predict win / HP-loss on *real self-play* pre-combat states? We snapshot
//!    realistic states from heuristic self-play, re-label each with the real
//!    grounded probe (`encounter_delta`), and let the Python side score the
//!    existing surrogate on them (no retraining) — the held-out-vs-realistic
//!    gap is the answer.
//!
//! 2. **Decision agreement.** The deployment use is *picking* (card rewards,
//!    etc.). What matters is whether the surrogate *ranks the candidate
//!    options the same way the real rollout does*. At every card-shaped
//!    decision we take the authoritative per-candidate rollout score
//!    (`fight_delta().score()` — the exact teacher the `ranker_chain500` net
//!    was distilled from) and emit, per candidate, the context features +
//!    which encounters the decision is probed against. The Python side scores
//!    each candidate with the surrogate and reports **Kendall τ** (the E3
//!    calibration metric), top-1 agreement, and decision regret.
//!
//! Realistic decisions come from `calibrate::collect_decision_points`
//! (heuristic self-play, C#-faithful run setup) — the same harvester the
//! rollout-ranker calibration uses.

use std::io::Write as _;

use clap::Parser;
use rayon::prelude::*;
use serde::Serialize;

use sts2_content::encounters::{
    act1_overgrowth::ACT1_ENCOUNTERS, act2_hive::ACT2_ENCOUNTERS, act3_glory::ACT3_ENCOUNTERS,
    EncounterDef,
};
use sts2_core::potions::CharacterId;
use sts2_runai::calibrate::collect_decision_points;
use sts2_runai::eval::ranker_features::ContextFeatures;
use sts2_runai::rollout::{
    encounter_delta, ensure_content_installed, fight_delta, CandidateMutation, FightCache,
    ProbeBudget,
};
use sts2_runai::view::VisibleRunView;

#[derive(Parser, Debug)]
#[command(
    name = "surrogate_eval_data",
    about = "Realistic-distribution eval data for the combat surrogate."
)]
struct Args {
    /// Self-play seeds to harvest decisions from.
    #[arg(long, default_value_t = 600)]
    seeds: u32,
    #[arg(long, default_value_t = 10001)]
    seed_start: u32,
    #[arg(long, default_value = "ironclad", value_parser = parse_character)]
    character: CharacterId,
    #[arg(long, default_value_t = 5)]
    ascension: u8,
    /// Cap on harvested decision points.
    #[arg(long, default_value_t = 1500)]
    max_points: usize,
    /// Ground-truth probe budget — match the training labeler (K=12, sims=64).
    #[arg(long, default_value_t = 12)]
    k: u32,
    #[arg(long, default_value_t = 64)]
    sims: u32,
    #[arg(long, default_value_t = 0)]
    jobs: usize,
    #[arg(long, default_value = "data/runai/surrogate/eval_realistic.jsonl")]
    out: String,
}

fn parse_character(s: &str) -> Result<CharacterId, String> {
    match s.to_ascii_lowercase().as_str() {
        "ironclad" => Ok(CharacterId::Ironclad),
        "silent" => Ok(CharacterId::Silent),
        "defect" => Ok(CharacterId::Defect),
        "regent" => Ok(CharacterId::Regent),
        "necrobinder" => Ok(CharacterId::Necrobinder),
        other => Err(format!("unknown character '{other}'")),
    }
}

/// Same encounter universe (and index order) as `combat_surrogate_data`, so
/// the emitted `enc_idx` matches the surrogate's encounter one-hot.
fn encounter_universe() -> Vec<&'static EncounterDef> {
    let mut v = Vec::new();
    v.extend(ACT1_ENCOUNTERS.iter());
    v.extend(ACT2_ENCOUNTERS.iter());
    v.extend(ACT3_ENCOUNTERS.iter());
    v
}

#[derive(Serialize, Clone)]
struct Feats {
    deck: Vec<(u32, f32, f32)>,
    relics: Vec<(u32, f32)>,
    scalars: Vec<f32>,
}

/// Per-(candidate, encounter) single-fight ground truth (`encounter_delta`,
/// full fights — the surrogate's actual learning target).
#[derive(Serialize)]
struct EncTruth {
    enc_idx: u32,
    true_win: f64,
    true_hp_frac: f64,
    max_hp: i32,
}

#[derive(Serialize)]
struct CandRow {
    desc: String,
    is_baseline: bool,
    /// Deployment teacher: `fight_delta().score()` (truncated horizon, the
    /// score the current run-AI ranker actually uses — includes a damage
    /// tempo term the surrogate has no head for).
    gt_score_teacher: f64,
    /// Per-encounter full-fight truth (win / hp_frac) for this candidate.
    truth: Vec<EncTruth>,
    /// One context-feature vector (identical across the decision's encounters).
    feats: Feats,
}

#[derive(Serialize)]
struct EncRef {
    enc_idx: u32,
    enc_id: &'static str,
}

#[derive(Serialize)]
struct DecisionRow {
    seed: i32,
    step: u64,
    /// Encounters this decision is probed against (plan elite + boss).
    plan_enc: Vec<EncRef>,
    candidates: Vec<CandRow>,
}

fn enc_idx_of(universe: &[&'static EncounterDef], id: &str) -> Option<u32> {
    universe.iter().position(|e| e.id == id).map(|i| i as u32)
}

fn featurize(rm: &sts2_core::run_manager::RunManager, cand: &CandidateMutation) -> Feats {
    let mut probe = rm.clone();
    let _ = cand.apply(&mut probe);
    probe.combat = None;
    // Snap HP to the probe's bucket rep so features == the labeled state.
    let (hp, mx) = (probe.run_state.current_hp, probe.run_state.max_hp);
    let bucket = hp.max(1) / 5;
    probe.run_state.current_hp = (bucket * 5 + 2).clamp(1, mx.max(1));
    let view = VisibleRunView::from_run_manager(&probe);
    let ctx = ContextFeatures::from_view(&view);
    Feats {
        deck: ctx.deck,
        relics: ctx.relics,
        scalars: ctx.scalars,
    }
}

fn main() {
    let args = Args::parse();
    ensure_content_installed();
    if args.jobs > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.jobs)
            .build_global()
            .expect("pool");
    }

    let universe = encounter_universe();
    let budget = ProbeBudget {
        sims: args.sims,
        k: args.k,
        ..ProbeBudget::default()
    };

    // Harvest realistic card-shaped decisions from heuristic self-play.
    let seeds: Vec<i32> = (args.seed_start..args.seed_start + args.seeds)
        .map(|s| s as i32)
        .collect();
    eprintln!(
        "surrogate_eval_data: harvesting ≤{} decisions from {} self-play seeds (char={:?}, A{})",
        args.max_points, args.seeds, args.character, args.ascension
    );
    let points = collect_decision_points(
        &seeds,
        args.character,
        args.ascension,
        None,
        args.max_points,
    );
    eprintln!(
        "harvested {} decision points → labeling at K={} sims={}",
        points.len(),
        args.k,
        args.sims
    );

    let done = std::sync::atomic::AtomicUsize::new(0);
    let total = points.len();
    let started = std::time::Instant::now();

    let mut rows: Vec<DecisionRow> = points
        .par_iter()
        .map(|p| {
            // Plan encounters (elite + boss) resolved into the surrogate's universe.
            let mut plan_enc = Vec::new();
            let mut encs: Vec<&'static EncounterDef> = Vec::new();
            for e in [p.plan.elite, p.plan.boss].into_iter().flatten() {
                if let Some(idx) = enc_idx_of(&universe, e.id) {
                    plan_enc.push(EncRef {
                        enc_idx: idx,
                        enc_id: e.id,
                    });
                    encs.push(e);
                }
            }

            // Per candidate: deployment-teacher score (fight_delta), the
            // per-encounter full-fight truth (encounter_delta), and features.
            let cache = FightCache::default();
            let max_hp = p.rm.run_state.max_hp;
            let candidates: Vec<CandRow> = p
                .candidates
                .iter()
                .map(|c| {
                    let teacher = fight_delta(&p.rm, c, &p.plan, &budget, &cache).score();
                    let truth: Vec<EncTruth> = encs
                        .iter()
                        .zip(&plan_enc)
                        .map(|(e, eref)| {
                            let d = encounter_delta(&p.rm, c, e, &budget);
                            EncTruth {
                                enc_idx: eref.enc_idx,
                                true_win: d.win_rate,
                                true_hp_frac: if max_hp > 0 {
                                    d.hp_lost_mean / f64::from(max_hp)
                                } else {
                                    0.0
                                },
                                max_hp,
                            }
                        })
                        .collect();
                    CandRow {
                        desc: c.desc(),
                        is_baseline: matches!(c, CandidateMutation::Baseline),
                        gt_score_teacher: teacher,
                        truth,
                        feats: featurize(&p.rm, c),
                    }
                })
                .collect();

            let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if n.is_multiple_of(100) || n == total {
                eprintln!(
                    "  {n}/{total} decisions labeled ({:.0}s)",
                    started.elapsed().as_secs_f64()
                );
            }
            DecisionRow {
                seed: p.seed,
                step: p.step,
                plan_enc,
                candidates,
            }
        })
        .collect();

    rows.sort_by_key(|r| (r.seed, r.step));
    if let Some(parent) = std::path::Path::new(&args.out).parent() {
        std::fs::create_dir_all(parent).expect("out dir");
    }
    let f = std::fs::File::create(&args.out).expect("create out");
    let mut w = std::io::BufWriter::new(f);
    for r in &rows {
        serde_json::to_writer(&mut w, r).expect("write");
        w.write_all(b"\n").expect("nl");
    }
    w.flush().expect("flush");

    let multi = rows.iter().filter(|r| r.candidates.len() >= 2).count();
    let n_truth: usize = rows
        .iter()
        .flat_map(|r| r.candidates.iter())
        .map(|c| c.truth.len())
        .sum();
    eprintln!(
        "wrote {} decisions → {}\n  {} with ≥2 candidates (rankable), {} (candidate,enc) truth pairs",
        rows.len(), args.out, multi, n_truth
    );
}
