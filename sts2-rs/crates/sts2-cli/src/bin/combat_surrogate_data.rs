//! `combat_surrogate_data` — generate `(pre-combat state, encounter) →
//! combat outcome` training pairs for the learned **combat surrogate**.
//!
//! The surrogate's goal (see `docs/research/2026-06-03-outer-loop-ai-
//! literature.md`, "combat surrogate") is to map a pre-combat setup
//! (deck / HP / relics / potions / gold / floor / ascension) **plus the
//! encounter being entered** to that combat's outcome (win probability +
//! HP lost) in one forward pass — so the outer-loop run-AI can score
//! "how does this deck do in *this* fight" without paying for a combat
//! MCTS search.
//!
//! This binary produces the labeled data half. It does NOT learn anything
//! — it runs the real grounded combat probe ([`encounter_delta`], K paired
//! synthetic-seed MCTS fights) over a wide distribution of synthesized
//! pre-combat states × act-1/2/3 encounters, and writes one JSONL row per
//! labeled combat. The Python trainer (`sts2-pycli`) regresses on these.
//!
//! Each row carries the **sparse** context features the run-feature
//! encoder emits ([`ContextFeatures`], the same player-visible encoding the
//! deployed nets use) plus the encounter identity, and the probe's
//! `win_rate` / `hp_lost_mean` labels. A sidecar `*.schema.json` pins the
//! dense layout so the trainer can densify without re-deriving the index
//! tables.
//!
//! Determinism: every row is a pure function of `(base_seed, variant_idx)`
//! via a SplitMix64 stream, so a given `--seeds/--variants` config is
//! reproducible and resumable.

use std::io::Write as _;

use clap::Parser;
use rayon::prelude::*;
use serde::Serialize;

use sts2_content::cards::spec_index::specs_for_character;
use sts2_content::encounters::{
    act1_overgrowth::ACT1_ENCOUNTERS, act2_hive::ACT2_ENCOUNTERS, act3_glory::ACT3_ENCOUNTERS,
    EncounterDef, EncounterTier,
};
use sts2_content::run_callbacks::{
    initialize_run_setup, new_run_manager_for_character_with_ascension,
};
use sts2_core::cards::CardId;
use sts2_core::full_run::{drive_full_run, FullRunConfig, RunPolicy};
use sts2_core::potions::{CharacterId, PotionKind};
use sts2_core::run_manager::{Phase, RunAction, RunManager};
use sts2_core::RelicId;
use sts2_runai::arena::CombatBudgetKind;
use sts2_runai::eval::ranker_features::{
    self, context_dim, n_cards, n_relics, ContextFeatures, SCALAR_NAMES,
};
use sts2_runai::eval::Weights;
use sts2_runai::policy::{CombatBudget, HeuristicPolicy, MctsCombatPolicy};
use sts2_runai::ppo::policy::PpoPolicy;
use sts2_runai::rollout::{
    encounter_delta, ensure_content_installed, CandidateMutation, ProbeBudget,
};
use sts2_runai::view::VisibleRunView;

/// Generate combat-surrogate training pairs (one row per labeled combat).
#[derive(Parser, Debug)]
#[command(
    name = "combat_surrogate_data",
    about = "Label (pre-combat state, encounter) → outcome pairs for the learned combat surrogate."
)]
struct Args {
    /// Number of base run seeds. Total rows ≈ seeds × variants.
    #[arg(long, default_value_t = 400)]
    seeds: u32,

    /// First base seed of the range.
    #[arg(long, default_value_t = 1)]
    seed_start: u32,

    /// Synthesized pre-combat states (each vs one sampled encounter) per seed.
    #[arg(long, default_value_t = 12)]
    variants: u32,

    /// Character whose card pool / starter deck seeds the states.
    #[arg(long, default_value = "ironclad", value_parser = parse_character)]
    character: CharacterId,

    /// Paired synthetic seeds per encounter (probe K). Fewer = faster, noisier label.
    #[arg(long, default_value_t = 12)]
    k: u32,

    /// MCTS simulations per probe move.
    #[arg(long, default_value_t = 64)]
    sims: u32,

    /// Rayon worker threads (0 = rayon default = all cores).
    #[arg(long, default_value_t = 0)]
    jobs: usize,

    /// Output JSONL path (a sibling `<out>.schema.json` is also written).
    #[arg(long, default_value = "data/runai/surrogate/combat_labels.jsonl")]
    out: String,

    /// Realistic harvest mode: label REAL pre-combat states drawn from
    /// heuristic self-play instead of synthesizing states. `--seeds`
    /// becomes the number of driven runs; `--variants` and the
    /// deck/relic/HP synthesis knobs are ignored.
    #[arg(long, default_value_t = false)]
    realistic: bool,

    /// Row cap for `--realistic` (each entered combat emits one row,
    /// across all driven seeds, until this many rows are collected).
    #[arg(long, default_value_t = 6000)]
    max_rows: usize,

    /// Ascension levels to draw from, comma-separated (implemented A0-A10).
    /// Applies to BOTH paths: synthetic samples one per (seed, variant),
    /// realistic assigns one per driven run.
    ///
    /// The default is the full spread, NOT A0. `--realistic` used to be
    /// hardcoded to ascension 0, and the resulting labels were saturated
    /// (measured 2026-08-12: mean win_rate 0.956, 17043/18603 rows at exactly
    /// 1.0) — the same confound that made the M1 surrogate eval
    /// uninterpretable. An A0-only harvest is the trap, so it is opt-in.
    #[arg(
        long,
        default_value = "0,2,5,8,10",
        value_delimiter = ',',
        value_parser = parse_ascension_level,
        num_args = 1..,
    )]
    ascensions: Vec<u8>,

    /// Self-play driver for `--realistic`. `heuristic` is cheap but dies in
    /// Act 1 (98.3% of its rows, measured 2026-08-13) and is off-policy;
    /// `ppo` drives the champion net with searched combat and reaches acts
    /// 2-3, at roughly the cost of a full run per seed.
    #[arg(long, default_value = "heuristic", value_parser = parse_driver)]
    driver: DriverKind,

    /// Combat budget for `--driver ppo`'s in-run combat search.
    #[arg(long, default_value = "eval-cheap", value_parser = parse_combat_budget)]
    driver_combat: CombatBudgetKind,
}

/// Which policy drives the `--realistic` self-play runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DriverKind {
    Heuristic,
    Ppo,
}

fn parse_driver(s: &str) -> Result<DriverKind, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "heuristic" => Ok(DriverKind::Heuristic),
        "ppo" => Ok(DriverKind::Ppo),
        other => Err(format!("unknown driver '{other}' (heuristic|ppo)")),
    }
}

/// Reuses the arena's parser so the harvest cannot drift from the frozen
/// budget names the rest of the stack quotes its win rates against.
fn parse_combat_budget(s: &str) -> Result<CombatBudgetKind, String> {
    CombatBudgetKind::parse(s.trim())
}

/// The ascension spread both harvest paths draw from. The two previously
/// disagreed silently — synthetic sampled `[0,2,5,8,10]` while realistic was
/// pinned to `0` — so this is the test's pin on the `--ascensions` clap
/// default, checked by `default_ascensions_matches_const`.
#[cfg(test)]
const DEFAULT_ASCENSIONS: [u8; 5] = [0, 2, 5, 8, 10];

/// Stable potion index: every [`PotionKind`] sorted by Debug name, matching
/// how `ranker_features` builds its card and relic tables.
///
/// Potion IDENTITY was invisible to the surrogate until 2026-08-14: the
/// harvester randomizes `PotionKind::ALL` into every synthetic state and the
/// features emitted only `potions.len()` and `max_potion_slots`, so a Fire
/// Potion and a Block Potion were literally the same input. That is an
/// unmodellable cause of outcome variance — a ceiling on the conditional mean
/// no architecture can lift, which is the diagnosed residual after the
/// distributional head fell short.
fn potion_index_table() -> &'static [PotionKind] {
    static TABLE: std::sync::OnceLock<Vec<PotionKind>> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let mut v = PotionKind::ALL.to_vec();
        v.sort_by_key(|p| format!("{p:?}"));
        v
    })
}

fn potion_index_lookup() -> &'static std::collections::BTreeMap<String, u32> {
    static MAP: std::sync::OnceLock<std::collections::BTreeMap<String, u32>> =
        std::sync::OnceLock::new();
    MAP.get_or_init(|| {
        potion_index_table()
            .iter()
            .enumerate()
            .map(|(i, p)| (format!("{p:?}"), i as u32))
            .collect()
    })
}

/// Potion identities held at combat entry, as indices into
/// [`potion_index_table`]. Order-insensitive by construction (sorted), since
/// a potion belt is a set, not a sequence.
fn potion_indices(view: &VisibleRunView) -> Vec<u32> {
    let lut = potion_index_lookup();
    let mut v: Vec<u32> = view
        .potions
        .iter()
        .filter_map(|p| lut.get(&format!("{p:?}")).copied())
        .collect();
    v.sort_unstable();
    v
}

/// One `--ascensions` element. Range-checked here rather than at use, so a
/// typo fails at parse instead of silently harvesting an unimplemented
/// difficulty.
fn parse_ascension_level(s: &str) -> Result<u8, String> {
    let a: u8 = s
        .trim()
        .parse()
        .map_err(|_| format!("bad ascension '{s}' (want an integer 0-10)"))?;
    if a > 10 {
        return Err(format!(
            "ascension {a} exceeds the implemented A0-A10 range"
        ));
    }
    Ok(a)
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

/// SplitMix64 — a tiny, dependency-free deterministic stream so each row is
/// a pure function of its `(seed, variant)` derivation.
struct SplitMix64(u64);
impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in `[0, n)`.
    fn below(&mut self, n: u64) -> u64 {
        if n == 0 {
            0
        } else {
            self.next_u64() % n
        }
    }
    /// Uniform in `[0, 1)`.
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[self.below(xs.len() as u64) as usize]
    }
}

/// One labeled combat (the JSONL row).
#[derive(Serialize)]
struct Row {
    seed: u32,
    variant: u32,
    // ── context (sparse, densify with the schema) ──
    /// `(card_idx, count, upgraded_count)`, ascending.
    deck: Vec<(u32, f32, f32)>,
    /// `(relic_idx, count)`, ascending.
    relics: Vec<(u32, f32)>,
    /// Potion IDENTITIES held at combat entry, as `potion_index_table` indices.
    /// Ascending. Previously the state carried potions the features could not
    /// see — only a count — so their effect on the outcome was pure label noise.
    potions: Vec<u32>,
    /// Dense scalar block (`SCALAR_NAMES` + boss one-hot tail).
    scalars: Vec<f32>,
    // ── encounter being entered ──
    enc_idx: u32,
    enc_id: &'static str,
    enc_act: u8,
    enc_tier: &'static str,
    // ── labels (the regression targets) ──
    win_rate: f64,
    hp_lost_mean: f64,
    /// HP lost as a fraction of max HP — the scale-free regression target.
    hp_frac_lost: f64,
    /// Spread of `hp_frac_lost` ACROSS the K probes, same scale-free units.
    ///
    /// Without this the surrogate can only learn a conditional mean, and a
    /// mean-trained surrogate models a world with no catastrophes: measured
    /// 2026-08-13 it produced 0.68 run survival against a 0.29 reference and
    /// only 0.063 of fights losing >50% HP against an actual 0.101. Death is a
    /// tail event, so the tail has to survive the harvest.
    hp_frac_lost_std: f64,
    losses: u32,
    probes: u32,
    terminals: u32,
    /// State scalars carried out for convenience / sanity checks.
    max_hp: i32,
    entry_hp: i32,
    deck_size: usize,
    /// `"synthetic"` (deck/relic/HP/potion combinatorics sampled from a
    /// `SplitMix64` stream) or `"realistic"` (a REAL pre-combat state
    /// snapshotted from a driven self-play run — see
    /// [`harvest_realistic`]). AZ Milestone 1 Task 2.
    source: &'static str,
}

/// The encounter universe we label against: every act-1/2/3 encounter,
/// tagged with its act. `act_index` is set to match before labeling so
/// ascension scaling and the act feature stay coherent with the fight.
fn encounter_universe() -> Vec<(u8, &'static EncounterDef)> {
    let mut v = Vec::new();
    for e in ACT1_ENCOUNTERS {
        v.push((1u8, e));
    }
    for e in ACT2_ENCOUNTERS {
        v.push((2u8, e));
    }
    for e in ACT3_ENCOUNTERS {
        v.push((3u8, e));
    }
    v
}

fn tier_name(t: EncounterTier) -> &'static str {
    match t {
        EncounterTier::Weak => "weak",
        EncounterTier::Normal => "normal",
        EncounterTier::Elite => "elite",
        EncounterTier::Boss => "boss",
        _ => "other",
    }
}

/// Mirror `rollout::bucket_and_rep_hp` so the encoded features describe the
/// exact HP the label is computed at (the probe snaps HP to the bucket rep).
fn bucket_rep_hp(hp: i32, max_hp: i32) -> i32 {
    const W: i32 = 5; // rollout::HP_BUCKET_WIDTH
    let bucket = hp.max(1) / W;
    (bucket * W + W / 2).clamp(1, max_hp.max(1))
}

/// Floor a probe's `hp_lost_mean` at 0. `DeltaStats::hp_lost_mean` is a raw
/// `start_hp − exit_hp` average (`rollout::probe_one`'s `hp_lost` field,
/// unclamped there) — a fight with net mid-combat healing (lifesteal /
/// regen cards or relics outpacing incoming damage) can leave it slightly
/// negative, which doesn't fit the row's own "HP lost" semantics or the
/// `hp_frac_lost ∈ [0, 1]` scale-free-target contract the trainer relies
/// on. Mirrors the existing `hp_lost.max(0)` floor the chained-attrition
/// rollout already applies per-link (`rollout.rs`); this binary's two
/// label sites (synthetic [`make_row`], realistic
/// [`snapshot_realistic_row`]) apply the same floor when they aggregate
/// `DeltaStats` into a `Row`.
fn floor_hp_lost_mean(hp_lost_mean: f64) -> f64 {
    hp_lost_mean.max(0.0)
}

/// Synthesize + label one `(state, encounter)` pair from `(seed, variant)`.
fn make_row(
    seed: u32,
    variant: u32,
    character: CharacterId,
    card_pool: &[CardId],
    encounters: &[(u8, &'static EncounterDef)],
    budget: &ProbeBudget,
    ascensions: &[u8],
) -> Row {
    let mut rng = SplitMix64::new(
        u64::from(seed)
            .wrapping_mul(0x1000_0000_0000_0001)
            .wrapping_add(u64::from(variant).wrapping_mul(0x9E37_79B9_7F4A_7C15))
            ^ 0xC0FF_EE00_1234_5678,
    );

    // Ascension (implemented A0–A10) — widens the difficulty distribution.
    let ascension = *rng.pick(ascensions);
    let mut rm = new_run_manager_for_character_with_ascension(seed as i32, character, ascension)
        .with_neow_start();
    initialize_run_setup(&mut rm);

    // ── Diversify the deck: append 0..=22 class cards (≈30% upgraded). ──
    let n_add = rng.below(23);
    for _ in 0..n_add {
        let card_id = *rng.pick(card_pool);
        let upgraded = rng.unit() < 0.30;
        let _ = CandidateMutation::AddCard { card_id, upgraded }.apply(&mut rm);
    }
    // ── A few extra in-deck upgrades on existing cards. ──
    let n_upg = rng.below(6);
    for _ in 0..n_upg {
        let len = rm.run_state.master_deck.len();
        if len > 0 {
            let idx = rng.below(len as u64) as usize;
            let _ = CandidateMutation::UpgradeCard { deck_idx: idx }.apply(&mut rm);
        }
    }
    // ── Relics: 0..=6 distinct (skip the ones the starter already grants). ──
    let n_relics = rng.below(7);
    for _ in 0..n_relics {
        let relic = *rng.pick(RelicId::all());
        if !rm.run_state.relics.contains(&relic) {
            let _ = CandidateMutation::AddRelic { relic_id: relic }.apply(&mut rm);
        }
    }
    // ── Potions: fill 0..=min(slots, 5). ──
    let slot_cap = rm.run_state.max_potion_slots.clamp(0, 5) as u64;
    let n_potions = rng.below(slot_cap + 1);
    for _ in 0..n_potions {
        let kind = *rng.pick(&PotionKind::ALL);
        let _ = CandidateMutation::AddPotion { kind }.apply(&mut rm);
    }
    // ── Gold (combat-inert, but a feature the surrogate sees). ──
    rm.run_state.gold = rng.below(601) as i32;

    // ── Pick the encounter and align the state's act to it. ──
    let enc_idx = rng.below(encounters.len() as u64) as usize;
    let (enc_act, enc) = encounters[enc_idx];
    rm.run_state.act_index = enc_act;

    // ── HP: wide fraction sweep (0.12..1.0) → loss signal on hard fights. ──
    let max_hp = rm.run_state.max_hp;
    let frac = 0.12 + rng.unit() * 0.88;
    let raw_hp = ((frac * f64::from(max_hp)).round() as i32).clamp(1, max_hp.max(1));
    // Snap to the probe's HP bucket rep so features == the labeled state.
    let entry_hp = bucket_rep_hp(raw_hp, max_hp);
    rm.run_state.current_hp = entry_hp;

    // ── Encode features (player-visible context). ──
    let view = VisibleRunView::from_run_manager(&rm);
    let ctx = ContextFeatures::from_view(&view);

    // ── Label: real grounded combat probe of THIS encounter. ──
    let stats = encounter_delta(&rm, &CandidateMutation::Baseline, enc, budget);
    let hp_lost_mean = floor_hp_lost_mean(stats.hp_lost_mean);
    let hp_frac_lost = if max_hp > 0 {
        hp_lost_mean / f64::from(max_hp)
    } else {
        0.0
    };
    // NOT floored the way the mean is: a spread is already non-negative, and
    // `floor_hp_lost_mean` exists to clip in-combat HEALING out of the mean,
    // which has no analogue for a standard deviation.
    let hp_frac_lost_std = if max_hp > 0 {
        stats.hp_lost_std / f64::from(max_hp)
    } else {
        0.0
    };

    Row {
        seed,
        variant,
        deck: ctx.deck,
        relics: ctx.relics,
        potions: potion_indices(&view),
        scalars: ctx.scalars,
        enc_idx: enc_idx as u32,
        enc_id: enc.id,
        enc_act,
        enc_tier: tier_name(enc.tier),
        win_rate: stats.win_rate,
        hp_lost_mean,
        hp_frac_lost,
        hp_frac_lost_std,
        losses: stats.losses,
        probes: stats.probes,
        terminals: stats.terminals,
        max_hp,
        entry_hp,
        deck_size: rm.run_state.master_deck.len(),
        source: "synthetic",
    }
}

/// Total size of the encounter-index space this binary labels against —
/// the concatenated Act 1/2/3 [`EncounterDef`] tables in
/// [`encounter_universe`]'s order (Act 1, then Act 2, then Act 3). This is
/// the `n_encounters` the surrogate's encounter embedding (`enc_emb`) is
/// sized to; it matches `sts2_content::encounters::n_encounters()` by
/// construction (both walk the same three static tables in the same
/// order) but is computed locally so this binary's row/schema `enc_idx`
/// values are self-consistent even if that invariant ever drifted.
fn n_encounter_index_space() -> usize {
    ACT1_ENCOUNTERS.len() + ACT2_ENCOUNTERS.len() + ACT3_ENCOUNTERS.len()
}

/// Snapshot + label ONE real pre-combat state: `rm` is a live, in-progress
/// `RunManager` that a driven self-play run has just entered `Phase::Combat`
/// on. Reads `rm.run_state.current_encounter_idx` (set by
/// `sts2_content::run_callbacks::build_combat_for_room` when it resolves
/// the `EncounterDef` for this fight) to identify which encounter is being
/// entered, then runs the same real grounded combat probe
/// ([`encounter_delta`], `CandidateMutation::Baseline`) the synthetic path
/// uses. Returns `None` when the index is missing (never set — a
/// pre-EncounterDef-path combat, i.e. the legacy `enter_combat` fallback)
/// or out of range (defensive; should not happen for the Act 1–3 tables
/// this binary labels against).
///
/// `rm` is never mutated: feature encoding runs on a `clone()` with HP
/// snapped to its bucket rep (matching what `encounter_delta` snaps to
/// internally on its OWN clone), so this is a pure observer of the run it
/// rides along on.
fn snapshot_realistic_row(
    rm: &RunManager,
    encounters: &[(u8, &'static EncounterDef)],
    budget: &ProbeBudget,
    variant: u32,
) -> Option<Row> {
    let idx = rm.run_state.current_encounter_idx?;
    let (enc_act, enc) = *encounters.get(idx)?;

    let max_hp = rm.run_state.max_hp;
    // Snap to the same HP bucket `encounter_delta` snaps its own internal
    // probe clone to, so the features encoded below describe the exact
    // state the label is computed at (mirrors the synthetic path's
    // `bucket_rep_hp` use in `make_row`).
    let entry_hp = bucket_rep_hp(rm.run_state.current_hp, max_hp);

    // Encode features off a snapped-HP clone so they describe the exact
    // state `encounter_delta` (which snaps its own internal clone the
    // same way) computes the label at — never perturbs the live `rm`.
    let mut feat_rm = rm.clone();
    feat_rm.run_state.current_hp = entry_hp;
    let view = VisibleRunView::from_run_manager(&feat_rm);
    let ctx = ContextFeatures::from_view(&view);

    let stats = encounter_delta(rm, &CandidateMutation::Baseline, enc, budget);
    let hp_lost_mean = floor_hp_lost_mean(stats.hp_lost_mean);
    let hp_frac_lost = if max_hp > 0 {
        hp_lost_mean / f64::from(max_hp)
    } else {
        0.0
    };
    // NOT floored the way the mean is: a spread is already non-negative, and
    // `floor_hp_lost_mean` exists to clip in-combat HEALING out of the mean,
    // which has no analogue for a standard deviation.
    let hp_frac_lost_std = if max_hp > 0 {
        stats.hp_lost_std / f64::from(max_hp)
    } else {
        0.0
    };

    Some(Row {
        // `RunManager::seed` is the per-run master seed; `variant` doubles
        // as the "which combat entry within this run" counter (realistic
        // rows have no deck/relic/HP variant axis).
        seed: rm.seed as u32,
        variant,
        deck: ctx.deck,
        relics: ctx.relics,
        potions: potion_indices(&view),
        scalars: ctx.scalars,
        enc_idx: idx as u32,
        enc_id: enc.id,
        enc_act,
        enc_tier: tier_name(enc.tier),
        win_rate: stats.win_rate,
        hp_lost_mean,
        hp_frac_lost,
        hp_frac_lost_std,
        losses: stats.losses,
        probes: stats.probes,
        terminals: stats.terminals,
        max_hp,
        entry_hp,
        deck_size: rm.run_state.master_deck.len(),
        source: "realistic",
    })
}

/// `RunPolicy` decorator that snapshots + labels the pre-combat state at
/// each fresh entry into [`Phase::Combat`] and otherwise delegates every
/// decision — including in-combat card play — unchanged to `inner`. A pure
/// observer: it never overrides a decision, so the run it rides along on
/// plays out exactly as `inner` alone would have. Once `max_rows` rows are
/// collected it returns `None` from every subsequent `choose`, which
/// `drive_full_run` treats as a clean [`sts2_core::full_run::RunOutcome::PolicyHalt`]
/// — ending that seed's run early instead of spending more compute once the
/// cap for this harvest call is reached.
struct HarvestPolicy<'a> {
    inner: Box<dyn RunPolicy>,
    encounters: &'a [(u8, &'static EncounterDef)],
    budget: &'a ProbeBudget,
    max_rows: usize,
    rows: Vec<Row>,
    combats_seen: u32,
    was_combat: bool,
}

impl<'a> HarvestPolicy<'a> {
    fn new(
        inner: Box<dyn RunPolicy>,
        encounters: &'a [(u8, &'static EncounterDef)],
        budget: &'a ProbeBudget,
        max_rows: usize,
    ) -> Self {
        Self {
            inner,
            encounters,
            budget,
            max_rows,
            rows: Vec::new(),
            combats_seen: 0,
            was_combat: false,
        }
    }
}

impl RunPolicy for HarvestPolicy<'_> {
    fn choose(&mut self, rm: &RunManager, actions: &[RunAction]) -> Option<usize> {
        if self.rows.len() >= self.max_rows {
            return None; // cap hit — halt this seed's run early.
        }
        let is_combat = rm.phase() == Phase::Combat;
        if is_combat && !self.was_combat {
            // Fresh combat entry (the previous `choose` saw a non-Combat
            // phase). Snapshot before any in-combat action is taken.
            if let Some(row) =
                snapshot_realistic_row(rm, self.encounters, self.budget, self.combats_seen)
            {
                self.rows.push(row);
            }
            self.combats_seen += 1;
        }
        self.was_combat = is_combat;
        self.inner.choose(rm, actions)
    }
}

/// Build the self-play driver for `--realistic`.
///
/// [`DriverKind::Heuristic`] is the cheap default-weights heuristic with NO
/// combat search. It is fast, and it is why this path was Act-1-only:
/// measured 2026-08-13, a heuristic harvest put **98.3% of its rows in act 1**
/// (302 rows in act 2, SIX in act 3) at a mean of 6.2 fights per run, because
/// the driver dies in mid-Act-1. Any surrogate benchmarked on those rows is
/// being scored on Act 1 against a policy that is no longer deployed.
///
/// [`DriverKind::Ppo`] is the deployment-shaped driver this module's comment
/// has promised since the switch to PPO: the champion net for non-combat
/// decisions, wrapped in [`MctsCombatPolicy`] so combat is actually searched
/// (`PpoPolicy` never plays combat itself — see `arena.rs`, which hard-errors
/// on `--policy ppo` without a combat budget). It reaches acts 2-3, and it
/// costs a full search-driven run per seed.
fn build_realistic_driver_policy(
    run_seed: i32,
    driver: DriverKind,
    character: CharacterId,
    combat: Option<CombatBudget>,
) -> Box<dyn RunPolicy> {
    match driver {
        DriverKind::Heuristic => Box::new(HeuristicPolicy::with_weights(Weights::default())),
        DriverKind::Ppo => {
            let budget = combat.expect("--driver ppo requires a combat budget (checked in main)");
            Box::new(MctsCombatPolicy::new(
                PpoPolicy::from_env_for(character),
                budget,
                run_seed,
            ))
        }
    }
}

/// Drive `seeds` full self-play runs (character fixed to Ironclad,
/// this binary's existing synthetic-path default) with
/// [`build_realistic_driver_policy`], snapshotting + labeling the
/// pre-combat state at every entered combat via [`HarvestPolicy`], until
/// `max_rows` rows are collected or the seeds are exhausted.
///
/// Kept probe/drive budgets small are the CALLER's job (`main` uses the
/// CLI's `--k`/`--sims`; the smoke test uses a tiny fixed budget) — this
/// function itself just wires the loop.
///
/// Test-only: mirrors the literal `harvest_realistic(seeds, max_rows)`
/// signature from the task brief's Step 1 smoke test with a tiny built-in
/// probe budget. `main`'s `--realistic` path calls
/// [`harvest_realistic_range`] directly with the CLI's `--k`/`--sims`.
#[cfg(test)]
fn harvest_realistic(seeds: u32, max_rows: usize) -> Vec<Row> {
    harvest_realistic_range(
        1,
        seeds,
        max_rows,
        &ProbeBudget {
            sims: 8,
            k: 2,
            t_turns: 4,
            ..ProbeBudget::default()
        },
        CharacterId::Ironclad,
        &DEFAULT_ASCENSIONS,
        DriverKind::Heuristic,
        None,
    )
}

/// Core `--realistic` harvest loop, parameterized over the starting seed
/// (so `main`'s `--seed-start` applies to `--realistic` the same way it
/// does to the synthetic path), the seed count, the row cap, the probe
/// budget, the character and the ascension spread.
///
/// `character` and `ascensions` used to be hardcoded to
/// `(Ironclad, 0)` here, which made `--character` silently a no-op in this
/// mode and pinned every realistic row to the easiest difficulty. The latter
/// is why realistic labels saturate — see `--ascensions`.
///
/// Ascension is assigned per driven run (not per row): every combat within
/// one run shares the run's difficulty, which is what makes a row's
/// `(encounter, ascension)` cell meaningful.
// Eight independent CLI knobs, each already named at the one call site.
// Bundling them into a struct would move the same eight names one level away
// without removing any of them.
#[allow(clippy::too_many_arguments)]
fn harvest_realistic_range(
    seed_start: i32,
    seeds: u32,
    max_rows: usize,
    budget: &ProbeBudget,
    character: CharacterId,
    ascensions: &[u8],
    driver: DriverKind,
    driver_combat: Option<CombatBudget>,
) -> Vec<Row> {
    let encounters = encounter_universe();
    let cfg = FullRunConfig {
        max_steps: 20_000,
        ..FullRunConfig::default()
    };

    let mut rows: Vec<Row> = Vec::new();
    let mut seeds_driven = 0u32;
    for seed in seed_start..seed_start + i32::try_from(seeds).unwrap_or(i32::MAX) {
        if rows.len() >= max_rows {
            break;
        }
        // Cycle rather than sample: with a handful of levels and a few
        // thousand runs this guarantees an even spread across cells, which a
        // per-seed random draw would only approach.
        let ascension = ascensions[(seeds_driven as usize) % ascensions.len()];
        let mut rm = new_run_manager_for_character_with_ascension(seed, character, ascension)
            .with_neow_start();
        initialize_run_setup(&mut rm);

        let inner = build_realistic_driver_policy(seed, driver, character, driver_combat);
        let mut harvest = HarvestPolicy::new(inner, &encounters, budget, max_rows - rows.len());
        let _report = drive_full_run(&mut rm, &mut harvest, &cfg);
        let gained = harvest.rows.len();
        rows.extend(harvest.rows);
        seeds_driven += 1;
        if seeds_driven.is_multiple_of(20) || rows.len() >= max_rows {
            eprintln!(
                "  realistic: {seeds_driven} seeds driven, seed {seed} +{gained} rows \
                 ({}/{max_rows} total)",
                rows.len()
            );
        }
    }
    rows
}

#[derive(Serialize)]
struct EncMeta {
    idx: u32,
    id: &'static str,
    act: u8,
    tier: &'static str,
}

#[derive(Serialize)]
#[allow(clippy::struct_field_names)] // `schema_id` naming the JSON key wins over the lint here.
struct Schema {
    /// The feature-schema id the encounter/context index space is pinned
    /// to ([`ranker_features::feature_schema_id`]) — copied verbatim into
    /// the surrogate-net weights JSON by the Python trainer so
    /// [`SurrogateNet::from_json_slice`](sts2_runai::eval::surrogate::SurrogateNet::from_json_slice)
    /// can refuse a net trained against a stale index space (AZ M1 Task 3).
    schema_id: &'static str,
    /// Content-addressed source hash of the engine that produced these
    /// labels ([`sts2_runai::label::engine_identity`]) — informational
    /// provenance, copied verbatim into the net JSON's `engine` field.
    engine: &'static str,
    context_dim: usize,
    n_cards: usize,
    n_relics: usize,
    scalar_names: Vec<&'static str>,
    n_scalars: usize,
    /// Size of the encounter-index space (`encounters.len()`) — the
    /// dimension the surrogate's `enc_emb` encounter embedding is sized
    /// to. Same value as `sts2_content::encounters::n_encounters()`.
    n_encounters: usize,
    /// Index → `(id, act, tier)` table, ordered Act 1 / 2 / 3 concatenated
    /// (`EncMeta::idx` is the stable index a `Row::enc_idx` refers to).
    encounters: Vec<EncMeta>,
    /// Index → id-name tables for the card / relic / potion spaces.
    ///
    /// These exist so Python can TRANSFER the champion PPO net's pretrained
    /// embeddings into the surrogate. The two index spaces are unrelated —
    /// `ranker_features` sorts by Rust Debug name, `ppo/vocab.py` by the
    /// registry's canonical name — so without a name table the only options
    /// are re-learning 580 card embeddings from a few thousand rows or a
    /// silently-misaligned copy. Names are `to_python_name()` output, which
    /// `vocab.canonical()` normalizes on the Python side.
    card_ids: Vec<&'static str>,
    relic_ids: Vec<&'static str>,
    /// Debug names — `PotionKind` has no `to_python_name`; `vocab.canonical()`
    /// turns PascalCase into the registry's UPPER_SNAKE just as well.
    potion_ids: Vec<String>,
    n_potions: usize,
    /// Dense layout the trainer must reproduce.
    layout: &'static str,
}

fn main() {
    let args = Args::parse();
    ensure_content_installed();

    if args.jobs > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.jobs)
            .build_global()
            .expect("rayon pool");
    }

    let card_pool: Vec<CardId> = {
        let mut v: Vec<CardId> = specs_for_character(args.character)
            .iter()
            .map(|f| f().id)
            .collect();
        v.sort_by_key(|c| format!("{c:?}"));
        v.dedup();
        v
    };
    let encounters = encounter_universe();
    // The row/schema encounter-index space must agree with the CLI's own
    // enumeration order — checked once, loudly, rather than trusting the
    // two implementations to never drift apart.
    assert_eq!(
        encounters.len(),
        n_encounter_index_space(),
        "encounter_universe() / n_encounter_index_space() disagree on index-space size"
    );
    let budget = ProbeBudget {
        sims: args.sims,
        k: args.k,
        ..ProbeBudget::default()
    };

    // Write the schema sidecar (dense layout + encounter table).
    let schema = Schema {
        schema_id: ranker_features::feature_schema_id(),
        engine: sts2_runai::label::engine_identity(),
        context_dim: context_dim(),
        n_cards: n_cards(),
        n_relics: n_relics(),
        scalar_names: SCALAR_NAMES.to_vec(),
        n_scalars: SCALAR_NAMES.len(),
        n_encounters: encounters.len(),
        encounters: encounters
            .iter()
            .enumerate()
            .map(|(i, (act, e))| EncMeta {
                idx: i as u32,
                id: e.id,
                act: *act,
                tier: tier_name(e.tier),
            })
            .collect(),
        card_ids: ranker_features::card_index_table()
            .iter()
            .map(|c| c.to_python_name())
            .collect(),
        relic_ids: ranker_features::relic_index_table()
            .iter()
            .map(|r| r.to_python_name())
            .collect(),
        potion_ids: potion_index_table()
            .iter()
            .map(|p| format!("{p:?}"))
            .collect(),
        n_potions: potion_index_table().len(),
        layout: "x = [deck(count,upg) per card_idx (2*n_cards)] ++ \
                 [relic count per relic_idx (n_relics)] ++ [scalars] ++ \
                 [encounter one-hot (n_encounters)]",
    };
    if let Some(parent) = std::path::Path::new(&args.out).parent() {
        std::fs::create_dir_all(parent).expect("create out dir");
    }
    let schema_path = format!("{}.schema.json", args.out);
    std::fs::write(
        &schema_path,
        serde_json::to_string_pretty(&schema).expect("schema json"),
    )
    .expect("write schema");

    let mut rows: Vec<Row> = if args.realistic {
        // `--driver ppo` needs a real combat budget: `PpoPolicy` never plays
        // combat itself, so an unwrapped net would be driving with no fighter.
        // Resolved here rather than inside the loop so a bad combination fails
        // before any run is driven.
        let driver_combat = match args.driver {
            DriverKind::Heuristic => None,
            DriverKind::Ppo => Some(args.driver_combat.budget(None, None).unwrap_or_else(|| {
                panic!(
                    "--driver ppo requires --driver-combat eval-cheap|eval-strong \
                     (got 'none'; the PPO net never plays combat itself)"
                )
            })),
        };
        eprintln!(
            "combat_surrogate_data --realistic: {} seeds (seed_start={}), max_rows={}, \
             K={} sims={}, char={:?}, ascensions={:?}, driver={:?}{}, \
             {} encounters → {}",
            args.seeds,
            args.seed_start,
            args.max_rows,
            args.k,
            args.sims,
            args.character,
            args.ascensions,
            args.driver,
            match driver_combat {
                Some(b) => format!("(combat {}/{} sims)", b.sims, b.boss_sims),
                None => "(no combat search)".to_string(),
            },
            encounters.len(),
            args.out
        );
        if args.ascensions == [0] {
            eprintln!(
                "  WARNING: ascension 0 only — realistic labels saturate at A0 \
                 (measured: mean win_rate 0.956, 92% of rows at exactly 1.0), which \
                 leaves almost nothing for a deck-sensitivity gate to rank."
            );
        }
        if args.driver == DriverKind::Heuristic {
            eprintln!(
                "  WARNING: --driver heuristic is OFF-POLICY and Act-1-only — measured \
                 2026-08-13, 98.3% of its rows land in act 1 (302 in act 2, SIX in act 3) \
                 at a mean of 6.2 fights per run. Use --driver ppo for a corpus that \
                 covers acts 2-3."
            );
        }
        // Realistic harvest drives full runs sequentially (each run's
        // trajectory depends on every prior decision, so there is no
        // per-cell parallelism to fan out over the way the synthetic path
        // has); `--jobs` still applies to any MCTS the driver invokes.
        let rows = harvest_realistic_range(
            i32::try_from(args.seed_start).unwrap_or_else(|_| {
                panic!(
                    "--seed-start {} overflows i32 (max {}); pick a smaller --seed-start \
                     rather than silently falling back to seed 1",
                    args.seed_start,
                    i32::MAX
                )
            }),
            args.seeds,
            args.max_rows,
            &budget,
            args.character,
            &args.ascensions,
            args.driver,
            driver_combat,
        );
        eprintln!("  realistic harvest done: {} rows", rows.len());
        rows
    } else {
        // Fan out over (seed, variant) cells.
        let cells: Vec<(u32, u32)> = (args.seed_start..args.seed_start + args.seeds)
            .flat_map(|s| (0..args.variants).map(move |v| (s, v)))
            .collect();
        let total = cells.len();
        eprintln!(
            "combat_surrogate_data: {total} cells ({} seeds × {} variants), \
             K={} sims={} char={:?}, {} encounters → {}",
            args.seeds,
            args.variants,
            args.k,
            args.sims,
            args.character,
            encounters.len(),
            args.out
        );

        let done = std::sync::atomic::AtomicUsize::new(0);
        let started = std::time::Instant::now();
        cells
            .par_iter()
            .map(|&(seed, variant)| {
                let row = make_row(
                    seed,
                    variant,
                    args.character,
                    &card_pool,
                    &encounters,
                    &budget,
                    &args.ascensions,
                );
                let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                if n.is_multiple_of(200) || n == total {
                    let elapsed = started.elapsed().as_secs_f64();
                    eprintln!(
                        "  {n}/{total} labeled ({:.1}/s, {:.0}s elapsed)",
                        n as f64 / elapsed.max(1e-9),
                        elapsed
                    );
                }
                row
            })
            .collect()
    };

    // Deterministic row order independent of thread scheduling.
    rows.sort_by_key(|r| (r.seed, r.variant));

    let f = std::fs::File::create(&args.out).expect("create out file");
    let mut w = std::io::BufWriter::new(f);
    for row in &rows {
        serde_json::to_writer(&mut w, row).expect("write row");
        w.write_all(b"\n").expect("newline");
    }
    w.flush().expect("flush");

    // Quick label-distribution summary (sanity at a glance).
    let n = rows.len().max(1) as f64;
    let mean_win = rows.iter().map(|r| r.win_rate).sum::<f64>() / n;
    let mean_hpfrac = rows.iter().map(|r| r.hp_frac_lost).sum::<f64>() / n;
    let any_loss = rows.iter().filter(|r| r.losses > 0).count();
    eprintln!(
        "wrote {} rows → {}\n  schema → {}\n  mean win_rate={:.3}  mean hp_frac_lost={:.3}  \
         rows-with-a-death={}/{}",
        rows.len(),
        args.out,
        schema_path,
        mean_win,
        mean_hpfrac,
        any_loss,
        rows.len()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ascension_scalar_idx() -> usize {
        SCALAR_NAMES
            .iter()
            .position(|n| *n == "ascension")
            .expect("ranker_features must expose an 'ascension' scalar")
    }

    /// The `--ascensions` clap default must stay the full spread. Regression
    /// guard: `--realistic` was hardcoded to ascension 0, which saturated the
    /// labels (mean win_rate 0.956) and is the confound that made the M1
    /// surrogate eval uninterpretable.
    #[test]
    fn default_ascensions_matches_const() {
        let args = Args::parse_from(["combat_surrogate_data"]);
        assert_eq!(
            args.ascensions,
            DEFAULT_ASCENSIONS.to_vec(),
            "the CLI default drifted from the documented spread"
        );
        assert_ne!(
            args.ascensions,
            vec![0],
            "A0-only must never be the default"
        );
    }

    #[test]
    fn parse_ascension_level_rejects_bad_input() {
        assert!(parse_ascension_level("11").is_err(), "A11 not implemented");
        assert!(
            parse_ascension_level("x").is_err(),
            "non-numeric must error"
        );
        assert_eq!(parse_ascension_level(" 10 ").expect("trims"), 10);
    }

    #[test]
    fn ascensions_flag_splits_on_commas() {
        let args = Args::parse_from(["combat_surrogate_data", "--ascensions", "0,10"]);
        assert_eq!(args.ascensions, vec![0, 10]);
    }

    /// The spread must REACH the row, not merely be computed. It is the whole
    /// point of the distributional label: a harvest that silently emits 0.0
    /// everywhere would train exactly the mean-only surrogate this replaces,
    /// and the failure would look like "the distributional head didn't help".
    #[test]
    fn realistic_rows_carry_a_nonzero_outcome_spread() {
        let rows = harvest_realistic_range(
            1,
            3,
            12,
            &ProbeBudget {
                sims: 8,
                k: 6, // >1, or every probe is its own mean and std is 0 by construction
                t_turns: 4,
                ..ProbeBudget::default()
            },
            CharacterId::Ironclad,
            &DEFAULT_ASCENSIONS,
            DriverKind::Heuristic,
            None,
        );
        assert!(!rows.is_empty(), "must emit rows");
        for r in &rows {
            assert!(
                r.hp_frac_lost_std >= 0.0 && r.hp_frac_lost_std.is_finite(),
                "spread must be a finite non-negative fraction, got {}",
                r.hp_frac_lost_std
            );
        }
        assert!(
            rows.iter().any(|r| r.hp_frac_lost_std > 0.0),
            "every row had spread exactly 0.0 across {} rows — the label is not \
             reaching the row (K={}, so probes DO differ)",
            rows.len(),
            6
        );
    }

    /// The potion index space must be a BIJECTION over `PotionKind::ALL`.
    /// This is where a silent misalignment would live: the surrogate stores an
    /// integer, the schema stores a name, and Python transfers a champion
    /// embedding row by that name. A collision or a gap sends some potions to
    /// the wrong embedding, which no downstream metric would attribute to the
    /// mapping.
    #[test]
    fn potion_index_space_is_a_bijection() {
        let table = potion_index_table();
        assert_eq!(
            table.len(),
            PotionKind::ALL.len(),
            "table must cover every PotionKind"
        );
        let lut = potion_index_lookup();
        assert_eq!(lut.len(), table.len(), "Debug names must be unique");
        for (i, p) in table.iter().enumerate() {
            assert_eq!(
                lut.get(&format!("{p:?}")).copied(),
                Some(i as u32),
                "{p:?} does not round-trip through the lookup"
            );
        }
        // Sorted by Debug name, matching card/relic table construction, so the
        // index survives enum-variant reordering.
        let mut sorted: Vec<String> = table.iter().map(|p| format!("{p:?}")).collect();
        let as_built = sorted.clone();
        sorted.sort();
        assert_eq!(as_built, sorted, "table must be sorted by Debug name");
    }

    /// Identities must REACH the row, ascending and in range.
    #[test]
    fn realistic_rows_carry_valid_potion_identities() {
        let rows = harvest_realistic_range(
            1,
            3,
            12,
            &ProbeBudget {
                sims: 8,
                k: 2,
                t_turns: 4,
                ..ProbeBudget::default()
            },
            CharacterId::Ironclad,
            &DEFAULT_ASCENSIONS,
            DriverKind::Heuristic,
            None,
        );
        assert!(!rows.is_empty(), "must emit rows");
        let n_potions = potion_index_table().len();
        for r in &rows {
            assert!(
                r.potions.windows(2).all(|w| w[0] <= w[1]),
                "potion indices must be ascending, got {:?}",
                r.potions
            );
            assert!(
                r.potions.iter().all(|&p| (p as usize) < n_potions),
                "potion index out of range in {:?}",
                r.potions
            );
        }
    }

    #[test]
    fn parse_driver_accepts_both_and_rejects_junk() {
        assert_eq!(parse_driver("heuristic").unwrap(), DriverKind::Heuristic);
        assert_eq!(parse_driver("PPO").unwrap(), DriverKind::Ppo);
        assert_eq!(parse_driver(" ppo ").unwrap(), DriverKind::Ppo);
        assert!(parse_driver("ranker").is_err());
    }

    /// `--driver ppo` without a combat budget would drive with a net that never
    /// plays combat. The budget parser is the gate, so pin that `none` parses
    /// (it is a legal budget name) but resolves to no budget — which is what
    /// `main` turns into a hard error.
    #[test]
    fn ppo_driver_requires_a_real_combat_budget() {
        assert!(parse_combat_budget("bogus").is_err());
        assert_eq!(
            parse_combat_budget("none").unwrap().budget(None, None),
            None,
            "'none' must resolve to no budget so main can reject it for --driver ppo"
        );
        assert!(parse_combat_budget("eval-cheap")
            .unwrap()
            .budget(None, None)
            .is_some());
    }

    /// The knob must reach the emitted row, not just the RunManager. This is
    /// the actual regression: `harvest_realistic_range` ignored both the
    /// requested character and the ascension, hardcoding `(Ironclad, 0)`.
    #[test]
    fn realistic_harvest_honors_requested_ascension() {
        let rows = harvest_realistic_range(
            1,
            2,
            6,
            &ProbeBudget {
                sims: 8,
                k: 2,
                t_turns: 4,
                ..ProbeBudget::default()
            },
            CharacterId::Ironclad,
            &[10],
            DriverKind::Heuristic,
            None,
        );
        assert!(!rows.is_empty(), "must emit rows");
        let idx = ascension_scalar_idx();
        for r in &rows {
            // Exact compare is correct here and clippy cannot know it: the
            // scalar is a small integer ascension written straight into an
            // f32 slot, never the result of arithmetic.
            #[allow(clippy::float_cmp)]
            {
                assert_eq!(
                    r.scalars[idx], 10.0,
                    "row carries ascension {} but 10 was requested",
                    r.scalars[idx]
                );
            }
        }
    }

    /// TDD Step 1 (task-2-brief.md): realistic harvest emits at least one
    /// `realistic`-tagged row with a valid encounter index and an
    /// `hp_frac_lost`-shaped ratio in `[0, 1]`. Kept fast (tiny probe
    /// budget, no combat search budget on the driver).
    #[test]
    fn realistic_harvest_emits_tagged_rows() {
        ensure_content_installed();
        let rows = harvest_realistic(2, 8);
        assert!(!rows.is_empty(), "must emit at least one realistic row");
        for r in &rows {
            assert_eq!(r.source, "realistic");
            assert!(
                (r.enc_idx as usize) < n_encounter_index_space(),
                "enc_idx {} out of range",
                r.enc_idx
            );
            let frac = r.hp_lost_mean / f64::from(r.max_hp.max(1));
            assert!((0.0..=1.0).contains(&frac), "hp frac out of range: {frac}");
        }
    }
}
