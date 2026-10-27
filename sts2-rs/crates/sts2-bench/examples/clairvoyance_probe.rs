//! `clairvoyance_probe` — hidden-information (clairvoyance) leak probe
//! for the chance-node MCTS (2026-07-12 audit follow-up, no bead yet).
//!
//! Since `755819594` removed the legacy PIMC mode (and with it the
//! `hide_draw_order` per-simulation reshuffle), the search root carries
//! the TRUE materialized draw-pile order and the TRUE RNG cursors
//! (chance outcome 0 is realized from the real cursors). Neither is
//! observable to a human player. This probe measures whether — and how
//! much — the deployed search exploits that hidden information.
//!
//! Two tests:
//!
//! * `invariance` — per corpus state, compare the chosen root action on
//!   the true state vs observationally-equivalent scrambles (unknown
//!   draw-pile portion uniformly permuted via the engine's own
//!   pin-preserving `cards::shuffle`; RNG streams reseeded). Decomposed
//!   into pile-only and streams-only scrambles so the two leak channels
//!   are attributed separately. The noise floor is the cross-search-seed
//!   disagreement on the FIXED true state.
//! * `closedloop` — play each corpus state to terminal six ways (same
//!   template, same fight seed, same env truth — paired by
//!   construction): TRUE (deployed, `Clairvoyant`), TRUE2 (identical but
//!   a different search-seed schedule — the honest noise-floor arm),
//!   BLIND (per decision the search runs `Clairvoyant` on a scrambled
//!   COPY of the state while the environment applies the chosen action
//!   to the truth — i.e. the agent is denied foresight; historical
//!   sanity cross-check that F0 ≈ BLIND), F0 (`hidden_info:
//!   RootScramble` — the plan's cheap honest baseline), F1
//!   (`hidden_info: DrawChance` — the plan's first recovery lever;
//!   retune CLOSED in task 4b, kept here as a standing reference), and
//!   F2 (`hidden_info: BeliefDraw` — the plan's second recovery lever,
//!   bead `sts2-rs-09x5n`, task 7b). `WR(TRUE) − WR(BLIND)` is the
//!   combat-level clairvoyance rent; `WR(TRUE) − WR(TRUE2)` bounds the
//!   search-stochasticity null. `WR(F2) − WR(F0)` vs `WR(TRUE) − WR(F0)`
//!   is the plan's G2 recovery gate (removal plan §4): PASS iff F2
//!   recovers at least half of the TRUE-vs-F0 gap (F1's recovery
//!   fraction is reported alongside as a reference, not the gate).
//!
//! ```sh
//! cargo run --release -p sts2-bench --example clairvoyance_probe -- \
//!     --mode both --states 50 --replicas 4 --sims 1000
//! ```
//!
//! As of the G4 bless (2026-07-12), the DEPLOYED run-AI no longer runs
//! this probe's TRUE arm in production: `sts2-runai`'s
//! `MctsCombatPolicy` now defaults `STS2_RUNAI_HONEST` to the honest F2
//! champion (`HiddenInfoMode::BeliefDraw`), so this probe's TRUE arm
//! models the *pre-2026-07-12* deployed (clairvoyant) behavior rather
//! than the current default — see `resolve_hidden_info` in
//! `crates/sts2-runai/src/policy/mod.rs`.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::too_many_lines
)]

#[path = "common/mod.rs"]
mod common;

use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use common::{build_default_corpus, CorpusEntry, CorpusTier};
use sts2_content::cards::spec_index::IRONCLAD_SPECS;
use sts2_content::cards::CardSpec;
use sts2_core::run_rng_set::{outcome_stream_seed_u64, splitmix64, LEGACY_RNG_SALT};
use sts2_core::{apply, is_terminal, legal_actions, Action, CombatState};
use sts2_mcts::{HiddenInfoMode, Mcts, MctsConfig, TerminalEvalMode};

const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;
const PILE_SALT: u64 = 0x51CA_4B1E_D0D0_CAFE;
const STREAM_SALT: u64 = 0x00DD_BA11_5EED_F00D;

/// Optional `(draw_widening_c, draw_widening_alpha)` override for the F1
/// retune sweep (bead `sts2-rs-k1z14`) — set once in `main` from
/// `--draw-widening-c`/`--draw-widening-alpha` (defaulting to
/// `MctsConfig::default()`'s values, i.e. a no-op override, when neither
/// flag is passed) and read by every [`mcts_config`] call. A global
/// rather than threading two more params through the half-dozen
/// `search_action` call sites (several inside `par_map` worker
/// closures) — read-only after `main` sets it before any search runs, so
/// this has no data-race implications despite the shared threads.
///
/// Applies to whichever arm's search actually runs in `DrawChance` (F1)
/// — harmless no-op for every other mode, `BeliefDraw` (F2) included,
/// since F2 has its own, independent override set below.
static DRAW_WIDENING: std::sync::OnceLock<(f32, f32)> = std::sync::OnceLock::new();

/// F2 retune sweep (bead `sts2-rs-09x5n`, task 7b): per-field
/// `MctsConfig` overrides applied ONLY when a search runs under
/// [`HiddenInfoMode::BeliefDraw`] — i.e. only the F2 arm (and
/// `--honest-belief` invariance runs) ever sees these; TRUE/TRUE2/BLIND/
/// F0/F1 read `MctsConfig::default()`'s values regardless of what's set
/// here. `None` fields keep the compiled-in default. Deliberately
/// separate from [`DRAW_WIDENING`] (F1's knob) even though both can
/// touch `draw_widening_c/alpha` — the two arms must be independently
/// tunable, and gating on `hidden_info == BeliefDraw` rather than
/// threading an arm index keeps `mcts_config` a pure function of its
/// existing `mode` parameter.
#[derive(Clone, Copy, Debug, Default)]
struct F2Overrides {
    c_puct: Option<f32>,
    draw_widening_c: Option<f32>,
    draw_widening_alpha: Option<f32>,
    chance_widening_c: Option<f32>,
    chance_widening_alpha: Option<f32>,
    endturn_widening_c: Option<f32>,
    endturn_widening_alpha: Option<f32>,
}
static F2_OVERRIDES: std::sync::OnceLock<F2Overrides> = std::sync::OnceLock::new();

// ─── Hidden-state scramble ──────────────────────────────────────────────────

/// Replace the hidden information in `state` with a fresh sample that is
/// observationally equivalent for the player:
///
/// * `pile` — uniformly permute the unknown portion of the draw pile.
///   Uses the engine's own [`sts2_core::cards::shuffle`], which preserves
///   `pinned_top` / `pinned_bottom` (Headbutt / Forethought knowledge)
///   and is a uniform Fisher–Yates over the rest. The card SET (which a
///   perfect player tracks) is unchanged; only the unobserved ORDER moves.
/// * `streams` — reseed every combat RNG stream plus the legacy rng, so
///   future rolls (enemy AI, reshuffles, random inserts, …) come from an
///   independent continuation instead of the game's true one. Each
///   pre-rolled, already-visible fact (current intents, current hand) is
///   state, not stream, and is untouched.
fn scramble_hidden(state: &mut CombatState, seed: u64, pile: bool, streams: bool) {
    if pile {
        let mut rng = sts2_core::dotnet_rng::Rng::from_search_seed(splitmix64(seed ^ PILE_SALT));
        sts2_core::cards::shuffle(&mut state.player, &mut rng);
    }
    if streams {
        let base = splitmix64(seed ^ STREAM_SALT);
        state.run_rng.reseed_combat_streams(base, 0);
        state.rng = sts2_core::dotnet_rng::Rng::from_search_seed(outcome_stream_seed_u64(
            base,
            0,
            LEGACY_RNG_SALT,
        ));
    }
}

// ─── Search wrapper ─────────────────────────────────────────────────────────

/// `mode`: gate `--honest`/`--honest-full` (bead `sts2-rs-vqckz` G1 /
/// `sts2-rs-i8bgu` F1). When non-`Clairvoyant`, the search runs under its
/// own root blindfold, seeded purely from `seed` — INSTEAD of trusting
/// whatever hidden-info scramble `scramble_hidden` already applied to the
/// caller's `state`. This is the canonical (sort-then-shuffle) blindfold,
/// not the deliberately-non-canonical `scramble_hidden` used by the BLIND
/// arm / the non-honest invariance arms below.
fn mcts_config(sims: u32, depth: u32, seed: u64, mode: HiddenInfoMode) -> MctsConfig {
    let default_widening = (
        MctsConfig::default().draw_widening_c,
        MctsConfig::default().draw_widening_alpha,
    );
    let (draw_widening_c, draw_widening_alpha) =
        DRAW_WIDENING.get().copied().unwrap_or(default_widening);
    let mut config = MctsConfig {
        max_simulations: sims,
        max_rollout_depth: depth,
        seed,
        terminal_eval_mode: TerminalEvalMode::RichBoth,
        hidden_info: mode,
        draw_widening_c,
        draw_widening_alpha,
        ..MctsConfig::default()
    };
    // F2 retune sweep (bead `sts2-rs-09x5n`): see [`F2_OVERRIDES`]'s doc —
    // gated on `mode` itself, so this only ever touches BeliefDraw searches.
    if mode == HiddenInfoMode::BeliefDraw {
        if let Some(ov) = F2_OVERRIDES.get() {
            if let Some(v) = ov.c_puct {
                config.c_puct = v;
            }
            if let Some(v) = ov.draw_widening_c {
                config.draw_widening_c = v;
            }
            if let Some(v) = ov.draw_widening_alpha {
                config.draw_widening_alpha = v;
            }
            if let Some(v) = ov.chance_widening_c {
                config.chance_widening_c = v;
            }
            if let Some(v) = ov.chance_widening_alpha {
                config.chance_widening_alpha = v;
            }
            if let Some(v) = ov.endturn_widening_c {
                config.endturn_widening_c = v;
            }
            if let Some(v) = ov.endturn_widening_alpha {
                config.endturn_widening_alpha = v;
            }
        }
    }
    config
}

/// Run one search and return the recommended action. The Debug rendering
/// is used as the comparison key in the invariance test (hand/enemy UIDs
/// are identical across scrambles, so renderings are directly comparable).
fn search_action(
    state: &CombatState,
    sims: u32,
    depth: u32,
    seed: u64,
    mode: HiddenInfoMode,
) -> Action {
    let mut mcts = Mcts::new(state.clone(), mcts_config(sims, depth, seed, mode));
    mcts.search()
}

// ─── Args ───────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Invariance,
    ClosedLoop,
    Both,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum DeckSource {
    /// Ironclad starter deck (the corpus default). Fights are mostly
    /// non-marginal on act 1 — good for the invariance test, weak power
    /// for the closed-loop rent measurement.
    Starter,
    /// `deck_size` cards sampled uniformly from the full Ironclad pool
    /// per state (mirrors `mcts_realistic_profile::random_deck`).
    /// Produces marginal, draw-order-sensitive fights.
    Random,
}

struct Args {
    mode: Mode,
    states: usize,
    replicas: u32,
    sims: u32,
    depth: u32,
    seeds: u32,
    scrambles: u32,
    max_actions: u32,
    threads: usize,
    deck: DeckSource,
    deck_size: usize,
    /// G1 honesty gate (bead `sts2-rs-vqckz` F0 / `sts2-rs-i8bgu` F1 /
    /// `sts2-rs-09x5n` F2). Invariance mode only: every search runs under
    /// this mode and the run asserts (process exit != 0 on failure) that
    /// the PILE/STREAMS/FULL flip rates are exactly 0%. `--honest` selects
    /// `RootScramble` (unchanged); `--honest-full` selects `DrawChance`;
    /// `--honest-belief` selects `BeliefDraw` — all three get the same
    /// 0-flip assertions. Closed-loop mode and the BLIND arm are
    /// unaffected by any of the three flags (always `Clairvoyant`).
    honest_mode: Option<HiddenInfoMode>,
    /// F1 retune sweep (bead `sts2-rs-k1z14`): override
    /// `MctsConfig::draw_widening_c`/`draw_widening_alpha` for every arm's
    /// search (harmless no-op for `Clairvoyant`/`TRUE`/`BLIND`/`F0`, which
    /// never read those fields — see [`DRAW_WIDENING`]). `None` keeps the
    /// compiled-in defaults.
    draw_widening_c: Option<f32>,
    draw_widening_alpha: Option<f32>,
    /// F2 retune sweep (bead `sts2-rs-09x5n`): see [`F2_OVERRIDES`].
    f2_c_puct: Option<f32>,
    f2_draw_widening_c: Option<f32>,
    f2_draw_widening_alpha: Option<f32>,
    f2_chance_widening_c: Option<f32>,
    f2_chance_widening_alpha: Option<f32>,
    f2_endturn_widening_c: Option<f32>,
    f2_endturn_widening_alpha: Option<f32>,
}

const USAGE: &str = "usage: clairvoyance_probe [--mode invariance|closedloop|both] \
    [--states N=50] [--replicas K=4] [--sims S=1000] [--depth D=200] \
    [--seeds S=4] [--scrambles M=4] [--max-actions A=400] [--threads T] \
    [--deck starter|random] [--deck-size N=25] \
    [--honest] [--honest-full] [--honest-belief] \
    [--draw-widening-c C] [--draw-widening-alpha A] \
    [--f2-c-puct C] [--f2-draw-widening-c C] [--f2-draw-widening-alpha A] \
    [--f2-chance-widening-c C] [--f2-chance-widening-alpha A] \
    [--f2-endturn-widening-c C] [--f2-endturn-widening-alpha A]";

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        mode: Mode::Both,
        states: 50,
        replicas: 4,
        sims: 1000,
        depth: 200,
        seeds: 4,
        scrambles: 4,
        max_actions: 400,
        threads: std::thread::available_parallelism()
            .map_or(8, std::num::NonZeroUsize::get)
            .saturating_sub(2)
            .clamp(1, 40),
        deck: DeckSource::Starter,
        deck_size: 25,
        honest_mode: None,
        draw_widening_c: None,
        draw_widening_alpha: None,
        f2_c_puct: None,
        f2_draw_widening_c: None,
        f2_draw_widening_alpha: None,
        f2_chance_widening_c: None,
        f2_chance_widening_alpha: None,
        f2_endturn_widening_c: None,
        f2_endturn_widening_alpha: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        // `--honest`/`--honest-full`/`--honest-belief` are bare flags (no
        // value) — handle them before the `val()` closure below, which
        // every other flag consumes a value through.
        if flag == "--honest" || flag == "--honest-full" || flag == "--honest-belief" {
            let requested = match flag.as_str() {
                "--honest" => HiddenInfoMode::RootScramble,
                "--honest-full" => HiddenInfoMode::DrawChance,
                _ => HiddenInfoMode::BeliefDraw,
            };
            if let Some(existing) = args.honest_mode {
                if existing != requested {
                    return Err(
                        "pass at most one of --honest / --honest-full / --honest-belief"
                            .to_string(),
                    );
                }
            }
            args.honest_mode = Some(requested);
            continue;
        }
        let mut val = || it.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--mode" => {
                args.mode = match val()?.as_str() {
                    "invariance" => Mode::Invariance,
                    "closedloop" => Mode::ClosedLoop,
                    "both" => Mode::Both,
                    other => return Err(format!("unknown mode {other}")),
                }
            }
            "--states" => args.states = val()?.parse().map_err(|e| format!("{e}"))?,
            "--replicas" => args.replicas = val()?.parse().map_err(|e| format!("{e}"))?,
            "--sims" => args.sims = val()?.parse().map_err(|e| format!("{e}"))?,
            "--depth" => args.depth = val()?.parse().map_err(|e| format!("{e}"))?,
            "--seeds" => args.seeds = val()?.parse().map_err(|e| format!("{e}"))?,
            "--scrambles" => args.scrambles = val()?.parse().map_err(|e| format!("{e}"))?,
            "--max-actions" => args.max_actions = val()?.parse().map_err(|e| format!("{e}"))?,
            "--threads" => args.threads = val()?.parse().map_err(|e| format!("{e}"))?,
            "--deck" => {
                args.deck = match val()?.as_str() {
                    "starter" => DeckSource::Starter,
                    "random" => DeckSource::Random,
                    other => return Err(format!("unknown deck source {other}")),
                }
            }
            "--deck-size" => args.deck_size = val()?.parse().map_err(|e| format!("{e}"))?,
            "--draw-widening-c" => {
                args.draw_widening_c = Some(val()?.parse().map_err(|e| format!("{e}"))?);
            }
            "--draw-widening-alpha" => {
                args.draw_widening_alpha = Some(val()?.parse().map_err(|e| format!("{e}"))?);
            }
            "--f2-c-puct" => {
                args.f2_c_puct = Some(val()?.parse().map_err(|e| format!("{e}"))?);
            }
            "--f2-draw-widening-c" => {
                args.f2_draw_widening_c = Some(val()?.parse().map_err(|e| format!("{e}"))?);
            }
            "--f2-draw-widening-alpha" => {
                args.f2_draw_widening_alpha = Some(val()?.parse().map_err(|e| format!("{e}"))?);
            }
            "--f2-chance-widening-c" => {
                args.f2_chance_widening_c = Some(val()?.parse().map_err(|e| format!("{e}"))?);
            }
            "--f2-chance-widening-alpha" => {
                args.f2_chance_widening_alpha = Some(val()?.parse().map_err(|e| format!("{e}"))?);
            }
            "--f2-endturn-widening-c" => {
                args.f2_endturn_widening_c = Some(val()?.parse().map_err(|e| format!("{e}"))?);
            }
            "--f2-endturn-widening-alpha" => {
                args.f2_endturn_widening_alpha = Some(val()?.parse().map_err(|e| format!("{e}"))?);
            }
            other => return Err(format!("unknown flag {other}")),
        }
    }
    Ok(args)
}

// ─── Parallel map (panic-isolating) ─────────────────────────────────────────

/// Map `f` over `items` on `threads` OS threads. A panicking item yields
/// `None` (content-gap fights are skipped, not counted as losses — the
/// paired analysis needs all arms of an item or none).
fn par_map<T, R, F>(items: &[T], threads: usize, f: F) -> Vec<Option<R>>
where
    T: Sync,
    R: Send,
    F: Fn(usize, &T) -> R + Sync,
{
    let next = AtomicUsize::new(0);
    let results: Vec<Mutex<Option<R>>> = items.iter().map(|_| Mutex::new(None)).collect();
    std::thread::scope(|scope| {
        for _ in 0..threads.min(items.len()) {
            scope.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= items.len() {
                    break;
                }
                let out = std::panic::catch_unwind(AssertUnwindSafe(|| f(i, &items[i]))).ok();
                *results[i].lock().expect("result slot poisoned") = out;
            });
        }
    });
    results
        .into_iter()
        .map(|m| m.into_inner().expect("result slot poisoned"))
        .collect()
}

// ─── Corpus expansion ───────────────────────────────────────────────────────

fn expand_corpus(states: usize, replicas: u32) -> Vec<CorpusEntry> {
    let mut base = build_default_corpus();
    base.truncate(states);
    let mut out = Vec::with_capacity(base.len() * replicas as usize);
    for k in 0..replicas {
        for e in &base {
            let mut e2 = e.clone();
            e2.state_id = e.state_id + 1000 * k;
            e2.engine_seed = e.engine_seed.wrapping_add(u64::from(k) * 100_003);
            e2.warmup_seed = e.warmup_seed.wrapping_add(u64::from(k) * 31_337);
            e2.warmup_turns = 1 + ((e.warmup_turns + k) % 3);
            out.push(e2);
        }
    }
    out
}

/// Sample `size` cards uniformly from `IRONCLAD_SPECS`, keyed on `seed`
/// (deterministic, host-independent). Mirrors
/// `mcts_realistic_profile::random_deck`.
fn random_deck(seed: u64, size: usize) -> Vec<CardSpec> {
    let mut rng = sts2_core::dotnet_rng::Rng::new((seed ^ 0xDEC0_DEC0_DEC0_DEC0) as i32);
    (0..size)
        .map(|_| {
            let idx = rng.next_int_exclusive(0, IRONCLAD_SPECS.len() as i32) as usize;
            IRONCLAD_SPECS[idx]()
        })
        .collect()
}

/// Deck-source-aware sibling of `common::realize_state`: build the
/// encounter state (starter or per-state random deck) and run the same
/// random warmup so the probe measures mid-combat decision states.
fn realize(entry: &CorpusEntry, args: &Args) -> CombatState {
    let mut state = match args.deck {
        DeckSource::Starter => common::build_ironclad_state(&entry.encounter, entry.engine_seed),
        DeckSource::Random => {
            let deck = random_deck(
                splitmix64(u64::from(entry.state_id).wrapping_mul(GOLDEN) ^ 0xDECC),
                args.deck_size,
            );
            common::build_ironclad_state_with_deck(&entry.encounter, entry.engine_seed, &deck)
        }
    };
    common::warmup_random(&mut state, entry.warmup_seed, entry.warmup_turns);
    state
}

// ─── Test A: invariance ─────────────────────────────────────────────────────

#[derive(Default)]
struct InvCounts {
    /// Pairs compared / disagreeing among search seeds on the TRUE state.
    null_cmp: u64,
    null_diff: u64,
    /// (scramble, seed) comparisons vs the same-seed TRUE action.
    full_cmp: u64,
    full_diff: u64,
    pile_cmp: u64,
    pile_diff: u64,
    stream_cmp: u64,
    stream_diff: u64,
    /// States with at least one pile-only flip (breadth of the leak).
    states_with_pile_flip: u64,
    states_total: u64,
}

impl InvCounts {
    fn merge(&mut self, o: &InvCounts) {
        self.null_cmp += o.null_cmp;
        self.null_diff += o.null_diff;
        self.full_cmp += o.full_cmp;
        self.full_diff += o.full_diff;
        self.pile_cmp += o.pile_cmp;
        self.pile_diff += o.pile_diff;
        self.stream_cmp += o.stream_cmp;
        self.stream_diff += o.stream_diff;
        self.states_with_pile_flip += o.states_with_pile_flip;
        self.states_total += o.states_total;
    }
}

fn invariance_one(entry: &CorpusEntry, args: &Args) -> InvCounts {
    let state = realize(entry, args);
    let seed_base = splitmix64(0xC1A1_0000u64 ^ u64::from(entry.state_id).wrapping_mul(GOLDEN));
    let mode = args.honest_mode.unwrap_or(HiddenInfoMode::Clairvoyant);

    let mut c = InvCounts {
        states_total: 1,
        ..InvCounts::default()
    };

    let a_true: Vec<String> = (0..args.seeds)
        .map(|s| {
            let a = search_action(
                &state,
                args.sims,
                args.depth,
                seed_base ^ u64::from(s),
                mode,
            );
            format!("{a:?}")
        })
        .collect();
    for i in 0..a_true.len() {
        for j in (i + 1)..a_true.len() {
            c.null_cmp += 1;
            c.null_diff += u64::from(a_true[i] != a_true[j]);
        }
    }

    let mut pile_flip_here = false;
    for m in 0..args.scrambles {
        let scr_seed = splitmix64(seed_base ^ u64::from(m).wrapping_mul(GOLDEN) ^ 0xA5A5);

        // Full scramble (pile + streams), compared at every search seed.
        let mut full = state.clone();
        scramble_hidden(&mut full, scr_seed, true, true);
        for s in 0..args.seeds {
            let a = search_action(&full, args.sims, args.depth, seed_base ^ u64::from(s), mode);
            c.full_cmp += 1;
            c.full_diff += u64::from(format!("{a:?}") != a_true[s as usize]);
        }

        // Decomposition arms at search seed 0 only.
        let mut pile = state.clone();
        scramble_hidden(&mut pile, scr_seed, true, false);
        let a = search_action(&pile, args.sims, args.depth, seed_base, mode);
        c.pile_cmp += 1;
        let flipped = format!("{a:?}") != a_true[0];
        c.pile_diff += u64::from(flipped);
        pile_flip_here |= flipped;

        let mut streams = state.clone();
        scramble_hidden(&mut streams, scr_seed, false, true);
        let a = search_action(&streams, args.sims, args.depth, seed_base, mode);
        c.stream_cmp += 1;
        c.stream_diff += u64::from(format!("{a:?}") != a_true[0]);
    }
    c.states_with_pile_flip += u64::from(pile_flip_here);
    c
}

fn rate(diff: u64, cmp: u64) -> f64 {
    if cmp == 0 {
        0.0
    } else {
        diff as f64 / cmp as f64
    }
}

fn run_invariance(work: &[CorpusEntry], args: &Args) {
    let start = Instant::now();
    let done = AtomicUsize::new(0);
    let rows = par_map(work, args.threads, |_, e| {
        let r = invariance_one(e, args);
        let d = done.fetch_add(1, Ordering::Relaxed) + 1;
        if d.is_multiple_of(25) {
            eprintln!("[invariance] {d}/{} states", work.len());
        }
        r
    });
    let mut total = InvCounts::default();
    let mut panics = 0u32;
    for r in &rows {
        match r {
            Some(c) => total.merge(c),
            None => panics += 1,
        }
    }
    println!("\n================ TEST A: INVARIANCE ================");
    println!(
        "states={} panicked={} sims={} seeds={} scrambles={} wall={:.1}s",
        total.states_total,
        panics,
        args.sims,
        args.seeds,
        args.scrambles,
        start.elapsed().as_secs_f64(),
    );
    println!(
        "  noise floor  (seed vs seed, true state):   {:6.2}%  ({}/{})",
        100.0 * rate(total.null_diff, total.null_cmp),
        total.null_diff,
        total.null_cmp,
    );
    println!(
        "  FULL scramble (pile+streams) flip rate:    {:6.2}%  ({}/{})",
        100.0 * rate(total.full_diff, total.full_cmp),
        total.full_diff,
        total.full_cmp,
    );
    println!(
        "  PILE-only scramble flip rate:              {:6.2}%  ({}/{})",
        100.0 * rate(total.pile_diff, total.pile_cmp),
        total.pile_diff,
        total.pile_cmp,
    );
    println!(
        "  STREAMS-only scramble flip rate:           {:6.2}%  ({}/{})",
        100.0 * rate(total.stream_diff, total.stream_cmp),
        total.stream_diff,
        total.stream_cmp,
    );
    println!(
        "  states with >=1 pile-only flip:            {:6.2}%  ({}/{})",
        100.0 * rate(total.states_with_pile_flip, total.states_total),
        total.states_with_pile_flip,
        total.states_total,
    );
    println!(
        "  verdict: flip rates well above the noise floor mean the chosen\n\
         \x20 action depends on information a player cannot observe."
    );

    // G1 honesty gate (bead `sts2-rs-vqckz` F0 plan §4 / `sts2-rs-i8bgu` F1
    // plan §3.2 step 5). Under `--honest`/`--honest-full` every arm above
    // searched with a non-`Clairvoyant` `HiddenInfoMode`, so the chosen
    // action is a pure function of the search seed regardless of the
    // incoming (unobservable) pile order / RNG cursors — the
    // PILE/STREAMS/FULL flip rates must be EXACTLY zero, by construction,
    // not merely "low". A nonzero rate here means the honest mode is
    // leaking and would silently poison every honest-mode measurement
    // downstream (G2/G4) — fail loudly instead.
    if let Some(mode) = args.honest_mode {
        let failures: Vec<(&str, u64, u64)> = [
            ("PILE-only", total.pile_diff, total.pile_cmp),
            ("STREAMS-only", total.stream_diff, total.stream_cmp),
            ("FULL", total.full_diff, total.full_cmp),
        ]
        .into_iter()
        .filter(|&(_, diff, _)| diff != 0)
        .collect();
        if failures.is_empty() {
            println!(
                "  G1 HONESTY GATE: PASS — PILE/STREAMS/FULL flip rates are all exactly 0% \
                 under {mode:?} ({} states, {panics} panicked).",
                total.states_total,
            );
        } else {
            eprintln!(
                "\n[clairvoyance_probe] G1 HONESTY GATE FAILED: {mode:?} must be exactly \
                 invariant to hidden information, but the following arm(s) still flipped:"
            );
            for (label, diff, cmp) in &failures {
                eprintln!(
                    "  {label}: {diff}/{cmp} flips ({:.4}%) — expected exactly 0",
                    100.0 * rate(*diff, *cmp),
                );
            }
            eprintln!(
                "This means the honest mode is leaking hidden information — do NOT trust \
                 any --honest/--honest-full measurement until this is fixed."
            );
            std::process::exit(1);
        }
    }
}

// ─── Test B: closed-loop A/B ────────────────────────────────────────────────

struct FightOut {
    won: bool,
    final_hp: i32,
    timeout: bool,
}

/// One closed-loop arm: everything that differs between TRUE / TRUE2 /
/// BLIND / F0 / F1 is expressed here so `play_fight` stays a single
/// generic driver instead of growing an arm per branch (brief: "resist
/// restructuring the fight loop; add an arm descriptor").
struct Arm {
    label: &'static str,
    /// `MctsConfig::hidden_info` passed to every decision's search.
    hidden_info: HiddenInfoMode,
    /// BLIND-style pre-search scramble: the search runs on a freshly
    /// scrambled COPY of the state each decision (env still applies the
    /// recommendation to the TRUE state). Orthogonal to `hidden_info` —
    /// BLIND is the only arm that sets this (Clairvoyant search denied
    /// foresight by construction, the historical sanity cross-check for
    /// F0). F0/F1 rely on `hidden_info` itself for honesty, not this.
    scramble_root: bool,
    /// TRUE2 is the only arm on the distinct (`seed_b`) search-seed
    /// stream — the honest noise-floor null. F0/F1 deliberately reuse
    /// TRUE's `seed_a` stream per the brief ("the mode is the only
    /// difference").
    alt_seed: bool,
}

const ARM_TRUE: usize = 0;
const ARM_TRUE2: usize = 1;
const ARM_BLIND: usize = 2;
const ARM_F0: usize = 3;
const ARM_F1: usize = 4;
const ARM_F2: usize = 5;

const ARMS: [Arm; 6] = [
    Arm {
        label: "TRUE",
        hidden_info: HiddenInfoMode::Clairvoyant,
        scramble_root: false,
        alt_seed: false,
    },
    Arm {
        label: "TRUE2",
        hidden_info: HiddenInfoMode::Clairvoyant,
        scramble_root: false,
        alt_seed: true,
    },
    Arm {
        label: "BLIND",
        hidden_info: HiddenInfoMode::Clairvoyant,
        scramble_root: true,
        alt_seed: false,
    },
    Arm {
        label: "F0",
        hidden_info: HiddenInfoMode::RootScramble,
        scramble_root: false,
        alt_seed: false,
    },
    Arm {
        label: "F1",
        hidden_info: HiddenInfoMode::DrawChance,
        scramble_root: false,
        alt_seed: false,
    },
    Arm {
        label: "F2",
        hidden_info: HiddenInfoMode::BeliefDraw,
        scramble_root: false,
        alt_seed: false,
    },
];

/// Drive one fight to terminal under `arm`. Closed-loop mode is
/// unaffected by `--honest`/`--honest-full` (brief: "closed-loop mode +
/// the existing BLIND arm stay unchanged") — each arm's `hidden_info` is
/// fixed by its `Arm` descriptor, not `args.honest_mode`.
fn play_fight(template: &CombatState, fight_seed: u64, arm: &Arm, args: &Args) -> FightOut {
    let mut state = template.clone();
    let mut count: u32 = 0;
    while !is_terminal(&state) && count < args.max_actions {
        if legal_actions(&state).is_empty() {
            let _ = apply(&mut state, Action::EndTurn);
            count += 1;
            continue;
        }
        let step_seed = fight_seed ^ u64::from(count).wrapping_mul(GOLDEN);
        let recommended = if arm.scramble_root {
            let mut root = state.clone();
            scramble_hidden(&mut root, splitmix64(step_seed ^ 0xB11D), true, true);
            search_action(&root, args.sims, args.depth, step_seed, arm.hidden_info)
        } else {
            search_action(&state, args.sims, args.depth, step_seed, arm.hidden_info)
        };
        let _ = apply(&mut state, recommended);
        count += 1;
    }
    let final_hp = state.player.current_hp.max(0);
    FightOut {
        won: is_terminal(&state) && final_hp > 0,
        final_hp,
        timeout: !is_terminal(&state),
    }
}

struct FightSet {
    tier: CorpusTier,
    /// Indexed by `ARM_TRUE` / `ARM_TRUE2` / `ARM_BLIND` / `ARM_F0` /
    /// `ARM_F1` / `ARM_F2`, in `ARMS` order. All six play the SAME
    /// template (same encounter/warmup/truth) so every cross-arm
    /// comparison is paired by construction.
    outs: Vec<FightOut>,
}

/// Exact two-sided McNemar (binomial n01+n10, p=0.5) via log-space pmf.
fn mcnemar_p(n01: u64, n10: u64) -> f64 {
    let n = n01 + n10;
    if n == 0 {
        return 1.0;
    }
    let ln_fact = {
        let mut v = vec![0.0f64; (n + 1) as usize];
        for i in 1..=n as usize {
            v[i] = v[i - 1] + (i as f64).ln();
        }
        v
    };
    let ln_half_n = -(n as f64) * std::f64::consts::LN_2;
    let k = n01.min(n10);
    let mut tail = 0.0f64;
    for i in 0..=k {
        let ln_pmf =
            ln_fact[n as usize] - ln_fact[i as usize] - ln_fact[(n - i) as usize] + ln_half_n;
        tail += ln_pmf.exp();
    }
    (2.0 * tail).min(1.0)
}

fn summarize_pair(label: &str, pairs: &[(bool, bool)]) {
    let n = pairs.len() as u64;
    let a_wins = pairs.iter().filter(|(a, _)| *a).count() as u64;
    let b_wins = pairs.iter().filter(|(_, b)| *b).count() as u64;
    let n01 = pairs.iter().filter(|(a, b)| !*a && *b).count() as u64;
    let n10 = pairs.iter().filter(|(a, b)| *a && !*b).count() as u64;
    println!(
        "  {label}: {:.1}% vs {:.1}%  (delta {:+.1}pp, flips {n10}->lost/{n01}->won, \
         McNemar p={:.2e}, n={n})",
        100.0 * a_wins as f64 / n as f64,
        100.0 * b_wins as f64 / n as f64,
        100.0 * (b_wins as f64 - a_wins as f64) / n as f64,
        mcnemar_p(n01, n10),
    );
}

/// Win-rate fraction for one arm over the usable (paired, non-timeout,
/// non-panicked) fight sets.
fn win_rate(ok: &[&FightSet], arm: usize) -> f64 {
    if ok.is_empty() {
        return 0.0;
    }
    ok.iter().filter(|fs| fs.outs[arm].won).count() as f64 / ok.len() as f64
}

/// `(won_a, won_b)` pairs for two arms, for feeding `summarize_pair` /
/// `mcnemar_p`.
fn arm_pairs(ok: &[&FightSet], a: usize, b: usize) -> Vec<(bool, bool)> {
    ok.iter()
        .map(|fs| (fs.outs[a].won, fs.outs[b].won))
        .collect()
}

/// Paired HP sign test between two arbitrary arms — finer-grained than
/// win/loss, catches rent in fights whose outcome is locked in either
/// way.
fn hp_sign_test(label: &str, ok: &[&FightSet], base: usize, other: usize) {
    let worse = ok
        .iter()
        .filter(|fs| fs.outs[other].final_hp < fs.outs[base].final_hp)
        .count() as u64;
    let better = ok
        .iter()
        .filter(|fs| fs.outs[other].final_hp > fs.outs[base].final_hp)
        .count() as u64;
    println!(
        "  HP sign test  {label}: {worse} worse / {better} better (binomial p={:.2e})",
        mcnemar_p(worse, better),
    );
}

fn run_closedloop(work: &[CorpusEntry], args: &Args) {
    // Guard the `ARM_*` index constants against `ARMS` reordering —
    // every downstream lookup is by index, so a silent mismatch here
    // would misattribute an entire arm's results.
    debug_assert_eq!(ARMS[ARM_TRUE].label, "TRUE");
    debug_assert_eq!(ARMS[ARM_TRUE2].label, "TRUE2");
    debug_assert_eq!(ARMS[ARM_BLIND].label, "BLIND");
    debug_assert_eq!(ARMS[ARM_F0].label, "F0");
    debug_assert_eq!(ARMS[ARM_F1].label, "F1");
    debug_assert_eq!(ARMS[ARM_F2].label, "F2");

    let start = Instant::now();
    let done = AtomicUsize::new(0);
    let rows = par_map(work, args.threads, |_, e| {
        let template = realize(e, args);
        let gid = u64::from(e.state_id);
        let seed_a = splitmix64(gid.wrapping_mul(3).wrapping_add(1));
        let seed_b = splitmix64(gid.wrapping_mul(3).wrapping_add(2));
        let outs: Vec<FightOut> = ARMS
            .iter()
            .map(|arm| {
                let seed = if arm.alt_seed { seed_b } else { seed_a };
                play_fight(&template, seed, arm, args)
            })
            .collect();
        let fs = FightSet { tier: e.tier, outs };
        let d = done.fetch_add(1, Ordering::Relaxed) + 1;
        if d.is_multiple_of(20) {
            eprintln!(
                "[closedloop] {d}/{} fights x{} arms done ({:.0}s)",
                work.len(),
                ARMS.len(),
                start.elapsed().as_secs_f64(),
            );
        }
        fs
    });

    let mut panics = 0u32;
    let mut timeouts = 0u32;
    let mut ok: Vec<&FightSet> = Vec::new();
    for r in &rows {
        match r {
            None => panics += 1,
            Some(fs) if fs.outs.iter().any(|o| o.timeout) => timeouts += 1,
            Some(fs) => ok.push(fs),
        }
    }

    println!("\n================ TEST B: CLOSED-LOOP A/B ================");
    println!(
        "fights={} usable={} panicked={panics} timed-out={timeouts} sims={} wall={:.1}s",
        rows.len(),
        ok.len(),
        args.sims,
        start.elapsed().as_secs_f64(),
    );
    println!("  arms: TRUE  = deployed (root sees true hidden state);");
    println!("        TRUE2 = same, different search seeds (noise floor);");
    println!("        BLIND = root scrambled per decision (no foresight, historical sanity check ~= F0);");
    println!(
        "        F0    = MctsConfig::hidden_info=RootScramble on the true root (plan G2 baseline);"
    );
    println!(
        "        F1    = MctsConfig::hidden_info=DrawChance on the true root (plan G2 recovery lever, retune CLOSED per task 4b);"
    );
    println!(
        "        F2    = MctsConfig::hidden_info=BeliefDraw on the true root (plan G2 recovery lever, bead sts2-rs-09x5n)."
    );
    // Print the RESOLVED F2 config (defaults + any --f2-* overrides) so
    // every closed-loop log is self-describing about which hyperparameters
    // produced it — required for the task 7b retune sweep to be legible
    // after the fact without re-deriving flags from shell history.
    let f2_cfg = mcts_config(args.sims, args.depth, 0, HiddenInfoMode::BeliefDraw);
    println!(
        "  F2 config: c_puct={:.4} draw_widening=({:.4},{:.4}) chance_widening=({:.4},{:.4}) \
         endturn_widening=({:.4},{:.4})",
        f2_cfg.c_puct,
        f2_cfg.draw_widening_c,
        f2_cfg.draw_widening_alpha,
        f2_cfg.chance_widening_c,
        f2_cfg.chance_widening_alpha,
        f2_cfg.endturn_widening_c,
        f2_cfg.endturn_widening_alpha,
    );

    summarize_pair(
        "TRUE vs TRUE2 (null) ",
        &arm_pairs(&ok, ARM_TRUE, ARM_TRUE2),
    );
    summarize_pair(
        "TRUE vs BLIND (rent) ",
        &arm_pairs(&ok, ARM_TRUE, ARM_BLIND),
    );
    summarize_pair("TRUE vs F0    (rent) ", &arm_pairs(&ok, ARM_TRUE, ARM_F0));
    summarize_pair("TRUE vs F1    (rent) ", &arm_pairs(&ok, ARM_TRUE, ARM_F1));
    summarize_pair("TRUE vs F2    (rent) ", &arm_pairs(&ok, ARM_TRUE, ARM_F2));

    let mean_hp = |arm: usize| -> f64 {
        ok.iter()
            .map(|fs| f64::from(fs.outs[arm].final_hp))
            .sum::<f64>()
            / ok.len().max(1) as f64
    };
    println!(
        "  mean final HP: TRUE {:.1} | TRUE2 {:.1} | BLIND {:.1} | F0 {:.1} | F1 {:.1} | F2 {:.1}",
        mean_hp(ARM_TRUE),
        mean_hp(ARM_TRUE2),
        mean_hp(ARM_BLIND),
        mean_hp(ARM_F0),
        mean_hp(ARM_F1),
        mean_hp(ARM_F2),
    );

    hp_sign_test("TRUE vs TRUE2 (null)", &ok, ARM_TRUE, ARM_TRUE2);
    hp_sign_test("TRUE vs BLIND (rent)", &ok, ARM_TRUE, ARM_BLIND);
    hp_sign_test("TRUE vs F0    (rent)", &ok, ARM_TRUE, ARM_F0);
    hp_sign_test("TRUE vs F1    (rent)", &ok, ARM_TRUE, ARM_F1);
    hp_sign_test("TRUE vs F2    (rent)", &ok, ARM_TRUE, ARM_F2);

    for (tier, label) in [
        (CorpusTier::Common, "common"),
        (CorpusTier::Elite, "elite "),
        (CorpusTier::Boss, "boss  "),
    ] {
        let idx: Vec<usize> = (0..ok.len()).filter(|&i| ok[i].tier == tier).collect();
        if idx.is_empty() {
            continue;
        }
        let sub_ok: Vec<&FightSet> = idx.iter().map(|&i| ok[i]).collect();
        summarize_pair(
            &format!("[{label}] TRUE vs BLIND"),
            &arm_pairs(&sub_ok, ARM_TRUE, ARM_BLIND),
        );
        summarize_pair(
            &format!("[{label}] TRUE vs F0   "),
            &arm_pairs(&sub_ok, ARM_TRUE, ARM_F0),
        );
        summarize_pair(
            &format!("[{label}] TRUE vs F1   "),
            &arm_pairs(&sub_ok, ARM_TRUE, ARM_F1),
        );
        summarize_pair(
            &format!("[{label}] TRUE vs F2   "),
            &arm_pairs(&sub_ok, ARM_TRUE, ARM_F2),
        );
    }
    println!(
        "  verdict: BLIND/F0/F1/F2 below TRUE beyond the TRUE-vs-TRUE2 null means the\n\
         \x20 deployed win rate includes rent from unobservable information."
    );

    // ─── G2: F1/F2 vs F0 recovery measurement (plan §4, bead sts2-rs-k1z14
    // / sts2-rs-09x5n task 7b) ───
    //
    // F0 (root blindfold, single-determinization) is the cheap honest
    // baseline; F1 (draws-as-chance) and F2 (belief-pile deferred draws)
    // are the two recovery levers under test. G2 asks how much of the
    // TRUE-vs-F0 clairvoyance gap each closes; per the task 7b brief the
    // VERDICT LINE is decided by F2 (F1's retune was already CLOSED as a
    // non-robust noise peak in task 4b — kept here only as a standing
    // reference point).
    println!("\n---- G2: F1/F2 vs F0 direct comparison (recovery measurement) ----");
    summarize_pair("F1 vs F0 (direct)    ", &arm_pairs(&ok, ARM_F0, ARM_F1));
    hp_sign_test("F1 vs F0 (direct)", &ok, ARM_F0, ARM_F1);
    summarize_pair("F2 vs F0 (direct)    ", &arm_pairs(&ok, ARM_F0, ARM_F2));
    hp_sign_test("F2 vs F0 (direct)", &ok, ARM_F0, ARM_F2);
    for (tier, label) in [
        (CorpusTier::Common, "common"),
        (CorpusTier::Elite, "elite "),
        (CorpusTier::Boss, "boss  "),
    ] {
        let idx: Vec<usize> = (0..ok.len()).filter(|&i| ok[i].tier == tier).collect();
        if idx.is_empty() {
            continue;
        }
        let sub_ok: Vec<&FightSet> = idx.iter().map(|&i| ok[i]).collect();
        summarize_pair(
            &format!("[{label}] F1 vs F0 (direct)"),
            &arm_pairs(&sub_ok, ARM_F0, ARM_F1),
        );
        summarize_pair(
            &format!("[{label}] F2 vs F0 (direct)"),
            &arm_pairs(&sub_ok, ARM_F0, ARM_F2),
        );
    }

    let wr_true = win_rate(&ok, ARM_TRUE);
    let wr_f0 = win_rate(&ok, ARM_F0);
    let wr_f1 = win_rate(&ok, ARM_F1);
    let wr_f2 = win_rate(&ok, ARM_F2);
    let gap = wr_true - wr_f0;
    let recovered_f1 = wr_f1 - wr_f0;
    let recovered_f2 = wr_f2 - wr_f0;
    println!(
        "\n  WR(TRUE)={:.2}%  WR(F0)={:.2}%  WR(F1)={:.2}%  WR(F2)={:.2}%  \
         clairvoyance-gap(TRUE-F0)={:+.2}pp  recovered(F1-F0)={:+.2}pp  recovered(F2-F0)={:+.2}pp",
        100.0 * wr_true,
        100.0 * wr_f0,
        100.0 * wr_f1,
        100.0 * wr_f2,
        100.0 * gap,
        100.0 * recovered_f1,
        100.0 * recovered_f2,
    );
    // Success criterion (plan G2): the arm under test recovers >= half of
    // the F0->TRUE gap. This is a MEASUREMENT, not an assert — per the
    // brief, do NOT exit non-zero on failure; just print PASS/FAIL and let
    // the controller decide next steps. Report BOTH F1 and F2 recovery
    // fractions; per the task 7b brief the VERDICT LINE uses F2.
    if gap <= 0.0 {
        println!(
            "  G2 VERDICT: INCONCLUSIVE — TRUE-vs-F0 clairvoyance gap is {:+.2}pp (<=0), \
             so \"recovery fraction\" is undefined on this run.",
            100.0 * gap,
        );
    } else {
        let recovery_frac_f1 = recovered_f1 / gap;
        let recovery_frac_f2 = recovered_f2 / gap;
        println!(
            "  F1 recovery fraction (reference, not the gate): {:.1}%",
            100.0 * recovery_frac_f1,
        );
        let verdict = if recovered_f2 >= gap / 2.0 {
            "PASS"
        } else {
            "FAIL"
        };
        println!(
            "  G2 VERDICT: {verdict} — F2 recovered {:.1}% of the TRUE-F0 clairvoyance gap \
             (need >=50.0%); F2-F0 = {:+.2}pp vs required >= {:+.2}pp.",
            100.0 * recovery_frac_f2,
            100.0 * recovered_f2,
            100.0 * gap / 2.0,
        );
    }
}

// ─── Main ───────────────────────────────────────────────────────────────────

fn main() {
    sts2_content::cards::dispatch::ensure_installed();
    sts2_content::relics::dispatch::ensure_installed();
    sts2_content::potions::dispatch::ensure_installed();

    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("[clairvoyance_probe] error: {e}\n{USAGE}");
            std::process::exit(2);
        }
    };

    // Set once, before any search runs — see `DRAW_WIDENING`'s doc.
    DRAW_WIDENING
        .set((
            args.draw_widening_c
                .unwrap_or(MctsConfig::default().draw_widening_c),
            args.draw_widening_alpha
                .unwrap_or(MctsConfig::default().draw_widening_alpha),
        ))
        .expect("DRAW_WIDENING set exactly once, before any mcts_config call");
    // Ditto for the F2 (BeliefDraw-only) overrides — see `F2_OVERRIDES`'s doc.
    F2_OVERRIDES
        .set(F2Overrides {
            c_puct: args.f2_c_puct,
            draw_widening_c: args.f2_draw_widening_c,
            draw_widening_alpha: args.f2_draw_widening_alpha,
            chance_widening_c: args.f2_chance_widening_c,
            chance_widening_alpha: args.f2_chance_widening_alpha,
            endturn_widening_c: args.f2_endturn_widening_c,
            endturn_widening_alpha: args.f2_endturn_widening_alpha,
        })
        .expect("F2_OVERRIDES set exactly once, before any mcts_config call");

    // Content-gap panics inside a worker are recorded as skips; keep the
    // default hook from spamming stderr mid-run.
    std::panic::set_hook(Box::new(|_| {}));

    let work = expand_corpus(args.states, args.replicas);
    eprintln!(
        "[clairvoyance_probe] {} states ({} base x {} replicas), {} threads",
        work.len(),
        args.states,
        args.replicas,
        args.threads,
    );

    if matches!(args.mode, Mode::Invariance | Mode::Both) {
        run_invariance(&work, &args);
    }
    if matches!(args.mode, Mode::ClosedLoop | Mode::Both) {
        run_closedloop(&work, &args);
    }
}
