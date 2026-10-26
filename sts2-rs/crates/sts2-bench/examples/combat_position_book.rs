//! `combat_position_book` — freeze a stratified position book with
//! method-independent gold answers (#446).
//!
//! Combat changes are currently gated on instruments that are either
//! noise-bound (win rate over a 30-run panel; the PPO screen is
//! documented blind below ~8pp) or narrow (`blunder_budgets.toml`
//! counts named pathologies, not decision quality). Nothing measures
//! "did the search pick the right move" across a representative set.
//!
//! The 2026-08-10 root-parallel study solved the hard part inside
//! `mcts_thread_agreement`: run a deep oracle **twice** — once
//! single-threaded, once root-parallel — and keep only positions where
//! both are unanimous across their own seeds AND agree with each other.
//! There the reference action is method-independent, so a `threads=1`
//! arm and a merged arm are judged against the same answer without the
//! shared-geometry bias that a single-threaded oracle would introduce.
//!
//! That study was a one-off, and it sampled synthetic random-deck
//! Ironclad positions on an Act-1 encounter mix. This binary makes the
//! method reusable and fixes the sampling:
//!
//! * **Real positions.** Entries come from the harvested on-manifold
//!   fixture pool (`logs/2026-08-12-fixture-pool/harvest.sh`) — real
//!   decks, relics and HP reached by actually playing runs, not 24
//!   cards drawn uniformly from the character registry.
//! * **Stratified.** Per character and per tier, because both matter.
//!   Boss positions are where #442's evidence is weakest (64.5% of boss
//!   decisions unresolved) and where the win-rate knee sits, but they
//!   are only ~8% of a raw harvest. Silent carries ~3x Ironclad's rate
//!   of transposition-merged root children (13.1% vs 4.6%), so an
//!   Ironclad-only book is not representative of anything.
//! * **Frozen.** The oracle pass is paid once and written to disk.
//!   Downstream A/Bs score against the stored answers.
//!
//! The output is the committed artifact; the fixture pool that fed it
//! is regenerable and git-ignored.
//!
//! ## What `gold_agr` cannot see, and what it is not
//!
//! The gold filter keeps only positions where **both oracles are
//! unanimous across their own seeds**. That is what makes the answer
//! method-independent, and it is also what makes the book blind to
//! near-ties: a position where the oracle is split is dropped, and a
//! split oracle is exactly the signature of two moves worth almost the
//! same. An aggregation, ordering or tie-break change fires precisely
//! there.
//!
//! Measured against the engine-proven ordering suite (`dominance_pairs`,
//! GH #492), a 50 000-sim search of this dialect picks the **provably
//! dominated** order on 19.0% of proven pairs — and a 50x budget increase
//! buys only 3.3pp on the block axis, so it is converged on an objective
//! that does not price those advantages rather than merely under-sampled.
//! More simulations will not fix it.
//!
//! **That does NOT mean `gold_agr` is 19% wrong.** The `annotate`
//! subcommand exists to check, and the answer for the committed books is:
//!
//! | book | entries | carry a proof | gold contradicted |
//! |---|---:|---:|---:|
//! | `combat_position_book.json` | 168 | 1 | **0** |
//! | `combat_position_book_v2.json` | 159 | 0 | **0** |
//! | `combat_position_book_v3.json` | 292 | 3 | **0** |
//!
//! The v3 row is the 2026-10-18 rebuild (GH #805; it was 303 / 2 / 0).
//! **v1 and v2 were not rebuilt and are stale**: 0/168 and 0/159 of their
//! positions realize to the state they froze once fixture combats carry
//! `player.character`, so their rows above describe books nothing can score.
//!
//! v1/v2 are verified against the belt-less pool (`data/run-fixtures/`);
//! **v3 is verified against the belt-carrying pool**
//! (`data/run-fixtures-potions/`, `just harvest-fixtures-potions`) — the
//! v2 book had zero positions with a potion available because the harvest
//! wrote `potions = []` into every fixture. Pass the right pool root or
//! every entry reports a hash miss. `logs/2026-09-05-book-v3/FINDINGS.md`
//! has the v3 build and baseline.
//!
//! The proof-bearing class is ~0.27% of decision points (741 proven pairs
//! out of 2,652,820 orderings tried across 216,000 positions), so a
//! 168-entry book contains almost none of it. `gold_agr` is not
//! contaminated by the class — it simply does not sample it.
//!
//! The practical consequence for anyone tuning against `gold_agr`: it is a
//! sound metric for *resolved* decisions and says nothing about the
//! near-tie class. Use `dominance_pairs` as the co-metric there; it is
//! oracle-free, so the two cannot share a bias.
//!
//! ## Motif tags
//!
//! Every entry carries `motifs` — the `sts2_cli::replay::motif` predicates
//! (incoming attack, lethal threat, free 0-cost play, power in hand, …),
//! plus `proven_pair` when the dominance enumeration found one and
//! `gold_end_turn` / `gold_potion` for the answer's shape. `mcts_book_sweep`
//! and the standing gate break agreement and resolution down per tag, so a
//! regression names the decision class it hit. Tags are written at build
//! time and refreshed by `annotate` — the position-only ones are a pure
//! function of the frozen state, so they never need the oracle re-paid.
//!
//! ## Per-child oracle Q (`oracle_q`) and regret
//!
//! Every entry also carries the oracle's **root table** — each legal
//! action's pooled mean value, standard error and visit count over the six
//! searches (2 methods × 3 seeds) that voted on the gold answer. That lets
//! a consumer price a miss instead of counting it: `mcts_book_sweep`
//! reports `regret = Q(gold) − Q(pick)` next to `gold_agr`, and the gate
//! pairs it. The reason it exists is the 2026-09-06 budget panel: the book
//! predicted the decision-level trade of an elite-focused budget exactly,
//! but agreement weights a coin-flip miss and a blunder the same, and the
//! run outcome did not (`logs/2026-09-06-budget-realloc/FINDINGS.md`).
//!
//! The table is written at build time. For a book built before the field
//! existed, `oracle-q` replays the same double oracle (same sims, seeds,
//! warm-up, dialect) and writes only `oracle_q`; the gold answers are
//! never rewritten, because every existing dump was scored against them.
//!
//! ## The hard book (`hard`): positions selected by playout, gold set by playout
//!
//! GH #494 step 2 (`logs/2026-09-07-blunder-diagnosis/FINDINGS.md`) showed
//! two things about the oracle books above. The per-child oracle Q is a
//! visit-count artifact across children of one root, so `regret` does not
//! price a miss; and the unanimity filter selects **easy** positions — 94
//! of the 142 positions the deployed agent missed on v3 are fights it wins
//! 100% of the time from gold *and* from every mis-pick. Agreement on such
//! a book rewards matching a 50k-sim visit attractor, not value.
//!
//! `hard` builds a book the other way round, oracle-free:
//!
//! 1. **Screen.** Every candidate is played out by the deployed 1000-sim
//!    agent from the position as it stands (no forced move), `screen`
//!    times with paired seeds. Only positions whose win rate lands inside
//!    `(band_lo, band_hi)` — default (0.2, 0.8) — are kept: the fight is
//!    still in doubt, so a first-move error can cost a win.
//! 2. **Gold.** At each kept position every *distinct* legal move (card
//!    identity + target, so two copies of a card in hand are one move) is
//!    forced as the first move and the agent finishes the fight `gold`
//!    times; the moves within `within` wins of the leader get `refine`
//!    more. Gold is the move with the highest win rate, HP at the end of
//!    the fight breaking ties. The whole table, the margin to the
//!    runner-up and a z for it are stored in `playout`, so a consumer can
//!    weight or filter by how well the position is resolved.
//! 3. **Pool.** Built from `data/run-fixtures-hard/` (`just
//!    harvest-fixtures-hard`): the belt-carrying harvest repeated at
//!    ascension 10 and at a 40-HP snapshot, the two knobs the v3 pool held
//!    at 0 and 70. Sub-pools are one directory level below the root and
//!    `fixture_rel_path` carries that level. Fixtures are realized at their
//!    own `[player].ascension` (`common::realize_fixture`); the v1–v3 pools
//!    carry none, so their hashes are unchanged.
//!
//! Every playout first **re-deals the hidden information** (unknown draw
//! order, hidden RNG streams; `sts2_core::blindfold::blindfold_root` seeded
//! by the playout), so gold is the move that wins most *over deals* — what
//! the player faces — and not the move that wins most on the one deal the
//! fixture happens to freeze, which is knowledge no honest search has. The
//! first hard book (2026-09-07) was built without this (`--no-determinize`;
//! `playout.determinized` absent) and its gold carried that clairvoyance;
//! see `logs/2026-09-07-potion-diagnostic/FINDINGS.md`.
//!
//! The units are the run's: win% and HP, from the agent that will be
//! scored. `gold_agr` on this book is "did the search find the move that
//! wins most often when the fight is close", and a pick of a second copy
//! of the gold card (`gold_aliases`) counts as agreement. `oracle_q` is
//! empty and `regret` reads `-`, deliberately.
//!
//! ```sh
//! cargo run --release -p sts2-bench --example combat_position_book -- \
//!     [pool_root] [out_json] [per_char_common] [per_char_elite] \
//!     [per_char_boss] [oracle_sims] [oracle_seeds]
//! cargo run --release -p sts2-bench --example combat_position_book -- \
//!     annotate <book_json> [pool_root]
//! cargo run --release -p sts2-bench --example combat_position_book -- \
//!     oracle-q <book_json> [pool_root]
//! cargo run --release -p sts2-bench --example combat_position_book -- \
//!     hard <pool_root> <out_json> [--common 40] [--elite 60] [--boss 60] \
//!     [--target 300] [--screen 16] [--gold 24] [--refine 72] [--within 2] \
//!     [--band 0.2,0.8] [--sims 1000] [--warmup 8] [--chunk 32] [--no-determinize]
//! ```

#[path = "common/mod.rs"]
mod common;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sts2_cli::replay::motif::{motifs_at, Motif};
use sts2_core::{is_terminal, legal_actions, Action, CombatPhase, CombatState};
use sts2_mcts::{par_search, HiddenInfoMode, Mcts, MctsConfig};

/// The book's oracle dialect: BeliefDraw + `c_puct` 2.5 + `draw_widening_c`
/// 2.0, with the engine-default EndTurn widening 0.5 / 0.4 (`oracle_config`
/// does NOT call `MctsConfig::apply_mode_dialect`).
///
/// That is the PRE-#467 dialect. It matched the advisor
/// (`realgame_advisor.py::HONEST_SEARCH`) until GH #758 moved every
/// interactive surface onto the run agent's 5.787 / 0.436, and never matched
/// the run agent (`sts2-runai::mcts_config_for`) after PR #467. Kept as-is
/// so the committed books stay reproducible from this source; a book meant
/// to judge the deployed search must be rebuilt with `apply_mode_dialect`
/// (the gold answers encode the search that built them).
const ORACLE_DEPTH: u32 = 200;
const C_PUCT: f32 = 2.5;
const DRAW_WIDENING_C: f32 = 2.0;

/// Reader-only mirror of the `RealisticCorpus` shape `corpus_gen
/// --runmanager-toml-export` writes. The generator owns the canonical
/// struct; duplicating the reader keeps this example off the generator's
/// dependency graph, matching what `mcts_ab.rs` already does.
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

/// One frozen book position. `fixture_path` is stored relative to the
/// pool root so the book survives being moved between machines — the
/// absolute paths the harvest writes would not.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BookEntry {
    book_id: u32,
    character: String,
    tier: String,
    floor: u16,
    fixture_rel_path: String,
    seed: u64,
    /// Legal actions at the scored decision point. `1.0 / legal_actions`
    /// is this position's chance-agreement baseline.
    legal_actions: usize,
    /// `state_hash` of the scored state — a consumer that realizes the
    /// fixture and gets a different hash is not scoring this position.
    state_hash: u64,
    /// The action both oracles agree on, as a debug string. `Action`
    /// has no `Serialize` impl and the debug rendering is what every
    /// sibling harness already compares on.
    gold_action: String,
    /// Engine-proven ordering facts at this position, written by the
    /// `annotate` subcommand. Empty on an un-annotated book, so every
    /// existing consumer deserializes unchanged.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    proven_pairs: Vec<ProvenPair>,
    /// `true` when [`Self::gold_action`] starts the **dominated** side of
    /// one of those pairs — i.e. the 50 000-sim oracle is provably wrong
    /// here, and any `gold_agr` computed over this entry is scoring
    /// agreement with a known-bad answer.
    ///
    /// Measured on the #492 dominance suite: the oracle gets **19.0%** of
    /// engine-proven ordering pairs wrong, and a 50x budget increase buys
    /// 3.3pp on the block axis — it is converged on an objective that does
    /// not price these advantages, not merely noisy. More simulations will
    /// not fix it.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    gold_is_proven_dominated: bool,
    /// Motif tags (`sts2_cli::replay::motif::Motif::as_str` spellings):
    /// what kind of decision this position poses, so a `gold_agr` move can
    /// be attributed to a class rather than reported as one scalar. Written
    /// at build time and refreshed by `annotate`; the position-only tags
    /// are a pure function of the frozen state, `proven_pair` comes from the
    /// dominance proofs, and the `gold_*` tags from the frozen answer.
    /// Empty on a book written before this field existed.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    motifs: Vec<String>,
    /// Per-root-child oracle value at this position, pooled over every
    /// oracle search that voted on the gold answer (both methods × seeds).
    /// Written at build time and by the `oracle-q` subcommand; empty on a
    /// book built before the field existed.
    ///
    /// This is what turns `gold_agr` from a hit count into a *value*: a
    /// consumer that picked action `a` lost `Q(gold) − Q(a)` in the oracle's
    /// own units, so a coin-flip miss and a blunder stop counting the same
    /// (GH #494 prerequisite; the 2026-09-06 budget panel showed agreement
    /// alone predicted the decision trade but not its run value).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    oracle_q: Vec<OracleChild>,
    /// Other debug renderings of the same move as [`Self::gold_action`] —
    /// a second copy of the gold card in another hand slot. Written by
    /// `hard`, which scores moves by card identity; an oracle book's gold
    /// is one `hand_index` and this is empty. A pick matching an alias is
    /// agreement (`mcts_book_sweep`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    gold_aliases: Vec<String>,
    /// `[player].ascension` of the fixture, which is the ascension the
    /// position is realized at. 0 on every pool harvested before the hard
    /// one, and omitted then so older books re-serialize unchanged.
    #[serde(default, skip_serializing_if = "is_zero_u8")]
    ascension: u8,
    /// The playout evidence behind a `hard` book's gold answer. `None` on
    /// oracle books.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    playout: Option<EntryPlayout>,
}

#[allow(clippy::trivially_copy_pass_by_ref)] // serde's skip_serializing_if signature
fn is_zero_u8(v: &u8) -> bool {
    *v == 0
}

/// One distinct first move's forced-first-move playout record: the deployed
/// agent took `action` first and finished the fight `n` times. `aliases` are
/// the other debug renderings of the same move (second copy of the card).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct PlayoutChild {
    action: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    aliases: Vec<String>,
    n: u32,
    wins: u32,
    /// Mean player HP at the end of the fight (0 on a loss).
    mean_hp: f32,
}

impl PlayoutChild {
    fn win_rate(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            f64::from(self.wins) / f64::from(self.n)
        }
    }
}

/// Why a `hard` entry's gold is what it is.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct EntryPlayout {
    /// Free playouts in the screen (the agent chose every move) and how many
    /// it won — the position's difficulty for the agent being scored.
    screen_n: u32,
    screen_wins: u32,
    /// The agent's own first move across the screen playouts, by debug
    /// rendering. `gold_agr` at 1000 sims for free, and the search's own
    /// spread at this position.
    screen_first_moves: BTreeMap<String, u32>,
    /// Every distinct legal first move, best-first (win rate, then mean HP).
    table: Vec<PlayoutChild>,
    /// `win%(gold) − win%(runner-up)` in percentage points.
    margin_pp: f64,
    /// `margin / sqrt(p₀(1−p₀)/n₀ + p₁(1−p₁)/n₁)`; capped at 99 when both
    /// rates are degenerate.
    z: f64,
}

/// The knobs of a `hard` build, so the book says what its gold means.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PlayoutParams {
    /// Deployed-agent simulations per move inside every playout.
    sims: u32,
    screen_playouts: u32,
    gold_playouts: u32,
    refine_playouts: u32,
    refine_within_wins: u32,
    band_lo: f64,
    band_hi: f64,
    /// Whether every playout re-dealt the hidden information first
    /// (`sts2_core::blindfold::blindfold_root` seeded by the playout).
    /// `false` (the 2026-09-07 first build) means every playout of a
    /// position ran on the one deal frozen in the fixture, so its gold is
    /// the best move *for that deal* — partly knowledge no honest search
    /// has. Absent on that first book; `true` on every later one.
    #[serde(default)]
    determinized: bool,
    /// Candidates whose screen win rate fell inside the band, by bucket
    /// (before the stratified cap).
    in_band_by_bucket: BTreeMap<String, usize>,
}

/// One root child's pooled oracle statistics. `action` is the same debug
/// rendering [`BookEntry::gold_action`] uses, so a consumer joins on it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct OracleChild {
    action: String,
    /// `Σ value_sum / Σ visits` over the pooled searches — the oracle's
    /// player-perspective expected value of taking this action.
    q: f32,
    /// Standard error of `q` from the pooled return variance.
    stderr: f32,
    visits: u64,
}

/// Pool the root children of several searches of the **same position**
/// into one table keyed by action. Sums are exact; the mean and its
/// standard error come from the pooled `Σv` and `Σv²`, so a child a
/// worker never visited simply contributes nothing.
///
/// The root-parallel oracle already merges its workers this way
/// (`par_search` sums per-child counters), so pooling its result with the
/// sequential oracle's is the same operation one level up. Children are
/// returned best-first so `[0]` is the oracle's argmax-Q and the top-two
/// gap is `[0].q − [1].q`.
fn pool_outcomes(runs: &[Vec<sts2_mcts::RootOutcome>]) -> Vec<OracleChild> {
    // (visits, Σv, Σv²)
    let mut acc: BTreeMap<String, (u64, f64, f64)> = BTreeMap::new();
    for run in runs {
        for o in run {
            let slot = acc
                .entry(format!("{:?}", o.action))
                .or_insert((0, 0.0, 0.0));
            slot.0 += u64::from(o.visits);
            slot.1 += f64::from(o.value_sum);
            slot.2 += f64::from(o.value_sq_sum);
        }
    }
    let mut out: Vec<OracleChild> = acc
        .into_iter()
        .map(|(action, (visits, sum, sq))| {
            #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
            let (q, stderr) = if visits == 0 {
                (0.0, 0.0)
            } else {
                let n = visits as f64;
                let mean = sum / n;
                let var = (sq / n - mean * mean).max(0.0);
                ((mean) as f32, (var / n).sqrt() as f32)
            };
            OracleChild {
                action,
                q,
                stderr,
                visits,
            }
        })
        .collect();
    out.sort_by(|a, b| {
        b.q.partial_cmp(&a.q)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.visits.cmp(&a.visits))
    });
    out
}

/// Every motif tag for one book position: the position predicates, plus
/// `proven_pair` when the dominance enumeration found one, plus the shape
/// of the gold answer. One function so `build` and `annotate` cannot
/// disagree about the vocabulary.
fn motif_tags(state: &CombatState, gold_action: &str, has_proof: bool) -> Vec<String> {
    let mut tags: Vec<Motif> = motifs_at(state);
    if has_proof {
        tags.push(Motif::ProvenPair);
    }
    tags.extend(Motif::from_gold_action(gold_action));
    tags.sort();
    tags.dedup();
    tags.into_iter().map(|m| m.as_str().to_string()).collect()
}

/// One engine-proven ordering fact: playing `better_first` and following
/// with the other card reaches a state that weakly dominates the one the
/// opposite order reaches, on the listed axes.
///
/// This is a *local* claim. It does not assert `better_first` is the best
/// action at the position, only that it is never worse than `worse_first`
/// as the first of those two cards — so it can contradict a gold answer
/// without supplying a replacement for it.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProvenPair {
    better_first: String,
    worse_first: String,
    better_card: String,
    worse_card: String,
    axes: Vec<String>,
    gain_enemy_hp: i32,
    gain_player_hp: i32,
    gain_player_block: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Book {
    version: u32,
    git_sha: String,
    generator: String,
    /// Everything needed to reproduce the gold answers.
    oracle_sims: u32,
    oracle_seeds: u32,
    oracle_depth: u32,
    /// v1 books predate this field, and **their warm-up cap was 0** — the
    /// generator's `unwrap_or(8)` came later. Defaulting to 8 here
    /// realizes 137 of `combat_position_book.json`'s 168 entries at the
    /// wrong decision point; defaulting to 0 reproduces all 168. The
    /// `state_hash` check catches the mistake either way, which is what it
    /// is for, but the default also has to agree with
    /// `mcts_book_sweep`'s (plain `#[serde(default)]`, i.e. 0) or the two
    /// harnesses disagree about which position an entry names.
    #[serde(default)]
    max_warmup_plies: u32,
    hidden_info: String,
    c_puct: f32,
    draw_widening_c: f32,
    /// Positions considered before the gold filter, by (character, tier).
    scored_by_bucket: BTreeMap<String, usize>,
    /// Mean `1 / legal_actions` over kept entries — the score a random
    /// picker gets on this book.
    chance_agreement: f64,
    /// Entries carrying at least one engine proof, and entries whose gold
    /// answer a proof contradicts. Both zero until `annotate` runs.
    #[serde(default)]
    annotated_with_proofs: usize,
    #[serde(default)]
    gold_contradicted: usize,
    /// Entries carrying a per-child oracle Q table. Equal to `entries.len()`
    /// on a book built after the field existed or after `oracle-q` ran.
    #[serde(default)]
    oracle_q_entries: usize,
    /// How the gold answers were chosen: `"oracle_unanimous"` (double 50k
    /// oracle, the default and the only method before `hard`) or
    /// `"playout"`. Empty on books written before the field existed, which
    /// are all oracle books.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    gold_method: String,
    /// The playout knobs of a `"playout"` book. `None` otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    playout: Option<PlayoutParams>,
    entries: Vec<BookEntry>,
}

fn oracle_config(sims: u32, seed: u64) -> MctsConfig {
    MctsConfig {
        max_simulations: sims,
        seed,
        max_rollout_depth: ORACLE_DEPTH,
        hidden_info: HiddenInfoMode::BeliefDraw,
        c_puct: C_PUCT,
        draw_widening_c: DRAW_WIDENING_C,
        ..MctsConfig::default()
    }
}

fn oracle_seed(k: u32) -> u64 {
    u64::from(k)
        .wrapping_mul(0xD1B5_4A32_D192_ED03)
        .wrapping_add(0x0C1E_0BAD_5EED_0001)
}

/// Realize a harvested fixture into the state the search will see.
///
/// `scenario_to_combat_state` returns a pre-`start_combat` state whose
/// only legal action is `EndTurn`; the combat loop has to be driven to
/// `PlayerPlayPhase` or every oracle answer would be "EndTurn". Same
/// progression `mcts_ab::load_realistic` performs.
///
/// Returns `None` for fixtures that fail to load, fail to realize (a
/// content-port gap the harvest should have filtered), land outside
/// `PlayerPlayPhase`, or offer fewer than two actions — a position with
/// one legal action carries no decision to be right or wrong about.
fn realize(path: &Path, seed: u64, max_warmup_plies: u32) -> Option<CombatState> {
    let mut state = common::realize_fixture(path, seed).ok()?;
    common::warmup_plies(&mut state, seed, max_warmup_plies);
    if state.is_over || state.phase != CombatPhase::PlayerPlayPhase {
        return None;
    }
    if legal_actions(&state).len() < 2 {
        return None;
    }
    Some(state)
}

/// Evenly stride `target` entries out of `pool` so the sample spans the
/// harvest order (which tracks run seed and floor) instead of taking a
/// correlated prefix.
fn stride_sample(pool: &[HarvestEntry], target: usize) -> Vec<HarvestEntry> {
    if pool.len() <= target {
        return pool.to_vec();
    }
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    let step = pool.len() as f64 / target as f64;
    (0..target)
        .map(|i| pool[((i as f64 * step) as usize).min(pool.len() - 1)].clone())
        .collect()
}

/// Annotate an existing book with engine proofs — **no oracle re-run**.
///
/// A dominance proof is a pure function of the position, so this pass
/// only has to realize each entry, verify its `state_hash`, and enumerate
/// the orderings. That matters: rebuilding the book costs a 50 000-sim
/// double-oracle pass over every candidate, and this costs seconds.
///
/// Why annotate rather than override. The proof says `better_first` is
/// never worse than `worse_first` **as the first of those two cards**. It
/// does not say `better_first` is the best action at the position, so
/// substituting it for the gold answer would trade one unproven answer
/// for another. Marking the entry lets a consumer drop it from `gold_agr`
/// — where the oracle is demonstrably wrong — and score it against the
/// proof instead, which is the only oracle-free judgement available.
fn cmd_annotate(args: &[String]) -> std::io::Result<()> {
    let book_path = PathBuf::from(
        args.first()
            .cloned()
            .unwrap_or_else(|| "data/combat_position_book.json".to_string()),
    );
    let pool_root = PathBuf::from(
        args.get(1)
            .cloned()
            .unwrap_or_else(|| "data/run-fixtures".to_string()),
    );
    let mut book: Book = serde_json::from_slice(&std::fs::read(&book_path)?)
        .map_err(|e| std::io::Error::other(format!("{}: {e}", book_path.display())))?;

    std::panic::set_hook(Box::new(|_| {}));

    let annotated: Vec<(u32, Vec<ProvenPair>, bool, Vec<String>, Skip)> = book
        .entries
        .par_iter()
        .map(|e| {
            let path = pool_root.join(&e.fixture_rel_path);
            let Some(state) =
                std::panic::catch_unwind(|| realize(&path, e.seed, book.max_warmup_plies))
                    .ok()
                    .flatten()
            else {
                // NOT silent. The fixture pool is git-ignored and
                // regenerable, so a fresh worktree has none of it — and an
                // uncounted skip here reports "0 proofs" for "0 fixtures",
                // which is the same wrong answer as a real null. Cost the
                // first time this was written: a positive control that came
                // back 0 and nearly shipped as a finding.
                return (e.book_id, Vec::new(), false, Vec::new(), Skip::Unrealized);
            };
            // A hash miss means we are not looking at the position the
            // oracle scored, so any proof from it would be attached to the
            // wrong entry. Counted, never silently annotated.
            if sts2_core::state_hash(&state) != e.state_hash {
                return (e.book_id, Vec::new(), false, Vec::new(), Skip::HashMiss);
            }
            let (_, found) = common::dominance::pairs_at(&state);
            let pairs: Vec<ProvenPair> = found
                .into_iter()
                .map(|(better, worse, bl, wl, axes, ge, gh, gb)| ProvenPair {
                    better_first: format!("{better:?}"),
                    worse_first: format!("{worse:?}"),
                    better_card: bl,
                    worse_card: wl,
                    axes,
                    gain_enemy_hp: ge,
                    gain_player_hp: gh,
                    gain_player_block: gb,
                })
                .collect();
            let contradicted = pairs.iter().any(|p| p.worse_first == e.gold_action);
            let motifs = motif_tags(&state, &e.gold_action, !pairs.is_empty());
            (e.book_id, pairs, contradicted, motifs, Skip::None)
        })
        .collect();

    let (mut unrealized, mut misses) = (0_usize, 0_usize);
    for (id, pairs, contradicted, motifs, skip) in annotated {
        match skip {
            Skip::Unrealized => unrealized += 1,
            Skip::HashMiss => misses += 1,
            Skip::None => {}
        }
        if let Some(e) = book.entries.iter_mut().find(|e| e.book_id == id) {
            e.proven_pairs = pairs;
            e.gold_is_proven_dominated = contradicted;
            // Only overwrite tags for entries that realized; a skipped
            // entry keeps whatever the last successful pass wrote.
            if skip == Skip::None {
                e.motifs = motifs;
            }
        }
    }
    let mut motif_hist: BTreeMap<&str, usize> = BTreeMap::new();
    for e in &book.entries {
        for m in &e.motifs {
            *motif_hist.entry(m.as_str()).or_default() += 1;
        }
    }
    book.annotated_with_proofs = book
        .entries
        .iter()
        .filter(|e| !e.proven_pairs.is_empty())
        .count();
    book.gold_contradicted = book
        .entries
        .iter()
        .filter(|e| e.gold_is_proven_dominated)
        .count();

    // Refuse to write a book whose every entry was skipped. Writing
    // `proven_pairs: []` everywhere would look identical to a genuine
    // null and would then be committed as one.
    assert!(
        unrealized < book.entries.len(),
        "every entry failed to realize from {} — the fixture pool is \
         git-ignored and regenerable; run logs/2026-08-12-fixture-pool/harvest.sh \
         or point the second argument at an existing pool. Refusing to write \
         an all-empty annotation that would read as a null.",
        pool_root.display()
    );

    println!("book            : {}", book_path.display());
    println!("entries         : {}", book.entries.len());
    println!(
        "with a proof    : {} ({:.1}%)",
        book.annotated_with_proofs,
        pct(book.annotated_with_proofs, book.entries.len())
    );
    println!(
        "gold CONTRADICTED: {} ({:.1}%)  <- gold_agr is scoring these against a known-wrong answer",
        book.gold_contradicted,
        pct(book.gold_contradicted, book.entries.len())
    );
    if unrealized > 0 {
        println!("UNREALIZED      : {unrealized} (fixture missing or dead — NOT a null)");
    }
    if misses > 0 {
        println!("state_hash misses: {misses} (left un-annotated, deliberately)");
    }
    println!("\nmotifs (entries carrying the tag):");
    for m in Motif::ALL {
        let n = motif_hist.get(m.as_str()).copied().unwrap_or(0);
        println!(
            "  {:<20} {n:>4}  ({:.1}%)",
            m.as_str(),
            pct(n, book.entries.len())
        );
    }
    std::fs::write(&book_path, serde_json::to_vec_pretty(&book)?)?;
    Ok(())
}

/// Run every `(position, workers, seed)` oracle task in parallel with a
/// progress line every ~2%, returning the pick and the full root table for
/// each. Shared by `build` and `oracle-q` so the two cannot disagree about
/// the dialect or the seed schedule.
fn run_oracle_tasks<'a>(
    tasks: &[(usize, u32, u32)],
    state_of: impl Fn(usize) -> &'a CombatState + Sync,
    oracle_sims: u32,
    done: &std::sync::atomic::AtomicUsize,
    every: usize,
    t0: &Instant,
) -> Vec<(usize, u32, String, Vec<sts2_mcts::RootOutcome>)> {
    tasks
        .par_iter()
        .map(|&(p, w, k)| {
            let cfg = oracle_config(oracle_sims, oracle_seed(k));
            let r = par_search(state_of(p).clone(), cfg, w);
            let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if n.is_multiple_of(every) || n == tasks.len() {
                eprintln!(
                    "[book] oracle {n}/{} tasks ({:.1}%) after {:.0}s",
                    tasks.len(),
                    100.0 * n as f64 / tasks.len() as f64,
                    t0.elapsed().as_secs_f64()
                );
            }
            (p, w, format!("{:?}", r.best_action), r.outcomes)
        })
        .collect()
}

/// Re-run the oracle on an existing book and write the per-child Q table
/// — **the gold answers are never changed.**
///
/// The book's `oracle_sims` / `oracle_seeds` / `max_warmup_plies` are
/// reused, so this is the same double oracle that froze the answers,
/// replayed to keep its root tables. Each entry is realized and verified
/// against its `state_hash` first (an entry that fails stays untouched and
/// is counted, as in `annotate`). The re-run's own unanimous pick is
/// compared with the stored gold and any disagreement is *printed*, not
/// written: the root-parallel oracle is not bit-reproducible across rayon
/// schedules, so a rare flip is the oracle's variance showing, and the
/// stored answer is the one every existing dump was scored against.
fn cmd_oracle_q(args: &[String]) -> std::io::Result<()> {
    let book_path = PathBuf::from(
        args.first()
            .cloned()
            .unwrap_or_else(|| "data/combat_position_book.json".to_string()),
    );
    let pool_root = PathBuf::from(
        args.get(1)
            .cloned()
            .unwrap_or_else(|| "data/run-fixtures".to_string()),
    );
    let mut book: Book = serde_json::from_slice(&std::fs::read(&book_path)?)
        .map_err(|e| std::io::Error::other(format!("{}: {e}", book_path.display())))?;

    std::panic::set_hook(Box::new(|_| {}));
    let realized: Vec<(usize, Result<CombatState, Skip>)> = book
        .entries
        .par_iter()
        .enumerate()
        .map(|(i, e)| {
            let path = pool_root.join(&e.fixture_rel_path);
            let Some(state) =
                std::panic::catch_unwind(|| realize(&path, e.seed, book.max_warmup_plies))
                    .ok()
                    .flatten()
            else {
                return (i, Err(Skip::Unrealized));
            };
            if sts2_core::state_hash(&state) != e.state_hash {
                return (i, Err(Skip::HashMiss));
            }
            (i, Ok(state))
        })
        .collect();
    let _ = std::panic::take_hook();
    let unrealized = realized
        .iter()
        .filter(|(_, r)| r.as_ref().err() == Some(&Skip::Unrealized))
        .count();
    let misses = realized
        .iter()
        .filter(|(_, r)| r.as_ref().err() == Some(&Skip::HashMiss))
        .count();
    let live: Vec<(usize, CombatState)> = realized
        .into_iter()
        .filter_map(|(i, r)| r.ok().map(|s| (i, s)))
        .collect();
    assert!(
        !live.is_empty(),
        "no entry realized and verified from {} — the fixture pool is git-ignored \
         and regenerable (`just harvest-fixtures` / `just harvest-fixtures-potions`); \
         refusing to write a book with an empty oracle_q on every entry.",
        pool_root.display()
    );
    eprintln!(
        "[book] oracle-q: {} of {} entries verified (unrealized {unrealized}, hash misses \
         {misses}); oracle {} sims x {} seeds x 2 methods",
        live.len(),
        book.entries.len(),
        book.oracle_sims,
        book.oracle_seeds
    );

    let t0 = Instant::now();
    let tasks: Vec<(usize, u32, u32)> = (0..live.len())
        .flat_map(|p| {
            [1_u32, 8_u32]
                .into_iter()
                .flat_map(move |w| (0..book.oracle_seeds).map(move |k| (p, w, k)))
        })
        .collect();
    let done = std::sync::atomic::AtomicUsize::new(0);
    let every = (tasks.len() / 50).max(1);
    let picks = run_oracle_tasks(&tasks, |p| &live[p].1, book.oracle_sims, &done, every, &t0);
    eprintln!(
        "[book] oracle pass done in {:.1}s",
        t0.elapsed().as_secs_f64()
    );

    let mut seq: Vec<BTreeMap<String, u32>> = vec![BTreeMap::new(); live.len()];
    let mut par: Vec<BTreeMap<String, u32>> = vec![BTreeMap::new(); live.len()];
    let mut runs: Vec<Vec<Vec<sts2_mcts::RootOutcome>>> = vec![Vec::new(); live.len()];
    for (p, w, pick, outcomes) in picks {
        let slot = if w == 1 { &mut seq[p] } else { &mut par[p] };
        *slot.entry(pick).or_default() += 1;
        runs[p].push(outcomes);
    }

    let unanimous = |m: &BTreeMap<String, u32>| -> Option<String> {
        let (action, count) = m.iter().max_by_key(|(_, v)| **v)?;
        (*count == book.oracle_seeds).then(|| action.clone())
    };
    let (mut rerun_agrees, mut rerun_split, mut rerun_flipped) = (0_usize, 0_usize, 0_usize);
    let mut gaps: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut argmax_is_gold = 0_usize;
    for (p, (i, _)) in live.iter().enumerate() {
        let table = pool_outcomes(&runs[p]);
        let gold = book.entries[*i].gold_action.clone();
        let tier = book.entries[*i].tier.clone();
        match (unanimous(&seq[p]), unanimous(&par[p])) {
            (Some(s), Some(q)) if s == q && s == gold => rerun_agrees += 1,
            (Some(s), Some(q)) if s == q => {
                rerun_flipped += 1;
                eprintln!(
                    "[book] book_id {} : re-run oracle unanimous on {s} but stored gold is {gold} \
                     (kept the stored answer)",
                    book.entries[*i].book_id
                );
            }
            _ => rerun_split += 1,
        }
        if let Some(g) = table.iter().find(|c| c.action == gold) {
            let best_other = table
                .iter()
                .filter(|c| c.action != gold)
                .map(|c| c.q)
                .fold(f32::NEG_INFINITY, f32::max);
            if best_other.is_finite() {
                gaps.entry(tier)
                    .or_default()
                    .push(f64::from(g.q - best_other));
            }
            if table.first().is_some_and(|c| c.action == gold) {
                argmax_is_gold += 1;
            }
        }
        book.entries[*i].oracle_q = table;
    }
    book.oracle_q_entries = book
        .entries
        .iter()
        .filter(|e| !e.oracle_q.is_empty())
        .count();

    println!("book            : {}", book_path.display());
    println!("entries         : {}", book.entries.len());
    println!(
        "oracle_q written: {} ({:.1}%)",
        book.oracle_q_entries,
        pct(book.oracle_q_entries, book.entries.len())
    );
    println!(
        "re-run vs gold  : agrees {rerun_agrees}, split {rerun_split}, unanimous-but-different \
         {rerun_flipped}  (gold never rewritten)"
    );
    println!(
        "gold = argmax Q : {argmax_is_gold} of {} (each search picked its own argmax-Q child; \
         the rest are near-ties where the six pooled tables rank the top two differently)",
        live.len()
    );
    if unrealized > 0 {
        println!(
            "UNREALIZED      : {unrealized} (fixture missing or dead — left without oracle_q)"
        );
    }
    if misses > 0 {
        println!("state_hash misses: {misses} (left without oracle_q, deliberately)");
    }
    println!("\noracle top-two gap Q(gold) − max Q(other), by tier:");
    println!(
        "  {:<8} {:>4}  {:>8}  {:>8}  {:>8}",
        "tier", "n", "median", "p25", "p75"
    );
    for (tier, v) in &mut gaps {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let at = |f: f64| {
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                clippy::cast_precision_loss
            )]
            let idx = ((v.len() - 1) as f64 * f).round() as usize;
            v[idx]
        };
        println!(
            "  {tier:<8} {:>4}  {:>8.4}  {:>8.4}  {:>8.4}",
            v.len(),
            at(0.5),
            at(0.25),
            at(0.75)
        );
    }
    std::fs::write(&book_path, serde_json::to_vec_pretty(&book)?)?;
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════
// `hard` — playout-selected positions, playout-set gold
// ═══════════════════════════════════════════════════════════════════

const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;
/// Salt for the per-playout re-deal (same constant as `book_position_probe`
/// so the two tools re-deal identically for the same playout seed).
const DETERMINIZE_SALT: u64 = 0xDEA1_0000_0000_0001;

/// Seed of playout `t` at a position. Same derivation as
/// `book_position_probe`, so its per-position playouts pair with the book's.
fn fight_seed(state_hash: u64, t: u32) -> u64 {
    0xF1_6A7_u64
        .wrapping_add(u64::from(t).wrapping_mul(GOLDEN))
        .wrapping_add(state_hash)
}

struct PlayoutOut {
    won: bool,
    final_hp: i32,
    /// The agent's own first move when none was forced.
    first: Option<Action>,
}

/// Let the deployed agent (`oracle_config` is the deployed dialect at
/// `sims`) finish the fight from `start`, after `first` if one is forced.
/// With `determinize`, the hidden information (unknown draw order, hidden
/// RNG streams) is re-dealt first as a pure function of `fight_seed`, so
/// the playouts of one position average over deals instead of replaying
/// the one deal the fixture froze. The per-step search seed is derived
/// from `fight_seed` and the step index, so two playouts with the same seed
/// differ only through the moves they take. 600 actions is a runaway
/// guard, not a game rule.
fn playout(
    start: &CombatState,
    first: Option<&Action>,
    fight_seed: u64,
    sims: u32,
    determinize: bool,
) -> PlayoutOut {
    let mut state = start.clone();
    if determinize {
        sts2_core::blindfold::blindfold_root(&mut state, fight_seed ^ DETERMINIZE_SALT);
    }
    if let Some(a) = first {
        let _ = sts2_core::apply(&mut state, a.clone());
    }
    let mut own_first: Option<Action> = None;
    let mut count: u32 = 0;
    while !is_terminal(&state) && count < 600 {
        if legal_actions(&state).is_empty() {
            let _ = sts2_core::apply(&mut state, Action::EndTurn);
            count += 1;
            continue;
        }
        let step_seed = fight_seed ^ u64::from(count).wrapping_mul(GOLDEN);
        let mut m = Mcts::new(state.clone(), oracle_config(sims, step_seed));
        let a = m.search_n(sims);
        if first.is_none() && own_first.is_none() {
            own_first = Some(a.clone());
        }
        let _ = sts2_core::apply(&mut state, a);
        count += 1;
    }
    PlayoutOut {
        won: is_terminal(&state) && state.player_won,
        final_hp: state.player.current_hp.max(0),
        first: own_first,
    }
}

/// Identity of a move for the gold table: a card play is its card (id,
/// upgrade, current cost) and target, not its hand slot, so two copies of
/// Strike are one move. Everything else is its debug rendering.
fn move_key(state: &CombatState, a: &Action) -> String {
    if let Action::PlayCard { hand_index, target } = a {
        let card = state
            .player
            .hand
            .get(usize::from(*hand_index))
            .and_then(|uid| state.cards.iter().find(|c| c.uid == *uid));
        if let Some(c) = card {
            return format!("card:{:?}:{}:{}:{target:?}", c.card_id, c.upgraded, c.cost);
        }
    }
    format!("{a:?}")
}

/// Distinct legal first moves, in `legal_actions` order: the representative
/// action plus the debug renderings of its aliases.
fn distinct_moves(state: &CombatState) -> Vec<(Action, Vec<String>)> {
    let mut out: Vec<(String, Action, Vec<String>)> = Vec::new();
    for a in legal_actions(state) {
        let key = move_key(state, &a);
        if let Some(slot) = out.iter_mut().find(|(k, _, _)| *k == key) {
            slot.2.push(format!("{a:?}"));
        } else {
            out.push((key, a, Vec::new()));
        }
    }
    out.into_iter().map(|(_, a, al)| (a, al)).collect()
}

#[derive(Debug, Clone, Copy)]
struct HardOpts {
    sims: u32,
    screen: u32,
    gold: u32,
    refine: u32,
    within: u32,
    band_lo: f64,
    band_hi: f64,
    determinize: bool,
}

/// Rank playout records best-first: win rate, then mean end HP.
fn rank_children(table: &mut [PlayoutChild]) {
    table.sort_by(|a, b| {
        b.win_rate()
            .partial_cmp(&a.win_rate())
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                b.mean_hp
                    .partial_cmp(&a.mean_hp)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });
}

/// Margin (pp) and z between the top two rows of a ranked table. A table
/// with one row has an infinite margin in spirit; report 100pp / z 99.
fn margin_of(table: &[PlayoutChild]) -> (f64, f64) {
    let Some(g) = table.first() else {
        return (0.0, 0.0);
    };
    let Some(r) = table.get(1) else {
        return (100.0, 99.0);
    };
    let (p0, p1) = (g.win_rate(), r.win_rate());
    let var = p0 * (1.0 - p0) / f64::from(g.n.max(1)) + p1 * (1.0 - p1) / f64::from(r.n.max(1));
    let margin = 100.0 * (p0 - p1);
    let z = if var > 0.0 {
        ((p0 - p1) / var.sqrt()).min(99.0)
    } else if margin > 0.0 {
        99.0
    } else {
        0.0
    };
    (margin, z)
}

/// Stage 2 for one kept position: every distinct first move forced
/// `opts.gold` times, then the moves within `opts.within` wins of the
/// leader (always at least the top two) get `opts.refine` more. Returns
/// the ranked table.
fn gold_by_playout(state: &CombatState, opts: &HardOpts) -> Vec<PlayoutChild> {
    let moves = distinct_moves(state);
    let h = sts2_core::state_hash(state);
    let run = |set: &[usize], t_from: u32, t_to: u32| -> Vec<(usize, PlayoutOut)> {
        let tasks: Vec<(usize, u32)> = set
            .iter()
            .flat_map(|&i| (t_from..t_to).map(move |t| (i, t)))
            .collect();
        tasks
            .par_iter()
            .map(|&(i, t)| {
                (
                    i,
                    playout(
                        state,
                        Some(&moves[i].0),
                        fight_seed(h, t),
                        opts.sims,
                        opts.determinize,
                    ),
                )
            })
            .collect()
    };
    // (n, wins, Σhp)
    let mut acc: Vec<(u32, u32, f64)> = vec![(0, 0, 0.0); moves.len()];
    let fold = |acc: &mut Vec<(u32, u32, f64)>, outs: Vec<(usize, PlayoutOut)>| {
        for (i, o) in outs {
            acc[i].0 += 1;
            acc[i].1 += u32::from(o.won);
            acc[i].2 += f64::from(o.final_hp);
        }
    };
    let all: Vec<usize> = (0..moves.len()).collect();
    fold(&mut acc, run(&all, 0, opts.gold));
    if moves.len() > 1 && opts.refine > 0 {
        let mut order: Vec<usize> = all.clone();
        order.sort_by(|&a, &b| {
            acc[b].1.cmp(&acc[a].1).then_with(|| {
                acc[b]
                    .2
                    .partial_cmp(&acc[a].2)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        });
        let best = acc[order[0]].1;
        let mut set: Vec<usize> = order
            .iter()
            .copied()
            .filter(|&i| acc[i].1 + opts.within >= best)
            .collect();
        if set.len() < 2 {
            set = order[..2].to_vec();
        }
        fold(&mut acc, run(&set, opts.gold, opts.gold + opts.refine));
    }
    #[allow(clippy::cast_possible_truncation)]
    let mut table: Vec<PlayoutChild> = moves
        .iter()
        .zip(acc.iter())
        .map(|((a, aliases), &(n, wins, hp))| PlayoutChild {
            action: format!("{a:?}"),
            aliases: aliases.clone(),
            n,
            wins,
            mean_hp: if n == 0 {
                0.0
            } else {
                (hp / f64::from(n)) as f32
            },
        })
        .collect();
    rank_children(&mut table);
    table
}

/// Water-filling share of `target` over buckets with `avail[i]` items: every
/// bucket gets an equal share, capped at what it has, and the slack is
/// redistributed to the buckets that can take it.
fn fair_share(avail: &[usize], target: usize) -> Vec<usize> {
    let mut out = vec![0usize; avail.len()];
    let mut remaining = target.min(avail.iter().sum());
    loop {
        let open: Vec<usize> = (0..avail.len()).filter(|&i| out[i] < avail[i]).collect();
        if open.is_empty() || remaining == 0 {
            return out;
        }
        let each = (remaining / open.len()).max(1);
        for &i in &open {
            if remaining == 0 {
                break;
            }
            let take = each.min(avail[i] - out[i]).min(remaining);
            out[i] += take;
            remaining -= take;
        }
    }
}

/// A screened candidate waiting for the gold stage.
struct Screened {
    character: String,
    harvest: HarvestEntry,
    rel: String,
    ascension: u8,
    state: CombatState,
    screen_wins: u32,
    first_moves: BTreeMap<String, u32>,
}

#[allow(clippy::too_many_lines)]
fn cmd_hard(args: &[String]) -> std::io::Result<()> {
    let mut pos: Vec<&String> = Vec::new();
    let mut n_common = 40usize;
    let mut n_elite = 60usize;
    let mut n_boss = 60usize;
    let mut target = 300usize;
    let mut max_warmup_plies = 8u32;
    let mut chunk = 32usize;
    let mut opts = HardOpts {
        sims: 1000,
        screen: 16,
        gold: 24,
        refine: 72,
        within: 2,
        band_lo: 0.2,
        band_hi: 0.8,
        determinize: true,
    };
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut val = |name: &str| -> String {
            it.next()
                .cloned()
                .unwrap_or_else(|| panic!("{name} needs a value"))
        };
        match a.as_str() {
            "--common" => n_common = val(a).parse().expect("--common"),
            "--elite" => n_elite = val(a).parse().expect("--elite"),
            "--boss" => n_boss = val(a).parse().expect("--boss"),
            "--target" => target = val(a).parse().expect("--target"),
            "--screen" => opts.screen = val(a).parse().expect("--screen"),
            "--gold" => opts.gold = val(a).parse().expect("--gold"),
            "--refine" => opts.refine = val(a).parse().expect("--refine"),
            "--within" => opts.within = val(a).parse().expect("--within"),
            "--sims" => opts.sims = val(a).parse().expect("--sims"),
            "--warmup" => max_warmup_plies = val(a).parse().expect("--warmup"),
            "--chunk" => chunk = val(a).parse::<usize>().expect("--chunk").max(1),
            "--no-determinize" => opts.determinize = false,
            "--band" => {
                let v = val(a);
                let (lo, hi) = v.split_once(',').expect("--band lo,hi");
                opts.band_lo = lo.parse().expect("--band lo");
                opts.band_hi = hi.parse().expect("--band hi");
            }
            _ => pos.push(a),
        }
    }
    let pool_root = PathBuf::from(pos.first().expect("hard <pool_root> <out_json>"));
    let out_path = PathBuf::from(pos.get(1).expect("hard <pool_root> <out_json>"));
    let pool_root = pool_root.canonicalize().unwrap_or(pool_root);

    // ── Pools: `the-*` under the root, or one level below it ────────
    let mut pool_dirs: Vec<PathBuf> = Vec::new();
    let walk = |dir: &Path, pool_dirs: &mut Vec<PathBuf>| -> std::io::Result<()> {
        for e in std::fs::read_dir(dir)? {
            let p = e?.path();
            let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
            if p.is_dir() && name.starts_with("the-") {
                pool_dirs.push(p);
            }
        }
        Ok(())
    };
    walk(&pool_root, &mut pool_dirs)?;
    for e in std::fs::read_dir(&pool_root)? {
        let p = e?.path();
        let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
        if p.is_dir() && !name.starts_with("the-") && !name.starts_with('.') {
            walk(&p, &mut pool_dirs)?;
        }
    }
    pool_dirs.sort();
    assert!(
        !pool_dirs.is_empty(),
        "no `the-<character>` pool under {} (or one level below); \
         `just harvest-fixtures-hard` writes it",
        pool_root.display()
    );

    // ── Candidates ──────────────────────────────────────────────────
    let mut candidates: Vec<(String, HarvestEntry, String)> = Vec::new();
    let mut scored_by_bucket: BTreeMap<String, usize> = BTreeMap::new();
    for dir in &pool_dirs {
        let character = dir
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .trim_start_matches("the-")
            .to_string();
        let sub = dir
            .strip_prefix(&pool_root)
            .map_or_else(|_| format!("the-{character}"), |p| p.display().to_string());
        let Ok(bytes) = std::fs::read(dir.join("corpus.json")) else {
            eprintln!("[hard] {sub}: no corpus.json, skipping");
            continue;
        };
        let corpus: HarvestCorpus = match serde_json::from_slice(&bytes) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[hard] {sub}: corpus.json unparseable ({e}), skipping");
                continue;
            }
        };
        for (tier, want) in [("common", n_common), ("elite", n_elite), ("boss", n_boss)] {
            let pool: Vec<HarvestEntry> = corpus
                .entries
                .iter()
                .filter(|e| e.tier == tier)
                .cloned()
                .collect();
            let picked = stride_sample(&pool, want);
            *scored_by_bucket
                .entry(format!("{character}/{tier}"))
                .or_default() += picked.len();
            for entry in picked {
                let file = Path::new(&entry.fixture_path)
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
                    .to_string();
                candidates.push((character.clone(), entry, format!("{sub}/{file}")));
            }
        }
    }
    eprintln!(
        "[hard] {} candidates from {} pools under {}; screen {} playouts x {} sims, band ({}, {})",
        candidates.len(),
        pool_dirs.len(),
        pool_root.display(),
        opts.screen,
        opts.sims,
        opts.band_lo,
        opts.band_hi
    );

    std::panic::set_hook(Box::new(|_| {}));
    let realized: Vec<(String, HarvestEntry, String, u8, CombatState)> = candidates
        .into_par_iter()
        .filter_map(|(character, entry, rel)| {
            let path = pool_root.join(&rel);
            let state = std::panic::catch_unwind(|| realize(&path, entry.seed, max_warmup_plies))
                .ok()
                .flatten()?;
            let asc = common::fixture_ascension(&path);
            Some((character, entry, rel, asc, state))
        })
        .collect();
    eprintln!(
        "[hard] {} realized to a live decision point",
        realized.len()
    );

    // ── Stage 1: screen ─────────────────────────────────────────────
    let t0 = Instant::now();
    let done = std::sync::atomic::AtomicUsize::new(0);
    println!("SCREEN\tbucket\tascension\tfixture\twins\tn\tfirst_moves");
    let mut screened: Vec<Screened> = realized
        .into_par_iter()
        .map(|(character, harvest, rel, ascension, state)| {
            let h = sts2_core::state_hash(&state);
            let outs: Vec<PlayoutOut> = (0..opts.screen)
                .into_par_iter()
                .map(|t| playout(&state, None, fight_seed(h, t), opts.sims, opts.determinize))
                .collect();
            let wins = outs.iter().filter(|o| o.won).count();
            let mut first_moves: BTreeMap<String, u32> = BTreeMap::new();
            for o in &outs {
                if let Some(a) = &o.first {
                    *first_moves.entry(format!("{a:?}")).or_default() += 1;
                }
            }
            let k = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if k.is_multiple_of(25) {
                eprintln!("[hard] screen {k} done  {:.0}s", t0.elapsed().as_secs_f64());
            }
            #[allow(clippy::cast_possible_truncation)]
            Screened {
                character,
                harvest,
                rel,
                ascension,
                state,
                screen_wins: wins as u32,
                first_moves,
            }
        })
        .collect();
    screened.sort_by(|a, b| a.rel.cmp(&b.rel));
    for s in &screened {
        println!(
            "SCREEN\t{}/{}\t{}\t{}\t{}\t{}\t{}",
            s.character,
            s.harvest.tier,
            s.ascension,
            s.rel,
            s.screen_wins,
            opts.screen,
            s.first_moves
                .iter()
                .map(|(a, n)| format!("{a}={n}"))
                .collect::<Vec<_>>()
                .join(";")
        );
    }
    eprintln!("[hard] screen done in {:.0}s", t0.elapsed().as_secs_f64());

    let in_band = |s: &Screened| {
        let p = f64::from(s.screen_wins) / f64::from(opts.screen.max(1));
        p > opts.band_lo && p < opts.band_hi
    };
    let mut in_band_by_bucket: BTreeMap<String, usize> = BTreeMap::new();
    let mut buckets: BTreeMap<String, Vec<Screened>> = BTreeMap::new();
    let mut n_easy = 0usize;
    let mut n_lost = 0usize;
    for s in screened {
        if in_band(&s) {
            let key = format!("{}/{}", s.character, s.harvest.tier);
            *in_band_by_bucket.entry(key.clone()).or_default() += 1;
            buckets.entry(key).or_default().push(s);
        } else if f64::from(s.screen_wins) / f64::from(opts.screen.max(1)) >= opts.band_hi {
            n_easy += 1;
        } else {
            n_lost += 1;
        }
    }
    let total_in_band: usize = in_band_by_bucket.values().sum();
    eprintln!(
        "[hard] in band: {total_in_band}  (above: {n_easy}, below: {n_lost}); target {target}"
    );

    // ── Stratified cap: equal share per tier, then per character ────
    let mut kept: Vec<Screened> = Vec::new();
    if target == 0 || total_in_band <= target {
        for (_, v) in buckets {
            kept.extend(v);
        }
    } else {
        let tiers: Vec<String> = {
            let mut t: Vec<String> = buckets
                .keys()
                .map(|k| k.split('/').nth(1).unwrap_or("").to_string())
                .collect();
            t.sort();
            t.dedup();
            t
        };
        let tier_avail: Vec<usize> = tiers
            .iter()
            .map(|t| {
                buckets
                    .iter()
                    .filter(|(k, _)| k.ends_with(&format!("/{t}")))
                    .map(|(_, v)| v.len())
                    .sum()
            })
            .collect();
        let tier_share = fair_share(&tier_avail, target);
        for (ti, tier) in tiers.iter().enumerate() {
            let keys: Vec<String> = buckets
                .keys()
                .filter(|k| k.ends_with(&format!("/{tier}")))
                .cloned()
                .collect();
            let avail: Vec<usize> = keys.iter().map(|k| buckets[k].len()).collect();
            let share = fair_share(&avail, tier_share[ti]);
            for (ki, key) in keys.iter().enumerate() {
                let mut v = buckets.remove(key).unwrap_or_default();
                // Spread the take across the band instead of clustering at
                // one win rate.
                v.sort_by_key(|s| s.screen_wins);
                let n = share[ki];
                if n == 0 || v.is_empty() {
                    continue;
                }
                #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
                let step = v.len() as f64 / n as f64;
                let idx: Vec<usize> = (0..n)
                    .map(|i| ((i as f64 * step) as usize).min(v.len() - 1))
                    .collect();
                for (j, s) in v.into_iter().enumerate() {
                    if idx.contains(&j) {
                        kept.push(s);
                    }
                }
            }
        }
    }
    kept.sort_by(|a, b| {
        a.character
            .cmp(&b.character)
            .then_with(|| a.harvest.tier.cmp(&b.harvest.tier))
            .then_with(|| a.rel.cmp(&b.rel))
    });
    eprintln!(
        "[hard] gold stage on {} positions: {} playouts per distinct move + {} refine, chunks of {chunk}",
        kept.len(),
        opts.gold,
        opts.refine
    );

    // ── Stage 2: gold by playout, written after every chunk ─────────
    let write_book = |entries: &[BookEntry], complete: bool| -> std::io::Result<()> {
        let chance = if entries.is_empty() {
            0.0
        } else {
            #[allow(clippy::cast_precision_loss)]
            let s: f64 = entries.iter().map(|e| 1.0 / e.legal_actions as f64).sum();
            #[allow(clippy::cast_precision_loss)]
            {
                s / entries.len() as f64
            }
        };
        let book = Book {
            version: 1,
            git_sha: common::git_short_sha(),
            generator: if complete {
                "combat_position_book hard".to_string()
            } else {
                "combat_position_book hard (PARTIAL — build interrupted)".to_string()
            },
            oracle_sims: 0,
            oracle_seeds: 0,
            oracle_depth: ORACLE_DEPTH,
            max_warmup_plies,
            hidden_info: "beliefdraw".to_string(),
            c_puct: C_PUCT,
            draw_widening_c: DRAW_WIDENING_C,
            scored_by_bucket: scored_by_bucket.clone(),
            chance_agreement: chance,
            annotated_with_proofs: 0,
            gold_contradicted: 0,
            oracle_q_entries: 0,
            gold_method: "playout".to_string(),
            playout: Some(PlayoutParams {
                sims: opts.sims,
                screen_playouts: opts.screen,
                gold_playouts: opts.gold,
                refine_playouts: opts.refine,
                refine_within_wins: opts.within,
                band_lo: opts.band_lo,
                band_hi: opts.band_hi,
                determinized: opts.determinize,
                in_band_by_bucket: in_band_by_bucket.clone(),
            }),
            entries: entries.to_vec(),
        };
        if let Some(parent) = out_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&out_path, serde_json::to_vec_pretty(&book)?)
    };

    let t1 = Instant::now();
    let mut entries: Vec<BookEntry> = Vec::new();
    println!("GOLD\tbook_id\tbucket\tasc\tfixture\tscreen_wins\tgold\tgold_win\trunner_up\trunner_win\tmargin_pp\tz\tmoves");
    for (ci, group) in kept.chunks(chunk).enumerate() {
        let tables: Vec<Vec<PlayoutChild>> = group
            .par_iter()
            .map(|s| gold_by_playout(&s.state, &opts))
            .collect();
        for (s, table) in group.iter().zip(tables) {
            let Some(gold) = table.first() else {
                continue;
            };
            let (margin_pp, z) = margin_of(&table);
            let has_proof = !common::dominance::pairs_at(&s.state).1.is_empty();
            let motifs = motif_tags(&s.state, &gold.action, has_proof);
            #[allow(clippy::cast_possible_truncation)]
            let book_id = entries.len() as u32;
            println!(
                "GOLD\t{book_id}\t{}/{}\t{}\t{}\t{}\t{}\t{}/{}\t{}\t{}/{}\t{margin_pp:.1}\t{z:.2}\t{}",
                s.character,
                s.harvest.tier,
                s.ascension,
                s.rel,
                s.screen_wins,
                gold.action,
                gold.wins,
                gold.n,
                table.get(1).map_or("-", |r| r.action.as_str()),
                table.get(1).map_or(0, |r| r.wins),
                table.get(1).map_or(0, |r| r.n),
                table.len()
            );
            entries.push(BookEntry {
                book_id,
                character: s.character.clone(),
                tier: s.harvest.tier.clone(),
                floor: s.harvest.floor,
                fixture_rel_path: s.rel.clone(),
                seed: s.harvest.seed,
                legal_actions: legal_actions(&s.state).len(),
                state_hash: sts2_core::state_hash(&s.state),
                gold_action: gold.action.clone(),
                proven_pairs: Vec::new(),
                gold_is_proven_dominated: false,
                motifs,
                oracle_q: Vec::new(),
                gold_aliases: gold.aliases.clone(),
                ascension: s.ascension,
                playout: Some(EntryPlayout {
                    screen_n: opts.screen,
                    screen_wins: s.screen_wins,
                    screen_first_moves: s.first_moves.clone(),
                    table,
                    margin_pp,
                    z,
                }),
            });
        }
        let last = (ci + 1) * chunk >= kept.len();
        write_book(&entries, last)?;
        eprintln!(
            "[hard] gold chunk {}/{} written ({} entries)  {:.0}s",
            ci + 1,
            kept.len().div_ceil(chunk),
            entries.len(),
            t1.elapsed().as_secs_f64()
        );
    }
    if kept.is_empty() {
        write_book(&entries, true)?;
    }
    let _ = std::panic::take_hook();

    // ── Report ──────────────────────────────────────────────────────
    let mut by_bucket: BTreeMap<String, usize> = BTreeMap::new();
    let mut margins: Vec<f64> = Vec::new();
    let mut agent_first_is_gold = 0.0_f64;
    let mut agent_first_n = 0.0_f64;
    let mut resolved_10 = 0usize;
    for e in &entries {
        *by_bucket
            .entry(format!("{}/{}", e.character, e.tier))
            .or_default() += 1;
        if let Some(p) = &e.playout {
            margins.push(p.margin_pp);
            if p.margin_pp >= 10.0 {
                resolved_10 += 1;
            }
            for (a, n) in &p.screen_first_moves {
                agent_first_n += f64::from(*n);
                if *a == e.gold_action || e.gold_aliases.contains(a) {
                    agent_first_is_gold += f64::from(*n);
                }
            }
        }
    }
    margins.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q = |f: f64| -> f64 {
        if margins.is_empty() {
            return 0.0;
        }
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss
        )]
        let idx = ((margins.len() - 1) as f64 * f).round() as usize;
        margins[idx]
    };
    println!("\n=== hard position book ===");
    println!(
        "screened      : {}",
        scored_by_bucket.values().sum::<usize>()
    );
    println!(
        "in band       : {total_in_band} ({:.1}% of screened; {n_easy} above, {n_lost} below)",
        pct(total_in_band, scored_by_bucket.values().sum::<usize>())
    );
    println!("kept          : {}", entries.len());
    println!(
        "margin (pp)   : median {:.1}  p25 {:.1}  p75 {:.1};  {} of {} at >= 10pp",
        q(0.5),
        q(0.25),
        q(0.75),
        resolved_10,
        entries.len()
    );
    println!(
        "agent first move = gold (screen, free play, {} sims): {:.3}",
        opts.sims,
        if agent_first_n > 0.0 {
            agent_first_is_gold / agent_first_n
        } else {
            0.0
        }
    );
    println!("written       : {}", out_path.display());
    println!("\nkept / in band / screened by bucket:");
    for (bucket, scored) in &scored_by_bucket {
        println!(
            "  {bucket:<24} {:>4} / {:>4} / {scored}",
            by_bucket.get(bucket).copied().unwrap_or(0),
            in_band_by_bucket.get(bucket).copied().unwrap_or(0)
        );
    }
    Ok(())
}

/// Why an entry carries no proof. "No fixture" and "no pairs" produce
/// identical JSON and must not produce identical output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Skip {
    None,
    Unrealized,
    HashMiss,
}

#[allow(clippy::cast_precision_loss)]
fn pct(n: usize, d: usize) -> f64 {
    if d == 0 {
        0.0
    } else {
        100.0 * n as f64 / d as f64
    }
}

fn main() -> std::io::Result<()> {
    sts2_content::cards::dispatch::ensure_installed();
    sts2_content::relics::dispatch::ensure_installed();
    sts2_content::potions::dispatch::ensure_installed();

    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("annotate") {
        return cmd_annotate(&args[1..]);
    }
    if args.first().map(String::as_str) == Some("oracle-q") {
        return cmd_oracle_q(&args[1..]);
    }
    if args.first().map(String::as_str) == Some("hard") {
        return cmd_hard(&args[1..]);
    }
    let mut it = args.iter();
    let pool_root = PathBuf::from(
        it.next()
            .cloned()
            .unwrap_or_else(|| "data/run-fixtures".to_string()),
    );
    let out_path = PathBuf::from(
        it.next()
            .cloned()
            .unwrap_or_else(|| "data/combat_position_book.json".to_string()),
    );
    let n_common: usize = it.next().and_then(|s| s.parse().ok()).unwrap_or(20);
    let n_elite: usize = it.next().and_then(|s| s.parse().ok()).unwrap_or(40);
    let n_boss: usize = it.next().and_then(|s| s.parse().ok()).unwrap_or(20);
    let oracle_sims: u32 = it.next().and_then(|s| s.parse().ok()).unwrap_or(50_000);
    let oracle_seeds: u32 = it.next().and_then(|s| s.parse().ok()).unwrap_or(3);
    // Max plies of random play before the position is scored. 0 reproduces the
    // v1 turn-opening book; the default spreads entries across the turn cycle
    // so end-turn decisions actually appear.
    let max_warmup_plies: u32 = it.next().and_then(|s| s.parse().ok()).unwrap_or(8);

    // Realizing a fixture exercises card/relic dispatch on content the
    // synthetic path never touches; a port gap panics rather than
    // erroring. The harvest should have filtered those, but suppress the
    // default handler's stderr spew so a stray one doesn't bury the log.
    std::panic::set_hook(Box::new(|_| {}));

    // ── Collect + stratify ──────────────────────────────────────────
    let mut candidates: Vec<(String, HarvestEntry)> = Vec::new();
    let mut scored_by_bucket: BTreeMap<String, usize> = BTreeMap::new();
    // Only `the-<character>` directories are pools. Dot-prefixed siblings are
    // set-aside partial harvests; counting them inflates the "N characters"
    // line and prints a skip warning per abandoned run.
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

    for dir in &pool_dirs {
        let character = dir
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .trim_start_matches("the-")
            .to_string();
        let corpus_json = dir.join("corpus.json");
        let Ok(bytes) = std::fs::read(&corpus_json) else {
            eprintln!("[book] {character}: no corpus.json, skipping");
            continue;
        };
        let corpus: HarvestCorpus = match serde_json::from_slice(&bytes) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[book] {character}: corpus.json unparseable ({e}), skipping");
                continue;
            }
        };
        for (tier, target) in [("common", n_common), ("elite", n_elite), ("boss", n_boss)] {
            let pool: Vec<HarvestEntry> = corpus
                .entries
                .iter()
                .filter(|e| e.tier == tier)
                .cloned()
                .collect();
            let picked = stride_sample(&pool, target);
            if picked.len() < target {
                eprintln!(
                    "[book] {character}/{tier}: pool has {}, wanted {target} — taking all",
                    pool.len()
                );
            }
            scored_by_bucket.insert(format!("{character}/{tier}"), picked.len());
            for entry in picked {
                candidates.push((character.clone(), entry));
            }
        }
    }
    eprintln!(
        "[book] {} candidate positions from {} characters; oracle {oracle_sims} sims x \
         {oracle_seeds} seeds x 2 methods",
        candidates.len(),
        pool_dirs.len()
    );

    // ── Realize ─────────────────────────────────────────────────────
    let realized: Vec<(String, HarvestEntry, CombatState)> = candidates
        .into_par_iter()
        .filter_map(|(character, entry)| {
            let path = PathBuf::from(&entry.fixture_path);
            let state = std::panic::catch_unwind(|| realize(&path, entry.seed, max_warmup_plies))
                .ok()
                .flatten()?;
            Some((character, entry, state))
        })
        .collect();
    eprintln!(
        "[book] {} realized to a live decision point",
        realized.len()
    );

    // ── Double oracle ───────────────────────────────────────────────
    // `workers == 1` makes `par_search` a thin wrapper over
    // `Mcts::search`, so one call site covers the sequential and the
    // root-parallel oracle.
    let t0 = Instant::now();
    let tasks: Vec<(usize, u32, u32)> = (0..realized.len())
        .flat_map(|p| {
            [1_u32, 8_u32]
                .into_iter()
                .flat_map(move |w| (0..oracle_seeds).map(move |k| (p, w, k)))
        })
        .collect();
    // Progress on stderr every ~2% of tasks: a wide build is an overnight
    // job, and a log that says nothing between "oracle pass" and "done" is
    // indistinguishable from a hung one.
    let done = std::sync::atomic::AtomicUsize::new(0);
    let every = (tasks.len() / 50).max(1);
    let picks = run_oracle_tasks(&tasks, |p| &realized[p].2, oracle_sims, &done, every, &t0);
    eprintln!(
        "[book] oracle pass done in {:.1}s",
        t0.elapsed().as_secs_f64()
    );

    let mut seq: Vec<BTreeMap<String, u32>> = vec![BTreeMap::new(); realized.len()];
    let mut par: Vec<BTreeMap<String, u32>> = vec![BTreeMap::new(); realized.len()];
    let mut runs: Vec<Vec<Vec<sts2_mcts::RootOutcome>>> = vec![Vec::new(); realized.len()];
    for (p, w, pick, outcomes) in picks {
        let slot = if w == 1 { &mut seq[p] } else { &mut par[p] };
        *slot.entry(pick).or_default() += 1;
        runs[p].push(outcomes);
    }

    // ── Gold filter ─────────────────────────────────────────────────
    let mut entries: Vec<BookEntry> = Vec::new();
    let mut chance_sum = 0.0_f64;
    for (p, (character, harvest, state)) in realized.iter().enumerate() {
        let unanimous = |m: &BTreeMap<String, u32>| -> Option<String> {
            let (action, count) = m.iter().max_by_key(|(_, v)| **v)?;
            (*count == oracle_seeds).then(|| action.clone())
        };
        let (Some(s), Some(q)) = (unanimous(&seq[p]), unanimous(&par[p])) else {
            continue;
        };
        if s != q {
            continue;
        }
        let n_legal = legal_actions(state).len();
        #[allow(clippy::cast_precision_loss)]
        {
            chance_sum += 1.0 / n_legal as f64;
        }
        let rel = PathBuf::from(&harvest.fixture_path)
            .strip_prefix(&pool_root)
            .map_or_else(
                |_| {
                    // The harvest writes absolute paths, and the pool may have
                    // been moved since. Fall back to the last two components
                    // (`the-<char>/<fixture>.toml`) — what the relative form
                    // would have been.
                    let p = PathBuf::from(&harvest.fixture_path);
                    let file = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
                    format!("the-{character}/{file}")
                },
                |p| p.display().to_string(),
            );
        // Tags are written at build time so a fresh book is scorable per
        // motif without a second pass; `annotate` refreshes them together
        // with the proofs. `proven_pair` needs the dominance enumeration,
        // which costs milliseconds per position against the oracle's
        // minutes, so it is cheap to do here too.
        let has_proof = !common::dominance::pairs_at(state).1.is_empty();
        let motifs = motif_tags(state, &s, has_proof);
        #[allow(clippy::cast_possible_truncation)]
        entries.push(BookEntry {
            book_id: entries.len() as u32,
            character: character.clone(),
            tier: harvest.tier.clone(),
            floor: harvest.floor,
            fixture_rel_path: rel,
            seed: harvest.seed,
            legal_actions: n_legal,
            state_hash: sts2_core::state_hash(state),
            gold_action: s,
            proven_pairs: Vec::new(),
            gold_is_proven_dominated: false,
            motifs,
            oracle_q: pool_outcomes(&runs[p]),
            gold_aliases: Vec::new(),
            ascension: common::fixture_ascension(Path::new(&harvest.fixture_path)),
            playout: None,
        });
    }
    let oracle_q_entries = entries.iter().filter(|e| !e.oracle_q.is_empty()).count();

    #[allow(clippy::cast_precision_loss)]
    let chance = if entries.is_empty() {
        0.0
    } else {
        chance_sum / entries.len() as f64
    };
    let book = Book {
        version: 1,
        git_sha: common::git_short_sha(),
        generator: "combat_position_book".to_string(),
        oracle_sims,
        oracle_seeds,
        oracle_depth: ORACLE_DEPTH,
        max_warmup_plies,
        hidden_info: "beliefdraw".to_string(),
        c_puct: C_PUCT,
        draw_widening_c: DRAW_WIDENING_C,
        scored_by_bucket,
        chance_agreement: chance,
        // Written by `annotate`, which is a separate pass so the oracle
        // does not have to be re-paid to add proofs to an existing book.
        annotated_with_proofs: 0,
        gold_contradicted: 0,
        oracle_q_entries,
        gold_method: "oracle_unanimous".to_string(),
        playout: None,
        entries,
    };

    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&out_path, serde_json::to_vec_pretty(&book)?)?;

    // ── Report ──────────────────────────────────────────────────────
    let mut by_bucket: BTreeMap<String, usize> = BTreeMap::new();
    for e in &book.entries {
        *by_bucket
            .entry(format!("{}/{}", e.character, e.tier))
            .or_default() += 1;
    }
    println!("\n=== combat position book ===");
    println!("realized      : {}", realized.len());
    println!(
        "gold          : {} ({:.1}% of realized survived both oracles)",
        book.entries.len(),
        100.0 * book.entries.len() as f64 / realized.len().max(1) as f64
    );
    println!("chance_agr    : {:.3}", book.chance_agreement);
    println!("written       : {}", out_path.display());
    println!("\nkept by bucket:");
    for (bucket, n) in &by_bucket {
        let scored = book.scored_by_bucket.get(bucket).copied().unwrap_or(0);
        println!("  {bucket:<24} {n:>4} / {scored}");
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // exact literals in, exact literals out
mod tests {
    use super::*;
    use sts2_core::Action;
    use sts2_mcts::RootOutcome;

    fn outcome(action: Action, visits: u32, values: &[f32]) -> RootOutcome {
        assert_eq!(values.len(), visits as usize);
        let value_sum: f32 = values.iter().sum();
        #[allow(clippy::cast_precision_loss)]
        RootOutcome {
            action,
            visits,
            wins: 0,
            losses: 0,
            ties: visits,
            value_sum,
            value_sq_sum: values.iter().map(|v| v * v).sum(),
            hp_sum: 0.0,
            hp_sq_sum: 0.0,
            hp_n: 0,
            mean_value: value_sum / visits as f32,
        }
    }

    #[test]
    fn pooling_sums_visits_and_recomputes_the_mean_from_the_pooled_sums() {
        let play = Action::PlayCard {
            hand_index: 0,
            target: None,
        };
        // Two searches of the same position; the second saw `play` twice
        // as often, so the pooled mean is visit-weighted, not a mean of
        // means (which would be 0.5).
        let runs = vec![
            vec![
                outcome(play.clone(), 2, &[0.0, 0.0]),
                outcome(Action::EndTurn, 2, &[0.2, 0.2]),
            ],
            vec![
                outcome(play.clone(), 4, &[1.0, 1.0, 1.0, 1.0]),
                outcome(Action::EndTurn, 1, &[0.2]),
            ],
        ];
        let table = pool_outcomes(&runs);
        assert_eq!(table.len(), 2);
        // Best-first: `play` pooled to 4/6 ≈ 0.667 beats EndTurn at 0.2.
        assert_eq!(table[0].action, format!("{play:?}"));
        assert_eq!(table[0].visits, 6);
        assert!((table[0].q - 4.0 / 6.0).abs() < 1e-6, "{}", table[0].q);
        // var = E[v²] − mean² = 4/6 − (4/6)² = 2/9; se = sqrt(var / 6)
        let se = (2.0_f32 / 9.0 / 6.0).sqrt();
        assert!(
            (table[0].stderr - se).abs() < 1e-5,
            "{} vs {se}",
            table[0].stderr
        );
        // Constant returns pool to (numerically) zero stderr: the sums
        // arrive as f32 from the tree, so `Σv²/n − mean²` is ~1e-9, not 0.
        assert_eq!(table[1].action, format!("{:?}", Action::EndTurn));
        assert_eq!(table[1].visits, 3);
        assert!(table[1].stderr.abs() < 1e-4, "{}", table[1].stderr);
    }

    #[test]
    fn a_child_one_search_never_visited_still_appears_once() {
        let runs = vec![
            vec![outcome(Action::EndTurn, 1, &[0.5])],
            vec![
                outcome(Action::EndTurn, 1, &[0.5]),
                outcome(
                    Action::PlayCard {
                        hand_index: 1,
                        target: Some(0),
                    },
                    1,
                    &[-1.0],
                ),
            ],
        ];
        let table = pool_outcomes(&runs);
        assert_eq!(table.len(), 2);
        assert_eq!(table[0].action, format!("{:?}", Action::EndTurn));
        assert_eq!(table[0].visits, 2);
        assert_eq!(table[1].visits, 1);
        assert_eq!(table[1].q, -1.0);
    }

    #[test]
    fn fair_share_water_fills_over_the_buckets_that_have_room() {
        assert_eq!(fair_share(&[100, 100, 100], 300), vec![100, 100, 100]);
        // The short bucket gives everything it has; the slack goes to the
        // others, one at a time once the equal share rounds to zero.
        assert_eq!(fair_share(&[5, 100, 100], 60), vec![5, 28, 27]);
        assert_eq!(fair_share(&[2, 3], 50), vec![2, 3]);
        assert_eq!(fair_share(&[], 5), Vec::<usize>::new());
        assert_eq!(fair_share(&[4, 4], 0), vec![0, 0]);
    }

    fn child(action: &str, n: u32, wins: u32, mean_hp: f32) -> PlayoutChild {
        PlayoutChild {
            action: action.into(),
            aliases: Vec::new(),
            n,
            wins,
            mean_hp,
        }
    }

    #[test]
    fn playout_table_ranks_by_win_rate_then_hp_and_margin_reads_the_top_two() {
        let mut t = vec![
            child("a", 96, 48, 10.0),
            child("b", 96, 60, 5.0),
            // Same rate as `b` on fewer playouts; the HP tiebreak puts it first.
            child("c", 24, 15, 20.0),
        ];
        rank_children(&mut t);
        assert_eq!(t[0].action, "c");
        assert_eq!(t[1].action, "b");
        assert_eq!(t[2].action, "a");
        let (m, z) = margin_of(&t);
        assert!(m.abs() < 1e-9 && z.abs() < 1e-9, "{m} {z}");
        // b vs a: 62.5% − 50% on 96 each.
        let (m, z) = margin_of(&t[1..]);
        assert!((m - 12.5).abs() < 1e-9, "{m}");
        assert!(z > 1.7 && z < 1.8, "{z}");
        assert_eq!(margin_of(&t[..1]), (100.0, 99.0));
        assert_eq!(margin_of(&[]), (0.0, 0.0));
        // Degenerate rates with a gap: z is capped, not NaN.
        let (m, z) = margin_of(&[child("x", 24, 24, 1.0), child("y", 24, 0, 0.0)]);
        assert!(
            (m - 100.0).abs() < 1e-9 && (z - 99.0).abs() < 1e-9,
            "{m} {z}"
        );
    }

    #[test]
    fn hard_book_fields_round_trip_and_stay_absent_on_oracle_entries() {
        let json = r#"{"book_id":7,"character":"ironclad","tier":"boss","floor":16,
            "fixture_rel_path":"the-ironclad/x.toml","seed":1,"legal_actions":3,
            "state_hash":42,"gold_action":"EndTurn","motifs":["incoming_attack"]}"#;
        let e: BookEntry = serde_json::from_str(json).unwrap();
        assert!(e.gold_aliases.is_empty());
        assert_eq!(e.ascension, 0);
        assert!(e.playout.is_none());
        let out = serde_json::to_string(&e).unwrap();
        for absent in ["gold_aliases", "ascension", "playout"] {
            assert!(!out.contains(absent), "{absent} must not be written: {out}");
        }
        let mut e2 = e.clone();
        e2.gold_aliases = vec!["PlayCard { hand_index: 3, target: None }".into()];
        e2.ascension = 10;
        e2.playout = Some(EntryPlayout {
            screen_n: 16,
            screen_wins: 7,
            screen_first_moves: BTreeMap::from([("EndTurn".to_string(), 16)]),
            table: vec![child("EndTurn", 96, 50, 12.5), child("x", 24, 3, 0.0)],
            margin_pp: 39.6,
            z: 4.2,
        });
        let back: BookEntry = serde_json::from_str(&serde_json::to_string(&e2).unwrap()).unwrap();
        assert_eq!(back.gold_aliases, e2.gold_aliases);
        assert_eq!(back.ascension, 10);
        let p = back.playout.unwrap();
        assert_eq!(p.screen_wins, 7);
        assert_eq!(p.table, e2.playout.as_ref().unwrap().table);
        assert!((p.margin_pp - 39.6).abs() < 1e-9);
    }

    #[test]
    fn an_entry_without_oracle_q_round_trips_as_empty() {
        // A v3 book written before the field existed must still parse, and
        // an entry with the field must serialize it back exactly.
        let json = r#"{"book_id":7,"character":"ironclad","tier":"boss","floor":16,
            "fixture_rel_path":"the-ironclad/x.toml","seed":1,"legal_actions":3,
            "state_hash":42,"gold_action":"EndTurn","motifs":["incoming_attack"]}"#;
        let e: BookEntry = serde_json::from_str(json).unwrap();
        assert!(e.oracle_q.is_empty());
        let out = serde_json::to_string(&e).unwrap();
        assert!(
            !out.contains("oracle_q"),
            "empty table must not be written: {out}"
        );
        let mut e2 = e.clone();
        e2.oracle_q = vec![OracleChild {
            action: "EndTurn".into(),
            q: 0.25,
            stderr: 0.01,
            visits: 300_000,
        }];
        let back: BookEntry = serde_json::from_str(&serde_json::to_string(&e2).unwrap()).unwrap();
        assert_eq!(back.oracle_q, e2.oracle_q);
    }
}
