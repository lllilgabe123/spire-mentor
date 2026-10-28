//! Run-level policies: baselines, the MCTS combat bridge, and the M1
//! heuristic agent.
//!
//! Bead `sts2-rs-40ikw.4` (M0, plan §3.3 / §5-M0): the two degenerate
//! meta policies measured as E0 baselines ([`sts2_core::full_run::RandomPolicy`]
//! lives in core; [`FirstOptionPolicy`] here), the frozen combat budget
//! configs ([`CombatBudget`]), and [`MctsCombatPolicy`] — a composite
//! [`RunPolicy`] that delegates non-combat phases to a wrapped meta
//! policy and plays `Phase::Combat` with the existing combat MCTS
//! (single-threaded `search_n`, throughput mode).
//!
//! Bead `sts2-rs-40ikw.5` (M1, plan §5/M1): [`heuristic::HeuristicPolicy`]
//! — the tier-1 all-surfaces heuristic meta policy — plus the
//! [`VScoreSource`] hook that threads per-decision V scores / margins
//! into the §3.6 audit log.

pub mod ablation;
pub mod heuristic;
mod potion_ablation;
mod potion_oc;
mod potion_timing;

/// Env-reading wrapper for [`potion_oc::resolve_potion_swap_flag`] —
/// the arena's `RunManager` construction calls this once per run.
#[must_use]
pub fn resolve_potion_swap_flag_from_env() -> bool {
    potion_oc::resolve_potion_swap_flag(
        std::env::var("STS2_RUNAI_POTION_SWAP").ok().as_deref(),
        std::env::var("STS2_RUNAI_POTION_OC").ok().as_deref(),
    )
}

/// Env-reading wrapper for [`potion_oc::swap_is_rule_side`] — whether a
/// surfaced full-belt swap is decided by the OC value rule (`true`) or by the
/// run policy through its `potion_swap` encoder surface (`false`).
///
/// `ppo/env.py` reads this through PyO3 to gate the mirror of the
/// `PotionSwap` interception in `MetaWithRules::choose`. ONE source of truth,
/// like the flag itself — if the two ever disagree, G1 breaks.
#[must_use]
pub fn potion_swap_is_rule_side_from_env() -> bool {
    potion_oc::swap_is_rule_side(std::env::var("STS2_RUNAI_POTION_OC").ok().as_deref())
}
pub mod sampled_deck;
pub mod sampling;
pub mod surfaces;
pub mod tiered;
pub mod value;

pub use ablation::{AblationPolicy, AblationSurface};
pub use heuristic::HeuristicPolicy;
pub use sampled_deck::SampledDeckPolicy;
pub use tiered::TieredPolicy;
pub use value::ValuePolicy;

use std::sync::{Arc, OnceLock};

use sts2_core::full_run::{RandomPolicy, RunPolicy};
use sts2_core::potions::CharacterId;
use sts2_core::run_manager::{Phase, RunAction, RunManager};
use sts2_core::runstate::RoomType;
use sts2_core::{Action, CombatState};
use sts2_mcts::{LeafEval, LearnedValueNet, Mcts, MctsConfig};

/// Degenerate baseline: always pick option 0. With combat budget `none`
/// this is the "first-option" E0 baseline from plan §5-M0.
#[derive(Clone, Copy, Debug, Default)]
pub struct FirstOptionPolicy;

impl RunPolicy for FirstOptionPolicy {
    fn choose(&mut self, _rm: &RunManager, actions: &[RunAction]) -> Option<usize> {
        if actions.is_empty() {
            None
        } else {
            Some(0)
        }
    }
}

/// Leaf evaluator selector for [`CombatBudget`]. Mirrors the two
/// [`sts2_mcts::LeafEval`] arms the plan freezes (§3.3); kept as a tiny
/// local enum so the budget stays `Copy`/CLI-printable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeafKind {
    /// Narrow 29-feature Texel leaf (`LeafEval::Texel`).
    Texel,
    /// Wide 172-feature Texel leaf (`LeafEval::TexelWide`).
    TexelWide,
}

impl LeafKind {
    pub(crate) fn to_leaf_eval(self) -> LeafEval {
        match self {
            LeafKind::Texel => LeafEval::Texel,
            LeafKind::TexelWide => LeafEval::TexelWide,
        }
    }
}

/// Process-wide learned combat leaf net from
/// `STS2_RUNAI_COMBAT_NET=<weights.json>` (a `value_policy_net.py`
/// trainer artifact — see `LearnedValueNet::from_weights_path`).
/// Unset ⇒ `None` ⇒ production behavior is byte-identical to today.
/// An unreadable/malformed asset PANICS with the load error rather
/// than silently falling back — same rationale as the honest-mode
/// knob: a typo silently reverting to TexelWide would poison a
/// multi-hour campaign without any signal (spec
/// `2026-07-18-hard-example-combat-net-design.md` §1).
///
/// The cache is keyed on the env value (not load-once): the weights
/// file is read once per distinct path, but the env var itself is
/// re-read every call, so a driver may flip the variable mid-process —
/// `fork-fatal` relies on this to replay phase-A prefixes WITHOUT the
/// net (byte-faithful to the corpus) and fork phase B WITH it, the
/// same two-phase contract as `STS2_RUNAI_HONEST`.
pub fn combat_net_from_env() -> Option<Arc<LearnedValueNet>> {
    static CACHE: OnceLock<std::sync::Mutex<Option<(String, Arc<LearnedValueNet>)>>> =
        OnceLock::new();
    let path = std::env::var("STS2_RUNAI_COMBAT_NET").ok()?;
    let mut cache = CACHE
        .get_or_init(|| std::sync::Mutex::new(None))
        .lock()
        .expect("combat net cache lock");
    if let Some((cached_path, net)) = cache.as_ref() {
        if *cached_path == path {
            return Some(Arc::clone(net));
        }
    }
    let net = Arc::new(
        LearnedValueNet::from_weights_path(std::path::Path::new(&path))
            .unwrap_or_else(|e| panic!("STS2_RUNAI_COMBAT_NET={path}: failed to load: {e}")),
    );
    *cache = Some((path, Arc::clone(&net)));
    Some(net)
}

/// The budget's leaf evaluator, overridden by [`combat_net_from_env`]
/// when the env asset is present.
pub fn resolve_leaf_eval(kind: LeafKind) -> LeafEval {
    match combat_net_from_env() {
        Some(net) => LeafEval::Learned(net),
        None => kind.to_leaf_eval(),
    }
}

/// `STS2_RUNAI_COMBAT_PRIOR=1` + a loaded env net ⇒ the net's policy
/// head also drives PUCT priors ([`sts2_mcts::LearnedPolicyPrior`]).
pub(crate) fn combat_prior_net_from_env() -> Option<Arc<LearnedValueNet>> {
    if std::env::var("STS2_RUNAI_COMBAT_PRIOR").ok().as_deref() != Some("1") {
        return None;
    }
    combat_net_from_env()
}

/// What `STS2_RUNAI_PRIOR_NET` asks for (GH #830).
#[derive(Clone, Debug, PartialEq, Eq)]
enum PriorNetEnv {
    /// Unset: the built-in move prior of the fight's character
    /// ([`sts2_mcts::rollout_net::builtin_prior_json`]; only Ironclad has
    /// one), in [`BUILTIN_PRIOR_ROOMS`] unless `STS2_RUNAI_PRIOR_ROOMS` is set,
    /// and only in the run agent's Ironclad layer
    /// ([`MctsCombatPolicy::ironclad_layer_on`]).
    Default,
    /// Exactly `off`: no move prior.
    Off,
    /// Any other value: the playout-net JSON file at that path, for every
    /// character and in every fight unless `STS2_RUNAI_PRIOR_ROOMS` narrows it
    /// (the variable's meaning before the built-in prior existed).
    File(String),
}

/// `STS2_RUNAI_PRIOR_NET`'s value → [`PriorNetEnv`].
///
/// # Panics
/// On an empty value: it would otherwise have to mean one of the three, and
/// a script that meant another would measure the wrong search silently.
fn parse_prior_net_env(value: Option<&str>) -> PriorNetEnv {
    match value {
        None => PriorNetEnv::Default,
        Some("") => panic!(
            "STS2_RUNAI_PRIOR_NET is empty: unset it for the built-in move prior, set it to \
             `off` for none, or name a net file"
        ),
        Some("off") => PriorNetEnv::Off,
        Some(path) => PriorNetEnv::File(path.to_owned()),
    }
}

/// [`parse_prior_net_env`] of the environment, read once per process.
fn prior_net_env() -> &'static PriorNetEnv {
    static ENV: OnceLock<PriorNetEnv> = OnceLock::new();
    ENV.get_or_init(|| parse_prior_net_env(std::env::var("STS2_RUNAI_PRIOR_NET").ok().as_deref()))
}

/// The combat tiers the built-in move prior applies in when
/// `STS2_RUNAI_PRIOR_ROOMS` is unset: the GH #830 bundle gate ran it in elite
/// and boss fights only (on the pro set it helped those and cost a little in
/// hallway fights).
const BUILTIN_PRIOR_ROOMS: [BudgetTier; 2] = [BudgetTier::Elite, BudgetTier::Boss];

/// `STS2_RUNAI_PRIOR_ROOMS=<comma list of common|elite|boss>`: the combat
/// tiers a move prior applies in, read once; `None` when unset (the file
/// prior then applies everywhere, the built-in one in
/// [`BUILTIN_PRIOR_ROOMS`]).
///
/// # Panics
/// On an unknown tier name.
fn prior_rooms() -> Option<&'static [BudgetTier]> {
    static ROOMS: OnceLock<Option<Vec<BudgetTier>>> = OnceLock::new();
    ROOMS
        .get_or_init(|| {
            let v = std::env::var("STS2_RUNAI_PRIOR_ROOMS").ok()?;
            Some(
                parse_prior_rooms(&v)
                    .unwrap_or_else(|t| panic!("STS2_RUNAI_PRIOR_ROOMS: unknown tier {t:?}")),
            )
        })
        .as_deref()
}

/// The move prior's blend weight `STS2_RUNAI_PRIOR_MIX` (default 0.5) and
/// softmax temperature `STS2_RUNAI_PRIOR_TEMP` (default 1), read once. Both
/// the built-in and the file prior use them.
///
/// # Panics
/// On a value that is not a number, a mix outside `[0, 1]`, or a
/// temperature that is not a positive finite number: the prior's
/// constructor would clamp it silently.
fn prior_knobs() -> (f32, f32) {
    static KNOBS: OnceLock<(f32, f32)> = OnceLock::new();
    *KNOBS.get_or_init(|| {
        let knob = |name: &str, default: f32| {
            std::env::var(name).map_or(default, |v| {
                v.parse::<f32>()
                    .unwrap_or_else(|_| panic!("{name}={v:?} is not a number"))
            })
        };
        let mix = knob("STS2_RUNAI_PRIOR_MIX", 0.5);
        assert!(
            (0.0..=1.0).contains(&mix),
            "STS2_RUNAI_PRIOR_MIX={mix} is outside [0, 1]"
        );
        let temperature = knob("STS2_RUNAI_PRIOR_TEMP", 1.0);
        assert!(
            temperature.is_finite() && temperature > 0.0,
            "STS2_RUNAI_PRIOR_TEMP={temperature} is not a positive number"
        );
        (mix, temperature)
    })
}

/// A PUCT move prior over a playout-net JSON text at [`prior_knobs`].
///
/// # Panics
/// When the text does not parse; `what` names its source in the message.
fn move_prior_from_json(text: &str, what: &str) -> sts2_mcts::SharedPolicy {
    let (mix, temperature) = prior_knobs();
    let net = sts2_mcts::rollout_net::RolloutNet::from_json(text)
        .unwrap_or_else(|e| panic!("{what}: {e}"));
    sts2_mcts::SharedPolicy::new(sts2_mcts::rollout_net::RolloutNetPrior::new(
        net,
        mix,
        temperature,
    ))
}

/// The built-in move prior of `character` (GH #830), built once; `None` for
/// a character without one, or a checkout without the net file (the test
/// `the_ironclad_prior_net_is_compiled_in` fails there).
fn builtin_move_prior(character: CharacterId) -> Option<sts2_mcts::SharedPolicy> {
    static IRONCLAD: OnceLock<Option<sts2_mcts::SharedPolicy>> = OnceLock::new();
    match character {
        CharacterId::Ironclad => IRONCLAD
            .get_or_init(|| {
                sts2_mcts::rollout_net::builtin_prior_json(CharacterId::Ironclad)
                    .map(|text| move_prior_from_json(text, "the built-in Ironclad move prior"))
            })
            .clone(),
        _ => None,
    }
}

/// The move prior from the `STS2_RUNAI_PRIOR_NET` file, built once; `None`
/// unless the variable names a file.
///
/// # Panics
/// When the file can't be read or parsed, or a knob is bad — a silent
/// fallback would measure the uniform search instead.
fn file_move_prior() -> Option<sts2_mcts::SharedPolicy> {
    static PRIOR: OnceLock<Option<sts2_mcts::SharedPolicy>> = OnceLock::new();
    PRIOR
        .get_or_init(|| {
            let PriorNetEnv::File(path) = prior_net_env() else {
                return None;
            };
            let _ = prior_knobs();
            let text = std::fs::read_to_string(path)
                .unwrap_or_else(|e| panic!("STS2_RUNAI_PRIOR_NET={path}: {e}"));
            Some(move_prior_from_json(
                &text,
                &format!("STS2_RUNAI_PRIOR_NET={path}"),
            ))
        })
        .clone()
}

/// Which move prior one search gets ([`prior_choice`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PriorChoice {
    /// No move prior (the uniform PUCT prior, `use_puct_prior = false`).
    None,
    /// [`builtin_move_prior`] of this character.
    BuiltIn(CharacterId),
    /// [`file_move_prior`].
    File,
}

/// The move prior for a search of a `character` fight in `tier` (GH #830),
/// as a pure function of the env's three answers: `env`
/// (`STS2_RUNAI_PRIOR_NET`), `rooms` (`STS2_RUNAI_PRIOR_ROOMS`, `None` when
/// unset) and `layer_on` ([`MctsCombatPolicy::ironclad_layer_on`]).
///
/// The built-in prior is Ironclad's only: the full-run gate measured it on
/// Ironclad alone, so no other character switches one on, whatever nets
/// exist. A file prior keeps its meaning from before the built-in one:
/// every character, every tier unless `rooms` narrows it, whatever the
/// dialect.
fn prior_choice(
    env: &PriorNetEnv,
    character: Option<CharacterId>,
    tier: BudgetTier,
    rooms: Option<&[BudgetTier]>,
    layer_on: bool,
) -> PriorChoice {
    match env {
        PriorNetEnv::Off => PriorChoice::None,
        PriorNetEnv::File(_) => {
            if rooms.is_none_or(|r| r.contains(&tier)) {
                PriorChoice::File
            } else {
                PriorChoice::None
            }
        }
        PriorNetEnv::Default => match character {
            Some(c @ CharacterId::Ironclad)
                if layer_on
                    && rooms
                        .unwrap_or(BUILTIN_PRIOR_ROOMS.as_slice())
                        .contains(&tier) =>
            {
                PriorChoice::BuiltIn(c)
            }
            _ => PriorChoice::None,
        },
    }
}

/// The move prior [`prior_choice`] picks, built (once per process) from its
/// source; `None` for no prior.
fn move_prior(
    env: &PriorNetEnv,
    character: Option<CharacterId>,
    tier: BudgetTier,
    layer_on: bool,
) -> Option<sts2_mcts::SharedPolicy> {
    match prior_choice(env, character, tier, prior_rooms(), layer_on) {
        PriorChoice::None => None,
        PriorChoice::BuiltIn(c) => builtin_move_prior(c),
        PriorChoice::File => file_move_prior(),
    }
}

/// `STS2_RUNAI_PRIOR_CACHE=0` turns the search's per-node PUCT prior cache
/// off ([`Mcts::with_prior_cache`]); unset or `1` keeps it on. The searches
/// are identical either way: the switch exists to check that (GH #830) and
/// to measure what the cache saves. Read once.
///
/// # Panics
/// On any other value.
fn prior_cache_from_env() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(
        || match std::env::var("STS2_RUNAI_PRIOR_CACHE").ok().as_deref() {
            None | Some("1") => true,
            Some("0") => false,
            Some(v) => panic!("STS2_RUNAI_PRIOR_CACHE={v:?}: expected 0 (off) or 1 (on)"),
        },
    )
}

/// `"elite,boss"` → those tiers; `Err(name)` on an unknown one.
fn parse_prior_rooms(v: &str) -> Result<Vec<BudgetTier>, String> {
    v.split(',')
        .map(|t| match t.trim() {
            "common" => Ok(BudgetTier::Common),
            "elite" => Ok(BudgetTier::Elite),
            "boss" => Ok(BudgetTier::Boss),
            other => Err(other.to_owned()),
        })
        .collect()
}

/// Rollout ply cap for run-context combat MCTS (and fight_delta probes).
/// `MctsConfig::default()`'s 10 truncates every act-1 rollout before the
/// fight ends — the exact failure mode documented at
/// `sts2-pycli/python/sts2_hint/recommender.py::RecommendConfig` — and was
/// the root cause of the 2026-06-12 arena win-rate floor: at depth 10 a
/// 1000-sim fight vs NIBBITS_WEAK bleeds 13 HP, at 200 it bleeds 3. Keep
/// in sync with the pycli `mcts_recommend` default (200).
pub const COMBAT_ROLLOUT_DEPTH: u32 = 200;

/// Frozen combat search budget (plan §3.3). All meta-AI A/Bs hold one of
/// these fixed; `none` (meta policy plays combat too — the fuzz behavior)
/// is represented by *not* wrapping the meta policy in
/// [`MctsCombatPolicy`] at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CombatBudget {
    /// Simulations per decision in non-boss combats.
    pub sims: u32,
    /// Simulations per decision in boss combats (the win-rate knee is at
    /// the boss — see
    /// `docs/experiments/2026-06-10-combat-fullrun-winrate-search-vs-leaf-findings.md`).
    pub boss_sims: u32,
    /// Simulations per decision in **elite** combats, `None` = same as
    /// [`Self::sims`]. GH #446's budget ladder on the v3 position book:
    /// elite is the tier with the worst per-ply resolution at 1000 sims
    /// (0.31 vs common 0.57) and the only one the deployed budget does not
    /// escalate.
    pub elite_sims: Option<u32>,
    /// Simulations per decision in non-boss, non-elite combats when the
    /// root position carries a budget-sensitive motif
    /// ([`sts2_mcts::motif::Motif::PotionAvailable`] or
    /// [`Motif::MultiEnemy`](sts2_mcts::motif::Motif::MultiEnemy)) — the
    /// two classes that gained most from 4x sims on the book (+12 / +14pp
    /// against +9pp average) and are detectable at the root for free.
    /// `None` = no motif escalation.
    pub motif_sims: Option<u32>,
    /// Depth-cap leaf evaluator.
    pub leaf: LeafKind,
}

/// Which combat tier a decision is being made in, for budget selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BudgetTier {
    Common,
    Elite,
    Boss,
}

impl BudgetTier {
    #[must_use]
    pub fn from_room(room: Option<RoomType>) -> Self {
        match room {
            Some(RoomType::Boss) => Self::Boss,
            Some(RoomType::Elite) => Self::Elite,
            _ => Self::Common,
        }
    }
}

impl CombatBudget {
    /// Simulations for one decision at `root` in a `tier` combat.
    ///
    /// Boss and elite are decided by tier alone; a common-tier decision
    /// gets [`Self::motif_sims`] when the root carries a budget-sensitive
    /// motif and [`Self::sims`] otherwise. The motif check runs
    /// `legal_actions` plus the lethal detector once per decision —
    /// microseconds against a 1000-sim search.
    #[must_use]
    pub fn sims_for(&self, tier: BudgetTier, root: &CombatState) -> u32 {
        let motif =
            tier == BudgetTier::Common && self.motif_sims.is_some() && Self::budget_motif_at(root);
        self.sims_for_with(tier, motif)
    }

    /// [`Self::sims_for`] with the motif predicate already evaluated (or
    /// skipped) by the caller, so the scan is paid at most once per decision.
    #[must_use]
    pub fn sims_for_with(&self, tier: BudgetTier, motif: bool) -> u32 {
        match tier {
            BudgetTier::Boss => self.boss_sims,
            BudgetTier::Elite => self.elite_sims.unwrap_or(self.sims),
            BudgetTier::Common => match self.motif_sims {
                Some(m) if motif => m,
                _ => self.sims,
            },
        }
    }

    /// The root carries a motif the budget ladder found budget-sensitive.
    #[must_use]
    pub fn budget_motif_at(root: &CombatState) -> bool {
        use sts2_mcts::motif::{motifs_at, Motif};
        motifs_at(root)
            .iter()
            .any(|m| matches!(m, Motif::PotionAvailable | Motif::MultiEnemy))
    }

    /// `elite-focus`: the deployed `eval-cheap` with elite escalated and
    /// common trimmed so total simulations per run stay level. The trim is
    /// sized from the measured decision mix; the arena reports
    /// `combat_sims_spent` per run so neutrality is checked, not assumed.
    ///
    /// **Measured 2026-09-06, do not deploy** (`logs/2026-09-06-budget-realloc/
    /// FINDINGS.md`): 200 paired `rel-200` seeds, A9 +1.5pp (p = 0.77),
    /// **A10 −6.0pp (16/4 discordant, p = 0.012)** at compute ratio 1.01.
    /// Trimming common-tier sims costs more HP over an A10 run than the
    /// elite escalation buys back. Kept so the result reproduces.
    #[must_use]
    pub const fn elite_focus() -> Self {
        Self {
            sims: ELITE_FOCUS_COMMON_SIMS,
            boss_sims: 4000,
            elite_sims: Some(ELITE_FOCUS_ELITE_SIMS),
            motif_sims: None,
            leaf: LeafKind::TexelWide,
        }
    }

    /// `motif-focus`: `eval-cheap` with common-tier decisions escalated
    /// only when the root offers a potion or faces several enemies, and
    /// the remaining common decisions trimmed for neutrality.
    ///
    /// **Measured 2026-09-06, null** (same panel as `elite_focus`): A9
    /// +2.5pp (p = 0.56), A10 −1.5pp (p = 0.71) at compute ratio 1.00,
    /// despite +1.7pp agreement on the v3 position book (p = 0.006). The
    /// book weights every decision equally; a run does not. Kept so the
    /// result reproduces.
    #[must_use]
    pub const fn motif_focus() -> Self {
        Self {
            sims: MOTIF_FOCUS_COMMON_SIMS,
            boss_sims: 4000,
            elite_sims: Some(MOTIF_FOCUS_ELITE_SIMS),
            motif_sims: Some(MOTIF_FOCUS_MOTIF_SIMS),
            leaf: LeafKind::TexelWide,
        }
    }
}

/// Reallocation constants for the two GH #446 budget arms, sized from the
/// measured decision mix of the deployed champion at A9 (40 `rel-200`
/// seeds, `eval-cheap`, 2026-09-06: 335 combat decisions per run — 56.2%
/// common, 20.1% elite, 23.7% boss; 573k sims per run, 1710 per decision).
/// Boss stays at 4000 (its escalation is a measured +3.7pp lever), so the
/// trade is common ↔ elite: `c·0.562 + e·0.201 = 1000·0.763`.
///
/// `elite-focus`: elite 2000 ⇒ common 642. Neutral to <1% on that mix;
/// the arena's `combat_sims_spent` is the check.
pub const ELITE_FOCUS_COMMON_SIMS: u32 = 642;
pub const ELITE_FOCUS_ELITE_SIMS: u32 = 2000;
/// `motif-focus`: measured live, **47%** of common-tier decisions carry a
/// budget motif (potion available / several enemies) — too many to double
/// and stay neutral (plain common would drop to ~116). So the reallocation
/// is within the common tier: motif 1500, plain common 557, elite untouched
/// (`557·0.298 + 1500·0.264 = 1000·0.562`). Whether a 1.5x on the classes
/// the book says gain most beats a flat 1000 is the question.
pub const MOTIF_FOCUS_COMMON_SIMS: u32 = 557;
pub const MOTIF_FOCUS_ELITE_SIMS: u32 = 1000;
pub const MOTIF_FOCUS_MOTIF_SIMS: u32 = 1500;

impl CombatBudget {
    /// `eval-cheap`: 1000 / 4000 sims, texel-wide — the iteration workhorse.
    /// Wide vs narrow leaf is ~a wash: a 2026-07-10 fixed-policy A9/A10 ablation
    /// (`STS2_RUNAI_LEAF`) found ≤1.3pp, within noise — an earlier "+11.7pp @
    /// 1000 sims" figure did not replicate. TexelWide is kept as the ~1%-wall
    /// default; what actually moves high-ascension win rate is the terminal-eval
    /// Q-calibration (RichBoth) and the sim budget, not the leaf's feature width.
    #[must_use]
    pub const fn eval_cheap() -> Self {
        Self {
            sims: 1000,
            boss_sims: 4000,
            elite_sims: None,
            motif_sims: None,
            leaf: LeafKind::TexelWide,
        }
    }

    /// `eval-strong`: 5000 / 15000 sims, narrow texel (leaf choice is a
    /// wash at this budget) — milestone numbers.
    #[must_use]
    pub const fn eval_strong() -> Self {
        Self {
            sims: 5000,
            boss_sims: 15000,
            elite_sims: None,
            motif_sims: None,
            leaf: LeafKind::Texel,
        }
    }
}

/// splitmix64 — the standard 64-bit finalizer, used to derive
/// per-decision MCTS seeds from `(run seed, decision counter)` so combat
/// search is deterministic per run *and* decorrelated across decisions.
/// Exposed to submodules (e.g. `sampling`) to derive per-sample params and
/// RNG seeds without duplicating the implementation.
pub(crate) fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Composite run policy: meta decisions (map / rewards / events / shop /
/// rest / …) go to the wrapped `meta` policy; `Phase::Combat` decisions
/// are picked by the combat MCTS under a frozen [`CombatBudget`].
///
/// **Boss detection:** a combat is "boss" iff
/// `rm.run_state.current_room.room_type == RoomType::Boss` — the room is
/// entered before combat starts and stays current for the whole fight
/// (`advance_boss_act` clears it only after the boss reward flow), so
/// this is stable across every decision of the fight.
///
/// **Determinism:** the MCTS seed for decision `k` of a run is
/// `splitmix64(run_seed_bits ^ (k+1) · φ64)` — a pure function of the
/// run seed and a per-decision counter, so a re-run reproduces every
/// search exactly. Search is single-threaded (`search_n`, never
/// `par_search`) — throughput mode parallelizes across seeds instead.
///
/// **Fallback:** if the MCTS picks an action with no `RunAction`
/// translation (Channel / Evoke / multi-pick choices — none reachable
/// for Ironclad at this pin) or the translated action is missing from
/// `actions`, the meta policy decides instead. Every such fallback is
/// tallied in [`MctsCombatPolicy::fallbacks`] (issue #164) so the
/// MCTS→meta handoff is *measured* rather than silent — see
/// [`FallbackCounts`].
#[derive(Debug)]
pub struct MctsCombatPolicy<M> {
    meta: M,
    budget: CombatBudget,
    run_seed: i32,
    /// Per-run combat-decision counter (seeds decision `k`'s search).
    decisions: u64,
    /// Of [`Self::decisions`], how many were made in elite / boss combats,
    /// and the simulations actually requested across all of them (a
    /// forced move, which is not searched, requests none). These
    /// are what let a budget-reallocation arm prove it is compute-neutral
    /// (GH #446) instead of asserting it.
    decisions_elite: u64,
    decisions_boss: u64,
    /// Common-tier decisions whose root carried a budget-sensitive motif
    /// (`CombatBudget::budget_motif_at`). Counted when the budget has
    /// `motif_sims` or when [`Self::count_motifs`] is set — never for free,
    /// because the scan is a per-decision cost the deployed budget must not
    /// pay.
    decisions_motif: u64,
    /// `STS2_RUNAI_COUNT_MOTIFS` was set at construction: count motif
    /// decisions even on a budget that does not use them (how the
    /// `motif-focus` neutrality constant was sized from an `eval-cheap` run).
    count_motifs: bool,
    sims_spent: u64,
    /// Per-run tally of combat decisions handed back to the meta policy,
    /// broken down by reason (issue #164).
    fallbacks: FallbackCounts,
    /// Per-reason tally of lethal-event-pick vetoes (GH #300).
    event_vetoes: EventVetoCounts,
    /// Root distribution of the most recent combat `mcts_pick`
    /// (`None` until the first combat decision / after a meta fallback).
    last_combat_search: Option<CombatSearchRecord>,
    /// Root-pick tolerance band ([`MctsConfig::tie_eps`]) applied to
    /// every combat search this policy runs. Resolved from
    /// `STS2_RUNAI_TIE_EPS` ONCE at construction (see
    /// [`resolve_tie_eps`]) so a sweep can't have the knob flip
    /// mid-run; `0` is the rollback seam back to strict-equality
    /// tie-breaking.
    tie_eps: f32,
    /// Graded proven-loss backups ([`MctsConfig::loss_gradient`],
    /// FINDINGS §3 of the 2026-08-09 A8 audit). Resolved from
    /// `STS2_RUNAI_LOSS_GRADIENT` ONCE at construction (see
    /// [`resolve_loss_gradient`]); `0` is the rollback seam back to the
    /// constant −1.45 loss floor (search-off boss fights).
    loss_gradient: bool,
    /// In-band 1-ply root guards ([`MctsConfig::root_guards`],
    /// FINDINGS §5 of the 2026-08-09 A8 audit). Resolved from
    /// `STS2_RUNAI_ROOT_GUARDS` ONCE at construction (see
    /// [`resolve_root_guards`]); `0` is the rollback seam that drops
    /// the block / free-play EndTurn refinements. The proven-win guard
    /// is unconditional in the engine and not reachable from here.
    root_guards: bool,
    /// Bundled search-knob overrides for A/B'ing tuned `MctsConfig`
    /// values through a full `run_arena` run without a rebuild per arm.
    /// Resolved from `STS2_RUNAI_MCTS_OVERRIDES` ONCE at construction
    /// (see [`parse_mcts_overrides`]), same "read once, not per pick"
    /// rationale as `tie_eps` / `loss_gradient` / `root_guards`. Applied
    /// in [`Self::search_config`] AFTER those three single-knob seams, so an
    /// explicit entry here wins if both are set for the same knob, and over
    /// the run agent's character layer ([`IRONCLAD_SEARCH_KNOBS`]).
    overrides: MctsKnobOverrides,
    /// Root-level potion opportunity-cost table
    /// ([`sts2_mcts::MctsConfig::potion_charge_q`], GH #447 follow-up).
    /// Resolved from `STS2_RUNAI_POTION_OC` ONCE at construction (see
    /// [`potion_oc::resolve_potion_oc`]); requires
    /// `STS2_RUNAI_POTION_WEIGHT=0` (enforced by panic) so the leaf
    /// potion term cannot double-count what the root charge prices.
    /// `None` (unset, the deployed default) ⇒ byte-identical searches.
    potion_oc: Option<potion_oc::PotionOc>,
    /// Randomized per-fight drink masking
    /// ([`sts2_mcts::MctsConfig::root_drink_mask`]) for causal OC
    /// corpora. Resolved from `STS2_RUNAI_DRINK_JITTER` ONCE at
    /// construction (see [`potion_oc::resolve_drink_jitter`]); the coin
    /// is a pure function of (run seed, floor, salt) so the offline fit
    /// re-labels masked fights without a recording change. `None`
    /// (unset, the deployed default) ⇒ no-op.
    drink_jitter: Option<potion_oc::DrinkJitter>,

    /// Potion headroom ablation (`STS2_RUNAI_POTION_ABLATION`, resolved
    /// once in `new`): `no-drinks` bounds the whole potion system's
    /// value, `boss-oracle` bounds all acquisition work. Unset ⇒ None ⇒
    /// deployed behavior untouched.
    potion_ablation: Option<potion_ablation::PotionAblation>,
    /// Which boss fights get a win value by what their end state carries
    /// forward (`STS2_RUNAI_BOSS_CARRY`, resolved once in `new`; see
    /// [`boss_win_carry`], GH #771). Default: the run's last boss only.
    boss_carry: BossCarry,
    /// Per-fight-class multiplier on the root potion charge
    /// (`STS2_RUNAI_POTION_OC_CLASS_SCALE`, resolved once in `new`;
    /// default all-ones ⇒ bit-identical to no scaling).
    potion_class_scale: potion_oc::ClassScale,
    /// In-fight potion-play degradation (`STS2_RUNAI_POTION_TIMING`) —
    /// measures how much value the search's within-fight timing /
    /// selection actually extracts. Unset ⇒ untouched.
    potion_timing: Option<potion_timing::PotionTiming>,
    /// `slots+2` bookkeeping: the capacity grant is applied exactly once
    /// per run (the policy is constructed per run).
    slots_granted: bool,
    /// `boss-oracle` bookkeeping: the floor whose belt was already
    /// topped up, so a boss is filled ONCE at entry rather than refilled
    /// after every drink (which would be an infinite-potion oracle, a
    /// different and far weaker measurement).
    oracle_filled_floor: Option<i32>,
    /// GH #830 `tree_reuse` (override `tree_reuse=1`): the search tree of
    /// this policy's previous combat decision, kept for the next one. Only
    /// ever `Some` between a combat decision and the very next call of
    /// [`RunPolicy::choose`], which takes it whatever happens. Always
    /// `None` with the knob off (the default), so the deployed search
    /// builds a fresh tree per decision exactly as before.
    retained: Option<RetainedSearch>,
    /// What `tree_reuse` did, over this policy's lifetime.
    tree_reuse: TreeReuseCounts,
    /// Root visits the most recent combat search started with: `Some(n)`
    /// when it reused a tree (`n` retained visits), `None` otherwise.
    last_retained_visits: Option<u32>,
    /// Turn-plan root filter (GH #830): the pro-trained plan scorer and K,
    /// from `STS2_RUNAI_TURNPLAN_ROOT_FILTER=<weights.json>:<K>`, resolved
    /// once in `new`. `None` (unset, the default) ⇒ the root is never
    /// filtered and nothing below runs.
    root_filter: Option<(Arc<sts2_mcts::turn_plan_scorer::PlanScorer>, usize)>,
    /// What the root filter did at the most recent combat decision (`None`
    /// with the knob off).
    last_root_filter: Option<RootFilterRecord>,
}

/// What the turn-plan root filter did at one combat decision (GH #830),
/// for the bench record. See [`sts2_mcts::turn_plan_scorer::root_filter`].
#[derive(Clone, Debug, PartialEq)]
pub struct RootFilterRecord {
    /// The search root was restricted.
    pub applied: bool,
    /// Why it was not: `"forced"`, `"cap"` or `"empty"`.
    pub fallback: Option<&'static str>,
    /// Canonical legal root actions.
    pub legal_n: usize,
    /// Enumerated plans (0 for a forced root).
    pub plans_n: usize,
    /// Plans kept (top-K ∪ anchors).
    pub kept_n: usize,
    /// Root actions the search could pick.
    pub allowed_n: usize,
    /// `"topk"` when a top-K plan starts with the chosen action,
    /// `"anchor_only"` when only anchor plans do; `None` when not applied.
    pub chosen_via: Option<&'static str>,
    /// Wall time of the filter (enumerate, score, anchors), microseconds.
    pub filter_us: u64,
}

/// `STS2_RUNAI_TURNPLAN_ROOT_FILTER` → (scorer, K). Unset or empty ⇒
/// `None`. A malformed value or an unreadable weight file panics: the knob
/// only gates measurement runs, which must not silently fall back to the
/// unfiltered search.
fn resolve_root_filter(
    raw: Option<&str>,
) -> Option<(Arc<sts2_mcts::turn_plan_scorer::PlanScorer>, usize)> {
    let raw = raw.filter(|v| !v.is_empty())?;
    let (path, k) = raw.rsplit_once(':').unwrap_or_else(|| {
        panic!("STS2_RUNAI_TURNPLAN_ROOT_FILTER={raw:?}: expected <weights.json>:<K>")
    });
    let k: usize = k.parse().ok().filter(|&k| k >= 1).unwrap_or_else(|| {
        panic!("STS2_RUNAI_TURNPLAN_ROOT_FILTER: K={k:?} is not an integer >= 1")
    });
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("STS2_RUNAI_TURNPLAN_ROOT_FILTER: {path}: {e}"));
    let scorer = sts2_mcts::turn_plan_scorer::PlanScorer::from_json(&text)
        .unwrap_or_else(|e| panic!("STS2_RUNAI_TURNPLAN_ROOT_FILTER: {path}: {e}"));
    Some((Arc::new(scorer), k))
}

/// A combat search kept for the policy's next decision (GH #830
/// `tree_reuse`).
#[derive(Debug)]
struct RetainedSearch {
    mcts: Mcts<sts2_mcts::SharedPolicy>,
    /// The action the policy returned from this tree's root.
    action: Action,
    /// The floor of the fight the tree was searched in.
    floor: i32,
}

/// What `tree_reuse` (GH #830) did over a policy's lifetime. All zero with
/// the knob off.
///
/// A *searched* decision is one that ran a search (not a forced move). Of
/// those, `reused` started from a retained tree and every other one built a
/// fresh tree for the reason counted: `fresh_no_tree` (the first decision of
/// a fight, or the one after a move nothing was kept for) or a
/// `fresh_*` mismatch ([`sts2_mcts::RerootMiss`]). So `searched == reused +
/// Σ fresh_*`, the reuse rate is `reused / searched`, and the share of root
/// visits that were carried over is `retained_visits / (retained_visits +
/// new_sims)`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TreeReuseCounts {
    /// Searched decisions with the knob on.
    pub searched: u64,
    /// Of those, how many started from a retained tree.
    pub reused: u64,
    /// Of `reused`, how many through a chance edge (a stored outcome
    /// matched the realised one).
    pub reused_via_chance: u64,
    /// Searched decisions with no tree retained from the decision before.
    pub fresh_no_tree: u64,
    /// The taken action had no expanded child in the retained tree.
    pub fresh_not_expanded: u64,
    /// A deterministic edge whose child did not match the observation.
    pub fresh_mismatch: u64,
    /// A chance edge with no stored outcome matching the observation.
    pub fresh_outcome_unseen: u64,
    /// The matched node's moves differed from the new root's.
    pub fresh_edge_set: u64,
    /// A retained tree from another fight, or an unsupported search mode.
    pub fresh_other: u64,
    /// Σ over searched decisions of the root visits carried over.
    pub retained_visits: u64,
    /// Σ over searched decisions of the new simulations run (the budget).
    pub new_sims: u64,
    /// Forced moves (not searched) that carried a retained tree on to the
    /// next decision, and forced moves that dropped one.
    pub forced_carried: u64,
    pub forced_dropped: u64,
}

/// The root MCTS distribution for the most recent combat decision,
/// captured so a stepwise driver (the workbench's `HeuristicRunAgent`) can
/// surface the same visit/Q panel `mcts_recommend` shows — without
/// re-running the search (which would diverge on seed).
///
/// A forced move (one legal action) is not searched: its record is a
/// one-row stub, `chosen` with zero visits and zero value — "no search
/// ran", not "this action loses".
#[derive(Clone, Debug)]
pub struct CombatSearchRecord {
    pub outcomes: Vec<sts2_mcts::RootOutcome>,
    pub chosen: sts2_core::Action,
    /// What `net_expand_order` (GH #830 rung 4) did in this search
    /// ([`sts2_mcts::Mcts::expand_order_stats`]); all zero when the knob
    /// is off and for a forced move, which is not searched.
    pub expand_order: sts2_mcts::ExpandOrderStats,
}

/// What the *meta* (non-combat) policy computed for one decision — the
/// scored candidate list, the net's value estimate, and which option the
/// score function ranked first.
///
/// Recorded so a trajectory can answer "was this a deliberate pick or a
/// coin flip?" after the fact. A decision the policy never scored (a
/// single surfaced option, a surface delegated to the heuristic
/// fallback) carries a `source` saying so and an empty `probs` — the
/// distinction between "chose this at p=0.97" and "there was nothing to
/// choose" is invisible in the chosen index alone.
#[derive(Clone, Debug)]
pub struct MetaEvalRecord {
    /// Which scorer produced this: `ppo` | `ppo_auto_take` |
    /// `heuristic_fallback`. (`heuristic_potion_swap` retired in GH #379 —
    /// the net owns the potion-swap surface now.)
    pub source: &'static str,
    /// The candidate actions actually scored, in scored order (the
    /// MASKED list — not necessarily the caller's `actions`).
    pub actions: Vec<RunAction>,
    /// Softmax probability per scored candidate, aligned with `actions`.
    /// Empty when nothing was scored.
    pub probs: Vec<f32>,
    /// The net's state-value estimate at this decision, when it ran.
    pub value: Option<f32>,
    /// Index into `actions` the scorer ranked first.
    pub argmax: Option<usize>,
}

/// Per-decision telemetry the trajectory recorder attaches to each
/// `Decision` so a recorded run explains *why* each pick was made, not
/// just what was picked.
///
/// Default bodies return `None`, so a policy without a search or a value
/// function (the M0 baselines) is zero-cost and needs no impl body — the
/// same shape as [`VScoreSource`].
pub trait DecisionTelemetry {
    /// Root MCTS distribution of the combat decision just made, if this
    /// policy searched one. Non-consuming: [`MctsCombatPolicy::mcts_pick`]
    /// clears the slot at the top of every search, so a stale record can
    /// never survive into the next decision.
    fn combat_search(&self) -> Option<&CombatSearchRecord> {
        None
    }

    /// The meta-policy scoring of the decision just made. *Take*
    /// semantics (like [`VScoreSource::take_v_scores`]): consumed, so a
    /// combat ply the meta never saw cannot inherit the previous
    /// non-combat decision's distribution.
    fn take_meta_eval(&mut self) -> Option<MetaEvalRecord> {
        None
    }
}

impl DecisionTelemetry for FirstOptionPolicy {}
impl DecisionTelemetry for RandomPolicy {}
impl DecisionTelemetry for HeuristicPolicy {}
impl DecisionTelemetry for TieredPolicy {}
impl DecisionTelemetry for ValuePolicy {}
impl DecisionTelemetry for SampledDeckPolicy {}
impl<P> DecisionTelemetry for AblationPolicy<P> {}

impl<M: DecisionTelemetry> DecisionTelemetry for MctsCombatPolicy<M> {
    fn combat_search(&self) -> Option<&CombatSearchRecord> {
        self.last_combat_search.as_ref()
    }

    fn take_meta_eval(&mut self) -> Option<MetaEvalRecord> {
        self.meta.take_meta_eval()
    }
}

/// Why a combat MCTS decision was handed back to the meta policy instead
/// of being translated into a surfaced [`RunAction`] (issue #164).
///
/// The run-AI's combat bridge is single-step rank-and-pick: it has no
/// candidate model for Channel / Evoke / Discover-style or "pick N of M"
/// selections. When the combat MCTS chooses such an action,
/// [`translate_combat_action`] returns `None` and the decision silently
/// degrades to the meta heuristic. These counters make that handoff
/// observable so a sweep can report how often — and *why* — it fires,
/// before the larger combinatorial-selection build-out is scoped.
///
/// Counts accumulate over the policy's lifetime (one run per
/// [`MctsCombatPolicy`]); divide by [`MctsCombatPolicy::combat_decisions`]
/// for a per-decision rate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FallbackCounts {
    /// MCTS produced no root outcome to pick from (empty search tree —
    /// e.g. a terminal/degenerate root). Rare; not the #164 gap.
    pub no_outcome: u64,
    /// The chosen combat [`Action`] has no [`RunAction`] twin — Channel /
    /// Evoke / Discover / unmodeled multi-pick selections. **This is the
    /// architectural gap of #164**: the meta policy decides with no
    /// awareness it was asked to.
    pub untranslatable: u64,
    /// The action translated cleanly but the resulting [`RunAction`] was
    /// absent from the surfaced `actions` list (translation drift vs
    /// `RunManager::get_available_actions`). Signals a translation bug
    /// rather than a missing surface.
    pub not_surfaced: u64,
}

/// Pure `STS2_RUNAI_HONEST` env-value → [`sts2_mcts::HiddenInfoMode`]
/// resolution, factored out of [`MctsCombatPolicy::mcts_pick`] so the
/// mapping is unit-testable without mutating process env (tests pass the
/// `Option<&str>` directly instead of `std::env::set_var` +
/// `remove_var`).
///
/// Honest-by-default per the G4 bless (2026-07-12,
/// `docs/plans/2026-07-12-clairvoyance-removal-plan.md`): unset (or the
/// explicit alias `"belief"`) resolves to [`sts2_mcts::HiddenInfoMode::BeliefDraw`],
/// the champion of the G2 3-way A/B (release-2000: clairvoyant 40.35%,
/// F0/RootScramble 13.65%, F2/BeliefDraw-retuned 17.40%, +3.75pp over F0,
/// p=4.7e-5). `"0"` is the rollback/ablation escape hatch back to the
/// pre-G4 leaky default, `"root"`/`"full"` select the F0/F1 honest modes
/// respectively, and anything else panics rather than silently falling
/// back — see the call site's rationale comment for why this flag alone
/// doesn't get the "silent fallback to deployed behavior" treatment every
/// other combat-search ablation knob gets.
/// Pure `STS2_RUNAI_TIE_EPS` env-value → [`MctsConfig::tie_eps`]
/// resolution, factored out of [`MctsCombatPolicy::new`] so the mapping
/// is unit-testable without mutating process env (same shape as
/// [`resolve_hidden_info`]).
///
/// Unset keeps the engine default (`0.02`, the measured Q noise floor at
/// a run's 1000-sim budget — GH #403 / the 2026-08-09 A8 audit). `0` is
/// the rollback seam: it restores strict-equality tie-breaking, i.e. the
/// pre-audit deployed `mcts_pick` semantics. Negative values clamp to
/// `0.0` (there is no meaningful "anti-band"). An unparseable value
/// PANICS rather than silently falling back — this knob exists to gate
/// A/B measurement campaigns of the tie-break itself, and a typo
/// silently running the default arm would poison hours of paired cells
/// without any signal (the same rationale as `STS2_RUNAI_HONEST`).
fn resolve_tie_eps(env_value: Option<&str>) -> f32 {
    match env_value {
        None => MctsConfig::default().tie_eps,
        Some(raw) => match raw.trim().parse::<f32>() {
            Ok(v) if v.is_finite() => v.max(0.0),
            _ => panic!(
                "STS2_RUNAI_TIE_EPS={raw:?} is not a finite f32 (expected unset or a value \
                 like \"0.02\"; \"0\" restores strict tie-breaking) — refusing to silently \
                 fall back and poison the run"
            ),
        },
    }
}

/// Pure `STS2_RUNAI_LOSS_GRADIENT` env-value → [`MctsConfig::loss_gradient`]
/// resolution, factored out of [`MctsCombatPolicy::new`] like
/// [`resolve_tie_eps`].
///
/// Unset (or `"1"`) keeps the engine default (graded proven-loss backups —
/// FINDINGS §3 of the 2026-08-09 A8 audit). `"0"` is the rollback seam: it
/// restores the pre-audit constant `-1.45` loss floor, i.e. the regime in
/// which whole boss fights ran with every root child pinned to one value.
/// Anything else PANICS rather than silently falling back — this knob
/// exists to gate A/B measurement campaigns of the loss backup itself, and
/// a typo silently running the default arm would poison hours of paired
/// cells without any signal (the same rationale as `STS2_RUNAI_HONEST` and
/// `STS2_RUNAI_TIE_EPS`).
fn resolve_loss_gradient(env_value: Option<&str>) -> bool {
    match env_value.map(str::trim) {
        None | Some("1") => true,
        Some("0") => false,
        Some(other) => panic!(
            "STS2_RUNAI_LOSS_GRADIENT={other:?} is not \"0\" or \"1\" (unset keeps the \
             graded default; \"0\" restores the constant loss floor) — refusing to \
             silently fall back and poison the run"
        ),
    }
}

/// Pure `STS2_RUNAI_ROOT_GUARDS` env-value → [`MctsConfig::root_guards`]
/// resolution, factored out of [`MctsCombatPolicy::new`] like
/// [`resolve_tie_eps`] / [`resolve_loss_gradient`].
///
/// Unset (or `"1"`) keeps the engine default (in-band 1-ply root guards —
/// FINDINGS §5 of the 2026-08-09 A8 audit: block-vs-telegraphed-damage and
/// don't-end-turn-with-a-free-play refinements of a band-winning EndTurn).
/// `"0"` is the rollback seam back to the plain tie-band pick; the
/// proven-win guard is exact and stays on either way. Anything else PANICS
/// rather than silently falling back — this knob exists to gate A/B
/// measurement campaigns of the guards themselves, and a typo silently
/// running the default arm would poison hours of paired cells without any
/// signal (the same rationale as `STS2_RUNAI_HONEST` and its siblings).
fn resolve_root_guards(env_value: Option<&str>) -> bool {
    match env_value.map(str::trim) {
        None | Some("1") => true,
        Some("0") => false,
        Some(other) => panic!(
            "STS2_RUNAI_ROOT_GUARDS={other:?} is not \"0\" or \"1\" (unset keeps the \
             in-band root guards; \"0\" restores the plain tie-band pick) — refusing to \
             silently fall back and poison the run"
        ),
    }
}

/// Which boss fights get the carry-aware win value of [`boss_win_carry`]
/// (GH #771).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BossCarry {
    /// None: every fight keeps the HP-and-potions win value (pre-#771).
    Off,
    /// The run's last boss only: win probability alone. The default — on
    /// 710 human A10 boss-2 fights x 2 it won +2.1pp (p = 0.018), and full
    /// runs were neutral (`logs/2026-10-22-771-boss-objective/`).
    LastBoss,
    /// Every boss, act-1/2 HP discounted too. Screened null (deaths −0.3pp).
    All,
}

/// Pure `STS2_RUNAI_BOSS_CARRY` env value → [`BossCarry`]. Unset or empty ⇒
/// `LastBoss`; `"0"` ⇒ `Off` (rollback); `"1"` ⇒ `All`. Anything else PANICS
/// rather than silently running another arm.
fn resolve_boss_carry(env_value: Option<&str>) -> BossCarry {
    match env_value.map(str::trim) {
        None | Some("") => BossCarry::LastBoss,
        Some("0") => BossCarry::Off,
        Some("1") => BossCarry::All,
        Some(other) => panic!(
            "STS2_RUNAI_BOSS_CARRY={other:?} is not \"0\" or \"1\" — refusing to silently \
             run another search in its place"
        ),
    }
}

/// Re-exported from [`sts2_mcts`], which owns it so the advisor's search
/// (`sts2-wire`, which does not depend on this crate) applies the same rule.
use sts2_mcts::boss_win_carry;

/// Parsed `STS2_RUNAI_MCTS_OVERRIDES` bundle — one optional slot per
/// tunable [`MctsConfig`] search knob, shared vocabulary with
/// `sts2-bench`'s `combat_asc_eval` example's `Knobs::apply`. `None`
/// means "not mentioned in the env value", so [`MctsKnobOverrides::apply`]
/// leaves that field exactly where the earlier seams
/// ([`resolve_tie_eps`] / [`resolve_loss_gradient`] / [`resolve_root_guards`]
/// / [`mcts_config_for`]'s baked F2 retune) already put it. `sims` and
/// `max_rollout_depth` are deliberately absent — combat budget is owned by
/// [`CombatBudget`] / `STS2_RUNAI_ROLLOUT_DEPTH`, not this seam.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct MctsKnobOverrides {
    chance_widening_c: Option<f32>,
    chance_widening_alpha: Option<f32>,
    draw_widening_c: Option<f32>,
    draw_widening_alpha: Option<f32>,
    endturn_widening_c: Option<f32>,
    endturn_widening_alpha: Option<f32>,
    tie_eps: Option<f32>,
    c_puct: Option<f32>,
    loss_w_removed: Option<f32>,
    loss_w_hp: Option<f32>,
    loss_w_turns: Option<f32>,
    loss_turns_half_life: Option<f32>,
    loss_span: Option<f32>,
    loot_weight: Option<f32>,
    persistent_weight: Option<f32>,
    loss_gradient: Option<bool>,
    root_guards: Option<bool>,
    chance_closed_stops_widening: Option<bool>,
    rollout_policy: Option<sts2_mcts::RolloutPolicyKind>,
    belief_dialect: Option<bool>,
    leaf_playouts: Option<u32>,
    /// GH #830 `tree_reuse`: keep the search tree between the policy's own
    /// consecutive combat decisions ([`MctsCombatPolicy::tree_reuse_counts`]).
    /// Not an [`MctsConfig`] field — the policy, not the search, owns the
    /// tree across decisions — so [`MctsKnobOverrides::apply`] skips it.
    tree_reuse: Option<bool>,
    tt_hash_v2: Option<bool>,
    net_expand_order: Option<bool>,
    fpu_reduction: Option<f32>,
    root_sequential_halving: Option<bool>,
    effective_incoming: Option<bool>,
    removed_counts_forms: Option<bool>,
    win_hp_carry: Option<f32>,
    win_potion_carry: Option<f32>,
}

/// The full set of knob names [`parse_mcts_overrides`] accepts, kept as
/// one literal so the "unknown knob" panic message and any future
/// listing stay in sync by construction.
const MCTS_OVERRIDE_KNOB_NAMES: &str = "chance_widening_c, chance_widening_alpha, \
    draw_widening_c, draw_widening_alpha, endturn_widening_c, endturn_widening_alpha, \
    tie_eps, c_puct, loss_w_removed, loss_w_hp, loss_w_turns, \
    loss_turns_half_life, loss_span, loot_weight, persistent_weight, loss_gradient, root_guards, \
    chance_closed_stops_widening, rollout_policy (heuristic | uniform | net; net, the \
    honest-mode default, plays the built-in Ironclad playout net, or the STS2_ROLLOUT_NET \
    weights when set), belief_dialect (true: apply the BeliefDraw dialect even under \
    STS2_RUNAI_HONEST=0, so a clairvoyant arm differs in information alone; other knobs in \
    the bundle still win), leaf_playouts (integer >= 1: playouts per search leaf, \
    GH #830), tree_reuse (bool: keep the search tree between the policy's own consecutive \
    combat decisions, GH #830), tt_hash_v2 (bool: transposition key with the potion belt and \
    uid-free powers, GH #830), net_expand_order (true: order non-root expansions by the playout \
    net, GH #830 rung 4), fpu_reduction (f32: price an unexpanded action at parent_q - r instead \
    of +inf), root_sequential_halving (true: allocate the root budget by Sequential Halving, \
    UCT below), effective_incoming (true: heuristic playouts and proven-lost scoring read enemy \
    attacks at their displayed number, GH #830), removed_counts_forms (true: the loss band's \
    enemy-HP-removed term counts multi-form / reviving enemies' remaining forms, GH #830), \
    win_hp_carry / win_potion_carry (f32 in [0, 1]: how much of a won fight's end HP / held \
    potions the win value counts; 0 = win probability alone, GH #771) \
    (sims and depth are NOT \
    included here — combat budget is owned by CombatBudget / STS2_RUNAI_ROLLOUT_DEPTH)";

impl MctsKnobOverrides {
    /// Apply every set slot to `config`, leaving unset slots untouched.
    /// Called AFTER the single-knob seams (`tie_eps` / `loss_gradient` /
    /// `root_guards`) and the `mcts_config_for` F2 retune have already
    /// written `config` — see the call site in
    /// [`MctsCombatPolicy::search_config`] for the interaction-order
    /// rationale (an explicit override in the bundle wins over the older
    /// single-knob env vars when both are set).
    fn apply(self, config: &mut MctsConfig) {
        // First, so any explicit knob in the same bundle overrides the
        // dialect value it would otherwise get.
        if self.belief_dialect == Some(true) {
            config.apply_belief_draw_dialect();
        }
        if let Some(v) = self.chance_widening_c {
            config.chance_widening_c = v;
        }
        if let Some(v) = self.chance_widening_alpha {
            config.chance_widening_alpha = v;
        }
        if let Some(v) = self.draw_widening_c {
            config.draw_widening_c = v;
        }
        if let Some(v) = self.draw_widening_alpha {
            config.draw_widening_alpha = v;
        }
        if let Some(v) = self.endturn_widening_c {
            config.endturn_widening_c = v;
        }
        if let Some(v) = self.endturn_widening_alpha {
            config.endturn_widening_alpha = v;
        }
        if let Some(v) = self.tie_eps {
            config.tie_eps = v;
        }
        if let Some(v) = self.c_puct {
            config.c_puct = v;
        }
        if let Some(v) = self.loss_w_removed {
            config.loss_w_removed = v;
        }
        if let Some(v) = self.loss_w_hp {
            config.loss_w_hp = v;
        }
        if let Some(v) = self.loss_w_turns {
            config.loss_w_turns = v;
        }
        if let Some(v) = self.loss_turns_half_life {
            config.loss_turns_half_life = v;
        }
        if let Some(v) = self.loss_span {
            config.loss_span = v;
        }
        if let Some(v) = self.loot_weight {
            config.loot_weight = v;
        }
        if let Some(v) = self.persistent_weight {
            config.persistent_weight = v;
        }
        if let Some(v) = self.loss_gradient {
            config.loss_gradient = v;
        }
        if let Some(v) = self.root_guards {
            config.root_guards = v;
        }
        if let Some(v) = self.chance_closed_stops_widening {
            config.chance_closed_stops_widening = v;
        }
        if let Some(v) = self.rollout_policy {
            config.rollout_policy = v;
        }
        if let Some(v) = self.leaf_playouts {
            config.leaf_playouts = v;
        }
        if let Some(v) = self.tt_hash_v2 {
            config.tt_hash_v2 = v;
        }
        if let Some(v) = self.net_expand_order {
            config.net_expand_order = v;
        }
        if let Some(v) = self.fpu_reduction {
            config.fpu_reduction = Some(v);
        }
        if let Some(v) = self.root_sequential_halving {
            config.root_sequential_halving = v;
        }
        if let Some(v) = self.effective_incoming {
            config.effective_incoming = v;
        }
        if let Some(v) = self.removed_counts_forms {
            config.removed_counts_forms = v;
        }
        if let Some(v) = self.win_hp_carry {
            config.win_hp_carry = Some(v);
        }
        if let Some(v) = self.win_potion_carry {
            config.win_potion_carry = Some(v);
        }
    }

    /// Each slot of `self`, or `fallback`'s where `self` leaves it unset:
    /// the env bundle over a character preset ([`character_search_preset`]),
    /// so an explicit env knob always wins. Destructures both sides
    /// exhaustively, so a knob added without a line here fails to compile.
    fn or(self, fallback: Self) -> Self {
        let Self {
            chance_widening_c,
            chance_widening_alpha,
            draw_widening_c,
            draw_widening_alpha,
            endturn_widening_c,
            endturn_widening_alpha,
            tie_eps,
            c_puct,
            loss_w_removed,
            loss_w_hp,
            loss_w_turns,
            loss_turns_half_life,
            loss_span,
            loot_weight,
            persistent_weight,
            loss_gradient,
            root_guards,
            chance_closed_stops_widening,
            rollout_policy,
            belief_dialect,
            leaf_playouts,
            tree_reuse,
            tt_hash_v2,
            net_expand_order,
            fpu_reduction,
            root_sequential_halving,
            effective_incoming,
            removed_counts_forms,
            win_hp_carry,
            win_potion_carry,
        } = self;
        Self {
            chance_widening_c: chance_widening_c.or(fallback.chance_widening_c),
            chance_widening_alpha: chance_widening_alpha.or(fallback.chance_widening_alpha),
            draw_widening_c: draw_widening_c.or(fallback.draw_widening_c),
            draw_widening_alpha: draw_widening_alpha.or(fallback.draw_widening_alpha),
            endturn_widening_c: endturn_widening_c.or(fallback.endturn_widening_c),
            endturn_widening_alpha: endturn_widening_alpha.or(fallback.endturn_widening_alpha),
            tie_eps: tie_eps.or(fallback.tie_eps),
            c_puct: c_puct.or(fallback.c_puct),
            loss_w_removed: loss_w_removed.or(fallback.loss_w_removed),
            loss_w_hp: loss_w_hp.or(fallback.loss_w_hp),
            loss_w_turns: loss_w_turns.or(fallback.loss_w_turns),
            loss_turns_half_life: loss_turns_half_life.or(fallback.loss_turns_half_life),
            loss_span: loss_span.or(fallback.loss_span),
            loot_weight: loot_weight.or(fallback.loot_weight),
            persistent_weight: persistent_weight.or(fallback.persistent_weight),
            loss_gradient: loss_gradient.or(fallback.loss_gradient),
            root_guards: root_guards.or(fallback.root_guards),
            chance_closed_stops_widening: chance_closed_stops_widening
                .or(fallback.chance_closed_stops_widening),
            rollout_policy: rollout_policy.or(fallback.rollout_policy),
            belief_dialect: belief_dialect.or(fallback.belief_dialect),
            leaf_playouts: leaf_playouts.or(fallback.leaf_playouts),
            tree_reuse: tree_reuse.or(fallback.tree_reuse),
            tt_hash_v2: tt_hash_v2.or(fallback.tt_hash_v2),
            net_expand_order: net_expand_order.or(fallback.net_expand_order),
            fpu_reduction: fpu_reduction.or(fallback.fpu_reduction),
            root_sequential_halving: root_sequential_halving.or(fallback.root_sequential_halving),
            effective_incoming: effective_incoming.or(fallback.effective_incoming),
            removed_counts_forms: removed_counts_forms.or(fallback.removed_counts_forms),
            win_hp_carry: win_hp_carry.or(fallback.win_hp_carry),
            win_potion_carry: win_potion_carry.or(fallback.win_potion_carry),
        }
    }
}

/// The Ironclad run agent's search knobs (GH #830), in the
/// `STS2_RUNAI_MCTS_OVERRIDES` vocabulary: the string the bundle arm of both
/// A10 gates set (PR #938; `exp/ironclad-combat-bundle` `10fa73b17`), so the
/// shipped default is literally the gated one. Their control arms set no
/// `STS2_*` variable.
///
/// The full-run gate (PR #938) measured these knobs together with the
/// built-in BC move prior in elite and boss fights
/// ([`sts2_mcts::rollout_net::builtin_prior_json`]) as one bundle: Ironclad
/// A10, champion a10_it700 `net_greedy`, 1000/4000 sims, 1,500 paired seeds,
/// +5.20 ± 1.18pp (19.5% → 24.7%), repeated 97.06% CI [+2.62, +7.78]. It had
/// no knobs-only or prior-only arm. Run agent only: the advisor and every
/// other `RecommendParams` surface keep the base dialect.
///
/// Applied by [`MctsCombatPolicy::search_config`] in the BeliefDraw dialect
/// ([`MctsCombatPolicy::ironclad_layer_on`]) under any explicit
/// `STS2_RUNAI_MCTS_OVERRIDES` knob. [`IRONCLAD_SEARCH_KNOBS_OFF`] switches
/// them off.
pub const IRONCLAD_SEARCH_KNOBS: &str =
    "removed_counts_forms=1,endturn_widening_c=23.148,draw_widening_c=4.0";

/// The off switch for [`IRONCLAD_SEARCH_KNOBS`]:
/// `STS2_RUNAI_MCTS_OVERRIDES=<this>` gives the Ironclad run agent the base
/// BeliefDraw dialect's values ([`sts2_mcts::BELIEF_DRAW_ENDTURN_WIDENING`],
/// [`sts2_mcts::BELIEF_DRAW_DRAW_WIDENING_C`], forms off) and is a no-op for
/// every other character. Together with `STS2_RUNAI_PRIOR_NET=off` it is the
/// gate's control arm. Honest mode only: under `STS2_RUNAI_HONEST=0` it would
/// overwrite the clairvoyant base. A test fails when the base dialect is
/// retuned and this string is not.
pub const IRONCLAD_SEARCH_KNOBS_OFF: &str =
    "removed_counts_forms=0,endturn_widening_c=5.787,draw_widening_c=2.0";

/// The run agent's per-character knob layer: [`IRONCLAD_SEARCH_KNOBS`] for an
/// Ironclad fight, nothing for any other character or a fight without one.
/// Parsed once.
fn character_search_preset(character: Option<CharacterId>) -> MctsKnobOverrides {
    static IRONCLAD: OnceLock<MctsKnobOverrides> = OnceLock::new();
    match character {
        Some(CharacterId::Ironclad) => {
            *IRONCLAD.get_or_init(|| parse_mcts_overrides(Some(IRONCLAD_SEARCH_KNOBS)))
        }
        _ => MctsKnobOverrides::default(),
    }
}

/// Pure `STS2_RUNAI_MCTS_OVERRIDES` env-value → [`MctsKnobOverrides`]
/// resolution, factored out of [`MctsCombatPolicy::new`] so the mapping
/// is unit-testable without mutating process env (same shape as
/// [`resolve_tie_eps`] / [`resolve_loss_gradient`] / [`resolve_root_guards`]).
///
/// Comma-separated `key=value` tokens sharing the knob vocabulary of
/// `sts2-bench`'s `combat_asc_eval` example (see that file's `Knobs`
/// table). Unset or empty (after trimming) ⇒ every slot `None` ⇒
/// [`MctsKnobOverrides::apply`] is a no-op ⇒ production behavior is
/// byte-identical to today. An unknown knob name or an unparseable value
/// PANICS rather than silently no-op'ing — this seam exists specifically
/// to gate promotion-deciding A/B campaigns through `run_arena` without a
/// rebuild per arm, and a typo silently running the deployed default arm
/// would poison hours of paired cells without any signal (the same
/// rationale as `STS2_RUNAI_HONEST` / `STS2_RUNAI_TIE_EPS` /
/// `STS2_RUNAI_LOSS_GRADIENT` / `STS2_RUNAI_ROOT_GUARDS`).
fn parse_mcts_overrides(env_value: Option<&str>) -> MctsKnobOverrides {
    let mut out = MctsKnobOverrides::default();
    let Some(raw) = env_value else {
        return out;
    };
    let raw = raw.trim();
    if raw.is_empty() {
        return out;
    }

    fn f32_of(key: &str, val: &str) -> f32 {
        val.trim().parse::<f32>().unwrap_or_else(|e| {
            panic!(
                "STS2_RUNAI_MCTS_OVERRIDES: knob {key}={val:?} is not a valid f32 ({e}) — \
                 refusing to silently fall back and poison the run"
            )
        })
    }
    fn playouts_of(key: &str, val: &str) -> u32 {
        match val.trim().parse::<u32>() {
            Ok(n) if n >= 1 => n,
            _ => panic!(
                "STS2_RUNAI_MCTS_OVERRIDES: knob {key}={val:?} is not an integer >= 1 — \
                 refusing to silently fall back and poison the run"
            ),
        }
    }
    fn carry_of(key: &str, val: &str) -> f32 {
        match val.trim().parse::<f32>() {
            Ok(c) if (0.0..=1.0).contains(&c) => c,
            _ => panic!(
                "STS2_RUNAI_MCTS_OVERRIDES: knob {key}={val:?} is not a number in [0, 1] — \
                 refusing to silently fall back and poison the run"
            ),
        }
    }
    fn bool_of(key: &str, val: &str) -> bool {
        match val.trim() {
            "1" | "true" => true,
            "0" | "false" => false,
            other => panic!(
                "STS2_RUNAI_MCTS_OVERRIDES: knob {key}={other:?} is not a valid bool \
                 (expected 0/1/true/false) — refusing to silently fall back and poison the run"
            ),
        }
    }

    for token in raw.split(',') {
        let token = token.trim();
        if token.is_empty() {
            continue;
        }
        let Some((key, val)) = token.split_once('=') else {
            panic!(
                "STS2_RUNAI_MCTS_OVERRIDES: token {token:?} is not \"key=value\" — expected \
                 comma-separated k=v pairs, e.g. \"endturn_widening_c=5.787,c_puct=2.5\""
            );
        };
        let key = key.trim();
        let val = val.trim();
        match key {
            "chance_widening_c" => out.chance_widening_c = Some(f32_of(key, val)),
            "chance_widening_alpha" => out.chance_widening_alpha = Some(f32_of(key, val)),
            "draw_widening_c" => out.draw_widening_c = Some(f32_of(key, val)),
            "draw_widening_alpha" => out.draw_widening_alpha = Some(f32_of(key, val)),
            "endturn_widening_c" => out.endturn_widening_c = Some(f32_of(key, val)),
            "endturn_widening_alpha" => out.endturn_widening_alpha = Some(f32_of(key, val)),
            "tie_eps" => out.tie_eps = Some(f32_of(key, val)),
            "c_puct" => out.c_puct = Some(f32_of(key, val)),
            "loss_w_removed" => out.loss_w_removed = Some(f32_of(key, val)),
            "loss_w_hp" => out.loss_w_hp = Some(f32_of(key, val)),
            "loss_w_turns" => out.loss_w_turns = Some(f32_of(key, val)),
            "loss_turns_half_life" => out.loss_turns_half_life = Some(f32_of(key, val)),
            "loss_span" => out.loss_span = Some(f32_of(key, val)),
            "loot_weight" => out.loot_weight = Some(f32_of(key, val)),
            "persistent_weight" => out.persistent_weight = Some(f32_of(key, val)),
            "win_hp_carry" => out.win_hp_carry = Some(carry_of(key, val)),
            "win_potion_carry" => out.win_potion_carry = Some(carry_of(key, val)),
            "loss_gradient" => out.loss_gradient = Some(bool_of(key, val)),
            "root_guards" => out.root_guards = Some(bool_of(key, val)),
            "belief_dialect" => out.belief_dialect = Some(bool_of(key, val)),
            "leaf_playouts" => out.leaf_playouts = Some(playouts_of(key, val)),
            "tree_reuse" => out.tree_reuse = Some(bool_of(key, val)),
            "tt_hash_v2" => out.tt_hash_v2 = Some(bool_of(key, val)),
            "net_expand_order" => out.net_expand_order = Some(bool_of(key, val)),
            "fpu_reduction" => out.fpu_reduction = Some(f32_of(key, val)),
            "root_sequential_halving" => {
                out.root_sequential_halving = Some(bool_of(key, val));
            }
            "effective_incoming" => out.effective_incoming = Some(bool_of(key, val)),
            "removed_counts_forms" => out.removed_counts_forms = Some(bool_of(key, val)),
            "chance_closed_stops_widening" => {
                out.chance_closed_stops_widening = Some(bool_of(key, val));
            }
            "rollout_policy" => {
                out.rollout_policy = Some(match val {
                    "heuristic" => sts2_mcts::RolloutPolicyKind::Heuristic,
                    "uniform" => sts2_mcts::RolloutPolicyKind::Uniform,
                    "net" => {
                        // Load (or fail) a STS2_ROLLOUT_NET file now, at
                        // policy construction, not inside the first rollout
                        // of the first search.
                        let _ = sts2_mcts::rollout_net::global_net();
                        sts2_mcts::RolloutPolicyKind::Net
                    }
                    // The playout net was `learned` before #808 gave that
                    // name to the fitted table. Fail rather than silently
                    // run the other policy from an old command line.
                    "learned" => panic!(
                        "STS2_RUNAI_MCTS_OVERRIDES: rollout_policy=learned was renamed — the \
                         playout net is `net` (the honest-mode default; STS2_ROLLOUT_NET swaps \
                         its weights)"
                    ),
                    other => panic!(
                        "STS2_RUNAI_MCTS_OVERRIDES: rollout_policy={other:?} — expected \
                         heuristic, uniform or net"
                    ),
                });
            }
            other => panic!(
                "STS2_RUNAI_MCTS_OVERRIDES: unknown knob {other:?} — typo? known knobs: \
                 {MCTS_OVERRIDE_KNOB_NAMES}"
            ),
        }
    }
    out
}

/// The hidden-info mode [`MctsCombatPolicy::build_search`] searches under,
/// resolved from `STS2_RUNAI_HONEST` (unset ⇒ `BeliefDraw`). Public so a
/// caller that views the root the way the search does (the turn-plan
/// enumerator, GH #830) reads the same mode instead of hardcoding one.
#[must_use]
pub fn deployed_hidden_info() -> sts2_mcts::HiddenInfoMode {
    resolve_hidden_info(std::env::var("STS2_RUNAI_HONEST").ok().as_deref())
}

fn resolve_hidden_info(env_value: Option<&str>) -> sts2_mcts::HiddenInfoMode {
    match env_value {
        None | Some("belief") => sts2_mcts::HiddenInfoMode::BeliefDraw,
        Some("0") => sts2_mcts::HiddenInfoMode::Clairvoyant,
        Some("root") => sts2_mcts::HiddenInfoMode::RootScramble,
        Some("full") => sts2_mcts::HiddenInfoMode::DrawChance,
        Some(other) => panic!(
            "STS2_RUNAI_HONEST={other:?} is not a recognized mode (expected unset, \"0\", \
             \"root\", \"full\", or \"belief\") — refusing to silently fall back and poison \
             the run"
        ),
    }
}

/// The BeliefDraw exploration constant for a given simulation budget.
///
/// Re-exported from [`sts2_mcts`], which owns the definition so `sts2-wire` —
/// the browser workbench's search, which deliberately does NOT depend on this
/// crate — can resolve the same constant. Kept exported here because every
/// deployed caller reaches it through `sts2_runai::policy`.
pub use sts2_mcts::belief_draw_c_puct;

/// Build the [`MctsConfig`] for one combat-search decision: the shared
/// fields every mode gets, plus the F2 retune (bead `sts2-rs-09x5n`, task
/// 7b) applied ONLY under [`sts2_mcts::HiddenInfoMode::BeliefDraw`].
///
/// Factored out of [`MctsCombatPolicy::mcts_pick`] as a pure function (no
/// env reads) so the baked retune constants are unit-testable directly
/// against the resolved mode, instead of only indirectly via a live
/// search.
///
/// `clairvoyance_probe`'s closed-loop sweep (500-fight paired random-deck
/// act-1 corpus, `--mode closedloop --deck random --states 50 --replicas
/// 10 --sims 1000`) found stock BeliefDraw's search hyperparameters
/// under-explore its own honest chance nodes — c_puct=√2 (default)
/// recovers only 27.8% of the TRUE-vs-F0 clairvoyance gap. Raising
/// exploration to c_puct=2.5 (lightspeed-fork-informed guess, then
/// confirmed as the dominant lever by ablating each axis separately:
/// c_puct alone → 38.9%, draw widening alone → 22.2%/no better than
/// stock) plus a mild draw-widening bump (draw_widening_c 1.0→2.0, alpha
/// unchanged) landed 44.4% at n=500 and reconfirmed significant (McNemar
/// p=0.016) at n=1000 — 41.5%, still short of the 50% G2 bar but the ONLY
/// one of the two honest-search retunes (F1's is a rejected noise peak,
/// see `sts2-rs-k1z14`) that stays positive across the sims ladder
/// (1000/2000/4000, whereas stock F2 and F1-retuned both reverse negative
/// by 2000-4000 sims). Applied ONLY under `BeliefDraw` — every other mode
/// keeps `MctsConfig::default()`'s c_puct/draw_widening_c/alpha
/// byte-identical to pre-`sts2-rs-09x5n` behavior.
///
/// That 2.5 is no longer a flat constant: it is the value
/// [`belief_draw_c_puct`] returns at every DEPLOYED budget, and the
/// function steps down below 500 sims. See its doc comment for the curve.
/// The EndTurn-widening retune layered on top (PR #467) is documented on
/// [`MctsConfig::apply_mode_dialect`], which applies all three constants.
pub fn mcts_config_for(
    hidden_info: sts2_mcts::HiddenInfoMode,
    max_simulations: u32,
    seed: u64,
    max_rollout_depth: u32,
    terminal_eval_mode: sts2_mcts::TerminalEvalMode,
) -> MctsConfig {
    let mut config = MctsConfig {
        max_simulations,
        seed,
        max_rollout_depth,
        terminal_eval_mode,
        hidden_info,
        ..MctsConfig::default()
    };
    // The deployed dialect — BeliefDraw's budget-keyed c_puct, the F2
    // draw widening and the PR #467 EndTurn retune — lives in sts2-mcts so
    // the advisor and the workbench hint (`RecommendParams::mcts_config`)
    // apply the identical constants (GH #758).
    //
    // Rollback of the EndTurn retune: `STS2_RUNAI_MCTS_OVERRIDES=
    // "endturn_widening_c=0.5,endturn_widening_alpha=0.4"` restores the
    // pre-retune dialect for this agent without a rebuild.
    config.apply_mode_dialect();
    config
}

impl FallbackCounts {
    /// Total combat decisions that fell back to the meta policy.
    #[must_use]
    pub fn total(self) -> u64 {
        self.no_outcome + self.untranslatable + self.not_surfaced
    }
}

/// Per-reason tally of [`veto_lethal_event_choice`] outcomes (GH #300).
///
/// The veto silently overrides a trained policy's pick, which is exactly
/// the kind of intervention that must be countable rather than inferred
/// from win-rate noise — the same rationale as [`FallbackCounts`] (issue
/// #164). A sweep that reports `vetoed: 0` has established the guard never
/// fired; without a counter, "the guard is working" and "the guard is
/// unreachable" produce identical evidence.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EventVetoCounts {
    /// The pick was lethal and was swapped for a surviving option.
    pub vetoed: u64,
    /// The pick was lethal and *every* option was — the pick stood, as the
    /// game itself would allow.
    pub all_lethal: u64,
    /// The picked option could not be previewed, so lethality could not be
    /// established. The pick stands. Non-zero means the guard has a blind
    /// spot on this content, not that the content is safe.
    pub unpreviewable_pick: u64,
    /// An *alternative* could not be previewed and was skipped rather than
    /// scored as HP-neutral. Non-zero means the swap had fewer candidates
    /// than the option list suggests.
    pub unpreviewable_alt: u64,
}

impl EventVetoCounts {
    /// Total event decisions on which the guard changed or considered
    /// changing the meta policy's pick.
    #[must_use]
    pub fn total(self) -> u64 {
        self.vetoed + self.all_lethal + self.unpreviewable_pick
    }
}

impl<M: RunPolicy> MctsCombatPolicy<M> {
    /// Borrow the wrapped meta policy (escalation-count extraction
    /// after a run — bead `sts2-rs-40ikw.6`).
    #[must_use]
    pub fn meta(&self) -> &M {
        &self.meta
    }

    #[must_use]
    pub fn new(meta: M, budget: CombatBudget, run_seed: i32) -> Self {
        Self {
            meta,
            budget,
            run_seed,
            decisions: 0,
            decisions_elite: 0,
            decisions_boss: 0,
            decisions_motif: 0,
            count_motifs: std::env::var_os("STS2_RUNAI_COUNT_MOTIFS").is_some(),
            sims_spent: 0,
            fallbacks: FallbackCounts::default(),
            event_vetoes: EventVetoCounts::default(),
            last_combat_search: None,
            // Read once here, not per pick: ~80 combat decisions per run
            // don't need 80 env reads, and a knob that can change
            // mid-run is a knob a sweep can't trust.
            tie_eps: resolve_tie_eps(std::env::var("STS2_RUNAI_TIE_EPS").ok().as_deref()),
            loss_gradient: resolve_loss_gradient(
                std::env::var("STS2_RUNAI_LOSS_GRADIENT").ok().as_deref(),
            ),
            root_guards: resolve_root_guards(
                std::env::var("STS2_RUNAI_ROOT_GUARDS").ok().as_deref(),
            ),
            overrides: parse_mcts_overrides(
                std::env::var("STS2_RUNAI_MCTS_OVERRIDES").ok().as_deref(),
            ),
            potion_oc: potion_oc::resolve_potion_oc(
                std::env::var("STS2_RUNAI_POTION_OC").ok().as_deref(),
                std::env::var("STS2_RUNAI_POTION_WEIGHT").ok().as_deref(),
                std::env::var("STS2_RUNAI_POTION_OC_SCALE").ok().as_deref(),
                std::env::var("STS2_RUNAI_POTION_OC_HP_RATE")
                    .ok()
                    .as_deref(),
            ),
            drink_jitter: potion_oc::resolve_drink_jitter(
                std::env::var("STS2_RUNAI_DRINK_JITTER").ok().as_deref(),
            ),
            potion_ablation: potion_ablation::PotionAblation::resolve(
                std::env::var("STS2_RUNAI_POTION_ABLATION").ok().as_deref(),
            ),
            boss_carry: resolve_boss_carry(std::env::var("STS2_RUNAI_BOSS_CARRY").ok().as_deref()),
            potion_class_scale: potion_oc::ClassScale::resolve(
                std::env::var("STS2_RUNAI_POTION_OC_CLASS_SCALE")
                    .ok()
                    .as_deref(),
            ),
            potion_timing: potion_timing::PotionTiming::resolve(
                std::env::var("STS2_RUNAI_POTION_TIMING").ok().as_deref(),
            ),
            oracle_filled_floor: None,
            slots_granted: false,
            retained: None,
            tree_reuse: TreeReuseCounts::default(),
            last_retained_visits: None,
            root_filter: resolve_root_filter(
                std::env::var("STS2_RUNAI_TURNPLAN_ROOT_FILTER")
                    .ok()
                    .as_deref(),
            ),
            last_root_filter: None,
        }
    }

    /// What the turn-plan root filter did at the most recent combat
    /// decision; `None` with `STS2_RUNAI_TURNPLAN_ROOT_FILTER` unset.
    #[must_use]
    pub fn last_root_filter(&self) -> Option<&RootFilterRecord> {
        self.last_root_filter.as_ref()
    }

    /// What `tree_reuse` (GH #830) did over this policy's lifetime. All
    /// zero unless `STS2_RUNAI_MCTS_OVERRIDES` sets `tree_reuse=1`.
    #[must_use]
    pub fn tree_reuse_counts(&self) -> TreeReuseCounts {
        self.tree_reuse
    }

    /// Root visits the most recent combat search inherited from the
    /// decision before (`tree_reuse`, GH #830): `Some(n)` when it reused a
    /// tree, `None` when it built a fresh one or did not search.
    #[must_use]
    pub fn last_retained_visits(&self) -> Option<u32> {
        self.last_retained_visits
    }

    fn tree_reuse_on(&self) -> bool {
        self.overrides.tree_reuse == Some(true)
    }

    /// The retained tree, rerooted onto `combat` under `config` — or
    /// `None` (a fresh tree) with the reason tallied. `retained` is the
    /// tree kept by the decision just before this one; `forced` marks a
    /// move that is not searched, which is tallied separately.
    fn reuse_retained(
        &mut self,
        retained: Option<RetainedSearch>,
        rm: &RunManager,
        combat: &CombatState,
        config: MctsConfig,
        forced: bool,
    ) -> Option<(Mcts<sts2_mcts::SharedPolicy>, u32)> {
        let Some(RetainedSearch {
            mut mcts,
            action,
            floor,
        }) = retained
        else {
            if !forced {
                self.tree_reuse.fresh_no_tree += 1;
            }
            return None;
        };
        let kept = if floor == rm.run_state.total_floor {
            mcts.reroot_observed(&action, combat, config)
        } else {
            Err(sts2_mcts::RerootMiss::UnsupportedMode)
        };
        let t = &mut self.tree_reuse;
        if forced {
            if kept.is_ok() {
                t.forced_carried += 1;
            } else {
                t.forced_dropped += 1;
            }
        }
        match kept {
            Ok(k) => {
                if !forced {
                    t.reused += 1;
                    t.reused_via_chance += u64::from(k.via_chance);
                }
                Some((mcts, k.retained_visits))
            }
            Err(miss) => {
                if !forced {
                    match miss {
                        sts2_mcts::RerootMiss::NotExpanded => t.fresh_not_expanded += 1,
                        sts2_mcts::RerootMiss::Mismatch => t.fresh_mismatch += 1,
                        sts2_mcts::RerootMiss::OutcomeUnseen => t.fresh_outcome_unseen += 1,
                        sts2_mcts::RerootMiss::EdgeSetChanged => t.fresh_edge_set += 1,
                        sts2_mcts::RerootMiss::UnsupportedMode => t.fresh_other += 1,
                    }
                }
                None
            }
        }
    }

    /// The combat search budget this policy is currently spending.
    #[must_use]
    pub fn budget(&self) -> CombatBudget {
        self.budget
    }

    /// Retarget the combat search budget mid-run.
    ///
    /// Everything else — the run seed, the decision counter that derives
    /// each search's seed, the fallback/veto tallies — is deliberately
    /// left alone, so a budget change is not a policy reset. That is what
    /// the workbench's "stop, raise the sims, resume" flow needs: the run
    /// continues from where it paused, only searching harder. Rebuilding
    /// the policy instead would rewind `decisions` to 0 and replay the
    /// same per-decision seeds the run already used.
    ///
    /// The A/B lanes never call this — they set the budget once at
    /// construction, so an arm's budget still cannot drift mid-run.
    pub fn set_budget(&mut self, budget: CombatBudget) {
        self.budget = budget;
    }

    /// Per-reason tally of lethal-event-pick vetoes (GH #300). All-zero
    /// until the guard fires; `vetoed == 0` with a non-zero
    /// `unpreviewable_pick` means the guard was reached but could not rule.
    #[must_use]
    pub fn event_vetoes(&self) -> EventVetoCounts {
        self.event_vetoes
    }

    /// Per-reason tally of combat decisions handed back to the meta
    /// policy (issue #164). `(0, 0, 0)` until a fallback fires.
    #[must_use]
    pub fn fallbacks(&self) -> FallbackCounts {
        self.fallbacks
    }

    /// Env-gated per-event-decision probe (GH #302 item 4). Set
    /// `STS2_RUNAI_EVENT_BLEED_LOG=<path>` and every event decision this
    /// policy makes appends one JSON line there; unset (the default) this
    /// is a single `env::var` miss and the deployed run is untouched.
    ///
    /// It logs the meta policy's RAW pick alongside the post-veto one, so
    /// the two numbers #302 asks for both fall out of one sweep:
    ///
    /// - **event HP bleed** = `-sum(raw_hp_delta)`. Attributed to the
    ///   *raw* pick because that is what the net actually wants; the
    ///   post-veto `final_hp_delta` measures what the run actually paid.
    ///   A retrained net should shrink the former, not just the latter.
    /// - **veto fires** = `count(veto_fired)`. #302's acceptance criterion
    ///   is that this trends to ~0, i.e. the guard becomes dead code
    ///   because the policy no longer needs saving from itself.
    ///
    /// Deltas come from `preview_option_effect` (GH #237), which resolves
    /// the option on a throwaway clone — exact, not description-parsed,
    /// so it prices SlipperyBridge's escalating page cost correctly.
    ///
    /// Append-mode single-`write_all` per line keeps the arena's
    /// process-per-seed model safe: `O_APPEND` writes of one short line do
    /// not interleave, so all seeds can share one log file. Errors are
    /// deliberately swallowed — a probe must never fail a run.
    fn log_event_choice(
        &self,
        rm: &RunManager,
        actions: &[RunAction],
        picked: usize,
        final_idx: usize,
    ) {
        let Ok(path) = std::env::var("STS2_RUNAI_EVENT_BLEED_LOG") else {
            return;
        };
        let RunAction::EventChoice { idx: raw_opt } = actions[picked] else {
            return;
        };
        let RunAction::EventChoice { idx: final_opt } = actions[final_idx] else {
            return;
        };
        let preview =
            |opt: usize| sts2_content::events::dispatch::preview_option_effect(&rm.run_state, opt);
        let raw = preview(raw_opt);
        let record = serde_json::json!({
            "seed": self.run_seed,
            "floor": rm.run_state.total_floor,
            "act": rm.run_state.act_index,
            "event": rm.run_state.current_event.map(sts2_core::EventId::python_name),
            "hp": rm.run_state.current_hp,
            "max_hp": rm.run_state.max_hp,
            "n_options": actions
                .iter()
                .filter(|a| matches!(a, RunAction::EventChoice { .. }))
                .count(),
            "raw_opt": raw_opt,
            "raw_hp_delta": raw.map(|d| d.hp_delta),
            "raw_gold_delta": raw.map(|d| d.gold_delta),
            "final_opt": final_opt,
            "final_hp_delta": preview(final_opt).map(|d| d.hp_delta),
            "veto_fired": picked != final_idx,
        });
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
        {
            use std::io::Write;
            let _ = f.write_all(format!("{record}\n").as_bytes());
        }
    }

    /// Combat decisions for which the MCTS was consulted (the denominator
    /// for a [`FallbackCounts`] rate).
    #[must_use]
    pub fn combat_decisions(&self) -> u64 {
        self.decisions
    }

    /// `(elite decisions, boss decisions, motif-tagged common decisions,
    /// simulations requested)` over the run — the compute-neutrality
    /// readout for budget arms (GH #446). Forced moves count as decisions
    /// but request no simulations.
    #[must_use]
    pub fn combat_budget_spent(&self) -> (u64, u64, u64, u64) {
        (
            self.decisions_elite,
            self.decisions_boss,
            self.decisions_motif,
            self.sims_spent,
        )
    }

    /// Root MCTS distribution of the most recent combat decision, for
    /// stepwise drivers that surface the visit/Q panel.
    #[must_use]
    pub fn last_combat_search(&self) -> Option<&CombatSearchRecord> {
        self.last_combat_search.as_ref()
    }

    /// Run one MCTS search and return the index of the matching
    /// `RunAction`, or `None` to fall back to the meta policy.
    /// `boss-oracle` ablation helper: fill the live combat belt to
    /// capacity, once per boss floor. Both the combat belt and the run
    /// belt are written so the top-up survives `exit_combat`'s
    /// write-back and reads consistently everywhere.
    fn fill_boss_belt(&mut self, rm: &mut RunManager) {
        if rm.phase() != Phase::Combat {
            return;
        }
        let floor = rm.run_state.total_floor;
        if self.oracle_filled_floor == Some(floor) {
            return;
        }
        let is_boss = rm
            .run_state
            .current_room
            .is_some_and(|r| r.room_type == RoomType::Boss);
        if !is_boss {
            return;
        }
        let character = rm.run_state.character_id;
        let run_seed = self.run_seed;
        let Some(combat) = rm.combat.as_mut() else {
            return;
        };
        if combat.is_over {
            return;
        }
        let cap = if combat.player.potion_slots == 0 {
            usize::try_from(rm.run_state.max_potion_slots.max(0)).unwrap_or(0)
        } else {
            combat.player.potion_slots as usize
        };
        let mut belt: Vec<_> = combat.player.potions.to_vec();
        potion_ablation::fill_belt_to_cap(&mut belt, cap, character, run_seed, floor);
        combat.player.potions = belt.iter().copied().collect();
        rm.run_state.potions = belt.into_iter().collect();
        self.oracle_filled_floor = Some(floor);
    }

    fn mcts_pick(
        &mut self,
        rm: &RunManager,
        combat: &CombatState,
        actions: &[RunAction],
    ) -> Option<usize> {
        // Clear stale state unconditionally: a fallback or empty decision
        // must not surface a record from a prior decision.
        self.last_combat_search = None;
        self.last_retained_visits = None;
        self.last_root_filter = None;

        let counter = self.decisions;
        self.decisions += 1;

        let tier = BudgetTier::from_room(rm.run_state.current_room.as_ref().map(|r| r.room_type));
        let is_boss = tier == BudgetTier::Boss;
        // The motif scan (`legal_actions` + the lethal detector) is only paid
        // when something reads it: a motif budget, or an explicit request to
        // count motif decisions (`STS2_RUNAI_COUNT_MOTIFS=1`, used to size
        // the `motif-focus` arm from an `eval-cheap` run). The deployed
        // budget therefore does exactly what it did before this field.
        let motif = tier == BudgetTier::Common
            && (self.budget.motif_sims.is_some() || self.count_motifs)
            && CombatBudget::budget_motif_at(combat);
        let sims = self.budget.sims_for_with(tier, motif);
        match tier {
            BudgetTier::Elite => self.decisions_elite += 1,
            BudgetTier::Boss => self.decisions_boss += 1,
            BudgetTier::Common => {
                if motif {
                    self.decisions_motif += 1;
                }
            }
        }

        let seed_bits = u64::from(u32::from_ne_bytes(self.run_seed.to_ne_bytes()));
        let seed = splitmix64(seed_bits ^ (counter + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        // `tree_reuse` (GH #830): the tree the decision before kept, if any.
        // Taken here unconditionally, so it serves this decision or none.
        let retained = self.retained.take();

        // Forced move: the search could only return the one legal action,
        // so skip it (a forced EndTurn used to pay the full budget). The
        // counter above has already advanced, so every later decision keeps
        // its seed.
        if let Some(only) = self.forced_root_action(combat, sims) {
            if self.root_filter.is_some() {
                self.last_root_filter = Some(RootFilterRecord {
                    applied: false,
                    fallback: Some("forced"),
                    legal_n: 1,
                    plans_n: 0,
                    kept_n: 0,
                    allowed_n: 1,
                    chosen_via: None,
                    filter_us: 0,
                });
            }
            // `tree_reuse`: carry a kept tree across the move it did not
            // search, rooted where this decision would have searched it.
            if retained.is_some() {
                let (config, _, prior) = self.pick_search_setup(rm, combat, is_boss, sims, seed);
                if let Some((mut mcts, _)) = self.reuse_retained(retained, rm, combat, config, true)
                {
                    // This decision's prior, as on a searched kept tree.
                    mcts.set_policy(prior);
                    self.retained = Some(RetainedSearch {
                        mcts: mcts.with_prior_cache(prior_cache_from_env()),
                        action: only.clone(),
                        floor: rm.run_state.total_floor,
                    });
                }
            }
            let stub = sts2_mcts::RootOutcome {
                action: only.clone(),
                visits: 0,
                wins: 0,
                losses: 0,
                ties: 0,
                value_sum: 0.0,
                value_sq_sum: 0.0,
                hp_sum: 0.0,
                hp_sq_sum: 0.0,
                hp_n: 0,
                mean_value: 0.0,
            };
            return self.surface_pick(
                only,
                vec![stub],
                sts2_mcts::ExpandOrderStats::default(),
                actions,
            );
        }
        self.sims_spent += u64::from(sims);

        let (config, leaf_eval, prior) = self.pick_search_setup(rm, combat, is_boss, sims, seed);
        // Turn-plan root filter (GH #830): which root actions the search may
        // pick. Computed from `combat` alone (no search state), so it is the
        // same whether or not a tree is reused below.
        let filter = self.root_filter.as_ref().map(|(scorer, k)| {
            let t0 = std::time::Instant::now();
            let f = sts2_mcts::turn_plan_scorer::root_filter(
                combat,
                scorer,
                *k,
                sts2_mcts::turn_plan::PlanCap::default(),
            );
            let us = u64::try_from(t0.elapsed().as_micros()).unwrap_or(u64::MAX);
            (f, us)
        });
        let reuse = self.tree_reuse_on();
        let reused = if reuse {
            self.tree_reuse.searched += 1;
            self.tree_reuse.new_sims += u64::from(sims);
            self.reuse_retained(retained, rm, combat, config, false)
        } else {
            None
        };
        let mut mcts = match reused {
            // The new decision's prior / leaf evaluator, like a fresh search.
            Some((mut kept, retained_visits)) => {
                self.tree_reuse.retained_visits += u64::from(retained_visits);
                self.last_retained_visits = Some(retained_visits);
                kept.set_policy(prior);
                kept.with_prior_cache(prior_cache_from_env())
                    .with_leaf_eval(leaf_eval)
            }
            None => Mcts::with_policy(combat.clone(), config, prior)
                .with_prior_cache(prior_cache_from_env())
                .with_leaf_eval(leaf_eval),
        };
        if let Some((f, _)) = filter.as_ref().filter(|(f, _)| f.fallback.is_none()) {
            let allowed: Vec<Action> = f.allowed.iter().map(|(a, _)| a.clone()).collect();
            mcts = mcts.with_root_allowed(&allowed);
        }

        // THE pick is `Mcts::pick_root_action` (via `try_search_n`) —
        // lost-position override (GH #32), draw-window bonus (GH #403)
        // and the tie-band visits tie-break, identical to every other
        // search entry point. The 2026-08-09 A8 audit found this path
        // discarding `search_n`'s return and re-deriving its own
        // argmax, which dead-coded the shipped #403 fix on the deployed
        // agent and diverged from the engine pick on exact ties; do NOT
        // reintroduce a local argmax here.
        let chosen = mcts.try_search_n(sims).ok();
        let root_outcomes = mcts.root_outcomes();
        // Per decision: a rerooted tree restarts this counter, like a fresh one.
        let expand_order = mcts.expand_order_stats();
        // Record *why* a fallback fires instead of collapsing every cause
        // to a silent `None` (issue #164).
        if let Some((f, us)) = filter {
            let via = chosen.as_ref().and_then(|c| {
                f.allowed
                    .iter()
                    .find(|(a, _)| a == c)
                    .map(|&(_, top)| if top { "topk" } else { "anchor_only" })
            });
            self.last_root_filter = Some(RootFilterRecord {
                applied: f.fallback.is_none(),
                fallback: f
                    .fallback
                    .map(sts2_mcts::turn_plan_scorer::FilterFallback::as_str),
                legal_n: f.legal_n,
                plans_n: f.plans_n,
                kept_n: f.kept_n,
                allowed_n: if f.fallback.is_none() {
                    f.allowed.len()
                } else {
                    f.legal_n
                },
                chosen_via: if f.fallback.is_none() { via } else { None },
                filter_us: us,
            });
        }
        let Some(chosen) = chosen else {
            self.fallbacks.no_outcome += 1;
            self.log_fallback("no_outcome", None);
            return None;
        };
        // In-fight degradation seam: rewrite the pick BEFORE translation
        // so the substitute goes through the identical legality path.
        // Keyed per (seed, floor, decision) — no engine RNG touched.
        let chosen = match self.potion_timing {
            Some(t) => {
                let coin = splitmix64(
                    u64::from(self.run_seed as u32)
                        ^ (u64::from(rm.run_state.total_floor as u32) << 32)
                        ^ counter.wrapping_mul(0x9E37_79B9_7F4A_7C15),
                );
                t.perturb(&chosen, &root_outcomes, coin).unwrap_or(chosen)
            }
            None => chosen,
        };
        if !reuse {
            return self.surface_pick(chosen, root_outcomes, expand_order, actions);
        }
        let picked = self.surface_pick(chosen.clone(), root_outcomes, expand_order, actions);
        // `tree_reuse`: keep this tree for the next decision, which reroots
        // it only if that decision's observation matches what `chosen` led
        // to here. A fallback keeps nothing: the meta's move is not ours.
        if picked.is_some() {
            self.retained = Some(RetainedSearch {
                mcts,
                action: chosen,
                floor: rm.run_state.total_floor,
            });
        }
        picked
    }

    /// The deployed combat search for `combat`, unsearched and fresh (no
    /// tree reuse): [`Self::pick_search_setup`]'s config, prior and leaf
    /// evaluator, exactly as `mcts_pick` builds a fresh search. The
    /// turn-plan teacher ([`crate::turn_plan_teacher`]) prices plans with
    /// it, so the two can never drift onto different configs.
    ///
    /// `&mut self` because the potion opportunity-cost table advances its
    /// own state on every build (`charge_slots_ctx`).
    pub fn build_search(
        &mut self,
        rm: &RunManager,
        combat: &CombatState,
        sims: u32,
        seed: u64,
    ) -> Mcts<sts2_mcts::SharedPolicy> {
        let is_boss =
            BudgetTier::from_room(rm.run_state.current_room.as_ref().map(|r| r.room_type))
                == BudgetTier::Boss;
        let (config, leaf_eval, prior) = self.pick_search_setup(rm, combat, is_boss, sims, seed);
        Mcts::with_policy(combat.clone(), config, prior)
            .with_prior_cache(prior_cache_from_env())
            .with_leaf_eval(leaf_eval)
    }

    /// The search `mcts_pick` runs for one decision: [`Self::search_config`]'s
    /// config and leaf evaluator for the fight's character
    /// (`combat.player.character`), plus the root-only seams it adds on top —
    /// the PUCT move prior (`use_puct_prior` and the returned prior: the
    /// built-in Ironclad prior in elite and boss fights by default, see
    /// [`prior_choice`]), the potion opportunity charge (`potion_charge_q`)
    /// and the drink masks (`root_drink_mask`). Those three config fields are
    /// the whole difference from `search_config`, and none of them is read by
    /// a playout; a test pins that, so the playouts [`deployed_value_config`]
    /// scores (`sts2_rs.values_many`, GH #830) stay the pilot's own.
    fn pick_search_setup(
        &mut self,
        rm: &RunManager,
        combat: &CombatState,
        is_boss: bool,
        sims: u32,
        seed: u64,
    ) -> (MctsConfig, LeafEval, sts2_mcts::SharedPolicy) {
        let character = combat.player.character;
        let (mut config, leaf_eval) = self.search_config(sims, seed, character);
        // Boss win value by what the fight carries forward (GH #771). An
        // explicit carry in the override bundle wins.
        if self.boss_carry != BossCarry::Off
            && config.win_hp_carry.is_none()
            && config.win_potion_carry.is_none()
        {
            let run = &rm.run_state;
            let carry = boss_win_carry(
                is_boss,
                run.is_last_act(),
                run.awaiting_second_boss(),
                run.ascension_level,
            );
            let last_boss = is_boss && run.is_last_act() && !run.awaiting_second_boss();
            if let Some((hp, potions)) =
                carry.filter(|_| last_boss || self.boss_carry == BossCarry::All)
            {
                config.win_hp_carry = Some(hp);
                config.win_potion_carry = Some(potions);
            }
        }
        // `SharedPolicy::uniform()` + `use_puct_prior = false` is documented
        // byte-identical to `Mcts::<UniformPolicy>::new`, so the prior-less
        // production path is unchanged by routing through `with_policy`.
        // (`use_puct_prior` is not an override knob, so setting it after the
        // override bundle in `search_config` is the same as before it.)
        // The combat net's prior (`STS2_RUNAI_COMBAT_PRIOR=1`) replaces the
        // move prior, as before the built-in one existed.
        let tier = BudgetTier::from_room(rm.run_state.current_room.as_ref().map(|r| r.room_type));
        let move_prior = move_prior(prior_net_env(), character, tier, self.ironclad_layer_on());
        let prior = match (combat_prior_net_from_env(), move_prior) {
            (Some(net), _) => {
                config.use_puct_prior = true;
                sts2_mcts::SharedPolicy::new(sts2_mcts::LearnedPolicyPrior::new(net))
            }
            (None, Some(prior)) => {
                config.use_puct_prior = true;
                prior
            }
            (None, None) => sts2_mcts::SharedPolicy::uniform(),
        };
        // Root-level potion opportunity-cost charge (GH #447 follow-up).
        // Slot-indexed from the CURRENT combat belt: each searched ply
        // builds a fresh `Mcts`, so the mapping cannot go stale across
        // mid-fight drinks. Applied after the override bundle — the
        // bundle has no charge slot, and the table is itself env-gated.
        if let Some(oc) = self.potion_oc.as_mut() {
            let class_scale = self.potion_class_scale;
            config.potion_charge_q = Some(oc.charge_slots_ctx(rm, combat, is_boss, class_scale));
        }
        // Drink-jitter seam: per-FIGHT coin (keyed on floor, not on the
        // per-decision counter) so a masked fight is masked for every
        // ply in it — partial masking would let the search defer the
        // drink one ply and defeat the exogeneity argument.
        if let Some(jitter) = self.drink_jitter.as_ref() {
            if jitter.masks(self.run_seed, rm.run_state.total_floor) {
                config.root_drink_mask = true;
            }
        }
        // Headroom ablation: `no-drinks` masks EVERY fight (the jitter
        // seam masks a random subset), so the arm never spends a potion
        // and the baseline−arm delta is the potion system's total value.
        if self.potion_ablation == Some(potion_ablation::PotionAblation::NoDrinks) {
            config.root_drink_mask = true;
        }
        (config, leaf_eval, prior)
    }

    /// The search configuration and leaf evaluator one combat search of
    /// `sims` simulations with search seed `seed` in a `character` fight runs
    /// under: the base dialect ([`mcts_config_for`]), every env seam this
    /// policy reads (`STS2_RUNAI_ROLLOUT_DEPTH` / `_TERMINAL_MODE` / `_LEAF` /
    /// `_HONEST` per call, the tie-band / loss-gradient / root-guard /
    /// override bundle resolved in [`Self::new`]) and, in the BeliefDraw
    /// dialect ([`Self::ironclad_layer_on`]), the run agent's character
    /// layer: [`IRONCLAD_SEARCH_KNOBS`] for an Ironclad fight (GH #830),
    /// under any knob the override bundle sets. Every other character, and
    /// `None`, get exactly the config they got before that layer existed. It
    /// leaves out only the root-only seams (the PUCT prior, the potion charge
    /// and drink masks).
    ///
    /// `character` is the fight's `combat.player.character`, the key the
    /// playout net uses too. Public so a caller that scores playouts outside
    /// the search (`sts2_rs.LearnedPlayout.play_many`'s `search_value`) reads
    /// exactly the terminal values ([`sts2_mcts::playout_return`]) this
    /// policy's searches back up.
    #[must_use]
    pub fn search_config(
        &self,
        sims: u32,
        seed: u64,
        character: Option<CharacterId>,
    ) -> (MctsConfig, LeafEval) {
        // Combat-search value-study ablation knobs. Absent env ⇒ deployed
        // defaults (depth 200, RichBoth terminals, budget's TexelWide leaf), so
        // production behavior is byte-identical unless a knob is explicitly set.
        let rollout_depth = std::env::var("STS2_RUNAI_ROLLOUT_DEPTH")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(COMBAT_ROLLOUT_DEPTH);
        let terminal_mode = match std::env::var("STS2_RUNAI_TERMINAL_MODE").ok().as_deref() {
            Some("binary") => sts2_mcts::TerminalEvalMode::Binary,
            Some("richloss") => sts2_mcts::TerminalEvalMode::RichLoss,
            Some("richboth") => sts2_mcts::TerminalEvalMode::RichBoth,
            _ => MctsConfig::default().terminal_eval_mode,
        };
        let leaf_eval = match std::env::var("STS2_RUNAI_LEAF").ok().as_deref() {
            // Explicit STS2_RUNAI_LEAF wins over the net env so an A/B
            // can flip the leaf off without unsetting the asset path.
            Some("texel") => LeafEval::Texel,
            Some("texelwide") => LeafEval::TexelWide,
            Some("learned") => LeafEval::Learned(combat_net_from_env().unwrap_or_else(|| {
                panic!("STS2_RUNAI_LEAF=learned requires STS2_RUNAI_COMBAT_NET")
            })),
            _ => resolve_leaf_eval(self.budget.leaf),
        };
        // Honest-search gate (bead `sts2-rs-vqckz` F0 / `sts2-rs-i8bgu` F1 /
        // `sts2-rs-09x5n` F2). Unlike the knobs above, an unrecognized value
        // PANICS instead of silently falling back to the deployed default.
        // This flag exists specifically to gate multi-hour
        // honest-vs-clairvoyant measurement campaigns (plan
        // `docs/plans/2026-07-12-clairvoyance-removal-plan.md` G1/G2/G4); a
        // typo silently collapsing away from the deployed default would
        // poison such a run without any signal that it happened. Every
        // other ablation knob here is a value-study convenience where a
        // silent fallback to "deployed behavior" is the safe choice — this
        // one is not.
        //
        // Honest-by-default per the G4 bless (2026-07-12,
        // `docs/plans/2026-07-12-clairvoyance-removal-plan.md`): unset now
        // resolves to `BeliefDraw`, the champion of the G2 3-way A/B
        // (release-2000: clairvoyant 40.35%, F0/RootScramble 13.65%,
        // F2/BeliefDraw-retuned 17.40%, +3.75pp over F0, p=4.7e-5). The F2
        // retune (c_puct/draw_widening_c below) applies automatically since
        // it's keyed on the resolved mode, not on env presence. Rollback:
        // set `STS2_RUNAI_HONEST=0` to restore the old (leaky) Clairvoyant
        // default for an ablation or emergency revert.
        let hidden_info = deployed_hidden_info();
        let mut config = mcts_config_for(hidden_info, sims, seed, rollout_depth, terminal_mode);
        // Per-run tie-band + loss-gradient + root-guard overrides
        // (`STS2_RUNAI_TIE_EPS` / `STS2_RUNAI_LOSS_GRADIENT` /
        // `STS2_RUNAI_ROOT_GUARDS`, resolved once in `new`);
        // `mcts_config_for` stays a pure function of the search-mode
        // args, so the retune constants it bakes are untouched.
        config.tie_eps = self.tie_eps;
        config.loss_gradient = self.loss_gradient;
        config.root_guards = self.root_guards;
        // `STS2_RUNAI_MCTS_OVERRIDES` (parsed once in `new`) applies LAST,
        // on top of every seam above — an explicit entry in the bundle
        // wins over `STS2_RUNAI_TIE_EPS` / `_LOSS_GRADIENT` / `_ROOT_GUARDS`
        // when both are set, so a sweep can hand a whole knob vector to
        // one env var without worrying about interaction with the older
        // single-knob vars.
        //
        // In the BeliefDraw dialect the character layer fills the slots the
        // bundle leaves unset (GH #830), so an env knob still wins, and one
        // `apply` lands it after a `belief_dialect` reset. Outside it (plain
        // `STS2_RUNAI_HONEST=0`, which never took the base retunes either)
        // the bundle applies alone. Unset/empty bundle on a character
        // without a layer ⇒ every slot `None` ⇒ a no-op.
        if self.ironclad_layer_on() {
            self.overrides
                .or(character_search_preset(character))
                .apply(&mut config);
        } else {
            self.overrides.apply(&mut config);
        }
        (config, leaf_eval)
    }

    /// Whether this policy's searches get the run agent's character layer
    /// (GH #830: [`IRONCLAD_SEARCH_KNOBS`] and the built-in move prior): in
    /// the BeliefDraw dialect, the deployed one (`STS2_RUNAI_HONEST` unset),
    /// or under `belief_dialect=1`, which gives a clairvoyant arm the same
    /// dialect so it differs in information alone. The gate measured the
    /// layer only there.
    #[must_use]
    pub fn ironclad_layer_on(&self) -> bool {
        deployed_hidden_info() == sts2_mcts::HiddenInfoMode::BeliefDraw
            || self.overrides.belief_dialect == Some(true)
    }

    /// The root's only legal action, when `mcts_pick` may answer without
    /// searching. Exact: every search pick goes through
    /// `Mcts::legalize_root_action`, which returns a member of
    /// `legal_actions` at the search root, and the honest-mode root
    /// blindfold reorders the draw pile and reseeds RNG streams without
    /// touching the hand, so that set is this one. The search runs on a
    /// clone with its own RNG, so skipping it changes only the record.
    ///
    /// `None` (search as before) when there is a real choice; when `sims`
    /// is 0, because a search that never expands the root returns
    /// `TerminalRoot` and hands the pick to the meta; and when a potion
    /// seam is active: `potion_oc` advances its own state on every search
    /// (`charge_slots_ctx` takes `&mut self`), `potion_timing` rewrites
    /// the pick from the root outcomes, and `drink_jitter` exists to
    /// generate search corpora.
    fn forced_root_action(&self, combat: &CombatState, sims: u32) -> Option<Action> {
        if sims == 0
            || self.potion_oc.is_some()
            || self.potion_timing.is_some()
            || self.drink_jitter.is_some()
        {
            return None;
        }
        match sts2_core::legal_actions(combat).as_slice() {
            [only] => Some(only.clone()),
            _ => None,
        }
    }

    /// Translate the combat pick into the index of its surfaced
    /// `RunAction` and record the root distribution, or tally the
    /// fallback (issue #164) and return `None`.
    fn surface_pick(
        &mut self,
        chosen: Action,
        outcomes: Vec<sts2_mcts::RootOutcome>,
        expand_order: sts2_mcts::ExpandOrderStats,
        actions: &[RunAction],
    ) -> Option<usize> {
        let Some(run_action) = translate_combat_action(&chosen) else {
            self.fallbacks.untranslatable += 1;
            self.log_fallback("untranslatable", Some(&chosen));
            return None;
        };
        if let Some(idx) = actions.iter().position(|a| *a == run_action) {
            // Record only on the success path: fallbacks (untranslatable /
            // not_surfaced) leave `last_combat_search` as `None` so the
            // workbench never surfaces a stale distribution.
            self.last_combat_search = Some(CombatSearchRecord {
                outcomes,
                chosen,
                expand_order,
            });
            Some(idx)
        } else {
            self.fallbacks.not_surfaced += 1;
            self.log_fallback("not_surfaced", Some(&chosen));
            None
        }
    }

    /// Opt-in stderr trace for a single fallback, gated on
    /// `STS2_RUNAI_LOG_FALLBACK` (off by default — the env-checked path
    /// keeps sweeps quiet and avoids interleaved rayon output). The
    /// always-on signal is the [`FallbackCounts`] tally; this is the
    /// "log" half of issue #164's "counter / log".
    fn log_fallback(&self, reason: &str, action: Option<&Action>) {
        if std::env::var_os("STS2_RUNAI_LOG_FALLBACK").is_some() {
            eprintln!(
                "[runai] combat MCTS→meta fallback (seed={}, decision={}, reason={reason}, action={action:?})",
                self.run_seed,
                self.decisions.saturating_sub(1),
            );
        }
    }
}

impl<M: RunPolicy> RunPolicy for MctsCombatPolicy<M> {
    fn choose(&mut self, rm: &RunManager, actions: &[RunAction]) -> Option<usize> {
        if rm.phase() == Phase::Combat {
            if let Some(combat) = rm.combat.as_ref() {
                if !combat.is_over {
                    if let Some(idx) = self.mcts_pick(rm, combat, actions) {
                        return Some(idx);
                    }
                }
            }
        }
        // `tree_reuse` (GH #830): a decision the combat search did not make
        // ends any chain — the next search builds a fresh tree.
        self.retained = None;
        // GH #252 / #447 acquisition: a full-belt potion-reward swap.
        // The engine emits ONLY `PotionSwap` actions while
        // `pending_potion_swaps` is non-empty, decline listed first.
        //
        // WHO DECIDES is `potion_oc::swap_is_rule_side`. With an OC table
        // loaded the value rule resolves it here, above the meta policy —
        // that is the #479 measurement arm and it must stay reproducible.
        // With no table we fall through to the meta policy, because GH #379
        // gave `potion_swap` a real encoder surface and the net can price it
        // like any other decision. `ppo/env.py::_advance` mirrors this split
        // on the same condition or G1 diverges.
        if actions
            .iter()
            .all(|a| matches!(a, RunAction::PotionSwap { .. }))
            && !actions.is_empty()
        {
            if let (Some(offered), Some(oc)) = (
                rm.run_state.pending_potion_swaps.first().copied(),
                self.potion_oc.as_mut(),
            ) {
                let pick = oc.resolve_swap_ctx(rm, offered);
                let want = RunAction::PotionSwap { discard_slot: pick };
                return actions.iter().position(|a| *a == want).or(Some(0));
            }
        }
        let picked = self.meta.choose(rm, actions)?;
        let final_idx = veto_lethal_event_choice(rm, actions, picked, &mut self.event_vetoes);
        self.log_event_choice(rm, actions, picked, final_idx);
        Some(final_idx)
    }

    fn prepare(&mut self, rm: &mut RunManager) {
        // `boss-oracle` ablation: top the belt up to capacity at the
        // FIRST decision of each boss fight. This is the one potion seam
        // that mutates engine state, which is why it lives here (the
        // only `&mut RunManager` hook) and why it is measurement-only.
        if self.potion_ablation == Some(potion_ablation::PotionAblation::BossOracle) {
            self.fill_boss_belt(rm);
        }
        // `slots+2`: grant the Potion Belt relic's capacity for the whole
        // run, once, without granting the relic itself (which carries no
        // other effect). Applied idempotently on the run-level cap; the
        // combat-side soft cap is refreshed from it at `enter_combat`.
        if self.potion_ablation == Some(potion_ablation::PotionAblation::SlotsPlus2)
            && !self.slots_granted
        {
            rm.run_state.gain_potion_slots(2);
            if let Some(combat) = rm.combat.as_mut() {
                if combat.player.potion_slots > 0 {
                    combat.player.potion_slots += 2;
                }
            }
            self.slots_granted = true;
        }
        // Combat never needs a mutating pre-decision hook (the MCTS
        // reads combat state directly); forward unconditionally so a
        // meta policy that DOES need it (e.g. `PpoPolicy`'s event
        // materialization) still gets it on every non-combat decision.
        self.meta.prepare(rm);
    }
}

/// Never let a meta-policy event pick be a literal suicide (GH #300).
///
/// The deployed PPO meta is event-identity-blind (event vocab was 100%
/// OOV at training), and it learned its option preferences against the
/// PRE-2026-07-18 `SlipperyBridge` port where "Hold On" was a single
/// 3-damage step. Against the now-faithful multi-page escalation
/// (3+N unblockable per page, looping until Overcome — matches
/// `SlipperyBridge.cs` `CurrentHpLoss => 3 + NumberOfHoldOns`), that
/// p≈0.99 preference becomes a loop that bleeds runs to death inside the
/// event: 10/60 engine-world losses of the 2026-07-19 diag sweep ended
/// on an event floor at 52–126 HP with no lethal fight.
///
/// The guard is minimal and policy-preserving: only when the chosen
/// option's previewed HP delta (`preview_option_effect`, exact — it
/// resolves the option on a clone) would leave the player at ≤0 HP does
/// it swap to the surviving event option with the smallest HP loss
/// (ties → lowest index). If every option is lethal the pick stands.
/// Non-event actions pass through untouched.
///
/// An alternative the engine cannot preview is **skipped, not scored 0**.
/// Scoring it 0 made an unpreviewable option look HP-neutral, i.e. strictly
/// better than every option whose real cost is known — so the guard would
/// prefer precisely the option it could say nothing about, which is the
/// opposite of a no-suicide guard. Failing closed can only cost the swap
/// (the pick then stands, as it did before this guard existed).
fn veto_lethal_event_choice(
    rm: &RunManager,
    actions: &[RunAction],
    picked: usize,
    counts: &mut EventVetoCounts,
) -> usize {
    let Some(RunAction::EventChoice { idx }) = actions.get(picked).copied() else {
        return picked;
    };
    let hp = rm.run_state.current_hp;
    let preview =
        |opt: usize| sts2_content::events::dispatch::preview_option_effect(&rm.run_state, opt);
    let Some(picked_delta) = preview(idx) else {
        // Can't prove the pick lethal, so don't touch it — but this is the
        // one bucket that would hide a real suicide, so it is counted.
        counts.unpreviewable_pick += 1;
        return picked;
    };
    if hp + picked_delta.hp_delta > 0 {
        return picked;
    }
    let candidates: Vec<(usize, Option<i32>)> = actions
        .iter()
        .enumerate()
        .filter_map(|(ai, action)| match action {
            RunAction::EventChoice { idx: alt } if *alt != idx => {
                Some((ai, preview(*alt).map(|d| d.hp_delta)))
            }
            _ => None,
        })
        .collect();
    if let Some(ai) = pick_best_survivor(hp, &candidates, counts) {
        counts.vetoed += 1;
        ai
    } else {
        counts.all_lethal += 1;
        picked
    }
}

/// Of `candidates` (`(action_index, previewed_hp_delta)`), the survivor
/// that costs the least HP — ties resolved to the lowest action index.
///
/// Split out of [`veto_lethal_event_choice`] so the two rules that decide
/// which option the guard substitutes are testable without depending on a
/// specific event's numbers:
///
/// 1. **Smallest loss wins.** Deltas are signed (loss is negative), so the
///    winner is the LARGEST delta.
/// 2. **`None` is not zero.** An option the engine cannot preview is
///    skipped. Scoring it 0 made it look HP-neutral — strictly better than
///    every option whose real cost is known — so the guard preferred
///    precisely the option it could say nothing about.
fn pick_best_survivor(
    hp: i32,
    candidates: &[(usize, Option<i32>)],
    counts: &mut EventVetoCounts,
) -> Option<usize> {
    let mut best: Option<(i32, usize)> = None;
    for &(ai, delta) in candidates {
        let Some(delta) = delta else {
            counts.unpreviewable_alt += 1;
            continue;
        };
        if hp + delta <= 0 {
            continue;
        }
        // `map_or(true, …)` not `is_none_or`: the latter is stable only
        // since 1.82 and this workspace's MSRV is 1.78.
        if best.is_none_or(|(b, _)| delta > b) {
            best = Some((delta, ai));
        }
    }
    best.map(|(_, ai)| ai)
}

/// Translate a combat [`Action`] into its [`RunAction`] twin — the exact
/// inverse of `RunManager::get_available_actions`'s `Phase::Combat`
/// mapping (`run_manager.rs` ~:861, bead `sts2-rs-rbb10`/`6dtxp`).
/// Returns `None` for variants with no run-surface equivalent
/// (UsePotion-legacy / Channel / Evoke / multi-pick selections).
fn translate_combat_action(action: &Action) -> Option<RunAction> {
    match action {
        Action::PlayCard { hand_index, target } => Some(RunAction::PlayCard {
            hand_idx: usize::from(*hand_index),
            target: target.map(u32::from),
        }),
        Action::EndTurn => Some(RunAction::EndTurn),
        Action::DrinkPotion {
            potion_slot,
            target,
        } => Some(RunAction::DrinkPotion {
            potion_slot: usize::from(*potion_slot),
            target: target.map(u32::from),
        }),
        Action::ResolveChoice { selection } => match selection.as_slice() {
            [] => Some(RunAction::CombatResolveChoice { option: None }),
            [i] => Some(RunAction::CombatResolveChoice {
                option: Some(usize::from(*i)),
            }),
            // Multi-pick (surfaced discard-N): legal_actions emits first-`k`
            // prefixes, so length == the selection (bead
            // `sts2-rs-mcts-surface-onplay-select`).
            #[allow(clippy::cast_possible_truncation)]
            sel => Some(RunAction::CombatResolveChoiceTakeFirst {
                count: sel.len() as u8,
            }),
        },
        _ => None,
    }
}

/// Source of per-decision value scores for the §3.6 audit log (M1,
/// bead `sts2-rs-40ikw.5`). Implemented by every policy the arena can
/// wrap in [`AuditingPolicy`]; the default body (`None`) keeps
/// baselines zero-cost. *Take* semantics: a returned pair is consumed —
/// scores never leak from one decision into the next record.
pub trait VScoreSource {
    /// `(per-option V scores aligned with the surfaced action list,
    /// margin between the top-2 considered options)` for the decision
    /// just made, when the policy computed them.
    fn take_v_scores(&mut self) -> Option<(Vec<f64>, f64)> {
        None
    }
}

impl VScoreSource for FirstOptionPolicy {}
impl VScoreSource for RandomPolicy {}

impl<M: VScoreSource> VScoreSource for MctsCombatPolicy<M> {
    fn take_v_scores(&mut self) -> Option<(Vec<f64>, f64)> {
        // Combat decisions don't produce meta V scores; meta decisions
        // set them on the wrapped policy, consumed here per decision.
        self.meta.take_v_scores()
    }
}

/// One decision-audit record (plan §3.6). `v_scores`/`margin` are
/// `None` for policies without a value function (M0 baselines, combat
/// MCTS decisions) and omitted from the JSONL — the schema stays
/// backward-compatible with M0 logs.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AuditRecord {
    pub seed: i32,
    /// 0-based decision index within the run (not the engine step count).
    pub step: u64,
    /// `RunState::total_floor` at decision time.
    pub floor: i32,
    /// `RunState::act_index` at decision time.
    pub act: u8,
    /// `Phase` (Debug form, e.g. `"Combat"`).
    pub phase: String,
    pub n_options: usize,
    pub chosen_idx: usize,
    /// Policy label (e.g. `"random"`, `"first+eval-cheap"`).
    pub policy: String,
    /// Per-option V scores aligned with the surfaced action list
    /// (plan §3.6; filled by `HeuristicPolicy`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub v_scores: Option<Vec<f64>>,
    /// Top-2 score margin (the tier-2 escalation gate signal).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub margin: Option<f64>,
}

/// Wrapper [`RunPolicy`] that records one [`AuditRecord`] per decision
/// (plan §3.6). Collect via [`AuditingPolicy::into_records`] after the
/// run; the arena writes them as JSONL.
#[derive(Debug)]
pub struct AuditingPolicy<P> {
    inner: P,
    seed: i32,
    policy_label: String,
    step: u64,
    records: Vec<AuditRecord>,
}

impl<P: RunPolicy> AuditingPolicy<P> {
    #[must_use]
    pub fn new(inner: P, seed: i32, policy_label: String) -> Self {
        Self {
            inner,
            seed,
            policy_label,
            step: 0,
            records: Vec::new(),
        }
    }

    #[must_use]
    pub fn into_records(self) -> Vec<AuditRecord> {
        self.records
    }

    /// Borrow the wrapped policy (escalation-count extraction before
    /// `into_records` — bead `sts2-rs-40ikw.6`).
    #[must_use]
    pub fn inner(&self) -> &P {
        &self.inner
    }

    /// Split into the wrapped policy and the collected records (the
    /// labeler needs both — bead `sts2-rs-40ikw.6`).
    #[must_use]
    pub fn into_parts(self) -> (P, Vec<AuditRecord>) {
        (self.inner, self.records)
    }
}

impl<P: RunPolicy + VScoreSource> RunPolicy for AuditingPolicy<P> {
    fn prepare(&mut self, rm: &mut RunManager) {
        self.inner.prepare(rm);
    }

    fn choose(&mut self, rm: &RunManager, actions: &[RunAction]) -> Option<usize> {
        let idx = self.inner.choose(rm, actions)?;
        let (v_scores, margin) = match self.inner.take_v_scores() {
            Some((scores, margin)) => (Some(scores), Some(margin)),
            None => (None, None),
        };
        self.records.push(AuditRecord {
            seed: self.seed,
            step: self.step,
            floor: rm.run_state.total_floor,
            act: rm.run_state.act_index,
            phase: format!("{:?}", rm.phase()),
            n_options: actions.len(),
            chosen_idx: idx,
            policy: self.policy_label.clone(),
            v_scores,
            margin,
        });
        self.step += 1;
        Some(idx)
    }
}

/// The playout config of the deployed pilot in a `character` fight, for
/// scoring playouts outside a search (`sts2_rs.values_many` /
/// `search_playout_value`, GH #830): the [`MctsCombatPolicy::search_config`]
/// of an [`CombatBudget::eval_cheap`] policy for that character, env seams
/// and the character layer included (an Ironclad state scores with
/// `removed_counts_forms` on). Its sims and search seed are placeholders a
/// playout never reads; `mcts_pick` adds only root-only fields on top
/// (`MctsCombatPolicy::pick_search_setup`). A test pins both, so a playout
/// under this config is bit-identical to the pilot's own.
#[must_use]
pub fn deployed_value_config(character: Option<CharacterId>) -> (MctsConfig, LeafEval) {
    let budget = CombatBudget::eval_cheap();
    MctsCombatPolicy::new(HeuristicPolicy::new(), budget, 0).search_config(
        budget.sims,
        0,
        character,
    )
}

// The sts2-mcts integration-test fixtures (`build_vertical_slice`), shared
// so the `build_search` golden pins the same combat the search tests use.
#[cfg(test)]
#[path = "../../../sts2-mcts/tests/common/mod.rs"]
mod mcts_fixtures;

#[cfg(test)]
mod tests {
    #[test]
    fn parse_prior_rooms_reads_tier_lists() {
        assert_eq!(
            super::parse_prior_rooms("elite, boss"),
            Ok(vec![super::BudgetTier::Elite, super::BudgetTier::Boss])
        );
        assert_eq!(
            super::parse_prior_rooms("common"),
            Ok(vec![super::BudgetTier::Common])
        );
        assert_eq!(
            super::parse_prior_rooms("elite,hallway"),
            Err("hallway".to_owned())
        );
    }

    use super::*;
    use sts2_content::run_callbacks::new_run_manager_for_character;
    use sts2_core::potions::CharacterId;

    /// A `PotionSwap`-only action set is resolved by the OC value rule,
    /// never by the meta policy: with belt [Rock(0), Block(default 8)]
    /// and Fire(4) offered, the rule discards the Rock — action index 1
    /// — while the `FirstOptionPolicy` meta would have picked index 0
    /// (decline). Env-var construction is safe: nextest runs one
    /// process per test.
    /// The twin of `potion_swap_decision_is_resolved_by_the_oc_rule`: with NO
    /// OC table the interception does not fire and the swap falls through to
    /// the META policy, because GH #379 gave `potion_swap` a real encoder
    /// surface and the net prices it like any other decision.
    ///
    /// `FirstOptionPolicy` picks index 0 — decline — which the OC rule would
    /// NOT have picked (it discards the worthless Rock, index 1), so this
    /// discriminates: it fails if the interception leaks into the OC-less arm.
    /// That arm is what the next campaign trains on (GH #545). Env-var
    /// construction is safe — nextest runs one process per test.
    #[test]
    fn potion_swap_without_an_oc_table_falls_through_to_the_meta_policy() {
        use sts2_core::potions::PotionKind;
        std::env::remove_var("STS2_RUNAI_POTION_OC");
        std::env::set_var("STS2_RUNAI_POTION_WEIGHT", "0");

        let mut rm = new_run_manager_for_character(7, CharacterId::Ironclad);
        rm.run_state.interactive_potion_swap = true;
        rm.run_state.potions.clear();
        assert!(rm
            .run_state
            .add_potion_to_belt(PotionKind::PotionShapedRock));
        assert!(rm.run_state.add_potion_to_belt(PotionKind::BlockPotion));
        rm.run_state
            .pending_potion_swaps
            .push(PotionKind::FirePotion);
        let actions = rm.get_available_actions();
        assert!(
            actions
                .iter()
                .all(|a| matches!(a, RunAction::PotionSwap { .. }))
                && actions.len() == 3,
            "swap preempt expected, got {actions:?}"
        );

        let budget = CombatBudget {
            sims: 8,
            boss_sims: 8,
            elite_sims: None,
            motif_sims: None,
            leaf: LeafKind::TexelWide,
        };
        let mut policy = MctsCombatPolicy::new(FirstOptionPolicy, budget, 7);
        let idx = policy.choose(&rm, &actions).expect("swap pick");
        assert_eq!(
            idx, 0,
            "with no OC table the META policy decides (FirstOptionPolicy -> \
             decline); the OC rule would have discarded the Rock at index 1"
        );
    }

    #[test]
    fn potion_swap_decision_is_resolved_by_the_oc_rule() {
        use sts2_core::potions::PotionKind;
        let dir = std::env::temp_dir().join("sts2-oc-swap-test");
        std::fs::create_dir_all(&dir).unwrap();
        let table = dir.join("oc.json");
        std::fs::write(
            &table,
            r#"{"version":1,"character":"ironclad","units":"hp_equiv",
                "source":"manual","default_oc":8.0,
                "entries":[{"potion":"POTION.FIRE_POTION","act":1,"oc":4.0},
                           {"potion":"POTION.POTION_SHAPED_ROCK","act":1,"oc":0.0}]}"#,
        )
        .unwrap();
        std::env::set_var("STS2_RUNAI_POTION_OC", &table);
        std::env::set_var("STS2_RUNAI_POTION_WEIGHT", "0");

        let mut rm = new_run_manager_for_character(7, CharacterId::Ironclad);
        rm.run_state.interactive_potion_swap = true;
        rm.run_state.potions.clear();
        assert!(rm
            .run_state
            .add_potion_to_belt(PotionKind::PotionShapedRock));
        assert!(rm.run_state.add_potion_to_belt(PotionKind::BlockPotion));
        rm.run_state
            .pending_potion_swaps
            .push(PotionKind::FirePotion);
        let actions = rm.get_available_actions();
        assert!(
            actions
                .iter()
                .all(|a| matches!(a, RunAction::PotionSwap { .. }))
                && actions.len() == 3,
            "swap preempt expected, got {actions:?}"
        );

        let budget = CombatBudget {
            sims: 8,
            boss_sims: 8,
            elite_sims: None,
            motif_sims: None,
            leaf: LeafKind::TexelWide,
        };
        let mut policy = MctsCombatPolicy::new(FirstOptionPolicy, budget, 7);
        let idx = policy.choose(&rm, &actions).expect("swap pick");
        assert_eq!(
            actions[idx],
            RunAction::PotionSwap {
                discard_slot: Some(0)
            },
            "rule must discard the worthless Rock for the Fire Potion"
        );
    }

    /// Critic-mode short-circuits must never touch the net: an empty
    /// belt and a final-act boss both charge zero (plan P1; twin of the
    /// table's `final_boss_charges_nothing`). This test has no ONNX
    /// gating on purpose — if it ever tries to load the champion it
    /// deserves to fail on machines without one.
    #[cfg(feature = "ppo")]
    #[test]
    #[allow(clippy::float_cmp)] // exact zeros / cache identity by construction.
    fn critic_charge_short_circuits_empty_belt_and_final_boss() {
        use sts2_core::potions::PotionKind;
        let mut oc = potion_oc::PotionOc::critic_for_tests(1.5);
        let mut rm = new_run_manager_for_character(11, CharacterId::Ironclad);
        let combat = CombatState::default(); // empty belt
        assert_eq!(
            oc.charge_slots_ctx(&rm, &combat, false, potion_oc::ClassScale::default()),
            [0.0; 8]
        );

        let mut combat = CombatState::default();
        combat.player.potions.push(PotionKind::FirePotion);
        rm.run_state.act_index = 3; // final act: boss carry is worthless
        assert_eq!(
            oc.charge_slots_ctx(&rm, &combat, true, potion_oc::ClassScale::default()),
            [0.0; 8]
        );
    }

    /// Gate sweeps set `STS2_RUNAI_PPO` per arm; critic mode must refuse
    /// to silently follow it (each arm would be priced by its own critic)
    /// and demand the explicit `STS2_RUNAI_POTION_OC_NET` pin instead.
    /// Safe under nextest's per-process isolation.
    #[cfg(feature = "ppo")]
    #[test]
    #[should_panic(expected = "STS2_RUNAI_POTION_OC_NET")]
    fn critic_mode_with_ppo_override_requires_pinned_charge_critic() {
        use sts2_core::potions::PotionKind;
        std::env::set_var("STS2_RUNAI_PPO", "/tmp/some-arm.onnx");
        std::env::remove_var("STS2_RUNAI_POTION_OC_NET");
        let mut oc = potion_oc::PotionOc::critic_for_tests(1.0);
        let rm = new_run_manager_for_character(12, CharacterId::Ironclad);
        let mut combat = CombatState::default();
        combat.player.potions.push(PotionKind::FirePotion);
        let _ = oc.charge_slots_ctx(&rm, &combat, false, potion_oc::ClassScale::default());
    }

    /// Plan P1 step 2 fixture: critic-mode charges on ONE live combat
    /// state — finite, non-negative, zero outside the belt, cached
    /// (identical on repeat), linear in the scale knob — with the
    /// table-mode charge on the same state printed alongside for shape
    /// comparison. Skips when the committed champion ONNX is absent.
    #[cfg(feature = "ppo")]
    #[test]
    #[allow(clippy::float_cmp)] // exact zeros / cache identity by construction.
    fn critic_charge_fixture_on_live_state_matches_table_shape() {
        use sts2_core::potions::PotionKind;
        if crate::ppo::net::onnx_model_path().is_none() {
            eprintln!("skip: champion ONNX absent");
            return;
        }
        let seed = 3;
        let mut rm = new_run_manager_for_character(seed, CharacterId::Ironclad);
        rm.run_state.potions.clear();
        assert!(rm.run_state.add_potion_to_belt(PotionKind::FirePotion));
        assert!(rm.run_state.add_potion_to_belt(PotionKind::BlockPotion));
        let mut warmup = FirstOptionPolicy;
        let mut guard = 0;
        while rm.phase() != Phase::Combat {
            let actions = rm.get_available_actions();
            assert!(!actions.is_empty(), "warmup stalled out of combat");
            let idx = warmup.choose(&rm, &actions).unwrap();
            let _ = rm.take_action(actions[idx]);
            guard += 1;
            assert!(guard < 200, "no combat reached in 200 decisions");
        }
        let combat = rm.combat.clone().expect("in combat");
        let belt_len = combat.player.potions.len();
        assert!(belt_len >= 2, "fixture belt lost its potions");

        let mut critic = potion_oc::PotionOc::critic_for_tests(1.5);
        let charges =
            critic.charge_slots_ctx(&rm, &combat, false, potion_oc::ClassScale::default());
        for (slot, q) in charges.iter().enumerate() {
            assert!(q.is_finite() && *q >= 0.0, "slot {slot}: {q}");
            if slot >= belt_len {
                assert!(*q == 0.0, "slot {slot} past belt charged {q}");
            }
        }
        let again = critic.charge_slots_ctx(&rm, &combat, false, potion_oc::ClassScale::default());
        assert_eq!(charges, again, "same (floor, belt) must hit the cache");

        let mut critic2x = potion_oc::PotionOc::critic_for_tests(3.0);
        let doubled =
            critic2x.charge_slots_ctx(&rm, &combat, false, potion_oc::ClassScale::default());
        for (a, b) in charges.iter().zip(&doubled) {
            assert!(
                (2.0 * a - b).abs() < 1e-5,
                "scale linearity end-to-end: 2×{a} vs {b}"
            );
        }

        // Table-mode shape on the same state, for the record.
        let table = potion_oc::parse_potion_oc(
            r#"{"version":1,"character":"ironclad","units":"hp_equiv",
                "source":"manual","default_oc":8.0,
                "entries":[{"potion":"POTION.FIRE_POTION","act":1,"oc":4.0}]}"#,
            "fixture",
        );
        let table_charges =
            table.charge_slots(&combat.player.potions, rm.run_state.act_index, false);
        println!("critic charges: {charges:?}\ntable charges:  {table_charges:?}");
        for q in &table_charges[belt_len..] {
            assert_eq!(*q, 0.0);
        }
    }

    /// Drive a fresh Ironclad run to its first `Phase::Combat` decision
    /// with first-option picks, then check the MCTS combat policy (tiny
    /// 8-sim budget for speed) returns a valid index into the surfaced
    /// action list.
    #[test]
    fn mcts_combat_policy_returns_valid_index() {
        let seed = 3;
        let mut rm = new_run_manager_for_character(seed, CharacterId::Ironclad);
        let mut warmup = FirstOptionPolicy;
        let mut guard = 0;
        while rm.phase() != Phase::Combat {
            let actions = rm.get_available_actions();
            assert!(
                !actions.is_empty(),
                "wedged before combat at {:?}",
                rm.phase()
            );
            let idx = warmup.choose(&rm, &actions).expect("first-option picks");
            let _ = rm.take_action(actions[idx]);
            guard += 1;
            assert!(guard < 200, "never reached combat in 200 steps");
        }

        let actions = rm.get_available_actions();
        assert!(!actions.is_empty());
        let budget = CombatBudget {
            sims: 8,
            boss_sims: 8,
            elite_sims: None,
            motif_sims: None,
            leaf: LeafKind::TexelWide,
        };
        let mut policy = MctsCombatPolicy::new(FirstOptionPolicy, budget, seed);
        let idx = policy.choose(&rm, &actions).expect("combat pick");
        assert!(idx < actions.len(), "index {idx} out of range");

        // The bridge translated this Ironclad combat decision cleanly, so
        // the #164 fallback counter must NOT have fired and the decision
        // counter advanced exactly once.
        assert_eq!(policy.combat_decisions(), 1);
        assert_eq!(
            policy.fallbacks(),
            FallbackCounts::default(),
            "translatable Ironclad pick must not fall back to meta"
        );
    }

    /// `FallbackCounts::total` sums the three reason buckets.
    #[test]
    fn fallback_counts_total_sums_reasons() {
        let counts = FallbackCounts {
            no_outcome: 1,
            untranslatable: 2,
            not_surfaced: 4,
        };
        assert_eq!(counts.total(), 7);
        assert_eq!(FallbackCounts::default().total(), 0);
    }

    /// Channel/Evoke-shaped combat actions (the #164 gap) have no
    /// `RunAction` twin — `translate_combat_action` returns `None`, which
    /// is exactly the path the `untranslatable` counter tallies.
    #[test]
    fn untranslatable_actions_have_no_run_action() {
        // `EndTurn` translates; an Evoke does not (no run-surface twin).
        assert!(translate_combat_action(&Action::EndTurn).is_some());
        assert!(translate_combat_action(&Action::Evoke { times: 1 }).is_none());
    }

    /// After one `MctsCombatPolicy::choose` in `Phase::Combat`, the
    /// `last_combat_search` accessor returns a `CombatSearchRecord` whose
    /// `outcomes` are non-empty and whose `chosen` action translates to the
    /// same `RunAction` the policy returned.
    #[test]
    fn mcts_combat_policy_records_last_search() {
        let seed = 3;
        let mut rm = new_run_manager_for_character(seed, CharacterId::Ironclad);
        let mut warmup = FirstOptionPolicy;
        let mut guard = 0;
        while rm.phase() != Phase::Combat && guard < 400 {
            let actions = rm.get_available_actions();
            if actions.is_empty() {
                break;
            }
            let idx = warmup.choose(&rm, &actions).expect("warmup pick");
            let _ = rm.take_action(actions[idx]);
            guard += 1;
        }
        assert_eq!(rm.phase(), Phase::Combat, "never reached combat");

        let budget = CombatBudget {
            sims: 8,
            boss_sims: 8,
            elite_sims: None,
            motif_sims: None,
            leaf: LeafKind::TexelWide,
        };
        let mut policy = MctsCombatPolicy::new(HeuristicPolicy::default(), budget, seed);
        let actions = rm.get_available_actions();
        let idx = policy.choose(&rm, &actions).expect("combat pick");
        let rec = policy.last_combat_search().expect("combat search recorded");
        assert!(!rec.outcomes.is_empty(), "root outcomes captured");
        // The recorded `chosen` action must translate to the returned index.
        let chosen_run = translate_combat_action(&rec.chosen).expect("translatable pick");
        assert_eq!(
            actions[idx], chosen_run,
            "chosen action matches returned index"
        );
        // `chosen` must appear in `outcomes` so the surface is self-consistent
        // (no phantom "best" that isn't in the root distribution).
        assert!(
            rec.outcomes.iter().any(|o| o.action == rec.chosen),
            "chosen action must be present in the root outcomes vec"
        );
    }

    /// GH #830 rung 4: `STS2_RUNAI_MCTS_OVERRIDES=net_expand_order=1`
    /// reaches the deployed search, and the decision's record carries what
    /// the reorder did; unset, the record's counters are all zero.
    #[test]
    fn mcts_combat_policy_records_net_expand_order_stats() {
        let seed = 3;
        let mut rm = new_run_manager_for_character(seed, CharacterId::Ironclad);
        let mut warmup = FirstOptionPolicy;
        let mut guard = 0;
        while rm.phase() != Phase::Combat && guard < 400 {
            let actions = rm.get_available_actions();
            if actions.is_empty() {
                break;
            }
            let idx = warmup.choose(&rm, &actions).expect("warmup pick");
            let _ = rm.take_action(actions[idx]);
            guard += 1;
        }
        assert_eq!(rm.phase(), Phase::Combat, "never reached combat");
        let budget = CombatBudget {
            sims: 64,
            boss_sims: 64,
            elite_sims: None,
            motif_sims: None,
            leaf: LeafKind::TexelWide,
        };
        let actions = rm.get_available_actions();
        let record = |env: Option<&str>| {
            // SAFETY: nextest runs this test in its own process.
            match env {
                Some(v) => std::env::set_var("STS2_RUNAI_MCTS_OVERRIDES", v),
                None => std::env::remove_var("STS2_RUNAI_MCTS_OVERRIDES"),
            }
            let mut policy = MctsCombatPolicy::new(HeuristicPolicy::default(), budget, seed);
            std::env::remove_var("STS2_RUNAI_MCTS_OVERRIDES");
            policy.choose(&rm, &actions).expect("combat pick");
            policy
                .last_combat_search()
                .expect("combat search recorded")
                .expand_order
        };
        assert_eq!(record(None), sts2_mcts::ExpandOrderStats::default());
        let on = record(Some("net_expand_order=1"));
        assert!(on.considered > 0, "{on:?}");
        if sts2_mcts::rollout_net::builtin_net(CharacterId::Ironclad).is_some() {
            assert!(on.scored > 0, "the Ironclad net scores the nodes: {on:?}");
        }
    }

    /// The auditing wrapper records one row per decision with the
    /// surfaced option count and chosen index.
    #[test]
    fn auditing_policy_records_decisions() {
        let seed = 5;
        let mut rm = new_run_manager_for_character(seed, CharacterId::Ironclad);
        let mut policy = AuditingPolicy::new(FirstOptionPolicy, seed, "first".to_string());
        for _ in 0..5 {
            if rm.is_over() {
                break;
            }
            let actions = rm.get_available_actions();
            if actions.is_empty() {
                break;
            }
            let idx = policy.choose(&rm, &actions).expect("pick");
            let _ = rm.take_action(actions[idx]);
        }
        let records = policy.into_records();
        assert!(!records.is_empty());
        assert_eq!(records[0].step, 0);
        assert_eq!(records[0].chosen_idx, 0);
        assert_eq!(records[0].seed, seed);
        assert_eq!(records[0].policy, "first");
    }

    /// `STS2_RUNAI_HONEST=root` selects the F0 root blindfold
    /// (`HiddenInfoMode::RootScramble`) and the combat bridge still returns
    /// a valid pick — the honest-search path must not break the existing
    /// MCTS -> `RunAction` translation.
    #[test]
    fn honest_env_root_mode_produces_valid_pick() {
        // SAFETY: single-threaded test (nextest gives this test its own
        // process); no other test reads this var concurrently.
        std::env::set_var("STS2_RUNAI_HONEST", "root");
        let seed = 3;
        let mut rm = new_run_manager_for_character(seed, CharacterId::Ironclad);
        let mut warmup = FirstOptionPolicy;
        let mut guard = 0;
        while rm.phase() != Phase::Combat {
            let actions = rm.get_available_actions();
            assert!(
                !actions.is_empty(),
                "wedged before combat at {:?}",
                rm.phase()
            );
            let idx = warmup.choose(&rm, &actions).expect("first-option picks");
            let _ = rm.take_action(actions[idx]);
            guard += 1;
            assert!(guard < 200, "never reached combat in 200 steps");
        }

        let actions = rm.get_available_actions();
        let budget = CombatBudget {
            sims: 8,
            boss_sims: 8,
            elite_sims: None,
            motif_sims: None,
            leaf: LeafKind::TexelWide,
        };
        let mut policy = MctsCombatPolicy::new(FirstOptionPolicy, budget, seed);
        let idx = policy
            .choose(&rm, &actions)
            .expect("combat pick under root blindfold");
        assert!(idx < actions.len(), "index {idx} out of range");
        std::env::remove_var("STS2_RUNAI_HONEST");
    }

    /// `STS2_RUNAI_HONEST=full` selects the F1 draws-as-chance mode
    /// (`HiddenInfoMode::DrawChance`, bead `sts2-rs-i8bgu`) and the combat
    /// bridge still returns a valid pick — mirrors
    /// `honest_env_root_mode_produces_valid_pick` for the newer mode.
    #[test]
    fn honest_env_full_mode_produces_valid_pick() {
        // SAFETY: single-threaded test (nextest gives this test its own
        // process); no other test reads this var concurrently.
        std::env::set_var("STS2_RUNAI_HONEST", "full");
        let seed = 3;
        let mut rm = new_run_manager_for_character(seed, CharacterId::Ironclad);
        let mut warmup = FirstOptionPolicy;
        let mut guard = 0;
        while rm.phase() != Phase::Combat {
            let actions = rm.get_available_actions();
            assert!(
                !actions.is_empty(),
                "wedged before combat at {:?}",
                rm.phase()
            );
            let idx = warmup.choose(&rm, &actions).expect("first-option picks");
            let _ = rm.take_action(actions[idx]);
            guard += 1;
            assert!(guard < 200, "never reached combat in 200 steps");
        }

        let actions = rm.get_available_actions();
        let budget = CombatBudget {
            sims: 8,
            boss_sims: 8,
            elite_sims: None,
            motif_sims: None,
            leaf: LeafKind::TexelWide,
        };
        let mut policy = MctsCombatPolicy::new(FirstOptionPolicy, budget, seed);
        let idx = policy
            .choose(&rm, &actions)
            .expect("combat pick under draws-as-chance");
        assert!(idx < actions.len(), "index {idx} out of range");
        std::env::remove_var("STS2_RUNAI_HONEST");
    }

    /// `STS2_RUNAI_HONEST=belief` selects the F2 belief-pile deferred-draw
    /// mode (`HiddenInfoMode::BeliefDraw`, bead `sts2-rs-09x5n`) and the
    /// combat bridge still returns a valid pick — mirrors
    /// `honest_env_full_mode_produces_valid_pick` for the newer mode.
    #[test]
    fn honest_env_belief_mode_produces_valid_pick() {
        // SAFETY: single-threaded test (nextest gives this test its own
        // process); no other test reads this var concurrently.
        std::env::set_var("STS2_RUNAI_HONEST", "belief");
        let seed = 3;
        let mut rm = new_run_manager_for_character(seed, CharacterId::Ironclad);
        let mut warmup = FirstOptionPolicy;
        let mut guard = 0;
        while rm.phase() != Phase::Combat {
            let actions = rm.get_available_actions();
            assert!(
                !actions.is_empty(),
                "wedged before combat at {:?}",
                rm.phase()
            );
            let idx = warmup.choose(&rm, &actions).expect("first-option picks");
            let _ = rm.take_action(actions[idx]);
            guard += 1;
            assert!(guard < 200, "never reached combat in 200 steps");
        }

        let actions = rm.get_available_actions();
        let budget = CombatBudget {
            sims: 8,
            boss_sims: 8,
            elite_sims: None,
            motif_sims: None,
            leaf: LeafKind::TexelWide,
        };
        let mut policy = MctsCombatPolicy::new(FirstOptionPolicy, budget, seed);
        let idx = policy
            .choose(&rm, &actions)
            .expect("combat pick under belief-pile deferred draws");
        assert!(idx < actions.len(), "index {idx} out of range");
        std::env::remove_var("STS2_RUNAI_HONEST");
    }

    /// Any `STS2_RUNAI_HONEST` value other than unset/`"0"`/`"root"`/`"full"`
    /// must panic rather than silently falling back to `Clairvoyant` — a
    /// typo in a measurement campaign's env must fail loudly, not poison
    /// hours of results (see the rationale comment at the `mcts_pick` call
    /// site).
    #[test]
    #[should_panic(expected = "STS2_RUNAI_HONEST")]
    fn honest_env_invalid_value_panics() {
        // SAFETY: single-threaded test (nextest gives this test its own
        // process); no other test reads this var concurrently.
        std::env::set_var("STS2_RUNAI_HONEST", "bogus");
        let seed = 3;
        let mut rm = new_run_manager_for_character(seed, CharacterId::Ironclad);
        let mut warmup = FirstOptionPolicy;
        let mut guard = 0;
        while rm.phase() != Phase::Combat {
            let actions = rm.get_available_actions();
            assert!(
                !actions.is_empty(),
                "wedged before combat at {:?}",
                rm.phase()
            );
            let idx = warmup.choose(&rm, &actions).expect("first-option picks");
            let _ = rm.take_action(actions[idx]);
            guard += 1;
            assert!(guard < 200, "never reached combat in 200 steps");
        }

        let actions = rm.get_available_actions();
        let budget = CombatBudget {
            sims: 8,
            boss_sims: 8,
            elite_sims: None,
            motif_sims: None,
            leaf: LeafKind::TexelWide,
        };
        let mut policy = MctsCombatPolicy::new(FirstOptionPolicy, budget, seed);
        // Panics inside `choose` -> `mcts_pick` when it reads the
        // unrecognized "bogus" value.
        let _ = policy.choose(&rm, &actions);
    }

    /// `set_budget` retargets sims WITHOUT disturbing the rest of the
    /// budget. The failure this pins: a caller that rebuilds a
    /// `CombatBudget` from scratch instead of mutating the one
    /// `budget()` hands back silently drops the `leaf` choice back to
    /// whatever `Default`/`eval_cheap` says, changing the leaf evaluator
    /// as a side effect of changing the sim count.
    #[test]
    fn set_budget_retargets_sims_and_keeps_the_leaf() {
        let start = CombatBudget {
            sims: 200,
            boss_sims: 800,
            elite_sims: None,
            motif_sims: None,
            leaf: LeafKind::Texel,
        };
        let mut policy = MctsCombatPolicy::new(FirstOptionPolicy, start, 7);
        assert_eq!(policy.budget().sims, 200);

        let mut next = policy.budget();
        next.sims = 1000;
        next.boss_sims = 4000;
        policy.set_budget(next);

        let got = policy.budget();
        assert_eq!(got.sims, 1000);
        assert_eq!(got.boss_sims, 4000);
        assert_eq!(
            got.leaf,
            LeafKind::Texel,
            "raising the sim budget must not swap the leaf evaluator"
        );
    }

    // -- `resolve_hidden_info` pure-function unit tests --------------------
    //
    // These exercise the env->mode mapping directly (no `std::env::set_var`,
    // no process-env mutation, no combat search) so all six cases run as
    // plain, order-independent unit tests. `honest_env_*_produces_valid_pick`
    // above stay as the integration-level checks that the resolved mode
    // actually flows through a live combat search.

    /// Unset env is the G4-blessed honest default: `BeliefDraw`.
    #[test]
    fn resolve_hidden_info_unset_is_belief_draw() {
        assert_eq!(
            resolve_hidden_info(None),
            sts2_mcts::HiddenInfoMode::BeliefDraw
        );
    }

    /// The explicit `"belief"` alias resolves identically to unset.
    #[test]
    fn resolve_hidden_info_belief_alias_matches_unset() {
        assert_eq!(
            resolve_hidden_info(Some("belief")),
            sts2_mcts::HiddenInfoMode::BeliefDraw
        );
        assert_eq!(
            resolve_hidden_info(Some("belief")),
            resolve_hidden_info(None)
        );
    }

    /// `"0"` is the rollback/ablation escape hatch back to the pre-G4
    /// leaky default.
    #[test]
    fn resolve_hidden_info_zero_is_clairvoyant() {
        assert_eq!(
            resolve_hidden_info(Some("0")),
            sts2_mcts::HiddenInfoMode::Clairvoyant
        );
    }

    /// `"root"` selects the F0 root-blindfold honest mode.
    #[test]
    fn resolve_hidden_info_root_is_root_scramble() {
        assert_eq!(
            resolve_hidden_info(Some("root")),
            sts2_mcts::HiddenInfoMode::RootScramble
        );
    }

    /// `"full"` selects the F1 draws-as-chance honest mode.
    #[test]
    fn resolve_hidden_info_full_is_draw_chance() {
        assert_eq!(
            resolve_hidden_info(Some("full")),
            sts2_mcts::HiddenInfoMode::DrawChance
        );
    }

    /// Anything else panics rather than silently falling back — a typo
    /// must not poison a multi-hour measurement campaign.
    #[test]
    #[should_panic(expected = "STS2_RUNAI_HONEST")]
    fn resolve_hidden_info_garbage_panics() {
        let _ = resolve_hidden_info(Some("garbage"));
    }

    // -- `resolve_tie_eps` pure-function unit tests ------------------------

    /// Unset env keeps the engine default (0.02, the A8-audit tie band).
    #[test]
    fn resolve_tie_eps_unset_is_engine_default() {
        assert!((resolve_tie_eps(None) - MctsConfig::default().tie_eps).abs() < 1e-6);
        assert!((resolve_tie_eps(None) - 0.02).abs() < 1e-6);
    }

    /// `"0"` is the rollback seam: strict-equality tie-breaking.
    #[test]
    #[allow(clippy::float_cmp)] // exact literal pins (0.0 assigned directly, no arithmetic)
    fn resolve_tie_eps_zero_is_strict_rollback() {
        assert_eq!(resolve_tie_eps(Some("0")), 0.0);
        assert_eq!(resolve_tie_eps(Some("0.0")), 0.0);
    }

    /// Explicit values parse; negatives clamp to strict.
    #[test]
    #[allow(clippy::float_cmp)] // exact literal pins (0.0 from a clamp of a literal)
    fn resolve_tie_eps_parses_and_clamps() {
        assert!((resolve_tie_eps(Some("0.05")) - 0.05).abs() < 1e-6);
        assert_eq!(resolve_tie_eps(Some("-1.0")), 0.0);
    }

    /// A typo must fail loudly, not silently run the default arm of an
    /// A/B — same rationale as `STS2_RUNAI_HONEST`.
    #[test]
    #[should_panic(expected = "STS2_RUNAI_TIE_EPS")]
    fn resolve_tie_eps_garbage_panics() {
        let _ = resolve_tie_eps(Some("garbage"));
    }

    /// The env is read ONCE at policy construction — a mid-run env flip
    /// must not change the band the policy searches with.
    #[test]
    #[allow(clippy::float_cmp)] // exact literal pin (0.0 parsed from "0", no arithmetic)
    fn tie_eps_env_read_once_at_construction() {
        // SAFETY: single-threaded test (nextest gives this test its own
        // process); no other test reads this var concurrently.
        std::env::set_var("STS2_RUNAI_TIE_EPS", "0");
        let budget = CombatBudget {
            sims: 8,
            boss_sims: 8,
            elite_sims: None,
            motif_sims: None,
            leaf: LeafKind::TexelWide,
        };
        let policy = MctsCombatPolicy::new(FirstOptionPolicy, budget, 1);
        std::env::set_var("STS2_RUNAI_TIE_EPS", "0.5");
        assert_eq!(
            policy.tie_eps, 0.0,
            "tie_eps must be pinned at construction, not re-read per pick"
        );
        std::env::remove_var("STS2_RUNAI_TIE_EPS");
    }

    // -- `resolve_loss_gradient` pure-function unit tests ------------------

    /// Unset keeps the engine default (graded loss backups — FINDINGS
    /// §3); `"1"` is the explicit spelling of the same arm.
    #[test]
    fn resolve_loss_gradient_unset_is_engine_default() {
        assert!(resolve_loss_gradient(None));
        assert!(resolve_loss_gradient(Some("1")));
        assert_eq!(
            resolve_loss_gradient(None),
            MctsConfig::default().loss_gradient
        );
    }

    /// `"0"` is the rollback seam to the pre-audit constant −1.45 loss
    /// floor.
    #[test]
    fn resolve_loss_gradient_zero_is_rollback() {
        assert!(!resolve_loss_gradient(Some("0")));
        assert!(!resolve_loss_gradient(Some(" 0 ")));
    }

    /// A typo must fail loudly, not silently run the default arm of an
    /// A/B — same rationale as `STS2_RUNAI_HONEST` / `STS2_RUNAI_TIE_EPS`.
    #[test]
    #[should_panic(expected = "STS2_RUNAI_LOSS_GRADIENT")]
    fn resolve_loss_gradient_garbage_panics() {
        let _ = resolve_loss_gradient(Some("off"));
    }

    /// The env is read ONCE at policy construction — a mid-run env flip
    /// must not change the loss backup the policy searches with.
    #[test]
    fn loss_gradient_env_read_once_at_construction() {
        // SAFETY: single-threaded test (nextest gives this test its own
        // process); no other test reads this var concurrently.
        std::env::set_var("STS2_RUNAI_LOSS_GRADIENT", "0");
        let budget = CombatBudget {
            sims: 8,
            boss_sims: 8,
            elite_sims: None,
            motif_sims: None,
            leaf: LeafKind::TexelWide,
        };
        let policy = MctsCombatPolicy::new(FirstOptionPolicy, budget, 1);
        std::env::set_var("STS2_RUNAI_LOSS_GRADIENT", "1");
        assert!(
            !policy.loss_gradient,
            "loss_gradient must be pinned at construction, not re-read per pick"
        );
        std::env::remove_var("STS2_RUNAI_LOSS_GRADIENT");
    }

    // -- `resolve_root_guards` pure-function unit tests --------------------

    /// Unset keeps the engine default (in-band 1-ply root guards —
    /// FINDINGS §5); `"1"` is the explicit spelling of the same arm.
    #[test]
    fn resolve_root_guards_unset_is_engine_default() {
        assert!(resolve_root_guards(None));
        assert!(resolve_root_guards(Some("1")));
        assert_eq!(resolve_root_guards(None), MctsConfig::default().root_guards);
    }

    /// `"0"` is the rollback seam back to the plain tie-band pick (the
    /// proven-win guard is ungated in the engine and unaffected).
    #[test]
    fn resolve_root_guards_zero_is_rollback() {
        assert!(!resolve_root_guards(Some("0")));
        assert!(!resolve_root_guards(Some(" 0 ")));
    }

    /// A typo must fail loudly, not silently run the default arm of an
    /// A/B — same rationale as `STS2_RUNAI_HONEST` and its siblings.
    #[test]
    #[should_panic(expected = "STS2_RUNAI_ROOT_GUARDS")]
    fn resolve_root_guards_garbage_panics() {
        let _ = resolve_root_guards(Some("off"));
    }

    // -- `STS2_RUNAI_BOSS_CARRY` (GH #771) -----------------------------------

    #[test]
    fn resolve_boss_carry_defaults_to_last_boss() {
        assert_eq!(resolve_boss_carry(None), BossCarry::LastBoss);
        assert_eq!(resolve_boss_carry(Some("")), BossCarry::LastBoss);
        assert_eq!(resolve_boss_carry(Some("0")), BossCarry::Off);
        assert_eq!(resolve_boss_carry(Some(" 1 ")), BossCarry::All);
    }

    #[test]
    #[should_panic(expected = "STS2_RUNAI_BOSS_CARRY")]
    fn resolve_boss_carry_garbage_panics() {
        let _ = resolve_boss_carry(Some("on"));
    }

    /// Last boss: win probability alone. A10 boss 1: deployed. Act-1/2
    /// bosses: HP at the Ancient's heal rate, potions in full. Hallways and
    /// elites: deployed.
    #[test]
    fn boss_win_carry_by_fight() {
        assert_eq!(boss_win_carry(false, true, false, 9), None, "not a boss");
        assert_eq!(boss_win_carry(true, true, false, 9), Some((0.0, 0.0)));
        assert_eq!(
            boss_win_carry(true, true, false, 10),
            Some((0.0, 0.0)),
            "A10 boss 2"
        );
        assert_eq!(boss_win_carry(true, true, true, 10), None, "A10 boss 1");
        assert_eq!(boss_win_carry(true, false, false, 0), Some((0.0, 1.0)));
        assert_eq!(boss_win_carry(true, false, false, 1), Some((0.0, 1.0)));
        assert_eq!(boss_win_carry(true, false, false, 2), Some((0.2, 1.0)));
        assert_eq!(boss_win_carry(true, false, false, 10), Some((0.2, 1.0)));
    }

    #[test]
    fn parse_mcts_overrides_win_carry() {
        let o = parse_mcts_overrides(Some("win_hp_carry=0,win_potion_carry=0.5"));
        assert_eq!(o.win_hp_carry, Some(0.0));
        assert_eq!(o.win_potion_carry, Some(0.5));
        let mut c = MctsConfig::default();
        o.apply(&mut c);
        assert_eq!(c.win_hp_carry, Some(0.0));
        assert_eq!(c.win_potion_carry, Some(0.5));
    }

    #[test]
    #[should_panic(expected = "win_hp_carry")]
    fn parse_mcts_overrides_win_carry_out_of_range_panics() {
        let _ = parse_mcts_overrides(Some("win_hp_carry=1.5"));
    }

    /// The env is read ONCE at policy construction — a mid-run env flip
    /// must not change the guards the policy picks with.
    #[test]
    fn root_guards_env_read_once_at_construction() {
        // SAFETY: single-threaded test (nextest gives this test its own
        // process); no other test reads this var concurrently.
        std::env::set_var("STS2_RUNAI_ROOT_GUARDS", "0");
        let budget = CombatBudget {
            sims: 8,
            boss_sims: 8,
            elite_sims: None,
            motif_sims: None,
            leaf: LeafKind::TexelWide,
        };
        let policy = MctsCombatPolicy::new(FirstOptionPolicy, budget, 1);
        std::env::set_var("STS2_RUNAI_ROOT_GUARDS", "1");
        assert!(
            !policy.root_guards,
            "root_guards must be pinned at construction, not re-read per pick"
        );
        std::env::remove_var("STS2_RUNAI_ROOT_GUARDS");
    }

    // -- `parse_mcts_overrides` / `MctsKnobOverrides` unit tests -----------

    /// Unset or empty (after trimming) ⇒ every slot `None` ⇒ `apply` is a
    /// no-op ⇒ byte-identical to today.
    #[test]
    fn parse_mcts_overrides_unset_or_empty_is_default() {
        assert_eq!(parse_mcts_overrides(None), MctsKnobOverrides::default());
        assert_eq!(parse_mcts_overrides(Some("")), MctsKnobOverrides::default());
        assert_eq!(
            parse_mcts_overrides(Some("   ")),
            MctsKnobOverrides::default()
        );
    }

    /// Every known knob parses into its slot; unmentioned slots stay
    /// `None`.
    #[test]
    #[allow(clippy::float_cmp)] // exact literal round-trip, no arithmetic
    fn parse_mcts_overrides_round_trip() {
        let parsed = parse_mcts_overrides(Some(
            "endturn_widening_c=5.787,endturn_widening_alpha=0.436,c_puct=2.5,\
             loss_gradient=0,root_guards=1",
        ));
        assert_eq!(parsed.endturn_widening_c, Some(5.787));
        assert_eq!(parsed.endturn_widening_alpha, Some(0.436));
        assert_eq!(parsed.c_puct, Some(2.5));
        assert_eq!(parsed.loss_gradient, Some(false));
        assert_eq!(parsed.root_guards, Some(true));
        // Unmentioned slots stay `None`.
        assert_eq!(parsed.tie_eps, None);
        assert_eq!(parsed.chance_widening_c, None);
        assert_eq!(parsed.loss_w_removed, None);
    }

    /// Whitespace around tokens/keys/values is tolerated.
    #[test]
    #[allow(clippy::float_cmp)] // exact literal round-trip, no arithmetic
    fn parse_mcts_overrides_tolerates_whitespace() {
        let parsed = parse_mcts_overrides(Some(" tie_eps = 0.05 , loss_span=0.9 "));
        assert_eq!(parsed.tie_eps, Some(0.05));
        assert_eq!(parsed.loss_span, Some(0.9));
    }

    /// An unknown knob name must fail loudly, not silently no-op — a typo
    /// here poisons a promotion-deciding sweep without any signal.
    #[test]
    #[should_panic(expected = "STS2_RUNAI_MCTS_OVERRIDES: unknown knob \"c_puc\"")]
    fn parse_mcts_overrides_unknown_knob_panics() {
        let _ = parse_mcts_overrides(Some("c_puc=2.5"));
    }

    /// The panic message lists the valid names so the fix is obvious from
    /// the failure alone.
    #[test]
    fn parse_mcts_overrides_unknown_knob_panic_lists_valid_names() {
        let result = std::panic::catch_unwind(|| parse_mcts_overrides(Some("bogus_knob=1")));
        let err = result.expect_err("must panic on unknown knob");
        let msg = err
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| err.downcast_ref::<&str>().map(|s| (*s).to_string()))
            .expect("panic payload is a string");
        for name in [
            "chance_widening_c",
            "endturn_widening_c",
            "tie_eps",
            "c_puct",
            "loss_w_removed",
            "loss_gradient",
            "root_guards",
            "leaf_playouts",
            "net_expand_order",
            "fpu_reduction",
            "root_sequential_halving",
            "effective_incoming",
            "removed_counts_forms",
        ] {
            assert!(
                msg.contains(name),
                "panic message must list {name:?} as a valid knob, got: {msg}"
            );
        }
    }

    /// GH #445: `endturn_fpu` was an inert knob — `MctsConfig`'s
    /// `endturn_fpu_penalty` was declared, documented and defaulted but
    /// never read by the chance-mode selector, so every arm that set it
    /// measured nothing. It was deleted rather than wired, because the
    /// general FPU mechanism (`MctsConfig::fpu_reduction`) was measured
    /// as a monotone loss (`logs/2026-08-14-exploration-arms/FINDINGS.md`
    /// §2). This pins that the name is now REJECTED: an old sweep script
    /// that still passes it must fail loudly instead of silently
    /// re-acquiring a no-op.
    #[test]
    #[should_panic(expected = "unknown knob \"endturn_fpu\"")]
    fn parse_mcts_overrides_retired_endturn_fpu_panics() {
        let _ = parse_mcts_overrides(Some("endturn_fpu=0.1"));
    }

    /// An unparseable f32 value panics.
    #[test]
    #[should_panic(expected = "STS2_RUNAI_MCTS_OVERRIDES")]
    fn parse_mcts_overrides_bad_f32_panics() {
        let _ = parse_mcts_overrides(Some("c_puct=not_a_number"));
    }

    /// An unparseable bool value panics.
    #[test]
    #[should_panic(expected = "STS2_RUNAI_MCTS_OVERRIDES")]
    fn parse_mcts_overrides_bad_bool_panics() {
        let _ = parse_mcts_overrides(Some("root_guards=maybe"));
    }

    /// `rollout_policy` selects the playout policy by name.
    #[test]
    fn parse_mcts_overrides_rollout_policy() {
        use sts2_mcts::RolloutPolicyKind;
        for (v, kind) in [
            ("uniform", RolloutPolicyKind::Uniform),
            ("heuristic", RolloutPolicyKind::Heuristic),
            ("net", RolloutPolicyKind::Net),
        ] {
            let o = parse_mcts_overrides(Some(&format!("rollout_policy={v}")));
            assert_eq!(o.rollout_policy, Some(kind));
            let mut config = MctsConfig::default();
            o.apply(&mut config);
            assert_eq!(config.rollout_policy, kind);
        }
    }

    /// `belief_dialect=true` gives a clairvoyant config the deployed
    /// honest-search dialect, so a clairvoyant arm differs in information
    /// alone (GH #830); an explicit knob in the same bundle still wins.
    #[test]
    #[allow(clippy::float_cmp)] // exact pins of copied constants
    fn parse_mcts_overrides_belief_dialect_applies_under_clairvoyant() {
        use sts2_mcts::{HiddenInfoMode, RolloutPolicyKind};
        let mut honest = MctsConfig {
            hidden_info: HiddenInfoMode::BeliefDraw,
            max_simulations: 1000,
            ..MctsConfig::default()
        };
        honest.apply_mode_dialect();

        let mut clair = MctsConfig {
            hidden_info: HiddenInfoMode::Clairvoyant,
            max_simulations: 1000,
            ..MctsConfig::default()
        };
        clair.apply_mode_dialect();
        assert_eq!(clair.rollout_policy, RolloutPolicyKind::Heuristic);
        parse_mcts_overrides(Some("belief_dialect=1")).apply(&mut clair);
        assert_eq!(clair.hidden_info, HiddenInfoMode::Clairvoyant);
        assert_eq!(clair.c_puct, honest.c_puct);
        assert_eq!(clair.endturn_widening_c, honest.endturn_widening_c);
        assert_eq!(clair.endturn_widening_alpha, honest.endturn_widening_alpha);
        assert_eq!(clair.rollout_policy, RolloutPolicyKind::Net);

        let mut mixed = MctsConfig {
            hidden_info: HiddenInfoMode::Clairvoyant,
            max_simulations: 1000,
            ..MctsConfig::default()
        };
        parse_mcts_overrides(Some("rollout_policy=heuristic,belief_dialect=true"))
            .apply(&mut mixed);
        assert_eq!(mixed.rollout_policy, RolloutPolicyKind::Heuristic);
        assert_eq!(mixed.c_puct, honest.c_puct);
    }

    /// An unknown playout policy name panics.
    #[test]
    #[should_panic(expected = "expected heuristic, uniform or net")]
    fn parse_mcts_overrides_bad_rollout_policy_panics() {
        let _ = parse_mcts_overrides(Some("rollout_policy=greedy"));
    }

    /// The pre-#808 name of the playout net fails loudly instead of
    /// selecting the fitted-table arm that now owns the name.
    #[test]
    #[should_panic(expected = "rollout_policy=learned was renamed")]
    fn parse_mcts_overrides_old_learned_name_panics() {
        let _ = parse_mcts_overrides(Some("rollout_policy=learned"));
    }

    /// A token missing `=` panics rather than silently skipping.
    #[test]
    #[should_panic(expected = "is not \"key=value\"")]
    fn parse_mcts_overrides_malformed_token_panics() {
        let _ = parse_mcts_overrides(Some("tie_eps"));
    }

    /// GH #830 (deep bets G-A, arm L4): `leaf_playouts` reaches the search
    /// config through the bundle, alone and next to other knobs, and an
    /// absent knob leaves the engine default (1) in place.
    #[test]
    fn leaf_playouts_override_reaches_the_search_config() {
        let base = || {
            mcts_config_for(
                resolve_hidden_info(None),
                1000,
                7,
                COMBAT_ROLLOUT_DEPTH,
                MctsConfig::default().terminal_eval_mode,
            )
        };
        assert_eq!(base().leaf_playouts, 1, "engine default is one playout");

        let parsed = parse_mcts_overrides(Some("leaf_playouts=4"));
        assert_eq!(parsed.leaf_playouts, Some(4));
        let mut config = base();
        parsed.apply(&mut config);
        assert_eq!(config.leaf_playouts, 4);

        let mut config = base();
        parse_mcts_overrides(Some("c_puct=2.5, leaf_playouts = 2")).apply(&mut config);
        assert_eq!(config.leaf_playouts, 2);

        let mut config = base();
        parse_mcts_overrides(Some("tie_eps=0.05")).apply(&mut config);
        assert_eq!(config.leaf_playouts, 1, "unmentioned knob stays default");
    }

    /// The five characters.
    const CHARACTERS: [CharacterId; 5] = [
        CharacterId::Ironclad,
        CharacterId::Silent,
        CharacterId::Defect,
        CharacterId::Regent,
        CharacterId::Necrobinder,
    ];

    /// A fresh `character` run (seed `seed`, A0) at its first combat, reached
    /// by first-option picks.
    fn first_combat(seed: i32, character: CharacterId) -> RunManager {
        crate::rollout::ensure_content_installed();
        let mut rm = sts2_content::run_callbacks::new_run_manager_for_character_with_ascension(
            seed, character, 0,
        )
        .with_neow_start();
        sts2_content::run_callbacks::initialize_run_setup(&mut rm);
        for _ in 0..64 {
            if rm.phase() == Phase::Combat {
                return rm;
            }
            let acts = rm.get_available_actions();
            let _ = rm.take_action(acts[0]);
        }
        panic!("{character:?}: no combat within 64 run actions");
    }

    /// A Test Subject boss fight of `rm`'s run that the player lost after
    /// killing the first of its three forms: the loss band, where
    /// `removed_counts_forms` changes the value (form 1's death scores as
    /// all enemy HP removed with it off, as a third of the forms with it on).
    fn test_subject_lost_after_its_first_form(rm: &RunManager) -> CombatState {
        use sts2_core::hooks::{CreatureRef, ValueProps};
        let mut s = sts2_content::run_bridge::enter_combat_from_roster(
            &rm.run_state,
            &[(
                sts2_core::MonsterKind::TestSubject,
                sts2_content::monster_dispatch::MonsterSpawnArgs::None,
                111,
            )],
            RoomType::Boss,
        );
        let ts = s.enemy_uids[0];
        let _ = sts2_core::damage::apply_damage(
            &mut s,
            CreatureRef::Player,
            ts,
            9_999,
            ValueProps::UNPOWERED,
            None,
        );
        assert!(!s.is_over, "Adaptable keeps the fight open");
        // The player dies (both HP fields: the mirror creature and the
        // legacy one the death check reads), and the combat ends.
        let me = s.player.creature_uid.expect("player creature");
        for c in &mut s.creatures {
            if c.id == me.0 {
                c.current_hp = 0;
            }
        }
        s.player.current_hp = 0;
        let _ = sts2_core::combat_loop::check_combat_end(&mut s);
        assert!(
            sts2_core::is_terminal(&s) && !s.player_won,
            "the player lost"
        );
        s
    }

    /// GH #830: the playouts `sts2_rs.values_many` scores
    /// ([`deployed_value_config`] of the state's character) are the pilot's
    /// own, for every character. `mcts_pick` searches with `search_config`
    /// plus three root-only fields, pinned field for field here, so a
    /// playout-relevant edit added in `pick_search_setup` fails this test.
    /// And a playout under the pick's config, at a real budget tier, search
    /// seed and boss flag, is bit-identical to one under
    /// `deployed_value_config`'s placeholder sims and seed.
    ///
    /// The opening fights cannot tell the Ironclad layer from the base
    /// dialect in a playout: `removed_counts_forms` (the one layer knob a
    /// playout reads) only matters for a multi-form enemy in the loss band.
    /// So the Ironclad states include a lost Test Subject fight, on which the
    /// Ironclad value config and the character-less one differ: the check
    /// fails if `deployed_value_config` drops the character.
    #[test]
    fn deployed_value_config_plays_the_pilots_playouts() {
        use sts2_mcts::values::{playout_value, PlayoutMode};
        let budget = CombatBudget::eval_cheap();
        let mut fights = Vec::new();
        for (k, character) in CHARACTERS.into_iter().enumerate() {
            let rm = first_combat(3 + 5 * i32::try_from(k).expect("small"), character);
            let opening = rm.combat.clone().expect("in combat");
            assert_eq!(opening.player.character, Some(character));
            let mut later = opening.clone();
            for k in 0..3_usize {
                let acts = sts2_core::legal_actions(&later);
                if acts.is_empty() || sts2_core::is_terminal(&later) {
                    break;
                }
                let _ = sts2_core::apply(&mut later, acts[(k * 7 + 1) % acts.len()].clone());
            }
            let mut states = vec![opening, later];
            if character == CharacterId::Ironclad {
                let lost = test_subject_lost_after_its_first_form(&rm);
                let (ic, ic_leaf) = deployed_value_config(Some(CharacterId::Ironclad));
                let (none, none_leaf) = deployed_value_config(None);
                assert_ne!(
                    sts2_mcts::playout_return(&lost, &ic, &ic_leaf).to_bits(),
                    sts2_mcts::playout_return(&lost, &none, &none_leaf).to_bits(),
                    "the lost Test Subject fight must score differently with forms on"
                );
                states.push(lost);
            }
            fights.push((character, rm, states));
        }
        assert_ne!(
            format!("{:?}", deployed_value_config(Some(CharacterId::Ironclad)).0),
            format!("{:?}", deployed_value_config(Some(CharacterId::Silent)).0),
            "the Ironclad layer must reach the value config"
        );
        let mut policy = MctsCombatPolicy::new(HeuristicPolicy::new(), budget, 3);
        let mut compared = 0;
        for (character, rm, states) in &fights {
            let (value_cfg, value_leaf) = deployed_value_config(Some(*character));
            for (i, s) in states.iter().enumerate() {
                for (is_boss, sims, seed) in [
                    (false, budget.sims, 0x5EED + i as u64),
                    (true, budget.boss_sims, 0xB055),
                ] {
                    let (pick, leaf, _prior) = policy.pick_search_setup(rm, s, is_boss, sims, seed);
                    let (mut base, _) = policy.search_config(sims, seed, Some(*character));
                    base.use_puct_prior = pick.use_puct_prior;
                    base.potion_charge_q = pick.potion_charge_q;
                    base.root_drink_mask = pick.root_drink_mask;
                    assert_eq!(
                        format!("{pick:?}"),
                        format!("{base:?}"),
                        "{character:?}: mcts_pick changed a config field other than the three \
                         root-only ones"
                    );
                    for ps in 0..3_u64 {
                        let a = playout_value(s, ps, PlayoutMode::Raw, &pick, &leaf).expect("raw");
                        let b = playout_value(s, ps, PlayoutMode::Raw, &value_cfg, &value_leaf)
                            .expect("raw");
                        assert_eq!(
                            (a.value.to_bits(), a.won, a.hp, a.terminal),
                            (b.value.to_bits(), b.won, b.hp, b.terminal),
                            "{character:?} state {i}, boss {is_boss}, playout seed {ps}"
                        );
                        compared += 1;
                    }
                }
            }
        }
        assert_eq!(compared, (5 * 2 + 1) * 2 * 3);
    }

    /// Zero playouts would silently score leaves with nothing; it fails.
    #[test]
    #[should_panic(expected = "leaf_playouts=\"0\" is not an integer >= 1")]
    fn parse_mcts_overrides_zero_leaf_playouts_panics() {
        let _ = parse_mcts_overrides(Some("leaf_playouts=0"));
    }

    /// A non-integer playout count fails rather than truncating.
    #[test]
    #[should_panic(expected = "is not an integer >= 1")]
    fn parse_mcts_overrides_fractional_leaf_playouts_panics() {
        let _ = parse_mcts_overrides(Some("leaf_playouts=2.5"));
    }

    /// `MctsKnobOverrides::apply` writes only the set slots; the config's
    /// unrelated fields (here `sims`/`seed`/`hidden_info`, standing in for
    /// "everything this seam doesn't touch") are untouched.
    #[test]
    #[allow(clippy::float_cmp)] // exact literal pins, no arithmetic
    fn mcts_knob_overrides_apply_writes_only_set_slots() {
        let hidden_info = resolve_hidden_info(None);
        let mut config = mcts_config_for(
            hidden_info,
            /* max_simulations */ 1000,
            /* seed */ 7,
            /* max_rollout_depth */ COMBAT_ROLLOUT_DEPTH,
            /* terminal_eval_mode */ MctsConfig::default().terminal_eval_mode,
        );
        let baseline_hidden_info = config.hidden_info;
        let baseline_sims = config.max_simulations;
        let baseline_seed = config.seed;

        let overrides = parse_mcts_overrides(Some(
            "endturn_widening_c=5.787,endturn_widening_alpha=0.436",
        ));
        overrides.apply(&mut config);

        assert_eq!(config.endturn_widening_c, 5.787);
        assert_eq!(config.endturn_widening_alpha, 0.436);
        // Fields the bundle didn't mention (including the BeliefDraw F2
        // retune's c_puct/draw_widening_c) stay exactly what
        // `mcts_config_for` produced.
        assert_eq!(
            config.c_puct, 2.5,
            "F2 retune c_puct must survive untouched"
        );
        assert_eq!(
            config.draw_widening_c, 2.0,
            "F2 retune draw_widening_c must survive untouched"
        );
        assert_eq!(config.hidden_info, baseline_hidden_info);
        assert_eq!(config.max_simulations, baseline_sims);
        assert_eq!(config.seed, baseline_seed);
    }

    /// GH #830 rung 4: `net_expand_order` and its companion `fpu_reduction`
    /// parse into their slots, and `apply` writes exactly those two fields
    /// and nothing else. Unmentioned, both stay at the base dialect's value
    /// (`mcts_config_for`: ladder-ordered expansion, `+inf` first-play
    /// urgency).
    #[test]
    #[allow(clippy::float_cmp)] // exact literal pins, no arithmetic
    fn mcts_knob_overrides_apply_net_expand_order_and_fpu_reduction() {
        let parsed = parse_mcts_overrides(Some("net_expand_order=1,fpu_reduction=0.25"));
        assert_eq!(parsed.net_expand_order, Some(true));
        assert_eq!(parsed.fpu_reduction, Some(0.25));
        assert_eq!(parsed.c_puct, None);
        assert_eq!(
            parse_mcts_overrides(Some("net_expand_order=false")).net_expand_order,
            Some(false)
        );

        let deployed = mcts_config_for(
            resolve_hidden_info(None),
            1000,
            7,
            COMBAT_ROLLOUT_DEPTH,
            MctsConfig::default().terminal_eval_mode,
        );
        assert!(!deployed.net_expand_order, "base dialect is off");
        assert_eq!(deployed.fpu_reduction, None, "base dialect is +inf FPU");

        let mut config = deployed;
        parsed.apply(&mut config);
        let mut expected = deployed;
        expected.net_expand_order = true;
        expected.fpu_reduction = Some(0.25);
        assert_eq!(format!("{config:?}"), format!("{expected:?}"));

        let mut untouched = deployed;
        parse_mcts_overrides(Some("c_puct=2.5")).apply(&mut untouched);
        assert!(!untouched.net_expand_order);
        assert_eq!(untouched.fpu_reduction, None);
    }

    /// `root_sequential_halving` parses into its slot and `apply` writes that
    /// field alone; unmentioned, the deployed UCT root allocation stays.
    #[test]
    fn mcts_knob_overrides_apply_root_sequential_halving() {
        let parsed = parse_mcts_overrides(Some("root_sequential_halving=1"));
        assert_eq!(parsed.root_sequential_halving, Some(true));

        let deployed = mcts_config_for(
            resolve_hidden_info(None),
            1000,
            7,
            COMBAT_ROLLOUT_DEPTH,
            MctsConfig::default().terminal_eval_mode,
        );
        assert!(
            !deployed.root_sequential_halving,
            "base dialect is UCT at the root"
        );

        let mut config = deployed;
        parsed.apply(&mut config);
        let mut expected = deployed;
        expected.root_sequential_halving = true;
        assert_eq!(format!("{config:?}"), format!("{expected:?}"));

        let mut untouched = deployed;
        parse_mcts_overrides(Some("c_puct=2.5")).apply(&mut untouched);
        assert!(!untouched.root_sequential_halving);
    }

    /// `net_expand_order` is a bool knob: anything else fails loudly.
    #[test]
    #[should_panic(expected = "STS2_RUNAI_MCTS_OVERRIDES: knob net_expand_order")]
    fn parse_mcts_overrides_bad_net_expand_order_panics() {
        let _ = parse_mcts_overrides(Some("net_expand_order=yes"));
    }

    /// GH #830 boss audit: `effective_incoming` and `removed_counts_forms`
    /// parse into their slots, and `apply` writes exactly those two fields
    /// and nothing else. Unmentioned, both stay at the base dialect's value
    /// (`mcts_config_for`: base intent numbers, current-form enemy HP). The
    /// Ironclad run agent turns `removed_counts_forms` on in its own layer
    /// ([`IRONCLAD_SEARCH_KNOBS`]), which this base config does not carry.
    #[test]
    fn mcts_knob_overrides_apply_effective_incoming_and_removed_counts_forms() {
        let parsed = parse_mcts_overrides(Some("effective_incoming=1,removed_counts_forms=true"));
        assert_eq!(parsed.effective_incoming, Some(true));
        assert_eq!(parsed.removed_counts_forms, Some(true));
        assert_eq!(parsed.net_expand_order, None);
        assert_eq!(parsed.loss_w_removed, None);
        let off = parse_mcts_overrides(Some("effective_incoming=0,removed_counts_forms=false"));
        assert_eq!(off.effective_incoming, Some(false));
        assert_eq!(off.removed_counts_forms, Some(false));

        let deployed = mcts_config_for(
            resolve_hidden_info(None),
            1000,
            7,
            COMBAT_ROLLOUT_DEPTH,
            MctsConfig::default().terminal_eval_mode,
        );
        assert!(!deployed.effective_incoming, "base dialect is off");
        assert!(!deployed.removed_counts_forms, "base dialect is off");

        let mut config = deployed;
        parsed.apply(&mut config);
        let mut expected = deployed;
        expected.effective_incoming = true;
        expected.removed_counts_forms = true;
        assert_eq!(format!("{config:?}"), format!("{expected:?}"));

        // Each knob alone touches only its own field.
        let mut one = deployed;
        parse_mcts_overrides(Some("removed_counts_forms=1")).apply(&mut one);
        let mut expected_one = deployed;
        expected_one.removed_counts_forms = true;
        assert_eq!(format!("{one:?}"), format!("{expected_one:?}"));

        let mut untouched = deployed;
        parse_mcts_overrides(Some("c_puct=2.5")).apply(&mut untouched);
        assert!(!untouched.effective_incoming);
        assert!(!untouched.removed_counts_forms);
    }

    /// Both are bool knobs: anything else fails loudly.
    #[test]
    #[should_panic(expected = "STS2_RUNAI_MCTS_OVERRIDES: knob effective_incoming")]
    fn parse_mcts_overrides_bad_effective_incoming_panics() {
        let _ = parse_mcts_overrides(Some("effective_incoming=yes"));
    }

    #[test]
    #[should_panic(expected = "STS2_RUNAI_MCTS_OVERRIDES: knob removed_counts_forms")]
    fn parse_mcts_overrides_bad_removed_counts_forms_panics() {
        let _ = parse_mcts_overrides(Some("removed_counts_forms=2"));
    }

    /// An empty bundle applied to a config leaves it byte-identical —
    /// the "unset env ⇒ no-op" contract exercised at the `apply` layer.
    #[test]
    #[allow(clippy::float_cmp)] // exact literal pins, no arithmetic
    fn mcts_knob_overrides_apply_empty_is_noop() {
        let hidden_info = resolve_hidden_info(None);
        let mut config = mcts_config_for(
            hidden_info,
            1000,
            7,
            COMBAT_ROLLOUT_DEPTH,
            MctsConfig::default().terminal_eval_mode,
        );
        let before_c_puct = config.c_puct;
        let before_tie_eps = config.tie_eps;
        let before_loss_gradient = config.loss_gradient;
        let before_root_guards = config.root_guards;

        MctsKnobOverrides::default().apply(&mut config);

        assert_eq!(config.c_puct, before_c_puct);
        assert_eq!(config.tie_eps, before_tie_eps);
        assert_eq!(config.loss_gradient, before_loss_gradient);
        assert_eq!(config.root_guards, before_root_guards);
    }

    /// The env is read ONCE at policy construction — a mid-run env flip
    /// must not change the overrides the policy searches with. Same
    /// pattern as `tie_eps_env_read_once_at_construction` /
    /// `root_guards_env_read_once_at_construction`.
    #[test]
    #[allow(clippy::float_cmp)] // exact literal pin, no arithmetic
    fn mcts_overrides_env_read_once_at_construction() {
        // SAFETY: single-threaded test (nextest gives this test its own
        // process); no other test reads this var concurrently.
        std::env::set_var("STS2_RUNAI_MCTS_OVERRIDES", "tie_eps=0.5");
        let budget = CombatBudget {
            sims: 8,
            boss_sims: 8,
            elite_sims: None,
            motif_sims: None,
            leaf: LeafKind::TexelWide,
        };
        let policy = MctsCombatPolicy::new(FirstOptionPolicy, budget, 1);
        std::env::set_var("STS2_RUNAI_MCTS_OVERRIDES", "tie_eps=0.9");
        assert_eq!(
            policy.overrides.tie_eps,
            Some(0.5),
            "overrides must be pinned at construction, not re-read per pick"
        );
        std::env::remove_var("STS2_RUNAI_MCTS_OVERRIDES");
    }

    /// Unset `STS2_RUNAI_MCTS_OVERRIDES` ⇒ the resolved config is
    /// byte-identical to the pre-existing seams alone — the top-level
    /// "unset env = production behavior unchanged" contract, exercised
    /// through the full `new` + config-assembly path rather than just
    /// `parse_mcts_overrides` in isolation.
    #[test]
    #[allow(clippy::float_cmp)] // exact literal pins, no arithmetic
    fn mcts_overrides_unset_leaves_config_untouched_by_new() {
        std::env::remove_var("STS2_RUNAI_MCTS_OVERRIDES");
        let budget = CombatBudget {
            sims: 8,
            boss_sims: 8,
            elite_sims: None,
            motif_sims: None,
            leaf: LeafKind::TexelWide,
        };
        let policy = MctsCombatPolicy::new(FirstOptionPolicy, budget, 1);
        assert_eq!(policy.overrides, MctsKnobOverrides::default());

        let hidden_info = resolve_hidden_info(None);
        let mut config = mcts_config_for(
            hidden_info,
            8,
            42,
            COMBAT_ROLLOUT_DEPTH,
            MctsConfig::default().terminal_eval_mode,
        );
        config.tie_eps = policy.tie_eps;
        config.loss_gradient = policy.loss_gradient;
        config.root_guards = policy.root_guards;
        let before_c_puct = config.c_puct;
        let before_endturn_widening_c = config.endturn_widening_c;
        policy.overrides.apply(&mut config);
        assert_eq!(config.c_puct, before_c_puct);
        assert_eq!(config.endturn_widening_c, before_endturn_widening_c);
    }

    // -- deployed-path pick unification (2026-08-09 A8 audit) --------------

    /// GH #403's `ftl_draw_window` busy-hand fixture, rebuilt inline for
    /// the DEPLOYED path: 5 playable cards + FTL appended last (so a
    /// naive left-to-right pick fails), window open (0 plays this turn,
    /// 3 Strikes left to draw), HP set so the turn is not lethal either
    /// way — the position is only about the draw rider.
    fn build_ftl_busy_hand_state(seed: i32) -> CombatState {
        use sts2_core::cards::{Card, CardId, CardType, TargetType};
        use sts2_core::{
            CardUid, CombatPhase, Creature, CreatureUid, IntentHistory, MonsterId, MonsterKind,
            PlayerCombatState,
        };
        sts2_core::card_effects::ensure_installed();
        sts2_content::cards::dispatch::install_default_registry();

        let hand_spec: [(CardId, i32, i32, i8, TargetType); 6] = [
            (CardId::Slice, 6, 0, 0, TargetType::Enemy),
            (CardId::StrikeIronclad, 6, 0, 1, TargetType::Enemy),
            (CardId::DefendIronclad, 0, 5, 1, TargetType::Self_),
            (CardId::Bash, 8, 0, 2, TargetType::Enemy),
            (CardId::Slice, 6, 0, 0, TargetType::Enemy),
            (CardId::Ftl, 5, 0, 0, TargetType::Enemy),
        ];
        let mut cards: Vec<Card> = Vec::new();
        for (idx, &(card_id, base_damage, base_block, cost, target_type)) in
            hand_spec.iter().enumerate()
        {
            let uid = CardUid::new(u32::try_from(idx + 1).expect("fits")).expect("non-zero");
            cards.push(Card {
                uid,
                card_id,
                cost,
                card_type: if target_type == TargetType::Enemy {
                    CardType::Attack
                } else {
                    CardType::Skill
                },
                target_type,
                base_damage,
                base_block,
                ..Card::pod_defaults()
            });
        }
        // 3 Strikes left in the draw pile so the FTL rider has something
        // to draw (the window predicate requires it).
        for idx in 7u32..=9 {
            let uid = CardUid::new(idx).expect("non-zero");
            cards.push(Card {
                uid,
                card_id: CardId::StrikeDefect,
                cost: 1,
                card_type: CardType::Attack,
                target_type: TargetType::Enemy,
                base_damage: 6,
                base_block: 0,
                ..Card::pod_defaults()
            });
        }

        let cultist_uid = CreatureUid::new(1).expect("non-zero");
        let cultist = Creature {
            id: cultist_uid.0,
            current_hp: 45,
            max_hp: 45,
            block: 0,
            powers: smallvec::SmallVec::default(),
            species: Some(MonsterKind::Nibbit),
            monster_id: MonsterId("Nibbit"),
            intent: Some(sts2_core::Intent::Attack { damage: 5 }),
            intent_history: IntentHistory::default(),
            ai: None,
        };

        CombatState {
            phase: CombatPhase::PlayerPlayPhase,
            round: 1,
            rng: sts2_core::dotnet_rng::Rng::new(seed),
            run_rng: sts2_core::run_rng_set::RunRngSet::new(seed),
            creatures: vec![cultist],
            cards: cards.into(),
            player: PlayerCombatState {
                hand: (1u32..=6)
                    .map(|i| CardUid::new(i).expect("non-zero"))
                    .collect(),
                draw: (7u32..=9)
                    .map(|i| CardUid::new(i).expect("non-zero"))
                    .collect(),
                energy: 3,
                max_energy: 3,
                current_hp: 70,
                max_hp: 70,
                ..PlayerCombatState::default()
            },
            enemy_uids: std::iter::once(cultist_uid).collect(),
            ..CombatState::default()
        }
    }

    /// THE A8-audit regression (FINDINGS §1): the deployed `mcts_pick`
    /// must return exactly what the engine's unified
    /// `Mcts::pick_root_action` returns — including the GH #403
    /// draw-window bonus, which was dead code on this path while
    /// `mcts_pick` discarded `search_n`'s return and re-derived its own
    /// argmax (measured 54/75 = 72% in-window on the audit corpus, the
    /// commit's own pre-fix rate). Byte-identical config on both sides,
    /// so agreement is exact, and the agreed pick must be FTL (hand
    /// slot 5) inside its open window.
    /// A valid root-filter scorer whose score is the plan's enemy damage
    /// (feature 3), written to a fresh temp file; returns its path.
    fn damage_scorer_file(tag: &str) -> std::path::PathBuf {
        let dim = sts2_mcts::turn_plan_features::PLAN_DIM;
        let eye: Vec<Vec<f32>> = (0..dim)
            .map(|i| (0..dim).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
            .collect();
        let mut l2 = vec![vec![0.0f32; dim]; 2];
        l2[0][3] = 1.0;
        let mut s1 = vec![0.0f32; 8];
        s1[6] = 1.0;
        let v = serde_json::json!({
            "format": sts2_mcts::turn_plan_scorer::FORMAT,
            "ctx_mean": vec![0.0; 6], "ctx_std": vec![1.0; 6],
            "plan_mean": vec![0.0; dim], "plan_std": vec![1.0; dim],
            "plan_l1": {"w": eye, "b": vec![0.0; dim]},
            "plan_l2": {"w": l2, "b": [0.0, 0.0]},
            "card_vocab": {},
            "card_emb": [[0.0, 0.0]],
            "score_l1": {"w": [s1], "b": [0.0]},
            "score_l2": {"w": [[1.0]], "b": [0.0]},
        });
        let path = std::env::temp_dir().join(format!(
            "sts2-root-filter-{tag}-{}.json",
            std::process::id()
        ));
        std::fs::write(&path, v.to_string()).expect("temp file");
        path
    }

    #[test]
    fn resolve_root_filter_unset_is_off_and_a_valid_value_loads() {
        assert!(resolve_root_filter(None).is_none());
        assert!(resolve_root_filter(Some("")).is_none());
        let path = damage_scorer_file("resolve");
        let (_, k) = resolve_root_filter(Some(&format!("{}:6", path.display()))).expect("on");
        assert_eq!(k, 6);
    }

    #[test]
    #[should_panic(expected = "STS2_RUNAI_TURNPLAN_ROOT_FILTER")]
    fn resolve_root_filter_refuses_a_missing_k() {
        let _ = resolve_root_filter(Some("weights.json"));
    }

    #[test]
    #[should_panic(expected = "K=\"0\"")]
    fn resolve_root_filter_refuses_k_zero() {
        let path = damage_scorer_file("k0");
        let _ = resolve_root_filter(Some(&format!("{}:0", path.display())));
    }

    fn ftl_pick(root_filter: Option<usize>) -> (Action, Option<RootFilterRecord>, CombatState) {
        let run_seed = 3;
        let state = build_ftl_busy_hand_state(run_seed);
        let rm = new_run_manager_for_character(run_seed, CharacterId::Ironclad);
        let budget = CombatBudget {
            sims: 400,
            boss_sims: 400,
            elite_sims: None,
            motif_sims: None,
            leaf: LeafKind::TexelWide,
        };
        let mut policy = MctsCombatPolicy::new(FirstOptionPolicy, budget, run_seed);
        assert!(policy.root_filter.is_none(), "the knob must default to off");
        if let Some(k) = root_filter {
            let text = std::fs::read_to_string(damage_scorer_file("pick")).unwrap();
            let scorer = sts2_mcts::turn_plan_scorer::PlanScorer::from_json(&text).unwrap();
            policy.root_filter = Some((Arc::new(scorer), k));
        }
        let mut actions: Vec<RunAction> = vec![RunAction::EndTurn];
        for hand_idx in 0..6 {
            for target in [None, Some(0), Some(1), Some(2)] {
                actions.push(RunAction::PlayCard { hand_idx, target });
            }
        }
        policy
            .mcts_pick(&rm, &state, &actions)
            .expect("translatable pick");
        let chosen = policy
            .last_combat_search()
            .expect("searched")
            .chosen
            .clone();
        (chosen, policy.last_root_filter().cloned(), state)
    }

    /// A K that keeps every plan restricts nothing: the pick is the
    /// unfiltered search's, and the record says the filter ran.
    #[test]
    fn root_filter_keeping_every_plan_leaves_the_pick_unchanged() {
        let (base, rec, _) = ftl_pick(None);
        assert!(rec.is_none());
        let (all, rec, _) = ftl_pick(Some(100_000));
        assert_eq!(all, base);
        let rec = rec.expect("recorded");
        assert!(rec.applied);
        assert_eq!(
            rec.allowed_n,
            legal_actions_len(&build_ftl_busy_hand_state(3))
        );
        assert_eq!(rec.chosen_via, Some("topk"));
    }

    fn legal_actions_len(state: &CombatState) -> usize {
        sts2_core::legal_actions(state).len()
    }

    /// K = 1 restricts the root to the top plan's and the anchors' first
    /// actions, and the pick is one of them.
    #[test]
    fn root_filter_k1_picks_an_allowed_action() {
        let (chosen, rec, state) = ftl_pick(Some(1));
        let rec = rec.expect("recorded");
        assert!(rec.applied, "{rec:?}");
        let text = std::fs::read_to_string(damage_scorer_file("check")).unwrap();
        let scorer = sts2_mcts::turn_plan_scorer::PlanScorer::from_json(&text).unwrap();
        let f = sts2_mcts::turn_plan_scorer::root_filter(
            &state,
            &scorer,
            1,
            sts2_mcts::turn_plan::PlanCap::default(),
        );
        assert_eq!(rec.allowed_n, f.allowed.len());
        assert!(rec.allowed_n < legal_actions_len(&state), "{rec:?}");
        assert!(
            f.allowed.iter().any(|(a, _)| *a == chosen),
            "{chosen:?} not allowed"
        );
        assert!(rec.chosen_via.is_some());
    }

    #[test]
    fn mcts_pick_matches_engine_pick_and_applies_draw_window_bonus() {
        let run_seed = 3;
        let sims = 1000;
        let state = build_ftl_busy_hand_state(run_seed);

        // Engine side: the same config construction `mcts_pick` performs
        // for decision 0 of `run_seed` with env unset.
        let seed_bits = u64::from(u32::from_ne_bytes(run_seed.to_ne_bytes()));
        let search_seed = splitmix64(seed_bits ^ 1u64.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let mut config = mcts_config_for(
            resolve_hidden_info(None),
            sims,
            search_seed,
            COMBAT_ROLLOUT_DEPTH,
            MctsConfig::default().terminal_eval_mode,
        );
        config.tie_eps = resolve_tie_eps(None);
        let mut engine =
            Mcts::with_policy(state.clone(), config, sts2_mcts::SharedPolicy::uniform())
                .with_leaf_eval(resolve_leaf_eval(LeafKind::TexelWide));
        let engine_pick = engine.try_search_n(sims).expect("non-terminal root");
        assert!(
            matches!(engine_pick, Action::PlayCard { hand_index: 5, .. }),
            "GH #403: the unified pick must take FTL inside its window, got {engine_pick:?}"
        );

        // Deployed side: `mcts_pick` on the same combat state. The rm is
        // only consulted for the boss-room check (fresh run: not a boss).
        let rm = new_run_manager_for_character(run_seed, CharacterId::Ironclad);
        let budget = CombatBudget {
            sims,
            boss_sims: sims,
            elite_sims: None,
            motif_sims: None,
            leaf: LeafKind::TexelWide,
        };
        let mut policy = MctsCombatPolicy::new(FirstOptionPolicy, budget, run_seed);
        // Surface every translation the fixture can produce so the pick
        // can't fall back on `not_surfaced`.
        let mut actions: Vec<RunAction> = vec![RunAction::EndTurn];
        for hand_idx in 0..6 {
            for target in [None, Some(0), Some(1), Some(2)] {
                actions.push(RunAction::PlayCard { hand_idx, target });
            }
        }
        let idx = policy
            .mcts_pick(&rm, &state, &actions)
            .expect("translatable pick");
        assert_eq!(
            actions[idx],
            translate_combat_action(&engine_pick).expect("PlayCard translates"),
            "deployed path must return the engine pick, not a re-derived argmax"
        );
        let rec = policy.last_combat_search().expect("search recorded");
        assert_eq!(
            rec.chosen, engine_pick,
            "recorded chosen action must be the unified pick"
        );
        assert!(
            rec.outcomes.iter().any(|o| o.action == rec.chosen),
            "chosen action must be present in the recorded root outcomes"
        );
    }

    /// A forced root (one legal action) is answered without a search, and
    /// no later decision can tell. The control policy carries a no-op
    /// drink-jitter seam (`p = 0` never masks), which keeps every ply
    /// searched under an otherwise identical config: both must take the
    /// same forced pick AND produce a bit-identical search on the next
    /// decision, i.e. the skipped search advanced nothing but the counter.
    #[test]
    fn mcts_pick_skips_the_search_on_a_forced_root() {
        let run_seed = 3;
        let sims = 200;
        let open = build_ftl_busy_hand_state(run_seed);
        let mut forced = open.clone();
        let hand = std::mem::take(&mut forced.player.hand);
        forced.player.discard.extend(hand);
        assert_eq!(
            sts2_core::legal_actions(&forced).as_slice(),
            &[Action::EndTurn]
        );

        let rm = new_run_manager_for_character(run_seed, CharacterId::Ironclad);
        let budget = CombatBudget {
            sims,
            boss_sims: sims,
            elite_sims: None,
            motif_sims: None,
            leaf: LeafKind::TexelWide,
        };
        let mut actions: Vec<RunAction> = vec![RunAction::EndTurn];
        for hand_idx in 0..6 {
            for target in [None, Some(0), Some(1), Some(2)] {
                actions.push(RunAction::PlayCard { hand_idx, target });
            }
        }
        let mut skipping = MctsCombatPolicy::new(FirstOptionPolicy, budget, run_seed);
        let mut searching = MctsCombatPolicy::new(FirstOptionPolicy, budget, run_seed);
        searching.drink_jitter = potion_oc::resolve_drink_jitter(Some("0:0"));

        // Decision 0: forced.
        let picked = skipping.mcts_pick(&rm, &forced, &actions);
        assert_eq!(picked, Some(0), "EndTurn is surfaced at index 0");
        assert_eq!(picked, searching.mcts_pick(&rm, &forced, &actions));
        let stub = skipping
            .last_combat_search()
            .expect("a forced pick records");
        assert_eq!(stub.chosen, Action::EndTurn);
        assert_eq!(stub.outcomes.len(), 1);
        assert_eq!(stub.outcomes[0].action, Action::EndTurn);
        assert_eq!(stub.outcomes[0].visits, 0, "no search ran");
        let searched = searching.last_combat_search().expect("search recorded");
        assert!(searched.outcomes.iter().map(|o| o.visits).sum::<u32>() > 0);
        assert_eq!(
            skipping.combat_decisions(),
            1,
            "the forced move takes its slot"
        );
        assert_eq!(skipping.combat_budget_spent().3, 0);
        assert_eq!(searching.combat_budget_spent().3, u64::from(sims));

        // Decision 1: a real choice, searched by both under decision 1's seed.
        let picked = skipping.mcts_pick(&rm, &open, &actions);
        assert!(picked.is_some());
        assert_eq!(picked, searching.mcts_pick(&rm, &open, &actions));
        let root = |p: &MctsCombatPolicy<FirstOptionPolicy>| {
            let rec = p.last_combat_search().expect("searched");
            let rows: Vec<_> = rec
                .outcomes
                .iter()
                .map(|o| (o.action.clone(), o.visits, o.value_sum.to_bits()))
                .collect();
            (rec.chosen.clone(), rows)
        };
        assert_eq!(root(&skipping), root(&searching));
    }

    // -- `mcts_config_for` baked-constants regression test ------------------

    /// The missing regression the 7b reviewer flagged: the F2 retune's
    /// baked constants (c_puct=2.5, draw_widening_c=2.0) must actually
    /// land on the `MctsConfig` built for an unset-env (now-default
    /// `BeliefDraw`) pick, and must NOT leak into any other resolved mode
    /// (which must stay byte-identical to `MctsConfig::default()`'s
    /// c_puct/draw_widening_c).
    #[test]
    #[allow(clippy::float_cmp)] // exact literal pins (2.5/2.0 assigned directly, no arithmetic)
    fn mcts_config_for_belief_draw_carries_baked_retune() {
        let hidden_info = resolve_hidden_info(None);
        assert_eq!(hidden_info, sts2_mcts::HiddenInfoMode::BeliefDraw);

        // The F2 retune's 2.5 is now budget-keyed (`belief_draw_c_puct`), so
        // assert it at a DEPLOYED budget. At `sims = 8` the correct answer is
        // 0.8, not 2.5 — checked separately below.
        let config = mcts_config_for(
            hidden_info,
            /* max_simulations */ 1000,
            /* seed */ 42,
            /* max_rollout_depth */ COMBAT_ROLLOUT_DEPTH,
            /* terminal_eval_mode */ MctsConfig::default().terminal_eval_mode,
        );
        assert_eq!(config.hidden_info, sts2_mcts::HiddenInfoMode::BeliefDraw);
        assert_eq!(config.c_puct, 2.5, "F2 retune c_puct must be baked in");
        assert_eq!(
            config.draw_widening_c, 2.0,
            "F2 retune draw_widening_c must be baked in"
        );
        // 2026-08-17 EndTurn-widening retune (PR #467): pinned here so a
        // silent revert to the engine defaults (0.5 / 0.4) fails loudly
        // rather than costing ~10pp at A8 / ~9.6pp at A0 unnoticed.
        assert_eq!(
            config.endturn_widening_c, 5.787,
            "EndTurn-widening retune endturn_widening_c must be baked in"
        );
        assert_eq!(
            config.endturn_widening_alpha, 0.436,
            "EndTurn-widening retune endturn_widening_alpha must be baked in"
        );
        // GH #777 playout net: +4.67pp pooled Ironclad full-run win rate
        // (1,200 paired runs). The other characters fall back to the
        // heuristic playout inside the policy, so it is set for all.
        assert_eq!(
            config.rollout_policy,
            sts2_mcts::RolloutPolicyKind::Net,
            "the playout net must be the honest-mode default"
        );

        // Every other resolved mode keeps the engine defaults untouched.
        let default_c_puct = MctsConfig::default().c_puct;
        let default_draw_widening_c = MctsConfig::default().draw_widening_c;
        let default_endturn_widening_c = MctsConfig::default().endturn_widening_c;
        let default_endturn_widening_alpha = MctsConfig::default().endturn_widening_alpha;
        for other in [
            sts2_mcts::HiddenInfoMode::Clairvoyant,
            sts2_mcts::HiddenInfoMode::RootScramble,
            sts2_mcts::HiddenInfoMode::DrawChance,
        ] {
            let config = mcts_config_for(
                other,
                8,
                42,
                COMBAT_ROLLOUT_DEPTH,
                MctsConfig::default().terminal_eval_mode,
            );
            assert_eq!(
                config.c_puct, default_c_puct,
                "{other:?} must not pick up the BeliefDraw-only retune"
            );
            assert_eq!(
                config.draw_widening_c, default_draw_widening_c,
                "{other:?} must not pick up the BeliefDraw-only retune"
            );
            assert_eq!(
                config.endturn_widening_c, default_endturn_widening_c,
                "{other:?} must not pick up the BeliefDraw-only retune"
            );
            assert_eq!(
                config.endturn_widening_alpha, default_endturn_widening_alpha,
                "{other:?} must not pick up the BeliefDraw-only retune"
            );
        }
    }

    /// `belief_draw_c_puct` returns exact literals, but the repo lints
    /// `float_cmp`, so compare the way `play_to_terminal_config.rs` does.
    fn c_puct_eq(sims: u32, expected: f32) -> bool {
        (belief_draw_c_puct(sims) - expected).abs() < f32::EPSILON
    }

    /// The deployed budgets must be byte-identical to the pre-rule constant —
    /// this rule is a guard on the low-sims corner, not a retune of production.
    #[test]
    fn belief_draw_c_puct_is_unchanged_at_every_deployed_budget() {
        for sims in [
            CombatBudget::eval_cheap().sims,
            CombatBudget::eval_cheap().boss_sims,
            CombatBudget::eval_strong().sims,
            CombatBudget::eval_strong().boss_sims,
            8000, // realgame_advisor's --sims default
        ] {
            assert!(
                c_puct_eq(sims, 2.5),
                "sims={sims} is a deployed budget and must keep the F2 constant, got {}",
                belief_draw_c_puct(sims),
            );
        }
    }

    /// Monotone non-decreasing in the budget, and pinned at the two step
    /// boundaries the sweep measured. A rule that stepped the wrong way, or
    /// stepped one budget early, would still satisfy the deployed-budget test
    /// above.
    #[test]
    fn belief_draw_c_puct_steps_up_with_budget() {
        assert!(c_puct_eq(1, 0.8));
        assert!(c_puct_eq(300, 0.8));
        assert!(c_puct_eq(499, 0.8));
        assert!(c_puct_eq(500, 1.41));
        assert!(c_puct_eq(999, 1.41));
        // 1000 is `eval_cheap().sims`, the main deployed combat lane. The
        // sweep's best arm there is 1.41, but only by 0.8pp at p = 0.33 — the
        // rule must NOT take it.
        assert!(c_puct_eq(1000, 2.5));
        assert!(c_puct_eq(2000, 2.5));

        let mut prev = 0.0_f32;
        for sims in [1, 200, 300, 499, 500, 999, 1000, 2000, 4000, 8000, 50_000] {
            let c = belief_draw_c_puct(sims);
            assert!(
                c >= prev,
                "c_puct must not fall as the budget rises (sims={sims})"
            );
            prev = c;
        }
    }

    // -- `veto_lethal_event_choice` (GH #300) -------------------------------

    /// `SlipperyBridge` at 3 HP: "Hold On" (idx 1) previews to −3 HP =
    /// death, so the guard must swap the pick to "Overcome" (idx 0). At
    /// healthy HP the identical pick passes through untouched, and a
    /// non-lethal pick is never rerouted — the guard is a suicide veto,
    /// not a policy override.
    #[test]
    fn veto_lethal_event_choice_swaps_suicide_pick_only() {
        use sts2_core::events::EventId;
        let mut rm = new_run_manager_for_character(3, CharacterId::Ironclad);
        rm.run_state.enter_event_room(EventId::SlipperyBridge);
        let actions = vec![
            RunAction::EventChoice { idx: 0 }, // Overcome: lose a card, 0 HP
            RunAction::EventChoice { idx: 1 }, // Hold On: 3+page unblockable
        ];

        let mut counts = EventVetoCounts::default();
        rm.run_state.current_hp = 3;
        assert_eq!(
            veto_lethal_event_choice(&rm, &actions, 1, &mut counts),
            0,
            "lethal Hold On at 3 HP must swap to Overcome"
        );
        assert_eq!(counts.vetoed, 1, "the swap must be countable, not inferred");
        assert_eq!(
            veto_lethal_event_choice(&rm, &actions, 0, &mut counts),
            0,
            "non-lethal Overcome pick must pass through"
        );
        assert_eq!(counts.vetoed, 1, "a pass-through must not count as a veto");

        rm.run_state.current_hp = 80;
        assert_eq!(
            veto_lethal_event_choice(&rm, &actions, 1, &mut counts),
            1,
            "healthy Hold On pick must pass through"
        );
        assert_eq!(counts.total(), 1);
    }

    /// Every option lethal ⇒ the pick stands (the game allows it), and the
    /// outcome is counted separately from a real veto so a sweep can tell
    /// "the guard had nowhere to go" from "the guard never fired".
    #[test]
    fn veto_lethal_event_choice_leaves_an_all_lethal_pick_alone() {
        use sts2_core::events::EventId;
        let mut rm = new_run_manager_for_character(3, CharacterId::Ironclad);
        rm.run_state.enter_event_room(EventId::SlipperyBridge);
        // Only the lethal option is surfaced, so there is no survivor to
        // swap to.
        let actions = vec![RunAction::EventChoice { idx: 1 }];
        rm.run_state.current_hp = 1;

        let mut counts = EventVetoCounts::default();
        assert_eq!(veto_lethal_event_choice(&rm, &actions, 0, &mut counts), 0);
        assert_eq!(counts.vetoed, 0);
        assert_eq!(counts.all_lethal, 1);
    }

    /// A pick the engine cannot preview is left alone — but counted, because
    /// this is the one bucket that would hide a real suicide.
    #[test]
    fn veto_lethal_event_choice_counts_an_unpreviewable_pick() {
        use sts2_core::events::EventId;
        let mut rm = new_run_manager_for_character(3, CharacterId::Ironclad);
        rm.run_state.enter_event_room(EventId::SlipperyBridge);
        // idx 9 is past the end of the presented list, so `preview_option_
        // effect` returns None — the same shape as a handler the preview
        // cannot resolve.
        let actions = vec![RunAction::EventChoice { idx: 9 }];
        rm.run_state.current_hp = 1;

        let mut counts = EventVetoCounts::default();
        assert_eq!(veto_lethal_event_choice(&rm, &actions, 0, &mut counts), 0);
        assert_eq!(counts.unpreviewable_pick, 1);
        assert_eq!(counts.vetoed, 0);
    }

    /// A non-`EventChoice` pick, and an out-of-range `picked`, both pass
    /// through without panicking. The old code indexed `actions[picked]`
    /// directly.
    #[test]
    fn veto_lethal_event_choice_ignores_non_event_and_out_of_range() {
        let rm = new_run_manager_for_character(3, CharacterId::Ironclad);
        let mut counts = EventVetoCounts::default();
        let actions = vec![RunAction::EndTurn];
        assert_eq!(veto_lethal_event_choice(&rm, &actions, 0, &mut counts), 0);
        assert_eq!(veto_lethal_event_choice(&rm, &[], 7, &mut counts), 7);
        assert_eq!(counts, EventVetoCounts::default());
    }

    // -- `pick_best_survivor`: the substitution rules, content-free --------

    /// Deltas are signed, so the cheapest survivor is the LARGEST delta.
    /// With only one survivor per event in the current content, this rule
    /// is otherwise never evaluated.
    #[test]
    fn pick_best_survivor_takes_the_smallest_hp_loss() {
        let mut counts = EventVetoCounts::default();
        let candidates = [(0, Some(-9)), (1, Some(-2)), (2, Some(-5))];
        assert_eq!(pick_best_survivor(50, &candidates, &mut counts), Some(1));
    }

    /// Ties resolve to the lowest action index — the guard must be a
    /// deterministic function of the surfaced order.
    #[test]
    fn pick_best_survivor_breaks_ties_on_lowest_index() {
        let mut counts = EventVetoCounts::default();
        let candidates = [(3, Some(-4)), (1, Some(-4)), (2, Some(-4))];
        assert_eq!(pick_best_survivor(50, &candidates, &mut counts), Some(3));
    }

    /// The regression this guard's own review turned up: an option that
    /// cannot be previewed must NOT beat a real survivor. Scored as 0 it
    /// looked HP-neutral and won every comparison.
    #[test]
    fn pick_best_survivor_never_prefers_an_unpreviewable_option() {
        let mut counts = EventVetoCounts::default();
        let candidates = [(0, None), (1, Some(-5))];
        assert_eq!(
            pick_best_survivor(50, &candidates, &mut counts),
            Some(1),
            "an unpreviewable option must not outrank a known -5"
        );
        assert_eq!(counts.unpreviewable_alt, 1);
    }

    /// Survivors are filtered on lethality first: -49 at 50 HP survives,
    /// -50 does not.
    #[test]
    fn pick_best_survivor_excludes_the_lethal() {
        let mut counts = EventVetoCounts::default();
        assert_eq!(
            pick_best_survivor(50, &[(0, Some(-50)), (1, Some(-49))], &mut counts),
            Some(1)
        );
        assert_eq!(
            pick_best_survivor(50, &[(0, Some(-50)), (1, Some(-60))], &mut counts),
            None,
            "no survivor ⇒ caller leaves the pick alone"
        );
    }

    // -- `build_search` extraction golden (GH #830) -------------------------

    /// Root distribution of the last decision, every float as its bit
    /// pattern, one line per root child.
    fn search_fingerprint<M: RunPolicy>(policy: &MctsCombatPolicy<M>) -> String {
        let rec = policy.last_combat_search().expect("search recorded");
        let mut out = format!("chosen={:?}\n", rec.chosen);
        for o in &rec.outcomes {
            out.push_str(&format!(
                "{:?}|{}|{}|{}|{}|{:08x}|{:08x}|{:08x}|{}|{:08x}\n",
                o.action,
                o.visits,
                o.wins,
                o.losses,
                o.ties,
                o.value_sum.to_bits(),
                o.value_sq_sum.to_bits(),
                o.hp_sum.to_bits(),
                o.hp_n,
                o.mean_value.to_bits(),
            ));
        }
        out
    }

    /// Captured from the deployed `mcts_pick` BEFORE `build_search` was
    /// extracted from it; the extraction must not move a single visit or
    /// value bit. Two decisions, so the per-decision seed derivation is
    /// pinned too.
    const BUILD_SEARCH_GOLDEN: &str = concat!(
        "chosen=PlayCard { hand_index: 4, target: None }\n",
        "PlayCard { hand_index: 3, target: Some(0) }|100|100|0|0|42fb2dea|431dbf5e|45e01000|100|3fa0c13a\n",
        "PlayCard { hand_index: 0, target: Some(0) }|98|98|0|0|42f59bb9|4319e7d6|45d93000|98|3fa065b2\n",
        "PlayCard { hand_index: 4, target: None }|113|113|0|0|430f2cbb|43356e9e|46041c00|113|3fa22e23\n",
        "EndTurn|89|89|0|0|42dd7623|4309ca1b|45be5000|89|3f9f40d1\n",
        "chosen=PlayCard { hand_index: 3, target: Some(0) }\n",
        "PlayCard { hand_index: 3, target: Some(0) }|136|136|0|0|432c1e0f|4359daa7|461e2000|136|3fa1fe2c\n",
        "PlayCard { hand_index: 0, target: Some(0) }|136|136|0|0|432c14f3|4359c210|461df800|136|3fa1f599\n",
        "EndTurn|128|128|0|0|43215458|434b5e51|4611f000|128|3fa15458\n",
    );

    #[test]
    fn build_search_refactor_is_byte_identical() {
        let run_seed = 7;
        let sims = 400;
        let state = mcts_fixtures::build_vertical_slice(7);
        let rm = new_run_manager_for_character(run_seed, CharacterId::Ironclad);
        let budget = CombatBudget {
            sims,
            boss_sims: sims,
            elite_sims: None,
            motif_sims: None,
            leaf: LeafKind::TexelWide,
        };
        let mut policy = MctsCombatPolicy::new(FirstOptionPolicy, budget, run_seed);
        let mut actions: Vec<RunAction> = vec![RunAction::EndTurn];
        for hand_idx in 0..10 {
            for target in [None, Some(0), Some(1), Some(2)] {
                actions.push(RunAction::PlayCard { hand_idx, target });
            }
        }
        let mut got = String::new();
        policy
            .mcts_pick(&rm, &state, &actions)
            .expect("decision 0 surfaced");
        got.push_str(&search_fingerprint(&policy));
        let mut next = state.clone();
        let chosen = policy
            .last_combat_search()
            .expect("searched")
            .chosen
            .clone();
        let _ = sts2_core::apply(&mut next, chosen);
        policy
            .mcts_pick(&rm, &next, &actions)
            .expect("decision 1 surfaced");
        got.push_str(&search_fingerprint(&policy));
        assert_eq!(got, BUILD_SEARCH_GOLDEN, "golden:\n{got}");
    }

    /// `build_search` (the turn-plan teacher's builder) searches exactly
    /// what a fresh `mcts_pick` searches: same seed, same root distribution
    /// bit for bit. `mcts_pick` builds its fresh search inline (tree reuse
    /// needs the config on its own), so this pins the two together.
    #[test]
    fn build_search_matches_the_fresh_pick_search() {
        let run_seed = 7;
        let sims = 400;
        let state = mcts_fixtures::build_vertical_slice(7);
        let rm = new_run_manager_for_character(run_seed, CharacterId::Ironclad);
        let budget = CombatBudget {
            sims,
            boss_sims: sims,
            elite_sims: None,
            motif_sims: None,
            leaf: LeafKind::TexelWide,
        };
        let mut actions: Vec<RunAction> = vec![RunAction::EndTurn];
        for hand_idx in 0..10 {
            for target in [None, Some(0), Some(1), Some(2)] {
                actions.push(RunAction::PlayCard { hand_idx, target });
            }
        }
        let mut picker = MctsCombatPolicy::new(FirstOptionPolicy, budget, run_seed);
        picker
            .mcts_pick(&rm, &state, &actions)
            .expect("decision 0 surfaced");
        let picked = &picker.last_combat_search().expect("searched").outcomes;

        // `mcts_pick`'s seed for decision 0 (counter 0).
        let seed_bits = u64::from(u32::from_ne_bytes(run_seed.to_ne_bytes()));
        let seed = splitmix64(seed_bits ^ 1u64.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let mut builder = MctsCombatPolicy::new(FirstOptionPolicy, budget, run_seed);
        let mut mcts = builder.build_search(&rm, &state, sims, seed);
        mcts.try_search_n(sims).expect("searched");
        let built = mcts.root_outcomes();
        assert_eq!(format!("{built:?}"), format!("{picked:?}"));
    }

    // ── the per-node PUCT prior cache (GH #830) ──────────────────────────

    /// The built-in Ironclad playout net's file, which `STS2_RUNAI_PRIOR_NET`
    /// can name as a prior; `None` (and a skip note) in a checkout without it.
    fn ironclad_playout_net_file() -> Option<&'static str> {
        const PATH: &str = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../data/rollout_nets/ironclad.json"
        );
        if std::path::Path::new(PATH).is_file() {
            Some(PATH)
        } else {
            eprintln!("no built-in Ironclad playout net (public snapshot): skipped");
            None
        }
    }

    /// A fresh Ironclad run (seed `seed`) at its first combat, reached by
    /// first-option picks.
    fn ironclad_first_combat(seed: i32) -> RunManager {
        crate::rollout::ensure_content_installed();
        let mut rm = sts2_content::run_callbacks::new_run_manager_for_character_with_ascension(
            seed,
            CharacterId::Ironclad,
            0,
        )
        .with_neow_start();
        sts2_content::run_callbacks::initialize_run_setup(&mut rm);
        for _ in 0..64 {
            if rm.phase() == Phase::Combat {
                return rm;
            }
            let acts = rm.get_available_actions();
            let _ = rm.take_action(acts[0]);
        }
        panic!("no combat within 64 run actions");
    }

    /// `rm`'s combat searched by the pilot's own route (`build_search`) for
    /// `sims` simulations: whether the search ran a PUCT prior, and the
    /// prior reads and evaluations it added to the process-wide counters.
    fn searched_with_the_env(rm: &RunManager, sims: u32) -> (bool, u64, u64) {
        let budget = CombatBudget::eval_cheap();
        let mut policy = MctsCombatPolicy::new(HeuristicPolicy::new(), budget, 3);
        let combat = rm.combat.clone().expect("in combat");
        let before = sts2_mcts::prior_counts::snapshot();
        let mut mcts = policy.build_search(rm, &combat, sims, 1);
        mcts.try_search_n(sims).expect("searchable");
        let after = sts2_mcts::prior_counts::snapshot();
        (
            mcts.config.use_puct_prior,
            after.lookups - before.lookups,
            after.evals - before.evals,
        )
    }

    /// Unset, the pilot's search caches the prior: fewer evaluations than
    /// reads. The twin of the off switch below, so that one can fail.
    #[test]
    fn the_pilots_search_caches_the_prior_by_default() {
        let Some(net) = ironclad_playout_net_file() else {
            return;
        };
        std::env::set_var("STS2_RUNAI_PRIOR_NET", net);
        std::env::remove_var("STS2_RUNAI_PRIOR_CACHE");
        let rm = ironclad_first_combat(3);
        let (prior, lookups, evals) = searched_with_the_env(&rm, 64);
        assert!(prior, "no PUCT prior");
        assert!(evals > 0 && evals < lookups, "{evals} / {lookups}");
    }

    /// `STS2_RUNAI_PRIOR_CACHE=0` reaches the pilot's tree: one evaluation
    /// per read.
    #[test]
    fn the_prior_cache_variable_turns_the_cache_off() {
        let Some(net) = ironclad_playout_net_file() else {
            return;
        };
        std::env::set_var("STS2_RUNAI_PRIOR_NET", net);
        std::env::set_var("STS2_RUNAI_PRIOR_CACHE", "0");
        let rm = ironclad_first_combat(3);
        let (prior, lookups, evals) = searched_with_the_env(&rm, 64);
        assert!(prior, "no PUCT prior");
        assert!(evals > 0 && evals == lookups, "{evals} / {lookups}");
    }

    /// Only `0` and `1` are values of `STS2_RUNAI_PRIOR_CACHE`: anything
    /// else stops the run instead of silently picking one.
    #[test]
    fn a_bad_prior_cache_value_panics() {
        std::env::set_var("STS2_RUNAI_PRIOR_CACHE", "2");
        let err = std::panic::catch_unwind(prior_cache_from_env).expect_err("expected a panic");
        let msg = err
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| err.downcast_ref::<&str>().map(|s| (*s).to_owned()))
            .expect("panic payload is a string");
        assert!(
            msg.contains("STS2_RUNAI_PRIOR_CACHE=\"2\"")
                && msg.contains("expected 0 (off) or 1 (on)"),
            "{msg}"
        );
    }

    // ── the Ironclad run-agent layer (GH #830) ───────────────────────────

    /// A policy with a tiny budget, constructed now (the override bundle is
    /// read here).
    fn layer_policy() -> MctsCombatPolicy<FirstOptionPolicy> {
        let budget = CombatBudget {
            sims: 8,
            boss_sims: 8,
            elite_sims: None,
            motif_sims: None,
            leaf: LeafKind::TexelWide,
        };
        MctsCombatPolicy::new(FirstOptionPolicy, budget, 1)
    }

    /// `search_config`'s config; the layer tests compare its Debug string
    /// (every field, floats exact).
    fn config_of<M: RunPolicy>(
        policy: &MctsCombatPolicy<M>,
        sims: u32,
        seed: u64,
        character: Option<CharacterId>,
    ) -> MctsConfig {
        policy.search_config(sims, seed, character).0
    }

    /// `config` with the Ironclad knobs written in by hand, the literal
    /// values of the gated string.
    fn with_the_ironclad_knobs(mut config: MctsConfig) -> MctsConfig {
        config.removed_counts_forms = true;
        config.endturn_widening_c = 23.148;
        config.draw_widening_c = 4.0;
        config
    }

    /// The config `search_config` assembled before the character layer
    /// existed: `mcts_config_for`, the three single-knob seams, then the
    /// override bundle (with every other env seam unset).
    fn pre_layer_assembly<M: RunPolicy>(
        policy: &MctsCombatPolicy<M>,
        sims: u32,
        seed: u64,
    ) -> MctsConfig {
        let mut config = mcts_config_for(
            deployed_hidden_info(),
            sims,
            seed,
            COMBAT_ROLLOUT_DEPTH,
            MctsConfig::default().terminal_eval_mode,
        );
        config.tie_eps = policy.tie_eps;
        config.loss_gradient = policy.loss_gradient;
        config.root_guards = policy.root_guards;
        policy.overrides.apply(&mut config);
        config
    }

    /// T1: the Ironclad run agent searches with exactly the gated knobs at
    /// every budget it runs (training 250, the gated 1000 / 4000, and a
    /// tiny one), and nothing else moves. The literal is the same f32 the
    /// gate's env string parsed to.
    #[test]
    fn ironclad_search_config_carries_the_gated_knobs_at_every_budget() {
        std::env::remove_var("STS2_RUNAI_MCTS_OVERRIDES");
        assert_eq!(
            23.148_f32.to_bits(),
            "23.148".parse::<f32>().expect("f32").to_bits()
        );
        let policy = layer_policy();
        for sims in [8, 250, 1000, 4000] {
            let base = config_of(&policy, sims, 7, None);
            let ironclad = config_of(&policy, sims, 7, Some(CharacterId::Ironclad));
            assert_eq!(
                format!("{ironclad:?}"),
                format!("{:?}", with_the_ironclad_knobs(base)),
                "sims {sims}"
            );
            assert_ne!(format!("{ironclad:?}"), format!("{base:?}"), "sims {sims}");
        }
    }

    /// T2 (config level): every other character, and a fight without one,
    /// searches with exactly the config the pre-layer assembly gives, at
    /// every budget.
    #[test]
    fn other_characters_get_the_pre_layer_config() {
        std::env::remove_var("STS2_RUNAI_MCTS_OVERRIDES");
        let policy = layer_policy();
        for sims in [8, 16, 250, 1000, 4000] {
            let want = format!("{:?}", pre_layer_assembly(&policy, sims, 11));
            for character in [
                Some(CharacterId::Silent),
                Some(CharacterId::Defect),
                Some(CharacterId::Regent),
                Some(CharacterId::Necrobinder),
                None,
            ] {
                assert_eq!(
                    format!("{:?}", config_of(&policy, sims, 11, character)),
                    want,
                    "{character:?} at {sims} sims"
                );
            }
            assert_ne!(
                format!(
                    "{:?}",
                    config_of(&policy, sims, 11, Some(CharacterId::Ironclad))
                ),
                want,
                "Ironclad must differ (non-vacuity) at {sims} sims"
            );
        }
    }

    /// T3: the off switch gives Ironclad the base config, bit for bit, and
    /// is a no-op for a fight without a character.
    #[test]
    fn the_off_switch_gives_ironclad_the_base_config() {
        std::env::set_var("STS2_RUNAI_MCTS_OVERRIDES", IRONCLAD_SEARCH_KNOBS_OFF);
        let off = layer_policy();
        std::env::remove_var("STS2_RUNAI_MCTS_OVERRIDES");
        let plain = layer_policy();
        for sims in [8, 250, 1000, 4000] {
            let base = format!("{:?}", config_of(&plain, sims, 3, None));
            assert_eq!(
                format!(
                    "{:?}",
                    config_of(&off, sims, 3, Some(CharacterId::Ironclad))
                ),
                base,
                "sims {sims}"
            );
            assert_eq!(format!("{:?}", config_of(&off, sims, 3, None)), base);
        }
    }

    /// T3: an env knob beats the preset's value for that knob and leaves
    /// the preset's other knobs in place.
    #[test]
    fn an_env_knob_beats_the_ironclad_preset() {
        std::env::set_var("STS2_RUNAI_MCTS_OVERRIDES", "draw_widening_c=3.0");
        let policy = layer_policy();
        for sims in [250, 1000] {
            let mut want = with_the_ironclad_knobs(pre_layer_assembly(&policy, sims, 5));
            want.draw_widening_c = 3.0;
            assert_eq!(
                format!(
                    "{:?}",
                    config_of(&policy, sims, 5, Some(CharacterId::Ironclad))
                ),
                format!("{want:?}")
            );
        }
    }

    /// T3: on another character an env knob is the whole change, as before
    /// the layer existed: no Ironclad knob comes with it.
    #[test]
    fn an_env_knob_reaches_another_character_alone() {
        std::env::remove_var("STS2_RUNAI_MCTS_OVERRIDES");
        let plain = layer_policy();
        std::env::set_var("STS2_RUNAI_MCTS_OVERRIDES", "removed_counts_forms=1");
        let policy = layer_policy();
        let mut want = config_of(&plain, 1000, 5, Some(CharacterId::Silent));
        want.removed_counts_forms = true;
        assert_eq!(
            format!(
                "{:?}",
                config_of(&policy, 1000, 5, Some(CharacterId::Silent))
            ),
            format!("{want:?}")
        );
    }

    /// `rm`'s first combat searched by `build_search` with its room set to
    /// `room`: whether the search runs a PUCT prior.
    fn prior_on_in(rm: &mut RunManager, room: RoomType) -> bool {
        rm.run_state
            .current_room
            .as_mut()
            .expect("in a room")
            .room_type = room;
        let combat = rm.combat.clone().expect("in combat");
        let mut policy =
            MctsCombatPolicy::new(HeuristicPolicy::new(), CombatBudget::eval_cheap(), 3);
        policy.build_search(rm, &combat, 8, 1).config.use_puct_prior
    }

    /// T4: plain clairvoyant mode (`STS2_RUNAI_HONEST=0`) keeps the legacy
    /// config for Ironclad (no knobs) and no built-in prior: the gate
    /// measured the layer only in the BeliefDraw dialect.
    #[test]
    fn clairvoyant_ironclad_keeps_the_legacy_search() {
        std::env::set_var("STS2_RUNAI_HONEST", "0");
        std::env::remove_var("STS2_RUNAI_MCTS_OVERRIDES");
        std::env::remove_var("STS2_RUNAI_PRIOR_NET");
        let policy = layer_policy();
        assert!(!policy.ironclad_layer_on());
        for sims in [250, 1000] {
            assert_eq!(
                format!(
                    "{:?}",
                    config_of(&policy, sims, 2, Some(CharacterId::Ironclad))
                ),
                format!("{:?}", config_of(&policy, sims, 2, None))
            );
        }
        let mut rm = first_combat(3, CharacterId::Ironclad);
        assert!(!prior_on_in(&mut rm, RoomType::Elite));
        assert!(!prior_on_in(&mut rm, RoomType::Boss));
    }

    /// T4: `belief_dialect=1` gives a clairvoyant arm the layer too (knobs
    /// and the built-in prior), so it still differs from the honest arm in
    /// information alone.
    #[test]
    fn belief_dialect_gives_clairvoyant_ironclad_the_layer() {
        std::env::set_var("STS2_RUNAI_HONEST", "0");
        std::env::set_var("STS2_RUNAI_MCTS_OVERRIDES", "belief_dialect=1");
        std::env::remove_var("STS2_RUNAI_PRIOR_NET");
        let policy = layer_policy();
        assert!(policy.ironclad_layer_on());
        for sims in [250, 1000] {
            let base = config_of(&policy, sims, 2, None);
            assert_eq!(
                format!(
                    "{:?}",
                    config_of(&policy, sims, 2, Some(CharacterId::Ironclad))
                ),
                format!("{:?}", with_the_ironclad_knobs(base))
            );
        }
        let mut rm = first_combat(3, CharacterId::Ironclad);
        assert!(prior_on_in(&mut rm, RoomType::Elite));
    }

    /// T4: in honest mode `belief_dialect=1` re-applies the base dialect
    /// first; the Ironclad knobs land after it, not under it.
    #[test]
    fn the_ironclad_knobs_survive_a_belief_dialect_reset() {
        std::env::remove_var("STS2_RUNAI_HONEST");
        std::env::set_var("STS2_RUNAI_MCTS_OVERRIDES", "belief_dialect=1");
        let policy = layer_policy();
        for sims in [250, 1000, 4000] {
            let base = config_of(&policy, sims, 2, None);
            let ironclad = config_of(&policy, sims, 2, Some(CharacterId::Ironclad));
            assert_eq!(
                format!("{ironclad:?}"),
                format!("{:?}", with_the_ironclad_knobs(base))
            );
        }
    }

    /// T6: [`IRONCLAD_SEARCH_KNOBS_OFF`] is the base BeliefDraw dialect for
    /// the three knobs: applied to the Ironclad config it gives the
    /// character-less config bit for bit. A base retune fails this until the
    /// string follows it.
    #[test]
    fn the_rollback_const_is_the_base_belief_draw_dialect() {
        std::env::remove_var("STS2_RUNAI_MCTS_OVERRIDES");
        let policy = layer_policy();
        let off = parse_mcts_overrides(Some(IRONCLAD_SEARCH_KNOBS_OFF));
        for sims in [8, 250, 499, 500, 1000, 4000] {
            let mut ironclad = config_of(&policy, sims, 9, Some(CharacterId::Ironclad));
            off.apply(&mut ironclad);
            assert_eq!(
                format!("{ironclad:?}"),
                format!("{:?}", config_of(&policy, sims, 9, None)),
                "sims {sims}"
            );
        }
        // The other direction: the gated string is exactly the preset.
        assert_eq!(
            character_search_preset(Some(CharacterId::Ironclad)),
            parse_mcts_overrides(Some(IRONCLAD_SEARCH_KNOBS))
        );
        for c in [None, Some(CharacterId::Silent), Some(CharacterId::Regent)] {
            assert_eq!(character_search_preset(c), MctsKnobOverrides::default());
        }
    }

    /// `or` keeps every set slot of the bundle and fills only the unset
    /// ones from the fallback.
    #[test]
    #[allow(clippy::float_cmp)] // exact literal pins
    fn knob_override_or_prefers_the_bundle() {
        let bundle = parse_mcts_overrides(Some("draw_widening_c=3.0,tie_eps=0.1"));
        let preset = parse_mcts_overrides(Some(IRONCLAD_SEARCH_KNOBS));
        let merged = bundle.or(preset);
        assert_eq!(merged.draw_widening_c, Some(3.0));
        assert_eq!(merged.tie_eps, Some(0.1));
        assert_eq!(merged.endturn_widening_c, Some(23.148));
        assert_eq!(merged.removed_counts_forms, Some(true));
        assert_eq!(merged.c_puct, None);
        assert_eq!(MctsKnobOverrides::default().or(preset), preset);
        assert_eq!(bundle.or(MctsKnobOverrides::default()), bundle);
    }

    /// Which move prior a search gets, as a pure function of the env's
    /// answers: the built-in prior is Ironclad's, in elite and boss fights,
    /// in the layer; `off` is none anywhere; a file keeps its old meaning.
    #[test]
    fn prior_choice_scopes_the_built_in_prior_to_ironclad_elite_and_boss() {
        use BudgetTier::{Boss, Common, Elite};
        let ic = Some(CharacterId::Ironclad);
        let tiers = [Common, Elite, Boss];
        let d = PriorNetEnv::Default;
        assert_eq!(
            prior_choice(&d, ic, Elite, None, true),
            PriorChoice::BuiltIn(CharacterId::Ironclad)
        );
        assert_eq!(
            prior_choice(&d, ic, Boss, None, true),
            PriorChoice::BuiltIn(CharacterId::Ironclad)
        );
        assert_eq!(prior_choice(&d, ic, Common, None, true), PriorChoice::None);
        for c in [
            Some(CharacterId::Silent),
            Some(CharacterId::Defect),
            Some(CharacterId::Regent),
            Some(CharacterId::Necrobinder),
            None,
        ] {
            for tier in tiers {
                for layer in [true, false] {
                    assert_eq!(
                        prior_choice(&d, c, tier, None, layer),
                        PriorChoice::None,
                        "{c:?} {tier:?}"
                    );
                    assert_eq!(
                        prior_choice(&d, c, tier, Some(&[Common, Elite, Boss]), layer),
                        PriorChoice::None,
                        "{c:?} {tier:?} with rooms"
                    );
                }
            }
        }
        let file = PriorNetEnv::File("/some/net.json".to_owned());
        for c in [ic, Some(CharacterId::Silent), None] {
            for tier in tiers {
                assert_eq!(
                    prior_choice(&PriorNetEnv::Off, c, tier, None, true),
                    PriorChoice::None
                );
                for layer in [true, false] {
                    assert_eq!(
                        prior_choice(&file, c, tier, None, layer),
                        PriorChoice::File,
                        "{c:?} {tier:?}"
                    );
                }
            }
        }
        // `STS2_RUNAI_PRIOR_ROOMS` replaces the built-in tiers and narrows a
        // file prior.
        assert_eq!(
            prior_choice(&d, ic, Elite, Some(&[Boss]), true),
            PriorChoice::None
        );
        assert_eq!(
            prior_choice(&d, ic, Boss, Some(&[Boss]), true),
            PriorChoice::BuiltIn(CharacterId::Ironclad)
        );
        assert_eq!(
            prior_choice(&d, ic, Common, Some(&[Common]), true),
            PriorChoice::BuiltIn(CharacterId::Ironclad)
        );
        assert_eq!(
            prior_choice(&file, ic, Common, Some(&[Boss]), true),
            PriorChoice::None
        );
        assert_eq!(
            prior_choice(&file, ic, Boss, Some(&[Boss]), true),
            PriorChoice::File
        );
        // Outside the layer the built-in prior is off; a file is not.
        for tier in tiers {
            assert_eq!(prior_choice(&d, ic, tier, None, false), PriorChoice::None);
        }
    }

    /// The value parse: unset = built-in, `off` = none, anything else a path.
    #[test]
    fn prior_net_values_parse() {
        assert_eq!(parse_prior_net_env(None), PriorNetEnv::Default);
        assert_eq!(parse_prior_net_env(Some("off")), PriorNetEnv::Off);
        assert_eq!(
            parse_prior_net_env(Some("/x/net.json")),
            PriorNetEnv::File("/x/net.json".to_owned())
        );
        // No aliases: these are paths.
        for v in ["OFF", "0", "none", "builtin"] {
            assert_eq!(
                parse_prior_net_env(Some(v)),
                PriorNetEnv::File(v.to_owned())
            );
        }
    }

    /// The panic message of `f`.
    fn panic_message(f: impl FnOnce() + std::panic::UnwindSafe) -> String {
        let err = std::panic::catch_unwind(f).expect_err("expected a panic");
        err.downcast_ref::<String>()
            .cloned()
            .or_else(|| err.downcast_ref::<&str>().map(|s| (*s).to_owned()))
            .expect("panic payload is a string")
    }

    /// An empty `STS2_RUNAI_PRIOR_NET` stops the run.
    #[test]
    fn an_empty_prior_net_value_panics() {
        std::env::set_var("STS2_RUNAI_PRIOR_NET", "");
        let msg = panic_message(|| {
            let _ = prior_net_env();
        });
        assert!(msg.contains("STS2_RUNAI_PRIOR_NET is empty"), "{msg}");
    }

    /// `off` from the environment.
    #[test]
    fn prior_net_off_is_read_from_the_env() {
        std::env::set_var("STS2_RUNAI_PRIOR_NET", "off");
        assert_eq!(prior_net_env(), &PriorNetEnv::Off);
    }

    /// A path from the environment.
    #[test]
    fn a_prior_net_path_is_read_from_the_env() {
        std::env::set_var("STS2_RUNAI_PRIOR_NET", "/nonexistent/net.json");
        assert_eq!(
            prior_net_env(),
            &PriorNetEnv::File("/nonexistent/net.json".to_owned())
        );
    }

    /// A blend weight outside `[0, 1]` stops the run (the prior would clamp
    /// it silently).
    #[test]
    fn a_prior_mix_outside_the_unit_interval_panics() {
        std::env::set_var("STS2_RUNAI_PRIOR_MIX", "1.5");
        let msg = panic_message(|| {
            let _ = prior_knobs();
        });
        assert!(
            msg.contains("STS2_RUNAI_PRIOR_MIX=1.5 is outside [0, 1]"),
            "{msg}"
        );
    }

    /// A temperature that is not positive stops the run.
    #[test]
    fn a_zero_prior_temperature_panics() {
        std::env::set_var("STS2_RUNAI_PRIOR_TEMP", "0");
        let msg = panic_message(|| {
            let _ = prior_knobs();
        });
        assert!(
            msg.contains("STS2_RUNAI_PRIOR_TEMP=0 is not a positive number"),
            "{msg}"
        );
    }

    /// The built-in route and the file route are one prior: the built-in
    /// text written to a file and named by `STS2_RUNAI_PRIOR_NET` gives
    /// bit-equal priors on real Ironclad fight states (what makes the
    /// gate's env-route arm and the shipped default the same search).
    #[test]
    fn the_built_in_prior_is_the_file_prior_of_its_text() {
        use sts2_mcts::Policy;
        let text = sts2_mcts::rollout_net::builtin_prior_json(CharacterId::Ironclad)
            .expect("the built-in Ironclad prior is compiled in");
        let path = std::env::temp_dir().join(format!(
            "sts2_runai_builtin_prior_{}.json",
            std::process::id()
        ));
        std::fs::write(&path, text).expect("write the prior file");
        std::env::set_var("STS2_RUNAI_PRIOR_NET", &path);
        std::env::remove_var("STS2_RUNAI_PRIOR_MIX");
        std::env::remove_var("STS2_RUNAI_PRIOR_TEMP");
        let file = file_move_prior().expect("file prior");
        let _ = std::fs::remove_file(&path);
        let built_in = builtin_move_prior(CharacterId::Ironclad).expect("built-in prior");
        let mut compared = 0;
        for seed in [3, 7, 11] {
            let rm = first_combat(seed, CharacterId::Ironclad);
            let mut s = rm.combat.clone().expect("in combat");
            for _ in 0..4 {
                if sts2_core::is_terminal(&s) {
                    break;
                }
                let acts = sts2_core::legal_actions(&s);
                let a: Vec<u32> = file.prior(&s, &acts).iter().map(|p| p.to_bits()).collect();
                let b: Vec<u32> = built_in
                    .prior(&s, &acts)
                    .iter()
                    .map(|p| p.to_bits())
                    .collect();
                assert_eq!(a, b, "seed {seed}");
                assert!(a.len() == acts.len() && !a.is_empty());
                compared += 1;
                let _ = sts2_core::apply(&mut s, acts[acts.len() / 2].clone());
            }
        }
        assert!(compared >= 9, "{compared}");
    }

    /// The pilot's search runs the built-in prior in an Ironclad elite or
    /// boss fight, and it IS the built-in prior; a hallway fight has none.
    #[test]
    fn the_pilot_runs_the_built_in_prior_in_ironclad_elite_and_boss_fights() {
        use sts2_mcts::Policy;
        std::env::remove_var("STS2_RUNAI_PRIOR_NET");
        std::env::remove_var("STS2_RUNAI_PRIOR_ROOMS");
        std::env::remove_var("STS2_RUNAI_HONEST");
        let mut rm = first_combat(3, CharacterId::Ironclad);
        assert!(!prior_on_in(&mut rm, RoomType::Monster), "hallway");
        let built_in = builtin_move_prior(CharacterId::Ironclad).expect("built-in prior");
        for room in [RoomType::Elite, RoomType::Boss] {
            assert!(prior_on_in(&mut rm, room), "{room:?}");
            let combat = rm.combat.clone().expect("in combat");
            let mut policy =
                MctsCombatPolicy::new(HeuristicPolicy::new(), CombatBudget::eval_cheap(), 3);
            let mcts = policy.build_search(&rm, &combat, 8, 1);
            let acts = sts2_core::legal_actions(&combat);
            assert_eq!(
                mcts.policy.prior(&combat, &acts),
                built_in.prior(&combat, &acts),
                "{room:?}"
            );
        }
        // Another character's elite fight has none.
        let mut silent = first_combat(3, CharacterId::Silent);
        assert!(!prior_on_in(&mut silent, RoomType::Elite));
        assert!(!prior_on_in(&mut silent, RoomType::Boss));
    }

    /// The pilot's built-in prior runs at the gated blend and temperature,
    /// written here as literals: the gate statement's "the BC prior net
    /// 95f3e909 at mix 0.5, T 1 in elite and boss rooms" (GH #830, PR #938).
    /// The two tests above compare the pilot's prior with
    /// [`builtin_move_prior`] and with the file route, and both of those read
    /// [`prior_knobs`], so a changed default would pass them. With
    /// `STS2_RUNAI_PRIOR_MIX` / `_TEMP` unset, the pilot's elite and boss
    /// prior must be bit-equal to `RolloutNetPrior::new(<the built-in net>,
    /// 0.5, 1.0)`. A prior at mix 0.6 or at T 2 must differ on the same
    /// states, so the comparison can see either knob move.
    #[test]
    fn the_built_in_prior_runs_at_the_gated_mix_and_temperature() {
        use sts2_mcts::rollout_net::{builtin_prior_json, RolloutNet, RolloutNetPrior};
        use sts2_mcts::Policy;
        fn prior_bits<P: Policy + ?Sized>(p: &P, s: &CombatState, acts: &[Action]) -> Vec<u32> {
            p.prior(s, acts).iter().map(|x| x.to_bits()).collect()
        }
        for v in [
            "STS2_RUNAI_PRIOR_NET",
            "STS2_RUNAI_PRIOR_ROOMS",
            "STS2_RUNAI_PRIOR_MIX",
            "STS2_RUNAI_PRIOR_TEMP",
            "STS2_RUNAI_COMBAT_PRIOR",
            "STS2_RUNAI_HONEST",
            "STS2_RUNAI_MCTS_OVERRIDES",
        ] {
            std::env::remove_var(v);
        }
        let (mix, temperature) = prior_knobs();
        assert_eq!(
            (mix.to_bits(), temperature.to_bits()),
            (0.5_f32.to_bits(), 1.0_f32.to_bits()),
            "the built-in prior's defaults are not the gated mix 0.5, T 1"
        );
        let text = builtin_prior_json(CharacterId::Ironclad)
            .expect("the built-in Ironclad prior is compiled in");
        let at = |mix: f32, temperature: f32| {
            RolloutNetPrior::new(
                RolloutNet::from_json(text).expect("the built-in prior parses"),
                mix,
                temperature,
            )
        };
        let gated = at(0.5, 1.0);
        let (other_mix, other_temperature) = (at(0.6, 1.0), at(0.5, 2.0));
        let (mut compared, mut mix_seen, mut temperature_seen) = (0, false, false);
        for seed in [3, 7, 11] {
            let mut rm = first_combat(seed, CharacterId::Ironclad);
            for room in [RoomType::Elite, RoomType::Boss] {
                assert!(prior_on_in(&mut rm, room), "seed {seed} {room:?}");
                let combat = rm.combat.clone().expect("in combat");
                let mut policy =
                    MctsCombatPolicy::new(HeuristicPolicy::new(), CombatBudget::eval_cheap(), 3);
                let mcts = policy.build_search(&rm, &combat, 8, 1);
                let mut s = combat;
                for _ in 0..4 {
                    if sts2_core::is_terminal(&s) {
                        break;
                    }
                    let acts = sts2_core::legal_actions(&s);
                    let shipped = prior_bits(&mcts.policy, &s, &acts);
                    assert_eq!(
                        shipped,
                        prior_bits(&gated, &s, &acts),
                        "seed {seed} {room:?}: the pilot's prior is not the gated mix 0.5, T 1"
                    );
                    assert!(shipped.len() == acts.len() && !shipped.is_empty());
                    mix_seen |= shipped != prior_bits(&other_mix, &s, &acts);
                    temperature_seen |= shipped != prior_bits(&other_temperature, &s, &acts);
                    compared += 1;
                    let _ = sts2_core::apply(&mut s, acts[acts.len() / 2].clone());
                }
            }
        }
        assert!(compared >= 18, "{compared}");
        assert!(
            mix_seen && temperature_seen,
            "these states cannot tell the knobs apart (mix {mix_seen}, T {temperature_seen})"
        );
    }

    /// `STS2_RUNAI_PRIOR_NET=off` turns the built-in prior off.
    #[test]
    fn prior_net_off_turns_the_built_in_prior_off() {
        std::env::set_var("STS2_RUNAI_PRIOR_NET", "off");
        let mut rm = first_combat(3, CharacterId::Ironclad);
        assert!(!prior_on_in(&mut rm, RoomType::Elite));
        assert!(!prior_on_in(&mut rm, RoomType::Boss));
    }
}
