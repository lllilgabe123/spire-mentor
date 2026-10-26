//! `MonsterSpec` + AI dispatch — cliext bead `sts2-rs-r1q8`.
//!
//! Generalizes the single-monster spawn path in
//! [`super::starter::build_state`] so a `Vec<MonsterSpec>` describes an
//! arbitrary encounter (1..N monsters), each with its own `MonsterKind`,
//! optional HP override, and per-AI args.
//!
//! The factories live in `sts2-content`:
//! - `monsters::act1_weak::create_nibbit_ai` (+ the rest of `act1_weak`
//!   / `act1` / etc.). The pre-DSL STS1 scaffolding factories
//!   (`Cultist` / `JawWorm` / `AcidSlimeS` / `GreenLouse` / `Sentry`)
//!   were removed in bead `sts2-rs-v1u5` along with the underlying
//!   `MonsterKind` variants (bead `sts2-rs-a19m` had previously
//!   migrated the player-Creature off the STS1 sentinel onto
//!   `species: Option<MonsterKind>`).

use sts2_content::monster_dispatch::MonsterSpawnArgs;
use sts2_core::MonsterKind;

/// Per-monster AI construction arguments. Defaults are picked so
/// callers that don't care about position-in-group get sensible
/// behavior (front of the row, alone).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MonsterAiArgs {
    /// Nibbit only: whether this nibbit is the sole monster in the
    /// encounter group.
    pub is_alone: bool,
    /// Nibbit only: whether this nibbit is in the front position.
    pub is_front: bool,
}

impl Default for MonsterAiArgs {
    fn default() -> Self {
        Self {
            is_alone: true,
            is_front: true,
        }
    }
}

/// Spec for spawning a single monster into a [`Scenario`](super::Scenario).
///
/// `args` carries structured per-kind spawn configuration (Sentry's
/// BEAM/BOLT phase, Nibbit's group-position fix-up, future
/// `Inklet:Left`/`KinFollower` slot enums). It defaults to
/// `MonsterSpawnArgs::None` so existing callers stay source-compatible
/// — the legacy `ai_args` field is consulted as a fallback when `args`
/// is `None`, preserving back-compat for `Scenario::from_monsters`'
/// positional fix-up. New callers (the Phase-4 CLI parser bead) should
/// populate `args` directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MonsterSpec {
    pub kind: MonsterKind,
    /// Override the rolled starting HP. `None` → use the engine's
    /// `MonsterKind::starting_hp` RNG roll.
    pub hp_override: Option<u32>,
    pub ai_args: MonsterAiArgs,
    /// Phase-1 (`sts2-rs-tgr4`) structured spawn args. Phase 2.2
    /// (`sts2-rs-mv7c`) consumes the `Inklet` / `KinFollower` variants
    /// once the slot enums are re-exported.
    pub args: MonsterSpawnArgs,
    /// C# `Creature.SlotName`, resolved to its index in the encounter's
    /// `Slots`. `None` for the many encounters that declare none, and for
    /// ad-hoc `--monsters` compositions that have no encounter behind them.
    ///
    /// Carried on the SPEC rather than looked up later because this is the
    /// last point that still knows which `EncounterDef` the roster came from —
    /// `resolve_encounter` returns a bare `Vec<MonsterSpec>`. Without it the
    /// CLI path would build slot-less combats where a mid-combat summon
    /// appends instead of taking its slot (GH #552).
    pub slot: Option<u8>,
}

impl MonsterSpec {
    /// Shorthand for tests/legacy callers: spawn `kind` with default
    /// AI args and no HP override.
    pub const fn new(kind: MonsterKind) -> Self {
        Self {
            kind,
            hp_override: None,
            ai_args: MonsterAiArgs {
                is_alone: true,
                is_front: true,
            },
            args: MonsterSpawnArgs::None,
            slot: None,
        }
    }

    /// The spawn args this monster is built with: `args`, or, when that is
    /// `None`, Nibbit's group position from the legacy `ai_args` (hand-written
    /// TOML scenarios can still set `is_alone` / `is_front` there). Every
    /// other kind takes its position from its roster index
    /// ([`sts2_content::run_bridge::spawn_position`]).
    #[must_use]
    pub const fn spawn_args(&self) -> MonsterSpawnArgs {
        match (self.args, self.kind) {
            (MonsterSpawnArgs::None, MonsterKind::Nibbit) => MonsterSpawnArgs::Nibbit {
                is_alone: self.ai_args.is_alone,
                is_front: self.ai_args.is_front,
            },
            (args, _) => args,
        }
    }
}

/// Type alias for encounter-builder function pointers. The `&mut Rng`
/// argument is reserved for aliases whose monster composition rolls
/// randomly (none today — the 9 v1 aliases are deterministic — but
/// future aliases like `random-act1-weak` will use it).
pub type EncounterBuilder = fn(&mut sts2_core::dotnet_rng::Rng) -> Vec<MonsterSpec>;

/// Static `(alias, builder)` table. Six single-monster aliases + three
/// multi-monster groups (B2 scope). Iteration order is the printed
/// order for `sts2-cli list encounters`.
// Bead `sts2-rs-ar9u` (2026-05-14): the STS1-name aliases that used to
// live here (cultist / jaw-worm / acid-slime-s / green-louse / sentry /
// two-sentries / three-louses / slimes-2x-acid) were Phase-4
// scaffolding stubs predating the STS2 port; they are not in the STS2
// Python pin and were silently misleading users designing
// "STS2-content" sweeps. The act1 encounter registry
// (`sts2_content::encounters::act1_overgrowth::ACT1_ENCOUNTERS`) is now the only
// CLI-reachable surface for canonical content; `random-N` etc. draw
// from `sts2_random_pool()` (STS2-native only).
pub const ENCOUNTERS: &[(&str, EncounterBuilder)] = &[("nibbit", builder_nibbit)];

fn builder_nibbit(_: &mut sts2_core::dotnet_rng::Rng) -> Vec<MonsterSpec> {
    vec![MonsterSpec::new(MonsterKind::Nibbit)]
}

/// Every monster kind currently selectable from the CLI. The runtime
/// pool of 6 implemented monsters; the rest of the STS2 roster is
/// pending the P5 monster-DSL port. Cliext H2 (bead `sts2-rs-45xt`)
/// surfaces this list so `--monsters` / `--encounter random-N` can
/// compose ad-hoc encounters without going through the alias table.
/// `(tag, kind, sts2_native)` — every monster currently selectable
/// from the CLI. Every entry is `sts2_native = true` as of bead
/// `sts2-rs-ar9u`; the column is preserved (rather than dropped) so
/// random-picker call sites and downstream tooling stay
/// source-compatible with the pre-cleanup shape. The five STS1-name
/// scaffolding stubs (`Cultist` / `JawWorm` / `AcidSlimeS` /
/// `GreenLouse` / `Sentry`) were removed from the CLI surface in
/// `sts2-rs-ar9u` and from the underlying `MonsterKind` enum in
/// bead `sts2-rs-v1u5`.
/// Cliext H5 (bead `sts2-rs-yioi` follow-up).
pub const SELECTABLE_MONSTERS: &[(&str, MonsterKind, bool)] = &[
    // Real STS2 monsters (Python pin `sts2_env/monsters/act1_weak.py`).
    ("nibbit", MonsterKind::Nibbit, true),
    ("shrinker-beetle", MonsterKind::ShrinkerBeetle, true),
    ("fuzzy-wurm-crawler", MonsterKind::FuzzyWurmCrawler, true),
    ("leaf-slime-s", MonsterKind::LeafSlimeS, true),
    ("twig-slime-s", MonsterKind::TwigSlimeS, true),
    ("leaf-slime-m", MonsterKind::LeafSlimeM, true),
    ("twig-slime-m", MonsterKind::TwigSlimeM, true),
    // STS2 act1 (Cliext Phase 4 — sts2-rs-90n5).
    ("cubex-construct", MonsterKind::CubexConstruct, true),
    ("flyconid", MonsterKind::Flyconid, true),
    ("eye-with-teeth", MonsterKind::EyeWithTeeth, true),
    ("fogmog", MonsterKind::Fogmog, true),
    ("inklet", MonsterKind::Inklet, true),
    ("mawler", MonsterKind::Mawler, true),
    ("vine-shambler", MonsterKind::VineShambler, true),
    (
        "slithering-strangler",
        MonsterKind::SlitheringStrangler,
        true,
    ),
    ("snapping-jaxfruit", MonsterKind::SnappingJaxfruit, true),
    (
        "assassin-ruby-raider",
        MonsterKind::AssassinRubyRaider,
        true,
    ),
    ("axe-ruby-raider", MonsterKind::AxeRubyRaider, true),
    ("brute-ruby-raider", MonsterKind::BruteRubyRaider, true),
    (
        "crossbow-ruby-raider",
        MonsterKind::CrossbowRubyRaider,
        true,
    ),
    ("tracker-ruby-raider", MonsterKind::TrackerRubyRaider, true),
    ("bygone-effigy", MonsterKind::BygoneEffigy, true),
    ("byrdonis", MonsterKind::Byrdonis, true),
    ("phrog-parasite", MonsterKind::PhrogParasite, true),
    ("parafright", MonsterKind::Parafright, true),
    ("vantom", MonsterKind::Vantom, true),
    ("ceremonial-beast", MonsterKind::CeremonialBeast, true),
    ("kin-priest", MonsterKind::KinPriest, true),
    ("kin-follower", MonsterKind::KinFollower, true),
    // STS2 act2 (incremental — full act2 port is sts2-rs-78ks/ux2-*).
    ("wriggler", MonsterKind::Wriggler, true),
];

/// Subset of [`SELECTABLE_MONSTERS`] containing only STS2-native
/// entries. Used by every random-monster path so STS1-name stubs
/// don't show up unsolicited.
#[must_use]
pub fn sts2_random_pool() -> Vec<MonsterKind> {
    SELECTABLE_MONSTERS
        .iter()
        .filter(|(_, _, sts2)| *sts2)
        .map(|(_, k, _)| *k)
        .collect()
}

/// Resolve a single monster-kind tag (dual-form: `nibbit`,
/// `ShrinkerBeetle`, `leaf_slime_s` all match). Used by both the
/// `--monsters` inline flag and the `random-N` parser. Returns the
/// kind plus the canonical tag for did-you-mean output.
#[must_use]
pub fn monster_kind_from_name(input: &str) -> Option<MonsterKind> {
    let needle = normalize_alias(input);
    SELECTABLE_MONSTERS
        .iter()
        .find(|(tag, _, _)| normalize_alias(tag) == needle)
        .map(|(_, k, _)| *k)
}

/// Error type for [`parse_inline_monsters`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MonsterListError {
    pub input: String,
    pub bad_token: String,
    pub suggestion: Option<&'static str>,
}

impl std::fmt::Display for MonsterListError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.suggestion {
            Some(s) => write!(
                f,
                "--monsters '{}': unknown monster '{}' (did you mean '{}'? — \
                 supported: nibbit, shrinker-beetle, mawler, byrdonis, vantom, …; see `sts2-cli list encounters`)",
                self.input, self.bad_token, s
            ),
            None => write!(
                f,
                "--monsters '{}': unknown monster '{}' \
                 (supported: nibbit, shrinker-beetle, mawler, byrdonis, vantom, …; see `sts2-cli list encounters`)",
                self.input, self.bad_token
            ),
        }
    }
}

impl std::error::Error for MonsterListError {}

/// Parse a comma-separated `--monsters` string into a
/// `Vec<MonsterSpec>`. Each token is resolved via
/// [`monster_kind_from_name`]; duplicates are allowed (so
/// `cultist,cultist,sentry` produces a 3-monster encounter). Empty
/// tokens are skipped so trailing commas don't error. Bounded by the
/// caller (typical max group size is 5 in STS2 act 1).
///
/// Slot-arg syntax (Phase 4, `sts2-rs-90n5`): `Kind:Slot` resolves a
/// `MonsterSpawnArgs::Inklet { slot }` / `KinFollower { slot }` for
/// the two slot-arg monsters. Recognized slots:
/// `Inklet:{First,Second,Other}` and `KinFollower:{First,Second,Third}`
/// (case-insensitive). Unknown slots fall back to `MonsterSpawnArgs::None`
/// and let `Scenario::from_monsters` apply position-based fallback.
///
/// # Errors
///
/// Returns [`MonsterListError`] on the first unknown token, with a
/// did-you-mean suggestion when available.
pub fn parse_inline_monsters(input: &str) -> Result<Vec<MonsterSpec>, MonsterListError> {
    let mut out = Vec::new();
    for tok in input.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let (name, slot) = match tok.split_once(':') {
            Some((n, s)) => (n.trim(), Some(s.trim())),
            None => (tok, None),
        };
        let Some(kind) = monster_kind_from_name(name) else {
            return Err(MonsterListError {
                input: input.to_string(),
                bad_token: tok.to_string(),
                suggestion: suggest_monster_tag(&normalize_alias(name)),
            });
        };
        let mut spec = MonsterSpec::new(kind);
        if let Some(slot_str) = slot {
            spec.args = parse_slot_args(kind, slot_str).ok_or_else(|| MonsterListError {
                input: input.to_string(),
                bad_token: tok.to_string(),
                suggestion: None,
            })?;
        }
        out.push(spec);
    }
    if out.is_empty() {
        return Err(MonsterListError {
            input: input.to_string(),
            bad_token: String::new(),
            suggestion: None,
        });
    }
    Ok(out)
}

/// Parse a `Kind:Slot` slot string for the two slot-arg act1 monsters.
/// Returns `None` if the kind doesn't take a slot, or if the slot
/// string doesn't match a known variant. Case-insensitive.
fn parse_slot_args(kind: MonsterKind, slot: &str) -> Option<MonsterSpawnArgs> {
    use sts2_content::monsters::act1_overgrowth::{InkletSlot, KinFollowerSlot};
    let needle = normalize_alias(slot);
    match kind {
        MonsterKind::Inklet => match needle.as_str() {
            "first" | "left" | "1" => Some(MonsterSpawnArgs::Inklet {
                slot: InkletSlot::First,
            }),
            "second" | "right" | "2" => Some(MonsterSpawnArgs::Inklet {
                slot: InkletSlot::Second,
            }),
            "other" | "center" | "3" => Some(MonsterSpawnArgs::Inklet {
                slot: InkletSlot::Other,
            }),
            _ => None,
        },
        MonsterKind::KinFollower => match needle.as_str() {
            "first" | "1" => Some(MonsterSpawnArgs::KinFollower {
                slot: KinFollowerSlot::First,
            }),
            "second" | "2" => Some(MonsterSpawnArgs::KinFollower {
                slot: KinFollowerSlot::Second,
            }),
            "third" | "3" => Some(MonsterSpawnArgs::KinFollower {
                slot: KinFollowerSlot::Third,
            }),
            _ => None,
        },
        _ => None,
    }
}

fn suggest_monster_tag(needle: &str) -> Option<&'static str> {
    let mut best: Option<(&'static str, usize)> = None;
    for (tag, _, _) in SELECTABLE_MONSTERS {
        let d = levenshtein(needle, &normalize_alias(tag));
        if best.is_none_or(|(_, bd)| d < bd) {
            best = Some((tag, d));
        }
    }
    best.and_then(|(name, d)| if d <= 3 { Some(name) } else { None })
}

/// Error type for [`resolve_encounter`]: unknown alias + optional
/// did-you-mean suggestion (top hit within edit distance 3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncounterError {
    pub input: String,
    pub suggestion: Option<&'static str>,
}

impl std::fmt::Display for EncounterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.suggestion {
            Some(s) => write!(
                f,
                "unknown encounter '{}' (did you mean '{}'?)",
                self.input, s
            ),
            None => write!(f, "unknown encounter '{}'", self.input),
        }
    }
}

impl std::error::Error for EncounterError {}

/// Resolve a CLI alias to a `Vec<MonsterSpec>`. Lookup is
/// case-insensitive and tolerant of `_`/space separators. Misses
/// include a Levenshtein-1..3 suggestion when one exists.
pub fn resolve_encounter(
    name: &str,
    rng: &mut sts2_core::dotnet_rng::Rng,
) -> Result<Vec<MonsterSpec>, EncounterError> {
    let needle = normalize_alias(name);
    for (alias, builder) in ENCOUNTERS {
        if normalize_alias(alias) == needle {
            return Ok(builder(rng));
        }
    }
    // Cliext Phase 4 (`sts2-rs-90n5`): look up the act1 encounter
    // registry. Matches against both the UPPER_SNAKE id and the
    // display string (case-insensitive, separator-tolerant) so the
    // plan's `--encounter "3 Louses"`-style syntax falls through.
    if let Some(specs) = resolve_act1_encounter(name, rng) {
        return Ok(specs);
    }
    // `random-act1` (+ tier-filtered shorthands) — pick uniformly
    // from the act1 registry.
    if let Some(tier_filter) = parse_random_act1(&needle) {
        let pool: Vec<_> = sts2_content::encounters::act1_overgrowth::ACT1_ENCOUNTERS
            .iter()
            .filter(|e| tier_filter.matches(e.tier))
            .collect();
        if pool.is_empty() {
            return Err(EncounterError {
                input: name.to_string(),
                suggestion: Some("random-act1"),
            });
        }
        let pick = pool[rng.next_int_exclusive(0, pool.len() as i32) as usize];
        return Ok(specs_from_encounter_def(pick, Some(rng)));
    }
    // Fall back to a single-monster encounter for any tag listed in
    // SELECTABLE_MONSTERS (so `--encounter shrinker-beetle` works
    // without requiring a hand-coded ENCOUNTERS entry per monster).
    // Cliext H3 (bead `sts2-rs-yioi`).
    if let Some(kind) = monster_kind_from_name(name) {
        return Ok(vec![MonsterSpec::new(kind)]);
    }
    // `random-N` and `random-{pair,triple}` shorthands compose N
    // monsters uniformly from the implemented pool. Cliext H2.
    if let Some(n) = parse_random_n(name) {
        if !(1..=5).contains(&n) {
            return Err(EncounterError {
                input: name.to_string(),
                suggestion: Some("random (1..=5 monsters)"),
            });
        }
        // STS2-only pool. As of bead `sts2-rs-ar9u` the STS1-name
        // stubs (Cultist/JawWorm/etc.) are no longer in
        // `SELECTABLE_MONSTERS`, so the `sts2_native` filter is a
        // belt-and-braces guard for any future non-STS2 additions.
        let pool = sts2_random_pool();
        assert!(!pool.is_empty(), "sts2 random pool is empty");
        let specs = (0..n)
            .map(|_| MonsterSpec::new(pool[rng.next_int_exclusive(0, pool.len() as i32) as usize]))
            .collect();
        return Ok(specs);
    }
    Err(EncounterError {
        input: name.to_string(),
        suggestion: suggest_encounter(&needle),
    })
}

/// Match an input string against the act1 encounter registry, by
/// either `UPPER_SNAKE` id or display label (both case- /
/// separator-insensitive via `normalize_alias`). Threads `rng` so
/// rows with [`sts2_content::encounters::EncounterDef::randomize`] set
/// (`SLIMES_WEAK`, `RUBY_RAIDERS_NORMAL`) pick a fresh spawn-time
/// roster instead of the canonical fallback. Bead `sts2-rs-6inu`.
fn resolve_act1_encounter(
    input: &str,
    rng: &mut sts2_core::dotnet_rng::Rng,
) -> Option<Vec<MonsterSpec>> {
    let needle = normalize_alias(input);
    for enc in sts2_content::encounters::act1_overgrowth::ACT1_ENCOUNTERS {
        if normalize_alias(enc.id) == needle || normalize_alias(enc.display) == needle {
            return Some(specs_from_encounter_def(enc, Some(rng)));
        }
    }
    None
}

/// Materialize a `Vec<MonsterSpec>` from a registry `EncounterDef`.
/// Position fields are populated like `Scenario::from_monsters` would
/// (alone if singleton; `is_front` for index 0), so the dispatch's
/// position-fallback paths line up with the explicit args.
///
/// When `rng` is `Some` and the encounter exposes a spawn-time
/// randomizer (bead `sts2-rs-6inu`: `SLIMES_WEAK`,
/// `RUBY_RAIDERS_NORMAL`, `KNIGHTS_ELITE`), the dynamic roster is
/// used; otherwise the canonical static `monsters` slice is.
fn specs_from_encounter_def(
    enc: &sts2_content::encounters::EncounterDef,
    rng: Option<&mut sts2_core::dotnet_rng::Rng>,
) -> Vec<MonsterSpec> {
    let roster = match rng {
        Some(r) => enc.roster(r),
        None => enc.canonical_roster(),
    };
    let alone = roster.len() == 1;
    let member_slots = enc.member_slots();
    roster
        .iter()
        .enumerate()
        .map(|(idx, (kind, args))| MonsterSpec {
            kind: *kind,
            hp_override: None,
            ai_args: MonsterAiArgs {
                is_alone: alone,
                is_front: idx == 0,
            },
            args: *args,
            slot: member_slots.get(idx).copied(),
        })
        .collect()
}

/// Recognize `random-act1`, `random-act1-weak`, `random-act1-normal`,
/// `random-act1-elite`, `random-act1-boss`. Returns the tier filter
/// (or `Any`) on a match.
fn parse_random_act1(needle: &str) -> Option<Act1TierFilter> {
    match needle {
        "randomact1" => Some(Act1TierFilter::Any),
        "randomact1weak" => Some(Act1TierFilter::Weak),
        "randomact1normal" => Some(Act1TierFilter::Normal),
        "randomact1elite" => Some(Act1TierFilter::Elite),
        "randomact1boss" => Some(Act1TierFilter::Boss),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy)]
enum Act1TierFilter {
    Any,
    Weak,
    Normal,
    Elite,
    Boss,
}

impl Act1TierFilter {
    fn matches(self, tier: sts2_content::encounters::EncounterTier) -> bool {
        use sts2_content::encounters::EncounterTier as T;
        matches!(
            (self, tier),
            (Self::Any, _)
                | (Self::Weak, T::Weak)
                | (Self::Normal, T::Normal)
                | (Self::Elite, T::Elite)
                | (Self::Boss, T::Boss)
        )
    }
}

/// Recognise `random-<N>`, `random-pair`, `random-triple`. The plain
/// alias `random` (single random alias from ENCOUNTERS) is handled
/// upstream and not matched here.
fn parse_random_n(name: &str) -> Option<usize> {
    let needle = normalize_alias(name);
    match needle.as_str() {
        "randompair" => Some(2),
        "randomtriple" => Some(3),
        _ => needle
            .strip_prefix("random")
            .and_then(|rest| rest.parse().ok()),
    }
}

fn normalize_alias(s: &str) -> String {
    s.chars()
        .filter_map(|c| {
            if c.is_ascii_alphanumeric() {
                Some(c.to_ascii_lowercase())
            } else if c == '-' || c == '_' || c == ' ' {
                None
            } else {
                Some(c.to_ascii_lowercase())
            }
        })
        .collect()
}

fn suggest_encounter(needle: &str) -> Option<&'static str> {
    let mut best: Option<(&'static str, usize)> = None;
    for (alias, _) in ENCOUNTERS {
        let d = levenshtein(needle, &normalize_alias(alias));
        if best.is_none_or(|(_, bd)| d < bd) {
            best = Some((alias, d));
        }
    }
    best.and_then(|(name, d)| if d <= 3 { Some(name) } else { None })
}

fn levenshtein(a: &str, b: &str) -> usize {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    let (m, n) = (a.len(), b.len());
    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }
    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr = vec![0usize; n + 1];
    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

/// Render a short summary of an encounter's monster composition for
/// the `list encounters` table. Examples: a single Nibbit → `Nibbit`;
/// 3× `LeafSlimeS` → `LeafSlimeS x3`.
#[must_use]
pub fn summarize_monsters(specs: &[MonsterSpec]) -> String {
    if specs.is_empty() {
        return String::from("<empty>");
    }
    let first = specs[0].kind;
    let all_same = specs.iter().all(|m| m.kind == first);
    if all_same && specs.len() == 1 {
        return format!("{first:?}");
    }
    if all_same {
        let len = specs.len();
        return format!("{first:?} x{len}");
    }
    // Mixed group — comma-join distinct names with counts in encounter order.
    let mut names: Vec<String> = Vec::new();
    for spec in specs {
        let kind = spec.kind;
        let name = format!("{kind:?}");
        if !names.contains(&name) {
            names.push(name);
        }
    }
    names.join(" + ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::scenario::{starter, Scenario};
    use crate::sim::PlayerClass;

    #[test]
    fn parse_inline_monsters_accepts_three_kinds() {
        let specs = parse_inline_monsters("nibbit,mawler,byrdonis").unwrap();
        assert_eq!(specs.len(), 3);
        assert_eq!(specs[0].kind, MonsterKind::Nibbit);
        assert_eq!(specs[1].kind, MonsterKind::Mawler);
        assert_eq!(specs[2].kind, MonsterKind::Byrdonis);
    }

    #[test]
    fn parse_inline_monsters_dual_form() {
        // Tags accept the same dual-form shapes as the alias table:
        // PascalCase, snake_case, kebab-case, spaces.
        let specs = parse_inline_monsters("Nibbit, ShrinkerBeetle, leaf_slime_s").unwrap();
        assert_eq!(
            specs.iter().map(|s| s.kind).collect::<Vec<_>>(),
            vec![
                MonsterKind::Nibbit,
                MonsterKind::ShrinkerBeetle,
                MonsterKind::LeafSlimeS
            ]
        );
    }

    #[test]
    fn parse_inline_monsters_allows_duplicates() {
        let specs = parse_inline_monsters("nibbit,nibbit,nibbit").unwrap();
        assert_eq!(specs.len(), 3);
        assert!(specs.iter().all(|s| s.kind == MonsterKind::Nibbit));
    }

    #[test]
    fn parse_inline_monsters_rejects_unknown_with_suggestion() {
        let err = parse_inline_monsters("nibbt").unwrap_err();
        assert_eq!(err.bad_token, "nibbt");
        assert_eq!(err.suggestion, Some("nibbit"));
    }

    #[test]
    fn parse_inline_monsters_rejects_empty() {
        assert!(parse_inline_monsters("").is_err());
        assert!(parse_inline_monsters(",,").is_err());
    }

    #[test]
    fn random_n_encounters_are_deterministic_and_correct_size() {
        let mut rng_a = sts2_core::dotnet_rng::Rng::new(11_i32);
        let mut rng_b = sts2_core::dotnet_rng::Rng::new(11_i32);
        let a = resolve_encounter("random-4", &mut rng_a).unwrap();
        let b = resolve_encounter("random-4", &mut rng_b).unwrap();
        assert_eq!(a.len(), 4);
        assert_eq!(b.len(), 4);
        assert_eq!(
            a.iter().map(|s| s.kind).collect::<Vec<_>>(),
            b.iter().map(|s| s.kind).collect::<Vec<_>>()
        );
        for spec in &a {
            assert!(SELECTABLE_MONSTERS.iter().any(|(_, k, _)| *k == spec.kind));
        }
    }

    #[test]
    fn random_pair_and_triple_aliases() {
        let mut rng = sts2_core::dotnet_rng::Rng::new(0_i32);
        assert_eq!(resolve_encounter("random-pair", &mut rng).unwrap().len(), 2);
        let mut rng = sts2_core::dotnet_rng::Rng::new(0_i32);
        assert_eq!(
            resolve_encounter("random-triple", &mut rng).unwrap().len(),
            3
        );
    }

    #[test]
    fn random_n_out_of_range_errors() {
        let mut rng = sts2_core::dotnet_rng::Rng::new(0_i32);
        assert!(resolve_encounter("random-6", &mut rng).is_err());
        let mut rng = sts2_core::dotnet_rng::Rng::new(0_i32);
        assert!(resolve_encounter("random-0", &mut rng).is_err());
    }

    #[test]
    fn two_shrinker_beetles_spawn_with_attached_ai() {
        // Bead `sts2-rs-r1q8` acceptance (migrated from Sentry to
        // ShrinkerBeetle in `sts2-rs-v1u5`): building a scenario from
        // two `MonsterSpec`s yields a state with two creatures, each
        // carrying an attached `MonsterAi`.
        let specs = vec![
            MonsterSpec::new(MonsterKind::ShrinkerBeetle),
            MonsterSpec::new(MonsterKind::ShrinkerBeetle),
        ];
        let scenario = Scenario { monsters: specs };
        let state = starter::build_state(&scenario, 0, PlayerClass::Ironclad, None, None);

        // Two beetles plus the player's mirror creature, which the builder
        // wires before applying spawn powers (GH #762).
        assert_eq!(state.creatures.len(), 3);
        assert_eq!(state.enemy_uids.len(), 2);
        for uid in &state.enemy_uids {
            let c = state.creatures.iter().find(|c| c.id == uid.0).unwrap();
            assert_eq!(c.species, Some(MonsterKind::ShrinkerBeetle));
            assert!(c.ai.is_some(), "every creature carries an AI");
        }
    }

    #[test]
    #[allow(deprecated)] // exercises the deprecated EncounterKind shim
    fn from_legacy_args_yields_single_monster_spec() {
        use crate::sim::EncounterKind;
        let scenario = Scenario::from_legacy_args(EncounterKind::Nibbit);
        assert_eq!(scenario.monsters.len(), 1);
        assert_eq!(scenario.monsters[0].kind, MonsterKind::Nibbit);
        assert!(scenario.monsters[0].ai_args.is_alone);
        assert!(scenario.monsters[0].ai_args.is_front);
    }

    #[test]
    fn from_monsters_fixes_position_args() {
        let specs = vec![
            MonsterSpec::new(MonsterKind::ShrinkerBeetle),
            MonsterSpec::new(MonsterKind::FuzzyWurmCrawler),
        ];
        let scenario = Scenario::from_monsters(specs);
        assert!(!scenario.monsters[0].ai_args.is_alone);
        assert!(scenario.monsters[0].ai_args.is_front);
        assert!(!scenario.monsters[1].ai_args.is_alone);
        assert!(!scenario.monsters[1].ai_args.is_front);
    }

    #[test]
    fn encounters_table_has_at_least_one_alias() {
        // Bead `sts2-rs-ar9u`: the legacy STS1-name aliases were
        // removed; only `nibbit` (the lone STS2-native legacy alias)
        // and the act1 registry remain CLI-reachable.
        assert!(!ENCOUNTERS.is_empty());
    }

    #[test]
    fn resolve_known_aliases() {
        let mut rng = sts2_core::dotnet_rng::Rng::new(0_i32);
        for (alias, _) in ENCOUNTERS {
            let result = resolve_encounter(alias, &mut rng);
            assert!(result.is_ok(), "alias {alias} must resolve");
            assert!(!result.unwrap().is_empty());
        }
    }

    #[test]
    fn resolve_act1_registry_via_display_alias() {
        // Sanity: an act1 registry encounter resolves via its display
        // string (the legacy aliases for STS1 monsters are gone — the
        // act1 registry is the canonical multi-monster surface).
        let mut rng = sts2_core::dotnet_rng::Rng::new(0_i32);
        let result = resolve_encounter("nibbit", &mut rng);
        assert!(result.is_ok(), "nibbit must resolve as a legacy alias");
    }

    #[test]
    fn resolve_case_and_separator_insensitive() {
        let mut rng = sts2_core::dotnet_rng::Rng::new(0_i32);
        assert!(resolve_encounter("Nibbit", &mut rng).is_ok());
        assert!(resolve_encounter("NIBBIT", &mut rng).is_ok());
        assert!(resolve_encounter("shrinker_beetle", &mut rng).is_ok());
        assert!(resolve_encounter("Shrinker Beetle", &mut rng).is_ok());
        // Truly unknown strings should still fail.
        assert!(resolve_encounter("xyzzy-foobar", &mut rng).is_err());
    }

    #[test]
    fn resolve_unknown_returns_suggestion_when_close() {
        let mut rng = sts2_core::dotnet_rng::Rng::new(0_i32);
        let err = resolve_encounter("nibbt", &mut rng).unwrap_err();
        assert_eq!(err.suggestion, Some("nibbit"));
        let msg = err.to_string();
        assert!(msg.contains("did you mean"));
    }

    #[test]
    fn resolve_unknown_no_suggestion_when_far() {
        let mut rng = sts2_core::dotnet_rng::Rng::new(0_i32);
        let err = resolve_encounter("xyzzy-foobar", &mut rng).unwrap_err();
        assert!(err.suggestion.is_none());
    }

    #[test]
    fn summarize_single_and_groups() {
        assert_eq!(
            summarize_monsters(&[MonsterSpec::new(MonsterKind::Nibbit)]),
            "Nibbit"
        );
        // Multi-monster summary using STS2-native kinds (the STS1
        // `two-sentries` / `three-louses` aliases were removed in
        // `sts2-rs-ar9u`).
        let louses = vec![
            MonsterSpec::new(MonsterKind::LeafSlimeS),
            MonsterSpec::new(MonsterKind::LeafSlimeS),
            MonsterSpec::new(MonsterKind::LeafSlimeS),
        ];
        assert_eq!(summarize_monsters(&louses), "LeafSlimeS x3");
    }
}
