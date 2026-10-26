//! Starter-potion resolution for the sts2-cli `combat` driver.
//!
//! Bead `sts2-rs-bzgs` (cliext C4). Three input paths feed into a
//! single [`resolve_potions`] entry point:
//!
//! 1. **Inline** — `--potion FirePotion,BloodPotion` (comma-separated).
//! 2. **Preset** — `--potion-preset random-3` (seeded draw from the
//!    63-entry [`PotionKind::ALL`] table; `none` is the default).
//! 3. **File** — `--potion-file path` (one [`PotionKind`] per line,
//!    `#` comments + blank lines tolerated).
//!
//! Name lookup is dual-form: each variant is matched against its
//! `PascalCase` Rust identifier (e.g. `FirePotion`) and its
//! `Title Case` rendering (e.g. `Fire Potion`), case-insensitive,
//! whitespace-normalized. Unknown names trigger a "did you mean"
//! suggestion built from a small Levenshtein walk against the same
//! dual-form table — load-bearing for the CLI's exit-1 contract on
//! typos.
//!
//! When the C1 surface (`sts2_content::names::potion_kind_from_str`)
//! lands, this module is the single replacement seam — `parse_one`
//! delegates to it.

use std::fs;
use std::path::{Path, PathBuf};

use sts2_core::PotionKind;

/// User-facing potion source. Each variant maps 1:1 to a CLI flag in
/// the `combat` argument group (see `main.rs`).
#[derive(Debug, Clone)]
pub enum PotionInput {
    /// `--potion <Kind,Kind,...>` — explicit list.
    Inline(Vec<String>),
    /// `--potion-preset <name>`. Supported: `none` (no-op),
    /// `random-3` (3 deterministic non-event-only draws).
    Preset(String),
    /// `--potion-file <path>` — one name per line.
    File(PathBuf),
    /// No potion flag was supplied; resolves to an empty belt.
    #[allow(dead_code)] // kept for callers that prefer an explicit "no input" sentinel
    None,
}

#[derive(Debug)]
pub enum PotionError {
    UnknownPotion {
        input: String,
        suggestion: Option<&'static str>,
    },
    UnknownPreset(String),
    File {
        path: PathBuf,
        source: std::io::Error,
    },
    TooMany(usize),
}

impl std::fmt::Display for PotionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownPotion { input, suggestion } => {
                if let Some(s) = suggestion {
                    write!(f, "unknown potion '{input}' (did you mean '{s}'?)")
                } else {
                    write!(f, "unknown potion '{input}'")
                }
            }
            Self::UnknownPreset(p) => write!(
                f,
                "unknown --potion-preset '{p}' \
                 (expected: none, random-3, damage-belt, defense-belt)"
            ),
            Self::File { path, source } => {
                write!(f, "--potion-file '{}' read error: {source}", path.display())
            }
            Self::TooMany(n) => write!(
                f,
                "too many potions: {n} > MAX_POTIONS=5 (engine cap, sts2_core::potions::MAX_POTIONS)"
            ),
        }
    }
}

impl std::error::Error for PotionError {}

/// Resolve `input` into a concrete list of [`PotionKind`]s ready for
/// belt-seeding. The `rng` argument is consumed by the `random-3`
/// preset so `--seed` reproduces the same draw across runs; other
/// inputs do not touch it.
pub fn resolve_potions(
    input: &PotionInput,
    rng: &mut sts2_core::dotnet_rng::Rng,
) -> Result<Vec<PotionKind>, PotionError> {
    let kinds = match input {
        PotionInput::None => Vec::new(),
        PotionInput::Inline(names) => names
            .iter()
            .map(|n| parse_one(n))
            .collect::<Result<Vec<_>, _>>()?,
        PotionInput::Preset(name) => resolve_preset(name, rng)?,
        PotionInput::File(path) => parse_file(path)?,
    };
    if kinds.len() > sts2_core::potions::MAX_POTIONS {
        return Err(PotionError::TooMany(kinds.len()));
    }
    Ok(kinds)
}

fn resolve_preset(
    name: &str,
    rng: &mut sts2_core::dotnet_rng::Rng,
) -> Result<Vec<PotionKind>, PotionError> {
    match name.to_ascii_lowercase().as_str() {
        "none" => Ok(Vec::new()),
        // Bead `sts2-rs-xzeu` (cliext F2): showcase belts. The F2 spec
        // lists `AngerPotion` and `EssenceOfSteel`; neither has a
        // ported `PotionKind` variant at pin `01c094d`, so they are
        // omitted intentionally.
        "damage-belt" => Ok(vec![PotionKind::FirePotion, PotionKind::StrengthPotion]),
        "defense-belt" => Ok(vec![PotionKind::BlockPotion, PotionKind::HeartOfIron]),
        "random-3" | "random3" => {
            // Sample 3 distinct potions from the non-event-only pool.
            // Using `choose_multiple` consumes the supplied RNG so the
            // same `--seed` reproduces the same triple.
            let pool: Vec<PotionKind> = PotionKind::ALL
                .iter()
                .copied()
                .filter(|k| !k.is_event_only())
                .collect();
            // Sample 3 distinct potions: DotNetRng Fisher-Yates shuffle
            // then take(3) — matches C# `Rng.NextInt`-based selection.
            let mut shuffled = pool.clone();
            rng.shuffle(shuffled.as_mut_slice());
            let draws: Vec<PotionKind> = shuffled.into_iter().take(3).collect();
            Ok(draws)
        }
        _ => Err(PotionError::UnknownPreset(name.to_string())),
    }
}

fn parse_file(path: &Path) -> Result<Vec<PotionKind>, PotionError> {
    let body = fs::read_to_string(path).map_err(|source| PotionError::File {
        path: path.to_path_buf(),
        source,
    })?;
    let mut out = Vec::new();
    for raw in body.lines() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        out.push(parse_one(line)?);
    }
    Ok(out)
}

/// Parse a single user-supplied potion name into a [`PotionKind`].
///
/// Accepts both the Rust `PascalCase` identifier (`FirePotion`) and
/// the human-friendly `Title Case` form (`Fire Potion`),
/// case-insensitive. The Python `UPPER_SNAKE_CASE` form
/// (`FIRE_POTION`) is also accepted for parity with the eventual
/// C1 `name_lookup` surface.
pub fn parse_one(name: &str) -> Result<PotionKind, PotionError> {
    let needle = normalize(name);
    for &k in &PotionKind::ALL {
        if name_matches(k, &needle) {
            return Ok(k);
        }
    }
    Err(PotionError::UnknownPotion {
        input: name.to_string(),
        suggestion: suggest_potion(&needle),
    })
}

/// Lowercase + strip non-alphanumeric (collapses both "Fire Potion"
/// and `FIRE_POTION` into `firepotion`).
fn normalize(s: &str) -> String {
    s.chars()
        .filter(char::is_ascii_alphanumeric)
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

pub(crate) fn pascal_name(k: PotionKind) -> &'static str {
    // Explicit per-variant table. Cannot derive from `python_name`
    // (UPPER_SNAKE) at runtime because the caller wants a `&'static
    // str`. `PotionKind` is `#[non_exhaustive]` upstream — we add a
    // catch-all that falls back to the python name so a future
    // variant compiles (banner will display the UPPER_SNAKE form
    // until this table is updated).
    match k {
        PotionKind::AttackPotion => "AttackPotion",
        PotionKind::BlockPotion => "BlockPotion",
        PotionKind::BloodPotion => "BloodPotion",
        PotionKind::ColorlessPotion => "ColorlessPotion",
        PotionKind::DexterityPotion => "DexterityPotion",
        PotionKind::EnergyPotion => "EnergyPotion",
        PotionKind::ExplosiveAmpoule => "ExplosiveAmpoule",
        PotionKind::FirePotion => "FirePotion",
        PotionKind::FlexPotion => "FlexPotion",
        PotionKind::FocusPotion => "FocusPotion",
        PotionKind::PoisonPotion => "PoisonPotion",
        PotionKind::PotionOfDoom => "PotionOfDoom",
        PotionKind::PowerPotion => "PowerPotion",
        PotionKind::SkillPotion => "SkillPotion",
        PotionKind::SpeedPotion => "SpeedPotion",
        PotionKind::StarPotion => "StarPotion",
        PotionKind::StrengthPotion => "StrengthPotion",
        PotionKind::SwiftPotion => "SwiftPotion",
        PotionKind::VulnerablePotion => "VulnerablePotion",
        PotionKind::WeakPotion => "WeakPotion",
        PotionKind::Ashwater => "Ashwater",
        PotionKind::BlessingOfTheForge => "BlessingOfTheForge",
        PotionKind::BoneBrew => "BoneBrew",
        PotionKind::Clarity => "Clarity",
        PotionKind::CunningPotion => "CunningPotion",
        PotionKind::CureAll => "CureAll",
        PotionKind::Duplicator => "Duplicator",
        PotionKind::Fortifier => "Fortifier",
        PotionKind::FyshOil => "FyshOil",
        PotionKind::GamblersBrew => "GamblersBrew",
        PotionKind::HeartOfIron => "HeartOfIron",
        PotionKind::KingsCourage => "KingsCourage",
        PotionKind::LiquidBronze => "LiquidBronze",
        PotionKind::PotionOfBinding => "PotionOfBinding",
        PotionKind::PotionOfCapacity => "PotionOfCapacity",
        PotionKind::PowderedDemise => "PowderedDemise",
        PotionKind::RadiantTincture => "RadiantTincture",
        PotionKind::RegenPotion => "RegenPotion",
        PotionKind::StableSerum => "StableSerum",
        PotionKind::TouchOfInsanity => "TouchOfInsanity",
        PotionKind::BeetleJuice => "BeetleJuice",
        PotionKind::BottledPotential => "BottledPotential",
        PotionKind::CosmicConcoction => "CosmicConcoction",
        PotionKind::DistilledChaos => "DistilledChaos",
        PotionKind::DropletOfPrecognition => "DropletOfPrecognition",
        PotionKind::EntropicBrew => "EntropicBrew",
        PotionKind::EssenceOfDarkness => "EssenceOfDarkness",
        PotionKind::FairyInABottle => "FairyInABottle",
        PotionKind::FruitJuice => "FruitJuice",
        PotionKind::GhostInAJar => "GhostInAJar",
        PotionKind::GigantificationPotion => "GigantificationPotion",
        PotionKind::LiquidMemories => "LiquidMemories",
        PotionKind::LuckyTonic => "LuckyTonic",
        PotionKind::MazalethsGift => "MazalethsGift",
        PotionKind::OrobicAcid => "OrobicAcid",
        PotionKind::PotOfGhouls => "PotOfGhouls",
        PotionKind::ShacklingPotion => "ShacklingPotion",
        PotionKind::ShipInABottle => "ShipInABottle",
        PotionKind::SneckoOil => "SneckoOil",
        PotionKind::SoldiersStew => "SoldiersStew",
        PotionKind::FoulPotion => "FoulPotion",
        PotionKind::GlowwaterPotion => "GlowwaterPotion",
        PotionKind::PotionShapedRock => "PotionShapedRock",
        // `PotionKind` is `#[non_exhaustive]`; fall back to the
        // python parity name for any future variant not yet in the
        // table above.
        _ => k.python_name(),
    }
}

fn name_matches(k: PotionKind, needle: &str) -> bool {
    let pascal = pascal_name(k);
    if normalize(pascal) == needle {
        return true;
    }
    // python_name is UPPER_SNAKE_CASE — `normalize` strips underscores
    // and lowercases so it matches the pascal form's normalized output
    // by construction. Kept explicit for clarity / future divergence.
    if normalize(k.python_name()) == needle {
        return true;
    }
    false
}

fn suggest_potion(needle: &str) -> Option<&'static str> {
    let mut best: Option<(&'static str, usize)> = None;
    for &k in &PotionKind::ALL {
        let cand = pascal_name(k);
        let d = levenshtein(needle, &normalize(cand));
        if best.is_none_or(|(_, bd)| d < bd) {
            best = Some((cand, d));
        }
    }
    // Only suggest within a small edit distance — otherwise a totally
    // unrelated input ("foo") would get a misleading "did you mean
    // 'CureAll'?".
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

/// Render the resolved belt for the opening banner.
#[must_use]
pub fn render_banner(kinds: &[PotionKind]) -> String {
    if kinds.is_empty() {
        return "potions=[]".to_string();
    }
    let names: Vec<&'static str> = kinds.iter().copied().map(pascal_name).collect();
    format!("potions=[{}]", names.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng() -> sts2_core::dotnet_rng::Rng {
        sts2_core::dotnet_rng::Rng::new(42)
    }

    #[test]
    fn inline_pascal_resolves() {
        let input = PotionInput::Inline(vec!["FirePotion".into(), "BloodPotion".into()]);
        let got = resolve_potions(&input, &mut rng()).unwrap();
        assert_eq!(got, vec![PotionKind::FirePotion, PotionKind::BloodPotion]);
    }

    #[test]
    fn inline_title_case_resolves() {
        let got = parse_one("Fire Potion").unwrap();
        assert_eq!(got, PotionKind::FirePotion);
    }

    #[test]
    fn inline_python_name_resolves() {
        let got = parse_one("FIRE_POTION").unwrap();
        assert_eq!(got, PotionKind::FirePotion);
    }

    #[test]
    fn unknown_potion_suggests() {
        let err = parse_one("FierPotion").unwrap_err();
        match err {
            PotionError::UnknownPotion {
                suggestion: Some(s),
                ..
            } => assert_eq!(s, "FirePotion"),
            other => panic!("expected UnknownPotion with suggestion, got {other:?}"),
        }
    }

    #[test]
    fn unknown_unrelated_no_suggestion() {
        let err = parse_one("totally-bogus-name-xyz").unwrap_err();
        assert!(matches!(
            err,
            PotionError::UnknownPotion {
                suggestion: None,
                ..
            }
        ));
    }

    #[test]
    fn random_3_is_deterministic() {
        let a = resolve_potions(&PotionInput::Preset("random-3".into()), &mut rng()).unwrap();
        let b = resolve_potions(&PotionInput::Preset("random-3".into()), &mut rng()).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), 3);
    }

    #[test]
    fn preset_none_yields_empty() {
        let got = resolve_potions(&PotionInput::Preset("none".into()), &mut rng()).unwrap();
        assert!(got.is_empty());
    }

    #[test]
    fn preset_unknown_errors() {
        let err = resolve_potions(&PotionInput::Preset("foo".into()), &mut rng()).unwrap_err();
        assert!(matches!(err, PotionError::UnknownPreset(_)));
    }

    #[test]
    fn too_many_potions_rejected() {
        let names = vec![
            "FirePotion".into(),
            "BloodPotion".into(),
            "BlockPotion".into(),
            "EnergyPotion".into(),
            "Clarity".into(),
            "Duplicator".into(),
        ];
        let err = resolve_potions(&PotionInput::Inline(names), &mut rng()).unwrap_err();
        assert!(matches!(err, PotionError::TooMany(6)));
    }

    #[test]
    fn preset_damage_belt_yields_fire_and_strength() {
        let got = resolve_potions(&PotionInput::Preset("damage-belt".into()), &mut rng()).unwrap();
        assert_eq!(
            got,
            vec![PotionKind::FirePotion, PotionKind::StrengthPotion]
        );
    }

    #[test]
    fn preset_defense_belt_yields_block_and_heart_of_iron() {
        let got = resolve_potions(&PotionInput::Preset("defense-belt".into()), &mut rng()).unwrap();
        assert_eq!(got, vec![PotionKind::BlockPotion, PotionKind::HeartOfIron]);
    }

    #[test]
    fn render_banner_lists_names() {
        let s = render_banner(&[PotionKind::FirePotion, PotionKind::BloodPotion]);
        assert!(s.contains("FirePotion"));
        assert!(s.contains("BloodPotion"));
    }

    #[test]
    fn render_banner_empty() {
        assert_eq!(render_banner(&[]), "potions=[]");
    }
}
