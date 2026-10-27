//! Deck resolution — bead `sts2-rs-63bi` (cliext C2).
//!
//! Resolves a starting deck from three mutually exclusive CLI input
//! paths into a `Vec<DeckCardSpec>` consumed by `Scenario` /
//! `build_starter_state_from_deck`:
//!
//! 1. `--deck-preset <name>` — curated demos (`ironclad-basic`,
//!    `silent-basic`, `ironclad-attack-demo`, `silent-stealth-demo`).
//! 2. `--deck-file <path>` — newline-delimited or TOML list of card
//!    names; `#` introduces a line comment; a trailing `+` marks the
//!    card as upgraded.
//! 3. `--deck <Card,Card+,Card,...>` — inline comma-separated.
//!
//! Card-name resolution uses [`sts2_content::names::resolve_card_name`]
//! (a minimal C1 surface) — see that module for the accepted token
//! shapes and suggestion behaviour.

use std::path::{Path, PathBuf};

use serde::Deserialize;
use sts2_content::cards::CardSpec;
use sts2_content::names::{
    is_lenient_skip_card_token, resolve_card_name, suggest_card_id, PlayerClassHint,
};
use sts2_core::cards::EnchantmentId;

use crate::sim::PlayerClass;

/// Maximum allowed deck size for `--deck*` flags. Mirrors the soft cap
/// suggested in the cliext plan (`docs/plans/2026-05-14-...md` §3.2).
pub const MAX_DECK_SIZE: usize = 256;

/// A single deck entry — the static [`CardSpec`] plus an upgrade level
/// and an optional enchantment. `build_starter_state_from_deck` consumes
/// a `Vec<DeckCardSpec>`.
///
/// `upgrade_level` is `0` for unupgraded, `1` for the typical single
/// upgrade (`Foo+`), and `≥2` for cards like Searing Blow that admit
/// multiple upgrades (`Foo+N`). The engine's `Card.upgraded: bool` field
/// can only express "upgraded or not" today — see [`install_deck`] for
/// how a `>1` level is folded. Bead `sts2-rs-kbi1`.
///
/// `enchantment` is `Some(id)` if the token carried an `@KIND` suffix
/// (e.g. `STRIKE_IRONCLAD+@GOOPY`). The amount is always `1`; runtime
/// stack amounts (GOOPY's per-combat counter etc.) aren't recoverable
/// from run-file fixtures so the fixture format only encodes presence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeckCardSpec {
    pub spec: CardSpec,
    pub upgrade_level: u32,
    pub enchantment: Option<EnchantmentId>,
}

impl DeckCardSpec {
    /// Convenience: was this card upgraded at least once?
    #[inline]
    #[must_use]
    pub const fn is_upgraded(&self) -> bool {
        self.upgrade_level >= 1
    }
}

/// Which input path the user supplied. Exactly one is set by the
/// mutually-exclusive clap group.
#[derive(Debug, Clone)]
pub enum DeckInput {
    Preset(String),
    File(PathBuf),
    Inline(String),
}

/// Error returned by [`resolve_deck`]. `Display` is wired so callers
/// can `eprintln!("{e}")` directly.
#[derive(Debug)]
pub enum DeckError {
    UnknownPreset {
        name: String,
        available: Vec<String>,
    },
    UnknownCard {
        input: String,
        suggestions: Vec<String>,
    },
    Io {
        path: PathBuf,
        err: std::io::Error,
    },
    TomlParse {
        path: PathBuf,
        err: String,
    },
    Empty,
    TooLarge {
        size: usize,
        cap: usize,
    },
}

impl std::fmt::Display for DeckError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownPreset { name, available } => {
                write!(
                    f,
                    "unknown deck preset '{name}'. available: {}",
                    available.join(", ")
                )
            }
            Self::UnknownCard { input, suggestions } => {
                if suggestions.is_empty() {
                    write!(f, "unknown card '{input}'")
                } else {
                    write!(
                        f,
                        "unknown card '{input}'. did you mean: {}?",
                        suggestions.join(", ")
                    )
                }
            }
            Self::Io { path, err } => write!(f, "could not read '{}': {err}", path.display()),
            Self::TomlParse { path, err } => {
                write!(f, "could not parse TOML deck '{}': {err}", path.display())
            }
            Self::Empty => write!(f, "deck is empty"),
            Self::TooLarge { size, cap } => {
                write!(f, "deck size {size} exceeds maximum of {cap} cards")
            }
        }
    }
}

impl std::error::Error for DeckError {}

/// Preset builder function pointer: takes the active [`PlayerClass`]
/// (used to disambiguate the bare Strike/Defend tokens) and returns
/// the resolved deck.
type PresetFn = fn(PlayerClass) -> Vec<DeckCardSpec>;

/// Preset name → builder. The class is provided so a single preset can
/// pick the class-appropriate Strike/Defend.
const PRESETS: &[(&str, PresetFn)] = &[
    ("ironclad-basic", preset_ironclad_basic),
    ("silent-basic", preset_silent_basic),
    ("ironclad-attack-demo", preset_ironclad_attack_demo),
    ("silent-stealth-demo", preset_silent_stealth_demo),
    // Bead `sts2-rs-xzeu` (cliext F2): showcase named bundles.
    ("ironclad-strength-demo", preset_ironclad_strength_demo),
    ("ironclad-block-demo", preset_ironclad_block_demo),
    ("silent-poison-demo", preset_silent_poison_demo),
    ("silent-shiv-demo", preset_silent_shiv_demo),
];

fn preset_ironclad_basic(_class: PlayerClass) -> Vec<DeckCardSpec> {
    use sts2_content::cards::ironclad_basic::{bash, defend, strike};
    let mut v = Vec::with_capacity(10);
    for _ in 0..5 {
        v.push(DeckCardSpec {
            spec: strike(),
            upgrade_level: 0,
            enchantment: None,
        });
    }
    for _ in 0..4 {
        v.push(DeckCardSpec {
            spec: defend(),
            upgrade_level: 0,
            enchantment: None,
        });
    }
    v.push(DeckCardSpec {
        spec: bash(),
        upgrade_level: 0,
        enchantment: None,
    });
    v
}

fn preset_silent_basic(_class: PlayerClass) -> Vec<DeckCardSpec> {
    let hint = Some(PlayerClassHint::Silent);
    let names = [
        "Strike",
        "Strike",
        "Strike",
        "Strike",
        "Strike",
        "Defend",
        "Defend",
        "Defend",
        "Defend",
        "Defend",
        "Neutralize",
        "Survivor",
    ];
    names
        .iter()
        .map(|n| DeckCardSpec {
            spec: resolve_card_name(n, hint).expect("preset name is in registry"),
            upgrade_level: 0,
            enchantment: None,
        })
        .collect()
}

fn preset_ironclad_attack_demo(_class: PlayerClass) -> Vec<DeckCardSpec> {
    // Curated showcase: heavier on attacks + 1 upgraded Bash.
    let hint = Some(PlayerClassHint::Ironclad);
    let lines: &[(&str, bool)] = &[
        ("Strike", false),
        ("Strike", false),
        ("Strike", false),
        ("Defend", false),
        ("Defend", false),
        ("Bash", true),
        ("Anger", false),
        ("IronWave", false),
        ("PommelStrike", false),
        ("TwinStrike", false),
    ];
    lines
        .iter()
        .map(|(n, up)| DeckCardSpec {
            spec: resolve_card_name(n, hint).expect("preset name is in registry"),
            upgrade_level: u32::from(*up),
            enchantment: None,
        })
        .collect()
}

fn preset_silent_stealth_demo(_class: PlayerClass) -> Vec<DeckCardSpec> {
    let hint = Some(PlayerClassHint::Silent);
    let lines: &[(&str, bool)] = &[
        ("Strike", false),
        ("Strike", false),
        ("Defend", false),
        ("Defend", false),
        ("Defend", false),
        ("Neutralize", true),
        ("Survivor", false),
    ];
    lines
        .iter()
        .map(|(n, up)| DeckCardSpec {
            spec: resolve_card_name(n, hint).expect("preset name is in registry"),
            upgrade_level: u32::from(*up),
            enchantment: None,
        })
        .collect()
}

/// `ironclad-strength-demo` (bead `sts2-rs-xzeu`): demonstrates strength
/// scaling via `Inflame` + `BodySlam`. The F2 spec also lists `Flex`,
/// which has no ported spec at pin `01c094d`; omitted intentionally.
fn preset_ironclad_strength_demo(_class: PlayerClass) -> Vec<DeckCardSpec> {
    let hint = Some(PlayerClassHint::Ironclad);
    let lines: &[(&str, bool)] = &[("Inflame", false), ("BodySlam", false)];
    lines
        .iter()
        .map(|(n, up)| DeckCardSpec {
            spec: resolve_card_name(n, hint).expect("preset name is in registry"),
            upgrade_level: u32::from(*up),
            enchantment: None,
        })
        .collect()
}

/// `ironclad-block-demo` (bead `sts2-rs-xzeu`): block-heavy showcase —
/// `Defend x4 + IronWave + ShrugItOff`.
fn preset_ironclad_block_demo(_class: PlayerClass) -> Vec<DeckCardSpec> {
    let hint = Some(PlayerClassHint::Ironclad);
    let lines: &[(&str, bool)] = &[
        ("Defend", false),
        ("Defend", false),
        ("Defend", false),
        ("Defend", false),
        ("IronWave", false),
        ("ShrugItOff", false),
    ];
    lines
        .iter()
        .map(|(n, up)| DeckCardSpec {
            spec: resolve_card_name(n, hint).expect("preset name is in registry"),
            upgrade_level: u32::from(*up),
            enchantment: None,
        })
        .collect()
}

/// `silent-poison-demo` (bead `sts2-rs-xzeu`): two ported poison
/// commons. The F2 spec also lists `Bane` and `Catalyst`, neither of
/// which has a ported spec at pin `01c094d`; omitted intentionally.
fn preset_silent_poison_demo(_class: PlayerClass) -> Vec<DeckCardSpec> {
    let hint = Some(PlayerClassHint::Silent);
    let lines: &[(&str, bool)] = &[("DeadlyPoison", false), ("PoisonedStab", false)];
    lines
        .iter()
        .map(|(n, up)| DeckCardSpec {
            spec: resolve_card_name(n, hint).expect("preset name is in registry"),
            upgrade_level: u32::from(*up),
            enchantment: None,
        })
        .collect()
}

/// `silent-shiv-demo` (bead `sts2-rs-xzeu`): `BladeDance x2 +
/// CloakAndDagger` — three of the four canonical shiv-generators that
/// have ported specs at pin `01c094d`.
fn preset_silent_shiv_demo(_class: PlayerClass) -> Vec<DeckCardSpec> {
    let hint = Some(PlayerClassHint::Silent);
    let lines: &[(&str, bool)] = &[
        ("BladeDance", false),
        ("BladeDance", false),
        ("CloakAndDagger", false),
    ];
    lines
        .iter()
        .map(|(n, up)| DeckCardSpec {
            spec: resolve_card_name(n, hint).expect("preset name is in registry"),
            upgrade_level: u32::from(*up),
            enchantment: None,
        })
        .collect()
}

fn class_hint(class: PlayerClass) -> PlayerClassHint {
    match class {
        PlayerClass::Ironclad => PlayerClassHint::Ironclad,
        PlayerClass::Silent => PlayerClassHint::Silent,
        PlayerClass::Defect => PlayerClassHint::Defect,
        // Regent/Necrobinder: no deck-eligible card pool yet (bead
        // `sts2-rs-5m6k` accepts the strings at the scenario layer; the
        // real deck hint will route here once P6 lands those rows).
        // Fall back to Ironclad so the deck-resolver path doesn't panic.
        PlayerClass::Regent | PlayerClass::Necrobinder => PlayerClassHint::Ironclad,
    }
}

/// Public entry point. Drives the preset/file/inline branches and
/// validates the result (non-empty, ≤ [`MAX_DECK_SIZE`]).
///
/// # Errors
///
/// See [`DeckError`] variants.
pub fn resolve_deck(input: &DeckInput, class: PlayerClass) -> Result<Vec<DeckCardSpec>, DeckError> {
    let deck = match input {
        DeckInput::Preset(name) => resolve_preset(name, class)?,
        DeckInput::Inline(s) => resolve_inline(s, class)?,
        DeckInput::File(path) => resolve_file(path, class)?,
    };
    if deck.is_empty() {
        return Err(DeckError::Empty);
    }
    if deck.len() > MAX_DECK_SIZE {
        return Err(DeckError::TooLarge {
            size: deck.len(),
            cap: MAX_DECK_SIZE,
        });
    }
    Ok(deck)
}

fn resolve_preset(name: &str, class: PlayerClass) -> Result<Vec<DeckCardSpec>, DeckError> {
    let key = name.to_ascii_lowercase();
    for (preset_name, build) in PRESETS {
        if preset_name.eq_ignore_ascii_case(&key) {
            return Ok(build(class));
        }
    }
    Err(DeckError::UnknownPreset {
        name: name.to_string(),
        available: PRESETS.iter().map(|(n, _)| (*n).to_string()).collect(),
    })
}

fn resolve_inline(s: &str, class: PlayerClass) -> Result<Vec<DeckCardSpec>, DeckError> {
    let hint = Some(class_hint(class));
    let mut out = Vec::new();
    for raw in s.split(',') {
        let token = raw.trim();
        if token.is_empty() {
            continue;
        }
        // Bead `sts2-rs-qt0b` (fwxe.6): silently drop canonical-but-
        // Python-unported tokens (e.g. NOT_YET) — mirrors the relic
        // path's `is_lenient_skip_relic_token` filter in
        // `crates/sts2-cli/src/sim/scenario/relics.rs`.
        if is_lenient_skip_card_token(token) {
            continue;
        }
        out.push(parse_card_token(token, hint)?);
    }
    Ok(out)
}

fn resolve_file(path: &Path, class: PlayerClass) -> Result<Vec<DeckCardSpec>, DeckError> {
    let text = std::fs::read_to_string(path).map_err(|err| DeckError::Io {
        path: path.to_path_buf(),
        err,
    })?;
    // Try TOML first if the extension hints at it OR the content
    // starts with a `cards =` / `[deck]` token.
    let is_toml = path
        .extension()
        .and_then(|s| s.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("toml"))
        || text.trim_start().starts_with("cards")
        || text.trim_start().starts_with("[deck]");
    if is_toml {
        parse_toml_deck(path, &text, class)
    } else {
        parse_lines_deck(&text, class)
    }
}

#[derive(Debug, Deserialize)]
struct TomlDeckFile {
    /// Newline-delimited deck embedded as a TOML array. Each entry
    /// uses the same `Card` or `Card+` shape as `--deck`.
    #[serde(default)]
    cards: Vec<String>,
    #[serde(default)]
    deck: Option<TomlDeckTable>,
}

#[derive(Debug, Deserialize)]
struct TomlDeckTable {
    #[serde(default)]
    cards: Vec<String>,
}

fn parse_toml_deck(
    path: &Path,
    text: &str,
    class: PlayerClass,
) -> Result<Vec<DeckCardSpec>, DeckError> {
    let parsed: TomlDeckFile = toml::from_str(text).map_err(|err| DeckError::TomlParse {
        path: path.to_path_buf(),
        err: err.to_string(),
    })?;
    let entries: Vec<String> = if !parsed.cards.is_empty() {
        parsed.cards
    } else if let Some(d) = parsed.deck {
        d.cards
    } else {
        Vec::new()
    };
    let hint = Some(class_hint(class));
    entries
        .iter()
        .map(|t| t.trim())
        .filter(|t| !t.is_empty() && !is_lenient_skip_card_token(t))
        .map(|tok| parse_card_token(tok, hint))
        .collect()
}

fn parse_lines_deck(text: &str, class: PlayerClass) -> Result<Vec<DeckCardSpec>, DeckError> {
    let hint = Some(class_hint(class));
    let mut out = Vec::new();
    for raw_line in text.lines() {
        // Strip `# …` comments anywhere on the line.
        let line = raw_line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        // A single line may contain comma-separated tokens too, which
        // makes round-tripping `--deck` output easier.
        for raw in line.split(',') {
            let tok = raw.trim();
            if tok.is_empty() || is_lenient_skip_card_token(tok) {
                continue;
            }
            out.push(parse_card_token(tok, hint)?);
        }
    }
    Ok(out)
}

/// Parse a single deck-card token.
///
/// Grammar (bead `sts2-rs-kbi1`):
///
/// ```text
/// token       := <name> upgrade? enchant?
/// upgrade     := "+"             ; level = 1
///              | "+" <digits>    ; level = N, N >= 1
/// enchant     := "@" <KIND>
/// ```
///
/// Examples: `Strike`, `Bash+`, `Strike+3`, `Strike@GOOPY`,
/// `Bash+@SHARP`, `Strike+2@MOMENTUM`.
fn parse_card_token(token: &str, hint: Option<PlayerClassHint>) -> Result<DeckCardSpec, DeckError> {
    // Strip enchantment suffix `@KIND` first (greedy from the right).
    let (head, enchantment) = if let Some((before, kind_tok)) = token.rsplit_once('@') {
        let kind =
            parse_enchantment_kind(kind_tok.trim()).ok_or_else(|| DeckError::UnknownCard {
                input: token.to_string(),
                suggestions: vec![format!("unknown enchantment kind '{}'", kind_tok.trim())],
            })?;
        (before.trim_end(), Some(kind))
    } else {
        (token, None)
    };

    // Strip upgrade suffix `+` or `+N` from what remains.
    let (name, upgrade_level) = if let Some(rest) = head.strip_suffix('+') {
        (rest.trim_end(), 1u32)
    } else if let Some(plus_pos) = head.rfind('+') {
        let (name_part, digits_part) = head.split_at(plus_pos);
        let digits = &digits_part[1..];
        if !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit()) {
            let level: u32 = digits.parse().map_err(|_| DeckError::UnknownCard {
                input: token.to_string(),
                suggestions: vec![format!("upgrade level '{digits}' out of range")],
            })?;
            if level == 0 {
                return Err(DeckError::UnknownCard {
                    input: token.to_string(),
                    suggestions: vec!["upgrade level must be >= 1; drop the '+0'".to_string()],
                });
            }
            (name_part.trim_end(), level)
        } else {
            (head, 0u32)
        }
    } else {
        (head, 0u32)
    };

    match resolve_card_name(name, hint) {
        Ok(spec) => Ok(DeckCardSpec {
            spec,
            upgrade_level,
            enchantment,
        }),
        Err(err) => Err(DeckError::UnknownCard {
            input: err.input,
            suggestions: if err.suggestions.is_empty() {
                suggest_card_id(name)
            } else {
                err.suggestions
            },
        }),
    }
}

/// Map an enchantment-kind token (uppercase scraper-emitted form, e.g.
/// `GOOPY`, `SHARP`, `ROYALLY_APPROVED`, or PascalCase) to an
/// [`EnchantmentId`]. Returns `None` for unknown tokens.
///
/// Recognises both screaming-snake (`TEZCATARAS_EMBER`, `SOULS_POWER`,
/// `SLUMBERING_ESSENCE`, `PERFECT_FIT`) and PascalCase forms so users
/// can write either `@GOOPY` or `@Goopy`.
fn parse_enchantment_kind(s: &str) -> Option<EnchantmentId> {
    let key: String = s
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .map(|c| c.to_ascii_uppercase())
        .collect();
    let id = match key.as_str() {
        "ROYALLYAPPROVED" => EnchantmentId::RoyallyApproved,
        "STEADY" => EnchantmentId::Steady,
        "TEZCATARASEMBER" => EnchantmentId::TezcatarasEmber,
        "SOULSPOWER" => EnchantmentId::SoulsPower,
        "INSTINCT" => EnchantmentId::Instinct,
        "GOOPY" => EnchantmentId::Goopy,
        "SHARP" => EnchantmentId::Sharp,
        "MOMENTUM" => EnchantmentId::Momentum,
        "CORRUPTED" => EnchantmentId::Corrupted,
        "FAVORED" => EnchantmentId::Favored,
        "VIGOROUS" => EnchantmentId::Vigorous,
        "NIMBLE" => EnchantmentId::Nimble,
        "SPIRAL" => EnchantmentId::Spiral,
        "GLAM" => EnchantmentId::Glam,
        "ADROIT" => EnchantmentId::Adroit,
        "SWIFT" => EnchantmentId::Swift,
        "SOWN" => EnchantmentId::Sown,
        "IMBUED" => EnchantmentId::Imbued,
        "SLITHER" => EnchantmentId::Slither,
        "SLUMBERINGESSENCE" => EnchantmentId::SlumberingEssence,
        "PERFECTFIT" => EnchantmentId::PerfectFit,
        "CLONE" => EnchantmentId::Clone,
        "INKY" => EnchantmentId::Inky,
        _ => return None,
    };
    Some(id)
}

/// Render a one-line banner enumerating the resolved deck.
///
/// Format: `deck (<N> cards): Strike, Defend, Bash+, Anger, ...`.
/// Upgraded cards get a trailing `+`. Used by `main.rs` before
/// Replace the starter deck on `state` with the resolved
/// [`DeckCardSpec`] list. Clears `state.cards` and `state.player.draw`,
/// then pushes one `Card` per spec with a fresh [`CardUid`].
/// Reserved for scenario-driven deck overrides (the legacy
/// `build_starter_state_with_deck` caller was retired by bead
/// `sts2-rs-na9oh`; future scenario plumbing will route through this
/// helper).
pub fn install_deck(state: &mut sts2_core::CombatState, deck: &[DeckCardSpec]) {
    use sts2_core::cards::Card;
    use sts2_core::CardUid;

    state.cards.clear();
    state.player.draw.clear();
    for (idx, entry) in deck.iter().enumerate() {
        let raw = u32::try_from(idx + 1).expect("deck size fits u32");
        let uid = CardUid::new(raw).expect("non-zero");
        let spec = entry.spec;
        let mut card = Card {
            uid,
            card_id: spec.id,
            cost: spec.cost,
            card_type: spec.card_type,
            target_type: spec.target_type,
            // Engine `Card.is_upgraded()` is a single bool today; an
            // `upgrade_level > 1` (e.g. Strike+3) collapses to
            // `true` until a future bead grows the engine. Fixture
            // round-trip preserves the requested level in the
            // emitted token regardless.
            upgraded: entry.upgrade_level >= 1,
            base_damage: spec.base_damage,
            base_block: spec.base_block,
            ..Card::pod_defaults()
        };
        if let Some(kind) = entry.enchantment {
            card.set_enchantment(kind, 1);
        }
        state.cards.push(card);
        state.player.draw.push(uid);
    }
}

/// dropping into the simulator loop.
pub fn format_summary(deck: &[DeckCardSpec]) -> String {
    let mut s = format!("deck ({} cards):", deck.len());
    for (i, e) in deck.iter().enumerate() {
        if i == 0 {
            s.push(' ');
        } else {
            s.push_str(", ");
        }
        s.push_str(short_name(e.spec));
        match e.upgrade_level {
            0 => {}
            1 => s.push('+'),
            n => {
                use std::fmt::Write;
                let _ = write!(s, "+{n}");
            }
        }
        if let Some(kind) = e.enchantment {
            use std::fmt::Write;
            let _ = write!(s, "@{}", enchantment_token(kind));
        }
    }
    s
}

/// Inverse of [`parse_enchantment_kind`]: emit the canonical
/// screaming-snake token for a given enchantment id. Used by
/// [`format_summary`] and by the `--print-scenario` round-trip.
#[must_use]
pub fn enchantment_token(id: EnchantmentId) -> &'static str {
    match id {
        EnchantmentId::RoyallyApproved => "ROYALLY_APPROVED",
        EnchantmentId::Steady => "STEADY",
        EnchantmentId::TezcatarasEmber => "TEZCATARAS_EMBER",
        EnchantmentId::SoulsPower => "SOULS_POWER",
        EnchantmentId::Instinct => "INSTINCT",
        EnchantmentId::Goopy => "GOOPY",
        EnchantmentId::Sharp => "SHARP",
        EnchantmentId::Momentum => "MOMENTUM",
        EnchantmentId::Corrupted => "CORRUPTED",
        EnchantmentId::Favored => "FAVORED",
        EnchantmentId::Vigorous => "VIGOROUS",
        EnchantmentId::Nimble => "NIMBLE",
        EnchantmentId::Spiral => "SPIRAL",
        EnchantmentId::Glam => "GLAM",
        EnchantmentId::Adroit => "ADROIT",
        EnchantmentId::Swift => "SWIFT",
        EnchantmentId::Sown => "SOWN",
        EnchantmentId::Imbued => "IMBUED",
        EnchantmentId::Slither => "SLITHER",
        EnchantmentId::SlumberingEssence => "SLUMBERING_ESSENCE",
        EnchantmentId::PerfectFit => "PERFECT_FIT",
        EnchantmentId::Clone => "CLONE",
        EnchantmentId::Inky => "INKY",
    }
}

/// Pretty short name for a [`CardSpec`]. Today returns the raw
/// `spec.name` (the canonical `STRIKE_IRONCLAD`-style string baked
/// into the spec); a future bead may strip class suffixes.
fn short_name(spec: CardSpec) -> &'static str {
    // The canonical `spec.name` is already a `'static str` baked into
    // the binary; we just return it. Stripping class suffixes would
    // require allocation, which the banner doesn't justify.
    spec.name
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn inline_strike_defend_bash_with_upgrade() {
        let deck = resolve_deck(
            &DeckInput::Inline("Strike,Strike,Defend,Bash+".to_string()),
            PlayerClass::Ironclad,
        )
        .unwrap();
        assert_eq!(deck.len(), 4);
        assert!(!deck[0].is_upgraded());
        assert!(!deck[1].is_upgraded());
        assert!(!deck[2].is_upgraded());
        assert!(deck[3].is_upgraded(), "Bash+ should be marked upgraded");
        assert_eq!(deck[3].spec.name, "BASH");
    }

    #[test]
    fn inline_unknown_card_returns_suggestions() {
        let err = resolve_deck(
            &DeckInput::Inline("Strike,UnknownCard".to_string()),
            PlayerClass::Ironclad,
        )
        .unwrap_err();
        match err {
            DeckError::UnknownCard { input, suggestions } => {
                assert_eq!(input, "UnknownCard");
                assert!(!suggestions.is_empty());
            }
            other => panic!("expected UnknownCard, got {other:?}"),
        }
    }

    #[test]
    fn preset_ironclad_basic_yields_10_cards() {
        let deck = resolve_deck(
            &DeckInput::Preset("ironclad-basic".into()),
            PlayerClass::Ironclad,
        )
        .unwrap();
        assert_eq!(deck.len(), 10);
        assert!(deck.iter().all(|e| !e.is_upgraded()));
    }

    #[test]
    fn preset_unknown_lists_available() {
        let err = resolve_deck(
            &DeckInput::Preset("does-not-exist".into()),
            PlayerClass::Ironclad,
        )
        .unwrap_err();
        match err {
            DeckError::UnknownPreset { available, .. } => {
                assert!(available.iter().any(|s| s == "ironclad-basic"));
            }
            other => panic!("expected UnknownPreset, got {other:?}"),
        }
    }

    #[test]
    fn file_lines_with_comments_and_upgrade() {
        let tmp = tempfile_path("deck_lines.txt");
        let body =
            b"# preamble comment\nStrike\nStrike  # trailing\nDefend\nBash+\n\n# blank above\n";
        std::fs::File::create(&tmp)
            .unwrap()
            .write_all(body)
            .unwrap();
        let deck = resolve_deck(&DeckInput::File(tmp.clone()), PlayerClass::Ironclad).unwrap();
        std::fs::remove_file(&tmp).ok();
        assert_eq!(deck.len(), 4);
        assert!(deck[3].is_upgraded());
        assert_eq!(deck[3].spec.name, "BASH");
    }

    #[test]
    fn file_toml_with_one_upgraded() {
        let tmp = tempfile_path("deck_toml.toml");
        let body = b"cards = [\n  \"Strike\",\n  \"Defend\",\n  \"Bash+\",\n]\n";
        std::fs::File::create(&tmp)
            .unwrap()
            .write_all(body)
            .unwrap();
        let deck = resolve_deck(&DeckInput::File(tmp.clone()), PlayerClass::Ironclad).unwrap();
        std::fs::remove_file(&tmp).ok();
        assert_eq!(deck.len(), 3);
        // Acceptance gate: exactly one upgraded card.
        assert_eq!(deck.iter().filter(|c| c.is_upgraded()).count(), 1);
        assert!(deck[2].is_upgraded());
        assert_eq!(deck[2].spec.id, sts2_core::cards::CardId::Bash);
    }

    #[test]
    fn rejects_empty_deck() {
        let err =
            resolve_deck(&DeckInput::Inline(String::new()), PlayerClass::Ironclad).unwrap_err();
        assert!(matches!(err, DeckError::Empty));
    }

    #[test]
    fn rejects_oversized_deck() {
        let huge = std::iter::repeat_n("Strike", MAX_DECK_SIZE + 1)
            .collect::<Vec<_>>()
            .join(",");
        let err = resolve_deck(&DeckInput::Inline(huge), PlayerClass::Ironclad).unwrap_err();
        assert!(matches!(err, DeckError::TooLarge { .. }));
    }

    #[test]
    fn format_summary_marks_upgrades() {
        let deck = resolve_deck(
            &DeckInput::Inline("Strike,Bash+".into()),
            PlayerClass::Ironclad,
        )
        .unwrap();
        let s = format_summary(&deck);
        assert!(s.contains("BASH+"), "summary={s}");
        assert!(s.contains("deck (2 cards)"), "summary={s}");
    }

    // ── F2 (bead `sts2-rs-xzeu`) preset coverage ──────────────────────

    #[test]
    fn preset_ironclad_strength_demo_contains_inflame_and_body_slam() {
        let deck = resolve_deck(
            &DeckInput::Preset("ironclad-strength-demo".into()),
            PlayerClass::Ironclad,
        )
        .unwrap();
        assert_eq!(deck.len(), 2);
        let ids: Vec<_> = deck.iter().map(|d| d.spec.id).collect();
        assert!(ids.contains(&sts2_core::cards::CardId::Inflame));
        assert!(ids.contains(&sts2_core::cards::CardId::BodySlam));
    }

    #[test]
    fn preset_ironclad_block_demo_has_four_defends_plus_two() {
        let deck = resolve_deck(
            &DeckInput::Preset("ironclad-block-demo".into()),
            PlayerClass::Ironclad,
        )
        .unwrap();
        assert_eq!(deck.len(), 6);
        let defend_count = deck
            .iter()
            .filter(|d| d.spec.id == sts2_core::cards::CardId::DefendIronclad)
            .count();
        assert_eq!(defend_count, 4, "expected 4 Defends, got {defend_count}");
        let ids: Vec<_> = deck.iter().map(|d| d.spec.id).collect();
        assert!(ids.contains(&sts2_core::cards::CardId::IronWave));
        assert!(ids.contains(&sts2_core::cards::CardId::ShrugItOff));
    }

    #[test]
    fn preset_silent_poison_demo_yields_ported_pair() {
        let deck = resolve_deck(
            &DeckInput::Preset("silent-poison-demo".into()),
            PlayerClass::Silent,
        )
        .unwrap();
        assert_eq!(deck.len(), 2);
        let ids: Vec<_> = deck.iter().map(|d| d.spec.id).collect();
        assert!(ids.contains(&sts2_core::cards::CardId::DeadlyPoison));
        assert!(ids.contains(&sts2_core::cards::CardId::PoisonedStab));
    }

    #[test]
    fn preset_silent_shiv_demo_has_two_blade_dances() {
        let deck = resolve_deck(
            &DeckInput::Preset("silent-shiv-demo".into()),
            PlayerClass::Silent,
        )
        .unwrap();
        assert_eq!(deck.len(), 3);
        let blade_count = deck
            .iter()
            .filter(|d| d.spec.id == sts2_core::cards::CardId::BladeDance)
            .count();
        assert_eq!(blade_count, 2, "expected 2 BladeDance, got {blade_count}");
        let ids: Vec<_> = deck.iter().map(|d| d.spec.id).collect();
        assert!(ids.contains(&sts2_core::cards::CardId::CloakAndDagger));
    }

    // ── kbi1: schema=2 deck-token extensions (`Foo+N`, `Foo@KIND`) ────

    #[test]
    fn token_upgrade_level_n_parses() {
        // `Strike+3` should yield upgrade_level=3 even though
        // engine `Card.upgraded: bool` collapses it to true.
        let deck = resolve_deck(
            &DeckInput::Inline("Strike,Strike+3".to_string()),
            PlayerClass::Ironclad,
        )
        .unwrap();
        assert_eq!(deck.len(), 2);
        assert_eq!(deck[0].upgrade_level, 0);
        assert_eq!(deck[1].upgrade_level, 3);
        assert!(deck[1].is_upgraded());
        assert!(deck[1].enchantment.is_none());
    }

    #[test]
    fn token_enchantment_plain() {
        let deck = resolve_deck(
            &DeckInput::Inline("Strike@GOOPY".to_string()),
            PlayerClass::Ironclad,
        )
        .unwrap();
        assert_eq!(deck.len(), 1);
        assert_eq!(deck[0].upgrade_level, 0);
        assert_eq!(deck[0].enchantment, Some(EnchantmentId::Goopy));
    }

    #[test]
    fn token_upgraded_and_enchanted_combine() {
        let deck = resolve_deck(
            &DeckInput::Inline("Bash+@SHARP".to_string()),
            PlayerClass::Ironclad,
        )
        .unwrap();
        assert_eq!(deck[0].upgrade_level, 1);
        assert_eq!(deck[0].enchantment, Some(EnchantmentId::Sharp));
    }

    #[test]
    fn token_upgrade_n_with_enchantment() {
        let deck = resolve_deck(
            &DeckInput::Inline("Strike+2@MOMENTUM".to_string()),
            PlayerClass::Ironclad,
        )
        .unwrap();
        assert_eq!(deck[0].upgrade_level, 2);
        assert_eq!(deck[0].enchantment, Some(EnchantmentId::Momentum));
    }

    #[test]
    fn token_enchantment_kind_screaming_snake_and_pascal() {
        // Multi-word screaming-snake — `ROYALLY_APPROVED`.
        let deck = resolve_deck(
            &DeckInput::Inline("Strike@ROYALLY_APPROVED".to_string()),
            PlayerClass::Ironclad,
        )
        .unwrap();
        assert_eq!(deck[0].enchantment, Some(EnchantmentId::RoyallyApproved));
        // PascalCase form is also accepted.
        let deck2 = resolve_deck(
            &DeckInput::Inline("Strike@RoyallyApproved".to_string()),
            PlayerClass::Ironclad,
        )
        .unwrap();
        assert_eq!(deck2[0].enchantment, Some(EnchantmentId::RoyallyApproved));
    }

    #[test]
    fn token_unknown_enchantment_kind_errors() {
        let err = resolve_deck(
            &DeckInput::Inline("Strike@NOPE".to_string()),
            PlayerClass::Ironclad,
        )
        .unwrap_err();
        match err {
            DeckError::UnknownCard { input, .. } => assert_eq!(input, "Strike@NOPE"),
            other => panic!("expected UnknownCard, got {other:?}"),
        }
    }

    #[test]
    fn token_plus_zero_rejected() {
        // `Foo+0` is meaningless — require either `Foo` or `Foo+N` (N>=1).
        let err = resolve_deck(
            &DeckInput::Inline("Strike+0".to_string()),
            PlayerClass::Ironclad,
        )
        .unwrap_err();
        assert!(matches!(err, DeckError::UnknownCard { .. }));
    }

    #[test]
    fn install_deck_applies_upgrade_and_enchantment() {
        use sts2_core::cards::EnchantmentId;
        let deck = resolve_deck(
            &DeckInput::Inline("Strike,Bash+@SHARP".to_string()),
            PlayerClass::Ironclad,
        )
        .unwrap();
        // Use any starter state; install_deck overwrites the card arena.
        let mut state = crate::sim::build_starter_state(
            1,
            crate::sim::EncounterKind::Nibbit,
            PlayerClass::Ironclad,
            None,
        );
        install_deck(&mut state, &deck);
        assert_eq!(state.cards.len(), 2);
        assert!(!state.cards[0].upgraded);
        assert!(state.cards[1].upgraded);
        assert!(state.cards[1].has_enchantment(EnchantmentId::Sharp));
        assert_eq!(state.cards[1].enchantment(EnchantmentId::Sharp), 1);
    }

    #[test]
    fn format_summary_renders_level_n_and_enchantment() {
        let deck = resolve_deck(
            &DeckInput::Inline("Strike,Strike+3,Bash+@GOOPY".to_string()),
            PlayerClass::Ironclad,
        )
        .unwrap();
        let s = format_summary(&deck);
        assert!(s.contains("+3"), "summary should contain +3: {s}");
        assert!(s.contains("@GOOPY"), "summary should contain @GOOPY: {s}");
        assert!(s.contains("BASH+@GOOPY"), "expected BASH+@GOOPY: {s}");
    }

    #[test]
    fn enchantment_token_roundtrips_for_all_ids() {
        // Every id round-trips through emit → parse cleanly.
        let ids = [
            EnchantmentId::RoyallyApproved,
            EnchantmentId::Steady,
            EnchantmentId::TezcatarasEmber,
            EnchantmentId::SoulsPower,
            EnchantmentId::Instinct,
            EnchantmentId::Goopy,
            EnchantmentId::Sharp,
            EnchantmentId::Momentum,
            EnchantmentId::Corrupted,
            EnchantmentId::Favored,
            EnchantmentId::Vigorous,
            EnchantmentId::Nimble,
            EnchantmentId::Spiral,
            EnchantmentId::Glam,
            EnchantmentId::Adroit,
            EnchantmentId::Swift,
            EnchantmentId::Sown,
            EnchantmentId::Imbued,
            EnchantmentId::Slither,
            EnchantmentId::SlumberingEssence,
            EnchantmentId::PerfectFit,
        ];
        for id in ids {
            let tok = enchantment_token(id);
            assert_eq!(parse_enchantment_kind(tok), Some(id), "token={tok}");
        }
    }

    fn tempfile_path(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        // Include thread / pid + nanos for parallel-test isolation.
        let pid = std::process::id();
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        p.push(format!("sts2-cli-{pid}-{ts}-{name}"));
        p
    }
}
