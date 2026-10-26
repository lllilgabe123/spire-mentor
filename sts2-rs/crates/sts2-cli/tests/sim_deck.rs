//! Deck-resolution grammar tests (`sts2_cli::sim::scenario::deck`).
//!
//! The inline `#[cfg(test)]` mod in `deck.rs` covers the inline/file/TOML
//! happy paths and two error shapes. This file covers what it doesn't:
//! every shipped preset's composition, the `DeckError` `Display` strings,
//! the `+N` / `@KIND` token grammar's rejection paths, the lenient-skip
//! filter on all three input paths, and the
//! `format_summary` → `--deck` round-trip (which is what makes
//! `--print-scenario` output re-loadable).
//!
//! These decks are CLI demo fixtures, not game content — no C# behavior
//! is encoded here. The only game-derived expectation is that a bare
//! `Strike`/`Defend` resolves to the class-specific card, which is
//! asserted through the resolver rather than restated.

use std::io::Write;
use std::path::PathBuf;

use hegel::generators as gs;
use hegel::TestCase;
use sts2_cli::sim::scenario::deck::{
    enchantment_token, format_summary, resolve_deck, DeckCardSpec, DeckError, DeckInput,
    MAX_DECK_SIZE,
};
use sts2_cli::sim::PlayerClass;
use sts2_core::cards::EnchantmentId;

/// Every `EnchantmentId` variant. The `index_of` match below is
/// exhaustive, so adding a variant to the enum breaks compilation here
/// until this list grows too.
const ALL_ENCHANTMENTS: &[EnchantmentId] = &[
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
    EnchantmentId::Clone,
    EnchantmentId::Inky,
];

fn index_of(id: EnchantmentId) -> usize {
    match id {
        EnchantmentId::RoyallyApproved => 0,
        EnchantmentId::Steady => 1,
        EnchantmentId::TezcatarasEmber => 2,
        EnchantmentId::SoulsPower => 3,
        EnchantmentId::Instinct => 4,
        EnchantmentId::Goopy => 5,
        EnchantmentId::Sharp => 6,
        EnchantmentId::Momentum => 7,
        EnchantmentId::Corrupted => 8,
        EnchantmentId::Favored => 9,
        EnchantmentId::Vigorous => 10,
        EnchantmentId::Nimble => 11,
        EnchantmentId::Spiral => 12,
        EnchantmentId::Glam => 13,
        EnchantmentId::Adroit => 14,
        EnchantmentId::Swift => 15,
        EnchantmentId::Sown => 16,
        EnchantmentId::Imbued => 17,
        EnchantmentId::Slither => 18,
        EnchantmentId::SlumberingEssence => 19,
        EnchantmentId::PerfectFit => 20,
        EnchantmentId::Clone => 21,
        EnchantmentId::Inky => 22,
    }
}

const PRESET_NAMES: &[&str] = &[
    "ironclad-basic",
    "silent-basic",
    "ironclad-attack-demo",
    "silent-stealth-demo",
    "ironclad-strength-demo",
    "ironclad-block-demo",
    "silent-poison-demo",
    "silent-shiv-demo",
];

const ALL_CLASSES: &[PlayerClass] = &[
    PlayerClass::Ironclad,
    PlayerClass::Silent,
    PlayerClass::Defect,
    PlayerClass::Regent,
    PlayerClass::Necrobinder,
];

fn tmp_path(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    let pid = std::process::id();
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    p.push(format!("sts2-deck-{pid}-{ts}-{name}"));
    p
}

fn write_tmp(name: &str, body: &str) -> PathBuf {
    let p = tmp_path(name);
    std::fs::File::create(&p)
        .unwrap()
        .write_all(body.as_bytes())
        .unwrap();
    p
}

fn inline(s: &str) -> Result<Vec<DeckCardSpec>, DeckError> {
    resolve_deck(&DeckInput::Inline(s.to_string()), PlayerClass::Ironclad)
}

fn names(deck: &[DeckCardSpec]) -> Vec<&'static str> {
    deck.iter().map(|e| e.spec.name).collect()
}

// --- presets -------------------------------------------------------------

#[test]
fn every_preset_resolves_with_the_documented_composition() {
    let expected: &[(&str, usize, usize)] = &[
        // (preset, card count, upgraded count)
        ("ironclad-basic", 10, 0),
        ("silent-basic", 12, 0),
        ("ironclad-attack-demo", 10, 1),
        ("silent-stealth-demo", 7, 1),
        ("ironclad-strength-demo", 2, 0),
        ("ironclad-block-demo", 6, 0),
        ("silent-poison-demo", 2, 0),
        ("silent-shiv-demo", 3, 0),
    ];
    for (name, count, upgraded) in expected {
        let deck = resolve_deck(&DeckInput::Preset((*name).into()), PlayerClass::Ironclad)
            .unwrap_or_else(|e| panic!("preset '{name}' failed: {e}"));
        assert_eq!(deck.len(), *count, "preset '{name}' card count");
        assert_eq!(
            deck.iter().filter(|e| e.is_upgraded()).count(),
            *upgraded,
            "preset '{name}' upgraded count"
        );
        assert!(
            deck.iter().all(|e| e.enchantment.is_none()),
            "preset '{name}' should ship no enchantments"
        );
    }
}

#[test]
fn preset_decks_are_class_independent() {
    // Each preset hardcodes its own class hint (the `class` argument is
    // unused), so the same preset name yields the same deck for every
    // player class — including Regent/Necrobinder, whose deck hint falls
    // back to Ironclad.
    for name in PRESET_NAMES {
        let baseline =
            resolve_deck(&DeckInput::Preset((*name).into()), PlayerClass::Ironclad).unwrap();
        for class in ALL_CLASSES {
            let deck = resolve_deck(&DeckInput::Preset((*name).into()), *class).unwrap();
            assert_eq!(deck, baseline, "preset '{name}' differs for {class:?}");
        }
    }
}

#[test]
fn preset_names_are_case_insensitive() {
    let lower = resolve_deck(
        &DeckInput::Preset("silent-shiv-demo".into()),
        PlayerClass::Silent,
    )
    .unwrap();
    let upper = resolve_deck(
        &DeckInput::Preset("SILENT-SHIV-DEMO".into()),
        PlayerClass::Silent,
    )
    .unwrap();
    assert_eq!(lower, upper);
    assert_eq!(
        names(&upper),
        ["BLADE_DANCE", "BLADE_DANCE", "CLOAK_AND_DAGGER"]
    );
}

#[test]
fn preset_class_specific_strikes_differ_between_ironclad_and_silent() {
    let ic = resolve_deck(
        &DeckInput::Preset("ironclad-basic".into()),
        PlayerClass::Ironclad,
    )
    .unwrap();
    let silent = resolve_deck(
        &DeckInput::Preset("silent-basic".into()),
        PlayerClass::Silent,
    )
    .unwrap();
    assert_ne!(
        ic[0].spec.id, silent[0].spec.id,
        "bare `Strike` must disambiguate per class"
    );
}

// --- class hint ----------------------------------------------------------

#[test]
fn regent_and_necrobinder_fall_back_to_the_ironclad_card_pool() {
    // No deck-eligible pool exists for these two yet; the resolver falls
    // back to the Ironclad hint rather than failing.
    let ic = resolve_deck(
        &DeckInput::Inline("Strike,Defend".into()),
        PlayerClass::Ironclad,
    )
    .unwrap();
    for class in [PlayerClass::Regent, PlayerClass::Necrobinder] {
        let deck = resolve_deck(&DeckInput::Inline("Strike,Defend".into()), class).unwrap();
        assert_eq!(names(&deck), names(&ic), "{class:?} should mirror Ironclad");
    }
}

#[test]
fn silent_hint_picks_the_silent_strike() {
    let ic = resolve_deck(&DeckInput::Inline("Strike".into()), PlayerClass::Ironclad).unwrap();
    let silent = resolve_deck(&DeckInput::Inline("Strike".into()), PlayerClass::Silent).unwrap();
    assert_ne!(ic[0].spec.id, silent[0].spec.id);
}

// --- DeckError Display ---------------------------------------------------

#[test]
fn deck_error_display_strings() {
    let unknown_preset = DeckError::UnknownPreset {
        name: "nope".into(),
        available: vec!["ironclad-basic".into(), "silent-basic".into()],
    };
    assert_eq!(
        unknown_preset.to_string(),
        "unknown deck preset 'nope'. available: ironclad-basic, silent-basic"
    );

    let no_suggestions = DeckError::UnknownCard {
        input: "Zorp".into(),
        suggestions: vec![],
    };
    assert_eq!(no_suggestions.to_string(), "unknown card 'Zorp'");

    let with_suggestions = DeckError::UnknownCard {
        input: "Strik".into(),
        suggestions: vec!["Strike".into(), "StrikeSilent".into()],
    };
    assert_eq!(
        with_suggestions.to_string(),
        "unknown card 'Strik'. did you mean: Strike, StrikeSilent?"
    );

    let io = DeckError::Io {
        path: PathBuf::from("/no/such/deck.txt"),
        err: std::io::Error::new(std::io::ErrorKind::NotFound, "boom"),
    };
    assert_eq!(io.to_string(), "could not read '/no/such/deck.txt': boom");

    let toml = DeckError::TomlParse {
        path: PathBuf::from("/tmp/deck.toml"),
        err: "expected `=`".into(),
    };
    assert_eq!(
        toml.to_string(),
        "could not parse TOML deck '/tmp/deck.toml': expected `=`"
    );

    assert_eq!(DeckError::Empty.to_string(), "deck is empty");
    assert_eq!(
        DeckError::TooLarge {
            size: 300,
            cap: MAX_DECK_SIZE
        }
        .to_string(),
        "deck size 300 exceeds maximum of 256 cards"
    );
}

// --- size validation -----------------------------------------------------

#[test]
fn empty_and_whitespace_only_inline_decks_are_rejected() {
    assert!(matches!(inline("").unwrap_err(), DeckError::Empty));
    assert!(matches!(inline("  ,  , ").unwrap_err(), DeckError::Empty));
    // A deck consisting only of lenient-skip tokens also lands on Empty.
    assert!(matches!(
        inline("NOT_YET,NOT_YET").unwrap_err(),
        DeckError::Empty
    ));
}

#[test]
fn deck_size_cap_is_inclusive() {
    let at_cap = std::iter::repeat_n("Strike", MAX_DECK_SIZE)
        .collect::<Vec<_>>()
        .join(",");
    assert_eq!(inline(&at_cap).unwrap().len(), MAX_DECK_SIZE);

    let over_cap = std::iter::repeat_n("Strike", MAX_DECK_SIZE + 1)
        .collect::<Vec<_>>()
        .join(",");
    match inline(&over_cap).unwrap_err() {
        DeckError::TooLarge { size, cap } => {
            assert_eq!(size, MAX_DECK_SIZE + 1);
            assert_eq!(cap, MAX_DECK_SIZE);
        }
        other => panic!("expected TooLarge, got {other:?}"),
    }
}

// --- token grammar -------------------------------------------------------

#[test]
fn upgrade_suffix_grammar() {
    assert_eq!(inline("Strike").unwrap()[0].upgrade_level, 0);
    assert_eq!(inline("Strike+").unwrap()[0].upgrade_level, 1);
    assert_eq!(inline("Strike+1").unwrap()[0].upgrade_level, 1);
    assert_eq!(inline("Strike+3").unwrap()[0].upgrade_level, 3);
    // Whitespace around the `+` is tolerated on the name side.
    assert_eq!(inline("Strike +2").unwrap()[0].upgrade_level, 2);
}

#[test]
fn upgrade_level_zero_is_rejected_with_a_hint() {
    match inline("Strike+0").unwrap_err() {
        DeckError::UnknownCard { input, suggestions } => {
            assert_eq!(input, "Strike+0");
            assert!(
                suggestions.iter().any(|s| s.contains("must be >= 1")),
                "suggestions: {suggestions:?}"
            );
        }
        other => panic!("expected UnknownCard, got {other:?}"),
    }
}

#[test]
fn upgrade_level_out_of_u32_range_is_rejected() {
    let token = "Strike+99999999999999999999";
    match inline(token).unwrap_err() {
        DeckError::UnknownCard { input, suggestions } => {
            assert_eq!(input, token);
            assert!(
                suggestions.iter().any(|s| s.contains("out of range")),
                "suggestions: {suggestions:?}"
            );
        }
        other => panic!("expected UnknownCard, got {other:?}"),
    }
}

#[test]
fn non_numeric_upgrade_suffix_is_treated_as_part_of_the_name() {
    // `Strike+abc` doesn't split into name + level, so the whole token
    // goes to the card registry and fails there.
    match inline("Strike+abc").unwrap_err() {
        DeckError::UnknownCard { input, .. } => assert_eq!(input, "Strike+abc"),
        other => panic!("expected UnknownCard, got {other:?}"),
    }
}

#[test]
fn enchantment_suffix_grammar() {
    let goopy = inline("Strike@GOOPY").unwrap();
    assert_eq!(goopy[0].enchantment, Some(EnchantmentId::Goopy));
    assert_eq!(goopy[0].upgrade_level, 0);

    // PascalCase and spaced forms normalize to the same id.
    assert_eq!(
        inline("Strike@Goopy").unwrap()[0].enchantment,
        Some(EnchantmentId::Goopy)
    );
    assert_eq!(
        inline("Strike @ GOOPY").unwrap()[0].enchantment,
        Some(EnchantmentId::Goopy)
    );

    // Upgrade + enchantment compose, in that order.
    let sharp = inline("Bash+@SHARP").unwrap();
    assert_eq!(sharp[0].upgrade_level, 1);
    assert_eq!(sharp[0].enchantment, Some(EnchantmentId::Sharp));

    let momentum = inline("Strike+2@MOMENTUM").unwrap();
    assert_eq!(momentum[0].upgrade_level, 2);
    assert_eq!(momentum[0].enchantment, Some(EnchantmentId::Momentum));

    // Screaming-snake multi-word forms.
    assert_eq!(
        inline("Strike@SLUMBERING_ESSENCE").unwrap()[0].enchantment,
        Some(EnchantmentId::SlumberingEssence)
    );
}

#[test]
fn unknown_enchantment_kind_is_rejected() {
    match inline("Strike@NOPE").unwrap_err() {
        DeckError::UnknownCard { input, suggestions } => {
            assert_eq!(input, "Strike@NOPE", "the whole token is echoed back");
            assert!(
                suggestions
                    .iter()
                    .any(|s| s.contains("unknown enchantment kind 'NOPE'")),
                "suggestions: {suggestions:?}"
            );
        }
        other => panic!("expected UnknownCard, got {other:?}"),
    }
}

#[test]
fn lenient_skip_tokens_are_dropped_on_every_input_path() {
    // Inline.
    let deck = inline("Strike,NOT_YET,Bash").unwrap();
    assert_eq!(deck.len(), 2);
    assert_eq!(names(&deck)[1], "BASH");

    // Newline file.
    let lines = write_tmp("skip_lines.txt", "Strike\nNOT_YET\nBash\n");
    let deck = resolve_deck(&DeckInput::File(lines.clone()), PlayerClass::Ironclad).unwrap();
    std::fs::remove_file(&lines).ok();
    assert_eq!(deck.len(), 2);

    // TOML file.
    let toml = write_tmp(
        "skip_deck.toml",
        "cards = [\"Strike\", \"NOT_YET\", \"\", \"Bash\"]\n",
    );
    let deck = resolve_deck(&DeckInput::File(toml.clone()), PlayerClass::Ironclad).unwrap();
    std::fs::remove_file(&toml).ok();
    assert_eq!(deck.len(), 2);
}

// --- file input ----------------------------------------------------------

#[test]
fn missing_deck_file_reports_io_error_with_the_path() {
    let path = tmp_path("definitely_absent.txt");
    match resolve_deck(&DeckInput::File(path.clone()), PlayerClass::Ironclad).unwrap_err() {
        DeckError::Io { path: got, .. } => assert_eq!(got, path),
        other => panic!("expected Io, got {other:?}"),
    }
}

#[test]
fn malformed_toml_deck_reports_a_parse_error() {
    let path = write_tmp("broken.toml", "cards = [\"Strike\"\n");
    let err = resolve_deck(&DeckInput::File(path.clone()), PlayerClass::Ironclad).unwrap_err();
    std::fs::remove_file(&path).ok();
    match err {
        DeckError::TomlParse { path: got, err } => {
            assert_eq!(got, path);
            assert!(!err.is_empty());
        }
        other => panic!("expected TomlParse, got {other:?}"),
    }
}

#[test]
fn toml_deck_table_form_is_accepted() {
    // `[deck] cards = [...]` is the second accepted TOML shape; it is
    // only consulted when the top-level `cards` array is absent/empty.
    let path = write_tmp("table.toml", "[deck]\ncards = [\"Strike\", \"Bash+\"]\n");
    let deck = resolve_deck(&DeckInput::File(path.clone()), PlayerClass::Ironclad).unwrap();
    std::fs::remove_file(&path).ok();
    assert_eq!(deck.len(), 2);
    assert!(deck[1].is_upgraded());
}

#[test]
fn toml_deck_without_any_cards_key_is_empty() {
    let path = write_tmp("nocards.toml", "unrelated = 3\n");
    let err = resolve_deck(&DeckInput::File(path.clone()), PlayerClass::Ironclad).unwrap_err();
    std::fs::remove_file(&path).ok();
    assert!(matches!(err, DeckError::Empty));
}

#[test]
fn toml_is_sniffed_from_content_when_the_extension_lies() {
    // A `.txt` whose body starts with `cards` takes the TOML branch.
    let path = write_tmp("sniffed.txt", "cards = [\"Strike\", \"Defend\"]\n");
    let deck = resolve_deck(&DeckInput::File(path.clone()), PlayerClass::Ironclad).unwrap();
    std::fs::remove_file(&path).ok();
    assert_eq!(deck.len(), 2);
}

#[test]
fn line_file_accepts_comma_separated_lines_and_inline_comments() {
    let path = write_tmp(
        "commas.txt",
        "# header\nStrike, Strike ,Defend # trailing comment\n\nBash+\n",
    );
    let deck = resolve_deck(&DeckInput::File(path.clone()), PlayerClass::Ironclad).unwrap();
    std::fs::remove_file(&path).ok();
    assert_eq!(deck.len(), 4);
    assert!(deck[3].is_upgraded());
}

// --- format_summary round-trip ------------------------------------------

/// Strip the `deck (N cards): ` banner prefix, leaving the
/// comma-separated token list that `--deck` accepts.
fn summary_tokens(summary: &str) -> &str {
    summary
        .split_once(": ")
        .expect("summary has a token list")
        .1
}

#[test]
fn format_summary_banner_shape() {
    let deck = inline("Strike,Bash+,Defend").unwrap();
    let summary = format_summary(&deck);
    assert!(summary.starts_with("deck (3 cards): "), "{summary}");
    assert_eq!(summary.matches(',').count(), 2, "{summary}");
    assert!(summary.contains("BASH+"), "{summary}");
}

#[test]
fn format_summary_round_trips_through_the_inline_parser() {
    let deck = inline("Strike,Bash+,Strike+3@GOOPY,Defend@SHARP").unwrap();
    let summary = format_summary(&deck);
    let reparsed = inline(summary_tokens(&summary)).unwrap();
    assert_eq!(reparsed, deck, "summary was: {summary}");
}

#[test]
fn every_enchantment_token_round_trips() {
    // `enchantment_token` is the inverse of the `@KIND` parser; a hole in
    // either table would silently drop the enchantment on
    // `--print-scenario` output.
    let base = inline("Strike").unwrap()[0];
    for id in ALL_ENCHANTMENTS {
        assert_eq!(
            ALL_ENCHANTMENTS[index_of(*id)],
            *id,
            "ALL_ENCHANTMENTS is out of sync with index_of"
        );
        let deck = vec![DeckCardSpec {
            enchantment: Some(*id),
            ..base
        }];
        let summary = format_summary(&deck);
        assert!(
            summary.ends_with(&format!("@{}", enchantment_token(*id))),
            "summary: {summary}"
        );
        let reparsed = inline(summary_tokens(&summary)).unwrap();
        assert_eq!(reparsed, deck, "round-trip failed for {id:?}");
    }
}

/// Any (card, upgrade level, enchantment) triple survives a
/// `format_summary` → `--deck` round-trip. This is the property that
/// makes `--print-scenario` output re-loadable.
#[hegel::test(test_cases = 200)]
fn deck_tokens_round_trip_through_format_summary(tc: TestCase) {
    const CARDS: &[&str] = &[
        "Strike",
        "Defend",
        "Bash",
        "Anger",
        "IronWave",
        "TwinStrike",
    ];
    let n = tc.draw(gs::integers::<u32>().min_value(1).max_value(6)) as usize;
    let mut deck = Vec::with_capacity(n);
    for _ in 0..n {
        let name = tc.draw(gs::sampled_from(CARDS));
        let spec = inline(name).unwrap()[0].spec;
        let upgrade_level = tc.draw(gs::integers::<u32>().min_value(0).max_value(9));
        let ench_idx = tc.draw(
            gs::integers::<u32>()
                .min_value(0)
                .max_value(ALL_ENCHANTMENTS.len() as u32),
        ) as usize;
        deck.push(DeckCardSpec {
            spec,
            upgrade_level,
            enchantment: ALL_ENCHANTMENTS.get(ench_idx).copied(),
        });
    }
    let summary = format_summary(&deck);
    let reparsed = resolve_deck(
        &DeckInput::Inline(summary_tokens(&summary).to_string()),
        PlayerClass::Ironclad,
    )
    .unwrap_or_else(|e| panic!("'{summary}' did not re-parse: {e}"));
    assert_eq!(reparsed, deck, "summary was: {summary}");
}
