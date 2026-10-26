//! `synergy_score` — point-query bin for the value net.
//!
//! Reads newline-delimited JSON states from stdin, prints one `V(state)` per
//! line on stdout. Loads the net ONCE and batches over all input lines so
//! callers (e.g. the Python synergy-catalog driver) can pipe thousands of
//! states without process-spawn overhead.
//!
//! Input (one JSON object per line):
//! ```json
//! {"deck":["BASH","IRON_WAVE+1"],"relics":["BurningBlood"],"hp":70,"max_hp":80,"gold":99,"ascension":0,"act_index":2,"act_floor":3}
//! ```
//! Fields: `deck` (required), `relics` (optional, default `[BurningBlood]`),
//! `hp`/`max_hp`/`gold`/`ascension`/`act_index`/`act_floor` (all optional
//! with sensible defaults). Unknown card/relic tokens cause a descriptive
//! error on stderr and a non-zero exit.
//!
//! Two backends, selected by `--net` (default `gbm` → historical behaviour):
//! - `--net gbm`: the legacy Ironclad-only GBM value net (env `STS2_RUNAI_VALUE`,
//!   default `<workspace>/data/runai/nets/value_human_a0.json`).
//! - `--net ppo --character <name>`: the deployed per-character PPO transformer's
//!   value head (champion ONNX via `STS2_RUNAI_PPO` / `onnx_model_path_for`).
//!   The value head mean-pools all obs tokens including map nodes, so a
//!   real act-1 map (`synergy_score_fixed_map.json`) is spliced in and held
//!   constant — a `map: None` obs is badly out-of-distribution (deck size
//!   dominates, card signal inverts). The probe is used for *relative* second
//!   differences, where the shared fixed map cancels.
//!
//! `--character` (default `ironclad`) selects per-character base HP / starter
//! relic (and the PPO champion). Usage: `synergy_score [--net ppo]
//! [--character silent] < states.jsonl`

use std::collections::BTreeSet;
use std::io::{self, BufRead};
use std::path::Path;
use std::process;

use serde::Deserialize;

use sts2_content::names::{card_id_from_str, relic_id_from_str};
use sts2_core::characters::character_starter;
use sts2_core::potions::CharacterId;
use sts2_core::relics::RelicId;
use sts2_core::run_manager::Phase;
use sts2_runai::eval::ranker_features::ContextFeatures;
use sts2_runai::eval::run_value::ValueNet;
use sts2_runai::ppo::encoder::featurize_with;
use sts2_runai::ppo::features::Features;
use sts2_runai::ppo::net::{onnx_model_path_for, PpoNet};
use sts2_runai::ppo::obs::{ActionDesc, FullMapObs, PpoObs};
use sts2_runai::view::{DeckCard, Offers, VisibleRunView};

const DEFAULT_NET: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../data/runai/nets/value_human_a0.json"
);

/// Base potion-slot count — character-invariant at base ascension (see
/// `runstate.rs` `max_potion_slots` default).
const BASE_POTION_SLOTS: i32 = 3;
const DEFAULT_GOLD: i32 = 99;
const DEFAULT_ACT_INDEX: u8 = 1;
const DEFAULT_ACT_FLOOR: i32 = 1;

/// Deserialised shape of one input line.
#[derive(Deserialize)]
struct StateInput {
    deck: Vec<String>,
    #[serde(default)]
    relics: Option<Vec<String>>,
    #[serde(default)]
    hp: Option<i32>,
    #[serde(default)]
    max_hp: Option<i32>,
    #[serde(default)]
    gold: Option<i32>,
    #[serde(default)]
    ascension: Option<u8>,
    #[serde(default)]
    act_index: Option<u8>,
    #[serde(default)]
    act_floor: Option<i32>,
}

/// Which value net backs the probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NetMode {
    /// Legacy Ironclad-only GBM value net (default; the historical behaviour).
    Gbm,
    /// Per-character deployed PPO transformer's value head.
    Ppo,
}

/// Parsed command line.
#[derive(Debug, Clone, Copy)]
struct Cli {
    net: NetMode,
    character: CharacterId,
}

/// Map a lowercase character name to its [`CharacterId`]. `None` for unknowns.
fn parse_character(name: &str) -> Option<CharacterId> {
    match name.to_ascii_lowercase().as_str() {
        "ironclad" => Some(CharacterId::Ironclad),
        "silent" => Some(CharacterId::Silent),
        "defect" => Some(CharacterId::Defect),
        "regent" => Some(CharacterId::Regent),
        "necrobinder" => Some(CharacterId::Necrobinder),
        _ => None,
    }
}

/// Parse `--net {gbm|ppo}` and `--character <name>` from *args* (the first item
/// is the program name and is skipped). Defaults: GBM net, Ironclad — so a
/// no-arg invocation reproduces the historical behaviour exactly.
fn parse_cli(mut args: impl Iterator<Item = String>) -> Result<Cli, String> {
    let _prog = args.next();
    let mut net = NetMode::Gbm;
    let mut character = CharacterId::Ironclad;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--net" => {
                let val = args.next().ok_or("--net requires a value (gbm|ppo)")?;
                net = match val.as_str() {
                    "gbm" => NetMode::Gbm,
                    "ppo" => NetMode::Ppo,
                    other => return Err(format!("unknown --net '{other}' (want gbm|ppo)")),
                };
            }
            "--character" => {
                let val = args.next().ok_or("--character requires a value")?;
                character =
                    parse_character(&val).ok_or_else(|| format!("unknown --character '{val}'"))?;
            }
            other => return Err(format!("unknown argument '{other}'")),
        }
    }
    Ok(Cli { net, character })
}

/// Strip a `CARD.`/`RELIC.`/`POTION.` namespace prefix.
fn strip_ns(s: &str) -> &str {
    s.split_once('.').map_or(s, |(_, rest)| rest)
}

/// Parse `"BASH"` / `"IRON_WAVE+1"` / `"CARD.IRON_WAVE+1"` into a [`DeckCard`].
/// Returns `None` if the card token is unknown.
fn parse_card(token: &str) -> Option<DeckCard> {
    let bare = strip_ns(token);
    let (base, upgraded) = bare
        .split_once('+')
        .map_or((bare, false), |(b, _)| (b, true));
    card_id_from_str(base).map(|card_id| DeckCard { card_id, upgraded })
}

/// Parse `"BurningBlood"` / `"RELIC.BurningBlood"` into a [`RelicId`].
/// Returns `None` if unknown.
fn parse_relic(token: &str) -> Option<RelicId> {
    relic_id_from_str(strip_ns(token))
}

/// Build a [`VisibleRunView`] from a parsed [`StateInput`] for *character*, or
/// return an error string describing the first unknown token. Base HP and the
/// default starter relic come from [`character_starter`], so all five base
/// characters work; only the potion-slot count is character-invariant.
fn build_view(s: &StateInput, character: CharacterId) -> Result<VisibleRunView, String> {
    let mut deck = Vec::with_capacity(s.deck.len());
    for token in &s.deck {
        match parse_card(token) {
            Some(dc) => deck.push(dc),
            None => return Err(format!("unknown card token: '{token}'")),
        }
    }

    let starter = character_starter(character);

    let relics: Vec<RelicId> = match &s.relics {
        Some(tokens) => {
            let mut out = Vec::with_capacity(tokens.len());
            for token in tokens {
                match parse_relic(token) {
                    Some(rid) => out.push(rid),
                    None => return Err(format!("unknown relic token: '{token}'")),
                }
            }
            out
        }
        None => vec![starter.starter_relic],
    };

    let max_hp = s.max_hp.unwrap_or(starter.hp);
    let hp = s.hp.unwrap_or(starter.hp).clamp(0, max_hp);
    let gold = s.gold.unwrap_or(DEFAULT_GOLD).clamp(0, 9999);
    let ascension = s.ascension.unwrap_or(0);
    let act_index = s.act_index.unwrap_or(DEFAULT_ACT_INDEX);
    let act_floor = s.act_floor.unwrap_or(DEFAULT_ACT_FLOOR);

    Ok(VisibleRunView {
        character,
        ascension,
        act_index,
        act_floor,
        hp,
        max_hp,
        gold,
        deck,
        relics,
        potions: Vec::new(),
        max_potion_slots: BASE_POTION_SLOTS,
        boss: None,
        map: None,
        seen_events: BTreeSet::new(),
        current_event: None,
        phase: Phase::MapChoice,
        offers: Offers::default(),
    })
}

/// Score a single state with the GBM value net. Testable without going through
/// stdin. The GBM net is Ironclad-trained; *character* only affects base-stat
/// defaults in [`build_view`] and is `Ironclad` on the default path.
fn score(net: &ValueNet, s: &StateInput, character: CharacterId) -> Result<f64, String> {
    let view = build_view(s, character)?;
    let ctx = ContextFeatures::from_view(&view);
    Ok(net.value(&ctx))
}

fn load_net() -> ValueNet {
    let path_str = std::env::var("STS2_RUNAI_VALUE").unwrap_or_else(|_| DEFAULT_NET.to_string());
    let path = Path::new(&path_str);
    match ValueNet::from_weights_path(path) {
        Ok(net) => net,
        Err(e) => {
            eprintln!("synergy_score: failed to load value net from {path_str}: {e}");
            process::exit(1);
        }
    }
}

/// Stream stdin lines through *score_line*, printing one `{:.6}` value per
/// non-blank input line. Exits non-zero on the first error (parse or scoring),
/// mirroring the strict one-float-per-line contract the Python driver expects.
fn run_stdin_loop(mut score_line: impl FnMut(&StateInput) -> Result<f64, String>) {
    let stdin = io::stdin();
    let mut had_error = false;

    for (lineno, line) in stdin.lock().lines().enumerate() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                eprintln!("synergy_score: stdin read error at line {lineno}: {e}");
                had_error = true;
                break;
            }
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let input: StateInput = match serde_json::from_str(trimmed) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("synergy_score: JSON parse error at line {lineno}: {e}");
                had_error = true;
                break;
            }
        };
        match score_line(&input) {
            Ok(v) => println!("{v:.6}"),
            Err(e) => {
                eprintln!("synergy_score: error at line {lineno}: {e}");
                had_error = true;
                break;
            }
        }
    }

    if had_error {
        process::exit(1);
    }
}

/// Default (legacy) path: the Ironclad-only GBM value net.
fn run_gbm(character: CharacterId) {
    let net = load_net();
    run_stdin_loop(|input| score(&net, input, character));
}

/// A fixed, real act-1 map (extracted from the committed encoder-parity
/// fixture) held constant across every probe state. The PPO value head
/// mean-pools all tokens including ~16 map-node tokens; a `map: None` obs is
/// badly out-of-distribution (deck size dominates, card signal inverts), so a
/// fixed real map is spliced in. Held constant, it cancels in the
/// second-difference protocol the probe uses.
const FIXED_MAP_JSON: &str = include_str!("synergy_score_fixed_map.json");

/// Deserialized shape of [`FIXED_MAP_JSON`].
#[derive(Deserialize)]
struct FixedMap {
    map: Option<FullMapObs>,
    map_coord: Option<(i64, i64)>,
}

fn load_fixed_map() -> FixedMap {
    serde_json::from_str(FIXED_MAP_JSON).expect("committed fixed map must parse")
}

/// Build a value-only [`PpoObs`] from a static deck/context view, mirroring
/// `ppo::policy::build_obs`'s token formatting EXACTLY (deck tokens are the
/// `CardId` debug name + upgraded flag; relics are `RELIC.<python_name>`) so the
/// encoder vocab resolves identically to the deployed agent.
///
/// The map is spliced in from *fixed_map* (a real act-1 map, constant across
/// states — see [`FIXED_MAP_JSON`]); every surface-gated offer field is `None`.
/// `phase`/`surface` are neutral placeholders — the Rust encoder derives the
/// state-token surface from the action list (`fine_surface`), not these fields.
/// This is used only for *relative* second differences, where the shared map
/// and action cancel.
fn build_ppo_obs(view: &VisibleRunView, fixed_map: &FixedMap) -> PpoObs {
    let deck: Vec<(String, bool)> = view
        .deck
        .iter()
        .map(|d| (format!("{:?}", d.card_id), d.upgraded))
        .collect();
    let relics: Vec<String> = view
        .relics
        .iter()
        .map(|r| format!("RELIC.{}", r.to_python_name()))
        .collect();
    let deck_size = deck.len() as i64;
    let num_upgraded = deck.iter().filter(|(_, up)| *up).count() as i64;
    let num_relics = relics.len() as i64;

    PpoObs {
        phase: "map_choice".to_string(),
        surface: None,
        seed: 0,
        ascension: i64::from(view.ascension),
        deck,
        deck_enchantments: None,
        relics,
        potions: Vec::new(),
        map: fixed_map.map.clone(),
        map_coord: fixed_map.map_coord,
        total_floor: i64::from(view.act_floor),
        act_index: i64::from(view.act_index),
        act_floor: i64::from(view.act_floor),
        hp: i64::from(view.hp),
        max_hp: i64::from(view.max_hp),
        gold: i64::from(view.gold),
        deck_size,
        num_upgraded,
        num_relics,
        num_potions: 0,
        card_reward_option_ids: None,
        card_reward_option_upgraded: None,
        card_reward_option_enchantments: None,
        card_reward_bundles: None,
        relic_reward_ids: None,
        boss_relic_options: None,
        smith_options: None,
        smith_option_enchantments: None,
        card_removal_options: None,
        card_grant_options: None,
        card_removal_option_enchantments: None,
        card_grant_option_enchantments: None,
        transform_select_deck_indices: None,
        enchant_select_deck_indices: None,
        pending_enchantment: None,
        upgrade_select_deck_indices: None,
        shop: None,
        pending_potion_swap: None,
        event_name: None,
        event_options: None,
        rest_option_labels: None,
    }
}

/// Score one state with the PPO value head. A single neutral `ActionDesc::
/// Unknown` slot is used for every state so the value head's obs+action pooling
/// contributes a constant action term across states (see plan Global
/// Constraints); the deck drives the differences via the obs tokens.
fn score_ppo(net: &PpoNet, view: &VisibleRunView, fixed_map: &FixedMap) -> Result<f64, String> {
    let obs = build_ppo_obs(view, fixed_map);
    // A dev tool scoring whatever net it was pointed at, so its schema comes
    // from the environment rather than from a champion manifest entry — the
    // same lane `net::features_for` reserves for an explicit path.
    let feat = featurize_with(&obs, &[ActionDesc::Unknown], Features::active());
    let (_probs, value) = net.infer(&feat)?;
    Ok(f64::from(value))
}

/// PPO value-head probe path: load the per-character champion ONNX once, then
/// score every stdin state. Aborts loudly (never silently falls back to the GBM
/// net) if no champion is resolvable or the graph fails to load.
fn run_ppo(character: CharacterId) {
    let Some(path) = onnx_model_path_for(character) else {
        eprintln!(
            "synergy_score: no PPO champion for {character:?} \
             (set STS2_RUNAI_PPO or add a committed champion)"
        );
        process::exit(1);
    };
    let net = match PpoNet::load(&path) {
        Ok(n) => n,
        Err(e) => {
            eprintln!(
                "synergy_score: failed to load PPO net from {}: {e}",
                path.display()
            );
            process::exit(1);
        }
    };
    let fixed_map = load_fixed_map();
    run_stdin_loop(|input| score_ppo(&net, &build_view(input, character)?, &fixed_map));
}

fn main() {
    let cli = match parse_cli(std::env::args()) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("synergy_score: {e}");
            process::exit(2);
        }
    };
    match cli.net {
        NetMode::Gbm => run_gbm(cli.character),
        NetMode::Ppo => run_ppo(cli.character),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_cli_defaults_to_gbm_ironclad() {
        let cli = parse_cli(["synergy_score".to_string()].into_iter()).expect("ok");
        assert_eq!(cli.net, NetMode::Gbm);
        assert_eq!(cli.character, CharacterId::Ironclad);
    }

    #[test]
    fn parse_cli_reads_net_and_character() {
        let cli = parse_cli(
            ["bin", "--net", "ppo", "--character", "silent"]
                .into_iter()
                .map(String::from),
        )
        .expect("ok");
        assert_eq!(cli.net, NetMode::Ppo);
        assert_eq!(cli.character, CharacterId::Silent);
    }

    #[test]
    fn parse_cli_rejects_unknown_net() {
        assert!(
            parse_cli(["b", "--net", "bogus"].into_iter().map(String::from)).is_err(),
            "unknown --net value must error"
        );
    }

    #[test]
    fn parse_cli_rejects_unknown_arg() {
        assert!(parse_cli(["b", "--wat"].into_iter().map(String::from)).is_err());
    }

    #[test]
    fn parse_character_maps_all() {
        for (s, c) in [
            ("ironclad", CharacterId::Ironclad),
            ("silent", CharacterId::Silent),
            ("defect", CharacterId::Defect),
            ("regent", CharacterId::Regent),
            ("necrobinder", CharacterId::Necrobinder),
        ] {
            assert_eq!(parse_character(s), Some(c));
        }
        assert_eq!(parse_character("nope"), None);
    }

    /// The blessed net is trained on the scraped run corpus and is not part
    /// of the public export (nor smudged in a fresh LFS checkout); the tests
    /// that need it skip when it is absent.
    fn net() -> Option<ValueNet> {
        let path = Path::new(DEFAULT_NET);
        let text = std::fs::read_to_string(path).ok()?;
        if text.starts_with("version https://git-lfs") {
            return None;
        }
        Some(ValueNet::from_weights_path(path).expect("blessed net loads"))
    }

    #[test]
    fn fixed_map_parses_and_is_nonempty() {
        // Guards the committed synergy_score_fixed_map.json — runs with no ONNX.
        let fm = load_fixed_map();
        let map = fm.map.expect("fixed map must be present");
        assert!(!map.rows.is_empty(), "fixed map must carry map nodes");
    }

    /// Load the character's PPO champion, or `None` when no ONNX is on disk
    /// (mirrors the other PPO smoke tests — skip gracefully in that case).
    fn ppo_net_or_skip(c: CharacterId) -> Option<PpoNet> {
        let path = onnx_model_path_for(c)?;
        PpoNet::load(&path).ok()
    }

    #[test]
    fn ppo_value_finite_and_deterministic() {
        let Some(net) = ppo_net_or_skip(CharacterId::Ironclad) else {
            eprintln!("no ironclad PPO champion on disk; skipping");
            return;
        };
        let view = build_view(&starter_input(), CharacterId::Ironclad).expect("view");
        let v1 = score_ppo(&net, &view, &load_fixed_map()).expect("scores");
        let v2 = score_ppo(&net, &view, &load_fixed_map()).expect("scores");
        assert!(v1.is_finite(), "value must be finite, got {v1}");
        assert_eq!(v1.to_bits(), v2.to_bits(), "value must be deterministic");
    }

    /// Size-controlled card-quality check: adding ONE strong card vs ONE curse
    /// to the same base deck must order correctly. Both arms have identical deck
    /// size, so the value head's deck-size term cancels and only card identity
    /// differs — the property the second-difference probe actually relies on.
    /// (Absolute strong-vs-weak of DIFFERENT-size decks is confounded by size
    /// and is NOT a valid check for this OOD static-obs probe — see plan §6.)
    #[test]
    fn ppo_one_strong_card_beats_one_curse() {
        let Some(net) = ppo_net_or_skip(CharacterId::Ironclad) else {
            eprintln!("no ironclad PPO champion on disk; skipping");
            return;
        };
        let base = [
            "STRIKE_IRONCLAD",
            "STRIKE_IRONCLAD",
            "STRIKE_IRONCLAD",
            "STRIKE_IRONCLAD",
            "STRIKE_IRONCLAD",
            "DEFEND_IRONCLAD",
            "DEFEND_IRONCLAD",
            "DEFEND_IRONCLAD",
            "DEFEND_IRONCLAD",
            "BASH",
        ];
        let with_card = |extra: &str| {
            let mut deck: Vec<String> = base.iter().map(|s| (*s).to_string()).collect();
            deck.push(extra.to_string());
            let input = StateInput {
                deck,
                relics: Some(vec!["BURNING_BLOOD".into()]),
                hp: Some(75),
                max_hp: Some(80),
                gold: Some(99),
                ascension: Some(0),
                act_index: Some(1),
                act_floor: Some(3),
            };
            let view = build_view(&input, CharacterId::Ironclad).expect("view");
            score_ppo(&net, &view, &load_fixed_map()).expect("scores")
        };
        let v_strong = with_card("BLUDGEON+1");
        let v_curse = with_card("CURSE_OF_THE_BELL");
        assert!(
            v_strong > v_curse,
            "FINDING: PPO value head ranks a curse above Bludgeon at equal deck \
             size (strong={v_strong:.4} !> curse={v_curse:.4}) — investigate \
             whether the map-less obs breaks value calibration."
        );
    }

    #[test]
    fn build_view_uses_character_starter_hp() {
        // Silent base max_hp = 70; omitting max_hp must default to it, and the
        // view must carry the requested character.
        let input: StateInput =
            serde_json::from_str(r#"{"deck":["STRIKE_SILENT"]}"#).expect("parse");
        let view = build_view(&input, CharacterId::Silent).expect("view");
        assert_eq!(view.max_hp, 70);
        assert_eq!(view.character, CharacterId::Silent);
    }

    fn starter_input() -> StateInput {
        StateInput {
            deck: vec![
                "STRIKE_IRONCLAD".into(),
                "STRIKE_IRONCLAD".into(),
                "STRIKE_IRONCLAD".into(),
                "STRIKE_IRONCLAD".into(),
                "STRIKE_IRONCLAD".into(),
                "DEFEND_IRONCLAD".into(),
                "DEFEND_IRONCLAD".into(),
                "DEFEND_IRONCLAD".into(),
                "DEFEND_IRONCLAD".into(),
                "BASH".into(),
            ],
            relics: Some(vec!["BURNING_BLOOD".into()]),
            hp: Some(75),
            max_hp: Some(80),
            gold: Some(99),
            ascension: Some(0),
            act_index: Some(1),
            act_floor: Some(3),
        }
    }

    fn strong_deck_input() -> StateInput {
        // Starter deck + a set of strong Ironclad cards commonly found in
        // winning runs.
        let mut deck = vec![
            "STRIKE_IRONCLAD".into(),
            "STRIKE_IRONCLAD".into(),
            "STRIKE_IRONCLAD".into(),
            "STRIKE_IRONCLAD".into(),
            "STRIKE_IRONCLAD".into(),
            "DEFEND_IRONCLAD".into(),
            "DEFEND_IRONCLAD".into(),
            "DEFEND_IRONCLAD".into(),
            "DEFEND_IRONCLAD".into(),
            "BASH".into(),
        ];
        // Add several known-strong cards (upgraded where reasonable).
        for card in &[
            "WHIRLWIND+1",
            "INFLAME+1",
            "CORRUPTION_CARD+1",
            "BLUDGEON+1",
            "FEEL_NO_PAIN_CARD+1",
            "IMPERVIOUS+1",
        ] {
            deck.push((*card).to_string());
        }
        StateInput {
            deck,
            relics: Some(vec!["BURNING_BLOOD".into()]),
            hp: Some(75),
            max_hp: Some(80),
            gold: Some(99),
            ascension: Some(0),
            act_index: Some(1),
            act_floor: Some(3),
        }
    }

    fn weak_deck_input() -> StateInput {
        // Starter deck + three curses — a clearly weaker position.
        let mut deck = vec![
            "STRIKE_IRONCLAD".into(),
            "STRIKE_IRONCLAD".into(),
            "STRIKE_IRONCLAD".into(),
            "STRIKE_IRONCLAD".into(),
            "STRIKE_IRONCLAD".into(),
            "DEFEND_IRONCLAD".into(),
            "DEFEND_IRONCLAD".into(),
            "DEFEND_IRONCLAD".into(),
            "DEFEND_IRONCLAD".into(),
            "BASH".into(),
        ];
        for _ in 0..3 {
            deck.push("CURSE_OF_THE_BELL".into());
        }
        StateInput {
            deck,
            relics: Some(vec!["BURNING_BLOOD".into()]),
            hp: Some(75),
            max_hp: Some(80),
            gold: Some(99),
            ascension: Some(0),
            act_index: Some(1),
            act_floor: Some(3),
        }
    }

    /// A fixed state yields a stable, finite V across two calls.
    #[test]
    fn fixed_state_stable_finite() {
        let Some(net) = net() else {
            eprintln!("skipping: {DEFAULT_NET} not present");
            return;
        };
        let input = starter_input();
        let v1 = score(&net, &input, CharacterId::Ironclad).expect("scores ok");
        let v2 = score(&net, &input, CharacterId::Ironclad).expect("scores ok");
        assert!(v1.is_finite(), "V must be finite, got {v1}");
        // Determinism means bit-identical output for identical input;
        // compare raw bits to assert exact equality without tripping the
        // float_cmp lint.
        assert_eq!(
            v1.to_bits(),
            v2.to_bits(),
            "V must be deterministic: {v1} != {v2}"
        );
    }

    /// A clearly stronger deck (starter + strong cards) should score higher V
    /// than a clearly weaker one (starter + curses). If the net disagrees,
    /// that is a real finding worth investigating.
    #[test]
    fn stronger_deck_scores_higher_than_weaker() {
        let Some(net) = net() else {
            eprintln!("skipping: {DEFAULT_NET} not present");
            return;
        };
        let v_strong =
            score(&net, &strong_deck_input(), CharacterId::Ironclad).expect("strong deck scores");
        let v_weak =
            score(&net, &weak_deck_input(), CharacterId::Ironclad).expect("weak deck scores");
        assert!(
            v_strong.is_finite() && v_weak.is_finite(),
            "both V must be finite: strong={v_strong}, weak={v_weak}"
        );
        assert!(
            v_strong > v_weak,
            "FINDING: net disagrees with expected direction — \
             strong={v_strong:.4} NOT > weak={v_weak:.4}. \
             This may indicate the net doesn't distinguish curses vs strong cards well."
        );
    }

    /// An unknown card token must produce a clear error, not a silent zero.
    #[test]
    fn unknown_card_errors_clearly() {
        let Some(net) = net() else {
            eprintln!("skipping: {DEFAULT_NET} not present");
            return;
        };
        let input = StateInput {
            deck: vec!["NOT_A_REAL_CARD".into()],
            relics: Some(vec!["BURNING_BLOOD".into()]),
            hp: Some(70),
            max_hp: Some(80),
            gold: Some(99),
            ascension: Some(0),
            act_index: Some(1),
            act_floor: Some(1),
        };
        let result = score(&net, &input, CharacterId::Ironclad);
        assert!(result.is_err(), "expected error for unknown card, got Ok");
        let msg = result.unwrap_err();
        assert!(
            msg.contains("NOT_A_REAL_CARD"),
            "error message must name the bad token, got: '{msg}'"
        );
    }

    /// An unknown relic token must also produce a clear error.
    #[test]
    fn unknown_relic_errors_clearly() {
        let Some(net) = net() else {
            eprintln!("skipping: {DEFAULT_NET} not present");
            return;
        };
        let input = StateInput {
            deck: vec!["BASH".into()],
            relics: Some(vec!["NOT_A_REAL_RELIC".into()]),
            hp: Some(70),
            max_hp: Some(80),
            gold: Some(99),
            ascension: Some(0),
            act_index: Some(1),
            act_floor: Some(1),
        };
        let result = score(&net, &input, CharacterId::Ironclad);
        assert!(result.is_err(), "expected error for unknown relic, got Ok");
        let msg = result.unwrap_err();
        assert!(
            msg.contains("NOT_A_REAL_RELIC"),
            "error message must name the bad token, got: '{msg}'"
        );
    }

    /// Omitting optional fields uses sensible defaults and still scores.
    #[test]
    fn defaults_score_finite() {
        let Some(net) = net() else {
            eprintln!("skipping: {DEFAULT_NET} not present");
            return;
        };
        let input: StateInput = serde_json::from_str(r#"{"deck":["BASH","STRIKE_IRONCLAD"]}"#)
            .expect("parse minimal input");
        let v = score(&net, &input, CharacterId::Ironclad).expect("scores with defaults");
        assert!(v.is_finite(), "V with defaults must be finite, got {v}");
    }
}
