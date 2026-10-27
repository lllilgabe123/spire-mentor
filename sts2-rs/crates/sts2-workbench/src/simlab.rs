//! The SimLab sandbox position, ported to the browser engine.
//!
//! A Rust twin of `sts2-pycli/python/simlab/position.py`. The engine's own
//! format for a fight is the *wire dict* (`state_to_dict` / `state_from_dict`):
//! card uids, per-enemy uids, an intent history, RNG cursors — nothing a person
//! wants to type. The sandbox edits a compact `simlab-position/1` document and
//! this module is the only place that knows how the two relate:
//!
//! * [`position_from_wire`] — engine wire → position (an import / hand-off;
//!   everything the position cannot carry is listed in the returned notes,
//!   never silently dropped).
//! * [`wire_from_position`] — position → engine wire (what a build submits).
//! * [`build_position`] — the round trip through the engine. A position is
//!   "built" when [`state_from_dict`](sts2_wire::state_dict::state_from_dict_impl)
//!   accepts the wire it produces; the normalised position handed back is what
//!   the engine actually holds.
//!
//! Every field name is stable API for the SPA and for pasted JSON, so a change
//! here is a schema change. Kept byte-parallel with the Python module — the
//! desktop `pytest` suite is the reference; this serves the browser build,
//! which has no Python server to answer `/api/simlab/*`.

use serde_json::{json, Map, Value};
use sts2_core::{legal_actions as core_legal_actions, terminal_value, CombatState};
use sts2_wire::emit::combat::combat_action_to_value;
use sts2_wire::emit::{combat_state_to_value, pending_choice};
use sts2_wire::state_dict::{parse_state_dict, state_from_dict_impl};

pub const SCHEMA: &str = "simlab-position/1";

/// PascalCase names the engine's scenario builder accepts, in the game's order.
pub const CHARACTERS: [&str; 5] = ["Ironclad", "Silent", "Defect", "Regent", "Necrobinder"];

/// Base orb values by type — what a hand-authored orb starts at (Focus 0). The
/// engine folds Focus in itself on a fresh channel; on a rebuilt board the wire
/// carries the already-scaled numbers and those win.
pub const ORB_BASE: [(&str, i64, i64); 5] = [
    ("Lightning", 3, 8),
    ("Frost", 2, 5),
    ("Dark", 6, 6),
    ("Plasma", 1, 2),
    ("Glass", 3, 6),
];

const TERMINAL_PHASES: [&str; 2] = ["Victory", "Defeat"];

/// The single monster used as a filler encounter when a position names its own
/// lineup instead of an encounter id (`simlab.specs._MONSTER_BASE`).
const MONSTER_BASE: &str = "MAWLER_NORMAL";

/// A position the sandbox cannot turn into a fight. The message is user-facing.
#[derive(Debug, Clone)]
pub struct PositionError(pub String);

impl std::fmt::Display for PositionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for PositionError {}

impl PositionError {
    fn new(msg: impl Into<String>) -> Self {
        Self(msg.into())
    }
}

type PResult<T> = Result<T, PositionError>;

/// A name resolver `(kind, id) -> display`, injected so labels can use the
/// catalog's words; the default title-cases the id like the Python default.
pub type NameOf<'a> = dyn Fn(&str, &str) -> String + 'a;

/// `FuzzyWurmCrawler` → `FUZZY_WURM_CRAWLER` (the engine's own rule; a faithful
/// port of the two `_camel_to_upper_snake` regexes in `position.py`).
#[must_use]
pub fn camel_to_upper_snake(name: &str) -> String {
    let chars: Vec<char> = name.chars().collect();
    let mut out = String::with_capacity(name.len() + 4);
    for (i, &c) in chars.iter().enumerate() {
        if c.is_ascii_uppercase() {
            let prev = if i > 0 { Some(chars[i - 1]) } else { None };
            let next = chars.get(i + 1).copied();
            // (?<=[a-z0-9])([A-Z]) — boundary after a lowercase/digit.
            let after_lower_or_digit =
                matches!(prev, Some(p) if p.is_ascii_lowercase() || p.is_ascii_digit());
            // (?<=[A-Z])([A-Z][a-z]) — acronym → word boundary (HTTPServer).
            let acronym_word = matches!(prev, Some(p) if p.is_ascii_uppercase())
                && matches!(next, Some(n) if n.is_ascii_lowercase());
            if i > 0 && (after_lower_or_digit || acronym_word) {
                out.push('_');
            }
        }
        out.push(c);
    }
    out.to_ascii_uppercase()
}

/// Canonical registry spelling for a card id off any wire dialect. The live
/// bridge spells statuses `status:Slimed` and curses `curse:Clumsy`; the engine
/// registry names them `SLIMED` / `CLUMSY`. Plain ids pass through upper-cased.
#[must_use]
pub fn normalize_card_id(raw: &str) -> String {
    let trimmed = raw.trim();
    let stripped = trimmed
        .strip_prefix("status:")
        .or_else(|| trimmed.strip_prefix("STATUS:"))
        .or_else(|| trimmed.strip_prefix("curse:"))
        .or_else(|| trimmed.strip_prefix("CURSE:"))
        .map_or(trimmed, str::trim);
    if stripped.to_ascii_uppercase() == stripped {
        stripped.to_string()
    } else {
        camel_to_upper_snake(stripped)
    }
}

/// `Ironclad` → the lowercase `player.class` the wire parser requires.
#[must_use]
fn wire_class(character: &str) -> String {
    character.to_ascii_lowercase()
}

/// `ironclad` → `Ironclad` (the position's PascalCase spelling).
fn character_from_class(cls: &str) -> PResult<&'static str> {
    for name in CHARACTERS {
        if name.eq_ignore_ascii_case(cls) {
            return Ok(name);
        }
    }
    Err(PositionError::new(format!(
        "unknown character {cls:?} (expected one of {})",
        CHARACTERS.join(", ")
    )))
}

// ---------------------------------------------------------------------------
// Small Value helpers (the position is JSON, like the Python dict)
// ---------------------------------------------------------------------------

fn as_i64_or(v: Option<&Value>, default: i64) -> i64 {
    match v {
        Some(Value::Number(n)) => n.as_i64().unwrap_or(default),
        Some(Value::Bool(b)) => i64::from(*b),
        _ => default,
    }
}

fn as_str<'a>(v: Option<&'a Value>, default: &'a str) -> &'a str {
    v.and_then(Value::as_str).unwrap_or(default)
}

fn truthy(v: Option<&Value>) -> bool {
    match v {
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64().is_some_and(|f| f != 0.0),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
        _ => false,
    }
}

/// Parse an integer field, faithful to `position.py::_int` (a user-facing error
/// on a non-number or a below-minimum value).
fn int_field(v: Option<&Value>, what: &str, minimum: Option<i64>) -> PResult<i64> {
    let out = match v {
        None | Some(Value::Null) => 0,
        Some(Value::Number(n)) => n
            .as_i64()
            .or_else(|| n.as_f64().map(|f| f as i64))
            .ok_or_else(|| PositionError::new(format!("{what} must be a whole number")))?,
        Some(Value::Bool(b)) => i64::from(*b),
        other => {
            return Err(PositionError::new(format!(
                "{what} must be a whole number, got {other:?}"
            )))
        }
    };
    if let Some(m) = minimum {
        if out < m {
            return Err(PositionError::new(format!(
                "{what} must be at least {m}, got {out}"
            )));
        }
    }
    Ok(out)
}

/// The engine wire dict → `CombatState` (`state_from_dict`).
fn state_from_wire(wire: &Value) -> PResult<CombatState> {
    let parsed = parse_state_dict(&wire)
        .map_err(|e| PositionError::new(format!("the engine rejected this position: {e}")))?;
    state_from_dict_impl(&parsed)
        .map_err(|e| PositionError::new(format!("the engine rejected this position: {e}")))
}

/// `CombatState` → the engine wire dict (`state_to_dict`).
fn state_to_wire(cs: &CombatState) -> PResult<Value> {
    combat_state_to_value(cs)
        .map_err(|e| PositionError::new(format!("the engine could not emit this board: {e}")))
}

// ---------------------------------------------------------------------------
// Position construction
// ---------------------------------------------------------------------------

/// A blank but well-formed position: no cards, no enemies, full HP.
#[must_use]
pub fn empty_position(character: &str, ascension: i64) -> Value {
    let character = character_from_class(character).unwrap_or("Ironclad");
    json!({
        "schema": SCHEMA,
        "character": character,
        "ascension": ascension,
        "turn": 1,
        "isElite": false,
        "encounter": Value::Null,
        "seed": 1,
        "player": {
            "hp": 80, "maxHp": 80, "block": 0, "energy": 3, "maxEnergy": 3,
            "stars": 0, "gold": 99, "powers": [], "relics": [], "potions": [],
            "orbSlots": 0, "orbs": [], "osty": Value::Null, "cardsPlayedThisTurn": Value::Null,
        },
        "piles": {"hand": [], "draw": [], "discard": [], "exhaust": []},
        "monsters": [],
    })
}

fn card_row(wire_card: &Value, notes: &mut Vec<String>, where_: &str) -> Value {
    let id = normalize_card_id(as_str(wire_card.get("id"), ""));
    let upgrade = i64::from(truthy(wire_card.get("upgraded")));
    let cost = wire_card
        .get("base_cost")
        .or_else(|| wire_card.get("cost_override"))
        .cloned()
        .unwrap_or(Value::Null);
    let dropped: Vec<&str> = [
        "enchantment",
        "affliction",
        "flags",
        "combat_vars",
        "tinker_time_rider",
    ]
    .into_iter()
    .filter(|k| truthy(wire_card.get(*k)))
    .collect();
    if !dropped.is_empty() {
        notes.push(format!(
            "{where_}: {id} loses {} (not carried by the position).",
            dropped.join(", ")
        ));
    }
    json!({"card": id, "upgrade": upgrade, "cost": cost})
}

fn power_rows(wire_powers: Option<&Value>) -> Value {
    let mut rows = Vec::new();
    if let Some(Value::Array(ps)) = wire_powers {
        for p in ps {
            let id = p
                .get("name")
                .and_then(Value::as_str)
                .or_else(|| p.get("id").and_then(Value::as_str))
                .unwrap_or("");
            let amount = as_i64_or(p.get("amount"), 0);
            // The engine emits `counter: 0` for every power; only a live counter
            // (Slow, Orbit, Crimson Mantle) is worth carrying.
            let counter = match p.get("counter") {
                Some(Value::Number(n)) if n.as_i64() != Some(0) => json!(n),
                _ => Value::Null,
            };
            rows.push(json!({"id": id, "amount": amount, "counter": counter}));
        }
    }
    Value::Array(rows)
}

/// Translate an engine wire dict into a position. Returns `(position, notes)` —
/// the notes name everything the wire carried that the position does not.
pub fn position_from_wire(
    wire: &Value,
    encounter: Option<&str>,
    seed: Option<i64>,
) -> PResult<(Value, Vec<String>)> {
    let obj = wire
        .as_object()
        .filter(|o| o.contains_key("player") && o.contains_key("deck"));
    let Some(_) = obj else {
        return Err(PositionError::new(
            "not an engine combat wire (no player / deck block)",
        ));
    };
    let phase = as_str(wire.get("phase"), "PlayerPlayPhase");
    if TERMINAL_PHASES.contains(&phase) {
        return Err(PositionError::new(format!(
            "that fight is already over ({}); nothing to play from",
            phase.to_ascii_lowercase()
        )));
    }
    let mut notes: Vec<String> = Vec::new();
    let player = &wire["player"];
    let deck = &wire["deck"];
    let rng = wire.get("rng").filter(|v| v.is_object());

    if truthy(deck.get("play_pile")) {
        return Err(PositionError::new(
            "a card is mid-resolution (play pile is not empty); copy the state at a decision point",
        ));
    }
    for key in ["pinned_top", "pinned_bottom"] {
        if truthy(deck.get(key)) {
            notes.push(format!(
                "Draw-pile {} pins are not carried; the order is.",
                key.replace('_', " ")
            ));
        }
    }
    if truthy(deck.get("deferred_draw_order")) {
        notes.push("The deferred draw-order flag is not carried.".to_string());
    }

    let class = as_str(player.get("class"), "ironclad");
    let mut pos = empty_position(character_from_class(class)?, 0);
    let p = pos.as_object_mut().unwrap();
    p.insert(
        "ascension".into(),
        json!(rng
            .and_then(|r| r.get("ascension"))
            .and_then(Value::as_i64)
            .unwrap_or(0)),
    );
    let turn = wire
        .get("round")
        .and_then(Value::as_i64)
        .or_else(|| wire.get("turn").and_then(Value::as_i64))
        .filter(|&t| t != 0)
        .unwrap_or(1);
    p.insert("turn".into(), json!(turn));
    p.insert("isElite".into(), json!(truthy(wire.get("is_elite"))));
    // A non-default build's wire names it; the position carries it on so a
    // rebuild plays the same build (GH #877).
    if let Some(v) = wire.get("game_version").filter(|v| v.is_string()) {
        p.insert("game_version".into(), v.clone());
    }
    p.insert(
        "encounter".into(),
        encounter.map_or(Value::Null, |e| json!(e)),
    );
    let seed_val = seed
        .or_else(|| rng.and_then(|r| r.get("seed")).and_then(Value::as_i64))
        .filter(|&s| s != 0)
        .unwrap_or(1);
    p.insert("seed".into(), json!(seed_val));

    let pp = p.get_mut("player").unwrap().as_object_mut().unwrap();
    pp.insert("hp".into(), json!(as_i64_or(player.get("hp"), 0)));
    pp.insert("maxHp".into(), json!(as_i64_or(player.get("max_hp"), 0)));
    pp.insert("block".into(), json!(as_i64_or(player.get("block"), 0)));
    pp.insert("energy".into(), json!(as_i64_or(player.get("energy"), 0)));
    pp.insert(
        "maxEnergy".into(),
        json!(as_i64_or(player.get("max_energy"), 3)),
    );
    pp.insert("stars".into(), json!(as_i64_or(player.get("stars"), 0)));
    pp.insert("gold".into(), json!(as_i64_or(player.get("gold"), 0)));
    pp.insert("powers".into(), power_rows(player.get("powers")));
    let relics: Vec<Value> = player
        .get("relics")
        .and_then(Value::as_array)
        .map(|rs| {
            rs.iter()
                .filter(|r| r.is_object())
                .map(|r| json!({"id": as_str(r.get("id"), ""), "counter": as_i64_or(r.get("counter"), 0)}))
                .collect()
        })
        .unwrap_or_default();
    pp.insert("relics".into(), Value::Array(relics));
    let potions: Vec<Value> = player
        .get("potions")
        .and_then(Value::as_array)
        .map(|ps| ps.iter().filter(|p| truthy(Some(p))).cloned().collect())
        .unwrap_or_default();
    pp.insert("potions".into(), Value::Array(potions));
    pp.insert(
        "orbSlots".into(),
        json!(as_i64_or(player.get("orb_slots"), 0)),
    );
    let orbs: Vec<Value> = player
        .get("orbs")
        .and_then(Value::as_array)
        .map(|os| {
            os.iter()
                .filter(|o| o.is_object())
                .map(|o| {
                    json!({
                        "type": as_str(o.get("type"), "Lightning"),
                        "passive": as_i64_or(o.get("passive"), 0),
                        "evoke": as_i64_or(o.get("evoke"), 0),
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    pp.insert("orbs".into(), Value::Array(orbs));
    let osty = match player.get("osty") {
        Some(o) if o.is_object() => json!({
            "alive": truthy(o.get("alive")) || o.get("alive").is_none(),
            "hp": as_i64_or(o.get("hp"), 0),
            "maxHp": as_i64_or(o.get("max_hp"), 1),
            "block": as_i64_or(o.get("block"), 0),
        }),
        _ => Value::Null,
    };
    pp.insert("osty".into(), osty);
    let cptt = match player.get("cards_played_this_turn") {
        Some(Value::Array(a)) => {
            Value::Array(a.iter().map(|x| json!(as_i64_or(Some(x), 0))).collect())
        }
        _ => Value::Null,
    };
    pp.insert("cardsPlayedThisTurn".into(), cptt);

    let piles = p.get_mut("piles").unwrap().as_object_mut().unwrap();
    for (pile, key) in [
        ("hand", "hand"),
        ("draw", "draw_pile"),
        ("discard", "discard_pile"),
        ("exhaust", "exhaust_pile"),
    ] {
        let cards: Vec<Value> = deck
            .get(key)
            .and_then(Value::as_array)
            .map(|cs| cs.iter().map(|c| card_row(c, &mut notes, pile)).collect())
            .unwrap_or_default();
        piles.insert(pile.into(), Value::Array(cards));
    }

    let mut monsters = Vec::new();
    if let Some(Value::Array(encs)) = wire.get("encounter") {
        for m in encs {
            if truthy(m.get("is_dead")) || truthy(m.get("is_escaped")) {
                continue;
            }
            let kind = as_str(m.get("kind"), "").to_string();
            let intent = m.get("move_state").cloned().unwrap_or(Value::Null);
            if intent.is_null() && truthy(m.get("intents")) {
                notes.push(format!(
                    "{kind}: the wire telegraphs a move without naming it; the intent will be re-rolled unless you pick one."
                ));
            }
            for key in [
                "stolen_cards",
                "fabricator_last_spawned",
                "curse_of_knowledge_counter",
            ] {
                if truthy(m.get(key)) {
                    notes.push(format!("{kind}: {key} is not carried by the position."));
                }
            }
            monsters.push(json!({
                "monster": kind,
                "hp": as_i64_or(m.get("hp"), 0),
                "maxHp": as_i64_or(m.get("max_hp"), 0),
                "block": as_i64_or(m.get("block"), 0),
                "powers": power_rows(m.get("powers")),
                "intent": intent,
            }));
        }
    }
    p.insert("monsters".into(), Value::Array(monsters));
    Ok((pos, notes))
}

// ---------------------------------------------------------------------------
// Position → wire
// ---------------------------------------------------------------------------

fn strip_move_segment(move_id: &str) -> String {
    move_id
        .strip_suffix("_MOVE")
        .unwrap_or(move_id)
        .replace("_MOVE_", "_")
}

fn wire_card(row: &Value, idx: usize) -> PResult<Value> {
    let raw = row
        .get("card")
        .and_then(Value::as_str)
        .or_else(|| row.get("id").and_then(Value::as_str))
        .unwrap_or("");
    let card_id = normalize_card_id(raw);
    if card_id.is_empty() {
        return Err(PositionError::new(format!("card #{} has no id", idx + 1)));
    }
    let upgraded = int_field(row.get("upgrade"), "upgrade", Some(0))? != 0;
    let mut out = Map::new();
    out.insert("id".into(), json!(card_id));
    out.insert("stable_idx".into(), json!(idx));
    out.insert("kind".into(), json!("card"));
    out.insert("upgraded".into(), json!(upgraded));
    if let Some(c) = row.get("cost") {
        if !c.is_null() {
            out.insert(
                "base_cost".into(),
                json!(int_field(Some(c), &format!("cost of {card_id}"), None)?),
            );
        }
    }
    Ok(Value::Object(out))
}

fn wire_powers(rows: Option<&Value>, where_: &str) -> PResult<Value> {
    let mut out = Vec::new();
    if let Some(Value::Array(rs)) = rows {
        for r in rs {
            let pid = r
                .get("id")
                .and_then(Value::as_str)
                .or_else(|| r.get("name").and_then(Value::as_str))
                .unwrap_or("")
                .trim()
                .to_ascii_uppercase();
            if pid.is_empty() {
                return Err(PositionError::new(format!("{where_}: a power has no id")));
            }
            let mut entry = Map::new();
            entry.insert("name".into(), json!(pid));
            entry.insert(
                "amount".into(),
                json!(int_field(
                    r.get("amount").or(Some(&json!(1))),
                    &format!("{pid} amount"),
                    None
                )?),
            );
            if let Some(c) = r.get("counter") {
                if !c.is_null() {
                    entry.insert(
                        "counter".into(),
                        json!(int_field(Some(c), &format!("{pid} counter"), None)?),
                    );
                }
            }
            out.push(Value::Object(entry));
        }
    }
    Ok(Value::Array(out))
}

/// Ask the engine what each enemy opens with, honouring any forced moves — a
/// fresh fight with the same lineup, whose `move_state`/telegraph we read off
/// each enemy's wire (the only way to respect the AI's random/positional
/// branches). Mirrors `position.py::_roll_intents`.
fn roll_intents(
    version: sts2_core::GameVersion,
    character: &str,
    monsters: &[Value],
    ascension: i64,
    seed: i64,
    forced: &[Option<String>],
) -> PResult<Vec<Value>> {
    let kinds: Vec<String> = monsters
        .iter()
        .map(|m| as_str(m.get("monster"), "").to_string())
        .collect();
    let cs = sts2_wire::scenario::build_scenario_state_for_version(
        version,
        u64::try_from(seed.max(1)).unwrap_or(1),
        character,
        &["STRIKE_IRONCLAD".to_string()],
        10,
        10,
        &[],
        &[],
        MONSTER_BASE,
        u8::try_from(ascension.clamp(0, 255)).unwrap_or(0),
        Some(&kinds),
        Some(forced),
    )
    .map_err(|e| PositionError::new(format!("the engine rejected this position: {e}")))?;
    let wire = state_to_wire(&cs)?;
    Ok(wire
        .get("encounter")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default())
}

/// Translate a position into the engine wire dict `state_from_dict` takes.
/// Registry errors (an unknown card) are left to the engine, whose message
/// names the JSON path.
pub fn wire_from_position(pos: &Value) -> PResult<Value> {
    let obj = pos
        .as_object()
        .ok_or_else(|| PositionError::new("position must be a JSON object"))?;
    match obj.get("schema") {
        None | Some(Value::Null) => {}
        Some(Value::String(s)) if s == SCHEMA => {}
        Some(other) => {
            return Err(PositionError::new(format!(
                "unsupported schema {other} (expected {SCHEMA})"
            )))
        }
    }
    let character = character_from_class(as_str(obj.get("character"), "Ironclad"))?;
    let version = position_game_version(pos)?;
    let ascension = int_field(obj.get("ascension"), "ascension", Some(0))?;
    let turn = int_field(obj.get("turn").or(Some(&json!(1))), "turn", Some(1))?;
    let seed = int_field(obj.get("seed").or(Some(&json!(1))), "seed", Some(0))?;
    let empty = json!({});
    let player = obj.get("player").unwrap_or(&empty);
    let piles = obj.get("piles").unwrap_or(&empty);

    let max_hp = int_field(player.get("maxHp").or(Some(&json!(80))), "max HP", Some(1))?;
    let hp = int_field(player.get("hp").or(Some(&json!(max_hp))), "HP", Some(0))?;
    if hp > max_hp {
        return Err(PositionError::new(format!(
            "HP ({hp}) is above max HP ({max_hp})"
        )));
    }
    let max_energy = int_field(
        player.get("maxEnergy").or(Some(&json!(3))),
        "max energy",
        Some(0),
    )?;
    let energy = int_field(
        player.get("energy").or(Some(&json!(max_energy))),
        "energy",
        Some(0),
    )?;

    let mut wire_player = Map::new();
    wire_player.insert("class".into(), json!(wire_class(character)));
    wire_player.insert("hp".into(), json!(hp));
    wire_player.insert("max_hp".into(), json!(max_hp));
    wire_player.insert(
        "block".into(),
        json!(int_field(player.get("block"), "block", Some(0))?),
    );
    wire_player.insert("energy".into(), json!(energy));
    wire_player.insert("max_energy".into(), json!(max_energy));
    wire_player.insert("energy_regen_per_turn".into(), json!(max_energy));
    wire_player.insert(
        "gold".into(),
        json!(int_field(player.get("gold"), "gold", Some(0))?),
    );
    let relics: Vec<Value> = match player.get("relics") {
        Some(Value::Array(rs)) => rs
            .iter()
            .map(|r| {
                if let Some(o) = r.as_object() {
                    json!({
                        "id": as_str(o.get("id"), "").trim().to_ascii_uppercase(),
                        "counter": int_field(o.get("counter"), "relic counter", None).unwrap_or(0),
                    })
                } else {
                    json!({"id": r.as_str().unwrap_or("").trim().to_ascii_uppercase(), "counter": 0})
                }
            })
            .collect(),
        _ => Vec::new(),
    };
    wire_player.insert("relics".into(), Value::Array(relics));
    wire_player.insert(
        "powers".into(),
        wire_powers(player.get("powers"), "player")?,
    );
    let potions: Vec<Value> = player
        .get("potions")
        .and_then(Value::as_array)
        .map(|ps| {
            ps.iter()
                .filter(|p| truthy(Some(p)))
                .map(|p| json!(p.as_str().unwrap_or("").trim().to_ascii_uppercase()))
                .collect()
        })
        .unwrap_or_default();
    wire_player.insert("potions".into(), Value::Array(potions));
    let stars = int_field(player.get("stars"), "stars", Some(0))?;
    if stars != 0 {
        wire_player.insert("stars".into(), json!(stars));
    }
    let orb_slots = int_field(player.get("orbSlots"), "orb slots", Some(0))?;
    let orbs = player
        .get("orbs")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if orb_slots != 0 || !orbs.is_empty() {
        if orbs.len() as i64 > orb_slots {
            return Err(PositionError::new(format!(
                "{} orbs channelled but only {orb_slots} orb slots",
                orbs.len()
            )));
        }
        let mut wire_orbs = Vec::new();
        for (i, o) in orbs.iter().enumerate() {
            let otype_raw = as_str(o.get("type"), "Lightning");
            let mut otype = otype_raw.to_ascii_lowercase();
            if let Some(first) = otype.get_mut(0..1) {
                first.make_ascii_uppercase();
            }
            let base = ORB_BASE.iter().find(|(t, _, _)| *t == otype);
            let Some((_, bp, be)) = base else {
                return Err(PositionError::new(format!(
                    "orb #{}: unknown orb type {otype:?}",
                    i + 1
                )));
            };
            wire_orbs.push(json!({
                "index": i,
                "type": otype,
                "passive": int_field(o.get("passive").or(Some(&json!(bp))), "orb passive", None)?,
                "evoke": int_field(o.get("evoke").or(Some(&json!(be))), "orb evoke", None)?,
            }));
        }
        wire_player.insert("orbs".into(), Value::Array(wire_orbs));
        wire_player.insert("orb_slots".into(), json!(orb_slots));
    }
    if let Some(osty) = player.get("osty").filter(|o| o.is_object()) {
        wire_player.insert(
            "osty".into(),
            json!({
                "alive": truthy(osty.get("alive")) || osty.get("alive").is_none(),
                "hp": int_field(osty.get("hp"), "Osty HP", Some(0))?,
                "max_hp": int_field(osty.get("maxHp").or(Some(&json!(1))), "Osty max HP", Some(1))?,
                "block": int_field(osty.get("block"), "Osty block", Some(0))?,
            }),
        );
    }
    if let Some(Value::Array(cptt)) = player.get("cardsPlayedThisTurn") {
        if cptt.len() == 5 {
            let vals: PResult<Vec<Value>> = cptt
                .iter()
                .map(|x| Ok(json!(int_field(Some(x), "cards played", None)?)))
                .collect();
            wire_player.insert("cards_played_this_turn".into(), Value::Array(vals?));
        }
    }

    let mut wire_deck = Map::new();
    let mut idx = 0usize;
    for (pile, key) in [
        ("hand", "hand"),
        ("draw", "draw_pile"),
        ("discard", "discard_pile"),
        ("exhaust", "exhaust_pile"),
    ] {
        let rows = piles
            .get(pile)
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let mut cards = Vec::new();
        for row in &rows {
            cards.push(wire_card(row, idx)?);
            idx += 1;
        }
        wire_deck.insert(key.into(), Value::Array(cards));
    }
    if idx == 0 {
        return Err(PositionError::new("the position has no cards in any pile"));
    }

    let monsters = pos
        .get("monsters")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if monsters.is_empty() {
        return Err(PositionError::new("the position has no enemies"));
    }
    let forced: Vec<Option<String>> = monsters
        .iter()
        .map(|m| match m.get("intent") {
            Some(v) if truthy(Some(v)) => v.as_str().map(str::to_string),
            _ => None,
        })
        .collect();
    let rolled = if forced.iter().any(Option::is_none) {
        roll_intents(version, character, &monsters, ascension, seed, &forced)?
    } else {
        Vec::new()
    };

    let mut encounter = Vec::new();
    for (i, m) in monsters.iter().enumerate() {
        let kind = as_str(m.get("monster"), "").trim().to_string();
        if kind.is_empty() {
            return Err(PositionError::new(format!(
                "enemy #{} has no monster",
                i + 1
            )));
        }
        let max_m_hp = int_field(
            m.get("maxHp").or(Some(&json!(1))),
            &format!("{kind} max HP"),
            Some(1),
        )?;
        let m_hp = int_field(
            m.get("hp").or(Some(&json!(max_m_hp))),
            &format!("{kind} HP"),
            Some(1),
        )?;
        if m_hp > max_m_hp {
            return Err(PositionError::new(format!(
                "{kind}: HP ({m_hp}) is above max HP ({max_m_hp})"
            )));
        }
        let mut move_state: Option<String> = forced[i].clone();
        let mut intents: Vec<Value> = Vec::new();
        if let Some(ms) = move_state.clone() {
            let table = monster_move_table(&kind, ascension)?;
            let mut found = table
                .iter()
                .find(|row| as_str(row.get("id"), "") == ms)
                .cloned();
            if found.is_none() {
                // The live bridge spells C# ids (`WRIGGLE_MOVE`, `THRASH_MOVE_2`)
                // where the engine has `WRIGGLE` / `THRASH_2`; match the way
                // `MonsterAi::force_current_state_by_name_relaxed` does.
                let want = strip_move_segment(&ms);
                for row in &table {
                    let mid = as_str(row.get("id"), "");
                    if strip_move_segment(mid) == want {
                        found = Some(row.clone());
                        move_state = Some(mid.to_string());
                        break;
                    }
                }
            }
            let Some(mv) = found else {
                let mut ids: Vec<String> = table
                    .iter()
                    .map(|r| as_str(r.get("id"), "").to_string())
                    .collect();
                ids.sort();
                let known = if ids.is_empty() {
                    "none known".into()
                } else {
                    ids.join(", ")
                };
                return Err(PositionError::new(format!(
                    "{kind}: {ms:?} is not one of its moves ({known})"
                )));
            };
            if let Some(intent) = mv.get("intent").filter(|v| !v.is_null()) {
                intents = vec![intent.clone()];
            }
        } else if i < rolled.len() {
            move_state = rolled[i]
                .get("move_state")
                .and_then(Value::as_str)
                .map(str::to_string);
            intents = rolled[i]
                .get("intents")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
        }
        let mut entry = Map::new();
        entry.insert("kind".into(), json!(kind));
        entry.insert("uid".into(), json!(i + 1));
        entry.insert("hp".into(), json!(m_hp));
        entry.insert("max_hp".into(), json!(max_m_hp));
        entry.insert(
            "block".into(),
            json!(int_field(
                m.get("block"),
                &format!("{kind} block"),
                Some(0)
            )?),
        );
        entry.insert("powers".into(), wire_powers(m.get("powers"), &kind)?);
        entry.insert("intents".into(), Value::Array(intents));
        entry.insert("intent_history".into(), json!([]));
        entry.insert("is_dead".into(), json!(false));
        entry.insert("is_escaped".into(), json!(false));
        if let Some(ms) = move_state.filter(|s| !s.is_empty()) {
            entry.insert("move_state".into(), json!(ms));
        }
        encounter.push(Value::Object(entry));
    }

    let mut wire = json!({
        "wire_version": 1,
        "phase": "PlayerPlayPhase",
        "round": turn,
        "turn": turn,
        "side": "Player",
        "is_elite": truthy(pos.get("isElite")),
        "combat_id": 0,
        "intent_amounts": "base",
        "player": Value::Object(wire_player),
        "deck": Value::Object(wire_deck),
        "encounter": Value::Array(encounter),
        "rng": {"seed": seed, "ascension": ascension, "run_seed": seed & 0xFFFF_FFFF},
    });
    // The engine wire names a non-default build (`$.game_version`), exactly
    // as `state_to_dict` writes it — absent for the default build.
    if !version.is_default() {
        wire["game_version"] = json!(version.as_str());
    }
    Ok(wire)
}

/// The game build a position plays (GH #877): its `game_version` field,
/// absent / `null` = the default build. Written only for a non-default build
/// (by [`position_from_wire`], from the wire's own `game_version`), so every
/// v0.107.1 position is unchanged.
pub fn position_game_version(pos: &Value) -> PResult<sts2_core::GameVersion> {
    crate::session::parse_game_version(pos.get("game_version")).map_err(PositionError::new)
}

// ---------------------------------------------------------------------------
// The round trip
// ---------------------------------------------------------------------------

/// What the engine holds for a position, plus what it changed on the way.
#[derive(Debug)]
pub struct BuiltPosition {
    pub position: Value,
    pub wire: Value,
    pub notes: Vec<String>,
}

/// Run a position through the engine and hand back its normalised form. The
/// wire returned is the engine's own `state_to_dict` of the rebuilt board (so it
/// carries the enemy AI's resolved move), and the position is re-derived from
/// that wire — how a user sees a rolled intent or a clamped value the moment it
/// happens.
pub fn build_position(pos: &Value) -> PResult<BuiltPosition> {
    let wire_in = wire_from_position(pos)?;
    let cs = state_from_wire(&wire_in)?;
    let wire_out = state_to_wire(&cs)?;
    let encounter = pos.get("encounter").and_then(Value::as_str);
    let seed = pos
        .get("seed")
        .and_then(Value::as_i64)
        .filter(|&s| s != 0)
        .unwrap_or(1);
    let (mut built, notes) = position_from_wire(&wire_out, encounter, Some(seed))?;
    // The wire the engine emits has no seed of its own; keep the user's so a
    // rebuild from the SAME position is the same fight.
    built.as_object_mut().unwrap().insert(
        "encounter".into(),
        encounter.map_or(Value::Null, |e| json!(e)),
    );
    Ok(BuiltPosition {
        position: built,
        wire: wire_out,
        notes,
    })
}

/// A fresh fight from turn 1, as the game would deal it. `starter` is the
/// character's `{hp, deck, relic}` row (from the catalog).
#[allow(clippy::too_many_arguments)]
pub fn new_position(
    version: sts2_core::GameVersion,
    character: &str,
    encounter: Option<&str>,
    monsters: Option<&[String]>,
    ascension: i64,
    seed: i64,
    hp: Option<i64>,
    max_hp: Option<i64>,
    relics: Option<&[String]>,
    potions: &[String],
    deck: Option<&[String]>,
    starter_hp: i64,
    starter_deck: &[String],
    starter_relic: Option<&str>,
) -> PResult<BuiltPosition> {
    let character = character_from_class(character)?;
    let max_hp = max_hp.unwrap_or(starter_hp).max(1);
    let hp = hp.unwrap_or(max_hp);
    let deck_ids: Vec<String> = deck.map_or_else(|| starter_deck.to_vec(), <[String]>::to_vec);
    let relic_ids: Vec<String> = match relics {
        Some(rs) => rs.to_vec(),
        None => starter_relic
            .map(|r| vec![r.to_string()])
            .unwrap_or_default(),
    };
    let lineup: Vec<String> = monsters
        .map(|ms| ms.iter().filter(|m| !m.is_empty()).cloned().collect())
        .unwrap_or_default();
    if encounter.is_none() && lineup.is_empty() {
        return Err(PositionError::new(
            "pick an encounter or name at least one monster",
        ));
    }
    let cs = sts2_wire::scenario::build_scenario_state_for_version(
        // The selector's build (`game_version` on the request, GH #877).
        version,
        u64::try_from(seed.max(0)).unwrap_or(0),
        character,
        &deck_ids,
        i32::try_from(hp).unwrap_or(i32::MAX),
        i32::try_from(max_hp).unwrap_or(i32::MAX),
        &relic_ids,
        potions,
        encounter.unwrap_or(MONSTER_BASE),
        u8::try_from(ascension.clamp(0, 255)).unwrap_or(0),
        if lineup.is_empty() {
            None
        } else {
            Some(&lineup)
        },
        None,
    )
    .map_err(|e| PositionError::new(format!("the engine rejected this position: {e}")))?;
    let wire = state_to_wire(&cs)?;
    let label = encounter.map_or_else(|| format!("monsters:{}", lineup.join(",")), str::to_string);
    let (pos, notes) = position_from_wire(&wire, Some(&label), Some(seed))?;
    Ok(BuiltPosition {
        position: pos,
        wire,
        notes,
    })
}

// ---------------------------------------------------------------------------
// Pasted text → position
// ---------------------------------------------------------------------------

fn looks_like_position(v: &Value) -> bool {
    v.as_object().is_some_and(|o| {
        o.get("schema").and_then(Value::as_str) == Some(SCHEMA)
            || (o.contains_key("piles") && o.contains_key("monsters"))
    })
}

fn looks_like_engine_wire(v: &Value) -> bool {
    v.as_object()
        .is_some_and(|o| o.contains_key("wire_version") && o.contains_key("deck"))
}

/// The engine's own emission of `wire`, or `wire` itself if it refuses.
fn normalized_wire(wire: &Value) -> Value {
    match state_from_wire(wire).and_then(|cs| state_to_wire(&cs)) {
        Ok(v) => v,
        Err(_) => wire.clone(),
    }
}

/// Turn whatever was pasted into a position. Accepted, in order: a
/// `simlab-position/1` document, an engine wire dict, and anything
/// [`crate::state_import::parse_pasted_state`] reads (SpireMentor Copy-state, an
/// advisor log line, a whole log). Returns `(position, notes, source)`.
pub fn position_from_paste(text: &str) -> PResult<(Value, Vec<String>, String)> {
    let stripped = text.trim();
    if stripped.is_empty() {
        return Err(PositionError::new("nothing pasted"));
    }
    if let Ok(parsed) = serde_json::from_str::<Value>(stripped) {
        if looks_like_position(&parsed) {
            let mut pos = parsed.clone();
            if let Some(o) = pos.as_object_mut() {
                o.entry("schema").or_insert_with(|| json!(SCHEMA));
            }
            return Ok((pos, Vec::new(), "position".to_string()));
        }
        if looks_like_engine_wire(&parsed) {
            let (pos, notes) = position_from_wire(&normalized_wire(&parsed), None, None)?;
            return Ok((pos, notes, "engine wire".to_string()));
        }
    }

    // Neither a position doc nor an engine wire dict. The other shape a person
    // is likely to paste is a SpireMentor Copy-state (an advisor `advice_request`
    // / log line): the parser turns its fight into the engine's wire through
    // `crate::combat_import` (GH #708), and that wire becomes the board.
    let imported =
        crate::state_import::parse_pasted_state(stripped).map_err(|e| PositionError::new(e.0))?;
    if let Some(combat) = &imported.run.combat {
        let (pos, mut notes) = position_from_wire(&normalized_wire(&combat.wire), None, None)?;
        // What the translation dropped rides with the board.
        notes.extend(
            imported
                .warnings
                .iter()
                .filter(|w| w.contains("dropped") || w.contains("left out"))
                .cloned(),
        );
        return Ok((pos, notes, "SpireMentor Copy-state".to_string()));
    }
    if imported.had_combat {
        let why = imported
            .warnings
            .iter()
            .find(|w| w.contains("could not be rebuilt"))
            .cloned()
            .unwrap_or_default();
        return Err(PositionError::new(format!(
            "That is a live-game fight paste the engine could not rebuild. {why}"
        )));
    }
    Err(PositionError::new(
        "That state has no fight in it. Paste an engine combat state (or start a New fight) to \
         load it here.",
    ))
}

// ---------------------------------------------------------------------------
// First moves and search
// ---------------------------------------------------------------------------

fn default_name_of(_kind: &str, ident: &str) -> String {
    title_case(ident)
}

/// `FUZZY_WURM_CRAWLER` / `FuzzyWurmCrawler` → `Fuzzy Wurm Crawler`.
#[must_use]
pub fn title_case(ident: &str) -> String {
    let snake = if !ident.contains('_') && ident.chars().any(|c| c.is_ascii_lowercase()) {
        camel_to_upper_snake(ident)
    } else {
        ident.to_string()
    };
    snake
        .to_ascii_lowercase()
        .split('_')
        .filter(|w| !w.is_empty())
        .map(|w| {
            let mut c = w.chars();
            match c.next() {
                Some(f) => f.to_ascii_uppercase().to_string() + c.as_str(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Human label for an engine legal action against the board it applies to.
pub fn action_label(action: &Value, wire: &Value, name_of: &NameOf<'_>) -> String {
    let kind = action.get("action").and_then(Value::as_str).unwrap_or("");
    // The engine's `target_index` is an encounter SLOT (`Action::PlayCard {
    // target }`), so it indexes the whole encounter, dead slots included —
    // not the alive-filtered list (which shifts left past a dead enemy and
    // named the wrong one). The name is the board label over the enemies the
    // board shows (neither dead nor escaped) in slot order
    // ([`crate::enemy_labels`]): "Two-Tailed Rat #2".
    let alive: Vec<(usize, String)> = wire
        .get("encounter")
        .and_then(Value::as_array)
        .map(|es| {
            es.iter()
                .enumerate()
                .filter(|(_, m)| !truthy(m.get("is_dead")) && !truthy(m.get("is_escaped")))
                .filter_map(|(slot, m)| {
                    m.get("kind")
                        .and_then(Value::as_str)
                        .map(|k| (slot, name_of("monster", k)))
                })
                .collect()
        })
        .unwrap_or_default();
    let names: Vec<&str> = alive.iter().map(|(_, n)| n.as_str()).collect();
    let labels = crate::enemy_labels::labels_from_names(&names);
    let target = || -> String {
        // `int(t)` in the Python twin: a numeric string resolves too.
        let t = action.get("target_index").filter(|v| !v.is_null());
        match t.map(|v| {
            v.as_i64()
                .or_else(|| v.as_str().and_then(|s| s.trim().parse().ok()))
        }) {
            None => String::new(),
            Some(None) => format!(" → enemy {}", crate::render::py::opt_str(t)),
            Some(Some(t)) => match alive
                .iter()
                .position(|(slot, _)| usize::try_from(t).is_ok_and(|t| t == *slot))
            {
                Some(i) => format!(" → {}", labels[i]),
                None => format!(" → enemy {t}"),
            },
        }
    };
    match kind {
        "play" => {
            let hand = wire
                .get("deck")
                .and_then(|d| d.get("hand"))
                .and_then(Value::as_array);
            let ci = action.get("card_index").and_then(Value::as_i64);
            let name = match (hand, ci) {
                (Some(h), Some(i)) if (i as usize) < h.len() => {
                    let card = &h[i as usize];
                    let base = name_of("card", &normalize_card_id(as_str(card.get("id"), "")));
                    if truthy(card.get("upgraded")) {
                        format!("{base}+")
                    } else {
                        base
                    }
                }
                _ => format!("card {}", ci.map_or("?".to_string(), |i| i.to_string())),
            };
            format!("Play {name}{}", target())
        }
        "end_turn" => "End turn".to_string(),
        "potion" | "drink_potion" => {
            let potions = wire
                .get("player")
                .and_then(|p| p.get("potions"))
                .and_then(Value::as_array);
            let slot = action.get("slot").and_then(Value::as_i64);
            let name = match (potions, slot) {
                (Some(ps), Some(s)) if (s as usize) < ps.len() => {
                    name_of("potion", as_str(Some(&ps[s as usize]), ""))
                }
                _ => format!("potion {}", slot.map_or("?".to_string(), |s| s.to_string())),
            };
            format!("Drink {name}{}", target())
        }
        "channel" => format!("Channel {}{}", as_str(action.get("orb"), ""), target()),
        "evoke" => format!("Evoke ×{}", as_i64_or(action.get("times"), 1)),
        "choose" => format!("Choose #{}", as_i64_or(action.get("index"), 0)),
        other => other.to_string(),
    }
}

/// Every legal move from this board and the position each one leads to. A move
/// that opens a choice is skipped with a note (the resulting board is
/// mid-resolution and cannot be swept from).
pub fn first_moves(wire: &Value, name_of: &NameOf<'_>) -> PResult<Vec<Value>> {
    let root = state_from_wire(wire)?;
    let mut out = Vec::new();
    let mut seen: std::collections::BTreeMap<String, i64> = std::collections::BTreeMap::new();
    for action in core_legal_actions(&root) {
        let action_value = combat_action_to_value(&action);
        let mut label = action_label(&action_value, wire, name_of);
        // Two Defends in hand are two legal actions with one label; a sweep keys
        // its cells on the label, so the second one is numbered.
        let count = seen.entry(label.clone()).or_insert(0);
        *count += 1;
        if *count > 1 {
            label = format!("{label} #{count}");
        }
        let mut clone = root.clone();
        let _ = sts2_core::apply(&mut clone, action);
        let mut note: Option<String> = None;
        if pending_choice(&clone).is_some() || clone.pending_play.is_some() {
            note = Some("opens a choice; cannot sweep from mid-resolution".to_string());
        }
        let (terminal, _) = terminal_value(&clone);
        let mut row = Map::new();
        row.insert("label".into(), json!(label));
        row.insert("action".into(), action_value);
        row.insert("terminal".into(), json!(terminal));
        if note.is_none() {
            let after = state_to_wire(&clone)?;
            match position_from_wire(&after, None, None) {
                Ok((pos, _)) => {
                    row.insert("position".into(), pos);
                }
                Err(e) => {
                    note = Some(e.to_string());
                    row.insert("position".into(), Value::Null);
                }
            }
            row.insert("wire".into(), after);
        }
        row.insert("note".into(), note.map_or(Value::Null, |n| json!(n)));
        out.push(Value::Object(row));
    }
    Ok(out)
}

/// Per-move MCTS verdict for the board — the sandbox's "search" card. Same
/// engine call the workbench hint uses, so the numbers agree with the board.
pub fn mcts_eval(wire: &Value, config: &Value, name_of: &NameOf<'_>) -> PResult<Value> {
    use sts2_wire::emit::mcts::{recommend_value, RecommendParams};
    let cs = state_from_wire(wire)?;
    let params = RecommendParams::from_config(config, "mcts_recommend")
        .map_err(|e| PositionError::new(format!("the engine rejected this position: {e}")))?;
    let sims = config.get("sims").and_then(Value::as_i64).unwrap_or(1000);
    let seed = config.get("seed").and_then(Value::as_i64).unwrap_or(0);
    let result = recommend_value(&cs, &params, true);
    let mut rows: Vec<Value> = Vec::new();
    for entry in result
        .get("actions")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let action = entry.get("action").cloned().unwrap_or(Value::Null);
        let q = entry
            .get("q")
            .and_then(Value::as_f64)
            .or_else(|| entry.get("value").and_then(Value::as_f64))
            .unwrap_or(0.0);
        rows.push(json!({
            "label": action_label(&action, wire, name_of),
            "action": action,
            "win_prob": entry.get("p_win").and_then(Value::as_f64).unwrap_or(0.0),
            "q": q,
            "visits": entry.get("visits").and_then(Value::as_i64).unwrap_or(0),
            "is_best": truthy(entry.get("is_best")),
            "low_confidence": truthy(entry.get("low_confidence")),
        }));
    }
    rows.sort_by(|a, b| {
        let key = |r: &Value| {
            (
                !truthy(r.get("is_best")),
                -(r.get("win_prob").and_then(Value::as_f64).unwrap_or(0.0)),
                -(r.get("visits").and_then(Value::as_i64).unwrap_or(0) as f64),
            )
        };
        key(a)
            .partial_cmp(&key(b))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(json!({"rows": rows, "sims": sims, "seed": seed}))
}

/// The move table for a monster kind (`[{id, initial, intent}]`) — the shared
/// engine move machine, formatted for the sandbox. See [`crate::simlab_catalog`].
fn monster_move_table(kind: &str, ascension: i64) -> PResult<Vec<Value>> {
    crate::simlab_catalog::monster_moves(kind, ascension)
        .map_err(PositionError::new)
        .map(|v| v.as_array().cloned().unwrap_or_default())
}

/// The default title-case name resolver, for callers with no catalog handy.
pub fn default_names() -> Box<NameOf<'static>> {
    Box::new(default_name_of)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simlab_catalog::{name_of, starter_for};

    fn names() -> impl Fn(&str, &str) -> String {
        |k: &str, i: &str| name_of(k, i)
    }

    #[test]
    fn action_label_targets_by_slot_and_numbers_duplicated_enemies() {
        // Slot 1 is dead: the engine's target is the encounter SLOT, so
        // target 2 is the second live rat — the alive-filtered list's [2]
        // would have been the Louse.
        let wire = json!({
            "deck": {"hand": [{"id": "STRIKE_R"}]},
            "encounter": [
                {"kind": "TWO_TAILED_RAT"},
                {"kind": "TWO_TAILED_RAT", "is_dead": true},
                {"kind": "TWO_TAILED_RAT"},
                {"kind": "LOUSE"},
            ],
        });
        let label = |t: i64| {
            let action = json!({"action": "play", "card_index": 0, "target_index": t});
            action_label(&action, &wire, &default_name_of)
        };
        assert_eq!(label(0), "Play Strike R → Two Tailed Rat #1");
        assert_eq!(label(2), "Play Strike R → Two Tailed Rat #2");
        assert_eq!(label(3), "Play Strike R → Louse");
        assert_eq!(label(1), "Play Strike R → enemy 1");
        // A numeric-string target resolves like the int (Python's `int(t)`).
        let action = json!({"action": "play", "card_index": 0, "target_index": "2"});
        assert_eq!(
            action_label(&action, &wire, &default_name_of),
            "Play Strike R → Two Tailed Rat #2"
        );
    }

    #[test]
    fn action_label_skips_escaped_enemies() {
        // The sandbox board drops an escaped Thief, so the one left is bare.
        let wire = json!({
            "deck": {"hand": [{"id": "STRIKE_R"}]},
            "encounter": [{"kind": "THIEF", "is_escaped": true}, {"kind": "THIEF"}],
        });
        let action = json!({"action": "play", "card_index": 0, "target_index": 1});
        assert_eq!(
            action_label(&action, &wire, &default_name_of),
            "Play Strike R → Thief"
        );
    }

    /// A real Ironclad turn-1 board against a filler encounter, via the shared
    /// scenario builder — the fixture every round-trip test starts from.
    fn fresh() -> BuiltPosition {
        fresh_in(sts2_core::GameVersion::default())
    }

    fn fresh_in(version: sts2_core::GameVersion) -> BuiltPosition {
        let (hp, deck, relic, _) = starter_for("Ironclad");
        new_position(
            version,
            "Ironclad",
            Some(MONSTER_BASE),
            None,
            0,
            1,
            None,
            None,
            None,
            &[],
            None,
            hp,
            &deck,
            Some(relic),
        )
        .expect("new_position builds")
    }

    /// GH #877: a sandbox position remembers its game build through the
    /// wire round trip; a default-build position never names one.
    #[test]
    fn a_position_carries_its_game_version_through_the_engine() {
        let base = fresh();
        assert!(base.position.get("game_version").is_none());
        assert!(base.wire.get("game_version").is_none());
        assert!(wire_from_position(&base.position)
            .unwrap()
            .get("game_version")
            .is_none());

        let v111 = sts2_core::GameVersion::V0_111_0;
        let built = fresh_in(v111);
        assert_eq!(built.position["game_version"], json!("v0.111.0"));
        assert_eq!(built.wire["game_version"], json!("v0.111.0"));
        assert_eq!(
            wire_from_position(&built.position).unwrap()["game_version"],
            json!("v0.111.0")
        );
        let again = build_position(&built.position).expect("rebuilds");
        assert_eq!(again.position["game_version"], json!("v0.111.0"));
        assert_eq!(position_game_version(&again.position).unwrap(), v111);

        let mut bad = built.position.clone();
        bad["game_version"] = json!("v9");
        assert!(wire_from_position(&bad).is_err());
    }

    #[test]
    fn normalize_card_id_handles_wire_dialects() {
        assert_eq!(normalize_card_id("status:Slimed"), "SLIMED");
        assert_eq!(normalize_card_id("curse:Clumsy"), "CLUMSY");
        assert_eq!(normalize_card_id("STRIKE_IRONCLAD"), "STRIKE_IRONCLAD");
        assert_eq!(normalize_card_id(" Bash "), "BASH");
    }

    #[test]
    fn camel_to_upper_snake_matches_the_engine_rule() {
        assert_eq!(
            camel_to_upper_snake("FuzzyWurmCrawler"),
            "FUZZY_WURM_CRAWLER"
        );
        assert_eq!(title_case("FUZZY_WURM_CRAWLER"), "Fuzzy Wurm Crawler");
    }

    #[test]
    fn new_position_deals_a_playable_turn_one() {
        let built = fresh();
        let pos = &built.position;
        assert_eq!(pos["character"], json!("Ironclad"));
        assert_eq!(pos["player"]["hp"], json!(80));
        assert_eq!(pos["player"]["maxHp"], json!(80));
        // The starter deck was dealt into piles (10 cards total).
        let total: usize = ["hand", "draw", "discard", "exhaust"]
            .iter()
            .map(|p| pos["piles"][p].as_array().map_or(0, Vec::len))
            .sum();
        assert_eq!(total, 10);
        assert!(!pos["monsters"].as_array().unwrap().is_empty());
    }

    #[test]
    fn build_position_round_trips_a_dealt_board() {
        let built = fresh();
        // The position the engine handed back rebuilds without loss.
        let rebuilt = build_position(&built.position).expect("build round-trips");
        assert_eq!(
            rebuilt.position["player"]["hp"],
            built.position["player"]["hp"]
        );
        assert_eq!(rebuilt.wire["phase"], json!("PlayerPlayPhase"));
    }

    #[test]
    fn first_moves_lists_end_turn_and_plays() {
        let built = fresh();
        let name = names();
        let moves = first_moves(&built.wire, &name).expect("moves");
        let labels: Vec<&str> = moves.iter().filter_map(|m| m["label"].as_str()).collect();
        assert!(labels.contains(&"End turn"), "labels: {labels:?}");
        assert!(
            labels.iter().any(|l| l.starts_with("Play ")),
            "labels: {labels:?}"
        );
    }

    #[test]
    fn mcts_eval_ranks_the_moves() {
        let built = fresh();
        let name = names();
        let out = mcts_eval(&built.wire, &json!({"sims": 32, "seed": 1}), &name).expect("mcts");
        assert_eq!(out["sims"], json!(32));
        assert!(!out["rows"].as_array().unwrap().is_empty());
    }

    #[test]
    fn position_from_paste_accepts_a_position_doc_and_a_wire_dict() {
        let built = fresh();
        // A position document round-trips verbatim.
        let text = serde_json::to_string(&built.position).unwrap();
        let (pos, _, source) = position_from_paste(&text).expect("position paste");
        assert_eq!(source, "position");
        assert_eq!(pos["character"], json!("Ironclad"));
        // The engine wire dict imports through position_from_wire.
        let wire_text = serde_json::to_string(&built.wire).unwrap();
        let (_, _, source) = position_from_paste(&wire_text).expect("wire paste");
        assert_eq!(source, "engine wire");
    }

    #[test]
    fn position_from_paste_rejects_junk() {
        assert!(position_from_paste("").is_err());
        assert!(position_from_paste("{\"not\": \"a fight\"}").is_err());
    }

    #[test]
    fn wire_from_position_rejects_an_empty_board() {
        let mut pos = empty_position("Ironclad", 0);
        // No cards, no enemies — the engine could never accept it.
        assert!(wire_from_position(&pos).is_err());
        pos["piles"]["hand"] = json!([{"card": "STRIKE_IRONCLAD"}]);
        // Cards but still no enemies.
        assert!(wire_from_position(&pos).is_err());
    }
}
