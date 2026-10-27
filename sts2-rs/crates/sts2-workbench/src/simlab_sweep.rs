//! SimLab distribution sweeps in the engine (GH #707): the port of
//! `sts2-pycli/python/simlab/{specs,runner,telemetry,intel}.py`, the
//! `simlab/agents/` package and the sweep half of `simlab/web/{app,jobs}.py`.
//!
//! On the desktop a sweep is a background job over a process pool with its
//! state on disk. The wasm is single-threaded and has no threads, files or
//! clock of its own, so here a sweep is a RESUMABLE job the host steps:
//! [`SweepJob::run_some`] plays a few trials and returns; the Web Worker
//! calls it between the page's other requests (a status poll, the sandbox's
//! own search), yields to its message queue, and calls again until the job
//! ends. Cancellation is a flag the stop route sets and the next step
//! honours. Seed pairing, the strategies, the row schema and the bucket
//! statistics are the Python code line for line, so a browser sweep and a
//! desktop sweep of one spec produce the same rows — `tests/sweep_parity.rs`
//! pins that against a dump of the Python runner.

use std::collections::BTreeMap;

use serde_json::{json, Map, Value};
use smallvec::SmallVec;
use sts2_core::{Action, CombatState};
use sts2_wire::emit::combat_state_to_value;
use sts2_wire::state_dict::{parse_state_dict, registry_names_for, state_from_dict_impl};

use crate::deck::PyRandom;

/// `runner.MAX_ACTIONS`: a trial that has not ended after this many actions
/// is `timed_out`.
pub const MAX_ACTIONS: u32 = 2000;

/// `jobs.DEFAULT_MAX_RUNTIME_S`: the wall-clock cap on one sweep.
pub const DEFAULT_MAX_RUNTIME_S: f64 = 600.0;

/// `specs.DECK_PRESETS`, in the Python dict's order.
pub fn deck_presets() -> Vec<(&'static str, Vec<&'static str>)> {
    fn rep(card: &'static str, n: usize) -> Vec<&'static str> {
        vec![card; n]
    }
    let mut starter = rep("STRIKE_IRONCLAD", 5);
    starter.extend(rep("DEFEND_IRONCLAD", 4));
    starter.push("BASH");
    let mut upgraded = rep("STRIKE_IRONCLAD+1", 5);
    upgraded.extend(rep("DEFEND_IRONCLAD+1", 4));
    upgraded.push("BASH+1");
    let mut aggro = rep("STRIKE_IRONCLAD", 4);
    aggro.extend(rep("ANGER", 2));
    aggro.extend([
        "TWIN_STRIKE",
        "UPPERCUT",
        "POMMEL_STRIKE",
        "THUNDERCLAP",
        "HEMOKINESIS",
    ]);
    aggro.extend(rep("DEFEND_IRONCLAD", 2));
    let mut turtle = rep("DEFEND_IRONCLAD", 4);
    turtle.extend(rep("SHRUG_IT_OFF", 2));
    turtle.extend(rep("IRON_WAVE", 2));
    turtle.push("IMPERVIOUS");
    turtle.extend(rep("STRIKE_IRONCLAD", 2));
    turtle.push("BASH");
    let mut scaling = vec!["DEMON_FORM_CARD"];
    scaling.extend(rep("INFLAME", 2));
    scaling.extend(["WHIRLWIND", "UPPERCUT"]);
    scaling.extend(rep("STRIKE_IRONCLAD", 3));
    scaling.extend(rep("DEFEND_IRONCLAD", 3));
    scaling.push("BASH");
    vec![
        ("ironclad_starter", starter),
        ("starter_upgraded", upgraded),
        ("aggro_rush", aggro),
        ("turtle", turtle),
        ("scaling", scaling),
    ]
}

/// `sorted(AGENTS)`: the strategy names the form offers.
pub const STRATEGIES: [&str; 5] = [
    "attack_first",
    "block_first",
    "mcts",
    "min_overblock",
    "random",
];

/// `specs.IMPORTED_ENCOUNTER`: the encounter column of a start-state cell.
pub const IMPORTED_ENCOUNTER: &str = "imported";

/// `specs._MONSTER_BASE`: the base encounter id a `monsters:` lineup rides on.
const MONSTER_BASE: &str = "MAWLER_NORMAL";
const MAX_DECK: usize = 100;

/// Python `round(x, n)` — correctly rounded on the exact binary value.
fn py_round(x: f64, n: usize) -> f64 {
    format!("{x:.n$}").parse().unwrap_or(x)
}

// ── the spec (`specs.py`) ─────────────────────────────────────────────────

/// `specs.SweepSpec`: a full sweep request.
#[derive(Debug, Clone)]
pub struct SweepSpec {
    pub name: String,
    pub encounters: Vec<String>,
    pub ascensions: Vec<i64>,
    pub strategies: Vec<String>,
    pub n_seeds: u32,
    pub decks: Vec<(String, Vec<String>)>,
    pub hp: i32,
    pub max_hp: i32,
    pub mcts_sims: u32,
    pub mcts_depth: u32,
    pub start_states: Vec<(String, String)>,
    pub positions: Vec<(String, String)>,
}

fn str_list(v: Option<&Value>, key: &str) -> Result<Vec<String>, String> {
    let Some(v) = v else {
        return Err(format!("missing required field {key:?}"));
    };
    let arr = v
        .as_array()
        .ok_or_else(|| format!("{key} must be a list"))?;
    arr.iter()
        .map(|x| match x {
            Value::String(s) => Ok(s.clone()),
            other => Ok(other.to_string()),
        })
        .collect()
}

fn pair_list(v: Option<&Value>, key: &str) -> Result<Vec<(String, String)>, String> {
    let Some(v) = v else {
        return Ok(Vec::new());
    };
    let arr = v
        .as_array()
        .ok_or_else(|| format!("{key} must be a list of [label, text] pairs"))?;
    arr.iter()
        .map(|p| {
            let items = p
                .as_array()
                .filter(|a| a.len() == 2)
                .ok_or_else(|| format!("{key} entries must be [label, text] pairs"))?;
            let as_str = |x: &Value| match x {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            Ok((as_str(&items[0]), as_str(&items[1])))
        })
        .collect()
}

fn int_of(v: Option<&Value>, key: &str, default: i64) -> Result<i64, String> {
    match v {
        None | Some(Value::Null) => Ok(default),
        Some(Value::Number(n)) => n
            .as_i64()
            .or_else(|| n.as_f64().map(|f| f as i64))
            .ok_or_else(|| format!("{key} must be an integer")),
        Some(Value::Bool(b)) => Ok(i64::from(*b)),
        Some(Value::String(s)) => s
            .trim()
            .parse::<i64>()
            .map_err(|_| format!("{key} must be an integer")),
        Some(_) => Err(format!("{key} must be an integer")),
    }
}

impl SweepSpec {
    /// `SweepSpec.from_json` + `__post_init__`: the legacy `deck_preset` maps
    /// to one named deck; unknown fields, empty / duplicate / slashed deck
    /// names and start-state labels are refused like the dataclass does.
    pub fn from_json(d: &Value) -> Result<Self, String> {
        let obj = d.as_object().ok_or("sweep spec must be an object")?;
        const KNOWN: [&str; 13] = [
            "name",
            "encounters",
            "ascensions",
            "strategies",
            "n_seeds",
            "decks",
            "deck_preset",
            "hp",
            "max_hp",
            "mcts_sims",
            "mcts_depth",
            "start_states",
            "positions",
        ];
        if let Some(k) = obj.keys().find(|k| !KNOWN.contains(&k.as_str())) {
            return Err(format!("unexpected field {k:?}"));
        }
        let name = match obj.get("name") {
            Some(Value::String(s)) => s.clone(),
            Some(other) => other.to_string(),
            None => return Err("missing required field \"name\"".to_string()),
        };
        let encounters = str_list(obj.get("encounters"), "encounters")?;
        let ascensions = obj
            .get("ascensions")
            .ok_or("missing required field \"ascensions\"")?
            .as_array()
            .ok_or("ascensions must be a list")?
            .iter()
            .map(|a| int_of(Some(a), "ascensions", 0))
            .collect::<Result<Vec<_>, _>>()?;
        let strategies = str_list(obj.get("strategies"), "strategies")?;
        let n_seeds = int_of(obj.get("n_seeds"), "n_seeds", 0)?;
        let n_seeds = u32::try_from(n_seeds.max(0)).unwrap_or(u32::MAX);
        let decks: Vec<(String, Vec<String>)> = match (obj.get("decks"), obj.get("deck_preset")) {
            (Some(Value::Array(arr)), _) => arr
                .iter()
                .map(|p| {
                    let items = p
                        .as_array()
                        .filter(|a| a.len() == 2)
                        .ok_or("decks entries must be [name, [cards…]] pairs")?;
                    let name = match &items[0] {
                        Value::String(s) => s.clone(),
                        other => other.to_string(),
                    };
                    let cards = str_list(Some(&items[1]), "decks")?;
                    Ok::<_, String>((name, cards))
                })
                .collect::<Result<_, _>>()?,
            (Some(_), _) => return Err("decks must be a list".to_string()),
            (None, Some(preset)) => {
                let preset = preset.as_str().unwrap_or("");
                let presets = deck_presets();
                let Some((_, cards)) = presets.iter().find(|(n, _)| *n == preset) else {
                    let valid: Vec<&str> = presets.iter().map(|(n, _)| *n).collect();
                    return Err(format!(
                        "unknown deck_preset {preset:?}; valid presets: {valid:?}"
                    ));
                };
                vec![(
                    preset.to_string(),
                    cards.iter().map(|c| (*c).to_string()).collect(),
                )]
            }
            (None, None) => {
                let presets = deck_presets();
                let (n, cards) = &presets[0];
                vec![(
                    (*n).to_string(),
                    cards.iter().map(|c| (*c).to_string()).collect(),
                )]
            }
        };
        let hp = i32::try_from(int_of(obj.get("hp"), "hp", 80)?).unwrap_or(80);
        let max_hp = i32::try_from(int_of(obj.get("max_hp"), "max_hp", 80)?).unwrap_or(80);
        let mcts_sims =
            u32::try_from(int_of(obj.get("mcts_sims"), "mcts_sims", 400)?.max(0)).unwrap_or(400);
        let mcts_depth =
            u32::try_from(int_of(obj.get("mcts_depth"), "mcts_depth", 200)?.max(0)).unwrap_or(200);
        let start_states = pair_list(obj.get("start_states"), "start_states")?;
        let positions = pair_list(obj.get("positions"), "positions")?;

        let names: Vec<&str> = decks.iter().map(|(n, _)| n.as_str()).collect();
        for n in &names {
            if n.is_empty() || n.contains('/') {
                return Err(format!(
                    "invalid deck name {n:?} (must be non-empty and contain no '/')"
                ));
            }
        }
        let dupes = duplicates(&names);
        if !dupes.is_empty() {
            return Err(format!("duplicate deck names in sweep: {dupes:?}"));
        }
        let labels: Vec<&str> = start_states.iter().map(|(n, _)| n.as_str()).collect();
        for n in &labels {
            if n.is_empty() || n.contains('/') {
                return Err(format!(
                    "invalid start-state label {n:?} (must be non-empty and contain no '/')"
                ));
            }
        }
        let dupes = duplicates(&labels);
        if !dupes.is_empty() {
            return Err(format!("duplicate start-state labels in sweep: {dupes:?}"));
        }
        Ok(Self {
            name,
            encounters,
            ascensions,
            strategies,
            n_seeds,
            decks,
            hp,
            max_hp,
            mcts_sims,
            mcts_depth,
            start_states,
            positions,
        })
    }

    /// `SweepSpec.to_json`.
    #[must_use]
    pub fn to_json(&self) -> Value {
        json!({
            "name": self.name,
            "encounters": self.encounters,
            "ascensions": self.ascensions,
            "strategies": self.strategies,
            "n_seeds": self.n_seeds,
            "decks": self.decks.iter().map(|(n, c)| json!([n, c])).collect::<Vec<_>>(),
            "hp": self.hp,
            "max_hp": self.max_hp,
            "mcts_sims": self.mcts_sims,
            "mcts_depth": self.mcts_depth,
            "start_states": self.start_states.iter().map(|(l, b)| json!([l, b])).collect::<Vec<_>>(),
            "positions": self.positions.iter().map(|(l, b)| json!([l, b])).collect::<Vec<_>>(),
        })
    }

    /// `SweepSpec.cells()`: one (scenario, strategy) bucket per cell, in the
    /// Python product order. A start-state sweep replaces the encounter /
    /// deck / ascension axes.
    pub fn cells(&self) -> Result<Vec<Cell>, String> {
        let mut out = Vec::new();
        if !self.start_states.is_empty() {
            for (label, blob) in &self.start_states {
                let wire: Value = serde_json::from_str(blob)
                    .map_err(|e| format!("start_state {label:?} is not JSON: {e}"))?;
                if !wire.is_object() {
                    return Err(
                        "start_state must be a JSON object (an engine wire dict)".to_string()
                    );
                }
                let ascension = start_state_ascension(&wire);
                for s in &self.strategies {
                    out.push(Cell {
                        encounter: IMPORTED_ENCOUNTER.to_string(),
                        ascension,
                        deck_name: label.clone(),
                        deck_cards: Vec::new(),
                        hp: self.hp,
                        max_hp: self.max_hp,
                        start_state: Some(std::rc::Rc::new(wire.clone())),
                        strategy: s.clone(),
                    });
                }
            }
            return Ok(out);
        }
        for e in &self.encounters {
            for a in &self.ascensions {
                for (dname, dcards) in &self.decks {
                    for s in &self.strategies {
                        out.push(Cell {
                            encounter: e.clone(),
                            ascension: *a,
                            deck_name: dname.clone(),
                            deck_cards: dcards.clone(),
                            hp: self.hp,
                            max_hp: self.max_hp,
                            start_state: None,
                            strategy: s.clone(),
                        });
                    }
                }
            }
        }
        Ok(out)
    }

    /// The submit-time checks of `POST /api/sweeps` (`app.py`): decks and
    /// encounters against the registries (a start-state sweep has neither),
    /// the strategies against the agent registry, then the cells
    /// materialised so a bad start state fails here rather than mid-run.
    pub fn validate(&self) -> Result<Vec<Cell>, String> {
        if self.start_states.is_empty() {
            for (name, cards) in &self.decks {
                validate_deck(name, cards)?;
            }
            for enc in &self.encounters {
                validate_encounter(enc)?;
            }
        }
        for s in &self.strategies {
            if !STRATEGIES.contains(&s.as_str()) {
                return Err(format!(
                    "unknown strategy {s:?}; valid strategies: {STRATEGIES:?}"
                ));
            }
        }
        self.cells()
    }
}

fn duplicates(names: &[&str]) -> Vec<String> {
    let mut seen = std::collections::BTreeSet::new();
    let mut dupes = std::collections::BTreeSet::new();
    for n in names {
        if !seen.insert(*n) {
            dupes.insert((*n).to_string());
        }
    }
    dupes.into_iter().collect()
}

/// `specs.parse_encounter`: `"monsters:A,B"` → the base id + the lineup.
#[must_use]
pub fn parse_encounter(encounter: &str) -> (String, Vec<String>) {
    if let Some(rest) = encounter.strip_prefix("monsters:") {
        let names = rest
            .split(',')
            .map(str::trim)
            .filter(|m| !m.is_empty())
            .map(str::to_string)
            .collect();
        return (MONSTER_BASE.to_string(), names);
    }
    (encounter.to_string(), Vec::new())
}

/// `specs.validate_deck`.
pub fn validate_deck(name: &str, cards: &[String]) -> Result<(), String> {
    if name.is_empty() || name.contains('/') {
        return Err(format!(
            "invalid deck name {name:?} (must be non-empty and contain no '/')"
        ));
    }
    if cards.is_empty() {
        return Err(format!("deck {name:?} is empty"));
    }
    if cards.len() > MAX_DECK {
        return Err(format!(
            "deck {name:?} has {} cards (max {MAX_DECK})",
            cards.len()
        ));
    }
    let known = registry_names_for("card").unwrap_or_default();
    for c in cards {
        let base = c.split_once('+').map_or(c.as_str(), |(b, _)| b);
        if !known.iter().any(|k| k == base) {
            return Err(format!("deck {name:?}: unknown card id {c:?}"));
        }
    }
    Ok(())
}

/// `specs.validate_encounter`: a `monsters:` lineup against the registry
/// (a plain id is the engine's to validate at build time).
pub fn validate_encounter(encounter: &str) -> Result<(), String> {
    let (_, monsters) = parse_encounter(encounter);
    if monsters.is_empty() && encounter.starts_with("monsters:") {
        return Err("monsters: entry names no monsters".to_string());
    }
    let known = registry_names_for("monster").unwrap_or_default();
    for m in &monsters {
        if m.contains('/') {
            return Err(format!("invalid monster name {m:?}"));
        }
        if !known.iter().any(|k| k == m) {
            return Err(format!("unknown monster {m:?}"));
        }
    }
    Ok(())
}

/// `specs.start_state_ascension`: `rng.ascension` of the pasted board, else 0.
fn start_state_ascension(wire: &Value) -> i64 {
    match wire.get("rng").and_then(|r| r.get("ascension")) {
        Some(Value::Number(n)) => n.as_i64().unwrap_or(0),
        _ => 0,
    }
}

/// One `(scenario, strategy)` cell of a sweep (`specs.SweepCell` with the
/// `ScenarioSpec` flattened in).
#[derive(Debug, Clone)]
pub struct Cell {
    pub encounter: String,
    pub ascension: i64,
    pub deck_name: String,
    pub deck_cards: Vec<String>,
    pub hp: i32,
    pub max_hp: i32,
    /// The parsed `start_state` wire dict, shared by every trial of the cell.
    pub start_state: Option<std::rc::Rc<Value>>,
    pub strategy: String,
}

impl Cell {
    /// `specs.build_combat`: the seeded fight for this cell — byte-identical
    /// for the same `(cell, seed)`, which is what makes strategies
    /// seed-paired.
    pub fn build_combat(&self, seed: u32) -> Result<CombatState, String> {
        if let Some(wire) = &self.start_state {
            return build_combat_from_state(wire, seed);
        }
        let (enc_id, monsters) = parse_encounter(&self.encounter);
        let ascension = u8::try_from(self.ascension.clamp(0, 255)).unwrap_or(0);
        sts2_wire::scenario::build_scenario_state_for_version(
            // SimLab has no session: positions are the default build's.
            sts2_core::GameVersion::default(),
            u64::from(seed),
            "Ironclad",
            &self.deck_cards,
            self.hp,
            self.max_hp,
            &[],
            &[],
            &enc_id,
            ascension,
            if monsters.is_empty() {
                None
            } else {
                Some(&monsters)
            },
            None,
        )
    }
}

/// `specs.build_combat_from_state`: the pasted board with this trial's own
/// future dice — every RNG stream re-seeded from `seed` and rewound to zero.
fn build_combat_from_state(wire: &Value, seed: u32) -> Result<CombatState, String> {
    let mut wire = wire.clone();
    let mut rng = wire
        .get("rng")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    rng.insert("seed".to_string(), json!(seed));
    rng.insert("run_seed".to_string(), json!(seed));
    rng.remove("counters");
    if let Some(m) = wire.as_object_mut() {
        m.insert("rng".to_string(), Value::Object(rng));
    }
    let parsed = parse_state_dict(&&wire).map_err(|e| e.to_string())?;
    state_from_dict_impl(&parsed).map_err(|e| e.to_string())
}

// ── the agents (`agents/`, `intel.py`) ────────────────────────────────────

/// `intel.ActionEffect`: one action's effect measured on a clone.
#[derive(Debug, Clone, Copy)]
struct ActionEffect {
    block_gain: i64,
    enemy_hp_loss: i64,
    energy_spent: i64,
    kills: i64,
}

fn player_field(state: &Value, key: &str) -> i64 {
    state
        .get("player")
        .and_then(|p| p.get(key))
        .and_then(Value::as_i64)
        .unwrap_or(0)
}

fn encounter_rows(state: &Value) -> &[Value] {
    state
        .get("encounter")
        .and_then(Value::as_array)
        .map_or(&[], Vec::as_slice)
}

fn is_dead(e: &Value) -> bool {
    e.get("is_dead").and_then(Value::as_bool).unwrap_or(false)
}

/// `intel.incoming_damage`: telegraphed attack damage across alive enemies.
fn incoming_damage(state: &Value) -> i64 {
    let mut total = 0;
    for e in encounter_rows(state).iter().filter(|e| !is_dead(e)) {
        for intent in e
            .get("intents")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            if intent.get("kind").and_then(Value::as_str) == Some("Attack") {
                let amount = intent.get("amount").and_then(Value::as_i64).unwrap_or(0);
                let times = intent.get("times").and_then(Value::as_i64).unwrap_or(1);
                total += amount * times;
            }
        }
    }
    total
}

fn enemy_hp_total(state: &Value) -> i64 {
    encounter_rows(state)
        .iter()
        .map(|e| e.get("hp").and_then(Value::as_i64).unwrap_or(0).max(0))
        .sum()
}

fn alive_count(state: &Value) -> i64 {
    encounter_rows(state).iter().filter(|e| !is_dead(e)).count() as i64
}

fn wire_of(cs: &CombatState) -> Result<Value, String> {
    combat_state_to_value(cs).map_err(|e| e.to_string())
}

/// `intel.probe_action`: apply on a clone, diff block / enemy HP / energy /
/// kills against `before`.
fn probe_action(cs: &CombatState, action: Action, before: &Value) -> Result<ActionEffect, String> {
    let mut clone = cs.clone();
    let _ = sts2_core::apply(&mut clone, action);
    let after = wire_of(&clone)?;
    Ok(ActionEffect {
        block_gain: player_field(&after, "block") - player_field(before, "block"),
        enemy_hp_loss: enemy_hp_total(before) - enemy_hp_total(&after),
        energy_spent: player_field(before, "energy") - player_field(&after, "energy"),
        kills: alive_count(before) - alive_count(&after),
    })
}

fn is_choose(a: &Action) -> bool {
    matches!(a, Action::ResolveChoice { .. })
}

fn is_play(a: &Action) -> bool {
    matches!(a, Action::PlayCard { .. })
}

/// `_best_attack`: kills first, then raw damage, then cheaper.
fn best_attack(probed: &[(Action, ActionEffect)]) -> Option<Action> {
    let mut attacks: Vec<&(Action, ActionEffect)> =
        probed.iter().filter(|(_, e)| e.enemy_hp_loss > 0).collect();
    if attacks.is_empty() {
        return None;
    }
    attacks.sort_by_key(|(_, e)| (-e.kills, -e.enemy_hp_loss, e.energy_spent));
    Some(attacks[0].0.clone())
}

/// `_blocks`: pure block plays (block up, no enemy damage).
fn blocks(probed: &[(Action, ActionEffect)]) -> Vec<(Action, ActionEffect)> {
    probed
        .iter()
        .filter(|(_, e)| e.block_gain > 0 && e.enemy_hp_loss == 0)
        .cloned()
        .collect()
}

/// Biggest block first, cheaper on ties (`blocks.sort(key=(-block_gain, energy_spent))`).
fn sort_blocks(blocks: &mut [(Action, ActionEffect)]) {
    blocks.sort_by_key(|(_, e)| (-e.block_gain, e.energy_spent));
}

/// `itertools.combinations(range(n), r)` in lexicographic order.
fn for_each_combination(n: usize, r: usize, mut f: impl FnMut(&[usize])) {
    if r == 0 || r > n {
        return;
    }
    let mut idx: Vec<usize> = (0..r).collect();
    loop {
        f(&idx);
        // Advance to the next combination.
        let mut i = r;
        loop {
            if i == 0 {
                return;
            }
            i -= 1;
            if idx[i] != i + n - r {
                break;
            }
            if i == 0 {
                return;
            }
        }
        idx[i] += 1;
        for j in i + 1..r {
            idx[j] = idx[j - 1] + 1;
        }
    }
}

/// `heuristics.pick_block_subset`: the largest block subset that does NOT
/// overblock, tie-broken by energy; overblock only when the waste is
/// smaller than the leak it prevents. Returns the chosen option indices.
fn pick_block_subset(options: &[(usize, i64, i64)], incoming: i64, energy: i64) -> Vec<usize> {
    let mut under_best: Vec<usize> = Vec::new();
    let mut under_score = (0i64, 0i64);
    let mut over_best: Vec<usize> = Vec::new();
    let mut over_score: Option<(i64, i64)> = None;
    for r in 1..=options.len() {
        for_each_combination(options.len(), r, |combo| {
            let cost: i64 = combo.iter().map(|&i| options[i].2).sum();
            if cost > energy {
                return;
            }
            let block: i64 = combo.iter().map(|&i| options[i].1).sum();
            if block <= incoming {
                let s = (block, -cost);
                if s > under_score {
                    under_score = s;
                    under_best = combo.to_vec();
                }
            } else {
                let o = (block - incoming, cost);
                if over_score.is_none_or(|best| o < best) {
                    over_score = Some(o);
                    over_best = combo.to_vec();
                }
            }
        });
    }
    if !under_best.is_empty() {
        return under_best;
    }
    if let Some((waste, _)) = over_score {
        if !over_best.is_empty() && waste < incoming {
            return over_best;
        }
    }
    Vec::new()
}

/// One decision's inputs (`agents.base.DecisionContext`), the wire snapshot
/// taken lazily like the Python property.
struct Decision<'a> {
    legal: &'a [Action],
    combat: &'a CombatState,
    state: Option<Value>,
}

impl Decision<'_> {
    fn state(&mut self) -> Result<&Value, String> {
        if self.state.is_none() {
            self.state = Some(wire_of(self.combat)?);
        }
        Ok(self.state.as_ref().expect("just set"))
    }

    fn first_choice(&self) -> Option<Action> {
        self.legal.iter().find(|a| is_choose(a)).cloned()
    }

    fn end_turn(&self) -> Result<Action, String> {
        self.legal
            .iter()
            .find(|a| matches!(a, Action::EndTurn))
            .cloned()
            .ok_or_else(|| "no end_turn in legal actions".to_string())
    }

    /// `_probed`: every play action probed against one before-snapshot.
    fn probed(&mut self) -> Result<Vec<(Action, ActionEffect)>, String> {
        let plays: Vec<Action> = self.legal.iter().filter(|a| is_play(a)).cloned().collect();
        let combat = self.combat;
        let before = self.state()?;
        plays
            .into_iter()
            .map(|a| probe_action(combat, a.clone(), before).map(|e| (a, e)))
            .collect()
    }
}

fn random_agent(d: &Decision<'_>, rng: &mut PyRandom) -> Action {
    let n = u32::try_from(d.legal.len()).unwrap_or(u32::MAX);
    d.legal[rng.randbelow(n) as usize].clone()
}

fn block_first(d: &mut Decision<'_>) -> Result<Action, String> {
    if let Some(c) = d.first_choice() {
        return Ok(c);
    }
    let probed = d.probed()?;
    let state = d.state()?;
    let deficit = incoming_damage(state) - player_field(state, "block");
    if deficit > 0 {
        let mut b = blocks(&probed);
        if !b.is_empty() {
            sort_blocks(&mut b);
            return Ok(b[0].0.clone());
        }
    }
    if let Some(atk) = best_attack(&probed) {
        return Ok(atk);
    }
    d.end_turn()
}

fn attack_first(d: &mut Decision<'_>) -> Result<Action, String> {
    if let Some(c) = d.first_choice() {
        return Ok(c);
    }
    let probed = d.probed()?;
    if let Some(atk) = best_attack(&probed) {
        return Ok(atk);
    }
    let mut b = blocks(&probed);
    if !b.is_empty() {
        sort_blocks(&mut b);
        return Ok(b[0].0.clone());
    }
    d.end_turn()
}

fn min_overblock(d: &mut Decision<'_>) -> Result<Action, String> {
    if let Some(c) = d.first_choice() {
        return Ok(c);
    }
    let probed = d.probed()?;
    let state = d.state()?;
    let deficit = incoming_damage(state) - player_field(state, "block");
    if deficit > 0 {
        let b = blocks(&probed);
        let options: Vec<(usize, i64, i64)> = b
            .iter()
            .enumerate()
            .map(|(i, (_, e))| (i, e.block_gain, e.energy_spent.max(0)))
            .collect();
        let chosen = pick_block_subset(&options, deficit, player_field(state, "energy"));
        if let Some(&first) = chosen.first() {
            return Ok(b[options[first].0].0.clone());
        }
    }
    if let Some(atk) = best_attack(&probed) {
        return Ok(atk);
    }
    d.end_turn()
}

/// `mcts_agent`: the engine's full MCTS (Texel + RichBoth) over the live
/// combat, exactly as `sts2_rs.Mcts(state, iters, seed, max_rollout_depth)`
/// builds it, seeded off the state hash so the pick is replayable.
fn mcts_agent(d: &Decision<'_>, sims: u32, depth: u32) -> Result<Action, String> {
    use sts2_mcts::{LeafEval, Mcts, MctsConfig, SharedPolicy};
    if d.legal.len() == 1 {
        return Ok(d.legal[0].clone());
    }
    let seed = sts2_core::state_hash(d.combat) & 0xFFFF_FFFF;
    let cfg = MctsConfig {
        c_puct: std::f32::consts::SQRT_2,
        max_simulations: sims,
        max_rollout_depth: depth,
        seed,
        use_puct_prior: false,
        ..MctsConfig::default()
    };
    let mut mcts = Mcts::with_policy(d.combat.clone(), cfg, SharedPolicy::uniform())
        .with_leaf_eval(LeafEval::default());
    let recommended = mcts.search();
    d.legal
        .iter()
        .find(|a| **a == recommended)
        .cloned()
        .ok_or_else(|| {
            format!(
                "AssertionError: mcts_agent recommendation matched no legal action: recommended={recommended:?}"
            )
        })
}

// ── one trial (`runner.play_combat`) ──────────────────────────────────────

/// A millisecond clock for `wall_ms` / `elapsed_s` (the host's `Date.now`
/// in the browser; `SystemTime` natively).
pub type Clock<'a> = &'a dyn Fn() -> f64;

/// `runner.play_combat`: drive one combat to terminal under `strategy`; the
/// telemetry row.
pub fn play_combat(
    cell: &Cell,
    seed: u32,
    sims: u32,
    depth: u32,
    clock: Clock<'_>,
) -> Result<Value, String> {
    let mut combat = cell.build_combat(seed)?;
    let mut rng = PyRandom::new(u64::from(seed));
    let hp_start = player_field(&wire_of(&combat)?, "hp");
    let t0 = clock();
    let mut actions = 0u32;
    let mut turns = 1u32;
    let mut stuck = false;
    while actions < MAX_ACTIONS {
        let (terminal, _) = sts2_core::terminal_value(&combat);
        if terminal {
            break;
        }
        let legal: SmallVec<[Action; 16]> = sts2_core::legal_actions(&combat);
        if legal.is_empty() {
            stuck = true;
            break;
        }
        let mut d = Decision {
            legal: &legal,
            combat: &combat,
            state: None,
        };
        let action = match cell.strategy.as_str() {
            "random" => random_agent(&d, &mut rng),
            "block_first" => block_first(&mut d)?,
            "attack_first" => attack_first(&mut d)?,
            "min_overblock" => min_overblock(&mut d)?,
            "mcts" => mcts_agent(&d, sims, depth)?,
            other => return Err(format!("KeyError: {other:?}")),
        };
        let ended_turn = matches!(action, Action::EndTurn);
        let _ = sts2_core::apply(&mut combat, action);
        actions += 1;
        if ended_turn {
            turns += 1;
        }
    }
    let final_hp = player_field(&wire_of(&combat)?, "hp");
    let (terminal, _) = sts2_core::terminal_value(&combat);
    let victory = terminal && final_hp > 0;
    Ok(json!({
        "encounter": cell.encounter,
        "ascension": cell.ascension,
        "deck": cell.deck_name,
        "strategy": cell.strategy,
        "seed": seed,
        "hp_start": hp_start,
        "hp_end": final_hp.max(0),
        "dmg_taken": hp_start - final_hp.max(0),
        "turns": turns,
        "actions": actions,
        "victory": victory,
        "timed_out": !terminal && !stuck,
        "stuck": stuck,
        "wall_ms": py_round(clock() - t0, 2),
    }))
}

/// `runner._error_row`: the uniform schema for a failed trial.
fn error_row(cell: &Cell, seed: u32, error: &str) -> Value {
    json!({
        "encounter": cell.encounter,
        "ascension": cell.ascension,
        "deck": cell.deck_name,
        "strategy": cell.strategy,
        "seed": seed,
        "hp_start": Value::Null,
        "hp_end": Value::Null,
        "dmg_taken": 0,
        "turns": 0,
        "actions": 0,
        "victory": false,
        "timed_out": false,
        "stuck": false,
        "wall_ms": Value::Null,
        "error": error,
    })
}

// ── telemetry (`telemetry.py`) ────────────────────────────────────────────

/// `telemetry.bucket_key`: `encounter/ascension/deck/strategy`.
#[must_use]
pub fn bucket_key(row: &Value) -> String {
    let s = |k: &str| match row.get(k) {
        Some(Value::String(s)) => s.clone(),
        Some(other) => other.to_string(),
        None => String::new(),
    };
    let deck = if row.get("deck").is_some() {
        s("deck")
    } else {
        "default".to_string()
    };
    format!(
        "{}/{}/{}/{}",
        s("encounter"),
        s("ascension"),
        deck,
        s("strategy")
    )
}

fn is_error(row: &Value) -> bool {
    row.get("error").is_some()
}

fn num(row: &Value, key: &str) -> f64 {
    row.get(key).and_then(Value::as_f64).unwrap_or(0.0)
}

/// `telemetry.throughput`: per-core rates over `rows` (error rows ignored).
#[must_use]
pub fn throughput(rows: &[&Value]) -> Value {
    let ok: Vec<&Value> = rows.iter().copied().filter(|r| !is_error(r)).collect();
    let combats = ok.len();
    let actions: f64 = ok.iter().map(|r| num(r, "actions")).sum();
    let turns: f64 = ok.iter().map(|r| num(r, "turns")).sum();
    let core_s: f64 = ok.iter().map(|r| num(r, "wall_ms")).sum::<f64>() / 1000.0;
    let rate = |n: f64| {
        if core_s > 0.0 {
            py_round(n / core_s, 2)
        } else {
            0.0
        }
    };
    json!({
        "combats": combats,
        "actions": actions,
        "turns": turns,
        "core_seconds": py_round(core_s, 3),
        "combats_per_core_s": rate(combats as f64),
        "actions_per_core_s": rate(actions),
        "turns_per_core_s": rate(turns),
    })
}

/// `telemetry.quantile` over a sorted list (nearest-rank, Python `round`).
fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let last = (sorted.len() - 1) as f64;
    let idx = (q * last).round_ties_even().clamp(0.0, last) as usize;
    sorted[idx]
}

fn fmean(vals: impl Iterator<Item = f64>) -> f64 {
    let (sum, n) = vals.fold((0.0, 0usize), |(s, n), v| (s + v, n + 1));
    if n == 0 {
        0.0
    } else {
        sum / n as f64
    }
}

fn empty_bucket(errors: usize) -> Value {
    json!({
        "n": 0,
        "mean_dmg": 0.0,
        "median_dmg": 0.0,
        "p95_dmg": 0.0,
        "max_dmg": 0.0,
        "win_rate": 0.0,
        "mean_turns": 0.0,
        "errors": errors,
        "mode_dmg": 0.0,
        "worst": 0.0,
        "worst_p": 0.0,
        "box": {"q1": 0.0, "median": 0.0, "q3": 0.0, "lo": 0.0, "hi": 0.0, "outliers": []},
        "throughput": throughput(&[]),
    })
}

/// `telemetry.summarize`: per-bucket stats plus the sweep-wide throughput.
#[must_use]
pub fn summarize(rows: &[Value]) -> Value {
    let mut buckets: BTreeMap<String, Vec<&Value>> = BTreeMap::new();
    let mut order: Vec<String> = Vec::new();
    for r in rows {
        let key = bucket_key(r);
        if !buckets.contains_key(&key) {
            order.push(key.clone());
        }
        buckets.entry(key).or_default().push(r);
    }
    let mut out = Map::new();
    for key in order {
        let rs = &buckets[&key];
        let ok: Vec<&Value> = rs.iter().copied().filter(|r| !is_error(r)).collect();
        let errors = rs.len() - ok.len();
        if ok.is_empty() {
            out.insert(key, empty_bucket(errors));
            continue;
        }
        let mut dmg: Vec<f64> = ok.iter().map(|r| num(r, "dmg_taken")).collect();
        dmg.sort_by(f64::total_cmp);
        let mut counts: BTreeMap<i64, usize> = BTreeMap::new();
        for v in &dmg {
            *counts.entry(*v as i64).or_insert(0) += 1;
        }
        // Most frequent value, ties broken by the smallest value.
        let mode_dmg = counts
            .iter()
            .max_by_key(|(v, c)| (**c, -**v))
            .map_or(0.0, |(v, _)| *v as f64);
        let worst = *dmg.last().expect("non-empty");
        let lo = quantile(&dmg, 0.01);
        let hi = quantile(&dmg, 0.99);
        let wins = ok
            .iter()
            .filter(|r| r.get("victory").and_then(Value::as_bool).unwrap_or(false))
            .count();
        let mut outliers: Vec<f64> = dmg.iter().copied().filter(|v| *v < lo || *v > hi).collect();
        outliers.dedup();
        let worst_count = counts.get(&(worst as i64)).copied().unwrap_or(0);
        out.insert(
            key,
            json!({
                "n": ok.len(),
                "mean_dmg": py_round(fmean(dmg.iter().copied()), 3),
                "median_dmg": quantile(&dmg, 0.5),
                "p95_dmg": quantile(&dmg, 0.95),
                "max_dmg": worst,
                "win_rate": py_round(wins as f64 / ok.len() as f64, 4),
                "mean_turns": py_round(fmean(ok.iter().map(|r| num(r, "turns"))), 2),
                "errors": errors,
                "mode_dmg": mode_dmg,
                "worst": worst,
                "worst_p": py_round(worst_count as f64 / ok.len() as f64, 6),
                "box": {
                    "q1": quantile(&dmg, 0.25),
                    "median": quantile(&dmg, 0.5),
                    "q3": quantile(&dmg, 0.75),
                    "lo": lo,
                    "hi": hi,
                    "outliers": outliers,
                },
                "throughput": throughput(rs),
            }),
        );
    }
    let all: Vec<&Value> = rows.iter().collect();
    json!({"buckets": out, "throughput": throughput(&all)})
}

/// `app._kde`: a Gaussian KDE at fixed bandwidth 0.3 (`scipy.stats.gaussian_kde
/// (arr, bw_method=0.3)` — unbiased variance × 0.3², evaluated on 200 points
/// from 0 to 1.1 × max), or a single spike when every value is the same.
#[must_use]
pub fn kde(values: &[f64]) -> Value {
    let mut distinct = values.to_vec();
    distinct.sort_by(f64::total_cmp);
    distinct.dedup();
    if distinct.len() < 2 {
        let v = values.first().copied().unwrap_or(0.0);
        return json!({"x": [v], "y": [1.0]});
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let cov = var * 0.3 * 0.3;
    let norm = (2.0 * std::f64::consts::PI * cov).sqrt();
    let max = values.iter().copied().fold(f64::MIN, f64::max);
    let stop = (max * 1.1).max(1.0);
    let step = stop / 199.0;
    let mut xs = Vec::with_capacity(200);
    let mut ys = Vec::with_capacity(200);
    for i in 0..200 {
        let x = if i == 199 { stop } else { f64::from(i) * step };
        let density: f64 = values
            .iter()
            .map(|d| (-0.5 * (x - d).powi(2) / cov).exp())
            .sum::<f64>()
            / n
            / norm;
        xs.push(py_round(x, 2));
        ys.push(py_round(density, 6));
    }
    json!({"x": xs, "y": ys})
}

/// `GET /api/sweeps/{id}/results`: the buckets with their raw values, the
/// KDE and (optionally) per-seed deltas against a baseline strategy or deck.
pub fn results(
    rows: &[Value],
    baseline: &str,
    baseline_deck: &str,
) -> Result<Value, (u16, String)> {
    if !baseline.is_empty() && !baseline_deck.is_empty() {
        return Err((
            422,
            "baseline and baseline_deck are mutually exclusive".to_string(),
        ));
    }
    let ok: Vec<Value> = rows.iter().filter(|r| !is_error(r)).cloned().collect();
    let total_errors = rows.len() - ok.len();
    let summary = summarize(&ok);
    let mut by_bucket: BTreeMap<String, Vec<&Value>> = BTreeMap::new();
    let mut order: Vec<String> = Vec::new();
    for r in &ok {
        let key = bucket_key(r);
        if !by_bucket.contains_key(&key) {
            order.push(key.clone());
        }
        by_bucket.entry(key).or_default().push(r);
    }
    // `key.rsplit("/", 3)` → [encounter, ascension, deck, strategy].
    let parts = |key: &str| -> Vec<String> {
        let mut p: Vec<&str> = key.rsplitn(4, '/').collect();
        p.reverse();
        p.into_iter().map(str::to_string).collect()
    };
    if !baseline_deck.is_empty() && !by_bucket.keys().any(|k| parts(k)[2] == baseline_deck) {
        return Err((
            422,
            format!("baseline_deck {baseline_deck:?} not found in results"),
        ));
    }
    if !baseline.is_empty() && !by_bucket.keys().any(|k| parts(k)[3] == baseline) {
        return Err((422, format!("baseline {baseline:?} not found in results")));
    }
    let mut out = Map::new();
    for key in order {
        let rs = &by_bucket[&key];
        let dmg: Vec<f64> = rs.iter().map(|r| num(r, "dmg_taken")).collect();
        let mut bucket = summary["buckets"][&key]
            .as_object()
            .cloned()
            .unwrap_or_default();
        bucket.insert(
            "dmg_values".to_string(),
            json!(dmg.iter().map(|v| *v as i64).collect::<Vec<_>>()),
        );
        bucket.insert("kde".to_string(), kde(&dmg));
        if !baseline.is_empty() || !baseline_deck.is_empty() {
            let p = parts(&key);
            let base_key = if baseline.is_empty() {
                format!("{}/{}/{}/{}", p[0], p[1], baseline_deck, p[3])
            } else {
                format!("{}/{}/{}/{}", p[0], p[1], p[2], baseline)
            };
            let base_rows: BTreeMap<i64, f64> = by_bucket
                .get(&base_key)
                .into_iter()
                .flatten()
                .map(|r| {
                    (
                        r.get("seed").and_then(Value::as_i64).unwrap_or(0),
                        num(r, "dmg_taken"),
                    )
                })
                .collect();
            let deltas: Vec<f64> = rs
                .iter()
                .filter_map(|r| {
                    let seed = r.get("seed").and_then(Value::as_i64).unwrap_or(0);
                    base_rows.get(&seed).map(|b| num(r, "dmg_taken") - b)
                })
                .collect();
            if !deltas.is_empty() {
                bucket.insert(
                    "delta_vs_baseline".to_string(),
                    json!({
                        "mean": py_round(deltas.iter().sum::<f64>() / deltas.len() as f64, 3),
                        "values": deltas.iter().map(|v| *v as i64).collect::<Vec<_>>(),
                    }),
                );
            }
        }
        out.insert(key, Value::Object(bucket));
    }
    Ok(json!({"buckets": out, "errors_total": total_errors}))
}

// ── the job (`web/jobs.py`) ───────────────────────────────────────────────

/// The lifecycle of a sweep, as `progress.json`'s `status` reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SweepStatus {
    Running,
    Done,
    Cancelled,
    Error,
}

impl SweepStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Done => "done",
            Self::Cancelled => "cancelled",
            Self::Error => "error",
        }
    }
}

/// One submitted sweep: its spec, the trials still to play, the rows so far.
#[derive(Debug)]
pub struct SweepJob {
    pub id: String,
    pub spec: SweepSpec,
    cells: Vec<Cell>,
    /// `(cell index, seed)` in the runner's dispatch order.
    queue: Vec<(usize, u32)>,
    next: usize,
    rows: Vec<Value>,
    total: usize,
    status: SweepStatus,
    error: Option<String>,
    stopped_reason: Option<&'static str>,
    cancel_requested: bool,
    /// Epoch seconds of submission (`spec.json`'s mtime: the listing's sort key).
    pub mtime: f64,
    /// The clock at the first trial (`started` in `jobs.py`: set just before
    /// the run begins so the rate is not skewed by the wait in the queue).
    started_ms: Option<f64>,
    elapsed_s: f64,
    /// The wall-clock cap (`max_runtime_s`); `<= 0` = uncapped.
    max_runtime_s: f64,
}

impl SweepJob {
    /// `JobManager.submit` minus the id: the queue in the runner's order
    /// (cell by cell, seeds ascending — one worker, so chunking is moot).
    pub fn new(id: String, spec: SweepSpec, cells: Vec<Cell>, now_ms: f64) -> Self {
        let mut queue = Vec::new();
        for (ci, _) in cells.iter().enumerate() {
            for seed in 0..spec.n_seeds {
                queue.push((ci, seed));
            }
        }
        let total = queue.len();
        Self {
            id,
            spec,
            cells,
            queue,
            next: 0,
            rows: Vec::new(),
            total,
            status: SweepStatus::Running,
            error: None,
            stopped_reason: None,
            cancel_requested: false,
            mtime: now_ms / 1000.0,
            started_ms: None,
            elapsed_s: 0.0,
            max_runtime_s: DEFAULT_MAX_RUNTIME_S,
        }
    }

    #[must_use]
    pub fn status(&self) -> SweepStatus {
        self.status
    }

    #[must_use]
    pub fn is_running(&self) -> bool {
        self.status == SweepStatus::Running
    }

    #[must_use]
    pub fn rows(&self) -> &[Value] {
        &self.rows
    }

    /// `JobManager.stop`: request cancellation; `false` when not running.
    pub fn stop(&mut self) -> bool {
        if !self.is_running() {
            return false;
        }
        self.cancel_requested = true;
        true
    }

    /// Play up to `max_trials` more trials (fewer if the job ends). The
    /// cancel flag and the runtime cap are checked between trials, so a
    /// stop lands within one trial.
    pub fn run_some(&mut self, max_trials: usize, clock: Clock<'_>) {
        let mut played = 0;
        let started = *self.started_ms.get_or_insert_with(clock);
        while self.is_running() && played < max_trials {
            self.elapsed_s = (clock() - started) / 1000.0;
            if self.cancel_requested {
                self.finish(SweepStatus::Cancelled, Some("user"));
                return;
            }
            if self.max_runtime_s > 0.0 && self.elapsed_s >= self.max_runtime_s {
                self.finish(SweepStatus::Cancelled, Some("max_runtime"));
                return;
            }
            let Some(&(ci, seed)) = self.queue.get(self.next) else {
                self.finish(SweepStatus::Done, None);
                return;
            };
            let cell = &self.cells[ci];
            let row =
                match play_combat(cell, seed, self.spec.mcts_sims, self.spec.mcts_depth, clock) {
                    Ok(row) => row,
                    Err(e) => error_row(cell, seed, &e),
                };
            self.rows.push(row);
            self.next += 1;
            played += 1;
        }
        self.elapsed_s = (clock() - started) / 1000.0;
        if self.is_running() && self.next >= self.queue.len() {
            self.finish(SweepStatus::Done, None);
        }
    }

    fn finish(&mut self, status: SweepStatus, reason: Option<&'static str>) {
        self.status = status;
        self.stopped_reason = reason;
    }

    /// `progress.json` (`GET /api/sweeps/{id}`).
    #[must_use]
    pub fn progress(&self) -> Value {
        if self.status == SweepStatus::Error {
            return json!({"status": "error", "error": self.error.clone().unwrap_or_default()});
        }
        let done = self.next.min(self.total);
        let mut p = json!({
            "done": done,
            "total": self.total,
            "status": self.status.as_str(),
            "jobs": 1,
            "elapsed_s": py_round(self.elapsed_s, 2),
            "combats_per_s": if self.elapsed_s > 0.0 { py_round(done as f64 / self.elapsed_s, 2) } else { 0.0 },
        });
        if let Some(reason) = self.stopped_reason {
            p["stopped_reason"] = json!(reason);
        }
        p
    }

    /// The listing row (`JobManager.list_sweeps`).
    #[must_use]
    pub fn list_entry(&self) -> Value {
        let mut entry = json!({
            "sweep_id": self.id,
            "spec": self.spec.to_json(),
            "mtime": self.mtime,
        });
        if let (Some(e), Some(p)) = (entry.as_object_mut(), self.progress().as_object()) {
            for (k, v) in p {
                e.insert(k.clone(), v.clone());
            }
        }
        entry
    }

    /// Everything the host stores so the sweep outlives a reload
    /// (`spec.json` + `rows.jsonl` + `progress.json` in one document).
    #[must_use]
    pub fn export(&self) -> Value {
        json!({
            "sweep_id": self.id,
            "spec": self.spec.to_json(),
            "mtime": self.mtime,
            "rows": self.rows,
            "progress": self.progress(),
        })
    }

    /// The inverse of [`Self::export`]. A sweep stored while still running
    /// was interrupted by the reload: it comes back `cancelled` with what it
    /// had, the honest equivalent of the desktop's orphaned `progress.json`.
    pub fn import(doc: &Value) -> Result<Self, String> {
        let id = doc
            .get("sweep_id")
            .and_then(Value::as_str)
            .ok_or("sweep_id must be a string")?
            .to_string();
        let spec = SweepSpec::from_json(doc.get("spec").unwrap_or(&Value::Null))?;
        let rows: Vec<Value> = doc
            .get("rows")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let progress = doc.get("progress").cloned().unwrap_or_else(|| json!({}));
        let status = match progress.get("status").and_then(Value::as_str) {
            Some("done") => SweepStatus::Done,
            Some("error") => SweepStatus::Error,
            _ => SweepStatus::Cancelled,
        };
        let total = progress
            .get("total")
            .and_then(Value::as_u64)
            .map_or(rows.len(), |t| t as usize);
        let stopped_reason = match progress.get("stopped_reason").and_then(Value::as_str) {
            Some("max_runtime") => Some("max_runtime"),
            Some("user") => Some("user"),
            _ if status == SweepStatus::Cancelled => Some("reload"),
            _ => None,
        };
        let mtime = doc.get("mtime").and_then(Value::as_f64).unwrap_or(0.0);
        Ok(Self {
            id,
            spec,
            cells: Vec::new(),
            queue: Vec::new(),
            next: rows.len(),
            rows,
            total,
            status,
            error: progress
                .get("error")
                .and_then(Value::as_str)
                .map(str::to_string),
            stopped_reason,
            cancel_requested: false,
            mtime,
            started_ms: None,
            elapsed_s: progress
                .get("elapsed_s")
                .and_then(Value::as_f64)
                .unwrap_or(0.0),
            max_runtime_s: DEFAULT_MAX_RUNTIME_S,
        })
    }
}

/// `time.strftime('%Y%m%d-%H%M%S')` of an epoch-millisecond clock, in UTC.
#[must_use]
pub fn stamp(now_ms: f64) -> String {
    let secs = (now_ms / 1000.0).floor().max(0.0) as i64;
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let (hour, minute, second) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    // Howard Hinnant's civil-from-days.
    let shifted = days + 719_468;
    let era = shifted.div_euclid(146_097);
    let doe = shifted.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { year + 1 } else { year };
    format!("{year:04}{month:02}{day:02}-{hour:02}{minute:02}{second:02}")
}

/// The sweeps this workbench knows, by id (`JobManager` + `results_root`).
#[derive(Debug, Default)]
pub struct SweepRegistry {
    jobs: BTreeMap<String, SweepJob>,
    /// Submission order, so the running job the host steps is the oldest.
    order: Vec<String>,
}

impl SweepRegistry {
    /// `JobManager.submit`: `{stamp}-{name}` with a `-N` suffix on a
    /// same-second collision.
    pub fn submit(&mut self, spec: SweepSpec, cells: Vec<Cell>, now_ms: f64) -> String {
        let base = format!("{}-{}", stamp(now_ms), spec.name);
        let mut id = base.clone();
        let mut counter = 0;
        while self.jobs.contains_key(&id) {
            counter += 1;
            id = format!("{base}-{counter}");
        }
        self.jobs
            .insert(id.clone(), SweepJob::new(id.clone(), spec, cells, now_ms));
        self.order.push(id.clone());
        id
    }

    #[must_use]
    pub fn get(&self, id: &str) -> Option<&SweepJob> {
        self.jobs.get(id)
    }

    pub fn get_mut(&mut self, id: &str) -> Option<&mut SweepJob> {
        self.jobs.get_mut(id)
    }

    /// Newest first by `mtime`, then id — the listing order.
    #[must_use]
    pub fn list(&self) -> Value {
        let mut jobs: Vec<&SweepJob> = self.jobs.values().collect();
        jobs.sort_by(|a, b| {
            b.mtime
                .partial_cmp(&a.mtime)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.id.cmp(&a.id))
        });
        json!({"sweeps": jobs.iter().map(|j| j.list_entry()).collect::<Vec<_>>()})
    }

    /// Whether any sweep still has trials to play.
    #[must_use]
    pub fn has_running(&self) -> bool {
        self.jobs.values().any(SweepJob::is_running)
    }

    /// Step the oldest running sweep by up to `max_trials`; the ids of the
    /// sweeps that ended during this call (for the host to persist).
    pub fn tick(&mut self, max_trials: usize, clock: Clock<'_>) -> Vec<String> {
        let Some(id) = self
            .order
            .iter()
            .find(|id| self.jobs.get(*id).is_some_and(SweepJob::is_running))
            .cloned()
        else {
            return Vec::new();
        };
        let job = self.jobs.get_mut(&id).expect("listed");
        job.run_some(max_trials, clock);
        if job.is_running() {
            Vec::new()
        } else {
            vec![id]
        }
    }

    /// Add a stored sweep back (a reload). An id already present is left
    /// alone — the live one is fresher than the stored copy.
    pub fn import(&mut self, doc: &Value) -> Result<(), String> {
        let job = SweepJob::import(doc)?;
        if self.jobs.contains_key(&job.id) {
            return Ok(());
        }
        self.order.push(job.id.clone());
        self.jobs.insert(job.id.clone(), job);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn native_clock() -> f64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0.0, |d| d.as_secs_f64() * 1000.0)
    }

    fn spec(strategies: &[&str], n_seeds: u32) -> SweepSpec {
        SweepSpec::from_json(&json!({
            "name": "t",
            "encounters": ["FUZZY_WURM_CRAWLER_WEAK"],
            "ascensions": [0],
            "strategies": strategies,
            "n_seeds": n_seeds,
            "mcts_sims": 8,
            "mcts_depth": 20,
        }))
        .expect("spec")
    }

    #[test]
    fn stamp_is_utc_strftime() {
        // 2026-09-08T12:34:56Z
        assert_eq!(stamp(1_788_870_896_000.0), "20260908-123456");
        assert_eq!(stamp(0.0), "19700101-000000");
    }

    #[test]
    fn spec_rejects_bad_shapes_like_the_dataclass() {
        let err = |v: Value| SweepSpec::from_json(&v).unwrap_err();
        assert!(
            err(json!({"encounters": [], "ascensions": [], "strategies": [], "n_seeds": 1}))
                .contains("name")
        );
        assert!(err(
            json!({"name": "x", "encounters": [], "ascensions": [], "strategies": [],
            "n_seeds": 1, "decks": [["a/b", ["BASH"]]]})
        )
        .contains("invalid deck name"));
        assert!(err(
            json!({"name": "x", "encounters": [], "ascensions": [], "strategies": [],
            "n_seeds": 1, "decks": [["a", ["BASH"]], ["a", ["BASH"]]]})
        )
        .contains("duplicate deck names"));
        assert!(err(
            json!({"name": "x", "encounters": [], "ascensions": [], "strategies": [],
            "n_seeds": 1, "bogus": 1})
        )
        .contains("unexpected field"));
        // Legacy `deck_preset` maps to one named deck.
        let s = SweepSpec::from_json(&json!({"name": "x", "encounters": [], "ascensions": [],
            "strategies": [], "n_seeds": 1, "deck_preset": "turtle"}))
        .unwrap();
        assert_eq!(s.decks.len(), 1);
        assert_eq!(s.decks[0].0, "turtle");
    }

    #[test]
    fn validate_catches_unknown_cards_and_monsters() {
        assert!(validate_deck("d", &["NOT_A_CARD".to_string()])
            .unwrap_err()
            .contains("unknown card id"));
        assert!(validate_deck("d", &["STRIKE_IRONCLAD+1".to_string()]).is_ok());
        assert!(validate_encounter("monsters:").is_err());
        assert!(validate_encounter("monsters:NotAMonster").is_err());
        assert!(validate_encounter("MAWLER_NORMAL").is_ok());
        let mut s = spec(&["nope"], 1);
        assert!(s.validate().unwrap_err().contains("unknown strategy"));
        s.strategies = vec!["random".to_string()];
        assert_eq!(s.validate().unwrap().len(), 1);
    }

    #[test]
    fn cells_follow_the_python_product_order() {
        let s = SweepSpec::from_json(&json!({
            "name": "t", "encounters": ["A", "B"], "ascensions": [0, 1],
            "strategies": ["random", "mcts"], "n_seeds": 3,
            "decks": [["d1", ["BASH"]], ["d2", ["BASH"]]],
        }))
        .unwrap();
        let cells = s.cells().unwrap();
        assert_eq!(cells.len(), 16);
        let keys: Vec<String> = cells
            .iter()
            .take(5)
            .map(|c| {
                format!(
                    "{}/{}/{}/{}",
                    c.encounter, c.ascension, c.deck_name, c.strategy
                )
            })
            .collect();
        assert_eq!(
            keys,
            [
                "A/0/d1/random",
                "A/0/d1/mcts",
                "A/0/d2/random",
                "A/0/d2/mcts",
                "A/1/d1/random"
            ]
        );
    }

    #[test]
    fn combinations_are_lexicographic() {
        let mut seen = Vec::new();
        for_each_combination(4, 2, |c| seen.push(c.to_vec()));
        assert_eq!(seen, vec![[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]);
    }

    #[test]
    fn pick_block_subset_matches_the_video_examples() {
        // 5-block Defends vs an 11 hit: block 10, take the 1.
        let opts = [(0, 5, 1), (1, 5, 1), (2, 5, 1)];
        assert_eq!(pick_block_subset(&opts, 11, 3), vec![0, 1]);
        // Nothing fits under a 7 hit; an 8-block Shrug is played (waste 1 < leak 7).
        assert_eq!(pick_block_subset(&[(0, 8, 1)], 7, 3), vec![0]);
        // A lone 5-block Defend vs a 1 hit is not (waste 4 > leak 1).
        assert!(pick_block_subset(&[(0, 5, 1)], 1, 3).is_empty());
        // Energy-bound.
        assert_eq!(pick_block_subset(&opts, 15, 1), vec![0]);
    }

    #[test]
    fn quantile_rounds_ties_to_even_like_python() {
        // round(0.5 * 1) = 0 in Python (ties to even), not 1.
        assert_eq!(quantile(&[1.0, 2.0], 0.5).to_bits(), 1.0_f64.to_bits());
        assert_eq!(quantile(&[1.0, 2.0, 3.0], 0.5).to_bits(), 2.0_f64.to_bits());
        assert_eq!(quantile(&[], 0.5).to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn every_strategy_plays_a_fight_to_the_end() {
        for strategy in STRATEGIES {
            let s = spec(&[strategy], 1);
            let cells = s.validate().unwrap();
            let row = play_combat(&cells[0], 0, s.mcts_sims, s.mcts_depth, &native_clock)
                .unwrap_or_else(|e| panic!("{strategy}: {e}"));
            assert!(row.get("error").is_none(), "{strategy}: {row}");
            assert!(row["actions"].as_u64().unwrap() > 0, "{strategy}: {row}");
            assert!(!row["stuck"].as_bool().unwrap(), "{strategy}: {row}");
            assert_eq!(row["encounter"], "FUZZY_WURM_CRAWLER_WEAK");
        }
    }

    #[test]
    fn seed_pairing_is_deterministic() {
        let s = spec(&["block_first"], 1);
        let cells = s.validate().unwrap();
        let a = play_combat(&cells[0], 3, 8, 20, &|| 0.0).unwrap();
        let b = play_combat(&cells[0], 3, 8, 20, &|| 0.0).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn a_job_runs_to_done_in_steps_and_stops_on_request() {
        let s = spec(&["random", "attack_first"], 3);
        let cells = s.validate().unwrap();
        let mut reg = SweepRegistry::default();
        let id = reg.submit(s.clone(), cells.clone(), 1_788_870_896_000.0);
        assert_eq!(id, "20260908-123456-t");
        assert_eq!(
            reg.submit(s, cells, 1_788_870_896_000.0),
            "20260908-123456-t-1"
        );
        assert!(reg.has_running());
        let mut finished = Vec::new();
        for _ in 0..100 {
            finished.extend(reg.tick(2, &native_clock));
            if finished.len() == 2 {
                break;
            }
        }
        assert_eq!(finished, vec!["20260908-123456-t", "20260908-123456-t-1"]);
        assert!(!reg.has_running());
        let job = reg.get(&id).unwrap();
        assert_eq!(job.progress()["status"], "done");
        assert_eq!(job.progress()["done"], 6);
        assert_eq!(job.rows().len(), 6);
        let res = results(job.rows(), "random", "").unwrap();
        let key = "FUZZY_WURM_CRAWLER_WEAK/0/ironclad_starter/attack_first";
        assert_eq!(res["buckets"][key]["n"], 3);
        assert_eq!(
            res["buckets"][key]["delta_vs_baseline"]["values"]
                .as_array()
                .unwrap()
                .len(),
            3
        );
        assert!(results(job.rows(), "nope", "").is_err());
        assert!(results(job.rows(), "a", "b").is_err());

        // Stop lands within one trial and leaves partial rows.
        let s = spec(&["random"], 50);
        let cells = s.validate().unwrap();
        let id = reg.submit(s, cells, 0.0);
        reg.tick(2, &native_clock);
        assert!(reg.get_mut(&id).unwrap().stop());
        assert_eq!(reg.tick(10, &native_clock), vec![id.clone()]);
        let job = reg.get(&id).unwrap();
        assert_eq!(job.progress()["status"], "cancelled");
        assert_eq!(job.progress()["stopped_reason"], "user");
        assert_eq!(job.rows().len(), 2);
        let doc = job.export();
        assert!(!reg.get_mut(&id).unwrap().stop());

        // Export / import round-trips a finished sweep; a running one comes
        // back cancelled.
        let mut fresh = SweepRegistry::default();
        fresh.import(&doc).unwrap();
        assert_eq!(fresh.get(&id).unwrap().rows().len(), 2);
        assert_eq!(fresh.list()["sweeps"].as_array().unwrap().len(), 1);
        let mut running = doc.clone();
        running["progress"]["status"] = json!("running");
        running["progress"]
            .as_object_mut()
            .unwrap()
            .remove("stopped_reason");
        running["sweep_id"] = json!("other");
        fresh.import(&running).unwrap();
        assert_eq!(
            fresh.get("other").unwrap().progress()["status"],
            "cancelled"
        );
        assert_eq!(
            fresh.get("other").unwrap().progress()["stopped_reason"],
            "reload"
        );
    }

    #[test]
    fn summarize_and_kde_shapes() {
        let rows: Vec<Value> = (0..5)
            .map(|i| {
                json!({"encounter": "E", "ascension": 0, "deck": "d", "strategy": "s",
                    "seed": i, "dmg_taken": i * 2, "turns": 3, "actions": 5, "victory": i < 4,
                    "wall_ms": 1.5})
            })
            .collect();
        let s = summarize(&rows);
        let b = &s["buckets"]["E/0/d/s"];
        assert_eq!(b["n"], 5);
        assert_eq!(b["mean_dmg"], 4.0);
        assert_eq!(b["median_dmg"], 4.0);
        assert_eq!(b["win_rate"], 0.8);
        assert_eq!(b["mode_dmg"], 0.0);
        assert_eq!(b["worst"], 8.0);
        assert_eq!(b["worst_p"], 0.2);
        assert_eq!(b["box"]["q1"], 2.0);
        assert_eq!(b["throughput"]["combats"], 5);
        let k = kde(&[0.0, 2.0, 4.0, 6.0, 8.0]);
        assert_eq!(k["x"].as_array().unwrap().len(), 200);
        assert_eq!(k["x"][199], 8.8);
        assert_eq!(kde(&[3.0, 3.0]), json!({"x": [3.0], "y": [1.0]}));
        let empty = summarize(&[json!({"encounter": "E", "ascension": 0, "deck": "d",
            "strategy": "s", "seed": 0, "error": "boom"})]);
        assert_eq!(empty["buckets"]["E/0/d/s"]["errors"], 1);
        assert_eq!(empty["buckets"]["E/0/d/s"]["n"], 0);
    }

    #[test]
    fn a_start_state_sweep_reseeds_the_board() {
        let s = spec(&["random"], 1);
        let cells = s.validate().unwrap();
        let cs = cells[0].build_combat(1).unwrap();
        let wire = wire_of(&cs).unwrap();
        let ss = SweepSpec::from_json(&json!({
            "name": "pos", "encounters": [], "ascensions": [], "strategies": ["random", "block_first"],
            "n_seeds": 2, "start_states": [["as is", wire.to_string()]],
        }))
        .unwrap();
        let cells = ss.validate().unwrap();
        assert_eq!(cells.len(), 2);
        assert_eq!(cells[0].encounter, IMPORTED_ENCOUNTER);
        assert_eq!(cells[0].deck_name, "as is");
        let a = cells[0].build_combat(0).unwrap();
        let b = cells[0].build_combat(1).unwrap();
        // Same visible board, different future dice.
        assert_eq!(a.player.current_hp, b.player.current_hp);
        assert_ne!(sts2_core::state_hash(&a), sts2_core::state_hash(&b));
        let row = play_combat(&cells[1], 0, 8, 20, &|| 0.0).unwrap();
        assert!(row.get("error").is_none(), "{row}");
    }
}
