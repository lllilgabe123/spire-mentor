//! The advisors (M2): ports of `workbench/reasoning.py`,
//! `choice_ranker.py`, `regret_review.py`, `card_advisor.py` and
//! `decision_advisor.py`, over the live [`Session`].
//!
//! Two families:
//!
//! * **Search-backed** — the combat hint, the search-tree inspector, the
//!   in-combat choice ranker, "play best" and the regret coach / review.
//!   All are one `mcts_recommend` / `mcts_search_tree` over a
//!   [`CombatState`], through the emitters in `sts2_wire::emit::mcts` that
//!   the `PyO3` module also uses. Deterministic for a config, so the hint /
//!   tree / ranker rows are golden-parity tested against the Python
//!   workbench. The regret routes score the parent node's **native**
//!   snapshot ([`Node::combat`]) — the Python session no longer holds that
//!   state and re-scores a lossy wire dict — so those are checked
//!   structurally, not by hash.
//! * **Rollout-backed** — the card advisor and the decision advisor: fork the
//!   run per option, take that option, play the rest of the act out
//!   `rollouts` times (a cheap MCTS in combat, the tier-1 [`HeuristicPolicy`]
//!   off combat), paired across options by seed. They report progress
//!   through a [`Progress`] sink — one `start` frame, one `tick` per playout —
//!   which the streaming routes forward as NDJSON. Python drives its rollouts
//!   with `random.Random` and its own `run_policy` heuristic, so these are
//!   structural twins, not bit-for-bit.
//!
//! Everything is single-threaded: `threads` in a config is honoured as 1
//! (wasm has no thread pool; the `PyO3` path keeps its rayon option).

use rand::{Rng, SeedableRng};
use serde_json::{json, Map, Value};
use sts2_core::full_run::RunPolicy;
use sts2_core::{Action, CombatState, RunAction, RunManager, RunRngSet};
use sts2_mcts::{Mcts, MctsConfig};
use sts2_runai::policy::HeuristicPolicy;
use sts2_wire::emit::combat_action_to_value;
use sts2_wire::emit::mcts::{
    recommend_value, score_action_forced, search_tree_value, RecommendParams,
};
use sts2_wire::emit::run_action::combat_action_to_run_action_value;

use crate::combat_view::build_combat_view;
use crate::display_maps::loc_id;
use crate::localization::{resolve_state_names, Loc};
use crate::options::normalize_options;
use crate::rmx;
use crate::session::{Node, Session, SessionError};

/// Sentinel `stable_idx` values (`sts2_hint.recommender`): end turn, and
/// potion slot `s` → `-100 - s` (range `(-200, -100]`).
pub const END_TURN: i64 = -1;
pub const POTION_BASE: i64 = -100;
pub const POTION_MIN: i64 = -200;

/// Default blunder threshold (percentage points of win-probability lost
/// versus the MCTS line) — above MCTS sampling noise at the usual budget.
pub const DEFAULT_BLUNDER_DROP_PCT: f64 = 8.0;

/// Severity (win-prob points, or expected HP) below which a non-best move is
/// a "tie" — within search noise. Mirrors `regret_review.TIE_PTS`.
pub const TIE_PTS: f64 = 3.0;
/// Minimum resolved playouts (`hp_n`) on BOTH rows before the HP gap is used.
pub const HP_MIN_SAMPLES: i64 = 10;
/// Standard errors (of the difference of the two rows' mean final HP) an HP
/// gap must clear to count; under it the gap reads as 0 (GH #888). Mirrors
/// `regret_review.HP_GAP_SE`.
pub const HP_GAP_SE: f64 = 2.0;
/// Minimum MCTS visits on the played move before it can be graded at all.
pub const PLAYED_MIN_VISITS: i64 = 50;
/// Per-action severity (win-chance points or HP) below which an action does
/// not count toward its turn's grade (sub-point jitter over five plays would
/// otherwise add up to a slip). Mirrors `regret_review.TURN_ACTION_FLOOR`.
pub const TURN_ACTION_FLOOR: f64 = 1.0;

/// A progress sink: receives each wire frame (`{"type": "start", …}`,
/// `{"type": "tick", …}`) as the rollout advisors go.
pub type Progress<'a> = &'a mut dyn FnMut(&Value);

fn config_obj(config: &Value) -> Value {
    // `dict(config or {})`: the echoed config is always an object.
    match config {
        Value::Object(_) => config.clone(),
        _ => json!({}),
    }
}

/// `RecommendConfig.from_dict` + the browser's single-thread clamp.
pub(crate) fn params_for(config: &Value, fn_name: &str) -> Result<RecommendParams, String> {
    let mut p = RecommendParams::from_config(config, fn_name)?;
    p.threads = p.threads.min(1);
    Ok(p)
}

// ── reasoning.py ────────────────────────────────────────────────────────

fn potion_name(state: &Value, slot: usize) -> Option<String> {
    let potions = state
        .get("player")
        .and_then(|p| p.get("potions"))
        .and_then(Value::as_array)
        .or_else(|| state.get("potions").and_then(Value::as_array))?;
    let p = potions.get(slot)?;
    let name = if p.is_object() {
        p.get("name")
    } else {
        Some(p)
    };
    match name {
        Some(Value::Object(m)) => Some(
            m.get("en")
                .or_else(|| m.get("zh"))
                .and_then(Value::as_str)
                .unwrap_or("?")
                .to_string(),
        ),
        Some(Value::String(s)) if !s.is_empty() => Some(s.clone()),
        _ => Some(format!("slot {slot}")),
    }
}

/// `_action_label`: the human label for a stable action index.
#[must_use]
pub fn action_label(stable_idx: i64, state: &Value) -> String {
    if stable_idx == END_TURN {
        return "end turn".to_string();
    }
    if POTION_MIN < stable_idx && stable_idx <= POTION_BASE {
        #[allow(clippy::cast_sign_loss)]
        let slot = (POTION_BASE - stable_idx) as usize;
        return match potion_name(state, slot) {
            Some(name) => format!("drink {name}"),
            None => format!("drink potion slot {slot}"),
        };
    }
    if let Some(hand) = state.get("hand").and_then(Value::as_array) {
        for card in hand {
            if card.get("index").and_then(Value::as_i64) == Some(stable_idx) {
                return card
                    .get("name")
                    .and_then(Value::as_str)
                    .map_or_else(|| format!("card {stable_idx}"), str::to_string);
            }
        }
    }
    format!("action {stable_idx}")
}

/// `action_stable_key`: a native action dict → `(stable_idx, target_index)`.
/// A `play` card's `card_index` is its stable index, `end_turn` is `-1`, a
/// potion slot `s` is `-100 - s`. `(None, target)` for anything else.
#[must_use]
pub fn action_stable_key(args: &Value) -> (Option<i64>, Option<i64>) {
    let Some(a) = args.as_object() else {
        return (None, None);
    };
    if a.is_empty() {
        return (None, None);
    }
    let target = a.get("target_index").and_then(Value::as_i64);
    match a.get("action").and_then(Value::as_str) {
        Some("play") => (a.get("card_index").and_then(Value::as_i64), target),
        Some("end_turn") => (Some(END_TURN), None),
        Some("potion" | "drink_potion") => (
            a.get("slot")
                .and_then(Value::as_i64)
                .map(|s| POTION_BASE - s),
            target,
        ),
        _ => (None, target),
    }
}

fn f64_of(v: &Value) -> f64 {
    v.as_f64().unwrap_or(0.0)
}

/// Python `int(row.get(k, 0) or 0)`: a missing / null count is 0 and a
/// float truncates toward zero.
#[allow(clippy::cast_possible_truncation)]
fn py_int(v: &Value) -> i64 {
    v.as_i64()
        .or_else(|| v.as_f64().map(|f| f as i64))
        .unwrap_or(0)
}

fn win_prob(row: &Value) -> f64 {
    f64_of(&row["win_prob"]).clamp(0.0, 1.0)
}

/// The stable index a search action surfaces under in [`rows_from_mcts`]:
/// a card play's hand position, `end_turn` → [`END_TURN`], potion slot `s` →
/// `POTION_BASE - s`. `None` for channel / evoke / choose actions.
fn row_stable_idx(a: &Value) -> Option<i64> {
    match a["action"].as_str() {
        Some("play") => Some(a["card_index"].as_i64().unwrap_or(0)),
        Some("end_turn") => Some(END_TURN),
        Some("potion" | "drink_potion") => Some(POTION_BASE - a["slot"].as_i64().unwrap_or(0)),
        _ => None,
    }
}

/// `rows_from_mcts`: score rows from a raw `mcts_recommend` result, labelled
/// against `state` (the position the search ROOTED at, names resolved).
/// Channel / evoke / choose actions are not surfaced at this boundary.
#[must_use]
pub fn rows_from_mcts(result: &Value, state: &Value) -> Vec<Value> {
    let mut rows: Vec<Value> = Vec::new();
    for entry in result["actions"].as_array().into_iter().flatten() {
        let a = &entry["action"];
        let Some(stable_idx) = row_stable_idx(a) else {
            continue;
        };
        rows.push(json!({
            "stable_idx": stable_idx,
            "action": action_label(stable_idx, state),
            "win_prob": f64_of(&entry["p_win"]),
            "q": entry.get("q").or_else(|| entry.get("value")).map_or(0.0, f64_of),
            "visits": entry["visits"].as_i64().unwrap_or(0),
            "is_best": entry["is_best"].as_bool().unwrap_or(false),
            "q_saturated": entry["q_saturated"].as_bool().unwrap_or(false),
            "low_confidence": entry["low_confidence"].as_bool().unwrap_or(false),
            "target_index": a.get("target_index").and_then(Value::as_i64),
            "wins": entry["wins"].as_i64().unwrap_or(0),
            "losses": entry["losses"].as_i64().unwrap_or(0),
            "ties": entry["ties"].as_i64().unwrap_or(0),
            "mean_final_hp": entry["mean_final_hp"].as_f64(),
            "hp_sd": entry["hp_sd"].as_f64(),
            "hp_n": entry["hp_n"].as_i64().unwrap_or(0),
        }));
    }
    rows.sort_by(|a, b| {
        let ka = (
            !a["is_best"].as_bool().unwrap_or(false),
            -f64_of(&a["win_prob"]),
        );
        let kb = (
            !b["is_best"].as_bool().unwrap_or(false),
            -f64_of(&b["win_prob"]),
        );
        ka.0.cmp(&kb.0)
            .then(ka.1.partial_cmp(&kb.1).unwrap_or(std::cmp::Ordering::Equal))
    });
    rows
}

/// `score_native`: per-action MCTS scores for a native combat state —
/// `{scores, pv: null, config}`, or `{error, scores: [], pv: null, config}`.
/// `pv` stays null like Python's (the SPA reads it as `string[] | null`);
/// the coach takes the search's PV from [`score_with_pv`].
#[must_use]
pub fn score(cs: &CombatState, named_state: &Value, config: &Value) -> Value {
    score_with_pv(cs, named_state, config).0
}

/// [`score`] plus the search's principal variation as native action dicts
/// (empty under `fair` or on error) — for the coach's AI line only.
fn score_with_pv(cs: &CombatState, named_state: &Value, config: &Value) -> (Value, Vec<Value>) {
    let echo = config_obj(config);
    match params_for(config, "mcts_recommend") {
        Ok(p) => {
            let result = recommend_value(cs, &p, true);
            let pv = result["pv"].as_array().cloned().unwrap_or_default();
            let out = json!({
                "scores": rows_from_mcts(&result, named_state),
                "pv": Value::Null,
                "config": echo,
            });
            (out, pv)
        }
        Err(e) => (
            json!({"error": e, "scores": [], "pv": Value::Null, "config": echo}),
            Vec::new(),
        ),
    }
}

/// `_choice_key`: `choice:skip` / `choice:{opt}` for a `choose` action dict.
fn choice_key(action: &Value) -> String {
    match action.get("selection").and_then(Value::as_array) {
        Some(sel) if sel.is_empty() => "choice:skip".to_string(),
        Some(sel) => format!("choice:{}", sel[0].as_i64().unwrap_or(0)),
        None => format!("choice:{}", action["index"].as_i64().unwrap_or(0)),
    }
}

/// `rank_choice_native`: `{by_key: {choice_key: {win_prob, q, visits,
/// is_best}}, config}` from one search over an open pending choice.
#[must_use]
pub fn rank_choice_native(cs: &CombatState, config: &Value) -> Value {
    let echo = config_obj(config);
    let p = match params_for(config, "mcts_recommend") {
        Ok(p) => p,
        Err(e) => return json!({"error": e, "by_key": {}, "config": echo}),
    };
    let result = recommend_value(cs, &p, true);
    let mut by_key = Map::new();
    for entry in result["actions"].as_array().into_iter().flatten() {
        let a = &entry["action"];
        if a["action"].as_str() != Some("choose") {
            continue;
        }
        by_key.insert(
            choice_key(a),
            json!({
                "win_prob": f64_of(&entry["p_win"]),
                "q": entry.get("q").or_else(|| entry.get("value")).map_or(0.0, f64_of),
                "visits": entry["visits"].as_i64().unwrap_or(0),
                "is_best": entry["is_best"].as_bool().unwrap_or(false),
            }),
        );
    }
    json!({"by_key": by_key, "config": echo})
}

/// `search_tree_native`: the full tree plus the echoed config.
#[must_use]
pub fn search_tree(cs: &CombatState, config: &Value) -> Value {
    let echo = config_obj(config);
    match params_for(config, "mcts_search_tree") {
        Ok(p) => {
            let mut tree = search_tree_value(cs, &p);
            tree["config"] = echo;
            tree
        }
        Err(e) => json!({"error": e, "nodes": [], "root": Value::Null, "pv": [], "config": echo}),
    }
}

// ── decision_advisor.py candidates ──────────────────────────────────────

/// One evaluable option of the current decision: the join key the SPA
/// renders its option under, a log label, and the engine action.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Candidate {
    pub key: String,
    pub label: String,
    pub action: RunAction,
}

fn pending_choice(rm: &RunManager) -> Option<Value> {
    rm.combat.as_ref().and_then(rmx::pending_choice)
}

/// A run-level *choose a card* screen (Lead Paperweight / Hefty Tablet):
/// offered through the `card_reward` phase but rendered as a `card_select`.
fn choose_screen(rm: &RunManager) -> bool {
    rmx::phase_str(rm) == "card_reward" && rmx::card_offer_choose_screen(rm)
}

fn choose_screen_candidates(rm: &RunManager) -> Vec<Candidate> {
    let ids = rmx::card_reward_option_ids(rm);
    let mut out = Vec::new();
    for action in rmx::available_actions(rm) {
        let RunAction::PickCardReward { idx } = action else {
            continue;
        };
        match idx {
            None => out.push(Candidate {
                key: "choice:skip".into(),
                label: "skip".into(),
                action,
            }),
            Some(i) => out.push(Candidate {
                key: format!("choice:{i}"),
                label: ids
                    .get(i)
                    .map_or_else(|| format!("card {i}"), |id| loc_id(id)),
                action,
            }),
        }
    }
    out
}

fn candidate_key_label(action: RunAction) -> Option<(String, String)> {
    Some(match action {
        RunAction::CombatResolveChoice { option: None } => ("choice:skip".into(), "skip".into()),
        RunAction::CombatResolveChoice { option: Some(o) } => {
            (format!("choice:{o}"), format!("option {o}"))
        }
        RunAction::ShopBuyCard { idx } => (format!("card:{idx}"), format!("buy card {idx}")),
        RunAction::ShopBuyRelic { idx } => (format!("relic:{idx}"), format!("buy relic {idx}")),
        RunAction::ShopBuyPotion { idx } => (format!("potion:{idx}"), format!("buy potion {idx}")),
        RunAction::LeaveShop => ("leave".into(), "leave".into()),
        RunAction::EventChoice { idx } => (format!("option:{idx}"), format!("option {idx}")),
        RunAction::PickBossRelic { idx } => (format!("boss_relic:{idx}"), format!("relic {idx}")),
        _ => return None,
    })
}

/// `_candidates`: `(key, label, action)` per evaluable option of the current
/// decision, straight off the engine's legal moves.
#[must_use]
pub fn candidates(rm: &RunManager) -> Vec<Candidate> {
    if choose_screen(rm) {
        return choose_screen_candidates(rm);
    }
    let pc = pending_choice(rm);
    let choice_cards: Vec<Value> = pc
        .as_ref()
        .and_then(|p| p.get("cards"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut out = Vec::new();
    for action in rmx::available_actions(rm) {
        let Some((key, mut label)) = candidate_key_label(action) else {
            continue;
        };
        if let RunAction::CombatResolveChoice { option: Some(o) } = action {
            if let Some(cid) = choice_cards.get(o).and_then(|c| c["id"].as_str()) {
                if !cid.is_empty() {
                    label = loc_id(cid);
                }
            }
        }
        out.push(Candidate { key, label, action });
    }
    out
}

// ── choice_ranker.py ────────────────────────────────────────────────────

/// `is_rankable`: an in-combat pending choice with ≥2 legal options.
#[must_use]
pub fn is_rankable(rm: &RunManager) -> bool {
    rmx::phase_str(rm) == "combat" && pending_choice(rm).is_some() && candidates(rm).len() >= 2
}

/// `choice_ranker.rank`: `{candidates: [{key, label, win_prob, q, visits,
/// is_best}], decision: "card_select", act, config}` or `{error,
/// candidates: [], config}`.
#[must_use]
pub fn rank(rm: &RunManager, config: &Value) -> Value {
    let echo = config_obj(config);
    if !is_rankable(rm) {
        return json!({
            "error": "the choice ranker only runs on an in-combat card-select choice with at least two legal options",
            "candidates": [],
            "config": echo,
        });
    }
    let Some(cs) = rm.combat.as_ref() else {
        return json!({
            "error": "no native combat state (the choice ranker requires the Rust engine backend)",
            "candidates": [],
            "config": echo,
        });
    };
    let ranking = rank_choice_native(cs, config);
    if let Some(e) = ranking.get("error") {
        return json!({"error": e, "candidates": [], "config": echo});
    }
    let by_key = &ranking["by_key"];
    let mut rows: Vec<Value> = candidates(rm)
        .into_iter()
        .map(|c| {
            let row = by_key.get(&c.key);
            json!({
                "key": c.key,
                "label": c.label,
                "win_prob": row.map_or(Value::Null, |r| r["win_prob"].clone()),
                "q": row.map_or(Value::Null, |r| r["q"].clone()),
                "visits": row.map_or(0, |r| r["visits"].as_i64().unwrap_or(0)),
                "is_best": row.is_some_and(|r| r["is_best"].as_bool().unwrap_or(false)),
            })
        })
        .collect();
    // If the search marked no row best (e.g. a single visited option), fall
    // back to the highest win_prob among scored rows so the UI always has one.
    if !rows.is_empty() && !rows.iter().any(|r| r["is_best"].as_bool() == Some(true)) {
        let best = rows
            .iter()
            .enumerate()
            .filter(|(_, r)| !r["win_prob"].is_null())
            .max_by(|(_, a), (_, b)| {
                (f64_of(&a["win_prob"]), a["visits"].as_i64().unwrap_or(0))
                    .partial_cmp(&(f64_of(&b["win_prob"]), b["visits"].as_i64().unwrap_or(0)))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i);
        if let Some(i) = best {
            rows[i]["is_best"] = json!(true);
        }
    }
    json!({
        "candidates": rows,
        "decision": "card_select",
        "act": rm.run_state.act_index,
        "config": ranking["config"].clone(),
    })
}

/// `choice_ranker.best_action`: the engine action behind the option
/// [`rank`] marks best (`None` when not rankable / the search failed).
#[must_use]
pub fn best_action(rm: &RunManager, config: &Value) -> Option<RunAction> {
    let out = rank(rm, config);
    if out.get("error").is_some() {
        return None;
    }
    let best_key = out["candidates"]
        .as_array()?
        .iter()
        .find(|c| c["is_best"].as_bool() == Some(true))?["key"]
        .as_str()?
        .to_string();
    candidates(rm)
        .into_iter()
        .find(|c| c.key == best_key)
        .map(|c| c.action)
}

// ── regret_review.py ────────────────────────────────────────────────────

fn match_row(scores: &[Value], stable_idx: Option<i64>, target: Option<i64>) -> Option<&Value> {
    let stable_idx = stable_idx?;
    let same_idx = |r: &&Value| r["stable_idx"].as_i64() == Some(stable_idx);
    scores
        .iter()
        .filter(same_idx)
        .find(|r| r["target_index"].as_i64() == target)
        .or_else(|| scores.iter().find(same_idx))
}

fn best_row(scores: &[Value]) -> &Value {
    scores
        .iter()
        .find(|r| r["is_best"].as_bool() == Some(true))
        .unwrap_or_else(|| {
            scores
                .iter()
                .max_by(|a, b| {
                    win_prob(a)
                        .partial_cmp(&win_prob(b))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .expect("non-empty")
        })
}

/// `_enemy_name`: the label of the enemy whose `index` is `target` — its
/// name plus `" #k"` when several enemies share it
/// ([`crate::enemy_labels`]), so a "wrong rat" blunder says which rat.
fn enemy_name(enemies: &[Value], target: Option<i64>) -> Option<String> {
    let target = target?;
    let e = enemies
        .iter()
        .find(|e| e["index"].as_i64() == Some(target))?;
    e["name"].as_str().filter(|n| !n.is_empty())?;
    crate::enemy_labels::enemy_label_by_index(enemies, target)
}

/// `_targeted_label`: the row's label, with `→ <enemy>` when the fight has
/// more than one enemy and the target resolves.
fn targeted_label(row: &Value, enemies: &[Value]) -> String {
    let base = row["action"]
        .as_str()
        .filter(|s| !s.is_empty())
        .unwrap_or("?");
    if enemies.len() <= 1 {
        return base.to_string();
    }
    match enemy_name(enemies, row["target_index"].as_i64()) {
        Some(name) => format!("{base} → {name}"),
        None => base.to_string(),
    }
}

/// Python `round(p * 100)`: ties to even.
#[allow(clippy::cast_possible_truncation)]
fn pct(p: f64) -> i64 {
    (p * 100.0).round_ties_even() as i64
}

/// `_native_action`: a branch node's `(verb, args)` → the native action dict
/// `assess_decision` keys on. Both index spellings are accepted per verb.
/// The workbench's own `use_potion` verb carries `potion_index`, which the
/// Python normaliser did not read (so it never matched a potion row); it
/// is read here.
#[must_use]
pub fn native_action(verb: Option<&str>, args: &Value) -> Value {
    native_action_with(verb, args, true)
}

/// `_native_action` exactly as Python normalises it — WITHOUT reading a
/// `use_potion`'s `potion_index`, so a hand-played potion records
/// `slot: null`. The run log is written with this so the recorded line hashes
/// like the oracle's; the live coach and verdict keep [`native_action`].
#[must_use]
pub fn native_action_recorded(verb: Option<&str>, args: &Value) -> Value {
    native_action_with(verb, args, false)
}

fn native_action_with(verb: Option<&str>, args: &Value, read_potion_index: bool) -> Value {
    let a = args.as_object().cloned().unwrap_or_default();
    let first = |keys: &[&str]| -> Value {
        keys.iter()
            .find_map(|k| a.get(*k).filter(|v| !v.is_null()).cloned())
            .unwrap_or(Value::Null)
    };
    let kind = if verb == Some(crate::render::NATIVE_ACTION) {
        a.get("action").and_then(Value::as_str).map(str::to_string)
    } else {
        verb.map(str::to_string)
    };
    match kind.as_deref() {
        Some("play_card" | "play") => json!({
            "action": "play",
            "card_index": first(&["card_index", "hand_idx"]),
            "target_index": a.get("target_index").cloned().unwrap_or(Value::Null),
        }),
        Some("end_turn") => json!({"action": "end_turn"}),
        Some("drink_potion" | "potion" | "use_potion") => json!({
            "action": "drink_potion",
            "slot": if read_potion_index {
                first(&["slot", "potion_slot", "potion_index"])
            } else {
                first(&["slot", "potion_slot"])
            },
            "target_index": a.get("target_index").cloned().unwrap_or(Value::Null),
        }),
        _ => Value::Object(a),
    }
}

/// What `assess_decision` needs of one combat decision.
#[derive(Clone, Debug)]
pub struct DecisionStep {
    pub idx: i64,
    pub node_id: Option<i64>,
    /// The (name-resolved) combat state the move was chosen from.
    pub state: Value,
    /// Native action dict of the move taken (see [`native_action`]).
    pub chosen: Value,
    pub turn: Value,
    pub floor: Value,
    pub act: Value,
    /// The search's recommended line from this position, labelled
    /// (see [`label_line`]); empty when unknown.
    pub best_line: Vec<String>,
}

/// `_round_half_up`: both twins compute `floor(x + 0.5)` — round half up,
/// which for `x >= 0` is half-away-from-zero (so 2.5 → 3, unlike Python's
/// banker's `round`). The explicit formula, not `f64::round`, keeps the two
/// bit-identical even where `x + 0.5` itself rounds in binary.
#[allow(clippy::cast_possible_truncation)]
fn round_half_up(x: f64) -> i64 {
    (x + 0.5).floor() as i64
}

/// `_hp_known`: a numeric `mean_final_hp` over at least [`HP_MIN_SAMPLES`]
/// resolved playouts (a missing `hp_n` counts as 0).
fn hp_known(row: &Value) -> bool {
    row["mean_final_hp"].as_f64().is_some() && py_int(&row["hp_n"]) >= HP_MIN_SAMPLES
}

/// `_hp_gap`: how much more mean final HP `best` keeps than `played`, or 0
/// when that is within [`HP_GAP_SE`] standard errors of the difference (a
/// noise gap) or negative. Clamping only after the noise test keeps
/// `grade_turn`'s per-turn sum from accumulating one-sided jitter. Rows
/// without `hp_sd` (logged before it existed) skip the test.
fn hp_gap(best: &Value, played: &Value) -> f64 {
    let gap = f64_of(&best["mean_final_hp"]) - f64_of(&played["mean_final_hp"]);
    if let (Some(sb), Some(sp)) = (best["hp_sd"].as_f64(), played["hp_sd"].as_f64()) {
        #[allow(clippy::cast_precision_loss)]
        let var = sb * sb / py_int(&best["hp_n"]) as f64 + sp * sp / py_int(&played["hp_n"]) as f64;
        if gap < HP_GAP_SE * var.sqrt() {
            return 0.0;
        }
    }
    gap.max(0.0)
}

/// A position's name-resolved `combat_play` view, built from the engine
/// state the way the session builds a combat node's `state` (the same
/// `build_combat_view` → `resolve_state_names` the hand / potion labels of
/// [`rows_from_mcts`] read).
fn named_combat_state(cs: &CombatState, loc: &Loc) -> Value {
    let view = build_combat_view(&rmx::wire(cs), "", &[], &[], &[]);
    resolve_state_names(Some(&view), loc)
}

/// The legal action at `cs` whose native dict is `v`, if any.
fn legal_action_for(cs: &CombatState, v: &Value) -> Option<Action> {
    sts2_core::legal_actions(cs)
        .into_iter()
        .find(|a| combat_action_to_value(a) == *v)
}

/// The search's principal variation as human labels: each PV action is
/// applied in order to a clone of `cs` and labelled from the position it is
/// played in — so after `play 0` the next `play 0` names the card that slid
/// into slot 0 — with the label [`rows_from_mcts`] gives. Stops after the
/// first `end_turn`, when the combat ends, at an action that is not legal
/// where it lands, or after the first action that puts a card into the hand
/// (a draw or a random generation): the clone replays with the real engine
/// RNG, so any label past that point could name the actual hidden draw — or
/// a card the search never saw. A pending-choice pick inside the line is
/// applied but not labelled ([`rows_from_mcts`] does not surface those
/// either).
#[must_use]
pub fn label_line(cs: &CombatState, pv: &[Value], loc: &Loc) -> Vec<String> {
    walk_line(cs, pv, loc).0
}

/// [`label_line`], plus the position the line ended the turn from: the
/// one its End turn was played in, or the won / lost position — `None` when
/// the line stopped short of either (`realgame_coach.CoachEngine.ai_turn`).
fn walk_line(cs: &CombatState, pv: &[Value], loc: &Loc) -> (Vec<String>, Option<CombatState>) {
    let mut state = cs.clone();
    let mut out = Vec::new();
    let mut end = None;
    for v in pv {
        if state.is_over {
            break;
        }
        let Some(action) = legal_action_for(&state, v) else {
            break;
        };
        if let Some(idx) = row_stable_idx(v) {
            out.push(action_label(idx, &named_combat_state(&state, loc)));
        }
        if matches!(action, Action::EndTurn) {
            end = Some(state.clone());
            break;
        }
        let hand_before = state.player.hand.clone();
        let _ = sts2_core::apply(&mut state, action);
        if state.is_over {
            end = Some(state);
            break;
        }
        let hand_gained = state
            .player
            .hand
            .iter()
            .any(|uid| !hand_before.contains(uid));
        if hand_gained {
            break;
        }
    }
    (out, end)
}

/// Powers whose removal is an enemy waking up (C# `AsleepPower` /
/// `SlumberPower`): `realgame_coach._SLEEP_POWERS`.
const SLEEP_POWERS: [&str; 2] = ["ASLEEP", "SLUMBER"];

fn asleep(enemy: &Value) -> bool {
    enemy["powers"].as_array().is_some_and(|ps| {
        ps.iter().any(|p| {
            p["name"]
                .as_str()
                .is_some_and(|n| SLEEP_POWERS.contains(&n))
        })
    })
}

/// `realgame_coach.board_events`: what a line visibly did to the enemies
/// between two engine boards ([`rmx::wire`]) — `("wakes", i)` /
/// `("kills", i)` per encounter slot, or a lone `("wins", -1)`.
fn board_events(start: &Value, end: &Value) -> Vec<(&'static str, i64)> {
    if end["phase"] == "Victory" {
        return vec![("wins", -1)];
    }
    let empty = Vec::new();
    let after = end["encounter"].as_array().unwrap_or(&empty);
    let mut out = Vec::new();
    for (i, e) in start["encounter"]
        .as_array()
        .unwrap_or(&empty)
        .iter()
        .enumerate()
    {
        if e["is_dead"].as_bool() == Some(true) {
            continue;
        }
        let Some(f) = after.iter().find(|f| f["uid"] == e["uid"]) else {
            continue;
        };
        if f["is_dead"].as_bool() == Some(true) {
            out.push(("kills", i as i64));
        } else if asleep(e) && !asleep(f) {
            out.push(("wakes", i as i64));
        }
    }
    out
}

/// `realgame_coach.why_text`: the recap's reason line — what one line did on
/// the board that the other did not, e.g. "Your line wakes Lagavulin
/// Matriarch on turn 1" (GH #888). `None` when both lines did the same.
fn why_text(
    you: &[(&'static str, i64)],
    ai: &[(&'static str, i64)],
    names: &[String],
    turn: &Value,
) -> Option<String> {
    let phrase = |&(kind, i): &(&str, i64)| {
        if kind == "wins" {
            return "wins the fight".to_string();
        }
        let name = usize::try_from(i)
            .ok()
            .and_then(|i| names.get(i).cloned())
            .unwrap_or_else(|| format!("enemy {}", i + 1));
        format!("{kind} {name}")
    };
    let mut parts = Vec::new();
    for (subject, mine, theirs) in [("your line", you, ai), ("the AI's line", ai, you)] {
        let only: Vec<String> = mine
            .iter()
            .filter(|e| !theirs.contains(e))
            .map(phrase)
            .collect();
        if !only.is_empty() {
            parts.push(format!("{subject} {}", only.join(" and ")));
        }
    }
    if parts.is_empty() {
        return None;
    }
    let mut text = parts.join("; ");
    if let Some(t) = turn.as_i64() {
        text.push_str(&format!(" on turn {t}"));
    }
    let mut chars = text.chars();
    chars
        .next()
        .map(|c| c.to_uppercase().chain(chars).collect())
}

/// The reason line of a graded turn: the boards the turn started from
/// (`start`), the player's End turn was played from (`you_end`) and the AI's
/// line ended at (`ai_end`). Enemy names are the turn context's board labels
/// by encounter slot, falling back to the view's name.
fn turn_why(
    start: &CombatState,
    you_end: &CombatState,
    ai_end: Option<&CombatState>,
    named: &Value,
) -> Option<String> {
    let ai_end = ai_end?;
    let start = rmx::wire(start);
    let empty = Vec::new();
    let mut names: Vec<String> = start["encounter"]
        .as_array()
        .unwrap_or(&empty)
        .iter()
        .map(|e| crate::display_maps::monster_title(e["kind"].as_str().unwrap_or("?")))
        .collect();
    let view = named["enemies"].as_array().unwrap_or(&empty);
    for (e, label) in view.iter().zip(crate::enemy_labels::enemy_labels(view)) {
        if let Some(slot) = e["index"].as_u64().and_then(|i| names.get_mut(i as usize)) {
            *slot = label;
        }
    }
    why_text(
        &board_events(&start, &rmx::wire(you_end)),
        &board_events(&start, &rmx::wire(ai_end)),
        &names,
        &named["round"],
    )
}

/// Index of the score row for `(stable_idx, target)` — [`match_row`]'s
/// exact-then-`stable_idx`-only rule.
fn match_row_index(
    scores: &[Value],
    stable_idx: Option<i64>,
    target: Option<i64>,
) -> Option<usize> {
    let stable_idx = stable_idx?;
    let same_idx = |r: &Value| r["stable_idx"].as_i64() == Some(stable_idx);
    scores
        .iter()
        .position(|r| same_idx(r) && r["target_index"].as_i64() == target)
        .or_else(|| scores.iter().position(same_idx))
}

/// Whether [`top_up_played`] would re-search the played move's row: it is in
/// `scores`, is not the recommended row, and has fewer than
/// [`PLAYED_MIN_VISITS`] visits. Cheap — lets callers skip building the
/// parent position when nothing would be topped up.
#[must_use]
pub(crate) fn played_needs_top_up(scores: &[Value], played: &Value) -> bool {
    let (idx, target) = action_stable_key(played);
    match_row_index(scores, idx, target).is_some_and(|i| {
        let row = &scores[i];
        row["is_best"].as_bool() != Some(true) && py_int(&row["visits"]) < PLAYED_MIN_VISITS
    })
}

/// Force a gradable sample on the played move: when its row (matched from
/// the native action dict `played`) has fewer than [`PLAYED_MIN_VISITS`]
/// visits, its stats — `win_prob`, `q`, `visits`, `wins` / `losses` /
/// `ties`, `mean_final_hp`, `hp_n` — are replaced by a forced-root search
/// ([`score_action_forced`]) rooted at `cs`, the PARENT position the move
/// was chosen from, under the main search's config and hidden-info mode.
/// Every simulation re-deals at the move's chance node, so the row never
/// prices the realized child's draw (applying the move to `cs` with its real
/// RNG — what [`sts2_wire::emit::mcts::score_action`] does — would). `stable_idx`, `action` and `target_index` are kept. The
/// recommended (`is_best`) row is never topped up: a move that matches the
/// line is graded "best" whatever its visit count.
pub fn top_up_played(
    cs: &CombatState,
    scores: &mut [Value],
    played: &Value,
    params: &RecommendParams,
) {
    if !played_needs_top_up(scores, played) {
        return;
    }
    let (idx, target) = action_stable_key(played);
    let Some(i) = match_row_index(scores, idx, target) else {
        return;
    };
    let row = &scores[i];
    let key = (row["stable_idx"].as_i64(), row["target_index"].as_i64());
    let Some(action) = sts2_core::legal_actions(cs).into_iter().find(|a| {
        let v = combat_action_to_value(a);
        (row_stable_idx(&v), v["target_index"].as_i64()) == key
    }) else {
        return;
    };
    let Some(fresh) = score_action_forced(cs, params, &action) else {
        return;
    };
    let Some(m) = scores[i].as_object_mut() else {
        return;
    };
    m.insert("win_prob".into(), json!(f64_of(&fresh["p_win"])));
    m.insert("q".into(), json!(f64_of(&fresh["q"])));
    for k in ["visits", "wins", "losses", "ties", "hp_n"] {
        m.insert(k.into(), json!(fresh[k].as_i64().unwrap_or(0)));
    }
    m.insert(
        "mean_final_hp".into(),
        json!(fresh["mean_final_hp"].as_f64()),
    );
    m.insert("hp_sd".into(), json!(fresh["hp_sd"].as_f64()));
    m.insert("is_best".into(), json!(false));
}

/// `assess_decision`: one combat step diffed against the MCTS line, or
/// `None` when it can't be assessed (fewer than two rows, no resolvable
/// recorded move).
#[must_use]
pub fn assess_decision(step: &DecisionStep, scores: &[Value], blunder_pct: f64) -> Option<Value> {
    if scores.len() < 2 {
        return None;
    }
    let (played_idx, played_target) = action_stable_key(&step.chosen);
    let played = match_row(scores, played_idx, played_target)?;
    let best = best_row(scores);
    let best_wp = win_prob(best);
    let played_wp = win_prob(played);
    let regret = (best_wp - played_wp).max(0.0);
    let matched_best = played["stable_idx"] == best["stable_idx"]
        && played["target_index"] == best["target_index"];
    let rank = scores.iter().filter(|r| win_prob(r) > played_wp).count();
    let drop_pct = pct(regret);
    let hp = (hp_known(played) && hp_known(best)).then(|| {
        let played_hp = f64_of(&played["mean_final_hp"]);
        let best_hp = f64_of(&best["mean_final_hp"]);
        (played_hp, best_hp, hp_gap(best, played))
    });
    let hp_drop = hp.map(|(_, _, d)| d);
    // `regret * 100.0` in f64, as Python computes it: 1.0 − 0.92 gives
    // 7.999…, a slip, though it renders as −8 pts.
    let severity = (regret * 100.0).max(hp_drop.unwrap_or(0.0));
    let verdict = if matched_best {
        "best"
    } else if py_int(&played["visits"]) < PLAYED_MIN_VISITS {
        "uncertain"
    } else if severity < TIE_PTS {
        "tie"
    } else if severity >= blunder_pct {
        "blunder"
    } else {
        "slip"
    };
    let is_blunder = verdict == "blunder";
    let enemies: Vec<Value> = step.state["enemies"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let played_label = targeted_label(played, &enemies);
    let best_label = targeted_label(best, &enemies);
    let turn = &step.turn;
    let turn_s = match turn {
        Value::Null => "None".to_string(),
        Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    let message = if matched_best {
        format!(
            "Turn {turn_s}: {played_label} matched the MCTS line (P(win) {}%)",
            pct(played_wp)
        )
    } else {
        let mut m = format!(
            "Turn {turn_s}: {played_label} instead of {best_label} dropped P(win) {}%→{}% (−{drop_pct} pts)",
            pct(best_wp),
            pct(played_wp)
        );
        if let Some(d) = hp_drop.map(round_half_up).filter(|&d| d >= 1) {
            m.push_str(&format!(", ≈{d} HP"));
        }
        m
    };
    Some(json!({
        "idx": step.idx,
        "node_id": step.node_id,
        "turn": turn,
        "floor": step.floor,
        "act": step.act,
        "played_label": played_label,
        "played_win_prob": played_wp,
        "best_label": best_label,
        "best_win_prob": best_wp,
        "regret": regret,
        "drop_pct": drop_pct,
        "player_rank": rank,
        "n_actions": scores.len(),
        "visits": py_int(&best["visits"]),
        "matched_best": matched_best,
        "is_blunder": is_blunder,
        "message": message,
        "played_mean_hp": hp.map(|(p, _, _)| p),
        "best_mean_hp": hp.map(|(_, b, _)| b),
        "hp_drop": hp_drop,
        "severity": severity,
        "verdict": verdict,
        "best_line": step.best_line,
    }))
}

/// Python `round(x, n)`: exact-value rounding, ties to even.
fn py_round(x: f64, n: usize) -> f64 {
    format!("{x:.n$}").parse().unwrap_or(x)
}

/// `regret_review.grade_turn`: one combat turn graded from its non-forced
/// actions' regrets (`assess_decision` values; `None` = could not be graded).
/// `None` (stay silent) when any action is ungraded or `uncertain`; else
/// `{verdict, hp_drop, severity, win_before, win_after}`.
///
/// ```text
/// counted  = actions with severity >= TURN_ACTION_FLOOR
/// severity = max(sum drop_pct, sum hp_drop) over counted
/// verdict  = best (none counted) | tie (< TIE_PTS)
///            | blunder (>= DEFAULT_BLUNDER_DROP_PCT) | slip
/// ```
///
/// Twin of the Python rule; both are pinned by
/// `sts2-pycli/tests/fixtures/coach/grade_turn_cases.json`.
#[must_use]
pub fn grade_turn(regrets: &[Option<Value>]) -> Option<Value> {
    let mut graded = Vec::with_capacity(regrets.len());
    for r in regrets {
        let r = r.as_ref()?;
        if r["verdict"].as_str() == Some("uncertain") {
            return None;
        }
        graded.push(r);
    }
    let counted: Vec<&Value> = graded
        .iter()
        .copied()
        .filter(|r| f64_of(&r["severity"]) >= TURN_ACTION_FLOOR)
        .collect();
    let turn_hp: f64 = counted.iter().map(|r| f64_of(&r["hp_drop"])).sum();
    let turn_pts: f64 = counted.iter().map(|r| f64_of(&r["drop_pct"])).sum();
    let severity = turn_pts.max(turn_hp);
    let verdict = if counted.is_empty() {
        "best"
    } else if severity < TIE_PTS {
        "tie"
    } else if severity >= DEFAULT_BLUNDER_DROP_PCT {
        "blunder"
    } else {
        "slip"
    };
    let hp_known = counted.iter().any(|r| !r["hp_drop"].is_null());
    let first = graded.first();
    Some(json!({
        "verdict": verdict,
        "hp_drop": hp_known.then(|| py_round(turn_hp, 2)),
        "severity": py_round(severity, 2),
        "win_before": first.map(|r| py_round(f64_of(&r["best_win_prob"]), 4)),
        "win_after": first.map(|r| py_round(f64_of(&r["played_win_prob"]), 4)),
    }))
}

/// `summarize`: `{decisions, blunders (top_n costliest), summary}`.
#[must_use]
pub fn summarize(decisions: Vec<Value>, top_n: usize) -> Value {
    let mut flagged: Vec<&Value> = decisions
        .iter()
        .filter(|d| d["is_blunder"].as_bool() == Some(true))
        .collect();
    flagged.sort_by(|a, b| {
        let key = |d: &Value| {
            (
                -f64_of(&d["severity"]),
                d["turn"].as_i64().unwrap_or(1 << 30),
                d["idx"].as_i64().unwrap_or(0),
            )
        };
        let (ka, kb) = (key(a), key(b));
        ka.0.partial_cmp(&kb.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(ka.1.cmp(&kb.1))
            .then(ka.2.cmp(&kb.2))
    });
    let cumulative: f64 = decisions.iter().map(|d| f64_of(&d["regret"])).sum();
    // Summed in decision order, like Python's generator `sum`.
    let cumulative_hp: f64 = decisions.iter().filter_map(|d| d["hp_drop"].as_f64()).sum();
    let worst = flagged
        .first()
        .map_or(0, |d| d["drop_pct"].as_i64().unwrap_or(0));
    let blunders: Vec<Value> = flagged.iter().take(top_n).map(|d| (*d).clone()).collect();
    json!({
        "summary": {
            "total_decisions": decisions.len(),
            "blunder_count": flagged.len(),
            "cumulative_drop_pct": pct(cumulative),
            "cumulative_hp_drop": round_half_up(cumulative_hp),
            "worst_drop_pct": worst,
        },
        "decisions": decisions,
        "blunders": blunders,
    })
}

// ── BranchingLiveSession advisor methods ────────────────────────────────

fn combat_play(node: &Node) -> bool {
    node.decision == "combat_play"
}

/// `hint`: MCTS scores for the active combat state, `None` off combat.
pub fn hint(sess: &Session, config: &Value) -> Result<Option<Value>, SessionError> {
    let node = sess.active_node()?;
    if !combat_play(node) {
        return Ok(None);
    }
    let named = resolve_state_names(Some(&node.state), sess.loc());
    let cs = sess
        .game()
        .and_then(|g| g.combat_state())
        .ok_or_else(|| SessionError::new(400, "no native combat state"))?;
    Ok(Some(score(cs, &named, config)))
}

/// [`hint`], remembering the rows and PV on the session
/// (`Session::hint_scores`) for the per-turn coach to reuse.
pub fn hint_remembered(sess: &mut Session, config: &Value) -> Result<Option<Value>, SessionError> {
    let node = sess.active_node()?;
    if !combat_play(node) {
        return Ok(None);
    }
    let id = node.id;
    let named = resolve_state_names(Some(&node.state), sess.loc());
    let cs = sess
        .game()
        .and_then(|g| g.combat_state())
        .ok_or_else(|| SessionError::new(400, "no native combat state"))?;
    let (out, pv) = score_with_pv(cs, &named, config);
    let sims = params_for(config, "mcts_recommend").map_or(0, |p| p.sims);
    if let Some(rows) = out["scores"].as_array().filter(|r| !r.is_empty()) {
        sess.hint_scores.insert(id, (rows.clone(), pv, sims));
    }
    Ok(Some(out))
}

/// `search_tree`: the full tree for the active combat state, `None` off combat.
pub fn live_search_tree(sess: &Session, config: &Value) -> Result<Option<Value>, SessionError> {
    let node = sess.active_node()?;
    if !combat_play(node) {
        return Ok(None);
    }
    let cs = sess
        .game()
        .and_then(|g| g.combat_state())
        .ok_or_else(|| SessionError::new(400, "no native combat state"))?;
    Ok(Some(search_tree(cs, config)))
}

/// `rank_choice`: the ranker over the active in-combat choice, `None` when
/// the active node is not a `card_select`.
pub fn rank_choice(sess: &Session, config: &Value) -> Result<Option<Value>, SessionError> {
    let node = sess.active_node()?;
    if node.decision != "card_select" {
        return Ok(None);
    }
    let rm = sess
        .game()
        .and_then(|g| g.native_run())
        .ok_or_else(|| SessionError::new(400, "no live run"))?;
    Ok(Some(rank(rm, config)))
}

/// `recommended_action`: `{action, args, policy?}` for the active decision —
/// an open in-combat choice is answered by the combat search at the ranker's
/// default config (so "play best" plays the option the panel stars); every
/// other run-level decision by the tier-1 [`HeuristicPolicy`]. `None` in plain
/// combat (the hint owns card plays), at game over, or when the chosen action
/// has no workbench verb.
pub fn recommended_action(sess: &Session) -> Result<Option<Value>, SessionError> {
    let node = sess.active_node()?;
    let game = sess
        .game()
        .ok_or_else(|| SessionError::new(400, "no live run"))?;
    let rm = game
        .native_run()
        .ok_or_else(|| SessionError::new(400, "no live run"))?;
    let options = normalize_options(&node.state);
    if let Some(action) = best_action(rm, &json!({})) {
        return Ok(game
            .verb_for(action, &options)
            .map(|(verb, args)| json!({"action": verb, "args": args, "policy": "combat_mcts"})));
    }
    if rmx::phase_str(rm) == "combat" || rm.is_over() {
        return Ok(None);
    }
    let actions = rmx::available_actions(rm);
    if actions.is_empty() {
        return Ok(None);
    }
    let Some(i) = HeuristicPolicy::new().choose(rm, &actions) else {
        return Ok(None);
    };
    Ok(game
        .verb_for(actions[i], &options)
        .map(|(verb, args)| json!({"action": verb, "args": args})))
}

fn pop_f64(cfg: &mut Value, key: &str, default: f64) -> f64 {
    cfg.as_object_mut()
        .and_then(|m| m.remove(key))
        .and_then(|v| v.as_f64())
        .unwrap_or(default)
}

fn pop_usize(cfg: &mut Value, key: &str, default: usize) -> usize {
    cfg.as_object_mut()
        .and_then(|m| m.remove(key))
        .and_then(|v| v.as_u64())
        .map_or(default, |v| usize::try_from(v).unwrap_or(usize::MAX))
}

fn step_of(
    parent: &Node,
    node: &Node,
    loc: &'static crate::localization::Loc,
    floor: &Value,
    act: &Value,
) -> DecisionStep {
    let named = resolve_state_names(Some(&parent.state), loc);
    let turn = named.get("round").cloned().unwrap_or(Value::Null);
    DecisionStep {
        idx: i64::try_from(parent.idx).unwrap_or(i64::MAX),
        node_id: i64::try_from(parent.id).ok(),
        state: named,
        chosen: native_action(node.action.as_deref(), &node.args),
        turn,
        floor: floor.clone(),
        act: act.clone(),
        best_line: Vec::new(),
    }
}

/// `last_move_regret`: regret of the move that produced the active node —
/// `None` when it has no parent or the parent was not a combat decision;
/// `{error, regret: null}` when the parent cannot be scored; else a
/// `DecisionRegret` dict (or `None` if the move is unresolvable).
pub fn last_move_regret(sess: &Session, config: &Value) -> Result<Option<Value>, SessionError> {
    let node = sess.active_node()?;
    let Some(parent) = node.parent_id.and_then(|p| sess.node(p)) else {
        return Ok(None);
    };
    if !combat_play(parent) {
        return Ok(None);
    }
    let mut cfg = config_obj(config);
    let blunder_pct = pop_f64(&mut cfg, "blunder_pct", DEFAULT_BLUNDER_DROP_PCT);
    let mut step = step_of(parent, node, sess.loc(), &Value::Null, &Value::Null);
    let no_scores = || json!({"error": "no MCTS scores available", "regret": Value::Null});
    let (Some(cs), Ok(params)) = (parent.combat.as_ref(), params_for(&cfg, "mcts_recommend"))
    else {
        return Ok(Some(no_scores()));
    };
    let (result, pv) = score_with_pv(cs, &step.state, &cfg);
    let mut scores: Vec<Value> = result["scores"].as_array().cloned().unwrap_or_default();
    if scores.is_empty() {
        return Ok(Some(no_scores()));
    }
    top_up_played(cs, &mut scores, &step.chosen, &params);
    step.best_line = label_line(cs, &pv, sess.loc());
    step.node_id = None; // the live coach's step carries no node id
    Ok(assess_decision(&step, &scores, blunder_pct))
}

/// `realgame_coach.intent_text`: one enemy's telegraphed move as display
/// text — `"attack 6"`, `"attack 6×2"`, another intent as its lower-cased
/// type, joined by `" + "` (`"attack 6×2 + debuff"`); `"unknown"` when
/// nothing is telegraphed. Reads the workbench view's `type` and the mod
/// wire's `intent_type`. Pinned with the Python one by
/// `sts2-pycli/tests/fixtures/coach/intent_text_cases.json`.
#[must_use]
pub fn intent_text(intents: &[Value]) -> String {
    let mut parts: Vec<String> = Vec::new();
    for intent in intents.iter().filter(|i| i.is_object()) {
        let text = if let Some(damage) = intent["damage"].as_i64() {
            let hits = intent["hits"].as_i64().unwrap_or(1);
            if hits > 1 {
                format!("attack {damage}×{hits}")
            } else {
                format!("attack {damage}")
            }
        } else {
            let kind = intent["intent_type"]
                .as_str()
                .or_else(|| intent["type"].as_str())
                .unwrap_or("")
                .trim();
            if kind.is_empty() {
                continue;
            }
            kind.to_lowercase()
        };
        if !parts.contains(&text) {
            parts.push(text);
        }
    }
    if parts.is_empty() {
        "unknown".to_string()
    } else {
        parts.join(" + ")
    }
}

/// `realgame_coach.turn_context` for a workbench combat view: the position
/// a turn started from — `{hand, energy, max_energy, hp, max_hp, block,
/// enemies: [{name, hp, intent}]}`. The view lists live enemies only; the
/// Python twin reads the mod's wire, which can still carry a dying one, and
/// [`crate::enemy_labels`] gives such an entry no number — so both number
/// the same live enemies the same way.
#[must_use]
pub fn turn_context(named: &Value) -> Value {
    let empty = Vec::new();
    let hand: Vec<Value> = named["hand"]
        .as_array()
        .unwrap_or(&empty)
        .iter()
        .map(|c| {
            json!(c["name"]
                .as_str()
                .or_else(|| c["id"].as_str())
                .unwrap_or("?"))
        })
        .collect();
    let view_enemies = named["enemies"].as_array().unwrap_or(&empty);
    // `name` is the board label: a name shared by several enemies carries
    // its `" #k"` ([`crate::enemy_labels`]), so the recap says WHICH rat.
    let labels = crate::enemy_labels::enemy_labels(view_enemies);
    let enemies: Vec<Value> = view_enemies
        .iter()
        .zip(labels)
        .map(|(e, label)| {
            json!({
                "name": label,
                "hp": e["hp"].as_i64(),
                "intent": intent_text(e["intents"].as_array().unwrap_or(&empty)),
            })
        })
        .collect();
    let player = &named["player"];
    json!({
        "hand": hand,
        "energy": named["energy"].as_i64(),
        "max_energy": named["max_energy"].as_i64(),
        "hp": player["hp"].as_i64(),
        "max_hp": player["max_hp"].as_i64(),
        "block": player["block"].as_i64(),
        "enemies": enemies,
    })
}

/// Default cap on fresh searches one turn grade may run: the worker is
/// single-threaded, so a long turn with no remembered hint scores stays
/// ungraded rather than stall the player's next action.
pub const TURN_GRADE_MAX_SEARCHES: usize = 6;

/// Default sims of a turn grade's search: the live advisor's `--sims`
/// (`realgame_advisor.py`), so the browser coach and the mod's grade a turn
/// from the same search strength. At the engine's 300 the root's children
/// split ~75 playouts each and an under-played turn's HP cost drowns in the
/// noise (a Strike-then-End-turn graded `best`).
pub const TURN_GRADE_SIMS: u32 = 8000;

/// Forced-root sims for a played-move top-up: `realgame_coach.TOP_UP_SIMS`.
pub const TURN_GRADE_TOP_UP_SIMS: u32 = 400;

/// `last_turn_grade`: the combat turn that the active node's End turn just
/// closed, graded whole (`grade_turn`) — the PROTOCOL.md `coach` `turn`
/// record, so the live chip and follow mode share one renderer.
///
/// Walks back from the End turn through the turn's moves (same fight, same
/// round). Each move is scored from its remembered hint search
/// (`Session::hint_scores`, if it ran at least the grade's `sims`, default
/// [`TURN_GRADE_SIMS`]) or a fresh search — at most `max_searches`
/// (default [`TURN_GRADE_MAX_SEARCHES`]) of them; past the cap the move is
/// ungraded and the turn silent — then topped up and diffed like
/// [`last_move_regret`]. A forced move (one legal action) is labelled, not
/// graded; an in-combat card select counts in `ungraded_actions`.
///
/// `None` when the active node is not the result of an End turn, or when
/// the turn is silent (a move ungraded or still `uncertain`);
/// `{error, turn: null}` when nothing can be scored.
pub fn last_turn_grade(sess: &Session, config: &Value) -> Result<Option<Value>, SessionError> {
    let node = sess.active_node()?;
    let Some(end) = node.parent_id.and_then(|p| sess.node(p)) else {
        return Ok(None);
    };
    if !combat_play(end)
        || native_action(node.action.as_deref(), &node.args)["action"] != "end_turn"
    {
        return Ok(None);
    }
    let Some(round) = end.combat.as_ref().map(|c| c.round) else {
        return Ok(None);
    };
    let mut steps: Vec<(&Node, &Node)> = vec![(end, node)];
    let mut cur = end;
    while let Some(prev) = cur.parent_id.and_then(|p| sess.node(p)) {
        if prev.combat.as_ref().map(|c| c.round) != Some(round) {
            break;
        }
        steps.push((prev, cur));
        cur = prev;
    }
    steps.reverse();

    let mut cfg = config_obj(config);
    let blunder_pct = pop_f64(&mut cfg, "blunder_pct", DEFAULT_BLUNDER_DROP_PCT);
    let max_searches = pop_usize(&mut cfg, "max_searches", TURN_GRADE_MAX_SEARCHES);
    let peeked = cfg
        .as_object_mut()
        .and_then(|m| m.remove("peeked"))
        .is_some_and(|v| v.as_bool() == Some(true));
    if let Some(m) = cfg.as_object_mut() {
        m.entry("sims").or_insert(json!(TURN_GRADE_SIMS));
    }
    let Ok(params) = params_for(&cfg, "mcts_recommend") else {
        return Ok(Some(
            json!({"error": "no MCTS scores available", "turn": Value::Null}),
        ));
    };
    let top_up = RecommendParams {
        sims: TURN_GRADE_TOP_UP_SIMS,
        ..params
    };

    let mut you: Vec<String> = Vec::new();
    let mut regrets: Vec<Option<Value>> = Vec::new();
    let mut ungraded_actions = 0usize;
    let mut searches = 0usize;
    // The turn's first graded position and its search's PV: the AI's line.
    let mut first: Option<(&CombatState, Vec<Value>, Vec<Value>)> = None;
    for (parent, child) in &steps {
        if !combat_play(parent) {
            ungraded_actions += 1;
            continue;
        }
        let step = step_of(parent, child, sess.loc(), &Value::Null, &Value::Null);
        let Some(cs) = parent.combat.as_ref() else {
            regrets.push(None);
            continue;
        };
        if sts2_core::legal_actions(cs).len() < 2 {
            let (idx, _) = action_stable_key(&step.chosen);
            you.push(idx.map_or_else(|| "?".to_string(), |i| action_label(i, &step.state)));
            continue;
        }
        let hint = sess
            .hint_scores
            .get(&parent.id)
            .filter(|(_, _, sims)| *sims >= params.sims);
        let (mut scores, pv) = if let Some((rows, pv, _)) = hint {
            (rows.clone(), pv.clone())
        } else if searches < max_searches {
            searches += 1;
            let (result, pv) = score_with_pv(cs, &step.state, &cfg);
            (result["scores"].as_array().cloned().unwrap_or_default(), pv)
        } else {
            // Past the cap: silent, whatever the rest of the turn holds.
            return Ok(None);
        };
        top_up_played(cs, &mut scores, &step.chosen, &top_up);
        if first.is_none() && !scores.is_empty() {
            first = Some((cs, pv, scores.clone()));
        }
        let regret = assess_decision(&step, &scores, blunder_pct);
        you.push(regret.as_ref().map_or_else(
            || {
                let (idx, _) = action_stable_key(&step.chosen);
                idx.map_or_else(|| "?".to_string(), |i| action_label(i, &step.state))
            },
            |r| r["played_label"].as_str().unwrap_or("?").to_string(),
        ));
        regrets.push(regret);
    }
    let Some(grade) = grade_turn(&regrets) else {
        return Ok(None);
    };
    let (head, _) = steps[0];
    let named = resolve_state_names(Some(&head.state), sess.loc());
    let mut why = None;
    let ai = first.as_ref().map_or_else(Vec::new, |(cs, pv, scores)| {
        let (line, ai_end) = walk_line(cs, pv, sess.loc());
        if let (Some(start), Some(you_end)) = (head.combat.as_ref(), end.combat.as_ref()) {
            why = turn_why(start, you_end, ai_end.as_ref(), &named);
        }
        if line.is_empty() {
            let best = best_row(scores);
            vec![best["action"].as_str().unwrap_or("?").to_string()]
        } else {
            line
        }
    });
    let mut record = json!({
        "kind": "turn",
        "id": format!("turn:{}", head.id),
        "node_id": head.id,
        "turn": named.get("round").cloned().unwrap_or(Value::Null),
        "floor": Value::Null,
        "you": you,
        "ai": ai,
        "why": why,
        "peeked": peeked,
        "ungraded_actions": ungraded_actions,
        "context": turn_context(&named),
        "searches": searches,
    });
    if let (Some(r), Some(g)) = (record.as_object_mut(), grade.as_object()) {
        for (k, v) in g {
            r.insert(k.clone(), v.clone());
        }
    }
    Ok(Some(record))
}

/// The root distribution of the search that actually chose `node`.
///
/// `Session::run_mode_step` stashes the native agent's `combat_search` on the
/// node the move PRODUCED (`node.state["mcts"]`), so the search for the
/// decision at `parent` lives one node downstream. Rows are labelled and
/// targeted against `parent` — the position the search rooted at — since
/// [`rows_from_mcts`] resolves both out of the state's hand/encounter by index,
/// and by `node` the card is gone and the enemy slots have shifted.
///
/// Empty when the move was not an agent-searched combat play: a hand-played
/// move, an off-combat pick, or a run recorded before this was persisted.
#[must_use]
pub fn agent_scores(
    parent: &Node,
    node: &Node,
    loc: &'static crate::localization::Loc,
) -> Vec<Value> {
    if !combat_play(parent) {
        return Vec::new();
    }
    let Some(search) = node.state.get("mcts").filter(|v| v.is_object()) else {
        return Vec::new();
    };
    if search
        .get("actions")
        .and_then(Value::as_array)
        .is_none_or(std::vec::Vec::is_empty)
    {
        return Vec::new();
    }
    let named = resolve_state_names(Some(&parent.state), loc);
    rows_from_mcts(search, &named)
}

/// `review_run`: the full-run regret review over the active line —
/// `{decisions, blunders, summary, config: {blunder_pct, top_n}}`. Every
/// combat decision on the path is re-scored from its node's snapshot;
/// `floor` / `act` are carried forward from the last screen that reported
/// them.
pub fn review_run(sess: &Session, config: &Value) -> Result<Value, SessionError> {
    let path = sess.active_path()?;
    let mut cfg = config_obj(config);
    let blunder_pct = pop_f64(&mut cfg, "blunder_pct", DEFAULT_BLUNDER_DROP_PCT);
    let top_n = pop_usize(&mut cfg, "top_n", 2);
    let params = params_for(&cfg, "mcts_recommend").ok();
    let mut decisions = Vec::new();
    let mut floor = Value::Null;
    let mut act = Value::Null;
    for pair in path.windows(2) {
        let (parent, node) = (pair[0], pair[1]);
        if let Some(f) = parent.state.get("floor").filter(|v| !v.is_null()) {
            floor = f.clone();
        }
        if let Some(a) = parent.state.get("act").filter(|v| !v.is_null()) {
            act = a.clone();
        }
        if !combat_play(parent) {
            continue;
        }
        let mut step = step_of(parent, node, sess.loc(), &floor, &act);
        // The AGENT'S OWN root distribution when the move came from a searched
        // autoplay step. This is what makes the review agree with the run: it
        // grades the move against the very numbers that picked it, so an
        // autoplayed line reports its real regret (zero, wherever the agent
        // took its own best) instead of a disagreement between two different
        // searches. A hand-played move has none, and falls back to a fresh
        // search below — a DIFFERENT referee, so those rows are advisory.
        let mut scores = agent_scores(parent, node, sess.loc());
        // The agent's stashed search carries no PV, so an autoplayed move
        // names no AI line; a fresh search does.
        let mut pv: Vec<Value> = Vec::new();
        if scores.is_empty() {
            if let Some(cs) = parent.combat.as_ref() {
                let (result, line) = score_with_pv(cs, &step.state, &cfg);
                scores = result["scores"].as_array().cloned().unwrap_or_default();
                pv = line;
            }
        }
        if let (Some(cs), Some(params)) = (parent.combat.as_ref(), params.as_ref()) {
            top_up_played(cs, &mut scores, &step.chosen, params);
            step.best_line = label_line(cs, &pv, sess.loc());
        }
        if let Some(d) = assess_decision(&step, &scores, blunder_pct) {
            decisions.push(d);
        }
    }
    let mut out = summarize(decisions, top_n);
    out["config"] = json!({"blunder_pct": blunder_pct, "top_n": top_n});
    Ok(out)
}

// ── card_advisor.py / decision_advisor.py ───────────────────────────────

pub const DEFAULT_ROLLOUTS: u32 = 12;
pub const DEFAULT_SIMS: u32 = 128;
pub const DEFAULT_THREADS: u32 = 8;
pub const DEFAULT_MAX_STEPS: u32 = 600;
pub const MAX_ROLLOUTS: u32 = 200;
pub const MAX_SIMS: u32 = 8000;

/// The rollout advisors' knobs, resolved from a request config.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RolloutConfig {
    pub rollouts: u32,
    /// `0` = a uniform-random legal move in combat (fast mode).
    pub sims: u32,
    /// Echoed only — the browser search is single-threaded.
    pub threads: u32,
    pub max_steps: u32,
    pub seed: i64,
}

impl RolloutConfig {
    fn from_config(config: &Value, rm: &RunManager) -> Self {
        let get = |k: &str| config.get(k).and_then(Value::as_i64);
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let u = |v: Option<i64>, d: u32| v.map_or(d, |v| v.max(0) as u32);
        Self {
            rollouts: u(get("rollouts"), DEFAULT_ROLLOUTS).clamp(1, MAX_ROLLOUTS),
            sims: u(get("sims"), DEFAULT_SIMS).min(MAX_SIMS),
            threads: u(get("threads"), DEFAULT_THREADS),
            max_steps: u(get("max_steps"), DEFAULT_MAX_STEPS),
            seed: get("seed").unwrap_or_else(|| i64::from(rm.seed & 0x7FFF_FFFF) ^ 0x5EED),
        }
    }

    fn to_value(self) -> Value {
        json!({
            "rollouts": self.rollouts,
            "sims": self.sims,
            "threads": self.threads,
            "max_steps": self.max_steps,
            "seed": self.seed,
        })
    }
}

/// `_playout_rest_of_act`'s result.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Playout {
    pub won: bool,
    pub died: bool,
    pub final_hp: i32,
    pub hp_at_deepest: i32,
    pub floors_advanced: i32,
    pub steps: u32,
    pub reached_act: u8,
    pub inconclusive: bool,
}

/// `RunManager.mcts_pick_action(sims, seed, threads=1)`: the cheap combat
/// search (depth 10, `c_puct = √2`, otherwise the engine defaults) → the
/// engine action, or `None` when the search's pick has no run-action form.
fn mcts_pick_action(rm: &RunManager, sims: u32, seed: u64) -> Option<RunAction> {
    let combat = rm.combat.as_ref()?;
    let cfg = MctsConfig {
        c_puct: std::f32::consts::SQRT_2,
        max_simulations: sims,
        max_rollout_depth: 10,
        seed,
        ..MctsConfig::default()
    };
    let mut mcts = Mcts::new(combat.clone(), cfg);
    let chosen = mcts.try_search().ok()?;
    rmx::action_from_value(&combat_action_to_run_action_value(&chosen)).ok()
}

/// `_playout_rest_of_act`: drive an already-forked + reseeded `rm` to the
/// end of its current act (death, act boundary, a won run, or `max_steps`
/// → inconclusive). Combat: the cheap MCTS at `sims` (uniform-random legal
/// move when `sims == 0` or the search declines); off combat: the tier-1
/// heuristic run policy.
pub fn playout_rest_of_act(
    rm: &mut RunManager,
    sims: u32,
    max_steps: u32,
    mcts_seed: u64,
    rng: &mut rand_xoshiro::Xoshiro256PlusPlus,
) -> Playout {
    let start_act = rm.run_state.act_index;
    let start_floor = rmx::total_floor(rm);
    let mut deepest = start_floor;
    let mut hp_at_deepest = rm.run_state.current_hp;
    let mut steps = 0_u32;
    let mut brain = HeuristicPolicy::new();
    while steps < max_steps {
        if rm.is_over() || rm.run_state.act_index != start_act {
            break;
        }
        // Sample progress out of combat, where `current_hp` is the settled
        // between-rooms HP.
        let floor = rmx::total_floor(rm);
        if floor > deepest {
            deepest = floor;
            hp_at_deepest = rm.run_state.current_hp;
        }
        let action = if rmx::phase_str(rm) == "combat" {
            let picked = if sims > 0 {
                mcts_pick_action(rm, sims, mcts_seed.wrapping_add(u64::from(steps)))
            } else {
                None
            };
            if let Some(a) = picked {
                a
            } else {
                let legal = rmx::available_actions(rm);
                if legal.is_empty() {
                    break;
                }
                legal[rng.gen_range(0..legal.len())]
            }
        } else {
            let acts = rmx::available_actions(rm);
            if acts.is_empty() {
                break;
            }
            match brain.choose(rm, &acts) {
                Some(i) => acts[i],
                None => acts[rng.gen_range(0..acts.len())],
            }
        };
        let result = rmx::take_action(rm, action, false, false);
        if result != "advanced" && result != "run_over" {
            // Engine rejected a legal move — bail rather than spin.
            return Playout {
                won: false,
                died: false,
                final_hp: rm.run_state.current_hp,
                hp_at_deepest,
                floors_advanced: deepest - start_floor,
                steps,
                reached_act: rm.run_state.act_index,
                inconclusive: true,
            };
        }
        steps += 1;
    }
    let over = rm.is_over();
    let cleared_act = rm.run_state.act_index != start_act;
    let won_run = over && rm.run_state.player_won;
    let died = over && !rm.run_state.player_won;
    let inconclusive = steps >= max_steps && !over && !cleared_act;
    let final_floor = rmx::total_floor(rm);
    if final_floor > deepest {
        deepest = final_floor;
        hp_at_deepest = rm.run_state.current_hp;
    }
    Playout {
        won: cleared_act || won_run,
        died,
        final_hp: rm.run_state.current_hp,
        hp_at_deepest,
        floors_advanced: deepest - start_floor,
        steps,
        reached_act: rm.run_state.act_index,
        inconclusive,
    }
}

/// `rank_rows`: mark the single best row, attach each row's `delta` vs the
/// best OTHER row, sort best-first.
pub fn rank_rows(rows: &mut [Value]) {
    if rows.is_empty() {
        return;
    }
    let values: Vec<f64> = rows.iter().map(|r| f64_of(&r["value"])).collect();
    let best = values
        .iter()
        .enumerate()
        .fold(0, |b, (i, v)| if *v > values[b] { i } else { b });
    for (i, row) in rows.iter_mut().enumerate() {
        let best_other = values
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, v)| *v)
            .fold(None, |m: Option<f64>, v| Some(m.map_or(v, |m| m.max(v))))
            .unwrap_or(0.0);
        row["delta"] = json!(values[i] - best_other);
        row["is_best"] = json!(i == best);
    }
    rows.sort_by(|a, b| {
        let ka = (
            !a["is_best"].as_bool().unwrap_or(false),
            -f64_of(&a["value"]),
        );
        let kb = (
            !b["is_best"].as_bool().unwrap_or(false),
            -f64_of(&b["value"]),
        );
        ka.0.cmp(&kb.0)
            .then(ka.1.partial_cmp(&kb.1).unwrap_or(std::cmp::Ordering::Equal))
    });
}

/// One candidate's `rollouts` paired playouts → its row (without
/// `delta` / `is_best`, which [`rank_rows`] adds).
#[allow(clippy::too_many_arguments)]
fn evaluate_candidate(
    rm: &RunManager,
    action: RunAction,
    cfg: RolloutConfig,
    max_hp: i32,
    progress: &mut Option<Progress<'_>>,
    done: &mut u32,
    total: u32,
    tick_fields: &Value,
) -> Value {
    let mut wins = 0_u32;
    let mut inconclusive = 0_u32;
    let mut hp_total = 0_i64;
    let mut floors_total = 0_i64;
    let mut hp_deep_total = 0_i64;
    let mut tick = |done: &mut u32| {
        *done += 1;
        if let Some(p) = progress.as_mut() {
            let mut frame = json!({"type": "tick", "done": *done, "total": total});
            if let (Some(f), Some(extra)) = (frame.as_object_mut(), tick_fields.as_object()) {
                for (k, v) in extra {
                    f.insert(k.clone(), v.clone());
                }
            }
            p(&frame);
        }
    };
    for r in 0..cfg.rollouts {
        let mut sim = rm.clone();
        let res = rmx::take_action(&mut sim, action, false, false);
        if res != "advanced" && res != "run_over" {
            inconclusive += 1;
            tick(done);
            continue;
        }
        // Paired across candidates: the same run seed + policy seed per
        // rollout `r`, so the only difference is the option taken.
        let seed = cfg.seed.wrapping_add(i64::from(r));
        #[allow(clippy::cast_possible_truncation)]
        let run_seed = seed as i32;
        // In the run's own build (GH #801): `RunRngSet::new` is v0.107.1's.
        sim.run_state.rng = RunRngSet::new_for(sim.game_version, u64::from(run_seed as u32));
        #[allow(clippy::cast_sign_loss)]
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(seed as u64);
        #[allow(clippy::cast_sign_loss)]
        let mcts_seed = cfg.seed.wrapping_add(i64::from(r) * 1009) as u64;
        let out = playout_rest_of_act(&mut sim, cfg.sims, cfg.max_steps, mcts_seed, &mut rng);
        if out.inconclusive {
            inconclusive += 1;
        }
        if out.won {
            wins += 1;
        }
        hp_total += i64::from(out.final_hp);
        floors_total += i64::from(out.floors_advanced);
        hp_deep_total += i64::from(out.hp_at_deepest);
        tick(done);
    }
    let n = f64::from(cfg.rollouts);
    let win_rate = f64::from(wins) / n;
    #[allow(clippy::cast_precision_loss)]
    let avg_floors = floors_total as f64 / n;
    #[allow(clippy::cast_precision_loss)]
    let avg_hp_deep = hp_deep_total as f64 / n;
    #[allow(clippy::cast_precision_loss)]
    let avg_final_hp = hp_total as f64 / n;
    // Composite value: act-clear dominates, with depth reached and HP carried
    // to the wall as fine-grained tiebreakers.
    let value = win_rate + 0.01 * avg_floors + 0.02 * (avg_hp_deep / f64::from(max_hp));
    json!({
        "win_rate": win_rate,
        "wins": wins,
        "n": cfg.rollouts,
        "avg_final_hp": avg_final_hp,
        "avg_floors_advanced": avg_floors,
        "avg_hp_at_deepest": avg_hp_deep,
        "value": value,
        "inconclusive": inconclusive,
    })
}

/// `card_advisor.is_advisable`: a plain card-pick reward (no bundle, no
/// relic reward) with at least one card offered.
#[must_use]
pub fn card_advisable(rm: &RunManager) -> bool {
    if rmx::phase_str(rm) != "card_reward" || rmx::card_reward_bundles(rm).is_some() {
        return false;
    }
    let acts = rmx::available_actions(rm);
    if acts.iter().any(|a| matches!(a, RunAction::PickRelicReward)) {
        return false;
    }
    acts.iter()
        .any(|a| matches!(a, RunAction::PickCardReward { idx: Some(_) }))
}

/// `card_advisor.advise`: Monte-Carlo act win-rate per card-reward option —
/// `{candidates: [{idx, label, win_rate, wins, n, avg_final_hp,
/// avg_floors_advanced, avg_hp_at_deepest, value, delta, is_best,
/// inconclusive}], act, config}`, or `{error, candidates: [], config}`.
pub fn card_advise(rm: &RunManager, config: &Value, mut progress: Option<Progress<'_>>) -> Value {
    if !card_advisable(rm) {
        return json!({
            "error": "card advisor only runs on a plain card-reward decision",
            "candidates": [],
            "config": config_obj(config),
        });
    }
    let cfg = RolloutConfig::from_config(config, rm);
    let start_act = rm.run_state.act_index;
    let max_hp = rm.run_state.max_hp.max(1);
    let ids = rmx::card_reward_option_ids(rm);
    let acts = rmx::available_actions(rm);
    let mut cands: Vec<(Option<usize>, String)> = acts
        .iter()
        .filter_map(|a| match a {
            RunAction::PickCardReward { idx: Some(i) } => Some((
                Some(*i),
                ids.get(*i).map_or_else(
                    || format!("card {i}"),
                    |id| id.strip_prefix("CARD.").unwrap_or(id).to_string(),
                ),
            )),
            _ => None,
        })
        .collect();
    if acts
        .iter()
        .any(|a| matches!(a, RunAction::PickCardReward { idx: None }))
    {
        cands.push((None, "skip".to_string()));
    }
    let total = u32::try_from(cands.len()).unwrap_or(u32::MAX) * cfg.rollouts;
    if let Some(p) = progress.as_mut() {
        p(&json!({
            "type": "start",
            "total": total,
            "rollouts": cfg.rollouts,
            "act": start_act,
            "candidates": cands.iter().map(|(idx, label)| json!({"idx": idx, "label": label})).collect::<Vec<_>>(),
        }));
    }
    let mut done = 0_u32;
    let mut rows: Vec<Value> = Vec::with_capacity(cands.len());
    for (idx, label) in &cands {
        let mut row = evaluate_candidate(
            rm,
            RunAction::PickCardReward { idx: *idx },
            cfg,
            max_hp,
            &mut progress,
            &mut done,
            total,
            &json!({"candidate_idx": idx}),
        );
        row["idx"] = json!(idx);
        row["label"] = json!(label);
        rows.push(row);
    }
    rank_rows(&mut rows);
    json!({"candidates": rows, "act": start_act, "config": cfg.to_value()})
}

/// `decision_advisor.is_advisable`: a shop / event / treasure / boss-relic
/// pick, a run-level choose screen, or a single-pick in-combat choice — with
/// at least two evaluable options.
#[must_use]
pub fn decision_advisable(rm: &RunManager) -> bool {
    let phase_ok = match rmx::phase_str(rm) {
        "combat" => match pending_choice(rm) {
            Some(pc) => pc["max_select"].as_i64().unwrap_or(1) <= 1,
            None => false,
        },
        "card_reward" => choose_screen(rm),
        "shop" | "event" | "treasure" | "boss_relic" => true,
        _ => false,
    };
    phase_ok && candidates(rm).len() >= 2
}

/// `_decision_of`: the frontend `decision` string for the current phase.
fn decision_of(rm: &RunManager) -> &'static str {
    let phase = rmx::phase_str(rm);
    if phase == "combat" || choose_screen(rm) {
        return "card_select";
    }
    match phase {
        "shop" => "shop",
        "event" | "treasure" | "boss_relic" => "event_choice",
        other => other,
    }
}

/// `decision_advisor.advise`: Monte-Carlo act win-rate per option of the
/// current decision — `{candidates: [{key, label, …}], decision, act,
/// config}`, or `{error, candidates: [], config}`.
pub fn decision_advise(
    rm: &RunManager,
    config: &Value,
    mut progress: Option<Progress<'_>>,
) -> Value {
    if !decision_advisable(rm) {
        return json!({
            "error": "the decision advisor only runs on a shop, event, or card-select decision with at least two options",
            "candidates": [],
            "config": config_obj(config),
        });
    }
    let cfg = RolloutConfig::from_config(config, rm);
    let start_act = rm.run_state.act_index;
    let decision = decision_of(rm);
    let max_hp = rm.run_state.max_hp.max(1);
    let cands = candidates(rm);
    let total = u32::try_from(cands.len()).unwrap_or(u32::MAX) * cfg.rollouts;
    if let Some(p) = progress.as_mut() {
        p(&json!({
            "type": "start",
            "total": total,
            "rollouts": cfg.rollouts,
            "act": start_act,
            "decision": decision,
            "candidates": cands.iter().map(|c| json!({"key": c.key, "label": c.label})).collect::<Vec<_>>(),
        }));
    }
    let mut done = 0_u32;
    let mut rows: Vec<Value> = Vec::with_capacity(cands.len());
    for c in &cands {
        let mut row = evaluate_candidate(
            rm,
            c.action,
            cfg,
            max_hp,
            &mut progress,
            &mut done,
            total,
            &json!({"candidate_key": c.key}),
        );
        row["key"] = json!(c.key);
        row["label"] = json!(c.label);
        rows.push(row);
    }
    rank_rows(&mut rows);
    json!({
        "candidates": rows,
        "decision": decision,
        "act": start_act,
        "config": cfg.to_value(),
    })
}

/// The coach's rollout check of one graded run-level choice: your pick vs
/// the policy's top pick, paired rollouts to the end of the act (the same
/// rows as [`decision_advise`], keyed `"you"` / `"ai"`). Unlike the
/// advisors it takes any decision, the map and rest site included, since
/// both actions come from the snapshotted position's legal moves.
pub fn choice_check(
    rm: &RunManager,
    you: (RunAction, &str),
    ai: (RunAction, &str),
    config: &Value,
    mut progress: Option<Progress<'_>>,
) -> Value {
    let cfg = RolloutConfig::from_config(config, rm);
    let start_act = rm.run_state.act_index;
    let max_hp = rm.run_state.max_hp.max(1);
    let cands = [("you", you), ("ai", ai)];
    let total = 2 * cfg.rollouts;
    if let Some(p) = progress.as_mut() {
        p(&json!({
            "type": "start",
            "total": total,
            "rollouts": cfg.rollouts,
            "act": start_act,
            "candidates": cands
                .iter()
                .map(|(key, (_, label))| json!({"key": key, "label": label}))
                .collect::<Vec<_>>(),
        }));
    }
    let mut done = 0_u32;
    let mut rows: Vec<Value> = Vec::with_capacity(2);
    for (key, (action, label)) in cands {
        let mut row = evaluate_candidate(
            rm,
            action,
            cfg,
            max_hp,
            &mut progress,
            &mut done,
            total,
            &json!({"candidate_key": key}),
        );
        row["key"] = json!(key);
        row["label"] = json!(label);
        rows.push(row);
    }
    rank_rows(&mut rows);
    json!({"candidates": rows, "act": start_act, "config": cfg.to_value()})
}

/// `advise_card_reward`: the card advisor over the live run; `None` when the
/// active node is not a `card_reward`.
pub fn advise_card_reward(
    sess: &Session,
    config: &Value,
    progress: Option<Progress<'_>>,
) -> Result<Option<Value>, SessionError> {
    let node = sess.active_node()?;
    if node.decision != "card_reward" {
        return Ok(None);
    }
    let rm = sess
        .game()
        .and_then(|g| g.native_run())
        .ok_or_else(|| SessionError::new(400, "no live run"))?;
    Ok(Some(card_advise(rm, config, progress)))
}

/// `advise_decision`: the decision advisor over the live run; `None` when the
/// active node is not a `card_select` / `shop` / `event_choice`.
pub fn advise_decision(
    sess: &Session,
    config: &Value,
    progress: Option<Progress<'_>>,
) -> Result<Option<Value>, SessionError> {
    let node = sess.active_node()?;
    if !matches!(
        node.decision.as_str(),
        "card_select" | "shop" | "event_choice"
    ) {
        return Ok(None);
    }
    let rm = sess
        .game()
        .and_then(|g| g.native_run())
        .ok_or_else(|| SessionError::new(400, "no live run"))?;
    Ok(Some(decision_advise(rm, config, progress)))
}

/// `wpa_review` for a session with no recorded node values (PPO values
/// arrive with M3): the `available: false` payload, verbatim.
#[must_use]
pub fn wpa_review_unavailable() -> Value {
    json!({
        "available": false,
        "reason": "no run-level value data — value recording needs the rust backend and a loadable PPO checkpoint",
        "points": [],
        "turning_points": [],
        "summary": {},
        "caveats": [
            "The outlook score is the AI's shaped-return estimate of the run — a relative signal, not a win percentage.",
            "A swing mixes your choice with the luck revealed after it (what the shop stocked, what the reward offered, how the fight rolled) — use “verify” to check whether the choice itself mattered.",
            "Judgments are relative to the AI's own play style; a line that is weak for the AI may still be fine for a human plan.",
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intent_text_shared_cases() {
        let cases: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../sts2-pycli/tests/fixtures/coach/intent_text_cases.json"
        )))
        .unwrap();
        for case in cases["cases"].as_array().unwrap() {
            for key in ["type", "intent_type"] {
                let intents: Vec<Value> = case["intents"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|i| {
                        let mut i = i.clone();
                        let kind = i["kind"].take();
                        i.as_object_mut().unwrap().remove("kind");
                        i[key] = kind;
                        i
                    })
                    .collect();
                assert_eq!(
                    json!(intent_text(&intents)),
                    case["expect"],
                    "case {} ({key})",
                    case["name"]
                );
            }
        }
    }

    #[test]
    fn grade_turn_shared_cases() {
        let cases: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../sts2-pycli/tests/fixtures/coach/grade_turn_cases.json"
        )))
        .unwrap();
        for case in cases["cases"].as_array().unwrap() {
            let regrets: Vec<Option<Value>> = case["regrets"]
                .as_array()
                .unwrap()
                .iter()
                .map(|r| (!r.is_null()).then(|| r.clone()))
                .collect();
            let got = grade_turn(&regrets).unwrap_or(Value::Null);
            assert_eq!(got, case["expect"], "case {}", case["name"]);
        }
    }

    #[test]
    fn labels_resolve_hand_potions_and_sentinels() {
        let state = json!({
            "hand": [{"index": 0, "name": "Strike"}, {"index": 3, "name": "Bash"}],
            "player": {"potions": [{"name": "Fire Potion"}, "Block Potion"]},
        });
        assert_eq!(action_label(END_TURN, &state), "end turn");
        assert_eq!(action_label(3, &state), "Bash");
        assert_eq!(action_label(7, &state), "action 7");
        assert_eq!(action_label(POTION_BASE, &state), "drink Fire Potion");
        assert_eq!(action_label(POTION_BASE - 1, &state), "drink Block Potion");
        assert_eq!(action_label(POTION_BASE - 2, &state), "drink potion slot 2");
    }

    #[test]
    fn stable_keys_mirror_the_recorder() {
        assert_eq!(
            action_stable_key(&json!({"action": "play", "card_index": 2, "target_index": 1})),
            (Some(2), Some(1))
        );
        assert_eq!(
            action_stable_key(&json!({"action": "end_turn"})),
            (Some(END_TURN), None)
        );
        assert_eq!(
            action_stable_key(&json!({"action": "drink_potion", "slot": 1})),
            (Some(POTION_BASE - 1), None)
        );
        assert_eq!(action_stable_key(&json!({})), (None, None));
        assert_eq!(
            action_stable_key(&json!({"action": "choose"})),
            (None, None)
        );
    }

    #[test]
    fn native_action_reads_the_workbench_verbs() {
        assert_eq!(
            native_action(
                Some("play_card"),
                &json!({"card_index": 1, "target_index": 0})
            ),
            json!({"action": "play", "card_index": 1, "target_index": 0})
        );
        assert_eq!(
            native_action(Some("use_potion"), &json!({"potion_index": 2})),
            json!({"action": "drink_potion", "slot": 2, "target_index": null})
        );
        assert_eq!(
            native_action(Some("__agent_native__"), &json!({"action": "end_turn"})),
            json!({"action": "end_turn"})
        );
        assert_eq!(
            native_action(Some("select_cards"), &json!({"indices": "1"})),
            json!({"indices": "1"})
        );
    }

    fn row(stable_idx: i64, target: Option<i64>, wp: f64, best: bool, label: &str) -> Value {
        json!({
            "stable_idx": stable_idx, "target_index": target, "win_prob": wp,
            "is_best": best, "visits": 100, "action": label,
        })
    }

    fn step(chosen: Value, enemies: Value) -> DecisionStep {
        DecisionStep {
            idx: 5,
            node_id: Some(9),
            state: json!({"enemies": enemies}),
            chosen,
            turn: json!(3),
            floor: json!(2),
            act: json!(1),
            best_line: Vec::new(),
        }
    }

    #[test]
    fn assess_decision_flags_a_blunder_with_the_python_message() {
        let scores = vec![
            row(1, Some(0), 0.71, true, "Defend"),
            row(0, Some(0), 0.58, false, "Strike"),
        ];
        let d = assess_decision(
            &step(
                json!({"action": "play", "card_index": 0, "target_index": 0}),
                json!([]),
            ),
            &scores,
            DEFAULT_BLUNDER_DROP_PCT,
        )
        .unwrap();
        assert_eq!(d["is_blunder"], json!(true));
        assert_eq!(d["drop_pct"], json!(13));
        assert_eq!(d["player_rank"], json!(1));
        assert_eq!(d["n_actions"], json!(2));
        assert_eq!(
            d["message"],
            json!("Turn 3: Strike instead of Defend dropped P(win) 71%→58% (−13 pts)")
        );
        // Multi-enemy fights name the target.
        let d = assess_decision(
            &step(
                json!({"action": "play", "card_index": 0, "target_index": 0}),
                json!([{"index": 0, "name": "Cultist"}, {"index": 1, "name": "Jaw Worm"}]),
            ),
            &scores,
            DEFAULT_BLUNDER_DROP_PCT,
        )
        .unwrap();
        assert!(d["played_label"]
            .as_str()
            .unwrap()
            .contains("Strike → Cultist"));
    }

    #[test]
    fn targeted_label_and_turn_context_number_duplicated_enemies() {
        let enemies = vec![
            json!({"index": 0, "name": "Two-Tailed Rat"}),
            json!({"index": 1, "name": "Louse"}),
            json!({"index": 3, "name": "Two-Tailed Rat"}),
        ];
        let r = |t| row(0, Some(t), 0.5, false, "Strike");
        assert_eq!(
            targeted_label(&r(0), &enemies),
            "Strike → Two-Tailed Rat #1"
        );
        assert_eq!(targeted_label(&r(1), &enemies), "Strike → Louse");
        assert_eq!(
            targeted_label(&r(3), &enemies),
            "Strike → Two-Tailed Rat #2"
        );
        assert_eq!(targeted_label(&r(2), &enemies), "Strike");
        let ctx = turn_context(&json!({"enemies": enemies}));
        let names: Vec<&str> = ctx["enemies"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e["name"].as_str().unwrap())
            .collect();
        assert_eq!(names, ["Two-Tailed Rat #1", "Louse", "Two-Tailed Rat #2"]);
    }

    #[test]
    fn assess_decision_matched_line_and_unresolvable() {
        let scores = vec![
            row(1, None, 0.6, true, "Defend"),
            row(0, None, 0.5, false, "Strike"),
        ];
        let d = assess_decision(
            &step(json!({"action": "play", "card_index": 1}), json!([])),
            &scores,
            8.0,
        )
        .unwrap();
        assert_eq!(d["matched_best"], json!(true));
        assert_eq!(d["regret"], json!(0.0));
        assert_eq!(
            d["message"],
            json!("Turn 3: Defend matched the MCTS line (P(win) 60%)")
        );
        assert!(assess_decision(
            &step(json!({"action": "play", "card_index": 4}), json!([])),
            &scores,
            8.0
        )
        .is_none());
        assert!(assess_decision(
            &step(json!({"action": "end_turn"}), json!([])),
            &scores[..1],
            8.0
        )
        .is_none());
    }

    #[test]
    fn summarize_ranks_blunders_by_regret_then_turn() {
        let d = |idx: i64, turn: i64, regret: f64, blunder: bool| json!({"idx": idx, "turn": turn, "regret": regret, "severity": regret * 100.0, "drop_pct": pct(regret), "is_blunder": blunder});
        let out = summarize(
            vec![
                d(1, 1, 0.10, true),
                d(2, 2, 0.30, true),
                d(3, 3, 0.30, true),
                d(4, 4, 0.02, false),
            ],
            2,
        );
        assert_eq!(out["summary"]["total_decisions"], json!(4));
        assert_eq!(out["summary"]["blunder_count"], json!(3));
        assert_eq!(out["summary"]["cumulative_drop_pct"], json!(72));
        assert_eq!(out["summary"]["worst_drop_pct"], json!(30));
        let b = out["blunders"].as_array().unwrap();
        assert_eq!(b.len(), 2);
        assert_eq!(b[0]["idx"], json!(2));
        assert_eq!(b[1]["idx"], json!(3));
    }

    #[test]
    fn rank_rows_marks_best_and_deltas() {
        let mut rows = vec![
            json!({"key": "a", "value": 0.2}),
            json!({"key": "b", "value": 0.5}),
            json!({"key": "c", "value": 0.4}),
        ];
        rank_rows(&mut rows);
        assert_eq!(rows[0]["key"], json!("b"));
        assert_eq!(rows[0]["is_best"], json!(true));
        assert!((f64_of(&rows[0]["delta"]) - 0.1).abs() < 1e-9);
        assert_eq!(rows[1]["key"], json!("c"));
        assert!((f64_of(&rows[1]["delta"]) + 0.1).abs() < 1e-9);
        assert!((f64_of(&rows[2]["delta"]) + 0.3).abs() < 1e-9);
        let mut one = vec![json!({"value": 0.3})];
        rank_rows(&mut one);
        assert_eq!(one[0]["delta"], json!(0.3));
    }

    #[test]
    fn pct_rounds_ties_to_even_like_python() {
        assert_eq!(pct(0.125), 12);
        assert_eq!(pct(0.135), 14);
        assert_eq!(pct(0.71), 71);
    }

    // ── HP-aware verdicts (twins of test_regret_review.py's `_hp_step` tests) ──

    fn score_row(stable_idx: i64, label: &str, wp: f64, best: bool, visits: i64) -> Value {
        json!({
            "stable_idx": stable_idx, "action": label, "win_prob": wp,
            "is_best": best, "target_index": null, "visits": visits,
        })
    }

    /// `_hp_step`: Defend (best, 900 visits, 70 HP over 400 playouts) vs the
    /// played end turn (120 visits, 58 HP over 120), each overridable.
    fn hp_assess(best_kw: Value, played_kw: Value, best_line: &[&str]) -> Value {
        let mut best = score_row(1, "Defend", 1.0, true, 900);
        best["mean_final_hp"] = json!(70.0);
        best["hp_n"] = json!(400);
        let mut played = score_row(END_TURN, "end turn", 1.0, false, 120);
        played["mean_final_hp"] = json!(58.0);
        played["hp_n"] = json!(120);
        for (row, kw) in [(&mut best, best_kw), (&mut played, played_kw)] {
            for (k, v) in kw.as_object().unwrap() {
                row[k] = v.clone();
            }
        }
        let mut s = step(json!({"action": "end_turn"}), json!([]));
        s.best_line = best_line.iter().map(|x| (*x).to_string()).collect();
        assess_decision(&s, &[played, best], DEFAULT_BLUNDER_DROP_PCT).unwrap()
    }

    #[test]
    fn hp_gap_within_noise_reads_zero() {
        // GH #888: 2 HP apart, SE of the difference 4 HP -- noise.
        let noisy = json!({"mean_final_hp": 60.0, "hp_sd": 20.0, "hp_n": 50});
        let mut played = noisy.clone();
        played["mean_final_hp"] = json!(58.0);
        let d = hp_assess(noisy, played, &[]);
        assert_eq!(d["hp_drop"], json!(0.0));
        assert_eq!(d["verdict"], json!("tie"));
        assert!(!d["message"].as_str().unwrap().contains("HP"));
    }

    #[test]
    fn hp_gap_clearing_noise_counts() {
        let d = hp_assess(json!({"hp_sd": 5.0}), json!({"hp_sd": 5.0}), &[]);
        assert_eq!(d["hp_drop"], json!(12.0));
        assert_eq!(d["verdict"], json!("blunder"));
    }

    #[test]
    fn noise_gaps_do_not_sum_into_a_turn_grade() {
        let noisy = json!({"mean_final_hp": 60.0, "hp_sd": 20.0, "hp_n": 50});
        let mut played = noisy.clone();
        played["mean_final_hp"] = json!(58.5);
        let regrets: Vec<Option<Value>> = (0..3)
            .map(|_| Some(hp_assess(noisy.clone(), played.clone(), &[])))
            .collect();
        let g = grade_turn(&regrets).unwrap();
        assert!(g["hp_drop"].is_null(), "{g}");
        assert_eq!(g["verdict"], json!("best"));
    }

    #[test]
    fn saturated_fight_hp_gap_is_a_blunder() {
        let d = hp_assess(json!({}), json!({}), &[]);
        assert_eq!(d["verdict"], json!("blunder"));
        assert_eq!(d["is_blunder"], json!(true));
        assert_eq!(d["hp_drop"], json!(12.0));
        assert_eq!(d["played_mean_hp"], json!(58.0));
        assert_eq!(d["best_mean_hp"], json!(70.0));
        assert_eq!(d["severity"], json!(12.0));
        assert_eq!(
            d["message"],
            json!("Turn 3: end turn instead of Defend dropped P(win) 100%→100% (−0 pts), ≈12 HP")
        );
    }

    #[test]
    fn small_hp_gap_is_a_slip() {
        let d = hp_assess(json!({}), json!({"mean_final_hp": 65.0}), &[]);
        assert_eq!(d["verdict"], json!("slip"));
        assert_eq!(d["is_blunder"], json!(false));
        assert_eq!(d["hp_drop"], json!(5.0));
        assert_eq!(
            d["message"],
            json!("Turn 3: end turn instead of Defend dropped P(win) 100%→100% (−0 pts), ≈5 HP")
        );
    }

    #[test]
    fn within_noise_on_all_signals_is_a_tie() {
        let d = hp_assess(
            json!({"win_prob": 1.0}),
            json!({"win_prob": 0.98, "mean_final_hp": 68.0}),
            &[],
        );
        assert_eq!(d["verdict"], json!("tie"));
        assert_eq!(d["is_blunder"], json!(false));
        // The suffix rides on every non-matched verdict, ties included.
        assert_eq!(
            d["message"],
            json!("Turn 3: end turn instead of Defend dropped P(win) 100%→98% (−2 pts), ≈2 HP")
        );
    }

    #[test]
    fn low_visit_played_move_is_uncertain() {
        let d = hp_assess(json!({}), json!({"visits": 12}), &[]);
        assert_eq!(d["verdict"], json!("uncertain"));
        assert_eq!(d["is_blunder"], json!(false));
        assert_eq!(
            d["message"],
            json!("Turn 3: end turn instead of Defend dropped P(win) 100%→100% (−0 pts), ≈12 HP")
        );
        // A row without `visits` counts as 0 — uncertain, not a crash.
        let d = hp_assess(json!({}), json!({"visits": null}), &[]);
        assert_eq!(d["verdict"], json!("uncertain"));
        // A float count truncates like Python's `int()`: 50.7 → 50, gradable.
        let d = hp_assess(json!({}), json!({"visits": 50.7}), &[]);
        assert_eq!(d["verdict"], json!("blunder"));
        let d = hp_assess(json!({}), json!({"visits": 49.9}), &[]);
        assert_eq!(d["verdict"], json!("uncertain"));
    }

    #[test]
    fn missing_hp_falls_back_to_win_prob() {
        let d = hp_assess(
            json!({"win_prob": 0.9, "mean_final_hp": null, "hp_n": 0}),
            json!({"win_prob": 0.8, "mean_final_hp": null, "hp_n": 0}),
            &[],
        );
        assert_eq!(d["hp_drop"], Value::Null);
        assert_eq!(d["played_mean_hp"], Value::Null);
        assert_eq!(d["verdict"], json!("blunder"));
        assert_eq!(
            d["message"],
            json!("Turn 3: end turn instead of Defend dropped P(win) 90%→80% (−10 pts)")
        );
    }

    #[test]
    fn hp_below_min_samples_ignored() {
        let d = hp_assess(json!({}), json!({"hp_n": 5}), &[]);
        assert_eq!(d["hp_drop"], Value::Null);
        assert_eq!(d["verdict"], json!("tie"));
        // A row without `hp_n` counts as 0 samples.
        let d = hp_assess(json!({}), json!({"hp_n": null}), &[]);
        assert_eq!(d["hp_drop"], Value::Null);
    }

    #[test]
    fn player_beats_ai_line_is_a_tie() {
        let d = hp_assess(json!({}), json!({"mean_final_hp": 75.0}), &[]);
        assert_eq!(d["hp_drop"], json!(0.0));
        assert_eq!(d["verdict"], json!("tie"));
        assert_eq!(
            d["message"],
            json!("Turn 3: end turn instead of Defend dropped P(win) 100%→100% (−0 pts)")
        );
    }

    #[test]
    fn best_line_is_echoed() {
        let d = hp_assess(json!({}), json!({}), &["Bash", "Defend", "end turn"]);
        assert_eq!(d["best_line"], json!(["Bash", "Defend", "end turn"]));
        assert_eq!(hp_assess(json!({}), json!({}), &[])["best_line"], json!([]));
    }

    #[test]
    fn severity_is_computed_in_float_points_like_python() {
        // 1.0 − 0.92 = 0.07999…; ×100 = 7.999… < 8 → slip, though it renders −8.
        let d = hp_assess(
            json!({"win_prob": 1.0, "mean_final_hp": null}),
            json!({"win_prob": 0.92, "mean_final_hp": null}),
            &[],
        );
        assert_eq!(d["verdict"], json!("slip"));
        assert_eq!(d["drop_pct"], json!(8));
        // A half-HP gap rounds half-up in the message (2.5 → 3), still a tie.
        let d = hp_assess(json!({}), json!({"mean_final_hp": 67.5}), &[]);
        assert_eq!(d["verdict"], json!("tie"));
        assert_eq!(
            d["message"],
            json!("Turn 3: end turn instead of Defend dropped P(win) 100%→100% (−0 pts), ≈3 HP")
        );
    }

    #[test]
    fn summarize_sorts_by_severity_and_sums_hp_in_order() {
        let ds = vec![
            hp_assess(json!({}), json!({}), &[]), // blunder, hp 12, drop 0
            hp_assess(json!({}), json!({"mean_final_hp": 65.0}), &[]), // slip, hp 5
            hp_assess(
                json!({"win_prob": 0.9, "mean_final_hp": null}),
                json!({"win_prob": 0.8, "mean_final_hp": null}),
                &[],
            ), // blunder, severity 9.99…, drop 10
            hp_assess(json!({}), json!({"mean_final_hp": 67.5}), &[]), // tie, hp 2.5
        ];
        let out = summarize(ds, 2);
        let s = &out["summary"];
        assert_eq!(s["blunder_count"], json!(2));
        assert_eq!(s["cumulative_drop_pct"], json!(10));
        // 12 + 5 + 2.5 = 19.5 → 20 (half-up).
        assert_eq!(s["cumulative_hp_drop"], json!(20));
        // The top blunder is the HP one (severity 12 > 9.99), whose drop is 0.
        assert_eq!(s["worst_drop_pct"], json!(0));
        assert_eq!(out["blunders"][0]["severity"], json!(12.0));
        assert_eq!(out["blunders"][1]["drop_pct"], json!(10));
    }

    #[test]
    fn round_half_up_matches_python_floor_plus_half() {
        assert_eq!(round_half_up(2.5), 3);
        assert_eq!(round_half_up(0.49), 0);
        assert_eq!(round_half_up(12.0), 12);
    }

    // ── label_line / top_up_played (native combat fixtures) ─────────────

    /// An Ironclad turn-1 fight vs a Mawler whose hand is exactly `hand` and
    /// whose draw pile is exactly `draw` (top first), by card id.
    fn ironclad_fight(hand: &[&str], draw: &[&str]) -> CombatState {
        let deck: Vec<String> = hand.iter().chain(draw).map(|s| (*s).to_string()).collect();
        let mut cs = sts2_wire::scenario::build_scenario_state(
            1,
            "Ironclad",
            &deck,
            80,
            80,
            &[],
            &[],
            "MAWLER_NORMAL",
            0,
            None,
            None,
        )
        .expect("scenario builds");
        let mut pool: Vec<sts2_core::CardUid> = cs
            .player
            .hand
            .iter()
            .chain(cs.player.draw.iter())
            .copied()
            .collect();
        let mut take = |cs: &CombatState, id: &str| {
            let i = pool
                .iter()
                .position(|uid| {
                    sts2_core::card_by_uid(&cs.cards, *uid)
                        .is_some_and(|c| c.card_id.to_python_name() == id)
                })
                .unwrap_or_else(|| panic!("{id} dealt"));
            pool.remove(i)
        };
        let new_hand: Vec<_> = hand.iter().map(|id| take(&cs, id)).collect();
        let new_draw: Vec<_> = draw.iter().map(|id| take(&cs, id)).collect();
        cs.player.hand = new_hand.into_iter().collect();
        cs.player.draw = new_draw.into_iter().collect();
        cs
    }

    fn ironclad_with_hand(order: &[&str]) -> CombatState {
        ironclad_fight(order, &[])
    }

    fn test_loc() -> crate::localization::Loc {
        let mut loc = crate::localization::Loc::new();
        loc.insert("BASH.title", "Bash");
        loc.insert("DEFEND_IRONCLAD.title", "Defend");
        loc.insert("STRIKE_IRONCLAD.title", "Strike");
        loc.insert("POMMEL_STRIKE.title", "Pommel Strike");
        loc
    }

    /// The legal action at `cs` playing hand slot `slot` (any target).
    fn play_slot(cs: &CombatState, slot: u8) -> Value {
        sts2_core::legal_actions(cs)
            .into_iter()
            .find(|a| matches!(a, sts2_core::Action::PlayCard { hand_index, .. } if *hand_index == slot))
            .map(|a| sts2_wire::emit::combat_action_to_value(&a))
            .expect("slot playable")
    }

    #[test]
    fn label_line_stops_at_the_first_draw() {
        let cs = ironclad_fight(
            &["POMMEL_STRIKE", "DEFEND_IRONCLAD", "STRIKE_IRONCLAD"],
            &["BASH", "DEFEND_IRONCLAD", "STRIKE_IRONCLAD"],
        );
        let draw = play_slot(&cs, 0);
        // `play 0` after the draw would name whatever the REAL draw put
        // there — the line ends at the draw instead.
        let pv = vec![
            draw,
            json!({"action": "play", "card_index": 0, "target_index": null}),
            json!({"action": "end_turn"}),
        ];
        assert_eq!(label_line(&cs, &pv, &test_loc()), vec!["Pommel Strike"]);
    }

    /// The AI line's end board: the position its End turn is played in;
    /// none when it stops at a draw first (GH #888).
    #[test]
    fn walk_line_ends_where_the_end_turn_is_played() {
        let cs = ironclad_with_hand(&["BASH", "DEFEND_IRONCLAD", "STRIKE_IRONCLAD"]);
        let defend = play_slot(&cs, 1);
        let mut after = cs.clone();
        let _ = sts2_core::apply(&mut after, legal_action_for(&cs, &defend).unwrap());
        let pv = vec![defend, json!({"action": "end_turn"})];
        let (line, end) = walk_line(&cs, &pv, &test_loc());
        assert_eq!(line, vec!["Defend", "end turn"]);
        assert_eq!(rmx::wire(&end.expect("end board")), rmx::wire(&after));
        let drawn = ironclad_fight(&["POMMEL_STRIKE"], &["DEFEND_IRONCLAD", "BASH"]);
        let pv = vec![play_slot(&drawn, 0), json!({"action": "end_turn"})];
        assert!(walk_line(&drawn, &pv, &test_loc()).1.is_none());
    }

    /// On a real board: three Strikes break the sleeping Lagavulin's 12
    /// block and wake it, a Defend does not — the reason line names it from
    /// the turn context.
    #[test]
    fn turn_why_names_the_woken_enemy() {
        let deck: Vec<String> = ["STRIKE_IRONCLAD"; 3]
            .iter()
            .chain(&["DEFEND_IRONCLAD"])
            .map(|s| (*s).to_string())
            .collect();
        let cs = sts2_wire::scenario::build_scenario_state(
            1,
            "Ironclad",
            &deck,
            80,
            80,
            &[],
            &[],
            "LAGAVULIN_MATRIARCH_BOSS",
            0,
            None,
            None,
        )
        .expect("scenario builds");
        let after = |ids: &[&str]| {
            let mut out = cs.clone();
            for id in ids {
                let slot = out
                    .player
                    .hand
                    .iter()
                    .position(|&u| {
                        sts2_core::card_by_uid(&out.cards, u)
                            .is_some_and(|c| c.card_id.to_python_name() == *id)
                    })
                    .expect("in hand");
                let action =
                    legal_action_for(&out, &play_slot(&out, u8::try_from(slot).unwrap())).unwrap();
                let _ = sts2_core::apply(&mut out, action);
            }
            out
        };
        let strike = after(&["STRIKE_IRONCLAD"; 3]);
        let defend = after(&["DEFEND_IRONCLAD"]);
        let named = named_combat_state(&cs, &test_loc());
        assert_eq!(
            turn_why(&cs, &strike, Some(&defend), &named).as_deref(),
            Some("Your line wakes Lagavulin Matriarch on turn 1")
        );
        assert_eq!(turn_why(&cs, &defend, Some(&defend), &named), None);
        assert_eq!(turn_why(&cs, &strike, None, &named), None);
    }

    fn lagavulin(asleep: bool, dead: bool, uid: i64) -> Value {
        let powers = if asleep {
            json!([{"name": "ASLEEP", "amount": 3}])
        } else {
            json!([])
        };
        json!({"kind": "LAGAVULIN_MATRIARCH", "uid": uid, "is_dead": dead, "powers": powers})
    }

    fn board(enemies: &[Value], phase: &str) -> Value {
        json!({"phase": phase, "encounter": enemies})
    }

    /// Twin of `test_realgame_coach.TestWhy` (GH #888).
    #[test]
    fn board_events_wake_kill_and_win() {
        let p = "PlayerPhase";
        let start = board(&[lagavulin(true, false, 7), lagavulin(false, false, 8)], p);
        let end = board(&[lagavulin(false, false, 7), lagavulin(false, true, 8)], p);
        assert_eq!(board_events(&start, &end), vec![("wakes", 0), ("kills", 1)]);
        assert_eq!(
            board_events(&start, &board(&[], "Victory")),
            vec![("wins", -1)]
        );
        assert!(board_events(&start, &start).is_empty());
        let mut slumber = lagavulin(false, false, 7);
        slumber["powers"] = json!([{"name": "SLUMBER"}]);
        assert_eq!(
            board_events(
                &board(&[slumber], p),
                &board(&[lagavulin(false, false, 7)], p)
            ),
            vec![("wakes", 0)]
        );
    }

    #[test]
    fn why_text_names_each_sides_extra_events() {
        let names = ["Lagavulin Matriarch".to_string(), "Cultist".to_string()];
        assert_eq!(
            why_text(&[("wakes", 0)], &[], &names, &json!(1)).as_deref(),
            Some("Your line wakes Lagavulin Matriarch on turn 1")
        );
        assert_eq!(
            why_text(
                &[("wakes", 0)],
                &[("wakes", 0), ("kills", 1)],
                &names,
                &json!(2)
            )
            .as_deref(),
            Some("The AI's line kills Cultist on turn 2")
        );
        assert_eq!(
            why_text(&[("wakes", 0)], &[("wins", -1)], &names, &json!(4)).as_deref(),
            Some("Your line wakes Lagavulin Matriarch; the AI's line wins the fight on turn 4")
        );
        assert_eq!(
            why_text(&[("kills", 1)], &[("kills", 1)], &names, &json!(1)),
            None
        );
    }

    fn end_turn_rows() -> Vec<Value> {
        vec![
            json!({"stable_idx": 1, "action": "Defend", "win_prob": 0.9, "q": 0.5,
                   "visits": 300, "is_best": true, "target_index": null,
                   "wins": 270, "losses": 30, "ties": 0, "mean_final_hp": 70.0, "hp_n": 250}),
            json!({"stable_idx": END_TURN, "action": "end turn", "win_prob": 0.0, "q": 0.0,
                   "visits": 3, "is_best": false, "target_index": null,
                   "wins": 0, "losses": 3, "ties": 0, "mean_final_hp": null, "hp_n": 0}),
        ]
    }

    /// The top-up prices the move from the PARENT with the draw re-dealt per
    /// simulation, so it cannot depend on the real (hidden) draw-pile order.
    #[test]
    fn top_up_ignores_the_hidden_draw_order() {
        let hand = ["BASH", "DEFEND_IRONCLAD", "STRIKE_IRONCLAD"];
        let draw = [
            "POMMEL_STRIKE",
            "DEFEND_IRONCLAD",
            "STRIKE_IRONCLAD",
            "BASH",
            "STRIKE_IRONCLAD",
        ];
        let a = ironclad_fight(&hand, &draw);
        // Same cards, same uids — only the hidden order differs.
        let mut b = a.clone();
        b.player.draw.reverse();
        assert_ne!(a.player.draw, b.player.draw);
        let params = params_for(&json!({"sims": 200, "seed": 1}), "mcts_recommend").unwrap();
        let end = json!({"action": "end_turn"});
        let (mut ra, mut rb) = (end_turn_rows(), end_turn_rows());
        top_up_played(&a, &mut ra, &end, &params);
        top_up_played(&b, &mut rb, &end, &params);
        assert!(
            ra[1]["visits"].as_i64().unwrap() >= PLAYED_MIN_VISITS,
            "{}",
            ra[1]
        );
        assert_eq!(ra, rb);
    }

    #[test]
    fn label_line_relabels_after_hand_shift() {
        let cs = ironclad_with_hand(&["BASH", "DEFEND_IRONCLAD", "STRIKE_IRONCLAD"]);
        let play0 = |cs: &CombatState| {
            sts2_core::legal_actions(cs)
                .into_iter()
                .find(|a| matches!(a, sts2_core::Action::PlayCard { hand_index: 0, .. }))
                .map(|a| sts2_wire::emit::combat_action_to_value(&a))
                .expect("card 0 playable")
        };
        let first = play0(&cs);
        let mut after = cs.clone();
        let a0 = sts2_core::legal_actions(&after)
            .into_iter()
            .find(|a| sts2_wire::emit::combat_action_to_value(a) == first)
            .unwrap();
        let _ = sts2_core::apply(&mut after, a0);
        let second = play0(&after);
        let pv = vec![
            first,
            second,
            json!({"action": "end_turn"}),
            // Never reached: the line stops after the first end turn.
            json!({"action": "end_turn"}),
        ];
        let loc = test_loc();
        assert_eq!(
            label_line(&cs, &pv, &loc),
            vec!["Bash", "Defend", "end turn"]
        );
        // An action that is not legal where it lands ends the line.
        assert_eq!(
            label_line(&cs, &[json!({"action": "play", "card_index": 9})], &loc),
            Vec::<String>::new()
        );
    }

    #[test]
    fn top_up_lifts_low_visit_played_row() {
        let cs = ironclad_with_hand(&["BASH", "DEFEND_IRONCLAD", "STRIKE_IRONCLAD"]);
        let params = params_for(&json!({"sims": 200, "seed": 1}), "mcts_recommend").unwrap();
        let mut scores = end_turn_rows();
        top_up_played(&cs, &mut scores, &json!({"action": "end_turn"}), &params);
        let row = &scores[1];
        assert!(
            row["visits"].as_i64().unwrap() >= PLAYED_MIN_VISITS,
            "{row}"
        );
        assert!(row["mean_final_hp"].is_number(), "{row}");
        assert_eq!(row["stable_idx"], json!(END_TURN));
        assert_eq!(row["action"], json!("end turn"));
        assert_eq!(row["is_best"], json!(false));
        // The best row is untouched.
        assert_eq!(scores[0]["visits"], json!(300));
        // A row already at the floor is left alone.
        let before = scores.clone();
        top_up_played(&cs, &mut scores, &json!({"action": "end_turn"}), &params);
        assert_eq!(scores, before);
    }
}
