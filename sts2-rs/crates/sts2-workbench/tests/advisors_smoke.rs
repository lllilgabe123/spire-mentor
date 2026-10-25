//! Advisor routes (M2) over a scripted run: every route answers with the
//! Python shape at the decisions it applies to and the documented status
//! elsewhere; the streaming twins emit `start` / `tick`* / `result`.
//!
//! Shape + no-panic coverage (the golden test is `golden_parity.rs`).

use serde_json::{json, Value};
use sts2_workbench::api::{Config, Workbench};

fn new_wb() -> Workbench {
    Workbench::new(Config {
        assets_base: "https://assets.test".to_string(),
        ..Config::default()
    })
}

fn call(wb: &mut Workbench, method: &str, route: &str, body: Value) -> Value {
    match wb.call(method, route, &body) {
        Ok(v) => v,
        Err(e) => panic!("{method} {route} -> {} {}", e.status, e.detail),
    }
}

fn status(wb: &mut Workbench, method: &str, route: &str, body: Value) -> Result<Value, u16> {
    wb.call(method, route, &body).map_err(|e| e.status)
}

fn stream(wb: &mut Workbench, route: &str, body: Value) -> Vec<Value> {
    let mut frames = Vec::new();
    wb.call_stream("POST", route, &body, &mut |f| frames.push(f.clone()));
    frames
}

/// The smoke policy: first playable card at the first enemy, else end turn;
/// prefer interesting rooms on the map; buy one card at the first shop.
fn policy(step: &Value, shop_done: &mut bool) -> Option<(String, Value)> {
    let state = &step["state"];
    let decision = step["decision"].as_str().unwrap_or("");
    let options = state["options"].as_array().cloned().unwrap_or_default();
    let first_enabled = |opts: &[Value]| {
        opts.iter()
            .find(|o| o["enabled"].as_bool().unwrap_or(true))
            .map(|o| (o["action"].as_str().unwrap().to_string(), o["args"].clone()))
    };
    match decision {
        "game_over" => None,
        "combat_play" => {
            let hand = state["hand"].as_array().cloned().unwrap_or_default();
            let enemy0 = state["enemies"][0]["index"].clone();
            for card in &hand {
                if card["can_play"].as_bool() == Some(true) {
                    let mut args = json!({"card_index": card["index"]});
                    if card["target_type"] == json!("AnyEnemy") {
                        args["target_index"] = enemy0.clone();
                    }
                    return Some(("play_card".to_string(), args));
                }
            }
            Some(("end_turn".to_string(), json!({})))
        }
        "map_select" => {
            let prefer = [
                "Shop", "Event", "Treasure", "Monster", "RestSite", "Elite", "Boss",
            ];
            for p in prefer {
                if let Some(o) = options
                    .iter()
                    .find(|o| o["label"].as_str().is_some_and(|l| l.starts_with(p)))
                {
                    return Some((o["action"].as_str().unwrap().to_string(), o["args"].clone()));
                }
            }
            first_enabled(&options)
        }
        "shop" => {
            if !*shop_done {
                *shop_done = true;
                if let Some(o) = options.iter().find(|o| o["action"] == json!("buy_card")) {
                    return Some(("buy_card".to_string(), o["args"].clone()));
                }
            }
            Some(("leave_room".to_string(), json!({})))
        }
        "card_select" => {
            let min = state["min_select"].as_u64().unwrap_or(1).max(1);
            let indices: Vec<String> = (0..min).map(|i| i.to_string()).collect();
            Some((
                "select_cards".to_string(),
                json!({"indices": indices.join(",")}),
            ))
        }
        _ => first_enabled(&options).or_else(|| Some(("proceed".to_string(), json!({})))),
    }
}

const HINT: &str = r#"{"config": {"sims": 32, "seed": 1, "threads": 1, "c_puct": 2.5,
                        "hidden_info": "beliefdraw", "draw_widening_c": 2.0}}"#;
const ROLLOUT: &str = r#"{"config": {"rollouts": 1, "sims": 0, "max_steps": 12, "seed": 3}}"#;

fn hint_body() -> Value {
    serde_json::from_str(HINT).unwrap()
}
fn rollout_body() -> Value {
    serde_json::from_str(ROLLOUT).unwrap()
}

fn assert_stream(frames: &[Value], key_field: &str) {
    assert_eq!(frames[0]["type"], json!("start"), "{frames:?}");
    let total = frames[0]["total"].as_u64().unwrap();
    let ticks: Vec<&Value> = frames
        .iter()
        .filter(|f| f["type"] == json!("tick"))
        .collect();
    assert_eq!(ticks.len() as u64, total);
    for (i, t) in ticks.iter().enumerate() {
        assert_eq!(t["done"], json!(i as u64 + 1));
        assert_eq!(t["total"], json!(total));
        assert!(
            !t[key_field].is_null() || key_field == "candidate_idx",
            "{t}"
        );
    }
    let last = frames.last().unwrap();
    assert_eq!(last["type"], json!("result"), "{last}");
    assert!(last["candidates"].as_array().is_some_and(|c| !c.is_empty()));
    let best: Vec<&Value> = last["candidates"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["is_best"] == json!(true))
        .collect();
    assert_eq!(best.len(), 1, "exactly one best: {last}");
}

#[test]
fn advisor_routes_over_a_scripted_run() {
    let mut wb = new_wb();
    let caps = call(&mut wb, "GET", "/api/capabilities", Value::Null);
    assert_eq!(caps["features"]["advisors"], json!(true));
    assert_eq!(caps["features"]["run_mode"], json!(true));
    for route in [
        "/api/live/hint",
        "/api/live/regret-review",
        "/api/live/auto-action",
    ] {
        assert_eq!(
            status(&mut wb, "POST", route, json!({})),
            Err(400),
            "{route}: no run"
        );
    }

    let mut step = call(
        &mut wb,
        "POST",
        "/api/live/start",
        json!({"character": "Ironclad", "seed": "SMOKE1", "ascension": 0, "act1": "overgrowth"}),
    );
    // Neow: a run-level decision → the default `ppo` brain has no net here,
    // so the heuristic answers for it (`fallback`) with a rendered option's
    // verb; under the `heuristic` brain the answer carries no policy tag.
    assert_eq!(step["decision"], json!("event_choice"));
    let rec = call(&mut wb, "POST", "/api/live/auto-action", Value::Null);
    assert_eq!(rec["action"], json!("choose_option"), "{rec}");
    assert!(rec["args"]["option_index"].is_number());
    assert_eq!(rec["policy"], json!("ppo"));
    assert_eq!(rec["fallback"], json!(true));
    call(
        &mut wb,
        "POST",
        "/api/live/run-mode",
        json!({"off_combat_brain": "heuristic"}),
    );
    let rec = call(&mut wb, "POST", "/api/live/auto-action", Value::Null);
    assert!(
        rec.get("policy").is_none(),
        "heuristic answers carry no policy tag"
    );
    // Off combat the hint / tree / ranker / coach are 409s.
    assert_eq!(
        status(&mut wb, "POST", "/api/live/hint", hint_body()),
        Err(409)
    );
    assert_eq!(
        status(&mut wb, "POST", "/api/live/search_tree", hint_body()),
        Err(409)
    );
    assert_eq!(
        status(&mut wb, "POST", "/api/live/choice-ranker", hint_body()),
        Err(409)
    );
    assert_eq!(
        status(&mut wb, "POST", "/api/live/last-move-regret", json!({})),
        Err(409)
    );
    assert_eq!(
        status(&mut wb, "POST", "/api/live/card-advisor", rollout_body()),
        Err(409)
    );
    // The decision advisor applies at Neow (an event with ≥2 options).
    let adv = call(
        &mut wb,
        "POST",
        "/api/live/decision-advisor",
        rollout_body(),
    );
    assert_eq!(adv["decision"], json!("event_choice"), "{adv}");
    assert_eq!(adv["config"]["rollouts"], json!(1));
    let frames = stream(&mut wb, "/api/live/decision-advisor/stream", rollout_body());
    assert_stream(&frames, "candidate_key");
    assert_eq!(frames[0]["decision"], json!("event_choice"));
    assert!(frames[0]["candidates"][0]["key"]
        .as_str()
        .unwrap()
        .starts_with("option:"));
    // A stream on a non-applicable route is an error FRAME, not a status.
    let frames = stream(&mut wb, "/api/live/card-advisor/stream", rollout_body());
    assert_eq!(
        frames,
        vec![json!({"type": "error", "error": "not at a card reward"})]
    );
    // No net loaded: the WPA review has no values, the panel names the gap.
    let wpa = call(&mut wb, "POST", "/api/live/wpa-review", json!({}));
    assert_eq!(wpa["available"], json!(false));
    assert!(wpa["caveats"].as_array().is_some_and(|c| c.len() == 3));
    let panel = call(&mut wb, "POST", "/api/live/ppo-policy", Value::Null);
    assert_eq!(panel["ppo_available"], json!(false));
    assert_eq!(panel["error"], json!("no PPO net loaded for Ironclad"));

    let mut shop_done = false;
    let mut combats = 0;
    let mut regret_seen = false;
    let mut card_advised = false;
    let mut shop_advised = false;
    let mut prev_decision = String::new();
    for _ in 0..400 {
        let d = step["decision"].as_str().unwrap_or("").to_string();
        if d == "combat_play" {
            combats += 1;
            if combats == 1 {
                let hint = call(&mut wb, "POST", "/api/live/hint", hint_body());
                let scores = hint["scores"].as_array().unwrap();
                assert!(!scores.is_empty(), "{hint}");
                assert_eq!(
                    scores
                        .iter()
                        .filter(|r| r["is_best"] == json!(true))
                        .count(),
                    1
                );
                assert!(hint["pv"].is_null());
                assert_eq!(hint["config"]["sims"], json!(32));
                assert!(
                    scores[0]["action"]
                        .as_str()
                        .is_some_and(|l| !l.starts_with("action ")),
                    "labels resolve against the hand: {}",
                    scores[0]
                );
                // Deterministic for a config.
                let again = call(&mut wb, "POST", "/api/live/hint", hint_body());
                assert_eq!(again["scores"], hint["scores"]);
                let tree = call(&mut wb, "POST", "/api/live/search_tree", hint_body());
                assert_eq!(tree["root"], json!(0));
                assert!(tree["nodes"].as_array().is_some_and(|n| n.len() > 1));
                assert_eq!(tree["nodes"][0]["parent"], Value::Null);
                assert_eq!(tree["nodes"][0]["node_kind"], json!("decision"));
                assert!(tree["nodes"][1]["kind"].is_string());
                assert_eq!(tree["config"]["sims"], json!(32));
                // A bad mode is an error payload, not a 500.
                let bad = call(
                    &mut wb,
                    "POST",
                    "/api/live/hint",
                    json!({"config": {"hidden_info": "psychic"}}),
                );
                assert!(bad["error"].as_str().unwrap().contains("hidden_info"));
                assert_eq!(bad["scores"], json!([]));
                // Plain combat: the hint owns card plays.
                assert_eq!(
                    status(&mut wb, "POST", "/api/live/auto-action", Value::Null),
                    Err(409)
                );
            }
        }
        if d == "card_reward" && !card_advised {
            card_advised = true;
            let adv = call(&mut wb, "POST", "/api/live/card-advisor", rollout_body());
            assert!(
                adv["candidates"].as_array().is_some_and(|c| c.len() >= 2),
                "{adv}"
            );
            assert!(
                adv["candidates"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|c| c["idx"].is_null()),
                "skip row"
            );
            assert_eq!(adv["config"]["max_steps"], json!(12));
            let frames = stream(&mut wb, "/api/live/card-advisor/stream", rollout_body());
            assert_stream(&frames, "candidate_idx");
            assert!(frames[0]["candidates"][0]["label"].is_string());
        }
        if d == "shop" && !shop_advised {
            shop_advised = true;
            let adv = call(
                &mut wb,
                "POST",
                "/api/live/decision-advisor",
                rollout_body(),
            );
            assert_eq!(adv["decision"], json!("shop"), "{adv}");
            let keys: Vec<&str> = adv["candidates"]
                .as_array()
                .unwrap()
                .iter()
                .map(|c| c["key"].as_str().unwrap())
                .collect();
            assert!(keys.contains(&"leave"), "{keys:?}");
            assert!(keys.iter().any(|k| k.starts_with("card:")), "{keys:?}");
        }
        if prev_decision == "combat_play" && !regret_seen {
            // The move just made came from a combat decision → the coach has
            // something to say (or reports the move as unresolvable — never 500).
            match status(
                &mut wb,
                "POST",
                "/api/live/last-move-regret",
                json!({"config": {"sims": 24, "seed": 2}}),
            ) {
                Ok(r) => {
                    regret_seen = true;
                    assert!(
                        r["message"]
                            .as_str()
                            .is_some_and(|m| m.starts_with("Turn ")),
                        "{r}"
                    );
                    assert!(r["node_id"].is_null());
                    assert!(r["played_win_prob"].is_number());
                }
                Err(409) => {}
                Err(s) => panic!("last-move-regret -> {s}"),
            }
        }
        let Some((action, args)) = policy(&step, &mut shop_done) else {
            break;
        };
        let resp = call(
            &mut wb,
            "POST",
            "/api/live/action",
            json!({"action": action, "args": args}),
        );
        assert!(
            resp["error"].is_null(),
            "action {action} {args} rejected: {}",
            resp["error"]
        );
        prev_decision = d;
        step = resp["step"].clone();
        if combats >= 3 && card_advised && shop_advised && regret_seen {
            break;
        }
    }
    assert!(combats >= 1, "the script never reached combat");
    assert!(card_advised, "the script never reached a card reward");
    assert!(regret_seen, "the coach never assessed a combat move");

    // The run review covers every combat decision on the active line.
    let review = call(
        &mut wb,
        "POST",
        "/api/live/regret-review",
        json!({"config": {"sims": 16, "seed": 5, "top_n": 1}}),
    );
    let decisions = review["decisions"].as_array().unwrap();
    assert!(!decisions.is_empty(), "{review}");
    assert_eq!(review["summary"]["total_decisions"], json!(decisions.len()));
    assert!(review["blunders"].as_array().unwrap().len() <= 1);
    assert_eq!(review["config"], json!({"blunder_pct": 8.0, "top_n": 1}));
    assert!(decisions[0]["node_id"].is_number());
    assert!(decisions[0]["idx"].is_number());
    // Rewinding to a combat node keeps the snapshots usable: the hint works
    // again at the seek target.
    let tl = call(&mut wb, "GET", "/api/live/timeline", Value::Null);
    let combat_node = tl["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .find(|n| n["decision"] == json!("combat_play"))
        .map(|n| n["id"].clone())
        .expect("a combat node");
    let seek = call(
        &mut wb,
        "POST",
        "/api/live/seek",
        json!({"node_id": combat_node}),
    );
    assert_eq!(seek["step"]["decision"], json!("combat_play"));
    let hint = call(&mut wb, "POST", "/api/live/hint", hint_body());
    assert!(!hint["scores"].as_array().unwrap().is_empty());
}

/// Ending turn 1 with an untouched hand — while Strike / Defend / Bash were all
/// playable — is graded as a mistake on HP or win chance, and the coach names
/// the line the search would have played instead.
#[test]
fn last_move_regret_flags_wasted_turn() {
    let mut wb = new_wb();
    let mut step = call(
        &mut wb,
        "POST",
        "/api/live/start",
        json!({"character": "Ironclad", "seed": "SMOKE1", "ascension": 0, "act1": "overgrowth"}),
    );
    call(
        &mut wb,
        "POST",
        "/api/live/run-mode",
        json!({"off_combat_brain": "heuristic"}),
    );
    let mut shop_done = false;
    for _ in 0..200 {
        if step["decision"] == json!("combat_play") {
            break;
        }
        let (action, args) = policy(&step, &mut shop_done).expect("run continues");
        step = call(
            &mut wb,
            "POST",
            "/api/live/action",
            json!({"action": action, "args": args}),
        )["step"]
            .clone();
    }
    assert_eq!(
        step["decision"],
        json!("combat_play"),
        "never reached combat"
    );
    let hand = step["state"]["hand"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    assert!(
        hand.iter().any(|c| c["can_play"] == json!(true)),
        "the fixture needs a playable card: {hand:?}"
    );
    call(
        &mut wb,
        "POST",
        "/api/live/action",
        json!({"action": "end_turn", "args": {}}),
    );
    let r = call(
        &mut wb,
        "POST",
        "/api/live/last-move-regret",
        json!({"config": {"sims": 400, "seed": 2}}),
    );
    let verdict = r["verdict"].as_str().unwrap_or("");
    assert!(matches!(verdict, "slip" | "blunder"), "{r}");
    assert!(r["hp_drop"].is_number(), "{r}");
    let line: Vec<&str> = r["best_line"]
        .as_array()
        .expect("best_line")
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert!(!line.is_empty(), "{r}");
    assert_ne!(line[0], "end turn", "{r}");
}

/// A fresh SMOKE1 Ironclad run walked to its first combat decision.
fn first_combat(wb: &mut Workbench) -> Value {
    first_combat_seeded(wb, "SMOKE1")
}

fn first_combat_seeded(wb: &mut Workbench, seed: &str) -> Value {
    let mut step = call(
        wb,
        "POST",
        "/api/live/start",
        json!({"character": "Ironclad", "seed": seed, "ascension": 0, "act1": "overgrowth"}),
    );
    call(
        wb,
        "POST",
        "/api/live/run-mode",
        json!({"off_combat_brain": "heuristic"}),
    );
    let mut shop_done = false;
    for _ in 0..200 {
        if step["decision"] == json!("combat_play") {
            return step;
        }
        let (action, args) = policy(&step, &mut shop_done).expect("run continues");
        step = call(
            wb,
            "POST",
            "/api/live/action",
            json!({"action": action, "args": args}),
        )["step"]
            .clone();
    }
    panic!("never reached combat");
}

fn act(wb: &mut Workbench, step: &Value) -> Value {
    let mut shop_done = true;
    let (action, args) = policy(step, &mut shop_done).expect("combat continues");
    call(
        wb,
        "POST",
        "/api/live/action",
        json!({"action": action, "args": args}),
    )["step"]
        .clone()
}

const GRADE: &str = r#"{"config": {"sims": 400, "seed": 2, "peeked": true}}"#;

/// The whole turn is the unit: ending turn 1 untouched is a costly turn whose
/// record is the PROTOCOL.md `turn` shape — the mod's renderer reads it.
#[test]
fn last_turn_grade_wasted_turn_is_costly() {
    let mut wb = new_wb();
    first_combat(&mut wb);
    call(
        &mut wb,
        "POST",
        "/api/live/action",
        json!({"action": "end_turn", "args": {}}),
    );
    let r = call(
        &mut wb,
        "POST",
        "/api/live/last-turn-grade",
        serde_json::from_str(GRADE).unwrap(),
    );
    assert_eq!(r["kind"], json!("turn"), "{r}");
    assert!(r["id"].as_str().unwrap().starts_with("turn:"), "{r}");
    assert!(
        matches!(r["verdict"].as_str(), Some("slip" | "blunder")),
        "{r}"
    );
    assert_eq!(r["you"], json!(["end turn"]), "{r}");
    let ai = r["ai"].as_array().expect("ai line");
    assert!(!ai.is_empty() && ai[0] != json!("end turn"), "{r}");
    assert_eq!(r["peeked"], json!(true));
    assert_eq!(r["turn"], json!(1));
    assert!(r["severity"].as_f64().unwrap() >= 3.0, "{r}");
    assert!(
        r["context"]["hand"]
            .as_array()
            .is_some_and(|h| !h.is_empty()),
        "{r}"
    );
    assert!(r["context"]["enemies"][0]["intent"].is_string(), "{r}");
    assert_eq!(r["searches"], json!(1));
    // Nothing visible separates the lines on turn 1 (no kill, no wake).
    assert_eq!(r["why"], Value::Null, "{r}");
    assert!(r.as_object().unwrap().contains_key("why"), "{r}");
}

/// Only an End turn closes a turn: after a card play there is nothing to grade.
#[test]
fn last_turn_grade_none_mid_turn() {
    let mut wb = new_wb();
    let step = first_combat(&mut wb);
    act(&mut wb, &step);
    assert_eq!(
        status(
            &mut wb,
            "POST",
            "/api/live/last-turn-grade",
            serde_json::from_str(GRADE).unwrap()
        ),
        Err(409)
    );
}

/// Positions the hint already scored are graded from those rows: no search.
#[test]
fn last_turn_grade_reuses_hint_scores() {
    let mut wb = new_wb();
    let mut step = first_combat(&mut wb);
    let hint = json!({"config": {"sims": 400, "seed": 2}});
    for _ in 0..12 {
        call(&mut wb, "POST", "/api/live/hint", hint.clone());
        let ends = policy(&step, &mut true).is_some_and(|(a, _)| a == "end_turn");
        step = act(&mut wb, &step);
        if ends {
            break;
        }
    }
    let r = call(
        &mut wb,
        "POST",
        "/api/live/last-turn-grade",
        serde_json::from_str(GRADE).unwrap(),
    );
    assert_eq!(r["searches"], json!(0), "{r}");
    assert!(r["you"].as_array().unwrap().len() >= 2, "{r}");
}

/// Past `max_searches` the turn stays silent rather than stall the worker.
#[test]
fn last_turn_grade_over_cap_is_silent() {
    let mut wb = new_wb();
    let mut step = first_combat(&mut wb);
    for _ in 0..12 {
        let ends = policy(&step, &mut true).is_some_and(|(a, _)| a == "end_turn");
        step = act(&mut wb, &step);
        if ends {
            break;
        }
    }
    assert_eq!(
        status(
            &mut wb,
            "POST",
            "/api/live/last-turn-grade",
            json!({"config": {"sims": 400, "seed": 2, "max_searches": 1}})
        ),
        Err(409)
    );
}

/// One Strike, then End turn with two energy and a Strike in hand: the
/// engine's default 300 sims grades it a tie (≈2.4 HP), the grade's own
/// default (`TURN_GRADE_SIMS`, the advisor's 8000) a slip (≈3.3 HP).
#[test]
fn last_turn_grade_flags_an_under_played_turn() {
    let mut wb = new_wb();
    let step = first_combat_seeded(&mut wb, "B2");
    let hand = step["state"]["hand"].as_array().unwrap();
    let strike = hand
        .iter()
        .find(|c| c["name"] == json!("Strike"))
        .expect("B2 opens with a Strike");
    call(
        &mut wb,
        "POST",
        "/api/live/action",
        json!({"action": "play_card", "args": {
            "card_index": strike["index"],
            "target_index": step["state"]["enemies"][0]["index"],
        }}),
    );
    call(
        &mut wb,
        "POST",
        "/api/live/action",
        json!({"action": "end_turn", "args": {}}),
    );
    let r = call(
        &mut wb,
        "POST",
        "/api/live/last-turn-grade",
        json!({"config": {}}),
    );
    assert!(
        matches!(r["verdict"].as_str(), Some("slip" | "blunder")),
        "{r}"
    );
    assert_eq!(r["you"], json!(["Strike", "end turn"]), "{r}");
}

/// A hint weaker than the grade's search is not reused: it re-searches.
#[test]
fn last_turn_grade_researches_a_weaker_hint() {
    let mut wb = new_wb();
    first_combat(&mut wb);
    call(
        &mut wb,
        "POST",
        "/api/live/hint",
        json!({"config": {"sims": 400, "seed": 2}}),
    );
    call(
        &mut wb,
        "POST",
        "/api/live/action",
        json!({"action": "end_turn", "args": {}}),
    );
    let r = call(
        &mut wb,
        "POST",
        "/api/live/last-turn-grade",
        json!({"config": {"sims": 1000, "seed": 2}}),
    );
    assert_eq!(r["searches"], json!(1), "{r}");
}
