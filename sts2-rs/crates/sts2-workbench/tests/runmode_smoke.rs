//! Run mode (M3) over the API: the faithful start, the run agent stepping
//! through combat and off-combat, play-to-end, autobattle (skip + watch +
//! verdict), the PPO panel and node values with the real Ironclad champion
//! loaded from bytes, the WPA review / verify, run-mode saves and the
//! browser persist round-trip.
//!
//! Shape + no-panic coverage; the golden test holds the routes to the
//! Python oracle.

use std::path::Path;

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

/// The committed Ironclad champion (the sibling checkout's `data/ppo/`).
fn ironclad_champion() -> Vec<u8> {
    let file = Workbench::ppo_champion_file("Ironclad").expect("champion file");
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../sts2-pycli/data/ppo")
        .join(file);
    std::fs::read(&path).unwrap_or_else(|e| panic!("champion {} missing: {e}", path.display()))
}

/// A tiny-budget run-mode run (heuristic brain), started and stepped.
fn start_faithful(wb: &mut Workbench, seed: &str) -> Value {
    let step = call(
        wb,
        "POST",
        "/api/live/start-faithful",
        json!({"character": "Ironclad", "seed": seed, "ascension": 0, "sims": 4, "boss_sims": 8, "threads": 1}),
    );
    assert_eq!(step["decision"], json!("event_choice"));
    step
}

#[test]
fn faithful_start_and_run_mode_settings() {
    let mut wb = new_wb();
    let e = status(&mut wb, "POST", "/api/live/run-mode", json!({})).unwrap_err();
    assert_eq!(e, 400, "no live run");
    start_faithful(&mut wb, "run_3");
    let tl = call(&mut wb, "GET", "/api/live/timeline", Value::Null);
    assert_eq!(
        tl["nodes"][0]["label"],
        json!("Started Ironclad (seed 3, A0, heuristic)")
    );
    let cfg = call(&mut wb, "POST", "/api/live/run-mode", json!({}));
    assert_eq!(
        cfg,
        json!({"off_combat_brain": "heuristic", "ppo_impl": "rust", "combat_sims": 4,
               "combat_boss_sims": 8, "combat_threads": 1, "reward_sims": null})
    );
    let cfg = call(
        &mut wb,
        "POST",
        "/api/live/run-mode",
        json!({"off_combat_brain": "ppo", "combat_sims": 6, "reward_sims": 0}),
    );
    assert_eq!(cfg["off_combat_brain"], json!("ppo"));
    assert_eq!(cfg["combat_sims"], json!(6));
    assert_eq!(cfg["reward_sims"], json!(0));
    for (body, detail) in [
        (
            json!({"off_combat_brain": "oracle"}),
            "off_combat_brain must be heuristic|ppo, got 'oracle'",
        ),
        (
            json!({"ppo_impl": "jax"}),
            "ppo_impl must be python|rust, got 'jax'",
        ),
        (
            json!({"combat_sims": 0}),
            "combat_sims must be positive, got 0",
        ),
        (
            json!({"reward_sims": -1}),
            "reward_sims must be >= 0, got -1",
        ),
    ] {
        let e = wb.call("POST", "/api/live/run-mode", &body).unwrap_err();
        assert_eq!((e.status, e.detail.as_str()), (400, detail));
    }
    // A hashed text seed and a run_N label agree with `hash_seed`.
    let step = call(
        &mut wb,
        "POST",
        "/api/live/start-faithful",
        json!({"character": "Silent", "seed": "run_7"}),
    );
    assert_eq!(step["state"]["player"]["name"], json!("Silent"));
    let tl = call(&mut wb, "GET", "/api/live/timeline", Value::Null);
    assert!(tl["nodes"][0]["label"]
        .as_str()
        .unwrap()
        .starts_with("Started Silent (seed 7, A0, heuristic)"));
}

#[test]
fn run_mode_steps_through_combat_and_off_combat() {
    let mut wb = new_wb();
    start_faithful(&mut wb, "run_1");
    let mut saw_combat_search = false;
    let mut saw_off_combat = false;
    let mut last = Value::Null;
    for _ in 0..60 {
        let out = call(&mut wb, "POST", "/api/live/run-mode-step", json!({}));
        for k in [
            "step", "done", "over", "victory", "steps", "error", "debug", "search",
        ] {
            assert!(out.get(k).is_some(), "run-mode-step lacks {k}");
        }
        assert!(out["error"].is_null(), "{out}");
        if let Some(search) = out["search"].as_object() {
            saw_combat_search = true;
            assert!(search["scores"].as_array().is_some_and(|s| !s.is_empty()));
            assert!(search["action"].is_string());
            assert!(search["enemies"].is_array());
            assert_eq!(out["debug"]["is_combat"], json!(true));
            assert!(out["debug"]["summary"]
                .as_str()
                .unwrap()
                .starts_with("combat MCTS → "));
        } else {
            saw_off_combat = true;
            assert_eq!(out["debug"]["is_combat"], json!(false));
        }
        last = out;
        if last["done"].as_bool() == Some(true) || last["over"].as_bool() == Some(true) {
            break;
        }
    }
    assert!(saw_combat_search, "never searched a combat move");
    assert!(saw_off_combat, "never took an off-combat decision");
    assert!(last["steps"].as_u64().unwrap() >= 20);
    // Every agent step is a native-action node the timeline labels.
    let tl = call(&mut wb, "GET", "/api/live/timeline", Value::Null);
    let nodes = tl["nodes"].as_array().unwrap();
    assert_eq!(nodes.len() as u64, last["steps"].as_u64().unwrap() + 1);
    assert!(nodes[1..]
        .iter()
        .all(|n| n["label"].as_str().is_some_and(|l| !l.is_empty())));
    // `faithful-step` is the same stepper.
    let out = call(&mut wb, "POST", "/api/live/faithful-step", json!({}));
    assert!(out.get("step").is_some());
}

#[test]
fn play_to_end_finishes_the_run_and_seek_replays_native_nodes() {
    let mut wb = new_wb();
    start_faithful(&mut wb, "run_2");
    let out = call(&mut wb, "POST", "/api/live/play-to-end", json!({}));
    assert_eq!(out["over"], json!(true), "{}", out["error"]);
    assert_eq!(out["step"]["decision"], json!("game_over"));
    assert!(out["steps"].as_u64().unwrap() > 10);
    assert!(out["timeline"]["nodes"].as_array().unwrap().len() > 10);
    // Seek back onto a native-agent node and replay forward again.
    let path: Vec<u64> = out["timeline"]["active_path"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap())
        .collect();
    let mid = path[path.len() / 2];
    let seek = call(&mut wb, "POST", "/api/live/seek", json!({"node_id": mid}));
    assert!(seek["step"]["decision"].is_string());
    let again = call(&mut wb, "POST", "/api/live/play-to-end", json!({}));
    assert_eq!(again["over"], json!(true));
    // A second run with the same seed takes the same line (deterministic).
    let mut other = new_wb();
    start_faithful(&mut other, "run_2");
    let out2 = call(&mut other, "POST", "/api/live/play-to-end", json!({}));
    assert_eq!(out2["steps"], out["steps"]);
    assert_eq!(out2["victory"], out["victory"]);
}

#[test]
fn autobattle_stops_at_run_level_decisions_and_grades_the_pick() {
    let mut wb = new_wb();
    call(
        &mut wb,
        "POST",
        "/api/live/start",
        json!({"character": "Ironclad", "seed": "AUT0BATTLE", "ascension": 0, "act1": "overgrowth"}),
    );
    call(
        &mut wb,
        "POST",
        "/api/live/run-mode",
        json!({"off_combat_brain": "heuristic", "combat_sims": 4, "combat_boss_sims": 8}),
    );
    // Neow is an event choice: already a stop.
    let adv = call(&mut wb, "POST", "/api/live/autobattle/advance", json!({}));
    assert_eq!(adv["stopped"], json!(true));
    assert_eq!(adv["autoplayed"], json!(0));
    assert_eq!(adv["diff"]["from_floor"], Value::Null);
    assert!(adv["diff"]["lines"]
        .as_array()
        .unwrap()
        .iter()
        .any(|l| l.as_str().unwrap().starts_with("HP ")));
    // Take the first option by hand: the verdict grades it.
    let opt = &adv["step"]["state"]["options"][0];
    call(
        &mut wb,
        "POST",
        "/api/live/action",
        json!({"action": opt["action"], "args": opt["args"]}),
    );
    let verdict = call(&mut wb, "GET", "/api/live/autobattle/verdict", Value::Null);
    assert_eq!(verdict["available"], json!(true), "{verdict}");
    assert!(["matched", "near_tie", "differed"].contains(&verdict["band"].as_str().unwrap()));
    assert_eq!(verdict["brain"], json!("heuristic"));
    assert_eq!(
        verdict["margin"],
        Value::Null,
        "heuristic brain has no probabilities"
    );
    let again = call(&mut wb, "GET", "/api/live/autobattle/verdict", Value::Null);
    assert_eq!(
        again,
        json!({"available": false, "reason": "no graded decision pending"})
    );
    // Skip to the next stop: the map or the first reward, combats resolved.
    let adv = call(&mut wb, "POST", "/api/live/autobattle/advance", json!({}));
    assert!(
        adv["stopped"].as_bool().unwrap() || adv["over"].as_bool().unwrap(),
        "{adv}"
    );
    assert!(adv["diff"].is_object());
    // Watch path: one decision per call, `diff` only at a stop.
    let step = call(&mut wb, "POST", "/api/live/autobattle/step", json!({}));
    assert_eq!(
        step["played"],
        json!(false),
        "already at a stop plays nothing"
    );
    assert_eq!(step["stopped"], json!(true));
    let opt = &step["step"]["state"]["options"][0];
    call(
        &mut wb,
        "POST",
        "/api/live/action",
        json!({"action": opt["action"], "args": opt["args"]}),
    );
    let mut played_any = false;
    for _ in 0..40 {
        let s = call(&mut wb, "POST", "/api/live/autobattle/step", json!({}));
        if s["played"] == json!(true) {
            played_any = true;
            assert!(s.get("search").is_some() && s.get("debug").is_some());
        }
        if s["stopped"] == json!(true) || s["over"] == json!(true) || !s["error"].is_null() {
            assert!(s["diff"].is_object());
            break;
        }
        assert!(s["diff"].is_null(), "mid-fight there is no diff yet");
    }
    assert!(played_any);
}

#[test]
fn ppo_panel_values_and_wpa_with_the_champion() {
    let mut wb = new_wb();
    // Without a net: the panel names the missing net, values stay null.
    call(
        &mut wb,
        "POST",
        "/api/live/start",
        json!({"character": "Ironclad", "seed": "PP0PANEL", "ascension": 0, "act1": "overgrowth"}),
    );
    let panel = call(&mut wb, "POST", "/api/live/ppo-policy", json!({}));
    assert_eq!(panel["ppo_available"], json!(false));
    assert_eq!(panel["error"], json!("no PPO net loaded for Ironclad"));
    assert_eq!(panel["impl"], json!("rust"));
    assert_eq!(panel["surface"], json!("event"));
    assert!(panel["options"].as_array().unwrap().len() >= 2);
    let review = call(&mut wb, "POST", "/api/live/wpa-review", json!({}));
    assert_eq!(review["available"], json!(false));
    wb.set_ppo_error("Ironclad", "fetch failed");
    let panel = call(&mut wb, "POST", "/api/live/ppo-policy", json!({}));
    assert_eq!(
        panel["error"],
        json!("no PPO net loaded for Ironclad: fetch failed")
    );

    // Load the champion: the live run picks it up.
    assert!(!wb.has_ppo("Ironclad"));
    let e = wb.load_ppo("Ironclad", b"not an onnx").unwrap_err();
    assert_eq!(e.status, 400);
    call_load(&mut wb);
    assert!(wb.has_ppo("Ironclad"));
    let panel = call(&mut wb, "POST", "/api/live/ppo-policy", json!({}));
    assert_eq!(panel["ppo_available"], json!(true), "{panel}");
    assert!(panel["value"].is_number());
    let options = panel["options"].as_array().unwrap();
    let total: f64 = options.iter().map(|o| o["prob"].as_f64().unwrap()).sum();
    assert!((total - 1.0).abs() < 1e-3, "probs sum to 1: {total}");
    assert_eq!(
        options
            .iter()
            .filter(|o| o["is_best"] == json!(true))
            .count(),
        1
    );
    assert!(options
        .iter()
        .all(|o| o["label"].as_str().is_some_and(|l| !l.is_empty())));
    // The PPO brain answers "play best" off combat.
    let rec = call(&mut wb, "POST", "/api/live/auto-action", json!({}));
    assert_eq!(rec["policy"], json!("ppo"));
    assert!(rec.get("fallback").is_none());
    // Play the run with the PPO brain for a while: values land on the nodes.
    call(
        &mut wb,
        "POST",
        "/api/live/run-mode",
        json!({"off_combat_brain": "ppo", "combat_sims": 4, "combat_boss_sims": 8}),
    );
    for _ in 0..80 {
        let out = call(&mut wb, "POST", "/api/live/run-mode-step", json!({}));
        if out["done"] == json!(true) || out["over"] == json!(true) {
            break;
        }
    }
    let tl = call(&mut wb, "GET", "/api/live/timeline", Value::Null);
    let nodes = tl["nodes"].as_array().unwrap();
    let valued = nodes.iter().filter(|n| n["value"].is_number()).count();
    assert!(
        valued >= 2,
        "expected scored non-combat nodes, got {valued}"
    );
    assert!(nodes
        .iter()
        .filter(|n| n["decision"] == json!("combat_play"))
        .all(|n| n["value"].is_null()));
    // The WPA review has a curve now; verify a turning point (tiny arms).
    let review = call(
        &mut wb,
        "POST",
        "/api/live/wpa-review",
        json!({"config": {"top_k": 2, "min_swing": 0.0}}),
    );
    assert_eq!(review["available"], json!(true), "{review}");
    assert_eq!(review["points"].as_array().unwrap().len(), valued);
    assert_eq!(review["caveats"].as_array().unwrap().len(), 3);
    assert!(review["summary"]["n_points"].as_u64().unwrap() >= 2);
    let tps = review["turning_points"].as_array().unwrap();
    assert!(!tps.is_empty());
    let node_id = tps[0]["node_id"].as_u64().unwrap();
    let active_before = tl["active_leaf"].clone();
    let verify = call(
        &mut wb,
        "POST",
        "/api/live/wpa-verify",
        json!({"node_id": node_id, "max_steps": 6, "max_seconds": 60}),
    );
    assert_eq!(verify["node_id"], json!(node_id));
    assert!(["same", "alternative_better", "played_better"]
        .contains(&verify["verdict"].as_str().unwrap()));
    for arm in ["played", "alternative"] {
        assert!(verify[arm]["label"].is_string());
        assert!(verify[arm]["steps"].as_u64().unwrap() <= 6);
        assert!(verify[arm]["over"].is_boolean());
    }
    assert!(verify["note"]
        .as_str()
        .unwrap()
        .contains("paired spot check"));
    let tl = call(&mut wb, "GET", "/api/live/timeline", Value::Null);
    assert_eq!(
        tl["active_leaf"], active_before,
        "the active leaf is restored"
    );
    let e = status(
        &mut wb,
        "POST",
        "/api/live/wpa-verify",
        json!({"node_id": 999_999}),
    )
    .unwrap_err();
    assert_eq!(e, 404);
    // A node with no follow-up move. Not `active_leaf`: when the turning
    // point verified above IS the active leaf, the two continuation branches
    // it adds hang off it, so it is no longer childless.
    let leaf = tl["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .find(|n| n["child_ids"].as_array().is_some_and(Vec::is_empty))
        .and_then(|n| n["id"].as_u64())
        .expect("a childless node");
    let e = wb
        .call("POST", "/api/live/wpa-verify", &json!({"node_id": leaf}))
        .unwrap_err();
    assert_eq!(e.status, 409);
    assert!(e.detail.contains("no follow-up move"));
    // Combat: the panel does not apply.
    // (The run is wherever the stepper left it; force a combat state check
    // by stepping until combat if needed.)
    for _ in 0..40 {
        let st = call(&mut wb, "GET", "/api/live/state", Value::Null);
        if st["decision"] == json!("combat_play") {
            let e = status(&mut wb, "POST", "/api/live/ppo-policy", json!({})).unwrap_err();
            assert_eq!(e, 409);
            break;
        }
        let out = call(&mut wb, "POST", "/api/live/run-mode-step", json!({}));
        if out["over"] == json!(true) {
            break;
        }
    }
}

fn call_load(wb: &mut Workbench) {
    let bytes = ironclad_champion();
    wb.load_ppo("Ironclad", &bytes).expect("load champion");
}

#[test]
fn run_mode_saves_restore_with_their_values_and_persist_round_trips() {
    let mut wb = new_wb();
    call_load(&mut wb);
    start_faithful(&mut wb, "run_5");
    call(
        &mut wb,
        "POST",
        "/api/live/run-mode",
        json!({"off_combat_brain": "ppo"}),
    );
    for _ in 0..30 {
        let out = call(&mut wb, "POST", "/api/live/run-mode-step", json!({}));
        if out["done"] == json!(true) || out["over"] == json!(true) {
            break;
        }
    }
    let before = call(&mut wb, "GET", "/api/live/timeline", Value::Null);
    let values_before: Vec<Value> = before["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n["value"].clone())
        .collect();
    assert!(values_before.iter().any(Value::is_number));
    let state_before = call(&mut wb, "GET", "/api/live/state", Value::Null);

    // A named save of a run-mode run restores through the faithful
    // constructor with its native-agent line and recorded values.
    let payload = wb
        .build_save("m3 run", "2026-01-01T00:00:00Z")
        .expect("save");
    assert_eq!(payload["run_mode"], json!(true));
    assert_eq!(payload["seed"], json!("5"));
    assert!(payload["line"]
        .as_array()
        .unwrap()
        .iter()
        .any(|e| e["action"] == json!("__agent_native__")));
    assert_eq!(payload["node_values"], json!(values_before));
    let mut fresh = new_wb();
    fresh.load_ppo("Ironclad", &ironclad_champion()).unwrap();
    let restored = fresh.restore_save(&payload).expect("restore");
    assert_eq!(restored["decision"], state_before["decision"]);
    assert_eq!(restored["state"]["floor"], state_before["state"]["floor"]);
    let after = call(&mut fresh, "GET", "/api/live/timeline", Value::Null);
    let values_after: Vec<Value> = after["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n["value"].clone())
        .collect();
    assert_eq!(values_after, values_before, "values replay, not recompute");
    // The restored run keeps stepping (the agent was rebuilt).
    let out = call(&mut fresh, "POST", "/api/live/run-mode-step", json!({}));
    assert!(out["error"].is_null());

    // The browser persist carries values, the native nodes and the run mode.
    let persist = wb.export_persist();
    assert_eq!(persist["run_mode"], json!(true));
    assert_eq!(persist["run_mode_config"]["off_combat_brain"], json!("ppo"));
    let mut reloaded = new_wb();
    let mut ticks = 0;
    let step = reloaded
        .import_persist(&persist, |_, _| ticks += 1)
        .expect("import");
    assert!(ticks > 0);
    assert_eq!(step["decision"], state_before["decision"]);
    let tl = call(&mut reloaded, "GET", "/api/live/timeline", Value::Null);
    let values: Vec<Value> = tl["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n["value"].clone())
        .collect();
    assert_eq!(values, values_before);
    assert_eq!(tl["active_leaf"], before["active_leaf"]);
    let cfg = call(&mut reloaded, "POST", "/api/live/run-mode", json!({}));
    assert_eq!(cfg["off_combat_brain"], json!("ppo"));
    assert_eq!(cfg["combat_sims"], json!(4));
}

#[test]
fn stream_frames_still_work_beside_run_mode() {
    // A run-mode step in the middle of an advisor session must not disturb
    // the M2 stream contract (regression guard for the shared session).
    let mut wb = new_wb();
    start_faithful(&mut wb, "run_9");
    let mut frames = Vec::new();
    wb.call_stream(
        "POST",
        "/api/live/decision-advisor/stream",
        &json!({"config": {"rollouts": 1, "sims": 0, "max_steps": 4, "seed": 1}}),
        &mut |f| frames.push(f.clone()),
    );
    assert_eq!(frames.first().unwrap()["type"], json!("start"));
    assert!(matches!(
        frames.last().unwrap()["type"].as_str(),
        Some("result" | "error")
    ));
}

/// Start a fresh overgrowth run (first decision: the Neow event).
fn start_neow(wb: &mut Workbench, seed: &str) {
    call(
        wb,
        "POST",
        "/api/live/start",
        json!({"character": "Ironclad", "seed": seed, "ascension": 0, "act1": "overgrowth"}),
    );
}

/// The policy panel's options sorted by probability, highest first.
fn ranked_options(wb: &mut Workbench) -> Vec<Value> {
    let panel = call(wb, "POST", "/api/live/ppo-policy", json!({}));
    assert_eq!(panel["ppo_available"], json!(true));
    let mut opts = panel["options"].as_array().unwrap().clone();
    opts.sort_by(|a, b| b["prob"].as_f64().partial_cmp(&a["prob"].as_f64()).unwrap());
    opts
}

fn take(wb: &mut Workbench, opt: &Value) {
    let r = call(
        wb,
        "POST",
        "/api/live/action",
        json!({"action": opt["action"], "args": opt["args"]}),
    );
    assert!(r["error"].is_null(), "{r}");
}

#[test]
fn choice_grade_after_a_neow_pick() {
    let mut wb = new_wb();
    // No net: nothing is snapshotted, so there is no grade.
    start_neow(&mut wb, "COACHCHOICE");
    let opts = call(&mut wb, "POST", "/api/live/ppo-policy", json!({}));
    let first = opts["options"][0].clone();
    take(&mut wb, &first);
    let e = status(&mut wb, "POST", "/api/live/last-choice-grade", json!({})).unwrap_err();
    assert_eq!(e, 409);

    start_neow(&mut wb, "COACHCHOICE");
    call_load(&mut wb);
    let root = call(&mut wb, "GET", "/api/live/timeline", Value::Null)["active_leaf"].clone();
    let opts = ranked_options(&mut wb);
    take(&mut wb, opts.last().unwrap());
    let g = call(&mut wb, "POST", "/api/live/last-choice-grade", json!({}));
    assert_eq!(g["decision"], json!("event_choice"));
    assert_eq!(g["kind"], json!("Event"));
    assert!(
        matches!(g["verdict"].as_str(), Some("flag" | "tossup")),
        "{g}"
    );
    assert_eq!(g["played"]["rank"], json!(opts.len()));
    assert_eq!(g["n_options"], json!(opts.len()));
    assert_eq!(g["top"]["label"], opts[0]["label"]);
    assert_eq!(g["checkable"], json!(true));

    // Undo: the grade belongs to the move just undone, so it is withdrawn.
    call(&mut wb, "POST", "/api/live/seek", json!({"node_id": root}));
    let e = status(&mut wb, "POST", "/api/live/last-choice-grade", json!({})).unwrap_err();
    assert_eq!(e, 409);

    take(&mut wb, &opts[0]);
    let g = call(&mut wb, "POST", "/api/live/last-choice-grade", json!({}));
    assert_eq!(g["verdict"], json!("agree"));
    assert_eq!(g["played"]["rank"], json!(1));
    assert_eq!(g["checkable"], json!(false));
}

#[test]
fn choice_check_streams_you_vs_ai() {
    let mut wb = new_wb();
    start_neow(&mut wb, "COACHCHECK");
    call_load(&mut wb);
    let opts = ranked_options(&mut wb);
    take(&mut wb, opts.last().unwrap());
    let g = call(&mut wb, "POST", "/api/live/last-choice-grade", json!({}));
    let config =
        json!({"node_id": g["node_id"], "rollouts": 1, "sims": 0, "max_steps": 4, "seed": 1});
    let mut frames = Vec::new();
    wb.call_stream(
        "POST",
        "/api/live/last-choice-check/stream",
        &json!({"config": config}),
        &mut |f| frames.push(f.clone()),
    );
    let start = frames.first().unwrap();
    assert_eq!(start["type"], json!("start"));
    assert_eq!(start["candidates"].as_array().unwrap().len(), 2);
    let result = frames.last().unwrap();
    assert_eq!(result["type"], json!("result"), "{result}");
    let mut keys: Vec<&str> = result["candidates"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["key"].as_str().unwrap())
        .collect();
    keys.sort_unstable();
    assert_eq!(keys, ["ai", "you"]);

    // An agreeing pick has nothing to check.
    start_neow(&mut wb, "COACHCHECK");
    let opts = ranked_options(&mut wb);
    take(&mut wb, &opts[0]);
    let g = call(&mut wb, "POST", "/api/live/last-choice-grade", json!({}));
    let mut frames = Vec::new();
    wb.call_stream(
        "POST",
        "/api/live/last-choice-check/stream",
        &json!({"config": {"node_id": g["node_id"]}}),
        &mut |f| frames.push(f.clone()),
    );
    assert_eq!(
        frames,
        vec![json!({"type": "error", "error": "nothing to check"})]
    );
}
