//! Reduce a non-combat decision-point state to uniform, clickable options.
//!
//! Port of `sts2-pycli/python/workbench/options.py`.
//!
//! Each option is a dict `{"label", "action", "args", "enabled"}` where
//! `action`/`args` are exactly what `/api/live/action` forwards to the game.
//! The mapping mirrors the `python/play.py` interactive dispatch loop. Any
//! decision we do not explicitly handle yields a single `proceed` button so a
//! live run can never get stuck.

use serde_json::{json, Map, Value};

use crate::render::py;

/// `{"label": label, "action": action, "args": args, "enabled": enabled}`.
///
/// `label` stays a raw JSON value: the Python passes whatever the state's
/// `name` slot held (a string in every real state), and the parity golden
/// keeps that shape.
fn opt(label: Value, action: &str, args: Map<String, Value>, enabled: bool) -> Value {
    json!({"label": label, "action": action, "args": args, "enabled": enabled})
}

/// `f"Option {o.get('index')}"` / `f"Card {cd.get('index')}"` / `f"Pack …"`.
fn indexed_label(prefix: &str, item: &Value) -> String {
    format!("{prefix} {}", py::opt_str(py::get_raw(item, "index")))
}

/// `d.get("name", f"{prefix} {d.get('index')}")` — the raw `name` slot when
/// the key is present (even `null`), else the indexed fallback.
fn name_or_indexed(item: &Value, prefix: &str) -> Value {
    py::get_raw(item, "name")
        .cloned()
        .unwrap_or_else(|| Value::String(indexed_label(prefix, item)))
}

/// `{key: item.get("index")}`.
fn index_args(key: &str, item: &Value) -> Map<String, Value> {
    let mut args = Map::new();
    args.insert(
        key.to_string(),
        py::get_raw(item, "index").cloned().unwrap_or(Value::Null),
    );
    args
}

/// `f"Buy {x.get('name')} ({x.get('cost', '?')}g)"`.
fn buy_label(item: &Value) -> Value {
    let name = py::opt_str(py::get_raw(item, "name"));
    let cost = py::get_raw(item, "cost").map_or_else(|| "?".to_string(), py::to_str);
    Value::String(format!("Buy {name} ({cost}g)"))
}

/// Return the selectable options for `state`, or `[]` for combat/terminal.
///
/// * `state`: a decision-point state dict (must carry `decision`).
///
/// Returns uniform option dicts. Empty for `combat_play` (the board handles
/// it) and `game_over` (terminal).
#[must_use]
pub fn normalize_options(state: &Value) -> Vec<Value> {
    // `state.get("decision", "")` — a non-string (or null) decision matches no
    // screen below and lands on the safety net, exactly as in Python.
    let decision = py::get(state, "decision")
        .and_then(Value::as_str)
        .unwrap_or("");
    if matches!(decision, "combat_play" | "game_over") {
        return Vec::new();
    }

    // `state.get(k, []) or []` — a present-but-null / non-list slot reads empty.
    let items = |k: &str| py::get_list(state, k).iter().filter(|v| v.is_object());

    match decision {
        "map_select" => items("choices")
            .map(|ch| {
                let col = py::get_raw(ch, "col").cloned().unwrap_or(Value::Null);
                let row = py::get_raw(ch, "row").cloned().unwrap_or(Value::Null);
                let node = py::get_raw(ch, "type").map_or_else(|| "Node".to_string(), py::to_str);
                let label = format!("{node} ({},{})", py::to_str(&col), py::to_str(&row));
                let mut args = Map::new();
                args.insert("col".to_string(), col);
                args.insert("row".to_string(), row);
                opt(Value::String(label), "select_map_node", args, true)
            })
            .collect(),

        "card_reward" => {
            let mut out: Vec<Value> = items("cards")
                .map(|cd| {
                    opt(
                        name_or_indexed(cd, "Card"),
                        "select_card_reward",
                        index_args("card_index", cd),
                        true,
                    )
                })
                .collect();
            if py::get_raw(state, "can_skip").is_none_or(py::truthy) {
                out.push(opt(
                    Value::from("Skip"),
                    "skip_card_reward",
                    Map::new(),
                    true,
                ));
            }
            out
        }

        "bundle_select" => items("bundles")
            .map(|b| {
                opt(
                    Value::String(indexed_label("Pack", b)),
                    "select_bundle",
                    index_args("bundle_index", b),
                    true,
                )
            })
            .collect(),

        "card_select" => {
            let mut out: Vec<Value> = items("cards")
                .map(|cd| {
                    let mut args = Map::new();
                    args.insert(
                        "indices".to_string(),
                        Value::String(py::opt_str(py::get_raw(cd, "index"))),
                    );
                    opt(name_or_indexed(cd, "Card"), "select_cards", args, true)
                })
                .collect();
            // Skippable two ways, and they are NOT the same flag. `min_select ==
            // 0` means "picking nothing satisfies me"; `can_skip` is the
            // engine's own `canSkip` on a choice that otherwise wants one card —
            // Discovery and the Colorless / Skill / Attack / Power potions are
            // min 1, max 1, can_skip true. Gating on `min_select` alone dropped
            // a legal action: the combat search scores a `choice:skip` on those
            // screens while neither this list nor the UI offered one (bug
            // wb-31c2d79f995fe1a6).
            let one = Value::from(1);
            let min_is_zero = py::eq(py::get_or(state, "min_select", &one), &Value::from(0));
            if min_is_zero || py::truthy_opt(py::get_raw(state, "can_skip")) {
                out.push(opt(Value::from("Skip"), "skip_select", Map::new(), true));
            }
            out
        }

        "shop" => {
            let mut out = Vec::new();
            for cd in items("cards") {
                if py::truthy_opt(py::get_raw(cd, "is_stocked")) {
                    out.push(opt(
                        buy_label(cd),
                        "buy_card",
                        index_args("card_index", cd),
                        true,
                    ));
                }
            }
            for r in items("relics") {
                if py::truthy_opt(py::get_raw(r, "is_stocked")) {
                    out.push(opt(
                        buy_label(r),
                        "buy_relic",
                        index_args("relic_index", r),
                        true,
                    ));
                }
            }
            for p in items("potions") {
                if py::truthy_opt(py::get_raw(p, "is_stocked")) {
                    out.push(opt(
                        buy_label(p),
                        "buy_potion",
                        index_args("potion_index", p),
                        true,
                    ));
                }
            }
            if let Some(cost) = py::get_raw(state, "card_removal_cost").filter(|c| py::truthy(c)) {
                let label = format!("Remove a card ({}g)", py::to_str(cost));
                out.push(opt(Value::String(label), "remove_card", Map::new(), true));
            }
            out.push(opt(
                Value::from("Leave shop"),
                "leave_room",
                Map::new(),
                true,
            ));
            out
        }

        "rest_site" => items("options")
            .map(|o| {
                // `o.get("option_id") or o.get("name", f"Option {index}")`.
                let label = py::get_raw(o, "option_id")
                    .filter(|v| py::truthy(v))
                    .cloned()
                    .unwrap_or_else(|| name_or_indexed(o, "Option"));
                let enabled = py::get_raw(o, "is_enabled").is_none_or(py::truthy);
                opt(
                    label,
                    "choose_option",
                    index_args("option_index", o),
                    enabled,
                )
            })
            .collect(),

        "event_choice" => items("options")
            .map(|o| {
                let title = py::get_raw(o, "title")
                    .filter(|v| py::truthy(v))
                    .or_else(|| py::get_raw(o, "name").filter(|v| py::truthy(v)))
                    .map_or_else(|| indexed_label("Option", o), py::to_str);
                let enabled = !py::truthy_opt(py::get_raw(o, "is_locked"));
                opt(
                    Value::String(title),
                    "choose_option",
                    index_args("option_index", o),
                    enabled,
                )
            })
            .collect(),

        // Safety net: unknown decision -> a single continue button.
        _ => vec![opt(Value::from("Continue"), "proceed", Map::new(), true)],
    }
}

#[cfg(test)]
mod tests {
    //! Port of `tests/workbench/test_options.py` (every case, same names).

    use super::*;

    fn action(o: &Value) -> &str {
        o["action"].as_str().unwrap()
    }

    #[test]
    fn test_combat_has_no_options() {
        assert!(normalize_options(&json!({"decision": "combat_play"})).is_empty());
    }

    #[test]
    fn test_card_reward_lists_cards_and_skip() {
        let state = json!({
            "decision": "card_reward",
            "cards": [{"index": 0, "name": "Strike"}, {"index": 1, "name": "Bash"}],
            "can_skip": true,
        });
        let opts = normalize_options(&state);
        assert_eq!(
            opts[0],
            json!({
                "label": "Strike",
                "action": "select_card_reward",
                "args": {"card_index": 0},
                "enabled": true,
            })
        );
        assert_eq!(
            opts[opts.len() - 1],
            json!({"label": "Skip", "action": "skip_card_reward", "args": {}, "enabled": true})
        );
    }

    #[test]
    fn test_card_reward_no_skip_when_not_allowed() {
        let state = json!({
            "decision": "card_reward",
            "cards": [{"index": 0, "name": "Strike"}],
            "can_skip": false,
        });
        assert!(!normalize_options(&state)
            .iter()
            .any(|o| action(o) == "skip_card_reward"));
    }

    #[test]
    fn test_map_select_uses_col_row() {
        let state =
            json!({"decision": "map_select", "choices": [{"col": 2, "row": 1, "type": "Monster"}]});
        let opts = normalize_options(&state);
        let o = &opts[0];
        assert_eq!(action(o), "select_map_node");
        assert_eq!(o["args"], json!({"col": 2, "row": 1}));
        assert!(o["label"].as_str().unwrap().contains("Monster"));
        assert_eq!(o["label"], "Monster (2,1)");
    }

    #[test]
    fn test_rest_site_skips_disabled() {
        let state = json!({
            "decision": "rest_site",
            "options": [
                {"index": 0, "option_id": "HEAL", "is_enabled": true},
                {"index": 1, "option_id": "DIG", "is_enabled": false},
            ],
        });
        let opts = normalize_options(&state);
        let enabled: Vec<bool> = opts
            .iter()
            .map(|o| o["enabled"].as_bool().unwrap())
            .collect();
        assert_eq!(enabled, vec![true, false]);
        assert_eq!(
            opts[0],
            json!({
                "label": "HEAL",
                "action": "choose_option",
                "args": {"option_index": 0},
                "enabled": true,
            })
        );
    }

    #[test]
    fn test_event_marks_locked_disabled() {
        let state = json!({
            "decision": "event_choice",
            "options": [
                {"index": 0, "title": "Take the gold", "is_locked": false},
                {"index": 1, "title": "Locked path", "is_locked": true},
            ],
        });
        let opts = normalize_options(&state);
        assert_eq!(opts[0]["label"], "Take the gold");
        assert_eq!(opts[0]["enabled"], true);
        assert_eq!(opts[1]["enabled"], false);
    }

    #[test]
    fn test_shop_lists_stocked_items_then_remove_and_leave() {
        let state = json!({
            "decision": "shop",
            "cards": [
                {"index": 0, "name": "Strike", "cost": 50, "is_stocked": true},
                {"index": 1, "name": "Gone", "is_stocked": false},
            ],
            "relics": [{"index": 0, "name": "Akabeko", "cost": 150, "is_stocked": true}],
            "potions": [{"index": 0, "name": "Fire", "cost": 60, "is_stocked": true}],
            "card_removal_cost": 75,
        });
        let opts = normalize_options(&state);
        let actions: Vec<&str> = opts.iter().map(action).collect();
        assert!(actions.contains(&"buy_card"));
        assert!(actions.contains(&"buy_relic"));
        assert!(actions.contains(&"buy_potion"));
        assert_eq!(&actions[actions.len() - 2..], ["remove_card", "leave_room"]);
        assert!(opts
            .iter()
            .all(|o| !o["label"].as_str().unwrap().contains("Gone")));
        // Exact label strings, since the golden compares them byte for byte.
        assert_eq!(opts[0]["label"], "Buy Strike (50g)");
        assert_eq!(opts[3]["label"], "Remove a card (75g)");
    }

    #[test]
    fn test_bundle_select() {
        let state = json!({"decision": "bundle_select", "bundles": [{"index": 0}, {"index": 1}]});
        let opts = normalize_options(&state);
        assert_eq!(
            opts[1],
            json!({
                "label": "Pack 1",
                "action": "select_bundle",
                "args": {"bundle_index": 1},
                "enabled": true,
            })
        );
    }

    #[test]
    fn test_card_select_skip_when_min_zero_or_can_skip() {
        // Both routes to a legal skip, and they are independent flags.
        // `min_select == 0` means "nothing is a valid selection"; `can_skip` is
        // the engine's own `canSkip` on a choice that otherwise wants a card —
        // Discovery and the Colorless potion are min 1 / max 1 / can_skip true.
        // Reading only `min_select` dropped their skip from the option list
        // while the combat search still scored it (bug wb-31c2d79f995fe1a6).
        let has_skip = |s: &Value| {
            normalize_options(s)
                .iter()
                .any(|o| action(o) == "skip_select")
        };
        let mut base = json!({
            "decision": "card_select",
            "cards": [{"index": 0, "name": "Wound"}],
            "min_select": 0,
            "max_select": 1,
        });
        assert!(has_skip(&base));
        base["min_select"] = json!(1);
        assert!(!has_skip(&base));
        base["can_skip"] = json!(true);
        assert!(has_skip(&base));
        base["can_skip"] = json!(false);
        assert!(!has_skip(&base));
        // The `indices` arg is the stringified index.
        assert_eq!(normalize_options(&base)[0]["args"], json!({"indices": "0"}));
    }

    #[test]
    fn test_unknown_decision_falls_back_to_proceed() {
        let opts = normalize_options(&json!({"decision": "weird_new_screen"}));
        assert_eq!(
            opts,
            vec![json!({"label": "Continue", "action": "proceed", "args": {}, "enabled": true})]
        );
    }

    #[test]
    fn missing_fields_render_like_python() {
        // `f"{ch.get('type', 'Node')} ({col},{row})"` with nothing set.
        let opts = normalize_options(&json!({"decision": "map_select", "choices": [{}]}));
        assert_eq!(opts[0]["label"], "Node (None,None)");
        assert_eq!(opts[0]["args"], json!({"col": null, "row": null}));
        // A present-but-null `name` is the label (Python's `.get` default does
        // not apply), an absent one falls back to the indexed label.
        let opts = normalize_options(
            &json!({"decision": "card_reward", "cards": [{"index": 3, "name": null}, {"index": 4}], "can_skip": false}),
        );
        assert_eq!(opts[0]["label"], Value::Null);
        assert_eq!(opts[1]["label"], "Card 4");
    }
}
