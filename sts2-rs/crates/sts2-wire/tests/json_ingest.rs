//! A captured wire dict goes from JSON text to a live `CombatState` with no
//! Python interpreter anywhere in the process.
//!
//! This is the property the crate exists for. Before the split, reaching
//! `CombatState` from a wire dict meant linking `libpython` and holding the
//! GIL, so nothing outside the `PyO3` extension could replay a capture. The
//! test crate here depends only on `sts2-wire` — if a `pyo3` symbol ever
//! creeps back into the parse path, this stops linking.

use sts2_wire::{parse_json, state_from_json_str, WireError};

fn fixture() -> String {
    std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/combat_action_ironclad.json"),
    )
    .expect("fixture readable")
}

#[test]
fn a_captured_dict_parses_without_python() {
    let text = fixture();
    let value: serde_json::Value = serde_json::from_str(&text).expect("valid JSON");
    let parsed = parse_json(&value).expect("fixture must parse");

    assert_eq!(parsed.wire_version, 1);
    assert_eq!(parsed.player.class, "ironclad");
    assert_eq!(parsed.player.hp, 54);
    assert_eq!(parsed.player.relics.len(), 2);
    assert_eq!(parsed.player.gold, Some(137));
    assert_eq!(parsed.deck.hand.len(), 2);
    assert_eq!(parsed.deck.draw_pile.len(), 3);
    assert_eq!(parsed.deck.pinned_top, vec![1]);
    assert_eq!(parsed.encounter.len(), 2);
    assert_eq!(parsed.ascension, 7);

    // A `u64` seed above `i64::MAX` — the case a naive `as_i64` read would
    // have silently rejected.
    assert_eq!(parsed.rng_seed, Some(18_446_744_073_709_551_557));

    // Key-sorted, per the `WireValue::entries` contract.
    // Sorted, though the fixture lists them shuffle / monster_ai /
    // combat_targets.
    let counter_names: Vec<&str> = parsed
        .rng_counters
        .iter()
        .map(|(k, _)| k.as_str())
        .collect();
    assert_eq!(counter_names, ["combat_targets", "monster_ai", "shuffle"]);
}

#[test]
fn a_captured_dict_rebuilds_a_combat_state() {
    let state = state_from_json_str(&fixture()).expect("fixture must rebuild");

    assert_eq!(state.player.current_hp, 54);
    assert_eq!(state.player.max_hp, 80);
    assert_eq!(state.player.block, 5);
    // Two enemies on the wire, one of them dead.
    assert_eq!(state.enemy_uids.len(), 2);
    // The piles came across, not just the scalars.
    assert_eq!(state.player.hand.len(), 2);
    assert_eq!(state.player.draw.len(), 3);
    assert_eq!(state.player.discard.len(), 1);
}

/// A wire `combat_vars` value lands behind the card's "may be nonzero" mask
/// (J02). `Card::combat_var` answers 0 from the mask alone when the key's bit
/// is clear, so a reconstruction path that wrote a value without going
/// through the mask-maintaining writers would read the fixture's
/// `free_this_turn: 1` back as 0, and the card would stop being free.
#[test]
fn wire_combat_vars_read_back_through_the_mask() {
    use sts2_core::cards::{CardId, VarKey};

    let state = state_from_json_str(&fixture()).expect("fixture must rebuild");
    let strike = state
        .player
        .hand
        .iter()
        .map(|uid| &state.cards[uid.get() as usize - 1])
        .find(|c| c.card_id == CardId::StrikeIronclad)
        .expect("the hand Strike came across");
    assert!(strike.combat_vars.maybe_nonzero(VarKey::FreeThisTurn));
    assert_eq!(strike.combat_var(VarKey::FreeThisTurn), 1);
    assert!(sts2_core::hooks::card_cost_locally_zeroed(&state, strike));
}

/// The synthetic `status:` / `curse:` ids resolve to their real cards rather
/// than degrading to the AscendersBane placeholder — the single largest
/// reconstruction gap the DLL path ever had, and worth pinning on the
/// interpreter-free route too.
#[test]
fn synthetic_status_and_curse_ids_resolve() {
    let value: serde_json::Value = serde_json::from_str(&fixture()).unwrap();
    let parsed = parse_json(&value).unwrap();

    let slimed = parsed
        .deck
        .draw_pile
        .iter()
        .find(|c| c.raw_id == "status:Slimed")
        .expect("slimed in draw pile");
    assert!(
        slimed.card_id.is_some(),
        "`status:Slimed` must resolve to a real CardId, not fall through to the placeholder"
    );

    let clumsy = parsed
        .deck
        .hand
        .iter()
        .find(|c| c.raw_id == "curse:Clumsy")
        .expect("clumsy in hand");
    assert!(clumsy.card_id.is_some(), "`curse:Clumsy` must resolve");
}

#[test]
fn failures_carry_their_path_and_class() {
    let mut value: serde_json::Value = serde_json::from_str(&fixture()).unwrap();

    // Unmodelled content → Unsupported.
    value["player"]["relics"][1]["id"] = serde_json::json!("DEFINITELY_NOT_A_RELIC");
    let err = parse_json(&value).unwrap_err();
    assert!(matches!(err, WireError::Unsupported { .. }), "got {err:?}");
    assert_eq!(err.path(), "$.player.relics[1]");
    assert!(err.to_string().contains("DEFINITELY_NOT_A_RELIC"));

    // Invariant violation → Schema.
    let mut value: serde_json::Value = serde_json::from_str(&fixture()).unwrap();
    value["player"]["hp"] = serde_json::json!(999);
    let err = parse_json(&value).unwrap_err();
    assert!(matches!(err, WireError::Schema { .. }), "got {err:?}");
    assert_eq!(err.path(), "$.player");

    // Missing required key → Schema, at the parent path.
    let mut value: serde_json::Value = serde_json::from_str(&fixture()).unwrap();
    value.as_object_mut().unwrap().remove("phase");
    let err = parse_json(&value).unwrap_err();
    assert_eq!(err.path(), "$");
    assert!(err.to_string().contains("missing required key `phase`"));
}

#[test]
fn malformed_json_text_fails_at_the_root() {
    let err = state_from_json_str("{not json").unwrap_err();
    assert_eq!(err.path(), "$");
    assert!(err.to_string().contains("invalid JSON"));
}

/// GH #787: `$.game_version` is optional — absent means the default build —
/// and an explicit one reaches `CombatState::game_version`.
#[test]
fn game_version_key_is_optional_and_threaded() {
    let mut value: serde_json::Value = serde_json::from_str(&fixture()).expect("valid JSON");
    assert!(
        value.get("game_version").is_none(),
        "fixture predates the key"
    );
    let default = sts2_wire::state_from_json(&value).expect("default rebuild");
    assert_eq!(default.game_version, sts2_core::GameVersion::V0_107_1);

    value["game_version"] = serde_json::json!("0.111.0");
    let v111 = sts2_wire::state_from_json(&value).expect("v0.111.0 rebuild");
    assert_eq!(v111.game_version, sts2_core::GameVersion::V0_111_0);

    value["game_version"] = serde_json::json!("v9.9.9");
    assert!(sts2_wire::state_from_json(&value).is_err(), "unknown build");
}

/// The emitter writes `game_version` only for a non-default build, so every
/// v0.107.1 wire keeps its historical key set, and the key round-trips.
#[test]
fn game_version_is_emitted_only_when_non_default() {
    let mut state = state_from_json_str(&fixture()).expect("fixture must rebuild");
    let wire = sts2_wire::emit::combat::combat_state_to_value(&state).expect("emit");
    assert!(wire.get("game_version").is_none());

    state.game_version = sts2_core::GameVersion::V0_111_0;
    let wire = sts2_wire::emit::combat::combat_state_to_value(&state).expect("emit");
    assert_eq!(wire["game_version"], serde_json::json!("v0.111.0"));
    let back = sts2_wire::state_from_json(&wire).expect("round trip");
    assert_eq!(back.game_version, sts2_core::GameVersion::V0_111_0);
}

/// GH #788: `$.rng.run_seed` is a `uint` for v0.107.1 and a `ulong` for
/// v0.111.0; the reconstruction seeds its streams in the wire's build.
#[test]
fn run_seed_width_follows_the_game_version() {
    let wide: u64 = 0xCCD0_8772_EAB0_03D2;
    let mut value: serde_json::Value = serde_json::from_str(&fixture()).expect("valid JSON");
    value["rng"]["run_seed"] = serde_json::json!(wide);
    value["rng"]["counters"] = serde_json::json!({});
    let err = sts2_wire::state_from_json(&value).unwrap_err();
    assert_eq!(err.path(), "$.rng.run_seed", "v0.107.1 seeds are 32-bit");

    value["game_version"] = serde_json::json!("v0.111.0");
    let state = sts2_wire::state_from_json(&value).expect("v0.111.0 accepts a u64 seed");
    assert_eq!(state.run_rng.seed, wide);
    assert_eq!(state.run_rng.game_version, sts2_core::GameVersion::V0_111_0);
    assert_eq!(
        state.run_rng,
        sts2_core::RunRngSet::new_for(sts2_core::GameVersion::V0_111_0, wide)
    );
    // ...and the emitter writes the full width back out.
    let wire = sts2_wire::emit::combat::combat_state_to_value(&state).expect("emit");
    assert_eq!(wire["rng"]["run_seed"], serde_json::json!(wide));
}

/// The scenario builder threads the version like ascension, and the legacy
/// entry point is exactly the default build.
#[test]
fn scenario_builder_threads_game_version() {
    let deck = vec!["STRIKE_IRONCLAD".to_string()];
    let build = |v| {
        sts2_wire::scenario::build_scenario_state_for_version(
            v,
            3,
            "Ironclad",
            &deck,
            50,
            80,
            &[],
            &[],
            "SHRINKER_BEETLE_WEAK",
            0,
            None,
            None,
        )
        .expect("scenario builds")
    };
    assert_eq!(
        build(sts2_core::GameVersion::V0_111_0).game_version,
        sts2_core::GameVersion::V0_111_0
    );
    let legacy = sts2_wire::scenario::build_scenario_state(
        3,
        "Ironclad",
        &deck,
        50,
        80,
        &[],
        &[],
        "SHRINKER_BEETLE_WEAK",
        0,
        None,
        None,
    )
    .expect("scenario builds");
    assert_eq!(legacy, build(sts2_core::GameVersion::V0_107_1));
}

/// GH #929: a v0.111.0 wire whose hand holds a v0.111.0-only card (the
/// DowsingRod's `DOWSING` quest card, or any of the build's new cards) must
/// rebuild. The id lookup used to walk only the v0.107.1 universe, so every
/// live advice request with such a card in a pile failed `UnsupportedStateError`.
#[test]
fn v0111_only_cards_parse_from_a_v0111_wire() {
    use sts2_core::cards::CardId;
    for id in CardId::V0111_ONLY {
        let mut value: serde_json::Value = serde_json::from_str(&fixture()).expect("valid JSON");
        value["game_version"] = serde_json::json!("v0.111.0");
        value["deck"]["hand"][0] = serde_json::json!({
            "id": id.to_python_name(),
            "stable_idx": 4,
            "kind": "card",
            "upgraded": false,
        });
        let state = sts2_wire::state_from_json(&value)
            .unwrap_or_else(|e| panic!("{}: {e:?}", id.to_python_name()));
        let first = state.player.hand[0];
        assert_eq!(state.cards[first.0.get() as usize - 1].card_id, id);
    }
}
