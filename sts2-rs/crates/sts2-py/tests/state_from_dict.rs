//! Integration tests for `state_from_dict` — hint-T2 (bead
//! `sts2-rs-osyf`). Verifies that a parsed wire-dict produces a
//! [`CoreCombatState`] whose `legal_actions()` reflects the injected
//! mid-turn snapshot and that the state passes the new
//! `CombatState::assert_invariants()` predicate.

#![allow(clippy::missing_panics_doc)]

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use sts2_rs::state_dict::state_from_dict;

// ---------------------------------------------------------------------------
// Fixture builders — minimum-viable dict for an Ironclad-vs-Cultist
// mid-turn snapshot.
// ---------------------------------------------------------------------------

fn empty_list(py: Python<'_>) -> Bound<'_, PyList> {
    PyList::empty(py)
}

fn make_card<'py>(
    py: Python<'py>,
    id: &str,
    stable_idx: i32,
    upgraded: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("id", id)?;
    d.set_item("stable_idx", stable_idx)?;
    d.set_item("kind", "card")?;
    d.set_item("upgraded", upgraded)?;
    Ok(d)
}

fn make_deck_with_ironclad_starter(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    // Ironclad starter: 5x Strike, 4x Defend, 1x Bash. Distribute across
    // the canonical mid-turn shape:
    //  - hand:    Strike, Strike, Defend, Defend, Bash  (5 cards)
    //  - draw:    Strike, Strike, Strike, Defend       (4 cards)
    //  - discard: Defend                                (1 card)
    //  - exhaust: (empty)
    let hand = PyList::empty(py);
    hand.append(make_card(py, "STRIKE_IRONCLAD", 1, false)?)?;
    hand.append(make_card(py, "STRIKE_IRONCLAD", 2, false)?)?;
    hand.append(make_card(py, "DEFEND_IRONCLAD", 3, false)?)?;
    hand.append(make_card(py, "DEFEND_IRONCLAD", 4, false)?)?;
    hand.append(make_card(py, "BASH", 5, false)?)?;

    let draw = PyList::empty(py);
    draw.append(make_card(py, "STRIKE_IRONCLAD", 6, false)?)?;
    draw.append(make_card(py, "STRIKE_IRONCLAD", 7, false)?)?;
    draw.append(make_card(py, "STRIKE_IRONCLAD", 8, false)?)?;
    draw.append(make_card(py, "DEFEND_IRONCLAD", 9, false)?)?;

    let discard = PyList::empty(py);
    discard.append(make_card(py, "DEFEND_IRONCLAD", 10, false)?)?;

    let d = PyDict::new(py);
    d.set_item("draw_pile", draw)?;
    d.set_item("discard_pile", discard)?;
    d.set_item("hand", hand)?;
    d.set_item("exhaust_pile", empty_list(py))?;
    Ok(d)
}

fn make_player(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("class", "ironclad")?;
    d.set_item("hp", 70)?;
    d.set_item("max_hp", 80)?;
    d.set_item("block", 6)?;
    d.set_item("energy", 3)?;
    d.set_item("max_energy", 3)?;
    d.set_item("energy_regen_per_turn", 3)?;

    // Equip Burning Blood as a starting relic.
    let relics = PyList::empty(py);
    let r = PyDict::new(py);
    r.set_item("id", "BURNING_BLOOD")?;
    relics.append(r)?;
    d.set_item("relics", relics)?;

    d.set_item("powers", empty_list(py))?;
    d.set_item("potions", empty_list(py))?;
    Ok(d)
}

fn make_nibbit_encounter(py: Python<'_>) -> PyResult<Bound<'_, PyList>> {
    let enc = PyList::empty(py);
    let nibbit = PyDict::new(py);
    nibbit.set_item("kind", "Nibbit")?;
    nibbit.set_item("uid", 1_i64)?;
    nibbit.set_item("hp", 48)?;
    nibbit.set_item("max_hp", 48)?;
    nibbit.set_item("block", 0)?;
    nibbit.set_item("powers", empty_list(py))?;
    nibbit.set_item("intents", empty_list(py))?;
    nibbit.set_item("intent_history", empty_list(py))?;
    nibbit.set_item("is_dead", false)?;
    nibbit.set_item("is_escaped", false)?;
    enc.append(nibbit)?;
    Ok(enc)
}

fn make_mid_turn_fixture(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("wire_version", 1)?;
    d.set_item("phase", "PlayerPlayPhase")?;
    d.set_item("round", 1)?;
    d.set_item("side", "Player")?;
    d.set_item("is_elite", false)?;
    d.set_item("turn", 1)?;
    d.set_item("combat_id", 0_i64)?;
    d.set_item("player", make_player(py)?)?;
    d.set_item("deck", make_deck_with_ironclad_starter(py)?)?;
    d.set_item("encounter", make_nibbit_encounter(py)?)?;

    let rng = PyDict::new(py);
    rng.set_item("seed", 42_u64)?;
    rng.set_item("ascension", 0_u8)?;
    d.set_item("rng", rng)?;
    Ok(d)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn state_from_dict_builds_legal_actions_for_ironclad_vs_cultist() {
    Python::with_gil(|py| {
        let d = make_mid_turn_fixture(py).unwrap();
        let state = state_from_dict(&d).expect("baseline mid-turn fixture must build");

        // Structural invariants must hold.
        state
            .assert_invariants()
            .expect("invariants must hold on freshly-built state");

        // Player + monster wiring.
        assert!(
            state.player.creature_uid.is_some(),
            "player.creature_uid must be Some after state_from_dict"
        );
        assert_eq!(
            state.enemy_uids.len(),
            1,
            "Cultist fixture should produce 1 enemy"
        );
        assert_eq!(state.player.current_hp, 70);
        assert_eq!(state.player.max_hp, 80);
        assert_eq!(state.player.block, 6);
        assert_eq!(state.player.energy, 3);

        // Deck piles wired through.
        assert_eq!(state.player.hand.len(), 5);
        assert_eq!(state.player.draw.len(), 4);
        assert_eq!(state.player.discard.len(), 1);
        assert_eq!(state.player.exhaust.len(), 0);

        // Relics carried over.
        assert_eq!(state.relics.len(), 1, "expected exactly Burning Blood");

        // Ascension + is_elite propagate.
        assert_eq!(state.ascension, 0);
        assert!(!state.is_elite);

        // legal_actions enumerates: 5 hand cards (each either targeted
        // at the Cultist or untargeted) + EndTurn. The exact count
        // depends on the action surface, but we expect at *minimum* the
        // EndTurn action plus one play action per playable card. We
        // assert the EndTurn action is present.
        let actions = sts2_core::legal_actions(&state);
        assert!(
            actions
                .iter()
                .any(|a| matches!(a, sts2_core::Action::EndTurn)),
            "legal_actions must include EndTurn, got: {actions:?}"
        );
        // At least one PlayCard action should be legal with 3 energy +
        // 5 in hand (Strikes cost 1, Defends cost 1, Bash costs 2).
        assert!(
            actions
                .iter()
                .any(|a| matches!(a, sts2_core::Action::PlayCard { .. })),
            "legal_actions must include at least one PlayCard, got: {actions:?}"
        );
    });
}

#[test]
fn state_from_dict_dead_enemies_reconstruct_as_corpses() {
    // Contract CHANGE (2026-07-13 revive-blindness fix): dead enemies on
    // the wire are revivers the game deliberately kept in combat
    // (Decimillipede REATTACH, IllusionPower). They must reconstruct as
    // 0-hp corpses — present in `enemy_uids` like the native engine keeps
    // them (alive-filters gate targeting/perform) — so revive machinery
    // runs in rollouts. Escaped enemies are still dropped.
    Python::with_gil(|py| {
        let d = make_mid_turn_fixture(py).unwrap();
        let enc = d.get_item("encounter").unwrap().unwrap();
        let cult = enc.downcast::<PyList>().unwrap().get_item(0).unwrap();
        cult.downcast::<PyDict>()
            .unwrap()
            .set_item("is_dead", true)
            .unwrap();
        let state = state_from_dict(&d).expect("dead-enemy fixture must build");
        state.assert_invariants().expect("invariants hold");
        assert_eq!(
            state.enemy_uids.len(),
            1,
            "dead enemy must stay in enemy_uids as a corpse"
        );
        let corpse_uid = state.enemy_uids[0];
        let corpse = state
            .creatures
            .iter()
            .find(|c| c.id == corpse_uid.0)
            .expect("corpse creature present");
        assert_eq!(corpse.current_hp, 0, "corpse hp clamped to dead");

        // Escaped enemies are still dropped entirely.
        cult.downcast::<PyDict>()
            .unwrap()
            .set_item("is_dead", false)
            .unwrap();
        cult.downcast::<PyDict>()
            .unwrap()
            .set_item("is_escaped", true)
            .unwrap();
        let state = state_from_dict(&d).expect("escaped-enemy fixture must build");
        assert_eq!(state.enemy_uids.len(), 0, "escaped enemy is dropped");
    });
}

#[test]
fn state_from_dict_propagates_is_elite_and_ascension() {
    Python::with_gil(|py| {
        let d = make_mid_turn_fixture(py).unwrap();
        d.set_item("is_elite", true).unwrap();
        let rng = d
            .get_item("rng")
            .unwrap()
            .unwrap()
            .downcast::<PyDict>()
            .unwrap()
            .clone();
        rng.set_item("ascension", 5_u8).unwrap();
        let state = state_from_dict(&d).expect("must build");
        assert!(state.is_elite);
        assert_eq!(state.ascension, 5);
    });
}

#[test]
fn state_from_dict_player_powers_land_in_arena() {
    Python::with_gil(|py| {
        let d = make_mid_turn_fixture(py).unwrap();
        // Add a Strength power to the player.
        let player = d
            .get_item("player")
            .unwrap()
            .unwrap()
            .downcast::<PyDict>()
            .unwrap()
            .clone();
        let powers = PyList::empty(py);
        let p = PyDict::new(py);
        p.set_item("name", "STRENGTH").unwrap();
        p.set_item("amount", 2_i32).unwrap();
        powers.append(p).unwrap();
        player.set_item("powers", powers).unwrap();

        let state = state_from_dict(&d).expect("must build");
        state.assert_invariants().expect("invariants hold");
        assert_eq!(state.powers.len(), 1, "Strength power should be in arena");
        assert_eq!(state.player.powers.len(), 1);
        // The player's Creature mirror also carries the power.
        let pc_uid = state.player.creature_uid.unwrap();
        let pc = state.creatures.iter().find(|c| c.id == pc_uid.0).unwrap();
        assert_eq!(pc.powers.len(), 1);
    });
}

/// TestSubject 2nd-form reconstruction (2026-07-17 matched-seed gap audit):
/// a wire snapshot carrying PainfulStabs must infer the spent revive
/// (Adaptable counter = 1, AI counter = 1 — otherwise the search believes
/// two revives remain and fights a phantom repeat 2nd form) and recover
/// `ExtraMultiClawCount` from the telegraphed MULTI_CLAW hit count
/// (counter = wire hits − base 3).
#[test]
fn state_from_dict_infers_test_subject_second_form_counters() {
    Python::with_gil(|py| {
        let d = make_mid_turn_fixture(py).unwrap();

        let enc = PyList::empty(py);
        let ts = PyDict::new(py);
        ts.set_item("kind", "TestSubject").unwrap();
        ts.set_item("uid", 1_i64).unwrap();
        ts.set_item("hp", 150).unwrap();
        ts.set_item("max_hp", 200).unwrap();
        ts.set_item("block", 0).unwrap();
        let powers = PyList::empty(py);
        for (name, amount) in [("ADAPTABLE", 1_i32), ("PAINFUL_STABS", 1_i32)] {
            let p = PyDict::new(py);
            p.set_item("name", name).unwrap();
            p.set_item("amount", amount).unwrap();
            powers.append(p).unwrap();
        }
        ts.set_item("powers", powers).unwrap();
        // Telegraphed MULTI_CLAW at 5 hits → 2 repeats already performed.
        let intents = PyList::empty(py);
        let it = PyDict::new(py);
        it.set_item("kind", "Attack").unwrap();
        it.set_item("amount", 10_i32).unwrap();
        it.set_item("times", 5_i32).unwrap();
        intents.append(it).unwrap();
        ts.set_item("intents", intents).unwrap();
        ts.set_item("intent_history", empty_list(py)).unwrap();
        ts.set_item("is_dead", false).unwrap();
        ts.set_item("is_escaped", false).unwrap();
        ts.set_item("move_state", "MULTI_CLAW_MOVE").unwrap();
        enc.append(ts).unwrap();
        d.set_item("encounter", enc).unwrap();

        let state = state_from_dict(&d).expect("must build");
        state.assert_invariants().expect("invariants hold");

        let boss = state
            .creatures
            .iter()
            .find(|c| c.monster_id.0 == "TestSubject")
            .expect("TestSubject reconstructed");
        let power = |kind: sts2_core::PowerKind| {
            boss.powers
                .iter()
                .find_map(|pu| state.powers.iter().find(|p| p.id == pu.0 && p.kind == kind))
        };
        assert_eq!(
            power(sts2_core::PowerKind::Adaptable).map(|p| p.counter),
            Some(1),
            "one revive spent → Adaptable counter 1 (single revive remains)"
        );
        assert_eq!(
            power(sts2_core::PowerKind::PainfulStabs).map(|p| p.counter),
            Some(2),
            "ExtraMultiClawCount recovered from the 5-hit telegraph (5 − 3)"
        );
        assert_eq!(
            boss.ai.as_ref().map(sts2_core::MonsterAi::counter),
            Some(1),
            "AI counter mirrors the respawn count for REVIVE_BRANCH routing"
        );
        assert_eq!(
            boss.ai.as_ref().map(|ai| ai.current_move().id),
            Some("MULTI_CLAW"),
            "wire MULTI_CLAW_MOVE relaxed-resolves onto the engine state"
        );
    });
}

#[test]
fn state_from_dict_infers_aeonglass_escalation_and_wither_levels() {
    Python::with_gil(|py| {
        let d = make_mid_turn_fixture(py).unwrap();

        // A level-2 Wither (wire base damage 9 = 3 + 3·2) in the discard.
        let wither = make_card(py, "WITHER", 11, false).unwrap();
        wither.set_item("kind", "curse").unwrap();
        wither.set_item("base_damage", 9).unwrap();
        let deck = d
            .get_item("deck")
            .unwrap()
            .unwrap()
            .downcast_into::<PyDict>()
            .unwrap();
        let discard = deck
            .get_item("discard_pile")
            .unwrap()
            .unwrap()
            .downcast_into::<PyList>()
            .unwrap();
        discard.append(wither).unwrap();

        let enc = PyList::empty(py);
        let boss = PyDict::new(py);
        boss.set_item("kind", "Aeonglass").unwrap();
        boss.set_item("uid", 1_i64).unwrap();
        boss.set_item("hp", 400).unwrap();
        boss.set_item("max_hp", 512).unwrap();
        boss.set_item("block", 0).unwrap();
        let powers = PyList::empty(py);
        let wp = PyDict::new(py);
        wp.set_item("name", "WITHERING_PRESENCE").unwrap();
        wp.set_item("amount", 6).unwrap();
        // C# DisplayAmount: 2 player cards left until the next Wither.
        wp.set_item("display", 2).unwrap();
        powers.append(wp).unwrap();
        boss.set_item("powers", powers).unwrap();
        let intents = PyList::empty(py);
        let it = PyDict::new(py);
        it.set_item("kind", "Attack").unwrap();
        it.set_item("amount", 26_i32).unwrap();
        intents.append(it).unwrap();
        boss.set_item("intents", intents).unwrap();
        boss.set_item("intent_history", empty_list(py)).unwrap();
        boss.set_item("is_dead", false).unwrap();
        boss.set_item("is_escaped", false).unwrap();
        boss.set_item("move_state", "EBB_MOVE").unwrap();
        enc.append(boss).unwrap();
        d.set_item("encounter", enc).unwrap();

        let state = state_from_dict(&d).expect("must build");
        state.assert_invariants().expect("invariants hold");

        let boss = state
            .creatures
            .iter()
            .find(|c| c.monster_id.0 == "Aeonglass")
            .expect("Aeonglass reconstructed");
        assert_eq!(
            boss.ai.as_ref().map(sts2_core::MonsterAi::counter),
            Some(2),
            "WitherUpgradeCount recovered from the max Wither level in the piles"
        );
        let wp = boss
            .powers
            .iter()
            .find_map(|pu| {
                state
                    .powers
                    .iter()
                    .find(|p| p.id == pu.0 && p.kind == sts2_core::PowerKind::WitheringPresence)
            })
            .expect("WitheringPresence reconstructed");
        assert_eq!(wp.amount, 6, "cycle length preserved");
        assert_eq!(
            wp.counter, 4,
            "cards-played countdown phase recovered from wire display (6 − 2)"
        );
        let wither = state
            .cards
            .iter()
            .find(|c| c.card_id == sts2_core::cards::CardId::Wither)
            .expect("Wither card reconstructed");
        assert_eq!(
            wither.combat_var(sts2_core::cards::VarKey::WitherFakeUpgradeLevel),
            2,
            "FakeUpgrade level restored from wire base damage (9 → level 2)"
        );
    });
}

#[test]
fn state_from_dict_infers_surrounded_facing_and_unbakes_back_attack() {
    Python::with_gil(|py| {
        let d = make_mid_turn_fixture(py).unwrap();

        // Player carries Surrounded (applied by Rocket at spawn).
        let player = d
            .get_item("player")
            .unwrap()
            .unwrap()
            .downcast_into::<PyDict>()
            .unwrap();
        let ppowers = PyList::empty(py);
        let sp = PyDict::new(py);
        sp.set_item("name", "SURROUNDED").unwrap();
        sp.set_item("amount", 1).unwrap();
        ppowers.append(sp).unwrap();
        player.set_item("powers", ppowers).unwrap();

        let make_arm = |kind: &str,
                        hp: i32,
                        back_attack: &str,
                        move_state: &str,
                        wire_damage: i32|
         -> Bound<'_, PyDict> {
            let arm = PyDict::new(py);
            arm.set_item("kind", kind).unwrap();
            arm.set_item("uid", i64::from(hp)).unwrap();
            arm.set_item("hp", hp).unwrap();
            arm.set_item("max_hp", 209).unwrap();
            arm.set_item("block", 0).unwrap();
            let powers = PyList::empty(py);
            for name in [back_attack, "CRAB_RAGE"] {
                let p = PyDict::new(py);
                p.set_item("name", name).unwrap();
                p.set_item("amount", 1).unwrap();
                powers.append(p).unwrap();
            }
            arm.set_item("powers", powers).unwrap();
            let intents = PyList::empty(py);
            let it = PyDict::new(py);
            it.set_item("kind", "Attack").unwrap();
            it.set_item("amount", wire_damage).unwrap();
            intents.append(it).unwrap();
            arm.set_item("intents", intents).unwrap();
            arm.set_item("intent_history", empty_list(py)).unwrap();
            arm.set_item("is_dead", false).unwrap();
            arm.set_item("is_escaped", false).unwrap();
            arm.set_item("move_state", move_state).unwrap();
            arm
        };

        // Facing Right: Crusher (BackAttackLeft) telegraphs the boosted
        // THRASH — wire 18 = ⌊12·3/2⌋ — while Rocket shows its plain 3.
        let enc = PyList::empty(py);
        enc.append(make_arm(
            "Crusher",
            209,
            "BACK_ATTACK_LEFT",
            "THRASH_MOVE",
            18,
        ))
        .unwrap();
        enc.append(make_arm(
            "Rocket",
            199,
            "BACK_ATTACK_RIGHT",
            "TARGETING_RETICLE_MOVE",
            3,
        ))
        .unwrap();
        d.set_item("encounter", enc).unwrap();

        let state = state_from_dict(&d).expect("must build");
        state.assert_invariants().expect("invariants hold");

        let surrounded = state
            .powers
            .iter()
            .find(|p| p.kind == sts2_core::PowerKind::Surrounded)
            .expect("player Surrounded reconstructed");
        assert_eq!(
            surrounded.counter, 0,
            "boosted BackAttackLeft telegraph ⇒ facing Right (counter 0)"
        );
        let crusher = state
            .creatures
            .iter()
            .find(|c| c.monster_id.0 == "Crusher")
            .expect("Crusher reconstructed");
        assert_eq!(
            crusher.intent,
            Some(sts2_core::Intent::Attack { damage: 12 }),
            "wire 18 unbaked to base 12 (the ×1.5 back-attack inverted)"
        );
        let rocket = state
            .creatures
            .iter()
            .find(|c| c.monster_id.0 == "Rocket")
            .expect("Rocket reconstructed");
        assert_eq!(
            rocket.intent,
            Some(sts2_core::Intent::Attack { damage: 3 }),
            "front arm's plain telegraph is untouched"
        );

        // Facing Left: Crusher plain 12, Rocket boosted ⌊3·3/2⌋ = 4.
        let enc = PyList::empty(py);
        enc.append(make_arm(
            "Crusher",
            209,
            "BACK_ATTACK_LEFT",
            "THRASH_MOVE",
            12,
        ))
        .unwrap();
        enc.append(make_arm(
            "Rocket",
            199,
            "BACK_ATTACK_RIGHT",
            "TARGETING_RETICLE_MOVE",
            4,
        ))
        .unwrap();
        d.set_item("encounter", enc).unwrap();

        let state = state_from_dict(&d).expect("must build");
        let surrounded = state
            .powers
            .iter()
            .find(|p| p.kind == sts2_core::PowerKind::Surrounded)
            .expect("player Surrounded reconstructed");
        assert_eq!(
            surrounded.counter, 1,
            "plain BackAttackLeft telegraph ⇒ facing Left (counter 1)"
        );
        let rocket = state
            .creatures
            .iter()
            .find(|c| c.monster_id.0 == "Rocket")
            .expect("Rocket reconstructed");
        assert_eq!(
            rocket.intent,
            Some(sts2_core::Intent::Attack { damage: 3 }),
            "back arm's boosted wire 4 unbaked to base 3"
        );

        // One arm dead: face the survivor — its back-attack can't trigger.
        let enc = PyList::empty(py);
        let dead_rocket = make_arm(
            "Rocket",
            0,
            "BACK_ATTACK_RIGHT",
            "TARGETING_RETICLE_MOVE",
            3,
        );
        dead_rocket.set_item("is_dead", true).unwrap();
        enc.append(make_arm(
            "Crusher",
            100,
            "BACK_ATTACK_LEFT",
            "THRASH_MOVE",
            12,
        ))
        .unwrap();
        enc.append(dead_rocket).unwrap();
        d.set_item("encounter", enc).unwrap();

        let state = state_from_dict(&d).expect("must build");
        let surrounded = state
            .powers
            .iter()
            .find(|p| p.kind == sts2_core::PowerKind::Surrounded)
            .expect("player Surrounded reconstructed");
        assert_eq!(
            surrounded.counter, 1,
            "sole survivor carries BackAttackLeft ⇒ face it (Left, no boost)"
        );
    });
}

#[test]
fn assert_invariants_catches_hp_overflow() {
    // Build a valid state, then mutate it to violate an invariant.
    Python::with_gil(|py| {
        let d = make_mid_turn_fixture(py).unwrap();
        let mut state = state_from_dict(&d).expect("must build");
        // Force HP > max_hp.
        state.player.current_hp = state.player.max_hp + 1;
        let err = state.assert_invariants().expect_err("must detect overflow");
        assert!(
            err.contains("current_hp"),
            "expected hp violation, got: {err}"
        );
    });
}

#[test]
fn state_from_dict_rejects_mismatched_wire_version() {
    // hint-T14 (bead `sts2-rs-rk1d`) — simulates an older sts2-cli
    // client emitting `wire_version=0` against a newer `sts2_rs` whose
    // `WIRE_VERSION` constant is 1. The integration contract requires
    // a typed Python `UnsupportedStateError` whose message names BOTH
    // the expected and the received version, so the adapter can log
    // an actionable error and gracefully disable hints for the
    // session. See `docs/schema-version-policy.md`.
    Python::with_gil(|py| {
        let d = make_mid_turn_fixture(py).unwrap();
        // Force a downgrade — emulate an older client.
        d.set_item("wire_version", 0_u32).unwrap();

        let err = state_from_dict(&d).expect_err("mismatched wire_version must raise");
        let msg = err.to_string();
        assert!(
            msg.contains("wire_version"),
            "error must mention the field name; got: {msg}"
        );
        // Must surface BOTH the expected (1, the current constant) and
        // the received (0) versions so adapters can log the gap.
        assert!(
            msg.contains('0') && msg.contains('1'),
            "error must contain both the received (0) and expected (1) \
             versions; got: {msg}"
        );
        // Type discipline — must be `UnsupportedStateError`, not a
        // generic `ValueError`, so the adapter can catch it
        // narrowly. The PyErr type-check goes through the GIL.
        let err_type = err.get_type(py);
        let qualname = err_type
            .qualname()
            .ok()
            .map(|q| q.to_string())
            .unwrap_or_default();
        assert!(
            qualname.contains("UnsupportedStateError"),
            "expected UnsupportedStateError, got: {qualname}"
        );
    });
}

#[test]
fn assert_invariants_catches_dangling_creature_uid() {
    Python::with_gil(|py| {
        let d = make_mid_turn_fixture(py).unwrap();
        let mut state = state_from_dict(&d).expect("must build");
        // Drop the player Creature record, leaving creature_uid dangling.
        state.creatures.clear();
        let err = state
            .assert_invariants()
            .expect_err("must detect dangling uid");
        assert!(
            err.contains("creature_uid") || err.contains("not found"),
            "expected dangling-uid violation, got: {err}"
        );
    });
}

/// A wire Defend telegraph carries no amount (C# `DefendIntent` is
/// icon-only), so `parse_intent` builds `Intent::Defend { block: 0 }` — and
/// `perform_intent` gains monster block FROM THAT PAYLOAD. The
/// reconstruction must recover the amount from the move table the resolved
/// AI is parked on (v27wr sweep: 76 zero-block Defends — Axebot BOOT_UP 10,
/// MechaKnight WINDUP 15, LouseProgenitor CURL_AND_GROW 14, …).
#[test]
fn state_from_dict_recovers_defend_telegraph_block() {
    Python::with_gil(|py| {
        let d = make_mid_turn_fixture(py).unwrap();

        let enc = PyList::empty(py);
        let bot = PyDict::new(py);
        bot.set_item("kind", "Axebot").unwrap();
        bot.set_item("uid", 1_i64).unwrap();
        bot.set_item("hp", 74).unwrap();
        bot.set_item("max_hp", 74).unwrap();
        bot.set_item("block", 0).unwrap();
        bot.set_item("powers", empty_list(py)).unwrap();
        let intents = PyList::empty(py);
        let it = PyDict::new(py);
        it.set_item("kind", "Defend").unwrap();
        intents.append(it).unwrap();
        bot.set_item("intents", intents).unwrap();
        bot.set_item("intent_history", empty_list(py)).unwrap();
        bot.set_item("is_dead", false).unwrap();
        bot.set_item("is_escaped", false).unwrap();
        bot.set_item("move_state", "BOOT_UP_MOVE").unwrap();
        enc.append(bot).unwrap();
        d.set_item("encounter", enc).unwrap();

        let state = state_from_dict(&d).expect("must build");
        let axebot = state
            .creatures
            .iter()
            .find(|c| c.monster_id.0 == "Axebot")
            .expect("Axebot reconstructed");
        assert_eq!(
            axebot.intent,
            Some(sts2_core::Intent::Defend { block: 10 }),
            "BOOT_UP's 10 block recovered from the move table, not left at 0"
        );
    });
}

/// Run gold gates in-combat relic hooks (SealOfGold spends 5 gold for +1
/// energy at every player turn start). A wire that carries it must land on
/// `state.gold`; one that doesn't keeps the constructor default (0), never
/// an error (older producers omit it).
#[test]
fn state_from_dict_carries_player_gold() {
    Python::with_gil(|py| {
        let d = make_mid_turn_fixture(py).unwrap();
        let player = d
            .get_item("player")
            .unwrap()
            .unwrap()
            .downcast_into::<PyDict>()
            .unwrap();
        player.set_item("gold", 123).unwrap();
        let state = state_from_dict(&d).expect("must build");
        assert_eq!(state.gold, 123);

        let d2 = make_mid_turn_fixture(py).unwrap();
        let state2 = state_from_dict(&d2).expect("must build");
        assert_eq!(state2.gold, 0, "absent gold defaults, never errors");
    });
}

/// RedSkull's `_strengthApplied` latch is a pure function of the wire: C#
/// keeps `StrengthApplied == (hp <= 50% max)` at every quiescent point. An
/// undisclosed `counter: null` must derive it from HP instead of defaulting
/// to 0 — a below-threshold rebuild with a zeroed latch re-applied the +3
/// (16 of the 188 `$.player.powers` divergences on the v27wr sweep).
#[test]
fn state_from_dict_derives_red_skull_latch_from_hp() {
    Python::with_gil(|py| {
        let build = |hp: i32| {
            let d = make_mid_turn_fixture(py).unwrap();
            let player = d
                .get_item("player")
                .unwrap()
                .unwrap()
                .downcast_into::<PyDict>()
                .unwrap();
            player.set_item("hp", hp).unwrap();
            player.set_item("max_hp", 80).unwrap();
            player.set_item("block", 0).unwrap();
            let relics = PyList::empty(py);
            let r = PyDict::new(py);
            r.set_item("id", "RED_SKULL").unwrap();
            relics.append(r).unwrap();
            player.set_item("relics", relics).unwrap();
            state_from_dict(&d).expect("must build")
        };
        let below = build(20);
        let latch = below
            .relics
            .iter()
            .find(|r| r.kind == sts2_core::relics::RelicKind::RedSkull)
            .unwrap();
        assert_eq!(latch.counter, 1, "hp 20/80 => Strength already applied");
        let above = build(70);
        let latch = above
            .relics
            .iter()
            .find(|r| r.kind == sts2_core::relics::RelicKind::RedSkull)
            .unwrap();
        assert_eq!(latch.counter, 0, "hp 70/80 => latch unarmed");
    });
}

/// Wriggler slots come from UID RANK among all Wrigglers on the wire —
/// corpses included — not from position in the (living-first) encounter
/// list. C# routes `SPAWNED_MOVE`'s follow-up off `Creature.SlotName`
/// (`wriggler1/3 → NASTY_BITE`, `wriggler2/4 → WRIGGLE`); the adapter
/// re-appends corpses at the END of `encounter`, so a position-based
/// ordinal slid every survivor down one slot after a death and inverted
/// their telegraphs (attack↔buff) for the rest of the fight (13 v34wr
/// intent entries, seeds 1018/1033/1045).
#[test]
fn wriggler_slots_survive_a_reaped_sibling() {
    Python::with_gil(|py| {
        let d = make_mid_turn_fixture(py).unwrap();

        let enc = PyList::empty(py);
        // Survivors first (uids 2, 3, 4), the reaped wriggler1 appended
        // last — exactly the adapter's living-first ordering.
        for (uid, dead) in [(2_i64, false), (3, false), (4, false), (1, true)] {
            let w = PyDict::new(py);
            w.set_item("kind", "Wriggler").unwrap();
            w.set_item("uid", uid).unwrap();
            w.set_item("hp", if dead { 0 } else { 11 }).unwrap();
            w.set_item("max_hp", 11).unwrap();
            w.set_item("block", 0).unwrap();
            w.set_item("powers", empty_list(py)).unwrap();
            let intents = PyList::empty(py);
            if !dead {
                let it = PyDict::new(py);
                it.set_item("kind", "Stun").unwrap();
                intents.append(it).unwrap();
            }
            w.set_item("intents", intents).unwrap();
            w.set_item("intent_history", empty_list(py)).unwrap();
            w.set_item("is_dead", dead).unwrap();
            w.set_item("is_escaped", false).unwrap();
            if !dead {
                w.set_item("move_state", "SPAWNED_MOVE").unwrap();
            }
            enc.append(w).unwrap();
        }
        d.set_item("encounter", enc).unwrap();

        let state = state_from_dict(&d).expect("must build");

        let follow_up_of = |uid: i64| {
            state
                .creatures
                .iter()
                .find(|c| c.monster_id.0 == "Wriggler" && i64::from(c.id.get()) == uid)
                .and_then(|c| c.ai.as_ref())
                .and_then(|ai| ai.states.get("SPAWNED_MOVE"))
                .and_then(|s| match s {
                    sts2_core::monster_ai::MonsterState::Move(m) => m.follow_up,
                    _ => None,
                })
        };
        // wriggler1 died; survivors hold slots 2 (WRIGGLE), 3 (BITE), 4 (WRIGGLE).
        assert_eq!(follow_up_of(2), Some("WRIGGLE"), "uid 2 is wriggler2");
        assert_eq!(follow_up_of(3), Some("NASTY_BITE"), "uid 3 is wriggler3");
        assert_eq!(follow_up_of(4), Some("WRIGGLE"), "uid 4 is wriggler4");
    });
}

/// One enemy on the wire, fully specified — for the intent-magnitude tests
/// below. `powers` is `(python_name, amount)`; `intent` is
/// `(amount, times)`.
fn make_single_enemy_fixture<'py>(
    py: Python<'py>,
    kind: &str,
    move_state: &str,
    powers: &[(&str, i32)],
    intent: (i32, i32),
) -> PyResult<Bound<'py, PyDict>> {
    let d = make_mid_turn_fixture(py)?;
    let enc = PyList::empty(py);
    let e = PyDict::new(py);
    e.set_item("kind", kind)?;
    e.set_item("uid", 1_i64)?;
    e.set_item("hp", 200)?;
    e.set_item("max_hp", 300)?;
    e.set_item("block", 0)?;
    let pw = PyList::empty(py);
    for (name, amount) in powers {
        let p = PyDict::new(py);
        p.set_item("name", *name)?;
        p.set_item("amount", *amount)?;
        pw.append(p)?;
    }
    e.set_item("powers", pw)?;
    let intents = PyList::empty(py);
    let it = PyDict::new(py);
    it.set_item("kind", "Attack")?;
    it.set_item("amount", intent.0)?;
    it.set_item("times", intent.1)?;
    intents.append(it)?;
    e.set_item("intents", intents)?;
    e.set_item("intent_history", empty_list(py))?;
    e.set_item("is_dead", false)?;
    e.set_item("is_escaped", false)?;
    e.set_item("move_state", move_state)?;
    enc.append(e)?;
    d.set_item("encounter", enc)?;
    Ok(d)
}

fn only_enemy_intent(state: &sts2_core::CombatState) -> Option<sts2_core::Intent> {
    let uid = state.enemy_uids[0];
    state
        .creatures
        .iter()
        .find(|c| c.id == uid.0)
        .and_then(|c| c.intent)
}

/// Wire telegraphs are UI-EFFECTIVE, so ingestion inverts them — but the
/// forward chain FLOORS, and under Weak (×3/4) every third effective value
/// has two integer preimages. Taking the smallest is a guess that holds only
/// while the modifier set does: the moment the player's card changes it
/// inside the same step, the mirror re-bakes from a base 1 low and
/// under-states incoming damage for the rest of the fight (7 v34wr intent
/// entries — TestSubject BIG_POUNCE 45/44 ×2, Rocket LASER 31/30, Tunneler
/// BITE 13/12, LouseProgenitor WEB_CANNON 9/8, ScrollOfBiting CHEW 5/4,
/// Exoskeleton SKITTER 1/0).
///
/// The move the wire pins the AI to knows the true base, so it breaks the
/// tie. Tunneler BITE_MOVE is 13; Weak makes the wire read ⌊13·3/4⌋ = 9,
/// whose preimages are {12, 13}.
#[test]
fn wire_base_intent_breaks_the_unbake_truncation_tie() {
    Python::with_gil(|py| {
        let d =
            make_single_enemy_fixture(py, "Tunneler", "BITE_MOVE", &[("WEAK", 1)], (9, 1)).unwrap();
        let state = state_from_dict(&d).expect("must build");
        assert_eq!(
            only_enemy_intent(&state),
            Some(sts2_core::Intent::Attack { damage: 13 }),
            "BITE_MOVE's move-table base (13) must win over the smallest \
             preimage (12) — both re-bake to the wire's 9 under Weak"
        );

        // The wire is still the authority. Park the AI on a move whose base
        // cannot produce the reported amount and the arithmetic inverse
        // stands: BURROW_MOVE telegraphs a Buff, so there is no attack base
        // to borrow and nothing changes.
        let d = make_single_enemy_fixture(py, "Tunneler", "BURROW_MOVE", &[("WEAK", 1)], (9, 1))
            .unwrap();
        let state = state_from_dict(&d).expect("must build");
        assert_eq!(
            only_enemy_intent(&state),
            Some(sts2_core::Intent::Attack { damage: 12 }),
            "a non-attack move contributes no base; fallback unchanged"
        );

        // No `move_state` at all → no reconstructed AI → fallback unchanged.
        let d =
            make_single_enemy_fixture(py, "Tunneler", "BITE_MOVE", &[("WEAK", 1)], (9, 1)).unwrap();
        let enc = d.get_item("encounter").unwrap().unwrap();
        let e = enc.downcast::<PyList>().unwrap().get_item(0).unwrap();
        e.downcast::<PyDict>()
            .unwrap()
            .del_item("move_state")
            .unwrap();
        let state = state_from_dict(&d).expect("must build");
        assert_eq!(
            only_enemy_intent(&state),
            Some(sts2_core::Intent::Attack { damage: 12 }),
        );
    });
}

/// The tie-break must NOT be a wholesale "read the telegraph off a fresh
/// AI" — that regresses every counter-scaled move. TestSubject's MULTI_CLAW
/// grows one hit per repeat (C# `MultiClawTotalCount = BaseMultiClawCount +
/// ExtraMultiClawCount`), and the growth is NOT in the move table: the
/// static state carries the base 3, and `roll_next_enemy_intents` adds the
/// owner's PainfulStabs counter at telegraph time. Reconstruction recovers
/// that counter FROM the wire hit count, so the wire's hits are the only
/// truth available and must survive untouched — only the per-hit damage is
/// reconciled.
#[test]
fn multi_claw_escalation_survives_the_base_reconciliation() {
    Python::with_gil(|py| {
        // 2nd-form boss mid-escalation: 6 hits = base 3 + 3 repeats. Weak
        // and Strength 3 put the per-hit amount in the ambiguous band —
        // ⌊(10+3)·3/4⌋ = 9, preimages {12, 13} ⇒ bases {9, 10}.
        let d = make_single_enemy_fixture(
            py,
            "TestSubject",
            "MULTI_CLAW_MOVE",
            &[("PAINFUL_STABS", 1), ("STRENGTH", 3), ("WEAK", 1)],
            (9, 6),
        )
        .unwrap();
        let state = state_from_dict(&d).expect("must build");
        assert_eq!(
            only_enemy_intent(&state),
            Some(sts2_core::Intent::MultiAttack {
                damage: 10,
                hits: 6
            }),
            "damage reconciled to MULTI_CLAW's 10 (not the 9 the inverse \
             guesses), hit count still the wire's 6 (not the move table's 3)"
        );

        // …and the escalation counter the forward roll re-reads is intact.
        let uid = state.enemy_uids[0];
        let boss = state.creatures.iter().find(|c| c.id == uid.0).unwrap();
        let extra = boss
            .powers
            .iter()
            .filter_map(|pu| state.powers.iter().find(|p| p.id == pu.0))
            .find(|p| p.kind == sts2_core::PowerKind::PainfulStabs)
            .map(|p| p.counter);
        assert_eq!(
            extra,
            Some(3),
            "ExtraMultiClawCount stays recovered from the wire hit count"
        );
    });
}

/// ToyBox melt state must survive reconstruction: the melted relic stays in
/// the rebuilt inventory (C# never removes it from `Player.Relics`) but its
/// hooks are silent, because `CombatState.cs:130-137` filters `!IsMelted`
/// out of every listener list. Absent flags default to `false`, so wires
/// written before `RunSimulator.cs` emitted them keep round-tripping.
#[test]
fn state_from_dict_round_trips_relic_melt_state() {
    Python::with_gil(|py| {
        let build = |melted: bool| {
            let d = make_mid_turn_fixture(py).unwrap();
            let player = d
                .get_item("player")
                .unwrap()
                .unwrap()
                .downcast_into::<PyDict>()
                .unwrap();
            let relics = PyList::empty(py);
            let r = PyDict::new(py);
            r.set_item("id", "STRIKE_DUMMY").unwrap();
            if melted {
                r.set_item("is_wax", true).unwrap();
                r.set_item("is_melted", true).unwrap();
            }
            relics.append(r).unwrap();
            player.set_item("relics", relics).unwrap();
            state_from_dict(&d).expect("must build")
        };

        let live = build(false);
        assert_eq!(live.relics.len(), 1);
        assert!(!live.relics[0].is_wax, "absent `is_wax` defaults to false");
        assert!(
            !live.relics[0].is_melted,
            "absent `is_melted` defaults to false"
        );
        assert_eq!(sts2_core::hooks::iter_relic_listeners(&live).count(), 1);

        let melted = build(true);
        assert_eq!(
            melted.relics.len(),
            1,
            "a melted relic is still in the inventory"
        );
        assert!(melted.relics[0].is_wax);
        assert!(melted.relics[0].is_melted);
        assert_eq!(
            sts2_core::hooks::iter_relic_listeners(&melted).count(),
            0,
            "…but contributes no hook listener"
        );
    });
}

// ---------------------------------------------------------------------------
// Roster-order + Fabricator reconstruction (v35 mirror roster corpus)
// ---------------------------------------------------------------------------

/// Generic wire-enemy builder for the roster tests.
fn make_enemy<'py>(
    py: Python<'py>,
    kind: &str,
    uid: i64,
    hp: i32,
    is_dead: bool,
    move_state: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let e = PyDict::new(py);
    e.set_item("kind", kind)?;
    e.set_item("uid", uid)?;
    e.set_item("hp", hp)?;
    e.set_item("max_hp", hp.max(1))?;
    e.set_item("block", 0)?;
    e.set_item("powers", empty_list(py))?;
    e.set_item("intents", empty_list(py))?;
    e.set_item("intent_history", empty_list(py))?;
    e.set_item("is_dead", is_dead)?;
    e.set_item("is_escaped", false)?;
    if let Some(ms) = move_state {
        e.set_item("move_state", ms)?;
    }
    Ok(e)
}

fn roster_kinds(state: &sts2_core::CombatState) -> Vec<sts2_core::MonsterKind> {
    state
        .enemy_uids
        .iter()
        .map(|u| {
            state
                .creatures
                .iter()
                .find(|c| c.id == u.0)
                .and_then(|c| c.species)
                .expect("enemy uid resolves to a species")
        })
        .collect()
}

/// The adapter wire re-appends dead-but-reviving corpses LAST, but C#
/// keeps `Enemies` slot-sorted and never moves a corpse
/// (`SortEnemiesBySlotName`; `ShouldCreatureBeRemovedFromCombatAfterDeath`
/// is false for revivers). `TheObscuraNormal.Slots` is
/// `["illusion", "obscura"]`, so a dead-last Parafright must reconstruct
/// AHEAD of TheObscura — otherwise its revive surfaces as the v35
/// alive-order divergence (mirror (TheObscura, Parafright) vs world
/// (Parafright, TheObscura)).
#[test]
fn state_from_dict_restores_slot_order_for_obscura_reviver_corpse() {
    Python::with_gil(|py| {
        use sts2_core::MonsterKind as MK;
        let d = make_mid_turn_fixture(py).unwrap();
        let enc = PyList::empty(py);
        enc.append(make_enemy(py, "TheObscura", 2, 100, false, None).unwrap())
            .unwrap();
        enc.append(make_enemy(py, "Parafright", 3, 0, true, None).unwrap())
            .unwrap();
        d.set_item("encounter", enc).unwrap();

        let state = state_from_dict(&d).expect("obscura fixture must build");
        state.assert_invariants().expect("invariants hold");
        assert_eq!(
            roster_kinds(&state),
            vec![MK::Parafright, MK::TheObscura],
            "illusion-slot corpse must sort ahead of TheObscura"
        );
        let parafright = state
            .creatures
            .iter()
            .find(|c| c.species == Some(MK::Parafright))
            .unwrap();
        assert_eq!(parafright.current_hp, 0, "still a corpse after reordering");
    });
}

/// `DecimillipedeElite.Slots` is `["segment1", "segment2", "segment3"]`
/// with Front/Middle/Back in that order. A wire of (Middle, Back,
/// Front-dead) — alive-first + corpse-last — must reconstruct as (Front,
/// Middle, Back): after the corpse REATTACHes, C# shows (Front, Middle,
/// Back) while an unordered rebuild shows (Middle, Back, Front); after a
/// FURTHER death the corpse relative order diverges the same way (v35:
/// mirror (Middle, Back-dead, Front-dead) vs world (Middle, Front-dead,
/// Back-dead)).
#[test]
fn state_from_dict_restores_slot_order_for_decimillipede_segments() {
    Python::with_gil(|py| {
        use sts2_core::MonsterKind as MK;
        let d = make_mid_turn_fixture(py).unwrap();
        let enc = PyList::empty(py);
        enc.append(make_enemy(py, "DecimillipedeSegmentMiddle", 3, 40, false, None).unwrap())
            .unwrap();
        enc.append(make_enemy(py, "DecimillipedeSegmentBack", 4, 42, false, None).unwrap())
            .unwrap();
        enc.append(make_enemy(py, "DecimillipedeSegmentFront", 2, 0, true, None).unwrap())
            .unwrap();
        d.set_item("encounter", enc).unwrap();

        let state = state_from_dict(&d).expect("decimillipede fixture must build");
        state.assert_invariants().expect("invariants hold");
        assert_eq!(
            roster_kinds(&state),
            vec![
                MK::DecimillipedeSegmentFront,
                MK::DecimillipedeSegmentMiddle,
                MK::DecimillipedeSegmentBack,
            ],
            "segments must reconstruct in segment1/2/3 slot order"
        );
    });
}

/// A roster with any kind OUTSIDE a single reviver group keeps wire
/// order — the slot table only speaks for the encounters it was built
/// from.
#[test]
fn state_from_dict_keeps_wire_order_for_mixed_rosters() {
    Python::with_gil(|py| {
        use sts2_core::MonsterKind as MK;
        let d = make_mid_turn_fixture(py).unwrap();
        let enc = PyList::empty(py);
        enc.append(make_enemy(py, "Nibbit", 2, 48, false, None).unwrap())
            .unwrap();
        enc.append(make_enemy(py, "Parafright", 3, 0, true, None).unwrap())
            .unwrap();
        d.set_item("encounter", enc).unwrap();

        let state = state_from_dict(&d).expect("mixed fixture must build");
        assert_eq!(
            roster_kinds(&state),
            vec![MK::Nibbit, MK::Parafright],
            "mixed roster must not be reordered"
        );
    });
}

/// C# `Fabricator._lastSpawned` (Fabricator.cs:37) excludes the
/// last-spawned bot kind from the next `SpawnBot` pick; the engine keeps
/// it on the Fabricator's AI counter. The wire does not carry it —
/// recover it as the kind of the HIGHEST-uid bot on the wire (uids are
/// per-combat spawn-ordered). v35 corpus, seed 1001 round 5: the world's
/// pool was forced to [Stabbot] by a living Zapbot; the mirror's rebuilt
/// counter 0 re-widened it to a coin flip and spawned the wrong kind.
#[test]
fn state_from_dict_infers_fabricator_last_spawned_from_max_uid_bot() {
    Python::with_gil(|py| {
        use sts2_content::monsters::act3_glory::fabricator_bot_code;
        use sts2_core::MonsterKind as MK;
        let build = |bots: &[(&str, i64)]| {
            let d = make_mid_turn_fixture(py).unwrap();
            let enc = PyList::empty(py);
            for (kind, uid) in bots {
                let ms = match *kind {
                    "Stabbot" => "STAB_MOVE",
                    "Zapbot" => "ZAP",
                    _ => "NOISE",
                };
                enc.append(make_enemy(py, kind, *uid, 18, false, Some(ms)).unwrap())
                    .unwrap();
            }
            enc.append(
                make_enemy(
                    py,
                    "Fabricator",
                    2,
                    127,
                    false,
                    Some("FABRICATING_STRIKE_MOVE"),
                )
                .unwrap(),
            )
            .unwrap();
            d.set_item("encounter", enc).unwrap();
            state_from_dict(&d).expect("fabricator fixture must build")
        };

        let counter_of = |state: &sts2_core::CombatState| {
            state
                .creatures
                .iter()
                .find(|c| c.species == Some(MK::Fabricator))
                .and_then(|c| c.ai.as_ref())
                .expect("fabricator AI rebuilt from move_state")
                .counter()
        };

        // Single living Stabbot: it must be the last spawn.
        let state = build(&[("Stabbot", 5)]);
        assert_eq!(counter_of(&state), fabricator_bot_code(MK::Stabbot));

        // Stabbot(5) + Zapbot(6): the HIGHER uid wins regardless of wire
        // position.
        let state = build(&[("Zapbot", 6), ("Stabbot", 5)]);
        assert_eq!(counter_of(&state), fabricator_bot_code(MK::Zapbot));

        // No bot on the wire: unrecoverable, counter stays 0 (full pool).
        let state = build(&[]);
        assert_eq!(counter_of(&state), 0);
    });
}

/// The 2026-08-02 producers reflection-export `_lastSpawned` itself
/// (`fabricator_last_spawned`, canonical bot id). When present it must
/// win over the max-uid inference — the inference's documented gap is a
/// most-recent bot that already died and left the wire.
#[test]
fn state_from_dict_prefers_wire_fabricator_last_spawned_over_inference() {
    Python::with_gil(|py| {
        use sts2_content::monsters::act3_glory::fabricator_bot_code;
        use sts2_core::MonsterKind as MK;
        let d = make_mid_turn_fixture(py).unwrap();
        let enc = PyList::empty(py);
        // A living Stabbot is the max-uid bot — the inference would say
        // Stabbot. The wire says the real last spawn was a (dead, off-wire)
        // Zapbot.
        enc.append(make_enemy(py, "Stabbot", 5, 18, false, Some("STAB_MOVE")).unwrap())
            .unwrap();
        let fab = make_enemy(
            py,
            "Fabricator",
            2,
            127,
            false,
            Some("FABRICATING_STRIKE_MOVE"),
        )
        .unwrap();
        fab.set_item("fabricator_last_spawned", "ZAPBOT").unwrap();
        enc.append(fab).unwrap();
        d.set_item("encounter", enc).unwrap();

        let state = state_from_dict(&d).expect("fabricator fixture must build");
        let counter = state
            .creatures
            .iter()
            .find(|c| c.species == Some(MK::Fabricator))
            .and_then(|c| c.ai.as_ref())
            .expect("fabricator AI rebuilt from move_state")
            .counter();
        assert_eq!(
            counter,
            fabricator_bot_code(MK::Zapbot),
            "wire fabricator_last_spawned must beat the max-uid inference"
        );
    });
}

/// KnowledgeDemon `_curseOfKnowledgeCounter` (0..3 — the Disintegration
/// 6/7/8 curse index) now rides the wire; reconstruction must seed the
/// demon's AI counter from it instead of rebuilding at 0 (the documented
/// DISINTEGRATION off-by-one: a mid-fight mirror re-offered curse set 0).
#[test]
fn state_from_dict_seeds_knowledge_demon_curse_counter_from_wire() {
    Python::with_gil(|py| {
        use sts2_core::MonsterKind as MK;
        let d = make_mid_turn_fixture(py).unwrap();
        let enc = PyList::empty(py);
        let demon = make_enemy(py, "KnowledgeDemon", 2, 300, false, Some("SLAP")).unwrap();
        demon.set_item("curse_of_knowledge_counter", 2).unwrap();
        enc.append(demon).unwrap();
        d.set_item("encounter", enc).unwrap();

        let state = state_from_dict(&d).expect("knowledge demon fixture must build");
        let counter = state
            .creatures
            .iter()
            .find(|c| c.species == Some(MK::KnowledgeDemon))
            .and_then(|c| c.ai.as_ref())
            .expect("knowledge demon AI rebuilt from move_state")
            .counter();
        assert_eq!(counter, 2, "wire curse counter must seed the AI");
    });
}

/// `FabricatorNormal.Slots` is `["bot1", "bot2", "fabricator", "bot3",
/// "bot4"]`: the wire's alive order is C# `Enemies` slot order, so bots
/// before the Fabricator carry slots 0/1 and bots after carry 3/4.
/// `fabricator_spawn_bot` keys the next spawn's position off
/// `enemy_slot()` — a slot-less rebuild front-inserts the new bot where
/// the game refills the first genuinely free slot (v35, seed 1001
/// round 4: mirror (NEW, old, Fabricator) vs world (old, NEW,
/// Fabricator)).
#[test]
fn state_from_dict_reconstructs_fabricator_bot_slots_from_wire_order() {
    Python::with_gil(|py| {
        use sts2_core::MonsterKind as MK;
        let d = make_mid_turn_fixture(py).unwrap();
        let enc = PyList::empty(py);
        enc.append(make_enemy(py, "Noisebot", 3, 22, false, Some("NOISE")).unwrap())
            .unwrap();
        enc.append(make_enemy(py, "Stabbot", 4, 18, false, Some("STAB_MOVE")).unwrap())
            .unwrap();
        enc.append(make_enemy(py, "Fabricator", 2, 127, false, Some("FABRICATE_MOVE")).unwrap())
            .unwrap();
        enc.append(make_enemy(py, "Zapbot", 5, 21, false, Some("ZAP")).unwrap())
            .unwrap();
        d.set_item("encounter", enc).unwrap();

        let state = state_from_dict(&d).expect("fabricator slot fixture must build");
        let slot_of = |kind: MK| {
            let uid = state
                .enemy_uids
                .iter()
                .copied()
                .find(|u| {
                    state
                        .creatures
                        .iter()
                        .find(|c| c.id == u.0)
                        .and_then(|c| c.species)
                        == Some(kind)
                })
                .expect("kind present");
            state.enemy_slot(uid)
        };
        assert_eq!(slot_of(MK::Noisebot), Some(0), "first pre-fab bot = bot1");
        assert_eq!(slot_of(MK::Stabbot), Some(1), "second pre-fab bot = bot2");
        assert_eq!(slot_of(MK::Zapbot), Some(3), "first post-fab bot = bot3");
        assert_eq!(
            slot_of(MK::Fabricator),
            None,
            "the Fabricator itself keeps its fixed middle-slot key"
        );
    });
}

/// The live bridge's card `cost` is the RESOLVED display and it ships no
/// `base_cost`. For an X-cost card (C# `HasEnergyCostX`, registry cost -1 —
/// Cascade, Whirlwind) that display is the CURRENT ENERGY, not an instance
/// cost: adopting it froze "X = energy at snapshot time" into `Card.cost`,
/// and the mirror then re-emitted a stale X for the rest of the fight (137
/// of the 421 live1-unexplained mirror records were CASCADE alone, always
/// one action behind the world's energy pool). The wire cost must be
/// ignored for registry-X cards on the base_cost-less dialect only —
/// engine/DLL wires carry `base_cost: -1` and are unchanged.
#[test]
fn x_cost_card_ignores_live_display_cost() {
    Python::with_gil(|py| {
        let d = make_mid_turn_fixture(py).unwrap();
        let deck_any = d.get_item("deck").unwrap().unwrap();
        let deck = deck_any.downcast::<PyDict>().unwrap();
        let hand_any = deck.get_item("hand").unwrap().unwrap();
        let hand = hand_any.downcast::<PyList>().unwrap();

        // Live dialect: resolved display cost only (X painted as energy=2).
        let cascade = make_card(py, "CASCADE", 90, false).unwrap();
        cascade.set_item("cost_override", 2).unwrap();
        hand.append(&cascade).unwrap();

        // Control 1: engine dialect carries base_cost -1 — first arm, as before.
        let cascade_native = make_card(py, "CASCADE", 91, false).unwrap();
        cascade_native.set_item("cost_override", 2).unwrap();
        cascade_native.set_item("base_cost", -1).unwrap();
        hand.append(&cascade_native).unwrap();

        // Control 2: a fixed-cost card still adopts the resolved cost when
        // base_cost is absent (pre-existing live-dialect behavior).
        let strike = make_card(py, "STRIKE_IRONCLAD", 92, false).unwrap();
        strike.set_item("cost_override", 0).unwrap();
        hand.append(&strike).unwrap();

        let state = state_from_dict(&d).expect("must build");
        let costs: Vec<(sts2_core::cards::CardId, i8)> = state
            .cards
            .iter()
            .filter(|c| {
                matches!(
                    c.card_id,
                    sts2_core::cards::CardId::Cascade | sts2_core::cards::CardId::StrikeIronclad
                )
            })
            .map(|c| (c.card_id, c.cost))
            .collect();
        let cascades: Vec<i8> = costs
            .iter()
            .filter(|(id, _)| *id == sts2_core::cards::CardId::Cascade)
            .map(|&(_, c)| c)
            .collect();
        assert_eq!(
            cascades,
            vec![-1, -1],
            "X-cost Cascade must keep registry cost -1 on both dialects"
        );
        assert!(
            costs
                .iter()
                .any(|&(id, c)| id == sts2_core::cards::CardId::StrikeIronclad && c == 0),
            "fixed-cost card still adopts the live resolved cost (Strike @0)"
        );
    });
}
