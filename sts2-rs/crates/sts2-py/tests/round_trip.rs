//! Property test for `state_to_dict` and `state_from_dict` round-trip
//! — hint-T3 (bead `sts2-rs-wfjn`).
//!
//! Drives 1000 seeded mid-turn fixtures through:
//!
//!   `state` → `state_to_dict` → `state_from_dict` → `state'`
//!
//! and asserts `state_equiv(state, state')` holds.
//!
//! **Intentionally-dropped fields** (not round-tripped through the wire
//! format, and therefore explicitly ignored by `state_equiv`):
//!
//! * `rng` — `Xoshiro256PlusPlus` cursor state. The wire schema's
//!   `rng.seed` is reseed-only metadata; the in-process cursor is
//!   opaque (no `seed_get`). After round-trip the cursor is reseeded
//!   from `rng_seed.wrapping_add(turn)` (see `state_from_dict_impl`).
//! * `combat_log` (`TurnEvents`) — append-only telemetry buffer not
//!   serialised in the wire format.
//! * Hook listener closures — the relic / power event buses populate
//!   themselves lazily on the next `apply()` from the relic/power
//!   kinds (which DO round-trip).
//! * `intent_history` (per-creature) — the Rust-side internal shape is
//!   private; the wire format only carries an i32-encoded summary
//!   which we re-emit as an empty list. `state_from_dict_impl` always
//!   defaults to `IntentHistory::default()`.
//! * `pending_choice`, `pending_play`, `active_attack` — all
//!   documented as `None` at hint-time and rejected by
//!   `parse_state_dict` when non-`None`. The round-trip is closed
//!   over fixtures where these are absent.
//! * `combat_id` — wire-only metadata for matching adapter calls;
//!   not stored on `CoreCombatState`.
//! * `monster.uid` — the rebuild path allocates fresh `CreatureUid`s
//!   sequentially; we compare structural enemy metadata (`kind` / `hp`
//!   / `max_hp` / `block`) instead.
//! * Card `uid` / `stable_idx` — same story: rebuild path allocates
//!   fresh `CardUid`s. We compare structural pile contents (`card_id`
//!   ordering + counts + `upgraded` flag).
//! * `master_deck.last_added_card`, `combat_vars`, `enchantments`,
//!   `allies`, `potions` arena — not in the wire schema; the rebuild
//!   path leaves them at default.
//!
//! Two entries left this list when the persistent combat mirror's oracle
//! showed what they cost: `play` (the mid-resolution pile) is now carried
//! as `$.deck.play_pile`, and `tags` is re-derived from the card id on
//! rebuild — without it every reconstructed deck read as untagged and
//! PerfectedStrike counted zero Strikes.

#![allow(clippy::missing_panics_doc)]

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use sts2_core::cards::CardId;
use sts2_core::CombatState as CoreCombatState;
use sts2_core::Intent;
use sts2_rs::state_dict::{state_from_dict, state_to_dict};

// ---------------------------------------------------------------------------
// Seeded fixture builder
// ---------------------------------------------------------------------------
//
// `build_seeded_state(seed, ascension)` is the test entry point referenced
// in the bead spec. It builds a valid mid-turn Ironclad-vs-Cultist
// fixture parameterised by `seed` — the seed advances the RNG and is
// echoed into `rng.seed` so we get 1000 distinct dict payloads. The
// state itself is constructed via `state_from_dict` (i.e. through the
// same path the round-trip uses on the inverse leg), guaranteeing the
// source state is one the wire format CAN represent.

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

fn make_seeded_fixture(py: Python<'_>, seed: u64, ascension: u8) -> PyResult<Bound<'_, PyDict>> {
    // Slight per-seed variation so we exercise more than one structural
    // shape. The hand/draw distribution is permuted by `seed % 5`; HP
    // varies in a 20-point band around 70. The Cultist HP varies in a
    // 10-point band so monster scalars also vary.
    let phase_idx = (seed % 5) as usize;

    // Card layout permutations — 5 patterns that all use the
    // Ironclad starter pool (Strikes, Defends, Bash). Picked to vary
    // the pile sizes / order while staying valid.
    let layouts: [(&[&str], &[&str], &[&str]); 5] = [
        (
            // hand
            &["STRIKE_IRONCLAD", "DEFEND_IRONCLAD", "BASH"],
            // draw
            &["STRIKE_IRONCLAD", "DEFEND_IRONCLAD"],
            // discard
            &["STRIKE_IRONCLAD"],
        ),
        (
            &[
                "STRIKE_IRONCLAD",
                "STRIKE_IRONCLAD",
                "DEFEND_IRONCLAD",
                "DEFEND_IRONCLAD",
            ],
            &["STRIKE_IRONCLAD"],
            &[],
        ),
        (
            &["BASH", "STRIKE_IRONCLAD"],
            &["DEFEND_IRONCLAD", "DEFEND_IRONCLAD", "STRIKE_IRONCLAD"],
            &["DEFEND_IRONCLAD"],
        ),
        (
            &["DEFEND_IRONCLAD", "DEFEND_IRONCLAD"],
            &["BASH", "STRIKE_IRONCLAD", "STRIKE_IRONCLAD"],
            &["STRIKE_IRONCLAD", "DEFEND_IRONCLAD"],
        ),
        (
            &["STRIKE_IRONCLAD", "STRIKE_IRONCLAD", "STRIKE_IRONCLAD"],
            &["DEFEND_IRONCLAD"],
            &["BASH"],
        ),
    ];
    let (hand_ids, draw_ids, discard_ids) = layouts[phase_idx];

    let mut next_idx = 1_i32;
    let hand = PyList::empty(py);
    for id in hand_ids {
        hand.append(make_card(py, id, next_idx, false)?)?;
        next_idx += 1;
    }
    let draw = PyList::empty(py);
    for id in draw_ids {
        draw.append(make_card(py, id, next_idx, false)?)?;
        next_idx += 1;
    }
    let discard = PyList::empty(py);
    for id in discard_ids {
        discard.append(make_card(py, id, next_idx, false)?)?;
        next_idx += 1;
    }

    let deck = PyDict::new(py);
    deck.set_item("draw_pile", draw)?;
    deck.set_item("discard_pile", discard)?;
    deck.set_item("hand", hand)?;
    deck.set_item("exhaust_pile", PyList::empty(py))?;

    // Player vitals vary with seed.
    let player = PyDict::new(py);
    player.set_item("class", "ironclad")?;
    let max_hp = 70 + i32::try_from(seed % 20).unwrap_or(0);
    let hp = max_hp - i32::try_from(seed % 11).unwrap_or(0);
    player.set_item("hp", hp)?;
    player.set_item("max_hp", max_hp)?;
    player.set_item("block", i32::try_from(seed % 7).unwrap_or(0))?;
    let energy = i32::try_from(seed % 4).unwrap_or(0);
    player.set_item("energy", energy)?;
    player.set_item("max_energy", 3)?;
    player.set_item("energy_regen_per_turn", 3)?;

    let relics = PyList::empty(py);
    let r = PyDict::new(py);
    r.set_item("id", "BURNING_BLOOD")?;
    relics.append(r)?;
    player.set_item("relics", relics)?;

    // Powers vary: even seeds get Strength, odd seeds get no power.
    let powers = PyList::empty(py);
    if seed.is_multiple_of(2) {
        let p = PyDict::new(py);
        p.set_item("name", "STRENGTH")?;
        let amt = i32::try_from((seed / 2) % 5).unwrap_or(0) + 1;
        p.set_item("amount", amt)?;
        powers.append(p)?;
    }
    player.set_item("powers", powers)?;
    player.set_item("potions", PyList::empty(py))?;

    // Cultist with HP varying.
    let enc = PyList::empty(py);
    let cult = PyDict::new(py);
    cult.set_item("kind", "Nibbit")?;
    cult.set_item("uid", 1_i64)?;
    let m_max = 48_i32;
    let m_hp = m_max - i32::try_from(seed % 10).unwrap_or(0);
    cult.set_item("hp", m_hp)?;
    cult.set_item("max_hp", m_max)?;
    cult.set_item("block", i32::try_from(seed % 3).unwrap_or(0))?;
    cult.set_item("powers", PyList::empty(py))?;
    cult.set_item("intents", PyList::empty(py))?;
    cult.set_item("intent_history", PyList::empty(py))?;
    cult.set_item("is_dead", false)?;
    cult.set_item("is_escaped", false)?;
    enc.append(cult)?;

    let d = PyDict::new(py);
    d.set_item("wire_version", 1)?;
    d.set_item("phase", "PlayerPlayPhase")?;
    d.set_item("round", 1)?;
    d.set_item("side", "Player")?;
    d.set_item("is_elite", seed.is_multiple_of(7))?;
    d.set_item("turn", 1)?;
    d.set_item("combat_id", 0_i64)?;
    d.set_item("player", player)?;
    d.set_item("deck", deck)?;
    d.set_item("encounter", enc)?;

    let rng = PyDict::new(py);
    rng.set_item("seed", seed)?;
    rng.set_item("ascension", ascension)?;
    d.set_item("rng", rng)?;
    Ok(d)
}

fn build_seeded_state(py: Python<'_>, seed: u64, ascension: u8) -> PyResult<CoreCombatState> {
    let d = make_seeded_fixture(py, seed, ascension)?;
    state_from_dict(&d)
}

// ---------------------------------------------------------------------------
// state_equiv — round-trip equality predicate
// ---------------------------------------------------------------------------

/// Compare two `CoreCombatState`s on the fields that round-trip through
/// the wire format. See the module-level comment for the list of fields
/// intentionally ignored.
///
/// Returns `Ok(())` when the two states are wire-equivalent, or
/// `Err(reason)` describing the first divergence.
#[allow(clippy::too_many_lines)]
fn state_equiv(a: &CoreCombatState, b: &CoreCombatState) -> Result<(), String> {
    // Scalars.
    if a.ascension != b.ascension {
        return Err(format!("ascension: {} != {}", a.ascension, b.ascension));
    }
    if a.is_elite != b.is_elite {
        return Err(format!("is_elite: {} != {}", a.is_elite, b.is_elite));
    }
    if a.side != b.side {
        return Err(format!("side: {:?} != {:?}", a.side, b.side));
    }
    // Phase: parse path is lossy across grouped variants (Player*-phases
    // all collapse to PlayerPlayPhase on the way out), so compare the
    // emitted wire string instead.
    let a_phase = combat_phase_str(a.phase);
    let b_phase = combat_phase_str(b.phase);
    if a_phase != b_phase {
        return Err(format!("phase wire-str: {a_phase} != {b_phase}"));
    }

    // Player scalars.
    if a.player.current_hp != b.player.current_hp {
        return Err(format!(
            "player.current_hp: {} != {}",
            a.player.current_hp, b.player.current_hp
        ));
    }
    if a.player.max_hp != b.player.max_hp {
        return Err(format!(
            "player.max_hp: {} != {}",
            a.player.max_hp, b.player.max_hp
        ));
    }
    if a.player.block != b.player.block {
        return Err(format!(
            "player.block: {} != {}",
            a.player.block, b.player.block
        ));
    }
    if a.player.energy != b.player.energy {
        return Err(format!(
            "player.energy: {} != {}",
            a.player.energy, b.player.energy
        ));
    }
    if a.player.max_energy != b.player.max_energy {
        return Err(format!(
            "player.max_energy: {} != {}",
            a.player.max_energy, b.player.max_energy
        ));
    }
    if a.player.character != b.player.character {
        return Err(format!(
            "player.character: {:?} != {:?}",
            a.player.character, b.player.character
        ));
    }
    if a.player.cards_played_this_turn != b.player.cards_played_this_turn {
        return Err(format!(
            "player.cards_played_this_turn: {:?} != {:?}",
            a.player.cards_played_this_turn, b.player.cards_played_this_turn
        ));
    }

    // Relics: compare (kind, counter) sequences.
    let ar: Vec<_> = a.relics.iter().map(|r| (r.kind, r.counter)).collect();
    let br: Vec<_> = b.relics.iter().map(|r| (r.kind, r.counter)).collect();
    if ar != br {
        return Err(format!("relics: {ar:?} != {br:?}"));
    }

    // Potions: compare PotionKind sequences.
    let ap: Vec<_> = a.player.potions.iter().copied().collect();
    let bp: Vec<_> = b.player.potions.iter().copied().collect();
    if ap != bp {
        return Err(format!("potions: {ap:?} != {bp:?}"));
    }

    // Player powers: compare (kind, amount) sequences via the powers arena.
    let pa: Vec<_> = a
        .player
        .powers
        .iter()
        .map(|uid| {
            a.powers
                .iter()
                .find(|p| p.id == uid.0)
                .map(|p| (p.kind, p.amount))
        })
        .collect();
    let pb: Vec<_> = b
        .player
        .powers
        .iter()
        .map(|uid| {
            b.powers
                .iter()
                .find(|p| p.id == uid.0)
                .map(|p| (p.kind, p.amount))
        })
        .collect();
    if pa != pb {
        return Err(format!("player.powers: {pa:?} != {pb:?}"));
    }

    // Enemy structural metadata.
    if a.enemy_uids.len() != b.enemy_uids.len() {
        return Err(format!(
            "enemy_uids.len: {} != {}",
            a.enemy_uids.len(),
            b.enemy_uids.len()
        ));
    }
    for i in 0..a.enemy_uids.len() {
        let ca = a
            .creatures
            .iter()
            .find(|c| c.id == a.enemy_uids[i].0)
            .ok_or_else(|| format!("a.enemy[{i}] uid not in creatures"))?;
        let cb = b
            .creatures
            .iter()
            .find(|c| c.id == b.enemy_uids[i].0)
            .ok_or_else(|| format!("b.enemy[{i}] uid not in creatures"))?;
        if ca.species != cb.species {
            return Err(format!(
                "enemy[{i}].species: {:?} != {:?}",
                ca.species, cb.species
            ));
        }
        if ca.current_hp != cb.current_hp {
            return Err(format!(
                "enemy[{i}].hp: {} != {}",
                ca.current_hp, cb.current_hp
            ));
        }
        if ca.max_hp != cb.max_hp {
            return Err(format!("enemy[{i}].max_hp: {} != {}", ca.max_hp, cb.max_hp));
        }
        if ca.block != cb.block {
            return Err(format!("enemy[{i}].block: {} != {}", ca.block, cb.block));
        }
        // Enemy powers — same (kind, amount) check as the player.
        let ea: Vec<_> = ca
            .powers
            .iter()
            .map(|uid| {
                a.powers
                    .iter()
                    .find(|p| p.id == uid.0)
                    .map(|p| (p.kind, p.amount))
            })
            .collect();
        let eb: Vec<_> = cb
            .powers
            .iter()
            .map(|uid| {
                b.powers
                    .iter()
                    .find(|p| p.id == uid.0)
                    .map(|p| (p.kind, p.amount))
            })
            .collect();
        if ea != eb {
            return Err(format!("enemy[{i}].powers: {ea:?} != {eb:?}"));
        }
    }

    // Piles — compare structural shapes (card_id sequence + upgraded
    // flag). uids are NOT compared because the rebuild path allocates
    // fresh ones.
    pile_equiv("hand", &a.player.hand, &a.cards, &b.player.hand, &b.cards)?;
    pile_equiv("draw", &a.player.draw, &a.cards, &b.player.draw, &b.cards)?;
    pile_equiv(
        "discard",
        &a.player.discard,
        &a.cards,
        &b.player.discard,
        &b.cards,
    )?;
    pile_equiv(
        "exhaust",
        &a.player.exhaust,
        &a.cards,
        &b.player.exhaust,
        &b.cards,
    )?;
    pile_equiv("play", &a.player.play, &a.cards, &b.player.play, &b.cards)?;

    // Draw-pile knowledge. Pins travel as draw-pile *indices*, so
    // compare positions rather than uids (which the rebuild reallocates).
    // Stale pins are dropped on the way out by design — compare only
    // pins whose card is actually in `draw`, which is all a consumer
    // honors anyway.
    pins_equiv(
        "pinned_top",
        &a.player.pinned_top,
        &a.player.draw,
        &b.player.pinned_top,
        &b.player.draw,
    )?;
    pins_equiv(
        "pinned_bottom",
        &a.player.pinned_bottom,
        &a.player.draw,
        &b.player.pinned_bottom,
        &b.player.draw,
    )?;
    if a.player.deferred_draw_order != b.player.deferred_draw_order {
        return Err(format!(
            "deferred_draw_order: {} != {}",
            a.player.deferred_draw_order, b.player.deferred_draw_order
        ));
    }

    Ok(())
}

fn pins_equiv(
    name: &str,
    a_pins: &[sts2_core::CardUid],
    a_draw: &[sts2_core::CardUid],
    b_pins: &[sts2_core::CardUid],
    b_draw: &[sts2_core::CardUid],
) -> Result<(), String> {
    let positions = |pins: &[sts2_core::CardUid], draw: &[sts2_core::CardUid]| -> Vec<usize> {
        pins.iter()
            .filter_map(|u| draw.iter().position(|d| d == u))
            .collect()
    };
    let a_pos = positions(a_pins, a_draw);
    let b_pos = positions(b_pins, b_draw);
    if a_pos != b_pos {
        return Err(format!("{name} differs: {a_pos:?} != {b_pos:?}"));
    }
    Ok(())
}

fn pile_equiv(
    name: &str,
    a_uids: &[sts2_core::CardUid],
    a_cards: &[sts2_core::cards::Card],
    b_uids: &[sts2_core::CardUid],
    b_cards: &[sts2_core::cards::Card],
) -> Result<(), String> {
    let a_sig: Vec<(CardId, bool)> = a_uids
        .iter()
        .map(|u| {
            a_cards
                .iter()
                .find(|c| c.uid == *u)
                .map(|c| (c.card_id, c.upgraded))
                .ok_or_else(|| format!("{name} pile uid {} not in arena (a)", u.get()))
        })
        .collect::<Result<_, _>>()?;
    let b_sig: Vec<(CardId, bool)> = b_uids
        .iter()
        .map(|u| {
            b_cards
                .iter()
                .find(|c| c.uid == *u)
                .map(|c| (c.card_id, c.upgraded))
                .ok_or_else(|| format!("{name} pile uid {} not in arena (b)", u.get()))
        })
        .collect::<Result<_, _>>()?;
    if a_sig != b_sig {
        return Err(format!("pile {name} differs: {a_sig:?} != {b_sig:?}"));
    }
    Ok(())
}

fn combat_phase_str(p: sts2_core::CombatPhase) -> &'static str {
    use sts2_core::CombatPhase as P;
    match p {
        P::Setup => "Setup",
        P::PlayerTurnStart => "PlayerDraw",
        P::PlayerPlayPhase | P::PlayerEndTurnPhaseOne | P::PlayerEndTurnPhaseTwo => {
            "PlayerPlayPhase"
        }
        P::EnemyTurnStart | P::EnemyTurnExec | P::EnemyTurnEnd => "EnemyPhase",
        P::Resolved { player_won: true } => "Victory",
        P::Resolved { player_won: false } => "Defeat",
    }
}

// ---------------------------------------------------------------------------
// Property test
// ---------------------------------------------------------------------------

#[test]
fn to_dict_from_dict_round_trip_1000_seeds() {
    Python::with_gil(|py| {
        let mut failures: Vec<(u64, String)> = Vec::new();
        for seed in 0..1000_u64 {
            let s1 = match build_seeded_state(py, seed, 0) {
                Ok(s) => s,
                Err(e) => {
                    failures.push((seed, format!("build_seeded_state failed: {e}")));
                    if failures.len() > 5 {
                        break;
                    }
                    continue;
                }
            };
            let dict = match state_to_dict(py, &s1) {
                Ok(d) => d,
                Err(e) => {
                    failures.push((seed, format!("state_to_dict failed: {e}")));
                    if failures.len() > 5 {
                        break;
                    }
                    continue;
                }
            };
            let s2 = match state_from_dict(&dict) {
                Ok(s) => s,
                Err(e) => {
                    failures.push((seed, format!("state_from_dict failed: {e}")));
                    if failures.len() > 5 {
                        break;
                    }
                    continue;
                }
            };
            if let Err(why) = state_equiv(&s1, &s2) {
                failures.push((seed, why));
                if failures.len() > 5 {
                    break;
                }
            }
        }
        assert!(
            failures.is_empty(),
            "round-trip failed for {} seeds; first failures: {:#?}",
            failures.len(),
            failures
        );
    });
}

#[test]
fn state_to_dict_marks_zero_hp_monster_dead() {
    // Regression for sts2-rs-vvi6g (o1mwv FOGMOG parity): `state_to_dict`
    // used to hardcode `is_dead = false` for every monster, so a creature
    // at 0 HP (e.g. the EyeWithTeeth killed by the player) serialized as
    // alive. C# `Creature.IsAlive` is `CurrentHp > 0`, and the parity
    // harness derives `alive = !is_dead`, so the field must reflect HP.
    Python::with_gil(|py| {
        let mut state = build_seeded_state(py, 1, 0).expect("seed=1 must build");
        let enemy_uid = *state.enemy_uids.first().expect("fixture has one enemy");
        let idx = state
            .creatures
            .iter()
            .position(|c| c.id == enemy_uid.0)
            .expect("enemy uid resolves to a creature");

        // Alive monster (hp > 0) emits is_dead=false.
        let dict_alive = state_to_dict(py, &state).expect("to_dict (alive)");
        assert!(
            !monster0_is_dead(&dict_alive),
            "hp>0 monster must serialize is_dead=false"
        );

        // Kill it: hp == 0 must emit is_dead=true.
        state.creatures[idx].current_hp = 0;
        let dict_dead = state_to_dict(py, &state).expect("to_dict (dead)");
        assert!(
            monster0_is_dead(&dict_dead),
            "0-hp monster must serialize is_dead=true"
        );
    });
}

/// Extract `encounter[0].is_dead` from a `state_to_dict` payload.
fn monster0_is_dead(d: &Bound<'_, PyDict>) -> bool {
    let enc = d
        .get_item("encounter")
        .unwrap()
        .expect("encounter present")
        .downcast::<PyList>()
        .unwrap()
        .clone();
    let m0 = enc
        .get_item(0)
        .unwrap()
        .downcast::<PyDict>()
        .unwrap()
        .clone();
    m0.get_item("is_dead")
        .unwrap()
        .expect("is_dead present")
        .extract()
        .unwrap()
}

#[test]
fn round_trip_preserves_cards_played_this_turn() {
    // Regression for the 2026-07-12 rung-3 stall vs Ceremonial Beast:
    // `state_to_dict` dropped `player.cards_played_this_turn`, so a
    // round-tripped mid-turn state under Ringing (one card per turn)
    // offered plays the source engine silently rejected — a deterministic
    // policy then repeated the ignored play forever. The buckets must
    // survive the round-trip; a dict without the key must parse to zeros.
    Python::with_gil(|py| {
        let mut s1 = build_seeded_state(py, 3, 0).expect("seed=3 must build");
        s1.player.cards_played_this_turn = [2, 1, 0, 0, 0];

        let dict = state_to_dict(py, &s1).expect("to_dict must succeed");
        let s2 = state_from_dict(&dict).expect("from_dict on emitted dict must succeed");
        assert_eq!(
            s2.player.cards_played_this_turn,
            [2, 1, 0, 0, 0],
            "mid-turn play counters must survive the round-trip"
        );

        // Absent key (all-zero buckets are omitted on emit) → zeros.
        s1.player.cards_played_this_turn = [0; 5];
        let dict = state_to_dict(py, &s1).expect("to_dict must succeed");
        let player = dict
            .get_item("player")
            .unwrap()
            .expect("player present")
            .downcast::<PyDict>()
            .unwrap()
            .clone();
        assert!(
            player.get_item("cards_played_this_turn").unwrap().is_none(),
            "all-zero counters must be omitted for wire-shape stability"
        );
        let s2 = state_from_dict(&dict).expect("from_dict without the key must succeed");
        assert_eq!(s2.player.cards_played_this_turn, [0; 5]);
    });
}

#[test]
fn round_trip_preserves_draw_pile_knowledge() {
    // The persistent-combat-mirror prerequisite: `pinned_top` /
    // `pinned_bottom` / `deferred_draw_order` were absent from the wire
    // schema entirely, so every reconstruction silently discarded the
    // player's OWN knowledge of the draw pile — after a Headbutt the
    // search forgot what it had just put on top. Pins travel as draw-pile
    // indices because the rebuild reallocates every `CardUid`.
    Python::with_gil(|py| {
        // seed=3 → layout 3, a 3-card draw pile (BASH, STRIKE, STRIKE).
        let mut s1 = build_seeded_state(py, 3, 0).expect("seed=3 must build");
        assert!(
            s1.player.draw.len() >= 3,
            "fixture must have a draw pile to pin into"
        );
        let top_a = s1.player.draw[2];
        let top_b = s1.player.draw[0];
        let bottom = s1.player.draw[1];
        // LIFO: the last pin_top wins the top slot.
        sts2_core::cards::pin_top(&mut s1.player, top_a);
        sts2_core::cards::pin_top(&mut s1.player, top_b);
        sts2_core::cards::pin_bottom(&mut s1.player, bottom);
        // NB: no `deferred_draw_order` here. Pins are orthogonal to that flag,
        // and this fixture's pile is a materialized shuffle — asserting the
        // flag over it is exactly the incoherent state the wire now rejects.
        // The flag has its own tests below.

        let dict = state_to_dict(py, &s1).expect("to_dict must succeed");
        let deck = dict
            .get_item("deck")
            .unwrap()
            .expect("deck present")
            .downcast::<PyDict>()
            .unwrap()
            .clone();
        let top_idx: Vec<usize> = deck
            .get_item("pinned_top")
            .unwrap()
            .expect("pinned_top emitted")
            .extract()
            .expect("pinned_top is a list of indices");
        assert_eq!(
            top_idx,
            vec![0, 2],
            "pins emit as draw-pile indices in pin order"
        );

        let s2 = state_from_dict(&dict).expect("from_dict on emitted dict must succeed");
        state_equiv(&s1, &s2).expect("pinned state must round-trip");
        // Spell the positions out rather than trusting only the predicate.
        assert_eq!(
            s2.player
                .pinned_top
                .iter()
                .map(|u| s2.player.draw.iter().position(|d| d == u).unwrap())
                .collect::<Vec<_>>(),
            vec![0, 2]
        );
        assert_eq!(
            s2.player
                .pinned_bottom
                .iter()
                .map(|u| s2.player.draw.iter().position(|d| d == u).unwrap())
                .collect::<Vec<_>>(),
            vec![1]
        );
        // Unpinned states must not emit the keys at all (wire-shape
        // stability for every producer that doesn't know about pins).
        let s3 = build_seeded_state(py, 3, 0).expect("seed=3 must build");
        let dict = state_to_dict(py, &s3).expect("to_dict must succeed");
        let deck = dict
            .get_item("deck")
            .unwrap()
            .expect("deck present")
            .downcast::<PyDict>()
            .unwrap()
            .clone();
        for key in ["pinned_top", "pinned_bottom", "deferred_draw_order"] {
            assert!(
                deck.get_item(key).unwrap().is_none(),
                "{key} must be omitted when there is nothing to say"
            );
        }
        let s4 = state_from_dict(&dict).expect("from_dict without the keys must succeed");
        assert!(s4.player.pinned_top.is_empty());
        assert!(s4.player.pinned_bottom.is_empty());
        assert!(!s4.player.deferred_draw_order);
    });
}

/// `deferred_draw_order` is an ASSERTION about the pile, not a preference.
///
/// Its own doc-comment says "no env-visible engine path ever sets it" —
/// inside the engine it is only ever set immediately after canonicalizing
/// (`blindfold::belief_root`), so the invariant held by construction.
/// Round-tripping it through the wire made it settable by any producer, and
/// a producer that asserts it over a materialized shuffle gets no error:
/// `pop_next_card` simply consumes RNG differently, so the search runs
/// against a pile whose order means something other than what the flag says.
#[test]
fn deferred_draw_order_round_trips_only_over_a_canonical_pile() {
    Python::with_gil(|py| {
        let mut s1 = build_seeded_state(py, 3, 0).expect("seed=3 must build");
        assert!(s1.player.draw.len() >= 3, "fixture needs a draw pile");

        // The one engine path that sets the flag: canonicalize, then assert.
        sts2_core::blindfold::belief_root(&mut s1, 0xC0FFEE);
        assert!(s1.player.deferred_draw_order);

        let dict = state_to_dict(py, &s1).expect("to_dict must succeed");
        let s2 = state_from_dict(&dict).expect("a canonical belief pile must round-trip");
        assert!(s2.player.deferred_draw_order);
        state_equiv(&s1, &s2).expect("belief state must round-trip");
    });
}

/// The half of the assertion the wire CAN check: pin/segment coherence.
///
/// Canonical form is `[pinned_top …, sorted middle …, pinned_bottom …]`, and
/// the wire carries the pins as indices into `draw_pile`, so a producer that
/// declares a pin the listed order contradicts is caught on ingest. Here the
/// LAST card is declared `pinned_top`; canonical form would put it first.
#[test]
fn deferred_draw_order_over_an_incoherent_pin_layout_fails_loud() {
    Python::with_gil(|py| {
        let mut s = build_seeded_state(py, 3, 0).expect("seed=3 must build");
        assert!(s.player.draw.len() >= 3, "fixture needs a draw pile");
        sts2_core::blindfold::belief_root(&mut s, 0xC0FFEE);
        assert!(s.player.deferred_draw_order);

        let tail = *s.player.draw.last().expect("non-empty draw pile");
        s.player.pinned_top.push(tail);

        let dict = state_to_dict(py, &s).expect("to_dict must succeed");
        let err = state_from_dict(&dict)
            .expect_err("a pin the pile order contradicts must not be accepted");
        let msg = format!("{err}");
        assert!(
            msg.contains("deferred_draw_order"),
            "error must name the offending field, got: {msg}"
        );
    });
}

/// The half the wire CANNOT check — pinned by a test so the limit is a
/// measured fact, not an assumption.
///
/// "Canonical" means uid-ascending, and uid order is *card creation* order:
/// a real engine property. The wire carries no uids — `alloc_card_into`
/// mints them positionally as it walks `draw_pile`, so whatever order a
/// producer lists comes back uid-ascending by construction. An unpinned
/// materialized shuffle is therefore INDISTINGUISHABLE from a belief pile
/// on ingest, and the gate cannot reject it.
///
/// That is not as bad as it sounds: the listed order survives ingest
/// unchanged either way, so the only thing the flag still buys a lying
/// producer is `pop_next_card`'s belief-mode RNG consumption — which is
/// exactly what it asked for. The real defense is that no wire producer
/// sets the flag; `blindfold::belief_root` is its only writer.
#[test]
fn an_unpinned_shuffle_is_indistinguishable_from_a_belief_on_the_wire() {
    Python::with_gil(|py| {
        let mut s = build_seeded_state(py, 3, 0).expect("seed=3 must build");
        assert!(s.player.draw.len() >= 3, "fixture needs a draw pile");
        s.player.draw.sort_unstable();
        s.player.draw.reverse();
        s.player.pinned_top.clear();
        s.player.pinned_bottom.clear();
        s.player.deferred_draw_order = true;
        assert!(
            !sts2_core::blindfold::unknown_middle_is_canonical(&s.player),
            "pre-wire, the reversed pile is detectably non-canonical"
        );

        let dict = state_to_dict(py, &s).expect("to_dict must succeed");
        let s2 = state_from_dict(&dict).expect("re-minted uids make the pile look canonical");
        assert!(
            sts2_core::blindfold::unknown_middle_is_canonical(&s2.player),
            "positional uid minting is what erases the distinction",
        );
        state_equiv(&s, &s2).expect("the listed order itself must survive intact");
    });
}

#[test]
fn round_trip_preserves_cards_in_flight() {
    // Found by the Phase-1 mirror oracle. `player.play` — the card that is
    // mid-resolution — had no wire representation at all, so a mid-choice
    // reconstruction dropped it: resolving the choice then flushed an empty
    // play pile and the played card never reached the discard. Since a
    // card-select screen is ALWAYS mid-resolution, that hit every single
    // injected choice the mirror handles.
    Python::with_gil(|py| {
        let mut s1 = build_seeded_state(py, 3, 0).expect("seed=3 must build");
        let in_flight = s1.player.hand.remove(0);
        s1.player.play.push(in_flight);

        let dict = state_to_dict(py, &s1).expect("to_dict must succeed");
        let deck = dict
            .get_item("deck")
            .unwrap()
            .expect("deck present")
            .downcast::<PyDict>()
            .unwrap()
            .clone();
        assert!(
            deck.get_item("play_pile").unwrap().is_some(),
            "an in-flight card must be emitted"
        );

        let s2 = state_from_dict(&dict).expect("from_dict on emitted dict must succeed");
        state_equiv(&s1, &s2).expect("in-flight card must round-trip");
        assert_eq!(s2.player.play.len(), 1);

        // Between actions there is nothing in flight, and the key stays off
        // the wire — shape stability for every producer that predates it.
        let s3 = build_seeded_state(py, 3, 0).expect("seed=3 must build");
        let dict = state_to_dict(py, &s3).expect("to_dict must succeed");
        let deck = dict
            .get_item("deck")
            .unwrap()
            .expect("deck present")
            .downcast::<PyDict>()
            .unwrap()
            .clone();
        assert!(deck.get_item("play_pile").unwrap().is_none());
        assert!(state_from_dict(&dict)
            .expect("from_dict without the key must succeed")
            .player
            .play
            .is_empty());
    });
}

/// Reconstructed cards must carry their canonical tags.
///
/// `CardTags::NONE` on every rebuilt card made tag-gated handlers read an
/// empty deck — PerfectedStrike counted zero Strikes. The oracle sweep found
/// it, but only as a damage number; this names the cause so a regression
/// fails on the tag itself rather than on a statistical ceiling.
#[test]
fn round_trip_preserves_canonical_card_tags() {
    Python::with_gil(|py| {
        let s = build_seeded_state(py, 3, 0).expect("seed=3 must build");
        let expected: Vec<_> = s
            .cards
            .iter()
            .map(|c| (c.card_id, c.tags))
            .filter(|(id, _)| *id == CardId::StrikeIronclad)
            .collect();
        assert!(
            !expected.is_empty(),
            "fixture must contain a Strike for this to mean anything"
        );
        assert!(
            expected
                .iter()
                .any(|(_, tags)| *tags != sts2_core::cards::CardTags::NONE),
            "a Strike must carry a non-empty canonical tag set"
        );

        let dict = state_to_dict(py, &s).expect("to_dict must succeed");
        let s2 = state_from_dict(&dict).expect("from_dict must succeed");
        for (a, b) in s.cards.iter().zip(s2.cards.iter()) {
            assert_eq!(
                (a.card_id, a.tags),
                (b.card_id, b.tags),
                "tags must survive the round-trip for {:?}",
                a.card_id
            );
        }
    });
}

/// A Defect belt must come back padded to its capacity, not compacted.
///
/// `channel` fills the first `is_empty()` slot and EVICTS the oldest orb when
/// it finds none, so a compact belt turned every channel on a non-full belt
/// into an eviction — the reconstructed orb count could never grow.
#[test]
fn round_trip_pads_the_orb_belt_to_capacity() {
    Python::with_gil(|py| {
        let mut s = build_seeded_state(py, 3, 0).expect("seed=3 must build");
        s.player.orbs = sts2_core::OrbSlots {
            slots: smallvec::smallvec![
                sts2_core::Orb::Lightning,
                sts2_core::Orb::Empty,
                sts2_core::Orb::Empty
            ],
            capacity: 3,
        };

        let dict = state_to_dict(py, &s).expect("to_dict must succeed");
        let s2 = state_from_dict(&dict).expect("from_dict must succeed");
        assert_eq!(s2.player.orbs.capacity, 3, "capacity must survive");
        assert_eq!(
            s2.player.orbs.slots.len(),
            3,
            "belt must be padded back to capacity, not left compact — a short \
             belt makes `channel` evict instead of fill"
        );
        assert_eq!(s2.player.orbs.slots[0], sts2_core::Orb::Lightning);
        assert!(
            s2.player.orbs.slots[1..]
                .iter()
                .all(|o| *o == sts2_core::Orb::Empty),
            "the padding must be empty slots"
        );
    });
}

/// `intent_amounts` selects the telegraph dialect, and `state_to_dict`
/// stamps `base` so its own output ingests verbatim.
///
/// Without the marker, a `state_to_dict` dict was re-read as `effective` and
/// run back through the un-baking inversion — subtracting modifiers that
/// were never added. The oracle caught it as a Strength-buffed 12 coming
/// back as 10; this asserts the marker and the fixed point directly.
#[test]
fn state_to_dict_stamps_the_base_intent_dialect_and_round_trips_it() {
    Python::with_gil(|py| {
        let s = build_seeded_state(py, 3, 0).expect("seed=3 must build");
        let dict = state_to_dict(py, &s).expect("to_dict must succeed");
        let marker: String = dict
            .get_item("intent_amounts")
            .unwrap()
            .expect("state_to_dict must declare its dialect")
            .extract()
            .expect("dialect marker is a string");
        assert_eq!(
            marker, "base",
            "our own output is pre-modifier, so it must be ingested verbatim"
        );

        // Idempotent: re-emitting the reconstruction must not shift the
        // telegraph, which is what a wrongly-inverted dialect would do.
        let s2 = state_from_dict(&dict).expect("from_dict must succeed");
        let intents =
            |st: &CoreCombatState| -> Vec<_> { st.creatures.iter().map(|c| c.intent).collect() };
        assert_eq!(
            intents(&s),
            intents(&s2),
            "telegraph must be a fixed point of the round-trip"
        );
    });
}

/// A reconstructed Osty must not collide with another creature's id.
///
/// `summon_osty` gives a fresh pet `CreatureUid(creatures.len() + 1)`, so it
/// is only safe once the arena is full. Applied from the player-scalars
/// section — where `$.player.osty` naturally belongs — the arena is still
/// empty, the pet takes id 1, and the player row is pushed with id 1 right
/// after: two creatures answering to one id, `DieForYou` landing on whichever
/// is found first, and an enemy's real power silently displaced.
///
/// Found by the mirror oracle (Necrobinder seed 10). Asserted here on the
/// round-trip directly, because that is where it lives — no mirror involved.
#[test]
fn osty_reconstruction_does_not_alias_another_creature_id() {
    Python::with_gil(|py| {
        let s = build_seeded_state(py, 3, 0).expect("seed=3 must build");
        let dict = state_to_dict(py, &s).expect("to_dict must succeed");
        let player = dict
            .get_item("player")
            .unwrap()
            .expect("player present")
            .downcast::<PyDict>()
            .unwrap()
            .clone();
        let osty = PyDict::new(py);
        osty.set_item("alive", true).unwrap();
        osty.set_item("hp", 7).unwrap();
        osty.set_item("max_hp", 7).unwrap();
        osty.set_item("block", 0).unwrap();
        player.set_item("osty", osty).unwrap();

        let s2 = state_from_dict(&dict).expect("a state with an Osty must rebuild");

        let mut ids: Vec<u32> = s2.creatures.iter().map(|c| c.id.get()).collect();
        let before = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), before, "every creature must have a distinct id");

        let pet = s2.player.pet_uid.expect("the pet must be summoned");
        assert_ne!(
            Some(pet),
            s2.player.creature_uid,
            "the Osty must not be the player row"
        );
        assert!(
            !s2.enemy_uids.contains(&pet),
            "the Osty must not alias an enemy"
        );

        // Every power handle must resolve. An aliased id leaves one dangling,
        // and `state_to_dict` is where that surfaces.
        state_to_dict(py, &s2).expect("the rebuilt state must re-serialize");

        // The enemy keeps its OWN powers — the aliasing symptom was an
        // enemy's power list being overwritten by the pet's DieForYou.
        for (a, b) in s.creatures.iter().zip(s2.creatures.iter()) {
            if Some(sts2_core::CreatureUid(a.id)) != s.player.creature_uid {
                assert_eq!(
                    a.powers.len(),
                    b.powers.len(),
                    "creature {:?} lost or gained a power to the Osty summon",
                    a.id
                );
            }
        }
    });
}

#[test]
fn synthetic_status_and_curse_ids_resolve_to_real_cards() {
    // The largest reconstruction gap left on the DLL path after the cost and
    // roster fixes. The producers spell an inflicted status as `status:Slimed`
    // and a curse as `curse:Curse of the Bell`; the parser treated ANY
    // prefixed id as unresolvable and `build_card` funnelled it into the
    // AscendersBane placeholder. So every Slimed / Dazed / Burn / Wound the
    // enemy inflicted entered the searched deck as an unplayable ethereal
    // curse, and Burn's end-of-turn damage never happened in rollouts at all.
    // Measured on the 60-run realdll_v6 sweep: 1374 AscendersBane in the
    // mirror against 751 `status:Slimed`, 392 `status:Dazed`, 278
    // `curse:Clumsy`, 182 `status:Burn`, 120 `status:Wound` in the game.
    Python::with_gil(|py| {
        for (wire_id, kind, expected) in [
            ("status:Slimed", "status", CardId::Slimed),
            ("status:Burn", "status", CardId::Burn),
            ("status:Dazed", "status", CardId::Dazed),
            ("status:Wound", "status", CardId::Wound),
            ("curse:Clumsy", "curse", CardId::Clumsy),
            ("curse:Curse of the Bell", "curse", CardId::CurseOfTheBell),
        ] {
            let dict = make_seeded_fixture(py, 3, 0).expect("fixture");
            let deck = dict
                .get_item("deck")
                .unwrap()
                .unwrap()
                .downcast::<PyDict>()
                .unwrap()
                .clone();
            let discard = deck
                .get_item("discard_pile")
                .unwrap()
                .unwrap()
                .downcast::<PyList>()
                .unwrap()
                .clone();
            let c = PyDict::new(py);
            c.set_item("id", wire_id).unwrap();
            c.set_item("stable_idx", 99).unwrap();
            c.set_item("kind", kind).unwrap();
            c.set_item("upgraded", false).unwrap();
            discard.append(c).unwrap();

            let state = state_from_dict(&dict).unwrap_or_else(|e| panic!("{wire_id}: {e}"));
            let last = *state.player.discard.last().expect("the appended card");
            let got = state.cards.iter().find(|c| c.uid == last).unwrap().card_id;
            assert_eq!(got, expected, "{wire_id} must resolve to its real card");
        }
    });
}

#[test]
fn an_unknown_synthetic_id_still_degrades_rather_than_failing() {
    // The AscendersBane placeholder stays for ids the registry genuinely
    // lacks: a status this build has never heard of must not fail the whole
    // reconstruction.
    Python::with_gil(|py| {
        let dict = make_seeded_fixture(py, 3, 0).expect("fixture");
        let deck = dict
            .get_item("deck")
            .unwrap()
            .unwrap()
            .downcast::<PyDict>()
            .unwrap()
            .clone();
        let discard = deck
            .get_item("discard_pile")
            .unwrap()
            .unwrap()
            .downcast::<PyList>()
            .unwrap()
            .clone();
        let c = PyDict::new(py);
        c.set_item("id", "status:NoSuchStatusExists").unwrap();
        c.set_item("stable_idx", 99).unwrap();
        c.set_item("kind", "status").unwrap();
        c.set_item("upgraded", false).unwrap();
        discard.append(c).unwrap();

        let state = state_from_dict(&dict).expect("an unknown status must not fail the rebuild");
        let last = *state.player.discard.last().unwrap();
        assert_eq!(
            state.cards.iter().find(|c| c.uid == last).unwrap().card_id,
            CardId::AscendersBane,
        );
    });
}

#[test]
fn a_this_turn_discount_does_not_become_permanent() {
    // The largest divergence bucket on the 60-run realdll_v5 mirror sweep:
    // the wire carried only the EFFECTIVE cost, so `build_card` baked a
    // transient discount into `Card.cost` and the reconstruction believed the
    // card was cheap forever. 1135 of ~1700 card-cost divergences had the
    // mirror cheaper than the game, concentrated on `end_turn` — the exact
    // moment the real discount expires. Both halves are needed: the base cost
    // to rebuild from, and the `combat_vars` entry that re-applies (and later
    // expires) the discount.
    Python::with_gil(|py| {
        let mut s1 = build_seeded_state(py, 3, 0).expect("seed=3 must build");
        let uid = s1.player.hand[0];
        let idx = s1
            .cards
            .iter()
            .position(|c| c.uid == uid)
            .expect("hand card in arena");
        let base = s1.cards[idx].cost;
        assert!(base > 0, "fixture card must cost something to discount");
        s1.cards[idx]
            .combat_vars
            .push((sts2_core::cards::VarKey::FreeThisTurn, 1));

        let dict = state_to_dict(py, &s1).expect("to_dict must succeed");
        let deck = dict
            .get_item("deck")
            .unwrap()
            .expect("deck present")
            .downcast::<PyDict>()
            .unwrap()
            .clone();
        let hand = deck
            .get_item("hand")
            .unwrap()
            .unwrap()
            .downcast::<PyList>()
            .unwrap()
            .get_item(0)
            .unwrap();
        let hand = hand.downcast::<PyDict>().unwrap();
        // The wire says both things: what it costs right now, and what the
        // instance's own cost is underneath.
        assert_eq!(
            hand.get_item("cost_override")
                .unwrap()
                .unwrap()
                .extract::<i32>()
                .unwrap(),
            0,
            "the effective cost is free this turn"
        );
        assert_eq!(
            hand.get_item("base_cost")
                .unwrap()
                .unwrap()
                .extract::<i32>()
                .unwrap(),
            i32::from(base),
            "the instance's own cost is unchanged"
        );

        let s2 = state_from_dict(&dict).expect("from_dict on emitted dict must succeed");
        let uid2 = s2.player.hand[0];
        let c2 = s2.cards.iter().find(|c| c.uid == uid2).unwrap();
        assert_eq!(c2.cost, base, "the rebuild must NOT bake the discount in");
        assert!(
            c2.combat_vars
                .iter()
                .any(|(k, v)| *k == sts2_core::cards::VarKey::FreeThisTurn && *v == 1),
            "the discount itself must survive, so it can also expire"
        );
        state_equiv(&s1, &s2).expect("discounted state must round-trip");
    });
}

/// The C# dialects have no `combat_vars`, so they describe a this-turn
/// discount by its EFFECT instead: the cost now, and the cost once the turn's
/// modifiers wear off. `build_card` banks the difference so the turn-start
/// restore expires it on the same boundary C# `EndOfTurnCleanup` does.
///
/// Without it a Stomp rebuilt mid-turn kept its per-Attack shave for the rest
/// of the fight — 163 of its 175 cost divergences on the v8 mirror sweep sat
/// on exactly the `end_turn` that should have restored it.
#[test]
fn a_wire_transient_discount_expires_at_turn_start() {
    Python::with_gil(|py| {
        let dict = make_seeded_fixture(py, 3, 0).expect("seed=3 fixture");
        let deck = dict
            .get_item("deck")
            .unwrap()
            .unwrap()
            .downcast::<PyDict>()
            .unwrap()
            .clone();
        let hand = deck
            .get_item("hand")
            .unwrap()
            .unwrap()
            .downcast::<PyList>()
            .unwrap()
            .clone();
        let card = hand.get_item(0).unwrap();
        let card = card.downcast::<PyDict>().unwrap();
        // A card the world is showing at 0 that goes back to 2 next turn.
        card.set_item("base_cost", 0).unwrap();
        card.set_item("cost_override", 0).unwrap();
        card.set_item("persistent_cost", 2).unwrap();

        let mut state = state_from_dict(&dict).expect("rebuild must succeed");
        let uid = state.player.hand[0];
        let idx = state.cards.iter().position(|c| c.uid == uid).unwrap();
        assert_eq!(state.cards[idx].cost, 0, "the card is cheap RIGHT NOW");
        assert_eq!(
            state.cards[idx].combat_var(sts2_core::cards::VarKey::WireTransientCostThisTurn),
            2,
            "and the shave is banked so it can expire"
        );

        state.restore_wire_transient_costs_for_turn_start();
        assert_eq!(state.cards[idx].cost, 2, "the discount expired on schedule");
        assert_eq!(
            state.cards[idx].combat_var(sts2_core::cards::VarKey::WireTransientCostThisTurn),
            0,
            "and the bank is spent, so a second turn cannot restore it twice"
        );
    });
}

/// A permanent discount must NOT be banked: `persistent_cost` equal to the
/// current cost means nothing expires, which is the overwhelmingly common
/// case and the one an Enlightenment-style "costs 1 this combat" produces.
#[test]
fn a_permanent_discount_is_not_banked_as_transient() {
    Python::with_gil(|py| {
        let dict = make_seeded_fixture(py, 3, 0).expect("seed=3 fixture");
        let deck = dict
            .get_item("deck")
            .unwrap()
            .unwrap()
            .downcast::<PyDict>()
            .unwrap()
            .clone();
        let hand = deck
            .get_item("hand")
            .unwrap()
            .unwrap()
            .downcast::<PyList>()
            .unwrap()
            .clone();
        let card = hand.get_item(0).unwrap();
        let card = card.downcast::<PyDict>().unwrap();
        card.set_item("base_cost", 1).unwrap();
        card.set_item("cost_override", 1).unwrap();
        card.set_item("persistent_cost", 1).unwrap();

        let mut state = state_from_dict(&dict).expect("rebuild must succeed");
        let uid = state.player.hand[0];
        let idx = state.cards.iter().position(|c| c.uid == uid).unwrap();
        state.restore_wire_transient_costs_for_turn_start();
        assert_eq!(
            state.cards[idx].cost, 1,
            "a discount that outlives the turn must survive the turn boundary"
        );
    });
}

/// The engine's own per-effect banks and `persistent_cost` describe the same
/// shave two different ways. A producer sending both would have it restored
/// twice, so the rebuild refuses rather than picking one.
#[test]
fn both_cost_representations_at_once_is_a_schema_error() {
    Python::with_gil(|py| {
        let dict = make_seeded_fixture(py, 3, 0).expect("seed=3 fixture");
        let deck = dict
            .get_item("deck")
            .unwrap()
            .unwrap()
            .downcast::<PyDict>()
            .unwrap()
            .clone();
        let hand = deck
            .get_item("hand")
            .unwrap()
            .unwrap()
            .downcast::<PyList>()
            .unwrap()
            .clone();
        let card = hand.get_item(0).unwrap();
        let card = card.downcast::<PyDict>().unwrap();
        card.set_item("base_cost", 0).unwrap();
        card.set_item("persistent_cost", 3).unwrap();
        let cv = PyDict::new(py);
        cv.set_item("stomp_cost_reduction_this_turn", 3).unwrap();
        card.set_item("combat_vars", cv).unwrap();

        let err = state_from_dict(&dict).expect_err("two representations must be refused");
        let msg = err.to_string();
        assert!(
            msg.contains("stomp_cost_reduction_this_turn") && msg.contains("persistent_cost"),
            "the error must name both representations, got: {msg}"
        );
    });
}

/// Two live banks without `stable_cost` is a schema error, not a default.
///
/// `rebank_wire_cost_after_edge` recomputes the surviving bank against
/// `WireCostStable`, read back through a `combat_var` accessor that DEFAULTS TO
/// 0 — so an absent `stable_cost` is indistinguishable from a real zero, and
/// the second edge drives the card's cost to `0 - cost`. Silent corruption, not
/// degradation.
///
/// The refusal is scoped to exactly that reach: `persistent_cost` alone is the
/// pre-v16 dialect, only ever lights ONE bank, and `rebank` early-returns on
/// the zero one — so `stable_cost` is never read and the older wire must keep
/// working. Both halves are asserted here, because a gate that also rejected
/// the safe case would be a silent compatibility break.
#[test]
fn two_cost_banks_without_the_stable_cost_are_refused() {
    Python::with_gil(|py| {
        let doctor = |played: Option<i32>| {
            let dict = make_seeded_fixture(py, 3, 0).expect("seed=3 fixture");
            let card = dict
                .get_item("deck")
                .unwrap()
                .unwrap()
                .downcast::<PyDict>()
                .unwrap()
                .get_item("hand")
                .unwrap()
                .unwrap()
                .downcast::<PyList>()
                .unwrap()
                .get_item(0)
                .unwrap();
            let card = card.downcast::<PyDict>().unwrap();
            card.set_item("base_cost", 0).unwrap();
            card.set_item("persistent_cost", 3).unwrap();
            if let Some(p) = played {
                card.set_item("played_cost", p).unwrap();
            }
            dict
        };

        let err = state_from_dict(&doctor(Some(3)))
            .expect_err("two banks with no `stable_cost` must be refused");
        let msg = err.to_string();
        assert!(
            msg.contains("stable_cost") && msg.contains("0 - cost"),
            "the error must name the missing field and what it would do, got: {msg}"
        );

        // The pre-v16 dialect — one bank, no `stable_cost` — still rebuilds.
        state_from_dict(&doctor(None)).expect("`persistent_cost` alone stays valid");
    });
}

/// The run's own dice round-trip, and land on the same NEXT roll.
///
/// Counters alone are not the point — being about to draw the same card is.
/// A reconstruction seeded from anything but the run's master seed has
/// `monster_ai` and `shuffle` streams unrelated to the producer's, so every
/// enemy intent and reshuffle order it predicts is a guess; a guess that
/// misses is not evidence of a fidelity bug, because the value was never
/// derivable from the state. Restoring both seed and cursor is what turns
/// those predictions into claims worth checking.
#[test]
fn the_runs_dice_round_trip_to_the_same_next_roll() {
    Python::with_gil(|py| {
        let mut s1 = build_seeded_state(py, 5, 0).expect("seed=5 must build");
        s1.run_rng = sts2_core::run_rng_set::RunRngSet::new(1234);
        // Burn an uneven number of reads so a rebuild that forgets the cursor
        // (or the seed) cannot accidentally agree.
        for _ in 0..7 {
            let _ = s1.run_rng.monster_ai.next_int(0, 100);
        }
        for _ in 0..3 {
            let _ = s1.run_rng.shuffle.next_int(0, 100);
        }

        let dict = state_to_dict(py, &s1).expect("to_dict must succeed");
        let s2 = state_from_dict(&dict).expect("from_dict must succeed");

        assert_eq!(s2.run_rng.seed, s1.run_rng.seed, "master seed round-trips");
        assert_eq!(s2.run_rng.monster_ai.counter(), 7);
        assert_eq!(s2.run_rng.shuffle.counter(), 3);

        // The claim that actually matters: both are about to roll the same.
        let (mut a, mut b) = (s1, s2);
        for _ in 0..5 {
            assert_eq!(
                a.run_rng.monster_ai.next_int(0, 999),
                b.run_rng.monster_ai.next_int(0, 999),
                "a restored stream must continue the producer's sequence"
            );
            assert_eq!(
                a.run_rng.shuffle.next_int(0, 999),
                b.run_rng.shuffle.next_int(0, 999),
            );
        }
    });
}

/// An unknown stream name is a vocabulary mismatch between producer and
/// engine, and silently ignoring it would leave that stream at roll zero
/// while every report claimed the dice were restored.
#[test]
fn an_unknown_rng_stream_name_fails_loud() {
    Python::with_gil(|py| {
        let dict = make_seeded_fixture(py, 5, 0).expect("seed=5 fixture");
        let rng = PyDict::new(py);
        rng.set_item("seed", 0).unwrap();
        rng.set_item("run_seed", 1234).unwrap();
        let counters = PyDict::new(py);
        counters.set_item("monster_ai", 3).unwrap();
        counters.set_item("moonster_ai", 3).unwrap();
        rng.set_item("counters", counters).unwrap();
        dict.set_item("rng", rng).unwrap();

        let err = state_from_dict(&dict).expect_err("an unknown stream must be refused");
        assert!(
            err.to_string().contains("moonster_ai"),
            "the error must name the stream, got: {err}"
        );
    });
}

#[test]
fn round_trip_preserves_power_counters() {
    // Found by the 60-run realdll_v5 mirror sweep. `Power::counter` is the
    // per-instance accumulator `amount` does not describe — CrimsonMantle's
    // self-damage total, Slow's cards-played tally, Surrounded's facing — and
    // it had no wire representation, so every reconstruction restarted it at
    // zero. Fingerprint: 54 single-field `$.player.hp +1` divergences, all
    // after `end_turn`, all with CrimsonMantle on the player and no other
    // common factor — a reconstructed Mantle dealt no self-damage at all.
    Python::with_gil(|py| {
        // seed=2: the fixture gives EVEN seeds a Strength power (odd seeds get
        // none), and a power is what we need to hang a counter on.
        let mut s1 = build_seeded_state(py, 2, 0).expect("seed=2 must build");
        assert!(!s1.player.powers.is_empty(), "fixture must carry a power");
        let uid = s1.player.powers[0];
        let idx = s1
            .powers
            .iter()
            .position(|p| p.id == uid.0)
            .expect("power in arena");
        s1.powers[idx].counter = 7;

        let dict = state_to_dict(py, &s1).expect("to_dict must succeed");
        let s2 = state_from_dict(&dict).expect("from_dict on emitted dict must succeed");
        let uid2 = s2.player.powers[0];
        let got = s2
            .powers
            .iter()
            .find(|p| p.id == uid2.0)
            .expect("power in arena")
            .counter;
        assert_eq!(got, 7, "the per-instance counter must survive the wire");
        state_equiv(&s1, &s2).expect("counter-carrying state must round-trip");
    });
}

#[test]
fn pin_index_out_of_range_fails_loud() {
    // No dirty fallback: a pin index the draw pile can't satisfy is a
    // producer bug, and swallowing it would silently strip knowledge the
    // mirror depends on.
    Python::with_gil(|py| {
        let s = build_seeded_state(py, 3, 0).expect("seed=3 must build");
        let draw_len = s.player.draw.len();
        let dict = state_to_dict(py, &s).expect("to_dict must succeed");
        let deck = dict
            .get_item("deck")
            .unwrap()
            .expect("deck present")
            .downcast::<PyDict>()
            .unwrap()
            .clone();
        deck.set_item("pinned_top", vec![draw_len]).unwrap();
        let err = state_from_dict(&dict).expect_err("out-of-range pin must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("pinned_top"),
            "error must name the offending path, got: {msg}"
        );

        // Same index on both ends is contradictory.
        deck.set_item("pinned_top", vec![0_usize]).unwrap();
        deck.set_item("pinned_bottom", vec![0_usize]).unwrap();
        let err = state_from_dict(&dict).expect_err("double-pinned index must be rejected");
        assert!(
            err.to_string().contains("both top and bottom"),
            "error must explain the contradiction, got: {err}"
        );
    });
}

#[test]
fn round_trip_smoke_seed_zero() {
    // Standalone smoke so a regression doesn't hide in the 1000-seed
    // bulk run's error tail.
    Python::with_gil(|py| {
        let s1 = build_seeded_state(py, 0, 0).expect("seed=0 must build");
        let dict = state_to_dict(py, &s1).expect("to_dict must succeed");
        let s2 = state_from_dict(&dict).expect("from_dict on emitted dict must succeed");
        state_equiv(&s1, &s2).expect("seed=0 must round-trip");
    });
}

#[test]
fn card_affliction_and_bound_flag_round_trip() {
    // ChainsOfBinding (Queen fight): the wire used to drop per-card
    // afflictions and the once-per-turn Bound gate entirely, so a
    // reconstructed state planned as if every Bound card were playable
    // (live sweep-9 forensics, 2026-07-13). Assert both survive
    // dict -> state -> dict -> state.
    Python::with_gil(|py| {
        use sts2_core::card_afflictions::CardAfflictionKind;

        let d = make_seeded_fixture(py, 3, 0).expect("fixture");
        // Afflict the first hand card: Bound x3.
        let deck = d
            .get_item("deck")
            .unwrap()
            .expect("deck present")
            .downcast::<PyDict>()
            .unwrap()
            .clone();
        let hand = deck
            .get_item("hand")
            .unwrap()
            .expect("hand present")
            .downcast::<PyList>()
            .unwrap()
            .clone();
        let card0 = hand
            .get_item(0)
            .expect("hand[0]")
            .downcast::<PyDict>()
            .unwrap()
            .clone();
        card0.set_item("affliction", "Bound").unwrap();
        card0.set_item("affliction_stacks", 3).unwrap();
        let player = d
            .get_item("player")
            .unwrap()
            .expect("player present")
            .downcast::<PyDict>()
            .unwrap()
            .clone();
        player
            .set_item("bound_card_played_this_turn", true)
            .unwrap();

        let s1 = state_from_dict(&d).expect("from_dict with affliction");
        let hand0 = s1.player.hand.first().copied().expect("hand not empty");
        let c = s1
            .cards
            .iter()
            .find(|c| c.uid == hand0)
            .expect("hand[0] in arena");
        let aff = c.affliction.expect("hand[0] must carry the affliction");
        assert_eq!(aff.kind, CardAfflictionKind::Bound);
        assert_eq!(aff.stacks, 3);
        assert!(!aff.applied_exhaust);
        assert!(
            s1.combat_log.chains_bound_card_played_this_turn,
            "bound_card_played_this_turn must reach combat_log"
        );

        // Second leg: emit and rebuild.
        let dict2 = state_to_dict(py, &s1).expect("to_dict");
        let s2 = state_from_dict(&dict2).expect("from_dict on emitted dict");
        let hand0b = s2.player.hand.first().copied().expect("hand not empty");
        let c2 = s2
            .cards
            .iter()
            .find(|c| c.uid == hand0b)
            .expect("hand[0] in arena");
        let aff2 = c2
            .affliction
            .expect("affliction must survive the round-trip");
        assert_eq!(aff2.kind, CardAfflictionKind::Bound);
        assert_eq!(aff2.stacks, 3);
        assert!(s2.combat_log.chains_bound_card_played_this_turn);
    });
}

#[test]
fn dead_reviver_reconstructs_as_corpse() {
    // Revive blindness (2026-07-13 rung-3.5 roster audit): the wire only
    // keeps corpses that WILL come back (Decimillipede segments awaiting
    // REATTACH, Eye with Teeth illusions), but `state_from_dict` skipped
    // every `is_dead` entry — so MCTS rollouts treated each kill as
    // permanent and undervalued the remaining fight. A dead reviver must
    // reconstruct as a 0-hp corpse with its powers and an AI, in its wire
    // slot, without becoming targetable-alive.
    Python::with_gil(|py| {
        let d = make_seeded_fixture(py, 5, 0).expect("fixture");
        let enc = d
            .get_item("encounter")
            .unwrap()
            .expect("encounter present")
            .downcast::<PyList>()
            .unwrap()
            .clone();
        let corpse = PyDict::new(py);
        corpse
            .set_item("kind", "DecimillipedeSegmentFront")
            .unwrap();
        corpse.set_item("uid", 2_i64).unwrap();
        corpse.set_item("hp", 0).unwrap();
        corpse.set_item("max_hp", 52).unwrap();
        corpse.set_item("block", 0).unwrap();
        let powers = PyList::empty(py);
        let reattach = PyDict::new(py);
        reattach.set_item("name", "REATTACH").unwrap();
        reattach.set_item("amount", 25).unwrap();
        powers.append(reattach).unwrap();
        corpse.set_item("powers", powers).unwrap();
        corpse.set_item("intents", PyList::empty(py)).unwrap();
        corpse
            .set_item("intent_history", PyList::empty(py))
            .unwrap();
        corpse.set_item("is_dead", true).unwrap();
        corpse.set_item("is_escaped", false).unwrap();
        corpse.set_item("move_state", "DEAD").unwrap();
        enc.append(corpse).unwrap();

        let s = state_from_dict(&d).expect("from_dict with corpse");
        assert_eq!(
            s.creatures.len(),
            3,
            "player + alive enemy + corpse must all reconstruct"
        );
        assert_eq!(s.enemy_uids.len(), 2, "corpse stays in enemy_uids");
        let corpse_c = s
            .creatures
            .iter()
            .find(|c| c.species == Some(sts2_core::MonsterKind::DecimillipedeSegmentFront))
            .expect("corpse creature present");
        assert_eq!(corpse_c.current_hp, 0, "corpse is dead");
        assert!(
            corpse_c.ai.is_some(),
            "corpse keeps an AI (DEAD state resolves; revived enemies must not be inert)"
        );
        assert_eq!(
            corpse_c.ai.as_ref().unwrap().current_state_id(),
            "DEAD",
            "corpse AI pinned at its wire move-state"
        );
        let has_reattach = corpse_c.powers.iter().any(|puid| {
            s.powers
                .iter()
                .any(|p| p.id == puid.0 && p.kind == sts2_core::PowerKind::Reattach)
        });
        assert!(has_reattach, "ReattachPower must survive onto the corpse");

        // Escaped enemies are still dropped.
        let escaped = PyDict::new(py);
        escaped.set_item("kind", "Nibbit").unwrap();
        escaped.set_item("uid", 3_i64).unwrap();
        escaped.set_item("hp", 10).unwrap();
        escaped.set_item("max_hp", 48).unwrap();
        escaped.set_item("block", 0).unwrap();
        escaped.set_item("powers", PyList::empty(py)).unwrap();
        escaped.set_item("intents", PyList::empty(py)).unwrap();
        escaped
            .set_item("intent_history", PyList::empty(py))
            .unwrap();
        escaped.set_item("is_dead", false).unwrap();
        escaped.set_item("is_escaped", true).unwrap();
        enc.append(escaped).unwrap();
        let s2 = state_from_dict(&d).expect("from_dict with escaped");
        assert_eq!(s2.creatures.len(), 3, "escaped enemy must still be skipped");
    });
}

#[test]
fn unknown_card_affliction_fails_loud() {
    // Soft-dropping unknown names is how the move_state / card-id
    // dialect bugs stayed invisible for weeks — an unknown affliction
    // must reject the state, not silently plan without it.
    Python::with_gil(|py| {
        let d = make_seeded_fixture(py, 4, 0).expect("fixture");
        let deck = d
            .get_item("deck")
            .unwrap()
            .expect("deck present")
            .downcast::<PyDict>()
            .unwrap()
            .clone();
        let hand = deck
            .get_item("hand")
            .unwrap()
            .expect("hand present")
            .downcast::<PyList>()
            .unwrap()
            .clone();
        let card0 = hand
            .get_item(0)
            .expect("hand[0]")
            .downcast::<PyDict>()
            .unwrap()
            .clone();
        card0.set_item("affliction", "Cursed").unwrap();
        assert!(
            state_from_dict(&d).is_err(),
            "unknown affliction kind must fail loud"
        );
    });
}

/// A live Osty must survive a `state_to_dict` -> `state_from_dict` round
/// trip. The 1000-seed round trip above is Ironclad-vs-Cultist and never
/// constructs a pet, so it cannot catch pet loss.
///
/// Osty lives in `state.creatures` and is in NEITHER `enemy_uids` NOR
/// `allies` -- the same structural blind spot that made the TT hash alias
/// Osty-alive with Osty-dead positions. `state_from_dict` already parses
/// `player.osty`, but `state_to_dict` never emitted it, so a Necrobinder
/// state round-tripped to one with NO pet: silent loss of the single most
/// valuable accumulated Necrobinder resource (Osty MaxHp compounds across
/// summons and resets on death).
#[test]
fn live_osty_survives_state_dict_round_trip() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let mut state = CoreCombatState::default();
        state.summon_osty(10);
        let pet_uid = state.player.pet_uid.expect("summon sets pet_uid");
        {
            let pet = state
                .creatures
                .iter_mut()
                .find(|c| c.id == pet_uid.0)
                .expect("pet creature in arena");
            // Distinguish current from max so a "reset to summon amount"
            // regression cannot masquerade as a successful round trip.
            pet.max_hp = 40;
            pet.current_hp = 37;
            pet.block = 3;
        }

        let d = state_to_dict(py, &state).expect("to_dict");
        let back = state_from_dict(&d).expect("from_dict");

        let pet_of = |s: &CoreCombatState| {
            s.player
                .pet_uid
                .and_then(|u| s.creatures.iter().find(|c| c.id == u.0))
                .map(|c| (c.current_hp, c.max_hp, c.block))
        };
        assert_eq!(
            pet_of(&back),
            Some((37, 40, 3)),
            "live Osty (hp/max_hp/block) must survive the round trip"
        );
    });
}

/// A DEAD Osty must survive too — the corpse is load-bearing.
///
/// `summon_osty`'s revive branch keys off an existing `pet_uid` with
/// `current_hp == 0`, and the corpse carries the max_hp the pet had grown to.
/// Dropping it loses that growth AND turns the next summon into a fresh spawn
/// at the summon amount, which is a strictly worse pet for the rest of the run.
///
/// Split from [`live_osty_survives_state_dict_round_trip`] deliberately: an
/// `alive:false, hp:0` pet is exactly the shape a "restore only when alive"
/// guard silently drops, and the live case cannot catch that.
#[test]
fn a_dead_osty_keeps_its_grown_max_hp_through_the_round_trip() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let mut state = CoreCombatState::default();
        state.summon_osty(3);
        let pet_uid = state.player.pet_uid.expect("summon sets pet_uid");
        {
            let pet = state
                .creatures
                .iter_mut()
                .find(|c| c.id == pet_uid.0)
                .expect("pet creature in arena");
            // Grown to 7, then killed: max_hp must outlive current_hp.
            pet.max_hp = 7;
            pet.current_hp = 0;
            pet.block = 0;
        }

        let d = state_to_dict(py, &state).expect("to_dict");
        let back = state_from_dict(&d).expect("from_dict");

        let pet = back
            .player
            .pet_uid
            .and_then(|u| back.creatures.iter().find(|c| c.id == u.0));
        assert_eq!(
            pet.map(|c| (c.current_hp, c.max_hp)),
            Some((0, 7)),
            "a dead pet must round-trip as a 0/7 corpse, not vanish"
        );
    });
}

/// A telegraph the game DISCLOSED must survive the wire.
///
/// `Summon` / `Heal` / `DeathBlow` all have engine variants, but the parse
/// side used to collapse them to `Intent::Unknown` while the emit side wrote
/// their real names. A mirror rebuilt from such a board then reported its own
/// correct prediction (`Summon`) as a divergence against the reconstruction's
/// `Unknown` — 61 Summon + 19 Heal records on the v14 sweep, none of which
/// was a real misprediction. Collapsing is only right for `Hidden` /
/// `Unknown`, which are the game's own "you may not see this".
#[test]
fn a_disclosed_telegraph_survives_the_wire() {
    Python::with_gil(|py| {
        for expected in [
            Intent::Summon,
            Intent::Heal,
            Intent::DeathBlow { damage: 30 },
            Intent::Escape,
        ] {
            let mut s1 = build_seeded_state(py, 2, 0).expect("seed=2 must build");
            let uid = *s1.enemy_uids.first().expect("fixture must carry an enemy");
            let idx = s1
                .creatures
                .iter()
                .position(|c| c.id == uid.0)
                .expect("enemy in arena");
            s1.creatures[idx].intent = Some(expected);

            let dict = state_to_dict(py, &s1).expect("to_dict must succeed");
            let s2 = state_from_dict(&dict).expect("from_dict on emitted dict must succeed");
            let uid2 = *s2.enemy_uids.first().expect("enemy survives round-trip");
            let got = s2
                .creatures
                .iter()
                .find(|c| c.id == uid2.0)
                .expect("enemy in arena")
                .intent;
            assert_eq!(
                got,
                Some(expected),
                "{expected:?} must round-trip, not degrade to Unknown"
            );
        }
    });
}

/// A DeathBlow telegraph must be un-baked like the attack it is.
///
/// C# `DeathBlowIntent : SingleAttackIntent : AttackIntent`, and
/// `RunSimulator` serializes `AttackIntent.GetSingleDamage` for any
/// `AttackIntent` — so every C# wire (the `effective` dialect) reports a
/// DeathBlow's damage with the dealer's Strength and the player's
/// Vulnerable/Tainted already folded in. `perform_intent` then runs it through
/// the same `apply_damage_to_player` as `Intent::Attack`, so failing to invert
/// stores the baked number as the engine's *base* and double-counts every
/// modifier.
///
/// Asserted against a plain `Attack` on the identical board, because the claim
/// is precisely that the two are the same shape on this path. `parse_intent`
/// and [`a_disclosed_telegraph_survives_the_wire`] both only exercise the
/// `Base` dialect, where `ingest_intent` short-circuits before the inversion —
/// which is why this went unnoticed when DeathBlow stopped collapsing to
/// `Unknown`.
#[test]
fn a_death_blow_is_unbaked_like_the_attack_it_is() {
    Python::with_gil(|py| {
        let base_of = |telegraph: Intent| -> Option<Intent> {
            let mut s = build_seeded_state(py, 4, 0).expect("seed=4 must build");
            let uid = *s.enemy_uids.first().expect("fixture must carry an enemy");
            let idx = s
                .creatures
                .iter()
                .position(|c| c.id == uid.0)
                .expect("enemy in arena");
            s.creatures[idx].intent = Some(telegraph);

            let dict = state_to_dict(py, &s).expect("to_dict must succeed");
            // Speak the C# dialect: the number on the wire is post-modifier,
            // and the dealer carries the Strength that inflated it.
            dict.set_item("intent_amounts", "effective")
                .expect("dialect is settable");
            let enemy: Bound<'_, PyDict> = dict
                .get_item("encounter")
                .unwrap()
                .expect("encounter present")
                .downcast_into::<PyList>()
                .expect("encounter is a list")
                .get_item(0)
                .expect("one enemy")
                .downcast_into::<PyDict>()
                .expect("enemy is a dict");
            let strength = PyDict::new(py);
            strength.set_item("name", "STRENGTH").unwrap();
            strength.set_item("amount", 4).unwrap();
            enemy
                .set_item("powers", PyList::new(py, [strength]).unwrap())
                .unwrap();

            let back = state_from_dict(&dict).expect("from_dict must succeed");
            let uid2 = *back.enemy_uids.first().expect("enemy survives");
            back.creatures
                .iter()
                .find(|c| c.id == uid2.0)
                .expect("enemy in arena")
                .intent
        };

        assert_eq!(
            base_of(Intent::Attack { damage: 18 }),
            Some(Intent::Attack { damage: 14 }),
            "control: a wire Attack of 18 from a +4-Strength dealer has base 14"
        );
        assert_eq!(
            base_of(Intent::DeathBlow { damage: 18 }),
            Some(Intent::DeathBlow { damage: 14 }),
            "a DeathBlow must invert the same +4, and stay a DeathBlow"
        );
    });
}

/// Build a one-card fixture whose wire costs are set by the caller.
fn cost_fixture(
    py: Python<'_>,
    base: i32,
    persistent: i32,
    played: i32,
    stable: i32,
) -> (CoreCombatState, usize) {
    let dict = make_seeded_fixture(py, 3, 0).expect("seed=3 fixture");
    let deck = dict
        .get_item("deck")
        .unwrap()
        .unwrap()
        .downcast::<PyDict>()
        .unwrap()
        .clone();
    let hand = deck
        .get_item("hand")
        .unwrap()
        .unwrap()
        .downcast::<PyList>()
        .unwrap()
        .clone();
    let card = hand.get_item(0).unwrap();
    let card = card.downcast::<PyDict>().unwrap();
    card.set_item("base_cost", base).unwrap();
    card.set_item("cost_override", base).unwrap();
    card.set_item("persistent_cost", persistent).unwrap();
    card.set_item("played_cost", played).unwrap();
    card.set_item("stable_cost", stable).unwrap();
    let state = state_from_dict(&dict).expect("rebuild must succeed");
    let uid = state.player.hand[0];
    let idx = state.cards.iter().position(|c| c.uid == uid).unwrap();
    (state, idx)
}

/// "This turn" really means "this turn OR until played", so the discount has
/// to die at whichever edge arrives first — and exactly once.
///
/// The two shapes below arrive as the SAME pair of banks (1, 1) and must be
/// treated differently, which is why the wire carries the after-both-edges
/// cost as well. Without it the shared modifier gets restored twice.
#[test]
fn a_this_turn_discount_dies_at_whichever_edge_comes_first() {
    Python::with_gil(|py| {
        // ONE `EndOfTurn | WhenPlayed` modifier: 1 -> 0, and both edges
        // describe the same single shave.
        let (mut s, i) = cost_fixture(py, 0, 1, 1, 1);
        let uid = s.cards[i].uid;
        s.restore_wire_cost_expiring_on_play(uid);
        assert_eq!(s.cards[i].cost, 1, "playing it retires the whole discount");
        s.restore_wire_transient_costs_for_turn_start();
        assert_eq!(
            s.cards[i].cost, 1,
            "and the turn boundary must NOT restore the same shave a second time"
        );

        // The mirror image: the turn ends first.
        let (mut s, i) = cost_fixture(py, 0, 1, 1, 1);
        let uid = s.cards[i].uid;
        s.restore_wire_transient_costs_for_turn_start();
        assert_eq!(s.cards[i].cost, 1);
        s.restore_wire_cost_expiring_on_play(uid);
        assert_eq!(s.cards[i].cost, 1, "same shave, still restored once");

        // TWO independent modifiers on a base-3 card (one EndOfTurn, one
        // WhenPlayed): 3 -> 1. Identical banks, but each edge owns half.
        let (mut s, i) = cost_fixture(py, 1, 2, 2, 3);
        let uid = s.cards[i].uid;
        s.restore_wire_cost_expiring_on_play(uid);
        assert_eq!(s.cards[i].cost, 2, "playing it retires only its own half");
        s.restore_wire_transient_costs_for_turn_start();
        assert_eq!(s.cards[i].cost, 3, "the turn boundary retires the other");
    });
}

/// The free riders are `WhenPlayed`-flagged too: C# `SetToFreeThisTurn` is
/// `EnergyCost.SetThisTurnOrUntilPlayed(0)` and `FreeUntilPlayed` is
/// `SetUntilPlayed(0)`, both dropped by `AfterCardPlayedCleanup` the moment
/// the card is played. Left to die only at end-of-turn cleanup, a generated
/// free card kept reading cost 0 from the exhaust pile after its play — 71
/// `$.deck.exhaust_pile` cost_override desyncs on the v28wr sweep, exposed
/// the day `transplant_hidden` started carrying per-card `combat_vars`.
#[test]
fn the_free_riders_die_at_the_played_edge() {
    use sts2_core::cards::VarKey;
    Python::with_gil(|py| {
        let (mut s, i) = cost_fixture(py, 1, 1, 1, 1);
        let uid = s.cards[i].uid;
        s.cards[i].set_combat_var(VarKey::FreeThisTurn, 1);
        s.cards[i].set_combat_var(VarKey::FreeUntilPlayed, 1);
        s.restore_wire_cost_expiring_on_play(uid);
        assert_eq!(
            s.cards[i].combat_var(VarKey::FreeThisTurn),
            0,
            "SetThisTurnOrUntilPlayed(0) retires when played"
        );
        assert_eq!(
            s.cards[i].combat_var(VarKey::FreeUntilPlayed),
            0,
            "SetUntilPlayed(0) retires when played"
        );
    });
}
