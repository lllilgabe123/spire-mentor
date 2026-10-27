"""Tests for :mod:`realgame_advisor_mirror` — the advisor's carried fight.

The first test is the reason the module exists: it pins, against the real
engine, that answering an in-combat selection off a REBUILT board silently
drops the rest of the card. Everything else here is lifecycle.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from realgame_advisor_mirror import AdvisorMirror, _enemy_hp, fight_key
from sts2_hint.recommender import RecommendConfig

sts2_rs = pytest.importorskip("sts2_rs")

HONEST = RecommendConfig(sims=24, hidden_info="beliefdraw", c_puct=2.5, draw_widening_c=2.0)


def _burning_pact_state(seed: int = 3):
    """Ironclad holding Burning Pact — "Exhaust a card. Draw 2"."""
    return sts2_rs.build_scenario_state(
        seed,
        "Ironclad",
        ["BURNING_PACT"] * 3 + ["STRIKE_IRONCLAD"] * 7,
        80,
        80,
        [],
        "SHRINKER_BEETLE_WEAK",
    )


def _hand_index(state, card_id: str) -> int:
    hand = sts2_rs.state_to_dict(state)["deck"]["hand"]
    return next(i for i, c in enumerate(hand) if c["id"] == card_id)


def _options(state) -> list[dict[str, Any]]:
    return [
        {
            "id": c["id"],
            "upgraded": c.get("upgraded", False),
            "pile": c.get("pile"),
            "index": c.get("index"),
        }
        for c in state.pending_choice()["cards"]
    ]


def _hand_size(state) -> int:
    return len(sts2_rs.state_to_dict(state)["deck"]["hand"])


# ---------------------------------------------------------------------------
# Why the module exists
# ---------------------------------------------------------------------------


def test_injecting_onto_a_rebuild_drops_the_rest_of_the_card() -> None:
    """A choice injected on a rebuilt board loses its `post_effect`.

    Burning Pact is "Exhaust a card THEN draw 2". The engine holds that tail
    on the parked `PendingChoice`; a wire dict cannot express it (note below
    that the rebuilt state has no open choice at all). So the stateless
    advisor could only ever inject a BARE choice, and a search over it would
    rank the exhaust while believing no draw follows — a confident wrong
    answer, which is worse than the leftmost-card default it would replace.

    Carrying the parked state and swapping only the option list in
    (`replace_parked`) keeps the tail. That difference is the whole design.
    """
    # A -- stateless: play, serialize, rebuild, inject, resolve.
    world = _burning_pact_state()
    world.apply_step(
        {"action": "play", "card_index": _hand_index(world, "BURNING_PACT"), "target_index": None}
    )
    options = _options(world)
    rebuilt = sts2_rs.state_from_dict(sts2_rs.state_to_dict(world))
    assert rebuilt.pending_choice() is None, "no wire dialect carries a suspended card"
    rebuilt.open_injected_choice("exhaust_from_hand", options, 1, 1)
    rebuilt.apply_step({"action": "choose", "selection": [0]})

    # B -- carried: keep the parked state, swap the option list in place.
    carried = _burning_pact_state()
    carried.apply_step(
        {
            "action": "play",
            "card_index": _hand_index(carried, "BURNING_PACT"),
            "target_index": None,
        }
    )
    carried.open_injected_choice("exhaust_from_hand", options, 1, 1, False, True)
    carried.apply_step({"action": "choose", "selection": [0]})

    assert _hand_size(rebuilt) == 3, "the rebuild exhausted the card and never drew"
    assert _hand_size(carried) == 5, "the carried state exhausted AND drew 2"


# ---------------------------------------------------------------------------
# Replaying the opening play
# ---------------------------------------------------------------------------


def _advisor_on(world, *, floor: int = 3) -> tuple[AdvisorMirror, dict[str, Any]]:
    """An AdvisorMirror carrying `world`, plus the message envelope used."""
    advisor = AdvisorMirror(HONEST)
    state = {"act": 1, "floor": floor}
    assert advisor.on_combat_board(sts2_rs.state_to_dict(world), state) is not None
    return advisor, state


def test_replay_parks_the_engine_on_its_own_choice() -> None:
    """The advisor never sees the play; it reconstructs it and PROVES it."""
    world = _burning_pact_state()
    advisor, _ = _advisor_on(world)

    # The world plays the card; the advisor only learns a screen is open.
    world.apply_step(
        {"action": "play", "card_index": _hand_index(world, "BURNING_PACT"), "target_index": None}
    )
    landed = advisor.replay_opening_play("BURNING_PACT", sts2_rs.state_to_dict(world))

    assert landed
    assert advisor.live is not None
    parked = advisor.live.state.pending_choice()
    assert parked is not None and parked["kind"] == "exhaust_from_hand"
    assert advisor.stats["replays_landed"] == 1


def test_replay_keeps_the_tail_that_a_rebuild_would_lose() -> None:
    """End to end: the carried mirror resolves Burning Pact's draw."""
    world = _burning_pact_state()
    advisor, _ = _advisor_on(world)
    world.apply_step(
        {"action": "play", "card_index": _hand_index(world, "BURNING_PACT"), "target_index": None}
    )
    assert advisor.replay_opening_play("BURNING_PACT", sts2_rs.state_to_dict(world))

    mirror = advisor.live
    assert mirror is not None
    mirror.inject_choice("exhaust_from_hand", _options(world), 1, 1)
    mirror.advance_through_choice([0])
    assert _hand_size(mirror.state) == 5, "exhaust + draw 2, not exhaust alone"


def test_replay_recovers_which_enemy_a_targeted_card_hit() -> None:
    """Headbutt is targeted, and the target is not on the wire either.

    Two identical beetles, one of them hit: the enemy-HP vector is the only
    thing that says which. 91 of the 1,521 unsearched selections in the 82-run
    live A8 corpus were Headbutt, so this is the common targeted case, and
    replaying it onto the wrong beetle would hand the search a board with the
    damage in the wrong place.
    """
    from simlab.specs import ScenarioSpec, build_combat

    spec = ScenarioSpec(
        encounter="monsters:ShrinkerBeetle,ShrinkerBeetle",
        deck_cards=("Headbutt",) * 4 + ("StrikeIronclad",) * 6,
    )
    world = build_combat(spec, 3)
    # Headbutt puts a card from the DISCARD back on the draw pile, so it needs
    # two there to be a choice at all (one auto-resolves).
    for _ in range(2):
        world.apply_step(
            {
                "action": "play",
                "card_index": _hand_index(world, "STRIKE_IRONCLAD"),
                "target_index": 0,
            }
        )
    advisor, _ = _advisor_on(world)

    assert (
        world.apply_step(
            {"action": "play", "card_index": _hand_index(world, "HEADBUTT"), "target_index": 1}
        )
        == "pending_choice"
    )
    after = sts2_rs.state_to_dict(world)
    assert advisor.replay_opening_play("HEADBUTT", after)

    mirror = advisor.live
    assert mirror is not None
    got = [e["hp"] for e in sts2_rs.state_to_dict(mirror.state)["encounter"]]
    assert got == [e["hp"] for e in after["encounter"]], "the hurt beetle must be the same one"


def test_replay_recovers_a_play_that_KILLED_the_enemy_it_targeted() -> None:
    """The board fingerprint has to survive a corpse only one side reports.

    The engine keeps a dead enemy in `encounter` at `hp: 0, is_dead: true`;
    the live producer drops it from its embedded combat state. A positional
    HP vector over the raw list therefore compares a 2-vector against a
    1-vector and matches NO target, so `replay_opening_play` declined every
    killing play that then opened a screen -- the three `no board matched
    replay of NEOWS_FURY` declines in `logs/f7-mirror-check16` (10-14 damage,
    then a discard-pile select; nothing about it is Neow-specific).

    Here Headbutt kills the beetle it targets and still opens its put-back
    screen, and the world wire is stripped of the corpse exactly as the mod
    reports it.
    """
    from simlab.specs import ScenarioSpec, build_combat

    spec = ScenarioSpec(
        encounter="monsters:ShrinkerBeetle,ShrinkerBeetle",
        deck_cards=("Headbutt",) * 4 + ("StrikeIronclad",) * 6,
    )
    world = build_combat(spec, 3)
    # Beat beetle 1 down to lethal range and stock the discard, so Headbutt
    # both kills and has two cards to choose between (one auto-resolves).
    for _ in range(6):
        while True:
            hand = sts2_rs.state_to_dict(world)["deck"]["hand"]
            i = next((j for j, c in enumerate(hand) if c["id"] == "STRIKE_IRONCLAD"), None)
            if i is None:
                break
            if (
                world.apply_step({"action": "play", "card_index": i, "target_index": 1})
                != "advanced"
            ):
                break
            if sts2_rs.state_to_dict(world)["encounter"][1]["hp"] <= 9:
                break
        d = sts2_rs.state_to_dict(world)
        if d["encounter"][1]["hp"] <= 9 and len(d["deck"]["discard_pile"]) >= 2:
            break
        world.apply_step({"action": "end_turn"})

    advisor, _ = _advisor_on(world)
    assert (
        world.apply_step(
            {"action": "play", "card_index": _hand_index(world, "HEADBUTT"), "target_index": 1}
        )
        == "pending_choice"
    )
    after = sts2_rs.state_to_dict(world)
    assert after["encounter"][1]["is_dead"], "the fixture must actually kill"

    producer = dict(after)
    producer["encounter"] = [e for e in after["encounter"] if not e.get("is_dead")]
    assert advisor.replay_opening_play("HEADBUTT", producer), dict(advisor.declines)
    assert advisor.live is not None
    assert advisor.live.state.pending_choice() is not None


def test_replay_is_refused_when_no_card_matches() -> None:
    """A power/relic/potion screen has no card to replay, so nothing is tried."""
    world = _burning_pact_state()
    advisor, _ = _advisor_on(world)
    assert not advisor.replay_opening_play("WELL_LAID_PLANS_POWER", sts2_rs.state_to_dict(world))
    assert advisor.stats["replays_attempted"] == 0
    assert advisor.live is not None, "declining to replay must not drop the fight"
    assert "WELL_LAID_PLANS_POWER is not a card" in advisor.declines


def test_the_kind_field_settles_what_the_source_is() -> None:
    """A declared kind beats every inference, including a correct one.

    Without it the only test available is absence -- look the id up in the
    hand, then the other piles, and call it "not a card" when neither has it.
    That misfires exactly where it matters: a real card that has fallen out of
    the mirror reports as "not a card", hiding a mirror fault behind a
    category. `source_kind` comes from the model's runtime type, so a power is
    a power even when the piles cannot say so.

    The example is a power rather than a potion because a potion is now
    replayable in its own right (:meth:`replay_opening_drink`). A power opens
    its select from a hook, so there is still no player action to re-perform.
    """
    world = _burning_pact_state()
    advisor, _ = _advisor_on(world)

    # BURNING_PACT *is* in hand, so absence-inference would happily replay it.
    # The kind says otherwise, and the kind wins.
    assert not advisor.replay_opening_play(
        "BURNING_PACT", sts2_rs.state_to_dict(world), source_kind="power"
    )
    assert "BURNING_PACT is a power, not a card or a potion" in advisor.declines
    assert advisor.stats["replays_attempted"] == 0
    assert advisor.live is not None, "declining to replay must not drop the fight"


def test_a_named_target_replaces_the_search_over_targets() -> None:
    """Told which enemy was hit, the replay stops guessing.

    The board check stays, but its job changes: it VERIFIES the named play
    instead of SEARCHING for an unnamed one.
    """
    from simlab.specs import ScenarioSpec, build_combat

    spec = ScenarioSpec(
        encounter="monsters:ShrinkerBeetle,ShrinkerBeetle",
        deck_cards=("Headbutt",) * 4 + ("StrikeIronclad",) * 6,
    )
    world = build_combat(spec, 3)
    for _ in range(2):  # Headbutt needs two cards in the discard to choose
        world.apply_step(
            {
                "action": "play",
                "card_index": _hand_index(world, "STRIKE_IRONCLAD"),
                "target_index": 0,
            }
        )
    advisor, _ = _advisor_on(world)
    world.apply_step(
        {"action": "play", "card_index": _hand_index(world, "HEADBUTT"), "target_index": 1}
    )
    after = sts2_rs.state_to_dict(world)

    assert advisor.replay_opening_play(
        "HEADBUTT", after, source_kind="card", target_index=1, upgraded=False
    )
    mirror = advisor.live
    assert mirror is not None
    got = [e["hp"] for e in sts2_rs.state_to_dict(mirror.state)["encounter"]]
    assert got == [e["hp"] for e in after["encounter"]], "the hurt beetle must be the same one"


def test_an_unofferable_named_play_degrades_to_the_board_proof() -> None:
    """A naming the engine does not offer must not cost the mirror.

    The wire's naming is an OPTIMISATION over proving the play from the board.
    When it does not fit, the two sides may simply disagree about the naming
    -- not decidable from here -- so the honest move is to fall back to the
    proof, which either finds the play anyway or declines harmlessly.

    Live run 001-ironclad-a8-0824-2305 produced one of these on a Sculpting
    Strike: an ordinary targeted attack naming itself through the DECLARED
    `source` argument, i.e. the least ambiguous path there is. Dropping a
    mirror on that premise would reintroduce, from the other side, exactly the
    failure the fork was added to fix.
    """
    world = _burning_pact_state()
    advisor, _ = _advisor_on(world)
    world.apply_step(
        {"action": "play", "card_index": _hand_index(world, "BURNING_PACT"), "target_index": None}
    )
    after = sts2_rs.state_to_dict(world)

    # Burning Pact is UNTARGETED, so naming target 3 is a play the engine
    # never offers. The board proof underneath still identifies it.
    landed = advisor.replay_opening_play(
        "BURNING_PACT", after, source_kind="card", target_index=3, upgraded=False
    )

    assert "BURNING_PACT play named by the wire is not legal in the mirror" in advisor.declines
    assert advisor.live is not None, "an unofferable naming must not drop the fight"
    assert landed, "the board proof underneath still finds the play"


def test_a_named_play_that_does_not_match_drops_rather_than_declines() -> None:
    """Once the play is NAMED, a mismatch is a mirror fault, not an ambiguity.

    Unnamed, "no board matched" means "I cannot tell which play it was" and
    the honest answer is to say nothing. Named, it means "my copy of the fight
    does not reproduce the real board" -- so the mirror is wrong and gets
    dropped, and the next board re-seeds clean instead of the fight limping on
    behind a picture already known to be false.
    """
    from simlab.specs import ScenarioSpec, build_combat

    spec = ScenarioSpec(
        encounter="monsters:ShrinkerBeetle,ShrinkerBeetle",
        deck_cards=("Headbutt",) * 4 + ("StrikeIronclad",) * 6,
    )
    world = build_combat(spec, 3)
    for _ in range(2):
        world.apply_step(
            {
                "action": "play",
                "card_index": _hand_index(world, "STRIKE_IRONCLAD"),
                "target_index": 0,
            }
        )
    advisor, _ = _advisor_on(world)
    world.apply_step(
        {"action": "play", "card_index": _hand_index(world, "HEADBUTT"), "target_index": 1}
    )
    after = sts2_rs.state_to_dict(world)
    # Name the OTHER beetle: a legal play whose board will not match.
    assert not advisor.replay_opening_play(
        "HEADBUTT", after, source_kind="card", target_index=0, upgraded=False
    )
    assert "HEADBUTT replayed but the board did not match" in advisor.declines
    assert advisor.live is None, "a named play that does not reproduce the board is a mirror fault"


def test_a_misplaced_card_declines_differently_from_a_non_card() -> None:
    """The tally must separate "normal" from "the mirror is wrong".

    A potion opening a hand select (`TouchOfInsanity` calls `FromHand` with
    `source: this`) is not a card and never was -- nothing to replay, business
    as usual. A card that IS in this fight but is not in hand means the
    mirror's picture is wrong, and the decline is a symptom of that rather
    than a category of screen. Sharing one message made the first read as the
    second on live run 001-ironclad-a8-0824-2151.
    """
    world = _burning_pact_state()
    # Move every Burning Pact out of the hand, so the id is unambiguously
    # present in the fight and unambiguously absent from the hand. The
    # fixture's two card ids are otherwise both in both piles, which is
    # exactly the ambiguity this test is about.
    board = sts2_rs.state_to_dict(world)
    deck = board["deck"]
    moved = [c for c in deck["hand"] if c["id"] == "BURNING_PACT"]
    assert moved, "fixture must deal a Burning Pact for this to mean anything"
    deck["hand"] = [c for c in deck["hand"] if c["id"] != "BURNING_PACT"]
    deck["discard_pile"] = list(deck.get("discard_pile") or []) + moved

    advisor = AdvisorMirror(HONEST)
    assert advisor.on_combat_board(board, {"act": 1, "floor": 3}) is not None

    advisor.replay_opening_play("TOUCH_OF_INSANITY", board)  # never a card
    advisor.replay_opening_play("BURNING_PACT", board)  # a card, misplaced

    assert "TOUCH_OF_INSANITY is not a card" in advisor.declines
    assert "BURNING_PACT is in the fight but not in hand" in advisor.declines


def test_replay_is_refused_when_the_board_does_not_match() -> None:
    """The match against the world's board is the proof; without it, silence.

    Here the world played the card AND took extra damage the mirror cannot
    account for, so no replay reproduces its enemy-HP vector.
    """
    world = _burning_pact_state()
    advisor, _ = _advisor_on(world)
    world.apply_step(
        {"action": "play", "card_index": _hand_index(world, "BURNING_PACT"), "target_index": None}
    )
    doctored = sts2_rs.state_to_dict(world)
    for enemy in doctored["encounter"]:
        enemy["hp"] = max(1, int(enemy["hp"]) - 7)

    assert not advisor.replay_opening_play("BURNING_PACT", doctored)
    assert advisor.stats["replays_attempted"] == 1
    assert advisor.stats["replays_landed"] == 0


def test_replay_is_a_noop_when_already_parked() -> None:
    world = _burning_pact_state()
    advisor, _ = _advisor_on(world)
    world.apply_step(
        {"action": "play", "card_index": _hand_index(world, "BURNING_PACT"), "target_index": None}
    )
    wire = sts2_rs.state_to_dict(world)
    assert advisor.replay_opening_play("BURNING_PACT", wire)
    assert advisor.replay_opening_play("BURNING_PACT", wire), "re-ask must stay parked"
    assert advisor.stats["replays_landed"] == 1, "and must not replay a second card"


# ---------------------------------------------------------------------------
# Following the fight (the thing that makes a mirror a mirror)
# ---------------------------------------------------------------------------


def _played(card_id: str, target: int | None = None) -> dict[str, Any]:
    return {
        "type": "human_action",
        "decision_type": "combat_action",
        "action": "card_play",
        "detail": f"IRONCLAD played {card_id}",
        "card": {"id": card_id, "upgraded": False},
        "target_index": target,
    }


def test_following_a_play_keeps_the_mirror_in_step() -> None:
    """Advanced through the play, the mirror PREDICTS the next board.

    Without this the advisor could only re-seed from each new board, so every
    ordinary change read as a divergence — 2232 desyncs over 540 decisions on
    live run 001-ironclad-a8-0824-1939, 42x the autopilot's rate.
    """
    world = _burning_pact_state()
    advisor, state = _advisor_on(world)

    world.apply_step(
        {
            "action": "play",
            "card_index": _hand_index(world, "STRIKE_IRONCLAD"),
            "target_index": 0,
        }
    )
    advisor.on_world_action(_played("STRIKE_IRONCLAD", 0))

    assert advisor.stats["followed"] == 1
    assert advisor.live is not None
    # The payoff: reconciling the world's next board finds NOTHING to report.
    assert advisor.on_combat_board(sts2_rs.state_to_dict(world), state) is not None
    assert advisor.stats["desyncs"] == 0


def test_an_unfollowable_action_drops_rather_than_drifts() -> None:
    """An action with no machine form still invalidates the mirror.

    Both cases are now older-mod-build shapes: a current build carries `card`
    for a play and `potion` for a drink. What is pinned is the response to
    neither being there — drop, rather than drift.
    """
    for msg in (
        {"type": "human_action", "decision_type": "combat_action", "action": "potion"},
        {"type": "human_action", "decision_type": "combat_action", "action": "card_play"},
    ):
        advisor, _ = _advisor_on(_burning_pact_state())
        advisor.on_world_action(msg)
        assert advisor.live is None
        assert advisor.stats["unfollowable"] == 1


def test_end_turn_report_does_not_drop_the_mirror() -> None:
    """The coach's end-turn report (protocol v5) is a no-op for the mirror.

    It carries no card and no potion, which used to read as unfollowable and
    DROP the mirror. The turn boundary is already followed from the next board
    (`_catch_up_turns`), so following the report too would end the turn twice.
    """
    advisor, _ = _advisor_on(_burning_pact_state())
    advisor.on_world_action(
        {
            "type": "human_action",
            "decision_type": "combat_action",
            "action": "end_turn",
            "detail": "End turn",
        }
    )
    assert advisor.live is not None
    assert advisor.stats["unfollowable"] == 0
    assert advisor.stats["followed"] == 0


def test_the_late_echo_of_a_replayed_play_does_not_drop_a_good_mirror() -> None:
    """The two follow mechanisms collide once per searched select.

    Answering a `card_select` REPLAYS the opening play, then drops the mirror
    (it is parked on an injection nothing will resolve). The mirror re-seeds
    from the next board, which already includes that play. Only THEN does the
    world report it, because `CardPlayFinishedEntry` fires when a play
    finishes and a choice-opening card finishes only once the selection is
    answered.

    Following that report would drop a freshly-seeded, correct mirror for a
    card it rightly no longer holds -- two re-seeds per searched select, one
    of them pure waste.
    """
    world = _burning_pact_state()
    advisor, state = _advisor_on(world)
    world.apply_step(
        {"action": "play", "card_index": _hand_index(world, "BURNING_PACT"), "target_index": None}
    )
    assert advisor.replay_opening_play("BURNING_PACT", sts2_rs.state_to_dict(world))

    advisor.drop("answered a card_select")  # what the advisor's `finally` does
    world.apply_step({"action": "choose", "selection": [0]})
    assert advisor.on_combat_board(sts2_rs.state_to_dict(world), state) is not None

    advisor.on_world_action(_played("BURNING_PACT"))

    assert advisor.live is not None, "the echo must not drop the re-seeded mirror"
    assert advisor.stats["stale_reports_skipped"] == 1
    assert advisor.stats["unfollowable"] == 0


def test_the_echo_suppression_is_consumed_once() -> None:
    """A genuine SECOND play of the same card is still followed.

    The suppression is a one-shot for a specific echo, not a standing licence
    to ignore a card id.
    """
    world = _burning_pact_state()
    advisor, state = _advisor_on(world)
    world.apply_step(
        {"action": "play", "card_index": _hand_index(world, "BURNING_PACT"), "target_index": None}
    )
    assert advisor.replay_opening_play("BURNING_PACT", sts2_rs.state_to_dict(world))
    advisor.drop("answered a card_select")
    world.apply_step({"action": "choose", "selection": [0]})
    advisor.on_combat_board(sts2_rs.state_to_dict(world), state)

    advisor.on_world_action(_played("BURNING_PACT"))  # the echo: skipped
    # The hand really does still hold a second copy, so "followed" here is a
    # genuine advance and not a drop dressed up as one.
    assert any(
        c["id"] == "BURNING_PACT"
        for c in sts2_rs.state_to_dict(advisor.live.state)["deck"]["hand"]  # type: ignore[union-attr]
    )
    advisor.on_world_action(_played("BURNING_PACT"))  # a real play: followed

    assert advisor.stats["stale_reports_skipped"] == 1
    assert advisor.stats["followed"] == 1, "the second report must be acted on, not swallowed"
    assert advisor.live is not None


def test_an_unreseeded_replay_still_follows_its_own_report() -> None:
    """Guarded on the seed generation, not just the card id.

    If the caller did NOT drop and re-seed after the replay, the mirror is
    still the one that played the card and the world's report is not an echo
    of a state it already has -- swallowing it there would silently put the
    mirror a card behind, which is the exact failure this class exists to
    prevent. Only a rebuild since the replay licenses the skip.
    """
    world = _burning_pact_state()
    advisor, _ = _advisor_on(world)
    world.apply_step(
        {"action": "play", "card_index": _hand_index(world, "BURNING_PACT"), "target_index": None}
    )
    assert advisor.replay_opening_play("BURNING_PACT", sts2_rs.state_to_dict(world))

    advisor.on_world_action(_played("BURNING_PACT"))  # no drop, no re-seed

    assert advisor.stats["stale_reports_skipped"] == 0


# ---------------------------------------------------------------------------
# Potions
#
# A potion opens a card-select through a different engine action than a card
# does, and used to be answered by the rarity sort for exactly that reason.
# Worse, the drink itself had no machine form on the wire, so a fight in which
# anything was drunk lost its mirror at that moment — and with it every
# searched decision for the rest of the fight.
# ---------------------------------------------------------------------------


def _potion_state(potion: str, seed: int = 3):
    """The Burning Pact fixture, plus one potion in the belt."""
    return sts2_rs.build_scenario_state(
        seed,
        "Ironclad",
        ["BURNING_PACT"] * 3 + ["STRIKE_IRONCLAD"] * 7,
        80,
        80,
        [],
        "SHRINKER_BEETLE_WEAK",
        0,
        None,
        [potion],
    )


def _drink(state, potion: str) -> dict[str, Any]:
    """The engine's own drink action for `potion`, in the world's copy."""
    slot = sts2_rs.state_to_dict(state)["player"]["potions"].index(potion)
    return next(
        a
        for a in state.legal_actions()
        if a.get("action") in ("drink_potion", "potion") and a.get("slot") == slot
    )


def _drank(potion_id: str, target: int | None = None) -> dict[str, Any]:
    """A `human_action` for a drink, shaped as the mod now sends it."""
    return {
        "type": "human_action",
        "decision_type": "combat_action",
        "action": "card_play",
        "detail": f"IRONCLAD drank {potion_id}",
        "card": None,
        "potion": {"id": potion_id},
        "target_index": target,
    }


@pytest.mark.parametrize(
    "potion",
    ["ATTACK_POTION", "SKILL_POTION", "POWER_POTION", "COLORLESS_POTION", "TOUCH_OF_INSANITY"],
)
def test_a_potion_screen_is_answered_by_replaying_the_drink(potion: str) -> None:
    """The screen the rarity sort used to answer is now a searchable board."""
    world = _potion_state(potion)
    advisor, _ = _advisor_on(world)

    assert world.apply_step(_drink(world, potion)) == "pending_choice"
    landed = advisor.replay_opening_drink(potion, sts2_rs.state_to_dict(world))

    assert landed
    assert advisor.live is not None
    assert advisor.live.state.pending_choice() is not None
    assert advisor.stats["replays_landed"] == 1


def test_a_potion_is_replayable_where_a_relic_is_not() -> None:
    """The kind gate admits a potion and still refuses what cannot be replayed.

    A relic fires from a hook, not from a player action, so there is no drink
    or play to re-perform — it has to stay declined.
    """
    advisor, _ = _advisor_on(_burning_pact_state())

    assert advisor.source_is_replayable("ATTACK_POTION", "potion")
    assert advisor.source_is_replayable("BURNING_PACT", "card")
    assert not advisor.source_is_replayable("BOTTLED_LIGHTNING", "relic")
    assert "BOTTLED_LIGHTNING is a relic, not a card or a potion" in advisor.declines


def test_a_potion_that_is_not_in_the_belt_is_declined() -> None:
    """No belt entry means no drink to replay, and nothing to guess from."""
    world = _potion_state("ATTACK_POTION")
    advisor, _ = _advisor_on(world)

    assert not advisor.replay_opening_drink("SKILL_POTION", sts2_rs.state_to_dict(world))
    assert advisor.live is not None, "an unanswerable screen must not cost the mirror"
    assert "SKILL_POTION is not drinkable in the mirror" in advisor.declines


def test_a_drink_that_opens_no_choice_is_declined() -> None:
    """Liquid Memories with an empty discard pile has nothing to offer.

    The potion IS in the belt and IS drinkable, so this is not the belt check
    firing — it is the engine reporting that the drink resolved outright. That
    distinction is the whole reason the parked-choice result is checked
    separately from whether the drink was possible.
    """
    world = _potion_state("LIQUID_MEMORIES")
    advisor, _ = _advisor_on(world)

    assert world.apply_step(_drink(world, "LIQUID_MEMORIES")) == "advanced"
    assert not advisor.replay_opening_drink("LIQUID_MEMORIES", sts2_rs.state_to_dict(world))
    assert advisor.live is not None
    assert any("parked no choice" in d for d in advisor.declines)


def test_a_drink_replay_on_a_mismatched_board_declines_without_dropping() -> None:
    """The board proof guards the search; it does not condemn the mirror.

    The potion was named by the wire, so a mismatch cannot mean "wrong potion"
    — it means this copy of the fight drifted, which `reconcile` re-seeds on
    the next board anyway. Dropping here would destroy a mirror on the
    strength of an enemy-HP check that an untargeted potion cannot even move.
    """
    world = _potion_state("ATTACK_POTION")
    advisor, _ = _advisor_on(world)
    world.apply_step(_drink(world, "ATTACK_POTION"))

    wire = sts2_rs.state_to_dict(world)
    for enemy in wire["encounter"]:
        enemy["hp"] = max(1, int(enemy["hp"]) - 7)  # damage the mirror cannot account for

    assert not advisor.replay_opening_drink("ATTACK_POTION", wire)
    assert advisor.live is not None, "a drifted mirror re-seeds; it is not destroyed here"
    assert "ATTACK_POTION drunk but the board did not match" in advisor.declines


def test_following_a_drink_keeps_the_mirror_in_step() -> None:
    """A drink used to be the one action that always cost the fight its mirror."""
    world = _potion_state("BLOCK_POTION")
    advisor, state = _advisor_on(world)

    assert world.apply_step(_drink(world, "BLOCK_POTION")) == "advanced"
    advisor.on_world_action(_drank("BLOCK_POTION"))

    assert advisor.live is not None, "the drink is followable now, not a drop"
    assert advisor.stats["followed"] == 1
    assert advisor.stats["unfollowable"] == 0
    # The payoff, same as for a card: the world's next board reports NOTHING.
    assert advisor.on_combat_board(sts2_rs.state_to_dict(world), state) is not None
    assert advisor.stats["desyncs"] == 0


def _potion_screen(potion: str = "ATTACK_POTION") -> tuple[AdvisorMirror, dict[str, Any]]:
    """An advisor at a real potion-opened `card_select`, and that screen.

    Two boards, because the drink happens between them: the advisor last saw
    the potion IN the belt, and the screen carries the board after it was
    consumed. Replaying the drink is what closes that gap.
    """
    import copy
    import json

    from realgame_adapter import to_headless_combat_play
    from sts2_hint import adapter

    fixture = Path(__file__).parent / "fixtures" / "realgame_adapter" / "single_enemy_ironclad.json"
    base = json.loads(fixture.read_text(encoding="utf-8"))["state"]

    before = copy.deepcopy(base)
    before["potions"][0] = {"slot": 0, "empty": False, "id": potion, "name": "Attack Potion"}
    before["run_state"]["potions"] = before["potions"]
    wire = adapter.translate(
        to_headless_combat_play(before), character="Ironclad", combat_id=1, seed=1, ascension=0
    )
    advisor = AdvisorMirror(HONEST)
    assert advisor.on_combat_board(wire, {"act": before["act"], "floor": before["floor"]})

    after = copy.deepcopy(base)  # the fixture's belt is already empty
    return advisor, {
        "type": "card_select",
        "screen": "NCombatPileCardSelectScreen",
        "prompt_key": f"{potion}.selectionScreenPrompt",
        "source_model": potion,
        "source_kind": "potion",
        "cards": [
            {"id": "BASH", "name": "Bash", "index": 0},
            {"id": "HEADBUTT", "name": "Headbutt", "index": 1},
            {"id": "STRIKE_IRONCLAD", "name": "Strike", "index": 2},
        ],
        "min_select": 1,
        "max_select": 1,
        "combat": after,
        "run_state": after["run_state"],
    }


def test_a_card_select_answer_rebuilds_at_the_run_ascension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GH #848: the answer's rebuild reads the run's ascension, not A0."""
    from sts2_hint import adapter

    advisor, screen = _potion_screen()
    screen["run_state"]["ascension"] = 10
    seen: list[int] = []
    real = adapter.translate

    def spy(*args: Any, **kwargs: Any) -> dict[str, Any]:
        seen.append(kwargs.get("ascension", 0))
        return real(*args, **kwargs)

    monkeypatch.setattr(adapter, "translate", spy)
    assert advisor.answer_card_select(screen, HONEST, 8) is not None
    assert seen and set(seen) == {10}


def test_a_potion_screen_is_searched_end_to_end() -> None:
    """The whole chain, on the screen class that was 100% rarity-sorted.

    Every earlier test here proves one link. This one proves they connect:
    seed, fork, replay the drink, reconcile without losing the park, inject
    the world's three offers, search, and answer. Each link failed separately
    on the way in -- the replay landed and the reconcile then deleted the park,
    which read as an injection fault three calls downstream.
    """

    advisor, screen = _potion_screen()
    advice = advisor.answer_card_select(screen, HONEST, 8)

    assert advice is not None, "a potion screen is answerable now"
    assert advice["note"].startswith("combat MCTS"), advice["note"]
    assert advice["best"]["action"] == "choose"
    assert advice["best"]["index"] in {0, 1, 2}
    assert sum(1 for o in advice["options"] if o["is_best"]) == 1
    assert advisor.stats["searched_selects"] == 1
    assert advisor.stats["replays_landed"] == 1
    assert not advisor.declines, advisor.declines
    assert advisor.live is not None, "answering is side-effect free on the fight"


# ---------------------------------------------------------------------------
# Hook-opened screens
#
# A relic fires from a hook, so there is no play or drink to re-issue — and
# the replayability gate turned the whole class away before a fork was ever
# built. Gambling Chip's start-of-combat mulligan was 15 of the 17 remaining
# unsearched selections in `logs/f7-mirror-check19`, every one of them
# answered by the rarity sort.
# ---------------------------------------------------------------------------


def _gambling_chip_screen():
    """Turn-1 mulligan: choose any number of the hand to discard and redraw.

    Shaped exactly as `combat_mirror.recognize_card_select` keys on it --
    round 1, `min_select` 0, and an option multiset equal to the mirror's
    whole hand -- because that recognition is what turns the screen into the
    engine's real `discard_from_hand_and_draw` kind.
    """
    import copy
    import json

    from realgame_adapter import to_headless_combat_play
    from sts2_hint import adapter

    fixture = Path(__file__).parent / "fixtures" / "realgame_adapter" / "single_enemy_ironclad.json"
    base = copy.deepcopy(json.loads(fixture.read_text(encoding="utf-8"))["state"])
    # The headless board rebuilds `player.relics` from `run_state`, so the
    # relic has to go in there to reach the recognition.
    base["run_state"]["relics"].append({"id": "GAMBLING_CHIP", "name": "Gambling Chip"})

    wire = adapter.translate(
        to_headless_combat_play(base), character="Ironclad", combat_id=1, seed=1, ascension=0
    )
    advisor = AdvisorMirror(HONEST)
    assert advisor.on_combat_board(wire, {"act": base["act"], "floor": base["floor"]})

    cards = [
        {"id": c["id"], "name": str(c["id"]).title(), "index": i}
        for i, c in enumerate(base["hand"])
    ]
    return advisor, {
        "type": "card_select",
        "screen": "NCombatPileCardSelectScreen",
        "prompt_key": "GAMBLING_CHIP.selectionScreenPrompt",
        "source_model": "GAMBLING_CHIP",
        "source_kind": "relic",
        "cards": cards,
        "min_select": 0,
        "max_select": len(cards),
        "combat": base,
        "run_state": base["run_state"],
    }


def test_a_relic_screen_is_searched_end_to_end() -> None:
    """The largest remaining unsearched class, answered by the search.

    Nothing here is new machinery: `recognize_card_select` already mapped
    this screen to `discard_from_hand_and_draw`, `inject_choice` already
    injected it, and `arm_turn_start_resume` already parked the rest of the
    turn-start batch behind the pick -- the DLL driver has taken this path all
    along. Only the advisor's replayability gate stood in front of it.
    """
    advisor, screen = _gambling_chip_screen()
    advice = advisor.answer_card_select(screen, HONEST, 8)

    assert advice is not None, "a relic-opened screen is answerable now"
    assert advice["note"].startswith("combat MCTS"), advice["note"]
    assert advisor.stats["searched_selects"] == 1
    # "Choose ANY number" includes none, so an empty pick is a real answer --
    # what must not happen is a decline or the rarity sort.
    assert not advisor.declines, advisor.declines
    assert advisor.live is not None, "answering is side-effect free on the fight"
    # No replay was attempted: there is no relic play to re-issue.
    assert advisor.stats["replays_landed"] == 0


def test_the_replayability_gate_still_refuses_what_it_always_did() -> None:
    """`source_is_replayable` is unchanged -- the advisor routes around it.

    The distinction matters: the gate answers "can this be REPLAYED", and for
    a relic the answer is still no. What changed is that a hook-opened screen
    no longer needs a replay to be answered, so the gate is not consulted for
    it. Asked directly, it must still say no.
    """
    advisor, _ = _advisor_on(_burning_pact_state())
    assert not advisor.source_is_replayable("GAMBLING_CHIP", "relic")


# ---------------------------------------------------------------------------
# Already parked
#
# The best case the rung has, and it was unreachable: the replayability gate
# ran before anything looked at the mirror, so a fight already sitting on the
# right choice was turned away over what had opened it.
# ---------------------------------------------------------------------------


def _well_laid_plans():
    """Silent, all Well-Laid Plans: end the turn and the engine parks itself."""
    return sts2_rs.build_scenario_state(
        3, "Silent", ["WELL_LAID_PLANS"] * 10, 70, 70, [], "SHRINKER_BEETLE_WEAK"
    )


def test_a_fight_already_on_the_choice_needs_no_replay() -> None:
    """A power's own retain select, which the mirror reached by itself.

    Well-Laid Plans retains a card at end of turn and the engine models it
    (`pending_wlp_retain_resume` -> `retain_at_turn_end`), so a mirror that
    followed the turn end is ALREADY on the screen the world is asking about.
    Nothing needs replaying, and what opened it is therefore irrelevant.

    It was declined as "a power, not a card": 246 of the 3,020 in-combat
    selections in the 82-run `logs/2026-08-12-live-a8` corpus, 8x every potion
    combined, and the single largest named source in it.
    """
    world = _well_laid_plans()
    advisor, _ = _advisor_on(world)
    live = advisor.live
    assert live is not None

    live.advance({"action": "play", "card_index": 0, "target_index": None})
    assert live.advance({"action": "end_turn"}) == "pending_choice"
    assert live.state.pending_choice()["kind"] == "retain_at_turn_end"

    # The source is a POWER -- unreplayable, and beside the point.
    assert not advisor.source_is_replayable("WELL_LAID_PLANS_POWER", "power")
    advisor.declines.clear()
    advisor._said.clear()

    assert advisor.replay_opening_play(
        "WELL_LAID_PLANS_POWER", sts2_rs.state_to_dict(world), source_kind="power"
    ), "an already-parked fight is answerable whatever opened the screen"
    assert not advisor.declines, advisor.declines
    assert advisor.stats["replays_attempted"] == 0, "nothing was replayed, so nothing was tried"


def test_an_inherited_park_survives_the_fork_and_the_reconcile() -> None:
    """The step that silently broke the potion path, checked here up front.

    `pick_card_select_mirrored` opens with a `reconcile`, so a park that does
    not survive it is worth nothing however correctly it was reached -- and
    the failure surfaces calls later as an injection the engine cannot locate.
    Well-Laid Plans survives because `pending_wlp_retain_resume` is already in
    `has_suspended_walk`; the potion marker was in nothing, which is what cost
    that path three layers of debugging.
    """
    world = _well_laid_plans()
    advisor, _ = _advisor_on(world)
    live = advisor.live
    assert live is not None

    for step in ({"action": "play", "card_index": 0, "target_index": None}, {"action": "end_turn"}):
        world.apply_step(step)
        live.advance(step)
    assert world.pending_choice()["kind"] == "retain_at_turn_end"

    fork = live.fork("answer")
    assert fork is not None
    assert fork.state.pending_choice() is not None, "a fork inherits the park"
    assert fork.reconcile(sts2_rs.state_to_dict(world)) == []
    assert fork.state.pending_choice() is not None, "and the reconcile keeps it"


def test_an_already_parked_fight_skips_the_drink_replay_too() -> None:
    """Same short-circuit on the potion path, checked before the name."""
    world = _well_laid_plans()
    advisor, _ = _advisor_on(world)
    live = advisor.live
    assert live is not None
    live.advance({"action": "play", "card_index": 0, "target_index": None})
    assert live.advance({"action": "end_turn"}) == "pending_choice"

    assert advisor.replay_opening_drink(None, sts2_rs.state_to_dict(world))
    assert not advisor.declines, advisor.declines
    assert advisor.stats["replays_attempted"] == 0


def test_a_potion_the_mirror_does_not_hold_drops_it() -> None:
    """Following a drink we cannot perform would desync silently."""
    advisor, _ = _advisor_on(_potion_state("BLOCK_POTION"))
    advisor.on_world_action(_drank("ENERGY_POTION"))
    assert advisor.live is None
    assert advisor.stats["unfollowable"] == 1


def test_a_card_the_mirror_does_not_hold_drops_it() -> None:
    advisor, _ = _advisor_on(_burning_pact_state())
    advisor.on_world_action(_played("BASH", 0))
    assert advisor.live is None, "following a card we do not hold would desync silently"


def test_non_combat_actions_are_ignored() -> None:
    advisor, _ = _advisor_on(_burning_pact_state())
    advisor.on_world_action({"type": "human_action", "decision_type": "map_select"})
    assert advisor.live is not None
    assert advisor.stats["unfollowable"] == 0


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_fight_key_separates_rooms() -> None:
    assert fight_key({"act": 1, "floor": 3}) != fight_key({"act": 1, "floor": 4})
    assert fight_key({"act": 1, "floor": 3}) != fight_key({"act": 2, "floor": 3})
    assert fight_key({"act": 1, "floor": 3}) == fight_key({"act": 1, "floor": 3, "hp": 9})


def test_a_new_floor_drops_the_carried_fight() -> None:
    world = _burning_pact_state()
    advisor, _ = _advisor_on(world, floor=3)
    first = advisor.live
    advisor.on_combat_board(sts2_rs.state_to_dict(world), {"act": 1, "floor": 4})
    assert advisor.live is not None
    assert advisor.live is not first, "a new room is a new fight"
    assert advisor.stats["seeds"] == 2


@pytest.mark.parametrize("mtype", ["snapshot_restored", "combat_undone"])
def test_a_rewind_drops_the_carried_fight(mtype: str) -> None:
    """Both rewind broadcasts invalidate a carried board (PROTOCOL.md §7).

    `combat_undone` is the one that matters: the fight keeps running while the
    board moves backwards, so a mirror that survived it would be wrong for the
    rest of the fight without ever saying so.
    """
    world = _burning_pact_state()
    advisor, _ = _advisor_on(world)
    advisor.on_rewind({"type": mtype, "state_id": 999})
    assert advisor.live is None
    assert advisor.stats["drops"] == 1


def test_disabled_advisor_never_carries_anything() -> None:
    """`--no-mirror` restores the old stateless behaviour exactly."""
    world = _burning_pact_state()
    advisor = AdvisorMirror(HONEST, enabled=False)
    assert advisor.on_combat_board(sts2_rs.state_to_dict(world), {"act": 1, "floor": 3}) is None
    assert advisor.live is None
    assert not advisor.replay_opening_play("BURNING_PACT", sts2_rs.state_to_dict(world))


# ---------------------------------------------------------------------------
# The advisor's fallback ladder must be unchanged where the mirror cannot help
# ---------------------------------------------------------------------------


def _deck_screen_state() -> dict[str, Any]:
    """A run-level card_select — a deck screen, with no fight behind it."""
    return {
        "type": "card_select",
        "prompt_key": "GAMBLERS_BREW.selectionScreenPrompt",
        "source_model": "GAMBLERS_BREW",
        "cards": [
            {"id": "STRIKE_IRONCLAD", "name": "Strike", "index": 0},
            {"id": "CLUMSY", "name": "Clumsy", "index": 1},
        ],
        "min_select": 1,
        "max_select": 1,
        "run_state": {"character": "Ironclad", "master_deck": [{"id": "STRIKE_IRONCLAD"}]},
    }


def test_a_screen_with_no_fight_behind_it_is_never_searched() -> None:
    """No embedded `combat` means a deck screen: the net's surface, not ours."""

    world = _burning_pact_state()
    advisor, _ = _advisor_on(world)
    assert advisor.answer_card_select(_deck_screen_state(), HONEST, 8) is None


def test_an_unanswerable_screen_does_not_cost_the_fight_its_mirror() -> None:
    """The regression this fork exists for.

    A power opens a combat select. The advisor cannot replay a power -- it
    fires from a hook, not from a player action -- so it declines, correctly.
    What must NOT follow is the rest of the answer path
    running anyway on the live fight, injecting an option list the engine
    cannot locate, and dropping the mirror as a side effect.

    That is the chain live run 001-ironclad-a8-0824-2239 recorded in three
    consecutive lines: `DROPLET_OF_PRECOGNITION is a potion, not a card`, then
    `injection dropped the mirror`, then `no live mirror at the screen` when
    the very same screen was re-asked. One unanswerable screen cost the fight
    its mirror and every decision after it.

    The live example was a potion, which is answerable now; the mechanism it
    exposed is not potion-specific, so the test keeps it with a source that is
    still unanswerable. That source has had to move twice as the rung grew --
    first off `power` (which now routes to the end-of-turn replay), then off
    `relic` (which now routes to the hook-sourced injection, the Gambling Chip
    class). What is left is C#'s catch-all `AbstractModel`, reported as
    `"model"`: nothing knows how to reach a screen one of those opened, so the
    gate still refuses it and this invariant still has something to guard.
    """

    world = _burning_pact_state()
    advisor, _ = _advisor_on(world)
    board = sts2_rs.state_to_dict(world)

    state = {
        "type": "card_select",
        "screen": "NCombatPileCardSelectScreen",
        "prompt_key": "SOME_MODEL.selectionScreenPrompt",
        "source_model": "SOME_MODEL",
        "source_kind": "model",
        "cards": [
            {"id": c["id"], "name": c["id"], "index": i}
            for i, c in enumerate(board["deck"]["draw_pile"][:2])
        ],
        "min_select": 1,
        "max_select": 1,
        "combat": {"hand": []},
        "run_state": {"character": "Ironclad", "master_deck": []},
    }

    # It declines -- that part is correct and expected.
    assert advisor.answer_card_select(state, HONEST, 8) is None
    assert "SOME_MODEL is a model, not a card or a potion" in advisor.declines
    # ... and the fight survives it, which is the fix.
    assert advisor.live is not None, "an unanswerable screen must not drop the fight"
    # Re-asking the same screen still finds a mirror, instead of the
    # `no live mirror at the screen` that followed it live.
    assert advisor.answer_card_select(state, HONEST, 8) is None
    assert advisor.live is not None
    assert "no live mirror at the screen" not in advisor.declines


def test_answering_a_select_leaves_the_fight_where_it_was() -> None:
    """The live mirror is not advanced, parked, or dropped by an answer.

    It stays at the PRE-play board, so the world's late `human_action` for the
    card that opened the screen is followed normally -- no drop, no re-seed,
    and none of the counters a re-seed would have cost.
    """
    world = _burning_pact_state()
    advisor, _ = _advisor_on(world)
    live = advisor.live
    assert live is not None
    before = live.state.state_hash()

    world.apply_step(
        {"action": "play", "card_index": _hand_index(world, "BURNING_PACT"), "target_index": None}
    )
    fork = live.fork("answer")
    assert fork is not None
    assert advisor.replay_opening_play("BURNING_PACT", sts2_rs.state_to_dict(world), on=fork)
    fork.drop("answered")

    assert advisor.live is live, "the fight is the same object"
    assert live.alive
    assert live.state.state_hash() == before, "the fight did not move"
    assert live.state.pending_choice() is None, "the fight is not left parked"


def test_no_mirror_leaves_the_old_ladder_exactly_as_it_was() -> None:
    """Every reason not to search returns None, so the heuristic rung runs."""
    import realgame_advisor as ra

    state = _deck_screen_state()
    state["combat"] = {"hand": []}  # in-combat shape, but nothing carried
    assert ra._searched_card_select_advice(state, None, HONEST, 8) is None
    assert AdvisorMirror(HONEST).answer_card_select(state, HONEST, 8) is None
    assert AdvisorMirror(HONEST, enabled=False).answer_card_select(state, HONEST, 8) is None
    # ... and with no cfg (a caller that cannot search at all).
    world = _burning_pact_state()
    advisor, _ = _advisor_on(world)
    assert advisor.answer_card_select(state, None, 8) is None


def test_drop_is_total_and_repeatable() -> None:
    """Nothing here may raise into the advisor's serve loop."""
    advisor = AdvisorMirror(HONEST)
    advisor.drop("never seeded")  # no mirror yet
    world = _burning_pact_state()
    advisor.on_combat_board(sts2_rs.state_to_dict(world), {"act": 1, "floor": 3})
    advisor.drop("once")
    advisor.drop("twice")
    assert advisor.live is None
    assert "mirror:" in advisor.note()


# ---------------------------------------------------------------------------
# A park the world has finished with
# ---------------------------------------------------------------------------


def _parked_advisor() -> tuple[AdvisorMirror, dict[str, Any]]:
    """An advisor whose LIVE mirror is parked on a selection, and a board.

    Parked the way the live lane gets there -- by the engine's own end-of-turn
    pipeline, not by hand -- using Well-Laid Plans, whose retain select
    suspends the flush (`combat_loop.rs`, `pending_wlp_retain_resume`). Any
    park would do; this one is reachable without a card that opens a screen.
    """
    import copy
    import json

    from realgame_adapter import to_headless_combat_play
    from sts2_hint import adapter

    fixture = Path(__file__).parent / "fixtures" / "realgame_adapter" / "single_enemy_ironclad.json"
    base = json.loads(fixture.read_text(encoding="utf-8"))["state"]
    board = copy.deepcopy(base)
    board["player"]["powers"] = [{"id": "WELL_LAID_PLANS", "name": "Well-Laid Plans", "amount": 1}]
    wire = adapter.translate(
        to_headless_combat_play(board), character="Ironclad", combat_id=1, seed=1, ascension=0
    )
    advisor = AdvisorMirror(HONEST)
    live = advisor.on_combat_board(wire, {"act": board["act"], "floor": board["floor"]})
    assert live is not None
    assert live.advance({"action": "end_turn"}) == "pending_choice"
    return advisor, board


def test_a_park_the_world_has_finished_with_is_dropped_not_searched() -> None:
    """A combat request against a parked mirror must not answer with `choose`.

    THE floor-47 wedge (live run 004-silent-a8-0825-0139, three attempts).
    A card that opens a selection reports LATE, so following it parks the
    mirror on a choice the world has already answered; a reshuffle on the way
    to that screen makes `reconcile`'s parked verify skip `$.deck.` and the
    stale park survives forever. Every combat search then returns `choose`
    actions, which have no executable form on a combat screen, so the advice
    ships with no `best`, the mod waits, and the stall guard kills the run.

    The mirror is worth less than the run: drop it and rebuild.
    """
    import realgame_advisor as ra

    advisor, board = _parked_advisor()
    assert advisor.live is not None and advisor.live.state.pending_choice() is not None

    advice, _cs, _cfg, _rows = ra.compute_combat_advice(board, HONEST, 24, advisor)

    assert advisor.live is None, "a stale park must cost the mirror, not the run"
    assert "best" in advice, "the rebuild must still produce an executable action"
    assert "no executable form" not in advice["note"], advice["note"]
    assert not advice["best"]["action"].startswith("choose")


# ---------------------------------------------------------------------------
# A screen the end of turn opened
# ---------------------------------------------------------------------------


def _wlp_board(amount: int = 1) -> dict[str, Any]:
    """The fixture board with a Well-Laid Plans power on the player.

    The `name` key is not optional: `sts2_hint.adapter._translate_power`
    `_require`s it, and `_soft_translate_list` DROPS a power that raises --
    silently, so the board comes back with no power at all and the test passes
    for the wrong reason.
    """
    import copy
    import json

    fixture = Path(__file__).parent / "fixtures" / "realgame_adapter" / "single_enemy_ironclad.json"
    board = copy.deepcopy(json.loads(fixture.read_text(encoding="utf-8"))["state"])
    board["player"]["powers"] = [
        {"id": "WELL_LAID_PLANS", "name": "Well-Laid Plans", "amount": amount}
    ]
    return board


def _advisor_at_play_phase(board: dict[str, Any]) -> AdvisorMirror:
    """An advisor carrying `board`, in the PLAY PHASE -- as the live lane has it.

    The precondition is the whole point, so the test must not construct the
    thing it is testing. `_catch_up_turns` ends the mirror's turn only once the
    world's `round` has moved, and a retain screen opens INSIDE the end-of-turn
    flush, before it does -- so at the screen the mirror has not ended its turn
    and is not parked. Hand-advancing it here would have "proved" a fix that
    does nothing live, which is exactly what happened to the already-parked
    path (c0b2ce576: green in isolation, 146 declines in run
    001-silent-a8-0825-0104).
    """
    from realgame_adapter import to_headless_combat_play
    from sts2_hint import adapter

    wire = adapter.translate(
        to_headless_combat_play(board), character="Ironclad", combat_id=1, seed=1, ascension=0
    )
    advisor = AdvisorMirror(HONEST)
    live = advisor.on_combat_board(wire, {"act": board["act"], "floor": board["floor"]})
    assert live is not None
    assert live.state.pending_choice() is None, "the live lane is not parked here"
    return advisor


def _wlp_screen(board: dict[str, Any]) -> dict[str, Any]:
    """The retain screen the world shows, offering the whole hand."""
    return {
        "type": "card_select",
        "screen": "NPlayerHand",
        "prompt_key": "WELL_LAID_PLANS_POWER.selectionScreenPrompt",
        "source_model": "WELL_LAID_PLANS_POWER",
        "source_kind": "power",
        "cards": [
            {"id": c["id"], "name": c.get("name") or c["id"], "index": i}
            for i, c in enumerate(board["hand"])
        ],
        "min_select": 0,
        "max_select": 1,
        "combat": board,
        "run_state": board["run_state"],
    }


def test_ending_the_turn_reaches_the_retain_screen() -> None:
    """`end_turn` is the replay, and the park is the proof.

    No name is matched. Ending a turn normally runs the flush, the enemy turn
    and the next setup and comes back to rest, so stopping on
    `retain_at_turn_end` is the engine's own statement that this is an
    end-of-turn retain screen (`combat_loop.rs`, `pending_wlp_retain_resume`).
    """
    board = _wlp_board()
    advisor = _advisor_at_play_phase(board)
    from realgame_adapter import to_headless_combat_play
    from sts2_hint import adapter

    wire = adapter.translate(
        to_headless_combat_play(board), character="Ironclad", combat_id=1, seed=1, ascension=0
    )

    assert advisor.replay_end_turn(wire)
    parked = advisor.live.state.pending_choice()
    assert parked is not None
    assert parked["kind"] == "retain_at_turn_end"
    assert advisor.stats["replays_landed"] == 1
    assert not advisor.declines, advisor.declines


def test_ending_the_turn_reaches_the_tools_of_the_trade_discard() -> None:
    """The same replay, on the far side of the round increment.

    `end_turn` does not stop at the boundary -- it runs the enemy turn and the
    next turn's setup, where `ToolsOfTheTradePower.AfterPlayerTurnStart` awaits
    a `FromHandForDiscard`. So a turn-START screen is this path's too, and the
    park (`discard_from_hand`) is the same kind of proof the retain screen is.

    Two defects met here and the test would pass with either half missing, so
    it asserts both: the engine has to SUSPEND the turn-start cluster (it used
    to park the choice, run the cluster tail anyway and report `advanced`), and
    `replay_end_turn` has to accept the kind (it hardcoded the retain screen).
    Live cost of the pair: 9 of the 17 unsearched in-combat decisions in
    `logs/f7-mirror-check15` -- the turn-start ones. The other 4 that corpus
    labels `TO_DISCARD` are the same PROMPT opened by a card's OnPlay, which
    is `replay_opening_play`'s path, not this one.
    """
    import copy
    import json

    from realgame_adapter import to_headless_combat_play
    from sts2_hint import adapter

    fixture = Path(__file__).parent / "fixtures" / "realgame_adapter" / "single_enemy_ironclad.json"
    board = copy.deepcopy(json.loads(fixture.read_text(encoding="utf-8"))["state"])
    board["player"]["powers"] = [
        {"id": "TOOLS_OF_THE_TRADE", "name": "Tools of the Trade", "amount": 1}
    ]
    advisor = _advisor_at_play_phase(board)
    wire = adapter.translate(
        to_headless_combat_play(board), character="Ironclad", combat_id=1, seed=1, ascension=0
    )

    assert advisor.replay_end_turn(wire), advisor.declines
    parked = advisor.live.state.pending_choice()
    assert parked is not None
    assert parked["kind"] == "discard_from_hand"
    assert advisor.stats["replays_landed"] == 1


def test_a_turn_that_parks_nothing_leaves_the_mirror_where_it_was() -> None:
    """Without the power the turn just ends -- and must not be ended.

    The probe exists for this case: `end_turn` cannot fail to apply, so
    committing first and checking after would strand the mirror a full round
    past the screen with no way back.
    """
    import copy
    import json

    fixture = Path(__file__).parent / "fixtures" / "realgame_adapter" / "single_enemy_ironclad.json"
    board = copy.deepcopy(json.loads(fixture.read_text(encoding="utf-8"))["state"])
    advisor = _advisor_at_play_phase(board)
    from realgame_adapter import to_headless_combat_play
    from sts2_hint import adapter

    wire = adapter.translate(
        to_headless_combat_play(board), character="Ironclad", combat_id=1, seed=1, ascension=0
    )
    before = sts2_rs.state_to_dict(advisor.live.state)["round"]

    assert not advisor.replay_end_turn(wire)
    assert advisor.live is not None, "declining must not drop the fight"
    assert advisor.live.state.pending_choice() is None
    assert sts2_rs.state_to_dict(advisor.live.state)["round"] == before
    assert advisor.stats["replays_landed"] == 0


def test_an_already_parked_fight_needs_no_end_turn_replay() -> None:
    """Parked is parked: return without spending a replay on it."""
    world = _burning_pact_state()
    advisor, _ = _advisor_on(world)
    advisor.live.advance(
        {"action": "play", "card_index": _hand_index(world, "BURNING_PACT"), "target_index": None}
    )
    assert advisor.live.state.pending_choice() is not None

    assert advisor.replay_end_turn(sts2_rs.state_to_dict(advisor.live.state)) is True
    assert advisor.stats["replays_attempted"] == 0, "nothing was replayed"


def test_a_card_mid_play_blocks_the_end_turn_replay() -> None:
    """A power hook fired from inside a card play is not a turn ending.

    Belt-and-braces, and pinned with a stub because the engine does not come
    to REST with a play suspended and no choice open -- the guard exists for
    the case the gate in `answer_card_select` cannot see, so the
    test must be able to present it.
    """

    class _MidPlay:
        @staticmethod
        def pending_choice() -> None:
            return None

        @staticmethod
        def has_pending_play() -> bool:
            return True

    class _Mirror:
        alive = True
        state = _MidPlay()

    advisor = AdvisorMirror(HONEST)
    assert not advisor.replay_end_turn({}, on=_Mirror())
    assert "a card is mid-play, so the turn is not ending" in advisor.declines
    assert advisor.stats["replays_attempted"] == 0


def _drifted_enemy_hp(board: dict[str, Any], hp: int) -> dict[str, Any]:
    """The same board with the first enemy on a different HP -- a drifted mirror.

    The one thing a retain park cannot explain. Nothing between the play phase
    and `retain_at_turn_end` touches enemy HP on either engine (C# runs
    `BeforeTurnEnd` -> `DoTurnEnd` -> `BeforeFlushLate`, and the Rust pin fires
    `before_turn_end` in phase one, both before the prompt), so a board that
    disagrees THERE disagreed before the turn ever ended.
    """
    import copy

    drifted = copy.deepcopy(board)
    drifted["enemies"][0]["hp"] = hp
    return drifted


def test_a_drifted_mirror_is_reconciled_rather_than_declined_at_the_retain_park() -> None:
    """The board proof at a retain park can only ever be reporting DRIFT.

    GH #593. The probe stops inside the end-of-turn flush, one hook before the
    round increments -- `round` is still N and the phase is still the play
    phase -- so the living-enemy HP vector it carries is the one the fork
    already had. A mismatch there is therefore never something the turn end
    did; it is the fork disagreeing with the world before the replay, which is
    exactly what `reconcile` exists to repair, and which it DOES repair one
    call later inside `pick_card_select_mirrored` -- too late, on a fork that
    is no longer parked.

    So repair first, then re-probe. The kind check is untouched and still the
    identity proof.
    """
    board = _wlp_board()
    advisor = _advisor_at_play_phase(_drifted_enemy_hp(board, 34))
    from realgame_adapter import to_headless_combat_play
    from sts2_hint import adapter

    wire = adapter.translate(
        to_headless_combat_play(board), character="Ironclad", combat_id=1, seed=1, ascension=0
    )

    assert advisor.replay_end_turn(wire), advisor.declines
    parked = advisor.live.state.pending_choice()
    assert parked is not None
    assert parked["kind"] == "retain_at_turn_end"
    assert advisor.stats["replays_landed"] == 1
    assert _enemy_hp(sts2_rs.state_to_dict(advisor.live.state)) == _enemy_hp(wire)


def test_a_drifted_mirror_is_still_declined_at_the_turn_START_park() -> None:
    """The repair is the retain park's alone, because the proof is not.

    `discard_from_hand` sits on the FAR side of the round increment: reaching
    it ran the flush, the whole enemy turn and the next turn's setup, so its
    board proof is grading a simulation the world may simply have played
    differently. That is a real proof and it stays strict -- and re-seeding
    from a wire the world took a full turn to reach would put the fork one
    turn ahead of itself, not repair it.
    """
    import copy
    import json

    fixture = Path(__file__).parent / "fixtures" / "realgame_adapter" / "single_enemy_ironclad.json"
    board = copy.deepcopy(json.loads(fixture.read_text(encoding="utf-8"))["state"])
    board["player"]["powers"] = [
        {"id": "TOOLS_OF_THE_TRADE", "name": "Tools of the Trade", "amount": 1}
    ]
    advisor = _advisor_at_play_phase(_drifted_enemy_hp(board, 34))
    from realgame_adapter import to_headless_combat_play
    from sts2_hint import adapter

    wire = adapter.translate(
        to_headless_combat_play(board), character="Ironclad", combat_id=1, seed=1, ascension=0
    )

    assert not advisor.replay_end_turn(wire)
    assert "the turn ended but the board did not match" in advisor.declines
    assert advisor.stats["replays_landed"] == 0


def test_a_drifted_well_laid_plans_screen_is_still_searched_end_to_end() -> None:
    """GH #593's whole symptom: two declines and a rarity-sorted answer.

    `logs/f7-mirror-check19/001-silent-a8-0826-1421.advisor.log`, consecutive:

        searched card_select declined (the turn ended but the board did not match)
        searched card_select declined (injection dropped the fork (NPlayerHand))
        advice #a909: best=Dodge and Roll p_win=None
          [no trained head for this selection (WELL_LAID_PLANS_POWER); heuristic pick]

    The second decline is the first one's consequence, not a second fault:
    `replay_end_turn`'s return value is not read, so a declined replay leaves
    the fork unparked, `pick_card_select_mirrored` rebuilds it from the wire,
    and `inject_choice` has no park to resolve the world's option list against.
    """
    board = _wlp_board()
    advisor = _advisor_at_play_phase(_drifted_enemy_hp(board, 34))
    advice = advisor.answer_card_select(_wlp_screen(board), HONEST, 8)

    assert advice is not None, advisor.declines
    assert advice["note"].startswith("combat MCTS"), advice["note"]
    assert advisor.stats["searched_selects"] == 1
    assert advisor.live is not None, "answering is side-effect free on the fight"


def test_a_well_laid_plans_screen_is_searched_end_to_end() -> None:
    """The class that was 77% of everything the advisor had to rank.

    Live run 001-silent-a8-0825-0104: 94 of 122 ranked in-combat decisions
    were this one power, every one of them answered by the rarity sort with
    `WELL_LAID_PLANS_POWER is a power, not a card or a potion`.
    """

    board = _wlp_board()
    advisor = _advisor_at_play_phase(board)
    advice = advisor.answer_card_select(_wlp_screen(board), HONEST, 8)

    assert advice is not None, "an end-of-turn screen is answerable now"
    assert advice["note"].startswith("combat MCTS"), advice["note"]
    assert advisor.stats["searched_selects"] == 1
    assert not advisor.declines, advisor.declines
    assert advisor.live is not None, "answering is side-effect free on the fight"
    assert advisor.live.state.pending_choice() is None, "the fight did not end its turn"


# ---------------------------------------------------------------------------
# Which character the screen belongs to
# ---------------------------------------------------------------------------


def _captured_character(monkeypatch: Any, screen: dict[str, Any], cfg: Any) -> Any:
    """Run the rung and report the `character` it handed the adapter."""
    from sts2_hint import adapter

    seen: dict[str, Any] = {}
    real = adapter.translate

    def spy(headless: Any, **kw: Any) -> Any:
        seen["character"] = kw.get("character")
        return real(headless, **kw)

    monkeypatch.setattr(adapter, "translate", spy)
    board = _wlp_board()
    advisor = _advisor_at_play_phase(board)
    advisor.answer_card_select(screen, cfg, 8)
    return seen.get("character", "translate was never called")


def test_the_screens_own_run_state_names_the_character(monkeypatch: Any) -> None:
    """`serve()` builds its config with no character, so the screen must say.

    Passing `cfg.character` straight through left it None on this path, and
    `adapter.translate` then inferred the class from card-id PREFIXES. That
    works on nearly every board and fails on one with no prefixed card: live
    run 002-silent-a8-0825-0259 hit `Scare | Corrosive Wave | Shadowmeld` and
    lost 12 selections to `$.player.class: could not infer character class`
    -- silently, through the generic except arm below.

    `compute_combat_advice` has resolved it per call all along; this path
    simply never did.
    """
    board = _wlp_board()
    screen = _wlp_screen(board)
    screen["run_state"] = {**screen["run_state"], "character": "Silent"}
    assert _captured_character(monkeypatch, screen, HONEST) == "Silent"


def test_a_search_that_cannot_run_is_counted_not_just_logged(monkeypatch: Any) -> None:
    """Every other way this rung gives up records a decline. So must this one.

    Run 002-silent-a8-0825-0259 reported `6/6 replays` and no declines at all
    while nine of its fourteen ranked decisions had fallen to the rarity sort
    through here -- the stat line read clean because this arm only logged.
    """
    from sts2_hint import adapter

    def boom(*_a: Any, **_k: Any) -> Any:
        raise ValueError("no")

    board = _wlp_board()
    advisor = _advisor_at_play_phase(board)  # seeded BEFORE the sabotage
    monkeypatch.setattr(adapter, "translate", boom)

    assert advisor.answer_card_select(_wlp_screen(board), HONEST, 8) is None
    assert "the search could not run (ValueError)" in advisor.declines
    assert advisor.live is not None, "a failed search must not cost the fight"


# ---------------------------------------------------------------------------
# The second pick of a multi-pick selection
# ---------------------------------------------------------------------------


def _offer(*names: str) -> list[dict[str, Any]]:
    return [{"id": n, "name": n.title(), "index": i} for i, n in enumerate(names)]


class TestRememberedPicks:
    """A card that discards two is ONE engine choice and one search.

    The mod presents it as one advice request per pick, the offered list
    shrinking by the card just taken:

        Decision up: NPlayerHand [4:Ultimate Defend|Defend|Defend|Survivor]
        Decision up: NPlayerHand [3:Defend|Defend|Survivor]

    Only the first was searched. The second replayed the opening play from
    scratch, so the engine parked the FULL option list against the world's
    shortened one and `inject_choice` dropped the fork -- run
    003-silent-a8-0825-0125 landed `70/70 replays` and still dropped 33
    injections. The answer already existed: `2 pick(s) searched`.
    """

    @staticmethod
    def _primed() -> AdvisorMirror:
        advisor = AdvisorMirror(HONEST)
        advisor.remember_picks("PREPARED", _offer("ULTIMATE_DEFEND", "DEFEND", "SURVIVOR"), [0, 2])
        return advisor

    def test_the_next_pick_comes_from_the_first_screens_search(self) -> None:
        advisor = self._primed()
        # The world took ULTIMATE_DEFEND, exactly as recommended.
        assert advisor.next_remembered_pick("PREPARED", _offer("DEFEND", "SURVIVOR")) == 1

    def test_the_cache_is_spent_after_its_last_pick(self) -> None:
        advisor = self._primed()
        advisor.next_remembered_pick("PREPARED", _offer("DEFEND", "SURVIVOR"))
        assert advisor.next_remembered_pick("PREPARED", _offer("DEFEND")) is None

    def test_a_world_that_picked_something_else_invalidates_it(self) -> None:
        # THE safety case: the advisor recommends, it does not execute. Here
        # the world discarded SURVIVOR instead, so the remaining offer is not
        # the original minus what was served and the cache must not be used.
        advisor = self._primed()
        assert advisor.next_remembered_pick("PREPARED", _offer("ULTIMATE_DEFEND", "DEFEND")) is None

    def test_an_absent_source_still_serves_the_pick(self) -> None:
        # THE case that actually occurs live, and the one this class missed:
        # `SimCardSelectContext.Consume()` nulls SourceModel and fires once
        # per CardSelectCmd, while a multi-pick selection asks for advice once
        # per PICK -- so pick 2 always arrives with source_model=None. The
        # sibling test below covers a DIFFERENT source and passed throughout,
        # which is why 7 TO_DISCARD decisions fell to the rarity sort with the
        # answer already computed (checks 13-14, 2026-08-25).
        advisor = self._primed()
        assert advisor.next_remembered_pick(None, _offer("DEFEND", "SURVIVOR")) == 1

    def test_an_absent_source_is_still_held_to_the_multiset(self) -> None:
        # Dropping the source check must not drop the SAFETY check with it:
        # the advisor recommends but never executes, so a world that picked
        # something else still has to invalidate the cache.
        advisor = self._primed()
        assert advisor.next_remembered_pick(None, _offer("ULTIMATE_DEFEND", "DEFEND")) is None

    def test_being_asked_twice_for_the_same_pick_answers_it_twice(self) -> None:
        """A SERVE IS NOT AN EXECUTION, so it must not consume the plan.

        The advisor answers into a request id the mod may already have
        abandoned -- `Dropping advice for unknown/stale request_id=` appears
        131 times in run 007 alone -- and the mod then re-requests the same
        pick. Consuming on serve burned the cached answer, so the re-ask found
        an empty plan and fell to the rarity sort. That is all 7 Hidden
        Daggers second picks in `logs/f7-mirror-check16`, each with `the
        second pick of the same search` logged one line above the fallback.

        The offer is the position, so the same offer is the same answer.
        """
        advisor = self._primed()
        second = _offer("DEFEND", "SURVIVOR")
        assert advisor.next_remembered_pick(None, second) == 1
        assert advisor.next_remembered_pick(None, second) == 1, "a re-ask is not progress"
        assert advisor.next_remembered_pick(None, second) == 1

    def test_a_re_ask_of_the_first_pick_repeats_the_first_pick(self) -> None:
        """The same property one screen earlier -- and it used to skip ahead.

        Asked again with the UNSHRUNK offer, the old cache handed back pick 2
        while the screen was still on pick 1: a card the world had not taken
        yet, recommended out of order.
        """
        advisor = self._primed()
        whole = _offer("ULTIMATE_DEFEND", "DEFEND", "SURVIVOR")
        assert advisor.next_remembered_pick(None, whole) == 0
        assert advisor.next_remembered_pick(None, whole) == 0

    def test_a_different_source_does_not_inherit_it(self) -> None:
        advisor = self._primed()
        assert advisor.next_remembered_pick("DAGGER_THROW", _offer("DEFEND", "SURVIVOR")) is None

    def test_duplicates_are_matched_as_a_multiset_not_by_position(self) -> None:
        # Two Defends offered, one of them picked: the leftover offer still
        # CONTAINS a Defend, so an "is it gone?" check would wrongly reject.
        advisor = AdvisorMirror(HONEST)
        advisor.remember_picks("PREPARED", _offer("DEFEND", "DEFEND", "SURVIVOR"), [0, 2])
        assert advisor.next_remembered_pick("PREPARED", _offer("DEFEND", "SURVIVOR")) == 1

    def test_a_single_pick_search_caches_nothing(self) -> None:
        advisor = AdvisorMirror(HONEST)
        advisor.remember_picks("PREPARED", _offer("DEFEND", "SURVIVOR"), [0])
        assert advisor.next_remembered_pick("PREPARED", _offer("SURVIVOR")) is None

    def test_dropping_the_fight_forgets_the_picks(self) -> None:
        advisor = self._primed()
        advisor.drop("new fight")
        assert advisor.next_remembered_pick("PREPARED", _offer("DEFEND", "SURVIVOR")) is None


def test_note_names_the_desynced_paths() -> None:
    """The note tallies desyncs by wire path, list keys folded, so a run
    log says WHAT the mirror failed to follow (co-op: a teammate's moves)."""
    import copy

    advisor = AdvisorMirror(HONEST)
    wire = sts2_rs.state_to_dict(_burning_pact_state())
    advisor.on_combat_board(wire, {"act": 1, "floor": 3})
    moved = copy.deepcopy(wire)
    moved["encounter"][0]["hp"] -= 5  # someone else hit the enemy
    advisor.on_combat_board(moved, {"act": 1, "floor": 3})
    assert advisor.stats["desyncs"] >= 1
    assert "desynced: $.encounter[].hp x1" in advisor.note()


def _coop_board(player_count: int) -> dict[str, Any]:
    return {"act": 1, "floor": 3, "player_count": player_count}


def test_coop_adopts_a_teammate_hit_instead_of_dropping() -> None:
    """Co-op: an enemy field the mirror could not have predicted (a teammate
    hit it) is adopted, not reported. The mirror keeps its carried state and
    stays untainted; the note says how often that happened."""
    import copy

    advisor = AdvisorMirror(HONEST)
    wire = sts2_rs.state_to_dict(_burning_pact_state())
    advisor.on_combat_board(wire, _coop_board(2))
    mirror = advisor.live
    moved = copy.deepcopy(wire)
    moved["encounter"][0]["hp"] -= 5
    advisor.on_combat_board(moved, _coop_board(2))
    assert advisor.stats["desyncs"] == 0
    assert advisor.live is mirror and not mirror.tainted
    assert sts2_rs.state_to_dict(mirror.state)["encounter"][0]["hp"] == moved["encounter"][0]["hp"]
    assert "teammate-adopted: $.encounter[].hp x1" in advisor.note()


def test_coop_adopts_the_teammates_own_block() -> None:
    """The teammate's creature (``$.allies``) moves on its own -- they block
    and take hits -- so a change there is adopted like an enemy-side one."""
    import copy

    advisor = AdvisorMirror(HONEST)
    wire = sts2_rs.state_to_dict(_burning_pact_state())
    wire["allies"] = [{"hp": 60, "max_hp": 70, "block": 0}]
    advisor.on_combat_board(wire, _coop_board(2))
    mirror = advisor.live
    moved = copy.deepcopy(wire)
    moved["allies"][0]["block"] = 9
    advisor.on_combat_board(moved, _coop_board(2))
    assert advisor.stats["desyncs"] == 0
    assert advisor.live is mirror and not mirror.tainted
    assert sts2_rs.state_to_dict(mirror.state)["allies"][0]["block"] == 9


def test_coop_still_reports_a_player_side_desync() -> None:
    """The exemption is scoped to the enemy side: our own HP is ours to
    predict, so a mismatch there drops the mirror exactly as solo."""
    import copy

    advisor = AdvisorMirror(HONEST)
    wire = sts2_rs.state_to_dict(_burning_pact_state())
    advisor.on_combat_board(wire, _coop_board(2))
    moved = copy.deepcopy(wire)
    moved["encounter"][0]["hp"] -= 5
    moved["player"]["hp"] -= 3
    advisor.on_combat_board(moved, _coop_board(2))
    assert advisor.stats["desyncs"] >= 2
    assert "teammate-adopted" not in advisor.note()


def test_solo_never_adopts() -> None:
    import copy

    advisor = AdvisorMirror(HONEST)
    wire = sts2_rs.state_to_dict(_burning_pact_state())
    advisor.on_combat_board(wire, _coop_board(1))
    moved = copy.deepcopy(wire)
    moved["encounter"][0]["hp"] -= 5
    advisor.on_combat_board(moved, _coop_board(1))
    assert advisor.stats["desyncs"] == 1
    assert "teammate-adopted" not in advisor.note()
