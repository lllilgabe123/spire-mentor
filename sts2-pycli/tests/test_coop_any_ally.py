"""Co-op: the teammate reaches the engine, so ``AnyAlly`` cards are played.

Without ``$.allies`` the engine sees no living teammate and, like C#'s
``NoLivingAllies``, refuses every ``AnyAlly`` card -- so the advisor never
recommended Lift, Mimic, Intercept and the rest in a co-op fight.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from realgame_adapter import to_headless_combat_play
from sts2_hint.adapter import translate

sts2_rs = pytest.importorskip("sts2_rs")

FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "realgame_adapter" / "single_enemy_ironclad.json"
)
TEAMMATE = {"character": "SILENT", "hp": 50, "max_hp": 70, "block": 4, "is_dead": False}


def _board(allies: list[dict[str, Any]] | None) -> dict[str, Any]:
    board = json.loads(FIXTURE.read_text(encoding="utf-8"))["state"]
    if allies is not None:
        board["allies"] = allies
    return board


def _lift_state(wire: dict[str, Any]):
    wire = copy.deepcopy(wire)
    lift = wire["deck"]["hand"][0]
    lift.update(
        {"id": "LIFT", "upgraded": False, "cost_override": 1, "base_cost": 1, "type": "Skill"}
    )
    wire["player"]["energy"] = 3
    return sts2_rs.state_from_dict(wire)


def _plays_of(state, card_id: str) -> list[dict[str, Any]]:
    hand = sts2_rs.state_to_dict(state)["deck"]["hand"]
    return [
        a
        for a in state.legal_actions()
        if a.get("action") == "play" and hand[a["card_index"]]["id"] == card_id
    ]


def test_teammate_rides_the_wire_and_solo_stays_unchanged() -> None:
    solo = translate(to_headless_combat_play(_board(None)))
    assert "allies" not in solo
    wire = translate(to_headless_combat_play(_board([TEAMMATE])))
    assert wire["allies"] == [{"hp": 50, "max_hp": 70, "block": 4}]
    assert sts2_rs.state_to_dict(sts2_rs.state_from_dict(wire))["allies"] == wire["allies"]


def test_lift_is_playable_only_with_a_teammate_and_blocks_them() -> None:
    solo = translate(to_headless_combat_play(_board(None)))
    assert _plays_of(_lift_state(solo), "LIFT") == []

    coop = translate(to_headless_combat_play(_board([TEAMMATE])))
    state = _lift_state(coop)
    plays = _plays_of(state, "LIFT")
    assert len(plays) == 1
    block_before = sts2_rs.state_to_dict(state)["player"]["block"]
    state.apply(plays[0])
    after = sts2_rs.state_to_dict(state)
    assert after["allies"][0]["block"] == 4 + 11
    assert after["player"]["block"] == block_before


def test_a_dead_teammate_makes_any_ally_cards_unplayable() -> None:
    dead = dict(TEAMMATE, hp=0, is_dead=True)
    wire = translate(to_headless_combat_play(_board([dead])))
    assert _plays_of(_lift_state(wire), "LIFT") == []


def _mawler_with_lift(allies: list[dict[str, Any]]):
    """Mawler about to hit for 14; we hold Lift, Strike and Defend with 1 energy."""
    wire = sts2_rs.state_to_dict(sts2_rs.CombatState(seed=1, encounter="Mawler"))
    wire["encounter"][0]["intents"] = [{"amount": 14, "effective_amount": 14, "kind": "Attack"}]
    hand = wire["deck"]["hand"]
    hand[0].update(id="LIFT", base_damage=0, base_block=11)
    hand[2].update(id="DEFEND_IRONCLAD", base_damage=0, base_block=5)
    wire["player"]["energy"] = 1
    wire["allies"] = allies
    return sts2_rs.state_from_dict(wire)


def _lift_gain(state) -> float:
    """Q of playing Lift over ending the turn."""
    out = sts2_rs.mcts_recommend(state, sims=1500, threads=1, seed=3)
    hand = sts2_rs.state_to_dict(state)["deck"]["hand"]
    q = {}
    for row in out["actions"]:
        a = row["action"]
        q[hand[a["card_index"]]["id"] if a.get("action") == "play" else a["action"]] = row["q"]
    return q["LIFT"] - q["end_turn"]


def test_the_search_values_blocking_an_exposed_teammate() -> None:
    """The enemy turn hits the teammate too and their lost HP counts, so
    Lift's 11 block on them is worth 11 HP -- unless they are already safe."""
    exposed = _lift_gain(_mawler_with_lift([{"hp": 40, "max_hp": 70, "block": 0}]))
    assert exposed == pytest.approx(11 * 0.5 * 100 / 14000, abs=0.01)
    safe = _lift_gain(_mawler_with_lift([{"hp": 40, "max_hp": 70, "block": 20}]))
    assert abs(safe) < 0.01
