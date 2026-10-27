"""Co-op: a teammate with a hand is a whole seat, and the engine plays it.

The mod sends each teammate's hand, energy, powers and whether they already
ended their turn. When we end ours, the engine plays out what they have left
(``sts2_core::coop``) before the enemies act, so the search knows the block
they will put up themselves and the Weak they will apply.
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


def _mod_card(card_id: str, card_type: str, target: str) -> dict[str, Any]:
    """A hand card in the mod's shape, as the fixture's Ironclad cards are."""
    board = json.loads(FIXTURE.read_text(encoding="utf-8"))["state"]
    card = copy.deepcopy(board["hand"][0])
    card.update(id=card_id, type=card_type, target_type=target)
    return card


def _board(*, ended_turn: bool = False) -> dict[str, Any]:
    board = json.loads(FIXTURE.read_text(encoding="utf-8"))["state"]
    board["allies"] = [
        {
            "character": "SILENT",
            "hp": 50,
            "max_hp": 70,
            "block": 0,
            "is_dead": False,
            "energy": 2,
            "max_energy": 3,
            "powers": [],
            "hand": [
                _mod_card("STRIKE_SILENT", "Attack", "AnyEnemy"),
                _mod_card("DEFEND_SILENT", "Skill", "Self"),
            ],
            "ended_turn": ended_turn,
        }
    ]
    return board


def _end_turn(wire: dict[str, Any]) -> dict[str, Any]:
    state = sts2_rs.state_from_dict(wire)
    end = next(a for a in state.legal_actions() if a.get("action") == "end_turn")
    state.apply(end)
    return sts2_rs.state_to_dict(state)


def test_the_teammates_seat_rides_the_wire() -> None:
    wire = translate(to_headless_combat_play(_board()))
    ally = wire["allies"][0]
    assert ally["class"] == "silent"
    assert [c["id"] for c in ally["hand"]] == ["STRIKE_SILENT", "DEFEND_SILENT"]
    assert ally["energy"] == 2
    assert ally["ended_turn"] is False
    again = sts2_rs.state_to_dict(sts2_rs.state_from_dict(wire))["allies"][0]
    assert [c["id"] for c in again["hand"]] == ["STRIKE_SILENT", "DEFEND_SILENT"]
    assert (again["energy"], again["ended_turn"]) == (2, False)


def test_the_teammate_plays_their_hand_when_we_end_our_turn() -> None:
    wire = translate(to_headless_combat_play(_board()))
    enemy_hp = wire["encounter"][0]["hp"]
    after = _end_turn(wire)
    assert after["encounter"][0]["hp"] == enemy_hp - 6, "their Strike landed"
    assert after["allies"][0]["block"] == 5, "their Defend landed (no attack this turn)"
    assert after["allies"][0]["hand"] == []


def test_a_teammate_who_ended_their_turn_plays_nothing() -> None:
    wire = translate(to_headless_combat_play(_board(ended_turn=True)))
    enemy_hp = wire["encounter"][0]["hp"]
    after = _end_turn(wire)
    assert after["encounter"][0]["hp"] == enemy_hp
    assert len(after["allies"][0]["hand"]) == 2


def _mawler_lift_gain(mate_hand: list[str] | None) -> float:
    """Q of playing Lift over ending the turn, Mawler about to hit for 14."""
    wire = sts2_rs.state_to_dict(sts2_rs.CombatState(seed=1, encounter="Mawler"))
    wire["encounter"][0]["intents"] = [{"amount": 14, "effective_amount": 14, "kind": "Attack"}]
    hand = wire["deck"]["hand"]
    template = {k: v for k, v in hand[1].items() if k != "stable_idx"}
    hand[0].update(id="LIFT", base_damage=0, base_block=11)
    wire["player"]["energy"] = 1
    mate: dict[str, Any] = {"hp": 40, "max_hp": 70, "block": 0}
    if mate_hand is not None:
        stats = {"DEFEND_SILENT": (0, 5), "STRIKE_SILENT": (6, 0)}
        mate.update(
            {
                "class": "silent",
                "energy": len(mate_hand),
                "max_energy": 3,
                "powers": [],
                "ended_turn": False,
                "hand": [
                    dict(
                        template,
                        id=c,
                        stable_idx=100 + i,
                        base_damage=stats[c][0],
                        base_block=stats[c][1],
                    )
                    for i, c in enumerate(mate_hand)
                ],
            }
        )
    wire["allies"] = [mate]
    state = sts2_rs.state_from_dict(wire)
    out = sts2_rs.mcts_recommend(state, sims=1500, threads=1, seed=3)
    names = [c["id"] for c in sts2_rs.state_to_dict(state)["deck"]["hand"]]
    q = {}
    for row in out["actions"]:
        a = row["action"]
        q[names[a["card_index"]] if a.get("action") == "play" else a["action"]] = row["q"]
    return q["LIFT"] - q["end_turn"]


HP_Q = 0.5 * 100 / 14000


def test_the_search_counts_the_block_the_teammate_will_play() -> None:
    """Lift is worth the hit the teammate will NOT cover themselves: all 11
    against a teammate who only attacks, 4 against one holding two Defends."""
    assert _mawler_lift_gain(["STRIKE_SILENT", "STRIKE_SILENT"]) == pytest.approx(
        11 * HP_Q, abs=0.01
    )
    assert _mawler_lift_gain(["DEFEND_SILENT", "DEFEND_SILENT"]) == pytest.approx(
        4 * HP_Q, abs=0.008
    )


def _mawler_with_mate(
    mate_hand: list[str], mate_powers: list[dict[str, Any]], enemy_powers: list[dict[str, Any]]
) -> dict[str, Any]:
    """A Mawler fight with one whole teammate seat, on the engine wire."""
    wire = sts2_rs.state_to_dict(sts2_rs.CombatState(seed=1, encounter="Mawler"))
    wire["encounter"][0]["powers"] = enemy_powers
    template = {k: v for k, v in wire["deck"]["hand"][1].items() if k != "stable_idx"}
    stats = {"DEFEND_SILENT": (0, 5), "STRIKE_SILENT": (6, 0)}
    wire["allies"] = [
        {
            "hp": 40,
            "max_hp": 70,
            "block": 0,
            "class": "silent",
            "energy": len(mate_hand),
            "max_energy": 3,
            "powers": mate_powers,
            "ended_turn": False,
            "hand": [
                dict(
                    template,
                    id=c,
                    stable_idx=100 + i,
                    base_damage=stats[c][0],
                    base_block=stats[c][1],
                )
                for i, c in enumerate(mate_hand)
            ],
        }
    ]
    return wire


def test_the_teammates_cards_do_not_tick_our_strangle() -> None:
    """Strangle hurts only on its applier's cards (C# `StranglePower`)."""
    wire = _mawler_with_mate(["STRIKE_SILENT"], [], [{"name": "STRANGLE", "amount": 2}])
    hp = wire["encounter"][0]["hp"]
    after = _end_turn(wire)
    assert after["encounter"][0]["hp"] == hp - 6, "their Strike, and no Strangle tick"


def test_the_teammates_debuffs_tick_down_at_the_enemy_turn_end() -> None:
    wire = _mawler_with_mate([], [{"name": "WEAK", "amount": 2}], [])
    wire["player"]["powers"] = [{"name": "WEAK", "amount": 2}]
    after = _end_turn(wire)
    for seat in (after["player"], after["allies"][0]):
        assert [(p["name"], p["amount"]) for p in seat["powers"]] == [("WEAK", 1)]


def test_poison_counts_the_teammates_accelerant() -> None:
    """Poison ticks `1 + Σ Accelerant` times over every living player."""
    wire = _mawler_with_mate(
        [], [{"name": "ACCELERANT", "amount": 1}], [{"name": "POISON", "amount": 5}]
    )
    hp = wire["encounter"][0]["hp"]
    after = _end_turn(wire)
    assert after["encounter"][0]["hp"] == hp - 5 - 4
    assert ("POISON", 3) in [(p["name"], p["amount"]) for p in after["encounter"][0]["powers"]]


def test_a_full_fight_rides_the_wire_and_the_teammate_draws() -> None:
    """``$.coop_full`` and a whole seat's piles, relics and stars round-trip, and
    the engine then runs the teammate's own turn: their hand flushes at our end
    of turn and they draw their next one from their own draw pile."""
    wire = _mawler_with_mate(["DEFEND_SILENT"], [], [])
    template = {k: v for k, v in wire["deck"]["hand"][1].items() if k != "stable_idx"}
    mate = wire["allies"][0]
    mate["draw_pile"] = [
        dict(template, id="STRIKE_SILENT", stable_idx=200 + i, base_damage=6, base_block=0)
        for i in range(6)
    ]
    mate["relics"] = [{"id": "ANCHOR"}]
    wire["coop_full"] = True

    again = sts2_rs.state_to_dict(sts2_rs.state_from_dict(wire))
    assert again["coop_full"] is True
    seat = again["allies"][0]
    assert [c["id"] for c in seat["draw_pile"]] == ["STRIKE_SILENT"] * 6
    assert [r["id"] for r in seat["relics"]] == ["ANCHOR"]

    after = _end_turn(wire)
    seat = after["allies"][0]
    assert len(seat["hand"]) == 5, "their next hand, from their own draw pile"
    assert {c["id"] for c in seat["hand"]} == {"STRIKE_SILENT"}
    assert len(seat.get("draw_pile", [])) == 1


def test_a_teammates_whole_seat_rides_from_the_mod_to_the_engine() -> None:
    """The mod's teammate piles (``{"cards": [...]}``), relics and stars pass
    through both adapters into the engine wire; absent on an older mod."""
    board = _board()
    relic = json.loads(FIXTURE.read_text(encoding="utf-8"))["state"]["run_state"]["relics"][0]
    mate = board["allies"][0]
    mate["draw_pile"] = {
        "type": "Draw",
        "count": 2,
        "cards": [
            _mod_card("STRIKE_SILENT", "Attack", "AnyEnemy"),
            _mod_card("NEUTRALIZE", "Attack", "AnyEnemy"),
        ],
    }
    mate["discard_pile"] = {"type": "Discard", "count": 0, "cards": []}
    mate["relics"] = [dict(relic, id="RING_OF_THE_SNAKE", name="Ring of the Snake")]
    mate["stars"] = 0

    ally = translate(to_headless_combat_play(board))["allies"][0]
    assert [c["id"] for c in ally["draw_pile"]] == ["STRIKE_SILENT", "NEUTRALIZE"]
    assert ally["discard_pile"] == []
    assert [r["id"] for r in ally["relics"]] == ["RING_OF_THE_SNAKE"]
    uids = [c["stable_idx"] for c in ally["hand"] + ally["draw_pile"]]
    assert len(set(uids)) == len(uids), "every card numbered once"
    seat = sts2_rs.state_to_dict(
        sts2_rs.state_from_dict(translate(to_headless_combat_play(board)))
    )["allies"][0]
    assert [c["id"] for c in seat["draw_pile"]] == ["STRIKE_SILENT", "NEUTRALIZE"]

    old = translate(to_headless_combat_play(_board()))["allies"][0]
    assert "draw_pile" not in old and "relics" not in old
