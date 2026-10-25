"""`ppo.bc_human.replay.fights`: how a SpireCodex log becomes fight records.

Pins two parser rules found on the #918 corpus check:

- The mod logs the end-of-turn damage of cards in hand (Infection, Burn, ...)
  before the player's `end_turn` event. It belongs to the `end_turn`
  decision, or the log view loses that HP one decision before the engine.
- A relic the player loses (`relic_lost`) is gone from later fights.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

from ppo.bc_human import replay

HEADER = {
    "t": "header",
    "seed": "S",
    "character": "IRONCLAD",
    "ascension": 10,
    "relics": [{"id": "BURNING_BLOOD"}],
    "starting_max_hp": 80,
}


def _write(tmp_path: Path, events: list[dict[str, Any]]) -> str:
    path = tmp_path / "x.replay.ndjson.gz"
    with gzip.open(path, "wt") as fh:
        for e in [HEADER, *events]:
            fh.write(json.dumps(e) + "\n")
    return str(path)


def _combat(cid: str, body: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"t": "combat_start", "combat_id": cid, "encounter": "PHROG_PARASITE_ELITE"},
        *body,
        {"t": "combat_end", "result": "win"},
    ]


def _infection(dmg: int) -> dict[str, Any]:
    return {"t": "hit", "src": "player", "dst": "player", "dmg": dmg, "card": "INFECTION"}


def test_turn_end_hand_damage_belongs_to_end_turn(tmp_path: Path) -> None:
    blood_wall = {"t": "hit", "src": "player", "dst": "player", "dmg": 2, "card": "BLOOD_WALL"}
    path = _write(
        tmp_path,
        _combat(
            "1.9:X",
            [
                {"t": "play", "id": "BLOOD_WALL"},
                blood_wall,  # the play's own self-damage stays with it
                _infection(3),
                _infection(1),
                {"t": "flush", "flushed_c": []},
                {"t": "end_turn", "side": "player"},
                {"t": "hit", "src": "PHROG_PARASITE", "dst": "player", "dmg": 5},
            ],
        ),
    )
    (rec,) = replay.fights(path)
    play, end = rec["decisions"]
    assert play["after"] == [blood_wall, {"t": "flush", "flushed_c": []}]
    assert end["ev"]["t"] == "end_turn"
    assert end["after"][:2] == [_infection(3), _infection(1)]
    assert end["after"][2]["src"] == "PHROG_PARASITE"


def test_turn_end_damage_after_an_idle_turn_leaves_the_turn_start(tmp_path: Path) -> None:
    # No card played this turn: the damage sits in the previous end_turn's
    # `after`, behind this turn's start; only what follows the start moves.
    auto = {"t": "play", "id": "HOWL_FROM_BEYOND", "auto": True}
    path = _write(
        tmp_path,
        _combat(
            "1.9:X",
            [
                {"t": "end_turn", "side": "player"},
                {"t": "turn", "side": "player"},
                auto,
                _infection(1),  # after an auto-play of another card: still this turn's end
                {"t": "end_turn", "side": "player"},
            ],
        ),
    )
    (rec,) = replay.fights(path)
    first, second = rec["decisions"]
    assert first["after"] == [{"t": "turn", "side": "player"}, auto]
    assert second["after"] == [_infection(1)]


def test_hits_before_the_turn_start_stay(tmp_path: Path) -> None:
    last_turn = _infection(3)
    path = _write(
        tmp_path,
        _combat(
            "1.9:X",
            [
                {"t": "end_turn", "side": "player"},
                last_turn,  # logged after end_turn: already in the right place
                {"t": "turn", "side": "player"},
                {"t": "end_turn", "side": "player"},
            ],
        ),
    )
    (rec,) = replay.fights(path)
    assert rec["decisions"][0]["after"] == [last_turn, {"t": "turn", "side": "player"}]
    assert rec["decisions"][1]["after"] == []


def test_lost_relics_leave_later_fights(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        [
            {"t": "relic", "id": "FESTIVE_POPPER"},
            *_combat("1.1:A", [{"t": "end_turn", "side": "player"}]),
            {"t": "relic_lost", "id": "FESTIVE_POPPER", "reason": "removed"},
            *_combat("3.44:B", [{"t": "end_turn", "side": "player"}]),
        ],
    )
    a, b = replay.fights(path)
    assert a["relics"] == ["BURNING_BLOOD", "FESTIVE_POPPER"]
    assert b["relics"] == ["BURNING_BLOOD"]


def test_a_replayed_play_is_part_of_the_first(tmp_path: Path) -> None:
    first = {"t": "play", "c": 7, "id": "TWIN_STRIKE", "play_index": 0, "play_count": 2}
    again = {"t": "play", "c": 7, "id": "TWIN_STRIKE", "play_index": 1, "play_count": 2}
    path = _write(tmp_path, _combat("1.1:A", [first, again]))
    (rec,) = replay.fights(path)
    assert [d["ev"] for d in rec["decisions"]] == [first]
    assert rec["decisions"][0]["after"] == [again]
