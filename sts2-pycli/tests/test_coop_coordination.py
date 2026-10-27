"""Co-op: hold a move the teammate's hand is about to make better."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from coop_coordination import ally_wait, apply_ally_wait


def _card(cid: str, ctype: str = "Skill", cost: int = 1, **stats: int) -> dict[str, Any]:
    return {"id": cid, "name": cid.title(), "type": ctype, "effective_cost": cost, "stats": stats}


STRIKE = _card("STRIKE_IRONCLAD", "Attack", damage=6)
DEFEND = _card("DEFEND_IRONCLAD", block=5)
BASH = _card("BASH", "Attack", 2, damage=8, vulnerablepower=2)
NEUTRALIZE = _card("NEUTRALIZE", "Attack", 0, damage=3, weakpower=1)
LIFT = _card("LIFT", block=8)


def _board(
    ours: dict[str, Any],
    ally_hand: list[dict[str, Any]],
    *,
    energy: int = 3,
    ended: bool = False,
    dead: bool = False,
    enemy_powers: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "hand": [ours],
        "enemies": [{"hp": 40, "powers": [{"id": p} for p in enemy_powers or []]}],
        "allies": [
            {
                "character": "SILENT",
                "energy": energy,
                "ended_turn": ended,
                "is_dead": dead,
                "hand": ally_hand,
            }
        ],
    }


def _best(card_index: int | None = 0) -> dict[str, Any]:
    return {"note": "MCTS", "best": {"card_index": card_index, "label": "x"}}


def test_attack_waits_for_the_teammates_vulnerable() -> None:
    wait = ally_wait(_board(STRIKE, [BASH]), _best())
    assert wait is not None and wait["card"] == "Bash"
    assert "Vulnerable" in wait["reason"] and wait["max_ms"] > 0


def test_block_waits_for_weak_or_ally_block() -> None:
    assert ally_wait(_board(DEFEND, [NEUTRALIZE]), _best())["card"] == "Neutralize"
    assert ally_wait(_board(DEFEND, [LIFT]), _best())["card"] == "Lift"


def test_no_wait_when_it_would_not_help() -> None:
    # Weak does nothing for an attack; Vulnerable nothing for a block.
    assert ally_wait(_board(STRIKE, [NEUTRALIZE]), _best()) is None
    assert ally_wait(_board(DEFEND, [BASH]), _best()) is None
    # Every enemy is already Vulnerable.
    assert ally_wait(_board(STRIKE, [BASH], enemy_powers=["VULNERABLE_POWER"]), _best()) is None


def test_no_wait_on_a_teammate_who_cannot_act() -> None:
    assert ally_wait(_board(STRIKE, [BASH], ended=True), _best()) is None
    assert ally_wait(_board(STRIKE, [BASH], dead=True), _best()) is None
    assert ally_wait(_board(STRIKE, [BASH], energy=1), _best()) is None  # Bash costs 2


def test_no_wait_for_end_turn_or_single_player() -> None:
    assert ally_wait(_board(STRIKE, [BASH]), _best(None)) is None
    solo = _board(STRIKE, [BASH])
    del solo["allies"]
    assert ally_wait(solo, _best()) is None


def test_our_own_debuff_is_never_held() -> None:
    assert ally_wait(_board(BASH, [_card("THUNDERCLAP", "Attack", 1, damage=4)]), _best()) is None


def test_apply_names_the_wait_in_the_note() -> None:
    advice = apply_ally_wait(_board(STRIKE, [BASH]), _best())
    assert advice["wait_for_ally"]["card"] == "Bash"
    assert advice["note"].startswith("MCTS · co-op: wait for the teammate's Bash")
    plain = _best()
    assert apply_ally_wait(_board(STRIKE, [NEUTRALIZE]), plain) is plain
