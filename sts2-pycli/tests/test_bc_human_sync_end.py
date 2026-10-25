"""A fight is in sync only if the engine's combat ends where the log's does.

`sync.replay` and `pro.fight_turns` stopped at the engine's combat end and
called the fight clean, even with log decisions left: an engine that won
early (a phantom relic's damage, ...) read as in sync. A stub engine that
wins on its first action pins both.
"""

from __future__ import annotations

from typing import Any

import pytest
from ppo.bc_human import sync
from ppo.turnplan import pro


class _WinsAtOnce:
    """Combat state stub: no enemies, no hand; the first action wins."""

    def __init__(self) -> None:
        self.over = False

    def get_state(self) -> dict[str, Any]:
        return {"type": "combat", "player": {"hp": 80}, "enemies": [], "hand": []}

    def clone(self) -> _WinsAtOnce:
        return self

    def apply_step(self, action: dict[str, Any]) -> str:
        self.over = True
        return "combat_won"

    def terminal_value(self) -> tuple[bool, float]:
        return (self.over, 1.0)


def _rec(n_end_turns: int) -> dict[str, Any]:
    return {
        "hp": 80,
        "character": "IRONCLAD",
        "enemies": [],
        "belt": [],
        "pre": [],
        "decisions": [{"ev": {"t": "end_turn", "side": "player"}, "after": []}] * n_end_turns,
    }


@pytest.fixture(autouse=True)
def _stub_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sync, "build", lambda rec, monster_ai_offset=0: _WinsAtOnce())


def test_replay_flags_an_engine_that_ends_before_the_log() -> None:
    out = sync.replay(_rec(2))
    assert out["desync"] is not None and out["desync"][0] == "ended_early"
    assert out["n_ok"] == 1 and len(out["pairs"]) == 1  # the pair before the end is still faithful


def test_replay_is_clean_when_both_end_together() -> None:
    out = sync.replay(_rec(1))
    assert out["desync"] is None and out["n_ok"] == 1


def test_fight_turns_flags_an_engine_that_ends_before_the_log() -> None:
    turns, desync = pro.fight_turns(_rec(2))
    assert desync is not None and desync[0] == "ended_early"
    assert [t["status"] for t in turns] == ["diverged:ended_early"]


def test_fight_turns_is_clean_when_both_end_together() -> None:
    turns, desync = pro.fight_turns(_rec(1))
    assert desync is None
    assert [t["status"] for t in turns] == ["legal"]
