"""`sync.mods_ok`: the human-play corpus keeps no gameplay mod and no coach.

From 0.7.4 the coach declares `affects_gameplay: false`, so the flag alone
no longer drops its runs (autoplay plays them, undo rewinds them); the mod
id must. Both corpus filters (`sync_census.eligible`, `pro.sync_eligible`)
go through this one helper.
"""

from __future__ import annotations

import pytest
from ppo.bc_human import sync


def test_no_mods_is_ok() -> None:
    assert sync.mods_ok({})
    assert sync.mods_ok({"mods": []})


def test_a_cosmetic_mod_is_ok() -> None:
    assert sync.mods_ok({"mods": [{"id": "PrettyCards", "affects_gameplay": False}]})


def test_a_gameplay_mod_is_dropped() -> None:
    assert not sync.mods_ok({"mods": [{"id": "MoreRelics", "affects_gameplay": True}]})


@pytest.mark.parametrize("mod_id", sorted(sync.COACH_MOD_IDS))
def test_the_coach_is_dropped_even_when_it_claims_no_gameplay(mod_id: str) -> None:
    assert not sync.mods_ok({"mods": [{"id": mod_id, "affects_gameplay": False}]})
