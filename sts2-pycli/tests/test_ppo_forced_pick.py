"""`ppo.forced_pick`: the arms override exactly the card rewards they claim to.

No engine: a stub policy and hand-built decisions."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from ppo.forced_pick import ForcedPick, compare


class _Pol:
    """Prefers the LAST candidate, so a forced pick is visible as a deviation."""

    def distribution(self, obs, actions, metrics):
        n = len(actions)
        return [(i + 1) / (n * (n + 1) / 2) for i in range(n)], 0.0


def _reward(offered: list[str], floor: int = 3) -> SimpleNamespace:
    actions = [{"action": "pick_card_reward", "idx": i} for i in range(len(offered))]
    actions.append({"action": "pick_card_reward", "idx": None})  # skip, last = greedy
    return SimpleNamespace(
        obs={"card_reward_option_ids": offered}, actions=actions, metrics={"total_floor": floor}
    )


def _path() -> SimpleNamespace:
    return SimpleNamespace(
        obs={}, actions=[{"action": "choose_path", "idx": 0}] * 2, metrics={"total_floor": 1}
    )


def test_none_is_the_greedy_champion():
    f = ForcedPick("none")
    assert f.choose(_Pol(), _reward(["BURNING_PACT", "INFLAME"])) == 2  # the skip
    assert f.stats()["forced"] == 0 and f.stats()["engine_offers"] == 1
    assert f.choose(_Pol(), _path()) == 1


def test_single_forces_only_the_first_engine_reward():
    f = ForcedPick("single")
    assert f.choose(_Pol(), _reward(["INFLAME", "INFLAME"])) == 2  # no engine card offered
    assert f.choose(_Pol(), _reward(["BURNING_PACT", "INFLAME", "OFFERING"])) == 2  # OFFERING
    assert f.choose(_Pol(), _reward(["BURNING_PACT", "INFLAME"])) == 2  # skip again
    st = f.stats()
    assert st["forced"] == 1 and st["engine_offers"] == 2
    assert [t["card"] for t in st["taken"]] == ["OFFERING"]


def test_package_forces_every_engine_reward():
    f = ForcedPick("package")
    for _ in range(3):
        assert f.choose(_Pol(), _reward(["BURNING_PACT", "INFLAME"])) == 0
    assert f.stats()["forced"] == 3


def test_unknown_arm_is_refused():
    with pytest.raises(ValueError, match="arm"):
        ForcedPick("always")


def test_compare_pairs_on_shared_seeds():
    base = {1: {"won": True}, 2: {"won": False}, 3: {"won": False}}
    arm = {1: {"won": False}, 2: {"won": True}, 3: {"won": True}, 4: {"won": True}}
    c = compare(base, arm, "won")
    assert (c["n"], c["only_base"], c["only_arm"], c["delta_pp"]) == (3, 1, 2, 33.33)


def test_half_flips_a_seeded_coin_per_engine_reward():
    def picks(seed: int) -> list[int]:
        f = ForcedPick("half", seed=seed)
        return [f.choose(_Pol(), _reward(["BURNING_PACT", "INFLAME"])) for _ in range(40)]

    assert picks(7) == picks(7)
    assert picks(7) != picks(8)
    assert 10 < picks(7).count(0) < 30
