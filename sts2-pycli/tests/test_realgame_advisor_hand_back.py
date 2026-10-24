"""The advisor's "hand back" verdict for decisions it cannot actually make (GH #874).

F7 used to play every such screen anyway -- the first option of an unranked
surface, a borrowed head's ranking -- and the player had no way to tell it was
happening. The advisor now says so on ``advice.hand_back``, and the mod's
``pause_on: "unsupported"`` stops autoplay there. The verdict lives here rather
than in a mod-side list so that a surface gaining a trained head stops pausing
on its own; these tests pin which answers carry it and which must not.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import realgame_advisor as ra

_RUN_STATE: dict[str, Any] = {"character": "Ironclad", "hp": 60, "max_hp": 75, "gold": 120}


class _StubPolicy:
    def __init__(self, probs: list[float], *, raises: Exception | None = None) -> None:
        self.probs = probs
        self.raises = raises

    def distribution(self, obs: Any, actions: Any, metrics: Any) -> tuple[list[float], float]:
        if self.raises is not None:
            raise self.raises
        return list(self.probs), 0.1


class _StubPolicies:
    def __init__(self, policy: Any, error: str | None = None) -> None:
        self.policy = policy
        self.error = error

    def get(self, character: str | None) -> tuple[Any, str | None]:
        return self.policy, self.error


def _rest_site() -> dict[str, Any]:
    return {
        "type": "rest_site",
        "options": [{"label": "heal", "index": 0}, {"label": "smith", "index": 1}],
        "run_state": _RUN_STATE,
    }


def _relics(advice_for: str) -> dict[str, Any]:
    return {
        "type": advice_for,
        "options": [
            {"index": 0, "relic": {"id": "ANCHOR", "name": "Anchor"}},
            {"index": 1, "relic": {"id": "LANTERN", "name": "Lantern"}},
        ],
        "run_state": _RUN_STATE,
    }


def _bundles() -> dict[str, Any]:
    return {
        "type": "card_bundle",
        "options": [
            {"index": 0, "cards": [{"id": "STRIKE_IRONCLAD", "name": "Strike"}]},
            {"index": 1, "cards": [{"id": "BASH", "name": "Bash"}]},
        ],
        "run_state": _RUN_STATE,
    }


class TestTrainedSurfacesPlayOn:
    def test_a_ranked_screen_carries_no_hand_back(self) -> None:
        advice = ra.compute_screen_advice(
            _rest_site(), "rest_site", _StubPolicies(_StubPolicy([0.3, 0.7]))
        )
        assert advice["best"]["label"] == "smith"
        assert "hand_back" not in advice

    def test_the_boss_relic_head_on_its_own_surface_carries_none(self) -> None:
        advice = ra.compute_screen_advice(
            _relics("boss_relic"), "boss_relic", _StubPolicies(_StubPolicy([0.6, 0.4]))
        )
        assert "hand_back" not in advice


class TestUncoveredSurfacesHandBack:
    def test_a_surface_with_no_head_hands_back(self) -> None:
        # The crystal sphere has no builder (GH #872): the unranked answer
        # still names a pick for unattended runs, and says it is not one.
        state = {"type": "crystal_sphere", "options": [{"label": "Cell (0,0)"}]}
        advice = ra.compute_screen_advice(
            state, "crystal_sphere", _StubPolicies(_StubPolicy([1.0]))
        )
        assert advice["hand_back"]
        assert advice["best"]["index"] == 0, "unattended runs still need something to play"

    def test_the_unranked_fallback_always_hands_back(self) -> None:
        with_options = ra._unranked({"options": [{"label": "A"}]}, "no trained head for widget")
        without = ra._unranked({}, "no trained head for widget")
        assert with_options["hand_back"] == "no trained head for widget"
        assert without["hand_back"] == "no trained head for widget"

    @pytest.mark.parametrize(
        ("advice_for", "state"),
        [("card_bundle", _bundles())],
    )
    def test_a_borrowed_head_hands_back_despite_ranking(
        self, advice_for: str, state: dict[str, Any]
    ) -> None:
        # GH #379: ranked by a different question's head. The ranking still
        # renders (the player may want it), but it is not the AI deciding.
        advice = ra.compute_screen_advice(state, advice_for, _StubPolicies(_StubPolicy([0.2, 0.8])))
        assert advice["best"]["index"] == 1
        assert advice["hand_back"] == ra._HAND_BACK_SURFACES[advice_for]

    def test_no_net_hands_back(self) -> None:
        advice = ra.compute_screen_advice(
            _rest_site(), "rest_site", _StubPolicies(None, "checkpoint missing")
        )
        assert "best" not in advice
        assert advice["hand_back"]

    def test_an_inference_failure_hands_back(self) -> None:
        policy = _StubPolicy([], raises=RuntimeError("budget exceeded"))
        advice = ra.compute_screen_advice(_rest_site(), "rest_site", _StubPolicies(policy))
        assert advice["hand_back"]
        assert "inference failed" in advice["hand_back"]


class _Client:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    def send_action(self, action: dict[str, Any]) -> None:
        self.sent.append(action)


def test_an_advisor_error_hands_back(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("bad snapshot")

    monkeypatch.setattr(ra, "compute_screen_advice", _boom)
    client = _Client()
    ra._handle_advice_request(
        client,
        ra.RecommendConfig(),
        SimpleNamespace(sims=10, boss_sims=20, log_search_tree=0),
        SimpleNamespace(write=lambda kind, data: None),
        SimpleNamespace(log_decision=lambda msg, advice_for: None, log_advice=lambda *a, **k: None),
        {**_rest_site(), "type": "advice_request", "advice_for": "rest_site", "request_id": "r1"},
        None,
        session={},
    )
    (reply,) = client.sent
    assert "advisor error" in reply["advice"]["note"]
    assert reply["advice"]["hand_back"]


def test_a_chest_relic_is_played_not_handed_back() -> None:
    # Personal build: the boss-relic head's chest pick is taken in auto mode.
    advice = ra.compute_screen_advice(_relics("treasure"), "treasure", _StubPolicies(_StubPolicy([0.2, 0.8])))
    assert advice["best"]["index"] == 1
    assert "hand_back" not in advice
