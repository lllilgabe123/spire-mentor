"""The live game's build routes the advisor's engine (GH #800).

SpireMentor puts ``game_version`` (``"v0.107.1"`` / ``"v0.111.0"``, from the
game's ``ReleaseInfoManager``) on every envelope (PROTOCOL.md section 1). The
advisor must build its engine for that build, refuse a build it has no engine
for, and -- the invariant -- treat a missing field or v0.107.1 exactly as it
did before the field existed.

Pure dict tests run everywhere; the engine-routing tests need the ``sts2_rs``
wheel from this checkout (``scripts/wheel-venv.sh build``).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import realgame_advisor as ra
from realgame_adapter import (
    UnsupportedGameVersionError,
    envelope_game_version,
    to_headless_combat_play,
)
from realgame_coach import InlineExecutor
from sts2_hint import adapter

try:
    import sts2_rs  # type: ignore[import-not-found]

    _SKIP_ENGINE = None
except ImportError as _e:  # pragma: no cover - depends on the venv
    sts2_rs = None  # type: ignore[assignment]
    _SKIP_ENGINE = f"sts2_rs not importable: {_e}"

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "realgame_adapter"
FIXTURES = [
    "single_enemy_ironclad.json",
    "multi_enemy_ironclad.json",
    "player_power_ironclad.json",
    "orbs_defect.json",
]


def _board(name: str = "single_enemy_ironclad.json", **envelope: Any) -> dict[str, Any]:
    with (FIXTURE_DIR / name).open(encoding="utf-8") as fh:
        state = json.load(fh)["state"]
    state.pop("game_version", None)
    state.update(envelope)
    return state


def _wire(state: dict[str, Any]) -> dict[str, Any]:
    character = {"IRONCLAD": "Ironclad", "DEFECT": "Defect"}.get(
        (state.get("run_state") or {}).get("character") or "", "Ironclad"
    )
    return adapter.translate(to_headless_combat_play(state), character=character)


# ---------------------------------------------------------------------------
# Envelope parsing
# ---------------------------------------------------------------------------


class TestEnvelopeGameVersion:
    def test_absent_means_the_stable_build(self) -> None:
        assert envelope_game_version({}) == "v0.107.1"

    @pytest.mark.parametrize("raw", [None, "", "   "])
    def test_null_or_blank_means_the_stable_build(self, raw: Any) -> None:
        assert envelope_game_version({"game_version": raw}) == "v0.107.1"

    @pytest.mark.parametrize(
        ("raw", "want"),
        [
            ("v0.107.1", "v0.107.1"),
            ("0.107.1", "v0.107.1"),
            ("v0.111.0", "v0.111.0"),
            ("0.111.0", "v0.111.0"),
            (" v0.111.0 ", "v0.111.0"),
        ],
    )
    def test_supported_builds_normalize(self, raw: str, want: str) -> None:
        assert envelope_game_version({"game_version": raw}) == want

    @pytest.mark.parametrize("raw", ["v0.112.0", "v0.103.2", "v0.111.0-beta", "banana"])
    def test_an_unknown_build_is_refused_by_name(self, raw: str) -> None:
        with pytest.raises(UnsupportedGameVersionError) as info:
            envelope_game_version({"game_version": raw})
        msg = str(info.value)
        assert raw in msg
        assert "v0.107.1" in msg and "v0.111.0" in msg, "the refusal lists what IS supported"
        assert info.value.raw == raw

    @pytest.mark.parametrize("raw", [111, 0.111, ["v0.111.0"], True])
    def test_a_non_string_is_refused(self, raw: Any) -> None:
        with pytest.raises(UnsupportedGameVersionError):
            envelope_game_version({"game_version": raw})

    def test_the_refusal_is_a_value_error(self) -> None:
        # Every existing caller catches broadly; ValueError keeps that working.
        assert issubclass(UnsupportedGameVersionError, ValueError)


# ---------------------------------------------------------------------------
# Adapter routing: bridge board -> headless dict -> engine wire
# ---------------------------------------------------------------------------


class TestAdapterCarriesTheBuild:
    @pytest.mark.parametrize("fixture", FIXTURES)
    def test_absent_and_stable_yield_the_identical_wire(self, fixture: str) -> None:
        # The invariant: a v0.107.1 envelope changes NOTHING downstream.
        before = _wire(_board(fixture))
        stable = _wire(_board(fixture, game_version="v0.107.1"))
        assert stable == before
        assert "game_version" not in before
        assert "game_version" not in to_headless_combat_play(
            _board(fixture, game_version="v0.107.1")
        )

    @pytest.mark.parametrize("fixture", FIXTURES)
    def test_the_beta_build_reaches_the_wire_and_nothing_else_moves(self, fixture: str) -> None:
        stable = _wire(_board(fixture))
        beta_headless = to_headless_combat_play(_board(fixture, game_version="0.111.0"))
        assert beta_headless["game_version"] == "v0.111.0"
        beta = _wire(_board(fixture, game_version="0.111.0"))
        assert beta.pop("game_version") == "v0.111.0"
        assert beta == stable

    def test_an_unknown_build_never_reaches_the_engine(self) -> None:
        with pytest.raises(UnsupportedGameVersionError):
            to_headless_combat_play(_board(game_version="v0.200.0"))

    def test_translate_refuses_an_unknown_build_from_any_producer(self) -> None:
        headless = to_headless_combat_play(_board())
        headless["game_version"] = "v9.9.9"
        with pytest.raises(adapter.UnsupportedStateError, match=r"\$\.game_version"):
            adapter.translate(headless, character="Ironclad")

    def test_translate_input_is_not_mutated(self) -> None:
        headless = to_headless_combat_play(_board(game_version="v0.111.0"))
        snapshot = copy.deepcopy(headless)
        adapter.translate(headless, character="Ironclad")
        assert headless == snapshot


@pytest.mark.skipif(_SKIP_ENGINE is not None, reason=_SKIP_ENGINE or "")
class TestEngineIsBuiltForTheBuild:
    def test_absent_builds_the_stable_engine(self) -> None:
        cs = sts2_rs.state_from_dict(_wire(_board()))
        assert cs.game_version == "v0.107.1"

    def test_stable_builds_the_stable_engine(self) -> None:
        cs = sts2_rs.state_from_dict(_wire(_board(game_version="v0.107.1")))
        assert cs.game_version == "v0.107.1"

    def test_beta_builds_the_beta_engine(self) -> None:
        cs = sts2_rs.state_from_dict(_wire(_board(game_version="v0.111.0")))
        assert cs.game_version == "v0.111.0"
        # And the round trip keeps it, so the mirror's carried state does too.
        assert sts2_rs.state_to_dict(cs).get("game_version") == "v0.111.0"

    def test_beta_only_card_in_hand_builds(self) -> None:
        # GH #929: the DowsingRod's quest card (a v0.111.0-only id) in hand
        # failed every advice request with UnsupportedStateError.
        board = _board(game_version="v0.111.0")
        card = board["hand"][0]
        card.update(id="DOWSING", name="Dowsing", type="Quest", rarity="Quest")
        card.update(target_type="None", cost=-1, base_cost=-1, effective_cost=-1)
        cs = sts2_rs.state_from_dict(_wire(board))
        assert "DOWSING" in json.dumps(sts2_rs.state_to_dict(cs))


# ---------------------------------------------------------------------------
# The advisor's request handler
# ---------------------------------------------------------------------------


class _Client:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    def send_action(self, reply: dict[str, Any]) -> None:
        self.sent.append(reply)


class _Mirror:
    def __init__(self) -> None:
        self.dropped: list[str] = []

    def drop(self, why: str) -> None:
        self.dropped.append(why)


def _handler_deps() -> tuple[_Client, SimpleNamespace, SimpleNamespace, SimpleNamespace]:
    args = SimpleNamespace(sims=10, boss_sims=20, log_search_tree=0)
    log = SimpleNamespace(write=lambda kind, data: None)
    wlog = SimpleNamespace(
        log_decision=lambda msg, advice_for: None,
        log_advice=lambda *a, **k: None,
    )
    return _Client(), args, log, wlog


def _request(**envelope: Any) -> dict[str, Any]:
    msg = _board(**envelope)
    msg.update(
        {"type": "advice_request", "advice_for": "combat_action", "request_id": "a1", "state_id": 7}
    )
    return msg


@pytest.fixture
def fake_search(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace the search with a recorder; returns the boards it was asked about."""
    seen: list[dict[str, Any]] = []

    def _fake(state: dict[str, Any], cfg: Any, sims: int, mirror: Any) -> tuple[Any, Any, Any, Any]:
        seen.append(state)
        return {"note": "searched", "options": []}, object(), cfg, []

    monkeypatch.setattr(ra, "compute_combat_advice", _fake)
    ra.LAST_COMBAT.clear()
    return seen


class TestAdvisorRouting:
    def test_an_unsupported_build_is_refused_with_a_message_the_hud_shows(
        self, fake_search: list[dict[str, Any]]
    ) -> None:
        client, args, log, wlog = _handler_deps()
        ra._handle_advice_request(
            client,
            ra.RecommendConfig(),
            args,
            log,
            wlog,
            _request(game_version="v0.112.0"),
            None,
            session={},
            mirror=_Mirror(),
        )
        assert fake_search == [], "no search may run on an engine for the wrong build"
        (reply,) = client.sent
        assert reply["action"] == "advice"
        assert reply["request_id"] == "a1" and reply["state_id"] == 7
        advice = reply["advice"]
        assert "v0.112.0" in advice["note"]
        assert "supported" in advice["note"]
        assert advice["unsupported_game_version"] == "v0.112.0"
        assert "best" not in advice and "options" not in advice

    @pytest.mark.parametrize("envelope", [{}, {"game_version": "v0.107.1"}])
    def test_absent_or_stable_searches_exactly_as_before(
        self, fake_search: list[dict[str, Any]], envelope: dict[str, Any]
    ) -> None:
        client, args, log, wlog = _handler_deps()
        session: dict[str, Any] = {}
        ra._handle_advice_request(
            client,
            ra.RecommendConfig(),
            args,
            log,
            wlog,
            _request(**envelope),
            None,
            session=session,
            mirror=_Mirror(),
        )
        (board,) = fake_search
        assert client.sent[0]["advice"]["note"] == "searched"
        assert "game_version" not in to_headless_combat_play(board)
        assert session["game_version"] == "v0.107.1"

    def test_the_beta_build_is_searched_on_the_beta_engine(
        self, fake_search: list[dict[str, Any]]
    ) -> None:
        client, args, log, wlog = _handler_deps()
        ra._handle_advice_request(
            client,
            ra.RecommendConfig(),
            args,
            log,
            wlog,
            _request(game_version="v0.111.0"),
            None,
            session={},
            mirror=_Mirror(),
        )
        (board,) = fake_search
        assert to_headless_combat_play(board)["game_version"] == "v0.111.0"

    def test_a_build_change_drops_carried_state(self, fake_search: list[dict[str, Any]]) -> None:
        client, args, log, wlog = _handler_deps()
        session: dict[str, Any] = {}
        mirror = _Mirror()
        for ver in ("v0.107.1", "v0.107.1", "v0.111.0"):
            ra._handle_advice_request(
                client,
                ra.RecommendConfig(),
                args,
                log,
                wlog,
                _request(game_version=ver),
                None,
                session=session,
                mirror=mirror,
            )
        assert mirror.dropped == ["game build changed v0.107.1 -> v0.111.0"]
        assert session["game_version"] == "v0.111.0"
        _, cs, _ = ra.LAST_COMBAT.get()
        assert cs is not None, "the cache is cleared BEFORE the new build's search refills it"


# ---------------------------------------------------------------------------
# The coach capability (PROTOCOL.md v5)
# ---------------------------------------------------------------------------


@pytest.fixture
def ranked_search(monkeypatch: pytest.MonkeyPatch) -> None:
    """A search whose rows crown end turn -- enough for the coach to grade."""

    def _fake(state: dict[str, Any], cfg: Any, sims: int, mirror: Any) -> tuple[Any, Any, Any, Any]:
        rows = [
            {"stable_idx": -1, "target_index": None, "win_prob": 0.9, "visits": 300,
             "is_best": True, "mean_final_hp": 50.0, "hp_n": 300, "action": "end turn"},
            {"stable_idx": 0, "target_index": None, "win_prob": 0.8, "visits": 200,
             "is_best": False, "mean_final_hp": 45.0, "hp_n": 200, "action": "Defend"},
        ]  # fmt: skip
        return {"note": "searched", "options": []}, object(), cfg, rows

    monkeypatch.setattr(ra, "compute_combat_advice", _fake)
    ra.LAST_COMBAT.clear()


def _coach_for(client: _Client, wlog: Any) -> Any:
    args = SimpleNamespace(coach_catchup_sims=10, coach_budget_s=1.0)
    return ra.build_coach(client, wlog, args, executor=InlineExecutor())


def _end_turn() -> dict[str, Any]:
    return {
        "type": "human_action",
        "decision_type": "combat_action",
        "action": "end_turn",
        "detail": "End turn",
        "advice_request_id": "a1",
        "position_id": 7,
    }


class TestCoachCapability:
    def _serve(self, request: dict[str, Any]) -> tuple[_Client, list[dict[str, Any]]]:
        client, args, log, wlog = _handler_deps()
        logged: list[dict[str, Any]] = []
        wlog.log_coach = logged.append
        coach = _coach_for(client, wlog)
        ra._handle_advice_request(
            client, ra.RecommendConfig(), args, log, wlog, request, None,
            session={}, mirror=_Mirror(), coach=coach,
        )  # fmt: skip
        coach.on_human_action(_end_turn(), None, node_id=3)
        return client, logged

    def test_advice_reply_advertises_coach_2(self, ranked_search: None) -> None:
        client, _ = self._serve(_request())
        reply = client.sent[0]
        assert reply["action"] == "advice"
        assert reply["capabilities"] == {"coach": 2}

    def test_combat_ended_routes_to_coach(self) -> None:
        msgs = iter([{"type": "combat_ended", "victory": True, "hp": 60, "round": 4}])
        got: list[dict[str, Any]] = []

        def _receive() -> dict[str, Any]:
            try:
                return next(msgs)
            except StopIteration:
                raise ConnectionError("done") from None

        client = SimpleNamespace(receive_state=_receive)
        control = SimpleNamespace(note_message=lambda m: None, set_log_path=lambda p: None)
        wlog = SimpleNamespace(path=None)
        log = SimpleNamespace(write=lambda kind, data: None)
        coach = SimpleNamespace(on_combat_ended=got.append)
        with pytest.raises(ConnectionError):
            ra._serve_connected(
                client, ra.RecommendConfig(), SimpleNamespace(), log, wlog, None,
                None, control, _Mirror(), {}, coach,
            )  # fmt: skip
        assert got == [{"type": "combat_ended", "victory": True, "hp": 60, "round": 4}]

    def test_no_coach_message_to_a_mod_without_capability(self, ranked_search: None) -> None:
        client, logged = self._serve(_request())  # an older mod: no capabilities
        assert [m["action"] for m in client.sent] == ["advice"]
        assert [r["verdict"] for r in logged] == ["best"], "the run log still gets the grade"

    def test_a_mod_with_the_capability_gets_the_coach_message(self, ranked_search: None) -> None:
        client, logged = self._serve(_request(capabilities={"coach": 1}))
        coach_msgs = [m for m in client.sent if m["action"] == "coach"]
        (msg,) = coach_msgs
        assert msg["request_id"] is None and msg["state_id"] is None
        assert msg["coach"] == logged[0]
        assert msg["coach"]["kind"] == "turn" and msg["coach"]["node_id"] == 3
