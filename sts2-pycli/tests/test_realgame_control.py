"""ControlPlane — the advisor's localhost HTTP server for the rewind lane.

Ported from the offline smoke harness (fake-mod round trip incl. fork and
409 non-seekable) that validated the lane before it shipped 2026-08-21. A
FakeClient stands in for the SpireMentor mod: it answers the rewind verbs the
way the mod would, from another thread, exercising the waiter/round-trip
machinery for real over HTTP on an ephemeral port.
"""

from __future__ import annotations

import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from realgame_control import ControlPlane
from realgame_livetree import LiveRunTree

ENV: dict[str, Any] = {
    "seed": "S1",
    "run_state": {"character": "IRONCLAD"},
    "act": 1,
    "floor": 1,
}


class FakeClient:
    """Answers rewind verbs the way the mod would, from another thread."""

    def __init__(self, control: ControlPlane) -> None:
        self.control = control
        self.sent: list[dict[str, Any]] = []
        #: Optional override: action dict -> reply dict (tests inject errors).
        self.reply_fn: Any = None

    def send_action(self, action: dict[str, Any], *, echo_state_id: bool = True) -> None:
        self.sent.append(action)
        if self.reply_fn is not None:
            reply = self.reply_fn(action)
        else:
            kind = action.get("action")
            if kind == "list_snapshots":
                reply = {
                    "type": "snapshots",
                    "run_start_time": 111,
                    "snapshots": [
                        {"id": "111-0", "seq": 0, "floor": 1, "post_combat": False},
                        {"id": "111-1", "seq": 1, "floor": 1, "post_combat": True},
                    ],
                }
            elif kind == "restore_snapshot":
                reply = {
                    "type": "snapshot_restored",
                    "snapshot_id": action["snapshot_id"],
                    "floor": 1,
                    "state_id": 999,
                }
            elif kind == "undo_turn":
                reply = {
                    "type": "combat_undone",
                    "floor": 1,
                    "turn": action["turn"],
                    "tape_id": action.get("tape_id"),
                    "replayed_events": 5,
                    "state_id": 1000,
                }
            else:
                raise AssertionError(kind)
        # The mod's replies are broadcasts routed back by the advisor loop —
        # from a different thread than the HTTP handler that is waiting.
        threading.Timer(0.02, lambda: self.control.on_bridge_message(reply)).start()


@pytest.fixture()
def plane():
    tree = LiveRunTree()
    control = ControlPlane(tree)
    client = FakeClient(control)
    control.set_client(client)
    control.start(0)  # ephemeral port
    assert control.port
    yield control, tree, client
    control.stop()


def _get(control: ControlPlane, path: str) -> dict[str, Any]:
    with urllib.request.urlopen(f"http://127.0.0.1:{control.port}{path}", timeout=5) as r:
        return json.loads(r.read())


def _post(control: ControlPlane, path: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    req = urllib.request.Request(
        f"http://127.0.0.1:{control.port}{path}",
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status or 0, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _play_room(control: ControlPlane, tree: LiveRunTree) -> None:
    """Room entry snapshot, one combat decision, one reward decision."""
    control.on_bridge_message(
        {
            "type": "snapshot_saved",
            "snapshot_id": "111-0",
            "seq": 0,
            "floor": 1,
            "act": 1,
            "post_combat": False,
        }
    )
    tree.on_decision({**ENV, "turn": 1, "request_id": "r1"}, "combat_action")
    tree.on_human_action({"detail": "Strike -> Cultist", "advice_request_id": "r1"})
    control.on_bridge_message(
        {
            "type": "snapshot_saved",
            "snapshot_id": "111-1",
            "seq": 1,
            "floor": 1,
            "act": 1,
            "post_combat": True,
            "has_tape": True,
        }
    )
    tree.on_decision({**ENV, "request_id": "r2"}, "card_reward")


class TestReadEndpoints:
    def test_tree_and_snapshots(self, plane) -> None:
        control, tree, _ = plane
        _play_room(control, tree)
        t = _get(control, "/tree")
        assert [n["decision"] for n in t["nodes"]] == [
            "root",
            "room_entry",
            "combat_action",
            "card_reward",
        ]
        combat = next(n for n in t["nodes"] if n["decision"] == "combat_action")
        assert combat["label"] == "f1 · Strike -> Cultist"
        snaps = _get(control, "/snapshots")
        assert [s["id"] for s in snaps["snapshots"]] == ["111-0", "111-1"]

    def test_status_reports_connection(self, plane) -> None:
        control, _, _ = plane
        assert _get(control, "/status")["connected"] is True


#: A Copy-state-shaped decision message for the /state tests.
ADVICE_REQUEST: dict[str, Any] = {
    **ENV,
    "type": "advice_request",
    "advice_for": "combat_action",
    "state_id": 7,
    "hand": [{"id": "STRIKE_IRONCLAD"}],
}


class TestState:
    """GET /state: the latest advice_request, verbatim (what Copy state yields)."""

    def test_404_before_any_decision(self, plane) -> None:
        control, _, _ = plane
        with pytest.raises(urllib.error.HTTPError) as exc:
            _get(control, "/state")
        assert exc.value.code == 404
        assert json.loads(exc.value.read())["error"] == "no_state"

    def test_serves_latest_advice_request_verbatim(self, plane) -> None:
        control, _, _ = plane
        control.note_message({**ADVICE_REQUEST, "state_id": 6})
        control.note_message(ADVICE_REQUEST)
        # Other message types do not replace the position.
        control.note_message({**ENV, "type": "human_action", "action": "play_card"})
        assert _get(control, "/state") == ADVICE_REQUEST

    def test_stored_copy_is_isolated_from_later_mutation(self, plane) -> None:
        control, _, _ = plane
        msg = json.loads(json.dumps(ADVICE_REQUEST))
        control.note_message(msg)
        msg["hand"].append({"id": "DEFEND_IRONCLAD"})
        assert _get(control, "/state") == ADVICE_REQUEST

    @pytest.mark.parametrize("end", ["game_over", "run_complete"])
    def test_cleared_at_run_end(self, plane, end: str) -> None:
        control, _, _ = plane
        control.note_message(ADVICE_REQUEST)
        control.note_message({**ENV, "type": end})
        with pytest.raises(urllib.error.HTTPError) as exc:
            _get(control, "/state")
        assert exc.value.code == 404


class TestSeek:
    def test_room_entry_seek_round_trip_and_fork(self, plane) -> None:
        control, tree, client = plane
        _play_room(control, tree)
        room = next(n for n in _get(control, "/tree")["nodes"] if n["decision"] == "room_entry")
        code, reply = _post(control, "/restore", {"node_id": room["id"]})
        assert (code, reply["type"]) == (200, "snapshot_restored")
        assert client.sent[-1] == {"action": "restore_snapshot", "snapshot_id": "111-0"}
        assert _get(control, "/tree")["active_leaf"] == room["id"]
        # The next decision forks a sibling branch at the restored node.
        tree.on_decision({**ENV, "request_id": "r3"}, "combat_action")
        t = _get(control, "/tree")
        children = [n for n in t["nodes"] if n["parent_id"] == room["id"]]
        assert len(children) == 2

    def test_combat_node_seeks_via_undo_turn_with_tape(self, plane) -> None:
        control, tree, client = plane
        _play_room(control, tree)
        combat = next(
            n for n in _get(control, "/tree")["nodes"] if n["decision"] == "combat_action"
        )
        assert combat["turn"] == 1 and combat["tape_id"] == "111-1"
        code, reply = _post(control, "/restore", {"node_id": combat["id"]})
        assert (code, reply["type"]) == (200, "combat_undone")
        assert client.sent[-1] == {"action": "undo_turn", "turn": 1, "tape_id": "111-1"}
        # Undo to turn 1 = the fight's start: active lands on the room entry.
        t = _get(control, "/tree")
        room = next(n for n in t["nodes"] if n["decision"] == "room_entry")
        assert t["active_leaf"] == room["id"]

    def test_live_combat_node_seeks_without_tape(self, plane) -> None:
        control, tree, client = plane
        control.on_bridge_message(
            {
                "type": "snapshot_saved",
                "snapshot_id": "111-0",
                "seq": 0,
                "floor": 1,
                "act": 1,
                "post_combat": False,
            }
        )
        tree.on_decision({**ENV, "turn": 2, "request_id": "r1"}, "combat_action")
        combat = next(
            n for n in _get(control, "/tree")["nodes"] if n["decision"] == "combat_action"
        )
        code, reply = _post(control, "/restore", {"node_id": combat["id"]})
        assert (code, reply["type"]) == (200, "combat_undone")
        assert client.sent[-1] == {"action": "undo_turn", "turn": 2}

    def test_undo_error_is_a_409_not_a_timeout(self, plane) -> None:
        control, tree, client = plane
        control.on_bridge_message(
            {
                "type": "snapshot_saved",
                "snapshot_id": "111-0",
                "seq": 0,
                "floor": 1,
                "act": 1,
                "post_combat": False,
            }
        )
        tree.on_decision({**ENV, "turn": 1, "request_id": "r1"}, "combat_action")
        client.reply_fn = lambda action: {
            "type": "error",
            "error": "undo_not_in_combat",
            "detail": "no combat in progress",
        }
        combat = next(
            n for n in _get(control, "/tree")["nodes"] if n["decision"] == "combat_action"
        )
        code, reply = _post(control, "/restore", {"node_id": combat["id"]})
        assert (code, reply.get("error")) == (409, "undo_not_in_combat")

    def test_node_without_save_point_or_turn_is_409(self, plane) -> None:
        control, tree, _ = plane
        _play_room(control, tree)
        reward = next(n for n in _get(control, "/tree")["nodes"] if n["decision"] == "card_reward")
        code, reply = _post(control, "/restore", {"node_id": reward["id"]})
        assert (code, reply.get("error")) == (409, "not_seekable")

    def test_missing_body_is_400(self, plane) -> None:
        control, _, _ = plane
        code, _reply = _post(control, "/restore", {})
        assert code == 400


# ---------------------------------------------------------------------------
# Browser access (GH #709): the CORS + Private Network Access opt-in
# ---------------------------------------------------------------------------

ORIGIN = "https://your-workbench.example.com"


def _request(
    control: ControlPlane, method: str, path: str, headers: dict[str, str]
) -> tuple[int, dict[str, str]]:
    req = urllib.request.Request(
        f"http://127.0.0.1:{control.port}{path}", method=method, headers=headers
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status or 0, {k.lower(): v for k, v in r.headers.items()}
    except urllib.error.HTTPError as e:
        return e.code, {k.lower(): v for k, v in e.headers.items()}


@pytest.fixture()
def browser_plane():
    tree = LiveRunTree()
    control = ControlPlane(tree)
    control.set_client(FakeClient(control))
    control.start(0, allow_origins=[ORIGIN + "/"])  # a trailing slash is tolerated
    assert control.port
    yield control
    control.stop()


class TestBrowserAccess:
    def test_default_sends_no_cors_headers(self, plane) -> None:
        control, _, _ = plane
        code, headers = _request(control, "GET", "/status", {"Origin": ORIGIN})
        assert code == 200
        assert "access-control-allow-origin" not in headers
        # The preflight is refused outright, so a browser never sends the POST.
        code, headers = _request(
            control,
            "OPTIONS",
            "/restore",
            {"Origin": ORIGIN, "Access-Control-Request-Method": "POST"},
        )
        assert code == 403
        assert "access-control-allow-origin" not in headers

    def test_allow_listed_origin_gets_cors_and_pna_headers(self, browser_plane) -> None:
        control = browser_plane
        code, headers = _request(
            control,
            "OPTIONS",
            "/restore",
            {
                "Origin": ORIGIN,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
                "Access-Control-Request-Private-Network": "true",
            },
        )
        assert code == 204
        assert headers["access-control-allow-origin"] == ORIGIN
        assert "POST" in headers["access-control-allow-methods"]
        assert headers["access-control-allow-headers"] == "content-type"
        assert headers["access-control-allow-private-network"] == "true"
        assert headers["vary"] == "Origin"
        # …and on the real replies, echoed (never `*`).
        code, headers = _request(control, "GET", "/status", {"Origin": ORIGIN})
        assert code == 200
        assert headers["access-control-allow-origin"] == ORIGIN

    def test_other_origins_get_nothing(self, browser_plane) -> None:
        control = browser_plane
        code, headers = _request(control, "GET", "/status", {"Origin": "https://evil.example"})
        assert code == 200  # the same-machine proxy path is unaffected…
        assert "access-control-allow-origin" not in headers  # …but a page cannot read it
        code, _ = _request(
            control,
            "OPTIONS",
            "/restore",
            {"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
        )
        assert code == 403
        # No Origin at all (curl, the proxy): plain replies, no CORS headers.
        code, headers = _request(control, "GET", "/status", {})
        assert code == 200
        assert "access-control-allow-origin" not in headers
