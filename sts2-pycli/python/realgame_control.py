"""Localhost control plane inside the live-game advisor (rewind lane).

The advisor owns the single TCP connection to the SpireMentor mod, so anything
else that wants to talk to the live game — the workbench's ``/api/bridge/*``
proxies — goes through this tiny HTTP server (``127.0.0.1:<control-port>``,
default 8079; 8078 belongs to the standalone SimLab). One command at a time:
the bridge's rewind verbs are fire-and-forget with broadcast replies, so a
command registers a waiter, sends the verb, and blocks (bounded) until the
advisor's receive loop routes the matching reply back here.

Endpoints (all JSON):
    GET  /status       — {connected, character, seed, floor, act, log_path}
    GET  /tree         — LiveRunTree.timeline()  (BranchTree.tsx wire shape)
    GET  /snapshots    — the mod's snapshot list (list_snapshots round trip)
    GET  /state        — the latest ``advice_request`` from the mod: the same
                         message the mod's Copy state yields, so the
                         workbench's state import takes it as is (404 until
                         the first one, and again after game over)
    POST /restore      — {snapshot_id} or {node_id}; rewinds the live game
    POST /search_tree  — companion MCTS tree over the current combat board

Browser access (GH #709). The Python workbench reaches this server through a
same-origin proxy; the browser-only workbench (``https://your-workbench.example.com``)
has no server to proxy for it and must call this loopback server from an
``https://`` page. That crosses three browser gates — mixed content (a
loopback ``http://`` target is exempt as a potentially trustworthy origin),
Chrome's Private Network Access preflight, and CORS — and the last two are
answered here, OPT-IN: ``start(port, allow_origins=[...])`` names the page
origins allowed to call in (``--control-allow-origin`` on the advisor). With
no origin listed the server sends no CORS headers at all, so no page anywhere
can read a reply, exactly as before. Same-origin proxies are unaffected.
"""

from __future__ import annotations

import copy
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from realgame_livetree import LiveRunTree

logger = logging.getLogger(__name__)

#: Message types the advisor's receive loop must route to the control plane.
BRIDGE_MESSAGE_TYPES = (
    "snapshots",
    "snapshot_saved",
    "snapshot_restored",
    "combat_undone",
    "error",
)


class _Waiter:
    def __init__(self) -> None:
        self.event = threading.Event()
        self.payload: dict[str, Any] | None = None

    def fulfill(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.event.set()


class ControlPlane:
    """State shared between the advisor loop and the HTTP handler threads."""

    def __init__(self, tree: LiveRunTree) -> None:
        self.tree = tree
        #: Page origins allowed to call in from a browser (``start`` sets it).
        self.allow_origins: tuple[str, ...] = ()
        self._lock = threading.Lock()
        self._client: Any = None
        self._status: dict[str, Any] = {"connected": False}
        self._waiters: dict[str, _Waiter] = {}
        self._last_state: dict[str, Any] | None = None
        self._server: ThreadingHTTPServer | None = None
        self.search_tree_fn: Any = None  # set by the advisor: (sims|None) -> dict

    # -- called from the advisor loop ----------------------------------

    def set_client(self, client: Any) -> None:
        with self._lock:
            self._client = client
            self._status["connected"] = client is not None

    def note_message(self, msg: dict[str, Any]) -> None:
        """Track the latest run envelope for /status (any message type), and
        the latest decision position for /state."""
        rs = msg.get("run_state") or {}
        mtype = msg.get("type")
        # Copied: the advisor's handlers may annotate the message they get.
        state = copy.deepcopy(msg) if mtype == "advice_request" else None
        with self._lock:
            if state is not None:
                self._last_state = state
            elif mtype in ("game_over", "run_complete"):
                self._last_state = None
            self._status.update(
                {
                    "connected": True,
                    "character": rs.get("character") or self._status.get("character"),
                    "seed": msg.get("seed") or self._status.get("seed"),
                    "floor": msg.get("floor", self._status.get("floor")),
                    "act": msg.get("act", self._status.get("act")),
                    "state_id": msg.get("state_id", self._status.get("state_id")),
                }
            )

    def last_state(self) -> dict[str, Any] | None:
        """The latest ``advice_request`` (see /state), or None."""
        with self._lock:
            return self._last_state

    def set_log_path(self, path: str | None) -> None:
        with self._lock:
            self._status["log_path"] = path

    def on_bridge_message(self, msg: dict[str, Any]) -> None:
        """Route a rewind-lane broadcast from the mod (see BRIDGE_MESSAGE_TYPES)."""
        mtype = msg.get("type")
        if mtype == "snapshot_saved":
            self.tree.on_snapshot_saved(msg)
            return
        if mtype == "snapshots":
            self._fulfill("snapshots", msg)
            return
        if mtype == "snapshot_restored":
            self.tree.on_snapshot_restored(msg)
            self._fulfill("restore", msg)
            return
        if mtype == "combat_undone":
            self.tree.on_combat_undone(msg)
            self._fulfill("restore", msg)
            return
        if mtype == "error" and (
            msg.get("snapshot_id") is not None or str(msg.get("error") or "").startswith("undo_")
        ):
            self._fulfill("restore", msg)

    def _fulfill(self, kind: str, payload: dict[str, Any]) -> None:
        with self._lock:
            waiter = self._waiters.pop(kind, None)
        if waiter is not None:
            waiter.fulfill(payload)

    # -- command round trips (HTTP handler threads) --------------------

    def _round_trip(
        self, kind: str, action: dict[str, Any], timeout: float
    ) -> dict[str, Any] | None:
        with self._lock:
            client = self._client
            if client is None:
                return None
            waiter = _Waiter()
            self._waiters[kind] = waiter
        try:
            client.send_action(action, echo_state_id=False)
        except Exception as exc:
            with self._lock:
                self._waiters.pop(kind, None)
            logger.warning("control command %s failed: %s", action.get("action"), exc)
            return None
        if not waiter.event.wait(timeout):
            with self._lock:
                self._waiters.pop(kind, None)
            return {"type": "error", "error": "timeout", "detail": f"{kind} timed out"}
        return waiter.payload

    def list_snapshots(self) -> dict[str, Any] | None:
        return self._round_trip("snapshots", {"action": "list_snapshots"}, timeout=5.0)

    def restore(self, snapshot_id: str) -> dict[str, Any] | None:
        return self._round_trip(
            "restore",
            {"action": "restore_snapshot", "snapshot_id": snapshot_id},
            timeout=20.0,
        )

    def undo_turn(self, turn: int, tape_id: str | None = None) -> dict[str, Any] | None:
        action: dict[str, Any] = {"action": "undo_turn", "turn": int(turn)}
        if tape_id:
            action["tape_id"] = tape_id
        # A persisted-tape replay reloads a room AND fast-forwards a whole
        # combat prefix; give it more room than a plain restore.
        return self._round_trip("restore", action, timeout=60.0)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._status)

    # -- HTTP server ---------------------------------------------------

    def start(self, port: int, allow_origins: Any = ()) -> None:
        """Serve on ``127.0.0.1:port``.

        Args:
            port: TCP port (``0`` = ephemeral; see :attr:`port`).
            allow_origins: Page origins (``https://your-workbench.example.com``)
                allowed to call this server from a browser. Each gets the CORS
                headers and the Private Network Access opt-in on every reply
                and on the preflight; anything else gets none. Default: none.
        """
        plane = self
        self.allow_origins = tuple(
            o.strip().rstrip("/") for o in (allow_origins or ()) if str(o).strip()
        )

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                logger.debug("control: " + format, *args)

            def _cors_headers(self) -> None:
                """CORS + Private Network Access, for an allow-listed Origin only.

                The allow-list is exact-match on the page origin; the reply
                echoes it (never ``*``), with ``Vary: Origin`` so a cache keeps
                the two answers apart. ``Access-Control-Allow-Private-Network``
                is Chrome's PNA opt-in: a public-origin page fetching a
                loopback address preflights and needs this on the preflight
                reply (https://wicg.github.io/private-network-access/).
                """
                origin = self.headers.get("Origin")
                if not origin or origin.rstrip("/") not in plane.allow_origins:
                    return
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "content-type")
                self.send_header("Access-Control-Allow-Private-Network", "true")
                self.send_header("Access-Control-Max-Age", "600")
                self.send_header("Vary", "Origin")

            def _send(self, code: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload, default=str).encode("utf-8")
                self.send_response(code)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self._cors_headers()
                self.end_headers()
                self.wfile.write(body)

            def do_OPTIONS(self) -> None:
                """The CORS / PNA preflight. 204 with the headers for an
                allow-listed origin; 403 (no CORS headers) for any other, so
                the browser refuses the real request before it is sent."""
                origin = self.headers.get("Origin")
                if origin and origin.rstrip("/") in plane.allow_origins:
                    self.send_response(204)
                    self.send_header("content-length", "0")
                    self._cors_headers()
                    self.end_headers()
                else:
                    self._send(403, {"error": "origin not allowed"})

            def _body(self) -> dict[str, Any]:
                length = int(self.headers.get("content-length") or 0)
                if length <= 0:
                    return {}
                try:
                    return json.loads(self.rfile.read(length)) or {}
                except json.JSONDecodeError:
                    return {}

            def do_GET(self) -> None:
                path = self.path.split("?", 1)[0]
                if path == "/status":
                    self._send(200, plane.status())
                elif path == "/tree":
                    self._send(200, plane.tree.timeline())
                elif path == "/state":
                    state = plane.last_state()
                    if state is None:
                        self._send(
                            404,
                            {
                                "error": "no_state",
                                "detail": "no decision from the game yet "
                                "(the mod sends one at each combat turn or choice screen)",
                            },
                        )
                    else:
                        self._send(200, state)
                elif path == "/snapshots":
                    reply = plane.list_snapshots()
                    if reply is None:
                        self._send(503, {"error": "bridge not connected"})
                    else:
                        self._send(200, reply)
                else:
                    self._send(404, {"error": f"unknown path {path}"})

            def do_POST(self) -> None:
                path = self.path.split("?", 1)[0]
                body = self._body()
                if path == "/restore":
                    snapshot_id = body.get("snapshot_id")
                    node_id = body.get("node_id")
                    if snapshot_id is None and node_id is not None:
                        plan = plane.tree.seek_plan(int(node_id))
                        if plan is None:
                            self._send(
                                409,
                                {
                                    "error": "not_seekable",
                                    "detail": "that node carries no game save point "
                                    "(the game only saves at room entry and "
                                    "post-combat rewards)",
                                },
                            )
                            return
                        if plan["action"] == "undo_turn":
                            reply = plane.undo_turn(plan["turn"], plan.get("tape_id"))
                            if reply is None:
                                self._send(503, {"error": "bridge not connected"})
                            elif reply.get("type") == "combat_undone":
                                self._send(200, reply)
                            else:
                                self._send(409, reply)
                            return
                        snapshot_id = plan["snapshot_id"]
                    if not snapshot_id:
                        self._send(400, {"error": "missing snapshot_id/node_id"})
                        return
                    reply = plane.restore(str(snapshot_id))
                    if reply is None:
                        self._send(503, {"error": "bridge not connected"})
                    elif reply.get("type") == "snapshot_restored":
                        self._send(200, reply)
                    else:
                        self._send(409, reply)
                elif path == "/search_tree":
                    fn = plane.search_tree_fn
                    if fn is None:
                        self._send(503, {"error": "search unavailable"})
                        return
                    try:
                        tree = fn(body.get("sims"))
                    except LookupError as exc:
                        self._send(409, {"error": "no_combat_state", "detail": str(exc)})
                        return
                    except Exception as exc:
                        self._send(500, {"error": "search_failed", "detail": str(exc)})
                        return
                    self._send(200, tree)
                else:
                    self._send(404, {"error": f"unknown path {path}"})

        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        except OSError as exc:
            logger.warning("control plane could not bind 127.0.0.1:%d: %s", port, exc)
            return
        self._server = server
        thread = threading.Thread(target=server.serve_forever, name="advisor-control", daemon=True)
        thread.start()
        logger.info(
            "control plane listening on 127.0.0.1:%d (browser origins: %s)",
            self.port or port,
            ", ".join(self.allow_origins) or "none",
        )

    @property
    def port(self) -> int | None:
        """The bound port (differs from the requested one when it was 0)."""
        return self._server.server_address[1] if self._server else None

    def stop(self) -> None:
        """Shut the HTTP server down (tests; the advisor never stops it)."""
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
