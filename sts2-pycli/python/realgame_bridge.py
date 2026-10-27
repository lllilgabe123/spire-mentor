"""TCP client for the real-game "SpireMentor" mod (real-game bridge phase 0).

Speaks the newline-delimited UTF-8 JSON protocol documented in
``sts2-rl-agent/docs/PROTOCOL.md``: the mod (running inside the live Slay the
Spire 2 process) sends a state message carrying a ``request_id`` and blocks
until exactly one action message echoing that ``request_id`` is received.

This module is intentionally stdlib-only (``socket``, ``json``, ``time``,
``logging``) so it has no dependency on the game repo or on
``sts2-rl-agent`` (the donor implementation this was ported from — see
``sts2-rl-agent/sts2_env/bridge/client.py`` — is not imported).

Deviations from the donor client, to match our mod's actual protocol:
  - State messages use ``"type"`` (e.g. ``"combat_action"``), not
    ``"phase"``/``"game_state"``.
  - ``receive_state()`` returns *every* message verbatim (including
    ``"pong"``, ``"game_over"``, ``"run_complete"``, ``"error"``) rather than
    silently swallowing some types — callers dispatch on ``"type"``
    themselves. Only malformed (non-JSON) lines are skipped, with a warning
    logged.
  - Our mod waits indefinitely for a response (no 30s random-play
    fallback), so this client has no artificial urgency; it still responds
    promptly, but a slow decision does not desync the protocol.
  - Unknown fields on any message (e.g. ``"seed"``, ``"floor"``, ``"act"``,
    ``"mod_version"``) are ignored gracefully everywhere — this client never
    validates a message's field set, it only reads the fields it needs.

v2 (phase 1b) protocol additions handled here:
  - ``error`` notifications (``{"type":"error","error":...,"detail":...}``)
    are fire-and-forget, like ``game_over``/``run_complete`` — no
    ``request_id`` is expected or consumed. They flow through
    :meth:`receive_state` like any other message; the client additionally
    remembers the most recent one on :attr:`RealGameBridgeClient.last_error`
    as a convenience flag (reset to ``None`` as soon as a non-error message
    arrives). Per PROTOCOL.md section 3a, an ``error`` is always immediately
    followed by a fresh re-send of the decision it rejected — callers just
    keep looping on :meth:`receive_state` and will see that decision next.
  - :meth:`send_action` now also auto-echoes ``state_id`` (the stale-state
    guard, section 4) from the most recently received message, mirroring the
    existing ``request_id`` auto-echo. Opt out per-call with
    ``echo_state_id=False``.
  - :meth:`start_run` is a convenience wrapper for the ``run_config`` ->
    ``start_run`` handshake (section 3/4), only relevant when the mod's
    ``client_run_config`` flag is enabled.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9002


class RealGameBridgeClient:
    """TCP client for the SpireMentor mod's blocking request/response protocol.

    Usage::

        with RealGameBridgeClient() as client:
            while True:
                state = client.receive_state()
                if state["type"] == "combat_action":
                    client.send_action({"action": "end_turn"})
                elif state["type"] in ("game_over", "run_complete"):
                    break
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        timeout: float = 60.0,
        reconnect_attempts: int = 30,
        reconnect_delay: float = 2.0,
    ) -> None:
        """Initialize the client (does not connect yet).

        Args:
            host: Bridge mod TCP server host.
            port: Bridge mod TCP server port.
            timeout: Socket timeout in seconds for receive operations. The
                mod itself waits indefinitely for a response, but the
                socket-level timeout here still guards against a genuinely
                dead connection (e.g. the game process crashed).
            reconnect_attempts: Number of connection attempts in
                :meth:`connect` before giving up (the game may not have
                started its TCP server yet).
            reconnect_delay: Seconds to sleep between connection attempts.
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay

        self._sock: socket.socket | None = None
        self._buffer: bytes = b""
        self._connected: bool = False
        self._last_request_id: str | None = None
        self._last_state_id: int | None = None
        self._last_error: dict[str, Any] | None = None
        #: One writer at a time: the advisor loop, the control plane's HTTP
        #: threads and the coach's grader thread all send on this socket, and
        #: interleaved `sendall`s would splice two JSON lines together.
        self._send_lock = threading.Lock()

    # ----------------------------------------------------------------
    # Connection management
    # ----------------------------------------------------------------

    def connect(self) -> None:
        """Connect to the bridge mod's TCP server, retrying on refusal.

        Raises:
            ConnectionError: If all ``reconnect_attempts`` attempts fail.
        """
        last_error: Exception | None = None
        for attempt in range(1, self.reconnect_attempts + 1):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(self.timeout)
                sock.connect((self.host, self.port))
                self._sock = sock
                self._buffer = b""
                self._connected = True
                logger.info(
                    "Connected to real-game bridge at %s:%d (attempt %d/%d)",
                    self.host,
                    self.port,
                    attempt,
                    self.reconnect_attempts,
                )
                return
            except OSError as e:
                last_error = e
                logger.warning(
                    "Connection attempt %d/%d to %s:%d failed: %s",
                    attempt,
                    self.reconnect_attempts,
                    self.host,
                    self.port,
                    e,
                )
                self._cleanup_socket()
                if attempt < self.reconnect_attempts:
                    time.sleep(self.reconnect_delay)

        raise ConnectionError(
            f"Could not connect to real-game bridge at {self.host}:{self.port} "
            f"after {self.reconnect_attempts} attempts"
        ) from last_error

    def disconnect(self) -> None:
        """Close the socket, if open. Safe to call multiple times."""
        self._cleanup_socket()
        self._connected = False
        logger.info("Disconnected from real-game bridge.")

    def reconnect(self) -> None:
        """Disconnect (if connected) and connect again."""
        self.disconnect()
        self.connect()

    @property
    def connected(self) -> bool:
        """Whether the client currently believes it holds an open socket."""
        return self._connected

    @property
    def last_error(self) -> dict[str, Any] | None:
        """The most recently received ``error`` message, or ``None``.

        Doubles as a convenience "did the last action get rejected?" flag:
        it is set on every ``{"type":"error",...}`` message (section 3a) and
        cleared back to ``None`` as soon as any other message arrives —
        including the fresh re-send of the decision the error was rejecting.
        """
        return self._last_error

    # ----------------------------------------------------------------
    # Message I/O
    # ----------------------------------------------------------------

    def receive_state(self) -> dict[str, Any]:
        """Block until the next newline-terminated JSON message arrives.

        Returns every message verbatim (state messages of any ``"type"``,
        including ``"pong"``, ``"game_over"``, ``"run_complete"``) — callers
        are responsible for dispatching on ``"type"``. Malformed (non-JSON)
        lines are logged and skipped rather than raised.

        If the message carries a ``"request_id"`` and/or ``"state_id"``, it
        is remembered so the next call to :meth:`send_action` can auto-echo
        it. If the message is an ``"error"`` notification (section 3a), it
        is additionally stashed on :attr:`last_error`; any other message
        type clears :attr:`last_error` back to ``None``.

        Returns:
            The parsed message as a dict.

        Raises:
            ConnectionError: If the connection is closed by the peer.
            TimeoutError: If no data arrives within ``timeout`` seconds.
        """
        while True:
            line = self._receive_line()
            if line is None:
                raise ConnectionError("Connection lost while waiting for state")
            if not line:
                # Blank line (e.g. stray newline) — nothing to parse.
                continue

            try:
                data: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("Invalid JSON from bridge: %s (raw: %s)", e, line[:200])
                continue

            if "request_id" in data:
                self._last_request_id = data["request_id"]
            if "state_id" in data:
                self._last_state_id = data["state_id"]
            self._last_error = data if data.get("type") == "error" else None
            return data

    def send_action(self, action: dict[str, Any], *, echo_state_id: bool = True) -> None:
        """Send an action message, auto-echoing ``request_id`` and ``state_id``.

        Per PROTOCOL.md, ``request_id`` should be echoed on all non-ping
        actions when the triggering state carried one. If ``action`` already
        has a ``"request_id"`` key, it is left untouched.

        **Phase 1b**: the same auto-echo now also applies to ``state_id``
        (the stale-state guard, section 4) — set ``echo_state_id=False`` to
        send the action without it (e.g. to deliberately exercise the
        server's ``error:"stale_state"`` rejection path). As with
        ``request_id``, an explicit ``"state_id"`` already in ``action`` is
        left untouched.

        Args:
            action: Action dict, e.g. ``{"action": "end_turn"}``. Not
                mutated — a copy is sent.
            echo_state_id: If False, never auto-add ``state_id`` even when
                one is known. Has no effect if ``action`` already carries an
                explicit ``"state_id"``.

        Raises:
            ConnectionError: If not connected, or the socket write fails.
        """
        if not self._connected or self._sock is None:
            raise ConnectionError("Not connected to real-game bridge")

        payload = dict(action)
        if "request_id" not in payload and self._last_request_id is not None:
            payload["request_id"] = self._last_request_id
        if echo_state_id and "state_id" not in payload and self._last_state_id is not None:
            payload["state_id"] = self._last_state_id

        try:
            data = json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n"
            with self._send_lock:
                self._sock.sendall(data)
            logger.debug("Sent action: %s", payload)
        except OSError as e:
            self._connected = False
            raise ConnectionError(f"Lost connection while sending action: {e}") from e

    def ping(self) -> bool:
        """Send ``{"action": "ping"}`` and block for the ``"pong"`` reply.

        Returns:
            True if a ``"pong"`` message was received, False on any
            connection error.
        """
        try:
            self.send_action({"action": "ping"})
            reply = self.receive_state()
            return reply.get("type") == "pong"
        except (ConnectionError, TimeoutError):
            return False

    def start_run(self, character: str, ascension: int = 0, seed: int | None = None) -> None:
        """Reply to a ``run_config`` message with a ``start_run`` action.

        Only meaningful when the mod's ``client_run_config`` config flag is
        enabled (PROTOCOL.md section 3/4) — the mod blocks indefinitely for
        a valid ``start_run`` in that mode instead of auto-selecting a run.
        A rejected ``start_run`` (unknown ``character``, out-of-range
        ``ascension``) gets a structured ``error`` (see :attr:`last_error`)
        followed by a fresh ``run_config`` re-send; this method does not
        retry on your behalf.

        Args:
            character: Must match one of ``run_config.characters[].id``
                (case-insensitive exact match preferred, substring fallback
                per PROTOCOL.md), e.g. ``"IRONCLAD"``.
            ascension: ``0``-``max_ascension`` (10 as of v0.107.1).
            seed: Explicit run seed, or ``None`` to let the mod generate a
                fresh random one (same semantics as the old
                ``SpireMentor.conf``-driven flow).

        Raises:
            ConnectionError: If not connected, or the socket write fails.
        """
        self.send_action(
            {
                "action": "start_run",
                "character": character,
                "ascension": ascension,
                "seed": seed,
            }
        )

    # ----------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------

    def newer_pending(self, mtype: str) -> bool:
        """Whether a complete ``mtype`` message is already queued behind the
        one just received. Drains what the socket holds without blocking, so
        nothing waits; the queued messages stay buffered for
        :meth:`receive_state`, in order.

        Personal build: the mod re-asks for advice on every board change, so
        in co-op a teammate's turn queues one request per card they play and
        the advisor answered each in turn (7 in a row, ~15 s behind the
        board). A request with a newer one behind it is outdated.
        """
        if self._sock is None:
            return False
        try:
            self._sock.setblocking(False)
            while True:
                try:
                    chunk = self._sock.recv(65536)
                except (BlockingIOError, InterruptedError):
                    break
                except OSError:
                    break
                if not chunk:
                    break
                self._buffer += chunk
        finally:
            try:
                self._sock.settimeout(self.timeout)
            except OSError:
                pass
        needle = f'"{mtype}"'.encode()
        for raw in self._buffer.split(b"\n")[:-1]:
            if needle not in raw:
                continue
            try:
                if json.loads(raw).get("type") == mtype:
                    return True
            except (ValueError, AttributeError):
                continue
        return False

    def _receive_line(self) -> str | None:
        """Read one newline-terminated line, buffering partial reads.

        Returns:
            The decoded line (without the trailing newline), or None if the
            peer closed the connection.

        Raises:
            TimeoutError: If no data arrives within ``timeout`` seconds.
        """
        if self._sock is None:
            return None

        while b"\n" not in self._buffer:
            try:
                chunk = self._sock.recv(8192)
            except TimeoutError as e:
                raise TimeoutError(f"No data received within {self.timeout}s timeout") from e
            except OSError as e:
                self._connected = False
                logger.error("Socket error during receive: %s", e)
                return None

            if not chunk:
                # Peer closed the connection.
                self._connected = False
                return None

            self._buffer += chunk

        line, self._buffer = self._buffer.split(b"\n", 1)
        return line.decode("utf-8", errors="replace").strip()

    def _cleanup_socket(self) -> None:
        """Close and discard the socket, if any."""
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        self._buffer = b""

    # ----------------------------------------------------------------
    # Context manager support
    # ----------------------------------------------------------------

    def __enter__(self) -> RealGameBridgeClient:
        self.connect()
        return self

    def __exit__(self, *args: object) -> None:
        self.disconnect()
