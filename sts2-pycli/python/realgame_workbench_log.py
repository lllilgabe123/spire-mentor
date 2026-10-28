"""Workbench-schema run logging for the live-game advisor (phase 9).

Writes each advised live run as a ``game_log.GameLogger`` JSONL file —
``{step, ts, type: meta|state|action|mcts, data}`` — in the *headless state
dialect*, so the workbench loads it with zero changes: ``/api/runs`` lists
it, ``/api/load`` parses it, the Board renders combat steps, the EventLog
labels the human's moves, and the mcts records carry the advice that was on
screen for each decision.

This rides alongside (never replaces) the raw-wire ``AdvisorLog``: that one
is the debugging tape (``tools/ascension_climb.py`` byte-slices it), this
one is the review surface.

Mapping notes
-------------
* ``combat_action`` payloads go through
  :func:`realgame_adapter.to_headless_combat_play` — the exact dict the
  workbench's replay UI was built on.
* Screen payloads (map/card reward/event/...) get thin per-surface mappers
  to the headless decision names (``map_select``, ``card_reward``,
  ``event_choice``, ...) with the fields the workbench actually reads
  (``choices``/``cards``/``options`` + ``player`` + ``context``).
* ``human_action`` messages carry only a display string (``detail``), not
  structured args, so the action record carries it as ``label`` and
  ``render.describe_action`` prefers a ``label`` when present.
* One ``state`` per *distinct* decision: the mod re-requests advice when a
  screen repaints, so consecutive identical boards (ignoring volatile
  envelope fields) are dropped.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import re
import threading
from collections.abc import Callable
from typing import Any, cast

from game_log import GameLogger
from realgame_adapter import character_display_name, to_headless_combat_play

logger = logging.getLogger(__name__)


def _locked[F: Callable[..., Any]](fn: F) -> F:
    """Serialize a logger entry point: the coach's grader thread writes
    ``coach`` records while the advisor loop writes everything else."""

    @functools.wraps(fn)
    def wrapper(self: WorkbenchRunLogger, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return fn(self, *args, **kwargs)

    return cast("F", wrapper)


def run_log_name(path: str | None) -> str | None:
    """A run log's name as the control plane's ``GET /log`` serves it.

    The bare filename with anything outside ``[A-Za-z0-9._-]`` replaced: the
    bundled page's run store keys by it (``engineRoutes.ts`` RUN_FILENAME),
    and coach records carry it so the mod can link a decision to its file.
    """
    if not path:
        return None
    return re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(path))


#: Envelope fields that change on every message even when the board did not.
_VOLATILE_KEYS = ("state_id", "request_id", "mod_version", "type", "advice_for")


def _context(msg: dict[str, Any]) -> dict[str, Any]:
    return {
        "act": msg.get("act"),
        "floor": msg.get("floor"),
        "total_floor": msg.get("floor"),
        "room_type": msg.get("room_type") or "",
    }


def _player(msg: dict[str, Any]) -> dict[str, Any]:
    rs = msg.get("run_state") or {}
    deck = rs.get("master_deck") or []
    return {
        "name": character_display_name(rs.get("character")) or (rs.get("character") or "?"),
        "hp": rs.get("hp"),
        "max_hp": rs.get("max_hp"),
        "block": 0,
        "gold": rs.get("gold"),
        "potion_slots": rs.get("potion_slot_count"),
        "relics": rs.get("relics") or [],
        "potions": [p for p in (rs.get("potions") or []) if p],
        "deck": deck,
        "deck_size": len(deck),
    }


def _options_row(o: dict[str, Any]) -> dict[str, Any]:
    """A wire options[] entry as the ``options`` row the workbench reads.

    ``render.describe_action``'s ``choose_option`` looks for
    ``text``/``name``/``title``/``option_id``; the wire carries ``label``.
    """
    row = dict(o)
    label = o.get("label")
    if label and not any(k in o for k in ("text", "name", "title", "option_id")):
        row["title"] = label
    return row


def _base_state(msg: dict[str, Any], decision: str) -> dict[str, Any]:
    return {
        "type": "decision",
        "decision": decision,
        # NOT "context": that key is the card_select PURPOSE STRING in native
        # workbench logs (render.py describe_action, CardSelect.tsx) and a
        # dict here crashed /api/step (unhashable membership test).
        "live_context": _context(msg),
        # Top-level floor/act so TimelineStore's carry-forward labelling works
        # even when the log starts mid-run (no earlier map_select to seed it).
        "floor": msg.get("floor"),
        "act": msg.get("act"),
        # WHICH act, which the index cannot say: act 1 ships two models
        # (Overgrowth / Underdocks, GH #544). Every painted region the frontend
        # picks -- combat backdrop, card-reward scene, rest-site clearing, map
        # parchment -- is keyed by this name, and `pickScene` falls back to the
        # act-1 Overgrowth room when it is missing. So a replayed live run used
        # to paint Overgrowth for all three acts.
        "act_name": msg.get("act_name") or None,
        "player": _player(msg),
    }


def _wire_map_to_full_map(m: dict[str, Any]) -> dict[str, Any] | None:
    """The wire map graph as the frontend's ``FullMap`` shape.

    Wire (PROTOCOL.md map_select): a flat ``nodes`` list plus ``current`` /
    ``boss`` / ``visited`` coordinate lists. The workbench MapView renders
    ``{rows: node[][], boss, current_coord}`` with per-node ``visited`` /
    ``current`` flags (web/src/api.ts FullMap) -- regroup accordingly.
    """
    nodes = [n for n in (m.get("nodes") or []) if isinstance(n, dict)]
    if not nodes:
        return None
    visited = {
        (v.get("col"), v.get("row")) for v in (m.get("visited") or []) if isinstance(v, dict)
    }
    cur = m.get("current") if isinstance(m.get("current"), dict) else None
    cur_key = (cur.get("col"), cur.get("row")) if cur else None
    by_row: dict[Any, list[dict[str, Any]]] = {}
    for n in nodes:
        node: dict[str, Any] = {
            "col": n.get("col"),
            "row": n.get("row"),
            "type": n.get("type"),
            "children": n.get("children") or [],
        }
        key = (node["col"], node["row"])
        if key in visited:
            node["visited"] = True
        if key == cur_key:
            node["current"] = True
        by_row.setdefault(n.get("row"), []).append(node)
    rows = [
        sorted(by_row[r], key=lambda x: (x.get("col") is None, x.get("col")))
        for r in sorted(by_row, key=lambda r: (r is None, r))
    ]
    out: dict[str, Any] = {"rows": rows}
    boss = m.get("boss")
    if isinstance(boss, dict):
        out["boss"] = {"col": boss.get("col"), "row": boss.get("row"), "type": "Boss"}
    if cur:
        out["current_coord"] = {"col": cur.get("col"), "row": cur.get("row")}
    return out


def _map_select(msg: dict[str, Any]) -> dict[str, Any]:
    state = _base_state(msg, "map_select")
    state["choices"] = [
        {"col": n.get("col"), "row": n.get("row"), "type": n.get("type")}
        for n in (msg.get("nodes") or [])
        if isinstance(n, dict)
    ]
    if isinstance(msg.get("map"), dict):
        full_map = _wire_map_to_full_map(msg["map"])
        if full_map:
            state["full_map"] = full_map
    return state


def _card_reward(msg: dict[str, Any]) -> dict[str, Any]:
    state = _base_state(msg, "card_reward")
    state["cards"] = msg.get("cards") or []
    state["can_skip"] = bool(msg.get("can_skip", True))
    return state


def _card_select(msg: dict[str, Any]) -> dict[str, Any]:
    state = _base_state(msg, "card_select")
    state["cards"] = msg.get("cards") or []
    for key in ("min_select", "max_select", "cancelable", "prompt_key"):
        if msg.get(key) is not None:
            state[key] = msg[key]
    return state


def _event(msg: dict[str, Any]) -> dict[str, Any]:
    state = _base_state(msg, "event_choice")
    options = [_options_row(o) for o in (msg.get("options") or []) if isinstance(o, dict)]
    state["options"] = options
    event_id = next((o.get("event_id") for o in options if o.get("event_id")), None)
    if event_id:
        state["event_name"] = event_id
    return state


def _options_only(decision: str):
    def build(msg: dict[str, Any]) -> dict[str, Any]:
        state = _base_state(msg, decision)
        state["options"] = [
            _options_row(o) for o in (msg.get("options") or []) if isinstance(o, dict)
        ]
        return state

    return build


def _shop(msg: dict[str, Any]) -> dict[str, Any]:
    state = _options_only("shop")(msg)
    for key in ("player_gold", "purge_cost"):
        if msg.get(key) is not None:
            state[key] = msg[key]
    # Shelves for the workbench Shop component (Shop.tsx renders
    # ``cards``/``relics``/``potions`` + ``card_removal_cost``; the generic
    # ``options`` list above only feeds action labelling). Prefer the full
    # ``stock`` (every slot, sold-out and unaffordable included); older mod
    # builds only sent the purchasable-only options -- fall back to those.
    rows: list[tuple[str, Any, bool, dict[str, Any], dict[str, Any]]] = []
    for r in msg.get("stock") or []:
        if isinstance(r, dict) and r.get("kind"):
            rows.append(
                (r["kind"], r.get("price"), bool(r.get("stocked", True)), r.get("item") or {}, r)
            )
    if not rows:
        for o in msg.get("options") or []:
            if isinstance(o, dict) and o.get("action") in (
                "buy_card",
                "buy_relic",
                "buy_potion",
                "remove_card",
            ):
                rows.append(
                    (
                        o["action"],
                        o.get("price"),
                        bool(o.get("enabled", True)),
                        o.get("item") or {},
                        o,
                    )
                )
    cards: list[dict[str, Any]] = []
    relics: list[dict[str, Any]] = []
    potions: list[dict[str, Any]] = []
    for kind, price, stocked, item, raw in rows:
        if kind == "buy_card":
            # Shop.tsx shows ``cost`` as the GOLD price tag and ``card_cost``
            # as the energy badge -- the wire card's ``cost`` is energy.
            card = dict(item)
            card["card_cost"] = card.get("cost")
            card["cost"] = price
            card["index"] = len(cards)
            card["is_stocked"] = stocked
            cards.append(card)
        elif kind == "buy_relic":
            relics.append(
                {
                    **item,
                    "index": len(relics),
                    "name": item.get("name") or raw.get("label"),
                    "cost": price,
                    "is_stocked": stocked,
                }
            )
        elif kind == "buy_potion":
            potions.append(
                {
                    **item,
                    "index": len(potions),
                    "name": item.get("name") or raw.get("label"),
                    "cost": price,
                    "is_stocked": stocked,
                }
            )
        elif kind == "remove_card":
            state["card_removal_cost"] = price
    state["cards"] = cards
    state["relics"] = relics
    state["potions"] = potions
    if "card_removal_cost" not in state and state.get("purge_cost") is not None:
        state["card_removal_cost"] = state["purge_cost"]
    return state


def _card_bundle(msg: dict[str, Any]) -> dict[str, Any]:
    state = _base_state(msg, "card_bundle")
    state["bundles"] = msg.get("bundles") or []
    return state


def _terminal(msg: dict[str, Any]) -> dict[str, Any]:
    state = _base_state(msg, "game_over")
    state["victory"] = bool(msg.get("victory"))
    return state


_BUILDERS: dict[str, Any] = {
    "combat_action": None,  # special-cased: to_headless_combat_play
    "map_select": _map_select,
    "card_reward": _card_reward,
    "card_select": _card_select,
    "event": _event,
    "rest_site": _options_only("rest_site"),
    "shop": _shop,
    "treasure": _options_only("treasure"),
    "boss_relic": _options_only("boss_relic"),
    "card_bundle": _card_bundle,
    "crystal_sphere": _options_only("crystal_sphere"),
    "reward_screen": _options_only("reward_screen"),
}


class WorkbenchRunLogger:
    """One workbench-schema JSONL per live run, opened lazily.

    A run is identified by ``(character, seed)`` from the message envelope; a
    change (or a terminal message) closes the current file and the next
    decision opens a fresh one. Never raises into the advisor loop.
    """

    def __init__(self, log_dir: str | None) -> None:
        #: Falsy dir disables the whole logger (``--workbench-logs ''``).
        self._dir = log_dir or None
        self._log: GameLogger | None = None
        self._run_key: tuple[str, str] | None = None
        self._last_sig: str | None = None
        self._pending_advice_sig: str | None = None
        #: The most recent logged decision state (headless dialect), which a
        #: human action's machine form is resolved against.
        self._last_state: dict[str, Any] | None = None
        #: ``state`` lines written to the open file: the next one's replay
        #: step index (``TimelineStore`` / ``runlog.rs`` give each state line
        #: one step, in file order).
        self._state_count = 0
        #: The log :meth:`close` last closed: still named by :attr:`log_name`,
        #: and the death autopsy (built after the run end closed it) is
        #: appended to it.
        self._closed_log: GameLogger | None = None
        #: Combat-log rows of decisions that wrote no ``state`` line (a
        #: repeated board): they ride on the next one, so the rows of a fight
        #: stay complete (``coop_replay`` replays plays from them).
        self._pending_rows: list[dict[str, Any]] = []
        self._lock = threading.RLock()

    @property
    def path(self) -> str | None:
        return self._log.path if self._log else None

    @property
    def log_name(self) -> str | None:
        """The open run log's name as ``GET /log`` serves it, else the name of
        the file closed last (:func:`run_log_name`)."""
        closed = self._closed_log.path if self._closed_log is not None else None
        return run_log_name(self.path or closed)

    # -- lifecycle -----------------------------------------------------

    def _run_key_of(self, msg: dict[str, Any]) -> tuple[str, str] | None:
        rs = msg.get("run_state") or {}
        character = rs.get("character")
        seed = msg.get("seed")
        if not character or not seed:
            return None
        return (str(character), str(seed))

    def _ensure_open(self, msg: dict[str, Any]) -> GameLogger | None:
        if not self._dir:
            return None
        key = self._run_key_of(msg)
        if key is None:
            return self._log  # keep logging into the current run, if any
        if self._log is not None and key == self._run_key:
            return self._log
        self.close()
        rs = msg.get("run_state") or {}
        try:
            self._log = GameLogger(
                character=character_display_name(key[0]) or key[0],
                seed=key[1],
                log_dir=self._dir,
                ascension=int(rs.get("ascension") or 0),
            )
            self._log.log_meta_extra({"source": "realgame_advisor", "live": True})
            self._run_key = key
            self._last_sig = None
            self._state_count = 0
            logger.info("workbench run log: %s", self._log.path)
        except Exception as exc:
            logger.warning("workbench log open failed: %s", exc)
            self._log = None
        return self._log

    def close(self) -> None:
        if self._log is not None:
            self._closed_log = self._log
            try:
                self._log.close()
            except Exception:
                pass
        self._log = None
        self._run_key = None
        self._last_sig = None
        self._last_state = None
        self._state_count = 0

    # -- events --------------------------------------------------------

    @_locked
    def log_decision(self, msg: dict[str, Any], advice_for: str) -> int | None:
        """Append the decision's ``state`` record.

        Returns the new step's replay index -- the index the workbench's replay
        view gives this ``state`` line -- or ``None`` when nothing was written
        (duplicate board, unknown surface, or logging disabled).
        """
        try:
            log = self._ensure_open(msg)
            if log is None:
                return None
            sig = self._signature(msg)
            if sig == self._last_sig:
                self._pending_rows.extend(msg.get("combat_log") or [])
                return None
            if advice_for == "combat_action":
                state = to_headless_combat_play({**msg, "type": "combat_action"})
                state["floor"] = msg.get("floor")
                state["act"] = msg.get("act")
                state["act_name"] = msg.get("act_name") or None
                _decorate_combat_for_render(state, msg)
            else:
                builder = _BUILDERS.get(advice_for)
                if builder is None:
                    self._pending_rows.extend(msg.get("combat_log") or [])
                    return None
                state = builder(msg)
            # The game's own combat-log rows for this decision, as the mod sent
            # them (`SimCombatLog.Drain()`). Stored RAW — game ids, no display
            # names — because this process has no localization; the workbench
            # server renders them when it serves the step, through the same
            # formatter a simulated run uses. Also stamp the character id: the
            # rows that name no creature (draws, generated cards, potions) need
            # it to say "The Ironclad" rather than "Player".
            rows = self._pending_rows + list(msg.get("combat_log") or [])
            self._pending_rows = []
            if rows:
                state["combat_log_raw"] = list(rows)
            character = (msg.get("run_state") or {}).get("character")
            if character:
                state["character_id"] = character
            log.log_state(state)
            self._last_sig = sig
            self._last_state = state
            self._state_count += 1
            return self._state_count - 1
        except Exception as exc:
            logger.warning("workbench log_decision failed: %s", exc)
            return None

    @_locked
    def log_advice(
        self,
        msg: dict[str, Any],
        advice: dict[str, Any],
        elapsed_s: float,
        tree: dict[str, Any] | None = None,
    ) -> None:
        """Attach the advice to the current step as its ``mcts`` record."""
        try:
            if self._log is None:
                return
            data: dict[str, Any] = {
                "request_id": msg.get("request_id"),
                "state_id": msg.get("state_id"),
                "elapsed_s": round(elapsed_s, 2),
                "advice": advice,
            }
            outcomes = _root_outcomes(advice)
            if outcomes:
                data["root_outcomes"] = outcomes
            if advice.get("pv"):
                data["pv"] = advice["pv"]
            if tree:
                data["tree"] = tree
            self._log.log_mcts(data)
        except Exception as exc:
            logger.warning("workbench log_advice failed: %s", exc)

    @_locked
    def log_human_action(self, msg: dict[str, Any]) -> None:
        """Append the human's move as the current step's ``action`` record.

        ``args`` is the move's machine form (:func:`human_action_args`), so the
        log can be graded and replayed, not just read.
        """
        try:
            if self._log is None:
                return
            self._log.log_action(
                {
                    "cmd": "action",
                    "action": msg.get("action") or "human",
                    "label": msg.get("detail") or msg.get("action") or "?",
                    "args": human_action_args(msg, self._last_state),
                    "decision_type": msg.get("decision_type"),
                    "advice_request_id": msg.get("advice_request_id"),
                    "position_id": msg.get("position_id"),
                    "peeked": bool(msg.get("peeked")),
                }
            )
        except Exception as exc:
            logger.warning("workbench log_human_action failed: %s", exc)

    @_locked
    def log_coach(self, record: dict[str, Any]) -> None:
        """Append one coach record (a graded turn / choice, or a withdrawal).

        Rides as its own ``type: "coach"`` line, so a reader that predates it
        ignores it; the follow-mode workbench renders it.

        The death autopsy (and a withdrawal racing it) is built after the run
        end closed the file: those go into the file closed last, re-opened
        for the one line.
        """
        try:
            if self._log is None:
                closed = self._closed_log
                if (
                    closed is not None
                    and record.get("kind") in ("autopsy", "withdraw")
                    and closed.reopen()
                ):
                    try:
                        closed.log_coach(record)
                    finally:
                        closed.close()
                return
            self._log.log_coach(record)
        except Exception as exc:
            logger.warning("workbench log_coach failed: %s", exc)

    @_locked
    def log_run_end(self, msg: dict[str, Any]) -> None:
        """Terminal ``game_over``/``run_complete``: final state + summary."""
        try:
            if self._log is None:
                return
            self._log.log_state(_terminal(msg))
            rs = msg.get("run_state") or {}
            self._log.log_summary(
                {
                    "final_won": bool(msg.get("victory")),
                    "final_turn": None,
                    "final_player_hp": rs.get("hp"),
                    "final_player_max_hp": rs.get("max_hp"),
                    "final_floor": msg.get("floor"),
                }
            )
        except Exception as exc:
            logger.warning("workbench log_run_end failed: %s", exc)
        finally:
            self.close()

    # -- helpers -------------------------------------------------------

    @staticmethod
    def _signature(msg: dict[str, Any]) -> str:
        core = {k: v for k, v in msg.items() if k not in _VOLATILE_KEYS}
        try:
            blob = json.dumps(core, sort_keys=True, default=str)
        except Exception:
            blob = repr(core)
        return hashlib.sha1(blob.encode("utf-8", "replace")).hexdigest()


#: Wire Card fields the search translator drops but the Board needs to render
#: card text and frames (templates resolve against ``stats``; ``rarity``
#: drives the frame tint; ``target_type`` the targeting arrow).
_RENDER_CARD_KEYS = ("description", "stats", "rarity", "target_type")


def _merge_render_fields(
    translated: list[dict[str, Any]] | None, originals: list[dict[str, Any]] | None
) -> None:
    """Copy display-only fields from the wire cards onto the translated pile.

    ``_translate_pile`` preserves order 1:1, so a positional zip is exact.
    Only fills gaps -- never overwrites what the translator produced (its
    ``cost``/``name`` are the search-correct values).
    """
    for t, o in zip(translated or [], originals or []):
        if not isinstance(t, dict) or not isinstance(o, dict):
            continue
        for key in _RENDER_CARD_KEYS:
            if key not in t and o.get(key) is not None:
                t[key] = o[key]


def _decorate_combat_for_render(state: dict[str, Any], msg: dict[str, Any]) -> None:
    """Make a translated ``combat_play`` state workbench-renderable.

    :func:`realgame_adapter.to_headless_combat_play` feeds the SEARCH -- it
    deliberately drops everything cosmetic. The workbench Board needs the
    player's display name (art atlas + Spine rig resolution), the deck (deck
    viewer), and per-card text fields; graft them from the wire message the
    state was translated from.
    """
    rs = msg.get("run_state") or {}
    player = state.get("player")
    if isinstance(player, dict):
        player.setdefault(
            "name", character_display_name(rs.get("character")) or (rs.get("character") or "?")
        )
        deck = rs.get("master_deck") or []
        player.setdefault("deck", deck)
        player.setdefault("deck_size", len(deck))
        player.setdefault("gold", rs.get("gold"))
    # The monster's canonical kind, which art resolution needs and the display
    # name cannot give: all three Decimillipede segments localize to the one
    # name "Decimillipede", so `AssetResolver.monster_rig` / `monster_rig_group`
    # can only reach `decimillipede_segment_front|middle|back` through the kind
    # (`_MONSTER_KIND_RIGS`). Without it the elite drew as three sprite
    # fallbacks and `cameraFor` missed the encounter's camera as well.
    #
    # Grafted HERE rather than in `_translate_enemy` on purpose: that dict is
    # the SEARCH's input (`sts2_rs.state_from_dict` reconstructs the fight from
    # it), and a new key there changes what the live MCTS is handed. This one is
    # cosmetic, so it belongs on the render side of the seam. The bridge's
    # `id` is `Creature.Monster.Id.Entry` — exactly the MonsterKind the engine
    # wire calls `kind` (`state_dict.rs`: `species.python_name()`).
    #
    # Index-aligned: `_translate_enemy` maps the wire's enemy list 1:1, corpses
    # included, so enemy i here IS wire enemy i.
    wire_enemies = msg.get("enemies") or []
    for i, enemy in enumerate(state.get("enemies") or []):
        if not isinstance(enemy, dict) or i >= len(wire_enemies):
            continue
        kind = (wire_enemies[i] or {}).get("id")
        if isinstance(kind, str) and kind:
            enemy["kind"] = kind
    _merge_render_fields(state.get("hand"), msg.get("hand"))
    for pile_key, wire_key in (
        ("draw_pile", "draw_pile"),
        ("discard_pile", "discard_pile"),
        ("exhaust_pile", "exhaust_pile"),
        ("play_pile", "play_pile"),
    ):
        wire_pile = msg.get(wire_key) or {}
        _merge_render_fields(state.get(pile_key), wire_pile.get("cards"))
    _merge_render_fields(state.get("master_deck"), rs.get("master_deck"))


def _root_outcomes(advice: dict[str, Any]) -> list[dict[str, Any]]:
    """The advice's ranked options as the ``root_outcomes`` rows
    ``TimelineStore.load_jsonl`` turns into per-step ``scores`` (only rows
    with real visit counts — screen advice has none)."""
    rows: list[dict[str, Any]] = []
    for o in advice.get("options") or []:
        if not isinstance(o, dict) or not o.get("visits"):
            continue
        rows.append(
            {
                "stable_idx": o.get("stable_idx", o.get("card_index", -1)),
                "action_label": o.get("label"),
                "target_index": o.get("target_index"),
                "q": o.get("q", 0.0),
                "visits": o.get("visits", 0),
                "wins": o.get("wins", 0),
                "losses": o.get("losses", 0),
                "ties": o.get("ties", 0),
                "is_best": bool(o.get("is_best")),
                # The search's own numbers, so a reader grades with what the
                # advisor saw rather than re-deriving wins / visits.
                "win_prob": o.get("p_win"),
                "mean_final_hp": o.get("mean_final_hp"),
                "hp_sd": o.get("hp_sd"),
                "hp_n": o.get("hp_n", 0),
            }
        )
    return rows


def _norm_id(raw: Any) -> str:
    return str(raw or "").strip().upper()


def human_action_args(msg: dict[str, Any], state: dict[str, Any] | None) -> dict[str, Any]:
    """The native action dict a ``human_action`` stands for, or ``{}``.

    Combat moves come out in the engine's frame against ``state`` (the logged
    headless decision state): a card is found in its hand by IDENTITY (id +
    upgraded), the same rule the advisor mirror follows
    (``realgame_advisor_mirror.on_world_action``), and a potion's ``slot`` is
    its position among the HELD potions (the engine keeps the belt dense).
    Anything that cannot be resolved logs ``{}`` -- never a guessed index.
    Non-combat choices log ``{"index": i | None}`` (``None`` = skip), in the
    advice's own option index space.
    """
    if msg.get("decision_type") != "combat_action":
        choice = msg.get("choice")
        if isinstance(choice, dict):
            index = choice.get("index")
            return {"index": int(index) if isinstance(index, int) else None}
        return {}
    action = msg.get("action")
    target = msg.get("target_index")
    target = int(target) if isinstance(target, int) else None
    if action == "end_turn":
        return {"action": "end_turn"}
    state = state or {}
    card = msg.get("card")
    if isinstance(card, dict) and card.get("id"):
        want = (_norm_id(card.get("id")), bool(card.get("upgraded")))
        for i, c in enumerate(state.get("hand") or []):
            if isinstance(c, dict) and (_norm_id(c.get("id")), bool(c.get("upgraded"))) == want:
                return {"action": "play", "card_index": i, "target_index": target}
        return {}
    drink = msg.get("potion")
    if isinstance(drink, dict) and drink.get("id"):
        belt = (state.get("player") or {}).get("potions") or state.get("potions") or []
        held = [p for p in belt if isinstance(p, dict) and not p.get("empty") and p.get("id")]
        for slot, p in enumerate(held):
            if _norm_id(p.get("id")) == _norm_id(drink.get("id")):
                return {"action": "potion", "slot": slot, "target_index": target}
    return {}
