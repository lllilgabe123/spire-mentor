"""Live coach for the real-game advisor: grades the human's decisions after the fact.

The advisor computes advice for every decision whether or not the mod shows
it (Coach mode renders nothing before the player acts). This module keeps
that advice per request (:class:`AdviceCache`) and grades each ``human_action``
against the advice for exactly the position it was taken from
(:class:`CoachGrader`). The grades go to the mod as ``coach`` messages and into
the workbench run log as ``coach`` records (PROTOCOL.md, "coach").

Combat is graded PER TURN, never per card. Each action is diffed with plan 1's
:func:`workbench.regret_review.assess_decision` (HP and win chance together),
after a forced-first-move top-up when the played move's row is too thin to
trust. A turn is graded when it ends, from its actions:

.. code-block:: text

    graded = per-action DecisionRegret, after top-up
    if any action verdict == "uncertain" or ungraded -> turn ungraded, silent
    otherwise regret_review.grade_turn: count only actions with
    severity >= TURN_ACTION_FLOOR, severity = max(summed points, summed HP),
    verdict best | tie | slip | blunder (the browser workbench uses its twin)

Silence beats a wrong number: a turn with any action that stays uncertain, or
that cannot be graded, emits nothing to the mod (the log records it as
``ungraded``). In-combat card selects carry no per-option stats and are
skipped (counted in ``ungraded_actions``), not guessed.

Run-level choices (map, card reward, shop, ...) are graded by policy
AGREEMENT (:func:`choice_grade.grade_choice`), never by a win-chance cost.

Two summaries ride on those grades (coach plan 4, ``capabilities.coach >= 2``):
a ``recap`` after each won fight (the mod's ``combat_ended``: HP lost against
the AI line's expectation, plus the costliest turns with the position each
started from) and an ``autopsy`` on death (the run's costliest decisions,
:func:`rank_autopsy`). Both are built on the executor behind the grading still
in flight, and only from records still on the active path.
"""

from __future__ import annotations

import dataclasses
import logging
import queue
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from choice_grade import grade_choice
from enemy_labels import enemy_label, enemy_labels
from workbench.reasoning import rows_from_mcts
from workbench.regret_review import (
    PLAYED_MIN_VISITS,
    DecisionRegret,
    action_stable_key,
    assess_decision,
    grade_turn,
)

logger = logging.getLogger(__name__)

#: How many advice requests the cache remembers. A combat turn is a handful of
#: requests and the grader only ever looks back one turn, so this is generous.
ADVICE_CACHE_SIZE = 64
#: Forced-root sims for a played-move top-up (same floor the browser coach uses).
TOP_UP_SIMS = max(PLAYED_MIN_VISITS * 4, 400)
#: Budget of each step of the background AI-line search.
AI_LINE_SIMS = 1000
#: Defensive cap on an AI line's length; the stop rules end it far sooner.
AI_LINE_MAX_STEPS = 12
#: Root-parallel workers for a catch-up search (kept off the main search's cores).
CATCH_UP_THREADS = 2
#: How many of a fight's costliest turns its recap shows.
RECAP_COSTLIEST = 2
#: How many decisions the death autopsy shows.
AUTOPSY_ITEMS = 3
#: How often a recap / autopsy job re-queues itself behind a turn still being
#: graded (an AI-line job queued after it) before going out regardless.
SUMMARY_REQUEUE_MAX = 20

#: The chip's name for each run-level decision (the mod's ``AdviceFor`` values).
LABEL_KINDS = {
    "map_select": "Map",
    "card_reward": "Card reward",
    "card_bundle": "Card bundle",
    "shop": "Shop",
    "event": "Event",
    "rest_site": "Rest site",
    "boss_relic": "Boss relic",
    "treasure": "Treasure",
    "card_select": "Card select",
}

_END_TURN = -1
_POTION_BASE = -100
#: Powers whose removal is an enemy waking up (C# ``AsleepPower`` /
#: ``SlumberPower``: unblocked damage strips both and ends the free turns).
_SLEEP_POWERS = frozenset({"ASLEEP", "SLUMBER"})
#: Why a turn the fight ended (no end-turn report) was dropped.
_COMBAT_ENDED_MID_TURN = "combat ended mid-turn"
#: The ``capabilities.coach`` level a mod needs before a record kind is sent
#: (kinds not listed: 1).
_SUMMARY_KINDS = {"recap": 2, "autopsy": 2}


@dataclass
class Entry:
    """One advice request's raw result, as the grader needs it.

    ``kind`` is ``"combat"`` (``rows`` = the search's per-action regret rows,
    ``cs`` = the native position it searched, ``cfg`` = its resolved config)
    or ``"screen"`` (``options`` = the advice's ranked options). ``msg`` is the
    wire ``advice_request`` the advice answered.
    """

    request_id: Any
    state_id: Any
    kind: str
    rows: list[dict[str, Any]] | None = None
    options: list[dict[str, Any]] | None = None
    cs: Any = None
    cfg: Any = None
    advice: dict[str, Any] = field(default_factory=dict)
    pv_labels: list[str] = field(default_factory=list)
    msg: dict[str, Any] = field(default_factory=dict)


class AdviceCache:
    """Bounded LRU of :class:`Entry` keyed by ``request_id``. Thread-safe."""

    def __init__(self, size: int = ADVICE_CACHE_SIZE) -> None:
        self._size = size
        self._lock = threading.Lock()
        self._entries: OrderedDict[Any, Entry] = OrderedDict()

    def put(
        self,
        request_id: Any,
        state_id: Any,
        *,
        kind: str,
        rows: list[dict[str, Any]] | None = None,
        options: list[dict[str, Any]] | None = None,
        cs: Any = None,
        cfg: Any = None,
        advice: dict[str, Any] | None = None,
        pv_labels: list[str] | None = None,
        msg: dict[str, Any] | None = None,
    ) -> Entry | None:
        """Remember one request's result; ``None`` (nothing stored) without an id."""
        if request_id is None:
            return None
        entry = Entry(
            request_id=request_id,
            state_id=state_id,
            kind=kind,
            rows=rows,
            options=options,
            cs=cs,
            cfg=cfg,
            advice=dict(advice or {}),
            pv_labels=list(pv_labels or []),
            msg=dict(msg or {}),
        )
        with self._lock:
            self._entries.pop(request_id, None)
            self._entries[request_id] = entry
            while len(self._entries) > self._size:
                self._entries.popitem(last=False)
        return entry

    def get(self, request_id: Any) -> Entry | None:
        if request_id is None:
            return None
        with self._lock:
            entry = self._entries.get(request_id)
            if entry is not None:
                self._entries.move_to_end(request_id)
            return entry

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


# ---------------------------------------------------------------------------
# Row helpers (twins of sts2-workbench advisors.rs top-up)
# ---------------------------------------------------------------------------


def _match_row_index(rows: list[dict[str, Any]], native: dict[str, Any]) -> int | None:
    """``regret_review._match_row``'s exact-then-``stable_idx``-only rule, as an index."""
    idx, target = action_stable_key(native, None)
    if idx is None:
        return None
    for i, r in enumerate(rows):
        if r.get("stable_idx") == idx and r.get("target_index") == target:
            return i
    for i, r in enumerate(rows):
        if r.get("stable_idx") == idx:
            return i
    return None


def played_needs_top_up(rows: list[dict[str, Any]], native: dict[str, Any]) -> bool:
    """The played move's row exists, is not the recommended one, and has fewer
    than ``PLAYED_MIN_VISITS`` visits (twin of ``advisors.rs::played_needs_top_up``)."""
    i = _match_row_index(rows, native)
    if i is None:
        return False
    row = rows[i]
    return not row.get("is_best") and int(row.get("visits", 0) or 0) < PLAYED_MIN_VISITS


def apply_top_up(rows: list[dict[str, Any]], native: dict[str, Any], fresh: dict[str, Any]) -> None:
    """Overwrite the played row's stats with a forced-root search's
    (twin of ``advisors.rs::top_up_played``); ``stable_idx`` / ``action`` /
    ``target_index`` are kept and the row is never crowned."""
    i = _match_row_index(rows, native)
    if i is None:
        return
    row = rows[i]
    row["win_prob"] = float(fresh.get("p_win", 0.0) or 0.0)
    row["q"] = float(fresh.get("q", 0.0) or 0.0)
    for k in ("visits", "wins", "losses", "ties", "hp_n"):
        row[k] = int(fresh.get(k, 0) or 0)
    hp = fresh.get("mean_final_hp")
    row["mean_final_hp"] = float(hp) if hp is not None else None
    sd = fresh.get("hp_sd")
    row["hp_sd"] = float(sd) if sd is not None else None
    row["is_best"] = False


def native_from_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """A regret row's move as the engine's action dict."""
    idx = row.get("stable_idx")
    if not isinstance(idx, int):
        return None
    target = row.get("target_index")
    if idx >= 0:
        return {"action": "play", "card_index": idx, "target_index": target}
    if idx == _END_TURN:
        return {"action": "end_turn"}
    if _POTION_BASE - 5 < idx <= _POTION_BASE:
        return {"action": "potion", "slot": _POTION_BASE - idx, "target_index": target}
    return None


def _norm(raw: Any) -> str:
    return str(raw or "").strip().upper()


def _in_combat_screen(msg: dict[str, Any]) -> bool:
    """A screen opened inside a fight: a card select with its embedded combat,
    or a generated-card pick (Discovery and friends, ``choose_a_card``)."""
    return msg.get("combat") is not None or msg.get("surface") == "choose_a_card"


# ---------------------------------------------------------------------------
# Recap / autopsy helpers (pure)
# ---------------------------------------------------------------------------


def _int_or_none(raw: Any) -> int | None:
    return raw if isinstance(raw, int) and not isinstance(raw, bool) else None


def intent_text(intents: list[dict[str, Any]] | None) -> str:
    """One enemy's telegraphed move as the recap prints it.

    Attacks read ``"attack {damage}"`` / ``"attack {damage}×{hits}"``; any
    other intent is its lower-cased ``intent_type``; a multi-intent move joins
    them with ``" + "`` (``"attack 6×2 + debuff"``). Unknown -> ``"unknown"``.
    """
    parts: list[str] = []
    for intent in intents or []:
        if not isinstance(intent, dict):
            continue
        damage = _int_or_none(intent.get("damage"))
        if damage is not None:
            hits = _int_or_none(intent.get("hits")) or 1
            text = f"attack {damage}" + (f"×{hits}" if hits > 1 else "")
        else:
            kind = str(intent.get("intent_type") or "").strip()
            if not kind:
                continue
            text = kind.lower()
        if text not in parts:
            parts.append(text)
    return " + ".join(parts) or "unknown"


def turn_context(msg: dict[str, Any] | None) -> dict[str, Any] | None:
    """The position a combat turn started from, for the recap and autopsy.

    ``msg`` is the turn's first combat ``advice_request`` (the mod's wire
    dialect). ``None`` when there is no such message: a turn without its
    position is never shown (the spec's "never 'Turn 2' without showing it").
    """
    if not msg:
        return None
    player = msg.get("player") or {}
    hand = [
        str(c.get("name") or c.get("id") or "?")
        for c in msg.get("hand") or []
        if isinstance(c, dict)
    ]
    roster = [e for e in msg.get("enemies") or [] if isinstance(e, dict)]
    enemies = [
        {
            # Numbered by ``enemy_labels`` over the wire list, as the option
            # labels are (a dead entry takes no number), so "Two-Tailed Rat #2
            # intends" names the rat "Strike → … #2" hits.
            "name": label,
            "hp": _int_or_none(e.get("hp")),
            "intent": intent_text(e.get("intents")),
        }
        for e, label in zip(roster, enemy_labels(roster), strict=True)
        if e.get("is_alive", True)
    ]
    return {
        "hand": hand,
        "energy": _int_or_none(player.get("energy")),
        "max_energy": _int_or_none(player.get("max_energy")),
        "hp": _int_or_none(player.get("hp")),
        "max_hp": _int_or_none(player.get("max_hp")),
        "block": _int_or_none(player.get("block")),
        "enemies": enemies,
    }


def _asleep(enemy: dict[str, Any]) -> bool:
    return any(
        isinstance(p, dict) and p.get("name") in _SLEEP_POWERS for p in enemy.get("powers") or []
    )


def board_events(start: dict[str, Any], end: dict[str, Any]) -> list[tuple[str, int]]:
    """What a line visibly did to the enemies between two engine boards.

    ``start`` / ``end`` are ``sts2_rs.state_to_dict`` boards of one turn's
    first position and of the position its line ended the turn from.
    Returns ``("wakes", i)`` / ``("kills", i)`` per encounter slot ``i`` (the
    slot the mod's ``enemies`` list and a ``target_index`` share), or a lone
    ``("wins", -1)`` when the line won the fight -- its kills say nothing more.
    """
    if end.get("phase") == "Victory":
        return [("wins", -1)]
    after = {e.get("uid"): e for e in end.get("encounter") or [] if isinstance(e, dict)}
    out: list[tuple[str, int]] = []
    for i, e in enumerate(start.get("encounter") or []):
        if not isinstance(e, dict) or e.get("is_dead"):
            continue
        f = after.get(e.get("uid"))
        if f is None:
            continue
        if f.get("is_dead"):
            out.append(("kills", i))
        elif _asleep(e) and not _asleep(f):
            out.append(("wakes", i))
    return out


def why_text(
    you: list[tuple[str, int]],
    ai: list[tuple[str, int]],
    names: list[str],
    turn: Any,
) -> str | None:
    """The recap's reason line: what one line did on the board that the other
    did not, e.g. ``"Your line wakes Lagavulin Matriarch on turn 1"`` (GH #888).

    A worse line can look like the AI's plus a free card; without the reason
    a correct call reads as noise. ``None`` when both lines did the same.
    """

    def phrase(event: tuple[str, int]) -> str:
        kind, i = event
        if kind == "wins":
            return "wins the fight"
        name = names[i] if 0 <= i < len(names) else f"enemy {i + 1}"
        return f"{kind} {name}"

    parts = []
    for subject, mine, theirs in (("your line", you, ai), ("the AI's line", ai, you)):
        only = [phrase(e) for e in mine if e not in theirs]
        if only:
            parts.append(f"{subject} {' and '.join(only)}")
    if not parts:
        return None
    text = "; ".join(parts) + (f" on turn {turn}" if turn is not None else "")
    return text[0].upper() + text[1:]


def _player_hp(msg: dict[str, Any]) -> int | None:
    hp = _int_or_none((msg.get("player") or {}).get("hp"))
    return hp if hp is not None else _int_or_none((msg.get("run_state") or {}).get("hp"))


def _costly(record: dict[str, Any]) -> bool:
    """A graded turn worth naming: a slip or blunder with its position."""
    return record.get("verdict") in ("slip", "blunder") and bool(record.get("context"))


def rank_autopsy(
    turns: list[dict[str, Any]], choices: list[dict[str, Any]], k: int = AUTOPSY_ITEMS
) -> list[dict[str, Any]]:
    """The run's costliest decisions, for the death autopsy.

    Combat turns and run-level choices are on no common scale (spec,
    "Autopsy ranking"): turns rank by expected HP lost (win-chance points
    when HP is unknown), choices by how far the AI's top pick out-weighed
    the player's. The two lists are interleaved, turns first, and one fills
    in when the other runs out.
    """
    ranked_turns = sorted(
        (t for t in turns if _costly(t)),
        key=lambda t: (
            -float(t["hp_drop"] if t.get("hp_drop") is not None else t.get("severity") or 0.0)
        ),
    )

    def _gap(c: dict[str, Any]) -> tuple[float, float]:
        top = float((c.get("top") or {}).get("prob") or 0.0)
        played = float((c.get("played") or {}).get("prob") or 0.0)
        return (-(top - played), -top)

    ranked_choices = sorted((c for c in choices if c.get("verdict") == "flag"), key=_gap)
    return _interleave(ranked_turns, ranked_choices, k)


def _interleave(
    turns: list[dict[str, Any]], choices: list[dict[str, Any]], k: int
) -> list[dict[str, Any]]:
    """``t1, c1, t2, c2, ...`` up to ``k``; the longer list fills the rest."""
    out: list[dict[str, Any]] = []
    ti = ci = 0
    while len(out) < k and (ti < len(turns) or ci < len(choices)):
        if ti < len(turns) and (len(out) % 2 == 0 or ci >= len(choices)):
            out.append({"kind": "turn", "record": turns[ti]})
            ti += 1
        else:
            out.append({"kind": "choice", "record": choices[ci]})
            ci += 1
    return out


def build_recap(
    fight: _Fight,
    end_msg: dict[str, Any],
    turn_records: list[dict[str, Any]],
    position: dict[str, Any],
) -> dict[str, Any] | None:
    """The ``recap`` record of a won fight (PROTOCOL.md, ``coach``).

    ``turn_records`` are the fight's graded turn records still on the active
    path; ``position`` is the fight's ``{"log_name", "log_idx"}``. ``None``
    when either HP end is unknown -- a recap without its headline number is
    not worth showing.
    """
    hp_start = fight.hp_start
    hp_end = _int_or_none(end_msg.get("hp"))
    if hp_start is None or hp_end is None:
        return None
    costliest = sorted(
        (r for r in turn_records if _costly(r)), key=lambda r: -float(r.get("severity") or 0.0)
    )[:RECAP_COSTLIEST]
    node = fight.node_id
    return {
        "kind": "recap",
        "id": f"recap:{node}",
        "node_id": node,
        **position,
        "floor": end_msg.get("floor", fight.first_msg.get("floor")),
        "room_type": end_msg.get("room_type") or fight.first_msg.get("room_type"),
        "hp_start": hp_start,
        "hp_end": hp_end,
        "hp_lost": max(0, hp_start - hp_end),
        # Reserved, always null for now. The obvious source -- the opening
        # search's best-row mean final HP -- averages playouts finished by the
        # quick default rollout policy, not the AI's own play, so it overstated
        # the AI's loss (live: "AI typically -12" where the player lost 6).
        # Silence beats a wrong number; the mod renders null as "-N HP".
        "ai_hp_lost": None,
        "turns": fight.turns,
        "turns_graded": len(turn_records),
        "turns_ungraded": fight.ungraded,
        "unfinished_turns": fight.unfinished,
        "costliest": costliest,
    }


def native_action(msg: dict[str, Any], info: dict[str, Any]) -> dict[str, Any] | None:
    """The engine action a combat ``human_action`` stands for at a position.

    ``info`` is :meth:`CoachEngine.position_info` of that position. A card is
    located in its hand by IDENTITY and a potion in its dense belt, the same
    rules the advisor mirror follows; ``None`` when neither resolves.
    """
    action = msg.get("action")
    target = msg.get("target_index")
    target = int(target) if isinstance(target, int) else None
    if action == "end_turn":
        return {"action": "end_turn"}
    card = msg.get("card")
    if isinstance(card, dict) and card.get("id"):
        want = (_norm(card.get("id")), bool(card.get("upgraded")))
        for i, held in enumerate(info.get("hand") or []):
            if (_norm(held[0]), bool(held[1])) == want:
                return {"action": "play", "card_index": i, "target_index": target}
        return None
    drink = msg.get("potion")
    if isinstance(drink, dict) and drink.get("id"):
        for slot, pid in enumerate(info.get("potions") or []):
            if pid and _norm(pid) == _norm(drink.get("id")):
                return {"action": "potion", "slot": slot, "target_index": target}
    return None


# ---------------------------------------------------------------------------
# Engine operations (stubbed in tests)
# ---------------------------------------------------------------------------


def _cfg_for(cfg: Any, *, sims: int, threads: int) -> Any:
    """``cfg`` re-budgeted, with the mode-resolved constants re-derived for it."""
    from sts2_hint.recommender import RecommendConfig

    base = cfg if cfg is not None else RecommendConfig()
    return dataclasses.replace(base, sims=sims, threads=threads, c_puct=None)


class CoachEngine:
    """The searches the coach runs, on native ``sts2_rs`` combat states."""

    def position_info(self, cs: Any, msg: dict[str, Any] | None = None) -> dict[str, Any]:
        """``{"hand": [(id, upgraded)], "potions": [id | None], "round": int | None}``.

        From the native state when there is one (its hand order is the frame
        the rows index); from the wire board otherwise.
        """
        if cs is not None:
            import sts2_rs

            d = sts2_rs.state_to_dict(cs)
            hand = [
                (str(c.get("id") or ""), bool(c.get("upgraded")))
                for c in (d.get("deck") or {}).get("hand") or []
            ]
            potions = [
                (str(p) if p else None) for p in ((d.get("player") or {}).get("potions") or [])
            ]
            rnd = d.get("round")
            return {
                "hand": hand,
                "potions": potions,
                "round": rnd if isinstance(rnd, int) else None,
            }
        msg = msg or {}
        hand = [
            (str(c.get("id") or ""), bool(c.get("upgraded") or c.get("upgrade_level")))
            for c in msg.get("hand") or []
            if isinstance(c, dict)
        ]
        potions = [
            p.get("id")
            for p in msg.get("potions") or []
            if isinstance(p, dict) and not p.get("empty") and p.get("id")
        ]
        rnd = msg.get("round")
        return {"hand": hand, "potions": potions, "round": rnd if isinstance(rnd, int) else None}

    def top_up(self, cs: Any, native: dict[str, Any], cfg: Any, sims: int) -> dict[str, Any] | None:
        """A forced-root search pricing ``native`` from ``cs``."""
        import sts2_rs

        c = _cfg_for(cfg, sims=sims, threads=1)
        return sts2_rs.mcts_score_action_forced(
            cs,
            native,
            sims=sims,
            c_puct=c.resolved_c_puct(),
            depth=c.depth,
            seed=c.seed,
            rollout_policy=c.rollout_policy,
            hidden_info=c.hidden_info,
            draw_widening_c=c.resolved_draw_widening_c(),
            endturn_widening_c=c.endturn_widening_c,
            endturn_widening_alpha=c.endturn_widening_alpha,
            win_hp_carry=c.win_hp_carry,
            win_potion_carry=c.win_potion_carry,
        )

    def _recommend(self, cs: Any, cfg: Any, sims: int, threads: int) -> dict[str, Any]:
        import sts2_rs

        c = _cfg_for(cfg, sims=sims, threads=threads)
        return sts2_rs.mcts_recommend(
            cs,
            sims=sims,
            c_puct=c.resolved_c_puct(),
            depth=c.depth,
            seed=c.seed,
            threads=threads,
            rollout_policy=c.rollout_policy,
            hidden_info=c.hidden_info,
            draw_widening_c=c.resolved_draw_widening_c(),
            endturn_widening_c=c.endturn_widening_c,
            endturn_widening_alpha=c.endturn_widening_alpha,
            win_hp_carry=c.win_hp_carry,
            win_potion_carry=c.win_potion_carry,
        )

    def catch_up(self, cs: Any, cfg: Any, sims: int) -> list[dict[str, Any]]:
        """Search a position the advisor never searched (the player was faster)."""
        return rows_from_mcts(self._recommend(cs, cfg, sims, CATCH_UP_THREADS), {})

    def board(self, cs: Any) -> dict[str, Any] | None:
        """The engine board of ``cs`` (``None`` without a position)."""
        import sts2_rs

        return sts2_rs.state_to_dict(cs) if cs is not None else None

    def ai_line(
        self,
        cs: Any,
        first: dict[str, Any] | None,
        cfg: Any,
        label: Callable[[dict[str, Any], dict[str, Any]], str],
    ) -> list[str]:
        """The labels of :meth:`ai_turn`."""
        return self.ai_turn(cs, first, cfg, label)[0]

    def ai_turn(
        self,
        cs: Any,
        first: dict[str, Any] | None,
        cfg: Any,
        label: Callable[[dict[str, Any], dict[str, Any]], str],
    ) -> tuple[list[str], dict[str, Any] | None]:
        """The AI's line for the turn from ``cs``: ``first``, then the best move
        of a short single-tree search, repeated on a clone.

        Also returns the board the line ended the turn from (the position its
        end turn was played in, or the won / lost board), for
        :func:`board_events`; ``None`` when the line stopped short of that.

        Stops after end turn, when combat ends, or after a move that put a new
        card into the hand -- past a draw the line would name cards the search
        dealt itself (``advisors.rs::label_line``'s rule). Each step is labelled
        from the position it is played in.
        """
        import sts2_rs

        state = cs.clone()
        out: list[str] = []
        end: dict[str, Any] | None = None
        for step in range(AI_LINE_MAX_STEPS):
            if step == 0 and first is not None:
                action = first
            else:
                result = self._recommend(state, cfg, AI_LINE_SIMS, 1)
                best = next((a for a in result.get("actions") or [] if a.get("is_best")), None)
                if best is None:
                    break
                action = best["action"]
            before = sts2_rs.state_to_dict(state)
            out.append(label(before, action))
            hand_before = {
                c.get("stable_idx") for c in (before.get("deck") or {}).get("hand") or []
            }
            outcome = state.apply_step(action)
            if action.get("action") == "end_turn":
                end = before
                break
            after = sts2_rs.state_to_dict(state)
            if outcome in ("combat_won", "combat_lost"):
                end = after
                break
            if any(
                c.get("stable_idx") not in hand_before
                for c in (after.get("deck") or {}).get("hand") or []
            ):
                break
        return out, end


class InlineExecutor:
    """Runs a submitted job immediately (tests, and a coach with no grader thread)."""

    def submit(self, job: Callable[[], None]) -> None:
        job()

    def cancel(self) -> None:
        pass

    def stop(self) -> None:
        pass


class GraderThread:
    """The coach's one background worker: a daemon thread over a job queue.

    Catch-up searches, played-row top-ups and AI lines run here, during the
    enemy turn, while the advisor loop goes on serving messages -- the loop
    never joins or waits on it. :meth:`cancel` drops every job queued so far
    (an undo made them moot); a job already running finishes and its result
    is discarded by the grader, which checks the turn it belongs to.
    """

    def __init__(self) -> None:
        self._queue: queue.Queue[tuple[int, Callable[[], None] | None]] = queue.Queue()
        self._lock = threading.Lock()
        self._generation = 0
        self._thread = threading.Thread(target=self._run, name="coach-grader", daemon=True)
        self._thread.start()

    def submit(self, job: Callable[[], None]) -> None:
        with self._lock:
            generation = self._generation
        self._queue.put((generation, job))

    def cancel(self) -> None:
        with self._lock:
            self._generation += 1

    def stop(self) -> None:
        self._queue.put((-1, None))

    def wait_idle(self, timeout: float) -> bool:
        """Block until every queued job has run (tests); ``False`` on timeout."""
        deadline = time.monotonic() + timeout
        with self._queue.all_tasks_done:
            while self._queue.unfinished_tasks:
                left = deadline - time.monotonic()
                if left <= 0:
                    return False
                self._queue.all_tasks_done.wait(left)
        return True

    def _run(self) -> None:
        while True:
            generation, job = self._queue.get()
            try:
                if job is None:
                    return
                with self._lock:
                    stale = generation != self._generation
                if not stale:
                    job()
            except Exception as exc:
                logger.warning("coach grader job failed: %s", exc, exc_info=True)
            finally:
                self._queue.task_done()


# ---------------------------------------------------------------------------
# The grader
# ---------------------------------------------------------------------------


@dataclass
class _Act:
    """One combat action of the open turn."""

    msg: dict[str, Any]
    label: str
    peeked: bool
    native: dict[str, Any] | None
    rows: list[dict[str, Any]] | None
    position: Any
    cfg: Any
    pending: bool = False
    regret: DecisionRegret | None = None
    forced: bool = False  # fewer than two legal moves: nothing to compare
    failed: str | None = None


@dataclass
class _Fight:
    """One combat, from its first combat advice to ``combat_ended``."""

    first_msg: dict[str, Any]
    #: Node of the fight's first human action (the recap's id / link target).
    node_id: Any = None
    hp_start: int | None = None
    turn_ids: list[str] = field(default_factory=list)
    turns: int = 0  # closed turns, graded or not
    ungraded: int = 0
    unfinished: int = 0


@dataclass
class _Turn:
    node_id: Any
    turn: Any
    floor: Any
    acts: list[_Act] = field(default_factory=list)
    ungraded_actions: int = 0
    closed: bool = False
    done: bool = False
    deadline: float | None = None
    ai: list[str] | None = None
    #: What the player's line did on the board that the AI's did not (or the
    #: reverse), e.g. "Your line wakes Lagavulin Matriarch on turn 1".
    why: str | None = None
    ai_pending: bool = False
    fight: _Fight | None = None
    #: The first combat advice of the turn's round, when there was one.
    start_msg: dict[str, Any] | None = None


class CoachGrader:
    """Per-connection live coach.

    ``send(record)`` puts one record on the wire (the advisor wraps it in the
    ``coach`` envelope); it is only called while the mod advertised
    ``capabilities.coach >= 1``. ``log(record)`` writes the run log and is
    always called, including for turns that stay ungraded. Both may be called
    from the grader's executor thread.
    """

    def __init__(
        self,
        *,
        send: Callable[[dict[str, Any]], None],
        log: Callable[[dict[str, Any]], None],
        engine: Any = None,
        executor: Any = None,
        cache: AdviceCache | None = None,
        catchup_sims: int = 2000,
        budget_s: float = 6.0,
        map_path: Callable[[dict[str, Any], int], list[dict[str, Any]] | None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        locate: Callable[[Any], tuple[str | None, int | None]] | None = None,
    ) -> None:
        self._send = send
        self._log = log
        self.engine = engine if engine is not None else CoachEngine()
        self.executor = executor if executor is not None else InlineExecutor()
        self.cache = cache if cache is not None else AdviceCache()
        self.catchup_sims = catchup_sims
        self.budget_s = budget_s
        self.map_path = map_path
        self.clock = clock
        #: node id -> ``(log_name, log_idx)``: where the record's decision sits
        #: in the run log, for the mod's "open in workbench" links.
        self.locate = locate
        #: The mod's advertised ``capabilities.coach`` (0 = never send).
        self.mod_coach = 0
        self._lock = threading.RLock()
        self._turn: _Turn | None = None
        self._last_cfg: Any = None
        self._last_combat_msg: dict[str, Any] = {}
        self._floor: Any = None
        self._names: dict[str, str] = {}
        self._anon = 0
        #: Ended turns still being graded in the background.
        self._grading: list[_Turn] = []
        #: Every graded record emitted so far: id -> node_id (for withdrawals).
        self._emitted: dict[str, Any] = {}
        #: The same records, id -> record: what the recap and the autopsy are
        #: built from (a withdrawal removes them from both maps).
        self._emitted_records: dict[str, dict[str, Any]] = {}
        #: The combat in progress (or ended, awaiting ``combat_ended``).
        self._fight: _Fight | None = None
        #: True from a fight's first combat advice to its end (a run-level
        #: screen or ``combat_ended``): the next combat advice opens a new fight.
        self._in_fight = False
        #: The first combat advice of each round of the current fight.
        self._round_start: dict[Any, dict[str, Any]] = {}
        #: Closed turns that went ungraded, over the whole run (the autopsy).
        self._run_ungraded = 0

    # -- advice side -----------------------------------------------------

    def on_advice_request(self, msg: dict[str, Any]) -> None:
        """Every ``advice_request``: track the mod's capability and the floor,
        and preempt grading that the next turn has overtaken.

        A COMBAT request for a later round than an ended turn means the enemy
        turn is over. Whatever that turn's grading has not finished is
        dropped -- the turn stays silent rather than show a late chip -- and
        the advice is served at once (this never waits on the grader).
        """
        caps = msg.get("capabilities")
        coach = caps.get("coach") if isinstance(caps, dict) else None
        self.mod_coach = int(coach) if isinstance(coach, int) else 0
        if msg.get("floor") is not None:
            self._floor = msg.get("floor")
        if (msg.get("advice_for") or "combat_action") != "combat_action":
            return
        rnd = msg.get("round")
        with self._lock:
            for turn in list(self._grading):
                if not isinstance(rnd, int) or not isinstance(turn.turn, int) or rnd > turn.turn:
                    self._drop(turn, "preempted: the next turn's advice arrived first")

    def on_rewind(self, msg: dict[str, Any], active_path: Iterable[Any]) -> list[dict[str, Any]]:
        """An undo / snapshot restore: abandon grading, withdraw abandoned records.

        Queued jobs are cancelled and every turn in progress is dropped. Every
        record already emitted whose ``node_id`` is not on the root -> active
        node path (``active_path``, from ``LiveRunTree``) is withdrawn with one
        ``{"kind": "withdraw", "ids": [...]}`` record, through the same
        send/log path -- the log line is the tombstone.
        """
        self.executor.cancel()
        on_path = set(active_path)
        with self._lock:
            self._abandon_open_turn(f"undone ({msg.get('type')})")
            for turn in list(self._grading):
                self._drop(turn, f"undone ({msg.get('type')})")
            gone = [rid for rid, node in self._emitted.items() if node not in on_path]
            for rid in gone:
                del self._emitted[rid]
                self._emitted_records.pop(rid, None)
                if self._fight is not None and rid in self._fight.turn_ids:
                    self._fight.turn_ids.remove(rid)
        if not gone:
            return []
        record = {"kind": "withdraw", "ids": gone}
        self._deliver(record)
        return [record]

    def on_advice(self, entry: Entry | None) -> None:
        """Advice was served and cached; remember what labelling needs."""
        if entry is None:
            return
        if entry.kind == "combat":
            with self._lock:
                self._last_cfg = entry.cfg if entry.cfg is not None else self._last_cfg
                self._last_combat_msg = entry.msg
                self._learn_names(entry.msg)
                if not self._in_fight:
                    self._open_fight(entry)
                rnd = entry.msg.get("round")
                if rnd is not None:
                    self._round_start.setdefault(rnd, entry.msg)
        elif not _in_combat_screen(entry.msg):
            # A run-level screen: any turn still open ended with the fight
            # (a killing blow ends a turn without an end-turn report).
            with self._lock:
                self._abandon_open_turn(_COMBAT_ENDED_MID_TURN)
                self._in_fight = False

    def _open_fight(self, entry: Entry) -> None:
        """The first combat advice after a run-level screen: a new fight
        (caller holds the lock)."""
        self._fight = _Fight(first_msg=entry.msg, hp_start=_player_hp(entry.msg))
        self._in_fight = True
        self._round_start = {}

    # -- fight / run summaries --------------------------------------------

    def on_combat_ended(self, msg: dict[str, Any]) -> None:
        """The mod's ``combat_ended``: queue the fight's recap.

        The turn still open is the one the killing blow ended (no end-turn
        report): it counts as unfinished, not ungraded. A lost fight gets no
        recap -- the death autopsy covers it. The recap job runs on the
        executor BEHIND the fight's turns still being graded (FIFO), so it
        sees their grades; the advice loop never waits for it.
        """
        with self._lock:
            self._abandon_open_turn(_COMBAT_ENDED_MID_TURN)
            self._in_fight = False
            fight, self._fight = self._fight, None
        if fight is None or not msg.get("victory"):
            return
        self.executor.submit(lambda: self._recap_job(fight, msg, 0))

    def _recap_job(self, fight: _Fight, msg: dict[str, Any], tries: int) -> None:
        try:
            with self._lock:
                if self._still_grading(fight) and tries < SUMMARY_REQUEUE_MAX:
                    requeue = True
                else:
                    requeue = False
                    records = [
                        self._emitted_records[rid]
                        for rid in fight.turn_ids
                        if rid in self._emitted_records
                    ]
                    if fight.node_id is None:
                        self._anon += 1
                        fight.node_id = f"x{self._anon}"
            if requeue:
                # A slip's AI-line job was queued after this one: go behind it.
                self.executor.submit(lambda: self._recap_job(fight, msg, tries + 1))
                return
            recap = build_recap(fight, msg, records, self._log_position(fight.node_id))
            if recap is None:
                logger.info("coach: no recap (player HP unknown)")
                return
            self._deliver(recap)
        except Exception as exc:  # bookkeeping: never take the grader down
            logger.warning("coach: recap failed: %s", exc, exc_info=True)

    def _still_grading(self, fight: _Fight | None = None) -> bool:
        """Any ended turn (of ``fight``, or any) still being graded within its
        budget (caller holds the lock; overdue turns are dropped here)."""
        return any(
            not self._overdue(t)
            for t in list(self._grading)
            if (fight is None or t.fight is fight) and not t.done
        )

    def _learn_names(self, msg: dict[str, Any]) -> None:
        cards: list[Any] = list(msg.get("hand") or [])
        for pile in ("draw_pile", "discard_pile", "exhaust_pile"):
            cards += (msg.get(pile) or {}).get("cards") or []
        cards += (msg.get("run_state") or {}).get("master_deck") or []
        for c in cards:
            if isinstance(c, dict) and c.get("id") and c.get("name"):
                self._names.setdefault(_norm(c["id"]), str(c["name"]).rstrip("+"))
        for p in msg.get("potions") or []:
            if isinstance(p, dict) and p.get("id") and p.get("name"):
                self._names.setdefault(_norm(p["id"]), str(p["name"]))

    def on_run_end(self) -> None:
        """A run ended with nothing to summarize (a win, or ``run_complete``):
        cancel the grading still queued and forget the run."""
        self.executor.cancel()
        self._reset_run()

    def _reset_run(self) -> None:
        with self._lock:
            self._turn = None
            for turn in list(self._grading):
                turn.done = True
            self._grading.clear()
            self._emitted.clear()
            self._emitted_records.clear()
            self._fight = None
            self._in_fight = False
            self._round_start = {}
            self._run_ungraded = 0
        self.cache.clear()

    def on_game_over(self, msg: dict[str, Any]) -> None:
        """Death: queue the run's autopsy, then forget the run.

        The autopsy job runs on the executor behind the grading still in
        flight -- the dying turn's grade is usually among it -- and the reset
        runs only AFTER the autopsy is built: resetting first would leave it
        nothing to rank.
        """
        with self._lock:
            self._abandon_open_turn(_COMBAT_ENDED_MID_TURN)
            self._in_fight = False
            self._fight = None
        self.executor.submit(lambda: self._autopsy_job(msg, 0))

    def _autopsy_job(self, msg: dict[str, Any], tries: int) -> None:
        try:
            with self._lock:
                requeue = self._still_grading() and tries < SUMMARY_REQUEUE_MAX
                if not requeue:
                    records = list(self._emitted_records.values())
                    ungraded = self._run_ungraded
            if requeue:
                self.executor.submit(lambda: self._autopsy_job(msg, tries + 1))
                return
            turns = [r for r in records if r.get("kind") == "turn"]
            choices = [r for r in records if r.get("kind") == "choice"]
            log_name = self._log_position(None)["log_name"]
            self._deliver(
                {
                    "kind": "autopsy",
                    "id": f"autopsy:{msg.get('seed') or log_name or 'run'}",
                    "log_name": log_name,
                    "floor": msg.get("floor"),
                    "act": msg.get("act"),
                    "items": rank_autopsy(turns, choices),
                    "turns_ungraded": ungraded,
                    "decisions_graded": len(turns) + len(choices),
                }
            )
        except Exception as exc:  # bookkeeping: never take the grader down
            logger.warning("coach: autopsy failed: %s", exc, exc_info=True)
            self._reset_run()
            return
        self._reset_run()

    # -- human side ------------------------------------------------------

    def on_human_action(
        self, msg: dict[str, Any], pre_state: Any = None, node_id: Any = None
    ) -> list[dict[str, Any]]:
        """Grade one ``human_action``; returns the records emitted synchronously.

        ``pre_state`` is the mirror's native state taken BEFORE the action was
        followed -- the position a catch-up search starts from when the advice
        on screen was not for this exact position. Records that finish later
        (catch-up, top-up, AI line) are emitted from the executor.
        """
        try:
            if msg.get("decision_type") == "combat_action":
                return self._on_combat_action(msg, pre_state, node_id)
            return self._on_choice(msg, node_id)
        except Exception as exc:  # grading is bookkeeping: never take the loop down
            logger.warning("coach: grading failed: %s", exc, exc_info=True)
            return []

    def _valid_entry(self, msg: dict[str, Any], kind: str) -> Entry | None:
        """The cached advice for EXACTLY the position the action was taken
        from, or ``None``. A hit whose ``state_id`` is not the action's
        ``position_id`` is a previous position's advice and is never used."""
        entry = self.cache.get(msg.get("advice_request_id"))
        if entry is None or entry.kind != kind:
            return None
        position = msg.get("position_id")
        if position is None or entry.state_id != position:
            return None
        return entry

    def _on_combat_action(
        self, msg: dict[str, Any], pre_state: Any, node_id: Any
    ) -> list[dict[str, Any]]:
        entry = self._valid_entry(msg, "combat")
        is_end = msg.get("action") == "end_turn"
        label = "end turn" if is_end else self._label_human(msg)
        info: dict[str, Any] = {}
        if entry is not None:
            position, rows, cfg = entry.cs, [dict(r) for r in entry.rows or []], entry.cfg
            if not is_end:
                info = self.engine.position_info(entry.cs, entry.msg)
        else:
            position, rows, cfg = pre_state, None, self._last_cfg
            if pre_state is not None:
                info = self.engine.position_info(pre_state)
        act = _Act(
            msg=msg,
            label=label,
            peeked=bool(msg.get("peeked")),
            native=native_action(msg, info),
            rows=rows,
            position=position,
            cfg=cfg,
        )
        if act.native is None:
            act.failed = "move not found at its position"
        elif rows is None:
            if position is None:
                act.failed = "no advice and no position to search"
            else:
                act.pending = True  # catch-up after the turn ends
        elif played_needs_top_up(rows, act.native):
            act.pending = True  # top-up after the turn ends, never inline
        else:
            self._assess(act)

        with self._lock:
            turn = self._turn
            if turn is None or turn.closed:
                rnd = (entry.msg.get("round") if entry is not None else None) or info.get("round")
                floor = (entry.msg.get("floor") if entry is not None else None) or self._floor
                turn = _Turn(
                    node_id=node_id,
                    turn=rnd,
                    floor=floor,
                    fight=self._fight,
                    start_msg=self._round_start.get(rnd)
                    or (entry.msg if entry is not None else None),
                )
                self._turn = turn
                if self._fight is not None and self._fight.node_id is None:
                    self._fight.node_id = node_id
            turn.acts.append(act)
            if not is_end:
                return []
            turn.closed = True
            if turn.fight is not None:
                turn.fight.turns += 1
            turn.deadline = self.clock() + self.budget_s
            self._turn = None
            self._grading.append(turn)
            pending = [a for a in turn.acts if a.pending]
        for a in pending:
            self.executor.submit(self._job(turn, lambda a=a: self._grade_pending(a)))
        with self._lock:
            return self._maybe_finish(turn)

    def _assess(self, act: _Act) -> None:
        rows = act.rows or []
        if not rows:
            act.failed = "the search ranked no moves"
            return
        if len(rows) == 1:
            act.forced = True  # one legal move: nothing to compare it with
            return
        step = {
            "scores": rows,
            "chosen_action": {"args": act.native},
            "state": {"enemies": []},
            "meta": {},
        }
        act.regret = assess_decision(step, scores=rows)
        if act.regret is None:
            act.failed = "played move has no row"

    def _grade_pending(self, act: _Act) -> None:
        """Catch-up search and/or played-row top-up for one action (executor)."""
        rows = act.rows
        if rows is None:
            rows = self.engine.catch_up(act.position, act.cfg, self.catchup_sims)
        if act.native is not None and played_needs_top_up(rows, act.native):
            fresh = self.engine.top_up(act.position, act.native, act.cfg, TOP_UP_SIMS)
            if fresh is not None:
                apply_top_up(rows, act.native, fresh)
        with self._lock:
            act.rows = rows
            self._assess(act)
            act.pending = False

    def _job(self, turn: _Turn, fn: Callable[[], None]) -> Callable[[], None]:
        """Wrap one unit of grading work for ``turn``."""

        def run() -> None:
            with self._lock:
                if turn.done or self._overdue(turn):
                    return
            try:
                fn()
            except Exception as exc:
                logger.warning("coach: grading job failed: %s", exc, exc_info=True)
                with self._lock:
                    self._drop(turn, f"grading failed: {exc}")
                return
            with self._lock:
                if not self._overdue(turn):
                    self._maybe_finish(turn)

        return run

    def _overdue(self, turn: _Turn) -> bool:
        """Drop ``turn`` (and say so) once its catch-up budget is spent: a late
        chip is worse than none (caller holds the lock)."""
        if turn.done or turn.deadline is None or self.clock() <= turn.deadline:
            return False
        self._drop(turn, f"catch-up budget of {self.budget_s:g}s exceeded")
        return True

    # -- turn assembly ---------------------------------------------------

    def _drop(self, turn: _Turn, reason: str) -> None:
        """Give up on ``turn``: silent to the mod, logged as ungraded."""
        if turn in self._grading:
            self._grading.remove(turn)
        if turn.done:
            return
        turn.done = True
        if turn.closed:
            self._run_ungraded += 1
            if turn.fight is not None:
                turn.fight.ungraded += 1
        self._log(self._turn_head(turn) | {"verdict": "ungraded", "reason": reason})

    def _abandon_open_turn(self, reason: str) -> None:
        """Drop the turn still open (caller holds the lock). One that the fight
        ended (the killing blow reports no end turn) counts as ``unfinished``,
        never ``ungraded``: it was never closed, so there was nothing to grade."""
        turn = self._turn
        self._turn = None
        if turn is not None and turn.acts:
            if reason == _COMBAT_ENDED_MID_TURN and turn.fight is not None:
                turn.fight.unfinished += 1
            self._drop(turn, reason)

    def _turn_head(self, turn: _Turn) -> dict[str, Any]:
        node = turn.node_id
        if node is None:
            self._anon += 1
            turn.node_id = node = f"x{self._anon}"
        return {
            "kind": "turn",
            "id": f"turn:{node}",
            "node_id": turn.node_id,
            "turn": turn.turn,
            "floor": turn.floor,
            **self._log_position(turn.node_id),
        }

    def _log_position(self, node_id: Any) -> dict[str, Any]:
        """``{"log_name", "log_idx"}`` of ``node_id`` (both ``None`` when unknown)."""
        name: str | None = None
        idx: int | None = None
        if self.locate is not None:
            try:
                name, idx = self.locate(node_id)
            except Exception as exc:
                logger.debug("coach: cannot locate node %r: %s", node_id, exc)
        return {"log_name": name, "log_idx": idx}

    def _maybe_finish(self, turn: _Turn) -> list[dict[str, Any]]:
        """Emit ``turn``'s record once every action is graded (caller holds the lock)."""
        if turn.done or not turn.closed or any(a.pending for a in turn.acts):
            return []
        failed = next((a.failed for a in turn.acts if a.failed), None)
        if failed:
            self._drop(turn, failed)
            return []
        if any(a.regret is not None and a.regret.verdict == "uncertain" for a in turn.acts):
            self._drop(turn, "a move stayed uncertain after top-up")
            return []
        grade = grade_turn([a.regret for a in turn.acts if not a.forced])
        if grade is None:
            self._drop(turn, "a move could not be graded")
            return []
        verdict = grade.verdict
        if verdict in ("slip", "blunder") and turn.ai is None:
            if not turn.ai_pending:
                turn.ai_pending = True
                self.executor.submit(self._job(turn, lambda: self._build_ai_line(turn)))
            # With an inline executor the line is already built: fall through.
            if turn.ai is None or turn.done:
                return []
        record = self._turn_head(turn) | {
            "verdict": verdict,
            "you": [a.label for a in turn.acts],
            "ai": list(turn.ai or []),
            "why": turn.why,
            "hp_drop": grade.hp_drop,
            "win_before": grade.win_before,
            "win_after": grade.win_after,
            "peeked": any(a.peeked for a in turn.acts),
            "ungraded_actions": turn.ungraded_actions,
            "severity": grade.severity,
            "context": turn_context(
                turn.start_msg
                or (self._round_start.get(turn.turn) if turn.fight is self._fight else None)
            ),
        }
        if turn.fight is not None:
            turn.fight.turn_ids.append(str(record["id"]))
        turn.done = True
        if turn in self._grading:
            self._grading.remove(turn)
        self._deliver(record)
        return [record]

    def _build_ai_line(self, turn: _Turn) -> None:
        """The AI's line from the turn's first position (executor)."""
        first = turn.acts[0] if turn.acts else None
        line: list[str] = []
        why: str | None = None
        if first is not None and first.position is not None and first.rows:
            best = next((r for r in first.rows if r.get("is_best")), None)
            native = native_from_row(best) if best is not None else None
            line, ai_end = self.engine.ai_turn(
                first.position, native, first.cfg, self._label_engine
            )
            why = self._why(turn, ai_end)
        if not line and first is not None and first.rows:
            best = next((r for r in first.rows if r.get("is_best")), None)
            if best is not None:
                line = [str(best.get("action") or "?")]
        with self._lock:
            turn.ai = line
            turn.why = why
            turn.ai_pending = False

    def _why(self, turn: _Turn, ai_end: dict[str, Any] | None) -> str | None:
        """The reason line of ``turn`` from the AI line's end board (executor).

        The player's side is the real board their end turn was played from;
        ``None`` when either end is unknown.
        """
        last = turn.acts[-1]
        if ai_end is None or last.native is None or last.native.get("action") != "end_turn":
            return None
        start = self.engine.board(turn.acts[0].position)
        you_end = self.engine.board(last.position)
        if start is None or you_end is None:
            return None
        roster = [e for e in (turn.start_msg or {}).get("enemies") or [] if isinstance(e, dict)]
        names = enemy_labels(roster)
        enc = start.get("encounter") or []
        if len(names) != len(enc):
            names = [str(e.get("kind") or "?").replace("_", " ").title() for e in enc]
        return why_text(board_events(start, you_end), board_events(start, ai_end), names, turn.turn)

    def _target_suffix(self, target: Any) -> str:
        """`` (Cultist)`` when the fight has more than one enemy, else ``""``.

        Parenthesised, never ``" → "``: a turn line joins its moves with
        arrows, so an arrow before the target read as a second move.
        """
        enemies = self._last_combat_msg.get("enemies") or []
        label = enemy_label(enemies, target) if len(enemies) > 1 else None
        return f" ({label})" if label is not None else ""

    def _label_human(self, msg: dict[str, Any]) -> str:
        """Name one combat ``human_action`` the way the AI line names moves.

        The mod's ``detail`` can be the game's raw history text
        (``IRONCLAD played STRIKE_IRONCLAD targeting TOADPOLE``), so the
        label is rebuilt from the machine fields; ``detail`` is the fallback.
        """
        card = msg.get("card")
        if isinstance(card, dict) and card.get("id"):
            cid = _norm(card["id"])
            name = self._names.get(cid) or cid.replace("_", " ").title()
            if card.get("upgraded"):
                name += "+"
            return name + self._target_suffix(msg.get("target_index"))
        potion = msg.get("potion")
        if isinstance(potion, dict) and potion.get("id"):
            pid = _norm(potion["id"])
            return f"drink {self._names.get(pid) or pid.replace('_', ' ').title()}"
        return str(msg.get("detail") or msg.get("action") or "?")

    def _label_engine(self, state: dict[str, Any], action: dict[str, Any]) -> str:
        """Name one engine action from the engine state it is played in."""
        kind = action.get("action")
        if kind == "end_turn":
            return "end turn"
        if kind == "play":
            hand = (state.get("deck") or {}).get("hand") or []
            i = action.get("card_index")
            card = hand[i] if isinstance(i, int) and 0 <= i < len(hand) else {}
            cid = _norm(card.get("id"))
            name = self._names.get(cid) or cid.replace("_", " ").title() or "?"
            if card.get("upgraded"):
                name += "+"
            return name + self._target_suffix(action.get("target_index"))
        if kind in ("potion", "drink_potion"):
            belt = (state.get("player") or {}).get("potions") or []
            slot = action.get("slot")
            pid = _norm(belt[slot]) if isinstance(slot, int) and 0 <= slot < len(belt) else ""
            return f"drink {self._names.get(pid) or pid.replace('_', ' ').title() or 'potion'}"
        return str(kind or "?").replace("_", " ")

    # -- run-level choices ------------------------------------------------

    def _on_choice(self, msg: dict[str, Any], node_id: Any) -> list[dict[str, Any]]:
        decision = msg.get("decision_type")
        entry = self._valid_entry(msg, "screen")
        in_combat = (entry is not None and _in_combat_screen(entry.msg)) or (
            entry is None and self._turn is not None
        )
        if decision in ("card_select", "card_reward") and in_combat:
            # Searched in-combat selects carry no per-option stats: skipped,
            # never guessed. The turn is still graded on its other moves.
            with self._lock:
                if self._turn is not None:
                    self._turn.ungraded_actions += 1
            return []
        with self._lock:
            self._abandon_open_turn(_COMBAT_ENDED_MID_TURN)
        if entry is None or decision not in LABEL_KINDS:
            return []
        options = [o for o in entry.options or [] if isinstance(o, dict)]
        choice = msg.get("choice") if isinstance(msg.get("choice"), dict) else {}
        index = choice.get("index")
        if index is None:
            played = next((i for i, o in enumerate(options) if o.get("action") == "skip"), None)
        else:
            played = next(
                (
                    i
                    for i, o in enumerate(options)
                    if o.get("index") == index and o.get("action") != "skip"
                ),
                None,
            )
        pairs = [(str(o.get("label") or "?"), float(o.get("visit_share") or 0.0)) for o in options]
        g = grade_choice(pairs, played)
        if g is None or played is None:
            return []
        ranked = sorted(pairs, key=lambda lp: -lp[1])
        node = node_id
        if node is None:
            self._anon += 1
            node = f"x{self._anon}"
        record: dict[str, Any] = {
            "kind": "choice",
            "id": f"choice:{node}",
            "node_id": node,
            "decision": decision,
            "label_kind": LABEL_KINDS[decision],
            "verdict": g.verdict,
            "strength": g.strength,
            "played": {"label": pairs[played][0], "prob": pairs[played][1], "rank": g.rank},
            "top": {"label": pairs[g.top][0], "prob": pairs[g.top][1]},
            "alternatives": [{"label": lbl, "prob": p} for lbl, p in ranked[:3]],
            "peeked": bool(msg.get("peeked")),
            **self._log_position(node),
        }
        if decision == "map_select":
            path = self._map_paths(entry, played, g.top)
            if path is not None:
                record["path"] = path
        self._deliver(record)
        return [record]

    def _map_paths(self, entry: Entry, played: int, top: int) -> dict[str, Any] | None:
        ai = entry.advice.get("path")
        if not ai:
            return None
        out: dict[str, Any] = {"ai": ai}
        if played == top:
            out["you"] = ai
        elif self.map_path is not None:
            try:
                you = self.map_path(entry.msg, played)
            except Exception as exc:
                logger.debug("coach: route projection failed: %s", exc)
                you = None
            if you:
                out["you"] = you
        return out

    # -- delivery ----------------------------------------------------------

    def _deliver(self, record: dict[str, Any]) -> None:
        """Log ``record``; send it only to a mod that advertised the capability.

        ``turn`` / ``choice`` / ``withdraw`` need ``coach >= 1`` (plan 3);
        ``recap`` / ``autopsy`` need ``coach >= 2`` -- a plan-3 mod ignores
        them anyway, but it should not receive them at all.
        """
        kind = record.get("kind")
        if kind in ("turn", "choice", "recap"):
            with self._lock:
                self._emitted[record["id"]] = record.get("node_id")
                self._emitted_records[record["id"]] = record
        try:
            self._log(record)
        except Exception as exc:
            logger.warning("coach: log failed: %s", exc)
        if self.mod_coach >= _SUMMARY_KINDS.get(str(kind), 1):
            try:
                self._send(record)
            except Exception as exc:
                logger.warning("coach: send failed: %s", exc)
