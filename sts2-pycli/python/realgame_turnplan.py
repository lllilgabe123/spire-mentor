"""Personal-build turn plan: the order to play this turn's cards (1, 2, 3...).

The search commits to one move at a time: its principal variation is usually
just the first play, because later plays depend on what gets drawn. With
known draws (the advisor's default fight sight) the draws are known, so the
turn can be played out on a COPY of the fight: apply the move, search again
with a smaller budget, repeat until it would end the turn. The cards in your
hand now get their step number; cards drawn on the way are named in the line.

Engine hand order is kept when a card is played and draws are appended (checked
on the beta engine), so each current card is tracked by position. A play that
removes other cards or opens a selection (Burning Pact, Armaments' choice)
ends the numbering there: past it the copy cannot say what you will pick.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

MAX_STEPS = 8


def _best_action(sts2_rs: Any, cs: Any, sims: int, seed: int, hidden_info: str | None) -> dict[str, Any] | None:
    kwargs: dict[str, Any] = {"sims": sims, "seed": seed}
    if hidden_info:
        kwargs["hidden_info"] = hidden_info
    result = sts2_rs.mcts_recommend(cs, **kwargs)
    for row in result.get("actions") or []:
        if row.get("is_best"):
            return row.get("action")
    return None


def _first_engine_action(best: dict[str, Any], wire_hand: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The advice's ``best`` (stable card handle) as an engine action."""
    if best.get("action") != "play":
        return None
    for i, card in enumerate(wire_hand):
        if card.get("stable_idx") == best.get("card_index"):
            target = best.get("target_index")
            return {"action": "play", "card_index": i, "target_index": None if target in (None, -1) else target}
    return None


def plan_turn(
    sts2_rs: Any,
    cs: Any,
    best: dict[str, Any],
    wire_hand: list[dict[str, Any]],
    names: dict[str, str],
    sims: int,
    seed: int = 1,
    hidden_info: str | None = None,
) -> tuple[list[dict[str, int]], list[str]]:
    """``([{"card_index": stable_idx, "step": n}, ...], line labels)``.

    ``names`` maps a card id to its display name."""
    action = _first_engine_action(best, wire_hand)
    if action is None:
        return [], []
    c = cs.clone()
    track: list[int | None] = [card.get("stable_idx") for card in wire_hand]
    order: list[dict[str, int]] = []
    line: list[str] = []
    tracking = True
    for step in range(1, MAX_STEPS + 1):
        hand = c.get_state().get("hand") or []
        if action.get("action") == "play":
            i = action.get("card_index")
            if not isinstance(i, int) or not 0 <= i < len(hand):
                break
            cid = str(hand[i].get("id") or "?")
            name = names.get(cid) or cid.replace("_", " ").title()
            handle = track[i] if tracking and i < len(track) else None
            if handle is not None:
                order.append({"card_index": handle, "step": step})
                line.append(f"{step}. {name}")
            else:
                line.append(f"{step}. {name} (drawn)")
            before = [str(h.get("id")) for h in hand]
            c.apply(action)
            if tracking and i >= len(track):
                tracking = False  # a card the tracker never saw (made mid-turn)
            if tracking:
                track.pop(i)
                before.pop(i)
                after = [str(h.get("id")) for h in c.get_state().get("hand") or []]
                if after[: len(before)] != before:
                    tracking = False  # something else left the hand: positions are lost
                else:
                    track.extend([None] * (len(after) - len(before)))
        elif action.get("action") == "drink_potion":
            line.append(f"{step}. potion")
            c.apply(action)
        else:
            break
        if c.pending_choice() is not None:
            break
        state = c.get_state()
        if not any((e.get("hp") or 0) > 0 for e in state.get("enemies") or []):
            line.append("fight won")
            break
        action = _best_action(sts2_rs, c, sims, seed + step, hidden_info)
        if action is None or action.get("action") == "end_turn":
            line.append("end turn")
            break
    return order, line


#: Free cards that pay off on each card played after them this turn, and
#: whether only Attacks benefit (Rage: Block per Attack; Monologue:
#: Strength per card, used by Attacks; Oblivion: Doom per card). They go
#: before the plan's first card they improve; the search sometimes ranks
#: them later by more than its noise, or not at all once the fight is won.
#: Deliberately this short: a data-driven list of 46 "setup" cards (every
#: Vulnerable, Strength, per-play power), moved first and checked on a copy
#: of the fight, changed 21 of 250 logged first moves: 5 better, 8 worse.
SETUP_FIRST = {"Rage": True, "Monologue": True, "Oblivion": False}


def setup_first(order: list[dict[str, int]], line: list[str], state: dict[str, Any]) -> None:
    """Move each SETUP_FIRST card before the first card it improves.
    Edits ``order`` and ``line`` in place."""
    cards = [*(state.get("hand") or []), *((state.get("draw_pile") or {}).get("cards") or [])]
    types = {str(c.get("name")): str(c.get("type")) for c in cards if isinstance(c, dict)}
    for setup, attacks_only in SETUP_FIRST.items():
        steps = [s for s in line if s[:1].isdigit()]
        names = [s.split(". ", 1)[1].removesuffix(" (drawn)") for s in steps]
        base = [n.rstrip("+") for n in names]
        if setup not in base:
            continue
        r = base.index(setup)
        first = next((i for i, n in enumerate(names[:r])
                      if base[i] not in SETUP_FIRST and (not attacks_only or types.get(n) == "Attack")), None)
        if first is None:
            continue
        moved = [*steps[:first], steps[r], *steps[first:r], *steps[r + 1:]]
        tail = [s for s in line if not s[:1].isdigit()]
        line[:] = [f"{i}. {s.split('. ', 1)[1]}" for i, s in enumerate(moved, 1)] + tail
        by_step = {o["step"]: o for o in order}
        new_pos = {old: new for new, old in enumerate(
            [*range(1, first + 1), r + 1, *range(first + 1, r + 1), *range(r + 2, len(steps) + 1)], 1)}
        for step, o in by_step.items():
            o["step"] = new_pos.get(step, step)


#: Cards whose payoff counts the enemy's Vulnerable when played (Dominate:
#: 1 Strength per Vulnerable, for the rest of the fight). The search scores
#: the extra Strength only as far as it looks ahead, so on a long fight it
#: played Dominate before Tremble and gave up 3 Strength (Terror Eel,
#: 2026-10-27).
COUNTS_VULNERABLE = {"DOMINATE"}
_VULN = re.compile(r"Apply (\d+) Vulnerable", re.I)


def _cost(card: dict[str, Any]) -> int:
    for key in ("effective_cost", "cost"):
        v = card.get(key)
        if isinstance(v, int) and v >= 0:
            return v
    return 99


def vulnerable_first(advice: dict[str, Any], state: dict[str, Any]) -> str | None:
    """When the best play is Dominate and a card that applies Vulnerable to
    the same enemy is also affordable this turn, recommend that card first.
    Edits ``advice`` in place; returns a HUD line when it switched."""
    best = advice.get("best") or {}
    hand = [c for c in state.get("hand") or [] if isinstance(c, dict)]
    ci = best.get("card_index")
    if best.get("action") != "play" or not isinstance(ci, int) or not 0 <= ci < len(hand):
        return None
    payoff = hand[ci]
    if str(payoff.get("id") or "").upper() not in COUNTS_VULNERABLE:
        return None
    energy = int(state.get("energy") or 0)
    target = best.get("target_index")
    picks = []
    for i, c in enumerate(hand):
        m = _VULN.search(re.sub(r"\[/?[a-z]+\]", "", str(c.get("description") or "")))
        if i == ci or not m or _cost(c) + _cost(payoff) > energy:
            continue
        picks.append((int(m.group(1)), -_cost(c), i))
    if not picks:
        return None
    _, _, i = max(picks)
    option = next((o for o in advice.get("options") or []
                   if o.get("card_index") == i and o.get("target_index") in (target, None)), None)
    if option is None:
        return None
    for o in advice.get("options") or []:
        o["is_best"] = o is option
    advice["best"] = {"action": "play", "card_index": i, "target_index": option.get("target_index"),
                      "label": option.get("label")}
    advice["pv"] = [option.get("label")]
    return f"{hand[i].get('name')} first: its Vulnerable makes {payoff.get('name')} give more Strength"


def prefer_upgraded(advice: dict[str, Any], state: dict[str, Any], tie_q: float = 0.005) -> None:
    """On a tie between two copies of the same card (Defend vs Defend+), keep
    the best as the upgraded copy. The search scores copies separately and
    can rank the plain one first by noise or on an exact tie (a won fight
    where every move scores the same). Edits ``advice`` in place."""
    best = advice.get("best") or {}
    hand = [c for c in state.get("hand") or [] if isinstance(c, dict)]
    ci = best.get("card_index")
    if best.get("action") != "play" or not isinstance(ci, int) or not 0 <= ci < len(hand):
        return
    card = hand[ci]
    if card.get("upgraded") or str(card.get("name") or "").endswith("+"):
        return
    options = advice.get("options") or []
    mine = next((o for o in options if o.get("card_index") == ci and o.get("target_index") == best.get("target_index")), None)
    if mine is None or mine.get("q") is None:
        return
    for o in options:
        j = o.get("card_index")
        if not isinstance(j, int) or j == ci or not 0 <= j < len(hand) or o.get("q") is None:
            continue
        other = hand[j]
        same = str(other.get("id") or "") == str(card.get("id") or "") and o.get("target_index") == best.get("target_index")
        upgraded = other.get("upgraded") or str(other.get("name") or "").endswith("+")
        if same and upgraded and o["q"] >= mine["q"] - tie_q:
            for x in options:
                x["is_best"] = x is o
            advice["best"] = {**best, "card_index": j, "label": o.get("label")}
            if advice.get("pv"):
                advice["pv"] = [o.get("label"), *advice["pv"][1:]]
            return
