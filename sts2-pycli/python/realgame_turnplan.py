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


def rage_first(order: list[dict[str, int]], line: list[str], state: dict[str, Any]) -> None:
    """Rage (0 cost, Block per Attack after it this turn) goes before the
    plan's first Attack; the search sometimes ranks it after them by more
    than its noise. Edits ``order`` and ``line`` in place."""
    cards = [*(state.get("hand") or []), *((state.get("draw_pile") or {}).get("cards") or [])]
    types = {str(c.get("name")): str(c.get("type")) for c in cards if isinstance(c, dict)}
    steps = [s for s in line if s[:1].isdigit()]
    names = [s.split(". ", 1)[1].removesuffix(" (drawn)") for s in steps]
    if "Rage" not in names:
        return
    r = names.index("Rage")
    first_attack = next((i for i, n in enumerate(names[:r]) if types.get(n) == "Attack"), None)
    if first_attack is None:
        return
    moved = [*steps[:first_attack], steps[r], *steps[first_attack:r], *steps[r + 1:]]
    tail = [s for s in line if not s[:1].isdigit()]
    line[:] = [f"{i}. {s.split('. ', 1)[1]}" for i, s in enumerate(moved, 1)] + tail
    by_step = {o["step"]: o for o in order}
    new_pos = {old: new for new, old in enumerate(
        [*range(1, first_attack + 1), r + 1, *range(first_attack + 1, r + 1), *range(r + 2, len(steps) + 1)], 1)}
    for step, o in by_step.items():
        o["step"] = new_pos.get(step, step)
