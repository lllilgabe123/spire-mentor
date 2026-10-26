"""Personal-build fight foresight for the real-game advisor.

Three read-only helpers the advisor runs next to its search. None of them
changes the game: they only shape what the search is told and what the HUD
says.

* :func:`scale_for_teammate` -- co-op. The engine simulates one player, so
  in a two-player fight it thinks only your cards ever hurt the enemies and
  plans for a fight several turns longer than the real one. The mod sends
  the share of this fight's damage your teammates have dealt; scaling every
  enemy's HP (and max HP, so %-thresholds such as slime splits keep their
  timing) by ``1 - share`` makes time-to-kill come out as if the teammate's
  damage kept arriving at the rate it has so far.
* :func:`forecast` -- what ending the turn right now does (the engine plays
  the enemy turn on a clone of the board, with the RNG cursors the mod
  read), the block that would keep you alive when it is lethal, and the
  enemies' moves for the next turns if the fight goes on.
"""

from __future__ import annotations

import copy
import math
from typing import Any

#: Never scale an enemy below this fraction of its real HP: a teammate who
#: has done nearly everything so far may not keep doing it.
MAX_TEAMMATE_SHARE = 0.85

#: Enemy turns the timeline looks ahead (the game already shows the next one).
TIMELINE_TURNS = 3

#: Cards whose reward only comes on a kill (decomp: StaticHoverTip.Fatal):
#: Feed (+max HP), Hand of Greed (+gold), The Hunt. The fight search scores a
#: fight by end HP and potions only, so it sees none of those rewards past
#: the fight and spends these as plain attacks.
FATAL_REWARD_CARDS = frozenset({"FEED", "HAND_OF_GREED", "THE_HUNT"})

#: How close (root Q) another attack must be to take over from a kill-reward
#: card that would not kill: the search's noise band, the same as SimTieBreak's
#: TieQ (~1.4 HP at the victory band's exchange rate).
FATAL_TIE_Q = 0.005


def scale_for_teammate(wire: dict[str, Any], share: float) -> dict[str, Any]:
    """A copy of ``wire`` with every living enemy's HP scaled by ``1 - share``."""
    share = max(0.0, min(MAX_TEAMMATE_SHARE, float(share or 0.0)))
    if share <= 0.0:
        return wire
    out = copy.deepcopy(wire)
    keep = 1.0 - share
    for m in out.get("encounter") or []:
        if m.get("is_dead") or not m.get("hp"):
            continue
        m["hp"] = max(1, math.ceil(m["hp"] * keep))
        if m.get("max_hp"):
            m["max_hp"] = max(m["hp"], math.ceil(m["max_hp"] * keep))
    return out


def keep_kill_reward_card(
    sts2_rs: Any, tags: list[Any], cs: Any, hand: list[dict[str, Any]], state_hand: list[dict[str, Any]]
) -> tuple[list[Any], str | None]:
    """When the best move is a kill-reward card (:data:`FATAL_REWARD_CARDS`)
    that would NOT kill its target, and another attack is within
    :data:`FATAL_TIE_Q`, rank that attack first instead.

    Whether it kills is played out on a clone of ``cs`` (the unscaled board),
    so Strength, Vulnerable, block and the rest all count.

    Args:
        tags: The ranked HintTags, best first.
        hand: The wire's ``deck.hand`` (maps engine card positions to
            ``stable_idx``).
        state_hand: The mod's ``hand`` (``stable_idx`` -> card id and type).

    Returns:
        The (possibly re-ranked) tags and a note for the HUD, or None.
    """
    from sts2_hint.recommender import _stable_idx_for_action

    def card(tag: Any) -> dict[str, Any]:
        i = tag.stable_idx
        return state_hand[i] if 0 <= i < len(state_hand) else {}

    if not tags:
        return tags, None
    best = tags[0]
    if (card(best).get("id") or "").upper() not in FATAL_REWARD_CARDS or best.target_index is None:
        return tags, None
    action = next(
        (
            a for a in cs.legal_actions()
            if a.get("action") == "play"
            and a.get("target_index") == best.target_index
            and _stable_idx_for_action(a, hand) == best.stable_idx
        ),
        None,
    )
    if action is None:
        return tags, None
    c = cs.clone()
    c.apply(action)
    target = (sts2_rs.state_to_dict(c).get("encounter") or [])[best.target_index]
    if target.get("is_dead") or int(target.get("hp", 1)) <= 0:
        return tags, None  # it kills: the reward is real, keep it
    alts = [
        t for t in tags[1:]
        if best.q - t.q <= FATAL_TIE_Q
        and card(t).get("type") == "Attack"
        and (card(t).get("id") or "").upper() not in FATAL_REWARD_CARDS
    ]
    if not alts:
        return tags, None
    alt = max(alts, key=lambda t: t.q)
    reranked = [alt._replace(is_best=True)] + [t._replace(is_best=False) for t in tags if t is not alt]
    name = card(best).get("name") or card(best).get("id")
    other = card(alt).get("name") or card(alt).get("id")
    return reranked, f"{name} won't kill now: {other} instead (about equal), saving {name} for a kill"


def _intent_text(intents: list[dict[str, Any]]) -> str:
    parts = []
    for it in intents or []:
        kind = it.get("kind") or "?"
        if kind == "Attack":
            dmg = it.get("effective_amount", it.get("amount", 0))
            times = it.get("times") or 1
            parts.append(f"attack {dmg}x{times}" if times > 1 else f"attack {dmg}")
        elif kind in ("Defend", "Block"):
            parts.append("block")
        elif kind == "Buff":
            parts.append("buff")
        elif kind == "Debuff" or kind == "DebuffStrong":
            parts.append("debuff")
        else:
            parts.append(kind.lower())
    return " + ".join(parts) or "nothing"


def _end_turn(sts2_rs: Any, cs: Any) -> tuple[Any, dict[str, Any]]:
    c = cs.clone()
    c.apply({"action": "end_turn"})
    return c, sts2_rs.state_to_dict(c)


def _alive(s: dict[str, Any]) -> list[dict[str, Any]]:
    return [m for m in s.get("encounter") or [] if not m.get("is_dead") and not m.get("is_escaped")]


def _block_to_survive(sts2_rs: Any, wire: dict[str, Any], hp_now: int) -> int | None:
    """Smallest extra block that keeps you alive through ending the turn now,
    found by re-simulating the same board with more block. None past 300."""
    def survives(extra: int) -> bool:
        w = copy.deepcopy(wire)
        w["player"]["block"] = int(w["player"].get("block", 0)) + extra
        _, s = _end_turn(sts2_rs, sts2_rs.state_from_dict(w))
        return s["player"]["hp"] > 0

    if not survives(300):
        return None
    lo, hi = 0, 300
    while lo < hi:
        mid = (lo + hi) // 2
        if survives(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def forecast(sts2_rs: Any, cs: Any, wire: dict[str, Any], names: list[str]) -> dict[str, Any]:
    """Lethal check and enemy timeline for the board ``cs`` (built from ``wire``).

    Returns ``{"lines": [...], "danger": bool}``; ``lines`` are ready to show.
    """
    now = sts2_rs.state_to_dict(cs)
    hp0 = int(now["player"]["hp"])
    lines: list[str] = []
    danger = False

    c, s = _end_turn(sts2_rs, cs)
    hp1 = int(s["player"]["hp"])
    if hp1 <= 0:
        danger = True
        need = _block_to_survive(sts2_rs, wire, hp0)
        fix = f"you need {need} more block to live" if need else "block won't save you: kill or weaken them"
        lines.append(f"LETHAL: ending your turn now kills you ({hp0} HP). {fix}.")
    elif hp1 < hp0:
        lines.append(f"Ending turn now: -{hp0 - hp1} HP ({hp0} -> {hp1}).")
    else:
        lines.append("Ending turn now: you take no damage.")
    # Necrobinder: Osty absorbs every point of attack damage your Block does
    # not, so "you take no damage" can mean Osty takes it all. Say so: extra
    # Block (a Fortifier on 6 Block before a 9x3 beam) can be the move that
    # keeps him alive although your own HP would not change.
    o0, o1 = now["player"].get("osty") or {}, s["player"].get("osty") or {}
    if o0.get("alive") and int(o0.get("hp") or 0) > 1:
        before, after = int(o0["hp"]), int(o1.get("hp") or 0) if o1.get("alive") else 0
        if after <= 0:
            lines[-1] += f" Osty dies ({before} HP): Block soaks hits before he does."
        elif after < before:
            lines[-1] += f" Osty takes {before - after} ({before} -> {after}): Block soaks hits before he does."

    # Enemies' plans for the turns after the one the game already shows. The
    # look-ahead has you do nothing, so moves that depend on their HP can
    # differ once you hit them; most patterns do not.
    timeline: dict[int, list[str]] = {}
    for _ in range(TIMELINE_TURNS - 1):
        if int(s["player"]["hp"]) <= 0 or not _alive(s):
            break
        for i, m in enumerate(s.get("encounter") or []):
            if not m.get("is_dead") and not m.get("is_escaped"):
                timeline.setdefault(i, []).append(_intent_text(m.get("intents") or []))
        c, s = _end_turn(sts2_rs, c)
    for i, moves in timeline.items():
        name = names[i] if i < len(names) and names[i] else (now["encounter"][i].get("kind") if i < len(now["encounter"]) else f"enemy {i}")
        lines.append(f"{name} next: " + ", then ".join(moves))
    return {"lines": lines, "danger": danger, "hp_after": hp1}
