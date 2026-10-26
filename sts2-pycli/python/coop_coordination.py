"""Co-op: hold a move the teammate is about to make better.

The combat search plans for the local player alone, so it cannot see that a
teammate who has not ended their turn is about to change the value of our
move: their Vulnerable makes our attack hit harder, their Weak or block makes
our block partly wasted. Every peer simulates every player, so the mod sends
the teammate's hand (PROTOCOL.md ``allies``) and this module reads it.

It does not change WHAT the advisor recommends, only WHEN: the reply carries
``wait_for_ally`` and autoplay holds the move a short, capped time (the mod
caps the whole round too, so two bots waiting on each other cannot
deadlock). A new board -- the teammate played -- brings fresh advice anyway.
For a human it is the note: "wait for the teammate's Bash".

The card tables come from the v0.107.1 + v0.111.0 decompilation: every card
whose ``OnPlay`` applies the power to an enemy (``cardPlay.Target`` or the
hittable enemies). Left out on purpose: DOUBT (a curse that makes its owner
Weak) and MAD_SCIENCE (its effect is chosen at runtime).
"""

from __future__ import annotations

from typing import Any

#: Cards that make an enemy Vulnerable: our attack should land after them.
ENEMY_VULNERABLE = frozenset({
    "ASSASSINATE", "BASH", "BEAM_CELL", "BREAK", "COMET", "DOMINATE", "EXPOSE",
    "FALLING_STAR", "FEAR", "GAMMA_BLAST", "HIGH_FIVE", "KNOW_THY_PLACE",
    "METEOR_SHOWER", "MOLTEN_FIST", "PUTREFY", "SHOCKWAVE", "SQUASH", "TAUNT",
    "THUNDERCLAP", "TREMBLE", "UPPERCUT",
})  # fmt: skip

#: Cards that make an enemy Weak: our block should come after them.
ENEMY_WEAK = frozenset({
    "COMET", "DEATHBRINGER", "DEFY", "FALLING_STAR", "GAMMA_BLAST",
    "GO_FOR_THE_EYES", "HAZE", "KNOW_THY_PLACE", "LEG_SWEEP", "MALAISE",
    "METEOR_SHOWER", "NEUTRALIZE", "NULL", "PUTREFY", "SCARE", "SHOCKWAVE",
    "SUCKER_PUNCH", "SUPPRESS", "UPPERCUT",
})  # fmt: skip

#: Multiplayer cards that block for us (an ally of theirs), or cover us
#: (INTERCEPT: the teammate takes our hits): our block should come after.
ALLY_BLOCK = frozenset({"CONSTELLATION", "DEMONIC_SHIELD", "LIFT", "RALLY", "INTERCEPT"})

#: How long one reply may hold its move. The mod caps the round at 6 s.
WAIT_MS = 2500


def _card_id(card: dict[str, Any]) -> str:
    return str(card.get("id") or "").upper()


def _affordable(card: dict[str, Any], energy: int) -> bool:
    if card.get("costs_x"):
        return energy > 0
    cost = card.get("effective_cost", card.get("cost"))
    return isinstance(cost, int) and 0 <= cost <= energy


def _blocks(card: dict[str, Any]) -> bool:
    stats = card.get("stats") or {}
    return any("block" in str(k) and isinstance(v, int) and v > 0 for k, v in stats.items())


def _every_enemy_has(state: dict[str, Any], power_prefix: str) -> bool:
    alive = [
        e for e in state.get("enemies") or [] if not e.get("is_dead") and (e.get("hp") or 0) > 0
    ]
    return bool(alive) and all(
        any(str(p.get("id") or "").startswith(power_prefix) for p in e.get("powers") or [])
        for e in alive
    )


def ally_wait(state: dict[str, Any], advice: dict[str, Any]) -> dict[str, Any] | None:
    """The ``wait_for_ally`` hint for this combat reply, or ``None``.

    Args:
        state: The ``combat_action`` board the advice answers.
        advice: The advice about to be sent; only its ``best`` is read.

    Returns:
        ``{"reason", "card", "character", "max_ms"}`` when a living teammate
        who has not ended their turn can afford a card that should go before
        our recommended play; else ``None``.
    """
    best = advice.get("best") or {}
    idx = best.get("card_index")
    hand = state.get("hand") or []
    if not isinstance(idx, int) or not 0 <= idx < len(hand):
        return None  # end turn, a potion, or no move: nothing to hold
    ours = hand[idx]
    want: list[tuple[frozenset[str], str]] = []
    if ours.get("type") == "Attack" and not _every_enemy_has(state, "VULNERABLE"):
        want.append((ENEMY_VULNERABLE, "Vulnerable before our attack"))
    if _blocks(ours):
        if not _every_enemy_has(state, "WEAK"):
            want.append((ENEMY_WEAK, "Weak before our block"))
        want.append((ALLY_BLOCK, "their block for us first"))
    if not want or _card_id(ours) in ENEMY_VULNERABLE | ENEMY_WEAK:
        return None
    for ally in state.get("allies") or []:
        if ally.get("is_dead") or ally.get("ended_turn"):
            continue
        energy = int(ally.get("energy") or 0)
        for card in ally.get("hand") or []:
            if not _affordable(card, energy):
                continue
            for ids, reason in want:
                if _card_id(card) in ids:
                    return {
                        "reason": reason,
                        "card": card.get("name") or _card_id(card),
                        "character": ally.get("character") or "",
                        "max_ms": WAIT_MS,
                    }
    return None


def apply_ally_wait(state: dict[str, Any], advice: dict[str, Any]) -> dict[str, Any]:
    """Attach :func:`ally_wait` to a combat reply and name it in the note."""
    wait = ally_wait(state, advice)
    if wait is None:
        return advice
    advice = dict(advice)
    advice["wait_for_ally"] = wait
    hint = f"co-op: wait for the teammate's {wait['card']} ({wait['reason']})"
    advice["note"] = f"{advice['note']} · {hint}" if advice.get("note") else hint
    return advice
