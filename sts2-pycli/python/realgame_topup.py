"""Personal build: double-check the fight search's runner-ups before advising.

The search splits one budget over the root moves. Once one move looks good
early it takes most of the visits, and the others keep rough Q estimates
from a few hundred playouts (GH #494 names the same artifact for the
coach's grading of the PLAYED move). Live, on the Queen fight: End turn
took 12,371 of 16,000 visits at +0.82; Graveblast got 1,007 at -0.23. Each
searched on its own (``mcts_score_action_forced``, 3 x 6,000), Graveblast
won 87.7% against End turn's 83.6%: Graveblast fetched Scourge, Wisp+ paid
for it, and Scourge drew two more cards.

So: re-search the best move and the few under-visited runner-ups that are
not far behind, each on its own budget, in parallel (the forced search runs
off the GIL), and switch only when a runner-up wins clearly more often.
"""

from __future__ import annotations

import concurrent.futures
import logging
from typing import Any

logger = logging.getLogger(__name__)

#: A runner-up is re-checked when it got under this share of the visits...
MAX_SHARE = 0.15
#: ...and its Q is within this of the best's (the live miss was 1.05 behind).
MAX_GAP = 1.2
#: At most this many runner-ups (plus the best) per decision.
MAX_CHECKS = 3
#: Switch only when the runner-up's win chance beats the best's by this much.
MIN_WIN_GAIN = 0.02
#: Seeds per re-check; their results are averaged.
SEEDS = (101, 102)
#: Fresh seeds that must agree before a switch.
CONFIRM_SEEDS = (103, 104)

_pool = concurrent.futures.ThreadPoolExecutor(max_workers=8, thread_name_prefix="topup")
_seed_pool = concurrent.futures.ThreadPoolExecutor(max_workers=16, thread_name_prefix="topup-seed")


def _engine_action(option: dict[str, Any], wire_hand: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The option as an engine action; None for what cannot be re-searched
    here (potions and anything not a plain play / end turn)."""
    if option.get("label") == "End turn":
        return {"action": "end_turn"}
    idx = option.get("card_index")
    if idx is None:
        return None
    for i, card in enumerate(wire_hand):
        if card.get("stable_idx") == idx:
            t = option.get("target_index")
            return {"action": "play", "card_index": i, "target_index": None if t in (None, -1) else t}
    return None


def _score(sts2_rs: Any, cs: Any, action: dict[str, Any], sims: int, hidden_info: str | None,
           seeds: tuple[int, ...] = SEEDS) -> tuple[float, float] | None:
    def one(seed: int) -> Any:
        kwargs: dict[str, Any] = {"sims": sims, "seed": seed}
        if hidden_info:
            kwargs["hidden_info"] = hidden_info
        return sts2_rs.mcts_score_action_forced(cs, action, **kwargs)

    # Seeds side by side (same results, each seed is deterministic): the
    # sts2_rs search releases the GIL. Own pool, so a _score running on
    # `_pool` never waits on a slot in it.
    rows = list(_seed_pool.map(one, seeds))
    if any(r is None for r in rows):
        return None
    return (sum(r["p_win"] for r in rows) / len(rows), sum(r["q"] for r in rows) / len(rows))


def double_check(
    sts2_rs: Any,
    cs: Any,
    advice: dict[str, Any],
    wire_hand: list[dict[str, Any]],
    sims: int,
    hidden_info: str | None = None,
) -> str | None:
    """Re-search under-visited runner-ups; switch ``advice`` in place when
    one clearly wins more. Returns a HUD line when it switched, else None."""
    options = advice.get("options") or []
    best = next((o for o in options if o.get("is_best")), None)
    if best is None or best.get("q") is None:
        return None
    best_action = _engine_action(best, wire_hand)
    if best_action is None:
        return None
    candidates = [
        o for o in options
        if o is not best and o.get("q") is not None
        and (o.get("visit_share") or 0) < MAX_SHARE
        and best["q"] - o["q"] <= MAX_GAP
        and _engine_action(o, wire_hand) is not None
    ]
    candidates = sorted(candidates, key=lambda o: -o["q"])[:MAX_CHECKS]
    if not candidates:
        return None
    jobs = {id(o): _pool.submit(_score, sts2_rs, cs, _engine_action(o, wire_hand), sims, hidden_info)
            for o in [best, *candidates]}
    scores = {k: f.result() for k, f in jobs.items()}
    base = scores.get(id(best))
    if base is None:
        return None
    winner, win_score = best, base
    for o in candidates:
        s = scores.get(id(o))
        if s and s[0] >= win_score[0] + MIN_WIN_GAIN and s[1] > win_score[1]:
            winner, win_score = o, s
    if winner is best:
        return None
    # Confirm on fresh seeds before switching: on 250 logged turns the
    # single pass switched 8 times, 5 better and 3 worse (one by -10 points
    # of win chance), each a runner-up that looked better by luck.
    w_act, b_act = _engine_action(winner, wire_hand), best_action
    w2 = _pool.submit(_score, sts2_rs, cs, w_act, sims, hidden_info, CONFIRM_SEEDS)
    b2 = _pool.submit(_score, sts2_rs, cs, b_act, sims, hidden_info, CONFIRM_SEEDS)
    w_again, b_again = w2.result(), b2.result()
    if not w_again or not b_again or w_again[0] < b_again[0] + MIN_WIN_GAIN:
        logger.info("double-check: %s not confirmed (%s vs %s)", winner["label"], w_again, b_again)
        return None
    win_score = ((win_score[0] + w_again[0]) / 2, (win_score[1] + w_again[1]) / 2)
    base = ((base[0] + b_again[0]) / 2, (base[1] + b_again[1]) / 2)
    best["is_best"] = False
    winner["is_best"] = True
    advice["best"] = {k: winner.get(k) for k in ("action", "card_index", "target_index", "label") if k in winner}
    advice["best"]["action"] = "end_turn" if winner.get("label") == "End turn" else "play"
    advice["pv"] = [winner["label"]]
    logger.info("double-check: %s %.3f over %s %.3f", winner["label"], win_score[0], best["label"], base[0])
    return (f"Double-checked: {winner['label']} wins {win_score[0]:.0%} vs {best['label']} {base[0]:.0%} "
            "when each is searched on its own (the first search barely looked at it)")
