"""Advisor backend for SpireMentor advisor mode (PROTOCOL.md section 5, v3).

The mod is the server; this client connects to it (same transport as every
other ``realgame_*`` driver) and answers ``advice_request`` messages with a
full ranked ``advice`` payload. It never sends game-driving verbs: advice
is rendered by the mod's HUD, never executed by this process (the phase-7
takeover executes game-side).

Pipeline per ``advice_request`` (combat only in v1):

1. Reinterpret the payload as the ``combat_action`` snapshot it wraps
   (``advice_for`` field) and translate it with
   :func:`realgame_adapter.to_headless_combat_play` — the same path the
   autopilot MCTS driver uses.
2. Run the deployed honest-search configuration
   (``hidden_info="beliefdraw"``, ``c_puct=2.5``, ``draw_widening_c=2.0``,
   and the run agent's EndTurn widening 5.787 / 0.436, which the engine
   applies with the mode — NOT the leaky clairvoyant default) via
   ``sts2_rs.mcts_recommend``, keeping the raw result so the principal
   variation survives (``recommend_all`` drops it). Budget, tree shape, pick
   rule and the last-boss win value are the Rust run agent's
   (:func:`fight_search_cfg`; ``--sims`` / ``--boss-sims`` / ``--threads``).
3. Map the ranked rows through :func:`sts2_hint.recommender.tags_from_result`
   and build the wire ``advice`` object: ``p_win``, ``best`` (executable),
   ``options[]`` (label / p_win / q / visits / visit_share / W-L-T /
   is_best / low_confidence / card_index / target_index), ``pv``.

``human_action`` messages and every advice exchange are appended to a JSONL
log under ``--log-dir`` for the phase-9 review pipeline. Never-crash
policy: any per-request failure is logged, answered with an empty advice
object (the HUD just shows nothing), and the loop continues. Messages this
process does not understand (e.g. autopilot decision messages if it is
accidentally connected to an autopilot session) are logged and ignored —
it must never answer them.

Usage::

    uv run python python/realgame_advisor.py [--host 127.0.0.1] [--port 9002]
        [--sims 4000] [--boss-sims 16000] [--threads 1]
        [--log-dir logs/realgame_advisor]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import os
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import realgame_hand_select
import runtime_paths
from coop_coordination import apply_ally_wait
from enemy_labels import enemy_label
from realgame_adapter import (
    UnsupportedGameVersionError,
    character_display_name,
    envelope_game_version,
    run_ascension,
    to_headless_combat_play,
)
from realgame_advisor_mirror import AdvisorMirror
from realgame_bridge import DEFAULT_HOST, DEFAULT_PORT, RealGameBridgeClient
from realgame_coach import CoachGrader, GraderThread
from realgame_control import BRIDGE_MESSAGE_TYPES, ControlPlane, StaticSite
from realgame_livetree import LiveRunTree
from realgame_workbench_log import WorkbenchRunLogger
from sts2_hint.recommender import (
    CHOOSE_SKIP_STABLE_IDX,
    CHOOSE_STABLE_IDX_BASE,
    END_TURN_STABLE_IDX,
    POTION_STABLE_IDX_BASE,
    HintTag,
    RecommendConfig,
    tags_from_result,
)

logger = logging.getLogger(__name__)

#: The deployed honest-search knobs (see RecommendConfig.hidden_info notes:
#: beliefdraw + c_puct 2.5 + draw_widening_c 2.0 travel together).
#:
#: The EndTurn widening is deliberately NOT listed: it is left to the engine,
#: which applies the run agent's values with the mode
#: (`MctsConfig::apply_mode_dialect`, GH #758). Before that the advisor
#: searched EndTurn at the pre-#467 0.5 / 0.4 while every published win rate
#: was measured at 5.787 / 0.436.
#:
#: The `c_puct` here is the value for the DEPLOYED budgets. The search resolves
#: it per decision via `belief_draw_c_puct(sims, threads)` instead, because the
#: optimum rises with the budget and `--sims` is chosen independently of this
#: dict. At every deployed budget the two agree; they diverge only when the
#: budget is trimmed, which is the case this protects.
#:
#: This stays on the BASE dialect on purpose (GH #830). The run agent adds an
#: Ironclad layer on top (`sts2_runai::policy::IRONCLAD_SEARCH_KNOBS` and a BC
#: move prior in elite and boss fights) that the advisor does not take, so for
#: Ironclad "the deployed search" means two things: this one, and the run agent's.
HONEST_SEARCH = {
    "hidden_info": "beliefdraw",
    "c_puct": 2.5,
    "draw_widening_c": 2.0,
}

#: What this advisor advertises on every ``advice`` reply (PROTOCOL.md v5). A
#: mod sends the new ``human_action`` kinds (end turn, run-level choices) only
#: once it has seen ``coach >= 1``, because an older advisor's mirror would
#: drop on them. ``coach: 2`` (v5.1) adds the fight recap and the death
#: autopsy: the mod sends ``combat_ended`` and its Advisor-mode ``game_over``
#: only to an advisor at 2 or above.
ADVISOR_CAPABILITIES = {"coach": 2}

#: Acts in a standard run; the fallback for a mod too old to send
#: ``run_state.is_last_act``.
STANDARD_ACT_COUNT = 3


def fight_search_cfg(cfg: RecommendConfig, state: dict[str, Any]) -> RecommendConfig:
    """``cfg`` with the boss win value the run agent uses for THIS fight.

    The run agent (``MctsCombatPolicy::pick_search_setup``, default
    ``STS2_RUNAI_BOSS_CARRY``) changes one thing per fight: on the run's LAST
    boss nothing carries forward, so only winning counts
    (``sts2_mcts::boss_win_carry``). Every other fight keeps the default value.

    The last-boss test reads the mod's ``run_state.is_last_act`` /
    ``awaiting_second_boss``. A mod too old to send them falls back to "act 3,
    and no A10 second boss can be pending" — below A10 the act-3 boss is the
    last one; at A10 it cannot tell which of the two this is, so it keeps the
    default value rather than guess.
    """
    import sts2_rs

    if (state.get("room_type") or "") != "Boss":
        return cfg
    run_state = state.get("run_state") or {}
    ascension = int(run_state.get("ascension") or 0)
    is_last_act = run_state.get("is_last_act")
    awaiting_second = run_state.get("awaiting_second_boss")
    if is_last_act is None:
        is_last_act = int(state.get("act") or 0) >= STANDARD_ACT_COUNT
    if awaiting_second is None:
        if ascension >= 10 and is_last_act:
            return cfg
        awaiting_second = False
    if not is_last_act or awaiting_second:
        return cfg
    carry = sts2_rs.boss_win_carry(True, True, False, ascension)
    if carry is None:
        return cfg
    return dataclasses.replace(cfg, win_hp_carry=carry[0], win_potion_carry=carry[1])


def _tag_label(tag: HintTag, state: dict[str, Any]) -> str:
    """Human-readable option label: card/potion name plus target name."""
    hand = state.get("hand") or []
    enemies = state.get("enemies") or []
    if tag.stable_idx == END_TURN_STABLE_IDX:
        return "End turn"
    if tag.stable_idx >= 0:
        name = "?"
        if 0 <= tag.stable_idx < len(hand):
            name = hand[tag.stable_idx].get("name") or hand[tag.stable_idx].get("id") or "?"
        label = str(name)
    elif tag.stable_idx > POTION_STABLE_IDX_BASE - 5 and tag.stable_idx <= POTION_STABLE_IDX_BASE:
        dense = POTION_STABLE_IDX_BASE - tag.stable_idx
        non_empty = [p for p in (state.get("potions") or []) if not p.get("empty")]
        name = (
            non_empty[dense].get("name") if 0 <= dense < len(non_empty) else None
        ) or f"potion {dense}"
        label = str(name)
    elif tag.stable_idx == CHOOSE_SKIP_STABLE_IDX:
        # The stop/skip pick of a pending choice, one above the band below and
        # so not caught by it -- it rendered as `action -299`.
        return "skip"
    elif tag.stable_idx <= CHOOSE_STABLE_IDX_BASE:
        # A pick out of a PENDING CHOICE (`-300 - i`, recommender.py:27). The
        # band had no arm at all, so every one of these rendered as the raw
        # index -- `action -302` on the HUD, which is what made the wedge
        # below look like a display glitch instead of the run-killer it was
        # (live run 004-silent-a8-0825-0139, floor 47, three times).
        #
        # Named against the WORLD's own offered list when the screen carries
        # one; the bare option number otherwise. The two agree only where the
        # advisor injected that list, so the number is not decoration -- it is
        # what the label falls back to when nobody has proven they agree.
        i = CHOOSE_STABLE_IDX_BASE - tag.stable_idx
        offered = state.get("cards") or []
        if 0 <= i < len(offered) and isinstance(offered[i], dict):
            named = offered[i].get("name") or offered[i].get("id")
            if named:
                return str(named)
        return f"choose option {i}"
    else:
        return f"action {tag.stable_idx}"
    target = enemy_label(enemies, tag.target_index)
    if target is not None:
        label += f" → {target}"
    return label


def _pv_labels(pv: list[Any], state: dict[str, Any]) -> list[str]:
    """Name the principal variation without lying about the deeper plies.

    ``mcts_recommend`` returns the PV as raw action dicts (``action_to_pydict``
    -- ``{"action": "play", "card_index": 3, "target_index": 0}``). Sending
    ``str(dict)`` is what put a wall of JSON on the HUD (reported 2026-08-05).

    Only ply 1 can be named: ``card_index`` indexes the hand at the state it
    was chosen in, and every later ply is a hand that does not exist yet (the
    search drew it). Naming those from the CURRENT hand would print confident
    nonsense, so they degrade to the action kind alone.
    """
    hand = state.get("hand") or []
    enemies = state.get("enemies") or []
    labels: list[str] = []
    for depth, action in enumerate(pv):
        if not isinstance(action, dict):
            labels.append(str(action))
            continue
        kind = str(action.get("action", "?"))
        if kind == "end_turn":
            labels.append("end turn")
            continue
        if depth > 0:
            labels.append(kind.replace("_", " "))
            continue
        idx = action.get("card_index")
        if kind == "play" and isinstance(idx, int) and 0 <= idx < len(hand):
            label = str(hand[idx].get("name") or hand[idx].get("id") or "?")
        else:
            label = kind.replace("_", " ")
        target = enemy_label(enemies, action.get("target_index"))
        if target is not None:
            label += f" → {target}"
        labels.append(label)
    return labels


def _best_to_action(tag: HintTag, state: dict[str, Any]) -> dict[str, Any] | None:
    """The executable ``best`` object (mod-side takeover dialect).

    ``card_index`` is the bridge hand index (== ``stable_idx`` for card
    plays); the potion ``slot`` is remapped from the engine's dense index
    to the bridge's absolute slot, same as
    :func:`realgame_mcts._remap_potion_slot`.
    """
    if tag.stable_idx == END_TURN_STABLE_IDX:
        return {"action": "end_turn"}
    if tag.stable_idx >= 0:
        best: dict[str, Any] = {"action": "play", "card_index": tag.stable_idx}
        if tag.target_index is not None:
            best["target_index"] = tag.target_index
        return best
    if POTION_STABLE_IDX_BASE - 5 < tag.stable_idx <= POTION_STABLE_IDX_BASE:
        dense = POTION_STABLE_IDX_BASE - tag.stable_idx
        non_empty = [p for p in (state.get("potions") or []) if not p.get("empty")]
        if 0 <= dense < len(non_empty) and isinstance(non_empty[dense].get("slot"), int):
            best = {"action": "potion", "slot": non_empty[dense]["slot"]}
            if tag.target_index is not None:
                best["target_index"] = tag.target_index
            return best
    return None


def wire_unplayable(state: dict[str, Any]) -> dict[int, str]:
    """Hand indices the GAME has already declared unplayable, and why.

    The bridge ships ``playable`` / ``unplayable_reason`` per hand card,
    computed by the live ``CardModel.CanPlay`` — the same predicate that will
    refuse the play a moment later. It is ground truth about this exact board,
    and it costs nothing.

    Returns:
        ``{hand_index: reason}`` for every card the game says cannot be played.
        Absent or malformed legality is treated as playable: a missing field
        must never suppress a legal move.
    """
    out: dict[int, str] = {}
    for i, card in enumerate(state.get("hand") or []):
        if isinstance(card, dict) and card.get("playable") is False:
            out[i] = str(card.get("unplayable_reason") or "unplayable")
    return out


def veto_unplayable(tags: list[HintTag], state: dict[str, Any]) -> tuple[list[HintTag], list[str]]:
    """Drop ranked plays the live game has already said it will refuse.

    The search runs on a RECONSTRUCTION of the board, and the reconstruction
    disagrees with the game about legality often enough to matter: across 51
    climb runs the live game refused 41 recommended plays, in three unrelated
    classes -- a second `Bound` card under the Queen's Chains of Binding (22),
    a card the mirror thought affordable (15), and an outright unplayable
    status/curse like Clumsy or Dazed (4). Each cost the escape hatch's three
    strikes and then an ENDED TURN, in exactly the act-3 fights the ladder is
    decided in.

    Fixing the reconstruction is the real repair and stays open (issue 12).
    This is the cheap guard in front of it: whatever the mirror believes, do
    not RECOMMEND a card the game has already told us it will not play. The
    runner-up is a legal move the search still ranked; a refusal is nothing.

    Returns:
        ``(kept, reasons)`` -- the surviving tags, best-first with ``is_best``
        re-stamped onto the new leader, and one human-readable string per
        vetoed play for the advice note.
    """
    veto = wire_unplayable(state)
    if not veto:
        return tags, []
    kept = [t for t in tags if not (t.stable_idx >= 0 and t.stable_idx in veto)]
    reasons = [
        f"hand[{t.stable_idx}]:{veto[t.stable_idx]}"
        for t in tags
        if t.stable_idx >= 0 and t.stable_idx in veto
    ]
    if not reasons:
        return tags, []
    if not kept:
        # Everything ranked is unplayable. Ending the turn is what the mod's
        # refusal escape hatch would reach anyway, three wasted requests later
        # -- reach it now, and say so. Cloned from a real tag so every
        # downstream field (bucket, confidence flags) stays well-formed.
        kept = [tags[0]._replace(stable_idx=END_TURN_STABLE_IDX, target_index=None, is_best=True)]
        return kept, reasons
    # The search's #1 may be the card that was just vetoed, so the flag has to
    # move with the crown -- the mod reads `is_best` to label the action.
    kept = [t._replace(is_best=(i == 0)) for i, t in enumerate(kept)]
    return kept, reasons


def build_advice(
    tags: list[HintTag],
    state: dict[str, Any],
    *,
    sims: int,
    pv: list[str] | None = None,
) -> dict[str, Any]:
    """The wire ``advice`` object (PROTOCOL.md section 5) from ranked tags."""
    if not tags:
        return {"note": "no legal actions"}
    tags, vetoed = veto_unplayable(tags, state)
    best = tags[0]
    total_visits = sum(t.visits for t in tags) or 1
    options = []
    for t in tags:
        options.append(
            {
                "stable_idx": t.stable_idx,
                "card_index": t.stable_idx if t.stable_idx >= 0 else None,
                "target_index": t.target_index,
                "label": _tag_label(t, state),
                "p_win": round(t.win_prob, 4),
                "q": round(t.q, 4),
                "visits": t.visits,
                "visit_share": round(t.visits / total_visits, 4),
                "wins": t.wins,
                "losses": t.losses,
                "ties": t.ties,
                "bucket": t.bucket_label,
                "is_best": t.is_best,
                "low_confidence": t.low_confidence,
            }
        )
    best_action = _best_to_action(best, state)
    note = f"MCTS {sims} sims · beliefdraw"
    if vetoed:
        # Loud on purpose: every one of these is a live disagreement between
        # the mirror and the game, and this note is the only place it is
        # counted now that the play never reaches the refusal log.
        note += f" · vetoed {len(vetoed)} the game calls unplayable ({', '.join(vetoed[:3])})"
    advice: dict[str, Any] = {
        "note": note,
        "p_win": round(best.win_prob, 4),
        "low_confidence": best.low_confidence or best.q_saturated,
        "options": options,
    }
    if best_action is not None:
        best_action["label"] = _tag_label(best, state)
        advice["best"] = best_action
    else:
        # A ranked search with nothing executable behind it. The mod waits for
        # a `best` it will never get -- "No fresh recommendation to execute"
        # -- and 180s later the stall guard kills the run. Silent before this:
        # the only trace was `best=None` in the advisor log and an unlabelled
        # option list on the HUD, which read as a rendering fault.
        advice["note"] += f" · no executable form for {_tag_label(best, state)}"
    if pv:
        advice["pv"] = pv
    return advice


def compute_combat_advice(
    state: dict[str, Any], cfg: RecommendConfig, sims: int, mirror: Any = None
) -> tuple[dict[str, Any], Any, RecommendConfig, list[dict[str, Any]]]:
    """One full search for one ``advice_request`` wrapping a combat board.

    Returns ``(advice, cs, call_cfg, rows)`` — the wire advice object, the
    native ``sts2_rs`` combat state the search ran on, the resolved config —
    so callers can run companion searches (``mcts_search_tree``) against
    the very same board — and the search's regret rows (:func:`combat_rows`),
    which the live coach grades the human's move against.

    Args:
        mirror: The fight's :class:`realgame_advisor_mirror.AdvisorMirror`.
            When it is carrying this fight the search runs on the CARRIED
            state instead of a fresh rebuild, which is the only way it sees
            the per-turn counters the wire does not express (GH #370 —
            ``card_plays_started_this_turn`` and friends read 0 forever on a
            rebuild, and two of them gate what is legal). ``None``, or a
            mirror that has dropped, keeps the stateless rebuild verbatim.
    """
    import sts2_rs
    from sts2_hint import adapter

    run_state = state.get("run_state") or {}
    character = character_display_name(run_state.get("character")) or cfg.character
    # `sims` varies per decision (boss vs regular) while a pinned `c_puct` is
    # chosen once at construction — that split is exactly the hazard
    # `belief_draw_c_puct` exists to close. Clearing the constant here hands it
    # to `RecommendConfig.resolved_c_puct`, which reads it off the budget THIS
    # decision actually gets (`call_cfg.sims`), so a trimmed boss/regular
    # budget can no longer keep a constant tuned for a budget it never runs.
    call_cfg = dataclasses.replace(
        cfg,
        character=character,
        ascension=run_ascension(run_state, cfg.ascension),
        sims=sims,
        c_puct=None,
    )

    headless = to_headless_combat_play(state)
    wire = adapter.translate(
        headless,
        character=call_cfg.character,
        combat_id=call_cfg.combat_id,
        seed=call_cfg.seed,
        ascension=call_cfg.ascension,
    )
    live = mirror.on_combat_board(wire, state) if mirror is not None else None
    if live is not None and live.state.pending_choice() is not None:
        # The world is asking for a COMBAT action, so it is not showing a
        # selection -- and a mirror parked on one answers a question nobody
        # asked. Every action the search can offer is a `choose`, which has no
        # executable form on this screen, so the advice ships with no `best`
        # and the run wedges until the stall guard kills it.
        #
        # How the mirror gets there: a card that opens a selection reports
        # LATE (PROTOCOL.md, `human_action`) -- the play arrives only once the
        # screen is answered -- so following it parks the mirror on a choice
        # the world has already resolved. `reconcile`'s parked verify normally
        # notices and re-seeds, but a card that RESHUFFLES on its way to the
        # screen sets `_crossed_shuffle`, which skips the whole `$.deck.`
        # subtree from that diff. The stale hand is then unverifiable rather
        # than wrong, the park survives, and it survives every reconcile
        # afterwards -- the fight never recovers. Silent+Prepared+ at Aeonglass
        # did exactly this three attempts running on 2026-08-25.
        #
        # Dropping costs this fight its carried state; keeping it costs the
        # run. The next board re-seeds.
        mirror.drop("parked on a selection the world has finished")
        live = None
    cs = result = None
    if live is not None:
        # Carried state wins: same board as the rebuild on every observable
        # (`reconcile` re-imposes them), plus the hidden fields no wire
        # carries. `search` clones internally, so `cs` must be its own clone
        # for the companion explorer to see the same thing.
        try:
            cs = live.state.clone()
            result = live.search(call_cfg)
        except Exception as exc:
            # Includes the clairvoyant refusal, which the advisor's pinned
            # `HONEST_SEARCH` never triggers but a caller could. Advice must
            # always render: fall through to the stateless rebuild.
            logger.warning("advisor mirror search failed, rebuilding: %s", exc)
            mirror.drop("search failed")
            cs = result = None
    if result is None:
        cs = sts2_rs.state_from_dict(wire)
        result = sts2_rs.mcts_recommend(
            cs,
            sims=call_cfg.sims,
            c_puct=call_cfg.resolved_c_puct(),
            depth=call_cfg.depth,
            seed=call_cfg.seed,
            threads=call_cfg.threads,
            fair=call_cfg.fair,
            rollout_policy=call_cfg.rollout_policy,
            hidden_info=call_cfg.hidden_info,
            draw_widening_c=call_cfg.resolved_draw_widening_c(),
            endturn_widening_c=call_cfg.endturn_widening_c,
            endturn_widening_alpha=call_cfg.endturn_widening_alpha,
            common_random_numbers=call_cfg.common_random_numbers,
            win_hp_carry=call_cfg.win_hp_carry,
            win_potion_carry=call_cfg.win_potion_carry,
        )
    tags = tags_from_result(result, wire["deck"]["hand"])
    pv = _pv_labels(list(result.get("pv") or [])[:4], state)
    advice = build_advice(tags, state, sims=sims, pv=pv)
    rows = combat_rows(result, headless, advice)
    return advice, cs, call_cfg, rows


def _row_key(d: dict[str, Any]) -> tuple[Any, Any]:
    return (d.get("stable_idx"), d.get("target_index"))


def combat_rows(
    result: dict[str, Any], headless: dict[str, Any], advice: dict[str, Any]
) -> list[dict[str, Any]]:
    """The search's per-action regret rows, crowned the way ``advice`` is.

    :func:`workbench.reasoning.rows_from_mcts` over the raw result -- the shape
    ``regret_review.assess_decision`` grades -- then two alignments with the
    advice the HUD shows, which the coach grades against:

    * ``is_best`` moves to the advice's best option. :func:`veto_unplayable`
      can re-crown the advice when the game refuses the search's #1, and rows
      that still crowned the refused card would grade the player against a
      move nobody could make.
    * Each row takes the advice option's label (potion and target names the
      advisor resolves against the wire board), and in return the options
      gain the rows' ``mean_final_hp`` / ``hp_sd`` / ``hp_n`` (additive keys).
    """
    from workbench.reasoning import rows_from_mcts

    rows = rows_from_mcts(result, headless)
    options = [o for o in advice.get("options") or [] if isinstance(o, dict)]
    by_key = {_row_key(r): r for r in reversed(rows)}  # first row per key wins
    for o in options:
        row = by_key.get(_row_key(o))
        if row is None:
            continue
        if o.get("label"):
            row["action"] = o["label"]
        o["mean_final_hp"] = row.get("mean_final_hp")
        o["hp_sd"] = row.get("hp_sd")
        o["hp_n"] = row.get("hp_n", 0)
    best = next((o for o in options if o.get("is_best")), None)
    if best is not None:
        crown = by_key.get(_row_key(best)) or next(
            (r for r in rows if r.get("stable_idx") == best.get("stable_idx")), None
        )
        if crown is not None:
            for r in rows:
                r["is_best"] = r is crown
            rows.sort(key=lambda r: (not r["is_best"], -r["win_prob"]))
    return rows


#: ``advice_for`` values answered by the PPO run policy rather than by MCTS.
#: These are the screens a human actually sees (autopilot never does: its
#: ICardSelector bypasses them), so the advice has to be built from the wire
#: payload the mod scrapes off the live screen -- see SimScreenAdvice.cs.
SCREEN_ADVICE_FOR = (
    "card_reward",
    "card_select",
    "map_select",
    "shop",
    "event",
    "rest_site",
    "boss_relic",
    "treasure",
    "card_bundle",
)

#: Surfaces answered by a DIFFERENT trained head than their own name suggests,
#: because the champion was never trained on them directly. Each one is a
#: judgement call about which trained question is closest -- and each one is
#: announced in the HUD note, because a ranking whose meaning has been
#: substituted must not read like a native one.
_SURFACE_STANDINS = {
    # A chest relic and a boss relic are the same question ("which of these
    # relics do I want"), only the price differs.
    "treasure": ("boss_relic", "scored with the boss-relic head"),
}

#: Surfaces the advisor answers but has no policy FOR, keyed to the player-facing
#: reason that rides on ``advice.hand_back`` (GH #874). A borrowed head is
#: honest in the HUD note, but it is still not the AI deciding, so F7 hands the
#: screen back instead of playing it. Delete an entry when its surface gets a
#: trained head (GH #379); the mod keeps no list of its own, so the pause goes
#: away with it.
_HAND_BACK_SURFACES = {
    "treasure": "the chest relic pick has no trained policy",
    "card_bundle": "card bundles have no trained policy",
}


#: Characters whose run-policy net has already been warmed this session.
_WARMED: set[str] = set()


def warm_policy_async(policies: Any, character: str | None) -> None:
    """Load the run-policy net for ``character`` on a background thread.

    The first screen decision otherwise pays the whole import-torch +
    load-checkpoint bill inline: 45.3 s on the first card reward of the
    2026-08-05 session, by which time the mod had long since cancelled the
    request and the screen showed nothing at all. Combat advice arrives well
    before the first card reward, so warming on any message with a character
    means the net is resident when a screen finally needs it.
    """
    key = (character or "").strip().lower()
    if not key or key in _WARMED:
        return
    _WARMED.add(key)

    def _load() -> None:
        t0 = time.monotonic()
        policy, error = policies.get(character)
        if policy is None:
            logger.warning("run-policy net unavailable for %s: %s", character, error)
        else:
            logger.info("run-policy net ready for %s (%.1fs)", character, time.monotonic() - t0)

    threading.Thread(target=_load, name=f"warm-{key}", daemon=True).start()


def _unranked(state: dict[str, Any], why: str) -> dict[str, Any]:
    """An answer for a surface no head can rank -- honest, but still usable.

    Returning only a note would leave F7 sitting on the screen forever, which
    is the failure this whole pass is about. So the first option is named as
    the pick and the note says plainly that nothing ranked it: an arbitrary
    choice the player can see is arbitrary beats a silent stall.

    Every answer from here also carries ``hand_back`` (GH #874): with the
    mod's default ``pause_on: "unsupported"`` F7 stops here and gives the
    player the screen, so the first-option pick only plays when they opted
    out of pausing (unattended runs).
    """
    options = state.get("options") or state.get("cards") or state.get("nodes") or []
    if not options:
        return {"note": f"{why} (nothing to pick)", "hand_back": why}
    labels = [
        str(o.get("label") or o.get("name") or o.get("id") or f"option {i}")
        for i, o in enumerate(options)
        if isinstance(o, dict)
    ]
    if not labels:
        return {"note": f"{why} (nothing to pick)"}
    return {
        "note": f"{why} — defaulting to the first option, NOT a recommendation",
        "options": [
            {"label": label, "is_best": i == 0, "index": i, "action": "choose"}
            for i, label in enumerate(labels)
        ],
        "best": {"action": "choose", "index": 0, "label": labels[0]},
        "hand_back": why,
    }


def _build_card_bundle(state: dict[str, Any]) -> Any:
    """Rank card bundles by how much the card-pick head likes their contents.

    There is no bundle head: the champion has never been asked "which of these
    SETS of cards do you want". What it can answer is "which of these cards
    would you pick", so each bundle is offered as a card-reward slate and
    scored by its cards. That is a composition, not a trained surface -- it
    ignores everything a bundle is actually about (synergy between the cards,
    the fact that you take all of them) -- so the caller labels it.

    Returns a ``PpoDecision`` whose actions are the bundles: the obs is the
    flattened card list and each bundle's action picks its FIRST card, which
    is what makes this a heuristic rather than a ranking.
    """
    import realgame_ppo

    bundles = [b for b in (state.get("options") or []) if isinstance(b, dict)]
    if not bundles:
        return None
    run_state = state.get("run_state") or {}

    flat: list[dict[str, Any]] = []
    first_of: list[int] = []
    for bundle in bundles:
        cards = [c for c in (bundle.get("cards") or []) if isinstance(c, dict)]
        if not cards:
            return None
        first_of.append(len(flat))
        flat.extend(cards)

    obs = realgame_ppo._base_obs(state, run_state)
    obs["card_reward_option_ids"] = [realgame_ppo._card_id(c) for c in flat]
    obs["card_reward_option_upgraded"] = [
        bool(c.get("upgrade_level") or c.get("upgraded")) for c in flat
    ]
    actions = [{"action": "pick_card_reward", "idx": i} for i in first_of]
    responses = [
        {"action": "choose", "index": int(b.get("index", i))} for i, b in enumerate(bundles)
    ]
    labels = [
        " + ".join(str(c.get("name") or c.get("id")) for c in (b.get("cards") or []))
        for b in bundles
    ]
    return realgame_ppo.PpoDecision(
        obs, actions, realgame_ppo._base_metrics(state, run_state), responses, labels
    )


def _searched_card_select_advice(
    state: dict[str, Any],
    mirror: Any,
    cfg: RecommendConfig | None,
    sims: int | None,
) -> dict[str, Any] | None:
    """Rank an IN-COMBAT ``card_select`` with the combat MCTS, or give up.

    The mirror owns the carried fight, so it answers this
    (:meth:`AdvisorMirror.answer_card_select`). ``None`` for every reason not
    to answer; the caller then falls to the heuristic rung exactly as before.
    """
    if mirror is None:
        return None
    return mirror.answer_card_select(state, cfg, sims)


def compute_screen_advice(
    state: dict[str, Any],
    advice_for: str,
    policies: Any,
    *,
    session: dict[str, Any] | None = None,
    mirror: Any = None,
    cfg: RecommendConfig | None = None,
    sims: int | None = None,
) -> dict[str, Any]:
    """Rank a card-reward / card-select screen with the run-policy net.

    Reuses :mod:`realgame_ppo`'s decision builders verbatim -- the same
    obs/action encoding the champion was trained and evaluated with -- and
    reports the FULL masked softmax (``NetPolicy.distribution``) instead of
    one sample, so the HUD can show every option's probability.

    The net's per-option probability rides on ``visit_share`` (the field the
    HUD already draws as a bar); there is no per-action Q for these surfaces,
    only the state value V, which goes in the note.

    Args:
        session: Per-run scratch (:func:`realgame_ppo.apply_session_context`).
            The advisor passed nothing here for its whole life, which is why
            its map advice silently degraded on the ~15% of live `map_select`
            messages that arrive with an empty graph. ``None`` keeps the old
            stateless behaviour for one-shot callers and tests.
        mirror: The fight's :class:`AdvisorCombatMirror`, when one is live.
            Only an IN-COMBAT ``card_select`` uses it -- see
            :func:`_searched_card_select_advice`.
        cfg / sims: Search knobs for that path; both required for it to run.
    """
    import ppo_decode
    import realgame_ppo

    state = realgame_ppo.apply_session_context(state, session)
    run_state = state.get("run_state") or {}
    character = character_display_name(run_state.get("character")) or "Ironclad"
    policy, error = policies.get(character)
    if policy is None:
        return {"note": f"no run-policy net: {error}", "hand_back": "no run-policy net is loaded"}

    standin_note = ""
    if advice_for == "card_select":
        # `_build_card_select` keys the deck-mutation surface off the screen's
        # own prompt_key first (TO_UPGRADE/TO_REMOVE/...), which the mod
        # captures via CardSelectContextPatch, so prev_msg_type is only the
        # fallback and None is fine here. It is the one surface the driver's
        # builder table does not carry (it needs that extra argument).
        decision = realgame_ppo._build_card_select(state, None)
    elif advice_for == "card_bundle":
        decision = _build_card_bundle(state)
        standin_note = "bundles scored card-by-card; synergy is not modelled"
    else:
        # Everything else goes through the driver's own builder table, so a
        # surface the PPO driver learns to handle is one the advisor can
        # explain without a second dispatch to keep in sync.
        head, standin_note = _SURFACE_STANDINS.get(advice_for, (advice_for, ""))
        builder = realgame_ppo._PPO_NONCOMBAT_BUILDERS.get(head)
        if builder is None:
            return _unranked(state, f"no trained head for {advice_for}")
        build_state = state if head == advice_for else {**state, "type": head}
        decision = builder(build_state)

    if decision is None or not decision.actions:
        # In-combat card selections (Burning Pact's Exhaust, Gambler's Brew's
        # replace, Dual Wield's copy) land here: no head was ever trained on
        # them, and `_unranked` answers them with the LEFTMOST card, which
        # exhausted a Strike+ over its unupgraded twin in 8 of 8 picks during
        # the ascension climb.
        #
        # The ladder is now three rungs, best first:
        #   1. the combat MCTS, when the fight's mirror is carrying the
        #      choice with its suspended tail (`_searched_card_select_advice`);
        #   2. `realgame_hand_select`, which decides only the one thing the
        #      leftmost default gets wrong -- whether the pick is punished or
        #      rewarded -- and returns None when it cannot tell;
        #   3. `_unranked`'s admitted arbitrary pick.
        # Each rung says in its note which one answered.
        if advice_for == "card_select":
            searched = _searched_card_select_advice(state, mirror, cfg, sims)
            if searched is not None:
                return searched
            ranked = realgame_hand_select.rank(state)
            if ranked is not None:
                return ranked
        return _unranked(state, f"{advice_for}: the net does not cover this surface")

    if len(decision.actions) == 1:
        probs, value = [1.0], None
    else:
        try:
            probs, value = policy.distribution(decision.obs, decision.actions, decision.metrics)
        except Exception as exc:
            # The fixed-shape ONNX backend can exceed its export budget
            # (schema.BudgetExceededError, e.g. a 103-option shop removal --
            # GH #315), and any backend can fail at inference. Advice must
            # always render, so degrade to the unranked shape.
            logger.warning("run-policy inference failed for %s: %s", advice_for, exc)
            return _unranked(state, f"{advice_for}: run-policy inference failed ({exc})")

    def _option_index(response: dict[str, Any], fallback: int) -> int:
        # card_reward answers with a bare `index`; card_select with an
        # `indices` list (realgame_random.pick_card_select's spelling).
        if isinstance(response.get("index"), int):
            return int(response["index"])
        indices = response.get("indices")
        if isinstance(indices, list) and indices and isinstance(indices[0], int):
            return int(indices[0])
        return fallback

    carried = (
        realgame_ppo.shop_removal_carry_index(state, session, decision)
        if advice_for == "card_select"
        else None
    )
    # The shop already chose WHICH card to remove; this screen only executes
    # it. Re-ranking here would answer a different question on a differently
    # trained surface -- see `realgame_ppo.shop_removal_carry_index`. The
    # driver has honoured the shop pick since 2026-07-16; the advisor never
    # did, because it does not route through `pick_noncombat_action_ppo`, so
    # every live shop removal was re-decided (0 carries across
    # `logs/f7-mirror-check18` and `check19`).
    # Decoded the way the run agent decodes (`ppo_decode.select_index`: the A8
    # §8 group-mass rule — pool the mass of equivalent candidates, e.g. one
    # shop removal per deck card, then take the best member). A plain argmax
    # here let a split "remove a starter" preference lose to a single
    # stronger-looking buy that the agent would never pick.
    best_i = carried if carried is not None else ppo_decode.select_index(decision.actions, probs)
    # Map options carry their graph coordinate so the HUD can pin the score to
    # the node itself instead of printing it in a side panel -- a ranked list
    # of coordinates is not something a player can act on while looking at a
    # map that shows no coordinates.
    offered = state.get("nodes") or [] if advice_for == "map_select" else []
    options = []
    for i, p in enumerate(probs):
        response = decision.responses[i] if i < len(decision.responses) else {}
        option = {
            "label": decision.labels[i] if i < len(decision.labels) else "?",
            "visit_share": round(float(p), 4),
            "is_best": i == best_i,
            "index": _option_index(response, i),
            "action": response.get("action", "choose"),
        }
        if i < len(offered):
            try:
                option["col"] = int(offered[i]["col"])
                option["row"] = int(offered[i]["row"])
            except (KeyError, TypeError, ValueError):
                pass
        options.append(option)

    # A single-option screen never consulted the net (see the short-circuit
    # above), so it must not be announced as one -- GH #399.
    note = realgame_ppo.decision_note(decision, probs)
    if value is not None:
        note += f" · V={value:+.2f}"
    best_response = dict(decision.responses[best_i]) if best_i < len(decision.responses) else {}
    best = {
        "action": best_response.get("action", "choose"),
        "index": _option_index(best_response, best_i),
        "label": options[best_i]["label"],
    }
    if carried is not None:
        note += " · target carried from the shop pick, not re-ranked here"
    if session is not None and advice_for == "shop":
        realgame_ppo.stash_shop_removal_target(
            session, state.get("run_state") or {}, decision.actions[best_i]
        )
    if standin_note:
        note += f" · {standin_note}"
    if state.get("surface") == "choose_a_card":
        # The card-pick head is the closest surface the champion was trained
        # on, but it was trained on "add this to my DECK". This card is
        # generated into the current fight and then gone, so its deck value
        # is the wrong question -- say so rather than let the ranking read as
        # a deck verdict.
        note += " · scored as a deck pick; this copy only lasts the fight"

    advice = {"note": note, "options": options, "best": best}
    if advice_for in _HAND_BACK_SURFACES:
        advice["hand_back"] = _HAND_BACK_SURFACES[advice_for]

    if advice_for == "map_select":
        path = _project_map_path(state, policy, best_i, hp_table=_projection_hp_table())
        if path:
            advice["path"] = path
            advice["note"] += _route_note(path)

    return advice


# Defensive bound only. The act DAG ends itself at the final row (~15 rows),
# so this never binds on a real map; it exists so a malformed graph cannot
# wedge the walk. It used to be 8 -- a fig leaf over a projection that froze
# HP and drifted without bound. Now that HP is walked forward the route can
# run the whole act, which is what the HUD needs to show.
MAX_PROJECTED_HOPS = 32


def _projection_hp_table() -> dict[str, Any] | None:
    """The run-AI's expected-HP-cost tables, straight from the engine.

    `data/runai/heuristic_v.toml` is `include_str!`'d into the Rust binary, so
    asking `sts2_rs` for it keeps the HUD's route and the planner's own
    projection on one set of numbers. A Python-side copy would be a second
    source of truth that nothing gates.

    `None` when the engine cannot answer (older wheel, extension missing) --
    the route then draws frozen, as it did before, rather than not at all.
    """
    try:
        import sts2_rs

        return dict(sts2_rs.projection_hp_table())
    except Exception as exc:  # never let a drawing cost the advice
        logger.debug("projection tables unavailable; route will not decay HP: %s", exc)
        return None


def _route_note(path: list[dict[str, Any]]) -> str:
    """The one-line route caption under the map advice."""
    note = f" · route {len(path)} floors"
    arrival = path[-1].get("hp") if path else None
    if arrival is not None:
        # Naming the frozen axes is the point: a player who knows only HP moves
        # can discount the late hops themselves.
        note += f" → {arrival} HP (deck/gold/relics frozen)"
    return note


def _project_hp(
    hp: float,
    max_hp: float,
    room: str,
    act: int,
    table: dict[str, Any] | None,
) -> float:
    """HP after `room` resolves, from the run-AI's expected-cost tables."""
    if not table:
        return hp
    slot = min(max(act - 1, 0), 2)
    # The bridge sends the C# `MapPointType` enum name (``RestSite``); the
    # training-side obs uses the snake form (``rest_site``). Fold both.
    kind = str(room).lower().replace("_", "")
    if kind == "monster":
        hp -= float(table["monster_hp_loss"][slot])
    elif kind == "elite":
        hp -= float(table["elite_hp_loss"][slot])
    elif kind == "ancient":
        # Event rooms are a chance node; the table's flat prior is the
        # expectation over the pool, matching `projection.rs`.
        hp -= float(table["event_hp_loss"])
    elif kind == "unknown":
        hp -= float(table["unknown_hp_loss"])
    elif kind == "restsite":
        # The engine's own floor(max_hp * 0.3). Projecting the campfire as a
        # smith instead would leave HP flat, and the smith case is the one
        # the player can see for themselves.
        hp += math.floor(max_hp * float(table["rest_heal_frac"]))
    # A drawing, not a death sentence: the planner keeps the negative deficit
    # to score dead paths as dominated, but a HUD line has nothing to do with
    # a projected -40, and the net has never seen an HP below 1.
    if max_hp > 0:
        hp = min(hp, max_hp)
    return max(hp, 1.0)


def _project_map_path(
    state: dict[str, Any],
    policy: Any,
    best_index: int,
    hp_table: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Walk the net forward from its chosen node to sketch the route.

    The map surface only ever ranks the CHILDREN of where you stand, so a
    single query answers "which node next", never "which way through the
    act" -- and a bare ``(4,1)`` on the HUD tells a player nothing about
    where that leads. This re-queries the same policy hop by hop, moving
    ``map_coord`` along the graph the bridge already sent us.

    HP is walked forward through :func:`_project_hp` between hops, so hop *n*
    is answered at the HP the player would really arrive with. That is not a
    nicety: the champion rest-seeks 95% of the time below half HP, so a walk
    that freezes HP at its live value is structurally blind to campfires and
    systematically under-draws them (the projection used to stop at 8 hops
    precisely to hide how far that drifted).

    It is still a PROJECTION, not a plan. Deck, gold, relics and potions stay
    frozen, and the HP deltas are the run-AI's flat per-act expectations, not
    this deck against that encounter -- so the route is plausible, not
    accurate. :func:`_route_note` says so on the HUD.
    """
    import realgame_ppo

    m = state.get("map") or {}
    graph = m.get("nodes") or []
    offered = state.get("nodes") or []
    if not graph or not (0 <= best_index < len(offered)):
        return []

    by_coord: dict[tuple[int, int], dict[str, Any]] = {}
    for n in graph:
        try:
            by_coord[int(n["col"]), int(n["row"])] = n
        except (KeyError, TypeError, ValueError):
            continue

    run_state = state.get("run_state") or {}
    base_obs = realgame_ppo._base_obs(state, run_state)
    base_metrics = realgame_ppo._base_metrics(state, run_state)
    base_floor = int(base_metrics.get("total_floor", 0) or 0)

    chosen = offered[best_index]
    try:
        cur_coord = (int(chosen["col"]), int(chosen["row"]))
    except (KeyError, TypeError, ValueError):
        return []

    hp = float(base_metrics.get("hp", 0) or 0)
    max_hp = float(base_metrics.get("max_hp", 0) or 0)
    act = int(base_metrics.get("act_index", 1) or 1)

    path: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for hop in range(MAX_PROJECTED_HOPS):
        node = by_coord.get(cur_coord)
        if node is None or cur_coord in seen:
            break
        seen.add(cur_coord)
        room = node.get("type", "?")
        step = {
            "col": cur_coord[0],
            "row": cur_coord[1],
            "type": room,
            "hop": hop,
        }
        # Only claim an arrival HP when there is a cost model behind it. With
        # no table the walk is frozen, and reporting the live HP as "what you
        # arrive with" would be a worse lie than saying nothing.
        if hp_table:
            step["hp"] = round(hp)
        path.append(step)
        hp = _project_hp(hp, max_hp, room, act, hp_table)

        children = []
        for c in node.get("children") or []:
            try:
                coord = (int(c["col"]), int(c["row"]))
            except (KeyError, TypeError, ValueError):
                continue
            if coord in by_coord:
                children.append(coord)
        if not children:
            break
        if len(children) == 1:
            cur_coord = children[0]
            continue

        obs = dict(base_obs)
        obs["map_coord"] = cur_coord
        metrics = dict(base_metrics)
        metrics["total_floor"] = base_floor + hop + 1
        metrics["act_floor"] = cur_coord[1] + 1
        metrics["hp"] = round(hp)
        actions = [{"action": "move", "col": col} for col, _row in children]
        try:
            probs, _ = policy.distribution(obs, actions, metrics)
        except Exception as exc:  # a projection must never break the advice
            logger.debug("map projection stopped at hop %d: %s", hop, exc)
            break
        cur_coord = children[max(range(len(probs)), key=probs.__getitem__)]

    return path


class LastCombat:
    """The most recent combat decision's native search state, for companion
    searches (the workbench's on-demand SearchTree explorer). Thread-safe:
    written by the serve loop, read by the control plane."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state_id: Any = None
        self._cs: Any = None
        self._cfg: RecommendConfig | None = None

    def put(self, state_id: Any, cs: Any, cfg: RecommendConfig) -> None:
        with self._lock:
            self._state_id, self._cs, self._cfg = state_id, cs, cfg

    def clear(self) -> None:
        with self._lock:
            self._state_id = self._cs = self._cfg = None

    def get(self) -> tuple[Any, Any, RecommendConfig | None]:
        with self._lock:
            return self._state_id, self._cs, self._cfg


#: Singleton written by ``_handle_advice_request``.
LAST_COMBAT = LastCombat()


def companion_search_tree(cs: Any, cfg: RecommendConfig, sims: int) -> dict[str, Any]:
    """A single-tree search over ``cs`` in the workbench's ``search_tree``
    payload shape, pruned to keep log records bounded.

    This is a COMPANION search (fresh single tree), not the advice's own
    beliefdraw tree — same caveat as the workbench's live explorer.
    """
    import sts2_rs

    tree = sts2_rs.mcts_search_tree(
        cs,
        sims=sims,
        c_puct=cfg.resolved_c_puct(),
        depth=cfg.depth,
        seed=cfg.seed,
        rollout_policy=cfg.rollout_policy,
    )
    return prune_tree(tree)


def prune_tree(tree: dict[str, Any], min_visits: int = 2, max_nodes: int = 2000) -> dict[str, Any]:
    """Drop leaf noise so a logged tree stays a record, not a payload bomb.

    Keeps the root and any node with ``visits >= min_visits``, then caps to
    the ``max_nodes`` most-visited (parents of kept nodes are re-added so
    the tree stays connected).
    """
    nodes = list(tree.get("nodes") or [])
    if not nodes:
        return tree
    root = tree.get("root")
    by_id = {n.get("id"): n for n in nodes}
    kept = [n for n in nodes if n.get("id") == root or (n.get("visits") or 0) >= min_visits]
    kept.sort(key=lambda n: (n.get("id") != root, -(n.get("visits") or 0)))
    kept = kept[:max_nodes]
    ids = {n.get("id") for n in kept}
    # re-add ancestors so every kept node reaches the root
    for n in list(kept):
        cur = n
        while True:
            pid = cur.get("parent")
            if pid is None or pid in ids:
                break
            parent = by_id.get(pid)
            if parent is None:
                break
            kept.append(parent)
            ids.add(pid)
            cur = parent
    out = dict(tree)
    out["nodes"] = kept
    out["pruned_from"] = len(nodes)
    return out


class AdvisorLog:
    """Append-only JSONL event log (advice + human actions), one file/session.

    TODO(phase 9): switch to the workbench ``GameLogger`` schema
    ({step, ts, type: meta|state|mcts|action, data}) so ``/api/runs`` can
    load real-game advisor sessions directly.
    """

    def __init__(self, log_dir: Path) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        self.path = log_dir / f"advisor-{stamp}.jsonl"
        self._step = 0

    def write(self, kind: str, data: dict[str, Any]) -> None:
        try:
            record = {"step": self._step, "ts": time.time(), "type": kind, "data": data}
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
            self._step += 1
        except Exception as e:
            logger.warning("advisor log write failed: %s", e)


WORKBENCH_URL = "http://127.0.0.1:8077"


def ensure_workbench_running() -> None:
    """Dev convenience for the mod's "Open Workbench" button: if nothing is
    serving :8077, spawn ``python -m workbench`` as a child process.

    Best-effort by design — the advisor must come up (and advise) whether or
    not the workbench does. The child shares our process group, so the mod's
    job object tears it down with the game like everything else we spawn.
    """
    import subprocess
    import sys
    import urllib.request

    try:
        with urllib.request.urlopen(f"{WORKBENCH_URL}/api/health", timeout=1.5) as resp:
            if resp.status == 200:
                logger.info("workbench already running at %s", WORKBENCH_URL)
                return
    except Exception:
        pass

    root = Path(__file__).resolve().parent.parent  # sts2-pycli/
    env = dict(os.environ)
    env["STS2_RELOAD"] = "0"
    # Serve the built SPA from :8077 so the mod's one URL is the whole app.
    # Needs a `npm run build` in web/ at least once; without web/dist the
    # mount is skipped and the page 404s (the API still works).
    env["STS2_SERVE_SPA"] = "1"
    if not (root / "web" / "dist").is_dir():
        logger.warning(
            "web/dist missing — the workbench will serve API only. "
            "Run `npm run build` in sts2-pycli/web once."
        )
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(root / "python"), env.get("PYTHONPATH")) if p
    )
    try:
        subprocess.Popen(
            [sys.executable, "-m", "workbench"],
            cwd=str(root),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.info("spawned workbench (%s) at %s", sys.executable, WORKBENCH_URL)
    except Exception as exc:
        logger.warning("could not spawn workbench: %s", exc)


def build_coach(
    client: Any,
    wlog: Any,
    args: Any,
    policies: Any = None,
    session: dict[str, Any] | None = None,
    executor: Any = None,
    tree: Any = None,
) -> CoachGrader:
    """The connection's live coach, wired to this client and run log.

    Every record goes to the run log; the ``coach`` message goes to the mod
    only while its last ``advice_request`` advertised ``capabilities.coach``
    (the grader gates that). ``request_id`` / ``state_id`` are explicit
    ``None``: the bridge client would otherwise echo the last request's, and a
    coach record answers no request. Records name the run log and the replay
    step of their decision (``log_name`` / ``log_idx``, from ``tree``) so the
    mod can open the workbench right there.
    """

    def _locate(node_id: Any) -> tuple[str | None, int | None]:
        log_idx = tree.log_idx(node_id) if tree is not None else None
        return wlog.log_name, log_idx

    def _send(record: dict[str, Any]) -> None:
        client.send_action(
            {"action": "coach", "request_id": None, "state_id": None, "coach": record}
        )

    def _log(record: dict[str, Any]) -> None:
        logger.info(
            "coach %s %s: %s",
            record.get("kind"),
            record.get("id") or record.get("ids"),
            record.get("verdict", ""),
        )
        wlog.log_coach(record)

    def _map_path(msg: dict[str, Any], chosen: int) -> list[dict[str, Any]] | None:
        if policies is None:
            return None
        import realgame_ppo

        state = realgame_ppo.apply_session_context({**msg, "type": "map_select"}, session)
        character = (
            character_display_name((state.get("run_state") or {}).get("character")) or "Ironclad"
        )
        policy, _error = policies.get(character)
        if policy is None:
            return None
        return _project_map_path(state, policy, chosen, hp_table=_projection_hp_table())

    return CoachGrader(
        send=_send,
        log=_log,
        executor=executor if executor is not None else GraderThread(),
        catchup_sims=int(getattr(args, "coach_catchup_sims", 2000)),
        budget_s=float(getattr(args, "coach_budget_s", 6.0)),
        map_path=_map_path,
        locate=_locate,
    )


def serve(args: argparse.Namespace) -> None:
    cfg = RecommendConfig(
        sims=args.sims,
        threads=args.threads,
        hidden_info=HONEST_SEARCH["hidden_info"],
        c_puct=HONEST_SEARCH["c_puct"],
        draw_widening_c=HONEST_SEARCH["draw_widening_c"],
    )
    log = AdvisorLog(Path(args.log_dir))
    logger.info("advisor log: %s", log.path)
    wlog = WorkbenchRunLogger(args.workbench_logs)
    if args.workbench == "auto":
        ensure_workbench_running()

    # Rewind lane: the live branch tree + the localhost control plane the
    # workbench's /api/bridge/* proxies talk to (D1: this process owns the
    # one bridge socket, so it is the single gateway).
    tree = LiveRunTree()
    control = ControlPlane(tree)

    def _control_search_tree(sims: Any) -> dict[str, Any]:
        state_id, cs, ccfg = LAST_COMBAT.get()
        if cs is None or ccfg is None:
            raise LookupError("no combat decision cached (not in a fight, or no advice yet)")
        tree_payload = companion_search_tree(cs, ccfg, int(sims or 2000))
        tree_payload["state_id"] = state_id
        return tree_payload

    control.search_tree_fn = _control_search_tree
    site = None
    if args.workbench_dir and StaticSite.is_static_build(Path(args.workbench_dir)):
        site = StaticSite(
            Path(args.workbench_dir), ppo_dir=runtime_paths.data_root() / "data" / "ppo"
        )
        logger.info("hosting the browser workbench from %s", args.workbench_dir)
    control.start(args.control_port, allow_origins=args.control_allow_origin or (), site=site)

    # Screen decisions (card reward / card select) are answered by the same
    # run-policy champion the PPO driver uses; loaded lazily per character on
    # the first screen that needs it, and never fatal if absent.
    import realgame_ppo

    # strict=False: unlike a sweep, a live coached run has no measurement to
    # invalidate, and advice must render even with no net at all (it degrades
    # to the unranked shape, saying so).
    policies = realgame_ppo.PpoPolicyCache(getattr(args, "model", None), strict=False)

    # The two pieces of carried context the advisor used to do without. Both
    # outlive a reconnect on purpose: dropping the connection does not rewind
    # the run, and the mirror re-seeds itself on the next combat board anyway.
    mirror = AdvisorMirror(cfg, enabled=not getattr(args, "no_mirror", False))
    session: dict[str, Any] = {}

    while True:
        # A human is allowed to stare at a screen. The default 60 s read
        # timeout is a stall detector for the autopilot lane, where silence
        # really does mean a wedged run; here it tore the connection down
        # every time the player thought for a minute (seven times in the
        # 2026-08-05 session), and any advice_request sent during the gap was
        # simply lost -- the mod saw no answer and the HUD stayed blank.
        client = RealGameBridgeClient(host=args.host, port=args.port, timeout=args.idle_timeout)
        coach: CoachGrader | None = None
        try:
            client.connect()
            logger.info("connected to SpireMentor at %s:%s", args.host, args.port)
            log.write("meta", {"event": "connected", "sims": args.sims, **HONEST_SEARCH})
            control.set_client(client)
            coach = build_coach(client, wlog, args, policies, session, tree=tree)
            _serve_connected(
                client, cfg, args, log, wlog, policies, tree, control, mirror, session, coach
            )
        except (ConnectionError, TimeoutError) as e:
            logger.warning("connection lost/failed: %s; retrying in 3s", e)
            time.sleep(3)
        finally:
            if coach is not None:
                coach.executor.stop()
                coach = None
            control.set_client(None)
            client.disconnect()


def _serve_connected(
    client: RealGameBridgeClient,
    cfg: RecommendConfig,
    args: argparse.Namespace,
    log: AdvisorLog,
    wlog: WorkbenchRunLogger,
    policies: Any,
    tree: LiveRunTree,
    control: ControlPlane,
    mirror: AdvisorMirror,
    session: dict[str, Any],
    coach: CoachGrader | None = None,
) -> None:
    while True:
        msg = client.receive_state()
        mtype = msg.get("type")
        warm_policy_async(policies, (msg.get("run_state") or {}).get("character"))
        control.note_message(msg)

        if mtype == "advice_request":
            _handle_advice_request(
                client, cfg, args, log, wlog, msg, policies, tree, mirror, session, coach
            )
        elif mtype == "human_action":
            # The position the action was taken FROM, for the coach's catch-up
            # search -- it must be cloned before the mirror follows the move.
            pre_state = None
            if coach is not None and mirror.live is not None:
                try:
                    pre_state = mirror.live.state.clone()
                except Exception as exc:
                    logger.debug("coach: cannot clone the mirror: %s", exc)
            # Follow the world BEFORE logging: the mirror is only a mirror
            # while it is advanced through what actually happened.
            mirror.on_world_action(msg)
            logger.info("human: %s", msg.get("detail") or msg.get("action"))
            log.write("action", msg)
            wlog.log_human_action(msg)
            node_id = tree.on_human_action(msg)
            if coach is not None:
                coach.on_human_action(msg, pre_state, node_id)
        elif mtype in BRIDGE_MESSAGE_TYPES:
            # Rewind lane (snapshots / snapshot_saved / snapshot_restored /
            # error): the control plane routes them, including into the tree.
            logger.debug("%s: %s", mtype, msg)
            if mtype in ("snapshot_restored", "combat_undone"):
                # Both carry a fresh state_id precisely because everything
                # older is stale (PROTOCOL.md section 7), and the mod
                # broadcasts them for ANY rewind including the in-game
                # toolbar's. A carried combat state is exactly that kind of
                # stale -- and `combat_undone` keeps the fight running while
                # moving the board backwards, which is the case that would
                # corrupt it silently.
                mirror.on_rewind(msg)
            control.on_bridge_message(msg)
            if coach is not None and mtype in ("snapshot_restored", "combat_undone"):
                # AFTER the tree moved its active pointer: grades for nodes
                # the rewind abandoned are withdrawn from the mod and the log.
                coach.on_rewind(msg, tree.active_path())
        elif mtype == "pong":
            logger.debug("%s: %s", mtype, msg)
        elif mtype == "combat_ended":
            # The fight's end (PROTOCOL.md v5.1): only the coach's recap
            # needs it. The mirror catches up from the next board and the
            # tree / run log record decisions, not fight boundaries.
            logger.info("combat_ended: %s", {k: msg.get(k) for k in ("victory", "hp", "round")})
            log.write("meta", {"event": mtype, **msg})
            if coach is not None:
                coach.on_combat_ended(msg)
        elif mtype in ("game_over", "run_complete"):
            logger.info("%s: %s", mtype, {k: msg.get(k) for k in ("victory", "floor", "act")})
            log.write("meta", {"event": mtype, **msg})
            wlog.log_run_end(msg)
            tree.on_run_end(msg)
            LAST_COMBAT.clear()
            mirror.drop("run over")
            logger.info("advisor %s", mirror.note())
            session.clear()
            if coach is not None:
                if is_death(msg):
                    # The autopsy is built behind the grading still in flight
                    # and appended to the run log this branch just closed.
                    coach.on_game_over(msg)
                else:
                    coach.on_run_end()
        else:
            # Autopilot-style decision messages (combat_action, map_select,
            # run_config, ...) mean this advisor is connected to a session
            # that expects a DRIVER. Never answer them — log and ignore.
            logger.warning("ignoring non-advisor message type=%r (advisor never drives)", mtype)

        # AFTER the handler, so the file this message just opened (or rotated
        # to) is the one published. Every message, not only advice requests:
        # the workbench's Follow mode asks /status for this path and silently
        # guesses when it is absent, so publishing it once the first advice
        # lands left a window where Follow tailed some other file -- and a
        # human_action, which opens the log just as well, never closed it.
        publish_log_path(control, wlog)


def is_death(msg: dict[str, Any]) -> bool:
    """A ``game_over`` the player lost (PROTOCOL.md section 3: ``result``).

    Only a death gets the coach's autopsy; a victory or a ``run_complete``
    (which also covers an abandoned run, ``terminated``) does not.
    """
    if msg.get("type") != "game_over":
        return False
    return msg.get("result") == "defeat" or msg.get("victory") is False


def publish_log_path(control: Any, wlog: Any) -> None:
    """Publish the run log the control plane serves at ``GET /log``.

    Only an OPEN log replaces the published path: once ``game_over`` closes
    the file the last one stays served until the next run opens its own, so
    the autopsy's "open in workbench" links (which arrive just then) still
    reach their file. Follow mode reloads on a path change, so a finished
    file left published is harmless.
    """
    if wlog.path is not None:
        control.set_log_path(wlog.path)


def _cache_for_coach(
    coach: CoachGrader,
    msg: dict[str, Any],
    advice_for: str,
    state: dict[str, Any],
    advice: dict[str, Any],
    rows: list[dict[str, Any]] | None,
    cs: Any,
    call_cfg: Any,
) -> None:
    """Keep this request's raw result for grading the move made from it."""
    try:
        if advice_for == "combat_action":
            if rows is None:
                return  # the search failed: nothing to grade against
            entry = coach.cache.put(
                msg.get("request_id"),
                msg.get("state_id"),
                kind="combat",
                rows=rows,
                cs=cs,
                cfg=call_cfg,
                advice=advice,
                pv_labels=list(advice.get("pv") or []),
                msg=state,
            )
        elif advice_for in SCREEN_ADVICE_FOR and advice.get("options"):
            entry = coach.cache.put(
                msg.get("request_id"),
                msg.get("state_id"),
                kind="screen",
                options=list(advice.get("options") or []),
                advice=advice,
                msg=state,
            )
        else:
            return
        coach.on_advice(entry)
    except Exception as exc:  # bookkeeping must never cost the advice loop
        logger.warning("coach: caching advice failed: %s", exc)


def refusal_advice(exc: UnsupportedGameVersionError) -> dict[str, Any]:
    """The advice for a board from a game build this advisor cannot simulate.

    No ranking and no ``best``: there is no engine for that build, and a search
    on the wrong one would rank moves by rules the game is not using. The
    ``note`` is the player-facing explanation the mod shows in the HUD;
    ``unsupported_game_version`` names the build for tooling.
    """
    return {"note": str(exc), "unsupported_game_version": exc.raw}


def note_game_version(session: dict[str, Any] | None, game_version: str, mirror: Any) -> None:
    """Track the game build across requests; a change invalidates carried state.

    The mirror's carried combat state and the cached search board were built
    for the previous build's engine, so neither survives a build change.
    """
    if session is None:
        return
    previous = session.get("game_version")
    if previous == game_version:
        return
    if previous is None:
        logger.info("game build: %s", game_version)
    else:
        logger.warning(
            "game build changed %s -> %s; dropping carried state", previous, game_version
        )
        LAST_COMBAT.clear()
        if mirror is not None:
            mirror.drop(f"game build changed {previous} -> {game_version}")
    session["game_version"] = game_version


def _handle_advice_request(
    client: RealGameBridgeClient,
    cfg: RecommendConfig,
    args: argparse.Namespace,
    log: AdvisorLog,
    wlog: WorkbenchRunLogger,
    msg: dict[str, Any],
    policies: Any,
    tree: LiveRunTree | None = None,
    mirror: Any = None,
    session: dict[str, Any] | None = None,
    coach: CoachGrader | None = None,
) -> None:
    request_id = msg.get("request_id")
    state_id = msg.get("state_id")
    advice_for = msg.get("advice_for") or "combat_action"
    if coach is not None:
        coach.on_advice_request(msg)
    rows: list[dict[str, Any]] | None = None
    cs: Any = None
    call_cfg: Any = None
    # State first, advice second: the mcts record attaches to the most recent
    # state line (TimelineStore contract), and the state must survive even if
    # the search below throws.
    logged_step = wlog.log_decision(msg, advice_for)
    if logged_step is not None and tree is not None:
        tree.on_decision(msg, advice_for, log_idx=logged_step)
    search_tree_log: dict[str, Any] | None = None
    state: dict[str, Any] = msg
    t0 = time.monotonic()
    try:
        # The build the mod reports (absent = v0.107.1). The combat engine is
        # built for it downstream (`to_headless_combat_play` -> `$.game_version`
        # -> `state_from_dict`); an unknown build is refused before any search.
        note_game_version(session, envelope_game_version(msg), mirror)
        state = dict(msg)
        state["type"] = advice_for
        is_boss = (state.get("room_type") or "") == "Boss"
        sims = args.boss_sims if is_boss else args.sims
        fight_cfg = fight_search_cfg(cfg, state)
        if advice_for in SCREEN_ADVICE_FOR:
            advice = compute_screen_advice(
                state,
                advice_for,
                policies,
                session=session,
                mirror=mirror,
                cfg=fight_cfg,
                sims=sims,
            )
        elif advice_for == "combat_action":
            advice, cs, call_cfg, rows = compute_combat_advice(state, fight_cfg, sims, mirror)
            advice = apply_ally_wait(state, advice)
            LAST_COMBAT.put(state_id, cs, call_cfg)
            if args.log_search_tree > 0 and logged_step:
                try:
                    search_tree_log = companion_search_tree(cs, call_cfg, args.log_search_tree)
                except Exception as exc:
                    logger.warning("search-tree logging failed: %s", exc)
        else:
            raise ValueError(f"unsupported advice_for={advice_for!r}")
    except UnsupportedGameVersionError as e:
        logger.error("advice_request refused: %s", e)
        advice = refusal_advice(e)
    except Exception as e:
        # With the traceback: this handler renders "advisor error" into the
        # HUD and moves on, so without one a live failure leaves only a
        # message and no way to find it. `object of type 'NoneType' has no
        # len()` cost a whole debugging round on 2026-08-24 for exactly that
        # reason.
        logger.error("advice_request failed: %s", e, exc_info=True)
        advice = {"note": f"advisor error: {e}", "hand_back": "the advisor hit an error"}

    elapsed = time.monotonic() - t0
    reply = {
        "action": "advice",
        "request_id": request_id,
        "state_id": state_id,
        "advice": advice,
        "capabilities": dict(ADVISOR_CAPABILITIES),
    }
    try:
        client.send_action(reply)
    except ConnectionError:
        raise
    if coach is not None:
        # AFTER the reply: the coach's bookkeeping never delays advice.
        _cache_for_coach(coach, msg, advice_for, state, advice, rows, cs, call_cfg)
    log.write(
        "mcts",
        {
            "request_id": request_id,
            "state_id": state_id,
            "elapsed_s": round(elapsed, 2),
            "advice": advice,
        },
    )
    if logged_step:
        wlog.log_advice(msg, advice, elapsed, tree=search_tree_log)
    best_label = (advice.get("best") or {}).get("label")
    # The note is where an answer admits it is not a ranking ("defaulting to
    # the first option, NOT a recommendation", "heuristic pick: ..."). It was
    # reaching the HUD and nothing else, so an arbitrary pick read exactly like
    # a trained one in every log we keep -- which is how the leftmost-card
    # selection bug survived a whole climb unnoticed. Log it.
    note = advice.get("note")
    logger.info(
        "advice #%s: best=%s p_win=%s (%.1fs)%s",
        request_id,
        best_label,
        advice.get("p_win"),
        elapsed,
        f" [{note}]" if note else "",
    )


#: A synthetic `map_select` decision for the packaged-exe smoke check. Small,
#: character-agnostic (unknown ids land in the encoder's OOV slots) and enough
#: to drive one full forward pass.
_SMOKE_STATE: dict[str, Any] = {
    "type": "map_select",
    "floor": 5,
    "act": 1,
    "run_state": {
        "character": "Ironclad",
        "hp": 60,
        "max_hp": 75,
        "gold": 120,
        "ascension": 0,
        "master_deck": [{"id": "STRIKE_IRONCLAD"}, {"id": "DEFEND_IRONCLAD"}],
        "relics": [{"id": "BURNING_BLOOD"}],
        "potions": [],
    },
    "nodes": [
        {"index": 0, "type": "Monster", "row": 6, "col": 1},
        {"index": 1, "type": "Shop", "row": 6, "col": 3},
    ],
    "map": {
        "row_count": 10,
        "col_count": 6,
        "current": {"col": 2, "row": 5},
        "boss": {"col": 2, "row": 15},
        "visited": [],
        "nodes": [],
    },
}


def _smoke() -> int:
    """Deterministic self-check of the packaged exe (build gate, GH #380).

    Exercises every PyInstaller landmine without a game or a bridge: the
    ``sts2_rs`` .pyd, the bundled localization datas, EVERY bundled ONNX
    champion through the full ``load_ppo_policy`` resolver, one real forward
    pass per champion, and a socket bind. Prints ``SMOKE OK`` and exits 0
    only if all four pass.

    The forward pass is the point of the third gate, not the load:
    constructing the policy only parses the graph, while ``ppo.vocab``'s
    registry build (``sts2_rs.list_registry_names``), the featurizer and the
    forward pass are first touched at the first real decision -- so a
    bundling or ABI regression in any of them used to ship green and surface
    as "run-policy inference failed" on the player's first card reward.
    Every champion is loaded because the manifest resolves per character,
    and four of the five were never touched.

    The type assertion is load-bearing, not decoration: the bundle ships
    neither numpy nor onnxruntime, so a resolver that fell back to
    :class:`ppo.onnx_policy.OnnxPolicy` would not merely be slower -- it
    would be unimportable, and the advisor's never-crash contract would turn
    that into a run played at random rather than a crash anyone notices.
    """
    import socket

    ok = True

    try:
        import sts2_rs

        print(f"sts2_rs OK (src_hash {sts2_rs.src_hash()})")
    except Exception as e:
        ok = False
        print(f"sts2_rs FAIL: {e}")

    try:
        from sts2_hint import canonical_maps

        n = len(canonical_maps._CARD_LOC)
        if n == 0:
            raise RuntimeError(f"no card titles loaded from {canonical_maps._LOCALIZATION_DIR}")
        print(f"localization OK ({n} card titles)")
    except Exception as e:
        ok = False
        print(f"localization FAIL: {e}")

    try:
        import champions
        import realgame_ppo
        from ppo.rust_policy import RustPolicy

        decision = realgame_ppo._build_map_select(_SMOKE_STATE)
        if decision is None:
            raise RuntimeError("_build_map_select returned None for the smoke state")
        # The manifest must have made it into the bundle -- without it the
        # frozen exe cannot name a champion at all.
        for character in champions.characters():
            policy, error = realgame_ppo.load_ppo_policy(None, character)
            if not isinstance(policy, RustPolicy):
                raise RuntimeError(
                    f"{character}: expected RustPolicy, got {type(policy).__name__}: {error}"
                )
            probs, value = policy.distribution(decision.obs, decision.actions, decision.metrics)
            if len(probs) != len(decision.actions) or abs(sum(probs) - 1.0) > 1e-6:
                raise RuntimeError(f"{character}: bad distribution {probs} (V={value})")
            print(
                f"run policy OK: {character} (T_MAX={policy.t_max}, A_MAX={policy.a_max}, "
                f"p={[round(p, 3) for p in probs]}, V={value:.3f})"
            )
    except Exception as e:
        ok = False
        print(f"run policy FAIL: {e}")

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            print(f"socket OK (ephemeral port {s.getsockname()[1]})")
    except Exception as e:
        ok = False
        print(f"socket FAIL: {e}")

    print("SMOKE OK" if ok else "SMOKE FAIL")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    # The Rust run agent's search, so a win rate measured in Rust is the one
    # the mod plays: ONE tree (`MctsCombatPolicy` never runs root-parallel)
    # at 4x the `eval-cheap` budget — 4000 sims per decision, 16000 in boss
    # fights, elites on the common budget as in `CombatBudget`. 4x is the
    # strongest budget measured on whole runs (A10 17.0% vs 10.8% at
    # 1000/4000, A9 45.2% vs 40.2%; 16x adds only +1.5pp at 4.2x the time —
    # docs/experiments/2026-10-20-combat-pro-ladder-findings.md, "Full-run
    # check"). Cost on one 2.8 GHz core: ~0.7 s per decision, ~2.3 s per boss
    # decision.
    #
    # The advisor used to default to 8000/8000 split across up to 8
    # root-parallel trees (2026-08-10, for latency). That made the advice
    # depend on the player's core count, merged its trees by a different pick
    # rule, and dropped the single-tree lost-position override; no run-level
    # win rate was ever measured for it.
    parser.add_argument("--sims", type=int, default=4000)
    parser.add_argument("--boss-sims", type=int, default=16000)
    # Root-parallel workers (sts2_rs par_search: sims split across independent
    # trees). 1 = the run agent's single tree. More trees trade decision
    # fidelity to the measured agent for wall-clock; only for a box too slow
    # for the single-tree budget.
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--log-dir",
        # Frozen exe: the CWD is the install dir (Program Files on a default
        # Steam install) -- default the JSONL logs to a user-writable spot.
        # SimAdvisorProcess passes --log-dir explicitly; this covers manual
        # launches. Dev default is unchanged (CWD-relative, set by the .cmd).
        default=str(runtime_paths.app_data_dir() / "logs" / "realgame_advisor")
        if runtime_paths.is_frozen()
        else "logs/realgame_advisor",
    )
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=3600.0,
        help=(
            "Seconds of socket silence before the advisor treats the bridge as "
            "dead. Long by design: silence means a human is thinking, not that "
            "the game is wedged (the autopilot lane has its own stall guard)."
        ),
    )
    parser.add_argument(
        "--workbench-logs",
        # Same split as --log-dir: dev writes where the workbench already
        # reads (`<sts2-pycli>/logs`), the frozen exe writes user-writable.
        default=str(runtime_paths.app_data_dir() / "logs")
        if runtime_paths.is_frozen()
        else str(Path(__file__).resolve().parent.parent / "logs"),
        help="Directory for workbench-schema run logs (GameLogger JSONL; "
        "the workbench's /api/runs reads this). Empty string disables.",
    )
    parser.add_argument(
        "--log-search-tree",
        type=int,
        default=0,
        metavar="SIMS",
        help="Also log a pruned companion MCTS search tree (single-tree, "
        "SIMS sims) with every combat decision's advice record. 0 = off "
        "(the workbench can recompute trees on demand instead).",
    )
    parser.add_argument(
        "--workbench",
        choices=("off", "auto"),
        # The packaged exe ships no workbench/SPA; dev installs have it one
        # `just workbench` away, so auto-start it for the browser button.
        default="off" if runtime_paths.is_frozen() else "auto",
        help="auto: probe the local workbench (:8077) and spawn "
        "`python -m workbench` if it is not running. off: never spawn.",
    )
    parser.add_argument(
        "--control-port",
        type=int,
        default=8079,
        help="Localhost control-plane HTTP port (the workbench's /api/bridge/* "
        "proxy and, with --control-allow-origin, the browser-only workbench).",
    )
    parser.add_argument(
        "--control-allow-origin",
        action="append",
        default=None,
        metavar="ORIGIN",
        help="Page origin allowed to call the control plane from a browser "
        "(repeatable), e.g. https://your-workbench.example.com — the browser-only "
        "workbench's live-game views (GH #709). Off by default: without it no "
        "web page can read the control plane, only the same-machine proxy.",
    )
    parser.add_argument(
        "--workbench-dir",
        default=str(runtime_paths.workbench_dir()),
        help="Browser-only workbench build the control plane hosts at / "
        "(the mod's bundled workbench; dev: web/dist when it is the static "
        "build). Empty string disables.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="PPO run-policy model for screen advice (card reward / card "
        "select). A torch checkpoint extension (.pt/.pth/.ckpt/.bin/.tar) "
        "loads the torch NetPolicy (dev); .onnx loads onnxruntime; anything "
        "else is refused rather than guessed. Default: the committed ONNX "
        "champion for the run's character, resolved by "
        "workbench.ppo_policy.onnx_path.",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Do not carry a per-fight engine mirror. Carrying one lets "
        "in-combat card selections be searched by the combat MCTS instead of "
        "a prompt heuristic, and keeps the per-turn counters the wire does "
        "not carry (GH #370, some of which gate what is legal). The mirror is "
        "only sound because `human_action` carries the machine form of a play "
        "(`card` + `target_index`) so the advisor can follow the fight; on "
        "an older mod build that omits those fields every action is "
        "unfollowable and the mirror drops itself rather than drift. Use this "
        "flag to A/B the mirror or to rule it out of a live problem.",
    )
    parser.add_argument(
        "--coach-catchup-sims",
        type=int,
        default=2000,
        help="Sims for the live coach's catch-up search of a position the "
        "player moved past before its advice arrived (run during the enemy "
        "turn on 2 threads, off the advice path).",
    )
    parser.add_argument(
        "--coach-budget-s",
        type=float,
        default=6.0,
        help="Seconds after an end turn within which the coach must finish "
        "grading that turn; past it, or once the next turn's advice is "
        "requested, the turn stays ungraded and silent (never a late chip).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the packaged-exe self-check (no game needed) and exit.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    # Clamp --threads to the core count: more workers than
    # cores cannot buy wall-clock, and each extra worker still shrinks the
    # per-tree sim budget, so oversubscription is a pure quality loss.
    cores = os.cpu_count() or 1
    if args.threads > cores:
        logger.warning(
            "--threads %d exceeds %d available cores; using %d",
            args.threads,
            cores,
            cores,
        )
        args.threads = cores
    logger.info(
        "combat search: sims=%d boss_sims=%d threads=%d (%d cores, %d sims/tree)",
        args.sims,
        args.boss_sims,
        args.threads,
        cores,
        args.sims // max(1, args.threads),
    )
    if args.smoke:
        return _smoke()
    try:
        serve(args)
    except KeyboardInterrupt:
        logger.info("advisor stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
