"""Personal-build boss and elite checks: how often the current deck wins.

On the map, the advisor simulates the act's boss (both bosses on an A10
final act) and the next Elite the act will give against your deck, relics,
potions and HP, many times with the fight search, and reports the win rate
and the HP you would finish on. Read only: it builds its own fights in the
engine (``sts2_rs.build_scenario_state``) and never touches the game.

It runs on a background thread and is cached per (deck, relics, HP band,
fights), so a map screen never waits for it: the first screen after the deck
or HP changes says it is simulating, the next one shows the result.

Bosses are checked at your HP now and after the campfire before them (the
30% heal, capped); the next Elite at your HP now, since that is what you walk
in with. Fights use the deployed honest search: with full foresight the
simulated fights flattered weak decks (Bygone Effigy at 25/75: 14/24 wins
clairvoyant, 10/24 honest; the real fight was lost from turn 1). Validated on
your logged runs: the bosses you beat scored 20-24/24, the Test Subject loss
0-3/24, the Terror Eel death at 31/67 3/24.
"""

from __future__ import annotations

import concurrent.futures
import logging
import re
import threading
from typing import Any

logger = logging.getLogger(__name__)

FIGHTS = 24
SIMS = 300
REST_HEAL = 0.3

_lock = threading.Lock()
_cache: dict[tuple, dict[str, Any]] = {}
_pending: set[tuple] = set()
_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="bosscheck")


def _card_id(card: dict[str, Any]) -> str:
    cid = str(card.get("id") or "")
    return cid + ("+1" if int(card.get("upgrade_level") or 0) > 0 else "")


def _inputs(state: dict[str, Any], character: str, ascension: int, game_version: str) -> dict[str, Any] | None:
    p = state.get("player") or {}
    boss = state.get("boss") or {}
    bosses = [b for b in (boss.get("id"), boss.get("second_id")) if b]
    deck = [_card_id(c) for c in p.get("deck") or [] if isinstance(c, dict) and c.get("id")]
    if (not bosses and not state.get("next_elite")) or not deck or not p.get("max_hp"):
        return None
    hp, max_hp = int(p.get("hp") or 0), int(p["max_hp"])
    return {
        "character": character,
        "bosses": tuple(bosses),
        "elite": state.get("next_elite"),
        "deck": tuple(sorted(deck)),
        "relics": tuple(sorted(str(r.get("id")) for r in p.get("relics") or [] if r.get("id"))),
        "potions": tuple(str(x.get("id")) for x in p.get("potions") or [] if not x.get("empty") and x.get("id")),
        "hp": hp,
        "hp_rest": min(max_hp, hp + int(REST_HEAL * max_hp)),
        "max_hp": max_hp,
        "ascension": ascension,
        "game_version": game_version,
    }


def _key(inp: dict[str, Any]) -> tuple:
    band = round(10 * inp["hp"] / inp["max_hp"])
    return (inp["character"], inp["bosses"], inp["elite"], inp["deck"], inp["relics"], band, inp["ascension"])


def _build(sts2_rs: Any, inp: dict[str, Any], encounter: str, seed: int) -> Any:
    """The fight, dropping any relic/potion the engine does not know."""
    relics, potions = list(inp["relics"]), list(inp["potions"])
    for _ in range(12):
        try:
            return sts2_rs.build_scenario_state(
                seed=seed, character=inp["character"], deck=list(inp["deck"]), hp=inp["hp"],
                max_hp=inp["max_hp"], relics=relics, encounter=encounter, ascension=inp["ascension"],
                potions=potions, game_version=inp["game_version"],
            )
        except ValueError as e:
            bad = re.findall(r"[A-Z][A-Z0-9_]{2,}", str(e))
            dropped = [b for b in bad if b in relics or b in potions]
            if not dropped:
                raise
            relics = [r for r in relics if r not in dropped]
            potions = [x for x in potions if x not in dropped]
    raise ValueError("could not build the fight")


def _simulate(inp: dict[str, Any]) -> dict[str, Any]:
    """{"bosses": {boss: {"now", "rest"}}, "elite": {"now"} or None}."""
    import sts2_rs

    def batch(encounter: str, hp: int) -> dict[str, Any]:
        wins, hp_left = 0, []
        for seed in range(1, FIGHTS + 1):
            cs = _build(sts2_rs, dict(inp, hp=hp), encounter, seed)
            r = sts2_rs.mcts_play_to_terminal(
                cs, sims=SIMS, seed=seed, hidden_info="beliefdraw", c_puct=2.5, draw_widening_c=2.0
            )
            if r.get("player_won"):
                wins += 1
                hp_left.append(int(r.get("final_hp") or 0))
        return {"wins": wins, "fights": FIGHTS, "hp": sum(hp_left) / len(hp_left) if hp_left else 0, "at": hp}

    out: dict[str, Any] = {"bosses": {}, "elite": None}
    if inp["elite"]:
        out["elite"] = {"id": inp["elite"], "now": batch(inp["elite"], inp["hp"])}
    for boss in inp["bosses"]:
        now = batch(boss, inp["hp"])
        rest = batch(boss, inp["hp_rest"]) if inp["hp_rest"] > inp["hp"] and now["wins"] < FIGHTS else None
        out["bosses"][boss] = {"now": now, "rest": rest}
    return out


def _run(key: tuple, inp: dict[str, Any]) -> None:
    try:
        result = _simulate(inp)
    except Exception as exc:  # never let a bad build break map advice
        logger.warning("boss check failed: %s", exc)
        result = {"error": str(exc)}
    with _lock:
        _cache[key] = {"inputs": inp, "result": result}
        _pending.discard(key)


def _pretty(encounter_id: str) -> str:
    return encounter_id.removesuffix("_BOSS").removesuffix("_ELITE").replace("_", " ").title()


def lines(state: dict[str, Any], character: str, ascension: int, game_version: str) -> list[str]:
    """HUD lines for map advice; starts the simulation when not cached."""
    inp = _inputs(state, character, ascension, game_version)
    if inp is None:
        return []
    key = _key(inp)
    with _lock:
        hit = _cache.get(key)
        if hit is None and key not in _pending:
            _pending.add(key)
            _pool.submit(_run, key, inp)
    if hit is None:
        names = " + ".join(_pretty(b) for b in ([inp["elite"]] if inp["elite"] else []) + list(inp["bosses"]))
        return [f"Elite/boss check: simulating your deck vs {names}... (shows on the next map screen)"]
    result = hit["result"]
    if "error" in result:
        return []

    def pct(r: dict[str, Any]) -> float:
        return r["wins"] / r["fights"]

    def verdict(p: float) -> str:
        return ("comfortable" if p >= 0.8 else "risky" if p >= 0.4
                else "LIKELY TO LOSE: you need more damage/scaling or defense")

    out = []
    elite = result.get("elite")
    if elite:
        now = elite["now"]
        p = pct(now)
        advice = ("you're ready for it" if p >= 0.8 else
                  "risky: only take an elite if the route forces it" if p >= 0.4 else
                  "AVOID elites until you heal or your deck is stronger")
        left = f", ~{now['hp']:.0f} HP left" if now["wins"] else ""
        out.append(f"Elite check vs {_pretty(elite['id'])} (your next elite): at {now['at']} HP you win "
                   f"{now['wins']}/{now['fights']} ({p:.0%}){left}. {advice[0].upper() + advice[1:]}")
    for boss, r in result.get("bosses", {}).items():
        now, rest = r["now"], r["rest"]
        left = f", ~{now['hp']:.0f} HP left" if now["wins"] else ""
        line = f"Boss check vs {_pretty(boss)}: at {now['at']} HP your deck wins {now['wins']}/{now['fights']} ({pct(now):.0%}){left}"
        if rest is not None:
            line += f"; after resting ({rest['at']} HP) {rest['wins']}/{rest['fights']} ({pct(rest):.0%})"
            if pct(rest) - pct(now) >= 0.25:
                line += f". REST before the boss! (then {verdict(pct(rest))})"
            else:
                line += f". {verdict(max(pct(now), pct(rest))).capitalize()}"
        else:
            line += f". {verdict(pct(now)).capitalize()}"
        out.append(line)
    out.append("(fresh simulated fights with your deck, relics and potions; not a prediction of the real fight)")
    return out
