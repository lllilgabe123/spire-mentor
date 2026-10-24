"""Personal-build boss check: how often the current deck beats this act's boss.

On the map, the advisor simulates the act's boss (both bosses on an A10
final act) against your deck, relics, potions and HP, many times with the
fight search, and reports the win rate and the HP you would finish on. Read
only: it builds its own fights in the engine (``sts2_rs.build_scenario_state``)
and never touches the game.

It runs on a background thread and is cached per (deck, relics, HP band,
boss), so a map screen never waits for it: the first screen after the deck
changes says it is simulating, the next one shows the result.

The fights assume you rest at the campfire before the boss (current HP plus
the 30% heal, capped), play them with the clairvoyant search the live advisor
uses, and give each its own seed. They are fresh fights, not the real one:
the boss's own random choices will differ.
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
    if not bosses or not deck or not p.get("max_hp"):
        return None
    hp, max_hp = int(p.get("hp") or 0), int(p["max_hp"])
    return {
        "character": character,
        "bosses": tuple(bosses),
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
    return (inp["character"], inp["bosses"], inp["deck"], inp["relics"], band, inp["ascension"])


def _build(sts2_rs: Any, inp: dict[str, Any], boss: str, seed: int) -> Any:
    """The fight, dropping any relic/potion the engine does not know."""
    relics, potions = list(inp["relics"]), list(inp["potions"])
    for _ in range(12):
        try:
            return sts2_rs.build_scenario_state(
                seed=seed, character=inp["character"], deck=list(inp["deck"]), hp=inp["hp"],
                max_hp=inp["max_hp"], relics=relics, encounter=boss, ascension=inp["ascension"],
                potions=potions, game_version=inp["game_version"],
            )
        except ValueError as e:
            bad = re.findall(r"[A-Z][A-Z0-9_]{2,}", str(e))
            dropped = [b for b in bad if b in relics or b in potions]
            if not dropped:
                raise
            relics = [r for r in relics if r not in dropped]
            potions = [x for x in potions if x not in dropped]
    raise ValueError("could not build the boss fight")


def _simulate(inp: dict[str, Any]) -> dict[str, Any]:
    """{boss: {"now": {...}, "rest": {...} or None}} -- the fights at your HP
    now and, when resting would heal, at your HP after the campfire."""
    import sts2_rs

    def batch(boss: str, hp: int) -> dict[str, Any]:
        wins, hp_left = 0, []
        for seed in range(1, FIGHTS + 1):
            cs = _build(sts2_rs, dict(inp, hp=hp), boss, seed)
            r = sts2_rs.mcts_play_to_terminal(cs, sims=SIMS, seed=seed)
            if r.get("player_won"):
                wins += 1
                hp_left.append(int(r.get("final_hp") or 0))
        return {"wins": wins, "fights": FIGHTS, "hp": sum(hp_left) / len(hp_left) if hp_left else 0, "at": hp}

    out = {}
    for boss in inp["bosses"]:
        now = batch(boss, inp["hp"])
        rest = batch(boss, inp["hp_rest"]) if inp["hp_rest"] > inp["hp"] and now["wins"] < FIGHTS else None
        out[boss] = {"now": now, "rest": rest}
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


def _pretty(boss_id: str) -> str:
    return boss_id.removesuffix("_BOSS").replace("_", " ").title()


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
        names = " + ".join(_pretty(b) for b in inp["bosses"])
        return [f"Boss check: simulating your deck vs {names}... (shows on the next map screen)"]
    result = hit["result"]
    if "error" in result:
        return []
    def pct(r: dict[str, Any]) -> float:
        return r["wins"] / r["fights"]

    def verdict(p: float) -> str:
        return ("comfortable" if p >= 0.8 else "risky" if p >= 0.4
                else "LIKELY TO LOSE: you need more damage/scaling or defense")

    out = []
    for boss, r in result.items():
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
    out.append("(fresh simulated boss fights with your deck, relics and potions; not a prediction of the real fight)")
    return out
