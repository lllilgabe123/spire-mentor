"""Parse SpireCodex combat replays into per-fight records.

A replay (`<hash>.replay.ndjson.gz`, from the SpireCodex Workshop mod) logs
every event of a run: draws, card plays with targets, potions, end turns,
in-combat picks, monster moves, and, from replay_version 6 on v0.111.0, the
run's RNG stream counters at each `combat_start`.

`fights(path)` walks one replay and yields one record per combat attempt:
the start state (deck in master order, relics, belt, hp, roster, rng) and
the human's ordered decisions, each with the events that followed it up to
the next decision (draws, hits, ...) so a replayer can check its sync.
"""

from __future__ import annotations

import gzip
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def scraper_dir(sub: str) -> str:
    """`<sts2_run_scraper>/data/<sub>`: the scraper checkout from
    `STS2_RUN_SCRAPER`, else the first `sts2_run_scraper` beside a parent
    of this file (the monorepo's sibling, also from a worktree)."""
    env = os.environ.get("STS2_RUN_SCRAPER")
    if env:
        return str(Path(env) / "data" / sub)
    for parent in Path(__file__).resolve().parents:
        cand = parent.parent / "sts2_run_scraper"
        if cand.is_dir():
            return str(cand / "data" / sub)
    return str(Path("sts2_run_scraper") / "data" / sub)


# `combat_end.result` of a finished fight: the game logs a death as "loss"
# (never "defeat"); a fight cut by a reload or a quit ends "abandoned".
FINISHED = ("victory", "loss")
DEATH = "loss"


# Decision event types: what the human chose. `pick` only counts when the
# human made it (`selector == "human"`) inside a combat.
DECISION_TYPES = ("play", "potion_start", "potion_used", "end_turn", "pick")


def _events(path: str) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                return  # truncated upload: keep what parsed


def _card_spec(c: dict[str, Any]) -> str:
    s = c["id"] + ("+1" if c.get("up") else "")
    if c.get("enchantment"):
        s += f"@{c['enchantment']}:{c.get('amount', 1)}"
    return s


class _RunTracker:
    """The player's out-of-combat state, replayed event by event."""

    def __init__(self, header: dict[str, Any]) -> None:
        self.header = header
        self.deck: list[dict[str, Any]] = list(header.get("starting_deck", []))
        self.relics: list[str] = [r["id"] for r in header.get("relics", [])]
        self.belt: list[str | None] = [p.get("id") for p in header.get("potions", [])]
        self.max_hp: int = header.get("starting_max_hp", 0)
        self.hp: int = self.max_hp

    def update(self, d: dict[str, Any]) -> None:
        t = d["t"]
        if t == "deck":
            self.deck = d["cards"]
        elif t == "relic" and d.get("id"):
            self.relics.append(d["id"])
        elif t == "relic_lost" and d.get("id") in self.relics:
            # `removed` / `replaced` by an event; a replacement logs its own `relic`.
            self.relics.remove(d["id"])
        elif t in ("hp", "max_hp") and d.get("dst", "player") == "player":
            if "hp" in d:
                self.hp = d["hp"]
            if "max_hp" in d:
                self.max_hp = d["max_hp"]
        elif t == "potion_got":
            self._belt_put(d["id"])
        elif t in ("potion_used", "potion_dropped"):
            self._belt_take(d["id"])

    def _belt_put(self, pid: str) -> None:
        for i, p in enumerate(self.belt):
            if p is None:
                self.belt[i] = pid
                return
        self.belt.append(pid)  # belt grew (relic) without our noticing

    def _belt_take(self, pid: str) -> None:
        for i, p in enumerate(self.belt):
            if p == pid:
                self.belt[i] = None
                return


def _take_turn_end_hits(events: list[dict[str, Any]], own: str | None) -> list[dict[str, Any]]:
    """Pop this turn's end-of-turn hand damage (Infection, Burn, ...) from `events`.

    The mod logs it before the player's `end_turn`, so it trails the previous
    decision's events. Only hits on the player by a card other than the last
    play's own card, after that play or the turn's start, are taken (GH #918).
    """
    start = 0
    for i, e in enumerate(events):
        if e["t"] == "play" or (e["t"] == "turn" and e.get("side") == "player"):
            start = i + 1
            own = e.get("id") if e["t"] == "play" else None
    taken = [
        e
        for e in events[start:]
        if e["t"] == "hit" and e.get("dst") == "player" and e.get("card") and e["card"] != own
    ]
    if taken:
        ids = {id(e) for e in taken}
        events[start:] = [e for e in events[start:] if id(e) not in ids]
    return taken


def fights(path: str) -> Iterator[dict[str, Any]]:
    """One record per combat attempt in the replay at `path`."""
    it = _events(path)
    header = next(it, None)
    if header is None or header.get("t") != "header":
        return
    run = _RunTracker(header)
    cur: dict[str, Any] | None = None
    attempts: dict[Any, int] = {}  # combat_id -> attempts started so far
    for d in it:
        t = d["t"]
        if t == "combat_start":
            attempts[d.get("combat_id")] = attempts.get(d.get("combat_id"), -1) + 1
            if cur is not None:
                cur["result"] = "abandoned"  # reloaded before combat_end
                yield cur
            cur = {
                "file": path,
                "seed": header.get("seed"),
                "build": header.get("build_id"),
                "character": header.get("character"),
                "ascension": header.get("ascension"),
                "floor": d.get("floor"),
                "act": d.get("act"),
                "combat_id": d.get("combat_id"),
                # 0 for the first attempt; k > 0 after k reloads of this fight.
                "attempt": attempts.get(d.get("combat_id"), 0),
                "attempt_id": d.get("attempt_id"),
                "encounter": d.get("encounter"),
                "enemies": d.get("enemies", []),
                "rng_state": d.get("rng_state"),
                "deck": [_card_spec(c) for c in run.deck],
                "deck_c": [c.get("c") for c in run.deck],
                "relics": list(run.relics),
                "belt": list(run.belt),
                "hp": run.hp,
                "max_hp": run.max_hp,
                "pre": [],  # events before the first decision (opening draws)
                "decisions": [],
                "result": None,
                "potions_started": {},  # potion id -> starts awaiting their `potion_used`
            }
            continue
        if cur is None:
            run.update(d)
            continue
        if t == "combat_end":
            cur["result"] = d.get("result")
            cur["hp_lost_total"] = d.get("hp_lost_total")
            cur["turns"] = d.get("turns")
            yield cur
            cur = None
            run.update(d)
            continue
        # A potion is decided at `potion_start` (its effects follow); older
        # replays log only `potion_used`.
        started = t == "potion_used" and cur["potions_started"].get(d.get("id"), 0) > 0
        if started:
            cur["potions_started"][d["id"]] -= 1
        elif t == "potion_start":
            cur["potions_started"][d.get("id")] = cur["potions_started"].get(d.get("id"), 0) + 1
        # A replayed play (Glam, Burst, Double Tap, ...: `play_index` > 0 of
        # `play_count`) is logged as its own non-auto `play`, but the human
        # chose only the first; the replays are that decision's aftermath.
        is_decision = (
            (t == "play" and not d.get("auto") and not d.get("play_index"))
            or t == "potion_start"
            or (t == "potion_used" and not started)
            or (t == "end_turn" and d.get("side") == "player")
            or (t == "pick" and d.get("selector") == "human")
        )
        if is_decision:
            after: list[dict[str, Any]] = []
            if t == "end_turn":
                prev = cur["decisions"][-1] if cur["decisions"] else None
                after = _take_turn_end_hits(
                    prev["after"] if prev else cur["pre"],
                    prev["ev"].get("id") if prev and prev["ev"]["t"] == "play" else None,
                )
            cur["decisions"].append({"ev": d, "after": after})
        elif cur["decisions"]:
            cur["decisions"][-1]["after"].append(d)
        else:
            cur["pre"].append(d)
        run.update(d)
    if cur is not None:
        cur["result"] = "abandoned"
        yield cur
