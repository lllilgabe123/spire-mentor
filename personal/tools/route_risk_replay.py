"""Replay SimRouteRisk's warning rules over every logged map decision.

The map advice carries the recommended route with the HP the run AI expects
on arrival at each node (`path[].hp`). SimRouteRisk warns when that route
  1. enters an Elite below ELITE_PCT of max HP, or
  2. reaches the boss below BOSS_PCT of max HP (a RestSite as the last node
     adds its 30% heal).
Prints how often each rule fires, per run, and how each run's boss fight
went (the fight AI's win estimate at the boss and whether a boss recap
exists), so the rules can be judged against what happened. Read-only.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ELITE_PCT, BOSS_PCT, REST_HEAL = 0.5, 0.5, 0.3
LOG_DIR = Path(os.environ.get("LOCALAPPDATA", "")) / "SpireMentor" / "logs"


def warnings(path: list[dict], max_hp: int) -> list[str]:
    out = []
    for n in path:
        if n.get("type") == "Elite" and n.get("hp") is not None and n["hp"] < ELITE_PCT * max_hp:
            out.append(f"elite@r{n.get('row')} at {n['hp']}/{max_hp}")
    last = path[-1] if path else None
    if last and last.get("hp") is not None:
        end = last["hp"] + (REST_HEAL * max_hp if last.get("type") == "RestSite" else 0)
        if end < BOSS_PCT * max_hp:
            out.append(f"boss at ~{end:.0f}/{max_hp}")
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    total = fired = 0
    for f in sorted(LOG_DIR.glob("2026*.jsonl")):
        hp = mhp = floor = None
        decisions = warned = 0
        first_warn = None
        boss = None
        for line in f.open(encoding="utf-8", errors="replace"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            d = r.get("data") or {}
            if r.get("type") == "state":
                p = d.get("player") or {}
                hp, mhp, floor = p.get("hp"), p.get("max_hp"), d.get("floor")
                if (d.get("context") or {}).get("room_type") == "Boss" and boss is None:
                    boss = {"hp": hp, "max": mhp, "win": None}
            elif r.get("type") == "mcts":
                a = d.get("advice") or {}
                if a.get("path") and mhp:
                    decisions += 1
                    w = warnings(a["path"], mhp)
                    if w:
                        warned += 1
                        if first_warn is None:
                            first_warn = f"floor {floor} ({hp}/{mhp}): " + "; ".join(w)
                elif boss is not None and boss["win"] is None and a.get("p_win") is not None:
                    boss["win"] = a["p_win"]
        if decisions == 0:
            continue
        total += decisions
        fired += warned
        b = "no boss reached" if boss is None else \
            f"boss entered at {boss['hp']}/{boss['max']}, fight AI win {100 * (boss['win'] or 0):.0f}%"
        print(f"{f.name[:35]:36} map decisions {decisions:3}  warned {warned:3}  | {b}")
        if first_warn:
            print(f"    first warning: {first_warn}")
    print(f"\nall runs: {fired}/{total} map decisions would show a warning ({100 * fired / max(total, 1):.0f}%)")


if __name__ == "__main__":
    main()
