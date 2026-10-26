"""Read the coach's run logs (%LOCALAPPDATA%\\SpireMentor\\logs) and summarize
how the player's runs end: outcomes, where HP goes, what kills runs, and
where the player and the AI disagree. Writes runs_summary.json next to this
file and prints the headline numbers. Read only."""

import collections
import glob
import json
import os
import statistics

LOGS = os.path.join(os.environ["LOCALAPPDATA"], "SpireMentor", "logs")
RUNS = os.path.join(os.environ["LOCALAPPDATA"], "SpireMentor", "runs.jsonl")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs_summary.json")


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


net_by_seed = {}
try:
    for r in load(RUNS):
        net_by_seed[r["seed"]] = {"net": r.get("net"), "players": r.get("players"), "asc": r.get("ascension")}
except OSError:
    pass

runs = {}  # seed -> merged run (a resumed run writes several logs)
for path in sorted(glob.glob(os.path.join(LOGS, "*.jsonl"))):
    name = os.path.basename(path)
    rows = load(path)
    meta = next((r["data"] for r in rows if r["type"] == "meta" and "seed" in r["data"]), {})
    seed = meta.get("seed") or name.split("_")[-1].removesuffix(".jsonl")
    run = runs.setdefault(seed, {"seed": seed, "character": meta.get("character"), "asc": meta.get("ascension"),
                                 "logs": [], "states": [], "coach": [], "summary": None, "start": name[:15]})
    run["logs"].append(name)
    for r in rows:
        if r["type"] == "state":
            run["states"].append(r["data"])
        elif r["type"] == "coach":
            run["coach"].append(r["data"])
        elif r["type"] == "meta" and r["data"].get("summary"):
            run["summary"] = r["data"]


def floor_of(s):
    return s.get("floor") or (s.get("live_context") or {}).get("total_floor")


def room_of(s):
    return (s.get("context") or {}).get("room_type") or s.get("room_type") or (s.get("live_context") or {}).get("room_type") or ""


out_runs = []
for seed, run in runs.items():
    st = run["states"]
    if not st:
        continue
    net = net_by_seed.get(seed, {})
    last_floor = max((floor_of(s) or 0) for s in st)
    summ = run["summary"] or {}
    over = [s for s in st if s.get("decision") == "game_over" or s.get("type") == "game_over"]
    won = summ.get("final_won") if over else None  # no game-over screen: quit or restarted
    # The fight that ended it: the last combat state.
    combats = [s for s in st if s.get("decision") == "combat_play"]
    death = None
    if over and won is False and combats:
        c = combats[-1]
        fl = floor_of(c)
        first = next(x for x in combats if floor_of(x) == fl)
        p0 = first.get("player") or {}
        fight_turns = [t for t in run["coach"] if t.get("kind") == "turn" and t.get("floor") == fl]
        death = {
            "floor": fl, "act": c.get("act"), "room": room_of(c), "round": c.get("round"),
            "enemies": [e.get("name") for e in c.get("enemies") or [] if (e.get("hp") or 0) > 0],
            "enemy_hp": [[e.get("name"), e.get("hp"), e.get("max_hp")] for e in c.get("enemies") or []],
            "hp_in": p0.get("hp"), "max_hp": p0.get("max_hp"),
            "deck_size": len(c.get("master_deck") or (c.get("player") or {}).get("deck") or []),
            "grades": dict(collections.Counter(t.get("verdict") for t in fight_turns)),
            "worst": [{k: t.get(k) for k in ("turn", "verdict", "win_before", "win_after", "you", "ai")}
                      for t in sorted(fight_turns, key=lambda t: -(t.get("severity") or 0))[:2]
                      if t.get("verdict") in ("blunder", "slip")],
        }
    # HP on entering each Elite/Boss fight (first combat state of that floor).
    entries = {}
    for c in combats:
        f = floor_of(c)
        if f not in entries:
            p = c.get("player") or {}
            entries[f] = {"floor": f, "room": room_of(c), "hp": p.get("hp"), "max_hp": p.get("max_hp"),
                          "enemies": [e.get("name") for e in c.get("enemies") or []]}
    co = run["coach"]
    recaps = [c for c in co if c.get("kind") == "recap"]
    turns = [c for c in co if c.get("kind") == "turn"]
    choices = [c for c in co if c.get("kind") == "choice"]
    out_runs.append({
        "seed": seed, "character": (run["character"] or "?").title(), "asc": run["asc"] or net.get("asc"),
        "net": net.get("net") or "?", "players": net.get("players") or 1, "start": run["start"],
        "last_floor": last_floor, "won": won, "finished": bool(over), "death": death,
        "elite_boss_entries": [e for e in entries.values() if e["room"] in ("Elite", "Boss")],
        "recaps": [{k: r.get(k) for k in ("floor", "room_type", "hp_start", "hp_end", "hp_lost", "ai_hp_lost", "turns")}
                   for r in recaps],
        "turn_verdicts": collections.Counter(t.get("verdict") for t in turns),
        "blunders": [{"floor": t.get("floor"), "turn": t.get("turn"), "you": t.get("you"), "ai": t.get("ai"),
                      "win_before": t.get("win_before"), "win_after": t.get("win_after"),
                      "hp_drop": t.get("hp_drop"), "severity": t.get("severity"),
                      "enemies": [e.get("name") for e in (t.get("context") or {}).get("enemies") or []]}
                     for t in turns if t.get("verdict") in ("blunder", "slip")],
        "flags": [{"decision": c.get("decision"), "strength": c.get("strength"),
                   "played": (c.get("played") or {}).get("label"), "played_prob": (c.get("played") or {}).get("prob"),
                   "top": (c.get("top") or {}).get("label"), "top_prob": (c.get("top") or {}).get("prob")}
                  for c in choices if c.get("verdict") == "flag"],
        "choices": collections.Counter(c.get("verdict") for c in choices),
        "autopsy": [c for c in co if c.get("kind") == "autopsy"],
    })

json.dump(out_runs, open(OUT, "w", encoding="utf-8"), default=lambda o: dict(o), indent=1)

# ---- headline numbers
real = [r for r in out_runs if r["finished"] and r["won"] is not None]
print(f"{len(out_runs)} runs logged, {len(real)} ended on a game-over screen")
by = collections.defaultdict(list)
for r in real:
    by[(r["character"], "co-op" if (r["players"] or 1) > 1 else "solo")].append(r)
for k, v in sorted(by.items()):
    w = sum(1 for r in v if r["won"])
    fl = statistics.median(r["last_floor"] for r in v)
    print(f"  {k[0]:12s} {k[1]:6s} runs {len(v):2d}  won {w}  median last floor {fl}")
print("\nDeaths:")
for r in real:
    if r["death"]:
        d = r["death"]
        print(f"  {r['start']} {r['character']:11s} {'co-op' if r['players'] > 1 else 'solo':6s} floor {d['floor']:2} act {d['act']} "
              f"{d['room']:7s} in at {d['hp_in']}/{d['max_hp']} round {d['round']} vs {', '.join(d['enemies'])[:40]:40s} grades {d['grades']}")
        for w in d["worst"]:
            print(f"      {w['verdict']} turn {w['turn']}: win {w['win_before']} -> {w['win_after']}; you {w['you']}; AI {w['ai']}")
