"""Spire Mentor reliability tracker.

Reads the run logs Spire Mentor's advisor already writes
(%LOCALAPPDATA%\\SpireMentor\\logs\\*.jsonl) and measures, per fight:

  * what the AI predicted at the start of the fight (win chance, and the HP
    you would finish with if its line were played -- the best option's
    `mean_final_hp` from the MCTS search), and
  * what actually happened (the fight's `recap`: HP lost; a fight with no
    recap at the end of a run was lost or abandoned),
  * how your turns were graded (best / tie / slip / blunder).

and summarises it separately for solo runs, co-op runs and the local test
bots, so you can see how far to trust the fight advice in each.

Read-only and local: it never touches the game, the mod or the network.

Run classification, most reliable first:
  1. The run started inside a local two-instance test (spire-test-bot results
     folders) -> "test bots (co-op)".
  2. runs.jsonl (written by Spire Mentor at every run start since the watchdog
     update): Singleplayer / Host / Client.
  3. The run holds a multiplayer-only card (e.g. MIDNIGHT) -> co-op.
  4. Otherwise "solo or not sure".

Usage:  python tracker.py [--html report.html]
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
import re
import statistics
import sys
from pathlib import Path

APPDATA = Path(os.environ.get("LOCALAPPDATA", "")) / "SpireMentor"
LOG_DIR = APPDATA / "logs"
RUNS_FILE = APPDATA / "runs.jsonl"
FIGHTS_FILE = APPDATA / "fights.jsonl"
TEST_RESULTS = Path(__file__).resolve().parent.parent / "spire-test-bot" / "results"

# v0.111.0 cards that only exist in multiplayer (CardMultiplayerConstraint.MultiplayerOnly).
MP_ONLY_CARDS = {
    "MIDNIGHT", "BLADE_SYMPHONY", "BLAZE", "CACOPHONY_CARD", "CONCOCT_CARD",
    "KNOCKDOWN", "ONE_FOR_ALL_CARD", "SOULBOUND_CARD", "SIDESTEP", "OUTRAGE",
}
MP_ONLY_RE = re.compile(r'"(%s)"' % "|".join(sorted(MP_ONLY_CARDS)))
FILE_RE = re.compile(r"^(\d{8})_(\d{6})_(\w+?)_([A-Z0-9]+)\.jsonl$")

SOLO, COOP, BOTS, UNSURE = "solo", "co-op", "test bots (co-op)", "solo or not sure"
GROUP_ORDER = [SOLO, UNSURE, COOP, BOTS]


def load_runs_file() -> dict[str, str]:
    """seed -> solo / co-op from Spire Mentor's run notes."""
    out: dict[str, str] = {}
    if not RUNS_FILE.exists():
        return out
    for line in RUNS_FILE.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        seed, net = row.get("seed"), row.get("net")
        if seed and net:
            out[seed] = SOLO if net == "Singleplayer" else COOP
    return out


def load_fight_damage() -> dict[tuple[str, int], tuple[int, int]]:
    """(seed, floor) -> (your damage, teammates' damage) from Spire Mentor's
    fight notes (each player's damage to enemies, read from the game's own
    combat history at fight end)."""
    out: dict[tuple[str, int], tuple[int, int]] = {}
    if not FIGHTS_FILE.exists():
        return out
    for line in FIGHTS_FILE.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        dmg, me = row.get("damage") or {}, row.get("me")
        if not row.get("seed") or row.get("floor") is None or me is None:
            continue
        mine = int(dmg.get(me, 0))
        others = sum(int(v) for k, v in dmg.items() if k != me)
        out[(row["seed"], int(row["floor"]))] = (mine, others)
    return out


def test_windows() -> list[tuple[dt.datetime, dt.datetime]]:
    """[start, end) of every local two-instance test, from its results folder."""
    if not TEST_RESULTS.is_dir():
        return []
    # The window ends when the test's files were last written: a test that
    # stopped early must not swallow a real game played straight after it.
    windows = []
    for d in TEST_RESULTS.iterdir():
        m = re.match(r"^(\d{8})-(\d{6})-", d.name)
        if not (d.is_dir() and m):
            continue
        start = dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        mtimes = [p.stat().st_mtime for p in d.iterdir() if p.is_file()]
        end = dt.datetime.fromtimestamp(max(mtimes)) if mtimes else start
        windows.append((start, end + dt.timedelta(seconds=30)))
    return windows


def classify(path: Path, seed: str, started: dt.datetime, text: str,
             notes: dict[str, str], windows) -> str:
    # Test bots first: their host runs Spire Mentor too, so they also leave
    # run notes ("Host") that would otherwise count as your co-op games.
    if any(s <= started < e for s, e in windows):
        return BOTS
    if seed in notes:
        return notes[seed]
    if MP_ONLY_RE.search(text):
        return COOP
    return UNSURE


def parse_run(path: Path) -> list[dict]:
    """One dict per fight in this log file."""
    fights: dict[int, dict] = {}
    last_hp = None
    last_floor = None
    order: list[int] = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind, data = rec.get("type"), rec.get("data") or {}
            if kind == "state":
                player = data.get("player") or {}
                if isinstance(player.get("hp"), (int, float)):
                    last_hp = player["hp"]
                if data.get("floor") is not None:
                    last_floor = data["floor"]
            elif kind == "mcts":
                advice = data.get("advice") or {}
                if advice.get("p_win") is None or last_floor is None:
                    continue  # not a fight decision
                f = fights.get(last_floor)
                if f is None:
                    f = fights[last_floor] = {"floor": last_floor, "turns": []}
                    order.append(last_floor)
                if "pred_win" not in f:
                    best = next((o for o in advice.get("options", []) if o.get("is_best")), None)
                    f["pred_win"] = advice.get("p_win")
                    f["hp_at_start"] = last_hp
                    if best and best.get("mean_final_hp") is not None and last_hp is not None:
                        f["pred_hp_lost"] = max(0.0, last_hp - best["mean_final_hp"])
            elif kind == "coach":
                floor = data.get("floor")
                if floor is None:
                    continue
                f = fights.setdefault(floor, {"floor": floor, "turns": []})
                if floor not in order:
                    order.append(floor)
                if data.get("kind") == "turn":
                    f["turns"].append(data.get("verdict"))
                elif data.get("kind") == "recap":
                    f["recap"] = data
    out = []
    for i, floor in enumerate(order):
        f = fights[floor]
        if "pred_win" not in f and "recap" not in f:
            continue
        recap = f.get("recap")
        if recap:
            f["result"] = "won"
            f["hp_lost"] = recap.get("hp_lost")
            f["room"] = recap.get("room_type")
        elif i == len(order) - 1:
            f["result"] = "lost or unfinished"
        else:
            continue
        graded = [v for v in f["turns"] if v in ("best", "tie", "slip", "blunder")]
        f["graded"] = graded
        f["followed"] = bool(graded) and all(v in ("best", "tie") for v in graded)
        out.append(f)
    return out


def summarise(fights: list[dict]) -> dict:
    turns = [v for f in fights for v in f["graded"]]
    won = [f for f in fights if f["result"] == "won"]
    pairs = [(f["pred_hp_lost"], f["hp_lost"]) for f in won
             if f.get("pred_hp_lost") is not None and f.get("hp_lost") is not None]
    pairs_followed = [(f["pred_hp_lost"], f["hp_lost"]) for f in won
                      if f["followed"] and f.get("pred_hp_lost") is not None and f.get("hp_lost") is not None]

    def err(ps):
        if not ps:
            return None
        diffs = [a - p for p, a in ps]
        return {
            "n": len(ps),
            "pred": statistics.mean(p for p, _ in ps),
            "actual": statistics.mean(a for _, a in ps),
            "bias": statistics.mean(diffs),
            "mae": statistics.mean(abs(d) for d in diffs),
        }

    pw = [f["pred_win"] for f in fights if f.get("pred_win") is not None]
    dmg = [f["damage"] for f in fights if f.get("damage")]
    mine_total = sum(m for m, _ in dmg)
    team_total = sum(o for _, o in dmg)
    return {
        "dmg_fights": len(dmg),
        "teammate_share": team_total / (mine_total + team_total) if mine_total + team_total else None,
        "fights": len(fights),
        "won": len(won),
        "lost": len(fights) - len(won),
        "pred_win": statistics.mean(pw) if pw else None,
        "turns": len(turns),
        "best": sum(v in ("best", "tie") for v in turns),
        "slip": turns.count("slip"),
        "blunder": turns.count("blunder"),
        "hp_all": err(pairs),
        "hp_followed": err(pairs_followed),
    }


def pct(a, b):
    return f"{100 * a / b:.0f}%" if b else "-"


def text_report(groups: dict[str, dict], runs_by_group: dict[str, list[str]]) -> str:
    lines = ["Spire Mentor fight-advice reliability", "=" * 38, ""]
    for g in GROUP_ORDER:
        if g not in groups:
            continue
        s = groups[g]
        lines.append(f"[{g}]  runs: {', '.join(runs_by_group[g])}")
        lines.append(f"  fights: {s['fights']}  (won {s['won']}, lost/unfinished {s['lost']})"
                     + (f"   AI's average win estimate at fight start: {100 * s['pred_win']:.0f}%" if s['pred_win'] is not None else ""))
        lines.append(f"  your turns: {s['turns']} graded -- matched the AI {pct(s['best'], s['turns'])}, "
                     f"small slips {s['slip']}, blunders {s['blunder']}")
        if s["teammate_share"] is not None and g in (COOP, BOTS):
            lines.append(f"  your teammate dealt {100 * s['teammate_share']:.0f}% of the players' damage "
                         f"({s['dmg_fights']} fights measured) -- damage the fight AI does not know about")
        for label, key in (("all fights", "hp_all"), ("fights where you followed the AI every turn", "hp_followed")):
            e = s[key]
            if e:
                lines.append(f"  HP lost, {label} ({e['n']}): AI predicted {e['pred']:.1f}, actual {e['actual']:.1f} "
                             f"-> off by {e['mae']:.1f} on average, "
                             f"{'more' if e['bias'] > 0 else 'less'} than predicted by {abs(e['bias']):.1f} overall")
            else:
                lines.append(f"  HP lost, {label}: no fights to compare yet")
        lines.append("")
    return "\n".join(lines)


def html_report(text: str, fights_by_group: dict[str, list[dict]]) -> str:
    rows = []
    for g in GROUP_ORDER:
        for f in fights_by_group.get(g, []):
            rows.append(
                "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                    html.escape(g), html.escape(f["run"]), f["floor"], html.escape(str(f.get("room") or "")),
                    "" if f.get("pred_hp_lost") is None else f"{f['pred_hp_lost']:.0f}",
                    "" if f.get("hp_lost") is None else f["hp_lost"],
                    html.escape(" ".join(f["graded"]) or "-")))
    return f"""<!doctype html><meta charset="utf-8"><title>Spire Mentor reliability</title>
<style>body{{font:14px system-ui;margin:16px;background:#fff;color:#111}}pre{{white-space:pre-wrap}}
table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:3px 6px}}
@media (prefers-color-scheme:dark){{body{{background:#111;color:#eee}}td,th{{border-color:#444}}}}</style>
<h1>Spire Mentor fight-advice reliability</h1><pre>{html.escape(text)}</pre>
<h2>Every fight</h2><table><tr><th>group</th><th>run</th><th>floor</th><th>room</th>
<th>AI predicted HP lost</th><th>actual HP lost</th><th>your turns</th></tr>{''.join(rows)}</table>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", type=Path, help="also write an HTML report here")
    args = ap.parse_args()
    if not LOG_DIR.is_dir():
        print(f"No Spire Mentor logs at {LOG_DIR}")
        return 1
    notes, windows, damage = load_runs_file(), test_windows(), load_fight_damage()
    fights_by_group: dict[str, list[dict]] = {}
    runs_by_group: dict[str, list[str]] = {}
    for path in sorted(LOG_DIR.glob("*.jsonl")):
        m = FILE_RE.match(path.name)
        if not m:
            continue
        started = dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        seed = m.group(4)
        text = path.read_text(encoding="utf-8", errors="replace")
        group = classify(path, seed, started, text, notes, windows)
        fights = parse_run(path)
        if not fights:
            continue
        for f in fights:
            f["run"] = seed
            f["damage"] = damage.get((seed, int(f["floor"])))
        fights_by_group.setdefault(group, []).extend(fights)
        if seed not in runs_by_group.setdefault(group, []):
            runs_by_group[group].append(seed)
    groups = {g: summarise(fs) for g, fs in fights_by_group.items()}
    text = text_report(groups, runs_by_group)
    print(text)
    if args.html:
        args.html.write_text(html_report(text, fights_by_group), encoding="utf-8")
        print(f"HTML report: {args.html}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
