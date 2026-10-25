"""`python -m ppo.combat_bench.encounter_gap`: pilot-vs-human gap per encounter (GH #830).

For one room type of a fight set and one or more played files (arms), the
per-encounter gap `L_pilot - L_human` (the `report` L: damage, or all HP plus
65 on a death), with the pilot's reps averaged within a fight, deaths pilot /
human, fight length, and a 95 % wild cluster bootstrap CI by player
(`report._est`, the same estimator the bench report uses). Human quits are
excluded (`quit_excluded`).

    python -m ppo.combat_bench.encounter_gap --set <set.jsonl> --room elite \\
        B0=<played.jsonl> B2=<played.jsonl> [--json out.json]

Encounters are ranked by their share of the room's total gap (sum over fights)
under the first arm, so the top rows are the ones carrying the gap.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from typing import Any

from ppo.combat_bench import report as R

GAP_METRICS = ("L", "died", "dmg", "turns")


def arm_rows(fights: dict[str, dict[str, Any]], path: str, name: str, room: str) -> list[R.GapRow]:
    """The arm's gap rows for the fights of `room` with a scored rep and a human outcome."""
    keep = {k: f for k, f in fights.items() if f["room_type"] == room}
    played = R.load_played(path, name, fights)
    scored = {k: v for k, v in played.scored().items() if k in keep}
    return R.gap_rows(
        keep, scored, lambda k: R.human_metrics(keep[k], R.DEATH_HP, "quit_excluded"), R.DEATH_HP
    )


def encounter_rows(rows: list[R.GapRow], o: R.Opts) -> list[dict[str, Any]]:
    """Per-encounter gap rows, plus an `ALL` row, ranked by total gap (largest first)."""
    by: dict[str, list[R.GapRow]] = collections.defaultdict(list)
    for r in rows:
        by[r.encounter].append(r)
    total = sum(r.d("L") for r in rows)
    out = []
    for enc, rs in [("ALL", rows), *by.items()]:
        vals = [r.d("L") for r in rs]
        est = R._est(vals, [r.run for r in rs], [r.player for r in rs], o)
        n = len(rs)
        out.append(
            {
                "encounter": enc,
                "fights": n,
                "players": est["players"],
                "pilot_L": sum(r.p["L"] for r in rs) / n,
                "human_L": sum(r.h["L"] for r in rs) / n,
                "gap": est["mean"],
                "ci_lo": est["ci_lo"],
                "ci_hi": est["ci_hi"],
                "gap_sum": sum(vals),
                "share": sum(vals) / total if total else float("nan"),
                "pilot_died": 100 * sum(r.p["died"] for r in rs) / n,
                "human_died": 100 * sum(r.h["died"] for r in rs) / n,
                "pilot_dmg": sum(r.p["dmg"] for r in rs) / n,
                "human_dmg": sum(r.h["dmg"] for r in rs) / n,
                "pilot_turns": sum(r.p["turns"] for r in rs) / n,
                "human_turns": sum(r.h["turns"] for r in rs) / n,
            }
        )
    head, rest = out[0], out[1:]
    rest.sort(key=lambda r: -r["gap_sum"])
    return [head, *rest]


def render(tables: dict[str, list[dict[str, Any]]]) -> str:
    """The tables as Markdown, one per arm."""
    lines = []
    for arm, rows in tables.items():
        lines += [
            f"### {arm}",
            "",
            "| encounter | fights | players | pilot L | human L | gap [95% CI by player] "
            "| share of room gap | died p/h % | dmg p/h | turns p/h |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in rows:
            lines.append(
                f"| {r['encounter']} | {r['fights']} | {r['players']} | {r['pilot_L']:.1f} "
                f"| {r['human_L']:.1f} | {r['gap']:+.1f} [{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}] "
                f"| {100 * r['share']:.0f}% | {r['pilot_died']:.0f} / {r['human_died']:.0f} "
                f"| {r['pilot_dmg']:.1f} / {r['human_dmg']:.1f} "
                f"| {r['pilot_turns']:.1f} / {r['human_turns']:.1f} |"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m ppo.combat_bench.encounter_gap", description=__doc__
    )
    ap.add_argument("--set", required=True)
    ap.add_argument("--room", default="elite")
    ap.add_argument("--boot", type=int, default=9999)
    ap.add_argument("--json", default=None)
    ap.add_argument("arms", nargs="+", help="NAME=PATH played files")
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)
    fights = R.load_set(args.set)
    o = R.Opts(boot=args.boot)
    tables = {}
    for spec in args.arms:
        name, path = R._arm_spec(spec)
        tables[name] = encounter_rows(arm_rows(fights, path, name, args.room), o)
    print(render(tables))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(tables, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
