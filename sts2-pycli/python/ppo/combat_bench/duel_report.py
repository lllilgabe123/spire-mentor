"""`python -m ppo.combat_bench.duel_report`: summarize a `replay_duel` run (GH #830).

Per encounter, the paired gap `L_arm - L_human` on the same start states,
with a 95 % cluster bootstrap CI by uploader (all fights of one player
resample together), deaths, fight length, potions drunk and damage taken per
round. Every number is reported for three subsets: `all` finished fights;
`noreload`, the fights the human finished on the first attempt (a reloaded
fight shows only the attempt the human kept, which flatters the human); and
`clean`, the `noreload` fights whose whole human line replayed in sync in the
engine (`sync is None`): there the engine reproduces what the human did, so a
gap cannot come from an engine that plays the encounter differently.

    python -m ppo.combat_bench.duel_report --duel <dir> [--json out.json] [--md out.md]
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

N_BOOT = 2000
SUBSETS = {
    "all": lambda r: True,
    "noreload": lambda r: r.get("attempt", 0) == 0,
    "clean": lambda r: r.get("attempt", 0) == 0 and r["sync"] is None,
}


def load(duel: str | Path) -> list[dict[str, Any]]:
    """The duel's fight rows without errors, each with its uploader as `player`."""
    duel = Path(duel)
    who: dict[str, str] = {}
    sel = duel / "selection.json"
    if sel.exists():
        files = json.loads(sel.read_text())["files"]
    else:  # a duel from before `replay_duel` wrote its selection
        from ppo.turnplan import pro

        files, _ = pro.select_files(pro.ROOT, "IRONCLAD", set(range(11)))
    for f in files:
        who[Path(f["file"]).name] = f.get("username") or f"anon:{Path(f['file']).name}"
    rows = []
    for line in (duel / "fights.jsonl").open():
        r = json.loads(line)
        if "error" in r:
            continue
        r["player"] = who.get(r["file"], f"anon:{r['file']}")
        rows.append(r)
    return rows


def cluster_ci(vals: np.ndarray, groups: list[str], seed: int = 0) -> tuple[float, float, float]:
    """Mean and 95 % percentile CI of `vals`, resampling whole `groups`."""
    uniq = sorted(set(groups))
    idx = np.searchsorted(uniq, groups)
    sums = np.bincount(idx, weights=vals, minlength=len(uniq))
    cnt = np.bincount(idx, minlength=len(uniq)).astype(float)
    rng = np.random.default_rng(seed)
    w = rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)), size=N_BOOT)
    boot = (w @ sums) / np.maximum(w @ cnt, 1e-9)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(vals.mean()), float(lo), float(hi)


def _by_round(rows: list[dict[str, Any]], side: str, max_round: int = 8) -> list[float]:
    out = []
    for k in range(1, max_round + 1):
        out.append(float(np.mean([r[side]["dmg_by_round"].get(str(k), 0) for r in rows])))
    return out


def summarize(rows: list[dict[str, Any]], arms: list[str]) -> dict[str, Any]:
    """Per encounter (and `ALL`), per subset (`all` / `clean`), per arm."""
    out: dict[str, Any] = {}
    encs = collections.Counter(r["encounter"] for r in rows)
    for enc in ["ALL", *[e for e, _ in encs.most_common()]]:
        e_rows = rows if enc == "ALL" else [r for r in rows if r["encounter"] == enc]
        out[enc] = {}
        for sub in SUBSETS:
            rs = [r for r in e_rows if SUBSETS[sub](r)]
            rs = [r for r in rs if all(a in r for a in arms)]
            if not rs:
                continue
            players = [r["player"] for r in rs]
            h = {
                "L": float(np.mean([r["human"]["L"] for r in rs])),
                "died": 100 * float(np.mean([r["human"]["died"] for r in rs])),
                "rounds": float(np.mean([r["human"]["rounds"] for r in rs])),
                "potions": float(np.mean([len(r["human"]["potions"]) for r in rs])),
                "dmg_by_round": _by_round(rs, "human"),
            }
            cell: dict[str, Any] = {"fights": len(rs), "players": len(set(players)), "human": h}
            for a in arms:
                gap = np.array([r[a]["L"] - r["human"]["L"] for r in rs], float)
                m, lo, hi = cluster_ci(gap, players)
                cell[a] = {
                    "L": float(np.mean([r[a]["L"] for r in rs])),
                    "gap": m,
                    "ci": [lo, hi],
                    "died": 100 * float(np.mean([r[a]["died"] for r in rs])),
                    "rounds": float(np.mean([r[a]["rounds"] for r in rs])),
                    "potions": float(np.mean([len(r[a]["potions"]) for r in rs])),
                    "dmg_by_round": _by_round(rs, a),
                }
            out[enc][sub] = cell
    return out


def render(s: dict[str, Any], arms: list[str]) -> str:
    """The summary as Markdown tables."""
    lines = []
    for sub in SUBSETS:
        lines += [
            f"### {sub} fights",
            "",
            "| encounter | fights | players | human L | "
            + " | ".join(f"{a} gap [95% CI]" for a in arms)
            + " | died h / "
            + " / ".join(arms)
            + " % | rounds h / "
            + " / ".join(arms)
            + " | potions h / "
            + " / ".join(arms)
            + " |",
            "|---|---|---|---|" + "---|" * len(arms) + "---|---|---|",
        ]
        for enc, cells in s.items():
            c = cells.get(sub)
            if not c:
                continue
            gaps = " | ".join(
                f"{c[a]['gap']:+.1f} [{c[a]['ci'][0]:+.1f}, {c[a]['ci'][1]:+.1f}]" for a in arms
            )
            died = " / ".join(f"{c[a]['died']:.0f}" for a in arms)
            rounds = " / ".join(f"{c[a]['rounds']:.1f}" for a in arms)
            pots = " / ".join(f"{c[a]['potions']:.2f}" for a in arms)
            h = c["human"]
            lines.append(
                f"| {enc} | {c['fights']} | {c['players']} | {h['L']:.1f} | {gaps} "
                f"| {h['died']:.0f} / {died} | {h['rounds']:.1f} / {rounds} "
                f"| {h['potions']:.2f} / {pots} |"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ppo.combat_bench.duel_report", description=__doc__)
    ap.add_argument("--duel", required=True)
    ap.add_argument("--arms", nargs="+", default=["B0", "B2"])
    ap.add_argument("--json", default=None)
    ap.add_argument("--md", default=None)
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)
    s = summarize(load(args.duel), args.arms)
    md = render(s, args.arms)
    print(md)
    if args.json:
        Path(args.json).write_text(json.dumps(s, indent=1) + "\n", encoding="utf-8")
    if args.md:
        Path(args.md).write_text(md + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
