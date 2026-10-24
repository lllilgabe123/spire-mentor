"""Paired comparison of a run-AI A/B (ab.sh output): every arm against "base".

Per ascension and pooled, on the seeds both arms finished without an error:
win rates, the discordant pairs (won only by base / only by the arm) with an
exact two-sided McNemar p-value, and the mean floor reached with a paired
95% interval -- the floor still separates arms at A10, where almost every
run is lost. Crashed runs ("outcome": "error") are counted and left out.

usage: python compare.py <dir> [arm ...]   (first arm is the baseline)
"""
import json
import math
import statistics
import sys
from pathlib import Path


def runs(path: Path) -> tuple[dict[int, tuple[bool, int]], int]:
    """{seed: (won, total_floor)} for finished runs, and the error count."""
    out, errors = {}, 0
    if path.exists():
        for line in path.open(encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("outcome") == "error":
                errors += 1
                continue
            floor = int(((r.get("final") or {}).get("total_floor")) or 0)
            out[int(r["seed"])] = (bool(r.get("won")), floor)
    return out, errors


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def main() -> None:
    d = Path(sys.argv[1])
    arms = sys.argv[2:] or ["base", "caution", "pros", "smith50"]
    base, others = arms[0], arms[1:]
    ascs = sorted({int(p.stem.split("_A")[1]) for p in d.glob(f"{base}_A*.jsonl")}, reverse=True)
    for arm in others:
        print(f"\n== {arm} vs {base}")
        print(f"{'asc':>4} {'runs':>5} {'win '+base:>9} {'win '+arm:>9} {'delta':>8} {'p':>6}   floor {base} -> {arm} (95% interval of the change)  errors")
        for asc in ascs:
            (a, ea), (c, ec) = runs(d / f"{base}_A{asc}.jsonl"), runs(d / f"{arm}_A{asc}.jsonl")
            seeds = sorted(set(a) & set(c))
            if not seeds:
                continue
            n = len(seeds)
            aw, cw = sum(a[s][0] for s in seeds), sum(c[s][0] for s in seeds)
            oa = sum(a[s][0] and not c[s][0] for s in seeds)
            oc = sum(c[s][0] and not a[s][0] for s in seeds)
            diffs = [c[s][1] - a[s][1] for s in seeds]
            mean = statistics.fmean(diffs)
            half = 1.96 * statistics.stdev(diffs) / math.sqrt(n) if n > 1 else float("nan")
            fa = statistics.fmean(a[s][1] for s in seeds)
            print(f"A{asc:>3} {n:>5} {aw / n:>9.1%} {cw / n:>9.1%} {100 * (cw - aw) / n:>+7.1f}pp {mcnemar(oa, oc):>6.3f}"
                  f"   {fa:5.1f} -> {fa + mean:5.1f} ({mean - half:+.1f} .. {mean + half:+.1f})  {ea}/{ec}")


if __name__ == "__main__":
    main()
