"""Paired comparison of a run-AI A/B (ab.sh output): every arm against "base".

Per ascension and pooled: win rates on the same seeds, the discordant pairs
(won only by base / only by the arm), and an exact two-sided McNemar p-value.

usage: python compare.py <dir> [arm ...]   (first arm is the baseline)
"""
import json
import math
import sys
from pathlib import Path


def wins(path: Path) -> dict[int, bool]:
    out = {}
    if path.exists():
        for line in path.open(encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[int(r["seed"])] = bool(r.get("won"))
    return out


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
        print(f"{'asc':>4} {'runs':>5} {base:>8} {arm:>8} {'delta':>8}  only-{base} only-{arm}  p")
        tot = [0, 0, 0, 0, 0]
        for asc in ascs:
            a, c = wins(d / f"{base}_A{asc}.jsonl"), wins(d / f"{arm}_A{asc}.jsonl")
            seeds = sorted(set(a) & set(c))
            if not seeds:
                continue
            n = len(seeds)
            aw, cw = sum(a[s] for s in seeds), sum(c[s] for s in seeds)
            oa = sum(a[s] and not c[s] for s in seeds)
            oc = sum(c[s] and not a[s] for s in seeds)
            print(f"A{asc:>3} {n:>5} {aw / n:>8.1%} {cw / n:>8.1%} {100 * (cw - aw) / n:>+7.1f}pp  {oa:>9} {oc:>9}  {mcnemar(oa, oc):.3f}")
            for i, v in enumerate((n, aw, cw, oa, oc)):
                tot[i] += v
        n, aw, cw, oa, oc = tot
        if n:
            print(f"{'all':>4} {n:>5} {aw / n:>8.1%} {cw / n:>8.1%} {100 * (cw - aw) / n:>+7.1f}pp  {oa:>9} {oc:>9}  {mcnemar(oa, oc):.3f}")


if __name__ == "__main__":
    main()
