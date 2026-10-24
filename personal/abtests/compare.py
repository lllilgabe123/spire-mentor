"""Paired comparison of the low-HP caution A/B (lowhp_ab.sh output).

Per ascension and pooled: win rates of both arms on the same seeds, the
discordant pairs, and an exact two-sided McNemar p-value.
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
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def main() -> None:
    d = Path(sys.argv[1])
    tb = tc = tn = tbw = tcw = 0
    print(f"{'asc':>4} {'runs':>5} {'base':>7} {'caution':>8} {'delta':>7}  only-base only-caution  p")
    for asc in (10, 9, 8, 7):
        a, c = wins(d / f"base_A{asc}.jsonl"), wins(d / f"caution_A{asc}.jsonl")
        seeds = sorted(set(a) & set(c))
        if not seeds:
            continue
        n = len(seeds)
        aw = sum(a[s] for s in seeds)
        cw = sum(c[s] for s in seeds)
        only_a = sum(a[s] and not c[s] for s in seeds)
        only_c = sum(c[s] and not a[s] for s in seeds)
        print(f"A{asc:>3} {n:>5} {aw / n:>7.1%} {cw / n:>8.1%} {100 * (cw - aw) / n:>+6.1f}pp  {only_a:>9} {only_c:>12}  {mcnemar(only_a, only_c):.3f}")
        tn += n
        tbw += aw
        tcw += cw
        tb += only_a
        tc += only_c
    if tn:
        print(f"{'all':>4} {tn:>5} {tbw / tn:>7.1%} {tcw / tn:>8.1%} {100 * (tcw - tbw) / tn:>+6.1f}pp  {tb:>9} {tc:>12}  {mcnemar(tb, tc):.3f}")


if __name__ == "__main__":
    main()
