"""Score A10 probe/gate JSONL: the A10 win rate plus the A9 estimate it carries.

An A10 run that entered the DoubleBoss second-boss room beat the first act-3
boss, i.e. it would have won A9 (A10 = A9 + that fight). `reached_second_boss`
is None for a run that died before act 3; it counts as an A9 loss. This is an
ESTIMATE of the A9 win rate, not a measurement of it: on Ironclad it read ~5pp
below a true A9 sweep (2026-08-24, n=200), because the net is
ascension-conditioned and plays the A10 map differently.

  python a10_score.py run.jsonl              # one arm
  python a10_score.py champ.jsonl cand.jsonl # paired deltas, McNemar exact
"""

import json
import math
import sys


def load(path: str) -> dict[int, tuple[bool, bool] | None]:
    """seed -> (won A10, A9 estimate), or None for an error/timeout row."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("outcome") in ("error", "timeout"):
                out[int(r["seed"])] = None
                continue
            won = bool(r.get("won"))
            out[int(r["seed"])] = (won, won or r.get("reached_second_boss") is True)
    return out


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar p on discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def main(paths: list[str]) -> None:
    arms = [load(p) for p in paths]
    if len(arms) == 1:
        a = {k: v for k, v in arms[0].items() if v is not None}
        n = len(a)
        print(f"runs: {n} (dropped {len(arms[0]) - n} errored)")
        for i, name in ((0, "A10"), (1, "A9 (from A10)")):
            w = sum(v[i] for v in a.values())
            print(f"{name:14s} {w}/{n} = {w / max(1, n):.1%}")
        return
    champ, cand = arms
    both = champ.keys() | cand.keys()
    seeds = sorted(s for s in champ.keys() & cand.keys() if champ[s] and cand[s])
    n = len(seeds)
    print(
        f"paired seeds: {n} (dropped {len(both) - n} errored or unpaired, from both arms)"
    )
    for i, name in ((0, "A10"), (1, "A9 (from A10)")):
        wa = sum(champ[s][i] for s in seeds)
        wb = sum(cand[s][i] for s in seeds)
        b = sum(champ[s][i] and not cand[s][i] for s in seeds)
        c = sum(cand[s][i] and not champ[s][i] for s in seeds)
        print(
            f"{name:14s} champ {wa / n:.1%} -> cand {wb / n:.1%}  "
            f"delta {100 * (wb - wa) / n:+.2f}pp  disc {b}/{c}  McNemar p={mcnemar_p(b, c):.3g}"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
