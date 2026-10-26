"""Compare 12 replay rows (replay/<RUN>/rows.jsonl) with the sweep's rows of the same seeds.

Two replays, each one ppo.probe per seed (`--seeds 1 --seed-start <seed> --procs 1`, the sweep's
other flags and checkpoint) on a plain wheel, 12 at once:

    b0c3d6d60  private main b0c3d6d60 (2026-10-26), src_hash cb57e2e9bd57b45a
    78e23253d  the merge of this write-up's branch with private main 78e23253d (commit 7602c9aaf,
               engine source identical to 78e23253d), src_hash b53031922847d723

As run, the comparison read the 12 one-seed files from a directory; this version reads them
concatenated in rows.jsonl and prints the same lines plus a count.

Compared: every field of a row except `wheel`, `py_src` and `wall_s` (13 fields). `decisions` is
[] in every row of the sweep and of the replays, so the fields that carry play are outcome, won,
steps, n_decisions, act1, final, reached_second_boss and resolution_counts.

Usage: python replay/compare.py RUN [EXPERIMENT_DIR]   (EXPERIMENT_DIR default: the folder above
this script)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

IGNORE = {"wheel", "py_src", "wall_s"}


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    run = sys.argv[1]
    exp = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parents[1]
    rows = [
        json.loads(x)
        for x in (exp / "replay" / run / "rows.jsonl").read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    sweep: dict[int, dict[int, dict]] = {}
    n_diff = 0
    for mine in rows:
        a, seed = mine["ascension"], mine["seed"]
        if a not in sweep:
            cell = (exp / f"cells/ironclad_A{a}.jsonl").read_text(encoding="utf-8").splitlines()
            sweep[a] = {r["seed"]: r for r in (json.loads(x) for x in cell if x.strip())}
        pub = sweep[a][seed]
        keys = (set(mine) | set(pub)) - IGNORE
        diff = sorted(k for k in keys if mine.get(k) != pub.get(k))
        n_diff += bool(diff)
        print(
            f"A{a} {seed}: sweep {pub['outcome']:<4} steps {pub['steps']:>3} floor {pub['final']['total_floor']:>2} | "
            f"replay {mine['outcome']:<4} steps {mine['steps']:>3} floor {mine['final']['total_floor']:>2} | "
            f"fields compared {len(keys)}, differing: {diff or 'none'} | wall {pub['wall_s']:.0f}s -> {mine['wall_s']:.0f}s "
            f"| replay src_hash {mine['wheel']['src_hash']} pgo {mine['wheel']['pgo']}"
        )
    print(
        f"{len(rows)} replay rows, {len(rows) - n_diff} identical to the sweep's, {n_diff} differing"
    )
    return 1 if n_diff else 0


if __name__ == "__main__":
    raise SystemExit(main())
