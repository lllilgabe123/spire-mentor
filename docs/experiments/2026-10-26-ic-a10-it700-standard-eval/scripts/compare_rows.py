"""Compare two ppo.probe JSONL files seed by seed: did the second run replay the first?

Compares, on the seeds both files hold, every field a run's play determines: outcome, won,
steps, n_decisions, act1, final, reached_second_boss and resolution_counts. Fields that
describe the machine or the build (wall_s, wheel, py_src) are ignored, so a public build,
a plain wheel or another host can replay the published rows.

Usage: python compare_rows.py PUBLISHED.jsonl YOURS.jsonl
Prints the count of identical and differing seeds, and the first differing fields.
"""

from __future__ import annotations

import json
import sys

PLAY_FIELDS = (
    "outcome",
    "won",
    "steps",
    "n_decisions",
    "act1",
    "final",
    "reached_second_boss",
    "resolution_counts",
)


def load(path: str) -> dict[int, dict]:
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {int(r["seed"]): r for r in rows}


def main(a_path: str, b_path: str) -> int:
    a, b = load(a_path), load(b_path)
    common = sorted(a.keys() & b.keys())
    diffs = {}
    for s in common:
        bad = [k for k in PLAY_FIELDS if a[s].get(k) != b[s].get(k)]
        if bad:
            diffs[s] = bad
    only_a, only_b = len(a.keys() - b.keys()), len(b.keys() - a.keys())
    print(f"seeds compared: {len(common)} (only in first: {only_a}, only in second: {only_b})")
    print(f"identical: {len(common) - len(diffs)}, different: {len(diffs)}")
    for s, bad in list(diffs.items())[:10]:
        print(f"  seed {s}: {', '.join(bad)}")
    return 0 if common and not diffs else 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
