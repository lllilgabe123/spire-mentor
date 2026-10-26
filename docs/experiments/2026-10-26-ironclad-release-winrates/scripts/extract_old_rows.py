"""Write old_ironclad_runs.csv: the 2026-10-11 release sweep's Ironclad rows, four columns.

The 2026-10-11 sweep (docs/experiments/2026-10-11-release-winrates.md) kept its per-run rows in
the private monorepo (logs/2026-10-11-release-winrates/runs.csv; columns character, ascension,
seed, act1, outcome, won, total_floor, wall_s; 14,000 rows, five characters). This script keeps
the 2,800 Ironclad rows, in the file's order, and only the columns the paired comparison needs:

    ascension, seed, act1, won        (won is "1" or "0", as in runs.csv)

Those are the numbers behind the published Ironclad column and its act-1 split. The old rows'
outcome is "won" or "lost" on every Ironclad row (0 errors, 0 timeouts), so it follows from won;
total_floor and wall_s are left out (see the write-up, "What a rerun on this folder reproduces").

Usage: python extract_old_rows.py RUNS_CSV [OUT_CSV]
  OUT_CSV defaults to old_ironclad_runs.csv in the folder above this script.
Prints the sha256 of both files; analysis/primary.json records the source's sha256 as
inputs.old_runs_csv_sha256.
"""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

COLUMNS = ("ascension", "seed", "act1", "won")


def main() -> int:
    if len(sys.argv) not in (2, 3):
        raise SystemExit(__doc__)
    src = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) == 3 else Path(__file__).resolve().parents[1] / "old_ironclad_runs.csv"
    rows = []
    with open(src, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["character"] != "Ironclad":
                continue
            if r["outcome"] not in ("won", "lost") or (r["won"] == "1") != (r["outcome"] == "won"):
                raise SystemExit(f"unexpected outcome/won on seed {r['seed']} A{r['ascension']}: {r}")
            rows.append([r[c] for c in COLUMNS])
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(COLUMNS)
        w.writerows(rows)
    for p in (src, out):
        print(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}")
    print(f"{len(rows)} Ironclad rows written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
