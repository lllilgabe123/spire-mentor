"""`python -m ppo.combat_bench.devpanel`: the contested dev panel (GH #830).

Approaches are screened on a dev panel, never on the pro set. The panel keeps
the dev-pool boss and elite fights whose outcome the deployed pilot does not
already decide: on its own selection reps, at least one won and at least one
lost. Uncontested fights carry almost no paired signal on deaths, since every
arm wins (or loses) them.

The selection reps must be DISJOINT from the screening reps. A panel chosen
on the reps it is later scored on regresses to the mean: fights that looked
contested by luck look less contested next time, which biases every arm the
same way but shrinks the deltas. Selection runs at rep ids 40..47; screens use
0..N with N < 40.

`ironclad_dev_v1` was selected by the pilot from before GH #830 (2026-10-25),
which changed the deployed Ironclad search. It stays the frozen panel: do not
re-run this selection against the new default, or the panel moves under every
delta already measured on it.

    python -m ppo.combat_bench.devpanel --pool data/combat_bench/ironclad_devpool_v1.jsonl \
        --played sel.jsonl --out data/combat_bench/ironclad_dev_v1.jsonl \
        --manifest data/combat_bench/MANIFEST.json
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from typing import Any

SCORED = ("won", "died")
MIN_SCORED = 6  # of 8 selection reps; fewer means caps/errors decided the fight


def contested(cells: list[dict[str, Any]]) -> bool:
    """At least one scored win and one scored death."""
    ends = [c["end"] for c in cells if c["end"] in SCORED]
    return len(ends) >= MIN_SCORED and "won" in ends and "died" in ends


def read_cells(path: str) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """Header and the cells per key; a retried cell keeps its last record."""
    header: dict[str, Any] = {}
    last: dict[tuple[str, int], dict[str, Any]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if "header" in row:
                header = row["header"]
                continue
            last[row["key"], row["rep"]["rep"]] = row["rep"]
    by_key: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for (key, _), rep in sorted(last.items()):
        by_key[key].append(rep)
    return header, by_key


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="the contested dev panel (GH #830)")
    ap.add_argument("--pool", required=True, help="the dev pool fight set")
    ap.add_argument(
        "--played", required=True, help="the deployed pilot on the pool, selection reps"
    )
    ap.add_argument("--out", required=True, help="the dev panel fight set to write")
    ap.add_argument("--manifest", required=True, help="MANIFEST.json to record the panel in")
    ap.add_argument(
        "--note", default="", help="free text stored with the selection (e.g. a WG cap)"
    )
    args = ap.parse_args(argv)

    header, cells = read_cells(args.played)
    if header.get("spec", {}).get("search_dialect") != "belief-draw":
        print("refusing: the selection must come from the deployed honest search", file=sys.stderr)
        return 2
    reps = range(header["rep_offset"], header["rep_offset"] + header["reps"])
    if min(reps) < 40:
        print(
            f"refusing: selection reps {list(reps)} overlap the screening reps (< 40)",
            file=sys.stderr,
        )
        return 2
    pool_raw = open(args.pool, "rb").read()
    if hashlib.sha256(pool_raw).hexdigest() != header["set_sha256"]:
        print("refusing: the played file was not run on this pool", file=sys.stderr)
        return 2

    rows = [json.loads(line) for line in pool_raw.decode().splitlines() if line.strip()]
    kept = [r for r in rows if r["key"] in cells and contested(cells[r["key"]])]
    played_keys = sum(1 for r in rows if r["key"] in cells)
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in kept:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    sha = hashlib.sha256(open(args.out, "rb").read()).hexdigest()

    by_room = collections.Counter(r["room_type"] for r in kept)
    by_enc = collections.Counter(r["encounter"] for r in kept if r["room_type"] == "boss")
    entry = {
        "sha256": sha,
        "cohort": "other",
        "fights": len(kept),
        "runs": len({r["file"] for r in kept}),
        "players": len({r["player"] for r in kept}),
        "by_room": dict(by_room),
        "boss_encounters": dict(by_enc.most_common()),
        "selection": {
            "rule": (
                f"dev-pool fights played by the deployed pilot with >= {MIN_SCORED} scored "
                "selection reps, at least one won and one died"
            ),
            "played_fights": played_keys,
            "pool_sha256": header["set_sha256"],
            "reps": list(reps),
            "spec": header["spec"],
            "src_hash": header["src_hash"],
            "pgo": header.get("pgo"),
            "note": args.note,
        },
    }
    manifest = json.load(open(args.manifest, encoding="utf-8"))
    manifest["sets"][args.out.rsplit("/", 1)[-1]] = entry
    with open(args.manifest, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=False)
        fh.write("\n")
    print(json.dumps({k: v for k, v in entry.items() if k != "selection"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
