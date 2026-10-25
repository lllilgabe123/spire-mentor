"""How far do human fights replay in sync? `python -m ppo.bc_human.sync_census`.

Samples v0.111.0 single-player replays without gameplay mods that carry RNG
counters, replays every fight that reached `combat_end`, and tallies where
each one first left sync.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import random
from collections import Counter
from multiprocessing import Pool
from typing import Any

from ppo.bc_human import replay, sync

ROOT = replay.scraper_dir("replays-spire-codex")


def eligible(path: str, character: str | None) -> bool:
    with gzip.open(path, "rt") as fh:
        try:
            h = json.loads(fh.readline())
        except (json.JSONDecodeError, EOFError, OSError):
            return False
    return (
        h.get("build_id") == sync.GAME_VERSION
        and (h.get("replay_version") or 0) >= 6
        and (h.get("player_count") or 1) == 1
        and h.get("game_mode") == "standard"
        and sync.mods_ok(h)
        and (character is None or h.get("character") == character)
    )


MAI_OFFSET: int | None = None


def _init(off: int | None) -> None:
    global MAI_OFFSET
    MAI_OFFSET = off


def one(path: str) -> list[dict[str, Any]]:
    rows = []
    try:
        for rec in replay.fights(path):
            if rec["result"] not in replay.FINISHED or not rec["rng_state"]:
                continue
            r = sync.replay(rec, MAI_OFFSET)
            rows.append(
                {
                    "file": path,
                    "combat_id": rec["combat_id"],
                    "encounter": rec["encounter"],
                    "result": rec["result"],
                    "n_dec": r["n_dec"],
                    "n_ok": r["n_ok"],
                    "desync": r["desync"],
                }
            )
    except Exception as e:
        rows.append({"file": path, "crash": repr(e)[:300]})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--character", default="IRONCLAD")
    ap.add_argument("--runs", type=int, default=40)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--mai-offset", type=int, default=None, help="default: #monsters")
    args = ap.parse_args()
    files = sorted(glob.glob(f"{ROOT}/{args.character.lower()}/*/*.gz"))
    random.Random(0).shuffle(files)
    picked = []
    for f in files:
        if eligible(f, args.character):
            picked.append(f)
            if len(picked) >= args.runs:
                break
    with Pool(args.workers, _init, (args.mai_offset,)) as p:
        rows = [r for rs in p.imap_unordered(one, picked) for r in rs]
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    fights = [r for r in rows if "crash" not in r]
    kinds = Counter(r["desync"][0] if r["desync"] else "SYNC" for r in fights)
    n_dec = sum(r["n_dec"] for r in fights)
    n_ok = sum(r["n_ok"] for r in fights)
    print(f"runs {len(picked)} fights {len(fights)} crashes {len(rows) - len(fights)}")
    print(f"decisions in sync {n_ok}/{n_dec} = {n_ok / max(n_dec, 1):.1%}")
    print("first desync:", kinds.most_common())


if __name__ == "__main__":
    main()
