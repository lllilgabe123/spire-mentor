"""Human decisions as rollout-net training shards. `python -m ppo.bc_human.gen`.

Every decision a human made from an engine state that matched their replay
(`sync.replay`) becomes one row in the `ppo.rollout_learn` shard format, with
the human's action as a one-hot `visits` target and as `chosen`, so

    python -m ppo.rollout_learn train --data <out>/train --target-mix 0 ...

fits a behaviour-cloned playout net without changes. Runs are split by file
hash: `--holdout-frac` of them go to `<out>/holdout` and never to training.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import time
from collections import Counter
from multiprocessing import Pool
from typing import Any

from ppo.bc_human import replay, sync
from ppo.bc_human.sync_census import ROOT, eligible
from ppo.rollout_learn import ROOMS, ShardWriter, _canon

VERSION = 1  # the feature version every shipped playout net reads


def _room(encounter: str) -> int:
    if encounter.endswith("_BOSS"):
        return ROOMS["boss"]
    if encounter.endswith("_ELITE"):
        return ROOMS["elite"]
    return ROOMS["monster"]


def run_rows(path: str) -> dict[str, Any]:
    import sts2_rs

    out: dict[str, Any] = {"file": path, "rows": [], "fights": 0, "n_dec": 0, "n_ok": 0}
    out["desync"] = Counter()
    try:
        for fi, rec in enumerate(replay.fights(path)):
            if rec["result"] not in replay.FINISHED or not rec["rng_state"]:
                continue
            r = sync.replay(rec)
            out["fights"] += 1
            out["n_dec"] += r["n_dec"]
            out["n_ok"] += r["n_ok"]
            out["desync"][r["desync"][0] if r["desync"] else "SYNC"] += 1
            for st, a in r["pairs"]:
                feats = sts2_rs.rollout_features(st, version=VERSION)
                if len(feats["actions"]) < 2:
                    continue
                index = {_canon(x): i for i, x in enumerate(feats["actions"])}
                i = index.get(_canon(a))
                if i is None:
                    out["desync"]["unmatched_action"] += 1
                    continue
                onehot = [0.0] * len(feats["actions"])
                onehot[i] = 1.0
                out["rows"].append(
                    {
                        "state": feats["state"],
                        "keys": feats["keys"],
                        "x": feats["x"],
                        "visits": onehot,
                        "chosen": i,
                        "pwin": 1.0 if rec["result"] == "victory" else 0.0,
                        "room": _room(rec["encounter"]),
                        "fight": fi,  # re-keyed globally by the writer loop
                    }
                )
    except Exception as e:
        out["error"] = repr(e)[:300]
    return out


def is_holdout(path: str, frac: float) -> bool:
    h = int(hashlib.sha256(os.path.basename(path).encode()).hexdigest()[:8], 16)
    return h / 0xFFFFFFFF < frac


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--character", default="IRONCLAD")
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", type=int, default=0, help="0 = every eligible run")
    ap.add_argument("--holdout-frac", type=float, default=0.1)
    ap.add_argument("--procs", type=int, default=24)
    ap.add_argument("--per-shard", type=int, default=20000)
    args = ap.parse_args()

    files = sorted(glob.glob(f"{ROOT}/{args.character.lower()}/*/*.gz"))
    random.Random(0).shuffle(files)
    with Pool(args.procs) as p:
        ok = p.starmap(eligible, [(f, args.character) for f in files], chunksize=64)
    files = [f for f, e in zip(files, ok, strict=True) if e]
    if args.runs:
        files = files[: args.runs]
    print(f"[bc gen] {len(files)} eligible runs", flush=True)

    writers = {
        "train": ShardWriter(os.path.join(args.out, "train"), args.per_shard),
        "holdout": ShardWriter(os.path.join(args.out, "holdout"), args.per_shard),
    }
    tot: Counter[str] = Counter()
    fight_id = 0
    t0 = time.time()
    with (
        Pool(args.procs) as p,
        open(os.path.join(args.out, "runs.jsonl"), "w", encoding="utf-8") as log,
    ):
        for n, res in enumerate(p.imap_unordered(run_rows, files, chunksize=2), 1):
            split = "holdout" if is_holdout(res["file"], args.holdout_frac) else "train"
            remap: dict[int, int] = {}
            for row in res["rows"]:
                if row["fight"] not in remap:
                    remap[row["fight"]] = fight_id
                    fight_id += 1
                row["fight"] = remap[row["fight"]]
            writers[split].add(res["rows"])
            tot[split + "_rows"] += len(res["rows"])
            for k in ("fights", "n_dec", "n_ok"):
                tot[k] += res[k]
            tot.update({f"desync:{k}": v for k, v in res["desync"].items()})
            tot["errors"] += "error" in res
            log.write(
                json.dumps(
                    {k: res.get(k) for k in ("file", "fights", "n_dec", "n_ok", "error")}
                    | {"split": split, "rows": len(res["rows"])}
                )
                + "\n"
            )
            if n % 50 == 0 or n == len(files):
                log.flush()
                print(
                    f"[bc gen] {n}/{len(files)} runs {time.time() - t0:.0f}s "
                    f"sync {tot['n_ok']}/{tot['n_dec']} "
                    f"rows train {tot['train_rows']} holdout {tot['holdout_rows']}",
                    flush=True,
                )
    for w in writers.values():
        w.flush()
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(tot), fh, indent=1)
    print(json.dumps(dict(tot), indent=1))


if __name__ == "__main__":
    main()
