"""Critic belt-counterfactual probe for the potion opportunity cost (option A).

GH #447 follow-up. The deployed PPO run critic already sees the belt
(``obs["potions"]`` tokens + the ``num_potions`` scalar), so a potion's
carry-forward value can be read off as a counterfactual difference
``dV = V(obs) - V(obs minus potion)`` at every non-combat decision — zero
new training. This probe drives Ironclad runs with the deployed champion
(non-combat decisions greedily from the same net; combat via the standard
``PpoRunEnv`` MCTS backend), collects dV per (potion, act), and converts
to HP-equivalents through the SAME critic's HP sensitivity
(``dV_hp10 = V(hp+10) - V(hp)``, so ``oc_hp_equiv = dV / (dV_hp10/10)``).

Output: a raw JSONL of counterfactual rows plus an OC table in the
``STS2_RUNAI_POTION_OC`` v1 format (source ``critic_counterfactual``).
Calibration verdicts (sign share, act gradient, class ordering) print at
the end and are written into the table file's ``"calibration"`` block —
the pre-registered shape checks live in the plan, not here.

Usage (from sts2-pycli):
    PYTHONPATH=python uv run --frozen python python/diag_potion_oc_probe.py \
        --runs 100 --out ../logs/2026-08-19-potion-oc
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

# Potion classes for the calibration check only (the table stays per-id).
SCALING = {
    "POTION.POWER_POTION",
    "POTION.STRENGTH_POTION",
    "POTION.DEXTERITY_POTION",
    "POTION.DUPLICATOR",
}
NUKES = {"POTION.FIRE_POTION", "POTION.EXPLOSIVE_AMPOULE", "POTION.POTION_SHAPED_ROCK"}


def canonical_potion(token: str) -> str:
    """``rm.potions()`` PascalCase (``SpeedPotion``) → the ``POTION.SPEED_POTION``
    form the Rust OC-table consumer builds from ``PotionKind::python_name()``.
    Without this every table lookup would silently fall to ``default_oc``."""
    if token.startswith("POTION."):
        return token
    return "POTION." + re.sub(r"(?<!^)(?=[A-Z])", "_", token).upper()


_POLICY = None  # one ONNX session per worker process


def _one_run(args: tuple[int, int, str]) -> tuple[list[dict], dict]:
    """Play one seed; return (counterfactual rows, run summary)."""
    seed, ascension, act1 = args
    import champions
    from ppo.env import PpoRunEnv, Terminal
    from ppo.onnx_policy import OnnxPolicy

    global _POLICY
    if _POLICY is None:
        champ = champions.get("ironclad")
        assert champ is not None, "no Ironclad champion in champions.toml"
        _POLICY = OnnxPolicy(str(champ.onnx_path))
    policy = _POLICY

    env = PpoRunEnv(seed=seed, ascension=ascension, character="Ironclad", act1=act1)
    d = env.reset()
    rows: list[dict] = []
    steps = 0
    while not isinstance(d, Terminal) and steps < 3000:
        steps += 1
        probs, v = policy.distribution(d.obs, d.actions, d.metrics)
        belt = list(d.obs.get("potions") or [])
        if belt:
            # HP sensitivity at this state: +10 HP (capped) on BOTH the
            # obs scalar and the metrics copy, mirroring the potion edit.
            hp, max_hp = int(d.obs["hp"]), int(d.obs["max_hp"])
            bump = min(10, max_hp - hp)
            dv_hp10 = None
            if bump > 0:
                obs_hp = {**d.obs, "hp": hp + bump}
                met_hp = {**d.metrics, "hp": hp + bump}
                _, v_hp = policy.distribution(obs_hp, d.actions, met_hp)
                dv_hp10 = (v_hp - v) * (10.0 / bump)
            for pid in sorted(set(belt)):
                cf = list(belt)
                cf.remove(pid)
                obs_cf = {**d.obs, "potions": cf, "num_potions": len(cf)}
                met_cf = {**d.metrics, "num_potions": len(cf)}
                _, v_cf = policy.distribution(obs_cf, d.actions, met_cf)
                rows.append(
                    {
                        "seed": seed,
                        "act": int(d.obs["act_index"]),
                        "floor": int(d.obs["total_floor"]),
                        "hp_frac": hp / max_hp if max_hp else 0.0,
                        "belt": len(belt),
                        "potion": canonical_potion(pid),
                        "dv": v - v_cf,
                        "dv_hp10": dv_hp10,
                    }
                )
        d = env.step(max(range(len(probs)), key=probs.__getitem__))
    won = bool(getattr(d, "won", False))
    return rows, {"seed": seed, "won": won, "steps": steps}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--seed-base", type=int, default=5_000)
    ap.add_argument("--ascension", type=int, default=8)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument(
        "--act1",
        default="overgrowth",
        help="act-1 model: overgrowth | underdocks | roll (roll = the deployed mix)",
    )
    ap.add_argument("--out", type=str, default="../logs/2026-08-19-potion-oc")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    all_rows: list[dict] = []
    wins = 0
    jobs = [(args.seed_base + i, args.ascension, args.act1) for i in range(args.runs)]
    with (
        Pool(args.procs) as pool,
        open(out / "probe_critic_rows.jsonl", "w", encoding="utf-8") as sink,
    ):
        for i, (rows, summary) in enumerate(pool.imap_unordered(_one_run, jobs), 1):
            wins += summary["won"]
            all_rows.extend(rows)
            for r in rows:
                sink.write(json.dumps(r) + "\n")
            if i % 10 == 0:
                print(
                    f"[progress] {i}/{args.runs} runs, {len(all_rows)} rows, "
                    f"{time.time() - t0:.0f}s",
                    flush=True,
                )

    # ── aggregate: median dv per (potion, act); hp exchange per act ──
    def median(xs: list[float]) -> float:
        xs = sorted(xs)
        return xs[len(xs) // 2] if xs else 0.0

    by_cell: dict[tuple[str, int], list[float]] = defaultdict(list)
    hp_by_act: dict[int, list[float]] = defaultdict(list)
    for r in all_rows:
        by_cell[r["potion"], r["act"]].append(r["dv"])
        if r["dv_hp10"] is not None:
            hp_by_act[r["act"]].append(r["dv_hp10"])

    hp_per_point = {a: median(v) / 10.0 for a, v in hp_by_act.items()}
    entries = []
    for (pid, act), dvs in sorted(by_cell.items()):
        rate = hp_per_point.get(act, 0.0)
        oc = median(dvs) / rate if rate > 1e-9 else float("nan")
        entries.append(
            {
                "potion": pid,
                "act": act,
                "oc": round(max(oc, 0.0), 3) if oc == oc else None,
                "n": len(dvs),
                "median_dv": round(median(dvs), 6),
            }
        )

    # ── calibration checks (shape, per the plan's Task 5 step 3) ──
    pos = sum(1 for e in entries if e["n"] >= 20 and e["median_dv"] > 0)
    tot = sum(1 for e in entries if e["n"] >= 20)
    scaling_grad = [
        (
            pid,
            [
                e["median_dv"]
                for a in (1, 2, 3)
                for e in entries
                if e["potion"] == pid and e["act"] == a
            ],
        )
        for pid in sorted(SCALING)
    ]
    act3 = {e["potion"]: e["median_dv"] for e in entries if e["act"] == 3 and e["n"] >= 20}
    scale3 = [act3[p] for p in SCALING if p in act3]
    nuke3 = [act3[p] for p in NUKES if p in act3]
    calibration = {
        "cells_n20": tot,
        "positive_share": round(pos / tot, 3) if tot else None,
        "hp_per_value_point_by_act": {str(a): round(r, 5) for a, r in sorted(hp_per_point.items())},
        "scaling_act_gradient": {p: [round(x, 5) for x in g] for p, g in scaling_grad},
        "act3_scaling_median_dv": round(median(scale3), 6) if scale3 else None,
        "act3_nuke_median_dv": round(median(nuke3), 6) if nuke3 else None,
        "runs": args.runs,
        "ascension": args.ascension,
        "act1": args.act1,
        "wins": wins,
    }

    table = {
        "version": 1,
        "character": "ironclad",
        "units": "hp_equiv",
        "source": "critic_counterfactual",
        "default_oc": 8.0,
        "entries": [
            {"potion": e["potion"], "act": e["act"], "oc": e["oc"]}
            for e in entries
            if e["oc"] is not None and e["n"] >= 20
        ],
        "calibration": calibration,
        "raw_cells": entries,
    }
    (out / "probe_critic.json").write_text(json.dumps(table, indent=1) + "\n")
    print(json.dumps(calibration, indent=1))
    print(f"wrote {out / 'probe_critic.json'} ({len(table['entries'])} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
