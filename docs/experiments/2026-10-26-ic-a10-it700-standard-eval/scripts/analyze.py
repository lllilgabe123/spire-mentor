"""Primary analysis of the Ironclad a10_it700 standard strength eval (PREREG.md §Analysis 1-4).

Usage: python analyze.py [EXPERIMENT_DIR]   (default: the folder above this script;
       writes EXPERIMENT_DIR/analysis/primary.json)

1. Filter error and timeout rows before anything else; pair only seeds valid in both arms.
2. Playbook row = arm D's A10 win rate and A9 estimate (won or reached_second_boss is True),
   Wilson 95% CIs. Same definitions as docs/experiments/ironclad-a10/a10_score.py.
3. D vs P paired, descriptive: discordant counts b (D won, P lost) / c (P won, D lost),
   diff = (b - c) / n, exact two-sided McNemar p (binomial on b of b + c, p = 1/2),
   95% CI by Newcombe's hybrid score method for paired proportions (Newcombe 1998,
   method 10, no continuity correction); Wald CI on paired diffs reported alongside.
4. Cost: wall time per arm (driver.log), per-run wall_s from the rows, D/P ratios.

The `qualifier` strings and the host sentence in `cost.notes` describe the 2026-10-26 run. They are
written only when runs/D.jsonl and runs/P.jsonl are that run's rows (as run, or as published);
on any other rows the qualifier names the wheel the rows record instead.
"""

import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path

# As run, A was the private archive folder; it is now argv[1], else <experiment>/ above scripts/.
A = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
RUNS = A / "runs"
DRIVER_LOG = A / "logs" / "driver.log"
OUT = A / "analysis" / "primary.json"
SEEDS = set(range(91_000_000, 91_000_400))
Z = 1.959963984540054
BAD = ("error", "timeout")

QUAL = (
    "Ironclad A10, a10_it700 net_greedy (ppo_fullrun_ironclad_a10_it700.pt sha 91c83c1a), "
    "rl2 playout net, 4000/16000 sims, honest BeliefDraw, knobs removed_counts_forms=1,"
    "endturn_widening_c=23.148,draw_widening_c=4.0, act1 roll, seeds 91000000-91000399, "
    "PGO wheel src_hash 4e5bb730d28e2976"
)
# sha256 of the 2026-10-26 run's rows: as the run wrote them, and as published (so_path redacted).
RUN_2026_10_26 = {
    "D": {
        "6e15ab979b8171c03be324d6a56e17d2fcee9f85c34f0849dbfe4229f100420c",
        "e732981cd9c0585d1a4a4184a92f071375f2b8addd71386ef5e423390caa2903",
    },
    "P": {
        "f74084528d78718b262ffba9b0edf25b0a6189fc5deebe0d1063a94920c94f63",
        "8dad9c0cfbcf3cb3feac5430b8e322acdb12be901e8d5c1549853cd6b4583f4f",
    },
}
ARM_DESC = {
    "D": "shipped Ironclad default: knobs + built-in BC move prior in elite/boss (mix 0.5, T 1)",
    "P": "knobs only, STS2_RUNAI_PRIOR_NET=off (pre-ship setting, with rule 5c)",
}


def load_rows(arm: str) -> list[dict]:
    with open(RUNS / f"{arm}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def is_2026_10_26_run() -> bool:
    return all(
        hashlib.sha256((RUNS / f"{arm}.jsonl").read_bytes()).hexdigest() in shas
        for arm, shas in RUN_2026_10_26.items()
    )


def qualifier(rows: list[dict]) -> str:
    """QUAL for the 2026-10-26 rows; otherwise the same settings with the wheel(s) the rows record."""
    if is_2026_10_26_run():
        return QUAL
    wheels = sorted(
        {
            f"{'PGO' if w.get('pgo') else 'plain'} wheel src_hash {w.get('src_hash')}"
            for w in (r.get("wheel", {}) for r in rows)
        }
    )
    return (
        "Ironclad A10, a10_it700 net_greedy (the checkpoint given to the driver; the rows do not "
        "record it), rl2 playout net, 4000/16000 sims, honest BeliefDraw, knobs "
        "removed_counts_forms=1,endturn_widening_c=23.148,draw_widening_c=4.0, act1 roll, "
        "seeds 91000000-91000399, " + " + ".join(wheels)
    )


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / den
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / den
    return (centre - half, centre + half)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar: 2 * P(X <= min(b, c)), X ~ Binom(b + c, 1/2), capped at 1."""
    m = b + c
    if m == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(m, i) for i in range(k + 1)) / 2**m
    return min(1.0, 2 * tail)


def newcombe_paired(a: int, b: int, c: int, d: int) -> tuple[float, float]:
    """95% CI of p1 - p2 for paired binary data, Newcombe (1998) method 10.

    a = both win, b = arm1 only, c = arm2 only, d = both lose; p1 = (a+b)/n, p2 = (a+c)/n.
    """
    n = a + b + c + d
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n)
    l2, u2 = wilson(a + c, n)
    den = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    phi = (a * d - b * c) / den if den > 0 else 0.0
    diff = p1 - p2
    dl = math.sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    du = math.sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    return (diff - dl, diff + du)


def wald_paired(b: int, c: int, n: int) -> tuple[float, float, float]:
    diff = (b - c) / n
    se = math.sqrt(b + c - (b - c) ** 2 / n) / n
    return (diff - Z * se, diff + Z * se, se)


def scipy_binom_p(b: int, c: int) -> float | None:
    try:
        from scipy.stats import binomtest
    except ImportError:
        return None
    if b + c == 0:
        return 1.0
    return float(binomtest(b, b + c, 0.5, alternative="two-sided").pvalue)


def a9_est(r: dict) -> bool:
    return bool(r.get("won")) or r.get("reached_second_boss") is True


def parse_driver_log() -> dict:
    out: dict = {}
    with open(DRIVER_LOG, encoding="utf-8") as f:
        for line in f:
            m = re.search(r"arm (\w) start; load ([\d.]+) ([\d.]+) ([\d.]+)", line)
            if m:
                out.setdefault(m.group(1), {})["load_at_start_1_5_15min"] = [
                    float(m.group(i)) for i in (2, 3, 4)
                ]
                out[m.group(1)]["start"] = line.split()[0]
            m = re.search(r"arm (\w) end rc=(\d+) wall_s=(\d+) rows=(\d+)", line)
            if m:
                arm = out.setdefault(m.group(1), {})
                arm.update(
                    end=line.split()[0],
                    rc=int(m.group(2)),
                    wall_s=int(m.group(3)),
                    rows_logged=int(m.group(4)),
                )
    return out


def arm_summary(arm: str, rows: list[dict], qual: str) -> tuple[dict, dict[int, dict]]:
    seeds = [int(r["seed"]) for r in rows]
    errors = [r for r in rows if r.get("outcome") == "error" or "error" in r]
    timeouts = [r for r in rows if r.get("outcome") == "timeout"]
    valid = {int(r["seed"]): r for r in rows if r.get("outcome") not in BAD and "error" not in r}
    n = len(valid)
    w10 = sum(bool(r.get("won")) for r in valid.values())
    w9 = sum(a9_est(r) for r in valid.values())
    with open(RUNS / f"{arm}.exit", encoding="utf-8") as f:
        exit_code = f.read().strip()
    summ = {
        "arm": ARM_DESC[arm],
        "exit_code": exit_code,
        "rows_total": len(rows),
        "error_rows": len(errors),
        "timeout_rows": len(timeouts),
        "scored": n,
        "outcomes": {o: sum(r.get("outcome") == o for r in rows) for o in sorted({r.get("outcome") for r in rows})},
        "seed_check": {
            "duplicates": len(seeds) - len(set(seeds)),
            "missing": sorted(SEEDS - set(seeds)),
            "extra": sorted(set(seeds) - SEEDS),
        },
        "wheel_src_hash": sorted({r.get("wheel", {}).get("src_hash") for r in rows}),
        "wheel_pgo": sorted({str(r.get("wheel", {}).get("pgo")) for r in rows}),
        "py_src": sorted({r.get("py_src") for r in rows}),
        "act1_split": {a: sum(r.get("act1") == a for r in rows) for a in sorted({r.get("act1") for r in rows})},
        "A10": {
            "wins": w10,
            "n": n,
            "rate": w10 / n,
            "wilson95": list(wilson(w10, n)),
            "qualifier": f"A10 win rate, {qual}, arm {arm} ({ARM_DESC[arm]}), n={n}",
        },
        "A9est": {
            "reached": w9,
            "n": n,
            "rate": w9 / n,
            "wilson95": list(wilson(w9, n)),
            "definition": "won or reached_second_boss is True (entered the A10 DoubleBoss room)",
            "qualifier": f"A9 estimate read off A10 runs, {qual}, arm {arm} ({ARM_DESC[arm]}), n={n}",
        },
    }
    return summ, valid


def paired_block(dv: dict[int, dict], pv: dict[int, dict], seeds: list[int], key) -> dict:
    a = b = c = d = 0
    for s in seeds:
        x, y = key(dv[s]), key(pv[s])
        if x and y:
            a += 1
        elif x and not y:
            b += 1
        elif y and not x:
            c += 1
        else:
            d += 1
    n = len(seeds)
    lo, hi = newcombe_paired(a, b, c, d)
    wlo, whi, se = wald_paired(b, c, n)
    p = mcnemar_exact(b, c)
    return {
        "n": n,
        "both_win": a,
        "b_D_only": b,
        "c_P_only": c,
        "both_lose": d,
        "D_rate": (a + b) / n,
        "P_rate": (a + c) / n,
        "diff_pp": 100 * (b - c) / n,
        "mcnemar_exact_p_two_sided": p,
        "mcnemar_p_scipy_binomtest": scipy_binom_p(b, c),
        "ci95_pp_newcombe10": [100 * lo, 100 * hi],
        "ci95_pp_wald": [100 * wlo, 100 * whi],
        "se_pp_wald": 100 * se,
        "ci_method": "Newcombe (1998) method 10 hybrid score CI for paired proportions (primary); "
        "Wald on paired diffs alongside",
    }


def cost_block(d_rows: list[dict], p_rows: list[dict], seeds: list[int], log: dict) -> dict:
    out: dict = {}
    by = {"D": {int(r["seed"]): r for r in d_rows}, "P": {int(r["seed"]): r for r in p_rows}}
    for arm, rows in (("D", d_rows), ("P", p_rows)):
        walls = [r["wall_s"] for r in rows if "wall_s" in r]
        steps = sum(r["steps"] for r in rows)
        combat = sum((r.get("resolution_counts") or {}).get("combat/None/combat-agent", 0) for r in rows)
        out[arm] = {
            "arm_wall_s": log.get(arm, {}).get("wall_s"),
            "load_at_start_1_5_15min": log.get(arm, {}).get("load_at_start_1_5_15min"),
            "start": log.get(arm, {}).get("start"),
            "end": log.get(arm, {}).get("end"),
            "run_wall_s_mean": statistics.mean(walls),
            "run_wall_s_median": statistics.median(walls),
            "run_wall_s_max": max(walls),
            "run_wall_s_sum": sum(walls),
            "steps_sum": steps,
            "combat_decisions_sum": combat,
            "s_per_step": sum(walls) / steps,
            "s_per_combat_decision": sum(walls) / combat,
        }
    d, p = out["D"], out["P"]
    ratios = [by["D"][s]["wall_s"] / by["P"][s]["wall_s"] for s in seeds]
    out["ratio_D_over_P"] = {
        "arm_wall": d["arm_wall_s"] / p["arm_wall_s"],
        "run_wall_mean": d["run_wall_s_mean"] / p["run_wall_s_mean"],
        "run_wall_median": d["run_wall_s_median"] / p["run_wall_s_median"],
        "run_wall_sum (CPU proxy)": d["run_wall_s_sum"] / p["run_wall_s_sum"],
        "s_per_step": d["s_per_step"] / p["s_per_step"],
        "s_per_combat_decision": d["s_per_combat_decision"] / p["s_per_combat_decision"],
        "paired_per_seed_ratio_median": statistics.median(ratios),
        "paired_per_seed_ratio_geomean": math.exp(statistics.mean(math.log(x) for x in ratios)),
    }
    host = (
        "The box is shared (48 CPUs, 30 procs per arm); load at P's start includes D's own tail in "
        "the 5/15-min averages. "
        if is_2026_10_26_run()
        else ""
    )
    out["notes"] = (
        "Per-run wall_s is one worker process (OMP/MKL/OpenBLAS threads 1), so its sum is the CPU "
        "proxy. D runs longer on average because it survives longer, so s_per_step and "
        "s_per_combat_decision are the length-adjusted cost. " + host + "Arms ran in sequence, "
        "not interleaved, so a load change between them confounds the time ratio."
    )
    return out


def divergence_check(dv: dict[int, dict], pv: dict[int, dict], seeds: list[int]) -> dict:
    """Weak sanity check that P's env change took effect: identical trajectories on every seed would
    say it did not. Not proof of the prior being off (rows do not record the env)."""
    same = sum(
        (dv[s]["outcome"], dv[s]["steps"], dv[s]["final"]) == (pv[s]["outcome"], pv[s]["steps"], pv[s]["final"])
        for s in seeds
    )
    return {"seeds_identical_outcome_steps_final": same, "n": len(seeds)}


def main() -> None:
    d_rows, p_rows = load_rows("D"), load_rows("P")
    qual = qualifier(d_rows + p_rows)
    d_sum, dv = arm_summary("D", d_rows, qual)
    p_sum, pv = arm_summary("P", p_rows, qual)
    seeds = sorted(dv.keys() & pv.keys())
    log = parse_driver_log()
    result = {
        "prereg": str(A / "PREREG.md"),
        "status": "descriptive, NOT a gate: n=400 at 4x sims; the ship rests on the two 1000/4000 A10 gates",
        "qualifier": qual,
        "arms": {"D": d_sum, "P": p_sum},
        "playbook_row": {
            "A10": d_sum["A10"],
            "A9est": d_sum["A9est"],
            "note": "arm D = the shipped Ironclad default; this is the playbook §2.6 row",
        },
        "paired_D_minus_P": {
            "paired_seeds": len(seeds),
            "dropped_seeds": len({int(r["seed"]) for r in d_rows + p_rows}) - len(seeds),
            "A10": paired_block(dv, pv, seeds, lambda r: bool(r.get("won"))),
            "A9est": paired_block(dv, pv, seeds, a9_est),
            "arm_divergence_check": divergence_check(dv, pv, seeds),
        },
        "cost": cost_block(d_rows, p_rows, seeds, log),
        "old_rows_quoted_not_paired": {
            "plr_it550 (2026-10-21)": {"A10": 0.170, "A9est": 0.452},
            "a10_it700 pre-bundle, no prior (2026-10-23)": {"A10": 0.230, "A9est": 0.498},
            "note": "other host, older wheels; per-seed rows not on this host, so not pairable",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
