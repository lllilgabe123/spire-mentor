"""Independent recompute from the raw rows, to reconcile analyze.py and recompute_blind.py.

Reads runs/{D,P}.jsonl read-only. Prints one JSON object to stdout.

Usage: python reconcile.py [EXPERIMENT_DIR]   (default: the folder above this script)

The `reconciliation` block (NARRATIVE below) is the written reconciliation of the 2026-10-26
run. It is written only when runs/D.jsonl and runs/P.jsonl are that run's rows (as run, or as
published); on any other rows it is replaced by a note, and `comparison_mismatches` is the check.
"""

import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

# The private archive kept the scripts in analysis/scripts/ (parents[2]); here they sit in scripts/.
A = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
Z = 1.959963984540054
SEEDS = set(range(91_000_000, 91_000_400))


def load(arm):
    rows = []
    with open(A / "runs" / f"{arm}.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def bad(r):
    return (
        r.get("outcome") not in ("won", "lost")
        or r.get("error") is not None
        or "timeout" in r
        or "age_s" in r
    )


def wilson(k, n):
    p = k / n
    den = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / den
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / den
    return [c - h, c + h]


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = Fraction(sum(math.comb(n, i) for i in range(k + 1)), 2**n)
    return float(min(Fraction(1), 2 * tail))


def newcombe10(a, b, c, d):
    """Paired diff p1-p2, p1 = (a+b)/n (D), p2 = (a+c)/n (P). Newcombe 1998 method 10."""
    n = a + b + c + d
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n)
    l2, u2 = wilson(a + c, n)
    den = (a + b) * (c + d) * (a + c) * (b + d)
    phi = 0.0 if den == 0 else (a * d - b * c) / math.sqrt(den)
    dl = math.sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    du = math.sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    return [p1 - p2 - dl, p1 - p2 + du], phi


def wald_paired(b, c, n):
    diff = (b - c) / n
    se = math.sqrt((b + c) / n - diff**2) / math.sqrt(n)
    return [diff - Z * se, diff + Z * se], se


def a9(r):
    return bool(r["won"]) or r.get("reached_second_boss") is True


def where(r):
    if r["won"]:
        return "won"
    ai = r["final"]["act_index"]
    if ai < 3:
        return f"act{ai}"
    return "act3_reached_boss2" if r.get("reached_second_boss") is True else "act3_before_boss2"


def main():
    out = {"rows": {}, "arms": {}, "paired": {}, "checks": {}, "cost": {}}
    arms = {a: load(a) for a in "DP"}
    valid = {}
    for a, rows in arms.items():
        seeds = [int(r["seed"]) for r in rows]
        sc = Counter(seeds)
        nbad = sum(bad(r) for r in rows)
        out["rows"][a] = {
            "total": len(rows),
            "error": sum(r.get("outcome") == "error" or r.get("error") is not None for r in rows),
            "timeout": sum(r.get("outcome") == "timeout" or "timeout" in r for r in rows),
            "dropped": nbad,
            "scored": len(rows) - nbad,
            "dup_seeds": sum(v > 1 for v in sc.values()),
            "missing_seeds": len(SEEDS - set(seeds)),
            "extra_seeds": len(set(seeds) - SEEDS),
            "exit": (A / "runs" / f"{a}.exit").read_text(encoding="utf-8").strip(),
            "outcomes": dict(Counter(r["outcome"] for r in rows)),
            "wheel_src_hash": sorted({r["wheel"]["src_hash"] for r in rows}),
            "wheel_pgo": sorted({r["wheel"]["pgo"] for r in rows}),
            "so_path": sorted({r["wheel"]["so_path"] for r in rows}),
            "py_src": sorted({r["py_src"] for r in rows}),
            "policy": sorted({r["policy"] for r in rows}),
            "asc": sorted({r["ascension"] for r in rows}),
            "character": sorted({r["character"] for r in rows}),
            "decisions_nonempty": sum(bool(r["decisions"]) for r in rows),
        }
        v = {int(r["seed"]): r for r in rows if not bad(r)}
        valid[a] = v
        n = len(v)
        w = sum(bool(r["won"]) for r in v.values())
        w9 = sum(a9(r) for r in v.values())
        rsb = Counter(
            (r["final"]["act_index"], r.get("reached_second_boss"), bool(r["won"])) for r in v.values()
        )
        out["arms"][a] = {
            "A10": {"wins": w, "n": n, "rate": w / n, "wilson95": wilson(w, n)},
            "A9est": {"reached": w9, "n": n, "rate": w9 / n, "wilson95": wilson(w9, n)},
            "losses_by_where": dict(Counter(where(r) for r in v.values())),
            "by_act1": {
                k: {
                    "n": sum(r["act1"] == k for r in v.values()),
                    "A10_wins": sum(r["act1"] == k and r["won"] for r in v.values()),
                    "A9est": sum(r["act1"] == k and a9(r) for r in v.values()),
                }
                for k in sorted({r["act1"] for r in v.values()})
            },
            "rsb_by_act_won": {str(k): c for k, c in sorted(rsb.items(), key=str)},
        }
    D, P = valid["D"], valid["P"]
    common = sorted(D.keys() & P.keys())
    out["rows"]["paired_n"] = len(common)
    for name, f in (("A10", lambda r: bool(r["won"])), ("A9est", a9)):
        a = sum(f(D[s]) and f(P[s]) for s in common)
        b = sum(f(D[s]) and not f(P[s]) for s in common)
        c = sum(f(P[s]) and not f(D[s]) for s in common)
        d = sum(not f(D[s]) and not f(P[s]) for s in common)
        n = len(common)
        ci, phi = newcombe10(a, b, c, d)
        wci, se = wald_paired(b, c, n)
        out["paired"][name] = {
            "n": n, "both": a, "b_D_only": b, "c_P_only": c, "neither": d,
            "diff_pp": 100 * (b - c) / n,
            "mcnemar_exact_p_two_sided": mcnemar_exact(b, c),
            "ci95_pp_newcombe10": [100 * x for x in ci],
            "phi": phi,
            "ci95_pp_wald": [100 * x for x in wci],
            "se_pp_wald": 100 * se,
        }
    try:
        from scipy.stats import binomtest

        for name in ("A10", "A9est"):
            p = out["paired"][name]
            p["scipy_binomtest_p"] = binomtest(p["b_D_only"], p["b_D_only"] + p["c_P_only"], 0.5).pvalue
    except ImportError:
        pass

    # --- evidence checks ---
    act_mis = [s for s in common if D[s]["act1"] != P[s]["act1"]]
    same_outcome = sum(D[s]["won"] == P[s]["won"] for s in common)
    same_a9 = sum(a9(D[s]) == a9(P[s]) for s in common)

    def traj(r):
        return (r["outcome"], r["steps"], r["n_decisions"], json.dumps(r["final"], sort_keys=True),
                json.dumps(r["resolution_counts"], sort_keys=True))

    identical = [s for s in common if traj(D[s]) == traj(P[s])]
    diff_steps = sum(D[s]["steps"] != P[s]["steps"] for s in common)
    diff_final = sum(D[s]["final"] != P[s]["final"] for s in common)
    diff_floor = sum(D[s]["final"]["total_floor"] != P[s]["final"]["total_floor"] for s in common)
    diff_hp = sum(D[s]["final"]["hp"] != P[s]["final"]["hp"] for s in common)
    diff_res = sum(D[s]["resolution_counts"] != P[s]["resolution_counts"] for s in common)
    # early deaths (both arms lost on the same act-1 floor <= 10): if search is seed-deterministic and
    # the prior acts only in elite/boss rooms, runs that never met an elite/boss should replay.
    early = []
    for s in common:
        fd, fp = D[s]["final"], P[s]["final"]
        if min(fd["total_floor"], fp["total_floor"]) <= 10 and fd["act_index"] == 1 and fp["act_index"] == 1:
            early.append({
                "seed": s, "D_floor": fd["total_floor"], "P_floor": fp["total_floor"],
                "D_steps": D[s]["steps"], "P_steps": P[s]["steps"],
                "D_wall": D[s]["wall_s"], "P_wall": P[s]["wall_s"], "identical": s in identical,
            })
    out["checks"] = {
        "act1_mismatch_seeds": act_mis,
        "act1_split": {a: dict(Counter(r["act1"] for r in valid[a].values())) for a in "DP"},
        "same_A10_outcome": same_outcome,
        "diff_A10_outcome": len(common) - same_outcome,
        "same_A9est": same_a9,
        "identical_trajectory_seeds": identical,
        "n_identical": len(identical),
        "diff_any_of_steps_final_rescounts": len(common) - len(identical),
        "diff_steps": diff_steps,
        "diff_final": diff_final,
        "diff_total_floor": diff_floor,
        "diff_final_hp": diff_hp,
        "diff_resolution_counts": diff_res,
        "early_act1_deaths_floor_le_10": sorted(early, key=lambda e: e["seed"]),
    }

    # --- cost ---
    for a in "DP":
        rs = list(valid[a].values())
        ws = [r["wall_s"] for r in rs]
        st = [r["steps"] for r in rs]
        cd = [r["resolution_counts"].get("combat/None/combat-agent", 0) for r in rs]
        out["cost"][a] = {
            "run_wall_s_mean": statistics.fmean(ws),
            "run_wall_s_median": statistics.median(ws),
            "run_wall_s_max": max(ws),
            "run_wall_s_sum": sum(ws),
            "s_per_step_ratio_of_sums": sum(ws) / sum(st),
            "s_per_step_mean_of_ratios": statistics.fmean(w / s for w, s in zip(ws, st)),
            "s_per_combat_decision_ratio_of_sums": sum(ws) / sum(cd),
        }
    rat = [D[s]["wall_s"] / P[s]["wall_s"] for s in common]
    cD, cP = out["cost"]["D"], out["cost"]["P"]
    out["cost"]["ratio_D_over_P"] = {
        "run_wall_mean": cD["run_wall_s_mean"] / cP["run_wall_s_mean"],
        "run_wall_median": cD["run_wall_s_median"] / cP["run_wall_s_median"],
        "run_wall_sum": cD["run_wall_s_sum"] / cP["run_wall_s_sum"],
        "s_per_step_ratio_of_sums": cD["s_per_step_ratio_of_sums"] / cP["s_per_step_ratio_of_sums"],
        "s_per_step_mean_of_ratios": cD["s_per_step_mean_of_ratios"] / cP["s_per_step_mean_of_ratios"],
        "s_per_combat_decision": cD["s_per_combat_decision_ratio_of_sums"] / cP["s_per_combat_decision_ratio_of_sums"],
        "paired_per_seed_median": statistics.median(rat),
        "paired_per_seed_geomean": math.exp(statistics.fmean(math.log(x) for x in rat)),
    }
    sf = [s for s in common if D[s]["final"]["total_floor"] == P[s]["final"]["total_floor"]]
    out["cost"]["ratio_D_over_P"]["paired_same_final_floor_n"] = len(sf)
    out["cost"]["ratio_D_over_P"]["paired_same_final_floor_geomean"] = math.exp(
        statistics.fmean(math.log(D[s]["wall_s"] / P[s]["wall_s"]) for s in sf)
    )
    # act-3 losses before the second boss room: how many ended on the first act-3 boss floor
    for a in "DP":
        rs = [r for r in valid[a].values() if not r["won"] and r["final"]["act_index"] == 3
              and r.get("reached_second_boss") is False]
        out["arms"][a]["act3_before_boss2_at_act_floor_15"] = sum(r["final"]["act_floor"] == 15 for r in rs)
        out["arms"][a]["act3_before_boss2_earlier"] = sum(r["final"]["act_floor"] < 15 for r in rs)
    return out


def get(d, path):
    for k in path.split("/"):
        d = d[k] if not isinstance(d, list) else d[int(k)]
    return d


# (label, path in this recompute, path in primary.json, path in recompute_blind.json)
COMPARE = [
    ("D rows total", "rows/D/total", "arms/D/rows_total", "rows/D/total"),
    ("D error rows", "rows/D/error", "arms/D/error_rows", "rows/D/error"),
    ("D timeout rows", "rows/D/timeout", "arms/D/timeout_rows", "rows/D/timeout"),
    ("D scored", "rows/D/scored", "arms/D/scored", "rows/D/scored"),
    ("P rows total", "rows/P/total", "arms/P/rows_total", "rows/P/total"),
    ("P error rows", "rows/P/error", "arms/P/error_rows", "rows/P/error"),
    ("P timeout rows", "rows/P/timeout", "arms/P/timeout_rows", "rows/P/timeout"),
    ("P scored", "rows/P/scored", "arms/P/scored", "rows/P/scored"),
    ("paired n", "rows/paired_n", "paired_D_minus_P/paired_seeds", "rows/paired_n"),
]
for arm in "DP":
    for m in ("A10", "A9est"):
        k = "wins" if m == "A10" else "reached"
        COMPARE += [
            (f"{arm} {m} {k}", f"arms/{arm}/{m}/{k}", f"arms/{arm}/{m}/{k}", f"scores/{arm}/{m}/{k}"),
            (f"{arm} {m} rate", f"arms/{arm}/{m}/rate", f"arms/{arm}/{m}/rate", f"scores/{arm}/{m}/rate"),
            (f"{arm} {m} Wilson lo", f"arms/{arm}/{m}/wilson95/0", f"arms/{arm}/{m}/wilson95/0", f"scores/{arm}/{m}/wilson95/0"),
            (f"{arm} {m} Wilson hi", f"arms/{arm}/{m}/wilson95/1", f"arms/{arm}/{m}/wilson95/1", f"scores/{arm}/{m}/wilson95/1"),
        ]
for m in ("A10", "A9est"):
    pp, pb = f"paired_D_minus_P/{m}", f"paired/{m}"
    COMPARE += [
        (f"paired {m} both", f"paired/{m}/both", f"{pp}/both_win", f"{pb}/both"),
        (f"paired {m} b (D only)", f"paired/{m}/b_D_only", f"{pp}/b_D_only", f"{pb}/D_only_b"),
        (f"paired {m} c (P only)", f"paired/{m}/c_P_only", f"{pp}/c_P_only", f"{pb}/P_only_c"),
        (f"paired {m} neither", f"paired/{m}/neither", f"{pp}/both_lose", f"{pb}/neither"),
        (f"paired {m} diff pp", f"paired/{m}/diff_pp", f"{pp}/diff_pp", f"{pb}/diff_pp"),
        (f"paired {m} McNemar p", f"paired/{m}/mcnemar_exact_p_two_sided", f"{pp}/mcnemar_exact_p_two_sided", f"{pb}/mcnemar_exact_p_two_sided"),
        (f"paired {m} Newcombe10 lo", f"paired/{m}/ci95_pp_newcombe10/0", f"{pp}/ci95_pp_newcombe10/0", f"{pb}/ci95_pp_newcombe10/0"),
        (f"paired {m} Newcombe10 hi", f"paired/{m}/ci95_pp_newcombe10/1", f"{pp}/ci95_pp_newcombe10/1", f"{pb}/ci95_pp_newcombe10/1"),
        (f"paired {m} Wald lo", f"paired/{m}/ci95_pp_wald/0", f"{pp}/ci95_pp_wald/0", f"{pb}/ci95_pp_wald/0"),
        (f"paired {m} Wald hi", f"paired/{m}/ci95_pp_wald/1", f"{pp}/ci95_pp_wald/1", f"{pb}/ci95_pp_wald/1"),
    ]
for arm in "DP":
    COMPARE += [
        (f"{arm} run wall_s mean", f"cost/{arm}/run_wall_s_mean", f"cost/{arm}/run_wall_s_mean", f"cost/per_run_wall_s/{arm}/mean"),
        (f"{arm} run wall_s median", f"cost/{arm}/run_wall_s_median", f"cost/{arm}/run_wall_s_median", f"cost/per_run_wall_s/{arm}/median"),
        (f"{arm} run wall_s max", f"cost/{arm}/run_wall_s_max", f"cost/{arm}/run_wall_s_max", f"cost/per_run_wall_s/{arm}/max"),
        (f"{arm} run wall_s sum", f"cost/{arm}/run_wall_s_sum", f"cost/{arm}/run_wall_s_sum", f"cost/per_run_wall_s/{arm}/sum"),
    ]
COMPARE += [
    ("ratio run wall mean D/P", "cost/ratio_D_over_P/run_wall_mean", "cost/ratio_D_over_P/run_wall_mean", "cost/per_run_mean_ratio_D_over_P"),
    ("ratio run wall median D/P", "cost/ratio_D_over_P/run_wall_median", "cost/ratio_D_over_P/run_wall_median", "cost/per_run_median_ratio_D_over_P"),
    ("ratio paired per-seed median", "cost/ratio_D_over_P/paired_per_seed_median", "cost/ratio_D_over_P/paired_per_seed_ratio_median", "cost/paired_per_seed_ratio_D_over_P/median"),
    ("ratio paired per-seed geomean", "cost/ratio_D_over_P/paired_per_seed_geomean", "cost/ratio_D_over_P/paired_per_seed_ratio_geomean", "cost/paired_per_seed_ratio_D_over_P/geomean"),
    ("ratio same-final-floor geomean", "cost/ratio_D_over_P/paired_same_final_floor_geomean", None, "cost/paired_ratio_same_final_floor/geomean"),
    ("D s/step (primary: ratio of sums)", "cost/D/s_per_step_ratio_of_sums", "cost/D/s_per_step", None),
    ("P s/step (primary: ratio of sums)", "cost/P/s_per_step_ratio_of_sums", "cost/P/s_per_step", None),
    ("D s/step (blind: mean of per-run ratios)", "cost/D/s_per_step_mean_of_ratios", None, "cost/per_step_s_mean/D"),
    ("P s/step (blind: mean of per-run ratios)", "cost/P/s_per_step_mean_of_ratios", None, "cost/per_step_s_mean/P"),
]


def rel_ok(x, y):
    if x is None or y is None:
        return None
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        # the analyses round some floats when they store them (e.g. 160.52, 64209.1)
        return abs(x - y) <= max(1e-9, 1e-3 * abs(x), 0.06 if abs(x) > 10 else 0)
    return x == y


def compare(out):
    prim = json.loads((A / "analysis" / "primary.json").read_text(encoding="utf-8"))
    blind = json.loads((A / "analysis" / "recompute_blind.json").read_text(encoding="utf-8"))
    res = []
    for label, pm, pp, pb in COMPARE:
        mine = get(out, pm)
        vp = None
        vb = None
        try:
            vp = get(prim, pp) if pp else None
        except (KeyError, IndexError, TypeError):
            vp = "MISSING"
        try:
            vb = get(blind, pb) if pb else None
        except (KeyError, IndexError, TypeError):
            vb = "MISSING"
        res.append({
            "what": label, "raw_recompute": mine, "primary": vp, "blind": vb,
            "primary_match": rel_ok(mine, vp) if vp != "MISSING" else False,
            "blind_match": rel_ok(mine, vb) if vb != "MISSING" else False,
        })
    return res


NARRATIVE = {
    "verdict": (
        "AGREE. Every number the two analyses both report matches the raw rows (counts exactly, "
        "floats to full precision or to the analysis' own rounding). No number needed resolving."
    ),
    "differences_resolved": [
        {
            "item": "seconds per step",
            "primary": "D 0.3191, P 0.3070, ratio 1.039",
            "blind": "D 0.3006, P 0.2881, ratio 1.043",
            "resolution": (
                "Both are right; the definitions differ. Primary divides total wall_s by total steps "
                "(ratio of sums, weights long runs). Blind averages each run's wall_s/steps (mean of "
                "ratios). The raw rows reproduce both."
            ),
        },
        {
            "item": "primary's reading: D-P '+2.75pp ... sits inside their range' (the two 1000/4000 gates)",
            "resolution": (
                "Wrong as worded, and the contrast differs. +2.75pp is below both gate point estimates "
                "(+3.45pp, +5.20pp). The gates compared the bundle (knobs + prior) with the plain control "
                "(neither), and the gate write-up says it had no knobs-only or prior-only arm. D-P here is "
                "the prior alone on top of the knobs, at 4000/16000. So the gates give no direct "
                "reference for D-P. Nothing numeric changes; final.txt drops the claim."
            ),
        },
        {
            "item": "primary's note: '84 died to the first act-3 boss' (D; P 82)",
            "resolution": (
                "Slight over-statement. These are act-3 losses with reached_second_boss False. In D, 68 "
                "ended on act floor 15 (the first boss floor) and 16 earlier in act 3; in P, 70 and 12. "
                "Blind's label 'act3_before_boss2' is the exact one. No score uses this split."
            ),
        },
        {
            "item": "primary's caveat: 'if search is not bit-deterministic it does not prove the prior was off'",
            "resolution": (
                "Resolved by outside evidence: the ship identity record (docs/experiments/"
                "2026-10-25-ic-bundle-ship-identity.md §12) shows that on this same PGO .so (823fa393...) "
                "Ironclad full runs replay seed-for-seed (51 of 51 identity checks, 0 differences, "
                "including a PRIOR_NET=off arm that reproduces its prior-off reference). That was at "
                "1000/4000 on seeds 96,509,000-043, not at 4000/16000. Given that, a no-op off switch "
                "would have made P replay D on all 400 seeds."
            ),
        },
    ],
    "evidence_checks": {
        "a_prior_on_in_D_off_in_P": {
            "result": "PASS (indirect; no row or log records the prior state)",
            "detail": (
                "399 of 400 seeds differ between D and P in steps, final state or resolution counts "
                "(398 in steps, 379 in final state, 169 in final floor, 144 in final hp, 399 in "
                "resolution counts); 85 differ in A10 outcome, 100 in the A9 estimate. The one identical "
                "seed, 91000303, is an 84-step act-1 death on floor 8 (wall 7.04 s vs 7.02 s), the kind "
                "of run that can end before any elite or boss room, where the prior never acts. Runs "
                "are seed-deterministic on this .so (ship identity record), so if STS2_RUNAI_PRIOR_NET=off "
                "had not engaged, or the wheel had no built-in prior, P would equal D on every seed. The "
                "wheel does carry the prior: builtin_prior_net_json('Ironclad') returns 268,757 bytes, "
                "sha256 ca4692595c22...d4f2, and None for the other four characters (checked now; .so "
                "sha256 still 823fa393...). Which arm had the prior rests on run_eval.sh: D gets no "
                "extra env, P gets `env STS2_RUNAI_PRIOR_NET=off`. A labelling swap would leave the "
                "discordance unchanged, so it is not ruled out by the rows; the per-step cost (D 1.04x "
                "P) is in the direction the prior's extra work predicts, but that is weak."
            ),
            "logs": (
                "logs/D.stdout and logs/P.stdout contain no line naming the prior or any STS2_* env; "
                "both stderr files are 0 bytes. driver.log's env line for P does not show "
                "STS2_RUNAI_PRIOR_NET because `env` applies it to the child after the line is logged. "
                "Both stdouts report 'G1b ownership audit: PASS'."
            ),
        },
        "b_act_per_seed": "PASS: act1 equal on all 400 seeds; both arms overgrowth 189 / underdocks 211.",
        "c_error_rows": (
            "PASS: 0 error and 0 timeout rows in either arm (no 'error' key, no non won/lost outcome); "
            "both .exit files 0; seeds exactly 91,000,000-91,000,399 in each arm, no duplicates. The "
            "filter was a no-op, so scored n = 400 per arm and paired n = 400."
        ),
        "d_D_plausible": (
            "PASS (no implausibility): D 26.75% A10 / 52.25% A9 estimate sits above P (24.00% / 49.75%) "
            "and above the old pre-bundle no-prior row (23.0% / 49.8%, other host, older wheel without "
            "rule 5c, unpaired). P on this wheel lands within 1pp of that old row, which is the "
            "same knobs-only setting. D's D-P direction matches the gates' bundle-vs-control direction, "
            "but the contrasts differ (see differences_resolved)."
        ),
    },
}


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
OTHER_ROWS_NOTE = {
    "note": (
        "These are not the 2026-10-26 run's rows, so its written reconciliation (verdict, resolved "
        "differences, evidence checks) is not reproduced here. comparison_mismatches lists every "
        "value on which this recompute, analyze.py and recompute_blind.py disagree."
    )
}


def is_2026_10_26_run():
    return all(
        hashlib.sha256((A / "runs" / f"{arm}.jsonl").read_bytes()).hexdigest() in shas
        for arm, shas in RUN_2026_10_26.items()
    )


if __name__ == "__main__":
    o = main()
    o["comparison"] = compare(o)
    o["comparison_mismatches"] = [
        r for r in o["comparison"] if r["primary_match"] is False or r["blind_match"] is False
    ]
    rec = {"reconciliation": NARRATIVE if is_2026_10_26_run() else OTHER_ROWS_NOTE, **o}
    dest = A / "analysis" / "reconcile.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1, default=str)
        f.write("\n")
    print(f"wrote {dest}: {len(o['comparison'])} compared, {len(o['comparison_mismatches'])} mismatches")
    for r in o["comparison_mismatches"]:
        print(r)
