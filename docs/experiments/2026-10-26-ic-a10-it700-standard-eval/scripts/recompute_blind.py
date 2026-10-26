"""Blind independent recompute of the 2026-10-26 Ironclad a10_it700 standard eval.

Works from the raw rows only (runs/D.jsonl, runs/P.jsonl) plus the driver log for
arm wall time. Does not read any other analysis file and does not call
a10_score.py. The analysis follows PREREG.md:

1. drop error/timeout rows first; pair only seeds valid in both arms;
2. per-arm A10 win rate and A9 estimate with Wilson 95% CIs;
3. paired D-P difference (b, c, exact two-sided McNemar p, 95% CI) for both;
4. cost: arm wall time and per-run wall time.

Schema notes (worked out from the rows and from the producer,
python/ppo/workers.py `_reached_second_boss`): `reached_second_boss` is True when
the run entered the act-3 second-boss room, False when it was in act 3 but did
not, and None when the run ended before act 3 (the map has no second boss yet).
The A9 estimate is therefore `won or reached_second_boss is True`.

Usage: python recompute_blind.py [EXPERIMENT_DIR]
"""

from __future__ import annotations

import json
import math
import re
import statistics
import sys
from fractions import Fraction
from pathlib import Path

ARMS = ("D", "P")
SEED_START = 91_000_000
N_SEEDS = 400
EXPECTED_SEEDS = set(range(SEED_START, SEED_START + N_SEEDS))
Z95 = 1.959963984540054
VALID_OUTCOMES = {"won", "lost"}
# workers.DEFAULT_GAME_TIMEOUT_S: a run older than this is a watchdog candidate.
GAME_TIMEOUT_S = 1800.0


# ---------------------------------------------------------------- statistics
def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def exact_mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar p: binomial test of b out of b+c at 0.5,
    doubling the smaller tail (symmetric null), computed with exact fractions."""
    m = b + c
    if m == 0:
        return 1.0
    k = min(b, c)
    tail = sum(Fraction(math.comb(m, i)) for i in range(k + 1)) / Fraction(2**m)
    return float(min(Fraction(1), 2 * tail))


def scipy_mcnemar_p(b: int, c: int) -> float | None:
    try:
        from scipy.stats import binomtest
    except Exception:
        return None
    if b + c == 0:
        return 1.0
    return float(binomtest(b, b + c, 0.5, alternative="two-sided").pvalue)


def newcombe10(a: int, b: int, c: int, d: int) -> tuple[float, float]:
    """Newcombe (1998) method 10: hybrid Wilson score CI for the difference of
    two paired proportions p1-p2, p1=(a+b)/n, p2=(a+c)/n (no continuity corr.)."""
    n = a + b + c + d
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n)
    l2, u2 = wilson(a + c, n)
    den = (a + b) * (c + d) * (a + c) * (b + d)
    phi = (a * d - b * c) / math.sqrt(den) if den > 0 else 0.0
    delta = p1 - p2
    lo = delta - math.sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    hi = delta + math.sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    return (lo, hi)


def wald_paired(b: int, c: int, n: int) -> tuple[float, float]:
    delta = (b - c) / n
    se = math.sqrt(max(0.0, (b + c) - (b - c) ** 2 / n)) / n
    return (delta - Z95 * se, delta + Z95 * se)


def agresti_min(a: int, b: int, c: int, d: int) -> tuple[float, float]:
    """Agresti & Min (2005) adjusted Wald: add 0.5 to every cell."""
    a2, b2, c2, d2 = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    n = a2 + b2 + c2 + d2
    delta = (b2 - c2) / n
    se = math.sqrt((b2 + c2) - (b2 - c2) ** 2 / n) / n
    return (delta - Z95 * se, delta + Z95 * se)


# ---------------------------------------------------------------- data
def load(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            r["_line"] = i
            rows.append(r)
    return rows


def is_invalid(r: dict) -> str | None:
    """Reason a row must be dropped before scoring, or None if it is scorable."""
    if r.get("outcome") in ("error", "timeout"):
        return str(r.get("outcome"))
    if r.get("error") is not None:
        return "error-field"
    if r.get("timeout") is not None or "age_s" in r:
        return "timeout-field"
    if r.get("outcome") not in VALID_OUTCOMES:
        return f"unknown-outcome:{r.get('outcome')!r}"
    return None


def a9_reached(r: dict) -> bool:
    return bool(r.get("won")) or r.get("reached_second_boss") is True


def describe(xs: list[float]) -> dict:
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return {}
    q = statistics.quantiles(xs, n=20) if n > 1 else [xs[0]] * 19
    return {
        "n": n,
        "min": round(xs[0], 2),
        "p05": round(q[0], 2),
        "median": round(statistics.median(xs), 2),
        "mean": round(statistics.fmean(xs), 2),
        "p95": round(q[18], 2),
        "max": round(xs[-1], 2),
        "sum": round(sum(xs), 1),
    }


def arm_checks(arm: str, rows: list[dict]) -> dict:
    out: dict = {"rows": len(rows)}
    seeds = [r.get("seed") for r in rows]
    counts: dict = {}
    for s in seeds:
        counts[s] = counts.get(s, 0) + 1
    out["duplicate_seeds"] = sorted(s for s, k in counts.items() if k > 1)
    out["missing_seeds"] = sorted(EXPECTED_SEEDS - set(seeds))
    out["unexpected_seeds"] = sorted(set(seeds) - EXPECTED_SEEDS)

    invalid = [(r["seed"], is_invalid(r)) for r in rows if is_invalid(r)]
    out["invalid_rows"] = invalid
    out["error_count"] = sum(1 for _, why in invalid if why in ("error", "error-field"))
    out["timeout_count"] = sum(1 for _, why in invalid if why in ("timeout", "timeout-field"))
    out["error_field_but_won_or_lost"] = [
        r["seed"] for r in rows if r.get("error") is not None and r.get("outcome") in VALID_OUTCOMES
    ]

    # constant-field audit
    const = {}
    for f in ("policy", "ascension", "character", "py_src"):
        const[f] = sorted({json.dumps(r.get(f)) for r in rows})
    const["wheel"] = sorted({json.dumps(r.get("wheel"), sort_keys=True) for r in rows})
    out["distinct_values"] = {k: (v if len(v) <= 3 else [*v[:3], "..."]) for k, v in const.items()}

    # internal consistency
    incons = []
    for r in rows:
        if is_invalid(r):
            continue
        fin = r.get("final") or {}
        won, outcome, rsb = r.get("won"), r.get("outcome"), r.get("reached_second_boss")
        act = fin.get("act_index")
        why = []
        if bool(won) != (outcome == "won"):
            why.append("won!=outcome")
        if won and (act != 3 or rsb is not True or not fin.get("hp", 0) > 0):
            why.append("won-but-not-act3/rsb/hp>0")
        if outcome == "lost" and fin.get("hp") != 0:
            why.append(f"lost-with-hp={fin.get('hp')}")
        if (rsb is None) != (act is not None and act < 3):
            why.append(f"rsb={rsb}-act={act}")
        rc = r.get("resolution_counts")
        if isinstance(rc, dict) and sum(rc.values()) != r.get("steps"):
            why.append(f"resolution_sum={sum(rc.values())}!=steps={r.get('steps')}")
        if r.get("act1") not in ("overgrowth", "underdocks"):
            why.append(f"act1={r.get('act1')!r}")
        if why:
            incons.append({"seed": r["seed"], "why": why})
    out["inconsistent_rows"] = incons
    out["decisions_nonempty"] = sum(1 for r in rows if r.get("decisions"))

    # durations
    walls = [r["wall_s"] for r in rows if isinstance(r.get("wall_s"), (int, float))]
    out["wall_s"] = describe(walls)
    med = statistics.median(walls)
    per_step = [r["wall_s"] / r["steps"] for r in rows if r.get("steps")]
    ps_med = statistics.median(per_step)
    ps_mad = statistics.median(abs(x - ps_med) for x in per_step) or 1e-9
    sus = []
    for r in rows:
        w = r.get("wall_s")
        reasons = []
        if not isinstance(w, (int, float)) or w <= 0:
            reasons.append("missing-or-nonpositive")
        else:
            if w >= GAME_TIMEOUT_S:
                reasons.append(">=game_timeout")
            if w > 3 * med:
                reasons.append(f">3x_median({med:.0f}s)")
            if w < 15:
                reasons.append("<15s")
            if r.get("steps"):
                rz = (w / r["steps"] - ps_med) / (1.4826 * ps_mad)
                if abs(rz) > 6:
                    reasons.append(f"s_per_step robust z={rz:.1f}")
        if reasons:
            fin = r.get("final") or {}
            sus.append({
                "seed": r["seed"], "wall_s": w, "steps": r.get("steps"), "outcome": r.get("outcome"),
                "act_index": fin.get("act_index"), "total_floor": fin.get("total_floor"), "why": reasons,
            })
    out["suspicious_durations"] = sus
    out["s_per_step"] = {"median": round(ps_med, 4), "mad": round(ps_mad, 4)}
    out["act1"] = {k: sum(1 for r in rows if r.get("act1") == k) for k in sorted({r.get("act1") for r in rows})}
    return out


def score_arm(rows: list[dict]) -> dict:
    n = len(rows)
    w = sum(1 for r in rows if r.get("won"))
    a9 = sum(1 for r in rows if a9_reached(r))
    by_act = {}
    for act in sorted({r["act1"] for r in rows}):
        sub = [r for r in rows if r["act1"] == act]
        by_act[act] = {
            "n": len(sub),
            "wins": sum(1 for r in sub if r.get("won")),
            "a9_reached": sum(1 for r in sub if a9_reached(r)),
        }
    death_act = {}
    for r in rows:
        if r.get("won"):
            continue
        k = f"act{(r.get('final') or {}).get('act_index')}"
        if k == "act3":
            k += "_rsb" if r.get("reached_second_boss") else "_before_boss2"
        death_act[k] = death_act.get(k, 0) + 1
    return {
        "n": n,
        "A10": {"wins": w, "n": n, "rate": w / n, "wilson95": list(wilson(w, n))},
        "A9est": {"reached": a9, "n": n, "rate": a9 / n, "wilson95": list(wilson(a9, n))},
        "by_act1": by_act,
        "losses_by_where": dict(sorted(death_act.items())),
    }


def paired(d_map: dict, p_map: dict, seeds: list[int], fn) -> dict:
    a = b = c = d = 0
    for s in seeds:
        x, y = fn(d_map[s]), fn(p_map[s])
        if x and y:
            a += 1
        elif x and not y:
            b += 1
        elif y and not x:
            c += 1
        else:
            d += 1
    n = a + b + c + d
    p_exact = exact_mcnemar_p(b, c)
    p_scipy = scipy_mcnemar_p(b, c)
    nc = newcombe10(a, b, c, d)
    wd = wald_paired(b, c, n)
    am = agresti_min(a, b, c, d)
    return {
        "n": n,
        "both": a,
        "D_only_b": b,
        "P_only_c": c,
        "neither": d,
        "D_rate": (a + b) / n,
        "P_rate": (a + c) / n,
        "diff_pp": 100 * (b - c) / n,
        "mcnemar_exact_p_two_sided": p_exact,
        "mcnemar_exact_p_scipy_crosscheck": p_scipy,
        "ci95_pp_newcombe10": [100 * nc[0], 100 * nc[1]],
        "ci95_pp_wald": [100 * wd[0], 100 * wd[1]],
        "ci95_pp_agresti_min": [100 * am[0], 100 * am[1]],
        "ci_method": "Newcombe (1998) method 10 hybrid Wilson score interval for paired proportions (primary); Wald and Agresti-Min (+0.5/cell) as cross-checks",
    }


def driver_walls(log: Path) -> dict:
    out = {}
    if not log.exists():
        return out
    for line in log.read_text(encoding="utf-8").splitlines():
        m = re.search(r"arm (\w+) end rc=(\d+) wall_s=(\d+) rows=(\d+)", line)
        if m:
            out[m.group(1)] = {"rc": int(m.group(2)), "wall_s": int(m.group(3)), "rows": int(m.group(4))}
    return out


def main() -> int:
    # parents[1]: <experiment>/scripts/ here; the private archive kept the scripts in analysis/scripts/.
    exp = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
    runs = exp / "runs"
    raw = {arm: load(runs / f"{arm}.jsonl") for arm in ARMS}
    exits = {}
    for arm in ARMS:
        p = runs / f"{arm}.exit"
        exits[arm] = p.read_text(encoding="utf-8").strip() if p.exists() else None

    checks = {arm: arm_checks(arm, raw[arm]) for arm in ARMS}
    valid = {arm: [r for r in raw[arm] if not is_invalid(r)] for arm in ARMS}
    # one row per seed (first occurrence) for pairing; duplicates are reported above
    vmap = {}
    for arm in ARMS:
        m = {}
        for r in valid[arm]:
            m.setdefault(r["seed"], r)
        vmap[arm] = m
    pair_seeds = sorted(set(vmap["D"]) & set(vmap["P"]))

    # act check: act1 must be a function of the seed
    act_mismatch = [
        {"seed": s, "D": vmap["D"][s]["act1"], "P": vmap["P"][s]["act1"]}
        for s in pair_seeds
        if vmap["D"][s]["act1"] != vmap["P"][s]["act1"]
    ]
    # all rows, not only valid ones
    all_d = {r["seed"]: r for r in raw["D"]}
    all_p = {r["seed"]: r for r in raw["P"]}
    act_mismatch_all = [s for s in sorted(set(all_d) & set(all_p)) if all_d[s].get("act1") != all_p[s].get("act1")]

    # identical trajectories across arms (would indicate the arm switch did nothing)
    def sig(r):
        return (r.get("outcome"), r.get("steps"), r.get("n_decisions"), json.dumps(r.get("final"), sort_keys=True),
                json.dumps(r.get("resolution_counts"), sort_keys=True))

    identical = [s for s in pair_seeds if sig(vmap["D"][s]) == sig(vmap["P"][s])]
    same_outcome = sum(1 for s in pair_seeds if vmap["D"][s]["won"] == vmap["P"][s]["won"])

    scored = {arm: score_arm(list(vmap[arm].values())) for arm in ARMS}
    paired_scored_d = [vmap["D"][s] for s in pair_seeds]
    paired_scored_p = [vmap["P"][s] for s in pair_seeds]

    res_a10 = paired(vmap["D"], vmap["P"], pair_seeds, lambda r: bool(r.get("won")))
    res_a9 = paired(vmap["D"], vmap["P"], pair_seeds, a9_reached)

    # cost
    dw = driver_walls(exp / "logs" / "driver.log")
    wall = {arm: describe([r["wall_s"] for r in raw[arm]]) for arm in ARMS}
    ratios = [vmap["D"][s]["wall_s"] / vmap["P"][s]["wall_s"] for s in pair_seeds]
    both_lost_same_floor = [s for s in pair_seeds
                            if (vmap["D"][s]["final"] or {}).get("total_floor") == (vmap["P"][s]["final"] or {}).get("total_floor")]
    cost = {
        "arm_wall_s_driver": {k: v["wall_s"] for k, v in dw.items()},
        "arm_wall_ratio_D_over_P": (dw["D"]["wall_s"] / dw["P"]["wall_s"]) if "D" in dw and "P" in dw else None,
        "per_run_wall_s": wall,
        "per_run_mean_ratio_D_over_P": wall["D"]["mean"] / wall["P"]["mean"],
        "per_run_median_ratio_D_over_P": wall["D"]["median"] / wall["P"]["median"],
        "per_run_sum_ratio_D_over_P": wall["D"]["sum"] / wall["P"]["sum"],
        "paired_per_seed_ratio_D_over_P": {
            "median": statistics.median(ratios),
            "geomean": math.exp(statistics.fmean(math.log(x) for x in ratios)),
        },
        "paired_ratio_same_final_floor": {
            "n": len(both_lost_same_floor),
            "geomean": math.exp(statistics.fmean(
                math.log(vmap["D"][s]["wall_s"] / vmap["P"][s]["wall_s"]) for s in both_lost_same_floor))
            if both_lost_same_floor else None,
        },
        "per_step_s_mean": {arm: statistics.fmean(r["wall_s"] / r["steps"] for r in vmap[arm].values()) for arm in ARMS},
        "note": "rows carry wall_s per run only (no CPU field); each run is one single-threaded worker "
                "(OMP_NUM_THREADS=1, 30 procs), so per-run wall_s is the CPU-per-run proxy",
    }

    out = {
        "experiment": str(exp),
        "inputs": {arm: str(runs / f"{arm}.jsonl") for arm in ARMS},
        "exit_codes": exits,
        "rows": {
            arm: {
                "total": len(raw[arm]),
                "error": checks[arm]["error_count"],
                "timeout": checks[arm]["timeout_count"],
                "dropped": len(checks[arm]["invalid_rows"]),
                "scored": len(vmap[arm]),
            }
            for arm in ARMS
        } | {"paired_n": len(pair_seeds)},
        "scores": scored,
        "scores_on_paired_seeds": {"D": score_arm(paired_scored_d), "P": score_arm(paired_scored_p)},
        "paired": {"A10": res_a10, "A9est": res_a9},
        "cost": cost,
        "checks": checks,
        "act_check": {
            "per_arm": {arm: checks[arm]["act1"] for arm in ARMS},
            "seed_mismatches_valid_pairs": act_mismatch,
            "seed_mismatches_all_rows": act_mismatch_all,
        },
        "cross_arm": {
            "identical_trajectories": len(identical),
            "identical_trajectory_seeds": identical[:50],
            "same_A10_outcome": same_outcome,
        },
        "a9_definition": "won or reached_second_boss is True (None = run ended before act 3)",
    }
    dest = exp / "analysis" / "recompute_blind.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    # concise stdout
    for arm in ARMS:
        s = scored[arm]
        print(f"{arm}: rows {out['rows'][arm]}  A10 {s['A10']['wins']}/{s['n']}={s['A10']['rate']:.4f} "
              f"W95 [{s['A10']['wilson95'][0]:.4f},{s['A10']['wilson95'][1]:.4f}]  A9est {s['A9est']['reached']}/{s['n']}="
              f"{s['A9est']['rate']:.4f} W95 [{s['A9est']['wilson95'][0]:.4f},{s['A9est']['wilson95'][1]:.4f}]  by_act {s['by_act1']}")
    for k, v in out["paired"].items():
        print(f"paired {k}: n={v['n']} a={v['both']} b={v['D_only_b']} c={v['P_only_c']} d={v['neither']} "
              f"diff={v['diff_pp']:+.2f}pp p={v['mcnemar_exact_p_two_sided']:.4g} (scipy {v['mcnemar_exact_p_scipy_crosscheck']}) "
              f"NewcombeCI [{v['ci95_pp_newcombe10'][0]:+.2f},{v['ci95_pp_newcombe10'][1]:+.2f}] "
              f"Wald [{v['ci95_pp_wald'][0]:+.2f},{v['ci95_pp_wald'][1]:+.2f}] AM [{v['ci95_pp_agresti_min'][0]:+.2f},{v['ci95_pp_agresti_min'][1]:+.2f}]")
    print("cost", json.dumps(cost, default=str))
    for arm in ARMS:
        c = checks[arm]
        print(arm, "dups", c["duplicate_seeds"], "missing", c["missing_seeds"], "unexpected", c["unexpected_seeds"],
              "invalid", c["invalid_rows"], "err-but-scored", c["error_field_but_won_or_lost"],
              "inconsistent", c["inconsistent_rows"], "distinct", c["distinct_values"],
              "decisions_nonempty", c["decisions_nonempty"])
        print(arm, "suspicious durations", json.dumps(c["suspicious_durations"]))
    print("act", out["act_check"])
    print("cross_arm", {k: v for k, v in out["cross_arm"].items() if k != "identical_trajectory_seeds"},
          out["cross_arm"]["identical_trajectory_seeds"][:10])
    print("wrote", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
