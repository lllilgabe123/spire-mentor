"""Blind, independent recompute of the 2026-10-26 Ironclad release win-rate sweep.

Written from PREREG.md and the raw rows only. It does not read anything under
analysis/ except this folder. Inputs:

  cells/ironclad_A<asc>.jsonl   one JSON row per run (new arm: a10_it700, #830 default)
  cells/ironclad_A<asc>.log     ppo.probe stdout (only to cross-check the rows)
  progress.log                  START load averages (context for wall times)
  old arm (plr_it550), Ironclad rows: the published old_ironclad_runs.csv (ascension, seed,
      act1, won), or the private logs/2026-10-11-release-winrates/runs.csv the run read

Output: SWEEP/analysis/blind/blind.json (next to this script in the published layout); a
short table on stdout.

Conventions:
  * error and timeout rows are dropped before scoring (PREREG step 1); a timeout
    is not a loss. Any other non-won outcome (lost, max_steps, no_actions,
    agent_halt) is scored as a non-win, which is what 2026-10-11 summarize.py did.
  * Wilson intervals use z = 1.96, the constant in 2026-10-11 summarize.py, so
    the brackets are comparable with the published table.
  * Paired arms: a = both won, b = new won & old lost, c = new lost & old won,
    d = both lost. diff = new - old = (b - c) / n.
  * McNemar: exact two-sided binomial test on the discordant pairs,
    p = min(1, 2 * P[X <= min(b, c)]), X ~ Bin(b + c, 1/2); p = 1 when b + c = 0.
  * Paired CI: Newcombe (1998, Stat Med 17:2635) method 10, implemented below
    from the paper's formula and checked against hand-computed toy tables.

Run:  <python> recompute.py [SWEEP] [OLD_CSV]
  SWEEP defaults to the folder two levels above this script; OLD_CSV to
  SWEEP/old_ironclad_runs.csv. With the published four-column old rows the old arm's wall
  times are absent and its cost fields are null.

Changes since the run (2026-10-26): SWEEP came from this script's location and OLD_CSV was a
hard-coded private path (the eight-column runs.csv); both are arguments now, the old arm's cost
fields are null when the old rows lack wall_s, and seven exact-equality asserts in the self-tests
carry a ruff noqa. On the run's inputs the script still writes blind.json byte for byte.
"""

from __future__ import annotations

import collections
import csv
import json
import math
import re
import statistics
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
SWEEP = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent.parent
CELLS = SWEEP / "cells"
PROGRESS = SWEEP / "progress.log"
OLD_CSV = Path(sys.argv[2]) if len(sys.argv) > 2 else SWEEP / "old_ironclad_runs.csv"
OUT = SWEEP / "analysis" / "blind" / "blind.json"

LEVELS = list(range(11))
N_EXPECTED = {a: (500 if a >= 9 else 200) for a in LEVELS}
SEED_START = 81_000_000
EXPECTED_SRC_HASH = "4e5bb730d28e2976"
EXCLUDED = {"error", "timeout"}
Z = 1.96

# The 2026-10-11 Ironclad column as published (docs/experiments/2026-10-11-release-winrates.md).
PUBLISHED_OLD_PCT = {0: 89.5, 1: 95.0, 2: 92.5, 3: 90.0, 4: 86.5, 5: 85.0, 6: 86.0, 7: 79.0, 8: 70.0, 9: 39.0, 10: 11.0}
PUBLISHED_OLD_MEAN = 74.9
PUBLISHED_OLD_ACT1 = {"overgrowth": (1297, 63.2), "underdocks": (1503, 65.0)}
# Seeds no agent may play (task hard rule); asserted absent.
FORBIDDEN = range(96_500_000, 96_503_000)


# ---------------------------------------------------------------- statistics


def wilson(k: int, n: int, z: float = Z) -> tuple[float, float]:
    """Wilson score interval, closed form (2np + z^2 +- z sqrt(z^2 + 4np(1-p))) / (2(n + z^2))."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    centre = 2 * n * p + z * z
    half = z * math.sqrt(z * z + 4 * n * p * (1 - p))
    den = 2 * (n + z * z)
    return (max(0.0, (centre - half) / den), min(1.0, (centre + half) / den))


def mcnemar_exact_p(b: int, c: int) -> float:
    """Exact two-sided McNemar p: doubled lower binomial tail at 1/2, capped at 1."""
    m = b + c
    if m == 0:
        return 1.0
    k = min(b, c)
    tail = Fraction(sum(math.comb(m, i) for i in range(k + 1)), 2**m)
    return float(min(Fraction(1), 2 * tail))


def newcombe10(a: int, b: int, c: int, d: int, z: float = Z, _phi_mult: float = 1.0) -> dict:
    """Newcombe (1998) method 10 CI for p1 - p2 in a paired 2x2 table.

    p1 = (a + b) / n (new arm), p2 = (a + c) / n (old arm), theta = (b - c) / n.
    (l1, u1), (l2, u2) = Wilson limits of p1, p2.
    phi = (ad - bc) / sqrt((a+b)(c+d)(a+c)(b+d)), set to 0 when any marginal is 0.
    delta   = sqrt((p1 - l1)^2 - 2 phi (p1 - l1)(u2 - p2) + (u2 - p2)^2)
    epsilon = sqrt((u1 - p1)^2 - 2 phi (u1 - p1)(p2 - l2) + (p2 - l2)^2)
    CI = (theta - delta, theta + epsilon).
    `_phi_mult` exists only for the self-test's negative controls (-1 flips the
    sign of phi, 0 drops it); real calls leave it at 1.
    """
    n = a + b + c + d
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n, z)
    l2, u2 = wilson(a + c, n, z)
    marg = (a + b) * (c + d) * (a + c) * (b + d)
    phi = 0.0 if marg == 0 else _phi_mult * (a * d - b * c) / math.sqrt(marg)
    theta = p1 - p2
    delta = math.sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    eps = math.sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    return {"theta": theta, "lo": theta - delta, "hi": theta + eps, "phi": phi}


def self_tests() -> dict:
    """Check the three statistics against hand-computed values; abort on a miss."""
    out = {}

    # Toy 1 (all four cells non-zero): a=3, b=4, c=1, d=2, n=10, z=1.96.
    #   p1 = 7/10 = 0.7, p2 = 4/10 = 0.4, theta = 0.3; z^2 = 3.8416, 2(n+z^2) = 27.6832.
    #   Wilson 7/10: 2np+z^2 = 17.8416; z*sqrt(3.8416 + 8.4) = 1.96*3.4987998 = 6.857648
    #       l1 = 10.983952/27.6832 = 0.3967736, u1 = 24.699248/27.6832 = 0.8922105
    #   Wilson 4/10: 2np+z^2 = 11.8416; z*sqrt(3.8416 + 9.6) = 1.96*3.6662788 = 7.185906
    #       l2 = 4.655694/27.6832 = 0.1681777, u2 = 19.027506/27.6832 = 0.6873306
    #   phi = (3*2 - 4*1)/sqrt(7*3*4*6) = 2/sqrt(504) = 2/22.449944 = 0.0890871
    #   delta^2 = 0.3032264^2 - 2*0.0890871*0.3032264*0.2873306 + 0.2873306^2
    #           = 0.0919463 - 0.0155237 + 0.0825589 = 0.1589815 -> delta = 0.3987248
    #   eps^2   = 0.1922105^2 - 2*0.0890871*0.1922105*0.2318223 + 0.2318223^2
    #           = 0.0369449 - 0.0079392 + 0.0537416 = 0.0827473 -> eps = 0.2876582
    #   CI = (0.3 - 0.3987248, 0.3 + 0.2876582) = (-0.0987248, 0.5876582)
    t1 = newcombe10(3, 4, 1, 2)
    assert abs(t1["theta"] - 0.3) < 1e-12, t1
    assert abs(t1["phi"] - 0.0890871) < 1e-6, t1
    assert abs(t1["lo"] - (-0.0987248)) < 5e-6, t1
    assert abs(t1["hi"] - 0.5876582) < 5e-6, t1
    out["toy1_a3_b4_c1_d2"] = {"computed": t1, "hand": {"lo": -0.0987248, "hi": 0.5876582, "phi": 0.0890871}}

    # Toy 2 (a zero marginal, so phi = 0): a=5, b=3, c=0, d=0, n=8.
    #   p1 = 8/8 = 1, p2 = 5/8 = 0.625, theta = 0.375; 2(n+z^2) = 23.6832.
    #   Wilson 8/8: l1 = (16 + 3.8416 - 1.96*1.96)/23.6832 = 16/23.6832 = 0.6755844, u1 = 1
    #   Wilson 5/8: 2np+z^2 = 13.8416; z*sqrt(3.8416 + 7.5) = 1.96*3.3677292 = 6.600749
    #       l2 = 7.240851/23.6832 = 0.3057380, u2 = 20.442349/23.6832 = 0.8631582
    #   (c+d) = 0 -> phi = 0
    #   delta = sqrt(0.3244156^2 + 0.2381582^2) = sqrt(0.1052455 + 0.0567193) = 0.4024485
    #   eps   = sqrt(0 + 0.3192620^2) = 0.3192620
    #   CI = (0.375 - 0.4024485, 0.375 + 0.3192620) = (-0.0274485, 0.6942620)
    t2 = newcombe10(5, 3, 0, 0)
    assert t2["phi"] == 0.0, t2  # noqa: RUF069 (exact by construction)
    assert abs(t2["lo"] - (-0.0274485)) < 5e-6, t2
    assert abs(t2["hi"] - 0.6942620) < 5e-6, t2
    out["toy2_a5_b3_c0_d0_phi0"] = {"computed": t2, "hand": {"lo": -0.0274485, "hi": 0.6942620, "phi": 0.0}}

    # Toy 3 (no discordant pairs): a=d=10, b=c=0. All marginals are 10, so
    #   phi = (100 - 0)/sqrt(10^4) = 1, and with p1 = p2 = 0.5 the Wilson limits are
    #   symmetric, so delta = |(p1 - l1) - (u2 - p2)| = 0 and eps = 0: CI = [0, 0].
    #   That is method 10's known degenerate case at b + c = 0 (it is a property of
    #   the formula, not a bug); main() flags any real cell with b + c = 0.
    t3 = newcombe10(10, 0, 0, 10)
    assert t3["theta"] == 0 and t3["phi"] == 1.0 and abs(t3["lo"]) < 1e-12 and abs(t3["hi"]) < 1e-12, t3  # noqa: RUF069 (exact by construction)
    out["toy3_a10_b0_c0_d10_degenerate"] = {"computed": t3, "hand": {"lo": 0.0, "hi": 0.0, "phi": 1.0}}

    # Monte Carlo coverage, with negative controls. True paired cell probabilities
    # shaped like this sweep's cells (positive arm correlation). Method 10 should
    # cover the true diff ~95% of the time; flipping phi's sign or dropping phi
    # widens the interval under positive correlation, so those controls must
    # over-cover. If they did not differ, this check would be no evidence.
    import numpy as np

    rng = np.random.default_rng(20261026)
    cov = {}
    for name, n, probs in (
        ("A8_like_n200", 200, (0.65, 0.20, 0.05, 0.10)),
        ("A10_like_n500", 500, (0.08, 0.19, 0.03, 0.70)),
    ):
        true = probs[1] - probs[2]
        tabs = rng.multinomial(n, probs, size=20000)
        res = {}
        for label, mult in (("method10", 1.0), ("phi_sign_flipped", -1.0), ("phi_dropped", 0.0)):
            hit = 0
            for a, b, c, d in tabs.tolist():
                ci = newcombe10(a, b, c, d, _phi_mult=mult)
                hit += ci["lo"] <= true <= ci["hi"]
            res[label] = hit / len(tabs)
        assert 0.935 <= res["method10"] <= 0.965, (name, res)
        assert res["phi_sign_flipped"] > res["method10"] + 0.01, (name, res)
        assert res["phi_dropped"] > res["method10"] + 0.005, (name, res)
        cov[name] = {"n": n, "cell_probs_abcd": list(probs), "true_diff": true, "reps": len(tabs), "coverage": res}
    out["newcombe10_mc_coverage"] = cov

    # McNemar hand values: b=1, c=5 -> 2*(1 + 6)/64 = 14/64 = 0.21875;
    # b=0, c=6 -> 2/64 = 0.03125; b=c=3 -> capped at 1; b=c=0 -> 1.
    assert mcnemar_exact_p(1, 5) == 0.21875  # noqa: RUF069 (exact by construction)
    assert mcnemar_exact_p(6, 0) == 0.03125  # noqa: RUF069 (exact by construction)
    assert mcnemar_exact_p(3, 3) == 1.0  # noqa: RUF069 (exact by construction)
    assert mcnemar_exact_p(0, 0) == 1.0  # noqa: RUF069 (exact by construction)
    out["mcnemar_hand"] = {"b1_c5": 0.21875, "b6_c0": 0.03125, "b3_c3": 1.0, "b0_c0": 1.0}

    # Wilson against the 2026-10-11 summary.csv: Ironclad A0 179/200 -> [0.8448, 0.9303].
    lo, hi = wilson(179, 200)
    assert round(lo, 4) == 0.8448 and round(hi, 4) == 0.9303, (lo, hi)  # noqa: RUF069 (exact by construction)
    out["wilson_179_200"] = [lo, hi]

    # Cross-check McNemar against scipy if it is importable (it should be).
    try:
        from scipy.stats import binomtest

        worst = 0.0
        for b in range(0, 40):
            for c in range(0, 40):
                if b + c:
                    worst = max(worst, abs(mcnemar_exact_p(b, c) - binomtest(b, b + c, 0.5).pvalue))
        assert worst < 1e-12, worst
        out["mcnemar_vs_scipy_max_abs_diff_b_c_lt_40"] = worst
    except ImportError:
        out["mcnemar_vs_scipy_max_abs_diff_b_c_lt_40"] = "scipy unavailable"
    return out


# ---------------------------------------------------------------- loading


def load_new() -> tuple[dict[int, list[dict]], list[str]]:
    rows: dict[int, list[dict]] = {}
    problems: list[str] = []
    for a in LEVELS:
        path = CELLS / f"ironclad_A{a}.jsonl"
        rs = []
        with open(path, encoding="utf-8") as fh:
            for ln, line in enumerate(fh, 1):
                if not line.strip():
                    problems.append(f"A{a}: blank line {ln}")
                    continue
                r = json.loads(line)
                r["_line"] = ln
                if r.get("ascension") != a:
                    problems.append(f"A{a} line {ln}: ascension field {r.get('ascension')}")
                if r.get("character") != "Ironclad":
                    problems.append(f"A{a} line {ln}: character {r.get('character')}")
                if r.get("policy") != "net_greedy":
                    problems.append(f"A{a} line {ln}: policy {r.get('policy')}")
                if r.get("outcome") not in EXCLUDED and bool(r.get("won")) != (r.get("outcome") == "won"):
                    problems.append(f"A{a} line {ln}: won={r.get('won')} vs outcome={r.get('outcome')}")
                fin = r.get("final") or {}
                if fin and fin.get("ascension") != a:
                    problems.append(f"A{a} line {ln}: final.ascension {fin.get('ascension')}")
                rs.append(r)
        rows[a] = rs
    return rows, problems


def load_old() -> dict[int, list[dict]]:
    rows: dict[int, list[dict]] = {a: [] for a in LEVELS}
    with open(OLD_CSV, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            # The published CSV has Ironclad rows only and no character/outcome/total_floor/wall_s;
            # every old Ironclad row is "won" or "lost", so outcome follows from won.
            if r.get("character", "Ironclad") != "Ironclad":
                continue
            won = r["won"] == "1"
            rows[int(r["ascension"])].append(
                {
                    "seed": int(r["seed"]),
                    "act1": r["act1"],
                    "outcome": r.get("outcome", "won" if won else "lost"),
                    "won": won,
                    "total_floor": int(r["total_floor"]) if r.get("total_floor") else None,
                    "wall_s": float(r["wall_s"]) if r.get("wall_s") else None,
                }
            )
    return rows


# ---------------------------------------------------------------- analysis


def cell_stats(rs: list[dict]) -> dict:
    outcomes = collections.Counter(r["outcome"] for r in rs)
    scored = [r for r in rs if r["outcome"] not in EXCLUDED]
    wins = sum(1 for r in scored if r["won"])
    n = len(scored)
    lo, hi = wilson(wins, n)
    return {
        "n_rows": len(rs),
        "n_error": outcomes.get("error", 0),
        "n_timeout": outcomes.get("timeout", 0),
        "n_other_nonwin_outcomes": {k: v for k, v in outcomes.items() if k not in ("won", "lost", "error", "timeout")},
        "wins": wins,
        "n_scored": n,
        "rate": wins / n if n else math.nan,
        "wilson95": [lo, hi],
    }


def seed_integrity(rs: list[dict], a: int) -> dict:
    seeds = [r["seed"] for r in rs]
    expected = set(range(SEED_START, SEED_START + N_EXPECTED[a]))
    dup = sorted(s for s, k in collections.Counter(seeds).items() if k > 1)
    got = set(seeds)
    return {
        "n_rows": len(seeds),
        "n_unique": len(got),
        "min": min(seeds) if seeds else None,
        "max": max(seeds) if seeds else None,
        "equals_expected_range": got == expected and not dup,
        "missing": sorted(expected - got),
        "extra": sorted(got - expected),
        "duplicates": dup,
    }


def paired_level(new_rs: list[dict], old_rs: list[dict]) -> dict:
    nmap = {r["seed"]: r for r in new_rs}
    omap = {r["seed"]: r for r in old_rs}
    both = sorted(set(nmap) & set(omap))
    keys = [s for s in both if nmap[s]["outcome"] not in EXCLUDED and omap[s]["outcome"] not in EXCLUDED]
    a = b = c = d = 0
    for s in keys:
        nw, ow = bool(nmap[s]["won"]), bool(omap[s]["won"])
        if nw and ow:
            a += 1
        elif nw:
            b += 1
        elif ow:
            c += 1
        else:
            d += 1
    n = a + b + c + d
    nc = newcombe10(a, b, c, d)
    return {
        "n_pairs": n,
        "n_seeds_in_both_arms": len(both),
        "n_seeds_only_new": len(set(nmap) - set(omap)),
        "n_seeds_only_old": len(set(omap) - set(nmap)),
        "table": {"a_both_won": a, "b_new_won_old_lost": b, "c_new_lost_old_won": c, "d_both_lost": d},
        "new": (a + b) / n,
        "old": (a + c) / n,
        "diff": (b - c) / n,
        "b": b,
        "c": c,
        "mcnemar_exact_p": mcnemar_exact_p(b, c),
        "newcombe10_95": [nc["lo"], nc["hi"]],
        "phi": nc["phi"],
    }


def act1_split(rows_by_level: dict[int, list[dict]]) -> dict:
    acc: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for a in LEVELS:
        for r in rows_by_level[a]:
            if r["outcome"] in EXCLUDED:
                continue
            acc[r["act1"]][0] += int(bool(r["won"]))
            acc[r["act1"]][1] += 1
    out = {}
    for act, (k, n) in sorted(acc.items()):
        out[act] = {"n": n, "wins": k, "rate": k / n, "wilson95": list(wilson(k, n))}
    return out


def act1_by_level(rows_by_level: dict[int, list[dict]]) -> dict:
    out = {}
    for a in LEVELS:
        acc: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
        for r in rows_by_level[a]:
            if r["outcome"] in EXCLUDED:
                continue
            acc[r["act1"]][0] += int(bool(r["won"]))
            acc[r["act1"]][1] += 1
        out[f"A{a}"] = {act: {"n": n, "wins": k, "rate": k / n} for act, (k, n) in sorted(acc.items())}
    return out


def act1_standardized(rows_by_level: dict[int, list[dict]]) -> dict:
    """Unweighted mean over A0-A10 of the per-level act-1 rates, and the act-1 mix per seed block.

    The pooled split weights each act by its runs. Act 1 is a function of the
    seed, and the 200-seed block (all levels) and the 300-seed block (A9/A10
    only) have different act-1 mixes, so the pooled split mixes ascension
    composition into the act comparison. This level-standardized view removes that.
    """
    by = act1_by_level(rows_by_level)
    acts = sorted({act for v in by.values() for act in v})
    std = {act: statistics.fmean(by[f"A{a}"][act]["rate"] for a in LEVELS) for act in acts}
    mix: dict[str, collections.Counter] = {"seeds_0_199": collections.Counter(), "seeds_200_499": collections.Counter()}
    seen = {}
    for a in LEVELS:
        for r in rows_by_level[a]:
            seen[r["seed"]] = r["act1"]
    for s, act in seen.items():
        mix["seeds_0_199" if s - SEED_START < 200 else "seeds_200_499"][act] += 1
    return {"level_standardized_rate": std, "act1_mix_by_seed_block": {k: dict(v) for k, v in mix.items()}}


def a9_direct_vs_a10_estimate(new: dict[int, list[dict]]) -> dict:
    """A9 played directly vs the A9 estimate read off A10 runs (reached_second_boss).

    `reached_second_boss` = the run entered the A10 DoubleBoss second-boss room,
    i.e. beat the first act-3 boss; ppo.workers documents it as an A9 win read
    off an A10 run. Same seeds, same champion and budget, so this pairs by seed.
    Arm 1 = A9 direct (won), arm 2 = A10 estimate (reached_second_boss is True).
    """
    a9 = {r["seed"]: r for r in new[9] if r["outcome"] not in EXCLUDED}
    a10 = {r["seed"]: r for r in new[10] if r["outcome"] not in EXCLUDED}
    keys = sorted(set(a9) & set(a10))
    a = b = c = d = 0
    for s in keys:
        x, y = bool(a9[s]["won"]), a10[s].get("reached_second_boss") is True
        if x and y:
            a += 1
        elif x:
            b += 1
        elif y:
            c += 1
        else:
            d += 1
    n = a + b + c + d
    nc = newcombe10(a, b, c, d)
    a10_rsb = collections.Counter(str(a10[s].get("reached_second_boss")) for s in keys)
    conv = sum(1 for s in keys if a10[s].get("reached_second_boss") is True and a10[s]["won"])
    reach = sum(1 for s in keys if a10[s].get("reached_second_boss") is True)
    return {
        "n_pairs": n,
        "a9_direct_rate": (a + b) / n,
        "a10_estimate_rate": (a + c) / n,
        "diff_direct_minus_estimate": (b - c) / n,
        "table": {"a_both": a, "b_a9won_a10notpast": b, "c_a9lost_a10past": c, "d_neither": d},
        "mcnemar_exact_p": mcnemar_exact_p(b, c),
        "newcombe10_95": [nc["lo"], nc["hi"]],
        "a10_reached_second_boss_values": dict(a10_rsb),
        "a10_second_boss_conversion": {"won": conv, "reached": reach, "rate": conv / reach if reach else None},
        "note": (
            "separate plays of the same seed, not a replay, so this tests the estimator. In the v0.107.1 decomp "
            "A10 adds only AscensionLevel.DoubleBoss (RunManager draws a second act-3 boss from Rng.UpFront). "
            "Post-hoc check (looked at after noticing the gap): not pre-registered, no multiplicity control"
        ),
    }


def act1_agreement(new: dict[int, list[dict]], old: dict[int, list[dict]]) -> dict:
    per_level = {}
    tot_agree = tot = 0
    disagreements = []
    for a in LEVELS:
        nmap = {r["seed"]: r["act1"] for r in new[a]}
        omap = {r["seed"]: r["act1"] for r in old[a]}
        both = sorted(set(nmap) & set(omap))
        ag = sum(1 for s in both if nmap[s] == omap[s])
        per_level[f"A{a}"] = {"n_seeds": len(both), "agree": ag, "disagree": len(both) - ag}
        tot_agree += ag
        tot += len(both)
        disagreements += [{"asc": a, "seed": s, "new": nmap[s], "old": omap[s]} for s in both if nmap[s] != omap[s]]

    def within_arm(arm: dict[int, list[dict]]) -> dict:
        by_seed: dict[int, set[str]] = collections.defaultdict(set)
        for a in LEVELS:
            for r in arm[a]:
                by_seed[r["seed"]].add(r["act1"])
        multi = sorted(s for s, v in by_seed.items() if len(v) > 1)
        return {"n_seeds": len(by_seed), "seeds_with_more_than_one_act1_across_levels": multi}

    return {
        "per_level": per_level,
        "total": {"n": tot, "agree": tot_agree, "disagree": tot - tot_agree},
        "disagreements": disagreements[:50],
        "within_new_arm_across_levels": within_arm(new),
        "within_old_arm_across_levels": within_arm(old),
    }


def pooled_paired(new: dict[int, list[dict]], old: dict[int, list[dict]], per_level: dict) -> dict:
    """Descriptive pooled read with the seed-clustered caveat (PREREG step 3).

    D = mean over the 11 levels of the per-level paired diff = sum_s z_s, where
    z_s = sum_{levels l containing seed s} x_{l,s} / (11 n_l) and x = new_won - old_won.
    Seeds are the independent units: seeds 0-199 sit in all 11 levels, seeds
    200-499 only in A9/A10, so the variance is a stratified cluster sum over
    the two seed blocks. The 'levels independent' SE is reported beside it.
    """
    z_s: dict[int, float] = collections.defaultdict(float)
    n_l = {}
    pooled_b = pooled_c = pooled_n = 0
    for a in LEVELS:
        nmap = {r["seed"]: r for r in new[a] if r["outcome"] not in EXCLUDED}
        omap = {r["seed"]: r for r in old[a] if r["outcome"] not in EXCLUDED}
        keys = sorted(set(nmap) & set(omap))
        n_l[a] = len(keys)
        for s in keys:
            x = int(bool(nmap[s]["won"])) - int(bool(omap[s]["won"]))
            z_s[s] += x / (len(LEVELS) * len(keys))
            pooled_b += x == 1
            pooled_c += x == -1
            pooled_n += 1
    D = sum(z_s.values())
    var = 0.0
    strata = {"seeds_0_199": [], "seeds_200_499": []}
    for s, v in z_s.items():
        strata["seeds_0_199" if s - SEED_START < 200 else "seeds_200_499"].append(v)
    for vals in strata.values():
        m = len(vals)
        if m > 1:
            mu = sum(vals) / m
            var += m / (m - 1) * sum((v - mu) ** 2 for v in vals)
    se_cluster = math.sqrt(var)
    # Levels-independent SE: each level's paired diff var = (b + c - (b - c)^2 / n) / n^2 ... / (n - 1) form.
    var_ind = 0.0
    for a in LEVELS:
        pl = per_level[f"A{a}"]
        n, b, c = pl["n_pairs"], pl["b"], pl["c"]
        dbar = (b - c) / n
        s2 = ((b + c) - n * dbar * dbar) / (n - 1)  # sample var of x in {-1,0,1}
        var_ind += s2 / n
    se_ind = math.sqrt(var_ind) / len(LEVELS)
    return {
        "unweighted_mean_diff_A0_A10": D,
        "se_seed_clustered_stratified": se_cluster,
        "ci95_seed_clustered_normal": [D - 1.959963984540054 * se_cluster, D + 1.959963984540054 * se_cluster],
        "se_if_levels_were_independent": se_ind,
        "design_effect_var_ratio": (se_cluster / se_ind) ** 2 if se_ind else None,
        "run_pooled": {
            "n_pairs": pooled_n,
            "b": pooled_b,
            "c": pooled_c,
            "diff": (pooled_b - pooled_c) / pooled_n,
            "note": "run-weighted (A9/A10 carry 1000 of 2800 pairs); no p-value: the same seeds repeat across levels",
        },
        "note": "descriptive only; confounds champion, search default and engine changes; attributes nothing",
    }


def wall_stats(rs: list[dict]) -> dict | None:
    w = [r["wall_s"] for r in rs if r.get("wall_s") is not None]
    if not w:  # the published old rows carry no wall_s
        return None
    return {
        "n": len(w),
        "mean_s": statistics.fmean(w),
        "median_s": statistics.median(w),
        "max_s": max(w),
        "min_s": min(w),
        "sum_h": sum(w) / 3600,
    }


def wall_by_outcome(rs: list[dict]) -> dict:
    out = {}
    for oc in ("won", "lost"):
        w = [r["wall_s"] for r in rs if r["outcome"] == oc]
        if w:
            out[oc] = {"n": len(w), "mean_s": statistics.fmean(w), "median_s": statistics.median(w)}
    return out


def parse_logs() -> dict:
    out = {}
    for a in LEVELS:
        txt = (CELLS / f"ironclad_A{a}.log").read_text(encoding="utf-8")
        m = re.search(r"^win rate: (\d+)/(\d+)", txt, re.M)
        w = re.search(r"^wall/run: mean ([\d.]+)s\s+median ([\d.]+)s\s+max ([\d.]+)s; batch wall (\d+)s", txt, re.M)
        oc = re.search(r"^outcomes: (.*)$", txt, re.M)
        g1b = re.search(r"^G1b ownership audit: (\w+)", txt, re.M)
        out[a] = {
            "wins": int(m.group(1)) if m else None,
            "n": int(m.group(2)) if m else None,
            "wall_mean": float(w.group(1)) if w else None,
            "wall_median": float(w.group(2)) if w else None,
            "wall_max": float(w.group(3)) if w else None,
            "batch_wall_s": int(w.group(4)) if w else None,
            "outcomes_line": oc.group(1) if oc else None,
            "g1b_audit": g1b.group(1) if g1b else None,
        }
    return out


def parse_progress() -> dict:
    out = {}
    for line in PROGRESS.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^(\S+) START Ironclad A(\d+) n=(\d+) load ([\d.]+) ([\d.]+) ([\d.]+)", line)
        if m:
            out.setdefault(int(m.group(2)), {})["start"] = m.group(1)
            out[int(m.group(2))]["load_1_5_15"] = [float(m.group(i)) for i in (4, 5, 6)]
        m = re.match(r"^(\S+) DONE\s+Ironclad A(\d+) rc=(\d+)", line)
        if m:
            out.setdefault(int(m.group(2)), {})["done"] = m.group(1)
            out[int(m.group(2))]["rc"] = int(m.group(3))
    return out


def loss_profile(rs: list[dict]) -> dict:
    """Where the lost runs ended (final.act_index), and A10 second-boss flags."""
    acts = collections.Counter((r.get("final") or {}).get("act_index") for r in rs if r["outcome"] == "lost")
    win_floors = collections.Counter((r.get("final") or {}).get("total_floor") for r in rs if r["outcome"] == "won")
    rsb = collections.Counter(f"{r['outcome']}:{r.get('reached_second_boss')}" for r in rs)
    return {
        "lost_by_final_act_index": {str(k): v for k, v in sorted(acts.items(), key=lambda kv: str(kv[0]))},
        "won_total_floor": {str(k): v for k, v in sorted(win_floors.items(), key=lambda kv: str(kv[0]))},
        "outcome_by_reached_second_boss": dict(sorted(rsb.items())),
    }


# ---------------------------------------------------------------- main


def r6(x):
    if isinstance(x, float):
        # Small nonzero values (exact McNemar p reaches 1.7e-13) keep 6 significant digits;
        # round(x, 6) wrote them as 0.0 (reconcile fix F1).
        if x != 0 and abs(x) < 1e-3:
            return float(f"{x:.6g}")
        return round(x, 6)
    if isinstance(x, dict):
        return {k: r6(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [r6(v) for v in x]
    return x


def main() -> int:
    tests = self_tests()
    new, problems = load_new()
    old = load_old()

    per_level_new = {f"A{a}": cell_stats(new[a]) for a in LEVELS}
    per_level_old = {f"A{a}": cell_stats(old[a]) for a in LEVELS}
    mean_new = statistics.fmean(per_level_new[f"A{a}"]["rate"] for a in LEVELS)
    mean_old = statistics.fmean(per_level_old[f"A{a}"]["rate"] for a in LEVELS)

    old_check = {}
    for a in LEVELS:
        pct = round(100 * per_level_old[f"A{a}"]["rate"], 1)
        old_check[f"A{a}"] = {"recomputed_pct": pct, "published_pct": PUBLISHED_OLD_PCT[a], "match": pct == PUBLISHED_OLD_PCT[a]}
    old_act1 = act1_split(old)
    old_act1_check = {
        act: {
            "recomputed": [old_act1[act]["n"], round(100 * old_act1[act]["rate"], 1)],
            "published": list(PUBLISHED_OLD_ACT1[act]),
            "match": [old_act1[act]["n"], round(100 * old_act1[act]["rate"], 1)] == list(PUBLISHED_OLD_ACT1[act]),
        }
        for act in PUBLISHED_OLD_ACT1
    }

    paired = {f"A{a}": paired_level(new[a], old[a]) for a in LEVELS}
    for a in LEVELS:
        if paired[f"A{a}"]["b"] + paired[f"A{a}"]["c"] == 0:
            problems.append(f"A{a}: no discordant pairs, Newcombe method 10 interval is degenerate")
    pooled = pooled_paired(new, old, paired)

    src_hashes = collections.Counter((r.get("wheel") or {}).get("src_hash") for a in LEVELS for r in new[a])
    so_paths = collections.Counter((r.get("wheel") or {}).get("so_path") for a in LEVELS for r in new[a])
    pgo = collections.Counter(
        f"pgo={(r.get('wheel') or {}).get('pgo')} instrumented={(r.get('wheel') or {}).get('pgo_instrumented')} "
        f"profdata={(r.get('wheel') or {}).get('profdata_sha256')}"
        for a in LEVELS
        for r in new[a]
    )
    py_src = collections.Counter(r.get("py_src") for a in LEVELS for r in new[a])
    forbidden_hits = [(a, r["seed"]) for a in LEVELS for r in new[a] if r["seed"] in FORBIDDEN]

    logs = parse_logs()
    log_check = {}
    for a in LEVELS:
        st, ws, lg = per_level_new[f"A{a}"], wall_stats(new[a]), logs[a]
        log_check[f"A{a}"] = {
            "log_wins_n": [lg["wins"], lg["n"]],
            "rows_wins_n": [st["wins"], st["n_scored"]],
            "wins_match": lg["wins"] == st["wins"] and lg["n"] == st["n_scored"],
            "log_wall_mean_median_max": [lg["wall_mean"], lg["wall_median"], lg["wall_max"]],
            "rows_wall_mean_median_max": [round(ws["mean_s"], 1), round(ws["median_s"], 1), round(ws["max_s"], 1)],
            "wall_match": [lg["wall_mean"], lg["wall_median"], lg["wall_max"]]
            == [round(ws["mean_s"], 1), round(ws["median_s"], 1), round(ws["max_s"], 1)],
            "batch_wall_s": lg["batch_wall_s"],
            "outcomes_line": lg["outcomes_line"],
            "g1b_audit": lg["g1b_audit"],
        }
    progress = parse_progress()

    integrity = {
        "row_problems": problems,
        "n_rows_total_new": sum(len(new[a]) for a in LEVELS),
        "n_rows_total_old_ironclad": sum(len(old[a]) for a in LEVELS),
        "seeds_new": {f"A{a}": seed_integrity(new[a], a) for a in LEVELS},
        "seeds_old": {f"A{a}": seed_integrity(old[a], a) for a in LEVELS},
        "wheel_src_hash": dict(src_hashes),
        "wheel_src_hash_all_expected": set(src_hashes) == {EXPECTED_SRC_HASH},
        "wheel_so_path": dict(so_paths),
        "wheel_pgo": dict(pgo),
        "py_src": dict(py_src),
        "forbidden_seed_rows": forbidden_hits,
        "log_cross_check": log_check,
        "progress_log": {f"A{a}": progress.get(a) for a in LEVELS},
    }

    cost = {
        f"A{a}": {
            **wall_stats(new[a]),
            "by_outcome": wall_by_outcome(new[a]),
            "old_arm": wall_stats(old[a]),
            "start_load_1_5_15": (progress.get(a) or {}).get("load_1_5_15"),
        }
        for a in LEVELS
    }
    all_w_new = [r["wall_s"] for a in LEVELS for r in new[a]]
    all_w_old = [r["wall_s"] for a in LEVELS for r in old[a] if r["wall_s"] is not None]

    result = {
        "meta": {
            "what": "blind recompute of the 2026-10-26 Ironclad release sweep (new arm) vs the 2026-10-11 Ironclad column (old arm)",
            "new_arm": "Ironclad, champion ppo_fullrun_ironclad_a10_it700, ppo.probe net_greedy, act1 roll, 8000 sims/decision incl. bosses, one tree, shipped #830 Ironclad default (forms+combo knobs every fight, BC prior elite/boss), engine 0181fa552, seeds 81,000,000+ (200 at A0-A8, 500 at A9-A10)",
            "old_arm": "Ironclad, champion plr_it550, same protocol and seeds, engine 0cab615a3, pre-#830 search",
            "wilson_z": Z,
            "excluded_outcomes": sorted(EXCLUDED),
            "inputs": [str(CELLS), str(PROGRESS), str(OLD_CSV)],
        },
        "self_tests": tests,
        "per_level_new": per_level_new,
        "unweighted_mean_A0_A10_new": mean_new,
        "act1_split_new_pooled_A0_A10": act1_split(new),
        "act1_split_new_by_level": act1_by_level(new),
        "act1_level_standardized": {"new": act1_standardized(new), "old": act1_standardized(old)},
        "a9_direct_vs_a10_estimate_new_arm": a9_direct_vs_a10_estimate(new),
        "paired_vs_old": paired,
        "pooled_paired_A0_A10": pooled,
        "act1_agreement_new_vs_old": act1_agreement(new, old),
        "old_column_recomputed": {
            "per_level": per_level_old,
            "unweighted_mean_A0_A10": mean_old,
            "mean_pct_rounded": round(100 * mean_old, 1),
            "published_mean_pct": PUBLISHED_OLD_MEAN,
            "mean_match": round(100 * mean_old, 1) == PUBLISHED_OLD_MEAN,
            "per_level_check": old_check,
            "all_levels_match": all(v["match"] for v in old_check.values()),
            "act1_split": old_act1,
            "act1_split_check": old_act1_check,
        },
        "loss_profile_new": {f"A{a}": loss_profile(new[a]) for a in LEVELS},
        "cost_wall_s": cost,
        "cost_overall": {
            "new_mean_s": statistics.fmean(all_w_new),
            "new_sum_h": sum(all_w_new) / 3600,
            "old_mean_s": statistics.fmean(all_w_old) if all_w_old else None,
            "old_sum_h": sum(all_w_old) / 3600 if all_w_old else None,
            "note": "wall time per run is NOT a clean cost measure: other jobs shared the box (see progress_log load averages)",
        },
        "integrity": integrity,
    }
    OUT.write_text(json.dumps(r6(result), indent=1, sort_keys=False) + "\n", encoding="utf-8")

    # ---- stdout table
    print(f"self-tests OK; MC coverage: {json.dumps(r6({k: v['coverage'] for k, v in tests['newcombe10_mc_coverage'].items()}))}")
    print(f"row problems: {len(problems)}  src_hash: {dict(src_hashes)}  forbidden seeds: {len(forbidden_hits)}")
    print(f"{'lvl':>4} {'rows':>4} {'err':>3} {'to':>3} {'wins':>4} {'rate':>6} {'wilson95':>15} | {'pairs':>5} {'new':>6} {'old':>6} {'diff':>6} {'b':>3} {'c':>3} {'p':>9} {'newcombe95':>17}")
    for a in LEVELS:
        s, p = per_level_new[f"A{a}"], paired[f"A{a}"]
        print(
            f"A{a:<3} {s['n_rows']:>4} {s['n_error']:>3} {s['n_timeout']:>3} {s['wins']:>4} {100 * s['rate']:6.1f} "
            f"[{100 * s['wilson95'][0]:5.1f}-{100 * s['wilson95'][1]:5.1f}] | {p['n_pairs']:>5} {100 * p['new']:6.1f} "
            f"{100 * p['old']:6.1f} {100 * p['diff']:+6.1f} {p['b']:>3} {p['c']:>3} {p['mcnemar_exact_p']:9.2e} "
            f"[{100 * p['newcombe10_95'][0]:+6.1f},{100 * p['newcombe10_95'][1]:+6.1f}]"
        )
    print(f"mean A0-A10: new {100 * mean_new:.2f}  old {100 * mean_old:.2f} (published {PUBLISHED_OLD_MEAN})  old per-level all match: {result['old_column_recomputed']['all_levels_match']}")
    print(f"pooled: {json.dumps(r6(pooled))}")
    print(f"act1 new: {json.dumps(r6(result['act1_split_new_pooled_A0_A10']))}")
    print(f"act1 old: {json.dumps(r6(old_act1_check))}")
    print(f"act1 agreement: {json.dumps(result['act1_agreement_new_vs_old']['total'])}")
    print(f"act1 level-standardized: {json.dumps(r6(result['act1_level_standardized']))}")
    print(f"A9 direct vs A10 estimate: {json.dumps(r6(result['a9_direct_vs_a10_estimate_new_arm']))}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
