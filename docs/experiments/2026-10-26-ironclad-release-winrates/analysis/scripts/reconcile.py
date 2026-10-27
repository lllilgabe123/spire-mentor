"""Third, minimal recompute of the 2026-10-26 Ironclad release sweep, and the reconciliation of
the primary analysis (analysis/primary.json + primary.md, from scripts/analyze.py) with the blind
one (analysis/blind/blind.json, from blind/recompute.py).

Recomputed here from the raw rows, with code written for this file and no import from either
analysis script:
  wins, n, rate and Wilson 95% CI per level (z = 1.96, the 2026-10-11 summarize.py constant, and
  z = Phi^-1(0.975)); the unweighted mean over A0-A10; the paired 2x2 table (b = new won & old
  lost, c = old won & new lost); the exact two-sided McNemar p (scipy binomtest); the Newcombe
  (1998) method 10 paired CI; the act-1 split; a seed-clustered bootstrap of the pooled mean diff;
  integrity counts; wall-time cost; per-act loss counts; the A9-vs-A10-estimate table.

Every value this recompute shares with another source is compared with an explicit tolerance.
Sources: primary.json, primary.md, blind.json, the cell .logs, progress.log, the 2026-10-11
progress.log / summary.csv, and the published 2026-10-11 Ironclad column.

If analysis/prefix_snapshot/ exists (the primary and blind outputs as they were before this
reconciliation fixed two reporting bugs), the same comparisons also run against it, so the
mismatches that led to the fixes stay on record next to the clean result.

Writes analysis/reconcile.json and analysis/final.txt.
Usage: <python> -I -B reconcile.py [SWEEP] [OLD]
  SWEEP defaults to the folder two levels above this script. OLD is the 2026-10-11 sweep's rows:
  the published old_ironclad_runs.csv (ascension, seed, act1, won; the default,
  SWEEP/old_ironclad_runs.csv), or the private folder logs/2026-10-11-release-winrates
  (runs.csv with all eight columns, progress.log, summary.csv), which the run read. With the
  published CSV, every comparison that needs the old rows' wall_s or total_floor, the old
  progress.log or summary.csv is skipped, and the old arm's cost and depth are null.

Changes since the run (2026-10-26): SWEEP came from this script's location and OLD was a
hard-coded private path; both are arguments now, the skips above were added, and ruff dropped the
parentheses of eight tuple subscripts. On the run's inputs (with analysis/prefix_snapshot/) the
script still writes final.txt byte for byte, and reconcile.json except the two sha256 it records
of the current analyze.py and recompute.py.
"""

from __future__ import annotations

import ast
import collections
import csv
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, norm

SWEEP = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
CELLS = SWEEP / "cells"
ANALYSIS = SWEEP / "analysis"
PROGRESS = SWEEP / "progress.log"
OLD = Path(sys.argv[2]) if len(sys.argv) > 2 else SWEEP / "old_ironclad_runs.csv"
# A folder is the private sweep folder; a file is the published four-column CSV.
OLD_CSV = OLD / "runs.csv" if OLD.is_dir() else OLD
OLD_PROGRESS = OLD / "progress.log" if OLD.is_dir() else None
OLD_SUMMARY = OLD / "summary.csv" if OLD.is_dir() else None
PRIMARY_JSON = ANALYSIS / "primary.json"
PRIMARY_MD = ANALYSIS / "primary.md"
BLIND_JSON = ANALYSIS / "blind" / "blind.json"
SNAP = ANALYSIS / "prefix_snapshot"
OUT_JSON = ANALYSIS / "reconcile.json"
OUT_TXT = ANALYSIS / "final.txt"

LEVELS = list(range(11))
N_PER = {a: (500 if a >= 9 else 200) for a in LEVELS}
SEED0 = 81_000_000
ORDER = [10, 9, 0, 6, 3, 8, 1, 5, 2, 4, 7]
BAD = ("error", "timeout")
FORBIDDEN = range(96_500_000, 96_503_000)
Z196 = 1.96
ZEX = float(norm.ppf(0.975))
BOOT_N = 10_000

SRC_HASH = "4e5bb730d28e2976"
HEAD = "0181fa552a78053f260f0796e1cb086d770c7cef"
SO_SHA = "823fa393eab74db57d0cffd6f71c4a5c873a74d345a354adfca8458bf1333ba2"
CKPT = "ppo_fullrun_ironclad_a10_it700.pt"

# The Ironclad column of docs/experiments/2026-10-11-release-winrates.md (origin/main), as printed.
PUBLISHED_OLD = {
    "rate": {0: "89.5", 1: "95.0", 2: "92.5", 3: "90.0", 4: "86.5", 5: "85.0", 6: "86.0", 7: "79.0", 8: "70.0", 9: "39.0", 10: "11.0"},
    "ci": {
        0: ("84.5", "93.0"), 1: ("91.0", "97.3"), 2: ("88.0", "95.4"), 3: ("85.1", "93.4"),
        4: ("81.1", "90.6"), 5: ("79.4", "89.3"), 6: ("80.5", "90.1"), 7: ("72.8", "84.1"),
        8: ("63.3", "75.9"), 9: ("34.8", "43.3"), 10: ("8.5", "14.0"),
    },
    "mean": "74.9",
    "act1": {"overgrowth": (1297, "63.2"), "underdocks": (1503, "65.0")},
}

# §2.6 standard eval (docs/experiments/2026-10-26-ic-a10-it700-standard-eval.md, arm D): context only.
STD_EVAL_A10 = {"wins": 107, "n": 400}

# Tolerances. Every comparison names one rule.
TOLS = {
    "exact": 0.0,  # counts, flags, strings, lists
    "full": 1e-9,  # full-precision floats in primary.json vs this recompute (summation order only)
    "r6": 5e-7 + 1e-12,  # blind.json rounds floats to 6 decimals (r6)
    "p_rel": 1e-3,  # p-values: RELATIVE, i.e. agreement to ~3 significant figures (tables print 2)
    "csv4": 5e-5 + 1e-12,  # 2026-10-11 summary.csv prints 4 decimals
    "log1": 0.05 + 1e-9,  # .log / progress.log print seconds to 0.1
    "method": 0.0025,  # different CI estimators for the pooled diff: endpoints within 0.25pp
    "mc": 0.0015,  # bootstrap Monte Carlo error across RNG seeds: endpoints within 0.15pp
}


# ---------------------------------------------------------------- statistics (written here)


def wilson(k: int, n: int, z: float) -> tuple[float, float]:
    p = k / n
    zz = z * z
    mid = p + zz / (2 * n)
    rad = z * math.sqrt(p * (1 - p) / n + zz / (4 * n * n))
    return ((mid - rad) / (1 + zz / n), (mid + rad) / (1 + zz / n))


def mcnemar_p(b: int, c: int) -> float:
    return 1.0 if b + c == 0 else float(binomtest(b, b + c, 0.5, alternative="two-sided").pvalue)


def newcombe10(a: int, b: int, c: int, d: int, z: float) -> tuple[float, float, float]:
    """Paired diff p1 - p2, p1 = (a+b)/n, p2 = (a+c)/n; Newcombe 1998 method 10; returns lo, hi, phi."""
    n = a + b + c + d
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n, z)
    l2, u2 = wilson(a + c, n, z)
    margins = (a + b) * (c + d) * (a + c) * (b + d)
    phi = (a * d - b * c) / math.sqrt(margins) if margins else 0.0
    lo_term = (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2
    hi_term = (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2
    return (p1 - p2 - math.sqrt(max(lo_term, 0.0)), p1 - p2 + math.sqrt(max(hi_term, 0.0)), phi)


def newcombe_unpaired(k1: int, n1: int, k2: int, n2: int, z: float) -> tuple[float, float]:
    """Independent samples, Newcombe 1998 (Stat Med 17:873) hybrid score interval."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1, z)
    l2, u2 = wilson(k2, n2, z)
    d = p1 - p2
    return (d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2))


# ---------------------------------------------------------------- loading


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_new() -> tuple[dict[int, list[dict]], int]:
    cells, blank = {}, 0
    for a in LEVELS:
        rows = []
        for line in (CELLS / f"ironclad_A{a}.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                blank += 1
                continue
            rows.append(json.loads(line))
        cells[a] = rows
    return cells, blank


def load_old() -> dict[int, list[dict]]:
    cells: dict[int, list[dict]] = {a: [] for a in LEVELS}
    with open(OLD_CSV, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            # the published CSV: Ironclad only, no character/outcome/total_floor/wall_s columns;
            # every old Ironclad row is "won" or "lost", so outcome follows from won
            if r.get("character", "Ironclad") == "Ironclad":
                won = r["won"] == "1"
                cells[int(r["ascension"])].append(
                    {
                        "seed": int(r["seed"]),
                        "act1": r["act1"],
                        "outcome": r.get("outcome", "won" if won else "lost"),
                        "won": won,
                        "total_floor": int(r["total_floor"]) if r.get("total_floor") else None,
                        "wall_s": float(r["wall_s"]) if r.get("wall_s") else None,
                    }
                )
    return cells


def is_bad(r: dict) -> bool:
    return r["outcome"] in BAD or bool(r.get("error"))


def parse_cell_logs() -> dict[int, dict]:
    out = {}
    for a in LEVELS:
        t = (CELLS / f"ironclad_A{a}.log").read_text(encoding="utf-8")
        wr = re.search(r"^win rate: (\d+)/(\d+) = ", t, re.M)
        wl = re.search(r"^wall/run: mean ([\d.]+)s +median ([\d.]+)s +max ([\d.]+)s; batch wall (\d+)s", t, re.M)
        oc = re.search(r"^outcomes: (\{.*\})$", t, re.M)
        hd = re.search(r"=== ppo\.probe --policy (\S+) n=(\d+) asc=(\d+) char=(\S+) ===", t)
        out[a] = {
            "wins": int(wr.group(1)),
            "n": int(wr.group(2)),
            "wall": [float(wl.group(i)) for i in (1, 2, 3)],
            "batch_wall_s": int(wl.group(4)),
            "outcomes": ast.literal_eval(oc.group(1)),
            "header": [hd.group(1), int(hd.group(2)), int(hd.group(3)), hd.group(4)],
            "g1b_pass": "G1b ownership audit: PASS" in t,
        }
    return out


def parse_progress(path: Path, new: bool) -> dict:
    out: dict = {"start": {}, "done": {}, "order": [], "all_done": False, "driver": None}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^(\S+) driver start pid \d+ head=(\S+) so=(\S+) ckpt=(\S+)$", line)
        if m:
            out["driver"] = {"head": m.group(2), "so": m.group(3), "ckpt": m.group(4)}
        m = re.match(r"^(\S+) START Ironclad A(\d+) n=(\d+)(?: load ([\d.]+) ([\d.]+) ([\d.]+))?", line)
        if m:
            a = int(m.group(2))
            out["order"].append(a)
            out["start"][a] = {"ts": m.group(1), "n": int(m.group(3))}
            if m.group(4):
                out["start"][a]["load"] = [float(m.group(i)) for i in (4, 5, 6)]
        m = re.match(
            r"^(\S+) DONE\s+Ironclad A(\d+) (?:rc=(\d+) )?win rate: (\d+)/(\d+) = \S+ wall/run: mean ([\d.]+)s +"
            r"median ([\d.]+)s +max ([\d.]+)s; batch wall (\d+)s",
            line,
        )
        if m:
            out["done"][int(m.group(2))] = {
                "ts": m.group(1),
                "rc": int(m.group(3)) if m.group(3) is not None else None,
                "wins": int(m.group(4)),
                "n": int(m.group(5)),
                "wall": [float(m.group(i)) for i in (6, 7, 8)],
                "batch_wall_s": int(m.group(9)),
            }
        if "ALL DONE" in line:
            out["all_done"] = True
    return out


def parse_summary_csv() -> dict[int, dict] | None:
    if OLD_SUMMARY is None:  # not published
        return None
    out = {}
    with open(OLD_SUMMARY, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["character"] == "Ironclad":
                out[int(r["ascension"])] = {
                    "n": int(r["n"]),
                    "wins": int(r["wins"]),
                    "errors": int(r["errors"]),
                    "rate": float(r["win_rate"]),
                    "lo": float(r["ci95_lo"]),
                    "hi": float(r["ci95_hi"]),
                }
    return out


# ---------------------------------------------------------------- the third recompute


def per_level(cells: dict[int, list[dict]]) -> dict[int, dict]:
    out = {}
    for a in LEVELS:
        rows = cells[a]
        scored = [r for r in rows if not is_bad(r)]
        k, n = sum(1 for r in scored if r["won"]), len(scored)
        out[a] = {
            "n_rows": len(rows),
            "n_error": sum(1 for r in rows if r["outcome"] == "error" or r.get("error")),
            "n_timeout": sum(1 for r in rows if r["outcome"] == "timeout"),
            "outcomes": dict(collections.Counter(r["outcome"] for r in rows)),
            "wins": k,
            "n": n,
            "rate": k / n,
            "w196": wilson(k, n, Z196),
            "wex": wilson(k, n, ZEX),
        }
    return out


def act1_split(cells: dict[int, list[dict]]) -> dict[str, dict]:
    acc: dict[str, list[int]] = {}
    for a in LEVELS:
        for r in cells[a]:
            if is_bad(r):
                continue
            v = acc.setdefault(r["act1"], [0, 0])
            v[0] += int(r["won"])
            v[1] += 1
    return {
        act: {"wins": k, "n": n, "rate": k / n, "w196": wilson(k, n, Z196), "wex": wilson(k, n, ZEX)}
        for act, (k, n) in sorted(acc.items())
    }


def act1_by_level(cells: dict[int, list[dict]]) -> dict[int, dict[str, dict]]:
    out = {}
    for a in LEVELS:
        acc: dict[str, list[int]] = {}
        for r in cells[a]:
            if not is_bad(r):
                v = acc.setdefault(r["act1"], [0, 0])
                v[0] += int(r["won"])
                v[1] += 1
        out[a] = {act: {"wins": k, "n": n, "rate": k / n} for act, (k, n) in sorted(acc.items())}
    return out


def paired(new: dict, old: dict) -> tuple[dict[int, dict], dict[int, np.ndarray]]:
    out, xs = {}, {}
    for a in LEVELS:
        nm = {r["seed"]: r for r in new[a]}
        om = {r["seed"]: r for r in old[a]}
        seeds = sorted(s for s in nm.keys() & om.keys() if not is_bad(nm[s]) and not is_bad(om[s]))
        t = collections.Counter((bool(nm[s]["won"]), bool(om[s]["won"])) for s in seeds)
        A, B, C, D = t[True, True], t[True, False], t[False, True], t[False, False]
        n = len(seeds)
        lo_ex, hi_ex, phi = newcombe10(A, B, C, D, ZEX)
        lo_96, hi_96, _ = newcombe10(A, B, C, D, Z196)
        out[a] = {
            "pairs": n,
            "only_one_arm": len(nm.keys() ^ om.keys()),
            "a": A, "b": B, "c": C, "d": D,
            "new_rate": (A + B) / n,
            "old_rate": (A + C) / n,
            "diff": (B - C) / n,
            "p": mcnemar_p(B, C),
            "nc_ex": (lo_ex, hi_ex),
            "nc_196": (lo_96, hi_96),
            "phi": phi,
            "act1_agree": sum(1 for s in seeds if nm[s]["act1"] == om[s]["act1"]),
        }
        # per-seed new_won - old_won, indexed by seed offset (the panel is seeds SEED0..SEED0+n-1)
        x = np.full(N_PER[a], np.nan)
        for s in seeds:
            x[s - SEED0] = int(bool(nm[s]["won"])) - int(bool(om[s]["won"]))
        xs[a] = x
    return out, xs


def bootstrap(xs: dict[int, np.ndarray], rng_seed: int, stratified: bool) -> dict:
    """Seed-clustered percentile bootstrap of the unweighted mean of the 11 level diffs.

    Draw 500 seed offsets with replacement (stratified: 200 from 0-199 and 300 from 200-499); a
    drawn seed brings its diff at every level that holds it. The RNG call sequence is chosen to
    match analyze.py's (rng.integers(0, 500, 500) per resample; or rng.choice on the two blocks),
    so the same RNG seed must give the same draws: an exact check of its implementation.
    """
    rng = np.random.default_rng(rng_seed)
    lo_blk, hi_blk = np.arange(0, 200), np.arange(200, 500)
    stats = np.empty(BOOT_N)
    for i in range(BOOT_N):
        if stratified:
            idx = np.concatenate(
                [rng.choice(lo_blk, size=200, replace=True), rng.choice(hi_blk, size=300, replace=True)]
            )
        else:
            idx = rng.integers(0, 500, size=500)
        stats[i] = np.mean([xs[a][idx[idx < N_PER[a]]].mean() for a in LEVELS])
    return {
        "ci95": [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))],
        "se": float(np.std(stats, ddof=1)),
    }


def cluster_se(xs: dict[int, np.ndarray]) -> dict:
    """Analytic SEs of the pooled diff D = mean_a mean_s x[a, s].

    stratified: D = sum_s z_s with z_s = sum_{a holding s} x[a,s] / (11 n_a); the two seed blocks
    are strata; var = sum over strata m/(m-1) sum (z_s - zbar)^2 (the blind's estimator, coded
    here from its description). independent: levels treated as independent samples.
    """
    z = np.zeros(500)
    for a in LEVELS:
        z[: N_PER[a]] += xs[a] / (len(LEVELS) * N_PER[a])
    var = 0.0
    for blk in (z[:200], z[200:]):
        var += len(blk) / (len(blk) - 1) * float(np.sum((blk - blk.mean()) ** 2))
    var_ind = sum(float(np.var(xs[a], ddof=1)) / N_PER[a] for a in LEVELS) / len(LEVELS) ** 2
    d = float(z.sum())
    se = math.sqrt(var)
    return {
        "D": d,
        "se_stratified": se,
        "ci_normal": [d - ZEX * se, d + ZEX * se],
        "se_independent": math.sqrt(var_ind),
        "design_effect": var / var_ind,
    }


def depth(cells: dict[int, list[dict]], new: bool) -> dict[int, dict] | None:
    if not new and any(r["total_floor"] is None for a in LEVELS for r in cells[a]):
        return None  # the published old rows carry no total_floor
    out = {}
    for a in LEVELS:
        scored = [r for r in cells[a] if not is_bad(r)]
        floors = [((r.get("final") or {}).get("total_floor") if new else r["total_floor"]) for r in scored]
        died = collections.Counter()
        for r, tf in zip(scored, floors, strict=True):
            if r["won"]:
                continue
            if new:
                died[(r.get("final") or {}).get("act_index")] += 1
            else:  # analyze.py's total_floor band rule for the old rows (they lack act_index)
                died[1 if tf <= 16 else (2 if tf <= 33 else 3)] += 1
        out[a] = {
            "died": {k: died.get(k, 0) for k in (1, 2, 3)},
            "died_other": sum(v for k, v in died.items() if k not in (1, 2, 3)),
            "mean_total_floor": statistics.fmean(f for f in floors if f is not None),
        }
    return out


def cost(cells: dict[int, list[dict]]) -> dict[int, dict] | None:
    if all(r.get("wall_s") is None for a in LEVELS for r in cells[a]):
        return None  # the published old rows carry no wall_s
    out = {}
    for a in LEVELS:
        w = [r["wall_s"] for r in cells[a] if r.get("wall_s") is not None]
        by = {}
        for oc in ("won", "lost"):
            v = [r["wall_s"] for r in cells[a] if r["outcome"] == oc and r.get("wall_s") is not None]
            if v:
                by[oc] = {"n": len(v), "mean": statistics.fmean(v), "median": statistics.median(v)}
        out[a] = {
            "n": len(w),
            "mean": statistics.fmean(w),
            "median": statistics.median(w),
            "max": max(w),
            "min": min(w),
            "sum_h": sum(w) / 3600,
            "by_outcome": by,
        }
    return out


def a9_vs_a10_estimate(new: dict) -> dict:
    a9 = {r["seed"]: bool(r["won"]) for r in new[9] if not is_bad(r)}
    a10 = {r["seed"]: r.get("reached_second_boss") is True for r in new[10] if not is_bad(r)}
    keys = sorted(a9.keys() & a10.keys())
    t = collections.Counter((a9[s], a10[s]) for s in keys)
    A, B, C, D = t[True, True], t[True, False], t[False, True], t[False, False]
    n = len(keys)
    lo, hi, _ = newcombe10(A, B, C, D, Z196)
    won_a10 = {r["seed"]: bool(r["won"]) for r in new[10] if not is_bad(r)}
    return {
        "n": n, "a": A, "b": B, "c": C, "d": D,
        "a9_direct": (A + B) / n,
        "a10_estimate": (A + C) / n,
        "diff": (B - C) / n,
        "p": mcnemar_p(B, C),
        "nc_196": (lo, hi),
        "won_implies_reached": all(a10[s] for s in keys if won_a10[s]),
    }


def integrity(new: dict, old: dict, blank: int, logs: dict, prog: dict) -> dict:
    per = {}
    for a in LEVELS:
        rows = new[a]
        seeds = [r["seed"] for r in rows]
        wheels = [r.get("wheel") or {} for r in rows]
        per[a] = {
            "rows": len(rows),
            "unique_seeds": len(set(seeds)),
            "seed_min": min(seeds),
            "seed_max": max(seeds),
            "seeds_exact": len(seeds) == N_PER[a] and set(seeds) == set(range(SEED0, SEED0 + N_PER[a])),
            "wrong": {
                "character": sum(r.get("character") != "Ironclad" for r in rows),
                "ascension": sum(r.get("ascension") != a for r in rows),
                "final_ascension": sum((r.get("final") or {}).get("ascension") != a for r in rows),
                "policy": sum(r.get("policy") != "net_greedy" for r in rows),
                "wheel_src_hash": sum(w.get("src_hash") != SRC_HASH for w in wheels),
                "wheel_pgo": sum(w.get("pgo") is not True for w in wheels),
                "wheel_pgo_instrumented": sum(w.get("pgo_instrumented") is not False for w in wheels),
                "won_vs_outcome": sum(bool(r["won"]) != (r["outcome"] == "won") for r in rows if not is_bad(r)),
            },
        }
        oseeds = [r["seed"] for r in old[a]]
        per[a]["old_seeds_exact"] = len(oseeds) == N_PER[a] and set(oseeds) == set(range(SEED0, SEED0 + N_PER[a]))
        per[a]["old_unique_seeds"] = len(set(oseeds))
    allrows = [r for a in LEVELS for r in new[a]]
    act1_by_seed_new: dict[int, set] = collections.defaultdict(set)
    act1_by_seed_old: dict[int, set] = collections.defaultdict(set)
    for a in LEVELS:
        for r in new[a]:
            act1_by_seed_new[r["seed"]].add(r["act1"])
        for r in old[a]:
            act1_by_seed_old[r["seed"]].add(r["act1"])
    drv = prog["driver"] or {}
    return {
        "per_level": per,
        "rows_new": len(allrows),
        "rows_old": sum(len(old[a]) for a in LEVELS),
        "blank_lines_new": blank,
        "error_rows_new": sum(1 for r in allrows if r["outcome"] == "error" or r.get("error")),
        "timeout_rows_new": sum(1 for r in allrows if r["outcome"] == "timeout"),
        "error_rows_old": sum(1 for a in LEVELS for r in old[a] if r["outcome"] == "error"),
        "timeout_rows_old": sum(1 for a in LEVELS for r in old[a] if r["outcome"] == "timeout"),
        "src_hash_counts": dict(collections.Counter((r.get("wheel") or {}).get("src_hash") for r in allrows)),
        "py_src_counts": dict(collections.Counter(r.get("py_src") for r in allrows)),
        "so_path_counts": dict(collections.Counter((r.get("wheel") or {}).get("so_path") for r in allrows)),
        "profdata": sorted({(r.get("wheel") or {}).get("profdata_sha256") for r in allrows}),
        "forbidden_seed_rows": sum(1 for r in allrows if r["seed"] in FORBIDDEN),
        "seeds_with_two_act1_new": sum(1 for v in act1_by_seed_new.values() if len(v) > 1),
        "seeds_with_two_act1_old": sum(1 for v in act1_by_seed_old.values() if len(v) > 1),
        "order_run": prog["order"],
        "order_ok": prog["order"] == ORDER,
        "all_done": prog["all_done"],
        "rc_all_zero": all(prog["done"][a]["rc"] == 0 for a in LEVELS),
        "driver_pins_ok": drv.get("head") == HEAD and drv.get("so") == SO_SHA and drv.get("ckpt", "").endswith(CKPT),
        "g1b_pass_all": all(logs[a]["g1b_pass"] for a in LEVELS),
        "log_headers_ok": all(logs[a]["header"] == ["net_greedy", N_PER[a], a, "Ironclad"] for a in LEVELS),
    }


def recompute() -> dict:
    new, blank = load_new()
    old = load_old()
    logs = parse_cell_logs()
    prog = parse_progress(PROGRESS, new=True)
    old_prog = parse_progress(OLD_PROGRESS, new=False) if OLD_PROGRESS is not None else None
    nl, ol = per_level(new), per_level(old)
    pr, xs = paired(new, old)
    mean_new = statistics.fmean(nl[a]["rate"] for a in LEVELS)
    mean_old = statistics.fmean(ol[a]["rate"] for a in LEVELS)
    boot = bootstrap(xs, 20261026, stratified=False)
    boot_strat = bootstrap(xs, 20261027, stratified=True)
    boot_mc = [bootstrap(xs, s, stratified=False) for s in (1, 2, 3, 4, 5)]
    run_b = sum(pr[a]["b"] for a in LEVELS)
    run_c = sum(pr[a]["c"] for a in LEVELS)
    run_n = sum(pr[a]["pairs"] for a in LEVELS)
    ab_new, ab_old = act1_by_level(new), act1_by_level(old)
    acts = sorted(ab_new[0])
    act_of_seed = {r["seed"]: r["act1"] for a in LEVELS for r in new[a]}
    act1_mix = {
        "0_199": dict(collections.Counter(v for s, v in act_of_seed.items() if s - SEED0 < 200)),
        "200_499": dict(collections.Counter(v for s, v in act_of_seed.items() if s - SEED0 >= 200)),
    }
    return {
        "act1_mix": act1_mix,
        "new": nl,
        "old": ol,
        "mean_new": mean_new,
        "mean_old": mean_old,
        "mean_diff": statistics.fmean(pr[a]["diff"] for a in LEVELS),
        "paired": pr,
        "boot": boot,
        "boot_strat": boot_strat,
        "boot_mc_seeds_1_5": boot_mc,
        "cluster": cluster_se(xs),
        "run_pooled": {"n": run_n, "b": run_b, "c": run_c, "diff": (run_b - run_c) / run_n},
        "act1_new": act1_split(new),
        "act1_old": act1_split(old),
        "act1_by_level_new": ab_new,
        "act1_std_new": {act: statistics.fmean(ab_new[a][act]["rate"] for a in LEVELS) for act in acts},
        "act1_std_old": {act: statistics.fmean(ab_old[a][act]["rate"] for a in LEVELS) for act in acts},
        "act1_agree": sum(pr[a]["act1_agree"] for a in LEVELS),
        "depth_new": depth(new, new=True),
        "depth_old": depth(old, new=False),
        "cost_new": cost(new),
        "cost_old": cost(old),
        "a9_vs_a10est": a9_vs_a10_estimate(new),
        "integrity": integrity(new, old, blank, logs, prog),
        "logs": logs,
        "prog": prog,
        "old_prog": old_prog,
        "summary_csv": parse_summary_csv(),
        "lost_won_floor_new": {
            a: dict(collections.Counter((r.get("final") or {}).get("total_floor") for r in new[a] if r["outcome"] == "won"))
            for a in LEVELS
        },
        "rsb_new": {
            a: dict(collections.Counter(f"{r['outcome']}:{r.get('reached_second_boss')}" for r in new[a])) for a in LEVELS
        },
    }


# ---------------------------------------------------------------- comparisons


class Checks:
    def __init__(self) -> None:
        self.items: list[dict] = []

    def add(self, source: str, cid: str, mine, theirs, rule: str) -> None:
        if theirs is _MISSING:
            ok, note = False, "value missing in source"
        elif rule == "exact":
            ok, note = mine == theirs, None
        elif rule == "p_rel":
            ok, note = theirs is not None and abs(mine - theirs) <= TOLS["p_rel"] * abs(mine), None
        else:
            ok, note = theirs is not None and abs(mine - theirs) <= TOLS[rule], None
        rec = {"source": source, "id": cid, "mine": _js(mine), "theirs": _js(theirs), "rule": rule, "ok": bool(ok)}
        if note:
            rec["note"] = note
        self.items.append(rec)

    def pct1(self, source: str, cid: str, mine: float, printed: str, signed: bool = False) -> None:
        s = f"{100 * mine:+.1f}" if signed else f"{100 * mine:.1f}"
        self.items.append({"source": source, "id": cid, "mine": s, "theirs": printed, "rule": "printed_1dp", "ok": s == printed})


_MISSING = object()


def _js(x):
    if x is _MISSING:
        return "<missing>"
    if isinstance(x, tuple):
        return list(x)
    return x


def g(d, *path):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return _MISSING
        d = d[p]
    return d


def compare_primary(m: dict, P: dict, ck: Checks, src: str = "primary") -> None:
    for arm, key in (("new", "new"), ("old", "old")):
        for a in LEVELS:
            x, y = m[arm][a], g(P, key, "per_level", str(a))
            ck.add(src, f"{arm}.A{a}.wins", x["wins"], g(y, "wins"), "exact")
            ck.add(src, f"{arm}.A{a}.n", x["n"], g(y, "n"), "exact")
            ck.add(src, f"{arm}.A{a}.rate", x["rate"], g(y, "rate"), "full")
            for i, e in enumerate(("lo", "hi")):
                ck.add(src, f"{arm}.A{a}.wilson_zex.{e}", x["wex"][i], g(y, "wilson95")[i], "full")
                ck.add(src, f"{arm}.A{a}.wilson_z196.{e}", x["w196"][i], g(y, "wilson95_summarize_py")[i], "full")
        ck.add(src, f"{arm}.mean_A0_A10", m["mean_" + arm], g(P, key, "mean_A0_A10"), "full")
        for act in ("overgrowth", "underdocks"):
            x, y = m["act1_" + arm][act], g(P, key, "act1_split", act)
            ck.add(src, f"{arm}.act1.{act}.wins", x["wins"], g(y, "wins"), "exact")
            ck.add(src, f"{arm}.act1.{act}.n", x["n"], g(y, "n"), "exact")
            ck.add(src, f"{arm}.act1.{act}.rate", x["rate"], g(y, "rate"), "full")
            for i, e in enumerate(("lo", "hi")):
                ck.add(src, f"{arm}.act1.{act}.wilson_zex.{e}", x["wex"][i], g(y, "wilson95")[i], "full")
    ck.add(src, "old_vs_published.matches", True, g(P, "old_vs_published", "matches_published"), "exact")
    for a in LEVELS:
        x, y = m["paired"][a], g(P, "paired", "per_level", str(a))
        ck.add(src, f"paired.A{a}.pairs", x["pairs"], g(y, "pairs"), "exact")
        ck.add(src, f"paired.A{a}.dropped", [], g(y, "dropped_seeds"), "exact")
        ck.add(src, f"paired.A{a}.only_one_arm", x["only_one_arm"], g(y, "seeds_only_in_one_arm"), "exact")
        ck.add(src, f"paired.A{a}.a", x["a"], g(y, "both_won"), "exact")
        ck.add(src, f"paired.A{a}.b", x["b"], g(y, "b_new_won_old_lost"), "exact")
        ck.add(src, f"paired.A{a}.c", x["c"], g(y, "c_old_won_new_lost"), "exact")
        ck.add(src, f"paired.A{a}.d", x["d"], g(y, "both_lost"), "exact")
        ck.add(src, f"paired.A{a}.new_rate", x["new_rate"], g(y, "new_rate"), "full")
        ck.add(src, f"paired.A{a}.old_rate", x["old_rate"], g(y, "old_rate"), "full")
        ck.add(src, f"paired.A{a}.diff", x["diff"], g(y, "diff"), "full")
        ck.add(src, f"paired.A{a}.mcnemar_p", x["p"], g(y, "mcnemar_exact_p"), "p_rel")
        ck.add(src, f"paired.A{a}.mcnemar_p_scipy", x["p"], g(y, "mcnemar_exact_p_scipy"), "p_rel")
        for i, e in enumerate(("lo", "hi")):
            ck.add(src, f"paired.A{a}.newcombe_zex.{e}", x["nc_ex"][i], g(y, "newcombe10_ci95")[i], "full")
        ck.add(src, f"paired.A{a}.act1_agree", x["act1_agree"], g(y, "act1_agree"), "exact")
    ck.add(src, "paired.act1_agreement.agree", m["act1_agree"], g(P, "paired", "act1_agreement", "agree"), "exact")
    ck.add(src, "paired.act1_agreement.pairs", m["run_pooled"]["n"], g(P, "paired", "act1_agreement", "pairs"), "exact")
    pooled = g(P, "paired", "pooled")
    ck.add(src, "pooled.mean_diff", m["mean_diff"], g(pooled, "mean_diff_A0_A10"), "full")
    for i, e in enumerate(("lo", "hi")):
        ck.add(src, f"pooled.bootstrap.ci95.{e}", m["boot"]["ci95"][i], g(pooled, "bootstrap", "ci95")[i], "full")
        ck.add(src, f"pooled.bootstrap_strat.ci95.{e}", m["boot_strat"]["ci95"][i],
               g(pooled, "bootstrap_stratified_sensitivity", "ci95")[i], "full")
    ck.add(src, "pooled.bootstrap.se", m["boot"]["se"], g(pooled, "bootstrap", "se"), "full")
    ck.add(src, "pooled.bootstrap_strat.se", m["boot_strat"]["se"], g(pooled, "bootstrap_stratified_sensitivity", "se"), "full")
    # integrity
    integ, PI = m["integrity"], g(P, "integrity")
    ck.add(src, "integrity.rows_new", integ["rows_new"], g(PI, "new_total_rows"), "exact")
    ck.add(src, "integrity.error_rows_new", integ["error_rows_new"], g(PI, "new_error_rows"), "exact")
    ck.add(src, "integrity.timeout_rows_new", integ["timeout_rows_new"], g(PI, "new_timeout_rows"), "exact")
    ck.add(src, "integrity.rows_old", integ["rows_old"], g(PI, "old_total_rows"), "exact")
    ck.add(src, "integrity.error_rows_old", integ["error_rows_old"], g(PI, "old_error_rows"), "exact")
    ck.add(src, "integrity.timeout_rows_old", integ["timeout_rows_old"], g(PI, "old_timeout_rows"), "exact")
    ck.add(src, "integrity.old_seeds_ok", all(integ["per_level"][a]["old_seeds_exact"] for a in LEVELS), g(PI, "old_seeds_ok"), "exact")
    ck.add(src, "integrity.so_paths", sorted(integ["so_path_counts"]), g(PI, "new_distinct_so_path"), "exact")
    ck.add(src, "integrity.py_src", sorted(integ["py_src_counts"]), g(PI, "new_distinct_py_src"), "exact")
    ck.add(src, "integrity.profdata", integ["profdata"], g(PI, "new_distinct_profdata_sha256"), "exact")
    ck.add(src, "integrity.order_run", integ["order_run"], g(PI, "cell_order_run"), "exact")
    ck.add(src, "integrity.order_ok", integ["order_ok"], g(PI, "cell_order_ok"), "exact")
    ck.add(src, "integrity.all_done", integ["all_done"], g(PI, "progress_all_done"), "exact")
    ck.add(src, "integrity.driver_pins_ok", integ["driver_pins_ok"], g(PI, "driver_pins_ok"), "exact")
    all_ok_mine = all(
        integ["per_level"][a]["seeds_exact"] and not any(integ["per_level"][a]["wrong"].values()) for a in LEVELS
    ) and integ["log_headers_ok"]
    ck.add(src, "integrity.new_all_cells_ok", all_ok_mine, g(PI, "new_all_cells_ok"), "exact")
    for a in LEVELS:
        y = g(PI, "new_per_level", str(a))
        ck.add(src, f"integrity.A{a}.seeds_ok", integ["per_level"][a]["seeds_exact"], g(y, "seeds", "ok"), "exact")
        ck.add(src, f"integrity.A{a}.outcomes", m["new"][a]["outcomes"], g(y, "outcomes"), "exact")
        ck.add(src, f"integrity.A{a}.error_rows", m["new"][a]["n_error"], g(y, "error_rows"), "exact")
        ck.add(src, f"integrity.A{a}.timeout_rows", m["new"][a]["n_timeout"], g(y, "timeout_rows"), "exact")
        ck.add(src, f"integrity.A{a}.wrong_field_counts", integ["per_level"][a]["wrong"], g(y, "wrong_field_counts"), "exact")
        ck.add(src, f"integrity.A{a}.log_g1b_pass", m["logs"][a]["g1b_pass"], g(y, "log_g1b_pass"), "exact")
        ck.add(src, f"integrity.A{a}.jsonl_sha256", sha256(CELLS / f"ironclad_A{a}.jsonl"), g(y, "jsonl_sha256"), "exact")
        yo = g(PI, "old_per_level", str(a))
        ck.add(src, f"integrity.old.A{a}.outcomes", m["old"][a]["outcomes"], g(yo, "outcomes"), "exact")
    # cost
    for a in LEVELS:
        x, y = m["cost_new"][a], g(P, "cost", "per_level", str(a))
        xo = m["cost_old"][a] if m["cost_old"] is not None else None
        ck.add(src, f"cost.A{a}.runs", x["n"], g(y, "runs"), "exact")
        ck.add(src, f"cost.A{a}.wall_mean", x["mean"], g(y, "wall_s_mean"), "full")
        ck.add(src, f"cost.A{a}.wall_median", x["median"], g(y, "wall_s_median"), "full")
        ck.add(src, f"cost.A{a}.wall_max", x["max"], g(y, "wall_s_max"), "full")
        ck.add(src, f"cost.A{a}.cpu_hours", x["sum_h"], g(y, "cpu_hours"), "full")
        ck.add(src, f"cost.A{a}.batch_wall_s", m["logs"][a]["batch_wall_s"], g(y, "batch_wall_s_log"), "exact")
        ck.add(src, f"cost.A{a}.load", m["prog"]["start"][a]["load"], g(y, "load_at_start_1_5_15"), "exact")
        if xo is not None:  # the old rows carry wall_s
            ck.add(src, f"cost.A{a}.old_wall_mean", xo["mean"], g(y, "old_wall_s_mean"), "full")
            ck.add(src, f"cost.A{a}.old_wall_median", xo["median"], g(y, "old_wall_s_median"), "full")
            ck.add(src, f"cost.A{a}.old_cpu_hours", xo["sum_h"], g(y, "old_cpu_hours"), "full")
        if m["old_prog"] is not None:
            ck.add(src, f"cost.A{a}.old_batch_wall_s", m["old_prog"]["done"][a]["batch_wall_s"], g(y, "old_batch_wall_s"), "exact")
        if xo is not None:
            ck.add(src, f"cost.A{a}.ratio", x["mean"] / xo["mean"], g(y, "wall_s_mean_ratio_new_over_old"), "full")
    tot = g(P, "cost", "total")
    ck.add(src, "cost.total.cpu_hours", sum(m["cost_new"][a]["sum_h"] for a in LEVELS), g(tot, "cpu_hours"), "full")
    if m["cost_old"] is not None:
        ck.add(src, "cost.total.old_cpu_hours", sum(m["cost_old"][a]["sum_h"] for a in LEVELS), g(tot, "old_cpu_hours"), "full")
    ck.add(src, "cost.total.batch_wall_h", sum(m["logs"][a]["batch_wall_s"] for a in LEVELS) / 3600, g(tot, "batch_wall_h"), "full")
    if m["old_prog"] is not None:
        ck.add(src, "cost.total.old_batch_wall_h", sum(m["old_prog"]["done"][a]["batch_wall_s"] for a in LEVELS) / 3600,
               g(tot, "old_batch_wall_h"), "full")
    # depth
    for arm in ("new", "old"):
        if m["depth_" + arm] is None:  # the old rows carry no total_floor
            continue
        for a in LEVELS:
            x, y = m["depth_" + arm][a], g(P, "depth_" + arm, "per_level", str(a))
            for k in (1, 2, 3):
                ck.add(src, f"depth_{arm}.A{a}.died_act{k}", x["died"][k], g(y, f"died_act{k}"), "exact")
            ck.add(src, f"depth_{arm}.A{a}.died_unknown", x["died_other"], g(y, "died_unknown_act"), "exact")
            ck.add(src, f"depth_{arm}.A{a}.mean_total_floor", x["mean_total_floor"], g(y, "mean_total_floor"), "full")


def compare_primary_md(m: dict, text: str, ck: Checks, src: str = "primary_md") -> None:
    def fmt_p(p: float) -> str:  # primary.md's printed precision
        return f"{p:.2g}" if p < 0.001 else f"{p:.3f}"

    num = r"[+-]?\d+(?:\.\d+)?"
    row = re.compile(
        rf"^\| A(\d+) \| (\d+) \| (\d+)/(\d+) = ({num}) \[({num})-({num})\] \| (\d+)/(\d+) = ({num}) "
        rf"\[({num})-({num})\] \| (\d+) \| ({num}) \| \[({num}), ({num})\] \| (\d+) / (\d+) \| (\S+) \|$",
        re.M,
    )
    seen = set()
    for mm in row.finditer(text):
        a = int(mm.group(1))
        seen.add(a)
        nl, ol, pr = m["new"][a], m["old"][a], m["paired"][a]
        ck.add(src, f"A{a}.n", nl["n"], int(mm.group(2)), "exact")
        ck.add(src, f"A{a}.new_wins", nl["wins"], int(mm.group(3)), "exact")
        ck.pct1(src, f"A{a}.new_rate", nl["rate"], mm.group(5))
        ck.pct1(src, f"A{a}.new_lo", nl["wex"][0], mm.group(6))
        ck.pct1(src, f"A{a}.new_hi", nl["wex"][1], mm.group(7))
        ck.add(src, f"A{a}.old_wins", ol["wins"], int(mm.group(8)), "exact")
        ck.pct1(src, f"A{a}.old_rate", ol["rate"], mm.group(10))
        ck.pct1(src, f"A{a}.old_lo", ol["wex"][0], mm.group(11))
        ck.pct1(src, f"A{a}.old_hi", ol["wex"][1], mm.group(12))
        ck.add(src, f"A{a}.pairs", pr["pairs"], int(mm.group(13)), "exact")
        ck.pct1(src, f"A{a}.diff", pr["diff"], mm.group(14), signed=True)
        ck.pct1(src, f"A{a}.nc_lo", pr["nc_ex"][0], mm.group(15), signed=True)
        ck.pct1(src, f"A{a}.nc_hi", pr["nc_ex"][1], mm.group(16), signed=True)
        ck.add(src, f"A{a}.b", pr["b"], int(mm.group(17)), "exact")
        ck.add(src, f"A{a}.c", pr["c"], int(mm.group(18)), "exact")
        ck.add(src, f"A{a}.p_printed", fmt_p(pr["p"]), mm.group(19), "exact")
    ck.add(src, "level_rows_found", LEVELS, sorted(seen), "exact")
    mr = re.search(
        rf"^\| \*\*mean A0-A10\*\* \| ([\d,]+) \| \*\*({num})\*\* \| \*\*({num})\*\* \| ([\d,]+) \| "
        rf"\*\*({num})\*\* \| \[({num}), ({num})\]",
        text,
        re.M,
    )
    if not mr:
        ck.add(src, "mean_row", "present", _MISSING, "exact")
    else:
        ck.add(src, "mean_row.n", sum(m["new"][a]["n"] for a in LEVELS), int(mr.group(1).replace(",", "")), "exact")
        ck.pct1(src, "mean_row.new", m["mean_new"], mr.group(2))
        ck.pct1(src, "mean_row.old", m["mean_old"], mr.group(3))
        ck.add(src, "mean_row.pairs", m["run_pooled"]["n"], int(mr.group(4).replace(",", "")), "exact")
        ck.pct1(src, "mean_row.diff", m["mean_diff"], mr.group(5), signed=True)
        ck.pct1(src, "mean_row.boot_lo", m["boot"]["ci95"][0], mr.group(6), signed=True)
        ck.pct1(src, "mean_row.boot_hi", m["boot"]["ci95"][1], mr.group(7), signed=True)
    for arm, label in (("new", "new: "), ("old", "old: ")):
        ar = re.search(rf"^\| {label}.*\| (\d+)/(\d+) = ({num}) \| (\d+)/(\d+) = ({num}) \|$", text, re.M)
        if not ar:
            ck.add(src, f"act1_row.{arm}", "present", _MISSING, "exact")
            continue
        for j, act in ((1, "overgrowth"), (4, "underdocks")):
            x = m["act1_" + arm][act]
            ck.add(src, f"act1.{arm}.{act}.wins", x["wins"], int(ar.group(j)), "exact")
            ck.add(src, f"act1.{arm}.{act}.n", x["n"], int(ar.group(j + 1)), "exact")
            ck.pct1(src, f"act1.{arm}.{act}.rate", x["rate"], ar.group(j + 2))
    nr = re.search(r"New rows: (\d+); error rows (\d+), timeout rows (\d+)", text)
    ck.add(src, "integrity.rows_line", [m["integrity"]["rows_new"], m["integrity"]["error_rows_new"], m["integrity"]["timeout_rows_new"]],
           [int(nr.group(i)) for i in (1, 2, 3)] if nr else _MISSING, "exact")
    pa = re.search(r"act1 agrees between the arms on (\d+) of (\d+) paired runs", text)
    ck.add(src, "act1_agreement_line", [m["act1_agree"], m["run_pooled"]["n"]],
           [int(pa.group(1)), int(pa.group(2))] if pa else _MISSING, "exact")


def compare_blind(m: dict, B: dict, ck: Checks, src: str = "blind") -> None:
    for a in LEVELS:
        x, y = m["new"][a], g(B, "per_level_new", f"A{a}")
        ck.add(src, f"new.A{a}.n_rows", x["n_rows"], g(y, "n_rows"), "exact")
        ck.add(src, f"new.A{a}.n_error", x["n_error"], g(y, "n_error"), "exact")
        ck.add(src, f"new.A{a}.n_timeout", x["n_timeout"], g(y, "n_timeout"), "exact")
        ck.add(src, f"new.A{a}.other_outcomes", {k: v for k, v in x["outcomes"].items() if k not in ("won", "lost", "error", "timeout")},
               g(y, "n_other_nonwin_outcomes"), "exact")
        ck.add(src, f"new.A{a}.wins", x["wins"], g(y, "wins"), "exact")
        ck.add(src, f"new.A{a}.n", x["n"], g(y, "n_scored"), "exact")
        ck.add(src, f"new.A{a}.rate", x["rate"], g(y, "rate"), "r6")
        for i, e in enumerate(("lo", "hi")):
            ck.add(src, f"new.A{a}.wilson_z196.{e}", x["w196"][i], g(y, "wilson95")[i], "r6")
        x, y = m["old"][a], g(B, "old_column_recomputed", "per_level", f"A{a}")
        ck.add(src, f"old.A{a}.n_error", x["n_error"], g(y, "n_error"), "exact")
        ck.add(src, f"old.A{a}.n_timeout", x["n_timeout"], g(y, "n_timeout"), "exact")
        ck.add(src, f"old.A{a}.wins", x["wins"], g(y, "wins"), "exact")
        ck.add(src, f"old.A{a}.n", x["n"], g(y, "n_scored"), "exact")
        ck.add(src, f"old.A{a}.rate", x["rate"], g(y, "rate"), "r6")
        for i, e in enumerate(("lo", "hi")):
            ck.add(src, f"old.A{a}.wilson_z196.{e}", x["w196"][i], g(y, "wilson95")[i], "r6")
    ck.add(src, "new.mean_A0_A10", m["mean_new"], g(B, "unweighted_mean_A0_A10_new"), "r6")
    ck.add(src, "old.mean_A0_A10", m["mean_old"], g(B, "old_column_recomputed", "unweighted_mean_A0_A10"), "r6")
    ck.add(src, "old.matches_published", [True, True], [g(B, "old_column_recomputed", "all_levels_match"),
                                                         g(B, "old_column_recomputed", "mean_match")], "exact")
    for arm, path in (("new", ("act1_split_new_pooled_A0_A10",)), ("old", ("old_column_recomputed", "act1_split"))):
        for act in ("overgrowth", "underdocks"):
            x, y = m["act1_" + arm][act], g(B, *path, act)
            ck.add(src, f"{arm}.act1.{act}.wins", x["wins"], g(y, "wins"), "exact")
            ck.add(src, f"{arm}.act1.{act}.n", x["n"], g(y, "n"), "exact")
            ck.add(src, f"{arm}.act1.{act}.rate", x["rate"], g(y, "rate"), "r6")
            for i, e in enumerate(("lo", "hi")):
                ck.add(src, f"{arm}.act1.{act}.wilson_z196.{e}", x["w196"][i], g(y, "wilson95")[i], "r6")
    for a in LEVELS:
        for act, v in m["act1_by_level_new"][a].items():
            y = g(B, "act1_split_new_by_level", f"A{a}", act)
            ck.add(src, f"new.act1_by_level.A{a}.{act}.wins_n", [v["wins"], v["n"]], [g(y, "wins"), g(y, "n")], "exact")
    for arm in ("new", "old"):
        for act, v in m["act1_std_" + arm].items():
            ck.add(src, f"{arm}.act1_level_standardized.{act}", v,
                   g(B, "act1_level_standardized", arm, "level_standardized_rate", act), "r6")
    for a in LEVELS:
        x, y = m["paired"][a], g(B, "paired_vs_old", f"A{a}")
        ck.add(src, f"paired.A{a}.pairs", x["pairs"], g(y, "n_pairs"), "exact")
        ck.add(src, f"paired.A{a}.only_one_arm", x["only_one_arm"], g(y, "n_seeds_only_new") + g(y, "n_seeds_only_old")
               if g(y, "n_seeds_only_new") is not _MISSING else _MISSING, "exact")
        ck.add(src, f"paired.A{a}.table", [x["a"], x["b"], x["c"], x["d"]],
               [g(y, "table", "a_both_won"), g(y, "table", "b_new_won_old_lost"), g(y, "table", "c_new_lost_old_won"),
                g(y, "table", "d_both_lost")], "exact")
        ck.add(src, f"paired.A{a}.b", x["b"], g(y, "b"), "exact")
        ck.add(src, f"paired.A{a}.c", x["c"], g(y, "c"), "exact")
        ck.add(src, f"paired.A{a}.new_rate", x["new_rate"], g(y, "new"), "r6")
        ck.add(src, f"paired.A{a}.old_rate", x["old_rate"], g(y, "old"), "r6")
        ck.add(src, f"paired.A{a}.diff", x["diff"], g(y, "diff"), "r6")
        ck.add(src, f"paired.A{a}.mcnemar_p", x["p"], g(y, "mcnemar_exact_p"), "p_rel")
        for i, e in enumerate(("lo", "hi")):
            ck.add(src, f"paired.A{a}.newcombe_z196.{e}", x["nc_196"][i], g(y, "newcombe10_95")[i], "r6")
        ck.add(src, f"paired.A{a}.phi", x["phi"], g(y, "phi"), "r6")
    pp = g(B, "pooled_paired_A0_A10")
    cl = m["cluster"]
    ck.add(src, "pooled.mean_diff", m["mean_diff"], g(pp, "unweighted_mean_diff_A0_A10"), "r6")
    ck.add(src, "pooled.se_stratified_analytic", cl["se_stratified"], g(pp, "se_seed_clustered_stratified"), "r6")
    ck.add(src, "pooled.se_levels_independent", cl["se_independent"], g(pp, "se_if_levels_were_independent"), "r6")
    ck.add(src, "pooled.design_effect", cl["design_effect"], g(pp, "design_effect_var_ratio"), "r6")
    for i, e in enumerate(("lo", "hi")):
        ck.add(src, f"pooled.ci_normal_analytic.{e}", cl["ci_normal"][i], g(pp, "ci95_seed_clustered_normal")[i], "r6")
        # cross-method: this recompute's percentile bootstrap vs the blind's analytic normal interval
        ck.add(src, f"pooled.ci_bootstrap_vs_blind_analytic.{e}", m["boot"]["ci95"][i], g(pp, "ci95_seed_clustered_normal")[i], "method")
    rp = g(pp, "run_pooled")
    ck.add(src, "pooled.run_pooled", [m["run_pooled"]["n"], m["run_pooled"]["b"], m["run_pooled"]["c"]],
           [g(rp, "n_pairs"), g(rp, "b"), g(rp, "c")], "exact")
    ck.add(src, "pooled.run_pooled.diff", m["run_pooled"]["diff"], g(rp, "diff"), "r6")
    ag = g(B, "act1_agreement_new_vs_old")
    ck.add(src, "act1_agreement.total", [m["run_pooled"]["n"], m["act1_agree"]], [g(ag, "total", "n"), g(ag, "total", "agree")], "exact")
    ck.add(src, "act1_one_per_seed.new", m["integrity"]["seeds_with_two_act1_new"],
           len(g(ag, "within_new_arm_across_levels", "seeds_with_more_than_one_act1_across_levels")), "exact")
    ck.add(src, "act1_one_per_seed.old", m["integrity"]["seeds_with_two_act1_old"],
           len(g(ag, "within_old_arm_across_levels", "seeds_with_more_than_one_act1_across_levels")), "exact")
    # A9 direct vs the A9 estimate read off A10 (post-hoc in the blind; checked here, not pre-registered)
    x, y = m["a9_vs_a10est"], g(B, "a9_direct_vs_a10_estimate_new_arm")
    ck.add(src, "a9_vs_a10est.table", [x["n"], x["a"], x["b"], x["c"], x["d"]],
           [g(y, "n_pairs"), g(y, "table", "a_both"), g(y, "table", "b_a9won_a10notpast"), g(y, "table", "c_a9lost_a10past"),
            g(y, "table", "d_neither")], "exact")
    ck.add(src, "a9_vs_a10est.a9_direct", x["a9_direct"], g(y, "a9_direct_rate"), "r6")
    ck.add(src, "a9_vs_a10est.a10_estimate", x["a10_estimate"], g(y, "a10_estimate_rate"), "r6")
    ck.add(src, "a9_vs_a10est.diff", x["diff"], g(y, "diff_direct_minus_estimate"), "r6")
    ck.add(src, "a9_vs_a10est.p", x["p"], g(y, "mcnemar_exact_p"), "p_rel")
    for i, e in enumerate(("lo", "hi")):
        ck.add(src, f"a9_vs_a10est.newcombe_z196.{e}", x["nc_196"][i], g(y, "newcombe10_95")[i], "r6")
    # loss profile
    for a in LEVELS:
        y = g(B, "loss_profile_new", f"A{a}")
        d = m["depth_new"][a]["died"]
        ck.add(src, f"loss_profile.A{a}.lost_by_act", {str(k): v for k, v in d.items() if v},
               g(y, "lost_by_final_act_index"), "exact")
        ck.add(src, f"loss_profile.A{a}.won_total_floor", {str(k): v for k, v in m["lost_won_floor_new"][a].items()},
               g(y, "won_total_floor"), "exact")
        ck.add(src, f"loss_profile.A{a}.outcome_by_rsb", m["rsb_new"][a], g(y, "outcome_by_reached_second_boss"), "exact")
    # cost
    for a in LEVELS:
        x, y = m["cost_new"][a], g(B, "cost_wall_s", f"A{a}")
        xo = m["cost_old"][a] if m["cost_old"] is not None else None
        for k in ("mean", "median", "max", "min", "sum_h"):
            ck.add(src, f"cost.A{a}.{k}", x[k], g(y, f"{k}_s" if k != "sum_h" else "sum_h"), "r6")
            if xo is not None:  # the old rows carry wall_s
                ck.add(src, f"cost.A{a}.old_{k}", xo[k], g(y, "old_arm", f"{k}_s" if k != "sum_h" else "sum_h"), "r6")
        for oc, v in x["by_outcome"].items():
            ck.add(src, f"cost.A{a}.{oc}.n", v["n"], g(y, "by_outcome", oc, "n"), "exact")
            ck.add(src, f"cost.A{a}.{oc}.mean", v["mean"], g(y, "by_outcome", oc, "mean_s"), "r6")
            ck.add(src, f"cost.A{a}.{oc}.median", v["median"], g(y, "by_outcome", oc, "median_s"), "r6")
        ck.add(src, f"cost.A{a}.load", m["prog"]["start"][a]["load"], g(y, "start_load_1_5_15"), "exact")
    allw = [m["cost_new"][a] for a in LEVELS]
    co = g(B, "cost_overall")
    ck.add(src, "cost.overall.new_sum_h", sum(c["sum_h"] for c in allw), g(co, "new_sum_h"), "r6")
    ck.add(src, "cost.overall.new_mean_s", sum(c["sum_h"] for c in allw) * 3600 / sum(c["n"] for c in allw), g(co, "new_mean_s"), "r6")
    if m["cost_old"] is not None:
        allo = [m["cost_old"][a] for a in LEVELS]
        ck.add(src, "cost.overall.old_sum_h", sum(c["sum_h"] for c in allo), g(co, "old_sum_h"), "r6")
        ck.add(src, "cost.overall.old_mean_s", sum(c["sum_h"] for c in allo) * 3600 / sum(c["n"] for c in allo), g(co, "old_mean_s"), "r6")
    # integrity
    integ, BI = m["integrity"], g(B, "integrity")
    n_wrong = sum(sum(integ["per_level"][a]["wrong"].values()) for a in LEVELS) + integ["blank_lines_new"]
    ck.add(src, "integrity.row_problems", n_wrong, len(g(BI, "row_problems")), "exact")
    ck.add(src, "integrity.rows_new", integ["rows_new"], g(BI, "n_rows_total_new"), "exact")
    ck.add(src, "integrity.rows_old", integ["rows_old"], g(BI, "n_rows_total_old_ironclad"), "exact")
    ck.add(src, "integrity.src_hash_counts", integ["src_hash_counts"], g(BI, "wheel_src_hash"), "exact")
    ck.add(src, "integrity.py_src_counts", integ["py_src_counts"], g(BI, "py_src"), "exact")
    ck.add(src, "integrity.so_path_counts", integ["so_path_counts"], g(BI, "wheel_so_path"), "exact")
    ck.add(src, "integrity.forbidden_seed_rows", integ["forbidden_seed_rows"], len(g(BI, "forbidden_seed_rows")), "exact")
    for a in LEVELS:
        x = integ["per_level"][a]
        for arm, key in (("new", "seeds_new"), ("old", "seeds_old")):
            y = g(BI, key, f"A{a}")
            exact = x["seeds_exact"] if arm == "new" else x["old_seeds_exact"]
            ck.add(src, f"integrity.{arm}.A{a}.seeds_exact", exact, g(y, "equals_expected_range"), "exact")
            ck.add(src, f"integrity.{arm}.A{a}.n_unique", x["unique_seeds"] if arm == "new" else x["old_unique_seeds"],
                   g(y, "n_unique"), "exact")
        ck.add(src, f"integrity.A{a}.g1b", "PASS" if m["logs"][a]["g1b_pass"] else "?", g(BI, "log_cross_check", f"A{a}", "g1b_audit"), "exact")
        ck.add(src, f"integrity.A{a}.rc", m["prog"]["done"][a]["rc"], g(BI, "progress_log", f"A{a}", "rc"), "exact")


def compare_logs_and_published(m: dict, ck: Checks) -> None:
    for a in LEVELS:
        x, c, lg, dn = m["new"][a], m["cost_new"][a], m["logs"][a], m["prog"]["done"][a]
        ck.add("cell_log", f"A{a}.wins_n", [x["wins"], x["n"]], [lg["wins"], lg["n"]], "exact")
        ck.add("cell_log", f"A{a}.outcomes", x["outcomes"], lg["outcomes"], "exact")
        ck.add("cell_log", f"A{a}.header", ["net_greedy", x["n_rows"], a, "Ironclad"], lg["header"], "exact")
        for i, k in enumerate(("mean", "median", "max")):
            ck.add("cell_log", f"A{a}.wall_{k}", c[k], lg["wall"][i], "log1")
            ck.add("progress_log", f"A{a}.wall_{k}", c[k], dn["wall"][i], "log1")
        ck.add("progress_log", f"A{a}.wins_n", [x["wins"], x["n"]], [dn["wins"], dn["n"]], "exact")
        ck.add("progress_log", f"A{a}.rc", 0, dn["rc"], "exact")
        ck.add("progress_log", f"A{a}.batch_wall_s", lg["batch_wall_s"], dn["batch_wall_s"], "exact")
        ck.add("progress_log", f"A{a}.start_n", N_PER[a], m["prog"]["start"][a]["n"], "exact")
        # old sweep
        xo = m["old"][a]
        if m["old_prog"] is not None:  # the 2026-10-11 progress.log is private
            co, od = m["cost_old"][a], m["old_prog"]["done"][a]
            ck.add("old_progress_log", f"A{a}.wins_n", [xo["wins"], xo["n"]], [od["wins"], od["n"]], "exact")
            for i, k in enumerate(("mean", "median", "max")):
                ck.add("old_progress_log", f"A{a}.wall_{k}", co[k], od["wall"][i], "log1")
        if m["summary_csv"] is not None:  # so is its summary.csv
            sc = m["summary_csv"][a]
            ck.add("summary_csv_2026_10_11", f"A{a}.n_wins_errors", [xo["n"], xo["wins"], xo["n_error"]], [sc["n"], sc["wins"], sc["errors"]], "exact")
            ck.add("summary_csv_2026_10_11", f"A{a}.rate", xo["rate"], sc["rate"], "csv4")
            ck.add("summary_csv_2026_10_11", f"A{a}.lo", xo["w196"][0], sc["lo"], "csv4")
            ck.add("summary_csv_2026_10_11", f"A{a}.hi", xo["w196"][1], sc["hi"], "csv4")
        ck.pct1("published_2026_10_11", f"A{a}.rate", xo["rate"], PUBLISHED_OLD["rate"][a])
        ck.pct1("published_2026_10_11", f"A{a}.lo", xo["w196"][0], PUBLISHED_OLD["ci"][a][0])
        ck.pct1("published_2026_10_11", f"A{a}.hi", xo["w196"][1], PUBLISHED_OLD["ci"][a][1])
    ck.pct1("published_2026_10_11", "mean", m["mean_old"], PUBLISHED_OLD["mean"])
    for act, (n, rate) in PUBLISHED_OLD["act1"].items():
        ck.add("published_2026_10_11", f"act1.{act}.n", m["act1_old"][act]["n"], n, "exact")
        ck.pct1("published_2026_10_11", f"act1.{act}.rate", m["act1_old"][act]["rate"], rate)
    # bootstrap Monte Carlo stability: other RNG seeds vs the pre-registered one
    for j, b in enumerate(m["boot_mc_seeds_1_5"], start=1):
        for i, e in enumerate(("lo", "hi")):
            ck.add("bootstrap_mc", f"rng_seed_{j}.ci95.{e}", m["boot"]["ci95"][i], b["ci95"][i], "mc")
    # the z constant must not change any printed (1 decimal) interval
    for a in LEVELS:
        for i, e in enumerate(("lo", "hi")):
            for arm in ("new", "old"):
                ck.add("z_choice", f"{arm}.A{a}.wilson.{e}", f"{100 * m[arm][a]['w196'][i]:.1f}", f"{100 * m[arm][a]['wex'][i]:.1f}", "exact")
            ck.add("z_choice", f"paired.A{a}.newcombe.{e}",
                   f"{100 * m['paired'][a]['nc_196'][i]:+.1f}", f"{100 * m['paired'][a]['nc_ex'][i]:+.1f}", "exact")
    for arm in ("new", "old"):
        for act, v in m["act1_" + arm].items():
            for i, e in enumerate(("lo", "hi")):
                ck.add("z_choice", f"{arm}.act1.{act}.wilson.{e}", f"{100 * v['w196'][i]:.1f}", f"{100 * v['wex'][i]:.1f}", "exact")


# ---------------------------------------------------------------- explanations and fixes

FIXES = {
    "F1_blind_r6": {
        "file": "analysis/blind/recompute.py (output analysis/blind/blind.json)",
        "bug": "r6() rounded every float to 6 decimal places before writing blind.json, so the exact McNemar p-values "
        "below 1e-3 lost their precision: A10 p = 1.70e-13 and A9 p = 5.81e-10 were written as 0.0, and A8 p = 2.037e-4 "
        "as 0.000204. An exact binomial p is never 0. Only the JSON was wrong: the blind's stdout table (format 9.2e) "
        "and the blind agent's report carried the right values.",
        "evidence": "prefix_snapshot/blind.json paired_vs_old.A9/A10.mcnemar_exact_p = 0.0; scipy binomtest(143, 199) = "
        "5.806e-10, binomtest(104, 127) = 1.696e-13; primary.json has the same values at full precision.",
        "change": "r6() keeps 6 significant digits for nonzero |x| < 1e-3 and still rounds to 6 decimals otherwise. No "
        "other line changed.",
        "rerun": "<venv-h python> -I -B analysis/blind/recompute.py (self-tests pass; rewrites blind.json)",
    },
    "F2_primary_md_3400": {
        "file": "analysis/scripts/analyze.py (output analysis/primary.md)",
        "bug": "render_md() hard-coded '3,400' in the mean A0-A10 row of primary.md, in both the n and the pairs column. "
        "The sweep has 2,800 runs per arm (9 x 200 + 2 x 500) and 2,800 pairs. The 3,400 came from the task text; "
        "primary.json was never affected (its integrity block says 2800).",
        "evidence": "prefix_snapshot/primary.md mean row '| **mean A0-A10** | 3,400 | ... | 3,400 |'; 2,800 rows in "
        "cells/*.jsonl; runs.csv holds 2,800 Ironclad rows.",
        "change": "The mean row's n and pairs columns are summed from the per-level tables. No other line changed.",
        "rerun": "<venv-h python> -I -B analysis/scripts/analyze.py (rewrites primary.json and primary.md)",
    },
}

EXPLAIN = [
    ("blind", re.compile(r"^(paired\.A\d+\.mcnemar_p|a9_vs_a10est\.p)$"), "F1_blind_r6"),
    ("primary_md", re.compile(r"^mean_row\.(n|pairs)$"), "F2_primary_md_3400"),
]


def explain(item: dict) -> str:
    for src, rx, fix in EXPLAIN:
        if item["source"] == src and rx.match(item["id"]):
            return fix
    return "UNEXPLAINED"


def run_comparisons(m: dict, P: dict, md: str, B: dict, with_independent: bool) -> list[dict]:
    ck = Checks()
    compare_primary(m, P, ck)
    compare_primary_md(m, md, ck)
    compare_blind(m, B, ck)
    if with_independent:
        compare_logs_and_published(m, ck)
    return ck.items


def summarize(items: list[dict]) -> dict:
    by_src: dict[str, list[int]] = {}
    for it in items:
        v = by_src.setdefault(it["source"], [0, 0])
        v[0] += 1
        v[1] += 0 if it["ok"] else 1
    mism = []
    for it in items:
        if not it["ok"]:
            fix = explain(it)
            side = "blind.json is wrong" if fix == "F1_blind_r6" else "primary.md is wrong" if fix.startswith("F2") else "?"
            mism.append(
                dict(
                    it,
                    explanation=fix,
                    wrong_side=side,
                    explanation_text=FIXES[fix]["bug"] if fix in FIXES else "no known cause; investigate",
                )
            )
    return {
        "values_compared": len(items),
        "mismatches": len(mism),
        "by_source": {k: {"compared": v[0], "mismatches": v[1]} for k, v in sorted(by_src.items())},
        "mismatch_list": mism,
        "unexplained": sum(1 for x in mism if x["explanation"] == "UNEXPLAINED"),
    }


def leaf_diff(a, b, path: str = "") -> list[dict]:
    """Paths whose leaf values differ between two JSON trees (for documenting a fix's footprint)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b), key=str):
            out += leaf_diff(a.get(k, "<absent>"), b.get(k, "<absent>"), f"{path}.{k}" if path else str(k))
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        out = []
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            out += leaf_diff(x, y, f"{path}[{i}]")
        return out
    return [] if a == b else [{"path": path, "before": a, "after": b}]


# ---------------------------------------------------------------- final.txt


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def spp(x: float) -> str:
    return f"{100 * x:+.1f}"


def fmt_p(p: float) -> str:
    return f"{p:.1e}" if p < 0.001 else f"{p:.2g}"


def render_final(m: dict, verdict: dict) -> str:
    L: list[str] = []
    nl, ol, pr = m["new"], m["old"], m["paired"]
    integ = m["integrity"]
    L.append("IRONCLAD RELEASE WIN RATES, 2026-10-26: RECONCILED NUMBERS")
    L.append("=" * 62)
    L.append("")
    L.append("Sources: analysis/primary.json (scripts/analyze.py), analysis/blind/blind.json (blind/recompute.py),")
    L.append("and a third recompute from the raw rows (scripts/reconcile.py -> analysis/reconcile.json).")
    L.append(f"Verdict: {verdict['values_compared']} values compared across all sources, {verdict['mismatches']} mismatches after two")
    if verdict["prefix_mismatches"] is not None:
        L.append(f"reporting fixes (reconcile.json 'fixes'). Before them there were {verdict['prefix_mismatches']} mismatches, "
                 f"{verdict['prefix_unexplained']} unexplained, all from")
        L.append("those two bugs: blind.json wrote three McNemar p-values below 1e-3 rounded to 6 decimals (A9 and A10 as 0.0;")
        L.append("its stdout had them right), and primary.md printed '3,400' for the 2,800 runs and pairs of the mean row.")
        L.append("No number in primary.json changed (byte-identical after the re-run). Wilson/Newcombe intervals here use")
    else:
        L.append("reporting fixes. The pre-fix outputs (analysis/prefix_snapshot/) are not here, so the record of the two")
        L.append("fixes is not recomputed; the published reconcile.json carries it ('fixes'). Wilson/Newcombe intervals here use")
    L.append("z = 1.96 (the 2026-10-11 constant); primary.md uses z = 1.959964; every interval agrees at 1 decimal.")
    L.append("")
    L.append("WHAT WAS MEASURED")
    L.append("  NEW: Ironclad, champion a10_it700 (ppo_fullrun_ironclad_a10_it700.pt, sha256 91c83c1a...), ppo.probe")
    L.append("       --policy net_greedy --act1 roll, 8000 sims per decision with boss fights included (8000/8000),")
    L.append("       one tree, run timeout 14400 s, 30 procs. Shipped Ironclad search default (no STS2_* set): knobs")
    L.append("       removed_counts_forms=1, endturn_widening_c=23.148, draw_widening_c=4.0 in every fight, built-in")
    L.append("       BC move prior (mix 0.5, T 1) in elite and boss fights, rl2 playout net. Engine private main")
    L.append("       0181fa552, PGO wheel src_hash 4e5bb730d28e2976 (.so sha256 823fa393...). Seeds 81,000,000+:")
    L.append("       n = 200 at A0-A8, n = 500 at A9-A10, 2,800 runs.")
    L.append("  OLD: the published 2026-10-11 Ironclad column. Ironclad, champion plr_it550, same driver, 8000/8000")
    L.append("       sims, same seeds and n, 32 procs, engine 0cab615a3 (2026-09-13): pre-#830 search (no BC prior;")
    L.append("       the 2026-10-11 write-up gives the old knob values as removed_counts_forms=0,")
    L.append("       endturn_widening_c=5.787, draw_widening_c=2.0) and no rl2 playout net (PR #816, 2026-10-18,")
    L.append("       is one of the 690 commits from 0cab615a3 to 0181fa552).")
    L.append("")
    L.append("PER LEVEL (Ironclad; NEW = a10_it700, 8000/8000 sims, #830 default; OLD = plr_it550, 8000/8000 sims,")
    L.append("pre-#830 search; rates in %, Wilson 95% CI with z = 1.96 as in 2026-10-11; diff = NEW - OLD in pp,")
    L.append("paired by seed, Newcombe 1998 method 10 paired 95% CI; b = NEW won & OLD lost, c = OLD won & NEW lost;")
    L.append("p = exact two-sided McNemar, uncorrected for the 11 tests)")
    L.append("")
    widths = (4, 4, 8, 19, 13, 22, 7, 8)
    just = ("l", "r", "r", "l", "l", "l", "r", "r")

    def line(cells: tuple[str, ...]) -> str:
        return "  ".join(c.ljust(w) if j == "l" else c.rjust(w) for c, w, j in zip(cells, widths, just, strict=True)).rstrip()

    hdr = line(("Asc", "n", "NEW wins", "NEW % [Wilson 95%]", "OLD wins  %", "diff pp [paired 95%]", "b/c", "p"))
    L.append(hdr)
    L.append("-" * len(hdr))
    for a in LEVELS:
        x, y, p = nl[a], ol[a], pr[a]
        L.append(
            line(
                (
                    f"A{a}",
                    str(x["n"]),
                    f"{x['wins']}/{x['n']}",
                    f"{pct(x['rate'])} [{pct(x['w196'][0])}-{pct(x['w196'][1])}]",
                    f"{y['wins']:>3}/{y['n']}  {pct(y['rate']):>4}",
                    f"{spp(p['diff'])} [{spp(p['nc_196'][0])}, {spp(p['nc_196'][1])}]",
                    f"{p['b']}/{p['c']}",
                    fmt_p(p["p"]),
                )
            )
        )
    L.append("-" * len(hdr))
    L.append(
        f"mean A0-A10 (unweighted mean of the 11 level rates, as in 2026-10-11): NEW {pct(m['mean_new'])} "
        f"({100 * m['mean_new']:.2f}), OLD {pct(m['mean_old'])} ({100 * m['mean_old']:.2f})."
    )
    L.append(
        f"Diff {spp(m['mean_diff'])}pp ({100 * m['mean_diff']:+.2f}), seed-clustered percentile bootstrap 95% CI "
        f"[{spp(m['boot']['ci95'][0])}, {spp(m['boot']['ci95'][1])}]"
    )
    L.append(
        f"  (500 seeds resampled with replacement, each carrying all its levels; 10,000 resamples; numpy "
        f"default_rng(20261026); SE {100 * m['boot']['se']:.2f}pp)."
    )
    mc_lo = min(b["ci95"][0] for b in m["boot_mc_seeds_1_5"])
    mc_hi = max(b["ci95"][1] for b in m["boot_mc_seeds_1_5"])
    L.append(
        f"  Agrees with the stratified bootstrap [{spp(m['boot_strat']['ci95'][0])}, {spp(m['boot_strat']['ci95'][1])}], "
        f"the blind's analytic seed-clustered normal CI [{spp(m['cluster']['ci_normal'][0])}, {spp(m['cluster']['ci_normal'][1])}],"
    )
    L.append(
        f"  and five other bootstrap RNG seeds (endpoints within [{spp(mc_lo)}, {spp(mc_hi)}]). Seeds 81,000,000-199 repeat"
    )
    L.append(
        "  across all 11 levels and 81,000,200-499 across A9-A10, so the level diffs are not independent (variance"
    )
    L.append(f"  ratio {m['cluster']['design_effect']:.2f} vs treating levels as independent); no p-value is given for the pooled diff.")
    L.append(
        f"Run-pooled over all {m['run_pooled']['n']:,} pairs (A9-A10 weigh 2.5x): b/c = {m['run_pooled']['b']}/{m['run_pooled']['c']}, "
        f"diff {spp(m['run_pooled']['diff'])}pp (descriptive)."
    )
    L.append("")
    L.append("ACT-1 SPLIT (Ironclad, pooled over all 11 cells by run, the 2026-10-11 definition; Wilson z = 1.96)")
    for arm, lab in (("new", "NEW a10_it700, #830 default, 8000/8000"), ("old", "OLD plr_it550, pre-#830, 8000/8000")):
        s = m["act1_" + arm]
        L.append(
            f"  {lab}: Overgrowth {s['overgrowth']['wins']}/{s['overgrowth']['n']} = {pct(s['overgrowth']['rate'])} "
            f"[{pct(s['overgrowth']['w196'][0])}-{pct(s['overgrowth']['w196'][1])}], Underdocks "
            f"{s['underdocks']['wins']}/{s['underdocks']['n']} = {pct(s['underdocks']['rate'])} "
            f"[{pct(s['underdocks']['w196'][0])}-{pct(s['underdocks']['w196'][1])}]"
        )
    sn, so = m["act1_std_new"], m["act1_std_old"]
    L.append(
        "  Act 1 is a function of the seed (one act per seed at every level, in both arms), so this is a by-seed split,"
    )
    L.append(
        "  not a paired comparison, and the pooled split mixes in the level composition (seeds 81,000,200-499 play only"
    )
    mix = m["act1_mix"]
    L.append(
        f"  A9-A10 and are {mix['200_499']['overgrowth']}/300 Overgrowth vs {mix['0_199']['overgrowth']}/200 below). "
        "Weighting the 11 levels equally (the blind's view, rechecked):"
    )
    L.append(
        f"  NEW Overgrowth {pct(sn['overgrowth'])} vs Underdocks {pct(sn['underdocks'])}; OLD {pct(so['overgrowth'])} vs "
        f"{pct(so['underdocks'])}. The pooled gap goes away once levels weigh equally."
    )
    L.append("")
    L.append("INTEGRITY (all three recomputes agree)")
    L.append(
        f"  {integ['rows_new']:,} new rows (9 x 200 + 2 x 500; the task text's '3,400 runs' is a slip), {integ['error_rows_new']} error rows, "
        f"{integ['timeout_rows_new']} timeout rows, nothing filtered;"
    )
    L.append("  outcomes are only won/lost. Every cell holds exactly seeds 81,000,000..+n-1 once. character, ascension,")
    L.append("  final.ascension, policy, wheel src_hash 4e5bb730d28e2976, pgo=true, pgo_instrumented=false and won-vs-outcome")
    L.append("  are right on every row; one .so path, one py_src (774d91045cdb5433), one profdata sha. No seed in")
    L.append("  96,500,000-96,502,999. Each cell .log (header, outcomes, win rate, wall mean/median/max) and each progress.log")
    L.append("  DONE line (rc 0, wins/n, wall, batch wall) agree with the rows; G1b ownership audit PASS in all 11 logs; the")
    L.append(f"  cells ran in the pre-registered order {integ['order_run']}; ALL DONE; the driver start line pins head 0181fa552,")
    L.append(f"  the .so sha and the champion ckpt. Old arm: 2,800 Ironclad rows in {OLD_CSV.name}, seeds complete, 0 errors,")
    L.append("  0 timeouts; recomputed, it reproduces the published column exactly (all 11 rates and Wilson CIs, mean 74.9,")
    L.append("  act-1 split 63.2/65.0 on n = 1,297/1,503)"
             + (" and summary.csv to 4 decimals" if m["summary_csv"] is not None else "") + ". act1 agrees between the arms on")
    L.append(f"  {m['act1_agree']:,}/{m['run_pooled']['n']:,} pairs; no seed dropped from any pairing.")
    L.append("")
    L.append("COST (new sweep; wall_s is per run, one single-threaded process per run, 30 at a time)")
    L.append(f"  {'Asc':<4} {'load@START 1/5/15':<20} {'mean s':>7} {'median s':>9} {'max s':>7} {'batch wall s':>13} {'CPU-h':>6}")
    for a in ORDER:
        c, lg = m["cost_new"][a], m["logs"][a]
        ld = m["prog"]["start"][a]["load"]
        L.append(
            f"  A{a:<3} {ld[0]:>5.2f}/{ld[1]:>5.2f}/{ld[2]:>5.2f}    {c['mean']:>7.1f} {c['median']:>9.1f} {c['max']:>7.1f} "
            f"{lg['batch_wall_s']:>13} {c['sum_h']:>6.1f}"
        )
    tot_h = sum(m["cost_new"][a]["sum_h"] for a in LEVELS)
    tot_b = sum(m["logs"][a]["batch_wall_s"] for a in LEVELS) / 3600
    n_runs = sum(m["cost_new"][a]["n"] for a in LEVELS)
    L.append(
        f"  total (rows above in run order): {tot_h:.1f} CPU-h (sum of wall_s), {tot_b:.2f} h of batch wall, "
        f"mean {tot_h * 3600 / n_runs:.1f} s per run."
    )
    if m["cost_old"] is not None and m["old_prog"] is not None:
        old_h = sum(m["cost_old"][a]["sum_h"] for a in LEVELS)
        old_b = sum(m["old_prog"]["done"][a]["batch_wall_s"] for a in LEVELS) / 3600
        L.append(
            f"  The old Ironclad cells took {old_h:.1f} CPU-h and {old_b:.2f} h of batch wall at 32 procs, mean "
            f"{old_h * 3600 / n_runs:.1f} s per run; that is context only,"
        )
        L.append("  not a baseline: another engine, champion and search, and another day's box load.")
    else:
        L.append(f"  The old Ironclad cells' wall times are not in {OLD_CSV.name} (ascension, seed, act1 and won only).")
    L.append("")
    L.append("READING")
    L.append(
        "  - New Ironclad release column (a10_it700, ppo.probe net_greedy, 8000/8000 sims in one tree, shipped #830 default,"
    )
    L.append(
        f"    engine 0181fa552, seeds 81,000,000+): A0 {pct(nl[0]['rate'])}, A1 {pct(nl[1]['rate'])}, A2 {pct(nl[2]['rate'])}, "
        f"A3 {pct(nl[3]['rate'])}, A4 {pct(nl[4]['rate'])}, A5 {pct(nl[5]['rate'])}, A6 {pct(nl[6]['rate'])},"
    )
    L.append(
        f"    A7 {pct(nl[7]['rate'])}, A8 {pct(nl[8]['rate'])} (n = 200 each), A9 {pct(nl[9]['rate'])}, A10 {pct(nl[10]['rate'])} "
        f"(n = 500 each); mean A0-A10 {pct(m['mean_new'])}."
    )
    L.append(
        "  - Against the published column (plr_it550, same protocol and seeds, engine 0cab615a3) the new arm is ahead at"
    )
    lo_lvl = min(LEVELS, key=lambda a: pr[a]["diff"])
    hi_lvl = max(LEVELS, key=lambda a: pr[a]["diff"])
    incl0 = [f"A{a}" for a in LEVELS if pr[a]["nc_196"][0] <= 0 <= pr[a]["nc_196"][1]]
    L.append(
        f"    every level on the point estimate, by {spp(pr[lo_lvl]['diff'])}pp (A{lo_lvl}) to {spp(pr[hi_lvl]['diff'])}pp "
        f"(A{hi_lvl}); the paired CI excludes zero at"
    )
    L.append(
        f"    {len(LEVELS) - len(incl0)} of 11 levels (not at {', '.join(incl0)}). Mean A0-A10: {spp(m['mean_diff'])}pp "
        f"[{spp(m['boot']['ci95'][0])}, {spp(m['boot']['ci95'][1])}]."
    )
    L.append(
        "    This comparison confounds three changes made at once: the champion (plr_it550 -> a10_it700), the search"
    )
    L.append(
        "    default (GH #830: knobs + BC prior), and the engine (690 commits from 0cab615a3 to 0181fa552, including the"
    )
    L.append(
        "    rl2 playout net, PR #816). It is descriptive and attributes nothing: it does not say how much of the gain"
    )
    L.append("    any one change brought, and the per-level p values carry no multiplicity correction.")
    lo_u, hi_u = newcombe_unpaired(nl[10]["wins"], nl[10]["n"], STD_EVAL_A10["wins"], STD_EVAL_A10["n"], Z196)
    std_rate = STD_EVAL_A10["wins"] / STD_EVAL_A10["n"]
    sw = wilson(STD_EVAL_A10["wins"], STD_EVAL_A10["n"], Z196)
    L.append(
        f"  - The new A10 ({nl[10]['wins']}/{nl[10]['n']} = {pct(nl[10]['rate'])}% [{pct(nl[10]['w196'][0])}-{pct(nl[10]['w196'][1])}], "
        f"Ironclad a10_it700, 8000/8000 sims, #830 default,"
    )
    L.append(
        f"    seeds 81,000,000-81,000,499) sits beside the playbook §2.6 standard eval's {100 * std_rate:.2f}% "
        f"({STD_EVAL_A10['wins']}/{STD_EVAL_A10['n']} [{pct(sw[0])}-{pct(sw[1])}], Ironclad"
    )
    L.append(
        "    a10_it700 net_greedy, 4000/16000 sims, the same shipped default, seeds 91,000,000-91,000,399). That is another"
    )
    L.append(
        f"    panel and another budget, so the two are unpaired: {100 * (nl[10]['rate'] - std_rate):+.2f}pp, Newcombe hybrid-score 95% CI"
    )
    L.append(
        f"    [{spp(lo_u)}, {spp(hi_u)}]. They are consistent, and this says nothing about the effect of budget (8000/8000 vs"
    )
    L.append("    4000/16000) or of the panel.")
    av = m["a9_vs_a10est"]
    L.append(
        "  - Side note, post-hoc (found by the blind, rechecked here, not pre-registered): read off this sweep's A10 runs,"
    )
    L.append(
        f"    the A9 estimate (won or reached the second act-3 boss) is {pct(av['a10_estimate'])}% vs {pct(av['a9_direct'])}% "
        f"for A9 played directly on the same 500 seeds"
    )
    L.append(
        f"    (Ironclad a10_it700, 8000/8000, #830 default): the estimate is {100 * av['diff']:.1f}pp lower, b/c "
        f"{av['b']}/{av['c']}, exact McNemar p {av['p']:.4f},"
    )
    L.append(
        f"    paired 95% CI of direct - estimate [{spp(av['nc_196'][0])}, {spp(av['nc_196'][1])}]. The §2.6 write-up already "
        "says the A9 estimate ran ~5pp"
    )
    L.append(
        "    below a true A9 sweep on Ironclad. So the §2.6 A9 estimate (52.25%, 4000/16000, other panel) should not be set"
    )
    L.append(f"    beside this sweep's directly played A9 ({pct(nl[9]['rate'])}%).")
    L.append(
        "  - Wall times were measured on a shared box: other jobs ran during the sweep (a CF Pages wasm build, a"
    )
    L.append(
        "    test-public-shape wheel build and pytest run, replays from a reproduction workflow, mostly during A10, A9 and"
    )
    L.append(
        "    A0; see the load averages). They are not clean cost numbers, and the win rates do not depend on them (the"
    )
    L.append(
        f"    search has no wall-clock budget; the longest run took {max(m['cost_new'][a]['max'] for a in LEVELS):.1f} s "
        "against the 14,400 s timeout)."
    )
    L.append(
        "  - For whoever updates the release pages: the 2026-10-11 write-up's 'measured before' note says how to"
    )
    L.append(
        "    reproduce the old column on a newer wheel (old knob values, STS2_RUNAI_PRIOR_NET=off) but does not mention"
    )
    L.append("    the rl2 playout net, which 0cab615a3 did not have.")
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------- main


def main() -> int:
    m = recompute()
    P = json.loads(PRIMARY_JSON.read_text(encoding="utf-8"))
    B = json.loads(BLIND_JSON.read_text(encoding="utf-8"))
    md = PRIMARY_MD.read_text(encoding="utf-8")
    items = run_comparisons(m, P, md, B, with_independent=True)
    current = summarize(items)

    prefix = None
    fixes = []
    if SNAP.is_dir():
        Ps = json.loads((SNAP / "primary.json").read_text(encoding="utf-8"))
        Bs = json.loads((SNAP / "blind.json").read_text(encoding="utf-8"))
        mds = (SNAP / "primary.md").read_text(encoding="utf-8")
        prefix = summarize(run_comparisons(m, Ps, mds, Bs, with_independent=False))
        blind_diff = leaf_diff(Bs, B)
        md_before, md_after = mds.splitlines(), md.splitlines()
        md_changed = [
            {"line": i + 1, "before": x, "after": y}
            for i, (x, y) in enumerate(zip(md_before, md_after, strict=False))
            if x != y
        ]
        for key, f in FIXES.items():
            rec = dict(f, id=key)
            if key == "F1_blind_r6":
                rec["sha256_before_after"] = {
                    "recompute.py": [sha256(SNAP / "recompute.py"), sha256(ANALYSIS / "blind" / "recompute.py")],
                    "blind.json": [sha256(SNAP / "blind.json"), sha256(BLIND_JSON)],
                }
                rec["blind_json_leaves_changed"] = blind_diff
                rec["only_values_below_1e-3_changed"] = all(
                    isinstance(d["before"], float) and isinstance(d["after"], float) and abs(d["after"]) < 1e-3
                    for d in blind_diff
                )
            else:
                rec["sha256_before_after"] = {
                    "analyze.py": [sha256(SNAP / "analyze.py"), sha256(ANALYSIS / "scripts" / "analyze.py")],
                    "primary.md": [sha256(SNAP / "primary.md"), sha256(PRIMARY_MD)],
                    "primary.json": [sha256(SNAP / "primary.json"), sha256(PRIMARY_JSON)],
                }
                rec["primary_json_byte_identical_after_rerun"] = sha256(SNAP / "primary.json") == sha256(PRIMARY_JSON)
                rec["primary_md_lines_changed"] = md_changed
            fixes.append(rec)

    verdict = {
        "values_compared": current["values_compared"],
        "mismatches": current["mismatches"],
        "unexplained": current["unexplained"],
        "prefix_mismatches": prefix["mismatches"] if prefix else None,
        "prefix_unexplained": prefix["unexplained"] if prefix else None,
        "statement": (
            f"{current['mismatches']} mismatches in {current['values_compared']} compared values on the current files"
            + (
                f"; before the two reporting fixes there were {prefix['mismatches']} "
                f"({prefix['unexplained']} unexplained), all from F1 (blind.json p-value rounding) or "
                "F2 (primary.md '3,400')"
                if prefix
                else ""
            )
        ),
    }
    out = {
        "what": "third recompute of the 2026-10-26 Ironclad release sweep from the raw rows; every value it shares with "
        "primary.json, primary.md, blind.json, the logs and the published 2026-10-11 column, compared with explicit tolerances",
        "qualifier_new": "Ironclad, champion a10_it700, ppo.probe net_greedy, act1 roll, 8000/8000 sims in one tree, shipped "
        "GH #830 default (knobs + BC prior in elite/boss + rl2 playout net), engine 0181fa552, PGO src_hash 4e5bb730d28e2976, "
        "seeds 81,000,000+ (200 at A0-A8, 500 at A9-A10)",
        "qualifier_old": "Ironclad, champion plr_it550, same driver, budget and seeds, engine 0cab615a3, pre-#830 search, "
        "no rl2 playout net (2026-10-11 release sweep)",
        "tolerances": {**TOLS, "printed_1dp": "string equality of the value formatted to 1 decimal in percent"},
        "tolerance_notes": {
            "full": "primary.json floats vs this recompute: same formulas, only summation order may differ",
            "r6": "blind.json writes floats rounded to 6 decimals",
            "p_rel": "relative; McNemar p down to 1.7e-13 must agree to ~3 significant figures",
            "method": "the pooled-diff CI by percentile bootstrap vs the blind's analytic seed-clustered normal CI",
            "mc": "the pre-registered bootstrap RNG seed vs five other RNG seeds",
            "z": "primary uses z = 1.959964 (Phi^-1(0.975)) for Wilson/Newcombe and also reports z = 1.96; the blind uses "
            "z = 1.96; each is compared with this recompute at its own z; 'z_choice' checks show the choice is invisible at "
            "1 decimal",
        },
        "verdict": verdict,
        "fixes": fixes,
        "current": current,
        "prefix_snapshot": prefix,
        "values": items,
        "reconciled": {
            "per_level": {
                f"A{a}": {
                    "new_wins": m["new"][a]["wins"],
                    "n": m["new"][a]["n"],
                    "new_rate": m["new"][a]["rate"],
                    "new_wilson95_z196": list(m["new"][a]["w196"]),
                    "old_wins": m["old"][a]["wins"],
                    "old_rate": m["old"][a]["rate"],
                    "old_wilson95_z196": list(m["old"][a]["w196"]),
                    "pairs": m["paired"][a]["pairs"],
                    "diff": m["paired"][a]["diff"],
                    "newcombe10_95_z196": list(m["paired"][a]["nc_196"]),
                    "b": m["paired"][a]["b"],
                    "c": m["paired"][a]["c"],
                    "mcnemar_exact_p": m["paired"][a]["p"],
                }
                for a in LEVELS
            },
            "mean_new": m["mean_new"],
            "mean_old": m["mean_old"],
            "mean_diff": m["mean_diff"],
            "bootstrap_ci95": m["boot"]["ci95"],
            "bootstrap_se": m["boot"]["se"],
            "bootstrap_stratified_ci95": m["boot_strat"]["ci95"],
            "bootstrap_other_rng_seeds_ci95": [b["ci95"] for b in m["boot_mc_seeds_1_5"]],
            "analytic_cluster": m["cluster"],
            "run_pooled": m["run_pooled"],
            "act1_new": {k: {**v, "w196": list(v["w196"]), "wex": list(v["wex"])} for k, v in m["act1_new"].items()},
            "act1_old": {k: {**v, "w196": list(v["w196"]), "wex": list(v["wex"])} for k, v in m["act1_old"].items()},
            "act1_level_standardized_new": m["act1_std_new"],
            "act1_level_standardized_old": m["act1_std_old"],
            "a9_direct_vs_a10_estimate_posthoc": {**m["a9_vs_a10est"], "nc_196": list(m["a9_vs_a10est"]["nc_196"])},
            "std_eval_a10_context": {
                "std_eval": STD_EVAL_A10,
                "this_sweep": {"wins": m["new"][10]["wins"], "n": m["new"][10]["n"]},
                "unpaired_diff": m["new"][10]["rate"] - STD_EVAL_A10["wins"] / STD_EVAL_A10["n"],
                "newcombe_hybrid_95_z196": list(
                    newcombe_unpaired(m["new"][10]["wins"], m["new"][10]["n"], STD_EVAL_A10["wins"], STD_EVAL_A10["n"], Z196)
                ),
            },
            "integrity": {k: v for k, v in m["integrity"].items() if k != "per_level"},
        },
        "inputs": {
            "cells_jsonl_sha256": {f"A{a}": sha256(CELLS / f"ironclad_A{a}.jsonl") for a in LEVELS},
            "old_runs_csv_sha256": sha256(OLD_CSV),
            "progress_log_sha256": sha256(PROGRESS),
            "primary_json_sha256": sha256(PRIMARY_JSON),
            "primary_md_sha256": sha256(PRIMARY_MD),
            "blind_json_sha256": sha256(BLIND_JSON),
        },
    }
    OUT_JSON.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    OUT_TXT.write_text(render_final(m, verdict), encoding="utf-8")
    print(json.dumps(verdict, indent=1))
    print(json.dumps(current["by_source"], indent=1))
    for x in current["mismatch_list"]:
        print("CURRENT MISMATCH", x)
    if prefix:
        for x in prefix["mismatch_list"]:
            print("PREFIX MISMATCH", x["source"], x["id"], x["mine"], x["theirs"], x["explanation"])
    print(f"wrote {OUT_JSON} and {OUT_TXT}")
    return 0 if current["mismatches"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
