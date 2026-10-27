"""Primary pre-registered analysis of the 2026-10-26 Ironclad release win-rate sweep.

Follows PREREG.md "Analysis (fixed now)" 1-4, plus descriptive run-depth context:

1. Integrity: every cell holds exactly seeds 81,000,000 .. +n-1 once; character, ascension,
   policy and wheel (src_hash 4e5bb730d28e2976, pgo) are right on every row; error and timeout
   rows are counted and filtered FIRST (a timeout is not a loss); each cell's .log summary line
   ("win rate: x/n", "outcomes: {...}") agrees with the rows; progress.log DONE lines agree too.
2. Per level: wins/n, rate, Wilson 95% CI; unweighted mean of the 11 per-level rates; act-1
   split pooled over all 11 cells (wins/n per `act1` value). Both definitions are the ones in
   docs/experiments/2026-10-11-release-winrates.md and logs/2026-10-11-release-winrates/
   summarize.py. The OLD Ironclad column is recomputed from that sweep's runs.csv with the same
   code and checked against the published column.
3. Paired by seed vs the old Ironclad cells, descriptive only: pair count, rates, diff,
   discordant b (new won, old lost) / c (old won, new lost), exact two-sided McNemar p
   (binomial), Newcombe (1998) method 10 paired 95% CI. act1 agreement per seed. Pooled mean
   diff over A0-A10 with a seed-clustered bootstrap CI (resample seeds 81,000,000-81,000,499,
   each seed carries all its levels; 10,000 resamples; fixed RNG seed).
4. Cost per cell: wall_s mean/median/max, batch wall (cell .log), CPU-hours = sum(wall_s)/3600,
   load average at START (progress.log); the old sweep's Ironclad wall_s per cell for context.
5. Descriptive: mean total_floor and the share of runs that died in act 1/2/3 (final.act_index).

Usage: python -B analyze.py [SWEEP_DIR] [OLD]
  SWEEP_DIR defaults to the folder two levels above this script.
  OLD is the 2026-10-11 sweep's rows, either
    * a CSV file with columns ascension, seed, act1, won (the published old_ironclad_runs.csv,
      the default: SWEEP_DIR/old_ironclad_runs.csv); the old rows' wall_s and total_floor and
      the old progress.log are then absent, and every field derived from them is null; or
    * the private sweep folder logs/2026-10-11-release-winrates (runs.csv with all eight
      columns, and progress.log), as the run used it.
Writes SWEEP_DIR/analysis/primary.json and SWEEP_DIR/analysis/primary.md.

Changes since the run (2026-10-26): OLD was a hard-coded private path and is an argument now,
and the old rows may lack wall_s, total_floor and the old progress.log (those fields are then
null). On the run's inputs the script still writes primary.json and primary.md byte for byte.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

HERE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
OLD = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "old_ironclad_runs.csv"
# A folder is the private sweep folder (runs.csv + progress.log); a file is the published CSV.
OLD_CSV = OLD / "runs.csv" if OLD.is_dir() else OLD
OLD_PROGRESS = OLD / "progress.log" if OLD.is_dir() else None
CELLS = HERE / "cells"
PROGRESS = HERE / "progress.log"
OUT_JSON = HERE / "analysis" / "primary.json"
OUT_MD = HERE / "analysis" / "primary.md"

LEVELS = list(range(11))
N_PER_LEVEL = {a: (500 if a >= 9 else 200) for a in LEVELS}
SEED0 = 81_000_000
CLUSTER_SEEDS = list(range(SEED0, SEED0 + 500))
ORDER = [10, 9, 0, 6, 3, 8, 1, 5, 2, 4, 7]
BAD = ("error", "timeout")
Z = 1.959963984540054
BOOT_N = 10_000
BOOT_RNG_SEED = 20261026

CHARACTER = "Ironclad"
POLICY = "net_greedy"
SRC_HASH = "4e5bb730d28e2976"
HEAD = "0181fa552a78053f260f0796e1cb086d770c7cef"
SO_SHA = "823fa393eab74db57d0cffd6f71c4a5c873a74d345a354adfca8458bf1333ba2"
CKPT = "ppo_fullrun_ironclad_a10_it700.pt"
CKPT_SHA = "91c83c1aafe137b22ab0617bea2a8fcb00e6b8777a67fe718b4d477a156a44fc"

NEW_QUAL = (
    "Ironclad, champion a10_it700 (ppo_fullrun_ironclad_a10_it700.pt, sha 91c83c1a), "
    "ppo.probe --policy net_greedy --act1 roll, 8000 sims per decision with boss fights "
    "included (one tree), shipped Ironclad search default (knobs removed_counts_forms=1, "
    "endturn_widening_c=23.148, draw_widening_c=4.0 in every fight + built-in BC move prior "
    "mix 0.5, T 1 in elite and boss fights; rl2 playout net), engine 0181fa552, PGO wheel "
    "src_hash 4e5bb730d28e2976, seeds 81,000,000+ (200 at A0-A8, 500 at A9-A10), 30 procs"
)
OLD_QUAL = (
    "Ironclad, champion plr_it550, ppo.probe --policy net_greedy --act1 roll, 8000 sims per "
    "decision with boss fights included (one tree), the search as it was at engine 0cab615a3 "
    "(no GH #830 knobs, no BC move prior, no rl2 playout net), seeds 81,000,000+ "
    "(200 at A0-A8, 500 at A9-A10), 32 procs (2026-10-11 release sweep)"
)

# The Ironclad column of docs/experiments/2026-10-11-release-winrates.md, as published.
PUBLISHED_OLD = {
    "rate_pct": {
        0: 89.5, 1: 95.0, 2: 92.5, 3: 90.0, 4: 86.5, 5: 85.0,
        6: 86.0, 7: 79.0, 8: 70.0, 9: 39.0, 10: 11.0,
    },
    "ci_pct": {
        0: (84.5, 93.0), 1: (91.0, 97.3), 2: (88.0, 95.4), 3: (85.1, 93.4),
        4: (81.1, 90.6), 5: (79.4, 89.3), 6: (80.5, 90.1), 7: (72.8, 84.1),
        8: (63.3, 75.9), 9: (34.8, 43.3), 10: (8.5, 14.0),
    },
    "mean_pct": 74.9,
    "act1": {"overgrowth": {"n": 1297, "rate_pct": 63.2}, "underdocks": {"n": 1503, "rate_pct": 65.0}},
}


# ---------------------------------------------------------------- statistics


def wilson(k: int, n: int) -> tuple[float, float]:
    """Wilson score 95% interval (no continuity correction)."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    den = 1 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / den
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / den
    return (centre - half, centre + half)


def wilson_summarize_py(k: int, n: int) -> tuple[float, float]:
    """The interval exactly as logs/2026-10-11-release-winrates/summarize.py computed it (z=1.96)."""
    z = 1.96
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar: 2 * P(X <= min(b, c)), X ~ Binom(b + c, 1/2), capped at 1."""
    m = b + c
    if m == 0:
        return 1.0
    tail = sum(math.comb(m, i) for i in range(min(b, c) + 1)) / 2**m
    return min(1.0, 2 * tail)


def newcombe_paired(a: int, b: int, c: int, d: int) -> tuple[float, float]:
    """95% CI of p1 - p2 for paired binary data, Newcombe (1998) method 10 (hybrid score, no CC).

    a = both win, b = arm1 only, c = arm2 only, d = both lose; p1 = (a+b)/n, p2 = (a+c)/n.
    phi is the plain phi coefficient of the 2x2 table, taken as 0 when a margin is empty.
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


# ---------------------------------------------------------------- loading


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_new() -> dict[int, list[dict]]:
    cells = {}
    for a in LEVELS:
        with open(CELLS / f"ironclad_A{a}.jsonl", encoding="utf-8") as f:
            cells[a] = [json.loads(line) for line in f if line.strip()]
    return cells


def load_old() -> dict[int, list[dict]]:
    cells: dict[int, list[dict]] = {a: [] for a in LEVELS}
    with open(OLD_CSV, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            # The published CSV holds Ironclad rows only and no character/outcome/total_floor/wall_s
            # columns; every old Ironclad row's outcome is "won" or "lost", so it follows from won.
            if r.get("character", CHARACTER) != CHARACTER:
                continue
            won = r["won"] == "1"
            cells[int(r["ascension"])].append(
                {
                    "seed": int(r["seed"]),
                    "ascension": int(r["ascension"]),
                    "character": r.get("character", CHARACTER),
                    "act1": r["act1"],
                    "outcome": r.get("outcome", "won" if won else "lost"),
                    "won": won,
                    "total_floor": int(r["total_floor"]) if r.get("total_floor") else None,
                    "wall_s": float(r["wall_s"]) if r.get("wall_s") else None,
                }
            )
    return cells


def is_bad(r: dict) -> bool:
    return r.get("outcome") in BAD or "error" in r


def seed_check(rows: list[dict], n: int) -> dict:
    seeds = [int(r["seed"]) for r in rows]
    expected = set(range(SEED0, SEED0 + n))
    dup = len(seeds) - len(set(seeds))
    return {
        "rows": len(rows),
        "expected_rows": n,
        "duplicates": dup,
        "missing": sorted(expected - set(seeds))[:20],
        "extra": sorted(set(seeds) - expected)[:20],
        "ok": len(rows) == n and dup == 0 and set(seeds) == expected,
    }


# ---------------------------------------------------------------- logs


def parse_cell_log(a: int) -> dict:
    text = (CELLS / f"ironclad_A{a}.log").read_text(encoding="utf-8")
    out: dict = {}
    m = re.search(r"=== ppo\.probe --policy (\S+) n=(\d+) asc=(\d+) char=(\S+) ===", text)
    if m:
        out["header"] = {
            "policy": m.group(1),
            "n": int(m.group(2)),
            "asc": int(m.group(3)),
            "char": m.group(4),
        }
    m = re.search(r"^outcomes: (\{.*\})$", text, re.M)
    if m:
        out["outcomes"] = ast.literal_eval(m.group(1))
    m = re.search(r"^win rate: (\d+)/(\d+) = ([\d.]+)$", text, re.M)
    if m:
        out["wins"], out["n"] = int(m.group(1)), int(m.group(2))
    m = re.search(
        r"^wall/run: mean ([\d.]+)s\s+median ([\d.]+)s\s+max ([\d.]+)s; batch wall (\d+)s "
        r"\S+ ([\d.]+) runs/min$",
        text,
        re.M,
    )
    if m:
        out["wall_mean_s"], out["wall_median_s"], out["wall_max_s"] = (
            float(m.group(1)),
            float(m.group(2)),
            float(m.group(3)),
        )
        out["batch_wall_s"], out["runs_per_min"] = int(m.group(4)), float(m.group(5))
    out["crash_banner"] = "SEEDS CRASHED" in text
    out["g1b_pass"] = "G1b ownership audit: PASS" in text
    return out


def parse_progress() -> dict:
    out: dict = {"start": {}, "done": {}, "driver": None, "all_done": False, "order": []}
    with open(PROGRESS, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            m = re.match(r"(\S+) driver start pid \d+ head=(\S+) so=(\S+) ckpt=(\S+)", line)
            if m:
                out["driver"] = {"ts": m.group(1), "head": m.group(2), "so": m.group(3), "ckpt": m.group(4)}
            m = re.match(r"(\S+) START Ironclad A(\d+) n=(\d+) load ([\d.]+) ([\d.]+) ([\d.]+)", line)
            if m:
                a = int(m.group(2))
                out["order"].append(a)
                out["start"][a] = {
                    "ts": m.group(1),
                    "n": int(m.group(3)),
                    "load_1_5_15": [float(m.group(4)), float(m.group(5)), float(m.group(6))],
                }
            m = re.match(
                r"(\S+) DONE\s+Ironclad A(\d+) rc=(\d+) win rate: (\d+)/(\d+) .*batch wall (\d+)s",
                line,
            )
            if m:
                out["done"][int(m.group(2))] = {
                    "ts": m.group(1),
                    "rc": int(m.group(3)),
                    "wins": int(m.group(4)),
                    "n": int(m.group(5)),
                    "batch_wall_s": int(m.group(6)),
                }
            if "ALL DONE" in line:
                out["all_done"] = True
    return out


def parse_old_progress() -> dict[int, dict]:
    out: dict[int, dict] = {}
    if OLD_PROGRESS is None:  # the published old rows come without the old sweep's progress.log
        return out
    with open(OLD_PROGRESS, encoding="utf-8") as f:
        for line in f:
            m = re.match(
                r"(\S+) DONE\s+Ironclad A(\d+) win rate: (\d+)/(\d+) .*batch wall (\d+)s", line
            )
            if m:
                out[int(m.group(2))] = {
                    "wins": int(m.group(3)),
                    "n": int(m.group(4)),
                    "batch_wall_s": int(m.group(5)),
                }
    return out


# ---------------------------------------------------------------- per-level summaries (same code for both sweeps)


def level_table(cells: dict[int, list[dict]]) -> dict:
    """Wins/n, rate, Wilson CI per level, unweighted mean, act-1 split pooled over all cells.

    Definitions match the 2026-10-11 write-up: errors (and here timeouts) are dropped, the mean
    is the unweighted mean of the 11 per-level rates, the act-1 split pools every scored run of
    every cell by its `act1` value.
    """
    per = {}
    for a in LEVELS:
        scored = [r for r in cells[a] if not is_bad(r)]
        k, n = sum(bool(r["won"]) for r in scored), len(scored)
        lo, hi = wilson(k, n)
        lo6, hi6 = wilson_summarize_py(k, n)
        per[a] = {
            "wins": k,
            "n": n,
            "rate": k / n,
            "wilson95": [lo, hi],
            "wilson95_summarize_py": [lo6, hi6],
        }
    mean = statistics.fmean(per[a]["rate"] for a in LEVELS)
    act1: dict[str, dict] = {}
    for a in LEVELS:
        for r in cells[a]:
            if is_bad(r):
                continue
            d = act1.setdefault(r["act1"], {"wins": 0, "n": 0})
            d["n"] += 1
            d["wins"] += bool(r["won"])
    for d in act1.values():
        d["rate"] = d["wins"] / d["n"]
        d["wilson95"] = list(wilson(d["wins"], d["n"]))
    return {"per_level": per, "mean_A0_A10": mean, "act1_split": dict(sorted(act1.items()))}


def check_published_old(tab: dict) -> dict:
    mism = []
    for a in LEVELS:
        p = tab["per_level"][a]
        got = round(100 * p["rate"], 1)
        if got != PUBLISHED_OLD["rate_pct"][a]:
            mism.append(f"A{a} rate {got} != {PUBLISHED_OLD['rate_pct'][a]}")
        lo, hi = (round(100 * x, 1) for x in p["wilson95_summarize_py"])
        if (lo, hi) != PUBLISHED_OLD["ci_pct"][a]:
            mism.append(f"A{a} CI {lo}-{hi} != {PUBLISHED_OLD['ci_pct'][a]}")
    got_mean = round(100 * tab["mean_A0_A10"], 1)
    if got_mean != PUBLISHED_OLD["mean_pct"]:
        mism.append(f"mean {got_mean} != {PUBLISHED_OLD['mean_pct']}")
    for act, ref in PUBLISHED_OLD["act1"].items():
        d = tab["act1_split"].get(act, {"n": 0, "rate": math.nan})
        if d["n"] != ref["n"] or round(100 * d["rate"], 1) != ref["rate_pct"]:
            mism.append(f"act1 {act}: n {d['n']} rate {100 * d['rate']:.1f} != {ref}")
    return {"matches_published": not mism, "mismatches": mism}


# ---------------------------------------------------------------- run depth (descriptive)


def death_act_from_floor(tf: int | None) -> int | None:
    """Act of a lost run from total_floor alone (used only for the OLD rows, which lack act_index).

    Bands from the new rows, where both fields exist: act 1 deaths sit at total_floor <= 16,
    act 2 at 19-33, act 3 at >= 34. Not structurally unambiguous: act_floor 1 of a new act shares
    total_floor with the previous act's boss floor, so 16 could be act 2's first floor and 31-33
    could be act 3's first floors. In the new rows no loss falls in those alternative readings.
    """
    if tf is None:
        return None
    if tf <= 16:
        return 1
    if tf <= 33:
        return 2
    return 3


def depth_table(cells: dict[int, list[dict]], new: bool) -> dict:
    if not new and all(r.get("total_floor") is None for a in LEVELS for r in cells[a]):
        return {
            "per_level": None,
            "field": f"not available: the old rows given ({OLD_CSV.name}) carry no total_floor",
        }
    per = {}
    band_agree = band_total = 0
    for a in LEVELS:
        scored = [r for r in cells[a] if not is_bad(r)]
        tfs = []
        died = {1: 0, 2: 0, 3: 0}
        unknown = 0
        for r in scored:
            tf = (r.get("final") or {}).get("total_floor") if new else r.get("total_floor")
            if tf is not None:
                tfs.append(tf)
            if r["won"]:
                continue
            act = (r.get("final") or {}).get("act_index") if new else death_act_from_floor(tf)
            if new:
                band_total += 1
                band_agree += act == death_act_from_floor(tf)
            if act in died:
                died[act] += 1
            else:
                unknown += 1
        n = len(scored)
        per[a] = {
            "n": n,
            "mean_total_floor": statistics.fmean(tfs) if tfs else math.nan,
            "died_act1": died[1],
            "died_act2": died[2],
            "died_act3": died[3],
            "died_unknown_act": unknown,
            "share_died_act1": died[1] / n,
            "share_died_act2": died[2] / n,
            "share_died_act3": died[3] / n,
        }
    out: dict = {"per_level": per}
    if new:
        out["field"] = "final.act_index (1-based: 1, 2, 3) on lost rows; total_floor for the mean"
        out["total_floor_band_vs_act_index"] = {"agree": band_agree, "lost_rows": band_total}
    else:
        out["field"] = (
            "runs.csv has no act_index: act from total_floor bands (<=16 act 1, 17-33 act 2, "
            ">=34 act 3), validated on the new rows (see new.total_floor_band_vs_act_index); "
            "floors 16 and 31-33 are not structurally unambiguous"
        )
    return out


# ---------------------------------------------------------------- paired comparison


def paired(new_cells: dict, old_cells: dict) -> dict:
    per = {}
    act1_agree = act1_total = 0
    diffs_matrix = np.full((len(CLUSTER_SEEDS), len(LEVELS)), np.nan)
    for a in LEVELS:
        nmap = {int(r["seed"]): r for r in new_cells[a]}
        omap = {int(r["seed"]): r for r in old_cells[a]}
        common = sorted(nmap.keys() & omap.keys())
        dropped = [s for s in common if is_bad(nmap[s]) or is_bad(omap[s])]
        seeds = [s for s in common if s not in set(dropped)]
        both = b = c = neither = 0
        agree = 0
        for s in seeds:
            nw, ow = bool(nmap[s]["won"]), bool(omap[s]["won"])
            both += nw and ow
            b += nw and not ow
            c += ow and not nw
            neither += (not nw) and (not ow)
            agree += nmap[s]["act1"] == omap[s]["act1"]
            diffs_matrix[s - SEED0, a] = int(nw) - int(ow)
        act1_agree += agree
        act1_total += len(seeds)
        n = len(seeds)
        lo, hi = newcombe_paired(both, b, c, neither)
        p = mcnemar_exact(b, c)
        p_scipy = float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0
        per[a] = {
            "pairs": n,
            "dropped_seeds": dropped,
            "seeds_only_in_one_arm": len(nmap.keys() ^ omap.keys()),
            "new_rate": (both + b) / n,
            "old_rate": (both + c) / n,
            "diff": (b - c) / n,
            "both_won": both,
            "b_new_won_old_lost": b,
            "c_old_won_new_lost": c,
            "both_lost": neither,
            "mcnemar_exact_p": p,
            "mcnemar_exact_p_scipy": p_scipy,
            "newcombe10_ci95": [lo, hi],
            "act1_agree": agree,
        }
    mean_diff = statistics.fmean(per[a]["diff"] for a in LEVELS)

    # Seed-clustered bootstrap: draw 500 seeds with replacement; a drawn seed brings every level
    # it has a pair in (seeds 0-199 all 11 levels, 200-499 only A9 and A10). Per level, the
    # resampled diff is the multiplicity-weighted mean over drawn seeds; the statistic is the
    # unweighted mean of the 11 level diffs.
    rng = np.random.default_rng(BOOT_RNG_SEED)
    present = ~np.isnan(diffs_matrix)
    dm = np.where(present, diffs_matrix, 0.0)
    pm = present.astype(float)
    k = len(CLUSTER_SEEDS)
    stats = np.empty(BOOT_N)
    stats_strat = np.empty(BOOT_N)
    lo_idx, hi_idx = np.arange(0, 200), np.arange(200, k)
    for i in range(BOOT_N):
        w = np.bincount(rng.integers(0, k, size=k), minlength=k).astype(float)
        num, den = w @ dm, w @ pm
        stats[i] = np.mean(num / den)
    # Sensitivity (not pre-specified): stratified draw, 200 from seeds 0-199 and 300 from 200-499,
    # so every resample keeps the panel's 200/500 shape.
    rng2 = np.random.default_rng(BOOT_RNG_SEED + 1)
    for i in range(BOOT_N):
        w = np.zeros(k)
        np.add.at(w, rng2.choice(lo_idx, size=lo_idx.size, replace=True), 1.0)
        np.add.at(w, rng2.choice(hi_idx, size=hi_idx.size, replace=True), 1.0)
        num, den = w @ dm, w @ pm
        stats_strat[i] = np.mean(num / den)
    return {
        "per_level": per,
        "act1_agreement": {"agree": act1_agree, "pairs": act1_total},
        "pooled": {
            "mean_diff_A0_A10": mean_diff,
            "bootstrap": {
                "method": "seed-clustered percentile bootstrap, unstratified draw of 500 seeds "
                "(81,000,000-81,000,499) with replacement; each seed carries all its levels",
                "resamples": BOOT_N,
                "rng": f"numpy default_rng({BOOT_RNG_SEED})",
                "ci95": [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))],
                "se": float(np.std(stats, ddof=1)),
            },
            "bootstrap_stratified_sensitivity": {
                "method": "not pre-specified: same, but 200 seeds drawn from 81,000,000-199 and "
                "300 from 81,000,200-499",
                "rng": f"numpy default_rng({BOOT_RNG_SEED + 1})",
                "ci95": [
                    float(np.percentile(stats_strat, 2.5)),
                    float(np.percentile(stats_strat, 97.5)),
                ],
                "se": float(np.std(stats_strat, ddof=1)),
            },
            "caveat": "Seeds 81,000,000-199 repeat across all 11 levels and 81,000,200-499 across "
            "A9 and A10, so the 11 level diffs are not independent; the CI resamples seeds as "
            "clusters for that reason. Descriptive only: the diff confounds the champion "
            "(plr_it550 -> a10_it700), the GH #830 search default, and every engine change "
            "between 0cab615a3 and 0181fa552 (690 commits, including the rl2 playout net, "
            "PR #816). It attributes nothing.",
        },
    }


# ---------------------------------------------------------------- integrity


def integrity(new_cells: dict, old_cells: dict, logs: dict, prog: dict) -> dict:
    per = {}
    all_ok = True
    for a in LEVELS:
        rows = new_cells[a]
        n = N_PER_LEVEL[a]
        sc = seed_check(rows, n)
        wrong = {
            "character": sum(r.get("character") != CHARACTER for r in rows),
            "ascension": sum(r.get("ascension") != a for r in rows),
            "final_ascension": sum((r.get("final") or {}).get("ascension") != a for r in rows),
            "policy": sum(r.get("policy") != POLICY for r in rows),
            "wheel_src_hash": sum((r.get("wheel") or {}).get("src_hash") != SRC_HASH for r in rows),
            "wheel_pgo": sum((r.get("wheel") or {}).get("pgo") is not True for r in rows),
            "wheel_pgo_instrumented": sum(
                (r.get("wheel") or {}).get("pgo_instrumented") is not False for r in rows
            ),
            "won_vs_outcome": sum(
                bool(r.get("won")) != (r.get("outcome") == "won")
                for r in rows
                if not is_bad(r)
            ),
        }
        outcomes: dict[str, int] = {}
        for r in rows:
            outcomes[r.get("outcome")] = outcomes.get(r.get("outcome"), 0) + 1
        n_error = sum(r.get("outcome") == "error" or "error" in r for r in rows)
        n_timeout = sum(r.get("outcome") == "timeout" for r in rows)
        lg = logs[a]
        # ppo.probe's "win rate" line drops only outcome == "error" from the denominator.
        probe_scored = [r for r in rows if r.get("outcome") != "error"]
        log_ok = (
            lg.get("wins") == sum(bool(r.get("won")) for r in probe_scored)
            and lg.get("n") == len(probe_scored)
            and lg.get("outcomes") == outcomes
            and lg.get("header", {}).get("n") == len(rows)
            and lg.get("header", {}).get("asc") == a
            and lg.get("header", {}).get("char") == CHARACTER
            and lg.get("header", {}).get("policy") == POLICY
            and not lg.get("crash_banner")
        )
        dn = prog["done"].get(a, {})
        prog_ok = (
            dn.get("rc") == 0
            and dn.get("wins") == lg.get("wins")
            and dn.get("n") == lg.get("n")
            and dn.get("batch_wall_s") == lg.get("batch_wall_s")
            and prog["start"].get(a, {}).get("n") == n
        )
        ok = sc["ok"] and not any(wrong.values()) and log_ok and prog_ok
        all_ok &= ok
        per[a] = {
            "seeds": sc,
            "wrong_field_counts": wrong,
            "outcomes": outcomes,
            "error_rows": n_error,
            "timeout_rows": n_timeout,
            "log_summary": {"wins": lg.get("wins"), "n": lg.get("n"), "outcomes": lg.get("outcomes")},
            "log_agrees_with_rows": log_ok,
            "log_g1b_pass": lg.get("g1b_pass"),
            "progress_done_agrees": prog_ok,
            "jsonl_sha256": sha256(CELLS / f"ironclad_A{a}.jsonl"),
            "ok": ok,
        }
    so_paths = sorted({(r.get("wheel") or {}).get("so_path") for a in LEVELS for r in new_cells[a]})
    py_src = sorted({r.get("py_src") for a in LEVELS for r in new_cells[a]})
    profdata = sorted(
        {(r.get("wheel") or {}).get("profdata_sha256") for a in LEVELS for r in new_cells[a]}
    )
    drv = prog["driver"] or {}
    driver_ok = (
        drv.get("head") == HEAD and drv.get("so") == SO_SHA and drv.get("ckpt", "").endswith(CKPT)
    )
    old_per = {}
    for a in LEVELS:
        rows = old_cells[a]
        sc = seed_check(rows, N_PER_LEVEL[a])
        outcomes: dict[str, int] = {}
        for r in rows:
            outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
        old_per[a] = {
            "seeds": sc,
            "outcomes": outcomes,
            "error_rows": outcomes.get("error", 0),
            "timeout_rows": outcomes.get("timeout", 0),
        }
    return {
        "new_per_level": per,
        "new_all_cells_ok": all_ok,
        "new_total_rows": sum(len(new_cells[a]) for a in LEVELS),
        "new_error_rows": sum(per[a]["error_rows"] for a in LEVELS),
        "new_timeout_rows": sum(per[a]["timeout_rows"] for a in LEVELS),
        "new_distinct_so_path": so_paths,
        "new_distinct_py_src": py_src,
        "new_distinct_profdata_sha256": profdata,
        "driver_start_line": drv,
        "driver_pins_ok": driver_ok,
        "driver_pins_note": "run.sh aborts unless HEAD, the .so sha and the champion ckpt sha "
        f"match ({HEAD[:9]}, {SO_SHA[:8]}, {CKPT_SHA[:8]}); the start line records head, so and "
        "ckpt path (the ckpt sha check passed or the driver would have aborted)",
        "cell_order_run": prog["order"],
        "cell_order_ok": prog["order"] == ORDER,
        "progress_all_done": prog["all_done"],
        "old_per_level": old_per,
        "old_total_rows": sum(len(old_cells[a]) for a in LEVELS),
        "old_error_rows": sum(old_per[a]["error_rows"] for a in LEVELS),
        "old_timeout_rows": sum(old_per[a]["timeout_rows"] for a in LEVELS),
        "old_seeds_ok": all(old_per[a]["seeds"]["ok"] for a in LEVELS),
    }


# ---------------------------------------------------------------- cost


def cost(new_cells: dict, old_cells: dict, logs: dict, prog: dict, old_prog: dict) -> dict:
    per = {}
    for a in LEVELS:
        w = [r["wall_s"] for r in new_cells[a] if "wall_s" in r]
        ow = [r["wall_s"] for r in old_cells[a] if r.get("wall_s") is not None]
        per[a] = {
            "runs": len(w),
            "wall_s_mean": statistics.fmean(w),
            "wall_s_median": statistics.median(w),
            "wall_s_max": max(w),
            "batch_wall_s_log": logs[a].get("batch_wall_s"),
            "runs_per_min_log": logs[a].get("runs_per_min"),
            "cpu_hours": sum(w) / 3600,
            "load_at_start_1_5_15": prog["start"].get(a, {}).get("load_1_5_15"),
            "start": prog["start"].get(a, {}).get("ts"),
            "done": prog["done"].get(a, {}).get("ts"),
            # null when the old rows given carry no wall_s / no old progress.log was given
            "old_wall_s_mean": statistics.fmean(ow) if ow else None,
            "old_wall_s_median": statistics.median(ow) if ow else None,
            "old_batch_wall_s": old_prog.get(a, {}).get("batch_wall_s"),
            "old_cpu_hours": sum(ow) / 3600 if ow else None,
            "wall_s_mean_ratio_new_over_old": statistics.fmean(w) / statistics.fmean(ow) if ow else None,
        }
    old_h = [per[a]["old_cpu_hours"] for a in LEVELS]
    old_b = [per[a]["old_batch_wall_s"] for a in LEVELS]
    tot = {
        "cpu_hours": sum(per[a]["cpu_hours"] for a in LEVELS),
        "batch_wall_h": sum(per[a]["batch_wall_s_log"] for a in LEVELS) / 3600,
        "old_cpu_hours": sum(old_h) if None not in old_h else None,
        "old_batch_wall_h": sum(old_b) / 3600 if None not in old_b else None,
    }
    return {
        "per_level": per,
        "total": tot,
        "notes": "Not a clean cost measurement: the box was shared during the sweep (a CF Pages "
        "wasm build, a test-public-shape wheel build and pytest run, and replays from a "
        "reproduction workflow, mostly during A10, A9 and A0); the load average at each START is "
        "recorded. CPU-hours = sum of per-run wall_s / 3600 (one process per run). The old "
        "sweep ran 32 procs on engine 0cab615a3 with a different champion and search, five "
        "characters interleaved per level; its wall times are context, not a baseline.",
    }


# ---------------------------------------------------------------- markdown


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}"


def fmt_p(p: float) -> str:
    return f"{p:.2g}" if p < 0.001 else f"{p:.3f}"


def fx(x: float | None, spec: str) -> str:
    """Format a number, or n/a for a field the old rows given cannot supply (null in the JSON)."""
    return "n/a" if x is None else format(x, spec)


def render_md(res: dict) -> str:
    new, old, pr = res["new"], res["old"], res["paired"]
    integ, cst = res["integrity"], res["cost"]
    L: list[str] = []
    L.append("# Ironclad release win rates, 2026-10-26: primary analysis (PREREG.md)\n")
    L.append(f"New sweep: {NEW_QUAL}.\n")
    L.append(f"Old column: {OLD_QUAL}.\n")
    L.append("Generated by `analysis/scripts/analyze.py`; every number is recomputed from the rows.\n")

    L.append("## 1. Integrity\n")
    L.append(
        f"- New rows: {integ['new_total_rows']}; error rows {integ['new_error_rows']}, "
        f"timeout rows {integ['new_timeout_rows']} (filtered before anything else; none).\n"
        f"- Every cell holds exactly seeds 81,000,000..+n-1, once: "
        f"{all(integ['new_per_level'][a]['seeds']['ok'] for a in LEVELS)}.\n"
        f"- character, ascension (row and final), policy, wheel.src_hash {SRC_HASH}, wheel.pgo "
        f"true, won consistent with outcome on every row: "
        f"{all(not any(integ['new_per_level'][a]['wrong_field_counts'].values()) for a in LEVELS)}.\n"
        f"- Each cell .log (header, `outcomes:`, `win rate: x/n`) agrees with the rows: "
        f"{all(integ['new_per_level'][a]['log_agrees_with_rows'] for a in LEVELS)}; "
        f"G1b ownership audit PASS in every log: "
        f"{all(integ['new_per_level'][a]['log_g1b_pass'] for a in LEVELS)}.\n"
        f"- progress.log DONE lines (rc 0, wins/n, batch wall) agree: "
        f"{all(integ['new_per_level'][a]['progress_done_agrees'] for a in LEVELS)}; "
        f"cell order {integ['cell_order_run']} as pre-registered: {integ['cell_order_ok']}; "
        f"ALL DONE: {integ['progress_all_done']}.\n"
        f"- Driver start line pins head/.so/ckpt: {integ['driver_pins_ok']}; distinct py_src "
        f"{integ['new_distinct_py_src']}, distinct profdata "
        f"{[s[:12] for s in integ['new_distinct_profdata_sha256']]}.\n"
        f"- Old Ironclad rows: {integ['old_total_rows']}; seeds complete: {integ['old_seeds_ok']}; "
        f"error rows {integ['old_error_rows']}, timeout rows {integ['old_timeout_rows']}.\n"
        f"- Old column recomputed from {OLD_CSV.name} with the same code matches the published column "
        f"(rates, Wilson CIs, mean, act-1 split): {res['old_vs_published']['matches_published']}"
        + (f" (mismatches: {res['old_vs_published']['mismatches']})" if res["old_vs_published"]["mismatches"] else "")
        + ".\n"
    )

    L.append("## 2-3. Per level, new vs old, paired by seed\n")
    L.append(
        "Win rate in %, Wilson 95% interval in brackets. Diff = new - old in pp on the paired "
        "seeds; b = new won & old lost, c = old won & new lost; p = exact two-sided McNemar; "
        "paired CI = Newcombe (1998) method 10. Descriptive only: the diff confounds the "
        "champion, the GH #830 search default and 690 commits of engine changes.\n"
    )
    L.append(
        "| Asc | n | new (a10_it700, #830 default) | old (plr_it550) | pairs | diff pp "
        "| paired 95% CI | b / c | McNemar p |"
    )
    L.append("|---|---|---|---|---|---|---|---|---|")
    for a in LEVELS:
        nl, ol, pl = new["per_level"][a], old["per_level"][a], pr["per_level"][a]
        L.append(
            f"| A{a} | {nl['n']} | {nl['wins']}/{nl['n']} = {pct(nl['rate'])} "
            f"[{pct(nl['wilson95'][0])}-{pct(nl['wilson95'][1])}] | "
            f"{ol['wins']}/{ol['n']} = {pct(ol['rate'])} "
            f"[{pct(ol['wilson95'][0])}-{pct(ol['wilson95'][1])}] | {pl['pairs']} | "
            f"{100 * pl['diff']:+.1f} | [{100 * pl['newcombe10_ci95'][0]:+.1f}, "
            f"{100 * pl['newcombe10_ci95'][1]:+.1f}] | {pl['b_new_won_old_lost']} / "
            f"{pl['c_old_won_new_lost']} | {fmt_p(pl['mcnemar_exact_p'])} |"
        )
    bs = pr["pooled"]["bootstrap"]
    # Totals summed from the tables (reconcile fix F2: this row used to hard-code 3,400).
    n_total = sum(new["per_level"][a]["n"] for a in LEVELS)
    pairs_total = sum(pr["per_level"][a]["pairs"] for a in LEVELS)
    L.append(
        f"| **mean A0-A10** | {n_total:,} | **{pct(new['mean_A0_A10'])}** | **{pct(old['mean_A0_A10'])}** "
        f"| {pairs_total:,} | **{100 * pr['pooled']['mean_diff_A0_A10']:+.1f}** | "
        f"[{100 * bs['ci95'][0]:+.1f}, {100 * bs['ci95'][1]:+.1f}] seed-clustered bootstrap | | |\n"
    )
    L.append(
        "The mean is the unweighted mean of the 11 per-level rates (as in the 2026-10-11 write-up). "
        f"Bootstrap: {bs['method']}, {bs['resamples']:,} resamples, {bs['rng']}. "
        f"{pr['pooled']['caveat']} Sensitivity (not pre-specified), stratified draw: "
        f"[{100 * pr['pooled']['bootstrap_stratified_sensitivity']['ci95'][0]:+.1f}, "
        f"{100 * pr['pooled']['bootstrap_stratified_sensitivity']['ci95'][1]:+.1f}]. "
        "The 11 per-level McNemar p values are uncorrected for multiplicity.\n"
    )
    ag = pr["act1_agreement"]
    L.append(f"act1 agrees between the arms on {ag['agree']} of {ag['pairs']} paired runs.\n")

    L.append("### Act 1 split (pooled over all 11 cells, by seed)\n")
    L.append("| | Overgrowth | Underdocks |")
    L.append("|---|---|---|")
    for label, tab in (
        ("new: Ironclad a10_it700, #830 default, 8000 sims", new),
        ("old: Ironclad plr_it550, 8000 sims", old),
    ):
        s = tab["act1_split"]
        L.append(
            f"| {label} | {s['overgrowth']['wins']}/{s['overgrowth']['n']} = {pct(s['overgrowth']['rate'])} "
            f"| {s['underdocks']['wins']}/{s['underdocks']['n']} = {pct(s['underdocks']['rate'])} |"
        )
    L.append(
        "\nThe act is a function of the seed, so this is a by-seed split, not a paired comparison "
        "of acts. Pooled raw wins/n, so A9 and A10 (500 seeds each) weigh 2.5x a 200-seed level.\n"
    )

    L.append("## 4. Cost per cell (shared box; not a clean cost number)\n")
    L.append(
        "| Asc | load at START (1/5/15 min) | wall_s mean | median | max | batch wall s | CPU-h "
        "| old wall_s mean (32 procs, 0cab615a3) | new/old mean |"
    )
    L.append("|---|---|---|---|---|---|---|---|---|")
    for a in ORDER:
        c = cst["per_level"][a]
        ld = c["load_at_start_1_5_15"]
        L.append(
            f"| A{a} | {ld[0]:.2f} / {ld[1]:.2f} / {ld[2]:.2f} | {c['wall_s_mean']:.1f} | "
            f"{c['wall_s_median']:.1f} | {c['wall_s_max']:.1f} | {c['batch_wall_s_log']} | "
            f"{c['cpu_hours']:.1f} | {fx(c['old_wall_s_mean'], '.1f')} | "
            f"{fx(c['wall_s_mean_ratio_new_over_old'], '.2f')} |"
        )
    t = cst["total"]
    L.append(
        f"| total | | | | | {t['batch_wall_h']:.2f} h | {t['cpu_hours']:.1f} | "
        + (
            "(old: n/a) | |\n"
            if t["old_cpu_hours"] is None or t["old_batch_wall_h"] is None
            else f"(old: {t['old_cpu_hours']:.1f} CPU-h, {t['old_batch_wall_h']:.2f} h batch wall) | |\n"
        )
    )
    L.append(f"Rows in run order. {cst['notes']}\n")

    L.append("## 5. Run depth (descriptive)\n")
    nd, od = res["depth_new"], res["depth_old"]
    L.append(
        f"New: {nd['field']}. total_floor band vs act_index on the new lost rows: "
        f"{nd['total_floor_band_vs_act_index']['agree']}/{nd['total_floor_band_vs_act_index']['lost_rows']} "
        f"agree. Old: {od['field']}. Shares are of all scored runs in the cell.\n"
    )
    L.append(
        "| Asc | n | new mean total_floor | new died act 1 / 2 / 3 (%) | old mean total_floor "
        "| old died act 1 / 2 / 3 (%) |"
    )
    L.append("|---|---|---|---|---|---|")
    for a in LEVELS:
        x = nd["per_level"][a]
        y = od["per_level"][a] if od["per_level"] is not None else None
        old_cols = (
            "n/a | n/a"
            if y is None
            else f"{y['mean_total_floor']:.1f} | "
            f"{pct(y['share_died_act1'])} / {pct(y['share_died_act2'])} / {pct(y['share_died_act3'])}"
        )
        L.append(
            f"| A{a} | {x['n']} | {x['mean_total_floor']:.1f} | "
            f"{pct(x['share_died_act1'])} / {pct(x['share_died_act2'])} / {pct(x['share_died_act3'])} | "
            f"{old_cols} |"
        )
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------- main


def main() -> int:
    new_cells = load_new()
    old_cells = load_old()
    logs = {a: parse_cell_log(a) for a in LEVELS}
    prog = parse_progress()
    old_prog = parse_old_progress()

    new_tab = level_table(new_cells)
    old_tab = level_table(old_cells)
    res = {
        "qualifier_new": NEW_QUAL,
        "qualifier_old": OLD_QUAL,
        "integrity": integrity(new_cells, old_cells, logs, prog),
        "new": new_tab,
        "old": old_tab,
        "old_vs_published": check_published_old(old_tab),
        "paired": paired(new_cells, old_cells),
        "cost": cost(new_cells, old_cells, logs, prog, old_prog),
        "depth_new": depth_table(new_cells, new=True),
        "depth_old": depth_table(old_cells, new=False),
        "inputs": {
            "sweep_dir": str(HERE),
            "old_runs_csv": str(OLD_CSV),
            "old_runs_csv_sha256": sha256(OLD_CSV),
            "progress_log_sha256": sha256(PROGRESS),
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, sort_keys=False)
        f.write("\n")
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(render_md(res))
    print(OUT_MD.read_text(encoding="utf-8"))
    return 0 if res["integrity"]["new_all_cells_ok"] and res["old_vs_published"]["matches_published"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
