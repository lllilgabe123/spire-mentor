"""Summarize ``ppo.modifier_prevalence`` output (GH #947, validation stage 0).

    PYTHONPATH=python python -m ppo.modifier_prevalence_report OUT1.jsonl [OUT2 ...]

Prints, per character and pooled:

  * runs, wins and errored rows (errored seeds are dropped, never scored);
  * deck exposure: the share of decisions whose deck holds at least one
    enchanted card, and the mean enchanted share of the deck, by act;
  * option exposure: the share of decisions with an enchanted card among
    their OPTIONS, by decision surface, and how often the chosen option was
    one; this is where an id-only token gets the decision itself wrong;
  * which enchantments the final decks hold.

Rates are per decision, but decisions cluster within runs (one enchanted
card stays in the deck for the rest of the run), so each share carries a
run-clustered standard error: per-run numerator and denominator totals, ratio
estimator with the linearized variance.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable


def load(paths: Iterable[str]) -> tuple[list[dict], list[dict]]:
    decs: list[dict] = []
    runs: list[dict] = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                (decs if r["kind"] == "dec" else runs).append(r)
    bad = {(r["character"], r["seed"]) for r in runs if "error" in r}
    decs = [d for d in decs if (d["character"], d["seed"]) not in bad]
    return decs, runs


def clustered_share(
    decs: list[dict], hit: Callable[[dict], float], weight: Callable[[dict], float] = lambda _: 1.0
) -> tuple[float, float, int]:
    """``(share, run-clustered SE, n decisions)`` of ``sum hit / sum weight``."""
    num: dict[tuple, float] = defaultdict(float)
    den: dict[tuple, float] = defaultdict(float)
    n = 0
    for d in decs:
        w = weight(d)
        if w <= 0:
            continue
        key = (d["character"], d["seed"])
        num[key] += hit(d)
        den[key] += w
        n += 1
    k = len(den)
    total = sum(den.values())
    if k == 0 or total == 0:
        return float("nan"), float("nan"), 0
    r = sum(num.values()) / total
    if k < 2:
        return r, float("nan"), n
    mean_den = total / k
    resid = [(num[c] - r * den[c]) / mean_den for c in den]
    se = math.sqrt(sum(x * x for x in resid) / (k * (k - 1)))
    return r, se, n


def fmt(share: float, se: float, n: int) -> str:
    if n == 0:
        return "—"
    return f"{100 * share:5.1f}% ± {100 * se:4.1f} (n={n})"


def report(decs: list[dict], runs: list[dict], label: str) -> list[str]:
    ok = [r for r in runs if "error" not in r]
    out = [f"### {label}", ""]
    out.append(
        f"runs {len(ok)} ({sum(bool(r.get('won')) for r in ok)} won), "
        f"{len(runs) - len(ok)} errored and dropped, {len(decs)} decisions with ≥2 options"
    )
    out.append("")
    out.append("| scope | deck holds an enchanted card | enchanted share of deck |")
    out.append("|---|---|---|")
    scopes: list[tuple[str, list[dict]]] = [("all acts", decs)]
    for act in sorted({d["act_index"] for d in decs}):
        scopes.append((f"act {act}", [d for d in decs if d["act_index"] == act]))
    for name, ds in scopes:
        any_mod = clustered_share(ds, lambda d: float(d["n_deck_mod"] > 0))
        frac = clustered_share(ds, lambda d: d["n_deck_mod"], lambda d: d["n_deck"])
        out.append(f"| {name} | {fmt(*any_mod)} | {fmt(*frac)} |")
    out.append("")
    out.append(
        "| surface | decisions | ≥1 enchanted card option | chosen option enchanted "
        "(of card choices) |"
    )
    out.append("|---|---|---|---|")
    by_surface: dict[str, list[dict]] = defaultdict(list)
    for d in decs:
        if d["n_card_opt"] > 0:
            by_surface[d["surface"]].append(d)
    card_decs = [d for ds in by_surface.values() for d in ds]
    rows = sorted(by_surface.items(), key=lambda kv: -len(kv[1]))
    for surface, ds in [("all card decisions", card_decs), *rows]:
        opt = clustered_share(ds, lambda d: float(d["n_card_opt_mod"] > 0))
        chosen = clustered_share(
            ds, lambda d: float(d["chosen_mod"]), lambda d: float(d["chosen_card"])
        )
        out.append(f"| {surface} | {len(ds)} | {fmt(*opt)} | {fmt(*chosen)} |")
    out.append("")
    final = Counter()
    holders = 0
    for r in ok:
        e = r.get("final_deck_enchantments") or {}
        final.update(e)
        holders += bool(e)
    out.append(
        f"final decks holding ≥1 enchanted card: {holders}/{len(ok)}; enchanted cards by id: "
        + (", ".join(f"{k} {v}" for k, v in final.most_common()) or "none")
    )
    out.append("")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args(argv)
    decs, runs = load(args.paths)
    lines: list[str] = []
    for ch in sorted({r["character"] for r in runs}):
        lines += report(
            [d for d in decs if d["character"] == ch], [r for r in runs if r["character"] == ch], ch
        )
    if len({r["character"] for r in runs}) > 1:
        lines += report(decs, runs, "pooled")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
