"""Forced-pick counterfactual: is the exhaust / self-damage engine a local optimum?

The Ironclad champion almost never takes the exhaust / self-damage engine that
strong humans build around (GH #778: Burning Pact 7% vs pros 55%, Forgotten
Ritual 0% vs 51%, ...). GH #777 showed the combat pilot is not the reason: the
engine cards are no more search-hungry than other cards at 250, 1000 or 5000
sims. That leaves two readings this driver separates:

  * the champion is RIGHT for this pilot — taking the cards loses runs; or
  * it is a LOCAL OPTIMUM — taking them wins at least as often, but the policy
    never tries them (0-9% take rates leave no data) and a lone engine card is
    weak one-for-one, so a single deviation looks bad even if the package pays.

Each seed is played under every arm with the champion (greedy) making every
decision except at card rewards that offer an engine card:

  none     the champion's own pick (the control);
  single   the FIRST such reward takes an engine card, then the champion plays on;
  half     each such reward takes one with probability 0.5 (seeded per game) —
           closer to the pros' ~50% take rate than `package`;
  package  EVERY such reward takes an engine card.

When several engine cards are offered, the one the champion rates highest is
taken. Shops, events and other card sources are left to the champion. Arms
share the seed panel, so the report pairs them per seed (McNemar), on all seeds
and on the seeds where the arm actually overrode the champion.

    PYTHONPATH=python python -m ppo.forced_pick run \\
        --ckpt data/ppo/<champion>.pt --ascension 10 --seeds 2000 \\
        --sims 250 --boss-sims 1000 --out ~/forced-pick/a10
    PYTHONPATH=python python -m ppo.forced_pick report --out ~/forced-pick/a10
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any

# The GH #777 engine set (the cards the PPO policy avoids, #778), plus Cruelty
# from the pro take-rate table. Vocab names; offered ids are matched through
# `vocab.cards().index`, so the engine's raw ids need not match the spelling.
ENGINE_CARDS = (
    "BURNING_PACT",
    "FEEL_NO_PAIN_CARD",
    "FORGOTTEN_RITUAL",
    "BLOODLETTING",
    "DARK_EMBRACE_CARD",
    "SECOND_WIND",
    "TRUE_GRIT",
    "OFFERING",
    "HEMOKINESIS",
    "BLOOD_WALL",
    "CRUELTY_CARD",
)
ARMS = ("none", "single", "half", "package")


class ForcedPick:
    """Greedy champion choices, with engine-card picks forced per `arm`."""

    def __init__(self, arm: str, seed: int = 0, cards: tuple[str, ...] = ENGINE_CARDS) -> None:
        import random

        from ppo import vocab

        if arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
        reg = vocab.cards()
        self._engine = {reg.index(c) for c in cards} - {reg.oov}
        if len(self._engine) != len(cards):
            raise ValueError(f"engine cards missing from the card vocab: {cards}")
        self._index = reg.index
        self.arm = arm
        # Its own stream: the coin flips must not move with anything else.
        self._rng = random.Random(f"forced_pick/{seed}")
        self.engine_offers = 0
        self.forced = 0
        self.taken: list[dict[str, Any]] = []

    def _is_engine(self, card_id: str) -> bool:
        return self._index(card_id) in self._engine

    def choose(self, pol: Any, d: Any) -> int:
        from ppo.encoder import fine_surface

        probs, _ = pol.distribution(d.obs, d.actions, d.metrics)
        greedy = max(range(len(probs)), key=probs.__getitem__)
        if fine_surface(d.actions) != "card-reward":
            return greedy
        offered = list(d.obs.get("card_reward_option_ids") or [])
        engine = [
            i
            for i, a in enumerate(d.actions)
            if a.get("action") == "pick_card_reward"
            and a.get("idx") is not None
            and int(a["idx"]) < len(offered)
            and self._is_engine(offered[int(a["idx"])])
        ]
        choice = greedy
        if engine:
            self.engine_offers += 1
            force = (
                self.arm == "package"
                or (self.arm == "single" and self.engine_offers == 1)
                or (self.arm == "half" and self._rng.random() < 0.5)
            )
            if force:
                choice = max(engine, key=probs.__getitem__)
                self.forced += choice != greedy
            if choice in engine:
                self.taken.append(
                    {
                        "card": offered[int(d.actions[choice]["idx"])],
                        "floor": int(d.metrics["total_floor"]),
                        "forced": choice != greedy,
                    }
                )
        return choice

    def stats(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "engine_offers": self.engine_offers,
            "forced": self.forced,
            "taken": self.taken,
        }


def _out_path(out_dir: str, arm: str) -> str:
    return os.path.join(out_dir, f"{arm}.jsonl")


def _done_seeds(path: str) -> set[int]:
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {json.loads(line)["seed"] for line in f if line.strip()}


def _compact(row: dict[str, Any]) -> dict[str, Any]:
    keep = ("seed", "ascension", "act1", "outcome", "won", "reached_second_boss", "wall_s")
    out = {k: row.get(k) for k in keep}
    final = row.get("final") or {}
    out["total_floor"] = final.get("total_floor")
    out["act_index"] = final.get("act_index")
    out["hp"] = final.get("hp")
    out["deck_size"] = final.get("deck_size")
    out["forced_stats"] = row.get("forced_stats")
    out["wheel"] = row.get("wheel")
    if row.get("error"):
        out["error"] = row["error"]
    return out


def run(args: argparse.Namespace) -> int:
    from ppo.workers import play_batch

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w", encoding="utf-8") as f:
        json.dump({**vars(args), "engine_cards": ENGINE_CARDS}, f, indent=2)
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    arms = args.arms.split(",")
    t0 = time.time()
    # Arms interleave per chunk so a run stopped early still has balanced arms.
    for i in range(0, len(seeds), args.chunk):
        for arm in arms:
            path = _out_path(args.out, arm)
            todo = [s for s in seeds[i : i + args.chunk] if s not in _done_seeds(path)]
            if not todo:
                continue
            rows = play_batch(
                todo,
                policy="net_forced",
                ascension=args.ascension,
                character=args.character,
                procs=args.procs,
                ckpt_path=args.ckpt,
                sims=args.sims,
                boss_sims=args.boss_sims,
                search={"arm": arm},
                act1=args.act1,
            )
            with open(path, "a", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(_compact(row)) + "\n")
            wins = sum(bool(r.get("won")) for r in rows)
            print(
                f"[forced_pick] seeds {i + len(todo)}/{len(seeds)} arm={arm} "
                f"won {wins}/{len(rows)} ({time.time() - t0:.0f}s)",
                flush=True,
            )
    return report(args)


def _mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2.0**n
    return min(1.0, 2.0 * tail)


def _load(out_dir: str, arm: str) -> dict[int, dict[str, Any]]:
    path = _out_path(out_dir, arm)
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {r["seed"]: r for r in rows if r.get("outcome") not in ("error", "timeout")}


def compare(base: dict[int, dict], arm: dict[int, dict], key: str) -> dict[str, Any]:
    """Paired comparison of a boolean outcome `key` on the shared seeds."""
    shared = sorted(set(base) & set(arm))
    n = len(shared)
    a = [bool(base[s].get(key)) for s in shared]
    b = [bool(arm[s].get(key)) for s in shared]
    only_base = sum(x and not y for x, y in zip(a, b))
    only_arm = sum(y and not x for x, y in zip(a, b))
    return {
        "n": n,
        "base": round(sum(a) / n, 4) if n else None,
        "arm": round(sum(b) / n, 4) if n else None,
        "delta_pp": round(100.0 * (sum(b) - sum(a)) / n, 2) if n else None,
        "only_base": only_base,
        "only_arm": only_arm,
        "mcnemar_p": round(_mcnemar(only_base, only_arm), 5),
    }


def report(args: argparse.Namespace) -> int:
    base = _load(args.out, "none")
    out: dict[str, Any] = {}
    print(f"\nforced-pick report: {args.out}  (base = arm 'none', n={len(base)})")
    for arm in ("single", "half", "package"):
        rows = _load(args.out, arm)
        if not rows:
            continue
        deviated = {s: r for s, r in rows.items() if (r.get("forced_stats") or {}).get("forced")}
        taken = [len((r.get("forced_stats") or {}).get("taken") or []) for r in rows.values()]
        floors = {
            "base": sum(base[s]["total_floor"] or 0 for s in rows if s in base) / max(1, len(rows)),
            "arm": sum(r["total_floor"] or 0 for r in rows.values()) / max(1, len(rows)),
        }
        res = {
            "won_all": compare(base, rows, "won"),
            "won_deviated": compare(base, deviated, "won"),
            "second_boss_deviated": compare(base, deviated, "reached_second_boss"),
            "deviated_frac": round(len(deviated) / max(1, len(rows)), 3),
            "engine_taken_per_run": round(sum(taken) / max(1, len(taken)), 2),
            "mean_floor": {k: round(v, 2) for k, v in floors.items()},
        }
        out[arm] = res
        print(
            f"\n[{arm}] deviated on {res['deviated_frac']:.1%} of seeds, "
            f"{res['engine_taken_per_run']} engine cards taken per run"
        )
        for label in ("won_all", "won_deviated", "second_boss_deviated"):
            c = res[label]
            if c["n"]:
                print(
                    f"  {label:22s} n={c['n']:5d}  {c['base']:.4f} -> {c['arm']:.4f}  "
                    f"{c['delta_pp']:+.2f}pp  ({c['only_base']}/{c['only_arm']})  "
                    f"p={c['mcnemar_p']:.4g}"
                )
        print(f"  mean final floor       {floors['base']:.2f} -> {floors['arm']:.2f}")
    base_taken = [len((r.get("forced_stats") or {}).get("taken") or []) for r in base.values()]
    out["none_engine_taken_per_run"] = round(sum(base_taken) / max(1, len(base_taken)), 2)
    print(f"\n[none] champion takes {out['none_engine_taken_per_run']} engine cards per run")
    with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--ckpt", required=True)
    r.add_argument("--ascension", type=int, default=10)
    r.add_argument("--character", default="Ironclad")
    r.add_argument("--seeds", type=int, default=2000)
    r.add_argument("--seed-start", type=int, default=140_000_000)
    r.add_argument("--sims", type=int, default=250)
    r.add_argument("--boss-sims", type=int, default=1000)
    r.add_argument("--act1", default="roll", choices=("overgrowth", "underdocks", "roll"))
    r.add_argument("--arms", default=",".join(ARMS))
    r.add_argument("--procs", type=int, default=None)
    r.add_argument("--chunk", type=int, default=240, help="seeds per arm per play_batch call")
    r.add_argument("--out", required=True)
    rep = sub.add_parser("report")
    rep.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "run":
        return run(args)
    return report(args)


if __name__ == "__main__":
    sys.exit(main())
