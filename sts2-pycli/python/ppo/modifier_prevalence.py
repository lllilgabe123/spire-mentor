"""Card-modifier prevalence driver (GH #947, design doc M0 / validation stage 0).

How often does a run decision involve a card the PPO run policy cannot see
correctly? The policy's card tokens carried only ``(id, upgraded)`` until the
``card_mods`` feature, so an enchanted card read as the bare card. This plays
champion runs through ``ppo.workers.play_batch`` with ``record_decisions`` on
and keeps, per decision, only what sizes that gap:

  * the deck: size, how many cards are enchanted, and with what;
  * the options: how many of the decision's card options are enchanted
    (reward options, removal / grant options, and the deck positions an
    upgrade / transform / enchant / shop-removal option names), and whether
    the CHOSEN option was one of them.

One ``dec`` row per non-trivial decision and one ``run`` row per run stream
to ``--out`` (JSONL), chunk by chunk, so a long sweep shows progress and
survives being killed (``--resume`` skips seeds already written).

    PYTHONPATH=python python -m ppo.modifier_prevalence \\
        --ckpt data/ppo/<champion>.pt --character Ironclad --ascension 10 \\
        --seeds 200 --act1 roll --out /tmp/prev.ironclad.a10.jsonl

``ppo.modifier_prevalence_report`` summarizes one or more output files.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from typing import Any

# Action kinds whose option names a master-deck position through an
# obs-carried offer map (`<kind>: <obs key of that map>`; the action's `idx`
# indexes the map, the map entry indexes the deck).
_DECK_MAPPED = {
    "upgrade_deck_card": "upgrade_select_deck_indices",
    "transform_deck_card": "transform_select_deck_indices",
    "enchant_deck_card": "enchant_select_deck_indices",
}


def _at(lst: list | None, i: Any) -> Any:
    if lst is None or i is None:
        return None
    i = int(i)
    return lst[i] if 0 <= i < len(lst) else None


def option_enchantment(obs: dict[str, Any], action: dict[str, Any]) -> tuple[bool, Any]:
    """``(is_card_option, enchantment)`` for one candidate action.

    ``enchantment`` is ``None`` or ``[id, amount]``. Non-card options (paths,
    relics, events, skips, ...) return ``(False, None)``.
    """
    kind = action.get("action")
    deck_ench = obs.get("deck_enchantments") or []
    if kind == "pick_card_reward":
        if action.get("idx") is None:
            return False, None
        return True, _at(obs.get("card_reward_option_enchantments"), action["idx"])
    if kind == "remove_deck_card":
        return True, _at(obs.get("card_removal_option_enchantments"), action.get("idx"))
    if kind == "grant_deck_card":
        return True, _at(obs.get("card_grant_option_enchantments"), action.get("idx"))
    if kind in _DECK_MAPPED:
        deck_i = _at(obs.get(_DECK_MAPPED[kind]), action.get("idx"))
        return True, _at(deck_ench, deck_i)
    if kind == "shop_remove_card":
        return True, _at(deck_ench, action.get("deck_idx"))
    if kind == "smith_card":
        return True, _at(obs.get("smith_option_enchantments"), action.get("idx"))
    if kind == "shop_buy_card":
        # Shop cards are never enchanted: a card option, unenchanted.
        return True, None
    return False, None


def reduce_row(row: dict[str, Any], character: str) -> list[dict[str, Any]]:
    """Compact one ``play_one`` row into prevalence rows."""
    from ppo.encoder import fine_surface

    seed = row["seed"]
    asc = row.get("ascension")
    if row.get("outcome") == "error":
        return [
            {
                "kind": "run",
                "character": character,
                "seed": seed,
                "ascension": asc,
                "error": row.get("error"),
            }
        ]
    out: list[dict[str, Any]] = []
    last_deck: list[Any] = []
    for d in row.get("decisions") or []:
        actions = d["actions"]
        if len(actions) < 2:
            continue
        obs, m = d["obs"], d["metrics"]
        deck_ench = obs.get("deck_enchantments")
        if deck_ench is None:
            raise RuntimeError(
                "recorded obs has no `deck_enchantments`: the imported sts2_rs / "
                "ppo.env predates GH #947, so every count here would read zero"
            )
        last_deck = deck_ench
        n_opt = n_opt_mod = 0
        for a in actions:
            is_card, ench = option_enchantment(obs, a)
            n_opt += is_card
            n_opt_mod += ench is not None
        chosen_card, chosen_ench = option_enchantment(obs, actions[d["chosen"]])
        out.append(
            {
                "kind": "dec",
                "character": character,
                "seed": seed,
                "ascension": asc,
                "act_index": m["act_index"],
                "total_floor": m["total_floor"],
                "surface": fine_surface(actions),
                "n_actions": len(actions),
                "n_deck": len(deck_ench),
                "n_deck_mod": sum(e is not None for e in deck_ench),
                "n_card_opt": n_opt,
                "n_card_opt_mod": n_opt_mod,
                "chosen_card": chosen_card,
                "chosen_mod": chosen_ench is not None,
            }
        )
    final_ench = Counter(e[0] for e in last_deck if e is not None)
    out.append(
        {
            "kind": "run",
            "character": character,
            "seed": seed,
            "ascension": asc,
            "act1": row.get("act1"),
            "outcome": row.get("outcome"),
            "won": row.get("won"),
            "final": row.get("final"),
            "final_deck_enchantments": dict(final_ench),
            "wall_s": row.get("wall_s"),
            "wheel": row.get("wheel"),
        }
    )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--policy", default="net_greedy", choices=("net", "net_greedy"))
    ap.add_argument("--ascension", type=int, default=10)
    ap.add_argument("--character", default="Ironclad")
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--act1", default="roll", choices=("overgrowth", "underdocks", "roll"))
    ap.add_argument("--procs", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=48, help="seeds per play_batch call")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true", help="skip seeds with a run row in --out")
    args = ap.parse_args(argv)

    from ppo.workers import play_batch

    done: set[int] = set()
    if args.resume and os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["kind"] == "run" and "error" not in r:
                    done.add(r["seed"])
    seeds = [s for s in range(args.seed_start, args.seed_start + args.seeds) if s not in done]
    t0 = time.time()
    n_runs = n_won = n_err = 0
    with open(args.out, "a" if args.resume else "w", encoding="utf-8") as sink:
        for i in range(0, len(seeds), args.chunk):
            rows = play_batch(
                seeds[i : i + args.chunk],
                policy=args.policy,
                ascension=args.ascension,
                character=args.character,
                procs=args.procs,
                record_decisions=True,
                ckpt_path=args.ckpt,
                act1=args.act1,
            )
            for row in rows:
                for r in reduce_row(row, args.character):
                    sink.write(json.dumps(r) + "\n")
                n_runs += 1
                n_won += bool(row.get("won"))
                n_err += row.get("outcome") == "error"
            sink.flush()
            print(
                f"[modifier_prevalence] {args.character} A{args.ascension} "
                f"{n_runs}/{len(seeds)} runs, {n_won} won, {n_err} errored, "
                f"{time.time() - t0:.0f}s",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
