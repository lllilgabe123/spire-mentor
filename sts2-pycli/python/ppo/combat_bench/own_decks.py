"""Fight pools for tuning the combat search on two deck sources.

`python -m ppo.combat_bench.own_decks {harvest,engine,split} ...`

The bench's sets replay human fights. Tuning the search on them tunes it for
the decks humans build, which is not where the pilot fights: the run policy
builds its own decks. This module writes two more pools in the bench's fight
schema (`SCHEMA.md`), so `run`, `devpanel` and `report` take them unchanged:

- `harvest`: the champion's OWN decks. It plays full runs with a PPO
  checkpoint (`net_greedy`, the env's deployed combat agent) and writes one
  row per fight at its entry: deck, relics, potion belt, HP and the enemy
  roster. `player` is the run seed (`seed_<n>`), and `human` holds what the
  champion did in that fight in the run.
- `engine`: the pro fights whose deck carries the exhaust / self-damage
  package (`ENGINE_CARDS`, the `ppo.forced_pick` set) at least `--min-engine`
  times.
- `split`: cuts a contested panel (`devpanel`) into tune and holdout halves
  by `player`, so no player's or run's fights sit on both sides.

The encounter of a hallway or elite fight is not on the `RunManager` surface.
Each act pre-rolls its encounters into queues (`generated_rooms`) that its
fights consume in order, so the harvester counts the act's fights per room
type and checks the queued encounter's roster against the live enemies. A
fight whose roster does not match (an event fight, say) is dropped and
counted.

    PYTHONPATH=python python -m ppo.combat_bench.own_decks harvest \\
        --ckpt data/ppo/<champion>.pt --seeds 150 \\
        --seed-start 150000000 --out data/combat_bench/ironclad_ownpool_v1.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
import time
from multiprocessing import get_context
from typing import Any

from ppo.forced_pick import ENGINE_CARDS

# Deck strings name the card without the engine's `_CARD` suffix
# (`FEEL_NO_PAIN`, not `FEEL_NO_PAIN_CARD`).
ENGINE_DECK_IDS = frozenset(c.removesuffix("_CARD") for c in ENGINE_CARDS)
ROOMS = ("monster", "elite", "boss")
RUN_STEP_CAP = 20_000


def card_id(entry: str) -> str:
    """The bare card id of a deck string `ID[+1][@ENCH:amt]`."""
    return entry.split("@", 1)[0].split("+", 1)[0]


def engine_count(deck: list[str]) -> int:
    """How many of the deck's cards are engine cards."""
    return sum(card_id(c) in ENGINE_DECK_IDS for c in deck)


def _id_forms(kind: str) -> dict[str, str]:
    import sts2_rs

    return dict(sts2_rs.list_registry_id_forms(kind))


def deck_strings(
    detailed: list[tuple[str, bool]],
    enchantments: list[tuple[str, int] | None],
    cards: dict[str, str],
) -> list[str]:
    """The run's master deck in the bench encoding `ID[+1][@ENCH:amt]`."""
    out = []
    for (name, upgraded), ench in zip(detailed, enchantments, strict=True):
        s = cards[name] + ("+1" if upgraded else "")
        if ench is not None:
            s += f"@{ench[0]}:{int(ench[1])}"
        out.append(s)
    return out


def queued_encounter(rooms: dict[str, Any], room_type: str, index: int) -> str | None:
    """The encounter the act's `index`-th fight of `room_type` draws."""
    if room_type == "boss":
        return rooms.get("boss")
    queue = rooms.get("normal" if room_type == "monster" else "elite") or []
    return queue[index % len(queue)] if queue else None


class _EntryRecorder:
    """Delegating proxy over the env's combat agent: one bench row per fight."""

    def __init__(self, inner: Any, seed: int, ascension: int) -> None:
        self._inner = inner
        self._seed = seed
        self._ascension = ascension
        self._key: tuple[int, int] | None = None
        self._open: dict[str, Any] | None = None
        self._pending: dict[str, Any] | None = None
        self.mismatches: list[dict[str, Any]] = []
        self._counts: dict[tuple[int, str], int] = collections.Counter()
        self._rosters: dict[str, list[str]] = {}
        self.rows: list[dict[str, Any]] = []
        self.dropped: collections.Counter[str] = collections.Counter()
        self._cards = _id_forms("card")
        self._potions = _id_forms("potion")

    def _roster(self, encounter: str) -> list[str]:
        import sts2_rs

        if encounter not in self._rosters:
            fresh = sts2_rs.RunManager(seed=1, character="Ironclad", ascension_level=0)
            self._rosters[encounter] = [m["id"] for m in fresh.dump_roster(encounter)]
        return self._rosters[encounter]

    def _close(self, rm: Any) -> None:
        if self._open is None:
            return
        row, self._open = self._open, None
        # The last in-combat HP: after the fight, Burning Blood and the like
        # have already healed. A death ends the fight before that step.
        last = row.pop("_hp", None)
        hp_after = 0 if int(rm.current_hp) <= 0 or last is None else last
        row["human"] = {
            "hp_after": hp_after,
            "damage_taken": max(0, row["hp_before"] - hp_after),
            "hp_healed": 0,
            "max_hp_gained": 0,
            "max_hp_lost": 0,
            "turns": row.pop("_turns"),
            "died": hp_after <= 0,
            "quit": False,
        }
        self.rows.append(row)

    def _enter(self, rm: Any) -> None:
        """Note the fight at its first step; `_resolve` finishes the row."""
        act, floor = int(rm.act_index), int(rm.total_floor())
        room_type = (rm.current_room() or {}).get("room_type")
        if room_type not in ROOMS:
            self.dropped["room_type"] += 1
            return
        index = self._counts[act, room_type]
        self._counts[act, room_type] += 1
        self._pending = {
            "act": act,
            "floor": floor,
            "room_type": room_type,
            "index": index,
            "hp_before": int(rm.current_hp),
            "max_hp": int(rm.max_hp),
        }
        self._resolve(rm)

    def _resolve(self, rm: Any) -> None:
        """Open the row once the enemies are visible.

        A fight can open on a pending choice (a start-of-combat card or relic
        prompt), whose `get_state` carries no enemies yet. The run-level fields
        (deck, relics, belt) do not move inside a fight, and the HP was taken
        at the first step, so waiting a step loses nothing.
        """
        p = self._pending
        if p is None:
            return
        state = rm.combat_state().get_state()
        if "enemies" not in state:
            return
        self._pending = None
        rooms = rm.generated_rooms()[p["act"] - 1]
        live = [e["id"] for e in state["enemies"]]
        candidates = [queued_encounter(rooms, p["room_type"], p["index"])]
        if p["room_type"] == "boss" and rooms.get("second_boss"):
            candidates.append(rooms["second_boss"])
        encounter = next((e for e in candidates if e and self._roster(e) == live), None)
        if encounter is None:
            self.dropped["roster_unmatched"] += 1
            self.mismatches.append(
                {
                    "floor": p["floor"],
                    "room_type": p["room_type"],
                    "queued": candidates,
                    "live": live,
                }
            )
            return
        relics = [r.removeprefix("RELIC.") for r in rm.relics()]
        potions = [self._potions[x] for x in rm.potions()]
        self._open = {
            "key": f"seed{self._seed}:{p['floor']}",
            "file": f"seed{self._seed}",
            "floor": p["floor"],
            "act": p["act"],
            "room_type": p["room_type"],
            "encounter": encounter,
            "character": "Ironclad",
            "ascension": self._ascension,
            "player": f"seed_{self._seed}",
            "cohort": "own",
            "run_won": False,
            "run_abandoned": False,
            "deck": deck_strings(
                rm.master_deck_detailed(), rm.master_deck_enchantments(), self._cards
            ),
            "relics": relics,
            "potions": potions,
            "belt": list(potions),
            "potion_slots": int(rm.max_potion_slots()),
            "belt_unused": False,
            "monsters": None,
            "hp_before": p["hp_before"],
            "max_hp": p["max_hp"],
            "n_enchanted": sum(e is not None for e in rm.master_deck_enchantments()),
            "flags": [],
            "_turns": 1,
        }

    def step(self, rm: Any) -> Any:
        if rm.phase() == "combat":
            key = (int(rm.act_index), int(rm.total_floor()))
            if key != self._key:
                self._close(rm)
                self._key = key
                self._enter(rm)
            elif self._pending is not None:
                self._resolve(rm)
            elif self._open is not None:
                # `get_state` carries no `round` in a pending-choice phase.
                g = rm.combat_state().get_state()
                if g.get("round") is not None:
                    self._open["_turns"] = int(g["round"])
                if "player" in g:
                    self._open["_hp"] = int(g["player"]["hp"])
        else:
            self._close(rm)
            self._pending = None
            self._key = None
        return self._inner.step(rm)

    def finish(self, rm: Any) -> None:
        self._close(rm)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def harvest_run(job: tuple[Any, ...]) -> dict[str, Any]:
    """Play one run and return its fight rows."""
    seed, ckpt, ascension, act1 = job
    import numpy as np

    from ppo.env import Decision, PpoRunEnv
    from ppo.workers import _net_policy

    t0 = time.time()
    try:
        kwargs: dict[str, Any] = {} if act1 == "overgrowth" else {"act1": act1}
        env = PpoRunEnv(seed=seed, ascension=ascension, character="Ironclad", **kwargs)
        rec = _EntryRecorder(env.agent, seed, ascension)
        env.agent = rec
        pol = _net_policy(ckpt)
        nrng = np.random.default_rng(seed)
        out = env.reset()
        steps = 0
        while isinstance(out, Decision) and steps < RUN_STEP_CAP:
            idx, _, _, _ = pol.sample(out.obs, out.actions, out.metrics, nrng, greedy=True)
            out = env.step(idx)
            steps += 1
        rec.finish(env.rm)
        won = bool(getattr(out, "won", False))
        for r in rec.rows:
            r["run_won"] = won
        return {
            "seed": seed,
            "won": won,
            "rows": rec.rows,
            "dropped": dict(rec.dropped),
            "mismatches": rec.mismatches,
            "wall_s": round(time.time() - t0, 1),
        }
    except Exception as e:  # a crashed run is logged and skipped, never fatal
        return {"seed": seed, "error": repr(e), "rows": [], "dropped": {}}


def harvest(args: argparse.Namespace) -> int:
    done: set[str] = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as fh:
            done = {json.loads(line)["file"] for line in fh if line.strip()}
    seeds = [
        s for s in range(args.seed_start, args.seed_start + args.seeds) if f"seed{s}" not in done
    ]
    jobs = [(s, args.ckpt, args.ascension, args.act1) for s in seeds]
    log = open(args.out + ".log", "a", encoding="utf-8")
    t0 = time.time()
    n = fights = errors = 0
    dropped: collections.Counter[str] = collections.Counter()
    with (
        get_context("spawn").Pool(args.procs) as pool,
        open(args.out, "a", encoding="utf-8") as sink,
    ):
        for res in pool.imap_unordered(harvest_run, jobs):
            n += 1
            if "error" in res:
                errors += 1
                log.write(json.dumps(res) + "\n")
            for m in res.get("mismatches", []):
                log.write(json.dumps({"seed": res["seed"], "mismatch": m}) + "\n")
            for row in res["rows"]:
                sink.write(json.dumps(row, sort_keys=True) + "\n")
            sink.flush()
            fights += len(res["rows"])
            dropped.update(res["dropped"])
            line = (
                f"[own_decks] {n}/{len(jobs)} runs, {fights} fights, dropped "
                f"{dict(dropped)}, {errors} errors, {time.time() - t0:.0f}s"
            )
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
    return 1 if errors else 0


def engine(args: argparse.Namespace) -> int:
    with open(args.set, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    rooms = set(args.room.split(",")) if args.room else set(ROOMS)
    kept = [
        r for r in rows if r["room_type"] in rooms and engine_count(r["deck"]) >= args.min_engine
    ]
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in kept:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "fights": len(kept),
                "runs": len({r["file"] for r in kept}),
                "players": len({r["player"] for r in kept}),
                "by_room": dict(collections.Counter(r["room_type"] for r in kept)),
            }
        )
    )
    return 0


def split_by_player(
    rows: list[dict[str, Any]], seed: int, frac: float = 0.5
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Tune and holdout halves with no `player` on both sides.

    Players are shuffled with `seed` and dealt greedily to whichever side is
    furthest under its share of FIGHTS, so one prolific player cannot tip the
    halves far from `frac`.
    """
    by_player: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        by_player[r["player"]].append(r)
    players = sorted(by_player)
    random.Random(seed).shuffle(players)
    tune: list[dict[str, Any]] = []
    hold: list[dict[str, Any]] = []
    for p in players:
        side = tune if len(tune) <= frac * (len(tune) + len(hold)) else hold
        side.extend(by_player[p])
    return tune, hold


def split(args: argparse.Namespace) -> int:
    with open(args.set, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    tune, hold = split_by_player(rows, args.seed)
    for path, part in ((args.tune, tune), (args.holdout, hold)):
        with open(path, "w", encoding="utf-8") as fh:
            for r in sorted(part, key=lambda r: r["key"]):
                fh.write(json.dumps(r, sort_keys=True) + "\n")
        rooms = dict(collections.Counter(r["room_type"] for r in part))
        print(f"{path}: {len(part)} fights, {len({r['player'] for r in part})} players, {rooms}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="fight pools from the champion's own decks")
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("harvest", help="fight rows from the champion's own runs")
    h.add_argument("--ckpt", required=True)
    h.add_argument("--ascension", type=int, default=10)
    h.add_argument("--seeds", type=int, default=150)
    h.add_argument("--seed-start", type=int, required=True)
    h.add_argument("--act1", default="roll", choices=("overgrowth", "underdocks", "roll"))
    h.add_argument("--procs", type=int, default=30)
    h.add_argument("--out", required=True, help="pool JSONL; resumed by run seed")
    e = sub.add_parser("engine", help="the fights of an existing set with an engine deck")
    e.add_argument("--set", required=True)
    e.add_argument("--min-engine", type=int, default=4)
    e.add_argument("--room", default=None, help="comma list of room types to keep")
    e.add_argument("--out", required=True)
    s = sub.add_parser("split", help="tune / holdout halves by player")
    s.add_argument("--set", required=True)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--tune", required=True)
    s.add_argument("--holdout", required=True)
    args = ap.parse_args(argv)
    return {"harvest": harvest, "engine": engine, "split": split}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
