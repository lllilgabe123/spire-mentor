"""`python -m ppo.combat_bench.replay_duel`: the pilot replays a human's fight from its start (GH #830).

SpireCodex v0.111.0 replays carry the run's RNG counters, so a fight's start
state rebuilds exactly in the engine (`ppo.bc_human.sync`). Here the pilot
plays that same start state to the end, once per arm, and the record puts its
per-round damage, potions, cards and fight length next to the human's from the
log. The pairing unit is the fight: same deck, relics, belt, HP, roster and
opening hand; the RNG diverges only once the two lines differ.

    python -m ppo.combat_bench.replay_duel --out <dir> [--room elite] \\
        [--encounters A,B] [--arms B0=1000 B2=4000] [--workers 32] \\
        [--files names.json] [--env K=V ...]

`fights.jsonl`: one line per fight with `human` and one record per arm, plus
`sync` (the fight's first desync while replaying the human's own line, `null`
when the whole human fight replays in sync: those fights show the engine
reproduces the human's outcome, so a gap there is not an engine-fidelity gap).
One process per replay file (fault isolation, `--file-timeout-s`); a rerun
skips the files already in `files.jsonl`.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
import zlib
from multiprocessing import get_context
from multiprocessing.connection import wait as wait_conns
from pathlib import Path
from typing import Any

from ppo.bc_human import replay, sync

MAX_ROUNDS = 30
MAX_STEPS = 3000
DEATH_HP = 65


def human_trace(rec: dict[str, Any]) -> dict[str, Any]:
    """The human's fight from the log: per-round damage taken, cards, potions.

    Rounds follow the log's player `turn` events. From replay_version 6 a
    `hit`'s `dmg` is the damage after block, so per-round damage is the sum of
    hits on the player; `hp_lost_total` (the game's own count, Thorns-like HP
    loss included) is the fight's total.
    """
    rnd = 0
    dmg: collections.Counter[int] = collections.Counter()
    blocked: collections.Counter[int] = collections.Counter()
    plays: list[tuple[int, str]] = []
    potions: list[tuple[int, str]] = []
    events = list(rec["pre"])
    for dec in rec["decisions"]:
        events.append(dec["ev"])
        events.extend(dec["after"])
    for d in events:
        t = d["t"]
        if t == "turn" and d.get("side") == "player":
            rnd = int(d.get("round") or d.get("n") or rnd + 1)
        elif t == "hit" and d.get("dst") == "player":
            dmg[rnd] += int(d.get("dmg", 0))
            blocked[rnd] += int(d.get("blocked", 0))
        elif t == "play" and not d.get("auto"):
            plays.append((rnd, sync._norm(d["id"])))
        elif t in ("potion_start", "potion_used") and d.get("id"):
            if not potions or potions[-1] != (rnd, d["id"]) or t == "potion_start":
                potions.append((rnd, d["id"]))
    # A `potion_start` and its `potion_used` are one drink.
    drinks = []
    for p in potions:
        if not drinks or drinks[-1] != p:
            drinks.append(p)
    died = rec["result"] == replay.DEATH
    hp_lost = rec.get("hp_lost_total")
    return {
        "died": died,
        "dmg": int(hp_lost) if hp_lost is not None else sum(dmg.values()),
        "rounds": int(rec.get("turns") or rnd),
        "dmg_by_round": {str(k): v for k, v in sorted(dmg.items())},
        "blocked_by_round": {str(k): v for k, v in sorted(blocked.items())},
        "plays": plays,
        "potions": drinks,
    }


def _step_card(view: dict[str, Any], action: dict[str, Any] | None) -> str | None:
    if not action or action.get("action") != "play":
        return None
    hand = view.get("hand", [])
    i = action.get("card_index")
    return sync._norm(hand[i]["id"]) if i is not None and i < len(hand) else None


def pilot_trace(rec: dict[str, Any], sims: int, seed: int) -> dict[str, Any]:
    """The pilot (deployed `MctsCombatAgent` at `sims`) playing the fight from its start."""
    import sts2_rs

    from ppo.combat_bench import agents

    state = sync.build(rec, monster_ai_offset=len(rec["enemies"]))
    rm = sts2_rs.RunManager(
        seed=seed,
        character=rec["character"].capitalize(),
        ascension_level=rec["ascension"] or 0,
    )
    room = (
        "boss"
        if rec["encounter"].endswith("_BOSS")
        else ("elite" if rec["encounter"].endswith("_ELITE") else "monster")
    )
    rm.debug_enter_combat(state, room)
    rm.set_record_combat_log(True)
    spec = agents.mcts_spec(sims, sims, agents.build_env({}))
    agent = agents.make_agent(spec, seed)
    hp0 = int(rec["hp"])
    plays: list[tuple[int, str]] = []
    potions: list[tuple[int, str]] = []
    steps = 0
    end = None
    rnd = 1
    while rm.phase() == "combat" and not rm.is_over():
        cs = rm.combat_state()
        if cs is None:
            break
        g = cs.get_state()  # `hand` / `round`; a card choice has neither
        rnd = int(g.get("round") or rnd)
        if rnd > MAX_ROUNDS or steps >= MAX_STEPS:
            end = "cap"
            break
        belt = sts2_rs.state_to_dict(cs)["player"].get("potions", [])
        step = agent.step(rm)
        if step is None:
            end = "no_action"
            break
        steps += 1
        act = ((step.get("combat_search") or {}).get("recommended")) or {}
        card = _step_card(g, act)
        if card:
            plays.append((rnd, card))
        elif act.get("action") in ("drink_potion", "potion"):
            slot = act.get("slot")
            potions.append((rnd, belt[slot] if slot is not None and slot < len(belt) else "?"))
    log = rm.combat_log()
    dmg: collections.Counter[int] = collections.Counter()
    blocked: collections.Counter[int] = collections.Counter()
    for r in log:
        if r["kind"] == "damage" and (r.get("receiver") or {}).get("kind") == "player":
            rnd = int(r.get("round", 0))
            dmg[rnd] += int(r["hp_before"]) - int(r["hp_after"])
            blocked[rnd] += int(r.get("blocked", 0) or 0)
    died = bool(rm.is_over() and not rm.player_won)
    if end is None:
        end = "died" if died else "won"
    total = sum(dmg.values())
    return {
        "end": end,
        "died": died,
        "dmg": total,
        "hp_loss": hp0 - int(rm.current_hp) if not died else hp0,
        "rounds": rnd,
        "dmg_by_round": {str(k): v for k, v in sorted(dmg.items())},
        "blocked_by_round": {str(k): v for k, v in sorted(blocked.items())},
        "plays": plays,
        "potions": potions,
        "steps": steps,
    }


def loss(died: bool, dmg: int, hp_before: int) -> int:
    """The bench's L: damage (clamped to the HP the fight started with), or all of it plus 65."""
    return hp_before + DEATH_HP if died else min(dmg, hp_before)


def file_rows(
    path: str, room: str, encounters: set[str] | None, arms: dict[str, int], rep: int = 0
) -> list[dict[str, Any]]:
    """Every finished fight of `room` in replay `path`, human vs each arm."""
    from ppo.turnplan.pro import fight_turns, room_type

    rows = []
    for rec in replay.fights(path):
        if rec["result"] not in replay.FINISHED or not rec["rng_state"]:
            continue
        if room_type(rec["encounter"]) != room:
            continue
        if encounters and rec["encounter"] not in encounters:
            continue
        base = {
            "file": Path(path).name,
            "combat_id": rec["combat_id"],
            "attempt": rec.get("attempt", 0),
            "encounter": rec["encounter"],
            "act": rec["act"],
            "floor": rec["floor"],
            "hp_before": int(rec["hp"]),
            "max_hp": int(rec["max_hp"]),
            "deck_n": len(rec["deck"]),
            "belt": [p for p in rec["belt"] if p],
            "relics_n": len(rec["relics"]),
        }
        try:
            turns, desync = fight_turns(rec)
            base["sync"] = list(desync) if desync else None
            base["sync_turns"] = sum(t["status"] == "legal" for t in turns)
            h = human_trace(rec)
            base["human"] = h | {"L": loss(h["died"], h["dmg"], base["hp_before"])}
            key = f"{base['file']}:{base['combat_id']}" + (f":{rep}" if rep else "")
            seed = zlib.crc32(key.encode()) % (2**31)
            base["rep"] = rep
            for name, sims in arms.items():
                p = pilot_trace(rec, sims, seed)
                base[name] = p | {"L": loss(p["died"], p["dmg"], base["hp_before"])}
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as e:
            base["error"] = repr(e)[:300]
        rows.append(base)
    return rows


def _proc(
    path: str,
    room: str,
    enc: set[str] | None,
    arms: dict[str, int],
    conn: Any,
    env: dict[str, str] | None = None,
    rep: int = 0,
) -> None:
    from ppo.combat_bench import agents

    agents.apply_env(env or {})  # the deployed search plus `--env`: no inherited engine seams
    try:
        out: dict[str, Any] = {"rows": file_rows(path, room, enc, arms, rep)}
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:
        out = {"crash": repr(e)[:300]}
    conn.send(out)
    conn.close()


def main(argv: list[str] | None = None) -> int:
    from ppo.turnplan import pro

    ap = argparse.ArgumentParser(prog="python -m ppo.combat_bench.replay_duel", description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--room", default="elite")
    ap.add_argument("--encounters", default=None, help="comma list; default every one")
    ap.add_argument("--ascension", type=int, nargs="+", default=[10])
    ap.add_argument("--arms", nargs="+", default=["B0=1000", "B2=4000"])
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--file-timeout-s", type=float, default=3600.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--files",
        default=None,
        help="JSON list of replay file names: play only these (e.g. one uploader group)",
    )
    ap.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="K=V",
        help="engine env for the workers (e.g. STS2_RUNAI_PRIOR_NET=off); all others are scrubbed",
    )
    ap.add_argument(
        "--rep",
        type=int,
        default=0,
        help="seed rep: 0 = the original per-fight seed, k > 0 a fresh one (same for every arm)",
    )
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)
    env = dict(kv.split("=", 1) for kv in args.env)
    arms = {a.split("=")[0]: int(a.split("=")[1]) for a in args.arms}
    enc = set(args.encounters.split(",")) if args.encounters else None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files, census = pro.select_files(pro.ROOT, "IRONCLAD", set(args.ascension))
    if args.files:
        keep = set(json.loads(Path(args.files).read_text(encoding="utf-8")))
        files = [f for f in files if Path(f["file"]).name in keep]
    files = files[: args.limit]
    (out / "selection.json").write_text(
        json.dumps({"args": vars(args), "census": census, "files": files}, indent=1) + "\n"
    )
    ledger = out / "files.jsonl"
    done = set()
    if ledger.exists():
        done = {json.loads(line)["file"] for line in ledger.open() if line.strip()}
    todo = iter([f for f in files if Path(f["file"]).name not in done])
    log = (out / "progress.log").open("a")

    def say(msg: str) -> None:
        line = f"{time.strftime('%F %T')} [replay_duel] {msg}"
        print(line, file=sys.stderr, flush=True)
        log.write(line + "\n")
        log.flush()

    say(
        f"{len(files)} files ({census}), {len(done)} done; arms {arms}; room {args.room}; env {env}"
    )
    ctx = get_context("spawn")
    running: dict[Any, tuple[Any, str, float]] = {}
    n = n_rows = 0
    t_last = time.time()
    with ledger.open("a") as led, (out / "fights.jsonl").open("a") as rows_fh:

        def finish(conn: Any, res: dict[str, Any]) -> None:
            nonlocal n, n_rows
            proc, path, t0 = running.pop(conn)
            conn.close()
            proc.join(timeout=10)
            for r in res.get("rows", []):
                rows_fh.write(json.dumps(r) + "\n")
            rows_fh.flush()
            led.write(
                json.dumps(
                    {
                        "file": Path(path).name,
                        "fights": len(res.get("rows", [])),
                        "wall_s": round(time.time() - t0, 1),
                        **({"crash": res["crash"]} if "crash" in res else {}),
                    }
                )
                + "\n"
            )
            led.flush()
            n += 1
            n_rows += len(res.get("rows", []))

        while True:
            while len(running) < args.workers:
                f = next(todo, None)
                if f is None:
                    break
                recv, send = ctx.Pipe(duplex=False)
                proc = ctx.Process(
                    target=_proc,
                    args=(f["file"], args.room, enc, arms, send, env, args.rep),
                    daemon=True,
                )
                proc.start()
                send.close()
                running[recv] = (proc, f["file"], time.time())
            if not running:
                break
            for conn in wait_conns(list(running), timeout=1.0):
                try:
                    res = conn.recv()
                except EOFError:
                    res = {"crash": f"process died ({running[conn][0].exitcode})"}
                finish(conn, res)
            now = time.time()
            for conn, (proc, _, t0) in list(running.items()):
                if now - t0 > args.file_timeout_s:
                    proc.kill()
                    proc.join()
                    finish(conn, {"crash": "watchdog"})
            if now - t_last > 60:
                t_last = now
                say(f"{n} files, {n_rows} fights done; {len(running)} running")
    say(f"done: {n} files, {n_rows} fights")
    return 0


if __name__ == "__main__":
    sys.exit(main())
