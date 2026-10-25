"""`python -m ppo.combat_bench.takeover`: does the human's multi-turn line pay off? (GH #830)

At an in-sync turn start of an exact v0.111 replay (`ppo.turnplan.pro`), the
rest of the fight is played several ways from the same state:

- `P`: the pilot (deployed `MctsCombatAgent`) plays from the turn start;
- `H<k>`: the human's actual next `k` turns are forced first, on the exact RNG
  (only when all `k` turns replayed in sync), then the pilot plays the rest.

Each arm is played `--reps` times with continuation seeds shared across arms
(rep `r` uses the same pilot seed in every arm). The outcome is the rest of
the fight's L: damage taken from the turn start (player damage rows of the
combat log), or the HP held at the turn start plus 65 on a death.

    python -m ppo.combat_bench.takeover --collect <pro collect dir> \\
        --keys <keys.jsonl>... --out <dir> [--k 1 2 3] [--reps 3] [--sims 4000]

One line per position in `positions.jsonl`; one process per replay file
(fault isolation, `--file-timeout-s`); a rerun skips the positions already
written. `progress.log` every minute.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from multiprocessing import get_context
from multiprocessing.connection import wait as wait_conns
from pathlib import Path
from typing import Any

DEATH_HP = 65
MAX_ROUNDS = 30
MAX_STEPS = 3000


def run_action(a: dict[str, Any]) -> dict[str, Any] | None:
    """A combat action dict (`sync._action`) as the RunManager's action dict;
    None when the run surface cannot express it (a multi-card pick)."""
    kind = a.get("action")
    if kind == "play":
        return {
            "action": "play_card",
            "hand_idx": a["card_index"],
            "target_index": a.get("target_index"),
        }
    if kind == "end_turn":
        return {"action": "end_turn"}
    if kind in ("drink_potion", "potion"):
        return {
            "action": "drink_potion",
            "potion_slot": a["slot"],
            "target_index": a.get("target_index"),
        }
    if kind == "choose":
        sel = a.get("selection") or []
        if len(sel) > 1:
            return None
        return {"action": "combat_resolve_choice", "option": sel[0] if sel else None}
    return None


def player_damage(log: list[dict[str, Any]]) -> int:
    """HP the player lost to damage rows of a combat log."""
    return sum(
        int(r["hp_before"]) - int(r["hp_after"])
        for r in log
        if r["kind"] == "damage" and (r.get("receiver") or {}).get("kind") == "player"
    )


def loss(died: bool, dmg: int, hp_start: int) -> int:
    return hp_start + DEATH_HP if died else min(dmg, hp_start)


def _snapshot(cs: Any) -> tuple[Any, ...]:
    import sts2_rs

    d = sts2_rs.state_to_dict(cs)
    g = cs.get_state()
    return (
        d["player"]["hp"],
        tuple(sorted(e["hp"] for e in g.get("enemies", []) if e.get("is_alive", True))),
    )


def play_rest(
    rec: dict[str, Any],
    state: Any,
    forced: list[dict[str, Any]],
    sims: int,
    seed: int,
    rollouts: list[int] | None = None,
    first_turn_only: bool = False,
    post_sink: list[Any] | None = None,
    log_sink: list[Any] | None = None,
) -> dict[str, Any]:
    """Force `forced` (combat actions) on `state`, then let the pilot finish the fight.

    With `rollouts` (seeds) and nothing forced, also record the pilot's first
    turn: the root `q` of every first action it searched, its turn features
    (`turn_features`) and the raw playout value of its pre-EndTurn state
    (`rollout_value` with those seeds).
    """
    import sts2_rs

    from ppo.combat_bench import agents
    from ppo.turnplan.pro import room_type

    hp_start = int(sts2_rs.state_to_dict(state)["player"]["hp"])
    rm = sts2_rs.RunManager(
        seed=seed, character=rec["character"].capitalize(), ascension_level=rec["ascension"] or 0
    )
    rm.debug_enter_combat(state.clone(), room_type(rec["encounter"]))
    rm.set_record_combat_log(True)
    # The expected state after the forced actions, stepped on a plain clone:
    # the RunManager path must reproduce it exactly.
    check = state.clone()
    for a in forced:
        ra = run_action(a)
        if ra is None:
            return {"end": "unforceable"}
        rm.take_action(ra)
        check.apply_step(a)
        if rm.phase() != "combat" or rm.is_over():
            break
    if forced and rm.phase() == "combat" and not rm.is_over():
        cs = rm.combat_state()
        if cs is None or _snapshot(cs) != _snapshot(check):
            return {"end": "force_mismatch"}
    spec = agents.mcts_spec(sims, sims, agents.build_env({}))
    agent = agents.make_agent(spec, seed)
    steps = 0
    end = None
    last_round = 0
    first: dict[str, Any] | None = None
    if rollouts is not None and not forced:
        first = {"actions": [], "trace": [], "post": None, "done": False}
        start_round = int(state.get_state().get("round") or 0)
    while rm.phase() == "combat" and not rm.is_over():
        cs = rm.combat_state()
        if cs is None:
            break
        g = cs.get_state()
        last_round = max(last_round, int(g.get("round") or 0))
        if int(g.get("round") or 0) > MAX_ROUNDS or steps >= MAX_STEPS:
            end = "cap"
            break
        rnd = g.get("round")
        if first is not None and not first["done"] and rnd is not None and int(rnd) > start_round:
            first["done"] = True  # turn 1 ended without a searched EndTurn
        if first is not None and first["done"] and first_turn_only:
            end = "turn1"
            break
        pre = cs.clone() if first is not None and not first["done"] else None
        rec_step = agent.step(rm)
        if rec_step is None:
            end = "no_action"
            break
        steps += 1
        if pre is not None:
            search = rec_step.get("combat_search") or {}
            act = search.get("recommended") or rec_step.get("action") or {}
            first["trace"].append(describe(pre, act))
            if steps == 1:
                first["root"] = [
                    [_ident(g, o["action"]), o.get("q"), o.get("visits")]
                    for o in search.get("actions", [])
                ]
                first["chosen"] = _ident(g, act)
            if act.get("action") == "end_turn":
                first["post"] = pre
                first["done"] = True
                if post_sink is not None:
                    post_sink.append(pre.clone())
                if first_turn_only:
                    end = "turn1"
                    break
            else:
                first["actions"].append((g, act))
    died = bool(rm.is_over() and not rm.player_won)
    log = rm.combat_log()
    if log_sink is not None:
        log_sink.append(log)
    dmg = player_damage(log)
    if end is None:
        end = "died" if died else "won"
    out = {
        "end": end,
        "died": died,
        "dmg": dmg,
        "L": loss(died, dmg, hp_start),
        "steps": steps,
        "hp_start": hp_start,
        "last_round": last_round,
    }
    if first is not None:
        out["first"] = {
            "root": first.get("root", []),
            "chosen": first.get("chosen"),
            "features": features_of(first["actions"], first["post"]),
            "rollout": rollout_value(first["post"], rollouts or []),
            "trace": first["trace"],
            "end": summarize(first["post"]),
            "after_enemy": after_enemy(first["post"]),
        }
    return out


def describe(st: Any, a: dict[str, Any]) -> str:
    """One combat action as readable text, e.g. `BASH+ -> JAW_WORM#0`."""
    import sts2_rs

    g = st.get_state()
    kind = a.get("action")
    ti = a.get("target_index")
    enemies = g.get("enemies") or []
    tgt = ""
    if ti is not None:
        name = enemies[ti]["id"] if ti < len(enemies) else "?"
        tgt = f" -> {name}#{ti}"
    if kind == "play":
        hand = g.get("hand") or []
        i = a.get("card_index")
        c = hand[i] if i is not None and i < len(hand) else {}
        return f"{c.get('id', '?')}{'+' if c.get('upgraded') else ''}{tgt}"
    if kind in ("drink_potion", "potion"):
        pots = sts2_rs.state_to_dict(st)["player"].get("potions") or []
        slot = a.get("slot")
        pid = pots[slot] if slot is not None and slot < len(pots) else "?"
        return f"potion:{pid}{tgt}"
    if kind == "choose":
        cards = g.get("cards") or []
        sel = a.get("selection") or ([a["index"]] if a.get("index") is not None else [])
        picked = [cards[j]["id"] if j < len(cards) else str(j) for j in sel]
        return f"choose[{g.get('kind', '')}]:{','.join(picked) or '-'}"
    return str(kind)


def summarize(st: Any) -> dict[str, Any] | None:
    """The board as the fields a turn diff needs (player, piles, enemies)."""
    if st is None:
        return None
    import sts2_rs

    g = st.get_state()
    d = sts2_rs.state_to_dict(st)
    p = g.get("player") or {}
    return {
        "hp": p.get("hp"),
        "block": p.get("block"),
        "energy": p.get("energy"),
        # `get_state` lists no player powers; the wire state does.
        "powers": {x["name"]: x.get("amount") for x in d["player"].get("powers") or []},
        "hand": [c["id"] + ("+" if c.get("upgraded") else "") for c in g.get("hand") or []],
        "draw": g.get("draw_pile_count"),
        "discard": g.get("discard_pile_count"),
        "exhaust": g.get("exhaust_pile_count"),
        "potions": list(d["player"].get("potions") or []),
        "relics": [r["id"] for r in d["player"].get("relics") or []],
        "enemies": [
            {
                "id": e["id"],
                "hp": e["hp"],
                "block": e["block"],
                "alive": e["is_alive"],
                "intent": e.get("intent"),
                "dmg": e.get("intent_damage"),
                "hits": e.get("intent_hits"),
                "powers": {x["id"]: x.get("amount") for x in e.get("powers") or []},
            }
            for e in g.get("enemies") or []
        ],
    }


def after_enemy(post: Any) -> dict[str, Any] | None:
    """The player after EndTurn and the enemy turn, stepped on a clone of the
    pre-EndTurn state with its own RNG: hp, block, whether the fight ended."""
    if post is None or post.get_state().get("type") == "card_select":
        return None
    st = post.clone()
    try:
        st.apply_step({"action": "end_turn"})
    except ValueError:
        return None
    import sts2_rs

    done, _ = st.terminal_value()
    # The wire state, not `get_state`: a prompt view (a start-of-turn
    # choice) has no player.
    p = sts2_rs.state_to_dict(st)["player"]
    return {"hp": p.get("hp"), "block": p.get("block"), "over": bool(done)}


def _ident(g: dict[str, Any], a: dict[str, Any]) -> list[Any] | None:
    """A combat action by what it does (`pro._identity`), JSON-ready."""
    from ppo.turnplan.pro import _identity

    if not a:
        return None
    try:
        return list(_identity(g, a))
    except (KeyError, IndexError, TypeError):
        return [a.get("action")]


def features_of(steps: list[tuple[dict[str, Any], dict[str, Any]]], post: Any) -> dict[str, Any]:
    """Turn features from `(state dict before, action)` pairs and the pre-EndTurn state."""
    potions = sum(a.get("action") in ("drink_potion", "potion") for _, a in steps)
    powers = attacks = 0
    for g, a in steps:
        if a.get("action") == "play":
            hand = g.get("hand", [])
            i = a.get("card_index")
            kind = hand[i].get("type") if i is not None and i < len(hand) else None
            powers += kind == "Power"
            attacks += kind == "Attack"
    feats: dict[str, Any] = {"potions": potions, "powers": powers, "attacks": attacks}
    if post is not None:
        import sts2_rs

        p = sts2_rs.state_to_dict(post)["player"]
        feats["block"] = p.get("block")
        feats["energy_left"] = p.get("energy")
    return feats


def rollout_value(post: Any, seeds: list[int]) -> dict[str, Any] | None:
    """Mean raw playout value (deployed playout policy, BeliefDraw view, the
    EndTurn applied) of a pre-EndTurn state: `sts2_rs.values_many`."""
    if post is None or not seeds:
        return None
    import sts2_rs

    if post.get_state().get("type") == "card_select":
        return None
    try:
        vals = sts2_rs.values_many([post] * len(seeds), list(seeds), mode="belief_endturn")
    except ValueError:
        return None
    return {
        "value": sum(v["search_value"] for v in vals) / len(vals),
        "won": sum(bool(v["won"]) for v in vals) / len(vals),
        "hp": sum(v["hp"] for v in vals) / len(vals),
        "n": len(vals),
    }


def human_turn(state: Any, actions: list[dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
    """The human's turn-1 features and the playout value of its pre-EndTurn
    state (stepped on a plain clone, the exact RNG)."""
    st = state.clone()
    steps = []
    trace = []
    post = None
    for a in actions:
        g = st.get_state()
        trace.append(describe(st, a))
        if a.get("action") == "end_turn":
            post = st.clone()
            break
        steps.append((g, a))
        st.apply_step(a)
        if st.terminal_value()[0]:
            break
    g0 = state.get_state()
    return {
        "first": _ident(g0, actions[0]) if actions else None,
        "features": features_of(steps, post),
        "rollout": rollout_value(post, seeds),
        "trace": trace,
        "end": summarize(post),
        "after_enemy": after_enemy(post),
    }


def file_positions(
    path: str,
    wants: list[tuple[str, int]],
    ks: list[int],
    reps: int,
    sims: int,
    n_rollouts: int = 0,
    trace: bool = False,
    rep_start: int = 0,
    first_turn_only: bool = False,
) -> list[dict[str, Any]]:
    """Every requested (combat_id, turn) position of replay `path`, all arms."""
    from ppo.bc_human import replay
    from ppo.turnplan.pro import fight_turns

    out = []
    want = {(c, t) for c, t in wants}
    for rec in replay.fights(path):
        if rec["result"] not in replay.FINISHED or not rec["rng_state"]:
            continue
        mine = sorted(t for c, t in want if c == rec["combat_id"])
        if not mine:
            continue
        try:
            turns, _ = fight_turns(rec)
        except Exception as e:
            out += [
                {"combat_id": rec["combat_id"], "turn": t, "error": repr(e)[:200]} for t in mine
            ]
            continue
        for t in mine:
            row: dict[str, Any] = {
                "file": Path(path).name,
                "combat_id": rec["combat_id"],
                "encounter": rec["encounter"],
                "turn": t,
                "arms": {},
            }
            if t >= len(turns):
                row["error"] = "turn not reached in sync"
                out.append(row)
                continue
            state = turns[t]["state"]
            # Human turns t .. t+k-1, each replayed legally in sync.
            legal = 0
            while t + legal < len(turns) and turns[t + legal]["status"] == "legal":
                legal += 1
            row["legal_turns_ahead"] = legal
            forced_k = {}
            for k in ks:
                if k <= legal:
                    forced_k[k] = [a for tt in turns[t : t + k] for a in tt["actions"]]
            row["human_k_actions"] = {
                str(k): [a.get("action") for a in v] for k, v in forced_k.items()
            }
            seeds = [] if trace else None
            if trace:
                row["pre"] = summarize(state)
            if n_rollouts:
                seeds = [
                    zlib.crc32(f"rollout:{row['file']}:{rec['combat_id']}:{t}:{i}".encode())
                    % (2**31)
                    for i in range(n_rollouts)
                ]
            if seeds is not None and legal:
                try:
                    row["human"] = human_turn(state, turns[t]["actions"], seeds)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except BaseException as e:
                    row["human"] = {"error": repr(e)[:200]}
            for r in range(rep_start, rep_start + reps):
                seed = zlib.crc32(f"{row['file']}:{rec['combat_id']}:{t}:{r}".encode()) % (2**31)
                try:
                    row["arms"].setdefault("P", []).append(
                        play_rest(
                            rec,
                            state,
                            [],
                            sims,
                            seed,
                            rollouts=seeds,
                            first_turn_only=first_turn_only,
                        )
                    )
                    for k, acts in forced_k.items():
                        row["arms"].setdefault(f"H{k}", []).append(
                            play_rest(rec, state, acts, sims, seed)
                        )
                except (KeyboardInterrupt, SystemExit):
                    raise
                except BaseException as e:
                    row.setdefault("errors", []).append(repr(e)[:200])
            out.append(row)
    return out


def _proc(
    path: str,
    wants: list,
    ks: list[int],
    reps: int,
    sims: int,
    n_rollouts: int,
    trace: bool,
    conn: Any,
    rep_start: int = 0,
    env: dict[str, str] | None = None,
    first_turn_only: bool = False,
) -> None:
    from ppo.combat_bench import agents

    agents.apply_env(env or {})
    try:
        res: dict[str, Any] = {
            "rows": file_positions(
                path, wants, ks, reps, sims, n_rollouts, trace, rep_start, first_turn_only
            )
        }
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:
        res = {"crash": repr(e)[:300]}
    conn.send(res)
    conn.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ppo.combat_bench.takeover", description=__doc__)
    ap.add_argument("--collect", required=True, help="a `pro collect` dir (selection.json)")
    ap.add_argument("--keys", nargs="+", required=True, help="JSONL of [file, combat_id, turn]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument(
        "--rep-start",
        type=int,
        default=0,
        help="first rep index (seeds derive from it): fresh reps for a confirmation run",
    )
    ap.add_argument("--sims", type=int, default=4000)
    ap.add_argument(
        "--rollouts",
        type=int,
        default=0,
        help="raw playouts per post-turn state (human turn 1, the pilot's first turn); 0 = off",
    )
    ap.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="K=V",
        help="engine env for the workers (e.g. STS2_RUNAI_HP_WEIGHT=4); all others are scrubbed",
    )
    ap.add_argument(
        "--pilot-only", action="store_true", help="P arm only (whole fight), no human arms"
    )
    ap.add_argument(
        "--first-turn-only",
        action="store_true",
        help="P arm only, stopped at its first EndTurn (with --trace: the turn-1 decision); "
        "no human arms",
    )
    ap.add_argument(
        "--trace",
        action="store_true",
        help="record both turn 1s as text, the boards before / after them and after the enemy turn",
    )
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--file-timeout-s", type=float, default=7200.0)
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)
    env = dict(kv.split("=", 1) for kv in args.env)
    if args.first_turn_only or args.pilot_only:
        args.k = []
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path_of = {
        Path(f["file"]).name: f["file"]
        for f in json.loads((Path(args.collect) / "selection.json").read_text())["files"]
    }
    group_of: dict[tuple[str, str, int], str] = {}
    for kf in args.keys:
        g = Path(kf).stem.removeprefix("keys_")
        for line in Path(kf).open(encoding="utf-8"):
            if line.strip():
                f, c, t = json.loads(line)
                group_of[f, c, int(t)] = g
    done = set()
    pos_path = out / "positions.jsonl"
    if pos_path.exists():
        for line in pos_path.open():
            try:
                r = json.loads(line)
                done.add((r["file"], r["combat_id"], int(r["turn"])))
            except (json.JSONDecodeError, KeyError):
                continue
    # One job per (file, combat): a strong uploader's run holds many positions.
    by_fight: dict[tuple[str, str], list[tuple[str, int]]] = {}
    for (f, c, t), _ in group_of.items():
        if (f, c, t) not in done:
            by_fight.setdefault((f, c), []).append((c, t))
    by_file = {f for f, _ in by_fight}
    todo = iter(sorted(((f, w) for (f, _), w in by_fight.items()), key=lambda kv: -len(kv[1])))
    log = (out / "progress.log").open("a")

    def say(msg: str) -> None:
        line = f"{time.strftime('%F %T')} [takeover] {msg}"
        print(line, file=sys.stderr, flush=True)
        log.write(line + "\n")
        log.flush()

    say(
        f"{len(group_of)} positions, {len(done)} done, {len(by_fight)} fights in {len(by_file)} files to go; k {args.k}, reps {args.reps}, sims {args.sims}"
    )
    # `values_many` runs on the rayon pool: one thread per worker process.
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    ctx = get_context("spawn")
    running: dict[Any, tuple[Any, str, float]] = {}
    n_files = n_rows = 0
    t_last = time.time()
    with pos_path.open("a") as fh:
        while True:
            while len(running) < args.workers:
                nxt = next(todo, None)
                if nxt is None:
                    break
                f, wants = nxt
                recv, send = ctx.Pipe(duplex=False)
                proc = ctx.Process(
                    target=_proc,
                    args=(
                        path_of[f],
                        wants,
                        args.k,
                        args.reps,
                        args.sims,
                        args.rollouts,
                        args.trace,
                        send,
                        args.rep_start,
                        env,
                        args.first_turn_only,
                    ),
                    daemon=True,
                )
                proc.start()
                send.close()
                running[recv] = (proc, f, time.time())
            if not running:
                break
            for conn in wait_conns(list(running), timeout=1.0):
                proc, f, _ = running.pop(conn)
                try:
                    res = conn.recv()
                except EOFError:
                    res = {"crash": f"process died ({proc.exitcode})"}
                conn.close()
                proc.join(timeout=10)
                for r in res.get("rows", []):
                    r["group"] = group_of.get((f, r["combat_id"], int(r["turn"])))
                    fh.write(json.dumps(r) + "\n")
                    n_rows += 1
                if "crash" in res:
                    say(f"{f}: {res['crash']}")
                fh.flush()
                n_files += 1
            now = time.time()
            for conn, (proc, f, t0) in list(running.items()):
                if now - t0 > args.file_timeout_s:
                    proc.kill()
                    proc.join()
                    running.pop(conn)
                    say(f"{f}: watchdog")
            if now - t_last > 60:
                t_last = now
                say(f"{n_files} files, {n_rows} positions done; {len(running)} running")
    say(f"done: {n_files} files, {n_rows} positions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
