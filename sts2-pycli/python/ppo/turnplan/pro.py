"""`python -m ppo.turnplan.pro`: human turns as turn-plan labels? (GH #830 feasibility).

Replays SpireCodex human fights exactly in the engine (`ppo.bc_human.sync`)
and, at every player turn start whose engine state matches the log, lists
the turn's plans (`MctsCombatAgent.turn_plan_labels`, unpriced) and locates
the human's own turn among them (`turn_plan::locate_plan`).

    python -m ppo.turnplan.pro collect --out <dir> [--ascension 10] [--workers 40]
    python -m ppo.turnplan.pro teacher --out <dir> --n 400 [--sims-per-plan 500]

`collect` writes one shard per replay file (`shard-<hash>.jsonl.gz`, one line
per turn start reached in sync) and a ledger line per file (`files.jsonl`:
fights, the first desync of each fight, wall time, or the crash). A file runs
in its own process, so a panic, a segfault or a hang (`--file-timeout-s`)
costs that file only; a rerun skips the files the ledger lists.

A turn start is IN SYNC when every earlier decision of the fight mapped to a
legal engine action and left the engine agreeing with the log (player hp,
each living enemy's hp: `sync.check`), and the engine's hand equals the log's
(the cards retained at the last end of turn plus this turn's turn-start
draws, by id). The human's turn is then mapped decision by decision, stopping
at the first failure:

- `legal`: every decision through `end_turn` (or the combat's end) maps to a
  legal action and the state stays in sync;
- `illegal:<kind>`: a decision did not map (`hand`, `illegal`, `pick`, ...);
- `diverged:<kind>`: an action mapped but the engine then disagreed with the
  log (`player_hp`, `enemy_hp`); the actions through that one are kept.

Either failure ends the fight's replay. The mapped actions are located in the
plan set: `locate.end` says where the human's plan ends (the first
`EndTurn`, RNG-consuming or combat-ending action), `locate.index` which
enumerated plan it is (`None`: not listed), `locate.failure` why it could not
be cut (`incomplete`: the mapped actions stop before the plan ends).

`teacher` re-replays a seeded sample of located, uncapped turn starts and
prices every plan with the G0 teacher (the deployed search forced per plan,
`--sims-per-plan`, BeliefDraw), and records the deployed agent's own first
action (1000 sims, 4000 for bosses) beside the human's.

Uploaders of the #830 `pro_v1` bench set are excluded (the players of
`data/combat_bench/ironclad_pro_v1.jsonl`, `"p_" + sha256(username)[:10]`,
any spelling that case-folds to one of theirs).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import sys
import time
import traceback
from collections import Counter
from collections.abc import Iterator
from multiprocessing import get_context
from multiprocessing.connection import wait as wait_conns
from pathlib import Path
from typing import Any

from ppo.bc_human import replay, sync

ROOT = replay.scraper_dir("replays-spire-codex")
PRO_SET = Path(__file__).resolve().parents[3] / "data" / "combat_bench" / "ironclad_pro_v1.jsonl"
MAX_PLANS = 512
# Fields of a plan kept in a shard (actions only for the human's plan).
PLAN_KEYS = ("end", "features", "cards", "logp_heur", "logp_rl2", "anchors")


# ── corpus selection ─────────────────────────────────────────────────────


def player_id(username: str) -> str:
    """The combat_bench player id of `username` (`SCHEMA.md`)."""
    return "p_" + hashlib.sha256(username.encode()).hexdigest()[:10]


def excluded_usernames(usernames: set[str], pro_ids: set[str]) -> set[str]:
    """Every username in `usernames` whose spelling, or any spelling that
    case-folds like it, hashes to a `pro_ids` player."""
    hit = {u.strip().casefold() for u in usernames if player_id(u) in pro_ids}
    hit |= {u.strip().casefold() for u in usernames if player_id(u.strip()) in pro_ids}
    return {u for u in usernames if u.strip().casefold() in hit}


def room_type(encounter: str) -> str:
    """`boss` / `elite` / `monster` (hallway) from the encounter id's suffix."""
    if encounter.endswith("_BOSS"):
        return "boss"
    if encounter.endswith("_ELITE"):
        return "elite"
    return "monster"


def _header(path: str) -> dict[str, Any] | None:
    try:
        with gzip.open(path, "rt") as fh:
            return json.loads(fh.readline())
    except (json.JSONDecodeError, EOFError, OSError):
        return None


def select_files(
    root: str, character: str, ascensions: set[int], pro_set: Path = PRO_SET
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Eligible replays (v0.111.0, replay_version >= 6, single player, standard,
    no gameplay mod, `ascensions`), strongest uploader first; and a census.

    Strength is the uploader's A10 win rate lower bound (`_players.json`);
    unknown uploaders sort last. Ties break by file name.
    """
    listing = {}
    with open(f"{root}/_listing.jsonl", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            listing[d["run_hash"]] = d
    strength = {}
    for p in json.load(open(f"{root}/_players.json", encoding="utf-8")):
        strength[p["player_token"]] = p.get("a10_win_rate_lo95") or 0.0
    pro_ids = {json.loads(line)["player"] for line in open(pro_set, encoding="utf-8")}
    rows, census = [], Counter()
    for path in sorted(Path(root, character.lower()).glob("*/*.replay.ndjson.gz")):
        h = _header(str(path))
        census["files"] += 1
        if h is None or not sync_eligible(h, character):
            continue
        census["eligible_any_asc"] += 1
        if (h.get("ascension") or 0) not in ascensions:
            continue
        run_hash = path.name.split(".")[0]
        ls = listing.get(run_hash, {})
        rows.append(
            {
                "file": str(path),
                "run_hash": run_hash,
                "username": ls.get("username"),
                "token": ls.get("player_token"),
                "ascension": h.get("ascension"),
                "strength": strength.get(ls.get("player_token"), -1.0),
            }
        )
    names = {r["username"] for r in rows if r["username"]}
    excl = excluded_usernames(names, pro_ids)
    census["pro_v1_players"] = len(pro_ids)
    census["excluded_pro_v1_files"] = sum(r["username"] in excl for r in rows)
    census["no_username_files"] = sum(not r["username"] for r in rows)
    rows = [r for r in rows if r["username"] not in excl]
    rows.sort(key=lambda r: (-r["strength"], r["file"]))
    census["selected"] = len(rows)
    census["uploaders"] = len({r["username"] for r in rows})
    return rows, dict(census)


def sync_eligible(h: dict[str, Any], character: str) -> bool:
    """The header filter of `bc_human.sync_census.eligible`."""
    return (
        h.get("t") == "header"
        and h.get("build_id") == sync.GAME_VERSION
        and (h.get("replay_version") or 0) >= 6
        and (h.get("player_count") or 1) == 1
        and h.get("game_mode") == "standard"
        and sync.mods_ok(h)
        and h.get("character") == character.upper()
    )


# ── the log's hand ───────────────────────────────────────────────────────


class HandLog:
    """The hand the log says the player starts each turn with, by card id:
    the cards retained at the last end of turn, the turn-start draws, and
    the cards generated between the player's `turn` event and the turn's
    first decision (start-of-turn relics such as Vexing Puzzlebox put them
    in the hand; the log names no pile)."""

    def __init__(self) -> None:
        self.ids: dict[int, str] = {}
        self.retained: list[int] = []
        self.draws: list[tuple[int | None, str]] = []  # (card c, id)
        self.in_start = False  # after the player's `turn` event
        self.started = False  # the turn's first decision was made
        # Cards generated since the player's last end of turn outside the
        # start window (enemy Burns, Dazed, Infections): the log names no
        # pile, so `hand_matches` lets the engine hold them in hand or not.
        self.generated: Counter[str] = Counter()

    def feed(self, d: dict[str, Any]) -> None:
        t = d["t"]
        if t in ("draw", "generate") and "c" in d:
            self.ids[d["c"]] = d["id"]
        if t == "draw":
            # Every draw before the turn's first decision: the turn-start
            # draws and start-of-combat / start-of-turn relic draws.
            if d.get("turn_start") or not self.started:
                self.draws.append((d.get("c"), d["id"]))
        elif t == "generate" and self.in_start:
            self.draws.append((d.get("c"), d["id"]))
        elif t == "generate" and not self.started:
            self.generated[sync._norm(d["id"])] += 1
        elif t == "exhaust" and not self.started:
            # A start-of-turn choice that exhausts from the hand (Toasty Mittens).
            self.draws = [x for x in self.draws if x[0] is None or x[0] != d.get("c")]
        elif t == "turn" and d.get("side") == "player":
            self.in_start = True
        elif t == "flush":
            self.retained = list(d.get("retained_c") or [])
        elif t == "end_turn" and d.get("side") == "player":
            self.draws = []
            self.started = False
            self.generated = Counter()

    def decision(self) -> None:
        """The turn's first decision closes its start window."""
        self.in_start = False
        self.started = True

    def start_hand(self) -> Counter[str]:
        kept = [self.ids.get(c, f"?{c}") for c in self.retained]
        return Counter(sync._norm(i) for i in [*kept, *(i for _, i in self.draws)])


def hand_matches(engine: Counter[str], log: HandLog) -> bool:
    """The engine's turn-start hand equals the log's, up to cards generated
    off-window whose pile the log does not name (`HandLog.generated`)."""
    want = log.start_hand()
    return not (want - engine) and not ((engine - want) - log.generated)


def engine_hand(st: Any) -> Counter[str]:
    return Counter(sync._norm(c["id"]) for c in st.get_state().get("hand", []))


# ── one fight ────────────────────────────────────────────────────────────


def fight_turns(rec: dict[str, Any]) -> tuple[list[dict[str, Any]], tuple[str, str] | None]:
    """The fight's turn starts reached in sync, and its first desync.

    Each turn: `{"turn", "state" (a clone at the turn start), "actions" (the
    mapped engine actions), "n_dec" (the log's decisions this turn), "status"}`.
    """
    turns: list[dict[str, Any]] = []
    try:
        st = sync.build(rec, monster_ai_offset=len(rec["enemies"]))
    except (ValueError, RuntimeError) as e:
        return turns, ("build", str(e)[:200])
    view = sync._LogView(rec)
    hand = HandLog()
    for d in rec["pre"]:
        view.feed(d)
        hand.feed(d)
    decisions = rec["decisions"]
    # The log's decisions per player turn (each turn closes at its end_turn).
    per_turn: list[int] = [0]
    for dec in decisions:
        per_turn[-1] += 1
        if dec["ev"]["t"] == "end_turn":
            per_turn.append(0)
    cur: dict[str, Any] | None = None
    checked = False  # this turn's start hand was compared with the log's
    try:
        sync.check(st, view)
        for k, dec in enumerate(decisions):
            ev = dec["ev"]
            choosing = st.get_state().get("type") == "card_select"
            if cur is None and not checked and not choosing:
                checked = True
                if not hand_matches(engine_hand(st), hand):
                    raise sync.Desync(
                        "hand_start",
                        f"engine {sorted(engine_hand(st).items())} "
                        f"log {sorted(hand.start_hand().items())}",
                    )
                hand.decision()
            if cur is None and choosing:
                # A start-of-turn choice (Toasty Mittens, ...): replayed and
                # checked like any decision, but the turn's plan starts after it.
                if ev["t"] != "pick":
                    raise sync.Desync("choice_at_turn_start", ev["t"])
                st.apply_step(sync._action(st, view, ev))
                for d in dec["after"]:
                    view.feed(d)
                    hand.feed(d)
                sync.check(st, view)
                continue
            if cur is None:
                cur = {
                    "turn": len(turns),
                    "state": st.clone(),
                    "actions": [],
                    "n_dec": per_turn[len(turns)],
                    "status": None,
                }
                turns.append(cur)
            try:
                a = sync._action(st, view, ev)
            except sync.Desync as e:
                cur["status"] = f"illegal:{e.kind}"
                raise
            cur["actions"].append(a)
            r = st.apply_step(a)
            if ev["t"] == "end_turn":
                hand.feed(ev)  # before its `after`: the next turn's draws
            for d in dec["after"]:
                view.feed(d)
                hand.feed(d)
            over = r in ("combat_won", "combat_lost") or st.terminal_value()[0]
            if over:
                try:
                    sync.check_end(decisions, k, r)
                except sync.Desync as e:
                    cur["status"] = f"diverged:{e.kind}"
                    raise
            else:
                try:
                    sync.check(st, view)
                except sync.Desync as e:
                    cur["status"] = f"diverged:{e.kind}"
                    raise
            if ev["t"] == "end_turn" or over:
                cur["status"] = "legal"
                cur = None
                checked = False
            if over:
                break
    except sync.Desync as e:
        return turns, (e.kind, e.detail[:2000])
    if cur is not None and cur["status"] is None:
        cur["status"] = "log_ended"
    return turns, None


# ── collect ──────────────────────────────────────────────────────────────


def deployed_spec(env: dict[str, str] | None = None) -> dict[str, Any]:
    """The G0 pilot / teacher spec (1000 sims, 4000 boss, env `{}` unless
    `env` names engine seams), its env applied.

    Call before any `sts2_rs` search: some engine seams latch on first use.
    """
    from ppo.combat_bench import agents

    spec = agents.mcts_spec(1000, 4000, agents.build_env(dict(env or {})))
    agents.apply_env(spec["env"])
    return spec


def _rm(state: Any, rec: dict[str, Any]) -> Any:
    import sts2_rs

    rm = sts2_rs.RunManager(
        seed=0, character=rec["character"].capitalize(), ascension_level=rec["ascension"] or 0
    )
    rm.debug_enter_combat(state.clone(), room_type(rec["encounter"]))
    return rm


def locate_form(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """`actions` with each multi-pick `selection` sorted: `legal_actions`
    lists a multi-card choice once, in ascending order, while the log keeps
    the order the human clicked the cards in."""
    out = []
    for a in actions:
        if a.get("action") == "choose" and a.get("selection"):
            sel = sorted(a["selection"])
            a = {**a, "selection": sel, "index": sel[0]}
        out.append(a)
    return out


def turn_record(rec: dict[str, Any], t: dict[str, Any], labels: dict[str, Any]) -> dict[str, Any]:
    """One shard line: the turn, its unpriced plan set and the human's plan."""
    loc = labels["locate"]
    idx = loc["index"]
    plans = [{k: p[k] for k in PLAN_KEYS} for p in labels["plans"]]
    if idx is not None:
        plans[idx]["actions"] = labels["plans"][idx]["actions"]
    return {
        "file": Path(rec["file"]).name,
        "combat_id": rec["combat_id"],
        "encounter": rec["encounter"],
        "room_type": room_type(rec["encounter"]),
        "floor": rec["floor"],
        "act": rec["act"],
        "ascension": rec["ascension"],
        "turn": t["turn"],
        "status": t["status"],
        "n_dec": t["n_dec"],
        "n_mapped": len(t["actions"]),
        "human_actions": t["actions"],
        "locate": loc,
        "cap_hit": labels["cap_hit"],
        "n_plans": labels["n_plans"],
        "deck": labels["deck"],
        "plans": plans,
    }


def collect_file(path: str, emit: Any) -> dict[str, Any]:
    """Replay every finished fight of `path`; `emit` each in-sync turn start."""
    from ppo.combat_bench import agents

    agent = agents.make_agent(deployed_spec(), 0)
    fights = turns_n = located = 0
    desyncs: Counter[str] = Counter()
    # Player turns the log holds (each fight: its end_turns, plus a last
    # turn cut by the combat's end), by room: the denominator of sync.
    log_turns: Counter[str] = Counter()
    errors: list[str] = []
    for rec in replay.fights(path):
        if rec["result"] not in replay.FINISHED or not rec["rng_state"]:
            continue
        fights += 1
        evs = [d["ev"]["t"] for d in rec["decisions"]]
        n_turns = evs.count("end_turn") + bool(evs and evs[-1] != "end_turn")
        log_turns[room_type(rec["encounter"])] += n_turns
        try:
            turns, desync = fight_turns(rec)
        except Exception as e:
            errors.append(f"{rec['combat_id']}: {e!r}"[:300])
            continue
        desyncs[desync[0] if desync else "SYNC"] += 1
        for t in turns:
            try:
                labels = agent.turn_plan_labels(
                    _rm(t["state"], rec), 0, MAX_PLANS, locate=locate_form(t["actions"])
                )
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as e:  # a PyO3 panic is a BaseException
                errors.append(f"{rec['combat_id']} turn {t['turn']}: {e!r}"[:300])
                continue
            if labels is None:
                errors.append(f"{rec['combat_id']} turn {t['turn']}: no labels")
                continue
            emit(turn_record(rec, t, labels))
            turns_n += 1
            located += labels["locate"]["index"] is not None
    return {
        "fights": fights,
        "turns": turns_n,
        "located": located,
        "desync": dict(desyncs),
        "log_turns": dict(log_turns),
        "errors": errors,
    }


def _file_process(path: str, shard: str, conn: Any) -> None:
    gz: Any = None

    def emit(row: dict[str, Any]) -> None:
        nonlocal gz
        if gz is None:
            gz = gzip.open(shard, "wt")
        gz.write(json.dumps(row) + "\n")
        gz.flush()

    try:
        out = collect_file(path, emit)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:
        out = {"crash": repr(e)[:300], "tb": traceback.format_exc()[-1500:]}
    finally:
        if gz is not None:
            gz.close()
    conn.send(out)
    conn.close()


def collect(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, census = select_files(args.root, args.character, set(args.ascension))
    if args.limit:
        rows = rows[: args.limit]
    (out / "selection.json").write_text(
        json.dumps({"args": vars(args), "census": census, "files": rows}, indent=1) + "\n"
    )
    ledger = out / "files.jsonl"
    done = set()
    if ledger.exists():
        for line in ledger.open():
            try:
                done.add(json.loads(line)["file"])
            except (json.JSONDecodeError, KeyError):
                continue
    todo = [r for r in rows if Path(r["file"]).name not in done]
    log = (out / "progress.log").open("a")

    def say(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [turnplan.pro] {msg}"
        print(line, file=sys.stderr, flush=True)
        log.write(line + "\n")
        log.flush()

    say(f"census {census}; {len(rows)} files, {len(done)} done, {len(todo)} to go")
    ctx = get_context("spawn")
    running: dict[Any, tuple[Any, dict[str, Any], float]] = {}
    it = iter(todo)
    tot: Counter[str] = Counter()
    t0 = t_last = time.time()
    with ledger.open("a") as led:

        def finish(conn: Any, res: dict[str, Any]) -> None:
            proc, row, start = running.pop(conn)
            conn.close()
            proc.join(timeout=10)
            line = {"file": Path(row["file"]).name}
            line |= {"strength": row["strength"], "wall_s": round(time.time() - start, 1), **res}
            led.write(json.dumps(line) + "\n")
            led.flush()
            tot["files"] += 1
            for k in ("fights", "turns", "located"):
                tot[k] += res.get(k, 0)
            tot["crash"] += "crash" in res

        while True:
            while len(running) < args.workers:
                row = next(it, None)
                if row is None:
                    break
                shard = out / f"shard-{row['run_hash']}.jsonl.gz"
                recv, send = ctx.Pipe(duplex=False)
                proc = ctx.Process(target=_file_process, args=(row["file"], str(shard), send))
                proc.daemon = True
                proc.start()
                send.close()
                running[recv] = (proc, row, time.time())
            if not running:
                break
            for conn in wait_conns(list(running), timeout=1.0):
                try:
                    res = conn.recv()
                except EOFError:
                    code = running[conn][0].exitcode
                    res = {"crash": f"process died ({code})"}
                finish(conn, res)
            now = time.time()
            for conn, (proc, _, start) in list(running.items()):
                if now - start > args.file_timeout_s:
                    proc.kill()
                    proc.join()
                    finish(conn, {"crash": f"watchdog {args.file_timeout_s:g} s"})
            if now - t_last > 60:
                t_last = now
                say(f"{(now - t0) / 60:.1f} min: {dict(tot)}, {len(running)} running")
    say(f"done: {dict(tot)}")
    return 0


# ── reading ──────────────────────────────────────────────────────────────


def iter_turns(out: str | Path) -> Iterator[dict[str, Any]]:
    """Every turn line of every shard in `out`, stopping cleanly at a cut stream."""
    for shard in sorted(Path(out).glob("shard-*.jsonl.gz")):
        with gzip.open(shard, "rt") as fh:
            try:
                for line in fh:
                    if line.endswith("\n"):
                        yield json.loads(line)
            except (EOFError, OSError, json.JSONDecodeError):
                continue


def as_label(row: dict[str, Any]) -> dict[str, Any]:
    """`row` as a `turnplan.data.to_position` record whose teacher-best plan is
    the human's: `q` 1 for it, 0 for every other plan (so `recall.hit` reads
    "the kept set holds the human's plan")."""
    idx = row["locate"]["index"]
    plans = [
        {**p, "q": float(i == idx), "q_var": 0.0, "visits": 0} for i, p in enumerate(row["plans"])
    ]
    key = f"{row['file']}:{row['combat_id']}:{row['turn']}"
    return {**row, "key": key, "rep": 0, "plans": plans}


# ── teacher ──────────────────────────────────────────────────────────────


def _identity(state: dict[str, Any], a: dict[str, Any]) -> tuple[Any, ...]:
    """A combat action by what it does: the card (id, upgraded) and target
    for a play, the potion and target for a drink."""
    kind = a.get("action")
    if kind == "play":
        c = state["hand"][a["card_index"]]
        return ("play", sync._norm(c["id"]), bool(c.get("upgraded")), a.get("target_index"))
    if kind in ("drink_potion", "potion"):
        return ("potion", a.get("slot"), a.get("target_index"))
    return (kind,)


def teacher(args: argparse.Namespace) -> int:
    out = Path(args.out)
    users = {
        Path(r["file"]).name: r.get("username")
        for r in json.loads((out / "selection.json").read_text())["files"]
    }
    players = set(args.players.split(",")) if args.players else None
    skip = set(args.exclude_players.split(",")) if args.exclude_players else set()
    keys = None
    if args.keys:
        keys = {
            tuple(json.loads(line))
            for line in Path(args.keys).open(encoding="utf-8")
            if line.strip()
        }
    rows = [
        r
        for r in iter_turns(out)
        if r["locate"]["index"] is not None
        and not r["cap_hit"]
        and r["n_plans"] > 1
        and (not args.encounters or r["encounter"] in args.encounters.split(","))
        and (args.room is None or r["room_type"] == args.room)
        and (players is None or users.get(r["file"]) in players)
        and users.get(r["file"]) not in skip
        and (keys is None or (r["file"], r["combat_id"], r["turn"]) in keys)
    ]
    rng = random.Random(args.seed)
    pick = rng.sample(rows, min(args.n, len(rows)))
    want: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    for r in pick:
        want.setdefault(r["file"], {})[r["combat_id"], r["turn"]] = r
    path_of = {
        Path(r["file"]).name: r["file"]
        for r in json.loads((out / "selection.json").read_text())["files"]
    }
    tdir = out / args.tdir
    tdir.mkdir(exist_ok=True)
    jobs = [
        (
            path_of[f],
            wants,
            args.sims_per_plan,
            str(tdir),
            tuple(args.pilot_sims),
            agents_env(args.env),
        )
        for f, wants in sorted(want.items())
    ]
    ctx = get_context("spawn")
    with ctx.Pool(args.workers, maxtasksperchild=1) as pool:
        for i, res in enumerate(pool.imap_unordered(_teacher_file, jobs), 1):
            print(f"[turnplan.pro teacher] {i}/{len(jobs)} files: {res}", file=sys.stderr)
    return 0


def agents_env(items: list[str] | None) -> dict[str, str]:
    """`--env K=V` pairs (engine seams for the teacher's evaluator)."""
    from ppo.combat_bench import agents

    return agents.parse_kv(items or [], "--env")


def _teacher_file(
    job: tuple[
        str, dict[tuple[str, int], dict[str, Any]], int, str, tuple[int, ...], dict[str, str]
    ],
) -> str:
    path, wants, sims_per_plan, tdir, pilot_sims, env = job
    from ppo.combat_bench import agents

    spec = deployed_spec(env)
    shard = Path(tdir) / f"{Path(path).name.split('.')[0]}.jsonl"
    done = n_err = 0
    with shard.open("w") as fh:
        for rec in replay.fights(path):
            keys = [k for k in wants if k[0] == rec["combat_id"]]
            if not keys or rec["result"] not in replay.FINISHED:
                continue
            try:
                turns, _ = fight_turns(rec)
            except Exception:
                n_err += 1
                continue
            for t in turns:
                row = wants.get((rec["combat_id"], t["turn"]))
                if row is None:
                    continue
                key = f"{row['file']}:{rec['combat_id']}:{t['turn']}"
                seed = int(hashlib.sha256(key.encode()).hexdigest()[:7], 16)
                try:
                    tutor = agents.make_agent(spec, seed)
                    t0 = time.perf_counter()
                    lab = tutor.turn_plan_labels(
                        _rm(t["state"], rec),
                        sims_per_plan,
                        MAX_PLANS,
                        locate=locate_form(t["actions"]),
                    )
                    label_s = time.perf_counter() - t0
                    view = sts2_view(_rm(t["state"], rec))
                    picks: dict[str, Any] = {}
                    for sims in pilot_sims:
                        pilot = agents.make_agent(
                            agents.mcts_spec(sims, 4 * sims, spec["env"]), seed
                        )
                        step = pilot.step(_rm(t["state"], rec))
                        search = (step or {}).get("combat_search") or {}
                        picks[str(sims)] = (
                            _identity(view, search["recommended"])
                            if search.get("recommended")
                            else None
                        )
                except (KeyboardInterrupt, SystemExit):
                    raise
                except BaseException as e:
                    err = {"key": [rec["combat_id"], t["turn"]], "error": repr(e)}
                    fh.write(json.dumps(err) + "\n")
                    n_err += 1
                    continue
                rec_out = {
                    "file": row["file"],
                    "combat_id": rec["combat_id"],
                    "turn": t["turn"],
                    "room_type": row["room_type"],
                    "label_s": round(label_s, 2),
                    "locate": lab["locate"],
                    "located_before": row["locate"]["index"],
                    "cap_hit": lab["cap_hit"],
                    "n_plans": lab["n_plans"],
                    "q": [p["q"] for p in lab["plans"]],
                    "q_var": [p["q_var"] for p in lab["plans"]],
                    "visits": [p["visits"] for p in lab["plans"]],
                    "plan_len": [len(p["actions"]) for p in lab["plans"]],
                    "human_first": _identity(view, t["actions"][0]) if t["actions"] else None,
                    "agent_first": picks.get(str(pilot_sims[0])) if pilot_sims else None,
                    "teacher_env": env,
                    "sims_per_plan": sims_per_plan,
                    "agent_first_by_sims": picks,
                    "encounter": row.get("encounter"),
                    "plan_features": [p["features"] for p in lab["plans"]],
                    "plan_cards": [p["cards"] for p in lab["plans"]],
                    "plan_end": [p["end"] for p in lab["plans"]],
                    "plan_first": [_identity(view, p["actions"][0]) for p in lab["plans"]],
                    "hand": [c["id"] for c in view.get("hand", [])],
                }
                fh.write(json.dumps(rec_out) + "\n")
                fh.flush()
                done += 1
    return f"{Path(path).name}: {done} positions, {n_err} errors"


def sts2_view(rm: Any) -> dict[str, Any]:
    """The combat state dict of `rm` (`hand` with ids, for `_identity`)."""
    return rm.combat_state().get_state()


# ── CLI ──────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m ppo.turnplan.pro",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--out", required=True)
    c.add_argument("--root", default=ROOT)
    c.add_argument("--character", default="IRONCLAD")
    c.add_argument("--ascension", type=int, nargs="+", default=[10])
    c.add_argument("--limit", type=int, default=None, help="only the N strongest-uploader files")
    c.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    c.add_argument("--file-timeout-s", type=float, default=1800.0)
    t = sub.add_parser("teacher")
    t.add_argument("--out", required=True, help="a collect dir")
    t.add_argument("--n", type=int, default=400)
    t.add_argument("--sims-per-plan", type=int, default=500)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    t.add_argument("--encounters", default=None, help="comma list: only these encounters")
    t.add_argument("--pilot-sims", type=int, nargs="*", default=[1000], help="pilot picks")
    t.add_argument(
        "--env", action="append", default=[], help="K=V engine seam for the teacher (repeatable)"
    )
    t.add_argument("--tdir", default="teacher", help="output subdir of --out")
    t.add_argument("--room", default=None, help="only this room type")
    t.add_argument(
        "--keys", default=None, help="JSONL of [file, combat_id, turn]: only these positions"
    )
    t.add_argument("--players", default=None, help="comma list of uploaders to keep")
    t.add_argument("--exclude-players", default=None, help="comma list of uploaders to drop")
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)
    return collect(args) if args.cmd == "collect" else teacher(args)


if __name__ == "__main__":
    sys.exit(main())
