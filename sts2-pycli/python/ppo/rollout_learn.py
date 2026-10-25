"""Learn the MCTS playout policy from the search's own decisions.

Every combat simulation scores its tree path by ONE playout to the end of the
fight, so the playout policy judges every candidate move. The deployed one is
a hand-written card-priority list (`sts2_mcts::expert`); in long boss fights
its playouts almost never win, so the search gets no win signal (GH #777
human-bank audit). This module replaces it with a policy learned from the
search itself (expert iteration):

    gen    play human-bank fights with the pilot; at every searched decision
           record the playout features of each legal action
           (`sts2_rs.rollout_features`, computed by the same Rust code the
           playout runs) and the root's visit counts.
    train  fit the factorized softmax of `sts2_mcts::rollout_net` to the
           visit distributions and write its weights as JSON.
    traj   record the deployed pilot's own trajectories on bank fights, so
           `rl --starts mid` can start playouts at any decision it reached
           (`ppo.rollout_starts`).
    rl     REINFORCE on playout outcomes (see `rl`'s docstring).

Honest-mode searches already play the built-in nets
(`sts2-rs/data/rollout_nets/<character>.json`, `RolloutPolicyKind::Net`;
every character). Every command plays each fight as its row's `character`
(`pilot_human_decks extract --character`), so one bank trains one character's
net. To try new weights, set `STS2_ROLLOUT_NET=<json>`; it replaces the built-in net for
every character. `STS2_ROLLOUT_TEMP` overrides that net's softmax temperature
(never the built-in net's), and `STS2_RUNAI_MCTS_OVERRIDES=rollout_policy=heuristic` restores the hand-written
playout.

Feature versions (`sts2_mcts::rollout_net` module docs): v1 is the deployed
net's 32 state / 24 action features; v2 (96 / 32) adds the boss flag, the act,
enemy intents and high-impact powers on both sides, and an optional
per-species encounter embedding; v3 ("v1e", 32 / 24) is v1's layout with every
telegraphed attack read at its displayed number (Strength, Weak, Vulnerable,
back attacks) and a form-aware "enemy HP removed" (GH #830); v4 (44 / 29) is
v1 read as v1 reads it plus a character and affliction block: Regent's Stars
and Star powers, Tainted, Personal Hive, Vital Spark, card afflictions and
each card's Star cost (GH #822 follow-up).
`gen --feature-version N` records version-N data, `train` follows the data's
version, and `rl --feature-version 2|3|4` upgrades a v1 `--init` net: to v2
and v4 by zero-padding (the same policy at iteration 0), to v3 by relabeling
(the same weights, read on the v3 inputs).

Training rows exclude every run in `--holdout` (fight files whose runs are
used to evaluate), so evaluation decks are never trained on. `gen`, `traj` and
`rl` also drop every combat-bench run (`ppo.combat_bench.holdout`, GH #830)
unless `--allow-bench-overlap`; what they dropped or kept is recorded in the
net's `meta.bench_holdout` (`gen` writes it to `<out-dir>/bench_holdout.json`,
which `train` copies into the net).
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import multiprocessing as mp
import os
import random
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

from ppo import rollout_starts
from ppo.combat_bench import agents
from ppo.combat_bench.holdout import subtract_bench_runs
from ppo.pilot_human_decks import _fight_seed, fight_character

BENCH_META = "bench_holdout.json"
FEATURES_META = "features.json"

ROOMS = {"monster": 0, "elite": 1, "boss": 2}
# (state_dim, action_dim) per feature version; `sts2_rs.LearnedPlayout`'s
# `state_dim` / `action_dim` are the source of truth (a test pins the match).
# v3 reuses v1's layout (its inputs are read differently, not laid out
# differently), so only v2 carries the act tables and the encounter embedding.
FEATURE_DIMS = {1: (32, 24), 2: (96, 32), 3: (32, 24), 4: (44, 29)}
TABLE_VERSION = 2  # the one version with per-species tables
STATE_DIM, ACTION_DIM = FEATURE_DIMS[1]
MONSTER_SLOTS = 5
HEUR_FEATURE = 21  # -priority / tau, the heuristic playout's own logit
# `gen` gives up on its pool this long after `--fight-timeout-s` with no fight
# finishing: a search step overran the per-fight deadline it cannot interrupt.
STALL_GRACE_S = 600


def _canon(action: dict[str, Any]) -> str:
    return json.dumps(action, sort_keys=True)


# ── gen ──────────────────────────────────────────────────────────────────


def record_decisions(
    rm: Any,
    agent: Any,
    version: int,
    room: int,
    fight_idx: int,
    out: dict[str, Any],
    deadline: float = math.inf,
) -> None:
    """Let `agent` play `rm`'s combat to the end and append one row per
    searched decision with >= 2 legal actions to `out["decisions"]`: the
    playout features of every legal action (`sts2_rs.rollout_features`), the
    root visits, the recommended action and the root win share. Past the
    `time.monotonic()` `deadline` it stops after the current step and sets
    `out["timeout"]`."""
    import sts2_rs

    steps = 0
    while rm.phase() == "combat" and not rm.is_over() and steps < 3000:
        if "timeout" in out:
            break
        feats = sts2_rs.rollout_features(rm.combat_state(), version=version)
        rec = agent.step(rm)
        steps += 1
        if time.monotonic() > deadline:
            out["timeout"] = True
        if rec is None:
            break
        search = rec.get("combat_search")
        if not search or len(feats["actions"]) < 2:
            continue
        index = {_canon(a): i for i, a in enumerate(feats["actions"])}
        visits = [0.0] * len(feats["actions"])
        wins = total = 0
        for a in search["actions"]:
            i = index.get(_canon(a["action"]))
            if i is None:
                out["unmatched"] += 1
                continue
            visits[i] += a["visits"]
            wins += a["wins"]
            total += a["visits"]
        if not total:
            continue
        chosen = index.get(_canon(search.get("recommended") or {}), -1)
        row = {
            "state": feats["state"],
            "keys": feats["keys"],
            "x": feats["x"],
            "visits": visits,
            "chosen": chosen,
            "pwin": wins / total,
            "room": room,
            "fight": fight_idx,
        }
        if version == TABLE_VERSION:
            row["monsters"] = feats["monsters"]
        out["decisions"].append(row)


def gen_fight(job: tuple[Any, ...]) -> dict[str, Any]:
    fight, fight_idx, reps, rep0, sims, boss_sims, clair, version, timeout_s = job
    import sts2_rs

    if clair:
        os.environ["STS2_RUNAI_HONEST"] = "0"
    key = f"{fight['file']}:{fight['floor']}"
    out: dict[str, Any] = {"key": key, "decisions": [], "unmatched": 0}
    # Wall-clock budget for all reps of this fight (0 = none), checked after
    # each search step; the decisions recorded before it ran out are kept.
    deadline = time.monotonic() + timeout_s if timeout_s > 0 else math.inf
    try:
        for rep in range(reps):
            s = _fight_seed(key) * 100 + rep0 + rep
            state = sts2_rs.build_scenario_state(
                s,
                fight_character(fight),
                fight["deck"],
                int(fight["hp_before"]),
                int(fight["max_hp"]),
                fight["relics"],
                fight["encounter"],
                ascension=int(fight["ascension"]),
                potions=fight["potions"][:3],
            )
            rm = sts2_rs.RunManager(
                seed=s, character=fight_character(fight), ascension_level=int(fight["ascension"])
            )
            rm.debug_enter_combat(state, fight["room_type"])
            agent = sts2_rs.MctsCombatAgent(seed=s, sims=sims, boss_sims=boss_sims)
            record_decisions(
                rm, agent, version, ROOMS.get(fight["room_type"], 0), fight_idx, out, deadline
            )
            if "timeout" in out:
                break
    except Exception as e:
        out["error"] = repr(e)
    return out


class ShardWriter:
    """Accumulates decisions and writes `.npz` shards; keys become vocab ids.

    v2 data also carries each decision's encounter species (`mon`, ids into
    `monster_vocab.json`, 0 = no enemy in that slot).
    """

    def __init__(self, out_dir: str, per_shard: int) -> None:
        self.out_dir = out_dir
        self.per_shard = per_shard
        os.makedirs(out_dir, exist_ok=True)
        vocab_path = os.path.join(out_dir, "vocab.json")
        self.vocab: dict[str, int] = (
            json.load(open(vocab_path, encoding="utf-8"))
            if os.path.exists(vocab_path)
            else {"<unk>": 0}
        )
        mvocab_path = os.path.join(out_dir, "monster_vocab.json")
        self.mvocab: dict[str, int] = (
            json.load(open(mvocab_path, encoding="utf-8"))
            if os.path.exists(mvocab_path)
            else {"<none>": 0}
        )
        self.n_shard = len([f for f in os.listdir(out_dir) if f.startswith("shard-")])
        self.buf: list[dict[str, Any]] = []

    def add(self, decisions: list[dict[str, Any]]) -> None:
        self.buf.extend(decisions)
        if len(self.buf) >= self.per_shard:
            self.flush()

    def flush(self) -> None:
        if not self.buf:
            return
        d = self.buf
        ids = [self.vocab.setdefault(k, len(self.vocab)) for r in d for k in r["keys"]]
        arrays = {
            "state": np.asarray([r["state"] for r in d], np.float32),
            "n_act": np.asarray([len(r["keys"]) for r in d], np.int16),
            "x": np.asarray([x for r in d for x in r["x"]], np.float32),
            "key": np.asarray(ids, np.int32),
            "visits": np.asarray([v for r in d for v in r["visits"]], np.float32),
            "chosen": np.asarray([r["chosen"] for r in d], np.int16),
            "pwin": np.asarray([r["pwin"] for r in d], np.float32),
            "room": np.asarray([r["room"] for r in d], np.int8),
            "fight": np.asarray([r["fight"] for r in d], np.int32),
        }
        if "monsters" in d[0]:
            arrays["mon"] = np.asarray(
                [
                    [
                        0 if m is None else self.mvocab.setdefault(m, len(self.mvocab))
                        for m in r["monsters"]
                    ]
                    for r in d
                ],
                np.int32,
            )
            json.dump(
                self.mvocab,
                open(os.path.join(self.out_dir, "monster_vocab.json"), "w", encoding="utf-8"),
            )
        np.savez_compressed(os.path.join(self.out_dir, f"shard-{self.n_shard:04d}.npz"), **arrays)
        json.dump(self.vocab, open(os.path.join(self.out_dir, "vocab.json"), "w", encoding="utf-8"))
        self.n_shard += 1
        self.buf = []


def gen(args: argparse.Namespace) -> int:
    fights = [json.loads(line) for line in open(args.fights, encoding="utf-8")]
    holdout: set[str] = set()
    for path in args.holdout or []:
        holdout |= {json.loads(line)["file"] for line in open(path, encoding="utf-8")}
    if args.holdout_encounter:
        holdout |= {f["file"] for f in fights if f["encounter"] in args.holdout_encounter}
    rooms = set(args.rooms.split(","))
    pool = [
        (i, f) for i, f in enumerate(fights) if f["file"] not in holdout and f["room_type"] in rooms
    ]
    pool, bench_meta = subtract_bench_runs(
        pool, lambda p: p[1]["file"], args.allow_bench_overlap, tag="[rollout_learn]"
    )
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, BENCH_META), "w", encoding="utf-8") as f:
        json.dump(bench_meta, f)
    features: dict[str, Any] = {"feature_version": args.feature_version}
    if args.feature_version == TABLE_VERSION:
        # `gen_fight` computes the features with the engine's default tables,
        # which are exactly these.
        features.update(feature_tables())
    feat_path = os.path.join(args.out_dir, FEATURES_META)
    if os.path.exists(feat_path) and json.load(open(feat_path, encoding="utf-8")) != features:
        raise SystemExit(f"{feat_path} records other features; use a new --out-dir")
    json.dump(features, open(feat_path, "w", encoding="utf-8"))
    rng = random.Random(args.sample_seed)
    if args.max_monster:
        mon = [p for p in pool if p[1]["room_type"] == "monster"]
        keep = {i for i, _ in rng.sample(mon, min(args.max_monster, len(mon)))}
        pool = [p for p in pool if p[1]["room_type"] != "monster" or p[0] in keep]
    rng.shuffle(pool)
    done_path = os.path.join(args.out_dir, "done.txt")
    done = (
        set(open(done_path, encoding="utf-8").read().split())
        if os.path.exists(done_path)
        else set()
    )
    jobs = [
        (
            f,
            i,
            args.reps,
            args.rep0,
            args.sims,
            args.boss_sims,
            args.clair,
            args.feature_version,
            args.fight_timeout_s,
        )
        for i, f in pool
        if f"{f['file']}:{f['floor']}" not in done
    ]
    print(
        f"[rollout_learn] {len(jobs)} fights to play ({len(holdout)} held-out runs, "
        f"rooms {sorted(rooms)}, feature version {args.feature_version})",
        flush=True,
    )
    writer = ShardWriter(args.out_dir, args.per_shard)
    t0 = time.time()
    n = n_dec = err = timeouts = unmatched = 0
    pending = {f"{job[0]['file']}:{job[0]['floor']}" for job in jobs}
    # A search step that never returns escapes the per-fight deadline; the
    # parent then stops waiting, keeps what it has and names the stuck fights.
    stall_s = args.fight_timeout_s + STALL_GRACE_S if args.fight_timeout_s > 0 else None
    with (
        get_context("spawn").Pool(args.procs) as pool_,
        open(done_path, "a", encoding="utf-8") as done_f,
    ):
        results = pool_.imap_unordered(gen_fight, jobs)
        while pending:
            try:
                row = results.next(stall_s)
            except mp.TimeoutError:
                writer.flush()
                print(
                    f"[rollout_learn] STALLED: no fight finished in {stall_s:.0f}s; "
                    f"{len(pending)} unfinished: {sorted(pending)[:20]}",
                    flush=True,
                )
                return 1
            pending.discard(row["key"])
            n += 1
            err += "error" in row
            timeouts += "timeout" in row
            unmatched += row["unmatched"]
            n_dec += len(row["decisions"])
            writer.add(row["decisions"])
            done_f.write(row["key"] + "\n")
            done_f.flush()
            if n % 200 == 0 or n == len(jobs):
                print(
                    f"[rollout_learn] {n}/{len(jobs)} fights, {n_dec} decisions, {err} errors, "
                    f"{timeouts} timeouts, "
                    f"{unmatched} unmatched root actions, {time.time() - t0:.0f}s",
                    flush=True,
                )
    writer.flush()
    print("DONE", flush=True)
    return 0


# ── gen-replay ───────────────────────────────────────────────────────────


def gen_replay_fight(job: tuple[Any, ...]) -> dict[str, Any]:
    """Every start of one replayed fight (GH #830 playout distillation).

    A start is `(turn, mode)`: the in-sync turn start `turn` of the fight
    (`ppo.turnplan.pro.fight_turns`, the exact engine state of the human's
    game) and `mode` "pilot" (the pilot plays from there) or "human" (the
    human's turn is forced first, checked against a plain replay, then the
    pilot plays from the next turn: a takeover post-state). The pilot plays to
    the end of the fight and every searched decision is recorded.
    """
    path, combat_id, starts, sims, boss_sims, version, env = job
    import zlib

    import sts2_rs

    from ppo.bc_human import replay
    from ppo.combat_bench import takeover
    from ppo.turnplan.pro import fight_turns, room_type

    agents.apply_env(env)
    name = Path(path).name
    # The fight id is the RUN's, so `train`'s validation split holds out runs.
    out: dict[str, Any] = {"keys": [], "decisions": [], "unmatched": 0, "skipped": 0}
    fight_idx = zlib.crc32(name.encode()) % (2**31)
    try:
        rec = next(r for r in replay.fights(path) if r["combat_id"] == combat_id)
        turns, _ = fight_turns(rec)
        room = room_type(rec["encounter"])
        for t, mode in starts:
            key = f"{name}:{combat_id}:{t}:{mode}"
            out["keys"].append(key)
            if t >= len(turns) or turns[t]["status"] != "legal":
                out["skipped"] += 1
                continue
            state = turns[t]["state"]
            seed = zlib.crc32(key.encode()) % (2**31)
            rm = sts2_rs.RunManager(
                seed=seed,
                character=rec["character"].capitalize(),
                ascension_level=rec["ascension"] or 0,
            )
            rm.debug_enter_combat(state.clone(), room)
            if mode == "human":
                check = state.clone()
                ok = True
                for a in turns[t]["actions"]:
                    ra = takeover.run_action(a)
                    if ra is None:
                        ok = False
                        break
                    rm.take_action(ra)
                    check.apply_step(a)
                    if rm.phase() != "combat" or rm.is_over():
                        break
                cs = rm.combat_state() if rm.phase() == "combat" and not rm.is_over() else None
                if not ok or cs is None or takeover._snapshot(cs) != takeover._snapshot(check):
                    out["skipped"] += 1
                    continue
            agent = sts2_rs.MctsCombatAgent(seed=seed, sims=sims, boss_sims=boss_sims)
            record_decisions(rm, agent, version, ROOMS.get(room, 0), fight_idx, out)
    except Exception as e:
        out["error"] = repr(e)[:300]
    return out


def gen_replay(args: argparse.Namespace) -> int:
    """`gen` from replayed human positions: `--starts` JSONL rows
    `{"file", "combat_id", "turn", "mode"}` (see `gen_replay_fight`), the
    replay files found through the `pro collect` dir `--collect`. Choosing
    the starts, and keeping panel runs out of them, is the caller's job
    (`docs/experiments/2026-10-23-830-playout-distill/plan_starts.py`)."""
    env = agents.build_env(agents.parse_kv(args.env or [], "--env"))
    sel = json.loads(Path(args.collect, "selection.json").read_text(encoding="utf-8"))["files"]
    path_of = {Path(r["file"]).name: r["file"] for r in sel}
    by_fight: dict[tuple[str, str], list[tuple[int, str]]] = collections.defaultdict(list)
    for line in open(args.starts, encoding="utf-8"):
        r = json.loads(line)
        by_fight[r["file"], r["combat_id"]].append((int(r["turn"]), r["mode"]))
    os.makedirs(args.out_dir, exist_ok=True)
    features: dict[str, Any] = {"feature_version": args.feature_version}
    feat_path = os.path.join(args.out_dir, FEATURES_META)
    if os.path.exists(feat_path) and json.load(open(feat_path, encoding="utf-8")) != features:
        raise SystemExit(f"{feat_path} records other features; use a new --out-dir")
    json.dump(features, open(feat_path, "w", encoding="utf-8"))
    with open(os.path.join(args.out_dir, BENCH_META), "w", encoding="utf-8") as f:
        json.dump({"starts": args.starts, "note": "panel runs excluded by the starts file"}, f)
    done_path = os.path.join(args.out_dir, "done.txt")
    done = (
        set(open(done_path, encoding="utf-8").read().split())
        if os.path.exists(done_path)
        else set()
    )
    jobs = []
    for (f, c), st in sorted(by_fight.items()):
        st = [s for s in st if f"{f}:{c}:{s[0]}:{s[1]}" not in done]
        if st:
            jobs.append((path_of[f], c, st, args.sims, args.boss_sims, args.feature_version, env))
    random.Random(args.sample_seed).shuffle(jobs)
    print(
        f"[rollout_learn gen-replay] {len(jobs)} fights, {sum(len(j[2]) for j in jobs)} starts",
        flush=True,
    )
    writer = ShardWriter(args.out_dir, args.per_shard)
    t0 = time.time()
    n = n_dec = err = skipped = 0
    with (
        get_context("spawn").Pool(args.procs) as pool_,
        open(done_path, "a", encoding="utf-8") as done_f,
    ):
        for row in pool_.imap_unordered(gen_replay_fight, jobs):
            n += 1
            err += "error" in row
            skipped += row["skipped"]
            n_dec += len(row["decisions"])
            writer.add(row["decisions"])
            for k in row["keys"]:
                done_f.write(k + "\n")
            done_f.flush()
            if n % 50 == 0 or n == len(jobs):
                print(
                    f"[rollout_learn gen-replay] {n}/{len(jobs)} fights, {n_dec} decisions, "
                    f"{err} errors, {skipped} skipped starts, {time.time() - t0:.0f}s",
                    flush=True,
                )
    writer.flush()
    print("DONE", flush=True)
    return 0


# ── train ────────────────────────────────────────────────────────────────


def data_features(data_dirs: list[str]) -> dict[str, Any]:
    """The feature metadata shared by every gen dir (v1 when none is recorded).

    Raises:
        SystemExit: When the dirs were generated with different features.
    """
    seen = []
    for d in data_dirs:
        p = os.path.join(d, FEATURES_META)
        seen.append(
            json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {"feature_version": 1}
        )
    if any(s != seen[0] for s in seen[1:]):
        raise SystemExit(f"gen dirs disagree on their features: {seen}")
    return seen[0]


def load_shards(data_dirs: list[str]) -> tuple[dict[str, np.ndarray], list[str], list[str]]:
    """Concatenate shards from several gen runs into one vocab space.

    Returns:
        `(data, keys, monster_keys)`; `monster_keys[0]` is the no-enemy row.
    """
    vocab: dict[str, int] = {"<unk>": 0}
    mvocab: dict[str, int] = {"<none>": 0}
    parts: dict[str, list[np.ndarray]] = collections.defaultdict(list)
    fight_offset = 0
    for d in data_dirs:
        local = json.load(open(os.path.join(d, "vocab.json"), encoding="utf-8"))
        remap = np.zeros(max(local.values()) + 1, np.int32)
        for k, i in local.items():
            remap[i] = vocab.setdefault(k, len(vocab))
        mpath = os.path.join(d, "monster_vocab.json")
        mlocal = (
            json.load(open(mpath, encoding="utf-8")) if os.path.exists(mpath) else {"<none>": 0}
        )
        mremap = np.zeros(max(mlocal.values()) + 1, np.int32)
        for k, i in mlocal.items():
            mremap[i] = mvocab.setdefault(k, len(mvocab))
        max_fight = 0
        for f in sorted(os.listdir(d)):
            if not f.startswith("shard-"):
                continue
            z = np.load(os.path.join(d, f))
            for name in z.files:
                a = z[name]
                if name == "key":
                    a = remap[a]
                elif name == "mon":
                    a = mremap[a]
                elif name == "fight":
                    max_fight = max(max_fight, int(a.max()) + 1)
                    a = a + fight_offset
                parts[name].append(a)
        fight_offset += max_fight
    data = {k: np.concatenate(v) for k, v in parts.items()}
    keys = [k for k, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
    mkeys = [k for k, _ in sorted(mvocab.items(), key=lambda kv: kv[1])]
    return data, keys, mkeys


def _spread_duplicates(
    key: np.ndarray, x: np.ndarray, visits: np.ndarray, starts: np.ndarray, n_act: np.ndarray
) -> np.ndarray:
    """Copies of one card share features and so logits; the search keeps one
    child for them (the transposition table), so its visits sit on one copy.
    Spread each group's mass evenly over its members."""
    out = visits.copy()
    for s, n in zip(starts, n_act, strict=True):
        if n < 2:
            continue
        groups: dict[bytes, list[int]] = collections.defaultdict(list)
        for j in range(s, s + n):
            groups[key[j].tobytes() + x[j].tobytes()].append(j)
        for members in groups.values():
            if len(members) > 1:
                out[members] = visits[members].sum() / len(members)
    return out


def _playout_net(
    torch: Any, hidden: int, emb_dim: int, n_keys: int, version: int = 1, n_monsters: int = 0
) -> Any:
    """The torch twin of `sts2_mcts::rollout_net::RolloutNet`.

    `n_monsters` > 0 adds the encounter embedding (v2 only): one `hidden`-wide
    row per species (row 0 = no enemy, kept at zero), summed over the first
    living enemies into the hidden layer before the ReLU.
    """
    state_dim, action_dim = FEATURE_DIMS[version]
    if n_monsters and version != TABLE_VERSION:
        raise ValueError("the encounter embedding needs feature version 2")

    class PlayoutNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.hidden, self.emb_dim, self.version = hidden, emb_dim, version
            self.state_dim, self.action_dim = state_dim, action_dim
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(state_dim, hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden, emb_dim),
            )
            self.emb = torch.nn.Embedding(n_keys, emb_dim)
            self.bias = torch.nn.Embedding(n_keys, 1)
            self.proj = torch.nn.Linear(action_dim, emb_dim, bias=False)
            self.lin = torch.nn.Linear(action_dim, 1, bias=False)
            self.mon = (
                torch.nn.Embedding(n_monsters + 1, hidden, padding_idx=0) if n_monsters else None
            )

        def query(self, S: Any, MON: Any = None) -> Any:
            h = self.mlp[0](S)
            if self.mon is not None and MON is not None:
                h = h + self.mon(MON).sum(-2)
            return self.mlp[2](torch.relu(h))

        def forward(self, S: Any, X: Any, K: Any, M: Any, MON: Any = None) -> Any:
            """Logits `[B, A]` for states `S [B, state_dim]`, action features
            `X [B, A, action_dim]`, key rows `K [B, A]`; `M` masks real
            actions; `MON [B, MONSTER_SLOTS]` encounter rows (v2)."""
            q = self.query(S, MON)
            k = self.emb(K) + self.proj(X)
            z = (k * q[:, None, :]).sum(-1) + self.bias(K)[..., 0] + self.lin(X)[..., 0]
            return z.masked_fill(~M, -1e9)

        def to_json(
            self,
            keys: list[str],
            temperature: float,
            meta: dict | None = None,
            no_early_end_turn: bool = False,
            monster_act: dict[str, int] | None = None,
            monster_keys: list[str] | None = None,
            encounter_act: list[list[Any]] | None = None,
        ) -> dict:
            w = lambda t: t.detach().cpu().numpy().astype(float).round(6).tolist()  # noqa: E731
            out = {
                "feature_version": self.version,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "hidden": self.hidden,
                "emb_dim": self.emb_dim,
                "temperature": temperature,
                # The playout rule the weights were trained under: the search
                # loads it from the file, so a net trained with early EndTurn
                # masked must not deploy with it allowed.
                "no_early_end_turn": no_early_end_turn,
                "keys": keys,
                "w1": w(self.mlp[0].weight),
                "b1": w(self.mlp[0].bias),
                "w2": w(self.mlp[2].weight),
                "b2": w(self.mlp[2].bias),
                "emb": w(self.emb.weight),
                "bias": w(self.bias.weight[:, 0]),
                "proj": w(self.proj.weight),
                "lin": w(self.lin.weight[0]),
                "meta": meta or {},
            }
            if self.version == TABLE_VERSION:
                if monster_act is None:
                    raise ValueError("a v2 net needs monster_act (sts2_rs.monster_acts())")
                out["monster_act"] = monster_act
                if encounter_act is not None:
                    out["encounter_act"] = encounter_act
            elif encounter_act is not None:
                raise ValueError("encounter_act belongs to feature version 2")
            if self.mon is not None:
                if monster_keys is None or len(monster_keys) != self.mon.num_embeddings - 1:
                    raise ValueError("monster_keys must name every encounter-embedding row")
                out["monster_keys"] = monster_keys
                out["monster_emb"] = w(self.mon.weight[1:])
            return out

    return PlayoutNet()


def _load_playout_net(torch: Any, net: dict[str, Any]) -> Any:
    version = int(net.get("feature_version", 1))
    mkeys = net.get("monster_keys") or []
    m = _playout_net(
        torch, net["hidden"], net["emb_dim"], len(net["keys"]), version, n_monsters=len(mkeys)
    )
    t = lambda a: torch.as_tensor(np.asarray(a, np.float32))  # noqa: E731
    with torch.no_grad():
        m.mlp[0].weight.copy_(t(net["w1"]))
        m.mlp[0].bias.copy_(t(net["b1"]))
        m.mlp[2].weight.copy_(t(net["w2"]))
        m.mlp[2].bias.copy_(t(net["b2"]))
        m.emb.weight.copy_(t(net["emb"]))
        m.bias.weight.copy_(t(net["bias"])[:, None])
        m.proj.weight.copy_(t(net["proj"]))
        m.lin.weight.copy_(t(net["lin"])[None, :])
        if m.mon is not None:
            m.mon.weight[1:].copy_(t(net["monster_emb"]))
    return m


def feature_tables() -> dict[str, Any]:
    """The engine's v2 act tables, as a v2 weights file / gen dir records them."""
    import sts2_rs

    return {
        "monster_act": sts2_rs.monster_acts(),
        "encounter_act": [[int(a), list(m)] for a, m in sts2_rs.encounter_acts()],
    }


def _zero_pad(net: dict[str, Any], version: int) -> dict[str, Any]:
    """`net` with `version`'s dims, every added `w1` / `proj` / `lin` column 0."""
    sd, ad = FEATURE_DIMS[version]
    out = dict(net)
    out["feature_version"] = version
    out["state_dim"], out["action_dim"] = sd, ad
    out["w1"] = [list(r) + [0.0] * (sd - len(r)) for r in net["w1"]]
    out["proj"] = [list(r) + [0.0] * (ad - len(r)) for r in net["proj"]]
    out["lin"] = list(net["lin"]) + [0.0] * (ad - len(net["lin"]))
    return out


def upgrade_net(
    net: dict[str, Any],
    tables: dict[str, Any],
    monster_keys: list[str] | None = None,
    version: int = 2,
) -> dict[str, Any]:
    """A weights file as a version-`version` one.

    To v2 (the default): the same policy. A v1 file gets the v2 columns of
    `w1`, `proj` and `lin` at zero and the act `tables` (`feature_tables()`).
    With `monster_keys`, a file without an encounter embedding (v1, or v2
    without one) gets one with a zero row per species. Every added weight is
    zero, so the logits equal the input net's (to float rounding) until
    training moves them. A v2 file keeps its own tables, and an embedding it
    already has.

    To v4: a v1 file gets the v4 columns of `w1`, `proj` and `lin` at zero,
    so the same policy until training moves them. No tables or embedding.

    To v3: a v1 file is relabeled, every weight unchanged. v3 is v1's layout
    read on the displayed incoming damage and the form-aware "removed", so
    the logits equal v1's wherever those two read the same and move where
    they don't (that is the point). A v3 net has no tables or embedding.

    Raises:
        ValueError: For any other pair (v2 -> v3, a downgrade, `monster_keys`
            with v3 or v4).
    """
    have = int(net.get("feature_version", 1))
    if version == 4:
        if have not in (1, 4) or monster_keys:
            raise ValueError(f"cannot turn a v{have} net into v4 (only v1 nets, no embedding)")
        if have == 4:
            return dict(net)
        return _zero_pad(net, 4)
    if version == 3:
        if have not in (1, 3) or monster_keys:
            raise ValueError(f"cannot turn a v{have} net into v3 (only v1 nets, no embedding)")
        out = dict(net)
        out["feature_version"] = 3
        return out
    if version != TABLE_VERSION or have == 3:
        raise ValueError(f"cannot turn a v{have} net into v{version}")
    out = dict(net)
    if have < 2:
        out = _zero_pad(net, 2)
        out["monster_act"] = dict(tables["monster_act"])
        out["encounter_act"] = [list(e) for e in tables["encounter_act"]]
    if monster_keys and not out.get("monster_keys"):
        out["monster_keys"] = list(monster_keys)
        out["monster_emb"] = [[0.0] * int(net["hidden"]) for _ in monster_keys]
    return out


def train(args: argparse.Namespace) -> int:
    import torch

    data, keys, mkeys = load_shards(args.data)
    features = data_features(args.data)
    version = int(features["feature_version"])
    init = json.load(open(args.init, encoding="utf-8")) if args.init else None
    if init is not None:
        if int(init.get("feature_version", 1)) != version or init.get("monster_keys"):
            raise SystemExit(
                "--init must be a net of the data's feature version, no encounter rows"
            )
        # The net's key vocabulary is the init's, extended by the data's new keys
        # (their rows start at zero: the init's policy exactly, until trained).
        net_keys = list(init["keys"])
        pos = {k: i for i, k in enumerate(net_keys)}
        for k in keys:
            if k not in pos:
                pos[k] = len(net_keys)
                net_keys.append(k)
        data["key"] = np.asarray([pos[k] for k in keys], np.int64)[data["key"]]
        keys = net_keys
    use_mon = bool(args.monster_emb)
    if use_mon and (version != TABLE_VERSION or "mon" not in data):
        raise SystemExit("--monster-emb needs v2 gen data (gen --feature-version 2)")
    n_dec = len(data["n_act"])
    n_act = data["n_act"].astype(np.int64)
    starts = np.concatenate([[0], np.cumsum(n_act)[:-1]])
    # Root visits are near-uniform at the deployed c_puct (GH #442: top child
    # ~1.06x uniform), so they say little; the search's recommended action
    # carries its verdict. `--target-mix` blends the two.
    chosen = np.zeros_like(data["visits"])
    has_choice = data["chosen"] >= 0
    chosen[starts[has_choice] + data["chosen"][has_choice]] = 1.0
    visits = data["visits"] / np.maximum(np.add.reduceat(data["visits"], starts), 1e-9).repeat(
        n_act
    )
    mix = np.where(has_choice.repeat(n_act), args.target_mix, 1.0)
    target = _spread_duplicates(
        data["key"], data["x"], (1.0 - mix) * chosen + mix * visits, starts, n_act
    )
    print(
        f"[rollout_learn] {n_dec} decisions, {len(target)} actions, {len(keys)} keys, "
        f"feature version {version}, {len(mkeys) - 1 if use_mon else 0} encounter rows",
        flush=True,
    )

    # Flat per-action arrays stay flat on the device; each batch is padded to
    # its own widest decision. Padding every decision to the global widest one
    # needs ~37 GiB for a 1.6M-decision Defect gen (a few decisions offer
    # hundreds of actions).
    target = target / np.maximum(np.add.reduceat(target, starts), 1e-9).repeat(n_act)
    C = data["chosen"].astype(np.int64)

    fights = np.unique(data["fight"])
    rng = np.random.default_rng(args.seed)
    val_f = set(
        rng.choice(fights, max(1, int(len(fights) * args.val_frac)), replace=False).tolist()
    )
    is_val = np.array([f in val_f for f in data["fight"]])
    dev = torch.device(args.device)
    tt = lambda a: torch.as_tensor(a, device=dev)  # noqa: E731
    S_, C_ = tt(data["state"]), tt(C)
    Xf, Kf, Tf = tt(data["x"]), tt(data["key"].astype(np.int64)), tt(target.astype(np.float32))
    start_, nact_ = tt(starts.astype(np.int64)), tt(n_act)
    MON_ = tt(data["mon"].astype(np.int64)) if use_mon else None

    def padded(idx: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """`X [B, A, action_dim]`, `K [B, A]`, `T [B, A]`, `M [B, A]` for decisions `idx`."""
        n = nact_[idx]
        col = torch.arange(int(n.max()), device=dev)
        M = col[None, :] < n[:, None]
        flat = torch.where(M, start_[idx][:, None] + col[None, :], 0)
        X = Xf[flat] * M[..., None]
        return X, Kf[flat] * M, Tf[flat] * M, M

    R_ = tt(data["room"].astype(np.int64))
    tr_idx = torch.as_tensor(np.flatnonzero(~is_val), device=dev)
    va_idx = torch.as_tensor(np.flatnonzero(is_val), device=dev)

    torch.manual_seed(args.seed)
    ref = None
    if init is not None:
        n0 = len(init["keys"])
        base_net = _load_playout_net(torch, init)
        net = _playout_net(torch, init["hidden"], init["emb_dim"], len(keys), version).to(dev)
        with torch.no_grad():
            for name, p in net.named_parameters():
                q = dict(base_net.named_parameters())[name]
                if name in ("emb.weight", "bias.weight"):
                    p.zero_()
                    p[:n0].copy_(q)
                else:
                    p.copy_(q)
        if args.kl_init > 0:
            import copy

            ref = copy.deepcopy(net).eval()
            for p in ref.parameters():
                p.requires_grad_(False)
    else:
        net = _playout_net(
            torch, args.hidden, args.emb_dim, len(keys), version, len(mkeys) - 1 if use_mon else 0
        ).to(dev)
        for p in (net.emb.weight, net.bias.weight, net.proj.weight):
            torch.nn.init.normal_(p, std=0.01)
        with torch.no_grad():
            # Start AT the heuristic playout: logit = -priority / tau.
            net.lin.weight.zero_()
            net.lin.weight[0, HEUR_FEATURE] = 1.0
            if net.mon is not None:
                net.mon.weight.zero_()
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)

    def logits(idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        X, K, T, M = padded(idx)
        return net(S_[idx], X, K, M, None if MON_ is None else MON_[idx]), T

    def metrics(idx: torch.Tensor, heuristic: bool = False) -> dict[str, float]:
        out: dict[str, list[float]] = collections.defaultdict(list)
        with torch.no_grad():
            for b in torch.split(idx, 16384):
                if heuristic:
                    X, _, T, M = padded(b)
                    z = X[..., HEUR_FEATURE].masked_fill(~M, -1e9)
                else:
                    z, T = logits(b)
                lp = torch.log_softmax(z, -1)
                ce = -(T * lp).sum(-1)
                has = C_[b] >= 0
                top1 = (z.argmax(-1) == C_[b]).float()
                out["ce"].append(ce.sum().item())
                out["top1"].append(top1[has].sum().item())
                out["n"].append(float(len(b)))
                out["n_ch"].append(has.sum().item())
                for r, name in ((2, "boss"),):
                    sel = R_[b] == r
                    out[f"ce_{name}"].append(ce[sel].sum().item())
                    out[f"n_{name}"].append(sel.sum().item())
        n, n_ch = sum(out["n"]), max(1.0, sum(out["n_ch"]))
        return {
            "ce": sum(out["ce"]) / n,
            "top1": sum(out["top1"]) / n_ch,
            "ce_boss": sum(out["ce_boss"]) / max(1.0, sum(out["n_boss"])),
        }

    if not len(tr_idx):
        raise SystemExit("no training decisions left after the validation split")
    base = metrics(va_idx, heuristic=True)
    print(f"[rollout_learn] heuristic on val: {base}", flush=True)
    if init is not None:
        meta_init = metrics(va_idx)
        print(f"[rollout_learn] init net on val: {meta_init}", flush=True)
    steps_per_epoch = math.ceil(len(tr_idx) / args.batch)
    for epoch in range(args.epochs):
        perm = tr_idx[torch.randperm(len(tr_idx), device=dev)]
        tot = 0.0
        for b in torch.split(perm, args.batch):
            z, T = logits(b)
            if ref is not None:
                # Pull toward the init policy (a soft trust region): the target
                # is (1 - kl) x teacher + kl x the init net's own distribution.
                X, K, _, M = padded(b)
                with torch.no_grad():
                    p_ref = torch.softmax(ref(S_[b], X, K, M) / args.temperature, -1)
                T = (1 - args.kl_init) * T + args.kl_init * p_ref
            loss = -(T * torch.log_softmax(z, -1)).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        print(
            f"[rollout_learn] epoch {epoch + 1}: train ce {tot / steps_per_epoch:.4f}  "
            f"val {metrics(va_idx)}",
            flush=True,
        )

    meta = {"data": args.data, "decisions": n_dec, "val_heuristic": base, "val": metrics(va_idx)}
    if init is not None:
        meta.update(init=args.init, kl_init=args.kl_init, val_init=meta_init)
    # The combat-bench holdout each gen run applied (absent for gen runs made
    # before GH #830, which held nothing out).
    meta["bench_holdout"] = {
        d: json.load(open(os.path.join(d, BENCH_META), encoding="utf-8"))
        for d in args.data
        if os.path.exists(os.path.join(d, BENCH_META))
    }
    out = net.to_json(
        keys,
        args.temperature,
        meta,
        monster_act=features.get("monster_act"),
        monster_keys=mkeys[1:] if use_mon else None,
        encounter_act=features.get("encounter_act"),
    )
    json.dump(out, open(args.out, "w", encoding="utf-8"))
    print(f"[rollout_learn] wrote {args.out}", flush=True)
    return 0


# ── rl ───────────────────────────────────────────────────────────────────


class Start:
    """One playout start: the state, where it came from, and its SB target."""

    __slots__ = ("file", "key", "spec", "state", "vstar")

    def __init__(
        self,
        key: str,
        file: str,
        state: Any,
        spec: dict[str, Any],
        vstar: float | None = None,
    ) -> None:
        self.key, self.file, self.state, self.spec, self.vstar = key, file, state, spec, vstar


def _combat_states(sts2_rs: Any, fights: list[dict[str, Any]]) -> list[tuple[str, Any]]:
    """Each fight's opening combat state (same build as `gen`); modded runs skipped."""
    out = []
    for f in fights:
        key = f"{f['file']}:{f['floor']}"
        s = _fight_seed(key) * 100 + 77
        try:
            rm = rollout_starts.build_fight(f, s)
        except ValueError:
            continue
        out.append((key, rm.combat_state()))
    return out


def fight_starts(sts2_rs: Any, fights: list[dict[str, Any]]) -> list[Start]:
    """`--starts fight`: each fight's opening state (the original `rl` pool)."""
    out = []
    for f in fights:
        key = f"{f['file']}:{f['floor']}"
        s = _fight_seed(key) * 100 + 77
        try:
            rm = rollout_starts.build_fight(f, s)
        except ValueError:
            continue
        cs = rm.combat_state()
        h = rollout_starts.ordered_hash(cs)
        spec = {
            "id": rollout_starts.start_id(key, s, 0, h),
            "fight": f,
            "seed": s,
            "actions": [],
            "hash": h,
        }
        out.append(Start(key, f["file"], cs, spec))
    return out


def mid_starts(
    paths: list[str],
    holdout: set[str],
    holdout_encounters: list[str] | None,
    rooms: set[str],
    allow_bench_overlap: bool,
    n_states: int,
    per_traj: int,
    seed: int,
) -> tuple[list[Start], dict[str, Any]]:
    """`--starts mid`: decision points of the pilot's recorded trajectories.

    Held-out runs (`--holdout` files, `--holdout-encounter`) and combat-bench
    runs are dropped exactly as for fight starts. Every rebuilt state is
    checked against its recorded hash (`rollout_starts.rebuild`).
    """
    recs = rollout_starts.read_records(paths)
    recs = [
        r
        for r in recs
        if r["file"] not in holdout
        and r["room"] in rooms
        and not (holdout_encounters and r["encounter"] in holdout_encounters)
    ]
    recs, bench_meta = subtract_bench_runs(
        recs, lambda r: r["file"], allow_bench_overlap, tag="[rollout_learn rl]"
    )
    rng = random.Random(seed)
    points = rollout_starts.candidate_points(recs, per_traj, rng)
    if n_states and len(points) > n_states:
        points = rng.sample(points, n_states)
    by_rec: dict[int, list[int]] = collections.defaultdict(list)
    for i, t in points:
        by_rec[i].append(t)
    out = []
    for i in sorted(by_rec):
        rec = recs[i]
        states = rollout_starts.rebuild(rec, by_rec[i])
        for t in sorted(states):
            out.append(
                Start(
                    rec["key"],
                    rec["file"],
                    states[t],
                    rollout_starts.start_spec(rec, t),
                    rollout_starts.pilot_value(rec, t),
                )
            )
    meta = {
        "trajectories": len(recs),
        "decision_points": len(points),
        "bench_holdout": bench_meta,
    }
    return out, meta


def _unpack_playouts(
    d: dict[str, bytes], state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM
) -> dict[str, np.ndarray]:
    u = lambda k, dt: np.frombuffer(d[k], dt).copy()  # noqa: E731  (writable for torch)
    out = {
        "steps": u("steps", np.uint32).astype(np.int64),
        "state": u("state", np.float32).reshape(-1, state_dim),
        "n_act": u("n_act", np.uint8).astype(np.int64),
        "chosen": u("chosen", np.uint8).astype(np.int64),
        "rows": u("rows", np.uint32).astype(np.int64),
        "x": u("x", np.float32).reshape(-1, action_dim),
        "allowed": u("allowed", np.uint8).astype(bool),
        "won": u("won", np.uint8).astype(bool),
        "hp_frac": u("hp_frac", np.float32),
        "removed": u("removed", np.float32),
        "round": u("round", np.float32),
    }
    if "search_value" in d:
        out["search_value"] = u("search_value", np.float32)
    if "mon" in d:
        out["mon"] = u("mon", np.uint16).astype(np.int64).reshape(-1, MONSTER_SLOTS)
    return out


def shaped_reward(p: dict[str, np.ndarray], w_hp: float, w_removed: float) -> np.ndarray:
    """The original `rl` reward: win `1 + w_hp·hp`, loss `w_removed·removed`."""
    return np.where(p["won"], 1.0 + w_hp * p["hp_frac"], w_removed * p["removed"])


def loo_advantage(r: np.ndarray, k: int) -> np.ndarray:
    """Leave-one-out baseline over each start's k playouts, std-normalized."""
    g = r.reshape(-1, k)
    adv = (g - (g.sum(1, keepdims=True) - g) / (k - 1)).reshape(-1)
    return adv / (adv.std() + 1e-6)


def sb_coefficients(
    z: np.ndarray, vstar: np.ndarray, k: int
) -> tuple[np.ndarray, dict[str, float]]:
    """Per-playout weights of the simulation-balancing gradient, on the same
    footing as the REINFORCE advantages they are added to.

    Silver & Tesauro (2009): minimise `½ Σ_s (V*(s) − V(s))²` over the
    playout policy, `V(s) = E[z | s]`; its gradient is
    `−Σ_s (V*(s) − V(s)) E[z ∇log π]`. The product of the two expectations
    is estimated without bias from two INDEPENDENT halves of the k playouts:
    one half's mean estimates `V(s)`, the other half estimates the policy
    gradient (with a leave-one-out baseline inside that half, which keeps it
    unbiased), then the halves swap.

    The weight of playout j is `c[j] = (V*(s) − V̂_other(s)) · â[j]`, where
    `â` is its half-LOO advantage divided by the batch std of those
    advantages: the normalisation `loo_advantage` gives the REINFORCE term.
    `rl` uses `adv[j] + β · c[j]` per decision, one mean over all decisions,
    so per playout the SB weight is `β · |V* − V̂|` times a unit-variance
    advantage, next to REINFORCE's unit-variance advantage: β = 1 with a
    0.5 gap weighs SB half as much as REINFORCE, whatever the playout length
    or the reward's scale.

    Args:
        z: Playout returns, `n·k` (start-major), in search-value units.
        vstar: Per-start target, `n` (NaN = no target: weight 0).
        k: Playouts per start (even, >= 4).

    Returns:
        `(c, stats)`: `c` per playout (`n·k`), and `stats` (`sb_gap` = mean
        |V* − V̂|, `sb_mse`, `n_sb` = starts with a target).
    """
    g = z.reshape(-1, k).astype(np.float64)
    h = k // 2
    has = ~np.isnan(vstar)
    gap = np.zeros_like(g)
    adv = np.zeros_like(g)
    for own, other in ((slice(0, h), slice(h, k)), (slice(h, k), slice(0, h))):
        v_other = g[:, other].mean(1)
        gh = g[:, own]
        adv[:, own] = gh - (gh.sum(1, keepdims=True) - gh) / (h - 1)
        gap[:, own] = (np.nan_to_num(vstar) - v_other)[:, None]
    c = gap * adv / (adv.std() + 1e-6)
    c[~has] = 0.0
    n_sb = int(has.sum())
    vhat = g.mean(1)
    err = np.abs(vstar[has] - vhat[has])
    stats = {
        "n_sb": n_sb,
        "sb_gap": float(err.mean()) if n_sb else float("nan"),
        "sb_mse": float((err**2).mean()) if n_sb else float("nan"),
    }
    return c.reshape(-1), stats


CKPT_VERSION = 2


def _settings(args: argparse.Namespace) -> dict[str, Any]:
    return {
        k: v
        for k, v in vars(args).items()
        if isinstance(v, (str, int, float, bool, list, type(None))) and k != "resume"
    }


def _truncate_metrics(path: str, it0: int) -> int:
    """Drop the metrics rows an interrupted run wrote after its checkpoint.

    The checkpoint for iteration `it0` is written right after that
    iteration's training row and before the eval of the next loop pass, so a
    resume replays (and logs again) every eval row from `it0` on and every
    training row past `it0`.

    Returns:
        The number of rows dropped.
    """
    if not os.path.exists(path):
        return 0
    keep, dropped = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            later = row["it"] >= it0 if "eval" in row else row["it"] > it0
            if later:
                dropped += 1
            else:
                keep.append(line if line.endswith("\n") else line + "\n")
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.writelines(keep)
    Path(tmp).replace(path)
    return dropped


def rl(args: argparse.Namespace) -> int:
    """REINFORCE on playout outcomes: play many playouts from human-bank
    starts with the current policy, and raise the log-probability of the
    decisions in playouts that did better than their start's other playouts.

    Round-2 options, each off by default (the defaults are the original
    trainer):

    - `--reward search`: the reward is what the deployed search backs up for
      the playout (`search_value`, `sts2_mcts::playout_return`: wins in
      [1.0, 1.5] by HP and kept potions, losses in [-1.45, -1.05) graded by
      enemy HP removed / HP / turns) instead of the shaped win/HP/removed
      reward.
    - `--starts mid TRAJ...`: starts are the pilot's recorded decision points
      (`traj`), fight starts included, rebuilt and hash-checked; `--view
      belief` (their default) plays each from the search's honest view.
      `--contested-k K` reweights them by `max(p(1-p), floor)` from a K-playout
      pre-pass.
    - `--sb-beta B`: adds B × the simulation-balancing term
      (`sb_coefficients`) toward `--sb-target pilot` (the pilot's recorded
      outcome from the start) or `search` (a deeper honest search's value,
      computed lazily and cached). Per playout, the SB weight is
      `B · |V* − V̂|` times a unit-variance advantage, REINFORCE's is a
      unit-variance advantage; each iteration logs both gradient norms
      (`g_rf`, `g_sb`).
    - `--feature-version 2|3`: upgrade a v1 `--init` to the v2 features
      (zero-padded) or the v3 ones (relabeled).

    The engine env (`--env`, `--overrides`) is applied to this process before
    anything is scored, and every inherited engine seam is cleared: the
    playouts' `search_value`, the search reward and the eval values are read
    under it, the `--sb-target search` workers search under it, and `--starts
    mid` refuses trajectories scored under another value env
    (`rollout_starts.check_value_env`).

    Every iteration appends a metrics row to `--metrics`
    (default `<out>.metrics.jsonl`); `--ckpt-every` writes a resumable
    checkpoint (`<out>.ckpt.pt`), which `--resume` continues from exactly as
    the uninterrupted run would have.
    """
    env = agents.build_env(agents.parse_kv(args.env or [], "--env"), overrides=args.overrides)
    scrubbed = agents.apply_env(env)
    if scrubbed:
        print(
            f"[rollout_learn rl] cleared inherited engine env {scrubbed}; pass what you "
            "mean with --env",
            flush=True,
        )
    import sts2_rs
    import torch

    rollout_starts.check_k(args.k, args.sb_beta)
    if args.contested_k and args.contested_floor <= 0:
        raise SystemExit("--contested-floor must be > 0 (a start must stay drawable)")
    torch.manual_seed(args.seed)
    # The minibatch order: a CPU generator, so it is seeded by --seed and
    # restored by --resume on any device.
    perm_gen = torch.Generator().manual_seed(args.seed)
    dev = torch.device(args.device)
    init = json.load(open(args.init, encoding="utf-8"))
    init_version = int(init.get("feature_version", 1))
    version = args.feature_version or init_version
    if version != init_version and init_version != 1:
        raise SystemExit(
            f"--feature-version {version} from a v{init_version} --init: only a v1 net upgrades"
        )
    if args.monster_emb and version != TABLE_VERSION:
        raise SystemExit(
            "--monster-emb needs a v2 net: add --feature-version 2 or start from a v2 --init"
        )
    if version != init_version or (args.monster_emb and not init.get("monster_keys")):
        init = upgrade_net(
            init,
            feature_tables() if version == TABLE_VERSION else {},
            sts2_rs.monster_species() if args.monster_emb else None,
            version=version,
        )
    sd, ad = FEATURE_DIMS[version]
    keys = init["keys"]
    monster_act = init.get("monster_act")
    encounter_act = init.get("encounter_act")
    monster_keys = init.get("monster_keys")
    net = _load_playout_net(torch, init).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    mode = args.starts[0]
    if mode not in ("fight", "mid") or (mode == "fight") != (len(args.starts) == 1):
        raise SystemExit("--starts expects 'fight' or 'mid TRAJ.jsonl [...]'")
    view = args.view or ("belief" if mode == "mid" else "raw")
    belief = view == "belief"
    if args.sb_beta > 0 and args.sb_target == "pilot" and mode != "mid":
        raise SystemExit("--sb-target pilot needs --starts mid (the pilot's recorded outcomes)")
    if mode == "mid":
        rollout_starts.check_value_env(
            rollout_starts.read_headers(args.starts[1:]), env, args.allow_env_mismatch
        )

    fights = [json.loads(line) for line in open(args.fights, encoding="utf-8")]
    holdout: set[str] = set()
    for path in args.holdout or []:
        holdout |= {json.loads(line)["file"] for line in open(path, encoding="utf-8")}
    if args.holdout_encounter:
        holdout |= {f["file"] for f in fights if f["encounter"] in args.holdout_encounter}
    rooms = set(args.rooms.split(","))
    t0 = time.time()
    if mode == "fight":
        pool = [f for f in fights if f["file"] not in holdout and f["room_type"] in rooms]
        pool, bench_meta = subtract_bench_runs(
            pool, lambda f: f["file"], args.allow_bench_overlap, tag="[rollout_learn rl]"
        )
        random.Random(args.seed).shuffle(pool)
        starts = fight_starts(sts2_rs, pool[: args.n_states])
        start_meta: dict[str, Any] = {"bench_holdout": bench_meta}
    else:
        starts, start_meta = mid_starts(
            args.starts[1:],
            holdout,
            args.holdout_encounter,
            rooms,
            args.allow_bench_overlap,
            args.n_states,
            args.mid_per_traj,
            args.seed,
        )
        bench_meta = start_meta["bench_holdout"]
    if not starts:
        raise SystemExit("no training starts")
    evals = []
    for path in args.eval_fights:
        evals.append(
            (
                os.path.basename(path),
                _combat_states(
                    sts2_rs, [json.loads(line) for line in open(path, encoding="utf-8")]
                ),
            )
        )
    print(
        f"[rollout_learn rl] {len(starts)} {mode} starts ({view} view), feature version "
        f"{version}, eval {[(n, len(e)) for n, e in evals]}, engine env {env}, built in "
        f"{time.time() - t0:.0f}s",
        flush=True,
    )
    rng = np.random.default_rng(args.seed)
    settings = _settings(args)
    metrics_path = args.metrics or f"{args.out}.metrics.jsonl"
    ckpt_path = f"{args.out}.ckpt.pt"
    ckpt_every = args.ckpt_every if args.ckpt_every is not None else args.eval_every
    targets = None
    if args.sb_beta > 0 and args.sb_target == "search":
        targets = rollout_starts.SearchTargets(
            args.sb_cache or f"{args.out}.sbcache.jsonl", args.sb_search_sims, args.sb_procs, env
        )
        print(
            f"[rollout_learn rl] sb-target search: {len(targets.values)} cached values, "
            f"{targets.ignored} cache rows of another search/engine/estimator ignored",
            flush=True,
        )
    weights = None
    it0 = 0
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=dev, weights_only=True)
        if ck.get("version") != CKPT_VERSION:
            raise SystemExit(f"{ckpt_path}: written by another rl version; start a new --out")
        changed = {
            k: (ck["settings"].get(k), v)
            for k, v in settings.items()
            if ck["settings"].get(k) != v and k not in ("iters", "eval_every", "ckpt_every")
        }
        if changed:
            print(
                f"[rollout_learn rl] WARNING: resuming with changed settings {changed}", flush=True
            )
        if len(ck["start_ids"]) != len(starts) or ck["start_ids"] != [s.spec["id"] for s in starts]:
            raise SystemExit(f"{ckpt_path}: the start pool differs from the checkpoint's")
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])
        rng.bit_generator.state = ck["np_rng"]
        # `map_location` moved it to `dev`; the generator's state is a CPU tensor.
        perm_gen.set_state(ck["perm_rng"].cpu())
        # Weights restored only for a run that still samples by them.
        if args.contested_k and ck.get("weights") is not None:
            weights = np.asarray(ck["weights"], np.float64)
        it0 = int(ck["it"])
        dropped = _truncate_metrics(metrics_path, it0)
        print(
            f"[rollout_learn rl] resumed {ckpt_path} at iteration {it0} ({dropped} metrics "
            "rows past it dropped)",
            flush=True,
        )
    elif args.resume:
        print(f"[rollout_learn rl] --resume: no {ckpt_path}, starting fresh", flush=True)

    def log(row: dict[str, Any]) -> None:
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def net_json(temperature: float, meta: dict | None = None) -> dict:
        return net.to_json(
            keys,
            temperature,
            meta=meta,
            no_early_end_turn=args.no_early_end,
            monster_act=monster_act,
            monster_keys=monster_keys,
            encounter_act=encounter_act,
        )

    def play(lp: Any, states: list[Any], seeds: list[int], **kw: Any) -> dict[str, np.ndarray]:
        return _unpack_playouts(lp.play_many(states, seeds, **kw), sd, ad)

    def evaluate(lp: Any, it: int) -> None:
        views = [("raw", False)] + ([("belief", True)] if belief else [])
        for name, ev in evals:
            row = []
            for vname, vb in views:
                for temp, ne in ((1.0, False), (0.25, True)):
                    jobs = [st for _, st in ev for _ in range(args.eval_reps)]
                    seeds = [1_000_003 * i + 17 for i in range(len(jobs))]
                    p = play(lp, jobs, seeds, temperature=temp, no_early_end=ne, belief=vb)
                    tag = f"{vname} T={temp}{' ne' if ne else ''}"
                    row.append(f"{tag} {100 * p['won'].mean():.1f}%")
                    log(
                        {
                            "it": it,
                            "eval": name,
                            "view": vname,
                            "temperature": temp,
                            "no_early_end": ne,
                            "win": float(p["won"].mean()),
                            "search_value": float(p["search_value"].mean()),
                        }
                    )
            print(f"[rollout_learn rl] it {it} eval {name}: {'  '.join(row)}", flush=True)

    def contested(lp: Any) -> np.ndarray:
        jobs = [s.state for s in starts for _ in range(args.contested_k)]
        seeds = [int(v) for v in rng.integers(0, 2**62, len(jobs))]
        won = play(lp, jobs, seeds, temperature=1.0, no_early_end=args.no_early_end, belief=belief)[
            "won"
        ]
        pwin = won.reshape(len(starts), args.contested_k).mean(1)
        w = rollout_starts.contested_weights(pwin, args.contested_floor)
        print(
            f"[rollout_learn rl] contestedness pre-pass: mean p(win) {pwin.mean():.3f}, "
            f"{(pwin == 0).mean():.1%} never won, {(pwin == 1).mean():.1%} always won",
            flush=True,
        )
        return w

    def grad_norm(loss: Any) -> float:
        gs = torch.autograd.grad(loss, list(net.parameters()), retain_graph=True, allow_unused=True)
        sq = torch.zeros((), dtype=torch.float64, device=dev)
        for g in gs:
            if g is not None:
                sq = sq + (g.double() ** 2).sum()
        return float(sq.sqrt())

    vstar_all = np.array([np.nan if s.vstar is None else s.vstar for s in starts])
    for it in range(it0, args.iters + 1):
        lp = sts2_rs.LearnedPlayout(json.dumps(net_json(1.0)))
        if it % args.eval_every == 0:
            evaluate(lp, it)
            meta = {
                "rl": settings,
                "engine_env": env,
                "iteration": it,
                "bench_holdout": bench_meta,
                "starts": {k: v for k, v in start_meta.items() if k != "bench_holdout"},
            }
            out = net_json(args.temperature, meta)
            tmp = f"{args.out}.tmp"
            json.dump(out, open(tmp, "w", encoding="utf-8"))
            Path(tmp).replace(args.out)
        if it == args.iters:
            break
        # `it > 0`, not `> it0`: a resumed run refreshes exactly where the
        # uninterrupted one does (the checkpoint precedes this pre-pass).
        if args.contested_k and (
            weights is None or (args.contested_every and it % args.contested_every == 0 and it > 0)
        ):
            weights = contested(lp)
        n_pick = min(args.batch_states, len(starts))
        if weights is None:
            idx = rng.choice(len(starts), n_pick, replace=False)
        else:
            idx = rng.choice(len(starts), n_pick, replace=False, p=weights / weights.sum())
        jobs = [starts[i].state for i in idx for _ in range(args.k)]
        seeds = [int(v) for v in rng.integers(0, 2**62, len(jobs))]
        p = play(lp, jobs, seeds, temperature=1.0, no_early_end=args.no_early_end, belief=belief)
        if args.reward == "search":
            r = p["search_value"].astype(np.float64)
        else:
            r = shaped_reward(p, args.w_hp, args.w_removed)
        adv = loo_advantage(r, args.k)
        coef = adv.copy()
        sb_coef = None
        sb_stats: dict[str, float] = {}
        if args.sb_beta > 0:
            if targets is not None:
                vals = targets.get([starts[i].spec for i in idx])
                vstar = np.array([np.nan if v is None else v for v in vals])
            else:
                vstar = vstar_all[idx]
            c, sb_stats = sb_coefficients(p["search_value"], vstar, args.k)
            sb_coef = args.sb_beta * c
            coef = adv + sb_coef
        n_dec = len(p["n_act"])
        if n_dec == 0:
            print(f"[rollout_learn rl] it {it + 1}: no unforced decision to learn from", flush=True)
            continue
        a_max = int(p["n_act"].max())
        st_idx = np.concatenate([[0], np.cumsum(p["n_act"])[:-1]])
        col = np.arange(len(p["rows"])) - np.repeat(st_idx, p["n_act"])
        row = np.repeat(np.arange(n_dec), p["n_act"])
        X = np.zeros((n_dec, a_max, ad), np.float32)
        X[row, col] = p["x"]
        K = np.zeros((n_dec, a_max), np.int64)
        K[row, col] = p["rows"]
        M = np.zeros((n_dec, a_max), bool)
        M[row, col] = p["allowed"]
        tt = lambda a: torch.as_tensor(a, device=dev)  # noqa: E731
        per_step = lambda v, steps=p["steps"]: np.repeat(v, steps).astype(np.float32)  # noqa: E731
        S_, X_, K_, M_, C_, A_ = map(tt, (p["state"], X, K, M, p["chosen"], per_step(coef)))
        MON_ = tt(p["mon"]) if "mon" in p and net.mon is not None else None
        perm = torch.randperm(n_dec, generator=perm_gen).to(dev)
        ent_tot = loss_tot = 0.0
        norms: dict[str, float] = {}
        for bi, b in enumerate(torch.split(perm, args.minibatch)):
            logp = torch.log_softmax(
                net(S_[b], X_[b], K_[b], M_[b], None if MON_ is None else MON_[b]), -1
            )
            lp_ch = logp.gather(1, C_[b][:, None])[:, 0]
            ent = -(logp.exp() * logp.masked_fill(~M_[b], 0.0)).sum(-1)
            if sb_coef is not None and bi == 0:
                # The two terms' gradient norms on the first minibatch, so
                # --sb-beta can be read against REINFORCE.
                norms["g_rf"] = grad_norm(-(tt(per_step(adv))[b] * lp_ch).mean())
                norms["g_sb"] = grad_norm(-(tt(per_step(sb_coef))[b] * lp_ch).mean())
                norms["sb_ratio"] = norms["g_sb"] / max(norms["g_rf"], 1e-12)
            loss = -(A_[b] * lp_ch).mean() - args.ent * ent.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            ent_tot += ent.mean().item()
            loss_tot += loss.item()
        nb = math.ceil(n_dec / args.minibatch)
        row_m = {
            "it": it + 1,
            "win": float(p["won"].mean()),
            "reward": float(r.mean()),
            "search_value": float(p["search_value"].mean()),
            "decisions_per_playout": n_dec / len(jobs),
            "entropy": ent_tot / nb,
            "loss": loss_tot / nb,
            "time_s": round(time.time() - t0, 1),
            **sb_stats,
            **norms,
        }
        log(row_m)
        sb_txt = (
            f"  sb gap {sb_stats['sb_gap']:.3f} on {sb_stats['n_sb']}, |g_sb|/|g_rf| "
            f"{norms.get('sb_ratio', float('nan')):.2f}"
            if sb_stats
            else ""
        )
        print(
            f"[rollout_learn rl] it {it + 1}: playout win {100 * p['won'].mean():.1f}%  reward "
            f"{r.mean():.3f}  decisions/playout {n_dec / len(jobs):.1f}  entropy {ent_tot / nb:.3f}  "
            f"loss {loss_tot / nb:+.4f}{sb_txt}  {time.time() - t0:.0f}s",
            flush=True,
        )
        if ckpt_every and (it + 1) % ckpt_every == 0:
            tmp = f"{ckpt_path}.tmp"
            torch.save(
                {
                    "version": CKPT_VERSION,
                    "it": it + 1,
                    "net": net.state_dict(),
                    "opt": opt.state_dict(),
                    "np_rng": rng.bit_generator.state,
                    "perm_rng": perm_gen.get_state(),
                    "weights": None if weights is None else weights.tolist(),
                    "settings": settings,
                    "start_ids": [s.spec["id"] for s in starts],
                },
                tmp,
            )
            Path(tmp).replace(ckpt_path)
    if targets is not None:
        targets.close()
    print("DONE", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen", help="record search decisions + playout features")
    g.add_argument("--fights", required=True, help="pilot_human_decks extract JSONL")
    g.add_argument("--out-dir", required=True)
    g.add_argument("--holdout", nargs="*", help="fight JSONL files whose runs are never trained on")
    g.add_argument(
        "--holdout-encounter", nargs="*", help="also hold out every run with this encounter"
    )
    g.add_argument("--rooms", default="monster,elite,boss")
    g.add_argument("--max-monster", type=int, default=0, help="subsample monster fights (0 = all)")
    g.add_argument("--sample-seed", type=int, default=0)
    g.add_argument("--reps", type=int, default=1)
    g.add_argument(
        "--rep0", type=int, default=50, help="seed rep offset (evaluation uses reps 0/1)"
    )
    g.add_argument("--sims", type=int, default=1000)
    g.add_argument("--boss-sims", type=int, default=4000)
    g.add_argument("--clair", action="store_true", help="clairvoyant teacher (STS2_RUNAI_HONEST=0)")
    g.add_argument("--procs", type=int, default=40)
    g.add_argument(
        "--fight-timeout-s",
        type=float,
        default=1800,
        help="wall-clock budget per fight (all reps); its decisions so far are kept (0 = none)",
    )
    g.add_argument("--per-shard", type=int, default=20000)
    g.add_argument(
        "--feature-version",
        type=int,
        choices=(1, 2, 3, 4),
        default=1,
        help="playout features to record",
    )
    g.add_argument(
        "--allow-bench-overlap",
        action="store_true",
        help="do NOT drop combat-bench runs (ppo.combat_bench.holdout); warns and records it",
    )
    gr = sub.add_parser(
        "gen-replay", help="gen from replayed human positions (pilot or human-turn starts)"
    )
    gr.add_argument("--starts", required=True, help="JSONL of {file, combat_id, turn, mode}")
    gr.add_argument("--collect", required=True, help="the `ppo.turnplan.pro collect` dir")
    gr.add_argument("--out-dir", required=True)
    gr.add_argument("--sims", type=int, default=4000)
    gr.add_argument("--boss-sims", type=int, default=4000)
    gr.add_argument("--feature-version", type=int, choices=(1, 2, 3, 4), default=1)
    gr.add_argument("--env", action="append", default=[], metavar="K=V", help="engine env")
    gr.add_argument("--procs", type=int, default=24)
    gr.add_argument("--per-shard", type=int, default=20000)
    gr.add_argument("--sample-seed", type=int, default=0)
    t = sub.add_parser("train", help="fit the playout policy, write its JSON weights")
    t.add_argument("--data", nargs="+", required=True, help="gen output dirs")
    t.add_argument("--out", required=True)
    t.add_argument("--hidden", type=int, default=32)
    t.add_argument("--emb-dim", type=int, default=16)
    t.add_argument("--epochs", type=int, default=8)
    t.add_argument("--batch", type=int, default=4096)
    t.add_argument("--lr", type=float, default=3e-3)
    t.add_argument("--wd", type=float, default=1e-4)
    t.add_argument("--val-frac", type=float, default=0.05)
    t.add_argument("--temperature", type=float, default=1.0)
    t.add_argument(
        "--target-mix",
        type=float,
        default=0.2,
        help="weight of the root visit distribution in the target; the rest is the "
        "search's recommended action (1.0 = visits only)",
    )
    t.add_argument(
        "--monster-emb",
        action="store_true",
        help="v2 data: learn the per-species encounter embedding",
    )
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="cuda")
    t.add_argument(
        "--init",
        help="start from this net (JSON; same feature version): its weights and key "
        "vocabulary, new data keys appended at zero; --hidden/--emb-dim are ignored",
    )
    t.add_argument(
        "--kl-init",
        type=float,
        default=0.0,
        help="with --init: blend this share of the init net's own policy into the target",
    )
    rollout_starts.add_traj_parser(sub)
    r = sub.add_parser("rl", help="improve a playout policy by REINFORCE on playout outcomes")
    r.add_argument("--init", required=True, help="starting weights (JSON from train)")
    r.add_argument("--out", required=True)
    r.add_argument("--fights", required=True)
    r.add_argument("--holdout", nargs="*")
    r.add_argument("--holdout-encounter", nargs="*")
    r.add_argument("--eval-fights", nargs="*", default=[], help="held-out fight files to score")
    r.add_argument("--eval-reps", type=int, default=4)
    r.add_argument("--eval-every", type=int, default=10)
    r.add_argument("--rooms", default="boss,elite")
    r.add_argument("--n-states", type=int, default=12000)
    r.add_argument("--iters", type=int, default=200)
    r.add_argument("--batch-states", type=int, default=512)
    r.add_argument("--k", type=int, default=8, help="playouts per start (the baseline group)")
    r.add_argument("--minibatch", type=int, default=16384)
    r.add_argument("--lr", type=float, default=3e-4)
    r.add_argument("--ent", type=float, default=0.01)
    r.add_argument("--w-hp", type=float, default=0.3, help="win reward = 1 + w_hp * hp left")
    r.add_argument(
        "--w-removed", type=float, default=0.2, help="loss reward = w * enemy HP removed"
    )
    r.add_argument(
        "--reward",
        choices=("shaped", "search"),
        default="shaped",
        help="shaped: the --w-hp / --w-removed reward; search: the value the deployed search "
        "backs up for the playout (sts2_mcts::playout_return)",
    )
    r.add_argument(
        "--starts",
        nargs="+",
        default=["fight"],
        metavar="MODE [TRAJ ...]",
        help="'fight' (each bank fight's opening state) or 'mid TRAJ.jsonl [...]' (the "
        "pilot's recorded decision points, `traj`)",
    )
    r.add_argument(
        "--view",
        choices=("raw", "belief"),
        help="raw: playouts see the true draw order; belief: each playout starts from the "
        "search's honest view (belief_root). Default raw for fight starts, belief for mid",
    )
    r.add_argument(
        "--mid-per-traj", type=int, default=0, help="at most N decision points per trajectory"
    )
    r.add_argument(
        "--contested-k",
        type=int,
        default=0,
        help="pre-pass playouts per start; sample starts by max(p(1-p), floor) (0 = uniform)",
    )
    r.add_argument("--contested-floor", type=float, default=0.02)
    r.add_argument(
        "--contested-every", type=int, default=0, help="redo the pre-pass every N its (0 = once)"
    )
    r.add_argument(
        "--sb-beta",
        type=float,
        default=0.0,
        help="weight of the simulation-balancing term: per playout, SB weighs beta * |V* - V^| "
        "times what REINFORCE weighs (both unit-variance advantages); the metrics log both "
        "gradient norms (g_rf, g_sb)",
    )
    r.add_argument("--sb-target", choices=("pilot", "search"), default="pilot")
    r.add_argument(
        "--sb-search-sims",
        type=int,
        default=16000,
        help="--sb-target search: sims of each of the TWO searches per start (one picks the "
        "move, an independent one prices it, so V* carries no winner's curse)",
    )
    r.add_argument("--sb-cache", help="--sb-target search cache (default <out>.sbcache.jsonl)")
    r.add_argument("--sb-procs", type=int, default=8)
    r.add_argument(
        "--feature-version",
        type=int,
        choices=(1, 2, 3, 4),
        help="upgrade a v1 --init to v2 or v4 (zero-padded: same policy at iteration 0) or to "
        "v3 (relabeled: same weights on the displayed-incoming / form-aware inputs)",
    )
    r.add_argument(
        "--monster-emb",
        action="store_true",
        help="give a v2 net (--feature-version 2, or a v2 --init) the encounter embedding: one "
        "zero row per species the engine knows, summons included; kept when --init has one",
    )
    r.add_argument(
        "--env",
        nargs="*",
        help="engine env K=V this run scores playouts under (and the --sb-target search "
        "workers search under); inherited STS2_RUNAI_* / STS2_ROLLOUT_* vars are cleared",
    )
    r.add_argument("--overrides", help="STS2_RUNAI_MCTS_OVERRIDES for this run's value function")
    r.add_argument(
        "--allow-env-mismatch",
        action="store_true",
        help="--starts mid: warn instead of refusing trajectories scored under another value env",
    )
    r.add_argument("--no-early-end", action="store_true")
    r.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature written to --out. The RL net plays best at its training "
        "temperature 1.0: at 0.5 its cut in boss deaths shrank from 6.1pp to 4.9pp "
        "(GH #777).",
    )
    r.add_argument("--metrics", help="per-iteration metrics JSONL (default <out>.metrics.jsonl)")
    r.add_argument(
        "--ckpt-every",
        type=int,
        help="write <out>.ckpt.pt every N iterations (default --eval-every; 0 = never)",
    )
    r.add_argument("--resume", action="store_true", help="continue from <out>.ckpt.pt")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--device", default="cuda")
    r.add_argument(
        "--allow-bench-overlap",
        action="store_true",
        help="do NOT drop combat-bench runs (ppo.combat_bench.holdout); warns and records it",
    )
    args = ap.parse_args(argv)
    return {
        "gen": gen,
        "gen-replay": gen_replay,
        "train": train,
        "traj": rollout_starts.traj,
        "rl": rl,
    }[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
