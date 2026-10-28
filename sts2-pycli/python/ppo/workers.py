"""Process-pool run collector (plan §2: process workers, not threads).

Each worker process owns its own `sts2_rs` import and plays whole runs;
a hung run is killed at the per-run wall clock without poisoning the pool
(known engine hangs — fire_late_effects — motivate process isolation).

The same harness serves G2 (throughput), G3 (random-policy win rate),
BC recording (M-P4) and, later, the M-P3 PPO collector: the unit of work
is `play_one(seed, policy_spec)` returning a JSON-serializable episode
record.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import random
import time
from typing import Any

from ppo.wheel_guard import python_source_digest, wheel_identity

# Take the digest of the first-party source now, at import (a spawn worker's
# first act), so it describes what this process loaded.
python_source_digest()

# Stall window (seconds): how long `play_batch` waits with no completion
# before it even considers a still-running game hung.
DEFAULT_RUN_TIMEOUT_S = 600.0
# Per-game budget (seconds): a running game is only abandoned once it has run
# this long itself (GH #763). One combat is a single `env.step`, so a slow boss
# fight shows no Python-visible progress for its whole length: the slowest game
# seen took 1,198 s on the 2026-10-12 engine and 226-255 s after PR #761.
DEFAULT_GAME_TIMEOUT_S = 1800.0

# Per-job start times (time.time(); 0.0 = not started), shared with the pool
# workers through the initializer so the watchdog can age each running game.
_JOB_STARTS: Any = None


def _reached_second_boss(rm: Any) -> bool | None:
    """Whether the run entered the A10 DoubleBoss second-boss room, i.e. beat
    the first act-3 boss: an A9 win read off an A10 run (A10 = A9 + DoubleBoss).
    None when the current map has no second boss (below A10, or before act 3)."""
    m = rm.full_map()
    sb = m and m.get("second_boss")
    if not sb:
        return None
    for row in m["rows"]:
        for node in row:
            if (node["col"], node["row"]) == (sb["col"], sb["row"]):
                return bool(node["visited"] or node["current"])
    return None


def _init_job_starts(starts: Any) -> None:
    global _JOB_STARTS
    _JOB_STARTS = starts


def _worker_at(i: int, args: tuple) -> dict[str, Any]:
    """`_worker`, stamping job `i`'s start time first (see `_JOB_STARTS`)."""
    if _JOB_STARTS is not None:
        _JOB_STARTS[i] = time.time()
    return _worker(args)


# Per-process NetPolicy cache: pool workers play many seeds per iteration;
# reload the checkpoint only when its path changes.
_NET_POLICY: dict[str, Any] = {}


def _net_policy(ckpt_path: str | None):
    from ppo.net_policy import NetPolicy

    key = ckpt_path or "<random-init>"
    if _NET_POLICY.get("key") != key:
        _NET_POLICY["key"] = key
        _NET_POLICY["policy"] = NetPolicy(ckpt_path)
    return _NET_POLICY["policy"]


def play_one(
    seed: int,
    policy: str = "random",
    sample_seed: int | None = None,
    ascension: int = 0,
    character: str = "Ironclad",
    record_decisions: bool = False,
    ckpt_path: str | None = None,
    sims: int | None = None,
    boss_sims: int | None = None,
    combat: str = "mcts",
    surrogate_path: str | None = None,
    max_steps: int | None = None,
    search: dict[str, Any] | None = None,
    act1: str = "overgrowth",
) -> dict[str, Any]:
    """Play one full run inside the current process.

    `policy`:
      - 'random': uniform-random over candidates at every yielded decision —
        equivalent to a fresh (near-zero-logit) policy net; the G3 probe.
      - 'expert': the deployed agent makes every decision (parity / BC).
      - 'net': sample from the RunPolicyNet at `ckpt_path` (None =
        random-init) — the PPO collection mode. The row carries a
        'transitions' list (featurized arrays + idx/logp/value/metrics),
        which is pool-picklable but NOT JSONL-serializable; `play_batch`
        strips it from the JSONL sink.
      - 'net_greedy': same net, argmax instead of sampling — diagnostic
        (separates net quality from sampling entropy, e.g. gate G4).

    `sample_seed` seeds the action-sampling RNG, decoupled from the game
    seed (fork CollectionJob pattern); defaults to `seed`.

    `sims` / `boss_sims` override the env's combat MCTS budget (None keeps
    the `PpoRunEnv` defaults) — the WPA review (`ppo.wpa`) threads these so
    its counterfactual replays run under the same budget as the base run.

    `combat` selects the combat backend: "mcts" (play every fight) or
    "surrogate" (fast-forward each fight through the net at `surrogate_path`).
    See `docs/ppo-screen-playbook.md` — "surrogate" is a screening backend,
    not a campaign one.

    `policy='net_search'` (GH #452) wraps the net in the Gumbel depth-1
    search teacher (`ppo.search_policy`); `search` carries its knobs
    (`SearchConfig` field names, plus an optional `shaping` sub-dict of
    `ShapingConfig` field names). The row gains `search_stats`, and with
    `record_decisions` each decision row carries the distillation targets
    `pi` (improved policy) and `search_value`. `max_steps` overrides the
    env's step cap (bounded smokes).

    `act1` (GH #545) selects the Act-1 model — "overgrowth", "underdocks" or
    "roll"; see `PpoRunEnv.act1`. The row records the RESOLVED token under
    `"act1"` (never "roll" itself), so a "roll" campaign's mixture can be
    read back from `trainer.behavior_stats`' `act1_frac/*` keys.
    """
    from ppo.env import Decision, PpoRunEnv, Terminal

    rng = random.Random(seed if sample_seed is None else sample_seed)
    env_kwargs: dict[str, Any] = {}
    if sims is not None:
        env_kwargs["sims"] = sims
    if boss_sims is not None:
        env_kwargs["boss_sims"] = boss_sims
    if combat != "mcts":
        env_kwargs["combat"] = combat
        env_kwargs["surrogate_path"] = surrogate_path
    if act1 != "overgrowth":
        env_kwargs["act1"] = act1
    # Expert mode drives every decision with the HeuristicRunAgent; policy
    # modes (random/net/net_greedy) use the combat-only MctsCombatAgent
    # (GH #220). Neither needs a net file or env vars.
    env = PpoRunEnv(
        seed=seed,
        ascension=ascension,
        character=character,
        expert=(policy == "expert"),
        **env_kwargs,
    )
    if max_steps is not None:
        env.max_steps = max_steps
    t0 = time.time()
    decisions: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []

    def record(d: Decision, chosen: int) -> None:
        if record_decisions:
            decisions.append(
                {
                    "step": d.step_index,
                    "surface": d.surface,
                    "phase": d.phase,
                    "n_actions": len(d.actions),
                    "actions": d.actions,
                    "obs": d.obs,
                    "metrics": d.metrics,
                    "chosen": chosen,
                }
            )

    terminal: Terminal | None = None
    search_stats: dict[str, Any] | None = None
    forced_stats: dict[str, Any] | None = None
    if policy == "expert":
        for out in env.expert_run():
            if isinstance(out, Decision):
                assert out.expert_idx is not None
                record(out, out.expert_idx)
            else:
                terminal = out
                break
    elif policy == "random":
        out = env.reset()
        while isinstance(out, Decision):
            chosen = rng.randrange(len(out.actions))
            record(out, chosen)
            out = env.step(chosen)
        terminal = out
    elif policy in ("net", "net_greedy"):
        import numpy as np

        # GH #399: the coarse surface lumps every deck-mutation decision into
        # `card-pick` with the ~10-14/run card rewards, so a removal count of 0
        # is invisible in `dec_per_run/card-pick`. Record the state token's
        # own fine surface alongside it — `trainer.behavior_stats` keys the
        # per-surface counters off both, and playbook §0.1's pre-flight reads
        # the fine ones. Imported here (not at module scope) because
        # `ppo.encoder` pulls in numpy + the `sts2_rs` PyO3 wheel (via
        # `ppo.vocab`) and the module-level import is shared with callers
        # that need neither.
        from ppo.encoder import fine_surface

        pol = _net_policy(ckpt_path)
        nrng = np.random.default_rng(seed if sample_seed is None else sample_seed)
        greedy = policy == "net_greedy"
        out = env.reset()
        while isinstance(out, Decision):
            idx, logp, value, feat = pol.sample(
                out.obs, out.actions, out.metrics, nrng, greedy=greedy
            )
            transitions.append(
                {
                    "feat": feat,
                    "action_idx": idx,
                    "old_logprob": logp,
                    "value": value,
                    "metrics": out.metrics,
                    "surface": out.surface,
                    "fine_surface": fine_surface(out.actions),
                    "n_actions": len(out.actions),
                }
            )
            record(out, idx)
            out = env.step(idx)
        terminal = out
    elif policy == "net_search":
        import numpy as np

        from ppo.rewards import ShapingConfig
        from ppo.search_policy import SearchConfig, SearchPolicy

        search_kwargs = dict(search or {})
        shaping = ShapingConfig(**search_kwargs.pop("shaping", {}))
        spol = SearchPolicy(_net_policy(ckpt_path), SearchConfig(**search_kwargs), shaping)
        nrng = np.random.default_rng(seed if sample_seed is None else sample_seed)
        search_stats = {"searched": 0, "overrides": 0, "by_surface": {}}
        out = env.reset()
        while isinstance(out, Decision):
            idx, diag = spol.choose(env, out, nrng)
            search_stats["searched"] += 1
            surf = search_stats["by_surface"].setdefault(out.surface, [0, 0])
            surf[1] += 1
            if diag["override"]:
                search_stats["overrides"] += 1
                surf[0] += 1
            record(out, idx)
            if record_decisions:
                # Distillation targets (GH #452): improved policy over the
                # candidate list + the search's root-value estimate, both on
                # the shaped-return scale ppo.bc trains against.
                decisions[-1]["pi"] = diag["pi"]
                decisions[-1]["search_value"] = diag["search_value"]
            out = env.step(idx)
        terminal = out
    elif policy == "net_forced":
        # The forced-pick counterfactual (`ppo.forced_pick`): greedy net, with
        # engine-card picks at card rewards forced per `search["arm"]`.
        from ppo.forced_pick import ForcedPick

        pol = _net_policy(ckpt_path)
        forcer = ForcedPick(seed=seed, **(search or {}))
        out = env.reset()
        while isinstance(out, Decision):
            idx = forcer.choose(pol, out)
            record(out, idx)
            out = env.step(idx)
        terminal = out
        forced_stats = forcer.stats()
    else:
        raise ValueError(f"unknown policy {policy!r}")
    if terminal is None:  # expert_run always ends with a Terminal
        raise RuntimeError("run ended without a Terminal record")

    counts: dict[str, int] = {}
    for _, phase, surface, klass in env.resolution_log:
        key = f"{phase}/{surface}/{klass}"
        counts[key] = counts.get(key, 0) + 1

    return {
        "seed": seed,
        "policy": policy,
        "ascension": ascension,
        "character": character,
        "act1": env.rm.act1,
        "outcome": terminal.outcome,
        "won": terminal.won,
        "steps": terminal.steps,
        "n_decisions": terminal.decisions,
        "final": terminal.metrics,
        "reached_second_boss": _reached_second_boss(env.rm),
        "wall_s": round(time.time() - t0, 2),
        "resolution_counts": counts,
        "decisions": decisions,
        "transitions": transitions,
        # The sts2_rs build this process actually ran (`ppo.wheel_guard`): the
        # trainer refuses a batch whose workers ran anything but the run's own.
        "wheel": wheel_identity(),
        # And the first-party Python it imported (digest taken at import, at the
        # top of this module): the pin covers only the extension, so a pull
        # mid-campaign would change this.
        "py_src": python_source_digest(),
        **({"search_stats": search_stats} if search_stats is not None else {}),
        **({"forced_stats": forced_stats} if forced_stats is not None else {}),
    }


def _worker(
    # POSITIONAL tuple, not kwargs (pool.apply_async pickles this across the
    # process boundary) — its arity has moved twice now (most recently GH
    # #545's `act1`). A test or caller building this tuple by hand has to
    # match the unpack below field-for-field; when adding a field, append it
    # LAST and update every hand-built tuple literal (e.g.
    # `tests/test_ppo_workers_watchdog.py`), or it fails with a plain
    # "not enough values to unpack" far from the actual cause.
    args: tuple[
        int,
        str,
        int,
        str,
        bool,
        str | None,
        int | None,
        int | None,
        str,
        str | None,
        dict | None,
        str,
        int | None,
    ],
) -> dict[str, Any]:
    (
        seed,
        policy,
        ascension,
        character,
        record_decisions,
        ckpt_path,
        sims,
        boss_sims,
        combat,
        surrogate_path,
        search,
        act1,
        sample_seed,
    ) = args
    try:
        return play_one(
            seed,
            policy=policy,
            sample_seed=sample_seed,
            ascension=ascension,
            character=character,
            record_decisions=record_decisions,
            ckpt_path=ckpt_path,
            sims=sims,
            boss_sims=boss_sims,
            combat=combat,
            surrogate_path=surrogate_path,
            search=search,
            act1=act1,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:
        # `BaseException`, not `Exception`, on purpose (GH #335): a Rust panic
        # inside the PyO3 engine surfaces as `pyo3_runtime.PanicException`,
        # which derives from `BaseException`. An `except Exception` here (and
        # the identical guard in `multiprocessing/pool.py`) lets it through, so
        # the worker process dies at `_bootstrap` and its in-flight
        # `AsyncResult` never becomes ready — the batch then hangs until
        # `play_batch`'s watchdog fires (after `DEFAULT_GAME_TIMEOUT_S`).
        # Booking it as an `outcome="error"` row costs one run instead.
        # `KeyboardInterrupt` / `SystemExit` still propagate (re-raised above).
        return {"seed": seed, "policy": policy, "outcome": "error", "won": False, "error": repr(e)}


def play_batch(
    seeds: list[int],
    policy: str = "random",
    ascension: int = 0,
    character: str = "Ironclad",
    procs: int | None = None,
    record_decisions: bool = False,
    run_timeout_s: float = DEFAULT_RUN_TIMEOUT_S,
    game_timeout_s: float = DEFAULT_GAME_TIMEOUT_S,
    out_path: str | None = None,
    ckpt_path: str | None = None,
    ascensions: list[int] | None = None,
    sims: int | None = None,
    boss_sims: int | None = None,
    sims_list: list[int | None] | None = None,
    boss_sims_list: list[int | None] | None = None,
    sample_seeds: list[int | None] | None = None,
    combat: str = "mcts",
    surrogate_path: str | None = None,
    mp_context: str = "spawn",
    search: dict[str, Any] | None = None,
    act1: str = "overgrowth",
) -> list[dict[str, Any]]:
    """Play `seeds` across a spawn-context process pool; one row per run.

    The returned list is ordered by SUBMISSION index (same order as `seeds`)
    regardless of completion order — load-bearing for the trainer, which
    concatenates transitions row-by-row and seeds the minibatch shuffle off
    that order. Rows stream to `out_path` (JSONL) in COMPLETION order instead,
    so a straggler never blocks flushing the runs that already finished and a
    partial batch is never lost (label-campaign fault-isolation lesson); the
    non-JSON 'transitions' payload (policy='net') is stripped from the sink but
    kept in the returned rows.

    The watchdog abandons the still-pending runs as outcome='timeout' (and
    terminates the pool, so a native infinite-combat loop that ignores Python
    signals stops burning cores) only when BOTH hold:
      - no run has completed for `run_timeout_s` (a stall), and
      - every run still executing has itself run longer than `game_timeout_s`.
    The second condition is GH #763: with only the first, a slow game that is
    still progressing was dropped as soon as it outlived every other game by
    `run_timeout_s` (2.8% of past Ironclad iterations lost their slowest game
    that way). A timeout row carries `age_s`, how long that game had run (None
    if it never started). A worker that dies hard (e.g. segfault) is recorded
    as outcome='error' rather than crashing the batch.

    `ascensions`, when given, is a per-seed ascension list (same length as
    `seeds`) used by the mixed-ascension curriculum; otherwise the scalar
    `ascension` applies to every seed. Each row records its own ascension.

    `sims` / `boss_sims` override the combat MCTS budget for every seed (None
    keeps the env's eval-cheap 1000/4000) — used by the combat sim-count sweep
    to isolate combat headroom at a fixed run-level policy. `sims_list` /
    `boss_sims_list` give a PER-SEED override (same length as `seeds`; entry
    None = env default) and take precedence over the scalar — used by the
    mixed-ascension curriculum to run high-ascension games under a stronger
    combat budget (the measured high-ascension lever).

    `combat` / `surrogate_path` select the combat backend for every seed; see
    `play_one`. `sims`/`boss_sims` are inert under `combat="surrogate"` (there
    is no search to budget), so a screen that switches backends should stop
    quoting them.

    `act1` (GH #545) selects the Act-1 model for every seed; see `play_one`.

    `sample_seeds` gives a PER-SEED action-sampling seed (same length as
    `seeds`; entry None = derive from the game seed, the legacy behavior) —
    used by PLR level replay, where a replayed game seed must NOT replay the
    recorded action stream.
    """
    procs = procs or max(1, (os.cpu_count() or 2) - 2)
    if ascensions is None:
        ascensions = [ascension] * len(seeds)
    elif len(ascensions) != len(seeds):
        raise ValueError(f"ascensions len {len(ascensions)} != seeds len {len(seeds)}")
    sims_seq = sims_list if sims_list is not None else [sims] * len(seeds)
    boss_sims_seq = boss_sims_list if boss_sims_list is not None else [boss_sims] * len(seeds)
    if len(sims_seq) != len(seeds) or len(boss_sims_seq) != len(seeds):
        raise ValueError("sims_list / boss_sims_list length must match seeds")
    sample_seq = sample_seeds if sample_seeds is not None else [None] * len(seeds)
    if len(sample_seq) != len(seeds):
        raise ValueError(f"sample_seeds len {len(sample_seq)} != seeds len {len(seeds)}")
    # `spawn` in production (each worker owns a clean sts2_rs import); tests
    # pass `fork` so a monkeypatched `play_one` (e.g. an injected hang) is
    # inherited by the workers, letting the stall watchdog be exercised without
    # a real engine.
    ctx = mp.get_context(mp_context)
    jobs = [
        (
            s,
            policy,
            a,
            character,
            record_decisions,
            ckpt_path,
            sm,
            bsm,
            combat,
            surrogate_path,
            search,
            act1,
            ssd,
        )
        for s, a, sm, bsm, ssd in zip(seeds, ascensions, sims_seq, boss_sims_seq, sample_seq)
    ]
    # Results are STORED by submission index (deterministic order for the
    # learner: the trainer concatenates transitions row-by-row and the update
    # shuffle is seeded off that order, so completion-order returns would make
    # training nondeterministic). The JSONL sink, by contrast, is written in
    # COMPLETION order — a straggler no longer blocks flushing the runs that
    # already finished, so a crash mid-batch never loses completed rows.
    results: list[dict[str, Any] | None] = [None] * len(jobs)
    sink = open(out_path, "w", encoding="utf-8") if out_path else None

    def _emit(i: int, row: dict[str, Any]) -> None:
        results[i] = row
        if sink:
            sink.write(json.dumps({k: v for k, v in row.items() if k != "transitions"}) + "\n")
            sink.flush()

    starts = ctx.Array("d", len(jobs), lock=False)
    try:
        with ctx.Pool(processes=procs, initializer=_init_job_starts, initargs=(starts,)) as pool:
            pending: dict[Any, tuple[int, int]] = {}
            for i, job in enumerate(jobs):
                pending[pool.apply_async(_worker_at, (i, job))] = (i, job[0])
            remaining = set(pending)
            # Stall watchdog (see the docstring): `last_progress` resets on
            # every completion; it is initialised to now so the first
            # completion has the full window to arrive.
            last_progress = time.time()
            while remaining:
                just_done = {ar for ar in remaining if ar.ready()}
                for ar in just_done:
                    i, seed = pending[ar]
                    try:
                        row = ar.get()
                    except Exception as e:  # worker died hard (e.g. segfault)
                        row = {
                            "seed": seed,
                            "policy": policy,
                            "outcome": "error",
                            "won": False,
                            "error": repr(e),
                        }
                    _emit(i, row)
                remaining -= just_done
                if just_done:
                    last_progress = time.time()
                elif remaining:
                    now = time.time()
                    ages = [
                        now - starts[pending[ar][0]]
                        for ar in remaining
                        if starts[pending[ar][0]] > 0
                    ]
                    if now - last_progress > run_timeout_s and all(
                        a > game_timeout_s for a in ages
                    ):
                        # A stall, and every game still running is past its own
                        # budget -> hung. Record the pending runs as 'timeout'
                        # and terminate the pool so the spinning workers stop
                        # burning cores now rather than at scope exit.
                        for ar in remaining:
                            i, seed = pending[ar]
                            _emit(
                                i,
                                {
                                    "seed": seed,
                                    "policy": policy,
                                    "outcome": "timeout",
                                    "won": False,
                                    "age_s": round(now - starts[i], 1) if starts[i] > 0 else None,
                                },
                            )
                        remaining.clear()
                        pool.terminate()
                    else:
                        time.sleep(0.1)
    finally:
        if sink:
            sink.close()
    return [r for r in results if r is not None]
