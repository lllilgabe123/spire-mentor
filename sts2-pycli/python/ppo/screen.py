"""Two-arm paired screen for PPO run-policy experiments.

A full campaign costs 31-82 wall-clock hours (see the table in
`docs/ppo-screen-playbook.md`), which makes trying an idea impossible. This
driver runs the same experiment small and cheap enough to answer *"is this
worth 31 hours?"* in under an hour, and scores it the only way the campaign
playbook accepts a verdict: a **paired** comparison on a fresh seed panel.

Two things it does that a pair of hand-launched `ppo.trainer` runs does not:

1. **Pairs the arms.** Both arms get the same `--seed-base`, so the trainer's
   `seed_base + it*games + g` derivation and its seed-derived ascension draw
   hand them an identical seed x ascension panel every iteration. The final
   eval panel is shared too, so the report is a McNemar over discordant seeds
   rather than a difference of two independent rates.
2. **Scores on a held-out panel, not the training curve.** Training win rate
   is measured under exploration, on the training distribution — the campaign
   playbook's §2 says plainly that it is a liveness signal, not a gate. The
   primary number here is a greedy replay of both final nets over seeds
   neither arm trained on.

What this is NOT
----------------
A promotion gate. A screen runs at a reduced combat budget, on fewer games,
for fewer iterations, over a narrower ascension band — four deliberate
distortions of the world the champion is measured in. It decides **what to
spend 31 hours on**, never what to ship; promotion keeps following
`docs/ppo-champion-training-playbook.md` §2 in full.

It is also worthless until calibrated. Run it on changes whose campaign
outcome is already known (the playbook lists three) before believing it on a
change whose outcome is not.

Usage
-----
    PYTHONPATH=python python -m ppo.screen \\
        --name <idea> --character Ironclad \\
        --init-ckpt <the champion .pt named in data/ppo/champions.toml> \\
        --arm-b-features <feature> \\
        --out /tmp/screen-<idea>

(`--arm-b-features` takes a name from `ppo.schema.KNOWN_FEATURES`, which is
currently EMPTY: its one entry, `boss`, graduated to an unconditional input in
GH #431 — see that module's docstring. Arms differing only by `--overrides`
work without any feature at all.)

Every arm's stdout, `stats.jsonl` and checkpoints land under `--out/<arm>/`,
and the eval rows under `--out/eval-<arm>.jsonl`, so a killed screen can be
inspected (and its finished arm reused) rather than redone.

Like `ppo.trainer`, the screen refuses to start unless the imported `sts2_rs`
is a fresh PGO wheel (`ppo.wheel_guard`; `just sim-engine-pgo`). It checks
before launching anything, so a wrong wheel fails in seconds rather than one
arm in; each arm's trainer re-checks. `--allow-plain-wheel` exports
`STS2_ALLOW_PLAIN_WHEEL=1`, which the arms' trainers inherit. The screen pins
the wheel once, under `--out/sts2_rs-pin/`, and both arms' trainers and evals
inherit that pin, so the two arms cannot end up on different wheels; an eval
whose runs report another build aborts the screen.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .wheel_guard import (
    ALLOW_PLAIN_ENV,
    PIN_DIRNAME,
    prepare_training_wheel,
    wheel_identity,
    worker_wheel_mismatch,
)

# Screen preset. Every value here is a deliberate cut from the campaign recipe
# with a measured cost; `docs/ppo-screen-playbook.md` carries the rationale and
# the sims-vs-wall-clock curve these came from.
PRESET: dict[str, Any] = {
    "sims": 100,
    "boss_sims": 300,
    "games_per_iter": 128,
    "iterations": 80,
    "ascension_weights": "0:1,1:1,2:1,3:1,4:1",
    "lr_anneal_final": 0.3,
    "entropy_coef": 0.02,
    "entropy_decay_start": 30,
    "entropy_decay_steps": 30,
    "entropy_coef_final": 0.005,
    "save_every": 20,
}

# Eval panel: disjoint from the trainer's `seed_base` block by construction
# (the trainer consumes `seed_base .. seed_base + iterations*games`), and
# disjoint from the champion gate panels, which live at 7000-7199.
EVAL_SEED_BASE = 900_000_000


@dataclass
class Arm:
    name: str
    features: str = ""
    overrides: dict[str, Any] = field(default_factory=dict)
    # Extra process env for this arm's training AND eval — the seam for
    # env-var-gated engine behavior (e.g. the potion OC/swap stack, plan
    # 2026-08-20-ppo-potion-bootstrap P3). Trainer flags cannot reach
    # those; the collection workers inherit the trainer's environ.
    env_vars: dict[str, str] = field(default_factory=dict)

    def label(self) -> str:
        bits = [self.name]
        if self.features:
            bits.append(f"features={self.features}")
        bits += [f"{k}={v}" for k, v in sorted(self.overrides.items())]
        bits += [f"env:{k}={v}" for k, v in sorted(self.env_vars.items())]
        return " ".join(bits)

    def config_key(self) -> tuple:
        """What actually differs between the arms — everything but the name.

        Compared rather than the label, which always differs (it leads with
        "a" / "b") and so would make an A/A run undetectable.
        """
        return (
            self.features,
            tuple(sorted(self.overrides.items())),
            tuple(sorted(self.env_vars.items())),
        )


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p over the discordant counts `(b, c)`.

    Same test the champion gate uses, so a screen result and a gate result are
    read the same way. Returns 1.0 when nothing is discordant.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0**n)
    return min(1.0, 2.0 * tail)


def build_cmd(arm: Arm, cfg: dict[str, Any], save_path: str) -> list[str]:
    merged = dict(cfg)
    merged.update(arm.overrides)
    merged["save_path"] = save_path
    cmd = [sys.executable, "-m", "ppo.trainer"]
    for key, value in sorted(merged.items()):
        if value is None or value == "":
            continue
        cmd += [f"--{key.replace('_', '-')}", str(value)]
    # Declared, not just exported: `PPOConfig.features` is recorded into every
    # checkpoint's saved cfg, and `train()` refuses to start if it disagrees
    # with the schema the interpreter actually imported.
    cmd += ["--features", arm.features]
    return cmd


def _trained_wheel(arm_dir: str) -> dict | None:
    """The sts2_rs identity the arm's last logged iteration's workers ran
    (`stats.jsonl` `wheel`), or None when it predates that record."""
    try:
        with open(os.path.join(arm_dir, "main.stats.jsonl"), encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    except (OSError, ValueError):
        return None
    return rows[-1].get("wheel") if rows else None


def _eval_path(out_dir: str, arm_name: str) -> str:
    return os.path.join(out_dir, f"eval-{arm_name}.jsonl")


def _eval_wheel(path: str) -> dict | None:
    """The sts2_rs identity an eval file's runs reported (their rows' `wheel`),
    or None when it predates that record."""
    try:
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    except (OSError, ValueError):
        return None
    return next((r["wheel"] for r in rows if "wheel" in r), None)


def run_arm(arm: Arm, cfg: dict[str, Any], out_dir: str, dry_run: bool) -> str:
    """Train one arm; returns the path to its final checkpoint."""
    arm_dir = os.path.join(out_dir, arm.name)
    os.makedirs(arm_dir, exist_ok=True)
    save_path = os.path.join(arm_dir, "main")
    ckpt = f"{save_path}.pt"
    cmd = build_cmd(arm, cfg, save_path)
    env = dict(os.environ, STS2_PPO_FEATURES=arm.features, **arm.env_vars)

    if os.path.exists(ckpt):
        # Skip-if-exists, so a screen killed after arm A does not redo it.
        print(f"[{arm.name}] reusing existing {ckpt}", flush=True)
        return ckpt
    if dry_run:
        print(f"[{arm.name}] STS2_PPO_FEATURES={arm.features!r} {shlex.join(cmd)}", flush=True)
        return ckpt

    log_path = os.path.join(arm_dir, "train.log")
    print(f"[{arm.name}] {shlex.join(cmd)}", flush=True)
    print(f"[{arm.name}] log -> {log_path}", flush=True)
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        raise SystemExit(f"[{arm.name}] trainer failed (rc={proc.returncode}); see {log_path}")
    print(f"[{arm.name}] trained in {(time.time() - t0) / 60:.1f} min", flush=True)
    return ckpt


def eval_arm(
    arm: Arm,
    ckpt: str,
    cfg: dict[str, Any],
    out_dir: str,
    seeds: int,
    dry_run: bool,
    wheel: dict | None = None,
) -> dict[int, bool]:
    """Greedy replay of `ckpt` over the shared eval panel → {seed: won}.

    Runs in a subprocess because the encoder schema is fixed at import time:
    arm B may have a different `N_CAT` than this process does. With `wheel`
    (the screen's pinned record), a fresh eval whose runs report another
    sts2_rs build aborts, after moving its file aside so that a re-run
    evaluates again instead of reusing it; an eval file from an earlier
    invocation is reused as is (`main` lists it in screen.json). Runs that
    ended in an error or a timeout are dropped from the result, so the pairing
    leaves their seeds out instead of scoring them as losses.
    """
    out_path = _eval_path(out_dir, arm.name)
    if dry_run:
        return {}
    fresh = not os.path.exists(out_path)
    if fresh:
        code = _EVAL_SNIPPET
        env = dict(
            os.environ,
            STS2_PPO_FEATURES=arm.features,
            **arm.env_vars,
            SCREEN_CKPT=ckpt,
            SCREEN_OUT=out_path,
            SCREEN_SEEDS=str(seeds),
            SCREEN_SEED_BASE=str(EVAL_SEED_BASE),
            SCREEN_CHARACTER=str(cfg["character"]),
            SCREEN_ASC_WEIGHTS=str(cfg["ascension_weights"]),
            SCREEN_SIMS=str(cfg["sims"]),
            SCREEN_BOSS_SIMS=str(cfg["boss_sims"]),
            SCREEN_COMBAT=str(cfg.get("combat", "mcts")),
            SCREEN_SURROGATE=str(cfg.get("surrogate_path", "")),
            SCREEN_PROCS=str(cfg.get("procs", 0)),
            # This module's own package root, so the child resolves `ppo.*`
            # from the checkout that launched it rather than from whatever a
            # relative path or an inherited PYTHONPATH happens to point at.
            SCREEN_PYTHON_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        proc = subprocess.run([sys.executable, "-c", code], env=env, check=False)
        if proc.returncode != 0:
            raise SystemExit(f"[{arm.name}] eval failed (rc={proc.returncode})")
    rows = [json.loads(line) for line in open(out_path, encoding="utf-8")]
    if fresh and wheel is not None:
        expected = wheel_identity(wheel)
        mismatch = worker_wheel_mismatch(rows, expected)
        if mismatch is not None:
            # Moved, not left: a re-run would otherwise find the file, take it
            # for an earlier invocation's eval and reuse it unchecked.
            aside = f"{out_path}.wheel-mismatch-{time.strftime('%Y%m%dT%H%M%S')}"
            Path(out_path).replace(aside)
            raise SystemExit(
                f"[{arm.name}] the eval's runs did not run the sts2_rs this screen pinned.\n"
                f"  pinned: {json.dumps(expected, sort_keys=True)}\n{mismatch}\n"
                f"Moved the eval to {aside}; re-running the screen evaluates this arm again."
            )
    failed = [r for r in rows if r.get("outcome") in _FAILED_OUTCOMES]
    if failed:
        print(
            f"[{arm.name}] WARNING: {len(failed)} of {len(rows)} eval run(s) ended in an "
            "error or a timeout; their seeds are left out of the pairing (a crash is not "
            "a loss). Check the eval before trusting the result.",
            file=sys.stderr,
            flush=True,
        )
    return {
        int(r["seed"]): bool(r["won"]) for r in rows if r.get("outcome") not in _FAILED_OUTCOMES
    }


# `play_batch` rows for a run that never finished. Scoring them as losses
# charges an engine crash to the arm (it once flipped the sign of a pooled
# `ppo.probe` delta), so they are left out of the pairing.
_FAILED_OUTCOMES = ("error", "timeout")


# Run in its own interpreter (see `eval_arm`). Kept as source rather than a
# module so the schema-fixing env var is unambiguously applied before any
# `ppo.*` import happens.
_EVAL_SNIPPET = """
import json, os, sys
sys.path.insert(0, os.environ["SCREEN_PYTHON_DIR"])
from ppo.trainer import PPOConfig, sample_ascensions
from ppo.workers import play_batch

n = int(os.environ["SCREEN_SEEDS"])
base = int(os.environ["SCREEN_SEED_BASE"])
seeds = [base + i for i in range(n)]
cfg = PPOConfig(
    ascension_weights=os.environ["SCREEN_ASC_WEIGHTS"],
    seed_base=base,
    games_per_iter=n,
)
# Same derivation as a training iteration, so the eval panel's ascension mix
# matches the curriculum instead of silently being all-A0.
ascensions = sample_ascensions(cfg, 0, n)
procs = int(os.environ["SCREEN_PROCS"]) or None
rows = play_batch(
    seeds,
    policy="net_greedy",
    ascension=0,
    ascensions=ascensions,
    character=os.environ["SCREEN_CHARACTER"],
    procs=procs,
    ckpt_path=os.environ["SCREEN_CKPT"],
    sims=int(os.environ["SCREEN_SIMS"]) or None,
    boss_sims=int(os.environ["SCREEN_BOSS_SIMS"]) or None,
    combat=os.environ["SCREEN_COMBAT"],
    surrogate_path=(os.environ["SCREEN_SURROGATE"] or None),
)
with open(os.environ["SCREEN_OUT"], "w") as f:
    for r, a in zip(rows, ascensions or [0] * n):
        f.write(json.dumps({
            "seed": r["seed"], "won": bool(r.get("won")), "ascension": a,
            "outcome": r.get("outcome"),
            "total_floor": (r.get("final") or {}).get("total_floor"),
            **({"wheel": r["wheel"]} if "wheel" in r else {}),
        }) + "\\n")
"""


def read_stats(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def tail_pooled(stats: list[dict], k: int) -> tuple[float, int]:
    """Win rate pooled over the last `k` iterations, and the run count."""
    rows = stats[-k:]
    wins = sum(r["win_rate"] * r["n_runs"] for r in rows)
    n = sum(r["n_runs"] for r in rows)
    return (wins / n if n else 0.0), n


def wall_hours(stats: list[dict]) -> dict[str, float]:
    """Training wall clock and its split, in hours. With `async_collect` the
    collect and train phases overlap, so `collect_h + train_h` exceeds `iter_h`;
    `iter_h` is the arm's real cost."""
    return {
        key: round(sum(r.get(field, 0.0) for r in stats) / 3600.0, 3)
        for key, field in (("iter_h", "iter_s"), ("collect_h", "collect_s"), ("train_h", "train_s"))
    }


def report(
    a: Arm,
    b: Arm,
    res_a: dict[int, bool],
    res_b: dict[int, bool],
    out_dir: str,
    dropped: dict[str, int] | None = None,
) -> dict:
    """Pair the two arms' evals on their shared seeds and write report.json.
    ``dropped`` is how many eval runs per arm ended in an error or a timeout
    (left out of the pairing), recorded as ``dropped_<arm>``."""
    shared = sorted(set(res_a) & set(res_b))
    n = len(shared)
    wins_a = sum(res_a[s] for s in shared)
    wins_b = sum(res_b[s] for s in shared)
    # b_only: A won, B lost. c_only: B won, A lost.
    b_only = sum(1 for s in shared if res_a[s] and not res_b[s])
    c_only = sum(1 for s in shared if res_b[s] and not res_a[s])
    p = mcnemar_exact(b_only, c_only)
    delta = (wins_b - wins_a) / n if n else 0.0

    out: dict[str, Any] = {
        "arm_a": a.label(),
        "arm_b": b.label(),
        "n_paired": n,
        "win_rate_a": round(wins_a / n, 4) if n else None,
        "win_rate_b": round(wins_b / n, 4) if n else None,
        "delta_pp": round(100.0 * delta, 2),
        "flips_a_only": b_only,
        "flips_b_only": c_only,
        "mcnemar_p": round(p, 6),
        **{f"dropped_{arm}": n_dropped for arm, n_dropped in (dropped or {}).items()},
    }

    print("\n" + "=" * 72)
    print(f"SCREEN  A: {a.label()}\n        B: {b.label()}")
    print("=" * 72)
    if n == 0:
        print("no paired eval rows — arms did not share a panel")
    else:
        print(f"paired eval seeds        {n}")
        print(f"win rate  A / B          {wins_a / n:.4f} / {wins_b / n:.4f}")
        print(f"delta (B - A)            {100.0 * delta:+.2f} pp")
        print(f"discordant  A-only/B-only {b_only} / {c_only}")
        print(f"McNemar exact p          {p:.4g}")
    for arm_name, n_dropped in (dropped or {}).items():
        if n_dropped:
            print(f"[{arm_name}] eval runs dropped (error / timeout): {n_dropped}")

    for arm in (a, b):
        stats = read_stats(os.path.join(out_dir, arm.name, "main.stats.jsonl"))
        if not stats:
            continue
        last = stats[-1]
        pooled, pooled_n = tail_pooled(stats, 20)
        print(f"\n[{arm.name}] {len(stats)} iters")
        print(f"  training WR (last 20 iters, n={pooled_n})  {pooled:.4f}")
        print(f"  explained_variance                         {last.get('explained_variance')}")
        print(
            f"  ent / ent_c                                {last.get('ent')} / "
            f"{last.get('entropy_coef')}"
        )
        # Playbook §0.1: a starved surface makes a null uninformative. These
        # are printed for BOTH arms because a feature change can only be read
        # against a live env.
        dec = {k: v for k, v in last.items() if k.startswith("dec_per_run/")}
        print(f"  dec_per_run                                {dec}")
        wall = wall_hours(stats)
        print(
            f"  wall h (iter / collect / train)            "
            f"{wall['iter_h']} / {wall['collect_h']} / {wall['train_h']}"
        )
        out.update({f"{arm.name}_{k}": v for k, v in wall.items()})
        out[f"{arm.name}_train_wr_tail20"] = round(pooled, 4)
        out[f"{arm.name}_explained_variance"] = last.get("explained_variance")

    print("\nA screen is not a gate. Promotion follows")
    print("docs/ppo-champion-training-playbook.md §2 — full A0-A10 curve, 200")
    print("paired seeds per level, fresh panel, pre-registered rule.")

    path = os.path.join(out_dir, "report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nreport -> {path}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--name", required=True, help="experiment name (used in the report)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--character", default="Ironclad")
    ap.add_argument("--init-ckpt", default="", help="warm start (strongly recommended)")
    ap.add_argument("--arm-b-features", default="", help="encoder features for arm B (e.g. 'boss')")
    ap.add_argument("--arm-a-features", default="", help="encoder features for arm A")
    ap.add_argument(
        "--arm-b-flag",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra ppo.trainer override for arm B only (repeatable), e.g. lr_anneal_final=1.0",
    )
    ap.add_argument(
        "--arm-b-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra process env var for arm B's training and eval "
        "(repeatable) — for env-var-gated engine seams the trainer has "
        "no flag for (e.g. STS2_RUNAI_POTION_OC=critic)",
    )
    ap.add_argument("--eval-seeds", type=int, default=512)
    ap.add_argument("--seed-base", type=int, default=800_000_000)
    ap.add_argument("--procs", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--combat", choices=("mcts", "surrogate"), default="mcts")
    ap.add_argument("--surrogate-path", default="")
    ap.add_argument("--dry-run", action="store_true", help="print the commands and exit")
    ap.add_argument(
        "--allow-plain-wheel",
        action="store_true",
        help=f"train both arms on a plain (non-PGO) sts2_rs wheel; exports {ALLOW_PLAIN_ENV}=1 "
        "to the arms' trainers",
    )
    for key, value in PRESET.items():
        ap.add_argument(f"--{key.replace('_', '-')}", type=type(value), default=value)
    args = ap.parse_args(argv)

    # Fail fast, before either arm launches. This pins the wheel and restarts
    # the screen on the pinned copy; the arms' trainers and evals inherit the
    # pin through the environment `run_arm` / `eval_arm` pass on, and so does
    # the opt-out. Each arm's trainer checks again.
    wheel = None
    if not args.dry_run:
        wheel = prepare_training_wheel(
            pin_root=os.path.join(args.out, PIN_DIRNAME),
            allow_plain=args.allow_plain_wheel,
            main_module="ppo.screen",
        )
        if not wheel["wheel_pgo"]:
            os.environ[ALLOW_PLAIN_ENV] = "1"
        # An arm trained, or an eval run, by an earlier invocation is reused as
        # is (`run_arm`, `eval_arm`), so it ran on whatever wheel that
        # invocation had; say which.
        reused = {
            arm: _trained_wheel(os.path.join(args.out, arm))
            for arm in ("a", "b")
            if os.path.exists(os.path.join(args.out, arm, "main.pt"))
        }
        reused_evals = {
            arm: _eval_wheel(_eval_path(args.out, arm))
            for arm in ("a", "b")
            if os.path.exists(_eval_path(args.out, arm))
        }
        if reused:
            wheel = {**wheel, "reused_arms": reused}
        if reused_evals:
            wheel = {**wheel, "reused_evals": reused_evals}

    if not args.init_ckpt:
        print(
            "WARNING: no --init-ckpt. A short screen cannot reach from-scratch "
            "learnability; without a warm start you are measuring which arm "
            "gets off the floor faster, not which change helps.",
            file=sys.stderr,
        )

    cfg: dict[str, Any] = {k: getattr(args, k) for k in PRESET}
    cfg.update(
        character=args.character,
        init_ckpt=args.init_ckpt,
        seed_base=args.seed_base,
        procs=args.procs,
        device=args.device,
        combat=args.combat,
        surrogate_path=args.surrogate_path,
    )

    overrides_b: dict[str, Any] = {}
    env_b: dict[str, str] = {}
    for item in args.arm_b_env:
        key, sep, value = item.partition("=")
        if not sep or not key:
            ap.error(f"--arm-b-env needs KEY=VALUE, got {item!r}")
        env_b[key] = value
    for item in args.arm_b_flag:
        key, _, value = item.partition("=")
        if not _:
            ap.error(f"--arm-b-flag needs KEY=VALUE, got {item!r}")
        overrides_b[key.replace("-", "_")] = value

    arm_a = Arm("a", features=args.arm_a_features)
    arm_b = Arm("b", features=args.arm_b_features, overrides=overrides_b, env_vars=env_b)
    if arm_a.config_key() == arm_b.config_key():
        print(
            "NOTE: the arms are identical — this is an A/A run. The reported "
            "delta must come out inside noise; a significant one means the "
            "pairing or the eval panel is broken.",
            file=sys.stderr,
        )

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "screen.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "name": args.name,
                "cfg": cfg,
                "arm_b": overrides_b,
                "arm_a_features": arm_a.features,
                "arm_b_features": arm_b.features,
                "wheel": wheel,
            },
            f,
            indent=2,
        )

    results = {}
    for arm in (arm_a, arm_b):
        ckpt = run_arm(arm, cfg, args.out, args.dry_run)
        results[arm.name] = eval_arm(
            arm, ckpt, cfg, args.out, args.eval_seeds, args.dry_run, wheel=wheel
        )
    if args.dry_run:
        return 0

    dropped = {name: max(0, args.eval_seeds - len(res)) for name, res in results.items()}
    report(arm_a, arm_b, results["a"], results["b"], args.out, dropped=dropped)
    return 0


if __name__ == "__main__":
    sys.exit(main())
