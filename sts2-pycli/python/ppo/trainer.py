"""PPO training loop over full self-play runs (M-P3, fork-validated recipe).

Salvaged from ``runmodel-lightspeed run_ppo/ppo.py`` and adapted to the
M-P1 env + process-pool collector:

- collection: ``workers.play_batch(policy='net')`` — each spawn-context
  worker process loads the current checkpoint, plays whole runs (combat =
  deployed MCTS at eval-cheap), samples every non-combat multi-option
  decision from the net, and returns per-step transitions;
- rewards are computed LEARNER-side from the recorded metrics snapshots
  (``ppo.rewards``: graded base + PBRS shaping, terminal-uncredited), so
  shaping coefficients can be refit without touching workers;
- update: clipped surrogate (ε=0.2) + 0.5·value-MSE − entropy·masked
  entropy + 0.1·dest-room aux CE; **epochs=2**; GAE(γ=1, λ=0.97);
  batch-size-invariant EWMA advantage normalization (never
  per-trajectory); AdamW 3e-5 policy / 1e-4 value, wd 1e-4, grad-clip 5;
  entropy 0.05 → geometric decay (absolute-iteration-anchored, engage at
  plateau via --entropy-decay-start on resume).

Health metrics per iteration (plan gates G5/G6): win rate, avg floors,
KL, clipfrac, value EV, entropy, per-surface behavior stats — appended to
``<save>.stats.jsonl``. Checkpoints ``<save>.iter_N.pt`` + ``<save>.pt``;
``--resume-from-iter`` continues schedules (absolute anchoring).

The CLI refuses to start unless the imported ``sts2_rs`` is a fresh PGO
wheel (``ppo.wheel_guard``; build it with ``just sim-engine-pgo``).
``--allow-plain-wheel`` / ``STS2_ALLOW_PLAIN_WHEEL=1`` is the explicit opt-out
for a host that cannot build PGO. The CLI first pins the wheel (a private copy
in ``<save dir>/sts2_rs-pin/``; not on Windows, which has no exec) and
restarts itself on it, so the whole run -- every iteration's spawn workers
included -- runs the copy it checked, whatever happens to the venv.

The wheel record lands in every checkpoint (``wheel``) and the wandb config;
each ``stats.jsonl`` row carries what that iteration's workers reported
(``wheel``, ``wheel_pgo``), and a mismatch aborts the run. The pin does not
cover the first-party Python the workers import from disk (``python/ppo``,
``champions.py``): each row and checkpoint also carries the run's ``py_src``
digest, and an iteration whose workers report another one (a pull or edit
mid-campaign) aborts the same way.

Usage (from sts2-pycli):
    just sim-engine-pgo
    PYTHONPATH=python .venv/bin/python -m ppo.trainer --iterations 2 \
        --games-per-iter 32 --save-path /tmp/ppo_smoke
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from . import schema
from .model import AUX_IGNORE, RunPolicyNet, collate, load_compat
from .plasticity import plasticity_stats
from .rewards import ShapingConfig, shaped_rewards
from .wheel_guard import (
    FIX,
    PIN_DIRNAME,
    engine_change,
    prepare_training_wheel,
    python_source_digest,
    wheel_banner,
    wheel_identity,
    worker_source_mismatch,
    worker_wheel_mismatch,
)
from .workers import play_batch


@dataclass
class Transition:
    feat: object  # model.Featurized
    action_idx: int
    old_logprob: float
    value: float
    advantage: float = 0.0
    ret: float = 0.0


class RunningMoments:
    """Batch-size-invariant EWMA of mean/variance (fork adv normalization)."""

    def __init__(self, decay: float = 5e-4) -> None:
        self.decay = decay
        self.mean = 0.0
        self.sq = 1.0  # E[x^2]
        self.inited = False

    def update(self, xs: np.ndarray) -> None:
        if xs.size == 0:
            return
        if not self.inited:
            self.mean = float(xs.mean())
            self.sq = float((xs * xs).mean())
            self.inited = True
            return
        # Per-item decay: a batch of N retains (1-d)^N of the prior estimate.
        retain = (1.0 - self.decay) ** xs.size
        self.mean = retain * self.mean + (1.0 - retain) * float(xs.mean())
        self.sq = retain * self.sq + (1.0 - retain) * float((xs * xs).mean())

    @property
    def std(self) -> float:
        return math.sqrt(max(1e-6, self.sq - self.mean * self.mean))


@dataclass
class PPOConfig:
    iterations: int = 300
    games_per_iter: int = 256
    minibatch: int = 192
    epochs: int = 2
    clip: float = 0.2
    gamma: float = 1.0
    lam: float = 0.97
    value_coef: float = 0.5
    aux_coef: float = 0.1
    entropy_coef: float = 0.05
    entropy_coef_final: float = 0.0025
    entropy_decay_steps: int = 200
    entropy_decay_start: int = 10**9  # engage manually at the plateau
    policy_lr: float = 3e-5
    value_lr: float = 1e-4
    weight_decay: float = 1e-4
    adam_eps: float = 1e-8  # ICLR detail: 1e-5 often matches reference impls better
    # Linear LR decay: at iteration `it`, lr = base_lr · (1 − (1−final)·it/(iters−1)).
    # 1.0 ⇒ constant LR (prior behavior). Anneal-to-small lets the policy settle on
    # the peak instead of drifting past it as the entropy bonus decays.
    lr_anneal_final: float = 1.0
    max_grad_norm: float = 5.0
    adv_decay: float = 5e-4
    ascension: int = 0
    # Mixed-ascension curriculum: "asc:weight,asc:weight,..." (e.g.
    # "0:1,6:2,9:4,10:2"). Each game's ascension is sampled from this
    # distribution per iteration. Empty ⇒ every game runs the scalar
    # `ascension` above (backward-compatible single-ascension training).
    ascension_weights: str = ""
    # Prioritized Level Replay (Jiang et al. 2021, arXiv:2010.03934) over
    # (seed, ascension) levels, scored by mean |GAE| (see ppo/plr.py). Each
    # collection slot is, with probability `plr_replay_p`, drawn from the
    # replay buffer instead of the fresh-seed stream; fresh slots keep the
    # legacy seed formula and `ascension_weights` untouched. 0.0 ⇒ OFF and a
    # run is byte-identical to a pre-PLR trainer. `plr_beta` / `plr_rho` are
    # the paper's Procgen defaults (rank prioritization temperature,
    # staleness mixing); the buffer is capped (evict lowest score) because
    # the seed space here is unbounded (paper App. B.3).
    plr_replay_p: float = 0.0
    plr_beta: float = 0.1
    plr_rho: float = 0.1
    plr_buffer_cap: int = 4096
    # High-ascension combat budget. Games whose sampled ascension is >=
    # `high_asc_min` collect under (`high_asc_sims`, `high_asc_boss_sims`)
    # instead of the eval-cheap 1000/4000 default; lower ascensions keep the
    # default. The measured high-ascension lever is the combat sim budget, not
    # the run policy (IC `asc_it55` provenance) — so the curriculum's hard
    # levels train under a stronger combat cell. `high_asc_min <= 0` disables
    # (uniform sims, backward-compatible).
    high_asc_min: int = 0
    high_asc_sims: int = 2000
    high_asc_boss_sims: int = 8000
    # Global combat search budget for collection. 0 ⇒ keep the env's
    # eval-cheap default (1000 / 4000). Lowering these is the primary lever
    # of the fast screen (`docs/ppo-screen-playbook.md`): collection is ~87%
    # of campaign wall clock and is dominated by the combat MCTS. The
    # high-ascension override above still layers on top, so a screen that
    # sets both gets `sims` at low ascension and `high_asc_sims` at high.
    # Campaign use is fine PROVIDED games_per_iter scales to iso wall clock
    # and the gate runs at the deployed budget (which the standard gates do):
    # 250/1000 x 960 games matched-or-beat 1000/4000 x 256 on a paired
    # A0-A10 gate (docs/experiments/2026-08-19-training-sims-isocompute.md,
    # +2.50pp pooled n=2200 p=0.028 — below its pre-registered 3pp bar, so
    # ">= equal" is the certified claim). Cheap sims at UNCHANGED games is
    # unmeasured and plausibly worse — the volume is the active ingredient.
    sims: int = 0
    boss_sims: int = 0
    # Combat backend for collection: "mcts" plays every fight with the
    # combat MCTS (the campaign behavior); "surrogate" fast-forwards each
    # fight through the combat surrogate net at `surrogate_path` (one
    # forward per fight instead of a search per decision). Screening only —
    # see the playbook's deck-sensitivity gate before believing a
    # surrogate-world result about anything deck-shaped.
    combat: str = "mcts"
    surrogate_path: str = ""
    # Opt-in encoder experiment features, comma-separated (e.g. "boss") —
    # see `ppo.schema`. This field EXISTS TO BE RECORDED: the flag itself is
    # read from `STS2_PPO_FEATURES` at import time (the schema constants are
    # module-level and the collection workers are spawned fresh from the
    # environment), so `train()` only cross-checks the two and refuses to run
    # on a mismatch. Set the env var; declare it here so the checkpoint's
    # saved cfg says which schema produced it.
    features: str = ""
    # GH #545: the Act-1 model the collection env walks — "overgrowth",
    # "underdocks" or "roll". Unlike `features` this is NOT an env var: it is
    # a real constructor argument, so declaring it here both drives collection
    # and records into the checkpoint's saved cfg which act distribution
    # produced this champion. A champion trained on one mixture and later
    # evaluated under another is a silent divergence — nothing about the
    # tensor reveals it.
    act1: str = "overgrowth"
    # Torch device for the UPDATE only ("cpu", "cuda", "cuda:1", or "auto").
    # Collection stays CPU-per-process by design (many small forwards across
    # a process pool). At campaign settings the update is ~13% of wall clock;
    # once the screen cuts collection it becomes the floor, which is what
    # this flag removes.
    device: str = "cpu"
    # Matmul precision of the UPDATE's forward/backward: "fp32" (the campaign
    # recipe) or "bf16" (autocast; the losses stay fp32). The update is plain
    # fp32 SIMT GEMMs and saturates the GPU; bf16 halves it (59.6 -> 27.5
    # ms/minibatch, RTX 4080, GH #926) but moves the recomputed logprobs off
    # the acting ones (|dlogp| p99 0.04, max 0.10 vs fp32's 2e-5), which
    # perturbs the PPO ratio. That changes training, so it is screen-only
    # until a gate says otherwise. Collection and the acting snapshot are
    # unaffected; the weights and optimizer state stay fp32.
    update_precision: str = "fp32"
    character: str = "Ironclad"
    seed_base: int = 10_000_000  # disjoint from eval panels
    procs: int = 0  # 0 = cpu_count - 2
    # Overlap next-batch collection with the update phase (issue #219). While
    # the learner runs the ~12%-of-wall update, a background thread collects the
    # NEXT batch under the pre-update weights, so ~31 cores no longer idle during
    # training. The consumed batch is then exactly ONE gradient step stale
    # (standard 1-step-async PPO). OFF by default: this changes training
    # dynamics and must be gated on a paired win-rate comparison vs the sync
    # baseline before campaign use. The sync path (False) is bit-identical to
    # the pre-#219 loop.
    async_collect: bool = False
    dim: int = 192
    n_layers: int = 3
    n_heads: int = 4
    shaping_hp_coef: float = 0.1
    shaping_upg_coef: float = 0.035
    shaping_relic_coef: float = 0.0
    shaping_offset: float = 0.307
    resume_from_iter: int = 0
    save_every: int = 10
    save_path: str = "ppo_run"
    init_ckpt: str = ""  # BC warm-start checkpoint (M-P4)
    # Reusable-corpus dump. A campaign throws its decisions away every
    # iteration; recording the same data for an offline study
    # (`ppo.feature_probe`, `ppo.bc`) otherwise costs a fresh set of sims,
    # which is ~95% of the loop's wall clock. Non-empty `dump_corpus` writes
    # `<dir>/corpus_iter_NNNN.jsonl` in exactly the format `bc.load_corpus`
    # reads. STRIDED and off by default on purpose: a recorded run costs
    # ~260 KiB, so a 960-game iteration is ~250 MB and an unstrided
    # 150-iteration campaign would write tens of gigabytes. The recording
    # iterations also pay an IPC/JSON tax the others do not.
    dump_corpus: str = ""
    dump_corpus_every: int = 10
    # Plasticity diagnostic (`ppo.plasticity`): how many transitions to probe
    # per iteration, 0 to disable. One extra no-grad forward over this many
    # rows — negligible against a ~700 s collection phase. Logged, never acted
    # on: it measures whether the trunk is losing trainability under the
    # moving target, which is what BBF-style resets exist to undo. Reason to
    # watch it here: `--epochs` is rising (more gradient steps per unit of
    # fresh data erodes plasticity fastest), and champions warm-start from
    # champions, so the deployed lineage has trained far longer against a
    # moving target than any one campaign's iteration count suggests.
    plasticity_samples: int = 256
    # Experiment tracking. wandb_project empty ⇒ logging disabled (stats.jsonl
    # is always written regardless). wandb_run names the run (defaults to the
    # save-path basename). wandb_mode "online"/"offline"/"disabled".
    wandb_project: str = ""
    wandb_run: str = ""
    wandb_mode: str = "online"


def parse_ascension_weights(spec: str) -> tuple[list[int], np.ndarray] | None:
    """Parse ``"asc:weight,asc:weight,..."`` into (ascensions, probabilities).

    Returns None for an empty spec (single-ascension training). Weights need
    not be normalized; they are converted to a probability vector. Raises
    ValueError on malformed entries or non-positive total weight.
    """
    spec = spec.strip()
    if not spec:
        return None
    ascs: list[int] = []
    weights: list[float] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        asc_s, _, w_s = tok.partition(":")
        asc = int(asc_s)
        w = float(w_s) if w_s else 1.0
        if asc < 0 or w < 0:
            raise ValueError(f"bad ascension-weight entry {tok!r}")
        ascs.append(asc)
        weights.append(w)
    total = sum(weights)
    if not ascs or total <= 0:
        raise ValueError(f"empty or zero-weight ascension spec {spec!r}")
    return ascs, np.array(weights, dtype=np.float64) / total


def sample_ascensions(cfg: PPOConfig, it: int, n: int) -> list[int] | None:
    """Per-iteration ascension assignment for `n` games (deterministic in `it`).

    None ⇒ no curriculum (caller uses the scalar cfg.ascension). The per-iter
    seed keeps a --resume-from-iter run bit-identical to an uninterrupted one.
    """
    parsed = parse_ascension_weights(cfg.ascension_weights)
    if parsed is None:
        return None
    ascs, probs = parsed
    rng = np.random.default_rng(cfg.seed_base + 7_919 * (it + 1))
    return [int(ascs[i]) for i in rng.choice(len(ascs), size=n, p=probs)]


def high_asc_sims_lists(
    cfg: PPOConfig, eff_asc: list[int]
) -> tuple[list[int | None] | None, list[int | None] | None]:
    """Per-seed (sims, boss_sims) overrides for the collection budget.

    Two independent layers, low priority first:

    - `cfg.sims` / `cfg.boss_sims` (0 = unset) set the budget for EVERY game.
      This is the screen's lever.
    - `high_asc_min > 0` overrides games at or above that ascension with
      `high_asc_sims` / `high_asc_boss_sims`.

    A `None` entry means "leave it to the env", i.e. the eval-cheap 1000/4000
    default. Returns (None, None) only when neither layer is engaged, which
    keeps a default-configured run byte-identical to the pre-flag behavior.
    """
    base_s = cfg.sims if cfg.sims > 0 else None
    base_b = cfg.boss_sims if cfg.boss_sims > 0 else None
    if cfg.high_asc_min <= 0:
        if base_s is None and base_b is None:
            return None, None
        return [base_s] * len(eff_asc), [base_b] * len(eff_asc)
    sims = [cfg.high_asc_sims if a >= cfg.high_asc_min else base_s for a in eff_asc]
    boss = [cfg.high_asc_boss_sims if a >= cfg.high_asc_min else base_b for a in eff_asc]
    return sims, boss


def effective_entropy_coef(cfg: PPOConfig, it: int) -> float:
    if it < cfg.entropy_decay_start or cfg.entropy_decay_steps <= 0:
        return cfg.entropy_coef
    t = min(1.0, (it - cfg.entropy_decay_start) / max(1, cfg.entropy_decay_steps))
    return cfg.entropy_coef * (cfg.entropy_coef_final / cfg.entropy_coef) ** t


def corpus_dump_kwargs(cfg: PPOConfig, it: int) -> dict:
    """`play_batch` kwargs that dump iteration `it`'s runs as a reusable corpus.

    Empty dict when the dump is off or `it` is not on the stride — spread it
    into the `play_batch` call and a non-dumping iteration is byte-identical to
    one from before this flag existed. `record_decisions` only appends the raw
    obs/actions the workers already built, so a recording iteration plays the
    same runs; it costs IPC and JSON, not determinism.

    A stride below 1 records every iteration rather than dividing by zero.
    """
    if not cfg.dump_corpus:
        return {}
    every = max(1, cfg.dump_corpus_every)
    if it % every:
        return {}
    os.makedirs(cfg.dump_corpus, exist_ok=True)
    # Zero-padded so a shell glob sorts the corpus chronologically.
    path = os.path.join(cfg.dump_corpus, f"corpus_iter_{it:04d}.jsonl")
    return {"record_decisions": True, "out_path": path}


def plasticity_sample(
    net: RunPolicyNet, trans: list[Transition], n_samples: int, device: torch.device
) -> dict[str, float]:
    """Per-iteration plasticity diagnostic, or `{}` when off / no data.

    Probes the FIRST `n_samples` transitions rather than a random draw: the
    signal is the trend across a campaign, so the probe must not add sampling
    noise of its own on top of the already-drifting state distribution.
    """
    if n_samples <= 0 or not trans:
        return {}
    batch = collate([t.feat for t in trans[:n_samples]], device)
    return plasticity_stats(net, batch)


def acting_payload(state_dict: dict, cfg: PPOConfig) -> dict:
    """The checkpoint the collection workers act under.

    Module-level so the `cfg` invariant is testable without standing up a
    training loop; see `tests/test_ppo_features_from_checkpoint.py`.
    """
    return {"state_dict": state_dict, "cfg": cfg.__dict__}


def collect_rows(cfg: PPOConfig, it: int, acting_ckpt: str, sampler=None) -> list[dict]:
    """Play iteration `it`'s batch of runs under the weights in `acting_ckpt`.

    Module-level so the seed/ascension/budget wiring is testable without
    standing up a net; `train`'s `collect` closure only adds the timing.

    `sampler` (a `ppo.plr.LevelSampler`, owned by `train`) engages PLR: each
    slot is, with probability `cfg.plr_replay_p`, replaced by a buffer draw
    (seed AND ascension) played under a FRESH `sample_seed` — reusing the game
    seed would partially replay the recorded action stream (`workers.play_one`
    seeds its action rng off the game seed). The replay rng is its own stream,
    so fresh slots stay byte-identical to a PLR-off run.
    """
    seeds = [cfg.seed_base + it * cfg.games_per_iter + g for g in range(cfg.games_per_iter)]
    asc_list = sample_ascensions(cfg, it, len(seeds))
    eff_asc = asc_list if asc_list is not None else [cfg.ascension] * len(seeds)
    sample_seeds: list[int | None] | None = None
    if sampler is not None and cfg.plr_replay_p > 0 and len(sampler) > 0:
        rng = np.random.default_rng(cfg.seed_base + 104_729 * (it + 1))
        replay = rng.random(len(seeds)) < cfg.plr_replay_p
        draws = sampler.sample(int(replay.sum()), it, rng)
        if draws:
            ss_list: list[int | None] = [None] * len(seeds)
            for g, (r_seed, r_asc) in zip(np.flatnonzero(replay), draws):
                seeds[g] = int(r_seed)
                eff_asc[g] = int(r_asc)
                ss = int(rng.integers(1, 2**31 - 1))
                ss_list[g] = ss + 1 if ss == seeds[g] else ss
            sample_seeds = ss_list
            asc_list = eff_asc  # replay may change any slot's ascension
    sims_list, boss_sims_list = high_asc_sims_lists(cfg, eff_asc)
    rows = play_batch(
        seeds,
        policy="net",
        ascension=cfg.ascension,
        ascensions=asc_list,
        character=cfg.character,
        procs=(cfg.procs or None),
        ckpt_path=acting_ckpt,
        sims_list=sims_list,
        boss_sims_list=boss_sims_list,
        sample_seeds=sample_seeds,
        combat=cfg.combat,
        surrogate_path=(cfg.surrogate_path or None),
        act1=cfg.act1,
        **corpus_dump_kwargs(cfg, it),
    )
    if sample_seeds is not None:
        # rows come back in submission order (play_batch invariant), so a
        # positional zip marks exactly the replayed slots.
        for r, ss in zip(rows, sample_seeds):
            if ss is not None:
                r["plr_replayed"] = True
    return rows


def gae_from_row(row: dict, cfg: PPOConfig, shaping: ShapingConfig) -> list[Transition]:
    """Shaped rewards + GAE for one collected run (terminal bootstrap 0)."""
    trans_in = row.get("transitions") or []
    if not trans_in:
        return []
    # Backfill the ascension `rewards.total_floors` needs, for rows recorded
    # before `metrics_snapshot` carried it (GH #464). The row always has it.
    asc = int(row.get("ascension") or 0)
    step_metrics = [{"ascension": asc, **t["metrics"]} for t in trans_in]
    final_metrics = {"ascension": asc, **row["final"]}
    rewards, _ = shaped_rewards(step_metrics, final_metrics, bool(row["won"]), shaping)
    values = [t["value"] for t in trans_in]
    out: list[Transition] = []
    adv = 0.0
    next_v = 0.0  # terminal bootstrap
    for t in reversed(range(len(trans_in))):
        delta = rewards[t] + cfg.gamma * next_v - values[t]
        adv = delta + cfg.gamma * cfg.lam * adv
        out.append(
            Transition(
                feat=trans_in[t]["feat"],
                action_idx=trans_in[t]["action_idx"],
                old_logprob=trans_in[t]["old_logprob"],
                value=values[t],
                advantage=adv,
                ret=adv + values[t],
            )
        )
        next_v = values[t]
    out.reverse()
    return out


def plr_score(row: dict, cfg: PPOConfig, shaping: ShapingConfig) -> float | None:
    """PLR level score for one collected run: mean |GAE| (the paper's L1
    value-loss score, the only one it found consistent across envs).

    None for a row with no transitions — errored/timeout rows must never
    enter the replay buffer (the probe learned that lesson the hard way:
    crashed seeds scored as losses inverted a pooled sign).
    """
    trans = gae_from_row(row, cfg, shaping)
    if not trans:
        return None
    return float(np.mean([abs(t.advantage) for t in trans]))


def plr_update_from_rows(sampler, rows: list[dict], cfg: PPOConfig, shaping, it: int) -> dict:
    """Score iteration `it`'s completed runs into the PLR buffer; emit stats.

    Errored/timeout rows never enter the buffer. The stats carry the realized
    replay fraction and per-ascension replayed-run counts — the live tripwire
    for a buffer degenerating into a static high-ascension prior (#510's
    failure mode, which was only visible in the `n_runs/A*` keys).

    They also carry `plr_replay_distinct` / `plr_replay_top_share`, over
    LEVELS. The per-ascension counts alone are too coarse: 480 replays over a
    4096-level buffer can be 100% one level and still look like an ordinary
    ascension spread. That is not hypothetical — the 2026-09-01 smoke put ~96%
    of each batch on one level (see `LevelSampler.sample`), and it was only
    visible because a single level happened to monopolize its ascension.
    """
    n_replayed = 0
    asc_counts: dict[str, int] = {}
    level_counts: dict[tuple[int, int], int] = {}
    for r in rows:
        lv = (int(r.get("seed") or 0), int(r.get("ascension") or 0))
        if r.get("plr_replayed"):
            n_replayed += 1
            key = f"plr_replayed/A{lv[1]}"
            asc_counts[key] = asc_counts.get(key, 0) + 1
            level_counts[lv] = level_counts.get(lv, 0) + 1
        if r.get("outcome") in ("error", "timeout"):
            continue
        score = plr_score(r, cfg, shaping)
        if score is None:
            continue
        sampler.update(lv, score, it)
    return {
        "plr_buffer_n": len(sampler),
        "plr_replay_frac": round(n_replayed / max(1, len(rows)), 4),
        "plr_replay_distinct": len(level_counts),
        "plr_replay_top_share": round(max(level_counts.values()) / n_replayed, 4)
        if n_replayed
        else 0.0,
        **asc_counts,
    }


UPDATE_PRECISIONS = ("fp32", "bf16")


def update_autocast(cfg: PPOConfig, device: torch.device):
    """The autocast context `ppo_update` runs the net's forward in.

    A disabled autocast at "fp32", so the default update is unchanged.
    """
    if cfg.update_precision not in UPDATE_PRECISIONS:
        raise ValueError(
            f"--update-precision must be one of {UPDATE_PRECISIONS}, got {cfg.update_precision!r}"
        )
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=cfg.update_precision == "bf16",
    )


def ppo_update(
    net: RunPolicyNet,
    opt: torch.optim.Optimizer,
    trans: list[Transition],
    cfg: PPOConfig,
    adv_norm: RunningMoments,
    ent_coef: float,
    update_rng: np.random.Generator,
    device: torch.device | None = None,
) -> dict:
    device = device or torch.device("cpu")
    autocast = update_autocast(cfg, device)
    advs = np.array([t.advantage for t in trans], dtype=np.float64)
    adv_norm.update(advs)
    mean, std = adv_norm.mean, adv_norm.std
    n = len(trans)
    order = np.arange(n)
    stats: dict[str, float] = collections.defaultdict(float)
    for _ in range(cfg.epochs):
        update_rng.shuffle(order)
        for s in range(0, n, cfg.minibatch):
            mb = [trans[i] for i in order[s : s + cfg.minibatch]]
            b = collate([t.feat for t in mb], device)
            with autocast:
                logits, value, aux = net(b)
            # No-ops at fp32; under bf16 every loss term is computed in fp32.
            logits, value, aux = logits.float(), value.float(), aux.float()
            logp = F.log_softmax(logits, dim=-1)
            act = torch.tensor([t.action_idx for t in mb], dtype=torch.long, device=device)
            new_lp = logp.gather(1, act.unsqueeze(1)).squeeze(1).clamp(-20, 20)
            old_lp = torch.tensor([t.old_logprob for t in mb], device=device).clamp(-20, 20)
            adv = torch.tensor(
                [(t.advantage - mean) / std for t in mb], dtype=torch.float32, device=device
            )
            ret = torch.tensor([t.ret for t in mb], dtype=torch.float32, device=device)
            ratio = (new_lp - old_lp).exp()
            surr = torch.min(ratio * adv, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv)
            pi_loss = -surr.mean()
            v_loss = F.mse_loss(value, ret)
            p = logp.exp()
            ent_rows = -(p * logp.masked_fill(~b.act_mask, 0.0)).sum(1)
            multi = b.act_mask.sum(1) >= 2
            ent = ent_rows[multi].mean() if bool(multi.any()) else logits.new_zeros(())
            aux_loss = F.cross_entropy(
                aux.reshape(-1, aux.size(-1)), b.aux_room.reshape(-1), ignore_index=AUX_IGNORE
            )
            if torch.isnan(aux_loss):
                aux_loss = logits.new_zeros(())
            loss = pi_loss + cfg.value_coef * v_loss - ent_coef * ent + cfg.aux_coef * aux_loss
            if torch.isnan(loss):
                stats["nan_batches"] += 1
                continue
            opt.zero_grad()
            loss.backward()
            gn = nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
            opt.step()
            with torch.no_grad():
                kl = (old_lp - new_lp).mean()
                clipfrac = ((ratio - 1.0).abs() > cfg.clip).float().mean()
            stats["pi"] += pi_loss.item()
            stats["v"] += v_loss.item()
            stats["ent"] += float(ent.item())
            stats["aux"] += float(aux_loss.item())
            stats["kl"] += float(kl.item())
            stats["clipfrac"] += float(clipfrac.item())
            stats["grad_norm"] += float(gn)
            # Explained-variance accumulators.
            stats["_ev_ss_res"] += float(((ret - value) ** 2).sum().item())
            stats["_ev_sum"] += float(ret.sum().item())
            stats["_ev_sq"] += float((ret**2).sum().item())
            stats["_ev_n"] += len(mb)
            stats["nb"] += 1
    nb = max(1.0, stats["nb"])
    out = {k: v / nb for k, v in stats.items() if not k.startswith("_") and k != "nb"}
    out["nb"] = stats["nb"]
    if stats["_ev_n"] > 1:
        var = stats["_ev_sq"] / stats["_ev_n"] - (stats["_ev_sum"] / stats["_ev_n"]) ** 2
        out["explained_variance"] = 1.0 - (stats["_ev_ss_res"] / stats["_ev_n"]) / max(1e-8, var)
    return out


def make_optimizer(net: RunPolicyNet, cfg: PPOConfig) -> torch.optim.Optimizer:
    value_params, policy_params = [], []
    for name, p in net.named_parameters():
        (value_params if name.startswith(("value_head", "value_norm")) else policy_params).append(p)
    return torch.optim.AdamW(
        [
            {"params": policy_params, "lr": cfg.policy_lr, "base_lr": cfg.policy_lr},
            {"params": value_params, "lr": cfg.value_lr, "base_lr": cfg.value_lr},
        ],
        weight_decay=cfg.weight_decay,
        eps=cfg.adam_eps,
    )


def apply_lr_schedule(opt: torch.optim.Optimizer, cfg: PPOConfig, it: int) -> float:
    """Set each group's lr = base_lr · linear_frac(it) and return the frac.

    Linear from 1.0 at it=0 to `lr_anneal_final` at it=iterations−1. Anchored to
    absolute iteration so --resume-from-iter keeps the schedule continuous.
    """
    if cfg.lr_anneal_final >= 1.0 or cfg.iterations <= 1:
        return 1.0
    t = min(1.0, it / (cfg.iterations - 1))
    frac = 1.0 - (1.0 - cfg.lr_anneal_final) * t
    for g in opt.param_groups:
        base = g.setdefault("base_lr", g["lr"])  # tolerate optimizers loaded w/o base_lr
        g["lr"] = base * frac
    return frac


def per_ascension_winrate(rows: list[dict]) -> dict:
    """Win rate + game count per ascension (curriculum health dashboard).

    Keys ``win_rate/A{n}`` and ``n_runs/A{n}`` so each ascension is its own
    wandb series and the A9/A10 trajectory is visible during training.

    A10 runs also yield ``win_rate/A9_from_A10``: the share that entered the
    DoubleBoss second-boss room, i.e. beat the first act-3 boss (A10 = A9 +
    that fight). An ESTIMATE, not the A9 win rate: the policy is
    ascension-conditioned, and on Ironclad it read ~5pp below a true A9 sweep
    (2026-08-24). A run that died before act 3 reports None and counts as lost.
    """
    by_asc: collections.defaultdict[int, list[bool]] = collections.defaultdict(list)
    a9_from_a10: list[bool] = []
    for r in rows:
        asc = int(r.get("ascension", 0))
        by_asc[asc].append(bool(r.get("won")))
        if asc == 10:
            a9_from_a10.append(bool(r.get("won")) or r.get("reached_second_boss") is True)
    out: dict[str, float] = {}
    for asc in sorted(by_asc):
        wl = by_asc[asc]
        out[f"win_rate/A{asc}"] = round(sum(wl) / max(1, len(wl)), 4)
        out[f"n_runs/A{asc}"] = len(wl)
    if a9_from_a10:
        out["win_rate/A9_from_A10"] = round(sum(a9_from_a10) / len(a9_from_a10), 4)
    return out


def potion_swap_banner() -> str:
    """One startup line stating whether full-belt potion rewards are a DECISION.

    Unlike every other `SCREEN:` banner, this one is printed **unconditionally**
    — on *and* off. The banners above fire only on a deviation from the campaign
    default, which is the right shape when the risk is "someone switched
    something on". Here the risk runs the other way: the recipe wants the flag
    ON and the default is OFF, so the dangerous state is the *absence* of a
    setting, and a deviation-only banner is exactly as silent as no banner.

    GH #545 is the instance. The v2 campaign's launcher dropped
    `export STS2_RUNAI_POTION_SWAP=1` and ran 200 iterations with full-belt
    potion rewards silently DROPPED instead of surfaced. Nothing caught it: the
    pre-flight check read `dec_per_run/potion-swap`, and `behavior_stats` builds
    that dashboard from a `Counter` over observed surfaces, so a surface that
    never fires emits **no key at all** — an absent key is not a zero, and the
    check passed vacuously for the whole campaign.

    Read through the SAME resolver `ppo/env.py` hands to `RunManager`, never
    from `os.environ` directly: a banner that consults its own source can
    disagree with the world the workers actually build, which would be worse
    than no banner. It also means a malformed value panics HERE, at startup,
    instead of inside a spawned collection worker an hour in.
    """
    import sts2_rs

    on = sts2_rs.resolve_potion_swap_flag()
    detail = (
        "the net decides which potion to discard"
        if on
        else "full-belt potion rewards are SILENTLY DROPPED"
    )
    return f"SCREEN: potion_swap={'ON' if on else 'off'} (STS2_RUNAI_POTION_SWAP) — {detail}"


def behavior_stats(rows: list[dict]) -> dict:
    """Per-surface action-frequency dashboard (plan §3.5 exploit tripwires).

    Emits `dec_per_run/<surface>` for BOTH surface granularities:

    * the coarse surface (`env.classify_surface`: path / card-pick / event /
      shop / rest / treasure) — the historical series, unchanged;
    * the fine surface (`encoder.fine_surface`, the state token's own
      categorical: card-reward / card-remove / card-upgrade / card-transform /
      card-enchant / card-grant / card-smith / boss-relic / relic-reward / …)
      — added for GH #399, because every deck-mutation decision classifies as
      the coarse `card-pick` and would otherwise be invisible inside the
      ~14-per-run card-reward bucket, exactly the failure playbook §0.1's
      pre-flight exists to catch.

    A name shared by both families (path / shop / event / rest / treasure)
    keeps its COARSE count: `dec_per_run/event` is the #317 sentinel every
    campaign log and provenance note compares against (~1.2 = the surface is
    not live, ~6.x = healthy), so it must go on meaning "decisions taken in
    the event phase" even for an event whose candidates are, say, relic picks
    (coarse `event`, fine `relic-reward`). The fine keys are an overlay on the
    coarse partition, not a replacement — summing all `dec_per_run/*` double
    counts, by design.

    Transitions without a `fine_surface` (rows from an older collector) fall
    back to their coarse surface rather than vanishing.

    Also emits `act1_frac/<token>` (GH #545): the fraction of `rows` that
    walked each RESOLVED Act-1 token ("overgrowth" / "underdocks"; rows from
    a collector predating the `act1` field group under "unknown"). This is
    what makes the act mixture a first-class training stat instead of
    something discovered missing later — a campaign run with `--act1 roll`
    believes it is training ~50/50 on the two acts; `act1_frac/*` in
    `<save>.stats.jsonl` is the one-line check that it actually is, without
    re-deriving it from raw rows.
    """
    coarse: collections.Counter[str] = collections.Counter()
    fine: collections.Counter[str] = collections.Counter()
    for r in rows:
        for t in r.get("transitions") or []:
            coarse[t["surface"]] += 1
            fine[t.get("fine_surface") or t["surface"]] += 1
    n_runs = max(1, len(rows))
    out = {f"dec_per_run/{k}": round(v / n_runs, 2) for k, v in sorted(fine.items())}
    # Coarse last: it wins any name collision (see docstring).
    out.update({f"dec_per_run/{k}": round(v / n_runs, 2) for k, v in sorted(coarse.items())})
    acts = collections.Counter(r.get("act1") or "unknown" for r in rows)
    out.update({f"act1_frac/{k}": round(v / n_runs, 3) for k, v in sorted(acts.items())})
    return out


def _init_wandb(cfg: PPOConfig, wheel: dict | None = None):
    """Return a live wandb run (or None if disabled/unavailable).

    Never fatal: a missing package, missing login, or init failure degrades to
    stats.jsonl-only logging with a warning rather than killing an overnight
    run. `resume="allow"` + a save-path-derived id makes --resume-from-iter
    continue the same wandb run. `wheel` (the CLI's wheel-guard record) rides
    in the run config next to the cfg.
    """
    if not cfg.wandb_project:
        return None
    try:
        import wandb
    except ImportError:
        print("wandb not installed — logging to stats.jsonl only", flush=True)
        return None
    run_id = (cfg.wandb_run or os.path.basename(cfg.save_path) or "ppo").replace("/", "_")
    try:
        return wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run or None,
            id=run_id,
            resume="allow",
            mode=cfg.wandb_mode,
            config={**cfg.__dict__, **({"wheel": wheel} if wheel is not None else {})},
        )
    except Exception as e:
        print(f"wandb.init failed ({e!r}) — logging to stats.jsonl only", flush=True)
        return None


def _parse_proc_stat(line: str) -> tuple[int, int] | None:
    """(busy, total) CPU jiffies from a ``/proc/stat`` aggregate ``cpu`` line.

    Busy excludes idle + iowait. Returns None for a malformed / non-Linux line.
    """
    parts = line.split()
    if len(parts) < 5 or parts[0] != "cpu":
        return None
    try:
        vals = [int(x) for x in parts[1:]]
    except ValueError:
        return None
    idle = vals[3] + (vals[4] if len(vals) > 4 else 0)  # idle + iowait
    return sum(vals) - idle, sum(vals)


def _cpu_busy() -> tuple[int, int] | None:
    """Current (busy, total) CPU jiffies, or None where ``/proc/stat`` is absent."""
    try:
        with open("/proc/stat", encoding="utf-8") as f:
            return _parse_proc_stat(f.readline())
    except OSError:
        return None


def _cpu_util(a: tuple[int, int] | None, b: tuple[int, int] | None) -> float | None:
    """Average busy fraction over the interval between two ``_cpu_busy`` samples."""
    if a is None or b is None:
        return None
    d_busy, d_total = b[0] - a[0], b[1] - a[1]
    return round(d_busy / d_total, 4) if d_total > 0 else None


def resolve_device(spec: str) -> torch.device:
    """Resolve a ``--device`` spec, refusing a silent CPU fallback.

    "auto" picks CUDA when it is there and CPU otherwise. Anything else is
    taken literally and *verified*: asking for "cuda" on a box without it is
    an error, not a 30×-slower run that looks identical in the log. The
    screen's whole budget assumes the update is off the critical path, so a
    silent demotion would blow it with no visible symptom.
    """
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(spec)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"--device {spec} requested but torch.cuda.is_available() is False")
    return dev


def _cpu_state_dict(net: RunPolicyNet) -> dict:
    """`net.state_dict()` with every tensor on CPU.

    Checkpoints must stay device-agnostic: they are read by the collection
    workers, `promote-champion`, the ONNX export and the parity fixtures, so a
    CUDA-tagged tensor in one of them is a portability bug waiting to surface
    somewhere far from here.
    """
    return {k: v.detach().to("cpu") for k, v in net.state_dict().items()}


def resume_hint(save_path: str, it: int) -> str:
    """How to continue a run that stopped before finishing iteration `it`:
    from the newest `<save_path>.iter_K.pt` (K <= it), else from scratch."""
    for k in range(it, 0, -1):
        if os.path.exists(f"{save_path}.iter_{k}.pt"):
            return (
                f"resume by re-running the command with --resume-from-iter {k} "
                f"(newest checkpoint: {save_path}.iter_{k}.pt)"
            )
    return "re-run the command as it was (no checkpoint was saved yet)"


def train(cfg: PPOConfig, wheel: dict | None = None) -> RunPolicyNet:
    """Run the PPO loop. `wheel` is the `ppo.wheel_guard` record the CLI checked
    (and pinned) before calling this: it is bannered and saved into every
    checkpoint and the wandb config. The wheel check itself stays in the CLI —
    tests call `train(cfg)` on a plain wheel.

    Every iteration, the identity each collection worker reports must equal
    the run's (`wheel`'s, else this process's own import), or the run aborts
    before training on that batch; each stats row carries what the workers
    reported (`wheel`, `wheel_pgo`)."""
    torch.manual_seed(0)
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 4))
    device = resolve_device(cfg.device)
    want_features = schema.parse_features(cfg.features)
    if want_features != schema.ACTIVE_FEATURES:
        raise ValueError(
            f"--features {sorted(want_features)} but ppo.schema imported "
            f"{sorted(schema.ACTIVE_FEATURES)}. The encoder schema is fixed at import "
            f"time; export STS2_PPO_FEATURES='{cfg.features}' before launching."
        )
    if cfg.combat not in ("mcts", "surrogate"):
        raise ValueError(f"--combat must be 'mcts' or 'surrogate', got {cfg.combat!r}")
    if cfg.combat == "surrogate" and not cfg.surrogate_path:
        raise ValueError("--combat surrogate requires --surrogate-path")
    update_autocast(cfg, device)  # validates update_precision before any collection
    if cfg.plr_replay_p > 0 and cfg.async_collect:
        # The async background collector would read the PLR sampler while
        # update_and_log scores rows into it — a data race. Sync-only for now.
        raise ValueError("plr_replay_p > 0 is incompatible with --async-collect")
    shaping = ShapingConfig(
        hp_coef=cfg.shaping_hp_coef,
        upg_coef=cfg.shaping_upg_coef,
        relic_coef=cfg.shaping_relic_coef,
        offset=cfg.shaping_offset,
    )
    net = RunPolicyNet(dim=cfg.dim, n_layers=cfg.n_layers, n_heads=cfg.n_heads).to(device)
    opt = make_optimizer(net, cfg)
    adv_norm = RunningMoments(cfg.adv_decay)
    sampler = None
    if cfg.plr_replay_p > 0:
        from .plr import LevelSampler

        sampler = LevelSampler(cap=cfg.plr_buffer_cap, beta=cfg.plr_beta, rho=cfg.plr_rho)
    start_it = cfg.resume_from_iter
    if start_it > 0:
        ck = torch.load(
            f"{cfg.save_path}.iter_{start_it}.pt", map_location="cpu", weights_only=True
        )
        net.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["opt"])
        adv_norm.mean, adv_norm.sq, adv_norm.inited = ck["adv_mean"], ck["adv_sq"], True
        if sampler is not None:
            if "plr" in ck:
                sampler.load_state_dict(ck["plr"])
            else:
                # Resuming a pre-PLR checkpoint: legal (warm start into PLR),
                # but say so — an empty buffer collects fresh-only until it fills.
                print("PLR: no buffer in the resumed checkpoint — starting empty", flush=True)
        print(f"resumed from iter {start_it}", flush=True)
        moved = engine_change(ck.get("wheel"), wheel)
        if moved is not None:
            # Allowed (a fix can be worth a mid-campaign rebuild), but never
            # silent: the stats rows from here on carry the new build.
            print(
                f"WARNING: iteration {start_it}'s checkpoint was trained on another "
                f"sts2_rs build; resuming on this one ({moved})",
                flush=True,
            )
        ck_src = ck.get("py_src")
        if ck_src is not None and ck_src != python_source_digest():
            # A source change, not an engine change: the same rule applies.
            print(
                f"WARNING: iteration {start_it}'s checkpoint was trained with other "
                f"first-party Python (python/ppo, champions.py); resuming on this "
                f"source (py_src {ck_src} -> {python_source_digest()})",
                flush=True,
            )
    elif cfg.init_ckpt:
        ck = torch.load(cfg.init_ckpt, map_location="cpu", weights_only=True)
        load_compat(net, ck.get("state_dict", ck))
        print(f"warm-started from {cfg.init_ckpt}", flush=True)

    update_rng = np.random.default_rng(0)
    ckpt_dir = os.path.dirname(os.path.abspath(cfg.save_path))
    os.makedirs(ckpt_dir, exist_ok=True)
    wb = _init_wandb(cfg, wheel)
    if wheel is not None:
        print(wheel_banner(wheel), flush=True)
    run_wheel = wheel_identity(wheel)
    run_py_src = python_source_digest()
    if cfg.ascension_weights:
        print(f"mixed-ascension curriculum: {cfg.ascension_weights}", flush=True)
    if sampler is not None:
        print(
            f"PLR: replay_p={cfg.plr_replay_p} beta={cfg.plr_beta} rho={cfg.plr_rho} "
            f"cap={cfg.plr_buffer_cap} buffer_n={len(sampler)}",
            flush=True,
        )
    # Banner every deviation from campaign settings. The playbook's recurring
    # failure is a silent flag (`--lr-anneal-final`'s default-OFF cost ~6pp
    # and nothing in the log distinguished it), so anything that changes the
    # world the policy trains in announces itself here.
    if cfg.combat != "mcts":
        print(f"SCREEN: combat backend={cfg.combat} surrogate={cfg.surrogate_path}", flush=True)
    if cfg.sims or cfg.boss_sims:
        print(
            f"SCREEN: combat budget sims={cfg.sims or 'default'} "
            f"boss_sims={cfg.boss_sims or 'default'} — NOT campaign-comparable",
            flush=True,
        )
    if schema.ACTIVE_FEATURES:
        # Servable since GH #545: the schema crosses the FFI explicitly
        # (`PpoNetPolicy(path, features)`), resolved from champions.toml, and
        # `promote_champion.py` derives that field from this checkpoint's own
        # cfg. What it is NOT is servable at the DEFAULT schema -- a gated
        # feature rewrites an existing `cat` slot instead of appending a
        # column, so a mis-served net has an identical graph and shapes and
        # fails silently rather than loudly.
        print(
            f"SCREEN: encoder features={sorted(schema.ACTIVE_FEATURES)} "
            f"N_CAT={schema.N_CAT} — promoting this net REQUIRES "
            f"champions.toml features={cfg.features!r}; served at the default "
            f"schema it is silently wrong, not broken",
            flush=True,
        )
    if cfg.update_precision != "fp32":
        print(
            f"SCREEN: update precision={cfg.update_precision} — changes training, "
            f"NOT campaign-comparable",
            flush=True,
        )
    print(potion_swap_banner(), flush=True)
    print(f"update device: {device}", flush=True)

    acting_ckpt = f"{cfg.save_path}.acting.pt"

    def snapshot_acting() -> None:
        """Freeze the CURRENT net weights to the acting checkpoint for workers.

        Called from the MAIN thread before the update runs, so the background
        collector (async mode) reads an on-disk θ snapshot and never races the
        in-place optimizer step.

        **The `cfg` rides along, and it is load-bearing.**
        `NetPolicy.__init__` resolves its feature schema from the checkpoint's
        own cfg — deliberately, so a committed champion is served at the schema
        it trained with rather than the shell's (`9df0210ba`). A checkpoint with
        no cfg falls back to the DEFAULT schema. That fallback is right for the
        champions, and silently wrong here: without this field a campaign
        launched with `--features reward_upgrade` banners the feature, records
        it into every saved cfg, and then collects EVERY trajectory with the
        feature off, because the actor is this cfg-less rolling snapshot. The
        gated branch changes what goes into an existing slot rather than adding
        one, so nothing in the tensor reveals it.
        """
        torch.save(acting_payload(_cpu_state_dict(net), cfg), acting_ckpt)

    def collect(it: int) -> tuple[list[dict], float]:
        """Play iteration ``it``'s batch under whatever weights are currently in
        ``acting_ckpt``. Reads only the on-disk checkpoint (safe to run in a
        background thread while the main thread trains)."""
        t0 = time.time()
        return collect_rows(cfg, it, acting_ckpt, sampler=sampler), time.time() - t0

    def check_workers_wheel(it: int, rows: list[dict]) -> None:
        """Abort before training on a batch whose workers ran a different
        sts2_rs than the run started on. Called on the main thread with no
        collection in flight, so the abort does not wait for one."""
        mismatch = worker_wheel_mismatch(rows, run_wheel)
        if mismatch is None:
            return
        raise SystemExit(
            f"ABORTING at iteration {it}: its collection workers did not run the "
            f"sts2_rs this run started on, so nothing from it was trained on.\n"
            f"  run started on: {json.dumps(run_wheel, sort_keys=True)}\n{mismatch}\n"
            "Something replaced the wheel the workers import (the pin should make "
            "that impossible for a CLI run). Check the wheel with `python -m "
            f"ppo.wheel_guard` (`{FIX}` rebuilds it), then {resume_hint(cfg.save_path, it)}."
        )

    def check_workers_source(it: int, rows: list[dict]) -> None:
        """Abort before training on a batch whose workers imported other
        first-party Python than the run started with. The pin covers only the
        sts2_rs extension; each iteration's spawn workers import `python/ppo`
        from the checkout on disk again."""
        mismatch = worker_source_mismatch(rows, run_py_src)
        if mismatch is None:
            return
        raise SystemExit(
            f"ABORTING at iteration {it}: its collection workers ran other first-party "
            "Python than this run started with, so nothing from it was trained on. This "
            "is a SOURCE change under the run (a pull, checkout, stash or edit of "
            "sts2-pycli/python/ppo or champions.py), not an sts2_rs build change.\n"
            f"  run started with py_src {run_py_src}\n{mismatch}\n"
            "Restore the source the run started with, or decide to continue on the new "
            f"code, then {resume_hint(cfg.save_path, it)}."
        )

    def update_and_log(
        it: int,
        rows: list[dict],
        collect_s: float,
        busy_iter0: tuple[int, int] | None,
        busy_collect1: tuple[int, int] | None,
        it_t0: float,
    ) -> None:
        ok = [r for r in rows if r.get("transitions")]
        # `check_workers_wheel` already made every reporting row equal run_wheel.
        worker_wheel = next((r["wheel"] for r in rows if "wheel" in r), None)
        wins = sum(1 for r in ok if r.get("won"))
        floors = [int(r["final"].get("total_floor") or 0) for r in ok]
        all_trans = [t for r in ok for t in gae_from_row(r, cfg, shaping)]
        errors = sum(1 for r in rows if r.get("outcome") in ("error", "timeout"))
        # GH #763: name the dropped games, not just count them.
        dropped = [
            {k: r.get(k) for k in ("seed", "outcome", "age_s", "error") if r.get(k) is not None}
            for r in rows
            if r.get("outcome") in ("error", "timeout")
        ]
        plr_stats = (
            plr_update_from_rows(sampler, rows, cfg, shaping, it) if sampler is not None else {}
        )

        ent_coef = effective_entropy_coef(cfg, it)
        lr_frac = apply_lr_schedule(opt, cfg, it)
        t1 = time.time()
        busy_train0 = _cpu_busy()
        st = ppo_update(net, opt, all_trans, cfg, adv_norm, ent_coef, update_rng, device)
        train_s = time.time() - t1
        busy_iter1 = _cpu_busy()
        # Measured AFTER the update, so iteration N reports the net that
        # iteration N+1 will collect with.
        plast = plasticity_sample(net, all_trans, cfg.plasticity_samples, device)

        stats = {
            "iteration": it,
            "n_runs": len(ok),
            "errors": errors,
            "dropped": dropped,
            "win_rate": wins / max(1, len(ok)),
            "avg_floor": float(np.mean(floors)) if floors else 0.0,
            "n_transitions": len(all_trans),
            "entropy_coef": ent_coef,
            "lr_frac": round(lr_frac, 4),
            "policy_lr": round(cfg.policy_lr * lr_frac, 8),
            "adv_norm_std": adv_norm.std,
            "collect_s": round(collect_s, 1),
            "train_s": round(train_s, 1),
            "iter_s": round(time.time() - it_t0, 1),
            # Effective CPU utilization (issue #219 acceptance metric). In sync
            # mode collect/train are sequential so each phase is measured; in
            # async mode they overlap, so only the whole-iteration figure is
            # meaningful (util_collect is left null).
            "util_collect": _cpu_util(busy_iter0, busy_collect1),
            "util_train": _cpu_util(busy_train0, busy_iter1),
            "util_iter": _cpu_util(busy_iter0, busy_iter1),
            **{k: round(float(v), 5) for k, v in st.items()},
            **per_ascension_winrate(ok),
            **behavior_stats(ok),
            **plast,
            **plr_stats,
            # What the workers reported, per row: a --resume-from-iter run may
            # resume on a different wheel, and each iteration says which it ran.
            # Null when no run reached the engine (every one an error or a
            # timeout), so an unchecked iteration shows as one.
            "wheel_pgo": bool(worker_wheel["pgo"]) if worker_wheel is not None else None,
            "wheel": worker_wheel,
            "py_src": run_py_src,
        }
        with open(f"{cfg.save_path}.stats.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(stats) + "\n")
        if wb is not None:
            wb.log(
                {k: v for k, v in stats.items() if k not in ("iteration", "wheel", "py_src")},
                step=it,
            )
        util = stats["util_iter"]
        cpu_suffix = f" cpu={util:.0%}" if util is not None else ""
        print(
            f"it{it:3d} win={stats['win_rate']:.3f} floor={stats['avg_floor']:.1f} "
            f"trans={len(all_trans)} kl={st.get('kl', 0):.4f} clip={st.get('clipfrac', 0):.3f} "
            f"ev={st.get('explained_variance', 0):.3f} ent={st.get('ent', 0):.3f} "
            f"ent_c={ent_coef:.4f} err={errors} "
            f"collect={collect_s:.0f}s train={train_s:.0f}s{cpu_suffix}",
            flush=True,
        )
        if (it + 1) % cfg.save_every == 0 or it + 1 == cfg.iterations:
            torch.save(
                {
                    "state_dict": _cpu_state_dict(net),
                    "opt": opt.state_dict(),
                    "adv_mean": adv_norm.mean,
                    "adv_sq": adv_norm.sq,
                    "cfg": cfg.__dict__,
                    # PLR buffer rides in the checkpoint so --resume-from-iter
                    # replays from the same buffer an uninterrupted run would.
                    **({"plr": sampler.state_dict()} if sampler is not None else {}),
                    **({"wheel": wheel} if wheel is not None else {}),
                    "py_src": run_py_src,
                },
                f"{cfg.save_path}.iter_{it + 1}.pt",
            )
            torch.save({"state_dict": _cpu_state_dict(net)}, f"{cfg.save_path}.pt")

    if cfg.async_collect:
        print("async collect: next-batch collection overlaps the update (1-step stale)", flush=True)
        # Pipeline: while the learner updates on batch `it`, a background thread
        # collects batch `it+1` under the pre-update weights.
        with ThreadPoolExecutor(max_workers=1) as pool_exec:
            snapshot_acting()  # θ_start
            fut = pool_exec.submit(collect, start_it)
            for it in range(start_it, cfg.iterations):
                it_t0 = time.time()
                busy_iter0 = _cpu_busy()
                rows, collect_s = fut.result()
                check_workers_wheel(it, rows)
                check_workers_source(it, rows)
                if it + 1 < cfg.iterations:
                    snapshot_acting()  # θ_it (pre-update); consumed next iter → 1 step stale
                    fut = pool_exec.submit(collect, it + 1)
                update_and_log(it, rows, collect_s, busy_iter0, None, it_t0)
    else:
        for it in range(start_it, cfg.iterations):
            it_t0 = time.time()
            busy_iter0 = _cpu_busy()
            snapshot_acting()
            rows, collect_s = collect(it)
            busy_collect1 = _cpu_busy()
            check_workers_wheel(it, rows)
            check_workers_source(it, rows)
            update_and_log(it, rows, collect_s, busy_iter0, busy_collect1, it_t0)
    if wb is not None:
        wb.finish()
    return net


def _cli(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    for f in fields(PPOConfig):
        name = f.name.replace("_", "-")
        if isinstance(f.default, bool):
            # A generic ``type=bool`` would treat any non-empty string
            # (incl. "false") as True; parse booleans explicitly instead.
            ap.add_argument(
                f"--{name}",
                type=lambda s: s.lower() in ("1", "true", "yes", "on"),
                default=f.default,
            )
        else:
            ap.add_argument(f"--{name}", type=type(f.default), default=f.default)
    # Not a PPOConfig field: it is a statement about the host, not the run,
    # and the run records what it actually got (`wheel_pgo`) instead.
    ap.add_argument(
        "--allow-plain-wheel",
        action="store_true",
        help="train on a plain (non-PGO) sts2_rs wheel, for a host that cannot "
        "build PGO; also STS2_ALLOW_PLAIN_WHEEL=1. Collection runs ~1.28x slower.",
    )
    args = ap.parse_args(argv)
    cfg = PPOConfig(**{f.name: getattr(args, f.name) for f in fields(PPOConfig)})
    # Before train(), so a wrong wheel costs nothing, not a collection phase.
    # Pins the wheel next to the checkpoints and restarts this process on the
    # pinned copy, so a venv rebuild mid-campaign cannot reach the workers.
    wheel = prepare_training_wheel(
        pin_root=os.path.join(os.path.dirname(os.path.abspath(cfg.save_path)), PIN_DIRNAME),
        allow_plain=args.allow_plain_wheel,
        main_module="ppo.trainer",
    )
    train(cfg, wheel=wheel)


if __name__ == "__main__":
    _cli()
