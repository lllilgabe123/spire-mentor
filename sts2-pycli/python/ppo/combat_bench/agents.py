"""Agent specs for `ppo.combat_bench run`: what plays the fights, as plain data.

A spec is a JSON-able dict, so it goes into the played file's header verbatim
and a resume can compare it field for field. Two kinds exist:

- `mcts`: the deployed `sts2_rs.MctsCombatAgent` at a `sims` / `boss_sims`
  budget.
- `factory`: any `pkg.module:factory` callable. It is called as
  `factory(seed=int, **args)` and must return an object with
  `step(rm) -> dict | None`, the `MctsCombatAgent.step` contract. If it also
  has `combat_budget_spent()`, its fourth element is read as the simulations
  requested.

Every spec carries the environment its workers run under (`env`). The engine
reads its behaviour seams from the environment, and some of them only once per
process: `STS2_ROLLOUT_NET` / `STS2_ROLLOUT_TEMP` sit behind a `OnceLock`, and
`STS2_RUNAI_MCTS_OVERRIDES` is read when an agent is constructed. That is why
`apply_env` runs in the Pool initializer, before the worker imports `sts2_rs`.
It also scrubs every engine seam the parent shell happened to export: an
inherited `STS2_RUNAI_HONEST=0` would otherwise turn a run clairvoyant
without anything in the header saying so. The spec's `env` together with the
wheel (the header's `src_hash`) is therefore the whole of the engine
configuration. The wheel matters on its own: since GH #830 (2026-10-25) the
run agent compiles in an Ironclad layer (the forms + combo knobs, and the BC
move prior in elite and boss fights), so an empty `env` plays a different
Ironclad search on wheels before and after it. SCHEMA.md says how to
reproduce the pre-ship B0.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
from typing import Any, Protocol

# Env vars the Rust engine reads as behaviour seams (grep `env::var` in
# sts2-runai / sts2-mcts / sts2-core). Scrubbed in every worker unless the
# spec sets them. `STS2_COMBAT_LOG` echoes every log row to stderr.
ENGINE_ENV_PREFIXES = ("STS2_RUNAI_", "STS2_ROLLOUT_", "STS2_TRACE_")
ENGINE_ENV_EXACT = frozenset({"STS2_COMBAT_LOG"})

ENV_OVERRIDES = "STS2_RUNAI_MCTS_OVERRIDES"
ENV_ROLLOUT_NET = "STS2_ROLLOUT_NET"
ENV_ROLLOUT_TEMP = "STS2_ROLLOUT_TEMP"
ENV_HONEST = "STS2_RUNAI_HONEST"
# `sts2-runai` override knob: the deployed dialect under any hidden-info mode,
# including the run agent's Ironclad layer (GH #830), which follows the dialect.
BELIEF_DIALECT_KNOB = "belief_dialect=1"
# `sts2_runai::policy::resolve_hidden_info`: unset or "belief" is the deployed
# BeliefDraw, "0" clairvoyant, "root" / "full" the older honest modes.
HONEST_MODES = frozenset({"belief", "0", "root", "full"})

_ENV_KEY = re.compile(r"^[A-Z_][A-Z0-9_]*$")


class CombatAgent(Protocol):
    """What `run` drives: one decision per `step`, `None` when it has none."""

    def step(self, rm: Any) -> dict[str, Any] | None: ...


def is_engine_env(name: str) -> bool:
    """Whether `name` is an engine behaviour seam that `apply_env` scrubs."""
    return name in ENGINE_ENV_EXACT or name.startswith(ENGINE_ENV_PREFIXES)


def parse_kv(items: list[str], what: str) -> dict[str, str]:
    """Parse repeated `K=V` flags, refusing duplicates and empty keys.

    Args:
        items: The raw flag values.
        what: The flag name, for error messages.

    Returns:
        The pairs as a dict, in flag order.

    Raises:
        ValueError: On a malformed item or a key given twice.
    """
    out: dict[str, str] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ValueError(f"{what} expects K=V, got {item!r}")
        if key in out:
            raise ValueError(f"{what} sets {key} twice")
        out[key] = value
    return out


def _arg_value(raw: str) -> Any:
    # `--agent-arg sims=40` should reach the factory as an int, and
    # `name=foo` as a string, without a type annotation on the command line.
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_env(
    env: dict[str, str],
    overrides: str | None = None,
    rollout_net: str | None = None,
    rollout_temp: float | None = None,
    clair: bool = False,
) -> dict[str, str]:
    """Merge `--env` with the dedicated engine flags into one env dict.

    Args:
        env: Explicit `--env K=V` pairs.
        overrides: `STS2_RUNAI_MCTS_OVERRIDES`, the knob bundle.
        rollout_net: Path of a playout net; becomes `STS2_ROLLOUT_NET` as an
            absolute path, so a worker started elsewhere loads the same file.
        rollout_temp: `STS2_ROLLOUT_TEMP`. The engine applies it only to a net
            loaded from the env, so it is refused without `rollout_net`
            rather than silently doing nothing.
        clair: The deployed search with perfect information:
            `STS2_RUNAI_HONEST=0` plus the `belief_dialect=1` override knob.
            The engine applies the deployed search dialect
            (`MctsConfig::apply_mode_dialect`: the RL playout net, the
            budget-keyed c_puct, draw widening and the EndTurn widening
            retune) only under BeliefDraw; the knob applies it under
            Clairvoyant too, so a deployed-vs-clair delta measures hidden
            information alone. The spec records `search_dialect:
            "belief-draw-clairvoyant"`. A bare `--env STS2_RUNAI_HONEST=0`
            without the knob is the older clairvoyant search (`legacy`).

    Returns:
        The merged env, sorted by key.

    Raises:
        ValueError: On a bad key, a key set twice (by `--env` and a dedicated
            flag), a missing net file, or a temperature without a net.
    """
    out = dict(env)
    for key in out:
        if not _ENV_KEY.match(key):
            raise ValueError(f"--env key {key!r} is not an env var name")

    def put(key: str, value: str) -> None:
        if key in out:
            raise ValueError(f"{key} is set by both --env and its dedicated flag")
        out[key] = value

    if clair:
        # Applied first inside the engine's bundle, so an explicit knob in
        # `overrides` still wins over its dialect value.
        overrides = BELIEF_DIALECT_KNOB + ("," + overrides if overrides else "")
    if overrides:
        put(ENV_OVERRIDES, overrides)
    if rollout_net:
        path = os.path.abspath(rollout_net)
        if not os.path.isfile(path):
            raise ValueError(f"--rollout-net {rollout_net!r} is not a file")
        put(ENV_ROLLOUT_NET, path)
    if rollout_temp is not None:
        if ENV_ROLLOUT_NET not in out:
            raise ValueError("--rollout-temp only applies to a net given by --rollout-net")
        put(ENV_ROLLOUT_TEMP, repr(float(rollout_temp)))
    if clair:
        put(ENV_HONEST, "0")
    if out.get(ENV_HONEST, "belief") not in HONEST_MODES:
        # The engine panics on any other value, so every cell would end as
        # `error`. Refuse up front instead (`STS2_RUNAI_HONEST=1` is not "on").
        raise ValueError(
            f"{ENV_HONEST}={out[ENV_HONEST]!r}: expected one of {sorted(HONEST_MODES)} "
            "(leave it unset for the deployed honest search)"
        )
    return dict(sorted(out.items()))


def search_dialect(env: dict[str, str]) -> str:
    """`belief-draw` (the deployed search) or `legacy` (any other hidden-info mode).

    `sts2_mcts::MctsConfig::apply_mode_dialect` switches on the deployed
    dialect (RL playout net, budget-keyed c_puct, draw widening, EndTurn
    widening retune) only for `HiddenInfoMode::BeliefDraw`. Clairvoyant
    (`STS2_RUNAI_HONEST=0`), `root` and `full` keep the older defaults, so an
    arm's hidden-info mode changes its search as well as what it can see.
    """
    honest = env.get(ENV_HONEST, "belief")
    if honest == "belief":
        return "belief-draw"
    knobs = {t.strip() for t in env.get(ENV_OVERRIDES, "").split(",")}
    if honest == "0" and knobs & {"belief_dialect=1", "belief_dialect=true"}:
        return "belief-draw-clairvoyant"
    return "legacy"


def _env_fields(env: dict[str, str]) -> dict[str, Any]:
    # The SCHEMA's named spec fields are read back from the merged env, so an
    # `--env STS2_ROLLOUT_NET=...` fills them exactly like `--rollout-net`.
    net = env.get(ENV_ROLLOUT_NET)
    return {
        "search_dialect": search_dialect(env),
        "env": env,
        "overrides": env.get(ENV_OVERRIDES),
        "rollout_net": net,
        # A path names a file, not its weights: hash it so two arms that
        # point at a retrained file under the same name stay distinguishable.
        "rollout_net_sha256": _sha256_file(net) if net and os.path.isfile(net) else None,
    }


def mcts_spec(sims: int, boss_sims: int, env: dict[str, str]) -> dict[str, Any]:
    """The spec of a deployed-style `MctsCombatAgent` arm.

    Args:
        sims: Simulations per hallway / elite decision.
        boss_sims: Simulations per boss decision.
        env: The merged env from `build_env`.

    Returns:
        The spec dict.
    """
    return {"kind": "mcts", "sims": int(sims), "boss_sims": int(boss_sims), **_env_fields(env)}


def factory_spec(target: str, args: list[str], env: dict[str, str]) -> dict[str, Any]:
    """The spec of a `pkg.module:factory` arm.

    Args:
        target: `pkg.module:factory`.
        args: Repeated `--agent-arg k=v`; values are parsed as JSON when they
            parse, else kept as strings.
        env: The merged env from `build_env`.

    Returns:
        The spec dict.

    Raises:
        ValueError: If `target` is not `module:attr`, or an arg is malformed.
    """
    module, sep, attr = target.partition(":")
    if not sep or not module or not attr:
        raise ValueError(f"--agent expects 'mcts' or 'pkg.module:factory', got {target!r}")
    kwargs = {k: _arg_value(v) for k, v in parse_kv(args, "--agent-arg").items()}
    if "seed" in kwargs:
        raise ValueError("--agent-arg seed is reserved: the runner passes the cell seed")
    return {"kind": "factory", "factory": target, "args": kwargs, **_env_fields(env)}


def apply_env(env: dict[str, str]) -> list[str]:
    """Make the process env exactly the spec's engine configuration.

    Call it before the first `sts2_rs` search in the process (the Pool
    initializer), because some seams are latched on first use.

    Args:
        env: The spec's env.

    Returns:
        The inherited engine seams that were removed because the spec does
        not set them, so the caller can warn about them.
    """
    scrubbed = sorted(k for k in os.environ if is_engine_env(k) and k not in env)
    for key in scrubbed:
        del os.environ[key]
    os.environ.update(env)
    return scrubbed


def make_agent(spec: dict[str, Any], seed: int) -> CombatAgent:
    """Construct a fresh agent for one cell.

    Args:
        spec: An `mcts_spec` / `factory_spec` dict.
        seed: The cell seed; the agent's search seed derives from it, so two
            arms on the same cell see the same seed.

    Returns:
        The agent.

    Raises:
        ValueError: On an unknown spec kind.
    """
    if spec["kind"] == "mcts":
        import sts2_rs

        return sts2_rs.MctsCombatAgent(seed=seed, sims=spec["sims"], boss_sims=spec["boss_sims"])
    if spec["kind"] == "factory":
        module, _, attr = spec["factory"].partition(":")
        factory = getattr(importlib.import_module(module), attr)
        return factory(seed=seed, **spec["args"])
    raise ValueError(f"unknown agent spec kind {spec['kind']!r}")


def tree_reuse_stats(agent: CombatAgent) -> dict[str, int] | None:
    """The agent's `tree_reuse` counts (GH #830), or None when it has none.

    `MctsCombatAgent.tree_reuse_stats()` is all-zero unless the spec's
    overrides set `tree_reuse=1`; an all-zero dict (`searched == 0`) is
    reported as None so records of runs without the knob keep their shape.
    """
    stats = getattr(agent, "tree_reuse_stats", None)
    if stats is None:
        return None
    d = {str(k): int(v) for k, v in stats().items()}
    return d if d.get("searched") else None


def sims_spent(agent: CombatAgent) -> int | None:
    """Simulations the agent has requested so far, or None if it cannot say.

    `MctsCombatAgent.combat_budget_spent()` returns `(elite decisions, boss
    decisions, motif decisions, sims requested)`; forced moves count as
    decisions but request no simulations.
    """
    spent = getattr(agent, "combat_budget_spent", None)
    if spent is None:
        return None
    return int(spent()[3])
