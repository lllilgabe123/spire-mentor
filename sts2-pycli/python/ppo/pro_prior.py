"""Personal-build nudges toward strong players' run choices, for A/B tests.

Both are OFF unless their environment variable is set, so training and every
existing caller are untouched:

* ``STS2_PRO_PRIOR=<beta>`` -- card rewards. Adds ``beta`` times the
  Elo-implied log-odds of each offered card, centred on the offered cards'
  mean, to the run policy's logits. Elo is spire-codex's "what players take
  when offered" rating (``STS2_PRO_PRIOR_FILE``: a JSON with ``cards`` rows
  ``[name, elo, ...]``; read for solo A10 players above a 50% win rate on
  v0.111.0). An Elo gap ``d`` means odds of ``10**(d/400)``, i.e. log-odds
  ``d*ln10/400``, so beta = 1 adds the full pro preference and beta = 0.5
  half of it. Skip and cards missing from the table get no nudge.
* ``STS2_SAFE_ROUTE=<fraction>`` -- map paths. Below this HP fraction, each
  extra Elite a branch forces before the boss costs that option one logit.
* ``STS2_SMITH_AT=<fraction>`` -- rest sites. Upgrade instead of healing when
  HP is at or above this fraction of max (pros start smithing around 0.5;
  the shipped Ironclad champion heals until about 0.75).
"""

from __future__ import annotations

import json
import math
import os
import re
from typing import Any

import numpy as np

_TABLE: dict[str, float] | None = None


def _norm(name: str) -> str:
    """One key for a display name ("Fight Me!") and an engine id
    ("CARD.FIGHT_ME", "CARD.COLOSSUS_CARD")."""
    s = name.lower()
    s = s.removeprefix("card.")
    s = re.sub(r"_card$", "", s)
    return re.sub(r"[^a-z0-9]", "", s)


def _table() -> dict[str, float]:
    global _TABLE
    if _TABLE is None:
        path = os.environ.get("STS2_PRO_PRIOR_FILE", "")
        rows = json.load(open(path, encoding="utf-8"))["cards"] if path else []
        _TABLE = {_norm(r[0]): float(r[1]) for r in rows}
    return _TABLE


def _beta() -> float:
    return float(os.environ.get("STS2_PRO_PRIOR", "0") or 0)


def _smith_at() -> float | None:
    raw = os.environ.get("STS2_SMITH_AT", "")
    return float(raw) if raw else None


def _safe_route() -> float | None:
    raw = os.environ.get("STS2_SAFE_ROUTE", "")
    return float(raw) if raw else None


def active() -> bool:
    return _beta() != 0.0 or _smith_at() is not None or _safe_route() is not None


def _forced_elites(m: dict[str, Any]) -> dict[tuple[int, int], int]:
    """Fewest Elite rooms on any path from each node to the act's end."""
    nodes = {(n["row"], n["col"]): n for row in m.get("rows") or [] for n in row}
    memo: dict[tuple[int, int], int] = {}

    def walk(key: tuple[int, int]) -> int:
        if key not in memo:
            n = nodes[key]
            kids = [(c["row"], c["col"]) for c in n.get("children") or [] if (c["row"], c["col"]) in nodes]
            memo[key] = int(n.get("type") == "Elite") + (min(walk(k) for k in kids) if kids else 0)
        return memo[key]

    for k in sorted(nodes, reverse=True):
        walk(k)
    return memo


def _route_options(m: dict[str, Any]) -> list[dict[str, Any]]:
    """The nodes a ``move`` can reach now: the current node's children, or
    the first row (after a lone Ancient start node) before the first move."""
    rows = m.get("rows") or []
    nodes = {(n["row"], n["col"]): n for row in rows for n in row}
    cur = next((n for row in rows for n in row if n.get("current")), None)
    if cur is None and rows and len(rows[0]) == 1 and rows[0][0].get("type") == "Ancient":
        cur = rows[0][0]
    if cur is not None:
        return [nodes[(c["row"], c["col"])] for c in cur.get("children") or [] if (c["row"], c["col"]) in nodes]
    return list(rows[0]) if rows else []


def adjust(obs: dict[str, Any], actions: list[dict[str, Any]], logits: np.ndarray) -> np.ndarray:
    """``logits`` (one per action) with the configured nudges applied."""
    surface = obs.get("surface")
    out = np.array(logits, dtype=np.float64, copy=True)
    if surface == "card-pick" and _beta() != 0.0:
        ids = obs.get("card_reward_option_ids") or []
        table = _table()
        elo: dict[int, float] = {}
        for i, a in enumerate(actions):
            if a.get("action") == "pick_card_reward" and a.get("idx") is not None:
                k = int(a["idx"])
                if 0 <= k < len(ids) and _norm(str(ids[k])) in table:
                    elo[i] = table[_norm(str(ids[k]))]
        if len(elo) >= 2:
            mean = sum(elo.values()) / len(elo)
            for i, e in elo.items():
                out[i] += _beta() * (e - mean) * math.log(10) / 400.0
    elif surface == "path" and _safe_route() is not None and obs.get("map"):
        # Below the HP fraction, steer away from branches that force more
        # Elites before the boss: one logit per extra forced Elite.
        hp, max_hp = obs.get("hp"), obs.get("max_hp")
        if hp is not None and max_hp and hp / max_hp < _safe_route():
            forced = _forced_elites(obs["map"])
            by_col = {n["col"]: forced.get((n["row"], n["col"]), 0) for n in _route_options(obs["map"])}
            idx = [i for i, a in enumerate(actions) if a.get("action") == "move" and a.get("col") in by_col]
            if len(idx) >= 2:
                least = min(by_col[actions[i]["col"]] for i in idx)
                for i in idx:
                    out[i] -= by_col[actions[i]["col"]] - least
    elif surface == "rest" and _smith_at() is not None:
        hp, max_hp = obs.get("hp"), obs.get("max_hp")
        if hp is not None and max_hp and hp / max_hp >= _smith_at():
            for i, a in enumerate(actions):
                if a.get("action") == "rest_option" and int(a.get("idx", -1)) == 1:
                    out[i] += 100.0  # smith
    return out
