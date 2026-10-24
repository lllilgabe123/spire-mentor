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


def active() -> bool:
    return _beta() != 0.0 or _smith_at() is not None


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
    elif surface == "rest" and _smith_at() is not None:
        hp, max_hp = obs.get("hp"), obs.get("max_hp")
        if hp is not None and max_hp and hp / max_hp >= _smith_at():
            for i, a in enumerate(actions):
                if a.get("action") == "rest_option" and int(a.get("idx", -1)) == 1:
                    out[i] += 100.0  # smith
    return out
