"""Personal build: which cards to look for, for the deck you have now.

The run AI's card-pick head is trained to choose among offered cards for the
current deck, relics, HP and act. Offering it every Common, Uncommon and
Rare of your character at once (one forward pass, a pretend card reward)
ranks the whole pool for this deck; the top of each rarity is the shopping
list shown on the map and in shops. Cards it would rank below "skip" are
left out. Display only.

The pools (``cardpools.json.gz`` beside the pro-guide data) list each
character's Common / Uncommon / Rare cards from the beta's card pool code.
"""

from __future__ import annotations

import gzip
import json
import logging
from typing import Any

import realgame_ppo
from realgame_proguide import _data_dir

logger = logging.getLogger(__name__)

_POOLS: dict[str, list[list[str]]] | None = None


def _pools() -> dict[str, list[list[str]]]:
    global _POOLS
    if _POOLS is None:
        _POOLS = {}
        d = _data_dir()
        for path in (d / "cardpools.json.gz", d / "cardpools.json"):
            try:
                opener = gzip.open if path.suffix == ".gz" else open
                with opener(path, "rt", encoding="utf-8") as f:
                    _POOLS = json.load(f)
                break
            except OSError:
                continue
    return _POOLS


def ranked_pool(state: dict[str, Any], character: str, policy: Any) -> list[tuple[str, str, float]] | None:
    """``[(card id, rarity, share), ...]`` best first, cut at "skip"; None
    when there is no pool or the net cannot score it."""
    pool = _pools().get(character.lower())
    if not pool:
        return None
    fake = {**state, "type": "card_reward", "cards": [{"id": cid} for cid, _ in pool], "can_skip": True}
    decision = realgame_ppo._build_card_reward(fake)
    if decision is None:
        return None
    probs, _value = policy.distribution(decision.obs, decision.actions, decision.metrics)
    skip = probs[-1]
    rows = sorted(((pool[i][0], pool[i][1], float(p)) for i, p in enumerate(probs[:-1])), key=lambda r: -r[2])
    return [r for r in rows if r[2] > skip]


def _name(cid: str) -> str:
    return cid.replace("_", " ").title().replace("Of The", "of the").replace(" Of ", " of ")


def lines(state: dict[str, Any], character: str, policy: Any, top: int = 3) -> list[str]:
    """One HUD line: the best Common/Uncommon and Rare picks for this deck."""
    try:
        rows = ranked_pool(state, character, policy)
    except Exception as exc:
        logger.warning("wishlist failed: %s", exc)
        return []
    if not rows:
        return []
    owned = {str(c.get("id") or "").upper() for c in (state.get("run_state") or {}).get("master_deck") or []}
    common = [r for r in rows if r[1] in ("Common", "Uncommon")][:top]
    rare = [r for r in rows if r[1] == "Rare"][:2]
    if not common and not rare:
        return []

    def fmt(r: tuple[str, str, float]) -> str:
        return _name(r[0]) + (" (have)" if r[0] in owned else "")
    text = "Looking for: " + ", ".join(fmt(r) for r in common)
    if rare:
        text += " · Rare: " + ", ".join(fmt(r) for r in rare)
    return [text + " (the run AI's top picks for this deck)"]
