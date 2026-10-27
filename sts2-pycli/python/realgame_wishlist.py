"""Personal build: which cards to look for, for the deck you have now.

The run AI's card-pick head is trained to choose among offered cards for the
current deck, relics, HP and act. Offering it every Common, Uncommon and
Rare of your character at once (one forward pass, a pretend card reward)
ranks the whole pool for this deck; the top of each rarity is the shopping
list shown on the map and in shops. Cards it would rank below "skip" are
left out. Relics are ranked the same way through its relic-pick head (only
ones that drop from elites, chests and shops, minus those you hold), and a
last line says when adding one yourself would match what the game could
have dropped there. Display only.

The pools (``cardpools.json.gz`` beside the pro-guide data) list each
character's Common / Uncommon / Rare cards from the beta's card pool code.
"""

from __future__ import annotations

import gzip
import json
import logging
import re
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


#: Cards that cost the Defect an orb slot (C# ``OrbCmd.RemoveSlots``). In a
#: deck that leans on orbs they switch off its Frost / Focus cards: a live
#: Aeonglass loss (2026-10-27) went to 0 slots after two Bulk Ups.
ORB_SLOT_LOSS = {"BULK_UP"}
_ORB_TEXT = re.compile(r"\b(Channel|Focus|Evoke|Orb)", re.I)
ORB_DECK_MIN = 4


def _deck_cards(state: dict[str, Any]) -> list[dict[str, Any]]:
    deck = (state.get("player") or {}).get("deck") or (state.get("run_state") or {}).get("master_deck") or []
    return [c for c in deck if isinstance(c, dict)]


def orb_cards(state: dict[str, Any]) -> int:
    """How many deck cards channel, evoke or use Focus (by card text)."""
    return sum(1 for c in _deck_cards(state)
               if str(c.get("id") or "").upper() not in ORB_SLOT_LOSS
               and _ORB_TEXT.search(re.sub(r"\[/?[a-z]+\]", "", str(c.get("description") or ""))))


def orb_slot_warning(state: dict[str, Any], offered: list[str]) -> str | None:
    """A line when a card that removes an orb slot is offered (or in hand)
    to a deck that relies on orbs."""
    hit = [cid for cid in offered if cid.upper().removeprefix("CARD.") in ORB_SLOT_LOSS]
    n = orb_cards(state)
    if not hit or n < ORB_DECK_MIN:
        return None
    have = sum(1 for c in _deck_cards(state) if str(c.get("id") or "").upper() in ORB_SLOT_LOSS)
    slots = state.get("orb_capacity")
    bit = f" You have {slots} orb slot(s) now." if isinstance(slots, int) else ""
    return (f"Careful: {_name(hit[0])} removes an orb slot, and {n} of your cards need orbs"
            f"{f' (you already run {have})' if have else ''}.{bit} At 0 slots Frost, Focus and"
            " channel cards do nothing.")


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
    # The pick head scores cards one at a time; it does not see Bulk Up
    # switching off an orb deck, so leave those out of the list.
    banned = ORB_SLOT_LOSS if orb_cards(state) >= ORB_DECK_MIN else set()
    return [r for r in rows if r[2] > skip and r[0] not in banned]


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


_RELICS: list[list[str]] | None = None

#: Relic rarities that drop from elites, chests and shops (``Shop`` ones are
#: sold only by the merchant). Ancient and Event relics come only from their
#: own events, so they are never on the list.
RELIC_RARITIES = ("Common", "Uncommon", "Rare", "Shop")


def _relic_pool() -> list[list[str]]:
    """``[[relic id, pool, rarity], ...]`` (``relicpools.json.gz``)."""
    global _RELICS
    if _RELICS is None:
        _RELICS = []
        d = _data_dir()
        for path in (d / "relicpools.json.gz", d / "relicpools.json"):
            try:
                opener = gzip.open if path.suffix == ".gz" else open
                with opener(path, "rt", encoding="utf-8") as f:
                    _RELICS = json.load(f)
                break
            except OSError:
                continue
    return _RELICS


def _owned_relics(state: dict[str, Any]) -> set[str]:
    rs = state.get("run_state") or {}
    out = set()
    for r in list(rs.get("relics") or []) + list((state.get("player") or {}).get("relics") or []):
        rid = r.get("id") if isinstance(r, dict) else r
        if rid:
            out.add(str(rid).upper().removeprefix("RELIC."))
    return out


def ranked_relics(state: dict[str, Any], character: str, policy: Any) -> dict[str, list[str]]:
    """``{rarity: [relic id, ...]}`` best first for this deck, from the run
    AI's relic-pick head, one pass per rarity (the net takes at most 160
    options at once). Only relics your character can find that you do not
    already hold."""
    owned = _owned_relics(state)
    out: dict[str, list[str]] = {}
    for rarity in RELIC_RARITIES:
        pool = [r[0] for r in _relic_pool()
                if r[2] == rarity and r[1] in ("Shared", character.title()) and r[0] not in owned]
        if not pool:
            continue
        fake = {**state, "type": "boss_relic", "options": [{"relic": {"id": rid}, "index": i} for i, rid in enumerate(pool)]}
        decision = realgame_ppo._build_boss_relic(fake)
        if decision is None:
            continue
        probs, _value = policy.distribution(decision.obs, decision.actions, decision.metrics)
        out[rarity] = [rid for rid, _p in sorted(zip(pool, probs), key=lambda x: -x[1])]
    return out


def timing(state: dict[str, Any]) -> str:
    """When a card or relic you add yourself matches what the game could
    have dropped here. Rewards follow the room you are in: normal fights
    give cards (Rares are rare there), elites a relic and better cards, the
    boss a Rare card, chests a relic, the merchant anything."""
    room = str(state.get("room_type") or "")
    kind = str(state.get("type") or "")
    if kind == "shop" or room == "Shop":
        return "Now fits: any card or relic (shop)"
    if room == "Elite":
        return "Now fits: a relic + a card, Rare OK (elite reward)"
    if room == "Boss":
        return "Now fits: a Rare card (boss reward)"
    if room == "Treasure":
        return "Now fits: one relic (chest)"
    if room == "Monster" and kind in ("card_reward", "combat_rewards", "rewards"):
        return "Now fits: a Common/Uncommon card (fight reward)"
    return "Natural times: cards after a fight (Rare after elite/boss) · relics after an elite, chest or shop"


def summary(state: dict[str, Any], character: str, policy: Any, per_rarity: int = 3) -> str | None:
    """The always-on HUD block: "Look for: Common: A, B, C · Uncommon: ... ·
    Rare: ...", a relic line, and when adding one would look natural. None
    when the pool cannot be ranked."""
    try:
        rows = ranked_pool(state, character, policy)
    except Exception as exc:
        logger.warning("wishlist failed: %s", exc)
        rows = None
    lines_out = []
    if rows:
        parts = []
        for rarity, n in (("Common", per_rarity), ("Uncommon", per_rarity), ("Rare", 2)):
            names = [_name(r[0]) for r in rows if r[1] == rarity][:n]
            if names:
                parts.append(f"{rarity}: " + ", ".join(names))
        if parts:
            lines_out.append("Look for: " + " · ".join(parts))
    try:
        relics = ranked_relics(state, character, policy)
    except Exception as exc:
        logger.warning("relic wishlist failed: %s", exc)
        relics = {}
    parts = [f"{rarity}: " + ", ".join(_name(r) for r in relics[rarity][:n])
             for rarity, n in (("Common", 2), ("Uncommon", 2), ("Rare", 2), ("Shop", 1)) if relics.get(rarity)]
    if parts:
        lines_out.append("Relics: " + " · ".join(parts))
    if not lines_out:
        return None
    lines_out.append(timing(state))
    return "\n".join(lines_out)
