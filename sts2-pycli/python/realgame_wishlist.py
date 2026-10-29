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


def ranked_pool(state: dict[str, Any], character: str, policy: Any,
                pool_name: str | None = None) -> list[tuple[str, str, float]] | None:
    """``[(card id, rarity, share), ...]`` best first, cut at "skip"; None
    when there is no pool or the net cannot score it. ``pool_name`` ranks
    another pool for this character's deck ("colorless", "multiplayer")."""
    pool = _pools().get(pool_name or character.lower())
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
    if is_coop(state):
        banned = banned | COOP_BANNED
    return [r for r in rows if r[2] > skip and r[0] not in banned]


#: Cards never to take in co-op: teammates hate Fight Me (user rule,
#: 2026-10-29). Kept out of the wishlist, the give tips and the card-reward
#: and shop picks (`coop_veto`).
COOP_BANNED = {"FIGHT_ME"}


def is_coop(state: dict[str, Any]) -> bool:
    return int(state.get("player_count") or 1) >= 2


def coop_veto(state: dict[str, Any], labels: list[str], probs: list[float]) -> list[float]:
    """``probs`` with every co-op-banned card's option set to 0 and the rest
    renormalised (unchanged outside co-op, or when nothing else is left)."""
    if not is_coop(state):
        return probs
    names = {_name(cid).lower() for cid in COOP_BANNED}
    # A shop's "remove Fight Me" stays allowed: only taking one is vetoed.
    out = [0.0 if any(n in str(lbl).lower() for n in names) and "remove" not in str(lbl).lower() else p
           for lbl, p in zip(labels, probs)]
    total = sum(out)
    if total <= 0 or len(out) != len(probs):
        return probs
    return [p / total for p in out]


def _name(cid: str) -> str:
    cid = cid.removesuffix("_CARD")
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


#: The cards to give yourself over a run, per character: the best package
#: of the A10 full-run test (abtests/start_runs.py in the training folder,
#: 150 seeds each, played by the shipped AI, 2026-10-27, cards added at the
#: start), with its win rate and the no-extra-cards baseline on the same
#: seeds. `give_now` hands them out one per reward that could drop them.
START_PACKAGES = {
    "ironclad": (["DOMINATE", "FIGHT_ME", "COLOSSUS", "BATTLE_TRANCE"], 40, 13),
    "silent": (["AFTERIMAGE", "BOUNCING_FLASK", "NOXIOUS_FUMES", "FOOTWORK"], 50, 9),
    "defect": (["DEFRAGMENT", "DARKNESS", "GLACIER", "BALL_LIGHTNING"], 51, 15),
    "necrobinder": (["DEMESNE", "BORROWED_TIME", "SLEIGHT_OF_FLESH", "DEATHS_DOOR"], 27, 9),
    "regent": (["BIG_BANG", "CHARGE", "REFLECT", "BULWARK"], 26, 7),
}

#: The relic that won the most fights per character in the relic test
#: (abtests/relic_confirmed.json). Sai and Snecko Eye are Ancient relics:
#: only an act-start Ancient offers them.
START_RELICS = {
    "ironclad": "LIZARD_TAIL",
    "silent": "SAI",
    "defect": "SNECKO_EYE",
    "necrobinder": "SNECKO_EYE",
    "regent": "LIZARD_TAIL",
}

#: What each reward can really hand out: (card rarities, relic rarities,
#: what the give replaces). One card and at most one relic per reward.
MOMENTS = {
    "fight": (("Common", "Uncommon"), (), "instead of this fight's card"),
    "elite": (("Common", "Uncommon", "Rare"), ("Common", "Uncommon", "Rare"),
              "instead of this elite's card and relic"),
    "boss": (("Rare",), (), "instead of this boss's card"),
    "chest": ((), ("Common", "Uncommon", "Rare"), "instead of the chest's relic"),
    "shop": (("Common", "Uncommon", "Rare"), ("Common", "Uncommon", "Rare", "Shop"), "as if bought here"),
    "ancient": ((), ("Ancient",), "instead of the Ancient's offer"),
}

#: Where a card of each rarity can come from, for the "next give" line,
#: soonest first.
WHERE = {
    "Common": "after a normal fight",
    "Uncommon": "after a normal fight",
    "Rare": "after an elite or boss, or at a shop",
}

#: The same for relics (bosses give none).
RELIC_WHERE = {
    "Common": "after an elite, from a chest, or at a shop",
    "Uncommon": "after an elite, from a chest, or at a shop",
    "Rare": "after an elite, from a chest, or at a shop",
    "Shop": "at a shop",
    "Ancient": "at the next act-start Ancient",
}

#: (seed, act, floor) -> the give already suggested at that reward, so one
#: reward hands out one card even across its two screens (reward list,
#: then card pick) and after you have typed it.
_GIVEN: dict[tuple[str, int, int], tuple[str | None, str | None]] = {}


def _card_cmd(cid: str) -> str:
    # The game's `card` console command puts a card in your hand for this
    # fight unless it is told the pile; `Deck` keeps it for the run.
    return f"card {cid} Deck"


def _moment(state: dict[str, Any]) -> str | None:
    room = str(state.get("room_type") or "")
    kind = str(state.get("type") or "")
    reward = kind in ("card_reward", "combat_rewards", "rewards")
    if kind == "shop" or room == "Shop":
        return "shop"
    if room == "Treasure":
        return "chest"
    if realgame_ppo._ancient_boss_relic_decision(state) is not None:
        return "ancient"
    if reward:
        return {"Monster": "fight", "Elite": "elite", "Boss": "boss"}.get(room)
    return None


def give_now(state: dict[str, Any], character: str,
             rows: list[tuple[str, str, float]] | None,
             relics: dict[str, list[str]]) -> list[str]:
    """The card (and relic) to add yourself at this point of the map, with
    the console command, or which one is next and where it can drop.

    Only what this reward could really give: a normal fight a Common or
    Uncommon card, an elite any card plus a relic, a boss a Rare card, a
    chest a relic, a shop anything it sells, an act-start Ancient its own
    relics. The tested package (`START_PACKAGES`) goes first, one card per
    reward that fits its rarity; then the run AI's top pick for the deck.

    Co-op too: the game sends a `card`/`relic` command to every player's game
    but runs it for you only, so it changes your deck and nobody else's.
    Fight Me is never suggested there (`COOP_BANNED`)."""
    key = character.lower()
    coop = is_coop(state)
    banned = COOP_BANNED if coop else set()
    deck = {str(c.get("id") or "").upper()
            for c in (state.get("run_state") or {}).get("master_deck") or [] if isinstance(c, dict)}
    owned = _owned_relics(state)
    rarity = {cid: r for pool in _pools().values() for cid, r in pool}
    relic_rarity = {r[0]: r[2] for r in _relic_pool()}
    plan = [c for c in START_PACKAGES.get(key, ([],))[0] if c not in deck and c not in banned]
    plan_relic = START_RELICS.get(key)
    if plan_relic in owned:
        plan_relic = None

    def next_line(prefix: str) -> list[str]:
        # The next planned give per kind of drop, soonest first: a Common or
        # Uncommon at the next fight, a Rare at an elite/boss/shop, the relic
        # wherever its rarity drops.
        parts, said = [], set()
        order = list(WHERE)
        for c in sorted(plan, key=lambda c: order.index(rarity[c]) if rarity.get(c) in order else len(order)):
            where = WHERE.get(rarity.get(c, ""), "at a reward")
            if where not in said:
                said.add(where)
                parts.append(f"{_name(c)} {where}")
        if plan_relic:
            r = relic_rarity.get(plan_relic, "")
            parts.append(f"{_name(plan_relic)} relic {RELIC_WHERE.get(r, 'at a reward')}")
        if not parts:
            return [f"{prefix}the AI's top pick at your next reward screen (after a fight), chest or shop"]
        return [prefix + " · ".join(parts)]

    moment = _moment(state)
    if moment is None:
        return next_line("Next give: ")
    spot = (str(state.get("seed") or ""), int(state.get("act") or 0), int(state.get("floor") or 0))
    card_rar, relic_rar, how = MOMENTS[moment]
    if spot in _GIVEN:
        card, relic = _GIVEN[spot]
        if (card is None or card in deck) and (relic is None or relic in owned):
            return next_line("Given for this reward. Next: ")
    else:
        card = next((c for c in plan if rarity.get(c) in card_rar), None)
        if card is None:
            card = next((r[0] for r in rows or [] if r[1] in card_rar and r[0] not in banned), None)
        relic = plan_relic if relic_rarity.get(plan_relic or "") in relic_rar else None
        if relic is None and relic_rar:
            relic = next((relics[r][0] for r in relic_rar if relics.get(r)), None)
        _GIVEN[spot] = (card, relic)
    parts = []
    if card and card not in deck:
        parts.append(f"{_name(card)} → {_card_cmd(card)}")
    if relic and relic not in owned:
        parts.append(f"{_name(relic)} → relic add {relic}")
    if not parts:
        return next_line("Next give: ")
    return [f"Give now ({how}): " + " · ".join(parts)]


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
    # Colorless cards (shops' colorless slot, some events) and, in co-op,
    # the multiplayer-only ones, ranked the same way for this deck.
    extra = []
    for pool_name, label, n in (("colorless", "Colorless (shop)", 3), ("multiplayer", "Co-op", 3)):
        if pool_name == "multiplayer" and int(state.get("player_count") or 1) < 2:
            continue
        try:
            rows_x = ranked_pool(state, character, policy, pool_name)
        except Exception as exc:
            logger.warning("%s wishlist failed: %s", pool_name, exc)
            rows_x = None
        if rows_x:
            extra.append(f"{label}: " + ", ".join(_name(r[0]) for r in rows_x[:n]))
    if extra:
        lines_out.append(" · ".join(extra))
    try:
        relics = ranked_relics(state, character, policy)
    except Exception as exc:
        logger.warning("relic wishlist failed: %s", exc)
        relics = {}
    parts = [f"{rarity}: " + ", ".join(_name(r) for r in relics[rarity][:n])
             for rarity, n in (("Common", 2), ("Uncommon", 2), ("Rare", 2), ("Shop", 1)) if relics.get(rarity)]
    if parts:
        lines_out.append("Relics: " + " · ".join(parts))
    if lines_out:
        lines_out.extend(give_now(state, character, rows, relics))
    if not lines_out:
        return None
    return "\n".join(lines_out)
