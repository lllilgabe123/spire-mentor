"""Personal-build "Combo watch": how close the deck is to a loop or a known engine.

Display only (picks are unchanged: the A/B showed nudging this AI's picks
toward outside advice lost runs). Two parts:

* Loop meter. A deck goes infinite when its cards, played every turn, draw
  themselves back and pay for themselves: over the cards that stay in the
  deck (not Exhaust), draws minus discards >= the number of cards, and energy
  gained >= energy spent. Unceasing Top (draw when the hand is empty) waives
  the draw half. Cards that break a loop (Battle Trance stops further draws,
  Overclock adds a Burn) count as dead weight. Per-card cost / energy / draw come from the beta's own card
  code (``loopcards.json.gz``, made by tools/loopcards.py from the decompiled
  v0.111.0 game), so every number is the real card's, unupgraded.
* Engines: card sets checked against the beta card text (the community lists
  named several cards that are not in the beta: Soul's Power, Incense Bead,
  Panache, Soul Explosion -- left out).
"""

from __future__ import annotations

import gzip
import json
from itertools import combinations
from typing import Any

from realgame_proguide import _data_dir, _deck, _norm

#: name -> (character, pieces, payoff/what it does). Pieces are normalized
#: card names; "relic:" pieces are relic ids.
ENGINES: list[tuple[str, str, list[str], str]] = [
    ("Bloodletting engine", "ironclad",
     ["bloodletting", "pommelstrike", "shrugitoff", "spite"],
     "Bloodletting is 0 cost, +2 energy and stays in your deck; draw cards cycle back to it. "
     "Spite hits more after you lose HP. Costs 3 HP a loop: keep the deck thin"),
    ("Corruption exhaust engine", "ironclad",
     ["corruption", "darkembrace", "feelnopain", "burningpact", "secondwind"],
     "Corruption makes Skills free and exhausts them; Dark Embrace draws and Feel No Pain blocks per exhaust"),
    ("Barricade block", "ironclad",
     ["barricade", "entrench", "bodyslam", "shrugitoff", "flamebarrier"],
     "Barricade keeps Block between turns, Entrench doubles it, Body Slam hits for it"),
    ("Sly energy loop", "silent",
     ["acrobatics", "calculatedgamble", "tactician", "reflex", "grandfinale"],
     "Discarding a Sly card plays it free: Tactician = +1 energy, Reflex = draw 2. "
     "Grand Finale (60 to all) is playable when your draw pile is empty"),
    ("Zero-cost cycle", "silent",
     ["escapeplan", "adrenaline", "backflip", "afterimage", "grandfinale"],
     "Cheap draw cycles the deck; Afterimage blocks per card played"),
    ("Claw loop", "defect",
     ["claw", "allforone", "ftl", "hologram", "relic:UNCEASING_TOP"],
     "Every Claw played makes all Claws stronger; All For One returns every 0-cost card from the discard pile"),
    ("Soul draw engine", "necrobinder",
     ["severance", "reave", "borrowedtime"],
     "Severance and Reave add 0-cost Souls (draw 2) to your piles; Borrowed Time gives 4 energy now but makes cards cost more"),
    ("Big Bang draw engine", "regent",
     ["bigbang", "glow", "relic:LETTER_OPENER"],
     "0-cost draw + energy; Letter Opener hits all enemies every 3 Skills"),
]

#: Relics that make loops easier, with why.
LOOP_RELICS = {
    "UNCEASING_TOP": "draws a card whenever your hand is empty",
    "ICE_CREAM": "unspent energy carries over",
    "LETTER_OPENER": "every 3 Skills deal damage to all enemies",
    "KUNAI": "Dexterity every 3 Attacks",
    "SHURIKEN": "Strength every 3 Attacks",
    "NUNCHAKU": "energy every 10 Attacks",
    "ORNAMENTAL_FAN": "Block every 3 Attacks",
}

_CARDS: dict[str, list[Any]] | None = None


def _cards() -> dict[str, list[Any]]:
    """{normalized card: [pool, cost, type, x, energy, draw, discard, exhaust, conditional, name, breaks_loop]}."""
    global _CARDS
    if _CARDS is None:
        _CARDS = {}
        d = _data_dir()
        for path in (d / "loopcards.json.gz", d / "loopcards.json"):
            try:
                opener = gzip.open if path.suffix == ".gz" else open
                with opener(path, "rt", encoding="utf-8") as f:
                    _CARDS = json.load(f)
                break
            except OSError:
                continue
    return _CARDS


def _relics(state: dict[str, Any]) -> set[str]:
    rows = [*((state.get("player") or {}).get("relics") or []), *((state.get("run_state") or {}).get("relics") or [])]
    return {str(r.get("id") or "").upper() for r in rows if isinstance(r, dict)}


def _key(card: dict[str, Any]) -> str:
    return _norm(str(card.get("id") or card.get("name") or ""))


def loop_gaps(keys: list[str], relics: set[str]) -> tuple[int, int, int] | None:
    """(draw gap, energy gap, cycle size) for a deck of normalized names, or
    None without card data. A gap <= 0 is met."""
    table = _cards()
    if not table:
        return None
    n = draw = energy = cost = 0
    for k in keys:
        row = table.get(k)
        if row is None:
            n += 1  # curse/status/unknown: dead weight in the cycle
            cost += 1
            continue
        _pool, c, ctype, x, e, dr, disc, exhaust, _cond = row[:9]
        if len(row) > 10 and row[10]:
            dr = e = 0  # breaks a loop (Battle Trance's no-draw, Overclock's Burn): count it as dead weight
        if exhaust:
            draw += dr - disc  # one-shot: helps once, then leaves the cycle
            energy += e - max(c, 0)
            continue
        n += 1
        draw += dr - disc
        energy += e
        cost += 0 if x else max(c, 0)
    draw_gap = 0 if "UNCEASING_TOP" in relics else n - draw
    return draw_gap, cost - energy, n


def _engine_lines(character: str, keys: set[str], relics: set[str], offered: list[str]) -> list[str]:
    out = []
    for name, char, pieces, how in ENGINES:
        if char != character:
            continue
        have = [p for p in pieces if (p[6:] in relics if p.startswith("relic:") else p in keys)]
        adds = [o for o in offered if o in pieces and o not in keys]
        if len(have) >= 2 or (have and adds):
            missing = [p for p in pieces if p not in have]
            line = f"Combo watch, {name}: {len(have)}/{len(pieces)} pieces"
            if missing:
                line += " (missing " + ", ".join(_pretty(m) for m in missing) + ")"
            if adds:
                line += ". OFFERED: " + ", ".join(_pretty(a) for a in adds)
            out.append(line + ". " + how)
    return out


def _pretty(key: str) -> str:
    if key.startswith("relic:"):
        return key[6:].replace("_", " ").title() + " (relic)"
    row = _cards().get(key)
    return row[9] if row and len(row) > 9 else key.title()


def _meter_line(keys: list[str], relics: set[str]) -> str | None:
    gaps = loop_gaps(keys, relics)
    if gaps is None:
        return None
    draw_gap, energy_gap, n = gaps
    if draw_gap <= 0 and energy_gap <= 0:
        return f"Loop meter: your {n}-card cycle pays for itself (draw and energy). Infinite possible: play it out"
    if max(draw_gap, 0) + max(energy_gap, 0) > 8:
        return None  # nowhere near; say nothing
    parts = []
    if draw_gap > 0:
        parts.append(f"{draw_gap} more draw")
    if energy_gap > 0:
        parts.append(f"{energy_gap} more energy")
    return (f"Loop meter: {n}-card cycle needs " + " and ".join(parts) +
            " per cycle to go infinite. Removing a card that draws nothing closes the draw gap by 1 and the energy gap by its cost")


def card_reward_lines(state: dict[str, Any], character: str) -> list[str]:
    character = character.lower()
    deck = [_key(c) for c in _deck(state)]
    relics = _relics(state)
    offered = [_key(c) for c in state.get("cards") or [] if isinstance(c, dict)]
    out = _engine_lines(character, set(deck), relics, offered)
    core = core_line(deck, relics)
    if core:
        out.append(core)
    meter = _meter_line(deck, relics)
    if meter:
        out.append(meter)
        base = loop_gaps(deck, relics)
        better = []
        for c, k in zip(state.get("cards") or [], offered):
            g = loop_gaps([*deck, k], relics)
            if base and g and max(g[0], 0) + max(g[1], 0) < max(base[0], 0) + max(base[1], 0):
                better.append(str(c.get("name") or _pretty(k)))
        if better:
            out.append("Moves you toward a loop: " + ", ".join(better))
    return out


def deck_lines(state: dict[str, Any], character: str) -> list[str]:
    """Shop / map: engines in progress, the loop meter and loop relics."""
    character = character.lower()
    deck = [_key(c) for c in _deck(state)]
    relics = _relics(state)
    out = _engine_lines(character, set(deck), relics, [])
    core = core_line(deck, relics)
    if core:
        out.append(core)
    meter = _meter_line(deck, relics)
    if meter:
        out.append(meter)
    owned = [f"{r.replace('_', ' ').title()} ({why})" for r, why in LOOP_RELICS.items() if r in relics]
    if owned and out:
        out.append("Your loop relics: " + "; ".join(owned))
    return out


def find_core(keys: list[str], relics: set[str], max_size: int = 4) -> list[str] | None:
    """The smallest set of cards (instances from ``keys``) that loops on its
    own: draws >= its size (waived with Unceasing Top), energy gained >=
    energy spent, no Exhaust, and something that deals damage (an Attack,
    or Skills with Letter Opener). None when no such set exists."""
    table = _cards()
    counts: dict[str, int] = {}
    for k in keys:
        row = table.get(k)
        # Loops need cards that stay (no Exhaust), a known cost (no X) and
        # nothing that stops the loop (no-draw, junk cards, cost increases).
        if row and not row[7] and not row[3] and not (len(row) > 10 and row[10]):
            counts[k] = counts.get(k, 0) + 1
    pool = [k for k, n in sorted(counts.items()) for _ in range(min(n, max_size))]
    top = "UNCEASING_TOP" in relics
    letter = "LETTER_OPENER" in relics
    best: tuple[int, int, list[str]] | None = None

    # One card cannot draw itself back: it is still being played when its
    # draw resolves (engine-checked: 1 Flash of Steel stalls, 2 loop). Only
    # Unceasing Top, which draws after the hand empties, makes one enough.
    for size in range(1 if top else 2, max_size + 1):
        for combo in set(combinations(pool, size)):
            rows = [table[k] for k in combo]
            draw = sum(r[5] - r[6] for r in rows)
            energy = sum(r[4] for r in rows) - sum(max(r[1], 0) for r in rows)
            hits = any(r[2] == "Attack" for r in rows) or (letter and any(r[2] == "Skill" for r in rows))
            if hits and energy >= 0 and (top or draw >= size):
                slack = energy + (draw - size)
                if best is None or slack > best[1]:
                    best = (size, slack, list(combo))
        if best is not None:
            return best[2]
    return None


def core_line(keys: list[str], relics: set[str], in_fight: bool = False) -> str | None:
    """"Infinite potential: A + B + C" with what stands in the way."""
    core = find_core(keys, relics)
    if not core:
        return None
    names = " + ".join(_pretty(k) for k in core)
    gaps = loop_gaps(keys, relics)
    if gaps and gaps[0] <= 0 and gaps[1] <= 0:
        return (f"INFINITE: your whole {'fight cycle' if in_fight else 'deck'} pays for itself ({names} is the core). "
                "Play draw and energy cards first, attacks as energy allows")
    rest = len([k for k in keys if k not in core]) if gaps else 0
    where = "in this fight's cards" if in_fight else "in your deck"
    verb = "loops" if len(core) == 1 else "loop"
    # Engine-checked (Flash of Steel x2, Flash + Finesse, Restlessness +
    # Pommel Strike vs the 630-HP Queen): turn-1 kills with up to 4 other
    # cards in the cycle, none with 5 -- the hand fills before the loop runs.
    tail = ("it should already run: play the loop cards first" if rest <= 4 else
            f"it needs the cycle down to about {len(core) + 4} cards: exhaust or remove {rest - 4} more")
    return f"Infinite potential {where}: {names} {verb} on their own; {tail}"


def combat_lines(state: dict[str, Any]) -> list[str]:
    """In a fight: the infinite hint over the cards still in play (hand,
    draw and discard piles; exhausted cards are gone)."""
    cards = [*(state.get("hand") or []),
             *((state.get("draw_pile") or {}).get("cards") or []),
             *((state.get("discard_pile") or {}).get("cards") or [])]
    keys = [_key(c) for c in cards if isinstance(c, dict)]
    line = core_line(keys, _relics(state), in_fight=True)
    return [line] if line else []
