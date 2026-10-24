"""Personal-build "Pro guide": what strong beta players do, shown next to advice.

Display only. Adds lines to card-reward and rest-site advice; never changes a
ranking or a pick.

* Card rewards: for each offered card, how often solo A10 players above a 50%
  win rate on v0.111.0 take it (overall and in this act), from spire-codex
  card metrics; which of the character's archetypes the card belongs to and
  how many of that archetype's cards the deck already holds; and beta-patch
  notes (v0.111.0 Expect a Fight / Hyperbeam).
* Rest sites: those players' habit (upgrade on ~70% of rest-site visits,
  resting mostly at low HP) against your HP now.

The pick tables are personal data (not shipped in the repo): one
``<character>.json.gz`` per character in the ``proguide`` folder next to the
advisor (``mods/SpireMentor/proguide``), or ``$SPIRE_PROGUIDE_DIR``. Gzipped
because the game reads every ``.json`` under a mod folder as a manifest.
Archetype lists are from community guides, checked against the v0.111.0
card text.
"""

from __future__ import annotations

import gzip
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

ARCHETYPES: dict[str, dict[str, list[str]]] = {
    "ironclad": {
        "Exhaust": ["Feel No Pain", "Dark Embrace", "Corruption", "Burning Pact", "Second Wind",
                    "Fiend Fire", "True Grit", "Forgotten Ritual", "Ashen Strike", "Pact's End",
                    "Stoke", "Offering", "Thrash", "Havoc", "Brand"],
        "Strength": ["Inflame", "Demon Form", "Rupture", "Brand", "Fight Me!", "Sword Boomerang",
                     "Whirlwind", "Twin Strike", "Pommel Strike", "Expect a Fight"],
        "Block": ["Barricade", "Body Slam", "Entrench", "Impervious", "Colossus", "Unmovable",
                  "Flame Barrier", "Shrug It Off", "Stone Armor", "Juggernaut"],
        "Self-damage": ["Bloodletting", "Hemokinesis", "Rupture", "Offering", "Crimson Mantle",
                        "Inferno", "Brand", "Spite"],
    },
    "defect": {
        "Frost": ["Glacier", "Coolheaded", "Chill", "Cold Snap", "Hailstorm", "Ice Lance", "Coolant"],
        "Lightning": ["Ball Lightning", "Storm", "Thunder", "Tempest", "Voltaic", "Lightning Rod",
                      "Tesla Coil"],
        "Dark": ["Darkness", "Loop", "Shadow Shield", "Consuming Shadow", "Multi-Cast"],
        "Scaling": ["Defragment", "Capacitor", "Echo Form", "Machine Learning", "Creative AI",
                    "Signal Boost", "Buffer"],
    },
}

#: Cards that grant Strength (for the v0.111.0 Expect a Fight note).
STRENGTH_SOURCES = {"inflame", "demonform", "rupture", "brand", "fightme", "spite"}

#: Top players' rest-site habit, v0.111.0 solo A10 >50% WR (1,843 runs).
REST_HABIT = "Top players upgrade on 70% of rest-site visits and rest mostly when low"

_TABLES: dict[str, dict[str, list[float]]] = {}


def _norm(name: str) -> str:
    s = name.lower().removeprefix("card.")
    s = re.sub(r"_card$", "", s)
    return re.sub(r"[^a-z0-9]", "", s)


def _data_dir() -> Path:
    env = os.environ.get("SPIRE_PROGUIDE_DIR")
    if env:
        return Path(env)
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent.parent / "proguide"
    return Path(r"E:\SpireMentorTraining\data")


def _table(character: str) -> dict[str, list[float]]:
    """{normalized card: [elo, pick, act1, act2, act3, offered]} or {}."""
    key = character.lower()
    if key not in _TABLES:
        rows: list[list[Any]] = []
        d = _data_dir()
        for path in (d / f"{key}.json.gz", d / f"codex_{key}_a10_wr50_v0111.json"):
            try:
                opener = gzip.open if path.suffix == ".gz" else open
                with opener(path, "rt", encoding="utf-8") as f:
                    rows = json.load(f)["cards"]
                break
            except OSError:
                continue
        _TABLES[key] = {_norm(r[0]): [float(x) for x in r[1:]] for r in rows}
    return _TABLES[key]


def _deck_names(state: dict[str, Any]) -> list[str]:
    deck = (state.get("player") or {}).get("deck") or ((state.get("run_state") or {}).get("master_deck")) or []
    return [_norm(c.get("id") or c.get("name") or "") for c in deck if isinstance(c, dict)]


def _builds(character: str, deck: list[str]) -> dict[str, int]:
    out = {}
    for name, cards in ARCHETYPES.get(character, {}).items():
        members = {_norm(c) for c in cards}
        out[name] = sum(1 for c in deck if c in members)
    return out


def card_reward_lines(state: dict[str, Any], character: str) -> list[str]:
    character = character.lower()
    table = _table(character)
    deck = _deck_names(state)
    builds = _builds(character, deck)
    act = int(state.get("act") or (state.get("live_context") or {}).get("act") or 1)
    lines = []
    active = sorted(((n, k) for k, n in builds.items() if n >= 2), reverse=True)
    if active:
        lines.append("Your build: " + ", ".join(f"{k} ({n})" for n, k in active))
    elif builds:
        lines.append("Your build: nothing yet. Pros take strong all-round cards in act 1 and commit in act 2")
    for card in state.get("cards") or []:
        name = card.get("name") or card.get("id") or "?"
        key = _norm(card.get("id") or name)
        bits = []
        row = table.get(key)
        if row:
            act_pick = row[1 + min(act, 3)]
            bits.append(f"pros take {act_pick:.0f}% in act {act} ({row[1]:.0f}% overall)")
        for build, cards in ARCHETYPES.get(character, {}).items():
            if key in {_norm(c) for c in cards}:
                n = builds.get(build, 0)
                bits.append(f"fits your {build} ({n})" if n >= 2 else f"{build} card")
        if key == "expectafight":
            strength = sum(1 for c in deck if c in STRENGTH_SOURCES)
            bits.append("beta: 3 energy, Block scales with Strength; " +
                        ("you have no Strength source yet" if strength == 0 else f"you have {strength} Strength source(s)"))
        if key == "hyperbeam":
            bits.append("beta: Focus loss now lasts one turn; a finisher vs groups, risky if Frost is your only Block")
        if bits:
            lines.append(f"{name}: " + "; ".join(bits))
    return lines


def rest_site_lines(state: dict[str, Any]) -> list[str]:
    p = state.get("player") or (state.get("run_state") or {})
    hp, max_hp = p.get("hp"), p.get("max_hp")
    if not hp or not max_hp:
        return [REST_HABIT + "."]
    frac = hp / max_hp
    lean = "they would usually upgrade here" if frac >= 0.5 else "they would usually rest here"
    return [f"{REST_HABIT}. You're at {frac:.0%} HP: {lean}."]
