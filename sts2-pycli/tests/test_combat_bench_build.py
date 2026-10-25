"""`ppo.combat_bench build` extraction rules (GH #830), on synthetic run files.

No corpus and no engine: every run here is a small hand-made dict shaped like a
spire-codex `.run.gz`, so each test pins one rule (quit vs death, belt replay,
Fairy, Petrified Toad, removed relics, wax melts, rosters, mods, pseudonyms,
the rollout_learn holdout) to a number worked out by hand.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
from ppo import pilot_human_decks as phd
from ppo import rollout_learn
from ppo.combat_bench import build as B
from ppo.combat_bench import holdout as H

STARTER = [{"id": "CARD.STRIKE_IRONCLAD", "floor_added_to_deck": 1}] * 5 + [
    {"id": "CARD.DEFEND_IRONCLAD", "floor_added_to_deck": 1}
] * 4
KNOWN = {
    "cards": frozenset({"STRIKE_IRONCLAD", "DEFEND_IRONCLAD", "JUGGLING"}),
    "relics": frozenset(
        {"BURNING_BLOOD", "BLACK_BLOOD", "PETRIFIED_TOAD", "POTION_BELT", "TOY_BOX", "VAJRA"}
    ),
    "potions": frozenset(
        {"FIRE_POTION", "BLOCK_POTION", "FAIRY_IN_A_BOTTLE", "POTION_SHAPED_ROCK"}
    ),
}


def _pt(
    room: str | None = None,
    enc: str = "ENCOUNTER.NIBBITS_WEAK",
    mons: list[str] | None = None,
    *,
    hp: int = 50,
    dmg: int = 0,
    healed: int = 0,
    event: bool = False,
    **stats: Any,
) -> dict[str, Any]:
    """One map point: `room` None is a non-combat point (a shop, say)."""
    rooms = []
    if event:
        rooms.append({"room_type": "event", "model_id": "EVENT.X", "turns_taken": 0})
    if room is not None:
        rooms.append(
            {
                "room_type": room,
                "model_id": enc,
                "monster_ids": mons if mons is not None else ["MONSTER.NIBBIT"],
                "turns_taken": 3,
            }
        )
    elif not event:
        rooms.append({"room_type": "shop", "turns_taken": 0})
    ps = {"current_hp": hp, "max_hp": 80, "damage_taken": dmg, "hp_healed": healed}
    ps |= {"max_hp_gained": 0, "max_hp_lost": 0, **stats}
    return {"map_point_type": room or "shop", "player_stats": [ps], "rooms": rooms}


def _run(
    points: list[dict[str, Any]],
    *,
    relics: list[dict[str, Any]] | None = None,
    potions: list[str] = (),
    slots: int = 2,
    abandoned: bool = False,
    win: bool = False,
    modifiers: list[Any] | None = None,
    username: str | None = "somebody",
    deck: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """A single-player A10 v0.107.1 Ironclad run. Floor 1 is the Neow point."""
    return {
        "build_id": "v0.107.1",
        "game_mode": "standard",
        "ascension": 10,
        "was_abandoned": abandoned,
        "win": win,
        "modifiers": modifiers or [],
        "username": username,
        "players": [
            {
                "character": "CHARACTER.IRONCLAD",
                "deck": STARTER if deck is None else deck,
                "relics": relics
                if relics is not None
                else [{"id": "RELIC.BURNING_BLOOD", "floor_added_to_deck": 1}],
                "potions": [{"id": f"POTION.{p}", "slot_index": i} for i, p in enumerate(potions)],
                "max_potion_slot_count": slots,
            }
        ],
        "map_point_history": [[_pt(None, hp=80), *points]],
    }


def _records(d: dict[str, Any], cohort: str = "pro") -> list[dict[str, Any]]:
    recs, why, _ = B.bench_fights(d, "/x/abc.run.gz", cohort, KNOWN)
    assert why is None
    return recs


# ── abandon / quit / death ───────────────────────────────────────────────


def test_abandoned_runs_are_skipped_by_default_and_kept_on_request() -> None:
    d = _run([_pt("monster")], abandoned=True)
    assert phd.run_rejection(d, 10, "v0.107.1") == "abandoned"
    assert phd.run_rejection(d, 10, "v0.107.1", include_abandoned=True) is None


def test_mid_fight_quit_is_a_quit_never_a_death() -> None:
    # Floor 3: abandoned in combat at 0 HP after only 5 damage from 40 HP.
    d = _run([_pt("monster", hp=40), _pt("elite", hp=0, dmg=5)], abandoned=True)
    rec = _records(d)[-1]
    assert rec["human"] == rec["human"] | {"quit": True, "died": False, "hp_after": 0}
    assert rec["run_abandoned"] is True
    # The completed fight before it is an ordinary fight.
    assert _records(d)[0]["human"]["quit"] is False


def test_lethal_last_fight_of_a_finished_run_is_a_death() -> None:
    d = _run([_pt("monster", hp=40), _pt("boss", hp=0, dmg=55)])
    last = _records(d)[-1]["human"]
    assert (last["died"], last["quit"]) == (True, False)


def test_a_death_without_lethal_damage_taken_is_flagged() -> None:
    # The Insatiable's Sandpit kills via CreatureCmd.Kill: 0 HP from 41 after
    # 31 damage, in a finished run. A death, flagged (R8).
    d = _run([_pt("monster", hp=41), _pt("boss", hp=0, dmg=31)])
    last = _records(d)[-1]
    assert last["human"]["died"] and "death_hp_unlogged" in last["flags"]
    d = _run([_pt("monster", hp=40), _pt("boss", hp=0, dmg=55)])
    assert "death_hp_unlogged" not in _records(d)[-1]["flags"]


def test_abandoned_after_a_won_fight_is_neither() -> None:
    d = _run([_pt("monster", hp=40), _pt("monster", hp=31, dmg=9)], abandoned=True)
    last = _records(d)[-1]["human"]
    assert (last["died"], last["quit"]) == (False, False)


def test_a_fight_won_then_abandoned_on_its_reward_screen_is_not_a_quit() -> None:
    # c794e5ff2fb721f3.run.gz:4: 4 damage from 40 HP, Burning Blood's 6
    # healed, 7 gold and a potion taken, then the run abandoned: HP reads 0.
    known = KNOWN | {
        "relics": KNOWN["relics"] | {"MAW_BANK"},
        "cards": KNOWN["cards"] | {"HAND_OF_GREED"},
    }

    def last(*, frozen: bool = False, deck: list[Any] | None = None, **stats: Any) -> Any:
        relics = [{"id": "RELIC.BURNING_BLOOD"}]
        if stats.pop("maw", False):
            relics.append({"id": "RELIC.MAW_BANK"})
        pts = [_pt("monster", hp=40), _pt("monster", hp=0, dmg=4, healed=6, **stats)]
        d = _run(pts, abandoned=True, relics=relics, deck=deck)
        recs, _, _ = B.bench_fights(d, "/x/abc.run.gz", "pro", known, frozen_v1=frozen)
        return recs[-1]

    won = last(gold_gained=7, potion_choices=_pick("FIRE_POTION"))
    assert (won["human"]["quit"], won["human"]["died"]) == (False, False)
    assert won["human"]["hp_after"] == 40 - 4 + 6 and "abandoned_after_win" in won["flags"]
    for evidence in (
        {"card_choices": [{"card": {}, "was_picked": False}]},
        {"relic_choices": [{}]},
    ):
        assert not last(**evidence)["human"]["quit"]
    # v1's rule: a quit.
    old = last(frozen=True, gold_gained=7)
    assert old["human"]["quit"] and old["human"]["hp_after"] == 0
    assert "abandoned_after_win" not in old["flags"]
    # No reward: in-combat potion generation logs picks, Maw Bank pays 12 on
    # entering the room, Hand of Greed pays on a kill.
    greed = [*STARTER, {"id": "CARD.HAND_OF_GREED", "floor_added_to_deck": 1}]
    for rec in (
        last(potion_choices=_pick("FIRE_POTION")),
        last(gold_gained=12, maw=True),
        last(gold_gained=20, deck=greed),
        last(),
    ):
        assert rec["human"]["quit"] and "abandoned_after_win" not in rec["flags"]
    assert not last(gold_gained=19, maw=True)["human"]["quit"]  # a reward beside Maw Bank's


# ── potion belt ──────────────────────────────────────────────────────────


def _pick(*ids: str) -> list[dict[str, Any]]:
    return [{"choice": f"POTION.{p}", "was_picked": True} for p in ids]


def test_belt_replay_slots_and_entry_belts() -> None:
    d = _run(
        [
            _pt("monster", potion_choices=_pick("FIRE_POTION")),  # floor 2
            _pt(None, potion_choices=_pick("BLOCK_POTION")),  # floor 3: shop
            _pt(
                "elite",
                potion_used=["POTION.FIRE_POTION"],
                relic_choices=[{"choice": "RELIC.POTION_BELT", "was_picked": True}],
            ),  # floor 4
            _pt("monster"),  # floor 5
        ],
        relics=[
            {"id": "RELIC.BURNING_BLOOD", "floor_added_to_deck": 1},
            {"id": "RELIC.POTION_BELT", "floor_added_to_deck": 4},
        ],
        potions=["BLOCK_POTION"],
        slots=4,
    )
    belts, slots, err = B.belt_timeline(d)
    assert err is None
    assert belts[2] == [] and belts[4] == ["FIRE_POTION", "BLOCK_POTION"]
    assert belts[5] == ["BLOCK_POTION"]
    # A10 starts on 2 slots (TightBelt); Potion Belt adds 2 after floor 4's entry.
    assert (slots[2], slots[4], slots[5]) == (2, 2, 4)


def test_belt_replay_rejects_an_inconsistent_run() -> None:
    d = _run([_pt("monster", potion_used=["POTION.FIRE_POTION"])])
    belts, _, err = B.belt_timeline(d)
    assert belts is None and "never held" in err
    d = _run([_pt("monster", potion_choices=_pick("FIRE_POTION"))], potions=[])
    assert B.belt_timeline(d)[2] == "final belt mismatch"


def test_pilot_gets_drunk_potions_then_held_fairy_capped_at_slots() -> None:
    # Entry belt: Fire + Fairy. Drank Fire and a Block made mid-fight.
    given, info = B.pilot_potions(
        ["BLOCK_POTION", "FIRE_POTION"], ["FIRE_POTION", "FAIRY_IN_A_BOTTLE"], [], 2
    )
    assert given == ["FIRE_POTION", "FAIRY_IN_A_BOTTLE"]  # belt potions, Fairy, then generated
    assert info["fairy"] and not info["overflow"]
    given, info = B.pilot_potions(["BLOCK_POTION", "FIRE_POTION"], ["FIRE_POTION"], [], 3)
    assert given == ["FIRE_POTION", "BLOCK_POTION"]
    _, info = B.pilot_potions(["FIRE_POTION"] * 3, ["FIRE_POTION"] * 2, [], 2)
    assert info["overflow"]
    # A Fairy that triggered was drunk: given once, not twice.
    given, _ = B.pilot_potions(["FAIRY_IN_A_BOTTLE"], ["FAIRY_IN_A_BOTTLE"], [], 2)
    assert given == ["FAIRY_IN_A_BOTTLE"]


def test_fairy_held_but_never_drunk_reaches_the_record() -> None:
    d = _run(
        [_pt(None, potion_choices=_pick("FAIRY_IN_A_BOTTLE")), _pt("monster")],
        potions=["FAIRY_IN_A_BOTTLE"],
    )
    rec = _records(d)[-1]
    assert rec["potions"] == ["FAIRY_IN_A_BOTTLE"] and rec["belt"] == ["FAIRY_IN_A_BOTTLE"]
    assert rec["belt_unused"] is False


# ── Petrified Toad ───────────────────────────────────────────────────────


def test_toad_rock_is_never_given_twice() -> None:
    toad = ["BURNING_BLOOD", "PETRIFIED_TOAD"]
    # The toad procured a rock at combat start and the human drank it.
    given, info = B.pilot_potions(["POTION_SHAPED_ROCK"], [], toad, 2)
    assert given == [] and info["toad_dedupe"]
    assert B.pilot_belt(given, toad, 2) == ["POTION_SHAPED_ROCK"]  # the engine's toad adds it
    # Two rocks drunk (one carried over, one fresh): the pilot holds two.
    given, _ = B.pilot_potions(["POTION_SHAPED_ROCK"] * 2, ["POTION_SHAPED_ROCK"], toad, 2)
    assert B.pilot_belt(given, toad, 2) == ["POTION_SHAPED_ROCK"] * 2
    # The rock takes a slot: on a 2-slot belt only one other potion fits.
    given, _ = B.pilot_potions(
        ["FIRE_POTION", "BLOCK_POTION", "POTION_SHAPED_ROCK"], ["FIRE_POTION"], toad, 2
    )
    assert B.pilot_belt(given, toad, 2) == ["FIRE_POTION", "POTION_SHAPED_ROCK"]


def test_generated_potions_are_not_given_twice_with_a_generator() -> None:
    # Entropic Brew on the belt, drunk, and a Fire Potion it made drunk too:
    # the engine's Brew rolls its own, so only the Brew is given (R5).
    given, info = B.pilot_potions(
        ["ENTROPIC_BREW", "FIRE_POTION"], ["BLOCK_POTION", "ENTROPIC_BREW"], [], 2
    )
    assert given == ["ENTROPIC_BREW"]
    assert info["generator"] and info["generated_dropped"] == 1
    # Alchemize in the deck, Delicate Frond held: same.
    assert B.pilot_potions(["FIRE_POTION"], [], [], 2, ["ALCHEMIZE+1"])[0] == []
    assert B.pilot_potions(["FIRE_POTION"], [], ["DELICATE_FROND"], 2)[0] == []
    # Without a generator a mid-fight potion is still given.
    given, info = B.pilot_potions(["FIRE_POTION"], [], [], 2, ["STRIKE_IRONCLAD"])
    assert given == ["FIRE_POTION"] and not info["generator"]
    # The toad's rock stays the toad's whatever generator is around.
    given, info = B.pilot_potions(
        ["POTION_SHAPED_ROCK"], [], ["PETRIFIED_TOAD", "DELICATE_FROND"], 2
    )
    assert given == [] and info["toad_dedupe"]


def test_anytime_potions_picked_on_the_floor_are_not_given() -> None:
    # Fruit Juice not on the entry belt: a reward drunk after the fight (R6).
    given, info = B.pilot_potions(["FRUIT_JUICE", "FIRE_POTION"], [], [], 2)
    assert given == ["FIRE_POTION"] and info["anytime_dropped"] == 1
    # From the entry belt it may have been drunk in the fight: given.
    assert B.pilot_potions(["BLOOD_POTION"], ["BLOOD_POTION"], [], 2)[0] == ["BLOOD_POTION"]


def test_toad_extra_rock_when_the_human_belt_was_full() -> None:
    toad = ["PETRIFIED_TOAD"]
    _, info = B.pilot_potions(["FIRE_POTION"], ["FIRE_POTION", "BLOCK_POTION"], toad, 2)
    assert info["toad_extra_rock"]  # the human's toad had no slot; the engine's has one
    _, info = B.pilot_potions(["FIRE_POTION"], ["FIRE_POTION"], toad, 2)
    assert not info["toad_extra_rock"]


# ── relics ───────────────────────────────────────────────────────────────


def test_removed_relics_are_held_until_their_removal() -> None:
    pts = [
        _pt("monster", relic_choices=[{"choice": "RELIC.VAJRA", "was_picked": True}]),  # 2
        _pt("monster"),  # 3
        _pt(
            None,
            relics_removed=["RELIC.VAJRA", "RELIC.BURNING_BLOOD"],
            relic_choices=[{"choice": "RELIC.BLACK_BLOOD", "was_picked": True}],
        ),  # 4
        _pt("monster"),  # 5
    ]
    relics = [{"id": "RELIC.BLACK_BLOOD", "floor_added_to_deck": 4}]
    d = _run(pts, relics=relics)
    by_floor = {r["floor"]: r["relics"] for r in _records(d)}
    assert by_floor == {
        2: ["BURNING_BLOOD"],  # the starter, never picked: held from the start
        3: ["BURNING_BLOOD", "VAJRA"],
        5: ["BLACK_BLOOD"],
    }
    # The original extraction keeps its old rule (final relics only).
    old = phd.extract_fights(d, "abc.run.gz", 10)
    assert {f["floor"]: f["relics"] for f in old} == {2: [], 3: [], 5: ["BLACK_BLOOD"]}


def test_wax_relics_melt_every_third_combat_after_toy_box() -> None:
    wax = {"bools": [{"name": "IsMelted", "value": True}, {"name": "IsWax", "value": True}]}
    relics = [
        {"id": "RELIC.BURNING_BLOOD", "floor_added_to_deck": 1},
        {"id": "RELIC.TOY_BOX", "floor_added_to_deck": 2},
        {"id": "RELIC.VAJRA", "floor_added_to_deck": 2, "props": wax},
    ]
    # Floor 2 is a shop (Toy Box obtained); combats at floors 3, 4, 5, 6.
    d = _run([_pt(None)] + [_pt("monster") for _ in range(4)], relics=relics)
    held = {r["floor"]: "VAJRA" in r["relics"] for r in _records(d)}
    assert held == {3: True, 4: True, 5: True, 6: False}  # melts at the end of floor 5


def test_downgraded_cards_were_upgraded_before() -> None:
    deck = [*STARTER, {"id": "CARD.JUGGLING", "floor_added_to_deck": 2}]
    pts = [
        _pt(None),  # 2: card added
        _pt(None, upgraded_cards=["CARD.JUGGLING"]),  # 3
        _pt("monster"),  # 4
        _pt(None, downgraded_cards=["CARD.JUGGLING"]),  # 5
        _pt("monster"),  # 6
    ]
    by_floor = {r["floor"]: r["deck"] for r in _records(_run(pts, deck=deck))}
    assert "JUGGLING+1" in by_floor[4] and "JUGGLING" in by_floor[6]


def _decks(d: dict[str, Any], history: bool = True) -> dict[int, list[str]]:
    """Fight floor -> deck, by the backward replay or the forward count rule."""
    fights = phd.extract_fights(d, "abc.run.gz", 10, count_downgrades=True, deck_history=history)
    return {f["floor"]: sorted(f["deck"]) for f in fights}


def _juggling(fl: int, *, up: int = 0, ench: str | None = None) -> dict[str, Any]:
    c: dict[str, Any] = {"id": "CARD.JUGGLING", "floor_added_to_deck": fl}
    if up:
        c["current_upgrade_level"] = up
    if ench:
        c["enchantment"] = {"id": f"ENCHANTMENT.{ench}", "amount": 1}
    return c


def test_an_upgrade_of_a_later_copy_does_not_unupgrade_the_held_one() -> None:
    # Copy A (floor 2) upgraded at the floor-3 rest; copy B gained at floor 5,
    # upgraded at floor 7. The forward rule counted floor 7's upgrade against
    # A at floor 4 (R1a).
    deck = [*STARTER, _juggling(2, up=1), _juggling(5, up=1)]
    gain = {"cards_gained": [{"id": "CARD.JUGGLING"}]}
    pts = [
        _pt(None, **gain),  # 2
        _pt(None, upgraded_cards=["CARD.JUGGLING"]),  # 3: rest
        _pt("monster"),  # 4
        _pt("monster", **gain),  # 5
        _pt("monster"),  # 6
        _pt(None, upgraded_cards=["CARD.JUGGLING"]),  # 7: rest
        _pt("monster"),  # 8
    ]
    d = _run(pts, deck=deck)
    new, old = _decks(d), _decks(d, history=False)
    jug = {fl: [c for c in v if c.startswith("JUGGLING")] for fl, v in new.items()}
    assert jug == {
        4: ["JUGGLING+1"],
        5: ["JUGGLING+1"],
        6: ["JUGGLING", "JUGGLING+1"],
        8: ["JUGGLING+1"] * 2,
    }
    assert [c for c in old[4] if c.startswith("JUGGLING")] == ["JUGGLING"]  # the old error
    assert new[6] == old[6] and new[8] == old[8]


def test_removed_and_transformed_copies_keep_their_recorded_upgrade() -> None:
    # Floor 2 upgrades a Defend and a Strike; floor 4 removes that Defend and
    # transforms that Strike (both records say upgraded). The forward rule
    # rendered both plain at floor 3 (R1b).
    deck = [*STARTER[:4], *STARTER[6:], _juggling(4, up=1)]  # 4 Strikes, 3 Defends
    pts = [
        _pt(None, upgraded_cards=["CARD.DEFEND_IRONCLAD", "CARD.STRIKE_IRONCLAD"]),  # 2
        _pt("monster"),  # 3
        _pt(
            None,
            cards_removed=[
                {"id": "CARD.DEFEND_IRONCLAD", "floor_added_to_deck": 1, "current_upgrade_level": 1}
            ],
            cards_transformed=[
                {
                    "original_card": {
                        "id": "CARD.STRIKE_IRONCLAD",
                        "floor_added_to_deck": 1,
                        "current_upgrade_level": 1,
                    },
                    "final_card": _juggling(4, up=1),
                }
            ],
        ),  # 4
        _pt("monster"),  # 5
    ]
    d = _run(pts, deck=deck)
    new, old = _decks(d), _decks(d, history=False)
    assert new[3].count("DEFEND_IRONCLAD+1") == 1 and new[3].count("STRIKE_IRONCLAD+1") == 1
    assert len(new[3]) == 9 and not any("+1" in c for c in old[3])
    assert new[5] == sorted(["STRIKE_IRONCLAD"] * 4 + ["DEFEND_IRONCLAD"] * 3 + ["JUGGLING+1"])
    # Deck ORDER seeds the shuffle, so it follows the original extraction:
    # only the two upgrades differ, in place.
    raw = {
        h: {f["floor"]: f["deck"] for f in phd.extract_fights(d, "a", 10, deck_history=h)}
        for h in (False, True)
    }
    assert [c.replace("+1", "") for c in raw[True][3]] == raw[False][3]
    assert raw[True][3][-2:] == ["DEFEND_IRONCLAD+1", "STRIKE_IRONCLAD+1"]
    assert sorted(phd.DeckReplay(d).start) == ["DEFEND_IRONCLAD"] * 4 + ["STRIKE_IRONCLAD"] * 5


def test_the_upgrade_sits_on_the_enchanted_copy_the_final_deck_shows() -> None:
    # A plain copy held from floor 2, and a copy gained at floor 6 already
    # upgraded and Glam-enchanted. The forward rule gave the +1 to the oldest
    # copy (R2).
    deck = [*STARTER, _juggling(2), _juggling(6, up=1, ench="GLAM")]
    pts = [
        _pt(None, cards_gained=[{"id": "CARD.JUGGLING"}]),  # 2
        *[_pt("monster") for _ in range(3)],  # 3-5
        _pt("monster", cards_gained=[_juggling(6, up=1, ench="GLAM")]),  # 6
        _pt("monster"),  # 7
    ]
    d = _run(pts, deck=deck)
    jug = [c for c in _decks(d)[7] if c.startswith("JUGGLING")]
    assert jug == ["JUGGLING", "JUGGLING+1@GLAM:1"]
    assert [c for c in _decks(d, False)[7] if c.startswith("JUGGLING")] == [
        "JUGGLING+1",
        "JUGGLING@GLAM:1",
    ]


def test_same_floor_buy_and_remove_and_reflections_replay_cleanly() -> None:
    # Floor 2 shop: a card bought and removed again. Floor 5 Reflections:
    # downgrades the upgraded copy, then upgrades it again (TouchAMirror).
    deck = [*STARTER, _juggling(3, up=1)]
    pts = [
        _pt(
            None,
            cards_gained=[{"id": "CARD.JUGGLING"}],
            cards_removed=[_juggling(2)],
        ),  # 2
        _pt("monster", cards_gained=[{"id": "CARD.JUGGLING"}]),  # 3
        _pt(None, upgraded_cards=["CARD.JUGGLING"]),  # 4
        _pt(None, upgraded_cards=["CARD.JUGGLING"], downgraded_cards=["CARD.JUGGLING"]),  # 5
        _pt("monster"),  # 6
    ]
    d = _run(pts, deck=deck)
    rep = phd.DeckReplay(d)
    assert rep.notes == {}
    assert rep.start == ["STRIKE_IRONCLAD"] * 5 + ["DEFEND_IRONCLAD"] * 4
    decks = _decks(d)
    assert not any(c.startswith("JUGGLING") for c in decks[3]) and "JUGGLING+1" in decks[6]
    assert not any("deck_unmatched" in r["flags"] for r in _records(d))


def test_reflections_downgrades_the_copy_that_was_gained_upgraded() -> None:
    # aba9d32b429efa1f.run.gz: Rupture gained plain at floor 14 and +1 at
    # floor 35, Reflections downgrades one at floor 46, both end plain, and no
    # upgrade event exists. Only the floor-35 copy was ever upgraded; v1 put
    # the undone downgrade on the oldest copy, a free Rupture+1 from floor 15.
    deck = [*STARTER, _juggling(2), _juggling(5)]
    pts = [
        _pt(None, cards_gained=[_juggling(2)]),  # 2
        _pt("monster"),  # 3
        _pt("monster"),  # 4
        _pt(None, cards_gained=[_juggling(5, up=1)]),  # 5
        _pt("monster"),  # 6
        _pt(None, downgraded_cards=["CARD.JUGGLING"]),  # 7: Reflections
        _pt("monster"),  # 8
    ]
    d = _run(pts, deck=deck)

    def jug(rep: Any) -> dict[int, list[str]]:
        return {f: sorted(c for c in rep.deck_at(f)[0] if "JUGGLING" in c) for f in (3, 4, 6, 8)}

    new, old = phd.DeckReplay(d), phd.DeckReplay(d, frozen_v1=True)
    assert jug(new) == {
        3: ["JUGGLING"],
        4: ["JUGGLING"],
        6: ["JUGGLING", "JUGGLING+1"],
        8: ["JUGGLING", "JUGGLING"],
    }
    assert (
        new.notes == {} and sorted(new.start) == ["DEFEND_IRONCLAD"] * 4 + ["STRIKE_IRONCLAD"] * 5
    )
    assert jug(old)[3] == ["JUGGLING+1"] and old.notes == {}  # the v1 error, unnoted
    recs = {r["floor"]: r for r in _records(d)}
    assert "JUGGLING+1" not in recs[4]["deck"] and not recs[4]["flags"]


def test_a_gain_undo_that_lost_its_records_upgrade_is_noted() -> None:
    # The gain record says +1 but the final copy is plain with no downgrade
    # logged: the replay cannot place that upgrade. A copy with more than its
    # record (+1 from a plain gain, no upgrade event) is common and fine.
    # The note at floor 3 leaves the fights up to floor 3 uncertain.
    pts = [_pt("monster"), _pt(None, cards_gained=[_juggling(3, up=1)]), _pt("monster")]
    lost = _run(pts, deck=[*STARTER, _juggling(3)])
    assert phd.DeckReplay(lost).notes == {3: ["gain JUGGLING: gained +1, the copy is plain"]}
    assert phd.DeckReplay(lost, frozen_v1=True).notes == {}
    flags = {r["floor"]: r["flags"] for r in _records(lost)}
    assert flags == {2: ["deck_unmatched"], 4: []}
    more = _run(
        [_pt(None, cards_gained=[_juggling(2)]), _pt("monster")],
        deck=[*STARTER, _juggling(2, up=1)],
    )
    assert phd.DeckReplay(more).notes == {}


def test_goopy_amount_is_interpolated_over_the_combats() -> None:
    # A Defend enchanted Goopy (amount 1) at floor 2 ends the run at 4 after
    # five combats (floors 3-7). Plays = 3 over 5 combats, at most 1 each:
    # floor f's amount = 1 + round(3 * k / 5), k = combats before f, clamped
    # to [4 - combats left, 1 + k].
    goopy = {
        "id": "CARD.DEFEND_IRONCLAD",
        "floor_added_to_deck": 1,
        "enchantment": {"id": "ENCHANTMENT.GOOPY", "amount": 4},
    }
    deck = [*STARTER[:8], goopy]
    event = {
        "card": {**goopy, "enchantment": {"id": "ENCHANTMENT.GOOPY", "amount": 1}},
        "enchantment": "ENCHANTMENT.GOOPY",
    }
    pts = [_pt(None, cards_enchanted=[event]), *[_pt("monster") for _ in range(5)]]
    recs = _records(_run(pts, deck=deck))
    amounts = {r["floor"]: next(c for c in r["deck"] if "@GOOPY" in c) for r in recs}
    assert amounts == {
        3: "DEFEND_IRONCLAD@GOOPY:1",
        4: "DEFEND_IRONCLAD@GOOPY:2",
        5: "DEFEND_IRONCLAD@GOOPY:2",
        6: "DEFEND_IRONCLAD@GOOPY:3",
        7: "DEFEND_IRONCLAD@GOOPY:3",
    }
    # Floor 3's amount is pinned (no combat since the enchant); the rest are estimates.
    assert {r["floor"] for r in recs if "goopy_amount" in r["flags"]} == {4, 5, 6, 7}
    # The forward rule kept the enchant-time amount for good (R3).
    assert all("DEFEND_IRONCLAD@GOOPY:1" in v for v in _decks(_run(pts, deck=deck), False).values())


def test_a_relic_handed_back_on_its_removal_floor_is_held_from_its_first_pick() -> None:
    # Vajra picked at floor 2; the floor-4 Relic Trader takes it and hands
    # it back (picked again, removed once); the final list dates it to 4 (R4).
    pts = [
        _pt("monster", relic_choices=[{"choice": "RELIC.VAJRA", "was_picked": True}]),  # 2
        _pt("monster"),  # 3
        _pt(
            None,
            relic_choices=[{"choice": "RELIC.VAJRA", "was_picked": True}],
            relics_removed=["RELIC.VAJRA"],
        ),  # 4
        _pt("monster"),  # 5
    ]
    relics = [
        {"id": "RELIC.BURNING_BLOOD", "floor_added_to_deck": 1},
        {"id": "RELIC.VAJRA", "floor_added_to_deck": 4},
    ]
    by_floor = {r["floor"]: r["relics"] for r in _records(_run(pts, relics=relics))}
    assert by_floor == {
        2: ["BURNING_BLOOD"],
        3: ["BURNING_BLOOD", "VAJRA"],
        5: ["BURNING_BLOOD", "VAJRA"],
    }


def test_deck_change_classes() -> None:
    assert B.deck_change(["A", "B+1"], ["B+1", "A"]) == "same"
    assert B.deck_change(["A", "A"], ["A", "A+1"]) == "more_upgrades"
    assert B.deck_change(["A+1"], ["A"]) == "fewer_upgrades"
    assert B.deck_change(["A+1", "A@GLAM:1"], ["A", "A+1@GLAM:1"]) == "upgrade_moved"
    assert B.deck_change(["A@GOOPY:1"], ["A@GOOPY:5"]) == "other"


# ── rosters ──────────────────────────────────────────────────────────────


def test_roster_drops_pets_and_summons_but_not_self_summoners() -> None:
    assert B.human_roster(["MONSTER.PHROG_PARASITE", "MONSTER.BYRDPIP", "MONSTER.WRIGGLER"]) == [
        "PHROG_PARASITE"
    ]
    assert B.human_roster(["MONSTER.FABRICATOR", "MONSTER.ZAPBOT", "MONSTER.STABBOT"]) == [
        "FABRICATOR"
    ]
    rats = ["MONSTER.TWO_TAILED_RAT"] * 3 + ["MONSTER.OSTY"]
    assert B.human_roster(rats) == ["TWO_TAILED_RAT"] * 3
    # A summon id without its summoner on the roster is an initial monster.
    assert B.human_roster(["MONSTER.WRIGGLER"] * 4) == ["WRIGGLER"] * 4


def test_override_only_where_it_changes_the_roster_and_is_lossless() -> None:
    table = {
        "SLIMES_WEAK": {
            "default": ["LEAF_SLIME_S", "TWIG_SLIME_S", "LEAF_SLIME_M"],
            "randomized": False,
            "lossless": True,
        },
        "THE_KIN_BOSS": {
            "default": ["KIN_FOLLOWER", "KIN_FOLLOWER", "KIN_PRIEST"],
            "randomized": False,
            "lossless": False,
        },
    }
    same = ["LEAF_SLIME_S", "TWIG_SLIME_S", "LEAF_SLIME_M"]
    assert B.resolve_roster(same, "SLIMES_WEAK", table) == (None, "default")
    other = ["TWIG_SLIME_S", "TWIG_SLIME_M", "LEAF_SLIME_S"]
    assert B.resolve_roster(other, "SLIMES_WEAK", table) == (other, "override")
    kin = ["KIN_FOLLOWER", "KIN_FOLLOWER", "KIN_PRIEST"]
    assert B.resolve_roster(kin, "THE_KIN_BOSS", table) == (None, "default")
    assert B.resolve_roster(kin[::-1], "THE_KIN_BOSS", table) == (None, "unmatched")
    # A spawn-arg kind the encounter's own roster never holds is not trusted.
    assert B.resolve_roster(["NIBBIT"], "SLIMES_WEAK", table) == (None, "unmatched")
    assert B.resolve_roster(None, "SLIMES_WEAK", table) == (None, "unmatched")
    assert B._snake("DecimillipedeSegmentFront") == "DECIMILLIPEDE_SEGMENT_FRONT"
    assert B._snake("LeafSlimeS") == "LEAF_SLIME_S"


def test_event_combat_is_extracted_and_flagged_only_on_request() -> None:
    d = _run([_pt("monster", enc="ENCOUNTER.PUNCH_OFF_EVENT_ENCOUNTER", event=True)])
    assert phd.extract_fights(d, "abc.run.gz", 10) == []
    rec = _records(d)[0]
    assert rec["encounter"] == "PUNCH_OFF_EVENT_ENCOUNTER" and "event_combat" in rec["flags"]


# ── mods and pseudonyms ──────────────────────────────────────────────────


def test_mod_filter() -> None:
    assert B.mod_rejection(_run([_pt("monster")]), KNOWN) is None
    hextech = [{"id": "MODIFIER.HEXTECH_MAYHEM_MODIFIER"}]
    assert B.mod_rejection(_run([_pt("monster")], modifiers=hextech), KNOWN) == "modifiers"
    rune = _pt("monster", relic_choices=[{"choice": "RELIC.CUTTING_EDGE_RUNE", "was_picked": True}])
    assert B.mod_rejection(_run([rune]), KNOWN) == "unknown_relic"  # even if later removed
    card = [*STARTER, {"id": "CARD.FUSED_CARD", "floor_added_to_deck": 1}]
    assert B.mod_rejection(_run([_pt("monster")], deck=card), KNOWN) == "unknown_card"
    pot = _pt("monster", potion_choices=_pick("THEHEROEXPANSION-VIGOR_POTION"))
    assert B.mod_rejection(_run([pot]), KNOWN) == "unknown_potion"
    _, why, _ = B.bench_fights(_run([_pt("monster")], modifiers=hextech), "a.run.gz", "pro", KNOWN)
    assert why == "modifiers"


def test_known_ids_come_from_the_game_localization() -> None:
    known = B.known_ids()
    assert "MAD_SCIENCE" in known["cards"] and "FAIRY_IN_A_BOTTLE" in known["potions"]
    assert "KEEPERS_GIFT" not in known["relics"]


def test_player_ids_are_pseudonymous_and_usernames_never_reach_records() -> None:
    name = "Some Pro Player"
    assert B.player_id(name, "f.run.gz") == "p_" + hashlib.sha256(name.encode()).hexdigest()[:10]
    assert B.player_id(None, "f.run.gz").startswith("anon_")
    assert B.player_id(None, "f.run.gz") != B.player_id(None, "g.run.gz")
    recs = _records(_run([_pt("monster")], username=name))
    assert recs and name not in json.dumps(recs)
    assert {r["player"] for r in recs} == {B.player_id(name, "")}


def test_kish_effective_players() -> None:
    assert B.kish_n_eff([10, 10, 10]) == pytest.approx(3.0)
    assert B.kish_n_eff([30, 1, 1]) == pytest.approx(32**2 / 902)


# ── versioned builds: v1 frozen, later versions beside it ────────────────

SEL_BUILD = "v0.103.2"
ALICE, BOB, CAROL = ("a" * 16 + ".run.gz", "b" * 16 + ".run.gz", "c" * 16 + ".run.gz")
# sha256 of the v1 sets `_bench_corpus` builds with the code that built the
# frozen data/combat_bench v1 sets (33d4b625a, which also rebuilds those
# byte for byte). A change that moves them changes what a v1 rebuild writes.
V1_GOLDEN = {
    "ironclad_devpool_v1.jsonl": "077665672378ad24eab3aa093e35d7d44b4c8b01f80998ef00ee2e75bff961a7",
    "ironclad_pro_v1.jsonl": "446f3e293b25f5ba938efec89ab12afa67a673b8388d17b2f00972a310febd79",
}


def _bench_corpus(tmp_path: Path) -> list[str]:
    """A tiny corpus, recipe dir and hbank for `build.main`; returns its argv.

    alice (3/3) outranks bob (1/3) on the selection build, so `--top 1` is
    alice alone. Each has one v0.107.1 A10 run; bob's and the anonymous
    carol's are dev-pool keys. `--bench-dir` is empty: no earlier version.
    """
    month = tmp_path / "root" / "ironclad" / "2026-06"
    month.mkdir(parents=True)

    def put(name: str, username: str | None, build: str, win: bool) -> None:
        fights = [_pt("monster", hp=70, dmg=10), _pt("monster", hp=62, dmg=8)]
        with gzip.open(month / name, "wt") as f:
            json.dump(_run(fights, username=username, win=win) | {"build_id": build}, f)

    for i in range(3):
        put(f"a{i:015d}.run.gz", "alice", SEL_BUILD, True)
        put(f"b{i:015d}.run.gz", "bob", SEL_BUILD, i == 0)
    put(ALICE, "alice", "v0.107.1", True)
    put(BOB, "bob", "v0.107.1", False)
    put(CAROL, None, "v0.107.1", False)
    recipe = tmp_path / "recipe"
    recipe.mkdir()
    (recipe / "pros25.json").write_text(json.dumps(["alice"]))
    (recipe / "boss800.jsonl").write_text(json.dumps({"file": CAROL, "floor": 2}) + "\n")
    (recipe / "hall800.jsonl").write_text(json.dumps({"file": BOB, "floor": 3}) + "\n")
    for name in ("elite800.jsonl", "wg_fights.jsonl"):
        (recipe / name).write_text("")
    (tmp_path / "hbank.jsonl").write_text("")
    (tmp_path / "bench").mkdir()
    return [
        *("--root", str(tmp_path / "root"), "--hbank", str(tmp_path / "hbank.jsonl")),
        *("--recipe-dir", str(recipe), "--procs", "1", "--top", "1", "--min-runs", "2"),
        *("--private-dir", str(tmp_path / "priv"), "--out-dir", str(tmp_path / "out")),
        *("--bench-dir", str(tmp_path / "bench")),
    ]


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_a_default_build_writes_the_frozen_v1_outputs(tmp_path: Path) -> None:
    pytest.importorskip("sts2_rs")
    argv = _bench_corpus(tmp_path)
    assert B.main(argv) == 0
    out = tmp_path / "out"
    assert sorted(p.name for p in out.iterdir()) == ["MANIFEST.json", *V1_GOLDEN]
    pro = _jsonl(out / "ironclad_pro_v1.jsonl")
    assert {(r["file"], r["player"]) for r in pro} == {(ALICE, B.player_id("alice", ""))}
    assert {r["file"] for r in _jsonl(out / "ironclad_devpool_v1.jsonl")} == {BOB, CAROL}
    assert {n: B._sha256(str(out / n)) for n in V1_GOLDEN} == V1_GOLDEN
    assert (tmp_path / "priv" / "players_all.jsonl").exists()
    assert (tmp_path / "priv" / "pro_cohort.json").exists()
    assert json.loads((out / "MANIFEST.json").read_text())["rules"] == "v1 (frozen)"
    # A second build refuses to overwrite them; --force does.
    (out / "MANIFEST.json").write_text("{}")
    assert B.main(argv) == 2
    assert (out / "MANIFEST.json").read_text() == "{}"
    assert B.main([*argv, "--force"]) == 0
    assert {n: B._sha256(str(out / n)) for n in V1_GOLDEN} == V1_GOLDEN


def test_an_explicit_v1_is_the_default() -> None:
    assert B.set_file("pro", "v1") == B.PRO_SET == "ironclad_pro_v1.jsonl"
    assert B.set_file("devpool", "v1") == B.DEV_SET == "ironclad_devpool_v1.jsonl"
    assert B.manifest_file("v1") == B.MANIFEST == "MANIFEST.json"
    assert B.index_file("v1") == "players_all.jsonl"
    assert B.cohort_file("v1") == "pro_cohort.json"
    assert B.manifest_file("v2") == "MANIFEST_v2.json"
    assert B.set_file("pro50", "v2") == "ironclad_pro50_v2.jsonl"
    assert B.index_file("v2") == "players_all_v2.jsonl"
    assert B.cohort_file("v2") == "pro_cohort_v2.json"


@pytest.mark.parametrize("bad", ["2", "v0", "V2", "v2/../x", "v2.1"])
def test_a_version_must_be_v_and_a_number(bad: str, tmp_path: Path) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        B._version(bad)
    argv = ["--version", bad, "--private-dir", str(tmp_path), "--hbank", str(tmp_path / "h")]
    with pytest.raises(SystemExit) as exc:
        B.main(argv)
    assert exc.value.code == 2


def test_a_version_is_v_and_a_number() -> None:
    assert B._version("v1") == "v1" and B._version("v2") == "v2" and B._version("v10") == "v10"


def test_extra_tops_must_extend_the_cohort(tmp_path: Path) -> None:
    argv = ["--private-dir", str(tmp_path), "--hbank", "/nonexistent", "--top", "25"]
    assert B.main([*argv, "--extra-top", "25"]) == 2
    assert B.main([*argv, "--extra-top", "50", "50"]) == 2


def test_v2_writes_its_own_files_beside_v1_and_the_wider_cohort(tmp_path: Path) -> None:
    pytest.importorskip("sts2_rs")
    assert B.main([*_bench_corpus(tmp_path), "--version", "v2", "--extra-top", "2"]) == 0
    out = tmp_path / "out"
    sets = ["ironclad_devpool_v2.jsonl", "ironclad_pro2_v2.jsonl", "ironclad_pro_v2.jsonl"]
    assert sorted(p.name for p in out.iterdir()) == ["MANIFEST_v2.json", *sets]
    # Nothing in this corpus meets a rule v2 changed: v2's sets are v1's bytes.
    assert B._sha256(str(out / "ironclad_pro_v2.jsonl")) == V1_GOLDEN["ironclad_pro_v1.jsonl"]
    dev = B._sha256(str(out / "ironclad_devpool_v2.jsonl"))
    assert dev == V1_GOLDEN["ironclad_devpool_v1.jsonl"]
    # The top-2 set adds bob; the dev pool still drops only the --top cohort.
    pro2 = _jsonl(out / "ironclad_pro2_v2.jsonl")
    assert {(r["file"], r["player"], r["cohort"]) for r in pro2} == {
        (ALICE, B.player_id("alice", ""), "pro"),
        (BOB, B.player_id("bob", ""), "pro"),
    }
    # hall800 holds bob's floor 3: the recipe evaluated on it.
    assert {r["key"] for r in pro2 if "in_rl2_eval" in r["flags"]} == {f"{BOB}:3"}
    assert not any("in_rl2_eval" in r["flags"] for r in _jsonl(out / "ironclad_devpool_v2.jsonl"))
    m = json.loads((out / "MANIFEST_v2.json").read_text())
    assert m["version"] == "v2" and sorted(m["sets"]) == sets
    assert m["rules"] == list(B.V2_RULES)
    assert m["sets"]["ironclad_pro2_v2.jsonl"]["players"] == 2
    assert m["sets"]["ironclad_pro2_v2.jsonl"]["flags"]["in_rl2_eval"] == 1
    assert m["held_out_runs"] == [ALICE, BOB, CAROL]
    extra = m["selection"]["extra_tops"]["2"]
    assert extra["set"] == "ironclad_pro2_v2.jsonl" and extra["eval_runs_wins"] == [1, 2]
    # bob's run is a dev-pool run too, and bob a dev-pool player.
    assert extra["devpool_overlap"] == {
        "ironclad_devpool_v2.jsonl": {
            "players": 1,
            "runs": 1,
            "fights": 1,
            "fights_of_shared_players": 2,
            "fights_in_shared_runs": 2,
            "kish_n_eff_players_without_shared": 1.0,
        }
    }
    assert m["selection"]["devpool_overlap"]["ironclad_devpool_v2.jsonl"]["players"] == 0
    assert m["screened"]["panels"] == {} and m["screened"]["dropped"] == {}
    assert m["filter_stages"]["pro2"]["kept_runs"] == 2
    cohort = json.loads((tmp_path / "priv" / "pro_cohort_v2.json").read_text())
    assert [c["username"] for c in cohort] == ["alice", "bob"]
    assert not (tmp_path / "priv" / "pro_cohort.json").exists()
    assert (tmp_path / "priv" / "players_all_v2.jsonl").exists()
    assert not (tmp_path / "priv" / "players_all.jsonl").exists()


ALICE2 = "d" * 16 + ".run.gz"


def _spelling_corpus(tmp_path: Path) -> list[str]:
    """One player under two spellings, a modded selection run and a v1 dev panel.

    "Alice" (3/3 on the selection build, first seen) uploads her v0.107.1 run
    as "alice ". bob's 1/2 selection runs have a modded win beside them.
    The recipe holds both of alice's fights (boss800 floor 2, hall800 floor
    3); v1 put them in its dev pool, and its dev panel holds floor 3.
    """
    month = tmp_path / "root" / "ironclad" / "2026-06"
    month.mkdir(parents=True)

    def put(name: str, username: str | None, build: str, win: bool, t: int, **kw: Any) -> None:
        fights = [_pt("monster", hp=70, dmg=10), _pt("monster", hp=62, dmg=8)]
        run = _run(fights, username=username, win=win, **kw)
        with gzip.open(month / name, "wt") as f:
            json.dump(run | {"build_id": build, "start_time": t}, f)

    for i in range(3):
        put(f"a{i:015d}.run.gz", "Alice", SEL_BUILD, True, 100 + i)
    put("b" + "0" * 15 + ".run.gz", "bob", SEL_BUILD, True, 200)
    put("b" + "1" * 15 + ".run.gz", "bob", SEL_BUILD, False, 201)
    hextech = [{"id": "MODIFIER.HEXTECH_MAYHEM_MODIFIER"}]
    put("b" + "2" * 15 + ".run.gz", "bob", SEL_BUILD, True, 202, modifiers=hextech)
    put(ALICE2, "alice ", "v0.107.1", True, 500)
    put(CAROL, None, "v0.107.1", False, 600)
    recipe = tmp_path / "recipe"
    recipe.mkdir()
    (recipe / "pros25.json").write_text(json.dumps(["ALICE"]))
    (recipe / "boss800.jsonl").write_text(json.dumps({"file": ALICE2, "floor": 2}) + "\n")
    hall = [{"file": ALICE2, "floor": 3}, {"file": CAROL, "floor": 2}]
    (recipe / "hall800.jsonl").write_text("".join(json.dumps(x) + "\n" for x in hall))
    for name in ("elite800.jsonl", "wg_fights.jsonl"):
        (recipe / name).write_text("")
    (tmp_path / "hbank.jsonl").write_text("")
    bench = tmp_path / "bench"
    bench.mkdir()

    def row(file: str, floor: int, username: str | None) -> str:
        key = f"{file}:{floor}"
        return json.dumps({"key": key, "file": file, "player": B.player_id(username, file)})

    (bench / "ironclad_devpool_v1.jsonl").write_text(
        "".join(
            row(*x) + "\n" for x in ((ALICE2, 2, "alice "), (ALICE2, 3, "alice "), (CAROL, 2, None))
        )
    )
    (bench / "ironclad_dev_v1.jsonl").write_text(row(ALICE2, 3, "alice ") + "\n")
    return [
        *("--root", str(tmp_path / "root"), "--hbank", str(tmp_path / "hbank.jsonl")),
        *("--recipe-dir", str(recipe), "--procs", "1", "--top", "1", "--min-runs", "2"),
        *("--private-dir", str(tmp_path / "priv"), "--bench-dir", str(bench)),
    ]


def test_v2_joins_spellings_drops_screened_fights_and_ranks_clean_runs(tmp_path: Path) -> None:
    pytest.importorskip("sts2_rs")
    argv = _spelling_corpus(tmp_path)
    v1, v2 = tmp_path / "out1", tmp_path / "out2"
    assert B.main([*argv, "--out-dir", str(v1)]) == 0
    assert B.main([*argv, "--out-dir", str(v2), "--version", "v2"]) == 0
    # v1: "Alice" is the pro and has no v0.107.1 run; "alice "'s is dev pool.
    assert _jsonl(v1 / "ironclad_pro_v1.jsonl") == []
    assert {r["key"] for r in _jsonl(v1 / "ironclad_devpool_v1.jsonl")} == {
        f"{ALICE2}:2",
        f"{ALICE2}:3",
        f"{CAROL}:2",
    }
    m1 = json.loads((v1 / "MANIFEST.json").read_text())
    assert m1["selection"]["ranking_runs_excluded"] == {}
    # v2: one player, named by the first-seen spelling. The floor-3 fight is
    # on v1's dev panel (screened on), so the pro set leaves it out.
    pro = _jsonl(v2 / "ironclad_pro_v2.jsonl")
    assert [(r["key"], r["player"]) for r in pro] == [(f"{ALICE2}:2", B.player_id("Alice", ""))]
    assert "in_rl2_eval" in pro[0]["flags"]
    assert {r["key"] for r in _jsonl(v2 / "ironclad_devpool_v2.jsonl")} == {f"{CAROL}:2"}
    m = json.loads((v2 / "MANIFEST_v2.json").read_text())
    sel = m["selection"]
    assert sel["players_with_several_spellings"] == 1
    assert sel["cohort_players_with_several_spellings"] == 1
    assert sel["ranking_runs_excluded"] == {"modifiers": 1}
    assert sel["overlap_with_pros25"] == 1 and sel["eval_runs_wins"] == [1, 1]
    assert m["devpool"]["pro_player_runs_excluded"] == 1
    assert m["screened"]["panels"] == {"ironclad_dev_v1.jsonl": 1}
    assert m["screened"]["dropped"] == {"ironclad_pro_v2.jsonl": 1}
    # The v1 sets gave "alice " her own id; the overlap still finds the player.
    assert sel["devpool_overlap"]["ironclad_dev_v1.jsonl"] == {
        "players": 1,
        "runs": 1,
        "fights": 0,
        "fights_of_shared_players": 1,
        "fights_in_shared_runs": 1,
        "kish_n_eff_players_without_shared": 0.0,
    }
    assert sel["devpool_overlap"]["ironclad_devpool_v1.jsonl"]["fights"] == 1
    assert sel["devpool_overlap"]["ironclad_devpool_v2.jsonl"]["players"] == 0
    cohort = json.loads((tmp_path / "priv" / "pro_cohort_v2.json").read_text())
    assert cohort[0]["username"] == "Alice" and cohort[0]["other_spellings"] == ["alice "]
    assert (cohort[0]["sel_wins"], cohort[0]["sel_runs"]) == (3, 3)
    v1_cohort = json.loads((tmp_path / "priv" / "pro_cohort.json").read_text())
    assert [c["username"] for c in v1_cohort] == ["Alice"]


def test_selection_runs_are_ranked_only_when_clean() -> None:
    run = _run([_pt("monster")])
    assert B.selection_rejection(run | {"build_id": "v0.103.2"}, KNOWN) is None
    hextech = [{"id": "MODIFIER.HEXTECH_MAYHEM_MODIFIER"}]
    assert B.selection_rejection(_run([_pt("monster")], modifiers=hextech), KNOWN) == "modifiers"
    baselib = [*STARTER, {"id": "CARD.STRIKE_IRONCLAD", "save_dict_x": {}}]
    assert B.selection_rejection(_run([_pt("monster")], deck=baselib), KNOWN) == "modded_card"
    # Unknown ids and enchantments count only up to the eval build: the
    # tables do not know what v0.108.0 on added.
    rune = _pt("monster", relic_choices=[{"choice": "RELIC.CUTTING_EDGE_RUNE", "was_picked": True}])
    old, new = ({"build_id": b} for b in ("v0.103.2", "v0.109.0"))
    assert B.selection_rejection(_run([rune]) | old, KNOWN) == "unknown_relic"
    assert B.selection_rejection(_run([rune]) | new, KNOWN) is None
    ench = [*STARTER, _juggling(1, ench="CUSTOM_SPIRAL")]
    assert B.selection_rejection(_run([_pt("monster")], deck=ench) | old, KNOWN) == (
        "modded_enchantment"
    )
    assert B.selection_rejection(_run([_pt("monster")], deck=ench) | new, KNOWN) is None
    assert (
        B.selection_rejection(_run([_pt("monster")]) | {"build_id": "NON-RELEASE"}, KNOWN) is None
    )


# ── rollout_learn holdout ────────────────────────────────────────────────


def _manifest(tmp_path: Path, runs: list[str]) -> Path:
    p = tmp_path / "MANIFEST.json"
    p.write_text(json.dumps({"held_out_runs": runs}))
    return p


def test_subtract_bench_runs(tmp_path: Path) -> None:
    m = str(_manifest(tmp_path, ["b.run.gz"]))
    pool = [{"file": "a.run.gz"}, {"file": "b.run.gz"}, {"file": "b.run.gz"}]
    kept, meta = H.subtract_bench_runs(pool, lambda f: f["file"], manifest_path=m)
    assert kept == [{"file": "a.run.gz"}]
    assert meta | {"manifest": None} == {
        "manifest": None,
        "bench_runs": 1,
        "overlap_fights": 2,
        "overlap_runs": 1,
        "subtracted": True,
    }
    kept, meta = H.subtract_bench_runs(pool, lambda f: f["file"], True, manifest_path=m)
    assert kept == pool and meta["subtracted"] is False and meta["overlap_fights"] == 2
    kept, meta = H.subtract_bench_runs(pool, lambda f: f["file"], manifest_path=str(tmp_path / "x"))
    assert kept == pool and meta == {"manifest": None, "subtracted": False}


def test_the_default_holdout_covers_every_bench_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "MANIFEST.json").write_text(json.dumps({"held_out_runs": ["v1.run.gz"]}))
    (tmp_path / "MANIFEST_v2.json").write_text(
        json.dumps({"held_out_runs": ["v2.run.gz", "v1.run.gz"]})
    )
    (tmp_path / "ironclad_pro_v2.jsonl").write_text("")
    monkeypatch.setattr(H, "MANIFEST_PATH", str(tmp_path / "MANIFEST.json"))
    assert H.manifest_paths() == [
        str(tmp_path / "MANIFEST.json"),
        str(tmp_path / "MANIFEST_v2.json"),
    ]
    assert H.holdout_runs() == {"v1.run.gz", "v2.run.gz"}
    assert H.holdout_runs(str(tmp_path / "MANIFEST.json")) == {"v1.run.gz"}
    pool = [{"file": f} for f in ("v1.run.gz", "v2.run.gz", "x.run.gz")]
    kept, meta = H.subtract_bench_runs(pool, lambda f: f["file"])
    assert kept == [{"file": "x.run.gz"}]
    assert meta["manifest"] == ["MANIFEST.json", "MANIFEST_v2.json"] and meta["bench_runs"] == 2
    # Only a v2 MANIFEST (v1's gone): still held out.
    (tmp_path / "MANIFEST.json").unlink()
    assert H.holdout_runs() == {"v1.run.gz", "v2.run.gz"}
    (tmp_path / "MANIFEST_v2.json").unlink()
    with pytest.raises(FileNotFoundError):
        H.holdout_runs()


def test_every_committed_bench_set_is_held_out() -> None:
    sets = sorted(Path(H.MANIFEST_PATH).parent.glob("ironclad_*.jsonl"))
    if not sets:
        pytest.skip("no combat bench data (the public snapshot ships without it)")
    held = H.holdout_runs()
    for path in sets:
        files = {r["file"] for r in _jsonl(path)}
        assert files and files <= held, f"{path.name}: {len(files - held)} runs not held out"


class _NoPool:
    """Stands in for the spawn pool: records the jobs and plays nothing."""

    jobs: ClassVar[list[Any]] = []

    def __init__(self, *_: Any) -> None:
        pass

    def __enter__(self) -> _NoPool:
        return self

    def __exit__(self, *_: Any) -> None:
        return None

    def imap_unordered(self, fn: Any, jobs: Any) -> Any:
        _NoPool.jobs = list(jobs)
        # Every job "finishes" empty; `gen` reads results with the real
        # iterator's `next(timeout)`.
        rows = iter(
            [
                {"key": f"{j[0]['file']}:{j[0]['floor']}", "decisions": [], "unmatched": 0}
                for j in _NoPool.jobs
            ]
        )
        return SimpleNamespace(next=lambda _timeout=None: next(rows))


@pytest.mark.parametrize("allow", [False, True])
def test_rollout_learn_gen_drops_bench_runs_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, allow: bool
) -> None:
    monkeypatch.setattr(H, "MANIFEST_PATH", str(_manifest(tmp_path, ["bench.run.gz"])))
    monkeypatch.setattr(rollout_learn, "get_context", lambda _: SimpleNamespace(Pool=_NoPool))
    fights = tmp_path / "fights.jsonl"
    rows = [
        {"file": f, "floor": fl, "room_type": "boss", "encounter": "E"}
        for f, fl in (("bench.run.gz", 17), ("train.run.gz", 17), ("train.run.gz", 33))
    ]
    fights.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "gen"
    argv = ["gen", "--fights", str(fights), "--out-dir", str(out), "--procs", "1"]
    assert rollout_learn.main(argv + (["--allow-bench-overlap"] if allow else [])) == 0
    played = {job[0]["file"] for job in _NoPool.jobs}
    assert ("bench.run.gz" in played) is allow and "train.run.gz" in played
    meta = json.loads((out / rollout_learn.BENCH_META).read_text())
    assert meta["overlap_fights"] == 1 and meta["subtracted"] is (not allow)
