"""`ppo.combat_bench.own_decks`: the pure helpers (no engine, no data files)."""

from __future__ import annotations

from ppo.combat_bench.own_decks import (
    ENGINE_DECK_IDS,
    card_id,
    deck_strings,
    engine_count,
    queued_encounter,
    split_by_player,
)


def test_engine_ids_drop_the_card_suffix() -> None:
    assert "FEEL_NO_PAIN" in ENGINE_DECK_IDS
    assert "DARK_EMBRACE" in ENGINE_DECK_IDS
    assert "CRUELTY" in ENGINE_DECK_IDS
    assert not any(c.endswith("_CARD") for c in ENGINE_DECK_IDS)


def test_card_id_strips_upgrade_and_enchantment() -> None:
    assert card_id("BURNING_PACT+1@SWIFT:2") == "BURNING_PACT"
    assert card_id("STRIKE_IRONCLAD") == "STRIKE_IRONCLAD"


def test_engine_count() -> None:
    deck = ["STRIKE_IRONCLAD", "BLOODLETTING+1", "FEEL_NO_PAIN@SWIFT:1", "BASH"]
    assert engine_count(deck) == 2


def test_deck_strings_encode_upgrade_and_enchantment() -> None:
    cards = {"StrikeIronclad": "STRIKE_IRONCLAD", "Bash": "BASH"}
    out = deck_strings([("StrikeIronclad", False), ("Bash", True)], [None, ("SWIFT", 2)], cards)
    assert out == ["STRIKE_IRONCLAD", "BASH+1@SWIFT:2"]


def test_queued_encounter_by_room_and_index() -> None:
    rooms = {"boss": "B", "normal": ["N0", "N1"], "elite": ["E0"]}
    assert queued_encounter(rooms, "boss", 3) == "B"
    assert queued_encounter(rooms, "monster", 1) == "N1"
    assert queued_encounter(rooms, "monster", 2) == "N0"
    assert queued_encounter(rooms, "elite", 0) == "E0"
    assert queued_encounter({"normal": []}, "monster", 0) is None


def test_split_by_player_keeps_players_on_one_side() -> None:
    rows = [{"player": f"p{i % 7}", "key": str(i)} for i in range(70)]
    tune, hold = split_by_player(rows, seed=3)
    assert len(tune) + len(hold) == 70
    assert not {r["player"] for r in tune} & {r["player"] for r in hold}
    assert 20 <= len(tune) <= 50
    assert split_by_player(rows, seed=3) == (tune, hold)
