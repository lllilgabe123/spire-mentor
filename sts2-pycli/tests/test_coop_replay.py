"""``coop_replay``: a recorded co-op decision pair replayed seat by seat."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

import coop_replay
from realgame_adapter import to_headless_combat_play

pytest.importorskip("sts2_rs")

FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "realgame_adapter" / "single_enemy_ironclad.json"
)


def _mod_board() -> dict[str, Any]:
    """The fixture's Ironclad board (one Shrinker Beetle, 40 HP), with net
    ids and one Silent teammate holding a Strike, their whole seat sent."""
    board = json.loads(FIXTURE.read_text(encoding="utf-8"))["state"]
    strike = copy.deepcopy(board["hand"][2])
    strike.update(id="STRIKE_SILENT")
    board["player"]["net_id"] = "1"
    board["allies"] = [
        {
            "character": "SILENT",
            "net_id": "2",
            "hp": 50,
            "max_hp": 70,
            "block": 0,
            "is_dead": False,
            "energy": 3,
            "max_energy": 3,
            "powers": [],
            "hand": [strike],
            "draw_pile": {"type": "Draw", "count": 0, "cards": []},
            "discard_pile": {"type": "Discard", "count": 0, "cards": []},
            "exhaust_pile": {"type": "Exhaust", "count": 0, "cards": []},
            "play_pile": {"type": "Play", "count": 0, "cards": []},
            "relics": [],
            "stars": 0,
            "ended_turn": False,
        }
    ]
    return board


def _teammate_strike_row() -> dict[str, Any]:
    return {
        "kind": "card_play_started",
        "round": 1,
        "side": "player",
        "card": "STRIKE_SILENT",
        "upgraded": False,
        "auto_played": False,
        "owner": "2",
        "target": {"kind": "monster", "id": "SHRINKER_BEETLE", "uid": None},
        "target_index": 0,
        "play_index": 0,
    }


def _pair(enemy_hp_after: int) -> tuple[dict[str, Any], dict[str, Any]]:
    before = _mod_board()
    after = _mod_board()
    after["enemies"][0]["hp"] = enemy_hp_after
    after["allies"][0]["hand"] = []
    after["allies"][0]["energy"] = 2
    a = to_headless_combat_play(before)
    b = to_headless_combat_play(after)
    b["combat_log_raw"] = [_teammate_strike_row()]
    return a, b


def test_a_teammates_recorded_play_replays_in_sync() -> None:
    a, b = _pair(enemy_hp_after=40 - 6)
    res = coop_replay.replay_pair(a, b)
    assert res.get("skipped") is None, res
    assert res["played"] == 1
    assert res["diffs"] == {}


def test_a_divergence_is_reported_by_field() -> None:
    a, b = _pair(enemy_hp_after=40 - 9)
    res = coop_replay.replay_pair(a, b)
    assert res["diffs"] == {"encounter[0].hp": [34, 31]}


def test_an_older_board_is_skipped_not_guessed() -> None:
    a, b = _pair(enemy_hp_after=34)
    for ally in a["allies"]:
        del ally["draw_pile"]
    assert coop_replay.replay_pair(a, b) == {"skipped": "teammate piles missing (older mod)"}


def test_a_cards_combat_vars_reach_the_engine() -> None:
    """A Thrumming Hatchet played this turn (``combat_vars.return_if_played_round``,
    from the mod's History read) survives both adapters into the engine."""
    import sts2_rs
    from sts2_hint.adapter import translate

    board = _mod_board()
    hatchet = copy.deepcopy(board["hand"][2])
    hatchet.update(id="THRUMMING_HATCHET", combat_vars={"return_if_played_round": 1})
    board["discard_pile"] = {"type": "Discard", "count": 1, "cards": [hatchet]}
    wire = translate(to_headless_combat_play(board))
    card = wire["deck"]["discard_pile"][0]
    assert card["combat_vars"] == {"return_if_played_round": 1}
    again = sts2_rs.state_to_dict(sts2_rs.state_from_dict(wire))
    assert again["deck"]["discard_pile"][0]["combat_vars"]["return_if_played_round"] == 1


def test_our_place_in_the_player_order_reaches_the_engine() -> None:
    """A board from the second player's seat loads with us behind the
    teammate (``coop_seat_index``), who then sets up and draws first."""
    import sts2_rs

    a, _ = _pair(enemy_hp_after=34)
    assert "coop_seat_index" not in coop_replay.full_fight_wire(a)
    a["seat_index"] = 1
    wire = coop_replay.full_fight_wire(a)
    assert wire["coop_seat_index"] == 1
    again = sts2_rs.state_to_dict(sts2_rs.state_from_dict(wire))
    assert again["coop_seat_index"] == 1


def _survivor_pair(choice_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """The teammate plays Survivor (8 block, discard 1) from Survivor, Strike,
    Defend and discards the Defend."""
    before = _mod_board()
    strike = before["allies"][0]["hand"][0]
    survivor = copy.deepcopy(strike)
    survivor.update(id="SURVIVOR", type="Skill", target="Self")
    defend = copy.deepcopy(strike)
    defend.update(id="DEFEND_SILENT", type="Skill", target="Self")
    before["allies"][0]["hand"] = [survivor, strike, defend]
    after = copy.deepcopy(before)
    after["allies"][0].update(hand=[strike], block=8, energy=2)
    a = to_headless_combat_play(before)
    b = to_headless_combat_play(after)
    play = _teammate_strike_row()
    play.update(card="SURVIVOR", target=None, target_index=None)
    b["combat_log_raw"] = [play, *choice_rows]
    return a, b


def _choice_row(owner: str, cards: list[str]) -> dict[str, Any]:
    return {
        "kind": "player_choice",
        "round": 1,
        "side": "player",
        "owner": owner,
        "choice_id": 0,
        "choice_type": "CombatCard",
        "cards": [{"id": c, "upgraded": False} for c in cards],
        "indexes": None,
        "player": None,
    }


def test_a_teammates_logged_pick_is_the_one_replayed() -> None:
    a, b = _survivor_pair([_choice_row("2", ["DEFEND_SILENT"])])
    res = coop_replay.replay_pair(a, b)
    assert res.get("skipped") is None, res
    assert res["choices"] == 1
    assert res["diffs"] == {}


def test_without_a_logged_pick_the_engine_picks_and_diverges() -> None:
    a, b = _survivor_pair([])
    res = coop_replay.replay_pair(a, b)
    assert "allies[0].hand" in res["diffs"]


def test_a_pick_matches_cards_by_id_once_each() -> None:
    options = [
        {"id": "StrikeSilent"},
        {"id": "DefendSilent"},
        {"id": "DefendSilent", "upgraded": True},
    ]
    row = _choice_row("2", ["DEFEND_SILENT", "DEFEND_SILENT"])
    assert coop_replay._choice_selection(row, options) == [1, 2]
    row["cards"][0]["upgraded"] = True
    assert coop_replay._choice_selection(row, options) == [1, 2]
    assert coop_replay._choice_selection(_choice_row("2", ["BASH"]), options) is None
    index = {**_choice_row("2", []), "choice_type": "Index", "indexes": [2]}
    assert coop_replay._choice_selection(index, options) == [2]


def test_our_own_logged_pick_is_answered_not_skipped() -> None:
    """We play True Grit+ (9 block, exhaust a card of our choice) and exhaust
    the Bash: the replay answers our screen from the log."""
    before = _mod_board()
    grit = copy.deepcopy(before["hand"][0])
    grit.update(id="TRUE_GRIT", upgraded=True)
    before["hand"][0] = grit
    after = copy.deepcopy(before)
    after["hand"] = before["hand"][1:4]
    after["player"].update(block=9, energy=2)
    a = to_headless_combat_play(before)
    b = to_headless_combat_play(after)
    play = _teammate_strike_row()
    play.update(card="TRUE_GRIT", upgraded=True, owner="1", target=None, target_index=None)
    b["combat_log_raw"] = [play, _choice_row("1", ["BASH"])]
    b["interrupted"] = True
    res = coop_replay.replay_pair(a, b)
    assert res.get("skipped") is None, res
    assert res["choices"] == 1
    assert res["diffs"] == {}


def test_an_index_skip_selects_nothing() -> None:
    skip = {**_choice_row("2", []), "choice_type": "Index", "indexes": [-1]}
    assert coop_replay._choice_selection(skip, [{"id": "A"}, {"id": "B"}]) == []


def _block_potion_pair(belt: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    """The teammate drinks a Block Potion (12 block); their belt is sent
    with the board when ``belt``, as newer mods do."""
    before = _mod_board()
    if belt:
        before["allies"][0]["potions"] = [
            {"slot": 0, "empty": False, "id": "BLOCK_POTION", "name": "Block Potion"},
            {"slot": 1, "empty": True},
        ]
    after = copy.deepcopy(before)
    after["allies"][0]["block"] = 12
    after["allies"][0].pop("potions", None)
    a = to_headless_combat_play(before)
    b = to_headless_combat_play(after)
    b["combat_log_raw"] = [
        {
            "kind": "potion",
            "round": 1,
            "side": "player",
            "potion": "BLOCK_POTION",
            "owner": "2",
            "target_index": None,
        }
    ]
    return a, b


@pytest.mark.parametrize("belt", [True, False])
def test_a_teammates_potion_is_drunk_from_their_belt(belt: bool) -> None:
    a, b = _block_potion_pair(belt)
    res = coop_replay.replay_pair(a, b)
    assert res.get("skipped") is None, res
    assert res["potions"] == 1
    assert res["diffs"] == {}


@pytest.mark.parametrize("discarded", [1, 2])
def test_a_logged_multi_pick_is_taken_whole(discarded: int) -> None:
    """The teammate drinks Gambler's Brew and discards one or both Defends in
    one pick (the engine takes a multi-pick a card at a time, its legal
    actions single picks only), drawing a Strike for each."""
    before = _mod_board()
    mate = before["allies"][0]
    strike = mate["hand"][0]
    defend = copy.deepcopy(strike)
    defend.update(id="DEFEND_SILENT", type="Skill", target="Self")
    mate["hand"] = [strike, defend, copy.deepcopy(defend)]
    mate["draw_pile"] = {
        "type": "Draw",
        "count": 2,
        "cards": [copy.deepcopy(strike), copy.deepcopy(strike)],
    }
    mate["potions"] = [{"slot": 0, "empty": False, "id": "GAMBLERS_BREW", "name": "Gambler's Brew"}]
    after = copy.deepcopy(before)
    after["allies"][0].update(
        hand=[copy.deepcopy(c) for c in [strike] * (1 + discarded) + [defend] * (2 - discarded)],
        draw_pile={"type": "Draw", "count": 2 - discarded, "cards": [strike] * (2 - discarded)},
        discard_pile={"type": "Discard", "count": discarded, "cards": [defend] * discarded},
    )
    a = to_headless_combat_play(before)
    b = to_headless_combat_play(after)
    drink = {
        "kind": "potion",
        "round": 1,
        "side": "player",
        "potion": "GAMBLERS_BREW",
        "owner": "2",
        "target_index": None,
    }
    b["combat_log_raw"] = [_choice_row("2", ["DEFEND_SILENT"] * discarded), drink]
    res = coop_replay.replay_pair(a, b)
    assert res.get("skipped") is None, res
    assert (res["choices"], res["unanswered"]) == (1, 0)
    assert res["diffs"] == {}
