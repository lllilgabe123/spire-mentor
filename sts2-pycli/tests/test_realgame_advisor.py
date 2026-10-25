"""Offline tests for :mod:`realgame_advisor`'s map-route projection.

The advisor's map advice answers "which node next"; the projection walks the
same policy forward so the HUD can draw a route through the act. Everything
here is pure dict-in/dict-out -- the policy is a stub, so no torch and no
checkpoint are needed.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
import realgame_advisor as ra
from sts2_hint.recommender import HintTag

_RUN_STATE: dict[str, Any] = {
    "character": "Ironclad",
    "hp": 60,
    "max_hp": 75,
    "gold": 120,
    "ascension": 0,
    "master_deck": [{"id": "STRIKE_IRONCLAD"}],
    "relics": [{"id": "BURNING_BLOOD"}],
    "potions": [],
}


def _state(nodes: list[dict[str, Any]], graph: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "type": "map_select",
        "floor": 5,
        "act": 1,
        "run_state": _RUN_STATE,
        "nodes": nodes,
        "map": {
            "row_count": 10,
            "col_count": 6,
            "current": {"col": 2, "row": 5},
            "boss": {"col": 2, "row": 15},
            "visited": [],
            "nodes": graph,
        },
    }


class _RightmostPolicy:
    """Always prefers the highest column, so the expected route is obvious."""

    def __init__(self) -> None:
        self.calls: list[list[dict[str, Any]]] = []

    def distribution(
        self,
        obs: dict[str, Any],
        actions: list[dict[str, Any]],
        metrics: dict[str, Any],
    ) -> tuple[list[float], float]:
        self.calls.append([{**a, "_map_coord": obs["map_coord"]} for a in actions])
        best = max(range(len(actions)), key=lambda i: int(actions[i]["col"]))
        return [1.0 if i == best else 0.0 for i in range(len(actions))], 0.5


class _RecordingPolicy(_RightmostPolicy):
    """Keeps the metrics dict each query was answered with."""

    def __init__(self) -> None:
        super().__init__()
        self.metrics: list[dict[str, Any]] = []

    def distribution(
        self,
        obs: dict[str, Any],
        actions: list[dict[str, Any]],
        metrics: dict[str, Any],
    ) -> tuple[list[float], float]:
        self.metrics.append(dict(metrics))
        return super().distribution(obs, actions, metrics)


class _RestSeekingPolicy:
    """Takes a campfire when it believes it is below half HP, otherwise the
    highest column. A crude stand-in for the champion, which rest-seeks 95% of
    the time below 50% HP and elite-avoids only when hurt."""

    def distribution(
        self,
        obs: dict[str, Any],
        actions: list[dict[str, Any]],
        metrics: dict[str, Any],
    ) -> tuple[list[float], float]:
        nodes = (obs["map"]["rows"] or [[]])[0]
        dest_row = obs["map_coord"][1] + 1
        rooms = {int(n["col"]): n.get("type") for n in nodes if int(n["row"]) == dest_row}
        hurt = float(metrics["hp"]) < float(metrics["max_hp"]) / 2
        if hurt:
            rest = [i for i, a in enumerate(actions) if rooms.get(int(a["col"])) == "RestSite"]
            if rest:
                return [1.0 if i == rest[0] else 0.0 for i in range(len(actions))], 0.5
        best = max(range(len(actions)), key=lambda i: int(actions[i]["col"]))
        return [1.0 if i == best else 0.0 for i in range(len(actions))], 0.5


class TestPvLabels:
    _STATE: ClassVar[dict[str, Any]] = {
        "hand": [{"name": "Strike"}, {"name": "Bash"}],
        "enemies": [{"name": "Cultist"}],
    }

    def test_first_ply_is_named_from_the_current_hand(self) -> None:
        pv = [{"action": "play", "card_index": 1, "target_index": 0}]
        assert ra._pv_labels(pv, self._STATE) == ["Bash → Cultist"]

    def test_deeper_plies_are_not_named(self) -> None:
        # card_index at depth>0 indexes a hand the search DREW; naming it from
        # the current hand would print a card the player does not hold.
        pv = [
            {"action": "play", "card_index": 0, "target_index": 0},
            {"action": "play", "card_index": 1, "target_index": 0},
            {"action": "end_turn"},
        ]
        assert ra._pv_labels(pv, self._STATE) == ["Strike → Cultist", "play", "end turn"]

    def test_same_named_targets_say_which_one(self) -> None:
        # Three rats all read "Two-Tailed Rat"; the HUD tags each "#k" by
        # the same count (spire-mentor-mod/EnemyTagText.cs).
        rats = {"hand": [{"name": "Strike"}], "enemies": [{"name": "Two-Tailed Rat"}] * 3}
        pv = [{"action": "play", "card_index": 0, "target_index": 1}]
        assert ra._pv_labels(pv, rats) == ["Strike → Two-Tailed Rat #2"]
        tag = HintTag(
            stable_idx=0, win_prob=0.5, bucket_label="even", is_best=True, visits=1, target_index=2
        )
        assert ra._tag_label(tag, rats) == "Strike → Two-Tailed Rat #3"

    def test_unknown_shapes_never_raise(self) -> None:
        assert ra._pv_labels([{"action": "play", "card_index": 99}], self._STATE) == ["play"]
        assert ra._pv_labels(["already a string"], self._STATE) == ["already a string"]


class TestChooseBandLabels:
    """`-300 - i` is a pick out of a pending choice, and it had no arm.

    Live run 004-silent-a8-0825-0139 wedged at floor 47 on three consecutive
    attempts. The HUD showed six options named `action -300` .. `action -305`
    and the advisor log said `best=None`; both are this band falling off the
    end of `_tag_label` / `_best_to_action`. The unnamed options read as a
    rendering fault, which is why the missing `best` -- the part that actually
    killed the run -- went unnoticed.
    """

    _CHOICE: ClassVar[dict[str, Any]] = {
        "cards": [{"name": "Survivor"}, {"id": "STRIKE_SILENT"}],
        "hand": [{"name": "Bash"}],
        "enemies": [{"name": "Cultist"}],
    }

    @staticmethod
    def _tag(stable_idx: int) -> Any:
        return HintTag(
            stable_idx=stable_idx, win_prob=0.5, bucket_label="even", is_best=True, visits=1
        )

    def test_a_pick_is_named_from_the_screen_offered_list(self) -> None:
        assert ra._tag_label(self._tag(-300), self._CHOICE) == "Survivor"
        assert ra._tag_label(self._tag(-301), self._CHOICE) == "STRIKE_SILENT"

    def test_a_pick_with_no_offered_list_still_says_which_option(self) -> None:
        # Not decoration: engine option order and world card order agree only
        # where the advisor injected the world's list, so an unproven index
        # must not borrow a name it cannot justify.
        assert ra._tag_label(self._tag(-302), {}) == "choose option 2"
        assert ra._tag_label(self._tag(-302), self._CHOICE) == "choose option 2"

    def test_the_skip_pick_is_named_too(self) -> None:
        # One above the band (-299), so the band arm does not catch it and it
        # rendered as `action -299` right beside the options it belongs to.
        assert ra._tag_label(self._tag(-299), self._CHOICE) == "skip"

    def test_the_other_bands_are_untouched(self) -> None:
        assert ra._tag_label(self._tag(-1), self._CHOICE) == "End turn"
        assert ra._tag_label(self._tag(0), self._CHOICE) == "Bash"

    def test_a_ranked_search_with_no_executable_best_says_so(self) -> None:
        # The mod waits for a `best` it will never get, and 180s later the
        # stall guard kills the run. The note is the only place that shows.
        advice = ra.build_advice([self._tag(-300)], self._CHOICE, sims=8)
        assert "best" not in advice
        assert "no executable form for Survivor" in advice["note"]

    def test_an_executable_best_adds_no_such_note(self) -> None:
        advice = ra.build_advice([self._tag(-1)], self._CHOICE, sims=8)
        assert advice["best"] == {"action": "end_turn", "label": "End turn"}
        assert "no executable form" not in advice["note"]


class TestUnrankedFallback:
    """A surface no head can rank must still leave autoplay able to move."""

    def test_names_the_first_option_and_says_it_is_not_a_recommendation(self) -> None:
        state = {"options": [{"label": "A"}, {"label": "B"}]}
        advice = ra._unranked(state, "no trained head for widget")
        assert advice["best"] == {"action": "choose", "index": 0, "label": "A"}
        assert [o["label"] for o in advice["options"]] == ["A", "B"]
        assert "NOT a recommendation" in advice["note"]

    def test_no_options_yields_a_note_only(self) -> None:
        advice = ra._unranked({}, "no trained head for widget")
        assert "best" not in advice


class TestBuildCardBundle:
    _BUNDLES: ClassVar[dict[str, Any]] = {
        "run_state": _RUN_STATE,
        "type": "card_bundle",
        "floor": 5,
        "act": 1,
        "options": [
            {"index": 0, "cards": [{"id": "STRIKE_IRONCLAD", "name": "Strike"}]},
            {
                "index": 1,
                "cards": [
                    {"id": "BASH", "name": "Bash"},
                    {"id": "DEFEND_IRONCLAD", "name": "Defend"},
                ],
            },
        ],
    }

    def test_one_action_per_bundle_pointing_at_its_first_card(self) -> None:
        decision = ra._build_card_bundle(self._BUNDLES)
        assert decision is not None
        # Bundle 1's cards are flattened after bundle 0's single card, so its
        # action must index 1, not 0 -- getting this wrong scores every bundle
        # by the first bundle's contents.
        assert decision.actions == [
            {"action": "pick_card_reward", "idx": 0},
            {"action": "pick_card_reward", "idx": 1},
        ]
        assert decision.responses == [
            {"action": "choose", "index": 0},
            {"action": "choose", "index": 1},
        ]
        assert decision.labels == ["Strike", "Bash + Defend"]
        assert decision.obs["card_reward_option_ids"] == [
            "STRIKE_IRONCLAD",
            "BASH",
            "DEFEND_IRONCLAD",
        ]
        # Index-aligned with the FLATTENED ids, not with the bundles.
        assert decision.obs["card_reward_option_upgraded"] == [False, False, False]

    def test_empty_or_cardless_bundles_return_none(self) -> None:
        assert ra._build_card_bundle({**self._BUNDLES, "options": []}) is None
        assert ra._build_card_bundle({**self._BUNDLES, "options": [{"index": 0}]}) is None


class TestProjectMapPath:
    def test_follows_single_child_chains_without_asking_the_net(self) -> None:
        # A forced move is not a decision; querying the net for it would be
        # noise (and the net has no legal action set with one option).
        graph = [
            {"col": 2, "row": 6, "type": "Monster", "children": [{"col": 2, "row": 7}]},
            {"col": 2, "row": 7, "type": "Shop", "children": [{"col": 2, "row": 8}]},
            {"col": 2, "row": 8, "type": "Elite", "children": []},
        ]
        state = _state([{"index": 0, "type": "Monster", "row": 6, "col": 2}], graph)
        policy = _RightmostPolicy()

        path = ra._project_map_path(state, policy, 0)

        assert [(p["col"], p["row"], p["type"], p["hop"]) for p in path] == [
            (2, 6, "Monster", 0),
            (2, 7, "Shop", 1),
            (2, 8, "Elite", 2),
        ]
        assert policy.calls == []

    def test_branches_are_resolved_by_the_policy(self) -> None:
        graph = [
            {
                "col": 2,
                "row": 6,
                "type": "Monster",
                "children": [{"col": 1, "row": 7}, {"col": 3, "row": 7}],
            },
            {"col": 1, "row": 7, "type": "Shop", "children": []},
            {"col": 3, "row": 7, "type": "RestSite", "children": []},
        ]
        state = _state([{"index": 0, "type": "Monster", "row": 6, "col": 2}], graph)
        policy = _RightmostPolicy()

        path = ra._project_map_path(state, policy, 0)

        assert [(p["col"], p["row"]) for p in path] == [(2, 6), (3, 7)]
        # The query is asked FROM the node being planned, not from where the
        # player currently stands -- otherwise every hop re-answers hop 0.
        assert policy.calls[0][0]["_map_coord"] == (2, 6)

    def test_starts_at_the_recommended_option_not_the_first_one(self) -> None:
        graph = [
            {"col": 2, "row": 6, "type": "Monster", "children": []},
            {"col": 3, "row": 6, "type": "RestSite", "children": []},
        ]
        state = _state(
            [
                {"index": 0, "type": "Monster", "row": 6, "col": 2},
                {"index": 1, "type": "RestSite", "row": 6, "col": 3},
            ],
            graph,
        )

        path = ra._project_map_path(state, _RightmostPolicy(), 1)

        assert [(p["col"], p["row"]) for p in path] == [(3, 6)]

    def test_cycles_and_missing_graphs_terminate(self) -> None:
        # A child that points back at its parent must not loop forever, and a
        # child that is not in the node list at all must not be walked into.
        graph = [
            {"col": 2, "row": 6, "type": "Monster", "children": [{"col": 2, "row": 6}]},
        ]
        state = _state([{"index": 0, "type": "Monster", "row": 6, "col": 2}], graph)
        assert len(ra._project_map_path(state, _RightmostPolicy(), 0)) == 1

        assert ra._project_map_path({**state, "map": {}}, _RightmostPolicy(), 0) == []

    def test_a_pathological_graph_still_stops_at_the_defensive_cap(self) -> None:
        # No real act is this deep; the cap exists so a malformed graph cannot
        # wedge the walk, not to truncate an honest route.
        graph = [
            {
                "col": 2,
                "row": row,
                "type": "Monster",
                "children": [{"col": 2, "row": row + 1}],
            }
            for row in range(6, 6 + ra.MAX_PROJECTED_HOPS + 10)
        ]
        state = _state([{"index": 0, "type": "Monster", "row": 6, "col": 2}], graph)
        path = ra._project_map_path(state, _RightmostPolicy(), 0)
        assert len(path) == ra.MAX_PROJECTED_HOPS

    def test_a_policy_that_raises_truncates_instead_of_failing(self) -> None:
        class _Broken:
            def distribution(self, obs: Any, actions: Any, metrics: Any) -> Any:
                raise RuntimeError("net exploded")

        graph = [
            {
                "col": 2,
                "row": 6,
                "type": "Monster",
                "children": [{"col": 1, "row": 7}, {"col": 3, "row": 7}],
            },
            {"col": 1, "row": 7, "type": "Shop", "children": []},
            {"col": 3, "row": 7, "type": "RestSite", "children": []},
        ]
        state = _state([{"index": 0, "type": "Monster", "row": 6, "col": 2}], graph)

        path = ra._project_map_path(state, _Broken(), 0)

        # The node the player must click now still survives: a broken
        # projection must never cost the advice itself.
        assert [(p["col"], p["row"]) for p in path] == [(2, 6)]


#: Round numbers so an assertion reads as arithmetic, not as a lookup. Act 1
#: (``_state``'s act) indexes slot 0: monster 10, elite 25.
_HP_TABLE: dict[str, Any] = {
    "monster_hp_loss": [10.0, 20.0, 30.0],
    "elite_hp_loss": [25.0, 35.0, 45.0],
    "event_hp_loss": 2.0,
    "unknown_hp_loss": 4.0,
    "rest_heal_frac": 0.3,
}


class TestProjectedHp:
    """The projection walks HP forward so late hops are answered at the HP the
    player would really arrive with -- the champion rest-seeks only when hurt,
    so a frozen-HP walk is structurally blind to campfires."""

    def test_each_node_carries_the_hp_the_player_would_arrive_with(self) -> None:
        graph = [
            {"col": 2, "row": 6, "type": "Monster", "children": [{"col": 2, "row": 7}]},
            {"col": 2, "row": 7, "type": "Monster", "children": [{"col": 2, "row": 8}]},
            {"col": 2, "row": 8, "type": "Shop", "children": []},
        ]
        state = _state([{"index": 0, "type": "Monster", "row": 6, "col": 2}], graph)

        path = ra._project_map_path(state, _RightmostPolicy(), 0, hp_table=_HP_TABLE)

        # Arrival HP, before that node's own room resolves: you reach the first
        # monster undamaged, the second down one fight, the shop down two.
        assert [(p["hop"], p["hp"]) for p in path] == [(0, 60), (1, 50), (2, 40)]

    def test_rest_sites_heal_the_projection(self) -> None:
        graph = [
            {"col": 2, "row": 6, "type": "Monster", "children": [{"col": 2, "row": 7}]},
            {"col": 2, "row": 7, "type": "RestSite", "children": [{"col": 2, "row": 8}]},
            {"col": 2, "row": 8, "type": "Monster", "children": []},
        ]
        state = _state([{"index": 0, "type": "Monster", "row": 6, "col": 2}], graph)

        path = ra._project_map_path(state, _RightmostPolicy(), 0, hp_table=_HP_TABLE)

        # 60 -> fight -> 50 -> campfire heals floor(75 * 0.3) = 22 -> 72.
        assert [p["hp"] for p in path] == [60, 50, 72]

    def test_elites_cost_more_than_monsters(self) -> None:
        graph = [
            {"col": 2, "row": 6, "type": "Elite", "children": [{"col": 2, "row": 7}]},
            {"col": 2, "row": 7, "type": "Shop", "children": []},
        ]
        state = _state([{"index": 0, "type": "Elite", "row": 6, "col": 2}], graph)

        path = ra._project_map_path(state, _RightmostPolicy(), 0, hp_table=_HP_TABLE)

        assert [p["hp"] for p in path] == [60, 35]

    def test_shops_and_treasures_are_free_while_events_and_unknowns_are_not(self) -> None:
        graph = [
            {"col": 2, "row": 6, "type": "Shop", "children": [{"col": 2, "row": 7}]},
            {"col": 2, "row": 7, "type": "Treasure", "children": [{"col": 2, "row": 8}]},
            {"col": 2, "row": 8, "type": "Ancient", "children": [{"col": 2, "row": 9}]},
            {"col": 2, "row": 9, "type": "Unknown", "children": [{"col": 2, "row": 10}]},
            {"col": 2, "row": 10, "type": "Monster", "children": []},
        ]
        state = _state([{"index": 0, "type": "Shop", "row": 6, "col": 2}], graph)

        path = ra._project_map_path(state, _RightmostPolicy(), 0, hp_table=_HP_TABLE)

        # Shop and treasure cost nothing; the event pays 2 and the ? room 4.
        assert [p["hp"] for p in path] == [60, 60, 60, 58, 54]

    def test_healing_is_capped_at_max_hp(self) -> None:
        graph = [
            {"col": 2, "row": 6, "type": "RestSite", "children": [{"col": 2, "row": 7}]},
            {"col": 2, "row": 7, "type": "Shop", "children": []},
        ]
        state = _state([{"index": 0, "type": "RestSite", "row": 6, "col": 2}], graph)
        state["run_state"] = {**_RUN_STATE, "hp": 70}

        path = ra._project_map_path(state, _RightmostPolicy(), 0, hp_table=_HP_TABLE)

        # 70 + 22 would be 92; a campfire cannot take you past max_hp.
        assert [p["hp"] for p in path] == [70, 75]

    def test_projected_hp_never_falls_below_one(self) -> None:
        graph = [
            {
                "col": 2,
                "row": row,
                "type": "Elite",
                "children": [{"col": 2, "row": row + 1}],
            }
            for row in range(6, 12)
        ]
        state = _state([{"index": 0, "type": "Elite", "row": 6, "col": 2}], graph)

        path = ra._project_map_path(state, _RightmostPolicy(), 0, hp_table=_HP_TABLE)

        # Six elites at 25 each is -90 against 60 HP. The projection is a
        # drawing, not a death sentence -- it floors rather than going negative.
        assert [p["hp"] for p in path] == [60, 35, 10, 1, 1, 1]

    def test_the_net_is_asked_at_the_hp_it_would_arrive_with(self) -> None:
        graph = [
            {"col": 2, "row": 6, "type": "Monster", "children": [{"col": 2, "row": 7}]},
            {
                "col": 2,
                "row": 7,
                "type": "Monster",
                "children": [{"col": 1, "row": 8}, {"col": 3, "row": 8}],
            },
            {"col": 1, "row": 8, "type": "Shop", "children": []},
            {"col": 3, "row": 8, "type": "Shop", "children": []},
        ]
        state = _state([{"index": 0, "type": "Monster", "row": 6, "col": 2}], graph)
        policy = _RecordingPolicy()

        ra._project_map_path(state, policy, 0, hp_table=_HP_TABLE)

        # The one branch sits two fights in. Asking it at the live 60 HP is the
        # whole bug: the champion only rest-seeks when it believes it is hurt.
        assert [m["hp"] for m in policy.metrics] == [40]

    def test_the_route_walks_the_whole_act_not_just_eight_hops(self) -> None:
        graph = [
            {
                "col": 2,
                "row": row,
                "type": "Shop",
                "children": [{"col": 2, "row": row + 1}],
            }
            for row in range(6, 20)
        ]
        state = _state([{"index": 0, "type": "Shop", "row": 6, "col": 2}], graph)

        path = ra._project_map_path(state, _RightmostPolicy(), 0, hp_table=_HP_TABLE)

        assert len(path) == 14

    def test_no_table_still_draws_the_route_but_claims_no_hp(self) -> None:
        # The advisor's never-crash contract: if the engine cannot hand over
        # its cost tables, the route still draws. It must NOT then report the
        # live HP as an arrival HP -- a frozen number labelled as a projection
        # is worse than no number, and the HUD keys its label off this field.
        graph = [
            {"col": 2, "row": 6, "type": "Elite", "children": [{"col": 2, "row": 7}]},
            {"col": 2, "row": 7, "type": "Elite", "children": []},
        ]
        state = _state([{"index": 0, "type": "Elite", "row": 6, "col": 2}], graph)

        path = ra._project_map_path(state, _RightmostPolicy(), 0, hp_table=None)

        assert [(p["col"], p["row"]) for p in path] == [(2, 6), (2, 7)]
        assert all("hp" not in p for p in path)

    def test_a_decayed_route_reaches_a_campfire_a_frozen_one_walks_past(self) -> None:
        # The point of the whole change, as one assertion. Three fights in, a
        # cost-aware walk believes it is hurt and takes the campfire; the frozen
        # walk still believes it is at 60/75 and strolls past it.
        graph = [
            {"col": 2, "row": 6, "type": "Monster", "children": [{"col": 2, "row": 7}]},
            {"col": 2, "row": 7, "type": "Monster", "children": [{"col": 2, "row": 8}]},
            {
                "col": 2,
                "row": 8,
                "type": "Monster",
                "children": [{"col": 1, "row": 9}, {"col": 3, "row": 9}],
            },
            {"col": 1, "row": 9, "type": "RestSite", "children": []},
            {"col": 3, "row": 9, "type": "Monster", "children": []},
        ]
        state = _state([{"index": 0, "type": "Monster", "row": 6, "col": 2}], graph)

        decayed = ra._project_map_path(state, _RestSeekingPolicy(), 0, hp_table=_HP_TABLE)
        frozen = ra._project_map_path(state, _RestSeekingPolicy(), 0, hp_table=None)

        assert decayed[-1]["type"] == "RestSite"
        assert frozen[-1]["type"] == "Monster"


class TestProjectionHpTable:
    """The cost tables come from the engine, so the HUD and the run-AI planner
    cannot drift apart. `heuristic_v.toml` is `include_str!`'d into the Rust
    binary, so a Python-side copy would be a second source of truth."""

    def test_reads_the_tables_from_the_engine(self, monkeypatch: Any) -> None:
        monkeypatch.setitem(
            sys.modules, "sts2_rs", SimpleNamespace(projection_hp_table=lambda: _HP_TABLE)
        )
        assert ra._projection_hp_table() == _HP_TABLE

    def test_an_engine_without_the_getter_degrades_to_no_table(self, monkeypatch: Any) -> None:
        # An older wheel has no such function. The route must still draw --
        # frozen, as it was before -- rather than the advice vanishing.
        monkeypatch.setitem(sys.modules, "sts2_rs", SimpleNamespace())
        assert ra._projection_hp_table() is None


class TestRouteNote:
    def test_names_the_hp_the_route_expects_to_reach_the_boss_with(self) -> None:
        path = [{"hop": 0, "hp": 60}, {"hop": 1, "hp": 50}, {"hop": 2, "hp": 41}]
        note = ra._route_note(path)
        assert "3 floors" in note
        assert "41" in note
        # Only HP is walked forward; saying so is the difference between a
        # projection the player can calibrate and one they will trust blindly.
        assert "deck" in note

    def test_a_frozen_route_does_not_claim_an_arrival_hp(self) -> None:
        path = [{"hop": 0}, {"hop": 1}]
        assert "HP" not in ra._route_note(path)


class TestEngineProjectionTable:
    """`sts2_rs.projection_hp_table()` is the single source of truth wire: the
    numbers must be the ones the planner itself projects with."""

    def test_the_getter_returns_the_toml_the_engine_embeds(self) -> None:
        sts2_rs = pytest.importorskip("sts2_rs")
        toml = Path(__file__).parents[2] / "sts2-rs" / "data" / "runai" / "heuristic_v.toml"
        expected = tomllib.loads(toml.read_text(encoding="utf-8"))["projection"]

        table = sts2_rs.projection_hp_table()

        assert table["monster_hp_loss"] == expected["monster_hp_loss"]
        assert table["elite_hp_loss"] == expected["elite_hp_loss"]
        assert table["event_hp_loss"] == expected["event_hp_loss"]
        assert table["unknown_hp_loss"] == expected["unknown_hp_loss"]
        assert table["rest_heal_frac"] == expected["rest_heal_frac"]

    def test_the_live_table_drives_a_real_route(self) -> None:
        # End to end on the shipped numbers: whatever the toml says, walking a
        # monster must cost the projection something.
        pytest.importorskip("sts2_rs")
        graph = [
            {"col": 2, "row": 6, "type": "Monster", "children": [{"col": 2, "row": 7}]},
            {"col": 2, "row": 7, "type": "Shop", "children": []},
        ]
        state = _state([{"index": 0, "type": "Monster", "row": 6, "col": 2}], graph)

        path = ra._project_map_path(state, _RightmostPolicy(), 0, ra._projection_hp_table())

        assert path[1]["hp"] < path[0]["hp"]


def _fixture_board() -> dict[str, Any]:
    import json

    path = Path(__file__).parent / "fixtures" / "realgame_adapter" / "single_enemy_ironclad.json"
    state = json.loads(path.read_text(encoding="utf-8"))["state"]
    state.pop("game_version", None)
    state["type"] = "combat_action"
    return state


def _row(action: dict[str, Any], p_win: float, visits: int, *, best: bool = False) -> dict:
    wins = round(p_win * visits)
    return {
        "action": action,
        "p_win": p_win,
        "q": p_win,
        "visits": visits,
        "wins": wins,
        "losses": visits - wins,
        "ties": 0,
        "is_best": best,
        "mean_final_hp": 50.0 + 10 * p_win,
        "hp_n": visits,
    }


class TestCombatAdviceRows:
    """`compute_combat_advice` also hands back the search's regret rows, and
    the rows crown the same move the advice does (the coach grades against
    them, the HUD shows the advice -- they must not disagree)."""

    def _canned(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sts2_rs = pytest.importorskip("sts2_rs")
        result = {
            "actions": [
                _row({"action": "play", "card_index": 4, "target_index": 0}, 0.9, 500, best=True),
                _row({"action": "play", "card_index": 2, "target_index": 0}, 0.8, 300),
                _row({"action": "end_turn"}, 0.5, 100),
            ],
            "pv": [{"action": "play", "card_index": 4, "target_index": 0}],
        }
        monkeypatch.setattr(sts2_rs, "mcts_recommend", lambda *a, **k: result)

    def test_compute_combat_advice_rows_match_options_best(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._canned(monkeypatch)
        advice, _cs, _cfg, rows = ra.compute_combat_advice(
            _fixture_board(), ra.RecommendConfig(threads=1), 64
        )
        (best_row,) = [r for r in rows if r["is_best"]]
        assert best_row["stable_idx"] == advice["best"]["card_index"] == 4
        assert best_row["target_index"] == advice["best"].get("target_index")
        assert rows[0] is best_row
        assert best_row["mean_final_hp"] == pytest.approx(59.0)
        assert best_row["hp_n"] == 500

    def test_rows_follow_the_veto_onto_the_new_best(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._canned(monkeypatch)
        board = _fixture_board()
        board["hand"][4]["playable"] = False  # the game refuses the Bash
        advice, _cs, _cfg, rows = ra.compute_combat_advice(board, ra.RecommendConfig(threads=1), 64)
        assert advice["best"]["card_index"] == 2
        (best_row,) = [r for r in rows if r["is_best"]]
        assert best_row["stable_idx"] == 2
        assert rows[0] is best_row

    def test_options_carry_final_hp(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._canned(monkeypatch)
        advice, *_ = ra.compute_combat_advice(_fixture_board(), ra.RecommendConfig(threads=1), 64)
        best = next(o for o in advice["options"] if o["is_best"])
        assert best["mean_final_hp"] == pytest.approx(59.0)
        assert best["hp_n"] == 500


class TestRunAgentParity:
    """The advisor searches the way the Rust run agent plays, so a win rate
    measured in Rust is the one the mod gets."""

    def test_default_budget_is_the_run_agents_single_tree_at_4x(self, monkeypatch) -> None:
        seen: dict[str, Any] = {}
        monkeypatch.setattr(ra, "serve", lambda args: seen.setdefault("args", args))
        ra.main([])
        args = seen["args"]
        assert (args.sims, args.boss_sims, args.threads) == (4000, 16000, 1)

    @staticmethod
    def _boss(run_state: dict[str, Any], act: int = 3) -> dict[str, Any]:
        return {"room_type": "Boss", "act": act, "run_state": run_state}

    def test_last_boss_values_only_winning(self) -> None:
        state = self._boss({"ascension": 9, "is_last_act": True, "awaiting_second_boss": False})
        cfg = ra.fight_search_cfg(ra.RecommendConfig(), state)
        assert (cfg.win_hp_carry, cfg.win_potion_carry) == (0.0, 0.0)

    @pytest.mark.parametrize(
        "state",
        [
            {"room_type": "Monster", "act": 3, "run_state": {"is_last_act": True}},
            # A10's first act-3 boss: the second one is still to come.
            {
                "room_type": "Boss",
                "act": 3,
                "run_state": {"ascension": 10, "is_last_act": True, "awaiting_second_boss": True},
            },
            # An act-1 boss: the run agent's default (`LastBoss`) leaves it alone.
            {
                "room_type": "Boss",
                "act": 1,
                "run_state": {"ascension": 5, "is_last_act": False, "awaiting_second_boss": False},
            },
            # An older mod at A10 cannot say which act-3 boss this is.
            {"room_type": "Boss", "act": 3, "run_state": {"ascension": 10}},
        ],
    )
    def test_every_other_fight_keeps_the_default_value(self, state: dict[str, Any]) -> None:
        cfg = ra.RecommendConfig()
        assert ra.fight_search_cfg(cfg, state) is cfg

    def test_older_mod_below_a10_reads_act_three_as_the_last_boss(self) -> None:
        cfg = ra.fight_search_cfg(ra.RecommendConfig(), self._boss({"ascension": 9}))
        assert (cfg.win_hp_carry, cfg.win_potion_carry) == (0.0, 0.0)
