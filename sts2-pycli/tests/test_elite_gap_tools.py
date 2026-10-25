"""The GH #830 elite-gap tools: per-encounter gap table and the replay duel."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ppo.combat_bench import encounter_gap, replay_duel
from ppo.combat_bench import report as R


def _fight(key: str, enc: str, player: str, hp: int, dmg: int, died: bool) -> dict:
    return {
        "key": key,
        "file": key.split(":")[0],
        "player": player,
        "room_type": "elite",
        "act": 1,
        "encounter": enc,
        "hp_before": hp,
        "human": {"damage_taken": dmg, "died": died, "quit": False, "turns": 4},
    }


def _played(path: Path, cells: list[tuple[str, int, int, bool]]) -> None:
    head = {"header": {"schema": R.PLAYED_SCHEMA, "reps": 2, "rep_offset": 0}}
    lines = [json.dumps(head)]
    for key, rep, dmg, died in cells:
        rec = {"rep": rep, "end": "died" if died else "won", "died": died, "dmg_c": dmg, "turns": 5}
        lines.append(json.dumps({"key": key, "block": 0, "rep": rec}))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_encounter_rows_rank_by_total_gap_and_average_reps(tmp_path: Path) -> None:
    fights = {
        f["key"]: f
        for f in [
            _fight("a.run.gz:1", "EEL_ELITE", "p1", 50, 10, False),
            _fight("b.run.gz:1", "EEL_ELITE", "p2", 50, 10, False),
            _fight("c.run.gz:1", "BUG_ELITE", "p3", 40, 20, False),
        ]
    }
    played = tmp_path / "arm.jsonl"
    _played(
        played,
        [
            ("a.run.gz:1", 0, 30, False),
            ("a.run.gz:1", 1, 10, False),  # mean 20 -> gap +10
            ("b.run.gz:1", 0, 0, True),  # died: 50 + 65 = 115 -> +105
            ("b.run.gz:1", 1, 0, True),
            ("c.run.gz:1", 0, 20, False),  # gap 0
            ("c.run.gz:1", 1, 20, False),
        ],
    )
    rows = encounter_gap.arm_rows(fights, str(played), "B0", "elite")
    table = encounter_gap.encounter_rows(rows, R.Opts(boot=99))
    assert [r["encounter"] for r in table] == ["ALL", "EEL_ELITE", "BUG_ELITE"]
    eel = table[1]
    assert eel["fights"] == 2
    assert eel["gap"] == pytest.approx((10 + 105) / 2)
    assert eel["share"] == pytest.approx(1.0)
    assert eel["pilot_died"] == pytest.approx(50.0)
    assert table[0]["gap_sum"] == pytest.approx(115.0)
    assert "EEL_ELITE" in encounter_gap.render({"B0": table})


def test_loss_matches_the_bench_l() -> None:
    assert replay_duel.loss(False, 30, 50) == 30
    assert replay_duel.loss(False, 80, 50) == 50
    assert replay_duel.loss(True, 10, 50) == 115


def test_human_trace_rounds_damage_plays_and_potions() -> None:
    rec = {
        "result": "victory",
        "hp_lost_total": 12,
        "turns": 2,
        "pre": [{"t": "turn", "side": "player", "round": 1}],
        "decisions": [
            {
                "ev": {"t": "play", "id": "STRIKE_IRONCLAD", "auto": False},
                "after": [],
            },
            {
                "ev": {"t": "potion_start", "id": "FIRE_POTION"},
                "after": [{"t": "potion_used", "id": "FIRE_POTION"}],
            },
            {
                "ev": {"t": "end_turn", "side": "player"},
                "after": [
                    {"t": "hit", "dst": "player", "dmg": 7, "blocked": 5},
                    {"t": "turn", "side": "player", "round": 2},
                ],
            },
            {
                "ev": {"t": "play", "id": "CRUELTY", "auto": False},
                "after": [{"t": "hit", "dst": "player", "dmg": 5, "blocked": 0}],
            },
        ],
    }
    h = replay_duel.human_trace(rec)
    assert h["died"] is False
    assert h["dmg"] == 12
    assert h["dmg_by_round"] == {"1": 7, "2": 5}
    assert h["blocked_by_round"] == {"1": 5, "2": 0}
    assert h["plays"] == [(1, "STRIKE_IRONCLAD"), (2, "CRUELTY")]
    assert h["potions"] == [(1, "FIRE_POTION")]


def test_duel_summary_pairs_arms_with_the_human_and_splits_clean(tmp_path: Path) -> None:
    from ppo.combat_bench import duel_report

    def row(enc: str, player_file: str, h: int, b0: int, sync: object) -> dict:
        side = lambda L: {  # noqa: E731
            "L": L,
            "died": False,
            "rounds": 3,
            "potions": [],
            "dmg_by_round": {"1": L},
        }
        return {
            "file": player_file,
            "encounter": enc,
            "sync": sync,
            "human": side(h),
            "B0": side(b0),
        }

    rows = [
        row("EEL", "a", 10, 30, None),
        row("EEL", "b", 0, 10, ["enemy_hp", ""]),
        row("EEL", "b", 0, 60, None) | {"attempt": 1},  # a reloaded fight
    ]
    (tmp_path / "fights.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    files = [{"file": "/x/a", "username": "u1"}, {"file": "/x/b", "username": "u2"}]
    (tmp_path / "selection.json").write_text(json.dumps({"files": files}))
    s = duel_report.summarize(duel_report.load(tmp_path), ["B0"])
    assert s["EEL"]["all"]["fights"] == 3
    assert s["EEL"]["noreload"]["fights"] == 2
    assert s["EEL"]["noreload"]["B0"]["gap"] == pytest.approx(15.0)
    assert s["EEL"]["clean"]["fights"] == 1
    assert s["EEL"]["clean"]["B0"]["gap"] == pytest.approx(20.0)
    assert s["EEL"]["noreload"]["human"]["dmg_by_round"][0] == pytest.approx(5.0)
    assert "EEL" in duel_report.render(s, ["B0"])


def test_a_loss_is_a_human_death() -> None:
    from ppo.bc_human import replay

    assert replay.DEATH == "loss" and replay.DEATH in replay.FINISHED
    rec = {"result": "loss", "hp_lost_total": 30, "turns": 3, "pre": [], "decisions": []}
    assert replay_duel.human_trace(rec)["died"] is True


def test_a_replayed_play_is_not_a_human_decision(tmp_path: Path) -> None:
    import gzip

    from ppo.bc_human import replay

    events = [
        {"t": "header", "seed": "S", "character": "IRONCLAD", "ascension": 10},
        {"t": "combat_start", "combat_id": "1.1:X_ELITE", "encounter": "X_ELITE", "enemies": []},
        {"t": "play", "id": "POMMEL_STRIKE", "auto": False, "play_index": 0, "play_count": 2},
        {"t": "hit", "dst": "X", "dmg": 10},
        {"t": "play", "id": "POMMEL_STRIKE", "auto": False, "play_index": 1, "play_count": 2},
        {"t": "hit", "dst": "X", "dmg": 10},
        {"t": "end_turn", "side": "player"},
        {"t": "combat_end", "result": "victory"},
    ]
    path = tmp_path / "r.replay.ndjson.gz"
    with gzip.open(path, "wt") as fh:
        fh.write("\n".join(json.dumps(e) for e in events) + "\n")
    (rec,) = list(replay.fights(str(path)))
    kinds = [(d["ev"]["t"], d["ev"].get("play_index")) for d in rec["decisions"]]
    assert kinds == [("play", 0), ("end_turn", None)]
    assert [e["t"] for e in rec["decisions"][0]["after"]] == ["hit", "play", "hit"]
    assert rec["attempt"] == 0


def test_takeover_maps_combat_actions_to_run_actions() -> None:
    from ppo.combat_bench import takeover

    assert takeover.run_action({"action": "play", "card_index": 2, "target_index": 0}) == {
        "action": "play_card",
        "hand_idx": 2,
        "target_index": 0,
    }
    assert takeover.run_action({"action": "end_turn"}) == {"action": "end_turn"}
    assert takeover.run_action({"action": "drink_potion", "slot": 1, "target_index": None}) == {
        "action": "drink_potion",
        "potion_slot": 1,
        "target_index": None,
    }
    assert takeover.run_action({"action": "choose", "index": 3, "selection": [3]}) == {
        "action": "combat_resolve_choice",
        "option": 3,
    }
    assert takeover.run_action({"action": "choose", "selection": [1, 2]}) is None


def test_takeover_loss_counts_player_damage_rows_only() -> None:
    from ppo.combat_bench import takeover

    log = [
        {"kind": "damage", "receiver": {"kind": "player"}, "hp_before": 50, "hp_after": 44},
        {"kind": "damage", "receiver": {"kind": "monster"}, "hp_before": 30, "hp_after": 0},
        {"kind": "block", "receiver": {"kind": "player"}},
    ]
    assert takeover.player_damage(log) == 6
    assert takeover.loss(False, 6, 50) == 6
    assert takeover.loss(True, 6, 50) == 115


def test_takeover_forces_a_human_line_exactly() -> None:
    sts2_rs = pytest.importorskip("sts2_rs")
    from ppo.combat_bench import takeover
    from ppo.combat_bench.run import build_run, load_set

    fight = next(
        f
        for f in load_set(
            str(Path(__file__).resolve().parents[1] / "data/combat_bench/ironclad_devpool_v1.jsonl")
        )
        if f["encounter"].endswith("_WEAK")
    )
    state = build_run(fight, 3).combat_state()
    rec = {
        "character": "IRONCLAD",
        "ascension": fight["ascension"],
        "encounter": fight["encounter"],
    }
    legal = [a for a in state.legal_actions() if a.get("action") == "play"]
    forced = [legal[0], {"action": "end_turn"}]
    a = takeover.play_rest(rec, state, forced, 20, 1)
    b = takeover.play_rest(rec, state, forced, 20, 1)
    assert a["end"] in ("won", "died") and a == b  # forced line reproduced, seeded pilot
    p = takeover.play_rest(rec, state, [], 20, 1)
    assert p["end"] in ("won", "died") and p["hp_start"] == a["hp_start"]
    assert sts2_rs is not None


def test_takeover_features_count_potions_powers_and_attacks() -> None:
    from ppo.combat_bench import takeover

    g = {"hand": [{"type": "Power"}, {"type": "Attack"}, {"type": "Skill"}]}
    steps = [
        (g, {"action": "play", "card_index": 0}),
        (g, {"action": "play", "card_index": 1, "target_index": 0}),
        (g, {"action": "drink_potion", "slot": 0}),
    ]
    assert takeover.features_of(steps, None) == {"potions": 1, "powers": 1, "attacks": 1}
    assert takeover.rollout_value(None, [1, 2]) is None


def test_takeover_records_both_first_turns_with_rollouts() -> None:
    pytest.importorskip("sts2_rs")
    from ppo.combat_bench import takeover
    from ppo.combat_bench.run import build_run, load_set

    fight = next(
        f
        for f in load_set(
            str(Path(__file__).resolve().parents[1] / "data/combat_bench/ironclad_devpool_v1.jsonl")
        )
        if f["encounter"].endswith("_WEAK")
    )
    state = build_run(fight, 3).combat_state()
    rec = {
        "character": "IRONCLAD",
        "ascension": fight["ascension"],
        "encounter": fight["encounter"],
    }
    p = takeover.play_rest(rec, state, [], 20, 1, rollouts=[5, 6])
    first = p["first"]
    assert first["root"] and first["chosen"] in [r[0] for r in first["root"]]
    assert first["rollout"]["n"] == 2 and "block" in first["features"]
    assert first["trace"][-1] == "end_turn" or first["end"] is None
    assert first["after_enemy"] is None or "hp" in first["after_enemy"]
    t1 = takeover.play_rest(rec, state, [], 20, 1, rollouts=[], first_turn_only=True)
    assert t1["end"] == "turn1" and t1["first"]["trace"] == first["trace"]
    # Forced arms record nothing extra.
    assert "first" not in takeover.play_rest(rec, state, [{"action": "end_turn"}], 20, 1, [5])
    h = takeover.human_turn(state, [{"action": "end_turn"}], [5, 6])
    assert h["first"] == ["end_turn"] and h["features"]["potions"] == 0
    assert h["trace"] == ["end_turn"] and h["end"]["hand"]
    assert takeover.summarize(state)["enemies"] and h["after_enemy"]["hp"] is not None
    assert h["rollout"]["n"] == 2
    assert h["rollout"] == takeover.human_turn(state, [{"action": "end_turn"}], [5, 6])["rollout"]
