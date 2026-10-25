"""`ppo.turnplan.pro`: human turns located among the turn plans (GH #830).

Pins the corpus filters (pro_v1 uploader exclusion, room type), the log's
turn-start hand, the label record a located turn becomes, and, with the
wheel, that `turn_plan_labels(locate=...)` finds a played turn and skips the
pricing at `sims_per_plan == 0`.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
from ppo.turnplan import pro

BENCH = Path(__file__).resolve().parents[2] / "data" / "combat_bench"


def test_player_id_matches_the_bench_schema() -> None:
    # SCHEMA.md: "p_" + sha256(username)[:10].
    assert pro.player_id("abc") == "p_ba7816bf8f"


def test_excluded_usernames_catch_every_case_folded_spelling() -> None:
    pro_ids = {pro.player_id("ProPlayer")}
    names = {"ProPlayer", "proplayer", " PROPLAYER", "someone"}
    assert pro.excluded_usernames(names, pro_ids) == {"ProPlayer", "proplayer", " PROPLAYER"}
    assert pro.excluded_usernames({"someone"}, pro_ids) == set()


def test_pro_set_is_the_frozen_bench_file() -> None:
    assert pro.PRO_SET == BENCH / "ironclad_pro_v1.jsonl"
    assert pro.PRO_SET.exists()


@pytest.mark.parametrize(
    ("encounter", "room"),
    [
        ("WATERFALL_GIANT_BOSS", "boss"),
        ("BYGONE_EFFIGY_ELITE", "elite"),
        ("TOADPOLES_WEAK", "monster"),
        ("SLIMES_NORMAL", "monster"),
    ],
)
def test_room_type_from_the_encounter_suffix(encounter: str, room: str) -> None:
    assert pro.room_type(encounter) == room


def test_hand_log_is_retained_cards_plus_turn_start_draws() -> None:
    h = pro.HandLog()
    h.feed({"t": "draw", "c": 1, "id": "STRIKE_IRONCLAD", "turn_start": True})
    h.feed({"t": "draw", "c": 9, "id": "ARMAMENTS"})  # a start-of-combat relic draw
    h.feed({"t": "turn", "side": "player"})
    h.feed({"t": "generate", "c": 2, "id": "DEFEND_IRONCLAD"})  # a start-of-turn relic
    h.feed({"t": "exhaust", "c": 9, "id": "ARMAMENTS"})  # a start-of-turn choice
    assert h.start_hand() == Counter({"STRIKE_IRONCLAD": 1, "DEFEND_IRONCLAD": 1})
    h.decision()
    h.feed({"t": "draw", "c": 3, "id": "BASH"})  # a mid-turn draw: not the turn's start
    h.feed({"t": "generate", "c": 5, "id": "WOUND"})  # mid-turn: not the turn's start
    assert h.start_hand() == Counter({"STRIKE_IRONCLAD": 1, "DEFEND_IRONCLAD": 1})
    h.feed({"t": "flush", "flushed_c": [1, 3], "retained_c": [2]})
    h.feed({"t": "end_turn", "side": "player"})
    h.feed({"t": "end_turn", "side": "enemy"})  # does not clear
    h.feed({"t": "draw", "c": 4, "id": "CRUELTY", "turn_start": True})
    # `CRUELTY` (replay) and `CRUELTY_CARD` (engine) are one card.
    assert h.start_hand() == Counter({"DEFEND_IRONCLAD": 1, "CRUELTY": 1})


def test_as_label_marks_only_the_human_plan_best() -> None:
    row = {
        "file": "f.replay.ndjson.gz",
        "combat_id": "1.2:X",
        "turn": 3,
        "locate": {"index": 1},
        "plans": [{"end": "end_turn"}, {"end": "chance"}, {"end": "end_turn"}],
    }
    lab = pro.as_label(row)
    assert [p["q"] for p in lab["plans"]] == [0.0, 1.0, 0.0]
    assert all(p["visits"] == 0 and not p["q_var"] for p in lab["plans"])
    assert lab["key"] == "f.replay.ndjson.gz:1.2:X:3"


def test_identity_names_the_card_not_the_slot() -> None:
    state = {"hand": [{"id": "STRIKE_IRONCLAD"}, {"id": "BASH", "upgraded": True}]}
    a = {"action": "play", "card_index": 1, "target_index": 0}
    assert pro._identity(state, a) == ("play", "BASH", True, 0)
    assert pro._identity(state, {"action": "end_turn"}) == ("end_turn",)


# ── with the wheel ───────────────────────────────────────────────────────


def _hallway_rm():
    from ppo.combat_bench.run import build_run, load_set

    fight = next(
        f
        for f in load_set(str(BENCH / "ironclad_devpool_v1.jsonl"))
        if f["encounter"].endswith("_WEAK")
    )
    return build_run(fight, 7)


def test_turn_plan_labels_locates_each_enumerated_plan_unpriced() -> None:
    sts2_rs = pytest.importorskip("sts2_rs")
    if not hasattr(sts2_rs, "run_seed_hash"):
        pytest.skip("wheel predates the bc_human replay kwargs")
    rm = _hallway_rm()
    agent = sts2_rs.MctsCombatAgent(seed=0, sims=10, boss_sims=10)
    base = agent.turn_plan_labels(rm, 0, 512)
    assert "locate" not in base
    assert all(p["q"] is None and p["visits"] is None for p in base["plans"])
    for i, plan in enumerate(base["plans"][:20]):
        lab = agent.turn_plan_labels(rm, 0, 512, locate=plan["actions"])
        assert lab["locate"] == {
            "index": i,
            "used": len(plan["actions"]),
            "end": plan["end"],
            "failure": None,
        }


def test_turn_plan_labels_reports_an_unplayable_turn() -> None:
    sts2_rs = pytest.importorskip("sts2_rs")
    if not hasattr(sts2_rs, "run_seed_hash"):
        pytest.skip("wheel predates the bc_human replay kwargs")
    rm = _hallway_rm()
    agent = sts2_rs.MctsCombatAgent(seed=0, sims=10, boss_sims=10)
    bad = agent.turn_plan_labels(rm, 0, 512, locate=[{"action": "play", "card_index": 40}])
    assert bad["locate"]["failure"] == "illegal"
    assert bad["locate"]["index"] is None
    short = agent.turn_plan_labels(rm, 0, 512, locate=[])
    assert short["locate"]["failure"] == "incomplete"


def test_locate_form_sorts_a_multi_pick_selection() -> None:
    acts = [
        {"action": "play", "card_index": 2, "target_index": 0},
        {"action": "choose", "index": 2, "selection": [2, 1]},
        {"action": "choose", "index": 0, "selection": []},
    ]
    assert pro.locate_form(acts) == [
        acts[0],
        {"action": "choose", "index": 1, "selection": [1, 2]},
        acts[2],
    ]


def test_split_rows_holds_out_eval_files_only_and_drops_unlocated() -> None:
    from ppo.turnplan import pro_train
    from ppo.turnplan.label import split_of

    files = [f"{i:016x}.replay.ndjson.gz" for i in range(40)]

    def row(f: str, idx: int | None) -> dict:
        loc = {"index": idx}
        return {"file": f, "combat_id": "c", "turn": 0, "locate": loc, "plans": [{}, {}]}

    ev = [row(f, 0) for f in files] + [row(files[0], None)]
    extra = [row(f, 1) for f in files[:5]]
    fit, hold = pro_train.split_rows(ev, extra)
    assert {r["file"] for r in hold} == {f for f in files if split_of(f) == "holdout"}
    n_extra = sum(split_of(f) == "train" for f in files[:5])
    assert len(fit) + len(hold) == len(files) + n_extra
    assert all(split_of(r["file"]) == "train" for r in fit)
    assert all([p["q"] for p in r["plans"]] == [1.0, 0.0] for r in hold)


def test_leak_guard_matches_run_hash_and_player_seed() -> None:
    from ppo.turnplan import pro_leak

    rows = [
        {"file": "aaaa.run.gz", "player": pro.player_id("Alice")},
        {"file": "bbbb.run.gz", "player": "anon_x"},
    ]
    keys = pro_leak.panel_keys(rows, {"aaaa.run.gz": "SEED1", "bbbb.run.gz": None})
    assert keys == {
        "run_hash": {"aaaa", "bbbb"},
        "player_seed": {(pro.player_id("Alice"), "SEED1")},
    }
    replays = [
        {"run_hash": "bbbb", "username": "Zed", "seed": "S"},  # same run file
        {"run_hash": "cccc", "username": "Alice", "seed": "SEED1"},  # same player and seed
        {"run_hash": "dddd", "username": "Alice", "seed": "OTHER"},
        {"run_hash": "eeee", "username": None, "seed": "SEED1"},
    ]
    assert pro_leak.leaks(replays, keys) == {"run_hash": ["bbbb"], "player_seed": ["cccc"]}


def test_leak_guard_panels_are_dev_devpool_holdout_and_pro() -> None:
    from ppo.turnplan import pro_leak
    from ppo.turnplan.label import split_of

    rows = pro_leak.panel_rows()
    panels = {r["panel"] for r in rows}
    assert panels == {"ironclad_dev_v1", "ironclad_devpool_v1", "ironclad_pro_v1"}
    assert all(
        split_of(r["file"]) == "holdout" for r in rows if r["panel"] == "ironclad_devpool_v1"
    )


def test_scraper_dir_honours_the_env_override(monkeypatch) -> None:
    from ppo.bc_human.replay import scraper_dir

    monkeypatch.setenv("STS2_RUN_SCRAPER", "/x/scraper")
    assert Path(scraper_dir("replays-spire-codex")) == Path("/x/scraper/data/replays-spire-codex")


def test_hand_matches_tolerates_only_off_window_generated_cards() -> None:
    h = pro.HandLog()
    h.feed({"t": "draw", "c": 1, "id": "STRIKE_IRONCLAD", "turn_start": True})
    h.decision()
    h.feed({"t": "end_turn", "side": "player"})
    h.feed({"t": "generate", "c": 7, "id": "BURN"})  # enemy turn: pile unknown
    h.feed({"t": "generate", "c": 8, "id": "BURN"})
    h.feed({"t": "turn", "side": "player"})
    h.feed({"t": "draw", "c": 2, "id": "BASH", "turn_start": True})
    assert pro.hand_matches(Counter({"BASH": 1, "BURN": 2}), h)
    assert pro.hand_matches(Counter({"BASH": 1}), h)  # the Burns went elsewhere
    assert not pro.hand_matches(Counter({"BASH": 1, "BURN": 3}), h)
    assert not pro.hand_matches(Counter({"BURN": 2}), h)  # a drawn card missing
