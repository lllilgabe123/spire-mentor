"""`ppo.rollout_learn` round 2 (GH #777): feature versions, the torch-vs-Rust
logit parity of the playout net, the search-value reward, mid-fight starts
(`traj` + deterministic rebuild) and the `rl` options, at toy scale.

The fights are inline human-bank rows (the `pilot_human_decks extract` shape);
the pilot runs at 16-24 sims, so every test is a few seconds at most.
"""

from __future__ import annotations

import functools
import json
import os
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ppo import rollout_learn, rollout_starts
from ppo.combat_bench import agents

sts2_rs = pytest.importorskip("sts2_rs")
torch = pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def _restore_environ() -> Iterator[None]:
    """`rl` makes the process env its engine env (`agents.apply_env`)."""
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


REPO = Path(__file__).resolve().parents[2]
IRONCLAD_NET = REPO / "sts2-rs" / "data" / "rollout_nets" / "ironclad.json"
REGENT_NET = REPO / "sts2-rs" / "data" / "rollout_nets" / "regent.json"
STARTER = [*["STRIKE_IRONCLAD"] * 5, *["DEFEND_IRONCLAD"] * 4, "BASH", "ASCENDERS_BANE"]


def _row(file: str, floor: int, room: str, encounter: str, **kw: Any) -> dict[str, Any]:
    return {
        "file": file,
        "floor": floor,
        "act": kw.pop("act", 1),
        "room_type": room,
        "encounter": encounter,
        "ascension": 10,
        "deck": kw.pop("deck", [*STARTER, "INFLAME", "SHRUG_IT_OFF", "TWIN_STRIKE"]),
        "relics": kw.pop("relics", ["BURNING_BLOOD"]),
        "potions": kw.pop("potions", ["FIRE_POTION"]),
        "hp_before": kw.pop("hp_before", 70),
        "max_hp": 80,
        **kw,
    }


HALL = _row("aaaa000000000001.run.gz", 3, "monster", "NIBBITS_WEAK")
ELITE = _row("aaaa000000000002.run.gz", 9, "elite", "BYGONE_EFFIGY_ELITE", potions=["BLOCK_POTION"])
BOSS = _row("aaaa000000000003.run.gz", 17, "boss", "VANTOM_BOSS", hp_before=80)


def _states() -> list[Any]:
    """Fight openings plus a few positions a step or two in."""
    out = []
    for i, f in enumerate((HALL, ELITE, BOSS)):
        rm = rollout_starts.build_fight(f, 1000 + i)
        out.append(rm.combat_state())
        cs = rm.combat_state()
        for a in cs.legal_actions()[:2]:
            nxt = rm.combat_state()
            nxt.apply(a)
            if nxt.legal_actions():
                out.append(nxt)
    return out


# A power-heavy deck with exhausting and Ethereal (Apparition) cards, and the
# fights where card-biased random play with it lights the most v2 columns
# (a greedy cover over every encounter; `_parity_states` pins the result).
RICH_DECK = [
    *["STRIKE_IRONCLAD", "DEFEND_IRONCLAD"] * 2,
    *["DEMON_FORM", "FEEL_NO_PAIN", "BARRICADE", "CORRUPTION", "RAGE", "DARK_EMBRACE"],
    *["JUGGERNAUT", "INFERNO", "RUPTURE", "AGGRESSION", "CRIMSON_MANTLE", "FLAME_BARRIER"],
    *["TRUE_GRIT", "OFFERING", "BLOODLETTING", "BURNING_PACT", "SHRUG_IT_OFF", "INFLAME"],
    *["TWIN_STRIKE", "BASH", "STONE_ARMOR", "UNMOVABLE", "BATTLE_TRANCE", "IMPERVIOUS"],
    *["SHOCKWAVE", "UPPERCUT", "TAUNT", "STOKE", "HEMOKINESIS", "BLOOD_WALL", "TREMBLE"],
    *["DOMINATE", "COLOSSUS", "VICIOUS", "PILLAGE", "SECOND_WIND", "APPARITION"],
    *["THUNDERCLAP", "FIGHT_ME", "BULLY", "EVIL_EYE", "UNRELENTING", "STAMPEDE"],
]
RICH_FIGHTS = [
    ("SLUMBERING_BEETLE_NORMAL", 1),
    ("SOUL_FYSH_BOSS", 2),
    ("CONSTRUCT_MENAGERIE_NORMAL", 1),
    ("VANTOM_BOSS", 1),
    ("SLIMES_NORMAL", 1),
    ("TOADPOLES_WEAK", 1),
]


@functools.lru_cache(maxsize=1)
def _parity_states() -> tuple[Any, ...]:
    """`_states()` plus mid-fight positions of `RICH_FIGHTS`: powers on both
    sides, exhausting and Ethereal cards, multi-enemy fights, enemies with
    Thorns / Artifact / Plating / Intangible / Slippery."""
    out = list(_states())
    for enc, seed in RICH_FIGHTS:
        fight = _row(
            "bbbb000000000001.run.gz", 1, "monster", enc, deck=RICH_DECK, potions=[], hp_before=80
        )
        cs = rollout_starts.build_fight(fight, seed).combat_state()
        rng = random.Random(seed)
        for i in range(90):
            acts = cs.legal_actions()
            if not acts:
                break
            if i % 2 == 0:
                out.append(cs.clone())
            cards = [a for a in acts if a.get("action") == "play"]
            cs.apply(rng.choice(cards) if cards and rng.random() < 0.85 else rng.choice(acts))
    return tuple(out)


def _keys(states: Any) -> list[str]:
    keys = {"end_turn"}
    for s in states:
        keys |= set(sts2_rs.rollout_features(s)["keys"])
    return ["<unk>", *sorted(keys)]


def _random_net(version: int, keys: list[str], monsters: list[str] | None, seed: int) -> Any:
    torch.manual_seed(seed)
    net = rollout_learn._playout_net(
        torch, 24, 8, len(keys), version, n_monsters=len(monsters or [])
    )
    with torch.no_grad():
        for p in net.parameters():
            p.normal_(0.0, 0.3)
        if net.mon is not None:
            net.mon.weight[0].zero_()
    return net


def _torch_logits(net: Any, ex: dict[str, Any]) -> np.ndarray:
    S = torch.tensor([ex["state"]], dtype=torch.float32)
    X = torch.tensor([ex["x"]], dtype=torch.float32)
    K = torch.tensor([ex["rows"]], dtype=torch.int64)
    M = torch.ones_like(K, dtype=torch.bool)
    MON = torch.tensor([ex["monsters"]], dtype=torch.int64)
    with torch.no_grad():
        return net(S, X, K, M, MON)[0].numpy()


# ── feature versions and R4 parity ───────────────────────────────────────


@pytest.mark.parametrize("version", [1, 2, 3, 4])
def test_feature_dims_match_the_engine(version: int) -> None:
    keys = ["<unk>", "end_turn"]
    net = _random_net(version, keys, None, 0)
    acts = sts2_rs.monster_acts()
    lp = sts2_rs.LearnedPlayout(json.dumps(net.to_json(keys, 1.0, monster_act=acts)))
    assert (lp.state_dim, lp.action_dim) == rollout_learn.FEATURE_DIMS[version]
    assert lp.feature_version == version
    feats = sts2_rs.rollout_features(_states()[0], version=version)
    assert len(feats["state"]) == lp.state_dim
    assert {len(x) for x in feats["x"]} == {lp.action_dim}


@pytest.mark.parametrize(
    ("version", "with_monsters"),
    [(1, False), (2, False), (2, True), (3, False)],
    ids=["v1", "v2", "v2+enc", "v3"],
)
def test_torch_logits_equal_the_rust_net(version: int, with_monsters: bool) -> None:
    """R4: the trainer's model and the playout's net agree to 1e-4 on real
    states, through the JSON the trainer writes."""
    states = _parity_states()
    keys = _keys(states)
    # The tables `gen` records and `train` writes: the engine defaults that
    # `rollout_features` reads.
    tables = rollout_learn.feature_tables() if version == 2 else {}
    monsters = sts2_rs.monster_species() if with_monsters else None
    net = _random_net(version, keys, monsters, 7)
    lp = sts2_rs.LearnedPlayout(
        json.dumps(
            net.to_json(
                keys,
                1.0,
                monster_act=tables.get("monster_act"),
                monster_keys=monsters,
                encounter_act=tables.get("encounter_act"),
            )
        )
    )
    checked = 0
    for s in states:
        ex = lp.explain(s)
        if len(ex["logits"]) < 2:
            continue
        np.testing.assert_allclose(_torch_logits(net, ex), ex["logits"], atol=1e-4, rtol=0)
        # `explain` reads the same features the training data records.
        feats = sts2_rs.rollout_features(s, version=version)
        np.testing.assert_allclose(ex["state"], feats["state"], atol=0)
        np.testing.assert_allclose(ex["x"], feats["x"], atol=0)
        assert ex["rows"] == [keys.index(k) for k in feats["keys"]]
        if with_monsters:
            assert ex["monsters"][0] > 0, "the encounter embedding sees the first enemy"
        checked += 1
    assert checked >= 4


def test_torch_matches_the_deployed_net_and_its_v2_upgrade() -> None:
    """The built-in v1 net loads into torch with the same logits, and
    `upgrade_net` turns it into a v2 file that plays the same policy."""
    v1 = json.loads(IRONCLAD_NET.read_text())
    net = rollout_learn._load_playout_net(torch, v1)
    lp1 = sts2_rs.LearnedPlayout(json.dumps(v1))
    species = sts2_rs.monster_species()
    up = rollout_learn.upgrade_net(v1, rollout_learn.feature_tables(), species)
    lp2 = sts2_rs.LearnedPlayout(json.dumps(up))
    assert lp2.feature_version == 2
    # Every species the engine knows gets a row, summons included (not just
    # the encounter rosters' species).
    assert up["monster_keys"] == species
    assert {"Wriggler", "GasBomb", "Parafright"} <= set(species) - set(sts2_rs.monster_acts())
    net2 = rollout_learn._load_playout_net(torch, up)
    for s in _parity_states():
        e1, e2 = lp1.explain(s), lp2.explain(s)
        np.testing.assert_allclose(_torch_logits(net, e1), e1["logits"], atol=1e-4, rtol=0)
        np.testing.assert_allclose(e2["logits"], e1["logits"], atol=1e-5, rtol=0)
        np.testing.assert_allclose(_torch_logits(net2, e2), e2["logits"], atol=1e-4, rtol=0)


def test_v3_relabels_the_deployed_net_and_reads_the_displayed_incoming() -> None:
    """GH #830: `upgrade_net(..., version=3)` turns the deployed v1 net into a
    v3 file with the same weights; torch matches Rust on it, its features are
    the training data's, and they are v1's except where a modifier changes a
    telegraphed attack (or an enemy has forms), which the parity states hit."""
    v1 = json.loads(IRONCLAD_NET.read_text())
    up = rollout_learn.upgrade_net(v1, {}, version=3)
    assert up["feature_version"] == 3
    assert {k: v for k, v in up.items() if k != "feature_version"} == {
        k: v for k, v in v1.items() if k != "feature_version"
    }
    lp1 = sts2_rs.LearnedPlayout(json.dumps(v1))
    lp3 = sts2_rs.LearnedPlayout(json.dumps(up))
    assert lp3.feature_version == 3
    assert (lp3.state_dim, lp3.action_dim) == (lp1.state_dim, lp1.action_dim)
    net3 = rollout_learn._load_playout_net(torch, up)
    same = differ = 0
    for s in _parity_states():
        e1, e3 = lp1.explain(s), lp3.explain(s)
        np.testing.assert_allclose(_torch_logits(net3, e3), e3["logits"], atol=1e-4, rtol=0)
        feats = sts2_rs.rollout_features(s, version=3)
        np.testing.assert_allclose(e3["state"], feats["state"], atol=0)
        np.testing.assert_allclose(e3["x"], feats["x"], atol=0)
        if e1["state"] == e3["state"] and e1["x"] == e3["x"]:
            assert e1["logits"] == e3["logits"]
            same += 1
        else:
            differ += 1
            # Only the incoming-derived and `removed` inputs may move.
            moved = {i for i, (a, b) in enumerate(zip(e1["state"], e3["state"])) if a != b}
            assert moved <= {10, 11, 12, 13, 15}, moved
    assert same > 0 and differ > 0, (same, differ)
    with pytest.raises(ValueError, match="into v3"):
        rollout_learn.upgrade_net(
            rollout_learn.upgrade_net(v1, rollout_learn.feature_tables()), {}, version=3
        )


def test_the_parity_states_exercise_most_v2_inputs() -> None:
    """Review F4: the parity tests above must see the v2 inputs they guard.
    `_states()` alone lit 7 of 64 v2 state columns and 2 of 8 action
    columns, so a torch/Rust mismatch on, e.g., the Demon Form column (73) or
    the exhaust flag (24) went unnoticed."""
    lit_s = np.zeros(64, bool)
    lit_a = np.zeros(8, bool)
    for s in _parity_states():
        f = sts2_rs.rollout_features(s, version=2)
        lit_s |= np.asarray(f["state"][32:]) != 0
        for x in f["x"]:
            lit_a |= np.asarray(x[24:]) != 0
    assert lit_s.sum() >= 40, np.flatnonzero(~lit_s) + 32
    assert lit_a.sum() >= 7, np.flatnonzero(~lit_a) + 24
    assert lit_s[73 - 32] and lit_a[0], "Demon Form and an exhausting card"


# ── R1: the search's own value ───────────────────────────────────────────


def _played_out(fight: dict[str, Any], seed: int) -> Any:
    rm = rollout_starts.build_fight(fight, seed)
    agent = sts2_rs.MctsCombatAgent(seed=seed, sims=16, boss_sims=16)
    rollout_starts.play_out(rm, agent, 3000)
    return rm


def test_search_value_is_in_the_backed_up_bands() -> None:
    rm = _played_out(HALL, 5)
    v = rollout_starts.final_value(rm)
    assert v is not None
    # `player_won` is the RUN's; the fight was lost only if the run is over.
    if not (rm.is_over() and not rm.player_won):
        assert 1.0 <= v <= 1.5
    else:
        assert -1.45 <= v < -1.05
    # A running fight has no final value.
    assert rollout_starts.final_value(rollout_starts.build_fight(HALL, 5)) is None

    lp = sts2_rs.LearnedPlayout(IRONCLAD_NET.read_text())
    states = _states()
    d = rollout_learn._unpack_playouts(
        lp.play_many(states, list(range(len(states))), temperature=1.0)
    )
    sv, won = d["search_value"], d["won"]
    assert np.all((sv[won] >= 1.0) & (sv[won] <= 1.5))
    assert np.all((sv[~won] >= -1.45) & (sv[~won] < -1.05))


# ── R2: trajectories and rebuilt starts ──────────────────────────────────

SPEC = agents.mcts_spec(16, 24, {})


def test_traj_records_rebuild_to_the_same_hash() -> None:
    rec = rollout_starts.traj_fight((ELITE, 0, 80, SPEC, 3000, 2, 0.5))
    assert rec["end"] in ("won", "died"), rec.get("error")
    assert rec["seed"] == rollout_starts.traj_seed(rollout_starts.fight_key(ELITE), 0)
    assert len(rec["actions"]) == len(rec["hashes"]) == len(rec["searched"])
    ts = [t for t, s in enumerate(rec["searched"]) if s]
    assert ts and ts[0] == 0, "the fight start is a searched decision"
    states = rollout_starts.rebuild(rec, ts)
    # The order-sensitive hash, not the order-free transposition key.
    assert [int(states[t].ordered_hash()) for t in ts] == [rec["hashes"][t] for t in ts]
    assert any(int(states[t].state_hash()) != rec["hashes"][t] for t in ts)
    assert rec["value"] is not None
    assert rec.get("branch"), "--branch-frac 0.5 picked some decisions"
    for t, vals in rec["branch"].items():
        assert all(-1.45 <= v <= 1.5 for v in vals)
        pv = rollout_starts.pilot_value(rec, int(t))
        assert pv == pytest.approx((rec["value"] + sum(vals)) / (1 + len(vals)))
    bad = json.loads(json.dumps(rec))
    bad["hashes"][ts[-1]] ^= 1
    with pytest.raises(RuntimeError, match="hash"):
        rollout_starts.rebuild(bad, ts)


def test_traj_cli_writes_resumes_and_holds_out(tmp_path: Path) -> None:
    fights = tmp_path / "fights.jsonl"
    fights.write_text("".join(json.dumps(f) + "\n" for f in (HALL, ELITE, BOSS)))
    held = tmp_path / "held.jsonl"
    held.write_text(json.dumps(BOSS) + "\n")
    out = tmp_path / "traj.jsonl"
    argv = [
        "traj",
        "--fights",
        str(fights),
        "--out",
        str(out),
        "--holdout",
        str(held),
        "--rooms",
        "monster,elite,boss",
        "--sims",
        "16",
        "--boss-sims",
        "24",
        "--procs",
        "2",
    ]
    assert rollout_learn.main(argv) == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows[0]["schema"] == rollout_starts.SCHEMA
    assert {r["file"] for r in rows[1:]} == {HALL["file"], ELITE["file"]}, "BOSS is held out"
    # A rerun finds nothing left to play.
    assert rollout_learn.main(argv) == 0
    assert len(out.read_text().splitlines()) == len(rows)


# ── R5 and the rl loop ───────────────────────────────────────────────────


def test_sb_coefficients() -> None:
    k = 4
    # Returns already at the target: nothing to balance.
    z = np.array([1.2, 1.2, 1.2, 1.2])
    c, st = rollout_learn.sb_coefficients(z, np.array([1.2]), k)
    assert np.allclose(c, 0.0) and st["n_sb"] == 1
    # Target above the playouts' mean: push up the playouts that did better
    # than their half-mates; no target: no weight.
    z = np.array([-1.2, 1.1, -1.3, 1.3, 0.0, 0.0, 0.0, 0.0])
    c, st = rollout_learn.sb_coefficients(z, np.array([1.4, np.nan]), k)
    assert c[1] > 0 > c[0] and c[3] > 0 > c[2]
    assert np.all(c[4:] == 0) and st["n_sb"] == 1
    assert st["sb_gap"] == pytest.approx(abs(1.4 - z[:4].mean()))
    with pytest.raises(SystemExit):
        rollout_starts.check_k(6 - 1, 0.5)
    w = rollout_starts.contested_weights(np.array([0.0, 0.5, 1.0]), 0.02)
    assert list(w) == [0.02, 0.25, 0.02]


def test_sb_weight_is_the_gap_times_a_unit_variance_advantage() -> None:
    """Review F1 (round 2): the SB coefficient is on REINFORCE's footing.
    It used to be a raw-unit sum that `rl` scaled by the decision count,
    so `--sb-beta` 0.5 outweighed REINFORCE ~15x on bosses. Now, with the
    target 0.5 above every start's value, it is exactly 0.5 x a
    unit-variance advantage, and it does not grow with the reward's scale
    beyond the gap itself."""
    rng = np.random.default_rng(0)
    n, k = 64, 8
    half = rng.normal(0.0, 0.8, (n, k // 2))
    z = np.concatenate([half, half[:, ::-1]], 1).reshape(-1)  # equal half means
    vstar = half.mean(1) + 0.5
    c, st = rollout_learn.sb_coefficients(z, vstar, k)
    assert c.std() == pytest.approx(0.5, rel=1e-4)
    assert st["sb_gap"] == pytest.approx(0.5) and st["n_sb"] == n
    c10, _ = rollout_learn.sb_coefficients(10 * z, 10 * vstar, k)
    assert np.allclose(c10, 10 * c, rtol=1e-5), "the gap is in value units, the advantage is not"


def _rl_argv(out: Path, fights: Path, *extra: str) -> list[str]:
    return [
        *["rl", "--init", str(IRONCLAD_NET), "--out", str(out), "--fights", str(fights)],
        *["--n-states", "3", "--batch-states", "2", "--k", "4", "--minibatch", "48"],
        *["--eval-every", "2", "--device", "cpu"],
        *extra,
    ]


def _metrics(out: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in Path(f"{out}.metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for r in rows:
        r.pop("time_s", None)
    return rows


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device"),
        ),
    ],
)
def test_rl_resume_reproduces_the_uninterrupted_run(tmp_path: Path, device: str) -> None:
    """Review F7 / F2 / F3 (round 2): a run interrupted after its checkpoint
    and resumed logs and learns what the uninterrupted run does: the
    contestedness refresh due at the checkpoint iteration happens, the
    minibatch order (a seeded CPU generator, so also on CUDA) is restored,
    and the rows the interrupted run wrote past its checkpoint are not
    logged twice."""
    fights = tmp_path / "fights.jsonl"
    fights.write_text("".join(json.dumps(f) + "\n" for f in (HALL, ELITE, BOSS)))
    evalf = tmp_path / "eval.jsonl"
    evalf.write_text(json.dumps(HALL) + "\n")
    common = ["--eval-fights", str(evalf), "--eval-reps", "1", "--ckpt-every", "2"]
    common += ["--contested-k", "2", "--contested-every", "2"]

    def argv(out: Path, *extra: str) -> list[str]:
        a = _rl_argv(out, fights, *common, *extra)
        a[a.index("--device") + 1] = device
        return a

    a, b = tmp_path / "straight.json", tmp_path / "resumed.json"
    assert rollout_learn.main(argv(a, "--iters", "4")) == 0
    # Interrupted after iteration 3 (checkpoint at 2), then resumed to 4.
    assert rollout_learn.main(argv(b, "--iters", "3")) == 0
    assert rollout_learn.main(argv(b, "--iters", "4", "--resume")) == 0
    ra, rb = _metrics(a), _metrics(b)
    assert [r["it"] for r in rb if "eval" not in r] == [1, 2, 3, 4]
    # CUDA's embedding backward may add in another order: equal to rounding.
    approx = (lambda v: v) if device == "cpu" else (lambda v: pytest.approx(v, rel=1e-5))
    assert ra == [{k: approx(v) for k, v in r.items()} for r in rb]
    na, nb = json.loads(a.read_text()), json.loads(b.read_text())
    for k in ("w1", "b1", "w2", "b2", "emb", "bias", "proj", "lin"):
        np.testing.assert_allclose(na[k], nb[k], rtol=0, atol=0 if device == "cpu" else 1e-5)


def test_rl_monster_emb_is_never_silently_ignored(tmp_path: Path) -> None:
    """Review F7c: `--monster-emb` on a net that stays v1 is refused, and a
    v2 `--init` without the embedding gets one (every species)."""
    fights = tmp_path / "fights.jsonl"
    fights.write_text("".join(json.dumps(f) + "\n" for f in (HALL, ELITE, BOSS)))
    out = tmp_path / "net.json"
    with pytest.raises(SystemExit, match="needs a v2 net"):
        rollout_learn.main(_rl_argv(out, fights, "--iters", "1", "--monster-emb"))
    v2 = tmp_path / "v2.json"
    plain = rollout_learn.upgrade_net(
        json.loads(IRONCLAD_NET.read_text()), rollout_learn.feature_tables()
    )
    assert "monster_keys" not in plain and plain["encounter_act"]
    v2.write_text(json.dumps(plain))
    argv = _rl_argv(out, fights, "--iters", "1", "--monster-emb")
    argv[argv.index("--init") + 1] = str(v2)
    assert rollout_learn.main(argv) == 0
    net = json.loads(out.read_text())
    assert net["monster_keys"] == sts2_rs.monster_species()
    assert net["encounter_act"] == plain["encounter_act"]


def _traj_file(tmp_path: Path) -> Path:
    out = tmp_path / "traj.jsonl"
    rows = [
        {"schema": rollout_starts.SCHEMA, "spec": SPEC, "engine": sts2_rs.src_hash()},
        rollout_starts.traj_fight((ELITE, 0, 80, SPEC, 3000, 0, 0.0)),
        rollout_starts.traj_fight((BOSS, 0, 80, SPEC, 3000, 0, 0.0)),
    ]
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return out


def test_start_ids_name_the_position_and_the_sb_cache_checks_what_it_searched(
    tmp_path: Path,
) -> None:
    """Review F1: two trajectories of one fight from another `--rep0` reach
    different positions under the same `(key, rep, t)`; their ids differ, and
    a cached `V*` is only reused for the same search spec / engine /
    estimator."""
    a = rollout_starts.traj_fight((ELITE, 0, 80, SPEC, 3000, 0, 0.0))
    b = rollout_starts.traj_fight((ELITE, 0, 85, SPEC, 3000, 0, 0.0))
    assert a["seed"] != b["seed"] and a["hashes"][0] != b["hashes"][0]
    sa, sb = rollout_starts.start_spec(a, 0), rollout_starts.start_spec(b, 0)
    assert sa["id"] != sb["id"]
    assert sa["id"] == rollout_starts.start_id(a["key"], a["seed"], 0, a["hashes"][0])

    cache = tmp_path / "sb.jsonl"
    # A row without the fingerprint (another search, engine or estimator).
    cache.write_text(json.dumps({"id": sa["id"], "sims": 16, "value": 9.0}) + "\n")
    t = rollout_starts.SearchTargets(str(cache), 16, 1, {})
    assert t.ignored == 1 and not t.values
    try:
        (v,) = t.get([sa])
    finally:
        t.close()
    assert v is not None and -1.45 <= v <= 1.5
    assert rollout_starts.SearchTargets(str(cache), 16, 1, {}).values == {sa["id"]: v}
    other = rollout_starts.SearchTargets(str(cache), 16, 1, {"STS2_RUNAI_LOSS_GRADIENT": "0"})
    assert not other.values and other.ignored == 2


def test_q_of_prices_a_given_move() -> None:
    """Review F5 (round 2): `V*` is the second search's Q of the first
    search's move, read with `q_of`."""
    rec = {
        "combat_search": {
            "actions": [
                {"action": {"action": "end_turn"}, "visits": 3, "q": 0.2},
                {"action": {"action": "play", "card_index": 0}, "visits": 5, "q": 0.4},
            ],
            "recommended": {"action": "play", "card_index": 0},
        }
    }
    assert rollout_starts.q_of(rec, {"action": "end_turn"}) == pytest.approx(0.2)
    assert rollout_starts.q_of(rec, {"card_index": 0, "action": "play"}) == pytest.approx(0.4)
    assert rollout_starts.q_of(rec, {"action": "potion", "slot": 0}) is None
    assert rollout_starts.q_of(None, {"action": "end_turn"}) is None


def test_value_env_mismatch_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Review F2 / F4 (round 2): `rl` scores playouts under its own explicit
    env (inherited seams cleared, the env recorded), and refuses
    trajectories whose values were scored under another value env."""
    hdr = [("t.jsonl", {"spec": {"env": {}}})]
    # The pilot's playout net is not part of the value function.
    assert not rollout_starts.check_value_env(hdr, {"STS2_ROLLOUT_NET": "/x.json"}, False)
    with pytest.raises(SystemExit, match="value env"):
        rollout_starts.check_value_env(
            hdr, {"STS2_RUNAI_MCTS_OVERRIDES": "loss_w_removed=0.9"}, False
        )
    assert rollout_starts.check_value_env(hdr, {"STS2_RUNAI_LOSS_GRADIENT": "0"}, True)

    fights = tmp_path / "fights.jsonl"
    fights.write_text("".join(json.dumps(f) + "\n" for f in (HALL, ELITE, BOSS)))
    out = tmp_path / "net.json"
    base = ["rl", "--init", str(IRONCLAD_NET), "--fights", str(fights), "--out", str(out)]
    base += ["--n-states", "2", "--batch-states", "2", "--k", "4", "--iters", "1"]
    base += ["--eval-every", "1", "--device", "cpu"]
    monkeypatch.setenv("STS2_RUNAI_TERMINAL_MODE", "binary")
    assert rollout_learn.main([*base, "--env", "STS2_RUNAI_LOSS_GRADIENT=0"]) == 0
    assert "STS2_RUNAI_TERMINAL_MODE" not in os.environ, "the inherited seam is cleared"
    assert os.environ["STS2_RUNAI_LOSS_GRADIENT"] == "0"
    meta = json.loads(out.read_text())["meta"]
    assert meta["engine_env"] == {"STS2_RUNAI_LOSS_GRADIENT": "0"}

    traj = _traj_file(tmp_path)  # recorded under the default env
    mid = ["--starts", "mid", str(traj), "--rooms", "elite,boss"]
    with pytest.raises(SystemExit, match="value env"):
        rollout_learn.main([*base, *mid, "--env", "STS2_RUNAI_LOSS_GRADIENT=0"])


def test_v2_act_lifts_the_construct_menagerie_to_act_3() -> None:
    """Review F6: two act-1 constructs alone are act 1, the act-3 Construct
    Menagerie (Punch + Cubex x2) is act 3; the species table alone said 1."""
    assert [3, ["PunchConstruct", "CubexConstruct", "CubexConstruct"]] in [
        [a, list(m)] for a, m in sts2_rs.encounter_acts()
    ]
    for enc, want in (
        ("CONSTRUCT_MENAGERIE_NORMAL", 3),
        ("PUNCH_CONSTRUCT_NORMAL", 1),
        ("CUBEX_CONSTRUCT_NORMAL", 1),
    ):
        cs = rollout_starts.build_fight(_row("x.run.gz", 1, "monster", enc), 7).combat_state()
        state = sts2_rs.rollout_features(cs, version=2)["state"]
        assert state[33] == pytest.approx(want / 3), enc
        alone = sts2_rs.rollout_features(cs, version=2, encounter_act=[])["state"]
        assert alone[33] == pytest.approx(1 / 3), enc


def test_rl_mid_starts_search_reward_sb_v2_and_resume(tmp_path: Path) -> None:
    traj = _traj_file(tmp_path)
    fights = tmp_path / "fights.jsonl"
    fights.write_text("".join(json.dumps(f) + "\n" for f in (HALL, ELITE, BOSS)))
    evalf = tmp_path / "eval.jsonl"
    evalf.write_text(json.dumps(HALL) + "\n")
    out = tmp_path / "net.json"
    argv = [
        "rl",
        "--init",
        str(IRONCLAD_NET),
        "--out",
        str(out),
        "--fights",
        str(fights),
        "--eval-fights",
        str(evalf),
        "--eval-reps",
        "1",
        "--eval-every",
        "2",
        "--starts",
        "mid",
        str(traj),
        "--mid-per-traj",
        "3",
        "--n-states",
        "6",
        "--batch-states",
        "3",
        "--k",
        "4",
        "--iters",
        "2",
        "--minibatch",
        "256",
        "--reward",
        "search",
        "--contested-k",
        "2",
        "--sb-beta",
        "0.5",
        "--sb-target",
        "pilot",
        "--feature-version",
        "2",
        "--monster-emb",
        "--device",
        "cpu",
    ]
    assert rollout_learn.main(argv) == 0
    net = json.loads(out.read_text())
    assert net["feature_version"] == 2 and net["monster_keys"]
    assert sts2_rs.LearnedPlayout(out.read_text()).state_dim == 96
    assert net["meta"]["starts"]["trajectories"] == 2
    metrics = [
        json.loads(line)
        for line in Path(f"{out}.metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    its = [m for m in metrics if "eval" not in m]
    assert [m["it"] for m in its] == [1, 2]
    assert all(m["n_sb"] > 0 and np.isfinite(m["sb_gap"]) for m in its)
    # Both terms' gradient norms are logged, to read --sb-beta against.
    assert all(m["g_rf"] > 0 and m["g_sb"] >= 0 and "sb_ratio" in m for m in its)
    assert {m["view"] for m in metrics if "eval" in m} == {"raw", "belief"}
    assert Path(f"{out}.ckpt.pt").exists()
    # Resume: picks up at iteration 2 and only plays iteration 3.
    i = argv.index("--iters")
    argv[i + 1] = "3"
    assert rollout_learn.main([*argv, "--resume"]) == 0
    metrics = [
        json.loads(line)
        for line in Path(f"{out}.metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [m["it"] for m in metrics if "eval" not in m] == [1, 2, 3]


def test_rl_default_fight_starts_still_run(tmp_path: Path) -> None:
    """The original trainer (fight starts, shaped reward, v1) is unchanged."""
    fights = tmp_path / "fights.jsonl"
    fights.write_text("".join(json.dumps(f) + "\n" for f in (HALL, ELITE, BOSS)))
    out = tmp_path / "net.json"
    argv = [
        "rl",
        "--init",
        str(IRONCLAD_NET),
        "--out",
        str(out),
        "--fights",
        str(fights),
        "--n-states",
        "3",
        "--batch-states",
        "2",
        "--k",
        "4",
        "--iters",
        "1",
        "--eval-every",
        "1",
        "--device",
        "cpu",
    ]
    assert rollout_learn.main(argv) == 0
    net = json.loads(out.read_text())
    assert net["feature_version"] == 1 and "monster_act" not in net
    assert net["meta"]["rl"]["reward"] == "shaped"


def test_rl_upgrades_a_v1_init_to_v3(tmp_path: Path) -> None:
    """`rl --feature-version 3` trains a relabeled v1 net end to end and
    writes a v3 file the engine loads; a v3 --init cannot become v2."""
    fights = tmp_path / "fights.jsonl"
    fights.write_text("".join(json.dumps(f) + "\n" for f in (HALL, ELITE, BOSS)))
    out = tmp_path / "net.json"
    argv = _rl_argv(out, fights, "--iters", "1", "--feature-version", "3")
    assert rollout_learn.main(argv) == 0
    net = json.loads(out.read_text())
    assert net["feature_version"] == 3 and "monster_act" not in net
    assert sts2_rs.LearnedPlayout(out.read_text()).feature_version == 3
    argv = _rl_argv(tmp_path / "net2.json", fights, "--iters", "1", "--feature-version", "2")
    argv[argv.index("--init") + 1] = str(out)
    with pytest.raises(SystemExit, match="only a v1 net upgrades"):
        rollout_learn.main(argv)


def test_gen_records_v3_features(tmp_path: Path) -> None:
    rows = rollout_learn.gen_fight((ELITE, 0, 1, 50, 16, 24, False, 3, 0))
    assert rows["decisions"] and "error" not in rows
    assert all(len(d["state"]) == 32 and "monsters" not in d for d in rows["decisions"])


def test_gen_stops_a_fight_past_its_timeout_and_keeps_its_decisions() -> None:
    full = rollout_learn.gen_fight((ELITE, 0, 2, 50, 16, 24, False, 1, 0))
    cut = rollout_learn.gen_fight((ELITE, 0, 2, 50, 16, 24, False, 1, 1e-9))
    assert "timeout" not in full and "error" not in full
    assert cut["timeout"] and "error" not in cut
    assert len(cut["decisions"]) <= 1 < len(full["decisions"])
    assert cut["decisions"] == full["decisions"][: len(cut["decisions"])]


def test_gen_and_train_v2_with_the_encounter_embedding(tmp_path: Path) -> None:
    w = rollout_learn.ShardWriter(str(tmp_path / "gen"), 10_000)
    for i, f in enumerate((ELITE, HALL)):
        rows = rollout_learn.gen_fight((f, i, 1, 50, 16, 24, False, 2, 0))
        assert rows["decisions"] and "error" not in rows
        assert all(len(d["state"]) == 96 and d["monsters"][0] for d in rows["decisions"])
        w.add(rows["decisions"])
    w.flush()
    (tmp_path / "gen" / rollout_learn.FEATURES_META).write_text(
        json.dumps({"feature_version": 2, "monster_act": sts2_rs.monster_acts()})
    )
    out = tmp_path / "net.json"
    argv = ["train", "--data", str(tmp_path / "gen"), "--out", str(out), "--epochs", "1"]
    argv += ["--monster-emb", "--device", "cpu", "--val-frac", "0.5"]
    assert rollout_learn.main(argv) == 0
    lp = sts2_rs.LearnedPlayout(out.read_text())
    assert lp.feature_version == 2
    net = json.loads(out.read_text())
    assert len(net["monster_keys"]) == len(net["monster_emb"]) >= 1


REGENT_DECK = [
    *["STRIKE_REGENT", "DEFEND_REGENT"] * 3,
    *["FALLING_STAR", "GLITTERSTREAM", "GAMMA_BLAST", "VENERATE", "GATHER_LIGHT"],
    *["CHILD_OF_THE_STARS", "BLACK_HOLE", "FURNACE", "GENESIS", "SWORD_SAGE"],
]


def _regent_states() -> list[Any]:
    """Mid-fight Regent positions against the Tainted (Infested Prisms) and
    Personal Hive (Entomancer) elites, random play biased toward cards."""
    out = []
    for enc, seed in (("INFESTED_PRISMS_ELITE", 7), ("ENTOMANCER_ELITE", 8)):
        fight = _row(
            "cccc000000000001.run.gz",
            9,
            "elite",
            enc,
            character="Regent",
            deck=REGENT_DECK,
            relics=["DIVINE_RIGHT"],
            potions=[],
            hp_before=75,
        )
        cs = rollout_starts.build_fight(fight, seed).combat_state()
        rng = random.Random(seed)
        for i in range(120):
            acts = cs.legal_actions()
            if not acts:
                break
            if i % 2 == 0:
                out.append(cs.clone())
            cards = [a for a in acts if a.get("action") == "play"]
            cs.apply(rng.choice(cards) if cards and rng.random() < 0.85 else rng.choice(acts))
    return out


def test_v4_zero_pads_a_v1_net_and_reads_the_character_block() -> None:
    """GH #822 follow-up: `upgrade_net(..., version=4)` zero-pads the shipped
    Regent net, so its logits are unchanged and torch matches Rust; v4's first
    32 / 24 features are v1's, and the block past them lights Stars, Star
    cost, Tainted and Personal Hive on Regent's elite fights."""
    v1 = json.loads(REGENT_NET.read_text())
    up = rollout_learn.upgrade_net(v1, {}, version=4)
    assert up["feature_version"] == 4
    assert (up["state_dim"], up["action_dim"]) == rollout_learn.FEATURE_DIMS[4]
    lp1 = sts2_rs.LearnedPlayout(json.dumps(v1))
    lp4 = sts2_rs.LearnedPlayout(json.dumps(up))
    net4 = rollout_learn._load_playout_net(torch, up)
    states = _regent_states()
    seen = np.zeros(44 + 29, dtype=bool)
    for s in states:
        e1, e4 = lp1.explain(s), lp4.explain(s)
        np.testing.assert_allclose(e4["logits"], e1["logits"], atol=1e-5, rtol=0)
        np.testing.assert_allclose(_torch_logits(net4, e4), e4["logits"], atol=1e-4, rtol=0)
        f1 = sts2_rs.rollout_features(s, version=1)
        f4 = sts2_rs.rollout_features(s, version=4)
        assert f4["state"][:32] == f1["state"]
        assert [x[:24] for x in f4["x"]] == f1["x"]
        seen[:44] |= np.asarray(f4["state"]) != 0
        for x in f4["x"]:
            seen[44:] |= np.asarray(x) != 0
    # Stars, Tainted, Personal Hive, Vital Spark; Star cost, Tainted card,
    # an Attack into the Hive.
    for i in (32, 40, 41, 42, 44 + 24, 44 + 26, 44 + 28):
        assert seen[i], i


def test_train_init_starts_exactly_at_the_init_net(tmp_path: Path) -> None:
    """GH #830 playout distillation: `train --init` with no epochs writes the
    init net's policy back unchanged (the data's new keys appended at zero),
    and `--kl-init` trains from it."""
    w = rollout_learn.ShardWriter(str(tmp_path / "gen"), 10_000)
    for i, f in enumerate((ELITE, HALL)):
        rows = rollout_learn.gen_fight((f, i, 1, 50, 16, 24, False, 1, 0))
        assert rows["decisions"] and "error" not in rows
        w.add(rows["decisions"])
    w.flush()
    init = (
        Path(__file__).resolve().parents[2] / "sts2-rs" / "data" / "rollout_nets" / "ironclad.json"
    )
    base = ["train", "--data", str(tmp_path / "gen"), "--init", str(init), "--device", "cpu"]
    base += ["--val-frac", "0.5"]
    e0, e1 = tmp_path / "e0.json", tmp_path / "e1.json"
    assert rollout_learn.main([*base, "--out", str(e0), "--epochs", "0"]) == 0
    assert rollout_learn.main([*base, "--out", str(e1), "--epochs", "1", "--kl-init", "0.5"]) == 0
    ref, got = json.loads(init.read_text()), json.loads(e0.read_text())
    assert got["keys"][: len(ref["keys"])] == ref["keys"]
    assert np.allclose(got["emb"][: len(ref["emb"])], ref["emb"], atol=1e-6)
    assert all(not any(r) for r in got["emb"][len(ref["emb"]) :])
    st = rollout_starts.build_fight(ELITE, 5).combat_state()
    sts_ref = sts2_rs.LearnedPlayout(init.read_text())
    sts_e0 = sts2_rs.LearnedPlayout(e0.read_text())
    a = rollout_learn._unpack_playouts(sts_ref.play_many([st] * 4, [1, 2, 3, 4], temperature=1.0))
    b = rollout_learn._unpack_playouts(sts_e0.play_many([st] * 4, [1, 2, 3, 4], temperature=1.0))
    assert (a["chosen"] == b["chosen"]).all() and (a["won"] == b["won"]).all()
    assert json.loads(e1.read_text())["meta"]["kl_init"] == pytest.approx(0.5)


def test_gen_replay_records_the_pilot_from_a_replayed_turn(tmp_path: Path) -> None:
    """`gen-replay` replays a SpireCodex fight to an in-sync turn start and
    records the pilot's decisions from there (mode pilot) or after the
    human's forced turn (mode human)."""
    from ppo.bc_human import replay
    from ppo.turnplan.pro import fight_turns

    root = Path(replay.scraper_dir("replays-spire-codex")) / "ironclad"
    files = sorted(root.glob("*/*.replay.ndjson.gz"))[:40]
    for path in files:
        for rec in replay.fights(str(path)):
            if rec["result"] not in replay.FINISHED or not rec["rng_state"]:
                continue
            try:
                turns, _ = fight_turns(rec)
            except Exception:
                continue
            legal = [t for t, x in enumerate(turns) if x["status"] == "legal"]
            if len(legal) < 2:
                continue
            job = (
                str(path),
                rec["combat_id"],
                [(legal[0], "pilot"), (legal[0], "human")],
                16,
                16,
                1,
                {},
            )
            out = rollout_learn.gen_replay_fight(job)
            assert "error" not in out and out["decisions"]
            assert len(out["keys"]) == 2 and out["skipped"] <= 1
            assert all(len(d["state"]) == 32 for d in out["decisions"])
            return
    pytest.skip("no replay with two in-sync turns found")
