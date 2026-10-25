"""GH #918: a replay rebuild rolls the encounter's spawn args like the game.

`bc_human.sync.build` rebuilds a recorded fight from the run seed, the stream
counters at combat start and a kind-only roster. Decimillipede's opener
rotation is drawn on the per-encounter `Rng(Seed + TotalFloor + hash(id))`,
which no stream counter carries, so the rebuild needs the fight's floor.
Without it every rebuild opened on one rotation, and the human's targeted
hits, Weak and Thorns landed on segments holding other moves than in the game.

Checked against the v0.111.0 C# oracle entering the same encounter on the
same run.
"""

from __future__ import annotations

import os

import pytest
import sts2_client

sts2_rs = pytest.importorskip("sts2_rs")

from ppo.bc_human import sync  # noqa: E402

ENCOUNTER = "DECIMILLIPEDE_ELITE"
DECK = ["STRIKE_IRONCLAD"] * 5 + ["DEFEND_IRONCLAD"] * 4 + ["BASH"]
# Already canonical (no I / O), as a replay header's seed is.
SEEDS = [f"DEC{i:04d}" for i in range(6)]


@pytest.fixture
def restore_version():
    before = sts2_client.GAME_VERSION
    env = os.environ.get(sts2_client.GAME_VERSION_ENV)
    yield
    sts2_client.set_game_version(before)
    if env is None:
        os.environ.pop(sts2_client.GAME_VERSION_ENV, None)
    else:
        os.environ[sts2_client.GAME_VERSION_ENV] = env


def _oracle_fight(seed: str) -> tuple[dict, list[tuple[int, str]]]:
    """A replay-shaped record of the oracle's fight, and its (hp, opener) per monster."""
    g = sts2_client.Game()
    try:
        st = g.start(character="Ironclad", seed=seed, ascension=10)
        g.skip_neow(st)
        g.set_player(hp=80, max_hp=80, deck=DECK, relics=[], potions=[])
        counters = dict(g.get_state()["rng_counters"])
        floor = g.enter_room("combat", encounter=ENCOUNTER)["context"]["total_floor"]
        monsters = g.get_state()["monsters"]
    finally:
        g.close()
    # A replay logs the counters after the opening shuffle and the HP rolls,
    # which `sync.start_counters` rewinds.
    counters["shuffle"] += len(DECK) - 1
    counters["niche"] += len(monsters)
    rec = {
        "seed": seed,
        "character": "IRONCLAD",
        "ascension": 10,
        "encounter": ENCOUNTER,
        "enemies": [{"id": m["kind"]} for m in monsters],
        "rng_state": counters,
        "floor": floor,
        "deck": DECK,
        "relics": [],
        "belt": [],
        "hp": 80,
        "max_hp": 80,
    }
    return rec, [(m["hp"], m["move_state"].removesuffix("_MOVE")) for m in monsters]


def _rebuilt(rec: dict) -> list[tuple[int, str]]:
    enemies = sts2_rs.state_to_dict(sync.build(rec))["encounter"]
    return [(e["hp"], e["move_state"]) for e in enemies]


def test_replay_rebuild_rolls_decimillipede_openers(restore_version) -> None:
    ver = sync.GAME_VERSION
    if not sts2_client.oracle_available(ver) or not sts2_client.oracle_built(ver):
        pytest.skip(f"C# oracle for {ver} not available/built (`just build-cs {ver}`)")
    sts2_client.set_game_version(ver)
    floorless_misses = 0
    for seed in SEEDS:
        rec, want = _oracle_fight(seed)
        assert _rebuilt(rec) == want, seed
        floorless_misses += _rebuilt({**rec, "floor": None}) != want
    # The floorless rebuild pins one rotation, so some seed must disagree.
    assert floorless_misses > 0
