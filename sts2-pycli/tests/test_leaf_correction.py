"""GH #830 leaf correction: label generation and the residual model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from ppo import leaf_correction

sts2_rs = pytest.importorskip("sts2_rs")


def test_play_and_label_rebuilds_the_final_value() -> None:
    from ppo.combat_bench.run import build_run, load_set

    fight = next(
        f
        for f in load_set(
            str(Path(__file__).resolve().parents[1] / "data/combat_bench/ironclad_devpool_v1.jsonl")
        )
        if f["encounter"].endswith("_WEAK")
    )
    rm = build_run(fight, 3)
    agent = sts2_rs.MctsCombatAgent(seed=3, sims=20, boss_sims=20)
    rows: list = []
    out = leaf_correction.play_and_label(rm, agent, 2, 3, rows, {"run": "r", "start": "s"})
    assert rows and out["vt"] is not None
    # A win backs up in [1.0, 1.5], a loss in [-1.45, -1.05).
    assert (1.0 <= out["vt"] <= 1.5) if out["won"] else (-1.45 <= out["vt"] < -1.05)
    assert all(len(r["x"]) == leaf_correction.STATE_DIM and r["vt"] == out["vt"] for r in rows)


def test_train_writes_a_clipped_residual(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    rows = tmp_path / "rows.jsonl"
    with rows.open("w", encoding="utf-8") as fh:
        for i in range(400):
            x = rng.random(leaf_correction.STATE_DIM).tolist()
            fh.write(
                json.dumps(
                    {
                        "run": f"r{i % 20}",
                        "start": f"s{i // 4}",
                        "round": i % 4,
                        "x": x,
                        "pm": 1.0,
                        "vt": 1.0 + 0.5 * x[0],
                    }
                )
                + "\n"
            )
    out = tmp_path / "r.json"
    argv = ["train", "--rows", str(rows), "--out", str(out), "--epochs", "30", "--device", "cpu"]
    argv += ["--clip", "0.2", "--batch", "64", "--lr", "3e-3"]
    assert leaf_correction.main(argv) == 0
    net = json.loads(out.read_text(encoding="utf-8"))
    x = rng.random((50, leaf_correction.STATE_DIM))
    r = leaf_correction.forward(net, x)
    assert r.shape == (50,) and np.abs(r).max() <= 0.2 + 1e-9
    assert np.corrcoef(r, x[:, 0])[0, 1] > 0.5  # it learns the planted residual


def test_train_reads_extra_input_blocks_and_their_masks(tmp_path: Path) -> None:
    """`--inputs x,K --extra f.npz`: row-aligned blocks join the rl2 features,
    and a block's `<b>_ok` mask drops the rows it could not compute."""
    pytest.importorskip("torch")
    rng = np.random.default_rng(1)
    rows = tmp_path / "rows.jsonl"
    n = 300
    K = rng.random((n, 3)).astype(np.float32)
    with rows.open("w", encoding="utf-8") as fh:
        for i in range(n):
            x = rng.random(leaf_correction.STATE_DIM).tolist()
            r = {"run": f"r{i % 15}", "start": f"s{i // 3}", "round": i % 3, "x": x, "pm": 1.0}
            fh.write(json.dumps({**r, "vt": 1.0 + 0.5 * float(K[i, 0])}) + "\n")
    ok = np.ones(n, bool)
    ok[:50] = False
    np.savez(tmp_path / "f.npz", K=K, K_ok=ok)
    out = tmp_path / "r.json"
    argv = ["train", "--rows", str(rows), "--out", str(out), "--epochs", "30", "--device", "cpu"]
    argv += ["--inputs", "x,K", "--extra", str(tmp_path / "f.npz"), "--batch", "64", "--lr", "3e-3"]
    assert leaf_correction.main(argv) == 0
    net = json.loads(out.read_text(encoding="utf-8"))
    assert net["inputs"] == ["x", "K"] and net["state_dim"] == leaf_correction.STATE_DIM + 3
    assert net["meta"]["n"] == n - 50
    X = np.concatenate([rng.random((40, leaf_correction.STATE_DIM)), rng.random((40, 3))], 1)
    assert np.corrcoef(leaf_correction.forward(net, X), X[:, leaf_correction.STATE_DIM])[0, 1] > 0.5


def test_train_pairs_learns_the_within_position_contrast(tmp_path: Path) -> None:
    """`train-pairs`: r(a) - r(b) tracks (vt_a - vt_b) - (pm_a - pm_b) for a
    planted contrast that only shows within a position."""
    pytest.importorskip("torch")
    rng = np.random.default_rng(2)
    path = tmp_path / "pairs.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for i in range(300):
            level = rng.normal()  # position-level offset: must not matter
            alts = []
            for kind in ("human", "pilot", "max_block"):
                x = rng.random(leaf_correction.STATE_DIM).tolist()
                alts.append(
                    {"kind": kind, "x": x, "pm": [1.0] * 8, "vt": [1.0 + level + 0.3 * x[0]]}
                )
            fh.write(
                json.dumps({"file": f"f{i % 30}", "combat_id": "c", "turn": i, "alts": alts}) + "\n"
            )
    out = tmp_path / "r.json"
    argv = [
        "train-pairs",
        "--pairs",
        str(path),
        "--out",
        str(out),
        "--epochs",
        "40",
        "--device",
        "cpu",
    ]
    argv += ["--batch", "64", "--lr", "3e-3", "--anchor", "0.0", "--wd", "0.0"]
    assert leaf_correction.main(argv) == 0
    net = json.loads(out.read_text(encoding="utf-8"))
    assert (
        net["meta"]["objective"] == "pairwise"
        and net["meta"]["val_mse"] < net["meta"]["zero_val_mse"]
    )
    X = rng.random((60, leaf_correction.STATE_DIM))
    assert np.corrcoef(leaf_correction.forward(net, X), X[:, 0])[0, 1] > 0.5
