"""Pure-python unit gates for the M-P3 PPO trainer (no engine, no torch net).

The load-bearing invariant: PBRS shaping with an un-credited terminal must
telescope — sum(shaped rewards) == base(terminal) − Φ(s0) exactly — so the
optimal policy is provably unchanged (plan §3.3).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from ppo.rewards import (
    ShapingConfig,
    base_value,
    shape_value,
    shaped_rewards,
    total_floors,
)


def _metrics(floor, hp=50, max_hp=80, upg=0, relics=1):
    return {
        "total_floor": floor,
        "hp": hp,
        "max_hp": max_hp,
        "num_upgraded": upg,
        "num_relics": relics,
        "gold": 100,
        "act_index": 1,
    }


@pytest.mark.parametrize("won", [False, True])
def test_shaped_rewards_telescope_exactly(won):
    cfg = ShapingConfig(hp_coef=0.1, upg_coef=0.035, relic_coef=0.02, offset=0.307)
    steps = [
        _metrics(1, hp=80, upg=0, relics=1),
        _metrics(5, hp=60, upg=1, relics=2),
        _metrics(9, hp=30, upg=2, relics=2),
        _metrics(16, hp=45, upg=4, relics=5),
    ]
    final = _metrics(23, hp=10, upg=4, relics=5)
    rewards, final_base = shaped_rewards(steps, final, won, cfg)
    assert len(rewards) == len(steps)
    # Telescoping identity: total return = base(terminal) - (base+shape)(s0).
    expected = final_base - (base_value(steps[0], False, cfg) + shape_value(steps[0], cfg))
    assert abs(sum(rewards) - expected) < 1e-12
    assert final_base == base_value(final, won, cfg)
    # Win adds exactly the win bonus to the terminal.
    if won:
        assert final_base == pytest.approx(base_value(final, False, cfg) + cfg.win_bonus)


def test_deeper_death_outscores_shallower():
    cfg = ShapingConfig()
    shallow = base_value(_metrics(8), False, cfg)
    deep = base_value(_metrics(30), False, cfg)
    win = base_value(_metrics(45), True, cfg)
    assert shallow < deep < win


# ── Variable run length: A10 DoubleBoss + Golden Compass (GH #464) ───────────


def _asc(metrics: dict, level: int) -> dict:
    return {**metrics, "ascension": level}


@pytest.mark.parametrize("level", [0, 1, 5, 9])
def test_normalizer_is_45_below_a10(level):
    assert total_floors(_asc(_metrics(45), level)) == 45
    assert base_value(_asc(_metrics(45), level), False, ShapingConfig()) == pytest.approx(0.5)


def test_a10_normalizer_is_46_because_of_the_second_boss():
    cfg = ShapingConfig()
    assert total_floors(_asc(_metrics(45), 10)) == 46
    # Floor 45 at A10 is the FIRST boss; the run is not over, so progress must
    # not read as complete yet.
    assert base_value(_asc(_metrics(45), 10), False, cfg) < 0.5
    assert base_value(_asc(_metrics(46), 10), False, cfg) == pytest.approx(0.5)


def test_a10_deeper_death_strictly_outscores():
    """Dying to the second boss must beat dying to the first."""
    cfg = ShapingConfig()
    first = base_value(_asc(_metrics(45), 10), False, cfg)
    second = base_value(_asc(_metrics(46), 10), False, cfg)
    assert first < second < base_value(_asc(_metrics(46), 10), True, cfg)


@pytest.mark.parametrize("level", [0, 10])
def test_progress_is_not_clipped_past_the_normalizer(level):
    """Golden Compass makes act 2 two rows longer; those floors must still count.

    A `min(1.0, ...)` clip flattened the reward at the end of exactly the runs
    that got furthest — measured at 3.9% of runs reaching act 2.
    """
    cfg = ShapingConfig()
    vals = [base_value(_asc(_metrics(f), level), False, cfg) for f in range(44, 49)]
    assert vals == sorted(vals) and len(set(vals)) == len(vals), vals


def test_missing_ascension_key_falls_back_to_a0():
    """Corpora predating the key still parse — as A0, which is what they were."""
    assert "ascension" not in _metrics(45)
    assert total_floors(_metrics(45)) == 45


def test_a_completed_run_still_scores_about_one():
    """Dropping the clip must not blow up the reward scale."""
    cfg = ShapingConfig()
    assert base_value(_asc(_metrics(45), 0), True, cfg) == pytest.approx(1.0)
    assert base_value(_asc(_metrics(46), 10), True, cfg) == pytest.approx(1.0)
    # Longest plausible run (A10 + Golden Compass) stays within a few percent.
    assert base_value(_asc(_metrics(48), 10), True, cfg) == pytest.approx(1.022, abs=0.005)


def test_telescoping_still_holds_at_a10():
    """The identity the whole reward rests on must survive the new cap."""
    cfg = ShapingConfig()
    steps = [_asc(_metrics(f, hp=80 - f, upg=f // 8), 10) for f in (1, 12, 30, 44, 45)]
    final = _asc(_metrics(46, hp=12, upg=6), 10)
    rewards, final_base = shaped_rewards(steps, final, True, cfg)
    expected = final_base - (base_value(steps[0], False, cfg) + shape_value(steps[0], cfg))
    assert abs(sum(rewards) - expected) < 1e-12


def test_gae_matches_bruteforce():
    from ppo.trainer import PPOConfig, gae_from_row

    cfg = PPOConfig(gamma=1.0, lam=0.97)
    values = [0.3, 0.5, 0.2]
    row = {
        "won": False,
        "final": _metrics(12, hp=0),
        "transitions": [
            {
                "feat": None,
                "action_idx": 0,
                "old_logprob": -1.0,
                "value": v,
                "metrics": m,
                "surface": "path",
                "n_actions": 2,
            }
            for v, m in zip(values, [_metrics(1), _metrics(4), _metrics(9)])
        ],
    }
    from ppo.rewards import ShapingConfig as SC

    shaping = SC()
    trans = gae_from_row(row, cfg, shaping)
    rewards, _ = shaped_rewards(
        [t["metrics"] for t in row["transitions"]], row["final"], False, shaping
    )
    # Brute-force GAE with terminal bootstrap 0.
    n = len(values)
    deltas = [rewards[t] + (values[t + 1] if t + 1 < n else 0.0) - values[t] for t in range(n)]
    expected = [0.0] * n
    acc = 0.0
    for t in reversed(range(n)):
        acc = deltas[t] + cfg.gamma * cfg.lam * acc
        expected[t] = acc
    got = [t.advantage for t in trans]
    assert np.allclose(got, expected)
    assert np.allclose([t.ret for t in trans], [e + v for e, v in zip(expected, values)])


def test_entropy_schedule_geometric_and_anchored():
    from ppo.trainer import PPOConfig, effective_entropy_coef

    cfg = PPOConfig(entropy_coef=0.05, entropy_coef_final=0.0025, entropy_decay_steps=200)
    # Not engaged by default (decay_start far in the future).
    assert effective_entropy_coef(cfg, 500) == pytest.approx(0.05)
    cfg2 = PPOConfig(
        entropy_coef=0.05,
        entropy_coef_final=0.0025,
        entropy_decay_steps=200,
        entropy_decay_start=100,
    )
    assert effective_entropy_coef(cfg2, 99) == pytest.approx(0.05)
    assert effective_entropy_coef(cfg2, 100) == pytest.approx(0.05)
    assert effective_entropy_coef(cfg2, 200) == pytest.approx(0.05 * (0.0025 / 0.05) ** 0.5)
    assert effective_entropy_coef(cfg2, 300) == pytest.approx(0.0025)
    assert effective_entropy_coef(cfg2, 999) == pytest.approx(0.0025)


def test_running_moments_batch_invariance():
    from ppo.trainer import RunningMoments

    xs = np.random.default_rng(0).normal(2.0, 3.0, size=4096)
    a = RunningMoments(5e-4)
    a.update(xs)
    b = RunningMoments(5e-4)
    for chunk in np.split(xs, 8):
        b.update(chunk)
    # Chunked vs single-shot: same EWMA state up to fp noise, and both near
    # the sample stats at this decay·N.
    assert a.mean == pytest.approx(b.mean, rel=0.05)
    assert a.std == pytest.approx(b.std, rel=0.05)
    assert a.std > 1.0


def test_a10_runs_also_report_an_a9_estimate():
    from ppo.trainer import per_ascension_winrate

    rows = [
        {"ascension": 10, "won": True, "reached_second_boss": True},
        {"ascension": 10, "won": False, "reached_second_boss": True},
        {"ascension": 10, "won": False, "reached_second_boss": False},
        {"ascension": 10, "won": False, "reached_second_boss": None},  # died before act 3
        {"ascension": 9, "won": True, "reached_second_boss": None},
    ]
    out = per_ascension_winrate(rows)
    assert out["win_rate/A10"] == pytest.approx(0.25)
    assert out["win_rate/A9_from_A10"] == pytest.approx(
        0.5
    )  # A10 runs only; the true A9 row is not mixed in
    assert out["win_rate/A9"] == pytest.approx(1.0)
    assert "win_rate/A9_from_A10" not in per_ascension_winrate(rows[4:])


def test_parse_ascension_weights():
    from ppo.trainer import parse_ascension_weights

    assert parse_ascension_weights("") is None
    assert parse_ascension_weights("   ") is None
    ascs, probs = parse_ascension_weights("0:1,9:3")
    assert ascs == [0, 9]
    assert probs == pytest.approx([0.25, 0.75])
    # bare entry defaults to weight 1; trailing comma tolerated.
    ascs2, probs2 = parse_ascension_weights("5,6,7,")
    assert ascs2 == [5, 6, 7]
    assert probs2 == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    with pytest.raises(ValueError):
        parse_ascension_weights("3:0")  # zero total weight
    with pytest.raises(ValueError):
        parse_ascension_weights("-1:1")  # negative ascension


def test_sample_ascensions_distribution_and_determinism():
    from ppo.trainer import PPOConfig, sample_ascensions

    base = PPOConfig(ascension_weights="")
    assert sample_ascensions(base, 0, 100) is None  # no curriculum

    cfg = PPOConfig(ascension_weights="0:1,9:4,10:1", seed_base=10_000_000)
    a = sample_ascensions(cfg, 3, 5000)
    b = sample_ascensions(cfg, 3, 5000)
    assert a == b  # deterministic per iteration (resume-safe)
    assert sample_ascensions(cfg, 4, 5000) != a  # different iteration → different draw
    assert set(a) == {0, 9, 10}
    frac9 = a.count(9) / len(a)
    assert 0.60 < frac9 < 0.72  # ~4/6 mass on A9


def test_high_asc_sims_lists_layers_screen_budget_under_the_high_asc_override():
    """The screen's global `--sims` and the high-ascension boost are two
    independent layers, and a default config must stay on the (None, None)
    path so a campaign run is byte-identical to the pre-flag behavior."""
    from ppo.trainer import PPOConfig, high_asc_sims_lists

    asc = [0, 4, 9]

    # Neither layer engaged → the env's own eval-cheap default, signalled by
    # (None, None) so `play_batch` never even builds a per-seed list.
    assert high_asc_sims_lists(PPOConfig(), asc) == (None, None)

    # Screen budget only → uniform across ascensions.
    cfg = PPOConfig(sims=100, boss_sims=300)
    assert high_asc_sims_lists(cfg, asc) == ([100, 100, 100], [300, 300, 300])

    # High-ascension boost only → boosted at/above the threshold, env default
    # (None) below it.
    cfg = PPOConfig(high_asc_min=9, high_asc_sims=2000, high_asc_boss_sims=8000)
    assert high_asc_sims_lists(cfg, asc) == ([None, None, 2000], [None, None, 8000])

    # Both → the boost wins at/above the threshold, the screen budget below.
    cfg = PPOConfig(sims=100, boss_sims=300, high_asc_min=9)
    sims, boss = high_asc_sims_lists(cfg, asc)
    assert sims == [100, 100, cfg.high_asc_sims]
    assert boss == [300, 300, cfg.high_asc_boss_sims]


def test_resolve_device_refuses_a_silent_cpu_fallback():
    """A screen's budget assumes the update is off the critical path. Asking
    for CUDA on a box without it must fail loudly, not run 30x slower with an
    identical-looking log."""
    import torch
    from ppo.trainer import resolve_device

    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device("auto").type in ("cpu", "cuda")
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="cuda"):
            resolve_device("cuda")


def test_train_rejects_surrogate_combat_without_a_net():
    """`--combat surrogate` with no `--surrogate-path` would otherwise be
    discovered one worker-spawn later, after the pool is up."""
    from ppo.trainer import PPOConfig, train

    with pytest.raises(ValueError, match="surrogate-path"):
        train(PPOConfig(combat="surrogate", iterations=1))
    with pytest.raises(ValueError, match="mcts"):
        train(PPOConfig(combat="heuristic", iterations=1))


def test_apply_lr_schedule_linear():
    from ppo.model import RunPolicyNet
    from ppo.trainer import PPOConfig, apply_lr_schedule, make_optimizer

    net = RunPolicyNet(dim=32, n_layers=1, n_heads=2)
    # No anneal ⇒ frac 1.0, lr untouched.
    cfg_off = PPOConfig(iterations=10, lr_anneal_final=1.0, policy_lr=3e-5)
    opt = make_optimizer(net, cfg_off)
    assert apply_lr_schedule(opt, cfg_off, 5) == pytest.approx(1.0)
    assert opt.param_groups[0]["lr"] == pytest.approx(3e-5)

    # Linear decay 1.0 → 0.1 over iters; policy group tracks base_lr·frac.
    cfg = PPOConfig(iterations=11, lr_anneal_final=0.1, policy_lr=3e-5, value_lr=1e-4)
    opt = make_optimizer(net, cfg)
    assert apply_lr_schedule(opt, cfg, 0) == pytest.approx(1.0)
    assert opt.param_groups[0]["lr"] == pytest.approx(3e-5)
    assert apply_lr_schedule(opt, cfg, 5) == pytest.approx(0.55)  # halfway
    assert opt.param_groups[0]["lr"] == pytest.approx(3e-5 * 0.55)
    assert opt.param_groups[1]["lr"] == pytest.approx(1e-4 * 0.55)
    assert apply_lr_schedule(opt, cfg, 10) == pytest.approx(0.1)  # final
    assert opt.param_groups[0]["lr"] == pytest.approx(3e-5 * 0.1)


def test_load_compat_pads_appended_cont_slot():
    """An N_CONT-1 checkpoint loads into the current net: overlap copied,
    the appended columns zeroed → identical behavior at init."""
    import torch
    from ppo.model import N_CONT, RunPolicyNet, load_compat

    net = RunPolicyNet(dim=32, n_layers=1, n_heads=2)
    sd = net.state_dict()
    # Simulate an older checkpoint with one fewer continuous slot (12 sinusoid
    # features per slot): truncate the cont projection's input columns.
    old = {k: v.clone() for k, v in sd.items()}
    w = old["cont_proj.proj.weight"]  # [dim, N_CONT*12]
    per_slot = w.shape[1] // N_CONT
    old["cont_proj.proj.weight"] = w[:, : (N_CONT - 1) * per_slot].contiguous()

    fresh = RunPolicyNet(dim=32, n_layers=1, n_heads=2)
    load_compat(fresh, old)
    fw = fresh.state_dict()["cont_proj.proj.weight"]
    assert fw.shape == (32, N_CONT * per_slot)
    # Overlap copied verbatim; trailing columns zeroed.
    assert torch.equal(fw[:, : (N_CONT - 1) * per_slot], old["cont_proj.proj.weight"])
    assert torch.count_nonzero(fw[:, (N_CONT - 1) * per_slot :]) == 0
    # A same-shape tensor loads verbatim.
    assert torch.equal(fresh.state_dict()["policy_head.weight"], old["policy_head.weight"])


def test_parse_proc_stat_and_cpu_util():
    from ppo.trainer import _cpu_util, _parse_proc_stat

    # "cpu user nice system idle iowait irq softirq ..." — busy excludes
    # idle(4th) + iowait(5th).
    a = _parse_proc_stat("cpu 100 0 50 800 50 0 0")
    assert a == (100 + 0 + 50 + 0 + 0, 100 + 0 + 50 + 800 + 50 + 0 + 0)  # (150, 1000)
    # Non-linux / malformed lines degrade to None (util instrumentation is
    # best-effort, never fatal).
    assert _parse_proc_stat("") is None
    assert _parse_proc_stat("intr 1 2 3 4 5") is None
    assert _parse_proc_stat("cpu 1 2") is None
    assert _parse_proc_stat("cpu a b c d e") is None

    # Busy fraction over the interval between two samples.
    b = _parse_proc_stat("cpu 250 0 100 1600 50 0 0")  # busy 350, total 2000
    assert _cpu_util(a, b) == pytest.approx((350 - 150) / (2000 - 1000))  # 0.2
    assert _cpu_util(None, b) is None
    assert _cpu_util(a, None) is None
    assert _cpu_util(a, a) is None  # zero elapsed → undefined, not a div-by-zero


def test_async_collect_defaults_off():
    from ppo.trainer import PPOConfig

    assert PPOConfig().async_collect is False


def test_behavior_stats_act1_frac_reports_the_act_mixture():
    """GH #545: `behavior_stats` must surface what fraction of a collected
    batch walked each Act-1 model, so a `--act1 roll` campaign that believes
    it is training ~50/50 on Overgrowth/Underdocks can check it in one line
    of `<save>.stats.jsonl` instead of discovering a skew after the fact."""
    from ppo.trainer import behavior_stats

    rows = [
        {"act1": "overgrowth", "transitions": []},
        {"act1": "overgrowth", "transitions": []},
        {"act1": "underdocks", "transitions": []},
        {"transitions": []},  # pre-#545 row: no "act1" key at all
    ]
    out = behavior_stats(rows)
    assert out["act1_frac/overgrowth"] == pytest.approx(0.5)
    assert out["act1_frac/underdocks"] == pytest.approx(0.25)
    assert out["act1_frac/unknown"] == pytest.approx(0.25)


def test_play_batch_returns_submission_order_despite_reversed_completion(monkeypatch, tmp_path):
    """play_batch must return rows in SUBMISSION order even when workers finish
    in the opposite order, while the JSONL sink records COMPLETION order.

    This invariant is load-bearing: the trainer concatenates transitions
    row-by-row and seeds its minibatch shuffle off the returned row order, so a
    completion-order return would silently make sync training nondeterministic.
    A fake pool + fake clock drives completion order without spawning workers.
    """
    import json

    from ppo import workers

    seeds = [10, 11, 12, 13, 14]
    n = len(seeds)

    class FakeClock:
        def __init__(self):
            self.now = 1000.0

        def time(self):
            return self.now

        def sleep(self, s):
            self.now += s

    clock = FakeClock()

    class FakeResult:
        def __init__(self, seed, ready_at):
            self.seed = seed
            self.ready_at = ready_at

        def ready(self):
            return clock.now >= self.ready_at

        def get(self, timeout=None):
            return {"seed": self.seed, "policy": "net", "outcome": "ok", "won": False}

    class FakePool:
        def __init__(self):
            self._i = 0

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def apply_async(self, fn, args):
            seed = args[1][0]  # (job index, job tuple)
            # Later submissions become ready EARLIER (spacing > the 0.1s poll
            # sleep so each readies alone) -> completion order is the reverse of
            # submission order.
            ready_at = clock.now + (n - self._i) * 0.15
            self._i += 1
            return FakeResult(seed, ready_at)

    class FakeCtx:
        def Pool(self, processes=None, initializer=None, initargs=()):
            return FakePool()

        def Array(self, typecode, size, lock=True):
            return [0.0] * size

    monkeypatch.setattr(workers.mp, "get_context", lambda method: FakeCtx())
    monkeypatch.setattr(workers, "time", clock)

    out = tmp_path / "runs.jsonl"
    rows = workers.play_batch(seeds, policy="net", procs=2, out_path=str(out))

    assert [r["seed"] for r in rows] == seeds  # returned: submission order
    sink_seeds = [json.loads(line)["seed"] for line in out.read_text().splitlines()]
    assert sink_seeds == list(reversed(seeds))  # sink: completion order


def test_dag_aggregates_normal_dag():
    """Reverse-topo DP over a small clean DAG: elite counting + dist-to-rest."""
    from ppo.mapfeat import MAP_AGG_CAP, _dag_aggregates

    # row2 leaves feed a row1 elite which feeds a row0 rest-site.
    nodes = {
        (0, 0): {"type": "rest_site", "children": [(0, 1)]},
        (0, 1): {"type": "elite", "children": [(0, 2), (1, 2)]},
        (0, 2): {"type": "monster", "children": []},
        (1, 2): {"type": "monster", "children": []},
    }
    agg = _dag_aggregates(nodes)
    assert agg[0, 2] == (0, 0, MAP_AGG_CAP)
    # one elite ahead; no rest ahead → dist-to-rest stays capped.
    assert agg[0, 1] == (1, 1, MAP_AGG_CAP)
    assert agg[0, 0] == (1, 1, 0)  # rest here → dist_rest 0; elite still ahead


def test_dag_aggregates_same_row_edge_no_keyerror():
    """A forward/same-row child not yet aggregated must be skipped, not indexed.

    Regression: `[c for c in children if c in nodes]` then `agg[c]` raised
    KeyError when a child shared its parent's row (processed later in the tie
    order), crashing runs on some maps (observed at A10). The aggregate now
    filters on `c in agg`.
    """
    from ppo.mapfeat import _dag_aggregates

    # (0, 1) points to same-row sibling (1, 1), inserted after it → the old
    # code indexes agg[(1, 1)] before it exists.
    nodes = {
        (0, 1): {"type": "elite", "children": [(1, 1)]},
        (1, 1): {"type": "monster", "children": [(0, 2)]},
        (0, 2): {"type": "monster", "children": []},
    }
    agg = _dag_aggregates(nodes)  # must not raise
    # (0, 1)'s only child was the un-aggregated same-row sibling → treated as
    # a leaf: just its own elite flag, capped dist-to-rest.
    assert agg[0, 1][0] == 1 and agg[0, 1][1] == 1


def test_update_precision_defaults_to_fp32_and_rejects_unknown():
    import torch
    from ppo.trainer import PPOConfig, update_autocast

    assert PPOConfig().update_precision == "fp32"
    assert not update_autocast(PPOConfig(), torch.device("cpu"))._enabled
    with pytest.raises(ValueError, match="update-precision"):
        update_autocast(PPOConfig(update_precision="fp16"), torch.device("cpu"))


def _tiny_transitions(n: int, seed: int = 0) -> list:
    from ppo.schema import N_CAT, N_CONT, Featurized
    from ppo.trainer import Transition

    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        n_obs, n_act = int(rng.integers(4, 9)), int(rng.integers(2, 5))
        feat = Featurized(
            obs_cat=np.zeros((n_obs, N_CAT), dtype=np.int64),
            obs_cont=rng.random((n_obs, N_CONT), dtype=np.float32),
            obs_sec=np.zeros(n_obs, dtype=np.int64),
            act_cat=np.zeros((n_act, N_CAT), dtype=np.int64),
            act_cont=rng.random((n_act, N_CONT), dtype=np.float32),
            act_sec=np.ones(n_act, dtype=np.int64),
            aux_room=np.zeros(n_act, dtype=np.int64),
            n_actions=n_act,
        )
        out.append(
            Transition(
                feat=feat,
                action_idx=int(rng.integers(0, n_act)),
                old_logprob=float(-np.log(n_act)),
                value=0.0,
                advantage=float(rng.normal()),
                ret=float(rng.random()),
            )
        )
    return out


def test_bf16_update_trains_in_fp32_weights_and_leaves_fp32_bit_identical():
    """`update_precision=bf16` must engage autocast (the weights move
    differently) while the weights stay fp32; the fp32 default must be the
    pre-flag update, bit for bit."""
    import torch
    from ppo.model import RunPolicyNet
    from ppo.trainer import PPOConfig, RunningMoments, make_optimizer, ppo_update

    trans = _tiny_transitions(24)

    def run(precision: str) -> tuple[dict, dict]:
        torch.manual_seed(0)
        net = RunPolicyNet(dim=32, n_layers=1, n_heads=2)
        cfg = PPOConfig(epochs=1, minibatch=8, update_precision=precision)
        opt = make_optimizer(net, cfg)
        st = ppo_update(net, opt, trans, cfg, RunningMoments(), 0.01, np.random.default_rng(0))
        return st, {k: v.clone() for k, v in net.state_dict().items()}

    st32, w32 = run("fp32")
    st32b, w32b = run("fp32")
    assert all(torch.equal(w32[k], w32b[k]) for k in w32)
    assert st32 == st32b

    st16, w16 = run("bf16")
    assert st16["nb"] == st32["nb"] == 3
    assert all(np.isfinite(v) for v in st16.values())
    assert all(w.dtype == w32[k].dtype for k, w in w16.items())
    assert any(not torch.equal(w16[k], w32[k]) for k in w32)
