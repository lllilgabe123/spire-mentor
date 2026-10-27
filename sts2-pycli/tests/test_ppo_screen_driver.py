"""Pure-python gates for `ppo.screen`'s report machinery (no training).

Two of these exist because the code was wrong the first time:

- `config_key` replaced a label comparison that could never match, so the
  A/A-run warning — the screen's own self-check — never fired.
- `mcnemar_exact` is the number a screen result is read off, so it is pinned
  against hand-computed binomial tails.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from ppo.screen import Arm, mcnemar_exact, tail_pooled


def test_mcnemar_exact_matches_hand_computed_tails():
    # No discordant pairs ⇒ nothing to test against.
    assert mcnemar_exact(0, 0) == pytest.approx(1.0)
    # Perfectly balanced ⇒ p = 1.
    assert mcnemar_exact(5, 5) == pytest.approx(1.0)
    # 1 vs 0: 2 * P(X <= 0 | n=1) = 2 * 0.5 = 1.0
    assert mcnemar_exact(1, 0) == pytest.approx(1.0)
    # 10 vs 0: 2 * (1/1024)
    assert mcnemar_exact(10, 0) == pytest.approx(2.0 / 1024)
    # 8 vs 2: 2 * [C(10,0)+C(10,1)+C(10,2)] / 2^10 = 2 * 56/1024
    assert mcnemar_exact(8, 2) == pytest.approx(2.0 * 56 / 1024)
    # Symmetric in its arguments — the test does not care which arm won.
    assert mcnemar_exact(2, 8) == pytest.approx(mcnemar_exact(8, 2))
    # Never exceeds 1 even when the doubled tail would.
    assert mcnemar_exact(3, 3) <= 1.0


def test_config_key_detects_an_aa_run():
    """The arms' labels always differ (they lead with 'a' / 'b'), so the A/A
    self-check has to compare the configuration, not the label."""
    a = Arm("a")
    b = Arm("b")
    assert a.label() != b.label()
    assert a.config_key() == b.config_key()

    assert Arm("b", features="boss").config_key() != a.config_key()
    assert Arm("b", overrides={"lr_anneal_final": "1.0"}).config_key() != a.config_key()
    # Override ORDER must not make two identical arms look different.
    assert (
        Arm("a", overrides={"x": 1, "y": 2}).config_key()
        == Arm("b", overrides={"y": 2, "x": 1}).config_key()
    )


def test_tail_pooled_weights_by_run_count_not_by_iteration():
    """Iterations can carry different `n_runs` (errored runs are dropped), so a
    plain mean over iterations would over-weight a short one."""
    stats = [
        {"win_rate": 0.0, "n_runs": 100},
        {"win_rate": 1.0, "n_runs": 100},
        {"win_rate": 0.5, "n_runs": 4},
    ]
    rate, n = tail_pooled(stats, 2)
    assert n == 104
    assert rate == pytest.approx((1.0 * 100 + 0.5 * 4) / 104)

    # k larger than the history is fine.
    rate, n = tail_pooled(stats, 99)
    assert n == 204
    assert rate == pytest.approx((0 + 100 + 2) / 204)

    assert tail_pooled([], 5) == (0.0, 0)


def test_wall_hours_sums_iterations_not_phases():
    """Under `async_collect` the phases overlap, so the arm's cost is the sum
    of `iter_s`, not of `collect_s + train_s`."""
    from ppo.screen import wall_hours

    stats = [{"iter_s": 1800.0, "collect_s": 1700.0, "train_s": 900.0}] * 2
    assert wall_hours(stats) == {"iter_h": 1.0, "collect_h": 0.944, "train_h": 0.5}
    assert wall_hours([]) == {"iter_h": 0.0, "collect_h": 0.0, "train_h": 0.0}
