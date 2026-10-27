"""Dump PPO encoder/net parity fixtures for the Rust port (Phase 2 §4a/§4b).

Drives PpoRunEnv greedily with the committed champion snapshot across seeds and
records every yielded decision: raw obs/actions/metrics, the featurized token
arrays (gate A: Rust must reproduce EXACTLY), and torch π/V (gate B: tract
must match < 1e-4). Asserts surface coverage — coarse AND fine, see
`REQUIRED_FINE_SURFACES` — and the T_MAX/A_MAX budget.

The walk has two phases: a dense head (`--max-rows` contiguous greedy
decisions at A0, which pins the common surfaces) and a coverage tail that keeps
only rows for the sparse deck-mutation fine surfaces until each has its quota.
The tail's last leg (`--asc-seeds`) runs at ascension 5 so Ascender's Bane
splits the `upgrade_deck_card` index space apart from a `!upgraded` scan of the
deck — see `CURSE_UPGRADE_ROWS` (GH #399 Task 5b).

Usage (from sts2-pycli, venv with sts2_rs; PYTHONPATH=python is required for
the `-m ppo...` module invocation to resolve the un-installed package):

    PYTHONPATH=python uv run python -m ppo.dump_parity_fixtures

With `STS2_PPO_FEATURES` set the script switches to FEATURE MODE and writes a
small SUPPLEMENTARY fixture (default `--out` gains the feature names) holding
only rows that exercise the gated branch — the default fixture keeps pinning
the OFF path, and every row records the schema it was dumped at so the Rust
harness featurizes it at that schema rather than at the runner's environment:

    STS2_PPO_FEATURES=reward_upgrade PYTHONPATH=python uv run python \
        -m ppo.dump_parity_fixtures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import schema
from .encoder import featurize_row, fine_surface
from .env import PpoRunEnv, Terminal
from .net_policy import NetPolicy

# NOTE: "treasure" is deliberately excluded. `env.py::_PHASE_SURFACE` maps
# `Phase::Treasure` to a "treasure" surface, but the modern (BUG-016-patched)
# engine fully auto-resolves treasure rooms inside `resolve_treasure_room`
# (sts2-rs/crates/sts2-core/src/run_manager.rs) — relic pick + gold are
# granted directly and the phase advances straight to `CardReward` (relic
# skip/pick screen, surface "card-pick") or `MapChoice`. Even the legacy
# fallback path that *does* set `Phase::Treasure` only ever exposes exactly
# one action (`CollectTreasure`), so `PpoRunEnv._advance` auto-resolves it
# (`len(actions) == 1`) and it is never yielded as a Decision. "treasure" is
# therefore structurally unreachable as a `Decision.surface` value, not
# merely rare — no seed range will ever produce it.
REQUIRED_SURFACES = {"path", "card-pick", "shop", "event", "rest"}

# GH #399: `REQUIRED_SURFACES` above is keyed by the COARSE `Decision.surface`
# (`env.py::_PHASE_SURFACE`, phase-derived), under which every deck-mutation
# screen collapses into "card-pick" — so coarse coverage cannot pin the new
# rows. `encoder.fine_surface` is the finer key that the state token actually
# carries (`encoder.py:414`), and it is what the Rust mirror
# (`ppo/encoder.rs::fine_surface`) has to reproduce, so require it separately.
#
# Enchant reachability (the open question at the time this was written):
# ENCHANTS ARE REACHABLE and are therefore required here. There is no
# `interactive_card_enchant` flag — the `pending_deck_enchants` queue is gated
# by `interactive_card_transform`, which `PpoRunEnv` now sets (see the
# construction comment in `env.py::__post_init__`), and a 20-seed expert smoke
# on the flags-on env yielded 13 enchant decisions (vs remove 4, upgrade 7,
# transform 4). "card-grant" is deliberately NOT required: grants pre-date this
# campaign and appear only on a handful of event branches, so pinning them
# would make the dump's seed budget hostage to an unrelated surface. Rows for
# it are kept when they show up naturally.
REQUIRED_FINE_SURFACES = {"card-remove", "card-upgrade", "card-transform", "card-enchant"}

# Kept when the walk happens to produce one, but NOT required — the same
# treatment "card-grant" gets above, and for the same reason: making the seed
# budget hostage to a surface this sparse would cost more than it pins.
# "potion-swap" (GH #379) is the sparsest of the lot: a 200-seed Ironclad A0
# PPO sweep measured 0.075 full-belt potion rewards per RUN, so requiring even
# one row would mean ~13 extra full runs on average. The branch is pinned
# directly instead, by `tests/test_ppo_potion_swap_surface.py` and its Rust
# twin in `ppo/encoder.rs`.
OPPORTUNISTIC_FINE_SURFACES = {"potion-swap"}

# How many rows to keep per required fine surface once the dense budget
# (`--max-rows` worth of contiguous greedy trajectory) is spent. Deck mutations
# are sparse — a few per RUN, not per floor — so covering them densely would
# mean tens of full runs and a ~10× larger fixture for rows that are almost all
# `card-reward`. Instead the tail seeds contribute ONLY the missing rare rows.
# This is a "collect up to" target that also decides when the walk may stop
# early; the hard contract asserted below is ≥1 row per required fine surface.
COVERAGE_ROWS_PER_FINE = 2

# GH #399 Task 5b. `upgrade_deck_card`'s `idx` indexes the engine's
# `Card::is_upgradable()` subset, which drops curses / statuses / Quest cards —
# NOT the `!upgraded` subset a naive scan of `obs["deck"]` yields. The two
# agree on every offer as long as the deck holds no non-upgradable card AHEAD
# of an offered one, so an A0 fixture pins the upgrade branch only in the
# degenerate case where the bug is invisible. Ascension 5 appends Ascender's
# Bane at floor 1 and every card acquired after it lands behind it, so the walk
# keeps going until it has at least this many `card-upgrade` rows on a SPLIT
# index space — "split" in the positional sense, see
# `_upgrade_index_spaces_differ`.
CURSE_UPGRADE_ROWS = 1
CURSE_UPGRADE_ASCENSION = 5

# GH #545, `reward_upgrade`. With a feature enabled the dump switches to
# FEATURE MODE: it writes a small supplementary fixture holding only rows that
# exercise the gated branch, rather than a second copy of the default one. The
# default fixture keeps pinning the OFF path (its rows record no features), so
# together they gate both arms.
#
# For `reward_upgrade` the gated branch needs a card reward the engine actually
# pre-upgraded, and `roll_for_upgrade`'s odds are `act_index * 0.25` with a
# combat base chance of 0.0 — so ACT 1 CAN NEVER PRODUCE ONE. A fixture that
# only reached act 1 would featurize identically with the feature on or off and
# the gate would pass while pinning nothing.
REWARD_UPGRADE_ROWS = 3

# GH #947, `card_mods`: the gated branch needs a card that carries an
# enchantment, and the deck holds one in only ~8% of act-1 decisions (21% over
# a run; `docs/experiments/2026-10-26-card-modifier-prevalence.md`). Of those
# rows, at least `CARD_MODS_OPTION_ROWS` must have the enchanted card among the
# OPTIONS, so the action-token branch is pinned and not only the deck tokens.
CARD_MODS_ROWS = 5
CARD_MODS_OPTION_ROWS = 1
_OPTION_ENCH_KEYS = (
    "card_reward_option_enchantments",
    "smith_option_enchantments",
    "card_removal_option_enchantments",
    "card_grant_option_enchantments",
)


def _gated(obs: dict) -> dict[str, bool]:
    """Which experiment features' gated branch this obs exercises."""
    option_mod = any(any(obs.get(k) or []) for k in _OPTION_ENCH_KEYS)
    deck_ench = obs.get("deck_enchantments") or []
    deck_idx = (
        list(obs.get("upgrade_select_deck_indices") or [])
        + list(obs.get("transform_select_deck_indices") or [])
        + list(obs.get("enchant_select_deck_indices") or [])
    )
    option_mod = option_mod or any(0 <= i < len(deck_ench) and deck_ench[i] for i in deck_idx)
    return {
        schema.REWARD_UPGRADE: any(obs.get("card_reward_option_upgraded") or []),
        schema.CARD_MODS: any(deck_ench) or option_mod,
        "card_mods_option": option_mod,
    }


#: Rows each feature's fixture must hold that exercise its gated branch.
_FEATURE_QUOTA = {
    schema.REWARD_UPGRADE: REWARD_UPGRADE_ROWS,
    schema.CARD_MODS: CARD_MODS_ROWS,
    "card_mods_option": CARD_MODS_OPTION_ROWS,
}


def _quota_keys(features: list[str]) -> list[str]:
    keys = [f for f in features if f in _FEATURE_QUOTA]
    if schema.CARD_MODS in features:
        keys.append("card_mods_option")
    return keys


T_MAX, A_MAX = 320, 160


def _upgrade_index_spaces_differ(obs: dict) -> bool:
    """True iff a deck-derived upgrade index space would MIS-NAME some option.

    Not a length test. A row can hold a non-upgradable card and still featurize
    identically to the pre-fix code — if every excluded card sits AFTER all the
    offered ones (Ascender's Bane last, upgrade taken before any acquisition),
    the two spaces agree on every offer position and the shorter length changes
    nothing. Pinning on length alone would let a future regen land such a row
    and leave the assert green while pinning nothing.

    The real condition is POSITIONAL: some excluded master-deck index must sit
    before some offered one, which is exactly when `!upgraded` position `i` and
    `is_upgradable()` position `i` diverge. Mirrors the assertion in
    `tests/test_ppo_encoder_index_spaces.py` and the Rust twin in
    `ppo_encoder_parity.rs::upgrade_index_spaces_differ`.
    """
    offered = set(obs.get("upgrade_select_deck_indices") or [])
    excluded = [
        i for i, entry in enumerate(obs.get("deck") or []) if not entry[1] and i not in offered
    ]
    return bool(excluded) and bool(offered) and max(offered) > min(excluded)


def _assert_default_coverage(
    seen: set[str], fine_kept: dict[str, int], curse_upgrades: int
) -> None:
    """Coverage contract for the DEFAULT fixture (no features enabled)."""
    missing = REQUIRED_SURFACES - seen
    assert not missing, f"fixture lacks surfaces: {missing} — add seeds"
    missing_fine = {s for s in REQUIRED_FINE_SURFACES if not fine_kept.get(s)}
    assert not missing_fine, (
        f"fixture lacks fine surfaces: {sorted(missing_fine)} — add seeds "
        f"(kept: { {k: v for k, v in sorted(fine_kept.items())} })"
    )
    assert curse_upgrades >= CURSE_UPGRADE_ROWS, (
        f"fixture has {curse_upgrades} `card-upgrade` rows on a SPLIT index "
        f"space (need {CURSE_UPGRADE_ROWS}) — no kept upgrade row has the "
        "engine skipping a card that sits BEFORE some offered one, so every "
        "offer position names the same card either way and the Task 5b "
        "branch is pinned only where the defect is invisible. Add "
        f"`--asc-seeds` (walked at ascension {CURSE_UPGRADE_ASCENSION})."
    )


def dump(ckpt: str, out: str, seeds: list[int], max_rows: int, asc_seeds: list[int]) -> None:
    # Read once: `schema` fixes it at import time and the recorded rows must
    # say which schema produced them, so the Rust harness can featurize each
    # row at the schema it was dumped at instead of at the runner's ambient
    # environment (`ppo_encoder_parity.rs::row_features`).
    features = sorted(schema.ACTIVE_FEATURES)
    pol = NetPolicy(ckpt)
    # The TAG and the COMPUTATION must be the same schema. They come from
    # different places -- the tag from this process's environment, the policy
    # from the checkpoint's own cfg -- and when they disagree the fixture is
    # silently mislabeled: rows claiming one schema holding pi/V recorded at
    # another. That is not hypothetical. `ppo_encoder_parity.jsonl` was dumped
    # with the env unset against a checkpoint whose cfg declared
    # `reward_upgrade`, so its "default schema" rows held reward_upgrade pi/V;
    # scored at the schema they claimed, the worst per-action probability drift
    # was 0.676, and scored at reward_upgrade it was 0.000001.
    if sorted(pol.features) != features:
        raise SystemExit(
            f"schema mismatch: rows would be TAGGED {features} but the policy "
            f"loaded from {ckpt} featurizes at {sorted(pol.features)}. Set "
            f"STS2_PPO_FEATURES to match the checkpoint's cfg, or dump against "
            f"a checkpoint whose cfg matches the schema you want to record."
        )
    quota_keys = _quota_keys(features)
    gated_rows = dict.fromkeys(quota_keys, 0)
    rows: list[dict] = []
    seen: set[str] = set()
    fine_kept: dict[str, int] = {}
    curse_upgrades = 0
    dense = 0  # rows taken from the contiguous head of the walk
    tmax_seen = amax_seen = 0
    # The A0 cells carry the dense head (common surfaces + the T_MAX/A_MAX
    # budget) and most of the deck-mutation coverage; the A5 cells are a
    # coverage tail that exists only for the split-index-space upgrade rows.
    cells = [(s, 0) for s in seeds] + [(s, CURSE_UPGRADE_ASCENSION) for s in asc_seeds]

    def covered() -> bool:
        """Every quota this dump exists to fill is met — stop before paying
        for another full run."""
        base = (
            dense >= max_rows
            and all(fine_kept.get(s, 0) >= COVERAGE_ROWS_PER_FINE for s in REQUIRED_FINE_SURFACES)
            and curse_upgrades >= CURSE_UPGRADE_ROWS
        )
        return base and all(gated_rows[k] >= _FEATURE_QUOTA[k] for k in quota_keys)

    for seed, ascension in cells:
        # Coverage is complete: dense budget spent AND every required fine
        # surface has its quota AND the split-index-space upgrade row is in.
        # Stop before paying for another full run.
        if covered():
            break
        env = PpoRunEnv(seed=seed, ascension=ascension)
        out_d = env.reset()
        while not isinstance(out_d, Terminal):
            feat = featurize_row(out_d.obs, out_d.actions, out_d.metrics)
            probs, value = pol.distribution(out_d.obs, out_d.actions, out_d.metrics)
            fine = fine_surface(out_d.actions)
            # Head of the walk: keep everything (this is what pins the common
            # surfaces and the T_MAX/A_MAX budget). Tail: keep only rows for a
            # required fine surface that is still short of its quota.
            split = fine == "card-upgrade" and _upgrade_index_spaces_differ(out_d.obs)
            hits = _gated(out_d.obs)
            wanted = [k for k in quota_keys if hits[k] and gated_rows[k] < _FEATURE_QUOTA[k]]
            covers = (
                dense < max_rows
                or (
                    fine in REQUIRED_FINE_SURFACES
                    and fine_kept.get(fine, 0) < COVERAGE_ROWS_PER_FINE
                )
                or (split and curse_upgrades < CURSE_UPGRADE_ROWS)
                or (
                    fine in OPPORTUNISTIC_FINE_SURFACES
                    and fine_kept.get(fine, 0) < COVERAGE_ROWS_PER_FINE
                )
            )
            # Feature mode is the FULL walk plus the gated rows, not just the
            # gated rows. It was supplementary only while the committed
            # champion was default-schema: the default fixture pinned gate B
            # and this one topped up the ON branch. Since Ironclad's champion
            # declares `reward_upgrade` (2026-08-28) that is inverted — gate B
            # can only score rows recorded at the SERVED net's schema, so a
            # 3-row supplement would leave it scoring 3 of 158 rows on one
            # surface. Whichever schema the champion declares needs a fixture
            # with full surface coverage.
            keep = covers or bool(wanted)
            if keep:
                for k in quota_keys:
                    gated_rows[k] += int(hits[k])
                if dense < max_rows:
                    dense += 1
                fine_kept[fine] = fine_kept.get(fine, 0) + 1
                curse_upgrades += int(split)
                tmax_seen = max(tmax_seen, feat.obs_cat.shape[0] + feat.n_actions)
                amax_seen = max(amax_seen, feat.n_actions)
                seen.add(out_d.surface)
                rows.append(
                    {
                        "seed": seed,
                        # Documentary only — neither gate reads it (the
                        # featurized value lives in `obs["ascension"]`). It is
                        # here so a human diffing the fixture can tell the A0
                        # dense head from the A5 coverage tail at a glance.
                        "ascension": ascension,
                        "decision": out_d.step_index,
                        "phase": out_d.phase,
                        "surface": out_d.surface,
                        "fine_surface": fine,
                        "features": features,
                        "obs": out_d.obs,
                        "actions": out_d.actions,
                        "metrics": out_d.metrics,
                        "feat": {
                            "obs_cat": feat.obs_cat.tolist(),
                            "obs_cont": feat.obs_cont.tolist(),
                            "obs_sec": feat.obs_sec.tolist(),
                            "act_cat": feat.act_cat.tolist(),
                            "act_cont": feat.act_cont.tolist(),
                            "act_sec": feat.act_sec.tolist(),
                            "n_actions": feat.n_actions,
                        },
                        "probs": probs,
                        "value": value,
                    }
                )
            out_d = env.step(int(np.argmax(probs)))
    # The whole point of the supplementary fixture: without a row that
    # exercises the gated branch, both arms emit the same tokens and the Rust
    # gate pins nothing (`reward_upgrade`: act 1 can never produce one, see
    # `REWARD_UPGRADE_ROWS`; `card_mods`: see `CARD_MODS_ROWS`).
    short = {k: gated_rows[k] for k in quota_keys if gated_rows[k] < _FEATURE_QUOTA[k]}
    assert not short, (
        f"{features} fixture is short of gated rows {short} (need "
        f"{ {k: _FEATURE_QUOTA[k] for k in short} }) — the walk never reached "
        "enough of them. Add seeds."
    )
    _assert_default_coverage(seen, fine_kept, curse_upgrades)
    assert tmax_seen <= T_MAX and amax_seen <= A_MAX, (tmax_seen, amax_seen)
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            # allow_nan=False: NaN/Infinity would serialize as invalid-for-serde
            # JSON tokens: fail loudly here rather than write a broken fixture.
            f.write(json.dumps(r, allow_nan=False) + "\n")
    print(
        f"{len(rows)} rows ({dense} dense), surfaces={sorted(seen)}, "
        f"fine={ {k: v for k, v in sorted(fine_kept.items())} }, "
        f"split-index-space upgrade rows={curse_upgrades}, "
        f"max T={tmax_seen}, max A={amax_seen}"
    )


if __name__ == "__main__":
    from workbench.ppo_policy import ckpt_path

    ap = argparse.ArgumentParser()
    # MUST match the champion the Rust side loads — `champion_onnx_filename`
    # in sts2-rs/crates/sts2-runai/src/ppo/net.rs — or gate B
    # (ppo_net_parity: tract vs the fixture's torch π/V) fails on a weight
    # mismatch rather than a real port bug.
    #
    # RESOLVED, not hardcoded. This default used to be the literal
    # `asc_it180.pt`, which was correct only until the next Ironclad promotion —
    # at which point regenerating fixtures would silently bake the OLD net's
    # π/V into them and gate B would fail looking exactly like the port bug the
    # comment above warns you it isn't.
    ap.add_argument("--ckpt", default=ckpt_path("Ironclad"))
    # Feature mode writes a SEPARATE, supplementary fixture (see
    # `REWARD_UPGRADE_ROWS`): defaulting both modes to the same path would let
    # a `STS2_PPO_FEATURES=…` dump silently replace the default fixture with a
    # 3-row one, taking every OFF-path row with it.
    suffix = "_" + "_".join(sorted(schema.ACTIVE_FEATURES)) if schema.ACTIVE_FEATURES else ""
    ap.add_argument(
        "--out",
        default=f"../sts2-rs/crates/sts2-runai/tests/fixtures/ppo_encoder_parity{suffix}.jsonl",
    )
    # 24 (was 12): the dense head only needs ~2 runs, but the deck-mutation
    # coverage tail needs a wider budget — `card-remove` is the rarest of the
    # required fine surfaces (measured on this champion's greedy trajectories:
    # ~1 removal decision per 7 runs, vs ~1 transform/enchant per 2–3). The
    # walk breaks out as soon as the dense budget and every fine quota are
    # satisfied, so the extra seeds cost nothing when the early ones cover.
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(24)))
    # Walked at ascension 5 AFTER `--seeds`, purely to reach a `card-upgrade`
    # row whose `is_upgradable()` offer is shorter than the `!upgraded` scan
    # (Ascender's Bane, in the deck from floor 1). 3 seeds: an upgrade
    # decision lands roughly once every 2-3 runs on this champion's greedy
    # trajectories, and the walk breaks out as soon as one lands.
    ap.add_argument("--asc-seeds", type=int, nargs="+", default=[101, 102, 103])
    ap.add_argument("--max-rows", type=int, default=150)
    a = ap.parse_args()
    if a.ckpt is None:
        ap.error(
            "could not resolve the Ironclad champion checkpoint under data/ppo "
            "(pass --ckpt explicitly)"
        )
    dump(a.ckpt, a.out, a.seeds, a.max_rows, a.asc_seeds)
