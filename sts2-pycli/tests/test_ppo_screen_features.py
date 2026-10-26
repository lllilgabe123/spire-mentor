"""Gates for the opt-in encoder experiment mechanism (`ppo.schema`).

Candidate encoder inputs are gated behind ``STS2_PPO_FEATURES`` so the default
schema stays byte-identical to every committed champion while an idea is being
*screened* (`docs/ppo-screen-playbook.md`). The mechanism is generic; the
feature it was introduced with is not here any more.

**`boss` graduated.** It was the harness's first and only candidate, keyed on
the boss meta-node's LEAD MONSTER (`vocab.monsters()`) because the encounter id
had no registry at the time. It has since been shipped as an unconditional
categorical keyed on the ENCOUNTER ID (`vocab.bosses()`, 9 rows rather than
~121, which also removes the "two encounters behind one lead monster share a
row" caveat the original carried), mirrored through the Rust encoder, the ONNX
export and the parity fixtures — the work the screen playbook calls "a separate,
larger job than screening it" — and trained into a gated champion (GH #431).
Its behavioural tests live in `test_ppo_boss_identity.py`; the pre-graduation
versions of the tests deleted from this file are in git history at `68d56e87a`.

**`reward_upgrade` is the current candidate** (GH #545), and it has a shape
`boss` did not: it appends NO slot. Card-reward options always had an upgrade
column and the encoder passed `False` into it, so the fix moves a VALUE in an
existing slot. Every tensor is identical either way, which means the usual
"an older champion drops the new column and stays bit-identical" argument does
not apply and cannot be measured — nothing about the graph reveals which arm
produced it. So the gates below pin the two things that are checkable:

* with the feature OFF, featurization is byte-identical even when the new obs
  key IS present (every producer ships it unconditionally), and
* with it ON, exactly one column moves.

The third gate is cross-language: `ppo/encoder.py` trains the champion and
`sts2-runai::ppo::encoder` serves it, so a name honoured by only one of them
would put a champion on an observation it never saw.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO / "python"


def _run(code: str, features: str) -> str:
    """Run `code` in a fresh interpreter at a given feature set — the flag is
    read at import time, which is also how the spawned collection workers see
    it."""
    env = dict(os.environ, PYTHONPATH=str(PYTHON_DIR), STS2_PPO_FEATURES=features)
    proc = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300
    )
    assert proc.returncode == 0, f"features={features!r} failed:\n{proc.stderr}"
    return proc.stdout.strip()


def test_declared_features_are_exactly_the_ones_gated_here() -> None:
    """A live reminder, not a tautology.

    Every name in `KNOWN_FEATURES` needs its own default-unchanged gate below
    (and, for a slot-appending one, the warm-start gate from git history at
    `68d56e87a`). A name added without them fails here.
    """
    from ppo.schema import KNOWN_FEATURES

    gated = {"reward_upgrade", "card_mods"}
    assert gated == set(KNOWN_FEATURES), (
        f"undeclared feature(s) {sorted(set(KNOWN_FEATURES) - gated)} — "
        "add the default-unchanged / one-column-moves gates for them here"
    )


def test_the_rust_encoder_honours_every_declared_feature() -> None:
    """A campaign TRAINS through `ppo/encoder.py` and then SERVES the champion
    through `sts2-runai::ppo::encoder` (run_arena, the workbench backend, the
    advisor exe). A name only one side knows is a champion evaluated on an
    observation it never saw in training — and, because these features change
    no tensor shape, nothing downstream would notice."""
    from ppo.schema import KNOWN_FEATURES

    src = (REPO.parent / "sts2-rs/crates/sts2-runai/src/ppo/features.rs").read_text()
    for name in sorted(KNOWN_FEATURES):
        assert f'= "{name}"' in src, f"{name} is not declared in ppo/features.rs"


def test_the_default_schema_is_what_the_champions_expect() -> None:
    """`N_CAT` is 12, not 10 -- cat[10]/cat[11] (deck-op, enchantment; GH #545)
    are unconditional APPENDS, the same shape as the cat[9] boss slot the
    module docstring's `emb_boss` note already covers. Every committed
    champion's ONNX graph was still exported at cat_dim=10, but both consumers
    (`ppo::net`'s `cat_dim`, `ppo.onnx_policy`'s `n_cat`) read that width off
    the graph they loaded and slice the trailing columns rather than assuming
    `N_CAT`, so those five champions stay bit-identical to what they were
    despite `N_CAT` moving out from under them (`test_champion_resolvers_agree.py`
    plus `ppo_net_parity` cover that slicing). This test pins the CURRENT
    default width, not "what every champion's graph was exported at"."""
    from ppo.schema import N_CAT, empty_cat

    assert N_CAT == 12
    assert len(empty_cat()) == N_CAT


def test_an_unknown_feature_name_fails_loudly() -> None:
    """The one guarantee that survives an empty feature set. A misspelled name
    must not be silently ignored — it would screen as a null and be recorded as
    evidence against the idea."""
    from ppo.schema import parse_features

    with pytest.raises(ValueError, match="nonexistent"):
        parse_features("nonexistent")
    assert parse_features("") == frozenset()


def test_an_unknown_feature_kills_the_process_at_import() -> None:
    """Same guarantee, but through the env var an actual screen would use —
    the in-process check above cannot see an import-time failure."""
    env = dict(os.environ, PYTHONPATH=str(PYTHON_DIR), STS2_PPO_FEATURES="nonexistent")
    out = subprocess.run(
        [sys.executable, "-c", "import ppo.schema"], env=env, capture_output=True, text=True
    )
    assert out.returncode != 0
    assert "nonexistent" in out.stderr


def test_reward_upgrade_appends_no_slot() -> None:
    """Unlike `boss`, this feature moves a value in an EXISTING column — so
    the schema widths must be IDENTICAL with it on or off. If one ever
    changed, the champion-serving path would need the append argument
    instead.

    Compares the two runs to EACH OTHER rather than to a hardcoded width: a
    hardcoded `"10 25 [...]"` is what broke here when `N_CAT` moved 10 → 12
    for cat[10]/cat[11] (deck-op, enchantment; GH #545) — an unrelated,
    correct schema change that this feature's actual invariant ("appends no
    slot") never stopped holding. Comparing on-vs-off keeps testing that
    invariant through the next schema change instead of snapshotting today's
    width."""
    off = _run("from ppo import schema; print(schema.N_CAT, schema.N_CONT)", features="")
    on = _run(
        "from ppo import schema; print(schema.N_CAT, schema.N_CONT, sorted(schema.ACTIVE_FEATURES))",
        features="reward_upgrade",
    )
    on_cat, on_cont, on_features = on.split(" ", 2)
    assert on_features == "['reward_upgrade']"
    assert off == f"{on_cat} {on_cont}"


def test_the_default_arm_ignores_the_flags_even_when_they_are_present() -> None:
    """Producers ship `card_reward_option_upgraded` unconditionally, so the
    committed champions see it on every card-pick decision from now on. With
    the feature off it must make no difference whatsoever — that is what keeps
    them usable until the campaign that retrains with it."""
    from ppo.encoder import featurize_row

    obs = {"card_reward_option_ids": ["CARD.BASH", "CARD.CLEAVE"]}
    actions = [
        {"action": "pick_card_reward", "idx": 0},
        {"action": "pick_card_reward", "idx": 1},
    ]
    bare = featurize_row(obs, actions, {})
    withflags = featurize_row({**obs, "card_reward_option_upgraded": [False, True]}, actions, {})
    assert (bare.act_cat == withflags.act_cat).all()
    assert (bare.act_cont == withflags.act_cont).all()


def test_reward_upgrade_moves_exactly_the_upgrade_column() -> None:
    """With the feature on, a pre-upgraded reward reads as upgraded — and
    nothing else about the token moves."""
    from ppo.encoder import featurize_row
    from ppo.schema import REWARD_UPGRADE

    obs = {
        "card_reward_option_ids": ["CARD.BASH", "CARD.CLEAVE"],
        "card_reward_option_upgraded": [False, True],
    }
    actions = [
        {"action": "pick_card_reward", "idx": 0},
        {"action": "pick_card_reward", "idx": 1},
    ]
    off = featurize_row(obs, actions, {})
    on = featurize_row(obs, actions, {}, frozenset({REWARD_UPGRADE}))
    assert list(off.act_cat[1]) != list(on.act_cat[1])
    assert off.act_cat[1][1] == 0 and on.act_cat[1][1] == 1
    # Only that one cell, on only that one option.
    assert list(off.act_cat[0]) == list(on.act_cat[0])
    off.act_cat[1][1] = on.act_cat[1][1]
    assert (off.act_cat == on.act_cat).all()
    assert (off.act_cont == on.act_cont).all()


def test_a_short_flag_list_reads_as_not_upgraded() -> None:
    """Bundle and choose-screen offerings carry no upgrade state at all
    (`RunManager::card_reward_upgraded_flags`), so a missing entry must read
    as "not upgraded" rather than raise on the index."""
    from ppo.encoder import featurize_row
    from ppo.schema import REWARD_UPGRADE

    feat = featurize_row(
        {"card_reward_option_ids": ["CARD.BASH"], "card_reward_option_upgraded": []},
        [{"action": "pick_card_reward", "idx": 0}],
        {},
        frozenset({REWARD_UPGRADE}),
    )
    assert feat.act_cat[0][1] == 0


# ── card_mods (GH #947) ──────────────────────────────────────────────────
#
# Same shape as `reward_upgrade`: the feature moves values into columns that
# exist either way. The enchantment column (`cat[11]`) is #545's; the amount
# slot (`C_ENCH_AMT`, cont 25) is an unconditional APPEND, gated for warm
# start by `test_ppo_trainer_units.py::test_load_compat_*` and for serving by
# `test_ppo_appended_slots_are_free.py`. So the checkable properties are the
# same two: OFF is byte-identical even with the new obs keys present, and ON
# moves only the enchanted tokens' enchantment cells.


def _card_mods_row():
    obs = {
        "deck": [["STRIKE_IRONCLAD", False], ["BASH", False], ["DEFEND_IRONCLAD", True]],
        "deck_enchantments": [None, ["SHARP", 3], None],
        "card_reward_option_ids": ["CARD.BASH", "CARD.SHRUG_IT_OFF"],
        "card_reward_option_enchantments": [None, ["NIMBLE", 2]],
        "card_removal_options": [["BASH", False]],
        "card_removal_option_enchantments": [["SHARP", 3]],
        "upgrade_select_deck_indices": [0, 1],
        "smith_options": ["STRIKE_IRONCLAD", "BASH"],
        "smith_option_enchantments": [None, ["SHARP", 3]],
        "max_hp": 80,
    }
    actions = [
        {"action": "pick_card_reward", "idx": 0},
        {"action": "pick_card_reward", "idx": 1},
        {"action": "remove_deck_card", "idx": 0},
        {"action": "upgrade_deck_card", "idx": 1},
        {"action": "shop_remove_card", "deck_idx": 0},
        {"action": "smith_card", "idx": 0},
        {"action": "smith_card", "idx": 1},
    ]
    return obs, actions


_ENCH_KEYS = (
    "deck_enchantments",
    "smith_option_enchantments",
    "card_reward_option_enchantments",
    "card_removal_option_enchantments",
    "card_grant_option_enchantments",
)


def test_card_mods_appends_no_slot() -> None:
    off = _run("from ppo import schema; print(schema.N_CAT, schema.N_CONT)", features="")
    on = _run(
        "from ppo import schema; print(schema.N_CAT, schema.N_CONT, sorted(schema.ACTIVE_FEATURES))",
        features="card_mods",
    )
    on_cat, on_cont, on_features = on.split(" ", 2)
    assert on_features == "['card_mods']"
    assert off == f"{on_cat} {on_cont}"


def test_the_default_arm_ignores_enchantments_even_when_they_are_present() -> None:
    """Producers ship the `*_enchantments` lists unconditionally, so every
    committed champion sees them from now on; off, they must change nothing."""
    from ppo.encoder import featurize_row

    obs, actions = _card_mods_row()
    bare = featurize_row({k: v for k, v in obs.items() if k not in _ENCH_KEYS}, actions, {})
    full = featurize_row(obs, actions, {})
    for name in ("obs_cat", "obs_cont", "act_cat", "act_cont"):
        assert (getattr(bare, name) == getattr(full, name)).all(), name


def test_card_mods_moves_only_the_enchanted_tokens_enchantment_cells() -> None:
    from ppo import schema, vocab
    from ppo.encoder import featurize_row

    obs, actions = _card_mods_row()
    off = featurize_row(obs, actions, {})
    on = featurize_row(obs, actions, {}, frozenset({schema.CARD_MODS}))
    sharp, nimble = vocab.enchantments().index("SHARP"), vocab.enchantments().index("NIMBLE")
    want_obs = {2: (sharp, 3)}  # obs row 0 is the state token; deck rows are 1..3
    want_act = {1: (nimble, 2), 2: (sharp, 3), 3: (sharp, 3), 6: (sharp, 3)}
    for cat_on, cont_on, cat_off, cont_off, want in (
        (on.obs_cat, on.obs_cont, off.obs_cat, off.obs_cont, want_obs),
        (on.act_cat, on.act_cont, off.act_cat, off.act_cont, want_act),
    ):
        for t in range(len(cat_on)):
            if t in want:
                ench, amount = want[t]
                assert cat_on[t][schema.CAT_ENCHANT] == ench
                assert cont_on[t][schema.C_ENCH_AMT] == np.float32(amount * schema.ENCH_AMT_SCALE)
            # Everything outside those two cells is unchanged on every token.
            c_on, c_off = list(cat_on[t]), list(cat_off[t])
            k_on, k_off = list(cont_on[t]), list(cont_off[t])
            if t in want:
                c_on[schema.CAT_ENCHANT] = c_off[schema.CAT_ENCHANT]
                k_on[schema.C_ENCH_AMT] = k_off[schema.C_ENCH_AMT]
            assert c_on == c_off and k_on == k_off, t
