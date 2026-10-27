"""GH #545: `cat[10]` and `cat[11]` must cost the deployed fleet nothing.

The `emb_boss` precedent (`ppo/schema.py`) is the whole basis for landing
these without retraining four characters: both consumers read the `cat` width
off the graph they loaded, so a 10-categorical champion drops the new columns
and is bit-identical. This test is what makes that a checked property rather
than an argument.

Review round 1 (2026-08-25) found that the original four assertions here all
stay green under a MID-TABLE INSERTION (e.g. a new column landing at index 5,
pushing boss/deck-op/enchant one slot right): the five `.onnx` files are
untouched by a pure encoder-source change, so the sha256 checks don't move,
and `OnnxPolicy` slices by COUNT, so a 10-wide slice of a 13-wide row still
comes back shaped right and softmaxes to 1.0 -- it is just quietly reading
the wrong columns. `test_positional_identity_...` and
`test_the_first_ten_pad_columns_are_frozen` below exist to close that hole:
both index the encoder's output by LITERAL integer, never via
`schema.CAT_BOSS` / `CAT_DECK_OP` / `CAT_ENCHANT` -- those constants would be
renumbered right alongside an insertion, and asserting through them would
just be checking that the code agrees with itself.
"""

import hashlib
import math
import tomllib
from pathlib import Path

import numpy as np
from ppo import schema, vocab
from ppo.encoder import featurize_row

PPO_DIR = Path(__file__).resolve().parent.parent / "data" / "ppo"


def test_every_committed_champion_onnx_still_matches_its_recorded_sha256():
    """A byte-drift guard, not an insert-vs-append guard.

    This catches a champion that was re-exported, retrained, or swapped for
    a different file -- it does NOT catch a mid-table insertion in the
    Python encoder, because that changes what the columns mean without ever
    touching a committed `.onnx`'s bytes. See the positional tests below for
    that failure mode.
    """
    champs = tomllib.loads((PPO_DIR / "champions.toml").read_text())
    for character, entry in champs.items():
        blob = (PPO_DIR / entry["onnx"]).read_bytes()
        got = hashlib.sha256(blob).hexdigest()
        assert got == entry["sha256"], (
            f"{character}: {entry['onnx']} no longer matches the recorded hash "
            f"-- it was re-exported, retrained, or swapped for a different file."
        )


def test_a_ten_categorical_champion_serves_under_the_twelve_cat_encoder():
    from ppo.onnx_policy import OnnxPolicy

    champs = tomllib.loads((PPO_DIR / "champions.toml").read_text())
    # A champion that is ACTUALLY still 10-cat, discovered rather than named.
    # This used to hardcode "ironclad" as a synonym for "a pre-#545 export";
    # that stopped being true on 2026-08-28 when its champion was re-baked to
    # cat_dim 12, and the test failed for the one reason it was never about.
    narrow = [
        (name, entry)
        for name, entry in sorted(champs.items())
        if OnnxPolicy(str(PPO_DIR / entry["onnx"])).n_cat == 10
    ]
    assert narrow, (
        "no committed champion is still a 10-cat export, so the property this "
        "test exists to pin -- a NARROWER graph serving under the 12-cat "
        "encoder -- is no longer observable on the real fleet. Retire this "
        "test or pin it to an archived net; do not relax the bound."
    )
    _name, entry = narrow[0]
    pol = OnnxPolicy(str(PPO_DIR / entry["onnx"]))
    # Pinned as `== 10`, not `<= 12` -- the premise is specifically that THIS
    # graph is narrower than the encoder, and a lax bound would pass even if
    # every champion had been re-baked to 12.
    assert pol.n_cat == 10
    obs = {"deck": [["STRIKE", False]], "max_hp": 80}
    probs, value = pol.distribution(obs, [{"action": "pick_card_reward", "idx": 0}], {})
    assert len(probs) == 1 and abs(sum(probs) - 1.0) < 1e-6
    assert isinstance(value, float) and math.isfinite(value)


def test_positional_identity_would_catch_a_mid_table_insertion():
    """Featurize one decision that stamps a real value into card/upgrade
    (pre-#545) and boss/deck-op/enchant (#545's slots), then read every one
    of those columns off a LITERAL index.

    If a future change inserted a column ahead of index 9 instead of
    appending after 11, `schema.CAT_BOSS`/`CAT_DECK_OP`/`CAT_ENCHANT` would
    be renumbered to match -- so an assertion written through those names
    would silently keep passing (both sides move together). Indexing by the
    bare integer is what makes an insertion show up: the value that used to
    live at column 9 would now be sitting at column 10, and this test would
    read column 9 and get something else instead (or IndexError on a row
    that no longer has room for it, since deck_op/enchant would then be one
    short of N_CAT).
    """
    obs = {
        "deck": [["STRIKE", False]],
        "enchant_select_deck_indices": [0],
        "pending_enchantment": "GOOPY",
        "max_hp": 80,
        "map": {
            "row_count": 10,
            "col_count": 6,
            "current": {"col": 2, "row": 5},
            "boss": {"col": 2, "row": 15, "id": "KAISER_CRAB_BOSS"},
            "visited": [],
            "nodes": [],
        },
    }
    feat = featurize_row(obs, [{"action": "enchant_deck_card", "idx": 0}], {})

    act_row = feat.act_cat[0]  # the one candidate action: enchant STRIKE with GOOPY
    assert act_row[0] == vocab.cards().index("STRIKE")  # pre-#545, must stay put
    assert act_row[1] == 0  # pre-#545 upgrade flag, must stay put
    assert act_row[10] == vocab.deck_op_index("enchant")  # GH #545 append #1
    assert act_row[11] == vocab.enchantments().index("GOOPY")  # GH #545 append #2

    state_row = feat.obs_cat[feat.obs_sec == schema.SEC_STATE][0]
    assert state_row[9] == vocab.bosses().index("KAISER_CRAB_BOSS")  # pre-#545 append
    assert state_row[11] == vocab.enchantments().index("GOOPY")  # mirrored onto state


def test_the_first_ten_pad_columns_are_frozen():
    """`pad_to_fixed` tiles `empty_cat()` across the whole T_MAX budget, so
    these ten numbers are baked into every one of the five shipped `.onnx`
    graphs at the positions a trained embedding table indexes by. An APPEND
    (GH #545's deck-op/enchant landing at 10/11) changes nothing here; a
    mid-table INSERT shifts this list, which the sha256 test above cannot
    see (an insertion is a pure Python source change -- it never touches a
    committed .onnx's bytes).

    Like `test_ppo_boss_identity.py::test_it_covers_every_boss_the_map_can_
    roll`, this is a CANARY: it also goes red the moment any of these ten
    vocabs legitimately grows (a new card, event, etc.), which means bump
    the numbers below and re-run `scripts/rebake_champion_vocab.py` -- not
    that an insertion happened. The two causes look the same here; the
    positional-identity test above is what tells them apart.
    """
    # GH #799 grew card / relic / potion for the v0.111.0 ids (581 -> 600,
    # 298 -> 300, 64 -> 65) and re-exported every champion over them
    # (`scripts/reexport_champions_vocab.py`, the `_v0111vocab` nets).
    assert schema.empty_cat()[:10] == [600, 17, 300, 65, 10, 70, 12, 8, 16, 13]


def test_every_champion_predating_card_mods_serves_under_the_wider_cont_encoder():
    """GH #947 appended cont slot 25 (`C_ENCH_AMT`). No committed champion was
    exported with it, and each must still serve: `OnnxPolicy` slices the
    trailing column off, exactly as `ppo::net` does for its `cont_dim`.
    Discovered, not named, for the reason the cat test above gives."""
    from ppo.onnx_policy import OnnxPolicy

    champs = tomllib.loads((PPO_DIR / "champions.toml").read_text(encoding="utf-8"))
    narrow = [
        entry
        for _name, entry in sorted(champs.items())
        if OnnxPolicy(str(PPO_DIR / entry["onnx"])).n_cont == schema.C_ENCH_AMT
    ]
    assert narrow, (
        "no committed champion is still a 25-cont export, so the narrower-graph "
        "property is no longer observable on the real fleet; retire this test"
    )
    pol = OnnxPolicy(str(PPO_DIR / narrow[0]["onnx"]))
    obs = {"deck": [["STRIKE", False]], "deck_enchantments": [["SHARP", 2]], "max_hp": 80}
    probs, value = pol.distribution(obs, [{"action": "pick_card_reward", "idx": 0}], {})
    assert len(probs) == 1 and abs(sum(probs) - 1.0) < 1e-6
    assert isinstance(value, float) and math.isfinite(value)


def test_the_enchant_amount_slot_sits_at_literal_column_25():
    """Positional, by literal index, for the reason
    `test_positional_identity_would_catch_a_mid_table_insertion` gives: an
    assertion through `schema.C_ENCH_AMT` would move with an insertion."""
    obs = {"deck": [["STRIKE", False]], "deck_enchantments": [["SHARP", 2]], "max_hp": 80}
    feat = featurize_row(obs, [], {}, frozenset({schema.CARD_MODS}))
    deck_row = feat.obs_cont[feat.obs_sec == schema.SEC_DECK][0]
    assert len(deck_row) == 26
    assert deck_row[25] == np.float32(0.2)
    assert not deck_row[:25].any()
    assert feat.obs_cat[feat.obs_sec == schema.SEC_DECK][0][11] == vocab.enchantments().index(
        "SHARP"
    )
