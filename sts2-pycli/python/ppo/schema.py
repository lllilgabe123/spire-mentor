"""Torch-free schema for the run-level policy net (GH #380).

The schema constants, ``Featurized``, and the fixed-shape padding used to
live in ``ppo.model`` / ``ppo.export_onnx``, both of which import torch at
module level. The ONNX advisor path (``ppo.onnx_policy``) must featurize
without torch installed, so the numpy-only pieces live here; ``model`` and
``export_onnx`` re-export them, keeping every existing import site valid.

Categorical schema (per token, PAD where N/A), index → meaning:
  0 card · 1 upgrade · 2 relic · 3 potion · 4 room · 5 event · 6 fixed-kind
  7 option-idx · 8 surface · 9 boss · 10 deck-op · 11 enchantment

(10 and 11 are GH #545, and both are APPENDS — the emb_boss property above
applies unchanged: a 10-categorical champion drops both columns and is
bit-identical. 10 exists because the only separator between a remove, an
upgrade, a transform, an enchant and a grant was `state_cat[8]`, and the
wb-27d55ac0 sweep showed that surface integer has zero authority over the
unconditional anti-upgraded prior every champion holds. 11 exists because
`pending_deck_enchants()` returned a bare COUNT, so Goopy — a downside —
featurized identically to Perfect Fit.)

(9 is the newest slot: the identity of the boss at the top of the current
act's map, carried on the state token and on the boss map node. The map
featurizer used to overwrite the boss node's identity with the literal room
type ``"boss"``, so the one fixed, known, run-defining fight the deck is being
built FOR was the one thing the policy could not see.)

**Why this one is unconditional when the screening harness gated its
candidate.** ``N_CAT`` is the width of the exported ONNX ``cat`` input and is
mirrored by the deployed Rust encoder, so the harness (`68d56e87a`) kept its
boss candidate behind ``STS2_PPO_FEATURES`` on the reasoning that bumping
``N_CAT`` would break champion serving the moment it landed. That reasoning is
right about an *inserted* slot. This slot is APPENDED, and both consumers now
read the ``cat`` width off the graph they loaded (``ppo::net``'s ``cat_dim``,
``ppo.onnx_policy``'s ``n_cat``) exactly as they already did for ``cont``. A
9-categorical champion therefore drops the boss column and is bit-identical to
what it was — verified by `ppo_net_parity` (tract on a 9-categorical champion
vs torch on this encoder) and by every committed champion's hash.

Experiment features (``STS2_PPO_FEATURES``)
-------------------------------------------
The harness above survives for the NEXT candidate input: a comma-separated env
var gates encoder experiments so the default schema stays byte-identical to
every committed champion while an idea is being screened. Its first entry,
``boss``, has graduated — it is the slot documented above, trained into a gated
champion, and mirrored through the Rust encoder / ONNX export / parity
fixtures. That last clause is the work the harness's own docs call out as "a
separate, larger job than screening it".

``reward_upgrade`` (GH #545) is the current entry, and it has a different shape
from ``boss``: it appends NO slot. Card-reward options always had an upgrade
column (``cat[1]``, the same one every deck / removal / transform token fills)
and the encoder simply passed ``False`` into it, so a reward the engine had
pre-upgraded featurized as the base card. ``roll_for_upgrade`` does that to
non-rare options 25% / 50% of the time in acts 2 / 3 below A7 (12.5% / 25% at
A7+), on the most-exercised surface in training (17.89 decisions/run).

Because nothing about the tensor changes, this one is NOT self-describing the
way an appended slot is: both consumers would happily serve a champion the
truth it was never trained on. Every champion to date holds an unconditional
anti-upgraded prior (``wb-27d55ac0``) that runs backwards here — handed the
flag mid-life it would learn to avoid free upgraded cards. So the env var is
the whole pairing mechanism, and it is mirrored in Rust
(``sts2-runai::ppo::features::Features``) because a campaign trains through
this encoder and then serves the champion through that one.

``card_mods`` (GH #947) stamps a card's OWN enchantment onto its token: the id
into the enchantment column (``cat[11]``, the table #545 added for the pending
pick) and the amount into the appended cont slot 25. It covers deck tokens,
the deck-op targets (remove / upgrade / transform / enchant / shop-remove /
grant) and relic-stamped reward options, read from the obs's parallel
``*_enchantments`` lists. C# keeps at most one enchantment per card and
persists nothing else per instance beyond the upgrade (``SerializableCard``:
id, upgrade level, one enchantment, a few card-specific counters), so this is
the card's whole modifier state. The column keeps its meaning on the
enchant-pick action tokens: ``CanEnchant`` admits only a target that is bare
or already holds the same (stackable) enchantment, so the target's own id
never disagrees with the pending one, and the amount says how many stacks it
already has. Gated for the ``reward_upgrade`` reason: the enchantment table
was only ever trained on the state token and the enchant-pick tokens, so a
champion handed it on every deck token would read noise.

Continuous schema (per token, zeros where N/A), index → meaning:
  0 hp_frac · 1 gold_scaled · 2 act_index · 3 deck_size_s · 4 n_upgraded_s
  5 n_relics_s · 6 dx · 7 dy · 8 reachable · 9 min_elites · 10 max_elites
  11 dist_rest · 12 price_scaled · 13 gold_delta_scaled · 14 hp_delta_frac
  15 enabled · 16 total_floor_frac · 17 act_floor_s · 18 ascension_scaled
  19 relics_gained · 20 relics_lost · 21 potions_gained · 22 potions_lost
  23 cards_gained · 24 cards_lost · 25 enchant_amount_s (``card_mods``)

(16/17 added after the first BC round: the state token carried NO floor
signal — act_index only — so card/shop/rest decisions couldn't condition
on progress within an act; floor conditioning was a proven lever in the
old learned pick-ranker, GH#63 +3.45pp.)

(19–24: an event option's payload was gold and HP and nothing else, so the
option that trades a relic away, drinks a potion or thins the deck featurized
IDENTICALLY to one that does nothing. Stone of All Time's "lose a potion, gain
10 Max HP" reads `0 / +10 / +10` — byte-identical to a free +10 — which is how
three stubbed trade events went unnoticed by every champion trained on them.
GAINED and LOST are separate slots on purpose: netting them collapses a swap
back to zero, and swapping is the whole point of Relic Trader.)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from . import vocab

# Raised by `pad_to_fixed` below, but defined in `ppo.errors` -- this module
# imports numpy, and the advisor's live path needs the exception type without
# it. Importing it here also keeps every existing `from ppo.schema import
# BudgetExceededError` resolving to the same class.
from .errors import BudgetExceededError

# Section ids.
SEC_STATE, SEC_DECK, SEC_RELIC, SEC_POTION, SEC_MAPNODE = 0, 1, 2, 3, 4
SEC_PATH, SEC_CARD, SEC_FIXED, SEC_RELIC_OPT, SEC_POTION_OPT, SEC_EVENT_OPT = 5, 6, 7, 8, 9, 10
N_SECTION = 11

#: Candidate encoder inputs, off by default. See the module docstring.
REWARD_UPGRADE = "reward_upgrade"
CARD_MODS = "card_mods"
KNOWN_FEATURES: frozenset[str] = frozenset({REWARD_UPGRADE, CARD_MODS})


def parse_features(spec: str) -> frozenset[str]:
    """``"foo,bar"`` → ``{"foo", "bar"}``; raises on an unknown name.

    Failing loud on a typo is the point: a misspelled feature would otherwise
    screen as a null and be recorded as "this idea does not help".
    """
    names = frozenset(t.strip() for t in spec.split(",") if t.strip())
    unknown = names - KNOWN_FEATURES
    if unknown:
        raise ValueError(f"unknown STS2_PPO_FEATURES entries {sorted(unknown)}")
    return names


ACTIVE_FEATURES = parse_features(os.environ.get("STS2_PPO_FEATURES", ""))

N_CAT = 12
CAT_BOSS = 9  # act-boss identity; see the module docstring
CAT_DECK_OP = 10  # which deck operation a card token belongs to (GH #545)
CAT_ENCHANT = 11  # which enchantment a pending deck-enchant pick stamps
N_CONT = 26
GOLD_SCALE = 0.01  # gold * 0.01 keeps it ~O(1)
ACT_SCALE = 1.0
DECK_SCALE = 0.05  # deck size * 0.05
ASC_SCALE = 0.1  # ascension level * 0.1 (A10 → 1.0); the difficulty input
AUX_IGNORE = -100

# Continuous indices (see module docstring).
C_HP, C_GOLD, C_ACT, C_DECKSZ, C_NUPG, C_NREL = 0, 1, 2, 3, 4, 5
C_DX, C_DY, C_REACH, C_MINE, C_MAXE, C_DREST = 6, 7, 8, 9, 10, 11
C_PRICE, C_GOLD_D, C_HP_D, C_ENABLED = 12, 13, 14, 15
C_FLOOR, C_ACTFLOOR = 16, 17
C_ASC = 18  # ascension level (state token only); appended slot ⇒ warm-start safe
# What an event option costs you in things that are not gold or HP. Appended
# slots ⇒ warm-start safe (see the module docstring on `cat` slot 9).
C_RELIC_GAIN, C_RELIC_LOSS = 19, 20
C_POTION_GAIN, C_POTION_LOSS = 21, 22
C_CARD_GAIN, C_CARD_LOSS = 23, 24
# The card's own enchantment amount (GH #947); card/deck tokens, `card_mods`
# feature only. Appended slot ⇒ a 25-cont champion drops it.
C_ENCH_AMT = 25
ENCH_AMT_SCALE = 0.1  # stacks run 1..~10 (Sharp / Nimble carry their payload)
FLOOR_SCALE = 1.0 / 45.0  # total_floor → [0,1] over a 3-act run
ACT_FLOOR_SCALE = 0.05  # act_floor (0..~15) → ~[0,0.75]


@dataclass
class Featurized:
    """Per-decision arrays. ``*_cat`` are int [T, N_CAT]; ``*_cont`` are
    float [T, N_CONT]; ``*_sec`` are int [T]."""

    obs_cat: np.ndarray
    obs_cont: np.ndarray
    obs_sec: np.ndarray
    act_cat: np.ndarray
    act_cont: np.ndarray
    act_sec: np.ndarray
    aux_room: np.ndarray  # [A] dest-room label per action (or AUX_IGNORE)
    n_actions: int


def pad_indices() -> dict[str, int]:
    p = {
        "card": vocab.cards().size,
        "relic": vocab.relics().size,
        "potion": vocab.potions().size,
        "room": vocab.N_ROOM,
        "upg": vocab.MAX_UPGRADE + 1,
        "event": vocab.events().size,
        "fixed": vocab.N_FIXED,
        "optidx": vocab.MAX_OPT_IDX,
        "surface": vocab.N_SURFACE,
        "boss": vocab.bosses().size,
        "deck_op": vocab.N_DECK_OP,
        "enchant": vocab.enchantments().size,
    }
    return p


def empty_cat() -> list[int]:
    p = pad_indices()
    # order: [card, upg, relic, potion, room, event, fixed, optidx, surface,
    #         boss, deck_op, enchant]
    return [
        p["card"],
        p["upg"],
        p["relic"],
        p["potion"],
        p["room"],
        p["event"],
        p["fixed"],
        p["optidx"],
        p["surface"],
        p["boss"],
        p["deck_op"],
        p["enchant"],
    ]


def pad_to_fixed(feat: Featurized, t_max: int, a_max: int) -> dict[str, np.ndarray]:
    o, a = feat.obs_cat.shape[0], feat.n_actions
    if o + a > t_max or a > a_max:
        raise BudgetExceededError(
            f"decision exceeds export budget: obs={o} act={a} (T_MAX={t_max}, A_MAX={a_max})"
        )
    pad = np.array(empty_cat(), dtype=np.int64)
    cat = np.tile(pad, (1, t_max, 1))
    cont = np.zeros((1, t_max, N_CONT), dtype=np.float32)
    sec = np.zeros((1, t_max), dtype=np.int64)
    pad_mask = np.ones((1, t_max), dtype=bool)
    act_mask = np.zeros((1, a_max), dtype=bool)
    cat[0, :o], cat[0, o : o + a] = feat.obs_cat, feat.act_cat
    cont[0, :o], cont[0, o : o + a] = feat.obs_cont, feat.act_cont
    sec[0, :o], sec[0, o : o + a] = feat.obs_sec, feat.act_sec
    pad_mask[0, : o + a] = False
    act_mask[0, :a] = True
    return {
        "cat": cat,
        "cont": cont,
        "sec": sec,
        "pad_mask": pad_mask,
        "obs_len": np.array([o], dtype=np.int64),
        "act_mask": act_mask,
    }
