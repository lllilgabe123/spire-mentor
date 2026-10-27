"""PPO-driven play driver for the real-game "SpireMentor" mod (real-game bridge
phase 3).

**Design correction discovered during exploration (read this first).** The
phase-3 task brief assumed the deployed PPO checkpoint (``data/ppo/
ppo_fullrun_*.pt``) is a *combat* policy -- mirroring :mod:`realgame_mcts`
but with PPO answering ``combat_action`` and a random policy answering
everything else. That is backwards. Reading :mod:`ppo.env` (the env the
champion nets were trained/evaluated in) and :mod:`workbench.ppo_policy`
(the workbench's existing PPO inference path) shows the opposite split:

    ``Phase::Combat`` decisions are resolved internally by the native combat
    MCTS agent (eval-cheap 1000/4000 sims -- "combat
    stays AS-IS", ``ppo/env.py`` module docstring). Every *non-combat*,
    multi-option decision (map choice, card reward, shop, rest site, event,
    boss relic, ...) is what the PPO net actually decides.

So this driver is the mirror image of the brief's literal wording, but the
faithful reading of "reuse the existing PPO inference path": **PPO answers
the non-combat run-level decisions it was trained for**
(``map_select``/``card_reward``/``shop``/``rest_site``/``event``/
``boss_relic`` -- the surfaces :func:`ppo.env.classify_surface` recognizes),
**combat is answered by MCTS** (reusing :func:`realgame_mcts.
pick_combat_action_mcts` verbatim -- ``--combat-policy mcts``, the default,
matches how the champion nets were themselves evaluated) **or, for a
cheaper/torch-only run, by the random-legal policy** (``--combat-policy
random``, reusing :func:`realgame_random.pick_combat_action`). Any
non-combat surface the PPO obs builders below don't (yet) cover --
``card_select``, ``card_bundle``, ``crystal_sphere``, ``reward_screen``,
``treasure``, ``run_config`` -- falls through to
:func:`realgame_random.choose_action`, same as an outright PPO failure.
Never-crash-the-session contract is identical to :mod:`realgame_mcts`: any
exception building the observation or running inference is caught, logged
loudly, and that one decision falls back to the random-legal policy with
``{"ppo_error": "..."}`` recorded on the JSONL step.

**Obs-schema gap (bridge vs. training).** The committed nets were trained
against :func:`ppo.env.observe`, which reads a *live* ``sts2_rs.RunManager``
(``rm.full_map()``, ``rm.shop_inventory()``, ``rm.event_option_infos()``,
...) -- there is no such live object over the bridge (the bridge only ever
sends the wire schema documented in ``spire-mentor-mod/PROTOCOL.md``). This
module does NOT reimplement the trained observation from scratch; it
reuses the *exact* training-time inference call
(:meth:`ppo.net_policy.NetPolicy.distribution`, which operates on the
already-dict-shaped ``(obs, actions, metrics)`` triple -- the same shape
:mod:`ppo.wpa` scores from recorded JSONL rows) and builds that triple
best-effort from bridge fields alone (see the ``_build_*`` functions
below). This is faithful where the bridge carries the same information the
live ``rm`` would expose (deck/relics/potions/map graph/shop stock/event
options/rest options) and degrades gracefully (OOV vocab slots, zero
features) where it doesn't. 2026-07-12 obs-parity audit closures: the act
map + current coord are session-cached from the last ``map_select`` and
patched into every surface (training obs always carry them); ``act_floor``
is derived engine-faithfully from that coord (``row + 1``); bridge card ids
are translated to the registry's ``_CARD``-suffixed canon
(:func:`_vocab_card_id`). Event ids used to be 100% OOV (``vocab.Registry``
keyed events by raw PascalCase names while ``index()`` canonicalized queries
to UPPER_SNAKE); commit a082efeef canonicalized the registry keys too, so
every ``EventId`` now resolves to its own slot and
``tests/test_ppo_vocab_roundtrip.py`` holds that shut. What is STILL open is
downstream of it (GH #302): the committed champions predating that commit
(Ironclad ``asc_it55``, Silent ``it390``, Defect ``asc_it80``) trained with
every event on the shared OOV row, so their per-event embedding rows never
took a gradient -- event identity is decodable by the encoder but not by
those nets, and only a retrain buys it.
None of this can crash a session -- see the never-crash contract above.

**Combat scores in ``annotations``.** Per PROTOCOL.md, the ``cards`` array
of an ``annotations`` payload only renders in the in-game overlay *while
combat is in progress* -- so for the PPO-driven non-combat decisions here,
only the ``note`` field is visible in the live overlay; ``cards`` is still
populated (indexed by candidate position, not hand position) for JSONL
corpus value and forward-compatibility with a non-combat overlay panel.
Combat annotations (when ``--combat-policy mcts``) are exactly
:mod:`realgame_mcts`'s own per-card MCTS win-prob overlay -- unmodified,
just reused.

Usage::

    uv run python python/realgame_ppo.py [--host 127.0.0.1] [--port 9002]
        [--log-dir logs/realgame] [--max-steps N] [--seed N]
        [--character IRONCLAD] [--ascension 0]
        [--model PATH] [--temperature 0.0]
        [--combat-policy {mcts,random}] [--sims 300] [--depth 200]
        [--c-puct 1.4] [--mcts-seed 0] [--rollout-policy heuristic]

``--model`` overrides the checkpoint lookup entirely (else the
``STS2_WB_PPO_CKPT`` env var, else the checkpoint
``data/ppo/champions.toml`` declares for the run's character -- see
:func:`workbench.ppo_policy.ckpt_path`, reused here unmodified). A character
with no entry in that manifest degrades to the random-legal policy for its
whole run, loudly logged once.
``--temperature 0`` (default) is greedy argmax over the net's masked
softmax; ``--temperature T > 0`` samples from a power-law re-tempering of
those probabilities (``p_i ** (1/T)``, renormalized) -- an approximation,
since :meth:`ppo.net_policy.NetPolicy.distribution` only exposes the
already-softmaxed probabilities, not raw logits.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import random
import re
import sys
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

import ppo_decode
from ppo.errors import BudgetExceededError  # numpy-free (ppo.schema imports numpy)
from realgame_bridge import DEFAULT_HOST, DEFAULT_PORT, RealGameBridgeClient
from realgame_mcts import (
    DEFAULT_SIMS_BOSS,
    DEFAULT_SIMS_REGULAR,
    chosen_map_node_type,
    combat_sims_for,
    pick_combat_action_mcts,
)
from realgame_random import (
    _TERMINAL_TYPES,
    JsonlLogger,
    _progress_line,
    choose_action,
    pick_combat_action,
)
from sts2_hint.recommender import RecommendConfig
from sts2_realgame import reward_screen_choice

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# obs/actions/metrics builders -- bridge decision -> ppo.net_policy.NetPolicy
# input triple. See module docstring's "obs-schema gap" note: best-effort
# from bridge fields, never a live sts2_rs.RunManager.
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class PpoDecision:
    """One non-combat decision, translated to the PPO net's input shape."""

    obs: dict[str, Any]
    actions: list[dict[str, Any]]  # ppo.env-shaped RunAction candidates
    metrics: dict[str, Any]
    responses: list[dict[str, Any]]  # bridge action dict per candidate (same order)
    labels: list[str]  # human labels for annotations/logging, same order
    #: Who actually decided, when it was not the net. A rule that overrides
    #: the net's ranking (today: the event-page walk-away filter, GH #487)
    #: names itself here so :func:`decision_note` can say so in the HUD/corpus
    #: note.
    source_note: str = ""
    #: How many cards ONE reply must name. 1 for every surface whose reply is
    #: a single index; >1 only for a deck mutation the world requires to be
    #: answered in one shot ("remove 2 cards"). The candidates stay
    #: one-per-card either way -- this says how many of them the reply
    #: carries, not how many the net ranks.
    pick_count: int = 1


@lru_cache(maxsize=1024)
def _vocab_card_id(raw: str) -> str:
    """Bridge card id -> the training vocab's canonical id.

    The Rust registry disambiguates cards whose name collides with another
    registry kind by a ``_CARD`` suffix (``RAGE_CARD``, ``FLAME_BARRIER_CARD``,
    ``SETUP_STRIKE_CARD``, ...); the bridge sends the plain game id
    (``RAGE``), which is OOV against the training vocab. 2026-07-12 obs-parity
    audit of the 12-run live corpus: 407/6074 deck tokens and 43/333
    card-reward options OOV, all from this suffix family -- the net could not
    see most Ironclad power cards. Try the suffixed form when the bare id
    misses; degrade to the raw id if the vocab isn't importable.
    """
    try:
        from ppo import vocab
    except Exception:  # pragma: no cover - vocab needs sts2_rs
        return raw
    reg = vocab.cards()
    if reg.index(raw) != reg.oov:
        return raw
    # `_STATUS` covers the one status card the registry disambiguates that way
    # (SLOTH -> SLOTH_STATUS; 2026-07-15 featurizer audit).
    for suffix in ("_CARD", "_STATUS"):
        suffixed = f"{raw}{suffix}"
        if reg.index(suffixed) != reg.oov:
            return suffixed
    return raw


def _card_id(c: dict[str, Any]) -> str | None:
    cid = c.get("id")
    return _vocab_card_id(str(cid)) if cid else None


def _alnum_upper(name: str) -> str:
    return "".join(ch for ch in name if ch.isalnum()).upper()


@lru_cache(maxsize=256)
def _vocab_enchantment_id(raw: str) -> str:
    """Bridge enchantment spelling -> the training vocab's UPPER_SNAKE id.

    The wire carries the C# id entry (``NIMBLE``) or the display title
    (``Soul's Power``); both reduce to the same alphanumeric key, which is
    what ``EnchantmentId::from_wire_name`` matches on in the Rust lane
    (``ppo::wire::card_enchantment``). An unknown spelling is kept verbatim so
    it lands on the vocab's OOV row instead of reading as unenchanted.
    """
    key = _alnum_upper(raw.removesuffix(".title"))
    try:
        from ppo import vocab
    except Exception:  # pragma: no cover - vocab needs sts2_rs
        return raw
    for name in vocab.enchantments().names:
        if _alnum_upper(name) == key:
            return name
    return raw


def _card_enchantment(c: dict[str, Any]) -> tuple[str, int] | None:
    """A wire card's enchantment as ``(registry id, amount)``, or ``None`` (GH #947).

    An absent ``enchantment_amount`` is one stack (PROTOCOL.md; the engine's
    ``state_dict`` reads it the same way).
    """
    ench = c.get("enchantment")
    if not ench:
        return None
    amount = c.get("enchantment_amount")
    return _vocab_enchantment_id(str(ench)), int(amount) if amount is not None else 1


@lru_cache(maxsize=1)
def _non_removable_cards() -> frozenset[str]:
    """Card ids the game will not put on a removal screen, FROM THE ENGINE.

    This replaced a hand-copied basic-Strike/Defend list. That list was
    justified by "the net was TRAINED on a surface that excluded them", and
    the justification did not survive its own history: `Phase::Shop` has
    pushed one `ShopRemoveCard` per `removal_offer_order()` entry since
    2026-06-19, and `removal_offer_order` has been Eternal-gated since the
    same commit, so every champion trained after that date saw basics as
    shop-removal candidates. The copied list was what made inference differ
    from training, not what prevented it -- and because it excluded ONLY
    basics it also offered Eternal curses: `logs/f7-mirror-check18` carries
    `remove:ASCENDERS_BANE` and `remove:ENTHRALLED` shop picks that no
    `FromDeckForRemoval` screen can satisfy, which dropped the carried
    target and let the follow-up `card_select` re-decide the removal.

    v0.107.1's only gate is `CardModel.IsRemovable`
    (`!Keywords.Contains(CardKeyword.Eternal)`, `CardModel.cs:738`), applied
    with no extra filter (`CardSelectCmd.cs:629`). Asking the engine rather
    than restating that here is the whole point: the two previous attempts
    to hand-maintain this rule in Python both got it wrong.
    """
    import sts2_rs

    return frozenset(sts2_rs.non_removable_cards())


@lru_cache(maxsize=1)
def _eternal_granting_enchantments() -> frozenset[str]:
    """Enchantments that make ANY card non-removable, FROM THE ENGINE.

    Removability is not a property of the card id alone. C# reads the card's
    LIVE keyword set, and `TezcatarasEmber.OnEnchant` calls
    `Card.AddKeyword(CardKeyword.Eternal)` (`TezcatarasEmber.cs:18`);
    `NutritiousSoup` applies it to every basic Strike in the deck. Filtering
    the shop fan-out on card id alone therefore offers Strikes the removal
    screen withholds -- the follow-up `card_select` cannot find the carried
    target and re-decides, which is exactly the failure the carry exists to
    prevent.
    """
    import sts2_rs

    return frozenset(sts2_rs.eternal_granting_enchantments())


def _wire_card_is_removable(card: dict[str, Any]) -> bool:
    """Can this wire ``master_deck`` entry appear on a removal screen?

    The bridge carries the game's own ``CardModel.IsRemovable`` as
    ``is_removable`` (GH #592); when it is there, that IS the answer --
    `CardSelectCmd.FromDeckForRemoval` applies no other filter, so a wire
    verdict and the screen cannot disagree. Everything below it is the
    reconstruction older bridge builds need, kept because a mismatched mod
    is a real deployment state and silently offering the whole deck would
    be worse than approximating.

    That reconstruction has been wrong twice, in both directions, which is
    why the wire flag now outranks it: it once offered Eternal curses no
    screen can satisfy (`logs/f7-mirror-check18`), and once offered a plain
    `OUTBREAK` carrying no enchantment on the wire that the screen still
    withheld (`logs/f7-mirror-check19`, GH #592 -- unrooted, and unrootable
    from a wire that never carried the keyword set the game filters on).

    An absent or unrecognised ``enchantment`` reads as "no enchantment", so
    an older bridge build that omits the field behaves exactly as before.
    """
    flag = card.get("is_removable")
    if isinstance(flag, bool):
        return flag
    if _canon_card(card.get("id")) in _non_removable_cards():
        return False
    ench = card.get("enchantment")
    if not ench:
        return True
    key = "".join(ch for ch in str(ench) if ch.isalnum()).upper()
    return not any(
        key == "".join(ch for ch in e if ch.isalnum()).upper()
        for e in _eternal_granting_enchantments()
    )


_CANON_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _canon_card(raw: Any) -> str:
    """Dialect-tolerant card-id canon (PascalCase or loc id -> UPPER_SNAKE)."""
    return _CANON_CAMEL.sub("_", str(raw or "")).upper()


def _base_obs(state: dict[str, Any], run_state: dict[str, Any]) -> dict[str, Any]:
    """Common obs fields every surface needs (deck/relics/potions/scalars/map).

    Mirrors :func:`ppo.env.observe`'s always-present fields, sourced from
    the bridge envelope's ``run_state`` instead of a live ``RunManager``.
    The training obs carry the current act map on EVERY decision; the
    2026-08-02 bridge attaches ``state["map"]`` to every run-level message
    (older builds only on ``map_select``), and
    :func:`pick_noncombat_action_ppo` patches the session-cached act graph
    (current coord = last chosen node) into ``state`` for any surface that
    still arrives bare -- 76% of live net decisions were missing all
    map-node tokens before that (2026-07-12 obs-parity audit).
    """
    deck = run_state.get("master_deck") or []
    m = state.get("map") or {}
    map_nodes = m.get("nodes")
    cur = m.get("current") if map_nodes else None
    # Training-time parity: `rm.map_coord()` is None at an act's FIRST map
    # choice, but the DLL reports the pre-act Neow/Ancient node (row 0) as
    # `current` — a row its own act graph never contains (the DLL's node
    # lists start at row 1). Feeding it through made `mapfeat` compute real
    # forward cones where training saw the degenerate act-start encoding,
    # flipping the act-start route on 27 of 42 exactly-reproduced live
    # decisions (2026-08-02 pathing diagnosis, seed 1005: p=[0.51,0.49]
    # native vs [0.00,1.00] bridged).
    if cur is not None and int(cur.get("row", -1)) == 0:
        cur = None
    return {
        "phase": state.get("type"),
        "surface": None,
        "seed": 0,
        "ascension": run_state.get("ascension", 0),
        "deck": [
            (_card_id(c), bool(c.get("upgrade_level")) or bool(c.get("upgraded"))) for c in deck
        ],
        "deck_enchantments": [_card_enchantment(c) for c in deck],
        "relics": [r.get("id") for r in (run_state.get("relics") or [])],
        "potions": [p.get("id") for p in (run_state.get("potions") or []) if not p.get("empty")],
        "map": {"rows": [map_nodes], "boss": m.get("boss")} if map_nodes else None,
        "map_coord": (int(cur["col"]), int(cur["row"])) if cur else None,
        "max_hp": run_state.get("max_hp") or 1,
    }


def _base_metrics(state: dict[str, Any], run_state: dict[str, Any]) -> dict[str, Any]:
    """Common metrics dict (see :func:`ppo.env.metrics_snapshot`).

    ``act_floor`` mirrors the engine's definition (runstate.rs: ``row + 1``
    of the map node the player stands on, 0 at act start), derived from the
    same ``state["map"]["current"]`` coord :func:`_base_obs` uses (live on
    ``map_select``, session-patched elsewhere). Decisions with no map in
    reach (pre-first-map Neow, cache miss) keep the old 0 approximation.
    """
    deck = run_state.get("master_deck") or []
    num_upgraded = sum(1 for c in deck if bool(c.get("upgrade_level")) or bool(c.get("upgraded")))
    potions = run_state.get("potions") or []
    m = state.get("map") or {}
    nodes = m.get("nodes")
    cur = m.get("current") if nodes else None
    # Same act-start guard as `_base_obs`: the pre-act row-0 node is not a
    # map node, and counting it made `act_floor` 1 where training saw 0.
    if cur is not None and int(cur.get("row", -1)) == 0:
        cur = None
    return {
        # Training-time parity (2026-08-02 pathing diagnosis): the engine's
        # `rm.total_floor()` is 0 at Neow (the DLL's `context.total_floor`
        # is 1 there) and `rm.act_index` is 1-BASED. Both fed the encoder
        # off-by-one on EVERY meta decision — 4735/4735 in the v3x sweeps —
        # so the net always believed it was one act earlier and one floor
        # deeper than training ever showed it.
        "total_floor": max(0, int(state.get("floor", 0) or 0) - 1),
        "act_index": int(state.get("act", 1) or 1),
        "act_floor": int(cur["row"]) + 1 if cur else 0,
        "hp": run_state.get("hp", 0),
        "max_hp": run_state.get("max_hp", 0),
        "gold": run_state.get("gold", 0),
        "deck_size": len(deck),
        "num_upgraded": num_upgraded,
        "num_relics": len(run_state.get("relics") or []),
        "num_potions": sum(1 for p in potions if not p.get("empty")),
    }


def _build_map_select(state: dict[str, Any]) -> PpoDecision | None:
    """``map_select`` -> ``ppo.env``'s "path" surface (``move`` actions).

    The bridge's ``map`` object (``PROTOCOL.md``'s "New." field) carries the
    same graph ``ppo.mapfeat.nodes_from_full_map`` consumes -- a flat node
    list with ``children`` edges -- once wrapped in a single-row ``rows``
    list (``nodes_from_full_map`` only flattens ``rows``, it never depends
    on row grouping being meaningful).
    """
    nodes = state.get("nodes") or []
    if not nodes:
        return None
    run_state = state.get("run_state") or {}
    obs = _base_obs(state, run_state)  # map/map_coord from state["map"]
    actions = [{"action": "move", "col": int(n["col"])} for n in nodes]
    responses = [{"action": "choose", "index": int(n.get("index", i))} for i, n in enumerate(nodes)]
    labels = _map_labels(nodes)
    return PpoDecision(obs, actions, _base_metrics(state, run_state), responses, labels)


def _map_labels(nodes: list[dict[str, Any]]) -> list[str]:
    """Name each offered node the way it looks on screen: "Elite (left)".

    The old spelling was the raw graph coordinate (``Elite@(4,1)``), which is
    unreadable next to a map that shows no coordinates at all -- a player
    cannot tell which of the two branches "(4,1)" is (reported 2026-08-05).
    Columns increase left-to-right on the map, so ranking the offers by
    column recovers exactly the words a player would use.
    """
    by_col = sorted(range(len(nodes)), key=lambda i: int(nodes[i].get("col", 0)))
    sides: dict[int, str] = {}
    if len(by_col) == 2:
        sides = {by_col[0]: "left", by_col[1]: "right"}
    elif len(by_col) == 3:
        sides = {by_col[0]: "left", by_col[1]: "middle", by_col[2]: "right"}
    elif len(by_col) > 3:
        sides = {idx: f"#{pos + 1} from left" for pos, idx in enumerate(by_col)}

    labels = []
    for i, n in enumerate(nodes):
        name = _pretty_room_type(str(n.get("type") or "?"))
        side = sides.get(i)
        labels.append(f"{name} ({side})" if side else name)
    return labels


def _pretty_room_type(raw: str) -> str:
    """``RestSite`` -> ``Rest Site`` (the enum name is the only name we get)."""
    out = []
    for i, ch in enumerate(raw):
        if i and ch.isupper() and not raw[i - 1].isupper():
            out.append(" ")
        out.append(ch)
    return "".join(out)


def _build_card_reward(state: dict[str, Any]) -> PpoDecision | None:
    """``card_reward`` -> ``ppo.env``'s "card-pick" surface (``pick_card_reward``)."""
    cards = state.get("cards") or []
    can_skip = bool(state.get("can_skip"))
    if not cards and not can_skip:
        return None
    run_state = state.get("run_state") or {}
    obs = _base_obs(state, run_state)
    obs["card_reward_option_ids"] = [_card_id(c) for c in cards]
    obs["card_reward_option_upgraded"] = [
        bool(c.get("upgrade_level") or c.get("upgraded")) for c in cards
    ]
    obs["card_reward_option_enchantments"] = [_card_enchantment(c) for c in cards]
    actions: list[dict[str, Any]] = []
    responses: list[dict[str, Any]] = []
    labels: list[str] = []
    for i, c in enumerate(cards):
        actions.append({"action": "pick_card_reward", "idx": i})
        responses.append({"action": "choose", "index": i})
        labels.append(str(c.get("name") or c.get("id")))
    if can_skip:
        actions.append({"action": "pick_card_reward", "idx": None})
        responses.append({"action": "skip"})
        labels.append("skip")
    return PpoDecision(obs, actions, _base_metrics(state, run_state), responses, labels)


def _build_boss_relic(state: dict[str, Any]) -> PpoDecision | None:
    """``boss_relic`` -> ``ppo.env``'s "boss-relic" surface (``pick_boss_relic``)."""
    options = [o for o in (state.get("options") or []) if o.get("enabled", True)]
    if not options:
        return None
    run_state = state.get("run_state") or {}
    obs = _base_obs(state, run_state)
    obs["boss_relic_options"] = [(o.get("relic") or {}).get("id") for o in options]
    actions = [{"action": "pick_boss_relic", "idx": i} for i in range(len(options))]
    responses = [
        {"action": "choose", "index": int(o.get("index", i))} for i, o in enumerate(options)
    ]
    labels = [str((o.get("relic") or {}).get("name") or o.get("label")) for o in options]
    return PpoDecision(obs, actions, _base_metrics(state, run_state), responses, labels)


_REST_OPTION_LABELS = ("rest", "smith")


def _build_rest_site(state: dict[str, Any]) -> PpoDecision | None:
    """``rest_site`` -> ``ppo.env``'s "rest" surface (``rest_option``)."""
    options = [o for o in (state.get("options") or []) if o.get("enabled", True)]
    if not options:
        return None
    run_state = state.get("run_state") or {}
    obs = _base_obs(state, run_state)
    obs["rest_option_labels"] = list(_REST_OPTION_LABELS)
    actions = [{"action": "rest_option", "idx": i} for i in range(len(options))]
    responses = [
        {"action": "choose", "index": int(o.get("index", i))} for i, o in enumerate(options)
    ]
    labels = [str(o.get("label") or o.get("id")) for o in options]
    return PpoDecision(obs, actions, _base_metrics(state, run_state), responses, labels)


#: Godot BBCode in bridge option descriptions: ``[img]...[/img]`` blocks
#: (icon references -- drop content too) and plain ``[tag]``/``[/tag]``
#: markers (``[red]6[/red]`` -> ``6``). The native engine's descriptions are
#: clean prose, and ``ppo.encoder.event_payload``'s gold/HP regexes were
#: trained against that -- markup in the middle of ``Lose [red]6[/red] HP``
#: made every live event payload parse to (0, 0) (2026-07-12 audit).
_RE_BBCODE_IMG = re.compile(r"\[img\].*?\[/img\]", re.S)
_RE_BBCODE_TAG = re.compile(r"\[/?[a-zA-Z_][^\[\]]*\]")


def _strip_bbcode(text: str | None) -> str:
    if not text:
        return ""
    return _RE_BBCODE_TAG.sub("", _RE_BBCODE_IMG.sub("", text))


#: Enum-form loc-key tail (``NEOW.pages.INITIAL.options.WINGED_BOOTS`` ->
#: ``WINGED_BOOTS``). Mirrors ``ppo.encoder._RE_ENUM_LABEL``'s shape gate.
_RE_ENUM_TAIL = re.compile(r"^[A-Z0-9_]{3,}$")


def _event_option_label(o: dict[str, Any], event_id: str) -> str:
    """Native-matching option label for one bridge event option.

    The native engine labels NEOW's options with the offered blessing's
    SCREAMING_SNAKE id (``neow_generate_options`` -> ``snake_case_id()``),
    which is exactly the tail of the C# option's ``TextKey`` -- and the PPO
    encoder registry-matches those enum-form labels to relic identity
    (``ppo.encoder._RE_ENUM_LABEL``). Every other event is labeled with
    prose natively, so the substitution is Neow-scoped: swapping prose out
    elsewhere would break ``event_payload``'s gold/hp parsing.
    """
    label = str(_strip_bbcode(o.get("label")) or o.get("label") or "")
    if event_id.upper().replace(".", "_") != "NEOW":
        return label
    tail = str(o.get("text_key") or "").rsplit(".", 1)[-1]
    if _RE_ENUM_TAIL.match(tail):
        return tail
    relic_id = str(o.get("relic_id") or "")
    if _RE_ENUM_TAIL.match(relic_id):
        return relic_id
    return label


#: Act-start Ancient events whose 1-of-3 relic offer IS the act's boss-relic
#: choice (Hive: Orobas/Pael/Tezcatara, decomp Acts/Hive.cs:26-28; Glory:
#: Nonupeipe/Tanx/Vakuu, Acts/Glory.cs:25-27). The real game presents it as a
#: plain event dialog -- the bridge's NChooseARelicSelection handler never
#: fires for it (0 ``boss_relic`` messages across 40 live run logs, 2026-07-15
#: audit) -- but the net only knows relic IDENTITY on the trained
#: ``boss_relic`` surface. Routed via the event surface (identity untrained,
#: 100% OOV) the pick is positional: delegating JUST this decision in the
#: engine-world ablation collapsed the native 10/20 baseline to 3/20.
_BOSS_RELIC_ANCIENTS = frozenset({"OROBAS", "PAEL", "TEZCATARA", "NONUPEIPE", "TANX", "VAKUU"})


def _ancient_boss_relic_decision(state: dict[str, Any]) -> PpoDecision | None:
    """An act-start Ancient's relic-offer event -> the trained boss_relic surface.

    Fires only on the live-bridge shape: every enabled option carries the
    mod's ``event_id`` (raw enum, one of the six act-start Ancients) AND a
    non-empty ``relic_id``. The Ancient's other pages (dialogue, the
    post-pick "Proceed") have empty ``relic_id`` and stay on the event
    surface, as do all ordinary events. Headless states take the equivalent
    ``headless_eval._boss_relic_reroute`` before ever reaching this builder.
    """
    options = [o for o in (state.get("options") or []) if o.get("enabled", True)]
    if len(options) < 2:
        return None
    relic_ids = [str(o.get("relic_id") or "") for o in options]
    if not all(relic_ids):
        return None
    if not all(str(o.get("event_id") or "").upper() in _BOSS_RELIC_ANCIENTS for o in options):
        return None
    synth = {
        **state,
        "options": [
            {
                "index": int(o.get("index", i)),
                "relic": {"id": rid, "name": o.get("label")},
                "enabled": True,
            }
            for i, (o, rid) in enumerate(zip(options, relic_ids))
        ],
    }
    return _build_boss_relic(synth)


#: Per-reason tally of the GH #300 no-suicide guard, the twin of
#: `sts2_runai`'s `EventVetoCounts`. The guard silently overrides a trained
#: policy, so it has to be countable rather than inferred from win-rate
#: noise: a sweep reporting `vetoed: 0` has established the guard never
#: fired, whereas with no counter "working" and "unreachable" look alike.
#: Process-global and monotonic; call `reset_event_vetoes()` per run.
EVENT_VETOES: dict[str, int] = {
    "vetoed": 0,
    "all_lethal": 0,
    # GH #487 off-model-page guard (`_veto_off_model_page`): pages forced to
    # the walk-away option / pages left to the net for want of one.
    "off_model_exit": 0,
    "off_model_no_exit": 0,
}


def reset_event_vetoes() -> None:
    """Zero the veto tally (call at run start)."""
    for key in EVENT_VETOES:
        EVENT_VETOES[key] = 0


#: HP-cost phrase in an event option's own description -- either a literal
#: number ("Take 3 damage, reroll card.") or an unresolved loc placeholder
#: ("Lose [red]{HpLoss}[/red] HP") whose value lives in the option's ``vars``
#: dict. The bridge attaches the EVENT-level DynamicVars to every option
#: (RunSimulator.cs ~3765), so the description is what ties the value to the
#: option that actually pays it.
_RE_EVENT_HP_COST = re.compile(r"(?:lose|take)\s+(?:\{(\w+)\}|(\d+))\s+(?:hp|damage)", re.I)


def _event_option_hp_cost(option: dict[str, Any]) -> int:
    """Best-effort HP an event option costs on pick; 0 when unparseable."""
    m = _RE_EVENT_HP_COST.search(_strip_bbcode(option.get("description")))
    if m is None:
        return 0
    var_name, literal = m.group(1), m.group(2)
    if literal is not None:
        return int(literal)
    try:
        return int((option.get("vars") or {}).get(var_name) or 0)
    except (TypeError, ValueError):
        return 0


def _option_is_lethal(option: dict[str, Any], hp: int) -> bool:
    """Whether picking *option* ends the run, at ``hp`` current HP.

    ONE rule with two sources, and the order is the point: the world's own
    ``will_kill_player`` verdict WINS when it is present, and the HP-cost parse
    is consulted only in its absence.

    An earlier version ran the two as independent filters in sequence, which
    inverted the authority it claimed: the blind regex got the last word and
    could veto an option the world had just said was survivable. That is
    reachable, not theoretical. ``AbyssalBaths`` flags Immerse with
    ``ThatDoesDamage(Damage.IntValue - MaxHp.BaseValue)`` — threshold ``3-2 = 1``
    on the initial page — while its text reads "Gain {MaxHp} Max HP. Take
    {Damage} damage.", so the parse extracts ``3``. At hp in {2,3} the world says
    survivable and the old code withheld it anyway; the window widens on each
    LINGER page as ``Damage`` escalates.

    ``None`` means UNKNOWN, never SAFE — a bridge dialect that predates the field,
    or a probe that threw C#-side. Both fall through to the parse.
    """
    flag = option.get("will_kill_player")
    if flag is not None:
        return bool(flag)
    return hp > 0 and _event_option_hp_cost(option) >= hp


#: Any ``{Name}`` loc placeholder, not just the HP one.
_RE_LOC_VAR = re.compile(r"\{(\w+)\}")


def _resolve_option_vars(text: str, option: dict[str, Any]) -> str:
    """Substitute an option's ``vars`` into its description's placeholders.

    The bridge ships descriptions with the loc template intact ("Lose
    {HpLoss} HP") and the value alongside in ``vars``. ``ppo.encoder.
    event_payload`` regex-scrapes NUMBERS out of the description, so an
    unresolved placeholder reads as no number at all — and the net therefore
    scored SlipperyBridge's "Hold On" as **hp_delta 0.0, i.e. free**, which
    is a large part of why it held a p~0.99 prior for it. The engine arm
    never had this problem: it gets `preview_option_effect`'s exact deltas.

    An unknown placeholder is left as-is rather than blanked, so a
    parse failure still looks like a parse failure downstream.
    """
    variables = option.get("vars") or {}
    if not isinstance(variables, dict):
        return text

    def sub(m: re.Match[str]) -> str:
        value = variables.get(m.group(1))
        return str(value) if value is not None else m.group(0)

    return _RE_LOC_VAR.sub(sub, text)


#: Wire event keys whose first `text_key` segment is not the event's registry
#: name: DollRoom's C# loc entry is MR_STRUGGLES, and TheArchitect's
#: continuation page keys on the bare option verb PROCEED.
_EVENT_KEY_ALIASES = {
    "MR_STRUGGLES": "DOLL_ROOM",
    "PROCEED": "THE_ARCHITECT",
}


def _resolve_event_name(options: list[dict[str, Any]], display_name: str) -> str:
    """Map wire event identity onto the net's event-vocab key.

    ``ppo.vocab.events()`` is keyed on ``EventId::python_name()`` (PascalCase,
    canonicalized to UPPER_SNAKE by ``Registry.index``). The wire gives us
    either the C# loc entry (SCREAMING_SNAKE — SpireMentor's ``event_id``, and
    the first segment of every option ``text_key``) or the English display
    name ("Tablet of Truth" — the headless dialect's ``event_id``). The
    display name canonicalizes with SPACES, never matching the registry, so
    before this resolver 72% of DLL event decisions fed the OOV slot and the
    net was event-blind (it walked Tablet of Truth's Decipher escalation to
    max_hp=1 in 4/60 sweep seeds, all defeats).

    Candidates are tried in wire-authority order and the first one that lands
    in the vocab wins; if none do, the first candidate is returned so a genuine
    unknown still reads as OOV downstream (never-crash contract: vocab import
    failure degrades to the same).
    """
    candidates: list[str] = []

    def add(raw: str | None) -> None:
        s = (raw or "").strip()
        if not s:
            return
        for c in (s, s.removeprefix("THE_"), f"THE_{s}"):
            if c and c not in candidates:
                candidates.append(c)

    for o in options:
        seg = str(o.get("text_key") or "").split(".", 1)[0].strip().upper()
        add(_EVENT_KEY_ALIASES.get(seg, seg))
    snake_display = re.sub(r"[^A-Za-z0-9]+", "_", display_name).strip("_").upper()
    add(_EVENT_KEY_ALIASES.get(snake_display, snake_display))
    if not candidates:
        return display_name
    try:
        from ppo import vocab

        registry = vocab.events()
        for c in candidates:
            if registry.index(c) != registry.oov:
                return c
    except Exception:  # degrade to today's OOV behavior
        pass
    return candidates[0]


#: Events whose LATER pages the engine actually models (`event_state.page`
#: progressions landed in sts2-content): their page-2+ states existed in
#: training, so the net's ranking there is a real opinion. Every other
#: multi-page event is flattened to its first page (`6ynt.4.multi-page`
#: debt), and a live page-2+ state hands the net an event id whose learned
#: value belongs to a DIFFERENT, cheaper decision. GH #487: the net walked
#: Tablet of Truth's Decipher escalation to 1/1 max HP at p=94% -- the last
#: page's real price (MaxHp-1) never entered any input it was trained on.
#: Keyed on `_resolve_event_name` output (registry UPPER_SNAKE, both
#: THE_-prefix spellings where the registry name is ambiguous). NEOW is
#: listed not as multi-page but because its pages are a bespoke surface
#: (`neow_blessing.rs`) this guard must never touch.
#:
#: TABLET_OF_TRUTH is DELIBERATELY absent even though `sts2-content` now
#: models its 3→6→12→24→(MaxHp-1) escalation for real (GH #487, fix
#: direction 1). Membership here is a claim about the NET, not about the
#: engine: it says "page 2+ of this event existed in the training
#: distribution, so the net's ranking there is a real opinion". Every
#: champion on `champions.toml` today was trained against the flattened
#: single-page port, so its opinion of page 4 is still an opinion about a
#: 3-Max-HP deal. Move it in only alongside a champion trained after the
#: escalation landed.
_ENGINE_MODELED_MULTIPAGE = {
    "COLOSSAL_FLOWER",
    "DOLL_ROOM",
    "ROUND_TEA_PARTY",
    "SLIPPERY_BRIDGE",
    "THE_LANTERN_KEY",
    "LANTERN_KEY",
    "TINKER_TIME",
    "NEOW",
}

#: The page segment of a live option's loc key
#: (``TABLET_OF_TRUTH.pages.DECIPHER_2.options.DECIPHER`` -> ``DECIPHER_2``).
_RE_OPTION_PAGE = re.compile(r"\.pages\.([A-Za-z0-9_]+)\.options\.")

#: Option-id tails that read as "walk away with what you already have".
#: ``EXIT\w*`` covers compounds like AbyssalBaths' EXIT_BATHS.
_RE_EXIT_OPTION = re.compile(
    r"^(?:GIVE_UP|LEAVE|EXIT\w*|WALK_AWAY|ABSTAIN|DECLINE|REFUSE|IGNORE|ESCAPE|FLEE|GO_BACK)$"
)


def _is_exit_option(option: dict[str, Any]) -> bool:
    """Whether *option* is the page's walk-away pick, by loc-key tail."""
    text_key = str(option.get("text_key") or "")
    tail = text_key.rsplit(".options.", 1)[-1] if ".options." in text_key else ""
    if not tail:
        tail = re.sub(
            r"[^A-Za-z0-9]+", "_", str(option.get("label") or option.get("title") or "")
        ).strip("_")
    return bool(_RE_EXIT_OPTION.match(tail.upper()))


def _veto_off_model_page(
    options: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    """GH #487: on an event page the engine never models, don't trust the net.

    The cheap stopgap in front of `6ynt.4.multi-page`. A page is "page 2+"
    when NO option's loc key says ``pages.INITIAL`` -- ``INITIAL`` is the
    first page across the whole v0.107.1 roster, and the shared ``pages.ALL``
    options only stand ALONE on later pages (AbyssalBaths' LINGER loop),
    while EndlessConveyor's first page mixes INITIAL and ALL keys. So "some
    option says non-INITIAL" would misfire and this test does not.

    When the guard fires, the walk-away option is taken instead of the net's
    ranking: a bounded small loss (forgoing a possibly-fine deal) instead of
    an unbounded one. A page with no recognizable walk-away pick is left to
    the net -- its outcomes are already bounded by the lethal filter, which
    must run BEFORE this (on Trial's REJECT page the only exit-shaped option
    is Double Down, which the world flags WillKillPlayer) -- but counted, so
    a sweep can tell "never fired" from "unreachable".

    Returns:
        ``(options, source_note)`` -- the surviving options, and the note
        naming the override when it fired (empty otherwise).
    """
    pages = {
        m.group(1)
        for o in options
        for m in [_RE_OPTION_PAGE.search(str(o.get("text_key") or ""))]
        if m
    }
    if not pages or "INITIAL" in pages:
        return options, ""
    event_id = next((o.get("event_id") for o in options if o.get("event_id")), "")
    name = _resolve_event_name(options, str(event_id or ""))
    if name.upper() in _ENGINE_MODELED_MULTIPAGE:
        return options, ""
    exits = [o for o in options if _is_exit_option(o)]
    if not exits:
        EVENT_VETOES["off_model_no_exit"] += 1
        return options, ""
    EVENT_VETOES["off_model_exit"] += 1
    page = "/".join(sorted(pages))
    return exits[:1], (
        f"page {page} of {name} has no engine model (GH #487) -- took the walk-away option"
    )


#: The structured per-option deltas GH #237 put on the wire. The first three
#: must all be present for ``ppo.encoder.structured_event_payload`` to use
#: them; a partial set falls back to the description regex rather than letting
#: an absent field read as a confident zero. The non-gold, non-HP counts are
#: read one by one (absent = 0), as in ``wire.rs``'s ``build_event``.
_STRUCTURED_DELTA_KEYS = (
    "gold_delta",
    "hp_delta",
    "max_hp_delta",
    "relics_gained",
    "relics_lost",
    "potions_gained",
    "potions_lost",
    "cards_gained",
    "cards_lost",
)


def _build_event(state: dict[str, Any]) -> PpoDecision | None:
    """``event`` -> ``ppo.env``'s "event" surface (``event_choice``)."""
    boss_relic = _ancient_boss_relic_decision(state)
    if boss_relic is not None:
        return boss_relic
    all_options = state.get("options") or []
    options = [o for o in all_options if o.get("enabled", True)]
    if not options:
        return None
    run_state = state.get("run_state") or {}
    # GH #300 no-suicide guard, the counterpart of the engine-side
    # `veto_lethal_event_choice` (sts2-runai policy/mod.rs). The event-blind
    # PPO (event vocab 100% OOV) holds a p~0.99 prior for SlipperyBridge's
    # "Hold On" learned against the old single-step port; against the
    # faithful 3+N escalation it loops until death. Withhold lethal options
    # -- unless every option is, in which case the pick stays the policy's to
    # make. That "unless" is not a softening: it is the same rule the shipped
    # game's own auto-player uses (AutoSlay.EventRoomHandler filters
    # `o.Option.WillKillPlayer(o.Event.Owner)` and keeps the filtered list only
    # `if (list3.Count > 0)`).
    #
    # Two sources, checked in order of authority:
    #
    #   1. `will_kill_player` -- `EventOption.WillKillPlayer` evaluated by the
    #      C# world itself. Authoritative, and the ONLY thing that catches
    #      non-damage lethality: Trial.DoubleDown is
    #      `ThatWillKillPlayerIf(_ => true)` (it opens the abandon-run confirm,
    #      so live it ENDS the run) and TabletOfTruth.Decipher keys on MAX HP.
    #      Neither prints an HP cost for a description parse to find.
    #   2. the parsed HP cost -- retained for bridges that don't send the flag
    #      (older real-game bridge dialects, replayed fixtures), where its
    #      absence must read as "unknown", never as "safe".
    #
    # The two ARMS are NOT the same intervention, and an earlier version of
    # this comment claimed they were. Rust vetoes AFTER inference, replacing
    # the policy's pick among survivors; Python filters BEFORE inference, so
    # the net never scores the lethal option and its ranking over the rest is
    # preserved. They agree whenever exactly one option survives (every case
    # in the current content) and can differ on a 3+-option event, where Rust
    # takes the smallest-HP-loss survivor and Python takes whichever survivor
    # the net ranks highest. Python's ordering is the better one; the Rust
    # arm cannot adopt it because at that point the meta policy has already
    # committed to a single index.
    #
    # Note the degenerate case this shares with any filter: when exactly one
    # option survives, the decision is forced and the logged probability is
    # 1.0 by construction, not by the net's confidence.
    hp = int(run_state.get("hp") or 0)
    survivors = [o for o in options if not _option_is_lethal(o, hp)]
    if survivors:
        if len(survivors) != len(options):
            EVENT_VETOES["vetoed"] += 1
        options = survivors
    else:
        EVENT_VETOES["all_lethal"] += 1
    # GH #487 off-model-page guard -- AFTER the lethal filter, deliberately
    # (see `_veto_off_model_page` on Trial's REJECT page).
    options, off_model_note = _veto_off_model_page(options)
    # A disabled option keeps its slot, as in the engine's option list: the
    # net reads an option by POSITION, so dropping one would shift every later
    # option onto another's slot (the Fake Merchant's throw/leave, GH #873).
    # It gets no action, and is never the pick.
    kept = {id(o) for o in options}
    options = [o for o in all_options if not o.get("enabled", True) or id(o) in kept]
    choices = [i for i, o in enumerate(options) if o.get("enabled", True)]
    obs = _base_obs(state, run_state)
    event_id = next((o.get("event_id") for o in options if o.get("event_id")), "")
    obs["event_name"] = _resolve_event_name(options, str(event_id or ""))
    option_labels = [_event_option_label(o, str(event_id or "")) for o in options]
    obs["event_options"] = [
        {
            "label": label,
            # Placeholders resolved BEFORE the encoder sees the text: it
            # scrapes numbers out of this string, and an unresolved
            # `{HpLoss}` reads as no cost at all.
            "description": _resolve_option_vars(_strip_bbcode(o.get("description")), o),
            "enabled": o.get("enabled", True),
            # Pass the engine's exact deltas through when the world supplies
            # them (GH #237); the encoder prefers these over its regex and
            # ignores a partial set.
            **{k: o[k] for k in _STRUCTURED_DELTA_KEYS if o.get(k) is not None},
        }
        for o, label in zip(options, option_labels)
    ]
    actions = [{"action": "event_choice", "idx": i} for i in choices]
    responses = [{"action": "choose", "index": int(options[i].get("index", i))} for i in choices]
    return PpoDecision(
        obs,
        actions,
        _base_metrics(state, run_state),
        responses,
        [option_labels[i] for i in choices],
        source_note=off_model_note,
    )


def _build_shop(state: dict[str, Any]) -> PpoDecision | None:
    """``shop`` -> ``ppo.env``'s "shop" surface.

    ``shop_buy_card``/``shop_buy_relic``/``shop_buy_potion`` need ``idx``
    into per-kind lists (``ctx.shop_cards``/``shop_relics``/
    ``shop_potions``), not the bridge's flat ``options`` index -- built
    here in the same pass. ``remove_card``/``buy_item`` (and anything else
    unrecognized) degrade to a generic action kind the encoder turns into a
    FIXED "none" token (never a crash, never silently dropped from the
    legal set -- it's just a token the net can't meaningfully score).
    """
    options = [o for o in (state.get("options") or []) if o.get("enabled", True)]
    if not options:
        return None
    run_state = state.get("run_state") or {}
    obs = _base_obs(state, run_state)

    # A full potion belt makes buy_potion a guaranteed no-op in the real game
    # (PotionCmd.TryToProcure -> TooFull, no gold spent): the shop re-presents
    # unchanged and the deterministic policy would pick the same potion forever
    # (until SpireMentor's 50-attempt cap force-leaves the shop). The bridge now
    # withholds potion slots too; this filter covers older mod builds.
    belt = run_state.get("potions") or []
    belt_full = bool(belt) and not any(p.get("empty") for p in belt)

    shop_cards: list[dict[str, Any]] = []
    shop_relics: list[dict[str, Any]] = []
    shop_potions: list[dict[str, Any]] = []

    actions: list[dict[str, Any]] = []
    responses: list[dict[str, Any]] = []
    labels: list[str] = []

    deck_entries = run_state.get("master_deck") or []
    deck = [(c.get("id"), bool(c.get("upgrade_level") or c.get("upgraded"))) for c in deck_entries]
    # Positionally parallel to `deck`; see `_wire_card_is_removable`.
    deck_removable = [_wire_card_is_removable(c) for c in deck_entries]
    for i, o in enumerate(options):
        kind = o.get("action")
        bridge_index = int(o.get("index", i))
        item = o.get("item") or {}
        price = o.get("price", 0)
        if kind == "buy_card":
            shop_cards.append({"id": _card_id(item), "price": price, "available": True})
            actions.append({"action": "shop_buy_card", "idx": len(shop_cards) - 1})
        elif kind == "buy_relic":
            shop_relics.append({"id": item.get("id"), "price": price, "available": True})
            actions.append({"action": "shop_buy_relic", "idx": len(shop_relics) - 1})
        elif kind == "buy_potion":
            if belt_full:
                continue
            shop_potions.append({"id": item.get("id"), "price": price, "available": True})
            actions.append({"action": "shop_buy_potion", "idx": len(shop_potions) - 1})
        elif kind == "leave_shop":
            actions.append({"action": "leave_shop"})
        elif kind == "remove_card" and deck:
            # The net was trained on N per-deck-card `shop_remove_card`
            # actions (deck_idx), not one opaque "removal service" button
            # (which previously encoded as a FIXED "none" token the net
            # can't value). Expand to the training shape; every candidate
            # answers the bridge with the same remove_card option, and the
            # follow-up card_select (prev=shop) honors the deck_idx chosen
            # HERE (session["shop_remove_target"], falling back to the
            # remove_deck_card surface). The candidate set is the
            # engine's own (`_non_removable_cards`): Eternal cards out,
            # basic Strike/Defend IN -- matching `Phase::Shop`, which is
            # the surface these nets were trained against.
            for deck_idx in range(len(deck)):
                if not deck_removable[deck_idx]:
                    continue
                actions.append({"action": "shop_remove_card", "deck_idx": deck_idx})
                responses.append({"action": "choose", "index": bridge_index})
                labels.append(f"remove:{deck[deck_idx][0]}")
            continue
        else:
            actions.append({"action": f"shop_other_{kind}"})
        responses.append({"action": "choose", "index": bridge_index})
        labels.append(str(o.get("label") or kind))

    removal_cost = state.get("purge_cost")
    if removal_cost is None:
        removal_cost = next(
            (o.get("price", 0) for o in options if o.get("action") == "remove_card"), 0
        )
    obs["shop"] = {
        "cards": shop_cards,
        "relics": shop_relics,
        "potions": shop_potions,
        "removal_cost": removal_cost,
    }
    return PpoDecision(obs, actions, _base_metrics(state, run_state), responses, labels)


#: card_select context (the PRECEDING decision message's type, tracked in
#: ``session["prev_msg_type"]`` by the session loops) -> the ppo.env deck-
#: mutation surface it corresponds to. The bridge's ``card_select`` message
#: itself carries NO source info (just cards + min/max/cancelable), so the
#: previous screen is the only signal for WHAT the pick means:
#: rest_site -> smith upgrade target, shop -> removal target, event -> card
#: grant ("choose 1 of N to add"). Anything else (notably combat_action --
#: Armaments/Warcry/discard/exhaust targets, a separate work item) still
#: falls through to the random-legal picker.
_CARD_SELECT_SURFACES = {
    "rest_site": "smith",
    "shop": "removal",
    "event": "grant",
}


#: ``prompt_key`` (upper-cased) -> the ppo.env surface that ranks it.
#:
#: The five ``TO_*`` keys are ``CardSelectorPrefs``' shared prompts. The rest
#: are per-source prompts (``"<ID>.selectionScreenPrompt"``), and they are here
#: because a source that ships its own prompt string has NO shared key, so it
#: matched nothing and fell all the way to ``_unranked``'s leftmost default --
#: `SEA_GLASS` and both event prompts did exactly that in
#: `logs/f7-mirror-check18`.
#:
#: The per-source set is a STATIC AUDIT, not a sample: every
#: ``*.selectionScreenPrompt`` in `localization_eng` is 39 strings, and all but
#: these four are IN-COMBAT hand/draw/discard operations that the combat MCTS
#: answers on the fight's mirror (`realgame_mcts.pick_card_select_mirrored`).
#: `test_every_non_combat_selection_prompt_has_a_surface` re-derives that from
#: the loc files, so a new one fails the gate instead of silently going
#: leftmost.
#:
#: Each entry is the decompiled effect, not the prompt's wording:
#:   * BRAIN_LEECH  `FromSimpleGridForRewards(prefs(1))` -> `Add(card, Deck)`
#:     (BrainLeech.cs:66-73) -- 1 of 5 generated cards.
#:   * ROOM_FULL_OF_CHEESE  `FromSimpleGridForRewards(prefs(2))` ->
#:     `Add(item, Deck)` (RoomFullOfCheese.cs:37-46) -- 2 of 8 commons.
#:   * SEA_GLASS  `FromSimpleGridForRewards(prefs(0, list.Count))` ->
#:     `Add(item, Deck)` (SeaGlass.cs:74-91).
#:   * CLAWS  `FromDeckForTransformation` -> `CardCmd.Transform(..)` into Maul
#:     (Claws.cs:22-29) -- master-deck cards, which is the `transform`
#:     surface's own shape.
#:
#: DOLLYS_MIRROR is deliberately ABSENT. It also adds to the deck, but its
#: options are the player's OWN deck (`FromDeckGeneric`, DollysMirror.cs:16-22)
#: and it adds a COPY, while the grant head was trained on cards being offered.
#: Whether that extrapolates is untested, and a confident wrong ranking is
#: worse than an admitted arbitrary one.
_PROMPT_SURFACES: dict[str, str] = {
    "TO_REMOVE": "removal",
    "TO_TRANSFORM": "transform",
    "TO_ENCHANT": "enchant",
    "TO_UPGRADE": "smith",
    "TO_OBTAIN": "grant",
    "BRAIN_LEECH.PAGES.SHARE_KNOWLEDGE.SELECTIONSCREENPROMPT": "grant",
    "ROOM_FULL_OF_CHEESE.PAGES.GORGE.SELECTIONSCREENPROMPT": "grant",
    "SEA_GLASS.SELECTIONSCREENPROMPT": "grant",
    "CLAWS.SELECTIONSCREENPROMPT": "transform",
}


def _build_card_select(state: dict[str, Any], prev_msg_type: str | None) -> PpoDecision | None:
    """``card_select`` -> the matching ppo.env deck-mutation surface.

    2026-07-12 root-cause finding: these picks previously ALL fell to
    ``realgame_random.pick_card_select`` (uniform random), replacing the
    trained net's entire deck-curation policy (which card to upgrade at a
    rest site, which to remove at a shop, which event card to take) with
    coin flips -- a top contributor to the live 0% win-rate vs 90% native.

    Only single-pick selections are handled (``min/max_select`` <= 1 or
    absent); multi-picks (combat discard-2 etc.) return ``None`` and keep
    the random fallback. Returns ``None`` when ``prev_msg_type`` has no
    mapped surface.
    """
    # The screen's own `prompt_key` names the surface authoritatively; the
    # previous-message heuristic misfiled every event-context screen as a
    # GRANT — Neow's Precise Scissors removal picked the card the net most
    # wanted to KEEP (removed Bash, p=0.67, in 8 of 60 v3x-sweep seeds; 31
    # sign-inverted deck mutations across 24 seeds). 2026-08-02 pathing
    # diagnosis, defect D4.
    prompt = str(state.get("prompt_key") or "").upper()
    surface = _PROMPT_SURFACES.get(prompt) or _CARD_SELECT_SURFACES.get(prev_msg_type or "")
    if surface is None:
        return None
    cards = state.get("cards") or []
    if not cards:
        return None
    min_n = state.get("min_count", state.get("min_select"))
    max_n = state.get("max_count", state.get("max_select"))
    # A multi-pick deck mutation ("remove 2 cards") is still a ranking over
    # single cards -- the arity says how many of them the answer names, not
    # that the surface is different. Bailing here sent every one of them to
    # `realgame_random.pick_card_select`, i.e. uniform random, which is what
    # decided 9 of the 19 unsearched selections in `logs/f7-mirror-check18`
    # (6 TO_REMOVE, 3 TO_TRANSFORM -- Empty Cage and Amalgamator).
    #
    # The two lanes need different answers to the same screen, so `pick_count`
    # carries the arity instead of the reply shape deciding it:
    #   * the LIVE mod re-asks once per pick with the option list one card
    #     shorter and min/max unchanged (verified in the check18 workbench
    #     logs: TO_REMOVE min=2 max=2 cards=22, then cards=21), so the
    #     advisor ranks the shrinking list and recommends one card each time;
    #   * `SimCardSelector` rejects a reply naming fewer than `minSelect`
    #     (SimCardSelector.cs:368), so the driver has to answer all of them
    #     at once -- see the top-k fill in `pick_noncombat_action_ppo`.
    #
    # Honest about what the top-k is: one inference ranked once, not the net
    # re-consulted with the first pick already removed from the deck. For
    # "remove 2 of 22" that is a close approximation and unarguably better
    # than a coin flip; it is not the same thing.
    pick_count = 1
    if max_n is not None and int(max_n) > 1:
        pick_count = max(1, int(min_n or 1))
    run_state = state.get("run_state") or {}

    obs = _base_obs(state, run_state)

    # Response spelling mirrors realgame_random.pick_card_select: indices
    # list when the message carries min/max bounds, bare index otherwise.
    def _response(bridge_index: int) -> dict[str, Any]:
        if min_n is None and max_n is None:
            return {"action": "choose", "index": bridge_index}
        return {"action": "choose", "indices": [bridge_index]}

    actions: list[dict[str, Any]] = []
    responses: list[dict[str, Any]] = []
    labels: list[str] = []
    if surface == "smith":
        # ppo.env "card-pick": obs.smith_options = [card_id, ...] (plain ids,
        # rm.smith_options() shape), action smith_card idx-aligned.
        obs["smith_options"] = [_card_id(c) for c in cards]
        obs["smith_option_enchantments"] = [_card_enchantment(c) for c in cards]
        kind = "smith_card"
    elif surface == "removal":
        # obs.card_removal_options = [(card_id, upgraded), ...].
        obs["card_removal_options"] = [
            (_card_id(c), bool(c.get("upgrade_level") or c.get("upgraded"))) for c in cards
        ]
        obs["card_removal_option_enchantments"] = [_card_enchantment(c) for c in cards]
        kind = "remove_deck_card"
    elif surface in ("transform", "enchant"):
        # These two are shaped differently from every other surface here, and
        # the difference is the whole reason GH #477 called them out.
        #
        # `remove`/`grant`/`smith` carry the offered cards' IDENTITIES straight
        # through. `transform`/`enchant` instead carry MASTER-DECK INDICES:
        # `encoder._Ctx` reads `transform_select_deck_indices`, and
        # `_build_action_token` dereferences `transform_idxs[idx]` into
        # `ctx.deck` to build the token. The bridge's own `index` field is the
        # position in THIS selection's option list (PROTOCOL.md §3
        # `card_select`), not a deck position, so the deck position has to be
        # recovered here.
        #
        # Duplicate card ids look like they make that ambiguous. They do not,
        # for this purpose: the index is ONLY ever dereferenced to read
        # `(card_id, upgraded)` and the enchantment back out, so any deck
        # position holding the same triple produces an identical token. Two
        # plain Strikes are interchangeable to the encoder; a Sharp Strike and
        # a plain one are not under `card_mods` (GH #947), hence the triple.
        # What has to be exact is the RESPONSE, and that uses the option-list
        # index like every other surface below.
        #
        # A card the master deck does not contain (shouldn't happen — these
        # screens offer from the deck) falls to an out-of-range index, which
        # `encoder._at` degrades to the unknown-card token rather than raising.
        # That costs the net one option's identity; it does not break the
        # never-crash contract.
        deck_triples = list(zip(obs["deck"], obs["deck_enchantments"], strict=True))
        missing = len(deck_triples)
        idxs: list[int] = []
        for c in cards:
            pair = (
                _card_id(c),
                bool(c.get("upgrade_level")) or bool(c.get("upgraded")),
            )
            triple = (pair, _card_enchantment(c))
            if triple in deck_triples:
                idxs.append(deck_triples.index(triple))
            else:
                # A screen whose option omits the enchantment its deck copy
                # carries still names that card: fall back to the pair match
                # the pre-#947 code used rather than dropping its identity.
                pairs = [p for p, _ in deck_triples]
                idxs.append(pairs.index(pair) if pair in pairs else missing)
        if surface == "transform":
            obs["transform_select_deck_indices"] = idxs
            kind = "transform_deck_card"
        else:
            obs["enchant_select_deck_indices"] = idxs
            # GH #545: unlike ppo/env.py, this driver has no live `rm` --
            # everything here is rebuilt from the bridge's wire JSON (see
            # module docstring), and PROTOCOL.md's card_select message never
            # names WHICH enchantment a TO_ENCHANT pick would stamp (the
            # per-card `enchantment` field on `cards` describes an ALREADY
            # enchanted card, not the pending one). Ship the key anyway so
            # the encoder reads a consistent "unknown" pad row over the
            # bridge rather than a missing dict key.
            obs["pending_enchantment"] = None
            kind = "enchant_deck_card"
    else:  # grant
        # obs.card_grant_options = [(card_id, upgraded), ...].
        obs["card_grant_options"] = [
            (_card_id(c), bool(c.get("upgrade_level") or c.get("upgraded"))) for c in cards
        ]
        obs["card_grant_option_enchantments"] = [_card_enchantment(c) for c in cards]
        kind = "grant_deck_card"
    for i, c in enumerate(cards):
        actions.append({"action": kind, "idx": i})
        responses.append(_response(int(c.get("index", i))))
        labels.append(str(c.get("name") or c.get("id")))
    return PpoDecision(
        obs,
        actions,
        _base_metrics(state, run_state),
        responses,
        labels,
        pick_count=min(pick_count, len(cards)),
    )


#: Bridge message ``type`` -> builder. Anything else (``card_bundle``,
#: ``crystal_sphere``, ``treasure``, ``run_config``) is not (yet) covered --
#: see module docstring -- and goes straight to
#: :func:`realgame_random.choose_action`. (``card_select`` and
#: ``reward_screen`` are special-cased in :func:`pick_noncombat_action_ppo`:
#: the former needs the session's ``prev_msg_type`` context, the latter the
#: shared multi-claim collector.)
_PPO_NONCOMBAT_BUILDERS: dict[str, Callable[[dict[str, Any]], PpoDecision | None]] = {
    "map_select": _build_map_select,
    "card_reward": _build_card_reward,
    "shop": _build_shop,
    "rest_site": _build_rest_site,
    "event": _build_event,
    "boss_relic": _build_boss_relic,
}


# ---------------------------------------------------------------------------
# Policy loading (reuses workbench.ppo_policy.ckpt_path's resolution rule).
# ---------------------------------------------------------------------------


#: Extensions that mean "torch checkpoint" for an explicit ``--model``. The
#: set is closed and an unknown suffix is an error, not a guess -- see
#: :func:`_load_explicit_model`.
_TORCH_MODEL_SUFFIXES = (".pt", ".pth", ".ckpt", ".bin", ".tar")


def _load_torch(path: str) -> tuple[Any, str | None]:
    try:
        from ppo.net_policy import NetPolicy  # ImportError in the frozen exe -> degrade

        return NetPolicy(path), None
    except Exception as e:  # corrupt/incompatible checkpoint -- degrade, never crash
        return None, f"failed to load PPO checkpoint {path!r}: {e}"


#: Set to ``0``/``false`` to score ONNX champions with onnxruntime instead of
#: the Rust ``tract`` backend. The Rust path is the default because it is what
#: lets the shipped advisor drop `numpy` + `onnxruntime` (~54 MB); this escape
#: hatch exists so the two can be compared on one machine
#: (`diag_ppo_policy_parity.py`) and so a suspected inference difference has a
#: one-variable A/B rather than a rebuild.
_RUST_BACKEND_ENV = "STS2_PPO_RUST_BACKEND"


def _use_rust_backend() -> bool:
    return (os.environ.get(_RUST_BACKEND_ENV, "1") or "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )


def _load_onnx(path: str) -> tuple[Any, str | None]:
    """Load an exported champion, Rust-first, onnxruntime as the fallback.

    Both backends run the SAME graph and are pinned to each other:
    ``ppo_encoder_parity`` fixes Rust featurization against
    :mod:`ppo.encoder`, ``ppo_net_parity`` fixes tract against torch, and
    ``diag_ppo_policy_parity.py`` replays recorded decisions through both
    ends of this function. So the choice is a packaging decision, not a
    behavioral one.

    A Rust load failure degrades to onnxruntime rather than to *no policy*:
    in a dev venv both are present, and losing the champion over a stale
    wheel would silently turn a measured lane into the random-legal one.
    Inside the frozen exe there is no onnxruntime to fall back to, and the
    error text says which backend failed.
    """
    if _use_rust_backend():
        from ppo import rust_policy

        if not rust_policy.available():
            logger.warning("%s; using onnxruntime", rust_policy.describe_unavailable())
        else:
            try:
                return rust_policy.RustPolicy(path), None
            except Exception as e:
                logger.warning(
                    "Rust PPO backend failed to load %r: %s; trying onnxruntime", path, e
                )
    try:
        from ppo.onnx_policy import OnnxPolicy

        return OnnxPolicy(path), None
    except Exception as e:  # corrupt/incompatible model -- degrade, never crash
        return None, f"failed to load PPO model {path!r}: {e}"


def _load_explicit_model(model_path: str) -> tuple[Any, str | None]:
    """Backend for an explicit ``--model``, chosen by extension.

    Both suffix sets are closed and an unrecognized one is an error rather
    than a guess. Testing only for a literal ``.pt`` routed every other torch
    spelling (``.pth``, ``.ckpt``, an extensionless wandb artifact) into
    onnxruntime, which raises ``InvalidProtobuf`` -- and a driver lane answers
    a load failure with the random-legal policy, so the sweep still finished
    and still reported a win rate for a checkpoint that was never loaded.
    """
    suffix = Path(model_path).suffix.lower()
    if suffix in _TORCH_MODEL_SUFFIXES:
        return _load_torch(model_path)
    if suffix == ".onnx":
        return _load_onnx(model_path)
    return None, (
        f"cannot tell which backend {model_path!r} needs -- name a torch checkpoint "
        f"({'/'.join(_TORCH_MODEL_SUFFIXES)}) or an ONNX export (.onnx)"
    )


def load_ppo_policy(model_path: str | None, character: str) -> tuple[Any, str | None]:
    """Load the run policy for ``character``, never raising (GH #380).

    Default backend is :class:`ppo.rust_policy.RustPolicy` over the
    committed ONNX champion (``workbench.ppo_policy.onnx_path``, which
    honors ``STS2_WB_PPO_ONNX``), falling back to the onnxruntime twin
    :class:`ppo.onnx_policy.OnnxPolicy` -- see :func:`_load_onnx`. Both run
    the same graph; scoring it through ``sts2_rs`` is what lets the packaged
    exe ship neither `numpy` nor `onnxruntime` (the advisor already loads
    ``sts2_rs`` for combat MCTS). Torch ``NetPolicy`` remains the dev path,
    reached three ways:
    an explicit ``--model`` naming a checkpoint extension, an explicitly set
    ``STS2_WB_PPO_CKPT``, or as the fallback when no ONNX champion resolves
    (in the frozen exe the torch import fails and this degrades to
    ``(None, error)``).

    ``STS2_WB_PPO_CKPT`` is checked BEFORE the ONNX default, not after: it is
    the documented way to score a not-yet-promoted candidate, and resolving
    ONNX first made it silently inert for every character that has a
    committed champion -- i.e. all of them -- so an eval credited to the
    candidate actually measured the old champion.

    Returns:
        ``(policy, error)`` -- ``policy`` is ``None`` with ``error`` set to
        a human-readable reason when no model could be resolved or loaded
        (e.g. a character with no entry in ``data/ppo/champions.toml``).
        Callers must fall back to the random-legal policy when
        ``policy is None``.
    """
    from workbench.ppo_policy import ckpt_path, onnx_path

    if model_path:  # explicit --model: the extension picks the backend
        return _load_explicit_model(model_path)

    ckpt_env = os.environ.get("STS2_WB_PPO_CKPT")
    if ckpt_env:
        pt = ckpt_path(character)
        if pt is None:
            return None, f"STS2_WB_PPO_CKPT={ckpt_env!r} does not exist"
        return _load_torch(pt)

    onnx_error: str | None = None
    # The manifest resolvers raise FileNotFoundError when champions.toml
    # names a file absent from the tree (deliberately loud for the
    # workbench); here that must degrade to (None, error) per the contract.
    try:
        path = onnx_path(character)
    except FileNotFoundError as e:
        return None, str(e)
    if path is not None:
        policy, err = _load_onnx(path)
        if policy is not None:
            return policy, None
        onnx_error = f"failed to load ONNX champion {path!r}: {err}"
        logger.warning("%s; trying torch checkpoint", onnx_error)

    try:
        pt = ckpt_path(character)
    except FileNotFoundError as e:
        # A packaged install ships the .onnx but never the .pt the manifest
        # declares -- if the ONNX load failed, THAT is the answer.
        return None, onnx_error if onnx_error is not None else str(e)
    if pt is None:
        # When the ONNX champion resolved and only its LOAD failed, that is the
        # answer: a packaged install ships no .pt and no torch, so "no PPO
        # model" would be the only message an onnxruntime failure could ever
        # produce -- while the .onnx sits in _internal/data/ppo.
        if onnx_error is not None:
            return None, onnx_error
        return None, (
            f"no PPO model for character={character!r} -- pass --model PATH, set "
            f"STS2_WB_PPO_ONNX/STS2_WB_PPO_CKPT, or commit "
            f"data/ppo/ppo_fullrun_<character>_it<N>.onnx"
        )
    return _load_torch(pt)


class PpoPolicyCache:
    """Per-character lazy run-policy cache (``OnnxPolicy``/``NetPolicy``) for one session.

    A live run's character is only known once the first decision carrying
    ``run_state.character`` arrives (or from ``--character`` before that) --
    mirrors :mod:`workbench.ppo_policy`'s per-character cache, kept local to
    this module rather than reaching into the workbench's private cache
    (only its public :func:`workbench.ppo_policy.ckpt_path` is reused).
    """

    def __init__(self, model_path: str | None, *, strict: bool = True) -> None:
        """``strict`` (default) makes an explicit ``--model`` that fails to
        load raise instead of degrading to the random-legal policy.

        A named checkpoint is a measurement instruction, not a preference:
        degrading still finishes the sweep and still reports a win rate, one
        the named checkpoint never produced. The advisor lane passes
        ``strict=False`` -- there, advice must render whatever happens.
        """
        self._model_path = model_path
        self._strict = strict
        self._cache: dict[str, tuple[Any, str | None]] = {}
        self._warned: set[str] = set()

    def get(self, character: str) -> tuple[Any, str | None]:
        key = (character or "").strip().lower() or "ironclad"
        if key not in self._cache:
            self._cache[key] = load_ppo_policy(self._model_path, character)
        policy, error = self._cache[key]
        # Strict is enforced on EVERY get, not just the populating one: the
        # sweep drivers catch per-run exceptions and keep going, so a raise
        # that fired only on the cache miss would abort run 1 and silently
        # score runs 2..N with the random-legal policy.
        if policy is None and self._strict and self._model_path is not None:
            raise RuntimeError(
                f"--model {self._model_path!r} could not be loaded for "
                f"character={character!r}: {error}"
            )
        if policy is None and key not in self._warned:
            self._warned.add(key)
            logger.error(
                "PPO policy unavailable for character=%r: %s -- every non-combat "
                "decision for this character will use the random-legal policy.",
                character,
                error,
            )
        return self._cache[key]


# ---------------------------------------------------------------------------
# Action selection.
# ---------------------------------------------------------------------------


def _select_index(
    probs: list[float],
    temperature: float,
    rng: random.Random,
    actions: list[dict[str, Any]] | None = None,
) -> int:
    """Greedy argmax (``temperature <= 0``) or power-law re-tempered sampling.

    See module docstring's ``--temperature`` note: :meth:`ppo.net_policy.
    NetPolicy.distribution` only exposes the already-softmaxed
    probabilities, not raw logits, so ``T != 1`` sampling here is an
    approximation (``p_i ** (1/T)``, renormalized) rather than exact
    logit-temperature softmax.

    Args:
        probs: Per-candidate probabilities.
        temperature: ``<= 0`` is greedy; ``> 0`` samples.
        rng: Sampling source (unused when greedy).
        actions: The candidates ``probs`` scores, when the caller has them.
            Greedy then decodes with :func:`ppo_decode.select_index` — the A8
            §8 group-mass rule the Rust lanes and the workbench use — so the
            live advisor cannot recommend differently from the workbench on a
            split-mass screen (shop removal fanned out per deck card).
            ``None`` keeps the plain per-candidate argmax.
    """
    if not probs:
        raise ValueError("_select_index: empty probability list")
    if temperature <= 0:
        if actions is not None:
            return ppo_decode.select_index(actions, probs)
        return max(range(len(probs)), key=probs.__getitem__)
    weights = [max(p, 1e-12) ** (1.0 / temperature) for p in probs]
    total = sum(weights)
    r = rng.random() * total
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if r <= acc:
            return i
    return len(weights) - 1


#: What a decision with exactly one action says instead of naming the net.
FORCED_NOTE = "forced (single option) — not a net ranking"


def decision_note(decision: PpoDecision, probs: list[float], net_note: str | None = None) -> str:
    """The HUD/corpus note for one decision, honest about who decided it.

    Both consumers (the PPO driver and the live advisor) short-circuit
    ``len(decision.actions) == 1`` to ``probs = [1.0]`` without ever calling
    the net -- and then printed that as ``run-policy net · p=100%`` /
    ``PPO p=1.00``, so the one pick the champion never made read as its most
    confident (GH #399). A forced pick says it is forced, and a decision that
    replicates engine behaviour additionally names the engine via
    :attr:`PpoDecision.source_note`.

    Args:
        decision: The decision being answered.
        probs: Per-action probabilities, aligned with ``decision.actions``.
        net_note: The caller's own attribution spelling for a real ranking
            (the driver's ``PPO p=0.70`` differs from the advisor's
            ``run-policy net · p=70%``). ``None`` uses the advisor's.

    Returns:
        The note text. Never mentions a probability for a forced pick.
    """
    if len(decision.actions) == 1:
        extra = decision.source_note
        return f"{FORCED_NOTE} · {extra}" if extra else FORCED_NOTE
    if net_note is not None:
        return net_note
    return f"run-policy net · p={max(probs):.0%}"


def build_ppo_annotations(
    decision: PpoDecision, probs: list[float], chosen: int, value: float | None
) -> dict[str, Any]:
    """Build the PROTOCOL.md ``annotations`` overlay payload from a PPO decision.

    See module docstring's "Combat scores in ``annotations``" note: the
    ``cards`` array here only renders in the live overlay during combat --
    for these non-combat decisions only ``note`` is currently visible
    in-game, but ``cards`` is still emitted for the JSONL corpus.
    """
    label = decision.labels[chosen] if chosen < len(decision.labels) else "?"
    net_note = f"PPO p={probs[chosen]:.2f}"
    if value is not None:
        net_note += f" V={value:.2f}"
    note = f"{decision_note(decision, probs, net_note)} -> {label}"
    cards = [
        {"index": i, "score": max(0.0, min(1.0, p)), "label": f"{p:.0%}"}
        for i, p in enumerate(probs)
    ]
    return {"note": note, "cards": cards}


def _remember_map_pos(
    session: dict[str, Any], state: dict[str, Any], action: dict[str, Any]
) -> None:
    """Record the (col, row) of the map node the answered ``map_select``
    chose -- the player's position for every decision until the next map
    screen. Feeds the map patch above (``current`` coord) and the derived
    ``act_floor``. Silently keeps the previous coord if the chosen node
    can't be identified (fallback answers, malformed nodes)."""
    idx = action.get("index")
    for i, n in enumerate(state.get("nodes") or []):
        if int(n.get("index", i)) == idx:
            try:
                session["map_pos"] = (int(n["col"]), int(n["row"]))
            except (KeyError, TypeError, ValueError):
                pass
            return


def apply_session_context(state: dict[str, Any], session: dict[str, Any] | None) -> dict[str, Any]:
    """Fold the per-run scratch dict into one decision's wire state.

    Split out of :func:`pick_noncombat_action_ppo` so the live ADVISOR
    gets the same view the driver has. The advisor rebuilds every decision
    from the wire and passed no session at all, so it was the one consumer
    still seeing the bare message -- and ~15% of live `map_select`
    messages arrive with an EMPTY map graph, which silently collapses
    every forward-cone routing feature to defaults. Same code, same cache,
    one behaviour.

    Args:
        state: The bridge message (a `type` key selects the surface).
        session: Per-run mutable scratch, or None to disable (returns
            `state` unchanged -- the one-shot/stateless caller contract).

    Returns:
        `state`, or a shallow copy with `map` patched in from the cache.
        MUTATES `session`.
    """
    msg_type = state.get("type")

    # ~15% of live map_select messages arrive with an EMPTY map graph
    # (2026-07-12 audit), collapsing every forward-cone routing feature to
    # defaults. The act's graph is immutable, so cache the last good one per
    # act and patch it in when the bridge omits it.
    if msg_type == "map_select" and session is not None:
        m = state.get("map") or {}
        act = state.get("act")
        if m.get("nodes"):
            session["last_map"] = (act, m)
        else:
            cached = session.get("last_map")
            if cached and cached[0] == act:
                state = {**state, "map": cached[1]}
    elif session is not None:
        m = state.get("map") or {}
        if m.get("nodes"):
            # New-dialect bridge (2026-08-02 run-level audit): SpireMentor
            # attaches the act graph to every run-level message's envelope,
            # not just map_select -- the live twin of headless_eval's
            # get_map() seeding (headless_eval.py:1510-1519). The wire's own
            # map (with the authoritative `current`) is used as-is by
            # `_base_obs`; here just refresh the cache + position tracker so
            # any message that still arrives bare (older mod build, mixed
            # deploys) keeps the same view. The pre-act row-0 `current` is
            # NOT a position -- leaving map_pos unset keeps the act-start
            # encoding `_base_obs`/`_base_metrics` expect.
            prev = session.get("last_map")
            session["last_map"] = (state.get("act"), m)
            if prev is not None and prev[0] != state.get("act"):
                # Fresh act: the old position belongs to the old graph
                # (headless_eval pops it the same way when reseeding).
                session.pop("map_pos", None)
            cur = m.get("current")
            try:
                if isinstance(cur, dict) and int(cur.get("row", -1)) > 0:
                    session["map_pos"] = (int(cur["col"]), int(cur["row"]))
                elif isinstance(cur, dict) and int(cur.get("row", -1)) == 0:
                    # The pre-act row-0 node is not a position -- act start.
                    session.pop("map_pos", None)
            except (KeyError, TypeError, ValueError):
                pass
        else:
            # The training obs carry the act map on EVERY decision (ppo.env.
            # observe); pre-audit bridges only send it on map_select, so 76%
            # of live net decisions had zero map-node tokens (2026-07-12
            # obs-parity audit). Patch the cached act graph in, with
            # `current` moved to the last chosen node (recorded below) --
            # that coord also drives the engine-faithful act_floor
            # (= row + 1) in _base_metrics.
            cached = session.get("last_map")
            if cached and cached[0] == state.get("act"):
                m = dict(cached[1])
                pos = session.get("map_pos")
                if pos is not None:
                    m["current"] = {"col": int(pos[0]), "row": int(pos[1])}
                state = {**state, "map": m}

    # The card-skip flag only bridges a card_reward -> the reward_screen that
    # immediately re-appears. Any other decision (map_select, shop, ...) means
    # we've left that reward context, so drop a stale flag before it can make a
    # later, unrelated reward_screen skip its card.
    if session is not None and msg_type not in ("reward_screen", "card_reward"):
        session.pop("card_reward_skipped", None)
    # The shop-removal target only bridges a shop answer -> the card_select
    # that immediately follows it. Any other decision means the removal never
    # opened (rejected purchase, re-presented shop, ...) -- drop it. (A shop
    # message re-stashes AFTER this clear if removal is chosen again.)
    if session is not None and msg_type != "card_select":
        session.pop("shop_remove_target", None)
    return state


def stash_shop_removal_target(
    session: dict[str, Any] | None,
    run_state: dict[str, Any],
    picked: dict[str, Any],
) -> None:
    """Remember which deck card the SHOP surface chose to remove.

    Shared by both lanes deliberately. The driver used to own this and the
    advisor had no equivalent, so every live shop removal was re-decided on
    the `remove_deck_card` surface -- the exact thing the carry exists to
    prevent, and it fired ZERO times across `logs/f7-mirror-check18` and
    `check19` because the advisor never calls `pick_noncombat_action_ppo`.
    """
    if session is None or picked.get("action") != "shop_remove_card":
        return
    deck_l = run_state.get("master_deck") or []
    di = int(picked.get("deck_idx", -1))
    if 0 <= di < len(deck_l):
        c = deck_l[di]
        session["shop_remove_target"] = (
            _card_id(c),
            bool(c.get("upgrade_level") or c.get("upgraded")),
        )


def shop_removal_carry_index(
    state: dict[str, Any],
    session: dict[str, Any] | None,
    decision: PpoDecision,
) -> int | None:
    """Which option on THIS screen is the card the shop already picked.

    In training a shop removal is ONE `shop_remove_card{deck_idx}` decision;
    the bridge splits it into a "removal service" answer plus a follow-up
    `card_select`. Re-deciding the target on the differently trained
    `remove_deck_card` surface picked a DIFFERENT card ~50% of the time
    (2026-07-16 meta differ), so the shop's pick wins when it can be found.

    CONSUMES the stash whether or not it locates the card: a target that
    cannot be found here will not become findable later, and leaving it set
    would let it fire on an unrelated screen.

    A miss is LOGGED, because it is the only signal that the two lists
    disagree. The fan-out reads `run_state.master_deck`; the screen reads
    C#'s Deck pile through `FromDeckForRemoval`. Those should agree and
    mostly do, but `logs/f7-mirror-check19` carries a `remove:OUTBREAK` pick
    against a screen with no Outbreak on it -- and Outbreak is not Eternal,
    so the removability filter did not explain it (GH #592). The fan-out now
    takes the game's own `is_removable` off the wire rather than
    reconstructing it, which is what makes a miss attributable at all: with
    that flag present the two lists are built from the same verdict, so a
    residual miss is an ID/display mismatch and nothing else.

    The warning names the DECK as well as the screen. Before, the log
    recorded the labels the fan-out built and the cards the screen offered
    but never the deck behind either, which is exactly why #592 stayed a
    guess -- the one artifact that could have settled it was the one not
    written down.
    """
    target = (session or {}).pop("shop_remove_target", None)
    if target is None:
        return None
    offered = state.get("cards") or []
    for i, c in enumerate(offered):
        key = (_card_id(c), bool(c.get("upgrade_level") or c.get("upgraded")))
        if key == target and i < len(decision.responses):
            return i

    def _tag(c: dict[str, Any]) -> str:
        name = str(_card_id(c))
        if c.get("upgrade_level") or c.get("upgraded"):
            name += "+"
        flag = c.get("is_removable")
        if flag is False:
            name += "(fixed)"
        elif flag is None:
            name += "(?)"
        return name

    deck = ((state.get("run_state") or {}).get("master_deck")) or []
    logger.warning(
        "shop removal target %s%s is not on the removal screen (%d offered: %s); "
        "the deck behind it was (%d: %s); the screen will re-decide",
        target[0],
        "+" if target[1] else "",
        len(offered),
        ", ".join(_tag(c) for c in offered),
        len(deck),
        ", ".join(_tag(c) for c in deck),
    )
    return None


def pick_noncombat_action_ppo(
    state: dict[str, Any],
    rng: random.Random,
    policies: PpoPolicyCache,
    *,
    temperature: float = 0.0,
    run_config_character: str | None = None,
    run_config_ascension: int | None = None,
    run_config_seed: int | None = None,
    map_explore: bool = False,
    session: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Pick a non-combat action: PPO where covered, random-legal otherwise.

    Args:
        session: Optional per-run mutable scratch dict for cross-decision
            state. Used to break the reward_screen <-> card_reward loop: when
            the policy SKIPS a card reward, the bridge re-shows the reward
            screen with the CardReward button still present (in the real game a
            card is forfeited by *proceeding*, not by a skip that removes it),
            so re-opening it would skip-and-loop forever. Pass a fresh ``{}``
            per session to enable the fix; omit for one-shot/stateless callers.

    Returns:
        ``(action, log_extra)`` -- ``log_extra`` is ``None`` when either
        the surface isn't (yet) PPO-covered (normal operation, see module
        docstring) or a single-option decision was auto-taken; it is
        ``{"ppo_error": "<message>"}`` when PPO was applicable but failed
        (missing checkpoint, malformed bridge fields, inference error) and
        the random-legal policy answered instead.
    """
    msg_type = state.get("type")
    builder = _PPO_NONCOMBAT_BUILDERS.get(msg_type or "")
    state = apply_session_context(state, session)

    def _fallback() -> dict[str, Any]:
        action = choose_action(
            state,
            rng,
            run_config_character=run_config_character,
            run_config_ascension=run_config_ascension,
            run_config_seed=run_config_seed,
            map_explore=map_explore,
        )
        # Fallback map answers still move the player -- keep the position
        # tracker (act map patch + act_floor) honest.
        if session is not None and msg_type == "map_select":
            _remember_map_pos(session, state, action)
        return action

    # reward_screen is a multi-claim loop (collect gold/potion/relic -> open the
    # CardReward -> proceed). It is NOT a run-policy decision in the sim -- the
    # headless engine auto-collects it (DetectPostCombatState) and only the CARD
    # is a decision. The random fallback instead treated "proceed" as just
    # another option and frequently bailed before claiming its card/gold/potion,
    # starving the deck. Route it through the SAME shared collector the RealGame
    # backend uses (reward_screen_choice); the CardReward it opens re-surfaces as
    # a real card_reward decision handled by the PPO below.
    if msg_type == "reward_screen":
        options = state.get("options") or []
        # If we already opened the CardReward this screen and the policy skipped
        # it, DON'T re-open it (that loops forever until AutoSlay's timeout) --
        # forfeit it by proceeding. A card is only skippable in the real game by
        # leaving the reward screen.
        if session is not None and session.pop("card_reward_skipped", False):
            proceed_idx = next(
                (o.get("index") for o in options if o.get("action") == "proceed"), None
            )
            if proceed_idx is not None:
                return {"action": "choose", "index": proceed_idx}, None
        idx = reward_screen_choice(options)
        return (
            ({"action": "choose", "index": idx}, None) if idx is not None else (_fallback(), None)
        )

    # card_select needs cross-decision context (which screen opened it) --
    # dispatched here rather than via the type->builder table. Only the
    # deck-mutation contexts (rest/shop/event) reach this function at all:
    # a MID-COMBAT selection is answered off the fight's persistent combat
    # mirror by `realgame_mcts.pick_card_select_mirrored`, which searches the
    # choice the played card already opened on it. Anything unmapped that
    # still lands here has no trained surface and takes the random-legal
    # fallback, as it always did.
    card_select_prev: str | None = None
    if msg_type == "card_select":
        card_select_prev = (session or {}).get("prev_msg_type")
        if _CARD_SELECT_SURFACES.get(card_select_prev or "") is None:
            return _fallback(), None
    elif builder is None:
        return _fallback(), None

    run_state = state.get("run_state") or {}
    character = run_state.get("character") or run_config_character or "Ironclad"
    policy, policy_error = policies.get(character)
    if policy is None:
        return _fallback(), {"ppo_error": policy_error} if policy_error else None

    try:
        if msg_type == "card_select":
            decision = _build_card_select(state, card_select_prev)
            if decision is not None:
                carried = shop_removal_carry_index(state, session, decision)
                if carried is not None:
                    action = dict(decision.responses[carried])
                    label = decision.labels[carried] if carried < len(decision.labels) else "?"
                    action["annotations"] = {
                        "note": f"shop-removal target carried from shop pick -> {label}"
                    }
                    return action, None
        else:
            assert builder is not None
            decision = builder(state)
        if decision is None or not decision.actions:
            return _fallback(), None
        if len(decision.actions) == 1:
            probs, value, chosen = [1.0], None, 0
        else:
            probs, value = policy.distribution(decision.obs, decision.actions, decision.metrics)
            chosen = _select_index(probs, temperature, rng, decision.actions)
        action = dict(decision.responses[chosen])
        if getattr(decision, "pick_count", 1) > 1 and isinstance(action.get("indices"), list):
            # The world wants this answered in ONE reply and rejects a short
            # one (`SimCardSelector.cs:368`), so fill the remaining slots with
            # the next-best distinct options. `chosen` stays first: it is the
            # sampled pick and the one the annotations describe.
            order = sorted(range(len(probs)), key=probs.__getitem__, reverse=True)
            picked = list(action["indices"])
            seen = {chosen}
            for i in order:
                if len(picked) >= decision.pick_count:
                    break
                if i in seen or i >= len(decision.responses):
                    continue
                seen.add(i)
                picked.extend(decision.responses[i].get("indices") or [])
            action["indices"] = picked[: decision.pick_count]
        action["annotations"] = build_ppo_annotations(decision, probs, chosen, value)
        # Remember a card-reward skip so the reward_screen that immediately
        # re-appears (CardReward button still present) proceeds instead of
        # re-opening -> skipping -> looping. Any card_reward TAKE clears it.
        if session is not None and msg_type == "card_reward":
            session["card_reward_skipped"] = action.get("action") == "skip"
        if session is not None and msg_type == "map_select":
            _remember_map_pos(session, state, action)
        if session is not None and msg_type == "shop":
            stash_shop_removal_target(session, run_state, decision.actions[chosen])
        return action, None
    except BudgetExceededError as e:
        # ONNX is the default backend now, and unlike torch's dynamic collate
        # the fixed-shape graph has a budget to exceed (GH #315's 103-option
        # shop removal, or any decision over T_MAX tokens). No trained head can
        # answer instead, so the pick stays random -- but it must not stay
        # SILENT: these are the highest-leverage decisions in a run, and a
        # sweep would otherwise report the win rate as the net's. Tagged
        # distinctly so a lane can count them.
        logger.warning(
            "PPO budget exceeded for %s at floor=%s (%s) -- falling back to a RANDOM legal pick",
            msg_type,
            state.get("floor"),
            e,
        )
        return _fallback(), {"ppo_error": str(e), "ppo_budget_exceeded": True}
    except Exception as e:  # Never crash the session on a bad snapshot.
        logger.error(
            "PPO decision failed for %s at floor=%s: %s",
            msg_type,
            state.get("floor"),
            e,
            exc_info=logger.isEnabledFor(logging.DEBUG),
        )
        return _fallback(), {"ppo_error": str(e)}


# ---------------------------------------------------------------------------
# Session loop -- mirrors realgame_mcts.run() exactly, with combat routed to
# either MCTS (default, matches how the champion nets were themselves
# evaluated) or the random-legal policy, and non-combat routed through
# pick_noncombat_action_ppo above.
# ---------------------------------------------------------------------------


def run(
    host: str,
    port: int,
    log_dir: Path,
    max_steps: int | None,
    rng: random.Random,
    policies: PpoPolicyCache,
    *,
    temperature: float = 0.0,
    combat_policy: str = "mcts",
    mcts_cfg: RecommendConfig | None = None,
    sims_regular: int = DEFAULT_SIMS_REGULAR,
    sims_boss: int = DEFAULT_SIMS_BOSS,
    sims_flat: int | None = None,
    character: str | None = None,
    ascension: int | None = None,
    seed: int | None = None,
    map_explore: bool = False,
) -> int:
    """Run the PPO-driven play loop to completion (or ``max_steps``).

    Args:
        combat_policy: ``"mcts"`` (default, reuses
            :func:`realgame_mcts.pick_combat_action_mcts` -- the same
            combat resolution the champion nets were trained/evaluated
            against) or ``"random"`` (reuses
            :func:`realgame_random.pick_combat_action`, no ``sts2_rs`` MCTS
            calls -- cheaper, no combat fidelity claim).
        mcts_cfg: Required (non-``None``) iff ``combat_policy == "mcts"``.

    Returns:
        Process exit code (0 on a clean finish, 1 on a connection error).
    """
    if combat_policy == "mcts" and mcts_cfg is None:
        raise ValueError("mcts_cfg is required when combat_policy='mcts'")

    jsonl = JsonlLogger(log_dir)
    step = 0
    result = "unknown"
    ppo_fallback_count = 0
    combat_fallback_count = 0
    current_room_type: str | None = None  # type of the map node last walked into
    session: dict[str, Any] = {}  # per-run scratch for pick_noncombat_action_ppo

    print(f"Logging to {jsonl.path}")
    if mcts_cfg is not None:
        # Say which search profile this run used. The 2026-08-02 clairvoyant
        # confound survived for weeks partly because no live report named the
        # profile it had searched (same line realgame_eval/headless_eval record
        # in their report meta).
        print(
            f"Combat search: hidden_info={mcts_cfg.hidden_info} "
            f"c_puct={mcts_cfg.c_puct} draw_widening_c={mcts_cfg.draw_widening_c}"
        )

    try:
        with RealGameBridgeClient(host=host, port=port) as client:
            while True:
                state = client.receive_state()
                msg_type = state.get("type")

                if msg_type == "pong":
                    continue

                if msg_type == "error":
                    jsonl.log_error(state)
                    print(f"[error] {state.get('error')}: {state.get('detail')}")
                    continue

                if msg_type in _TERMINAL_TYPES:
                    step += 1
                    jsonl.log_terminal(step, state)
                    result = state.get("result") or state.get("message") or msg_type
                    print(f"[{step:4d}] {msg_type} -> result={result!r}")
                    if msg_type == "run_complete":
                        break
                    continue

                log_extra: dict[str, Any] | None = None
                if msg_type == "combat_action":
                    if combat_policy == "mcts":
                        assert mcts_cfg is not None
                        sims = combat_sims_for(
                            current_room_type, sims_regular, sims_boss, sims_flat
                        )
                        # No mirror on this lane, so the engine-dialect twin is
                        # always None -- discarded rather than named.
                        action, log_extra, _ = pick_combat_action_mcts(
                            state, mcts_cfg, rng, sims=sims
                        )
                        if log_extra is not None:
                            combat_fallback_count += 1
                    else:
                        action = pick_combat_action(state, rng)
                else:
                    action, log_extra = pick_noncombat_action_ppo(
                        state,
                        rng,
                        policies,
                        temperature=temperature,
                        run_config_character=character,
                        run_config_ascension=ascension,
                        run_config_seed=seed,
                        map_explore=map_explore,
                        session=session,
                    )
                    if log_extra is not None:
                        ppo_fallback_count += 1
                    # Remember the room we're entering so the next combat knows
                    # whether it's the boss (4000 sims) or a regular fight (1000).
                    node_type = chosen_map_node_type(state, action)
                    if node_type is not None:
                        current_room_type = node_type

                # Context for the NEXT decision: card_select carries no source
                # info; pick_noncombat_action_ppo reads the opening screen's
                # type from here (rest_site/shop/event -> deck-mutation surface).
                session["prev_msg_type"] = msg_type

                step += 1
                jsonl.log_step(step, state, action, extra=log_extra)
                print(_progress_line(step, state, action))
                client.send_action(action)

                if max_steps is not None and step >= max_steps:
                    result = "max_steps_reached"
                    print(f"Stopping after max_steps={max_steps}")
                    break
    except (ConnectionError, TimeoutError) as e:
        logger.error("Bridge connection error after %d steps: %s", step, e)
        result = f"error: {e}"
        print(
            f"Steps: {step}\nPPO fallbacks: {ppo_fallback_count}\n"
            f"Combat fallbacks: {combat_fallback_count}\nResult: {result}\nLog: {jsonl.path}"
        )
        jsonl.close()
        return 1

    jsonl.close()
    print(
        f"Steps: {step}\nPPO fallbacks: {ppo_fallback_count}\n"
        f"Combat fallbacks: {combat_fallback_count}\nResult: {result}\nLog: {jsonl.path}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST, help="Bridge mod TCP host.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Bridge mod TCP port.")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs/realgame"),
        help="Directory to write the JSONL run log into.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Stop after this many decisions (default: run to game_over/run_complete).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Seed for the random-legal fallback policy's RNG and the "
            "temperature-sampling RNG, and (if the mod's client_run_config "
            "flag is on) the run seed offered in start_run."
        ),
    )
    parser.add_argument(
        "--character",
        default=None,
        help="Character id to request via start_run when a run_config message is offered.",
    )
    parser.add_argument(
        "--ascension",
        type=int,
        default=None,
        help="Ascension level to request via start_run when a run_config message is offered.",
    )
    parser.add_argument(
        "--explore-map",
        action="store_true",
        help="Bias map choices toward rarely exercised screen types (see realgame_random).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Explicit PPO checkpoint path (.pt), overriding STS2_WB_PPO_CKPT and the "
            "per-character committed-champion lookup (see workbench.ppo_policy.ckpt_path)."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0 (default) = greedy argmax; >0 = power-law re-tempered sampling (approximate).",
    )
    parser.add_argument(
        "--combat-policy",
        choices=["mcts", "random"],
        default="mcts",
        help=(
            "How combat_action decisions are answered (PPO has no combat policy -- see "
            "module docstring): 'mcts' (default, matches the champion nets' own training/"
            "eval combat resolution) or 'random' (cheaper, no sts2_rs MCTS engine calls)."
        ),
    )
    parser.add_argument(
        "--sims",
        "--playouts",
        dest="sims",
        type=int,
        default=None,
        help="Flat MCTS simulations for EVERY combat decision (--combat-policy mcts), "
        "overriding the per-encounter --sims-regular/--sims-boss schedule (default: adaptive).",
    )
    parser.add_argument(
        "--sims-regular",
        type=int,
        default=DEFAULT_SIMS_REGULAR,
        help=f"MCTS simulations for regular (non-boss) fights (default: {DEFAULT_SIMS_REGULAR}).",
    )
    parser.add_argument(
        "--sims-boss",
        type=int,
        default=DEFAULT_SIMS_BOSS,
        help=f"MCTS simulations for boss fights (default: {DEFAULT_SIMS_BOSS}).",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=RecommendConfig.depth,
        help=f"MCTS rollout depth cap (default: {RecommendConfig.depth}).",
    )
    parser.add_argument(
        "--c-puct",
        type=float,
        default=None,
        help="Explicit exploration constant; default follows the search profile "
        "(2.5 honest / 1.4 with --clairvoyant).",
    )
    parser.add_argument(
        "--clairvoyant",
        action="store_true",
        help="reproduce the OLD leaky combat search (hidden_info=clairvoyant, "
        "c_puct=1.4, no draw widening). Default matches the DEPLOYED honest "
        "MctsCombatPolicy (beliefdraw, c_puct=2.5, draw_widening_c=2.0) — the "
        "same trio headless_eval and realgame_eval default to. This driver was "
        "missed by the 2026-08-02 sweep that fixed those two (GH #383) and "
        "silently ran the clairvoyant profile until then, so no sweep it "
        "produced was comparable to the rung-3.5, engine or advisor arms.",
    )
    parser.add_argument(
        "--mcts-seed",
        type=int,
        default=RecommendConfig.seed,
        help=f"RNG seed for MCTS rollouts, separate from --seed (default: {RecommendConfig.seed}).",
    )
    parser.add_argument(
        "--rollout-policy",
        choices=["heuristic", "uniform", "net"],
        default=RecommendConfig.rollout_policy,
        help="MCTS rollout policy (default: what --hidden-info deploys — the playout net under beliefdraw).",
    )
    parser.add_argument(
        "--time-budget-ms",
        type=int,
        default=RecommendConfig.timeout_ms,
        help="Forwarded as RecommendConfig.timeout_ms (see realgame_mcts's identical flag).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    rng = random.Random(args.seed)
    policies = PpoPolicyCache(args.model)
    mcts_cfg = (
        RecommendConfig(
            # Placeholder base sims; run() overrides per decision via the
            # regular/boss schedule (or the flat --sims override).
            sims=args.sims if args.sims is not None else args.sims_regular,
            depth=args.depth,
            # The three honest knobs travel together (see RecommendConfig's
            # hidden_info comment): on reconstructed real-DLL states the RNG
            # cursors are faked, so a clairvoyant root plans against a
            # fictional future.
            hidden_info="clairvoyant" if args.clairvoyant else "beliefdraw",
            c_puct=args.c_puct if args.c_puct is not None else (1.4 if args.clairvoyant else 2.5),
            draw_widening_c=None if args.clairvoyant else 2.0,
            seed=args.mcts_seed,
            rollout_policy=args.rollout_policy,
            timeout_ms=args.time_budget_ms,
        )
        if args.combat_policy == "mcts"
        else None
    )
    return run(
        args.host,
        args.port,
        args.log_dir,
        args.max_steps,
        rng,
        policies,
        temperature=args.temperature,
        combat_policy=args.combat_policy,
        mcts_cfg=mcts_cfg,
        sims_regular=args.sims_regular,
        sims_boss=args.sims_boss,
        sims_flat=args.sims,
        character=args.character,
        ascension=args.ascension,
        seed=args.seed,
        map_explore=args.explore_map,
    )


if __name__ == "__main__":
    sys.exit(main())
