"""FastAPI app for the workbench replay core.

Serves a loaded run's steps (enriched with resolved art URLs) and on-demand
MCTS recommendations, plus the extracted game art as static files.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import queue
import threading
import uuid
import zlib
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any

import sts2_client
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sts2_hint import display_maps

from workbench import (
    bugreport,
    bugreport_store,
    deck_analysis,
    localization,
    reasoning,
    regret_review,
    render,
    wpa_review,
)
from workbench import saves as saves_mod
from workbench.advice_stream import stream_advice
from workbench.aggregate import _parse_run, aggregate_runs
from workbench.assets import AssetResolver, normalize_kind
from workbench.branching import BranchingLiveSession
from workbench.combatlog import LogNames, format_combat_log
from workbench.live_runs import LiveRuns, RestoreError, resume_capability
from workbench.options import normalize_options
from workbench.session import (
    SID_SCOPE_KEY,
    Session,
    SessionRegistry,
    install_session_cookie,
)
from workbench.statediff import diff_states
from workbench.timeline import TimelineStore

_log = logging.getLogger(__name__)


class LoadRequest(BaseModel):
    """Request body for POST /api/load."""

    path: str


class RecommendRequest(BaseModel):
    """Request body for POST /api/recommend."""

    idx: int
    config: dict[str, Any] = {}


class RegretReviewRequest(BaseModel):
    """Request body for POST /api/regret-review (config only; reviews the whole
    loaded run). ``config`` carries the review knobs (``blunder_pct``,
    ``top_n``, ``recompute``) plus any MCTS params used when re-searching."""

    config: dict[str, Any] = {}


class StartRequest(BaseModel):
    """Request body for POST /api/live/start."""

    character: str = "Ironclad"
    seed: str = "test"
    ascension: int = 0
    backend: str = "rust"
    # GH #544: which Act-1 model to walk. Defaults to `"roll"` — the real
    # lobby's seed-driven pick — because the workbench's job is to reproduce
    # the player's game for a given seed, and the roll is a uniform 50/50 once
    # Underdocks is unlocked and discovered (`ActModel.GetRandomList`,
    # ActModel.cs:558). Defaulting to `"overgrowth"` showed the WRONG ACT for
    # half of all seeds, which is what made act-1 bug reports irreproducible.
    # Pin it to `"overgrowth"` / `"underdocks"` to force one.
    act1: str = "roll"


class ImportStateRequest(BaseModel):
    """Request body for POST /api/live/import-state and /api/live/import-state/preview."""

    payload: str
    """Whatever the user pasted — see :mod:`workbench.state_import` for the shapes."""
    lane: str = "auto"
    """``"auto"`` (the fight if the paste has one, else the run), ``"combat"``, or ``"run"``."""


class SimlabSweepRequest(BaseModel):
    """Request body for POST /api/live/simlab/sweep."""

    name: str = "workbench position"
    strategies: list[str] | None = None
    n_seeds: int = 200
    node_ids: list[int] | None = None
    """Branch-tree nodes to sweep, one cell each. Empty/absent = the active node.

    Naming several is the counterfactual comparison: play the alternatives out
    one move each, then sweep them side by side and the distributions differ by
    that decision alone.
    """


class RunModeRequest(BaseModel):
    """Request body for POST /api/live/run-mode — any subset updates."""

    off_combat_brain: str | None = None
    ppo_impl: str | None = None
    combat_sims: int | None = None
    combat_boss_sims: int | None = None
    combat_threads: int | None = None
    reward_sims: int | None = None


class StartFaithfulRequest(BaseModel):
    """Request body for POST /api/live/start-faithful.

    Drives the native ``HeuristicRunAgent`` (heuristic meta + combat MCTS)
    at the eval-cheap budget.  ``seed`` is the run integer as a string, or a
    ``"run_N"`` label.  ``sims`` / ``boss_sims`` mirror the arena knobs
    (defaults reproduce eval-cheap: 1000 / 4000).  ``reward_sims`` is
    accepted but ignored — it was the old boss-sim card-chooser budget
    which the agent no longer uses (kept for stale-SPA compat).
    Extra fields are silently ignored by Pydantic v2 defaults.
    """

    character: str = "Ironclad"
    seed: str = "run_1"
    ascension: int = 0
    sims: int = 1000
    boss_sims: int = 4000
    threads: int = 1
    reward_sims: int | None = None  # accepted, ignored — stale SPA compat


class ActionRequest(BaseModel):
    """Request body for POST /api/live/action."""

    action: str
    args: dict[str, Any] = {}


class BridgeSeekRequest(BaseModel):
    """POST /api/bridge/seek -- rewind the live game (node or snapshot)."""

    node_id: int | None = None
    snapshot_id: str | None = None


class BridgeSearchTreeRequest(BaseModel):
    """POST /api/bridge/search-tree -- companion MCTS over the live board."""

    sims: int | None = None


class HintRequest(BaseModel):
    """Request body for POST /api/live/hint (config only; no step index)."""

    config: dict[str, Any] = {}


class SeekRequest(BaseModel):
    """Request body for POST /api/live/seek (rewind to a branch-tree node)."""

    node_id: int


class WpaVerifyRequest(BaseModel):
    """Request body for POST /api/live/wpa-verify (counterfactual spot check
    of one WPA turning point; slow — two agent-played continuations)."""

    node_id: int
    max_steps: int = 2_000
    max_seconds: float = 120.0


class SaveRequest(BaseModel):
    """Request body for POST /api/live/save (name the current line)."""

    name: str


class LoadSaveRequest(BaseModel):
    """Request body for POST /api/saves/load (restore a named save)."""

    name: str


class RewardPreviewRequest(BaseModel):
    """Request body for POST /api/deck/reward-preview.

    ``deck`` is the current master deck and ``candidates`` the offered reward
    cards; both are lists of card dicts (only ``id`` / ``upgraded`` are read,
    plus ``index`` on candidates).
    """

    deck: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []


class BugReportRequest(BaseModel):
    """Request body for POST /api/bugreport (freeze the current moment)."""

    mode: str = "live"  # "live" | "replay"
    step: int | None = None  # replay: the line/step index being viewed
    screen: str | None = None  # current UI screen id
    overlay: str | None = None  # open overlay id, if any
    note: str | None = None  # optional free-text bug description
    run_path: str | None = None  # replay: the loaded run file path


# Base URL for served art. Defaults to the locally-mounted ``/assets`` route;
# set ``STS2_ASSETS_BASE_URL`` to a public bucket/CDN origin (e.g.
# ``https://storage.googleapis.com/<bucket>``) to serve the 142 MB of art off
# GCS and keep the container lean — see ``create_app``.
_ASSETS_BASE_URL = os.environ.get("STS2_ASSETS_BASE_URL", "/assets").rstrip("/")


def _asset_url(rel: str | None) -> str | None:
    """Convert a relative image path to a served asset URL, or None.

    The base is ``/assets`` by default (locally-mounted art) but can point at an
    external origin (GCS bucket / CDN) via ``STS2_ASSETS_BASE_URL``.
    """
    return f"{_ASSETS_BASE_URL}/{rel}" if rel else None


# The game tints a card's frame by the *pool* it belongs to, not by the run
# character — a Colorless card in an Ironclad deck is grey, not red. These are
# the ``CardFrameMaterialPath`` overrides from the decompiled ``*CardPool.cs``.
_POOL_FRAME_COLOR = {
    "ironclad": "red",
    "silent": "green",
    "defect": "blue",
    "necrobinder": "pink",
    "regent": "orange",
}


def _frame_color(card_character: str | None, card_type: str | None, rarity: str | None) -> str:
    """Return the card's frame tint key (``red``/``green``/…/``colorless``).

    Curse, Quest and Status cards live in their own pools regardless of which
    character is holding them, so they are matched before the character lookup.
    Anything with no known character (Colorless, Token, Event, and cards absent
    from the manifest) falls back to the colourless frame, which is what
    ``ColorlessCardPool`` / ``StatusCardPool`` / ``TokenCardPool`` all use.
    """
    kind = f"{card_type or ''}|{rarity or ''}".lower()
    if "curse" in kind:
        return "curse"
    if "quest" in kind:
        return "quest"
    return _POOL_FRAME_COLOR.get((card_character or "").strip().lower(), "colorless")


def _enrich_map_nodes(
    full_map: dict[str, Any] | None, assets: AssetResolver
) -> dict[str, Any] | None:
    """Attach an ``icon`` URL (the real in-game room-type badge) to map nodes.

    Walks a ``full_map`` graph (``{"rows": [[node, …], …], "boss": node|None}``)
    in place, adding an ``icon`` key to every node and the boss meta-node. The
    icon is the game's own per-room-type art (monster / elite / shop / rest /
    treasure / event / ancient, plus the specific boss); the frontend falls back
    to its text glyph when ``icon`` is None (art missing).

    A10 DoubleBoss: the second boss node is a real ``rows`` entry typed
    ``Boss`` sitting one row above the act boss, and it fights a DIFFERENT
    encounter (``second_boss.id``). Its icon is resolved from that id, not from
    the act boss's — otherwise the last two nodes of the run both wear the
    first boss's face.
    """
    if not full_map:
        return full_map
    boss = full_map.get("boss") or {}
    boss_id = boss.get("id")
    second = full_map.get("second_boss") or {}
    second_id = second.get("id")
    second_coord = (second.get("col"), second.get("row")) if second.get("row") is not None else None
    for row in full_map.get("rows") or []:
        for node in row:
            node_boss_id = (
                second_id
                if second_coord is not None and (node.get("col"), node.get("row")) == second_coord
                else boss_id
            )
            node["icon"] = _asset_url(assets.room_icon(node.get("type"), node_boss_id))
    if boss:
        boss["icon"] = _asset_url(assets.room_icon(boss.get("type"), boss_id))
    if second:
        second["icon"] = _asset_url(assets.room_icon(second.get("type"), second_id))
    return full_map


# `Rng.Chaotic.NextFloat(0.05)` in the game — seeded off the wall clock, so the
# hue is not reproducible even in principle. A stable per-instance sample from
# the same range is as faithful as the game's own draw and keeps a replayed step
# from shimmering every time it is re-rendered.
_HUE_JITTER_MAX = 0.05


def _apply_instance_jitter(
    enemies: list[dict[str, Any]], hp_ranges: dict[str, dict[str, int]]
) -> None:
    """`NCombatRoom.RandomizeEnemyScalesAndHues`, stamped onto the wire enemies.

    The game sizes and tints EVERY member of a group of two or more monsters of
    the same type, by where that instance's rolled max HP landed inside its
    `[MinInitialHp, MaxInitialHp]` band:

        num   = clamp(((hp - min) / (max - min) - 0.5) * 2, 0, 1)
        amt   = clamp(inverseLerp(250, 100, max(bounds.x, bounds.y)), 0, 1)
        scale = 1 + num * lerp(0.10, 0.15, amt)

    so a big monster varies by up to 10% and a small one by up to 15%, and only
    the above-median half of the band grows at all (`num` clamps at 0). Three
    Exoskeletons at 26/24/25 HP are three visibly different sizes in the game;
    without this they were three identical clones.

    Modifies ``enemies`` in place, adding ``size_jitter`` (and ``hue_jitter``)
    only where the rule actually fires — a lone monster, an unknown kind or a
    fixed-HP band all leave the enemy untouched, so the frontend can treat the
    absence as 1.0.

    ASCENSION: the band moves by +1 at `hp_ascension` and no wire carries the
    run's ascension here, so the band that CONTAINS the rolled HP is used, base
    first. That is exact below the threshold and above it whenever the roll
    falls outside the base band; inside the overlap it can be one step off,
    which is a fraction of a 10-15% envelope.

    Args:
        enemies: The state's enemy list, already carrying ``kind``/``max_hp``
            and the authored ``rig_bounds_w``/``rig_bounds_h``.
        hp_ranges: :func:`workbench.localization.load_monster_hp_ranges` output.
    """
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for enemy in enemies:
        kind = normalize_kind(enemy.get("kind"))
        if kind:
            by_kind.setdefault(kind, []).append(enemy)
    for kind, group in by_kind.items():
        # "Two or more of the same type" is the whole trigger; a solo monster is
        # drawn at its authored size.
        if len(group) < 2:
            continue
        band = hp_ranges.get(kind)
        if not band:
            continue
        for index, enemy in enumerate(group):
            hp = enemy.get("max_hp")
            if not isinstance(hp, int):
                continue
            lo, hi = band["min"], band["max"]
            if not (lo <= hp <= hi) and band["min_asc"] <= hp <= band["max_asc"]:
                lo, hi = band["min_asc"], band["max_asc"]
            span = hi - lo
            num = 0.0 if span <= 0 else max(0.0, min(1.0, ((hp - lo) / span - 0.5) * 2))
            box = max(
                float(enemy.get("rig_bounds_w") or 0.0), float(enemy.get("rig_bounds_h") or 0.0)
            )
            # inverseLerp(250, 100, box): 1 for a small creature, 0 for a big one.
            amount = 0.0 if box <= 0 else max(0.0, min(1.0, (250.0 - box) / 150.0))
            envelope = 0.10 + (0.15 - 0.10) * amount
            enemy["size_jitter"] = round(1.0 + num * envelope, 4)
            # crc32, NOT hash(): Python randomizes string hashing per process,
            # so the tint would change on every server restart.
            spread = zlib.crc32(f"{kind}|{index}".encode()) & 0xFFFF
            enemy["hue_jitter"] = round(_HUE_JITTER_MAX * (spread / 0xFFFF), 4)


def _enemy_has_power(enemy: dict[str, Any], token: str) -> bool:
    """Whether an enriched enemy carries a power whose id/name matches ``token``.

    ``token`` is an upper-case fragment of the power's registry id (e.g.
    ``"THORNS"``, ``"BURROWED"``, ``"PAINFUL_STABS"``). Checks both the raw
    ``id`` (``PowerKind`` python_name) and the display ``name`` so it is robust
    to whichever the wire carries.

    Display titles are spaced, not snake_cased ("Painful Stabs"), so both sides
    are normalized to UNDERSCORES before matching — otherwise a multi-word
    token silently matches nothing on a backend that ships only the title, which
    is how every Test Subject stayed stuck in its one-headed first form.
    """
    want = token.upper().replace(" ", "_")
    for pw in enemy.get("powers") or []:
        for field in ("id", "name"):
            if want in str(pw.get(field, "")).upper().replace(" ", "_"):
                return True
    return False


# Re-exported from :mod:`workbench.localization` so the live-log path
# (:mod:`workbench.branching`) and this module resolve names identically.
_LOC_KEY_RE = localization._LOC_KEY_RE
_resolve_name = localization.resolve_name
_names_resolved = localization.resolve_state_names


def _manifest_cost(cid: str, loc: dict[str, str], *, upgraded: bool = False) -> int | str | None:
    """The card's display-ready energy cost from the wiki manifest, by id.

    Returns an ``int`` for fixed-cost cards, the string ``"X"`` for variable
    cost, or ``None`` when the manifest has no entry.

    ``upgraded`` picks the post-upgrade cost (``<id>.upgraded_cost``) when the
    manifest has one — 57 cards upgrade by dropping an energy (Stampede 2 → 1),
    and only those carry the key; every other card's upgraded cost IS its base
    cost, so the fallback is exact rather than approximate.
    """
    mcost = loc.get(f"{cid}.upgraded_cost") if upgraded else None
    if mcost is None:
        mcost = loc.get(f"{cid}.cost")
    if mcost is None:
        return None
    return int(mcost) if mcost.lstrip("-").isdigit() else mcost


def _apply_upgrade_cost(twin: dict[str, Any], card: dict[str, Any], loc: dict[str, str]) -> None:
    """Put the post-upgrade energy cost on an upgrade-preview ``twin``.

    ``twin`` is a deep copy of ``card`` with ``upgraded`` already flipped, so
    every cost key on it is still the UN-upgraded one. Two cases:

    * No live cost on the wire (deck / rest-site Smith options): drop the keys
      and let :func:`enrich_card` backfill ``<id>.upgraded_cost`` from the
      manifest — it reads ``twin["upgraded"]``.
    * A live cost (the in-combat Armaments-style pick): the number may carry a
      hook's discount, which the upgrade does not remove, so shift it by the
      manifest's upgrade delta rather than replacing it. A card the manifest
      gives no upgraded cost keeps the cost it has (delta 0), which is right
      for the ~500 cards whose upgrade leaves the cost alone. The shifted
      value floors at 0 — a card a hook has already zeroed does not go
      negative when its printed cost drops.

    X-cost cards (``"X"``) have no arithmetic to do and are left untouched, as
    is a NEGATIVE live cost: on this wire ``-1`` is the unplayable sentinel a
    Burn / Wound carries, not a number to do arithmetic on.
    """
    cid = (card.get("id") or "").split(".")[-1]
    base = _manifest_cost(cid, loc)
    upgraded = _manifest_cost(cid, loc, upgraded=True)
    delta = upgraded - base if isinstance(base, int) and isinstance(upgraded, int) else 0
    for key in ("cost", "base_cost"):
        live = twin.get(key)
        if live is None:
            twin.pop(key, None)
        elif isinstance(live, int) and live >= 0:
            twin[key] = max(0, live + delta)


def _resolve_owned_text(item: dict[str, Any], loc: dict[str, str]) -> str:
    """Resolve a relic/potion description template using its own ``vars`` dict.

    The game ships relic/potion text with ``{VarName}`` placeholders and a
    per-item ``vars`` map of resolved numbers (e.g. ``{"Cards": 2}``).
    :func:`render.resolve_card_text` keys ``vars`` case-insensitively. A
    description that stays an unresolved loc key (no entry in our loc data)
    is blanked rather than shown raw.

    The Rust backend can't attach a relic/potion ``vars`` dict, so its templates
    (``"heal {Heal} HP"``) would render bare variable names. When no ``vars`` are
    present, fall back to the wiki's pre-resolved plain description (keyed by the
    item id / description loc-key stem) — the relic/potion analogue of a card's
    ``description_plain``. The C# backend always ships ``vars``, so this never
    fires there.
    """
    item_vars = {k.lower(): v for k, v in (item.get("vars") or {}).items()}
    if not item_vars:
        stem = (item.get("id") or "").split(".")[-1] or (item.get("description") or "").split(".")[
            0
        ]
        plain = localization.plain_owned_description(stem, loc)
        if plain:
            return render.resolve_card_text(plain, None)
    template = localization.resolve(item.get("description"), loc)
    if _LOC_KEY_RE.match(template):
        return ""
    return render.resolve_card_text(template, item_vars)


#: Branch flags for the event-text DSL's boolean form
#: (``{IsMultiplayer:Everyone loses|Lose}``). Without the flag the renderer
#: has no branch to pick and substitutes the VALUE instead, so Morphic Grove's
#: Group read "0 ALL of 0 Gold.". The sim is single-player, always.
_EVENT_FLAGS = {"ismultiplayer": False}


# Mad Science (the Tinker Time event card) morphs by CardType and carries one
# of these "rider" effects, both chosen when it's created. Its description
# template branches on ``{CardType:choose(...)}`` + ``{HasRider:..}`` + a boolean
# per rider, and its art is one of three per-type sprites. The backend ships the
# per-instance morph as ``type`` (= the live CardType) + ``tinker_time_rider``.
def _event_option_vars(
    opt: dict[str, Any], loc: dict[str, str], *, key: str = "vars"
) -> dict[str, Any]:
    """Lower-case an event option's ``vars``, resolving any name it carries.

    Numbers arrive ready to print. Names do not: an option that talks about a
    specific card, relic or potion ("{RandomCard} is removed from your Deck.")
    gets it as a ``<CARD_ID>.title`` loc key, because the engine holds ids and
    the display strings live here — the same contract the Rust backend uses for
    every other name it emits. Prose passes through untouched, so the C# backend
    (which ships display strings already) is unaffected.
    """
    out: dict[str, Any] = {}
    for name, val in (opt.get(key) or {}).items():
        out[name.lower()] = _resolve_name(val, loc) if isinstance(val, str) else val
    return out


_MAD_SCIENCE_RIDERS = (
    "violence",
    "sapping",
    "choking",
    "energized",
    "wisdom",
    "chaos",
    "expertise",
    "curious",
    "improvement",
)


def _mad_science_flags(card: dict[str, Any]) -> dict[str, bool] | None:
    """Build the morph ``extra_flags`` for a Mad Science card, else None.

    Keys: ``cardtype`` (the live type string), ``hasrider``, and one bool per
    rider name — exactly what the ``MAD_SCIENCE.description`` template reads.
    """
    if (card.get("id") or "").split(".")[-1] != "MAD_SCIENCE":
        return None
    rider = str(card.get("tinker_time_rider") or "None").lower()
    flags: dict[str, bool] = {r: (rider == r) for r in _MAD_SCIENCE_RIDERS}
    flags["hasrider"] = rider != "none"
    # The chooser reads CardType from the flags; ``type`` is the live morph type.
    flags["cardtype"] = card.get("type") or "Attack"  # type: ignore[assignment]
    return flags


# ``card_select`` contexts whose pick UPGRADES the chosen card: the rest-site
# Smith / Pomander deck upgrade (``upgrade``) and the in-combat Armaments-style
# pick (``upgrade_in_hand``). For these the frontend shows the real game's
# before → after comparison (``NUpgradePreview``), so every un-upgraded option
# carries a rendered copy of its upgraded self under ``upgrade_preview``.
_UPGRADE_CONTEXTS = frozenset({"upgrade", "upgrade_in_hand"})


def _enrich_relic(relic: dict[str, Any], assets: AssetResolver, loc: dict[str, str]) -> None:
    """Attach an ``image`` URL and resolve the name/description of a relic."""
    relic["name"] = _resolve_name(relic.get("name"), loc)
    relic["image"] = _asset_url(assets.relic(relic["name"]))
    relic["description"] = _resolve_owned_text(relic, loc)


def _enrich_potion(potion: dict[str, Any], assets: AssetResolver, loc: dict[str, str]) -> None:
    """Attach an ``image`` URL and resolve the name/description of a potion."""
    potion["name"] = _resolve_name(potion.get("name"), loc)
    potion["image"] = _asset_url(assets.potion(potion["name"]))
    potion["description"] = _resolve_owned_text(potion, loc)


def _combat_log_names(loc: dict[str, str]) -> LogNames:
    """Display-name resolvers for the combat log, off a loc map.

    Shared by the replay/step path and the live-game branch-tree proxy so both
    render identically — the whole point of shipping raw ids on the wire.
    """

    def titled(raw: str) -> str:
        return localization.resolve_name(f"{raw}.title", loc)

    return LogNames(
        card=titled,
        monster=display_maps.monster_title,
        power=display_maps.power_title,
        potion=titled,
        orb=lambda raw: titled(f"{raw}_ORB"),
        character=titled,
    )


def _enrich(
    step: dict[str, Any],
    assets: AssetResolver,
    loc: dict[str, str],
    hp_ranges: dict[str, dict[str, int]] | None = None,
) -> dict[str, Any]:
    """Attach an ``image`` URL to combat entities (None when art is missing).

    Only enriches steps where ``decision == "combat_play"``. Operates on a
    deep copy so the stored step is never mutated.

    Args:
        step: A :class:`workbench.timeline.Step` dict.
        assets: Resolver for game-key → relative image path.
        loc: Localization map for display names.
        hp_ranges: Per-monster initial-HP bands
            (:func:`workbench.localization.load_monster_hp_ranges`), used to
            reproduce the game's per-instance size jitter. Omitted, enemies get
            no jitter and are drawn at their authored size.

    Returns:
        A deep-copied step dict with ``image`` keys added to enemies and hand
        cards, and ``energy_image`` added to the state.
    """
    step = copy.deepcopy(step)
    # Where art is served from. Every image URL below is already absolute, but
    # the card skin composes its own paths (frame/banner/plaque per type+rarity)
    # in the frontend, so it needs the base — which differs in bucket mode.
    step["assets_base"] = _ASSETS_BASE_URL
    state = step.get("state", {})
    # A human-readable label for the action taken from this step, for all
    # decision types (the event log shows non-combat steps too).
    step["action_label"] = render.describe_action(
        _names_resolved(state, loc), step.get("chosen_action")
    )
    # Live-GAME runs carry the game's own combat-log rows, raw (the advisor
    # that wrote them has no localization). Render them here so a followed or
    # replayed live run shows the same log panel a simulated one does.
    raw_log = state.pop("combat_log_raw", None)
    if raw_log:
        state["combat_log"] = format_combat_log(
            raw_log,
            _combat_log_names(loc),
            player_id=str(state.get("character_id") or "PLAYER"),
        )
    decision = step.get("decision")
    player = state.get("player") or {}
    character = player.get("name", "")

    # Relics and potions persist across the whole run, so enrich the player's
    # owned ones on every decision (the relic tray is shown on all screens).
    for relic in player.get("relics") or []:
        _enrich_relic(relic, assets, loc)
    for potion in player.get("potions") or []:
        if potion:
            _enrich_potion(potion, assets, loc)

    # A treasure room auto-resolves in the engine, so the panel showing what
    # the chest held rides on whatever decision comes NEXT (usually the map).
    # Enrich it here, outside the per-decision dispatch below, for that reason.
    reveal = state.get("room_reveal")
    if isinstance(reveal, dict) and isinstance(reveal.get("relic"), dict):
        _enrich_relic(reveal["relic"], assets, loc)

    def enrich_card(
        card: dict[str, Any], *, in_combat: bool = False, osty_alive: bool = False
    ) -> None:
        cid = (card.get("id") or "").split(".")[-1]
        # Mad Science morphs by CardType: its art is one of three per-type event
        # sprites (``mad_science_attack|skill|power``), not a single ``mad_science``
        # slug, so resolve it from the live type rather than the id/name.
        morph = _mad_science_flags(card)
        if morph is not None:
            variant = f"mad_science_{str(morph['cardtype']).lower()}"
            card["image"] = _asset_url(assets.card(character, "", variant))
        else:
            # A run can hold cards from another character (relics, Neow swaps,
            # cross-class rewards); the card's own character (from the manifest)
            # points the resolver at the right atlas folder.
            card["image"] = _asset_url(
                assets.card(
                    character,
                    card.get("id", ""),
                    card.get("name", ""),
                    card_character=loc.get(f"{cid}.character"),
                )
            )
        # The engine sends a generic ``type: "card"``; replace it with the real
        # category (Attack/Skill/Power/…) and attach rarity (drives the frame
        # tint) from the wiki manifest, keyed by card id == loc_key.
        ctype = loc.get(f"{cid}.card_type")
        if ctype:
            card["type"] = ctype
        rarity = loc.get(f"{cid}.rarity")
        if rarity:
            card["rarity"] = rarity
        # Frame tint key for the card skin. Resolved here (not in the frontend)
        # because the card's own character only exists in the manifest.
        card["color"] = _frame_color(
            loc.get(f"{cid}.character"), card.get("type"), card.get("rarity")
        )
        # Regent cards can cost stars as well as energy; the star cost is a
        # per-card constant that lives only in the manifest (never on the wire),
        # so attach it here for the frame's second cost gem.
        star = loc.get(f"{cid}.star_cost")
        if star is not None:
            card["star_cost"] = int(star) if star.lstrip("-").isdigit() else star
        # The Rust backend ships card-reward / deck cards without an energy
        # ``cost`` (only combat hand cards carry one), so the card frame's cost
        # corner renders blank. Backfill it from the manifest (keyed by id) when
        # missing — never clobber a real value (combat cards, the C# backend).
        # The manifest value is already display-ready ("X" for X-cost cards).
        if card.get("cost") is None:
            mcost = _manifest_cost(cid, loc, upgraded=bool(card.get("upgraded")))
            if mcost is not None:
                card["cost"] = mcost
        # Resolve loc keys the game left unresolved (e.g. "PREPARED.title").
        card["name"] = localization.resolve(card.get("name"), loc)
        template = localization.resolve(card.get("description"), loc)
        stats = card.get("stats")
        upgraded = bool(card.get("upgraded"))
        # The Rust backend can't attach a card's full numeric ``stats`` — none at
        # all outside combat (deck / reward / shop views), and only ``damage`` /
        # ``block`` in combat — so a template like ``Apply {VulnerablePower}
        # Vulnerable`` would render its bare variable names. The C# backend ships
        # every DynamicVar in ``stats``, so the gap-filling below never fires there.
        # The game's formatter reads a handful of card-level facts that are
        # not DynamicVars (``CardModel.GetDescriptionForPile``). Two of them
        # decide whether a branch renders at all, so without them the renderer
        # treats the name as a missing *stat* and the card falls back to its
        # at-rest text — Sacrifice's live "(Gain N Block)" never appears.
        # ``IsTargeting`` is always false here: the workbench draws a card face,
        # never a hover-targeted one, which is also the number the game shows
        # until you point at an enemy.
        flags = {"istargeting": False, "isostyalive": osty_alive, **(morph or {})}
        needed = render.required_stat_vars(
            template, in_combat=in_combat, upgraded=upgraded, extra_flags=flags
        )
        have = {k.lower() for k in (stats or {})}
        missing = needed - have
        if missing and in_combat:
            # In combat the wire's ``damage`` / ``block`` carry the card's live
            # (possibly scaled, e.g. Claw / Rampage) values, so fill only the
            # MISSING vars (Vulnerable, Increase, …) from the manifest's base
            # values rather than discarding the live numbers via the plain
            # description. Vars with no manifest entry stay missing → plain.
            filled = dict(stats or {})
            calculated = card.get("calculated_value")
            for var in missing:
                # ``{Calculated*}`` is a value the card only knows on a live
                # board — Ashen Strike's 6 is really 6 + 3 × <Exhaust Pile>.
                # The engine computes it from the same function the card's play
                # handler uses and ships it as ``calculated_value``; when it is
                # absent the var stays MISSING on purpose, so this card takes
                # the plain-description branch below. Filling it from the
                # manifest's ``CalculationBase`` instead would print the game's
                # at-rest number on a live board, and for the ~25 cards whose
                # base is 0 that is a card face reading "(Hits 0 times)".
                if var.startswith("calculated"):
                    if calculated is not None:
                        filled[var] = calculated
                    continue
                base = localization.card_var_value(card.get("id"), var, upgraded, loc)
                if base is not None:
                    filled[var] = int(base) if base.lstrip("-").isdigit() else base
            stats = filled
            have = {k.lower() for k in stats}
            missing = needed - have
        # Outside combat (no live stats) or when a var still has no base value,
        # fall back to the wiki's pre-resolved plain description.
        plain = localization.plain_description(card.get("id"), upgraded, loc) if missing else None
        # Resolved twice: plain for everything that reads text (action labels,
        # logs, tooltips), and with the game's ``[gold]…[/gold]`` keyword markup
        # intact for the card skin, which colours those words like the game.
        if plain:
            card["description"] = render.resolve_card_text(plain, None)
            card["description_rich"] = render.resolve_card_text(plain, None, keep_markup=True)
        else:
            # ``{InCombat:..}`` resolves to its combat branch only on the live
            # board; shop / deck / reward views take the else-branch.
            # ``{IfUpgraded:..}`` follows the card's own upgrade state.
            kwargs: dict[str, Any] = {
                "in_combat": in_combat,
                "upgraded": upgraded,
                "extra_flags": flags,
            }
            card["description"] = render.resolve_card_text(template, stats, **kwargs)
            card["description_rich"] = render.resolve_card_text(
                template, stats, keep_markup=True, **kwargs
            )
        # Keywords print ON the card face, above or below the description
        # (C# ``CardModel.GetDescriptionForPile``) — a Tremble reads "Apply 3
        # Vulnerable." then "Exhaust.". No manifest description carries them,
        # so they are stitched on here, after either branch above.
        before, after = localization.keyword_card_lines(card.get("id"), upgraded, loc)
        # An afflicted card prints its affliction the same way, between the
        # description and the after-keywords (``CardModel.GetDescriptionForPile``
        # appends ``[purple]<extraCardText>[/purple]`` right after the body).
        # This is what makes a Bound card under the Queen's ChainsOfBinding read
        # as Bound: the workbench used to render it identically to a free card,
        # so the fight looked like Chains was doing nothing. ``{Amount}`` is
        # resolved here (Galvanized: "Take 3 damage.").
        affliction_line = localization.affliction_card_line(card.get("affliction"), loc)
        if before or after or affliction_line:
            for field, keep in (("description", False), ("description_rich", True)):
                lines = [render.resolve_card_text(line, None, keep_markup=keep) for line in before]
                body = card.get(field) or ""
                if body:
                    lines.append(body)
                if affliction_line:
                    lines.append(
                        render.resolve_card_text(
                            affliction_line,
                            {"amount": card.get("affliction_amount")},
                            keep_markup=keep,
                        )
                    )
                lines += [render.resolve_card_text(line, None, keep_markup=keep) for line in after]
                card[field] = "\n".join(lines)

    # The full master deck persists across the run, so enrich it on every
    # decision — the deck viewer is reachable from any screen.
    for card in player.get("deck") or []:
        enrich_card(card)

    def enrich_powers(powers: list[dict[str, Any]] | None) -> None:
        for pw in powers or []:
            name = pw.get("name", "")
            # Two icons, because the game has two and draws them in different
            # places: `image` is the 64x64 tray sprite, `image_big` the 256x256
            # `BigIcon` the power VFX use. See `assets.power_big`.
            pw["image"] = _asset_url(assets.power(name))
            pw["image_big"] = _asset_url(assets.power_big(name))

    if decision == "combat_play":
        # Prefer the full-body Spine cutout; fall back to the class-icon portrait.
        player["image"] = _asset_url(
            assets.character_sprite(character) or assets.portrait(character)
        )
        # Live Spine rig base path (frontend appends .skel/.atlas/.png); None ->
        # the combatant stays a static image.
        player["rig"] = _asset_url(assets.character_rig(character))
        pscale = assets.character_rig_scale(character)
        if pscale:
            player["rig_scale"] = pscale.get("scale")
            player["rig_bounds_h"] = pscale.get("bounds_h")
            player["rig_bounds_w"] = pscale.get("bounds_w")
        enrich_powers(state.get("player_powers"))
        # Osty (or any pet mirrored into the ``osty`` slot -- Byrdpip and
        # Pael's Legion summon on ANY character) gets the same sprite/rig
        # treatment as an enemy; the wire carries only hp/block/alive.
        osty = state.get("osty")
        if isinstance(osty, dict) and osty.get("alive"):
            osty.setdefault("name", "Osty")
            osty["image"] = _asset_url(assets.monster("Osty"))
            osty["rig"] = _asset_url(assets.monster_rig("Osty", None))
            oscale = assets.monster_rig_scale("Osty", None)
            if oscale:
                osty["rig_scale"] = oscale.get("scale")
                osty["rig_bounds_h"] = oscale.get("bounds_h")
                osty["rig_bounds_w"] = oscale.get("bounds_w")
        for enemy in state.get("enemies", []) or []:
            # `kind` (raw MonsterKind) disambiguates segments that share one
            # display name (the three Decimillipede segments → distinct rigs).
            # Normalized HERE, once, because the branches below compare it to
            # literal PascalCase kinds ("TestSubject", "SpinyToad", …) and
            # SpireMentor spells it SCREAMING_SNAKE — every one of those checks
            # silently missed on a live run. See `assets.normalize_kind`.
            ekind = normalize_kind(enemy.get("kind"))
            enemy["image"] = _asset_url(assets.monster(enemy.get("name", "")))
            enemy["rig"] = _asset_url(assets.monster_rig(enemy.get("name", ""), ekind))
            escale = assets.monster_rig_scale(enemy.get("name", ""), ekind)
            if escale:
                enemy["rig_scale"] = escale.get("scale")
                enemy["rig_bounds_h"] = escale.get("bounds_h")
                enemy["rig_bounds_w"] = escale.get("bounds_w")
            # Multi-part bosses (Kaiser Crab arms) share ONE rig; tell the frontend
            # which shared skeleton + animation-track family this combatant drives,
            # so it renders the rig once and pins each arm's overlay over its claw.
            group = assets.monster_rig_group(enemy.get("name", ""), ekind)
            if group:
                enemy["rig_group"] = group["group"]
                enemy["rig_part"] = group["part"]
            # Test Subject has 3 revive forms (one more head each time), each
            # with its own idle clip (idle_loop1/2/3) but no dedicated form
            # counter in the wire format — derive it from the revive-only powers
            # that mark form 2 (PainfulStabs) and form 3 (Nemesis, PainfulStabs
            # gone). See TestSubject.cs RespawnMove.
            #
            # Matched through ``_enemy_has_power``, NOT against ``pw["name"]``:
            # by the time this runs the wire power carries a DISPLAY title
            # ("Painful Stabs" — ``rust_game._power`` runs it through
            # ``display_maps.power_title``), so comparing it to the registry id
            # never matched and every Test Subject rendered as its one-headed
            # first form for its whole fight, revives included.
            if ekind == "TestSubject":
                if _enemy_has_power(enemy, "NEMESIS"):
                    enemy["rig_form"] = 3
                elif _enemy_has_power(enemy, "PAINFUL_STABS"):
                    enemy["rig_form"] = 2
            # Monsters whose resting pose depends on combat state — mirror the
            # game's CreatureAnimator branches so the idle clip matches:
            #  * Spiny Toad starts "naked" (idle_naked_loop) and shows its spikes
            #    (idle_loop) only while its Thorns is up (C# ``IsSpiny``, driven
            #    by ThornsPower — SpinyToad.cs).
            #  * Tunneler hides underground (hidden_loop) while it has Burrowed,
            #    else stands in idle_loop (Tunneler.cs ``HasPower<BurrowedPower>``).
            elif ekind == "SpinyToad":
                enemy["rig_anim"] = (
                    "idle_loop" if _enemy_has_power(enemy, "THORNS") else "idle_naked_loop"
                )
            elif ekind == "Tunneler":
                enemy["rig_anim"] = (
                    "hidden_loop" if _enemy_has_power(enemy, "BURROWED") else "idle_loop"
                )
            # The Fake Merchant's rig carries both a (shop-disguise) idle_loop
            # and the combat idle the game actually rests on — pick the combat
            # one (FakeMerchantMonster.cs CreatureAnimator base state).
            elif ekind == "FakeMerchantMonster":
                enemy["rig_anim"] = "combat_idle_loop"
            # Attach the real intent-atlas icon per telegraphed intent (the
            # frontend shows it in place of the emoji glyph). Attack intents
            # pick their dagger tier from total damage (damage × hits).
            for intent in enemy.get("intents") or []:
                dmg = intent.get("damage")
                total = dmg * (intent.get("hits") or 1) if dmg is not None else None
                intent["intent_image"] = _asset_url(assets.intent(intent.get("type", ""), total))
                _enrich_intent_modifiers(intent, assets)
            enrich_powers(enemy.get("powers"))
            # The card a SwipePower thief (Thieving Hopper) is holding hostage.
            # Enriched like a DECK card, not a hand card: a stolen card has been
            # removed from every combat pile, so it carries no live stats and its
            # at-rest text is the right one to print on the frame pinned to the
            # thief.
            for stolen_card in enemy.get("stolen_cards") or []:
                enrich_card(stolen_card)
        # Per-instance size (and tint) for repeated monsters, after the bounds
        # each one is measured by have been stamped above.
        if hp_ranges:
            _apply_instance_jitter(state.get("enemies", []) or [], hp_ranges)
        # Necrobinder's Osty is a player-side minion (serialized as a monster).
        # A dead Osty carries only ``alive: false`` (no name) and gets no sprite.
        osty = state.get("osty")
        if osty and osty.get("name"):
            osty["image"] = _asset_url(assets.monster(osty["name"]))
        # Defect orbs: attach the orb-atlas icon keyed by ``type`` and resolve
        # any unresolved loc-key name (the C# backend already ships a plain
        # name; the Rust backend emits ``<ORB>.title``).
        for orb in state.get("orbs") or []:
            orb["image"] = _asset_url(assets.orb(orb.get("type", "")))
            orb["name"] = localization.resolve(orb.get("name"), loc)
        osty_state = state.get("osty")
        osty_alive = bool(isinstance(osty_state, dict) and osty_state.get("alive"))
        for card in state.get("hand", []) or []:
            enrich_card(card, in_combat=True, osty_alive=osty_alive)
        # Pile contents (cards have the same shape as hand cards). The viewer
        # uses these for inspection. The draw pile is order-scrambled below so
        # the client can't peek at the real draw order ("no cheating").
        for pile_key in ("draw_pile", "discard_pile", "exhaust_pile"):
            for card in state.get(pile_key) or []:
                enrich_card(card, in_combat=True, osty_alive=osty_alive)
        draw = state.get("draw_pile")
        if isinstance(draw, list):
            state["draw_pile"] = sorted(
                draw, key=lambda card: (str(card.get("name", "")), card.get("cost", 0))
            )
        state["energy_image"] = _asset_url(assets.energy(character))
    elif decision in ("card_reward", "card_select"):
        # An upgrade pick renders as before → after, so hand the frontend the
        # upgraded twin of every un-upgraded option (see ``_UPGRADE_CONTEXTS``).
        # The twin is enriched exactly like a real card, so it picks up the
        # wiki's ``upgraded_description_plain`` (numbers already substituted)
        # plus the same art / type / rarity / cost as its base.
        ctx = state.get("context")  # live-mirror logs put a floor/act dict here
        previews = decision == "card_select" and isinstance(ctx, str) and ctx in _UPGRADE_CONTEXTS
        for card in state.get("cards", []) or []:
            if previews and not card.get("upgraded"):
                twin = copy.deepcopy(card)
                twin["upgraded"] = True
                # Any live ``stats`` are the card's UN-upgraded numbers; drop
                # them so the description resolves from the manifest's upgraded
                # base values instead of re-rendering the old ones.
                twin.pop("stats", None)
                # Same for the cost: 57 cards upgrade by dropping an energy, so
                # a deep-copied ``cost`` drew Stampede+ at its un-upgraded 2.
                # Out of combat there is no cost on the wire and clearing it is
                # enough — ``enrich_card`` backfills the upgraded manifest cost
                # off ``twin["upgraded"]``. In combat (Armaments) the option
                # DOES carry a live cost, which may be discounted by a hook, so
                # shift both the effective and the printed cost by the upgrade's
                # delta instead of overwriting them: the preview then keeps the
                # discount the "now" side shows.
                _apply_upgrade_cost(twin, card, loc)
                # ~48 cards upgrade by keyword alone (Afterimage → Innate) and
                # have no upgraded description at all, so the twin would read
                # identically without this delta.
                delta = loc.get(f"{(card.get('id') or '').split('.')[-1]}.upgrade_keywords")
                if delta:
                    twin["keyword_delta"] = delta
                enrich_card(twin)
                card["upgrade_preview"] = twin
            enrich_card(card)
    elif decision == "shop":
        for card in state.get("cards", []) or []:
            enrich_card(card)
            # In the shop, ``cost`` is the gold price; the card frame reads the
            # energy badge from ``card_cost`` (Shop.tsx). The Rust backend ships
            # neither an energy ``cost`` nor ``card_cost`` for shop cards, so the
            # badge would render 0 for every card. Backfill the energy cost from
            # the manifest (keyed by id) — never clobber a real value (C# ships
            # ``card_cost`` directly).
            if card.get("card_cost") is None:
                cid = (card.get("id") or "").split(".")[-1]
                mcost = _manifest_cost(cid, loc, upgraded=bool(card.get("upgraded")))
                if mcost is not None:
                    card["card_cost"] = mcost
        # The shelf wares go through the same enrichment as the ones already in
        # the player's trays: name (the Rust backend emits ``JUZU_BRACELET.title``)
        # resolved before the art lookup, which slugifies the display name, and
        # then the DESCRIPTION. The description is the whole point of the
        # detour: the shop used to resolve name + art only, so the one screen
        # where you decide whether an item is worth your gold was the one screen
        # that never said what the item does (GH: shop relic/potion effects).
        for relic in state.get("relics", []) or []:
            _enrich_relic(relic, assets, loc)
        for potion in state.get("potions", []) or []:
            _enrich_potion(potion, assets, loc)
    elif decision == "bundle_select":
        for bundle in state.get("bundles", []) or []:
            for card in bundle.get("cards", []) or []:
                enrich_card(card)
    elif decision == "treasure":
        for relic in state.get("relics", []) or []:
            relic["name"] = _resolve_name(relic.get("name"), loc)
            relic["image"] = _asset_url(assets.relic(relic["name"]))
    elif decision == "event_choice":
        # The Rust backend emits the event title as a loc key
        # (``LOST_WISP.title``); resolve it (prettifying any our bundle lacks,
        # e.g. ``NEOW.title`` -> "Neow"). The C# backend already ships a plain
        # title, which passes through unchanged.
        state["event_name"] = localization.resolve_name(state.get("event_name"), loc)
        state["art"] = {
            "illustration": _asset_url(assets.event_illustration(state.get("event_name", "")))
        }
        raw_options = state.get("options", []) or []
        # The game ships event/option text as templates with ``{VarName}``
        # placeholders (e.g. ``Gain {SoloGold} Gold.``) plus a per-option
        # ``vars`` dict of resolved numbers. Substitute them so the player sees
        # actual values. ``resolve_card_text`` keys ``vars`` case-insensitively.
        event_vars: dict[str, Any] = {}
        for opt in raw_options:
            for name, val in _event_option_vars(opt, loc).items():
                event_vars[name] = val
        body = localization.resolve(state.get("description"), loc)
        # The Rust backend emits the body as ``<EVENT>.pages.INITIAL.description``;
        # blank it when our bundle has no entry (a leaked dotted key) rather than
        # showing the raw key. Real prose (with spaces) and C# bodies pass through.
        if localization.looks_like_unresolved_key(body):
            body = ""
        state["description"] = render.resolve_card_text(body, event_vars)
        # Stash the rich per-option payload under a dedicated key: ``_live_step``
        # overwrites ``state["options"]`` with the normalized action list, so the
        # frontend reads ``event_options`` for titles/descriptions in both modes.
        options_out: list[dict[str, Any]] = []
        for opt in raw_options:
            opt_vars = _event_option_vars(opt, loc)
            title = localization.resolve(opt.get("title") or opt.get("name"), loc)
            # The Rust backend now names the game's own loc rows here and
            # ships the engine's shorthand alongside as ``fallback_*``; the C#
            # backend sends plain text and no fallbacks. Only substitute when
            # a fallback actually exists, so the C# path keeps its old
            # behaviour.
            if localization.looks_like_unresolved_key(title) and opt.get("fallback_title"):
                title = opt["fallback_title"]
            # Five option titles take variables of their own ("Accept the
            # {Relic}", "Give {Gold} Gold") — same fall-back-if-unsupplied rule
            # the description below uses, or the player reads a bare var name.
            # Gated on a literal brace so a plain title (every C# one, and most
            # of ours) reaches the frontend byte-for-byte as before.
            if "{" in title:
                # A handful of options give the TITLE its own values, because
                # a var name can mean two things across the two templates —
                # The Future of Potions' `{Rarity}` is the POTION's in the
                # title and the CARD's in the description.
                title_vars = {**opt_vars, **_event_option_vars(opt, loc, key="title_vars")}
                if render.required_stat_vars(title, extra_flags=_EVENT_FLAGS) - set(title_vars):
                    title = opt.get("fallback_title") or title
                title = render.resolve_card_text(title, title_vars, extra_flags=_EVENT_FLAGS)
            template = localization.resolve(opt.get("description"), loc)
            # `{BoneTeaDescription}` and friends splice ANOTHER model's rules
            # text in; that lives in a different loc table, so the engine can't
            # ship it. Resolve it here from the item the name points at.
            opt_vars.update(
                localization.named_description_vars(
                    render.required_stat_vars(template), set(opt_vars), loc
                )
            )
            # A template whose vars nobody supplied prints the variable NAME
            # ("Gain Gold Gold."). Fall back to the engine's own shorthand —
            # wrong register, but never nonsense. Same rule the card face uses
            # with ``description_plain``.
            unrenderable = render.required_stat_vars(template, extra_flags=_EVENT_FLAGS) - set(
                opt_vars
            ) or (localization.looks_like_unresolved_key(template))
            if unrenderable and opt.get("fallback_description"):
                template = opt["fallback_description"]
            desc = render.resolve_card_text(template, opt_vars, extra_flags=_EVENT_FLAGS)
            # Neow relic options carry no localized description from C#; the
            # relic id is the tail of the text_key (e.g.
            # "NEOW.pages.INITIAL.options.WINGED_BOOTS" -> "WINGED_BOOTS").
            reward_id = (opt.get("text_key") or "").rsplit(".", 1)[-1]
            image = _asset_url(assets.relic(reward_id)) if reward_id else None
            if not desc and reward_id:
                # Prefer the wiki's pre-resolved plain description (numbers
                # already substituted): the relic/potion's own ``{Cards}`` /
                # ``{Damage}`` vars aren't in the option's ``vars`` dict, so the
                # raw ``.description`` template would render bare variable names
                # ("remove Cards cards … lose Damage HP"). Fall back to the raw
                # template only when no plain text exists. ``resolve_card_text``
                # still strips the BBCode tags either way.
                fallback = localization.plain_owned_description(reward_id, loc) or loc.get(
                    f"{reward_id}.description"
                )
                if fallback:
                    desc = render.resolve_card_text(fallback, opt_vars)
            options_out.append(
                {
                    "index": opt.get("index"),
                    "title": title,
                    "description": desc,
                    "image": image,
                    "is_locked": bool(opt.get("is_locked", False)),
                }
            )
        state["event_options"] = options_out
    elif decision == "rest_site":
        state["rest_options"] = [
            {
                "index": opt.get("index"),
                "option_id": opt.get("option_id"),
                "name": localization.resolve(opt.get("name") or opt.get("option_id"), loc),
                "is_enabled": bool(opt.get("is_enabled", True)),
            }
            for opt in state.get("options", []) or []
        ]
    elif decision == "map_select":
        # The live map picker (MapChoices.tsx) renders the immediately-reachable
        # ``choices``; the full connected graph (MapView.tsx) rides under
        # ``full_map``. Give both the real per-room-type map icons.
        for choice in state.get("choices") or []:
            choice["icon"] = _asset_url(assets.room_icon(choice.get("type")))
        _enrich_map_nodes(state.get("full_map"), assets)
    return step


def _enrich_intent_modifiers(intent: dict[str, Any], assets: AssetResolver) -> None:
    """Give each ``intent["modifiers"]`` row a display title and an icon.

    The engine ships registry ids (``WEAK``, ``BURNING_BLOOD``) — the same
    vocabulary a power carries — so the board can draw a modifier with the very
    icon the status tray uses and name it in a sentence. Without the icon the
    breakdown is a wall of words next to a numeral; the point of it is that a
    player recognises "oh, the Weak is already in there" at a glance (bug
    report ``wb-4eae3bad6ebd57f8``).

    The ``other`` row (the card-enchantment and relic-registry stages, which
    are not per-listener and so cannot be named) keeps its title and gets no
    icon — it exists so the printed arithmetic still reconciles.
    """
    for mod in intent.get("modifiers") or []:
        name = str(mod.get("name", ""))
        kind = mod.get("kind")
        if kind == "power":
            mod["title"] = display_maps.power_title(name)
            mod["image"] = _asset_url(assets.power(mod["title"]))
        elif kind == "relic":
            mod["title"] = name.replace("_", " ").title()
            mod["image"] = _asset_url(assets.relic(mod["title"]))
        else:
            mod["title"] = "Other"


def _live_step(
    state: dict[str, Any],
    assets: AssetResolver,
    loc: dict[str, str],
    hp_ranges: dict[str, dict[str, int]] | None = None,
) -> dict[str, Any]:
    """Wrap a bare live state in a Step-shaped, enriched dict for the frontend.

    Reuses :func:`_enrich` so live combat renders through the same components
    as replay, then attaches normalized ``options`` for non-combat decisions.
    """
    step = {
        "idx": 0,
        "decision": state.get("decision", "unknown"),
        "state": state,
        "scores": [],
        "pv": None,
        "chosen_action": None,
        "source": "live",
        # ``floor``/``act`` must ride along: the docked minimap refetches the
        # act graph only when these change (App.tsx effect deps). Omitting them
        # leaves the deps ``undefined`` forever, so the effect never re-fires and
        # the minimap freezes on the first map it fetched.
        "meta": {
            "turn": state.get("round"),
            "step_no": 0,
            "floor": state.get("floor"),
            "act": state.get("act"),
            "act_name": state.get("act_name"),
            # The map node this fight stands on, for the room's variant roll
            # (see `roomScenes.resolveScene`). Absent leaves the room on its
            # stable per-floor hash.
            "bg_seed": state.get("bg_seed"),
            "bg_row": state.get("bg_row"),
            "bg_col": state.get("bg_col"),
        },
    }
    enriched = _enrich(step, assets, loc, hp_ranges)
    enriched["state"]["options"] = normalize_options(state)
    return enriched


def _faithful_replay_plan(
    store: Any, idx: int, config: dict[str, Any] | None
) -> tuple[str, list[dict[str, Any]], dict[str, Any]] | None:
    """Plan a faithful native re-search of step ``idx``, or ``None`` to fall back.

    For a fixture-sourced recorded run, replaying the logged native actions over
    a fresh fixture load reproduces the *exact* engine state (RNG cursor +
    intent history included) — so the re-search matches the recorder's saved
    reasoning instead of diverging like the lossy ``state_from_dict``
    reconstruction. The MCTS seed defaults to the recorder's so the RNG path
    matches; sims/depth/c_puct come from the caller's request.

    Returns ``(fixture_path, native_actions, config)`` or ``None`` when no
    fixture is available (full-run / C# recordings) or its file is missing.
    """
    fixture_path = store.fixture_path()
    if not isinstance(fixture_path, str) or not Path(fixture_path).is_file():
        return None
    actions = store.native_actions_before(idx)
    if actions is None:
        return None
    # Pin the re-search to the RECORDED config (c_puct / seed / rollout_policy)
    # so it reproduces the saved reasoning, honoring only the search-strength
    # knobs (sims / depth) from the caller. The frontend always sends its own
    # c_puct default (1.4) which, if it won, would flip the "best" move vs the
    # recorder's c_puct=1.0 on an otherwise identical state — the exact
    # Dominate-vs-Fasten divergence. The returned tree echoes the real c_puct,
    # so the UI shows what was actually used.
    cfg = dict(store.mcts_config())
    for knob in ("sims", "depth"):
        val = (config or {}).get(knob)
        if val is not None:
            cfg[knob] = val
    return fixture_path, actions, cfg


def create_app(
    images_root: str | Path,
    logs_dir: str | Path,
    *,
    serve_spa: bool = False,
    assets_manifest: str | Path | None = None,
) -> FastAPI:
    """Create and return the workbench FastAPI application.

    Args:
        images_root: Path to the directory containing extracted game art.
            Mounted as ``/assets`` only when art is served locally (i.e.
            ``STS2_ASSETS_BASE_URL`` is unset/``/assets``) and the directory
            exists. In manifest mode this only needs to hold the small
            ``creature_scales.json``.
        logs_dir: Directory to scan for ``*.jsonl`` run logs.
        serve_spa: When True, mount the built Vite SPA (``web/dist``) at ``/`` so
            a single container serves both the API and the UI (the Cloud Run
            path). When True, the SimLab API mount is skipped (both want ``/``;
            SimLab is a separate dev-only concern needing numpy/scipy).
        assets_manifest: Optional path to an ``asset_manifest.json`` (a JSON list
            of relative art paths). When given, :class:`AssetResolver` resolves
            existence against this set instead of the local filesystem, so the
            art tree need not be present — pair it with ``STS2_ASSETS_BASE_URL``
            pointing at the bucket that holds the art.

    Returns:
        A configured :class:`fastapi.FastAPI` instance ready for uvicorn.
    """
    # Per-client state lives in a session registry (not on two process globals),
    # so each browser pointing at the same deployment gets its OWN loaded replay
    # and live game instead of sharing one. The session id is an opaque cookie
    # minted on first contact; see :mod:`workbench.session`.
    sessions = SessionRegistry()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        """Release every session's engine resources when the server stops."""
        # Say so at startup if bug-report persistence is configured but broken.
        # The capture path falls back to a container-local file on any failure,
        # which on Cloud Run is silently destroyed at scale-to-zero — so this is
        # the only place the difference is visible.
        if problem := bugreport_store.check_config(os.environ.get("STS2_BUGREPORT_BUCKET")):
            _log.error("bug-report storage misconfigured: %s", problem)
        yield
        sessions.close_all()

    app = FastAPI(title="STS2 Workbench", lifespan=lifespan)
    app.state.sessions = sessions
    manifest: set[str] | None = None
    if assets_manifest:
        manifest = set(json.loads(Path(assets_manifest).read_text(encoding="utf-8")))
    assets = AssetResolver(images_root, manifest=manifest)
    logs_dir = Path(logs_dir)

    def get_session(request: Request) -> Session:
        """Resolve the caller's :class:`Session` from the id on the ASGI scope.

        :class:`~workbench.session.SessionCookieMiddleware` minted that id into
        the client's cookie (and onto the scope) before any handler ran, so a
        fresh browser's concurrent first requests all resolve to one session
        rather than racing to create their own. The ``or`` is defensive: the
        middleware always populates the scope for HTTP requests.
        """
        sid = request.scope.get(SID_SCOPE_KEY) or uuid.uuid4().hex
        return sessions.get_or_create(sid)

    # Card loc-key fallback: in-repo localization + sibling wiki manifest.
    repo_root = Path(__file__).resolve().parents[2]
    images_path = Path(images_root)
    # images_root is <wiki>/site/public/images -> wiki data dir is <wiki>/data
    wiki_data = images_path.parents[2] / "data" if len(images_path.parents) >= 3 else None
    card_loc = localization.load_card_localization(
        repo_root / "localization_eng" / "cards.json", wiki_data
    )
    # Per-monster initial-HP bands, which no wire carries: the game sizes each
    # member of a repeated-monster group by where its rolled max HP landed in
    # its band (`NCombatRoom.RandomizeEnemyScalesAndHues`). Empty on a deploy
    # with no wiki manifest, which simply leaves every enemy at its authored
    # size — what the board did before.
    hp_ranges = localization.load_monster_hp_ranges(wiki_data)
    # Display titles for the keywords a card prints on its face ("Exhaust.",
    # "Ethereal.", …) — keyed ``card_keyword.<Name>`` so they cannot collide
    # with a card/relic loc key.
    card_loc.update(localization.load_card_keyword_localization(repo_root / "localization_eng"))
    # Same idea for the afflictions the game prints on a card face ("Bound"):
    # keyed ``affliction.<ID>.<field>``, so an afflicted card can say so.
    card_loc.update(localization.load_affliction_localization(repo_root / "localization_eng"))
    # Rest-site option names ("Rest", "Smith", "Kindle", …) — keyed
    # ``rest_site.OPTION_<ID>.name`` off the engine's own ``OptionId``, so a
    # relic-added option (Dig / Lift / Kindle / Cook / Hatch) is named by the
    # game's own row instead of falling back to a numbered button.
    card_loc.update(localization.load_rest_site_localization(repo_root / "localization_eng"))
    # Owns the live-run slot of every client session: starting, restoring and
    # replacing a run all go through it (see ``workbench.live_runs``).
    live_runs = LiveRuns(logs_dir, card_loc)
    # The engine ships every card with a generic ``type: "card"`` and no rarity;
    # fold the real category + rarity (keyed by card id == manifest loc_key) into
    # the loc map as flat ``<id>.card_type`` / ``<id>.rarity`` keys so ``_enrich``
    # can resolve them without a new signature thread-through.
    for key, meta in localization.load_card_meta(wiki_data).items():
        if meta.get("card_type"):
            card_loc.setdefault(f"{key}.card_type", meta["card_type"])
        if meta.get("rarity"):
            card_loc.setdefault(f"{key}.rarity", meta["rarity"])
        if meta.get("cost") is not None:
            card_loc.setdefault(f"{key}.cost", meta["cost"])
        if meta.get("upgraded_cost") is not None:
            card_loc.setdefault(f"{key}.upgraded_cost", meta["upgraded_cost"])
    # Relic/potion loc keys (flat ``KEY.title``/``KEY.description`` dicts) so a
    # relic/potion whose name arrives as a raw key resolves to a display
    # string. Keys are namespaced, so merging into the same map is safe.
    for fname in ("relics.json", "potions.json"):
        path = repo_root / "localization_eng" / fname
        if path.is_file():
            try:
                card_loc.update(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                pass
    # Character titles (``IRONCLAD.title`` -> "The Ironclad"), for the combat
    # log's actor names. Merged with setdefault so a colliding card/relic key
    # keeps its meaning; the five character ids collide with nothing today.
    characters_path = repo_root / "localization_eng" / "characters.json"
    if characters_path.is_file():
        try:
            for key, value in json.loads(characters_path.read_text(encoding="utf-8")).items():
                if isinstance(value, str):
                    card_loc.setdefault(key, value)
        except (OSError, json.JSONDecodeError):
            pass
    # Pre-resolved relic/potion descriptions (numbers already substituted) from
    # the wiki per-item data — the display fallback for the Rust backend, which
    # can't ship a ``vars`` dict. Keyed ``<KEY>.description_plain``; merged
    # without overwriting the templates above.
    for owned_key, owned_plain in localization.load_owned_localization(wiki_data).items():
        card_loc.setdefault(owned_key, owned_plain)

    # Event titles + page bodies (the Rust backend emits ``<EVENT>.title`` /
    # ``<EVENT>.pages.INITIAL.description`` keys). Merge WITHOUT overwriting: a
    # handful of event ids collide with the relic they grant (e.g.
    # ``LOST_WISP.title`` = "The Lost Wisp" vs the relic's "Lost Wisp"), and the
    # relic name must win for relic displays — the event header falling back to
    # the relic name reads fine. Unique event keys (most titles, all bodies)
    # still land.
    events_path = repo_root / "localization_eng" / "events.json"
    if events_path.is_file():
        try:
            for key, val in json.loads(events_path.read_text(encoding="utf-8")).items():
                card_loc.setdefault(key, val)
        except (OSError, json.JSONDecodeError):
            pass

    @app.get("/api/health")
    def health() -> dict[str, bool]:
        """Return ``{"ok": True}`` — liveness probe."""
        return {"ok": True}

    @app.get("/api/capabilities")
    def capabilities() -> dict[str, list[str]]:
        """Which game engines this deployment can actually run.

        The SPA's engine dropdown is built from this. The deployed image is
        deliberately C#-free (root ``Dockerfile`` stage 3), so offering "C#"
        there was a dead end: the pick spawned ``dotnet``, which is absent, and
        the resulting 500 left "Start run" looking like it did nothing.
        """
        backends = ["rust"]
        if sts2_client.headless_available():
            backends.append("csharp")
        return {"backends": backends}

    @app.get("/api/runs")
    def runs() -> dict[str, list[str]]:
        """List ``*.jsonl`` log files in ``logs_dir``, newest first.

        By mtime, not by name. Not every writer date-prefixes its files: the
        MCP server drops ``mcp_<date>_<pid>.jsonl`` in this same directory,
        and ``m`` sorts above every ``2026…`` run log, so a name sort pinned a
        tool log to the head forever. Follow mode takes that head whenever the
        advisor cannot name its own file, so the workbench sat tailing a 5 KB
        MCP log while a real run played (2026-08-22).
        """

        def _mtime(p: Path) -> float:
            # A file can vanish between glob and stat; sort it last rather
            # than fail the whole listing.
            try:
                return p.stat().st_mtime
            except OSError:
                return 0.0

        # Name breaks mtime ties so the order is stable (same-second writes).
        ordered = sorted(logs_dir.glob("*.jsonl"), key=lambda p: (_mtime(p), p.name), reverse=True)
        return {"runs": [str(p) for p in ordered]}

    # ------------------------------------------------------------------
    # Live-game bridge proxies (rewind lane). The advisor process owns the
    # one TCP socket to the SpireMentor mod and exposes a localhost control
    # plane (realgame_control.py, default 127.0.0.1:8079); these thin
    # proxies keep the SPA same-origin. Not session-scoped: there is one
    # live game per machine.
    # ------------------------------------------------------------------

    advisor_control = os.environ.get("STS2_ADVISOR_CONTROL", "127.0.0.1:8079")

    def _bridge_call(
        path: str, payload: dict[str, Any] | None = None, timeout: float = 30.0
    ) -> dict[str, Any]:
        import urllib.error
        import urllib.request

        url = f"http://{advisor_control}{path}"
        try:
            if payload is None:
                req = urllib.request.Request(url)
            else:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"content-type": "application/json"},
                    method="POST",
                )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            try:
                detail = json.loads(exc.read())
            except Exception:
                detail = {"error": str(exc)}
            raise HTTPException(exc.code, detail) from exc
        except (OSError, ValueError) as exc:
            raise HTTPException(
                503,
                "advisor control plane not reachable at "
                f"{advisor_control} — is the live-game advisor running? ({exc})",
            ) from exc

    @app.get("/api/bridge/status")
    def bridge_status() -> dict[str, Any]:
        """Live-game advisor status (connected / character / seed / floor)."""
        return _bridge_call("/status", timeout=3.0)

    @app.get("/api/bridge/tree")
    def bridge_tree() -> dict[str, Any]:
        """The live run's branch tree, in the LiveTimeline wire shape.

        The advisor stores each node's combat-log rows RAW (game ids, no
        display names) because it has no localization; render them here, where
        the loc map lives, so the live-game panel goes through the exact same
        formatter as a simulated run.
        """
        tree = _bridge_call("/tree", timeout=5.0)
        names = _combat_log_names(card_loc)
        for node in tree.get("nodes") or []:
            if not isinstance(node, dict):
                continue
            raw = node.pop("combat_log_raw", None)
            player_id = node.pop("player_id", None) or "PLAYER"
            if raw:
                node["combat_log"] = format_combat_log(raw, names, player_id=str(player_id))
        return tree

    @app.get("/api/bridge/snapshots")
    def bridge_snapshots() -> dict[str, Any]:
        """The mod's snapshot list for the current live run."""
        return _bridge_call("/snapshots", timeout=10.0)

    @app.get("/api/bridge/state")
    def bridge_state() -> dict[str, Any]:
        """The live game's latest decision position (what Copy state yields),
        for the state import; 404 before the game's first decision."""
        return _bridge_call("/state", timeout=5.0)

    @app.post("/api/bridge/seek")
    def bridge_seek(req: BridgeSeekRequest) -> dict[str, Any]:
        """Rewind the LIVE game to a branch-tree node's save point."""
        return _bridge_call(
            "/restore",
            {"node_id": req.node_id, "snapshot_id": req.snapshot_id},
            timeout=30.0,
        )

    @app.post("/api/bridge/search-tree")
    def bridge_search_tree(req: BridgeSearchTreeRequest) -> dict[str, Any]:
        """Companion MCTS tree over the live game's current combat board."""
        return _bridge_call("/search_tree", {"sims": req.sims}, timeout=120.0)

    @app.get("/api/runs/stat")
    def runs_stat(path: str) -> dict[str, Any]:
        """Cheap change probe for one run file: ``{size, mtime}``.

        The SPA's Follow mode polls this every couple of seconds and only
        re-issues the (much heavier) ``/api/load`` when either value moved —
        a still-being-written live-game log grows in place, so size+mtime is
        a sufficient dirty bit. Constrained to ``logs_dir`` (unlike
        ``/api/load``'s TODO, a poll endpoint should not leak arbitrary
        file metadata).
        """
        p = Path(path)
        try:
            resolved = p.resolve()
            resolved.relative_to(logs_dir.resolve())
        except (OSError, ValueError) as exc:
            raise HTTPException(400, f"path outside logs dir: {path}") from exc
        try:
            st = resolved.stat()
        except (FileNotFoundError, OSError) as exc:
            raise HTTPException(404, f"file not found: {exc}") from exc
        return {"size": st.st_size, "mtime": st.st_mtime}

    @app.get("/api/mcts-runs")
    def mcts_runs() -> dict:
        """List recorded MCTS runs (summary rows) parsed from ``logs_dir``."""
        runs = []
        for p in sorted(logs_dir.glob("*.jsonl")):
            r = _parse_run(p)
            if r:
                runs.append(
                    {
                        "stem": r["stem"],
                        "won": r["won"],
                        "final_turn": r["final_turn"],
                        "final_floor": r["final_floor"],
                        "turn1_q": r["turn1_q"],
                        "enemy_min_pct": r["enemy_min_pct"],
                        "path": p.name,
                    }
                )
        return {"runs": runs}

    @app.get("/api/mcts-runs/aggregate")
    def mcts_runs_aggregate() -> dict:
        """Aggregate loss patterns across the recorded MCTS runs in ``logs_dir``."""
        return aggregate_runs(str(logs_dir))

    @app.post("/api/load")
    def load(req: LoadRequest, session: Session = Depends(get_session)) -> dict[str, Any]:
        """Load a JSONL run into the in-memory timeline store.

        Args:
            req: ``{"path": "<absolute path to .jsonl>"}``

        Returns:
            ``{"count": <steps loaded>, "resume": {...}}`` — ``resume`` is
            :func:`~workbench.live_runs.resume_capability`, which tells the UI
            whether (and where) "play from here" can take this recording over.
        """
        # TODO: constrain `req.path` to within logs_dir before exposing beyond localhost.
        try:
            session.store = TimelineStore.load_jsonl(req.path)
        except (FileNotFoundError, OSError) as exc:
            raise HTTPException(404, f"file not found: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise HTTPException(422, f"invalid JSONL: {exc}") from exc
        return {"count": len(session.store), "resume": resume_capability(session.store)}

    @app.get("/api/timeline")
    def timeline(session: Session = Depends(get_session)) -> dict[str, list[dict[str, Any]]]:
        """Return a lightweight summary of every step for the event log.

        Each entry carries the friendly action label plus the carried-forward
        act/floor/turn so the UI can render a navigable, human-readable run
        history without fetching every full step.

        Raises:
            HTTPException: 400 if no run is loaded.
        """
        store = session.store
        if store is None:
            raise HTTPException(400, "no run loaded")
        steps = store.all()
        resolved = [_names_resolved(s["state"], card_loc) for s in steps]

        # Per-step combat-log rows: workbench-saved runs record each action's
        # display rows directly; bridge logs instead carry the raw engine log
        # on every state, so a step's slice is the tail its successor's log
        # grew by (a shorter successor means a new fight began — take it all).
        names = _combat_log_names(card_loc)
        player_id = store.character().upper().replace(" ", "_")

        def _step_rows(i: int) -> list[dict[str, Any]]:
            recorded = steps[i].get("combat_log") or []
            if recorded:
                return recorded
            if i + 1 >= len(steps):
                return []
            raw0 = steps[i]["state"].get("combat_log_raw") or []
            raw1 = steps[i + 1]["state"].get("combat_log_raw") or []
            tail = raw1[len(raw0) :] if len(raw1) >= len(raw0) else raw1
            return format_combat_log(tail, names, player_id=player_id) if tail else []

        out = [
            {
                "idx": s["idx"],
                "decision": s["decision"],
                "label": render.describe_action(resolved[i], s["chosen_action"]),
                "details": (
                    diff_states(resolved[i], resolved[i + 1]) if i + 1 < len(steps) else []
                ),
                "combat_log": _step_rows(i),
                "turn": s["meta"].get("turn"),
                "floor": s["meta"].get("floor"),
                "act": s["meta"].get("act"),
                "act_name": s["meta"].get("act_name"),
            }
            for i, s in enumerate(steps)
        ]
        return {"steps": out}

    @app.get("/api/map")
    def game_map(session: Session = Depends(get_session)) -> dict[str, Any]:
        """Reconstruct the run's map path from all ``map_select`` steps.

        The log records only the reachable choices at each map decision (not
        the full graph), so we assemble one row per ``map_select`` step with
        its node choices, marking the node that was actually taken.

        Raises:
            HTTPException: 400 if no run is loaded.
        """
        store = session.store
        if store is None:
            raise HTTPException(400, "no run loaded")
        rows: list[dict[str, Any]] = []
        graphs: list[dict[str, Any]] = []
        for s in store.all():
            if s["decision"] != "map_select":
                continue
            state = s["state"]
            taken = s.get("chosen_action") or {}
            t_args = taken.get("args") or {}
            tcol, trow = t_args.get("col"), t_args.get("row")
            nodes = [
                {
                    "col": c.get("col"),
                    "row": c.get("row"),
                    "type": c.get("type"),
                    "chosen": c.get("col") == tcol and c.get("row") == trow,
                    "icon": _asset_url(assets.room_icon(c.get("type"))),
                }
                for c in (state.get("choices") or [])
            ]
            rows.append(
                {
                    "idx": s["idx"],
                    "floor": s["meta"].get("floor"),
                    "act": s["meta"].get("act"),
                    "act_name": s["meta"].get("act_name"),
                    "nodes": nodes,
                }
            )
            # Full connected graph, captured by the recorder when available.
            # Enrich a copy so the persisted store state isn't mutated.
            if state.get("full_map"):
                graphs.append(
                    {
                        "idx": s["idx"],
                        "act": s["meta"].get("act"),
                        "act_name": s["meta"].get("act_name"),
                        "floor": s["meta"].get("floor"),
                        "full_map": _enrich_map_nodes(copy.deepcopy(state["full_map"]), assets),
                    }
                )
        return {"rows": rows, "graphs": graphs}

    @app.get("/api/step/{idx}")
    def step(idx: int, session: Session = Depends(get_session)) -> dict[str, Any]:
        """Return the step at *idx*, enriched with resolved art URLs.

        Args:
            idx: Zero-based step index.

        Returns:
            Enriched :class:`workbench.timeline.Step` dict.

        Raises:
            HTTPException: 400 if no run is loaded; 404 if *idx* is out of
                range.
        """
        store = session.store
        if store is None:
            raise HTTPException(400, "no run loaded")
        try:
            s = store.get(idx)
        except IndexError as exc:
            raise HTTPException(404, str(exc)) from exc
        return _enrich(dict(s), assets, card_loc, hp_ranges)

    def _deck_of(state: dict[str, Any]) -> list[dict[str, Any]]:
        return (state.get("player") or {}).get("deck") or []

    @app.get("/api/step/{idx}/deck-analysis")
    def step_deck_analysis(idx: int, session: Session = Depends(get_session)) -> dict[str, Any]:
        """Grade the master deck at step *idx* on Damage / Block / Velocity.

        Returns the :func:`workbench.deck_analysis.analyze_deck` report. The deck
        persists across the run, so this works on every decision type (combat,
        reward, map, shop), not just combat.
        """
        store = session.store
        if store is None:
            raise HTTPException(400, "no run loaded")
        try:
            s = store.get(idx)
        except IndexError as exc:
            raise HTTPException(404, str(exc)) from exc
        return deck_analysis.analyze_deck(_deck_of(s.get("state", {})))

    @app.get("/api/live/deck-analysis")
    def live_deck_analysis(session: Session = Depends(get_session)) -> dict[str, Any]:
        """Grade the current live run's master deck. 400 if no live run."""
        sess = session.live
        if sess is None:
            raise HTTPException(400, "no live run")
        return deck_analysis.analyze_deck(_deck_of(sess.state()))

    @app.post("/api/deck/reward-preview")
    def deck_reward_preview(req: RewardPreviewRequest) -> dict[str, Any]:
        """Preview how each offered card would shift the deck's three axes.

        Stateless — the client passes the current deck and the candidate cards
        (from the card-reward screen), so this works in both replay and live.
        Returns baseline scores plus a per-candidate delta (see
        :func:`workbench.deck_analysis.reward_preview`).
        """
        return deck_analysis.reward_preview(req.deck, req.candidates)

    @app.get("/api/deck/trend")
    def deck_trend(session: Session = Depends(get_session)) -> dict[str, Any]:
        """Axis scores across the loaded run, one point per deck change.

        Walks the replay timeline and re-grades the master deck whenever it
        changes (a pickup / removal / upgrade), so the client can sparkline how
        Damage / Block / Velocity / Volume evolved over the run. 400 if no run
        is loaded.
        """
        store = session.store
        if store is None:
            raise HTTPException(400, "no run loaded")
        points: list[dict[str, Any]] = []
        last_sig: tuple | None = None
        for s in store.all():
            deck = _deck_of(s.get("state", {}))
            if not deck:
                continue
            sig = deck_analysis.deck_signature(deck)
            if sig == last_sig:
                continue
            last_sig = sig
            # Lighter sampling than the on-demand panel — a trend needs the shape,
            # not three-decimal precision, and this re-grades at every deck change
            # over the whole run (each grade is a multi-turn simulation).
            report = deck_analysis.analyze_deck(deck, samples=60)
            meta = s.get("meta") or {}
            points.append(
                {
                    "idx": s.get("idx"),
                    "floor": meta.get("floor"),
                    "act": meta.get("act"),
                    "deck_size": report.get("deck_size"),
                    "scores": report.get("scores"),
                }
            )
        return {"points": points}

    @app.post("/api/recommend")
    def recommend(req: RecommendRequest, session: Session = Depends(get_session)) -> dict[str, Any]:
        """Run the MCTS recommender on step *idx* and cache the result.

        Calls ``reasoning.score(state, config)`` then writes ``scores`` and
        ``pv`` back onto the stored step so subsequent ``/api/step`` responses
        include them.

        Args:
            req: ``{"idx": <step index>, "config": {...optional MCTS params}}``

        Returns:
            The ``reasoning.score`` result dict
            (``{scores, pv, config}`` or ``{error, scores, pv, config}``).

        Raises:
            HTTPException: 400 if no run is loaded; 404 if *idx* is out of
                range.
        """
        store = session.store
        if store is None:
            raise HTTPException(400, "no run loaded")
        try:
            s = store.get(req.idx)
        except IndexError as exc:
            raise HTTPException(404, str(exc)) from exc
        plan = _faithful_replay_plan(store, req.idx, req.config)
        if plan is not None:
            fixture_path, actions, cfg = plan
            result = reasoning.score_fixture_replay(fixture_path, actions, s["state"], cfg)
        else:
            result = reasoning.score(s["state"], req.config)
        # Cache back onto the step so subsequent /api/step returns it too.
        s["scores"] = result.get("scores", [])
        s["pv"] = result.get("pv")
        return result

    @app.post("/api/regret-review")
    def regret_review_endpoint(
        req: RegretReviewRequest, session: Session = Depends(get_session)
    ) -> dict[str, Any]:
        """Post-combat regret review: diff every combat decision in the loaded
        run against the MCTS line and surface the costliest plays.

        Uses each step's recorded MCTS ``scores`` when present (faithful to what
        the recorder saw). For steps without recorded scores — or for every step
        when ``config.recompute`` is set — it re-runs MCTS (faithful fixture
        replay when the run is fixture-sourced, else a dict reconstruction).

        Args:
            req: ``{"config": {"blunder_pct"?, "top_n"?, "recompute"?,
                ...MCTS params}}``.

        Returns:
            ``{"decisions": [...], "blunders": [...top 1-2...], "summary": {...},
            "config": {...}}``.

        Raises:
            HTTPException: 400 if no run is loaded.
        """
        store = session.store
        if store is None:
            raise HTTPException(400, "no run loaded")
        cfg = dict(req.config or {})
        recompute = bool(cfg.pop("recompute", False))
        blunder_pct = float(cfg.pop("blunder_pct", regret_review.DEFAULT_BLUNDER_DROP_PCT))
        top_n = int(cfg.pop("top_n", 2))
        mcts_cfg = cfg  # whatever remains are MCTS params for a re-search

        def _rescore(step: dict[str, Any]) -> list[dict[str, Any]]:
            idx = int(step.get("idx", 0))
            plan = _faithful_replay_plan(store, idx, mcts_cfg)
            if plan is not None:
                fixture_path, actions, c = plan
                result = reasoning.score_fixture_replay(fixture_path, actions, step["state"], c)
            else:
                result = reasoning.score(step["state"], mcts_cfg)
            return result.get("scores", [])

        steps = store.all()
        # Resolve display names BEFORE anything reads the state. A run saved by
        # the live workbench persists the raw engine state, whose card/potion
        # names are localization keys, so every label the review builds off
        # `state["hand"]` came out as `DEFEND_IRONCLAD.title`. Every other
        # replay surface goes through `_names_resolved`/`_enrich` on the way
        # out; this one scored the raw dict. Mirrors the live twin
        # (`BranchingLiveSession.review_run`), which resolves for the same
        # reason — and the resolved names round-trip through the scorer fine,
        # since the adapter maps display names AND leaked loc keys alike.
        # A RECORDED row carries its own label, so resolving the state is not
        # enough for it — a log written before the labels were resolved at
        # record time still holds raw keys there.

        def _named_step(s: dict[str, Any]) -> dict[str, Any]:
            rows = [
                {**r, "action": localization.resolve_name(r.get("action"), card_loc)}
                if r.get("action")
                else r
                for r in (s.get("scores") or [])
            ]
            return {**s, "state": _names_resolved(s["state"], card_loc), "scores": rows}

        steps = [_named_step(s) for s in steps]
        if recompute:
            # Force a fresh search on every combat step by blanking the
            # recorded scores so review_steps falls through to _rescore.
            steps = [
                {**s, "scores": []}
                if (s.get("state") or {}).get("decision") == "combat_play"
                else s
                for s in steps
            ]
        return regret_review.review_steps(
            steps, blunder_pct=blunder_pct, top_n=top_n, rescore=_rescore
        )

    @app.post("/api/wpa-review")
    def wpa_review_endpoint(
        req: RegretReviewRequest, session: Session = Depends(get_session)
    ) -> dict[str, Any]:
        """WPA run review of the loaded replay — the run-level twin of
        ``/api/regret-review`` (issue #201).

        Reads the per-decision PPO values the live session recorded into the
        run log (``wpa`` records; see ``BranchingLiveSession.quit``) and
        returns the value curve + turning points. No net is needed at review
        time — the values can't be recomputed from the rendered log, so runs
        recorded without value data return ``available: False``.

        Args:
            req: ``{"config": {"top_k"?, "min_swing"?}}``.

        Raises:
            HTTPException: 400 if no run is loaded.
        """
        store = session.store
        if store is None:
            raise HTTPException(400, "no run loaded")
        cfg = dict(req.config or {})
        steps = store.all()
        rows = [
            {
                "node_id": None,
                "idx": step["idx"],
                "decision": step.get("decision", "unknown"),
                "floor": (step.get("meta") or {}).get("floor"),
                "act": (step.get("meta") or {}).get("act"),
                "value": (step.get("wpa") or {}).get("value"),
                "taken_label": (step.get("wpa") or {}).get("taken_label"),
            }
            for step in steps
            if step.get("wpa")
        ]
        # Surface the run's outcome (like the live path does from ``rm``): the
        # logged ``game_over`` state carries ``victory``. Absent on runs that
        # were quit before the end, in which case the review just omits it.
        won: bool | None = next(
            (
                bool((step.get("state") or {}).get("victory"))
                for step in reversed(steps)
                if step.get("decision") == "game_over"
                and (step.get("state") or {}).get("victory") is not None
            ),
            None,
        )
        return wpa_review.review(
            rows,
            top_k=int(cfg.pop("top_k", wpa_review.DEFAULT_TOP_K)),
            min_swing=float(cfg.pop("min_swing", wpa_review.DEFAULT_MIN_SWING)),
            won=won,
            reason_if_empty=(
                "this run log has no recorded value data — play it in the workbench "
                "(rust backend + PPO checkpoint) to record a value curve"
            ),
        )

    @app.post("/api/search_tree")
    def search_tree(
        req: RecommendRequest, session: Session = Depends(get_session)
    ) -> dict[str, Any]:
        """Return the full MCTS search tree for replay step *idx*.

        Args:
            req: ``{"idx": <step index>, "config": {...optional MCTS params}}``

        Returns:
            ``reasoning.search_tree`` result (``{nodes, root, pv, ...}`` or
            ``{error, nodes, ...}``). 409 if the step is not a combat decision.

        Raises:
            HTTPException: 400 if no run loaded; 404 if *idx* out of range.
        """
        store = session.store
        if store is None:
            raise HTTPException(400, "no run loaded")
        try:
            s = store.get(req.idx)
        except IndexError as exc:
            raise HTTPException(404, str(exc)) from exc
        if s.get("state", {}).get("decision") != "combat_play":
            raise HTTPException(409, "search tree only available for combat decisions")
        plan = _faithful_replay_plan(store, req.idx, req.config)
        if plan is not None:
            fixture_path, actions, cfg = plan
            return reasoning.search_tree_fixture_replay(fixture_path, actions, cfg)
        return reasoning.search_tree(s["state"], req.config)

    @app.post("/api/live/start")
    def live_start(req: StartRequest, session: Session = Depends(get_session)) -> dict[str, Any]:
        """Start a fresh live run, replacing any existing session."""
        try:
            state = live_runs.start(
                session,
                character=req.character,
                seed=req.seed,
                ascension=req.ascension,
                act1=req.act1,
                backend=req.backend,
            )
        except RestoreError as exc:
            raise HTTPException(exc.status, str(exc)) from exc
        return _live_step(state, assets, card_loc, hp_ranges)

    @app.post("/api/live/start-faithful")
    def live_start_faithful(
        req: StartFaithfulRequest, session: Session = Depends(get_session)
    ) -> dict[str, Any]:
        """Start a run that reproduces the offline benchmark move-for-move.

        Routes through :meth:`BranchingLiveSession.start_faithful` (rust backend,
        integer seed, no interactive UI flags). Follow with ``/api/live/play-to-end``
        to run it out with the benchmark policy. 400 on a bad seed / config."""
        try:
            state = live_runs.start_run_mode(
                session,
                character=req.character,
                seed=req.seed,
                ascension=req.ascension,
                sims=req.sims,
                boss_sims=req.boss_sims,
                threads=req.threads,
            )
        except RestoreError as exc:
            raise HTTPException(exc.status, str(exc)) from exc
        return _live_step(state, assets, card_loc, hp_ranges)

    def _parse_import(payload: str) -> Any:
        """Parse a pasted state, turning a bad paste into a 422 the UI can show."""
        from workbench.state_import import StateImportError, parse_pasted_state

        try:
            return parse_pasted_state(payload)
        except StateImportError as exc:
            # 422, not 400: the request is well-formed, its content is not.
            # The message is written for the user and is shown verbatim.
            raise HTTPException(422, str(exc)) from exc

    def _import_lane(imported: Any, lane: str) -> str:
        """Which lane to rebuild, and why the other one is not available."""
        if lane not in ("auto", "combat", "run"):
            raise HTTPException(400, f"unknown lane {lane!r} (want auto, combat or run)")
        if lane == "combat" and not imported.has_combat:
            raise HTTPException(
                409,
                "this paste carries no fight — it was copied "
                f"on the {imported.surface} screen. Import it as a run instead.",
            )
        if lane == "auto":
            return "combat" if imported.has_combat else "run"
        return lane

    @app.post("/api/live/import-state/preview")
    def live_import_preview(req: ImportStateRequest) -> dict[str, Any]:
        """Parse a pasted state and describe it WITHOUT touching the live run.

        What the import dialog shows before the user commits: which run this
        is, where it is, and what the import will and will not reproduce. 422
        with a user-facing message when the paste cannot be read.
        """
        imported = _parse_import(req.payload)
        run = imported.run
        return {
            "surface": imported.surface,
            "source": imported.source,
            "has_combat": imported.has_combat,
            "lanes": ["combat", "run"] if imported.has_combat else ["run"],
            "run": {
                "seed": run.seed,
                "character": run.character,
                "ascension": run.ascension,
                "act": run.act,
                "floor": run.floor,
                "hp": run.hp,
                "max_hp": run.max_hp,
                "gold": run.gold,
                "deck_size": len(run.deck),
                "relics": list(run.relics),
                "potions": list(run.potions),
                "coord": run.coord,
            },
            "fidelity": imported.fidelity(),
        }

    @app.post("/api/live/import-state")
    def live_import_state(
        req: ImportStateRequest, session: Session = Depends(get_session)
    ) -> dict[str, Any]:
        """Rebuild a state copied out of the live game, replacing the live run.

        Two lanes, picked by ``lane`` (``"auto"`` prefers the fight):

        * **combat** — the exact board, bit-identical down to the draw order and
          the RNG cursor, playable and branchable. Combat-only: it ends when the
          fight does.
        * **run** — the run position: the real seed's maps, with deck, relics,
          potions, gold, HP and the RNG cursors grafted on, parked on the node
          the player was standing on.

        Everything that can fail is settled BEFORE the current live run is torn
        down — a rejected paste must not cost the user the run they already had
        (the same discipline ``/api/live/from_replay`` keeps).

        422 when the paste cannot be read, 409 when it cannot fill the
        requested lane, 400 when the engine cannot rebuild it.
        """
        imported = _parse_import(req.payload)
        lane = _import_lane(imported, req.lane)

        # Build the factory first; it runs only once the live slot is free.
        def build() -> tuple[BranchingLiveSession, dict[str, Any]]:
            if lane == "combat":
                sess = BranchingLiveSession.from_imported_combat(
                    logs_dir=str(logs_dir),
                    loc=card_loc,
                    wire=imported.combat_wire,
                    character=imported.run.character,
                )
                return sess, sess.start(imported.run.character, imported.run.seed, 0)
            sess = BranchingLiveSession(logs_dir=str(logs_dir), backend="rust", loc=card_loc)
            return sess, sess.start_imported(imported.run)

        prior = session.live
        session.live = None
        session.imported = None
        if prior is not None:
            prior.quit()
        try:
            sess, state = build()
        except (ValueError, RuntimeError, OSError) as exc:
            # A coord the engine's map does not have, a character with no port,
            # a stale wheel. Name it: "could not import" with the engine's own
            # words is actionable, a 500 is not.
            raise HTTPException(400, f"could not rebuild this state: {exc}") from exc
        session.live = sess
        session.imported = imported

        step = _live_step(state, assets, card_loc, hp_ranges)
        fidelity = imported.fidelity()
        fidelity["lane"] = lane
        # Whatever the graft itself could not reproduce (map disagreements, a
        # refused RNG restore, an unupgradable card) rides alongside the
        # parser's own warnings — the user sees one list, not two.
        fidelity["warnings"] = list(fidelity["warnings"]) + sess.import_notes()
        step["import"] = fidelity
        return step

    @app.get("/api/live/simlab/scenario")
    def live_simlab_scenario(session: Session = Depends(get_session)) -> dict[str, Any]:
        """SimLab sweep-form fields for the imported run ("open in SimLab").

        The cheap handoff: sweep this deck against this lineup from turn 1.
        409 when the live run did not come from an import — there is no deck
        list to hand over without one.
        """
        from workbench.simlab_bridge import scenario_prefill

        if session.imported is None:
            raise HTTPException(409, "no imported state — paste one first")
        return scenario_prefill(session.imported)

    @app.get("/api/live/position")
    def live_position(session: Session = Depends(get_session)) -> dict[str, Any]:
        """The active board as a SimLab sandbox position ("SimLab from here").

        The workbench opens SimLab with this in the URL hash, so the sandbox
        starts on the exact hand, pile order, powers and telegraphs the user
        is looking at. 400 with no live run; 409 when the active node is not
        a fight (there is no position to edit off a map screen).
        """
        from simlab.position import PositionError, position_from_wire

        sess = session.live
        if sess is None:
            raise HTTPException(400, "no live run")
        try:
            states, skipped = sess.combat_boards()
        except RuntimeError as exc:
            raise HTTPException(400, str(exc)) from exc
        if not states:
            raise HTTPException(
                409,
                "the current position is not a fight"
                + (": " + "; ".join(skipped) if skipped else ""),
            )
        label, wire = states[0]
        run = getattr(session.imported, "run", None)
        try:
            position, notes = position_from_wire(wire)
        except PositionError as exc:
            raise HTTPException(409, str(exc)) from exc
        if run is not None:
            position["ascension"] = int(run.ascension)
        return {"position": position, "label": label, "notes": notes}

    @app.post("/api/live/simlab/sweep")
    def live_simlab_sweep(
        req: SimlabSweepRequest, session: Session = Depends(get_session)
    ) -> dict[str, Any]:
        """Launch a SimLab sweep starting from live board position(s).

        Each named branch node contributes one cell, so several lines can be
        compared as distributions rather than as single playouts. Nodes that
        are not combat positions are skipped and named in ``skipped`` — a
        silently shorter sweep would read as a finished comparison.

        The active node is restored before returning, so launching a sweep
        never moves the user.

        400 with no live run; 409 when no named node holds a fight; 503 when
        SimLab is not installed in this deployment.
        """
        from workbench.simlab_bridge import DEFAULT_STRATEGIES, sweep_from_states

        sess = session.live
        if sess is None:
            raise HTTPException(400, "no live run")
        submit = getattr(app.state, "simlab_submit", None)
        if submit is None:
            raise HTTPException(503, "SimLab is not available in this deployment")

        try:
            states, skipped = sess.combat_boards(req.node_ids)
        except RuntimeError as exc:
            raise HTTPException(400, str(exc)) from exc

        if not states:
            raise HTTPException(
                409,
                "none of those positions is a fight — SimLab sweeps combats. "
                + ("; ".join(skipped) if skipped else ""),
            )
        spec = sweep_from_states(
            req.name,
            states,
            strategies=tuple(req.strategies or DEFAULT_STRATEGIES),
            n_seeds=req.n_seeds,
        )
        try:
            sweep_id = submit(spec)
        except (TypeError, ValueError, KeyError) as exc:
            raise HTTPException(422, f"bad sweep spec: {exc}") from exc
        return {
            "sweep_id": sweep_id,
            "url": f"/simlab.html?sweep={sweep_id}",
            "cells": [label for label, _ in states],
            "skipped": skipped,
        }

    @app.post("/api/live/play-to-end")
    def live_play_to_end(session: Session = Depends(get_session)) -> dict[str, Any]:
        """Play a faithful run (see ``/api/live/start-faithful``) to completion
        with the benchmark policy, recording every step as a branch node.

        Returns ``{"step", "steps", "victory", "over", "error", "timeline"}``. 400
        if no faithful run is active. The whole run plays server-side in one call,
        so the branch tree is populated when it returns."""
        sess = session.live
        if sess is None:
            raise HTTPException(400, "no live run")
        try:
            out = sess.play_to_end()
        except RuntimeError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {
            "step": _live_step(out["state"], assets, card_loc, hp_ranges),
            "steps": out["steps"],
            "victory": out["victory"],
            "over": out["over"],
            "error": out["error"],
            "timeline": sess.timeline(),
        }

    @app.post("/api/live/faithful-step")
    def live_faithful_step(session: Session = Depends(get_session)) -> dict[str, Any]:
        """Apply ONE benchmark-policy decision to the active faithful run.

        The live UI loops this to play a faithful run out visibly (and to
        pause/resume), instead of the blocking ``/api/live/play-to-end``. Returns
        ``{"step", "done", "over", "victory", "error"}``; ``done`` is True when the
        run is finished (decide() returned None) or an error stopped it. 400 if no
        faithful run is active.
        """
        sess = session.live
        if sess is None or getattr(sess, "_faithful_agent", None) is None:
            raise HTTPException(400, "no faithful run")
        out = sess.faithful_step()
        # ``done`` signals "no further action to apply" (agent returned None)
        # or an error stopped the run.  ``out["done"]`` carries this flag from
        # faithful_step(); error also terminates the loop.
        done = out.get("done", False) or (out.get("error") is not None)
        return {
            "step": _live_step(out["state"], assets, card_loc, hp_ranges),
            "done": done,
            "over": out["over"],
            "victory": out["victory"],
            "steps": out.get("steps", 0),
            "error": out["error"],
            "debug": out.get("debug"),
            "search": out.get("search"),
        }

    @app.post("/api/live/run-mode-step")
    def live_run_mode_step(session: Session = Depends(get_session)) -> dict[str, Any]:
        """Apply ONE run-mode decision (the unified stepper the live UI loops).
        Same response shape as ``/api/live/faithful-step``. 400 if no run-mode
        run is active.

        ``search`` carries the combat MCTS root distribution for the move this
        step applied — ``{"scores": [...], "action": label}`` in the same row
        shape ``/api/live/hint`` returns, or ``None`` off combat. It is
        deliberately NOT folded into ``step.scores``: those rows are matched
        onto the rendered hand by ``stable_idx``, and ``search``'s rows belong
        to the position BEFORE this step's action, whose hand is already gone.
        """
        sess = session.live
        if sess is None:
            raise HTTPException(400, "no live run")
        try:
            out = sess.run_mode_step()
        except RuntimeError as exc:
            raise HTTPException(400, str(exc)) from exc
        done = out.get("done", False) or (out.get("error") is not None)
        return {
            "step": _live_step(out["state"], assets, card_loc, hp_ranges),
            "done": done,
            "over": out["over"],
            "victory": out["victory"],
            "steps": out.get("steps", 0),
            "error": out["error"],
            "debug": out.get("debug"),
            "search": out.get("search"),
        }

    @app.post("/api/live/autobattle/advance")
    def live_autobattle_advance(session: Session = Depends(get_session)) -> dict[str, Any]:
        """Auto-play combats until the next run-level decision (autobattle mode).

        Returns the stop's rendered step plus the change-diff since the previous
        stop. The champion's pick is deliberately NOT in this payload — fetch it
        from ``/api/live/autobattle/verdict`` after committing the action. 400 if no
        run is live.
        """
        sess = session.live
        if sess is None:
            raise HTTPException(400, "no live run")
        try:
            out = sess.autobattle_advance()
        except RuntimeError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {
            "step": _live_step(out["state"], assets, card_loc, hp_ranges),
            "diff": out["diff"],
            "autoplayed": out["autoplayed"],
            "stopped": out["stopped"],
            "over": out["over"],
            "won": out["won"],
            "error": out["error"],
        }

    @app.post("/api/live/autobattle/step")
    def live_autobattle_step(session: Session = Depends(get_session)) -> dict[str, Any]:
        """Advance the autobattle by ONE decision so the fight can be watched.

        The UI loops this to play the combats out visibly, exactly as it loops
        ``/api/live/run-mode-step``, instead of the blocking
        ``/api/live/autobattle/advance`` which resolves the whole leg server-side
        and shows only its far end. Same response as the advance endpoint plus
        ``played`` (False when the run was already sitting on a stop) and the
        step's ``search`` / ``debug``; ``diff`` is null until a stop is reached.
        400 if no run is live.
        """
        sess = session.live
        if sess is None:
            raise HTTPException(400, "no live run")
        try:
            out = sess.autobattle_step()
        except RuntimeError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {
            "step": _live_step(out["state"], assets, card_loc, hp_ranges),
            "diff": out["diff"],
            "played": out["played"],
            "stopped": out["stopped"],
            "over": out["over"],
            "won": out["won"],
            "error": out["error"],
            "search": out["search"],
            "debug": out["debug"],
        }

    @app.get("/api/live/autobattle/verdict")
    def live_autobattle_verdict(session: Session = Depends(get_session)) -> dict[str, Any]:
        """Reveal the champion's pick for the decision just committed.

        Agreement only — never a win probability; see
        :meth:`~workbench.branching.BranchingLiveSession.autobattle_verdict`. 400 if
        no run is live.
        """
        sess = session.live
        if sess is None:
            raise HTTPException(400, "no live run")
        return sess.autobattle_verdict()

    @app.post("/api/live/from_replay")
    def live_from_replay(
        req: RecommendRequest, session: Session = Depends(get_session)
    ) -> dict[str, Any]:
        """Take over the loaded recorded run at step ``idx`` ("play from here"),
        replacing any current live run.

        Two routes, tried in the order :func:`~workbench.live_runs.resume_capability`
        reports (see :meth:`~workbench.live_runs.LiveRuns.resume_recording`):
        a fixture-sourced combat recording reloads its fixture and replays the
        logged native actions (combat steps only); a workbench run log rebuilds
        the whole run from its seed, so **any** step can be taken over.

        400 if no run is loaded; 404 if ``idx`` is out of range; 409 with the
        reason when neither route fits this recording or the line diverges.
        """
        store = session.store
        if store is None:
            raise HTTPException(400, "no run loaded")
        try:
            state = live_runs.resume_recording(session, store, req.idx)
        except RestoreError as exc:
            raise HTTPException(exc.status, str(exc)) from exc
        return _live_step(state, assets, card_loc, hp_ranges)

    @app.get("/api/live/state")
    def live_state(session: Session = Depends(get_session)) -> dict[str, Any]:
        """Return the current live step. 400 if no run is active."""
        sess = session.live
        if sess is None:
            raise HTTPException(400, "no live run")
        # ``hp_ranges`` too: without it a run rehydrated after a page reload
        # rendered its repeated monsters without the per-instance size jitter
        # every other route stamps (the browser workbench's golden fixtures
        # caught the asymmetry).
        return _live_step(sess.state(), assets, card_loc, hp_ranges)

    @app.get("/api/live/map")
    def live_map(session: Session = Depends(get_session)) -> dict[str, Any]:
        """Current act's full map graph for the live run, on demand.

        Returns ``{"rows": [], "graphs": [...]}`` in the same shape as
        ``/api/map`` so the UI's MapView renders it directly. ``graphs`` is
        empty when no map is available (e.g. the boss room). 400 if no run is
        active.
        """
        sess = session.live
        if sess is None:
            raise HTTPException(400, "no live run")
        state = sess.state()
        full_map = sess.map()
        graphs: list[dict[str, Any]] = []
        if full_map:
            graphs.append(
                {
                    "idx": 0,
                    "act": state.get("act"),
                    "act_name": state.get("act_name"),
                    "floor": state.get("floor"),
                    "full_map": _enrich_map_nodes(full_map, assets),
                }
            )
        return {"rows": [], "graphs": graphs}

    @app.post("/api/live/action")
    def live_action(req: ActionRequest, session: Session = Depends(get_session)) -> dict[str, Any]:
        """Apply an action; return ``{"step", "error"}``. 400 if illegal verb."""
        sess = session.live
        if sess is None:
            raise HTTPException(400, "no live run")
        try:
            result = sess.apply(req.action, req.args)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {
            "step": _live_step(result["state"], assets, card_loc, hp_ranges),
            "error": result["error"],
            "label": result["label"],
            "details": result["details"],
        }

    @app.post("/api/live/auto-action")
    def live_auto_action(session: Session = Depends(get_session)) -> dict[str, Any]:
        """The policy's recommended ``{"action", "args"}`` for the current
        decision — a run-level one (card reward, shop, event, map, rest, …) or
        an open in-combat card choice (Discover, a pile search, an upgrade pick).

        Powers the "play best (G)" keystroke and the "play to end (F)" full-run
        autoplay; the frontend dispatches the returned verb through
        ``/api/live/action``. Run-level decisions are answered by the session's
        off-combat brain; an in-combat choice by the combat MCTS at the same
        config as that screen's ``search %`` chips, so G plays the option the
        panel stars. 409 in plain combat (card play — the MCTS hint owns that),
        at game over, on a multi-pick in-combat choice, and on a backend that
        can't recommend. 400 if no run.
        """
        sess = session.live
        if sess is None:
            raise HTTPException(400, "no live run")
        rec = sess.recommended_action()
        if rec is None:
            raise HTTPException(409, "no recommendation here")
        return rec

    @app.post("/api/live/run-mode")
    def live_run_mode(
        req: RunModeRequest, session: Session = Depends(get_session)
    ) -> dict[str, Any]:
        """Update the two-axis run mode (off-combat brain x combat budget).
        Any subset of fields updates; 400 on no run or a bad value."""
        sess = session.live
        if sess is None:
            raise HTTPException(400, "no live run")
        try:
            sess.set_run_mode(
                off_combat_brain=req.off_combat_brain,
                ppo_impl=req.ppo_impl,
                combat_sims=req.combat_sims,
                combat_boss_sims=req.combat_boss_sims,
                combat_threads=req.combat_threads,
                reward_sims=req.reward_sims,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {
            "off_combat_brain": sess.off_combat_brain,
            "ppo_impl": sess.ppo_impl,
            "combat_sims": sess.combat_sims,
            "combat_boss_sims": sess.combat_boss_sims,
            "combat_threads": sess.combat_threads,
            "reward_sims": sess.reward_sims,
        }

    @app.post("/api/live/ppo-policy")
    def live_ppo_policy(session: Session = Depends(get_session)) -> dict[str, Any]:
        """The PPO net's masked softmax + V(s) over the current NON-combat
        decision (the probability panel). Carries ``ppo_available: False`` +
        ``error`` when the checkpoint can't load. 409 in combat / at game over /
        on a backend without an in-process RunManager. 400 if no run."""
        sess = session.live
        if sess is None:
            raise HTTPException(400, "no live run")
        result = sess.ppo_policy_panel()
        if result is None:
            raise HTTPException(409, "ppo policy only available at a non-combat decision")
        return result

    @app.get("/api/live/timeline")
    def live_timeline(session: Session = Depends(get_session)) -> dict[str, Any]:
        """Return the branch tree + active path for the live log / branch UI."""
        sess = session.live
        if sess is None:
            raise HTTPException(400, "no live run")
        return sess.timeline()

    @app.post("/api/live/seek")
    def live_seek(req: SeekRequest, session: Session = Depends(get_session)) -> dict[str, Any]:
        """Rewind (replay) to a past node and return its enriched step.

        Read-only: the active leaf moves but no node is created. Acting next
        forks a new branch from here.
        """
        sess = session.live
        if sess is None:
            raise HTTPException(400, "no live run")
        try:
            state = sess.seek(req.node_id)
        except KeyError as exc:
            raise HTTPException(404, f"unknown node: {req.node_id}") from exc
        return {"step": _live_step(state, assets, card_loc, hp_ranges)}

    @app.post("/api/live/hint")
    def live_hint(req: HintRequest, session: Session = Depends(get_session)) -> dict[str, Any]:
        """MCTS hint for the current combat state. 409 outside combat."""
        sess = session.live
        if sess is None:
            raise HTTPException(400, "no live run")
        result = sess.hint(req.config)
        if result is None:
            raise HTTPException(409, "hint only available during combat")
        return result

    @app.post("/api/live/last-move-regret")
    def live_last_move_regret(
        req: HintRequest, session: Session = Depends(get_session)
    ) -> dict[str, Any]:
        """Regret of the move that produced the current live state — the
        "blunder right after you did one" warning.

        Re-scores the previous combat position and diffs the move just made
        against the MCTS line. 409 when there is no prior combat move to assess
        (start of run, or the last decision wasn't a combat play). Body is
        ``{"config": {...optional: blunder_pct + MCTS params}}``.
        """
        sess = session.live
        if sess is None:
            raise HTTPException(400, "no live run")
        result = sess.last_move_regret(req.config)
        if result is None:
            raise HTTPException(409, "no prior combat move to assess")
        return result

    @app.post("/api/live/regret-review")
    def live_regret_review(
        req: HintRequest, session: Session = Depends(get_session)
    ) -> dict[str, Any]:
        """Full-run regret review over the live run's active line.

        The live twin of ``/api/regret-review`` (which reviews a loaded
        *replay*): diffs every combat decision made so far against the MCTS line
        and surfaces the costliest plays, re-scoring each combat position on
        demand. Body is ``{"config": {...optional: blunder_pct, top_n + MCTS
        params}}``. 400 when there is no live run.
        """
        sess = session.live
        if sess is None:
            raise HTTPException(400, "no live run")
        return sess.review_run(req.config)

    @app.post("/api/live/wpa-review")
    def live_wpa_review(
        req: HintRequest, session: Session = Depends(get_session)
    ) -> dict[str, Any]:
        """WPA run review over the live run's active line (issue #201).

        The run-level twin of ``/api/live/regret-review``: the value curve
        from the per-decision PPO values recorded at each node, plus ranked
        turning points. ``available: False`` (never an error) when the run has
        no value data — C# backend or no loadable PPO checkpoint. Body is
        ``{"config": {"top_k"?, "min_swing"?}}``. 400 when no live run.
        """
        sess = session.live
        if sess is None:
            raise HTTPException(400, "no live run")
        return sess.wpa_review(req.config)

    @app.post("/api/live/wpa-verify")
    def live_wpa_verify(
        req: WpaVerifyRequest, session: Session = Depends(get_session)
    ) -> dict[str, Any]:
        """Counterfactual spot check of one WPA turning point (SLOW —
        agent-finishes the run twice from the flagged decision).

        Compares the real outcome of continuing the played line vs the PPO's
        preferred alternative, both agent-played under the session's run mode.
        404 for an unknown node; 409 when the node can't be verified (a leaf
        with no recorded follow-up move, or no alternative option); 400 when no
        live run or the backend can't autoplay.
        """
        sess = session.live
        if sess is None:
            raise HTTPException(400, "no live run")
        try:
            return sess.wpa_verify(
                req.node_id, max_steps=req.max_steps, max_seconds=req.max_seconds
            )
        except KeyError as exc:
            raise HTTPException(404, f"unknown node: {req.node_id}") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/live/card-advisor")
    def live_card_advisor(
        req: HintRequest, session: Session = Depends(get_session)
    ) -> dict[str, Any]:
        """Monte-Carlo win-rate per card-reward option for the live run.

        A deliberately deep, slow analysis (forks the run and plays the rest of
        the act out many times per option, with MCTS in every fight), so it is
        only ever run on demand from the card-reward screen — never as part of
        a normal step. 409 outside a card-reward decision. Body is
        ``{"config": {...optional: rollouts, sims, threads, max_steps, seed}}``,
        same envelope as the hint endpoint.
        """
        sess = session.live
        if sess is None:
            raise HTTPException(400, "no live run")
        result = sess.advise_card_reward(req.config)
        if result is None:
            raise HTTPException(409, "card advisor only available at a card reward")
        return result

    @app.post("/api/live/card-advisor/stream")
    def live_card_advisor_stream(
        req: HintRequest, session: Session = Depends(get_session)
    ) -> StreamingResponse:
        """Streaming twin of ``/api/live/card-advisor`` for a live progress bar.

        Same analysis, emitted as newline-delimited JSON as it goes (see
        :mod:`workbench.advice_stream` for the event protocol).
        """
        sess = session.live
        if sess is None:
            raise HTTPException(400, "no live run")
        return StreamingResponse(
            stream_advice(
                lambda progress: sess.advise_card_reward(req.config, progress=progress),
                unavailable="not at a card reward",
            ),
            media_type="application/x-ndjson",
        )

    @app.post("/api/live/choice-ranker")
    def live_choice_ranker(
        req: HintRequest, session: Session = Depends(get_session)
    ) -> dict[str, Any]:
        """Fast per-option combat-MCTS ranking for the live run's in-combat choice.

        The cheap counterpart to ``/api/live/decision-advisor`` for an in-combat
        single-pick card-select (Headbutt's "put a card on top of your draw pile",
        an upgrade-in-hand pick, …): one combat MCTS search ranks every option,
        instead of the advisor's fork-and-play-out-the-act-per-option rollouts —
        so the workbench can fetch it eagerly when an in-combat choice opens while
        the slow EV rollouts stay behind the "analyze options" button. 409 when
        the active decision is not an in-combat card-select. Body is
        ``{"config": {...}}``, same envelope as the hint endpoint.
        """
        sess = session.live
        if sess is None:
            raise HTTPException(400, "no live run")
        result = sess.rank_choice(req.config)
        if result is None:
            raise HTTPException(409, "choice ranker only available at an in-combat card-select")
        return result

    @app.post("/api/live/decision-advisor")
    def live_decision_advisor(
        req: HintRequest, session: Session = Depends(get_session)
    ) -> dict[str, Any]:
        """Monte-Carlo win-rate per option for the live run's current decision.

        The generic sibling of ``/api/live/card-advisor``: covers shops, events
        (incl. treasure / boss relic), and single-pick in-combat card selects.
        Forks the run and plays the rest of the act out per option. 409 when the
        active decision is none of those. Body is ``{"config": {...}}``, same
        envelope as the hint endpoint.
        """
        sess = session.live
        if sess is None:
            raise HTTPException(400, "no live run")
        result = sess.advise_decision(req.config)
        if result is None:
            raise HTTPException(409, "decision advisor only available at a shop, event, or choice")
        return result

    @app.post("/api/live/decision-advisor/stream")
    def live_decision_advisor_stream(
        req: HintRequest, session: Session = Depends(get_session)
    ) -> StreamingResponse:
        """Streaming twin of ``/api/live/decision-advisor`` for a live progress bar.

        Same analysis, emitted as newline-delimited JSON as it goes (see
        :mod:`workbench.advice_stream` for the event protocol).
        """
        sess = session.live
        if sess is None:
            raise HTTPException(400, "no live run")
        return StreamingResponse(
            stream_advice(
                lambda progress: sess.advise_decision(req.config, progress=progress),
                unavailable="not at an advisable decision",
            ),
            media_type="application/x-ndjson",
        )

    @app.post("/api/live/search_tree")
    def live_search_tree(
        req: HintRequest, session: Session = Depends(get_session)
    ) -> dict[str, Any]:
        """Full MCTS search tree for the current live combat state.

        409 outside combat. Body is ``{"config": {...optional MCTS params}}``,
        same as the hint endpoint.
        """
        sess = session.live
        if sess is None:
            raise HTTPException(400, "no live run")
        result = sess.search_tree(req.config)
        if result is None:
            raise HTTPException(409, "search tree only available during combat")
        return result

    @app.post("/api/live/quit")
    def live_quit(session: Session = Depends(get_session)) -> dict[str, Any]:
        """End the live run; return the recorded log path."""
        sess = session.live
        if sess is None:
            raise HTTPException(400, "no live run")
        out = sess.quit()
        session.live = None
        return out

    @app.post("/api/live/save")
    def live_save(req: SaveRequest, session: Session = Depends(get_session)) -> dict[str, Any]:
        """Save the current live run's active line under *name*.

        Overwrites any existing save with the same (slugified) name. 400 if no
        run is active or the name is blank.
        """
        sess = session.live
        if sess is None:
            raise HTTPException(400, "no live run")
        if not req.name.strip():
            raise HTTPException(400, "save name must not be blank")
        line, progress = sess.active_line()
        path = saves_mod.write_save(
            saves_mod.save_dir(logs_dir),
            name=req.name,
            character=sess.character,
            seed=sess.seed,
            ascension=sess.ascension,
            backend=sess.backend,
            run_mode=sess.run_mode,
            progress=progress,
            line=line,
            act1=sess.act1,
            node_values=sess.active_node_values(),
        )
        return {"name": req.name, "path": str(path), "step_count": len(line)}

    @app.get("/api/saves")
    def list_saves() -> dict[str, Any]:
        """List saved runs (metadata headers only), newest first."""
        return {"saves": saves_mod.list_saves(saves_mod.save_dir(logs_dir))}

    @app.post("/api/saves/load")
    def load_saved(req: LoadSaveRequest, session: Session = Depends(get_session)) -> dict[str, Any]:
        """Restore a named save into a fresh live session, replacing any current one.

        404 if the save is unknown/invalid; 409 if replay of the saved line
        diverges from a valid run.
        """
        payload = saves_mod.read_save(saves_mod.save_dir(logs_dir), req.name)
        if payload is None:
            raise HTTPException(404, f"unknown save: {req.name}")
        try:
            state = live_runs.restore_save(session, payload)
        except RestoreError as exc:
            raise HTTPException(exc.status, str(exc)) from exc
        return _live_step(state, assets, card_loc, hp_ranges)

    @app.post("/api/bugreport")
    def capture_bugreport(
        req: BugReportRequest, session: Session = Depends(get_session)
    ) -> dict[str, str]:
        """Freeze the current workbench moment to a dump and return its id/URL."""
        # A server restart empties the session while the SPA still shows a
        # loaded run; recover the store from the request's run_path instead
        # of 409ing "nothing to capture".
        if req.mode == "replay" and session.store is None and req.run_path:
            try:
                session.store = TimelineStore.load_jsonl(req.run_path)
            except (FileNotFoundError, OSError):
                pass  # fall through to the structured 409
        try:
            dump = bugreport.build_dump(
                session,
                mode=req.mode,
                step=req.step,
                screen=req.screen,
                overlay=req.overlay,
                note=req.note,
                run_path=req.run_path,
            )
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        # Live-game context rides along when the advisor is up: which run the
        # bridge is following plus its branch tree at capture time (follow-
        # mode reports otherwise lose the rewind-lane state the bug was seen
        # in). Best-effort -- a report must never fail because the advisor
        # is down.
        try:
            dump["bridge"] = {
                "status": _bridge_call("/status", timeout=2.0),
                "tree": _bridge_call("/tree", timeout=3.0),
            }
        except HTTPException:
            pass
        report_id = bugreport_store.new_id()
        dump["meta"]["id"] = report_id
        dump["meta"]["captured_at"] = (
            datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        )
        url = bugreport_store.put(
            report_id,
            dump,
            logs_dir=logs_dir,
            bucket=os.environ.get("STS2_BUGREPORT_BUCKET"),
        )
        return {"id": report_id, "url": url}

    @app.get("/api/bugreport/{report_id}")
    def fetch_bugreport(report_id: str) -> dict[str, Any]:
        """Return a stored bug-report dump for the read-only inspection view."""
        try:
            dump = bugreport_store.get(
                report_id,
                logs_dir=logs_dir,
                bucket=os.environ.get("STS2_BUGREPORT_BUCKET"),
            )
        except ValueError as exc:
            raise HTTPException(404, f"unknown report: {report_id}") from exc
        if dump is None:
            raise HTTPException(404, f"unknown report: {report_id}")
        return dump

    # Serve art locally only when it isn't offloaded to an external origin
    # (STS2_ASSETS_BASE_URL). In manifest/bucket mode `_asset_url` already points
    # clients at the bucket, so the local mount is unnecessary (and the tree is
    # usually absent).
    if _ASSETS_BASE_URL == "/assets" and Path(images_root).is_dir():
        # sfx_map.json's events resolve to `audio/<event>.webm` (see
        # web/src/sound/sfxMap.ts's `eventAssetUrl`), and in the deployed
        # bucket that resolves fine: the bucket root IS `images_root`
        # (deploy/sync-assets.sh puts the audio tree at `$BUCKET/audio`), so
        # `$BUCKET/audio/...` exists. Locally, `images_root` is
        # `sts2-wiki/site/public/images` — the audio tree at
        # `sts2-wiki/site/public/audio` is a SIBLING of that directory, not a
        # child, so without this mount every `/assets/audio/...` request
        # 404s and sound is silent in every non-bucket run (`just workbench`)
        # with nothing louder than a per-cue `console.warn`. Registered
        # BEFORE the `/assets` mount below: Starlette's Router tries mounts
        # in registration order and a `Mount` that matches a path prefix
        # claims the whole request (it does not fall through to a later
        # route on the inner app's 404), so the general `/assets` mount
        # would otherwise shadow this one and this mount would never be
        # reached. Absent audio tree (a checkout that hasn't run
        # `just build-sfx-audio` yet) degrades to the same silence as
        # before this fix, not an error.
        audio_root = Path(images_root).parent / "audio"
        if audio_root.is_dir():
            app.mount("/assets/audio", StaticFiles(directory=str(audio_root)), name="assets-audio")
        app.mount("/assets", StaticFiles(directory=str(images_root)), name="assets")

    # Embed the SimLab API (sweep jobs + chart aggregates) on both the dev
    # (serve_spa=False, pages via Vite) and single-container (serve_spa=True,
    # pages via the SPA mount below) paths. SimLab's `/api/meta` + `/api/sweeps*`
    # routes are disjoint from the workbench's, so they're registered as explicit
    # routes here — crucially BEFORE the serve_spa `/` StaticFiles catch-all, so
    # a browser POST resolves to the SimLab handler instead of falling through to
    # StaticFiles (which allows only GET/HEAD → 405). A nested `Mount("/")` can't
    # do this: the SPA mount at `/` would shadow it.
    try:
        from simlab.specs import SweepSpec as _SweepSpec
        from simlab.web.app import DEFAULT_RESULTS as _SIMLAB_RESULTS
        from simlab.web.app import build_api_router as _build_simlab_router
        from simlab.web.jobs import JobManager as _SimlabJobManager

        # CWD-relative by default (`results/simlab`), which is fine for a dev
        # server but not for a test or a container that runs from elsewhere —
        # and two app instances sharing one root see each other's sweeps.
        results_root = Path(os.environ.get("STS2_SIMLAB_RESULTS") or _SIMLAB_RESULTS)
        simlab_jobs = _SimlabJobManager(results_root=results_root, jobs=0)
        app.include_router(_build_simlab_router(simlab_jobs))

        # In-process submit for the workbench's own handoff
        # (``/api/live/simlab/sweep``). The same JobManager the HTTP route
        # uses, so a sweep launched from a board shows up in SimLab's list
        # exactly like one launched from its form.
        def _simlab_submit(spec: dict[str, Any]) -> str:
            sweep = _SweepSpec.from_json(spec)
            sweep.cells()  # surface a bad spec here, not in the worker thread
            return simlab_jobs.submit(sweep)

        app.state.simlab_submit = _simlab_submit
    except ImportError:
        # SimLab deps (numpy/scipy) absent — boot the workbench without it.
        app.state.simlab_submit = None

    if serve_spa:
        # Single-container (Cloud Run) path: serve the built SPA at `/`. This is
        # a catch-all mount, so it must come last (after the SimLab API routes
        # above, and after the workbench's own explicit routes).
        web_dist = repo_root / "web" / "dist"
        if web_dist.is_dir():
            app.mount("/", StaticFiles(directory=str(web_dist), html=True), name="spa")

            # The shell (index.html) must always revalidate: it names the
            # content-hashed bundles, and browsers heuristic-cache HTML that
            # ships no Cache-Control -- Firefox kept serving a stale build
            # across several same-evening redeploys (live 2026-08-21). The
            # hashed wb-assets/* keep default caching.
            @app.middleware("http")
            async def _no_cache_html(request, call_next):  # type: ignore[no-untyped-def]
                response = await call_next(request)
                if response.headers.get("content-type", "").startswith("text/html"):
                    response.headers["cache-control"] = "no-cache"
                return response

    # Issue every client its ``sts2_sid`` cookie on the first response (incl. the
    # SPA's index.html), so a fresh browser's id is established before its API
    # calls fire — no concurrent-first-request race minting orphan sessions.
    install_session_cookie(app)

    # Optional shared-password gate. Active only when STS2_AUTH_PASSWORD is set,
    # so local dev and the pytest suite are unaffected. Added last so it wraps
    # every route + mount (it's the outermost middleware) — and so an
    # unauthenticated request is turned away before a session cookie is minted.
    auth_password = os.environ.get("STS2_AUTH_PASSWORD")
    if auth_password:
        from workbench.auth import install_password_gate

        install_password_gate(app, auth_password, os.environ.get("STS2_AUTH_SECRET"))

    return app


def create_app_from_env() -> FastAPI:
    """Build the app from environment variables (a uvicorn factory).

    ``uvicorn --reload`` re-imports the app in a worker subprocess, so it
    must be referenced by import string rather than passed as a built
    object. This zero-arg factory reads the same ``STS2_WIKI_IMAGES`` /
    ``STS2_LOGS_DIR`` env vars documented in :mod:`workbench.__main__`.

    Cloud Run / lean-deploy env vars (see ``Dockerfile`` + ``DEPLOY.md``):
        STS2_SERVE_SPA: when truthy, serve the built ``web/dist`` SPA at ``/``.
        STS2_ASSETS_MANIFEST: path to ``asset_manifest.json`` (bucket art mode).
        STS2_ASSETS_BASE_URL: external origin for art (read at import).
        STS2_SKIP_WHEEL_CHECK: when truthy, skip the sts2_rs staleness gate
            (the parity harness isn't bundled in the deploy image).
    """
    # Fail LOUDLY at startup if the installed sts2_rs is stale — otherwise a
    # hot-reloaded backend serving an outdated .so surfaces as cryptic errors
    # (e.g. `mcts_recommend() got an unexpected keyword argument`). This runs in
    # every uvicorn (re)started worker. Bypass with STS2_SKIP_WHEEL_CHECK=1.
    if os.environ.get("STS2_SKIP_WHEEL_CHECK", "").lower() not in ("1", "true", "yes"):
        try:
            from parity_audit import assert_wheel_fresh

            assert_wheel_fresh()
        except ImportError:
            pass  # harness module unavailable — skip rather than block the server

    repo_root = Path(__file__).resolve().parents[2]
    default_images = repo_root.parent / "sts2-wiki" / "site" / "public" / "images"
    images = os.environ.get("STS2_WIKI_IMAGES", str(default_images))
    logs = os.environ.get("STS2_LOGS_DIR", str(repo_root / "logs"))
    serve_spa = os.environ.get("STS2_SERVE_SPA", "").lower() in ("1", "true", "yes")
    manifest = os.environ.get("STS2_ASSETS_MANIFEST") or None
    return create_app(
        images_root=images,
        logs_dir=logs,
        serve_spa=serve_spa,
        assets_manifest=manifest,
    )
