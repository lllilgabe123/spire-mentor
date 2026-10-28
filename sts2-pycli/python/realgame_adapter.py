"""Bridge ``combat_action`` (v2) -> Sts2Headless ``combat_play`` decision JSON.

Real-game bridge phase 2: the live game (via the SpireMentor mod) speaks the
wire protocol documented in ``spire-mentor-mod/PROTOCOL.md``; the MCTS
recommender (:mod:`sts2_hint.recommender`) only understands the
Sts2Headless ``combat_play`` decision shape consumed by
:func:`sts2_hint.adapter.translate` (which in turn feeds
``sts2_rs.state_from_dict``). This module bridges the two *without*
touching either -- :func:`to_headless_combat_play` produces a dict in
exactly the shape :mod:`sts2_hint.adapter` already knows how to read
(the same shape ``tests/fixtures/hint_adapter/_helpers.py`` hand-crafts
and Sts2Headless's own ``RunSimulator.cs`` emits for a ``combat_play``
decision), so the existing sts2_hint pipeline is reused completely
unmodified.

Field-mapping notes (verified against ``PROTOCOL.md``, the
``sts2_hint.adapter``/``schema`` source, and the ``sts2-rs`` engine's own
draw-pile semantics -- see the real-game-bridge plan's phase-2 task list):

* **Card ids.** The bridge already sends game-native registry ids in bare
  SCREAMING_SNAKE form (e.g. ``"STRIKE_IRONCLAD"``, ``"BASH"``) with no
  ``/`` or ``CARD.`` prefix -- confirmed these match
  ``sts2_rs.list_registry_names("card")`` directly, so
  :mod:`sts2_hint.adapter`'s bare-id resolution path (``_resolve_card_id``)
  accepts them verbatim. No re-mapping needed.
* **Relic / monster ids.** The bridge sends canonical relic ids
  (``"BURNING_BLOOD"``) that also match the ``sts2_rs`` registry directly,
  but :mod:`sts2_hint.adapter`'s relic translator only accepts a display
  **title** (looked up via ``RELIC_TITLE_TO_ID``) -- so relics are emitted
  keyed by the bridge's ``name`` field, not ``id``. Monsters are the same
  story: the bridge's ``id`` (``"SHRINKER_BEETLE"``) does NOT match the
  registry's CamelCase monster ids (``"ShrinkerBeetle"``), but the bridge's
  ``name`` (``"Shrinker Beetle"``) matches ``MONSTER_TITLE_TO_ID`` exactly,
  so enemies are also emitted keyed by ``name``.
* **Powers.** The bridge's power ``id`` is the C# ``Id.Entry``, which
  keeps a ``_POWER`` suffix the Rust registry strips for most powers
  (``VULNERABLE_POWER`` vs the engine's ``VULNERABLE``) -- exactly the
  case ``sts2_hint.adapter._translate_power`` already falls back from id
  to a title lookup for. Both ``id`` and ``name`` are forwarded verbatim
  and the existing fallback handles the rest.
* **Draw-pile order.** The bridge documents "index 0 is the next card
  drawn" (verified against the v0.107.1 decomp); the ``sts2-rs`` engine
  itself pops from the front of ``state.player.draw`` (``draw.remove(0)``
  in ``combat_loop.rs``) and ``state_dict.rs`` preserves wire order
  1:1 into that vec -- so the bridge's draw/discard/exhaust pile order is
  passed straight through with no reversal.
* **Target/enemy indexing.** ``sts2_hint.display.format_target`` resolves
  an MCTS action's ``target_index`` by matching it against each enemy's
  wire ``index`` field -- i.e. ``target_index`` is the enemy's *position*
  in the ``enemies`` array, exactly the same 0-based addressing the
  bridge's own ``play``/``potion`` ``target_index`` uses. So an MCTS
  recommendation's ``target_index`` can be sent back to the bridge
  unchanged.
* **Hand ``stable_idx``.** Sts2Headless's ``card.index`` on a hand card
  becomes ``CardSpec.stable_idx``, which the recommender maps back via
  ``hand[card_index]["stable_idx"]`` where ``card_index`` is the
  *position* in the translated hand list (0-based) -- i.e. positional in
  our own emitted list, matching the bridge's own ``play`` verb
  addressing (``card_index`` always indexes ``hand[card_index]``). We set
  each hand card's ``index`` to its position so ``stable_idx`` round-trips
  as exactly the bridge ``card_index`` to send back.
* **``unplayable`` keyword.** Only set when the bridge's
  ``unplayable_reason`` mentions ``HasUnplayableKeyword`` -- NOT whenever
  ``playable`` is false. The bridge's ``playable`` flag is dynamic (energy
  availability included); naively mapping "not playable right now" to the
  wire's static ``unplayable`` ``CardFlag`` would permanently lock out a
  card that's merely too expensive at this instant, corrupting every
  future-turn rollout the MCTS simulates.
* **``master_deck``.** Bridge sends the full deck at
  ``run_state.master_deck``, and ``sts2_hint.schema.DeckSpec`` documents an
  optional ``master_deck`` slot. We translate it and attach it as a
  top-level ``"master_deck"`` key (this module's own output shape has no
  nested ``deck`` object -- see the module docstring's overview), which
  ``sts2_hint.adapter.translate()`` reads via ``state.get("master_deck")``
  and threads into ``wire["deck"]["master_deck"]`` for
  ``sts2_rs.state_from_dict``.

Known gaps -- fields the headless ``combat_play`` schema can carry that
the bridge does not (yet) send:

* ``context.room_type`` / elite flag: forwarded from a top-level
  ``room_type`` when the bridge sends one (SimCombatHandler emits
  ``RunState.CurrentRoom.RoomType`` since the 2026-08-02 dialect audit;
  deploy-pending); otherwise defaults to ``"Combat"`` (non-elite).
* ``stars`` (Regent mechanic): forwarded from ``player.stars`` when the
  bridge sends it (same deploy-pending mod build); omitted before that.
* Card ``stats`` / ``enchantment`` / ``enchantment_amount``: forwarded
  when present (same deploy-pending mod build -- see ``_translate_card``);
  absent on older mod builds, which leaves the engine's registry/upgrade
  fallbacks in charge exactly as before.

Formerly-listed gaps, now closed:

* ``play_pile``, power ``display_amount`` and power ``vars`` **are now
  forwarded**. All three were added to the headless (``Sts2Headless``) wire by
  the v11/v12 mirror-fidelity fixes and the bridge was left behind on them, so
  the live arm ran the same mirror against a wire that still withheld the card
  mid-play and the CrimsonMantle/WitheringPresence counters. The bridge emits
  them from ``SimCombatHandler.BuildCombatState`` /
  ``SimStateExtractor.SerializePower``; this module passes them through.
* The layered cost fields (``base_cost`` / ``persistent_cost`` /
  ``played_cost`` / ``stable_cost``) **are now forwarded** (see
  ``_translate_card``). The bridge has emitted them since the v15/v20
  mirror-fidelity catch-ups (``SimStateExtractor.SerializeCard``), but this
  module dropped them, so the live reconstruction ran cost_override-only:
  the engine baked transient discounts in permanently and lost every
  turn-boundary/when-played cost reset -- the dominant pile-desync class of
  the first persistent-mirror live sweep.
* The RNG lineage block (``rng.run_seed`` + ``rng.counters``) **is now
  forwarded** (2026-08-02 dialect audit). ``SimStateExtractor.ApplyEnvelope``
  has published it on every ``combat_action`` (and on the ``card_select``
  embedded board) since the mirror's restore path landed, but this module
  dropped it -- so on the live arm ``sts2_hint.adapter._rng_block`` never saw
  a ``run_seed``, ``CombatMirror.reconcile`` never let the world own the
  dice, and every live shuffle/intent prediction was structurally a guess
  (verified: no ``mirror_live2.jsonl`` world_wire carries ``run_seed``).
  Forwarded ALL-OR-NOTHING, mirroring the producer/consumer contract: a
  partial block (seed without counters) is worse than none.
* ``player.gold`` **is now forwarded** (same audit) from
  ``run_state.gold``. The gold-gated relic hooks (SealOfGold's 5-gold
  turn-start energy buy) rebuilt against the engine default of no gold on
  the live arm only -- the exact class the v27wr sweep pinned on the DLL arm
  (90 of 178 ``$.player.energy`` divergences) fixed there but not here.
* ``EnemySpec.move_state`` **is now forwarded** (see ``_translate_enemy``).
  The bridge's telegraphed ``next_move_id`` is the monster's current
  ``MonsterState`` id, and those id strings match sts2-rs's enemy-AI state
  names 1:1, so ``sts2_rs.state_from_dict`` reconstructs the real move-state
  machine (``force_current_state_by_name``) and rollouts replay the true
  moveset -- fixing the prior overconfidence on scaling bosses (e.g. the Kin
  Priest, whose ``RITUAL_MOVE`` stacks Strength every cycle) where the
  heuristic-AI fallback made MCTS score a losing fight as a near-certain win.
"""

from __future__ import annotations

import logging
from typing import Any

from sts2_client import DEFAULT_GAME_VERSION, SUPPORTED_GAME_VERSIONS, normalize_game_version

logger = logging.getLogger(__name__)


class UnsupportedGameVersionError(ValueError):
    """The live game reports a build this advisor has no engine for (GH #800).

    ``str(exc)`` is written for the player: the mod renders it in the HUD as
    the advice note, so it names the build and what to do about it.
    """

    def __init__(self, raw: object) -> None:
        self.raw = raw
        super().__init__(
            f"Spire Mentor's advisor does not support game version {raw!r} "
            f"(supported: {', '.join(SUPPORTED_GAME_VERSIONS)}). Update the advisor, "
            "or switch the game to a supported build in Steam (Properties > Betas)."
        )


def envelope_game_version(msg: dict[str, Any]) -> str:
    """The game build a bridge message was produced by, normalized (``"v0.111.0"``).

    Read from the envelope's ``game_version`` (``SimStateExtractor.ApplyEnvelope``,
    PROTOCOL.md section 1). Absent, ``None`` or blank -- every mod build before
    the field -- means :data:`sts2_client.DEFAULT_GAME_VERSION` (v0.107.1), so an
    old mod gets exactly the engine it always got.

    Raises:
        UnsupportedGameVersionError: a build outside
            :data:`sts2_client.SUPPORTED_GAME_VERSIONS`, or a non-string value.
    """
    raw = msg.get("game_version")
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return DEFAULT_GAME_VERSION
    if not isinstance(raw, str):
        raise UnsupportedGameVersionError(raw)
    try:
        return normalize_game_version(raw)
    except ValueError:
        raise UnsupportedGameVersionError(raw) from None


#: Bridge ``run_state.character`` (SCREAMING_CASE) -> the Title-Case
#: literal :func:`sts2_hint.adapter.translate` accepts as its
#: ``character=`` kwarg. Pass the result as ``RecommendConfig.character``.
CHARACTER_ID_TO_DISPLAY: dict[str, str] = {
    "IRONCLAD": "Ironclad",
    "SILENT": "Silent",
    "DEFECT": "Defect",
    "NECROBINDER": "Necrobinder",
    "REGENT": "Regent",
}

#: Bridge Orb ``id`` -> the stem ``sts2_hint.canonical_maps.ORB_TITLE_TO_ID``
#: keys on (the localized display title: ``"Lightning"``, ``"Frost"``, ...).
#: Two dialects reach us: ``PROTOCOL.md`` documents CamelCase
#: ``"LightningOrb"``, but the live v0.107.1 game's ``OrbModel.Id.Entry`` is
#: SCREAMING_SNAKE (``"LIGHTNING_ORB"`` -- observed on every Defect wire).
_ORB_ID_SUFFIX = "Orb"
_ORB_ID_SUFFIX_SNAKE = "_ORB"

#: The bridge Creature ``id`` (``Monster.Id.Entry``, via
#: ``SimStateExtractor.SafeCreatureId``) of the Necrobinder's Osty. Used to pick
#: Osty out of ``PlayerCombatState.Pets``, which also holds Byrdpip/PaelsLegion.
_OSTY_PET_ID = "OSTY"

#: ``run_state.character`` value the headless twin gates its "no Osty" sentinel
#: on (RunSimulator.cs:3988-3990).
_NECROBINDER_CHARACTER_ID = "NECROBINDER"


def character_display_name(character_id: str | None) -> str | None:
    """Map a bridge ``run_state.character`` id to the adapter's Title-Case form.

    Args:
        character_id: e.g. ``"IRONCLAD"``, or ``None`` if no run/character
            is known yet.

    Returns:
        e.g. ``"Ironclad"``, or ``None`` if unmapped/absent (the caller
        should then rely on :mod:`sts2_hint.adapter`'s own
        card-id-prefix inference).
    """
    if not character_id:
        return None
    return CHARACTER_ID_TO_DISPLAY.get(character_id.upper())


def run_ascension(run_state: dict[str, Any], default: int = 0) -> int:
    """The ascension a live board must be rebuilt at.

    The mod reports the run's level as ``run_state.ascension``
    (``SimStateExtractor.cs``). A combat lane that rebuilds the board without
    it plans every enemy turn after the current telegraph against A0 tables
    (GH #848).

    Args:
        run_state: The bridge state's ``run_state`` block (may be empty).
        default: Used only when the mod sent no level, e.g. the
            ``RecommendConfig.ascension`` the lane was built with.

    Returns:
        The run's ascension level.
    """
    value = run_state.get("ascension")
    return default if value is None else int(value)


def _normalize_affliction_name(name: str) -> str:
    """Normalize a per-card affliction onto the class-name wire vocabulary
    (``card_affliction_kind_from_wire`` in sts2-rs). Converts the unresolved
    headless loc-key form ``"<NAME>.title"`` -> ``"Name"`` (headless echoes
    the key when ``localization_eng/afflictions.json`` lacks the entry —
    currently Tainted/Devoured/Weighted); the bare class name passes through.
    """
    if name.endswith(".title"):
        stem = name[: -len(".title")]
        return stem.replace("_", " ").title().replace(" ", "")
    return name


def _translate_power(power: dict[str, Any]) -> dict[str, Any]:
    """One bridge Power object -> a headless power dict.

    ``display_amount`` and ``vars`` are forwarded verbatim for
    :func:`sts2_hint.adapter._translate_power` to interpret — it owns the
    ``amount``-vs-``DisplayAmount`` rule (WitheringPresence's countdown) and the
    ``_POWER_VAR_TO_ENGINE_FIELD`` table (CrimsonMantle/Inferno ``SelfDamage``).
    Passed through as ``None``-tolerant keys so a mod build that predates the
    bridge emitting them behaves exactly as before rather than claiming zero.
    """
    return {
        "id": power.get("id"),
        "name": power.get("name"),
        "amount": power.get("amount", 0),
        "display_amount": power.get("display_amount"),
        "vars": power.get("vars"),
    }


def _keywords_for_hand_card(card: dict[str, Any]) -> list[str]:
    """Static keyword flags for a bridge hand Card object.

    Only flags that are true *card properties* (independent of the
    current energy/turn state) are forwarded -- see the module docstring's
    "``unplayable`` keyword" note for why ``playable: false`` alone must
    NOT set the ``unplayable`` flag.
    """
    kws: list[str] = []
    if card.get("retain"):
        kws.append("retain")
    if card.get("ethereal"):
        kws.append("ethereal")
    if card.get("innate"):
        kws.append("innate")
    reason = card.get("unplayable_reason") or ""
    if "HasUnplayableKeyword" in reason:
        kws.append("unplayable")
    return kws


def _translate_card(card: dict[str, Any], index: int, *, hand: bool = False) -> dict[str, Any]:
    """One bridge Card object (any pile) -> a headless CombatCard dict.

    Args:
        card: Bridge Card object (see ``PROTOCOL.md`` "Card object").
        index: Assigned wire ``index`` -- for hand cards this MUST be the
            card's position within the hand list (see module docstring's
            "Hand ``stable_idx``" note); for other piles any unique value
            is fine, it's otherwise unread.
        hand: Whether this card came from ``hand`` -- gates keyword-flag
            translation, mirroring ``sts2_hint.adapter._translate_card``.
    """
    out: dict[str, Any] = {
        "index": index,
        "id": card.get("id"),
        "name": card.get("name"),
        "type": card.get("type"),
        # effective_cost is authoritative (handles X-cost cards); fall
        # back to cost if a fixture/older snapshot omits it. For non-X cards
        # the two are identical by construction (both are
        # ``Max(0, GetWithModifiers(All))`` -- decomp ``CardEnergyCost``).
        "cost": card.get("effective_cost", card.get("cost")),
        "upgraded": bool(card.get("upgrade_level")) or bool(card.get("upgraded")),
    }
    # The layered cost model -- forwarded verbatim so
    # ``sts2_hint.adapter._translate_card`` sees the identical shape the
    # headless (Sts2Headless) wire gives it. ``base_cost`` is the instance's
    # OWN cost (local modifiers applied, global power hooks excluded -- C#
    # ``GetWithModifiers(CostModifiers.Local)``, ``-1`` = X-cost sentinel);
    # ``persistent_cost``/``played_cost``/``stable_cost`` are what it becomes
    # after the end-of-turn / when-played / both cleanup edges. Without these
    # the engine rebuilds ``Card.cost`` from the resolved ``cost`` alone, and
    # every transient discount bakes in as a permanent one while every
    # per-turn dynamic (Stomp, Cascade) loses its spring-back -- on the first
    # persistent-CombatMirror live sweep 171 of 451 pile-only desync records
    # differed ONLY in the cost element of the pile card key, dominated by
    # exactly those dynamic-cost cards (logs/diag/mirror_live1.jsonl). The
    # ``isinstance`` guard keeps a pre-field mod build's absent/null values
    # off the wire so absent behaves exactly as before rather than claiming a
    # cost of ``None``.
    for cost_key in ("base_cost", "persistent_cost", "played_cost", "stable_cost"):
        v = card.get(cost_key)
        if isinstance(v, int) and not isinstance(v, bool):
            out[cost_key] = v
    # Current base stats (the card's DynamicVars: post-upgrade,
    # post-enchantment, post-permanent-growth base values). The headless
    # dialect ships them as ``stats`` and ``sts2_hint.adapter._translate_card``
    # maps ``damage``/``block`` onto the engine's ``base_damage``/``base_block``
    # -- the ONLY wire signal that a card's base differs from its registry
    # spec beyond the upgrade flag (Sharp/Nimble enchantment deltas, Rampage's
    # own permanent growth). Absent on pre-field mod builds, which leaves the
    # engine's registry/upgraded-build fallback in charge exactly as before.
    stats = card.get("stats")
    if isinstance(stats, dict):
        clean_stats = {
            k: v for k, v in stats.items() if isinstance(v, int) and not isinstance(v, bool)
        }
        if clean_stats:
            out["stats"] = clean_stats
    # Run-persistent enchantment (Nimble/Sharp/Steady/...). The engine
    # resolves the name via ``EnchantmentId::from_wire_name`` (case- and
    # punctuation-insensitive, ``.title``-suffix tolerant), so the bridge's
    # ``Id.Entry`` ("NIMBLE") and the headless localized title ("Nimble")
    # both land on the same kind. Amount follows the headless convention:
    # absent means one stack (``ParsedCard::enchantment_amount``).
    enchantment = card.get("enchantment")
    if isinstance(enchantment, str) and enchantment:
        out["enchantment"] = enchantment
        ench_amount = card.get("enchantment_amount")
        if isinstance(ench_amount, int) and not isinstance(ench_amount, bool):
            out["enchantment_amount"] = ench_amount
    # Per-card affliction (Bound/Tainted/...): forwarded so the engine's
    # reconstruction respects play vetoes like ChainsOfBinding (the Queen
    # fight) and self-damage taxes like Tainted (Infested Prism) inside
    # rollouts, not just at the root mask. Headless without a localized
    # afflictions.json entry emits an unresolved loc key ("TAINTED.title");
    # normalize it onto the class-name wire vocabulary or state_from_dict
    # silently drops the affliction (the un-tainted-deck gap).
    affliction = card.get("affliction")
    if isinstance(affliction, str) and affliction:
        out["affliction"] = _normalize_affliction_name(affliction.split(" / ", 1)[0].strip())
        stacks = card.get("affliction_stacks", card.get("affliction_amount"))
        if isinstance(stacks, int):
            out["affliction_stacks"] = stacks
    # Per-instance combat scratch the mod disclosed (``combat_vars``, the
    # engine's ``VarKey`` wire names): forwarded for ``sts2_hint.adapter``.
    combat_vars = card.get("combat_vars")
    if isinstance(combat_vars, dict) and combat_vars:
        out["combat_vars"] = dict(combat_vars)
    if hand:
        kws = _keywords_for_hand_card(card)
        if kws:
            out["keywords"] = kws
    return out


def _translate_pile(
    cards: list[dict[str, Any]], start_index: int, *, hand: bool = False
) -> tuple[list[dict[str, Any]], int]:
    """Translate a list of bridge Card objects, assigning sequential indices.

    Returns the translated list and the next free index (for the caller
    to thread through subsequent piles).
    """
    out = []
    idx = start_index
    for card in cards:
        out.append(_translate_card(card, idx, hand=hand))
        idx += 1
    return out, idx


def _translate_relic(relic: dict[str, Any]) -> dict[str, Any]:
    """One bridge Relic object -> a headless relic dict (id- and title-keyed).

    The headless dialect (RunSimulator.cs relic serializer) carries the live
    progress as a TOP-LEVEL ``counter`` plus the ToyBox ``is_wax``/
    ``is_melted`` flags, and that is what ``sts2_hint.adapter._translate_relic``
    reads. Nesting the counter under ``vars`` (the old shape here) meant every
    counter relic (Nunchaku, Kusarigama, HappyFlower, Pen Nib, ...)
    reconstructed with ``counter=None`` on the live path only, and melted wax
    relics stayed hook-active in the mirror.

    The canonical ``id`` (``RelicModel.Id.Entry``) is forwarded alongside the
    ``name`` because the live ``name`` is neither stable nor complete:

    * ``SimStateExtractor.SerializeRelic`` sends ``relic.Title.GetFormattedText()``,
      and ``RelicModel.Title`` PREFIXES a wax relic with ToyBox's
      ``"Wax {Title}"`` template (RelicModel.cs:48-61, ToyBox.cs:26,
      ``localization_eng/relics.json`` ``"TOY_BOX.waxRelicPrefix"``). Every wax
      title therefore misses ``RELIC_TITLE_TO_ID`` -- and ToyBox grants FOUR
      wax relics at once (ToyBox.cs:87-95), so all four vanished from every
      reconstructed live combat state.
    * Relics that exist in the live build but not in this repo's
      ``localization_eng/`` snapshot (FISHING_ROD, SILKEN_TRESS) have no title
      key at all and soft-dropped the same way.

    ``sts2_hint.adapter._translate_relic`` prefers a registry-valid ``id`` and
    keeps the title path as the fallback, mirroring what the headless wire
    already does (RunSimulator.cs:5047 emits both ``id`` and ``name``).
    """
    variables: dict[str, int] = {}
    stack_count = relic.get("stack_count")
    if isinstance(stack_count, int) and stack_count > 1:
        variables["stack_count"] = stack_count
    out: dict[str, Any] = {
        "id": relic.get("id"),
        "name": relic.get("name"),
        "vars": variables or None,
    }
    counter = relic.get("counter")
    if isinstance(counter, int) and not isinstance(counter, bool):
        out["counter"] = counter
    # Same only-when-true convention as both C# producers' wires.
    if relic.get("is_wax"):
        out["is_wax"] = True
    if relic.get("is_melted"):
        out["is_melted"] = True
    return out


def _translate_intent(intent: dict[str, Any]) -> dict[str, Any]:
    """One bridge Intent object -> a headless intent dict.

    Uses ``intent_type`` (the ``IntentType`` enum member, e.g.
    ``"Attack"``/``"Buff"``) for the headless ``type`` field --
    ``sts2_hint.canonical_maps.INTENT_TYPES`` is exactly that enum's
    value set, whereas the bridge's ``kind`` is the raw C# intent *class*
    name (``"SingleAttackIntent"``) which doesn't match it.
    """
    out: dict[str, Any] = {"type": intent.get("intent_type")}
    if intent.get("damage") is not None:
        out["damage"] = intent["damage"]
    if intent.get("hits") is not None:
        out["hits"] = intent["hits"]
    return out


def _translate_enemy(enemy: dict[str, Any], index: int) -> dict[str, Any]:
    """One bridge Creature (enemy) object -> a headless enemy dict."""
    out = {
        "index": index,
        "name": enemy.get("name"),
        "hp": enemy.get("hp"),
        "max_hp": enemy.get("max_hp"),
        "block": enemy.get("block", 0),
        # The bridge ships the FULL creature list (corpses inline, matching
        # its own target_index space). Flag corpses so the reconstruction
        # builds them as proper 0-hp revivers (Decimillipede REATTACH,
        # IllusionPower) instead of pseudo-alive hp-0 creatures.
        "is_dead": not enemy.get("is_alive", True),
        "powers": [_translate_power(p) for p in (enemy.get("powers") or [])],
        "intents": [_translate_intent(i) for i in (enemy.get("intents") or [])],
        # Enemy AI move-state id, so sts2-rs rebuilds the real move-state
        # machine (state_dict.rs `force_current_state_by_name`) and rollouts
        # replay the actual moveset past turn 1 instead of falling back to the
        # heuristic enemy AI. The bridge's telegraphed `next_move_id` (the
        # monster's current MonsterState id, e.g. Kin Priest `RITUAL_MOVE`)
        # IS that current state -- the id strings match the sts2-rs AI state
        # names 1:1 (verified: ORB_OF_FRAILTY_MOVE/ORB_OF_WEAKNESS_MOVE/
        # BEAM_MOVE/RITUAL_MOVE). An unknown/absent id falls back to None in
        # sts2-rs (the prior heuristic-AI behavior), so this never regresses a
        # monster whose id doesn't match.
        "move_state": enemy.get("move_state") or enemy.get("next_move_id"),
    }
    # Allowlisted private-AI-state disclosures (both C# producers emit the
    # same field names -- ``SimStateExtractor.AppendMonsterHiddenState`` /
    # ``RunSimulator.AppendMonsterHiddenState``): ``fabricator_last_spawned``
    # is the last-spawned bot's canonical ``Id.Entry`` (Fabricator's next
    # SpawnBot pool excludes it; the engine otherwise infers it from the
    # max-uid living bot, unrecoverable once that bot died) and
    # ``curse_of_knowledge_counter`` is KnowledgeDemon's curse index (the
    # Disintegration 6/7/8 table -- the off-by-one when rebuilt at 0).
    # Forwarded verbatim; absent on pre-field mod builds, which keeps the
    # engine's existing inference in charge exactly as before.
    last_spawned = enemy.get("fabricator_last_spawned")
    if isinstance(last_spawned, str) and last_spawned:
        out["fabricator_last_spawned"] = last_spawned
    curse_counter = enemy.get("curse_of_knowledge_counter")
    if isinstance(curse_counter, int) and not isinstance(curse_counter, bool):
        out["curse_of_knowledge_counter"] = curse_counter
    # C# ``Creature.CombatId`` (newer mods): what a combat-log row names its
    # target by, so a co-op replay can find it (``coop_replay``).
    combat_id = enemy.get("combat_id")
    if isinstance(combat_id, int) and not isinstance(combat_id, bool):
        out["combat_id"] = combat_id
    return out


def _orb_type_stem(orb: dict[str, Any]) -> str:
    """Resolve a bridge Orb object onto an ``ORB_TITLE_TO_ID`` key.

    The rendered ``name`` IS that key (``"Lightning"``), so it is preferred --
    the same reason relics and enemies are emitted keyed by ``name``. The
    ``id`` fallback covers both spellings the producers use: CamelCase
    ``"LightningOrb"`` (``PROTOCOL.md``) and the live game's SCREAMING_SNAKE
    ``"LIGHTNING_ORB"``.
    """
    name = orb.get("name")
    if isinstance(name, str) and name.strip():
        return name
    raw_id = orb.get("id") or ""
    if raw_id.endswith(_ORB_ID_SUFFIX):
        return raw_id[: -len(_ORB_ID_SUFFIX)]
    if raw_id.endswith(_ORB_ID_SUFFIX_SNAKE):
        return raw_id[: -len(_ORB_ID_SUFFIX_SNAKE)].replace("_", " ").title()
    return raw_id


def _translate_orb(orb: dict[str, Any]) -> dict[str, Any]:
    """One bridge Orb object -> a headless orb dict."""
    return {
        "type": _orb_type_stem(orb),
        "passive": orb.get("passive_value", 0),
        "evoke": orb.get("evoke_value", 0),
    }


def _translate_osty(
    pets: list[dict[str, Any]], character_id: str | None = None
) -> dict[str, Any] | None:
    """Necrobinder's Osty companion from the bridge ``pets`` list, if any.

    ``PlayerCombatState.Pets`` is a LIST (PlayerCombatState.cs:17,27) and Osty
    is only one of the things that push onto it: the **Byrdpip** relic (Event
    rarity, ANY character -- Byrdpip.cs:71) and **PaelsLegion** (Ancient, any
    character -- PaelsLegion.cs:223) both ``AddPet`` too. The game's own
    accessor is ``Player.Osty => PlayerCombatState.GetPet<Osty>()``
    (Player.cs:134 -> PlayerCombatState.cs:263, filtering ``p.Monster is Osty``)
    and that is what the headless twin serializes (RunSimulator.cs:3976), so we
    select by the creature's canonical id rather than taking ``pets[0]``.

    Taking ``pets[0]`` blindly invented an Osty on non-Necrobinder runs -- an
    Ironclad holding PaelsLegion reconstructed a 9999/9999-HP "Osty" whose
    DieForYou power soaks every unblocked attack (observed verbatim in
    ``logs/realgame/20260802T221831Z_9884_4.jsonl``) -- and on a Necrobinder
    holding one of those relics whichever pet happened to be first won.

    Args:
        pets: The bridge's ``state["pets"]`` list (may be empty/absent).
        character_id: The bridge's ``run_state.character`` (SCREAMING_CASE).
            Only used to decide whether an Osty-less board should emit the
            headless's explicit "no Osty" sentinel.

    Returns:
        A headless ``osty`` dict, or ``None`` when the key should be omitted.
    """
    osty = next((p for p in pets if str(p.get("id") or "").upper() == _OSTY_PET_ID), None)
    if osty is None:
        # The headless twin emits an explicit sentinel rather than omitting the
        # key when a Necrobinder simply has no Osty out (RunSimulator.cs:3988-3990);
        # the bridge omits it, which left the two dialects disagreeing.
        if (character_id or "").upper() == _NECROBINDER_CHARACTER_ID:
            return {"alive": False, "hp": 0, "max_hp": 0, "block": 0}
        return None
    return {
        "alive": bool(osty.get("is_alive", False)),
        "hp": osty.get("hp", 0),
        "max_hp": osty.get("max_hp", 0),
        "block": osty.get("block", 0),
    }


def to_headless_combat_play(state: dict[str, Any]) -> dict[str, Any]:
    """Translate a v2 bridge ``combat_action`` snapshot to a headless ``combat_play`` dict.

    Args:
        state: The ``state`` object of a bridge message with
            ``state["type"] == "combat_action"`` (see ``PROTOCOL.md``
            section 3, "combat_action") -- e.g. the ``state`` field of one
            JSONL line logged by :mod:`realgame_random`/:mod:`realgame_mcts`.

    Returns:
        A dict shaped exactly like Sts2Headless's own ``combat_play``
        decision JSON (``{"type": "decision", "decision": "combat_play",
        ...}``), ready for :func:`sts2_hint.adapter.translate` (called
        downstream by :func:`sts2_hint.recommender.recommend`/``recommend_all``).

    Raises:
        KeyError, TypeError: On a structurally malformed ``state`` (e.g.
            missing ``"player"``). Callers should catch broadly and fall
            back to a non-MCTS policy -- this function does not itself
            guard against a malformed bridge payload beyond what plain
            dict access raises.
    """
    if state.get("type") != "combat_action":
        raise ValueError(
            f"to_headless_combat_play: expected state['type'] == 'combat_action', "
            f"got {state.get('type')!r}"
        )

    player = state.get("player") or {}
    run_state = state.get("run_state") or {}

    hand_raw = state.get("hand") or []
    draw_raw = (state.get("draw_pile") or {}).get("cards") or []
    discard_raw = (state.get("discard_pile") or {}).get("cards") or []
    exhaust_raw = (state.get("exhaust_pile") or {}).get("cards") or []
    # The card mid-play — see the note on the bridge's `play_pile` emitter. Only
    # a board serialized during a card's own selection screen has anything here.
    play_raw = (state.get("play_pile") or {}).get("cards") or []
    master_deck_raw = run_state.get("master_deck") or []

    hand, next_idx = _translate_pile(hand_raw, 0, hand=True)
    draw, next_idx = _translate_pile(draw_raw, next_idx)
    discard, next_idx = _translate_pile(discard_raw, next_idx)
    exhaust, next_idx = _translate_pile(exhaust_raw, next_idx)
    play, next_idx = _translate_pile(play_raw, next_idx)
    master_deck, next_idx = _translate_pile(master_deck_raw, next_idx)

    enemies = [_translate_enemy(e, i) for i, e in enumerate(state.get("enemies") or [])]

    # Elite/boss signal for the engine's ``is_elite`` flag (an eval-net
    # feature -- eval_wide index 77). The bridge emits the live
    # ``RunState.CurrentRoom.RoomType`` since the 2026-08-02 dialect audit
    # (deploy-pending); absent keeps the historical "Combat" default.
    room_type = state.get("room_type")
    if not isinstance(room_type, str) or not room_type:
        room_type = "Combat"

    out: dict[str, Any] = {
        "type": "decision",
        "decision": "combat_play",
        "context": {"room_type": room_type},
        "round": state.get("round", 1),
        "energy": player.get("energy", 0),
        "max_energy": player.get("max_energy", player.get("energy", 0)),
        "hand": hand,
        "enemies": enemies,
        "player": {
            "hp": player.get("hp"),
            "max_hp": player.get("max_hp"),
            "block": player.get("block", 0),
            "relics": [_translate_relic(r) for r in (run_state.get("relics") or [])],
            "potions": state.get("potions") or [],
            # Per-turn play counters (5 buckets, Rust CardType order) --
            # feeds the Ringing/Normality/Sloth ShouldPlay gates so the
            # rebuilt search state stops offering plays the live game
            # silently refuses mid-turn. Absent on pre-field mod builds.
            "cards_played_this_turn": player.get("cards_played_this_turn"),
            # ChainsOfBinding's once-per-turn Bound gate (Queen fight) --
            # same rationale as the buckets above. Absent on older mods.
            "bound_card_played_this_turn": player.get("bound_card_played_this_turn"),
            # Run gold, from the bridge envelope's ``run_state`` -- gates the
            # in-combat gold-spending relic hooks (SealOfGold's 5-gold
            # turn-start energy buy). ``sts2_hint.adapter`` only forwards an
            # int, so a pre-field/absent value degrades to the engine default
            # exactly as before. This was the v27wr sweep's dominant
            # ``$.player.energy`` class on the DLL arm; the live arm dropped
            # the field here and kept the bug after the DLL fix.
            "gold": run_state.get("gold"),
        },
        "player_powers": [_translate_power(p) for p in (player.get("powers") or [])],
        "draw_pile": draw,
        "discard_pile": discard,
        "exhaust_pile": exhaust,
        # Read by `sts2_hint.adapter.translate()` into `wire["deck"]["play_pile"]`.
        "play_pile": play,
        # Consumed by sts2_hint.adapter.translate() (see module docstring's
        # "master_deck" note) -- threaded into wire["deck"]["master_deck"].
        "master_deck": master_deck,
    }
    # Belt CAPACITY (C# ``Player.MaxPotionCount``; 2 under A4 Tight Belt) from
    # the bridge's run block. ``sts2_hint.adapter.translate`` forwards it as
    # the wire's ``player.potion_slots``; without it a rebuilt board's
    # slot-filling potions fill as if the belt had 5 slots (GH #893).
    slot_count = run_state.get("potion_slot_count")
    if isinstance(slot_count, int) and not isinstance(slot_count, bool):
        out["player"]["potion_slots"] = slot_count

    # Co-op teammates (PROTOCOL.md ``allies``), in the mod's order: an
    # ``AnyAlly`` card lands on the first living one, in the engine and in
    # the mod's autoplay alike. With their hand they are a whole seat whose
    # remaining plays the engine simulates; their cards are numbered after
    # ours so no ``stable_idx`` collides.
    allies = []
    for a in state.get("allies") or []:
        if not isinstance(a, dict):
            continue
        ally: dict[str, Any] = {
            "hp": int(a.get("hp") or 0),
            "max_hp": int(a.get("max_hp") or 0),
            "block": int(a.get("block") or 0),
        }
        if isinstance(a.get("hand"), list):
            ally_hand, next_idx = _translate_pile(a["hand"], next_idx, hand=True)
            ally.update(
                character=character_display_name(a.get("character")),
                energy=int(a.get("energy") or 0),
                max_energy=int(a.get("max_energy") or 0),
                powers=[_translate_power(p) for p in (a.get("powers") or [])],
                hand=ally_hand,
                ended_turn=bool(a.get("ended_turn")),
            )
            # The rest of the seat (newer mods), for replaying a recorded fight
            # as a whole co-op fight: piles numbered after the hand, relics,
            # stars. Absent on older mods, which leaves the advisor's board as
            # it was.
            for key in ("draw_pile", "discard_pile", "exhaust_pile", "play_pile"):
                cards = (a.get(key) or {}).get("cards")
                if isinstance(cards, list):
                    ally[key], next_idx = _translate_pile(cards, next_idx)
            if isinstance(a.get("relics"), list):
                ally["relics"] = [_translate_relic(r) for r in a["relics"]]
            if isinstance(a.get("stars"), int):
                ally["stars"] = a["stars"]
            if isinstance(a.get("potions"), list):
                ally["potions"] = a["potions"]
        if a.get("net_id") is not None:
            ally["net_id"] = str(a["net_id"])
        allies.append(ally)
    if allies:
        out["allies"] = allies
        # Who we are on the combat log's card-play rows (``owner``): the key a
        # recorded fight is replayed seat by seat with (``coop_replay``).
        if player.get("net_id") is not None:
            out["net_id"] = str(player["net_id"])
        if isinstance(player.get("seat_index"), int):
            out["seat_index"] = player["seat_index"]

    # The run's RNG position, from the bridge envelope
    # (``SimStateExtractor.ApplyEnvelope``): the once-hashed master seed every
    # per-stream salt mixes into, plus each stream's read count.
    # ``sts2_hint.adapter._rng_block`` threads it into the wire, and
    # ``CombatMirror.reconcile`` reads the presence of ``run_seed`` as "the
    # world owns the dice" -- dropping it here meant no live-arm wire ever
    # carried ``run_seed`` and every shuffle/intent the mirror predicted was a
    # seedless guess. Forwarded ALL-OR-NOTHING (both keys or neither),
    # honouring the same pair contract the producer and consumer enforce: a
    # seed without counters rebuilds every stream at count 0, which is worse
    # than no lineage at all.
    rng = state.get("rng")
    if (
        isinstance(rng, dict)
        and isinstance(rng.get("run_seed"), int)
        and not isinstance(rng.get("run_seed"), bool)
        and isinstance(rng.get("counters"), dict)
        and rng["counters"]
    ):
        out["rng"] = {"run_seed": rng["run_seed"], "counters": rng["counters"]}
    # Co-op: the same block under its own key (never ``rng``: live, the
    # teammates draw from the shared streams between our decisions), for a
    # replay that starts at this board (``coop_replay``).
    coop_rng = state.get("coop_rng")
    if (
        isinstance(coop_rng, dict)
        and isinstance(coop_rng.get("run_seed"), int)
        and isinstance(coop_rng.get("counters"), dict)
        and coop_rng["counters"]
    ):
        out["coop_rng"] = {"run_seed": coop_rng["run_seed"], "counters": coop_rng["counters"]}

    # Regent stars (headless emits a top-level ``stars``; the bridge carries
    # it on ``player`` since the 2026-08-02 dialect audit, deploy-pending).
    stars = player.get("stars")
    if isinstance(stars, int) and not isinstance(stars, bool):
        out["stars"] = stars

    # An EMPTY belt is still a belt. The headless twin deliberately emits the
    # ``orbs``/``orb_slots`` pair whenever ``OrbQueue.Capacity > 0`` -- empty
    # list included (RunSimulator.cs:3955-3967) -- and the bridge does the same
    # (``"orbs": [], "orb_capacity": 3`` on every turn-1/belt-cleared Defect
    # decision). Gating on truthiness dropped BOTH keys for those boards, so the
    # engine kept ``OrbSlots::default()`` (capacity 0): the search root had no
    # belt at all, and the first channel inside a rollout snapped a Defect back
    # to the hardcoded 3 (``orbs.rs``), overwriting any belt grown past 3 by
    # Capacitor / Signal Boost / Potion of Capacity.
    orbs_raw = state.get("orbs")
    if orbs_raw is not None:
        out["orbs"] = [_translate_orb(o) for o in orbs_raw]
        out["orb_slots"] = state.get("orb_capacity", len(orbs_raw))

    osty = _translate_osty(state.get("pets") or [], run_state.get("character"))
    if osty is not None:
        out["osty"] = osty

    # The game build the board comes from (GH #800). `sts2_hint.adapter.translate`
    # forwards it to the wire, where `state_from_dict` builds the engine for that
    # build. Emitted only for a non-default build, so a v0.107.1 board (and any
    # board from a mod predating the field) yields the dict it always did.
    game_version = envelope_game_version(state)
    if game_version != DEFAULT_GAME_VERSION:
        out["game_version"] = game_version

    return out


#: Epochs the LIVE save appears not to have revealed yet (GH #367).
#:
#: **FALLBACK ONLY as of GH #374.** SpireMentor now puts the save's real
#: ``UnlockState.ToSerializable().UnlockedEpochs`` on the envelope, and
#: :func:`install_unlock_state_from_wire` supersedes this constant on the first
#: message that carries it — logging a warning when the two disagree, which is
#: the signal that this census has gone stale. This list still covers the gap
#: before the first message arrives, and any mod build predating the field.
#:
#: The engine had no access to the save's ``UnlockState``, so this was inferred
#: from the world wires themselves: ``python -m diag_epoch_census`` counts how
#: often each epoch-gated card is named anywhere in a corpus, and an epoch is
#: all-or-nothing, so a clean 0-of-3 against siblings that appear freely is the
#: signature of an unrevealed epoch.
#:
#: Provenance — the 2026-08-03 A1 corpora (1082 records; Defect 545 / Regent
#: 223 / Necrobinder 186 / Silent 128):
#:
#: * ``REGENT5``  — 0/3, vs Regent2 at 276/84/225 and Regent7 at 155/44/3.
#:   Strongest of the set, and the one with an independent confirmation: it
#:   holds exactly one Attack, which is the +1 ``combat_card_generation`` drift
#:   measured on 37/37 CROSSBOW end-turn records.
#: * ``NECROBINDER5`` / ``NECROBINDER7`` — 0/3 each, vs Necrobinder2 at
#:   330/99/21.
#: * ``SILENT5`` / ``SILENT7`` — 0/3 each, vs Silent2 at 110/3/42. WEAKEST of
#:   the set: only 128 Silent records back it.
#:
#: Defect is the control: all three of its epochs are sighted, so the census
#: can distinguish revealed from unrevealed rather than just reporting silence.
#: No colorless epoch is locked — each has at least one sighting.
#:
#: Absence is weaker evidence than presence, so this is a measurable hypothesis
#: rather than a known fact: if it is wrong for a character, the next live sweep
#: shows a NEW cursor drift for that character instead of the old one.
LIVE_LOCKED_EPOCHS: tuple[str, ...] = (
    "REGENT5_EPOCH",
    "NECROBINDER5_EPOCH",
    "NECROBINDER7_EPOCH",
    "SILENT5_EPOCH",
    "SILENT7_EPOCH",
)


def install_live_unlock_state(override: str | None = None) -> list[str]:
    """Install the live arm's epoch lock into the engine. GH #367.

    C# prunes cards behind unrevealed epochs out of every card pool
    (``CardPoolModel.GetUnlockedCards``), which the engine did not model at all
    — so its generation pools ran one epoch too large and its
    ``combat_card_generation`` cursor drifted by exactly that difference.

    ONLY the live arm should call this. The DLL/headless oracle builds
    ``UnlockState.all``, so locking anything there would manufacture the very
    drift this closes, in the opposite direction.

    ``override`` (or ``$STS2_LIVE_LOCKED_EPOCHS``) replaces
    :data:`LIVE_LOCKED_EPOCHS` with a comma-separated epoch-id list; the empty
    string disables locking entirely, which is the escape hatch if the
    inference turns out wrong for a character. Returns the epoch ids installed.
    """
    import os

    import sts2_rs

    raw = override if override is not None else os.environ.get("STS2_LIVE_LOCKED_EPOCHS")
    if raw is None:
        epochs = list(LIVE_LOCKED_EPOCHS)
    else:
        epochs = [e.strip().upper() for e in raw.split(",") if e.strip()]
    # Raises on an unknown id rather than silently no-op'ing — a typo that
    # quietly disabled the filter would present as an unexplained residual
    # drift, which is the exact bug class this closes.
    locked = sts2_rs.set_locked_epochs(epochs)
    logger.info(
        "unlock state: %d epoch(s) locked, %d card(s) pruned from every pool: %s",
        len(epochs),
        locked,
        ", ".join(epochs) or "(none)",
    )
    return epochs


#: Whether a wire-carried unlock state has already been installed this process.
#:
#: The wire repeats ``unlocked_epochs`` on every run-level message, but the
#: install is a process-global mutation and the log line is worth exactly once.
#: A latch also means the census fallback is only ever *replaced*, never
#: flip-flopped with, if some later message serializes the field as null.
_WIRE_UNLOCKS_INSTALLED: list[str] | None = None


def locked_epochs_from_wire(run_state: dict) -> list[str] | None:
    """Invert the wire's REVEALED epoch list into the locked set. GH #374.

    SpireMentor emits ``run_state.unlocked_epochs`` =
    ``UnlockState.ToSerializable().UnlockedEpochs`` — what the live save has
    revealed. The engine wants the complement, over the epochs *it* models;
    C# knows many more (Relic1-5, Silent1/3/4/6, Underdocks, Darv, …), so the
    coverage list comes from ``sts2_rs.known_epochs()`` rather than a second
    copy of the table here.

    Returns ``None`` when the producer did not carry the field — an OLD MOD, or
    a serialization failure. That is deliberately distinct from ``[]``, which
    is a save that has revealed everything. Conflating them would silently
    unlock all 60 gated cards on a mod that predates this field, reintroducing
    #367's drift with no signal.
    """
    import sts2_rs

    revealed = run_state.get("unlocked_epochs")
    if revealed is None:
        return None
    seen = {str(e).strip().upper() for e in revealed}
    return [e for e in sts2_rs.known_epochs() if e.upper() not in seen]


def install_unlock_state_from_wire(run_state: dict) -> list[str] | None:
    """Install the live save's real unlock state off the wire. GH #374.

    Idempotent and cheap: latches after the first message that carries the
    field, so callers can invoke it per-message without thinking about it.

    Returns the locked epoch ids installed, or ``None`` if the wire did not
    carry an unlock state (in which case whatever
    :func:`install_live_unlock_state` put in at startup stays).

    ``$STS2_LIVE_LOCKED_EPOCHS`` still wins. If someone pinned the lock set
    explicitly they are almost certainly bisecting a drift, and having the wire
    quietly overwrite it mid-session would make that unfalsifiable.
    """
    global _WIRE_UNLOCKS_INSTALLED

    import os

    import sts2_rs

    if _WIRE_UNLOCKS_INSTALLED is not None:
        return _WIRE_UNLOCKS_INSTALLED
    if os.environ.get("STS2_LIVE_LOCKED_EPOCHS") is not None:
        return None

    locked = locked_epochs_from_wire(run_state)
    if locked is None:
        return None

    count = sts2_rs.set_locked_epochs(locked)
    _WIRE_UNLOCKS_INSTALLED = locked

    # The census was a snapshot of one save at one moment, and going stale is
    # its whole failure mode — so say so loudly when the wire disagrees. This
    # line is the signal #374 exists to create: without it, a revealed epoch
    # presents only as an unexplained +1 combat_card_generation drift found by
    # mining a fresh corpus.
    inferred = set(LIVE_LOCKED_EPOCHS)
    if set(locked) != inferred:
        logger.warning(
            "unlock state: wire DISAGREES with the hardcoded census. "
            "wire-locked=[%s] census-locked=[%s]; newly revealed=[%s], "
            "newly locked=[%s]. The census constant LIVE_LOCKED_EPOCHS is "
            "stale — the wire is authoritative and has been installed.",
            ", ".join(sorted(locked)) or "(none)",
            ", ".join(sorted(inferred)) or "(none)",
            ", ".join(sorted(inferred - set(locked))) or "(none)",
            ", ".join(sorted(set(locked) - inferred)) or "(none)",
        )
    logger.info(
        "unlock state (from wire): %d epoch(s) locked, %d card(s) pruned: %s",
        len(locked),
        count,
        ", ".join(sorted(locked)) or "(none)",
    )
    return locked


def _reset_wire_unlock_latch() -> None:
    """Test hook: clear the once-per-process latch."""
    global _WIRE_UNLOCKS_INSTALLED

    _WIRE_UNLOCKS_INSTALLED = None


__all__ = [
    "CHARACTER_ID_TO_DISPLAY",
    "LIVE_LOCKED_EPOCHS",
    "character_display_name",
    "install_live_unlock_state",
    "install_unlock_state_from_wire",
    "locked_epochs_from_wire",
    "to_headless_combat_play",
]
