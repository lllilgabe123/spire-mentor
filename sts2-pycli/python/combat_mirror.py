"""Persistent per-fight engine mirror for the DLL / real-game drivers.

Plan: ``2026-07-24 persistent combat mirror`` (Phase 1).

Why this exists
===============

When the C# headless DLL (rung 3.5) or the live game (SpireMentor) is the
*world*, the Rust engine has no state of its own: historically every combat
decision rebuilt a throwaway ``CombatState`` from the decision message,
searched it, and threw it away. That works for ``combat_play``, whose message
carries the whole board. It fails for ``card_select``, whose message carries
only the offered cards — no enemies, no hand, no energy, no piles — so the
driver had to *replay* the previous snapshot forward and hope the engine
re-derived an identical option list. Measured on the realdll_v4 sweep (60
runs, 635 mid-combat prompts): 251 searched, 263 replays failed the match
gate, 121 were never attempted, **175 were answered at random**.

A :class:`CombatMirror` holds ONE engine state alive for a whole fight and
advances it by the actions actually committed. A ``card_select`` then needs no
reconstruction at all: the mirror is already parked on the choice the card
opened.

The one rule
============

**Ground truth replays what already happened; the information set decides what
happens next.**

* The mirror holds concrete truth — the true pile order the world reported,
  the RNG cursors carried across decisions. It is never searched directly.
* :meth:`CombatMirror.search` runs on a *clone*, blindfolded at the root by
  the engine's ``hidden_info`` machinery. Clairvoyant profiles are refused
  outright (:meth:`CombatMirror.search`), because a mirror is exactly the
  object that would make clairvoyance work — and therefore exactly the object
  that must not be allowed to.

Transplant reconcile
====================

:meth:`CombatMirror.reconcile` is correct-by-construction: drift cannot
accumulate, because reality wins on every observable, every decision.

1. Rebuild the whole state from the wire (``sts2_rs.state_from_dict``).
2. Carry over from the previous mirror only what the wire cannot express —
   RNG cursors, per-enemy ``intent_history``, draw-pile pins, the belief flag
   (``CombatState.transplant_hidden``).
3. Diff the carried mirror's *prediction* against the fresh reconstruction on
   deterministic observables. That list is the DESYNC report.

Discipline
==========

:data:`MirrorPolicy.STRICT` (the default, and the only mode whose numbers may
be quoted) raises :class:`MirrorDivergence`. :data:`MirrorPolicy.OBSERVE`
(``STS2_MIRROR_OBSERVE=1``) dumps forensics, drops the mirror, and **tags the
run** via :attr:`CombatMirror.tainted` so its results can't be mistaken for
clean. Either way the divergence is appended to
``logs/diag/mirror_divergences.jsonl``.

There is exactly one principled non-crash: an advance that crossed a shuffle
produces a pile order the world rolled independently, so the next decision's
observables are *unverifiable* rather than wrong. Those are tagged and skipped
(:attr:`MirrorStats.reshuffle_skips`), and the reconcile after them
self-corrects.

Known Phase-2 divergence
========================

The engine keeps dead enemies in ``encounter`` (corpses drive revive
machinery: Decimillipede segments, Fogmog's Eye with Teeth); the real game
removes ordinary corpses from its enemy list. When the two lists disagree on
length, ``transplant_hidden`` raises rather than truncating — a kill in a
multi-enemy fight is expected to surface here first when the DLL driver lands.
That is the intended failure mode, not something to paper over: the crash
names the exact mismatch.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, NamedTuple

from sts2_hint.recommender import RecommendConfig

#: Env flag that downgrades STRICT to OBSERVE for a whole process.
OBSERVE_ENV = "STS2_MIRROR_OBSERVE"

#: Env override for the forensic JSONL path.
LOG_ENV = "STS2_MIRROR_LOG"

#: Env flag that arms the engine's damage-source trace and attaches the rows
#: for the diverging step to every divergence record. GH #368.1 — an HP desync
#: names the amount and never the cause, and the mirror re-imposes reality one
#: decision later, so the evidence is gone before anyone can ask. Off by
#: default: it makes the records substantially larger, and the sweeps that do
#: NOT need it should not pay for it.
DAMAGE_TRACE_ENV = "STS2_MIRROR_DAMAGE_TRACE"


class MirrorPolicy(StrEnum):
    """What a divergence does.

    ``STRICT`` raises. ``OBSERVE`` records, drops the mirror, and taints the
    run. There is deliberately no "repair" mode: a mirror that silently
    patches over a mismatch is worse than no mirror, because its search then
    plans against a board that never existed.

    Co-op (:attr:`CombatMirror.teammates`) is not a repair in that sense. A
    teammate's moves land on the enemies between ours, so an enemy field the
    engine could not have predicted is not a misprediction; the reconcile
    already ADOPTS the world's board, and only the carried hidden state is
    kept. That is counted (:attr:`MirrorStats.teammate_adoptions`), never
    silent, and scoped to ``$.encounter``: our own side still diverges.
    """

    STRICT = "strict"
    OBSERVE = "observe"


def policy_from_env(default: MirrorPolicy = MirrorPolicy.STRICT) -> MirrorPolicy:
    """``MirrorPolicy.OBSERVE`` iff ``STS2_MIRROR_OBSERVE=1``, else ``default``."""
    return MirrorPolicy.OBSERVE if os.environ.get(OBSERVE_ENV) == "1" else default


@dataclass(frozen=True)
class Desync:
    """One deterministic observable the mirror predicted wrong.

    ``path`` is a JSON-path-ish label into the wire dict (``$.player.hp``,
    ``$.encounter[1].intents``) so a report line points straight at the field.
    """

    path: str
    mirror: Any
    world: Any

    def __str__(self) -> str:
        return f"{self.path}: mirror={self.mirror!r} world={self.world!r}"


@dataclass
class MirrorStats:
    """Per-mirror counters. The fidelity metric this project reports.

    ``desync_rate_per_1000`` replaces the old "fallback share": under a
    correct mirror there is no fallback to take a share of, so the question
    stops being "how often did we guess?" and becomes "how often did the
    engine mispredict reality?".
    """

    decisions: int = 0
    reconciles: int = 0
    #: Reconciles whose observable diff was non-empty. Numerator of
    #: :attr:`desync_rate_per_1000`.
    desyncs: int = 0
    desync_fields: int = 0
    #: Injected choices the engine's own derivation graded wrong. Counted
    #: SEPARATELY from :attr:`desyncs`: they are a different event over a
    #: different denominator (:attr:`injections`), and folding them into the
    #: reconcile rate inflated a numerator its denominator never counted.
    choice_mismatches: int = 0
    reshuffle_skips: int = 0
    #: Observable paths excluded from a diff because the world's producer
    #: does not emit them (:func:`unreported_paths`). Non-zero means the
    #: fidelity number is measured over an incomplete observable set — the
    #: DLL and live arms report no ``play_pile``, so this is expected to be
    #: exactly ``reconciles`` there and ``0`` against the native oracle.
    unreported_fields: int = 0
    injections: int = 0
    #: Injections where the engine had NOT parked a choice of its own, so the
    #: driver had to fall back to the ``card_select`` catch-all kind. That
    #: kind's resolver is a deliberate no-op, so the pick changes nothing in
    #: the mirror while it changes the world — every such injection is a
    #: guaranteed one-advance misprediction of whatever the screen granted.
    #: Some are legitimate (a combat-START selection the engine never had a
    #: chance to open); the rest are engine modelling gaps. Counted, not
    #: raised, so the size of the gap is measurable before it is closed.
    unparked_injections: int = 0
    #: Per-stream RNG read-count differences seen at reconcile, once the
    #: producer carries the run's dice. A drift means the two engines
    #: consumed different amounts of randomness for the same step — the
    #: one divergence no field diff can see coming, because the board
    #: still looks right this turn and is wrong for the rest of the fight.
    #: Recorded rather than raised: the cursor is hidden state, not one of
    #: the deterministic observables STRICT is defined over.
    cursor_drifts: int = 0
    #: Reconciles where a card was SUSPENDED mid-play, so the mirror's
    #: observables were verified against the wire without rebuilding it.
    #: Rebuilding would delete the suspended tail, which no wire can carry
    #: (oracle Silent-16).
    parked_verifies: int = 0
    #: Reconciles SKIPPED because the WORLD's wire showed a card still in its
    #: ``play_pile`` while the mirror had nothing in flight — the live
    #: producer snapshotted mid-card-resolution (effects half-applied), so no
    #: observable on that board is gradable. The diff is deferred to the next
    #: stable wire; see the world-in-flight comment in :meth:`reconcile`.
    in_flight_skips: int = 0
    #: Reconciles SKIPPED because the world's wire was snapshotted between a
    #: card-select closing and the effect that screen *granted* landing —
    #: the mirror already holds the picked card (added to hand, or its cost
    #: zeroed) and the world does not yet. Same deferral contract as
    #: :attr:`in_flight_skips`, for the choice channel; see
    #: :func:`_choice_effect_in_flight`.
    choice_in_flight_skips: int = 0
    #: Co-op reconciles whose whole diff was enemy-side, so the world's board
    #: was adopted instead of reported (see :class:`MirrorPolicy`). Kept out
    #: of :attr:`desyncs`: a teammate's hit is not an engine misprediction.
    teammate_adoptions: int = 0
    searches: int = 0
    seeds: int = 0
    drops: int = 0

    @property
    def desync_rate_per_1000(self) -> float:
        """DESYNC-ing reconciles per 1000 reconciles. 0.0 when none ran."""
        return 1000.0 * self.desyncs / self.reconciles if self.reconciles else 0.0

    @property
    def choice_mismatch_rate_per_1000(self) -> float:
        """Mis-derived choices per 1000 injections. 0.0 when none ran."""
        return 1000.0 * self.choice_mismatches / self.injections if self.injections else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "decisions": self.decisions,
            "reconciles": self.reconciles,
            "desyncs": self.desyncs,
            "desync_fields": self.desync_fields,
            "choice_mismatches": self.choice_mismatches,
            "reshuffle_skips": self.reshuffle_skips,
            "unreported_fields": self.unreported_fields,
            "injections": self.injections,
            "unparked_injections": self.unparked_injections,
            "cursor_drifts": self.cursor_drifts,
            "parked_verifies": self.parked_verifies,
            "in_flight_skips": self.in_flight_skips,
            "choice_in_flight_skips": self.choice_in_flight_skips,
            "teammate_adoptions": self.teammate_adoptions,
            "searches": self.searches,
            "seeds": self.seeds,
            "drops": self.drops,
            "desync_rate_per_1000": round(self.desync_rate_per_1000, 3),
            "choice_mismatch_rate_per_1000": round(self.choice_mismatch_rate_per_1000, 3),
        }


class MirrorDivergence(RuntimeError):
    """The mirror and the world disagree about a board that must match.

    Carries the forensic ``report`` that was written to the divergence log, so
    a caller that catches this (a sweep driver tallying failures) has the same
    detail the file does without re-reading it.
    """

    def __init__(self, message: str, report: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.report: dict[str, Any] = report or {}


# ---------------------------------------------------------------------------
# Observable extraction
# ---------------------------------------------------------------------------
#
# Both sides of a diff pass through `sts2_rs.state_to_dict`, so normalization
# is identical by construction and a difference is a real difference — never
# an emitter artifact.


def _powers(raw: Iterable[dict[str, Any]] | None) -> list[tuple[str, int]]:
    """Powers as a sorted ``(name, amount)`` list.

    Sorted because the arena's power order is an implementation detail: two
    states with the same powers applied in a different order are the same
    board, and flagging that as a DESYNC would bury the real ones.
    """
    return sorted((str(p.get("name")), int(p.get("amount", 0))) for p in raw or [])


def _osty(raw: dict[str, Any] | None) -> tuple[Any, ...] | None:
    """The Necrobinder pet's identity, or ``None`` when there is no pet.

    ``max_hp`` is included deliberately: the Osty GROWS across a combat, and a
    mirror holding a freshly re-summoned 1-HP pet against a grown one is the
    exact failure this key exists to catch.
    """
    if not raw:
        return None
    return (raw.get("alive"), raw.get("hp"), raw.get("max_hp"), raw.get("block"))


def _card_key(c: dict[str, Any]) -> tuple[Any, ...]:
    """A card's identity for diff purposes.

    Includes ``cost_override`` and the ``enchantment`` (name *and* stack — a
    Nimble 2 and a Nimble 4 Defend block differently) — a Snecko'd hand and a
    clean one are genuinely different boards — but not the per-instance
    ``combat_vars`` blob, which is opaque on both sides.
    """
    return (
        c.get("id"),
        bool(c.get("upgraded")),
        c.get("cost_override"),
        c.get("enchantment"),
        c.get("enchantment_amount"),
        tuple(sorted(c.get("flags") or [])),
    )


def _intent_key(it: dict[str, Any]) -> tuple[Any, ...]:
    """An enemy intent's identity — everything the wire actually carries.

    ``kind`` (not ``type``) is what ``state_to_dict`` emits; both sides of a
    diff pass through that emitter, so reading the adapter's spelling would
    silently compare ``None`` to ``None`` and mask every intent divergence —
    one of the highest-signal fields there is.

    ``amount``/``times`` are compared for **Attack only**. A telegraph is an
    icon plus, for attacks, the damage: the engine's ``Intent::Buff`` carries
    a ``PowerKind`` and stack count that no wire dialect transmits, so
    ``parse_intent`` rebuilds every Buff as ``{Strength, 0}``. Diffing those
    would flag every reconstruction ever made while telling us nothing —
    the real buff comes from the AI's move when it performs, not from the
    telegraph.

    The damage compared is the **baked** one — what the icon actually paints.
    A real wire reports C# ``AttackIntent.GetSingleDamage``, which has already
    run the dealer's Strength/Weak/Shrink and the player's Vulnerable through
    ``ModifyDamage``; the engine's ``Intent::Attack`` stores the raw base from
    the move table, because that is what ``unbake_wire_intent`` reconstructs on
    the way back in. Comparing those two directly diffs raw against baked and
    flags every modified board as a divergence — measured on the v17 corpus,
    161 of 213 unequal attack telegraphs were exactly that, with Strength the
    dominant driver. ``state_to_dict`` therefore emits ``effective_amount``
    alongside ``amount``; prefer it when present so both sides are compared in
    the same units. The world wire has no such key and falls back to
    ``amount``, which is already baked.

    ``times`` is normalised to 1 when absent. The two producers disagree on
    whether a single-hit attack carries the key at all: ``state_to_dict``
    (``sts2-rs/crates/sts2-py/src/state_dict.rs``) emits ``times`` ONLY for
    ``Intent::MultiAttack``, while the bridge always emits ``times: 1``. Left
    raw, that compares ``None`` against ``1`` and flags EVERY single-hit attack
    telegraph as a divergence when a stored wire pair is re-diffed. A missing
    ``times`` means one hit on both sides — C# ``AttackIntent.Repeats``
    defaults to 0 and ``SingleAttackIntent`` paints one icon — so fold the two
    spellings together.
    """
    kind = it.get("kind")
    if kind == "Attack":
        baked = it.get("effective_amount")
        times = it.get("times")
        return (
            kind,
            it.get("amount") if baked is None else baked,
            1 if times is None else times,
        )
    return (kind,)


def enemy_slot_key(enemy: dict[str, Any], index: int, encounter: list[dict[str, Any]]) -> str:
    """An enemy's identity across the two worlds: ``"<kind>#<nth of kind>"``.

    NOT the slot index. The two worlds disagree about slots by construction:
    the game removes an ordinary corpse from its enemy list the moment it
    dies (keeping only revivers, which it ships separately and the adapter
    APPENDS), while the engine keeps every creature in its original spawn
    position so its revive machinery can find it. So one kill shifts every
    later slot, and a slot-wise diff then reports six fields wrong on an
    otherwise perfectly synchronized board — 515 of the 2992 divergences on
    the 60-run realdll_v5 sweep were exactly that.

    Same-kind enemies are numbered in list order, and the LIVING and the DEAD
    are numbered separately. Counting corpses too would defeat the whole
    point: the engine's surviving Nibbit sits behind its own corpse and would
    be numbered ``Nibbit#1`` while the game — which reaped that corpse —
    numbers the same creature ``Nibbit#0``. The keys would then miss each
    other, the corpse would be diffed against the survivor, and the reaping
    rule in :func:`diff_observables` could never fire, because the key it
    computes for the mirror-only enemy names the LIVE one.

    **What this key is not.** Numbering within each namespace is stable under
    reaping only while BOTH worlds reap the same creature. On a mispredicted
    kill — the class that dominates the residual — one side reaps and the other
    does not, and every later same-kind enemy shifts by one: a mirror holding
    ``[NibbitA alive, NibbitB alive]`` against a world holding ``[NibbitB
    alive]`` numbers B as ``Nibbit#1`` while the world calls it ``Nibbit#0``, so
    the keys cross and B is diffed against A. Nothing local can fix that — two
    same-kind enemies are indistinguishable on the wire, so the numbering IS the
    identity and a shift destroys it. What the roster comparison in
    :func:`diff_observables` reports on such a board is correct and is the real
    finding; the per-field differences reported alongside it are noise.
    :func:`_enemy_alignment` refuses to carry hidden state across a shift for
    exactly this reason — see :func:`_unalignable_kinds`.
    """
    kind = str(enemy.get("kind"))
    dead = bool(enemy.get("is_dead"))
    nth = sum(
        1
        for e in encounter[:index]
        if str(e.get("kind")) == kind and bool(e.get("is_dead")) == dead
    )
    return f"{kind}#dead{nth}" if dead else f"{kind}#{nth}"


def observables(wire: dict[str, Any]) -> dict[str, Any]:
    """The deterministic half of a wire dict, flattened to ``{path: value}``.

    Deterministic means: given the same board and the same committed action,
    both engines must produce the same value. Anything the engines are allowed
    to roll independently (the RNG cursors themselves, ``combat_id``, the
    unresolved seed) is deliberately absent.
    """
    player = wire.get("player") or {}
    deck = wire.get("deck") or {}
    out: dict[str, Any] = {
        "$.phase": wire.get("phase"),
        "$.round": wire.get("round"),
        "$.side": wire.get("side"),
        "$.player.hp": player.get("hp"),
        "$.player.block": player.get("block"),
        "$.player.energy": player.get("energy"),
        "$.player.powers": _powers(player.get("powers")),
        "$.player.potions": list(player.get("potions") or []),
        "$.player.orbs": [
            (o.get("type"), o.get("passive"), o.get("evoke")) for o in player.get("orbs") or []
        ],
        # The Necrobinder's pet. It is a creature that absorbs damage aimed at
        # the player (`DieForYouPower`), so a mirror whose Osty disagrees with
        # the game plans against a different amount of soak — as deterministic
        # an observable as player block, and it was not being compared at all.
        # That blindness is why the reconstruction dropping the pet entirely
        # could only ever surface indirectly, as `$.player.hp`.
        "$.player.osty": _osty(player.get("osty")),
    }
    # `play_pile` is here because the wire round-trips it and a mid-combat
    # `card_select` is opened BY a card that is sitting in it. Leave it out
    # and a mirror holding a card in flight against a world reporting none
    # diffs clean — the divergence would be structurally unreportable.
    for pile in ("hand", "draw_pile", "discard_pile", "exhaust_pile", "play_pile"):
        out[f"$.deck.{pile}"] = [_card_key(c) for c in deck.get(pile) or []]

    # Enemies are keyed by IDENTITY, not by slot — see `enemy_slot_key`.
    # The roster list is POSITIONAL, not sorted: C# keeps `_enemies` in
    # encounter-slot order and every random-target effect indexes it
    # (`HittableEnemies`), so order IS an observable. Sorting here made a
    # re-ordered board structurally invisible — 18 v28wr enemy-hp events sat
    # on silently re-ordered rosters where Juggernaut's random target hit
    # the wrong creature (a clean ±8 damage swap between two enemies).
    #
    # What is NOT an observable is where a CORPSE sits in that list. C# keeps
    # dead-but-reviving creatures in their `Enemies` slots, but the headless
    # producer serializes alive and dead into separate lists (`enemies` +
    # `dead_enemies`, RunSimulator.cs:3710) that the adapter re-concatenates
    # dead-last, while the engine wire and the live bridge ship corpses
    # inline at their spawn slots. Comparing those interleavings verbatim
    # flagged every post-death Decimillipede/Parafright board as a roster
    # divergence (44/158 of the d1 corpus). Partition each side alive-first
    # (stable, so the targetable order stays load-bearing) with corpses
    # appended, and the dialects agree again; `enemy_slot_key` numbers the
    # living and the dead in separate namespaces, so the keys themselves are
    # unchanged by the re-ordering.
    encounter = wire.get("encounter") or []
    canon = sorted(range(len(encounter)), key=lambda i: bool(encounter[i].get("is_dead")))
    out["$.encounter.roster"] = [enemy_slot_key(encounter[i], i, encounter) for i in canon]
    for i, e in enumerate(encounter):
        key = enemy_slot_key(e, i, encounter)
        out[f"$.encounter[{key}].hp"] = e.get("hp")
        out[f"$.encounter[{key}].block"] = e.get("block")
        out[f"$.encounter[{key}].is_dead"] = e.get("is_dead")
        out[f"$.encounter[{key}].powers"] = _powers(e.get("powers"))
        out[f"$.encounter[{key}].intents"] = [_intent_key(it) for it in e.get("intents") or []]
    return out


def _teammate_reachable(path: str) -> bool:
    """Is ``path`` a field a co-op teammate's moves can change under us?

    The enemy side (``$.encounter[X].hp``, ``.powers``, ``.intents``, the
    roster) and the teammate's own creature (``$.allies``: they block, take
    hits and heal on their own). Everything of ours is still ours to predict.
    """
    return path.startswith(("$.encounter", "$.allies"))


def unreported_paths(raw_world_wire: dict[str, Any]) -> frozenset[str]:
    """Observable paths the WORLD's producer does not emit at all.

    Both sides of :func:`diff_observables` are ``state_to_dict`` output, which
    always emits every key — so by diff time "the producer never sent this"
    is indistinguishable from "the producer sent empty". That distinction has
    to be read off the raw wire, before the round-trip normalizes it.

    Only ``play_pile`` needs this today: the Rust wire round-trips it, but
    both C# producers OMIT it unless a card is actually in flight
    (``RunSimulator.CombatPlayState`` sends null when empty, and the
    live-arm translation drops an empty pile — ``sts2_hint.adapter``). A
    producer that predates the field also means "no card is mid-play" by
    its silence, so diffing an absent pile against the mirror's would
    report a divergence whose fault is the producer's, not the mirror's.
    Absent means UNKNOWN-or-empty, not comparable.
    """
    deck = raw_world_wire.get("deck") or {}
    return frozenset() if "play_pile" in deck else frozenset({"$.deck.play_pile"})


def _choice_effect_in_flight(desyncs: list[Desync], picked: list[dict[str, Any]]) -> bool:
    """Is this hand diff exactly the just-picked card's effect not-yet-landed?

    The live producer answers a ``card_select`` the instant the pick arrives
    (``SimCardSelector.GetSelectedCards`` returns; the ``CardSelectCmd`` that
    awaited it then finishes on the action queue), and nothing downstream
    waits for the grant to land the way ``PlayCardAndWaitAsync`` waits for the
    play pile to drain. So the next serialized combat board can be a
    MID-RESOLUTION snapshot of the *choice* channel: the screen is gone, the
    pick is committed, and the card it granted is still on its way into the
    hand. The mirror resolved the same pick atomically, so at that instant the
    mirror is AHEAD and the hand is not gradable — 33 of the 49 pure
    ``$.deck.hand`` records in the ``mirror_live3`` corpus, the other 16
    being the post-reseed echo each one manufactured one decision later.

    Two shapes, one per grant flavour, both proven by the corpus:

    * ``add_generated[_free]_to_hand`` — the mirror's hand carries the picked
      card and the world's does not carry it *anywhere* (27 records);
    * ``make_free_in_hand`` — both hands carry it, the mirror's at cost 0 and
      the world's still at the printed cost (6 records).

    Deliberately narrow, so this defers a snapshot and never grants amnesty:

    * the ONLY diverging path may be ``$.deck.hand``;
    * every card in the symmetric difference must be one the pick just named;
    * the mirror must be the ADVANCED side (it holds a card the world lacks,
      or the same card at a strictly lower cost). A mirror that is BEHIND the
      world looks identical to the post-reseed echo this fixes and to a real
      misprediction, so it is never deferred.

    Args:
        desyncs: The reconcile's observable diff.
        picked: The option dicts (``{"id", "upgraded", …}``) the world's
            selection resolved to.

    Returns:
        True when the diff is fully explained by the pending grant.
    """
    if not desyncs or not picked:
        return False
    if len(desyncs) != 1 or desyncs[0].path != "$.deck.hand":
        return False

    # `_card_key` tuples, re-frozen: a live diff already hands us tuples, but
    # the same check has to run over a JSON-loaded forensic record (the
    # offline corpus replay), where the key's `flags` element is a list.
    def _key(card: Any) -> tuple[Any, ...]:
        return tuple(tuple(v) if isinstance(v, list) else v for v in card)

    mine = Counter(_key(c) for c in desyncs[0].mirror or [])
    theirs = Counter(_key(c) for c in desyncs[0].world or [])
    extra = list((mine - theirs).elements())  # cards only the MIRROR has
    missing = list((theirs - mine).elements())  # cards only the WORLD has
    if not extra or len(missing) > len(extra):
        return False
    named = {(str(o.get("id")), bool(o.get("upgraded"))) for o in picked}
    if any((c[0], c[1]) not in named for c in extra + missing):
        return False
    # A world-only copy is only explainable as the SAME card still at its
    # pre-grant cost — i.e. the mirror already zeroed what the world has yet
    # to. Anything else is a genuine hand disagreement.
    return all(
        any(
            e[0] == c[0]
            and e[1] == c[1]
            and isinstance(e[2], int)
            and isinstance(c[2], int)
            and e[2] < c[2]
            for e in extra
        )
        for c in missing
    )


def diff_observables(
    mirror_wire: dict[str, Any],
    world_wire: dict[str, Any],
    *,
    skip_prefix: str | None = None,
    skip_paths: frozenset[str] = frozenset(),
) -> list[Desync]:
    """Every deterministic observable on which the two dicts disagree.

    Args:
        mirror_wire: What the carried mirror predicts.
        world_wire: What reality reports.
        skip_prefix: Drop paths starting with this. Used for the
            crossed-a-reshuffle case, where the pile paths (and only the
            pile paths) are genuinely unverifiable — see
            :meth:`CombatMirror.reconcile`.
        skip_paths: Drop these exact paths. See :func:`unreported_paths`.
    """
    mine, theirs = observables(mirror_wire), observables(world_wire)
    # An enemy the mirror still has and the world does not is a corpse the
    # game REAPED — the engine deliberately keeps it so its revive machinery
    # can find it, which is a documented modelling difference, not a
    # misprediction. Drop those keys from both sides; a mirror-only enemy
    # that is still ALIVE, or a world-only enemy, survives into the roster
    # comparison below and is reported.
    reaped = {
        key
        for key in _enemy_keys(mirror_wire) - _enemy_keys(world_wire)
        if _enemy_is_dead(mirror_wire, key)
    }
    if reaped:
        drop = {f"$.encounter[{k}].{f}" for k in reaped for f in _ENEMY_FIELDS}
        mine = {k: v for k, v in mine.items() if k not in drop}
        # Filter, don't re-sort: `observables` emits the roster in canonical
        # alive-first order on BOTH sides, and sorting only the mirror's half
        # alphabetically flagged every post-reap board whose survivors aren't
        # coincidentally in lexical order (FlailKnight reaped out of a
        # SpectralKnight/MagiKnight fight kept the roster diverged for the
        # rest of the combat).
        mine["$.encounter.roster"] = [k for k in mine["$.encounter.roster"] if k not in reaped]
    return [
        Desync(path=path, mirror=mine.get(path), world=theirs.get(path))
        for path in sorted(mine.keys() | theirs.keys())
        if mine.get(path) != theirs.get(path)
        and path not in skip_paths
        and not (skip_prefix is not None and path.startswith(skip_prefix))
    ]


#: The per-enemy observable suffixes :func:`observables` emits.
_ENEMY_FIELDS = ("hp", "block", "is_dead", "powers", "intents")


#: Streams whose read count a combat decision can actually move. The wire
#: carries the run-scoped ones too (rewards / shops / transformations), but
#: those only advance between fights, so diffing them mid-combat would report
#: the map layer's business as a combat divergence.
_COMBAT_RNG_STREAMS = (
    "shuffle",
    "monster_ai",
    "combat_card_generation",
    "combat_potion_generation",
    "combat_card_selection",
    "combat_energy_costs",
    "combat_targets",
    "combat_orbs",
)


def _cursor_drift(mirror_wire: dict[str, Any], world_wire: dict[str, Any]) -> list[Desync]:
    """How many rolls the mirror took that the world didn't (and vice versa).

    This is the one observable that says whether the two engines *consumed*
    the same randomness, as opposed to landing on the same values. A monster
    whose AI reads one extra roll produces a correct-looking board this turn
    and a wrong one for the rest of the fight, and no field diff can see it
    coming — the counters can.

    Only meaningful once the producer carries the run's dice: before that the
    mirror's streams are seeded from something unrelated to the game's, so the
    read counts are not comparable at all.
    """
    mine = (mirror_wire.get("rng_counters") or {}) if isinstance(mirror_wire, dict) else {}
    theirs = (world_wire.get("rng") or {}).get("counters") or {}
    out: list[Desync] = []
    for name in _COMBAT_RNG_STREAMS:
        a, b = mine.get(name), theirs.get(name)
        if a is None or b is None or a == b:
            continue
        out.append(Desync(f"$.rng.counters.{name}", a, b))
    return out


def _undisclosed_relic_counters(wire: dict[str, Any]) -> list[str]:
    """Relics the wire named but told us no ``counter`` for.

    A relic's counter is not only a displayed tally — it is also where a
    one-shot per-combat latch lives (Vambrace: "the first card-granted block
    this combat is doubled"). Both producers only emit a counter for relics
    that SHOW one, so those latches arrive as ``None`` and every
    reconstruction re-arms them. Naming them here is what lets the transplant
    keep the carried mirror's value; the ones reality DID report keep
    reality's.
    """
    relics = ((wire.get("player") or {}).get("relics")) or []
    return [
        str(r["id"])
        for r in relics
        if isinstance(r, dict) and r.get("id") and r.get("counter") is None
    ]


def _undisclosed_power_counters(wire: dict[str, Any]) -> list[str]:
    """Powers the wire named but told us no ``counter`` for — either side.

    The power twin of :func:`_undisclosed_relic_counters`: a power whose
    accumulator lives in C# ``InternalData`` (Juggling's attacks-this-turn,
    Possess*'s stolen-stat totals) arrives as ``{name, amount}`` and every
    reconstruction re-zeroed it — the Juggling 3rd-attack clone never fired
    (26 v28wr hand records). Naming them lets the transplant keep the carried
    mirror's accumulation; a counter reality DID report keeps reality's.
    """
    #: Powers whose counter `state_from_dict` DERIVES from the wire
    #: (`display` for the countdown family) — the wire does disclose those,
    #: just not under the `counter` key, so the carry must not override them.
    #:
    #: ORBIT joined on GH #363: C# publishes only ``4 - energySpent % 4`` as
    #: ``DisplayAmount``, and that residue is all the trigger arithmetic needs.
    #: This entry and the ``PowerKind::Orbit`` arm in ``state_dict.rs`` are one
    #: fix in two halves — either alone is a no-op (the carry would overwrite
    #: the derivation) or a regression (the derivation would be dropped for a
    #: mirror counter that has no wire backing).
    derived = {"SLOW", "WITHERING_PRESENCE", "AUTOMATION", "PANACHE", "ORBIT"}
    out: set[str] = set()
    sides = [((wire.get("player") or {}).get("powers")) or []]
    for e in wire.get("encounter") or []:
        sides.append(e.get("powers") or [])
    for powers in sides:
        for p in powers:
            if isinstance(p, dict) and p.get("name") and p.get("counter") is None:
                name = str(p["name"])
                if name not in derived:
                    out.add(name)
    return sorted(out)


#: Wire deck-pile key → the engine locator pile name
#: (`locate_card` / `parse_injected_option` dialect).
_WIRE_PILE_TO_LOCATOR = {
    "hand": "hand",
    "draw_pile": "draw",
    "discard_pile": "discard",
    "exhaust_pile": "exhaust",
}


def _relocate_parked(parked: dict[str, Any], seeded: dict[str, Any]) -> dict[str, Any] | None:
    """Re-locate a stale parked choice's options against the reseeded piles.

    A parked choice that survived a reseed carries ``pile``/``index``
    locators that indexed the DROPPED state, so each option is matched
    one-to-one on ``(id, upgraded)`` in option order against the piles the
    mirror now holds, consuming positions so duplicates resolve to distinct
    cards. An option not found anywhere is dropped rather than guessed —
    a later ``_resolve_locators`` then reports the concrete
    ``unlocatable_option`` instead of a blanket refusal. Returns ``None``
    when no option survives (the carried derivation was entirely stale).

    ``seeded`` must be ``state_to_dict`` output, NOT the raw wire, for the
    same reason ``reconcile``'s parked-verify round-trips before diffing:
    the wire is the producer's dialect and a parked option is the engine's,
    so ``curse:Decay`` never equals ``DECAY`` and the option is silently
    dropped. Only prefixed ids diverge, so the failure is invisible on an
    ordinary board and total on a cursed one. In ``logs/f7-mirror-check16``
    it is every one of the six ``unlocatable_option`` records: the world
    offered a whole hand of 8, the carried derivation came back as 7 —
    short by exactly its unplayable cards (DECAY, DAZED, CLUMSY) — and six
    advices across four Survivor / Dagger Throw screens fell to the rarity
    sort. The engine
    itself offers unplayable cards for discard, exactly as C# does; nothing
    was ever filtering them.
    """
    deck = seeded.get("deck") or {}
    piles = {loc: list(deck.get(key) or []) for key, loc in _WIRE_PILE_TO_LOCATOR.items()}
    used: set[tuple[str, int]] = set()
    cards: list[dict[str, Any]] = []
    for opt in parked.get("cards") or []:
        key = (opt.get("id"), bool(opt.get("upgraded")))
        found: tuple[str, int] | None = None
        for pile_name, pile_cards in piles.items():
            for idx, wc in enumerate(pile_cards):
                if (pile_name, idx) in used or not isinstance(wc, dict):
                    continue
                if (wc.get("id"), bool(wc.get("upgraded"))) == key:
                    found = (pile_name, idx)
                    break
            if found is not None:
                break
        if found is None:
            continue
        used.add(found)
        cards.append({**opt, "pile": found[0], "index": found[1]})
    if not cards:
        return None
    return {**parked, "cards": cards}


def _enemy_keys(wire: dict[str, Any]) -> set[str]:
    enc = wire.get("encounter") or []
    return {enemy_slot_key(e, i, enc) for i, e in enumerate(enc)}


def _enemy_is_dead(wire: dict[str, Any], key: str) -> bool:
    enc = wire.get("encounter") or []
    for i, e in enumerate(enc):
        if enemy_slot_key(e, i, enc) == key:
            return bool(e.get("is_dead"))
    return False


def _living_by_kind(wire: dict[str, Any]) -> Counter[str]:
    """How many LIVING enemies of each kind a wire reports."""
    return Counter(
        str(e.get("kind")) for e in wire.get("encounter") or [] if not bool(e.get("is_dead"))
    )


def _unalignable_kinds(world_wire: dict[str, Any], mirror_wire: dict[str, Any]) -> set[str]:
    """Kinds whose ``#nth`` numbering the two worlds cannot be trusted to share.

    :func:`enemy_slot_key`'s numbering is only an identity while the two worlds
    hold the same *number* of living enemies of that kind. When they do not —
    a mispredicted kill, or a summon one side has and the other does not — the
    numbering shifts on one side only and the keys start naming different
    creatures. There is nothing local left to disambiguate with: two Nibbits at
    the same HP are genuinely indistinguishable on the wire, so the numbering IS
    the identity, and once it shifts the identity is gone.

    Note this is per KIND, not global: numbering counts only same-kind enemies,
    so a Nibbit the world reaped cannot shift the Wurm beside it.

    A same-kind summon lands here too, which is stricter than it strictly has to
    be — a spawn appends, so the enemies before it keep their numbers. But
    "world has one more" is also what a kill the MIRROR predicted and the world
    did not looks like (the mirror parks it in the dead namespace, shortening
    its own living run), and those two are not distinguishable either. The cost
    of refusing is one enemy's carried ``intent_history``, i.e. one turn of
    telegraph prediction; the cost of guessing wrong is that history landing on
    a different creature, which is worse and silent.
    """
    world, mirror = _living_by_kind(world_wire), _living_by_kind(mirror_wire)
    return {k for k in world.keys() | mirror.keys() if world[k] != mirror[k]}


def _enemy_alignment(world_wire: dict[str, Any], mirror_wire: dict[str, Any]) -> list[int | None]:
    """For each of the world's enemies, the carried mirror's slot for it.

    ``None`` where the mirror has no counterpart (a newly-spawned enemy has no
    history to carry) — or where the two worlds' rosters have drifted far enough
    that the counterpart cannot be identified, see :func:`_unalignable_kinds`.
    Matching is otherwise by :func:`enemy_slot_key`, because the two worlds do
    not agree on slot numbering the moment one of them reaps a corpse.
    """
    carried = wire_enemy_index(mirror_wire)
    enc = world_wire.get("encounter") or []
    blocked = _unalignable_kinds(world_wire, mirror_wire)
    return [
        None if str(e.get("kind")) in blocked else carried.get(enemy_slot_key(e, i, enc))
        for i, e in enumerate(enc)
    ]


def wire_enemy_index(wire: dict[str, Any]) -> dict[str, int]:
    """``{identity key -> slot}`` for a wire dict's encounter."""
    enc = wire.get("encounter") or []
    return {enemy_slot_key(e, i, enc): i for i, e in enumerate(enc)}


def _default_log_path() -> Path:
    override = os.environ.get(LOG_ENV)
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[1] / "logs" / "diag" / "mirror_divergences.jsonl"


# ---------------------------------------------------------------------------
# The mirror
# ---------------------------------------------------------------------------

#: Search profiles that would let the mirror leak its own ground truth into
#: the search root. Refused by :meth:`CombatMirror.search`.
_CLAIRVOYANT_PROFILES = frozenset({"clairvoyant"})


class CombatMirror:
    """One live engine state per fight, re-synchronized from reality.

    Typical driver loop for a ``combat_play`` decision::

        mirror.reconcile(wire)                 # or .seed(wire) for a new fight
        result = mirror.search()
        action = best_recommend_action(result)
        world.take_action(action)              # the world commits it
        mirror.advance(action)                 # the mirror follows

    and for a ``card_select``::

        mirror.reconcile(wire)                 # the choice's own combat payload
        mirror.inject_choice(kind, options, min_select, max_select)
        result = mirror.search()
        mirror.advance_through_choice(picks)

    Args:
        cfg: The search profile :meth:`search` uses by default. Must be
            honest — see :meth:`search`.
        policy: Divergence policy. Defaults to the env-resolved value
            (:func:`policy_from_env`).
        label: Free-form tag written into every forensic record (a seed, a
            run id) so a sweep's divergences can be attributed.
        log_path: Forensic JSONL destination. Defaults to
            ``logs/diag/mirror_divergences.jsonl``.
    """

    def __init__(
        self,
        cfg: RecommendConfig | None = None,
        *,
        policy: MirrorPolicy | None = None,
        label: str = "",
        log_path: str | Path | None = None,
    ) -> None:
        self.cfg = cfg or RecommendConfig(hidden_info="beliefdraw", c_puct=2.5, draw_widening_c=2.0)
        self.policy = policy if policy is not None else policy_from_env()
        self.label = label
        self.log_path = Path(log_path) if log_path is not None else _default_log_path()
        self.stats = MirrorStats()
        #: Set once the mirror has recorded a divergence under OBSERVE. A run
        #: with this set must never be reported as a clean measurement.
        self.tainted = False
        #: Why the mirror was last dropped (``""`` for a clean end-of-combat
        #: drop). Surfaced so an OBSERVE sweep can report *what* went wrong
        #: without reopening the divergence log.
        self.drop_reason = ""
        #: Optional ``(paths: list[str]) -> None`` hook, called once per
        #: divergence with the field paths that diverged, BEFORE the policy
        #: decides to raise. Lets a sweep classify divergences live instead of
        #: re-parsing the forensic log. Never swallows the divergence.
        self.on_divergence: Callable[[list[str]], None] | None = None
        #: How many other players share this fight (co-op). Non-zero lets a
        #: reconcile adopt an enemy-side diff instead of reporting it; see
        #: :class:`MirrorPolicy`. The driver sets it from the board.
        self.teammates = 0
        #: The diff the last reconcile adopted under :attr:`teammates`, so a
        #: caller can tally what the teammate changed. Empty otherwise.
        self.last_adopted: list[Desync] = []

        self._state: Any | None = None
        # The state the last `drop()` discarded, kept one divergence long so
        # `_reseed_after_observe` can salvage the undisclosed hidden state
        # (relic latches, history counters, per-card vars) into the reseed.
        self._last_dropped_state: Any | None = None
        self._last_dropped_parked: dict[str, Any] | None = None
        # The choice the ENGINE derived on its own, captured the moment an
        # advance parked on it. `reconcile` rebuilds the state from the wire
        # and so wipes the live choice; this survives, because grading the
        # engine's derivation against the world's option list is the whole
        # point of the injected path.
        self._parked: dict[str, Any] | None = None
        # Forensics for the last screen the engine had NOT opened itself — see
        # `MirrorStats.unparked_injections`. Attached to any divergence report
        # written afterwards, because a no-op `card_select` resolution is the
        # likeliest explanation for a deck path that then diverges.
        self._last_unparked: dict[str, Any] | None = None
        # Did any advance since the last reconcile consume the shuffle stream?
        self._crossed_shuffle = False
        # Kept only for forensics — the last wire the world handed us and the
        # last action we committed.
        self._last_wire: dict[str, Any] | None = None
        self._last_action: dict[str, Any] | None = None
        self._last_result: str | None = None
        self._last_drift: list[Desync] = []
        self._last_injection: dict[str, Any] | None = None
        # The injected screen that has been opened but not yet answered.
        # Distinct from `_last_injection`, which is kept for forensics long
        # after its pick landed: this one is CONSUMED by the answer, so a
        # choice the driver resolved without injecting can never inherit a
        # previous screen's options.
        self._open_injection: dict[str, Any] | None = None
        # The option dicts a just-answered choice picked, armed for exactly
        # one graded reconcile — see `_choice_effect_in_flight`.
        self._choice_in_flight: list[dict[str, Any]] | None = None
        # GH #368.1 forensics. `_damage_trace_on` is latched from the env at
        # construction so a sweep cannot half-arm itself mid-run;
        # `_last_damage_trace` holds the rows produced since the last
        # reconcile, which is exactly the window a desync report needs.
        self._damage_trace_on = os.environ.get(DAMAGE_TRACE_ENV) == "1"
        self._last_damage_trace: list[dict[str, Any]] = []
        self._damage_trace_dropped = 0

    # -- lifecycle ---------------------------------------------------------

    @property
    def alive(self) -> bool:
        """Is a state currently mirrored? False before :meth:`seed` / after :meth:`drop`."""
        return self._state is not None

    @property
    def state(self) -> Any:
        """The live ``sts2_rs.CombatState``. Read it; do not advance it directly.

        Raises:
            MirrorDivergence: When no fight is mirrored — asking for the state
                of a dropped mirror is a driver bug, not a game state.
        """
        if self._state is None:
            raise MirrorDivergence("mirror has no state: seed() a fight first")
        return self._state

    @property
    def wire(self) -> dict[str, Any] | None:
        """The last wire dict reality handed us, or ``None``.

        Drivers need it to map a search result's ``card_index`` back to the
        world's own card handle: the engine indexes the hand it was built from,
        and that hand is ``wire["deck"]["hand"]`` with its ``stable_idx``
        values intact.
        """
        return self._last_wire

    def seed(self, wire: dict[str, Any]) -> None:
        """Start mirroring a fight from a full combat state.

        Args:
            wire: A wire-schema dict (``sts2_rs.state_to_dict`` shape) — from
                ``sts2_hint.adapter.translate`` on the DLL / live path, or
                straight from the engine in the oracle test.
        """
        import sts2_rs

        self._state = sts2_rs.state_from_dict(wire)
        self._parked = None
        self._crossed_shuffle = False
        self._last_wire = wire
        self._last_action = None
        self._last_result = None
        self._last_drift = []
        self._last_injection = None
        self._open_injection = None
        self._choice_in_flight = None
        self._take_damage_trace()
        self.stats.seeds += 1
        # The producer's FIRST wire of a fight lists enemies in spawn order
        # and later wires in C#'s slot-sorted order (the sort happens after
        # combat start). A mirror seeded from wire #1 cannot predict a
        # re-sort it never saw, so the first reconcile's roster-order diff is
        # noise by construction — 271 of the 582 v30wr desyncs, ~one per
        # fight. Graded from the second reconcile on.
        self._just_seeded = True

    def fork(self, label: str = "") -> CombatMirror | None:
        """A DETACHED copy of this fight, for work that must not touch it.

        Answering a ``card_select`` is destructive by design: the caller
        reconciles, replays the opening play, and injects the world's option
        list, leaving the mirror parked on a choice only *that* caller can
        resolve. A driver executes its own picks, so it resolves it. An
        ADVISOR does not execute anything, so it never can.

        Doing that work on a fork makes the answer path side-effect free.
        Without one it is not: an injection the engine cannot locate drops the
        mirror outright, and it does so on screens the advisor was always
        going to decline. Live run 001-ironclad-a8-0824-2239 shows the whole
        chain in three lines -- ``DROPLET_OF_PRECOGNITION is a potion, not a
        card`` (an honest decline), then ``injection dropped the mirror``,
        then ``no live mirror at the screen`` when the same screen was
        re-asked. A screen nobody could answer cost the fight its mirror.

        Carries the engine state and the small amount of bookkeeping an answer
        reads: the engine's own derived choice (``_parked``, which is what
        lets an injection resolve locators at all), the reshuffle blind spot,
        and the last wire. Stats are NOT shared -- the fork's work is not the
        fight's history, and counting a throwaway's reconciles as the live
        mirror's would corrupt every rate computed from them.

        Returns:
            The fork, or ``None`` when there is nothing to fork or the clone
            fails. Never raises: a fork is an optimisation over declining.
        """
        if self._state is None:
            return None
        try:
            twin = CombatMirror(
                self.cfg,
                policy=self.policy,
                label=label or f"{self.label}-fork",
                log_path=self.log_path,
            )
            twin._state = self._state.clone()
        except Exception:
            # A fork is an optimisation over declining, so a failure to make
            # one is not an event: the caller falls back exactly as it does
            # for a screen it cannot answer. This module has no logger by
            # design -- it reports through the divergence log, and a fork that
            # never existed has nothing to report.
            return None
        twin._parked = dict(self._parked) if self._parked is not None else None
        twin._crossed_shuffle = self._crossed_shuffle
        twin._last_wire = self._last_wire
        twin.tainted = self.tainted
        return twin

    def drop(self, reason: str = "") -> None:
        """Forget the current fight. Idempotent.

        Called at combat end, and by OBSERVE after a divergence (a mirror that
        has already been proven wrong must not keep answering). The dropped
        state is stashed for :meth:`_reseed_after_observe`, whose reseed can
        salvage the UNDISCLOSED hidden state (relic latches, history
        counters, per-card vars) the wire structurally cannot restore — the
        divergence proved the state's *prediction* wrong, not those.
        """
        if self._state is not None:
            self.stats.drops += 1
            self._last_dropped_state = self._state
            # The engine's own derived choice is hidden state too: it is set
            # only by `advance()` and no wire can re-derive it, so a reseed
            # that wipes it leaves the NEXT `inject_choice` with nothing to
            # resolve locators against — 25 of the 32 v34wr
            # `unlocatable_injection` records were a parked choice discarded
            # microseconds before it was needed. Stashed for
            # `_reseed_after_observe`'s salvage, same contract as the state.
            self._last_dropped_parked = self._parked
        self._state = None
        self._parked = None
        self._crossed_shuffle = False
        self._open_injection = None
        self._choice_in_flight = None
        self.drop_reason = reason

    # -- reconcile ---------------------------------------------------------

    def reconcile(self, wire: dict[str, Any]) -> list[Desync]:
        """Re-impose every observable from reality; return the DESYNC report.

        Seeds the fight when nothing is mirrored yet, so a driver can call
        this unconditionally at every combat decision.

        Args:
            wire: The world's current combat state, wire-schema.

        Returns:
            The deterministic observables the carried mirror predicted wrong,
            so an OBSERVE-mode caller sees them without re-reading the
            forensic log. Empty on a healthy mirror; never non-empty under
            STRICT, which raises instead of returning.

        Raises:
            MirrorDivergence: Under STRICT, on any DESYNC or on a
                transplant the shapes refuse (draw pile or enemy roster
                disagreeing on length).
        """
        import sts2_rs

        self.stats.decisions += 1
        if self._state is None:
            self.seed(wire)
            return []

        self.stats.reconciles += 1
        self.last_adopted = []
        carried = self._state

        # A card whose OnPlay opened a screen has the REST of itself suspended
        # in `pending_play` — AfterCardPlayed, replays, disposal. No wire
        # dialect expresses that (the parser refuses a dict that even claims
        # to), so REPLACING the state here silently deletes the tail and the
        # pick resolves into nothing.
        #
        # The plan's choice handling already says the mirror "is already
        # sitting at pending_choice() when the live game opens its screen" —
        # i.e. mid-play is precisely when a rebuild is the wrong move. So
        # verify against the parked mirror and keep it. Reality still wins on
        # every observable, because a mismatch is reported exactly as it would
        # be otherwise; what it does not do is throw away the half-executed
        # card in order to say so.
        #
        # Oracle Silent-16, traced: play -> [reconcile rebuilds, tail gone] ->
        # inject -> choose -> Nibbit block mirror 5 / world 1, with
        # `combat_targets` mirror 0 / world 1 (the world ran a tail that
        # targeted; the mirror ran nothing).
        # A mid-ENEMY-turn suspension (KnowledgeDemon's CURSE_OF_KNOWLEDGE
        # screen) must be kept for the same reason a mid-play card must: a
        # rebuild cannot represent the parked enemy-turn walk (the wire
        # schema has no `pending_*_resume`), so re-seeding here would delete
        # the rest of the enemy turn — the mirror would then sit one whole
        # turn behind after the choice resolves, which was the entire
        # `$.round`/`$.phase` divergence family on the v27wr sweep.
        curse_parked = (
            not carried.has_pending_play()
            and (cpc := carried.pending_choice()) is not None
            and str(cpc.get("kind")) == "monster_curse_select"
        )
        # Any OTHER suspended turn-transition walk (Well-Laid Plans' retain
        # select inside the end-of-turn flush, a Stampede auto-play screen,
        # suspended turn setup / draw) is unreconstructable for the same
        # reason: no `pending_*_resume` marker rides the wire, so a rebuild
        # deletes the rest of the transition and the mirror lands a full
        # round behind (55 of the 230 player-side v28wr events: mirror round
        # N-1, energy 0 vs the world's refilled 3 — plus run 1058's entire
        # 40-event one-turn-lag fight).
        walk_parked = not curse_parked and carried.has_suspended_walk()
        # A DRINK suspended on its own card-select. Same unexpressible-on-the-
        # wire class, and it was in neither predicate: C#
        # `PotionModel.OnUseWrapper` awaits `OnUse` — the Attack/Skill/Power/
        # Colorless choose-a-card screen included — before firing
        # `Hook.AfterPotionUsed`, so the engine parks the deferred dispatch on
        # `pending_potion_hook_resume`. Rebuilding here deleted BOTH the
        # deferred hook (ReptileTrinket's +3 temporary Strength, BeltBuckle)
        # and the parked choice itself, which is why a potion screen could not
        # be searched at all: the replay landed and the very next reconcile
        # threw the park away.
        #
        # Deliberately NOT folded into `walk_parked` below: a drink happens
        # inside the player's own play phase, which both sides agree on, so
        # `$.phase` / `$.side` stay compared. A turn-transition walk is the
        # only thing that earns that excusal.
        potion_parked = not curse_parked and not walk_parked and carried.has_pending_potion_hook()
        if carried.has_pending_play() or curse_parked or walk_parked or potion_parked:
            self.stats.parked_verifies += 1
            predicted = sts2_rs.state_to_dict(carried)
            # Round-trip the wire before diffing, exactly as the rebuild path
            # does. Comparing against the RAW wire diffs the engine dialect
            # against the producer's: `CLUMSY` vs `curse:Clumsy`, an intent's
            # `times` as None vs 1, a `cost_override` of None vs 'Slither'.
            # None of those are divergences, and every one of them fired as a
            # false STRICT crash the first time this ran against the DLL — the
            # oracle could never show it, because there both sides are already
            # `state_to_dict` output and share a dialect by construction.
            #
            # The reshuffle blind spot and the unreported-field skip apply here
            # for exactly the reasons they apply to the rebuild path below —
            # more so, in fact. A card that reshuffles the draw pile and THEN
            # opens a screen (draw-through-the-deck into a select) lands right
            # here, and the world rolled its own order for what was reshuffled,
            # so `$.deck.*` is unverifiable rather than wrong. Diffing it anyway
            # is a false STRICT crash on the one path that cannot recover by
            # re-seeding, because re-seeding is what would delete the suspended
            # card.
            #
            # `_crossed_shuffle` is deliberately NOT cleared on this path.
            # Clearing it is the rebuild path's privilege: it earns the right by
            # ADOPTING the world's piles. A parked verify keeps the mirror's
            # own, so if the two rolled different orders they are still
            # different afterwards, and the blind spot has to survive until a
            # rebuild actually resolves it.
            if self._crossed_shuffle:
                self.stats.reshuffle_skips += 1
            unreported = unreported_paths(wire)
            if unreported:
                self.stats.unreported_fields += len(unreported)
            if curse_parked or walk_parked:
                # Representation, not divergence: C# reports a blocking
                # mid-transition screen as a player-side play phase, while
                # the engine reports the suspended walk it actually is
                # (EnemyPhase / a parked turn boundary, same round). Both
                # name the same instant. `$.round` stays compared — the walk
                # must not lag a whole turn.
                unreported = unreported | {"$.phase", "$.side"}
            parked_desyncs = diff_observables(
                predicted,
                sts2_rs.state_to_dict(sts2_rs.state_from_dict(wire)),
                skip_prefix="$.deck." if self._crossed_shuffle else None,
                skip_paths=unreported,
            )
            # Same accounting as the rebuild path: the cursor is hidden state,
            # so a drift is recorded and attached to whatever report this
            # decision produces, never raised. Skipping it here would have made
            # the drift series silently under-count every mid-play decision.
            if isinstance(wire.get("rng"), dict) and "run_seed" in wire["rng"]:
                self._last_drift = _cursor_drift(predicted, wire)
                self.stats.cursor_drifts += len(self._last_drift)
            else:
                self._last_drift = []
            if parked_desyncs:
                self.stats.desyncs += 1
                self.stats.desync_fields += len(parked_desyncs)
                suspended_what = (
                    "the enemy turn was suspended on a curse screen"
                    if curse_parked
                    else "a card was suspended mid-play"
                )
                self._diverge(
                    f"{len(parked_desyncs)} deterministic observable(s) diverged "
                    f"while {suspended_what}: " + "; ".join(str(d) for d in parked_desyncs[:6]),
                    kind="desync",
                    wire=wire,
                    paths=[d.path for d in parked_desyncs],
                    mirror_wire=predicted,
                    extra={
                        "mid_play": not curse_parked,
                        "curse_parked": curse_parked,
                        "desyncs": [
                            {"path": d.path, "mirror": d.mirror, "world": d.world}
                            for d in parked_desyncs
                        ],
                    },
                )
                # Reseed, then report — the same two steps, in the same order,
                # as the rebuild path below. This used to be
                # `return self._reseed_after_observe(wire)`, which returned
                # that helper's `None` from a method annotated
                # `-> list[Desync]` AND threw the report away: every
                # divergence found while a card was suspended mid-play was
                # invisible to any caller that reads the return. Nothing did
                # until the advisor's mirror started counting desyncs per
                # decision, and then it crashed on `len(None)` instead
                # (live run 001-ironclad-a8-0824-2138, floor 5).
                self._reseed_after_observe(wire)
                return parked_desyncs
            self._last_wire = wire
            return []

        # -- world-in-flight skip -------------------------------------------
        # The LIVE producer can snapshot MID-CARD-RESOLUTION: SimCombatHandler.
        # PlayCardAndWaitAsync returns as soon as energy or hand size moves,
        # which is when the play STARTS (card leaves the hand for the
        # PlayPile, cost paid), not when it finishes — so the next serialized
        # board can show the played card still sitting in `play_pile` with
        # its effects (power gained, block added, the move to discard) not
        # yet landed. The mirror resolved the same play atomically, so at
        # this instant nothing on that board is gradable: every field the
        # in-flight card touches is transiently stale on the world's side,
        # not mispredicted (the play_pile+powers / discard+play_pile(+block)
        # family, 59 of the 296 live2 records, was exactly this — always one
        # `play` behind, world play_pile holding the card the mirror had
        # already discarded). Rebuilding from such a wire is worse than
        # skipping: it adopts a state whose play_pile card can never resume
        # (no `pending_play` rides the wire), which is how stuck play-pile
        # cards ended up haunting later reconciles after an OBSERVE reseed.
        # Defer instead: keep the carried mirror, count the skip, and grade
        # at the next stable wire — a real misprediction is still caught one
        # decision later.
        #
        # This cannot swallow a legitimately suspended play: when the mirror
        # itself holds the card in flight (a select screen it also opened),
        # `has_pending_play()` routed it to the parked verify above. The
        # oracle arm only emits a non-empty play_pile when genuinely parked
        # (same routing), and the DLL producer only at select decisions —
        # so rebuild-path reconciles there keep diffing exactly as before.
        if (wire.get("deck") or {}).get("play_pile"):
            self.stats.in_flight_skips += 1
            # `_crossed_shuffle` deliberately survives, same as the parked
            # path: only a rebuild that ADOPTS the world's piles may clear it.
            self._last_wire = wire
            self._last_drift = []
            return []

        fresh = sts2_rs.state_from_dict(wire)

        # Diff BEFORE transplanting. The transplant only moves hidden state,
        # which the diff ignores — and when it refuses (mismatched pile or
        # roster lengths) the field list is what actually explains why, so it
        # has to be in hand before the refusal is reported.
        predicted = sts2_rs.state_to_dict(carried)
        # The advance since the last reconcile may have consumed the shuffle
        # stream, in which case the world rolled its own order for whatever
        # was reshuffled and the PILE paths are UNVERIFIABLE, not wrong.
        # Nothing else is: a reshuffle does not make HP, block, energy,
        # powers, potions, orbs, phase or intents unknowable, and
        # skipping the whole diff (as this did originally) dropped 13.5% of
        # reconciles — 167/1235 on the 5-char x seeds-1-5 oracle sweep —
        # out of the DESYNC denominator for no reason. Scope the blind spot
        # to `$.deck.*` and keep measuring everything else.
        skip_prefix = "$.deck." if self._crossed_shuffle else None
        if self._crossed_shuffle:
            self.stats.reshuffle_skips += 1
        unreported = unreported_paths(wire)
        if unreported:
            self.stats.unreported_fields += len(unreported)
        if getattr(self, "_just_seeded", False):
            # See `seed`: wire #1 lists enemies in spawn order, later wires
            # in C#'s slot-sorted order — the first reconcile cannot grade a
            # re-sort the mirror never saw.
            unreported = unreported | {"$.encounter.roster"}
            self._just_seeded = False
        desyncs = diff_observables(
            predicted,
            sts2_rs.state_to_dict(fresh),
            skip_prefix=skip_prefix,
            skip_paths=unreported,
        )

        # -- choice-effect in-flight skip -----------------------------------
        # The play channel's mid-resolution snapshot has a twin on the CHOICE
        # channel. `SimCardSelector.GetSelectedCards` returns the moment the
        # pick arrives and nothing waits for the `CardSelectCmd` that awaited
        # it to finish, so the very next serialized board can show the screen
        # closed and its grant still in transit — the mirror holding the
        # picked card in hand (or at cost 0) against a world that has neither
        # yet. Nothing there is mispredicted; the world is one action-queue
        # tick behind on exactly that card.
        #
        # Deferring is not merely tidier than reporting it: under OBSERVE the
        # report RESEEDS from the half-applied board, which deletes the grant
        # from the mirror for good and manufactures the mirror-BEHIND echo one
        # decision later (16 of the 49 pure `$.deck.hand` records in
        # `mirror_live3` were that echo, never an independent fault). Keep the
        # carried mirror, count the skip, and grade at the next wire — where
        # the world has caught up, and a grant the world never actually made
        # still diverges, because nothing re-arms the flag.
        #
        # `_crossed_shuffle` deliberately survives, same as the play-pile
        # path: only a rebuild that ADOPTS the world's piles may clear it.
        pending_grant, self._choice_in_flight = self._choice_in_flight, None
        if _choice_effect_in_flight(desyncs, pending_grant or []):
            self.stats.choice_in_flight_skips += 1
            self._last_wire = wire
            self._last_drift = []
            return []

        # Who owns the RNG cursors. When the producer carries the run's own
        # dice, `fresh` already holds the WORLD's streams, and carrying the
        # mirror's over them would replace known truth with the mirror's own
        # drift — the very thing that makes an enemy intent unpredictable.
        world_rng = isinstance(wire.get("rng"), dict) and "run_seed" in wire["rng"]
        drift: list[Desync] = []
        if world_rng:
            # Recorded, not raised. The RNG cursor is hidden state, not one of
            # the deterministic observables STRICT is defined over, and the
            # honest reading of a drift is "the two engines consumed different
            # amounts of randomness" — which is a lead, not yet a proven
            # board-level divergence. It is attached to whatever report this
            # decision produces and counted separately so its size is
            # measurable before it is promoted to a crash.
            drift = _cursor_drift(predicted, wire)
            self.stats.cursor_drifts += len(drift)

        try:
            fresh.transplant_hidden(
                carried,
                _enemy_alignment(wire, predicted),
                carry_rng=not world_rng,
                carry_relic_counters=_undisclosed_relic_counters(wire),
                carry_power_counters=_undisclosed_power_counters(wire),
            )
        except Exception as exc:  # re-raised as a divergence below
            self._diverge(
                f"transplant refused the reconstruction: {exc}"
                + (f" — diverged on: {'; '.join(str(d) for d in desyncs[:6])}" if desyncs else ""),
                kind="transplant",
                wire=wire,
                paths=[d.path for d in desyncs] or ["$.transplant"],
                mirror_wire=predicted,
                extra={
                    "error": str(exc),
                    "desyncs": [
                        {"path": d.path, "mirror": d.mirror, "world": d.world} for d in desyncs
                    ],
                },
            )
            self._reseed_after_observe(wire)
            return desyncs or [Desync(path="$.transplant", mirror=None, world=str(exc))]

        self._state = fresh
        self._crossed_shuffle = False
        self._last_wire = wire
        self._last_drift = drift

        if desyncs and self.teammates and all(_teammate_reachable(d.path) for d in desyncs):
            # Co-op: every diff is on the enemies, which a teammate moves
            # between our decisions. `fresh` already holds the world's board
            # plus our carried hidden state, so keep it rather than drop it.
            self.stats.teammate_adoptions += 1
            self.last_adopted, desyncs = desyncs, []

        if desyncs:
            self.stats.desyncs += 1
            self.stats.desync_fields += len(desyncs)
            self._diverge(
                f"{len(desyncs)} deterministic observable(s) diverged: "
                + "; ".join(str(d) for d in desyncs[:6]),
                kind="desync",
                wire=wire,
                paths=[d.path for d in desyncs],
                mirror_wire=predicted,
                extra={
                    "desyncs": [
                        {"path": d.path, "mirror": d.mirror, "world": d.world} for d in desyncs
                    ]
                },
            )
            self._reseed_after_observe(wire)
        # Close the window whether or not it was reported: a clean decision's
        # rows must not leak into the next decision's evidence.
        self._take_damage_trace()
        return desyncs

    # -- damage-source forensics (GH #368.1) -------------------------------

    def _arm_damage_trace(self) -> None:
        """Arm the engine trace for the step that is about to run.

        Arming clears the engine-side buffer, so this must be paired with
        :meth:`_accumulate_damage_trace` — arming twice without a drain
        discards the first step's rows.
        """
        if not self._damage_trace_on:
            return
        import sts2_rs

        sts2_rs.set_damage_trace(True)

    def _accumulate_damage_trace(self) -> None:
        """Drain the engine buffer into this decision's window, then disarm.

        Disarming is the load-bearing half: everything that happens between
        two mirror steps — above all the driver's own MCTS search — runs on
        this thread and would otherwise pour rollout damage into the window.
        """
        if not self._damage_trace_on:
            return
        import sts2_rs

        self._last_damage_trace.extend(sts2_rs.take_damage_trace())
        self._damage_trace_dropped += int(sts2_rs.damage_trace_dropped())
        sts2_rs.set_damage_trace(False)

    def _take_damage_trace(self) -> tuple[list[dict[str, Any]], int]:
        """Hand over this decision's rows + overflow count, and reset the window."""
        rows, dropped = self._last_damage_trace, self._damage_trace_dropped
        self._last_damage_trace = []
        self._damage_trace_dropped = 0
        return rows, dropped

    def _reseed_after_observe(self, wire: dict[str, Any]) -> None:
        """Restart the mirror from reality after an OBSERVE-mode divergence.

        OBSERVE drops the mirror — a state proven wrong must not keep
        answering — but the fight is still going, and leaving the driver
        without one for the rest of it would turn a single divergence into a
        blind fight. Reality is right here in ``wire``, so seed from it and
        carry on; the run stays tagged via :attr:`tainted`.

        The bare reseed used to lose ALL carried hidden state, including the
        parts the wire structurally cannot disclose: undisclosed relic
        latches (ArtOfWar's attacks-played bit re-armed and over-granted +1
        energy on the v27wr sweep), the history counters (Spite /
        TearAsunder hit counts), and per-card vars. Those were not what the
        divergence proved wrong, and re-zeroing them manufactures the NEXT
        divergence — so salvage them from the dropped state. Enemy-side
        carries (intent history, AI instance state) stay dropped: a
        mispredicted board most often means the enemy model was the wrong
        part. A refused salvage leaves the bare reseed.

        No-op under STRICT, which raised instead of returning.
        """
        if self.alive:
            return

        import sts2_rs

        stale, self._last_dropped_state = self._last_dropped_state, None
        stale_parked, self._last_dropped_parked = self._last_dropped_parked, None
        self.seed(wire)
        if stale is None or self._state is None:
            return
        try:
            n_enemies = len(wire.get("encounter") or [])
            world_rng = isinstance(wire.get("rng"), dict) and "run_seed" in wire["rng"]
            self._state.transplant_hidden(
                stale,
                [None] * n_enemies,
                carry_rng=not world_rng,
                carry_relic_counters=_undisclosed_relic_counters(wire),
                carry_power_counters=_undisclosed_power_counters(wire),
            )
        except Exception:
            # The salvage is best-effort by definition; the reseed stands.
            pass
        # The parked choice belongs to the same salvage class: the divergence
        # proved the state's PREDICTION wrong, not the choice's kind/arity —
        # and only the kind lets the next `inject_choice` resolve locators
        # for a deterministic screen at all. The stale locators indexed the
        # dropped state, so re-locate each option against the freshly seeded
        # (i.e. the world's) piles by identity; an option that cannot be
        # found is dropped so `_resolve_locators` reports a genuine
        # `unlocatable_option` instead of a blanket refusal. The suspended
        # `pending_play` tail is still lost to the reseed — the pick applies
        # its primary effect but not the card's post-effect — which is
        # strictly better than applying nothing.
        if stale_parked is not None:
            # Against the mirror's OWN piles, not `wire`: the locators have to
            # index the state that will resolve them, and the two speak
            # different dialects (`DECAY` vs `curse:Decay`). Reading the state
            # also picks up whatever `transplant_hidden` just carried, which
            # the wire by definition does not describe.
            try:
                seeded = sts2_rs.state_to_dict(self._state)
            except Exception:
                # Forensics-grade fallback only: a wire-dialect match still
                # relocates every unprefixed card, which is most of them.
                seeded = wire
            self._parked = _relocate_parked(stale_parked, seeded)

    # -- advance -----------------------------------------------------------

    def advance(self, action: dict[str, Any]) -> str:
        """Commit one action the world already took.

        Args:
            action: A bridge-shape action dict (``{"action": "play",
                "card_index": …}``) — the same shape ``legal_actions``
                emits.

        Returns:
            The engine's ``StepResult`` as a lowercase string: ``advanced`` |
            ``pending_choice`` | ``combat_won`` | ``combat_lost`` |
            ``invalid_target``.

        Raises:
            MirrorDivergence: Under STRICT, when the engine reports
                ``invalid_target`` — the world and the engine disagreeing
                about what is legal is a divergence, not a recoverable
                condition.
        """
        state = self.state
        # Any committed action ends the previous choice's grace period: the
        # deferral in `reconcile` covers the ONE board the world serializes
        # between a pick and its grant, not everything that follows.
        # `advance_through_choice` re-arms it after its own advance.
        self._choice_in_flight = None
        before = state.shuffle_stream_counter()
        # GH #368.1: the trace is armed ONLY across the mirror's own step.
        # The driver runs MCTS on this same thread between decisions, and its
        # rollouts deal damage through the same pipeline — leaving the trace
        # armed across a search would bury the handful of rows that matter
        # under tens of thousands of hypothetical ones.
        self._arm_damage_trace()
        result = state.apply_step(action)
        self._accumulate_damage_trace()
        if state.shuffle_stream_counter() != before:
            self._crossed_shuffle = True
        self._last_action = action
        self._last_result = result

        if result == "invalid_target":
            self._diverge(
                f"engine rejected an action the world committed: {action!r}",
                kind="invalid_action",
                wire=self._last_wire,
                paths=["$.action"],
                extra={"action": action},
            )
            return result

        self._parked = state.pending_choice() if result == "pending_choice" else None
        return result

    # -- choices -----------------------------------------------------------

    def parked_choice(self) -> dict[str, Any] | None:
        """The choice the engine derived on its own, or ``None``.

        Captured when an :meth:`advance` parked on a pending choice, and kept
        across the following :meth:`reconcile` (which rebuilds the state and
        therefore wipes the live choice). Cleared by :meth:`inject_choice`,
        which is what grades it.
        """
        return self._parked

    def inject_choice(
        self,
        kind: str,
        options: Sequence[dict[str, Any]],
        min_select: int,
        max_select: int,
        *,
        can_skip: bool = False,
        origin: dict[str, Any] | None = None,
    ) -> list[Desync]:
        """Open the choice the WORLD is showing, and grade the engine's own.

        This is the single path for every mid-combat selection — there is no
        "trust the engine's derivation" branch. Removing option-matching from
        the critical path is what retires the three-tier fallback chain: the
        engine's derivation becomes an assertion instead of a data source.

        Grading is by kind. A *deterministic* kind's options are concrete
        cards out of the player's own piles, so engine and world must derive
        the same list and a mismatch is a divergence. A *generative* kind
        (``add_generated_*``) rolled its own content, so a mismatch says
        nothing about fidelity and is not graded
        (``sts2_rs.choice_kind_is_generative``).

        Args:
            kind: ``ChoiceKind`` python name (``put_back``, ``card_select``, …).
            options: ``{"id", "upgraded", "pile", "index"}`` dicts. ``pile`` +
                ``index`` locate a concrete card and must be given together;
                omitting both makes a synthetic option, which is only
                meaningful for generative kinds. This is exactly the shape
                ``CombatState.pending_choice()`` emits.

                A real driver rarely has locators: a ``card_select`` message
                names cards, not board positions. Options without one are
                RESOLVED against the engine's own parked choice by
                :meth:`_resolve_locators` — which is also where a live list
                the engine cannot account for becomes a divergence.
            min_select / max_select: Selection arity from the world.
            can_skip: Whether the world offers a skip.
            origin: The world's own description of the screen — the
                ``prompt_key`` / ``source`` the C# ``CardSelectCmd`` context
                patch captured. Purely forensic: it is what names the culprit
                when the engine parked no choice of its own, since an option
                list of three generated cards says what was offered but not
                which effect offered it.

        Returns:
            The graded differences against the engine's own derivation, so an
            OBSERVE-mode caller sees them without re-reading the forensic log.
            Empty for a generative kind and for a matching derivation; never
            non-empty under STRICT, which raises instead.

        Raises:
            MirrorDivergence: Under STRICT, when a deterministic kind's
                engine-derived option list differs from the world's, or when
                the engine refuses the injected options.
        """
        import sts2_rs

        state = self.state
        parked, self._parked = self._parked, None
        # A screen that never opens leaves nothing to answer: clear eagerly so
        # a refused injection cannot hand its options to the next
        # `advance_through_choice`.
        self._open_injection = None
        options = [dict(o) for o in options]

        generative = sts2_rs.choice_kind_is_generative(kind)
        graded: list[Desync] = []
        if parked is not None and not generative:
            if not self._resolve_locators(options, parked, kind):
                return []
            graded = _grade_choice(parked, kind, options, min_select, max_select)
        elif not generative and any(
            o.get("pile") is None or o.get("index") is None for o in options
        ):
            # No parked choice to locate against, and the driver only knows
            # card NAMES — so these options point at no board position.
            # Injecting them anyway is the one thing this class must not do:
            # every deterministic resolver bails when `selected_card_uid`
            # comes back `None` (`resolve_upgrade_in_hand`, lib.rs:8150),
            # and `ChoiceKind::CardSelect` — the kind a driver reaches for
            # when it has nothing better — is an explicit no-op placeholder
            # arm in the engine, not a working catch-all. The world would
            # upgrade/exhaust/discard a card, the mirror would close its
            # choice having applied NOTHING, no grading would run (grading
            # needs `parked`), and nothing would raise. A silent, permanent
            # desync dressed as success.
            #
            # Reachable whenever the mirror was re-seeded rather than parked
            # (after an OBSERVE drop, after a shadow-answered decision), so
            # it is not a theoretical path.
            self._diverge(
                f"cannot answer a deterministic {kind}: the world's options carry no "
                "locator and the engine has no parked choice to resolve them against "
                "— injecting would resolve to a silent no-op",
                kind="unlocatable_injection",
                wire=self._last_wire,
                paths=["$.choice.injection"],
                extra={
                    "choice_kind": kind,
                    "options": options,
                    "engine_derived": parked,
                },
            )
            return []

        try:
            # `replace_parked` when the mirror is sitting on its own derived
            # list: the world's options win, and — critically — clearing the
            # choice this way leaves `pending_play` intact, which is the whole
            # reason the mirror was left parked instead of rebuilt.
            state.open_injected_choice(
                kind,
                options,
                min_select,
                max_select,
                can_skip=can_skip,
                replace_parked=parked is not None,
            )
        except Exception as exc:  # re-raised as a divergence below
            self._diverge(
                f"engine refused the world's option list: {exc}",
                kind="injection",
                wire=self._last_wire,
                paths=["$.choice.injection"],
                extra={
                    "choice_kind": kind,
                    "options": options,
                    "engine_derived": parked,
                    "error": str(exc),
                },
            )
            return []

        self.stats.injections += 1
        # The whole screen, as the mirror actually opened it: the world's
        # options with the board positions `_resolve_locators` pinned them to,
        # and the engine's own derivation beside them. A divergence one advance
        # later says "these cards are in the wrong pile" and nothing about the
        # pick that put them there, which is unarguable and unusable — this is
        # what makes a choice-adjacent divergence self-explaining.
        self._last_injection = {
            "kind": kind,
            "options": options,
            "min_select": min_select,
            "max_select": max_select,
            "engine_derived": (parked or {}).get("cards"),
            "picks": None,
        }
        self._open_injection = self._last_injection
        if parked is None:
            self.stats.unparked_injections += 1
            self._last_unparked = {
                "kind": kind,
                "options": [o.get("id") for o in options],
                "min_select": min_select,
                "max_select": max_select,
                "origin": origin,
                # What the engine was doing when the world opened this screen.
                # `origin` names the C# entry point, but several effects share
                # one (FromChooseACardScreen serves Discovery, four potions,
                # Toolbox and three relics), so the culprit is only pinned
                # once you also know which action the mirror had just run and
                # whether the engine parked anything at all for it.
                "after_action": self._last_action,
                "after_result": self._last_result,
            }
        else:
            self._last_unparked = None
        if graded:
            # NOT `stats.desyncs` — that is per-reconcile and this is
            # per-injection; see `MirrorStats.choice_mismatches`.
            self.stats.choice_mismatches += 1
            self.stats.desync_fields += len(graded)
            self._diverge(
                f"engine derived a different {kind} than the world showed: "
                + "; ".join(str(d) for d in graded[:6]),
                kind="choice_mismatch",
                wire=self._last_wire,
                paths=[d.path for d in graded],
                extra={
                    "choice_kind": kind,
                    "options": options,
                    "engine_derived": parked,
                    "desyncs": [
                        {"path": d.path, "mirror": d.mirror, "world": d.world} for d in graded
                    ],
                },
            )
        return graded

    def arm_turn_start_resume(self, after_relic: str) -> bool:
        """Park the rest of turn setup behind the OPEN turn-start screen.

        C# ``Hook.AfterPlayerTurnStart`` *awaits* the ``CardSelectCmd`` a relic
        opens from it, so the DLL's board at such a decision is a mid-setup
        snapshot: the hand is drawn and energy is reset, but every later
        listener on that pass, the whole ``AfterPlayerTurnStartLate`` pass and
        the outer loop's ``Hook.AfterSideTurnStart`` still have to run. A mirror
        seeded from that wire and handed a free-standing injected choice never
        ran any of it.

        Only Gambling Chip's round-1 mulligan reaches this today; the engine
        parks the identical marker when its own copy of the relic raises the
        screen.

        Args:
            after_relic: Relic id (``"GAMBLING_CHIP"``) whose hook the world is
                blocked inside.

        Returns:
            True when the marker was armed. False when the engine declined —
            no choice open, or the relic is not in the mirror's own inventory —
            in which case nothing is parked and the resolve behaves as before.

        Raises:
            RuntimeError: The installed ``sts2_rs`` wheel predates this surface.
                Deliberately loud: degrading to a no-op would silently restore
                the very desync this exists to close, on a run that looks fine.
        """
        if self._state is None:
            return False
        arm = getattr(self._state, "arm_turn_start_resume", None)
        if arm is None:
            raise RuntimeError(
                "CombatMirror.arm_turn_start_resume: the installed sts2_rs wheel has no "
                "CombatState.arm_turn_start_resume — rebuild it "
                "(scripts/wheel-venv.sh build) before sweeping, or the Gambling-Chip "
                "turn-1 hook batch silently goes missing from the mirror again."
            )
        return bool(arm(after_relic))

    def arm_turn_start_resume_after_power(self, after_power: str) -> bool:
        """Park the rest of turn setup behind a POWER's turn-start screen.

        The any-turn twin of :meth:`arm_turn_start_resume` (GH #355). C#
        ``Hook.AfterPlayerTurnStart`` awaits the ``CardSelectCmd`` that
        ``TyrannyPower`` / ``ToolsOfTheTradePower`` open, and unlike the two
        relics with the same shape neither carries a turn gate — so the
        mid-setup snapshot this repairs is reachable on ANY turn, which is
        what made the relic-only fix a partial one.

        Args:
            after_power: Power id (``"TYRANNY"``) whose hook the world is
                blocked inside.

        Returns:
            True when the marker was armed. False when the engine declined —
            no choice open, or the mirror's player does not carry the power.

        Raises:
            RuntimeError: The installed ``sts2_rs`` wheel predates this
                surface. Loud for the same reason as the relic twin: a no-op
                degradation silently restores the desync on a run that looks
                fine.
        """
        if self._state is None:
            return False
        arm = getattr(self._state, "arm_turn_start_resume_after_power", None)
        if arm is None:
            raise RuntimeError(
                "CombatMirror.arm_turn_start_resume_after_power: the installed sts2_rs "
                "wheel has no CombatState.arm_turn_start_resume_after_power — rebuild it "
                "(scripts/wheel-venv.sh build) before sweeping, or the Tyranny / Tools of "
                "the Trade turn-start batch silently goes missing from the mirror again."
            )
        return bool(arm(after_power))

    def arm_turn_start_resume_for(self, block: TurnStartBlock) -> bool:
        """Arm whichever resume :class:`TurnStartBlock` names.

        The two surfaces are not interchangeable — see
        :class:`TurnStartBlock` — so the dispatch lives here rather than at
        each driver, which is what kept the relic case from drifting between
        ``headless_eval`` and ``realgame_mcts`` in the first place.
        """
        if block.source == "power":
            return self.arm_turn_start_resume_after_power(block.id)
        return self.arm_turn_start_resume(block.id)

    def _resolve_locators(
        self, options: list[dict[str, Any]], parked: dict[str, Any], kind: str
    ) -> bool:
        """Fill in each option's ``pile``/``index`` from the engine's own choice.

        A ``card_select`` message names cards; the engine names board
        positions. Matching is one-to-one on ``(id, upgraded)`` in the order
        the world listed them, so two Strikes in the same pile consume two
        distinct engine options rather than both resolving to the first.

        Deliberately NOT a name-matching heuristic with a fallback. If the
        engine's derivation cannot account for every card the world is showing,
        the two are looking at different boards, and that is the divergence the
        whole mirror exists to catch — so it is reported, not patched over.

        Returns:
            True when every option is located (or already was). False when the
            mirror diverged, which under STRICT never returns at all.
        """
        available = list(parked.get("cards") or [])
        used: set[int] = set()
        for i, opt in enumerate(options):
            if opt.get("pile") is not None and opt.get("index") is not None:
                continue
            key = (opt.get("id"), bool(opt.get("upgraded")))
            match = next(
                (
                    j
                    for j, c in enumerate(available)
                    if j not in used
                    and (c.get("id"), bool(c.get("upgraded"))) == key
                    and c.get("pile") is not None
                ),
                None,
            )
            if match is None:
                self._diverge(
                    f"the world offers {key[0]!r} (upgraded={key[1]}) as {kind} "
                    f"option {i}, which the engine's own choice does not contain",
                    kind="unlocatable_option",
                    wire=self._last_wire,
                    paths=["$.choice.options"],
                    extra={
                        "choice_kind": kind,
                        "options": options,
                        "engine_derived": parked,
                    },
                )
                return False
            used.add(match)
            opt["pile"] = available[match]["pile"]
            opt["index"] = available[match]["index"]
        return True

    def advance_through_choice(self, selection: Sequence[int]) -> str:
        """Answer the open choice and confirm it closed.

        A world that picks fewer than ``max_select`` is finalized here rather
        than left mid-accumulation — see the inline note. Anything still open
        after that is a genuine arity disagreement.

        Args:
            selection: Option indices the world picked. Empty means skip.

        Returns:
            The engine's ``StepResult`` string from the final step.

        Raises:
            MirrorDivergence: Under STRICT, when no choice is open (the world
                showed a selection the engine never opened), or when the
                choice survives both the picks and the finalize.
        """
        state = self.state
        # Consumed here whether or not the answer lands: an injection is
        # answered exactly once.
        injected, self._open_injection = self._open_injection, None
        open_choice = state.pending_choice()
        if open_choice is None:
            self._diverge(
                "advance_through_choice with no choice open",
                kind="no_choice",
                wire=self._last_wire,
                paths=["$.choice.absent"],
                extra={"selection": list(selection)},
            )
            return "advanced"

        picks = [int(i) for i in selection]
        if self._last_injection is not None:
            self._last_injection["picks"] = picks
        result = self.advance({"action": "choose", "selection": picks})
        # A partial multi-pick ACCUMULATES: `resolve_choice_action` records the
        # index and re-opens the same choice, firing no effects, waiting for
        # either the next pick or a finalize. The world resolved in one shot,
        # so finalize here to land on the byte-identical end state the engine
        # documents that path to produce.
        if (
            self._state is not None
            and self._state.pending_choice() is not None
            and int(open_choice.get("max_select", 1)) > 1
            and len(picks) < int(open_choice["max_select"])
        ):
            result = self.advance({"action": "choose", "selection": []})
        if self._state is not None and self._state.pending_choice() is not None:
            # A choice that survives its own answer means the engine expects
            # more picks than the world made — the two disagree on arity.
            self._diverge(
                f"choice still open after selection {list(selection)}",
                kind="unclean_choice",
                wire=self._last_wire,
                paths=["$.choice.unclean"],
                extra={
                    "selection": list(selection),
                    "still_open": self._state.pending_choice(),
                },
            )
        # Arm the one-board grace period for the grant this pick just made —
        # the live producer can serialize the next combat decision before it
        # lands. `advance` cleared the flag, so this is the only writer, and
        # only a pick that named real options can arm it.
        opts = (injected or {}).get("options") or []
        named = [opts[i] for i in picks if 0 <= i < len(opts)]
        self._choice_in_flight = named or None
        return result

    # -- search ------------------------------------------------------------

    def search(
        self, cfg: RecommendConfig | None = None, *, seed: int | None = None
    ) -> dict[str, Any]:
        """Run one honest MCTS search on a CLONE of the mirror.

        The only place this module launches a search. Three invariants live
        here and nowhere else:

        * it searches a clone, so no search can ever mutate ground truth;
        * it refuses a clairvoyant profile. The mirror is the one object that
          would make clairvoyance actually work — the true draw order and the
          true RNG cursors are sitting right there — which makes it the one
          object that must not be allowed to use them;
        * it refuses to search a mirror that is itself already on the BELIEF
          representation. ``deferred_draw_order`` is a claim that the pile is
          a canonical belief rather than a materialized order; the mirror
          holds concrete truth, so only the private clone the search builds
          may ever carry it. A live mirror with the flag set means a belief
          state leaked back into ground truth, and searching it would quietly
          change how the pile is consumed.

        Args:
            cfg: Overrides the mirror's default profile for this call.
            seed: Overrides ``cfg.seed`` (drivers vary it per decision).

        Returns:
            The raw ``sts2_rs.mcts_recommend`` result dict.

        Raises:
            ValueError: On a clairvoyant ``hidden_info``.
            MirrorDivergence: When the live mirror is already on the belief
                representation.
        """
        import sts2_rs

        knowledge = self.state.draw_pile_knowledge()
        if knowledge.get("deferred_draw_order"):
            self._diverge(
                "the live mirror is on the belief draw representation "
                "(deferred_draw_order=True) — only the private search clone may be",
                kind="belief_leak",
                wire=self._last_wire,
                paths=["$.deck.deferred_draw_order"],
                extra={"draw_pile_knowledge": dict(knowledge)},
            )
            return {"actions": [], "pv": [], "sims": 0}

        cfg = cfg or self.cfg
        if seed is not None:
            cfg = replace(cfg, seed=seed)
        if cfg.hidden_info in _CLAIRVOYANT_PROFILES:
            raise ValueError(
                "CombatMirror.search: refusing hidden_info="
                f"{cfg.hidden_info!r}. The mirror holds the TRUE draw order and "
                "RNG cursors, so a clairvoyant root would plan against cards the "
                "player cannot see. Use 'beliefdraw' (the deployed honest "
                "profile), 'drawchance', or 'rootscramble'."
            )

        return self._recommend(self.state, cfg)

    def _recommend(self, state: Any, cfg: RecommendConfig) -> dict[str, Any]:
        """One `mcts_recommend` over a clone of `state`, counted in `stats`.

        Split out of :meth:`search` so :meth:`search_choice_picks` can search a
        PRIVATE working state (the mirror's own state plus the picks made so
        far this decision) without duplicating the call or the accounting.
        The guards stay in :meth:`search`: they are properties of the LIVE
        mirror, and the working state is a clone of one already checked.
        """
        import sts2_rs

        self.stats.searches += 1
        return sts2_rs.mcts_recommend(
            state.clone(),
            sims=cfg.sims,
            c_puct=cfg.resolved_c_puct(),
            depth=cfg.depth,
            seed=cfg.seed,
            threads=cfg.threads,
            fair=cfg.fair,
            rollout_policy=cfg.rollout_policy,
            hidden_info=cfg.hidden_info,
            draw_widening_c=cfg.resolved_draw_widening_c(),
            endturn_widening_c=cfg.endturn_widening_c,
            endturn_widening_alpha=cfg.endturn_widening_alpha,
            common_random_numbers=cfg.common_random_numbers,
            win_hp_carry=cfg.win_hp_carry,
            win_potion_carry=cfg.win_potion_carry,
        )

    def search_choice_picks(
        self,
        cfg: RecommendConfig | None = None,
        *,
        hand: list[dict[str, Any]],
        n_options: int,
        min_select: int,
        max_select: int,
    ) -> list[int]:
        """Answer the open choice — ONE SEARCH PER PICK on a multi-pick screen.

        Shared by both mirrored ``card_select`` drivers
        (``headless_eval.pick_card_select_mirrored`` and
        ``realgame_mcts.pick_card_select_mirrored``).

        A single-pick screen (``max_select <= 1``) is one search, unchanged.

        A MULTI-pick screen used to be answered by taking the top
        ``max_select`` rows off that same single search. But the engine
        decomposes a multi-pick SEQUENTIALLY (bead ``sts2-rs-sopgf``): each
        incremental ``ResolveChoice`` appends to ``pc.picked``, fires no
        effect, and RE-OPENS the same choice, so the root the search scored
        only ever answered "which card first". Its second row is the runner-up
        to that question, not the best follow-up to the pick actually made —
        and for a discard-2 or an exhaust-2 the two picks interact by
        construction. This walks the engine's own decomposition instead: pick,
        apply, re-search, repeat.

        The walk runs on a PRIVATE clone. The mirror is not advanced here —
        the caller still applies the picks only after the world accepts them
        (:meth:`advance_through_choice`), so a rejected selection cannot leave
        ground truth half-answered.

        Args:
            cfg: Overrides the mirror's default profile for this call.
            hand: The wire's ``deck.hand`` — the stable-idx frame the tags are
                read in. Incremental picks fire no effects, so one hand stays
                valid across the whole walk.
            n_options: How many options the world's screen shows.
            min_select / max_select: The screen's (already clamped) arity.

        Returns:
            Distinct option indices, in pick order. Empty means skip.
        """
        from sts2_hint.recommender import (
            FINALIZE,
            choice_picks_from_tags,
            next_choice_pick_from_tags,
            tags_from_result,
        )

        if max_select <= 1:
            return choice_picks_from_tags(
                tags_from_result(self.search(cfg), hand),
                n_options=n_options,
                min_select=min_select,
                max_select=max_select,
            )

        # The guards belong to the live mirror; run them once, then walk.
        first = self.search(cfg)
        cfg = cfg or self.cfg
        working = self.state.clone()
        picks: list[int] = []
        result = first
        while len(picks) < max_select:
            tags = tags_from_result(result, hand)
            nxt = next_choice_pick_from_tags(
                tags,
                n_options=n_options,
                taken=picks,
                allow_finalize=len(picks) >= min_select,
            )
            if nxt is FINALIZE or nxt is None:
                break
            picks.append(int(nxt))
            if len(picks) >= max_select:
                break
            # Apply the pick to the PRIVATE state so the next search is
            # conditioned on it, then confirm the engine re-opened the choice.
            # It auto-finalizes on the pick that reaches `max_select`, and a
            # closed choice means there is nothing left to search.
            working.apply_step({"action": "choose", "selection": [int(nxt)]})
            if working.pending_choice() is None:
                break
            result = self._recommend(working, cfg)

        if len(picks) < min_select:
            picks.extend(i for i in range(n_options) if i not in picks)
            del picks[min_select:]
        return picks

    # -- divergence --------------------------------------------------------

    def _diverge(
        self,
        message: str,
        *,
        kind: str,
        wire: dict[str, Any] | None,
        paths: list[str],
        mirror_wire: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Record a divergence, then raise (STRICT) or drop the mirror (OBSERVE).

        ``mirror_wire`` must be the mirror's **prediction** — the state as it
        was before reality replaced it. Serializing ``self._state`` instead
        would report the reconstruction on both sides of the record and make
        every desync look like a diff of a dict against itself.
        """
        import sts2_rs

        report: dict[str, Any] = {
            "ts": time.time(),
            "label": self.label,
            "kind": kind,
            "message": message,
            "policy": self.policy.value,
            "stats": self.stats.as_dict(),
            "last_action": self._last_action,
            "world_wire": wire,
        }
        # GH #368.1: every HP-loss event the mirror produced since the last
        # reconcile, with the dispatcher that produced it. Drained here rather
        # than at the diff, so a divergence raised from `step` (an action the
        # engine refuses) carries its window too.
        trace, trace_dropped = self._take_damage_trace()
        if trace:
            report["damage_trace"] = trace
            if trace_dropped:
                # Never let a truncated trace read as a complete one.
                report["damage_trace_dropped"] = trace_dropped
        if self._last_injection is not None:
            report["last_injection"] = self._last_injection
        if self._last_drift:
            report["cursor_drift"] = [
                {"stream": d.path, "mirror": d.mirror, "world": d.world} for d in self._last_drift
            ]
        if self._last_unparked is not None:
            # The most recent screen the engine had not opened itself. Its
            # `card_select` catch-all resolver is a no-op, so if the world
            # granted anything there the mirror never received it — which is
            # the first thing to check when a deck path diverges.
            report["last_unparked_choice"] = self._last_unparked
        if mirror_wire is None and self._state is not None:
            try:
                mirror_wire = sts2_rs.state_to_dict(self._state)
            except Exception as exc:  # forensics must never mask the divergence
                report["mirror_wire_error"] = str(exc)
        report["mirror_wire"] = mirror_wire
        report["paths"] = paths
        report.update(extra or {})
        _append_jsonl(self.log_path, report)
        if self.on_divergence is not None:
            self.on_divergence(paths)

        if self.policy is MirrorPolicy.STRICT:
            raise MirrorDivergence(f"[mirror {self.label}] {message}", report)
        self.tainted = True
        self.drop(reason=message)


def _grade_choice(
    parked: dict[str, Any],
    kind: str,
    options: Sequence[dict[str, Any]],
    min_select: int,
    max_select: int,
) -> list[Desync]:
    """Diff the engine's own derived choice against the world's.

    Compares kind, arity, and the option *multiset* by ``(id, upgraded, pile,
    index)`` — a concrete board position, not a card name, because names repeat
    (two Strikes in one pile) and positions don't. Locators absent on both
    sides (synthetic options) compare as ``None`` and so only match each other.

    A multiset, not a list, because the mirror opens the choice in the
    **world's** order and answers by index into it: the engine listing the same
    cards in a different order changes nothing about what gets picked, and
    flagging it would be a false positive. What is still a real signal — the
    engine offering a card the world does not, or a different number of them —
    survives the sort.
    """

    def key(o: dict[str, Any]) -> tuple[Any, ...]:
        return (o.get("id"), bool(o.get("upgraded")), o.get("pile"), o.get("index"))

    out: list[Desync] = []
    if parked.get("kind") != kind:
        out.append(Desync("$.choice.kind", parked.get("kind"), kind))
    if int(parked.get("min_select", 0)) != int(min_select):
        out.append(Desync("$.choice.min_select", parked.get("min_select"), min_select))
    if int(parked.get("max_select", 0)) != int(max_select):
        out.append(Desync("$.choice.max_select", parked.get("max_select"), max_select))
    mine = sorted(key(c) for c in parked.get("cards") or [])
    theirs = sorted(key(o) for o in options)
    if mine != theirs:
        out.append(Desync("$.choice.options", mine, theirs))
    return out


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one forensic record. Best-effort: never masks the divergence."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except OSError:
        # A full disk must not turn a legible divergence into an IOError
        # traceback that hides it.
        pass


#: Generative screens roll their own content, so their options exist in NO
#: pile and can never be located — but the engine's generative kinds need no
#: locators at all. The world names the effect (``source_model``,
#: RunSimulator.cs / SpireMentor MainFile.cs CardSelectCmd context patch); trust
#: it first, and fall back to the shape (no option locatable anywhere) when
#: the patch didn't fire.
#:
#: **This table is currently unreachable for every entry, in BOTH world
#: drivers**, and the fallback below is what actually classifies generative
#: screens. All of these open via ``CardSelectCmd.FromChooseACardScreen``,
#: whose v0.107.1 signature is ``(PlayerChoiceContext, IReadOnlyList<CardModel>,
#: Player, bool canSkip)`` (``CardSelectCmd.cs:216``) — it declares no
#: ``AbstractModel source`` parameter (only ``FromHand`` /
#: ``FromHandForDiscard`` / ``FromHandForUpgrade`` do), and ``Player`` is not
#: an ``AbstractModel``, so the capture patch has nothing to name. Kept, and
#: kept COMPLETE, because it is the correct classification the moment the game
#: or the patch supplies a name — and because the completeness is what the
#: audit checks.
#:
#: Free vs non-free is the whole point of the split: ``add_generated_to_hand``
#: makes the search pay for the card, ``add_generated_free_to_hand`` does not.
#: The two non-free grants are Toolbox (``Toolbox.cs:28`` — no
#: ``SetToFreeThisTurn``) and Quasar (``Quasar.cs:32``,
#: ``AddGeneratedCardToCombat(.., Hand)``); ``Discovery.cs:26``, ``Splash.cs:53``
#: and all four potions set it.
_GENERATIVE_SOURCES = {
    "Toolbox": "add_generated_to_hand",
    "Quasar": "add_generated_to_hand",
    "Discovery": "add_generated_free_to_hand",
    "Splash": "add_generated_free_to_hand",
    "AttackPotion": "add_generated_free_to_hand",
    "SkillPotion": "add_generated_free_to_hand",
    "PowerPotion": "add_generated_free_to_hand",
    "ColorlessPotion": "add_generated_free_to_hand",
}


class TurnStartBlock(NamedTuple):
    """What the world's turn-start hook batch is currently blocked inside.

    ``source`` is ``"relic"`` or ``"power"``; ``id`` is that model's python id
    (``"GAMBLING_CHIP"`` / ``"TYRANNY"``). Fed to
    :meth:`CombatMirror.arm_turn_start_resume_for`, which picks the matching
    engine surface — the two are NOT interchangeable: a relic block resumes
    the relic ``AfterPlayerTurnStart`` bus, a power block resumes the power
    cluster walk that runs BEFORE it.
    """

    source: str
    id: str


#: C# powers whose ``AfterPlayerTurnStart`` **awaits** a card select, keyed by
#: the ``source_model`` the world names, mapped to the engine ``ChoiceKind``
#: the screen really is plus the power's python id. GH #355.
#:
#: This is the "any turn" half of the Gambling-Chip fix. Both v0.107.1 relics
#: with a turn-start select are hard-gated to turn 1 (``GamblingChip.cs:18``
#: ``TurnNumber <= 1``, ``ChoicesParadox.cs:29`` ``TurnNumber != 1``), so the
#: relic surface can only ever fire on the first turn. These three powers have
#: no turn gate at all, so a mirror seeded from one of their screens is the
#: same mid-setup snapshot on turn 7 as Gambling Chip's was on turn 1.
#:
#: ``EntropyPower`` is deliberately ABSENT even though it is the third such
#: power (``EntropyPower.cs:21``, ``FromHand(TransformSelectionPrompt,
#: Amount)``): the engine has no in-hand transform ChoiceKind — its cluster
#: arm transforms ``hand.iter().take(amount)`` outright with no screen — so
#: there is no kind to map the injection to, and guessing one would resolve
#: the pick against the wrong resolver. Recognizing it needs the engine-side
#: select first; until then an Entropy screen keeps the refusal path, which is
#: where it already was.
_TURN_START_POWER_SOURCES: dict[str, tuple[str, str]] = {
    # source_model -> (engine ChoiceKind python name, power python id)
    "TyrannyPower": ("exhaust_from_hand", "TYRANNY"),
    "ToolsOfTheTradePower": ("discard_from_hand", "TOOLS_OF_THE_TRADE"),
}


def _locate_options_in_hand(options: list[dict[str, Any]], board: dict[str, Any]) -> bool:
    """Stamp each option with its ``(pile, index)`` position in the HAND.

    The engine names board positions; a ``card_select`` message names cards.
    When the engine parked no choice of its own there is nothing to resolve
    against (:meth:`CombatMirror._resolve_locators` is the parked path), so a
    screen that is known to be drawn from the hand locates itself here.

    Matching is one-to-one in hand order, so two identical Strikes consume two
    DISTINCT hand indices rather than both pointing at the first — an option
    list that resolved to one position would make the search rank a pick it
    cannot then perform.

    MUTATES ``options`` in place. Returns True when every option found a
    distinct hand slot; False leaves the partial stamping alone and tells the
    caller to keep the refusal path — an option the hand cannot account for
    means the screen is not the hand screen it was taken for.
    """
    from sts2_hint import adapter

    hand = [
        (adapter.registry_card_id(str(c.get("id")), "$.hand"), bool(c.get("upgraded")))
        for c in (board.get("hand") or [])
    ]
    used: set[int] = set()
    ok = True
    for o in options:
        want = (str(o["id"]), bool(o["upgraded"]))
        for i, hc in enumerate(hand):
            if i not in used and hc == want:
                o["pile"], o["index"] = "hand", i
                used.add(i)
                break
        else:
            ok = False
    return ok


def skip_arity_from_world(
    *, cancelable: Any, source: str | None, min_select: int, max_select: int
) -> tuple[bool, int]:
    """Reconcile a screen's skippability with its reported minimum arity.

    Shared by both world drivers so the two cannot disagree.

    The legacy rule was ``can_skip = min_select <= 0``, which is sound for
    every ``CardSelectorPrefs``-carrying entry point — there ``MinSelect`` is
    the real arity the screen enforces — but NOT for
    ``CardSelectCmd.FromChooseACardScreen``. That overload hard-codes its call
    into the selector as ``GetSelectedCards(cards, 0, 1)``
    (``CardSelectCmd.cs:230``): the 0 is a constant, not the screen's minimum,
    and the actual skippability lives in the separate ``canSkip`` parameter
    which defaults to **false** (``CardSelectCmd.cs:216``). Ten of its twelve
    v0.107.1 callers pass ``canSkip: true`` explicitly; the two that take the
    default — ``Toolbox`` and ``KnowledgeDemon`` — are MANDATORY, and Toolbox
    is the single largest unparked-injection family in the corpora. Injected
    as skippable, the search was free to decline a card the real screen has no
    skip button for.

    ``cancelable`` is NOT a general substitute for the arity test, which is
    why this is keyed on ``source`` rather than applied everywhere.
    ``CardSelectorPrefs.Cancelable`` is an ``init``-only field defaulting to
    false that the two public constructors never set, so it reads false on
    screens that are plainly declinable — Gambling Chip's mulligan is
    ``CardSelectorPrefs(prompt, 0, 999999999)``, answerable with nothing via
    confirmation rather than via a Cancel button. Trusting it there would
    force a mandatory discard and break the mulligan recognition below.

    Clearing ``can_skip`` alone would not close the hole either: both drivers
    gate their "answer with nothing" branch on ``min_select <= 0``, and
    :func:`choice_picks_from_tags` treats a zero minimum as permission to
    return an empty selection. So a mandatory screen's minimum is re-floored
    to 1 — which is what the C# screen means anyway (exactly one card, no skip
    button).

    Args:
        cancelable: The world's captured flag (``state["cancelable"]``) — for
            ``FromChooseACardScreen`` this is the ``canSkip`` argument
            verbatim. None when no capture patch fired, in which case the
            legacy inference is kept, so a world that does not send it is
            unchanged.
        source: The ``CardSelectCmd`` entry-point name the world captured
            (``state["source"]``), or None.
        min_select: The already-clamped minimum selection arity.
        max_select: The already-clamped maximum selection arity.

    Returns:
        ``(can_skip, min_select)``, both normalized.
    """
    if source != "FromChooseACardScreen" or cancelable is None:
        return min_select <= 0, min_select
    can_skip = bool(cancelable)
    if not can_skip and min_select <= 0 and max_select >= 1:
        min_select = 1
    return can_skip, min_select


def recognize_card_select(
    *,
    parked: dict[str, Any] | None,
    options: list[dict[str, Any]],
    board: dict[str, Any],
    wire: dict[str, Any],
    min_select: int,
    source_model: str | None = None,
    prompt_key: str | None = None,
) -> tuple[str, str | None]:
    """Recognize a mid-combat ``card_select`` screen the engine never opened.

    Shared by BOTH world drivers — ``headless_eval.pick_card_select_mirrored``
    (DLL world) and ``realgame_mcts.pick_card_select_mirrored`` (live
    SpireMentor world) — so the recognition logic cannot fork and drift again.
    It operates on dialect-neutral inputs: a headless-dialect ``board`` (the
    DLL's own ``combat_play`` dict, or ``realgame_adapter.
    to_headless_combat_play``'s translation of the bridge snapshot), the
    ``sts2_hint.adapter.translate`` wire built from it, and the option list
    the caller is about to hand to :meth:`CombatMirror.inject_choice`.

    The engine's own kind is used when it derived one (``parked``). A screen
    the engine never opened (a combat-START selection, Gambling-Chip-style)
    has no parked choice — and ``card_select`` is NOT a usable catch-all for
    that shape: with nothing to resolve the world's card names against, the
    options carry no locator, and ``ChoiceKind::CardSelect`` is an explicit
    no-op placeholder arm in the engine. ``inject_choice`` refuses the
    injection rather than resolving it to nothing; the recognition below maps
    the known unparked screens to their real engine kinds instead:

    * Generative screens (Toolbox's combat-start colorless offer, the
      Attack/Skill/Power/Colorless potions, Discovery) — by ``source_model``
      when the world named the effect, else by shape (no option present in
      any pile).
    * KnowledgeDemon's curse pick — ``resolve_monster_curse_select`` maps
      card_id → (PowerKind, amount), no locators needed either.
    * Gambling Chip's turn-1 mulligan — round 1, skippable, and the option
      multiset is exactly the mirror's whole hand. Mapped to the engine's
      real ``discard_from_hand_and_draw`` (locators filled in from the hand
      itself, mutating ``options`` in place) so resolution actually
      discards+draws in the mirror and the search ranks the picks.

    Args:
        parked: :meth:`CombatMirror.parked_choice` — the engine's own derived
            choice, or ``None`` when it never opened one.
        options: The world's option list (``{"id", "upgraded"}`` dicts, ids
            already resolved via ``sts2_hint.adapter.registry_card_id``).
            MUTATED in place when the Gambling-Chip recognition fires: each
            option gains its ``pile``/``index`` hand locator.
        board: Headless-dialect combat board (``hand``, ``round``,
            ``player.relics``).
        wire: ``sts2_hint.adapter.translate(board, ...)`` — the same wire the
            caller just reconciled the mirror against.
        min_select: The already-clamped minimum selection arity.
        source_model: The world's name for the effect that opened the screen
            (``state["source_model"]``, RunSimulator.cs CardSelectCmd context
            patch), when it captured one.
        prompt_key: The world's localization prompt key, when captured.

    Returns:
        ``(kind, turn_start_blocked_on)``. ``kind`` is what the caller passes
        to :meth:`CombatMirror.inject_choice`. ``turn_start_blocked_on`` is
        the :class:`TurnStartBlock` naming the relic OR power whose turn-start
        hook the world is blocked inside, or ``None``; when set, the caller
        MUST call :meth:`CombatMirror.arm_turn_start_resume_for` with it after
        ``inject_choice`` and BEFORE the search, so the rollouts see the
        deferred turn-setup continuation.
    """
    from sts2_hint import adapter

    n = len(options)
    kind = str(parked.get("kind")) if parked else "card_select"
    turn_start_blocked_on: TurnStartBlock | None = None
    if parked is None:
        src = str(source_model or "")
        # `kind` is what the wire dialect actually carries (`state_to_dict`
        # emits {kind, uid, hp, ...}; there is no `id` key) — reading only
        # `id` left this set always-empty, so the KnowledgeDemon curse pick
        # below could never fire and the screen fell through to the
        # generative-shape fallback: the mirror granted the curse INTO HAND
        # instead of resolving `monster_curse_select`, and the suspended
        # enemy turn never resumed (the two mirror_d1 seed-1004 records with
        # the mirror a full round behind at hand [DISINTEGRATION]).
        wire_enemy_ids = {
            str(e.get("id") or e.get("kind") or "") for e in (wire.get("encounter") or [])
        }
        if src in _GENERATIVE_SOURCES:
            kind = _GENERATIVE_SOURCES[src]
        elif src in _TURN_START_POWER_SOURCES:
            # GH #355. A power's `AfterPlayerTurnStart` select, on ANY turn.
            # Unlike Gambling Chip this needs no shape heuristic: the world
            # NAMES the power (`FromHand` / `FromHandForDiscard` both declare
            # the `AbstractModel source` parameter the capture patch reads, and
            # both powers pass `this`), so the recognition is the name itself
            # and cannot be confused with a potion of the same shape.
            #
            # Locators come from the hand, exactly as the Gambling-Chip block
            # below does it — C# `FromHand`/`FromHandForDiscard` pass the hand
            # `Cards` with no OrderBy, so the world's option order IS hand
            # order, and duplicates must still consume distinct indices.
            power_kind, power_id = _TURN_START_POWER_SOURCES[src]
            try:
                located = _locate_options_in_hand(options, board)
            except Exception:
                located = False
            if located:
                kind = power_kind
                turn_start_blocked_on = TurnStartBlock("power", power_id)
        elif (
            n == 2
            and options
            and str(options[0].get("id")) == "DISINTEGRATION"
            and "KnowledgeDemon" in wire_enemy_ids
        ):
            # KnowledgeDemon's curse pick maps card_id → (PowerKind, amount)
            # in `resolve_monster_curse_select` — no locators needed either.
            kind = "monster_curse_select"
        else:
            try:
                deck = wire.get("deck") or {}
                in_any_pile = {
                    (str(c.get("id")), bool(c.get("upgraded")))
                    for key in ("hand", "draw_pile", "discard_pile", "exhaust_pile")
                    for c in (deck.get(key) or [])
                }
                if options and all(
                    (str(o["id"]), bool(o["upgraded"])) not in in_any_pile for o in options
                ):
                    relics_up = {
                        str(r.get("id") or r.get("name") or "").upper()
                        for r in (board.get("player") or {}).get("relics", [])
                    }
                    round_one = int(board.get("round") or 0) <= 1
                    kind = (
                        "add_generated_to_hand"
                        if round_one and any("TOOLBOX" in r for r in relics_up)
                        else "add_generated_free_to_hand"
                    )
            except Exception:
                pass
    if parked is None and kind == "card_select":
        # Gambling Chip's turn-1 mulligan: the mirror is seeded from this
        # very decision's wire, so its own `after_player_turn_start` hook never
        # ran and nothing is parked — 32 of the 57 v28wr unanswerable
        # injections, and the arm always answered "skip" (never mulligans,
        # while the engine arm searches it natively). The screen is
        # recognizable without a prompt key: round 1, skippable, and the
        # option multiset is exactly the mirror's whole hand. Map it to the
        # engine's real kind so resolution actually discards+draws in the
        # mirror and the search ranks the picks; the locators come from the
        # hand itself.
        try:
            mirror_wire_hand = [
                (adapter.registry_card_id(str(c.get("id")), "$.hand"), bool(c.get("upgraded")))
                for c in (board.get("hand") or [])
            ]
            opts_key = sorted((str(o["id"]), bool(o["upgraded"])) for o in options)
            relic_ids = {
                str(r.get("id") or r.get("name") or "")
                for r in (board.get("player") or {}).get("relics", [])
            }
            if (
                int(board.get("round") or 0) <= 1
                and min_select == 0
                and n == len(mirror_wire_hand)
                and opts_key == sorted(mirror_wire_hand)
                and any("GAMBLING" in r.upper() for r in relic_ids)
            ):
                kind = "discard_from_hand_and_draw"
                # Arming the turn-setup resume is the one part of this
                # recognition that MUST NOT misfire: replaying the turn-start
                # batch against a board that already ran it doubles the Vigor /
                # energy / AoE. GamblersBrew's potion screen has the identical
                # shape (choose-any over the whole hand, same
                # `DiscardAndDraw`) and can be drunk on turn 1 by a player who
                # also holds the relic — so require the DLL's own prompt key
                # when it captured one, and only fall back to the shape when it
                # did not (`prompt_key` is None on an unpatched entry point).
                pk = str(prompt_key or "")
                relic = next((r for r in relic_ids if "GAMBLING" in r.upper()), None)
                if relic is not None and (not pk or "GAMBLING_CHIP" in pk.upper()):
                    turn_start_blocked_on = TurnStartBlock("relic", relic)
                _locate_options_in_hand(options, board)
        except Exception:
            # Best-effort recognition; an unmappable id just keeps the
            # refusal path, which is where we already were.
            pass
    return kind, turn_start_blocked_on


__all__ = [
    "CombatMirror",
    "Desync",
    "MirrorDivergence",
    "MirrorPolicy",
    "MirrorStats",
    "TurnStartBlock",
    "diff_observables",
    "observables",
    "policy_from_env",
    "recognize_card_select",
]
