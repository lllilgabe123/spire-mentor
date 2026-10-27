"""A persistent per-fight engine mirror for the live ADVISOR lane.

Why
---
The advisor is *stateless*: every decision it throws the fight away and
rebuilds the board from the snapshot the mod just sent. The autopilot is not
-- it keeps a :class:`combat_mirror.CombatMirror` running alongside the fight.
Three things live only in that mirror, and the advisor therefore never had
them:

1. **The open choice and its suspended tail.** Burning Pact is "exhaust a card
   THEN draw 2"; the engine holds that tail as the parked
   ``PendingChoice.post_effect``. A choice injected onto a freshly rebuilt
   board has ``post_effect: None``, so a search over it ranks the exhaust
   while believing no draw follows. That is a confident wrong answer, which is
   worse than the leftmost-card default it would replace -- amputating
   ``post_effect`` was 60 of 188 power divergences on the v27wr sweep.
2. **Hidden per-turn counters** (GH #370). ``card_plays_started_this_turn``,
   ``shivs_played_this_turn``, ``generated_cards_this_combat``,
   ``stars_gained_this_turn``, ``vigor_latch_card``: none is emitted by
   SpireMentor, so a rebuilt board reads them as 0 forever. Some are legality
   gates (``RingingPower.ShouldPlay`` / ``SlothPower.ShouldPlay``), so a stuck
   value changes which actions the search believes exist.
   ``transplant_hidden`` carries them; only a carried mirror has them to carry.
3. The true RNG cursors and undisclosed relic/power counters, same mechanism.

What this does NOT buy
----------------------
Clairvoyance. The mirror holds the true draw order, which makes it the one
object forbidden from searching with it -- :meth:`CombatMirror.search` refuses
a clairvoyant profile outright. Both lanes still search under ``beliefdraw``.

OBSERVE, not STRICT
-------------------
The autopilot's mirror watches a policy; this one watches a *human*, who may
play a card nobody recommended, rewind, or walk away. OBSERVE is the mode for
that: a divergence tags the mirror and re-seeds from reality instead of
raising. A dropped mirror costs advice quality for the rest of the fight --
which is exactly where the advisor already sits permanently -- so the floor
does not move and the ceiling does.

Lifecycle
---------
Seeded/reconciled on every combat ``advice_request``; dropped on a fight
change, on ``snapshot_restored`` / ``combat_undone`` (the mod broadcasts both
for *any* rewind, including ones the human triggers from the in-game toolbar),
and at ``game_over`` / ``run_complete``.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import Any

from combat_mirror import CombatMirror, MirrorDivergence, MirrorPolicy
from realgame_adapter import character_display_name, run_ascension, to_headless_combat_play
from sts2_hint.recommender import RecommendConfig

logger = logging.getLogger(__name__)

_INDEX_RE = re.compile(r"\[[^\]]*\]")


#: The park INSIDE the end-of-turn flush, one hook before the round increments.
#:
#: Named on its own because it is the only one of the two whose board proof can
#: be repaired rather than believed: nothing between the play phase and this
#: prompt touches an enemy, so a board that disagrees here disagreed already.
#: See :meth:`AdvisorMirror.replay_end_turn`.
_RETAIN_PARK = "retain_at_turn_end"

#: Choice kinds :meth:`AdvisorMirror.replay_end_turn` will answer.
#:
#: An allowlist, not a wildcard: the only thing that sent us down this path is
#: "a power opened a screen", so the park is the identity proof and an
#: unrecognized kind means the proof is missing. Both members are screens the
#: engine's own end-of-turn pipeline raises, one on either side of the round
#: increment -- ``retain_at_turn_end`` inside the flush (Well-Laid Plans),
#: ``discard_from_hand`` in the next turn's setup (Tools of the Trade).
_END_TURN_PARKS = frozenset({_RETAIN_PARK, "discard_from_hand"})


def fight_key(state: dict[str, Any]) -> tuple[Any, Any]:
    """Identity of the fight a bridge message belongs to.

    One room is one fight, so ``(act, floor)`` separates them. A key that
    changes mid-fight only costs a re-seed (the next reconcile rebuilds from
    the wire), which is the same thing the mirror does after any divergence --
    so an imperfect key degrades quality, never correctness.
    """
    return (state.get("act"), state.get("floor"))


def _enemy_hp(wire: dict[str, Any]) -> list[Any]:
    """The LIVING enemy HP vector — the cheapest fingerprint a target moves.

    Living-only, and that is the whole point. The engine keeps a corpse in
    `encounter` (`{"hp": 0, "is_dead": true}`); the live producer drops the
    dead enemy from its embedded combat state entirely. Comparing the two
    positionally means a play that KILLS can never fingerprint-match — the
    mirror's vector is one entry longer than the world's no matter which
    target is tried — so `replay_opening_play` declines every killing play
    that then opens a screen.

    That is the three `no board matched replay of NEOWS_FURY` declines in
    `logs/f7-mirror-check16`, and Neow's Fury is only the card that happened
    to show it: 10-14 damage, and it opens a discard-pile select immediately
    after. Nothing about the class is Neow-specific.

    Filtering both sides is safe whichever way the producer behaves: if it
    ever does report a corpse, dropping it on both sides still compares like
    with like. Order is still positional over the survivors, so it stays
    discriminating — killing a different enemy yields a different vector.
    """
    return [e.get("hp") for e in (wire.get("encounter") or []) if not e.get("is_dead")]


class AdvisorMirror:
    """Owns the advisor's per-fight :class:`CombatMirror`, or nothing.

    Every method is total: on any internal failure the mirror is dropped and
    the caller falls back to the stateless path it used before. Nothing here
    may raise into the advisor's serve loop.
    """

    def __init__(
        self,
        cfg: RecommendConfig,
        *,
        enabled: bool = True,
        log_path: Any = None,
    ) -> None:
        self.enabled = enabled
        self.cfg = cfg
        self._log_path = log_path
        self._mirror: CombatMirror | None = None
        self._key: tuple[Any, Any] | None = None
        #: Counters for the run log -- how often the carried state was
        #: actually available where it matters.
        self.stats: dict[str, int] = {
            "seeds": 0,
            "drops": 0,
            "reconciles": 0,
            "desyncs": 0,
            "replays_attempted": 0,
            "replays_landed": 0,
            "searched_selects": 0,
            "followed": 0,
            "unfollowable": 0,
            "turns_followed": 0,
            "stale_reports_skipped": 0,
            "teammate_adopted": 0,
        }
        #: Why the searched rung declined, by reason. The first live runs
        #: showed 8-10% of in-combat selections being searched and the rest
        #: falling through SILENTLY -- no exception, no divergence, nothing to
        #: read. Counting the reason is the difference between diagnosing that
        #: and guessing at it.
        self.declines: dict[str, int] = {}
        #: Desyncs by wire path, list keys folded (``$.encounter[].hp``).
        #: The total alone cannot say WHAT the mirror fails to follow; a
        #: co-op run desyncs at 2-3x the solo rate, and only the paths can
        #: tell a teammate's move from an engine gap.
        self.desync_paths: dict[str, int] = {}
        #: Co-op enemy-side diffs the mirror adopted instead of reporting
        #: (a teammate's moves), by the same folded path.
        self.adopted_paths: dict[str, int] = {}
        self._said: set[str] = set()
        #: The card whose play this advisor REPLAYED to answer a select, and
        #: the seed generation it was replayed under. See
        #: :meth:`_expect_stale_report`.
        self._stale_report: tuple[tuple[str, bool], int] | None = None
        #: Bumped on every seed, so a suppression can tell "the mirror was
        #: rebuilt since" from "it is the same one".
        self._generation = 0
        #: A multi-pick selection whose picks were all searched at once, and
        #: how much of it has been served. See :meth:`remember_picks`.
        self._picks: dict[str, Any] | None = None

    def decline(self, reason: str) -> None:
        """Record (and, once per reason, announce) a decline to search."""
        self.declines[reason] = self.declines.get(reason, 0) + 1
        if reason not in self._said:
            self._said.add(reason)
            logger.info("searched card_select declined (%s) — first of this kind", reason)

    # -- lifecycle ---------------------------------------------------------

    @property
    def live(self) -> CombatMirror | None:
        """The mirror, iff one is seeded and has not been dropped."""
        m = self._mirror
        return m if (m is not None and m.alive) else None

    def drop(self, reason: str) -> None:
        """Forget the fight. Safe to call at any time, including when empty."""
        if self._mirror is not None:
            self.stats["drops"] += 1
            try:
                self._mirror.drop(reason)
            except Exception as exc:  # a drop must never propagate
                logger.debug("advisor mirror drop(%s) failed: %s", reason, exc)
        self._mirror = None
        self._key = None
        self._picks = None

    def on_rewind(self, msg: dict[str, Any]) -> None:
        """Drop on ``snapshot_restored`` / ``combat_undone``.

        Both carry a fresh ``state_id`` precisely because everything older is
        stale (PROTOCOL.md section 7); a carried combat state is exactly that.
        A snapshot restore lands at room entry so the fight ends anyway, but a
        turn undo keeps the fight and moves the board BACKWARDS, which is the
        case that would silently corrupt a carried mirror.
        """
        # A rewind can un-play the very card a pending suppression is waiting
        # for, and then the world may report it AGAIN. Forget the expectation
        # rather than swallow a real second play.
        self._stale_report = None
        self.drop(f"rewind: {msg.get('type')}")

    # -- per-decision ------------------------------------------------------

    def on_combat_board(self, wire: dict[str, Any], state: dict[str, Any]) -> CombatMirror | None:
        """Seed or reconcile for one combat decision; return the live mirror.

        Args:
            wire: ``sts2_hint.adapter.translate`` output for this board.
            state: The bridge message, for :func:`fight_key`.

        Returns:
            The mirror when it is carrying this fight, else ``None`` (the
            caller then behaves exactly as it did before this class existed).
        """
        if not self.enabled:
            return None
        key = fight_key(state)
        if self._mirror is not None and self._key != key:
            # Say what the fight that just ended did BEFORE forgetting it. The
            # run-end note is not enough: the climb driver closes the game on
            # death, so `game_over` never reaches the advisor and the only
            # summary that existed was never emitted (both 2026-08-24 runs).
            logger.info("advisor %s", self.note())
            self._stale_report = None  # cannot span two fights
            self.drop("new fight")
        if self._mirror is None:
            self._mirror = CombatMirror(
                self.cfg,
                policy=MirrorPolicy.OBSERVE,
                label="advisor",
                log_path=self._log_path,
            )
            self._key = key
            self.stats["seeds"] += 1
            self._generation += 1
        self._catch_up_turns(wire)
        if self._mirror is None:
            return None
        # Co-op: the teammate's moves land on the enemies between ours.
        self._mirror.teammates = max(0, int(state.get("player_count") or 1) - 1)
        try:
            desyncs = self._mirror.reconcile(wire)
        except MirrorDivergence as exc:
            # OBSERVE should not raise, but a transplant the shapes refuse
            # still can. Never let it reach the serve loop.
            logger.info("advisor mirror dropped on reconcile: %s", exc)
            self.drop("reconcile refused")
            return None
        except Exception as exc:
            logger.warning("advisor mirror reconcile failed: %s", exc)
            self.drop("reconcile error")
            return None
        self.stats["reconciles"] += 1
        # `or []` on purpose: `reconcile` has several exits and one of them
        # returned `None` in spite of its annotation. That is fixed at the
        # source, but a miscounted desync must never be able to take the
        # advisor's serve loop down with it — the whole point of this class is
        # that a mirror problem degrades the advice, not the session.
        self.stats["desyncs"] += len(desyncs or [])
        for d in desyncs or []:
            key = _INDEX_RE.sub("[]", d.path)
            self.desync_paths[key] = self.desync_paths.get(key, 0) + 1
        adopted = self._mirror.last_adopted if self._mirror is not None else []
        self.stats["teammate_adopted"] += len(adopted)
        for d in adopted:
            key = _INDEX_RE.sub("[]", d.path)
            self.adopted_paths[key] = self.adopted_paths.get(key, 0) + 1
        return self.live

    def _catch_up_turns(self, wire: dict[str, Any]) -> None:
        """End the mirror's turn when the world has moved to a later round.

        `human_action` reports card plays and nothing else, so a turn ending
        is invisible on the wire — and it is the single biggest thing the
        mirror was failing to follow. Divergence paths from live run 4, before
        this: `$.round` x46 and `$.player.energy` x46 out of 79 records, i.e.
        most of the remaining desyncs were one unfollowed turn boundary each,
        dragging `$.deck.hand` / `$.deck.draw_pile` / `$.encounter` along as
        the discard-and-draw and the whole enemy turn landed at once.

        Ending the turn in the ENGINE is not the same as being told the world
        ended it: the engine runs its own end-of-turn pipeline and enemy turn,
        which is exactly the point — per-turn counters reset, powers tick,
        block expires. Where its enemy turn rolls differently from the
        world's, the `reconcile` immediately after re-imposes reality; what it
        no longer does is report the ENTIRE transition as the mirror's error.

        Bounded at 3 rounds: a bigger gap means the mirror missed something
        structural and catching up by simulation would be inventing a fight.
        """
        mirror = self.live
        if mirror is None:
            return
        want = wire.get("round")
        if not isinstance(want, int):
            return

        import sts2_rs

        try:
            have = sts2_rs.state_to_dict(mirror.state).get("round")
        except Exception:
            return
        if not isinstance(have, int) or have >= want:
            return
        if want - have > 3:
            self.drop(f"round gap too wide to follow ({have} -> {want})")
            return
        for _ in range(want - have):
            try:
                result = mirror.advance({"action": "end_turn"})
            except MirrorDivergence as exc:
                logger.info("advisor mirror dropped ending its turn: %s", exc)
                self.drop("turn end diverged")
                return
            except Exception as exc:
                logger.warning("advisor mirror could not end its turn: %s", exc)
                self.drop("turn end error")
                return
            self.stats["turns_followed"] += 1
            if result in ("combat_won", "combat_lost"):
                # The mirror thinks the fight is over and the world disagrees;
                # nothing useful is left to carry.
                self.drop(f"mirror ended the fight early ({result})")
                return

    def on_world_action(self, msg: dict[str, Any]) -> None:
        """Advance the mirror through an action the world just committed.

        THE thing that makes a mirror a mirror. The autopilot advances its
        own after every accepted action, so its mirror PREDICTS the next
        board and a desync means a real disagreement. Until the mod carried
        the machine form of a play, the advisor could only `reconcile` — it
        predicted nothing, so every ordinary change read as a divergence
        (2232 desyncs over 540 decisions on live run
        001-ironclad-a8-0824-1939, 42x the autopilot's rate) and the mirror
        re-seeded constantly instead of carrying anything.

        `human_action` now carries `card` (`id` + `upgraded`) and
        `target_index`. The card is located in the mirror's own hand by
        identity rather than by a wire index, so the two never have to agree
        on hand ORDER — only on contents.

        A potion use arrives the same way, as ``potion`` (``id``), and is
        followed through the engine's own drink action. It used to be the
        headline example of an action with no machine form, and so DROPPED the
        mirror every time: a fight in which the player drank anything lost its
        mirror at that moment, and with it every searched decision afterwards.

        Any action that still cannot be followed DROPS the mirror instead of
        letting it fall silently out of step: a card the mirror does not hold,
        a potion not in its belt, or an engine rejection. A dropped mirror
        re-seeds on the next board and costs advice quality; a wrong one costs
        correctness.
        """
        mirror = self.live
        if mirror is None:
            return
        if msg.get("decision_type") != "combat_action":
            return
        if msg.get("action") == "end_turn":
            # The coach's end-turn report (PROTOCOL.md v5). Not followed here:
            # `_catch_up_turns` already ends the mirror's turn from the next
            # board, and doing both would end it twice. Not a drop either --
            # it is the one combat action with no card or potion by design.
            return
        card = msg.get("card")
        drink = msg.get("potion")
        if isinstance(drink, dict) and drink.get("id"):
            self._follow_drink(mirror, drink, msg.get("target_index"))
            return
        if not isinstance(card, dict) or not card.get("id"):
            # Older mod build, or an action with no machine form.
            # Either way the board moved somewhere we cannot follow.
            self.drop(f"unfollowable action: {msg.get('action')}")
            self.stats["unfollowable"] += 1
            return

        import sts2_rs

        try:
            hand = sts2_rs.state_to_dict(mirror.state)["deck"]["hand"]
        except Exception as exc:
            logger.debug("advisor mirror: cannot read hand to follow: %s", exc)
            self.drop("hand unreadable")
            return
        want_id = str(card["id"]).upper()
        want_up = bool(card.get("upgraded"))
        if self._expect_stale_report((want_id, want_up)):
            return
        idx = next(
            (
                i
                for i, c in enumerate(hand)
                if str(c.get("id", "")).upper() == want_id and bool(c.get("upgraded")) == want_up
            ),
            None,
        )
        if idx is None:
            self.drop(f"{want_id} not in the mirror's hand to follow")
            self.stats["unfollowable"] += 1
            return

        target = msg.get("target_index")
        action = {
            "action": "play",
            "card_index": idx,
            "target_index": int(target) if isinstance(target, int) else None,
        }
        try:
            mirror.advance(action)
        except MirrorDivergence as exc:
            logger.info("advisor mirror dropped following %s: %s", want_id, exc)
            self.drop("following diverged")
            return
        except Exception as exc:
            logger.warning("advisor mirror could not follow %s: %s", want_id, exc)
            self.drop("follow error")
            return
        self.stats["followed"] += 1

    def _follow_drink(self, mirror: CombatMirror, drink: dict[str, Any], target: Any) -> None:
        """Advance the mirror by a potion the world reports the player drank.

        Located by potion id in the mirror's own belt, not by the wire's slot
        number, for the same reason a card is located by identity in hand: the
        two sides then never have to agree on belt ORDER, only on contents.
        """
        src = str(drink.get("id") or "").strip().upper()
        action = self._drink_action(mirror, src, target)
        if action is None:
            self.drop(f"{src} not in the mirror's belt to follow")
            self.stats["unfollowable"] += 1
            return
        try:
            mirror.advance(action)
        except MirrorDivergence as exc:
            logger.info("advisor mirror dropped following %s: %s", src, exc)
            self.drop("following diverged")
            return
        except Exception as exc:
            logger.warning("advisor mirror could not follow %s: %s", src, exc)
            self.drop("follow error")
            return
        self.stats["followed"] += 1

    def _expect_stale_report(self, want: tuple[str, bool]) -> bool:
        """True when this play report is the echo of a replay we already made.

        The two mechanisms that keep the mirror in step collide exactly once
        per searched selection, and predictably:

        1. To answer a `card_select` the advisor REPLAYS the opening play in
           its mirror (:meth:`replay_opening_play`), then drops the mirror --
           it is parked on an injected choice that nothing will ever resolve,
           because the advisor does not execute.
        2. The mirror re-seeds from the next combat board, which ALREADY
           reflects that play.
        3. Only then does the world report the play, because
           `CardPlayFinishedEntry` fires when a play FINISHES and a
           choice-opening card finishes only once the selection is answered
           (PROTOCOL.md section 7).

        Following step 3 would be following a card the re-seeded mirror no
        longer holds: `"<CARD> not in the mirror's hand to follow"`, and a
        second drop of a mirror that was correct. Two re-seeds per searched
        select, both avoidable.

        Guarded on the seed generation, not just the card: the suppression is
        honoured only if the mirror really was rebuilt since the replay. If it
        somehow was not, the play is still live and must be followed normally
        -- swallowing it there would put the mirror one card behind and say
        nothing, which is the failure mode this whole class exists to avoid.
        Consumed on use, so a genuine second play of the same card is
        followed.
        """
        pending = self._stale_report
        if pending is None or pending[0] != want:
            return False
        if pending[1] == self._generation:
            return False  # never re-seeded: this play is real, follow it
        self._stale_report = None
        self.stats["stale_reports_skipped"] += 1
        return True

    def source_is_replayable(self, source_model: str | None, source_kind: str | None) -> bool:
        """Can a replay possibly work for this source? Decided from the wire alone.

        Cheap and board-free, so a screen that was never answerable is
        declined BEFORE building a board and forking a mirror for it. Shared
        with :meth:`replay_opening_play` so the two can never disagree about
        what counts as replayable.
        """
        src = str(source_model or "").strip().upper()
        kind = str(source_kind or "").strip().lower()
        if not src:
            self.decline("wire named no source_model")
            return False
        if kind and kind not in ("card", "potion"):
            # Authoritative, and the reason the pile search in the replay is
            # only a fallback: knowing the TYPE means never confusing "this
            # was never a card" with "this card fell out of my copy of the
            # fight".
            #
            # A POTION is replayable for the same reason a card is: the engine
            # models the drink as a first-class action
            # (`Action::DrinkPotion`), and a potion that opens a card-select
            # parks a real `pending_choice` behind it, with the deferred
            # `Hook.AfterPotionUsed` tail carried on
            # `pending_potion_hook_resume`. What it needs is a different
            # replay -- :meth:`replay_opening_drink` -- not a different answer
            # to "can this be replayed at all".
            self.decline(f"{src} is a {kind}, not a card or a potion")
            return False
        return True

    @staticmethod
    def _drink_action(mirror: CombatMirror, src: str, target: Any = None) -> dict[str, Any] | None:
        """The engine's own ``drink_potion`` action for ``src``, or ``None``.

        Resolved through :meth:`legal_actions` rather than by constructing a
        slot index by hand, on the same principle as the card path: the engine
        decides what is drinkable right now (a character-locked potion and an
        ``Automatic`` one -- Fairy in a Bottle -- are both in the belt and
        neither is a legal manual drink), and its own action carries the slot
        in its own frame.

        Two belt slots holding the SAME potion are interchangeable -- unlike
        two copies of a card in hand, a potion carries no upgrade state and no
        uid the outcome could depend on -- so the first is taken without
        disambiguating.
        """
        import sts2_rs

        try:
            belt = (sts2_rs.state_to_dict(mirror.state).get("player") or {}).get("potions") or []
        except Exception as exc:
            logger.debug("advisor mirror: cannot read the belt: %s", exc)
            return None
        slots = {i for i, p in enumerate(belt) if str(p or "").strip().upper() == src}
        if not slots:
            return None
        try:
            legal = [
                a
                for a in mirror.state.legal_actions()
                if a.get("action") in ("drink_potion", "potion") and int(a.get("slot", -1)) in slots
            ]
        except Exception as exc:
            logger.debug("advisor mirror: cannot enumerate drinks: %s", exc)
            return None
        if not legal:
            return None
        action = dict(legal[0])
        # `legal_actions` emits every drink with `target: None` (the per-enemy
        # fan-out is deferred, bead `sts2-rs-h9u`) and the dispatcher resolves
        # a null target to the first alive enemy, exactly as C#
        # `RunSimulator.DoUsePotion` does. So passing the target is an
        # OPTIMISATION over that default, never a requirement -- and the
        # choice-opening potions are self/untargeted anyway.
        if isinstance(target, int):
            action["target_index"] = int(target)
        return action

    def replay_opening_drink(
        self,
        source_model: str | None,
        world_wire: dict[str, Any],
        *,
        target_index: Any = None,
        on: CombatMirror | None = None,
    ) -> bool:
        """Re-drink the potion that opened this screen, to reach its choice.

        The potion sibling of :meth:`replay_opening_play`, and simpler for one
        reason: ``source_model`` names the potion outright, so there is no
        "which of these was it" to settle. A card play has to be disambiguated
        against the board because several plays in hand could have opened the
        same screen; a potion is identified by the wire.

        The board proof therefore does a different job here. It is not
        choosing between candidates -- it is checking that this mirror's
        picture of the fight still matches the world's before a search runs on
        it. A mismatch DECLINES rather than dropping: the identity was never
        in doubt, so a mismatch says the mirror drifted, and
        :meth:`CombatMirror.reconcile` already re-seeds that on the next
        board. Dropping here would destroy a mirror on the strength of an
        enemy-HP check that an untargeted potion cannot move.

        Args:
            source_model: The potion's ``Id.Entry``, from the screen's wire.
            world_wire: The world's own board, as the match gate.
            target_index: The drink's target, when the wire named one.
            on: The mirror to replay into -- a fork, in the advisor lane.

        Returns:
            True when the mirror is now parked on the potion's own choice.
        """
        mirror = on if on is not None else self.live
        src = str(source_model or "").strip().upper()
        if mirror is None or not mirror.alive:
            self.decline("no live mirror")
            return False
        if mirror.state.pending_choice() is not None:
            # Already parked -- same reasoning as the play path: nothing to
            # replay, so nothing to disqualify. Checked before the name.
            return True
        if not src:
            self.decline("wire named no source_model")
            return False

        action = self._drink_action(mirror, src, target_index)
        if action is None:
            self.decline(f"{src} is not drinkable in the mirror")
            return False
        self.stats["replays_attempted"] += 1

        import sts2_rs

        # Prove it on a clone before touching the mirror, so a drink that
        # turns out not to reproduce the screen leaves no trace. `on` is
        # normally a fork already, but this method is callable against a live
        # mirror and must not mutate one it is about to decline.
        try:
            probe = mirror.state.clone()
            parked = probe.apply_step(action)
        except Exception as exc:
            logger.debug("advisor mirror: %s could not be drunk: %s", src, exc)
            self.decline(f"{src} could not be drunk in the mirror")
            return False
        if parked != "pending_choice":
            self.decline(f"{src} drunk but parked no choice ({parked})")
            return False
        try:
            matched = _enemy_hp(sts2_rs.state_to_dict(probe)) == _enemy_hp(world_wire)
        except Exception as exc:
            logger.debug("advisor mirror: cannot compare boards for %s: %s", src, exc)
            return False
        if not matched:
            self.decline(f"{src} drunk but the board did not match")
            return False

        try:
            result = mirror.advance(action)
        except MirrorDivergence as exc:
            logger.info("advisor mirror dropped drinking %s: %s", src, exc)
            self.drop("drink replay diverged")
            return False
        except Exception as exc:
            logger.warning("advisor mirror drink replay of %s failed: %s", src, exc)
            self.drop("drink replay error")
            return False
        if result != "pending_choice":
            self.decline(f"{src} drunk but parked no choice ({result})")
            return False
        self.stats["replays_landed"] += 1
        return True

    @staticmethod
    def _identity(card: Any) -> tuple[str, bool]:
        """A card as the pair that survives a shrinking offer list."""
        if not isinstance(card, dict):
            return ("", False)
        return (str(card.get("id") or card.get("name") or "").upper(), bool(card.get("upgraded")))

    def remember_picks(
        self, source_model: Any, offered: Sequence[Any], picks: Sequence[int]
    ) -> None:
        """Keep the picks a MULTI-pick search made, for the screens still to come.

        A card that discards two opens ONE engine choice with `max_select`
        two, and the search answers all of it at once -- the note says so:
        `combat MCTS - 2 pick(s) searched`. The mod then presents one advice
        request per pick, with the offered list shrinking by the card just
        taken. Only the first of those was ever searched; the rest replayed
        the opening play from scratch, so the engine parked a choice with the
        FULL option list against the world's shortened one, `inject_choice`
        could not reconcile them, and the fork was dropped. Run
        003-silent-a8-0825-0125: `70/70 replays` landed alongside 33 injection
        drops, and 21 decisions fell to the rarity sort with an answer already
        computed and thrown away.

        Stored by identity rather than index, because the indices are exactly
        what the next screen renumbers.
        """
        self._picks = None
        if len(picks) < 2:
            return
        ids = [self._identity(offered[i]) for i in picks if 0 <= i < len(offered)]
        if len(ids) < 2:
            return
        # The WHOLE plan, pick 1 included -- not "what is left to serve".
        # :meth:`next_remembered_pick` re-derives the position from the offer
        # it is handed, so nothing here tracks progress and a re-ask cannot
        # advance it.
        self._picks = {
            "source": str(source_model or "").strip().upper(),
            "offer": [self._identity(c) for c in offered],
            "plan": ids,
        }

    def next_remembered_pick(self, source_model: Any, offered: Sequence[Any]) -> int | None:
        """The next searched pick for this selection, as an index into `offered`.

        Returns None -- and forgets the selection -- unless the world's offer
        is EXACTLY the original minus the plan's own first N picks. That is
        the whole safety argument: the advisor recommends but never executes,
        so the world may have picked something else entirely, and the multiset
        check is what tells the two apart. A shrinking list is not proof on its
        own; the same list minus the same cards is.

        IDEMPOTENT, and that is the point. How far through the plan we are is
        re-derived from the offer every call (``len(offer) - len(now)``), so
        being asked twice for the same pick answers it twice with the same
        card. Consuming the plan on SERVE instead was wrong, because a serve
        is not an execution: the mod drops advice whose ``request_id`` has
        gone stale (131 such drops in run 007 alone) and then re-requests,
        which burned the cached pick and dropped the re-ask to the rarity
        sort. All 7 Hidden Daggers second picks in `logs/f7-mirror-check16`
        went that way -- every one of them with the answer already computed,
        the multiset matching, and `the second pick of the same search`
        logged one line above.
        """
        held = self._picks
        if held is None:
            return None
        # An ABSENT source is not a DIFFERENT source. `Consume()` nulls
        # SourceModel and fires once per CardSelectCmd, while a multi-pick
        # selection asks for advice once per PICK -- so pick 2 always arrives
        # with source_model=None and `or ""` made that compare unequal to
        # everything. The multiset check below is the real safety property
        # (the world's offer must be the original minus what was served);
        # the source only ever ruled out a selection wearing another's name.
        now_source = str(source_model or "").strip().upper()
        if now_source and now_source != held["source"]:
            self._picks = None
            return None
        plan = held["plan"]
        expected = list(held["offer"])
        now = [self._identity(c) for c in offered]
        # How many the WORLD has actually taken, read off the offer rather
        # than off our own bookkeeping.
        taken = len(expected) - len(now)
        if taken < 0 or taken >= len(plan):
            # The offer grew, or the plan is spent and this is a new screen.
            self._picks = None
            return None
        for ident in plan[:taken]:
            if ident not in expected:
                self._picks = None
                return None
            expected.remove(ident)
        if sorted(expected) != sorted(now):
            # The world did not take what was recommended, or this is a
            # different selection wearing the same source.
            self._picks = None
            return None
        want = plan[taken]
        if want not in now:
            self._picks = None
            return None
        return now.index(want)

    def _probe_turn_end(
        self,
        mirror: CombatMirror,
        action: dict[str, Any],
    ) -> tuple[tuple[str, list[Any]] | None, str | None]:
        """End the turn on a CLONE and report ``(park kind, living enemy HP)``.

        Prove it on a clone first. Unlike a card or a drink, ``end_turn``
        CANNOT fail to apply -- it just runs the whole turn boundary -- so
        committing first and checking after would leave the mirror a full round
        past the screen with no way back.

        Returns:
            ``(probed, reason)``. Exactly one is populated: ``probed`` is the
            park kind and the probe's living-enemy HP vector, ``reason`` is the
            decline text to record. Both ``None`` means the probe landed but
            its board could not be read, which is a debug line and not a
            decline -- the historical behaviour of this arm.

            The reason is RETURNED rather than recorded because
            :meth:`replay_end_turn` probes twice at a retain park, and a repair
            that fails must decline once, not twice.
        """
        import sts2_rs

        try:
            probe = mirror.state.clone()
            parked = probe.apply_step(action)
        except Exception as exc:
            logger.debug("advisor mirror: the turn would not end: %s", exc)
            return None, "the turn would not end in the mirror"
        if parked != "pending_choice":
            return None, f"the turn ended but parked no choice ({parked})"
        kind = str((probe.pending_choice() or {}).get("kind") or "")
        if kind not in _END_TURN_PARKS:
            # Some other end-of-turn screen. Not answered here: the gate that
            # sent us down this path only knows the source was a power, and a
            # park of an unexpected kind is exactly the case where that is not
            # enough.
            return None, f"the turn parked on {kind or 'an unnamed choice'}"
        try:
            return (kind, _enemy_hp(sts2_rs.state_to_dict(probe))), None
        except Exception as exc:
            logger.debug("advisor mirror: cannot read the probe's board: %s", exc)
            return None, None

    def replay_end_turn(
        self,
        world_wire: dict[str, Any],
        *,
        on: CombatMirror | None = None,
    ) -> bool:
        """End the fork's turn to reach a screen the END OF TURN opened.

        The third replay kind, and the only one with no name to match. A card
        and a potion are replayed by re-issuing the action that opened the
        screen; here that action is ``end_turn``, which is unconditionally
        legal in the play phase -- so "which one was it" has no candidates and
        the identity proof the other two need has nothing to prove.

        What replaces it is stricter. Ending a turn normally runs the flush,
        the enemy turn and the next turn-setup and comes back to rest, so
        PARKING is not the common case: it happens only where the engine's own
        end-of-turn pipeline surfaced a player choice. "It parked, and on a
        kind this boundary is known to raise" is therefore the engine's own
        evidence about what the screen is -- not a name match against
        ``WELL_LAID_PLANS_POWER``.

        Why this class was invisible: the retain screen opens INSIDE the
        end-of-turn flush, before the round increments, and
        :meth:`_catch_up_turns` only ends the mirror's turn once the world's
        ``round`` has moved. So at the screen the mirror is still in the play
        phase and never parked itself. Live run 001-silent-a8-0825-0104: 146
        declines, every one of them this power, against 122 ranked decisions
        in the whole run -- 77% of everything the advisor had to rank.

        ``end_turn`` runs past the boundary and into the NEXT turn's setup, so
        a turn-START screen is this path's too. Tools of the Trade is the one
        that matters: its power draws +1 and then opens a ``FromHandForDiscard``
        inside ``AfterPlayerTurnStart`` (``ToolsOfTheTradePower.cs:27-37``), and
        the engine parks ``discard_from_hand`` there. Accepting only the retain
        kind sent 9 of them to the heuristic in the ``logs/f7-mirror-check15``
        corpus.

        ``TO_DISCARD`` in those logs is the PROMPT KEY, not the source, and
        ``FromHandForDiscard`` has two unrelated callers. The corpus's other 4
        are cards opening the same screen from their own ``OnPlay``
        (HiddenDaggers, Survivor, DaggerThrow, Prepared); those are
        :meth:`replay_opening_play`'s and are NOT fixed by this path.

        Args:
            world_wire: The world's own board, as the match gate. Comparable
                at either park point, because the probe stops exactly where
                the engine does: at a retain prompt neither side has run the
                enemy turn, at a turn-start discard both have. At the retain
                prompt it is also the REPAIR: a mismatch there is drift the
                turn end cannot have caused, so the fork is reconciled against
                this wire and re-probed (GH #593).
            on: The mirror to replay into -- a fork, in the advisor lane.

        Returns:
            True when the mirror is now parked on the end-of-turn choice.
        """
        mirror = on if on is not None else self.live
        if mirror is None or not mirror.alive:
            self.decline("no live mirror")
            return False
        if mirror.state.pending_choice() is not None:
            # Already parked -- nothing to replay, so nothing to disqualify.
            return True
        if mirror.state.has_pending_play():
            # A card is suspended mid-play, so the turn is not ending and
            # whatever opened this screen fired from inside that play. Ending
            # the turn here would be inventing a board.
            self.decline("a card is mid-play, so the turn is not ending")
            return False

        action = {"action": "end_turn"}
        self.stats["replays_attempted"] += 1
        probed, reason = self._probe_turn_end(mirror, action)
        if probed is None:
            if reason is not None:
                self.decline(reason)
            return False
        kind, mine = probed
        try:
            theirs = _enemy_hp(world_wire)
        except Exception as exc:
            logger.debug("advisor mirror: cannot compare boards at turn end: %s", exc)
            return False

        if mine != theirs and kind == _RETAIN_PARK:
            # GH #593. At THIS park the proof cannot be reporting what it
            # reads as, so believing it throws the screen away for nothing.
            #
            # The retain prompt opens inside the flush: C# runs
            # `Hook.BeforeTurnEnd` -> `DoTurnEnd` -> `Hook.BeforeFlush` ->
            # `BeforeFlushLate` (`CombatManager.cs:1182-1207`,
            # `WellLaidPlansPower.cs:23`), and the Rust pin fires
            # `before_turn_end` in phase ONE, ahead of the phase-two prompt
            # (`combat_loop.rs:2113`, `:2711`). So the probe comes to rest
            # with `round` still N and the phase still the player's play
            # phase, and NOTHING on either engine has touched an enemy since
            # the fork was made. Whatever `mine != theirs` is, the turn end
            # did not do it -- the fork already disagreed with the world.
            #
            # That is drift, and drift is `reconcile`'s job. It even already
            # does it: `pick_card_select_mirrored` reconciles this same fork
            # against this same wire one call later. Only the ORDER was wrong
            # -- by then the replay has declined, the fork is unparked, and
            # `inject_choice` has no park to resolve the world's option list
            # against, which is the second of the two consecutive declines in
            # `logs/f7-mirror-check19/001-silent-a8-0826-1421.advisor.log`:
            #
            #   searched card_select declined (the turn ended but the board did not match)
            #   searched card_select declined (injection dropped the fork (NPlayerHand))
            #
            # So reconcile HERE and probe again. The rebuild is not a way of
            # making the proof pass: it adopts the world's own observables and
            # transplants the mirror's hidden state (draw order, RNG cursors)
            # onto them, so what the search then ranks is the world's board
            # with the fight's memory -- which is what a reconciled fork is
            # everywhere else in this lane.
            #
            # Retain-park ONLY, and that is not caution, it is correctness:
            # `discard_from_hand` sits on the FAR side of the round increment,
            # so reaching it ran the flush, the whole enemy turn and the next
            # turn's setup. There the board proof grades a simulation the
            # world may genuinely have played differently, and a wire the
            # world took a full turn to reach would seed the fork a turn ahead
            # of where the replay must start. That proof stays strict.
            logger.info(
                "advisor mirror: end-turn board proof failed at the retain park, "
                "mirror enemy hp %s vs world %s -- reconciling the drift and re-probing",
                mine,
                theirs,
            )
            try:
                mirror.reconcile(world_wire)
            except MirrorDivergence as exc:
                # STRICT raises rather than reseeding. Not fatal here: the
                # fall-through below declines exactly as it did before, and
                # the fork is dropped by the caller either way.
                logger.info("advisor mirror: the retain-park reconcile diverged: %s", exc)
            except Exception as exc:
                logger.warning("advisor mirror: the retain-park reconcile failed: %s", exc)
            else:
                probed, reason = self._probe_turn_end(mirror, action)
                if probed is None:
                    if reason is not None:
                        self.decline(reason)
                    return False
                kind, mine = probed

        if mine != theirs:
            # Say WHAT disagreed, not just that something did: a length change
            # means an enemy died on one side only, a value change at equal
            # length means the two engines ordered the turn's damage
            # differently. Past the repair above this is the turn-start park's
            # decline -- the retain park reaches it only when the reconcile
            # could not repair the fork at all.
            logger.info(
                "advisor mirror: end-turn board proof failed, mirror enemy hp %s vs world %s",
                mine,
                theirs,
            )
            self.decline("the turn ended but the board did not match")
            return False

        try:
            result = mirror.advance(action)
        except MirrorDivergence as exc:
            logger.info("advisor mirror dropped ending its turn to answer: %s", exc)
            self.drop("end-turn replay diverged")
            return False
        except Exception as exc:
            logger.warning("advisor mirror end-turn replay failed: %s", exc)
            self.drop("end-turn replay error")
            return False
        if result != "pending_choice":
            self.decline(f"the turn ended but parked no choice ({result})")
            return False
        self.stats["replays_landed"] += 1
        return True

    def replay_opening_play(
        self,
        source_model: str | None,
        world_wire: dict[str, Any],
        *,
        source_kind: str | None = None,
        target_index: Any = None,
        upgraded: Any = None,
        on: CombatMirror | None = None,
    ) -> bool:
        """Play the card that opened a select screen, so the engine parks it.

        The autopilot's mirror is parked on the choice because it *played the
        card itself*; the advisor only learns a screen is up, so the play has
        to be reconstructed -- and then PROVEN, not assumed.

        How much has to be reconstructed depends on what the mod tells us.

        **Told (current builds).** ``source_kind`` says what the source IS by
        runtime type, and for a card ``target_index`` / ``upgraded`` say which
        play it was. Then there is exactly one candidate, the board check
        VERIFIES it, and a mismatch means the mirror is wrong -- which is a
        drop, not a decline.

        **Not told (older builds).** Fall back to proving it from the board:
        replay every legal play of that card on a clone and keep the one whose
        enemy-HP vector matches the world's. That settles which copy and which
        enemy at once, but it is inference: a damage prediction off by any
        amount matches nothing, and the screen goes unanswered
        (``no board matched replay of NEOWS_FURY``, live run
        001-ironclad-a8-0824-2150). Kept only for compatibility.

        A source that is not a card -- a potion (``TouchOfInsanity`` opens a
        ``FromHand`` select with ``source: this``), a power
        (``WELL_LAID_PLANS_POWER``), a relic -- has nothing to replay and is
        left to the unparked recognition in
        ``combat_mirror.recognize_card_select``, which already handles it.

        Args:
            source_model: The world's name for the effect that opened the
                screen (``CardSelectContextPatch``), when it captured one.
            world_wire: The wire built from the screen's embedded combat --
                the board AFTER the play, used as the match target.
            source_kind: ``"card"`` / ``"potion"`` / ``"power"`` / ... from the
                model's runtime type. ``None`` on an older mod build, which
                falls back to inferring the kind from the piles.
            target_index: The play's target as an index into the full enemy
                list, or ``None`` for an untargeted card.
            upgraded: Whether the copy played was upgraded.

        Returns:
            True when the mirror is now parked on the engine's own choice,
            with the suspended tail intact.
        """
        # `on` is the FORK the advisor answers into, so nothing here can
        # damage the live fight. Falling back to the live mirror keeps the
        # older single-mirror call shape working (and the tests that use it).
        mirror = on if on is not None else self.live
        src = str(source_model or "").strip().upper()
        kind = str(source_kind or "").strip().lower()
        if mirror is None or not mirror.alive:
            self.decline("no live mirror")
            return False
        if mirror.state.pending_choice() is not None:
            # Already parked (a multi-screen card, a re-ask, or a screen the
            # engine opened by itself), so there is nothing to replay -- and
            # therefore nothing the source's KIND could disqualify. Asking
            # "is this replayable?" first threw away the best case there is.
            #
            # WellLaidPlans is the example that made this expensive: the power
            # retains a card at end of turn, the engine models that
            # (`pending_wlp_retain_resume` -> `retain_at_turn_end`), and a
            # mirror that followed the turn end is sitting on exactly the
            # right choice. It was declined anyway as "a power, not a card" --
            # 246 of the 3,020 in-combat selections in the 82-run
            # `logs/2026-08-12-live-a8` corpus, 8x every potion combined.
            return True
        if not self.source_is_replayable(source_model, source_kind):
            return False

        import sts2_rs

        try:
            before = sts2_rs.state_to_dict(mirror.state)
        except Exception as exc:
            logger.debug("advisor mirror: cannot read state for replay: %s", exc)
            return False
        hand = (before.get("deck") or {}).get("hand") or []
        candidates = [i for i, c in enumerate(hand) if str(c.get("id", "")).upper() == src]
        if not candidates:
            # Two very different things, and the tally is where they get told
            # apart -- it is the only diagnostic surface this rung has.
            #
            # NOT A CARD: a potion (TouchOfInsanity opens a `FromHand` select
            # with `source: this`), a power (StratagemPower), a relic. Normal,
            # nothing to replay, handled downstream by the unparked
            # recognition in `combat_mirror.recognize_card_select`.
            #
            # A CARD THE MIRROR MISPLACED: the id is somewhere in this fight's
            # deck but not in hand, which means the mirror's picture is wrong
            # and the decline is a symptom, not a category.
            #
            # These shared one message until a live run declined
            # `TOUCH_OF_INSANITY not in the mirror's hand` and read as a
            # mirror fault when it was a potion behaving exactly as designed.
            deck = before.get("deck") or {}
            elsewhere = any(
                str(c.get("id", "")).upper() == src
                for key in ("draw_pile", "discard_pile", "exhaust_pile", "play_pile")
                for c in (deck.get(key) or [])
            )
            self.decline(
                f"{src} is in the fight but not in hand" if elsewhere else f"{src} is not a card"
            )
            return False

        want = _enemy_hp(world_wire)
        # The engine's own legal-action list settles which cards take a target
        # and which enemies are hittable: an untargeted card gets exactly one
        # action per copy (`target_index` None), a targeted one gets an action
        # per live enemy. Enumerating `[None, *range(n)]` by hand instead made
        # every untargeted play look like several different plays.
        try:
            legal = [
                a
                for a in mirror.state.legal_actions()
                if a.get("action") == "play" and int(a.get("card_index", -1)) in set(candidates)
            ]
        except Exception as exc:
            logger.debug("advisor mirror: cannot enumerate plays: %s", exc)
            return False
        if not legal:
            return False

        # When the wire named the play, keep only that one. `target_index` is
        # authoritative including its absence: None means UNTARGETED, so it
        # must match the engine's own None rather than be treated as "unknown".
        told = kind == "card" and (target_index is not None or upgraded is not None)
        if told:
            want_target = int(target_index) if isinstance(target_index, int) else None
            named = [
                a
                for a in legal
                if a.get("target_index") == want_target
                and (
                    upgraded is None
                    or bool(hand[int(a["card_index"])].get("upgraded")) == bool(upgraded)
                )
            ]
            if named:
                legal = named[:1]
            else:
                # The wire named a play the engine does not offer. That COULD
                # mean the mirror is wrong -- but it could equally mean the two
                # sides disagree about the naming itself, and the difference is
                # not decidable from here. Live run 001-ironclad-a8-0824-2305
                # produced exactly one of these, on a Sculpting Strike: an
                # ordinary targeted attack naming itself through the DECLARED
                # `source` argument, i.e. the path that should be least
                # ambiguous.
                #
                # So DEGRADE, do not destroy. The naming is an optimisation
                # over the board proof, and when it does not fit, the honest
                # move is to fall back to the proof -- which either finds the
                # play anyway or declines harmlessly. Dropping a mirror on an
                # unconfirmed premise is the failure this rung was just fixed
                # for; it must not be reintroduced from the other side.
                self.decline(f"{src} play named by the wire is not legal in the mirror")
                told = False
        self.stats["replays_attempted"] += 1

        # Two copies of one card in hand are interchangeable, but they carry
        # distinct uids, so `state_hash` would call their identical outcomes
        # different. What genuinely separates two matches is the target and
        # whether the copy played was upgraded — Burning Pact and Burning
        # Pact+ draw different amounts, and neither shows in the enemy-HP
        # filter. Anything else is one play spelled twice.
        matches: dict[Any, dict[str, Any]] = {}
        for action in legal:
            idx = int(action["card_index"])
            key = (action.get("target_index"), bool(hand[idx].get("upgraded")))
            if key in matches:
                continue
            try:
                probe = mirror.state.clone()
                if probe.apply_step(action) != "pending_choice":
                    continue
                if _enemy_hp(sts2_rs.state_to_dict(probe)) == want:
                    matches[key] = action
            except Exception:
                continue  # not a play that could have opened this screen
            if len(matches) > 1:
                break

        if len(matches) != 1:
            if told:
                # Reached only when the named play WAS legal here and was the
                # single candidate replayed. So this is no longer "which play
                # was it?" -- it is "my copy of the fight does not reproduce
                # the real board" after replaying exactly what the world says
                # it did. That is a mirror fault and it gets dropped, so the
                # next board re-seeds clean, instead of being logged as an
                # unanswerable screen and left to rot.
                self.decline(f"{src} replayed but the board did not match")
                self.drop("replay did not reproduce the world board")
                return False
            # Not named: zero means the world's board is not reachable by
            # replaying this card, more than one means the target is
            # undetermined and guessing it would hand the search a board with
            # the wrong enemy hurt. Either way, say nothing.
            self.decline(f"{'ambiguous' if matches else 'no board matched'} replay of {src}")
            return False

        try:
            result = mirror.advance(next(iter(matches.values())))
        except MirrorDivergence as exc:
            logger.info("advisor mirror dropped replaying %s: %s", src, exc)
            self.drop("replay diverged")
            return False
        except Exception as exc:
            logger.warning("advisor mirror replay of %s failed: %s", src, exc)
            self.drop("replay error")
            return False
        if result != "pending_choice" or self.live is None:
            self.decline(f"{src} replayed but parked no choice ({result})")
            return False
        self.stats["replays_landed"] += 1
        # The world will report this play LATE, after the screen is answered
        # and after this mirror has been dropped and re-seeded past it. Note
        # which card, so the echo is skipped instead of dropping a good
        # mirror. See :meth:`_expect_stale_report`.
        played = next(iter(matches.values()))
        self._stale_report = (
            (src, bool(hand[int(played["card_index"])].get("upgraded"))),
            self._generation,
        )
        return True

    def answer_card_select(
        self,
        state: dict[str, Any],
        cfg: RecommendConfig | None,
        sims: int | None,
    ) -> dict[str, Any] | None:
        """Answer an IN-COMBAT ``card_select`` with the combat MCTS, or give up.

        The autopilot answers this exact screen by search
        (``realgame_mcts.pick_card_select_mirrored``); what it has is a mirror
        already parked on the choice with the rest of the card suspended behind
        it. This method puts the advisor in the same position: it picks the
        replay that re-parks a FORK of the carried fight on the world's choice
        (already parked / end of turn / a potion / the opening play), searches
        it, drops the fork, and remembers the picks for a multi-pick follow-up.
        One code path with the driver, no second dialect to drift.

        Returns ``None`` for every reason not to answer -- no embedded combat
        (a deck screen), no live mirror, an unreplayable source, a fork that
        could not be taken, or a search that failed -- each one counted as a
        decline so the stat line tells the whole story. The caller then falls to
        its heuristic rung.
        """
        if cfg is None:
            return None
        if state.get("combat") is None:
            # A deck screen, not a fight. Normal, and not this rung's business --
            # counted so it never gets mistaken for a fight the mirror missed.
            self.decline("no embedded combat (deck screen)")
            return None
        live = self.live
        if live is None:
            self.decline("no live mirror at the screen")
            return None
        # Is the fight ALREADY on this choice? Then no replay is needed, and what
        # opened the screen does not matter -- the mirror followed the world into
        # the same place by itself. This is the best case the rung has and it used
        # to be unreachable, because the replayability gate ran first and answered
        # a question nobody had asked.
        #
        # WellLaidPlans is the case that made it expensive: the power retains a
        # card at end of turn, the engine models that (`pending_wlp_retain_resume`
        # -> `retain_at_turn_end`), and a mirror that followed the turn end is
        # sitting on exactly the right choice. It was declined as "a power, not a
        # card" -- 246 of the 3,020 in-combat selections in the 82-run
        # `logs/2026-08-12-live-a8` corpus, 8x every potion combined.
        #
        # A park left over from something else is not a new risk: `inject_choice`
        # grades the engine's own option list against the world's, so a stale one
        # is caught there, exactly as it already is for a card.
        # Pick 2 of a multi-pick selection: already searched, on the screen that
        # offered the whole list. Checked before anything else -- it needs no
        # wire, no fork and no replay, and the replay is what fails here.
        remembered = self.next_remembered_pick(state.get("source_model"), state.get("cards") or [])
        if remembered is not None:
            cards = state.get("cards") or []
            labels = [str(c.get("name") or c.get("id") or f"card {i}") for i, c in enumerate(cards)]
            self.stats["searched_selects"] += 1
            return {
                "note": "combat MCTS · the second pick of the same search",
                "options": [
                    {
                        "label": label,
                        "is_best": i == remembered,
                        "index": int(cards[i].get("index", i)),
                        "action": "choose",
                    }
                    for i, label in enumerate(labels)
                ],
                "best": {
                    "action": "choose",
                    "index": int(cards[remembered].get("index", remembered)),
                    "label": labels[remembered],
                },
            }
        already_parked = live.state.pending_choice() is not None
        # A screen the END OF TURN opened has no source that can be re-issued: a
        # power is not something the engine can be told to do again. `end_turn`
        # is, and reaching the same park by ending the turn is the proof --
        # :meth:`AdvisorMirror.replay_end_turn` requires the engine to stop by
        # itself, on one of the kinds this boundary is known to raise
        # (`_END_TURN_PARKS`). "End of turn" is the whole transition, so the
        # turn-START screens on the far side of the round increment -- Tools of
        # the Trade's discard -- come down this path too.
        #
        # Narrow on purpose. `power` excludes the relic / affliction /
        # enchantment sources, which fire from hooks and need injection rather
        # than a replay; `has_pending_play()` excludes a power hook that fired
        # from inside a card play, where the turn is plainly not ending. What
        # slips through both is caught by `inject_choice`'s option grading, the
        # same net that covers a stale park.
        end_of_turn = (
            not already_parked
            and str(state.get("source_kind") or "").strip().lower() == "power"
            and not live.state.has_pending_play()
        )
        # A screen a HOOK opened has no source to re-issue either, and unlike the
        # end-of-turn case there is no transition to reproduce: the relic fired
        # where the fork already stands. Gambling Chip is the whole reason this
        # branch exists -- its start-of-combat mulligan is `source_kind == "relic"`,
        # and it was 15 of the 17 remaining unsearched selections in
        # `logs/f7-mirror-check19` (every one of them answered by the rarity
        # sort). The machinery to answer it was already complete on the DLL
        # driver's side: `realgame_mcts.recognize_card_select` maps the mulligan
        # to `discard_from_hand_and_draw`, `inject_choice` injects it, and
        # `arm_turn_start_resume` parks the rest of the turn-start batch behind
        # the pick. Only this gate stood in front of it, declining the screen
        # before a fork was ever built.
        #
        # No replay is the POINT, not a shortcut: `replay_opening_play` proves the
        # picture by re-issuing the play, and there is no "play the relic" action
        # to issue. The proof here is `inject_choice`'s option grading -- the same
        # net the `power` branch above leans on -- and an injection the engine
        # cannot locate drops the FORK and returns `picks is None`, which is
        # already handled below as a decline. So the worst case is where we
        # started: an unsearched screen, minus one fork.
        hook_sourced = not already_parked and str(
            state.get("source_kind") or ""
        ).strip().lower() in (
            "relic",
            "affliction",
            "enchantment",
        )
        # Cheapest check first, and board-free: a screen opened by something that
        # can never be replayed must not cost a wire build, a fork, or -- as it
        # used to -- the fight's mirror.
        if (
            not already_parked
            and not end_of_turn
            and not hook_sourced
            and not self.source_is_replayable(state.get("source_model"), state.get("source_kind"))
        ):
            return None
        # Answer into a FORK, never the live fight. Everything below is
        # destructive by design -- reconcile, replay the opening play, inject the
        # world's option list -- and it leaves the mirror parked on a choice only
        # the party that EXECUTES the picks can resolve. A driver executes; the
        # advisor never does. Doing it on a throwaway makes this whole rung
        # side-effect free, which it was not: an injection the engine cannot
        # locate drops the mirror outright, on exactly the screens the advisor was
        # always going to decline. Live run 001-ironclad-a8-0824-2239, three
        # consecutive lines: `DROPLET_OF_PRECOGNITION is a potion, not a card`
        # (an honest decline) -> `injection dropped the mirror` -> `no live mirror
        # at the screen` when the same screen was re-asked. A screen nobody could
        # answer cost the fight its mirror, and every decision after it.
        fork = live.fork("advisor-answer")
        if fork is None:
            self.decline("could not fork the mirror to answer")
            return None

        import realgame_mcts
        from sts2_hint import adapter

        try:
            headless = to_headless_combat_play(state["combat"])
            # Resolve the character the way `compute_combat_advice` does, instead
            # of passing `cfg.character` straight through. `serve()` builds its
            # RecommendConfig with no character at all, so that field is None on
            # this path and `adapter.translate` falls back to inferring the class
            # from card-id prefixes -- which works on nearly every board and fails
            # on the ones that have no prefixed card. Live run
            # 002-silent-a8-0825-0259 hit `Scare | Corrosive Wave | Shadowmeld`
            # and lost 12 selections to
            # `$.player.class: could not infer character class from state`.
            run_state = state.get("run_state") or (state.get("combat") or {}).get("run_state") or {}
            character = character_display_name(run_state.get("character")) or cfg.character
            wire = adapter.translate(
                headless,
                character=character,
                combat_id=cfg.combat_id,
                seed=cfg.seed,
                ascension=run_ascension(run_state, cfg.ascension),
            )
            # Park the engine on its OWN choice first, so `inject_choice` carries
            # the `post_effect` forward (Burning Pact's Draw 2, Brand's Strength).
            # Without it the search ranks the pick against a truncated card --
            # a confident wrong answer, which is worse than an admitted arbitrary
            # one. `parked` False is not a failure: the unparked screens
            # (Gambling Chip's mulligan, the generative offers) are recognized
            # downstream and have no tail to lose.
            if already_parked:
                # The fork inherited the park; replaying anything on top of it
                # would be wrong, not merely redundant.
                pass
            elif hook_sourced:
                # Nothing to re-issue: the hook already fired on the state the
                # fork inherited. Straight to the injection, which is where the
                # proof lives for this class. See `hook_sourced` above.
                pass
            elif end_of_turn:
                self.replay_end_turn(wire, on=fork)
            elif str(state.get("source_kind") or "").strip().lower() == "potion":
                # A potion opens its screen through a different engine action, so
                # it needs a different replay -- but the same proof. The four
                # generative potions (Attack/Skill/Power/Colorless) plus
                # LiquidMemories, DropletOfPrecognition, TouchOfInsanity and
                # GamblersBrew all park a real choice the engine models, and every
                # one of them used to be answered by the rarity sort.
                self.replay_opening_drink(
                    state.get("source_model"),
                    wire,
                    target_index=state.get("source_target_index"),
                    on=fork,
                )
            else:
                self.replay_opening_play(
                    state.get("source_model"),
                    wire,
                    # The mod's own answer to "what opened this, and which play
                    # was it". `source_kind` is the model's runtime type, so the
                    # client never infers "is this a card?" from pile membership;
                    # `source_target_index` is `CardModel.CurrentTarget` read
                    # while the play is still in flight, which is the one fact the
                    # late `human_action` report cannot deliver in time. Absent on
                    # an older mod build, and the replay falls back to proving it
                    # from the board.
                    source_kind=state.get("source_kind"),
                    target_index=state.get("source_target_index"),
                    upgraded=state.get("source_upgraded"),
                    on=fork,
                )
            action, picks = realgame_mcts.pick_card_select_mirrored(state, cfg, fork, sims=sims)
        except Exception as exc:
            # The FORK failed, so the live fight is not implicated and is left
            # alone. Only a proven-wrong picture drops it, and `replay_opening_play`
            # does that itself where it can prove one.
            #
            # COUNTED, not just logged. Every other way this rung gives up records
            # a decline, so the mirror's own stat line reads as the whole story --
            # and this arm made it lie. Run 002-silent-a8-0825-0259 reported
            # `6/6 replays` with no declines at all while nine of its fourteen
            # ranked decisions had fallen to the rarity sort through here.
            logger.warning("searched card_select failed, falling back: %s", exc)
            self.decline(f"the search could not run ({type(exc).__name__})")
            return None
        finally:
            # The fork is done either way: it is parked on an injection nothing
            # will ever resolve, which is precisely why it was a fork.
            fork.drop("answered a card_select")
        if picks is None:
            # `pick_card_select_mirrored` dropped the FORK (an injection the
            # engine could not locate) and returned a minimum-legal selection,
            # which is not a recommendation. The live mirror is untouched.
            self.decline(f"injection dropped the fork ({state.get('screen')})")
            return None

        cards = state.get("cards") or []
        labels = [str(c.get("name") or c.get("id") or f"card {i}") for i, c in enumerate(cards)]
        if not labels:
            self.decline("screen offered no labelled cards")
            return None
        self.stats["searched_selects"] += 1
        self.remember_picks(state.get("source_model"), cards, picks)
        best_i = picks[0] if picks else None
        options = [
            {
                "label": label,
                "is_best": i in picks,
                "index": int(cards[i].get("index", i)) if i < len(cards) else i,
                "action": "choose",
            }
            for i, label in enumerate(labels)
        ]
        note = f"combat MCTS · {len(picks)} pick(s) searched"
        if len(picks) > 1:
            note += " (one search per pick)"
        best = (
            {"action": "choose", "index": options[best_i]["index"], "label": labels[best_i]}
            if best_i is not None and best_i < len(options)
            # A skippable selection the search answered with NOTHING -- legal, and
            # the right answer where `min_select` is 0 (a Well-Laid Plans retain
            # the search would rather decline). Labelled, because the advisor logs
            # the label and an unlabelled best prints as `best=None`, which is
            # exactly how the floor-47 wedge disguised itself.
            else {"action": action.get("action", "skip"), "label": "select none"}
        )
        return {"note": note, "options": options, "best": best}

    def note(self) -> str:
        """One-line state summary for the advice note / run log."""
        s = self.stats
        why = ", ".join(f"{k} x{v}" for k, v in sorted(self.declines.items()))
        top = sorted(self.desync_paths.items(), key=lambda kv: (-kv[1], kv[0]))[:6]
        paths = ", ".join(f"{k} x{v}" for k, v in top)
        top_adopted = sorted(self.adopted_paths.items(), key=lambda kv: (-kv[1], kv[0]))[:6]
        adopted = ", ".join(f"{k} x{v}" for k, v in top_adopted)
        return (
            f"mirror: {s['reconciles']} reconciles, {s['desyncs']} desyncs, "
            f"{s['followed']} plays + {s['turns_followed']} turns followed "
            f"({s['unfollowable']} not), "
            f"{s['replays_landed']}/{s['replays_attempted']} replays, "
            f"{s['searched_selects']} searched selects"
            + (
                f", {s['stale_reports_skipped']} echoes skipped"
                if s["stale_reports_skipped"]
                else ""
            )
            + (f" | declined: {why}" if why else "")
            + (f" | desynced: {paths}" if paths else "")
            + (f" | teammate-adopted: {adopted}" if adopted else "")
        )
