"""PpoRunEnv — decision-stepper environment over the sts2_rs PyO3 bindings.

Plan: sts2-rs `docs/plans/2026-07-03-ppo-fullrun-plan.md`, milestone M-P1
(bead sts2-rs-2wvkf).

The env plays one full run per `reset()`:

- `Phase::Combat` decisions are resolved internally by a combat MCTS agent
  (eval-cheap 1000/4000 — combat stays AS-IS).
- Single-option decisions are auto-taken (any policy returns the only index).
- Every multi-option non-combat decision is *yielded* to the caller as a
  `Decision` (surface kind, candidate action dicts, observation, metrics
  snapshot); the caller answers with `step(idx)`.

Two drive modes, selected by the `expert` construction flag:

- **Policy mode** (`expert=False`, the default — `reset()` / `step(idx)`):
  the caller is the policy (the PPO net samples an index per decision). The
  combat agent is the combat-only `MctsCombatAgent` — combat is a
  heuristic-leaf MCTS and every meta decision is yielded to the net, so no
  net file or env var is needed.
- **Expert mode** (`expert=True` — `expert_run()`): an expert agent makes
  *every* decision. This is the `HeuristicRunAgent` (character-agnostic
  tier-1 heuristic meta over the same MCTS combat); it loads no net and
  needs no env vars. Each yielded decision carries the expert's chosen
  index. Used for the G1 parity gate (env-driven expert must reproduce the
  arena agent's outcomes per seed) and for BC warm-start recording (M-P4).

Because both agents wrap the identical `MctsCombatPolicy` combat search at
the same seed/budget, a policy-mode probe reproduces the expert agent's
per-seed combat trajectories exactly.

Construction mirrors `run_arena`'s `run_cell` exactly (Neow start, shared
`initialize_run_setup`, and — since GH #399 — the four deck-mutation flags
`interactive_card_removal` / `interactive_card_upgrade` /
`interactive_card_transform` / `direct_event_relic_grant` ON, so removal /
upgrade / transform / enchant reach the policy as decisions instead of being
auto-resolved; and — since GH #379 — `interactive_potion_swap`, so a
full-belt potion reward becomes a discard-to-claim decision instead of
evaporating into `potions_overflow_dropped`). Both sides must move together
or G1 breaks.

Ownership audit (G1): every decision point is logged with a resolution
class (`net` / `single-option-auto` / `combat-agent` / `expert`); there is
no silent-fallback class by construction — anything not combat and not
single-option is yielded. `resolution_counts()` feeds the audit table.
Complementing it, `assert_encodable` (GH #399) refuses any non-combat
candidate whose kind the encoder cannot represent, so a newly-reachable
surface can never arrive as a list of blank tokens.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

# Mirrors run_arena's FullRunConfig default (arena.rs: max_steps 3000).
DEFAULT_MAX_STEPS = 3000

# Deck-mutation RunActions form a homogeneous card-pick list regardless of
# phase. The mirror is the DEPLOYED arm's own classifier, sts2-runai
# `ppo/policy.rs::classify_surface:77-95` (which is the authority for the PPO
# module and does include `GrantDeckCard`) — NOT `policy/ablation.rs::classify`,
# whose list predates `GrantDeckCard` and omits it.
_DECK_MUTATION_ACTIONS = frozenset(
    {
        "remove_deck_card",
        "upgrade_deck_card",
        "transform_deck_card",
        "enchant_deck_card",
        "grant_deck_card",
    }
)

_PHASE_SURFACE = {
    "map_choice": "path",
    "card_reward": "card-pick",
    "boss_relic": "card-pick",
    "event": "event",
    "shop": "shop",
    "rest_site": "rest",
    "treasure": "treasure",
}

# Base rest-site option order is always [Rest, Smith] (run_manager.rs);
# the engine exposes no labels (workbench does the same mapping).
REST_OPTION_LABELS = ("rest", "smith")


def classify_surface(phase: str, actions: list[dict[str, Any]]) -> str | None:
    """Surface class for a decision, or None (combat/over).

    Mirrors the deployed arm's own classifier, sts2-runai
    `ppo/policy.rs::classify_surface`: smith target and phase-agnostic
    deck-mutation lists are card picks; otherwise the phase decides.
    (`policy/ablation.rs::classify` has the same shape but an older, shorter
    deck-mutation list — it is not the authority for this module.)
    """
    if actions and actions[0]["action"] == "smith_card":
        return "card-pick"
    if actions and all(a["action"] in _DECK_MUTATION_ACTIONS for a in actions):
        return "card-pick"
    # A full-belt potion swap preempts the action surface in ANY phase
    # (`run_manager.rs:1010`), so it has to be claimed before the phase
    # lookup — otherwise it would be reported as whatever room it happened
    # to interrupt. GH #252/#379.
    if actions and all(a["action"] == "potion_swap" for a in actions):
        return "potion-swap"
    return _PHASE_SURFACE.get(phase)


def metrics_snapshot(rm: Any) -> dict[str, Any]:
    """Cheap per-decision metrics for reward shaping (fork GameMetrics)."""
    deck = rm.master_deck_detailed()
    return {
        # `rewards.total_floors` reads this: A10 attaches a second boss, so a
        # completed run is 46 floors there and 45 everywhere else (GH #464).
        "ascension": int(rm.ascension_level),
        "total_floor": int(rm.total_floor()),
        "act_index": int(rm.act_index),
        "act_floor": int(rm.act_floor()),
        "hp": int(rm.current_hp),
        "max_hp": int(rm.max_hp),
        "gold": int(rm.gold),
        "deck_size": len(deck),
        "num_upgraded": sum(1 for _, up in deck if up),
        "num_relics": len(rm.relics()),
        "num_potions": len(rm.potions()),
    }


# One loaded surrogate per (process, path). The weights JSON is tens of MB and
# an env is constructed per run, so reloading it per episode would hand back a
# large slice of the speedup the surrogate backend exists to buy.
_SURROGATE_CACHE: dict[str, Any] = {}


def _load_surrogate(path: str) -> Any:
    net = _SURROGATE_CACHE.get(path)
    if net is None:
        import sts2_rs

        net = sts2_rs.CombatSurrogate(path)
        _SURROGATE_CACHE[path] = net
    return net


def observe(
    rm: Any, phase: str, surface: str | None, *, seed: int = 0, ascension: int = 0
) -> dict[str, Any]:
    """Observation dict for the encoder. JSON-serializable throughout.

    Module-level so the workbench PPO option (``workbench/ppo_policy.py``)
    builds the EXACT training-time observation over its live ``RunManager``.
    Always includes run state (deck / relics / potions / scalars / current
    act map); adds surface-specific candidate detail the raw action dicts
    don't carry (shop prices, event descriptions, offered card ids, …).
    Only *player-visible* state: current act's map (never `act_map_for`
    on future acts — that would be clairvoyance).
    """
    obs: dict[str, Any] = {
        "phase": phase,
        "surface": surface,
        "seed": seed,
        "ascension": ascension,
        "deck": rm.master_deck_detailed(),  # [(card_id, upgraded), ...]
        # Index-aligned with `deck`: `None` or `(UPPER_SNAKE id, amount)`.
        # GH #947 — `deck` alone reads an enchanted card as the bare card.
        "deck_enchantments": rm.master_deck_enchantments(),
        "relics": rm.relics(),
        "potions": rm.potions(),
        "map": rm.full_map(),
        "map_coord": rm.map_coord(),
        **metrics_snapshot(rm),
    }
    if surface == "card-pick":
        obs["card_reward_option_ids"] = rm.card_reward_option_ids()
        obs["card_reward_option_upgraded"] = rm.card_reward_option_upgraded()
        obs["card_reward_option_enchantments"] = rm.card_reward_option_enchantments()
        obs["card_reward_bundles"] = rm.card_reward_bundles()
        obs["relic_reward_ids"] = rm.relic_reward_ids()
        obs["boss_relic_options"] = rm.boss_relic_options()
        obs["smith_options"] = rm.smith_options()
        obs["smith_option_enchantments"] = rm.smith_option_enchantments()
        obs["card_removal_options"] = rm.card_removal_options()
        obs["card_removal_option_enchantments"] = rm.card_removal_option_enchantments()
        obs["card_grant_options"] = rm.card_grant_options()
        obs["card_grant_option_enchantments"] = rm.card_grant_option_enchantments()
        obs["transform_select_deck_indices"] = rm.transform_select_deck_indices()
        obs["enchant_select_deck_indices"] = rm.enchant_select_deck_indices()
        obs["pending_enchantment"] = rm.pending_enchantment
        # GH #399 Task 5b: `upgrade_deck_card`'s `idx` indexes the engine's
        # `is_upgradable()` subset (curses / statuses / Quest cards excluded),
        # which the encoder cannot re-derive from `deck` — a `!upgraded` scan
        # over-offers and shifts every option after the first curse. Ship the
        # engine's own offer map, like transform/enchant above.
        obs["upgrade_select_deck_indices"] = rm.upgrade_select_deck_indices()
    elif surface == "shop":
        obs["shop"] = rm.shop_inventory()
    elif surface == "event":
        # Idempotent; matches what the lazy resolve path would pick, so
        # the run's RNG/event-cursor state is unchanged (run_manager.rs).
        rm.materialize_event()
        obs["event_name"] = rm.current_event_name()
        obs["event_options"] = rm.event_option_infos()
    elif surface == "rest":
        obs["rest_option_labels"] = list(REST_OPTION_LABELS)
    elif surface == "potion-swap":
        # The potion being OFFERED. Every option token names a potion you
        # would THROW AWAY (or the decline), so without this the decision is
        # unanswerable — the encoder puts it on the state token. GH #252/#379.
        obs["pending_potion_swap"] = rm.pending_potion_swap
    return obs


def _potion_swap_is_rule_side() -> bool:
    """Whether a surfaced full-belt swap is decided by the OC value rule.

    Delegates to `sts2_runai::policy::potion_swap_is_rule_side_from_env` so
    this and `MetaWithRules::choose` can never disagree — G1 pins the two
    together. Read at the decision rather than cached at construction: it is
    consulted only when the action set is actually a homogeneous swap (rare),
    and caching it would both go stale under a test that sets the var later
    and break any `PpoRunEnv` built without `__post_init__`.
    """
    import sts2_rs

    return sts2_rs.potion_swap_is_rule_side()


def assert_encodable(actions: list[dict[str, Any]], *, phase: str | None = None) -> None:
    """Raise unless every candidate's kind has an encoder surface (GH #399).

    `encoder._featurize_action` ends in `_fixed_tok("none")`: an unknown
    action kind becomes a *blank* token instead of a crash (encoder.py:390).
    That fallback stays — the encoder is the porting authority for the
    bit-for-bit Rust mirror — so the loudness has to live here. Without it a
    newly-reachable surface arrives as a list of identical blank tokens, the
    policy ranks noise, and every dashboard stays green: the #317 shape
    (events silently auto-resolved, `emb_event` drift 0.000000).

    Mirrors the deployed Rust arm, which encodes exactly the same kind set:
    `ActionDesc::try_from` is total over `_KIND_SURFACE` and `PpoPolicy`
    panics on anything else. (Until GH #379 that arm short-circuited a
    `PotionSwap`-headed list to the heuristic, because `potion_swap` had no
    `ActionDesc` tag; it now has one, so both sides encode it.) Rust can
    fall back mid-run; a training worker must not, so this raises and the
    run is fault-isolated as `outcome='error'` rather than training on
    noise.

    Combat kinds (`play_card`, `end_turn`, …) are deliberately absent from
    `_KIND_SURFACE` — combat belongs to the MCTS agent, never the run
    encoder — so the caller must apply this only to non-combat candidates.
    """
    # Imported lazily: `ppo.encoder` pulls in numpy and the `sts2_rs` PyO3
    # wheel (via `ppo.vocab`), and `ppo.env` stays stdlib-only at module
    # scope for wheel-free consumers (workbench policy shim).
    from .encoder import _KIND_SURFACE

    for act in actions:
        kind = act.get("action")
        if kind not in _KIND_SURFACE:
            where = f" (phase {phase!r})" if phase else ""
            raise RuntimeError(
                f"unencodable action kind {kind!r}{where}: it has no "
                "encoder._KIND_SURFACE entry, so the encoder would emit a "
                "blank 'none' token for every candidate and the policy would "
                "rank noise while the dashboards stayed green (GH #399/#317 "
                "family). Give the kind an ActionDesc tag + _KIND_SURFACE "
                "entry + encoder branch before making this surface reachable."
            )


def mask_rejectable(rm: Any, phase: str, actions: list[dict]) -> list[dict]:
    """Drop shop actions the engine accepts but silently no-ops.

    The engine already filters unaffordable/sold-out buys out of
    `available_actions()`, but a belt-full `shop_buy_potion` is a
    deliberate C#-lockstep graceful no-op (`do_shop_buy_potion`
    `PotionBeltFull => Advanced`): state doesn't change, so a greedy
    argmax policy deadlocks on it (observed: 24/256 greedy G4 runs
    burned to max_steps in shop potion loops). Expert mode is
    unaffected — the deployed agent never picks it.
    """
    if phase != "shop":
        return actions
    if len(rm.potions()) < int(rm.max_potion_slots()):
        return actions
    masked = [a for a in actions if a.get("action") != "shop_buy_potion"]
    return masked or actions  # never yield an empty set


@dataclass
class Decision:
    """One yielded multi-option non-combat decision."""

    surface: str
    phase: str
    actions: list[dict[str, Any]]
    obs: dict[str, Any]
    metrics: dict[str, Any]
    step_index: int
    # Expert mode only: the index the deployed agent chose (already applied).
    expert_idx: int | None = None


@dataclass
class Terminal:
    """End-of-run record."""

    outcome: str  # 'won' | 'lost' | 'max_steps' | 'no_actions' | 'agent_halt'
    won: bool
    metrics: dict[str, Any]
    steps: int
    decisions: int


@dataclass
class PpoRunEnv:
    """One full run, decision-stepped. Construct fresh per episode."""

    seed: int
    ascension: int = 0
    character: str = "Ironclad"
    sims: int = 1000
    boss_sims: int = 4000
    max_steps: int = DEFAULT_MAX_STEPS
    # Drive mode: False → policy mode (MctsCombatAgent, combat only; meta
    # decisions yielded to the caller). True → expert mode (HeuristicRunAgent
    # makes every decision). `expert_run()` requires this to be True (GH #220).
    expert: bool = False
    # Combat backend. "mcts" plays every fight with the combat MCTS — the
    # campaign behavior, and the only one a champion may be trained under.
    # "surrogate" fast-forwards each fight through the combat surrogate net at
    # `surrogate_path`: one forward pass per fight instead of a `sims`-budget
    # search per decision, which is where ~87% of campaign wall clock goes.
    # Screening only — `docs/ppo-screen-playbook.md` carries the
    # deck-sensitivity gate a surrogate result has to clear before it means
    # anything about a deck-shaped question. `sims`/`boss_sims` are inert
    # under "surrogate" (there is no search to budget).
    combat: str = "mcts"
    surrogate_path: str | None = None
    # GH #545: which Act-1 model this run walks — "overgrowth" (the single
    # act this engine modelled before the Underdocks port), "underdocks", or
    # "roll" (the real game's seed-driven pick, ~50/50).
    #
    # The DEFAULT stays "overgrowth" so every existing test, fixture and panel
    # keeps meaning what it meant. A campaign that wants Underdocks in its
    # training distribution passes "roll" here AND on the arena arm it is
    # compared against — an arena arm and a training env that disagree about
    # the act is the same silent-divergence class as the potion-swap flag,
    # which is why G1 (`tests/test_ppo_env_parity.py`) is parameterised over
    # this token.
    act1: str = "overgrowth"
    # GH #803: the game build this run simulates ("v0.107.1" / "v0.111.0").
    # None means the process-wide selection, `sts2_client.GAME_VERSION`
    # (`STS2_GAME_VERSION`, or `--game-version` on the entry point), which
    # defaults to v0.107.1, so every existing caller keeps its engine.
    game_version: str | None = None
    # Resolution log: (step_index, phase, surface|None, class) — the G1
    # ownership-audit record. Classes: 'combat-agent', 'combat-surrogate',
    # 'single-option-auto', 'net' (policy mode), 'expert' (expert mode).
    resolution_log: list[tuple[int, str, str | None, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        import sts2_client
        import sts2_rs

        # Parity-pinned construction (== run_arena `run_cell`, whose shared
        # `new_arena_run_manager` helper mirrors this call statement for
        # statement): Neow start, shared initialize_run_setup, deck-mutation
        # flags ON since GH #399 — the net must SEE removal / upgrade /
        # transform / enchant as decisions instead of the engine silently
        # auto-resolving them (front-of-deck removal, random upgrade, …).
        # `interactive_card_transform` gates the ENCHANT queue too: there is
        # no `interactive_card_enchant` flag — every `pending_deck_enchants`
        # push site is behind `interactive_card_transform` (e.g.
        # sts2-content/src/relics/shop_event.rs:2998,
        # events/shared_wave_b.rs:256 / :353 / :1724, events/act2_hive.rs:544,
        # events/act2_hive_wave_b.rs:474).
        # `interactive_potion_swap` comes from the SAME env-var resolver the
        # arena reads (`STS2_RUNAI_POTION_SWAP`), so training and arena
        # RunManagers can never disagree on the flag. Default OFF: the
        # deployed default and every parity construction stay byte-identical.
        #
        # WHO DECIDES a surfaced swap depends on `STS2_RUNAI_POTION_OC`:
        # with an OC table configured the Rust OC rule resolves it rule-side
        # (the #479 experiment arm, preserved exactly); without one it is a
        # normal encoded decision and the net picks (#379 gave it a real
        # `potion_swap` surface). `_advance` mirrors that split, and both
        # sides must agree or G1 (`tests/test_ppo_env_parity.py`) breaks.
        self.rm = sts2_client.rs_engine_call(
            sts2_rs.RunManager,
            version=self.game_version,
            seed=self.seed,
            character=self.character,
            ascension_level=self.ascension,
            neow_start=True,
            interactive_card_removal=True,
            interactive_card_upgrade=True,
            interactive_card_transform=True,
            direct_event_relic_grant=True,
            interactive_potion_swap=sts2_rs.resolve_potion_swap_flag(),
            # GH #545: the act this run walks. `arena::run_cell` takes the same
            # token from `--act1`, and G1 compares the two per token, so the
            # training env and the measurement arm cannot silently disagree.
            act1=self.act1,
        )
        # Combat agent. Policy mode (expert=False) uses the combat-only
        # MctsCombatAgent (combat MCTS only — every meta decision is yielded
        # to the caller; GH #220). It builds the same combat cell as the
        # expert agent (MctsCombatPolicy<HeuristicPolicy>), so per-seed combat
        # trajectories match exactly.
        #
        # Expert mode (expert=True) makes every meta decision itself, so it
        # needs a meta policy: the HeuristicRunAgent — the character-agnostic
        # tier-1 heuristic over MCTS combat (no net files, no env vars).
        # STS2_PPO_EXPERT=heuristic is accepted for back-compat; the removed
        # learned ranker ("ranker") raises.
        if self.expert:
            import os

            expert_kind = os.environ.get("STS2_PPO_EXPERT", "heuristic").lower()
            if expert_kind == "ranker":
                raise ValueError(
                    "STS2_PPO_EXPERT=ranker is no longer supported: the learned "
                    "pick-ranker was removed from sts2_rs (PPO replaced it). "
                    "Use STS2_PPO_EXPERT=heuristic (the default)."
                )
            if expert_kind != "heuristic":
                raise ValueError(
                    f"unknown STS2_PPO_EXPERT={expert_kind!r}; the only expert is 'heuristic'"
                )
            self.agent = sts2_rs.HeuristicRunAgent(
                seed=self.seed, sims=self.sims, boss_sims=self.boss_sims
            )
        else:
            self.agent = sts2_rs.MctsCombatAgent(
                seed=self.seed, sims=self.sims, boss_sims=self.boss_sims
            )
        if self.combat not in ("mcts", "surrogate"):
            raise ValueError(f"combat must be 'mcts' or 'surrogate', got {self.combat!r}")
        self._surrogate = None
        if self.combat == "surrogate":
            if self.expert:
                # The expert agent drives combat itself; there is no seam to
                # swap. Silently ignoring the backend here would make the G1
                # parity gate compare two different worlds.
                raise ValueError("combat='surrogate' is incompatible with expert=True")
            if not self.surrogate_path:
                raise ValueError("combat='surrogate' requires surrogate_path")
            self._surrogate = _load_surrogate(self.surrogate_path)
        self.steps = 0
        self.decisions = 0
        self._pending_actions: list[dict[str, Any]] | None = None

    # ── observation / metrics ────────────────────────────────────────────

    def metrics_snapshot(self) -> dict[str, Any]:
        """Cheap per-decision metrics for reward shaping (fork GameMetrics)."""
        return metrics_snapshot(self.rm)

    def observe(self, phase: str, surface: str | None) -> dict[str, Any]:
        """Observation dict for the encoder (see module-level :func:`observe`)."""
        return observe(self.rm, phase, surface, seed=self.seed, ascension=self.ascension)

    # ── drive loops ──────────────────────────────────────────────────────

    def _terminal(self, outcome: str) -> Terminal:
        won = bool(self.rm.player_won)
        return Terminal(
            outcome="won" if won else outcome,
            won=won,
            metrics=self.metrics_snapshot(),
            steps=self.steps,
            decisions=self.decisions,
        )

    def _mask_rejectable(self, phase: str, actions: list[dict]) -> list[dict]:
        return mask_rejectable(self.rm, phase, actions)

    def _outcome_base(self) -> int:
        """Seed for this fight's Bernoulli survival draw.

        Keyed on `(run seed, total_floor)`, NOT on the run seed alone: the
        Rust side draws `splitmix64(outcome_stream_seed_u64(base, 0, SALT))`
        with a fixed `outcome_n = 0`, so a per-run constant would hand every
        fight in the run the *same* uniform — the run would then either
        survive every marginal fight or die at the first one whose win_prob
        fell below that single draw, which is not a Bernoulli process at all.

        Keying on the floor also buys common random numbers across a paired
        screen: two arms that reach the same floor of the same seed draw the
        same bit, so a win-rate delta between them is not partly the survival
        RNG. Masked to u64 for the PyO3 boundary.
        """
        mixed = (self.seed * 0x9E37_79B9_7F4A_7C15) ^ (
            int(self.rm.total_floor()) * 0xBF58_476D_1CE4_E5B9
        )
        return mixed & 0xFFFF_FFFF_FFFF_FFFF

    def _advance(self, mode: str) -> Decision | Terminal:
        """Run until the next yielded decision or the end of the run.

        `mode` is the resolution class recorded for yielded decisions
        ('net' or 'expert' — expert mode applies the agent's choice itself).
        """
        rm = self.rm
        while True:
            if rm.is_over():
                return self._terminal("lost")
            if self.steps >= self.max_steps:
                return self._terminal("max_steps")
            phase = rm.phase()
            # GH #302: bind the event BEFORE the options are counted. The
            # engine offers a lone placeholder `event_choice` while
            # `current_event` is None, and `observe()` -- which is where
            # `materialize_event()` used to live -- runs only past the
            # `len(actions) == 1` auto-resolve below. So every ordinary
            # event room was swallowed as `single-option-auto` and never
            # became a decision: measured over 40 seeded runs, 22/22
            # unbound event rooms offered exactly 1 action while the 40
            # bound ones (Neow, pre-bound by `neow_start=True`) offered 3.
            # The PPO meta had therefore never made an ordinary event
            # choice -- `emb_event` moved 0.000000 across Regent's whole
            # it220->it800 leg, 3 of 71 rows touched (Neow, TinkerTime,
            # OOV). Binding here is idempotent and leaves the run's
            # RNG/event-cursor untouched (same call `observe()` makes),
            # and it matches the deployed Rust arm, whose
            # `PpoPolicy::prepare` (policy.rs:673) already binds before
            # `choose` -- so this closes a train/deploy asymmetry, not
            # just a coverage gap.
            if phase == "event":
                rm.materialize_event()
            actions = rm.available_actions()
            if not actions:
                return self._terminal("no_actions")
            actions = self._mask_rejectable(phase, actions)
            if phase == "combat":
                if self._surrogate is not None:
                    # One forward pass resolves the whole fight; see
                    # `_outcome_base` for how the survival draw is seeded. The
                    # fight still exits through the deployed reward bridge, so
                    # gold / potion / card rewards are generated exactly as if
                    # it had been played.
                    rm.surrogate_resolve_combat(self._surrogate, self._outcome_base())
                    self.steps += 1
                    self.resolution_log.append((self.steps, phase, None, "combat-surrogate"))
                    continue
                if self.agent.step(rm) is None:
                    return self._terminal("agent_halt")
                self.steps += 1
                self.resolution_log.append((self.steps, phase, None, "combat-agent"))
                continue
            # Full-belt potion swap: the engine emits ONLY `potion_swap`
            # actions while a swap is pending, regardless of phase.
            #
            # WITH an OC table configured, keep resolving it rule-side (plan
            # P2 / #479): route the set to the agent, whose inner Rust
            # `MctsCombatPolicy::choose` applies the SAME OC rule as the
            # deployed arena (evict argmin-OC iff the offer prices strictly
            # higher). This must mirror `policy/mod.rs`, which intercepts
            # `PotionSwap` above the meta policy on exactly the same
            # condition, or G1 diverges.
            #
            # WITHOUT one, fall through: #379 gave `potion_swap` a real
            # encoder surface, so it is an ordinary encoded decision and the
            # run policy picks. That is the arm the next campaign trains on
            # (#545) — the rule-side path exists to keep #479's measurement
            # reproducible, not because the net should not learn this.
            if (
                actions
                and all(a.get("action") == "potion_swap" for a in actions)
                and _potion_swap_is_rule_side()
            ):
                if self.agent.step(rm) is None:
                    return self._terminal("agent_halt")
                self.steps += 1
                self.resolution_log.append((self.steps, phase, "potion-swap", "potion-swap-rule"))
                continue
            # GH #399: from here on the candidates are the run policy's, and
            # anything without an encoder surface would be featurized as a
            # blank token (or auto-taken invisibly). Fail loud instead. Sits
            # after the combat dispatch (combat kinds are the agent's, not
            # the encoder's) and before the single-option auto-take, matching
            # where the Rust arm short-circuits (policy.rs:764-772).
            assert_encodable(actions, phase=phase)
            surface = classify_surface(phase, actions)
            if len(actions) == 1:
                rm.take_action(actions[0])
                self.steps += 1
                self.resolution_log.append((self.steps, phase, surface, "single-option-auto"))
                continue
            # Multi-option non-combat: yield to the caller.
            obs = self.observe(phase, surface)
            metrics = self.metrics_snapshot()
            self.decisions += 1
            self.resolution_log.append((self.steps, phase, surface, mode))
            if mode == "expert":
                rec = self.agent.step(rm)
                if rec is None:
                    return self._terminal("agent_halt")
                self.steps += 1
                expert_idx = _index_of_action(actions, rec["action"])
                return Decision(
                    surface=surface or "unknown",
                    phase=phase,
                    actions=actions,
                    obs=obs,
                    metrics=metrics,
                    step_index=self.steps,
                    expert_idx=expert_idx,
                )
            self._pending_actions = actions
            return Decision(
                surface=surface or "unknown",
                phase=phase,
                actions=actions,
                obs=obs,
                metrics=metrics,
                step_index=self.steps,
            )

    def reset(self) -> Decision | Terminal:
        """Advance to the first yielded decision (policy mode)."""
        return self._advance("net")

    def step(self, idx: int) -> Decision | Terminal:
        """Apply candidate `idx` of the pending decision, advance (policy mode)."""
        if self._pending_actions is None:
            raise RuntimeError("step() without a pending decision — call reset() first")
        actions = self._pending_actions
        if not 0 <= idx < len(actions):
            raise IndexError(f"candidate idx {idx} out of range 0..{len(actions) - 1}")
        self._pending_actions = None
        result = self.rm.take_action(actions[idx])
        if result in ("shop_purchase_failed", "shop_remove_card_failed", "invalid_for_phase"):
            # Tripwire: masking should make this unreachable; a silent
            # no-op here becomes a greedy-policy deadlock, so fail loud
            # (workers fault-isolate the run as outcome='error').
            raise RuntimeError(f"engine rejected action {actions[idx]!r}: {result}")
        self.steps += 1
        return self._advance("net")

    def expert_run(self) -> Iterator[Decision | Terminal]:
        """Drive the whole run with the deployed agent, yielding every
        multi-option non-combat decision (with `expert_idx` set) and finally
        the `Terminal`. The G1 parity + ownership driver and the BC recorder.

        Requires `expert=True` construction: expert mode's meta decisions are
        made by the HeuristicRunAgent's meta policy, which the combat-only
        policy agent cannot reproduce (GH #220).
        """
        if not self.expert:
            raise RuntimeError(
                "expert_run() requires PpoRunEnv(expert=True): the default "
                "policy-mode agent (MctsCombatAgent) has no meta policy "
                "and cannot make the expert's meta decisions."
            )
        while True:
            out = self._advance("expert")
            yield out
            if isinstance(out, Terminal):
                return

    # ── search-fork support (GH #452) ────────────────────────────────────

    @classmethod
    def from_fork(
        cls,
        rm: Any,
        *,
        like: PpoRunEnv,
        sims: int,
        boss_sims: int,
        max_steps: int = DEFAULT_MAX_STEPS,
        agent_seed: int | None = None,
    ) -> PpoRunEnv:
        """Wrap an already-constructed (cloned) RunManager in a fresh env
        facade, reusing ``_advance()``/``step()`` semantics over the clone.

        The search wrapper (``ppo.search_policy``) clones the live run's rm
        (``rm.clone()`` + ``rm.reseed_run()`` for anti-clairvoyance) and
        drives the copy to its next strategic decision. The fork gets its
        OWN (cheap) combat agent — sharing the parent's would advance the
        parent's MCTS rng state. Always policy-mode, always real-MCTS
        combat: a fork exists to measure the consequence of one candidate,
        so it must play the same game the parent does.
        """
        import sts2_rs

        env = object.__new__(cls)
        env.seed = like.seed
        env.ascension = like.ascension
        env.character = like.character
        env.sims = sims
        env.boss_sims = boss_sims
        env.max_steps = max_steps
        env.expert = False
        env.combat = "mcts"
        env.surrogate_path = None
        env._surrogate = None
        env.resolution_log = []
        env.rm = rm
        env.agent = sts2_rs.MctsCombatAgent(
            seed=like.seed if agent_seed is None else agent_seed,
            sims=sims,
            boss_sims=boss_sims,
        )
        env.steps = like.steps
        env.decisions = like.decisions
        env._pending_actions = None
        return env

    # ── audits ───────────────────────────────────────────────────────────

    def resolution_counts(self) -> dict[tuple[str, str | None, str], int]:
        """(phase, surface, class) -> count, for the G1 ownership table."""
        counts: dict[tuple[str, str | None, str], int] = {}
        for _, phase, surface, klass in self.resolution_log:
            key = (phase, surface, klass)
            counts[key] = counts.get(key, 0) + 1
        return counts


def _index_of_action(actions: list[dict[str, Any]], applied: dict[str, Any]) -> int:
    """Index of the expert-applied action dict within the candidate list.

    `run_action_to_pydict` is deterministic, so dict equality identifies the
    action. Raises if absent — that would mean the agent chose an action not
    in `available_actions()`, which the parity gate must surface, not mask.
    """
    for i, a in enumerate(actions):
        if a == applied:
            return i
    raise RuntimeError(f"expert action {applied!r} not in available_actions ({actions!r})")
