//! Expert-knowledge play-ordering for heuristic MCTS rollouts.
//!
//! Bead `sts2-rs-dfv1`. Port of the priority bucket table from
//! `sts_lightspeed`:
//!   * `include/sim/search/ExpertKnowledge.h`
//!   * `src/sim/search/ExpertKnowledge.cpp::getPlayOrdering`
//!
//! Lower priority numbers play earlier; very high values (~10_000)
//! collapse the softmax weight to ~zero, making the rollout policy avoid
//! curses, EndTurn, and unknown cards while still allowing them as a
//! tail in case nothing else is legal.
//!
//! Only STS2 [`CardId`] variants that have a known C++ counterpart get a
//! tuned priority. Every other card lands on the neutral default
//! ([`NEUTRAL_PRIORITY`]) so the heuristic degrades to "play whatever's
//! legal, prefer non-curses" on cards we haven't classified yet.

use rand::Rng;

use sts2_core::cards::CardId;
use sts2_core::{Action, CharacterId, CombatState};

pub use crate::enemy_eval::IncomingDamage;

/// Default priority for cards we haven't classified. Mid-bucket — same
/// neighborhood as Iron Wave / Sentinel in the C++ table.
pub const NEUTRAL_PRIORITY: i16 = 50;

/// Coverage-fraction threshold below which the Heuristic rollout
/// degrades to Uniform (bead `sts2-rs-j0f6`). When fewer than 80% of
/// the legal `PlayCard` actions in the current state reference a card
/// whose [`play_ordering`] has been explicitly classified, the STS1-ported
/// priority table is dominating the softmax with a stale signal — the
/// uncovered cards collapse to [`NEUTRAL_PRIORITY`] and the heuristic
/// effectively picks randomly among the covered subset. On random-deck
/// scenarios (Silent / Defect / Necrobinder / Regent pools) this
/// produces a 6pp winrate regression vs uniform-random. Falling back
/// to Uniform when coverage is low restores the baseline.
pub const COVERAGE_THRESHOLD: f32 = 0.8;

/// Priority bucket assigned to [`Action::EndTurn`]. Very high so the
/// softmax weight collapses near zero and rollouts only end the turn
/// when no card play is available.
pub const END_TURN_PRIORITY: i16 = 10_000;

/// Softmax temperature for [`pick_action_heuristic`]. The plan calibrates
/// this against the C++ priority-bucket spread (~-30..+70 for in-play
/// cards). `tau = 32` keeps a mild bias toward higher-priority cards
/// while leaving enough rollout variance for UCT exploration to matter.
///
/// Bead `sts2-rs-3f74`: the original `tau = 8` was over-sharp at the
/// 90-point STS1 priority spread — covered cards with extreme priorities
/// (Apotheosis -20, Slimed +70) dominated the softmax to near-determinism,
/// while ~50% of STS2 Ironclad cards (BloodWall, EvilEye, FightMe, Pyre,
/// Stoke, etc.) fell through to [`NEUTRAL_PRIORITY`] = 50 and were
/// systematically de-weighted next to a covered Strike (55 vs the rare
/// Apotheosis at -20). The combination Q-saturated UCT exploration too
/// fast and regressed winrate by ~6pp vs uniform on a 10-scenario × 10-seed
/// random-Ironclad sweep at sims=1000 / depth=10. `tau = 32` flattens the
/// softmax enough to recover (53/90 vs uniform 52/90 on the same grid)
/// without losing the ordering signal at the dedup-sort step.
pub const SOFTMAX_TAU: f32 = 32.0;

/// Lethal-turn priority for Defend-class cards (bead `sts2-rs-wver`,
/// hoisted from 5dw7 + it3o magic numbers). When [`HpAwareCtx::is_lethal_turn`]
/// fires, [`action_priority`] returns this value for Defend-class cards
/// — sitting below the entire play_ordering range so Defend dominates
/// the rollout softmax (~94% weight vs Strike at tau=32).
pub const LETHAL_DEFEND_PRIORITY: i16 = -5;

/// Lethal-turn penalty for Strike-class basics (bead `sts2-rs-wver`,
/// hoisted from 5dw7 + it3o magic numbers). Symmetric to
/// [`LETHAL_DEFEND_PRIORITY`]: pushes Strike well above the status-card
/// bucket so a dying-turn Strike costs ≥100 priority vs Defend.
pub const LETHAL_STRIKE_PRIORITY: i16 = 90;

/// GH #293: ceiling of the GRADED defensive bonus subtracted from a
/// block-bearing card's [`play_ordering`] priority, scaled by how much of
/// the player's effective HP this turn's telegraphed damage represents.
///
/// Before this existed the rollout had exactly two defensive modes, both
/// cliffs on *am I about to die*: [`HpAwareCtx::is_lethal_turn`] /
/// [`HpAwareCtx::is_sustained_pressure`] fired, or nothing did. And when
/// nothing did, a basic Defend and a basic Strike both scored
/// `play_ordering` **55** — the rollout was *exactly indifferent* between
/// blocking a telegraphed hit and swinging. Measured over the mirror
/// corpora, that indifferent regime covered 86–98% of the states facing a
/// telegraphed attack (`sts2-pycli/python/diag_rollout_defense.py`), so the
/// search never blocked for HP efficiency — only to avoid dying. Paired
/// with a leaf that cannot price block either (median `tanh'` ≈ 0.07, see
/// `diag_leaf_saturation.py`), *nothing* in the search preferred defence.
///
/// See [`DEFEND_THREAT_MAX_BONUS_CONTINUOUS`] for the size that makes the
/// ramp meet the cliff.
///
/// **The default is 0 — the ramp ships OFF.** The seed-matched A/B
/// (`docs/experiments/2026-08-05-defend-gradient-ab.md`, n=24 full Regent
/// runs per arm) came back null on every axis: wins 12/24 → 10/24 (5 vs 3
/// discordant seeds, McNemar p ≈ 0.73), and paired same-floor HP-lost per
/// fight moved −0.98 (act 1, t = −1.51), −0.80 (act 2, t = −0.67), +0.12
/// (act 3, t = +0.08). What little direction there is sits in act 1, while
/// #293's bleed lives in act 3. Shipping 60 would push an unmeasured
/// behaviour change into every rollout, so the deployed default is
/// byte-identical to pre-#293 and `STS2_RUNAI_DEFEND_GRADIENT=60` turns the
/// ramp on for anyone continuing the study at a usable n.
pub const DEFEND_THREAT_MAX_BONUS_DEFAULT: i16 = 0;

/// The ceiling at which the GH #293 ramp is *continuous with the cliff it
/// extends*: a base-55 card at full threat scores `55 - 60 = -5`, exactly
/// [`LETHAL_DEFEND_PRIORITY`], so nothing jumps as a turn crosses into
/// lethal. Below full threat it interpolates — a boss telegraphing 24 into
/// 50 effective HP gives `24·60/50 = 28`, so Defend scores 27 against
/// Strike's 55, about 2.4× the softmax weight at `tau = 32`. A nudge, not an
/// override.
///
/// This is the value to pass to `STS2_RUNAI_DEFEND_GRADIENT` when enabling
/// the ramp; it is NOT the shipped default (see
/// [`DEFEND_THREAT_MAX_BONUS_DEFAULT`] for why).
pub const DEFEND_THREAT_MAX_BONUS_CONTINUOUS: i16 = 60;

/// Swept ceiling for the GH #293 defensive ramp.
/// `STS2_RUNAI_DEFEND_GRADIENT=<i16>` overrides
/// [`DEFEND_THREAT_MAX_BONUS_DEFAULT`].
///
/// **`0` disables the ramp entirely** — the bonus computes to 0, the guard in
/// [`play_card_priority`] skips the arm, and behaviour is byte-identical to
/// the pre-#293 rollout. That is the A/B baseline arm, so the comparison runs
/// in one binary instead of two wheel builds.
///
/// An invalid value **panics** rather than silently falling back, mirroring
/// [`sts2_core`]'s `hp_weight` — a typo must not quietly poison an arm.
/// Read once and cached.
#[must_use]
pub fn defend_threat_max_bonus() -> i16 {
    static BONUS: std::sync::OnceLock<i16> = std::sync::OnceLock::new();
    *BONUS.get_or_init(|| match std::env::var("STS2_RUNAI_DEFEND_GRADIENT") {
        Err(_) => DEFEND_THREAT_MAX_BONUS_DEFAULT,
        Ok(s) => {
            let v: i16 = s.parse().unwrap_or_else(|_| {
                panic!(
                    "STS2_RUNAI_DEFEND_GRADIENT={s:?} is not a valid i16 — refusing to \
                     silently fall back and poison the A/B arm"
                )
            });
            assert!(
                v >= 0,
                "STS2_RUNAI_DEFEND_GRADIENT={v} must be non-negative"
            );
            v
        }
    })
}

/// Bonus subtracted from an attack card's [`play_ordering`] priority
/// when its [`Action::PlayCard`] target slot equals
/// [`HpAwareCtx::lowest_hp_enemy_target`] (bead `sts2-rs-6g96`).
/// Modest — biases the softmax toward finishing the lowest-hp enemy
/// without overriding the lethal-turn / scaling-power signals. A
/// Strike (55) targeting the kill candidate scores 45, comfortably
/// ahead of the same Strike (55) targeting a healthier enemy.
pub const FOCUS_FIRE_PRIORITY_BONUS: i16 = 10;

/// Bead `sts2-rs-d9qd` (zlhm.1): turn-horizon for the "sustained
/// pressure" predicate that extends 5dw7's single-turn-lethal Defend
/// bump to the multi-turn-lethal case. Fires when the player would
/// die within `SUSTAINED_PRESSURE_HORIZON` turns of unblocked incoming
/// AND can't kill the weakest alive enemy this turn (so the offensive
/// counter-play of "drop one threat, reduce next-turn incoming" isn't
/// available). Set to 2: the smallest horizon that captures elite
/// fights like PHANTASMAL_GARDENERS where round-1 effective HP (47)
/// is above single-turn incoming (40) but below 2-turn incoming (80).
pub const SUSTAINED_PRESSURE_HORIZON: i32 = 2;

/// Bead `sts2-rs-pifyc` Part 2: priority bump applied to debuff-
/// attaching attacks (Bash, Thunderclap, Uppercut) when the target
/// survives this hit AND has not yet been debuffed (Vulnerable
/// inactive or about to expire) AND the player has at least one
/// follow-up attack of equal-or-higher damage in hand. Pushes the
/// debuff-attaching attack one "step" earlier in the heuristic
/// ordering (subtracted from its baseline `play_ordering` priority),
/// so it gets played before the follow-up landing damage onto a
/// vulnerable target instead of after it.
///
/// Sized at 2 — small enough that the existing Bash (25) vs Strike
/// (55) gap dwarfs it (Bash is preferred over Strike regardless),
/// but enough to break the priority tie inside the cluster of
/// debuff-attaching attacks around their shared bucket. The 2026-
/// 05-21 strike_when_bash_affordable blunder set (5 cases) showed
/// the bonus is needed only when the gates above hold; the gates
/// exclude the "1-shot kill via Strike" line (target_hp ≤ damage
/// short-circuits) and the "already-debuffed" line (Vulnerable
/// would double-stack pointlessly).
pub const DEBUFF_SETUP_PRIORITY_BONUS: i16 = 2;

/// GH #403: priority bonus for a card whose on-play rider is gated on
/// "fewer than N cards played this turn" — currently only FTL (`Deal 5/6
/// damage. If you have played fewer than 3/4 cards this turn, draw 1`).
///
/// The window is *self-closing*: every other card played this turn burns
/// one of its slots, so the rider is free upside if the card goes early and
/// gone forever if it goes late. The static bucket ordering points the
/// wrong way for that — FTL sits at 40 (AOE / cheap utility) while ordinary
/// attacks sit at 35 and free attacks at 30, so the rollout softmax
/// systematically defers it.
///
/// Sized at 20 — enough to clear both of those buckets, landing
/// FTL-with-a-live-window at 20, the cheap-cantrip bucket (Soul / Spur /
/// Transfigure / Wisp), which is what a 0-cost draw-a-card attack actually
/// is. It stays above [`LETHAL_DEFEND_PRIORITY`] so a dying turn still
/// blocks first, and it is a softmax nudge, not an override.
///
/// This is the *minor* half of the #403 fix. Rollout ordering barely moves
/// the decision, because the root returns `argmax(Q)` and smarter rollouts
/// improve every subtree equally: over 300 MCTS-driven fights on a normal
/// Act-1 mix it took the in-window rate from 73.7% to only 75.3%.
/// [`DRAW_WINDOW_ROOT_Q_BONUS`] is what carries the fix. The bonus is kept
/// anyway because it is correct on its own terms — a rollout that defers
/// FTL is misplaying the card, and the combined arm measured best on both
/// in-window rate and mean HP.
pub const DRAW_WINDOW_PRIORITY_BONUS: i16 = 20;

/// GH #403: root-side Q bonus applied to a `PlayCard` whose play-count-gated
/// draw rider is still live. See [`draw_window_root_q_bonus`] for why the
/// rollout-side [`DRAW_WINDOW_PRIORITY_BONUS`] cannot do this job alone, why
/// a UCT edge prior cannot either, and for the sizing rationale.
pub const DRAW_WINDOW_ROOT_Q_BONUS: f32 = 0.05;

/// The `PlayMax` threshold for cards whose on-play rider fires only while
/// *fewer than* that many cards have finished playing this turn, or `None`
/// for every other card.
///
/// C# `Ftl.cs:28-35` — `CardPlaysFinished.Count(this turn, this owner) <
/// DynamicVars["PlayMax"].IntValue`, with `PlayMax` 3 base and `OnUpgrade`
/// bumping it to 4. FTL itself has not finished playing during its own
/// `OnPlay`, so the threshold reads the *pre-FTL* count — matching the
/// engine-side gate in `sts2-content::cards::defect::ftl::play_ftl`.
///
/// Only the "fewer than" direction belongs here. The play-count readers
/// pointing the other way (Finisher / Juggling / LunarBlast / Conflagration
/// scale *up* with cards already played) want to be played LATE and would
/// need the opposite sign.
#[must_use]
const fn play_count_gated_draw_max(card_id: CardId, upgraded: bool) -> Option<u32> {
    match card_id {
        CardId::Ftl => Some(if upgraded { 4 } else { 3 }),
        _ => None,
    }
}

/// Would `card`'s play-count-gated draw rider still fire if it were played
/// right now? Three gates, all required:
/// 1. the card has such a rider ([`play_count_gated_draw_max`]);
/// 2. the window is still open — strictly fewer than `PlayMax` cards have
///    finished playing this turn;
/// 3. there is actually something to draw. With both the draw and discard
///    piles empty the rider is a no-op, so promoting the card would be a
///    free behaviour change for zero value. (A non-empty discard counts:
///    drawing from an empty pile reshuffles it.)
///
/// This predicate is deliberately separate from the two magnitudes it
/// gates ([`DRAW_WINDOW_PRIORITY_BONUS`], [`DRAW_WINDOW_ROOT_Q_BONUS`]):
/// folding it into either one couples the two knobs, so zeroing one to
/// A/B it silently disables the other.
#[must_use]
pub fn draw_window_is_live(state: &CombatState, card: &sts2_core::cards::Card) -> bool {
    let Some(play_max) = play_count_gated_draw_max(card.card_id, card.upgraded) else {
        return false;
    };
    state.count_player_cards_played_this_turn(None) < play_max
        && !(state.player.draw.is_empty() && state.player.discard.is_empty())
}

/// [`DRAW_WINDOW_PRIORITY_BONUS`] when [`draw_window_is_live`] holds for
/// `card`, `0` otherwise.
#[must_use]
pub fn draw_window_priority_delta(state: &CombatState, card: &sts2_core::cards::Card) -> i16 {
    if draw_window_is_live(state, card) {
        DRAW_WINDOW_PRIORITY_BONUS
    } else {
        0
    }
}

/// Return the expert play priority for `card_id` under the given
/// `character`. Lower means "play earlier" (better card to play first);
/// higher means "play later" (worse, prefer skipping). Returns
/// [`NEUTRAL_PRIORITY`] for cards not in the C++ table.
///
/// The buckets mirror `sts::search::Expert::getPlayOrdering` at
/// `sts_lightspeed/src/sim/search/ExpertKnowledge.cpp:9-199`. STS1-only
/// cards that have no STS2 counterpart are simply absent from the
/// match. STS2-only cards fall through to `NEUTRAL_PRIORITY` until a
/// future tuning pass classifies them.
///
/// Per-character overrides (bead `sts2-rs-a79i`, bottled_ai audit
/// 2026-05-18 §4): when `character` is `Some(c)`,
/// [`per_character_override`] is consulted first and, if it returns
/// `Some(p)`, that priority shadows the shared table. The override map
/// is currently empty — this bead lifts the signature so future
/// character-specific tuning (Ironclad-aggressive, Silent-shiv,
/// Defect-orb, …) can land as one-line additions to that helper without
/// touching every call site again.
///
/// # Example
///
/// ```
/// use sts2_core::cards::CardId;
/// use sts2_mcts::expert::{play_ordering, NEUTRAL_PRIORITY};
///
/// // Bash is in the priority table (bucket 25 — high-tempo setup
/// // attack) and ranks well below the neutral fallback.
/// assert!(play_ordering(CardId::Bash, None) < NEUTRAL_PRIORITY);
///
/// // Apotheosis is at the ladder floor (-20).
/// assert_eq!(play_ordering(CardId::Apotheosis, None), -20);
///
/// // Per-character argument: at this pin every CardId returns the
/// // same priority across every character (the override map is empty).
/// use sts2_core::CharacterId;
/// assert_eq!(
///     play_ordering(CardId::Bash, Some(CharacterId::Ironclad)),
///     play_ordering(CardId::Bash, None),
/// );
/// ```
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn play_ordering(card_id: CardId, character: Option<CharacterId>) -> i16 {
    if let Some(c) = character {
        if let Some(p) = per_character_override(card_id, c) {
            return p;
        }
    }
    play_ordering_base(card_id)
}

/// Per-character priority override. Returns `Some(p)` when the
/// `(character, card_id)` pair has a tuned override, `None` to fall
/// through to [`play_ordering_base`]. Currently empty by design —
/// bead `sts2-rs-a79i` ships the lift; future beads file the
/// overrides themselves (e.g. Ironclad-aggressive damage promotion,
/// Silent-shiv stack chaining, Defect-orb channel priority).
///
/// Cross-pollinated from bottled_ai's per-strategy
/// `move_in_comparison_list` / `add_to_comparison_list` pattern
/// (`bottled_ai/rs/common/comparators/common_general_comparator.py`);
/// see `docs/perf/2026-05-18-bottled-ai-feature-audit.md` §4.
#[must_use]
const fn per_character_override(_card_id: CardId, _character: CharacterId) -> Option<i16> {
    None
}

/// Shared base priority table — the existing STS1-derived ladder, used
/// when no per-character override applies.
#[must_use]
#[allow(clippy::too_many_lines)]
fn play_ordering_base(card_id: CardId) -> i16 {
    match card_id {
        // -20: free / cycle utility
        // GH #515: Fuel (0-cost Exhaust: +1 energy, draw 1) and Luminesce
        // (0-cost Exhaust+Retain: +2 energy) are pure free value with no
        // opportunity cost — the Apotheosis tier by construction.
        CardId::Apotheosis | CardId::Apparition | CardId::Fuel | CardId::Luminesce => -20,

        // -15: card pre-play setup (Armaments upgrades)
        CardId::Armaments => -15,

        // -10: turn-bonus power that wants to fire first
        CardId::CorruptionCard => -10,

        // -5: power-priority A — top-tier 0-cost powers
        CardId::PanacheCard | CardId::RuptureCard | CardId::DrumOfBattleCard => -5,

        // -2: cheap powers / power-class scalers wanting to fire first.
        // Bead `sts2-rs-g198` (3f74 followup): pull the STS2 Ironclad
        // power pool out of NEUTRAL — Aggression/Cruelty/Juggling/Vicious
        // are draw/scaling powers analogous to DarkEmbrace; CrimsonMantle
        // is a startup power. Cross-character first-pass: Silent
        // NoxiousFumes/Envenom and Defect BiasedCognition/MachineLearning
        // are pure scalers, in the same bucket as DarkEmbrace.
        //
        // Bead `sts2-rs-g8sy4` (GitHub issue #8): pull the rest of the
        // STS2 cross-character "pure scaler" powers out of NEUTRAL —
        // Silent AfterimageCard / FanOfKnives / PhantomBlades stack
        // per-card riders analogous to DarkEmbrace; Defect Hailstorm
        // / ConsumingShadow / BulkUp are orb/armor scalers in the same
        // tier; Silent MasterPlanner is a draw scaler that pays back
        // over the whole combat.
        CardId::DarkEmbraceCard
        | CardId::FeelNoPainCard
        | CardId::Bloodletting
        | CardId::Offering
        | CardId::AggressionCard
        | CardId::CrueltyCard
        | CardId::JugglingCard
        | CardId::ViciousCard
        | CardId::CrimsonMantle
        | CardId::NoxiousFumesCard
        | CardId::EnvenomCard
        | CardId::BiasedCognitionCard
        | CardId::MachineLearningCard
        | CardId::AfterimageCard
        | CardId::FanOfKnivesCard
        | CardId::PhantomBladesCard
        | CardId::MasterPlanner
        | CardId::Hailstorm
        | CardId::ConsumingShadow
        | CardId::BulkUp
        // g8sy4: Caltrops is a Power (Thorns 3) — scaling defense, fire early.
        | CardId::Caltrops
        // g8sy4 colorless straggler pass: BrightestFlame is a 0-cost
        // cycle+energy burst (draw 2 / +2 energy / -1 max HP), Feeding
        // Frenzy is a 0-cost self-Power scaler (FeedingFrenzy 5), and
        // Hello World is a 1-cost self-Power scaler (HelloWorld 1).
        // All three want to land early like the other 0/1-cost
        // scalers in this bucket.
        | CardId::BrightestFlame
        | CardId::FeedingFrenzyCard
        | CardId::HelloWorldCard => -2,

        // 0: late-game-defining powers — play if available, but they're
        // not the only thing keeping us alive.
        // g198: HellraiserCard / Pyre / StampedeCard / TankCard are
        // Ironclad's DemonForm-class scaling powers; Silent
        // AccuracyCard / InfiniteBladesCard / SerpentFormCard and Defect
        // CreativeAi / EchoForm / Loop / Defragment are their analogues.
        //
        // Bead `sts2-rs-g8sy4` (GitHub issue #8): WraithForm is the
        // Silent equivalent of DemonForm — Intangible scaling that
        // defines the late game once landed; Colorless CalamityCard /
        // EternalArmor / RollingBoulder are 3-cost rare powers in the
        // same combat-defining tier.
        CardId::BarricadeCard
        | CardId::DemonFormCard
        | CardId::HellraiserCard
        | CardId::Pyre
        | CardId::StampedeCard
        | CardId::TankCard
        | CardId::AccuracyCard
        | CardId::InfiniteBladesCard
        | CardId::SerpentFormCard
        | CardId::CreativeAiCard
        | CardId::EchoFormCard
        | CardId::LoopCard
        | CardId::Defragment
        | CardId::NecroMasteryCard
        | CardId::WraithForm
        | CardId::CalamityCard
        | CardId::EternalArmor
        | CardId::RollingBoulder => 0,

        // 5: situational utilities + lesser powers.
        // g198: StoneArmor is a defensive power, Cascade is 0-cost
        // utility. Cross-character: Footwork (Silent dex power),
        // BufferCard / Capacitor / IterationCard / ThunderCard (Defect
        // utility powers).
        //
        // Bead `sts2-rs-g8sy4` (GitHub issue #8): WellLaidPlans is a
        // retain power (Silent), Coolant is an orb-utility power
        // (already at 10 above); pull additional Silent / Defect
        // utility powers into this band — Tracking / SneakyCard
        // (Silent debuff/Stealth scalers), Subroutine / StormCard
        // (Defect orb-trigger powers).
        CardId::SecretTechnique
        | CardId::SecretWeapon
        | CardId::StoneArmor
        | CardId::Cascade
        | CardId::Footwork
        | CardId::BufferCard
        | CardId::Capacitor
        | CardId::IterationCard
        | CardId::ThunderCard
        | CardId::Abrasive
        | CardId::ToolsOfTheTrade
        | CardId::WellLaidPlans
        | CardId::Tracking
        | CardId::SneakyCard
        | CardId::Subroutine
        | CardId::StormCard => 5,

        // 10: standard powers / debuffs / cheap block powers.
        // g198: Unmovable / PrimalForce / DemonicShield / Brand are
        // additional Ironclad utilities; Smokestack / Coolant /
        // TrashToTreasure / SpinnerCard are Defect block/utility powers.
        //
        // Bead `sts2-rs-g8sy4` (GitHub issue #8): the remaining
        // standard-power / debuff bucket for Silent, Defect, Colorless.
        // Silent: DeadlyPoison / PiercingWail / BubbleBubble / Haze /
        // Flanking / Malaise / CorrosiveWave (debuffs); OutbreakCard /
        // SpeedsterCard / Accelerant (utility powers). Defect: Feral
        // (Strength). Colorless: Automation / Fasten / PrepTime /
        // Prowess / Stratagem / BeaconOfHope / Entropy / MayhemCard /
        // NostalgiaCard (standard powers).
        CardId::DarkShackles
        | CardId::Finesse
        | CardId::BattleTrance
        | CardId::Enlightenment
        | CardId::Inflame
        | CardId::Impervious
        | CardId::Shockwave
        | CardId::Unmovable
        | CardId::PrimalForce
        | CardId::DemonicShield
        | CardId::Brand
        | CardId::Smokestack
        | CardId::Coolant
        | CardId::TrashToTreasure
        | CardId::SpinnerCard
        | CardId::DeadlyPoison
        | CardId::PiercingWail
        | CardId::BubbleBubble
        | CardId::Haze
        | CardId::Flanking
        | CardId::Malaise
        | CardId::CorrosiveWave
        | CardId::OutbreakCard
        | CardId::SpeedsterCard
        | CardId::Accelerant
        | CardId::Feral
        | CardId::Automation
        | CardId::Fasten
        | CardId::PrepTime
        | CardId::Prowess
        | CardId::Stratagem
        | CardId::BeaconOfHope
        | CardId::Entropy
        | CardId::MayhemCard
        | CardId::NostalgiaCard
        // g8sy4 colorless straggler pass: ToricToughness is a 2-cost
        // skill that gains `base_block` (5) block — modest standard-
        // tier block, same bucket as the other 10-tier block cards.
        | CardId::ToricToughness
        // GH #515: Scare is a 0-cost Exhaust skill applying Weak 1 to every
        // enemy — the cheap-debuff shape this bucket is for.
        | CardId::Scare => 10,

        // 15: medium-impact mid-bucket (incl. Ironclad Stoke).
        CardId::JuggernautCard | CardId::RageCard | CardId::Stoke => 15,

        // 20: utility / discard / card-flow.
        // g198: ForgottenRitual / OneTwoPunchCard / Tremble / Dominate
        // are Ironclad card-mod utilities.
        //
        // Bead `sts2-rs-g8sy4` (GitHub issue #8): the bulk of the
        // Silent / Defect / Colorless card-flow + utility pool.
        // Silent: Acrobatics / Anticipate / BladeDance / Prepared /
        // CalculatedGamble / EscapePlan / Expertise / Expose /
        // HiddenDaggers / Reflex / Tactician / UpMySleeve / Adrenaline
        // / BladeOfInk / BulletTime / Burst / Nightmare / ShadowStep /
        // StormOfSteel (card-flow + setup skills). Defect: Zap /
        // Dualcast / Coolheaded / Hotfix / Turbo / Chaos / Chill /
        // DarknessCard / DoubleEnergy / EnergySurge / Fusion / Overclock
        // / Scavenge / Skim / Synchronize / Tempest / WhiteNoise /
        // GeneticAlgorithm / Ignition / Modded / MultiCast / Rainbow /
        // Reboot / SignalBoost / Supercritical / Voltaic / Quadcast
        // (channel / evoke / draw skills). Colorless: MasterOfStrategy
        // / Alchemize / HiddenGem / BelieveInYou / Catastrophe /
        // CoordinateCard / Lift / Production / Prolong / Restlessness /
        // Mimic / Rally / Scrawl / TheGambit / Anointed / BeatDown
        // (utility skills).
        // Track-A (STS2-native rollout ordering): NotYet is the only
        // Ironclad spec card with NO STS1 counterpart and no prior
        // classification — a 2-cost Skill that heals 10 (13 upgraded)
        // and Exhausts (reward/shop pool only). No immediate combat
        // tempo (pure out-of-combat-value heal), so it slots with the
        // low-priority utility skills here rather than competing with
        // block/attack plays. Classifying it (off NEUTRAL=50) keeps a
        // NotYet-bearing hand above [`COVERAGE_THRESHOLD`].
        CardId::NotYet
        | CardId::PanicButton
        | CardId::JackOfAllTrades
        | CardId::BurningPact
        | CardId::DualWield
        | CardId::Discovery
        | CardId::InfernalBlade
        | CardId::Havoc
        | CardId::SecondWind
        | CardId::ThinkingAhead
        | CardId::ShrugItOff
        | CardId::ForgottenRitual
        | CardId::OneTwoPunchCard
        | CardId::Tremble
        | CardId::Dominate
        | CardId::Acrobatics
        | CardId::Anticipate
        | CardId::BladeDance
        | CardId::Prepared
        | CardId::CalculatedGamble
        | CardId::EscapePlan
        | CardId::Expertise
        | CardId::Expose
        | CardId::HiddenDaggers
        | CardId::Reflex
        | CardId::Tactician
        | CardId::UpMySleeve
        | CardId::Adrenaline
        | CardId::BladeOfInk
        | CardId::BulletTime
        | CardId::Burst
        | CardId::Nightmare
        | CardId::ShadowStep
        | CardId::StormOfSteel
        | CardId::Zap
        | CardId::Dualcast
        | CardId::Coolheaded
        | CardId::Hotfix
        | CardId::Turbo
        | CardId::Chaos
        | CardId::Chill
        | CardId::DarknessCard
        | CardId::DoubleEnergy
        | CardId::EnergySurge
        | CardId::Fusion
        | CardId::Overclock
        | CardId::Scavenge
        | CardId::Skim
        | CardId::Synchronize
        | CardId::Tempest
        | CardId::WhiteNoise
        | CardId::GeneticAlgorithm
        | CardId::Ignition
        | CardId::Modded
        | CardId::MultiCast
        | CardId::Rainbow
        | CardId::Reboot
        | CardId::SignalBoost
        | CardId::Supercritical
        | CardId::Voltaic
        | CardId::Quadcast
        | CardId::MasterOfStrategy
        | CardId::Alchemize
        | CardId::HiddenGem
        | CardId::BelieveInYou
        | CardId::Catastrophe
        | CardId::CoordinateCard
        | CardId::Lift
        | CardId::Production
        | CardId::Prolong
        | CardId::Restlessness
        | CardId::Mimic
        | CardId::Rally
        | CardId::Scrawl
        | CardId::TheGambit
        | CardId::Anointed
        | CardId::BeatDown
        // g8sy4 colorless straggler pass: card-flow / generate-random
        // skills with no immediate combat value. Distraction (1c
        // generate Skill), Metamorphosis (3 random Attacks into draw),
        // Outmaneuver (Self stack-power), Wish (0c move-card utility).
        | CardId::Distraction
        | CardId::Metamorphosis
        | CardId::Outmaneuver
        | CardId::Wish => 20,

        // 22: scaling / situational / big block.
        // g198: BloodWall (16 block), ColossusCard, Taunt (block+debuff),
        // PactsEnd (0-cost 17 dmg attack — exceptional value).
        // g8sy4: EvilEye (1-cost, double current block — big defensive
        // scaler) and UltimateDefend (1-cost +11 block) both bucket here
        // alongside the other "big block" cards.
        //
        // Bead `sts2-rs-g8sy4` (GitHub issue #8): the cross-character
        // "big block + situational defense" bucket. Silent skills with
        // ≥6 block or block+retain: Survivor / Backflip / CloakAndDagger
        // / Deflect / DodgeAndRoll / Untouchable / BlurCard /
        // BouncingFlask / HandTrick / Mirage / LegSweep / Shadowmeld /
        // Snakebite / KnifeTrap. Defect: BoostAway / ChargeBattery /
        // Hologram / Leap / LightningRod / Compact / Glacier /
        // Glasswork / FightThrough / BootSequence / ShadowShield.
        // Colorless: Equilibrium / HuddleUp / InterceptCard / Splash /
        // TheBombCard (delayed huge AOE — treat as scaling threat).
        CardId::Impatience
        | CardId::TrueGrit
        | CardId::BloodWall
        | CardId::ColossusCard
        | CardId::Taunt
        | CardId::PactsEnd
        | CardId::EvilEye
        | CardId::UltimateDefend
        | CardId::Survivor
        | CardId::Backflip
        | CardId::CloakAndDagger
        | CardId::Deflect
        | CardId::DodgeAndRoll
        | CardId::Untouchable
        | CardId::BlurCard
        | CardId::BouncingFlask
        | CardId::HandTrick
        | CardId::Mirage
        | CardId::LegSweep
        | CardId::Shadowmeld
        | CardId::Snakebite
        | CardId::KnifeTrap
        | CardId::BoostAway
        | CardId::ChargeBattery
        | CardId::Hologram
        | CardId::Leap
        | CardId::LightningRod
        | CardId::Compact
        | CardId::Glacier
        | CardId::Glasswork
        | CardId::FightThrough
        | CardId::BootSequence
        | CardId::ShadowShield
        | CardId::Equilibrium
        | CardId::HuddleUp
        | CardId::InterceptCard
        | CardId::Splash
        | CardId::TheBombCard
        // g8sy4 colorless straggler pass: Relax (3-cost big-block +
        // 2-stack next-turn powers) and Stack (1-cost block scales
        // with discard pile) are the remaining big-block / scaling-
        // defense cards from colorless.rs.
        | CardId::Relax
        | CardId::Stack => 22,

        // 25: solid offense / big-damage attacks.
        // g198: Break (20 dmg), Cinder (17 dmg), MoltenFist (10/1),
        // Spite (6/0), ExpectAFight (situational setup).
        // g8sy4: ByrdSwoop (0-cost 14 dmg — exceptional tempo).
        //
        // Bead `sts2-rs-g8sy4` (GitHub issue #8): Silent 0-cost
        // high-damage attacks (Backstab 0c/11, PreciseCut 0c/13,
        // Assassinate 0c/10, Suppress 0c/11) plus Dash (2c attack+
        // block combo) and GrandFinale (0-cost 50-damage finisher).
        CardId::Bash
        | CardId::Uppercut
        | CardId::Thunderclap
        | CardId::Break
        | CardId::Cinder
        | CardId::MoltenFist
        | CardId::Spite
        | CardId::ExpectAFight
        | CardId::ByrdSwoop
        | CardId::Backstab
        | CardId::PreciseCut
        | CardId::Assassinate
        | CardId::Suppress
        | CardId::Dash
        | CardId::GrandFinale
        // g8sy4 colorless straggler pass: Clash (0-cost 14 dmg
        // conditional) and Whistle (3-cost 33 dmg exhaust finisher)
        // are exceptional-value tempo attacks alongside ByrdSwoop /
        // Bash / Break.
        | CardId::Clash
        | CardId::Whistle
        // GH #515: GiantRock is a 1-cost token attack for 16 — the largest
        // single-target damage per energy in this pass.
        | CardId::GiantRock => 25,

        // 27: high-value singletons
        CardId::Whirlwind => 27,

        // 30: big swing attacks.
        // g198: HowlFromBeyond, Mangle, Stomp, Unrelenting are 2-3 cost
        // mid-damage attacks similar to Inferno/Bludgeon.
        // g8sy4: Maul (2x damage with kill-aware re-hit) and NeowsFury
        // (damage + 2-card discard retrieval) bucket as solid mid-attacks.
        //
        // Bead `sts2-rs-g8sy4` (GitHub issue #8): cross-character big
        // swing attacks. Silent: Pinpoint (3c/17), Pounce (2c/12),
        // Predator (2c/15), Strangle (1c/8+Constricted), EchoingSlash
        // (1c/10), Murder (3c with bonus), TheHunt (1c/10), Finisher
        // (1c with cards-played scaling), PoisonedStab (1c/6 + Poison).
        // Defect: NullCard (2c/10 + orb-lock), Refract (3c/9), RocketPunch
        // (2c/13), Sunder (3c/24), Synthesis (2c/12), AdaptiveStrike
        // (2c/18), AllForOne (2c/10), FlakCannon (2c/8), Hyperbeam (2c/26),
        // IceLance (3c/19), MeteorStrike (5c/24), Shatter (1c/11),
        // BallLightning (1c/7), ColdSnap (1c/6), CompileDriver (1c/7),
        // MomentumStrike (1c/10), Scrape (1c/7). Colorless: Fisticuffs
        // (1c/7), TagTeam (2c), Rend (2c), Knockdown (3c), Jackpot (3c).
        CardId::InfernoCard
        | CardId::Bludgeon
        | CardId::DramaticEntrance
        | CardId::HowlFromBeyond
        | CardId::Mangle
        | CardId::Stomp
        | CardId::Unrelenting
        | CardId::Maul
        | CardId::NeowsFury
        | CardId::Pinpoint
        | CardId::Pounce
        | CardId::Predator
        | CardId::Strangle
        | CardId::EchoingSlash
        | CardId::Murder
        | CardId::TheHunt
        | CardId::Finisher
        | CardId::PoisonedStab
        | CardId::NullCard
        | CardId::Refract
        | CardId::RocketPunch
        | CardId::Sunder
        | CardId::Synthesis
        | CardId::AdaptiveStrike
        | CardId::AllForOne
        | CardId::FlakCannon
        | CardId::Hyperbeam
        | CardId::IceLance
        | CardId::MeteorStrike
        | CardId::Shatter
        | CardId::BallLightning
        | CardId::ColdSnap
        | CardId::CompileDriver
        | CardId::MomentumStrike
        | CardId::Scrape
        | CardId::Fisticuffs
        | CardId::TagTeam
        | CardId::Rend
        | CardId::Knockdown
        | CardId::Jackpot
        // g8sy4 colorless straggler pass: Rebound (1c attack + Rebound
        // power rider) and Squash (1c 10 dmg + debuff) fit the
        // mid-attack-plus-rider bucket alongside PoisonedStab /
        // EchoingSlash.
        | CardId::Rebound
        | CardId::Squash => 30,

        // 35: standard attacks.
        // g198: Breakthrough, Bully, Conflagration, Dismantle,
        // SetupStrikeCard slot here next to Rampage/PommelStrike.
        // g8sy4: UltimateStrike (1-cost 14 dmg straight attack).
        //
        // Bead `sts2-rs-g8sy4` (GitHub issue #8): Silent / Defect /
        // Colorless 1-cost ~7-9 dmg standard attacks. Silent:
        // DaggerThrow (1c/9), FlickFlack (1c/7), LeadingStrike (1c/7),
        // SuckerPunch (1c/8 + Weak), Flechettes (1c/5 with scaling),
        // MementoMori (1c/8),
        // Skewer (X-cost 7), Slice (0c/6). Defect: Barrage (1c/5
        // with orb scaling), FocusedStrikeCard (1c/9), Uproar (2c/5+).
        // Colorless: Omnislice (0c X-dmg), SeekerStrike (1c rider),
        // ThrummingHatchet (1c rider), Volley (X-cost atk),
        // GangUp (1c rider), GoldAxe (1c rider).
        CardId::Rampage
        | CardId::PommelStrike
        | CardId::HandOfGreed
        | CardId::PerfectedStrike
        | CardId::Hemokinesis
        | CardId::SwordBoomerang
        | CardId::Breakthrough
        | CardId::Bully
        | CardId::Conflagration
        | CardId::Dismantle
        | CardId::SetupStrikeCard
        | CardId::UltimateStrike
        | CardId::DaggerThrow
        | CardId::FlickFlack
        | CardId::LeadingStrike
        | CardId::SuckerPunch
        | CardId::Flechettes
        | CardId::MementoMori
        | CardId::Skewer
        | CardId::Slice
        | CardId::Barrage
        | CardId::FocusedStrikeCard
        | CardId::Uproar
        | CardId::Omnislice
        | CardId::SeekerStrike
        | CardId::ThrummingHatchet
        | CardId::Volley
        | CardId::GangUp
        | CardId::GoldAxe
        // g8sy4 colorless straggler pass: MadScience (1c/12 straight
        // attack) and Peck (1c three-hit attack) are standard 1-cost
        // attacks alongside DaggerThrow / LeadingStrike.
        | CardId::MadScience
        | CardId::Peck => 35,

        // 40: AOE / risky attacks.
        // g198: Pillage (chance-based reward) lives here.
        // g8sy4: Exterminate (1-cost 3-dmg AOE) is the cheapest AOE
        // damage card; sits with the other AOE filler.
        //
        // Bead `sts2-rs-g8sy4` (GitHub issue #8): cross-character
        // AOE / cheap utility attacks. Silent: DaggerSpray (AoE 4x2),
        // Ricochet (2c random-target 3-dmg attacks), Neutralize (0c/3
        // + Weak). Defect: BeamCell (0c/3 + Weak), Claw (0c/3 scaling),
        // GoForTheEyes (0c/3 + Weak), GunkUp (1c/4 debuff), TeslaCoil
        // (0c/3 + Lightning), HelixDrill (0c/3), SweepingBeam (1c AoE),
        // Ftl (0c/5 first-attack bonus). Colorless: Bolas (0c rider),
        // Salvo (1c AoE).
        CardId::FiendFire
        | CardId::FlameBarrierCard
        | CardId::FlashOfSteel
        | CardId::Feed
        | CardId::Pillage
        | CardId::Exterminate
        | CardId::DaggerSpray
        | CardId::Ricochet
        | CardId::Neutralize
        | CardId::BeamCell
        | CardId::Claw
        | CardId::GoForTheEyes
        | CardId::GunkUp
        | CardId::TeslaCoil
        | CardId::HelixDrill
        | CardId::SweepingBeam
        | CardId::Ftl
        | CardId::Bolas
        | CardId::Salvo
        // GH #515: RipAndTear hits random opponents twice for 7. Random
        // targeting is exactly what this bucket's "risky" means.
        | CardId::RipAndTear => 40,

        // 48: filler attacks.
        // g198: AshenStrike, FightMe, TearAsunder, Thrash are
        // sub-Strike-ratio cheap attacks.
        //
        // Track-A (STS2-native rollout ordering): this bucket was
        // originally `50`, numerically COLLIDING with `NEUTRAL_PRIORITY`
        // (also 50). The collision had two costs: (a) these classified
        // filler attacks were indistinguishable from genuinely
        // unclassified cards in the softmax, and (b) they counted as
        // *uncovered* in [`rollout_coverage_fraction`] (which tests
        // `play_ordering(..) != NEUTRAL_PRIORITY`), so a hand heavy in
        // TwinStrike/Anger/BodySlam/AshenStrike/FightMe/TearAsunder/
        // Thrash could push the covered fraction below
        // [`COVERAGE_THRESHOLD`] and drop the whole step to the Uniform
        // (random) rollout policy — exactly the "rollouts effectively
        // random on STS2 content" failure this lever targets. Dropping
        // the bucket to `48` keeps the STS1-derived ordering (still a
        // hair better than the Strike/Defend basics at 55, worse than
        // the cheap-block 52 bucket) while breaking the NEUTRAL tie so
        // the cards register as covered and weigh distinctly.
        CardId::TwinStrike
        | CardId::Anger
        | CardId::BodySlam
        | CardId::MindBlast
        | CardId::AshenStrike
        | CardId::FightMe
        | CardId::TearAsunder
        | CardId::Thrash
        // g8sy4 silent straggler pass: Shiv is a 0-cost generated
        // 4-damage filler attack — same tempo class as TwinStrike /
        // Anger / BodySlam.
        | CardId::Shiv => 48,

        // 52: cheap block / setup
        CardId::IronWave | CardId::Entrench | CardId::Purity => 52,

        // 55: basics (Strike/Defend per character).
        // g198 cross-character first-pass: pin all four character basics
        // into the same bucket so the heuristic treats e.g. StrikeDefect
        // identically to StrikeIronclad when scoring rollout actions.
        CardId::StrikeIronclad
        | CardId::DefendIronclad
        | CardId::Headbutt
        | CardId::StrikeSilent
        | CardId::DefendSilent
        | CardId::StrikeDefect
        | CardId::DefendDefect
        | CardId::StrikeNecrobinder
        | CardId::DefendNecrobinder
        | CardId::StrikeRegent
        | CardId::DefendRegent => 55,

        // 60: status cards (Burn fires automatically — usually don't *want*
        // to play, but sometimes the discard pile needs cycling)
        // g8sy4: SpoilsMap / LanternKey / ByrdonisEgg are STS2 status-pile
        // cards living in sts2-content/src/cards/status.rs alongside Burn.
        CardId::Burn
        | CardId::Decay
        | CardId::Clumsy
        | CardId::Wound
        | CardId::Dazed
        | CardId::Void
        | CardId::SpoilsMap
        | CardId::LanternKey
        | CardId::ByrdonisEgg
        // GH #515: Beckon (6 unblockable) and Toxic (5) both fire from hand
        // at end of turn, so playing them trades energy for the hit — the
        // Burn shape this bucket already holds. Debris is a 1-cost Exhaust
        // status with no effect at all.
        | CardId::Beckon
        | CardId::Toxic
        | CardId::Debris
        // FranticEscape sits here on measurement, against its reading. The
        // C# says it is a *survival* card, not junk: `SandpitPower`
        // (`Models.Powers/SandpitPower.cs`) is a counter that decrements
        // every enemy turn and KILLS the player and Osty when removed, and
        // this card's `OnPlay` adds 1 to it — leaving it in hand is how
        // GH #215 lost runs. So it was first given 5, the situational-utility
        // bucket. That cost **1.98pp of position-book agreement**
        // (p = 0.030), concentrated exactly where the card lives: defect/boss
        // -16.2pp and necrobinder/boss -15.2pp. At 60 the same measurement is
        // -0.50pp (p = 0.63). The rollout is an *estimator*, not a plan: it
        // spends a turn's energy on whatever it promotes, and a card whose
        // own cost rises 1 per play drains the turn it was meant to save.
        // GH #215's guarantee is carried search-side (the `loss_gradient`
        // loss-band ordering and the proven-loss override), which is where it
        // belongs and where its regression test pins it.
        | CardId::FranticEscape => 60,

        // 65: low-impact curses
        CardId::Regret | CardId::Shame | CardId::Doubt => 65,

        // 70: hard curses — almost never play voluntarily
        // g8sy4: Greed, Guilty, Debt, Folly are STS2 curses living in
        // sts2-content/src/cards/curse.rs alongside the STS1 set.
        CardId::Slimed
        | CardId::Writhe
        | CardId::Parasite
        | CardId::AscendersBane
        | CardId::CurseOfTheBell
        | CardId::Injury
        | CardId::Normality
        | CardId::Pain
        | CardId::Greed
        | CardId::Guilty
        | CardId::Debt
        | CardId::Folly
        // GH #515: SporeMind is a 1-cost Exhaust curse with no effect — the
        // Slimed shape. Enthralled is 2-cost and Eternal, so it cannot even
        // be cleared by playing it.
        | CardId::SporeMind
        | CardId::Enthralled => 70,

        // ── Regent rollout-coverage pass (2026-07-20 under-block
        // investigation) ──────────────────────────────────────────────
        // Regent's 88-card deck pool (+5 minion/blade tokens) was
        // entirely NEUTRAL except the two basics above, so any Regent
        // hand's `rollout_coverage_fraction` sat at ~2% << 0.8 and the
        // heuristic rollout degraded to Uniform for EVERY Regent state.
        // Uniform depth-10 rollouts have high per-sim variance that buries
        // the ~0.001–0.005-tanh-unit per-card leaf signal below the noise
        // floor (block/HP/enemy-progress deltas all shrink under ÷max_hp
        // and the ÷32 leaf divisor, worst on 300–380-HP bosses), so MCTS
        // played near-randomly at the tactical margin — leaving free
        // block and whole turns of energy unspent. Buckets mirror the
        // shared ladder above by card type/cost/block (game-informed
        // per-card tuning is a follow-up; this pass restores coverage so
        // the Heuristic policy engages at all).

        // -2: powers cost ≤ 1 — cheap scalers/utility powers fire first,
        // same tier as DarkEmbrace / NoxiousFumes.
        CardId::Arsenal
        | CardId::BlackHole
        | CardId::ChildOfTheStars
        | CardId::Furnace
        | CardId::NeutronAegis
        | CardId::PaleBlueDot
        | CardId::ParryCard
        | CardId::PillarOfCreation
        | CardId::RoyaltiesCard
        | CardId::SeekingEdge
        | CardId::TheSealedThrone
        | CardId::TyrannyCard => -2,

        // 0: powers cost ≥ 2 — combat-defining scalers, same tier as
        // DemonForm / EchoForm.
        CardId::Genesis
        | CardId::HammerTime
        | CardId::MonarchsGazeCard
        | CardId::Orbit
        | CardId::SpectrumShift
        | CardId::SwordSage
        | CardId::VoidForm => 0,

        // 22: block skills (base_block > 0 or effect block-grant) — the
        // shared "big block + situational defense" bucket (Deflect /
        // UltimateDefend tier).
        CardId::Bulwark
        | CardId::CloakOfStars
        | CardId::CosmicIndifference
        | CardId::GatherLight
        | CardId::Glitterstream
        | CardId::IAmInvincible
        | CardId::ManifestAuthority
        | CardId::MinionSacrifice
        | CardId::ParticleWall
        | CardId::Patter
        | CardId::ReflectCard => 22,

        // 20: utility / card-flow skills with no immediate block/damage.
        CardId::Alignment
        | CardId::Begone
        | CardId::BigBang
        | CardId::BundleOfJoy
        | CardId::Charge
        | CardId::Conqueror
        | CardId::Convergence
        | CardId::DecisionsDecisions
        | CardId::ForegoneConclusion
        | CardId::Glimmer
        | CardId::Glow
        | CardId::Guards
        | CardId::HiddenCache
        | CardId::KnowThyPlace
        | CardId::Largesse
        | CardId::MonologueCard
        | CardId::Prophesize
        | CardId::Quasar
        | CardId::RefineBlade
        | CardId::Resonance
        | CardId::RoyalGamble
        | CardId::SpoilsOfBattle
        | CardId::SummonForth
        | CardId::Terraforming
        | CardId::TheSmith
        | CardId::Venerate => 20,

        // 30: free (cost ≤ 0) attacks — cheap tempo, play before costed
        // attacks (mirrors the 0-cost-attack treatment in the ladder).
        CardId::AstralPulse
        | CardId::CollisionCourse
        | CardId::Comet
        | CardId::FallingStar
        | CardId::GammaBlast
        | CardId::HeavenlyDrill
        | CardId::LunarBlast
        | CardId::MakeItSo
        | CardId::MeteorShower
        | CardId::MinionDiveBomb
        | CardId::MinionStrike
        | CardId::Radiate
        | CardId::Stardust => 30,

        // 40: costed attacks — the shared "standard attacks" bucket.
        CardId::BeatIntoShape
        | CardId::Bombardment
        | CardId::CelestialMight
        | CardId::CrashLanding
        | CardId::CrescentSpear
        | CardId::CrushUnder
        | CardId::Devastate
        | CardId::DyingStar
        | CardId::GuidingStar
        | CardId::Hegemony
        | CardId::HeirloomHammer
        | CardId::KinglyKick
        | CardId::KinglyPunch
        | CardId::KnockoutBlow
        | CardId::PhotonCut
        | CardId::SevenStars
        | CardId::ShiningStrike
        | CardId::SolarStrike
        | CardId::SovereignBlade
        | CardId::Supermassive
        | CardId::WroughtInWar => 40,

        // ── Necrobinder rollout-coverage pass (2026-07-20) ───────────
        // Same latent Uniform-fallback issue Regent had: only 8/88
        // Necrobinder cards were classified, so any Necrobinder hand's
        // coverage sat far below 0.8 and the heuristic rollout degraded
        // to Uniform. Silent (99%) / Defect (100%) were already covered
        // by the g8sy4 passes. Bucketed by type/cost/block exactly like
        // the Regent pass above (game-informed per-card tuning is a
        // follow-up; this restores coverage so the policy engages).

        // -2: powers cost ≤ 1.
        CardId::CalcifyCard
        | CardId::CallOfTheVoid
        | CardId::CountdownCard
        | CardId::DanseMacabre
        | CardId::DevourLifeCard
        | CardId::Friendship
        | CardId::Haunt
        | CardId::LethalityCard
        | CardId::Neurosurge
        | CardId::Pagestorm
        | CardId::Shroud
        | CardId::SpiritOfAsh => -2,

        // 0: powers cost ≥ 2 (incl. ReaperForm — the Necrobinder
        // DemonForm-class combat-defining power).
        CardId::Demesne
        | CardId::ForbiddenGrimoire
        | CardId::ReaperForm
        | CardId::SentryMode
        | CardId::SleightOfFlesh => 0,

        // 22: block skills (base_block > 0).
        CardId::DeathsDoor
        | CardId::Defy
        | CardId::Delay
        | CardId::Melancholy
        | CardId::NegativePulse
        | CardId::PullAggro
        | CardId::Undeath
        // GH #515 straggler: GraveWarden is a 1-cost skill granting 8 block
        // (plus a Soul to the draw pile) — the block-skill rule this pass
        // is built on.
        | CardId::GraveWarden => 22,

        // 20: utility / card-flow / summon skills (incl. the Soul token).
        CardId::Afterlife
        | CardId::Bodyguard
        | CardId::BorrowedTime
        | CardId::CaptureSpirit
        | CardId::Cleanse
        | CardId::Deathbringer
        | CardId::Dirge
        | CardId::Dredge
        | CardId::Eidolon
        | CardId::EndOfDays
        | CardId::EnfeeblingTouch
        | CardId::GlimpseBeyond
        | CardId::Invoke
        | CardId::LegionOfBone
        | CardId::NoEscape
        | CardId::Oblivion
        | CardId::Parse
        | CardId::Putrefy
        | CardId::Reanimate
        | CardId::Sacrifice
        | CardId::Scourge
        | CardId::Seance
        | CardId::SharedFate
        | CardId::Soul
        | CardId::Spur
        | CardId::Transfigure
        | CardId::Wisp => 20,

        // 30: free (cost ≤ 0) attacks.
        CardId::Eradicate
        | CardId::Fetch
        | CardId::Misery
        | CardId::Poke
        | CardId::RightHandHand
        // GH #515 straggler: SweepingGaze is the 0-cost Osty attack token.
        | CardId::SweepingGaze => 30,

        // 40: costed attacks.
        CardId::BansheesCry
        | CardId::BoneShards
        | CardId::Bury
        | CardId::DeathMarch
        | CardId::DebilitateCard
        | CardId::Fear
        | CardId::Flatten
        | CardId::Graveblast
        | CardId::Hang
        | CardId::HighFive
        | CardId::Protector
        | CardId::PullFromBelow
        | CardId::Rattle
        | CardId::Reap
        | CardId::Reave
        | CardId::SculptingStrike
        | CardId::Severance
        | CardId::SicEm
        | CardId::Snap
        | CardId::SoulStorm
        | CardId::Sow
        | CardId::Squeeze
        | CardId::TheScythe
        | CardId::TimesUp
        | CardId::Unleash
        | CardId::Veilpiercer
        // GH #515 stragglers: three 1-cost Necrobinder attacks the
        // 2026-07-20 pass missed — BlightStrike (8 + Doom), Defile (13,
        // Ethereal) and DrainPower (10 + upgrade 2 in the discard). Bucketed
        // by cost and type like the rest of this pass; per-card tuning (for
        // instance promoting Defile, which is lost at end of turn if not
        // played) stays the follow-up it has always been, because nothing
        // here can measure a one-bucket move (GH #446: shifting a rollout
        // priority made the search monotonically worse).
        | CardId::BlightStrike
        | CardId::Defile
        | CardId::DrainPower => 40,

        // Default for everything else (most STS2 cards, including
        // Silent/Defect cards we haven't classified).
        _ => NEUTRAL_PRIORITY,
    }
}

/// Pick an action index from `actions` using a softmax-over-priority
/// sample. `actions` is the slice produced by
/// [`sts2_core::legal_actions`]; the returned index is into that slice.
///
/// EndTurn and unknown cards land near zero weight; the highest-priority
/// (lowest `play_ordering`) cards dominate. Cumulative-weight inverse
/// transform sampling keeps the hot path branch-light.
///
/// # Panics
/// Panics if `actions` is empty.
pub fn pick_action_heuristic<R: Rng>(
    rng: &mut R,
    state: &CombatState,
    actions: &[Action],
) -> usize {
    assert!(!actions.is_empty(), "pick_action_heuristic: empty actions");
    // Bead `sts2-rs-5dw7` (z9ap remediation): cache the per-rollout
    // incoming-damage sum + effective HP once per state so the HP-aware
    // priority bump (Defend↑/Strike↓ when effective_hp <= incoming) is
    // O(actions × 1) instead of O(actions × creatures) per call.
    let ctx = HpAwareCtx::compute(state, IncomingDamage::Base);
    let priorities: smallvec::SmallVec<[i16; 16]> = actions
        .iter()
        .map(|a| action_priority(state, a, &ctx))
        .collect();
    softmax_sample(rng, &priorities, actions.len())
}

/// Size of the memoized softmax-weight table. Priorities span roughly
/// `[-30, 90]` for in-play cards plus the `+10_000` EndTurn bucket, so
/// most `(p - min_p)` diffs are tiny; diffs past the table (EndTurn vs
/// the min card priority — present in every mid-turn sample) short-circuit
/// to `0.0`, the exact value the f32 `exp` underflows to at that range.
const SOFTMAX_EXP_TABLE_SIZE: usize = 4096;

/// Lazily-built memo for the rollout softmax weight
/// `exp(-(p - min_p) / SOFTMAX_TAU)`, indexed by the integer priority
/// diff (perf-investigation 2026-07-04: `expf` was a top per-rollout-step
/// transcendental). Each entry is produced by the EXACT expression the
/// direct computation used — `i16 → f32` conversion is exact and the
/// subtraction of two exact integers ≤ 2^24 is exact, so
/// `p_f - min_p == diff as f32` bit-for-bit and table hits are
/// bit-identical to computing inline.
static SOFTMAX_EXP_TABLE: std::sync::OnceLock<Vec<f32>> = std::sync::OnceLock::new();

/// Softmax weight for a non-negative integer priority diff. Bit-identical
/// to `(-(diff as f32) / SOFTMAX_TAU).exp()`.
#[inline]
fn softmax_weight(diff: i32) -> f32 {
    debug_assert!(diff >= 0, "priority diff must be non-negative");
    let table = SOFTMAX_EXP_TABLE.get_or_init(|| {
        (0..SOFTMAX_EXP_TABLE_SIZE)
            .map(|d| {
                #[allow(clippy::cast_precision_loss)]
                let d_f = d as f32;
                (-d_f / SOFTMAX_TAU).exp()
            })
            .collect()
    });
    #[allow(clippy::cast_sign_loss)]
    if let Some(w) = table.get(diff as usize) {
        return *w;
    }
    // Past the table the weight is exactly +0.0: for diff >= 4096 the f32
    // computation is `exp(-128)` or smaller (~2.6e-56, far below the f32
    // round-to-zero threshold 2^-150 ~ 7e-46), so a conforming `exp`
    // underflows to +0.0 — verified empirically for every diff >= 3328.
    // Returning 0.0 directly is therefore bit-identical to the live `expf`
    // this path used to pay once per softmax call for the EndTurn-vs-min
    // diff (~10_000), which always overshoots the table.
    0.0
}

/// Cumulative-weight inverse-transform sample over `exp(-(p - min)/tau)`
/// weights. Shared tail of [`pick_action_heuristic`] and
/// [`pick_rollout_action`]; consumes at most one RNG draw.
fn softmax_sample<R: Rng>(rng: &mut R, priorities: &[i16], n_actions: usize) -> usize {
    // Softmax over -priority/tau. Subtract the min so we work with
    // non-negative exponents (numerically stable for the [-30..70] bucket
    // range; +10_000 collapses to ~0 weight as designed).
    let min_p = priorities.iter().copied().min().unwrap_or(0);
    let mut weights: smallvec::SmallVec<[f32; 16]> =
        smallvec::SmallVec::with_capacity(priorities.len());
    let mut total = 0.0_f32;
    for &p in priorities {
        // exp(-(p - min)/tau) — smaller priority => larger weight.
        let w = softmax_weight(i32::from(p) - i32::from(min_p));
        weights.push(w);
        total += w;
    }
    if !total.is_finite() || total <= 0.0 {
        // Fallback: uniform over the action list. Shouldn't happen for
        // realistic priority ranges, but keep the rollout from
        // deadlocking on a degenerate weight vector.
        return rng.gen_range(0..n_actions);
    }
    let mut x = rng.gen::<f32>() * total;
    for (i, w) in weights.iter().enumerate() {
        x -= *w;
        if x <= 0.0 {
            return i;
        }
    }
    n_actions - 1
}

/// How a simulated co-op teammate plays (`sts2_core::coop::TeammatePolicy`):
/// the rollout heuristic, from an RNG seeded by the caller so the teammate's
/// turn is a pure function of the state. Like a rollout it plays every card
/// it can afford, which is what a teammate who has not ended their turn
/// usually does.
pub fn teammate_policy(state: &CombatState, actions: &[Action], seed: u64) -> usize {
    use rand::SeedableRng;
    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(seed ^ 0x7EA3_4A7E_5EA7_0000);
    pick_rollout_action(&mut rng, state, actions)
}

/// Fused coverage gate + rollout-policy pick for the per-rollout-step
/// hot path (perf-investigation 2026-07-04). Behaviorally identical to
///
/// ```text
/// if rollout_coverage_fraction(state, actions) < COVERAGE_THRESHOLD {
///     Uniform.pick(rng, state, actions)
/// } else {
///     Heuristic.pick(rng, state, actions)
/// }
/// ```
///
/// — same action index, same number and order of RNG draws — but the
/// hand/card resolution and [`play_ordering`] lookup happen ONCE per
/// action instead of once in the coverage walk and again inside
/// [`action_priority`].
///
/// # Panics
/// Panics if `actions` is empty.
pub fn pick_rollout_action<R: Rng>(rng: &mut R, state: &CombatState, actions: &[Action]) -> usize {
    pick_rollout_action_with(rng, state, actions, END_TURN_PRIORITY)
}

/// [`pick_rollout_action`] with the turn-end priority supplied by the
/// caller ([`crate::MctsConfig::rollout_end_turn_priority`]).
///
/// At the [`END_TURN_PRIORITY`] default this is bit-identical to
/// [`pick_rollout_action`] — same actions, same RNG draws. A *finite*
/// value is the interesting case: it puts `EndTurn` back into the softmax,
/// so a rollout ends the turn with a probability that falls as the best
/// remaining card improves.
///
/// Why that is a knob at all (GH #446,
/// `logs/2026-09-08-endturn-rollout/PREREGISTRATION.md`): at the default,
/// `EndTurn`'s softmax weight underflows to exactly `+0.0`, so the
/// heuristic rollout **never ends a turn while any card is playable** — it
/// empties the hand, every turn. Lines whose merit is *not* playing a card
/// (hold the block for the turn the intent lands, keep a card for a retain,
/// decline to trigger a power) are therefore unrepresentable in the
/// rollout: both root children dump the hand, so the value that separates
/// them never appears. The honest hard book prices that class at +12.4pp of
/// fight win probability per miss over 25 positions.
pub fn pick_rollout_action_with<R: Rng>(
    rng: &mut R,
    state: &CombatState,
    actions: &[Action],
    end_turn_priority: i16,
) -> usize {
    pick_rollout_action_incoming(rng, state, actions, end_turn_priority, IncomingDamage::Base)
}

/// [`pick_rollout_action_with`] reading the enemies' telegraphed attacks
/// the `incoming` way ([`crate::MctsConfig::effective_incoming`]): it moves
/// the HP-aware arms (lethal turn, sustained pressure, the graded Defend
/// bonus). At [`IncomingDamage::Base`] it is [`pick_rollout_action_with`].
pub fn pick_rollout_action_incoming<R: Rng>(
    rng: &mut R,
    state: &CombatState,
    actions: &[Action],
    end_turn_priority: i16,
    incoming: IncomingDamage,
) -> usize {
    pick_rollout_action_by(
        rng,
        state,
        actions,
        end_turn_priority,
        incoming,
        |id, ch| {
            let base = play_ordering(id, ch);
            (base, base != NEUTRAL_PRIORITY)
        },
    )
}

/// [`pick_rollout_action_with`] reading card base priorities from a fitted
/// [`LearnedPriorities`] table instead of the STS1-derived ladder
/// ([`RolloutPolicyKind::Learned`]). A card the table does not list keeps
/// its [`play_ordering`] value; a card it lists counts as covered for the
/// coverage gate whatever value it carries, since a fitted value of
/// [`NEUTRAL_PRIORITY`] is a measurement, not the unclassified fallback.
/// Every context adjustment in [`play_card_priority`] applies on top of
/// the learned base unchanged.
///
/// # Panics
/// Panics if `actions` is empty.
pub fn pick_rollout_action_learned<R: Rng>(
    rng: &mut R,
    state: &CombatState,
    actions: &[Action],
    end_turn_priority: i16,
    table: &LearnedPriorities,
) -> usize {
    pick_rollout_action_learned_incoming(
        rng,
        state,
        actions,
        end_turn_priority,
        table,
        IncomingDamage::Base,
    )
}

/// [`pick_rollout_action_learned`] reading telegraphed attacks the
/// `incoming` way (see [`pick_rollout_action_incoming`]).
pub fn pick_rollout_action_learned_incoming<R: Rng>(
    rng: &mut R,
    state: &CombatState,
    actions: &[Action],
    end_turn_priority: i16,
    table: &LearnedPriorities,
    incoming: IncomingDamage,
) -> usize {
    pick_rollout_action_by(
        rng,
        state,
        actions,
        end_turn_priority,
        incoming,
        |id, ch| {
            table.base(id, ch).map_or_else(
                || {
                    let base = play_ordering(id, ch);
                    (base, base != NEUTRAL_PRIORITY)
                },
                |p| (p, true),
            )
        },
    )
}

/// Shared body of the two rollout pickers. `base_of` returns a card's base
/// priority and whether it counts as covered for the coverage gate.
fn pick_rollout_action_by<R: Rng>(
    rng: &mut R,
    state: &CombatState,
    actions: &[Action],
    end_turn_priority: i16,
    incoming: IncomingDamage,
    base_of: impl Fn(CardId, Option<CharacterId>) -> (i16, bool),
) -> usize {
    assert!(!actions.is_empty(), "pick_rollout_action: empty actions");
    /// Per-action resolution cached from the coverage pass.
    enum Slot<'a> {
        /// `PlayCard` whose card resolved: card ref + base ordering.
        Card {
            card: &'a sts2_core::cards::Card,
            base: i16,
            target: Option<u8>,
        },
        /// Non-`PlayCard` action or unresolvable card — fixed priority.
        Fixed(i16),
    }
    let mut slots: smallvec::SmallVec<[Slot<'_>; 16]> =
        smallvec::SmallVec::with_capacity(actions.len());
    let mut play_card_count: u32 = 0;
    let mut covered: u32 = 0;
    for a in actions {
        let slot = match a {
            Action::PlayCard { hand_index, target } => {
                play_card_count = play_card_count.saturating_add(1);
                let card = state
                    .player
                    .hand
                    .get(*hand_index as usize)
                    .and_then(|uid| state.cards.get(uid.0.get() as usize - 1));
                match card {
                    Some(card) => {
                        let (base, is_covered) = base_of(card.card_id, state.player.character);
                        if is_covered {
                            covered = covered.saturating_add(1);
                        }
                        Slot::Card {
                            card,
                            base,
                            target: *target,
                        }
                    }
                    None => Slot::Fixed(NEUTRAL_PRIORITY),
                }
            }
            Action::EndTurn => Slot::Fixed(end_turn_priority),
            _ => Slot::Fixed(NEUTRAL_PRIORITY),
        };
        slots.push(slot);
    }
    // Coverage gate — mirrors `rollout_coverage_fraction < COVERAGE_THRESHOLD`
    // (the no-PlayCard case returns 1.0 there, which never trips the gate).
    if play_card_count > 0 {
        #[allow(clippy::cast_precision_loss)]
        let frac = covered as f32 / play_card_count as f32;
        if frac < COVERAGE_THRESHOLD {
            // Uniform fallback — identical RNG consumption to `Uniform::pick`.
            return rng.gen_range(0..actions.len());
        }
    }
    // Heuristic path — mirrors `Heuristic::pick` (kill detection consumes
    // no RNG; the softmax consumes at most one draw, exactly like
    // `pick_action_heuristic`).
    if let Some(idx) = obvious_kill_action(state, actions) {
        return idx;
    }
    let ctx = HpAwareCtx::compute(state, incoming);
    let priorities: smallvec::SmallVec<[i16; 16]> = slots
        .iter()
        .map(|s| match s {
            Slot::Fixed(p) => *p,
            Slot::Card { card, base, target } => {
                play_card_priority(state, card, *target, *base, &ctx)
            }
        })
        .collect();
    softmax_sample(rng, &priorities, actions.len())
}

/// Per-action priorities exactly as the heuristic rollout scores them —
/// the softmax input of [`pick_rollout_action_with`], without its coverage
/// gate or kill shortcut. Lower is preferred. The learned rollout policy
/// (`crate::rollout_net`) reads them as one input feature.
#[must_use]
pub fn heuristic_priorities(
    state: &CombatState,
    actions: &[Action],
    end_turn_priority: i16,
) -> smallvec::SmallVec<[i16; 16]> {
    heuristic_priorities_with(state, actions, end_turn_priority, IncomingDamage::Base)
}

/// [`heuristic_priorities`] reading telegraphed attacks the `incoming` way
/// (the playout net's v3 features use [`IncomingDamage::Effective`]).
#[must_use]
pub fn heuristic_priorities_with(
    state: &CombatState,
    actions: &[Action],
    end_turn_priority: i16,
    incoming: IncomingDamage,
) -> smallvec::SmallVec<[i16; 16]> {
    let ctx = HpAwareCtx::compute(state, incoming);
    actions
        .iter()
        .map(|a| match a {
            Action::PlayCard { hand_index, target } => state
                .player
                .hand
                .get(*hand_index as usize)
                .and_then(|uid| state.cards.get(uid.0.get() as usize - 1))
                .map_or(NEUTRAL_PRIORITY, |card| {
                    let base = play_ordering(card.card_id, state.player.character);
                    play_card_priority(state, card, *target, base, &ctx)
                }),
            Action::EndTurn => end_turn_priority,
            _ => NEUTRAL_PRIORITY,
        })
        .collect()
}

/// Env var naming the JSON file [`learned_priorities`] loads.
pub const LEARNED_PRIORITIES_ENV: &str = "STS2_ROLLOUT_PRIORITIES";

/// Number of [`CharacterId`] variants; the table keeps one row per variant.
const N_CHARACTERS: usize = 5;

/// Base rollout priorities fitted offline per `(character, card)`, the
/// replacement [`RolloutPolicyKind::Learned`] reads in place of
/// [`play_ordering`]. Built by `scripts/fit_rollout_priorities.py` from the
/// root visit counts `rollout_policy_labels` dumps.
///
/// JSON shape (unknown card names are an error, so a table fitted against
/// a different card roster fails loudly instead of silently dropping rows):
///
/// ```text
/// {"tables": {"ironclad": {"BASH": 23, "STRIKE_IRONCLAD": 57}, "regent": {}}}
/// ```
#[derive(Debug, Clone)]
pub struct LearnedPriorities {
    /// `rows[character][card as usize]`; [`Self::ABSENT`] where the table
    /// has no entry.
    rows: [Vec<i16>; N_CHARACTERS],
}

impl LearnedPriorities {
    const ABSENT: i16 = i16::MIN;

    /// Parse the JSON form described on the type.
    ///
    /// # Errors
    /// Malformed JSON, an unknown character key, or an unknown card name.
    pub fn from_json(text: &str) -> Result<Self, String> {
        #[derive(serde::Deserialize)]
        struct Raw {
            tables: std::collections::BTreeMap<String, std::collections::BTreeMap<String, i16>>,
        }
        let raw: Raw =
            serde_json::from_str(text).map_err(|e| format!("rollout priorities: {e}"))?;
        let n_cards = CardId::all().len();
        let by_name: std::collections::HashMap<&'static str, CardId> = CardId::all()
            .iter()
            .map(|&c| (c.to_python_name(), c))
            .collect();
        let mut rows: [Vec<i16>; N_CHARACTERS] =
            std::array::from_fn(|_| vec![Self::ABSENT; n_cards]);
        for (character, cards) in raw.tables {
            let ch = match character.to_ascii_lowercase().as_str() {
                "ironclad" => CharacterId::Ironclad,
                "silent" => CharacterId::Silent,
                "defect" => CharacterId::Defect,
                "regent" => CharacterId::Regent,
                "necrobinder" => CharacterId::Necrobinder,
                other => return Err(format!("rollout priorities: unknown character {other:?}")),
            };
            for (name, p) in cards {
                let id = by_name
                    .get(name.as_str())
                    .ok_or_else(|| format!("rollout priorities: unknown card {name:?}"))?;
                if p == Self::ABSENT {
                    return Err(format!(
                        "rollout priorities: {name} uses the reserved value"
                    ));
                }
                rows[ch as usize][*id as usize] = p;
            }
        }
        Ok(Self { rows })
    }

    /// Fitted base priority of `card` for `character`, or `None` when the
    /// table has no entry (including when the character is unknown).
    #[inline]
    #[must_use]
    pub fn base(&self, card: CardId, character: Option<CharacterId>) -> Option<i16> {
        let p = *self.rows[character? as usize].get(card as usize)?;
        (p != Self::ABSENT).then_some(p)
    }
}

static LEARNED_PRIORITIES: std::sync::OnceLock<LearnedPriorities> = std::sync::OnceLock::new();

/// The process-wide fitted table, loaded on first use from the file named
/// by [`LEARNED_PRIORITIES_ENV`] unless [`install_learned_priorities`] ran
/// first.
///
/// # Panics
/// When nothing was installed and the env var is unset or names a file
/// that does not load: a `Learned` arm that silently fell back to the
/// deployed table would report a null that measured nothing.
pub fn learned_priorities() -> &'static LearnedPriorities {
    LEARNED_PRIORITIES.get_or_init(|| {
        let path = std::env::var(LEARNED_PRIORITIES_ENV).unwrap_or_else(|_| {
            panic!("RolloutPolicyKind::Learned needs {LEARNED_PRIORITIES_ENV}=<table.json>")
        });
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("{LEARNED_PRIORITIES_ENV}={path}: {e}"));
        LearnedPriorities::from_json(&text).unwrap_or_else(|e| panic!("{path}: {e}"))
    })
}

/// Install the process-wide table explicitly (tests, embedders).
///
/// # Errors
/// Returns the table back when one is already installed or loaded.
pub fn install_learned_priorities(table: LearnedPriorities) -> Result<(), LearnedPriorities> {
    LEARNED_PRIORITIES.set(table)
}

/// One legal action as the Heuristic rollout prices it — the training row
/// `rollout_policy_labels` writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PricedAction {
    /// The card a `PlayCard` resolves to; `None` for every other action.
    pub card: Option<CardId>,
    /// [`play_ordering`] of that card (the value a fitted table replaces).
    pub base: i16,
    /// The final softmax priority the rollout uses.
    pub priority: i16,
    /// Whether `priority` moves one-for-one with `base`. `false` where a
    /// context rule returns a fixed constant (the lethal-turn Defend /
    /// Strike cliffs) or for non-card actions, so a fitted base cannot
    /// change it.
    pub tracks_base: bool,
}

/// How the deployed Heuristic rollout would treat `actions` at `state`:
/// whether the coverage gate sends the step to Uniform, whether the
/// obvious-kill shortcut fires, and every action's priority.
#[derive(Debug, Clone)]
pub struct RolloutPricing {
    pub uniform_fallback: bool,
    pub obvious_kill: Option<usize>,
    pub actions: Vec<PricedAction>,
}

/// Price `actions` exactly as [`pick_rollout_action_with`] would, without
/// sampling.
#[must_use]
pub fn rollout_pricing(
    state: &CombatState,
    actions: &[Action],
    end_turn_priority: i16,
) -> RolloutPricing {
    let ctx = HpAwareCtx::compute(state, IncomingDamage::Base);
    let character = state.player.character;
    let mut play_cards = 0u32;
    let mut covered = 0u32;
    let priced: Vec<PricedAction> = actions
        .iter()
        .map(|a| match a {
            Action::PlayCard { hand_index, target } => {
                play_cards += 1;
                let card = state
                    .player
                    .hand
                    .get(*hand_index as usize)
                    .and_then(|uid| state.cards.get(uid.0.get() as usize - 1));
                match card {
                    Some(card) => {
                        let base = play_ordering(card.card_id, character);
                        if base != NEUTRAL_PRIORITY {
                            covered += 1;
                        }
                        let priority = play_card_priority(state, card, *target, base, &ctx);
                        let bumped = play_card_priority(state, card, *target, base + 1, &ctx);
                        PricedAction {
                            card: Some(card.card_id),
                            base,
                            priority,
                            tracks_base: bumped == priority + 1,
                        }
                    }
                    None => PricedAction {
                        card: None,
                        base: NEUTRAL_PRIORITY,
                        priority: NEUTRAL_PRIORITY,
                        tracks_base: false,
                    },
                }
            }
            other => {
                let p = if matches!(other, Action::EndTurn) {
                    end_turn_priority
                } else {
                    NEUTRAL_PRIORITY
                };
                PricedAction {
                    card: None,
                    base: p,
                    priority: p,
                    tracks_base: false,
                }
            }
        })
        .collect();
    #[allow(clippy::cast_precision_loss)]
    let uniform_fallback =
        play_cards > 0 && (covered as f32 / play_cards as f32) < COVERAGE_THRESHOLD;
    RolloutPricing {
        uniform_fallback,
        obvious_kill: obvious_kill_action(state, actions),
        actions: priced,
    }
}

/// Per-call cache for [`action_priority`]'s HP-aware bump. Computed
/// once per [`pick_action_heuristic`] call (bead `sts2-rs-5dw7`).
#[derive(Clone, Copy, Debug)]
struct HpAwareCtx {
    /// `true` when player's `current_hp + block` is at-or-below the
    /// sum of all telegraphed enemy attack damage this turn. When set,
    /// [`action_priority`] bumps Defend-class cards into the top
    /// bucket and Strike-class cards into the bottom bucket.
    is_lethal_turn: bool,
    /// Bead `sts2-rs-d9qd` (zlhm.1): `true` when player would die
    /// within [`SUSTAINED_PRESSURE_HORIZON`] turns of unblocked
    /// telegraphed damage AND can't kill the weakest alive enemy this
    /// turn (so offensive counter-play isn't available). Mutually
    /// exclusive with `is_lethal_turn` (strict single-turn arm wins
    /// on overlap). When set, [`action_priority`] applies the same
    /// Defend bump as the lethal-turn arm. The
    /// `can-kill-weakest-this-turn` gate is the tightened replacement
    /// for the zlhm probe's `horizon_burst < enemy_hp_sum` gate that
    /// over-fired on multi-enemy elites — see
    /// `docs/perf/2026-05-17-zlhm-sustained-pressure-probe.md`.
    is_sustained_pressure: bool,
    /// Bead `sts2-rs-6g96`: slot index (`enemy_uids[i]`) of the
    /// lowest-hp ALIVE enemy when 2+ enemies are alive. `None` in
    /// single-target fights (no targeting choice to bias). When set,
    /// [`action_priority`] subtracts [`FOCUS_FIRE_PRIORITY_BONUS`]
    /// from attack-class PlayCard actions targeting this slot.
    lowest_hp_enemy_target: Option<u8>,
    /// GH #293: graded defensive urgency in `0..=DEFEND_THREAT_MAX_BONUS_DEFAULT`,
    /// `incoming · MAX / effective_hp` clamped to the ceiling. Subtracted
    /// from any block-bearing card's priority by [`play_card_priority`],
    /// giving the rollout a *gradient* on incoming damage where it
    /// previously had only the two death-avoidance cliffs. Zero when
    /// nothing is telegraphed, so a quiet turn is unaffected.
    ///
    /// Integer math on purpose: `HpAwareCtx::compute` is the hottest
    /// cadence in the search (once per rollout step / selection descent),
    /// and an integer ratio keeps this bit-reproducible across platforms
    /// the way [`softmax_weight`]'s memo table does.
    defend_threat_bonus: i16,
}

impl HpAwareCtx {
    /// `incoming` picks how a telegraphed attack is read (the base move-table
    /// number, or the displayed one).
    fn compute(state: &CombatState, incoming_damage: IncomingDamage) -> Self {
        // Single fused pass over the enemy list (perf-investigation
        // 2026-07-04). `compute` runs once per rollout step / selection
        // descent — the hottest cadence in the search — and the four
        // aggregates below each used to re-walk `enemy_uids` with a
        // creature `find` per uid. One walk computes identical results
        // in the same iteration order:
        //
        //   * `incoming` — telegraphed enemy attack damage this turn,
        //     read the `incoming_damage` way (base move-table number or
        //     the displayed one); non-attack intents count 0.
        //   * `enemy_hp_sum` — HP sum over ALL resolved enemies (dead
        //     ones contribute 0 via `.max(0)`, matching the original
        //     unfiltered scan).
        //   * `smallest_alive_enemy_hp` — min HP over alive enemies
        //     (`0` when none are alive, matching `.min().unwrap_or(0)`).
        //   * `alive_count` / `lowest_hp_slot` — focus-fire slot pick
        //     (bead `sts2-rs-6g96`). Strict `<` preserves `min_by_key`'s
        //     first-minimum tie-break; the slot must fit `u8` exactly as
        //     in the original loop (perf-investigation 2026-06-07).
        let mut incoming: i32 = 0;
        let mut enemy_hp_sum: i32 = 0;
        let mut smallest_alive_hp = i32::MAX;
        let mut any_alive = false;
        let mut alive_count: u32 = 0;
        let mut lowest_hp = i32::MAX;
        let mut lowest_hp_slot: Option<u8> = None;
        for (slot, uid) in state.enemy_uids.iter().enumerate() {
            let Some(c) = sts2_core::creature_by_id(&state.creatures, uid.0) else {
                continue;
            };
            enemy_hp_sum += c.current_hp.max(0);
            if c.current_hp > 0 {
                incoming += incoming_damage.total(state, c);
                any_alive = true;
                if c.current_hp < smallest_alive_hp {
                    smallest_alive_hp = c.current_hp;
                }
                if let Ok(slot_u8) = u8::try_from(slot) {
                    alive_count += 1;
                    if c.current_hp < lowest_hp {
                        lowest_hp = c.current_hp;
                        lowest_hp_slot = Some(slot_u8);
                    }
                }
            }
        }
        let smallest_alive_enemy_hp = if any_alive { smallest_alive_hp } else { 0 };
        let effective_hp = state.player.current_hp.saturating_add(state.player.block);
        let horizon_incoming = incoming.saturating_mul(SUSTAINED_PRESSURE_HORIZON);
        // Bead `sts2-rs-it3o`: when the player has enough damage in hand
        // (accounting for energy) to kill every alive enemy this turn,
        // suppress the lethal-turn flag — Strike-the-kill is the correct
        // play, not Defend (the enemy's telegraphed attack never lands).
        // Approximation: sum of base_damage over attack cards in hand
        // whose cost is affordable, greedy from cheapest first.
        //
        // Perf 2026-07-04: the hand walk (+ sort) is only CONSULTED when
        // one of the pressure arms can fire — `is_lethal_turn` requires
        // `incoming > 0 && effective_hp <= incoming` and
        // `is_sustained_pressure` requires
        // `incoming > 0 && effective_hp <= horizon_incoming` (a superset,
        // since `horizon_incoming >= incoming` for positive incoming). In
        // the safe common case both predicates are false regardless of
        // the potential, so skipping the walk is behavior-identical.
        let player_lethal_potential = if incoming > 0 && effective_hp <= horizon_incoming {
            max_player_attack_damage_this_turn(state)
        } else {
            0
        };
        let can_clear_enemies = enemy_hp_sum > 0 && player_lethal_potential >= enemy_hp_sum;
        let is_lethal_turn = incoming > 0 && effective_hp <= incoming && !can_clear_enemies;
        // Bead `sts2-rs-d9qd` (zlhm.1): sustained-pressure predicate.
        // Fires when `effective_hp <= incoming × HORIZON` AND the
        // player can't kill the weakest alive enemy this turn. The
        // can-kill-weakest gate is the tightened replacement for the
        // zlhm probe's `horizon_burst < enemy_hp_sum` — that compared
        // total burst against total enemy HP, but for multi-enemy
        // elites (DECIMILLIPEDE / SKULKING_COLONY) the correct
        // counter-play is to one-shot a single threat and let next-
        // turn incoming drop. If `player_lethal_potential` can hit
        // the smallest alive enemy's HP, offensive play wins; only
        // when it can't (PHANTASMAL_GARDENERS: 60-HP gardener vs ~18
        // hand damage) does Defend stack-up become the right call.
        let can_kill_weakest =
            smallest_alive_enemy_hp > 0 && player_lethal_potential >= smallest_alive_enemy_hp;
        let is_sustained_pressure = !is_lethal_turn
            && incoming > 0
            && effective_hp <= horizon_incoming
            && smallest_alive_enemy_hp > 0
            && !can_kill_weakest;
        // Bead `sts2-rs-ubia`: scaling-power UCT prior fires when
        // multi-turn payback is plausible inside the rollout horizon.
        // Bead `sts2-rs-6g96`: focus-fire only when 2+ enemies are alive
        // — single-target fights have no targeting choice to bias.
        let lowest_hp_enemy_target = if alive_count >= 2 {
            lowest_hp_slot
        } else {
            None
        };
        // GH #293: the graded defensive ramp. `effective_hp` already
        // includes standing block, so the bonus self-limits as block goes
        // up — stacking a third Defend against a small hit scores much
        // lower than the first Defend against a big one.
        let max_bonus = defend_threat_max_bonus();
        let defend_threat_bonus: i16 = if incoming > 0 && effective_hp > 0 && max_bonus > 0 {
            let scaled = incoming.saturating_mul(i32::from(max_bonus)) / effective_hp;
            i16::try_from(scaled.min(i32::from(max_bonus))).unwrap_or(max_bonus)
        } else {
            0
        };
        Self {
            is_lethal_turn,
            is_sustained_pressure,
            lowest_hp_enemy_target,
            defend_threat_bonus,
        }
    }
}

/// Bead `sts2-rs-it3o`: greedy upper bound on attack damage the player
/// can deliver this turn. Walks the hand, sorts affordable attack cards
/// by ascending cost, and picks the most-damage-per-energy slice that
/// fits the current energy budget. Coarse: ignores Vulnerable/Strength,
/// Block/AOE distinctions, and target routing (treats damage as fungible
/// across enemies). Tight enough to gate the lethal-turn prior in the
/// common case where a Strike or two clears the front enemy.
fn max_player_attack_damage_this_turn(state: &CombatState) -> i32 {
    use sts2_core::cards::CardType;
    let energy = i32::from(state.player.energy.max(0));
    if energy <= 0 {
        return 0;
    }
    let mut attacks: smallvec::SmallVec<[(i32, i32); 16]> = smallvec::SmallVec::new();
    for uid in &state.player.hand {
        let idx = uid.0.get() as usize - 1;
        let Some(card) = state.cards.get(idx) else {
            continue;
        };
        if !matches!(card.card_type, CardType::Attack) {
            continue;
        }
        let cost = i32::from(card.cost.max(0));
        if cost > energy {
            continue;
        }
        attacks.push((cost.max(0), card.base_damage.max(0)));
    }
    // Sort by ascending cost, descending damage so the greedy pick
    // grabs the highest-damage cheap card first.
    attacks.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| b.1.cmp(&a.1)));
    let mut budget = energy;
    let mut total = 0_i32;
    for (cost, dmg) in attacks {
        if cost == 0 {
            total = total.saturating_add(dmg);
        } else if budget >= cost {
            total = total.saturating_add(dmg);
            budget -= cost;
        }
    }
    total
}

/// Bead `sts2-rs-pifyc` Part 2: is `card_id` an attack that *also*
/// applies a debuff (typically Vulnerable) to its target? The set is
/// closed at this pin: Bash (apply Vulnerable), Thunderclap (apply
/// Vulnerable to all enemies), Uppercut (apply Weak + Vulnerable).
/// IronWave is explicitly excluded — at `sts2-rl-agent@01c094d` its
/// play handler does damage + block only, no debuff
/// (`crates/sts2-content/src/cards/ironclad.rs::play_iron_wave`).
/// Upgrade variants share the same `CardId` at this pin (`Card.upgraded`
/// is a flag, not a separate ID).
#[must_use]
pub const fn is_debuff_attaching_attack(card_id: CardId) -> bool {
    matches!(
        card_id,
        CardId::Bash | CardId::Thunderclap | CardId::Uppercut
    )
}

/// Bead `sts2-rs-pifyc` Part 2: returns
/// `DEBUFF_SETUP_PRIORITY_BONUS` when all three setup-bonus gates
/// hold, else `0`. Gates (must ALL hold):
///   1. `target_hp > attack_damage` — target survives this hit so
///      the applied Vulnerable can affect a future hit. If we're
///      about to 1-shot the target, the debuff is wasted and
///      Bash / Thunderclap / Uppercut should not pre-empt a cheaper
///      Strike.
///   2. Target is NOT already Vulnerable (`amount > 0`). Re-applying
///      Vulnerable doesn't double-stack damage; if the target is
///      already debuffed, the setup-bonus benefit is zero.
///   3. Player has at least one OTHER attack in hand whose
///      `base_damage >= card.base_damage` — i.e. there's a worthwhile
///      follow-up to cash in the doubled-damage window.
///
/// All gates fail for the single-card / single-shot / already-
/// debuffed cases, leaving the heuristic at its baseline
/// `play_ordering` and the 2026-05-21 `strike_when_bash_affordable`
/// blunders are addressed without regressing winnable Strike-first
/// lines.
fn debuff_setup_priority_delta(
    state: &CombatState,
    card: &sts2_core::cards::Card,
    target: Option<u8>,
) -> i16 {
    use sts2_core::cards::CardType;
    use sts2_core::PowerKind;
    let Some(slot) = target else {
        return 0;
    };
    let Some(enemy_uid) = state.enemy_uids.get(slot as usize) else {
        return 0;
    };
    let Some(enemy) = sts2_core::creature_by_id(&state.creatures, enemy_uid.0) else {
        return 0;
    };
    if enemy.current_hp <= 0 {
        return 0;
    }
    let player_strength: i32 = state
        .player
        .creature_uid
        .and_then(|uid| sts2_core::creature_by_id(&state.creatures, uid.0))
        .map_or(0, |c| {
            c.powers
                .iter()
                .filter_map(|puid| state.powers.get_by_id(puid.0))
                .filter(|p| p.kind == PowerKind::Strength)
                .map(|p| p.amount)
                .sum::<i32>()
        });
    let dmg = card.base_damage.saturating_add(player_strength).max(0);
    // Gate 1: target must survive this hit.
    if dmg >= enemy.current_hp {
        return 0;
    }
    // Gate 2: target must not already be Vulnerable.
    let already_vulnerable = enemy
        .powers
        .iter()
        .filter_map(|puid| state.powers.get_by_id(puid.0))
        .any(|p| p.kind == PowerKind::Vulnerable && p.amount > 0);
    if already_vulnerable {
        return 0;
    }
    // Gate 3: at least one OTHER attack in hand with base_damage >= card.base_damage.
    let setup_dmg = card.base_damage.max(0);
    let mut has_followup = false;
    for uid in &state.player.hand {
        if *uid == card.uid {
            continue;
        }
        let idx = uid.0.get() as usize - 1;
        let Some(other) = state.cards.get(idx) else {
            continue;
        };
        if !matches!(other.card_type, CardType::Attack) {
            continue;
        }
        if other.base_damage.max(0) >= setup_dmg {
            has_followup = true;
            break;
        }
    }
    if !has_followup {
        return 0;
    }
    DEBUFF_SETUP_PRIORITY_BONUS
}

/// Priority for an entire [`Action`]. PlayCard looks up the card's
/// CardId in [`play_ordering`]; EndTurn / DrinkPotion / Channel / Evoke
/// / ResolveChoice get neutral or penalty buckets.
///
/// Bead `sts2-rs-5dw7` HP-aware bump: when `ctx.is_lethal_turn`,
/// Defend-class cards (Defend* / IronWave / Entrench) jump to priority
/// 25 (top of the in-play range), and Strike-class basics drop to 75
/// (between status cards and curses). This is a narrow override on top
/// of the C++ play_ordering table — every other card keeps its baseline
/// priority so the heuristic still mixes in scaling/setup plays.
fn action_priority(state: &CombatState, action: &Action, ctx: &HpAwareCtx) -> i16 {
    match action {
        Action::PlayCard { hand_index, target } => {
            let Some(card_uid) = state.player.hand.get(*hand_index as usize) else {
                return NEUTRAL_PRIORITY;
            };
            let idx = card_uid.0.get() as usize - 1;
            let Some(card) = state.cards.get(idx) else {
                return NEUTRAL_PRIORITY;
            };
            let base = play_ordering(card.card_id, state.player.character);
            play_card_priority(state, card, *target, base, ctx)
        }
        Action::EndTurn => END_TURN_PRIORITY,
        // Potions / orb channels / choices are not in the C++ priority
        // table. Use the neutral bucket so the softmax still considers
        // them but doesn't strongly prefer them over card plays.
        Action::UsePotion { .. }
        | Action::DrinkPotion { .. }
        | Action::Channel { .. }
        | Action::Evoke { .. }
        | Action::ResolveChoice { .. } => NEUTRAL_PRIORITY,
    }
}

/// Priority for a resolved `PlayCard` — the [`action_priority`] card arm
/// with the card lookup + [`play_ordering`] hoisted out so
/// [`pick_rollout_action`] can reuse the values it already resolved for
/// the coverage gate. `base` MUST be
/// `play_ordering(card.card_id, state.player.character)`.
fn play_card_priority(
    state: &CombatState,
    card: &sts2_core::cards::Card,
    target: Option<u8>,
    base: i16,
    ctx: &HpAwareCtx,
) -> i16 {
    use sts2_core::cards::CardType;
    // Bead `sts2-rs-d9qd` (zlhm.1): the sustained-pressure arm
    // shares the 5dw7 Defend bump — same priority constants,
    // same Strike demotion. Mutual exclusion at
    // HpAwareCtx::compute means at most one arm fires per call.
    if ctx.is_lethal_turn || ctx.is_sustained_pressure {
        if is_defend_class(card.card_id) {
            return LETHAL_DEFEND_PRIORITY;
        }
        if is_strike_basic(card.card_id) {
            return LETHAL_STRIKE_PRIORITY;
        }
    }
    // GH #293: graded defensive ramp for ANY block-bearing card. Runs after
    // the two cliffs above so their tuned constants are untouched — an
    // `is_defend_class` card on a lethal turn already returned.
    //
    // Keyed on `base_block` rather than the [`is_defend_class`] allowlist,
    // which names only 7 cards (the five basic Defends, IronWave, Entrench)
    // against 74 block-bearing cards in content. That allowlist is the same
    // failure mode as GH #375: a hand-maintained set that silently stops
    // covering the roster as content grows.
    //
    // NOT a fix for #293's Regent examples specifically — Guards is a summon
    // Skill and Parry is a Power, both `base_block: 0`, so neither is
    // block-bearing by this test. That is suspect 3 and stays open.
    if card.base_block > 0 && ctx.defend_threat_bonus > 0 {
        return base.saturating_sub(ctx.defend_threat_bonus);
    }
    // Bead `sts2-rs-pifyc` Part 2: debuff-setup bonus. When a
    // debuff-attaching attack (Bash / Thunderclap / Uppercut)
    // targets an enemy that will SURVIVE this hit (so the
    // applied Vulnerable matters next turn), is not already
    // debuffed (Vulnerable inactive), AND the player has a
    // follow-up attack of equal-or-higher base damage in
    // hand to cash in the doubled-damage window, subtract
    // `DEBUFF_SETUP_PRIORITY_BONUS` so the setup card plays
    // before the follow-up.
    let debuff_setup_bonus = if is_debuff_attaching_attack(card.card_id) {
        debuff_setup_priority_delta(state, card, target)
    } else {
        0
    };
    // GH #403: draw-window bonus — see [`draw_window_priority_delta`].
    // Stacks with the two bonuses around it (a card can be both an
    // attack on the focus target and inside its draw window).
    let draw_window_bonus = draw_window_priority_delta(state, card);
    // Bead `sts2-rs-6g96`: focus-fire bias. Subtract the bonus
    // when an Attack-class play targets the lowest-hp alive
    // enemy slot — biases the softmax toward closing kills
    // rather than spreading chip damage across all enemies.
    if matches!(card.card_type, CardType::Attack) {
        if let (Some(t), Some(focus)) = (target, ctx.lowest_hp_enemy_target) {
            if t == focus {
                return base - FOCUS_FIRE_PRIORITY_BONUS - debuff_setup_bonus - draw_window_bonus;
            }
        }
    }
    base - debuff_setup_bonus - draw_window_bonus
}

const fn is_defend_class(id: CardId) -> bool {
    matches!(
        id,
        CardId::DefendIronclad
            | CardId::DefendSilent
            | CardId::DefendDefect
            | CardId::DefendNecrobinder
            | CardId::DefendRegent
            | CardId::IronWave
            | CardId::Entrench
    )
}

/// Fraction of legal `PlayCard` actions in `state` / `actions` whose
/// referenced card has an explicitly classified [`play_ordering`]
/// priority (i.e. NOT the [`NEUTRAL_PRIORITY`] fallback). Bead
/// `sts2-rs-j0f6` — used by the rollout step to detect when the STS1
/// priority table is under-covering the current card pool and the
/// Heuristic policy should defer to Uniform.
///
/// Returns `1.0` when `actions` contains no `PlayCard` entries — there
/// is no priority-table dependency to over-fit on, so the heuristic is
/// safe to use (it falls through to neutral on EndTurn / DrinkPotion /
/// Channel etc. anyway).
///
/// # Example
///
/// ```
/// use sts2_core::{Action, CombatState};
/// use sts2_mcts::expert::rollout_coverage_fraction;
///
/// // No-PlayCard branch — empty slice + EndTurn-only state both
/// // return 1.0 by construction (no priority-table dependency).
/// let state = CombatState::default();
/// assert_eq!(rollout_coverage_fraction(&state, &[]), 1.0);
/// assert_eq!(
///     rollout_coverage_fraction(&state, &[Action::EndTurn]),
///     1.0,
/// );
/// ```
#[must_use]
pub fn rollout_coverage_fraction(state: &CombatState, actions: &[Action]) -> f32 {
    let mut play_card_count: u32 = 0;
    let mut covered: u32 = 0;
    for a in actions {
        if let Action::PlayCard { hand_index, .. } = a {
            play_card_count = play_card_count.saturating_add(1);
            let Some(card_uid) = state.player.hand.get(*hand_index as usize) else {
                continue;
            };
            let idx = card_uid.0.get() as usize - 1;
            let Some(card) = state.cards.get(idx) else {
                continue;
            };
            if play_ordering(card.card_id, state.player.character) != NEUTRAL_PRIORITY {
                covered = covered.saturating_add(1);
            }
        }
    }
    if play_card_count == 0 {
        return 1.0;
    }
    #[allow(clippy::cast_precision_loss)]
    let frac = covered as f32 / play_card_count as f32;
    frac
}

/// GH #403 gate: does the hand hold a card whose play-count-gated draw
/// rider would still fire if played now? Short-circuits on the first hit,
/// and the whole walk is skipped for the overwhelmingly common hand that
/// holds no such card because [`play_count_gated_draw_max`] is a one-arm
/// match on `CardId`.
#[must_use]
pub fn has_live_draw_window(state: &CombatState) -> bool {
    state.player.hand.iter().any(|uid| {
        state
            .cards
            .get(uid.0.get() as usize - 1)
            .is_some_and(|card| draw_window_is_live(state, card))
    })
}

const fn is_strike_basic(id: CardId) -> bool {
    matches!(
        id,
        CardId::StrikeIronclad
            | CardId::StrikeSilent
            | CardId::StrikeDefect
            | CardId::StrikeNecrobinder
            | CardId::StrikeRegent
    )
}

/// GH #403: root-side Q tie-break bonus for a `PlayCard` action whose card
/// is sitting inside a live play-count-gated draw window (today: FTL).
/// Returns [`DRAW_WINDOW_ROOT_Q_BONUS`] for such an action, `0.0` otherwise.
/// Callers gate on [`has_live_draw_window`] so the per-action work is
/// skipped entirely for every hand that holds no such card.
///
/// This is the half of the #403 fix that actually moves the *decision*, and
/// it has to be a root Q adjustment rather than a UCT edge prior because
/// [`crate::Mcts::search`] returns `argmax(Q)` (visits are only the
/// exact-tie break). A UCT prior in the shape of
/// a UCT edge prior reallocates visits and nothing else — and
/// on the unit fixture the visit skew it induces (762 vs 125) actively
/// *inverts* the pick, because the better-explored child converges to an
/// honest value while its shallow sibling keeps an optimistic small-sample
/// mean.
///
/// Sizing: the true edge here is one card, worth ~0.002 of leaf value once
/// [`sts2_core::evaluate_in_progress`] has divided by its 114.56 normalizer
/// — an order of magnitude under the ~0.02 mid-search Q noise at 1000 sims,
/// so the search will never resolve it no matter how long it thinks. 0.05
/// sits clearly above that noise floor while staying well under the
/// 0.10 / 0.20 / 0.25 magnitudes the removed game-knowledge edge priors
/// used (see `docs/experiments/2026-08-22-game-edge-priors-null.md`), so it
/// decides coin-flips without overruling a preference the search earned.
///
/// Swept over 300 MCTS-driven fights on a hard Act-1 mix (normal / elite /
/// boss), reporting the share of FTL plays that landed inside the window
/// and mean end-of-fight HP as the guard rail:
///
/// ```text
/// bonus   in-window        mean HP
/// 0.00    295/436 (67.7%)  59.8   <- baseline
/// 0.02    471/544 (86.6%)  59.7
/// 0.05    530/581 (91.2%)  59.9   <- shipped
/// 0.10    547/585 (93.5%)  60.1
/// ```
///
/// HP is flat across the sweep (all arms won 300/300, so winrate does not
/// discriminate at this fixture difficulty) — the ordering is being fixed
/// without paying for it elsewhere. 0.10 buys another 2.3pp but starts
/// overriding earned preferences: on a hand holding Bash it flips a
/// Vulnerable-setup-first line the search had genuinely valued above the
/// draw. Reproduce with
/// `cargo run --release -p sts2-bench --example ftl_window_probe`.
///
/// Known limit, deliberately not tuned away: this is a tie-break, so a
/// position where some other play is worth more than 0.05 of backed-up
/// value still gets that play first, even though FTL costs 0 and would
/// have been free to fire first.
#[must_use]
pub fn draw_window_root_q_bonus(state: &CombatState, action: &Action) -> f32 {
    let Action::PlayCard { hand_index, .. } = action else {
        return 0.0;
    };
    let Some(card_uid) = state.player.hand.get(*hand_index as usize) else {
        return 0.0;
    };
    let idx = card_uid.0.get() as usize - 1;
    let Some(card) = state.cards.get(idx) else {
        return 0.0;
    };
    if draw_window_is_live(state, card) {
        DRAW_WINDOW_ROOT_Q_BONUS
    } else {
        0.0
    }
}

/// Pluggable rollout policy. Used by [`crate::Mcts::rollout`] to pick the
/// next action during a simulation. The default is [`Heuristic`].
pub trait RolloutPolicy {
    /// Pick an index into `actions`. Must be `< actions.len()`. Called
    /// every rollout step; keep cheap.
    fn pick<R: Rng>(&self, rng: &mut R, state: &CombatState, actions: &[Action]) -> usize;
}

/// Uniform-random rollout policy. The pre-`dfv1` baseline.
#[derive(Clone, Copy, Debug, Default)]
pub struct Uniform;

impl RolloutPolicy for Uniform {
    fn pick<R: Rng>(&self, rng: &mut R, _state: &CombatState, actions: &[Action]) -> usize {
        rng.gen_range(0..actions.len())
    }
}

/// Softmax-over-expert-priority rollout policy (`tau = 32`, see
/// [`SOFTMAX_TAU`]). Default rollout policy at
/// [`crate::MctsConfig::default`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Heuristic;

impl RolloutPolicy for Heuristic {
    fn pick<R: Rng>(&self, rng: &mut R, state: &CombatState, actions: &[Action]) -> usize {
        if let Some(idx) = obvious_kill_action(state, actions) {
            return idx;
        }
        pick_action_heuristic(rng, state, actions)
    }
}

/// Bead `sts2-rs-j9GGr` (GitHub issue #9): rollout-side kill detection.
/// Returns `Some(idx)` into `actions` when a single-target attack
/// PlayCard would lethal an alive enemy this turn — estimated damage
/// (base + player Strength, ×1.5 if target has Vulnerable) ≥ target HP.
/// First match wins; we don't rank between multiple lethal options.
///
/// Closes the highest-leverage gap from the 2026-05-20 MCTS engine audit:
/// the heuristic rollout previously followed `play_ordering` priority
/// blindly and would play Strike-for-6 next to a 2-HP Vulnerable enemy
/// while ranking Bash (priority 25) ahead of Strike (55). Skipping a
/// free kill in rollout warps the Q signal MCTS backs up; detecting
/// the lethal here is one branch in front of the softmax sample.
///
/// Multi-hit cards (TwinStrike, Whirlwind, Thunderclap, …) are NOT
/// modeled — the `Card` struct doesn't encode a generic `hits` field,
/// so we'd need a per-card-id table to extend. Conservative single-hit
/// detection is preferred: under-estimating damage misses some kills
/// but never promotes a non-lethal attack. The emergency-block half
/// of the audit suggestion is already handled by
/// [`HpAwareCtx::is_lethal_turn`] inside [`pick_action_heuristic`].
///
/// The damage projection mirrors the engine pipeline so it stays
/// genuinely conservative (never a false kill): additive Strength,
/// then the multiplicative Weak (×3/4) and Vulnerable (×3/2) listeners
/// composed with a single floor, and the kill threshold is the target's
/// `current_hp + block` (block absorbs before HP). Earlier revisions
/// ignored the player's Weak debuff and the target's block, which
/// over-counted damage and promoted attacks that only chip a blocked or
/// weakened enemy — warping the Q the rollout backs up.
fn obvious_kill_action(state: &CombatState, actions: &[Action]) -> Option<usize> {
    use sts2_core::cards::CardType;
    use sts2_core::PowerKind;

    let (player_strength, player_weak): (i32, bool) = state
        .player
        .creature_uid
        .and_then(|uid| sts2_core::creature_by_id(&state.creatures, uid.0))
        .map_or((0, false), |c| {
            let mut strength = 0i32;
            let mut weak = false;
            for p in c
                .powers
                .iter()
                .filter_map(|puid| state.powers.get_by_id(puid.0))
            {
                match p.kind {
                    PowerKind::Strength => strength = strength.saturating_add(p.amount),
                    PowerKind::Weak if p.amount > 0 => weak = true,
                    _ => {}
                }
            }
            (strength, weak)
        });

    let energy = i32::from(state.player.energy.max(0));

    for (idx, action) in actions.iter().enumerate() {
        let Action::PlayCard { hand_index, target } = action else {
            continue;
        };
        let Some(slot) = target else { continue };
        let Some(card_uid) = state.player.hand.get(*hand_index as usize) else {
            continue;
        };
        let cidx = card_uid.0.get() as usize - 1;
        let Some(card) = state.cards.get(cidx) else {
            continue;
        };
        if !matches!(card.card_type, CardType::Attack) || card.base_damage <= 0 {
            continue;
        }
        if i32::from(card.cost.max(0)) > energy {
            continue;
        }
        let Some(enemy_uid) = state.enemy_uids.get(*slot as usize) else {
            continue;
        };
        let Some(enemy) = sts2_core::creature_by_id(&state.creatures, enemy_uid.0) else {
            continue;
        };
        if enemy.current_hp <= 0 {
            continue;
        }
        let base_plus_str = card.base_damage.saturating_add(player_strength).max(0);
        let vulnerable = enemy
            .powers
            .iter()
            .filter_map(|puid| state.powers.get_by_id(puid.0))
            .any(|p| p.kind == PowerKind::Vulnerable && p.amount > 0);
        // Mirror the engine: Strength is already folded in additively;
        // the multiplicative Weak (×3/4) and Vulnerable (×3/2) listeners
        // compose and floor once (hooks.rs carries the value as float
        // through every multiplicative listener, rounding at the end).
        let mut num: i64 = 1;
        let mut den: i64 = 1;
        if player_weak {
            num *= 3;
            den *= 4;
        }
        if vulnerable {
            num *= 3;
            den *= 2;
        }
        #[allow(clippy::cast_possible_truncation)]
        let dmg = (i64::from(base_plus_str) * num / den) as i32;
        // Block absorbs damage before HP: a kill must beat
        // current_hp + block, not current_hp alone.
        if dmg >= enemy.current_hp.saturating_add(enemy.block.max(0)) {
            return Some(idx);
        }
    }
    None
}

/// Selector for the rollout policy. Surfaced through
/// [`crate::MctsConfig::rollout_policy`] so callers can opt into the
/// pre-`dfv1` uniform baseline (e.g. for legacy parity tests) without
/// reaching for the [`RolloutPolicy`] trait directly.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RolloutPolicyKind {
    /// Uniform-random rollout — the baseline before bead `sts2-rs-dfv1`.
    Uniform,
    /// Softmax over expert play priority (`tau = 32`, see
    /// [`SOFTMAX_TAU`]). Default.
    #[default]
    Heuristic,
    /// [`Self::Heuristic`] with each card's base priority read from the
    /// process-wide fitted table ([`learned_priorities`], loaded from
    /// `STS2_ROLLOUT_PRIORITIES`) instead of the STS1-derived ladder.
    /// Experimental: an arm for `mcts_book_sweep`, not a deployed default.
    Learned,
    /// The playout net ([`crate::rollout_net`], GH #777): the net named by
    /// `STS2_ROLLOUT_NET` when set, else the built-in net for the fight's
    /// character. A character without one (every character but Ironclad
    /// today) plays [`Self::Heuristic`] unchanged. The deployed honest
    /// dialect selects it ([`crate::MctsConfig::apply_mode_dialect`]).
    Net,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn play_ordering_known_buckets() {
        assert_eq!(play_ordering(CardId::Bash, None), 25);
        assert_eq!(play_ordering(CardId::StrikeIronclad, None), 55);
        assert_eq!(play_ordering(CardId::DefendIronclad, None), 55);
        assert_eq!(play_ordering(CardId::Slimed, None), 70);
        assert_eq!(play_ordering(CardId::Apotheosis, None), -20);
    }

    /// Bead `sts2-rs-g8sy4` (re-close pass 14): extend
    /// `play_ordering_known_buckets` to cover EVERY bucket on the
    /// priority ladder with one canary card each. The existing
    /// canary test pins 5 buckets {-20, 25, 55, 70}; this adds 10
    /// more {-2, 0, 10, 15, 20, 22, 27, 30, 35, 50} so a bucket
    /// reorder (e.g. moving Wish from 20 → 22) trips the canary
    /// regardless of which bucket it lands in.
    ///
    /// Each canary is chosen to be a g8sy4-classified card whose
    /// bucket assignment is documented in the play_ordering source
    /// at the bucket's match arm (lines noted in the test).
    #[test]
    fn play_ordering_g8sy4_canary_per_bucket() {
        // Negative buckets — high priority (eager to play).
        assert_eq!(play_ordering(CardId::HelloWorldCard, None), -2);
        // Bucket 0.
        assert_eq!(play_ordering(CardId::RollingBoulder, None), 0);
        // Mid buckets.
        assert_eq!(play_ordering(CardId::ToricToughness, None), 10);
        assert_eq!(play_ordering(CardId::JuggernautCard, None), 15);
        assert_eq!(play_ordering(CardId::Wish, None), 20);
        assert_eq!(play_ordering(CardId::Stack, None), 22);
        assert_eq!(play_ordering(CardId::Whirlwind, None), 27);
        assert_eq!(play_ordering(CardId::Squash, None), 30);
        assert_eq!(play_ordering(CardId::Peck, None), 35);
        // Bucket 48 = filler attacks (Shiv straggler; off NEUTRAL=50).
        assert_eq!(play_ordering(CardId::Shiv, None), 48);
    }

    /// Bead `sts2-rs-a79i`: with an empty override map (the initial
    /// shape this bead ships), every character must agree with the
    /// `None` baseline on every card we care about. This pins the
    /// no-regression contract for the signature lift — any future
    /// override bead must update this test or the contract drifts.
    #[test]
    fn a79i_per_character_override_is_empty_at_lift() {
        let sample = [
            CardId::Bash,
            CardId::StrikeIronclad,
            CardId::DefendIronclad,
            CardId::Apotheosis,
            CardId::Slimed,
            // Bead `sts2-rs-g8sy4` (pass 6 extension): widen the
            // per-character-override no-regression check to cover the
            // 17 stragglers landed in pass 1 (c452126) — same contract
            // applies to them and a future override-arm bead must
            // update this list explicitly.
            CardId::BrightestFlame,
            CardId::FeedingFrenzyCard,
            CardId::HelloWorldCard,
            CardId::Distraction,
            CardId::Metamorphosis,
            CardId::Outmaneuver,
            CardId::Wish,
            CardId::Relax,
            CardId::Stack,
            CardId::Clash,
            CardId::Whistle,
            CardId::Rebound,
            CardId::Squash,
            CardId::MadScience,
            CardId::Peck,
            CardId::Shiv,
            CardId::ToricToughness,
        ];
        let characters = [
            CharacterId::Ironclad,
            CharacterId::Silent,
            CharacterId::Defect,
            CharacterId::Necrobinder,
            CharacterId::Regent,
        ];
        for card in sample {
            let base = play_ordering(card, None);
            for ch in characters {
                assert_eq!(
                    play_ordering(card, Some(ch)),
                    base,
                    "override map should be empty at a79i landing; {card:?} for {ch:?} diverged from {base}",
                );
            }
        }
    }

    #[test]
    fn play_ordering_g8sy4_top_fixture_card_buckets() {
        // Bead `sts2-rs-g8sy4`: every top-frequency missing card in
        // the IC fixture pool now classifies into the existing
        // bucket appropriate for its in-content role. Pins the
        // partial-audit lift (curses, statuses, EvilEye + 4 large-
        // damage/block colorless cards covering ~53 % of the 12 k
        // missing-card occurrences in /tmp/run-fixtures-full/the-
        // ironclad).
        // -2 bucket: Caltrops is a power (Thorns scaler).
        assert_eq!(play_ordering(CardId::Caltrops, None), -2);
        // 22 bucket: big-block defensives.
        assert_eq!(play_ordering(CardId::EvilEye, None), 22);
        assert_eq!(play_ordering(CardId::UltimateDefend, None), 22);
        // 25 bucket: 0-cost big-tempo attack.
        assert_eq!(play_ordering(CardId::ByrdSwoop, None), 25);
        // 30 bucket: mid attacks.
        assert_eq!(play_ordering(CardId::Maul, None), 30);
        assert_eq!(play_ordering(CardId::NeowsFury, None), 30);
        // 35 bucket: standard attacks.
        assert_eq!(play_ordering(CardId::UltimateStrike, None), 35);
        // 40 bucket: cheap AOE.
        assert_eq!(play_ordering(CardId::Exterminate, None), 40);
        // 60 bucket: STS2 status cards.
        assert_eq!(play_ordering(CardId::SpoilsMap, None), 60);
        assert_eq!(play_ordering(CardId::LanternKey, None), 60);
        assert_eq!(play_ordering(CardId::ByrdonisEgg, None), 60);
        // 70 bucket: STS2 curses.
        assert_eq!(play_ordering(CardId::Greed, None), 70);
        assert_eq!(play_ordering(CardId::Guilty, None), 70);
        assert_eq!(play_ordering(CardId::Debt, None), 70);
        assert_eq!(play_ordering(CardId::Folly, None), 70);
    }

    #[test]
    fn play_ordering_g198_ironclad_neutral_fill() {
        // Bead `sts2-rs-g198` (3f74 followup): every Ironclad card that
        // used to fall through to NEUTRAL_PRIORITY must now classify
        // into a non-neutral bucket. Sample one from each tier.
        assert_eq!(play_ordering(CardId::DrumOfBattleCard, None), -5);
        assert_eq!(play_ordering(CardId::AggressionCard, None), -2);
        assert_eq!(play_ordering(CardId::Pyre, None), 0);
        assert_eq!(play_ordering(CardId::StoneArmor, None), 5);
        assert_eq!(play_ordering(CardId::Unmovable, None), 10);
        assert_eq!(play_ordering(CardId::Stoke, None), 15);
        assert_eq!(play_ordering(CardId::OneTwoPunchCard, None), 20);
        assert_eq!(play_ordering(CardId::PactsEnd, None), 22);
        assert_eq!(play_ordering(CardId::Break, None), 25);
        assert_eq!(play_ordering(CardId::Mangle, None), 30);
        assert_eq!(play_ordering(CardId::Conflagration, None), 35);
        assert_eq!(play_ordering(CardId::Pillage, None), 40);
        assert_eq!(play_ordering(CardId::FightMe, None), 48);
    }

    #[test]
    fn play_ordering_g198_cross_character_basics() {
        // g198 cross-character first-pass: basics for every character
        // sit in the same 55 bucket as StrikeIronclad/DefendIronclad.
        for cid in [
            CardId::StrikeSilent,
            CardId::DefendSilent,
            CardId::StrikeDefect,
            CardId::DefendDefect,
            CardId::StrikeNecrobinder,
            CardId::DefendNecrobinder,
            CardId::StrikeRegent,
            CardId::DefendRegent,
        ] {
            assert_eq!(play_ordering(cid, None), 55, "{cid:?} should be a basic");
        }
    }

    #[test]
    fn play_ordering_g198_cross_character_powers() {
        // Sampled power buckets across Silent/Defect/Necrobinder.
        assert_eq!(play_ordering(CardId::NoxiousFumesCard, None), -2);
        assert_eq!(play_ordering(CardId::EnvenomCard, None), -2);
        assert_eq!(play_ordering(CardId::BiasedCognitionCard, None), -2);
        assert_eq!(play_ordering(CardId::MachineLearningCard, None), -2);
        assert_eq!(play_ordering(CardId::InfiniteBladesCard, None), 0);
        assert_eq!(play_ordering(CardId::EchoFormCard, None), 0);
        assert_eq!(play_ordering(CardId::NecroMasteryCard, None), 0);
        assert_eq!(play_ordering(CardId::Footwork, None), 5);
    }

    /// Bead `sts2-rs-g8sy4` (GitHub issue #8): every Silent / Defect /
    /// Colorless card in the deck-loadable pool (the spec_index lists)
    /// classifies into a non-NEUTRAL bucket. Pin one per bucket per
    /// character so future regressions surface immediately.
    #[test]
    fn play_ordering_g8sy4_silent_defect_colorless_buckets() {
        // -2 bucket: Silent / Defect scaling powers.
        assert_eq!(play_ordering(CardId::AfterimageCard, None), -2);
        assert_eq!(play_ordering(CardId::FanOfKnivesCard, None), -2);
        assert_eq!(play_ordering(CardId::PhantomBladesCard, None), -2);
        assert_eq!(play_ordering(CardId::MasterPlanner, None), -2);
        assert_eq!(play_ordering(CardId::Hailstorm, None), -2);
        assert_eq!(play_ordering(CardId::ConsumingShadow, None), -2);
        assert_eq!(play_ordering(CardId::BulkUp, None), -2);
        // 0 bucket: late-game-defining powers.
        assert_eq!(play_ordering(CardId::WraithForm, None), 0);
        assert_eq!(play_ordering(CardId::CalamityCard, None), 0);
        assert_eq!(play_ordering(CardId::EternalArmor, None), 0);
        assert_eq!(play_ordering(CardId::RollingBoulder, None), 0);
        // 5 bucket: situational utility powers.
        assert_eq!(play_ordering(CardId::WellLaidPlans, None), 5);
        assert_eq!(play_ordering(CardId::Tracking, None), 5);
        assert_eq!(play_ordering(CardId::SneakyCard, None), 5);
        assert_eq!(play_ordering(CardId::Subroutine, None), 5);
        assert_eq!(play_ordering(CardId::StormCard, None), 5);
        // 10 bucket: standard powers / debuffs.
        assert_eq!(play_ordering(CardId::DeadlyPoison, None), 10);
        assert_eq!(play_ordering(CardId::PiercingWail, None), 10);
        assert_eq!(play_ordering(CardId::OutbreakCard, None), 10);
        assert_eq!(play_ordering(CardId::Accelerant, None), 10);
        assert_eq!(play_ordering(CardId::Feral, None), 10);
        assert_eq!(play_ordering(CardId::Automation, None), 10);
        assert_eq!(play_ordering(CardId::BeaconOfHope, None), 10);
        // 20 bucket: card-flow utilities.
        assert_eq!(play_ordering(CardId::Acrobatics, None), 20);
        assert_eq!(play_ordering(CardId::Prepared, None), 20);
        assert_eq!(play_ordering(CardId::Zap, None), 20);
        assert_eq!(play_ordering(CardId::Coolheaded, None), 20);
        assert_eq!(play_ordering(CardId::MasterOfStrategy, None), 20);
        assert_eq!(play_ordering(CardId::Mimic, None), 20);
        // 22 bucket: big block / situational defense.
        assert_eq!(play_ordering(CardId::Survivor, None), 22);
        assert_eq!(play_ordering(CardId::Deflect, None), 22);
        assert_eq!(play_ordering(CardId::LegSweep, None), 22);
        assert_eq!(play_ordering(CardId::BootSequence, None), 22);
        assert_eq!(play_ordering(CardId::ShadowShield, None), 22);
        assert_eq!(play_ordering(CardId::TheBombCard, None), 22);
        // 25 bucket: 0-cost / high-tempo offense.
        assert_eq!(play_ordering(CardId::Backstab, None), 25);
        assert_eq!(play_ordering(CardId::PreciseCut, None), 25);
        assert_eq!(play_ordering(CardId::Suppress, None), 25);
        assert_eq!(play_ordering(CardId::Dash, None), 25);
        assert_eq!(play_ordering(CardId::GrandFinale, None), 25);
        // 30 bucket: big swing attacks.
        assert_eq!(play_ordering(CardId::Pinpoint, None), 30);
        assert_eq!(play_ordering(CardId::Sunder, None), 30);
        assert_eq!(play_ordering(CardId::Hyperbeam, None), 30);
        assert_eq!(play_ordering(CardId::MeteorStrike, None), 30);
        assert_eq!(play_ordering(CardId::Fisticuffs, None), 30);
        // 35 bucket: standard attacks.
        assert_eq!(play_ordering(CardId::DaggerThrow, None), 35);
        assert_eq!(play_ordering(CardId::SuckerPunch, None), 35);
        assert_eq!(play_ordering(CardId::Barrage, None), 35);
        assert_eq!(play_ordering(CardId::FocusedStrikeCard, None), 35);
        assert_eq!(play_ordering(CardId::Omnislice, None), 35);
        // 40 bucket: AOE / cheap utility attacks.
        assert_eq!(play_ordering(CardId::DaggerSpray, None), 40);
        assert_eq!(play_ordering(CardId::Neutralize, None), 40);
        assert_eq!(play_ordering(CardId::Claw, None), 40);
        assert_eq!(play_ordering(CardId::BeamCell, None), 40);
        assert_eq!(play_ordering(CardId::Salvo, None), 40);
    }

    /// Bead `sts2-rs-g8sy4` straggler pass: the remaining
    /// Silent / Colorless cards that were still falling through to
    /// [`NEUTRAL_PRIORITY`] after the first g8sy4 landing now bucket
    /// into their content-role bucket. Covers the 17 cards identified
    /// by diffing `card_ids_generated.rs` against `expert.rs` (Shiv +
    /// 16 colorless event/status-pool cards).
    #[test]
    fn play_ordering_g8sy4_silent_colorless_stragglers() {
        // -2 bucket: 0/1-cost self-Power scalers and BrightestFlame
        // (0-cost cycle/energy burst).
        assert_eq!(play_ordering(CardId::BrightestFlame, None), -2);
        assert_eq!(play_ordering(CardId::FeedingFrenzyCard, None), -2);
        assert_eq!(play_ordering(CardId::HelloWorldCard, None), -2);
        // 10 bucket: ToricToughness (2c modest standard block).
        assert_eq!(play_ordering(CardId::ToricToughness, None), 10);
        // 20 bucket: card-flow / generate-random / move-card skills.
        assert_eq!(play_ordering(CardId::Distraction, None), 20);
        assert_eq!(play_ordering(CardId::Metamorphosis, None), 20);
        assert_eq!(play_ordering(CardId::Outmaneuver, None), 20);
        assert_eq!(play_ordering(CardId::Wish, None), 20);
        // 22 bucket: big-block / scaling defense.
        assert_eq!(play_ordering(CardId::Relax, None), 22);
        assert_eq!(play_ordering(CardId::Stack, None), 22);
        // 25 bucket: exceptional 0-cost / huge-finisher offense.
        assert_eq!(play_ordering(CardId::Clash, None), 25);
        assert_eq!(play_ordering(CardId::Whistle, None), 25);
        // 30 bucket: mid attack + rider.
        assert_eq!(play_ordering(CardId::Rebound, None), 30);
        assert_eq!(play_ordering(CardId::Squash, None), 30);
        // 35 bucket: standard 1-cost attacks.
        assert_eq!(play_ordering(CardId::MadScience, None), 35);
        assert_eq!(play_ordering(CardId::Peck, None), 35);
        // 48 bucket: 0-cost filler attack.
        assert_eq!(play_ordering(CardId::Shiv, None), 48);
    }

    /// The fixture marker for `play_ordering_coverage_threshold_fence`.
    ///
    /// Bead `sts2-rs-g8sy4` used four unclassified Necrobinder cards
    /// (Defile, BlightStrike, GraveWarden, DrainPower) as its "truly
    /// uncovered pool", and pinned them NEUTRAL so the fence could not
    /// rot silently. GH #515 classified all four — every *playable* card
    /// now carries a priority, which `tests/play_ordering_coverage.rs`
    /// enforces — so a real card can no longer play that role, and the
    /// fence moved to [`CardId::Generic`].
    ///
    /// `Generic` is the sentinel variant: `spec_for_card_id` returns
    /// `None` for it, nothing can put it in a hand, and the census test
    /// exempts it for exactly that reason. It is therefore permanently
    /// NEUTRAL by construction rather than by omission — which is what a
    /// fixture marker should have been in the first place.
    #[test]
    fn play_ordering_coverage_fixture_marker_stays_neutral() {
        assert_eq!(
            play_ordering(CardId::Generic, None),
            NEUTRAL_PRIORITY,
            "CardId::Generic is the coverage-threshold fixture's marker — \
             it is the sentinel variant and must never be classified",
        );
        // The cards that used to hold this role are classified now. If one
        // of them reverts to NEUTRAL the census test fails too, but say so
        // here as well: this is where someone looking for the old fixture
        // lands.
        for cid in [
            CardId::Defile,
            CardId::BlightStrike,
            CardId::GraveWarden,
            CardId::DrainPower,
        ] {
            assert_ne!(
                play_ordering(cid, None),
                NEUTRAL_PRIORITY,
                "{cid:?} was classified by GH #515 — a revert to NEUTRAL puts \
                 Necrobinder hands back under the coverage gate",
            );
        }
    }

    /// Bead `sts2-rs-g8sy4` (re-close pass): pin a quantitative
    /// coverage floor on a fixed sample of the deck-loadable pool
    /// for the 4 ported characters (Ironclad, Silent, Defect,
    /// Colorless). Each entry below was independently bucketed by
    /// the per-character tests above; this test re-checks them as a
    /// single batch so a refactor that drops a bucket arm surfaces
    /// here rather than in a downstream A/B. Necrobinder + Regent
    /// are still in flight under bead `bay` and are intentionally
    /// excluded from this sample (their cards mostly fall through
    /// to [`NEUTRAL_PRIORITY`] — that's the coverage-threshold
    /// fixture's signal, locked in
    /// [`play_ordering_coverage_fixture_marker_stays_neutral`]). GH #515
    /// closed that gap: Necrobinder and Regent are fully classified now,
    /// and the sample below is no longer the only coverage floor —
    /// `tests/play_ordering_coverage.rs` checks every playable card.
    ///
    /// **Floor lock:** all 23 sampled deck-loadable cards across the
    /// 4 ported characters must classify into a non-NEUTRAL bucket.
    /// Failure mode: a trainer or refactor regression drops a bucket
    /// arm and one of these cards collapses to NEUTRAL, which would
    /// silently degrade the heuristic rollout to Uniform on that
    /// card's character pool.
    #[test]
    fn play_ordering_g8sy4_ported_character_coverage_floor() {
        const SAMPLE: &[CardId] = &[
            // Ironclad
            CardId::Bash,
            CardId::StrikeIronclad,
            CardId::DemonFormCard,
            CardId::InfernoCard,
            CardId::Inflame,
            CardId::Apotheosis,
            // Silent
            CardId::Backstab,
            CardId::StrikeSilent,
            CardId::WraithForm,
            CardId::FanOfKnivesCard,
            CardId::Acrobatics,
            CardId::Survivor,
            // Defect
            CardId::Zap,
            CardId::StrikeDefect,
            CardId::EchoFormCard,
            CardId::Hailstorm,
            CardId::ChargeBattery,
            CardId::Hyperbeam,
            // Colorless (event/ancient-pool stragglers from the
            // first g8sy4 landing — the cards this bead added)
            CardId::BrightestFlame,
            CardId::Clash,
            CardId::Whistle,
            CardId::Relax,
            CardId::Wish,
            // Note: Shiv (0-cost filler attack) classifies into bucket
            // 50, which equals NEUTRAL_PRIORITY by current constants;
            // the coverage-gate predicate would mark it as uncovered.
            // Same is true for the STS1 50-bucket filler attacks
            // (TwinStrike, Anger, BodySlam, ...). This sample excludes
            // 50-bucket cards by design — the floor-lock measures
            // semantically-covered cards, not numerically-distinct.
        ];
        let total = SAMPLE.len();
        let uncovered: Vec<_> = SAMPLE
            .iter()
            .filter(|cid| play_ordering(**cid, None) == NEUTRAL_PRIORITY)
            .collect();
        let covered = total - uncovered.len();
        assert_eq!(
            covered, total,
            "ported-character coverage floor breached: {covered}/{total} classified; \
             uncovered = {uncovered:?}"
        );
    }

    /// Regent rollout-coverage floor. Regent's 88-card deck pool (+5
    /// minion/blade tokens) was entirely `NEUTRAL_PRIORITY` except the
    /// two basics, so `rollout_coverage_fraction` on any Regent hand sat
    /// at ~2% << `COVERAGE_THRESHOLD` and the heuristic rollout degraded
    /// to Uniform for every Regent state (the root cause of the boss-fight
    /// under-block / wasted-energy tactical noise investigated 2026-07-20).
    /// This sample spans every bucket the coverage pass populates —
    /// powers, block skills, utility skills, free + costed attacks, and
    /// the minion/blade tokens — and must stay fully classified.
    #[test]
    fn play_ordering_regent_coverage_floor() {
        const SAMPLE: &[CardId] = &[
            // powers (cheap scaler -2 / combat-defining 0)
            CardId::BlackHole,
            CardId::TyrannyCard,
            CardId::VoidForm,
            CardId::Genesis,
            // block skills (22)
            CardId::ParticleWall,
            CardId::Bulwark,
            CardId::CloakOfStars,
            CardId::ReflectCard,
            // utility skills (20)
            CardId::KnowThyPlace,
            CardId::HiddenCache,
            CardId::Conqueror,
            CardId::Guards,
            // attacks (free 30 / costed 40)
            CardId::Comet,
            CardId::FallingStar,
            CardId::CrescentSpear,
            CardId::Devastate,
            // minion / blade tokens
            CardId::SovereignBlade,
            CardId::MinionStrike,
            CardId::MinionDiveBomb,
            CardId::MinionSacrifice,
            CardId::Exterminate,
        ];
        let total = SAMPLE.len();
        let uncovered: Vec<_> = SAMPLE
            .iter()
            .filter(|cid| play_ordering(**cid, Some(CharacterId::Regent)) == NEUTRAL_PRIORITY)
            .collect();
        let covered = total - uncovered.len();
        assert_eq!(
            covered, total,
            "Regent coverage floor breached: {covered}/{total} classified; \
             uncovered = {uncovered:?}"
        );
    }

    /// Necrobinder rollout-coverage floor. Same latent issue Regent had
    /// (8/88 classified → Uniform rollout fallback); Silent (99%) and
    /// Defect (100%) were already covered by the g8sy4 passes, so this
    /// pass targets only Necrobinder. Sample spans every populated bucket.
    #[test]
    fn play_ordering_necrobinder_coverage_floor() {
        const SAMPLE: &[CardId] = &[
            // powers
            CardId::ReaperForm,
            CardId::SpiritOfAsh,
            CardId::LethalityCard,
            CardId::Demesne,
            // block skills (22)
            CardId::DeathsDoor,
            CardId::Undeath,
            CardId::Defy,
            // utility skills (20)
            CardId::Reanimate,
            CardId::Seance,
            CardId::Sacrifice,
            CardId::Invoke,
            CardId::Soul,
            // attacks (free 30 / costed 40)
            CardId::Poke,
            CardId::Eradicate,
            CardId::Reap,
            CardId::TheScythe,
        ];
        let total = SAMPLE.len();
        let uncovered: Vec<_> = SAMPLE
            .iter()
            .filter(|cid| play_ordering(**cid, Some(CharacterId::Necrobinder)) == NEUTRAL_PRIORITY)
            .collect();
        let covered = total - uncovered.len();
        assert_eq!(
            covered, total,
            "Necrobinder coverage floor breached: {covered}/{total} classified; \
             uncovered = {uncovered:?}"
        );
    }

    /// Bead `sts2-rs-g8sy4` (re-close pass 3): pin the
    /// `rollout_coverage_fraction` empty-actions branch — when the
    /// legal-actions slice contains no `PlayCard` entries
    /// (EndTurn-only / DrinkPotion-only / Channel-only states), the
    /// fraction must return 1.0 so the heuristic does NOT trip the
    /// coverage-threshold fallback. This branch is non-obvious
    /// (priority-table coverage doesn't apply when there's no card
    /// to look up) and the existing coverage_fraction test only
    /// exercises the all-classified and all-uncovered extremes.
    /// Bead `sts2-rs-g8sy4` (pass 7): `play_ordering` is total over
    /// `(CardId, Option<CharacterId>)` — every variant pair returns
    /// an i16 in the [-20, 70] ladder range without panicking. Pins
    /// the public-API totality contract so a future per-character
    /// override-arm bead can't silently introduce a code path that
    /// panics or returns an out-of-range value on one specific
    /// (card, character) combo.
    #[test]
    fn play_ordering_g8sy4_total_over_character_argument_space() {
        let chars = [
            None,
            Some(CharacterId::Ironclad),
            Some(CharacterId::Silent),
            Some(CharacterId::Defect),
            Some(CharacterId::Necrobinder),
            Some(CharacterId::Regent),
        ];
        // Sample one card per major bucket-cluster: ladder floor,
        // mid, ceiling, and three g8sy4 stragglers across types.
        let sample = [
            CardId::Apotheosis,     // floor
            CardId::Bash,           // 25
            CardId::StrikeIronclad, // 55
            CardId::Slimed,         // 70 (ceiling)
            CardId::Clash,          // g8sy4 (25)
            CardId::Shiv,           // g8sy4 (50)
            CardId::HelloWorldCard, // g8sy4 (-2)
        ];
        for card in sample {
            for ch in chars {
                let p = play_ordering(card, ch);
                assert!(
                    (-20..=70).contains(&p),
                    "play_ordering({card:?}, {ch:?}) = {p} outside [-20, 70]"
                );
            }
        }
    }

    /// Bead `sts2-rs-g8sy4` (re-close pass 4): pin the heuristic
    /// constants used by the rollout-policy ladder. A trainer tweak
    /// (e.g. a sweep that overwrites `SOFTMAX_TAU` or
    /// `COVERAGE_THRESHOLD`) without updating the bead's empirical
    /// gates would silently change rollout behavior; this test
    /// surfaces that as a build failure. Step 3 of the bead body
    /// flagged `SOFTMAX_TAU` for a re-tune after coverage rises —
    /// the current value is the post-stragglers ladder constant.
    #[test]
    fn play_ordering_g8sy4_constants_pinned() {
        // NEUTRAL_PRIORITY is the midpoint of the [-20, 70] ladder
        // (see `play_ordering_g8sy4_priority_range_invariant`). The
        // 50-bucket filler-attack cards (TwinStrike, Shiv, ...)
        // numerically alias to this sentinel — coverage gate semantics
        // depend on the alias holding.
        assert_eq!(NEUTRAL_PRIORITY, 50);
        // COVERAGE_THRESHOLD is the rollout-fallback trip point
        // (per j0f6). Anything below pulls the rollout to Uniform.
        // 0.8 = pinned in the constant's docstring as "≥ 80% of the
        // legal-action set classifies into a non-neutral bucket".
        assert!((COVERAGE_THRESHOLD - 0.8).abs() < f32::EPSILON);
        // SOFTMAX_TAU sets the softmax sharpness. The bead body's
        // step 3 (deferred) is the formal re-tune; pin the current
        // value so the deferred work surfaces as a test diff rather
        // than silent drift.
        assert!((SOFTMAX_TAU - 32.0).abs() < f32::EPSILON);
        // DEBUFF_SETUP_PRIORITY_BONUS is the priority bump Bash and
        // similar setup attacks get when (a) target survives, (b)
        // target not already debuffed, (c) higher-damage follow-up
        // attack exists. A positive value pulls these forward.
        // Pin the exact current value so a future bead resetting it
        // to 0 (which would disable the setup-priority logic) surfaces
        // here rather than as a silent behavior change.
        assert_eq!(DEBUFF_SETUP_PRIORITY_BONUS, 2);
    }

    #[test]
    #[allow(clippy::float_cmp)] // exact return-value pin; function literally returns 1.0.
    fn coverage_fraction_returns_one_on_no_play_card_actions() {
        let state = sts2_core::CombatState::default();
        // Empty slice — no actions of any kind.
        assert_eq!(rollout_coverage_fraction(&state, &[]), 1.0);
        // EndTurn-only — non-PlayCard action present, still no
        // priority-table dependency.
        let only_end_turn = [Action::EndTurn];
        assert_eq!(rollout_coverage_fraction(&state, &only_end_turn), 1.0);
    }

    /// Bead `sts2-rs-g8sy4` (re-close pass 13): pin the exact
    /// membership of [`is_debuff_attaching_attack`] —
    /// `{Bash, Thunderclap, Uppercut}` — and a representative
    /// counter-example set (Strike, Defend, IronWave). The function
    /// gates the call to [`debuff_setup_priority_delta`] from
    /// [`action_priority`]; a regression that added (or removed) a
    /// card from this set would silently shift rollout aggression
    /// without surfacing in either constant-pin (pass 5) or sample-
    /// priority pins (passes 3 / 10).
    ///
    /// IronWave is explicitly NOT a debuff-attacher at the Python
    /// pin (`sts2-rl-agent@01c094d:sts2_env/cards/ironclad.rs::play_iron_wave`
    /// does damage + block, no debuff) — pinning that is_debuff_attaching_attack
    /// returns false for it keeps the Rust port aligned.
    #[test]
    fn g8sy4_is_debuff_attaching_attack_membership() {
        // The three (and only three) cards that should return true.
        // Bind to a local first so clippy doesn't fire
        // assertions-on-constants against the const-fn return.
        let bash_ok = is_debuff_attaching_attack(CardId::Bash);
        let tc_ok = is_debuff_attaching_attack(CardId::Thunderclap);
        let upper_ok = is_debuff_attaching_attack(CardId::Uppercut);
        assert!(bash_ok, "Bash must be classified as debuff-attaching");
        assert!(tc_ok, "Thunderclap must be classified as debuff-attaching");
        assert!(upper_ok, "Uppercut must be classified as debuff-attaching");

        // Counter-examples — Attack-class cards that do NOT attach
        // a debuff in their play handler, MUST return false.
        const NON_DEBUFF_ATTACKS: &[CardId] = &[
            CardId::StrikeIronclad,
            CardId::TwinStrike,
            CardId::Clash,
            CardId::IronWave, // damage + block, NO debuff per Python pin
            CardId::PommelStrike,
            CardId::PerfectedStrike,
        ];
        for cid in NON_DEBUFF_ATTACKS {
            assert!(
                !is_debuff_attaching_attack(*cid),
                "is_debuff_attaching_attack({cid:?}) should be false — \
                 not in the Bash/Thunderclap/Uppercut set",
            );
        }

        // And cards in other CardType categories should never be in
        // this set either.
        const NON_ATTACK_CARDS: &[CardId] = &[
            CardId::DefendIronclad,
            CardId::Anger,
            CardId::Apotheosis,
            CardId::Slimed,
        ];
        for cid in NON_ATTACK_CARDS {
            assert!(
                !is_debuff_attaching_attack(*cid),
                "is_debuff_attaching_attack({cid:?}) should be false — non-attack card",
            );
        }
    }

    /// Bead `sts2-rs-g8sy4` (re-close pass 12): pin that the SECOND
    /// `continue` branch of [`rollout_coverage_fraction`] (when the
    /// hand entry's `card_uid` indexes outside `state.cards`) also
    /// drops the action from the `covered` counter while still
    /// incrementing `play_card_count`. Pass 11 only exercised the
    /// FIRST `continue` (out-of-range hand_index on empty hand).
    /// Together they pin both miss-paths against a regression that
    /// would short-circuit one but not the other.
    #[test]
    #[allow(clippy::float_cmp)]
    fn coverage_fraction_zero_when_hand_entries_point_outside_cards_vec() {
        use smallvec::smallvec;
        use sts2_core::CardUid;
        let mut state = sts2_core::CombatState::default();
        // Hand contains 3 entries pointing into a vacant `state.cards`
        // (uids 1, 2, 3 → indices 0, 1, 2 — all out of range for an
        // empty `state.cards`). The first `let Some(card_uid) =
        // hand.get(idx)` succeeds; the second `let Some(card) =
        // state.cards.get(idx-1)` returns None → `continue`.
        state.player.hand = smallvec![
            CardUid::new(1).unwrap(),
            CardUid::new(2).unwrap(),
            CardUid::new(3).unwrap(),
        ];
        // `state.cards` is empty — every uid lookup misses.
        assert!(state.cards.is_empty());

        let actions: Vec<Action> = (0..3)
            .map(|i| Action::PlayCard {
                hand_index: i,
                target: None,
            })
            .collect();
        assert_eq!(rollout_coverage_fraction(&state, &actions), 0.0);
    }

    /// Bead `sts2-rs-g8sy4` (re-close pass 11): pin the
    /// [0.0, 1.0] output range of [`rollout_coverage_fraction`].
    /// With an empty player hand, every PlayCard's `hand_index`
    /// lookup misses (the `state.player.hand.get(idx)` returns
    /// `None` → `continue`), incrementing the denominator but not
    /// the numerator → frac = 0.0 / N → 0.0. Confirms the lower
    /// bound and that EndTurn actions are not counted in the
    /// denominator (only PlayCard increments).
    #[test]
    #[allow(clippy::float_cmp)]
    fn coverage_fraction_zero_when_all_play_cards_miss_hand() {
        let state = sts2_core::CombatState::default();
        // 5 PlayCards with hand_index ∈ {0..5}, all pointing into an
        // empty hand. Each one bumps play_card_count but
        // contributes 0 to covered.
        let actions: Vec<Action> = (0..5)
            .map(|i| Action::PlayCard {
                hand_index: i,
                target: None,
            })
            .collect();
        assert_eq!(rollout_coverage_fraction(&state, &actions), 0.0);

        // Mixing EndTurn into the same action list does NOT change
        // the denominator (EndTurn isn't a PlayCard) — frac stays 0.0.
        let mut mixed = actions.clone();
        mixed.push(Action::EndTurn);
        mixed.push(Action::EndTurn);
        assert_eq!(rollout_coverage_fraction(&state, &mixed), 0.0);
    }

    /// Bead `sts2-rs-g8sy4` (re-close pass 2): pin the priority range
    /// invariant. Every classified card's `play_ordering` value must
    /// fall within [-20, 70] — the empirical range covered by the
    /// existing buckets ladder (Apotheosis/Apparition at -20 up
    /// through hard curses at 70). A typo or mis-bucket producing
    /// e.g. `255` or `-99` would still pass the per-character bucket
    /// tests but blow up softmax weighting; this guard catches it.
    /// The check runs over the same SAMPLE used by
    /// `play_ordering_g8sy4_ported_character_coverage_floor` (plus
    /// the 50-bucket boundary cards) so coverage stays in lockstep
    /// with that pin.
    #[test]
    fn play_ordering_g8sy4_priority_range_invariant() {
        const SAMPLE: &[CardId] = &[
            CardId::Apotheosis,     // floor (-20)
            CardId::Bash,           // 25
            CardId::StrikeIronclad, // 55
            CardId::Slimed,         // 70 (ceiling)
            CardId::TwinStrike,     // 50 (boundary, equal to NEUTRAL)
            CardId::Shiv,           // 50 (g8sy4 straggler)
            CardId::BrightestFlame, // -2 (g8sy4 straggler)
            CardId::Whistle,        // 25 (g8sy4 straggler)
        ];
        for cid in SAMPLE {
            let p = play_ordering(*cid, None);
            assert!(
                (-20..=70).contains(&p),
                "{cid:?} = {p} outside the [-20, 70] priority ladder; \
                 a bucket typo or out-of-range arm would break softmax weighting"
            );
        }
    }

    #[test]
    fn end_turn_is_penalised() {
        use rand::SeedableRng;
        let state = sts2_core::CombatState::default();
        // pick_action_heuristic is search-side (DET, not parity-critical
        // per the RNG audit); fixture seed via Xoshiro is fine.
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(0);
        let acts = vec![Action::EndTurn];
        // With only EndTurn legal we still must pick it (no crash).
        let idx = pick_action_heuristic(&mut rng, &state, &acts);
        assert_eq!(idx, 0);
    }

    #[test]
    fn heuristic_prefers_lower_priority_card() {
        // Crafted state with two PlayCard actions: one Apotheosis (-20),
        // one Strike (55). Apotheosis should dominate the softmax sample
        // across many seeds.
        // We can't easily build a full CombatState here without the
        // common test fixtures; instead exercise the priority sort
        // directly via action_priority on a near-default state.
        // The Apotheosis card_uid path is exercised in the integration
        // tests under `crates/sts2-mcts/tests/`.
        let p_lo = play_ordering(CardId::Apotheosis, None);
        let p_hi = play_ordering(CardId::StrikeIronclad, None);
        assert!(p_lo < p_hi, "Apotheosis must outrank Strike");
    }

    /// Bead `sts2-rs-5dw7`: pure unit test on the `HpAwareCtx` →
    /// priority-override table. Avoids needing a full `CombatState`
    /// fixture (the integration smoke vs Entomancer covers that path).
    #[test]
    fn hp_aware_bumps_defend_and_demotes_strike_when_lethal() {
        // Lethal turn → Defend↑/Strike↓.
        let lethal = HpAwareCtx {
            is_lethal_turn: true,
            is_sustained_pressure: false,
            lowest_hp_enemy_target: None,
            defend_threat_bonus: 0,
        };
        let nonlethal = HpAwareCtx {
            is_lethal_turn: false,
            is_sustained_pressure: false,
            lowest_hp_enemy_target: None,
            defend_threat_bonus: 0,
        };

        // Quick sanity over each Defend/Strike pair the override
        // recognizes. Build a `PlayCard` into hand_index 0 by leaning
        // on the helper predicates directly — `action_priority`'s
        // `state.player.hand`/`state.cards` lookups would need a full
        // CombatState fixture, which the integration smoke covers.
        for did in [
            CardId::DefendIronclad,
            CardId::DefendSilent,
            CardId::DefendDefect,
            CardId::DefendNecrobinder,
            CardId::DefendRegent,
            CardId::IronWave,
            CardId::Entrench,
        ] {
            assert!(is_defend_class(did), "{did:?} should be defend-class");
        }
        for sid in [
            CardId::StrikeIronclad,
            CardId::StrikeSilent,
            CardId::StrikeDefect,
            CardId::StrikeNecrobinder,
            CardId::StrikeRegent,
        ] {
            assert!(is_strike_basic(sid), "{sid:?} should be strike-basic");
        }

        // Cards outside the override stay on play_ordering even when
        // lethal — the heuristic should still consider Bash, scaling
        // powers, etc.
        assert!(!is_defend_class(CardId::Bash));
        assert!(!is_strike_basic(CardId::Bash));
        assert!(!is_defend_class(CardId::Apotheosis));

        // ctx field discipline: `is_lethal_turn` must be the only knob
        // gating the override (future-proofing against the addition of
        // extra fields).
        assert!(lethal.is_lethal_turn);
        assert!(!nonlethal.is_lethal_turn);
    }

    /// Bead `sts2-rs-d9qd` (zlhm.1): sustained-pressure predicate
    /// constants + mutual-exclusion contract. The full
    /// predicate-from-state path is covered by the integration smoke
    /// + the 200-Silent baseline A/B; this test pins the constants
    /// and the priority-bump-sharing invariant.
    #[test]
    fn sustained_pressure_shares_lethal_bump_constants() {
        // Horizon must be >=2 — at horizon=1 the predicate collapses
        // into `is_lethal_turn`. Picked 2 as the smallest non-trivial
        // multi-turn window.
        const _: () = assert!(SUSTAINED_PRESSURE_HORIZON >= 2);
        // The sustained-pressure arm reuses the 5dw7 priority constants
        // for the rollout-side Defend/Strike bump. If they ever diverge,
        // `action_priority` must branch — fail fast here.
        assert_eq!(LETHAL_DEFEND_PRIORITY, -5);
        assert_eq!(LETHAL_STRIKE_PRIORITY, 90);
        // Mutual exclusion: HpAwareCtx::compute enforces
        // `is_sustained_pressure = !is_lethal_turn && ...` so the two
        // flags are never both set. The struct field shape is
        // future-proofed against extra fields.
        let sustained = HpAwareCtx {
            is_lethal_turn: false,
            is_sustained_pressure: true,
            lowest_hp_enemy_target: None,
            defend_threat_bonus: 0,
        };
        assert!(sustained.is_sustained_pressure && !sustained.is_lethal_turn);
    }

    /// Bead `sts2-rs-6g96`: focus-fire bonus const + ctx-field shape.
    /// The PlayCard-target branch of `action_priority` needs a full
    /// CombatState fixture to exercise, which the integration smoke
    /// covers; this test pins the const + None-target ctx contract.
    #[test]
    fn focus_fire_bonus_within_sensible_range() {
        // Bonus must be strictly positive (it's a *subtraction* from
        // play_ordering — biases the targeted action earlier).
        const _: () = assert!(FOCUS_FIRE_PRIORITY_BONUS > 0);
        // And small enough that a Strike (55) targeting the kill enemy
        // doesn't undercut Bash (25) targeting the same enemy.
        assert!(
            play_ordering(CardId::StrikeIronclad, None) - FOCUS_FIRE_PRIORITY_BONUS
                > play_ordering(CardId::Bash, None),
            "Strike-on-focus ({}) must still yield to Bash ({})",
            play_ordering(CardId::StrikeIronclad, None) - FOCUS_FIRE_PRIORITY_BONUS,
            play_ordering(CardId::Bash, None),
        );

        // None-target ctx: focus-fire is disabled (single-enemy fights).
        let ctx = HpAwareCtx {
            is_lethal_turn: false,
            is_sustained_pressure: false,
            lowest_hp_enemy_target: None,
            defend_threat_bonus: 0,
        };
        assert!(ctx.lowest_hp_enemy_target.is_none());

        // Some-target ctx: focus-fire arms on slot 2.
        let focused = HpAwareCtx {
            is_lethal_turn: false,
            is_sustained_pressure: false,
            lowest_hp_enemy_target: Some(2),
            defend_threat_bonus: 0,
        };
        assert_eq!(focused.lowest_hp_enemy_target, Some(2));
    }

    /// Bead `sts2-rs-j0f6`: coverage-fraction helper triggers the
    /// fallback to Uniform on random Silent/Defect/Necrobinder/Regent
    /// decks where most cards fall through to `NEUTRAL_PRIORITY`.
    /// Builds two synthetic 4-card hands directly on a `CombatState` —
    /// one of all-classified Ironclad cards (coverage = 1.0, should
    /// NOT trip the threshold), one of all-uncovered Necrobinder
    /// cards (coverage = 0.0, must trip it).
    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn coverage_fraction_triggers_fallback_on_random_unclassified_decks() {
        use smallvec::smallvec;
        use sts2_core::cards::{Card, CardTags, CardType, TargetType};
        use sts2_core::{CardUid, CombatState};

        fn build_card(uid: u32, card_id: CardId) -> Card {
            Card {
                uid: CardUid::new(uid).expect("non-zero"),
                card_id,
                cost: 1,
                card_type: CardType::Skill,
                target_type: TargetType::Self_,
                upgraded: false,
                base_damage: 0,
                base_block: 0,
                combat_vars: sts2_core::cards::CombatVars::new(),
                enchantments: sts2_core::cards::CardEnchantments::new(),
                tags: CardTags::NONE,
                energy_spent: 0,
                base_replay_count: 0,
                keyword_override: None,
                affliction: None,
            }
        }

        let play_actions: Vec<Action> = (0..4u8)
            .map(|i| Action::PlayCard {
                hand_index: i,
                target: None,
            })
            .collect();

        // Hand A — fully classified Ironclad cards. Every entry has a
        // non-neutral `play_ordering` bucket; coverage must be 1.0.
        let mut state_hi = CombatState::default();
        state_hi.cards = vec![
            build_card(1, CardId::Bash),           // 25
            build_card(2, CardId::Apotheosis),     // -20
            build_card(3, CardId::DefendIronclad), // 55
            build_card(4, CardId::Inflame),        // 10
        ]
        .into();
        state_hi.player.hand = smallvec![
            CardUid::new(1).unwrap(),
            CardUid::new(2).unwrap(),
            CardUid::new(3).unwrap(),
            CardUid::new(4).unwrap(),
        ];
        let frac_hi = rollout_coverage_fraction(&state_hi, &play_actions);
        assert!(
            frac_hi >= COVERAGE_THRESHOLD,
            "all-classified Ironclad hand should clear COVERAGE_THRESHOLD: got {frac_hi}",
        );

        // Hand B — the sentinel `CardId::Generic`, which is NEUTRAL by
        // construction: it has no `CardSpec`, so nothing can deal it, and
        // `tests/play_ordering_coverage.rs` exempts it from the census for
        // that reason. Coverage must be 0.0, well under the threshold,
        // triggering the Uniform fallback.
        //
        // This hand used to be four unclassified Necrobinder cards. GH #515
        // classified every playable card, so a real card can no longer stand
        // in for an uncovered pool — the fence would have quietly stopped
        // testing anything.
        let mut state_lo = CombatState::default();
        state_lo.cards = vec![
            build_card(1, CardId::Generic),
            build_card(2, CardId::Generic),
            build_card(3, CardId::Generic),
            build_card(4, CardId::Generic),
        ]
        .into();
        state_lo.player.hand = smallvec![
            CardUid::new(1).unwrap(),
            CardUid::new(2).unwrap(),
            CardUid::new(3).unwrap(),
            CardUid::new(4).unwrap(),
        ];
        let frac_lo = rollout_coverage_fraction(&state_lo, &play_actions);
        assert!(
            frac_lo < COVERAGE_THRESHOLD,
            "all-unclassified Necrobinder hand must trip the fallback: got {frac_lo}",
        );

        // Sanity: the marker maps to NEUTRAL on its own, so hand B's 0.0
        // is the fallback firing and not an accident of the hand shape.
        assert_eq!(
            play_ordering(CardId::Generic, None),
            NEUTRAL_PRIORITY,
            "CardId::Generic is the sentinel variant and must never be \
             classified — it is what makes this fence testable now that \
             every playable card carries a priority (GH #515)",
        );

        // No-PlayCard actions list (e.g. only EndTurn legal) yields the
        // "no priority dependency" sentinel — coverage = 1.0 so the
        // heuristic is safe to run.
        let only_end_turn = vec![Action::EndTurn];
        assert!(
            (rollout_coverage_fraction(&state_hi, &only_end_turn) - 1.0).abs() < f32::EPSILON,
            "empty-PlayCard action list should report coverage = 1.0",
        );
    }

    /// Bead `sts2-rs-j9GGr` (GitHub issue #9): kill-detection picks the
    /// lethal Strike over a higher-priority Bash when the enemy is at
    /// 2 HP. Without the override the softmax would heavily favor Bash
    /// (priority 25) over Strike (priority 55) and skip the free kill.
    fn make_attack_card(
        uid: u32,
        card_id: CardId,
        base_damage: i32,
        cost: i8,
    ) -> sts2_core::cards::Card {
        use sts2_core::cards::{Card, CardTags, CardType, TargetType};
        use sts2_core::CardUid;
        Card {
            uid: CardUid::new(uid).unwrap(),
            card_id,
            cost,
            card_type: CardType::Attack,
            target_type: TargetType::Enemy,
            upgraded: false,
            base_damage,
            base_block: 0,
            combat_vars: sts2_core::cards::CombatVars::new(),
            enchantments: sts2_core::cards::CardEnchantments::new(),
            tags: CardTags::NONE,
            energy_spent: 0,
            base_replay_count: 0,
            keyword_override: None,
            affliction: None,
        }
    }

    /// GH #293 — block-bearing twin of [`make_attack_card`].
    fn make_block_card(
        uid: u32,
        card_id: CardId,
        base_block: i32,
        cost: i8,
    ) -> sts2_core::cards::Card {
        use sts2_core::cards::{Card, CardTags, CardType, TargetType};
        use sts2_core::CardUid;
        Card {
            uid: CardUid::new(uid).unwrap(),
            card_id,
            cost,
            card_type: CardType::Skill,
            target_type: TargetType::Self_,
            upgraded: false,
            base_damage: 0,
            base_block,
            combat_vars: sts2_core::cards::CombatVars::new(),
            enchantments: sts2_core::cards::CardEnchantments::new(),
            tags: CardTags::NONE,
            energy_spent: 0,
            base_replay_count: 0,
            keyword_override: None,
            affliction: None,
        }
    }

    /// GH #293 — a ctx with only the graded knob set.
    fn calm_ctx(defend_threat_bonus: i16) -> HpAwareCtx {
        HpAwareCtx {
            is_lethal_turn: false,
            is_sustained_pressure: false,
            lowest_hp_enemy_target: None,
            defend_threat_bonus,
        }
    }

    /// GH #293: the regime this fix targets. When neither death-avoidance
    /// arm fires, a basic Defend and a basic Strike both scored
    /// `play_ordering` 55 — the rollout was *exactly* indifferent between
    /// blocking a telegraphed hit and swinging, on 86-98% of the states
    /// facing an attack. The graded bonus has to break that tie in
    /// proportion to the threat, and leave a quiet turn alone.
    #[test]
    fn defend_gains_a_gradient_where_the_rollout_used_to_be_indifferent() {
        use sts2_core::CombatState;
        let state = CombatState::default();
        let defend = make_block_card(1, CardId::DefendIronclad, 5, 1);
        let strike = make_attack_card(2, CardId::StrikeIronclad, 6, 1);
        let base_d = play_ordering(defend.card_id, None);
        let base_s = play_ordering(strike.card_id, None);
        assert_eq!(
            base_d, base_s,
            "the pre-#293 tie this fix exists to break: Defend and Strike are both 55",
        );

        // Nothing telegraphed -> untouched, so quiet turns keep old behaviour.
        assert_eq!(
            play_card_priority(&state, &defend, None, base_d, &calm_ctx(0)),
            base_d,
            "a zero bonus must not perturb the priority at all",
        );

        // Under threat the tie breaks toward Defend, and harder as it ramps.
        let mild = play_card_priority(&state, &defend, None, base_d, &calm_ctx(10));
        let heavy = play_card_priority(&state, &defend, None, base_d, &calm_ctx(40));
        assert_eq!(mild, base_d - 10);
        assert!(
            heavy < mild,
            "a bigger threat must rank Defend strictly sooner ({heavy} vs {mild})",
        );
        assert!(
            heavy < play_card_priority(&state, &strike, None, base_s, &calm_ctx(40)),
            "under threat Defend must outrank Strike, which it never did before",
        );
    }

    /// GH #293: the ramp ships OFF. The A/B was null (see
    /// [`DEFEND_THREAT_MAX_BONUS_DEFAULT`]), so with the env var unset
    /// `compute` must produce a zero bonus even on a turn that would have
    /// ramped hard — deployed rollout behaviour stays byte-identical to
    /// pre-#293. Flipping the default without a fresh, powered A/B should
    /// break this test.
    #[test]
    fn the_defend_gradient_is_off_by_default() {
        use smallvec::smallvec;
        use std::num::NonZeroU32;
        use sts2_core::{CombatState, CreatureUid, Intent};

        assert_eq!(DEFEND_THREAT_MAX_BONUS_DEFAULT, 0);
        assert_eq!(defend_threat_max_bonus(), 0, "env var must be unset here");

        // 24 telegraphed into 50 effective HP — the worked example that
        // yields 28 when the ramp is enabled.
        let mut state = CombatState::default();
        state.player.current_hp = 50;
        state.player.max_hp = 80;
        let mut enemy = make_enemy(100, 200);
        enemy.intent = Some(Intent::Attack { damage: 24 });
        state.creatures.push(enemy);
        state.enemy_uids = smallvec![CreatureUid(NonZeroU32::new(100).unwrap())];

        let ctx = HpAwareCtx::compute(&state, IncomingDamage::Base);
        assert_eq!(ctx.defend_threat_bonus, 0);

        let defend = make_block_card(1, CardId::DefendIronclad, 5, 1);
        let base = play_ordering(defend.card_id, None);
        assert_eq!(
            play_card_priority(&state, &defend, None, base, &ctx),
            base,
            "with the ramp off a Defend must score its untouched base priority",
        );
    }

    /// GH #293: the ramp is sized to meet the cliff it extends. A base-55
    /// card at full threat must land exactly on [`LETHAL_DEFEND_PRIORITY`],
    /// so there is no discontinuity as a turn crosses into lethal.
    #[test]
    fn the_defend_gradient_is_continuous_with_the_lethal_cliff() {
        use sts2_core::CombatState;
        let state = CombatState::default();
        let defend = make_block_card(1, CardId::DefendIronclad, 5, 1);
        let base = play_ordering(defend.card_id, None);
        assert_eq!(
            play_card_priority(
                &state,
                &defend,
                None,
                base,
                &calm_ctx(DEFEND_THREAT_MAX_BONUS_CONTINUOUS)
            ),
            LETHAL_DEFEND_PRIORITY,
            "full threat must reach the same priority the lethal arm returns",
        );
    }

    /// GH #293: the graded arm keys on `base_block`, not on card class, so
    /// it must not touch cards that grant no block — and must not disturb
    /// the tuned emergency arms, which return before it.
    #[test]
    fn the_defend_gradient_leaves_attacks_and_the_lethal_arm_alone() {
        use sts2_core::CombatState;
        let state = CombatState::default();
        let strike = make_attack_card(1, CardId::StrikeIronclad, 6, 1);
        let base_s = play_ordering(strike.card_id, None);
        assert_eq!(
            play_card_priority(&state, &strike, None, base_s, &calm_ctx(50)),
            base_s,
            "a card with base_block 0 must be untouched however large the bonus",
        );

        // Regression guard: with an emergency arm live, the tuned constants
        // must win outright regardless of the graded knob.
        let lethal = HpAwareCtx {
            is_lethal_turn: true,
            is_sustained_pressure: false,
            lowest_hp_enemy_target: None,
            defend_threat_bonus: 17,
        };
        let defend = make_block_card(2, CardId::DefendIronclad, 5, 1);
        assert_eq!(
            play_card_priority(
                &state,
                &defend,
                None,
                play_ordering(defend.card_id, None),
                &lethal
            ),
            LETHAL_DEFEND_PRIORITY,
        );
        assert_eq!(
            play_card_priority(&state, &strike, None, base_s, &lethal),
            LETHAL_STRIKE_PRIORITY,
        );
    }

    /// GH #293: the bonus is computed from state, and `effective_hp`
    /// includes standing block — so the ramp SELF-LIMITS. Blocking against
    /// the same telegraphed hit is worth progressively less as block goes
    /// up, which is what stops the rollout over-correcting into a turtle.
    #[test]
    fn the_defend_gradient_self_limits_as_block_accumulates() {
        use smallvec::smallvec;
        use std::num::NonZeroU32;
        use sts2_core::{CombatState, CreatureUid, Intent};

        // The ramp ships OFF (see `DEFEND_THREAT_MAX_BONUS_DEFAULT`), so this
        // shape test has to turn it on. Safe because nextest runs one process
        // per test — the `OnceLock` behind `defend_threat_max_bonus()` is
        // populated by this test alone.
        std::env::set_var(
            "STS2_RUNAI_DEFEND_GRADIENT",
            DEFEND_THREAT_MAX_BONUS_CONTINUOUS.to_string(),
        );
        assert_eq!(
            defend_threat_max_bonus(),
            DEFEND_THREAT_MAX_BONUS_CONTINUOUS
        );

        let bonus_at = |block: i32| -> i16 {
            let mut state = CombatState::default();
            state.player.current_hp = 50;
            state.player.max_hp = 80;
            state.player.block = block;
            let mut enemy = make_enemy(100, 200);
            enemy.intent = Some(Intent::Attack { damage: 24 });
            state.creatures.push(enemy);
            state.enemy_uids = smallvec![CreatureUid(NonZeroU32::new(100).unwrap())];
            HpAwareCtx::compute(&state, IncomingDamage::Base).defend_threat_bonus
        };

        let naked = bonus_at(0);
        let shielded = bonus_at(30);
        // 24 incoming into 50 effective HP -> 24*60/50 = 28.
        assert_eq!(
            naked, 28,
            "the worked example from the constant's doc-comment"
        );
        assert!(
            shielded < naked,
            "standing block must shrink the urgency ({shielded} vs {naked})",
        );
        assert!(
            naked < DEFEND_THREAT_MAX_BONUS_CONTINUOUS,
            "a survivable hit must stay strictly below the lethal-cliff ceiling",
        );

        // No telegraph at all -> no bonus, so the ramp cannot fire on a
        // turn where the enemy is buffing or asleep.
        let mut quiet = CombatState::default();
        quiet.player.current_hp = 50;
        quiet.player.max_hp = 80;
        quiet.creatures.push(make_enemy(100, 200));
        quiet.enemy_uids = smallvec![CreatureUid(NonZeroU32::new(100).unwrap())];
        assert_eq!(
            HpAwareCtx::compute(&quiet, IncomingDamage::Base).defend_threat_bonus,
            0
        );
    }

    fn make_enemy(uid_value: u32, hp: i32) -> sts2_core::Creature {
        use std::num::NonZeroU32;
        use sts2_core::{Creature, IntentHistory, MonsterId};
        Creature {
            id: NonZeroU32::new(uid_value).unwrap(),
            current_hp: hp,
            max_hp: hp.max(1),
            block: 0,
            powers: smallvec::SmallVec::new(),
            species: None,
            monster_id: MonsterId("Cultist"),
            intent: None,
            intent_history: IntentHistory::default(),
            ai: None,
        }
    }

    #[test]
    fn obvious_kill_picks_lethal_strike_over_bash() {
        use smallvec::smallvec;
        use std::num::NonZeroU32;
        use sts2_core::{CardUid, CombatState, CreatureUid};

        let mut state = CombatState::default();
        state.player.energy = 3;
        state.cards = vec![
            make_attack_card(1, CardId::StrikeIronclad, 6, 1),
            make_attack_card(2, CardId::Bash, 8, 2),
        ]
        .into();
        state.player.hand = smallvec![CardUid::new(1).unwrap(), CardUid::new(2).unwrap()];

        let enemy = make_enemy(100, 2);
        let enemy_uid = CreatureUid(NonZeroU32::new(100).unwrap());
        state.creatures.push(enemy);
        state.enemy_uids = smallvec![enemy_uid];

        let actions = vec![
            Action::PlayCard {
                hand_index: 0,
                target: Some(0),
            },
            Action::PlayCard {
                hand_index: 1,
                target: Some(0),
            },
        ];

        assert_eq!(
            obvious_kill_action(&state, &actions),
            Some(0),
            "kill detection should fire on the first lethal attack",
        );

        // Enemy at 12 HP — neither Strike (6) nor Bash (8) lethals → no override.
        let mut state2 = state.clone();
        state2.creatures[0].current_hp = 12;
        assert_eq!(obvious_kill_action(&state2, &actions), None);
    }

    /// Bead `sts2-rs-j9GGr`: Vulnerable on target promotes Strike-for-6
    /// to a kill on a 9-HP enemy (6 × 1.5 = 9).
    #[test]
    fn obvious_kill_accounts_for_vulnerable() {
        use smallvec::smallvec;
        use std::num::NonZeroU32;
        use sts2_core::{CardUid, CombatState, CreatureUid, Power, PowerKind, PowerUid};

        let mut state = CombatState::default();
        state.player.energy = 3;
        state.cards = vec![make_attack_card(1, CardId::StrikeIronclad, 6, 1)].into();
        state.player.hand = smallvec![CardUid::new(1).unwrap()];

        let mut enemy = make_enemy(100, 9);
        let power_uid = PowerUid(NonZeroU32::new(200).unwrap());
        enemy.powers.push(power_uid);
        state.creatures.push(enemy);
        let enemy_uid = CreatureUid(NonZeroU32::new(100).unwrap());
        state.enemy_uids = smallvec![enemy_uid];
        state.powers.push(Power {
            id: power_uid.0,
            kind: PowerKind::Vulnerable,
            amount: 2,
            counter: 0,
            applier: None,
            skip_next_tick: false,
        });

        let actions = vec![Action::PlayCard {
            hand_index: 0,
            target: Some(0),
        }];
        assert_eq!(obvious_kill_action(&state, &actions), Some(0));

        // Without Vulnerable, 6 < 9, no kill.
        let mut state2 = state.clone();
        state2.creatures[0].powers.clear();
        assert_eq!(obvious_kill_action(&state2, &actions), None);
    }

    /// Block absorbs damage before HP, so an attack is only an "obvious
    /// kill" when it beats `current_hp + block`. A Strike-for-6 into a
    /// 4-HP enemy carrying 3 block deals through only 6 vs the 7-point
    /// wall → not a kill. Drop the block to 1 (wall 5) and it kills.
    #[test]
    fn obvious_kill_accounts_for_enemy_block() {
        use smallvec::smallvec;
        use std::num::NonZeroU32;
        use sts2_core::{CardUid, CombatState, CreatureUid};

        let mut state = CombatState::default();
        state.player.energy = 3;
        state.cards = vec![make_attack_card(1, CardId::StrikeIronclad, 6, 1)].into();
        state.player.hand = smallvec![CardUid::new(1).unwrap()];

        let mut enemy = make_enemy(100, 4);
        enemy.block = 3; // 4 HP + 3 block = 7 > 6 damage → survives.
        state.creatures.push(enemy);
        state.enemy_uids = smallvec![CreatureUid(NonZeroU32::new(100).unwrap())];

        let actions = vec![Action::PlayCard {
            hand_index: 0,
            target: Some(0),
        }];
        assert_eq!(
            obvious_kill_action(&state, &actions),
            None,
            "block must be added to the kill threshold",
        );

        let mut state2 = state.clone();
        state2.creatures[0].block = 1; // 4 + 1 = 5 <= 6 → kill.
        assert_eq!(obvious_kill_action(&state2, &actions), Some(0));
    }

    /// Player Weak floors outgoing damage to ×3/4: Strike-for-8 under
    /// Weak deals only `floor(8 * 0.75) = 6`, so an 8-HP enemy survives.
    /// Without Weak the same Strike is lethal.
    #[test]
    fn obvious_kill_accounts_for_player_weak() {
        use smallvec::smallvec;
        use std::num::NonZeroU32;
        use sts2_core::{CardUid, CombatState, CreatureUid, Power, PowerKind, PowerUid};

        let mut state = CombatState::default();
        state.player.energy = 3;
        // Give the player a mirror creature so power-folding sees Weak.
        let player_uid = CreatureUid(NonZeroU32::new(1).unwrap());
        let mut player_creature = make_enemy(1, 50);
        let weak_uid = PowerUid(NonZeroU32::new(300).unwrap());
        player_creature.powers.push(weak_uid);
        state.creatures.push(player_creature);
        state.player.creature_uid = Some(player_uid);
        state.powers.push(Power {
            id: weak_uid.0,
            kind: PowerKind::Weak,
            amount: 1,
            counter: 0,
            applier: None,
            skip_next_tick: false,
        });

        state.cards = vec![make_attack_card(1, CardId::StrikeIronclad, 8, 1)].into();
        state.player.hand = smallvec![CardUid::new(1).unwrap()];

        let enemy = make_enemy(100, 8);
        state.creatures.push(enemy);
        state.enemy_uids = smallvec![CreatureUid(NonZeroU32::new(100).unwrap())];

        let actions = vec![Action::PlayCard {
            hand_index: 0,
            target: Some(0),
        }];
        assert_eq!(
            obvious_kill_action(&state, &actions),
            None,
            "Weak must floor projected damage below the 8-HP target",
        );

        // Clear Weak: 8 >= 8 → kill.
        let mut state2 = state.clone();
        state2.creatures[0].powers.clear();
        assert_eq!(obvious_kill_action(&state2, &actions), Some(0));
    }

    /// Bead `sts2-rs-pifyc` Part 2: a debuff-attaching attack (Bash)
    /// gets `DEBUFF_SETUP_PRIORITY_BONUS` shaved off its priority
    /// when targeting an enemy that survives the hit AND is not
    /// already Vulnerable AND there is a follow-up attack of >= damage
    /// in hand. The bonus stacks with the focus-fire bonus (both
    /// subtract from `base`).
    #[test]
    fn debuff_setup_bonus_fires_for_bash_with_followup_strike() {
        use smallvec::smallvec;
        use std::num::NonZeroU32;
        use sts2_core::{CardUid, CombatState, CreatureUid};

        let mut state = CombatState::default();
        state.player.energy = 3;
        state.cards = vec![
            make_attack_card(1, CardId::Bash, 8, 2),
            make_attack_card(2, CardId::StrikeIronclad, 12, 1),
        ]
        .into();
        state.player.hand = smallvec![CardUid::new(1).unwrap(), CardUid::new(2).unwrap()];

        // Healthy enemy (50 HP) — Bash (8) does NOT 1-shot, follow-up
        // Strike (12) deals more damage than Bash, no existing
        // Vulnerable → bonus should fire.
        let enemy = make_enemy(100, 50);
        let enemy_uid = CreatureUid(NonZeroU32::new(100).unwrap());
        state.creatures.push(enemy);
        state.enemy_uids = smallvec![enemy_uid];

        let bash_card = state.cards[0];
        let delta = debuff_setup_priority_delta(&state, &bash_card, Some(0));
        assert_eq!(delta, DEBUFF_SETUP_PRIORITY_BONUS);

        // 1-shot range — bonus suppressed because the target dies on
        // this hit; the Vulnerable application is wasted.
        let mut state_1shot = state.clone();
        state_1shot.creatures[0].current_hp = 4;
        assert_eq!(
            debuff_setup_priority_delta(&state_1shot, &bash_card, Some(0)),
            0
        );

        // Already-Vulnerable target — bonus suppressed because
        // re-applying Vulnerable doesn't double-stack damage.
        let mut state_vuln = state.clone();
        use sts2_core::{Power, PowerKind, PowerUid};
        let pid = PowerUid(NonZeroU32::new(500).unwrap());
        state_vuln.creatures[0].powers.push(pid);
        state_vuln.powers.push(Power {
            id: pid.0,
            kind: PowerKind::Vulnerable,
            amount: 2,
            counter: 0,
            applier: None,
            skip_next_tick: false,
        });
        assert_eq!(
            debuff_setup_priority_delta(&state_vuln, &bash_card, Some(0)),
            0
        );

        // No follow-up of >= damage — bonus suppressed because there's
        // nothing to cash in the doubled-damage window with.
        let mut state_nofu = state.clone();
        state_nofu.cards[1] = make_attack_card(2, CardId::StrikeIronclad, 4, 1);
        assert_eq!(
            debuff_setup_priority_delta(&state_nofu, &bash_card, Some(0)),
            0
        );

        // Untargeted (e.g. Thunderclap as a single-target Action with
        // None) — gate fails open since we can't pick a survival /
        // already-debuffed branch.
        assert_eq!(debuff_setup_priority_delta(&state, &bash_card, None), 0);
    }

    /// Bead `sts2-rs-g8sy4` (re-close pass 10): pin that
    /// `play_ordering(card, Some(_))` returns the SAME value as
    /// `play_ordering(card, None)` for every (card, character) pair
    /// in the existing sample set, because `per_character_override`
    /// currently returns `None` for all inputs (pinned at the
    /// internal boundary by `a79i_per_character_override_is_empty_at_lift`).
    /// This pin tightens the contract at the public function level:
    /// any future per-character override MUST come with an updated
    /// test (the constant-result invariant breaks first).
    #[test]
    fn play_ordering_g8sy4_per_character_path_matches_base_at_function_level() {
        const CHARS: &[CharacterId] = &[
            CharacterId::Ironclad,
            CharacterId::Silent,
            CharacterId::Defect,
            CharacterId::Necrobinder,
            CharacterId::Regent,
        ];
        // Cover each priority bucket plus the g8sy4 stragglers so a
        // future override on any of these would surface.
        const SAMPLE: &[CardId] = &[
            CardId::Apotheosis,     // -20 (floor)
            CardId::HelloWorldCard, // -2
            CardId::RollingBoulder, // 0
            CardId::ToricToughness, // 10
            CardId::JuggernautCard, // 15
            CardId::Wish,           // 20
            CardId::Stack,          // 22
            CardId::Whistle,        // 25
            CardId::Squash,         // 30
            CardId::Peck,           // 35
            CardId::Shiv,           // 50
            CardId::Folly,          // 70 (ceiling)
        ];
        for cid in SAMPLE {
            let baseline = play_ordering(*cid, None);
            for ch in CHARS {
                let with_char = play_ordering(*cid, Some(*ch));
                assert_eq!(
                    with_char, baseline,
                    "play_ordering({cid:?}, Some({ch:?})) = {with_char} \
                     diverged from baseline = {baseline}; \
                     per_character_override has been wired without \
                     updating this g8sy4 regression pin",
                );
            }
        }
    }

    // ----- Bead `sts2-rs-y9umw` conformance tests -----

    /// Walk every `CardSpec` in every character + status registry, call
    /// `play_ordering` on its `CardId`, and assert the result is in the
    /// documented spread for in-play cards (Apotheosis −20 floor,
    /// Slimed +70 ceiling). Pins the "no fallthrough outside the
    /// documented window" contract: any future addition that lands
    /// outside this range must update the test explicitly.
    #[test]
    fn play_ordering_total_over_card_id() {
        use sts2_content::cards::spec_index::{
            COLORLESS_SPECS, DEFECT_SPECS, IRONCLAD_SPECS, SILENT_SPECS, STATUS_SPECS,
        };
        for factory in IRONCLAD_SPECS
            .iter()
            .chain(SILENT_SPECS.iter())
            .chain(DEFECT_SPECS.iter())
            .chain(COLORLESS_SPECS.iter())
            .chain(STATUS_SPECS.iter())
        {
            let spec = factory();
            let p = play_ordering(spec.id, None);
            assert!(
                (-20..=70).contains(&p),
                "play_ordering({:?}) = {} out of documented [-20, 70] spread",
                spec.id,
                p,
            );
            assert!(
                p < END_TURN_PRIORITY,
                "END_TURN_PRIORITY ({END_TURN_PRIORITY}) is reserved for the turn-end action, not card {:?}",
                spec.id,
            );
        }
    }

    /// Group every registered `CardSpec` by `CardType` and assert the
    /// per-class mean priority respects "powers play earliest, status
    /// cards latest": `mean(Power) < mean(Attack) < mean(Status)`.
    /// Defends are typed `Skill` in `CardType` — the bead's "powers >
    /// attacks > defends > skills" wording maps to the structural
    /// ordering once the type-system reality is taken into account.
    #[test]
    fn play_ordering_class_buckets_monotonic() {
        use sts2_content::cards::spec_index::{
            COLORLESS_SPECS, DEFECT_SPECS, IRONCLAD_SPECS, SILENT_SPECS, STATUS_SPECS,
        };
        use sts2_core::cards::CardType;
        let mut sums: [(i64, u32); 5] = [(0, 0); 5];
        for factory in IRONCLAD_SPECS
            .iter()
            .chain(SILENT_SPECS.iter())
            .chain(DEFECT_SPECS.iter())
            .chain(COLORLESS_SPECS.iter())
            .chain(STATUS_SPECS.iter())
        {
            let spec = factory();
            let bucket = match spec.card_type {
                CardType::Attack => 0,
                CardType::Skill => 1,
                CardType::Power => 2,
                CardType::Curse => 3,
                CardType::Status => 4,
            };
            let p = play_ordering(spec.id, None);
            sums[bucket].0 += i64::from(p);
            sums[bucket].1 += 1;
        }
        let mean = |b: usize| -> f64 {
            assert!(sums[b].1 > 0, "bucket {b} has no entries");
            sums[b].0 as f64 / f64::from(sums[b].1)
        };
        let power_mean = mean(2);
        let attack_mean = mean(0);
        let status_mean = mean(4);
        assert!(
            power_mean < attack_mean,
            "power mean {power_mean:.2} must be < attack mean {attack_mean:.2}",
        );
        assert!(
            attack_mean < status_mean,
            "attack mean {attack_mean:.2} must be < status mean {status_mean:.2}",
        );
    }

    /// On lethal turns, the override constants must keep Defend below
    /// Strike in priority so Defend dominates the rollout softmax (per
    /// `sts2-rs-5dw7` / `sts2-rs-wver`). Asserts the order at the
    /// constant level: dispatching through `action_priority` would
    /// require a full `CombatState`, but the constants themselves
    /// carry the contract.
    #[test]
    fn lethal_overrides_preserve_class_ordering() {
        // Bind const comparisons to locals so clippy's
        // assertions-on-constants lint doesn't fire (it triggers when
        // both sides of a comparison are const-evaluable; binding
        // routes the value through a runtime variable).
        let defend_below_strike = LETHAL_DEFEND_PRIORITY < LETHAL_STRIKE_PRIORITY;
        assert!(
            defend_below_strike,
            "LETHAL_DEFEND_PRIORITY ({LETHAL_DEFEND_PRIORITY}) must rank below LETHAL_STRIKE_PRIORITY ({LETHAL_STRIKE_PRIORITY}) so Defend wins the softmax on lethal turns",
        );
        // Defend sits in the negative range (high-priority bucket) so
        // its softmax weight vs Strike at LETHAL_STRIKE_PRIORITY (90)
        // is overwhelming: at tau=32, exp(-(-5-90)/32) ≈ 19.5×, giving
        // Defend ~95% weight per the constant's docstring.
        let defend_below_neutral = LETHAL_DEFEND_PRIORITY < NEUTRAL_PRIORITY;
        assert!(
            defend_below_neutral,
            "LETHAL_DEFEND_PRIORITY ({LETHAL_DEFEND_PRIORITY}) must sit below NEUTRAL_PRIORITY ({NEUTRAL_PRIORITY}) so Defend ranks as a high-priority play",
        );
        // Strike sits above the in-play ceiling so it's strongly
        // penalised vs Defend.
        let strike_above_ceiling = LETHAL_STRIKE_PRIORITY > 70;
        assert!(
            strike_above_ceiling,
            "LETHAL_STRIKE_PRIORITY ({LETHAL_STRIKE_PRIORITY}) must sit above the play_ordering ceiling (70, Slimed)",
        );
    }
}
