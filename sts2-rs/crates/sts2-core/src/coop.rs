//! Co-op seats: the teammates of a multiplayer fight as whole players.
//!
//! The engine is written for one player: every card, power and relic acts on
//! `state.player`. A teammate (`state.allies[i]`) is the same
//! [`PlayerCombatState`], so instead of teaching every effect about a second
//! player, [`swap_seat`] puts the teammate in the player's seat, the ordinary
//! engine runs their plays, and a second swap puts us back.
//!
//! What a seat owns outside [`PlayerCombatState`] travels with it in a
//! [`SeatStash`]: the relics (our Kunai must not count their Attacks) and the
//! per-turn event log (their cards are not "cards we played this turn").
//!
//! [`play_teammates`] runs the teammates' remaining plays for the round the
//! board was read on, when we end our turn: C# lets players act in any order
//! and ends the side's turn once everyone is ready, so "after our plays,
//! before the enemy turn" is the order the advice can rely on — our block on
//! them counts only where they leave a gap, and their Vulnerable never
//! inflates a play we are told to make now.
//!
//! In a whole co-op fight ([`CoopState::full`]) the teammates also run their own
//! share of every turn start and turn end (`run_seat_stage`), between ours and
//! the side-wide step that follows it, and their relics hear the side's turn
//! hooks (`each_teammate`).

use std::sync::OnceLock;

use smallvec::SmallVec;

use std::num::NonZeroU32;

use crate::{Action, CombatState, PowerKind, PowerUid, RelicStore, TurnEvents};

/// Co-op bookkeeping on [`CombatState::coop`]. `Default` is solo.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CoopState {
    /// The round whose enemy turn also hits the teammates, and whose play
    /// phase they still have ahead of them. Monster attacks target every
    /// living player (C# `AttackCommand.FromMonster` →
    /// `TargetingAllOpponents`), but the teammates' later turns are not
    /// simulated: past this round they would stand unblocking until they
    /// died, and every line would end with the same dead teammate. Set by
    /// the wire loader when `$.allies` is present, `None` solo.
    pub threat_round: Option<u16>,
    /// Bit `i` set: `allies[i]` has not ended their turn yet and has a hand to
    /// play ([`play_teammates`] clears it once they have).
    pub to_play: u8,
    /// Per ally seat, what that player owns outside [`PlayerCombatState`].
    /// Empty when the wire carried only HP and block.
    pub stash: SmallVec<[SeatStash; 1]>,
    /// True while a [`seat_pass`] runs: the seated player is a teammate
    /// listening to an event SOMEONE ELSE caused, and the power walk yields
    /// only the seated player's listeners (the enemies' already heard it).
    /// See [`actor_is_seated`].
    pub seat_pass: bool,
    /// The seat id of whoever sits in `state.player`: 0 is the player the
    /// state was built for, a teammate's is `ally + 1` ([`SeatStash::seat`]).
    /// [`swap_seat`] trades it with the incoming seat's, so it names a
    /// player across swaps where the shared creature row cannot.
    pub seated: u8,
    /// The seat that applied each live instance of an applier-keyed power
    /// ([`tracks_applier_seat`]). Empty solo; a power absent here is seat 0's.
    pub power_seat: SmallVec<[(NonZeroU32, u8); 2]>,
    /// A whole co-op fight: every round, every living teammate runs their own
    /// turn start and is hit by the enemy turn, as in C#. False for the
    /// advisor's board, whose teammates live only through
    /// [`Self::threat_round`].
    pub full: bool,
    /// The per-seat stage a teammate is running right now, `None` at rest.
    /// The shared turn pipeline returns at the end of the stage instead of
    /// running on into the side's next step (`run_seat_stage`).
    pub seat_run: Option<SeatRun>,
}

/// A teammate's share of the turn pipeline, run with them seated. C# runs each
/// phase of a turn over every player before the next phase begins
/// (`CombatManager.cs:424-607`); these are the per-player phases, cut where the
/// side-wide step that follows them begins.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeatRun {
    /// `SetupPlayerTurn`: block clear, energy, hand draw, the
    /// `AfterEnergyReset` and `AfterPlayerTurnStart` hooks. Stops before
    /// `AfterSideTurnStart`.
    Setup,
    /// `RunAutoPrePlayPhase`: Bombardment, Mayhem, Whispering Earring. Stops
    /// before the combat-end check and the play phase.
    AutoPrePlay,
    /// `AfterAutoPostPlayPhaseEntered`: Stampede, I Am Invincible, Howl, the
    /// turn-end block snapshot. Stops before `BeforeTurnEnd`.
    AutoPostPlay,
    /// `DoTurnEnd` and the flush: Chains, the orb passives, Ethereal, the
    /// turn-end-in-hand cards, Well-Laid Plans, the flush and the per-turn
    /// card cleanup. Stops before `AfterTurnEnd`.
    TurnEnd,
}

/// Powers whose instance belongs to the PLAYER who applied it rather than to
/// the creature it sits on: C# keeps one instance per applier
/// (`InstancedPerApplier`) and reacts only to that player's cards
/// (`cardPlay.Card.Owner == Applier.Player`).
#[must_use]
pub const fn tracks_applier_seat(kind: PowerKind) -> bool {
    matches!(kind, PowerKind::Strangle | PowerKind::Oblivion)
}

/// Record that the seated player applied the fresh power `id`. A no-op solo
/// and for kinds outside [`tracks_applier_seat`].
pub fn stamp_power_seat(state: &mut CombatState, id: NonZeroU32, kind: PowerKind) {
    if !has_seats(state) || !tracks_applier_seat(kind) {
        return;
    }
    let powers = &state.powers;
    state
        .coop
        .power_seat
        .retain(|(p, _)| *p != id && powers.get_by_id(*p).is_some());
    state.coop.power_seat.push((id, state.coop.seated));
}

/// Whether the seated player applied power `id`. A power with no recorded
/// seat (solo, or read off the wire, which does not say who applied it) is
/// seat 0's, the player the state was built for.
#[must_use]
pub fn applied_by_seated(state: &CombatState, id: NonZeroU32) -> bool {
    let seat = state
        .coop
        .power_seat
        .iter()
        .find(|(p, _)| *p == id)
        .map_or(0, |(_, seat)| *seat);
    seat == state.coop.seated
}

/// Whether the event being dispatched was caused by the seated player.
///
/// Every player-action hook (a card played, drawn or exhausted, block gained,
/// energy or stars spent, an orb channeled …) fires for the seated player's
/// action and then once more for each teammate in a [`seat_pass`], with that
/// teammate seated. An arm C# guards on its owner (`cardPlay.Card.Owner ==
/// Owner`, `player == Owner.Player`) must check this: true in the main walk,
/// false in a teammate's pass. Solo it is always true.
#[must_use]
pub fn actor_is_seated(state: &CombatState) -> bool {
    !state.coop.seat_pass
}

/// Whether the fight has teammate seats to run [`seat_pass`]es for.
#[must_use]
pub fn has_seats(state: &CombatState) -> bool {
    !state.coop.stash.is_empty()
}

/// `kind` summed over every player: the seated one and each living teammate
/// waiting in a [`SeatStash`] (C# `GetOpponentsOf(enemy).Where(IsAlive)`).
/// Solo, the seated player's amount.
#[must_use]
pub fn players_power_amount(state: &CombatState, kind: PowerKind) -> i32 {
    let mut sum = state.player_power_amount(kind);
    for (i, seat) in state.coop.stash.iter().enumerate() {
        if state.allies.get(i).is_some_and(|a| a.current_hp > 0) {
            sum += seat
                .powers
                .iter()
                .filter_map(|u| state.powers.get_by_id(u.0))
                .filter(|p| p.kind == kind)
                .map(|p| p.amount)
                .sum::<i32>();
        }
    }
    sum
}

/// Give every ally an empty [`SeatStash`] with its seat id, making them
/// whole players.
pub fn open_seats(state: &mut CombatState) {
    state.coop.stash = (0..state.allies.len())
        .map(|i| SeatStash {
            seat: u8::try_from(i + 1).unwrap_or(u8::MAX),
            ..SeatStash::default()
        })
        .collect();
}

/// Let every living teammate hear the event `body` dispatches: seat them,
/// remap the context, run `body` with only their listeners on the power walk,
/// and stand them back up. The caller has already run `body` for the seated
/// player and the enemies (C# walks every player's hooks, then the enemies';
/// the teammates come after the enemies here, the one ordering difference).
///
/// A no-op solo, and never nested: a seat pass does not open further passes
/// for events it causes (C#'s owner-guarded arms would ignore them anyway,
/// and the few unguarded ones are listed in the co-op plan).
pub fn seat_pass<C: SeatRemap>(
    state: &mut CombatState,
    ctx: &C,
    mut body: impl FnMut(&mut CombatState, &C),
) {
    if state.coop.seat_pass || !has_seats(state) {
        return;
    }
    for ally in 0..state.allies.len() {
        if state.allies[ally].current_hp <= 0 || state.coop.stash.get(ally).is_none() {
            continue;
        }
        let remapped = ctx.seat_remap(u8::try_from(ally).unwrap_or(u8::MAX));
        swap_seat(state, ally);
        state.coop.seat_pass = true;
        body(state, &remapped);
        state.coop.seat_pass = false;
        swap_seat(state, ally);
    }
}

/// An event context seen from a teammate's seat: after [`swap_seat`]`(ally)`
/// the player who caused it sits at `allies[ally]` and the teammate is
/// `Player`, so the two references trade places. Everything else is
/// unchanged.
pub trait SeatRemap: Sized + Clone {
    #[must_use]
    fn seat_remap(&self, _ally: u8) -> Self {
        self.clone()
    }
}

impl crate::hooks::CreatureRef {
    /// `self` as seen once `allies[ally]` and the player swap seats.
    #[must_use]
    pub fn seat_remap(self, ally: u8) -> Self {
        use crate::hooks::CreatureRef;
        match self {
            CreatureRef::Player => CreatureRef::Ally(ally),
            CreatureRef::Ally(a) if a == ally => CreatureRef::Player,
            other => other,
        }
    }
}

/// What an UNSEATED player owns that must stay off the live state: their
/// relics, their per-turn event log, and their powers.
///
/// The powers matter most. The power bus walks `state.allies[*].powers` as
/// well as the player's (`hooks::iter_power_listeners`), and some arms act on
/// whoever is seated rather than on their owner (Aggression's
/// discard-to-hand, for one). Kept here, an unseated player's powers are
/// inert until their seat comes up — the engine stays single-player at any
/// moment, which is what every arm was written for.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SeatStash {
    /// This seat's id ([`CoopState::seated`]) while it waits here.
    pub seat: u8,
    pub relics: RelicStore,
    pub combat_log: TurnEvents,
    pub powers: SmallVec<[PowerUid; 8]>,
}

/// Exchange the player's seat with `allies[ally]`. Calling it twice restores
/// the state.
///
/// The player's creature row (`player.creature_uid`) always stands for
/// whoever is seated: it is re-synced from the incoming seat, and teammates
/// never own one, so every walk that tells the player's row from the
/// monsters' is unchanged. The outgoing seat's powers (the row's and the
/// legacy list's, see [`CombatState::player_power_uids`]), relics and event
/// log go to its [`SeatStash`]; the incoming seat's come out of it.
///
/// Without a stash (the wire carried only HP and block) only the
/// [`PlayerCombatState`]s swap.
///
/// # Panics
/// If `ally` is out of range.
pub fn swap_seat(state: &mut CombatState, ally: usize) {
    let outgoing_powers = state.player_power_uids();
    let row = state.player.creature_uid.take();
    std::mem::swap(&mut state.player, &mut state.allies[ally]);
    state.player.creature_uid = row;
    if let Some(stash) = state.coop.stash.get_mut(ally) {
        std::mem::swap(&mut state.relics, &mut stash.relics);
        std::mem::swap(&mut state.combat_log, &mut stash.combat_log);
        state.player.powers = std::mem::replace(&mut stash.powers, outgoing_powers);
        state.allies[ally].powers.clear();
        std::mem::swap(&mut state.coop.seated, &mut stash.seat);
    } else {
        state.allies[ally].powers = outgoing_powers;
    }
    state.sync_player_creature();
}

// Contexts whose creature references trade places in a seat pass.

impl SeatRemap for crate::hooks::DamageCtx {
    fn seat_remap(&self, ally: u8) -> Self {
        let mut out = *self;
        out.dealer = out.dealer.seat_remap(ally);
        out.target = out.target.seat_remap(ally);
        out
    }
}

impl SeatRemap for crate::hooks::BlockCtx {
    fn seat_remap(&self, ally: u8) -> Self {
        let mut out = *self;
        out.target = out.target.seat_remap(ally);
        out
    }
}

impl SeatRemap for crate::hooks::HpLostCtx {
    fn seat_remap(&self, ally: u8) -> Self {
        let mut out = *self;
        out.dealer = out.dealer.seat_remap(ally);
        out.target = out.target.seat_remap(ally);
        out
    }
}

impl SeatRemap for crate::hooks::CardCostCtx {
    fn seat_remap(&self, ally: u8) -> Self {
        let mut out = *self;
        out.owner = out.owner.seat_remap(ally);
        out
    }
}

impl SeatRemap for crate::hooks::PowerAppliedCtx {
    fn seat_remap(&self, ally: u8) -> Self {
        let mut out = *self;
        out.target = out.target.seat_remap(ally);
        out.applier = out.applier.map(|r| r.seat_remap(ally));
        out
    }
}

impl SeatRemap for crate::hooks::CardPlayCountCtx {
    fn seat_remap(&self, ally: u8) -> Self {
        let mut out = *self;
        out.owner = out.owner.seat_remap(ally);
        out
    }
}

impl SeatRemap for crate::hooks::BlockClearedCtx {
    fn seat_remap(&self, ally: u8) -> Self {
        let mut out = *self;
        out.cleared = out.cleared.map(|r| r.seat_remap(ally));
        out
    }
}

impl SeatRemap for crate::hooks::BlockGainedCtx {
    fn seat_remap(&self, ally: u8) -> Self {
        let mut out = *self;
        out.target = out.target.seat_remap(ally);
        out
    }
}

impl SeatRemap for crate::hooks::ForgeCtx {
    fn seat_remap(&self, ally: u8) -> Self {
        let mut out = *self;
        out.forger = out.forger.seat_remap(ally);
        out
    }
}

impl SeatRemap for crate::hooks::PowerAmountCtx {
    fn seat_remap(&self, ally: u8) -> Self {
        let mut out = *self;
        out.applier = out.applier.map(|r| r.seat_remap(ally));
        out.target = out.target.seat_remap(ally);
        out
    }
}

// Contexts that name no player: the actor is whoever is seated (see
// [`actor_is_seated`]).
impl SeatRemap for crate::hooks::AttackCtx {}
impl SeatRemap for crate::hooks::CardCtx {}
impl SeatRemap for crate::hooks::CardDrawnCtx {}
impl SeatRemap for crate::hooks::HandDrawCtx {}
impl SeatRemap for crate::hooks::MaxEnergyCtx {}
impl SeatRemap for crate::hooks::SummonAmountCtx {}
impl SeatRemap for crate::hooks::SummonCtx {}
impl SeatRemap for crate::hooks::OstyRevivedCtx {}
impl SeatRemap for crate::hooks::TurnSideCtx {}
impl SeatRemap for crate::hooks::TurnSelfCtx {}
impl SeatRemap for crate::hooks::CombatLifecycleCtx {}
impl SeatRemap for crate::hooks::CreatureAddedCtx {}
impl SeatRemap for crate::hooks::OrbChanneledCtx {}
impl SeatRemap for crate::hooks::StarsSpentCtx {}
impl SeatRemap for crate::hooks::TryBlockDebuffCtx {}
impl SeatRemap for crate::hooks::DrawCtx {}
impl SeatRemap for crate::hooks::AfterDeathCtx {}
impl SeatRemap for crate::hooks::AfterBlockBrokenCtx {}

/// How a simulated teammate picks their next action: an index into `actions`
/// (never empty, never `EndTurn`, no potions). `seed` makes the pick a pure
/// function of the state, as every search transition must be.
pub type TeammatePolicy = fn(state: &CombatState, actions: &[Action], seed: u64) -> usize;

static TEAMMATE_POLICY: OnceLock<TeammatePolicy> = OnceLock::new();

/// Install the policy [`play_teammates`] plays the teammates with. The search
/// crate registers its rollout heuristic; without one the teammates do not
/// act. First registration wins.
pub fn register_teammate_policy(policy: TeammatePolicy) {
    let _ = TEAMMATE_POLICY.set(policy);
}

/// Upper bound on one teammate's actions, against a policy that never runs
/// out of free plays.
const MAX_TEAMMATE_ACTIONS: usize = 40;

/// Play out every teammate in [`CoopState::to_play`]: on the round the board was
/// read on, or every round of a [`CoopState::full`] fight. Called when we end
/// our turn, before the side's end-of-turn processing. A no-op solo, past
/// that round, or with no policy registered.
pub fn play_teammates(state: &mut CombatState) {
    if state.coop.to_play == 0 || !teammates_threatened(state) {
        return;
    }
    let Some(policy) = TEAMMATE_POLICY.get().copied() else {
        return;
    };
    let to_play = std::mem::take(&mut state.coop.to_play);
    for ally in 0..state.allies.len().min(8) {
        if to_play & (1 << ally) == 0 || state.allies[ally].current_hp <= 0 {
            continue;
        }
        swap_seat(state, ally);
        // Past the cap only an open choice keeps going: the seat must not be
        // handed back with a selection pending.
        for step in 0..MAX_TEAMMATE_ACTIONS * 2 {
            if state.is_over {
                // A seated teammate's own death is not the fight's: C# loses
                // only when every player is dead (`CreatureCmd.cs:450`).
                if !state.player_won && state.player.current_hp <= 0 {
                    state.is_over = false;
                }
                break;
            }
            if step >= MAX_TEAMMATE_ACTIONS && state.pending_choice.is_none() {
                break;
            }
            let actions: SmallVec<[Action; 16]> = crate::legal_actions(state)
                .into_iter()
                .filter(|a| {
                    !matches!(
                        a,
                        Action::EndTurn | Action::UsePotion { .. } | Action::DrinkPotion { .. }
                    )
                })
                .collect();
            if actions.is_empty() {
                break;
            }
            let seed = (u64::from(state.round) << 16) ^ ((ally as u64) << 8) ^ step as u64;
            let pick = policy(state, &actions, seed).min(actions.len() - 1);
            crate::apply(state, actions[pick].clone());
        }
        swap_seat(state, ally);
    }
}

/// Whether the coming enemy turn reaches the teammates: every round of a
/// [`CoopState::full`] fight, else only the advisor's
/// [`CoopState::threat_round`].
#[must_use]
pub fn teammates_threatened(state: &CombatState) -> bool {
    state.coop.full || state.coop.threat_round == Some(state.round)
}

/// Run `stage` for every living teammate, in seat order, each seated in turn.
/// Called from the seated player's own turn pipeline at the point the stage
/// ends for them, so the players' stages run back to back before the side
/// step that follows (C# finishes a phase over every player first).
///
/// A teammate's choices are answered on the spot by the registered
/// [`TeammatePolicy`] (the first option without one): their stage must be
/// over before the seats swap back. A no-op outside a [`CoopState::full`]
/// fight, and never nested.
pub(crate) fn run_seat_stage(state: &mut CombatState, stage: SeatRun) {
    if !state.coop.full || state.coop.seat_run.is_some() || !has_seats(state) {
        return;
    }
    for ally in 0..state.allies.len() {
        if state.is_over {
            return;
        }
        if state.allies[ally].current_hp <= 0 || state.coop.stash.get(ally).is_none() {
            continue;
        }
        swap_seat(state, ally);
        state.coop.seat_run = Some(stage);
        let _ = match stage {
            SeatRun::Setup => crate::combat_loop::continue_player_turn_setup(state),
            SeatRun::AutoPrePlay => crate::combat_loop::run_auto_pre_play_for_seat(state),
            SeatRun::AutoPostPlay => crate::combat_loop::run_auto_post_play(state),
            SeatRun::TurnEnd => crate::combat_loop::run_turn_end_for_seat(state),
        };
        settle_seat_choices(state, ally);
        state.coop.seat_run = None;
        if stage == SeatRun::AutoPrePlay && ally < 8 {
            // Ready at once if an auto-played card ended their turn (VoidForm);
            // otherwise they have this round's plays ahead of them.
            if std::mem::take(&mut state.player.request_end_turn_after_play) {
                state.coop.to_play &= !(1 << ally);
            } else {
                state.coop.to_play |= 1 << ally;
            }
        }
        forgive_teammate_death(state);
        swap_seat(state, ally);
    }
}

/// Run `body` once for each living teammate, seated as the one acting: their
/// relics' side-turn hooks, which C# walks for every player. Unlike a
/// [`seat_pass`], whatever `body` causes is heard by everyone, as when the
/// teammate plays a card. Only in a [`CoopState::full`] fight (the
/// advisor's teammates carry no relics); never inside a seat pass or a
/// teammate's stage.
pub(crate) fn each_teammate(state: &mut CombatState, mut body: impl FnMut(&mut CombatState)) {
    if !state.coop.full
        || state.coop.seat_pass
        || state.coop.seat_run.is_some()
        || !has_seats(state)
    {
        return;
    }
    for ally in 0..state.allies.len() {
        if state.is_over {
            return;
        }
        if state.allies[ally].current_hp <= 0 || state.coop.stash.get(ally).is_none() {
            continue;
        }
        swap_seat(state, ally);
        body(state);
        forgive_teammate_death(state);
        swap_seat(state, ally);
    }
}

/// A seated teammate's own death is not the fight's: C# loses only when every
/// player is dead (`CreatureCmd.cs:450`).
fn forgive_teammate_death(state: &mut CombatState) {
    if state.is_over && !state.player_won && state.player.current_hp <= 0 {
        state.is_over = false;
    }
}

/// Upper bound on the picks one teammate's stage may need.
const MAX_SEAT_CHOICES: usize = 64;

/// Answer the seated teammate's open choices until none is left. Each answer
/// resumes their stage, which returns at the stage's end
/// ([`CoopState::seat_run`]) or opens the next choice.
fn settle_seat_choices(state: &mut CombatState, ally: usize) {
    let policy = TEAMMATE_POLICY.get().copied();
    for step in 0..MAX_SEAT_CHOICES {
        if state.pending_choice.is_none() || state.is_over {
            return;
        }
        let actions = crate::legal_actions(state);
        if actions.is_empty() {
            break;
        }
        let seed = (u64::from(state.round) << 16) ^ ((ally as u64) << 8) ^ step as u64;
        let pick = policy.map_or(0, |p| p(state, &actions, seed).min(actions.len() - 1));
        crate::apply(state, actions[pick].clone());
    }
    debug_assert!(
        state.pending_choice.is_none() || state.is_over,
        "a teammate's stage left a choice open"
    );
}

// Hooks without a context (`dispatch_after_energy_spent`, …) pass `&()`.
impl SeatRemap for () {}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use super::*;
    use crate::powers::PowerKind;
    use crate::relics::RelicKind;
    use crate::{PlayerCombatState, Power, Relic};

    /// Us (60 HP, Strength 2 on our creature row, Burning Blood) and one
    /// teammate seat (30 HP, 3 energy, Dexterity 1, no relics).
    fn two_seats() -> CombatState {
        let mut state = CombatState::default();
        state.player.current_hp = 60;
        state.player.max_hp = 80;
        // On the creature row only, as card-applied player powers are.
        let row = state.init_player_creature();
        crate::card_effects::apply_power_to_creature(&mut state, row, PowerKind::Strength, 2);
        state.relics.push(Relic {
            id: NonZeroU32::new(10).expect("nz"),
            kind: RelicKind::BurningBlood,
            counter: 0,
            is_wax: false,
            is_melted: false,
        });
        let dex = NonZeroU32::new(900).expect("nz");
        state.powers.push(Power {
            id: dex,
            kind: PowerKind::Dexterity,
            amount: 1,
            counter: 0,
            applier: None,
            skip_next_tick: false,
        });
        state.allies.push(PlayerCombatState {
            current_hp: 30,
            max_hp: 70,
            energy: 3,
            ..PlayerCombatState::default()
        });
        state.coop.stash.push(SeatStash {
            powers: smallvec::smallvec![PowerUid(dex)],
            seat: 1,
            ..SeatStash::default()
        });
        state
    }

    #[test]
    fn a_seated_teammate_is_the_player_and_owns_only_their_own_things() {
        let mut state = two_seats();
        swap_seat(&mut state, 0);
        assert_eq!(state.player.current_hp, 30);
        assert_eq!(state.player.energy, 3);
        assert_eq!(state.player_power_amount(PowerKind::Dexterity), 1);
        assert_eq!(
            state.player_power_amount(PowerKind::Strength),
            0,
            "our Strength stays ours"
        );
        assert!(state.relics.is_empty(), "our Burning Blood stays ours");
        let row = state.player.creature_uid.expect("the row follows the seat");
        let creature = state.creatures.iter().find(|c| c.id == row.0).expect("row");
        assert_eq!(creature.current_hp, 30);
        assert_eq!(state.allies[0].current_hp, 60);
        assert!(state.allies[0].creature_uid.is_none());
        assert!(
            state.allies[0].powers.is_empty(),
            "an unseated player's powers are off the power bus"
        );
    }

    #[test]
    fn swapping_a_seat_twice_restores_the_state() {
        let mut state = two_seats();
        // The row's powers fold into the legacy list on the way out; do the
        // same to the reference so the comparison is about the swap alone.
        state.player.powers = state.player_power_uids();
        state.sync_player_creature();
        let before = state.clone();
        swap_seat(&mut state, 0);
        swap_seat(&mut state, 0);
        assert_eq!(state, before);
    }

    /// Strangle is `InstancedPerApplier`: the teammate's opens a second
    /// instance on the same enemy, and each player's re-apply stacks into
    /// their own.
    #[test]
    fn each_player_strangles_with_their_own_instance() {
        let mut state = two_seats();
        let id = NonZeroU32::new(50).expect("nz");
        state.creatures.push(crate::Creature {
            id,
            current_hp: 30,
            max_hp: 30,
            block: 0,
            powers: SmallVec::new(),
            species: None,
            intent: None,
            intent_history: crate::IntentHistory::default(),
            monster_id: crate::MonsterId("Nibbit"),
            ai: None,
        });
        let enemy = crate::CreatureUid(id);
        state.enemy_uids.push(enemy);
        let strangles = |state: &CombatState| -> Vec<(i32, bool)> {
            let c = state.creatures.iter().find(|c| c.id == id).expect("enemy");
            c.powers
                .iter()
                .filter_map(|u| state.powers.get_by_id(u.0))
                .filter(|p| p.kind == PowerKind::Strangle)
                .map(|p| (p.amount, applied_by_seated(state, p.id)))
                .collect()
        };
        let strangle = |state: &mut CombatState| {
            crate::card_effects::apply_power_to_creature(state, enemy, PowerKind::Strangle, 2);
        };

        strangle(&mut state);
        swap_seat(&mut state, 0);
        strangle(&mut state);
        assert_eq!(strangles(&state), vec![(2, false), (2, true)]);
        swap_seat(&mut state, 0);
        strangle(&mut state);
        assert_eq!(strangles(&state), vec![(4, true), (2, false)]);
    }
}
