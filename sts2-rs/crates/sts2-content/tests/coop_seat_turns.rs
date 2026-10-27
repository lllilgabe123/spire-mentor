//! A whole co-op fight (`CoopState::full`): every teammate runs their own turn
//! start, seated, between ours and the side's next step (C#
//! `CombatManager.cs:424-607` finishes each phase over every player first).

use std::num::NonZeroU32;

use sts2_content::cards::dispatch as content_dispatch;
use sts2_core::cards::CardId;
use sts2_core::{
    CardUid, CombatPhase, CombatState, MonsterKind, PlayerCombatState, Power, PowerKind, PowerUid,
    Relic, RelicKind,
};

fn mint(state: &mut CombatState, id: CardId) -> CardUid {
    let factory = sts2_core::lookup_card_factory().expect("card factory installed");
    let mut card = factory(id, false).expect("card factory has the card");
    let uid = CardUid::new(u32::try_from(state.cards.len() + 1).expect("fits")).expect("nz");
    card.uid = uid;
    state.cards.push(card);
    uid
}

/// Us (Ironclad, ten Strikes to draw) and one teammate seat (Silent, ten
/// Defends to draw, `mate_block` block), at the top of `round`, one Nibbit.
fn two_seat_fight(round: u16, mate_block: i32) -> CombatState {
    content_dispatch::install_default_registry();
    sts2_core::register_card_factory(sts2_content::cards::spec_index::card_pod_from_id);
    sts2_core::relic_effects::install_default_registry();
    let mut state = CombatState {
        run_rng: sts2_core::run_rng_set::RunRngSet::new(4242_i32),
        ..Default::default()
    };
    state.player.character = Some(sts2_core::potions::CharacterId::Ironclad);
    state.player.current_hp = 80;
    state.player.max_hp = 80;
    state.init_player_creature();
    let enemy = state.spawn_enemy(MonsterKind::Nibbit);
    if let Some(c) = state.creatures.iter_mut().find(|c| c.id == enemy.0) {
        c.max_hp = 200;
        c.current_hp = 200;
    }
    for _ in 0..10 {
        let uid = mint(&mut state, CardId::StrikeIronclad);
        state.player.draw.push(uid);
    }
    let mut mate = PlayerCombatState {
        character: Some(sts2_core::potions::CharacterId::Silent),
        current_hp: 70,
        max_hp: 70,
        block: mate_block,
        ..PlayerCombatState::default()
    };
    for _ in 0..10 {
        mate.draw.push(mint(&mut state, CardId::DefendSilent));
    }
    state.allies.push(mate);
    sts2_core::coop::open_seats(&mut state);
    state.coop.full = true;
    state.round = round;
    state.phase = CombatPhase::PlayerTurnStart;
    state
}

fn give_mate_power(state: &mut CombatState, id: u32, kind: PowerKind, amount: i32) {
    let id = NonZeroU32::new(id).expect("nz");
    state.powers.push(Power {
        id,
        kind,
        amount,
        counter: 0,
        applier: None,
        skip_next_tick: false,
    });
    state.coop.stash[0].powers.push(PowerUid(id));
}

#[test]
fn every_seat_draws_and_resets_energy_at_turn_start() {
    let mut state = two_seat_fight(1, 0);
    state.coop.stash[0].relics.push(Relic {
        id: NonZeroU32::new(50).expect("nz"),
        kind: RelicKind::Lantern,
        counter: 0,
        is_wax: false,
        is_melted: false,
    });

    let _ = sts2_core::combat_loop::start_player_turn(&mut state);

    assert_eq!(state.phase, CombatPhase::PlayerPlayPhase);
    assert_eq!(state.coop.seated, 0, "we sit in our seat again");
    assert_eq!(state.coop.seat_run, None);
    assert_eq!(state.player.hand.len(), 5);
    assert_eq!(state.player.energy, 3, "the teammate's Lantern is not ours");
    let mate = &state.allies[0];
    assert_eq!(mate.hand.len(), 5, "the teammate drew their own hand");
    assert_eq!(mate.draw.len(), 5);
    assert_eq!(mate.energy, 4, "3, and 1 from their Lantern on turn 1");
    assert_eq!(state.coop.to_play, 1, "they have their plays ahead of them");
}

#[test]
fn the_teammates_block_clears_and_their_choice_is_answered() {
    let mut state = two_seat_fight(2, 7);
    // Draw one more, then discard one: a pick out of six.
    give_mate_power(&mut state, 900, PowerKind::ToolsOfTheTrade, 1);

    let _ = sts2_core::combat_loop::start_player_turn(&mut state);

    assert_eq!(state.phase, CombatPhase::PlayerPlayPhase);
    assert!(state.pending_choice.is_none(), "no choice is left for us");
    let mate = &state.allies[0];
    assert_eq!(mate.block, 0, "their block cleared on turn 2");
    assert_eq!(mate.hand.len(), 5, "drew 6, discarded 1");
    assert_eq!(mate.discard.len(), 1);
    assert_eq!(state.player.hand.len(), 5, "our draw is our own");
    assert!(state.player.discard.is_empty());
}

#[test]
fn the_advisors_board_leaves_the_teammate_alone() {
    let mut state = two_seat_fight(2, 7);
    state.coop.full = false;

    let _ = sts2_core::combat_loop::start_player_turn(&mut state);

    let mate = &state.allies[0];
    assert_eq!(mate.hand.len(), 0);
    assert_eq!(mate.block, 7);
    assert_eq!(state.coop.to_play, 0);
}

fn give_mate_relic(state: &mut CombatState, id: u32, kind: RelicKind) {
    state.coop.stash[0].relics.push(Relic {
        id: NonZeroU32::new(id).expect("nz"),
        kind,
        counter: 0,
        is_wax: false,
        is_melted: false,
    });
}

#[test]
fn every_seat_ends_its_turn_before_the_side_does() {
    let mut state = two_seat_fight(1, 0);
    let _ = sts2_core::combat_loop::start_player_turn(&mut state);
    let dazed = mint(&mut state, CardId::Dazed);
    state.allies[0].hand.push(dazed);
    give_mate_relic(&mut state, 51, RelicKind::Orichalcum);

    let _ = sts2_core::combat_loop::end_player_turn_phase_one(&mut state);
    let _ = sts2_core::combat_loop::end_player_turn_phase_two(&mut state);

    assert_eq!(state.coop.seated, 0);
    assert_eq!(state.phase, CombatPhase::EnemyTurnStart);
    let mate = &state.allies[0];
    assert_eq!(
        mate.block, 6,
        "their Orichalcum heard the side's BeforeTurnEnd"
    );
    assert_eq!(
        mate.exhaust.as_slice(),
        [dazed],
        "their Ethereal Dazed exhausted"
    );
    assert!(mate.hand.is_empty(), "their hand flushed");
    assert_eq!(mate.discard.len(), 5);
    assert_eq!(state.player.block, 0, "their Orichalcum is not ours");
    assert!(state.player.hand.is_empty());
}

#[test]
fn a_whole_round_settles_with_every_seat_ready_to_play() {
    let mut state = two_seat_fight(1, 0);
    give_mate_power(&mut state, 901, PowerKind::WellLaidPlans, 1);
    let _ = sts2_core::combat_loop::start_player_turn(&mut state);

    let _ = sts2_core::apply(&mut state, sts2_core::Action::EndTurn);

    assert!(
        state.pending_choice.is_none(),
        "their retain pick was answered"
    );
    assert_eq!(state.phase, CombatPhase::PlayerPlayPhase);
    assert_eq!(state.round, 2);
    assert_eq!(state.coop.seated, 0);
    assert_eq!(state.coop.seat_run, None);
    let mate = &state.allies[0];
    assert!(mate.hand.len() >= 5, "a new hand, plus anything retained");
    assert_eq!(mate.hand.len() + mate.draw.len() + mate.discard.len(), 10);
    assert_eq!(state.coop.to_play, 1);
}

#[test]
fn every_seat_starts_the_combat() {
    let mut state = two_seat_fight(1, 0);
    give_mate_relic(&mut state, 52, RelicKind::Anchor);
    let before = state.allies[0].draw.clone();

    let _ = sts2_core::combat_loop::start_combat(&mut state);
    let _ = sts2_core::combat_loop::start_player_turn(&mut state);

    let mate = &state.allies[0];
    assert_eq!(mate.block, 10, "their Anchor fired at combat start");
    assert_eq!(state.player.block, 0, "their Anchor is not ours");
    let mut all: Vec<_> = mate.hand.iter().chain(mate.draw.iter()).copied().collect();
    all.sort();
    let mut was: Vec<_> = before.to_vec();
    was.sort();
    assert_eq!(all, was, "their deck is all theirs");
}

/// Each seat's turn-start hand as positions in its own discard pile, after
/// both piles (ten Strikes each) reshuffle from the one shared stream, with
/// us at `seat_index` in the run's player order.
fn reshuffled_hands(seat_index: u8) -> (Vec<usize>, Vec<usize>) {
    let mut state = two_seat_fight(2, 0);
    let ours = std::mem::take(&mut state.player.draw);
    let mut theirs = std::mem::take(&mut state.allies[0].draw);
    for uid in &mut theirs {
        let i = uid.0.get() as usize - 1;
        state.cards[i].card_id = CardId::StrikeIronclad;
    }
    state.player.discard.clone_from(&ours);
    state.allies[0].discard.clone_from(&theirs);
    state.coop.seat_index = seat_index;

    let _ = sts2_core::combat_loop::start_player_turn(&mut state);

    let positions = |hand: &[CardUid], pile: &[CardUid]| -> Vec<usize> {
        hand.iter()
            .map(|u| pile.iter().position(|p| p == u).expect("from the pile"))
            .collect()
    };
    (
        positions(&state.player.hand, &ours),
        positions(&state.allies[0].hand, &theirs),
    )
}

#[test]
fn the_seat_ahead_in_the_player_order_shuffles_first() {
    let (us_first, mate_second) = reshuffled_hands(0);
    let (us_second, mate_first) = reshuffled_hands(1);

    assert_ne!(us_first, mate_second, "two shuffles of the stream differ");
    assert_eq!(
        mate_first, us_first,
        "seated second, the teammate shuffles first"
    );
    assert_eq!(us_second, mate_second);
}
