//! Co-op scaling (C# `MultiplayerScalingModel`, `ScaleHpForMultiplayer`,
//! `GetScaledAmountForMultiplayer`) and player death in a whole co-op fight
//! (`CreatureCmd.Kill`, `HandlePlayerDeath`, `ReviveBeforeCombatEnd`).

use std::num::NonZeroU32;

use sts2_content::cards::dispatch as content_dispatch;
use sts2_core::cards::CardId;
use sts2_core::hooks::{CreatureRef, ValueProps};
use sts2_core::{
    Action, CardUid, CombatPhase, CombatState, CreatureUid, MonsterKind, PlayerCombatState,
    PowerKind, Relic, RelicKind,
};

fn install() {
    content_dispatch::install_default_registry();
    sts2_core::register_card_factory(sts2_content::cards::spec_index::card_pod_from_id);
    sts2_core::relic_effects::install_default_registry();
}

fn mint(state: &mut CombatState, id: CardId) -> CardUid {
    let factory = sts2_core::lookup_card_factory().expect("card factory installed");
    let mut card = factory(id, false).expect("card factory has the card");
    let uid = CardUid::new(u32::try_from(state.cards.len() + 1).expect("fits")).expect("nz");
    card.uid = uid;
    state.cards.push(card);
    uid
}

/// A bare board seeded the same way every time, with `mates` teammate seats
/// (whole players in a full fight when there is at least one).
fn board(mates: usize) -> CombatState {
    install();
    let mut state = CombatState {
        run_rng: sts2_core::run_rng_set::RunRngSet::new(4242_i32),
        ..Default::default()
    };
    state.player.current_hp = 80;
    state.player.max_hp = 80;
    state.init_player_creature();
    for _ in 0..mates {
        state.allies.push(PlayerCombatState {
            current_hp: 70,
            max_hp: 70,
            ..PlayerCombatState::default()
        });
    }
    if mates > 0 {
        sts2_core::coop::open_seats(&mut state);
        state.coop.full = true;
    }
    state
}

fn enemy_hp(state: &CombatState, uid: CreatureUid) -> i32 {
    state
        .creatures
        .iter()
        .find(|c| c.id == uid.0)
        .expect("enemy")
        .max_hp
}

fn power(state: &CombatState, uid: CreatureUid, kind: PowerKind) -> i32 {
    let c = state
        .creatures
        .iter()
        .find(|c| c.id == uid.0)
        .expect("creature");
    c.powers
        .iter()
        .filter_map(|u| state.powers.get_by_id(u.0))
        .filter(|p| p.kind == kind)
        .map(|p| p.amount)
        .sum()
}

/// `hp × players × 1.1` in act 1, `× 1.2` in act 2, truncated.
#[test]
fn monster_hp_scales_with_the_player_count_and_act() {
    let mut solo = board(0);
    let uid = solo.spawn_enemy(MonsterKind::Nibbit);
    let base = enemy_hp(&solo, uid);

    let mut two = board(1);
    let uid = two.spawn_enemy(MonsterKind::Nibbit);
    assert_eq!(enemy_hp(&two, uid), base * 2 * 11 / 10);

    let mut act2 = board(2);
    act2.act_index = 2;
    let uid = act2.spawn_enemy(MonsterKind::Nibbit);
    assert_eq!(enemy_hp(&act2, uid), base * 3 * 12 / 10);
}

/// An enemy's move block and its `ShouldScaleInMultiplayer` powers grow with
/// the players; a debuff on it does not.
#[test]
fn enemy_block_and_scaled_powers_grow_with_the_players() {
    let mut state = board(1);
    let uid = state.spawn_enemy(MonsterKind::Nibbit);
    sts2_core::hooks::apply_effect(
        &mut state,
        sts2_core::hooks::Effect::GainBlockCreature {
            target: CreatureRef::Enemy(uid),
            amount: 10,
        },
    );
    let block = state
        .creatures
        .iter()
        .find(|c| c.id == uid.0)
        .expect("enemy")
        .block;
    assert_eq!(block, 22, "10 × 2 players × 1.1");

    sts2_core::card_effects::apply_power_to_creature(&mut state, uid, PowerKind::Plating, 4);
    assert_eq!(power(&state, uid, PowerKind::Plating), 12, "(2·1 + 1) × 4");
    sts2_core::card_effects::apply_power_to_creature(&mut state, uid, PowerKind::Artifact, 1);
    assert_eq!(
        power(&state, uid, PowerKind::Artifact),
        2,
        "1 + players − 1"
    );
    sts2_core::card_effects::apply_power_to_creature(&mut state, uid, PowerKind::Vulnerable, 2);
    assert_eq!(
        power(&state, uid, PowerKind::Artifact),
        1,
        "Artifact ate the debuff"
    );
}

/// Kill the teammate in seat `ally` through the damage pipeline, seated.
fn kill_mate(state: &mut CombatState, ally: usize, dealer: CreatureUid) {
    sts2_core::coop::swap_seat(state, ally);
    let _ = sts2_core::combat_loop::apply_damage_to_player(
        state,
        CreatureRef::Enemy(dealer),
        999,
        ValueProps::MOVE,
        None,
    );
    sts2_core::coop::swap_seat(state, ally);
}

/// A dead teammate loses their cards and energy and their relics go quiet;
/// the fight goes on.
#[test]
fn a_dead_teammate_is_out_but_the_fight_goes_on() {
    let mut state = board(1);
    let enemy = state.spawn_enemy(MonsterKind::Nibbit);
    for _ in 0..3 {
        let uid = mint(&mut state, CardId::DefendSilent);
        state.allies[0].hand.push(uid);
    }
    state.allies[0].energy = 3;
    state.coop.stash[0].relics.push(Relic {
        id: NonZeroU32::new(60).expect("nz"),
        kind: RelicKind::Anchor,
        counter: 0,
        is_wax: false,
        is_melted: false,
    });

    kill_mate(&mut state, 0, enemy);

    assert!(!state.is_over, "we still stand");
    let mate = &state.allies[0];
    assert_eq!(mate.current_hp, 0);
    assert!(mate.hand.is_empty(), "their cards left the combat");
    assert_eq!(mate.energy, 0);
    assert!(
        state.coop.stash[0].relics.is_empty(),
        "their relics went quiet"
    );
    assert_eq!(state.coop.benched.len(), 1);
}

/// With us dead and a teammate alive, we can only end the turn and no hit
/// lands on us; once the teammate dies too the fight is lost.
#[test]
fn we_are_out_until_every_player_is_dead() {
    let mut state = board(1);
    let enemy = state.spawn_enemy(MonsterKind::Nibbit);
    state.phase = CombatPhase::PlayerPlayPhase;
    let hit = |state: &mut CombatState| {
        sts2_core::combat_loop::apply_damage_to_player(
            state,
            CreatureRef::Enemy(enemy),
            999,
            ValueProps::MOVE,
            None,
        )
    };

    assert_eq!(hit(&mut state), None, "a teammate lives: not lost");
    assert!(!state.is_over);
    assert!(sts2_core::coop::is_out(&state));
    assert_eq!(
        sts2_core::legal_actions(&state).as_slice(),
        [Action::EndTurn]
    );
    assert_eq!(hit(&mut state), None, "a dead player is not hit again");

    kill_mate(&mut state, 0, enemy);
    assert!(state.is_over, "every player is dead");
    assert!(!state.player_won);
}

/// A won fight revives every dead player at 1 HP, with their relics back.
#[test]
fn a_win_revives_the_dead_at_one_hp() {
    let mut state = board(1);
    let enemy = state.spawn_enemy(MonsterKind::Nibbit);
    state.coop.stash[0].relics.push(Relic {
        id: NonZeroU32::new(61).expect("nz"),
        kind: RelicKind::Anchor,
        counter: 0,
        is_wax: false,
        is_melted: false,
    });
    kill_mate(&mut state, 0, enemy);
    if let Some(c) = state.creatures.iter_mut().find(|c| c.id == enemy.0) {
        c.current_hp = 0;
    }

    let _ = sts2_core::combat_loop::check_combat_end(&mut state);

    assert!(state.is_over && state.player_won);
    assert_eq!(state.allies[0].current_hp, 1);
    assert_eq!(state.coop.stash[0].relics.len(), 1, "their relics are back");
    assert!(state.coop.benched.is_empty());
}

/// With us out, a round still runs: our setup is skipped, the teammate draws
/// and plays on, and we come back to a turn where we can only end it.
#[test]
fn a_round_runs_on_while_we_are_out() {
    let mut state = board(1);
    let enemy = state.spawn_enemy(MonsterKind::Nibbit);
    if let Some(c) = state.creatures.iter_mut().find(|c| c.id == enemy.0) {
        c.max_hp = 500;
        c.current_hp = 500;
    }
    for _ in 0..10 {
        let uid = mint(&mut state, CardId::DefendSilent);
        state.allies[0].draw.push(uid);
    }
    state.round = 2;
    state.phase = CombatPhase::PlayerPlayPhase;
    let _ = sts2_core::combat_loop::apply_damage_to_player(
        &mut state,
        CreatureRef::Enemy(enemy),
        999,
        ValueProps::MOVE,
        None,
    );
    assert!(sts2_core::coop::is_out(&state));

    let _ = sts2_core::apply(&mut state, Action::EndTurn);

    assert!(!state.is_over, "the teammate stands");
    assert_eq!(state.round, 3);
    assert_eq!(state.phase, CombatPhase::PlayerPlayPhase);
    assert_eq!(state.player.energy, 0, "no setup for the dead");
    assert!(state.player.hand.is_empty());
    assert_eq!(state.allies[0].hand.len(), 5, "the teammate drew");
    assert_eq!(
        sts2_core::legal_actions(&state).as_slice(),
        [Action::EndTurn]
    );
}

/// An encounter's opening roster scales too (`run_bridge::spawn_roster`, the
/// path the run and the scenario builders take).
#[test]
fn the_opening_roster_scales_with_the_players() {
    let mut solo = board(0);
    let roster = [(
        MonsterKind::Nibbit,
        sts2_content::monster_dispatch::MonsterSpawnArgs::None,
        40,
    )];
    let uid = sts2_content::run_bridge::spawn_roster(&mut solo, &roster)[0];
    assert_eq!(enemy_hp(&solo, uid), 40);

    let mut two = board(1);
    let uid = sts2_content::run_bridge::spawn_roster(&mut two, &roster)[0];
    assert_eq!(enemy_hp(&two, uid), 88, "40 × 2 × 1.1");
}

/// The last living player can be a teammate, dying seated (a monster hit
/// lands on them after us): that death ends the fight, lost.
#[test]
fn the_last_teammate_dying_seated_loses_the_fight() {
    let mut state = board(1);
    let enemy = state.spawn_enemy(MonsterKind::Nibbit);
    let _ = sts2_core::combat_loop::apply_damage_to_player(
        &mut state,
        CreatureRef::Enemy(enemy),
        999,
        ValueProps::MOVE,
        None,
    );
    assert!(!state.is_over, "the teammate still stands");

    // The enemy turn's per-hit path: the teammate seated, hit, unseated.
    sts2_core::coop::each_player(&mut state, |state| {
        let _ = sts2_core::combat_loop::apply_damage_to_player(
            state,
            CreatureRef::Enemy(enemy),
            999,
            ValueProps::MOVE,
            None,
        );
    });

    assert!(state.is_over, "every player is dead");
    assert!(!state.player_won);
}

/// v0.111.0 scales a two-player enemy's move block by the player count alone
/// (C# `count <= 2 ? count : count × scaling`); v0.107.1 by 2 × 1.1.
#[test]
fn v0111_two_player_block_drops_the_act_factor() {
    for (version, want) in [
        (sts2_core::GameVersion::V0_107_1, 11),
        (sts2_core::GameVersion::V0_111_0, 10),
    ] {
        let mut state = board(1);
        state.game_version = version;
        let uid = state.spawn_enemy(MonsterKind::Nibbit);
        sts2_core::hooks::apply_effect(
            &mut state,
            sts2_core::hooks::Effect::GainBlockCreature {
                target: CreatureRef::Enemy(uid),
                amount: 5,
            },
        );
        let block = state
            .creatures
            .iter()
            .find(|c| c.id == uid.0)
            .expect("enemy")
            .block;
        assert_eq!(block, want, "{version:?}");
    }
}
