//! GH #946: Whispering Earring's turn-1 auto-play.
//!
//! C# `WhisperingEarring.AfterAutoPrePlayPhaseEnteredLate` (identical in
//! v0.107.1 and v0.111.0): on the owner's first turn, under a pushed
//! `VakuuCardSelector`, play up to 13 cards from hand — each time the FIRST
//! hand card whose `CanPlay()` holds, aimed at the leftmost hittable enemy,
//! paid through `SpendResources()` and then `CardCmd.AutoPlay(...,
//! skipXCapture: true)`. Rust had only the relic's +1 max energy.

use smallvec::SmallVec;

use sts2_content::monster_dispatch::{build_ai, MonsterSpawnArgs, SpawnPosition};
use sts2_core::cards::CardId;
use sts2_core::combat_loop::{start_combat, start_player_turn};
use sts2_core::{
    CardUid, CombatPhase, CombatState, Creature, CreatureUid, IntentHistory, MonsterId,
    MonsterKind, PlayerCombatState, Relic, RelicKind, StepResult,
};

fn enemy(id: u32, hp: i32, index: u8, alone: bool) -> Creature {
    Creature {
        id: CreatureUid::new(id).expect("non-zero").0,
        current_hp: hp,
        max_hp: hp,
        block: 0,
        powers: SmallVec::new(),
        species: Some(MonsterKind::ShrinkerBeetle),
        monster_id: MonsterId(MonsterKind::ShrinkerBeetle.python_name()),
        intent: None,
        intent_history: IntentHistory::default(),
        ai: Some(build_ai(
            MonsterKind::ShrinkerBeetle,
            MonsterSpawnArgs::None,
            SpawnPosition {
                alone,
                front: true,
                back: true,
                index,
            },
        )),
    }
}

fn earring(melted: bool) -> Relic {
    Relic {
        id: std::num::NonZeroU32::new(1).expect("non-zero"),
        kind: RelicKind::WhisperingEarring,
        counter: 0,
        is_wax: false,
        is_melted: melted,
    }
}

struct Setup {
    draw: Vec<CardId>,
    enemy_hps: Vec<i32>,
    relics: Vec<Relic>,
}

impl Setup {
    fn new(draw: &[CardId]) -> Self {
        Self {
            draw: draw.to_vec(),
            enemy_hps: vec![400],
            relics: vec![earring(false)],
        }
    }
}

/// Build a combat whose turn-1 draw deals `draw` (top first) and run turn 1's
/// start. Returns the state and the card uids in `draw` order.
fn run_turn_one(setup: Setup) -> (CombatState, Vec<CardUid>, StepResult) {
    run_turn(setup, None)
}

fn run_turn(setup: Setup, round: Option<u16>) -> (CombatState, Vec<CardUid>, StepResult) {
    sts2_content::cards::dispatch::ensure_installed();
    sts2_content::relics::dispatch::ensure_installed();
    sts2_core::register_card_factory(sts2_content::cards::spec_index::card_pod_from_id);

    let mut cards = sts2_core::CardStore::new();
    let mut uids = Vec::new();
    for &id in &setup.draw {
        let uid = CardUid::new(u32::try_from(cards.len() + 1).expect("fits")).expect("non-zero");
        let mut card = sts2_content::cards::spec_index::card_pod_from_id(id, false).expect("spec");
        card.uid = uid;
        cards.push(card);
        uids.push(uid);
    }
    let alone = setup.enemy_hps.len() == 1;
    let creatures: Vec<Creature> = setup
        .enemy_hps
        .iter()
        .enumerate()
        .map(|(i, &hp)| {
            enemy(
                u32::try_from(i + 1).expect("fits"),
                hp,
                u8::try_from(i).expect("fits"),
                alone,
            )
        })
        .collect();
    let enemy_uids: SmallVec<[CreatureUid; 8]> =
        creatures.iter().map(|c| CreatureUid(c.id)).collect();
    let player = PlayerCombatState {
        current_hp: 80,
        max_hp: 80,
        character: Some(sts2_core::potions::CharacterId::Ironclad),
        ..PlayerCombatState::default()
    };
    let mut state = CombatState {
        rng: sts2_core::dotnet_rng::Rng::new(7),
        run_rng: sts2_core::run_rng_set::RunRngSet::new(7),
        creatures,
        cards,
        relics: setup.relics.into(),
        player,
        enemy_uids,
        ..CombatState::default()
    };
    let _ = start_combat(&mut state);
    state.player.hand.clear();
    state.player.discard.clear();
    state.player.exhaust.clear();
    state.player.draw = uids.iter().copied().collect();
    if let Some(r) = round {
        state.round = r;
    }
    let res = start_player_turn(&mut state);
    (state, uids, res)
}

fn ids(state: &CombatState, pile: &[CardUid]) -> Vec<CardId> {
    pile.iter()
        .map(|u| state.cards[u.0.get() as usize - 1].card_id)
        .collect()
}

fn hp(state: &CombatState, idx: usize) -> i32 {
    let uid = state.enemy_uids[idx];
    state
        .creatures
        .iter()
        .find(|c| c.id == uid.0)
        .expect("enemy")
        .current_hp
}

const STRIKE: CardId = CardId::StrikeIronclad;
const DEFEND: CardId = CardId::DefendIronclad;

/// Repeatedly plays the FIRST playable card in hand, paying its cost, until
/// none is playable: 4 energy (3 + the Earring) buys Bash (2), Strike (1),
/// Defend (1); the second Strike / Defend are left in hand.
#[test]
fn plays_first_playable_card_repeatedly_and_pays_cost() {
    let (state, _uids, res) =
        run_turn_one(Setup::new(&[CardId::Bash, STRIKE, DEFEND, STRIKE, DEFEND]));
    assert_eq!(res, StepResult::Advanced);
    assert_eq!(state.phase, CombatPhase::PlayerPlayPhase);
    assert!(state.pending_choice.is_none());
    assert_eq!(state.player.energy, 0, "every auto-play paid its cost");
    assert_eq!(ids(&state, &state.player.hand), vec![STRIKE, DEFEND]);
    assert_eq!(
        ids(&state, &state.player.discard),
        vec![CardId::Bash, STRIKE, DEFEND],
        "played in hand order"
    );
    // Bash 8, then the Strike's 6 under Bash's Vulnerable = 9.
    assert_eq!(hp(&state, 0), 400 - 8 - 9);
    assert_eq!(state.player.block, 5);
    assert_eq!(state.combat_log.energy_spent_this_turn, 4);
}

/// A card the player cannot afford is skipped, not a stop: the first card
/// whose `CanPlay()` holds is taken, wherever it sits in hand.
#[test]
fn skips_unaffordable_cards_for_the_first_playable_one() {
    let (state, _uids, _) = run_turn_one(Setup::new(&[
        CardId::Bash,
        STRIKE,
        CardId::Bash,
        CardId::Bash,
        DEFEND,
    ]));
    // Bash (2) → 2 left, Strike (1) → 1 left; both remaining Bashes are now
    // unaffordable, so the Defend behind them is played.
    assert_eq!(state.player.energy, 0);
    assert_eq!(
        ids(&state, &state.player.hand),
        vec![CardId::Bash, CardId::Bash]
    );
    assert_eq!(
        ids(&state, &state.player.discard),
        vec![CardId::Bash, STRIKE, DEFEND]
    );
}

/// `GetTarget`: AnyEnemy → `HittableEnemies.FirstOrDefault()`, the leftmost
/// one, with no `CombatTargets` read.
#[test]
fn aims_at_the_leftmost_enemy_without_an_rng_read() {
    let mut setup = Setup::new(&[STRIKE, STRIKE, STRIKE, STRIKE, STRIKE]);
    setup.enemy_hps = vec![100, 100];
    let targets_before = {
        let (s, _, _) = {
            let mut probe = Setup::new(&[DEFEND, DEFEND, DEFEND, DEFEND, DEFEND]);
            probe.enemy_hps = vec![100, 100];
            probe.relics = Vec::new();
            run_turn_one(probe)
        };
        s.run_rng.combat_targets.counter()
    };
    let (state, _, _) = run_turn_one(setup);
    assert_eq!(hp(&state, 0), 100 - 4 * 6, "four Strikes, all on the left");
    assert_eq!(hp(&state, 1), 100);
    assert_eq!(
        state.run_rng.combat_targets.counter(),
        targets_before,
        "the leftmost pick reads no combat_targets"
    );
}

/// An X-cost card spends ALL energy through `SpendResources`, and that is
/// its X (`skipXCapture`): Whirlwind first with 4 energy hits 4 times.
#[test]
fn x_cost_card_captures_the_energy_it_spent() {
    let (state, _, _) = run_turn_one(Setup::new(&[
        CardId::Whirlwind,
        STRIKE,
        DEFEND,
        DEFEND,
        DEFEND,
    ]));
    assert_eq!(state.player.energy, 0);
    assert_eq!(hp(&state, 0), 400 - 4 * 5, "Whirlwind X=4");
    assert_eq!(
        ids(&state, &state.player.hand),
        vec![STRIKE, DEFEND, DEFEND, DEFEND]
    );
}

/// The loop is capped at 13 plays (`maxCardsToPlay`).
#[test]
fn stops_after_thirteen_plays() {
    // FlashOfSteel costs 0 and draws one, so the hand never runs dry; only
    // the cap stops the loop.
    let (state, _, _) = run_turn_one(Setup::new(&[CardId::FlashOfSteel; 25]));
    assert_eq!(state.player.discard.len(), 13);
    assert_eq!(state.player.hand.len(), 5);
    assert_eq!(state.player.draw.len(), 25 - 5 - 13);
    assert_eq!(state.player.energy, 4);
    assert_eq!(
        hp(&state, 0),
        400 - 13 * 5,
        "thirteen 5-damage FlashOfSteels"
    );
}

/// It stops as soon as the combat is over: the remaining cards stay in hand.
#[test]
fn stops_when_combat_ends() {
    let mut setup = Setup::new(&[STRIKE, STRIKE, STRIKE, STRIKE, STRIKE]);
    setup.enemy_hps = vec![10];
    let (state, _, res) = run_turn_one(setup);
    assert_eq!(res, StepResult::CombatWon);
    assert!(state.is_over);
    assert_eq!(state.player.hand.len(), 3, "two Strikes killed it");
    assert_eq!(state.player.energy, 2);
}

/// Only the first turn: on turn 2 nothing is auto-played.
#[test]
fn turn_two_does_nothing() {
    let (state, _, _) = run_turn(
        Setup::new(&[STRIKE, STRIKE, STRIKE, DEFEND, DEFEND]),
        Some(2),
    );
    assert_eq!(state.player.hand.len(), 5);
    assert_eq!(state.player.energy, 4);
    assert_eq!(hp(&state, 0), 400);
}

/// A melted Earring does nothing at all (no auto-play, no +1 energy).
#[test]
fn melted_earring_does_nothing() {
    let mut setup = Setup::new(&[STRIKE, STRIKE, STRIKE, DEFEND, DEFEND]);
    setup.relics = vec![earring(true)];
    let (state, _, _) = run_turn_one(setup);
    assert_eq!(state.player.hand.len(), 5);
    assert_eq!(state.player.energy, 3);
}

/// A selection opened by an auto-played card is answered by the
/// `VakuuCardSelector` — `options.Take(maxSelect)`, the first max_select
/// options — inline, without surfacing. Purity is "exhaust up to 3"
/// (min 0 / max 3): Vakuu exhausts the first THREE hand cards (the parity
/// harness's min-prefix drain would exhaust one).
#[test]
fn selection_inside_a_played_card_resolves_vakuu_style() {
    let (state, uids, res) = run_turn_one(Setup::new(&[
        CardId::Purity,
        STRIKE,
        DEFEND,
        STRIKE,
        DEFEND,
    ]));
    assert_eq!(res, StepResult::Advanced);
    assert!(state.pending_choice.is_none(), "nothing surfaced");
    assert_eq!(state.phase, CombatPhase::PlayerPlayPhase);
    // Purity (exhausts itself) + the first three remaining hand cards.
    let exhausted: Vec<CardUid> = state.player.exhaust.iter().copied().collect();
    for u in &uids[1..4] {
        assert!(exhausted.contains(u), "{u:?} exhausted: {exhausted:?}");
    }
    // The last Defend is then auto-played with the energy left.
    assert!(state.player.hand.is_empty());
    assert_eq!(state.player.block, 5);
    assert_eq!(state.player.energy, 3);
}
