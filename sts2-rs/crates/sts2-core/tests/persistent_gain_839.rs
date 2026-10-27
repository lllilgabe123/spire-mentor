//! GH #839 — pricing of gains that outlast the fight
//! (`sts2_core::persistent_gain_hp_equiv`).

use sts2_core::cards::{Card, CardId};
use sts2_core::{
    persistent_combats_left, persistent_gain_hp_equiv, CombatState, PersistentBaseline,
    LOOT_GOLD_HP_EQUIV, PERSISTENT_GROWTH_HP_EQUIV, PERSISTENT_MAX_HP_HP_EQUIV,
    PERSISTENT_SCALER_PLAYS_PER_FIGHT,
};

fn card(id: CardId, base_damage: i32, base_block: i32) -> Card {
    let mut c = Card::pod_defaults();
    c.card_id = id;
    c.base_damage = base_damage;
    c.base_block = base_block;
    c
}

/// Act 1 hallway fight: 80 max HP, 100 gold, a fresh GA and a Strike.
fn root() -> CombatState {
    let mut s = CombatState {
        act_index: 1,
        is_boss: false,
        gold: 100,
        cards: vec![
            card(CardId::GeneticAlgorithm, 0, 1),
            card(CardId::StrikeIronclad, 6, 0),
        ]
        .into(),
        ..CombatState::default()
    };
    s.player.max_hp = 80;
    s
}

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-4
}

#[test]
fn the_root_is_worth_nothing() {
    let s = root();
    assert!(persistent_gain_hp_equiv(&s, &PersistentBaseline::of(&s)).abs() < 1e-6);
}

#[test]
fn combats_left_counts_down_to_zero_at_the_final_boss() {
    let mut s = root();
    assert!(close(persistent_combats_left(&s), 22.5));
    s.is_boss = true;
    assert!(close(persistent_combats_left(&s), 18.0));
    s.act_index = 3;
    assert!(persistent_combats_left(&s).abs() < 1e-6);
}

#[test]
fn genetic_algorithm_growth_is_priced_per_future_play() {
    let s0 = root();
    let base = PersistentBaseline::of(&s0);
    let mut s = s0.clone();
    s.cards[0].base_block = 4; // one play: 1 -> 4
    let want = 3.0 * PERSISTENT_GROWTH_HP_EQUIV * PERSISTENT_SCALER_PLAYS_PER_FIGHT * 22.5;
    assert!(close(persistent_gain_hp_equiv(&s, &base), want));
}

/// Growth on a non-scaler card (a Claw-style in-fight buff) is not a
/// persistent gain: it resets when the fight ends.
#[test]
fn non_scaler_growth_is_ignored() {
    let s0 = root();
    let base = PersistentBaseline::of(&s0);
    let mut s = s0.clone();
    s.cards[1].base_damage = 20;
    assert!(persistent_gain_hp_equiv(&s, &base).abs() < 1e-6);
}

#[test]
fn max_hp_and_gold_shrink_with_the_run_left() {
    let s0 = root();
    let base = PersistentBaseline::of(&s0);
    let mut s = s0.clone();
    s.player.max_hp += 3; // Feed
    s.gold += 20; // Hand of Greed
    let run_left = 22.5 / 27.0;
    let want = (3.0 * PERSISTENT_MAX_HP_HP_EQUIV + 20.0 * LOOT_GOLD_HP_EQUIV) * run_left;
    assert!(close(persistent_gain_hp_equiv(&s, &base), want));

    // Debt / Brightest Flame: losses count against.
    let mut lost = s0.clone();
    lost.gold -= 10;
    assert!(persistent_gain_hp_equiv(&lost, &base) < 0.0);
}

/// Gold a thief takes lowers `state.gold`, but the loot penalty already
/// charges it; the persistent term must not see it again.
#[test]
fn stolen_gold_is_left_to_the_loot_penalty() {
    let s0 = root();
    let base = PersistentBaseline::of(&s0);
    let mut s = s0.clone();
    s.gold -= 15;
    s.heist_recovered_gold = 15; // thief killed, gold coming back at exit
    assert!(persistent_gain_hp_equiv(&s, &base).abs() < 1e-6);
}
