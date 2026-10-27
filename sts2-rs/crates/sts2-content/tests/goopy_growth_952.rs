//! Goopy grows on every play (GH #952), through the real
//! `combat_loop::play_card` path.
//!
//! C# `Goopy.AfterCardPlayed` (v0.107.1 `Models.Enchantments/Goopy.cs`):
//!
//! [decompiled C# omitted — see the cited path]
//!
//! and `EnchantBlockAdditive` adds `Amount - 1`. `Hook.AfterCardPlayed` runs
//! once per play iteration (`CardModel.OnPlayWrapper`), so a replayed Goopy
//! card ticks once per play, and each replay's block reads the amount the
//! previous play left.

use smallvec::SmallVec;
use std::num::NonZeroU32;

use sts2_content::cards::dispatch as content_dispatch;
use sts2_core::cards::{Card, CardId, CardTag, CardTags, CardType, EnchantmentId, TargetType};
use sts2_core::{CombatPhase, CombatState, Creature, IntentHistory, MonsterId, MonsterKind};

fn fresh() -> CombatState {
    content_dispatch::ensure_installed();
    let mut state = CombatState::default();
    let id = NonZeroU32::new(1).expect("nz");
    state.creatures.push(Creature {
        id,
        current_hp: 9999,
        max_hp: 9999,
        block: 0,
        powers: SmallVec::new(),
        species: Some(MonsterKind::Nibbit),
        intent: None,
        intent_history: IntentHistory::default(),
        monster_id: MonsterId("Nibbit"),
        ai: None,
    });
    state.enemy_uids.push(sts2_core::CreatureUid(id));
    state.player.current_hp = 80;
    state.player.max_hp = 80;
    state.player.energy = 99;
    state.player.max_energy = 99;
    let _ = state.init_player_creature();
    state.phase = CombatPhase::PlayerPlayPhase;
    state
}

/// A Goopy Defend (5 block, amount `amount`) in hand, replayed `replays`
/// extra times; returns its index into `state.cards`.
fn push_goopy_defend(state: &mut CombatState, amount: i32, replays: i32) -> usize {
    let raw = u32::try_from(state.cards.len() + 1).expect("u32");
    let uid = sts2_core::CardUid::new(raw).expect("nz");
    let mut card = Card {
        uid,
        card_id: CardId::DefendIronclad,
        cost: 0,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        upgraded: false,
        base_damage: 0,
        base_block: 5,
        tags: CardTags::NONE.with(CardTag::Defend),
        base_replay_count: replays,
        ..Card::pod_defaults()
    };
    card.set_enchantment(EnchantmentId::Goopy, amount);
    state.cards.push(card);
    state.player.hand.push(uid);
    state.cards.len() - 1
}

#[test]
fn a_goopy_play_ticks_its_amount_after_the_block_lands() {
    let mut state = fresh();
    let idx = push_goopy_defend(&mut state, 1, 0);
    let _ = sts2_core::combat_loop::play_card(&mut state, 0, None);
    assert_eq!(state.player.block, 5, "amount 1 adds 0 block");
    assert_eq!(state.cards[idx].enchantment(EnchantmentId::Goopy), 2);
}

#[test]
fn each_replay_ticks_goopy_and_reads_the_grown_amount() {
    let mut state = fresh();
    let idx = push_goopy_defend(&mut state, 2, 1);
    let _ = sts2_core::combat_loop::play_card(&mut state, 0, None);
    // play 1: 5 + (2 - 1) = 6, tick to 3; replay: 5 + (3 - 1) = 7, tick to 4.
    assert_eq!(state.player.block, 13);
    assert_eq!(state.cards[idx].enchantment(EnchantmentId::Goopy), 4);
}
