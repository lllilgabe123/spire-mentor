//! Defect Imitation Learning — v0.111.0-only, MP-only (GH #793).
use super::prelude::*;

/// Imitation Learning (Rare, 1 energy, Skill, Exhaust, AnyAlly) — v0.111.0
/// `Cards/ImitationLearning.cs`, `MultiplayerConstraint => MultiplayerOnly`:
///
/// [decompiled C# omitted — see the cited path]
///
/// `AnyAlly` — unplayable in single-player (`card_effects::card_is_any_ally`).
pub(crate) fn play_imitation_learning(
    state: &mut CombatState,
    _t: Option<CreatureUid>,
) -> EffectResult {
    // The power sits on the PLAYER, bound to the target teammate; it copies
    // the teammate's Power cards, which the simulation never plays, so only
    // its amount is observable. One teammate is ever targeted
    // (`any_ally_card_target`), so a second play raises the one instance
    // (C# `PowerCmd.ModifyAmount`) instead of adding another.
    if state.any_ally_card_target().is_none() {
        return EffectResult::Resolved;
    }
    let amt = if current_card_upgraded(state) { 3 } else { 2 };
    let existing = state.player_power_uids().into_iter().find(|p| {
        state
            .powers
            .get_by_id(p.0)
            .is_some_and(|p| p.kind == PowerKind::ImitationLearning)
    });
    match existing {
        Some(uid) => {
            if let Some(p) = state.powers.get_mut_by_id(uid.0) {
                p.amount += amt;
            }
        }
        None => apply_power_to_player(state, PowerKind::ImitationLearning, amt),
    }
    EffectResult::Resolved
}

#[must_use]
pub(crate) const fn spec() -> CardSpec {
    CardSpec {
        id: CardId::ImitationLearningCard,
        name: "IMITATION_LEARNING_CARD",
        cost: 1,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Rare,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::EXHAUST,
        ..CardSpec::base()
    }
}

#[allow(clippy::unnecessary_wraps, dead_code)]
pub(crate) fn upgrade() -> Option<CardUpgradeDelta> {
    None
}
