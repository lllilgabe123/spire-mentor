//! Silent Fade — v0.111.0-only, MP-only (GH #793).
use super::prelude::*;

/// Fade (Uncommon, 0 energy, Skill, Retain, AnyAlly) — v0.111.0
/// `Cards/Fade.cs`, `MultiplayerConstraint => MultiplayerOnly`:
///
/// [decompiled C# omitted — see the cited path]
///
/// `AnyAlly` — unplayable in single-player (`card_effects::card_is_any_ally`).
pub(crate) fn play_fade(_state: &mut CombatState, _t: Option<CreatureUid>) -> EffectResult {
    // C# `PowerCmd.Apply<FadePower>(cardPlay.Target, 6 (upg 9))` — all of it lands on the co-op teammate
    // (`cardPlay.Target`), whose turn is not simulated, so playing it
    // changes nothing here but the energy and the card's own pile move.
    // Unplayable without a living teammate (`card_is_any_ally`).
    EffectResult::Resolved
}

#[must_use]
pub(crate) const fn spec() -> CardSpec {
    CardSpec {
        id: CardId::FadeCard,
        name: "FADE_CARD",
        cost: 0,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Uncommon,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::RETAIN,
        ..CardSpec::base()
    }
}

#[allow(clippy::unnecessary_wraps, dead_code)]
pub(crate) fn upgrade() -> Option<CardUpgradeDelta> {
    None
}
