//! Necrobinder Soulbound — v0.111.0-only, MP-only (GH #793).
use super::prelude::*;

/// Soulbound (Uncommon, 1 energy, Power, AnyAlly) — v0.111.0
/// `Cards/Soulbound.cs`, `MultiplayerConstraint => MultiplayerOnly`:
///
/// [decompiled C# omitted — see the cited path]
///
/// `AnyAlly` — unplayable in single-player (`card_effects::card_is_any_ally`).
pub(crate) fn play_soulbound(_state: &mut CombatState, _t: Option<CreatureUid>) -> EffectResult {
    // C# `PowerCmd.Apply<SoulboundPower>(cardPlay.Target, 1)` — all of it lands on the co-op teammate
    // (`cardPlay.Target`), whose turn is not simulated, so playing it
    // changes nothing here but the energy and the card's own pile move.
    // Unplayable without a living teammate (`card_is_any_ally`).
    EffectResult::Resolved
}

#[must_use]
pub(crate) const fn spec() -> CardSpec {
    CardSpec {
        id: CardId::SoulboundCard,
        name: "SOULBOUND_CARD",
        cost: 1,
        card_type: CardType::Power,
        target_type: TargetType::Self_,
        rarity: CardRarity::Uncommon,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[allow(clippy::unnecessary_wraps, dead_code)]
pub(crate) fn upgrade() -> Option<CardUpgradeDelta> {
    // `AddKeyword(Innate)` — the row lives in `card_upgrades_generated`.
    None
}
