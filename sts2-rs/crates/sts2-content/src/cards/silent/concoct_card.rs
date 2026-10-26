//! Silent Concoct — v0.111.0-only, MP-only (GH #793).
use super::prelude::*;

/// Concoct (Uncommon, 0 energy, Skill, AnyAlly) — v0.111.0 `Cards/Concoct.cs`,
/// `MultiplayerConstraint => MultiplayerOnly`:
///
/// [decompiled C# omitted — see the cited path]
///
/// `AnyAlly`: C# `CardModel.CanPlay` (`CardModel.cs:1742`) makes it
/// unplayable without a living ally, which single-player never has
/// (`card_effects::card_is_any_ally`), so this handler does not run in SP.
pub(crate) fn play_concoct(_state: &mut CombatState, _t: Option<CreatureUid>) -> EffectResult {
    // C# `PowerCmd.Apply<ConcoctPower>(cardPlay.Target, 3 (upg 4))` — all of it lands on the co-op teammate
    // (`cardPlay.Target`), whose turn is not simulated, so playing it
    // changes nothing here but the energy and the card's own pile move.
    // Unplayable without a living teammate (`card_is_any_ally`).
    EffectResult::Resolved
}

#[must_use]
pub(crate) const fn spec() -> CardSpec {
    CardSpec {
        id: CardId::ConcoctCard,
        name: "CONCOCT_CARD",
        cost: 0,
        card_type: CardType::Skill,
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
    None
}
