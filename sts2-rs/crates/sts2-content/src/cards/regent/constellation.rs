//! Regent Constellation — v0.111.0-only, MP-only (GH #793).
use super::prelude::*;

/// Constellation (Uncommon, 0 energy + 2 stars, Skill, AnyAlly) — v0.111.0
/// `Cards/Constellation.cs`, `MultiplayerConstraint => MultiplayerOnly`:
///
/// [decompiled C# omitted — see the cited path]
///
/// `AnyAlly` — unplayable in single-player (`card_effects::card_is_any_ally`);
/// in co-op the effect lands on the teammate (`any_ally_card_target`).
pub(crate) fn play_constellation(state: &mut CombatState, _t: Option<CreatureUid>) -> EffectResult {
    // v0.111.0: the TARGET teammate draws 1, gains 1 energy and 9 (upg 12)
    // Block (decomp Constellation.cs:31-38). Their draw and energy are not
    // simulated; their block is, so a later Mimic reads it.
    if let Some(ally) = state.any_ally_card_target() {
        let _ = sts2_core::card_effects::gain_block_ally(state, ally, current_block(state));
    }
    EffectResult::Resolved
}

#[must_use]
pub(crate) const fn spec() -> CardSpec {
    CardSpec {
        id: CardId::Constellation,
        name: "CONSTELLATION",
        cost: 0,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Uncommon,
        base_damage: 0,
        base_block: 9,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[allow(clippy::unnecessary_wraps, dead_code)]
pub(crate) fn upgrade() -> Option<CardUpgradeDelta> {
    // Block 9 -> 12 — the row lives in `card_upgrades_generated`.
    None
}
