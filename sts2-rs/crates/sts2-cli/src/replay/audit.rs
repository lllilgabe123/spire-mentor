//! Replay red-flag auditor.
//!
//! Scans recorded [`FightRecord`]s for action patterns that suggest
//! engine/agent bugs rather than legitimate play. The detectors here
//! are intentionally cheap and conservative — every flag we count
//! should either be a real bug or rare enough that a small false-
//! positive rate doesn't drown the signal across a benchmark corpus.
//!
//! Current detectors (see [`RedFlagKind`]):
//!
//! * `EndTurnWithPlayable` — chose [`Action::EndTurn`] while at least
//!   one hand card was nominally affordable and not UNPLAYABLE.
//! * `ConsecutiveSkippedTurns` — ≥2 consecutive player turns whose
//!   only action was an `EndTurn` flagged by the above (the longer
//!   streak emits one flag per turn beyond the first).
//! * `ZeroPlayCombat` — the entire combat had zero `PlayCard`
//!   decisions (player skipped every turn).
//! * `SuicidePlay` — a `PlayCard` decision brought the player to
//!   `hp ≤ 0` while ≥1 enemy was still alive (final-decision-only).
//! * `LethalIntentIgnored` — chose `EndTurn` while telegraphed
//!   incoming damage exceeded `player_hp + player_block` and an
//!   affordable card was in hand. Strict subset of
//!   `EndTurnWithPlayable`; a high-confidence "MCTS didn't see the
//!   lethal" signal.
//! * `OverblockWithoutAttackIntent` — gained block above a
//!   conservative threshold on a turn where no alive enemy had an
//!   Attack/MultiAttack intent. Cheap proxy for "the agent is
//!   defending against nothing".
//! * `PotionHoardedOnDeath` — fight ended in defeat with ≥1 potion
//!   still held. Doesn't prove the potion would have saved the run,
//!   but a rising count between benchmark runs implies the agent is
//!   under-using potions.
//! * `TransientPowerLeak` — a power whose Python pin pops/decrements
//!   it via `after_turn_end` is observed at the start of two
//!   consecutive player turns with a non-decreasing stack on the same
//!   owner. Regression guard for the bug class fixed by commit
//!   `093d420` (OneTwoPunch / Rage / FlameBarrier).
//!
//! Run via the `sts2-replay-audit` binary after a benchmark or any
//! `sts2-fight-replay` batch.

use std::fmt;

use sts2_core::card_effects::card_unplayable;
use sts2_core::cards::{CardId, CardType};
use sts2_core::monsters::Intent;
use sts2_core::{Action, PowerKind};

use crate::replay::record::{CardInHand, DecisionRecord, FightRecord, StateSnapshot};

/// A single suspicious pattern detected in one fight.
#[derive(Debug, Clone)]
pub struct RedFlag {
    pub kind: RedFlagKind,
    pub turn: u16,
    /// Index into [`FightRecord::decisions`] (the first/representative
    /// decision when the flag spans multiple).
    pub decision_index: u32,
    pub detail: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RedFlagKind {
    EndTurnWithPlayable,
    ConsecutiveSkippedTurns,
    ZeroPlayCombat,
    SuicidePlay,
    LethalIntentIgnored,
    OverblockWithoutAttackIntent,
    PotionHoardedOnDeath,
    TransientPowerLeak,
    DefendNoAttackIntent,
    EndTurnWithHighEnergy,
    StrikeWhenBashAffordable,
}

impl RedFlagKind {
    pub const ALL: [Self; 11] = [
        Self::EndTurnWithPlayable,
        Self::ConsecutiveSkippedTurns,
        Self::ZeroPlayCombat,
        Self::SuicidePlay,
        Self::LethalIntentIgnored,
        Self::OverblockWithoutAttackIntent,
        Self::PotionHoardedOnDeath,
        Self::TransientPowerLeak,
        Self::DefendNoAttackIntent,
        Self::EndTurnWithHighEnergy,
        Self::StrikeWhenBashAffordable,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::EndTurnWithPlayable => "end_turn_with_playable",
            Self::ConsecutiveSkippedTurns => "consecutive_skipped_turns",
            Self::ZeroPlayCombat => "zero_play_combat",
            Self::SuicidePlay => "suicide_play",
            Self::LethalIntentIgnored => "lethal_intent_ignored",
            Self::OverblockWithoutAttackIntent => "overblock_without_attack_intent",
            Self::PotionHoardedOnDeath => "potion_hoarded_on_death",
            Self::TransientPowerLeak => "transient_power_leak",
            Self::DefendNoAttackIntent => "defend_no_attack_intent",
            Self::EndTurnWithHighEnergy => "end_turn_with_high_energy",
            Self::StrikeWhenBashAffordable => "strike_when_bash_affordable",
        }
    }
}

/// Powers that defang the `DefendNoAttackIntent` heuristic — Defending
/// without an incoming Attack is still rational if the block carries
/// over (`Barricade`) or feeds a damage payoff (`Juggernaut`,
/// `Metallicize`). Other guards from the bead body (`Calipers`,
/// `Entrench`, `Plated`) don't exist as `PowerKind` variants at this
/// pin and are omitted.
const BLOCK_PAYOFF_POWERS: &[PowerKind] = &[
    PowerKind::Barricade,
    PowerKind::Juggernaut,
    PowerKind::Metallicize,
];

/// Energy threshold for `EndTurnWithHighEnergy`. `≥ 3` matches the
/// bead spec: a full starting hand at base energy that decides to
/// end-turn is unambiguously bad (couldn't be a single-card-can't-pay
/// tail).
const HIGH_ENERGY_THRESHOLD: i8 = 3;

/// Strike base damage for the `StrikeWhenBashAffordable` overkill
/// guard. 6 = unupgraded; 9 = upgraded. Matches the Python pin for
/// all class-base `Strike*` variants (Ironclad/Silent/Defect/Regent/
/// Necrobinder).
const STRIKE_BASE_DAMAGE: i32 = 6;
const STRIKE_PLUS_DAMAGE: i32 = 9;

/// Power kinds whose Python pin pops or decrements them via
/// `after_turn_end`. If one of these is present on the same owner at
/// the start of two consecutive player turns with a non-decreasing
/// stack, the engine almost certainly forgot to pop it — the bug class
/// behind commit `093d420`.
///
/// v1 keeps this list to the three powers actually fixed in that
/// commit. Candidates from the bead body (Vigor, NoBlock, NoDraw,
/// DoubleTap, BurstPower, Phantasmal, Wraith Form) need a pin sweep
/// before being added: some of those decay rather than fully pop, and
/// flagging "stack unchanged" on a slow-decay power produces noise.
const TRANSIENT_POWERS: &[PowerKind] = &[
    PowerKind::OneTwoPunch,
    PowerKind::Rage,
    PowerKind::FlameBarrier,
];

/// Block-stack threshold above which `OverblockWithoutAttackIntent`
/// fires (on a turn with no Attack/MultiAttack intent telegraphed).
/// 30 is comfortably above Ironclad's typical defensive curve at low
/// floors — a tighter threshold flags too many legitimate Barricade /
/// Body Slam setups.
const OVERBLOCK_THRESHOLD: i32 = 30;

impl fmt::Display for RedFlagKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Aggregate per-flag counts across many fights.
#[derive(Debug, Default, Clone)]
pub struct AuditSummary {
    pub fights_audited: u32,
    pub fights_with_any_flag: u32,
    pub counts: [u32; RedFlagKind::ALL.len()],
}

impl AuditSummary {
    pub fn record(&mut self, per_fight: &[RedFlag]) {
        self.fights_audited += 1;
        if !per_fight.is_empty() {
            self.fights_with_any_flag += 1;
        }
        for flag in per_fight {
            let idx = RedFlagKind::ALL
                .iter()
                .position(|k| *k == flag.kind)
                .expect("kind always in ALL");
            self.counts[idx] += 1;
        }
    }

    pub fn count(&self, kind: RedFlagKind) -> u32 {
        let idx = RedFlagKind::ALL
            .iter()
            .position(|k| *k == kind)
            .expect("kind always in ALL");
        self.counts[idx]
    }
}

/// Scan a single fight and return every red flag detected.
pub fn audit_fight(record: &FightRecord) -> Vec<RedFlag> {
    let mut flags: Vec<RedFlag> = Vec::new();

    // Per-decision passes.
    let mut play_card_decisions: u32 = 0;
    let last_idx = record.decisions.len().saturating_sub(1);
    for (i, dec) in record.decisions.iter().enumerate() {
        let Some(action) = decision_action(dec) else {
            continue;
        };
        match action {
            Action::PlayCard { hand_index, target } => {
                play_card_decisions += 1;
                if i == last_idx && !record.final_won && killed_self(dec, record) {
                    flags.push(RedFlag {
                        kind: RedFlagKind::SuicidePlay,
                        turn: dec.pre.turn,
                        decision_index: dec.decision_index,
                        detail: format!(
                            "ended at hp={} after `{}`; enemies alive at decision: {}",
                            dec.player_hp_after,
                            dec.recommended_label,
                            alive_enemy_count(&dec.pre),
                        ),
                    });
                }
                let Some(played) = dec.pre.hand.iter().find(|c| c.hand_index == *hand_index) else {
                    continue;
                };
                if is_defend_card(played.card_id)
                    && !any_attack_intent(&dec.pre)
                    && alive_enemy_count(&dec.pre) > 0
                    && !has_block_payoff_power(&dec.pre.player_powers)
                {
                    flags.push(RedFlag {
                        kind: RedFlagKind::DefendNoAttackIntent,
                        turn: dec.pre.turn,
                        decision_index: dec.decision_index,
                        detail: format!(
                            "rec=`{}`, enemy_intents={}",
                            dec.recommended_label,
                            describe_enemy_intents(&dec.pre),
                        ),
                    });
                }
                if is_basic_strike_card(played.card_id) {
                    if let Some(flag) = detect_strike_when_bash(dec, played, *target) {
                        flags.push(flag);
                    }
                }
            }
            Action::EndTurn if has_affordable_playable(&dec.pre) => {
                flags.push(RedFlag {
                    kind: RedFlagKind::EndTurnWithPlayable,
                    turn: dec.pre.turn,
                    decision_index: dec.decision_index,
                    detail: format!(
                        "energy={}, affordable_hand={}",
                        dec.pre.player_energy,
                        describe_affordable_hand(&dec.pre),
                    ),
                });
                if dec.pre.player_energy >= HIGH_ENERGY_THRESHOLD
                    && has_affordable_non_power(&dec.pre)
                {
                    flags.push(RedFlag {
                        kind: RedFlagKind::EndTurnWithHighEnergy,
                        turn: dec.pre.turn,
                        decision_index: dec.decision_index,
                        detail: format!(
                            "energy={}, affordable_playable_cards={}",
                            dec.pre.player_energy,
                            describe_affordable_non_power_hand(&dec.pre),
                        ),
                    });
                }
                let incoming = incoming_attack_damage(&dec.pre);
                let survivable = dec.pre.player_block + dec.pre.player_hp;
                if incoming > survivable {
                    flags.push(RedFlag {
                        kind: RedFlagKind::LethalIntentIgnored,
                        turn: dec.pre.turn,
                        decision_index: dec.decision_index,
                        detail: format!(
                            "incoming={incoming} > hp({}) + block({}); affordable_hand={}",
                            dec.pre.player_hp,
                            dec.pre.player_block,
                            describe_affordable_hand(&dec.pre),
                        ),
                    });
                }
            }
            _ => {}
        }

        // Overblock check applies to any decision phase regardless of
        // action: we're flagging the state itself ("block stacked
        // without an incoming attack"), not the action.
        if dec.pre.player_block >= OVERBLOCK_THRESHOLD
            && !any_attack_intent(&dec.pre)
            && alive_enemy_count(&dec.pre) > 0
        {
            flags.push(RedFlag {
                kind: RedFlagKind::OverblockWithoutAttackIntent,
                turn: dec.pre.turn,
                decision_index: dec.decision_index,
                detail: format!(
                    "block={} ≥ {OVERBLOCK_THRESHOLD}; no Attack/MultiAttack intent telegraphed",
                    dec.pre.player_block,
                ),
            });
        }
    }

    // Consecutive skipped turns: a turn is "skipped" iff its only
    // decision was an `EndTurn` flagged by `EndTurnWithPlayable`.
    // Walk decisions grouped by `pre.turn`. Streaks of ≥2 such turns
    // emit one `ConsecutiveSkippedTurns` flag per turn past the first.
    let skipped_turns = collect_skipped_turns(&record.decisions);
    if skipped_turns.len() >= 2 {
        for window in skipped_turns.windows(2) {
            let (prev_turn, _) = window[0];
            let (cur_turn, cur_dec_idx) = window[1];
            if cur_turn == prev_turn.saturating_add(1) {
                flags.push(RedFlag {
                    kind: RedFlagKind::ConsecutiveSkippedTurns,
                    turn: cur_turn,
                    decision_index: cur_dec_idx,
                    detail: format!("skipped turns {prev_turn} and {cur_turn} in a row"),
                });
            }
        }
    }

    if !record.final_won {
        if let Some(last) = record.decisions.last() {
            if last.pre.potions_held > 0 {
                flags.push(RedFlag {
                    kind: RedFlagKind::PotionHoardedOnDeath,
                    turn: last.pre.turn,
                    decision_index: last.decision_index,
                    detail: format!(
                        "{} potion(s) held on final decision; final hp={}",
                        last.pre.potions_held, record.final_player_hp,
                    ),
                });
            }
        }
    }

    detect_transient_power_leaks(&record.decisions, &mut flags);

    if play_card_decisions == 0 && !record.decisions.is_empty() {
        let first = &record.decisions[0];
        flags.push(RedFlag {
            kind: RedFlagKind::ZeroPlayCombat,
            turn: first.pre.turn,
            decision_index: first.decision_index,
            detail: format!(
                "{} decisions, zero PlayCard; combat {}",
                record.decisions.len(),
                if record.final_won { "won" } else { "lost" },
            ),
        });
    }

    flags
}

fn decision_action(dec: &DecisionRecord) -> Option<&Action> {
    dec.recommended_idx
        .and_then(|i| dec.root_outcomes.get(i))
        .map(|row| &row.action)
}

fn has_affordable_playable(snap: &StateSnapshot) -> bool {
    snap.hand.iter().any(|c| is_affordable_playable(c, snap))
}

fn is_affordable_playable(card: &CardInHand, snap: &StateSnapshot) -> bool {
    if card_unplayable(card.card_id) {
        return false;
    }
    // Cost -1 = X-cost (always playable, consumes all energy ≥ 0).
    // Cost -2 = unplayable (already filtered above for registered cards;
    // catches the unregistered case too).
    if card.cost < -1 {
        return false;
    }
    if card.cost == -1 {
        return snap.player_energy >= 0;
    }
    i32::from(card.cost) <= i32::from(snap.player_energy)
}

fn describe_affordable_hand(snap: &StateSnapshot) -> String {
    let names: Vec<String> = snap
        .hand
        .iter()
        .filter(|c| is_affordable_playable(c, snap))
        .map(|c| format!("h{}(cost {})", c.hand_index, c.cost))
        .collect();
    if names.is_empty() {
        "none".to_string()
    } else {
        names.join(", ")
    }
}

fn alive_enemy_count(snap: &StateSnapshot) -> usize {
    snap.monsters.iter().filter(|m| m.alive).count()
}

/// Sum of telegraphed `Attack` / `MultiAttack` damage across alive
/// enemies. Ignores player Vulnerable / monster Strength — the
/// detector is intentionally a lower bound, not a damage simulator.
fn incoming_attack_damage(snap: &StateSnapshot) -> i32 {
    snap.monsters
        .iter()
        .filter(|m| m.alive)
        .filter_map(|m| m.intent.as_ref())
        .map(|intent| match *intent {
            Intent::Attack { damage } => damage.max(0),
            Intent::MultiAttack { damage, hits } => damage.max(0) * i32::from(hits),
            _ => 0,
        })
        .sum()
}

fn any_attack_intent(snap: &StateSnapshot) -> bool {
    snap.monsters
        .iter()
        .filter(|m| m.alive)
        .filter_map(|m| m.intent.as_ref())
        .any(|intent| matches!(intent, Intent::Attack { .. } | Intent::MultiAttack { .. }))
}

fn killed_self(dec: &DecisionRecord, record: &FightRecord) -> bool {
    dec.player_hp_after <= 0 && record.final_player_hp <= 0 && alive_enemy_count(&dec.pre) > 0
}

/// Returns `(turn, first_decision_index_in_turn)` pairs for every
/// turn whose sole decision was a flagged `EndTurnWithPlayable`.
fn collect_skipped_turns(decisions: &[DecisionRecord]) -> Vec<(u16, u32)> {
    let mut out: Vec<(u16, u32)> = Vec::new();
    let mut i = 0;
    while i < decisions.len() {
        let turn = decisions[i].pre.turn;
        let mut j = i;
        while j < decisions.len() && decisions[j].pre.turn == turn {
            j += 1;
        }
        let span = &decisions[i..j];
        if span.len() == 1 {
            let dec = &span[0];
            if matches!(decision_action(dec), Some(Action::EndTurn))
                && has_affordable_playable(&dec.pre)
            {
                out.push((turn, dec.decision_index));
            }
        }
        i = j;
    }
    out
}

/// Scan for transient powers that persisted across a turn boundary.
///
/// v1 (per bead `sts2-rs-7l8li`): for each pair of consecutive player
/// turns, look at the first decision of each turn ("turn-start"
/// snapshot, before the player has played any card on that turn). If a
/// power in [`TRANSIENT_POWERS`] is present on the same owner at both
/// snapshots with `cur_stack >= prev_stack`, flag it. False positives
/// from same-turn re-application are accepted at this resolution — the
/// detector is a regression guard, not a precise oracle.
fn detect_transient_power_leaks(decisions: &[DecisionRecord], flags: &mut Vec<RedFlag>) {
    let turn_starts: Vec<&DecisionRecord> = first_decisions_per_turn(decisions);
    for window in turn_starts.windows(2) {
        let prev = window[0];
        let cur = window[1];
        if cur.pre.turn != prev.pre.turn.saturating_add(1) {
            continue;
        }
        // Player-side leaks.
        for (kind, prev_stack) in &prev.pre.player_powers {
            if !TRANSIENT_POWERS.contains(kind) || *prev_stack <= 0 {
                continue;
            }
            if let Some(cur_stack) = lookup_power(&cur.pre.player_powers, *kind) {
                if cur_stack >= *prev_stack {
                    flags.push(RedFlag {
                        kind: RedFlagKind::TransientPowerLeak,
                        turn: cur.pre.turn,
                        decision_index: cur.decision_index,
                        detail: format!(
                            "({kind:?}, player, stack={cur_stack}, turns_persisted={}→{})",
                            prev.pre.turn, cur.pre.turn,
                        ),
                    });
                }
            }
        }
        // Monster-side leaks. Match by slot so re-spawned creatures
        // don't get confused with their predecessors.
        for prev_m in &prev.pre.monsters {
            let Some(cur_m) = cur.pre.monsters.iter().find(|m| m.slot == prev_m.slot) else {
                continue;
            };
            for (kind, prev_stack) in &prev_m.powers {
                if !TRANSIENT_POWERS.contains(kind) || *prev_stack <= 0 {
                    continue;
                }
                if let Some(cur_stack) = lookup_power(&cur_m.powers, *kind) {
                    if cur_stack >= *prev_stack {
                        flags.push(RedFlag {
                            kind: RedFlagKind::TransientPowerLeak,
                            turn: cur.pre.turn,
                            decision_index: cur.decision_index,
                            detail: format!(
                                "({kind:?}, monster[{}], stack={cur_stack}, turns_persisted={}→{})",
                                prev_m.slot, prev.pre.turn, cur.pre.turn,
                            ),
                        });
                    }
                }
            }
        }
    }
}

fn first_decisions_per_turn(decisions: &[DecisionRecord]) -> Vec<&DecisionRecord> {
    let mut out: Vec<&DecisionRecord> = Vec::new();
    for dec in decisions {
        if out.last().is_none_or(|prev| prev.pre.turn != dec.pre.turn) {
            out.push(dec);
        }
    }
    out
}

fn lookup_power(powers: &[(PowerKind, i32)], kind: PowerKind) -> Option<i32> {
    powers.iter().find_map(|(k, s)| (*k == kind).then_some(*s))
}

/// Matches the 5 class-base `Defend*` variants (Ironclad / Silent /
/// Defect / Regent / Necrobinder). Other "Defend"-named cards don't
/// exist at the current Python pin.
fn is_defend_card(card_id: CardId) -> bool {
    matches!(
        card_id,
        CardId::DefendIronclad
            | CardId::DefendSilent
            | CardId::DefendDefect
            | CardId::DefendRegent
            | CardId::DefendNecrobinder
    )
}

/// Matches the 5 class-base `Strike*` variants. Special Strikes
/// (PerfectedStrike / PommelStrike / TwinStrike / LeadingStrike / etc.)
/// have different damage/scaling tradeoffs and are NOT clean swaps
/// against Bash — the bead spec is intentionally limited to basic
/// Strikes.
fn is_basic_strike_card(card_id: CardId) -> bool {
    matches!(
        card_id,
        CardId::StrikeIronclad
            | CardId::StrikeSilent
            | CardId::StrikeDefect
            | CardId::StrikeRegent
            | CardId::StrikeNecrobinder
    )
}

fn has_block_payoff_power(powers: &[(PowerKind, i32)]) -> bool {
    powers
        .iter()
        .any(|(k, s)| *s > 0 && BLOCK_PAYOFF_POWERS.contains(k))
}

/// `EndTurnWithHighEnergy` excludes Power-type cards because keeping a
/// Power un-played at high energy can be a legitimate scaling-curve
/// decision (e.g., DemonForm at low HP). All other affordable
/// non-Power cards are fair game.
fn is_affordable_non_power(card: &CardInHand, snap: &StateSnapshot) -> bool {
    is_affordable_playable(card, snap) && card.card_type != CardType::Power
}

fn has_affordable_non_power(snap: &StateSnapshot) -> bool {
    snap.hand.iter().any(|c| is_affordable_non_power(c, snap))
}

fn describe_affordable_non_power_hand(snap: &StateSnapshot) -> String {
    let names: Vec<String> = snap
        .hand
        .iter()
        .filter(|c| is_affordable_non_power(c, snap))
        .map(|c| format!("h{}(cost {})", c.hand_index, c.cost))
        .collect();
    if names.is_empty() {
        "none".to_string()
    } else {
        names.join(", ")
    }
}

fn describe_enemy_intents(snap: &StateSnapshot) -> String {
    let parts: Vec<String> = snap
        .monsters
        .iter()
        .filter(|m| m.alive)
        .map(|m| {
            m.intent.as_ref().map_or_else(
                || format!("slot{}:none", m.slot),
                |i| format!("slot{}:{:?}", m.slot, i),
            )
        })
        .collect();
    if parts.is_empty() {
        "no alive enemies".to_string()
    } else {
        parts.join(", ")
    }
}

/// `StrikeWhenBashAffordable` body. Returns a flag iff the swap to
/// Bash dominates: Bash is in hand and affordable, the target isn't
/// 1-shot dead to Strike (else Bash's Vulnerable is wasted), and the
/// remaining-energy budget after the swap still covers any other
/// affordable non-Strike card in hand.
fn detect_strike_when_bash(
    dec: &DecisionRecord,
    played_strike: &CardInHand,
    target: Option<u8>,
) -> Option<RedFlag> {
    let snap = &dec.pre;
    let bash = snap
        .hand
        .iter()
        .find(|c| c.card_id == CardId::Bash && is_affordable_playable(c, snap))?;
    let energy_after_bash = i32::from(snap.player_energy) - i32::from(bash.cost.max(0));
    if energy_after_bash < 0 {
        return None;
    }
    let blocks_other_play = snap.hand.iter().any(|c| {
        c.hand_index != played_strike.hand_index
            && c.hand_index != bash.hand_index
            && is_affordable_playable(c, snap)
            && i32::from(c.cost.max(0)) > energy_after_bash
    });
    if blocks_other_play {
        return None;
    }
    let strike_dmg = if played_strike.upgraded {
        STRIKE_PLUS_DAMAGE
    } else {
        STRIKE_BASE_DAMAGE
    };
    let target_info = target.and_then(|slot| {
        snap.monsters
            .iter()
            .find(|m| m.slot == usize::from(slot) && m.alive)
    });
    let t = target_info?;
    let effective_hp = t.hp + t.block;
    if effective_hp <= strike_dmg {
        return None;
    }
    Some(RedFlag {
        kind: RedFlagKind::StrikeWhenBashAffordable,
        turn: dec.pre.turn,
        decision_index: dec.decision_index,
        detail: format!(
            "(target_slot={}, target_hp={}, target_block={}, rec=`{}`)",
            t.slot, t.hp, t.block, dec.recommended_label,
        ),
    })
}

/// Format a one-line summary suitable for stdout / CI logs.
pub fn format_summary(summary: &AuditSummary) -> String {
    let mut parts = vec![format!(
        "audited={}, flagged_fights={}",
        summary.fights_audited, summary.fights_with_any_flag,
    )];
    for kind in RedFlagKind::ALL {
        parts.push(format!("{kind}={}", summary.count(kind)));
    }
    parts.join(" | ")
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::path::PathBuf;

    use sts2_core::CombatPhase;

    use super::*;
    use crate::replay::record::{
        DecisionRecord, FightRecord, MonsterSnapshot, RecorderConfig, RootOutcomeRow, StateSnapshot,
    };

    /// `sts2-rs-dhrxw`: `format_summary` produces ` | `-separated
    /// `key=value` tokens consumed by downstream CI log parsers. A
    /// regression that changed the delimiter, dropped a token, or
    /// reordered `RedFlagKind::ALL` would silently break log parsing.
    #[test]
    fn format_summary_default_shape() {
        let s = AuditSummary::default();
        let line = format_summary(&s);

        // Leading prefix.
        assert!(
            line.starts_with("audited=0, flagged_fights=0"),
            "format_summary default missing leading 'audited=0, flagged_fights=0' (got: {line})",
        );

        // Tokens are ` | `-separated; total token count = 2 prefix
        // tokens (audited, flagged_fights bundled as one) +
        // ALL.len() flag tokens = 1 + 11 = 12.
        let tokens: Vec<&str> = line.split(" | ").collect();
        assert_eq!(
            tokens.len(),
            1 + RedFlagKind::ALL.len(),
            "expected 1 prefix + {} flag tokens, got {}: {tokens:?}",
            RedFlagKind::ALL.len(),
            tokens.len(),
        );

        // Every RedFlagKind appears as `<kind>=0` since this is a
        // default (un-recorded) summary.
        for kind in RedFlagKind::ALL {
            let expected = format!("{}=0", kind.as_str());
            assert!(
                line.contains(&expected),
                "format_summary default missing token `{expected}` (got: {line})",
            );
        }
    }

    /// `sts2-rs-qh91i`: pin `AuditSummary::default()` zero-initializes
    /// every counter — `fights_audited`, `fights_with_any_flag`, and
    /// `counts[kind]` for every kind in `RedFlagKind::ALL`. A regression
    /// that initialized a counter to non-zero would let the
    /// `format_summary` post-record pin (q2yhd) still pass (it asserts
    /// =2 for audited after two record()s, not =0 baseline) but would
    /// spuriously inflate `format_summary_default_shape` (dhrxw) AND
    /// confuse the CI gate's "delta since start" math.
    #[test]
    fn audit_summary_default_zero_initializes_every_counter() {
        let summary = AuditSummary::default();
        assert_eq!(summary.fights_audited, 0);
        assert_eq!(summary.fights_with_any_flag, 0);
        for kind in RedFlagKind::ALL {
            assert_eq!(
                summary.count(kind),
                0,
                "AuditSummary::default() must zero-init counts[{kind:?}]",
            );
        }
    }

    /// `sts2-rs-bo7qm`: `Display for RedFlagKind` (L158) forwards to
    /// `as_str()`. Pin the equivalence so a regression that swapped
    /// the Display impl to use a debug repr (or a wrong stringifier)
    /// is caught — `format_summary` and `format!("{kind}=...")` callers
    /// depend on this equivalence for stable CI log output.
    #[test]
    fn red_flag_kind_display_matches_as_str() {
        for kind in RedFlagKind::ALL {
            assert_eq!(
                format!("{kind}"),
                kind.as_str(),
                "Display impl diverged from as_str() for {kind:?}",
            );
        }
    }

    /// `sts2-rs-q2yhd`: complement to `format_summary_default_shape`
    /// (sts2-rs-dhrxw). Asserts the post-record token values
    /// surface in the right slot: the formatted line reflects the
    /// actual `count(kind)` for each recorded flag, not the default
    /// zero. Catches a regression that always read `counts[0]` (or
    /// some other fixed slot) instead of indexing by `kind`.
    #[test]
    fn format_summary_reflects_recorded_counts() {
        let mut summary = AuditSummary::default();
        let mk = |kind: RedFlagKind| RedFlag {
            kind,
            turn: 1,
            decision_index: 0,
            detail: String::new(),
        };
        // Two fights, the second flagged twice with different kinds.
        summary.record(&[]);
        summary.record(&[
            mk(RedFlagKind::SuicidePlay),
            mk(RedFlagKind::EndTurnWithHighEnergy),
        ]);

        let line = format_summary(&summary);

        assert!(
            line.contains("audited=2"),
            "expected 'audited=2' in: {line}",
        );
        assert!(
            line.contains("flagged_fights=1"),
            "expected 'flagged_fights=1' in: {line}",
        );
        assert!(
            line.contains("suicide_play=1"),
            "expected 'suicide_play=1' in: {line}",
        );
        assert!(
            line.contains("end_turn_with_high_energy=1"),
            "expected 'end_turn_with_high_energy=1' in: {line}",
        );
        // Sibling kinds remain at zero — pin that we don't bleed
        // the count into the wrong slot.
        assert!(
            line.contains("zero_play_combat=0"),
            "expected 'zero_play_combat=0' (unrecorded) in: {line}",
        );
    }

    /// `sts2-rs-6dz41`: `AuditSummary::record` increments
    /// `fights_audited` for every call but `fights_with_any_flag`
    /// only when `per_fight` is non-empty. A regression that flipped
    /// the negation or order would silently miscount the corpus-wide
    /// gate denominator.
    #[test]
    fn audit_summary_record_counter_semantics() {
        let mut summary = AuditSummary::default();

        // Clean fight (no flags) — bumps fights_audited only.
        summary.record(&[]);
        assert_eq!(summary.fights_audited, 1);
        assert_eq!(summary.fights_with_any_flag, 0);
        assert_eq!(summary.count(RedFlagKind::SuicidePlay), 0);

        // Flagged fight (one suicide_play) — bumps both + counts.
        let mk_flag = |kind: RedFlagKind| RedFlag {
            kind,
            turn: 1,
            decision_index: 0,
            detail: String::new(),
        };
        summary.record(&[mk_flag(RedFlagKind::SuicidePlay)]);
        assert_eq!(summary.fights_audited, 2);
        assert_eq!(summary.fights_with_any_flag, 1);
        assert_eq!(summary.count(RedFlagKind::SuicidePlay), 1);

        // Two flags of same kind — fights_audited += 1 once but
        // counts[suicide_play] += 2.
        summary.record(&[
            mk_flag(RedFlagKind::SuicidePlay),
            mk_flag(RedFlagKind::SuicidePlay),
        ]);
        assert_eq!(summary.fights_audited, 3);
        assert_eq!(summary.fights_with_any_flag, 2);
        assert_eq!(summary.count(RedFlagKind::SuicidePlay), 3);

        // `count(kind)` agrees with `counts[ALL.position(kind)]`.
        let idx = RedFlagKind::ALL
            .iter()
            .position(|k| *k == RedFlagKind::SuicidePlay)
            .unwrap();
        assert_eq!(summary.count(RedFlagKind::SuicidePlay), summary.counts[idx]);
    }

    /// `sts2-rs-dzgzr`: every `RedFlagKind::as_str()` value is unique
    /// across the `ALL` array. Catches a copy-paste error where two
    /// variants accidentally share a string — which would silently
    /// collapse the budgets table at `tests/parity/blunder_budgets.toml`
    /// because the gate keys off `as_str()`.
    #[test]
    fn red_flag_kind_as_str_values_are_distinct() {
        let strs: Vec<&'static str> = RedFlagKind::ALL.iter().map(|k| k.as_str()).collect();
        let unique: HashSet<&'static str> = strs.iter().copied().collect();
        assert_eq!(
            strs.len(),
            unique.len(),
            "RedFlagKind::as_str() produced duplicate string in: {strs:?}",
        );
        // Each variant's string must also be non-empty and contain
        // only `[a-z_]`.
        for s in &strs {
            assert!(
                !s.is_empty() && s.chars().all(|c| c.is_ascii_lowercase() || c == '_'),
                "RedFlagKind::as_str() = `{s}` is not snake_case",
            );
        }
    }

    fn empty_snapshot(turn: u16, player_powers: Vec<(PowerKind, i32)>) -> StateSnapshot {
        StateSnapshot {
            turn,
            phase: CombatPhase::PlayerPlayPhase,
            player_hp: 50,
            player_max_hp: 80,
            player_block: 0,
            player_energy: 3,
            player_max_energy: 3,
            draw_n: 0,
            discard_n: 0,
            exhaust_n: 0,
            hand: Vec::new(),
            monsters: Vec::new(),
            player_powers,
            potions_held: 0,
        }
    }

    fn turn_start_decision(
        idx: u32,
        turn: u16,
        player_powers: Vec<(PowerKind, i32)>,
    ) -> DecisionRecord {
        DecisionRecord {
            decision_index: idx,
            pre: empty_snapshot(turn, player_powers),
            root_outcomes: Vec::new(),
            total_visits: 0,
            recommended_idx: None,
            recommended_label: "(synthetic)".to_string(),
            pv: Vec::new(),
            player_hp_after: 50,
            monster_hp_after: Vec::new(),
            step_result: "Advanced".to_string(),
        }
    }

    fn synthetic_record(decisions: Vec<DecisionRecord>) -> FightRecord {
        FightRecord {
            fixture_path: PathBuf::from("synthetic"),
            fixture_stem: "synthetic".to_string(),
            encounter: None,
            floor: None,
            player_outcome_label: None,
            config: RecorderConfig::default(),
            final_won: true,
            final_round: 2,
            final_player_hp: 50,
            final_player_max_hp: 80,
            aborted_reason: None,
            decisions,
        }
    }

    #[test]
    fn flags_one_two_punch_persisting_across_player_turns() {
        let record = synthetic_record(vec![
            turn_start_decision(0, 1, vec![(PowerKind::OneTwoPunch, 1)]),
            turn_start_decision(1, 2, vec![(PowerKind::OneTwoPunch, 1)]),
        ]);
        let flags = audit_fight(&record);
        let leaks: Vec<_> = flags
            .iter()
            .filter(|f| f.kind == RedFlagKind::TransientPowerLeak)
            .collect();
        assert_eq!(leaks.len(), 1, "expected one TransientPowerLeak: {flags:?}");
        assert_eq!(leaks[0].turn, 2);
        assert!(
            leaks[0].detail.contains("OneTwoPunch"),
            "detail missing kind: {}",
            leaks[0].detail,
        );
    }

    #[test]
    fn quiet_when_one_two_punch_correctly_popped() {
        let record = synthetic_record(vec![
            turn_start_decision(0, 1, vec![(PowerKind::OneTwoPunch, 1)]),
            turn_start_decision(1, 2, vec![]),
        ]);
        let flags = audit_fight(&record);
        assert!(
            !flags
                .iter()
                .any(|f| f.kind == RedFlagKind::TransientPowerLeak),
            "spurious leak flag: {flags:?}",
        );
    }

    #[test]
    fn flags_flame_barrier_on_monster_slot() {
        let monster_with_fb = |stack: i32| MonsterSnapshot {
            slot: 0,
            kind: None,
            display_name: "syn".to_string(),
            hp: 30,
            max_hp: 30,
            block: 0,
            alive: true,
            intent: None,
            powers: vec![(PowerKind::FlameBarrier, stack)],
        };
        let mut t1 = turn_start_decision(0, 1, vec![]);
        t1.pre.monsters.push(monster_with_fb(3));
        let mut t2 = turn_start_decision(1, 2, vec![]);
        t2.pre.monsters.push(monster_with_fb(3));
        let record = synthetic_record(vec![t1, t2]);
        let flags = audit_fight(&record);
        let leaks: Vec<_> = flags
            .iter()
            .filter(|f| f.kind == RedFlagKind::TransientPowerLeak)
            .collect();
        assert_eq!(leaks.len(), 1, "expected one leak: {flags:?}");
        assert!(leaks[0].detail.contains("FlameBarrier"));
        assert!(leaks[0].detail.contains("monster[0]"));
    }

    fn card(hand_index: u8, card_id: CardId, cost: i8, card_type: CardType) -> CardInHand {
        CardInHand {
            hand_index,
            card_id,
            upgraded: false,
            cost,
            card_type,
        }
    }

    fn monster(slot: usize, hp: i32, block: i32, intent: Option<Intent>) -> MonsterSnapshot {
        MonsterSnapshot {
            slot,
            kind: None,
            display_name: format!("m{slot}"),
            hp,
            max_hp: hp.max(1),
            block,
            alive: hp > 0,
            intent,
            powers: Vec::new(),
        }
    }

    fn play_card_decision(
        idx: u32,
        turn: u16,
        hand_index: u8,
        target: Option<u8>,
        recommended_label: &str,
        snap_setup: impl FnOnce(&mut StateSnapshot),
    ) -> DecisionRecord {
        let mut pre = empty_snapshot(turn, vec![]);
        snap_setup(&mut pre);
        let action = Action::PlayCard { hand_index, target };
        let row = RootOutcomeRow {
            action,
            action_label: recommended_label.to_string(),
            visits: 1,
            wins: 1,
            losses: 0,
            ties: 0,
            mean_value: 0.0,
        };
        DecisionRecord {
            decision_index: idx,
            pre,
            root_outcomes: vec![row],
            total_visits: 1,
            recommended_idx: Some(0),
            recommended_label: recommended_label.to_string(),
            pv: vec![recommended_label.to_string()],
            player_hp_after: 50,
            monster_hp_after: Vec::new(),
            step_result: "Advanced".to_string(),
        }
    }

    fn end_turn_decision(
        idx: u32,
        turn: u16,
        snap_setup: impl FnOnce(&mut StateSnapshot),
    ) -> DecisionRecord {
        let mut pre = empty_snapshot(turn, vec![]);
        snap_setup(&mut pre);
        let row = RootOutcomeRow {
            action: Action::EndTurn,
            action_label: "End turn".to_string(),
            visits: 1,
            wins: 1,
            losses: 0,
            ties: 0,
            mean_value: 0.0,
        };
        DecisionRecord {
            decision_index: idx,
            pre,
            root_outcomes: vec![row],
            total_visits: 1,
            recommended_idx: Some(0),
            recommended_label: "End turn".to_string(),
            pv: vec!["End turn".to_string()],
            player_hp_after: 50,
            monster_hp_after: Vec::new(),
            step_result: "Advanced".to_string(),
        }
    }

    #[test]
    fn flags_defend_against_non_attacking_enemy() {
        let dec = play_card_decision(0, 1, 0, None, "Play DefendIronclad", |s| {
            s.hand
                .push(card(0, CardId::DefendIronclad, 1, CardType::Skill));
            s.monsters.push(monster(0, 30, 0, None));
        });
        let record = synthetic_record(vec![dec]);
        let flags = audit_fight(&record);
        assert_eq!(
            flags
                .iter()
                .filter(|f| f.kind == RedFlagKind::DefendNoAttackIntent)
                .count(),
            1,
            "expected DefendNoAttackIntent: {flags:?}",
        );
    }

    #[test]
    fn quiet_defend_when_barricade_owned() {
        let dec = play_card_decision(0, 1, 0, None, "Play DefendIronclad", |s| {
            s.hand
                .push(card(0, CardId::DefendIronclad, 1, CardType::Skill));
            s.monsters.push(monster(0, 30, 0, None));
            s.player_powers.push((PowerKind::Barricade, 1));
        });
        let record = synthetic_record(vec![dec]);
        let flags = audit_fight(&record);
        assert!(
            !flags
                .iter()
                .any(|f| f.kind == RedFlagKind::DefendNoAttackIntent),
            "spurious DefendNoAttackIntent under Barricade: {flags:?}",
        );
    }

    #[test]
    fn flags_end_turn_with_high_energy() {
        let dec = end_turn_decision(0, 1, |s| {
            s.player_energy = 3;
            s.hand.push(card(0, CardId::Bash, 2, CardType::Attack));
        });
        let record = synthetic_record(vec![dec]);
        let flags = audit_fight(&record);
        assert_eq!(
            flags
                .iter()
                .filter(|f| f.kind == RedFlagKind::EndTurnWithHighEnergy)
                .count(),
            1,
            "expected EndTurnWithHighEnergy: {flags:?}",
        );
    }

    #[test]
    fn quiet_end_turn_high_energy_when_only_power_in_hand() {
        let dec = end_turn_decision(0, 1, |s| {
            s.player_energy = 3;
            // Only a Power card is affordable — legitimate scaling-curve hold.
            s.hand.push(card(0, CardId::Bash, 9, CardType::Attack)); // unaffordable filler
            s.hand
                .push(card(1, CardId::DefendIronclad, 9, CardType::Skill)); // unaffordable filler
                                                                            // Add a Power-type card that IS affordable.
            s.hand
                .push(card(2, CardId::DefendIronclad, 1, CardType::Power));
        });
        let record = synthetic_record(vec![dec]);
        let flags = audit_fight(&record);
        assert!(
            !flags
                .iter()
                .any(|f| f.kind == RedFlagKind::EndTurnWithHighEnergy),
            "spurious EndTurnWithHighEnergy on Power-only hand: {flags:?}",
        );
    }

    #[test]
    fn flags_strike_when_bash_affordable() {
        let dec = play_card_decision(0, 1, 0, Some(0), "Play StrikeIronclad", |s| {
            s.player_energy = 3;
            s.hand
                .push(card(0, CardId::StrikeIronclad, 1, CardType::Attack));
            s.hand.push(card(1, CardId::Bash, 2, CardType::Attack));
            s.monsters.push(monster(0, 40, 0, None)); // not 1-shottable
        });
        let record = synthetic_record(vec![dec]);
        let flags = audit_fight(&record);
        assert_eq!(
            flags
                .iter()
                .filter(|f| f.kind == RedFlagKind::StrikeWhenBashAffordable)
                .count(),
            1,
            "expected StrikeWhenBashAffordable: {flags:?}",
        );
    }

    #[test]
    fn quiet_strike_when_target_is_one_shot_overkill() {
        let dec = play_card_decision(0, 1, 0, Some(0), "Play StrikeIronclad", |s| {
            s.player_energy = 3;
            s.hand
                .push(card(0, CardId::StrikeIronclad, 1, CardType::Attack));
            s.hand.push(card(1, CardId::Bash, 2, CardType::Attack));
            s.monsters.push(monster(0, 4, 0, None)); // hp=4, dies to Strike
        });
        let record = synthetic_record(vec![dec]);
        let flags = audit_fight(&record);
        assert!(
            !flags
                .iter()
                .any(|f| f.kind == RedFlagKind::StrikeWhenBashAffordable),
            "spurious StrikeWhenBashAffordable on 1-shot kill: {flags:?}",
        );
    }

    #[test]
    fn ignores_non_transient_powers() {
        let record = synthetic_record(vec![
            turn_start_decision(0, 1, vec![(PowerKind::Strength, 5)]),
            turn_start_decision(1, 2, vec![(PowerKind::Strength, 5)]),
        ]);
        let flags = audit_fight(&record);
        assert!(
            !flags
                .iter()
                .any(|f| f.kind == RedFlagKind::TransientPowerLeak),
            "Strength is not in TRANSIENT_POWERS: {flags:?}",
        );
    }
}
