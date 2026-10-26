//! Canonical action enumeration for the Wave-6 policy-head target
//! corpus — bead `sts2-rs-quuu` (qctu.6.x).
//!
//! The MCTS engine's [`sts2_core::Action`] enum has many variants
//! (PlayCard, EndTurn, DrinkPotion, Channel, Evoke, ResolveChoice,
//! …), and PlayCard's `hand_index` is volatile — it indexes into
//! a SmallVec whose order depends on draw history. For a policy
//! head to train against MCTS visit distributions, the action
//! axis has to map to a **stable** discrete space that survives
//! draw-order shuffling. The natural choice is to key on the
//! actual card a PlayCard plays (its [`sts2_core::cards::CardId`])
//! rather than the hand slot, and bucket every non-PlayCard
//! variant into a fixed slot per kind.
//!
//! ## Slot layout
//!
//! | Slot range | Action |
//! |---|---|
//! | `0 .. CARD_COUNT (580)` | `PlayCard { hand_index → state.player.hand[hand_index] → CardId }`. The slot index is the position of that CardId in [`CardId::all()`]. |
//! | `CARD_COUNT (580)` | `EndTurn` |
//! | `CARD_COUNT + 1 (581)` | `UsePotion` / `DrinkPotion` (both potion variants collapse into one slot — distinguishing them at the policy level is not actionable without a learned-policy-side potion choice, which is out of scope for the Wave-6 spike) |
//! | `CARD_COUNT + 2 (582)` | Any other variant (`Channel`, `Evoke`, `ResolveChoice`, future variants) — catch-all bucket |
//!
//! [`CANONICAL_ACTION_DIM`] is then `CARD_COUNT + 3`. Concretely
//! that's `583` at the current pin, but we pad to **640** so the
//! NPZ row width stays a multiple of 64 (numpy header alignment
//! convention) and future variants don't bump the schema.
//!
//! ## Stability
//!
//! Slot indices are stable as long as the `CardId::all()` slice
//! order doesn't change. That slice is hand-maintained in
//! `card_ids_generated.rs` against the C# decompilation, so a
//! content port can insert or drop a variant and shift indices —
//! the corpus generator stamps the `CardId::all().len()` value
//! into the NPZ metadata so the trainer can detect drift and
//! re-cache its index map. This module exports a
//! `current_card_count()` accessor for that purpose.
//!
//! ## Why bucket potions / catch-all
//!
//! The Wave-6 spike's policy head is value-only-adjacent — it
//! predicts the visit distribution as a regression target for the
//! shared trunk, not as a directly-actionable policy. The
//! engine still emits real `Action` variants at decision time
//! (the policy head's output is a learned PRIOR consumed by PUCT;
//! the action choice is still made by the search). For that
//! purpose, collapsing potion + catch-all variants into single
//! slots is fine — they're rare (potions ≈ 1 use/combat; the
//! catch-all is unreachable for the engine's `legal_actions` at
//! this pin). If a future bead wants per-potion policy granularity,
//! it can extend the slot layout without breaking the canonical
//! card-slot prefix.

use sts2_core::cards::CardId;
use sts2_core::{Action, CombatState};

/// Number of `CardId` variants at the current Python pin. Stamped
/// into NPZ metadata by the corpus generator so the downstream
/// trainer can detect schema drift from a pin bump.
#[must_use]
pub fn current_card_count() -> usize {
    CardId::all().len()
}

/// Slot index reserved for `EndTurn`. `CardId::all().len()` is 580 at the
/// v0.107.1 pin (card slots `0..580`), so the named slots start at 580.
pub const SLOT_END_TURN: usize = 580;

/// Slot index reserved for `UsePotion` / `DrinkPotion` (collapsed).
pub const SLOT_POTION: usize = 581;

/// Slot index reserved for any other Action variant.
pub const SLOT_OTHER: usize = 582;

/// Total policy-head dimension. Padded to 640 (multiple of 64) so
/// numpy alignment is clean + future variants have headroom
/// without a schema bump.
pub const CANONICAL_ACTION_DIM: usize = 640;

/// Map an `Action` taken from `state` to its canonical slot in
/// `0..CANONICAL_ACTION_DIM`. Returns `None` only when a
/// PlayCard's hand_index can't be resolved against `state.player.hand`
/// (which would be a caller-side bug — `legal_actions` never emits
/// such a PlayCard).
#[must_use]
pub fn action_to_slot(state: &CombatState, action: &Action) -> Option<usize> {
    match action {
        Action::PlayCard { hand_index, .. } => {
            let uid = *state.player.hand.get(usize::from(*hand_index))?;
            let card = state.cards.iter().find(|c| c.uid == uid)?;
            let card_id = card.card_id;
            // Linear scan over CardId::all(). 579 entries, cold
            // once-per-position cost — the corpus generator runs
            // this ~8 times per combat × ~12 500 combats = ~100 k
            // calls per 100 k-position corpus. ~60 μs total.
            CardId::all().iter().position(|c| *c == card_id)
        }
        Action::EndTurn => Some(SLOT_END_TURN),
        Action::UsePotion { .. } | Action::DrinkPotion { .. } => Some(SLOT_POTION),
        _ => Some(SLOT_OTHER),
    }
}

// ─── ActionKey (sts2-rs-pbvpy / GH #24) ──────────────────────────────
//
// Semantic action identity that retains target information. Unlike
// [`action_to_slot`] (which collapses `Bash → enemy_0` and `Bash →
// enemy_1` into the same CardId slot), [`ActionKey`] distinguishes
// them. This is the type the bead's redesigned policy head will train
// against — it's added here independently so the trainer-side
// migration can pick it up without further engine churn.
//
// **Scope of this commit:** define the type + the
// [`action_to_key`] mapping fn + unit tests. The trainer-side
// migration (corpus_gen NPZ schema bump, value_policy_net.py
// policy-head dimension change, deployment-vs-training search-config
// audit) is the second half of the bead's acceptance criteria and
// lands separately.
//
// The existing slot-based path (`action_to_slot` /
// `CANONICAL_ACTION_DIM = 640`) is untouched — every shipped fixture,
// trainer artifact, and downstream policy-prior consumer stays
// byte-stable.

/// Coarse target class for actions that may target a creature.
/// Distinguishes "no target", a specific enemy slot, or the player.
/// Two policy-relevant choices collapse to the same key when their
/// target class matches even if the underlying CombatState's enemy
/// arena reorders.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActionTarget {
    /// Card is untargeted (or the target field is `None`).
    None,
    /// Card targets a specific enemy. `u8` is the target index from
    /// the `Action::PlayCard.target` / `Action::DrinkPotion.target`
    /// field — i.e. the position in the alive-enemies list at
    /// decision time. Matching the engine's existing target semantics
    /// keeps the key trivially derivable from `Action` without a
    /// fresh state walk.
    Enemy(u8),
    /// Player-targeted (multiplayer-future; player_index is always 0
    /// at the current pin).
    Player(u8),
}

/// Semantic policy key for an action. Stable across draw-pile
/// reorderings (the `card_id` doesn't change with hand_index) and
/// distinguishes target choices for targeted plays.
///
/// Compared to [`action_to_slot`]'s collapsed 640-slot space, this is
/// a richer key that downstream trainers can hash to a learnable
/// embedding table — the slot space stays the rendering target while
/// `ActionKey` is the index key. A trainer can use this directly
/// (per-key embeddings) or project it down to a fixed-dim grid (e.g.
/// `(card_slot, target_class)` ⊆ `579 × 8` ≈ 4.6k slots) — the
/// projection lives on the trainer side; this module owns only the
/// canonical key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActionKey {
    PlayCard {
        /// Stable identity of the played card (resolved via
        /// `state.cards` lookup, not hand_index).
        card_id: CardId,
        /// Target classification — see [`ActionTarget`].
        target: ActionTarget,
    },
    EndTurn,
    Potion {
        /// Target classification (potions can target enemies for
        /// Fire/Block etc.).
        target: ActionTarget,
    },
    /// Channel / Evoke / ResolveChoice — collapsed bucket. These are
    /// not emitted by `legal_actions` at this pin (Channel/Evoke
    /// route through card effects directly; ResolveChoice is a
    /// pending-choice resolution that has its own decision surface).
    Other,
}

fn target_from_option(t: Option<u8>) -> ActionTarget {
    match t {
        None => ActionTarget::None,
        Some(idx) => ActionTarget::Enemy(idx),
    }
}

/// Number of distinct target classes addressed by the dense
/// [`action_to_keyed_slot`] projection. The classes:
///
/// | code | meaning |
/// |---|---|
/// | 0 | `ActionTarget::None` (untargeted) |
/// | 1..=5 | `ActionTarget::Enemy(i)` where `i = code - 1`, clamped at 4 |
/// | 6 | `ActionTarget::Player(_)` |
/// | 7 | reserved (future expansion / unmapped) |
///
/// Five-enemy ceiling matches `crate::corpus::ENEMY_SLOT_MAX`; any
/// engine state that ever spawns >5 hostile creatures simultaneously
/// would collapse the extra slots into `Enemy(4)`.
pub const TARGET_CLASS_COUNT: usize = 8;

/// Tail-slot layout for non-`PlayCard` action keys in the dense
/// [`action_to_keyed_slot`] projection. The card-grid occupies the
/// `[0, CARD_COUNT × TARGET_CLASS_COUNT)` prefix; these named
/// constants live in the tail.
pub const KEYED_SLOT_END_TURN: usize = 579 * TARGET_CLASS_COUNT; // 4632
pub const KEYED_SLOT_POTION_BASE: usize = KEYED_SLOT_END_TURN + 1; // 4633
pub const KEYED_SLOT_OTHER: usize = KEYED_SLOT_POTION_BASE + TARGET_CLASS_COUNT; // 4641

/// Total dimension of the dense [`action_to_keyed_slot`] projection.
/// Padded to the next multiple of 64 (4672) so numpy header alignment
/// stays clean and future variants have headroom without a schema
/// bump. GH #24 / bead `sts2-rs-9c7vr`.
pub const ACTION_KEY_DIM: usize = 4672;

const _: () = assert!(KEYED_SLOT_OTHER < ACTION_KEY_DIM);
const _: () = assert!(ACTION_KEY_DIM.is_multiple_of(64));

/// Project an [`ActionTarget`] to its dense target-class code in
/// `0..TARGET_CLASS_COUNT`. See [`TARGET_CLASS_COUNT`] for the
/// mapping table.
#[must_use]
pub fn target_class(target: ActionTarget) -> usize {
    match target {
        ActionTarget::None => 0,
        ActionTarget::Enemy(i) => 1 + (i.min(4) as usize),
        ActionTarget::Player(_) => 6,
    }
}

/// Dense projection of an [`ActionKey`] to a fixed slot in
/// `0..ACTION_KEY_DIM`. Unlike [`action_to_slot`] (which collapses
/// `Bash → enemy_0` and `Bash → enemy_1` into one slot), this keeps
/// target-class identity for `PlayCard` and `Potion` actions.
///
/// Slot layout:
///
/// | range | meaning |
/// |---|---|
/// | `card_slot × 8 + target_class` ∈ `[0, 4632)` | `PlayCard` keyed by `(CardId, target_class)` |
/// | `4632` | `EndTurn` |
/// | `4633 + target_class` ∈ `[4633, 4641)` | `Potion` (Use/Drink collapsed) by target class |
/// | `4641` | `ActionKey::Other` (Channel/Evoke/ResolveChoice — unreachable at this pin) |
/// | `[4642, 4672)` | padding/headroom |
///
/// Returns `None` only when a `PlayCard` `card_id` can't be resolved
/// in `CardId::all()` — a corpus drift signal, not a runtime bug.
#[must_use]
pub fn action_key_to_slot(key: ActionKey) -> Option<usize> {
    match key {
        ActionKey::PlayCard { card_id, target } => {
            let card_slot = CardId::all().iter().position(|c| *c == card_id)?;
            Some(card_slot * TARGET_CLASS_COUNT + target_class(target))
        }
        ActionKey::EndTurn => Some(KEYED_SLOT_END_TURN),
        ActionKey::Potion { target } => Some(KEYED_SLOT_POTION_BASE + target_class(target)),
        ActionKey::Other => Some(KEYED_SLOT_OTHER),
    }
}

/// Convenience: resolve `action → ActionKey → keyed slot` in one
/// step. Returns `None` for the same reasons as
/// [`action_to_key`] / [`action_key_to_slot`].
#[must_use]
pub fn action_to_keyed_slot(state: &CombatState, action: &Action) -> Option<usize> {
    action_to_key(state, action).and_then(action_key_to_slot)
}

/// Map an `Action` taken from `state` to its semantic [`ActionKey`].
/// Returns `None` only when a PlayCard's hand_index can't be resolved
/// (same contract as [`action_to_slot`]).
#[must_use]
pub fn action_to_key(state: &CombatState, action: &Action) -> Option<ActionKey> {
    match action {
        Action::PlayCard { hand_index, target } => {
            let uid = *state.player.hand.get(usize::from(*hand_index))?;
            let card = state.cards.iter().find(|c| c.uid == uid)?;
            Some(ActionKey::PlayCard {
                card_id: card.card_id,
                target: target_from_option(*target),
            })
        }
        Action::EndTurn => Some(ActionKey::EndTurn),
        Action::UsePotion { target, .. } | Action::DrinkPotion { target, .. } => {
            Some(ActionKey::Potion {
                target: target_from_option(*target),
            })
        }
        _ => Some(ActionKey::Other),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_layout_constants_are_consistent() {
        // CardId count + the three named slots + headroom must fit.
        let card_count = current_card_count();
        assert!(card_count <= SLOT_END_TURN, "CardId count drifted past 580");
        // Compile-time pins on the constants themselves — clippy
        // flags these as "constant value" if asserted at runtime,
        // so structurally they live as const-eval asserts in the
        // module body and we just reflect their structure here as
        // documentation.
        const _: () = assert!(SLOT_END_TURN + 1 == SLOT_POTION);
        const _: () = assert!(SLOT_POTION + 1 == SLOT_OTHER);
        const _: () = assert!(SLOT_OTHER < CANONICAL_ACTION_DIM);
        const _: () = assert!(CANONICAL_ACTION_DIM.is_multiple_of(64));
    }

    #[test]
    fn end_turn_maps_to_fixed_slot() {
        let s = CombatState::default();
        let slot = action_to_slot(&s, &Action::EndTurn);
        assert_eq!(slot, Some(SLOT_END_TURN));
    }

    #[test]
    fn potion_variants_collapse_to_one_slot() {
        let s = CombatState::default();
        let use_p = action_to_slot(
            &s,
            &Action::UsePotion {
                slot: 0,
                target: None,
            },
        );
        let drink_p = action_to_slot(
            &s,
            &Action::DrinkPotion {
                potion_slot: 0,
                target: None,
            },
        );
        assert_eq!(use_p, Some(SLOT_POTION));
        assert_eq!(drink_p, Some(SLOT_POTION));
    }

    #[test]
    fn unresolvable_play_card_returns_none() {
        let s = CombatState::default();
        // Empty hand — hand_index=0 fails the .get() lookup.
        let slot = action_to_slot(
            &s,
            &Action::PlayCard {
                hand_index: 0,
                target: None,
            },
        );
        assert_eq!(slot, None);
    }

    // ── ActionKey (sts2-rs-pbvpy) ─────────────────────────────────

    #[test]
    fn action_key_end_turn() {
        let s = CombatState::default();
        let key = action_to_key(&s, &Action::EndTurn);
        assert_eq!(key, Some(ActionKey::EndTurn));
    }

    #[test]
    fn action_key_potion_distinguishes_target_class() {
        let s = CombatState::default();
        let untargeted = action_to_key(
            &s,
            &Action::UsePotion {
                slot: 0,
                target: None,
            },
        );
        let enemy0 = action_to_key(
            &s,
            &Action::UsePotion {
                slot: 0,
                target: Some(0),
            },
        );
        let enemy1 = action_to_key(
            &s,
            &Action::UsePotion {
                slot: 0,
                target: Some(1),
            },
        );
        assert_eq!(
            untargeted,
            Some(ActionKey::Potion {
                target: ActionTarget::None
            })
        );
        assert_ne!(
            untargeted, enemy0,
            "untargeted and targeted potions must differ"
        );
        assert_ne!(
            enemy0, enemy1,
            "different enemy targets must produce different keys"
        );
    }

    #[test]
    fn action_key_use_and_drink_collapse_on_same_target() {
        let s = CombatState::default();
        let u = action_to_key(
            &s,
            &Action::UsePotion {
                slot: 0,
                target: Some(2),
            },
        );
        let d = action_to_key(
            &s,
            &Action::DrinkPotion {
                potion_slot: 0,
                target: Some(2),
            },
        );
        assert_eq!(u, d, "UsePotion/DrinkPotion are aliases — same key");
    }

    #[test]
    fn action_key_unresolvable_play_card_returns_none() {
        let s = CombatState::default();
        let k = action_to_key(
            &s,
            &Action::PlayCard {
                hand_index: 0,
                target: None,
            },
        );
        assert_eq!(k, None);
    }

    #[test]
    fn keyed_slot_layout_constants_are_consistent() {
        // Compile-time pins on the constants — runtime asserts get
        // flagged as "constant value" by clippy.
        const _: () = assert!(KEYED_SLOT_END_TURN == 579 * TARGET_CLASS_COUNT);
        const _: () = assert!(KEYED_SLOT_POTION_BASE == KEYED_SLOT_END_TURN + 1);
        const _: () = assert!(KEYED_SLOT_OTHER == KEYED_SLOT_POTION_BASE + TARGET_CLASS_COUNT);
        const _: () = assert!(KEYED_SLOT_OTHER < ACTION_KEY_DIM);
        const _: () = assert!(ACTION_KEY_DIM.is_multiple_of(64));
    }

    #[test]
    fn target_class_codes_are_distinct() {
        assert_eq!(target_class(ActionTarget::None), 0);
        assert_eq!(target_class(ActionTarget::Enemy(0)), 1);
        assert_eq!(target_class(ActionTarget::Enemy(4)), 5);
        // Clamping at 4: enemy index 5+ folds into the last enemy
        // slot (engine never spawns >5 simultaneously, see
        // corpus::ENEMY_SLOT_MAX).
        assert_eq!(target_class(ActionTarget::Enemy(7)), 5);
        assert_eq!(target_class(ActionTarget::Player(0)), 6);
    }

    #[test]
    fn keyed_slot_distinguishes_play_card_targets() {
        // Two PlayCard keys that share `card_id` but differ in target
        // map to different keyed slots — the central acceptance
        // criterion of GH #24.
        let card_id = CardId::all()[0];
        let s0 = action_key_to_slot(ActionKey::PlayCard {
            card_id,
            target: ActionTarget::Enemy(0),
        });
        let s1 = action_key_to_slot(ActionKey::PlayCard {
            card_id,
            target: ActionTarget::Enemy(1),
        });
        let s_none = action_key_to_slot(ActionKey::PlayCard {
            card_id,
            target: ActionTarget::None,
        });
        assert!(s0.is_some() && s1.is_some() && s_none.is_some());
        assert_ne!(s0, s1, "Bash→enemy_0 and Bash→enemy_1 must differ");
        assert_ne!(
            s_none, s0,
            "untargeted and enemy-targeted PlayCard must differ"
        );
        // Slots stay inside the card-grid prefix.
        assert!(s0.unwrap() < KEYED_SLOT_END_TURN);
    }

    #[test]
    fn keyed_slot_end_turn_and_potion_in_tail() {
        assert_eq!(
            action_key_to_slot(ActionKey::EndTurn),
            Some(KEYED_SLOT_END_TURN)
        );
        assert_eq!(
            action_key_to_slot(ActionKey::Potion {
                target: ActionTarget::None
            }),
            Some(KEYED_SLOT_POTION_BASE)
        );
        assert_eq!(
            action_key_to_slot(ActionKey::Potion {
                target: ActionTarget::Enemy(2)
            }),
            Some(KEYED_SLOT_POTION_BASE + 3)
        );
        assert_eq!(action_key_to_slot(ActionKey::Other), Some(KEYED_SLOT_OTHER));
    }

    #[test]
    fn keyed_slots_never_collide_across_variants() {
        // Sample one slot from each variant class and confirm
        // disjointness — guards against future refactors that
        // overlap the layout.
        let card_id = CardId::all()[7];
        let slots = [
            action_key_to_slot(ActionKey::PlayCard {
                card_id,
                target: ActionTarget::None,
            })
            .unwrap(),
            action_key_to_slot(ActionKey::PlayCard {
                card_id,
                target: ActionTarget::Enemy(3),
            })
            .unwrap(),
            action_key_to_slot(ActionKey::EndTurn).unwrap(),
            action_key_to_slot(ActionKey::Potion {
                target: ActionTarget::None,
            })
            .unwrap(),
            action_key_to_slot(ActionKey::Potion {
                target: ActionTarget::Enemy(1),
            })
            .unwrap(),
            action_key_to_slot(ActionKey::Other).unwrap(),
        ];
        let mut sorted = slots;
        sorted.sort_unstable();
        for w in sorted.windows(2) {
            assert_ne!(w[0], w[1], "keyed slot collision across variants");
        }
        for s in slots {
            assert!(s < ACTION_KEY_DIM);
        }
    }

    #[test]
    fn action_key_channel_evoke_collapse_to_other() {
        let s = CombatState::default();
        let ch = action_to_key(
            &s,
            &Action::Channel {
                orb: sts2_core::Orb::Empty,
                target: None,
            },
        );
        let ev = action_to_key(&s, &Action::Evoke { times: 1 });
        assert_eq!(ch, Some(ActionKey::Other));
        assert_eq!(ev, Some(ActionKey::Other));
    }
}
