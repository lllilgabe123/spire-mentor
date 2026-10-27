//! Defect orb mechanic — bead `sts2-rs-spy` (P7-7).
//!
//! Ports `sts2-rl-agent@01c094d:sts2_env/orbs/*.py` (or canonical STS
//! reference where the pin file is absent). Five orb types — Lightning,
//! Frost, Dark, Plasma, and the empty `Empty` slot — live in an
//! [`OrbSlots`] collection on [`crate::PlayerCombatState`]. The collection
//! exposes [`OrbSlots::channel`] (FIFO eviction at capacity),
//! [`OrbSlots::evoke`] (one orb), [`OrbSlots::evoke_all`] (every orb),
//! and [`OrbSlots::passive_tick`] (turn-end passive trigger for Lightning
//! and Frost).
//!
//! ## Canonical orb behaviours
//!
//! | Orb       | Passive (each turn end)              | Evoke (one-shot)                          |
//! |-----------|--------------------------------------|-------------------------------------------|
//! | Lightning | `3 + focus` to random alive enemy    | `8 + focus` to random alive enemy         |
//! | Frost     | `2 + focus` block                    | `5 + focus` block                         |
//! | Dark      | +`6 + focus` stored damage           | `6 + stored` damage to lowest-HP alive enemy |
//! | Plasma    | `+1` energy (turn start)             | `+2` energy this turn (no Focus)          |
//! | Glass     | `current_passive + focus` to all     | `(current_passive + focus) * 2` to all    |
//! | Empty     | nothing                              | nothing                                   |
//!
//! Glass orb (bead `sts2-rs-q89c8`) carries per-instance mutable
//! `current_passive: i32`, starting at 4 on channel. Each passive
//! tick deals `max(0, current_passive + focus)` to ALL alive enemies,
//! then decays `current_passive` by 1 (floor 0) — Focus modifies the
//! damage but NOT the decay. Mirrors Python pin
//! `sts2-rl-agent@01c094d:sts2_env/orbs/all.py:181-216`.
//!
//! Focus is read from the player's [`PowerKind::Focus`] stack; for Dark,
//! the +6 stored-damage passive scales additively as `6 + focus`, while
//! the evoke fires the accumulated value directly (no second Focus add).
//!
//! ## Slot mechanics
//!
//! [`OrbSlots`] starts at the canonical Defect baseline of 3 slots
//! ([`DEFAULT_ORB_SLOTS`]); Capacitor and Signal Boost grow capacity at
//! runtime via [`OrbSlots::add_slots`]. Channelling into a full
//! collection evicts the *oldest* orb first, mirroring Python's
//! `combat.add_orb` / `evoke_first_orb` ordering at pin `01c094d`.

use crate::card_effects::{deal_damage_to_all_alive_enemies, gain_block_player_unpowered};
use crate::hooks::ValueProps;
use crate::{CombatState, CreatureUid, PowerKind};
use smallvec::SmallVec;

/// Default number of orb slots a Defect starts combat with.
pub const DEFAULT_ORB_SLOTS: usize = 3;

/// The damage a freshly channelled Dark orb already evokes for, before any
/// passive tick. C# `DarkOrb._evokeVal = 6m` (`DarkOrb.cs:17`), surfaced as
/// `EvokeVal` (`:25`) and grown by `PassiveVal` each turn end (`:39`).
///
/// [`Orb::Dark`]'s `stored` field holds only the accumulated growth, so this
/// is the offset between the two representations. Every wire leg that carries
/// a C# `evoke` value must add it on the way out and subtract it on the way
/// in; getting that asymmetric is silent, because emit and parse are exact
/// inverses of each other and a round-trip test still passes.
pub const DARK_ORB_BASE_EVOKE: i32 = 6;

/// Hard upper bound for the inline `SmallVec` storage. Capacitor + Signal
/// Boost can push the slot count to ~10 in extreme builds; the heap
/// fallback handles anything beyond.
pub const ORB_SLOTS_INLINE: usize = 10;

/// A single orb instance carried by [`OrbSlots`]. `Empty` is the
/// canonical "slot is exposed but unfilled" marker used by the parity
/// exporter so the slot count is observable from the bridge dict.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Orb {
    Empty,
    Lightning,
    Frost,
    Dark {
        /// Accumulated damage stored by the per-turn passive. Reset to
        /// `0` on channel; grows by `6 + focus` per turn end.
        ///
        /// NOTE this is *not* the C# `EvokeVal`: `DarkOrb._evokeVal` starts
        /// at [`DARK_ORB_BASE_EVOKE`] and the passive adds to it
        /// (`DarkOrb.cs:17,25,39`), so `EvokeVal == stored + DARK_ORB_BASE_EVOKE`.
        /// Anything crossing the wire must convert — see `state_dict.rs`.
        stored: i32,
    },
    Plasma,
    /// Glass orb — per-instance decaying passive. Bead `sts2-rs-q89c8`.
    /// `current_passive` starts at 4 on channel; each passive tick
    /// deals `max(0, current_passive + focus)` to ALL alive enemies,
    /// then decrements by 1 (floor 0). Evoke deals
    /// `max(0, current_passive * 2 + focus)` to ALL alive enemies. Pin:
    /// `sts2-rl-agent@01c094d:sts2_env/orbs/all.py:181-216`.
    Glass {
        /// Remaining per-passive damage before Focus modifier. Decays by
        /// 1 on each passive tick; clamped at 0.
        current_passive: i32,
    },
}

impl Orb {
    /// Construct a freshly channelled orb of the given kind. Dark starts
    /// with `stored = 0`; the first passive tick grows it.
    #[must_use]
    pub const fn lightning() -> Self {
        Self::Lightning
    }
    #[must_use]
    pub const fn frost() -> Self {
        Self::Frost
    }
    #[must_use]
    pub const fn dark() -> Self {
        Self::Dark { stored: 0 }
    }
    #[must_use]
    pub const fn plasma() -> Self {
        Self::Plasma
    }
    /// Construct a freshly channelled Glass orb with `current_passive = 4`.
    /// Mirrors Python `GlassOrb.__init__` at pin
    /// `sts2-rl-agent@01c094d:sts2_env/orbs/all.py:194`.
    #[must_use]
    pub const fn glass() -> Self {
        Self::Glass { current_passive: 4 }
    }

    /// `true` when this orb cell is unfilled.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        matches!(self, Self::Empty)
    }
}

/// Per-player orb slot collection. Inline-sized to [`ORB_SLOTS_INLINE`]
/// so the common 3..6 slot range avoids the heap.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OrbSlots {
    /// Active orbs in channel order (front = oldest). `Empty` entries
    /// represent exposed-but-unfilled slots; the FIFO `channel` path
    /// overwrites them in order before evicting real orbs.
    pub slots: SmallVec<[Orb; ORB_SLOTS_INLINE]>,
    /// Maximum number of orbs simultaneously held. Mirrors Python's
    /// `combat.orb_slots`. Grows via [`Self::add_slots`] when Capacitor
    /// or Signal Boost fires.
    pub capacity: u8,
    /// The belt's size is real, not a fixture default: set by every
    /// constructor that sizes it, the wire import, and Bulk Up. A Defect
    /// whose belt is real and shrank to 0 (two Bulk Ups) channels nothing,
    /// as in C#; only an unsized belt snaps to the 3-slot default.
    pub sized: bool,
}

impl Default for OrbSlots {
    fn default() -> Self {
        // Empty by default: until the player builds Defect and runs
        // through the combat-start hook (or channels their first orb),
        // there are no slots. Fixtures opt in via `with_default_capacity`.
        Self {
            slots: SmallVec::new(),
            capacity: 0,
            sized: false,
        }
    }
}

impl OrbSlots {
    /// Construct with the canonical Defect baseline of 3 empty slots
    /// exposed.
    #[must_use]
    pub fn with_default_capacity() -> Self {
        Self::with_capacity(DEFAULT_ORB_SLOTS)
    }

    /// Construct with `cap` empty slots exposed.
    #[must_use]
    pub fn with_capacity(cap: usize) -> Self {
        let mut slots = SmallVec::new();
        for _ in 0..cap {
            slots.push(Orb::Empty);
        }
        Self {
            slots,
            #[allow(clippy::cast_possible_truncation)]
            capacity: cap.min(u8::MAX as usize) as u8,
            sized: true,
        }
    }

    /// Number of filled (non-empty) orbs.
    #[must_use]
    pub fn filled_len(&self) -> usize {
        self.slots.iter().filter(|o| !o.is_empty()).count()
    }

    /// Total slot capacity (empty + filled).
    #[must_use]
    pub fn capacity(&self) -> usize {
        usize::from(self.capacity)
    }

    /// Add `n` extra empty slots. Mirrors `Capacitor`'s "gain N orb
    /// slots" power-on-play effect.
    pub fn add_slots(&mut self, n: usize) {
        self.sized = true;
        for _ in 0..n {
            self.slots.push(Orb::Empty);
        }
        self.capacity = self.capacity.saturating_add(
            #[allow(clippy::cast_possible_truncation)]
            {
                n.min(u8::MAX as usize) as u8
            },
        );
    }

    /// Channel an orb into the collection. Fills the leftmost `Empty`
    /// slot if one exists; otherwise evicts the *oldest* orb (FIFO) by
    /// rotating the new orb onto the back of the queue. No-op when
    /// `capacity == 0`.
    pub fn channel(&mut self, orb: Orb) {
        if self.capacity == 0 {
            return;
        }
        if let Some(slot) = self.slots.iter_mut().find(|o| o.is_empty()) {
            *slot = orb;
            return;
        }
        // All slots filled — evict the front (oldest) orb and push
        // the new orb onto the back.
        if !self.slots.is_empty() {
            self.slots.remove(0);
        }
        self.slots.push(orb);
    }

    /// Count the number of orbs of `kind` currently in slots. Used by
    /// Ice Lance, Compile Driver, Tesla Coil to scale.
    #[must_use]
    pub fn count_kind(&self, kind: OrbKind) -> usize {
        self.slots.iter().filter(|o| OrbKind::of(o) == kind).count()
    }

    /// Number of distinct *non-empty* orb kinds currently held. Used by
    /// Compile Driver scaling.
    #[must_use]
    pub fn distinct_kinds(&self) -> usize {
        let mut seen = [false; 5];
        for orb in &self.slots {
            match orb {
                Orb::Lightning => seen[0] = true,
                Orb::Frost => seen[1] = true,
                Orb::Dark { .. } => seen[2] = true,
                Orb::Plasma => seen[3] = true,
                Orb::Glass { .. } => seen[4] = true,
                Orb::Empty => {}
            }
        }
        seen.iter().filter(|s| **s).count()
    }
}

/// Discriminant for orb kinds without the `Dark { stored }` /
/// `Glass { current_passive }` payload. Stable equality key used by
/// [`OrbSlots::count_kind`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OrbKind {
    Empty,
    Lightning,
    Frost,
    Dark,
    Plasma,
    Glass,
}

impl OrbKind {
    /// Project an [`Orb`] to its kind discriminant.
    #[must_use]
    pub const fn of(orb: &Orb) -> Self {
        match orb {
            Orb::Empty => Self::Empty,
            Orb::Lightning => Self::Lightning,
            Orb::Frost => Self::Frost,
            Orb::Dark { .. } => Self::Dark,
            Orb::Plasma => Self::Plasma,
            Orb::Glass { .. } => Self::Glass,
        }
    }
}

// ---------------------------------------------------------------------------
// Focus helper
// ---------------------------------------------------------------------------

/// Effective Focus applied to orb values: the player's real `FocusPower`.
///
/// In the C# decomp only `FocusPower` carries `ModifyOrbValue`
/// (`FocusPower.cs`: `value + Amount`). `TemporaryFocusPower` and its subclass
/// `SynchronizePower` are NOT orb modifiers themselves — when applied they
/// INJECT an equal-amount real `FocusPower` into the owner (their
/// `InternallyAppliedPower => FocusPower`) and revert it on `AfterTurnEnd`.
/// The Rust port mirrors that injection: the `Synchronize` card applies
/// `PowerKind::Focus` alongside `PowerKind::Synchronize`
/// (`cards/defect/synchronize.rs`), and the `after_turn_end` Synchronize arm
/// reverts the Focus. So the transient powers are already accounted for via
/// the injected `Focus` — counting `Synchronize`/`TemporaryFocus` again here
/// DOUBLE-COUNTED, over-scaling every orb passive/evoke by the transient
/// amount (orb-parity `ORB_STATE|SYNCHRONIZE`: C# Glass passive 4 vs rs 6 with
/// Focus(2)+Synchronize(2) live). Read only `Focus`.
#[must_use]
pub fn focus_bonus(state: &CombatState) -> i32 {
    // Walk the player's mirror+legacy power UNION (bead `sts2-rs-vzzg1`).
    // Relic combat-start Focus grants (e.g. DataDisk) land on the mirror
    // Creature via `apply_power_to_creature` and never touch the legacy
    // `state.player.powers` SmallVec, so a legacy-only walk would read Focus=0
    // and every orb passive/evoke silently drop the Focus bonus (CrackedCore
    // Lightning under-dealt by 1/turn — sweep dd78 seed 3346).
    // `player_power_amount` dedupes mirror-vs-legacy by `PowerUid`.
    state.player_power_amount(PowerKind::Focus)
}

/// A Focus-scaled orb value, clamped at 0 to match C#
/// `FocusPower.ModifyOrbValue` → `Math.Max(value + Amount, 0m)`
/// (`FocusPower.cs`). Negative Focus can drive a Lightning/Frost/Dark/Glass
/// orb value below its base, but C# never lets the realized passive/evoke go
/// below 0 — so an orb under heavy Focus drain deals 0, not negative
/// (Lightning would otherwise HEAL the enemy and Frost grant negative block;
/// orb-parity `ORB_STATE|{HYPERBEAM,ICE_LANCE,COLD_SNAP}`: cs passive 0 vs a
/// negative rs).
#[inline]
#[must_use]
pub fn focus_scaled(base: i32, focus: i32) -> i32 {
    base.saturating_add(focus).max(0)
}

/// Additive bonus applied to a **Lightning** orb's passive/evoke value
/// when the player owns `InfusedCore` (v0.107.1).
///
/// C# `MegaCrit.Sts2.Core.Models.Relics/InfusedCore.cs:50-60`
/// (`ModifyOrbValue`): for a `LightningOrb` owned by the relic owner,
/// `value + ExtraDamage.BaseValue` (`ExtraDamage = 1`). Modelled as a
/// raise of the orb's base value before the Focus clamp — for the
/// in-game (non-negative Focus) case this is identical to C#'s separate
/// `ModifyOrbValue` hook running alongside `FocusPower.ModifyOrbValue`.
/// Bead `sts2-rs-v0107-content-port-wb8lk.9` (§7).
#[inline]
#[must_use]
fn lightning_orb_bonus(state: &CombatState) -> i32 {
    i32::from(crate::relic_effects::helpers::player_owns_relic(
        state,
        crate::RelicKind::InfusedCore,
    ))
}

// ---------------------------------------------------------------------------
// Evoke / passive resolution
// ---------------------------------------------------------------------------

/// Roll one orb off the run-level `combat_orbs` stream — C#
/// `OrbModel.GetRandomOrb(Rng rng)` (`OrbModel.cs:166-169`), which is
/// `rng.NextItem(_validOrbs)` over the FIVE-entry table Lightning / Frost /
/// Dark / Plasma / Glass (`OrbModel.cs:26-32`). `Rng.CombatOrbGeneration` maps
/// to `RunRngType.CombatOrbs` (`RunRngSet.cs`) → salt `"combat_orbs"`.
///
/// Shared by every C# caller of `GetRandomOrb`: Chaos's `OnPlay` and
/// TrashToTreasure's `AfterCardGeneratedForCombat`. Kept in one place so the
/// table and the one-read-per-orb draw count cannot drift between them —
/// bead `sts2-rs-dh30e` records a prior 4-roll here that dropped Glass.
#[must_use]
pub fn random_combat_orb(state: &mut CombatState) -> Orb {
    match state.run_rng.combat_orbs.next_int_exclusive(0, 5) {
        0 => Orb::lightning(),
        1 => Orb::frost(),
        2 => Orb::dark(),
        3 => Orb::plasma(),
        _ => Orb::glass(),
    }
}
/// Channel a fresh orb onto the player.
pub fn channel_orb(state: &mut CombatState, orb: Orb) {
    // C# `OrbCmd.Channel` (`OrbCmd.cs:51-62`): a character with
    // `BaseOrbSlotCount == 0` (i.e. NOT Defect — e.g. an Ironclad channeling a
    // generated Defect card's orb such as MeteorStrike's 3 Plasma) that channels
    // into a zero-capacity belt gets a TEMPORARY 1-slot belt (`AddSlots(1)`),
    // not the full Defect belt. The prior port grew to the 3-slot Defect default
    // for everyone, so an Ironclad held all 3 Plasma instead of overflow-evoking
    // them — dropping the evoke energy C# grants (sweep `player.energy` cs=4
    // rs=0 on METEOR_STRIKE; bead `sts2-rs-dnxjt`). Defect fixtures that skipped
    // deck-build capacity setup still get the canonical 3-slot belt.
    if state.player.orbs.capacity == 0 {
        if state.player.character == Some(crate::potions::CharacterId::Defect) {
            if state.player.orbs.sized {
                // A real belt removed by Bulk Up: C# `TryEnqueue` fails and
                // the orb is lost. Snapping back to 3 here made the search
                // block with Frost it did not have (live Aeonglass loss,
                // 2026-10-27: 98% "win" at 0 slots).
                return;
            }
            state.player.orbs = OrbSlots::with_default_capacity();
        } else {
            state.player.orbs.add_slots(1);
        }
    }
    // C# evokes the front orb when the belt is already full BEFORE enqueueing
    // the new one (`OrbCmd.Channel` → `EvokeNext` with `dequeue:true`, which
    // `orbQueue.Remove`s the evoked front, then `TryEnqueue` appends the new orb
    // at the BACK — a FIFO queue). `OrbSlots::channel` evicts the front WITHOUT
    // evoking, silently dropping its effect (Plasma's +2 evoke energy,
    // Lightning's hit, …); do the evoke here while we hold `&mut state`.
    if state.player.orbs.filled_len() >= state.player.orbs.capacity as usize {
        evoke_front(state);
        // FIFO shift: `evoke_front` blanks the evoked orb IN PLACE (front-most
        // slot → Empty), but the next `channel` call refills the FIRST empty —
        // i.e. that same front slot — so a repeated overflow would evoke and
        // refill the front over and over while the orbs BEHIND it (e.g. a Frost
        // queued mid-belt) are never reached. C#'s queue instead removes the
        // front and appends the new orb at the back, advancing the queue. Mirror
        // that by rotating the freed slot to the end, so `channel` appends the
        // new orb behind the survivors and the next overflow evokes the NEXT
        // orb. Without this, Tempest/Rainbow/Refract channels over a full belt
        // re-evoke the front Lightning instead of advancing to the next orb
        // (sweep `MONSTER_STATE|TEMPEST`/`combat_targets` desync, seed 154).
        if let Some(pos) = state.player.orbs.slots.iter().position(Orb::is_empty) {
            let freed = state.player.orbs.slots.remove(pos);
            state.player.orbs.slots.push(freed);
        }
    }
    state.player.orbs.channel(orb);
    // Event log (C# `OrbChanneledEntry`). Recorded AFTER the belt write so an
    // overflow evoke's row (and its damage) precedes the channel that caused
    // it, matching the order the game's log shows.
    crate::event_log::record(
        state,
        crate::event_log::EventLogEntry::OrbChanneled {
            orb: OrbKind::of(&orb),
        },
    );
    // Bead `sts2-rs-7y9vg`: combat-wide counter feeding Voltaic's C#
    // `CalculatedVar` (`History.OfType<OrbChanneledEntry>.Count(...)`).
    if matches!(orb, Orb::Lightning) {
        state.combat_log.lightning_orbs_channeled_this_combat = state
            .combat_log
            .lightning_orbs_channeled_this_combat
            .saturating_add(1);
    }
    // Bead `sts2-rs-1h8q` ch.5: fire on_orb_channeled on both buses.
    // Power-side is observer-only; relic-side carries `&mut` so
    // Metronome-style bodies can mutate state directly.
    let ctx = crate::hooks::OrbChanneledCtx::default();
    crate::hooks::dispatch_on_orb_channeled(state, &ctx);
    let _ = crate::relic_effects::dispatch_on_orb_channeled(state, &ctx);
}

/// Evoke the front-most filled orb (and remove it), LEFT-PACKING the belt:
/// the orb is removed from its slot and an `Empty` pushed to the back, so the
/// survivors shift left and empties stay at the back. C#'s `OrbQueue` is a
/// left-packed FIFO (`Remove` drops the front and shifts the rest down), so a
/// later `channel` appends BEHIND the survivors. Blanking the slot in place
/// instead would leave a hole the next `channel` refills — putting the new orb
/// in front of survivors (orb-parity `ORB_STATE|{DARKNESS,CONSUMING_SHADOW,
/// GLACIER,MULTI_CAST}`: evoke-then-channel produced `[DARK, FROST, …]` vs C#'s
/// `[FROST, …, DARK]`).
pub fn evoke_front(state: &mut CombatState) {
    let Some(idx) = state.player.orbs.slots.iter().position(|o| !o.is_empty()) else {
        return;
    };
    let orb = state.player.orbs.slots.remove(idx);
    state.player.orbs.slots.push(Orb::Empty);
    evoke_one(state, orb);
}

/// Evoke the most-recently-channeled (last filled) orb and consume it.
/// Mirrors C# `OrbCmd.EvokeLast` (`OrbCmd.cs`): `orb = OrbQueue.Orbs.Last();
/// Evoke(orb, dequeue: true)`. Orbs are front-packed (see [`OrbSlots::channel`]
/// / [`evoke_front`]), so the last non-empty slot is the newest orb. Removal
/// left-packs survivors (the slot is dropped and an `Empty` pushed to the back)
/// to keep the FIFO invariant the rest of the orb code relies on. Used by
/// `ConsumingShadowPower.AfterTurnEnd`, which evokes the last orb once per
/// stack.
pub fn evoke_last(state: &mut CombatState) {
    let Some(idx) = state.player.orbs.slots.iter().rposition(|o| !o.is_empty()) else {
        return;
    };
    let orb = state.player.orbs.slots.remove(idx);
    state.player.orbs.slots.push(Orb::Empty);
    evoke_one(state, orb);
}

/// Evoke the front-most orb `times` times (repeat-evoking the same orb
/// without consuming, the Dualcast / Multi-Cast pattern). The orb is
/// consumed at the *end* of the sequence — mirrors Python's
/// `evoke_first_orb(times=N)` which calls `on_evoke` N times then
/// removes the orb. Left-packs on removal (see [`evoke_front`]).
pub fn evoke_front_n(state: &mut CombatState, times: u32) {
    // C# `OrbCmd.EvokeNext(.., dequeue: i == count-1)` only dequeues on the
    // LAST iteration, so `count == 0` (Multi-Cast / Tempest at X=0) evokes
    // nothing AND removes nothing. Guard the removal on `times > 0` — the prior
    // unconditional dequeue consumed the front orb even at X=0 (orb-parity
    // `ORB_STATE|MULTI_CAST`: cs kept 5 orbs, rs dropped to 4).
    if times == 0 {
        return;
    }
    let Some(idx) = state.player.orbs.slots.iter().position(|o| !o.is_empty()) else {
        return;
    };
    let orb = state.player.orbs.slots[idx];
    for _ in 0..times {
        evoke_one(state, orb);
    }
    state.player.orbs.slots.remove(idx);
    state.player.orbs.slots.push(Orb::Empty);
}

/// Evoke every filled orb left-to-right. Used by Echo Form / All for
/// One (rare paths) and `passive_tick`'s evoke phase.
pub fn evoke_all(state: &mut CombatState) {
    let positions: Vec<usize> = state
        .player
        .orbs
        .slots
        .iter()
        .enumerate()
        .filter_map(|(i, o)| (!o.is_empty()).then_some(i))
        .collect();
    for i in positions {
        let orb = state.player.orbs.slots[i];
        state.player.orbs.slots[i] = Orb::Empty;
        evoke_one(state, orb);
    }
}

/// Evoke every filled orb, firing each orb's evoke effect `times` times before
/// consuming it. Mirrors v0.107.1 `Shatter.OnPlay`: it captures
/// `OrbQueue.Orbs.Count` up front, then per orb runs
/// `OrbCmd.EvokeNext(.., dequeue: false)` followed by `OrbCmd.EvokeNext(..)`
/// (dequeue) — i.e. the FRONT orb evokes `times` (=2) times back-to-back, then
/// is removed, before moving to the next orb (order: orb0,orb0,orb1,orb1,…).
/// Implemented via [`evoke_front_n`] once per captured orb so the
/// repeat-then-consume sequencing (and the left-pack on removal) match exactly.
pub fn evoke_all_n(state: &mut CombatState, times: u32) {
    let count = state
        .player
        .orbs
        .slots
        .iter()
        .filter(|o| !o.is_empty())
        .count();
    for _ in 0..count {
        evoke_front_n(state, times);
    }
}

/// `damage_trace` label for an orb firing in `phase` (`"evoke"` /
/// `"passive"`). Static strings so the disarmed path allocates nothing.
fn orb_trace_label(orb: Orb, phase: &str) -> &'static str {
    match (OrbKind::of(&orb), phase) {
        (OrbKind::Lightning, "evoke") => "orb:Lightning:evoke",
        (OrbKind::Frost, "evoke") => "orb:Frost:evoke",
        (OrbKind::Dark, "evoke") => "orb:Dark:evoke",
        (OrbKind::Plasma, "evoke") => "orb:Plasma:evoke",
        (OrbKind::Glass, "evoke") => "orb:Glass:evoke",
        (OrbKind::Lightning, _) => "orb:Lightning:passive",
        (OrbKind::Frost, _) => "orb:Frost:passive",
        (OrbKind::Dark, _) => "orb:Dark:passive",
        (OrbKind::Plasma, _) => "orb:Plasma:passive",
        (OrbKind::Glass, _) => "orb:Glass:passive",
        (OrbKind::Empty, _) => "orb:Empty",
    }
}

/// Resolve one orb's evoke effect, then fire the `AfterOrbEvoked` hook
/// against the creatures it hit. Mirrors C# `OrbCmd.Evoke`: the orb's
/// `Evoke()` returns its targets, then `Hook.AfterOrbEvoked(orb, targets)`
/// fires (ThunderPower amplifies Lightning evokes). The slot is consumed by
/// the caller — this only resolves the effect + listener reaction.
fn evoke_one(state: &mut CombatState, orb: Orb) {
    let targets = resolve_evoke(state, orb);
    crate::hooks::dispatch_after_orb_evoked(state, OrbKind::of(&orb), &targets);
}

/// Fire the per-orb-kind evoke effect and return the creatures it targeted
/// (for `AfterOrbEvoked`). Pure dispatcher — does not mutate the slot
/// collection (caller is responsible for consuming). Block / energy orbs
/// return no creature targets; the returned list is consumed only by the
/// Lightning-gated ThunderPower reactor, so Glass's AoE need not be
/// enumerated here.
fn resolve_evoke(state: &mut CombatState, orb: Orb) -> SmallVec<[CreatureUid; 4]> {
    let _trace = crate::damage_trace::push_source(orb_trace_label(orb, "evoke"));
    // Event log. C# has no evoke entry — the log shows only the damage — but a
    // Frost or Plasma evoke deals none, so without this row the belt emptying
    // is invisible. Recorded before the effect so the evoke's own damage rows
    // follow it.
    crate::event_log::record(
        state,
        crate::event_log::EventLogEntry::OrbEvoked {
            orb: OrbKind::of(&orb),
        },
    );
    let focus = focus_bonus(state);
    let mut targets: SmallVec<[CreatureUid; 4]> = SmallVec::new();
    match orb {
        Orb::Lightning => {
            let dmg = focus_scaled(8 + lightning_orb_bonus(state), focus);
            if let Some(target) = random_alive_enemy(state) {
                // C# `LightningOrb.ApplyLightningDamage`: `CreatureCmd.Damage(
                // targets, value, ValueProp.Unpowered, Owner.Creature)` — orb
                // evokes are UNPOWERED (no Strength / Weak / Shrink / Vulnerable
                // scaling) and source no card, identical to the passive tick
                // below. `deal_damage_player_to_enemy` stamps `MOVE` (powered)
                // and reads the in-play card as the source, so a player power
                // like Shrink (Shrinker Beetle, "Attacks deal 30% less damage")
                // wrongly shrank the evoke — `Dualcast` over one Lightning under
                // Shrink dealt `2 * floor(8 * 0.7) = 10` instead of C#'s `16`,
                // surfacing in run_parity as a mid-combat `enemies_hp` divergence
                // (e.g. RP29 Shrinker Beetle `cs=[4] rs=[10]`). Mirror the
                // passive's UNPOWERED path. Bead `sts2-rs-01ygn` family.
                let _ = crate::damage::apply_damage(
                    state,
                    crate::hooks::CreatureRef::Player,
                    target,
                    dmg,
                    ValueProps::UNPOWERED,
                    None,
                );
                targets.push(target);
            }
        }
        Orb::Frost => {
            gain_block_player_unpowered(state, focus_scaled(5, focus));
        }
        Orb::Dark { stored } => {
            // C# `DarkOrb._evokeVal` starts at 6 and the passive ADDS
            // `PassiveVal` (6 + focus) to it each turn end; the evoke deals
            // `_evokeVal` (DarkOrb.cs:16,24,37,52). Our `stored` field tracks
            // only the ACCUMULATED passive growth (0 on channel), so the evoke
            // value is `stored + 6` — the base the prior port dropped, dealing
            // 6 too little per Dark evoke (and ×N under Dualcast/Quadcast/
            // Multi-Cast: QUADCAST seed 227 dealt 4×6 where C# dealt 4×12).
            let dmg = stored.saturating_add(DARK_ORB_BASE_EVOKE).max(0);
            if let Some(target) = lowest_hp_enemy(state) {
                // C# `DarkOrb.Evoke`: `CreatureCmd.Damage(weakestEnemy, EvokeVal,
                // ValueProp.Unpowered, Owner.Creature)` — like Lightning, the
                // Dark evoke is UNPOWERED. Route through the unpowered pipeline
                // so player damage powers (Strength / Shrink / Weak) don't scale
                // it (and no card is stamped as the source).
                let _ = crate::damage::apply_damage(
                    state,
                    crate::hooks::CreatureRef::Player,
                    target,
                    dmg,
                    ValueProps::UNPOWERED,
                    None,
                );
                targets.push(target);
            }
        }
        Orb::Plasma => {
            // C# `PlasmaOrb.cs` `EvokeVal => 2m` → `PlayerCmd.GainEnergy(2)`.
            // (Passive is +1; the Python pin collapsed both to +1 — a drift.)
            // Focus does NOT scale a Plasma evoke. Bead `sts2-rs-plasma-evoke`.
            state.player.energy = state.player.energy.saturating_add(2);
        }
        Orb::Glass { current_passive } => {
            // C# `GlassOrb.EvokeVal => PassiveVal * 2` with
            // `PassiveVal => ModifyOrbValue(_passiveVal)` = `max(_passiveVal +
            // focus, 0)` (`GlassOrb.cs` + `FocusPower.ModifyOrbValue`). So Focus
            // is applied BEFORE the ×2 and is therefore DOUBLED — the Python-pin
            // `_passiveVal*2 + focus` (focus added once) under-dealt by `focus`
            // per Glass evoke (RAINBOW seed 259: 2 overflow-evoked Glass under
            // Focus(2) dealt 16 vs C#'s 20). C# also clamps `EvokeVal <= 0` to
            // no damage, matched by the `dmg > 0` guard.
            let dmg = focus_scaled(current_passive, focus).saturating_mul(2);
            if dmg > 0 {
                deal_damage_to_all_alive_enemies(state, dmg, ValueProps::UNPOWERED, None);
            }
        }
        Orb::Empty => {}
    }
    targets
}

/// Fire the front-most filled orb's passive exactly once, including
/// the Plasma `+1 energy` arm. Mirrors Python
/// `sts2-rl-agent@01c094d:sts2_env/orbs/base.py:128-132`
/// (`OrbQueue.trigger_first_passive` — calls `orb.on_passive(combat)`
/// on the first orb regardless of `orb_type`).
///
/// Distinct from [`trigger_first_orb_passive_n`]: the `_n` variant
/// skips Plasma (it ports the Loop-power callsite which mirrors
/// `passive_tick`'s turn-end semantics). This function ports the
/// `OrbQueue::trigger_first_passive` surface that `GoldPlatedCables`
/// drives from `after_side_turn_start` (Plasma branch) and
/// `before_turn_end` (non-Plasma branch) — both arms route through
/// the same single-orb passive, and Plasma must fire `+1 energy`
/// in the turn-start arm.
///
/// Empty front slot is a no-op.
///
/// Bead `sts2-rs-rt2ry`.
pub fn trigger_first_orb_passive(state: &mut CombatState) {
    let Some(idx) = state.player.orbs.slots.iter().position(|o| !o.is_empty()) else {
        return;
    };
    trigger_orb_passive_at(state, idx);
}

/// Fire the passive of the orb in slot `idx` exactly once — C#
/// `OrbModel.Passive` for one orb, Plasma's `+1 energy` included. The orb is
/// read fresh from the slot, so a second call sees the first call's Dark
/// accumulation / Glass decay. An empty slot is a no-op.
///
/// [`trigger_first_orb_passive`] is this on the front-most filled slot.
pub fn trigger_orb_passive_at(state: &mut CombatState, idx: usize) {
    match state.player.orbs.slots.get(idx) {
        Some(o) if !o.is_empty() => {}
        _ => return,
    }
    let focus = focus_bonus(state);
    let orb = state.player.orbs.slots[idx];
    let _trace = crate::damage_trace::push_source(orb_trace_label(orb, "passive"));
    match orb {
        Orb::Lightning => {
            let dmg = focus_scaled(3 + lightning_orb_bonus(state), focus);
            if let Some(target) = random_alive_enemy(state) {
                // C# `LightningOrb.ApplyLightningDamage`: `ValueProp.Unpowered`,
                // no card source — orb zaps don't scale with Strength and, as
                // non-attacks, don't trigger the target's Thorns (C#
                // `ThornsPower` gates on `IsPoweredAttack()`). `deal_damage_player_to_enemy`
                // wrongly stamps `MOVE` (powered) + reads the in-play card as
                // source, which fired SpinyToad's Thorns against the player on a
                // GoldPlatedCables re-trigger (bead `sts2-rs-u2y5w`; seed 5040).
                // Match the `passive_tick` Lightning branch.
                let _ = crate::damage::apply_damage(
                    state,
                    crate::hooks::CreatureRef::Player,
                    target,
                    dmg,
                    ValueProps::UNPOWERED,
                    None,
                );
            }
        }
        Orb::Frost => {
            gain_block_player_unpowered(state, focus_scaled(2, focus));
        }
        Orb::Dark { stored } => {
            let new = stored.saturating_add(focus_scaled(6, focus));
            state.player.orbs.slots[idx] = Orb::Dark { stored: new };
        }
        Orb::Plasma => {
            state.player.energy = state.player.energy.saturating_add(1);
        }
        Orb::Glass { current_passive } => {
            // C# `GlassOrb.Passive` (GlassOrb.cs:34-39): when
            // `PassiveVal <= 0` (i.e. Focus drags `current_passive + focus`
            // to/below 0) the entire passive is SKIPPED — no damage AND no
            // decay. The `_passiveVal -= 1` step lives INSIDE the
            // `if (!(passiveVal <= 0m))` guard. Focus modifies the damage
            // but the decay only happens on a firing tick.
            let dmg = current_passive.saturating_add(focus).max(0);
            if dmg > 0 {
                deal_damage_to_all_alive_enemies(state, dmg, ValueProps::UNPOWERED, None);
                let new = (current_passive - 1).max(0);
                state.player.orbs.slots[idx] = Orb::Glass {
                    current_passive: new,
                };
            }
        }
        Orb::Empty => {}
    }
}

/// Trigger only the front-most filled orb's passive `times` times.
/// Used by `LoopPower.after_side_turn_start` (Python pin
/// `sts2-rl-agent@01c094d:sts2_env/powers/turn_effects.py:378-381` —
/// `for _ in range(self.amount): combat.trigger_first_orb_passive(owner)`).
///
/// Bead `sts2-rs-zlx.4`: added so Loop's per-turn-start tick can fire
/// without consuming the orb or running the full [`passive_tick`] walk.
///
/// Mirrors [`passive_tick`]'s per-orb branches restricted to a single
/// slot, plus the Plasma `+1 energy` passive (Empty is a no-op). C#
/// `LoopPower.AfterPlayerTurnStart` triggers `OrbCmd.Passive(Orbs[0])`
/// `Amount` times and `PlasmaOrb`'s passive is `GainEnergy(1)`, so a front
/// Plasma orb under Loop grants energy per tick — matching the singular
/// [`trigger_first_orb_passive`] (the earlier `_n` Plasma skip was a bug,
/// Defect sweep `PLAYER_STATE|end_turn` seed 1171).
pub fn trigger_first_orb_passive_n(state: &mut CombatState, times: u32) {
    let Some(idx) = state.player.orbs.slots.iter().position(|o| !o.is_empty()) else {
        return;
    };
    let focus = focus_bonus(state);
    let orb = state.player.orbs.slots[idx];
    let _trace = crate::damage_trace::push_source(orb_trace_label(orb, "passive"));
    for _ in 0..times {
        match orb {
            Orb::Lightning => {
                let dmg = focus_scaled(3 + lightning_orb_bonus(state), focus);
                if let Some(target) = random_alive_enemy(state) {
                    // UNPOWERED, no card source — see `trigger_first_orb_passive`
                    // (bead `sts2-rs-u2y5w`): orb zaps are not powered
                    // attacks and must not trigger the target's Thorns.
                    let _ = crate::damage::apply_damage(
                        state,
                        crate::hooks::CreatureRef::Player,
                        target,
                        dmg,
                        ValueProps::UNPOWERED,
                        None,
                    );
                }
            }
            Orb::Frost => {
                gain_block_player_unpowered(state, focus_scaled(2, focus));
            }
            Orb::Dark { .. } => {
                // Re-read the live stored value each iteration: with `times > 1`
                // (Loop ≥ 2 on a Dark front orb) the accumulation must compound
                // (+6 per trigger), like the Glass arm below. Matching on the
                // captured `orb` used the ORIGINAL `stored` every iteration, so
                // N triggers added only one +6 (SHATTER seed 189: Loop(2) Dark
                // evoke 18 vs C#'s 24).
                let cur = match state.player.orbs.slots[idx] {
                    Orb::Dark { stored } => stored,
                    _ => 0,
                };
                let new = cur.saturating_add(focus_scaled(6, focus));
                state.player.orbs.slots[idx] = Orb::Dark { stored: new };
            }
            Orb::Glass { current_passive } => {
                // Re-read latest stored value each iteration so the decay
                // chain is observable across `times` invocations.
                let cur = match state.player.orbs.slots[idx] {
                    Orb::Glass { current_passive } => current_passive,
                    _ => current_passive,
                };
                let dmg = cur.saturating_add(focus).max(0);
                if dmg > 0 {
                    // C# decays `_passiveVal` only inside the `passiveVal > 0`
                    // guard (GlassOrb.cs:34-39) — skip both damage and decay
                    // when Focus clamps the tick to 0.
                    deal_damage_to_all_alive_enemies(state, dmg, ValueProps::UNPOWERED, None);
                    let new = (cur - 1).max(0);
                    state.player.orbs.slots[idx] = Orb::Glass {
                        current_passive: new,
                    };
                }
            }
            Orb::Plasma => {
                // C# `LoopPower.AfterPlayerTurnStart` calls
                // `OrbCmd.Passive(Orbs[0])` `Amount` times, and `PlasmaOrb`'s
                // passive is `GainEnergy(1)` — so Loop on a front Plasma orb
                // grants +1 energy PER tick (separate from, and additive to,
                // the once-per-turn Plasma turn-start passive applied in
                // `combat_loop`). The prior `_n` Plasma skip dropped this
                // (Defect sweep `PLAYER_STATE|end_turn` seed 1171: Loop(2) +
                // front Plasma, C# energy 6 vs Rust 4). Bead `sts2-rs-zlx.4`
                // follow-up — matches the singular `trigger_first_orb_passive`.
                state.player.energy = state.player.energy.saturating_add(1);
            }
            Orb::Empty => {}
        }
    }
}

/// Tick every orb's passive effect (turn-end). Lightning and Frost
/// fire; Dark accumulates damage; Plasma is inert; Focus enters
/// additively for the damage / block / stored amounts.
pub fn passive_tick(state: &mut CombatState) {
    let focus = focus_bonus(state);
    // Take a snapshot of orb kinds so we can mutate `state.player.orbs`
    // (Dark stored amount) without borrow-checker pain mid-iter.
    let snapshot: Vec<(usize, Orb)> = state
        .player
        .orbs
        .slots
        .iter()
        .enumerate()
        .map(|(i, o)| (i, *o))
        .collect();
    for (idx, orb) in snapshot {
        let _trace = crate::damage_trace::push_source(orb_trace_label(orb, "passive"));
        match orb {
            Orb::Lightning => {
                let dmg = focus_scaled(3 + lightning_orb_bonus(state), focus);
                if let Some(target) = random_alive_enemy(state) {
                    // C# `LightningOrb.ApplyLightningDamage`:
                    // `CreatureCmd.Damage(targets, value,
                    // ValueProp.Unpowered, Owner.Creature)` — orb zaps are
                    // UNPOWERED (no Strength / Vulnerable / Weak scaling)
                    // and source no card. `deal_damage_player_to_enemy`
                    // stamps `MOVE` (powered) and reads the in-play card as
                    // the source, both wrong for a passive tick. Bead
                    // `sts2-rs-01ygn` (coverage `relic:CRACKED_CORE`).
                    let _ = crate::damage::apply_damage(
                        state,
                        crate::hooks::CreatureRef::Player,
                        target,
                        dmg,
                        ValueProps::UNPOWERED,
                        None,
                    );
                }
            }
            Orb::Frost => {
                gain_block_player_unpowered(state, focus_scaled(2, focus));
            }
            Orb::Dark { stored } => {
                let new = stored.saturating_add(focus_scaled(6, focus));
                state.player.orbs.slots[idx] = Orb::Dark { stored: new };
            }
            Orb::Glass { current_passive } => {
                // C# `GlassOrb.Passive` (GlassOrb.cs:34-39): decay only
                // happens on a firing tick. When Focus clamps the passive to
                // 0 the orb keeps its full `_passiveVal`.
                let dmg = current_passive.saturating_add(focus).max(0);
                if dmg > 0 {
                    deal_damage_to_all_alive_enemies(state, dmg, ValueProps::UNPOWERED, None);
                    let new = (current_passive - 1).max(0);
                    state.player.orbs.slots[idx] = Orb::Glass {
                        current_passive: new,
                    };
                }
            }
            Orb::Plasma | Orb::Empty => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Target selection helpers
// ---------------------------------------------------------------------------

fn random_alive_enemy(state: &mut CombatState) -> Option<CreatureUid> {
    let alive: Vec<CreatureUid> = state
        .enemy_uids
        .iter()
        .copied()
        .filter(|uid| {
            state
                .creatures
                .iter()
                .any(|c| c.id == uid.0 && c.current_hp > 0)
        })
        .collect();
    if alive.is_empty() {
        return None;
    }
    let idx = state
        .run_rng
        .combat_targets
        .next_int_exclusive(0, alive.len() as i32) as usize;
    Some(alive[idx])
}

fn lowest_hp_enemy(state: &CombatState) -> Option<CreatureUid> {
    state
        .enemy_uids
        .iter()
        .copied()
        .filter_map(|uid| {
            state
                .creatures
                .iter()
                .find(|c| c.id == uid.0 && c.current_hp > 0)
                .map(|c| (uid, c.current_hp))
        })
        .min_by_key(|(_, hp)| *hp)
        .map(|(uid, _)| uid)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Creature, CreatureUid, IntentHistory, MonsterId, MonsterKind, Power, PowerUid};

    use smallvec::SmallVec;
    use std::num::NonZeroU32;

    fn make_state(player_hp: i32, enemy_hps: &[i32]) -> CombatState {
        let mut state = CombatState {
            rng: crate::dotnet_rng::Rng::new(0x00DE_FEC7_i32),
            ..Default::default()
        };
        state.player.current_hp = player_hp;
        state.player.max_hp = player_hp;
        state.player.orbs = OrbSlots::with_default_capacity();
        for (i, &hp) in enemy_hps.iter().enumerate() {
            #[allow(clippy::cast_possible_truncation)]
            let id = NonZeroU32::new((i as u32) + 1).expect("nz");
            state.creatures.push(Creature {
                id,
                current_hp: hp,
                max_hp: hp,
                block: 0,
                powers: SmallVec::new(),
                species: Some(MonsterKind::Nibbit),
                intent: None,
                intent_history: IntentHistory::default(),
                monster_id: MonsterId("Nibbit"),
                ai: None,
            });
            state.enemy_uids.push(CreatureUid(id));
        }
        state
    }

    fn give_power(state: &mut CombatState, kind: PowerKind, amount: i32) {
        let pid = NonZeroU32::new(u32::try_from(state.powers.len()).expect("powers fit u32") + 1)
            .expect("nz");
        state.powers.push(Power {
            id: pid,
            kind,
            amount,
            counter: 0,
            applier: None,
            skip_next_tick: false,
        });
        state.player.powers.push(PowerUid(pid));
    }

    fn give_focus(state: &mut CombatState, amount: i32) {
        give_power(state, PowerKind::Focus, amount);
    }

    /// `focus_bonus` counts ONLY the real `FocusPower`. The transient
    /// `Synchronize`/`TemporaryFocus` markers inject an equal `FocusPower` when
    /// applied (C# `TemporaryFocusPower.InternallyAppliedPower`; the Rust
    /// Synchronize card applies `Focus` too — `cards/defect/synchronize.rs`),
    /// so adding them here as well DOUBLE-COUNTED and over-scaled every orb
    /// (orb-parity `ORB_STATE|SYNCHRONIZE`). The earlier bead `sts2-rs-1hoa3`
    /// summed all three back when the Rust port stored the markers standalone
    /// WITHOUT injecting Focus; once the injection landed, the sum became a
    /// double-count.
    #[test]
    fn focus_bonus_counts_real_focus_only() {
        let mut state = make_state(50, &[100]);
        assert_eq!(focus_bonus(&state), 0, "no focus powers → 0");
        give_focus(&mut state, 2);
        // The Synchronize/TemporaryFocus markers do NOT add on top of the
        // injected Focus they already produce.
        give_power(&mut state, PowerKind::Synchronize, 4);
        give_power(&mut state, PowerKind::TemporaryFocus, 1);
        assert_eq!(
            focus_bonus(&state),
            2,
            "only the real Focus(2) scales orbs; the markers are not re-counted"
        );
    }

    #[test]
    fn default_orb_slots_has_three_empty() {
        let slots = OrbSlots::with_default_capacity();
        assert_eq!(slots.capacity(), 3);
        assert_eq!(slots.filled_len(), 0);
        assert_eq!(slots.slots.len(), 3);
        assert!(slots.slots.iter().all(Orb::is_empty));
    }

    #[test]
    fn channel_fills_empty_slots_left_to_right() {
        let mut s = OrbSlots::with_default_capacity();
        s.channel(Orb::lightning());
        s.channel(Orb::frost());
        assert_eq!(s.filled_len(), 2);
        assert_eq!(s.slots[0], Orb::Lightning);
        assert_eq!(s.slots[1], Orb::Frost);
        assert_eq!(s.slots[2], Orb::Empty);
    }

    #[test]
    fn channel_evicts_oldest_when_full() {
        let mut s = OrbSlots::with_default_capacity();
        s.channel(Orb::lightning());
        s.channel(Orb::frost());
        s.channel(Orb::dark());
        assert_eq!(s.filled_len(), 3);
        // Now full. Channel a Plasma — Lightning (oldest) is evicted.
        s.channel(Orb::plasma());
        assert_eq!(s.filled_len(), 3);
        assert_eq!(s.slots[0], Orb::Frost);
        assert_eq!(s.slots[1], Orb::Dark { stored: 0 });
        assert_eq!(s.slots[2], Orb::Plasma);
    }

    #[test]
    fn channel_into_zero_capacity_is_noop() {
        let mut s = OrbSlots::default();
        s.channel(Orb::lightning());
        assert_eq!(s.filled_len(), 0);
    }

    /// `evoke_front` LEFT-PACKS the belt (survivors shift left, empty pushed to
    /// the back), so a later channel appends BEHIND the survivors — matching
    /// C#'s left-packed FIFO. orb-parity `ORB_STATE|{DARKNESS,GLACIER,…}`.
    #[test]
    fn evoke_front_left_packs_so_channel_appends_at_back() {
        let mut state = make_state(50, &[100]);
        channel_orb(&mut state, Orb::lightning());
        channel_orb(&mut state, Orb::frost());
        // Belt: [Lightning, Frost, Empty]. Evoke the front (Lightning).
        evoke_front(&mut state);
        assert_eq!(
            state.player.orbs.slots[0],
            Orb::Frost,
            "survivor packed left"
        );
        assert!(state.player.orbs.slots[1].is_empty());
        // Channel Dark — must land BEHIND Frost (not in the freed front slot).
        channel_orb(&mut state, Orb::dark());
        let kinds: Vec<_> = state
            .player
            .orbs
            .slots
            .iter()
            .filter(|o| !o.is_empty())
            .map(OrbKind::of)
            .collect();
        assert_eq!(kinds, vec![OrbKind::Frost, OrbKind::Dark]);
    }

    /// `evoke_front_n(0)` (Multi-Cast / Tempest at X=0) evokes nothing AND
    /// removes nothing — C# only dequeues on the last iteration, which never
    /// runs at count 0. orb-parity `ORB_STATE|MULTI_CAST`.
    #[test]
    fn evoke_front_n_zero_does_not_consume_the_front_orb() {
        let mut state = make_state(50, &[100]);
        channel_orb(&mut state, Orb::lightning());
        evoke_front_n(&mut state, 0);
        assert_eq!(state.player.orbs.filled_len(), 1, "X=0 evoke keeps the orb");
        assert_eq!(state.player.orbs.slots[0], Orb::Lightning);
    }

    #[test]
    fn add_slots_grows_capacity_with_empty_cells() {
        let mut s = OrbSlots::with_default_capacity();
        s.add_slots(2);
        assert_eq!(s.capacity(), 5);
        assert_eq!(s.slots.len(), 5);
    }

    #[test]
    fn evoke_lightning_deals_8_plus_focus() {
        // Deep pin #1: Focus 2 → Lightning evoke = 10 dmg.
        let mut state = make_state(50, &[100]);
        channel_orb(&mut state, Orb::lightning());
        give_focus(&mut state, 2);
        evoke_front(&mut state);
        assert_eq!(state.creatures[0].current_hp, 90, "8 + 2 focus = 10");
        // Slot was consumed.
        assert_eq!(state.player.orbs.filled_len(), 0);
    }

    #[test]
    fn thunder_amplifies_lightning_evoke() {
        // C# `ThunderPower.AfterOrbEvoked`: when a Lightning orb evokes,
        // deal `Amount` Unpowered damage to the orb's targets. Lightning
        // evoke = 8 (+0 focus), then Thunder(6) hits the SAME enemy = 14.
        let mut state = make_state(50, &[100]);
        channel_orb(&mut state, Orb::lightning());
        give_power(&mut state, PowerKind::Thunder, 6);
        evoke_front(&mut state);
        assert_eq!(state.creatures[0].current_hp, 86, "8 evoke + 6 Thunder");
        assert_eq!(state.player.orbs.filled_len(), 0);
    }

    #[test]
    fn thunder_does_not_amplify_frost_evoke() {
        // Thunder only reacts to Lightning evokes — a Frost evoke (block,
        // no creature target) triggers no Thunder damage.
        let mut state = make_state(50, &[100]);
        channel_orb(&mut state, Orb::frost());
        give_power(&mut state, PowerKind::Thunder, 6);
        evoke_front(&mut state);
        assert_eq!(
            state.creatures[0].current_hp, 100,
            "Frost evoke: no Thunder"
        );
        assert_eq!(state.player.block, 5);
    }

    #[test]
    fn thunder_fires_per_repeat_evoke() {
        // Dualcast repeat-evokes the front Lightning orb twice: each evoke
        // fires Thunder. 2 × (8 + 6) = 28 against a single enemy.
        let mut state = make_state(50, &[100]);
        channel_orb(&mut state, Orb::lightning());
        give_power(&mut state, PowerKind::Thunder, 6);
        evoke_front_n(&mut state, 2);
        assert_eq!(state.creatures[0].current_hp, 72, "2 × (8 + 6) = 28");
    }

    #[test]
    fn channel_overflow_advances_fifo_queue() {
        // C# `OrbCmd.Channel` is a FIFO queue: a channel into a FULL belt
        // evokes (and removes) the FRONT orb, then appends the new orb at the
        // BACK. A second overflow must therefore evoke the NEXT orb, not
        // re-evoke the refilled front. Regression for the in-place-blank bug
        // that re-evoked the front Lightning forever while a queued Frost was
        // never reached (sweep `MONSTER_STATE|TEMPEST`, seed 154).
        //
        // Belt (capacity 3): channel Lightning, Frost, Lightning → full
        // [Ligh, Fros, Ligh]. Then channel 2 more Lightning (Tempest-style):
        //   overflow 1 → evoke front Lightning (8 dmg), queue → [Fros, Ligh, Ligh]
        //   overflow 2 → evoke front Frost (+5 block), queue → [Ligh, Ligh, Ligh]
        let mut state = make_state(50, &[100]);
        channel_orb(&mut state, Orb::lightning());
        channel_orb(&mut state, Orb::frost());
        channel_orb(&mut state, Orb::lightning());
        assert_eq!(state.player.orbs.filled_len(), 3, "belt full");

        channel_orb(&mut state, Orb::lightning());
        channel_orb(&mut state, Orb::lightning());

        // The Frost reached the front on the 2nd overflow and evoked → block.
        // The pre-fix bug re-evoked the front Lightning twice (0 block).
        assert_eq!(state.player.block, 5, "queued Frost must evoke for 5 block");
        // One Lightning evoke landed (8 dmg); the in-place-blank bug landed two.
        assert_eq!(
            state.creatures[0].current_hp, 92,
            "exactly one Lightning evoke (8)"
        );
        // Belt ends all-Lightning (Frost was evoked out, two Lightning appended).
        assert_eq!(state.player.orbs.count_kind(OrbKind::Lightning), 3);
        assert_eq!(state.player.orbs.count_kind(OrbKind::Frost), 0);
    }

    #[test]
    fn evoke_frost_grants_5_plus_focus_block() {
        let mut state = make_state(50, &[100]);
        channel_orb(&mut state, Orb::frost());
        give_focus(&mut state, 1);
        evoke_front(&mut state);
        assert_eq!(state.player.block, 6);
    }

    #[test]
    fn evoke_plasma_grants_two_energy_no_focus() {
        // C# `PlasmaOrb.EvokeVal => 2m`: Plasma evoke = +2 energy, Focus does
        // not scale (bead `sts2-rs-plasma-evoke`).
        let mut state = make_state(50, &[100]);
        channel_orb(&mut state, Orb::plasma());
        give_focus(&mut state, 5);
        let pre = state.player.energy;
        evoke_front(&mut state);
        assert_eq!(
            state.player.energy,
            pre + 2,
            "plasma: +2 energy, focus does not scale"
        );
    }

    #[test]
    fn evoke_dark_deals_base_plus_stored_damage_to_lowest_hp() {
        let mut state = make_state(50, &[100, 30]);
        channel_orb(&mut state, Orb::Dark { stored: 12 });
        evoke_front(&mut state);
        // Dark evoke = 6 base + 12 accumulated = 18 to the lowest-HP enemy
        // (idx 1, 30hp → 12). C# `DarkOrb._evokeVal` base-6 + passive growth.
        assert_eq!(state.creatures[0].current_hp, 100);
        assert_eq!(state.creatures[1].current_hp, 12);
    }

    #[test]
    fn passive_tick_lightning_damages_and_frost_blocks() {
        let mut state = make_state(50, &[100]);
        channel_orb(&mut state, Orb::lightning());
        channel_orb(&mut state, Orb::frost());
        passive_tick(&mut state);
        // Lightning 3, Frost 2 block.
        assert_eq!(state.creatures[0].current_hp, 97);
        assert_eq!(state.player.block, 2);
    }

    #[test]
    fn passive_tick_dark_accumulates_stored() {
        let mut state = make_state(50, &[100]);
        channel_orb(&mut state, Orb::dark());
        passive_tick(&mut state);
        assert_eq!(state.player.orbs.slots[0], Orb::Dark { stored: 6 });
        passive_tick(&mut state);
        assert_eq!(state.player.orbs.slots[0], Orb::Dark { stored: 12 });
    }

    #[test]
    fn passive_tick_dark_scales_with_focus() {
        let mut state = make_state(50, &[100]);
        channel_orb(&mut state, Orb::dark());
        give_focus(&mut state, 2);
        passive_tick(&mut state);
        // 6 + 2 focus = 8 stored.
        assert_eq!(state.player.orbs.slots[0], Orb::Dark { stored: 8 });
    }

    #[test]
    fn capacitor_expands_slots() {
        // Deep pin #2: Capacitor expands slots by N.
        let mut state = make_state(50, &[100]);
        assert_eq!(state.player.orbs.capacity(), 3);
        state.player.orbs.add_slots(2);
        assert_eq!(state.player.orbs.capacity(), 5);
        // Channeling fills past 3 without eviction.
        for _ in 0..5 {
            channel_orb(&mut state, Orb::lightning());
        }
        assert_eq!(state.player.orbs.filled_len(), 5);
    }

    #[test]
    fn evoke_front_n_repeats_then_consumes() {
        // Dualcast: evoke front orb twice, then remove.
        let mut state = make_state(50, &[100]);
        channel_orb(&mut state, Orb::lightning());
        evoke_front_n(&mut state, 2);
        // Two lightning hits: 8 + 8 = 16.
        assert_eq!(state.creatures[0].current_hp, 84);
        assert_eq!(state.player.orbs.filled_len(), 0);
    }

    #[test]
    fn evoke_all_consumes_every_orb() {
        let mut state = make_state(50, &[100]);
        channel_orb(&mut state, Orb::frost());
        channel_orb(&mut state, Orb::frost());
        evoke_all(&mut state);
        // Two frost evokes: 5 + 5 = 10 block.
        assert_eq!(state.player.block, 10);
        assert_eq!(state.player.orbs.filled_len(), 0);
    }

    #[test]
    fn evoke_all_n_double_evokes_each_orb_then_consumes() {
        // v0.107.1 Shatter: each orb evokes TWICE then is consumed. Two Frost
        // orbs → 4 frost evokes (5 block each) = 20 block, belt emptied.
        let mut state = make_state(50, &[100]);
        channel_orb(&mut state, Orb::frost());
        channel_orb(&mut state, Orb::frost());
        evoke_all_n(&mut state, 2);
        assert_eq!(state.player.block, 20, "2 orbs × 2 evokes × 5 block");
        assert_eq!(state.player.orbs.filled_len(), 0, "every orb consumed");
    }

    #[test]
    fn distinct_kinds_counts_unique() {
        let mut s = OrbSlots::with_default_capacity();
        s.channel(Orb::lightning());
        s.channel(Orb::lightning());
        s.channel(Orb::frost());
        assert_eq!(s.distinct_kinds(), 2);
        assert_eq!(s.count_kind(OrbKind::Lightning), 2);
        assert_eq!(s.count_kind(OrbKind::Frost), 1);
    }

    #[test]
    fn trigger_first_orb_passive_lightning_fires_once() {
        // rt2ry: GoldPlatedCables surface re-fires the front orb's
        // passive without consuming it. Lightning front → 3 dmg, slot
        // remains filled.
        let mut state = make_state(50, &[100]);
        channel_orb(&mut state, Orb::lightning());
        channel_orb(&mut state, Orb::frost());
        trigger_first_orb_passive(&mut state);
        assert_eq!(state.creatures[0].current_hp, 97, "3 dmg from lightning");
        assert_eq!(state.player.block, 0, "frost behind front must not fire");
        assert_eq!(state.player.orbs.filled_len(), 2, "orb not consumed");
    }

    #[test]
    fn trigger_first_orb_passive_plasma_grants_energy() {
        // rt2ry: diverges from `_n` — Plasma fires +1 energy here.
        // Mirrors Python `OrbQueue.trigger_first_passive` →
        // `PlasmaOrb.on_passive` → `combat.energy += 1`. Focus does
        // not scale Plasma.
        let mut state = make_state(50, &[100]);
        channel_orb(&mut state, Orb::plasma());
        give_focus(&mut state, 3);
        let pre = state.player.energy;
        trigger_first_orb_passive(&mut state);
        assert_eq!(
            state.player.energy,
            pre + 1,
            "plasma: +1 energy, focus inert"
        );
        assert_eq!(state.player.orbs.filled_len(), 1, "plasma not consumed");
    }

    #[test]
    fn trigger_first_orb_passive_n_plasma_grants_energy_per_tick() {
        // Loop(N) on a front Plasma orb fires `OrbCmd.Passive(Orbs[0])` N
        // times = +N energy (C# `LoopPower.AfterPlayerTurnStart` +
        // `PlasmaOrb.Passive` GainEnergy(1)). Regression for the `_n` Plasma
        // skip (Defect sweep `PLAYER_STATE|end_turn` seed 1171).
        let mut state = make_state(50, &[100]);
        channel_orb(&mut state, Orb::plasma());
        give_focus(&mut state, 3); // Focus must not scale Plasma.
        let pre = state.player.energy;
        trigger_first_orb_passive_n(&mut state, 2);
        assert_eq!(
            state.player.energy,
            pre + 2,
            "Loop(2) + front Plasma: +2 energy",
        );
        assert_eq!(state.player.orbs.filled_len(), 1, "plasma not consumed");
    }

    #[test]
    fn trigger_first_orb_passive_dark_accumulates() {
        let mut state = make_state(50, &[100]);
        channel_orb(&mut state, Orb::dark());
        give_focus(&mut state, 2);
        trigger_first_orb_passive(&mut state);
        // 6 + 2 focus = 8 stored.
        assert_eq!(state.player.orbs.slots[0], Orb::Dark { stored: 8 });
    }

    #[test]
    fn trigger_first_orb_passive_empty_belt_is_noop() {
        let mut state = make_state(50, &[100]);
        let pre_hp = state.creatures[0].current_hp;
        let pre_block = state.player.block;
        let pre_energy = state.player.energy;
        trigger_first_orb_passive(&mut state);
        assert_eq!(state.creatures[0].current_hp, pre_hp);
        assert_eq!(state.player.block, pre_block);
        assert_eq!(state.player.energy, pre_energy);
    }

    #[test]
    fn trigger_first_orb_passive_frost_scales_with_focus() {
        let mut state = make_state(50, &[100]);
        channel_orb(&mut state, Orb::frost());
        give_focus(&mut state, 4);
        trigger_first_orb_passive(&mut state);
        assert_eq!(state.player.block, 6, "2 base + 4 focus");
    }

    // ── Glass orb (bead sts2-rs-q89c8) ──────────────────────────────────

    #[test]
    fn glass_constructor_starts_at_4() {
        assert_eq!(Orb::glass(), Orb::Glass { current_passive: 4 });
    }

    #[test]
    fn glass_orbkind_discriminant_round_trips() {
        let g = Orb::Glass { current_passive: 3 };
        assert_eq!(OrbKind::of(&g), OrbKind::Glass);
        let mut s = OrbSlots::with_default_capacity();
        s.channel(Orb::glass());
        assert_eq!(s.count_kind(OrbKind::Glass), 1);
        assert_eq!(s.distinct_kinds(), 1);
    }

    #[test]
    fn glass_passive_decays_4_to_0_then_clamps() {
        // Channel a Glass orb; tick passive 6 times against a 100-HP
        // enemy. Per-tick damage: 4, 3, 2, 1, 0, 0 → cumulative 10.
        let mut state = make_state(50, &[100]);
        channel_orb(&mut state, Orb::glass());
        let mut hp = 100;
        for expected in [4, 3, 2, 1, 0, 0] {
            passive_tick(&mut state);
            hp -= expected;
            assert_eq!(
                state.creatures[0].current_hp, hp,
                "tick expected {expected}"
            );
        }
        // Orb still in slot 0 with current_passive = 0.
        assert_eq!(
            state.player.orbs.slots[0],
            Orb::Glass { current_passive: 0 },
        );
    }

    #[test]
    fn glass_passive_uses_focus_for_damage_but_not_decay() {
        // Focus 2: per-tick damage = current_passive + 2; decay still by 1.
        let mut state = make_state(50, &[100]);
        channel_orb(&mut state, Orb::glass());
        give_focus(&mut state, 2);
        // Expected sequence: (4+2)=6, (3+2)=5, (2+2)=4, (1+2)=3, (0+2)=2, 2
        let mut hp = 100;
        for expected in [6, 5, 4, 3, 2, 2] {
            passive_tick(&mut state);
            hp -= expected;
            assert_eq!(
                state.creatures[0].current_hp, hp,
                "tick expected {expected}"
            );
        }
    }

    #[test]
    fn glass_passive_zero_clamped_tick_preserves_value() {
        // C# `GlassOrb.Passive` (GlassOrb.cs:34-39) runs the decay INSIDE the
        // `passiveVal > 0` guard: when Focus drags `current_passive + focus`
        // to/below 0 the tick is fully skipped — no damage AND no decay. A
        // fresh Glass under Focus −5 should deal nothing and KEEP its 4 stacks
        // (the prior code decayed unconditionally → 3, permanently lowering
        // later passive/evoke damage once Focus recovers).
        let mut state = make_state(50, &[100]);
        channel_orb(&mut state, Orb::glass());
        give_focus(&mut state, -5);
        passive_tick(&mut state);
        assert_eq!(
            state.creatures[0].current_hp, 100,
            "no damage on a clamped tick"
        );
        assert_eq!(
            state.player.orbs.slots[0],
            Orb::Glass { current_passive: 4 },
            "a zero-clamped tick must not decay the orb",
        );
        // The single-passive and multi-tick paths share the same guard.
        trigger_first_orb_passive(&mut state);
        trigger_first_orb_passive_n(&mut state, 3);
        assert_eq!(state.creatures[0].current_hp, 100);
        assert_eq!(
            state.player.orbs.slots[0],
            Orb::Glass { current_passive: 4 },
        );
    }

    #[test]
    fn glass_passive_hits_all_alive_enemies() {
        let mut state = make_state(50, &[100, 80, 60]);
        channel_orb(&mut state, Orb::glass());
        passive_tick(&mut state);
        assert_eq!(state.creatures[0].current_hp, 96);
        assert_eq!(state.creatures[1].current_hp, 76);
        assert_eq!(state.creatures[2].current_hp, 56);
    }

    #[test]
    fn glass_evoke_deals_focus_scaled_passive_doubled_to_all() {
        // C# `EvokeVal = (PassiveVal) * 2`, `PassiveVal = _passiveVal + focus`.
        // Fresh Glass (current_passive=4), Focus=1: evoke = (4+1)*2 = 10 AoE
        // (Focus is doubled, applied before the ×2).
        let mut state = make_state(50, &[100, 80]);
        channel_orb(&mut state, Orb::glass());
        give_focus(&mut state, 1);
        evoke_front(&mut state);
        assert_eq!(state.creatures[0].current_hp, 90);
        assert_eq!(state.creatures[1].current_hp, 70);
        assert_eq!(state.player.orbs.filled_len(), 0);
    }

    #[test]
    fn glass_evoke_at_zero_passive_with_focus_0_is_noop() {
        // After full decay, evoke = 0*2 + 0 = 0 → no damage event written.
        let mut state = make_state(50, &[100]);
        state.player.orbs = OrbSlots::with_default_capacity();
        state.player.orbs.slots[0] = Orb::Glass { current_passive: 0 };
        evoke_front(&mut state);
        assert_eq!(state.creatures[0].current_hp, 100);
    }

    #[test]
    fn trigger_first_orb_passive_glass_decays_once() {
        // The rt2ry single-passive path must also decay Glass by 1.
        let mut state = make_state(50, &[100]);
        channel_orb(&mut state, Orb::glass());
        trigger_first_orb_passive(&mut state);
        assert_eq!(state.creatures[0].current_hp, 96);
        assert_eq!(
            state.player.orbs.slots[0],
            Orb::Glass { current_passive: 3 },
        );
    }

    #[test]
    fn trigger_first_orb_passive_n_dark_accumulates_n_times() {
        // Loop(N) on a Dark front orb must compound the stored evoke value
        // (+6 per trigger): 3 triggers → stored 18, not a single +6. Regression
        // for the captured-`orb` bug (SHATTER seed 189: Loop(2) Dark under-dealt
        // by 6). Mirrors the Glass multi-tick test above.
        let mut state = make_state(50, &[100]);
        channel_orb(&mut state, Orb::dark());
        trigger_first_orb_passive_n(&mut state, 3);
        assert_eq!(
            state.player.orbs.slots[0],
            Orb::Dark { stored: 18 },
            "3 × +6 must compound to 18",
        );
    }

    #[test]
    fn trigger_first_orb_passive_n_glass_decays_n_times() {
        // The zlx.4 Loop-style multi-tick path must chain Glass decay
        // (4 → 3 → 2 → 1) and damage (4+3+2+1=10) without re-channeling.
        let mut state = make_state(50, &[100]);
        channel_orb(&mut state, Orb::glass());
        trigger_first_orb_passive_n(&mut state, 4);
        assert_eq!(state.creatures[0].current_hp, 90, "4+3+2+1 = 10 cumulative");
        assert_eq!(
            state.player.orbs.slots[0],
            Orb::Glass { current_passive: 0 },
        );
    }

    #[test]
    fn channel_orb_auto_grows_zero_capacity_belt() {
        // Fixture-friendly: a fresh `CombatState::default()` has an
        // empty (capacity 0) belt. The first `channel_orb` upgrades
        // to the canonical 3-slot baseline so test setups don't have
        // to remember to call `with_default_capacity`.
        let mut state = CombatState::default();
        // Defect player → zero-capacity belt auto-grows to the canonical
        // 3-slot baseline (a non-Defect channeler would get a temporary
        // 1-slot belt instead; see `channel_orb`, bead sts2-rs-dnxjt).
        state.player.character = Some(crate::potions::CharacterId::Defect);
        assert_eq!(state.player.orbs.capacity(), 0);
        channel_orb(&mut state, Orb::lightning());
        assert_eq!(state.player.orbs.capacity(), 3);
        assert_eq!(state.player.orbs.filled_len(), 1);
    }

    /// End-to-end pin for the GH #368.1 forensics: a real evoke through the
    /// real damage pipeline must arrive in the trace *attributed*, not as an
    /// anonymous 6. The whole point of the tool is that "an enemy lost 6 HP"
    /// and "a Dark orb evoked for 6" are different facts.
    #[test]
    fn a_dark_evoke_is_attributed_to_the_orb_not_left_anonymous() {
        crate::damage_trace::set_armed(true);
        let mut state = make_state(50, &[40]);
        channel_orb(&mut state, Orb::dark());
        evoke_front(&mut state);

        let rows = crate::damage_trace::take();
        crate::damage_trace::set_armed(false);

        assert_eq!(rows.len(), 1, "one evoke, one HP-loss row: {rows:?}");
        let r = &rows[0];
        assert_eq!(r.source, "orb:Dark:evoke");
        assert_eq!(r.dealer, "player");
        assert_eq!(r.target_id, "Nibbit");
        // Dark's `_evokeVal` base is 6 and a freshly channelled orb has zero
        // accumulated growth — the exact magnitude this class is named after.
        assert_eq!(r.hp_loss, 6);
        assert_eq!(r.hp_before - r.hp_after, 6);
    }

    /// The passive tick and the evoke must not collapse into one label —
    /// separating them is what would have split the `+6` class in the first
    /// place.
    #[test]
    fn a_lightning_passive_is_labelled_distinctly_from_its_evoke() {
        crate::damage_trace::set_armed(true);
        let mut state = make_state(50, &[40]);
        channel_orb(&mut state, Orb::lightning());
        passive_tick(&mut state);
        evoke_front(&mut state);

        let rows = crate::damage_trace::take();
        crate::damage_trace::set_armed(false);

        let labels: Vec<&str> = rows.iter().map(|r| r.source.as_str()).collect();
        assert_eq!(labels, vec!["orb:Lightning:passive", "orb:Lightning:evoke"]);
        assert_eq!(rows[0].hp_loss, 3);
        assert_eq!(rows[1].hp_loss, 8);
    }
}
