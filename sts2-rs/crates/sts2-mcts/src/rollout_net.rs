//! The playout net: the playout's move choice made by a small trained
//! policy instead of the hand-written priority table in [`crate::expert`].
//!
//! Every simulation scores its tree path by one playout to the end of the
//! fight, so the playout policy is the judge of every candidate move. The
//! heuristic one is an STS1-ported card-priority list that plays out its
//! whole hand every turn; in long boss fights its playouts almost never
//! win, the search gets no win signal, and the pilot plays as if the fight
//! were already lost (GH #777 human-bank audit, 2026-10-18).
//!
//! The built-in Ironclad net (`rl2`) was first fitted to the search's own
//! moves, then trained by REINFORCE on playout outcomes from the starts of
//! human A10 boss and elite fights. Inside the deployed search it raised the
//! Ironclad full-run win rate by 4.67pp pooled over A0, A3, A5, A8, A9 and
//! A10 (1,200 paired runs, McNemar p = 3.6e-4; A9 33.5% → 43.5%).
//! `docs/experiments/2026-10-18-learned-rollout-fullrun-gate.md` has the
//! numbers. The built-in Defect net, trained the same way on Defect fights,
//! raised the Defect A9 win rate from 33.2% to 43.0% (A10 runs past act-3
//! boss 1, 400 paired seeds, McNemar p = 3.1e-4; GH #825,
//! `docs/experiments/defect-playout-net.md`). The built-in Necrobinder net
//! raised its A9 win rate from 9.2% to 15.2% (400 paired seeds, McNemar
//! p = 9.4e-4; GH #823, `docs/experiments/necrobinder-playout-net.md`), and
//! the Silent net from 15.8% to 22.2% (p = 3.8e-3; GH #824,
//! `docs/experiments/silent-playout-net.md`).
//! The Regent net shipped on the owner's call although its gate was NULL
//! (A0 +8.5pp, A9 +3.5pp at p = 0.12; GH #822).
//!
//! The model is a small factorized softmax, cheap enough to run at every
//! playout step:
//!
//! ```text
//! q(s)      = W2 · relu(W1 · s + b1 [+ Σ M[monster]]) + b2   (state query, D dims)
//! k(a)      = E[key(a)] + P · x(a)                           (action key, D dims)
//! logit(a)  = q(s) · k(a) + bias[key(a)] + w · x(a)
//! ```
//!
//! `s` and `x(a)` are fixed feature vectors computed here; `key(a)` is the
//! card (with its upgrade), the potion, `end_turn`, `choice` or `other`, so
//! each card learns its own embedding and bias. The trainer
//! (`sts2-pycli/python/ppo/rollout_learn.py`) reads the same features through
//! the `sts2_rs.rollout_features` binding, so training and playout can never
//! disagree on them.
//!
//! # Feature versions
//!
//! A weights file declares the [`FeatureVersion`] its features were computed
//! under (`feature_version`, absent = 1), and its `state_dim` / `action_dim`
//! must be that version's widths.
//!
//! - **v1** ([`STATE_DIM`] × [`ACTION_DIM`] = 32 × 24): the deployed net's
//!   features. A v1 file runs exactly the code it was trained under: every
//!   v1 feature, dot product and sum is computed in the same order as before
//!   v2 existed, so its logits are bit-identical (pinned by
//!   `tests/rollout_net_v1_golden.rs`).
//! - **v2** ([`STATE_DIM_V2`] × [`ACTION_DIM_V2`] = 96 × 32): v1 plus the
//!   boss flag, the act, the intent type of the first three enemies in play
//!   (alive, or a beaten sentinel such as the exploding Waterfall Giant),
//!   high-impact enemy powers (Artifact, Thorns, Plating, Ritual,
//!   Intangible, Slippery, Hardened Shell, Reflect, Curl Up, Enrage, sleep,
//!   minions, Vulnerable / Weak amounts) and player powers (Demon Form, Feel
//!   No Pain, Barricade, Corruption, Rage, Metallicize, Dark Embrace,
//!   Juggernaut, Inferno, Rupture, Aggression, Crimson Mantle, Flame Barrier,
//!   Vigor, Plating, Artifact, Buffer, Intangible, Thorns, Regen, No Draw,
//!   Entangled, temporary Strength); per action, whether the card exhausts
//!   (Corruption included) or is Ethereal, and the target's Intangible,
//!   Thorns, Slippery, Artifact, attack intent and block. The index of every
//!   feature is the `V2_*` constants below.
//!
//!   The act is NOT `CombatState::act_index`: a fight built by
//!   `build_scenario_state` (the human bank, the combat bench) always has
//!   `act_index = 1`, so a feature read from it would be constant in
//!   training and vary in deployment. A v2 file instead carries
//!   `monster_act` (species → the lowest act any encounter lists it in, via
//!   `sts2_rs.monster_acts()`) and, optionally, `encounter_act` (each
//!   encounter's `[act, roster]`, via `sts2_rs.encounter_acts()`). The act is
//!   the highest `monster_act` among every enemy the fight has spawned,
//!   raised to an encounter's act when its whole roster has been spawned: two
//!   constructs are act 1 alone and act 3 as the Construct Menagerie. The
//!   optional `monster_keys` / `monster_emb` table adds a learned per-species
//!   vector to the hidden layer for each of the first [`MONSTER_SLOTS`]
//!   enemies in play: an encounter-identity embedding.
//! - **v3** ("v1e", [`STATE_DIM`] × [`ACTION_DIM`], v1's layout): every v1
//!   feature at its v1 index, read the way the game shows the fight (GH
//!   #830 boss audit). A telegraphed attack is its displayed number
//!   ([`crate::enemy_eval::IncomingDamage::Effective`]: the dealer's
//!   Strength / Weak, the player's Vulnerable / Intangible, a back attack's
//!   ×1.5) wherever v1 reads one: `incoming` (state 10), `unblocked` (11,
//!   and action features 20 and 23), the incoming-lethal flag (13), the
//!   target's intent (action 18) and the heuristic-priority feature (21,
//!   whose HP-aware arms read incoming). `removed` (state 15) counts the
//!   forms an enemy still has to go through
//!   ([`crate::enemy_eval::enemy_hp_remaining`]). Because the layout is
//!   v1's, a v1 file (the built-in `rl2`) loads as v3 by changing only its
//!   `feature_version`; v1 files keep computing the v1 numbers.
//! - **v4** ([`STATE_DIM_V4`] × [`ACTION_DIM_V4`]): v1's features, read as
//!   v1 reads them, then a character and affliction block (GH #822
//!   follow-up: Regent's elite deaths ran 4× the humans' on a faithful
//!   simulation). The state adds Regent's Stars and Star powers, the
//!   player's Tainted, the enemies' Personal Hive and Vital Spark, and the
//!   afflicted cards in hand ([`V4_STARS`] on). A card play adds its Star
//!   cost, the Stars left after it, its Tainted stacks or other affliction,
//!   and an Attack target's Personal Hive ([`V4_X_STAR_COST`] on). A v1
//!   file zero-padded to these widths computes the v1 net's logits.
//!
//! Which weights a playout uses ([`net_for`]):
//! - `STS2_ROLLOUT_NET=<json>`, when set, for every character. This is the
//!   experiment path, and it fails loudly if the file does not load.
//! - Otherwise the built-in net of the fight's character
//!   (`sts2-rs/data/rollout_nets/<character>.json`, compiled in when present).
//!   Every character has one.
//! - A character with neither plays the heuristic playout unchanged.
//!
//! `STS2_ROLLOUT_TEMP` and `STS2_ROLLOUT_NO_EARLY_END` override the softmax
//! temperature and the early-EndTurn rule of the `STS2_ROLLOUT_NET` net only.
//! A built-in net always plays with the values in its file, so a variable
//! left over from an experiment cannot silently change the deployed search.
//! Each net is loaded once per process.
//!
//! A built-in net is compiled in only when its file exists
//! (`cfg(rollout_net_<character>)`, set by `build.rs`). A checkout without the
//! file still builds and plays the heuristic playout.
//!
//! The same format also carries a move prior ([`RolloutNetPrior`]): the
//! built-in Ironclad one (`sts2-rs/data/prior_nets/ironclad.json`,
//! `cfg(prior_net_ironclad)`, GH #830) is [`builtin_prior_json`]. It is not a
//! playout net, and [`net_for`] never returns it.
#![allow(clippy::cast_precision_loss)]

use std::sync::OnceLock;

use rand::Rng;
use smallvec::SmallVec;
use sts2_core::card_afflictions::CardAfflictionKind;
use sts2_core::cards::{CardId, CardType};
use sts2_core::potions::PotionKind;
use sts2_core::{
    Action, CharacterId, CombatState, IntentKindTag, MonsterKind, PowerKind, PowerKindSet,
};

use crate::enemy_eval::IncomingDamage;
use crate::{LeafEval, MctsConfig};

/// Width of the version-1 state feature vector `s` (the deployed net's).
pub const STATE_DIM: usize = 32;
/// Width of the version-1 per-action feature vector `x(a)`.
pub const ACTION_DIM: usize = 24;
/// Width of the version-2 state feature vector.
pub const STATE_DIM_V2: usize = 96;
/// Width of the version-2 per-action feature vector.
pub const ACTION_DIM_V2: usize = 32;
/// The widest state vector any version computes (the buffer size).
pub const STATE_DIM_MAX: usize = STATE_DIM_V2;
/// The widest action vector any version computes (the buffer size).
pub const ACTION_DIM_MAX: usize = ACTION_DIM_V2;
/// Living enemies whose species feed the v2 encounter embedding.
pub const MONSTER_SLOTS: usize = 5;

// ── v2 state feature indices ─────────────────────────────────────────────
/// `CombatState::is_boss`.
pub const V2_IS_BOSS: usize = 32;
/// The fight's act, / 3: the highest `monster_act` among every enemy the
/// fight has spawned (dead ones included), raised to an `encounter_act`
/// roster's act when that whole roster has been spawned.
pub const V2_ACT: usize = 33;
/// First of three 8-wide intent blocks (enemies in play, i.e. alive or a
/// beaten sentinel, in `enemy_uids` order): attack (single or death blow),
/// multi-attack, defend, buff, debuff, status card, other (sleep / stun /
/// unknown / summon / heal / escape), and the intent's total damage / max HP.
pub const V2_INTENTS: usize = 34;
/// Width of one intent block.
pub const V2_INTENT_WIDTH: usize = 8;
/// Enemies in play past the third, / 3.
pub const V2_EXTRA_ENEMIES: usize = 58;
/// First enemy-power aggregate (summed over the enemies in play): Artifact /3,
/// Thorns /5, Plating /10, Ritual /5, then flags Intangible, Slippery,
/// Hardened Shell, Reflect, Curl Up, Enrage, asleep (Asleep / Slumber), the
/// minion share of the enemies in play, and the Vulnerable /5 and Weak /5
/// amounts.
pub const V2_ENEMY_POWERS: usize = 59;
/// First player power, in [`PLAYER_V2_POWERS`] order.
pub const V2_PLAYER_POWERS: usize = 73;
/// v2 action features: the card exhausts on play (its keyword, or a Skill
/// under Corruption).
pub const V2_X_EXHAUSTS: usize = 24;
/// The card is Ethereal.
pub const V2_X_ETHEREAL: usize = 25;
/// First target fact: Intangible, Thorns /5, Slippery, Artifact, intends to
/// attack, block /20.
pub const V2_X_TARGET: usize = 26;

/// Width of the version-4 state feature vector: v1's 32, then the character
/// and affliction block from [`V4_STARS`].
pub const STATE_DIM_V4: usize = 44;
/// Width of the version-4 per-action feature vector: v1's 24, then the card
/// facts from [`V4_X_STAR_COST`].
pub const ACTION_DIM_V4: usize = 29;

// ── v4 state feature indices (after v1's 32) ─────────────────────────────
/// Regent's Stars /5, then the Stars gained this turn /5, the player's Star
/// Next Turn, Child of the Stars, Black Hole, Furnace and Genesis /3, and a
/// Void Form flag.
pub const V4_STARS: usize = 32;
/// The player's Tainted /5: each powered attack on them deals that much more
/// (`TaintedPower.cs`).
pub const V4_TAINTED: usize = 40;
/// The Personal Hive on enemies in play /3: a Dazed into the draw pile per
/// powered hit on its owner (`PersonalHivePower.cs`).
pub const V4_HIVE: usize = 41;
/// An enemy in play has Vital Spark: playing a Tainted card applies Tainted
/// to the player (`VitalSparkPower.cs`).
pub const V4_VITAL_SPARK: usize = 42;
/// Afflicted cards in hand /5.
pub const V4_AFFLICTED_HAND: usize = 43;
/// v4 action features: the card's Star cost /3, then the Stars left after
/// paying it /5.
pub const V4_X_STAR_COST: usize = 24;
/// The card's Tainted stacks /5.
pub const V4_X_TAINTED: usize = 26;
/// The card carries an affliction other than Tainted (Bound, Devoured, ...).
pub const V4_X_AFFLICTED: usize = 27;
/// An Attack's target's Personal Hive /3.
pub const V4_X_TARGET_HIVE: usize = 28;

/// The player powers in the v4 block after the two Star counts, each /3.
const PLAYER_V4_POWERS: [PowerKind; 5] = [
    PowerKind::StarNextTurn,
    PowerKind::ChildOfTheStars,
    PowerKind::BlackHole,
    PowerKind::Furnace,
    PowerKind::Genesis,
];

const PLAYER_V4_MASK: PowerKindSet = PowerKindSet::from_kinds(&[
    PowerKind::StarNextTurn,
    PowerKind::ChildOfTheStars,
    PowerKind::BlackHole,
    PowerKind::Furnace,
    PowerKind::Genesis,
    PowerKind::VoidForm,
    PowerKind::Tainted,
]);

const ENEMY_V4_MASK: PowerKindSet =
    PowerKindSet::from_kinds(&[PowerKind::PersonalHive, PowerKind::VitalSpark]);

/// Player powers the v2 state reads, each with the divisor its amount is
/// scaled by (`0` = a presence flag). The order is the feature order from
/// [`V2_PLAYER_POWERS`].
pub const PLAYER_V2_POWERS: [(PowerKind, f32); 23] = [
    (PowerKind::DemonForm, 3.0),
    (PowerKind::FeelNoPain, 5.0),
    (PowerKind::Barricade, 0.0),
    (PowerKind::Corruption, 0.0),
    (PowerKind::Rage, 5.0),
    (PowerKind::Metallicize, 5.0),
    (PowerKind::DarkEmbrace, 2.0),
    (PowerKind::Juggernaut, 5.0),
    (PowerKind::Inferno, 3.0),
    (PowerKind::Rupture, 2.0),
    (PowerKind::Aggression, 0.0),
    (PowerKind::CrimsonMantle, 0.0),
    (PowerKind::FlameBarrier, 5.0),
    (PowerKind::Vigor, 10.0),
    (PowerKind::Plating, 10.0),
    (PowerKind::Artifact, 3.0),
    (PowerKind::Buffer, 2.0),
    (PowerKind::Intangible, 0.0),
    (PowerKind::Thorns, 5.0),
    (PowerKind::Regen, 5.0),
    (PowerKind::NoDraw, 0.0),
    (PowerKind::Entangled, 0.0),
    (PowerKind::TemporaryStrength, 5.0),
];

const PLAYER_V2_MASK: PowerKindSet = PowerKindSet::from_kinds(&[
    PowerKind::DemonForm,
    PowerKind::FeelNoPain,
    PowerKind::Barricade,
    PowerKind::Corruption,
    PowerKind::Rage,
    PowerKind::Metallicize,
    PowerKind::DarkEmbrace,
    PowerKind::Juggernaut,
    PowerKind::Inferno,
    PowerKind::Rupture,
    PowerKind::Aggression,
    PowerKind::CrimsonMantle,
    PowerKind::FlameBarrier,
    PowerKind::Vigor,
    PowerKind::Plating,
    PowerKind::Artifact,
    PowerKind::Buffer,
    PowerKind::Intangible,
    PowerKind::Thorns,
    PowerKind::Regen,
    PowerKind::NoDraw,
    PowerKind::Entangled,
    PowerKind::TemporaryStrength,
]);

const ENEMY_V2_MASK: PowerKindSet = PowerKindSet::from_kinds(&[
    PowerKind::Artifact,
    PowerKind::Thorns,
    PowerKind::Plating,
    PowerKind::Ritual,
    PowerKind::Intangible,
    PowerKind::Slippery,
    PowerKind::HardenedShell,
    PowerKind::Reflect,
    PowerKind::CurlUp,
    PowerKind::Enrage,
    PowerKind::Asleep,
    PowerKind::Slumber,
    PowerKind::Minion,
    PowerKind::Vulnerable,
    PowerKind::Weak,
]);

/// Which feature set a weights file was trained on. See the module docs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FeatureVersion {
    /// 32 state / 24 action features (the deployed net).
    V1,
    /// 96 state / 32 action features, with per-species tables.
    V2,
    /// "v1e": v1's 32 / 24 layout, with every telegraphed attack read at
    /// its displayed number and "enemy HP removed" counting the forms still
    /// to come.
    V3,
    /// v1's 32 / 24 features read as v1 reads them, then the character and
    /// affliction block: 44 state / 29 action features (Regent's Stars and
    /// Star powers, Tainted, Personal Hive, Vital Spark, card afflictions).
    V4,
}

impl FeatureVersion {
    /// The version numbered `n` in a weights file.
    #[must_use]
    pub const fn from_number(n: u32) -> Option<Self> {
        match n {
            1 => Some(Self::V1),
            2 => Some(Self::V2),
            3 => Some(Self::V3),
            4 => Some(Self::V4),
            _ => None,
        }
    }

    /// The number a weights file writes for this version.
    #[must_use]
    pub const fn number(self) -> u32 {
        match self {
            Self::V1 => 1,
            Self::V2 => 2,
            Self::V3 => 3,
            Self::V4 => 4,
        }
    }

    /// Width of this version's state vector.
    #[must_use]
    pub const fn state_dim(self) -> usize {
        match self {
            Self::V1 | Self::V3 => STATE_DIM,
            Self::V2 => STATE_DIM_V2,
            Self::V4 => STATE_DIM_V4,
        }
    }

    /// Width of this version's action vector.
    #[must_use]
    pub const fn action_dim(self) -> usize {
        match self {
            Self::V1 | Self::V3 => ACTION_DIM,
            Self::V2 => ACTION_DIM_V2,
            Self::V4 => ACTION_DIM_V4,
        }
    }

    /// How this version reads a telegraphed attack: displayed under v3, the
    /// move table's base number under v1 and v2.
    #[must_use]
    pub const fn incoming(self) -> IncomingDamage {
        match self {
            Self::V3 => IncomingDamage::Effective,
            Self::V1 | Self::V2 | Self::V4 => IncomingDamage::Base,
        }
    }

    /// Whether this version's "enemy HP removed" counts the forms still to
    /// come ([`crate::enemy_eval::enemy_hp_remaining`]); v3 only.
    #[must_use]
    pub const fn removed_counts_forms(self) -> bool {
        matches!(self, Self::V3)
    }
}

/// What the features of one [`FeatureVersion`] read beyond the state: the
/// per-species act table, the encounter rosters that refine it and the
/// encounter-embedding rows (v2 only).
#[derive(Clone, Debug)]
pub struct FeatureSpec {
    /// The feature version.
    pub version: FeatureVersion,
    /// Act of each species, indexed by `MonsterKind as usize`; 0 = unknown.
    act_by_kind: Vec<u8>,
    /// Encounter-embedding row (1-based) of each species; 0 = none.
    row_by_kind: Vec<u16>,
    /// Encounters whose act is above every member's [`Self::act`] (a
    /// species shared with an earlier act's encounters, e.g. the act-3
    /// Construct Menagerie of two act-1 constructs): `(act, roster as
    /// (species, count))`. The act feature takes such an encounter's act
    /// when its whole roster is among the fight's enemies.
    rosters: Vec<(u8, SmallVec<[(MonsterKind, u8); 4]>)>,
    /// Whether a species is a member of any of `rosters`, by
    /// `MonsterKind as usize` (so a fight without one skips the matching).
    in_roster: Vec<bool>,
}

/// The v1 spec (no tables), shared by every v1 net and [`StepCtx::new`].
pub const FEATURES_V1: FeatureSpec = FeatureSpec {
    version: FeatureVersion::V1,
    act_by_kind: Vec::new(),
    row_by_kind: Vec::new(),
    rosters: Vec::new(),
    in_roster: Vec::new(),
};

/// The v3 spec (no tables), shared by every v3 net.
pub const FEATURES_V3: FeatureSpec = FeatureSpec {
    version: FeatureVersion::V3,
    act_by_kind: Vec::new(),
    row_by_kind: Vec::new(),
    rosters: Vec::new(),
    in_roster: Vec::new(),
};

/// The v4 spec (no tables), shared by every v4 net.
pub const FEATURES_V4: FeatureSpec = FeatureSpec {
    version: FeatureVersion::V4,
    act_by_kind: Vec::new(),
    row_by_kind: Vec::new(),
    rosters: Vec::new(),
    in_roster: Vec::new(),
};

/// `name → species` for every species this build knows.
fn monster_by_name() -> std::collections::HashMap<&'static str, MonsterKind> {
    MonsterKind::ALL_KNOWN
        .iter()
        .map(|k| (k.python_name(), *k))
        .collect()
}

impl FeatureSpec {
    /// A v2 spec from `(species, act)` pairs and `(species, embedding row)`
    /// pairs (rows are 1-based; pass none for a net without the embedding).
    #[must_use]
    pub fn v2(acts: &[(MonsterKind, u8)], rows: &[(MonsterKind, u16)]) -> Self {
        let mut act_by_kind = Vec::new();
        for &(k, a) in acts {
            let i = k as usize;
            if act_by_kind.len() <= i {
                act_by_kind.resize(i + 1, 0);
            }
            act_by_kind[i] = a;
        }
        let mut row_by_kind = Vec::new();
        for &(k, r) in rows {
            let i = k as usize;
            if row_by_kind.len() <= i {
                row_by_kind.resize(i + 1, 0);
            }
            row_by_kind[i] = r;
        }
        Self {
            version: FeatureVersion::V2,
            act_by_kind,
            row_by_kind,
            rosters: Vec::new(),
            in_roster: Vec::new(),
        }
    }

    /// This spec with the encounter rosters `(act, species)` (one entry per
    /// monster, so a roster of two Cubex Constructs lists it twice). Only
    /// the rosters the species table alone would put in a lower act are
    /// kept: for every other encounter, the highest species act already is
    /// the encounter's act.
    #[must_use]
    pub fn with_encounters(mut self, rosters: &[(u8, Vec<MonsterKind>)]) -> Self {
        self.rosters.clear();
        self.in_roster.clear();
        for (act, members) in rosters {
            let species_act = members.iter().map(|k| self.act(*k)).max().unwrap_or(0);
            if *act <= species_act {
                continue;
            }
            let mut counts: SmallVec<[(MonsterKind, u8); 4]> = SmallVec::new();
            for &k in members {
                match counts.iter_mut().find(|(c, _)| *c == k) {
                    Some((_, n)) => *n = n.saturating_add(1),
                    None => counts.push((k, 1)),
                }
                let i = k as usize;
                if self.in_roster.len() <= i {
                    self.in_roster.resize(i + 1, false);
                }
                self.in_roster[i] = true;
            }
            self.rosters.push((*act, counts));
        }
        self
    }

    /// The act table this spec was built with, as `(species name, act)`.
    #[must_use]
    pub fn monster_acts(&self) -> Vec<(&'static str, u8)> {
        MonsterKind::ALL_KNOWN
            .iter()
            .filter_map(|k| {
                let a = self.act(*k);
                (a > 0).then(|| (k.python_name(), a))
            })
            .collect()
    }

    fn act(&self, k: MonsterKind) -> u8 {
        self.act_by_kind.get(k as usize).copied().unwrap_or(0)
    }

    fn row(&self, k: MonsterKind) -> u16 {
        self.row_by_kind.get(k as usize).copied().unwrap_or(0)
    }

    fn is_roster_member(&self, k: MonsterKind) -> bool {
        self.in_roster.get(k as usize).copied().unwrap_or(false)
    }

    /// The highest act of a kept roster contained in `present` (species and
    /// how many of each the fight has spawned), 0 when none is.
    fn roster_act(&self, present: &[(MonsterKind, u8)]) -> u8 {
        self.rosters
            .iter()
            .filter(|(_, members)| {
                members.iter().all(|(k, n)| {
                    present
                        .iter()
                        .find(|(p, _)| p == k)
                        .is_some_and(|(_, have)| have >= n)
                })
            })
            .map(|(act, _)| *act)
            .max()
            .unwrap_or(0)
    }
}

/// What an action's embedding row is keyed on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ActionKey {
    /// Playing this card (upgrade is part of the identity).
    Card { id: CardId, upgraded: bool },
    /// Drinking this potion.
    Potion(PotionKind),
    /// Ending the turn.
    EndTurn,
    /// Resolving a pending choice.
    Choice,
    /// Anything else (`Channel` / `Evoke`, never emitted as legal today).
    Other,
}

impl ActionKey {
    /// The key's name in the weights file and the training data.
    #[must_use]
    pub fn name(self) -> String {
        match self {
            Self::Card { id, upgraded } => {
                format!(
                    "card:{}{}",
                    id.to_python_name(),
                    if upgraded { "+" } else { "" }
                )
            }
            Self::Potion(k) => format!("potion:{}", k.python_name()),
            Self::EndTurn => "end_turn".to_owned(),
            Self::Choice => "choice".to_owned(),
            Self::Other => "other".to_owned(),
        }
    }
}

/// What an attack or potion aimed at one enemy slot sees there (v2).
#[derive(Clone, Copy, Debug, Default)]
struct TargetFacts {
    intangible: bool,
    thorns: i32,
    slippery: bool,
    artifact: bool,
    attacking: bool,
    block: i32,
}

/// Per-step context shared by every action's features.
#[derive(Clone, Debug)]
pub struct StepCtx {
    /// The state feature vector `s`; only the version's first
    /// `state_dim` entries are meaningful (the rest stay 0).
    pub state: [f32; STATE_DIM_MAX],
    /// Encounter-embedding rows of the first [`MONSTER_SLOTS`] living
    /// enemies (0 = no row; always 0 under v1).
    pub monsters: [u16; MONSTER_SLOTS],
    /// Those enemies' species (v2; `None` under v1).
    pub species: [Option<MonsterKind>; MONSTER_SLOTS],
    version: FeatureVersion,
    max_hp: i32,
    energy: i32,
    unblocked: i32,
    strength: i32,
    player_weak: bool,
    lowest_slot: Option<u8>,
    alive: u32,
    playable: u32,
    /// Whether the enemy in each `enemy_uids` slot is Vulnerable, so an
    /// attack's lethal check reads it instead of rescanning the powers.
    enemy_vuln: SmallVec<[bool; 6]>,
    /// v2: what each `enemy_uids` slot shows a targeted action.
    targets: SmallVec<[TargetFacts; 6]>,
    /// v2: the player has Corruption (Skills exhaust).
    corruption: bool,
    /// v4: the player's Stars.
    stars: i32,
    /// v4: the Personal Hive on each `enemy_uids` slot (empty when none is
    /// in play).
    hive: SmallVec<[i32; 6]>,
}

fn clampf(x: f32, lo: f32, hi: f32) -> f32 {
    x.clamp(lo, hi)
}

/// The `(Vulnerable, Weak, Strength)` amounts on enemy creature `c`, 0 when
/// absent, in one pass over its powers.
fn creature_powers(state: &CombatState, c: &sts2_core::Creature) -> (i32, i32, i32) {
    let present = state.powers.present_kinds();
    if !(present.contains(PowerKind::Vulnerable)
        || present.contains(PowerKind::Weak)
        || present.contains(PowerKind::Strength))
    {
        return (0, 0, 0);
    }
    let (mut vuln, mut weak, mut strength) = (0, 0, 0);
    for p in c
        .powers
        .iter()
        .filter_map(|pu| state.powers.iter().find(|p| p.id == pu.0))
    {
        match p.kind {
            PowerKind::Vulnerable => vuln += p.amount,
            PowerKind::Weak => weak += p.amount,
            PowerKind::Strength => strength += p.amount,
            _ => {}
        }
    }
    (vuln, weak, strength)
}

impl StepCtx {
    /// Build the v1 state features for `state`, whose legal actions are
    /// `actions` (only their count of card plays is read).
    #[must_use]
    pub fn new(state: &CombatState, actions: &[Action]) -> Self {
        Self::with_spec(state, actions, &FEATURES_V1)
    }

    /// Build the state features of `spec`'s version.
    #[must_use]
    pub fn with_spec(state: &CombatState, actions: &[Action], spec: &FeatureSpec) -> Self {
        let mut ctx = Self::v1(state, actions, spec.version);
        match spec.version {
            FeatureVersion::V2 => ctx.fill_v2(state, spec),
            FeatureVersion::V4 => ctx.fill_v4(state),
            FeatureVersion::V1 | FeatureVersion::V3 => {}
        }
        ctx
    }

    /// The v1 features, computed exactly as before feature versions existed,
    /// or under `V3` their "v1e" reading: the same 32 values except that the
    /// telegraphed attacks (`incoming`, `unblocked`, the incoming-lethal
    /// flag) are the displayed numbers and `removed` counts the forms still
    /// to come.
    fn v1(state: &CombatState, actions: &[Action], version: FeatureVersion) -> Self {
        let read = version.incoming();
        let p = &state.player;
        let hp = p.current_hp.max(0);
        let max_hp = p.max_hp.max(1);
        let block = p.block.max(0);
        let energy = i32::from(p.energy.max(0));
        let mut incoming = 0i32;
        let (mut hp_sum, mut max_sum, mut block_sum, mut str_sum) = (0i32, 0i32, 0i32, 0i32);
        let mut min_hp = i32::MAX;
        let (mut alive, mut lowest_hp, mut lowest_slot) = (0u32, i32::MAX, None);
        let (mut any_vuln, mut any_weak, mut any_sentinel) = (false, false, false);
        let mut enemy_vuln = SmallVec::new();
        for (slot, uid) in state.enemy_uids.iter().enumerate() {
            enemy_vuln.push(false);
            let Some(c) = sts2_core::creature_by_id(&state.creatures, uid.0) else {
                continue;
            };
            if c.is_beaten_sentinel() {
                any_sentinel = true;
                max_sum += c.scored_hp(state.ascension).1;
                incoming += read.total(state, c);
                continue;
            }
            hp_sum += c.current_hp.max(0);
            max_sum += c.max_hp.max(0);
            let (vuln, weak, strength) = creature_powers(state, c);
            enemy_vuln[slot] = vuln > 0;
            if c.current_hp <= 0 {
                continue;
            }
            incoming += read.total(state, c);
            alive += 1;
            block_sum += c.block.max(0);
            min_hp = min_hp.min(c.current_hp);
            if c.current_hp < lowest_hp {
                lowest_hp = c.current_hp;
                lowest_slot = u8::try_from(slot).ok();
            }
            any_vuln |= vuln > 0;
            any_weak |= weak > 0;
            str_sum += strength;
        }
        if min_hp == i32::MAX {
            min_hp = 0;
        }
        let unblocked = (incoming - block).max(0);
        let strength = state.player_power_amount(PowerKind::Strength);
        let dexterity = state.player_power_amount(PowerKind::Dexterity);
        let player_weak = state.player_power_amount(PowerKind::Weak) > 0;
        let playable = u32::try_from(
            actions
                .iter()
                .filter(|a| matches!(a, Action::PlayCard { .. }))
                .count(),
        )
        .unwrap_or(u32::MAX);
        let mhp = max_hp as f32;
        let (removed_hp, removed_max) = if version.removed_counts_forms() {
            crate::enemy_eval::enemy_hp_remaining(state, true)
        } else {
            (hp_sum, max_sum)
        };
        let removed = if removed_max > 0 {
            1.0 - removed_hp as f32 / removed_max as f32
        } else {
            1.0
        };
        let b = |x: bool| if x { 1.0 } else { 0.0 };
        let s = [
            hp as f32 / mhp,
            mhp / 100.0,
            clampf(block as f32 / mhp, 0.0, 2.0),
            energy as f32 / 3.0,
            f32::from(p.max_energy.max(0)) / 3.0,
            p.hand.len() as f32 / 10.0,
            p.draw.len() as f32 / 30.0,
            p.discard.len() as f32 / 30.0,
            p.exhaust.len() as f32 / 30.0,
            clampf(f32::from(state.round) / 20.0, 0.0, 2.0),
            clampf(incoming as f32 / mhp, 0.0, 3.0),
            clampf(unblocked as f32 / mhp, 0.0, 3.0),
            b(incoming > 0),
            b(incoming > 0 && unblocked >= hp),
            alive as f32 / 3.0,
            removed,
            clampf(hp_sum as f32 / 200.0, 0.0, 3.0),
            clampf(min_hp as f32 / 100.0, 0.0, 3.0),
            clampf(block_sum as f32 / 30.0, 0.0, 3.0),
            clampf(strength as f32 / 5.0, -3.0, 3.0),
            clampf(dexterity as f32 / 5.0, -3.0, 3.0),
            b(player_weak),
            b(state.player_power_amount(PowerKind::Vulnerable) > 0),
            b(state.player_power_amount(PowerKind::Frail) > 0),
            b(any_vuln),
            b(any_weak),
            clampf(str_sum as f32 / 5.0, -3.0, 3.0),
            p.potions.len() as f32 / 3.0,
            b(state.is_elite),
            b(any_sentinel),
            playable as f32 / 10.0,
            1.0,
        ];
        let mut state_vec = [0.0f32; STATE_DIM_MAX];
        state_vec[..STATE_DIM].copy_from_slice(&s);
        Self {
            state: state_vec,
            monsters: [0; MONSTER_SLOTS],
            species: [None; MONSTER_SLOTS],
            version,
            max_hp,
            energy,
            unblocked,
            strength,
            player_weak,
            lowest_slot,
            alive,
            playable,
            enemy_vuln,
            targets: SmallVec::new(),
            corruption: false,
            stars: 0,
            hive: SmallVec::new(),
        }
    }

    /// Append the v4 block (indices `V4_*`) to a v1 context. The player
    /// powers and the enemy pass are skipped outright when no power kind they
    /// read is live anywhere (`present_kinds`); an enemy counts while it is
    /// in play, as in [`Self::fill_v2`].
    fn fill_v4(&mut self, state: &CombatState) {
        let p = &state.player;
        let present = state.powers.present_kinds();
        let s = &mut self.state;
        self.stars = i32::from(p.stars.max(0));
        s[V4_STARS] = clampf(self.stars as f32 / 5.0, 0.0, 3.0);
        s[V4_STARS + 1] = clampf(p.stars_gained_this_turn as f32 / 5.0, 0.0, 3.0);
        if present.intersects(&PLAYER_V4_MASK) {
            for (i, k) in PLAYER_V4_POWERS.iter().enumerate() {
                s[V4_STARS + 2 + i] = clampf(state.player_power_amount(*k) as f32 / 3.0, -3.0, 3.0);
            }
            s[V4_STARS + 7] = if state.player_power_amount(PowerKind::VoidForm) > 0 {
                1.0
            } else {
                0.0
            };
            s[V4_TAINTED] = clampf(
                state.player_power_amount(PowerKind::Tainted) as f32 / 5.0,
                0.0,
                3.0,
            );
        }
        if present.intersects(&ENEMY_V4_MASK) {
            let (mut hive_sum, mut spark) = (0i32, false);
            for uid in &state.enemy_uids {
                let mut hive = 0;
                if let Some(c) = sts2_core::creature_by_id(&state.creatures, uid.0) {
                    if c.is_beaten_sentinel() || c.current_hp > 0 {
                        for pw in c
                            .powers
                            .iter()
                            .filter_map(|pu| state.powers.get_by_id(pu.0))
                        {
                            match pw.kind {
                                PowerKind::PersonalHive => hive += pw.amount,
                                PowerKind::VitalSpark => spark = true,
                                _ => {}
                            }
                        }
                    }
                }
                hive_sum += hive;
                self.hive.push(hive);
            }
            s[V4_HIVE] = clampf(hive_sum as f32 / 3.0, 0.0, 3.0);
            s[V4_VITAL_SPARK] = if spark { 1.0 } else { 0.0 };
        }
        let afflicted = p
            .hand
            .iter()
            .filter_map(|uid| state.cards.get(uid.0.get() as usize - 1))
            .filter(|c| c.affliction.is_some())
            .count();
        s[V4_AFFLICTED_HAND] = clampf(afflicted as f32 / 5.0, 0.0, 2.0);
        self.version = FeatureVersion::V4;
    }

    /// Append the v2 features (indices `V2_*`) to a v1 context: one pass over
    /// the enemies, one over the player's powers, each skipped outright when
    /// no power kind it reads is live anywhere (`present_kinds`).
    ///
    /// An enemy is in play while it is alive or a beaten sentinel: the
    /// Waterfall Giant at its 999,999,999 HP sentinel still acts
    /// (`WaterfallGiant.cs` `TriggerAboutToBlowState`: ABOUT_TO_BLOW, then
    /// EXPLODE's DeathBlow), so it keeps its intent block, species slot and
    /// target facts. The act reads every enemy the fight has spawned, dead
    /// ones included (`enemy_uids` never drops one), so a summon outliving
    /// its summoner keeps the encounter's act.
    fn fill_v2(&mut self, state: &CombatState, spec: &FeatureSpec) {
        let b = |x: bool| if x { 1.0 } else { 0.0 };
        let mhp = self.max_hp as f32;
        let present = state.powers.present_kinds();
        let enemy_powers = present.intersects(&ENEMY_V2_MASK);
        let s = &mut self.state;
        s[V2_IS_BOSS] = b(state.is_boss);
        let (mut act, mut in_play, mut minions, mut intent_slot, mut mon_slot) =
            (0u8, 0u32, 0u32, 0, 0);
        let (mut artifact, mut thorns, mut plating, mut ritual, mut vuln, mut weak) =
            (0i32, 0i32, 0i32, 0i32, 0i32, 0i32);
        let mut flags = [false; 7]; // intangible, slippery, hardened, reflect, curl, enrage, asleep

        // Spawned roster members (species, count), for `spec.roster_act`.
        let mut spawned: SmallVec<[(MonsterKind, u8); 4]> = SmallVec::new();
        for uid in &state.enemy_uids {
            self.targets.push(TargetFacts::default());
            let Some(c) = sts2_core::creature_by_id(&state.creatures, uid.0) else {
                continue;
            };
            if let Some(k) = c.species {
                act = act.max(spec.act(k));
                if spec.is_roster_member(k) {
                    match spawned.iter_mut().find(|(p, _)| *p == k) {
                        Some((_, n)) => *n = n.saturating_add(1),
                        None => spawned.push((k, 1)),
                    }
                }
            }
            if !c.is_beaten_sentinel() && c.current_hp <= 0 {
                continue;
            }
            in_play += 1;
            if let Some(k) = c.species {
                if mon_slot < MONSTER_SLOTS {
                    self.monsters[mon_slot] = spec.row(k);
                    self.species[mon_slot] = Some(k);
                    mon_slot += 1;
                }
            }
            let kind = c.intent.map(sts2_core::Intent::kind);
            let mut facts = TargetFacts {
                block: c.block.max(0),
                attacking: matches!(
                    kind,
                    Some(
                        IntentKindTag::Attack
                            | IntentKindTag::MultiAttack
                            | IntentKindTag::DeathBlow
                    )
                ),
                ..TargetFacts::default()
            };
            let mut minion = false;
            if enemy_powers {
                for p in c
                    .powers
                    .iter()
                    .filter_map(|pu| state.powers.get_by_id(pu.0))
                {
                    let a = p.amount;
                    match p.kind {
                        PowerKind::Artifact => {
                            artifact += a;
                            facts.artifact |= a > 0;
                        }
                        PowerKind::Thorns => {
                            thorns += a;
                            facts.thorns += a;
                        }
                        PowerKind::Plating => plating += a,
                        PowerKind::Ritual => ritual += a,
                        PowerKind::Intangible => {
                            flags[0] = true;
                            facts.intangible = true;
                        }
                        PowerKind::Slippery => {
                            flags[1] = true;
                            facts.slippery = true;
                        }
                        PowerKind::HardenedShell => flags[2] = true,
                        PowerKind::Reflect => flags[3] = true,
                        PowerKind::CurlUp => flags[4] = true,
                        PowerKind::Enrage => flags[5] = true,
                        PowerKind::Asleep | PowerKind::Slumber => flags[6] = true,
                        PowerKind::Minion => minion = true,
                        PowerKind::Vulnerable => vuln += a,
                        PowerKind::Weak => weak += a,
                        _ => {}
                    }
                }
            }
            minions += u32::from(minion);
            if let Some(slot) = self.targets.last_mut() {
                *slot = facts;
            }
            if intent_slot < 3 {
                if let (Some(k), Some(i)) = (kind, c.intent) {
                    let base = V2_INTENTS + V2_INTENT_WIDTH * intent_slot;
                    let col = match k {
                        IntentKindTag::Attack | IntentKindTag::DeathBlow => 0,
                        IntentKindTag::MultiAttack => 1,
                        IntentKindTag::Defend => 2,
                        IntentKindTag::Buff => 3,
                        IntentKindTag::Debuff | IntentKindTag::DebuffStrong => 4,
                        IntentKindTag::StatusCard => 5,
                        _ => 6,
                    };
                    s[base + col] = 1.0;
                    s[base + 7] = clampf(
                        i.damage().saturating_mul(i32::from(i.hits())) as f32 / mhp,
                        0.0,
                        3.0,
                    );
                }
                intent_slot += 1;
            }
        }
        if !spawned.is_empty() {
            act = act.max(spec.roster_act(&spawned));
        }
        s[V2_ACT] = f32::from(act) / 3.0;
        s[V2_EXTRA_ENEMIES] = in_play.saturating_sub(3) as f32 / 3.0;
        let e = V2_ENEMY_POWERS;
        s[e] = clampf(artifact as f32 / 3.0, 0.0, 3.0);
        s[e + 1] = clampf(thorns as f32 / 5.0, 0.0, 3.0);
        s[e + 2] = clampf(plating as f32 / 10.0, 0.0, 3.0);
        s[e + 3] = clampf(ritual as f32 / 5.0, 0.0, 3.0);
        for (i, f) in flags.iter().enumerate() {
            s[e + 4 + i] = b(*f);
        }
        s[e + 11] = if in_play > 0 {
            minions as f32 / in_play as f32
        } else {
            0.0
        };
        s[e + 12] = clampf(vuln as f32 / 5.0, 0.0, 3.0);
        s[e + 13] = clampf(weak as f32 / 5.0, 0.0, 3.0);
        if present.intersects(&PLAYER_V2_MASK) {
            let mut amounts = [0i32; PLAYER_V2_POWERS.len()];
            for_each_player_power(state, |kind, amount| {
                if let Some(i) = PLAYER_V2_POWERS.iter().position(|(k, _)| *k == kind) {
                    amounts[i] = amounts[i].saturating_add(amount);
                }
            });
            for (i, ((_, div), a)) in PLAYER_V2_POWERS.iter().zip(amounts).enumerate() {
                s[V2_PLAYER_POWERS + i] = if *div == 0.0 {
                    b(a > 0)
                } else {
                    clampf(a as f32 / div, -3.0, 3.0)
                };
            }
            self.corruption = amounts[3] > 0;
        }
        self.version = FeatureVersion::V2;
    }
}

/// Every power on the player (the mirror creature's and the legacy list's,
/// without double counting) as `(kind, amount)` — the same union
/// `CombatState::player_power_amount` walks.
fn for_each_player_power(state: &CombatState, mut f: impl FnMut(PowerKind, i32)) {
    let creature: &[sts2_core::PowerUid] = match state
        .player
        .creature_uid
        .and_then(|uid| sts2_core::creature_by_id(&state.creatures, uid.0))
    {
        Some(c) => c.powers.as_slice(),
        None => &[],
    };
    let legacy = state
        .player
        .powers
        .iter()
        .copied()
        .filter(|uid| !creature.contains(uid));
    for uid in creature.iter().copied().chain(legacy) {
        if let Some(p) = state.powers.get_by_id(uid.0) {
            f(p.kind, p.amount);
        }
    }
}

/// The embedding key and feature vector `x(a)` of one legal action, in
/// `ctx`'s feature version (entries past the version's `action_dim` stay 0).
/// `heur_priority` is the heuristic rollout's priority for it
/// ([`crate::expert::heuristic_priorities`]).
#[must_use]
pub fn action_features(
    state: &CombatState,
    ctx: &StepCtx,
    action: &Action,
    heur_priority: i16,
) -> (ActionKey, [f32; ACTION_DIM_MAX]) {
    let mut x = [0.0f32; ACTION_DIM_MAX];
    let mhp = ctx.max_hp as f32;
    x[21] = clampf(
        -f32::from(heur_priority) / crate::expert::SOFTMAX_TAU,
        -10.0,
        3.0,
    );
    let key = match action {
        Action::PlayCard { hand_index, target } => {
            let Some(card) = state
                .player
                .hand
                .get(*hand_index as usize)
                .and_then(|uid| state.cards.get(uid.0.get() as usize - 1))
            else {
                x[4] = 1.0;
                return (ActionKey::Other, x);
            };
            let cost = if card.is_x_cost() {
                ctx.energy
            } else {
                i32::from(card.cost.max(0))
            };
            x[0] = 1.0;
            x[5] = cost as f32 / 3.0;
            x[6] = if cost == 0 { 1.0 } else { 0.0 };
            x[7] = (ctx.energy - cost).max(0) as f32 / 3.0;
            match card.card_type {
                CardType::Attack => x[8] = 1.0,
                CardType::Skill => x[9] = 1.0,
                CardType::Power => x[10] = 1.0,
                CardType::Curse | CardType::Status => x[11] = 1.0,
            }
            x[12] = card.base_damage as f32 / 20.0;
            x[13] = card.base_block as f32 / 20.0;
            x[14] = if card.upgraded { 1.0 } else { 0.0 };
            x[20] = card.base_block.min(ctx.unblocked).max(0) as f32 / 20.0;
            if let Some(t) = *target {
                x[15] = 1.0;
                if let Some(c) = state
                    .enemy_uids
                    .get(usize::from(t))
                    .and_then(|u| sts2_core::creature_by_id(&state.creatures, u.0))
                {
                    if !c.is_beaten_sentinel() && c.max_hp > 0 {
                        x[16] = c.current_hp.max(0) as f32 / c.max_hp as f32;
                        x[17] = if ctx.alive > 1 && ctx.lowest_slot == Some(t) {
                            1.0
                        } else {
                            0.0
                        };
                        if c.intent.is_some() {
                            x[18] = clampf(
                                ctx.version.incoming().total(state, c) as f32 / mhp,
                                0.0,
                                3.0,
                            );
                        }
                        if card.card_type == CardType::Attack && card.base_damage > 0 {
                            let mut dmg = (card.base_damage + ctx.strength).max(0) as f32;
                            if ctx.player_weak {
                                dmg *= 0.75;
                            }
                            if ctx.enemy_vuln.get(usize::from(t)).copied().unwrap_or(false) {
                                dmg *= 1.5;
                            }
                            x[19] = if dmg >= (c.current_hp + c.block.max(0)) as f32 {
                                1.0
                            } else {
                                0.0
                            };
                        }
                    }
                }
            }
            if ctx.version == FeatureVersion::V2 {
                x[V2_X_EXHAUSTS] = if sts2_core::card_effects::card_exhausts_inst(card)
                    || (ctx.corruption && card.card_type == CardType::Skill)
                {
                    1.0
                } else {
                    0.0
                };
                x[V2_X_ETHEREAL] = if sts2_core::card_effects::card_ethereal_inst(card) {
                    1.0
                } else {
                    0.0
                };
                target_facts(ctx, *target, &mut x);
            }
            if ctx.version == FeatureVersion::V4 {
                let stars = sts2_core::effective_star_cost_in_combat(state, card);
                x[V4_X_STAR_COST] = stars as f32 / 3.0;
                x[V4_X_STAR_COST + 1] = clampf((ctx.stars - stars).max(0) as f32 / 5.0, 0.0, 3.0);
                match &card.affliction {
                    Some(a) if a.kind == CardAfflictionKind::Tainted => {
                        x[V4_X_TAINTED] = clampf(a.stacks as f32 / 5.0, 0.0, 3.0);
                    }
                    Some(_) => x[V4_X_AFFLICTED] = 1.0,
                    None => {}
                }
                if card.card_type == CardType::Attack {
                    if let Some(h) = target.and_then(|t| ctx.hive.get(usize::from(t))) {
                        x[V4_X_TARGET_HIVE] = clampf(*h as f32 / 3.0, 0.0, 3.0);
                    }
                }
            }
            ActionKey::Card {
                id: card.card_id,
                upgraded: card.upgraded,
            }
        }
        Action::EndTurn => {
            x[1] = 1.0;
            x[7] = ctx.energy as f32 / 3.0;
            x[22] = ctx.playable as f32 / 10.0;
            x[23] = clampf(ctx.unblocked as f32 / mhp, 0.0, 3.0);
            ActionKey::EndTurn
        }
        Action::DrinkPotion {
            potion_slot: slot,
            target,
        }
        | Action::UsePotion { slot, target } => {
            x[2] = 1.0;
            x[7] = ctx.energy as f32 / 3.0;
            x[15] = if target.is_some() { 1.0 } else { 0.0 };
            x[23] = clampf(ctx.unblocked as f32 / mhp, 0.0, 3.0);
            if ctx.version == FeatureVersion::V2 {
                target_facts(ctx, *target, &mut x);
            }
            match state.player.potions.get(usize::from(*slot)) {
                Some(k) => ActionKey::Potion(*k),
                None => ActionKey::Other,
            }
        }
        Action::ResolveChoice { .. } => {
            x[3] = 1.0;
            ActionKey::Choice
        }
        Action::Channel { .. } | Action::Evoke { .. } => {
            x[4] = 1.0;
            ActionKey::Other
        }
    };
    (key, x)
}

/// The v2 target facts ([`V2_X_TARGET`]) of an action aimed at enemy slot
/// `target`.
fn target_facts(ctx: &StepCtx, target: Option<u8>, x: &mut [f32; ACTION_DIM_MAX]) {
    let Some(t) = target.and_then(|t| ctx.targets.get(usize::from(t))) else {
        return;
    };
    let b = |v: bool| if v { 1.0 } else { 0.0 };
    x[V2_X_TARGET] = b(t.intangible);
    x[V2_X_TARGET + 1] = clampf(t.thorns as f32 / 5.0, 0.0, 3.0);
    x[V2_X_TARGET + 2] = b(t.slippery);
    x[V2_X_TARGET + 3] = b(t.artifact);
    x[V2_X_TARGET + 4] = b(t.attacking);
    x[V2_X_TARGET + 5] = clampf(t.block as f32 / 20.0, 0.0, 3.0);
}

/// The playout net's weights. See the module docs for the model.
#[derive(Debug)]
pub struct RolloutNet {
    hidden: usize,
    emb_dim: usize,
    state_dim: usize,
    action_dim: usize,
    spec: FeatureSpec,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    emb: Vec<f32>,
    bias: Vec<f32>,
    proj: Vec<f32>,
    lin: Vec<f32>,
    /// The encounter embedding, `monster rows × hidden` (empty = none).
    monster_emb: Vec<f32>,
    /// Softmax temperature applied to the logits.
    pub temperature: f32,
    /// When set, `EndTurn` is only sampled once no card is playable — the
    /// heuristic playout's rule (GH #446: playouts that stop early are worse
    /// evaluators).
    pub no_early_end_turn: bool,
    card_rows: Vec<u32>,
    potion_rows: Vec<(PotionKind, u32)>,
    end_turn_row: u32,
    choice_row: u32,
    other_row: u32,
    /// Number of embedding rows (row 0 = unknown key).
    pub rows: usize,
}

fn default_feature_version() -> u32 {
    1
}

#[derive(serde::Deserialize)]
struct NetFile {
    #[serde(default = "default_feature_version")]
    feature_version: u32,
    state_dim: usize,
    action_dim: usize,
    hidden: usize,
    emb_dim: usize,
    temperature: f32,
    #[serde(default)]
    no_early_end_turn: bool,
    keys: Vec<String>,
    w1: Vec<Vec<f32>>,
    b1: Vec<f32>,
    w2: Vec<Vec<f32>>,
    b2: Vec<f32>,
    emb: Vec<Vec<f32>>,
    bias: Vec<f32>,
    proj: Vec<Vec<f32>>,
    lin: Vec<f32>,
    /// v2: species name → act.
    #[serde(default)]
    monster_act: Option<std::collections::BTreeMap<String, u8>>,
    /// v2, optional: `[act, [species, ...]]` per encounter (one name per
    /// monster), which refines `monster_act` ([`FeatureSpec::with_encounters`]).
    #[serde(default)]
    encounter_act: Option<Vec<(u8, Vec<String>)>>,
    /// v2, optional: the encounter embedding's species, row `i + 1` each.
    #[serde(default)]
    monster_keys: Option<Vec<String>>,
    /// v2, optional: one `hidden`-wide row per `monster_keys` entry.
    #[serde(default)]
    monster_emb: Option<Vec<Vec<f32>>>,
}

fn flatten(m: &[Vec<f32>], rows: usize, cols: usize, what: &str) -> Result<Vec<f32>, String> {
    if m.len() != rows || m.iter().any(|r| r.len() != cols) {
        return Err(format!("{what}: expected {rows}x{cols}"));
    }
    Ok(m.iter().flatten().copied().collect())
}

impl RolloutNet {
    /// Parse a weights file (JSON written by the trainer).
    ///
    /// # Errors
    /// An unknown feature version, dimensions that disagree with the
    /// version's, a malformed matrix, an unknown key name, per-species tables
    /// on a v1 file or a v2 file without `monster_act`.
    pub fn from_json(text: &str) -> Result<Self, String> {
        let f: NetFile = serde_json::from_str(text).map_err(|e| e.to_string())?;
        let version = FeatureVersion::from_number(f.feature_version)
            .ok_or_else(|| format!("unknown feature_version {}", f.feature_version))?;
        let (sd, ad) = (version.state_dim(), version.action_dim());
        if f.state_dim != sd || f.action_dim != ad {
            return Err(format!(
                "feature dims {}x{} in file, {sd}x{ad} for feature version {}",
                f.state_dim,
                f.action_dim,
                version.number()
            ));
        }
        let (h, d, rows) = (f.hidden, f.emb_dim, f.keys.len());
        if h == 0 || d == 0 {
            return Err("hidden and emb_dim must be positive".to_owned());
        }
        if f.keys.first().map(String::as_str) != Some("<unk>") {
            return Err("keys[0] must be \"<unk>\"".to_owned());
        }
        let card_by_name: std::collections::HashMap<&str, CardId> = CardId::all()
            .iter()
            .map(|c| (c.to_python_name(), *c))
            .collect();
        let mut card_rows: Vec<u32> = Vec::new();
        let mut potion_rows = Vec::new();
        let (mut end_turn_row, mut choice_row, mut other_row) = (0, 0, 0);
        for (row, k) in f.keys.iter().enumerate().skip(1) {
            let row = u32::try_from(row).map_err(|_| "too many keys".to_owned())?;
            if let Some(card) = k.strip_prefix("card:") {
                let (name, up) = card.strip_suffix('+').map_or((card, false), |n| (n, true));
                // Cards this build doesn't know (renamed / removed) keep row 0.
                if let Some(&id) = card_by_name.get(name) {
                    let idx = id as usize * 2 + usize::from(up);
                    if card_rows.len() <= idx {
                        card_rows.resize(idx + 1, 0);
                    }
                    card_rows[idx] = row;
                }
            } else if let Some(name) = k.strip_prefix("potion:") {
                if let Some(p) = PotionKind::ALL.iter().find(|p| p.python_name() == name) {
                    potion_rows.push((*p, row));
                }
            } else {
                match k.as_str() {
                    "end_turn" => end_turn_row = row,
                    "choice" => choice_row = row,
                    "other" => other_row = row,
                    _ => return Err(format!("unknown key {k:?}")),
                }
            }
        }
        let (spec, monster_emb) = match version {
            FeatureVersion::V1 | FeatureVersion::V3 | FeatureVersion::V4 => {
                if f.monster_act.is_some()
                    || f.encounter_act.is_some()
                    || f.monster_keys.is_some()
                    || f.monster_emb.is_some()
                {
                    return Err("monster_act / encounter_act / monster_keys / monster_emb \
                                need feature_version 2"
                        .to_owned());
                }
                let spec = match version {
                    FeatureVersion::V1 => FEATURES_V1,
                    FeatureVersion::V3 => FEATURES_V3,
                    _ => FEATURES_V4,
                };
                (spec, Vec::new())
            }
            FeatureVersion::V2 => {
                let by_name = monster_by_name();
                let table = f
                    .monster_act
                    .as_ref()
                    .ok_or("feature_version 2 needs monster_act")?;
                let acts: Vec<(MonsterKind, u8)> = table
                    .iter()
                    // Species this build doesn't know never appear in a fight.
                    .filter_map(|(n, a)| by_name.get(n.as_str()).map(|k| (*k, *a)))
                    .collect();
                if acts.is_empty() && !table.is_empty() {
                    return Err("no monster_act name is a species of this build (names are \
                                MonsterKind::python_name, e.g. \"Nibbit\")"
                        .to_owned());
                }
                let (mrows, memb) = match (f.monster_keys, f.monster_emb) {
                    (None, None) => (Vec::new(), Vec::new()),
                    (Some(keys), Some(emb_rows)) => {
                        let memb = flatten(&emb_rows, keys.len(), h, "monster_emb")?;
                        let mut mrows = Vec::new();
                        for (idx, name) in keys.iter().enumerate() {
                            let row1 =
                                u16::try_from(idx + 1).map_err(|_| "too many monster_keys")?;
                            if let Some(kind) = by_name.get(name.as_str()) {
                                mrows.push((*kind, row1));
                            }
                        }
                        (mrows, memb)
                    }
                    _ => return Err("monster_keys and monster_emb come together".to_owned()),
                };
                // A roster naming a species this build doesn't know can never
                // be spawned whole, so it is dropped.
                let rosters: Vec<(u8, Vec<MonsterKind>)> = f
                    .encounter_act
                    .iter()
                    .flatten()
                    .filter_map(|(act, names)| {
                        let members: Option<Vec<MonsterKind>> = names
                            .iter()
                            .map(|n| by_name.get(n.as_str()).copied())
                            .collect();
                        members.map(|m| (*act, m))
                    })
                    .collect();
                (
                    FeatureSpec::v2(&acts, &mrows).with_encounters(&rosters),
                    memb,
                )
            }
        };
        Ok(Self {
            hidden: h,
            emb_dim: d,
            state_dim: sd,
            action_dim: ad,
            spec,
            w1: flatten(&f.w1, h, sd, "w1")?,
            b1: (f.b1.len() == h).then_some(f.b1).ok_or("b1")?,
            w2: flatten(&f.w2, d, h, "w2")?,
            b2: (f.b2.len() == d).then_some(f.b2).ok_or("b2")?,
            emb: flatten(&f.emb, rows, d, "emb")?,
            bias: (f.bias.len() == rows).then_some(f.bias).ok_or("bias")?,
            proj: flatten(&f.proj, d, ad, "proj")?,
            lin: (f.lin.len() == ad).then_some(f.lin).ok_or("lin")?,
            monster_emb,
            temperature: f.temperature,
            no_early_end_turn: f.no_early_end_turn,
            card_rows,
            potion_rows,
            end_turn_row,
            choice_row,
            other_row,
            rows,
        })
    }

    /// The features this net reads.
    #[must_use]
    pub fn feature_spec(&self) -> &FeatureSpec {
        &self.spec
    }

    /// Width of this net's state vector.
    #[must_use]
    pub fn state_dim(&self) -> usize {
        self.state_dim
    }

    /// Width of this net's action vector.
    #[must_use]
    pub fn action_dim(&self) -> usize {
        self.action_dim
    }

    /// The embedding row `key` resolves to (0 = unknown).
    #[must_use]
    pub fn row(&self, key: ActionKey) -> usize {
        let r = match key {
            ActionKey::Card { id, upgraded } => self
                .card_rows
                .get(id as usize * 2 + usize::from(upgraded))
                .copied()
                .unwrap_or(0),
            ActionKey::Potion(p) => self
                .potion_rows
                .iter()
                .find(|(k, _)| *k == p)
                .map_or(0, |(_, r)| *r),
            ActionKey::EndTurn => self.end_turn_row,
            ActionKey::Choice => self.choice_row,
            ActionKey::Other => self.other_row,
        };
        r as usize
    }

    /// The per-step half of every logit: `q(s)` and `lin + Pᵀ·q(s)`.
    fn prepare(&self, ctx: &StepCtx) -> StepQuery {
        self.prepare_raw(&ctx.state[..self.state_dim], &ctx.monsters)
    }

    /// [`Self::prepare`] from a state vector (`state_dim` wide) and the
    /// encounter-embedding rows.
    fn prepare_raw(&self, s: &[f32], monsters: &[u16; MONSTER_SLOTS]) -> StepQuery {
        let s = &s[..self.state_dim];
        let hid: SmallVec<[f32; 64]> = if self.monster_emb.is_empty() {
            self.w1
                .chunks_exact(self.state_dim)
                .zip(&self.b1)
                .map(|(row, b)| (b + dot(row, s)).max(0.0))
                .collect()
        } else {
            let mut pre: SmallVec<[f32; 64]> = self
                .w1
                .chunks_exact(self.state_dim)
                .zip(&self.b1)
                .map(|(row, b)| b + dot(row, s))
                .collect();
            for &r in monsters.iter().filter(|r| **r != 0) {
                let start = (usize::from(r) - 1) * self.hidden;
                for (h, m) in pre
                    .iter_mut()
                    .zip(&self.monster_emb[start..start + self.hidden])
                {
                    *h += m;
                }
            }
            pre.iter().map(|h| h.max(0.0)).collect()
        };
        let q: SmallVec<[f32; 32]> = self
            .w2
            .chunks_exact(self.hidden)
            .zip(&self.b2)
            .map(|(row, b)| b + dot(row, &hid))
            .collect();
        let mut w = [0.0f32; ACTION_DIM_MAX];
        w[..self.action_dim].copy_from_slice(&self.lin);
        for (q_j, p_row) in q.iter().zip(self.proj.chunks_exact(self.action_dim)) {
            for (w_k, p_jk) in w[..self.action_dim].iter_mut().zip(p_row) {
                *w_k += q_j * p_jk;
            }
        }
        StepQuery { q, w }
    }

    /// Logits for every legal action in `actions`.
    #[must_use]
    pub fn logits(
        &self,
        state: &CombatState,
        actions: &[Action],
        end_turn_priority: i16,
    ) -> SmallVec<[f32; 16]> {
        let ctx = StepCtx::with_spec(state, actions, &self.spec);
        let heur = crate::expert::heuristic_priorities_with(
            state,
            actions,
            end_turn_priority,
            self.spec.version.incoming(),
        );
        let step = self.prepare(&ctx);
        actions
            .iter()
            .zip(&heur)
            .map(|(action, &prio)| {
                let (key, feats) = action_features(state, &ctx, action, prio);
                self.logit(&step, self.row(key), &feats)
            })
            .collect()
    }

    /// Everything [`Self::logits`] computes at `state` for its legal actions
    /// (at the playout's EndTurn priority), for checking a trainer's model
    /// against this one (`sts2_rs.LearnedPlayout.explain`).
    #[must_use]
    pub fn explain(&self, state: &CombatState) -> Explained {
        let actions = sts2_core::legal_actions(state);
        let ctx = StepCtx::with_spec(state, &actions, &self.spec);
        let heur = crate::expert::heuristic_priorities_with(
            state,
            &actions,
            crate::expert::END_TURN_PRIORITY,
            self.spec.version.incoming(),
        );
        let step = self.prepare(&ctx);
        let rows = actions
            .iter()
            .zip(&heur)
            .map(|(action, &prio)| {
                let (key, feats) = action_features(state, &ctx, action, prio);
                let row = self.row(key);
                (
                    action.clone(),
                    row,
                    feats[..self.action_dim].to_vec(),
                    self.logit(&step, row, &feats),
                )
            })
            .collect();
        Explained {
            state: ctx.state[..self.state_dim].to_vec(),
            monsters: ctx.monsters,
            actions: rows,
        }
    }

    /// One action's logit, `bias[row] + q·E[row] + (lin + Pᵀ·q)·x` — the
    /// model's `q·(E[row] + P·x) + bias[row] + lin·x` with the `P` product
    /// hoisted into [`Self::prepare`], once per step instead of per action.
    fn logit(&self, step: &StepQuery, row: usize, feats: &[f32; ACTION_DIM_MAX]) -> f32 {
        let emb = &self.emb[row * self.emb_dim..(row + 1) * self.emb_dim];
        self.bias[row]
            + dot(&step.q, emb)
            + dot(&step.w[..self.action_dim], &feats[..self.action_dim])
    }
}

/// [`RolloutNet::explain`]'s output.
#[derive(Clone, Debug)]
pub struct Explained {
    /// The state features `s`.
    pub state: Vec<f32>,
    /// Encounter-embedding rows of the first living enemies (0 = none).
    pub monsters: [u16; MONSTER_SLOTS],
    /// Per legal action, in `legal_actions` order: the action, its embedding
    /// row, its features `x(a)` and its logit.
    pub actions: Vec<(Action, usize, Vec<f32>, f32)>,
}

/// [`RolloutNet::prepare`]'s output for one playout step.
struct StepQuery {
    /// The state query `q(s)`.
    q: SmallVec<[f32; 32]>,
    /// The action-feature weights `lin + Pᵀ·q(s)` (first `action_dim` used).
    w: [f32; ACTION_DIM_MAX],
}

/// `a · b` over eight independent partial sums, which LLVM vectorizes; a
/// single running sum is a serial chain of float adds it may not reorder.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let ((ca, ra), (cb, rb)) = (a.as_chunks::<8>(), b.as_chunks::<8>());
    let tail = ra.iter().zip(rb).map(|(x, y)| x * y).sum::<f32>();
    let mut acc = [0.0f32; 8];
    for (xa, xb) in ca.iter().zip(cb) {
        for ((s, x), y) in acc.iter_mut().zip(xa).zip(xb) {
            *s += x * y;
        }
    }
    acc.iter().sum::<f32>() + tail
}

/// One recorded playout decision with at least two allowed actions (a
/// forced step carries no policy gradient).
#[derive(Clone, Debug)]
pub struct TrajStep {
    /// The state features `s` (the net's `state_dim` entries are meaningful).
    pub state: [f32; STATE_DIM_MAX],
    /// Encounter-embedding rows of the first living enemies (v2; 0 = none).
    pub monsters: [u16; MONSTER_SLOTS],
    /// Each legal action's embedding row in the net.
    pub rows: SmallVec<[u32; 16]>,
    /// Each legal action's features `x(a)` (the net's `action_dim` entries
    /// are meaningful).
    pub x: SmallVec<[[f32; ACTION_DIM_MAX]; 16]>,
    /// Whether each action could be sampled (the no-early-end rule masks
    /// `EndTurn`).
    pub allowed: SmallVec<[bool; 16]>,
    /// Index of the sampled action.
    pub chosen: u8,
}

/// One playout by the net with every decision recorded, for training the
/// net on playout outcomes.
#[derive(Clone, Debug)]
pub struct Trajectory {
    /// The recorded decisions, in order.
    pub steps: Vec<TrajStep>,
    /// Whether the player won the combat.
    pub won: bool,
    /// Final HP / max HP.
    pub hp_frac: f32,
    /// Fraction of enemy HP removed (beaten sentinels count as removed; a
    /// v3 net's counts the forms still to come).
    pub removed: f32,
    /// Final round.
    pub round: u16,
    /// What the search backs up for this playout
    /// ([`crate::playout_return`] of its final state) when
    /// [`TrajOptions::score`] was given, else NaN.
    pub search_value: f32,
}

/// How [`playout_trajectories_with`] plays.
#[derive(Clone, Copy, Debug)]
pub struct TrajOptions<'a> {
    /// Softmax temperature.
    pub temperature: f32,
    /// Mask `EndTurn` while a card is playable.
    pub no_early_end_turn: bool,
    /// Step cap per playout.
    pub max_steps: u32,
    /// Play from the honest view the deployed search sees: each playout
    /// starts from its own `sts2_core::blindfold::belief_root` of the state
    /// (unknown draw-pile order canonicalized and drawn at random at draw
    /// time, hidden RNG streams reseeded), seeded from the job seed. Off,
    /// every playout of a state shares its true draw order and RNG.
    pub belief: bool,
    /// Score each playout with [`crate::playout_return`] under this search
    /// configuration and leaf ([`Trajectory::search_value`]).
    pub score: Option<(&'a MctsConfig, &'a LeafEval)>,
}

/// Salt for the per-playout belief seed, so it is independent of the
/// policy's sampling stream (seeded from the same job seed).
const BELIEF_PLAYOUT_SALT: u64 = 0xB3_11E7_F00D_5A17;

/// Play `state` out with `net` (sampling at `temperature`) and record every
/// decision.
pub fn playout_trajectory<R: Rng>(
    net: &RolloutNet,
    state: &CombatState,
    rng: &mut R,
    temperature: f32,
    no_early_end_turn: bool,
    max_steps: u32,
) -> Trajectory {
    let (traj, _) = play_recorded(
        net,
        state.clone(),
        rng,
        temperature,
        no_early_end_turn,
        max_steps,
    );
    traj
}

/// [`playout_trajectory`] from an owned start; also returns the final state.
fn play_recorded<R: Rng>(
    net: &RolloutNet,
    mut s: CombatState,
    rng: &mut R,
    temperature: f32,
    no_early_end_turn: bool,
    max_steps: u32,
) -> (Trajectory, CombatState) {
    let mut steps = Vec::new();
    let mut n = 0u32;
    while !sts2_core::is_terminal(&s) && n < max_steps {
        let acts = sts2_core::legal_actions(&s);
        if acts.is_empty() {
            break;
        }
        let ctx = StepCtx::with_spec(&s, &acts, &net.spec);
        let heur = crate::expert::heuristic_priorities_with(
            &s,
            &acts,
            crate::expert::END_TURN_PRIORITY,
            net.spec.version.incoming(),
        );
        let query = net.prepare(&ctx);
        let any_card = acts.iter().any(|a| matches!(a, Action::PlayCard { .. }));
        let mut step = TrajStep {
            state: ctx.state,
            monsters: ctx.monsters,
            rows: SmallVec::new(),
            x: SmallVec::new(),
            allowed: SmallVec::new(),
            chosen: 0,
        };
        let mut logits: SmallVec<[f32; 16]> = SmallVec::new();
        for (a, &h) in acts.iter().zip(&heur) {
            let (key, feats) = action_features(&s, &ctx, a, h);
            let row = net.row(key);
            let ok = !(no_early_end_turn && any_card && matches!(a, Action::EndTurn));
            logits.push(if ok {
                net.logit(&query, row, &feats)
            } else {
                f32::NEG_INFINITY
            });
            step.rows.push(u32::try_from(row).unwrap_or(0));
            step.x.push(feats);
            step.allowed.push(ok);
        }
        let idx = sample(rng, &logits, temperature);
        if step.allowed.iter().filter(|ok| **ok).count() >= 2 {
            step.chosen = u8::try_from(idx).unwrap_or(u8::MAX);
            steps.push(step);
        }
        let _ = sts2_core::apply(&mut s, acts[idx].clone());
        n += 1;
    }
    // A v3 net's shaped reward counts the forms still to come, like its
    // `removed` feature.
    let (hp_sum, max_sum) =
        crate::enemy_eval::enemy_hp_remaining(&s, net.spec.version.removed_counts_forms());
    let traj = Trajectory {
        steps,
        won: s.player_won,
        hp_frac: s.player.current_hp.max(0) as f32 / s.player.max_hp.max(1) as f32,
        removed: if max_sum > 0 {
            1.0 - hp_sum as f32 / max_sum as f32
        } else {
            1.0
        },
        round: s.round,
        search_value: f32::NAN,
    };
    (traj, s)
}

/// [`playout_trajectory`] for many `(state, seed)` jobs in parallel.
#[must_use]
pub fn playout_trajectories(
    net: &RolloutNet,
    jobs: &[(CombatState, u64)],
    temperature: f32,
    no_early_end_turn: bool,
    max_steps: u32,
) -> Vec<Trajectory> {
    playout_trajectories_with(
        net,
        jobs,
        &TrajOptions {
            temperature,
            no_early_end_turn,
            max_steps,
            belief: false,
            score: None,
        },
    )
}

/// [`playout_trajectories`] with the honest-view and scoring options.
#[must_use]
pub fn playout_trajectories_with(
    net: &RolloutNet,
    jobs: &[(CombatState, u64)],
    opts: &TrajOptions<'_>,
) -> Vec<Trajectory> {
    use rand::SeedableRng;
    use rayon::prelude::*;
    jobs.par_iter()
        .map(|(st, seed)| {
            let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(*seed);
            let mut start = st.clone();
            if opts.belief {
                sts2_core::blindfold::belief_root(
                    &mut start,
                    sts2_core::run_rng_set::splitmix64(*seed ^ BELIEF_PLAYOUT_SALT),
                );
            }
            let (mut traj, end) = play_recorded(
                net,
                start,
                &mut rng,
                opts.temperature,
                opts.no_early_end_turn,
                opts.max_steps,
            );
            if let Some((cfg, leaf)) = opts.score {
                traj.search_value = crate::playout_return(&end, cfg, leaf);
            }
            traj
        })
        .collect()
}

static NET: OnceLock<Option<RolloutNet>> = OnceLock::new();

/// The net named by `STS2_ROLLOUT_NET`, loaded on first use; `None` when the
/// variable is unset. When set it replaces the built-in nets for every
/// character ([`net_for`]).
///
/// # Panics
/// When `STS2_ROLLOUT_NET` names a file that can't be read or parsed — a
/// silent fallback would quietly measure another policy instead.
pub fn global_net() -> Option<&'static RolloutNet> {
    NET.get_or_init(|| {
        let path = std::env::var("STS2_ROLLOUT_NET").ok()?;
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("STS2_ROLLOUT_NET={path}: {e}"));
        let net =
            RolloutNet::from_json(&text).unwrap_or_else(|e| panic!("STS2_ROLLOUT_NET={path}: {e}"));
        Some(with_env_overrides(net))
    })
    .as_ref()
}

/// Install `net` as the process-wide policy instead of loading
/// `STS2_ROLLOUT_NET`. Returns `false` (dropping `net`) when the policy was
/// already resolved.
pub fn set_global_net(net: RolloutNet) -> bool {
    NET.set(Some(net)).is_ok()
}

/// A net whose logit is exactly the heuristic's softmax logit
/// (`-priority / tau`, feature 21): the net with the heuristic playout's
/// preferences. Its greedy line is the heuristic's top choice at each step.
///
/// # Panics
/// Never: the weights are built here to this build's dimensions.
#[must_use]
pub fn heuristic_mirror_net() -> RolloutNet {
    let (h, d) = (2usize, 2usize);
    let zeros = |r: usize, c: usize| vec![vec![0.0f32; c]; r];
    let mut lin = vec![0.0f32; ACTION_DIM];
    lin[21] = 1.0;
    let text = serde_json::json!({
        "state_dim": STATE_DIM, "action_dim": ACTION_DIM, "hidden": h, "emb_dim": d,
        "temperature": 1.0, "keys": ["<unk>", "end_turn"],
        "w1": zeros(h, STATE_DIM), "b1": vec![0.0; h], "w2": zeros(d, h), "b2": vec![0.0; d],
        "emb": zeros(2, d), "bias": [0.0, 0.0], "proj": zeros(d, ACTION_DIM), "lin": lin,
    })
    .to_string();
    RolloutNet::from_json(&text).expect("heuristic mirror net is well-formed")
}

/// `net` with `STS2_ROLLOUT_NO_EARLY_END` and `STS2_ROLLOUT_TEMP` applied.
/// Only the experiment net ([`global_net`]) takes them.
///
/// # Panics
/// When `STS2_ROLLOUT_TEMP` is not a positive number: ignoring a typo would
/// run the file's temperature while the experiment believes otherwise.
fn with_env_overrides(mut net: RolloutNet) -> RolloutNet {
    if let Ok(v) = std::env::var("STS2_ROLLOUT_NO_EARLY_END") {
        net.no_early_end_turn = matches!(v.as_str(), "1" | "true");
    }
    if let Ok(v) = std::env::var("STS2_ROLLOUT_TEMP") {
        net.temperature = v
            .parse::<f32>()
            .ok()
            .filter(|t| t.is_finite() && *t > 0.0)
            .unwrap_or_else(|| panic!("STS2_ROLLOUT_TEMP={v:?} is not a positive number"));
    }
    net
}

/// One built-in net, parsed on first use. `build.rs` sets
/// `cfg(rollout_net_<name>)` only when `data/rollout_nets/<name>.json` exists,
/// and each arm of [`builtin_net`] is compiled under its cfg, so a checkout
/// without the file still compiles.
#[allow(unused_macros)]
macro_rules! builtin {
    ($name:literal) => {{
        static NET: OnceLock<RolloutNet> = OnceLock::new();
        NET.get_or_init(|| {
            RolloutNet::from_json(include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../data/rollout_nets/",
                $name,
                ".json"
            )))
            .unwrap_or_else(|e| panic!(concat!("built-in ", $name, " rollout net: {}"), e))
        })
    }};
}

/// The built-in net for `character`, or `None` when its file is not in the
/// checkout. All five characters ship one (GH #777, #822, #823, #824, #825).
///
/// # Panics
/// When a compiled-in net does not parse. It is a build-time asset, so the
/// `built_in_nets_load` test catches that first.
#[must_use]
#[allow(unreachable_patterns)]
pub fn builtin_net(character: CharacterId) -> Option<&'static RolloutNet> {
    match character {
        #[cfg(rollout_net_ironclad)]
        CharacterId::Ironclad => Some(builtin!("ironclad")),
        #[cfg(rollout_net_silent)]
        CharacterId::Silent => Some(builtin!("silent")),
        #[cfg(rollout_net_defect)]
        CharacterId::Defect => Some(builtin!("defect")),
        #[cfg(rollout_net_regent)]
        CharacterId::Regent => Some(builtin!("regent")),
        #[cfg(rollout_net_necrobinder)]
        CharacterId::Necrobinder => Some(builtin!("necrobinder")),
        _ => None,
    }
}

/// The built-in PUCT move-prior net of `character`, as the text of its file
/// `data/prior_nets/<character>.json` (GH #830), or `None` when there is
/// none. Only Ironclad has one: a net in the playout-net format, trained by
/// behaviour cloning on human combat decisions, which [`RolloutNetPrior`]
/// turns into a move prior. It passed its full-run gate as part of the
/// Ironclad run-agent bundle, in elite and boss fights at mix 0.5, T 1
/// (PR #938).
///
/// Returns the text, not a parsed net: [`RolloutNet`] is not `Clone` and a
/// [`RolloutNetPrior`] owns its net, so each caller parses its own copy. The
/// arm is compiled only under `cfg(prior_net_ironclad)` (set by `build.rs`
/// when the file exists), and no other character has an arm, so a net
/// dropped into `data/prior_nets/` for another character switches nothing on.
/// Nothing here reads the environment; the run agent decides where the prior
/// applies.
#[must_use]
pub fn builtin_prior_json(character: CharacterId) -> Option<&'static str> {
    match character {
        #[cfg(prior_net_ironclad)]
        CharacterId::Ironclad => Some(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../data/prior_nets/ironclad.json"
        ))),
        _ => None,
    }
}

/// The net a playout of `state` uses: [`global_net`] when `STS2_ROLLOUT_NET`
/// is set, else the [`builtin_net`] of the fight's character. `None` means
/// the heuristic playout.
#[must_use]
pub fn net_for(state: &CombatState) -> Option<&'static RolloutNet> {
    global_net().or_else(|| state.player.character.and_then(builtin_net))
}

/// Pick a playout action at `state` with [`net_for`]'s net. When there is
/// none, use the heuristic playout ([`crate::expert::pick_rollout_action_with`])
/// with the same arguments and random draws, so a search under
/// [`crate::RolloutPolicyKind::Net`] for a character without a net is
/// identical to one under [`crate::RolloutPolicyKind::Heuristic`].
pub fn pick<R: Rng>(
    rng: &mut R,
    state: &CombatState,
    actions: &[Action],
    end_turn_priority: i16,
) -> usize {
    pick_incoming(
        rng,
        state,
        actions,
        end_turn_priority,
        crate::enemy_eval::IncomingDamage::Base,
    )
}

/// [`pick`] whose heuristic fallback (no net for `state`) reads telegraphed
/// attacks the `incoming` way ([`crate::MctsConfig::effective_incoming`]).
/// A net, when there is one, reads what its feature version reads either
/// way.
pub fn pick_incoming<R: Rng>(
    rng: &mut R,
    state: &CombatState,
    actions: &[Action],
    end_turn_priority: i16,
    incoming: crate::enemy_eval::IncomingDamage,
) -> usize {
    match net_for(state) {
        Some(net) => pick_with(
            net,
            rng,
            state,
            actions,
            end_turn_priority,
            net.temperature,
            net.no_early_end_turn,
        ),
        None => crate::expert::pick_rollout_action_incoming(
            rng,
            state,
            actions,
            end_turn_priority,
            incoming,
        ),
    }
}

/// [`pick`] with an explicit net, temperature and early-end rule.
pub fn pick_with<R: Rng>(
    net: &RolloutNet,
    rng: &mut R,
    state: &CombatState,
    actions: &[Action],
    end_turn_priority: i16,
    temperature: f32,
    no_early_end_turn: bool,
) -> usize {
    let logits = pick_logits(net, state, actions, end_turn_priority, no_early_end_turn);
    sample(rng, &logits, temperature)
}

/// The logits [`pick_with`] samples from, aligned with `actions`: the net's
/// [`RolloutNet::logits`] with `EndTurn` masked to `-inf` under the
/// no-early-end rule while a card is playable. The sampling temperature
/// divides every logit alike, so the order of these is the order of the
/// playout's probabilities. Draws no randomness.
#[must_use]
pub fn pick_logits(
    net: &RolloutNet,
    state: &CombatState,
    actions: &[Action],
    end_turn_priority: i16,
    no_early_end_turn: bool,
) -> SmallVec<[f32; 16]> {
    let mut logits = net.logits(state, actions, end_turn_priority);
    if no_early_end_turn && actions.iter().any(|a| matches!(a, Action::PlayCard { .. })) {
        for (l, a) in logits.iter_mut().zip(actions) {
            if matches!(a, Action::EndTurn) {
                *l = f32::NEG_INFINITY;
            }
        }
    }
    logits
}

/// What a [`pick`] at `state` would sample from, without sampling: the
/// [`pick_logits`] of [`net_for`]'s net with that net's own early-end rule,
/// aligned with `actions`. `None` when `state` has no net, i.e. when the
/// playout there is the heuristic one. Draws no randomness.
///
/// `actions` must be the full [`sts2_core::legal_actions`] list, in its
/// order, exactly as a playout sees it: the state features read the count
/// of card plays in it.
#[must_use]
pub fn playout_logits(
    state: &CombatState,
    actions: &[Action],
    end_turn_priority: i16,
) -> Option<SmallVec<[f32; 16]>> {
    let net = net_for(state)?;
    Some(pick_logits(
        net,
        state,
        actions,
        end_turn_priority,
        net.no_early_end_turn,
    ))
}

/// Sample an index from `softmax(logits / temperature)`. A masked action
/// (`-inf` logit, weight 0) is never returned while any other action has
/// weight: an edge draw of exactly 0 or float drift past the last bucket
/// would otherwise land on it, and `rollout_learn rl` turns one such step
/// into a `-1e9` log-probability that wrecks the weights.
fn sample<R: Rng>(rng: &mut R, logits: &[f32], temperature: f32) -> usize {
    let t = temperature.max(1e-3);
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !m.is_finite() {
        return rng.gen_range(0..logits.len());
    }
    let w: SmallVec<[f32; 16]> = logits.iter().map(|l| ((l - m) / t).exp()).collect();
    let total: f32 = w.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return rng.gen_range(0..logits.len());
    }
    let mut x = rng.gen::<f32>() * total;
    let mut last = 0;
    for (i, &wi) in w.iter().enumerate() {
        if wi <= 0.0 {
            continue;
        }
        if x < wi {
            return i;
        }
        x -= wi;
        last = i;
    }
    last
}

/// Which policy plays a [`playout`].
#[derive(Clone, Copy, Debug)]
pub enum PlayoutPolicy {
    /// The heuristic playout ([`crate::expert::pick_rollout_action_with`]).
    Heuristic,
    /// The playout net at this temperature, with or without the
    /// no-early-end-turn rule.
    Net {
        temperature: f32,
        no_early_end_turn: bool,
    },
}

/// Play `state` to the end of the combat (or `max_steps` actions) with one
/// policy and no search — the playout the search would run from here,
/// measured on its own. Returns the final state.
///
/// # Panics
/// [`PlayoutPolicy::Net`] when [`net_for`] has no net for `state`.
pub fn playout<R: Rng>(
    state: &CombatState,
    policy: PlayoutPolicy,
    rng: &mut R,
    max_steps: u32,
) -> (CombatState, u32) {
    let mut s = state.clone();
    let mut steps = 0u32;
    while !sts2_core::is_terminal(&s) && steps < max_steps {
        let acts = sts2_core::legal_actions(&s);
        if acts.is_empty() {
            break;
        }
        let idx = match policy {
            PlayoutPolicy::Heuristic => crate::expert::pick_rollout_action_with(
                rng,
                &s,
                &acts,
                crate::expert::END_TURN_PRIORITY,
            ),
            PlayoutPolicy::Net {
                temperature,
                no_early_end_turn,
            } => pick_with(
                net_for(&s).expect("PlayoutPolicy::Net: no STS2_ROLLOUT_NET and no built-in net for this character"),
                rng,
                &s,
                &acts,
                crate::expert::END_TURN_PRIORITY,
                temperature,
                no_early_end_turn,
            ),
        };
        let _ = sts2_core::apply(&mut s, acts[idx].clone());
        steps += 1;
    }
    (s, steps)
}

/// Features of every legal action at `state`, for the training data
/// (`sts2_rs.rollout_features`): `(s, [(action, key, x(a))])`, in
/// [`sts2_core::legal_actions`] order — the order the playout sees. The v1
/// features; [`export_features_spec`] computes any version.
#[must_use]
pub fn export_features(
    state: &CombatState,
) -> (
    [f32; STATE_DIM],
    Vec<(Action, ActionKey, [f32; ACTION_DIM])>,
) {
    let ex = export_features_spec(state, &FEATURES_V1);
    let mut s = [0.0f32; STATE_DIM];
    s.copy_from_slice(&ex.state);
    let rows = ex
        .actions
        .into_iter()
        .map(|(a, k, x)| {
            let mut xa = [0.0f32; ACTION_DIM];
            xa.copy_from_slice(&x);
            (a, k, xa)
        })
        .collect();
    (s, rows)
}

/// [`export_features_spec`]'s output.
#[derive(Clone, Debug)]
pub struct ExportedFeatures {
    /// The state features `s` (the version's `state_dim` entries).
    pub state: Vec<f32>,
    /// Per legal action: the action, its key and `x(a)` (`action_dim` wide).
    pub actions: Vec<(Action, ActionKey, Vec<f32>)>,
    /// Species of the first [`MONSTER_SLOTS`] living enemies (v2; all
    /// `None` under v1), for the encounter embedding's vocabulary.
    pub species: [Option<MonsterKind>; MONSTER_SLOTS],
}

/// The features of `spec`'s version at `state`, for the training data.
#[must_use]
pub fn export_features_spec(state: &CombatState, spec: &FeatureSpec) -> ExportedFeatures {
    let actions = sts2_core::legal_actions(state);
    let ctx = StepCtx::with_spec(state, &actions, spec);
    let heur = crate::expert::heuristic_priorities_with(
        state,
        &actions,
        crate::expert::END_TURN_PRIORITY,
        spec.version.incoming(),
    );
    let ad = spec.version.action_dim();
    let rows = actions
        .iter()
        .zip(&heur)
        .map(|(a, &h)| {
            let (k, x) = action_features(state, &ctx, a, h);
            (a.clone(), k, x[..ad].to_vec())
        })
        .collect();
    ExportedFeatures {
        state: ctx.state[..spec.version.state_dim()].to_vec(),
        actions: rows,
        species: ctx.species,
    }
}

/// A playout net as the PUCT prior ([`crate::MctsConfig::use_puct_prior`]):
/// `P = (1 - mix) / n + mix * softmax(logits / temperature)` over the tree's
/// child edges. Experimental (behaviour cloning from human replays).
///
/// The net scores the full legal list at the node, in order, as a playout
/// would (its state features read that list). A child edge takes the summed
/// probability of every legal action with its [`crate::EdgeAction`], since
/// the tree merges identical cards into one edge.
#[derive(Debug)]
pub struct RolloutNetPrior {
    net: RolloutNet,
    mix: f32,
    temperature: f32,
}

impl RolloutNetPrior {
    /// A prior from `net` at blend weight `mix` (in `[0, 1]`) and softmax
    /// `temperature` (> 0).
    #[must_use]
    pub fn new(net: RolloutNet, mix: f32, temperature: f32) -> Self {
        Self {
            net,
            mix: mix.clamp(0.0, 1.0),
            temperature: temperature.max(1e-3),
        }
    }
}

impl crate::Policy for RolloutNetPrior {
    fn prior(&self, state: &CombatState, actions: &[Action]) -> SmallVec<[f32; 16]> {
        crate::prior_counts::count_eval();
        let n = actions.len().max(1);
        #[allow(clippy::cast_precision_loss)]
        let uniform = 1.0 / n as f32;
        let legal = sts2_core::legal_actions(state);
        let logits = pick_logits(
            &self.net,
            state,
            &legal,
            crate::expert::END_TURN_PRIORITY,
            false,
        );
        let top = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let w: SmallVec<[f32; 16]> = logits
            .iter()
            .map(|l| ((l - top) / self.temperature).exp())
            .collect();
        let edges: SmallVec<[crate::EdgeAction; 16]> = legal
            .iter()
            .map(|a| crate::EdgeAction::from_action(a, state))
            .collect();
        let mass: SmallVec<[f32; 16]> = actions
            .iter()
            .map(|a| {
                let e = crate::EdgeAction::from_action(a, state);
                edges
                    .iter()
                    .zip(&w)
                    .filter(|(x, _)| **x == e)
                    .map(|(_, w)| w)
                    .sum()
            })
            .collect();
        let total: f32 = mass.iter().sum();
        if total.is_nan() || total <= 1e-12 {
            crate::prior_counts::count_fallback();
            return actions.iter().map(|_| uniform).collect();
        }
        mass.iter()
            .map(|m| (1.0 - self.mix) * uniform + self.mix * m / total)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_net(keys: &[&str], bias: &[f32]) -> String {
        let (h, d) = (2usize, 2usize);
        let zeros = |r: usize, c: usize| vec![vec![0.0f32; c]; r];
        serde_json::json!({
            "state_dim": STATE_DIM, "action_dim": ACTION_DIM, "hidden": h, "emb_dim": d,
            "temperature": 1.0, "keys": keys,
            "w1": zeros(h, STATE_DIM), "b1": vec![0.0; h], "w2": zeros(d, h), "b2": vec![0.0; d],
            "emb": zeros(keys.len(), d), "bias": bias, "proj": zeros(d, ACTION_DIM),
            "lin": vec![0.0; ACTION_DIM],
        })
        .to_string()
    }

    #[test]
    fn keys_resolve_to_their_rows() {
        let net = RolloutNet::from_json(&tiny_net(
            &[
                "<unk>",
                "end_turn",
                "card:STRIKE_IRONCLAD",
                "card:STRIKE_IRONCLAD+",
                "potion:FIRE_POTION",
            ],
            &[0.0, 1.0, 2.0, 3.0, 4.0],
        ))
        .expect("parses");
        assert_eq!(net.row(ActionKey::EndTurn), 1);
        assert_eq!(
            net.row(ActionKey::Card {
                id: CardId::StrikeIronclad,
                upgraded: false
            }),
            2
        );
        assert_eq!(
            net.row(ActionKey::Card {
                id: CardId::StrikeIronclad,
                upgraded: true
            }),
            3
        );
        assert_eq!(net.row(ActionKey::Potion(PotionKind::FirePotion)), 4);
        assert_eq!(
            net.row(ActionKey::Card {
                id: CardId::Bash,
                upgraded: false
            }),
            0,
            "unseen card"
        );
        assert_eq!(net.row(ActionKey::Choice), 0, "absent special key");
    }

    #[test]
    fn rejects_mismatched_dims_and_unknown_keys() {
        let bad = tiny_net(&["<unk>", "bogus"], &[0.0, 0.0]);
        assert!(RolloutNet::from_json(&bad)
            .unwrap_err()
            .contains("unknown key"));
        let dims = tiny_net(&["<unk>"], &[0.0])
            .replace(&format!("\"state_dim\":{STATE_DIM}"), "\"state_dim\":3");
        assert!(RolloutNet::from_json(&dims)
            .unwrap_err()
            .contains("feature dims"));
    }

    #[test]
    fn key_names_round_trip_through_the_loader() {
        let k = ActionKey::Card {
            id: CardId::Bash,
            upgraded: true,
        };
        assert_eq!(k.name(), "card:BASH+");
        assert_eq!(
            ActionKey::Potion(PotionKind::FirePotion).name(),
            "potion:FIRE_POTION"
        );
    }

    #[test]
    fn logits_match_the_model_formula() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(3);
        // Widths that aren't multiples of 8 exercise `dot`'s tail.
        let (hidden, dim) = (11usize, 3usize);
        let keys = ["<unk>", "end_turn", "card:STRIKE_IRONCLAD"];
        let mut mat = |rows: usize, cols: usize| -> Vec<Vec<f32>> {
            (0..rows)
                .map(|_| (0..cols).map(|_| rng.gen_range(-1.0f32..1.0)).collect())
                .collect()
        };
        let (w1, b1) = (mat(hidden, STATE_DIM), mat(1, hidden).remove(0));
        let (w2, b2) = (mat(dim, hidden), mat(1, dim).remove(0));
        let (emb, bias) = (mat(keys.len(), dim), mat(1, keys.len()).remove(0));
        let (proj, lin) = (mat(dim, ACTION_DIM), mat(1, ACTION_DIM).remove(0));
        let state = mat(1, STATE_DIM).remove(0);
        let feats = mat(1, ACTION_DIM).remove(0);
        let net = RolloutNet::from_json(
            &serde_json::json!({
                "state_dim": STATE_DIM, "action_dim": ACTION_DIM, "hidden": hidden,
                "emb_dim": dim, "temperature": 1.0, "keys": keys, "w1": w1, "b1": b1,
                "w2": w2, "b2": b2, "emb": emb, "bias": bias, "proj": proj, "lin": lin,
            })
            .to_string(),
        )
        .expect("parses");
        let naive_dot = |u: &[f32], v: &[f32]| u.iter().zip(v).map(|(a, b)| a * b).sum::<f32>();
        let hid: Vec<f32> = w1
            .iter()
            .zip(&b1)
            .map(|(row, b)| (b + naive_dot(row, &state)).max(0.0))
            .collect();
        let query: Vec<f32> = w2
            .iter()
            .zip(&b2)
            .map(|(row, b)| b + naive_dot(row, &hid))
            .collect();
        let step = net.prepare_raw(&state, &[0; MONSTER_SLOTS]);
        let mut feat_arr = [0.0f32; ACTION_DIM_MAX];
        feat_arr[..ACTION_DIM].copy_from_slice(&feats);
        for row in 0..keys.len() {
            let want = bias[row]
                + naive_dot(&lin, &feats)
                + query
                    .iter()
                    .zip(&emb[row])
                    .zip(&proj)
                    .map(|((q_j, e_j), p_j)| q_j * (e_j + naive_dot(p_j, &feats)))
                    .sum::<f32>();
            let got = net.logit(&step, row, &feat_arr);
            assert!((want - got).abs() < 1e-4, "row {row}: {want} vs {got}");
        }
    }

    /// An `RngCore` that yields one fixed word, to pin `gen::<f32>()` at the
    /// ends of `[0, 1)`.
    struct ConstRng(u64);

    impl rand::RngCore for ConstRng {
        #[allow(clippy::cast_possible_truncation)]
        fn next_u32(&mut self) -> u32 {
            self.0 as u32
        }
        fn next_u64(&mut self) -> u64 {
            self.0
        }
        fn fill_bytes(&mut self, dest: &mut [u8]) {
            dest.fill(self.0 as u8);
        }
        fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
            self.fill_bytes(dest);
            Ok(())
        }
    }

    #[test]
    fn sampling_never_returns_a_masked_action() {
        let ninf = f32::NEG_INFINITY;
        for word in [0, u64::MAX] {
            for logits in [[ninf, 0.0, 1.0], [0.0, 1.0, ninf], [ninf, 0.0, ninf]] {
                let i = sample(&mut ConstRng(word), &logits, 1.0);
                assert!(logits[i].is_finite(), "draw {word:#x}, {logits:?} -> {i}");
            }
        }
    }

    #[test]
    fn sampling_follows_the_logits() {
        use rand::SeedableRng;
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(7);
        let logits = [0.0f32, 3.0, -50.0];
        let mut counts = [0u32; 3];
        for _ in 0..4000 {
            counts[sample(&mut rng, &logits, 1.0)] += 1;
        }
        // p = softmax([0, 3, -50]) ≈ [0.047, 0.953, 0].
        assert!(
            counts[1] > 3700 && counts[0] > 100 && counts[2] == 0,
            "{counts:?}"
        );
    }
}
