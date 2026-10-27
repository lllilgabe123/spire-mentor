//! C#-faithful run-setup room generation off the shared `up_front` RNG stream.
//!
//! Ports `RunManager.GenerateRooms` (`MegaCrit.Sts2.Core.Runs/RunManager.cs:483`)
//! + `ActModel.GenerateRooms` (`MegaCrit.Sts2.Core.Models/ActModel.cs:231`).
//! At run start the C# game pre-generates **every** act's room contents from the
//! single `up_front` stream, in this exact draw order (verified read-for-read
//! against the headless engine via `sts2-pycli/python/trace_run_rng.py`, seed
//! `RP0`, reads 231–413):
//!
//! 1. `SharedAncients.UnstableShuffle(up_front)` — 1 element → 0 reads.
//! 2. For acts 2,3 (`Acts.Skip(1)`): `up_front.NextInt(remaining + 1)` distributes
//!    the shared-ancient subset (2 reads).
//! 3. Per act, `ActModel.GenerateRooms`:
//!    a. events: `AllEvents.Concat(AllSharedEvents)` (epoch-filtered) →
//!       `UnstableShuffle` (`count - 1` reads). Under the headless `UnlockState.all`
//!       no epoch filtering happens, so `count = own_events + 18` shared.
//!    b. `NumberOfWeakEncounters` weak grabs, `BaseNumberOfRooms - NumberOfWeak`
//!       regular grabs, then 15 elite grabs — each an `AddWithoutRepeatingTags`
//!       over a refilling `GrabBag` (`NextDouble` rejection-sampling on shared
//!       `EncounterTag`s + identity).
//!    c. `Boss = up_front.NextItem(AllBossEncounters)` (1 read).
//!    d. `Ancient = up_front.NextItem(own_ancients ++ subset)` (1 read).
//!
//! The per-act encounter tables below are the C# `GenerateAllEncounters()` arrays
//! verbatim (in order — the `.Where(RoomType/IsWeak)` filters preserve it), with
//! each entry's `RoomType`, `IsWeak`, and `Tags`. Boss ids are the C# `Id.Entry`
//! (`UPPER_SNAKE`) so they canon-match the headless `BossEncounter.Id.Entry` in
//! the parity harness (`run_parity.py::_canon_boss`).

use crate::act_selection::ActVariant;
use crate::dotnet_rng::Rng;

/// The Ancient event selected for an act — the C# boss-relic mechanism.
/// Each act start runs one `AncientEventModel` (Neow for act 1; one of a
/// per-act pool for acts 2/3) which offers ~3 relic choices. Selected by
/// `up_front.NextItem(GetUnlockedAncients ++ sharedSubset)`
/// (`ActModel.cs:297`); the per-act ordered pools are the C# `AllAncients`
/// declaration order (`Hive.cs:24-28`, `Glory.cs:23-27`,
/// `Overgrowth.cs:26`) plus the shared subset `[Darv]`
/// (`SharedAncients`). Bead `sts2-rs-40ikw` (task #19, boss relics).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AncientId {
    Neow,
    Orobas,
    Pael,
    Tezcatara,
    Nonupeipe,
    Tanx,
    Vakuu,
    Darv,
}

impl AncientId {
    /// `EventModel.Id.Entry` — the SCREAMING_SNAKE class name, used as the
    /// salt for the per-event RNG seed (`EventModel.cs:193`). Single-word
    /// class names uppercase directly.
    #[must_use]
    pub fn entry(self) -> &'static str {
        match self {
            Self::Neow => "NEOW",
            Self::Orobas => "OROBAS",
            Self::Pael => "PAEL",
            Self::Tezcatara => "TEZCATARA",
            Self::Nonupeipe => "NONUPEIPE",
            Self::Tanx => "TANX",
            Self::Vakuu => "VAKUU",
            Self::Darv => "DARV",
        }
    }
}

/// Act 1 (Overgrowth) `AllAncients` (`Overgrowth.cs:26`): `[Neow]`.
const ACT1_ANCIENTS: &[AncientId] = &[AncientId::Neow];
/// Act 2 (Hive) `AllAncients` (`Hive.cs:24-28`), C# declaration order.
const ACT2_ANCIENTS: &[AncientId] = &[AncientId::Orobas, AncientId::Pael, AncientId::Tezcatara];
/// Act 3 (Glory) `AllAncients` (`Glory.cs:23-27`), C# declaration order.
const ACT3_ANCIENTS: &[AncientId] = &[AncientId::Nonupeipe, AncientId::Tanx, AncientId::Vakuu];
/// `ModelDb.SharedAncients` — the single shared ancient distributed to the
/// act-2/3 subset (`ActModel.cs:207-208`).
const SHARED_ANCIENT: AncientId = AncientId::Darv;

/// C# `EncounterTag` (`MegaCrit.Sts2.Core.Entities.Encounters/EncounterTag.cs`).
/// Only the discriminant identity matters here (shared-tag rejection), so we use
/// a small integer set.
type Tag = u8;

/// `ModelDb.AllSharedEvents` (`MegaCrit.Sts2.Core.Models/ModelDb.cs:97`) — the 18
/// shared events appended to **every** act's pool, in C# declaration order. These
/// are concatenated AFTER the act's own events before the `up_front`
/// `UnstableShuffle`, so the pre-shuffle order is `own ++ shared` exactly. Strings
/// are `EventId::python_name`s (resolved via `EventId::from_python_name` in
/// `pick_event`). Bead `sts2-rs-1hiht` (M5 event-pool parity).
const SHARED_EVENTS: &[&str] = &[
    "BrainLeech",
    "CrystalSphere",
    "DollRoom",
    "FakeMerchant",
    "PotionCourier",
    "RanwidTheElder",
    "RelicTrader",
    "RoomFullOfCheese",
    "SelfHelpBook",
    "SlipperyBridge",
    "StoneOfAllTime",
    "Symbiote",
    "TeaMaster",
    "TheFutureOfPotions",
    "TheLegendsWereTrue",
    "ThisOrThat",
    "WarHistorianRepy",
    "WelcomeToWongos",
];

/// Act 1 (Overgrowth) `AllEvents` (`Overgrowth.cs:28`) — 13 own events, C# order.
const ACT1_OWN_EVENTS: &[&str] = &[
    "AromaOfChaos",
    "ByrdonisNest",
    "DenseVegetation",
    "JungleMazeAdventure",
    "LuminousChoir",
    "MorphicGrove",
    "SapphireSeed",
    "SunkenStatue",
    "TabletOfTruth",
    "UnrestSite",
    "Wellspring",
    "WhisperingHollow",
    "WoodCarvings",
];

/// Act 1 (Underdocks) `AllEvents` (`Underdocks.cs:26`) — 10 own events, C#
/// order. Shares exactly ONE entry with Overgrowth's 13 (`SunkenStatue`);
/// the whole event pool swings with the act variant. GH #544.
const ACT1_UNDERDOCKS_OWN_EVENTS: &[&str] = &[
    "AbyssalBaths",
    "DrowningBeacon",
    "EndlessConveyor",
    "PunchOff",
    "SpiralingWhirlpool",
    "SunkenStatue",
    "SunkenTreasury",
    "DoorsOfLightAndDark",
    "TrashHeap",
    "WaterloggedScriptorium",
];

/// Act 2 (Hive) `AllEvents` (`Hive.cs`) — 10 own events, C# order.
const ACT2_OWN_EVENTS: &[&str] = &[
    "Amalgamator",
    "Bugslayer",
    "ColorfulPhilosophers",
    "ColossalFlower",
    "FieldOfManSizedHoles",
    "InfestedAutomaton",
    "LostWisp",
    "SpiritGrafter",
    "TheLanternKey",
    "ZenWeaver",
];

/// Act 3 (Glory) `AllEvents` (`Glory.cs`) — 7 own events, C# order.
const ACT3_OWN_EVENTS: &[&str] = &[
    "BattlewornDummy",
    "GraveOfTheForgotten",
    "HungryForMushrooms",
    "Reflections",
    "RoundTeaParty",
    "Trial",
    "TinkerTime",
];

/// One row of an act's `GenerateAllEncounters()` array.
#[derive(Clone, Copy)]
struct GenEnc {
    id: &'static str,
    room: RoomKind,
    weak: bool,
    tags: &'static [Tag],
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RoomKind {
    Monster,
    Elite,
    Boss,
}

/// Generated rooms for one act, in C# draw order. `boss_id` is the canonical
/// `UPPER_SNAKE` encounter id selected by `up_front.NextItem(AllBossEncounters)`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GeneratedActRooms {
    /// `normalEncounters` in draw order (weak block first, then regular).
    pub normal_encounters: Vec<&'static str>,
    /// `eliteEncounters` in draw order (15 entries).
    pub elite_encounters: Vec<&'static str>,
    /// Boss encounter id (`up_front.NextItem(AllBossEncounters)`).
    pub boss_id: &'static str,
    /// A10 DoubleBoss second-boss encounter id — `Some` only for the
    /// FINAL act of an ascension-10 run. C# `RunManager.GenerateRooms`
    /// (`RunManager.cs:501-504`): `UpFront.NextItem(
    /// AllBossEncounters.Where(e => e.Id != BossEncounter.Id))`, rolled
    /// immediately after the last act's rooms generate. Bead
    /// sts2-rs-40ikw.2.
    pub second_boss_id: Option<&'static str>,
    /// The act's event pool (`own ++ shared`) after the `up_front`
    /// `UnstableShuffle`, as `EventId::python_name`s in C# `RoomSet.events`
    /// order. `pick_event` walks this with the `events_visited` cursor +
    /// IsAllowed/visited filters. Byte-matches the headless `dump_rooms`
    /// `events` list. Bead `sts2-rs-1hiht` (M5 event-pool parity).
    pub events_shuffled: Vec<&'static str>,
    /// The Ancient event selected for this act — the boss-relic source
    /// (`up_front.NextItem(own_ancients ++ sharedSubset)`, `ActModel.cs:297`).
    /// `None` only on a `Default` placeholder; real generation always sets
    /// it (every act has a non-empty ancient pool). Bead `sts2-rs-40ikw`.
    pub ancient: Option<AncientId>,
}

/// Per-act static room-generation data.
struct ActGenData {
    encounters: &'static [GenEnc],
    /// The act's own event pool (`AllEvents`) in C# declaration order;
    /// `own ++ SHARED_EVENTS` is the pre-shuffle list. Bead `sts2-rs-1hiht`.
    own_events: &'static [&'static str],
    num_weak: usize,
    base_rooms: usize,
    /// The act's own ancient pool (`AllAncients`) in C# declaration order.
    /// `own ++ sharedSubset` is the `up_front.NextItem` pool that selects
    /// the act's Ancient event (`ActModel.cs:297`).
    own_ancients: &'static [AncientId],
}

// ── Encounter tag constants (discriminant identity only) ────────────────────
const BURROWER: Tag = 1;
const CHOMPER: Tag = 2;
const NIBBIT: Tag = 3;
const SHRINKER: Tag = 4;
const SLIMES: Tag = 5;
const THIEVES: Tag = 6;
const WORKERS: Tag = 7;
const CRAWLER: Tag = 8;
const MUSHROOM: Tag = 9;
const KNIGHTS: Tag = 10;
const SCROLLS: Tag = 11;
const SEAPUNK: Tag = 12;
const SLUGS: Tag = 13;
const EXOSKELETONS: Tag = 14;
/// v0.111.0-only `EncounterTag.Jaxfruit` (decomp `EncounterTag.cs:20`).
const JAXFRUIT: Tag = 15;

// ── Act 1 (Overgrowth) — Overgrowth.cs GenerateAllEncounters (22) ───────────
#[rustfmt::skip]
const ACT1: &[GenEnc] = &[
    GenEnc { id: "BYGONE_EFFIGY_ELITE",      room: RoomKind::Elite,   weak: false, tags: &[] },
    GenEnc { id: "BYRDONIS_ELITE",           room: RoomKind::Elite,   weak: false, tags: &[] },
    GenEnc { id: "CEREMONIAL_BEAST_BOSS",    room: RoomKind::Boss,    weak: false, tags: &[] },
    GenEnc { id: "CUBEX_CONSTRUCT_NORMAL",   room: RoomKind::Monster, weak: false, tags: &[] },
    GenEnc { id: "FLYCONID_NORMAL",          room: RoomKind::Monster, weak: false, tags: &[MUSHROOM, SLIMES] },
    GenEnc { id: "FOGMOG_NORMAL",            room: RoomKind::Monster, weak: false, tags: &[] },
    GenEnc { id: "FUZZY_WURM_CRAWLER_WEAK",  room: RoomKind::Monster, weak: true,  tags: &[CRAWLER] },
    GenEnc { id: "INKLETS_NORMAL",           room: RoomKind::Monster, weak: false, tags: &[] },
    GenEnc { id: "MAWLER_NORMAL",            room: RoomKind::Monster, weak: false, tags: &[] },
    GenEnc { id: "NIBBITS_NORMAL",           room: RoomKind::Monster, weak: false, tags: &[] },
    GenEnc { id: "NIBBITS_WEAK",             room: RoomKind::Monster, weak: true,  tags: &[NIBBIT] },
    GenEnc { id: "OVERGROWTH_CRAWLERS",      room: RoomKind::Monster, weak: false, tags: &[SHRINKER, CRAWLER] },
    GenEnc { id: "PHROG_PARASITE_ELITE",     room: RoomKind::Elite,   weak: false, tags: &[] },
    GenEnc { id: "RUBY_RAIDERS_NORMAL",      room: RoomKind::Monster, weak: false, tags: &[] },
    GenEnc { id: "SHRINKER_BEETLE_WEAK",     room: RoomKind::Monster, weak: true,  tags: &[SHRINKER] },
    GenEnc { id: "SLIMES_NORMAL",            room: RoomKind::Monster, weak: false, tags: &[SLIMES] },
    GenEnc { id: "SLIMES_WEAK",              room: RoomKind::Monster, weak: true,  tags: &[SLIMES] },
    GenEnc { id: "SLITHERING_STRANGLER_NORMAL", room: RoomKind::Monster, weak: false, tags: &[] },
    GenEnc { id: "SNAPPING_JAXFRUIT_NORMAL", room: RoomKind::Monster, weak: false, tags: &[MUSHROOM] },
    GenEnc { id: "THE_KIN_BOSS",             room: RoomKind::Boss,    weak: false, tags: &[] },
    GenEnc { id: "VANTOM_BOSS",              room: RoomKind::Boss,    weak: false, tags: &[] },
    GenEnc { id: "VINE_SHAMBLER_NORMAL",     room: RoomKind::Monster, weak: false, tags: &[] },
];

// ── Act 1 ALTERNATE (Underdocks) — Underdocks.cs GenerateAllEncounters (20) ─
//
// The second act model at `Index => 0`. `IsDefault` is false and it is gated
// on `UnderdocksEpoch`, but on an unlocked save the run picks between this and
// Overgrowth off the dedicated `act_selection` stream — see
// [`crate::act_selection`]. Tags come from each encounter's `Tags` override:
// only the two Corpse Slug rows (`Slugs`) and the two Seapunk rows (`Seapunk`)
// carry any. GH #544.
#[rustfmt::skip]
const ACT1_UNDERDOCKS: &[GenEnc] = &[
    GenEnc { id: "CORPSE_SLUGS_NORMAL",       room: RoomKind::Monster, weak: false, tags: &[SLUGS] },
    GenEnc { id: "CORPSE_SLUGS_WEAK",         room: RoomKind::Monster, weak: true,  tags: &[SLUGS] },
    GenEnc { id: "CULTISTS_NORMAL",           room: RoomKind::Monster, weak: false, tags: &[] },
    GenEnc { id: "FOSSIL_STALKER_NORMAL",     room: RoomKind::Monster, weak: false, tags: &[] },
    GenEnc { id: "GREMLIN_MERC_NORMAL",       room: RoomKind::Monster, weak: false, tags: &[] },
    GenEnc { id: "HAUNTED_SHIP_NORMAL",       room: RoomKind::Monster, weak: false, tags: &[] },
    GenEnc { id: "LAGAVULIN_MATRIARCH_BOSS",  room: RoomKind::Boss,    weak: false, tags: &[] },
    GenEnc { id: "LIVING_FOG_NORMAL",         room: RoomKind::Monster, weak: false, tags: &[] },
    GenEnc { id: "PHANTASMAL_GARDENERS_ELITE", room: RoomKind::Elite,  weak: false, tags: &[] },
    GenEnc { id: "PUNCH_CONSTRUCT_NORMAL",    room: RoomKind::Monster, weak: false, tags: &[] },
    GenEnc { id: "SEAPUNK_NORMAL",            room: RoomKind::Monster, weak: false, tags: &[SEAPUNK] },
    GenEnc { id: "SEAPUNK_WEAK",              room: RoomKind::Monster, weak: true,  tags: &[SEAPUNK] },
    GenEnc { id: "SEWER_CLAM_NORMAL",         room: RoomKind::Monster, weak: false, tags: &[] },
    GenEnc { id: "SKULKING_COLONY_ELITE",     room: RoomKind::Elite,   weak: false, tags: &[] },
    GenEnc { id: "SLUDGE_SPINNER_WEAK",       room: RoomKind::Monster, weak: true,  tags: &[] },
    GenEnc { id: "SOUL_FYSH_BOSS",            room: RoomKind::Boss,    weak: false, tags: &[] },
    GenEnc { id: "TERROR_EEL_ELITE",          room: RoomKind::Elite,   weak: false, tags: &[] },
    GenEnc { id: "TOADPOLES_WEAK",            room: RoomKind::Monster, weak: true,  tags: &[] },
    GenEnc { id: "TWO_TAILED_RATS_NORMAL",    room: RoomKind::Monster, weak: false, tags: &[] },
    GenEnc { id: "WATERFALL_GIANT_BOSS",      room: RoomKind::Boss,    weak: false, tags: &[] },
];

// ── Act 2 (Hive) — Hive.cs GenerateAllEncounters (20) ───────────────────────
#[rustfmt::skip]
const ACT2: &[GenEnc] = &[
    GenEnc { id: "BOWLBUGS_NORMAL",          room: RoomKind::Monster, weak: false, tags: &[WORKERS] },
    GenEnc { id: "BOWLBUGS_WEAK",            room: RoomKind::Monster, weak: true,  tags: &[WORKERS] },
    GenEnc { id: "CHOMPERS_NORMAL",          room: RoomKind::Monster, weak: false, tags: &[CHOMPER] },
    GenEnc { id: "DECIMILLIPEDE_ELITE",      room: RoomKind::Elite,   weak: false, tags: &[] },
    GenEnc { id: "ENTOMANCER_ELITE",         room: RoomKind::Elite,   weak: false, tags: &[] },
    GenEnc { id: "EXOSKELETONS_NORMAL",      room: RoomKind::Monster, weak: false, tags: &[EXOSKELETONS] },
    GenEnc { id: "EXOSKELETONS_WEAK",        room: RoomKind::Monster, weak: true,  tags: &[EXOSKELETONS] },
    GenEnc { id: "HUNTER_KILLER_NORMAL",     room: RoomKind::Monster, weak: false, tags: &[] },
    GenEnc { id: "KAISER_CRAB_BOSS",         room: RoomKind::Boss,    weak: false, tags: &[] },
    GenEnc { id: "INFESTED_PRISMS_ELITE",    room: RoomKind::Elite,   weak: false, tags: &[] },
    GenEnc { id: "KNOWLEDGE_DEMON_BOSS",     room: RoomKind::Boss,    weak: false, tags: &[] },
    GenEnc { id: "LOUSE_PROGENITOR_NORMAL",  room: RoomKind::Monster, weak: false, tags: &[] },
    GenEnc { id: "MYTES_NORMAL",             room: RoomKind::Monster, weak: false, tags: &[] },
    GenEnc { id: "OVICOPTER_NORMAL",         room: RoomKind::Monster, weak: false, tags: &[] },
    GenEnc { id: "SLUMBERING_BEETLE_NORMAL", room: RoomKind::Monster, weak: false, tags: &[WORKERS] },
    GenEnc { id: "SPINY_TOAD_NORMAL",        room: RoomKind::Monster, weak: false, tags: &[] },
    GenEnc { id: "THE_INSATIABLE_BOSS",      room: RoomKind::Boss,    weak: false, tags: &[] },
    GenEnc { id: "THE_OBSCURA_NORMAL",       room: RoomKind::Monster, weak: false, tags: &[] },
    GenEnc { id: "THIEVING_HOPPER_WEAK",     room: RoomKind::Monster, weak: true,  tags: &[THIEVES] },
    GenEnc { id: "TUNNELER_WEAK",            room: RoomKind::Monster, weak: true,  tags: &[BURROWER] },
];

// ── Act 3 (Glory) — Glory.cs GenerateAllEncounters (18) ─────────────────────
#[rustfmt::skip]
const ACT3: &[GenEnc] = &[
    GenEnc { id: "AXEBOTS_NORMAL",               room: RoomKind::Monster, weak: false, tags: &[] },
    GenEnc { id: "CONSTRUCT_MENAGERIE_NORMAL",   room: RoomKind::Monster, weak: false, tags: &[] },
    GenEnc { id: "DEVOTED_SCULPTOR_WEAK",        room: RoomKind::Monster, weak: true,  tags: &[] },
    GenEnc { id: "AEONGLASS_BOSS",               room: RoomKind::Boss,    weak: false, tags: &[] },
    GenEnc { id: "FABRICATOR_NORMAL",            room: RoomKind::Monster, weak: false, tags: &[] },
    GenEnc { id: "FROG_KNIGHT_NORMAL",           room: RoomKind::Monster, weak: false, tags: &[] },
    GenEnc { id: "GLOBE_HEAD_NORMAL",            room: RoomKind::Monster, weak: false, tags: &[] },
    GenEnc { id: "KNIGHTS_ELITE",                room: RoomKind::Elite,   weak: false, tags: &[KNIGHTS] },
    GenEnc { id: "MECHA_KNIGHT_ELITE",           room: RoomKind::Elite,   weak: false, tags: &[] },
    GenEnc { id: "OWL_MAGISTRATE_NORMAL",        room: RoomKind::Monster, weak: false, tags: &[] },
    GenEnc { id: "QUEEN_BOSS",                   room: RoomKind::Boss,    weak: false, tags: &[] },
    GenEnc { id: "SCROLLS_OF_BITING_NORMAL",     room: RoomKind::Monster, weak: false, tags: &[SCROLLS] },
    GenEnc { id: "SCROLLS_OF_BITING_WEAK",       room: RoomKind::Monster, weak: true,  tags: &[SCROLLS] },
    GenEnc { id: "SLIMED_BERSERKER_NORMAL",      room: RoomKind::Monster, weak: false, tags: &[] },
    GenEnc { id: "SOUL_NEXUS_ELITE",             room: RoomKind::Elite,   weak: false, tags: &[] },
    GenEnc { id: "TEST_SUBJECT_BOSS",            room: RoomKind::Boss,    weak: false, tags: &[] },
    GenEnc { id: "THE_LOST_AND_FORGOTTEN_NORMAL", room: RoomKind::Monster, weak: false, tags: &[] },
    GenEnc { id: "TURRET_OPERATOR_WEAK",         room: RoomKind::Monster, weak: true,  tags: &[] },
];

/// Act 1 (Overgrowth) — `IsDefault`, the variant this engine used to model
/// as the only Act 1.
const ACT1_OVERGROWTH_DATA: ActGenData = ActGenData {
    encounters: ACT1,
    own_events: ACT1_OWN_EVENTS,
    num_weak: 3,
    base_rooms: 15,
    own_ancients: ACT1_ANCIENTS,
};

/// Act 1 (Underdocks) — the alternate at the same index. `BaseNumberOfRooms`
/// (15), `NumberOfWeakEncounters` (3) and `AllAncients` (`[Neow]`) are all
/// identical to Overgrowth's; the encounter table and the event pool are not.
/// GH #544.
const ACT1_UNDERDOCKS_DATA: ActGenData = ActGenData {
    encounters: ACT1_UNDERDOCKS,
    own_events: ACT1_UNDERDOCKS_OWN_EVENTS,
    num_weak: 3,
    base_rooms: 15,
    own_ancients: ACT1_ANCIENTS,
};

const ACT2_DATA: ActGenData = ActGenData {
    encounters: ACT2,
    own_events: ACT2_OWN_EVENTS,
    num_weak: 2,
    base_rooms: 14,
    own_ancients: ACT2_ANCIENTS,
};

const ACT3_DATA: ActGenData = ActGenData {
    encounters: ACT3,
    own_events: ACT3_OWN_EVENTS,
    num_weak: 2,
    base_rooms: 13,
    own_ancients: ACT3_ANCIENTS,
};

/// Every act model the game ships, INCLUDING both Act-1 variants. This is the
/// universe [`all_boss_encounter_ids`] unions — deliberately wider than what
/// any single run walks, because the PPO boss vocab has to cover every boss a
/// run *could* roll, not just the ones the default act-1 variant offers.
const ALL_ACT_MODELS: [&ActGenData; 4] = [
    &ACT1_OVERGROWTH_DATA,
    &ACT1_UNDERDOCKS_DATA,
    &ACT2_DATA,
    &ACT3_DATA,
];

/// The three acts one run walks, given its act-1 variant. Index 0 is the
/// variant the run rolled (or was forced to); indices 1 and 2 are singletons.
fn acts_for(act1: ActVariant) -> [&'static ActGenData; 3] {
    let first = match act1 {
        ActVariant::Underdocks => &ACT1_UNDERDOCKS_DATA,
        // Hive/Glory are not act-1 models; `ActVariant` carries them so the
        // selection port can name every act, and one landing in the act-1
        // slot is a caller bug. Defaulting keeps generation total.
        _ => &ACT1_OVERGROWTH_DATA,
    };
    [first, &ACT2_DATA, &ACT3_DATA]
}

/// Every boss-encounter id that map generation can roll, across every act
/// model the game ships — **both** Act-1 variants included — sorted and
/// deduplicated.
///
/// This is the same `RoomKind::Boss` filter [`generate_act_rooms`] applies to
/// pick [`GeneratedActRooms::boss_id`], so the two can never drift: the ids
/// this returns are exactly the ids that can appear as an act's boss. The PPO
/// run-policy vocab builds its boss-identity embedding table from here
/// (`sts2_vocab::registry_names("boss")`), which is what lets the
/// net condition its deck-shaping choices on the boss it is actually walking
/// toward. Hand-copying the list into the vocab would let a content change
/// silently send a real boss to the OOV row.
///
/// The union spans [`ALL_ACT_MODELS`], not the three acts one run walks: a
/// vocab that tracked the *default* act-1 variant would OOV Lagavulin
/// Matriarch the moment a run rolled Underdocks. Widening it is what makes
/// the boss rows content-complete — and why adding Underdocks needed the
/// append-only vocab pin (GH #544).
#[must_use]
pub fn all_boss_encounter_ids() -> Vec<&'static str> {
    let mut ids: Vec<&'static str> = ALL_ACT_MODELS
        .iter()
        .flat_map(|a| a.encounters.iter())
        .filter(|e| e.room == RoomKind::Boss)
        .map(|e| e.id)
        .collect();
    ids.sort_unstable();
    ids.dedup();
    ids
}

/// `GrabBag<EncounterModel>` (`MegaCrit.Sts2.Core.Helpers/GrabBag.cs`) — weighted
/// draw-without-replacement with `NextDouble`-driven `GrabIndex`. All weights are
/// `1.0` here (`grabBag.Add(e, 1.0)`).
struct GrabBag {
    entries: Vec<&'static GenEnc>,
    total_weight: f64,
}

impl GrabBag {
    fn new() -> Self {
        Self {
            entries: Vec::new(),
            total_weight: 0.0,
        }
    }

    fn any(&self) -> bool {
        !self.entries.is_empty()
    }

    fn add(&mut self, e: &'static GenEnc) {
        self.entries.push(e);
        self.total_weight += 1.0;
    }

    /// `GrabBag.GrabIndex(rng)` — `num = NextDouble() * totalWeight`, walk the
    /// cumulative weights. Returns `None` only for an empty bag.
    fn grab_index(&self, rng: &mut Rng) -> Option<usize> {
        let num = rng.next_double() * self.total_weight;
        let mut acc = 0.0;
        for (i, _) in self.entries.iter().enumerate() {
            acc += 1.0;
            if num < acc {
                return Some(i);
            }
        }
        None
    }

    /// `GrabBag.GrabAndRemove(rng, predicate)`: rejection-sample `GrabIndex`
    /// until the drawn entry satisfies `predicate`, then remove + return it.
    /// The `predicate is None` → all-pass form is the `GrabAndRemove(rng)`
    /// fallback. Mirrors the IL exactly: the `!entries.any(predicate)` precheck
    /// short-circuits with **no** RNG read.
    fn grab_and_remove(
        &mut self,
        rng: &mut Rng,
        predicate: Option<&dyn Fn(&GenEnc) -> bool>,
    ) -> Option<&'static GenEnc> {
        if let Some(pred) = predicate {
            if !self.entries.iter().any(|e| pred(e)) {
                return None;
            }
        }
        let idx = loop {
            let candidate = self.grab_index(rng)?;
            let ok = match predicate {
                Some(pred) => pred(self.entries[candidate]),
                None => true,
            };
            if ok {
                break candidate;
            }
        };
        let e = self.entries.remove(idx);
        self.total_weight -= 1.0;
        Some(e)
    }
}

/// An encounter's `Tags` in game build `version`. The static tables hold the
/// v0.107.1 tags; v0.111.0 retagged two Overgrowth normals with the new
/// `Jaxfruit` tag (so they no longer roll back to back), and gave
/// SlitheringStranglerNormal the `Slimes` tag it lacked:
///
/// [decompiled C# omitted — see the cited path]
fn tags_for(e: &GenEnc, version: crate::GameVersion) -> &'static [Tag] {
    if version >= crate::GameVersion::V0_111_0 {
        match e.id {
            "SLITHERING_STRANGLER_NORMAL" => return &[JAXFRUIT, SLIMES],
            "SNAPPING_JAXFRUIT_NORMAL" => return &[MUSHROOM, JAXFRUIT],
            _ => {}
        }
    }
    e.tags
}

fn shares_tags(a: &GenEnc, b: &GenEnc, version: crate::GameVersion) -> bool {
    let b_tags = tags_for(b, version);
    tags_for(a, version).iter().any(|t| b_tags.contains(t))
}

/// `ActModel.AddWithoutRepeatingTags` (`ActModel.cs:328`): grab an encounter that
/// shares no tag with — and is not — the last-added one; if none qualifies, fall
/// back to an unconstrained grab. Pushes the result onto `out`.
fn add_without_repeating_tags(
    out: &mut Vec<&'static str>,
    last: Option<&'static GenEnc>,
    bag: &mut GrabBag,
    rng: &mut Rng,
    version: crate::GameVersion,
) {
    let picked = {
        let last_id = last.map(|l| l.id);
        let pred = |e: &GenEnc| match last {
            Some(l) => !shares_tags(e, l, version) && e.id != last_id.unwrap(),
            None => true,
        };
        bag.grab_and_remove(rng, Some(&pred))
            .or_else(|| bag.grab_and_remove(rng, None))
    };
    if let Some(e) = picked {
        out.push(e.id);
    }
}

/// `ActModel.GenerateRooms` for one act. `subset_ancients` is the act's share of
/// the distributed shared ancients (Act 1 always 0). Consumes `up_front` exactly
/// as the C# does.
fn generate_act_rooms(
    rng: &mut Rng,
    act: &ActGenData,
    subset_ancients: usize,
    version: crate::GameVersion,
) -> GeneratedActRooms {
    let enc = act.encounters;

    // (a) events: `AllEvents.Concat(AllSharedEvents).UnstableShuffle(up_front)`.
    // Pre-shuffle order is `own ++ shared` (no epoch filtering under the headless
    // `UnlockState.all`). The shuffle both drives the `up_front` cursor AND yields
    // the C#-faithful `RoomSet.events` identity order. Bead `sts2-rs-1hiht`.
    let mut events_shuffled: Vec<&'static str> = act
        .own_events
        .iter()
        .copied()
        .chain(SHARED_EVENTS.iter().copied())
        .collect();
    rng.shuffle(&mut events_shuffled);

    // Derived sublists, in `AllEncounters` order (LINQ `.Where` preserves order).
    let weak: Vec<&GenEnc> = enc
        .iter()
        .filter(|e| e.room == RoomKind::Monster && e.weak)
        .collect();
    let regular: Vec<&GenEnc> = enc
        .iter()
        .filter(|e| e.room == RoomKind::Monster && !e.weak)
        .collect();
    let elite: Vec<&GenEnc> = enc.iter().filter(|e| e.room == RoomKind::Elite).collect();
    let boss: Vec<&GenEnc> = enc.iter().filter(|e| e.room == RoomKind::Boss).collect();

    let mut normal_encounters: Vec<&'static str> = Vec::new();

    // (b1) weak encounters
    let mut bag = GrabBag::new();
    let mut last: Option<&'static GenEnc> = None;
    for _ in 0..act.num_weak {
        if !bag.any() {
            for e in &weak {
                bag.add(e);
            }
        }
        add_without_repeating_tags(&mut normal_encounters, last, &mut bag, rng, version);
        last = normal_encounters
            .last()
            .and_then(|id| enc.iter().find(|e| e.id == *id));
    }

    // (b2) regular encounters (`NumberOfWeakEncounters .. GetNumberOfRooms`)
    let mut bag = GrabBag::new();
    for _ in act.num_weak..act.base_rooms {
        if !bag.any() {
            for e in &regular {
                bag.add(e);
            }
        }
        add_without_repeating_tags(&mut normal_encounters, last, &mut bag, rng, version);
        last = normal_encounters
            .last()
            .and_then(|id| enc.iter().find(|e| e.id == *id));
    }

    // (b3) elite encounters (fixed 15)
    let mut elite_encounters: Vec<&'static str> = Vec::new();
    let mut bag = GrabBag::new();
    let mut last_elite: Option<&'static GenEnc> = None;
    for _ in 0..15 {
        if !bag.any() {
            for e in &elite {
                bag.add(e);
            }
        }
        add_without_repeating_tags(&mut elite_encounters, last_elite, &mut bag, rng, version);
        last_elite = elite_encounters
            .last()
            .and_then(|id| enc.iter().find(|e| e.id == *id));
    }

    // (c) Boss = up_front.NextItem(AllBossEncounters)
    let boss_idx = rng.next_int_exclusive(0, boss.len() as i32) as usize;
    let boss_id = boss[boss_idx].id;

    // (d) Ancient = up_front.NextItem(own_ancients ++ subset). The pool is
    // the act's own ancients (C# `AllAncients` order) followed by its share
    // of the shared subset (`[Darv]`, distributed across acts 2/3). One read
    // (`NextItem` = `NextInt(0, count - 1)`), identical to the prior
    // count-only consumption — capturing the identity is read-neutral.
    let mut ancient_pool: Vec<AncientId> = act.own_ancients.to_vec();
    ancient_pool.extend(std::iter::repeat_n(SHARED_ANCIENT, subset_ancients));
    let ancient = if ancient_pool.is_empty() {
        None
    } else {
        let idx = rng.next_int_exclusive(0, ancient_pool.len() as i32) as usize;
        Some(ancient_pool[idx])
    };

    GeneratedActRooms {
        normal_encounters,
        elite_encounters,
        boss_id,
        second_boss_id: None,
        events_shuffled,
        ancient,
    }
}

/// `RunManager.GenerateRooms` — generate all three acts from `up_front`, in C#
/// order. Call once at run start, immediately after the relic grab-bags are
/// populated (which already consumed the first ~230 `up_front` reads).
///
/// `has_double_boss` (A10 DoubleBoss) appends ONE extra `up_front` read after
/// the LAST act's rooms generate: `SecondBoss = UpFront.NextItem(
/// AllBossEncounters.Where(e => e.Id != BossEncounter.Id))`
/// (`RunManager.cs:501-504`; `Rng.NextItem` = one `NextInt(0, count)` over the
/// filtered list in declaration order, `Rng.cs:157-167`). Below A10 the read
/// sequence is unchanged. Bead sts2-rs-40ikw.2.
///
/// `act1` picks which of the two Act-1 models the run walks. It is decided
/// **before** this function, on its own `act_selection` stream
/// (`StartRunLobby.BeginRunLocally`, ported in [`crate::act_selection`]), and
/// costs no `up_front` reads of its own.
///
/// It does, however, move the `up_front` cursor for everything after it.
/// Overgrowth declares 13 own events and Underdocks 10, so act 1's
/// `UnstableShuffle` over `own ++ 18 shared` consumes 30 reads instead of 27.
/// That is faithful — C# shuffles the act it actually picked — but it means
/// the same seed under the two variants diverges from act 1's event shuffle
/// onward, acts 2 and 3 included. The two variants are NOT a swap of act-1
/// content on a shared spine. GH #544.
#[must_use]
pub fn generate_all_rooms(
    up_front: &mut Rng,
    has_double_boss: bool,
    act1: ActVariant,
) -> Vec<GeneratedActRooms> {
    generate_all_rooms_for(
        crate::GameVersion::V0_107_1,
        up_front,
        has_double_boss,
        act1,
    )
}

/// [`generate_all_rooms`] for game build `version` (GH #792). The only
/// v0.111.0 difference in room generation is the encounter tags the
/// no-repeat rule compares ([`tags_for`]); the RNG call sequence is the same.
#[must_use]
pub fn generate_all_rooms_for(
    version: crate::GameVersion,
    up_front: &mut Rng,
    has_double_boss: bool,
    act1: ActVariant,
) -> Vec<GeneratedActRooms> {
    let acts = acts_for(act1);
    // (1) SharedAncients.UnstableShuffle — 1 element ([Darv]) → 0 reads. Then
    // (2) distribute the subset among acts 2,3 (`Acts.Skip(1)`).
    let mut remaining_shared_ancients: i32 = 1;
    let mut subset = [0usize; 3]; // index by act (0-based); act 0 gets none.
    for act_subset in subset.iter_mut().skip(1) {
        let count = up_front.next_int_exclusive(0, remaining_shared_ancients + 1);
        *act_subset = count as usize;
        remaining_shared_ancients -= count;
    }

    (0..3)
        .map(|i| {
            let mut rooms = generate_act_rooms(up_front, acts[i], subset[i], version);
            if i == 2 && has_double_boss {
                // Last act only (`i == State.Acts.Count - 1`,
                // `RunManager.cs:501`): pick the second boss from the
                // act's boss pool minus the already-chosen boss,
                // preserving `AllEncounters` declaration order (LINQ
                // `.Where` keeps it).
                let candidates: Vec<&'static str> = acts[i]
                    .encounters
                    .iter()
                    .filter(|e| e.room == RoomKind::Boss && e.id != rooms.boss_id)
                    .map(|e| e.id)
                    .collect();
                if !candidates.is_empty() {
                    let idx = up_front.next_int_exclusive(0, candidates.len() as i32) as usize;
                    rooms.second_boss_id = Some(candidates[idx]);
                }
            }
            rooms
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn act1_sublists_match_csharp_filters() {
        let weak: Vec<_> = ACT1
            .iter()
            .filter(|e| e.room == RoomKind::Monster && e.weak)
            .map(|e| e.id)
            .collect();
        assert_eq!(
            weak,
            [
                "FUZZY_WURM_CRAWLER_WEAK",
                "NIBBITS_WEAK",
                "SHRINKER_BEETLE_WEAK",
                "SLIMES_WEAK"
            ]
        );
        let boss: Vec<_> = ACT1
            .iter()
            .filter(|e| e.room == RoomKind::Boss)
            .map(|e| e.id)
            .collect();
        assert_eq!(
            boss,
            ["CEREMONIAL_BEAST_BOSS", "THE_KIN_BOSS", "VANTOM_BOSS"]
        );
        let regular = ACT1
            .iter()
            .filter(|e| e.room == RoomKind::Monster && !e.weak)
            .count();
        assert_eq!(regular, 12);
        let elite = ACT1.iter().filter(|e| e.room == RoomKind::Elite).count();
        assert_eq!(elite, 3);
    }

    /// Every Underdocks row, checked against
    /// `Underdocks.GenerateAllEncounters` (v0.107.1) through the same
    /// `.Where(RoomType/IsWeak)` filters `generate_act_rooms` applies.
    /// The counts differ from Overgrowth's (4/12/3/3): Underdocks has one
    /// more weak and two fewer regular, so a table transcribed with the
    /// wrong `weak` flag would change how many `GrabBag` refills the weak
    /// block needs and desync the whole act's draw order.
    #[test]
    fn act1_underdocks_sublists_match_csharp_filters() {
        assert_eq!(ACT1_UNDERDOCKS.len(), 20);
        let weak: Vec<_> = ACT1_UNDERDOCKS
            .iter()
            .filter(|e| e.room == RoomKind::Monster && e.weak)
            .map(|e| e.id)
            .collect();
        assert_eq!(
            weak,
            [
                "CORPSE_SLUGS_WEAK",
                "SEAPUNK_WEAK",
                "SLUDGE_SPINNER_WEAK",
                "TOADPOLES_WEAK"
            ]
        );
        let regular: Vec<_> = ACT1_UNDERDOCKS
            .iter()
            .filter(|e| e.room == RoomKind::Monster && !e.weak)
            .map(|e| e.id)
            .collect();
        assert_eq!(
            regular,
            [
                "CORPSE_SLUGS_NORMAL",
                "CULTISTS_NORMAL",
                "FOSSIL_STALKER_NORMAL",
                "GREMLIN_MERC_NORMAL",
                "HAUNTED_SHIP_NORMAL",
                "LIVING_FOG_NORMAL",
                "PUNCH_CONSTRUCT_NORMAL",
                "SEAPUNK_NORMAL",
                "SEWER_CLAM_NORMAL",
                "TWO_TAILED_RATS_NORMAL",
            ]
        );
        let elite: Vec<_> = ACT1_UNDERDOCKS
            .iter()
            .filter(|e| e.room == RoomKind::Elite)
            .map(|e| e.id)
            .collect();
        assert_eq!(
            elite,
            [
                "PHANTASMAL_GARDENERS_ELITE",
                "SKULKING_COLONY_ELITE",
                "TERROR_EEL_ELITE"
            ]
        );
        let boss: Vec<_> = ACT1_UNDERDOCKS
            .iter()
            .filter(|e| e.room == RoomKind::Boss)
            .map(|e| e.id)
            .collect();
        assert_eq!(
            boss,
            [
                "LAGAVULIN_MATRIARCH_BOSS",
                "SOUL_FYSH_BOSS",
                "WATERFALL_GIANT_BOSS"
            ]
        );
    }

    /// The two Act-1 variants share NO encounter. If a future edit made them
    /// overlap, an encounter-id-keyed lookup could resolve to the wrong act's
    /// row without any test noticing.
    #[test]
    fn the_two_act1_variants_share_no_encounter() {
        for u in ACT1_UNDERDOCKS {
            assert!(
                !ACT1.iter().any(|o| o.id == u.id),
                "{} appears in both Act-1 variants",
                u.id
            );
        }
    }

    /// Tag identity is what `AddWithoutRepeatingTags` rejects on, so a
    /// mis-transcribed tag silently changes the encounter ORDER (not just the
    /// composition) for every seed. Only four Underdocks rows carry one.
    #[test]
    fn act1_underdocks_tags_match_the_csharp_overrides() {
        let tagged: Vec<(&str, &[Tag])> = ACT1_UNDERDOCKS
            .iter()
            .filter(|e| !e.tags.is_empty())
            .map(|e| (e.id, e.tags))
            .collect();
        assert_eq!(
            tagged,
            [
                ("CORPSE_SLUGS_NORMAL", &[SLUGS][..]),
                ("CORPSE_SLUGS_WEAK", &[SLUGS][..]),
                ("SEAPUNK_NORMAL", &[SEAPUNK][..]),
                ("SEAPUNK_WEAK", &[SEAPUNK][..]),
            ]
        );
    }

    #[test]
    fn boss_pools_are_three_each() {
        for a in ALL_ACT_MODELS {
            assert_eq!(
                a.encounters
                    .iter()
                    .filter(|e| e.room == RoomKind::Boss)
                    .count(),
                3
            );
        }
    }

    /// A fixed seed produces a deterministic, in-range boss per act and the
    /// expected room-list lengths (weak+regular normals, 15 elites).
    #[test]
    fn generate_all_rooms_shapes() {
        for act1 in [ActVariant::Overgrowth, ActVariant::Underdocks] {
            let acts = acts_for(act1);
            let mut rng = Rng::with_name(12345, "up_front");
            let rooms = generate_all_rooms(&mut rng, false, act1);
            assert_eq!(rooms.len(), 3);
            for (i, r) in rooms.iter().enumerate() {
                assert_eq!(r.normal_encounters.len(), acts[i].base_rooms, "{act1:?}");
                assert_eq!(r.elite_encounters.len(), 15, "{act1:?}");
                assert!(
                    acts[i].encounters.iter().any(|e| e.id == r.boss_id),
                    "{act1:?} act {i} boss {} is off-pool",
                    r.boss_id
                );
                assert_eq!(r.second_boss_id, None, "no second boss below A10 (act {i})");
            }
        }
    }

    /// Picking Underdocks must change act 1 AND everything the shifted
    /// `up_front` cursor touches after it. A port that swapped only the act-1
    /// ids while leaving acts 2/3 byte-identical would be reading the stream
    /// wrong — Underdocks' event pool is three shorter, so its shuffle is
    /// three reads cheaper. GH #544.
    #[test]
    fn underdocks_changes_act1_and_shifts_the_stream_for_acts_2_and_3() {
        let mut rng_o = Rng::with_name(12345, "up_front");
        let over = generate_all_rooms(&mut rng_o, false, ActVariant::Overgrowth);
        let mut rng_u = Rng::with_name(12345, "up_front");
        let under = generate_all_rooms(&mut rng_u, false, ActVariant::Underdocks);

        assert_ne!(over[0].boss_id, under[0].boss_id);
        assert!(ACT1_UNDERDOCKS
            .iter()
            .any(|e| e.id == under[0].boss_id && e.room == RoomKind::Boss));
        assert_eq!(over[0].events_shuffled.len(), 13 + 18);
        assert_eq!(under[0].events_shuffled.len(), 10 + 18);
        assert_ne!(
            (&over[1], &over[2]),
            (&under[1], &under[2]),
            "acts 2/3 must move with the act-1 cursor shift"
        );
        assert_eq!(
            rng_o.counter() - rng_u.counter(),
            3,
            "Overgrowth's event pool is 3 longer, so its shuffle costs 3 more reads"
        );
    }

    /// The exported boss universe is exactly the union of every act model's
    /// `RoomKind::Boss` rows — the same filter `generate_act_rooms` rolls
    /// from — so a content change can never leave the PPO boss-identity
    /// vocab short of a boss the map can actually hand the player.
    #[test]
    fn all_boss_encounter_ids_is_the_union_of_every_act_boss_pool() {
        let exported = all_boss_encounter_ids();
        let mut expected: Vec<&'static str> = ALL_ACT_MODELS
            .iter()
            .flat_map(|a| a.encounters.iter())
            .filter(|e| e.room == RoomKind::Boss)
            .map(|e| e.id)
            .collect();
        expected.sort_unstable();
        expected.dedup();
        assert_eq!(exported, expected);
        // Three per act model, four models (both Act-1 variants). A change
        // here is a real content change, not a refactor — the embedding table
        // width moves with it, so every champion needs a re-bake. See the
        // append-only pin in `sts2-vocab`.
        assert_eq!(exported.len(), 12, "{exported:?}");
        assert!(exported.windows(2).all(|w| w[0] < w[1]), "sorted + unique");
        // The three Underdocks bosses are the reason this is 12 and not 9.
        for id in [
            "LAGAVULIN_MATRIARCH_BOSS",
            "SOUL_FYSH_BOSS",
            "WATERFALL_GIANT_BOSS",
        ] {
            assert!(exported.contains(&id), "{id} missing from the boss vocab");
        }
    }

    /// A10 DoubleBoss: only the LAST act rolls a second boss, it is a
    /// distinct member of that act's boss pool, and everything generated
    /// BEFORE the extra read is byte-identical to the non-A10 sequence
    /// (the roll appends to the stream, never reorders it). C#
    /// `RunManager.cs:501-504`.
    #[test]
    fn generate_all_rooms_double_boss_rolls_second_boss_for_last_act_only() {
        let mut rng_a = Rng::with_name(12345, "up_front");
        let plain = generate_all_rooms(&mut rng_a, false, ActVariant::Overgrowth);
        let mut rng_b = Rng::with_name(12345, "up_front");
        let double = generate_all_rooms(&mut rng_b, true, ActVariant::Overgrowth);

        for i in 0..2 {
            assert_eq!(plain[i], double[i], "acts 1/2 unchanged by A10");
            assert_eq!(double[i].second_boss_id, None);
        }
        assert_eq!(
            plain[2].boss_id, double[2].boss_id,
            "the second-boss read comes AFTER the act-3 boss pick"
        );
        let second = double[2]
            .second_boss_id
            .expect("A10 rolls a second boss for the last act");
        assert_ne!(second, double[2].boss_id, "second boss differs");
        assert!(
            ACT3_DATA
                .encounters
                .iter()
                .any(|e| e.room == RoomKind::Boss && e.id == second),
            "second boss {second} comes from the act-3 boss pool"
        );
    }

    // ── v0.111.0 (GH #792) ───────────────────────────────────────────

    #[test]
    fn v0107_generate_all_rooms_is_the_versioned_generator() {
        for seed in 0..40 {
            for act1 in [ActVariant::Overgrowth, ActVariant::Underdocks] {
                let mut a = Rng::new(seed);
                let mut b = Rng::new(seed);
                assert_eq!(
                    generate_all_rooms(&mut a, true, act1),
                    generate_all_rooms_for(crate::GameVersion::V0_107_1, &mut b, true, act1)
                );
                assert_eq!(a, b);
            }
        }
    }

    /// v0.111.0 tags SlitheringStranglerNormal [Jaxfruit, Slimes] and
    /// SnappingJaxfruitNormal [Mushroom, Jaxfruit], so the no-repeat rule
    /// keeps them (and the strangler next to a slime fight) apart.
    #[test]
    fn jaxfruit_tag_separates_strangler_and_jaxfruit_v0111() {
        let v = crate::GameVersion::V0_111_0;
        let find = |id: &str| ACT1.iter().find(|e| e.id == id).unwrap();
        let strangler = find("SLITHERING_STRANGLER_NORMAL");
        let jaxfruit = find("SNAPPING_JAXFRUIT_NORMAL");
        let slimes = find("SLIMES_NORMAL");
        assert!(shares_tags(strangler, jaxfruit, v));
        assert!(shares_tags(strangler, slimes, v));
        assert!(!shares_tags(
            strangler,
            jaxfruit,
            crate::GameVersion::V0_107_1
        ));
        assert!(!shares_tags(
            strangler,
            slimes,
            crate::GameVersion::V0_107_1
        ));
        // The rule is soft (`AddWithoutRepeatingTags` falls back to an
        // unconstrained grab when nothing qualifies), so compare adjacency
        // counts rather than forbidding it outright.
        let adjacent = |rooms: &[GeneratedActRooms]| {
            rooms[0]
                .normal_encounters
                .windows(2)
                .filter(|w| {
                    let ids = [w[0], w[1]];
                    ids.contains(&"SLITHERING_STRANGLER_NORMAL")
                        && ids.contains(&"SNAPPING_JAXFRUIT_NORMAL")
                })
                .count()
        };
        let (mut changed, mut adj_old, mut adj_new) = (false, 0, 0);
        for seed in 0..300 {
            let mut a = Rng::new(seed);
            let mut b = Rng::new(seed);
            let old = generate_all_rooms_for(
                crate::GameVersion::V0_107_1,
                &mut a,
                false,
                ActVariant::Overgrowth,
            );
            let new = generate_all_rooms_for(v, &mut b, false, ActVariant::Overgrowth);
            changed |= old[0].normal_encounters != new[0].normal_encounters;
            adj_old += adjacent(&old);
            adj_new += adjacent(&new);
        }
        assert!(adj_new < adj_old, "old {adj_old} new {adj_new}");
        assert!(changed, "the retag must change some act-1 rolls");
    }
}
