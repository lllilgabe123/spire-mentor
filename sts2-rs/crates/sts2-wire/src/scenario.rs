//! Building a fresh combat from a scenario spec — the one path shared by the
//! Python parity oracle (`sts2-py`) and the browser workbench
//! (`sts2-workbench` → wasm).
//!
//! [`build_scenario_state`] used to live in `sts2-py` as `build_scenario_state_impl`,
//! reachable only through PyO3. The browser SimLab sandbox needs the exact same
//! construction (a "New fight" board, and the opener roll for an enemy whose
//! intent the user left unspecified), and a duplicate would silently drift from
//! the oracle the first time either side changed. So it lives here, in the wire
//! crate both hosts already depend on; `sts2-py` now delegates to it, which
//! keeps the C#-parity fixtures and the browser byte-identical by construction.

use crate::state_dict;
use smallvec::SmallVec;
use sts2_content::cards::dispatch as card_dispatch;
use sts2_content::monster_dispatch::MonsterSpawnArgs;
use sts2_core::cards::Card;
use sts2_core::combat_loop::{start_combat, start_player_turn};
use sts2_core::{
    CardUid, CombatPhase, CombatState as CoreCombatState, MonsterKind, PlayerCombatState, Side,
    TurnEvents,
};

fn starter_args_for_move_state(kind: MonsterKind, move_state: &str) -> Option<MonsterSpawnArgs> {
    match kind {
        MonsterKind::ScrollOfBiting => {
            let starter_idx = match move_state {
                "CHOMP" => 0,
                "CHEW" => 1,
                "MORE_TEETH" => 2,
                _ => return None,
            };
            Some(MonsterSpawnArgs::ScrollOfBiting { starter_idx })
        }
        // Bead `sts2-rs-z4jsg`: `DecimillipedeElite.GenerateMonsters`
        // (DecimillipedeElite.cs:37-50) rolls `StarterMoveIdx =
        // Rng.NextInt(3)` on the per-encounter RNG for the Front segment
        // and rotates +1/+2 onto Middle/Back; the segment machine's
        // initial-state switch (DecimillipedeSegment.cs:171-176) is
        // 0→WRITHE_MOVE, 1→BULK_MOVE, _→CONSTRICT_MOVE — the inverse
        // mapped here (keep in lock-step with
        // `create_decimillipede_segment_ai`). The synthetic builder
        // can't replicate the encounter RNG seed, so it adopts the
        // rotation C# actually rolled, exactly like ScrollsOfBiting.
        MonsterKind::DecimillipedeSegment
        | MonsterKind::DecimillipedeSegmentFront
        | MonsterKind::DecimillipedeSegmentMiddle
        | MonsterKind::DecimillipedeSegmentBack => {
            let starter_idx = match move_state {
                "WRITHE_MOVE" => 0,
                "BULK_MOVE" => 1,
                "CONSTRICT_MOVE" => 2,
                _ => return None,
            };
            Some(MonsterSpawnArgs::DecimillipedeSegment { starter_idx })
        }
        // GH #544: the two Act-1 Underdocks encounters whose `GenerateMonsters`
        // rolls a starter-move ROTATION on the per-encounter RNG, the same
        // shape as ScrollsOfBiting and Decimillipede above.
        //
        // `CorpseSlug.EnsureCorpseSlugsStartWithDifferentMoves` draws one
        // `NextInt(3)` and walks consecutive indices across the slugs;
        // `CorpseSlug.GenerateMoveStateMachine` maps `StarterMoveIdx % 3` to
        // WHIP_SLAP / GLOMP / GOOP.
        //
        // NOTE the ids are C#'s `MoveState` names as the oracle reports them
        // (`_MOVE`-suffixed), NOT the Rust state ids — this function keys off
        // `get_state`'s `monsters[].move_state`. The two differ for both
        // monsters below.
        MonsterKind::CorpseSlug => {
            let starter_idx = match move_state {
                "WHIP_SLAP_MOVE" => 0,
                "GLOMP_MOVE" => 1,
                "GOOP_MOVE" => 2,
                _ => return None,
            };
            Some(MonsterSpawnArgs::CorpseSlug { starter_idx })
        }
        // `TwoTailedRatsNormal` rolls one `NextInt(3)` and hands the three rats
        // `n`, `n+1`, `n+2`; `TwoTailedRat.GenerateMoveStateMachine` maps
        // `StarterMoveIndex % 3` to SCRATCH / DISEASE_BITE / SCREECH (and -1,
        // for a mid-combat summon, to the RAND branch — which is what the
        // `_ => return None` fallthrough leaves in place).
        //
        // This is not cosmetic even though the three rats are identical: the
        // RAND branch is `CannotRepeat`, so which move a rat just performed
        // decides what its next roll may return, and a rotation offset that
        // disagrees with C#'s desyncs the `monster_ai` stream.
        MonsterKind::TwoTailedRat => {
            let slot = match move_state {
                "SCRATCH_MOVE" => sts2_content::monsters::act1_underdocks::TwoTailedRatSlot::First,
                "DISEASE_BITE_MOVE" => {
                    sts2_content::monsters::act1_underdocks::TwoTailedRatSlot::Second
                }
                "SCREECH_MOVE" => sts2_content::monsters::act1_underdocks::TwoTailedRatSlot::Third,
                _ => return None,
            };
            Some(MonsterSpawnArgs::TwoTailedRat { slot })
        }
        _ => None,
    }
}

/// [`build_scenario_state_for_version`] at the default game build
/// ([`sts2_core::GameVersion::V0_107_1`]) — the entry point every
/// pre-multi-version caller uses, byte-identical to before GH #787.
#[allow(clippy::too_many_arguments)]
pub fn build_scenario_state(
    seed: u64,
    character: &str,
    deck_ids: &[String],
    hp: i32,
    max_hp: i32,
    relic_ids: &[String],
    potion_ids: &[String],
    encounter: &str,
    ascension: u8,
    monster_override: Option<&[String]>,
    monster_move_states: Option<&[Option<String>]>,
) -> Result<CoreCombatState, String> {
    build_scenario_state_for_version(
        sts2_core::GameVersion::default(),
        seed,
        character,
        deck_ids,
        hp,
        max_hp,
        relic_ids,
        potion_ids,
        encounter,
        ascension,
        monster_override,
        monster_move_states,
    )
}

/// The C# headless harness's belt size below A4 (`initialMaxPotionSlotCount`;
/// see the `potion_slots` field below). From A4 Tight Belt the default belt
/// is one slot smaller: `scenario_potion_slots` (GH #893).
pub const SCENARIO_POTION_SLOTS: u8 = 3;

/// [`build_scenario_state`] with an explicit belt size.
///
/// A fight rebuilt from a human run needs the human's belt: A4+ (TightBelt)
/// has one slot fewer, and slot relics add some. The belt size matters
/// before combat starts, because start-of-combat potion grants (Petrified
/// Toad's rock) only land in a free slot (GH #830).
#[allow(clippy::too_many_arguments)]
pub fn build_scenario_state_with_slots(
    seed: u64,
    character: &str,
    deck_ids: &[String],
    hp: i32,
    max_hp: i32,
    relic_ids: &[String],
    potion_ids: &[String],
    encounter: &str,
    ascension: u8,
    monster_override: Option<&[String]>,
    monster_move_states: Option<&[Option<String>]>,
    potion_slots: u8,
) -> Result<CoreCombatState, String> {
    build_scenario_state_for_version_with_slots(
        sts2_core::GameVersion::default(),
        seed,
        character,
        deck_ids,
        hp,
        max_hp,
        relic_ids,
        potion_ids,
        encounter,
        ascension,
        monster_override,
        monster_move_states,
        Some(potion_slots),
    )
}

/// Build a fresh scenario combat for game build `game_version` (GH #787).
/// The version lands on [`CoreCombatState::game_version`] exactly like
/// `ascension` lands on `ascension`.
#[allow(clippy::too_many_arguments)]
pub fn build_scenario_state_for_version(
    game_version: sts2_core::GameVersion,
    seed: u64,
    character: &str,
    deck_ids: &[String],
    hp: i32,
    max_hp: i32,
    relic_ids: &[String],
    potion_ids: &[String],
    encounter: &str,
    ascension: u8,
    monster_override: Option<&[String]>,
    monster_move_states: Option<&[Option<String>]>,
) -> Result<CoreCombatState, String> {
    build_scenario_state_for_version_with_slots(
        game_version,
        seed,
        character,
        deck_ids,
        hp,
        max_hp,
        relic_ids,
        potion_ids,
        encounter,
        ascension,
        monster_override,
        monster_move_states,
        None,
    )
}

/// [`build_scenario_state_for_version`] with an explicit belt size — the one
/// body every scenario entry point above delegates to: the game build
/// (GH #787) and the belt size (GH #830) are independent inputs.
/// `potion_slots = None` is the harness's default belt at `ascension`
/// (`scenario_potion_slots`: 3, or 2 from A4 Tight Belt, GH #893); `Some`
/// is a rebuilt human fight's own belt.
#[allow(clippy::too_many_arguments)]
pub fn build_scenario_state_for_version_with_slots(
    game_version: sts2_core::GameVersion,
    seed: u64,
    character: &str,
    deck_ids: &[String],
    hp: i32,
    max_hp: i32,
    relic_ids: &[String],
    potion_ids: &[String],
    encounter: &str,
    ascension: u8,
    monster_override: Option<&[String]>,
    monster_move_states: Option<&[Option<String>]>,
    potion_slots: Option<u8>,
) -> Result<CoreCombatState, String> {
    build_scenario_state_with_run_rng(
        game_version,
        seed,
        character,
        deck_ids,
        hp,
        max_hp,
        relic_ids,
        potion_ids,
        encounter,
        ascension,
        monster_override,
        monster_move_states,
        potion_slots,
        None,
    )
}

/// The real run's RNG at the moment a recorded combat starts: the run's
/// master seed (hashed, as `$.rng.run_seed` on the wire) and every stream's
/// read count (a SpireCodex replay's `combat_start.rng_state`).
#[derive(Debug)]
pub struct RunRngInit<'a> {
    pub run_seed: u64,
    pub counters: &'a [(String, u32)],
    /// C#'s `RunState.TotalFloor` when the room's monsters were generated.
    /// With it the roster is rolled off the per-encounter `Rng(Seed +
    /// TotalFloor + hash(id))`, as the run path does
    /// (`run_callbacks::build_combat_for_room`), so an encounter that draws
    /// its monsters' spawn args there (Decimillipede's opener rotation,
    /// ScrollsOfBiting, CorpseSlug, TwoTailedRat) gets the game's draw.
    /// `None` keeps the old roster: the encounter table's static one, or a
    /// `monster_override`'s kinds without spawn args (GH #918).
    pub encounter_floor: Option<i64>,
}

/// [`build_scenario_state_for_version_with_slots`] that can replace the
/// scenario's synthetic run RNG with a recorded run's, so a human fight is
/// rebuilt with the game's own opening shuffle and monster rolls.
///
/// The recorded counters are read before the room's monsters are created,
/// so the streams are replaced before the roster's HP rolls.
#[allow(clippy::too_many_arguments)]
pub fn build_scenario_state_with_run_rng(
    game_version: sts2_core::GameVersion,
    seed: u64,
    character: &str,
    deck_ids: &[String],
    hp: i32,
    max_hp: i32,
    relic_ids: &[String],
    potion_ids: &[String],
    encounter: &str,
    ascension: u8,
    monster_override: Option<&[String]>,
    monster_move_states: Option<&[Option<String>]>,
    potion_slots: Option<u8>,
    run_rng_init: Option<RunRngInit<'_>>,
) -> Result<CoreCombatState, String> {
    if let Some(potion_slots) = potion_slots {
        if usize::from(potion_slots) > sts2_core::MAX_POTIONS {
            return Err(format!(
                "build_scenario_state: {potion_slots} potion slots exceed MAX_POTIONS ({})",
                sts2_core::MAX_POTIONS
            ));
        }
    }
    // Map the scenario's capitalized class string to a `CharacterId`. The
    // deck loader (`resolve_card_name` below) already searches every
    // character's spec slice, so any ported character can build a scenario;
    // the only character-specific field is `player.character`, set below so
    // class-gated generators (OrobicAcid, deck-search potions) behave. Was
    // Ironclad-only in v1 (bead `sts2-rs-q6bix`); lifted so the Regent
    // coverage pack can A/B its 88 cards against the C# oracle.
    let character_id = match character {
        "Ironclad" => sts2_core::CharacterId::Ironclad,
        "Silent" => sts2_core::CharacterId::Silent,
        "Defect" => sts2_core::CharacterId::Defect,
        "Regent" => sts2_core::CharacterId::Regent,
        "Necrobinder" => sts2_core::CharacterId::Necrobinder,
        other => {
            return Err(format!(
                "build_scenario_state: unknown character {other:?} (expected one of \
                 Ironclad/Silent/Defect/Regent/Necrobinder)"
            ));
        }
    };
    if hp <= 0 || max_hp <= 0 || hp > max_hp {
        return Err(format!(
            "build_scenario_state: invalid hp={hp}/max_hp={max_hp}"
        ));
    }

    // Bead `sts2-rs-7hhd` / `sts2-rs-ov4`: populate the PlayCard dispatch
    // registry before the first `Action::PlayCard` can fire. Idempotent.
    card_dispatch::install_default_registry();
    // Parity coverage (2026-05-29): also install the relic + potion
    // dispatch registries so that relics pushed into `state.relics` below
    // actually fire their `before_combat_start` / reactive listeners
    // during `start_combat`, and held potions resolve their on-drink
    // bodies. Without the relic install, raw-pushed relics are inert
    // (e.g. BronzeScales Thorns never applies) and the parity harness
    // can't compare relic behavior. All three installs are idempotent.
    sts2_content::relics::dispatch::install_default_registry();
    sts2_content::potions::dispatch::install_default_registry();

    // Resolve encounter across ALL act registries (Act 1 → 2 → 3) so the
    // all-acts coverage sweep can build Act-2/Act-3 rosters, not just Act-1
    // ids like `SHRINKER_BEETLE_WEAK` / `VANTOM_BOSS`. Bead `sts2-rs-zdjjx`.
    let enc = sts2_content::encounters::find(encounter)
        .ok_or_else(|| format!("unknown encounter id {encounter:?}"))?;

    // Build the runtime card arena from the deck-id list. Each id is
    // looked up via `resolve_card_name`, which already supports the
    // canonical UPPER_SNAKE form used by T6's `IC_CARDS` constant.
    let mut draw: SmallVec<[CardUid; 32]> = SmallVec::new();
    let mut cards_arena: Vec<Card> = Vec::with_capacity(deck_ids.len());
    for (idx, full_name) in deck_ids.iter().enumerate() {
        // GH #777: an optional "@<ENCHANTMENT>[:<amount>]" suffix
        // ("FIGHT_ME+1@VIGOROUS:8") enchants the card the way the run's
        // enchant path does (`apply_static_enchantment` on the deck card,
        // which the combat then copies), so a human deck with enchanted cards
        // can be rebuilt. The amount defaults to 1; any spelling
        // `EnchantmentId::from_wire_name` accepts works.
        let (name, enchant) = match full_name.split_once('@') {
            Some((card, ench)) => {
                let (ench_id, amount) = match ench.split_once(':') {
                    Some((id, amt)) => (
                        id,
                        amt.parse::<i32>().map_err(|_| {
                            format!("bad enchantment amount in card id {full_name:?}")
                        })?,
                    ),
                    None => (ench, 1),
                };
                let id =
                    sts2_core::cards::EnchantmentId::from_wire_name(ench_id).ok_or_else(|| {
                        format!("unknown enchantment {ench_id:?} in card id {full_name:?}")
                    })?;
                (card, Some((id, amount)))
            }
            None => (full_name.as_str(), None),
        };
        // Sweep v2: an optional "+N" suffix ("BASH+1") requests an upgraded
        // card. Parsed here (resolve_card_name only knows base ids) and
        // applied below via the canonical `Card::upgrade()` delta path.
        // STS2 cards are single-level (C# MaxUpgradeLevel default 1), so
        // only "+1" is accepted — "+0"/"+2"/etc. are rejected rather than
        // silently clamped. The `is_upgradable()` gate below additionally
        // covers the curse/status (MaxUpgradeLevel 0) and already-upgraded
        // cases, degrading those "+1"s to base.
        let (base_name, plus_levels) = match name.split_once('+') {
            Some((base, lvl)) => {
                let n: u32 = lvl.parse().map_err(|_| {
                    format!("bad upgrade suffix in card id {name:?}: expected +<integer>")
                })?;
                if n != 1 {
                    return Err(format!(
                        "bad upgrade suffix in card id {name:?}: STS2 cards are single-level, expected +1"
                    ));
                }
                (base, n)
            }
            None => (name, 0),
        };
        let spec = sts2_content::names::resolve_card_name(base_name, None)
            .map_err(|e| format!("unknown card id {base_name:?}: {e:?}"))?;
        let raw = u32::try_from(idx + 1).map_err(|_| "deck size overflows u32".to_string())?;
        let uid = CardUid::new(raw).ok_or_else(|| "uid non-zero".to_string())?;
        draw.push(uid);
        let mut card = Card {
            uid,
            card_id: spec.id,
            cost: spec.cost,
            card_type: spec.card_type,
            target_type: spec.target_type,
            upgraded: false,
            base_damage: spec.base_damage,
            base_block: spec.base_block,
            // Mirror `card_from_spec` (status.rs): populate the canonical
            // CardTags. Without this, tag-gated handlers diverge from C#
            // (e.g. FastenPower.ModifyBlockAdditive only folds onto block
            // from `CardTag::Defend` sources). q6bix coverage-sweep fix.
            tags: sts2_content::cards::status::tags_for_card_id(spec.id),
            ..Card::pod_defaults()
        };
        // v0.111.0: the scenario's game build decides the card's values
        // (GH #793); v0.107.1 keeps the spec values above unchanged.
        card.normalize_for_version(game_version);
        if plus_levels >= 1 && card.is_upgradable() {
            card.upgrade_for(game_version);
        }
        if let Some((id, amount)) = enchant {
            let _ = sts2_content::enchantments::apply_static_enchantment(&mut card, id, amount);
        }
        cards_arena.push(card);
    }

    // Resolve the held-potion list (canonical UPPER_SNAKE ids, e.g.
    // `FIRE_POTION`) into `PotionKind`s for `player.potions`. Potions are
    // placed directly into the player's belt — the same target the C#
    // headless `set_player` "potions" arg fills (`RunSimulator.cs:433`) —
    // so both engines start combat with byte-identical potion slots. The
    // on-drink RNG/effect dispatch lives in `sts2-content::potions` and
    // fires when a `UsePotion`/`DrinkPotion` action is applied; nothing
    // here perturbs combat RNG at setup.
    let mut potions_vec: SmallVec<[sts2_core::PotionKind; sts2_core::MAX_POTIONS]> =
        SmallVec::new();
    for name in potion_ids {
        let kind = state_dict::potion_kind_from_python_name(name)
            .ok_or_else(|| format!("unknown potion id {name:?}"))?;
        potions_vec.push(kind);
    }

    let potion_slots =
        potion_slots.unwrap_or_else(|| scenario_potion_slots(ascension, potions_vec.len()));
    let player = PlayerCombatState {
        hand: SmallVec::new(),
        draw,
        discard: SmallVec::new(),
        exhaust: SmallVec::new(),
        play: SmallVec::new(),
        energy: 0,
        max_energy: 3,
        stars: 0,
        current_hp: hp,
        max_hp,
        block: 0,
        powers: SmallVec::new(),
        // Character-card generators (OrobicAcid, deck-search potions)
        // early-return on a `None` character; set it to the scenario's
        // character (parsed above) so they generate vs C#. q6bix fix;
        // generalized beyond Ironclad for the Regent coverage pack.
        character: Some(character_id),
        // Compact identity slot mapping — a parity fixture's belt starts
        // hole-free (0..len); holes only develop from in-combat drinks.
        potion_slot_indices: (0..potions_vec.len() as u8).collect(),
        potions: potions_vec,
        // C# `Player.initialMaxPotionSlotCount == 3`, less one under A4
        // Tight Belt (`AscensionManager.ApplyEffectsTo` →
        // `SubtractFromMaxPotionCount(1)` when the run starts), and the
        // headless harness `set_player` never resizes `_potionSlots`
        // (RunSimulator.cs:471-498). Seeding the cap here keeps slot-filling
        // potions (EntropicBrew / Alchemize) from over-filling to
        // MAX_POTIONS=5 and surfacing `use_potion` slots C# never emits
        // (bead sts2-rs-eysmt). A rebuilt human fight passes its own belt
        // through `build_scenario_state_with_slots` /
        // `build_scenario_state_for_version_with_slots`.
        potion_slots,
        orbs: sts2_core::OrbSlots::default(),
        creature_uid: None,
        pet_uid: None,
        shivs_played_this_turn: 0,
        cards_played_this_turn: [0; 5],
        powered_hits_this_turn: 0,
        powered_hits_by_target_this_turn: smallvec::SmallVec::new(),
        powered_hits_at_card_play_start: 0,
        attack_command_suppressed_this_play: false,
        vigor_latch_card: None,
        generated_cards_this_combat: 0,
        skills_played_this_turn: 0,
        card_plays_started_this_turn: 0,
        cards_first_in_series_this_turn: 0,
        request_end_turn_after_play: false,
        retain_this_turn: smallvec::SmallVec::new(),
        sly_this_turn: smallvec::SmallVec::new(),
        sly_this_combat: smallvec::SmallVec::new(),
        pending_retain_count: 0,
        stars_gained_this_turn: 0,
        pinned_top: smallvec::SmallVec::new(),
        pinned_bottom: SmallVec::new(),
        deferred_draw_order: false,
    };

    let rng = sts2_core::dotnet_rng::Rng::new((seed) as i32);
    // Build the per-stream RNG bundle up-front. Monster HP rolls below
    // consume from `run_rng.niche`, mirroring C#
    // `CombatState.CreateCreature` (`SetUniqueMonsterHpValue(...,
    // RunState.Rng.Niche)`) instead of the legacy monolithic combat
    // RNG, so a shuffle won't perturb the next HP roll and vice versa.
    // Bead `sts2-rs-943w7`. The bundle is moved into `state.run_rng`
    // after the HP-roll loop completes.
    //
    // GH #788: seeded in the scenario's game build (v0.107.1 keeps the low
    // 32 bits of `seed`, exactly the old `RunRngSet::new(seed as i32)`).
    let mut run_rng = sts2_core::run_rng_set::RunRngSet::new_for(game_version, seed);
    let encounter_floor = run_rng_init.as_ref().and_then(|init| init.encounter_floor);
    if let Some(init) = run_rng_init {
        run_rng = sts2_core::run_rng_set::RunRngSet::new_for(game_version, init.run_seed);
        for (name, counter) in init.counters {
            let stream = run_rng
                .stream_mut(name.as_str())
                .ok_or_else(|| format!("build_scenario_state: unknown rng stream {name:?}"))?;
            stream.fast_forward_counter(*counter);
        }
    }

    // sts2-rs-3010t: when the harness passes an explicit monster_list
    // (resolved on the C# side post-EnterRoom), use it verbatim so
    // randomized encounters (FLYCONID_NORMAL, SLITHERING_STRANGLER_NORMAL,
    // SLIMES_*) reach roster parity without aligning the two RNGs.
    // Each entry must be a canonical Pascal-case monster name
    // (`MonsterKind::python_name`), matching the ids the C# headless
    // `get_state` surface emits under `monsters[].monster_id`.
    //
    // Per-monster spawn-args from the encounter table (Inklet slot,
    // Nibbit position, KinFollower slot) are dropped under the
    // override path — those scaffolding bits would need a richer
    // protocol from C# to round-trip, which is out of 3010t's scope.
    // Per-monster (kind, spawn-args) pairs. The encounter-table path
    // carries explicit MonsterSpawnArgs (Nibbit group-position fix-up,
    // Inklet slot, KinFollower slot, etc.); the harness-override path
    // (sts2-rs-3010t) defaults to `MonsterSpawnArgs::None` since C#
    // doesn't round-trip per-monster spawn args today.
    //
    // GH #918: a recorded fight that knows its floor rolls the encounter's
    // roster exactly as the run path does, and an override naming the same
    // kinds keeps that roll's spawn args. Without this a SpireCodex replay's
    // Decimillipede always opened on the `num2 = 0` rotation, so the human's
    // targeted hits, Weak and Thorns landed on segments carrying other moves
    // than in the game.
    let rolled: Option<Vec<(MonsterKind, MonsterSpawnArgs)>> = encounter_floor.map(|floor| {
        let mut enc_rng = run_rng.named_at(floor, enc.id);
        enc.roster(&mut enc_rng).into_iter().collect()
    });
    let roster: Vec<(MonsterKind, MonsterSpawnArgs)> = match monster_override {
        Some(names) => {
            let kinds = names
                .iter()
                .map(|n| {
                    state_dict::monster_kind_from_python_name(n)
                        .ok_or_else(|| format!("unknown monster id {n:?}"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            match rolled {
                Some(r) if r.iter().map(|(k, _)| *k).eq(kinds.iter().copied()) => r,
                _ => kinds
                    .into_iter()
                    .map(|k| (k, MonsterSpawnArgs::None))
                    .collect(),
            }
        }
        None => rolled.unwrap_or_else(|| enc.monsters.to_vec()),
    };

    // sts2-rs-u3wjj: when the coverage sweep forwards C#'s observed
    // per-monster initial move-state ids, force-match any monster whose
    // starting move is picked by an RNG-branched `StarterMoveIdx`
    // (ScrollsOfBiting{Weak,Normal} roll `StarterMoveIdx = Rng.NextInt(3)`
    // then rotate +1/+2 across the roster). The synthetic scenario builder
    // doesn't replicate C#'s per-encounter RNG seeding (a fresh
    // `Rng(runState.Seed + TotalFloor + hash(id))`), so it can't roll the
    // same rotation itself — instead it adopts the rotation C# actually
    // rolled. `starter_args_for_move_state` returns `None` for monsters
    // with no starter knob, leaving the encounter-table default intact, so
    // this is a no-op for every non-rotating encounter.
    let roster: Vec<(MonsterKind, MonsterSpawnArgs)> = match monster_move_states {
        Some(states) => roster
            .into_iter()
            .enumerate()
            .map(|(i, (kind, args))| {
                let forced = states
                    .get(i)
                    .and_then(Option::as_deref)
                    .and_then(|sid| starter_args_for_move_state(kind, sid))
                    .unwrap_or(args);
                (kind, forced)
            })
            .collect(),
        None => roster,
    };

    // C# assigns enemy HP in TWO passes: `CombatState.CreateCreature` rolls
    // every enemy's `MaxHp` off `RunState.Rng.Niche`
    // (`Creature.SetUniqueMonsterHpValue`), excluding the RAW `MaxHp` of all
    // same-side creatures already created (any kind), then a later
    // `AfterAddedToRoom` pass applies the Decimillipede even-unique fix-up /
    // Door sentinel off the mixed (earlier-finalized, later-raw) teammate
    // set. `roll_unique_max_hps` performs both passes faithfully; for every
    // non-Decimillipede roster it is byte-identical to the old inline
    // `roll → finalize` loop (same niche reads, identity finalize). Beads
    // `sts2-rs-q6bix` / `8mpo1`.
    let roster_kinds: Vec<MonsterKind> = roster.iter().map(|(k, _)| *k).collect();
    let max_hps = sts2_core::monsters::roll_unique_max_hps_for(
        &roster_kinds,
        &mut run_rng.niche,
        ascension,
        game_version,
    );
    // Spawned below by the run path's `run_bridge::spawn_roster` (position,
    // spawn args, AI, on-spawn powers), once the state exists.
    let roster: Vec<(MonsterKind, MonsterSpawnArgs, i32)> = roster
        .into_iter()
        .zip(max_hps)
        .map(|((kind, args), hp)| (kind, args, hp))
        .collect();

    // Build relics. We push directly into `state.relics` (matching the
    // pattern in `state_dict::state_from_dict_impl`); on-obtain hooks
    // aren't dispatched here, which is fine for the v1 SAFE_RELICS list
    // (`BURNING_BLOOD`'s only effect is a rest-heal hook, no combat
    // perturbation). Expand to `RunManager::force_add_relic` semantics
    // if SAFE_RELICS ever grows to include on-obtain combat effects.
    let mut relics_vec: Vec<sts2_core::Relic> = Vec::with_capacity(relic_ids.len());
    for (idx, name) in relic_ids.iter().enumerate() {
        let kind = state_dict::relic_kind_from_python_name(name)
            .ok_or_else(|| format!("unknown relic id {name:?}"))?;
        let raw = u32::try_from(idx + 1).map_err(|_| "relic index overflows u32".to_string())?;
        let id = std::num::NonZeroU32::new(raw).ok_or_else(|| "relic id non-zero".to_string())?;
        // Seed the combat counter from the relic's canonical fresh-obtain
        // value. Countdown relics (EmberTea/BoneTea/TeaOfDiscourtesy/
        // SwordOfStone) gate their listeners on `counter > 0`; hardcoding
        // `0` here silently disabled them on the only combat the scenario
        // builds — EMBER_TEA dropped its Strength(2), BONE_TEA its hand
        // upgrade (bead `sts2-rs-8s5wf`). Zero-based accumulator relics are
        // unaffected (`initial_counter` returns 0 for them).
        relics_vec.push(sts2_core::Relic {
            id,
            kind,
            counter: kind.initial_counter(),
            is_wax: false,
            is_melted: false,
        });
    }

    let is_elite = matches!(enc.tier, sts2_content::encounters::EncounterTier::Elite);
    let is_boss = matches!(enc.tier, sts2_content::encounters::EncounterTier::Boss);

    // Apply Pantograph's boss-room heal to the player's starting HP.
    // C# fires `Hook.AfterRoomEntered(runState, combatRoom)` on entering a
    // boss room; `Pantograph.AfterRoomEntered` (`Pantograph.cs:22-26`, run-side
    // listener — mirrored in Rust at
    // `sts2-content/src/relics/uncommon.rs::listener_pantograph`) heals 25 HP
    // when `room_type == Boss && current_hp > 0`. That listener operates on a
    // `RunState`, NOT a `CombatState`, so it cannot be reached via the
    // combat-side `relic_effects` bus (an earlier port wrongly dispatched the
    // combat channel here — that both failed to heal AND fired unrelated
    // combat-side boss listeners, regressing non-Pantograph boss fixtures;
    // bead `sts2-rs-nkvht`). Pantograph is the only boss-room-entry run relic
    // with a *combat-observable* effect (the others touch gold/cards/map state
    // the scenario builder does not surface), so we apply it directly here,
    // clamped to `max_hp` exactly like `RunState::heal`.
    let player_hp_final = if is_boss
        && hp > 0
        && relics_vec
            .iter()
            .any(|r| r.kind == sts2_core::RelicKind::Pantograph)
    {
        (hp + sts2_content::relics::uncommon::PANTOGRAPH_HEAL).min(max_hp)
    } else {
        hp
    };

    let mut player = player;
    player.current_hp = player_hp_final;

    let mut state = CoreCombatState {
        phase: CombatPhase::Setup,
        round: 0,
        side: Side::Player,
        rng,
        run_rng,
        creatures: Vec::new(),
        cards: cards_arena.into(),
        powers: Vec::new().into(),
        relics: relics_vec.into(),
        potions: Vec::new(),
        player,
        allies: SmallVec::new(),
        enemy_uids: SmallVec::new(),
        pending_choice: None,
        resolving_card: None,
        pending_play: None,
        pending_turn_setup_resume: false,
        pending_draw_resume: None,
        pending_snecko_cost_randomize: false,
        pending_cluster_resume: None,
        pending_hand_draw_resume: None,
        pending_relic_hand_draw_resume: None,
        pending_wlp_retain_resume: false,
        pending_end_turn_resume: None,
        pending_autoplay_resume: None,
        autoplay_inline_resolve: false,
        vakuu_selector: false,
        pending_curse_choice: None,
        pending_potion_hook_resume: None,
        pending_enemy_exec_resume: None,
        combat_log: TurnEvents::default(),
        swipe_stolen: Vec::new(),
        escaped_creatures: 0,
        coop: sts2_core::coop::CoopState::default(),
        heist_recovered_gold: 0,
        event_combat_timed_out: false,
        // The player's run gold, carried into combat — see the same field in
        // `sts2-cli::sim::scenario::starter`. The C# parity oracle builds its
        // synthetic combats out of a real run, so its player walks in on the
        // character's starting gold, and gold is a genuine combat input:
        // GremlinMerc's `ThieveryPower.Steal()` caps at 20 a steal and banks
        // the total into the HeistPower the FatGremlin inherits when the merc
        // dies, and SealOfGold spends 5. At `gold: 0` both paths no-opped, so
        // the harness reported the engine missing a power it computes
        // correctly (GH #544, found by the Underdocks encounter sweep).
        gold: sts2_core::characters::character_starter(character_id).gold,
        is_over: false,
        player_won: false,
        ascension,
        game_version,
        master_deck: Vec::new(),
        last_added_card: None,
        is_elite,
        is_boss,
        act_index: 1,
        active_attack: None,
        energy_wasted: 0,
        cards_drawn: 0,
        pending_run_gold_loss: 0,
        last_drawn_card: None,
    };

    // Monsters 1..n first, then the player's mirror creature, then every
    // monster's on-spawn powers (so Rocket's Surrounded lands on the player,
    // sts2-rs-q6bix): the order `spawn_roster` shares with the run path and
    // the `sts2-cli` scenario builder.
    sts2_content::run_bridge::spawn_roster(&mut state, &roster);

    // GH #552: stamp the encounter's slot model, so a mid-combat summon lands
    // in the turn position C# gives it. Inert for the ~80% of encounters that
    // declare no `Slots`, and a pure no-op for the initial roster either way
    // (every shipped `GenerateMonsters` is already slot-ascending). Must come
    // after the roster is in `enemy_uids` and before anything can summon.
    state.assign_encounter_slots(enc.member_slots());

    // Parity coverage (2026-05-29): the player's mirror Creature (the
    // canonical d03 path the real engine uses) is initialized by
    // `start_combat` itself (idempotent guard at combat_loop.rs:138,
    // fired before the `before_combat_start` relic listeners at step 4).
    // Relic combat-start grants (BronzeScales Thorns, Vajra Strength,
    // OddlySmoothStone Dexterity, …) call
    // `apply_power_to_creature(player.creature_uid, …)`; what previously
    // blocked them here was the *relic dispatch registry* not being
    // installed (only the card registry was) — see the
    // `relics::dispatch::install_default_registry()` call above. No
    // explicit `init_player_creature` is needed; `start_combat` does it.
    start_combat(&mut state);
    start_player_turn(&mut state);
    Ok(state)
}

/// The potion-belt capacity of a scenario player at `ascension` holding
/// `held` potions: C#'s initial 3, or 2 from A4 (Tight Belt). Never below
/// `held`, so a scenario that lists more potions than that (a belt grown by
/// PotionBelt / PhialHolster, which the builder does not model) keeps them
/// all rather than silently losing one.
fn scenario_potion_slots(ascension: u8, held: usize) -> u8 {
    let base: u8 = if ascension >= sts2_core::runstate::TIGHT_BELT_ASCENSION_LEVEL {
        2
    } else {
        3
    };
    base.max(u8::try_from(held).unwrap_or(u8::MAX))
}

#[cfg(test)]
mod tests {
    use super::starter_args_for_move_state;
    use sts2_core::MonsterKind;

    #[test]
    fn starter_args_for_move_state_maps_scroll_states() {
        // sts2-rs-u3wjj: the inverse of `create_scroll_of_biting_ai`'s
        // `StarterMoveIdx % 3` selection. Keep in lock-step.
        use sts2_content::monster_dispatch::MonsterSpawnArgs as A;
        assert!(matches!(
            starter_args_for_move_state(MonsterKind::ScrollOfBiting, "CHOMP"),
            Some(A::ScrollOfBiting { starter_idx: 0 })
        ));
        assert!(matches!(
            starter_args_for_move_state(MonsterKind::ScrollOfBiting, "CHEW"),
            Some(A::ScrollOfBiting { starter_idx: 1 })
        ));
        assert!(matches!(
            starter_args_for_move_state(MonsterKind::ScrollOfBiting, "MORE_TEETH"),
            Some(A::ScrollOfBiting { starter_idx: 2 })
        ));
        // Unknown state / non-rotating kind → no override.
        assert!(starter_args_for_move_state(MonsterKind::ScrollOfBiting, "WAT").is_none());
        assert!(starter_args_for_move_state(MonsterKind::Nibbit, "CHOMP").is_none());
    }

    fn build(deck: &[&str]) -> Result<sts2_core::CombatState, String> {
        let deck: Vec<String> = deck.iter().map(|s| (*s).to_string()).collect();
        super::build_scenario_state(
            1,
            "Ironclad",
            &deck,
            70,
            80,
            &[],
            &[],
            "SHRINKER_BEETLE_WEAK",
            0,
            None,
            None,
        )
    }

    /// GH #918: a recorded fight that passes its floor rolls Decimillipede's
    /// opener rotation off the per-encounter RNG, the run path's draw, and the
    /// kind-only roster override a replay sends keeps that roll. Without the
    /// floor every rebuild opened on the `num2 = 0` rotation.
    #[test]
    fn encounter_floor_rolls_the_decimillipede_rotation() {
        let version = sts2_core::GameVersion::V0_111_0;
        let deck = vec!["STRIKE_IRONCLAD".to_string(); 5];
        let kinds: Vec<String> = [
            "DecimillipedeSegmentFront",
            "DecimillipedeSegmentMiddle",
            "DecimillipedeSegmentBack",
        ]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
        let run_seed = 0x5eed_u64;
        let openers = |floor: Option<i64>| -> Vec<&'static str> {
            super::build_scenario_state_with_run_rng(
                version,
                1,
                "Ironclad",
                &deck,
                70,
                80,
                &[],
                &[],
                "DECIMILLIPEDE_ELITE",
                10,
                Some(&kinds),
                None,
                None,
                Some(super::RunRngInit {
                    run_seed,
                    counters: &[],
                    encounter_floor: floor,
                }),
            )
            .expect("scenario builds")
            .creatures
            .iter()
            .filter_map(|c| {
                c.ai.as_ref()
                    .map(sts2_core::monster_ai::MonsterAi::current_state_id)
            })
            .collect()
        };
        const MOVES: [&str; 3] = ["WRITHE", "BULK", "CONSTRICT"];
        assert_eq!(openers(None), MOVES);
        let mut rotations = std::collections::BTreeSet::new();
        for floor in 2..20 {
            let set = sts2_core::run_rng_set::RunRngSet::new_for(version, run_seed);
            let num2 = set
                .named_at(floor, "DECIMILLIPEDE_ELITE")
                .next_int_exclusive(0, 3);
            let want: Vec<&str> = (0..3)
                .map(|i| MOVES[usize::try_from((num2 + i) % 3).unwrap()])
                .collect();
            assert_eq!(openers(Some(floor)), want, "floor {floor}");
            rotations.insert(num2);
        }
        assert!(rotations.len() > 1, "the draw must not be pinned");
    }

    /// A rebuilt A10 fight has a 2-slot belt (TightBelt). With both slots
    /// full, Petrified Toad's start-of-combat rock finds no free slot; a
    /// 3-slot belt (the harness default below A4) leaves it one (GH #830).
    #[test]
    fn potion_slots_bound_the_start_of_combat_toad_rock() {
        let deck: Vec<String> = ["STRIKE_IRONCLAD", "DEFEND_IRONCLAD", "BASH"]
            .iter()
            .map(|s| (*s).to_string())
            .collect();
        let relics = vec!["PETRIFIED_TOAD".to_string()];
        let potions = vec!["FIRE_POTION".to_string(), "BLOCK_POTION".to_string()];
        let build = |slots: u8| {
            super::build_scenario_state_with_slots(
                1,
                "Ironclad",
                &deck,
                70,
                80,
                &relics,
                &potions,
                "SHRINKER_BEETLE_WEAK",
                10,
                None,
                None,
                slots,
            )
            .expect("scenario builds")
        };
        let two = build(2);
        assert_eq!(two.player.potion_slots, 2);
        assert_eq!(two.player.potions.len(), 2);
        let three = build(super::SCENARIO_POTION_SLOTS);
        assert_eq!(
            three.player.potions.len(),
            3,
            "the toad's rock fills the free third slot"
        );
        assert!(super::build_scenario_state_with_slots(
            1,
            "Ironclad",
            &deck,
            70,
            80,
            &[],
            &[],
            "SHRINKER_BEETLE_WEAK",
            10,
            None,
            None,
            u8::try_from(sts2_core::MAX_POTIONS + 1).unwrap(),
        )
        .is_err());
    }

    /// GH #893: A4 Tight Belt leaves the scenario player 2 potion slots.
    #[test]
    fn scenario_belt_has_two_slots_from_a4() {
        let build_at = |ascension: u8, potions: &[&str]| {
            let potions: Vec<String> = potions.iter().map(|s| (*s).to_string()).collect();
            super::build_scenario_state(
                1,
                "Ironclad",
                &["STRIKE_IRONCLAD".to_string()],
                70,
                80,
                &[],
                &potions,
                "SHRINKER_BEETLE_WEAK",
                ascension,
                None,
                None,
            )
            .unwrap()
        };
        assert_eq!(build_at(0, &[]).player.potion_slots, 3);
        assert_eq!(build_at(3, &[]).player.potion_slots, 3);
        assert_eq!(build_at(4, &[]).player.potion_slots, 2);
        assert_eq!(build_at(10, &["FIRE_POTION"]).player.potion_slots, 2);
        // A listed belt is never truncated.
        let full = build_at(10, &["FIRE_POTION", "FIRE_POTION", "FIRE_POTION"]);
        assert_eq!(full.player.potion_slots, 3);
        assert_eq!(full.player.potions.len(), 3);
    }

    #[test]
    fn enchantment_suffix_enchants_the_card() {
        // GH #777: "ID[+1]@ENCHANTMENT[:amount]" rebuilds a human deck's
        // enchanted card; any `from_wire_name` spelling, amount default 1.
        use sts2_core::cards::EnchantmentId;
        let state = build(&[
            "STRIKE_IRONCLAD@SHARP:3",
            "DEFEND_IRONCLAD+1@steady",
            "BASH",
        ])
        .unwrap();
        let with = |id, amount| {
            state
                .cards
                .iter()
                .filter(|c| c.enchantment(id) == amount)
                .collect::<Vec<_>>()
        };
        assert_eq!(with(EnchantmentId::Sharp, 3).len(), 1);
        let steady = with(EnchantmentId::Steady, 1);
        assert_eq!(steady.len(), 1);
        assert!(
            steady[0].upgraded,
            "the upgrade suffix still applies before '@'"
        );
        assert_eq!(
            state
                .cards
                .iter()
                .filter(|c| c.enchantment(EnchantmentId::Sharp) == 0
                    && c.enchantment(EnchantmentId::Steady) == 0)
                .count(),
            1,
            "the unsuffixed card stays unenchanted"
        );
    }

    #[test]
    fn enchantment_suffix_rejects_unknown_id_and_bad_amount() {
        let err = |card: &str| build(&[card]).err().unwrap();
        assert!(err("STRIKE_IRONCLAD@NOPE").contains("unknown enchantment"));
        assert!(err("STRIKE_IRONCLAD@SHARP:x").contains("bad enchantment amount"));
    }
}
