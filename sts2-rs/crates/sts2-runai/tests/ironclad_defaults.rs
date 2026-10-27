//! GH #830: the Ironclad run-agent defaults (the gated knobs
//! `IRONCLAD_SEARCH_KNOBS` and the built-in move prior in elite and boss
//! fights) stay in Ironclad fights.
//!
//! The layer is keyed on the fight's `combat.player.character`, so two
//! things must hold: every run combat carries its character (T8), and every
//! other character, and a fight without one, searches exactly as it did
//! before the layer existed (T2), through the same `build_search` route the
//! pilot uses. Each test runs in its own process (nextest), with no
//! `STS2_RUNAI_*` variable set.

use sts2_core::potions::CharacterId;
use sts2_core::run_manager::{Phase, RunManager};
use sts2_core::runstate::RoomType;
use sts2_mcts::MctsConfig;
use sts2_runai::policy::{CombatBudget, FirstOptionPolicy, HeuristicPolicy, MctsCombatPolicy};
use sts2_runai::rollout::ensure_content_installed;

const CHARACTERS: [CharacterId; 5] = [
    CharacterId::Ironclad,
    CharacterId::Silent,
    CharacterId::Defect,
    CharacterId::Regent,
    CharacterId::Necrobinder,
];

/// A fresh `character` run (seed `seed`, ascension `asc`) at its first
/// combat, reached by first-option picks.
fn first_combat(seed: i32, character: CharacterId, asc: u8) -> RunManager {
    ensure_content_installed();
    let mut rm = sts2_content::run_callbacks::new_run_manager_for_character_with_ascension(
        seed, character, asc,
    )
    .with_neow_start();
    sts2_content::run_callbacks::initialize_run_setup(&mut rm);
    for _ in 0..64 {
        if rm.phase() == Phase::Combat {
            return rm;
        }
        let acts = rm.get_available_actions();
        assert!(!acts.is_empty(), "{character:?}: run stalled");
        let _ = rm.take_action(acts[0]);
    }
    panic!("{character:?}: no combat within 64 run actions");
}

fn clear_env() {
    for var in [
        "STS2_RUNAI_MCTS_OVERRIDES",
        "STS2_RUNAI_PRIOR_NET",
        "STS2_RUNAI_PRIOR_ROOMS",
        "STS2_RUNAI_HONEST",
    ] {
        std::env::remove_var(var);
    }
}

/// T8: the layer's key is set on every real run combat. A combat-entry
/// path that dropped it would silently give Ironclad the base search.
#[test]
fn run_combats_carry_their_character() {
    for asc in [0, 10] {
        for (k, character) in CHARACTERS.into_iter().enumerate() {
            let rm = first_combat(11 + i32::try_from(k).expect("small"), character, asc);
            let combat = rm.combat.as_ref().expect("in combat");
            assert_eq!(
                combat.player.character,
                Some(character),
                "{character:?} at A{asc}"
            );
            assert_eq!(rm.run_state.character_id, character);
        }
    }
}

/// `search_config`'s config as a Debug string (every field, floats exact).
fn config(character: Option<CharacterId>, sims: u32) -> String {
    let budget = CombatBudget::eval_cheap();
    let policy = MctsCombatPolicy::new(FirstOptionPolicy, budget, 1);
    format!("{:?}", policy.search_config(sims, 17, character).0)
}

/// What `build_search` sets up for `combat` in `rm`'s room as `room`: the
/// config (Debug) and whether it runs a PUCT prior.
fn built(rm: &mut RunManager, combat: &sts2_core::CombatState, room: RoomType) -> (String, bool) {
    rm.run_state
        .current_room
        .as_mut()
        .expect("in a room")
        .room_type = room;
    let mut policy = MctsCombatPolicy::new(HeuristicPolicy::new(), CombatBudget::eval_cheap(), 3);
    let cfg: MctsConfig = policy.build_search(rm, combat, 1000, 5).config;
    (format!("{cfg:?}"), cfg.use_puct_prior)
}

/// T2: Silent, Defect, Regent, Necrobinder and a character-less fight
/// search exactly like the character-less base, at every budget and through
/// the pilot's `build_search` in elite and boss rooms (no prior); Ironclad
/// differs in both, so the comparison can fail.
#[test]
fn other_characters_search_exactly_as_before() {
    clear_env();
    for sims in [8, 16, 250, 1000, 4000] {
        let base = config(None, sims);
        for character in &CHARACTERS[1..] {
            assert_eq!(
                config(Some(*character), sims),
                base,
                "{character:?} at {sims} sims"
            );
        }
        assert_ne!(
            config(Some(CharacterId::Ironclad), sims),
            base,
            "Ironclad at {sims} sims"
        );
    }
    for (k, character) in CHARACTERS.into_iter().enumerate() {
        let mut rm = first_combat(3 + i32::try_from(k).expect("small"), character, 10);
        let combat = rm.combat.clone().expect("in combat");
        let mut characterless = combat.clone();
        characterless.player.character = None;
        for room in [RoomType::Elite, RoomType::Boss] {
            let (got, prior) = built(&mut rm, &combat, room);
            let (want, base_prior) = built(&mut rm, &characterless, room);
            assert!(!base_prior, "a character-less fight has no prior");
            if character == CharacterId::Ironclad {
                assert!(prior, "Ironclad {room:?}: the built-in prior");
                assert_ne!(got, want, "Ironclad {room:?}: the layer");
            } else {
                assert!(!prior, "{character:?} {room:?}: no prior");
                assert_eq!(got, want, "{character:?} {room:?}");
            }
        }
    }
}
