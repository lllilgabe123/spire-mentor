//! `coop_fight` — a whole co-op fight (2–4 players) in the engine.
//!
//! Reads one JSON fight spec from stdin (or `--json <json>`), plays `fights`
//! fights from `seed` on, and prints one JSON line per fight plus a summary
//! line. Seat 0 is played by MCTS; the teammates by the search's rollout
//! heuristic (`sts2_mcts::expert::teammate_policy`), which plays their whole
//! turn when seat 0 ends its own. The fight is faithful to C#'s multiplayer
//! rules as far as the engine models them (`docs/plans/2026-10-26-coop-
//! multiseat-engine.md`): each seat's own turn start and turn end, monsters
//! hitting every player, player-count scaling, per-seat death and revival.
//!
//! # Input
//! ```json
//! {
//!   "seats": [
//!     {"character": "ironclad"},
//!     {"character": "silent", "deck": ["STRIKE_SILENT", "NEUTRALIZE"], "relics": []}
//!   ],
//!   "encounter_id": "BYGONE_EFFIGY_ELITE",
//!   "seed": 1, "fights": 10, "sims": 500, "depth": 50
//! }
//! ```
//!
//! Per seat: `character` (ironclad / silent / defect / regent / necrobinder),
//! `deck` (card tokens, default the starter deck), `relics` (relic names,
//! default the starter relic), `hp` (default the character's max HP).
//!
//! # Output
//! One line per fight:
//! `{"seed":1,"won":true,"rounds":6,"hp":[52,31],"max_hp":[80,70],"stalled":false}`, then
//! `{"summary":{"fights":10,"wins":7,"win_rate":0.7,"mean_rounds":6.3,
//!   "mean_hp":[48.1,22.0]}}`.

use std::io::{self, Read};
use std::process;

use serde::{Deserialize, Serialize};

use sts2_cli::sim::scenario::deck::{install_deck, DeckCardSpec};
use sts2_cli::sim::scenario::relics::{relic_id_to_kind, resolve_relics, RelicInput};
use sts2_cli::sim::{build_starter_state_from_monsters, PlayerClass};
use sts2_content::encounters::{self, EncounterTier};
use sts2_content::names::{resolve_card_name, PlayerClassHint};
use sts2_content::starters::starter_deck_for;
use sts2_core::combat_loop::{start_combat, start_player_turn};
use sts2_core::potions::CharacterId;
use sts2_core::{apply, is_terminal, legal_actions, Action, CardUid, CombatState, Relic};
use sts2_mcts::{Mcts, MctsConfig, TerminalEvalMode};

/// Actions per fight before giving up on a stalled one.
const MAX_ACTIONS: u32 = 4000;

#[derive(Deserialize)]
struct SeatSpec {
    character: String,
    #[serde(default)]
    deck: Option<Vec<String>>,
    #[serde(default)]
    relics: Option<Vec<String>>,
    #[serde(default)]
    hp: Option<i32>,
}

#[derive(Deserialize)]
struct FightSpec {
    seats: Vec<SeatSpec>,
    encounter_id: String,
    seed: u64,
    #[serde(default = "one")]
    fights: u32,
    #[serde(default = "default_sims")]
    sims: u32,
    #[serde(default = "default_depth")]
    depth: u32,
}

const fn one() -> u32 {
    1
}
const fn default_sims() -> u32 {
    500
}
const fn default_depth() -> u32 {
    50
}

/// One fight's result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct FightOutcome {
    seed: u64,
    won: bool,
    rounds: u16,
    /// Each seat's HP at the end, seat 0 first.
    hp: Vec<i32>,
    max_hp: Vec<i32>,
    /// The fight hit the action cap without ending: a bug, not a loss.
    stalled: bool,
}

#[derive(Debug, Serialize)]
struct Summary {
    fights: u32,
    wins: u32,
    stalled: u32,
    win_rate: f64,
    mean_rounds: f64,
    mean_hp: Vec<f64>,
}

/// A seat resolved against the content tables.
struct Seat {
    class: PlayerClass,
    character: CharacterId,
    deck: Vec<DeckCardSpec>,
    relics: Vec<sts2_core::relics::RelicKind>,
    hp: i32,
}

fn resolve_character(tag: &str) -> Result<(PlayerClass, CharacterId), String> {
    match tag.trim().to_ascii_lowercase().as_str() {
        "ironclad" => Ok((PlayerClass::Ironclad, CharacterId::Ironclad)),
        "silent" => Ok((PlayerClass::Silent, CharacterId::Silent)),
        "defect" => Ok((PlayerClass::Defect, CharacterId::Defect)),
        "regent" => Ok((PlayerClass::Regent, CharacterId::Regent)),
        "necrobinder" => Ok((PlayerClass::Necrobinder, CharacterId::Necrobinder)),
        other => Err(format!("unknown character: '{other}'")),
    }
}

const fn class_hint(character: CharacterId) -> Option<PlayerClassHint> {
    match character {
        CharacterId::Ironclad => Some(PlayerClassHint::Ironclad),
        CharacterId::Silent => Some(PlayerClassHint::Silent),
        CharacterId::Defect => Some(PlayerClassHint::Defect),
        _ => None,
    }
}

/// `"BASH"` / `"INFLAME+"` / `"INFLAME+1"` → a deck entry.
fn parse_card_token(token: &str, hint: Option<PlayerClassHint>) -> Result<DeckCardSpec, String> {
    let (name, upgrade_level) = match token.strip_suffix('+') {
        Some(rest) => (rest, 1),
        None => match token.rsplit_once('+') {
            Some((name, n)) if !n.is_empty() && n.chars().all(|c| c.is_ascii_digit()) => (
                name,
                n.parse().map_err(|_| format!("bad upgrade: '{token}'"))?,
            ),
            _ => (token, 0),
        },
    };
    let spec =
        resolve_card_name(name.trim(), hint).map_err(|_| format!("unknown card: '{token}'"))?;
    Ok(DeckCardSpec {
        spec,
        upgrade_level,
        enchantment: None,
    })
}

fn resolve_seat(s: &SeatSpec) -> Result<Seat, String> {
    let (class, character) = resolve_character(&s.character)?;
    let deck = match &s.deck {
        Some(tokens) => tokens
            .iter()
            .map(|t| parse_card_token(t, class_hint(character)))
            .collect::<Result<_, _>>()?,
        None => starter_deck_for(character)
            .into_iter()
            .map(|spec| DeckCardSpec {
                spec,
                upgrade_level: 0,
                enchantment: None,
            })
            .collect(),
    };
    let starter = sts2_core::characters::character_starter(character);
    let relic_ids = match &s.relics {
        Some(names) => {
            resolve_relics(&RelicInput::Inline(names.clone())).map_err(|e| e.to_string())?
        }
        None => vec![starter.starter_relic],
    };
    let relics = relic_ids
        .into_iter()
        .map(|id| relic_id_to_kind(id).ok_or_else(|| format!("relic has no engine kind: {id:?}")))
        .collect::<Result<_, _>>()?;
    Ok(Seat {
        class,
        character,
        deck,
        relics,
        hp: s.hp.unwrap_or(starter.hp),
    })
}

fn relic_store(
    kinds: &[sts2_core::relics::RelicKind],
    first_id: &mut u32,
) -> sts2_core::RelicStore {
    let mut store = sts2_core::RelicStore::default();
    for &kind in kinds {
        store.push(Relic {
            id: std::num::NonZeroU32::new(*first_id).expect("non-zero"),
            kind,
            counter: 0,
            is_wax: false,
            is_melted: false,
        });
        *first_id += 1;
    }
    store
}

/// Build the fight: seat 0 through the CLI's own builder, the teammates seated
/// as whole players BEFORE the monsters spawn (their HP and spawn powers scale
/// with the player count, C# `CreateCreature`), then the encounter.
fn build_fight(
    seats: &[Seat],
    enc: &'static encounters::EncounterDef,
    act: u8,
    seed: u64,
) -> CombatState {
    let first = &seats[0];
    let mut state = build_starter_state_from_monsters(
        seed,
        Vec::new(),
        first.class,
        Some(u32::try_from(first.hp).unwrap_or(0)),
        Some(u32::try_from(first.hp).unwrap_or(0)),
        Some(first.deck.clone()),
    );
    state.player.character = Some(first.character);
    let mut relic_id = 1_u32;
    state.relics = relic_store(&first.relics, &mut relic_id);
    state.act_index = act;
    state.is_elite = enc.tier == EncounterTier::Elite;
    state.is_boss = enc.tier == EncounterTier::Boss;

    for seat in &seats[1..] {
        let mut mate = sts2_core::PlayerCombatState {
            current_hp: seat.hp,
            max_hp: seat.hp,
            max_energy: 3,
            character: Some(seat.character),
            ..sts2_core::PlayerCombatState::default()
        };
        // The teammate's cards join the shared arena after ours.
        let mut scratch = state.clone();
        install_deck(&mut scratch, &seat.deck);
        for &card in &scratch.cards {
            let mut card = card;
            let uid =
                CardUid::new(u32::try_from(state.cards.len() + 1).expect("fits")).expect("nz");
            card.uid = uid;
            state.cards.push(card);
            mate.draw.push(uid);
        }
        state.allies.push(mate);
    }
    sts2_core::coop::open_seats(&mut state);
    state.coop.full = true;
    for (i, seat) in seats[1..].iter().enumerate() {
        state.coop.stash[i].relics = relic_store(&seat.relics, &mut relic_id);
    }

    let roster = enc.canonical_roster();
    let kinds: Vec<_> = roster.iter().map(|(k, _)| *k).collect();
    let hps =
        sts2_core::monsters::roll_unique_max_hps(&kinds, &mut state.run_rng.niche, state.ascension);
    let spawn: Vec<_> = roster
        .iter()
        .zip(hps)
        .map(|((kind, args), hp)| (*kind, *args, hp))
        .collect();
    let _ = sts2_content::run_bridge::spawn_roster(&mut state, &spawn);
    state
}

/// The act (1-based) an encounter belongs to.
fn act_of(id: &str) -> u8 {
    if encounters::act3_glory::find(id).is_some() {
        3
    } else if encounters::act2_hive::find(id).is_some() {
        2
    } else {
        1
    }
}

fn run_fight(spec: &FightSpec, seats: &[Seat], seed: u64) -> Result<FightOutcome, String> {
    let enc = encounters::find(&spec.encounter_id)
        .ok_or_else(|| format!("unknown encounter_id: '{}'", spec.encounter_id))?;
    let mut state = build_fight(seats, enc, act_of(enc.id), seed);
    let _ = start_combat(&mut state);
    let _ = start_player_turn(&mut state);

    let mut actions = 0_u32;
    while !is_terminal(&state) && actions < MAX_ACTIONS {
        let legal = legal_actions(&state);
        let action = if legal.len() <= 1 {
            legal.first().cloned().unwrap_or(Action::EndTurn)
        } else {
            let cfg = MctsConfig {
                max_simulations: spec.sims,
                max_rollout_depth: spec.depth,
                seed: seed
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add(u64::from(actions)),
                terminal_eval_mode: TerminalEvalMode::RichBoth,
                ..MctsConfig::default()
            };
            Mcts::new(state.clone(), cfg).search()
        };
        let _ = apply(&mut state, action);
        actions += 1;
    }

    // Seat order: us, then the teammates (we are seated at rest).
    let hp: Vec<i32> = std::iter::once(&state.player)
        .chain(state.allies.iter())
        .map(|p| p.current_hp.max(0))
        .collect();
    let max_hp = std::iter::once(&state.player)
        .chain(state.allies.iter())
        .map(|p| p.max_hp)
        .collect();
    Ok(FightOutcome {
        seed,
        won: state.is_over && state.player_won,
        rounds: state.round,
        hp,
        max_hp,
        stalled: !is_terminal(&state),
    })
}

fn run(spec: &FightSpec) -> Result<(Vec<FightOutcome>, Summary), String> {
    sts2_content::cards::dispatch::ensure_installed();
    sts2_content::relics::dispatch::ensure_installed();
    sts2_content::potions::dispatch::ensure_installed();
    if !(2..=4).contains(&spec.seats.len()) {
        return Err(format!(
            "a co-op fight has 2 to 4 seats, not {}",
            spec.seats.len()
        ));
    }
    let seats: Vec<Seat> = spec
        .seats
        .iter()
        .map(resolve_seat)
        .collect::<Result<_, _>>()?;
    let mut outcomes = Vec::new();
    for i in 0..spec.fights {
        outcomes.push(run_fight(spec, &seats, spec.seed + u64::from(i))?);
    }
    let n = f64::from(spec.fights.max(1));
    let wins = u32::try_from(outcomes.iter().filter(|o| o.won).count()).unwrap_or(0);
    let mean_hp = (0..seats.len())
        .map(|s| outcomes.iter().map(|o| f64::from(o.hp[s])).sum::<f64>() / n)
        .collect();
    let summary = Summary {
        fights: spec.fights,
        wins,
        stalled: u32::try_from(outcomes.iter().filter(|o| o.stalled).count()).unwrap_or(0),
        win_rate: f64::from(wins) / n,
        mean_rounds: outcomes.iter().map(|o| f64::from(o.rounds)).sum::<f64>() / n,
        mean_hp,
    };
    Ok((outcomes, summary))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let json = if let Some(pos) = args.iter().position(|a| a == "--json") {
        args.get(pos + 1).cloned().unwrap_or_else(|| {
            eprintln!("coop_fight: --json needs an argument");
            process::exit(1);
        })
    } else {
        let mut buf = String::new();
        if let Err(e) = io::stdin().read_to_string(&mut buf) {
            eprintln!("coop_fight: stdin read error: {e}");
            process::exit(1);
        }
        buf
    };
    let spec: FightSpec = match serde_json::from_str(json.trim()) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("coop_fight: JSON parse error: {e}");
            process::exit(1);
        }
    };
    match run(&spec) {
        Ok((outcomes, summary)) => {
            for o in &outcomes {
                println!("{}", serde_json::to_string(o).expect("serialize"));
            }
            println!("{}", serde_json::json!({ "summary": summary }));
        }
        Err(e) => {
            eprintln!("coop_fight: {e}");
            process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(seats: &str, fights: u32) -> FightSpec {
        serde_json::from_str(&format!(
            r#"{{"seats": {seats}, "encounter_id": "NIBBITS_WEAK", "seed": 3, "fights": {fights}, "sims": 40, "depth": 20}}"#
        ))
        .expect("spec")
    }

    /// Two starter seats play a weak fight to its end; every seat is
    /// accounted for and the monsters were scaled for two.
    #[test]
    fn a_two_seat_fight_runs_to_the_end() {
        let s = spec(r#"[{"character": "ironclad"}, {"character": "silent"}]"#, 2);
        let (outcomes, summary) = run(&s).expect("runs");
        assert_eq!(outcomes.len(), 2);
        for o in &outcomes {
            assert_eq!(o.hp.len(), 2);
            assert_eq!(o.max_hp, vec![80, 70]);
            assert!(o.rounds >= 1);
        }
        assert_eq!(summary.fights, 2);
        assert_eq!(summary.mean_hp.len(), 2);
    }

    /// A teammate's monsters are scaled at spawn: two seats, double-plus HP.
    #[test]
    fn the_encounter_is_scaled_for_the_seats() {
        sts2_content::cards::dispatch::ensure_installed();
        let seats: Vec<Seat> = [r#"{"character":"ironclad"}"#, r#"{"character":"silent"}"#]
            .iter()
            .map(|j| resolve_seat(&serde_json::from_str(j).expect("seat")).expect("resolves"))
            .collect();
        let enc = encounters::find("NIBBITS_WEAK").expect("encounter");
        let two = build_fight(&seats, enc, 1, 3);
        let solo = build_fight(&seats[..1], enc, 1, 3);
        let hp = |s: &CombatState| {
            s.creatures
                .iter()
                .filter(|c| c.species.is_some())
                .map(|c| c.max_hp)
                .sum::<i32>()
        };
        assert!(hp(&two) >= hp(&solo) * 2, "{} vs {}", hp(&two), hp(&solo));
        assert!(two.coop.full);
        assert_eq!(
            two.coop.stash[0].relics.len(),
            1,
            "the Silent's Ring of the Snake"
        );
    }

    /// A teammate who only defends cannot win the elite: the fight is lost
    /// once both players are dead, and it ends rather than stalling.
    #[test]
    fn a_lost_fight_ends_when_every_player_is_dead() {
        let defends = ["\"DEFEND_SILENT\""; 10].join(",");
        let mut s = spec(
            &format!(
                r#"[{{"character": "ironclad"}}, {{"character": "silent", "deck": [{defends}]}}]"#
            ),
            1,
        );
        s.encounter_id = "BYGONE_EFFIGY_ELITE".into();
        let (outcomes, _) = run(&s).expect("runs");
        let o = &outcomes[0];
        assert!(!o.stalled, "the fight ended");
        assert!(!o.won);
        assert_eq!(o.hp, vec![0, 0]);
    }

    #[test]
    fn a_fight_needs_two_to_four_seats() {
        assert!(run(&spec(r#"[{"character": "ironclad"}]"#, 1)).is_err());
    }
}
