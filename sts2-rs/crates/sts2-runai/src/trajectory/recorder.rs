//! [`RecordingPolicy`] — a [`RunPolicy`] decorator that captures a full,
//! reproducible trajectory of a run without touching [`drive_full_run`].
//!
//! It wraps the real policy (PPO/ranker/…). On every `choose` it observes
//! the pre-decision state and appends a [`Decision`] (phase-typed offered
//! payload + the chosen index). Off-combat offers are resolved through the
//! proven [`VisibleRunView`]; combat plies embed the parity bridge board
//! ([`sts2_core::parity::combat_state_to_bridge_state`]). It also maintains
//! deck/relic "ledgers" (first-seen floor per card/relic) and a room-visit
//! list so [`RecordingPolicy::into_run_file`] can project the native
//! `RunHistory` summary. See [`super::schema`] for the output shape.

use std::collections::BTreeMap;

use sts2_core::full_run::{FullRunReport, RunPolicy};
use sts2_core::run_manager::{Phase, RunAction, RunManager};

use super::schema::{
    CombatPly, Decision, DeckCardEntry, MapPointEntry, Meta, MetaEvalTelemetry, Offered,
    PotionEntry, RelicEntry, RootAction, RunFile, RunPlayer, ScoredOption, SearchTelemetry,
    ShopItem, Sts2Sim, SCHEMA_VERSION,
};
use crate::policy::{CombatSearchRecord, DecisionTelemetry, MetaEvalRecord};
use crate::view::VisibleRunView;

/// Recorder schema version for the `sts2sim` extension block.
///
/// * **1** — decisions, phase-typed offers, combat boards.
/// * **2** — per-decision *why*: the combat MCTS root distribution
///   (`search`), the meta policy's scored candidates (`meta_eval`),
///   potions held, event options with previewed costs, and the reachable
///   rooms at a map choice. All additive and all optional, so a v1
///   reader still parses a v2 file.
pub const RECORDER_VERSION: u32 = 2;

/// How much per-decision detail to retain. Ledgers + room history are
/// always tracked (they feed the native projection); this gates the
/// `decisions[]` stream only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Detail {
    /// No per-decision stream — native summary only.
    Summary,
    /// Off-combat decisions only (card rewards, events, shop, rest, map).
    Decisions,
    /// Everything, including per-ply combat boards.
    Full,
}

impl Detail {
    /// Parse the CLI token. Unknown → `Err`.
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "summary" => Ok(Detail::Summary),
            "decisions" => Ok(Detail::Decisions),
            "full" => Ok(Detail::Full),
            other => Err(format!("unknown detail '{other}' (summary|decisions|full)")),
        }
    }

    /// CLI token form.
    #[must_use]
    pub fn token(self) -> &'static str {
        match self {
            Detail::Summary => "summary",
            Detail::Decisions => "decisions",
            Detail::Full => "full",
        }
    }
}

/// One `"CARD.X"` / `"CARD.X+"` token (upgrade suffix for the stream).
pub(crate) fn card_token(card_id: sts2_core::cards::CardId, upgraded: bool) -> String {
    let base = card_id.to_python_name();
    if upgraded {
        format!("CARD.{base}+")
    } else {
        format!("CARD.{base}")
    }
}

/// The upgrade-agnostic `"CARD.X"` token — used for floor-of-addition
/// tracking so upgrading a card in place does not re-stamp its floor.
fn card_base_token(card_id: sts2_core::cards::CardId) -> String {
    format!("CARD.{}", card_id.to_python_name())
}

fn relic_token(relic_id: sts2_core::RelicId) -> String {
    format!("RELIC.{}", relic_id.to_python_name())
}

pub(crate) fn potion_token(kind: sts2_core::potions::PotionKind) -> String {
    format!("POTION.{}", kind.python_name())
}

/// A visited map room, accumulated for `map_point_history`.
#[derive(Clone, Debug)]
struct RoomVisit {
    act: u8,
    floor: i32,
    map_point_type: String,
    monster_ids: Vec<String>,
    event_id: Option<String>,
    coord: Option<(i32, i32)>,
}

/// [`RunPolicy`] decorator that records a full trajectory.
#[derive(Debug)]
pub struct RecordingPolicy<P> {
    inner: P,
    detail: Detail,
    step: u64,
    decisions: Vec<Decision>,
    /// Per-base-token high-water floors: `deck_hw["CARD.X"][i]` is the floor
    /// at which the deck first held `i+1` copies of `CARD.X`. Never shrinks
    /// (a transient remove+re-add does not re-stamp), so floors stay stable.
    deck_hw: BTreeMap<String, Vec<i32>>,
    relic_hw: BTreeMap<String, Vec<i32>>,
    rooms: Vec<RoomVisit>,
    /// Lead enemy id of the most recent combat (killed-by fallback).
    last_combat_lead: Option<String>,
}

impl<P: RunPolicy> RecordingPolicy<P> {
    /// Wrap `inner`, recording at `detail`.
    pub fn new(inner: P, detail: Detail) -> Self {
        Self {
            inner,
            detail,
            step: 0,
            decisions: Vec::new(),
            deck_hw: BTreeMap::new(),
            relic_hw: BTreeMap::new(),
            rooms: Vec::new(),
            last_combat_lead: None,
        }
    }

    /// Borrow the wrapped policy (for escalation/fallback counters).
    pub fn inner(&self) -> &P {
        &self.inner
    }

    // ── ledger maintenance ────────────────────────────────────────────

    /// Raise the per-token high-water floors to cover the current multiset:
    /// for each token whose current count exceeds the recorded count, stamp
    /// the newly-covered copies with `floor`. Never lowers the water mark, so
    /// a transient remove+re-add keeps each copy's earliest floor and cannot
    /// batch-stamp cards at a later floor.
    fn bump_highwater(map: &mut BTreeMap<String, Vec<i32>>, current: &[String], floor: i32) {
        let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
        for t in current {
            *counts.entry(t.as_str()).or_insert(0) += 1;
        }
        for (tok, &cnt) in &counts {
            let floors = map.entry((*tok).to_string()).or_default();
            while floors.len() < cnt {
                floors.push(floor);
            }
        }
    }

    /// Update deck/relic ledgers and the room list from the live state.
    /// Cheap: reads `RunState` pub fields directly (no `VisibleRunView`).
    fn observe_state(&mut self, rm: &RunManager) {
        let rs = &rm.run_state;
        let floor = rs.total_floor;

        // Track floor-of-addition only off-combat: cards/relics are never
        // *added* to the master deck mid-fight (rewards resolve after), and
        // combat transitions can momentarily expose a doubled master deck
        // (deck copied into the draw pile) which would inflate the floors.
        // The upgrade-agnostic base token keeps an in-place upgrade
        // (`CARD.X` → `CARD.X+`) from registering as a new card.
        if !matches!(rm.phase(), Phase::Combat) {
            let deck_tokens: Vec<String> = rs
                .master_deck
                .iter()
                .map(|c| card_base_token(c.card_id))
                .collect();
            Self::bump_highwater(&mut self.deck_hw, &deck_tokens, floor);

            let relic_tokens: Vec<String> = rs.relics.iter().map(|r| relic_token(*r)).collect();
            Self::bump_highwater(&mut self.relic_hw, &relic_tokens, floor);
        }

        // Room visits: push on coord change.
        if let Some(room) = rs.current_room.as_ref() {
            let coord = room.map_coord.map(|c| (c.col, c.row));
            let changed = self.rooms.last().is_none_or(|r| r.coord != coord);
            if changed {
                self.rooms.push(RoomVisit {
                    act: rs.act_index,
                    floor,
                    map_point_type: format!("{:?}", room.room_type),
                    monster_ids: Vec::new(),
                    event_id: None,
                    coord,
                });
            }
        }
        // Enrich the current room with combat enemies / event id. The
        // combat side must be gated on `Phase::Combat`: `RunManager`
        // keeps the previous fight's `CombatState` in `rm.combat` after
        // `exit_combat_and_transition`, so an ungated read stamps the
        // stale fight's monster ids onto every following non-combat room.
        if let Some(last) = self.rooms.last_mut() {
            if let (Phase::Combat, Some(combat)) = (rm.phase(), rm.combat.as_ref()) {
                if last.monster_ids.is_empty() {
                    let ids: Vec<String> = combat
                        .enemy_uids
                        .iter()
                        .filter_map(|uid| combat.creatures.iter().find(|c| c.id == uid.0))
                        .map(|c| monster_token(c.monster_id.0))
                        .collect();
                    if let Some(first) = ids.first() {
                        self.last_combat_lead = Some(first.clone());
                    }
                    last.monster_ids = ids;
                }
            }
            if last.event_id.is_none() {
                if let Some(ev) = rs.current_event {
                    last.event_id = Some(format!("EVENT.{}", ev.python_name()));
                }
            }
        }
    }

    /// Build the phase-typed offered payload for the chosen action.
    fn build_offered(rm: &RunManager, action: &RunAction) -> Offered {
        match action {
            RunAction::PlayCard { hand_idx, .. } => {
                let (card, board) = combat_ply(rm, Some(*hand_idx));
                Offered::Combat(CombatPly {
                    action: "play".into(),
                    card,
                    potion: None,
                    round: rm.combat.as_ref().map_or(0, |c| c.round),
                    board,
                })
            }
            RunAction::EndTurn
            | RunAction::DrinkPotion { .. }
            | RunAction::CombatResolveChoice { .. }
            | RunAction::CombatResolveChoiceTakeFirst { .. } => {
                let (_c, board) = combat_ply(rm, None);
                Offered::Combat(CombatPly {
                    action: combat_action_label(action),
                    card: None,
                    // Which potion was drunk, resolved from the belt slot.
                    // The decision's `potions` list is a whole-belt
                    // snapshot, so two drinks in a row are otherwise
                    // indistinguishable from each other.
                    //
                    // The slot indexes the COMBAT belt: `RunState::potions`
                    // is frozen for the duration of a fight and only
                    // written back at `exit_combat`, so resolving against
                    // it names the potion that occupied that slot when the
                    // fight started — which, after one drink has shifted
                    // the belt, is the wrong potion.
                    potion: match action {
                        RunAction::DrinkPotion { potion_slot, .. } => rm
                            .combat
                            .as_ref()
                            .and_then(|c| c.player.potions.get(*potion_slot))
                            .map(|p| potion_token(*p)),
                        _ => None,
                    },
                    round: rm.combat.as_ref().map_or(0, |c| c.round),
                    board,
                })
            }
            RunAction::PickCardReward { idx } => {
                let view = VisibleRunView::from_run_manager(rm);
                let options: Vec<String> = view
                    .offers
                    .card_rewards
                    .iter()
                    .map(|d| card_token(d.card_id, d.upgraded))
                    .collect();
                let picked = idx.and_then(|i| options.get(i).cloned());
                Offered::CardReward {
                    options,
                    relic_offer: view.offers.relic_reward.map(relic_token),
                    picked,
                }
            }
            RunAction::SacrificeCardReward => {
                let view = VisibleRunView::from_run_manager(rm);
                Offered::CardReward {
                    options: view
                        .offers
                        .card_rewards
                        .iter()
                        .map(|d| card_token(d.card_id, d.upgraded))
                        .collect(),
                    relic_offer: view.offers.relic_reward.map(relic_token),
                    picked: None,
                }
            }
            RunAction::EventChoice { idx } => Offered::Event {
                event_id: rm
                    .run_state
                    .current_event
                    .map(|e| format!("EVENT.{}", e.python_name())),
                choice_idx: *idx,
                options: event_options(rm),
            },
            RunAction::ShopBuyCard { .. }
            | RunAction::ShopBuyRelic { .. }
            | RunAction::ShopBuyPotion { .. }
            | RunAction::ShopRemoveCard { .. }
            | RunAction::LeaveShop => {
                let view = VisibleRunView::from_run_manager(rm);
                let (cards, relics, potions) = shop_items(&view);
                Offered::Shop {
                    cards,
                    relics,
                    potions,
                    action: format!("{action:?}"),
                }
            }
            RunAction::RestOption { idx } => Offered::Rest {
                option_idx: Some(*idx),
                smith_card: None,
            },
            RunAction::SmithCard { idx } => {
                let view = VisibleRunView::from_run_manager(rm);
                Offered::Rest {
                    option_idx: None,
                    smith_card: view
                        .offers
                        .smith_options
                        .get(*idx)
                        .map(|d| card_token(d.card_id, d.upgraded)),
                }
            }
            RunAction::PickBossRelic { idx } => {
                let view = VisibleRunView::from_run_manager(rm);
                Offered::BossRelic {
                    options: view
                        .offers
                        .boss_relics
                        .iter()
                        .map(|r| relic_token(*r))
                        .collect(),
                    picked_idx: *idx,
                }
            }
            RunAction::Move { col } => {
                let reachable = reachable_rooms(rm);
                Offered::Map {
                    col: *col,
                    room: reachable
                        .iter()
                        .find(|(c, _)| c == col)
                        .map(|(_, room)| room.clone()),
                    reachable,
                }
            }
            other => Offered::Other {
                action: format!("{other:?}"),
            },
        }
    }

    /// Consume the recorder and project the native `RunHistory` + our
    /// `sts2sim` block. `meta` carries reproducibility fields the recorder
    /// cannot see (policy label, budgets, model sha, env).
    #[must_use]
    pub fn into_run_file(
        mut self,
        rm: &RunManager,
        report: &FullRunReport,
        mut meta: Meta,
    ) -> RunFile {
        // Final reconciliation so the last action's adds/removes land.
        self.observe_state(rm);
        let rs = &rm.run_state;

        // Native deck = the authoritative final `master_deck` (contents +
        // upgrade), with each card's floor drawn from the high-water map
        // (earliest-first) so counts/upgrades are exact and floors stable.
        let mut deck_floors = self.deck_hw.clone();
        let deck: Vec<DeckCardEntry> = rs
            .master_deck
            .iter()
            .map(|c| {
                let base = card_base_token(c.card_id);
                let floor = deck_floors
                    .get_mut(&base)
                    .and_then(|v| {
                        if v.is_empty() {
                            None
                        } else {
                            Some(v.remove(0))
                        }
                    })
                    .unwrap_or(0);
                DeckCardEntry {
                    id: base,
                    floor_added_to_deck: floor,
                    upgrades: u32::from(c.upgraded),
                }
            })
            .collect();
        let mut relic_floors = self.relic_hw.clone();
        let relics: Vec<RelicEntry> = rs
            .relics
            .iter()
            .map(|r| {
                let id = relic_token(*r);
                let floor = relic_floors
                    .get_mut(&id)
                    .and_then(|v| {
                        if v.is_empty() {
                            None
                        } else {
                            Some(v.remove(0))
                        }
                    })
                    .unwrap_or(0);
                RelicEntry {
                    id,
                    floor_added_to_deck: floor,
                }
            })
            .collect();
        let potions: Vec<PotionEntry> = rs
            .potions
            .iter()
            .map(|p| PotionEntry {
                id: potion_token(*p),
            })
            .collect();

        // Group room visits into per-act lists.
        let mut map_point_history: Vec<Vec<MapPointEntry>> = Vec::new();
        for rv in &self.rooms {
            let act_idx = rv.act.saturating_sub(1) as usize;
            while map_point_history.len() <= act_idx {
                map_point_history.push(Vec::new());
            }
            map_point_history[act_idx].push(MapPointEntry {
                map_point_type: rv.map_point_type.clone(),
                floor: rv.floor,
                monster_ids: rv.monster_ids.clone(),
                event_id: rv.event_id.clone(),
            });
        }

        let killed_by_encounter = if report.player_won {
            "NONE.NONE".to_string()
        } else {
            self.last_combat_lead
                .clone()
                .unwrap_or_else(|| "NONE.NONE".to_string())
        };

        meta.steps = report.steps;
        meta.final_hp = report.final_hp;
        meta.final_gold = report.final_gold;

        RunFile {
            schema_version: SCHEMA_VERSION,
            platform_type: "sts2sim".into(),
            game_mode: "standard".into(),
            win: report.player_won,
            seed: seed_to_string(meta.seed_int),
            start_time: 0,
            run_time: 0.0,
            ascension: meta.ascension,
            build_id: meta.env.get("build_id").cloned().unwrap_or_default(),
            was_abandoned: false,
            killed_by_encounter,
            killed_by_event: "NONE.NONE".into(),
            players: vec![RunPlayer {
                id: 1,
                character: format!("CHARACTER.{}", meta.character.to_uppercase()),
                deck,
                relics,
                potions,
                badges: Vec::new(),
                max_potion_slot_count: rs.max_potion_slots,
            }],
            acts: Vec::new(),
            modifiers: Vec::new(),
            map_point_history,
            sts2sim: Sts2Sim {
                recorder_version: RECORDER_VERSION,
                meta,
                decisions: self.decisions,
            },
        }
    }
}

impl<P: RunPolicy + DecisionTelemetry> RunPolicy for RecordingPolicy<P> {
    fn prepare(&mut self, rm: &mut RunManager) {
        // Forward — PPO's `observe` materializes pending events here.
        self.inner.prepare(rm);
    }

    fn choose(&mut self, rm: &RunManager, actions: &[RunAction]) -> Option<usize> {
        let idx = self.inner.choose(rm, actions);
        // Telemetry is read (and, for the meta eval, consumed) on every
        // decision — including ones we don't record — so a decision the
        // scorer skipped can never inherit an older distribution. The
        // search slot is cleared per *combat* decision, not per decision,
        // so the phase guard is what keeps the card reward after a fight
        // from inheriting the fight's last ply.
        let search = if rm.phase() == Phase::Combat {
            self.inner.combat_search().map(search_telemetry)
        } else {
            None
        };
        let meta_eval = self.inner.take_meta_eval();
        // Always track ledgers + rooms (feed the native projection).
        self.observe_state(rm);

        if let Some(i) = idx {
            let action = actions[i];
            let phase = rm.phase();
            let is_combat = matches!(phase, Phase::Combat);
            let record = match self.detail {
                Detail::Summary => false,
                Detail::Decisions => !is_combat,
                Detail::Full => true,
            };
            if record {
                let rs = &rm.run_state;
                let offered = Self::build_offered(rm, &action);
                // `RunState::current_hp` is only reconciled when a combat
                // ENDS, so reading it mid-fight reports the HP the player
                // walked in with — a potion drunk at 58/80 recorded as
                // "drunk at 80/80", i.e. exactly inverting the judgement a
                // reviewer is trying to make. Inside a fight the live
                // values are the combat player's.
                let (hp, max_hp) = match rm.combat.as_ref() {
                    Some(c) if is_combat => (c.player.current_hp, c.player.max_hp),
                    _ => (rs.current_hp, rs.max_hp),
                };
                self.decisions.push(Decision {
                    step: self.step,
                    phase: format!("{phase:?}"),
                    act: rs.act_index,
                    floor: rs.total_floor,
                    hp,
                    max_hp,
                    gold: rs.gold,
                    // Live belt: inside a fight `RunState::potions` is the
                    // stale entry snapshot (see the `potion` field above),
                    // so a potion drunk mid-fight would still be listed as
                    // held for every later ply of that fight.
                    potions: match rm.combat.as_ref() {
                        Some(c) if is_combat => {
                            c.player.potions.iter().map(|p| potion_token(*p)).collect()
                        }
                        _ => rs.potions.iter().map(|p| potion_token(*p)).collect(),
                    },
                    chosen_idx: i,
                    n_options: actions.len(),
                    offered,
                    search,
                    meta_eval: meta_eval.map(|e| meta_eval_telemetry(&e, &action)),
                });
            }
        }
        self.step += 1;
        idx
    }
}

/// `(column, room type)` for every map node reachable from the current
/// one — what the path choice was actually choosing between.
///
/// A recorded `Move { col: 6 }` alone cannot answer "did it walk past a
/// shop to take a third elite at 20 HP?", which is the whole of route
/// review. Mirrors [`RunManager::next_map_children_cols`]'s node set: the
/// current room's children, or the start node's children on row 0.
fn reachable_rooms(rm: &RunManager) -> Vec<(i32, String)> {
    let Some(act_map) = rm.run_state.current_act_map.as_ref() else {
        return Vec::new();
    };
    let children = match rm.run_state.current_room.and_then(|r| r.map_coord) {
        Some(coord) => act_map.children_of(coord),
        None => match act_map.start {
            Some(s) => act_map.children_of(act_map.point(s).coord),
            None => Vec::new(),
        },
    };
    let mut rooms: Vec<(i32, String)> = children
        .iter()
        .map(|p| (p.coord.col, format!("{:?}", p.point_type)))
        .collect();
    rooms.sort_by_key(|(col, _)| *col);
    rooms.dedup_by_key(|(col, _)| *col);
    rooms
}

/// The active event's presented options, each with the HP/gold/max-HP
/// delta resolving it would cost (`preview_option_effect`, exact — it
/// resolves on a clone).
///
/// Without this an event decision records as `EventChoice { idx: 1 }`,
/// which says nothing about what was on the table: the option text and
/// its price are exactly what makes an event pick reviewable after the
/// fact. Empty when no event is bound.
fn event_options(rm: &RunManager) -> Vec<super::schema::EventOption> {
    let rs = &rm.run_state;
    let Some(presented) = sts2_content::events::dispatch::present(rs) else {
        return Vec::new();
    };
    presented
        .iter()
        .enumerate()
        .map(|(i, o)| {
            let delta = sts2_content::events::dispatch::preview_option_effect(rs, i);
            super::schema::EventOption {
                option_id: o.option_id.to_string(),
                label: o.label.clone(),
                description: o.description.clone(),
                enabled: o.enabled,
                hp_delta: delta.map(|d| d.hp_delta),
                gold_delta: delta.map(|d| d.gold_delta),
                max_hp_delta: delta.map(|d| d.max_hp_delta),
            }
        })
        .collect()
}

/// Project a combat search record into its serializable form: root
/// children Q-RANKED (mean value desc, then visits desc) plus the gap
/// between the two best Q values.
///
/// Q-rank is a stable readability/schema convention, NOT the deployed
/// pick rule: since the 2026-08-09 A8 audit the pick is
/// `Mcts::pick_root_action` (override → proven-win guard → tie-band
/// visits → root guards), so `chosen` routinely differs from `root[0]`
/// — on ~71% of audited plies the band alone can flip to a
/// higher-visit child — and `top_gap` measures the Q spread of the
/// top-2 Q-ranked children, not the pick's margin. Consumers must key
/// on `chosen`, never on `root[0]`, to recover the agent's decision.
fn search_telemetry(rec: &CombatSearchRecord) -> SearchTelemetry {
    let mut sorted = rec.outcomes.clone();
    sorted.sort_by(|a, b| {
        b.mean_value
            .partial_cmp(&a.mean_value)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(b.visits.cmp(&a.visits))
    });
    let top_gap = match sorted.as_slice() {
        [first, second, ..] => Some(first.mean_value - second.mean_value),
        _ => None,
    };
    SearchTelemetry {
        visits: sorted.iter().map(|o| o.visits).sum(),
        root: sorted
            .iter()
            .map(|o| RootAction {
                action: format!("{:?}", o.action),
                visits: o.visits,
                mean_value: o.mean_value,
                wins: o.wins,
                losses: o.losses,
            })
            .collect(),
        chosen: format!("{:?}", rec.chosen),
        top_gap,
    }
}

/// Project a meta-policy scoring into its serializable form, resolving
/// the probability the scorer put on the action actually taken.
fn meta_eval_telemetry(rec: &MetaEvalRecord, taken: &RunAction) -> MetaEvalTelemetry {
    let chosen_prob = rec
        .actions
        .iter()
        .position(|a| a == taken)
        .and_then(|i| rec.probs.get(i).copied());
    // Shannon entropy in nats; `p·ln p → 0` as `p → 0`, so zero-prob
    // candidates contribute nothing and cannot produce a `NaN`.
    let entropy = if rec.probs.is_empty() {
        None
    } else {
        Some(
            -rec.probs
                .iter()
                .filter(|&&p| p > 0.0)
                .map(|&p| p * p.ln())
                .sum::<f32>(),
        )
    };
    MetaEvalTelemetry {
        source: rec.source.to_string(),
        options: rec
            .actions
            .iter()
            .enumerate()
            .map(|(i, a)| ScoredOption {
                action: format!("{a:?}"),
                prob: rec.probs.get(i).copied().unwrap_or(0.0),
            })
            .collect(),
        value: rec.value,
        chosen_prob,
        entropy,
    }
}

/// Resolve the combat board (parity bridge) and, when `hand_idx` is set,
/// the played card token.
fn combat_ply(rm: &RunManager, hand_idx: Option<usize>) -> (Option<String>, serde_json::Value) {
    let Some(combat) = rm.combat.as_ref() else {
        return (None, serde_json::Value::Null);
    };
    let card = hand_idx.and_then(|hi| {
        combat.player.hand.get(hi).and_then(|uid| {
            combat
                .cards
                .iter()
                .find(|c| c.uid == *uid)
                .map(|c| card_token(c.card_id, c.upgraded))
        })
    });
    let board = sts2_core::parity::combat_state_to_bridge_state(combat);
    (card, board)
}

fn combat_action_label(action: &RunAction) -> String {
    match action {
        RunAction::EndTurn => "end_turn".into(),
        RunAction::DrinkPotion { .. } => "potion".into(),
        RunAction::CombatResolveChoice { .. } | RunAction::CombatResolveChoiceTakeFirst { .. } => {
            "resolve".into()
        }
        other => format!("{other:?}"),
    }
}

fn shop_items(view: &VisibleRunView) -> (Vec<ShopItem>, Vec<ShopItem>, Vec<ShopItem>) {
    let Some(shop) = view.offers.shop.as_ref() else {
        return (Vec::new(), Vec::new(), Vec::new());
    };
    let cards = shop
        .cards
        .iter()
        .flatten()
        .map(|c| ShopItem {
            id: card_token(c.card.card_id, c.card.upgraded),
            price: c.price,
        })
        .collect();
    let relics = shop
        .relics
        .iter()
        .flatten()
        .map(|r| ShopItem {
            id: relic_token(r.relic_id),
            price: r.price,
        })
        .collect();
    let potions = shop
        .potions
        .iter()
        .flatten()
        .map(|p| ShopItem {
            id: potion_token(p.potion_kind),
            price: p.price,
        })
        .collect();
    (cards, relics, potions)
}

/// `"MONSTER.X"` from a raw monster id string (already-prefixed ids pass
/// through).
pub(crate) fn monster_token(raw: &str) -> String {
    if raw.contains('.') {
        raw.to_string()
    } else {
        format!("MONSTER.{raw}")
    }
}

/// Native `.run` files use an alphanumeric seed string; our runs are
/// integer-seeded, so stamp the decimal form (round-trips via `meta.seed_int`).
fn seed_to_string(seed: i32) -> String {
    seed.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detail_parse_round_trips() {
        for d in [Detail::Summary, Detail::Decisions, Detail::Full] {
            assert_eq!(Detail::parse(d.token()), Ok(d));
        }
        assert!(Detail::parse("nope").is_err());
    }

    /// Minimal policy so `RecordingPolicy::<NullPolicy>::reconcile` (an
    /// associated fn) is reachable in tests.
    #[derive(Debug)]
    struct NullPolicy;
    impl RunPolicy for NullPolicy {
        fn choose(&mut self, _rm: &RunManager, _actions: &[RunAction]) -> Option<usize> {
            None
        }
    }
    impl DecisionTelemetry for NullPolicy {}

    #[test]
    fn highwater_stamps_first_floor_and_never_restamps() {
        let mut hw: BTreeMap<String, Vec<i32>> = BTreeMap::new();
        // Floor 0: starter Strike x2.
        RecordingPolicy::<NullPolicy>::bump_highwater(
            &mut hw,
            &["CARD.STRIKE".into(), "CARD.STRIKE".into()],
            0,
        );
        assert_eq!(hw["CARD.STRIKE"], vec![0, 0]);
        // Floor 3: add Bash.
        RecordingPolicy::<NullPolicy>::bump_highwater(
            &mut hw,
            &[
                "CARD.STRIKE".into(),
                "CARD.STRIKE".into(),
                "CARD.BASH".into(),
            ],
            3,
        );
        assert_eq!(hw["CARD.BASH"], vec![3]);
        // Floor 5: deck transiently shows only one Strike — high-water does
        // NOT lower (both floor-0 stamps retained).
        RecordingPolicy::<NullPolicy>::bump_highwater(
            &mut hw,
            &["CARD.STRIKE".into(), "CARD.BASH".into()],
            5,
        );
        assert_eq!(hw["CARD.STRIKE"], vec![0, 0]);
        // Floor 33: a churn that briefly re-shows two Strikes must NOT
        // re-stamp them at floor 33 — the earliest floors win.
        RecordingPolicy::<NullPolicy>::bump_highwater(
            &mut hw,
            &["CARD.STRIKE".into(), "CARD.STRIKE".into()],
            33,
        );
        assert_eq!(hw["CARD.STRIKE"], vec![0, 0]);
        // A genuinely new 3rd Strike copy at floor 33 gets floor 33.
        RecordingPolicy::<NullPolicy>::bump_highwater(
            &mut hw,
            &[
                "CARD.STRIKE".into(),
                "CARD.STRIKE".into(),
                "CARD.STRIKE".into(),
            ],
            33,
        );
        assert_eq!(hw["CARD.STRIKE"], vec![0, 0, 33]);
    }
}
