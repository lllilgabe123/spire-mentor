//! `Game` — the workbench's driver over one `RunManager`.
//!
//! Port of `RustGame` in `sts2-pycli/python/workbench/backends/rust_game.py`
//! (the `target_convention="native"` half; the C#-dialect emulation, the
//! faithful/run-mode constructor and the live-state import are later
//! milestones). It translates the workbench's C#-shaped verbs
//! (`play_card`, `select_cards`, `choose_option`, …) into engine
//! `RunAction`s and rebuilds the C#-shaped state dict the SPA consumes
//! from the engine's accessors — `decision` / `player` / `enemies` / `hand`
//! / `options` … — emitting card / relic / power names as loc keys for
//! the enrichment step to resolve, exactly as Python did.
//!
//! Every dict is a `serde_json::Value` built key-for-key from the Python:
//! the golden-parity tests diff the two outputs structurally.

use serde_json::{json, Map, Value};
use sts2_core::potions::CharacterId;
use sts2_core::run_manager::{RunAction, RunManager};
use sts2_core::{CombatState, GameVersion};

use crate::combat_view::{
    self, action_label, deck_card, offer_card, owned_entry, treasure_reveal, RoomSnapshot,
};
use crate::display_maps::{loc_id, monster_title};
use crate::rmx;

/// An action the engine rejected or the backend cannot express. The
/// session turns it into the `{"type": "error", "message"}` reply Python's
/// `_err` produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GameError(pub String);

impl std::fmt::Display for GameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for GameError {}

fn err<T>(msg: impl Into<String>) -> Result<T, GameError> {
    Err(GameError(msg.into()))
}

/// C# `decision` ← Rust `phase()` (`_PHASE_TO_DECISION`).
fn phase_to_decision(phase: &str) -> &'static str {
    match phase {
        "map_choice" => "map_select",
        "combat" => "combat_play",
        "card_reward" => "card_reward",
        "shop" => "shop",
        "rest_site" => "rest_site",
        "event" | "treasure" | "boss_relic" => "event_choice",
        "run_over" => "game_over",
        _ => "unknown",
    }
}

/// `_REST_LABELS`: display fallback for a rest-site option whose engine
/// `option_id` is unavailable.
fn rest_label(idx: i64) -> String {
    match idx {
        0 => "Rest".to_string(),
        1 => "Smith".to_string(),
        other => format!("Option {other}"),
    }
}

/// `_event_loc_stem`: `LostWisp` (event Python class name) -> `LOST_WISP`.
fn event_loc_stem(python_name: &str) -> String {
    let mut out = String::with_capacity(python_name.len() + 4);
    for (i, c) in python_name.chars().enumerate() {
        if i > 0 && c.is_uppercase() {
            out.push('_');
        }
        out.push(c);
    }
    out.to_uppercase()
}

/// Python `str.isupper()`: at least one cased character and every cased
/// character is upper-case.
fn py_isupper(s: &str) -> bool {
    let mut cased = false;
    for c in s.chars() {
        if c.is_lowercase() {
            return false;
        }
        if c.is_uppercase() {
            cased = true;
        }
    }
    cased
}

/// Python `str(raw).split(",")` over an `indices` arg (a JSON string or
/// number), keeping non-blank entries as ints. Errors on a non-integer.
fn parse_indices(raw: Option<&Value>) -> Result<Vec<i64>, GameError> {
    let text = match raw {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(other) => combat_view::py_str(other),
    };
    let mut out = Vec::new();
    for part in text.split(',') {
        let t = part.trim();
        if t.is_empty() {
            continue;
        }
        match t.parse::<i64>() {
            Ok(v) => out.push(v),
            Err(_) => return err(format!("invalid literal for int() with base 10: '{t}'")),
        }
    }
    Ok(out)
}

/// `sorted(set(idxs))` (ascending) or `reverse=True`.
fn sorted_unique(mut idxs: Vec<i64>, descending: bool) -> Vec<i64> {
    idxs.sort_unstable();
    idxs.dedup();
    if descending {
        idxs.reverse();
    }
    idxs
}

fn arg_i64(args: &Value, key: &str) -> Result<i64, GameError> {
    match args.get(key) {
        Some(v) => v
            .as_i64()
            .ok_or_else(|| GameError(format!("argument {key:?} must be an integer, got {v}"))),
        None => err(format!("missing argument {key:?}")),
    }
}

fn opt_target(args: &Value) -> Option<u32> {
    args.get("target_index")
        .and_then(Value::as_i64)
        .and_then(|t| u32::try_from(t).ok())
}

fn usize_of(i: i64) -> usize {
    usize::try_from(i).unwrap_or(usize::MAX)
}

/// The one live `RunManager` and the per-screen bookkeeping around it.
#[derive(Debug)]
pub struct Game {
    rm: Option<RunManager>,
    /// The display character name the run was started with (`"Ironclad"`).
    pub character: String,
    /// The Act-1 model this run was built with (`"overgrowth"` /
    /// `"underdocks"` / `"roll"`), remembered for saves and seek.
    pub act1: String,
    /// The game build every `start*` constructs its run in (GH #801). Set
    /// before `start`; defaults to v0.107.1.
    pub game_version: GameVersion,
    record_card_plays: bool,
    record_combat_log: bool,
    /// Engine actions backing the option indices of the CURRENT state, for
    /// the screens rendered via the generic per-action option fallback.
    option_actions: Vec<RunAction>,
    /// True while the shop's synthetic card-removal `card_select` is open.
    shop_removal_pick: bool,
    /// Gold / relics / room BEFORE the action being applied
    /// (`_treasure_reveal`). `None` until the first action.
    pre_action: Option<RoomSnapshot>,
    /// What the last `start_imported` graft could NOT reproduce (a map that
    /// disagrees with the paste, a refused RNG restore, an unupgradable card).
    /// Empty on every other start path.
    import_notes: Vec<String>,
}

impl Default for Game {
    fn default() -> Self {
        Self::new()
    }
}

impl Game {
    #[must_use]
    pub fn new() -> Self {
        Self {
            rm: None,
            character: String::new(),
            act1: "overgrowth".to_string(),
            game_version: GameVersion::default(),
            record_card_plays: false,
            record_combat_log: false,
            option_actions: Vec::new(),
            shop_removal_pick: false,
            pre_action: None,
            import_notes: Vec::new(),
        }
    }

    /// What the last [`Self::start_imported`] could not reproduce.
    #[must_use]
    pub fn import_notes(&self) -> Vec<String> {
        self.import_notes.clone()
    }

    /// The in-process `RunManager` (`None` before `start`).
    #[must_use]
    pub fn native_run(&self) -> Option<&RunManager> {
        self.rm.as_ref()
    }

    /// Mutable access to the run (native-action replay).
    pub fn native_run_mut(&mut self) -> Option<&mut RunManager> {
        self.rm.as_mut()
    }

    /// The native combat state, or `None` outside combat.
    #[must_use]
    pub fn combat_state(&self) -> Option<&CombatState> {
        let rm = self.rm.as_ref()?;
        if rmx::phase_str(rm) != "combat" {
            return None;
        }
        rm.combat.as_ref()
    }

    fn rm(&self) -> Result<&RunManager, GameError> {
        self.rm
            .as_ref()
            .ok_or_else(|| GameError("no live run".into()))
    }

    // ── lifecycle ──────────────────────────────────────────────────────

    /// `RustGame.start`: build the run the way the interactive workbench
    /// does (Neow start, every C#-faithful interactive deck-mutation flag,
    /// direct Neow relic grant, the requested Act-1 model) and return the
    /// opening state.
    pub fn start(
        &mut self,
        character: &str,
        seed: &str,
        ascension: u8,
        act1: &str,
    ) -> Result<Value, GameError> {
        let cid = rmx::character_from_str(character).map_err(GameError)?;
        self.character = character.to_string();
        self.act1 = act1.to_string();
        self.rm = Some(build_run_manager(
            self.game_version,
            cid,
            seed,
            ascension,
            act1,
        )?);
        self.option_actions.clear();
        self.shop_removal_pick = false;
        self.pre_action = None;
        // Apply the remembered play-log recording preferences now that the
        // RunManager exists (they no-op while it is `None`).
        if self.record_card_plays {
            self.set_record_card_plays(true);
        }
        if self.record_combat_log {
            self.set_record_combat_log(true);
        }
        Ok(self.serialize())
    }

    /// `RustGame.start_faithful`: a run constructed IDENTICALLY to the PPO
    /// training env (`PpoRunEnv.__post_init__` / the arena's
    /// `new_arena_run_manager`): integer seed, Neow start, the deck-mutation
    /// flags on, potion swap off, the direct Neow relic grant, and — on
    /// purpose — Overgrowth, the act the deployed nets trained on.
    pub fn start_faithful(
        &mut self,
        character: &str,
        seed: i32,
        ascension: u8,
    ) -> Result<Value, GameError> {
        let cid = rmx::character_from_str(character).map_err(GameError)?;
        self.character = character.to_string();
        self.act1 = "overgrowth".to_string();
        self.rm = Some(build_faithful_run_manager(
            self.game_version,
            cid,
            seed,
            ascension,
        ));
        self.option_actions.clear();
        self.shop_removal_pick = false;
        self.pre_action = None;
        if self.record_card_plays {
            self.set_record_card_plays(true);
        }
        if self.record_combat_log {
            self.set_record_combat_log(true);
        }
        Ok(self.serialize())
    }

    /// `RustGame.start_imported`: rebuild the run behind a state copied out of
    /// the live game. The run identity (seed, character, ascension) is REAL —
    /// constructed the same way — so the act maps, boss and pre-rolled
    /// encounter queue are the live game's. What the engine cannot derive from
    /// a seed is grafted from the paste: position, deck, relics, potions, gold,
    /// HP, and the RNG cursors.
    ///
    /// Order is load-bearing (mirrors `RustGame.start_imported`):
    /// 1. Place on the map first, so the run parks in `map_choice`.
    /// 2. Empty the deck, THEN grant relics — `after_obtained` fires eagerly
    ///    and several relics upgrade cards on landing (Whetstone, War Paint);
    ///    over a starter deck they would upgrade cards the paste never had.
    /// 3. Deck, then upgrades, so "5 Strikes, 2 upgraded" lands as exactly that.
    /// 4. HP / gold last: the world's numbers are authoritative.
    /// 5. RNG cursors last of all, so nothing above advances a stream after it
    ///    is positioned.
    ///
    /// `act1` is not guessed: the run is built with the rolled Act-1 model
    /// (GH #544), and the resolved value is read back off the run.
    ///
    /// # Errors
    /// [`GameError`] when the character has no port, or the placement is
    /// rejected (a coord not on this seed's map — the paste and the engine
    /// disagree about the run).
    pub fn start_imported(
        &mut self,
        snapshot: &crate::state_import::RunSnapshot,
    ) -> Result<Value, GameError> {
        use sts2_core::map::MapCoord;

        let cid = rmx::character_from_str(&snapshot.character).map_err(GameError)?;
        self.character.clone_from(&snapshot.character);
        let mut rm =
            build_import_run_manager(self.game_version, cid, &snapshot.seed, snapshot.ascension);
        self.act1 = rmx::act1_str(&rm).to_string();
        let mut notes: Vec<String> = Vec::new();

        // 1. Place on the map (or note the run starts at the act entrance).
        if let Some((col, row)) = snapshot.coord() {
            let visited: Vec<MapCoord> = snapshot
                .visited()
                .into_iter()
                .map(|(c, r)| MapCoord::new(c, r))
                .collect();
            let act = u8::try_from(snapshot.act.max(1)).unwrap_or(1);
            let total_floor = i32::try_from(snapshot.floor).unwrap_or(0);
            rm.debug_place_on_map(act, MapCoord::new(col, row), visited, row + 1, total_floor)
                .map_err(|e| GameError(format!("could not place the run on the map: {e}")))?;
            notes.extend(map_disagreements(&rm, snapshot));
        } else {
            notes.push("No map position in the paste — the run starts at the act entrance.".into());
        }

        // 2. Empty the deck BEFORE granting relics (see the docstring).
        rm.run_state.master_deck.clear();
        for relic in &snapshot.relics {
            graft_add_relic(&mut rm, relic);
        }
        // 3. Deck, then upgrades.
        for (card_id, _upgraded) in &snapshot.deck {
            graft_add_card(&mut rm, card_id);
        }
        for (card_id, upgraded) in &snapshot.deck {
            if *upgraded && !graft_upgrade_card(&mut rm, card_id) {
                notes.push(format!(
                    "could not upgrade {card_id} (no copy left to upgrade)"
                ));
            }
        }
        // Potions, then belt-size sanity.
        graft_set_potions(&mut rm, &snapshot.potions);
        let slots = i64::from(rm.run_state.max_potion_slots);
        if snapshot.potion_slots > 0 && slots != snapshot.potion_slots {
            notes.push(format!(
                "belt is {slots} slots here, {} in the live run",
                snapshot.potion_slots
            ));
        }
        // 4. HP / gold last.
        rm.debug_set_player(
            Some(i32::try_from(snapshot.hp).unwrap_or(0)),
            Some(i32::try_from(snapshot.max_hp).unwrap_or(0)),
            Some(i32::try_from(snapshot.gold).unwrap_or(0)),
        );
        // 5. RNG cursors last of all.
        if let Some(rng) = &snapshot.rng {
            notes.extend(graft_rng(&mut rm, rng));
        }
        // 6. The combat lane (GH #708): the exact fight, rebuilt from its
        // wire dict and dropped onto the run as it stands. The board carries
        // its own RNG lineage, so nothing here re-rolls; a board the engine
        // refuses is an error (the caller keeps the run it had).
        if let Some(combat) = &snapshot.combat {
            // The fight is rebuilt in the RUN's build: the bridge's combat
            // wire never names one, and the wire parser would otherwise fall
            // back to v0.107.1 (refusing a 64-bit v0.111.0 `rng.run_seed`, or
            // dropping a 0.107 board into a 0.111 run). Stamped at use time so
            // the stored snapshot bytes — and the default build — are as-is.
            let mut wire = combat.wire.clone();
            if !self.game_version.is_default() {
                if let Some(obj) = wire.as_object_mut() {
                    obj.insert("game_version".into(), json!(self.game_version.as_str()));
                }
            }
            let cs = sts2_wire::state_from_json(&wire)
                .map_err(|e| GameError(format!("the engine refused the fight: {e}")))?;
            rm.debug_enter_combat(cs, combat.room_kind());
        }

        self.rm = Some(rm);
        self.import_notes = notes;
        self.option_actions.clear();
        self.shop_removal_pick = false;
        self.pre_action = None;
        if self.record_card_plays {
            self.set_record_card_plays(true);
        }
        if self.record_combat_log {
            self.set_record_combat_log(true);
        }
        Ok(self.serialize())
    }

    /// Apply a raw engine action (a native-agent step or its replay),
    /// honouring the play-log flags exactly as `act` does. Returns the
    /// engine's result; the caller decides what a rejection means.
    pub fn take_native(&mut self, action: RunAction) -> sts2_core::run_manager::ActionResult {
        let (rcp, rcl) = (self.record_card_plays, self.record_combat_log);
        let rm = self.rm.as_mut().expect("take_native needs a run");
        // Deliberately NOT re-snapshotting `pre_action`: the Python session
        // drives these through `rm.take_action` directly, so `room_reveal`
        // keeps comparing against the last hand-played action's snapshot.
        rmx::ensure_play_log_flag(rm, rcp, rcl);
        rm.take_action(action)
    }

    /// Drop the run.
    pub fn close(&mut self) {
        self.rm = None;
    }

    /// Opt into the engine card play-log (event-log labels).
    pub fn set_record_card_plays(&mut self, on: bool) {
        self.record_card_plays = on;
        if let Some(rm) = self.rm.as_mut() {
            rmx::set_record_card_plays(rm, on);
        }
    }

    /// Opt into the full game-style combat log (the log panel).
    pub fn set_record_combat_log(&mut self, on: bool) {
        self.record_combat_log = on;
        if let Some(rm) = self.rm.as_mut() {
            rmx::set_record_combat_log(rm, on);
        }
    }

    /// The card play-log rows (empty outside combat / before `start`).
    #[must_use]
    pub fn card_play_log(&self) -> Vec<Value> {
        self.rm.as_ref().map(rmx::card_play_log).unwrap_or_default()
    }

    /// This combat's raw event rows, oldest first.
    #[must_use]
    pub fn combat_log(&self) -> Vec<Value> {
        self.rm.as_ref().map(rmx::combat_log).unwrap_or_default()
    }

    /// `get_state`: the current C#-shaped state dict.
    pub fn get_state(&mut self) -> Result<Value, GameError> {
        if self.rm.is_none() {
            return err("no live run");
        }
        Ok(self.serialize())
    }

    /// `get_map`: the full act graph (`rows` / `boss` / `current_coord` …)
    /// with the boss nodes named and the path-plan overlay attached.
    /// Empty object when no map is loaded.
    #[must_use]
    pub fn get_map(&self, first_col: Option<i32>) -> Value {
        let Some(rm) = self.rm.as_ref() else {
            return Value::Object(Map::new());
        };
        let mut full_map = match rmx::full_map(rm) {
            Value::Object(m) => m,
            _ => Map::new(),
        };
        if let Some(boss) = full_map.get_mut("boss").and_then(Value::as_object_mut) {
            let kind = boss
                .get("kind")
                .and_then(Value::as_str)
                .filter(|k| !k.is_empty())
                .map(str::to_string);
            let has_name = boss.get("name").is_some_and(combat_view::truthy);
            if let (Some(kind), false) = (kind, has_name) {
                let name = combat_view::boss_kind_encounter_name(&kind)
                    .map_or_else(|| monster_title(&kind), str::to_string);
                boss.insert("name".into(), json!(name));
            }
        }
        if let Some(second) = full_map
            .get_mut("second_boss")
            .and_then(Value::as_object_mut)
        {
            let id = second
                .get("id")
                .and_then(Value::as_str)
                .filter(|k| !k.is_empty())
                .map(str::to_string);
            let has_name = second.get("name").is_some_and(combat_view::truthy);
            if let (Some(id), false) = (id, has_name) {
                second.insert("name".into(), json!(combat_view::boss_encounter_title(&id)));
            }
        }
        let plan = rmx::path_plan(rm, first_col);
        if plan.is_object() {
            full_map.insert("path_plan".into(), plan);
        }
        Value::Object(full_map)
    }

    // ── shop card-removal picker (synthetic card_select) ───────────────

    fn open_shop_removal_pick(&mut self) -> Result<Value, GameError> {
        let rm = self.rm()?;
        let offers: Vec<RunAction> = rmx::available_actions(rm)
            .into_iter()
            .filter(|a| rmx::action_kind(a) == "shop_remove_card")
            .collect();
        if offers.is_empty() {
            return err("card removal is not available here");
        }
        let deck = rmx::master_deck_detailed(rm);
        let mut cards = Vec::new();
        for a in &offers {
            let i = rmx::action_int(a, "deck_idx").unwrap_or(-1);
            if i < 0 || usize_of(i) >= deck.len() {
                continue;
            }
            let (cid, upgraded) = &deck[usize_of(i)];
            cards.push(deck_card(i, cid, *upgraded));
        }
        self.shop_removal_pick = true;
        let mut state = self.serialize();
        let m = state.as_object_mut().expect("state dict");
        m.insert("decision".into(), json!("card_select"));
        m.insert("cards".into(), Value::Array(cards));
        m.insert("min_select".into(), json!(0)); // min 0 → Skip (= cancel)
        m.insert("max_select".into(), json!(1));
        m.insert("can_skip".into(), json!(true));
        m.insert("context".into(), json!("remove"));
        Ok(state)
    }

    fn resolve_shop_removal_pick(
        &mut self,
        action: &str,
        args: &Value,
    ) -> Result<Value, GameError> {
        self.shop_removal_pick = false;
        let first = if action == "select_cards" {
            parse_indices(args.get("indices"))?.first().copied()
        } else {
            None
        };
        let Some(first) = first else {
            return Ok(self.serialize()); // cancel → back to the shop
        };
        let result = self.take(RunAction::ShopRemoveCard {
            deck_idx: usize_of(first),
        })?;
        if result != "advanced" && result != "run_over" {
            return err(format!("remove_card rejected by engine: {result}"));
        }
        Ok(self.serialize())
    }

    // ── action translation ─────────────────────────────────────────────

    /// `take_action` on the live run with the play-log flags applied.
    fn take(&mut self, action: RunAction) -> Result<&'static str, GameError> {
        let (rcp, rcl) = (self.record_card_plays, self.record_combat_log);
        let rm = self
            .rm
            .as_mut()
            .ok_or_else(|| GameError("no live run".into()))?;
        Ok(rmx::take_action(rm, action, rcp, rcl))
    }

    /// Apply a sequence of engine actions, failing on the first rejection
    /// with Python's `"{action} rejected by engine: {result}"` message.
    fn take_all(&mut self, verb: &str, actions: Vec<RunAction>) -> Result<Value, GameError> {
        for a in actions {
            let result = self.take(a)?;
            if result != "advanced" && result != "run_over" {
                return err(format!("{verb} rejected by engine: {result}"));
            }
        }
        Ok(self.serialize())
    }

    /// `RustGame.act`: apply one workbench verb.
    pub fn act(&mut self, action: &str, args: &Value) -> Result<Value, GameError> {
        if self.rm.is_none() {
            return err("no live run");
        }
        // What this action is about to change, for the treasure-reveal panel.
        self.pre_action = Some(self.room_snapshot());
        if action == "remove_card" {
            return self.open_shop_removal_pick();
        }
        if self.shop_removal_pick && (action == "select_cards" || action == "skip_select") {
            return self.resolve_shop_removal_pick(action, args);
        }
        if action == "select_cards" {
            let rm = self.rm()?;
            // Interactive deck removal: resolve EVERY chosen index in this one
            // act, descending so earlier removals don't shift later ones.
            if rmx::pending_deck_removals(rm) > 0 {
                let idxs = sorted_unique(parse_indices(args.get("indices"))?, true);
                if idxs.is_empty() {
                    return err("select_cards requires an index");
                }
                let acts = idxs
                    .into_iter()
                    .map(|i| RunAction::RemoveDeckCard { idx: usize_of(i) })
                    .collect();
                return self.take_all(action, acts);
            }
            // Interactive transform: ascending, each later index shifted down
            // by the number of earlier removals.
            if rmx::pending_deck_transforms(rm) > 0 {
                let idxs = sorted_unique(parse_indices(args.get("indices"))?, false);
                if idxs.is_empty() {
                    return err("select_cards requires an index");
                }
                let acts = idxs
                    .into_iter()
                    .enumerate()
                    .map(|(n, i)| RunAction::TransformDeckCard {
                        idx: usize_of(i - n as i64),
                    })
                    .collect();
                return self.take_all(action, acts);
            }
            // Interactive enchant: descending (an enchanted card drops out of
            // the eligible subset).
            if rmx::pending_deck_enchants(rm) > 0 {
                let idxs = sorted_unique(parse_indices(args.get("indices"))?, true);
                if idxs.is_empty() {
                    return err("select_cards requires an index");
                }
                let acts = idxs
                    .into_iter()
                    .map(|i| RunAction::EnchantDeckCard { idx: usize_of(i) })
                    .collect();
                return self.take_all(action, acts);
            }
            // Multi-pick card grant: ascending with shift, like transforms.
            if rmx::pending_card_grants(rm) > 0 {
                let idxs = sorted_unique(parse_indices(args.get("indices"))?, false);
                if idxs.is_empty() {
                    return err("select_cards requires an index");
                }
                let acts = idxs
                    .into_iter()
                    .enumerate()
                    .map(|(n, i)| RunAction::GrantDeckCard {
                        idx: usize_of(i - n as i64),
                    })
                    .collect();
                return self.take_all(action, acts);
            }
            // Multi-pick in-combat pending choice: MULTIPLE indices at once
            // resolve the whole choice through the explicit-selection action;
            // a single index is always a single pick (choice stays open).
            let idxs = parse_indices(args.get("indices"))?;
            if rmx::combat_choice_open(rm) && idxs.len() > 1 {
                let selection: Vec<u8> = idxs
                    .iter()
                    .map(|&i| u8::try_from(i).unwrap_or(u8::MAX))
                    .collect();
                let (rcp, rcl) = (self.record_card_plays, self.record_combat_log);
                let rm = self.rm.as_mut().expect("checked above");
                let result = rmx::resolve_choice_selection(rm, &selection, rcp, rcl);
                if result != "advanced" && result != "run_over" {
                    return err(format!("{action} rejected by engine: {result}"));
                }
                return Ok(self.serialize());
            }
        }
        let rust_action = self.translate(action, args)?;
        let result = self.take(rust_action)?;
        if result != "advanced" && result != "run_over" {
            return err(format!("{action} rejected by engine: {result}"));
        }
        Ok(self.serialize())
    }

    /// The workbench verb + args behind an engine action at the CURRENT
    /// state (`run_policy.resolve_workbench_verb`): an in-combat choice or a
    /// choose-screen pick has a fixed verb; every other action is looked up
    /// among `options` — the normalized option list the UI renders
    /// (`options::normalize_options` of the node state) — so "play best"
    /// dispatches exactly what clicking that option would. `None` when no
    /// rendered option translates to `action`.
    #[must_use]
    pub fn verb_for(&self, action: RunAction, options: &[Value]) -> Option<(String, Value)> {
        match action {
            RunAction::CombatResolveChoice { option: None } => {
                return Some(("skip_select".to_string(), json!({})));
            }
            RunAction::CombatResolveChoice { option: Some(o) } => {
                return Some((
                    "select_cards".to_string(),
                    json!({"indices": o.to_string()}),
                ));
            }
            RunAction::PickCardReward { idx }
                if self.rm().is_ok_and(|rm| {
                    rmx::phase_str(rm) == "card_reward" && rmx::card_offer_choose_screen(rm)
                }) =>
            {
                return Some(match idx {
                    None => ("skip_select".to_string(), json!({})),
                    Some(i) => (
                        "select_cards".to_string(),
                        json!({"indices": i.to_string()}),
                    ),
                });
            }
            _ => {}
        }
        for opt in options {
            if opt.get("enabled").and_then(Value::as_bool) == Some(false) {
                continue;
            }
            let verb = opt.get("action").and_then(Value::as_str)?;
            let args = opt.get("args").cloned().unwrap_or_else(|| json!({}));
            if self.translate(verb, &args).ok() == Some(action) {
                return Some((verb.to_string(), args));
            }
        }
        None
    }

    /// `_translate`: C# verb + args → an engine action.
    fn translate(&self, action: &str, args: &Value) -> Result<RunAction, GameError> {
        let rm = self.rm()?;
        let phase = rmx::phase_str(rm);
        match action {
            "select_map_node" => Ok(RunAction::Move {
                col: i32::try_from(arg_i64(args, "col")?).unwrap_or(i32::MAX),
            }),
            "play_card" => Ok(RunAction::PlayCard {
                hand_idx: usize_of(arg_i64(args, "card_index")?),
                // `"native"` target convention: already the FULL encounter
                // slot — passed straight through.
                target: opt_target(args),
            }),
            "end_turn" => Ok(RunAction::EndTurn),
            "use_potion" => Ok(RunAction::DrinkPotion {
                potion_slot: usize_of(arg_i64(args, "potion_index")?),
                target: opt_target(args),
            }),
            "select_cards" => {
                let first = parse_indices(args.get("indices"))?.first().copied();
                let Some(idx) = first else {
                    return err("select_cards requires an index");
                };
                let idx = usize_of(idx);
                if rmx::pending_deck_removals(rm) > 0 {
                    return Ok(RunAction::RemoveDeckCard { idx });
                }
                if rmx::pending_deck_upgrades(rm) > 0 {
                    return Ok(RunAction::UpgradeDeckCard { idx });
                }
                if rmx::pending_deck_transforms(rm) > 0 {
                    return Ok(RunAction::TransformDeckCard { idx });
                }
                if rmx::pending_deck_enchants(rm) > 0 {
                    return Ok(RunAction::EnchantDeckCard { idx });
                }
                // A choose screen renders as a card_select but resolves
                // through the card-reward pick. GH #34.
                if phase == "card_reward" && rmx::card_offer_choose_screen(rm) {
                    return Ok(RunAction::PickCardReward { idx: Some(idx) });
                }
                if phase == "rest_site" && !rmx::smith_options(rm).is_empty() {
                    return Ok(RunAction::SmithCard { idx });
                }
                if phase == "shop" {
                    return Ok(RunAction::ShopRemoveCard { deck_idx: idx });
                }
                if rmx::combat_choice_open(rm) {
                    return Ok(RunAction::CombatResolveChoice { option: Some(idx) });
                }
                Ok(RunAction::RemoveDeckCard { idx })
            }
            "skip_select" => {
                if rmx::combat_choice_open(rm) {
                    return Ok(RunAction::CombatResolveChoice { option: None });
                }
                if phase == "card_reward" && rmx::card_offer_choose_screen(rm) {
                    return Ok(RunAction::PickCardReward { idx: None });
                }
                err("skip_select not valid here")
            }
            "select_card_reward" => Ok(RunAction::PickCardReward {
                idx: Some(usize_of(arg_i64(args, "card_index")?)),
            }),
            "skip_card_reward" => Ok(RunAction::PickCardReward { idx: None }),
            "select_bundle" => Ok(RunAction::PickCardReward {
                idx: Some(usize_of(arg_i64(args, "bundle_index")?)),
            }),
            "buy_card" => Ok(RunAction::ShopBuyCard {
                idx: usize_of(arg_i64(args, "card_index")?),
            }),
            "buy_relic" => Ok(RunAction::ShopBuyRelic {
                idx: usize_of(arg_i64(args, "relic_index")?),
            }),
            "buy_potion" => Ok(RunAction::ShopBuyPotion {
                idx: usize_of(arg_i64(args, "potion_index")?),
            }),
            "choose_option" => {
                let idx = arg_i64(args, "option_index")?;
                if phase == "event" {
                    return Ok(RunAction::EventChoice { idx: usize_of(idx) });
                }
                if phase == "boss_relic" {
                    return Ok(RunAction::PickBossRelic { idx: usize_of(idx) });
                }
                if idx >= 0 && usize_of(idx) < self.option_actions.len() {
                    return Ok(self.option_actions[usize_of(idx)]);
                }
                err(format!("choose_option {idx} not valid in phase {phase}"))
            }
            "leave_room" => {
                if phase == "shop" {
                    return Ok(RunAction::LeaveShop);
                }
                self.first_action(phase)
            }
            "proceed" => self.first_action(phase),
            other => err(format!(
                "action {other:?} not supported by the rust backend"
            )),
        }
    }

    /// The single advancing action for a continue-style screen.
    fn first_action(&self, phase: &str) -> Result<RunAction, GameError> {
        let acts = rmx::available_actions(self.rm()?);
        acts.first()
            .copied()
            .ok_or_else(|| GameError(format!("no available action in phase {phase}")))
    }

    // ── state serialization ────────────────────────────────────────────

    /// `_room_snapshot`: gold / owned relics / current room.
    fn room_snapshot(&self) -> RoomSnapshot {
        let rm = self.rm.as_ref().expect("room_snapshot needs a run");
        let room = rmx::current_room(rm);
        RoomSnapshot {
            gold: rmx::gold(rm),
            relics: rmx::relics(rm),
            room: if room.is_null() { None } else { Some(room) },
        }
    }

    /// `_serialize`: the C#-shaped state dict plus the room-reveal panel,
    /// the resolved `act1` / `act_name`, and the background variant roll.
    fn serialize(&mut self) -> Value {
        let mut state = self.serialize_state();
        let m = state.as_object_mut().expect("state dict");
        let rm = self.rm.as_ref().expect("serialize needs a run");
        let act = rm.run_state.act_index;
        if let Some(reveal) = treasure_reveal(self.pre_action.as_ref(), &self.room_snapshot(), act)
        {
            m.insert("room_reveal".into(), reveal);
        }
        let act1 = rmx::act1_str(rm);
        m.insert("act1".into(), json!(act1));
        m.insert(
            "act_name".into(),
            match combat_view::act_name(act, act1) {
                Some(n) => json!(n),
                None => Value::Null,
            },
        );
        m.insert("bg_seed".into(), seed_json(rmx::run_seed(rm)));
        // GH #801: only a non-default build is named, so every v0.107.1
        // state (and golden) keeps its bytes. Readers treat absence as
        // v0.107.1.
        if !rm.game_version.is_default() {
            m.insert("game_version".into(), json!(rm.game_version.as_str()));
        }
        if let Some((col, row)) = rmx::map_coord(rm) {
            m.insert("bg_col".into(), json!(col));
            m.insert("bg_row".into(), json!(row));
        }
        state
    }

    /// `_serialize_state`: dispatch on the phase (or a pending deck
    /// mutation preempting it) into one `fill_*`.
    fn serialize_state(&mut self) -> Value {
        let rm = self.rm.as_ref().expect("serialize needs a run");
        let phase = rmx::phase_str(rm);
        let decision = phase_to_decision(phase);
        self.option_actions.clear();

        // An interactive deck mutation the engine is waiting on preempts the
        // phase (`_PREEMPTS`, in order).
        if rmx::pending_deck_removals(rm) > 0 {
            let mut state = self.base_state("card_select");
            self.fill_card_removal(&mut state);
            return state;
        }
        if rmx::pending_deck_upgrades(rm) > 0 {
            let mut state = self.base_state("card_select");
            self.fill_card_upgrade(&mut state);
            return state;
        }
        if rmx::pending_deck_transforms(rm) > 0 {
            let mut state = self.base_state("card_select");
            self.fill_card_transform(&mut state);
            return state;
        }
        if rmx::pending_deck_enchants(rm) > 0 {
            let mut state = self.base_state("card_select");
            self.fill_card_enchant(&mut state);
            return state;
        }
        if rmx::pending_card_grants(rm) > 0 {
            let mut state = self.base_state("card_select");
            self.fill_card_grant(&mut state);
            return state;
        }

        let mut state = self.base_state(decision);
        if decision == "combat_play" {
            self.fill_combat(&mut state);
        } else if phase == "map_choice" {
            self.fill_map(&mut state);
        } else if phase == "card_reward" {
            self.fill_card_reward(&mut state);
        } else if phase == "shop" {
            self.fill_shop(&mut state);
        } else if phase == "rest_site" {
            self.fill_rest(&mut state);
        } else if phase == "event" {
            self.fill_event(&mut state);
        } else if decision == "game_over" {
            let won = self.rm.as_ref().expect("run").run_state.player_won;
            state
                .as_object_mut()
                .expect("state dict")
                .insert("victory".into(), json!(won));
        } else if phase == "boss_relic" {
            self.fill_boss_relic(&mut state);
        } else {
            // treasure / combat-relic reward → generic option list.
            self.fill_generic(&mut state, phase);
        }
        state
    }

    // ── per-decision builders ──────────────────────────────────────────

    /// `_base_state`: the run-wide part of every decision view.
    fn base_state(&self, decision: &str) -> Value {
        let rm = self.rm.as_ref().expect("run");
        json!({
            "decision": decision,
            "player": self.player(),
            "floor": rmx::total_floor(rm),
            "act": rm.run_state.act_index,
            "ascension": rm.run_state.ascension_level,
        })
    }

    /// `_player`.
    fn player(&self) -> Value {
        let rm = self.rm.as_ref().expect("run");
        let relics: Vec<Value> = rmx::relics(rm).iter().map(|r| owned_entry(r)).collect();
        let potions: Vec<Value> = rmx::potions(rm).iter().map(|p| owned_entry(p)).collect();
        let deck: Vec<Value> = rmx::master_deck_detailed(rm)
            .iter()
            .enumerate()
            .map(|(i, (cid, up))| deck_card(i as i64, cid, *up))
            .collect();
        json!({
            "name": self.character,
            "hp": rm.run_state.current_hp,
            "max_hp": rm.run_state.max_hp,
            "block": 0,
            "gold": rmx::gold(rm),
            "relics": relics,
            "potions": potions,
            "potion_slots": rmx::max_potion_slots(rm),
            "deck": deck,
        })
    }

    /// `_fill_combat`.
    fn fill_combat(&self, state: &mut Value) {
        let rm = self.rm.as_ref().expect("run");
        let Some(cs) = rm.combat.as_ref() else {
            return;
        };
        let legal: Vec<Value> = rmx::available_actions(rm)
            .iter()
            .map(rmx::action_value)
            .collect();
        let view = combat_view::build_combat_view(
            &rmx::wire(cs),
            &self.character,
            &legal,
            &rmx::damage_preview(cs),
            &rmx::block_preview(cs),
        );
        let m = state.as_object_mut().expect("state dict");
        // Copy the LIVE combat HP/block/max_hp onto the player up front —
        // even when a pending choice is open.
        if let Some(player) = m.get_mut("player").and_then(Value::as_object_mut) {
            for k in ["hp", "max_hp", "block"] {
                if let Some(v) = view.get("player").and_then(|p| p.get(k)) {
                    player.insert(k.into(), v.clone());
                }
            }
        }
        // An in-combat pending choice is still a combat state: carry the live
        // enemies, player powers and orb belt across the choice boundary.
        m.insert(
            "enemies".into(),
            view.get("enemies").cloned().unwrap_or_else(|| json!([])),
        );
        m.insert(
            "player_powers".into(),
            view.get("player_powers")
                .cloned()
                .unwrap_or_else(|| json!([])),
        );
        if let Some(orbs) = view.get("orbs") {
            m.insert("orbs".into(), orbs.clone());
            m.insert(
                "orb_slots".into(),
                view.get("orb_slots").cloned().unwrap_or(json!(0)),
            );
        }
        if let Some(pc) = rmx::pending_choice(cs) {
            fill_combat_choice(state, &pc);
            return;
        }
        if let Some(osty) = view.get("osty") {
            m.insert("osty".into(), osty.clone());
        }
        for k in [
            "round",
            "energy",
            "max_energy",
            "hand",
            "enemies",
            "player_powers",
            "draw_pile_count",
            "discard_pile_count",
            "exhaust_pile_count",
            "draw_pile",
            "discard_pile",
            "exhaust_pile",
        ] {
            m.insert(k.into(), view.get(k).cloned().unwrap_or(Value::Null));
        }
    }

    /// `_fill_map`: offered nodes from the current node's graph children.
    fn fill_map(&self, state: &mut Value) {
        let rm = self.rm.as_ref().expect("run");
        let move_cols: Vec<i64> = rmx::available_actions(rm)
            .iter()
            .filter(|a| rmx::action_kind(a) == "move")
            .filter_map(|a| rmx::action_int(a, "col"))
            .collect();
        let full = rmx::full_map(rm);
        let empty = Vec::new();
        let rows = full.get("rows").and_then(Value::as_array).unwrap_or(&empty);
        let node_at = |col: i64, row: i64| -> Option<&Value> {
            rows.iter()
                .flat_map(|r| r.as_array().into_iter().flatten())
                .find(|n| {
                    n.get("col").and_then(Value::as_i64) == Some(col)
                        && n.get("row").and_then(Value::as_i64) == Some(row)
                })
        };
        let mut current = rmx::map_coord(rm).and_then(|(c, r)| node_at(i64::from(c), i64::from(r)));
        if current.is_none() {
            current = rows
                .first()
                .and_then(|r| r.as_array())
                .and_then(|r| r.first());
        }
        let children: Vec<&Value> = current
            .and_then(|c| c.get("children"))
            .and_then(Value::as_array)
            .map(|c| c.iter().collect())
            .unwrap_or_default();
        let child_by_col = |col: i64| -> Option<&Value> {
            // Python builds a dict col -> node, so the LAST child with a given
            // column wins.
            let mut found = None;
            for c in &children {
                if c.get("col").and_then(Value::as_i64) == Some(col) {
                    let row = c.get("row").and_then(Value::as_i64).unwrap_or(-1);
                    found = Some(node_at(col, row).unwrap_or(c));
                }
            }
            found
        };
        let choices: Vec<Value> = move_cols
            .iter()
            .map(|&c| match child_by_col(c) {
                Some(node) => json!({
                    "col": c,
                    "row": node.get("row").cloned().unwrap_or(Value::Null),
                    "type": node.get("type").cloned().unwrap_or(Value::Null),
                }),
                None => json!({"col": c, "row": Value::Null, "type": Value::Null}),
            })
            .collect();
        state
            .as_object_mut()
            .expect("state dict")
            .insert("choices".into(), Value::Array(choices));
    }

    fn set_select(
        state: &mut Value,
        cards: Vec<Value>,
        min: i64,
        max: i64,
        can_skip: bool,
        context: &str,
    ) {
        let m = state.as_object_mut().expect("state dict");
        m.insert("cards".into(), Value::Array(cards));
        m.insert("min_select".into(), json!(min));
        m.insert("max_select".into(), json!(max));
        m.insert("can_skip".into(), json!(can_skip));
        m.insert("context".into(), json!(context));
    }

    /// Python `max(1, min(count, len(cards)))`.
    fn clamp_count(count: u32, cards: usize) -> i64 {
        let count = i64::from(if count == 0 { 1 } else { count });
        count.min(cards as i64).max(1)
    }

    /// `_fill_card_removal`: single pick-N deck card_select over the
    /// engine's removal offer order.
    fn fill_card_removal(&self, state: &mut Value) {
        let rm = self.rm.as_ref().expect("run");
        let cards: Vec<Value> = rmx::card_removal_options(rm)
            .iter()
            .enumerate()
            .map(|(i, (cid, up))| deck_card(i as i64, cid, *up))
            .collect();
        let count = Self::clamp_count(rmx::pending_deck_removals(rm), cards.len());
        Self::set_select(state, cards, count, count, false, "remove");
    }

    /// `_fill_card_transform`.
    fn fill_card_transform(&self, state: &mut Value) {
        let rm = self.rm.as_ref().expect("run");
        let cards = self.deck_subset_cards(&rmx::transform_select_deck_indices(rm));
        let count = Self::clamp_count(rmx::pending_deck_transforms(rm), cards.len());
        Self::set_select(state, cards, count, count, false, "transform");
    }

    /// `_fill_card_enchant`.
    fn fill_card_enchant(&self, state: &mut Value) {
        let rm = self.rm.as_ref().expect("run");
        let cards = self.deck_subset_cards(&rmx::enchant_select_deck_indices(rm));
        let count = Self::clamp_count(rmx::pending_deck_enchants(rm), cards.len());
        Self::set_select(state, cards, count, count, false, "enchant");
    }

    /// `_fill_card_grant`.
    fn fill_card_grant(&self, state: &mut Value) {
        let rm = self.rm.as_ref().expect("run");
        let cards: Vec<Value> = rmx::card_grant_options(rm)
            .iter()
            .enumerate()
            .map(|(i, (cid, up))| deck_card(i as i64, cid, *up))
            .collect();
        let count = Self::clamp_count(rmx::pending_card_grants(rm), cards.len());
        Self::set_select(state, cards, count, count, false, "grant");
    }

    /// `_deck_subset_cards`: master-deck cards at `deck_indices`, indexed by
    /// SUBSET position.
    fn deck_subset_cards(&self, deck_indices: &[usize]) -> Vec<Value> {
        let rm = self.rm.as_ref().expect("run");
        let deck = rmx::master_deck_detailed(rm);
        deck_indices
            .iter()
            .enumerate()
            .filter_map(|(subset_idx, &deck_idx)| {
                deck.get(deck_idx)
                    .map(|(cid, up)| deck_card(subset_idx as i64, cid, *up))
            })
            .collect()
    }

    /// `_fill_card_upgrade`: the UPGRADABLE deck cards, indexed by their
    /// position in that subset.
    fn fill_card_upgrade(&self, state: &mut Value) {
        let rm = self.rm.as_ref().expect("run");
        let deck = rmx::master_deck_detailed(rm);
        let eligible = rmx::upgrade_select_deck_indices(rm);
        let mut cards = Vec::new();
        let mut subset_idx = 0_i64;
        for (deck_idx, (cid, _up)) in deck.iter().enumerate() {
            if !eligible.contains(&deck_idx) {
                continue;
            }
            cards.push(deck_card(subset_idx, cid, false));
            subset_idx += 1;
        }
        Self::set_select(state, cards, 1, 1, false, "upgrade");
    }

    /// `_fill_card_reward`.
    fn fill_card_reward(&mut self, state: &mut Value) {
        let rm = self.rm.as_ref().expect("run");
        if let Some(bundles) = rmx::card_reward_bundles(rm).filter(|b| !b.is_empty()) {
            Self::fill_bundle_select(state, &bundles);
            return;
        }
        let actions = rmx::available_actions(rm);
        let has_cards = actions
            .iter()
            .any(|a| rmx::action_kind(a) == "pick_card_reward");
        if !has_cards {
            // Relic / sacrifice reward — no card list; render as options.
            state
                .as_object_mut()
                .expect("state dict")
                .insert("decision".into(), json!("event_choice"));
            self.fill_relic_reward(state);
            return;
        }
        let cards: Vec<Value> = rmx::card_reward_option_ids(rm)
            .iter()
            .enumerate()
            .map(|(i, raw)| offer_card(Some(i as i64), raw))
            .collect();
        let can_skip = actions.iter().any(|a| {
            rmx::action_kind(a) == "pick_card_reward"
                && rmx::action_value(a).get("idx").is_none_or(Value::is_null)
        });
        let m = state.as_object_mut().expect("state dict");
        m.insert("cards".into(), Value::Array(cards));
        m.insert("can_skip".into(), json!(can_skip));
        // A choose screen is a `card_select` in the C# vocabulary. GH #34.
        if rmx::card_offer_choose_screen(rm) {
            m.insert("decision".into(), json!("card_select"));
            m.insert("min_select".into(), json!(0));
            m.insert("max_select".into(), json!(1));
        }
    }

    /// `_fill_bundle_select`: Scroll Boxes packs.
    fn fill_bundle_select(state: &mut Value, bundles_raw: &[Vec<String>]) {
        let bundles: Vec<Value> = bundles_raw
            .iter()
            .enumerate()
            .map(|(i, pack)| {
                let cards: Vec<Value> = pack.iter().map(|raw| offer_card(None, raw)).collect();
                json!({"index": i, "cards": cards})
            })
            .collect();
        let m = state.as_object_mut().expect("state dict");
        m.insert("decision".into(), json!("bundle_select"));
        m.insert("bundles".into(), Value::Array(bundles));
    }

    /// `_fill_relic_reward`: name the relic on the take/skip screen.
    fn fill_relic_reward(&mut self, state: &mut Value) {
        let rm = self.rm.as_ref().expect("run");
        let actions = rmx::available_actions(rm);
        let ids = rmx::relic_reward_ids(rm);
        let head = ids.first().map(|r| loc_id(r));
        let options: Vec<Value> = actions
            .iter()
            .enumerate()
            .map(
                |(i, a)| match (rmx::action_kind(a).as_str(), head.as_deref()) {
                    ("pick_relic_reward", Some(head)) => json!({
                        "index": i,
                        "title": format!("{head}.title"),
                        "text_key": head,
                        "is_locked": false,
                    }),
                    ("skip_relic", _) => json!({"index": i, "title": "Skip", "is_locked": false}),
                    _ => json!({"index": i, "title": action_label(a), "is_locked": false}),
                },
            )
            .collect();
        self.option_actions = actions;
        state
            .as_object_mut()
            .expect("state dict")
            .insert("options".into(), Value::Array(options));
    }

    /// `_fill_shop`: the merchant inventory with `is_stocked` (buyable right
    /// now) and `is_available` (still holds an item).
    fn fill_shop(&self, state: &mut Value) {
        let rm = self.rm.as_ref().expect("run");
        let inv = rmx::shop_inventory(rm);
        let gold = state
            .get("player")
            .and_then(|p| p.get("gold"))
            .and_then(Value::as_i64)
            .unwrap_or(0);
        let empty = Vec::new();
        let affordable = |e: &Value| -> bool {
            let price = e.get("price").and_then(Value::as_i64);
            combat_view::truthy(e.get("available").unwrap_or(&Value::Null))
                && price.is_some_and(|p| p <= gold)
        };
        let id_of = |e: &Value| -> Option<String> {
            e.get("id")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .map(loc_id)
        };
        let cards: Vec<Value> = inv
            .get("cards")
            .and_then(Value::as_array)
            .unwrap_or(&empty)
            .iter()
            .map(|e| {
                let cid = id_of(e).unwrap_or_else(|| {
                    format!(
                        "card_{}",
                        combat_view::py_str(e.get("idx").unwrap_or(&Value::Null))
                    )
                });
                json!({
                    "index": e.get("idx").cloned().unwrap_or(Value::Null),
                    "id": cid,
                    "name": format!("{cid}.title"),
                    "description": format!("{cid}.description"),
                    "type": "card",
                    "cost": e.get("price").cloned().unwrap_or(Value::Null),
                    "on_sale": combat_view::truthy(e.get("on_sale").unwrap_or(&Value::Null)),
                    "is_stocked": affordable(e),
                    "is_available": combat_view::truthy(e.get("available").unwrap_or(&Value::Null)),
                })
            })
            .collect();
        let relics: Vec<Value> = inv
            .get("relics")
            .and_then(Value::as_array)
            .unwrap_or(&empty)
            .iter()
            .map(|e| {
                let rid = id_of(e);
                json!({
                    "index": e.get("idx").cloned().unwrap_or(Value::Null),
                    "id": rid,
                    "name": rid.as_ref().map_or_else(|| "Relic".to_string(), |r| format!("{r}.title")),
                    "cost": e.get("price").cloned().unwrap_or(Value::Null),
                    "is_stocked": affordable(e),
                    "is_available": combat_view::truthy(e.get("available").unwrap_or(&Value::Null)),
                })
            })
            .collect();
        let potions: Vec<Value> = inv
            .get("potions")
            .and_then(Value::as_array)
            .unwrap_or(&empty)
            .iter()
            .map(|e| {
                let pid = id_of(e);
                json!({
                    "index": e.get("idx").cloned().unwrap_or(Value::Null),
                    "id": pid,
                    "name": pid.as_ref().map_or_else(|| "Potion".to_string(), |p| format!("{p}.title")),
                    "cost": e.get("price").cloned().unwrap_or(Value::Null),
                    "is_stocked": affordable(e),
                    "is_available": combat_view::truthy(e.get("available").unwrap_or(&Value::Null)),
                })
            })
            .collect();
        let removal_offered = rmx::available_actions(rm)
            .iter()
            .any(|a| rmx::action_kind(a) == "shop_remove_card");
        let m = state.as_object_mut().expect("state dict");
        m.insert("cards".into(), Value::Array(cards));
        m.insert("relics".into(), Value::Array(relics));
        m.insert("potions".into(), Value::Array(potions));
        if removal_offered {
            let cost = inv.get("removal_cost").and_then(Value::as_i64).unwrap_or(0);
            m.insert("card_removal_cost".into(), json!(cost));
        }
    }

    /// `_fill_rest`: rest-site options by engine `option_id`, or the Smith
    /// sub-state as a card_select.
    fn fill_rest(&mut self, state: &mut Value) {
        let rm = self.rm.as_ref().expect("run");
        let acts = rmx::available_actions(rm);
        let rest_acts: Vec<RunAction> = acts
            .iter()
            .copied()
            .filter(|a| rmx::action_kind(a) == "rest_option")
            .collect();
        let smith_acts: Vec<RunAction> = acts
            .iter()
            .copied()
            .filter(|a| rmx::action_kind(a) == "smith_card")
            .collect();
        let smith_ids = rmx::smith_options(rm);
        if !smith_acts.is_empty() && !smith_ids.is_empty() {
            Self::fill_smith(state, &smith_acts, &smith_ids);
            return;
        }
        self.option_actions = if rest_acts.is_empty() {
            acts
        } else {
            rest_acts
        };
        let engine_opts = rmx::rest_site_options(rm);
        let options: Vec<Value> = self
            .option_actions
            .iter()
            .enumerate()
            .map(|(i, a)| {
                if rmx::action_kind(a) == "rest_option" {
                    let idx = rmx::action_int(a, "idx");
                    let pair = idx
                        .filter(|&i| i >= 0)
                        .and_then(|i| engine_opts.get(usize_of(i)));
                    match pair {
                        None => {
                            let option_id = rest_label(idx.unwrap_or(-1));
                            json!({"index": i, "option_id": option_id, "name": option_id, "is_enabled": true})
                        }
                        Some((option_id, enabled)) => json!({
                            "index": i,
                            "option_id": option_id,
                            "name": format!("rest_site.OPTION_{option_id}.name"),
                            "is_enabled": enabled,
                        }),
                    }
                } else {
                    let option_id = action_label(a);
                    json!({"index": i, "option_id": option_id, "name": option_id, "is_enabled": true})
                }
            })
            .collect();
        state
            .as_object_mut()
            .expect("state dict")
            .insert("options".into(), Value::Array(options));
    }

    /// `_fill_smith`: the Smith upgrade sub-state as a deck card_select.
    fn fill_smith(state: &mut Value, smith_acts: &[RunAction], smith_ids: &[String]) {
        let mut cards = Vec::new();
        for a in smith_acts {
            let Some(idx) = rmx::action_int(a, "idx") else {
                continue;
            };
            if idx < 0 || usize_of(idx) >= smith_ids.len() {
                continue;
            }
            cards.push(deck_card(idx, &smith_ids[usize_of(idx)], false));
        }
        state
            .as_object_mut()
            .expect("state dict")
            .insert("decision".into(), json!("card_select"));
        Self::set_select(state, cards, 1, 1, false, "upgrade");
    }

    /// `_fill_event`: materialize the room's event up front and render its
    /// options as loc keys with the engine's shorthand as fallbacks.
    fn fill_event(&mut self, state: &mut Value) {
        let rm = self.rm.as_mut().expect("run");
        let event_pyname = rmx::materialize_event(rm);
        let rm = self.rm.as_ref().expect("run");
        let infos = rmx::event_option_infos(rm);
        let empty = Vec::new();
        let infos = infos.as_array().unwrap_or(&empty);
        let mut options = Vec::new();
        for a in rmx::available_actions(rm) {
            if rmx::action_kind(&a) != "event_choice" {
                continue;
            }
            let idx = rmx::action_int(&a, "idx").unwrap_or(-1);
            let info = if idx >= 0 {
                infos.get(usize_of(idx))
            } else {
                None
            };
            let str_of = |key: &str| -> String {
                info.and_then(|i| i.get(key))
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string()
            };
            let option_id = str_of("option_id");
            let mut label = str_of("label");
            if label.is_empty() {
                label = "Continue".to_string();
            }
            // Neow relic offers come back as relic ids (all-caps).
            let looks_like_id = py_isupper(&label) && label.contains('_');
            let (mut title, text_key) = if looks_like_id {
                (format!("{}.title", loc_id(&label)), label.clone())
            } else {
                (label.clone(), String::new())
            };
            let loc_key = str_of("loc_key");
            let description = if !loc_key.is_empty() && !looks_like_id {
                title = format!("{loc_key}.title");
                format!("{loc_key}.description")
            } else if looks_like_id {
                String::new()
            } else {
                str_of("description")
            };
            let dict_of = |key: &str| -> Value {
                match info.and_then(|i| i.get(key)) {
                    Some(Value::Object(m)) => Value::Object(m.clone()),
                    _ => Value::Object(Map::new()),
                }
            };
            let enabled = info
                .and_then(|i| i.get("enabled"))
                .is_none_or(combat_view::truthy);
            options.push(json!({
                "index": idx,
                "title": if title.is_empty() { format!("Option {idx}") } else { title },
                "description": description,
                "option_id": option_id,
                "text_key": text_key,
                "vars": dict_of("vars"),
                "title_vars": dict_of("title_vars"),
                "fallback_description": str_of("description"),
                "fallback_title": label,
                "is_locked": info.is_some() && !enabled,
            }));
        }
        let m = state.as_object_mut().expect("state dict");
        m.insert("options".into(), Value::Array(options));
        match event_pyname {
            Some(name) => {
                let stem = event_loc_stem(&name);
                m.insert("event_name".into(), json!(format!("{stem}.title")));
                m.insert(
                    "description".into(),
                    json!(format!("{stem}.pages.INITIAL.description")),
                );
            }
            None => {
                m.insert("event_name".into(), json!(""));
            }
        }
    }

    /// `_fill_boss_relic`: the three named boss relics.
    fn fill_boss_relic(&mut self, state: &mut Value) {
        let rm = self.rm.as_ref().expect("run");
        let actions = rmx::available_actions(rm);
        let ids = rmx::boss_relic_options(rm);
        let options: Vec<Value> = actions
            .iter()
            .enumerate()
            .map(|(i, a)| {
                if rmx::action_kind(a) == "pick_boss_relic" {
                    let rid = rmx::action_int(a, "idx")
                        .filter(|&i| i >= 0)
                        .and_then(|i| ids.get(usize_of(i)))
                        .filter(|r| !r.is_empty());
                    if let Some(rid) = rid {
                        let loc = loc_id(rid);
                        return json!({
                            "index": i,
                            "title": format!("{loc}.title"),
                            "text_key": loc,
                            "is_locked": false,
                        });
                    }
                }
                json!({"index": i, "title": action_label(a), "is_locked": false})
            })
            .collect();
        self.option_actions = actions;
        let m = state.as_object_mut().expect("state dict");
        m.insert("decision".into(), json!("event_choice"));
        m.insert("options".into(), Value::Array(options));
        m.insert("event_name".into(), json!(""));
    }

    /// `_fill_generic`: one option per available action (treasure / relic
    /// rewards), never-stuck fallback.
    fn fill_generic(&mut self, state: &mut Value, phase: &str) {
        let rm = self.rm.as_ref().expect("run");
        self.option_actions = rmx::available_actions(rm);
        let options: Vec<Value> = self
            .option_actions
            .iter()
            .enumerate()
            .map(|(i, a)| json!({"index": i, "title": action_label(a), "is_locked": false}))
            .collect();
        let m = state.as_object_mut().expect("state dict");
        m.insert("decision".into(), json!("event_choice"));
        m.insert("options".into(), Value::Array(options));
        m.insert(
            "event_name".into(),
            json!(if phase == "treasure" { "Treasure" } else { "" }),
        );
    }
}

/// `fill_combat_choice`: render an in-combat pending choice (Headbutt,
/// Warcry, …) as a `card_select` on `state`.
pub fn fill_combat_choice(state: &mut Value, pc: &Value) {
    let empty = Vec::new();
    let cards: Vec<Value> = pc
        .get("cards")
        .and_then(Value::as_array)
        .unwrap_or(&empty)
        .iter()
        .enumerate()
        .map(|(i, opt)| {
            deck_card(
                i as i64,
                opt.get("id").and_then(Value::as_str).unwrap_or(""),
                combat_view::truthy(opt.get("upgraded").unwrap_or(&Value::Null)),
            )
        })
        .collect();
    let context = pc
        .get("kind")
        .and_then(Value::as_str)
        .filter(|k| !k.is_empty())
        .unwrap_or("combat_choice")
        .to_lowercase();
    let m = state.as_object_mut().expect("state dict");
    m.insert("decision".into(), json!("card_select"));
    m.insert("cards".into(), Value::Array(cards));
    m.insert(
        "min_select".into(),
        pc.get("min_select").cloned().unwrap_or(json!(1)),
    );
    m.insert(
        "max_select".into(),
        pc.get("max_select").cloned().unwrap_or(json!(1)),
    );
    m.insert(
        "can_skip".into(),
        json!(combat_view::truthy(
            pc.get("can_skip").unwrap_or(&Value::Null)
        )),
    );
    m.insert("context".into(), json!(context));
}

/// Install the process-global content dispatch registries (cards, relics,
/// potions) — what `sts2_rs`'s module import (`sts2-py/src/lib.rs`) and the
/// CLI host do before any run exists. Idempotent. Without it the run loop
/// still moves, but every content hook silently no-ops: a Neow relic pickup
/// lands in the deck without its `after_obtained` (Lead Paperweight's
/// choose screen, Precise Scissors' removal, …) and cards played in combat
/// do nothing. The golden parity replay caught exactly that — the whole
/// G0LDEN0001 family went `map_select` where the oracle showed a
/// `card_select` — so every run constructor in this crate goes through here.
pub fn ensure_content_installed() {
    sts2_content::cards::dispatch::install_default_registry();
    sts2_content::relics::dispatch::ensure_installed();
    sts2_content::potions::dispatch::ensure_installed();
}

/// The interactive workbench's run constructor — `RustGame.start`'s
/// `sts2_rs.RunManager(...)` call, statement for statement
/// (`sts2-py/src/run_manager.rs::PyRunManager::new` with `string_seed`,
/// `neow_start=True`, every interactive deck-mutation flag,
/// `direct_event_relic_grant=True`, `act1`).
pub fn build_run_manager(
    version: GameVersion,
    cid: CharacterId,
    seed: &str,
    ascension: u8,
    act1: &str,
) -> Result<RunManager, GameError> {
    use sts2_content::run_callbacks::initialize_run_setup;
    use sts2_core::act_selection::ActVariant;
    let rm = new_string_seed_run(version, cid, seed, ascension);
    let mut rm = rm.with_neow_start();
    // Present on-obtain card removals / upgrades / transforms as player
    // card_selects, matching C#; grant the chosen Neow relic directly.
    rm.run_state.interactive_card_removal = true;
    rm.run_state.interactive_card_upgrade = true;
    rm.run_state.interactive_card_transform = true;
    rm.run_state.interactive_potion_swap = false;
    rm.run_state.direct_event_relic_grant = true;
    // GH #544: which of the two Act-1 models this run walks — set BEFORE
    // `initialize_run_setup`, which pre-generates every act's rooms.
    let mut rm = match act1 {
        "overgrowth" => rm.with_act1(ActVariant::Overgrowth),
        "underdocks" => rm.with_act1(ActVariant::Underdocks),
        "roll" => rm.with_rolled_act1(),
        other => {
            return err(format!(
                "unknown act1 {other:?} (expected \"overgrowth\", \"underdocks\" or \"roll\")"
            ))
        }
    };
    initialize_run_setup(&mut rm);
    Ok(rm)
}

/// The PPO env's run constructor — `sts2_rs.RunManager(seed=int, …)` with
/// `neow_start=True`, the three interactive deck-mutation flags,
/// `direct_event_relic_grant=True`, `interactive_potion_swap=False` and the
/// default (Overgrowth) Act 1, statement for statement
/// (`sts2-py/src/run_manager.rs::PyRunManager::new` without `string_seed`).
#[must_use]
/// The run constructor for a live-state import: the interactive workbench's
/// build (every C#-faithful deck-mutation flag, direct Neow relic grant) but
/// with NO Neow start — the blessing was taken floors ago and the pasted deck
/// already reflects it — and the rolled Act-1 model (GH #544), so the act maps
/// match the live run. Mirrors `RustGame.start_imported`'s `RunManager(...)`.
fn build_import_run_manager(
    version: GameVersion,
    cid: CharacterId,
    seed: &str,
    ascension: u8,
) -> RunManager {
    use sts2_content::run_callbacks::initialize_run_setup;
    // No `with_neow_start`: not a fresh run.
    let mut rm = new_string_seed_run(version, cid, seed, ascension);
    rm.run_state.interactive_card_removal = true;
    rm.run_state.interactive_card_upgrade = true;
    rm.run_state.interactive_card_transform = true;
    rm.run_state.interactive_potion_swap = false;
    rm.run_state.direct_event_relic_grant = true;
    // The game rolled this Act-1 for the player; `with_rolled_act1` rolls the
    // same, and must run BEFORE `initialize_run_setup` pre-generates rooms.
    let mut rm = rm.with_rolled_act1();
    initialize_run_setup(&mut rm);
    rm
}

/// The text-seed run every interactive constructor starts from, in game
/// build `version` — `PyRunManager::new` with `string_seed`
/// (`sts2-py/src/run_manager.rs`): hash the canonical seed with the build's
/// `RunRngSet` rule (XxHash64 in v0.111.0), build through the character /
/// ascension factory, then re-target the build and its full 64-bit master
/// seed BEFORE anything (Neow, act 1, run setup) rolls. Both re-targets are
/// no-ops at v0.107.1, so the default build's run is unchanged.
fn new_string_seed_run(
    version: GameVersion,
    cid: CharacterId,
    seed: &str,
    ascension: u8,
) -> RunManager {
    use sts2_content::run_callbacks::{
        new_run_manager_for_character_from_string_seed,
        new_run_manager_for_character_with_ascension,
    };
    ensure_content_installed();
    let canonical = sts2_core::canonicalize_seed(seed);
    let master = version.run_seed_hash(&canonical);
    #[allow(clippy::cast_possible_truncation)]
    let effective_seed = master as i32;
    let rm = if ascension > 0 {
        new_run_manager_for_character_with_ascension(effective_seed, cid, ascension)
    } else {
        new_run_manager_for_character_from_string_seed(seed, cid)
    };
    rm.with_game_version(version).with_run_seed(master)
}

/// Grant a relic to the run by id, mirroring `PyRunManager::force_add_relic`:
/// push, stash `last_obtained_relic`, fire `after_obtained` eagerly (so
/// Whetstone / War Paint apply their follow-ups), clear the stash. Idempotent
/// — a relic already owned is a no-op.
fn graft_add_relic(rm: &mut RunManager, relic_id: &str) {
    let stripped = relic_id.strip_prefix("RELIC.").unwrap_or(relic_id);
    let Some(rid) = sts2_content::names::relic_id_from_str(stripped) else {
        return;
    };
    let rs = &mut rm.run_state;
    if rs.relics.contains(&rid) {
        return;
    }
    rs.relics.push(rid);
    rs.last_obtained_relic = Some(rid);
    let _ = sts2_core::run_effects::dispatch_after_obtained(rs);
    rm.run_state.last_obtained_relic = None;
}

/// Append one master-deck card by id, mirroring `PyRunManager::force_add_card`:
/// build via the registered factory so spec stats are populated, falling back
/// to a bare POD only if no spec is known.
fn graft_add_card(rm: &mut RunManager, card_id: &str) {
    let Some(cid) = sts2_content::names::card_id_from_str(card_id) else {
        return;
    };
    let card = sts2_core::lookup_card_factory()
        .and_then(|f| f(cid, false))
        .unwrap_or_else(|| {
            let mut c = sts2_core::cards::Card::pod_defaults();
            c.card_id = cid;
            c
        });
    rm.run_state.master_deck.push(card);
}

/// Upgrade the first un-upgraded master-deck copy of `card_id`
/// (`PyRunManager::force_upgrade_card`). `false` when no such copy exists.
fn graft_upgrade_card(rm: &mut RunManager, card_id: &str) -> bool {
    let Some(cid) = sts2_content::names::card_id_from_str(card_id) else {
        return false;
    };
    for c in &mut rm.run_state.master_deck {
        if c.card_id == cid && !c.upgraded {
            c.upgraded = true;
            return true;
        }
    }
    false
}

/// Replace the run's belt with `potions` (`PyRunManager::debug_set_potion_state`
/// with no pending swaps). Unknown names are dropped silently — the parser
/// already warned about them.
fn graft_set_potions(rm: &mut RunManager, potions: &[String]) {
    let rs = &mut rm.run_state;
    rs.potions.clear();
    rs.pending_potion_swaps.clear();
    for name in potions {
        if let Some(kind) = sts2_content::names::potion_kind_from_str(name) {
            rs.potions.push(kind);
        }
    }
}

/// Fast-forward each named RNG stream to its recorded cursor
/// (`PyRunManager::debug_set_rng_counters`). Returns notes for anything that
/// could not be restored: a foreign seed, a backwards cursor, or a stream this
/// engine does not carry (run-level only — combat streams live on a
/// `CombatState` and are the combat import's business).
fn graft_rng(rm: &mut RunManager, rng: &crate::state_import::RngBlock) -> Vec<String> {
    // The wire's `run_seed` is the hashed master seed (`RunRngSet.Seed`): the
    // u32 reinterpretation of the i32 run seed in v0.107.1 (same contract as
    // the Python restore), the full 64-bit XxHash64 in v0.111.0 (GH #788).
    let ours = rm.run_state.rng.seed;
    if rng.run_seed != ours {
        return vec![format!(
            "RNG cursors not restored — run_seed {} is not this run's seed ({ours}); \
             these cursors belong to a different run",
            rng.run_seed
        )];
    }
    let mut ignored: Vec<String> = Vec::new();
    for (name, target) in &rng.counters {
        let Ok(target) = u32::try_from(*target) else {
            continue;
        };
        let Some(stream) = rm.run_state.rng.stream_mut(name) else {
            ignored.push(name.clone());
            continue;
        };
        let current = stream.counter();
        if target < current {
            return vec![format!(
                "RNG cursors not restored — stream {name:?} is already at {current}, \
                 cannot rewind to {target}"
            )];
        }
        stream.fast_forward_counter(target);
    }
    let run_level: Vec<String> = ignored
        .into_iter()
        .filter(|n| !n.starts_with("combat_"))
        .collect();
    if run_level.is_empty() {
        Vec::new()
    } else {
        vec![format!(
            "RNG streams this engine does not carry: {}",
            run_level.join(", ")
        )]
    }
}

/// Where the engine's generated act map differs from the pasted one
/// (`rust_game._map_disagreements`). The map is a pure function of
/// `(seed, act)`, so on a correct import these agree node for node; a
/// disagreement means every path explored from here is one the real run does
/// not have. Best-effort and non-fatal.
fn map_disagreements(rm: &RunManager, snapshot: &crate::state_import::RunSnapshot) -> Vec<String> {
    let Some(world_nodes) = snapshot
        .map
        .as_ref()
        .and_then(|m| m.get("nodes"))
        .and_then(Value::as_array)
        .filter(|n| !n.is_empty())
    else {
        return Vec::new();
    };
    let full = rmx::full_map(rm);
    let Some(full) = full.as_object() else {
        return Vec::new();
    };
    // (col, row) → node type; BTreeMap per the repo's disallowed-HashMap rule.
    let mut ours: std::collections::BTreeMap<(i64, i64), String> =
        std::collections::BTreeMap::new();
    if let Some(rows) = full.get("rows").and_then(Value::as_array) {
        for row in rows {
            for node in row.as_array().into_iter().flatten() {
                if let Some(o) = node.as_object() {
                    let key = (
                        o.get("col").and_then(Value::as_i64).unwrap_or(i64::MIN),
                        o.get("row").and_then(Value::as_i64).unwrap_or(i64::MIN),
                    );
                    ours.insert(
                        key,
                        o.get("type")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string(),
                    );
                }
            }
        }
    }
    if let Some(boss) = full.get("boss").and_then(Value::as_object) {
        let key = (
            boss.get("col").and_then(Value::as_i64).unwrap_or(i64::MIN),
            boss.get("row").and_then(Value::as_i64).unwrap_or(i64::MIN),
        );
        ours.insert(
            key,
            boss.get("type")
                .and_then(Value::as_str)
                .unwrap_or("Boss")
                .to_string(),
        );
    }
    let (mut mismatched, mut missing) = (0u32, 0u32);
    for node in world_nodes {
        let Some(o) = node.as_object() else { continue };
        let key = (
            o.get("col").and_then(Value::as_i64).unwrap_or(i64::MIN),
            o.get("row").and_then(Value::as_i64).unwrap_or(i64::MIN),
        );
        let theirs = o.get("type").and_then(Value::as_str).unwrap_or("");
        match ours.get(&key) {
            None => missing += 1,
            Some(t) if t != theirs => mismatched += 1,
            Some(_) => {}
        }
    }
    if mismatched == 0 && missing == 0 {
        return Vec::new();
    }
    vec![format!(
        "the act map this engine generates differs from the live one \
         ({mismatched} node(s) of a different type, {missing} it does not have) \
         — paths explored from here are not the ones in your run"
    )]
}

pub fn build_faithful_run_manager(
    version: GameVersion,
    cid: CharacterId,
    seed: i32,
    ascension: u8,
) -> RunManager {
    use sts2_content::run_callbacks::{
        initialize_run_setup, new_run_manager_for_character,
        new_run_manager_for_character_with_ascension,
    };
    use sts2_core::act_selection::ActVariant;
    ensure_content_installed();
    let rm = if ascension > 0 {
        new_run_manager_for_character_with_ascension(seed, cid, ascension)
    } else {
        new_run_manager_for_character(seed, cid)
    };
    // An integer seed is a `uint` master in both builds (`PyRunManager::new`
    // without `string_seed`); the build is re-targeted before Neow.
    let rm = rm
        .with_game_version(version)
        .with_run_seed(u64::from(seed as u32));
    let mut rm = rm.with_neow_start();
    rm.run_state.interactive_card_removal = true;
    rm.run_state.interactive_card_upgrade = true;
    rm.run_state.interactive_card_transform = true;
    rm.run_state.interactive_potion_swap = false;
    rm.run_state.direct_event_relic_grant = true;
    let mut rm = rm.with_act1(ActVariant::Overgrowth);
    initialize_run_setup(&mut rm);
    rm
}

/// A run seed as the browser can read it back exactly (GH #878).
///
/// v0.111.0 run seeds are `ulong`s and `JSON.parse` rounds any number past
/// 2^53, so those go out as a decimal string. Every smaller seed (all of
/// v0.107.1's `uint`s) stays a number, so its bytes and goldens don't change.
/// Python twin: `workbench.seed.json_exact_seed`.
pub(crate) fn seed_json(seed: u64) -> Value {
    const JSON_SAFE_MAX: u64 = (1 << 53) - 1;
    if seed <= JSON_SAFE_MAX {
        json!(seed)
    } else {
        json!(seed.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state_import::parse_pasted_state;

    /// A mod-shape map block built from a real run's act-1 map, standing on
    /// one of its row-1 nodes — the same trick `test_run_import.py` uses to
    /// assert the graft lands on the right node without a live game.
    fn map_block_from(rm: &RunManager, warnings_only: bool) -> (Value, i64, i64) {
        let full = rmx::full_map(rm);
        let rows = full["rows"].as_array().expect("rows");
        let mut nodes = Vec::new();
        let mut here: Option<(i64, i64)> = None;
        for row in rows {
            for n in row.as_array().into_iter().flatten() {
                let col = n["col"].as_i64().unwrap();
                let row_i = n["row"].as_i64().unwrap();
                nodes.push(json!({
                    "col": col, "row": row_i,
                    "type": n["type"].clone(),
                    "children": n.get("children").cloned().unwrap_or(json!([])),
                }));
                if row_i == 1 && here.is_none() {
                    here = Some((col, row_i));
                }
            }
        }
        let (col, row_i) = here.expect("a row-1 node");
        let map = if warnings_only {
            Value::Null
        } else {
            json!({
                "nodes": nodes,
                "current": {"col": col, "row": row_i},
                "visited": [{"col": col, "row": row_i}],
            })
        };
        (map, col, row_i)
    }

    fn import_wire(rng: Value, map: Value) -> Value {
        json!({
            "type": "advice_request",
            "advice_for": "map_select",
            "seed": "TESTSEED42",
            "act": 1,
            "floor": 2,
            "map": map,
            "rng": rng,
            "run_state": {
                "character": "IRONCLAD",
                "ascension": 0,
                "hp": 55,
                "max_hp": 80,
                "gold": 217,
                "potion_slot_count": 3,
                "master_deck": [
                    {"id": "STRIKE_IRONCLAD", "upgrade_level": 0},
                    {"id": "STRIKE_IRONCLAD", "upgrade_level": 1},
                    {"id": "DEFEND_IRONCLAD", "upgrade_level": 0},
                    {"id": "BASH", "upgrade_level": 0},
                ],
                "relics": [{"id": "BURNING_BLOOD"}, {"id": "WHETSTONE"}],
                "potions": [{"slot": 0, "id": "FIRE_POTION"}, {"slot": 1, "empty": true}],
            },
        })
    }

    #[test]
    fn start_imported_grafts_the_deck_relics_potions_hp_and_parks_on_the_map() {
        let source = build_import_run_manager(
            GameVersion::default(),
            CharacterId::Ironclad,
            "TESTSEED42",
            0,
        );
        let (map, col, row) = map_block_from(&source, false);
        let wire = import_wire(Value::Null, map);
        let imported = parse_pasted_state(&wire.to_string()).expect("parse");

        let mut game = Game::new();
        let state = game.start_imported(&imported.run).expect("graft");
        assert_eq!(state["decision"], json!("map_select"));

        let rm = game.native_run().expect("run");
        assert_eq!(rm.run_state.current_hp, 55);
        assert_eq!(rm.run_state.max_hp, 80);
        assert_eq!(rm.run_state.gold, 217);
        // The deck is the pasted one, not the starter, and exactly one Strike
        // is upgraded — Whetstone's on-obtain upgrade must NOT have chewed on
        // the grafted deck (the empty-deck-before-relics ordering).
        let upgraded = rm
            .run_state
            .master_deck
            .iter()
            .filter(|c| c.upgraded)
            .count();
        assert_eq!(rm.run_state.master_deck.len(), 4);
        assert_eq!(upgraded, 1, "only the copy the paste said was upgraded");
        // Placed on the node it stood on, with real moves to make.
        assert_eq!(rmx::map_coord(rm), Some((col as i32, row as i32)));
        assert!(
            !rmx::available_actions(rm).is_empty(),
            "a placed run has somewhere to go"
        );
        assert!(!game
            .import_notes()
            .iter()
            .any(|n| n.contains("does not have")));
    }

    #[test]
    fn start_imported_restores_the_rng_cursors() {
        let source = build_import_run_manager(
            GameVersion::default(),
            CharacterId::Ironclad,
            "TESTSEED42",
            0,
        );
        let (map, _c, _r) = map_block_from(&source, false);
        let seed_wire = i64::try_from(source.run_state.rng.seed).expect("v0.107.1 seed fits");
        let rng = json!({"run_seed": seed_wire, "counters": {"shops": 2, "rewards": 1}});
        let wire = import_wire(rng, map);
        let imported = parse_pasted_state(&wire.to_string()).expect("parse");

        let mut game = Game::new();
        game.start_imported(&imported.run).expect("graft");
        let rm = game.native_run_mut().expect("run");
        assert_eq!(rm.run_state.rng.stream_mut("shops").unwrap().counter(), 2);
        assert_eq!(rm.run_state.rng.stream_mut("rewards").unwrap().counter(), 1);
        assert!(!game
            .import_notes()
            .iter()
            .any(|n| n.contains("not restored")));
    }

    /// GH #801: a v0.111.0 master seed is a full `ulong` — the cursors of a
    /// run whose seed is above `i64::MAX` still restore, and the map the
    /// graft is checked against is the v0.111.0 one.
    #[test]
    fn start_imported_restores_v0111_cursors_with_a_64_bit_seed() {
        let v = GameVersion::V0_111_0;
        // Find a seed whose v0.111.0 master does not fit an i64.
        let seed = (0..10_000)
            .map(|i| format!("BIGSEED{i}"))
            .find(|s| i64::try_from(v.run_seed_hash(&sts2_core::canonicalize_seed(s))).is_err())
            .expect("half of all seeds qualify");
        let source = build_import_run_manager(v, CharacterId::Ironclad, &seed, 0);
        assert_eq!(source.game_version, v);
        assert!(source.run_state.rng.seed > i64::MAX as u64);
        let (map, _c, _r) = map_block_from(&source, false);
        let rng = json!({"run_seed": source.run_state.rng.seed, "counters": {"shops": 2}});
        let mut wire = import_wire(rng, map);
        wire["seed"] = json!(seed);
        let imported = parse_pasted_state(&wire.to_string()).expect("parse");

        let mut game = Game::new();
        game.game_version = v;
        game.start_imported(&imported.run).expect("graft");
        assert!(game.import_notes().is_empty(), "{:?}", game.import_notes());
        let rm = game.native_run_mut().expect("run");
        assert_eq!(rm.game_version, v);
        assert_eq!(rm.run_state.rng.seed, source.run_state.rng.seed);
        assert_eq!(rm.run_state.rng.stream_mut("shops").unwrap().counter(), 2);
    }

    #[test]
    fn start_imported_refuses_a_foreign_seeds_cursors() {
        let source = build_import_run_manager(
            GameVersion::default(),
            CharacterId::Ironclad,
            "TESTSEED42",
            0,
        );
        let (map, _c, _r) = map_block_from(&source, false);
        let rng = json!({"run_seed": 999_999, "counters": {"shops": 2}});
        let wire = import_wire(rng, map);
        let imported = parse_pasted_state(&wire.to_string()).expect("parse");
        let mut game = Game::new();
        game.start_imported(&imported.run).expect("graft");
        assert!(game
            .import_notes()
            .iter()
            .any(|n| n.contains("not restored")));
    }

    #[test]
    fn start_imported_refuses_a_coord_off_this_seeds_map() {
        let source = build_import_run_manager(
            GameVersion::default(),
            CharacterId::Ironclad,
            "TESTSEED42",
            0,
        );
        let (mut map, _c, _r) = map_block_from(&source, false);
        map["current"] = json!({"col": 99, "row": 99});
        let wire = import_wire(Value::Null, map);
        let imported = parse_pasted_state(&wire.to_string()).expect("parse");
        let err = Game::new().start_imported(&imported.run).unwrap_err();
        assert!(err.0.contains("place the run on the map"), "{}", err.0);
    }

    #[test]
    fn faithful_run_starts_on_neow_with_an_integer_seed() {
        let mut game = Game::new();
        let state = game.start_faithful("Silent", 7, 0).expect("start");
        assert_eq!(state["decision"], json!("event_choice"));
        assert_eq!(state["event_name"], json!("NEOW.title"));
        assert_eq!(state["act1"], json!("overgrowth"));
        let again = Game::new().start_faithful("Silent", 7, 0).expect("start");
        assert_eq!(state["bg_seed"], again["bg_seed"]);
        let asc = Game::new().start_faithful("Silent", 7, 10).expect("start");
        assert_eq!(asc["decision"], json!("event_choice"));
    }

    #[test]
    fn event_loc_stems() {
        assert_eq!(event_loc_stem("LostWisp"), "LOST_WISP");
        assert_eq!(event_loc_stem("Neow"), "NEOW");
        assert_eq!(event_loc_stem("ExpectAFight"), "EXPECT_A_FIGHT");
    }

    #[test]
    fn isupper_matches_python() {
        assert!(py_isupper("WINGED_BOOTS"));
        assert!(!py_isupper("Gain 100 gold"));
        assert!(!py_isupper("123"));
        assert!(py_isupper("A1"));
    }

    #[test]
    fn indices_parse_like_python() {
        assert_eq!(
            parse_indices(Some(&json!("0,2, 4"))).unwrap(),
            vec![0, 2, 4]
        );
        assert_eq!(parse_indices(Some(&json!(3))).unwrap(), vec![3]);
        assert_eq!(parse_indices(Some(&json!(""))).unwrap(), Vec::<i64>::new());
        assert!(parse_indices(Some(&json!("x"))).is_err());
        assert_eq!(sorted_unique(vec![3, 1, 3, 2], true), vec![3, 2, 1]);
        assert_eq!(sorted_unique(vec![3, 1, 3, 2], false), vec![1, 2, 3]);
    }

    #[test]
    fn ironclad_run_starts_on_neow() {
        let mut game = Game::new();
        let state = game
            .start("Ironclad", "ABC", 0, "overgrowth")
            .expect("start");
        assert_eq!(state["decision"], json!("event_choice"));
        assert_eq!(state["event_name"], json!("NEOW.title"));
        assert_eq!(state["player"]["name"], json!("Ironclad"));
        assert_eq!(state["act1"], json!("overgrowth"));
        assert_eq!(state["act_name"], json!("Overgrowth"));
        assert!(state["options"].as_array().is_some_and(|o| !o.is_empty()));
        assert!(state.get("room_reveal").is_none());
        assert!(state["bg_seed"].is_number());
    }

    #[test]
    fn seed_json_keeps_a_ulong_seed_exact() {
        assert_eq!(seed_json(44_815_703), json!(44_815_703));
        assert_eq!(seed_json((1 << 53) - 1), json!(9_007_199_254_740_991_u64));
        assert_eq!(seed_json(1 << 53), json!("9007199254740992"));
        assert_eq!(seed_json(u64::MAX), json!("18446744073709551615"));
    }

    #[test]
    fn rolled_act1_resolves_from_the_run() {
        let mut game = Game::new();
        let state = game.start("Silent", "ROLLME", 0, "roll").expect("start");
        let act1 = state["act1"].as_str().unwrap();
        assert!(act1 == "overgrowth" || act1 == "underdocks");
    }

    #[test]
    fn illegal_verb_is_an_error_dict() {
        let mut game = Game::new();
        game.start("Ironclad", "ABC", 0, "overgrowth")
            .expect("start");
        let e = game.act("end_turn", &json!({})).unwrap_err();
        assert!(e.0.contains("rejected by engine"), "{e}");
        assert!(game.act("bogus", &json!({})).is_err());
    }
}
