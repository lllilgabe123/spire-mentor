//! PPO observation data model — a serde-transparent port of the dict
//! shape `sts2-pycli/python/ppo/env.py::observe()` emits (porting
//! authority; never modify that file). [`PpoObs`] deserializes the
//! committed parity fixtures' `"obs"` field and re-serializes to
//! JSON-equivalent output — compared as `serde_json::Value` (field
//! ORDER is not significant, only keys/values are), which is exactly
//! what this module's round-trip test does.
//!
//! Struct shapes mirror `sts2-rs/crates/sts2-py/src/run_manager.rs`'s
//! PyO3 getters one-for-one (every key those methods emit has a field
//! here): `full_map` (~1842) -> [`FullMapObs`]/[`MapNodeObs`]/
//! [`BossNodeObs`]/[`Coord`]; `shop_inventory` (~1184) -> [`ShopObs`]/
//! [`ShopCardObs`]/[`ShopEntryObs`]; `event_option_infos` (~1417) ->
//! [`EventOptionObs`]; `run_action_to_pydict` (~347) -> [`ActionDesc`].
//!
//! # Structural fields and the second producer
//!
//! There are now TWO producers of this shape. `env.py::observe()` builds
//! it from a `RunManager` and fills every key; `realgame_ppo._build_*`
//! builds it from a SpireMentor message, where several keys have no
//! equivalent on the wire and are simply omitted (`idx`, `on_sale`,
//! `is_colorless`, `removal_used`, `visited`, `current`, `option_id`).
//! Those fields carry `#[serde(default)]` so the bridge dialect
//! deserializes, and that is safe for one specific reason: **no encoder
//! reads them**. The `idx` values the featurizer indexes shop rows by
//! come off the ACTION dicts (`encoder.py`'s `a.get("idx")`), not off
//! these entries; the map position arrives as
//! [`FullMapObs::current_coord`]; the event `optidx` feature comes from
//! an option's position in the list.
//!
//! A field the encoder DOES read stays required on purpose — a bridge
//! that stopped sending it should fail loudly here rather than featurize
//! a silent zero. `diag_ppo_policy_parity.py` is what checks that this
//! stayed true, by scoring recorded bridge observations through both
//! this path and the numpy/onnxruntime one.

use serde::{Deserialize, Deserializer, Serialize};
use sts2_core::run_manager::RunAction;

/// Deserialize a present-but-possibly-`null` field into `Some(value)`,
/// where a MISSING field is handled separately by `#[serde(default)]`
/// (which yields the outer `None`). Needed for the `card_reward_bundles`
/// "double `Option`" field: a bare `#[serde(default)]` on
/// `Option<Option<T>>` can't tell "key absent" from "key present with
/// JSON `null`" apart — both collapse to the outer `None` under the
/// standard `Option<U>: Deserialize` impl, since JSON `null` maps
/// straight to that impl's own `None` regardless of what `U` is. This
/// helper forces the outer wrapper to `Some` whenever the key is
/// present at all, letting the inner `Option<T>::deserialize` do its
/// normal null-vs-value job.
fn deserialize_some<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    T::deserialize(deserializer).map(Some)
}

/// `{"col": i64, "row": i64}` — the dict form `full_map()`'s internal
/// `coord_dict` helper emits for node `children` entries and
/// `current_coord`. Distinct from [`PpoObs::map_coord`], which mirrors
/// `rm.map_coord()`'s Python-tuple return (a JSON *array*, not this
/// object shape).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Coord {
    pub col: i64,
    pub row: i64,
}

/// One `full_map()` row entry. `room_type` serializes to the JSON key
/// `"type"` (a PascalCase UI string, e.g. `"RestSite"` —
/// `mapfeat::_pascal_to_snake` converts it to the vocab's snake form).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MapNodeObs {
    pub col: i64,
    pub row: i64,
    #[serde(rename = "type")]
    pub room_type: String,
    pub children: Vec<Coord>,
    /// `default` because the SpireMentor dialect omits it — see the
    /// module note on structural fields.
    #[serde(default)]
    pub visited: bool,
    /// `default` for the same reason as [`Self::visited`]; the live
    /// position arrives as [`FullMapObs::current_coord`] instead.
    #[serde(default)]
    pub current: bool,
}

/// The `full_map()` `"boss"` dict — a terminal node with NO `children`
/// key (see that method's doc comment: "the boss dict intentionally
/// has NO children key") and optional `kind`/`id` display fields
/// (present only when the engine could resolve the pinned/rolled boss
/// encounter's identity).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BossNodeObs {
    pub col: i64,
    pub row: i64,
    /// `default`: the SpireMentor dialect sends the boss as a bare
    /// `{col, row}`, and nothing reads this — both map featurizers
    /// OVERWRITE it with the literal `"boss"` when they build the node
    /// table (`mapfeat.rs::nodes_from_full_map`, `mapfeat.py:141`), so
    /// only the coordinate is ever consumed.
    #[serde(default, rename = "type")]
    pub room_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
}

/// `RunManager.full_map()`'s full dict schema: `{"rows": [[node,
/// ...], ...], "boss": node|None, "second_boss": node|None,
/// "current_coord": coord|None}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FullMapObs {
    pub rows: Vec<Vec<MapNodeObs>>,
    #[serde(default)]
    pub boss: Option<BossNodeObs>,
    /// The A10 DoubleBoss node's IDENTITY (`full_map()`'s `second_boss`
    /// key); `None` below A10. It is also an ordinary `rows` entry —
    /// this key only carries the `id` that rows cannot express, so both
    /// map featurizers keep reading the node from `rows`.
    ///
    /// `default` because the SpireMentor dialect omits it, as with the
    /// structural fields above. It reuses [`BossNodeObs`] because the
    /// PyO3 getter emits the same `col`/`row`/`type` (+ optional `id`)
    /// shape there; it never sets `kind`, and `skip_serializing_if`
    /// keeps the absent keys absent so the Gate A2 byte-for-byte
    /// comparison against Python holds.
    #[serde(default)]
    pub second_boss: Option<BossNodeObs>,
    /// `default` (i.e. `None`) when absent, which is the SpireMentor shape
    /// before the player has moved. Unlike the fields above this one IS
    /// read — it is the player's position — but absent and explicit
    /// `null` mean the same thing to both featurizers
    /// (`full_map.get("current_coord")` yields `None` either way), so
    /// defaulting preserves the meaning rather than inventing one.
    #[serde(default)]
    pub current_coord: Option<Coord>,
}

/// One `shop_inventory()` card-row entry (character row then
/// colorless row, concatenated — `idx` is the flat index
/// `shop_buy_card` indexes). `id` is absent entirely (not `null`) when
/// the slot has no card (sold out / unfilled).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShopCardObs {
    /// `default`: structural, and absent in the SpireMentor dialect — the
    /// encoder indexes shop rows by the ACTION's `idx`, never this one.
    #[serde(default)]
    pub idx: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub price: i64,
    #[serde(default)]
    pub on_sale: bool,
    #[serde(default)]
    pub is_colorless: bool,
    pub available: bool,
}

/// One `shop_inventory()` relic- or potion-row entry — same shape as
/// [`ShopCardObs`] minus the card-only `on_sale`/`is_colorless` keys.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShopEntryObs {
    /// `default`, as [`ShopCardObs::idx`].
    #[serde(default)]
    pub idx: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub price: i64,
    pub available: bool,
}

/// `RunManager.shop_inventory()`'s full dict schema.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShopObs {
    pub cards: Vec<ShopCardObs>,
    pub relics: Vec<ShopEntryObs>,
    pub potions: Vec<ShopEntryObs>,
    pub removal_cost: i64,
    /// `default`: the bridge has no equivalent, and neither encoder
    /// reads it (the shop's removal option carries its own availability).
    #[serde(default)]
    pub removal_used: bool,
}

/// One `event_option_infos()` entry. `option_id` is a Python string
/// literal (e.g. `"claim"`, `"positive_1"`), sourced from
/// `sts2_core::events::EventOptionInfo::option_id: &'static str`
/// (`crates/sts2-core/src/events.rs`) — NOT numeric. This is a
/// deviation from the task brief's assumed `i64`; the committed
/// fixtures (and the PyO3 binding's `&str` passthrough) confirm the
/// string shape.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct EventOptionObs {
    /// `default`: the bridge names options by label, not by the engine's
    /// internal id, and neither encoder reads this — the `optidx` feature
    /// comes from the option's POSITION.
    #[serde(default)]
    pub option_id: String,
    pub label: String,
    pub description: String,
    pub enabled: bool,
    /// The game's own localization key + template values for this option
    /// (`run_manager.rs::event_option_infos`'s `loc_key`/`vars`/`title_vars`
    /// — a `workbench` display concern, not read by either encoder). Not
    /// part of GH #545; added alongside it purely to keep this struct's
    /// round-trip test (`round_trips_every_fixture_obs_row`) honest against
    /// the current PyO3 binding shape — omitting them made every fixture
    /// row silently drop these keys on re-serialization.
    ///
    #[serde(default)]
    pub loc_key: String,
    #[serde(default)]
    pub vars: std::collections::BTreeMap<String, serde_json::Value>,
    #[serde(default)]
    pub title_vars: std::collections::BTreeMap<String, serde_json::Value>,
    /// Structured effect payload (GH #237): the engine's own signed
    /// gold / current-HP / max-HP deltas for resolving this option,
    /// surfaced so the encoder need not regex-scrape `description`.
    /// `None` (key absent — the pre-#237 fixture shape) makes the
    /// encoder fall back to parsing the description text, so old
    /// committed fixtures featurize identically.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gold_delta: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hp_delta: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_hp_delta: Option<i64>,
    /// What the option costs in things that are not gold or HP. GAINED and
    /// LOST are separate because two of these events TRADE, and a net count
    /// collapses a trade back to "nothing happened" — which is precisely how
    /// the payload used to read for Relic Trader. Absent (the pre-existing
    /// fixture shape) leaves the slots at zero, so old fixtures featurize
    /// identically.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relics_gained: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relics_lost: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub potions_gained: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub potions_lost: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cards_gained: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cards_lost: Option<i64>,
}

/// One card's enchantment as `(UPPER_SNAKE id, amount)`, `None` when it
/// carries none — `sts2_wire::emit::run_manager::card_enchantment`'s shape.
/// C# allows at most one enchantment per card, so this is its whole
/// run-persistent modifier state apart from the upgrade flag (GH #947).
pub type CardEnchantObs = Option<(String, i64)>;

/// [`CardEnchantObs`] of one engine card's enchantment stack: the first
/// entry, since C# holds at most one. Mirrors
/// `sts2_wire::emit::run_manager::card_enchantment` (the PyO3 side `env.py`
/// reads), which Gate A2 holds byte-identical to this.
#[must_use]
pub fn card_enchant_obs(e: &sts2_core::cards::CardEnchantments) -> CardEnchantObs {
    e.first()
        .map(|(id, amount)| (id.python_name().to_string(), i64::from(*amount)))
}

/// `env.py::observe()`'s full observation dict — the encoder's (Task
/// 6) only input besides the raw action list. The surface-gated
/// extras are `Option`-wrapped and skip serialization when absent, so
/// e.g. a `"path"` decision's JSON has no `shop`/`event_name`/… keys
/// at all, matching Python's dict-literal-per-surface construction in
/// `observe()`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PpoObs {
    pub phase: String,
    pub surface: Option<String>,
    pub seed: i64,
    pub ascension: i64,
    pub deck: Vec<(String, bool)>,
    /// Each deck card's enchantment, index-aligned with [`Self::deck`]
    /// (GH #947). `deck` carries only `(id, upgraded)`, so without this an
    /// enchanted card reads as the bare card on every run decision. Absent
    /// (a recording that predates the field) reads as "nothing enchanted";
    /// only the `card_mods` feature ([`super::features::Features`]) reads it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deck_enchantments: Option<Vec<CardEnchantObs>>,
    pub relics: Vec<String>,
    pub potions: Vec<String>,
    pub map: Option<FullMapObs>,
    pub map_coord: Option<(i64, i64)>,

    pub total_floor: i64,
    pub act_index: i64,
    pub act_floor: i64,
    pub hp: i64,
    pub max_hp: i64,
    pub gold: i64,
    pub deck_size: i64,
    pub num_upgraded: i64,
    pub num_relics: i64,
    pub num_potions: i64,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub card_reward_option_ids: Option<Vec<String>>,
    /// Per-option upgrade flags, index-aligned with
    /// `card_reward_option_ids`. Shorter than the ids (or absent) means
    /// "not upgraded" — bundle / choose-screen offerings carry no upgrade
    /// state, per `RunManager::card_reward_upgraded_flags`.
    ///
    /// Only read when the `reward_upgrade` feature is on
    /// ([`super::features::Features`]); producers ship it unconditionally so
    /// the flag stays a pure encoder decision.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub card_reward_option_upgraded: Option<Vec<bool>>,
    /// Per-option enchantment (a relic stamp: FresnelLens's Nimble,
    /// SilkenTress's Glam, …; GH #806), index-aligned with
    /// `card_reward_option_ids`. Shorter (or absent) means "not enchanted".
    /// Read only under the `card_mods` feature. GH #947.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub card_reward_option_enchantments: Option<Vec<CardEnchantObs>>,
    /// Double-`Option`: the outer gates on surface (`"card-pick"`
    /// only, mirrors every other extra below); the inner is Python's
    /// `card_reward_bundles() -> list[list[str]] | None`. `Some(None)`
    /// serializes to JSON `null` (the common case: an ordinary,
    /// non-bundle card reward); `None` omits the key entirely (any
    /// other surface). `#[serde(default)]` on the *field* only
    /// controls the missing-key case, so a present `null` still
    /// deserializes correctly to `Some(None)`.
    #[serde(
        default,
        deserialize_with = "deserialize_some",
        skip_serializing_if = "Option::is_none"
    )]
    pub card_reward_bundles: Option<Option<Vec<Vec<String>>>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relic_reward_ids: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub boss_relic_options: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub smith_options: Option<Vec<String>>,
    /// Index-aligned with `smith_options` (each Smith option is the k-th
    /// upgradable deck card). Read only under `card_mods`. GH #947.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub smith_option_enchantments: Option<Vec<CardEnchantObs>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub card_removal_options: Option<Vec<(String, bool)>>,
    /// Index-aligned with `card_removal_options` (and the grant twin below
    /// with `card_grant_options`). Read only under the `card_mods` feature.
    /// GH #947. Declared in `env.py::observe()`'s key order, which is the
    /// order this struct serializes in.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub card_removal_option_enchantments: Option<Vec<CardEnchantObs>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub card_grant_options: Option<Vec<(String, bool)>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub card_grant_option_enchantments: Option<Vec<CardEnchantObs>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transform_select_deck_indices: Option<Vec<i64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enchant_select_deck_indices: Option<Vec<i64>>,
    /// Which enchantment the pending deck-enchant pick stamps (GH #545).
    /// Double-`Option`, same shape as [`Self::card_reward_bundles`] above and
    /// for the same reason: `env.py::observe()` sets this key unconditionally
    /// on the `"card-pick"` surface (`obs["pending_enchantment"] =
    /// rm.pending_enchantment`), so it is present as JSON `null` there even
    /// with nothing queued — NOT the "key absent ⟺ value is `None`" pattern
    /// [`Self::pending_potion_swap`] uses. Outer `None` (every other surface,
    /// and a champion-era recording that predates the field) omits the key
    /// entirely; outer `Some(None)` serializes to `null`.
    #[serde(
        default,
        deserialize_with = "deserialize_some",
        skip_serializing_if = "Option::is_none"
    )]
    pub pending_enchantment: Option<Option<String>>,
    /// Master-deck positions the engine offers for `UpgradeDeckCard`, in
    /// offer order — the `Card::is_upgradable()` subset (curses / statuses /
    /// Quest cards excluded), which is NOT the `!upgraded` subset the raw
    /// `deck` field would yield. GH #399 Task 5b.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub upgrade_select_deck_indices: Option<Vec<i64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shop: Option<ShopObs>,
    /// Debug name of the potion being OFFERED on a full belt, e.g.
    /// `"EnergyPotion"` — `RunManager::pending_potion_swap`. Present only
    /// on the `"potion-swap"` surface. Every option token names a potion
    /// you would THROW AWAY, so without this the decision is unanswerable;
    /// the encoder puts it on the state token. GH #252/#379.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pending_potion_swap: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub event_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub event_options: Option<Vec<EventOptionObs>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rest_option_labels: Option<Vec<String>>,
}

/// Non-combat `RunAction` shapes, JSON-tagged exactly like
/// `run_manager.rs::run_action_to_pydict`'s output: the `"action"` key
/// selects the variant, and `#[serde(rename_all = "snake_case")]`
/// turns each PascalCase variant name into that function's literal
/// strings (e.g. `PickCardReward` -> `"pick_card_reward"`,
/// `ShopRemoveCard` -> `"shop_remove_card"`). Combat-only shapes
/// (`play_card`, `end_turn`, `combat_resolve_choice*`,
/// `drink_potion`) and anything else `run_action_to_pydict` might
/// emit degrade to [`ActionDesc::Unknown`] on deserialize — mirrors
/// `encoder.py`'s never-crash rule for the live path, where
/// `PpoPolicy` (Task 8) never actually reaches a combat decision
/// (that surface belongs to `MctsCombatPolicy`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum ActionDesc {
    Move {
        col: i64,
    },
    PickCardReward {
        idx: Option<i64>,
    },
    SacrificeCardReward,
    PickRelicReward,
    SkipRelic,
    PickBossRelic {
        idx: i64,
    },
    SmithCard {
        idx: i64,
    },
    RemoveDeckCard {
        idx: i64,
    },
    UpgradeDeckCard {
        idx: i64,
    },
    TransformDeckCard {
        idx: i64,
    },
    EnchantDeckCard {
        idx: i64,
    },
    GrantDeckCard {
        idx: i64,
    },
    ShopBuyCard {
        idx: i64,
    },
    ShopBuyRelic {
        idx: i64,
    },
    ShopBuyPotion {
        idx: i64,
    },
    ShopRemoveCard {
        deck_idx: i64,
    },
    LeaveShop,
    RestOption {
        idx: i64,
    },
    EventChoice {
        idx: i64,
    },
    CollectTreasure,
    /// Full-belt potion-reward swap (GH #252/#379). `discard_slot: None`
    /// declines the reward outright; `Some(slot)` throws away the held
    /// potion at `slot` to make room. Surfaced in ANY phase, and only
    /// when the run carries `interactive_potion_swap`.
    PotionSwap {
        discard_slot: Option<i64>,
    },
    #[serde(other)]
    Unknown,
}

/// Live-path bridge: converts an engine [`RunAction`] straight to its
/// [`ActionDesc`] without a JSON round-trip (used when featurizing a
/// `RunManager`-sourced candidate list instead of a fixture row).
/// Combat variants — and anything else not covered by one of the tags
/// above — are `Err`: `PpoPolicy` never calls this on a combat
/// decision (combat stays `MctsCombatPolicy`'s), so reaching `Err`
/// here would mean a caller-side ownership bug, not a value this type
/// should silently paper over.
impl TryFrom<&RunAction> for ActionDesc {
    type Error = ();

    fn try_from(a: &RunAction) -> Result<Self, Self::Error> {
        Ok(match *a {
            RunAction::Move { col } => ActionDesc::Move {
                col: i64::from(col),
            },
            RunAction::PickCardReward { idx } => ActionDesc::PickCardReward {
                idx: idx.map(|i| i as i64),
            },
            RunAction::SacrificeCardReward => ActionDesc::SacrificeCardReward,
            RunAction::PickRelicReward => ActionDesc::PickRelicReward,
            RunAction::SkipRelic => ActionDesc::SkipRelic,
            RunAction::PickBossRelic { idx } => ActionDesc::PickBossRelic { idx: idx as i64 },
            RunAction::SmithCard { idx } => ActionDesc::SmithCard { idx: idx as i64 },
            RunAction::RemoveDeckCard { idx } => ActionDesc::RemoveDeckCard { idx: idx as i64 },
            RunAction::UpgradeDeckCard { idx } => ActionDesc::UpgradeDeckCard { idx: idx as i64 },
            RunAction::TransformDeckCard { idx } => {
                ActionDesc::TransformDeckCard { idx: idx as i64 }
            }
            RunAction::EnchantDeckCard { idx } => ActionDesc::EnchantDeckCard { idx: idx as i64 },
            RunAction::GrantDeckCard { idx } => ActionDesc::GrantDeckCard { idx: idx as i64 },
            RunAction::ShopBuyCard { idx } => ActionDesc::ShopBuyCard { idx: idx as i64 },
            RunAction::ShopBuyRelic { idx } => ActionDesc::ShopBuyRelic { idx: idx as i64 },
            RunAction::ShopBuyPotion { idx } => ActionDesc::ShopBuyPotion { idx: idx as i64 },
            RunAction::ShopRemoveCard { deck_idx } => ActionDesc::ShopRemoveCard {
                deck_idx: deck_idx as i64,
            },
            RunAction::LeaveShop => ActionDesc::LeaveShop,
            RunAction::RestOption { idx } => ActionDesc::RestOption { idx: idx as i64 },
            RunAction::EventChoice { idx } => ActionDesc::EventChoice { idx: idx as i64 },
            RunAction::CollectTreasure => ActionDesc::CollectTreasure,
            RunAction::PotionSwap { discard_slot } => ActionDesc::PotionSwap {
                discard_slot: discard_slot.map(|s| s as i64),
            },
            _ => return Err(()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn action_desc_tags_match_run_action_to_pydict() {
        let v = serde_json::to_value(ActionDesc::PickCardReward { idx: Some(2) }).unwrap();
        assert_eq!(v["action"], "pick_card_reward");
        assert_eq!(v["idx"], 2);

        let v = serde_json::to_value(ActionDesc::ShopRemoveCard { deck_idx: 3 }).unwrap();
        assert_eq!(v["action"], "shop_remove_card");
        assert_eq!(v["deck_idx"], 3);

        let v = serde_json::to_value(ActionDesc::CollectTreasure).unwrap();
        assert_eq!(v["action"], "collect_treasure");
    }

    #[test]
    fn action_desc_unknown_never_crashes_on_combat_tags() {
        let a: ActionDesc = serde_json::from_value(
            serde_json::json!({"action": "play_card", "hand_idx": 0, "target_index": null}),
        )
        .unwrap();
        assert_eq!(a, ActionDesc::Unknown);
    }

    #[test]
    fn try_from_run_action_covers_every_ppo_variant() {
        let cases: &[(RunAction, ActionDesc)] = &[
            (RunAction::Move { col: 2 }, ActionDesc::Move { col: 2 }),
            (
                RunAction::PickCardReward { idx: None },
                ActionDesc::PickCardReward { idx: None },
            ),
            (
                RunAction::SacrificeCardReward,
                ActionDesc::SacrificeCardReward,
            ),
            (RunAction::PickRelicReward, ActionDesc::PickRelicReward),
            (RunAction::SkipRelic, ActionDesc::SkipRelic),
            (
                RunAction::PickBossRelic { idx: 1 },
                ActionDesc::PickBossRelic { idx: 1 },
            ),
            (RunAction::LeaveShop, ActionDesc::LeaveShop),
            (RunAction::CollectTreasure, ActionDesc::CollectTreasure),
        ];
        for (action, expected) in cases {
            let got = ActionDesc::try_from(action).unwrap();
            assert_eq!(&got, expected);
        }
        assert!(ActionDesc::try_from(&RunAction::EndTurn).is_err());
        assert!(ActionDesc::try_from(&RunAction::PlayCard {
            hand_idx: 0,
            target: None
        })
        .is_err());
    }

    /// Insert `loc_key`/`vars`/`title_vars` defaults into every
    /// `obs["event_options"]` entry that lacks them — see the call site's
    /// comment for why. A no-op on any row already carrying the keys.
    fn backfill_event_option_defaults(obs: &mut serde_json::Value) {
        let Some(opts) = obs.get_mut("event_options").and_then(|v| v.as_array_mut()) else {
            return;
        };
        for opt in opts {
            let Some(map) = opt.as_object_mut() else {
                continue;
            };
            map.entry("loc_key")
                .or_insert_with(|| serde_json::Value::String(String::new()));
            map.entry("vars")
                .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
            map.entry("title_vars")
                .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
        }
    }

    /// Insert an explicit `second_boss: null` into `obs["map"]` when the key
    /// is absent — the A10 DoubleBoss identity key, added to the PyO3
    /// `full_map()` getter (and mirrored here) after this fixture was
    /// captured. Same remedy and same reasoning as
    /// [`backfill_event_option_defaults`]: `full_map()` sets the key
    /// UNCONDITIONALLY (`py.None()` below A10), so Python's live obs always
    /// carries it and a `skip_serializing_if` on the Rust side would omit it
    /// on every non-A10 board — silently breaking Gate A2's byte-for-byte
    /// live comparison to fix a stale committed corpus. A no-op on a row that
    /// already has the key, and on a row whose `map` is null.
    fn backfill_map_second_boss(obs: &mut serde_json::Value) {
        let Some(map) = obs.get_mut("map").and_then(|v| v.as_object_mut()) else {
            return;
        };
        map.entry("second_boss").or_insert(serde_json::Value::Null);
    }

    /// Read the committed encoder-parity fixture (if present — skip
    /// gracefully otherwise, mirroring `run_policy_net_parity.rs`) and
    /// assert every row's `"obs"` field round-trips through `PpoObs`
    /// byte-for-key-and-value-identical (compared as
    /// `serde_json::Value`, so key ORDER doesn't matter, only presence
    /// and values do). This is the cheap high-value smoke test that
    /// this struct's fields/types/`skip_serializing_if` gates actually
    /// match every surface (`path`, `card-pick`, `shop`, `event`,
    /// `rest`) the fixture carries.
    #[test]
    fn round_trips_every_fixture_obs_row() {
        let path = crate::ppo::test_fixture_path("ppo_encoder_parity.jsonl");
        let Ok(text) = std::fs::read_to_string(&path) else {
            eprintln!(
                "ppo_encoder_parity.jsonl fixture absent at {}; skipping obs round-trip check",
                path.display()
            );
            return;
        };
        let mut checked = 0usize;
        let mut surfaces = std::collections::BTreeSet::new();
        for (i, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let row: serde_json::Value = serde_json::from_str(line)
                .unwrap_or_else(|e| panic!("line {i}: invalid JSON: {e}"));
            let obs_value = row["obs"].clone();
            let parsed: PpoObs = serde_json::from_value(obs_value.clone()).unwrap_or_else(|e| {
                panic!("line {i}: failed to deserialize obs into PpoObs: {e}\n{obs_value}")
            });
            if let Some(s) = parsed.surface.clone() {
                surfaces.insert(s);
            }
            let round_tripped = serde_json::to_value(&parsed).expect("serialize PpoObs");
            // `loc_key`/`vars`/`title_vars` are UNCONDITIONALLY set by the live
            // `event_option_infos()` binding (predates this fixture, and predates
            // GH #545 entirely — commit bf87b04aa), so `EventOptionObs` always
            // serializes them. This committed fixture was captured before that
            // binding change and genuinely has no such keys — patching them onto
            // the fixture side (rather than making the field skip-if-empty, which
            // would silently break Gate A2's LIVE parity check, since Python's
            // live obs always carries the keys) keeps this test asserting the
            // property it exists for — every field PpoObs knows about round-trips
            // — without freezing the corpus to this fixture's capture-time schema.
            let mut expected = obs_value.clone();
            backfill_event_option_defaults(&mut expected);
            backfill_map_second_boss(&mut expected);
            assert_eq!(round_tripped, expected, "line {i}: round-trip mismatch");
            checked += 1;
        }
        assert!(checked > 0, "fixture had zero rows");
        eprintln!("round_trips_every_fixture_obs_row: {checked} rows, surfaces={surfaces:?}");
    }
}
