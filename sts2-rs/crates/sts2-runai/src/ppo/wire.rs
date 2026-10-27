//! Bridge wire JSON → [`PpoObs`] + candidate [`ActionDesc`]s.
//!
//! **Porting authority:** `sts2-pycli/python/realgame_ppo.py` — `_base_obs`,
//! `_base_metrics`, `_card_id`/`_vocab_card_id`, and the six
//! `_PPO_NONCOMBAT_BUILDERS` entries. That file is never modified by this
//! port; `tests/wire_obs_parity.rs` and the pycli-side differential harness
//! hold the two together.
//!
//! ## Why this module exists
//!
//! [`super::policy::build_obs`] builds the observation from a live
//! `RunManager` — the engine's own run state. The advisor lane has no such
//! object: SpireMentor sends the wire schema in `spire-mentor-mod/PROTOCOL.md` and
//! nothing else. That is the one piece of the PPO path with no Rust
//! counterpart, so the advisor has had to build the observation in Python,
//! dragging `numpy` + `onnxruntime` (~54 MB of the shipped bundle) along with
//! it. Everything downstream — [`super::encoder::featurize`], the tract
//! runtime in [`super::net`] — has been in Rust since the 2026-07-07
//! ppo-rust-inference plan.
//!
//! ## Faithfulness notes
//!
//! The Python builders encode a series of hard-won corrections, each of which
//! is reproduced here with its reason, because losing one is silent: the
//! observation still builds, the net still answers, and only the win rate
//! knows. In particular:
//!
//! * **The row-0 map guard.** `rm.map_coord()` is `None` at an act's first map
//!   choice, but the DLL reports the pre-act Neow/Ancient node (row 0) as
//!   `current` — a row its own act graph never contains. Feeding it through
//!   made `mapfeat` compute real forward cones where training saw the
//!   degenerate act-start encoding, flipping the act-start route on 27 of 42
//!   exactly-reproduced live decisions (2026-08-02 pathing diagnosis).
//! * **`total_floor` is `floor - 1`, `act_index` is 1-based.** The engine's
//!   `rm.total_floor()` is 0 at Neow where the DLL's `context.total_floor` is
//!   1. Both fed the encoder off-by-one on every meta decision — 4735/4735 in
//!   the v3x sweeps.
//! * **`_STATUS` vocab suffix.** The registry disambiguates ids colliding
//!   with another registry kind by a suffix (`SLOTH` → `SLOTH_STATUS`) and
//!   the bridge sends the plain id. 407/6074 deck tokens went OOV before the
//!   fix. The `_CARD` half of that family (`RAGE` → `RAGE_CARD`) is now
//!   handled upstream — [`vocab::canonical`] strips a trailing `_CARD`, so a
//!   bare `RAGE` already lands on `RAGE_CARD`'s slot — but the lookup is kept
//!   in the same shape as `_vocab_card_id` so the two emit identical tokens.

use serde_json::Value;

use super::obs::{
    ActionDesc, BossNodeObs, Coord, EventOptionObs, FullMapObs, MapNodeObs, PpoObs, ShopCardObs,
    ShopEntryObs, ShopObs,
};
use super::vocab;

/// One non-combat decision translated into the net's input shape, plus the
/// bridge-facing bookkeeping the caller needs to act on the answer.
///
/// Mirrors `realgame_ppo.PpoDecision`, minus its `metrics` field: the Rust
/// [`PpoObs`] already carries the scalars `_base_metrics` computed
/// (`total_floor`, `act_index`, `act_floor`, `hp`, …), so there is nothing
/// left for a separate dict to hold.
#[derive(Debug, Clone, PartialEq)]
pub struct WireDecision {
    pub obs: PpoObs,
    /// Candidates in the order the net will score them.
    pub actions: Vec<ActionDesc>,
    /// The bridge reply for each candidate, same order as `actions`.
    pub responses: Vec<Value>,
    /// Human-readable label per candidate, same order as `actions`.
    pub labels: Vec<String>,
}

// ---------------------------------------------------------------------------
// Small JSON readers — absent, null and wrong-type all read as the default
// ---------------------------------------------------------------------------
//
// Python's `.get(k) or default` collapses those three cases, and the builders
// lean on it everywhere. Reproducing that leniency is deliberate: this path
// runs against a live game under a never-crash contract, and a bridge dialect
// that drops a field must degrade the observation, not kill the decision.

fn obj<'a>(v: &'a Value, key: &str) -> Option<&'a Value> {
    v.get(key).filter(|x| !x.is_null())
}

fn arr<'a>(v: &'a Value, key: &str) -> &'a [Value] {
    obj(v, key)
        .and_then(Value::as_array)
        .map_or(&[][..], |a| a.as_slice())
}

fn s(v: &Value, key: &str) -> String {
    obj(v, key).and_then(Value::as_str).unwrap_or("").to_owned()
}

fn opt_s(v: &Value, key: &str) -> Option<String> {
    obj(v, key)
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .filter(|x| !x.is_empty())
}

fn i(v: &Value, key: &str) -> i64 {
    obj(v, key).and_then(Value::as_i64).unwrap_or(0)
}

fn b(v: &Value, key: &str) -> bool {
    obj(v, key).and_then(Value::as_bool).unwrap_or(false)
}

/// `option.get("enabled", True)` — absent means enabled, only an explicit
/// `false` disables.
fn enabled(o: &Value) -> bool {
    o.get("enabled").and_then(Value::as_bool).unwrap_or(true)
}

/// `int(o.get("index", i))` — the bridge's own option index, falling back to
/// positional order.
fn bridge_index(o: &Value, fallback: usize) -> i64 {
    o.get("index")
        .and_then(Value::as_i64)
        .unwrap_or(fallback as i64)
}

fn choose(index: i64) -> Value {
    serde_json::json!({ "action": "choose", "index": index })
}

// ---------------------------------------------------------------------------
// Card ids
// ---------------------------------------------------------------------------

/// Bridge card id → the training vocab's canonical id.
///
/// The registry disambiguates cards whose name collides with another registry
/// kind by a `_CARD` suffix (`RAGE_CARD`, `FLAME_BARRIER_CARD`, …) and the one
/// such status by `_STATUS` (`SLOTH` → `SLOTH_STATUS`); the bridge sends the
/// plain game id, which misses. Try the bare id, then each suffix; degrade to
/// the raw id when none land, so an genuinely unknown card still reads as OOV
/// rather than crashing.
#[must_use]
pub fn vocab_card_id(raw: &str) -> String {
    let reg = vocab::cards();
    if reg.index(raw) != reg.oov {
        return raw.to_owned();
    }
    for suffix in ["_CARD", "_STATUS"] {
        let suffixed = format!("{raw}{suffix}");
        if reg.index(&suffixed) != reg.oov {
            return suffixed;
        }
    }
    raw.to_owned()
}

/// A card entry's vocab id, or `None` for an entry with no `id` at all.
fn card_id(c: &Value) -> Option<String> {
    opt_s(c, "id").map(|raw| vocab_card_id(&raw))
}

/// Dialect-tolerant card-id canon (PascalCase or loc id → UPPER_SNAKE).
/// Mirrors `realgame_ppo._canon_card`.
fn canon_card(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len() + 4);
    let chars: Vec<char> = raw.chars().collect();
    for (idx, ch) in chars.iter().enumerate() {
        if idx > 0 && ch.is_ascii_uppercase() {
            let prev = chars[idx - 1];
            if prev.is_ascii_lowercase() || prev.is_ascii_digit() {
                out.push('_');
            }
        }
        out.extend(ch.to_uppercase());
    }
    out
}

/// Registry names the GENERIC removal offer will not show, from the engine.
///
/// This replaces a hand-copied basic-Strike/Defend list whose stated premise
/// -- "the net was TRAINED on an engine action surface that excluded them" --
/// did not survive its own history. `Phase::Shop` has pushed one
/// `ShopRemoveCard` per `removal_offer_order()` entry since 2026-06-19
/// (`run_manager.rs`), and `removal_offer_order` has been Eternal-gated since
/// the same commit, so every champion trained after that date saw basics as
/// shop-removal candidates. The hand-copied list was the thing making
/// inference differ from training, not the thing preventing it.
///
/// Getting it from the engine also fixes the other end. The old list excluded
/// only basics, so the fan-out offered Eternal curses -- live logs carry
/// `remove:ASCENDERS_BANE` and `remove:ENTHRALLED` shop picks that no
/// `FromDeckForRemoval` screen can ever satisfy, which then dropped the
/// carried target and let the follow-up `card_select` re-decide.
fn non_removable_names() -> &'static std::collections::HashSet<String> {
    static SET: std::sync::OnceLock<std::collections::HashSet<String>> = std::sync::OnceLock::new();
    SET.get_or_init(|| {
        sts2_core::cards::CardId::all()
            .iter()
            .copied()
            .filter(|&c| !sts2_content::cards::spec_index::card_is_removable_csharp(c))
            .map(|c| c.to_python_name().to_string())
            .collect()
    })
}

/// Whether a wire `master_deck` entry can appear on a removal screen.
///
/// [`non_removable_names`] answers by card id, which is not the whole rule:
/// C# reads the card's LIVE keyword set, and `TezcatarasEmber` adds Eternal
/// on enchant (`TezcatarasEmber.cs:18`). `NutritiousSoup` puts that
/// enchantment on every basic Strike, so after picking it up a Strike is
/// offered by an id-only filter and withheld by the game — the fan-out then
/// picks a target the follow-up `card_select` cannot find, drops the carry,
/// and re-decides on the screen.
///
/// The enchantment is read off the entry (`SimStateExtractor` emits
/// `Id.Entry`, normalised by `EnchantmentId::from_wire_name`). An absent or
/// unrecognised enchantment reads as "no enchantment", which keeps an older
/// bridge build that omits the field behaving exactly as before.
fn wire_card_is_removable(entry: &Value) -> bool {
    if non_removable_names().contains(&canon_card(&s(entry, "id"))) {
        return false;
    }
    match opt_s(entry, "enchantment")
        .as_deref()
        .and_then(sts2_core::cards::EnchantmentId::from_wire_name)
    {
        Some(e) => {
            let mut probe = sts2_core::cards::Card::pod_defaults();
            probe.card_id = sts2_core::cards::CardId::StrikeIronclad;
            probe.set_enchantment(e, 1);
            sts2_content::cards::spec_index::card_is_removable_instance(&probe)
        }
        None => true,
    }
}

// ---------------------------------------------------------------------------
// The common observation
// ---------------------------------------------------------------------------

/// Whether a card entry counts as upgraded, over both wire spellings
/// (`upgrade_level` int, `upgraded` bool).
fn is_upgraded(c: &Value) -> bool {
    let level = c.get("upgrade_level").and_then(Value::as_i64).unwrap_or(0) != 0;
    level || b(c, "upgraded")
}

/// A wire card's enchantment as `(registry id, amount)` (GH #947), mirroring
/// `realgame_ppo._card_enchantment`. The wire spells it as the C# id entry or
/// the display title (`PROTOCOL.md`: `enchantment`); either resolves to the
/// UPPER_SNAKE registry id the engine-side obs carries. An unknown spelling is
/// kept verbatim (it lands on the vocab's OOV row) rather than dropped, so a
/// new enchantment still reads as "enchanted with something". An absent
/// `enchantment_amount` is one stack, as in `sts2_wire::state_dict`.
fn card_enchantment(c: &Value) -> super::obs::CardEnchantObs {
    let name = opt_s(c, "enchantment")?;
    let id = sts2_core::cards::EnchantmentId::from_wire_name(&name)
        .map_or(name, |e| e.python_name().to_string());
    let amount = obj(c, "enchantment_amount")
        .and_then(Value::as_i64)
        .unwrap_or(1);
    Some((id, amount))
}

/// The act-map current coord, with the row-0 guard applied.
///
/// Returns `None` both when there is no map in reach and when `current` sits
/// on row 0 — the pre-act Neow/Ancient node, which the act graph never
/// contains. See the module note: feeding it through flipped 27 of 42
/// reproduced act-start routes.
fn current_coord(state: &Value) -> Option<Coord> {
    let map = obj(state, "map")?;
    // Only trust `current` when the graph came with it, matching Python's
    // `m.get("current") if nodes else None`.
    if arr(map, "nodes").is_empty() {
        return None;
    }
    let cur = obj(map, "current")?;
    let row = cur.get("row").and_then(Value::as_i64).unwrap_or(-1);
    if row == 0 {
        return None;
    }
    Some(Coord {
        col: i(cur, "col"),
        row,
    })
}

/// One bridge map node → the `full_map()` node shape `mapfeat` consumes.
fn map_node(n: &Value) -> MapNodeObs {
    MapNodeObs {
        col: i(n, "col"),
        row: i(n, "row"),
        room_type: s(n, "type"),
        children: arr(n, "children")
            .iter()
            .map(|c| Coord {
                col: i(c, "col"),
                row: i(c, "row"),
            })
            .collect(),
        visited: b(n, "visited"),
        current: b(n, "current"),
    }
}

/// The fields every surface's observation carries.
///
/// Ports `_base_obs` and `_base_metrics` together — the Rust [`PpoObs`] holds
/// both in one struct, so the Python split into two dicts has no counterpart
/// here.
fn base_obs(state: &Value) -> PpoObs {
    let run_state = state.get("run_state").cloned().unwrap_or(Value::Null);
    let deck_entries = arr(&run_state, "master_deck");
    let deck: Vec<(String, bool)> = deck_entries
        .iter()
        .map(|c| (card_id(c).unwrap_or_default(), is_upgraded(c)))
        .collect();
    let deck_enchantments: Vec<super::obs::CardEnchantObs> =
        deck_entries.iter().map(card_enchantment).collect();
    let num_upgraded = deck_entries.iter().filter(|c| is_upgraded(c)).count() as i64;

    let relics: Vec<String> = arr(&run_state, "relics")
        .iter()
        .filter_map(|r| opt_s(r, "id"))
        .collect();
    // An "empty" belt slot is not a potion — `num_potions` and the token list
    // must agree with the engine's, which never sees the empties.
    let potion_entries: Vec<&Value> = arr(&run_state, "potions")
        .iter()
        .filter(|p| !b(p, "empty"))
        .collect();
    let potions: Vec<String> = potion_entries
        .iter()
        .filter_map(|p| opt_s(p, "id"))
        .collect();

    let map_nodes = obj(state, "map").map_or(&[][..], |m| arr(m, "nodes"));
    // The bridge ships a FLAT node list; `nodes_from_full_map` only flattens
    // `rows` and never depends on row grouping being meaningful, so a single
    // row carries the same graph.
    let map = if map_nodes.is_empty() {
        None
    } else {
        Some(FullMapObs {
            rows: vec![map_nodes.iter().map(map_node).collect()],
            boss: obj(state, "map")
                .and_then(|m| obj(m, "boss"))
                .map(|bn| BossNodeObs {
                    col: i(bn, "col"),
                    row: i(bn, "row"),
                    room_type: s(bn, "type"),
                    kind: opt_s(bn, "kind"),
                    id: opt_s(bn, "id"),
                }),
            // Read the same way as `boss` rather than pinned to `None`: the
            // bridge dialect ships no `second_boss` today, so this yields
            // `None` anyway, and reading it means the field cannot silently
            // drift if the bridge starts sending one. Nothing on this side
            // consumes it — `mapfeat` takes the node from `rows` — so it is
            // carried for shape only.
            second_boss: obj(state, "map")
                .and_then(|m| obj(m, "second_boss"))
                .map(|bn| BossNodeObs {
                    col: i(bn, "col"),
                    row: i(bn, "row"),
                    room_type: s(bn, "type"),
                    kind: opt_s(bn, "kind"),
                    id: opt_s(bn, "id"),
                }),
            // Left unset, matching `_base_obs`'s map dict. The coord the
            // features are built from is the top-level `map_coord` below —
            // `mapfeat::compute(obs.map, obs.map_coord, …)` on this side and
            // `obs.get("map_coord")` on Python's, neither of which looks
            // inside the map dict. Filling it here would put a second copy of
            // the same fact where nothing reads it.
            current_coord: None,
        })
    };
    let coord = current_coord(state);

    PpoObs {
        phase: s(state, "type"),
        // `None`, matching `_base_obs`. The fine surface the net actually
        // sees is derived from the candidate list (`fine_surface(actions)`),
        // not read off this field — neither encoder touches it — so carrying
        // the label here would only make the two observations differ.
        surface: None,
        seed: 0,
        ascension: i(&run_state, "ascension"),
        deck,
        deck_enchantments: Some(deck_enchantments),
        relics: relics.clone(),
        potions: potions.clone(),
        map,
        map_coord: coord.map(|c| (c.col, c.row)),

        // `total_floor`: the engine is 0 at Neow where the DLL is 1.
        total_floor: (i(state, "floor") - 1).max(0),
        act_index: obj(state, "act").and_then(Value::as_i64).unwrap_or(1),
        // `act_floor` is the engine's `row + 1` off the same coord, 0 when
        // there is no map node under the player.
        act_floor: coord.map_or(0, |c| c.row + 1),
        hp: i(&run_state, "hp"),
        // `or 1`, matching `_base_obs` — and it is `_base_obs`'s value the
        // encoder reads (`ctx.max_hp`), not `_base_metrics`' `or 0`. The two
        // featurize identically because the encoder floors it at 1 anyway,
        // but emitting the same number keeps the observation dicts literally
        // comparable instead of merely equivalent.
        max_hp: match i(&run_state, "max_hp") {
            0 => 1,
            v => v,
        },
        gold: i(&run_state, "gold"),
        deck_size: deck_entries.len() as i64,
        num_upgraded,
        num_relics: relics.len() as i64,
        num_potions: potion_entries.len() as i64,

        card_reward_option_ids: None,
        card_reward_option_upgraded: None,
        card_reward_option_enchantments: None,
        card_reward_bundles: None,
        relic_reward_ids: None,
        boss_relic_options: None,
        smith_options: None,
        smith_option_enchantments: None,
        card_removal_options: None,
        card_grant_options: None,
        card_removal_option_enchantments: None,
        card_grant_option_enchantments: None,
        transform_select_deck_indices: None,
        enchant_select_deck_indices: None,
        // GH #545: same rationale as the deck-mutation selects below — the
        // bridge has no enchant-pick screen, so this stays unset here too.
        pending_enchantment: None,
        // The deck-mutation select surfaces (GH #399) are engine-side: the
        // bridge never presents one, so every builder here leaves them unset
        // and the encoder reads them as the empty slice.
        upgrade_select_deck_indices: None,
        shop: None,
        // The bridge has no potion-swap screen: SpireMentor filters an
        // unclaimable `PotionReward` out of the reward buttons instead of
        // offering the discard-to-claim choice
        // (`spire-mentor-mod/SimScreenAdvice.cs:1755`), so no wire message ever
        // carries this. Left unset like the deck-mutation selects above.
        pending_potion_swap: None,
        event_name: None,
        event_options: None,
        rest_option_labels: None,
    }
}

// ---------------------------------------------------------------------------
// Surface builders
// ---------------------------------------------------------------------------

/// Build the decision for one bridge message, or `None` when the surface has
/// nothing to decide (no options, empty map, …) and the caller should fall
/// through to its own policy.
///
/// `surface` is the bridge `state["type"]`: `map_select`, `card_reward`,
/// `boss_relic`, `rest_site`, `event`, or `shop`. Anything else returns
/// `None`, matching `_PPO_NONCOMBAT_BUILDERS.get(...)` missing.
#[must_use]
pub fn build_decision(state: &Value, surface: &str) -> Option<WireDecision> {
    match surface {
        "map_select" => build_map_select(state),
        "card_reward" => build_card_reward(state),
        "boss_relic" => build_boss_relic(state),
        "rest_site" => build_rest_site(state),
        "event" => build_event(state),
        "shop" => build_shop(state),
        _ => None,
    }
}

/// Enabled options only, as `(original_index, value)` pairs — every builder
/// filters the same way and then needs the bridge index off the original.
fn enabled_options(state: &Value) -> Vec<&Value> {
    arr(state, "options")
        .iter()
        .filter(|o| enabled(o))
        .collect()
}

/// `map_select` → the "path" surface (`move` actions).
fn build_map_select(state: &Value) -> Option<WireDecision> {
    let nodes = arr(state, "nodes");
    if nodes.is_empty() {
        return None;
    }
    let obs = base_obs(state);
    let actions = nodes
        .iter()
        .map(|n| ActionDesc::Move { col: i(n, "col") })
        .collect();
    let responses = nodes
        .iter()
        .enumerate()
        .map(|(idx, n)| choose(bridge_index(n, idx)))
        .collect();
    Some(WireDecision {
        obs,
        actions,
        responses,
        labels: map_labels(nodes),
    })
}

/// Name each offered node the way it looks on screen: `"Elite (left)"`.
///
/// Columns increase left-to-right on the map, so ranking the offers by column
/// recovers exactly the words a player would use. The old spelling was the raw
/// graph coordinate (`Elite@(4,1)`), unreadable next to a map that shows no
/// coordinates at all.
fn map_labels(nodes: &[Value]) -> Vec<String> {
    let mut by_col: Vec<usize> = (0..nodes.len()).collect();
    by_col.sort_by_key(|&idx| i(&nodes[idx], "col"));

    let side_of = |idx: usize| -> Option<String> {
        let pos = by_col.iter().position(|&x| x == idx)?;
        match by_col.len() {
            2 => Some(["left", "right"][pos].to_owned()),
            3 => Some(["left", "middle", "right"][pos].to_owned()),
            n if n > 3 => Some(format!("#{} from left", pos + 1)),
            _ => None,
        }
    };

    nodes
        .iter()
        .enumerate()
        .map(|(idx, n)| {
            let raw = obj(n, "type").and_then(Value::as_str).unwrap_or("?");
            let name = pretty_room_type(raw);
            side_of(idx).map_or_else(|| name.clone(), |side| format!("{name} ({side})"))
        })
        .collect()
}

/// `RestSite` → `Rest Site` (the enum name is the only name we get).
fn pretty_room_type(raw: &str) -> String {
    let chars: Vec<char> = raw.chars().collect();
    let mut out = String::with_capacity(raw.len() + 2);
    for (idx, ch) in chars.iter().enumerate() {
        if idx > 0 && ch.is_uppercase() && !chars[idx - 1].is_uppercase() {
            out.push(' ');
        }
        out.push(*ch);
    }
    out
}

/// `card_reward` → the "card-pick" surface (`pick_card_reward`).
fn build_card_reward(state: &Value) -> Option<WireDecision> {
    let cards = arr(state, "cards");
    let can_skip = b(state, "can_skip");
    if cards.is_empty() && !can_skip {
        return None;
    }
    let mut obs = base_obs(state);
    obs.card_reward_option_ids = Some(
        cards
            .iter()
            .map(|c| card_id(c).unwrap_or_default())
            .collect(),
    );
    obs.card_reward_option_upgraded = Some(cards.iter().map(is_upgraded).collect());
    obs.card_reward_option_enchantments = Some(cards.iter().map(card_enchantment).collect());

    let mut actions = Vec::with_capacity(cards.len() + 1);
    let mut responses = Vec::with_capacity(cards.len() + 1);
    let mut labels = Vec::with_capacity(cards.len() + 1);
    for (idx, c) in cards.iter().enumerate() {
        actions.push(ActionDesc::PickCardReward {
            idx: Some(idx as i64),
        });
        responses.push(choose(idx as i64));
        labels.push(
            opt_s(c, "name")
                .or_else(|| opt_s(c, "id"))
                .unwrap_or_default(),
        );
    }
    if can_skip {
        // `idx: None` IS the skip action in the training surface.
        actions.push(ActionDesc::PickCardReward { idx: None });
        responses.push(serde_json::json!({ "action": "skip" }));
        labels.push("skip".to_owned());
    }
    Some(WireDecision {
        obs,
        actions,
        responses,
        labels,
    })
}

/// `boss_relic` → the "boss-relic" surface (`pick_boss_relic`).
fn build_boss_relic(state: &Value) -> Option<WireDecision> {
    let options = enabled_options(state);
    if options.is_empty() {
        return None;
    }
    let mut obs = base_obs(state);
    obs.boss_relic_options = Some(
        options
            .iter()
            .map(|o| {
                obj(o, "relic")
                    .and_then(|r| opt_s(r, "id"))
                    .unwrap_or_default()
            })
            .collect(),
    );
    Some(WireDecision {
        obs,
        actions: (0..options.len())
            .map(|idx| ActionDesc::PickBossRelic { idx: idx as i64 })
            .collect(),
        responses: options
            .iter()
            .enumerate()
            .map(|(idx, o)| choose(bridge_index(o, idx)))
            .collect(),
        labels: options
            .iter()
            .map(|o| {
                obj(o, "relic")
                    .and_then(|r| opt_s(r, "name"))
                    .or_else(|| opt_s(o, "label"))
                    .unwrap_or_default()
            })
            .collect(),
    })
}

/// The rest-site option vocabulary the net was trained on.
const REST_OPTION_LABELS: [&str; 2] = ["rest", "smith"];

/// `rest_site` → the "rest" surface (`rest_option`).
fn build_rest_site(state: &Value) -> Option<WireDecision> {
    let options = enabled_options(state);
    if options.is_empty() {
        return None;
    }
    let mut obs = base_obs(state);
    obs.rest_option_labels = Some(REST_OPTION_LABELS.iter().map(|&x| x.to_owned()).collect());
    Some(WireDecision {
        obs,
        actions: (0..options.len())
            .map(|idx| ActionDesc::RestOption { idx: idx as i64 })
            .collect(),
        responses: options
            .iter()
            .enumerate()
            .map(|(idx, o)| choose(bridge_index(o, idx)))
            .collect(),
        labels: options
            .iter()
            .map(|o| {
                opt_s(o, "label")
                    .or_else(|| opt_s(o, "id"))
                    .unwrap_or_default()
            })
            .collect(),
    })
}

/// `shop` → the "shop" surface.
///
/// `shop_buy_card`/`_relic`/`_potion` need an `idx` into the per-kind list the
/// training env exposed, not the bridge's flat option index — both are built
/// in one pass here.
fn build_shop(state: &Value) -> Option<WireDecision> {
    let options = enabled_options(state);
    if options.is_empty() {
        return None;
    }
    let run_state = state.get("run_state").cloned().unwrap_or(Value::Null);
    let mut obs = base_obs(state);

    // A full potion belt makes buy_potion a guaranteed no-op in the real game
    // (`PotionCmd.TryToProcure` → TooFull, no gold spent): the shop
    // re-presents unchanged and a deterministic policy picks the same potion
    // forever. Newer bridges withhold the slots; this covers older builds.
    let belt = arr(&run_state, "potions");
    let belt_full = !belt.is_empty() && !belt.iter().any(|p| b(p, "empty"));

    let deck_entries = arr(&run_state, "master_deck");
    let deck: Vec<(String, bool)> = deck_entries
        .iter()
        .map(|c| (s(c, "id"), is_upgraded(c)))
        .collect();
    // Positionally parallel to `deck`; see `wire_card_is_removable`.
    let deck_removable: Vec<bool> = deck_entries.iter().map(wire_card_is_removable).collect();

    let mut shop_cards: Vec<ShopCardObs> = Vec::new();
    let mut shop_relics: Vec<ShopEntryObs> = Vec::new();
    let mut shop_potions: Vec<ShopEntryObs> = Vec::new();
    let mut actions: Vec<ActionDesc> = Vec::new();
    let mut responses: Vec<Value> = Vec::new();
    let mut labels: Vec<String> = Vec::new();

    for (pos, o) in options.iter().enumerate() {
        let kind = s(o, "action");
        let bidx = bridge_index(o, pos);
        let item = obj(o, "item").cloned().unwrap_or(Value::Null);
        let price = i(o, "price");
        match kind.as_str() {
            "buy_card" => {
                shop_cards.push(ShopCardObs {
                    // `idx`/`on_sale`/`is_colorless` are structural fields of
                    // the training dict that the encoder never reads (verified
                    // against `encoder.rs`), so the bridge's silence about them
                    // costs no feature.
                    idx: shop_cards.len() as i64,
                    id: card_id(&item),
                    price,
                    on_sale: false,
                    is_colorless: false,
                    available: true,
                });
                actions.push(ActionDesc::ShopBuyCard {
                    idx: shop_cards.len() as i64 - 1,
                });
            }
            "buy_relic" => {
                shop_relics.push(ShopEntryObs {
                    idx: shop_relics.len() as i64,
                    id: opt_s(&item, "id"),
                    price,
                    available: true,
                });
                actions.push(ActionDesc::ShopBuyRelic {
                    idx: shop_relics.len() as i64 - 1,
                });
            }
            "buy_potion" => {
                if belt_full {
                    continue;
                }
                shop_potions.push(ShopEntryObs {
                    idx: shop_potions.len() as i64,
                    id: opt_s(&item, "id"),
                    price,
                    available: true,
                });
                actions.push(ActionDesc::ShopBuyPotion {
                    idx: shop_potions.len() as i64 - 1,
                });
            }
            "leave_shop" => actions.push(ActionDesc::LeaveShop),
            "remove_card" if !deck.is_empty() => {
                // The net was trained on N per-deck-card `shop_remove_card`
                // actions (deck_idx), not one opaque "removal service" button
                // (which encoded as a FIXED "none" token it cannot value).
                // Every candidate answers the bridge with the same option; the
                // follow-up card_select honours the deck_idx chosen here.
                for (deck_idx, (raw_id, _)) in deck.iter().enumerate() {
                    if !deck_removable[deck_idx] {
                        continue;
                    }
                    actions.push(ActionDesc::ShopRemoveCard {
                        deck_idx: deck_idx as i64,
                    });
                    responses.push(choose(bidx));
                    labels.push(format!("remove:{raw_id}"));
                }
                continue;
            }
            // Anything unrecognized degrades to a token the encoder turns into
            // a FIXED "none" — never a crash, never silently dropped from the
            // legal set.
            _ => actions.push(ActionDesc::Unknown),
        }
        responses.push(choose(bidx));
        labels.push(opt_s(o, "label").unwrap_or(kind));
    }

    let removal_cost = obj(state, "purge_cost")
        .and_then(Value::as_i64)
        .or_else(|| {
            options
                .iter()
                .find(|o| s(o, "action") == "remove_card")
                .map(|o| i(o, "price"))
        })
        .unwrap_or(0);

    obs.shop = Some(ShopObs {
        cards: shop_cards,
        relics: shop_relics,
        potions: shop_potions,
        removal_cost,
        removal_used: false,
    });
    Some(WireDecision {
        obs,
        actions,
        responses,
        labels,
    })
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

/// The `event_id`s of the act-start Ancients whose 1-of-3 relic offer IS the
/// act's boss-relic choice (Hive: Orobas/Pael/Tezcatara, `Acts/Hive.cs:26-28`;
/// Glory: Nonupeipe/Tanx/Vakuu, `Acts/Glory.cs:25-27`).
///
/// The real game presents it as a plain event dialog — the bridge's
/// `NChooseARelicSelection` handler never fires for it — but the net only
/// knows relic IDENTITY on the trained `boss_relic` surface. Routed via the
/// event surface the pick is positional: delegating just this decision in the
/// engine-world ablation collapsed the native 10/20 baseline to 3/20.
const BOSS_RELIC_ANCIENTS: &[&str] = &["OROBAS", "PAEL", "TEZCATARA", "NONUPEIPE", "TANX", "VAKUU"];

/// Wire event keys whose first `text_key` segment is not the event's registry
/// name: DollRoom's C# loc entry is `MR_STRUGGLES`, and TheArchitect's
/// continuation page keys on the bare option verb `PROCEED`.
const EVENT_KEY_ALIASES: &[(&str, &str)] =
    &[("MR_STRUGGLES", "DOLL_ROOM"), ("PROCEED", "THE_ARCHITECT")];

fn alias(key: &str) -> String {
    EVENT_KEY_ALIASES
        .iter()
        .find(|(from, _)| *from == key)
        .map_or_else(|| key.to_owned(), |(_, to)| (*to).to_owned())
}

/// `event` → the "event" surface (`event_choice`), or the `boss-relic`
/// surface for an act-start Ancient's relic offer.
fn build_event(state: &Value) -> Option<WireDecision> {
    if let Some(decision) = ancient_boss_relic_decision(state) {
        return Some(decision);
    }
    let mut offered = enabled_options(state);
    if offered.is_empty() {
        return None;
    }

    // The GH #300 no-suicide guard, counterpart of the engine-side
    // `veto_lethal_event_choice`. The event-blind PPO holds a p~0.99 prior for
    // SlipperyBridge's "Hold On" learned against the old single-step port;
    // against the faithful 3+N escalation it loops until death. Withhold
    // lethal options — unless every option is, in which case the pick stays
    // the policy's to make. That "unless" is the shipped game's own rule
    // (`AutoSlay.EventRoomHandler` keeps the filtered list only if non-empty).
    //
    // Filtering BEFORE inference (rather than vetoing after, as the Rust
    // run-policy arm does) preserves the net's ranking over the survivors.
    let hp = i(
        &state.get("run_state").cloned().unwrap_or(Value::Null),
        "hp",
    );
    let survivors: Vec<&Value> = offered
        .iter()
        .copied()
        .filter(|o| !option_is_lethal(o, hp))
        .collect();
    if !survivors.is_empty() {
        offered = survivors;
    }

    // A disabled option keeps its slot, as in the engine's option list: the
    // net reads an option by POSITION, so dropping one would shift every later
    // option onto another's slot (the Fake Merchant's throw/leave, GH #873).
    // It gets no action, and is never the pick.
    let options: Vec<&Value> = arr(state, "options")
        .iter()
        .filter(|o| !enabled(o) || offered.iter().any(|k| std::ptr::eq(*k, *o)))
        .collect();
    let choices: Vec<usize> = (0..options.len())
        .filter(|&idx| enabled(options[idx]))
        .collect();

    let mut obs = base_obs(state);
    let event_id = options
        .iter()
        .find_map(|o| opt_s(o, "event_id"))
        .unwrap_or_default();
    obs.event_name = Some(resolve_event_name(&options, &event_id));

    let option_labels: Vec<String> = options
        .iter()
        .map(|o| event_option_label(o, &event_id))
        .collect();
    obs.event_options = Some(
        options
            .iter()
            .zip(&option_labels)
            .map(|(o, label)| EventOptionObs {
                // The bridge has no `option_id`; the encoder only reads it for
                // identity it does not have here either.
                option_id: String::new(),
                label: label.clone(),
                // Placeholders resolved BEFORE the encoder sees the text: it
                // scrapes numbers out of this string, and an unresolved
                // `{HpLoss}` reads as no cost at all.
                description: resolve_option_vars(&strip_bbcode(&s(o, "description")), o),
                enabled: enabled(o),
                // Pass the engine's exact deltas through when the world
                // supplies them (GH #237); the encoder prefers these over its
                // regex and ignores a partial set.
                gold_delta: obj(o, "gold_delta").and_then(Value::as_i64),
                hp_delta: obj(o, "hp_delta").and_then(Value::as_i64),
                max_hp_delta: obj(o, "max_hp_delta").and_then(Value::as_i64),
                relics_gained: obj(o, "relics_gained").and_then(Value::as_i64),
                relics_lost: obj(o, "relics_lost").and_then(Value::as_i64),
                potions_gained: obj(o, "potions_gained").and_then(Value::as_i64),
                potions_lost: obj(o, "potions_lost").and_then(Value::as_i64),
                cards_gained: obj(o, "cards_gained").and_then(Value::as_i64),
                cards_lost: obj(o, "cards_lost").and_then(Value::as_i64),
                // The bridge carries no localization path/template values for
                // an event option (a `workbench` display concern neither
                // encoder reads) — same "no wire equivalent" treatment as
                // `option_id` above.
                ..EventOptionObs::default()
            })
            .collect(),
    );

    Some(WireDecision {
        obs,
        actions: choices
            .iter()
            .map(|&idx| ActionDesc::EventChoice { idx: idx as i64 })
            .collect(),
        responses: choices
            .iter()
            .map(|&idx| choose(bridge_index(options[idx], idx)))
            .collect(),
        labels: choices
            .iter()
            .map(|&idx| option_labels[idx].clone())
            .collect(),
    })
}

/// An act-start Ancient's relic-offer event → the trained `boss_relic`
/// surface.
///
/// Fires only on the live-bridge shape: at least two enabled options, every
/// one of them carrying a non-empty `relic_id` AND an `event_id` naming one of
/// the six act-start Ancients. The Ancient's other pages (dialogue, the
/// post-pick "Proceed") have an empty `relic_id` and stay on the event
/// surface, as do all ordinary events.
///
/// The gate is on **`event_id`**, not on the relic: the relics offered are
/// ordinary boss relics (`YUMMY_COOKIE`, `GOLDEN_COMPASS`, …), and it is the
/// *event* that identifies this as the act's boss-relic choice. Gating on the
/// relic id instead silently routed all 230 of these in the recorded corpus
/// back through the untrained event surface — the exact positional-pick
/// failure this function exists to prevent (caught by
/// `diag_ppo_wire_parity.py`, 2026-08-09).
fn ancient_boss_relic_decision(state: &Value) -> Option<WireDecision> {
    let options = enabled_options(state);
    if options.len() < 2 {
        return None;
    }
    let relic_ids: Vec<String> = options
        .iter()
        .map(|o| opt_s(o, "relic_id").unwrap_or_default())
        .collect();
    if relic_ids.iter().any(String::is_empty) {
        return None;
    }
    if !options
        .iter()
        .all(|o| BOSS_RELIC_ANCIENTS.contains(&s(o, "event_id").to_ascii_uppercase().as_str()))
    {
        return None;
    }
    let mut obs = base_obs(state);
    obs.boss_relic_options = Some(relic_ids);
    Some(WireDecision {
        obs,
        actions: (0..options.len())
            .map(|idx| ActionDesc::PickBossRelic { idx: idx as i64 })
            .collect(),
        responses: options
            .iter()
            .enumerate()
            .map(|(idx, o)| choose(bridge_index(o, idx)))
            .collect(),
        // `_build_boss_relic` labels from `relic.name`, which the synthetic
        // option sets to the bridge option's own label.
        labels: options
            .iter()
            .map(|o| opt_s(o, "label").unwrap_or_default())
            .collect(),
    })
}

/// Whether picking `option` ends the run at `hp` current HP.
///
/// One rule with two sources, and the order is the point: the world's own
/// `will_kill_player` verdict WINS when present, and the HP-cost parse is
/// consulted only in its absence. Running them as independent filters inverts
/// the authority — the blind parse could veto an option the world had just
/// called survivable (reachable on `AbyssalBaths`, whose text says 3 while its
/// real threshold is 1).
///
/// A missing flag means UNKNOWN, never SAFE.
fn option_is_lethal(option: &Value, hp: i64) -> bool {
    if let Some(flag) = option.get("will_kill_player").and_then(Value::as_bool) {
        return flag;
    }
    hp > 0 && event_option_hp_cost(option) >= hp
}

/// The HP-cost phrase in an event option's own description — either a literal
/// number (`"Take 3 damage, reroll card."`) or an unresolved loc placeholder
/// (`"Lose [red]{HpLoss}[/red] HP"`) whose value lives in the option's `vars`.
///
/// Verbatim from `realgame_ppo._RE_EVENT_HP_COST`. The leading `lose|take` is
/// load-bearing, not decoration: without it `"Heal 20 HP"` parses as a 20-HP
/// *cost* and the guard withholds a heal from a low-HP player — which is
/// exactly what a hand-rolled token scan did here before
/// `diag_ppo_wire_parity.py` caught it against this regex (2026-08-09).
fn hp_cost_re() -> &'static regex::Regex {
    static RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| {
        regex::Regex::new(r"(?i)(?:lose|take)\s+(?:\{(\w+)\}|(\d+))\s+(?:hp|damage)")
            .expect("static regex")
    })
}

/// Best-effort HP an event option costs on pick; 0 when unparseable.
///
/// Reads the description **before** placeholder resolution and resolves the
/// captured `{Var}` from `vars` itself, matching the Python. The bridge
/// attaches the EVENT-level dynamic vars to every option, so the description
/// is what ties a value to the option that actually pays it.
fn event_option_hp_cost(option: &Value) -> i64 {
    let text = strip_bbcode(&s(option, "description"));
    let Some(m) = hp_cost_re().captures(&text) else {
        return 0;
    };
    if let Some(literal) = m.get(2) {
        return literal.as_str().parse().unwrap_or(0);
    }
    let Some(var_name) = m.get(1) else { return 0 };
    obj(option, "vars")
        .and_then(|v| v.get(var_name.as_str()))
        .and_then(|v| match v {
            Value::String(x) => x.parse::<i64>().ok(),
            other => other.as_i64(),
        })
        .unwrap_or(0)
}

/// Strip Godot BBCode from a bridge description.
///
/// `[img]…[/img]` blocks go with their content (icon references); plain
/// `[tag]`/`[/tag]` markers are removed and their text kept (`[red]6[/red]` →
/// `6`). The native engine's descriptions are clean prose and the encoder's
/// gold/HP regexes were trained against that — markup mid-string made every
/// live event payload parse to (0, 0).
fn strip_bbcode(text: &str) -> String {
    // `[img]…[/img]` first, so its inner tags are not stripped into stray text.
    let mut without_img = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("[img]") {
        without_img.push_str(&rest[..start]);
        let Some(end) = rest[start..].find("[/img]") else {
            // Unterminated: drop the remainder, matching the `re.S` match
            // failing and the tag pass removing the opener.
            rest = "";
            break;
        };
        rest = &rest[start + end + "[/img]".len()..];
    }
    without_img.push_str(rest);

    let mut out = String::with_capacity(without_img.len());
    let mut chars = without_img.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch != '[' {
            out.push(ch);
            continue;
        }
        // A tag is `[` `/`? letter-or-underscore, then anything but brackets,
        // then `]` — mirrors `_RE_BBCODE_TAG`. Anything else is literal text.
        let mut body = String::new();
        let mut closed = false;
        for c in chars.by_ref() {
            if c == ']' {
                closed = true;
                break;
            }
            if c == '[' {
                break;
            }
            body.push(c);
        }
        let name = body.strip_prefix('/').unwrap_or(&body);
        let looks_like_tag = name
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == '_');
        if !(closed && looks_like_tag) {
            out.push('[');
            out.push_str(&body);
            if closed {
                out.push(']');
            }
        }
    }
    out
}

/// Substitute an option's `vars` into its description's `{Name}` placeholders.
///
/// The bridge ships descriptions with the loc template intact (`"Lose {HpLoss}
/// HP"`) and the value alongside in `vars`. The encoder regex-scrapes NUMBERS
/// out of this string, so an unresolved placeholder reads as no cost at all —
/// which is a large part of why the net scored SlipperyBridge's "Hold On" as
/// free. An unknown placeholder is left as-is, so a parse failure still looks
/// like one downstream.
fn resolve_option_vars(text: &str, option: &Value) -> String {
    let Some(vars) = obj(option, "vars").and_then(Value::as_object) else {
        return text.to_owned();
    };
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find('{') {
        out.push_str(&rest[..start]);
        let after = &rest[start + 1..];
        let Some(end) = after.find('}') else {
            out.push('{');
            rest = after;
            continue;
        };
        let name = &after[..end];
        let is_word =
            !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
        if let Some(v) = vars.get(name).filter(|_| is_word) {
            out.push_str(&json_scalar_to_string(v));
        } else {
            // Leave an unknown placeholder intact, so a parse failure still
            // looks like a parse failure downstream.
            out.push('{');
            out.push_str(name);
            out.push('}');
        }
        rest = &after[end + 1..];
    }
    out.push_str(rest);
    out
}

/// `str(value)` for the scalar shapes a `vars` entry can hold — a bare number
/// or string, never the quoted JSON rendering.
fn json_scalar_to_string(v: &Value) -> String {
    match v {
        Value::String(x) => x.clone(),
        other => other.to_string(),
    }
}

/// Native-matching option label for one bridge event option.
///
/// The native engine labels NEOW's options with the offered blessing's
/// SCREAMING_SNAKE id, which is the tail of the C# option's `TextKey` — and
/// the PPO encoder registry-matches those enum-form labels to relic identity.
/// Every other event is labelled with prose natively, so the substitution is
/// Neow-scoped: swapping prose out elsewhere would break the gold/HP parsing.
fn event_option_label(o: &Value, event_id: &str) -> String {
    let label = strip_bbcode(&s(o, "label"));
    if event_id.to_ascii_uppercase().replace('.', "_") != "NEOW" {
        return label;
    }
    let tail = s(o, "text_key");
    let tail = tail.rsplit('.').next().unwrap_or("");
    if is_enum_tail(tail) {
        return tail.to_owned();
    }
    let relic_id = s(o, "relic_id");
    if is_enum_tail(&relic_id) {
        return relic_id;
    }
    label
}

/// `^[A-Z0-9_]{3,}$` — the enum-form loc-key tail shape the encoder gates on.
fn is_enum_tail(s: &str) -> bool {
    s.len() >= 3
        && s.chars()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
}

/// Map wire event identity onto the net's event-vocab key.
///
/// `vocab::events()` is keyed on `EventId::python_name()` (canonicalized to
/// UPPER_SNAKE). The wire gives either the C# loc entry (SCREAMING_SNAKE — the
/// `event_id`, and the first segment of every option `text_key`) or the
/// English display name (`"Tablet of Truth"`). The display name canonicalizes
/// with SPACES and never matches, which is why 72% of DLL event decisions fed
/// the OOV slot before this resolver existed.
///
/// Candidates are tried in wire-authority order; the first that lands in the
/// vocab wins. If none do, the first candidate is returned so a genuine
/// unknown still reads as OOV downstream.
fn resolve_event_name(options: &[&Value], display_name: &str) -> String {
    let mut candidates: Vec<String> = Vec::new();
    let mut add = |raw: &str| {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return;
        }
        let stripped = trimmed.strip_prefix("THE_").unwrap_or(trimmed).to_owned();
        for c in [trimmed.to_owned(), stripped, format!("THE_{trimmed}")] {
            if !c.is_empty() && !candidates.contains(&c) {
                candidates.push(c);
            }
        }
    };

    for o in options {
        let key = s(o, "text_key");
        let seg = key
            .split('.')
            .next()
            .unwrap_or("")
            .trim()
            .to_ascii_uppercase();
        add(&alias(&seg));
    }
    let snake_display: String = display_name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    let snake_display = snake_display.trim_matches('_').to_ascii_uppercase();
    add(&alias(&snake_display));

    if candidates.is_empty() {
        return display_name.to_owned();
    }
    let registry = vocab::events();
    for c in &candidates {
        if registry.index(c) != registry.oov {
            return c.clone();
        }
    }
    candidates[0].clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn run_state() -> Value {
        json!({
            "hp": 54, "max_hp": 80, "gold": 137, "ascension": 7,
            "master_deck": [
                {"id": "STRIKE_IRONCLAD", "upgrade_level": 0},
                {"id": "DEFEND_IRONCLAD", "upgraded": true},
                {"id": "RAGE", "upgrade_level": 0}
            ],
            "relics": [{"id": "BURNING_BLOOD"}],
            "potions": [{"id": "ATTACK_POTION"}, {"empty": true}]
        })
    }

    fn map_state(current_row: i64) -> Value {
        json!({
            "type": "map_select", "floor": 5, "act": 1,
            "run_state": run_state(),
            "nodes": [
                {"col": 3, "row": 2, "type": "Monster", "index": 0, "children": []},
                {"col": 1, "row": 2, "type": "RestSite", "index": 1, "children": []}
            ],
            "map": {
                "nodes": [
                    {"col": 3, "row": 2, "type": "Monster", "children": []},
                    {"col": 1, "row": 2, "type": "RestSite", "children": []}
                ],
                "current": {"col": 2, "row": current_row}
            }
        })
    }

    /// The row-0 guard: the DLL reports the pre-act Neow node as `current`,
    /// but the act graph never contains row 0. Feeding it through flipped 27
    /// of 42 reproduced act-start routes, so it must read as "no coord".
    #[test]
    fn a_row_zero_current_node_is_suppressed() {
        let d = build_decision(&map_state(0), "map_select").expect("decision");
        assert_eq!(d.obs.map_coord, None);
        assert_eq!(d.obs.act_floor, 0, "no coord means act_floor 0");

        let d = build_decision(&map_state(2), "map_select").expect("decision");
        assert_eq!(d.obs.map_coord, Some((2, 2)));
        assert_eq!(d.obs.act_floor, 3, "act_floor is row + 1");
    }

    /// `total_floor` is the DLL floor minus one and never negative; the engine
    /// is 0 at Neow where the DLL says 1.
    #[test]
    fn total_floor_is_offset_and_clamped() {
        let mut s = map_state(2);
        s["floor"] = json!(1);
        assert_eq!(build_decision(&s, "map_select").unwrap().obs.total_floor, 0);
        s["floor"] = json!(0);
        assert_eq!(build_decision(&s, "map_select").unwrap().obs.total_floor, 0);
        s["floor"] = json!(12);
        assert_eq!(
            build_decision(&s, "map_select").unwrap().obs.total_floor,
            11
        );
    }

    /// Token-for-token agreement with `realgame_ppo._vocab_card_id`, checked
    /// against it directly (2026-08-09): `RAGE` stays bare because
    /// [`vocab::canonical`] strips `_CARD` and it already hits `RAGE_CARD`'s
    /// slot, while `SLOTH` genuinely misses until `_STATUS` is tried.
    #[test]
    fn card_ids_take_the_registry_suffix_only_when_they_miss() {
        assert_eq!(vocab_card_id("SLOTH"), "SLOTH_STATUS");
        assert_eq!(vocab_card_id("RAGE"), "RAGE");
        assert_eq!(vocab_card_id("BASH"), "BASH");
        assert_eq!(vocab_card_id("STRIKE_IRONCLAD"), "STRIKE_IRONCLAD");
        // A genuine unknown degrades to itself rather than crashing.
        assert_eq!(vocab_card_id("NOT_A_CARD_AT_ALL"), "NOT_A_CARD_AT_ALL");
    }

    #[test]
    fn base_obs_counts_ignore_empty_potion_slots() {
        let d = build_decision(&map_state(2), "map_select").expect("decision");
        assert_eq!(d.obs.num_potions, 1, "the empty belt slot is not a potion");
        assert_eq!(d.obs.potions, ["ATTACK_POTION"]);
        assert_eq!(d.obs.deck_size, 3);
        assert_eq!(d.obs.num_upgraded, 1);
        assert_eq!(d.obs.deck[2].0, "RAGE", "deck tokens are vocab ids");
        assert_eq!(d.obs.ascension, 7);
        assert_eq!(d.obs.hp, 54);
    }

    /// Columns increase left-to-right, so the offers are named the way a
    /// player would say them, not by graph coordinate.
    #[test]
    fn map_labels_read_left_to_right() {
        let d = build_decision(&map_state(2), "map_select").expect("decision");
        // node[0] is col 3 (the right one), node[1] is col 1 (the left one).
        assert_eq!(d.labels, ["Monster (right)", "Rest Site (left)"]);
        assert_eq!(
            d.actions,
            [ActionDesc::Move { col: 3 }, ActionDesc::Move { col: 1 }]
        );
    }

    #[test]
    fn an_empty_map_yields_no_decision() {
        let mut s = map_state(2);
        s["nodes"] = json!([]);
        assert!(build_decision(&s, "map_select").is_none());
    }

    #[test]
    fn card_reward_skip_is_a_none_index() {
        let s = json!({
            "type": "card_reward", "floor": 5, "act": 1, "run_state": run_state(),
            "cards": [{"id": "RAGE", "name": "Rage"}, {"id": "BASH", "name": "Bash"}],
            "can_skip": true
        });
        let d = build_decision(&s, "card_reward").expect("decision");
        assert_eq!(
            d.obs.card_reward_option_ids.as_deref(),
            Some(&["RAGE".to_owned(), "BASH".to_owned()][..])
        );
        assert_eq!(d.actions.len(), 3);
        assert_eq!(d.actions[2], ActionDesc::PickCardReward { idx: None });
        assert_eq!(d.labels, ["Rage", "Bash", "skip"]);
        assert_eq!(d.responses[2], json!({"action": "skip"}));
    }

    /// The real game pre-upgrades reward cards too, and it says so over both
    /// wire spellings. Dropping the flag here would put a SpireMentor-served
    /// champion on a different observation from the one it trained on the
    /// moment `reward_upgrade` graduates (GH #545).
    #[test]
    fn card_reward_options_carry_their_upgrade_flag() {
        let s = json!({
            "type": "card_reward", "floor": 5, "act": 2, "run_state": run_state(),
            "cards": [
                {"id": "RAGE", "name": "Rage"},
                {"id": "BASH", "name": "Bash+", "upgrade_level": 1},
                {"id": "CLEAVE", "name": "Cleave+", "upgraded": true},
            ],
            "can_skip": false
        });
        let d = build_decision(&s, "card_reward").expect("decision");
        assert_eq!(
            d.obs.card_reward_option_upgraded.as_deref(),
            Some(&[false, true, true][..])
        );
    }

    /// The lethal-option filter runs BEFORE inference, and the world's own
    /// `will_kill_player` outranks the description parse.
    #[test]
    fn lethal_event_options_are_withheld_unless_all_are() {
        let base = |kill: Value, desc: &str| {
            json!({
                "type": "event", "floor": 5, "act": 1, "run_state": run_state(),
                "options": [
                    {"index": 0, "label": "Safe", "description": "Gain 30 Gold",
                     "text_key": "SLIPPERY_BRIDGE.a", "event_id": "SLIPPERY_BRIDGE"},
                    {"index": 1, "label": "Deadly", "description": desc,
                     "will_kill_player": kill,
                     "text_key": "SLIPPERY_BRIDGE.b", "event_id": "SLIPPERY_BRIDGE"}
                ]
            })
        };
        let d = build_decision(&base(json!(true), "Lose 1 HP"), "event").expect("decision");
        assert_eq!(
            d.labels,
            ["Safe"],
            "the lethal option never reaches the net"
        );

        // The flag outranks the parse: a huge printed cost the world calls
        // survivable stays on the table.
        let d = build_decision(&base(json!(false), "Lose 999 HP"), "event").expect("decision");
        assert_eq!(d.labels.len(), 2);

        // No flag at all, so the parse decides, and 999 >= 54 hp is lethal.
        let mut s = base(json!(null), "Lose 999 HP");
        s["options"][1]
            .as_object_mut()
            .unwrap()
            .remove("will_kill_player");
        let d = build_decision(&s, "event").expect("decision");
        assert_eq!(d.labels, ["Safe"]);
    }

    /// GH #873: the Fake Merchant always sends the engine's three options;
    /// one that is not available stays in its slot, so leave is still read
    /// as option 2 and never as the throw.
    #[test]
    fn a_disabled_event_option_keeps_its_slot() {
        let opt = |index: i64, key: &str, enabled: bool, relics: i64, potions: i64| {
            json!({
                "index": index, "label": key, "description": "", "enabled": enabled,
                "event_id": "FAKE_MERCHANT", "text_key": format!("FAKE_MERCHANT.options.{key}"),
                "gold_delta": if relics > 0 { -50 } else { 0 }, "hp_delta": 0, "max_hp_delta": 0,
                "relics_gained": relics, "potions_lost": potions
            })
        };
        let s = json!({
            "type": "event", "floor": 20, "act": 2, "run_state": run_state(),
            "options": [
                opt(0, "BUY", true, 1, 0),
                opt(1, "THROW_FOUL", false, 0, 1),
                opt(2, "LEAVE", true, 0, 0)
            ]
        });
        let d = build_decision(&s, "event").expect("decision");
        assert_eq!(d.obs.event_name.as_deref(), Some("FAKE_MERCHANT"));
        let opts = d.obs.event_options.as_ref().expect("options");
        assert_eq!(opts.len(), 3, "the disabled throw is still observed");
        assert!(!opts[1].enabled);
        assert_eq!(opts[0].relics_gained, Some(1));
        assert_eq!(opts[1].potions_lost, Some(1));
        assert!(matches!(
            d.actions.as_slice(),
            [
                ActionDesc::EventChoice { idx: 0 },
                ActionDesc::EventChoice { idx: 2 }
            ]
        ));
        assert_eq!(d.labels, ["BUY", "LEAVE"]);
        assert_eq!(d.responses, [choose(0), choose(2)]);
    }

    #[test]
    fn every_option_lethal_leaves_the_pick_to_the_policy() {
        let s = json!({
            "type": "event", "floor": 5, "act": 1, "run_state": run_state(),
            "options": [
                {"index": 0, "label": "A", "will_kill_player": true, "description": ""},
                {"index": 1, "label": "B", "will_kill_player": true, "description": ""}
            ]
        });
        let d = build_decision(&s, "event").expect("decision");
        assert_eq!(
            d.labels.len(),
            2,
            "withholding all of them would decide nothing"
        );
    }

    /// The encoder scrapes numbers out of the description, so BBCode must go
    /// and the loc placeholder must be resolved before it ever sees the text.
    #[test]
    fn event_descriptions_are_stripped_and_resolved() {
        let s = json!({
            "type": "event", "floor": 5, "act": 1, "run_state": run_state(),
            "options": [{
                "index": 0, "label": "[b]Hold On[/b]",
                "description": "[img]res://icon.png[/img]Lose [red]{HpLoss}[/red] HP",
                "vars": {"HpLoss": 6},
                "text_key": "SLIPPERY_BRIDGE.hold", "event_id": "SLIPPERY_BRIDGE",
                "will_kill_player": false
            }]
        });
        let d = build_decision(&s, "event").expect("decision");
        let opts = d.obs.event_options.as_ref().expect("event options");
        assert_eq!(opts[0].description, "Lose 6 HP");
        assert_eq!(opts[0].label, "Hold On");
    }

    #[test]
    fn an_unknown_placeholder_survives_intact() {
        let o = json!({"vars": {"Known": 3}});
        assert_eq!(
            resolve_option_vars("Lose {Unknown} HP, gain {Known}", &o),
            "Lose {Unknown} HP, gain 3"
        );
    }

    /// DollRoom's loc entry is `MR_STRUGGLES`; without the alias the whole
    /// event fed the OOV slot.
    #[test]
    fn event_names_resolve_through_aliases_and_the_vocab() {
        let opts = [json!({"text_key": "MR_STRUGGLES.page1"})];
        let refs: Vec<&Value> = opts.iter().collect();
        assert_eq!(resolve_event_name(&refs, "Doll Room"), "DOLL_ROOM");
    }

    /// The display-name dialect canonicalizes with spaces and never matched
    /// the registry: 72% of DLL event decisions went OOV before the resolver.
    #[test]
    fn a_display_name_still_reaches_the_registry() {
        let opts: Vec<Value> = vec![];
        let refs: Vec<&Value> = opts.iter().collect();
        let got = resolve_event_name(&refs, "Tablet of Truth");
        assert!(
            !got.contains(' '),
            "a spaced display name can never match the vocab: {got}"
        );
    }

    /// NEOW's options are labelled natively with the blessing's enum id, which
    /// is what the encoder registry-matches to relic identity.
    #[test]
    fn neow_options_are_labelled_by_enum_tail() {
        let o = json!({
            "label": "Take the boots",
            "text_key": "NEOW.pages.INITIAL.options.WINGED_BOOTS"
        });
        assert_eq!(event_option_label(&o, "NEOW"), "WINGED_BOOTS");
        // Every other event keeps its prose: the gold/HP parse depends on it.
        assert_eq!(event_option_label(&o, "DOLL_ROOM"), "Take the boots");
    }

    /// An act-start Ancient's relic offer is routed to the trained
    /// `boss_relic` surface; via the event surface the pick is positional.
    #[test]
    fn an_ancient_relic_offer_becomes_a_boss_relic_decision() {
        // The gate is the EVENT id; the relics offered are ordinary boss
        // relics. Gating on the relic instead sent all 230 of these in the
        // recorded corpus down the untrained event surface.
        let s = json!({
            "type": "event", "floor": 1, "act": 2, "run_state": run_state(),
            "options": [
                {"index": 0, "label": "Yummy Cookie", "relic_id": "YUMMY_COOKIE",
                 "event_id": "TEZCATARA"},
                {"index": 1, "label": "Biiig Hug", "relic_id": "BIIIG_HUG",
                 "event_id": "TEZCATARA"},
                {"index": 2, "label": "Seal of Gold", "relic_id": "SEAL_OF_GOLD",
                 "event_id": "TEZCATARA"}
            ]
        });
        // The routing shows up in the ACTIONS (and the relic-identity slot),
        // not in `obs.surface` — that field is `None` on both sides, and the
        // net reads the fine surface off the candidate list instead.
        let d = build_decision(&s, "event").expect("decision");
        assert_eq!(
            d.obs.boss_relic_options.as_deref(),
            Some(
                &[
                    "YUMMY_COOKIE".to_owned(),
                    "BIIIG_HUG".to_owned(),
                    "SEAL_OF_GOLD".to_owned()
                ][..]
            )
        );
        assert!(matches!(d.actions[0], ActionDesc::PickBossRelic { idx: 0 }));
        assert!(d.obs.event_options.is_none());
        assert_eq!(d.labels, ["Yummy Cookie", "Biiig Hug", "Seal of Gold"]);

        // An ordinary event stays an event, however boss-relic-ish its
        // options look — the gate is the EVENT id.
        let mut plain = s.clone();
        for idx in 0..3 {
            plain["options"][idx]["event_id"] = json!("SLIPPERY_BRIDGE");
        }
        let d = build_decision(&plain, "event").expect("decision");
        assert!(d.obs.boss_relic_options.is_none());
        assert!(matches!(d.actions[0], ActionDesc::EventChoice { idx: 0 }));

        // The Ancient's OTHER pages (dialogue, the post-pick Proceed) carry
        // no relic and stay on the event surface too.
        let mut dialogue = s.clone();
        dialogue["options"][2]["relic_id"] = json!("");
        let d = build_decision(&dialogue, "event").expect("decision");
        assert!(d.obs.boss_relic_options.is_none());
    }

    #[test]
    fn shop_actions_index_per_kind_lists() {
        let s = json!({
            "type": "shop", "floor": 5, "act": 1, "run_state": run_state(),
            "purge_cost": 75,
            "options": [
                {"index": 0, "action": "buy_card", "price": 50,
                 "item": {"id": "RAGE"}, "label": "Rage"},
                {"index": 1, "action": "buy_relic", "price": 150,
                 "item": {"id": "BURNING_BLOOD"}},
                {"index": 2, "action": "buy_card", "price": 60, "item": {"id": "BASH"}},
                {"index": 3, "action": "leave_shop"}
            ]
        });
        let d = build_decision(&s, "shop").expect("decision");
        let shop = d.obs.shop.as_ref().expect("shop");
        assert_eq!(shop.cards.len(), 2);
        assert_eq!(shop.cards[0].id.as_deref(), Some("RAGE"));
        assert_eq!(shop.relics.len(), 1);
        assert_eq!(shop.removal_cost, 75);
        assert_eq!(
            d.actions,
            [
                ActionDesc::ShopBuyCard { idx: 0 },
                ActionDesc::ShopBuyRelic { idx: 0 },
                ActionDesc::ShopBuyCard { idx: 1 },
                ActionDesc::LeaveShop,
            ],
            "idx counts within each kind, not the flat option list"
        );
    }

    /// A full belt makes `buy_potion` a guaranteed no-op, and a deterministic
    /// policy would pick it forever.
    #[test]
    fn a_full_potion_belt_hides_potion_buys() {
        let mut rs = run_state();
        rs["potions"] = json!([{"id": "ATTACK_POTION"}, {"id": "FEAR_POTION"}]);
        let s = json!({
            "type": "shop", "floor": 5, "act": 1, "run_state": rs,
            "options": [
                {"index": 0, "action": "buy_potion", "price": 50,
                 "item": {"id": "FIRE_POTION"}},
                {"index": 1, "action": "leave_shop"}
            ]
        });
        let d = build_decision(&s, "shop").expect("decision");
        assert_eq!(d.actions, [ActionDesc::LeaveShop]);
        assert_eq!(d.responses.len(), 1, "responses stay aligned with actions");
    }

    /// The removal service expands to one candidate per removable deck card,
    /// which in v0.107.1 means EVERY card that is not Eternal -- basics
    /// included.
    ///
    /// This assertion used to be `[ShopRemoveCard { deck_idx: 2 }]` with a
    /// comment saying basics "are excluded as they were at training". They
    /// were not: `Phase::Shop` has pushed one `ShopRemoveCard` per
    /// `removal_offer_order()` entry since 2026-06-19, so the champions
    /// trained WITH basics on the menu and the bridge filter was the thing
    /// introducing the mismatch it claimed to prevent.
    #[test]
    fn removal_expands_over_every_non_eternal_deck_card() {
        let s = json!({
            "type": "shop", "floor": 5, "act": 1, "run_state": run_state(),
            "options": [{"index": 0, "action": "remove_card", "price": 75}]
        });
        let d = build_decision(&s, "shop").expect("decision");
        // Deck is Strike/Defend/Rage; none is Eternal, so all three expand.
        assert_eq!(
            d.actions,
            [
                ActionDesc::ShopRemoveCard { deck_idx: 0 },
                ActionDesc::ShopRemoveCard { deck_idx: 1 },
                ActionDesc::ShopRemoveCard { deck_idx: 2 },
            ]
        );
        assert_eq!(
            d.labels,
            [
                "remove:STRIKE_IRONCLAD",
                "remove:DEFEND_IRONCLAD",
                "remove:RAGE"
            ]
        );
        assert_eq!(d.obs.shop.as_ref().unwrap().removal_cost, 75);
    }

    /// An Eternal card is never a removal candidate: the live bridge emitted
    /// `remove:ASCENDERS_BANE` shop picks that no `FromDeckForRemoval` screen
    /// could satisfy, which then dropped the carried target and let the
    /// follow-up `card_select` re-decide the removal.
    #[test]
    fn removal_withholds_eternal_cards() {
        let mut rs = run_state();
        rs["master_deck"] = json!([
            {"id": "STRIKE_IRONCLAD", "upgrade_level": 0},
            {"id": "ASCENDERS_BANE", "upgrade_level": 0},
        ]);
        let s = json!({
            "type": "shop", "floor": 5, "act": 1, "run_state": rs,
            "options": [{"index": 0, "action": "remove_card", "price": 75}]
        });
        let d = build_decision(&s, "shop").expect("decision");
        assert_eq!(d.labels, ["remove:STRIKE_IRONCLAD"]);
    }

    /// An enchantment can make a card Eternal, so the fan-out cannot filter
    /// on card id alone. `NutritiousSoup` puts `TezcatarasEmber` on every
    /// basic Strike, and `TezcatarasEmber.OnEnchant` adds
    /// `CardKeyword.Eternal` (`TezcatarasEmber.cs:18`) — after that pickup
    /// the removal screen withholds those Strikes.
    #[test]
    fn removal_withholds_a_card_an_enchantment_made_eternal() {
        let mut rs = run_state();
        rs["master_deck"] = json!([
            {"id": "STRIKE_IRONCLAD", "upgrade_level": 0, "enchantment": "TEZCATARAS_EMBER"},
            {"id": "STRIKE_IRONCLAD", "upgrade_level": 0},
            {"id": "BASH", "upgrade_level": 0, "enchantment": "GOOPY"},
        ]);
        let s = json!({
            "type": "shop", "floor": 5, "act": 1, "run_state": rs,
            "options": [{"index": 0, "action": "remove_card", "price": 75}]
        });
        let d = build_decision(&s, "shop").expect("decision");
        // The Ember'd Strike is out; the plain Strike and the Goopy'd Bash
        // stay (Goopy adds Exhaust, not Eternal).
        assert_eq!(d.labels, ["remove:STRIKE_IRONCLAD", "remove:BASH"]);
        assert_eq!(
            d.actions,
            [
                ActionDesc::ShopRemoveCard { deck_idx: 1 },
                ActionDesc::ShopRemoveCard { deck_idx: 2 },
            ],
            "deck_idx stays a master-deck index, so the skip must not renumber"
        );
    }

    /// A bridge build that omits `enchantment` must behave exactly as before.
    #[test]
    fn an_absent_or_unknown_enchantment_reads_as_removable() {
        let mut rs = run_state();
        rs["master_deck"] = json!([
            {"id": "BASH", "upgrade_level": 0},
            {"id": "ANGER", "upgrade_level": 0, "enchantment": "NOT_AN_ENCHANTMENT"},
        ]);
        let s = json!({
            "type": "shop", "floor": 5, "act": 1, "run_state": rs,
            "options": [{"index": 0, "action": "remove_card", "price": 75}]
        });
        let d = build_decision(&s, "shop").expect("decision");
        assert_eq!(d.labels, ["remove:BASH", "remove:ANGER"]);
    }

    #[test]
    fn disabled_options_never_reach_the_net() {
        let s = json!({
            "type": "rest_site", "floor": 5, "act": 1, "run_state": run_state(),
            "options": [
                {"index": 0, "label": "Rest", "enabled": true},
                {"index": 1, "label": "Smith", "enabled": false}
            ]
        });
        let d = build_decision(&s, "rest_site").expect("decision");
        assert_eq!(d.labels, ["Rest"]);
        assert_eq!(d.actions, [ActionDesc::RestOption { idx: 0 }]);
        // The trained label vocabulary is fixed, not derived from the options.
        assert_eq!(
            d.obs.rest_option_labels.as_deref(),
            Some(&["rest".to_owned(), "smith".to_owned()][..])
        );
    }

    #[test]
    fn an_unknown_surface_declines_to_decide() {
        assert!(build_decision(&map_state(2), "card_select").is_none());
        assert!(build_decision(&map_state(2), "combat_action").is_none());
    }

    #[test]
    fn bbcode_stripping_keeps_text_and_drops_images() {
        assert_eq!(strip_bbcode("[red]6[/red] HP"), "6 HP");
        assert_eq!(strip_bbcode("[img]res://x.png[/img]Gold"), "Gold");
        // Not a tag: a literal bracket stays put.
        assert_eq!(strip_bbcode("[3] apples"), "[3] apples");
        assert_eq!(strip_bbcode(""), "");
    }

    /// The `lose|take` verb is what separates a cost from a gain. Without it
    /// `"Heal 20 HP"` reads as a 20-HP cost and the no-suicide guard withholds
    /// a heal from the player who most needs it.
    #[test]
    fn hp_costs_need_a_lose_or_take_verb() {
        let cost = |desc: &str| event_option_hp_cost(&json!({"description": desc}));
        assert_eq!(cost("Lose 6 HP"), 6);
        assert_eq!(cost("Take 12 damage"), 12);
        assert_eq!(cost("Heal [green]20[/green] HP."), 0);
        assert_eq!(cost("Gain 30 Gold"), 0);
        assert_eq!(cost("Gain 30 Gold, lose 5 HP"), 5);
        // "Max HP" is not "HP" — the regex wants the unit right after the
        // number, so a max-HP line is not read as a current-HP cost.
        assert_eq!(cost("Lose [red]3[/red] Max HP. Upgrade a random card."), 0);
    }

    /// An unresolved placeholder is looked up in `vars` by the parser itself,
    /// on the pre-resolution description.
    #[test]
    fn hp_costs_resolve_a_placeholder_from_vars() {
        let opt = json!({
            "description": "Lose [red]{HpLoss}[/red] HP",
            "vars": {"HpLoss": 9}
        });
        assert_eq!(event_option_hp_cost(&opt), 9);
        // No `vars` entry: unknown, which reads as 0 rather than a guess.
        let opt = json!({"description": "Lose {HpLoss} HP"});
        assert_eq!(event_option_hp_cost(&opt), 0);
    }

    /// The bridge sends a flat node list; the training obs wants `rows`. One
    /// row carries the same graph because `nodes_from_full_map` only flattens.
    #[test]
    fn the_flat_node_list_becomes_a_single_row() {
        let d = build_decision(&map_state(2), "map_select").expect("decision");
        let map = d.obs.map.as_ref().expect("map");
        assert_eq!(map.rows.len(), 1);
        assert_eq!(map.rows[0].len(), 2);
        // Deliberately unset — see the note in `base_obs`; `obs.map_coord`
        // is the field the features come from.
        assert_eq!(map.current_coord, None);
        assert_eq!(d.obs.map_coord, Some((2, 2)));
    }
}
