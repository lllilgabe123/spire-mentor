//! Decision -> token arrays for the run-level policy net — a straight
//! Rust port of `sts2-pycli/python/ppo/encoder.py` (porting authority;
//! never modify that file). Schema constants mirror
//! `sts2-pycli/python/ppo/model.py` verbatim (never modify that file
//! either — the drift guard is `tests/ppo_encoder_parity.rs`'s
//! committed-fixture Gate A1, which requires bit-for-bit continuous
//! output and exact categorical/section output on every row).
//!
//! Correspondence notes (see the Task 6 brief for the full 7-point
//! contract this module honors point-by-point):
//! - Every continuous value is computed in `f64` and cast to `f32` only
//!   at the point a token's `cont` array is written into a
//!   [`Featurized`] vector ([`to_f32_arr`]) — numpy builds its `f32`
//!   arrays from Python `float` (f64) scalars the same way, so casting
//!   any earlier would round differently and break bit-equality.
//! - `event_payload`'s regexes are ported 1:1 (`regex` crate, `(?i)`
//!   flags where Python uses `re.I`); none of them need lookaround, so
//!   unlike `vocab::canonical` this port uses the `regex` crate
//!   directly rather than a hand-rolled char scan.
//! - The Rust `featurize` signature takes only `(obs, actions)`, not a
//!   third `metrics` dict like Python's `featurize_row` — `env.py`'s
//!   `observe()` merges `metrics_snapshot()` into the same obs dict
//!   `_advance` later re-derives `metrics` from at the same instant, so
//!   the state token's scalar fields (`hp`, `gold`, `act_index`, …) are
//!   read straight off [`PpoObs`]'s own fields. The committed parity
//!   fixture is the proof this equivalence holds on every row.

use std::cell::Cell;
use std::sync::OnceLock;

use regex::Regex;

use super::features::Features;
use super::obs::{ActionDesc, CardEnchantObs, EventOptionObs, PpoObs, ShopCardObs, ShopEntryObs};
use super::{mapfeat, vocab};

// ── schema constants (mirror `model.py` verbatim) ───────────────────────

pub const SEC_STATE: i64 = 0;
pub const SEC_DECK: i64 = 1;
pub const SEC_RELIC: i64 = 2;
pub const SEC_POTION: i64 = 3;
pub const SEC_MAPNODE: i64 = 4;
pub const SEC_PATH: i64 = 5;
pub const SEC_CARD: i64 = 6;
pub const SEC_FIXED: i64 = 7;
pub const SEC_RELIC_OPT: i64 = 8;
pub const SEC_POTION_OPT: i64 = 9;
pub const SEC_EVENT_OPT: i64 = 10;
pub const N_SECTION: i64 = 11;

pub const N_CAT: usize = 12;
/// The act-boss identity's categorical column, mirroring `schema.py::CAT_BOSS`.
/// Fixed at 9 regardless of `N_CAT` — it was the trailing column only while
/// `N_CAT == 10`; GH #545 appended `deck_op`/`enchant` after it, so anything
/// that wants "the boss column" must name this constant rather than derive
/// it from the width (`N_CAT - 1` silently became the enchant column).
pub const CAT_BOSS: usize = 9;
pub const N_CONT: usize = 26;
pub const GOLD_SCALE: f64 = 0.01;
pub const DECK_SCALE: f64 = 0.05;
pub const FLOOR_SCALE: f64 = 1.0 / 45.0;
pub const ACT_FLOOR_SCALE: f64 = 0.05;
/// Ascension level * 0.1 (A10 → 1.0); the difficulty input (`model.py`'s
/// `ASC_SCALE`). Written only on the state token, in the appended slot
/// [`C_ASC`], so an 18-cont champion drops it cleanly (see [`super::net`]).
pub const ASC_SCALE: f64 = 0.1;
/// Aux dest-room-label sentinel (`model.py::AUX_IGNORE`). The aux head
/// is excluded from this port's `Featurized` (ONNX export drops it —
/// see the plan's Design §7), so this constant is unused here but kept
/// for schema-doc parity with `model.py`.
#[allow(dead_code)]
pub const AUX_IGNORE: i64 = -100;

// Continuous indices (see `model.py`'s module docstring).
pub const C_HP: usize = 0;
pub const C_GOLD: usize = 1;
pub const C_ACT: usize = 2;
pub const C_DECKSZ: usize = 3;
pub const C_NUPG: usize = 4;
pub const C_NREL: usize = 5;
pub const C_DX: usize = 6;
pub const C_DY: usize = 7;
pub const C_REACH: usize = 8;
pub const C_MINE: usize = 9;
pub const C_MAXE: usize = 10;
pub const C_DREST: usize = 11;
pub const C_PRICE: usize = 12;
pub const C_GOLD_D: usize = 13;
pub const C_HP_D: usize = 14;
pub const C_ENABLED: usize = 15;
pub const C_FLOOR: usize = 16;
pub const C_ACTFLOOR: usize = 17;
/// Ascension level (state token only); appended slot ⇒ warm-start safe and
/// dropped cleanly for an 18-cont champion (see [`super::net`]).
pub const C_ASC: usize = 18;
/// What an event option costs in things that are not gold or HP. Appended
/// slots ⇒ warm-start safe; a 19-cont champion drops them (see
/// [`super::net`]). GAINED and LOST are separate so a TRADE does not net to
/// zero and read as "nothing happens" — see `schema.py`'s module docstring.
pub const C_RELIC_GAIN: usize = 19;
pub const C_RELIC_LOSS: usize = 20;
pub const C_POTION_GAIN: usize = 21;
pub const C_POTION_LOSS: usize = 22;
pub const C_CARD_GAIN: usize = 23;
pub const C_CARD_LOSS: usize = 24;
/// The card's own enchantment amount (GH #947), on card and deck tokens,
/// scaled by [`ENCH_AMT_SCALE`]. Written only under the `card_mods` feature;
/// appended slot ⇒ a 25-cont champion drops it (see [`super::net`]).
pub const C_ENCH_AMT: usize = 25;
/// Enchantment amount * 0.1: stacks run 1..~10 (Sharp / Nimble carry their
/// payload in the amount), so this keeps the slot ~O(1).
pub const ENCH_AMT_SCALE: f64 = 0.1;

/// One token's categorical id row (`[card, upg, relic, potion, room,
/// event, fixed, optidx, surface, boss, deck_op, enchant]`).
pub type Cat = [i64; N_CAT];
/// One token's continuous feature row, computed in `f64` (cast to
/// `f32` at the point it's written into a [`Featurized`] vector).
type ContF64 = [f64; N_CONT];

/// Per-decision token arrays. Mirrors `model.py::Featurized` minus the
/// `aux_room` field (the aux head is excluded from this port — see the
/// module doc comment).
#[derive(Debug, Clone, PartialEq)]
pub struct Featurized {
    pub obs_cat: Vec<Cat>,
    pub obs_cont: Vec<[f32; N_CONT]>,
    pub obs_sec: Vec<i64>,
    pub act_cat: Vec<Cat>,
    pub act_cont: Vec<[f32; N_CONT]>,
    pub act_sec: Vec<i64>,
    pub n_actions: usize,
}

/// Pad-row categorical ids — mirrors `model.py::empty_cat()`
/// (`pad_indices()`'s values in `[card, upg, relic, potion, room,
/// event, fixed, optidx, surface, boss, deck_op, enchant]` order).
#[must_use]
pub fn empty_cat() -> Cat {
    [
        vocab::cards().size as i64,
        vocab::MAX_UPGRADE + 1,
        vocab::relics().size as i64,
        vocab::potions().size as i64,
        vocab::N_ROOM as i64,
        vocab::events().size as i64,
        vocab::N_FIXED as i64,
        vocab::MAX_OPT_IDX as i64,
        vocab::N_SURFACE as i64,
        vocab::bosses().size as i64,
        vocab::N_DECK_OP as i64,
        vocab::enchantments().size as i64,
    ]
}

fn to_f32_arr(a: ContF64) -> [f32; N_CONT] {
    a.map(|v| v as f32)
}

// ── event payload parsing ────────────────────────────────────────────────

fn re_enum_label() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^[A-Z0-9_]{3,}$").expect("valid regex"))
}

fn re_gold() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)(gain|receive|lose|pay|give|offer)\s+(?:up to\s+)?(\d+)\s*gold")
            .expect("valid regex")
    })
}

fn re_dmg() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)(?:take|lose|suffer)\s+(\d+)\s*(?:damage|hp)").expect("valid regex")
    })
}

fn re_heal() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)heal\s+(?:for\s+)?(\d+)").expect("valid regex"))
}

fn re_pct_hp() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)(?:lose|take)\s+(\d+)%\s*(?:of\s+your\s+)?(?:max(?:imum)?\s+)?hp")
            .expect("valid regex")
    })
}

fn re_maxhp() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)(gain|lose)\s+(\d+)\s*max(?:imum)?\s+hp").expect("valid regex")
    })
}

/// `(gold_delta, hp_delta_frac)` parsed from an option label/description.
/// Conservative: unmatched -> zeros. Signs: gain +, lose/pay -. HP
/// deltas are fractions of `max_hp`; percent-HP costs resolve directly.
/// Mirrors `encoder.py::event_payload` — same accumulation order (gold
/// verbs -> damage -> heal -> maxhp -> hp_frac -> pct-hp), then a final
/// clamp to `[-1, 1]`.
#[must_use]
pub fn event_payload(text: &str, max_hp: i64) -> (f64, f64) {
    let mut gold = 0.0f64;
    let mut hp = 0.0f64;
    for cap in re_gold().captures_iter(text) {
        let verb = cap[1].to_ascii_lowercase();
        let amt: f64 = cap[2].parse().expect("digits");
        let sign = if verb == "gain" || verb == "receive" {
            1.0
        } else {
            -1.0
        };
        gold += sign * amt;
    }
    for cap in re_dmg().captures_iter(text) {
        let amt: f64 = cap[1].parse().expect("digits");
        hp -= amt;
    }
    for cap in re_heal().captures_iter(text) {
        let amt: f64 = cap[1].parse().expect("digits");
        hp += amt;
    }
    for cap in re_maxhp().captures_iter(text) {
        let verb = cap[1].to_ascii_lowercase();
        let amt: f64 = cap[2].parse().expect("digits");
        hp += if verb == "gain" { 1.0 } else { -1.0 } * amt;
    }
    let mut hp_frac = hp / max_hp.max(1) as f64;
    for cap in re_pct_hp().captures_iter(text) {
        let pct: f64 = cap[1].parse().expect("digits");
        hp_frac -= pct / 100.0;
    }
    (gold, hp_frac.clamp(-1.0, 1.0))
}

/// `(gold_delta, hp_delta_frac)` from an option's STRUCTURED engine
/// deltas (GH #237), when present — the exact numbers the engine
/// computed, with no description scraping. Returns `None` when the
/// option carries no structured payload (pre-#237 obs), so the caller
/// falls back to [`event_payload`]. Mirrors
/// `encoder.py::structured_event_payload`.
///
/// `hp_delta_frac` folds current-HP and max-HP changes into one signed
/// fraction of `max_hp`: a max-HP GAIN already bumps current HP (engine
/// contract) so it is counted once via `hp_delta`; a max-HP LOSS takes
/// the more-negative of the current-HP drop and the ceiling drop, since
/// `lose_max_hp` only lowers current HP once it exceeds the new ceiling.
fn structured_event_payload(opt: &EventOptionObs, max_hp: i64) -> Option<(f64, f64)> {
    let (Some(g), Some(h), Some(mh)) = (opt.gold_delta, opt.hp_delta, opt.max_hp_delta) else {
        return None;
    };
    let hp_units = if mh >= 0 { h } else { h.min(mh) };
    let hp_frac = (hp_units as f64 / max_hp.max(1) as f64).clamp(-1.0, 1.0);
    Some((g as f64, hp_frac))
}

// ── surface refinement (state-token categorical) ─────────────────────────

/// The `kind` tag string for one action — matches the JSON `"action"`
/// value `run_action_to_pydict`/`ActionDesc`'s serde tag produces
/// (mirrors `a.get("action", "")` reads in `encoder.py`).
fn action_kind(a: &ActionDesc) -> &'static str {
    match a {
        ActionDesc::Move { .. } => "move",
        ActionDesc::PickCardReward { .. } => "pick_card_reward",
        ActionDesc::SacrificeCardReward => "sacrifice_card_reward",
        ActionDesc::PickRelicReward => "pick_relic_reward",
        ActionDesc::SkipRelic => "skip_relic",
        ActionDesc::PickBossRelic { .. } => "pick_boss_relic",
        ActionDesc::SmithCard { .. } => "smith_card",
        ActionDesc::RemoveDeckCard { .. } => "remove_deck_card",
        ActionDesc::UpgradeDeckCard { .. } => "upgrade_deck_card",
        ActionDesc::TransformDeckCard { .. } => "transform_deck_card",
        ActionDesc::EnchantDeckCard { .. } => "enchant_deck_card",
        ActionDesc::GrantDeckCard { .. } => "grant_deck_card",
        ActionDesc::ShopBuyCard { .. } => "shop_buy_card",
        ActionDesc::ShopBuyRelic { .. } => "shop_buy_relic",
        ActionDesc::ShopBuyPotion { .. } => "shop_buy_potion",
        ActionDesc::ShopRemoveCard { .. } => "shop_remove_card",
        ActionDesc::LeaveShop => "leave_shop",
        ActionDesc::RestOption { .. } => "rest_option",
        ActionDesc::EventChoice { .. } => "event_choice",
        ActionDesc::CollectTreasure => "collect_treasure",
        ActionDesc::PotionSwap { .. } => "potion_swap",
        ActionDesc::Unknown => "unknown",
    }
}

/// `_KIND_SURFACE` verbatim — maps an action kind to its surface-defining
/// label, or `None` for kinds that never define a surface (`unknown`
/// included: it carries no entry in the Python dict either).
fn kind_surface(kind: &str) -> Option<&'static str> {
    Some(match kind {
        "move" => "path",
        "pick_card_reward" | "sacrifice_card_reward" => "card-reward",
        "pick_relic_reward" | "skip_relic" => "relic-reward",
        "pick_boss_relic" => "boss-relic",
        "smith_card" => "card-smith",
        "remove_deck_card" => "card-remove",
        "upgrade_deck_card" => "card-upgrade",
        "transform_deck_card" => "card-transform",
        "enchant_deck_card" => "card-enchant",
        "grant_deck_card" => "card-grant",
        "shop_buy_card" | "shop_buy_relic" | "shop_buy_potion" | "shop_remove_card"
        | "leave_shop" => "shop",
        "rest_option" => "rest",
        "event_choice" => "event",
        "collect_treasure" => "treasure",
        "potion_swap" => "potion-swap",
        _ => return None,
    })
}

/// Finer-grained surface for the state token, from the first
/// surface-defining action kind (skip/leave variants defer to a
/// sibling). Mirrors `encoder.py::fine_surface`: first pass skips
/// `skip_relic`/`leave_shop`; second pass accepts any; else
/// `"unknown"`.
#[must_use]
pub fn fine_surface(actions: &[ActionDesc]) -> &'static str {
    const DEFERRING: [&str; 2] = ["skip_relic", "leave_shop"];
    for a in actions {
        let k = action_kind(a);
        if !DEFERRING.contains(&k) {
            if let Some(s) = kind_surface(k) {
                return s;
            }
        }
    }
    for a in actions {
        if let Some(s) = kind_surface(action_kind(a)) {
            return s;
        }
    }
    "unknown"
}

// ── helpers ──────────────────────────────────────────────────────────────

/// Bounds-checked index into a slice — mirrors `encoder.py::_at`
/// (negative indices, which Python would otherwise wrap, never occur
/// in valid decision data; treated as out-of-range here).
fn at<T>(lst: &[T], i: i64) -> Option<&T> {
    usize::try_from(i).ok().and_then(|i| lst.get(i))
}

/// `op` names which deck operation this CARD-section token belongs to (GH
/// #545) — stamped into the appended `cat[10]` slot on every call site
/// (mirrors `encoder.py::_card_tok`'s `op` parameter).
fn card_tok(card_id: Option<&str>, upgraded: bool, price: i64, op: &str) -> (i64, Cat, ContF64) {
    let mut cat = empty_cat();
    if let Some(cid) = card_id {
        cat[0] = vocab::cards().index(cid) as i64;
        cat[1] = (i64::from(upgraded)).min(vocab::MAX_UPGRADE);
    }
    cat[10] = vocab::deck_op_index(op) as i64;
    let mut cont = [0.0f64; N_CONT];
    if price != 0 {
        cont[C_PRICE] = price as f64 * GOLD_SCALE;
    }
    (SEC_CARD, cat, cont)
}

/// Stamp a card's own enchantment onto its token (GH #947): the id into the
/// enchantment column, the amount into [`C_ENCH_AMT`]. A no-op unless
/// [`Features::card_mods`] is on, and for an unenchanted card (the pad row
/// stays). `list[i]` out of range — a short or absent list — is "not
/// enchanted", like the upgrade flags. Mirrors `encoder.py::_stamp_enchant`.
fn stamp_enchant(
    features: Features,
    cat: &mut Cat,
    cont: &mut ContF64,
    list: &[CardEnchantObs],
    i: Option<i64>,
) {
    if !features.card_mods {
        return;
    }
    if let Some((name, amount)) = i.and_then(|i| at(list, i)).and_then(Option::as_ref) {
        cat[11] = vocab::enchantments().index(name) as i64;
        cont[C_ENCH_AMT] = *amount as f64 * ENCH_AMT_SCALE;
    }
}

fn fixed_tok(kind: &str, price: i64) -> (i64, Cat, ContF64) {
    let mut cat = empty_cat();
    cat[6] = vocab::fixed_index(kind) as i64;
    let mut cont = [0.0f64; N_CONT];
    if price != 0 {
        cont[C_PRICE] = price as f64 * GOLD_SCALE;
    }
    (SEC_FIXED, cat, cont)
}

/// Featurization context: obs lookups shared by the action builders.
/// Mirrors `encoder.py::_Ctx`.
struct Ctx<'a> {
    deck: &'a [(String, bool)],
    max_hp: i64,
    shop_cards: &'a [ShopCardObs],
    shop_relics: &'a [ShopEntryObs],
    shop_potions: &'a [ShopEntryObs],
    shop_removal_cost: i64,
    /// The belt, in slot order — a `PotionSwap { discard_slot }` names a
    /// position in it.
    potions: &'a [String],
    /// The potion being OFFERED on a full belt, or `""`.
    pending_potion_swap: &'a str,
    event_name: &'a str,
    event_options: &'a [EventOptionObs],
    card_reward_ids: &'a [String],
    /// Per-option upgrade flags aligned with `card_reward_ids`; short or
    /// empty means "not upgraded". Only read under
    /// [`Features::reward_upgrade`].
    card_reward_upgraded: &'a [bool],
    relic_reward_ids: &'a [String],
    boss_relics: &'a [String],
    smith: &'a [String],
    removal: &'a [(String, bool)],
    grants: &'a [(String, bool)],
    /// Per-card enchantments aligned with `deck` / `card_reward_ids` /
    /// `smith` / `removal` / `grants` (GH #947). Short or empty means "not enchanted".
    /// Only read under [`Features::card_mods`].
    deck_ench: &'a [CardEnchantObs],
    card_reward_ench: &'a [CardEnchantObs],
    smith_ench: &'a [CardEnchantObs],
    removal_ench: &'a [CardEnchantObs],
    grants_ench: &'a [CardEnchantObs],
    transform_idxs: &'a [i64],
    enchant_idxs: &'a [i64],
    /// The UPPER_SNAKE enchantment id a pending deck-enchant pick will stamp
    /// (GH #545), or `None` when no enchant pick is pending. Read on the
    /// `EnchantDeckCard` action tokens and the state token's `cat[11]`.
    pending_enchantment: Option<&'a str>,
    /// Master-deck positions the engine offers for `UpgradeDeckCard`, in
    /// offer order. NOT derivable from `deck` here: the engine's index space
    /// is `Card::is_upgradable()` (curses / statuses / Quest cards excluded),
    /// not merely `!upgraded` — see the `UpgradeDeckCard` arm below.
    upgrade_idxs: &'a [i64],
    mapf: mapfeat::MapFeatures,
    /// Consuming cursor into `mapf.options` — one `Move` action consumes
    /// one cone per `next_move_cone()` call, in action order.
    move_seen: Cell<usize>,
    /// Opt-in encoder experiments for this featurization. Mirrors
    /// `encoder.py::_Ctx.features` / `schema.ACTIVE_FEATURES`.
    features: Features,
}

impl<'a> Ctx<'a> {
    fn new(obs: &'a PpoObs, actions: &[ActionDesc], features: Features) -> Self {
        let deck: &[(String, bool)] = obs.deck.as_slice();
        let max_hp = if obs.max_hp == 0 { 1 } else { obs.max_hp };
        let shop = obs.shop.as_ref();
        let shop_cards: &[ShopCardObs] = shop.map_or(&[][..], |s| s.cards.as_slice());
        let shop_relics: &[ShopEntryObs] = shop.map_or(&[][..], |s| s.relics.as_slice());
        let shop_potions: &[ShopEntryObs] = shop.map_or(&[][..], |s| s.potions.as_slice());
        let shop_removal_cost = shop.map_or(0, |s| s.removal_cost);

        let move_cols: Vec<i64> = actions
            .iter()
            .filter_map(|a| match a {
                ActionDesc::Move { col } => Some(*col),
                _ => None,
            })
            .collect();
        let mapf = mapfeat::compute(obs.map.as_ref(), obs.map_coord, &move_cols);

        Self {
            deck,
            max_hp,
            shop_cards,
            shop_relics,
            shop_potions,
            shop_removal_cost,
            potions: obs.potions.as_slice(),
            pending_potion_swap: obs.pending_potion_swap.as_deref().unwrap_or(""),
            event_name: obs.event_name.as_deref().unwrap_or(""),
            event_options: obs.event_options.as_deref().unwrap_or(&[]),
            card_reward_ids: obs.card_reward_option_ids.as_deref().unwrap_or(&[]),
            card_reward_upgraded: obs.card_reward_option_upgraded.as_deref().unwrap_or(&[]),
            relic_reward_ids: obs.relic_reward_ids.as_deref().unwrap_or(&[]),
            boss_relics: obs.boss_relic_options.as_deref().unwrap_or(&[]),
            smith: obs.smith_options.as_deref().unwrap_or(&[]),
            removal: obs.card_removal_options.as_deref().unwrap_or(&[]),
            grants: obs.card_grant_options.as_deref().unwrap_or(&[]),
            deck_ench: obs.deck_enchantments.as_deref().unwrap_or(&[]),
            card_reward_ench: obs
                .card_reward_option_enchantments
                .as_deref()
                .unwrap_or(&[]),
            smith_ench: obs.smith_option_enchantments.as_deref().unwrap_or(&[]),
            removal_ench: obs
                .card_removal_option_enchantments
                .as_deref()
                .unwrap_or(&[]),
            grants_ench: obs.card_grant_option_enchantments.as_deref().unwrap_or(&[]),
            transform_idxs: obs.transform_select_deck_indices.as_deref().unwrap_or(&[]),
            enchant_idxs: obs.enchant_select_deck_indices.as_deref().unwrap_or(&[]),
            pending_enchantment: obs
                .pending_enchantment
                .as_ref()
                .and_then(|inner| inner.as_deref()),
            upgrade_idxs: obs.upgrade_select_deck_indices.as_deref().unwrap_or(&[]),
            mapf,
            move_seen: Cell::new(0),
            features,
        }
    }

    fn next_move_cone(&self) -> Option<&mapfeat::OptionConeFeat> {
        let i = self.move_seen.get();
        self.move_seen.set(i + 1);
        self.mapf.options.get(i)
    }
}

/// `(section, cat, cont)` for one action. Mirrors
/// `encoder.py::_build_action_token` arm by arm (the `aux_room_label`
/// fourth return value is dropped — see the module doc comment on why
/// this port's `Featurized` has no `aux_room` field).
fn build_action_token(ctx: &Ctx<'_>, a: &ActionDesc) -> (i64, Cat, ContF64) {
    match a {
        ActionDesc::Move { .. } => {
            let of = ctx.next_move_cone();
            let mut cat = empty_cat();
            let mut cont = [0.0f64; N_CONT];
            if let Some(of) = of {
                cat[4] = of.dest_room_idx as i64;
                cont[C_DX] = of.dx;
                cont[C_DY] = of.dy;
                cont[C_REACH] = 1.0;
                cont[C_MINE] = of.min_elites;
                cont[C_MAXE] = of.max_elites;
                cont[C_DREST] = of.dist_rest;
            }
            (SEC_PATH, cat, cont)
        }
        ActionDesc::PickCardReward { idx } => match idx {
            None => fixed_tok("skip_card", 0),
            Some(i) => {
                let cid = at(ctx.card_reward_ids, *i).map(String::as_str);
                // Reward options are pre-upgraded by `roll_for_upgrade` in
                // acts 2+ (`card_rewards.rs`), but every champion to date was
                // trained with the flag dropped — see
                // [`Features::reward_upgrade`] for why handing one the truth
                // mid-life is worse than keeping it blind.
                let up = ctx.features.reward_upgrade
                    && at(ctx.card_reward_upgraded, *i).copied().unwrap_or(false);
                let (sec, mut cat, mut cont) = card_tok(cid, up, 0, "reward");
                stamp_enchant(
                    ctx.features,
                    &mut cat,
                    &mut cont,
                    ctx.card_reward_ench,
                    Some(*i),
                );
                (sec, cat, cont)
            }
        },
        ActionDesc::SacrificeCardReward => fixed_tok("sacrifice", 0),
        ActionDesc::PickRelicReward => {
            let mut cat = empty_cat();
            if let Some(rid) = ctx.relic_reward_ids.first() {
                cat[2] = vocab::relics().index(rid) as i64;
            }
            (SEC_RELIC_OPT, cat, [0.0; N_CONT])
        }
        ActionDesc::SkipRelic => fixed_tok("skip_relic", 0),
        ActionDesc::PickBossRelic { idx } => {
            let mut cat = empty_cat();
            if let Some(rid) = at(ctx.boss_relics, *idx) {
                cat[2] = vocab::relics().index(rid) as i64;
            }
            (SEC_RELIC_OPT, cat, [0.0; N_CONT])
        }
        ActionDesc::SmithCard { idx } => {
            let cid = at(ctx.smith, *idx).map(String::as_str);
            let (sec, mut cat, mut cont) = card_tok(cid, false, 0, "smith");
            stamp_enchant(
                ctx.features,
                &mut cat,
                &mut cont,
                ctx.smith_ench,
                Some(*idx),
            );
            (sec, cat, cont)
        }
        ActionDesc::RemoveDeckCard { idx } => {
            let (sec, mut cat, mut cont) = match at(ctx.removal, *idx) {
                Some((cid, up)) => card_tok(Some(cid.as_str()), *up, 0, "remove"),
                None => card_tok(None, false, 0, "remove"),
            };
            stamp_enchant(
                ctx.features,
                &mut cat,
                &mut cont,
                ctx.removal_ench,
                Some(*idx),
            );
            (sec, cat, cont)
        }
        ActionDesc::UpgradeDeckCard { idx } => {
            // `idx` is a position in the engine's `is_upgradable()` subset
            // (`run_manager.rs:954-968` offers it, `do_upgrade_deck_card_at`
            // resolves it) — Curse / Status / Quest cards are excluded, so a
            // raw `!upgraded` scan over `deck` over-offers and every option
            // after the first curse names the wrong card (GH #399 Task 5b;
            // Ascender's Bane puts a curse in EVERY A5+ deck from floor 1).
            // The obs therefore carries the engine's own offer map, exactly
            // as transform/enchant do.
            let di = at(ctx.upgrade_idxs, *idx).copied();
            let ent = di.and_then(|di| at(ctx.deck, di));
            let (sec, mut cat, mut cont) = match ent {
                Some((cid, up)) => card_tok(Some(cid.as_str()), *up, 0, "upgrade"),
                None => card_tok(None, false, 0, "upgrade"),
            };
            stamp_enchant(ctx.features, &mut cat, &mut cont, ctx.deck_ench, di);
            (sec, cat, cont)
        }
        ActionDesc::TransformDeckCard { idx } => {
            let di = at(ctx.transform_idxs, *idx).copied();
            let ent = di.and_then(|di| at(ctx.deck, di));
            let (sec, mut cat, mut cont) = match ent {
                Some((cid, up)) => card_tok(Some(cid.as_str()), *up, 0, "transform"),
                None => card_tok(None, false, 0, "transform"),
            };
            stamp_enchant(ctx.features, &mut cat, &mut cont, ctx.deck_ench, di);
            (sec, cat, cont)
        }
        ActionDesc::EnchantDeckCard { idx } => {
            let di = at(ctx.enchant_idxs, *idx).copied();
            let ent = di.and_then(|di| at(ctx.deck, di));
            let (sec, mut cat, mut cont) = match ent {
                Some((cid, up)) => card_tok(Some(cid.as_str()), *up, 0, "enchant"),
                None => card_tok(None, false, 0, "enchant"),
            };
            // The target's own enchantment first, then the pending one over
            // its id: `CanEnchant` only admits a target that is unenchanted or
            // already holds this same (stackable) enchantment, so the id
            // agrees whenever both are set and the amount says how many
            // stacks the target already has.
            stamp_enchant(ctx.features, &mut cat, &mut cont, ctx.deck_ench, di);
            if let Some(name) = ctx.pending_enchantment {
                cat[11] = vocab::enchantments().index(name) as i64;
            }
            (sec, cat, cont)
        }
        ActionDesc::GrantDeckCard { idx } => {
            let (sec, mut cat, mut cont) = match at(ctx.grants, *idx) {
                Some((cid, up)) => card_tok(Some(cid.as_str()), *up, 0, "grant"),
                None => card_tok(None, false, 0, "grant"),
            };
            stamp_enchant(
                ctx.features,
                &mut cat,
                &mut cont,
                ctx.grants_ench,
                Some(*idx),
            );
            (sec, cat, cont)
        }
        ActionDesc::ShopBuyCard { idx } => {
            let e = at(ctx.shop_cards, *idx);
            let cid = e.and_then(|e| e.id.as_deref());
            let price = e.map_or(0, |e| e.price);
            let (sec, cat, mut cont) = card_tok(cid, false, price, "shop_buy");
            cont[C_ENABLED] = if e.is_some_and(|e| e.available) {
                1.0
            } else {
                0.0
            };
            (sec, cat, cont)
        }
        ActionDesc::ShopBuyRelic { idx } => {
            let e = at(ctx.shop_relics, *idx);
            let mut cat = empty_cat();
            if let Some(id) = e.and_then(|e| e.id.as_deref()) {
                cat[2] = vocab::relics().index(id) as i64;
            }
            let mut cont = [0.0f64; N_CONT];
            cont[C_PRICE] = e.map_or(0, |e| e.price) as f64 * GOLD_SCALE;
            cont[C_ENABLED] = if e.is_some_and(|e| e.available) {
                1.0
            } else {
                0.0
            };
            (SEC_RELIC_OPT, cat, cont)
        }
        ActionDesc::ShopBuyPotion { idx } => {
            let e = at(ctx.shop_potions, *idx);
            let mut cat = empty_cat();
            if let Some(id) = e.and_then(|e| e.id.as_deref()) {
                cat[3] = vocab::potions().index(id) as i64;
            }
            let mut cont = [0.0f64; N_CONT];
            cont[C_PRICE] = e.map_or(0, |e| e.price) as f64 * GOLD_SCALE;
            cont[C_ENABLED] = if e.is_some_and(|e| e.available) {
                1.0
            } else {
                0.0
            };
            (SEC_POTION_OPT, cat, cont)
        }
        ActionDesc::ShopRemoveCard { deck_idx } => {
            let (cid, up) = match at(ctx.deck, *deck_idx) {
                Some((c, u)) => (Some(c.as_str()), *u),
                None => (None, false),
            };
            let (sec, mut cat, mut cont) = card_tok(cid, up, ctx.shop_removal_cost, "shop_remove");
            cat[6] = vocab::fixed_index("shop_remove") as i64;
            stamp_enchant(
                ctx.features,
                &mut cat,
                &mut cont,
                ctx.deck_ench,
                Some(*deck_idx),
            );
            (sec, cat, cont)
        }
        ActionDesc::LeaveShop => fixed_tok("leave_shop", 0),
        ActionDesc::RestOption { idx } => {
            // Mirrors `("rest_heal", "rest_smith")[idx] if idx < 2 else
            // "rest_other"` for the only idx values valid decision data
            // ever carries (0/1); negative-idx Python tuple-wraparound
            // is not replicated (structurally unreachable — `idx` is an
            // unsigned engine option index).
            let name = match *idx {
                0 => "rest_heal",
                1 => "rest_smith",
                _ => "rest_other",
            };
            let (sec, mut cat, cont) = fixed_tok(name, 0);
            cat[7] = (vocab::MAX_OPT_IDX as i64 - 1).min(*idx);
            (sec, cat, cont)
        }
        ActionDesc::EventChoice { idx } => {
            let mut cat = empty_cat();
            if !ctx.event_name.is_empty() {
                cat[5] = vocab::events().index(ctx.event_name) as i64;
            }
            cat[7] = (vocab::MAX_OPT_IDX as i64 - 1).min(*idx);
            let mut cont = [0.0f64; N_CONT];
            let opt = at(ctx.event_options, *idx);
            let label = opt.map_or("", |o| o.label.as_str());
            // Identity payload: enum-form labels are relic/card ids (Neow).
            if re_enum_label().is_match(label) {
                let ridx = vocab::relics().index(label);
                if ridx == vocab::relics().oov {
                    let cidx = vocab::cards().index(label);
                    if cidx != vocab::cards().oov {
                        cat[0] = cidx as i64;
                    }
                } else {
                    cat[2] = ridx as i64;
                }
            }
            let description = opt.map_or("", |o| o.description.as_str());
            // Prefer the engine's structured deltas (GH #237); fall back
            // to scraping the label+description when the obs predates them.
            let (gold_d, hp_d) = opt
                .and_then(|o| structured_event_payload(o, ctx.max_hp))
                .unwrap_or_else(|| event_payload(&format!("{label}. {description}"), ctx.max_hp));
            cont[C_GOLD_D] = gold_d * GOLD_SCALE;
            cont[C_HP_D] = hp_d;
            cont[C_ENABLED] = if opt.is_none_or(|o| o.enabled) {
                1.0
            } else {
                0.0
            };
            // Non-gold, non-HP cost. Without it an option that trades a relic
            // away or drinks a potion featurizes as doing nothing at all.
            for (slot, value) in [
                (C_RELIC_GAIN, opt.and_then(|o| o.relics_gained)),
                (C_RELIC_LOSS, opt.and_then(|o| o.relics_lost)),
                (C_POTION_GAIN, opt.and_then(|o| o.potions_gained)),
                (C_POTION_LOSS, opt.and_then(|o| o.potions_lost)),
                (C_CARD_GAIN, opt.and_then(|o| o.cards_gained)),
                (C_CARD_LOSS, opt.and_then(|o| o.cards_lost)),
            ] {
                cont[slot] = value.unwrap_or(0) as f64;
            }
            (SEC_EVENT_OPT, cat, cont)
        }
        ActionDesc::CollectTreasure => fixed_tok("open_chest", 0),
        ActionDesc::PotionSwap { discard_slot } => match discard_slot {
            // Decline: the reward is lost (and counted in
            // `potions_overflow_dropped`).
            None => fixed_tok("decline_potion", 0),
            // The held potion you would THROW AWAY. What you would gain is
            // identical across every option, so it rides on the state token
            // (`state_cat[3]`) rather than being repeated here.
            Some(slot) => {
                let mut cat = empty_cat();
                if let Some(pid) = at(ctx.potions, *slot) {
                    cat[3] = vocab::potions().index(pid) as i64;
                }
                (SEC_POTION_OPT, cat, [0.0f64; N_CONT])
            }
        },
        // Unknown action kind: generic token, never a crash (dashboarded
        // instead) — mirrors `encoder.py`'s catch-all.
        ActionDesc::Unknown => fixed_tok("none", 0),
    }
}

// ── main entry point ─────────────────────────────────────────────────────

/// Featurize one decision's `(obs, actions)` into token arrays. Mirrors
/// `encoder.py::featurize_row`'s token order: state token first, then
/// deck, relics, potions, map nodes, then one action token per
/// candidate in `actions` order.
#[must_use]
pub fn featurize(obs: &PpoObs, actions: &[ActionDesc]) -> Featurized {
    featurize_with(obs, actions, Features::active())
}

/// [`featurize`] with an explicit feature set instead of the process-wide
/// [`Features::active`]. Mirrors `encoder.py::featurize_row`'s `features`
/// argument, and lets a test or a fixture row pin the schema it was recorded
/// at rather than inheriting the runner's environment.
#[must_use]
pub fn featurize_with(obs: &PpoObs, actions: &[ActionDesc], features: Features) -> Featurized {
    let ctx = Ctx::new(obs, actions, features);

    let mut obs_cat: Vec<Cat> = Vec::new();
    let mut obs_cont: Vec<[f32; N_CONT]> = Vec::new();
    let mut obs_sec: Vec<i64> = Vec::new();

    // State scalar token — carries the fine decision surface and the
    // identity of the boss this act's map terminates in. The boss is fixed
    // at map generation, so it is known from the act's first floor: every
    // card reward, removal, upgrade and transform can be judged against the
    // fight the deck is actually being built for, instead of against the
    // average boss. Pad row when the engine could not resolve one.
    let mut state_cat = empty_cat();
    state_cat[8] = vocab::surface_index(fine_surface(actions)) as i64;
    state_cat[CAT_BOSS] = ctx.mapf.boss_idx as i64;
    // Which enchantment a pending deck-enchant pick will stamp (GH #545) —
    // pad row (never touched) when no enchant pick is pending, so this is
    // inert for every decision that isn't a card-enchant surface.
    if let Some(name) = ctx.pending_enchantment {
        state_cat[11] = vocab::enchantments().index(name) as i64;
    }
    // Pad on every other surface, so this is inert for a net that never
    // sees a `potion_swap`: the potion embedding's pad row contributes
    // nothing to the token sum.
    if !ctx.pending_potion_swap.is_empty() {
        state_cat[3] = vocab::potions().index(ctx.pending_potion_swap) as i64;
    }
    let mut state_cont = [0.0f64; N_CONT];
    state_cont[C_HP] = obs.hp as f64 / ctx.max_hp.max(1) as f64;
    state_cont[C_GOLD] = obs.gold as f64 * GOLD_SCALE;
    state_cont[C_ACT] = obs.act_index as f64;
    state_cont[C_DECKSZ] = ctx.deck.len() as f64 * DECK_SCALE;
    state_cont[C_NUPG] = obs.num_upgraded as f64 * DECK_SCALE;
    state_cont[C_NREL] = obs.num_relics as f64 * 0.1;
    state_cont[C_FLOOR] = obs.total_floor as f64 * FLOOR_SCALE;
    state_cont[C_ACTFLOOR] = obs.act_floor as f64 * ACT_FLOOR_SCALE;
    state_cont[C_ASC] = obs.ascension as f64 * ASC_SCALE;
    obs_cat.push(state_cat);
    obs_cont.push(to_f32_arr(state_cont));
    obs_sec.push(SEC_STATE);

    // Deck / relic / potion tokens.
    for (i, (cid, up)) in ctx.deck.iter().enumerate() {
        let mut cat = empty_cat();
        cat[0] = vocab::cards().index(cid) as i64;
        cat[1] = i64::from(*up).min(vocab::MAX_UPGRADE);
        let mut cont = [0.0f64; N_CONT];
        stamp_enchant(
            ctx.features,
            &mut cat,
            &mut cont,
            ctx.deck_ench,
            Some(i as i64),
        );
        obs_cat.push(cat);
        obs_cont.push(to_f32_arr(cont));
        obs_sec.push(SEC_DECK);
    }
    for rid in &obs.relics {
        let mut cat = empty_cat();
        cat[2] = vocab::relics().index(rid) as i64;
        obs_cat.push(cat);
        obs_cont.push([0.0f32; N_CONT]);
        obs_sec.push(SEC_RELIC);
    }
    for pid in &obs.potions {
        let mut cat = empty_cat();
        cat[3] = vocab::potions().index(pid) as i64;
        obs_cat.push(cat);
        obs_cont.push([0.0f32; N_CONT]);
        obs_sec.push(SEC_POTION);
    }

    // Map node tokens (R5b). The terminal boss node also carries the boss's
    // identity; every other node carries the pad row.
    for nf in &ctx.mapf.nodes {
        let mut cat = empty_cat();
        cat[4] = nf.room_idx as i64;
        cat[CAT_BOSS] = nf.boss_idx as i64;
        let mut cont = [0.0f64; N_CONT];
        cont[C_DX] = nf.dx;
        cont[C_DY] = nf.dy;
        cont[C_REACH] = nf.reachable;
        cont[C_MINE] = nf.min_elites;
        cont[C_MAXE] = nf.max_elites;
        cont[C_DREST] = nf.dist_rest;
        obs_cat.push(cat);
        obs_cont.push(to_f32_arr(cont));
        obs_sec.push(SEC_MAPNODE);
    }

    // Action tokens.
    let mut act_cat: Vec<Cat> = Vec::with_capacity(actions.len());
    let mut act_cont: Vec<[f32; N_CONT]> = Vec::with_capacity(actions.len());
    let mut act_sec: Vec<i64> = Vec::with_capacity(actions.len());
    for a in actions {
        let (sec, cat, cont) = build_action_token(&ctx, a);
        act_sec.push(sec);
        act_cat.push(cat);
        act_cont.push(to_f32_arr(cont));
    }

    Featurized {
        obs_cat,
        obs_cont,
        obs_sec,
        act_cat,
        act_cont,
        act_sec,
        n_actions: actions.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ppo::obs::EventOptionObs;

    fn opt(gold: Option<i64>, hp: Option<i64>, max_hp: Option<i64>) -> EventOptionObs {
        EventOptionObs {
            option_id: "x".into(),
            label: String::new(),
            description: String::new(),
            enabled: true,
            gold_delta: gold,
            hp_delta: hp,
            max_hp_delta: max_hp,
            ..EventOptionObs::default()
        }
    }

    #[test]
    fn structured_payload_none_without_all_three_deltas() {
        // Pre-#237 obs shape: no structured keys -> fall back to regex.
        assert!(structured_event_payload(&opt(None, None, None), 80).is_none());
        // A partial payload is also treated as absent (all-or-nothing).
        assert!(structured_event_payload(&opt(Some(5), None, None), 80).is_none());
    }

    // Exact-value float checks via `to_bits` (the codebase's convention for
    // deterministic float equality — avoids the `float_cmp` lint).
    fn bits_eq(got: f64, want: f64) {
        assert_eq!(got.to_bits(), want.to_bits(), "got {got} want {want}");
    }

    #[test]
    fn structured_payload_gold_and_hp_frac() {
        let (g, h) = structured_event_payload(&opt(Some(-50), Some(-5), Some(0)), 80).unwrap();
        bits_eq(g, -50.0);
        bits_eq(h, -5.0 / 80.0);
    }

    #[test]
    fn structured_payload_max_hp_gain_counted_once() {
        // GainMaxHp bumps current HP too, so hp_delta already carries it;
        // don't double-count the max_hp_delta.
        let (_g, h) = structured_event_payload(&opt(Some(0), Some(8), Some(8)), 80).unwrap();
        bits_eq(h, 8.0 / 80.0);
    }

    #[test]
    fn structured_payload_max_hp_loss_uses_ceiling_drop() {
        // LoseMaxHp with current HP below the new ceiling: current HP
        // unchanged (hp_delta 0) but the ceiling dropped -10 -> reflect -10.
        let (_g, h) = structured_event_payload(&opt(Some(0), Some(0), Some(-10)), 80).unwrap();
        bits_eq(h, -10.0 / 80.0);
    }

    #[test]
    fn structured_payload_clamps_to_unit_interval() {
        let (_g, h) = structured_event_payload(&opt(Some(0), Some(-999), Some(0)), 80).unwrap();
        bits_eq(h, -1.0);
    }

    // ── `reward_upgrade` feature (GH #545) ───────────────────────────────

    /// A card reward whose second option the engine pre-upgraded.
    fn upgraded_reward_obs() -> (PpoObs, Vec<ActionDesc>) {
        // Built through serde (rather than a struct literal) so this test
        // never needs touching when an unrelated obs field lands, and so
        // `PpoObs` can keep NOT deriving `Default` — the producers build it
        // field by field on purpose, which is what made the compiler catch
        // every one of them when this feature's field was added.
        let obs: PpoObs = serde_json::from_value(serde_json::json!({
            "phase": "CardReward",
            "surface": "card-pick",
            "seed": 0, "ascension": 0,
            "deck": [], "relics": [], "potions": [],
            "map": null, "map_coord": null,
            "total_floor": 20, "act_index": 1, "act_floor": 4,
            "hp": 80, "max_hp": 80, "gold": 99,
            "deck_size": 0, "num_upgraded": 0, "num_relics": 0, "num_potions": 0,
            "card_reward_option_ids": ["CARD.BASH", "CARD.CLEAVE"],
            "card_reward_option_upgraded": [false, true],
        }))
        .expect("obs");
        let actions = vec![
            ActionDesc::PickCardReward { idx: Some(0) },
            ActionDesc::PickCardReward { idx: Some(1) },
        ];
        (obs, actions)
    }

    #[test]
    fn reward_upgrade_off_drops_the_flag() {
        let (obs, actions) = upgraded_reward_obs();
        let f = featurize_with(&obs, &actions, Features::default());
        // Byte-identical to the pre-feature encoder: the upgrade column is
        // zero on BOTH options even though the engine upgraded the second.
        assert_eq!(f.act_cat[0][1], 0);
        assert_eq!(f.act_cat[1][1], 0);
        // …and identical to what the same obs featurizes to with the flags
        // absent entirely, which is what every committed fixture carries.
        let mut bare = obs.clone();
        bare.card_reward_option_upgraded = None;
        assert_eq!(
            f.act_cat,
            featurize_with(&bare, &actions, Features::default()).act_cat
        );
    }

    #[test]
    fn reward_upgrade_on_carries_the_flag_into_the_upgrade_column() {
        let (obs, actions) = upgraded_reward_obs();
        let f = featurize_with(
            &obs,
            &actions,
            Features {
                reward_upgrade: true,
                card_mods: false,
            },
        );
        assert_eq!(f.act_cat[0][1], 0);
        assert_eq!(f.act_cat[1][1], 1);
        // The card identity is untouched — only the upgrade column moves.
        let off = featurize_with(&obs, &actions, Features::default());
        assert_eq!(f.act_cat[1][0], off.act_cat[1][0]);
    }

    #[test]
    fn reward_upgrade_on_tolerates_a_short_or_missing_flag_list() {
        // Bundle / choose-screen offerings carry no upgrade state at all
        // (`RunManager::card_reward_upgraded_flags`), so a missing entry has
        // to read as "not upgraded" rather than panic on the index.
        let (mut obs, actions) = upgraded_reward_obs();
        obs.card_reward_option_upgraded = Some(vec![]);
        let f = featurize_with(
            &obs,
            &actions,
            Features {
                reward_upgrade: true,
                card_mods: false,
            },
        );
        assert_eq!(f.act_cat[0][1], 0);
        assert_eq!(f.act_cat[1][1], 0);
    }

    // ── GH #379: the full-belt potion-swap surface ───────────────────────
    //
    // Rust twin of `sts2-pycli/tests/test_ppo_potion_swap_surface.py`. The
    // two assert the SAME structural facts against the same registry, which
    // is what stands in for a `ppo_encoder_parity.jsonl` row here: at the
    // measured 0.075 full-belt potion rewards per Ironclad A0 run, requiring
    // a fixture row would cost the dump ~13 extra full runs (see
    // `dump_parity_fixtures::OPPORTUNISTIC_FINE_SURFACES`).

    /// Held belt + the potion being offered. Debug-name spellings
    /// (`format!("{k:?}")`), matching `RunManager::potions()` and
    /// `pending_potion_swap`.
    const HELD: [&str; 2] = ["BlockPotion", "FirePotion"];
    const OFFERED: &str = "EnergyPotion";

    fn swap_obs(pending: Option<&str>) -> PpoObs {
        let mut obs: PpoObs = serde_json::from_value(serde_json::json!({
            "phase": "card_reward",
            "surface": "potion-swap",
            "seed": 0,
            "ascension": 0,
            "deck": [["CARD.STRIKE_IRONCLAD", false]],
            "relics": [],
            "potions": HELD,
            "map": null,
            "map_coord": null,
            "total_floor": 1,
            "act_index": 1,
            "act_floor": 1,
            "hp": 80,
            "max_hp": 80,
            "gold": 99,
            "deck_size": 1,
            "num_upgraded": 0,
            "num_relics": 0,
            "num_potions": 2,
        }))
        .expect("obs literal");
        obs.pending_potion_swap = pending.map(str::to_owned);
        obs
    }

    /// The engine's own order (`run_manager.rs:1011`): decline first, then
    /// one discard option per held slot.
    fn swap_actions() -> Vec<ActionDesc> {
        let mut v = vec![ActionDesc::PotionSwap { discard_slot: None }];
        for i in 0..HELD.len() {
            v.push(ActionDesc::PotionSwap {
                discard_slot: Some(i as i64),
            });
        }
        v
    }

    #[test]
    fn potion_swap_tokens_name_the_discarded_potion() {
        let feat = featurize(&swap_obs(Some(OFFERED)), &swap_actions());
        assert_eq!(feat.n_actions, 3);

        // Decline is a fixed token, naming no potion.
        assert_eq!(
            feat.act_cat[0][6],
            vocab::fixed_index("decline_potion") as i64
        );
        assert_eq!(
            feat.act_cat[0][3],
            empty_cat()[3],
            "decline names no potion"
        );

        // Each discard option names the held potion at that slot — the one
        // you THROW AWAY, not the one you gain.
        for (slot, pid) in HELD.iter().enumerate() {
            assert_eq!(
                feat.act_cat[slot + 1][3],
                vocab::potions().index(pid) as i64,
                "slot {slot}"
            );
            assert_eq!(feat.act_cat[slot + 1][6], empty_cat()[6], "not a fixed tok");
        }

        // Distinguishable, or the net would be picking by position.
        assert_ne!(feat.act_cat[1][3], feat.act_cat[2][3]);
    }

    #[test]
    fn potion_swap_state_token_carries_the_offered_potion() {
        let feat = featurize(&swap_obs(Some(OFFERED)), &swap_actions());
        let state = &feat.obs_cat[0];
        assert_eq!(state[8], vocab::surface_index("potion-swap") as i64);
        assert_eq!(state[3], vocab::potions().index(OFFERED) as i64);
        // Really from `pending_potion_swap`, not leaked off the belt.
        for pid in HELD {
            assert_ne!(state[3], vocab::potions().index(pid) as i64);
        }
    }

    /// The inertness contract: `state_cat[3]` stays the potion pad row
    /// whenever no swap is pending, so adding this surface cannot perturb
    /// the observation of any already-trained one.
    #[test]
    fn potion_swap_state_token_pads_without_a_pending_swap() {
        let feat = featurize(&swap_obs(None), &swap_actions());
        assert_eq!(feat.obs_cat[0][3], empty_cat()[3]);
    }

    #[test]
    fn potion_swap_claims_the_fine_surface() {
        assert_eq!(fine_surface(&swap_actions()), "potion-swap");
        assert_eq!(action_kind(&swap_actions()[0]), "potion_swap");
    }

    /// `model.load_compat` warm-starts an older checkpoint by copying
    /// embedding rows 0..N verbatim, so a new vocab entry is only safe if it
    /// is APPENDED — an insertion silently re-maps every trained row after
    /// it. Mirrors the Python test of the same name.
    #[test]
    fn potion_swap_vocab_rows_were_appended_not_inserted() {
        assert_eq!(*vocab::SURFACES.last().unwrap(), "potion-swap");
        assert_eq!(vocab::surface_index("unknown"), 14, "rows must not move");
        assert_eq!(*vocab::FIXED_KINDS.last().unwrap(), "decline_potion");
        assert_eq!(vocab::fixed_index("sacrifice"), 10, "rows must not move");
        // The fallback resolves by NAME: "unknown" is no longer last, so a
        // positional fallback would now resolve to "potion-swap".
        assert_eq!(
            vocab::surface_index("no-such-surface"),
            vocab::surface_index("unknown")
        );
    }

    /// The load-bearing one, and the reason the new entries went into the
    /// SPARE row rather than growing the table.
    ///
    /// A torch checkpoint tolerates a grown embedding (`model.load_compat`
    /// copies rows 0..N and zero-fills). An exported **ONNX** champion does
    /// not — its tables are baked at export size, so moving a pad index
    /// feeds every deployed net an out-of-range gather. Moving them also
    /// perturbs the featurization of every ALREADY-TRAINED surface, since
    /// the pad appears in `empty_cat()` and hence in nearly every token:
    /// `ppo_encoder_parity` failed on exactly that when these were first
    /// written as `len() + 1`. Mirrors the Python test of the same name.
    #[test]
    fn potion_swap_pad_indices_did_not_move() {
        assert_eq!(
            empty_cat()[8],
            16,
            "surface pad moved — every .onnx needs re-export"
        );
        assert_eq!(
            empty_cat()[6],
            12,
            "fixed pad moved — every .onnx needs re-export"
        );
        // The spare is now spent. Whoever adds the next entry gets a red
        // test here, which is the point: they need to know it costs a
        // re-export of every champion.
        assert_eq!(vocab::N_SURFACE, vocab::SURFACES.len(), "no spare row left");
        assert_eq!(
            vocab::N_FIXED,
            vocab::FIXED_KINDS.len(),
            "no spare row left"
        );
    }

    /// GH #545: the Rust encoder serves what the Python encoder trains, so
    /// the two appended columns must agree by construction. A mismatch here
    /// is SILENT — the shapes match either way.
    #[test]
    fn appended_cat_slots_mirror_the_python_encoder() {
        assert_eq!(N_CAT, 12);
        let pad = empty_cat();
        assert_eq!(pad[10], vocab::N_DECK_OP as i64);
        assert_eq!(pad[11], vocab::enchantments().size as i64);
        assert_eq!(vocab::N_DECK_OP, vocab::DECK_OPS.len() + 1);
        assert_eq!(vocab::deck_op_index("enchant"), 6);
        assert_eq!(vocab::deck_op_index("nonsense"), 0); // "none"
    }

    // ── card_mods (GH #947) ─────────────────────────────────────────────

    const CARD_MODS: Features = Features {
        reward_upgrade: false,
        card_mods: true,
    };

    /// A three-card deck whose middle card is Sharp ×3, an enchant pick
    /// pending (Sharp) over deck positions 0 and 1, and a reward slate whose
    /// second option is relic-stamped Nimble ×2.
    fn enchanted_obs() -> (PpoObs, Vec<ActionDesc>) {
        let obs: PpoObs = serde_json::from_value(serde_json::json!({
            "phase": "CardReward",
            "surface": "card-pick",
            "seed": 0, "ascension": 0,
            "deck": [["StrikeIronclad", false], ["Bash", false], ["DefendIronclad", true]],
            "deck_enchantments": [null, ["SHARP", 3], null],
            "relics": [], "potions": [],
            "map": null, "map_coord": null,
            "total_floor": 20, "act_index": 1, "act_floor": 4,
            "hp": 80, "max_hp": 80, "gold": 99,
            "deck_size": 3, "num_upgraded": 1, "num_relics": 0, "num_potions": 0,
            "card_reward_option_ids": ["CARD.BASH", "CARD.SHRUG_IT_OFF"],
            "card_reward_option_enchantments": [null, ["NIMBLE", 2]],
            "enchant_select_deck_indices": [0, 1],
            "pending_enchantment": "SHARP",
        }))
        .expect("obs");
        let actions = vec![
            ActionDesc::PickCardReward { idx: Some(0) },
            ActionDesc::PickCardReward { idx: Some(1) },
            ActionDesc::EnchantDeckCard { idx: 0 },
            ActionDesc::EnchantDeckCard { idx: 1 },
        ];
        (obs, actions)
    }

    fn ench(name: &str) -> i64 {
        vocab::enchantments().index(name) as i64
    }

    #[test]
    fn card_mods_off_is_byte_identical_to_an_obs_without_the_fields() {
        let (obs, actions) = enchanted_obs();
        let mut bare = obs.clone();
        bare.deck_enchantments = None;
        bare.card_reward_option_enchantments = None;
        let f = featurize_with(&obs, &actions, Features::default());
        assert_eq!(f, featurize_with(&bare, &actions, Features::default()));
        // Deck tokens (obs rows 1..=3) keep the pad enchantment and zero cont.
        let pad = empty_cat()[11];
        for t in 1..=3 {
            assert_eq!(f.obs_cat[t][11], pad);
            assert_eq!(f.obs_cont[t][C_ENCH_AMT].to_bits(), (0.0f32).to_bits());
        }
    }

    #[test]
    fn card_mods_stamps_deck_tokens_with_their_own_enchantment() {
        let (obs, actions) = enchanted_obs();
        let f = featurize_with(&obs, &actions, CARD_MODS);
        let pad = empty_cat()[11];
        assert_eq!(f.obs_sec[2], SEC_DECK);
        assert_eq!(f.obs_cat[1][11], pad);
        assert_eq!(f.obs_cat[2][11], ench("SHARP"));
        assert_eq!(
            f.obs_cont[2][C_ENCH_AMT].to_bits(),
            ((3.0 * ENCH_AMT_SCALE) as f32).to_bits()
        );
        assert_eq!(f.obs_cat[3][11], pad);
        // The card identity and upgrade columns are untouched.
        let off = featurize_with(&obs, &actions, Features::default());
        for t in 1..=3 {
            assert_eq!(f.obs_cat[t][..11], off.obs_cat[t][..11]);
        }
    }

    #[test]
    fn card_mods_stamps_a_relic_stamped_reward_option() {
        let (obs, actions) = enchanted_obs();
        let f = featurize_with(&obs, &actions, CARD_MODS);
        assert_eq!(f.act_cat[0][11], empty_cat()[11]);
        assert_eq!(f.act_cat[1][11], ench("NIMBLE"));
        assert_eq!(
            f.act_cont[1][C_ENCH_AMT].to_bits(),
            ((2.0 * ENCH_AMT_SCALE) as f32).to_bits()
        );
    }

    #[test]
    fn card_mods_enchant_target_keeps_the_pending_id_and_shows_existing_stacks() {
        let (obs, actions) = enchanted_obs();
        let f = featurize_with(&obs, &actions, CARD_MODS);
        // Both targets carry the PENDING id (GH #545, unchanged) …
        assert_eq!(f.act_cat[2][11], ench("SHARP"));
        assert_eq!(f.act_cat[3][11], ench("SHARP"));
        // … and only the already-Sharp target shows stacks in the amount.
        assert_eq!(f.act_cont[2][C_ENCH_AMT].to_bits(), (0.0f32).to_bits());
        assert_eq!(
            f.act_cont[3][C_ENCH_AMT].to_bits(),
            ((3.0 * ENCH_AMT_SCALE) as f32).to_bits()
        );
    }

    #[test]
    fn card_mods_tolerates_a_short_or_missing_enchantment_list() {
        let (mut obs, actions) = enchanted_obs();
        obs.deck_enchantments = Some(vec![None]);
        obs.card_reward_option_enchantments = None;
        let f = featurize_with(&obs, &actions, CARD_MODS);
        let pad = empty_cat()[11];
        assert_eq!(f.obs_cat[2][11], pad);
        assert_eq!(f.act_cat[1][11], pad);
    }

    #[test]
    fn card_mods_stamps_a_smith_option_through_its_own_list() {
        let (mut obs, _) = enchanted_obs();
        obs.smith_options = Some(vec!["StrikeIronclad".into(), "Bash".into()]);
        obs.smith_option_enchantments = Some(vec![None, Some(("SHARP".into(), 3))]);
        let actions = vec![
            ActionDesc::SmithCard { idx: 0 },
            ActionDesc::SmithCard { idx: 1 },
        ];
        let f = featurize_with(&obs, &actions, CARD_MODS);
        assert_eq!(f.act_cat[0][11], empty_cat()[11]);
        assert_eq!(f.act_cat[1][11], ench("SHARP"));
        assert_eq!(
            f.act_cont[1][C_ENCH_AMT].to_bits(),
            ((3.0 * ENCH_AMT_SCALE) as f32).to_bits()
        );
    }

    #[test]
    fn enchant_amount_slot_is_appended_after_the_event_payload() {
        // Appended, never inserted: a 25-cont champion is fed the first 25
        // columns and must see exactly what it was trained on.
        assert_eq!(C_ENCH_AMT, C_CARD_LOSS + 1);
        assert_eq!(N_CONT, C_ENCH_AMT + 1);
    }
}
