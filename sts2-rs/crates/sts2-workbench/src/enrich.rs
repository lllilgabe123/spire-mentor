//! The display-enrichment pass: art URLs, resolved card / relic / event text,
//! upgrade twins, intent icons, per-instance size jitter.
//!
//! Port of the `_enrich` half of `sts2-pycli/python/workbench/server.py`
//! (`_asset_url`, `_frame_color`, `_enrich_map_nodes`,
//! `_apply_instance_jitter`, `_manifest_cost`, `_apply_upgrade_cost`,
//! `_resolve_owned_text`, `_event_option_vars`, `_mad_science_flags`,
//! `_enrich_relic` / `_enrich_potion`, `_combat_log_names`, `_enrich`,
//! `_enrich_intent_modifiers`). Runs over the `Step` dict the SPA
//! consumes; every key it adds or rewrites is part of the wire contract the
//! golden-parity tests diff, so this follows the Python branch for branch.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{json, Map, Value};

use crate::assets::{normalize_kind, AssetResolver};
use crate::combat_view::{py_str, title_case, truthy};
use crate::combatlog::{format_combat_log, LogNames};
use crate::data::CreatureScale;
use crate::display_maps::{monster_title, power_title};
use crate::localization::{
    self, affliction_card_line, card_var_value, keyword_card_lines, looks_like_unresolved_key,
    named_description_vars, plain_description, plain_owned_description, resolve, resolve_name,
    resolve_state_names, HpRange, Loc,
};
use crate::render::{describe_action, required_stat_vars, resolve_card_text, CardTextFlags};

/// What the enrichment pass reads: the art resolver, the loc map, the
/// monster HP bands (for the size jitter) and where art is served from.
#[derive(Debug, Clone, Copy)]
pub struct EnrichCtx<'a> {
    pub assets: &'a AssetResolver,
    pub loc: &'a Loc,
    pub hp_ranges: Option<&'a BTreeMap<String, HpRange>>,
    /// `STS2_ASSETS_BASE_URL`, no trailing slash.
    pub assets_base: &'a str,
}

/// `_asset_url`: a served asset URL for a relative art path, or `null`.
#[must_use]
pub fn asset_url(base: &str, rel: Option<String>) -> Value {
    match rel {
        Some(r) if !r.is_empty() => json!(format!("{base}/{r}")),
        _ => Value::Null,
    }
}

/// `_frame_color`: the card's frame tint key.
#[must_use]
pub fn frame_color(
    card_character: Option<&str>,
    card_type: Option<&str>,
    rarity: Option<&str>,
) -> &'static str {
    let kind = format!("{}|{}", card_type.unwrap_or(""), rarity.unwrap_or("")).to_lowercase();
    if kind.contains("curse") {
        return "curse";
    }
    if kind.contains("quest") {
        return "quest";
    }
    match card_character.unwrap_or("").trim().to_lowercase().as_str() {
        "ironclad" => "red",
        "silent" => "green",
        "defect" => "blue",
        "necrobinder" => "pink",
        "regent" => "orange",
        _ => "colorless",
    }
}

/// `_LOC_KEY_RE.match`: `^[A-Z0-9][A-Z0-9_]*\.(title|name|description|flavor)$`.
fn is_loc_key(s: &str) -> bool {
    let Some((stem, field)) = s.split_once('.') else {
        return false;
    };
    let mut chars = stem.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !(first.is_ascii_uppercase() || first.is_ascii_digit()) {
        return false;
    }
    if !chars.all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_') {
        return false;
    }
    matches!(field, "title" | "name" | "description" | "flavor")
}

/// `card.get(key)` as a string, `""` when absent / not a string.
fn sval<'a>(v: &'a Value, key: &str) -> &'a str {
    v.get(key).and_then(Value::as_str).unwrap_or("")
}

/// `(card.get("id") or "").split(".")[-1]`.
fn id_tail(v: &Value) -> String {
    let id = sval(v, "id");
    id.rsplit('.').next().unwrap_or(id).to_string()
}

/// Python `s.lstrip("-").isdigit()` → int, else the string as-is.
fn int_or_str(s: &str) -> Value {
    let digits = s.trim_start_matches('-');
    if !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit()) {
        match s.parse::<i64>() {
            Ok(n) => json!(n),
            Err(_) => json!(s),
        }
    } else {
        json!(s)
    }
}

/// `_manifest_cost`: the display-ready energy cost from the manifest.
fn manifest_cost(cid: &str, loc: &Loc, upgraded: bool) -> Option<Value> {
    let mut mcost = if upgraded {
        loc.get(&format!("{cid}.upgraded_cost"))
    } else {
        None
    };
    if mcost.is_none() {
        mcost = loc.get(&format!("{cid}.cost"));
    }
    mcost.map(int_or_str)
}

/// `_apply_upgrade_cost`: put the post-upgrade energy cost on a twin.
fn apply_upgrade_cost(twin: &mut Map<String, Value>, card: &Value, loc: &Loc) {
    let cid = id_tail(card);
    let base = manifest_cost(&cid, loc, false).and_then(|v| v.as_i64());
    let upgraded = manifest_cost(&cid, loc, true).and_then(|v| v.as_i64());
    let delta = match (base, upgraded) {
        (Some(b), Some(u)) => u - b,
        _ => 0,
    };
    for key in ["cost", "base_cost"] {
        let live = twin.get(key).cloned();
        match live {
            None | Some(Value::Null) => {
                twin.remove(key);
            }
            Some(Value::Number(n)) => {
                if let Some(v) = n.as_i64() {
                    if v >= 0 {
                        twin.insert(key.into(), json!((v + delta).max(0)));
                    }
                }
            }
            _ => {}
        }
    }
}

fn flags(
    in_combat: bool,
    upgraded: bool,
    extra: Option<&Value>,
    keep_markup: bool,
) -> CardTextFlags<'_> {
    CardTextFlags {
        in_combat,
        upgraded,
        extra_flags: extra,
        keep_markup,
    }
}

/// `_resolve_owned_text`: a relic / potion description rendered with its own
/// `vars`, or the wiki's pre-resolved plain text when the wire has none.
fn resolve_owned_text(item: &Value, loc: &Loc) -> String {
    let mut item_vars = Map::new();
    if let Some(Value::Object(vars)) = item.get("vars") {
        for (k, v) in vars {
            item_vars.insert(k.to_lowercase(), v.clone());
        }
    }
    if item_vars.is_empty() {
        let mut stem = id_tail(item);
        if stem.is_empty() {
            stem = sval(item, "description")
                .split('.')
                .next()
                .unwrap_or("")
                .to_string();
        }
        if let Some(plain) = plain_owned_description(Some(&stem), loc) {
            if !plain.is_empty() {
                return resolve_card_text(Some(plain), None, flags(false, false, None, false));
            }
        }
    }
    let template = resolve(item.get("description").and_then(Value::as_str), loc);
    if is_loc_key(&template) {
        return String::new();
    }
    let vars = Value::Object(item_vars);
    resolve_card_text(
        Some(&template),
        Some(&vars),
        flags(false, false, None, false),
    )
}

/// `_EVENT_FLAGS`: the sim is single-player, always.
fn event_flags() -> Value {
    json!({"ismultiplayer": false})
}

/// `_event_option_vars`: lower-cased option vars with names resolved.
fn event_option_vars(opt: &Value, loc: &Loc, key: &str) -> Map<String, Value> {
    let mut out = Map::new();
    if let Some(Value::Object(vars)) = opt.get(key) {
        for (name, val) in vars {
            let v = match val {
                Value::String(s) => json!(resolve_name(Some(s), loc)),
                other => other.clone(),
            };
            out.insert(name.to_lowercase(), v);
        }
    }
    out
}

const MAD_SCIENCE_RIDERS: [&str; 9] = [
    "violence",
    "sapping",
    "choking",
    "energized",
    "wisdom",
    "chaos",
    "expertise",
    "curious",
    "improvement",
];

/// `_mad_science_flags`: the morph `extra_flags` for a Mad Science card.
fn mad_science_flags(card: &Value) -> Option<Map<String, Value>> {
    if id_tail(card) != "MAD_SCIENCE" {
        return None;
    }
    let rider = card
        .get("tinker_time_rider")
        .filter(|v| truthy(v))
        .map_or_else(|| "None".to_string(), py_str)
        .to_lowercase();
    let mut out = Map::new();
    for r in MAD_SCIENCE_RIDERS {
        out.insert(r.to_string(), json!(rider == r));
    }
    out.insert("hasrider".into(), json!(rider != "none"));
    let ctype = card
        .get("type")
        .filter(|v| truthy(v))
        .cloned()
        .unwrap_or(json!("Attack"));
    out.insert("cardtype".into(), ctype);
    Some(out)
}

/// `card_select` contexts whose pick UPGRADES the chosen card.
const UPGRADE_CONTEXTS: [&str; 2] = ["upgrade", "upgrade_in_hand"];

/// `_enrich_relic`: name / image / description of a relic dict.
fn enrich_relic(relic: &mut Value, ctx: &EnrichCtx<'_>) {
    let name = resolve_name(relic.get("name").and_then(Value::as_str), ctx.loc);
    let image = asset_url(ctx.assets_base, ctx.assets.relic(&name));
    let m = relic.as_object_mut().expect("relic dict");
    m.insert("name".into(), json!(name));
    m.insert("image".into(), image);
    let desc = resolve_owned_text(relic, ctx.loc);
    relic
        .as_object_mut()
        .expect("relic dict")
        .insert("description".into(), json!(desc));
}

/// `_enrich_potion`.
fn enrich_potion(potion: &mut Value, ctx: &EnrichCtx<'_>) {
    let name = resolve_name(potion.get("name").and_then(Value::as_str), ctx.loc);
    let image = asset_url(ctx.assets_base, ctx.assets.potion(&name));
    let m = potion.as_object_mut().expect("potion dict");
    m.insert("name".into(), json!(name));
    m.insert("image".into(), image);
    let desc = resolve_owned_text(potion, ctx.loc);
    potion
        .as_object_mut()
        .expect("potion dict")
        .insert("description".into(), json!(desc));
}

/// `_combat_log_names`: display-name resolvers for the combat log, off a
/// loc map. Shared with the session so both render identically.
#[must_use]
pub fn log_names(loc: &'static Loc) -> LogNames {
    let titled = move |raw: &str| resolve_name(Some(&format!("{raw}.title")), loc);
    LogNames {
        card: Box::new(titled),
        monster: Box::new(|raw: &str| monster_title(raw)),
        power: Box::new(|raw: &str| power_title(raw)),
        potion: Box::new(titled),
        orb: Box::new(move |raw: &str| titled(&format!("{raw}_ORB"))),
        character: Box::new(titled),
        ..LogNames::default()
    }
}

/// `_enrich_map_nodes`: attach an `icon` URL to every map node and the boss
/// meta-nodes. `null` / non-object passes through.
#[must_use]
pub fn enrich_map_nodes(mut full_map: Value, assets: &AssetResolver, base: &str) -> Value {
    if !truthy(&full_map) {
        return full_map;
    }
    let Some(fm) = full_map.as_object_mut() else {
        return full_map;
    };
    let boss_id = fm
        .get("boss")
        .and_then(|b| b.get("id"))
        .and_then(Value::as_str)
        .map(str::to_string);
    let second = fm.get("second_boss").cloned().unwrap_or(Value::Null);
    let second_id = second.get("id").and_then(Value::as_str).map(str::to_string);
    let second_coord = if second.get("row").is_some_and(|r| !r.is_null()) {
        Some((
            second.get("col").cloned().unwrap_or(Value::Null),
            second.get("row").cloned().unwrap_or(Value::Null),
        ))
    } else {
        None
    };
    if let Some(Value::Array(rows)) = fm.get_mut("rows") {
        for row in rows.iter_mut() {
            let Some(nodes) = row.as_array_mut() else {
                continue;
            };
            for node in nodes.iter_mut() {
                let coord = (
                    node.get("col").cloned().unwrap_or(Value::Null),
                    node.get("row").cloned().unwrap_or(Value::Null),
                );
                let node_boss_id = match &second_coord {
                    Some(sc) if *sc == coord => second_id.as_deref(),
                    _ => boss_id.as_deref(),
                };
                let icon = asset_url(
                    base,
                    assets.room_icon(node.get("type").and_then(Value::as_str), node_boss_id),
                );
                if let Some(m) = node.as_object_mut() {
                    m.insert("icon".into(), icon);
                }
            }
        }
    }
    if let Some(boss) = fm.get_mut("boss").filter(|b| truthy(b)) {
        let icon = asset_url(
            base,
            assets.room_icon(boss.get("type").and_then(Value::as_str), boss_id.as_deref()),
        );
        if let Some(m) = boss.as_object_mut() {
            m.insert("icon".into(), icon);
        }
    }
    if let Some(sec) = fm.get_mut("second_boss").filter(|b| truthy(b)) {
        let icon = asset_url(
            base,
            assets.room_icon(
                sec.get("type").and_then(Value::as_str),
                second_id.as_deref(),
            ),
        );
        if let Some(m) = sec.as_object_mut() {
            m.insert("icon".into(), icon);
        }
    }
    full_map
}

/// `Rng.Chaotic.NextFloat(0.05)` in the game.
const HUE_JITTER_MAX: f64 = 0.05;

/// Python `round(x, 4)` (close enough: the golden compare uses a tolerance).
fn round4(x: f64) -> f64 {
    (x * 10_000.0).round() / 10_000.0
}

/// `_apply_instance_jitter`: `NCombatRoom.RandomizeEnemyScalesAndHues`,
/// stamped onto the wire enemies as `size_jitter` / `hue_jitter`.
pub fn apply_instance_jitter(enemies: &mut [Value], hp_ranges: &BTreeMap<String, HpRange>) {
    // Group positions by normalized kind, preserving first-seen order.
    let mut groups: Vec<(String, Vec<usize>)> = Vec::new();
    for (i, enemy) in enemies.iter().enumerate() {
        let Some(kind) = normalize_kind(enemy.get("kind").and_then(Value::as_str)) else {
            continue;
        };
        if kind.is_empty() {
            continue;
        }
        match groups.iter_mut().find(|(k, _)| *k == kind) {
            Some((_, idxs)) => idxs.push(i),
            None => groups.push((kind, vec![i])),
        }
    }
    for (kind, idxs) in groups {
        if idxs.len() < 2 {
            continue;
        }
        let Some(band) = hp_ranges.get(&kind) else {
            continue;
        };
        for (index, &pos) in idxs.iter().enumerate() {
            let enemy = &mut enemies[pos];
            let Some(hp) = enemy.get("max_hp").and_then(Value::as_i64) else {
                continue;
            };
            let (mut lo, mut hi) = (band.min, band.max);
            if !(lo <= hp && hp <= hi) && band.min_asc <= hp && hp <= band.max_asc {
                lo = band.min_asc;
                hi = band.max_asc;
            }
            let span = hi - lo;
            let num = if span <= 0 {
                0.0
            } else {
                (((hp - lo) as f64 / span as f64 - 0.5) * 2.0).clamp(0.0, 1.0)
            };
            let bw = enemy
                .get("rig_bounds_w")
                .and_then(Value::as_f64)
                .unwrap_or(0.0);
            let bh = enemy
                .get("rig_bounds_h")
                .and_then(Value::as_f64)
                .unwrap_or(0.0);
            let bx = bw.max(bh);
            let amount = if bx <= 0.0 {
                0.0
            } else {
                ((250.0 - bx) / 150.0).clamp(0.0, 1.0)
            };
            let envelope = 0.10 + (0.15 - 0.10) * amount;
            let spread = i64::from(crc32fast::hash(format!("{kind}|{index}").as_bytes()) & 0xFFFF);
            let m = enemy.as_object_mut().expect("enemy dict");
            m.insert("size_jitter".into(), json!(round4(1.0 + num * envelope)));
            m.insert(
                "hue_jitter".into(),
                json!(round4(HUE_JITTER_MAX * (spread as f64 / 65535.0))),
            );
        }
    }
}

/// `enrich_powers`: the two icons every power carries.
fn enrich_powers(powers: Option<&mut Value>, ctx: &EnrichCtx<'_>) {
    let Some(Value::Array(list)) = powers else {
        return;
    };
    for pw in list.iter_mut() {
        let name = sval(pw, "name").to_string();
        let image = asset_url(ctx.assets_base, ctx.assets.power(&name));
        let image_big = asset_url(ctx.assets_base, ctx.assets.power_big(&name));
        if let Some(m) = pw.as_object_mut() {
            m.insert("image".into(), image);
            m.insert("image_big".into(), image_big);
        }
    }
}

/// `_enemy_has_power`: whether an enriched enemy carries a power whose
/// id / name contains `token` (underscore-normalized, upper-cased).
fn enemy_has_power(enemy: &Value, token: &str) -> bool {
    let want = token.to_uppercase().replace(' ', "_");
    let Some(powers) = enemy.get("powers").and_then(Value::as_array) else {
        return false;
    };
    powers.iter().any(|pw| {
        ["id", "name"].iter().any(|field| {
            py_str(pw.get(field).unwrap_or(&json!("")))
                .to_uppercase()
                .replace(' ', "_")
                .contains(&want)
        })
    })
}

/// Stamp `rig_scale` / `rig_bounds_h` / `rig_bounds_w` from a creature
/// scale dict (`pscale.get(...)`: missing → `null`).
fn stamp_scale(target: &mut Map<String, Value>, scale: &CreatureScale) {
    let get = |k: &str| {
        scale
            .get(k)
            .copied()
            .flatten()
            .map_or(Value::Null, |f| json!(f))
    };
    target.insert("rig_scale".into(), get("scale"));
    target.insert("rig_bounds_h".into(), get("bounds_h"));
    target.insert("rig_bounds_w".into(), get("bounds_w"));
}

/// `_enrich_intent_modifiers`: a display title and an icon per modifier row.
fn enrich_intent_modifiers(intent: &mut Value, ctx: &EnrichCtx<'_>) {
    let Some(Value::Array(mods)) = intent.get_mut("modifiers") else {
        return;
    };
    for m in mods.iter_mut() {
        let name = py_str(m.get("name").unwrap_or(&json!("")));
        let kind = m.get("kind").and_then(Value::as_str).unwrap_or("");
        let (title, image) = match kind {
            "power" => {
                let t = power_title(&name);
                let img = asset_url(ctx.assets_base, ctx.assets.power(&t));
                (t, Some(img))
            }
            "relic" => {
                let t = title_case(&name.replace('_', " "));
                let img = asset_url(ctx.assets_base, ctx.assets.relic(&t));
                (t, Some(img))
            }
            _ => ("Other".to_string(), None),
        };
        if let Some(mm) = m.as_object_mut() {
            mm.insert("title".into(), json!(title));
            if let Some(img) = image {
                mm.insert("image".into(), img);
            }
        }
    }
}

/// `enrich_card`: art, real type / rarity / frame colour, manifest cost,
/// and the fully rendered description (plain + rich), for one card dict.
fn enrich_card(
    card: &mut Value,
    in_combat: bool,
    osty_alive: bool,
    character: &str,
    ctx: &EnrichCtx<'_>,
) {
    let loc = ctx.loc;
    let cid = id_tail(card);
    let morph = mad_science_flags(card);
    let image = if let Some(m) = &morph {
        let variant = format!(
            "mad_science_{}",
            py_str(m.get("cardtype").unwrap_or(&Value::Null)).to_lowercase()
        );
        asset_url(
            ctx.assets_base,
            ctx.assets.card(character, "", &variant, None),
        )
    } else {
        asset_url(
            ctx.assets_base,
            ctx.assets.card(
                character,
                sval(card, "id"),
                sval(card, "name"),
                loc.get(&format!("{cid}.character")),
            ),
        )
    };
    let ctype = loc.get(&format!("{cid}.card_type")).map(str::to_string);
    let rarity = loc.get(&format!("{cid}.rarity")).map(str::to_string);
    {
        let m = card.as_object_mut().expect("card dict");
        m.insert("image".into(), image);
        if let Some(t) = &ctype {
            if !t.is_empty() {
                m.insert("type".into(), json!(t));
            }
        }
        if let Some(r) = &rarity {
            if !r.is_empty() {
                m.insert("rarity".into(), json!(r));
            }
        }
    }
    let color = frame_color(
        loc.get(&format!("{cid}.character")),
        card.get("type").and_then(Value::as_str),
        card.get("rarity").and_then(Value::as_str),
    );
    let star = loc.get(&format!("{cid}.star_cost")).map(int_or_str);
    let upgraded = truthy(card.get("upgraded").unwrap_or(&Value::Null));
    let cost_missing = card.get("cost").is_none_or(Value::is_null);
    let mcost = if cost_missing {
        manifest_cost(&cid, loc, upgraded)
    } else {
        None
    };
    let name = resolve(card.get("name").and_then(Value::as_str), loc);
    let template = resolve(card.get("description").and_then(Value::as_str), loc);
    {
        let m = card.as_object_mut().expect("card dict");
        m.insert("color".into(), json!(color));
        if let Some(s) = star {
            m.insert("star_cost".into(), s);
        }
        if let Some(c) = mcost {
            m.insert("cost".into(), c);
        }
        m.insert("name".into(), json!(name));
    }
    // Card-level facts the formatter reads that are not DynamicVars.
    let mut flag_map = Map::new();
    flag_map.insert("istargeting".into(), json!(false));
    flag_map.insert("isostyalive".into(), json!(osty_alive));
    if let Some(m) = &morph {
        for (k, v) in m {
            flag_map.insert(k.clone(), v.clone());
        }
    }
    let flag_val = Value::Object(flag_map);
    let mut stats: Option<Value> = card.get("stats").filter(|s| !s.is_null()).cloned();
    let needed = required_stat_vars(
        Some(&template),
        flags(in_combat, upgraded, Some(&flag_val), false),
    );
    let have_of = |stats: &Option<Value>| -> BTreeSet<String> {
        stats
            .as_ref()
            .and_then(Value::as_object)
            .map(|m| m.keys().map(|k| k.to_lowercase()).collect())
            .unwrap_or_default()
    };
    let mut have = have_of(&stats);
    let mut missing: BTreeSet<String> = needed.difference(&have).cloned().collect();
    if !missing.is_empty() && in_combat {
        let mut filled = stats
            .as_ref()
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        let calculated = card
            .get("calculated_value")
            .filter(|v| !v.is_null())
            .cloned();
        for var in &missing {
            if var.starts_with("calculated") {
                if let Some(c) = &calculated {
                    filled.insert(var.clone(), c.clone());
                }
                continue;
            }
            if let Some(base) =
                card_var_value(card.get("id").and_then(Value::as_str), var, upgraded, loc)
            {
                filled.insert(var.clone(), int_or_str(base));
            }
        }
        stats = Some(Value::Object(filled));
        have = have_of(&stats);
        missing = needed.difference(&have).cloned().collect();
    }
    let plain = if missing.is_empty() {
        None
    } else {
        plain_description(card.get("id").and_then(Value::as_str), upgraded, loc)
            .filter(|p| !p.is_empty())
            .map(str::to_string)
    };
    let (description, description_rich) = if let Some(plain) = plain {
        (
            resolve_card_text(Some(&plain), None, flags(false, false, None, false)),
            resolve_card_text(Some(&plain), None, flags(false, false, None, true)),
        )
    } else {
        (
            resolve_card_text(
                Some(&template),
                stats.as_ref(),
                flags(in_combat, upgraded, Some(&flag_val), false),
            ),
            resolve_card_text(
                Some(&template),
                stats.as_ref(),
                flags(in_combat, upgraded, Some(&flag_val), true),
            ),
        )
    };
    {
        let m = card.as_object_mut().expect("card dict");
        m.insert("description".into(), json!(description));
        m.insert("description_rich".into(), json!(description_rich));
    }
    // Keywords print ON the card face; an affliction between body and after.
    let (before, after) = keyword_card_lines(card.get("id").and_then(Value::as_str), upgraded, loc);
    let affliction_line = affliction_card_line(card.get("affliction").and_then(Value::as_str), loc);
    if !before.is_empty() || !after.is_empty() || affliction_line.is_some() {
        let amount = card
            .get("affliction_amount")
            .cloned()
            .unwrap_or(Value::Null);
        for (field, keep) in [("description", false), ("description_rich", true)] {
            let mut lines: Vec<String> = before
                .iter()
                .map(|l| resolve_card_text(Some(l), None, flags(false, false, None, keep)))
                .collect();
            let body = sval(card, field).to_string();
            if !body.is_empty() {
                lines.push(body);
            }
            if let Some(al) = &affliction_line {
                let vars = json!({"amount": amount});
                lines.push(resolve_card_text(
                    Some(al),
                    Some(&vars),
                    flags(false, false, None, keep),
                ));
            }
            lines.extend(
                after
                    .iter()
                    .map(|l| resolve_card_text(Some(l), None, flags(false, false, None, keep))),
            );
            card.as_object_mut()
                .expect("card dict")
                .insert(field.into(), json!(lines.join("\n")));
        }
    }
}

fn enrich_cards(
    list: Option<&mut Value>,
    in_combat: bool,
    osty_alive: bool,
    character: &str,
    ctx: &EnrichCtx<'_>,
) {
    if let Some(Value::Array(cards)) = list {
        for card in cards.iter_mut() {
            enrich_card(card, in_combat, osty_alive, character, ctx);
        }
    }
}

/// `_enrich`: the whole pass over a `Step` dict.
#[must_use]
pub fn enrich(mut step: Value, ctx: &EnrichCtx<'_>) -> Value {
    let loc = ctx.loc;
    let assets = ctx.assets;
    let base = ctx.assets_base;
    let url = |rel: Option<String>| asset_url(base, rel);

    let Some(sm) = step.as_object_mut() else {
        return step;
    };
    sm.insert("assets_base".into(), json!(base));
    if !sm.get("state").is_some_and(Value::is_object) {
        sm.insert("state".into(), json!({}));
    }
    let chosen = sm.get("chosen_action").cloned().unwrap_or(Value::Null);
    let decision = sm
        .get("decision")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let state = sm
        .get_mut("state")
        .expect("state")
        .as_object_mut()
        .expect("state dict");
    // A human-readable label for the action taken from this step.
    let label = describe_action(
        Some(&resolve_state_names(
            Some(&Value::Object(state.clone())),
            loc,
        )),
        if chosen.is_null() {
            None
        } else {
            Some(&chosen)
        },
    );
    // Live-GAME runs carry the game's own combat-log rows, raw.
    if let Some(raw_log) = state.remove("combat_log_raw").filter(truthy) {
        let rows = raw_log.as_array().cloned().unwrap_or_default();
        let player_id = state
            .get("character_id")
            .filter(|v| truthy(v))
            .map_or_else(|| "PLAYER".to_string(), py_str);
        let names = log_names(loc_static(loc));
        state.insert(
            "combat_log".into(),
            Value::Array(format_combat_log(
                &rows,
                Some(&names),
                false,
                &[],
                &player_id,
            )),
        );
    }
    let character = state
        .get("player")
        .and_then(|p| p.get("name"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();

    // Relics and potions persist across the whole run.
    if let Some(player) = state.get_mut("player").and_then(Value::as_object_mut) {
        if let Some(Value::Array(relics)) = player.get_mut("relics") {
            for relic in relics.iter_mut() {
                enrich_relic(relic, ctx);
            }
        }
        if let Some(Value::Array(potions)) = player.get_mut("potions") {
            for potion in potions.iter_mut() {
                if truthy(potion) {
                    enrich_potion(potion, ctx);
                }
            }
        }
    }
    // The chest panel rides on whatever decision comes NEXT.
    if let Some(reveal) = state.get_mut("room_reveal").and_then(Value::as_object_mut) {
        if let Some(relic) = reveal.get_mut("relic").filter(|r| r.is_object()) {
            enrich_relic(relic, ctx);
        }
    }
    // The full master deck persists across the run.
    if let Some(player) = state.get_mut("player") {
        enrich_cards(player.get_mut("deck"), false, false, &character, ctx);
    }

    match decision.as_str() {
        "combat_play" => {
            if let Some(player) = state.get_mut("player").and_then(Value::as_object_mut) {
                player.insert(
                    "image".into(),
                    url(assets
                        .character_sprite(&character)
                        .or_else(|| assets.portrait(&character))),
                );
                player.insert("rig".into(), url(assets.character_rig(&character)));
                if let Some(pscale) = assets.character_rig_scale(&character) {
                    stamp_scale(player, pscale);
                }
            }
            enrich_powers(state.get_mut("player_powers"), ctx);
            if let Some(osty) = state.get_mut("osty").filter(|o| o.is_object()) {
                if truthy(osty.get("alive").unwrap_or(&Value::Null)) {
                    let om = osty.as_object_mut().expect("osty dict");
                    om.entry("name").or_insert(json!("Osty"));
                    om.insert("image".into(), url(assets.monster("Osty")));
                    om.insert("rig".into(), url(assets.monster_rig("Osty", None)));
                    if let Some(oscale) = assets.monster_rig_scale("Osty", None) {
                        stamp_scale(om, &oscale);
                    }
                }
            }
            if let Some(Value::Array(enemies)) = state.get_mut("enemies") {
                for enemy in enemies.iter_mut() {
                    let ekind = normalize_kind(enemy.get("kind").and_then(Value::as_str));
                    let ekind = ekind.as_deref();
                    let ename = sval(enemy, "name").to_string();
                    let image = url(assets.monster(&ename));
                    let rig = url(assets.monster_rig(&ename, ekind));
                    let escale = assets.monster_rig_scale(&ename, ekind);
                    let group = assets.monster_rig_group(&ename, ekind);
                    {
                        let em = enemy.as_object_mut().expect("enemy dict");
                        em.insert("image".into(), image);
                        em.insert("rig".into(), rig);
                        if let Some(s) = &escale {
                            stamp_scale(em, s);
                        }
                        if let Some(g) = group {
                            em.insert("rig_group".into(), json!(g.group));
                            em.insert("rig_part".into(), json!(g.part));
                        }
                    }
                    let rig_form = match ekind {
                        Some("TestSubject") => {
                            if enemy_has_power(enemy, "NEMESIS") {
                                Some(json!(3))
                            } else if enemy_has_power(enemy, "PAINFUL_STABS") {
                                Some(json!(2))
                            } else {
                                None
                            }
                        }
                        _ => None,
                    };
                    let rig_anim = match ekind {
                        Some("TestSubject") => None,
                        Some("SpinyToad") => Some(if enemy_has_power(enemy, "THORNS") {
                            "idle_loop"
                        } else {
                            "idle_naked_loop"
                        }),
                        Some("Tunneler") => Some(if enemy_has_power(enemy, "BURROWED") {
                            "hidden_loop"
                        } else {
                            "idle_loop"
                        }),
                        Some("FakeMerchantMonster") => Some("combat_idle_loop"),
                        _ => None,
                    };
                    {
                        let em = enemy.as_object_mut().expect("enemy dict");
                        if let Some(f) = rig_form {
                            em.insert("rig_form".into(), f);
                        }
                        if let Some(a) = rig_anim {
                            em.insert("rig_anim".into(), json!(a));
                        }
                    }
                    if let Some(Value::Array(intents)) = enemy.get_mut("intents") {
                        for intent in intents.iter_mut() {
                            let dmg = intent.get("damage").and_then(Value::as_i64);
                            let hits = intent
                                .get("hits")
                                .and_then(Value::as_i64)
                                .filter(|h| *h != 0)
                                .unwrap_or(1);
                            let total = dmg.map(|d| d * hits);
                            let itype = sval(intent, "type").to_string();
                            let img = url(assets.intent(&itype, total));
                            if let Some(im) = intent.as_object_mut() {
                                im.insert("intent_image".into(), img);
                            }
                            enrich_intent_modifiers(intent, ctx);
                        }
                    }
                    enrich_powers(enemy.get_mut("powers"), ctx);
                    enrich_cards(enemy.get_mut("stolen_cards"), false, false, &character, ctx);
                }
                if let Some(hp_ranges) = ctx.hp_ranges {
                    apply_instance_jitter(enemies, hp_ranges);
                }
            }
            if let Some(osty) = state.get_mut("osty").filter(|o| truthy(o)) {
                let name = osty.get("name").filter(|n| truthy(n)).map(py_str);
                if let Some(name) = name {
                    let img = url(assets.monster(&name));
                    if let Some(om) = osty.as_object_mut() {
                        om.insert("image".into(), img);
                    }
                }
            }
            if let Some(Value::Array(orbs)) = state.get_mut("orbs") {
                for orb in orbs.iter_mut() {
                    let img = url(assets.orb(sval(orb, "type")));
                    let name = resolve(orb.get("name").and_then(Value::as_str), loc);
                    if let Some(om) = orb.as_object_mut() {
                        om.insert("image".into(), img);
                        om.insert("name".into(), json!(name));
                    }
                }
            }
            let osty_alive = state
                .get("osty")
                .filter(|o| o.is_object())
                .is_some_and(|o| truthy(o.get("alive").unwrap_or(&Value::Null)));
            enrich_cards(state.get_mut("hand"), true, osty_alive, &character, ctx);
            for pile_key in ["draw_pile", "discard_pile", "exhaust_pile"] {
                enrich_cards(state.get_mut(pile_key), true, osty_alive, &character, ctx);
            }
            if let Some(Value::Array(draw)) = state.get_mut("draw_pile") {
                // Order-scrambled so the client can't peek at the real draw order.
                draw.sort_by(|a, b| {
                    let ka = (
                        py_str(a.get("name").unwrap_or(&json!(""))),
                        a.get("cost").and_then(Value::as_i64).unwrap_or(0),
                    );
                    let kb = (
                        py_str(b.get("name").unwrap_or(&json!(""))),
                        b.get("cost").and_then(Value::as_i64).unwrap_or(0),
                    );
                    ka.cmp(&kb)
                });
            }
            state.insert("energy_image".into(), url(assets.energy(&character)));
        }
        "card_reward" | "card_select" => {
            let ctx_str = state
                .get("context")
                .and_then(Value::as_str)
                .map(str::to_string);
            let previews = decision == "card_select"
                && ctx_str
                    .as_deref()
                    .is_some_and(|c| UPGRADE_CONTEXTS.contains(&c));
            if let Some(Value::Array(cards)) = state.get_mut("cards") {
                for card in cards.iter_mut() {
                    if previews && !truthy(card.get("upgraded").unwrap_or(&Value::Null)) {
                        let mut twin = card.as_object().cloned().unwrap_or_default();
                        twin.insert("upgraded".into(), json!(true));
                        twin.remove("stats");
                        apply_upgrade_cost(&mut twin, card, loc);
                        let cid = id_tail(card);
                        if let Some(delta) = loc
                            .get(&format!("{cid}.upgrade_keywords"))
                            .filter(|d| !d.is_empty())
                        {
                            twin.insert("keyword_delta".into(), json!(delta));
                        }
                        let mut twin = Value::Object(twin);
                        enrich_card(&mut twin, false, false, &character, ctx);
                        card.as_object_mut()
                            .expect("card dict")
                            .insert("upgrade_preview".into(), twin);
                    }
                    enrich_card(card, false, false, &character, ctx);
                }
            }
        }
        "shop" => {
            if let Some(Value::Array(cards)) = state.get_mut("cards") {
                for card in cards.iter_mut() {
                    enrich_card(card, false, false, &character, ctx);
                    if card.get("card_cost").is_none_or(Value::is_null) {
                        let cid = id_tail(card);
                        let upgraded = truthy(card.get("upgraded").unwrap_or(&Value::Null));
                        if let Some(mcost) = manifest_cost(&cid, loc, upgraded) {
                            card.as_object_mut()
                                .expect("card dict")
                                .insert("card_cost".into(), mcost);
                        }
                    }
                }
            }
            if let Some(Value::Array(relics)) = state.get_mut("relics") {
                for relic in relics.iter_mut() {
                    enrich_relic(relic, ctx);
                }
            }
            if let Some(Value::Array(potions)) = state.get_mut("potions") {
                for potion in potions.iter_mut() {
                    enrich_potion(potion, ctx);
                }
            }
        }
        "bundle_select" => {
            if let Some(Value::Array(bundles)) = state.get_mut("bundles") {
                for bundle in bundles.iter_mut() {
                    enrich_cards(bundle.get_mut("cards"), false, false, &character, ctx);
                }
            }
        }
        "treasure" => {
            if let Some(Value::Array(relics)) = state.get_mut("relics") {
                for relic in relics.iter_mut() {
                    let name = resolve_name(relic.get("name").and_then(Value::as_str), loc);
                    let img = url(assets.relic(&name));
                    if let Some(rm) = relic.as_object_mut() {
                        rm.insert("name".into(), json!(name));
                        rm.insert("image".into(), img);
                    }
                }
            }
        }
        "event_choice" => {
            let event_name = resolve_name(state.get("event_name").and_then(Value::as_str), loc);
            state.insert("event_name".into(), json!(event_name));
            state.insert(
                "art".into(),
                json!({"illustration": url(assets.event_illustration(&event_name))}),
            );
            let raw_options: Vec<Value> = state
                .get("options")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            let mut event_vars = Map::new();
            for opt in &raw_options {
                for (k, v) in event_option_vars(opt, loc, "vars") {
                    event_vars.insert(k, v);
                }
            }
            let mut body = resolve(state.get("description").and_then(Value::as_str), loc);
            if looks_like_unresolved_key(Some(&body)) {
                body = String::new();
            }
            let ev = Value::Object(event_vars);
            state.insert(
                "description".into(),
                json!(resolve_card_text(
                    Some(&body),
                    Some(&ev),
                    flags(false, false, None, false)
                )),
            );
            let eflags = event_flags();
            let mut options_out = Vec::with_capacity(raw_options.len());
            for opt in &raw_options {
                let mut opt_vars = event_option_vars(opt, loc, "vars");
                let title_src = opt
                    .get("title")
                    .filter(|t| truthy(t))
                    .or_else(|| opt.get("name"))
                    .and_then(Value::as_str);
                let mut title = resolve(title_src, loc);
                let fallback_title = sval(opt, "fallback_title").to_string();
                if looks_like_unresolved_key(Some(&title)) && !fallback_title.is_empty() {
                    title.clone_from(&fallback_title);
                }
                if title.contains('{') {
                    let mut title_vars = opt_vars.clone();
                    for (k, v) in event_option_vars(opt, loc, "title_vars") {
                        title_vars.insert(k, v);
                    }
                    let tv_keys: BTreeSet<String> = title_vars.keys().cloned().collect();
                    let need =
                        required_stat_vars(Some(&title), flags(false, false, Some(&eflags), false));
                    if need.difference(&tv_keys).next().is_some() && !fallback_title.is_empty() {
                        title.clone_from(&fallback_title);
                    }
                    let tv = Value::Object(title_vars);
                    title = resolve_card_text(
                        Some(&title),
                        Some(&tv),
                        flags(false, false, Some(&eflags), false),
                    );
                }
                let mut template = resolve(opt.get("description").and_then(Value::as_str), loc);
                let supplied: BTreeSet<String> = opt_vars.keys().cloned().collect();
                let need_plain =
                    required_stat_vars(Some(&template), flags(false, false, None, false));
                for (k, v) in named_description_vars(&need_plain, &supplied, loc) {
                    opt_vars.insert(k, json!(v));
                }
                let supplied: BTreeSet<String> = opt_vars.keys().cloned().collect();
                let need =
                    required_stat_vars(Some(&template), flags(false, false, Some(&eflags), false));
                let unrenderable = need.difference(&supplied).next().is_some()
                    || looks_like_unresolved_key(Some(&template));
                let fallback_description = sval(opt, "fallback_description").to_string();
                if unrenderable && !fallback_description.is_empty() {
                    template = fallback_description;
                }
                let ov = Value::Object(opt_vars);
                let mut desc = resolve_card_text(
                    Some(&template),
                    Some(&ov),
                    flags(false, false, Some(&eflags), false),
                );
                let text_key = sval(opt, "text_key");
                let reward_id = text_key.rsplit('.').next().unwrap_or("").to_string();
                let image = if reward_id.is_empty() {
                    Value::Null
                } else {
                    url(assets.relic(&reward_id))
                };
                if desc.is_empty() && !reward_id.is_empty() {
                    let fallback = plain_owned_description(Some(&reward_id), loc)
                        .filter(|p| !p.is_empty())
                        .map(str::to_string)
                        .or_else(|| {
                            loc.get(&format!("{reward_id}.description"))
                                .map(str::to_string)
                        });
                    if let Some(fb) = fallback.filter(|f| !f.is_empty()) {
                        desc = resolve_card_text(
                            Some(&fb),
                            Some(&ov),
                            flags(false, false, None, false),
                        );
                    }
                }
                options_out.push(json!({
                    "index": opt.get("index").cloned().unwrap_or(Value::Null),
                    "title": title,
                    "description": desc,
                    "image": image,
                    "is_locked": truthy(opt.get("is_locked").unwrap_or(&Value::Null)),
                }));
            }
            state.insert("event_options".into(), Value::Array(options_out));
        }
        "rest_site" => {
            let rest_options: Vec<Value> = state
                .get("options")
                .and_then(Value::as_array)
                .map(|opts| {
                    opts.iter()
                        .map(|o| {
                            let name_src = o
                                .get("name")
                                .filter(|n| truthy(n))
                                .or_else(|| o.get("option_id"))
                                .and_then(Value::as_str);
                            json!({
                                "index": o.get("index").cloned().unwrap_or(Value::Null),
                                "option_id": o.get("option_id").cloned().unwrap_or(Value::Null),
                                "name": resolve(name_src, loc),
                                "is_enabled": o.get("is_enabled").is_none_or(truthy),
                            })
                        })
                        .collect()
                })
                .unwrap_or_default();
            state.insert("rest_options".into(), Value::Array(rest_options));
        }
        "map_select" => {
            if let Some(Value::Array(choices)) = state.get_mut("choices") {
                for choice in choices.iter_mut() {
                    let icon =
                        url(assets.room_icon(choice.get("type").and_then(Value::as_str), None));
                    if let Some(cm) = choice.as_object_mut() {
                        cm.insert("icon".into(), icon);
                    }
                }
            }
            if let Some(fm) = state.remove("full_map") {
                state.insert("full_map".into(), enrich_map_nodes(fm, assets, base));
            }
        }
        _ => {}
    }
    step.as_object_mut()
        .expect("step dict")
        .insert("action_label".into(), json!(label));
    step
}

/// The loc maps this crate serves are process-lifetime statics (the
/// embedded bundle); the combat-log resolvers need `'static` closures.
fn loc_static(loc: &Loc) -> &'static Loc {
    // Every build's embedded bundle (GH #801), so a v0.111.0 session never
    // leaks a copy per call.
    if let Some(embedded) = sts2_core::GameVersion::ALL
        .into_iter()
        .map(Loc::embedded_for)
        .find(|e| std::ptr::eq(loc, *e))
    {
        embedded
    } else {
        // A caller-supplied map (tests): leak a copy — bounded by the number
        // of distinct maps a test constructs.
        Box::leak(Box::new(loc.clone()))
    }
}

/// `localization.resolve` re-exported for callers that only have this module.
#[must_use]
pub fn resolve_text(value: Option<&str>, loc: &Loc) -> String {
    localization::resolve(value, loc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_colors() {
        assert_eq!(
            frame_color(Some("Ironclad"), Some("Attack"), Some("Common")),
            "red"
        );
        assert_eq!(frame_color(None, Some("Curse"), None), "curse");
        assert_eq!(
            frame_color(Some("Silent"), Some("Skill"), Some("Quest")),
            "quest"
        );
        assert_eq!(frame_color(None, None, None), "colorless");
    }

    #[test]
    fn loc_key_regex() {
        assert!(is_loc_key("WINGED_BOOTS.title"));
        assert!(is_loc_key("A1.name"));
        assert!(!is_loc_key("winged_boots.title"));
        assert!(!is_loc_key("X.pages.INITIAL.description"));
        assert!(!is_loc_key("Hello"));
    }

    #[test]
    fn asset_urls() {
        assert_eq!(
            asset_url("/assets", Some("a/b.png".into())),
            json!("/assets/a/b.png")
        );
        assert_eq!(asset_url("/assets", None), Value::Null);
        assert_eq!(asset_url("/assets", Some(String::new())), Value::Null);
    }

    #[test]
    fn jitter_fires_only_for_groups_of_two() {
        let mut ranges = BTreeMap::new();
        ranges.insert(
            "Exoskeleton".to_string(),
            HpRange {
                min: 20,
                max: 28,
                min_asc: 21,
                max_asc: 29,
                hp_ascension: 8,
            },
        );
        let mut enemies = vec![
            json!({"kind": "Exoskeleton", "max_hp": 28, "rig_bounds_w": 100.0, "rig_bounds_h": 80.0}),
            json!({"kind": "Exoskeleton", "max_hp": 20, "rig_bounds_w": 100.0, "rig_bounds_h": 80.0}),
            json!({"kind": "Nibbit", "max_hp": 10}),
        ];
        apply_instance_jitter(&mut enemies, &ranges);
        assert_eq!(enemies[0]["size_jitter"], json!(1.15));
        assert_eq!(enemies[1]["size_jitter"], json!(1.0));
        assert!(enemies[0]["hue_jitter"].is_number());
        assert!(enemies[2].get("size_jitter").is_none());
    }

    #[test]
    fn upgrade_cost_shifts_live_costs() {
        let mut loc = Loc::new();
        loc.insert("STAMPEDE.cost", "2");
        loc.insert("STAMPEDE.upgraded_cost", "1");
        let card = json!({"id": "STAMPEDE", "cost": 2, "base_cost": 2});
        let mut twin = card.as_object().cloned().unwrap();
        apply_upgrade_cost(&mut twin, &card, &loc);
        assert_eq!(twin["cost"], json!(1));
        assert_eq!(twin["base_cost"], json!(1));
        let card = json!({"id": "STAMPEDE", "cost": null});
        let mut twin = card.as_object().cloned().unwrap();
        apply_upgrade_cost(&mut twin, &card, &loc);
        assert!(twin.get("cost").is_none());
    }
}
